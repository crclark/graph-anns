#![feature(is_sorted)]
#![feature(portable_simd)]
extern crate atomic_float;
extern crate graph_anns;
extern crate indicatif;
extern crate nix;
extern crate nohash_hasher;
extern crate parking_lot;
extern crate rand;
extern crate rand_core;
extern crate rand_xoshiro;
extern crate rayon;
extern crate tinyset;

use graph_anns::*;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;
use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::path::Path;
use std::sync::RwLock;
use std::time::Instant;
use std::{io, mem};
use texmex::ID;

use std::io::prelude::*;

mod texmex;

fn pause() {
  let mut stdin = io::stdin();
  let mut stdout = io::stdout();

  // We want the cursor to stay at the end of the line, so we print without a newline and flush manually.
  write!(stdout, "Press any key to continue...").unwrap();
  stdout.flush().unwrap();

  // Read a single byte and discard
  let _ = stdin.read(&mut [0u8]).unwrap();
}

// fn main_alloc_1B() {
//   println!(
//     "let's see how much memory a 1-billion element HashMap<u32, u32> takes."
//   );
//   // Answer: about 9.5GiB with NoHashHasher, 18.5 with default hasher (not sure why it makes a difference)

//   let n = 1000_000_000;
//   let start = Instant::now();
//   let mut h =
//     HashMap::<u32, u32, BuildHasherDefault<NoHashHasher<u32>>>::with_capacity_and_hasher(n, BuildNoHashHasher::default());

//   for i in 0..n {
//     h.insert(i as u32, i as u32);
//     if i % 1000000 == 0 {
//       println!("inserted {}", i);
//     }
//   }

//   let duration = start.elapsed();

//   println!("Time elapsed: {:?}", duration);
//   pause();
// }

// fn allocate_1_billion() {
//   println!(
//     "let's see how much memory a 1-billion element DenseKNNGraph takes."
//   );

//   // answer: 67.3G resident, took 211 seconds to allocate.

//   fn dist(_x: &u32, _y: &u32) -> f32 {
//     return 1.1f32;
//   }

//   let n = 1000_000_000;
//   let start = Instant::now();
//   let build_hasher: BuildHasherDefault<NoHashHasher<u32>> =
//     nohash_hasher::BuildNoHashHasher::default();
//   let config =
//     KNNGraphConfig::new(n, 5, 5, &dist, build_hasher, false, 2, false);
//   let _ = DenseKNNGraph::empty(config);

//   let duration = start.elapsed();

//   println!("Time elapsed: {:?}", duration);
//   pause();
// }

fn make_dist_fn<'a>(
  base_vecs: texmex::Vecs<'a, u8>,
  query_vecs: texmex::Vecs<'a, u8>,
) -> impl Fn(&ID, &ID) -> f32 + 'a + Clone + Sync + Send {
  let dist_fn = Box::new(move |x: &ID, y: &ID| -> f32 {
    let x_slice = match x {
      ID::Base(i) => &base_vecs[*i as usize],
      ID::Query(i) => &query_vecs[*i as usize],
    };
    let y_slice = match y {
      ID::Base(i) => &base_vecs[*i as usize],
      ID::Query(i) => &query_vecs[*i as usize],
    };
    return texmex::sq_euclidean_iter(x_slice, y_slice);
  });
  dist_fn
}

fn load_texmex_to_dense<'a>(
  subset_size: u32,
  dist_fn: &'a (dyn Fn(&ID, &ID) -> f32 + Sync),
) -> KNN<'a, ID, RandomState> {
  let start_allocate_graph = Instant::now();

  let build_hasher = RandomState::new();

  // TODO: re-run the time benchmarks below with  RUSTFLAGS="-C target-cpu=native -C opt-level=3" cargo run texmex_tests
  // I was accidentally disabling optimizations when I ran the benchmarks below because I was mixing
  // --release with RUSTFLAGS, and RUSTFLAGS was taking precedence.
  // out_degree of 10 -> OOM. 7 -> 89.7 resident, slow insertion (310s per 100k).
  // 5 -> ~80 resident, 200s per 100k insertion. 3 might be optimal.
  // out_degree 3, num_searchers 5 -> ~45 resident, 40s per 100k.
  // out_degree 3, num_searchers 5, rrnp=false -> ~45 resident, 40s per 100k -- i.e. rrnp makes no difference?
  // TODO: collect stats on how much rrnp actually changes the graph on each insertion.
  // higher num_searchers also kills insertion speed.
  // TODO: should num_searchers be dynamic based on graph size? If small, don't
  // need a lot.
  //
  // Parameter effects on search quality (recall@r):
  // out_degree 7, num_searchers 7, rrnp false, lgd true -> 0.33 recall@10
  // out_degree 7, num_searchers 7, rrnp true, lgd true -> 0.3038 recall@10
  // out_degree 7, num_searchers 14, rrnp true, lgd true -> 0.31 recall@10
  // out_degree 3, num_searchers 7, rrnp false, lgd true -> 0.0005 recall@10
  // out_degree 15, num_searchers 7, rrnp false, lgd true -> 0.7062 recall@10
  // out_degree 20, num_searchers 7, rrnp true, lgd true -> 0.79 recall@10, 1000s runtime
  // out_degree 20, num_searchers 1, rrnp true, lgd true -> 0.78 recall@10, 1070s runtime
  // out_degree 30, num_searchers 1, rrnp true, lgd true -> 0.87 recall@10, 1973s runtime
  // out_degree 40, num_searchers 1, rrnp true, lgd true -> 0.91 recall@10, 3145s runtime
  // TL;dr: higher out_degree has greatest effect on search quality. That's
  // frustrating because it is also the greatest contributor to memory usage.
  //
  // Other performance observations:
  //
  // - Search speed degrades as the graph gets bigger (of course), but the rate
  // of degradation is steeper for larger out_degree -- for small out_degree (e.g., 7),
  // it's unnoticeable for 1M items, but for out_degree 20, inserting 100k items
  // takes an additional 10s for each 100k items already in the graph. Oof.
  // - For out_degree and num_searchers = 7, insertion speed starts to significantly
  // degrade after 250M points have been inserted. So inserting the remaining points
  // will take *at least* 1500 additional hours (assuming no further speed degradation)!
  // We will need to split into multiple graphs (optionally managed by separate threads)
  // to reach 1B points. The tradeoff to faster insertion may be slower search (or at least
  // multithreaded search). Hmm. I think the benefit to insertion will far outweigh the
  // slowdown at search time, which may not be too bad.
  let config = KNNGraphConfig::new(
    subset_size,
    7,
    7,
    dist_fn,
    build_hasher,
    true,
    2,
    true,
  );

  let mut prng = Xoshiro256StarStar::seed_from_u64(1);
  let mut g = KNN::new(config);

  // oops, this is irrelevant because we start out with the brute force
  // implementation. So this is 19 microseconds. All of the real allocation is
  // going to happen invisibly on one of the inserts. We should log the insert
  // time of each insert to a CSV. We're going to see a *crazy* latency spike
  // for one of the insertions.
  println!("Allocated graph in {:?}", start_allocate_graph.elapsed());

  let start_inserting = Instant::now();

  for i in 0..subset_size {
    if i % 100000 == 0 {
      println!("inserting {}", i);
      println!("elapsed: {:?}", start_inserting.elapsed());
    }
    g.insert(ID::Base(i as u32), &mut prng);
  }
  println!(
    "Finished building the nearest neighbors graph in {:?}",
    start_inserting.elapsed()
  );
  println!("Graph size: {:?}", g.debug_size_stats());
  g
}

fn recall_at_r<G: NN<ID>, R: RngCore>(
  g: &G,
  query_vecs: &texmex::Vecs<u8>,
  ground_truth: &texmex::Vecs<i32>,
  r: usize,
  prng: &mut R,
) -> f32 {
  let start_recall = Instant::now();
  let mut num_correct = 0;
  for i in 0..query_vecs.num_rows {
    let query = ID::Query(i as u32);
    let SearchResults {
      approximate_nearest_neighbors,
      ..
    } = g.query(&query, r, prng);
    // TODO: passing a PRNG into every query? Just store it in the graph.
    for nbr in approximate_nearest_neighbors {
      let nbr_id = match nbr.item {
        ID::Base(i) => i,
        ID::Query(i) => i,
      };
      if ground_truth[i as usize][0] == nbr_id as i32 {
        num_correct += 1;
      }
    }
  }
  println!("Finished recall@{} in {:?}", r, start_recall.elapsed());

  num_correct as f32 / query_vecs.num_rows as f32
}

fn main_serial() {
  // TODO: this is absurdly slow to build a graph, even for just 1M elements.
  // Optimize it. Focus on the stuff in lib; don't spend time optimizing the
  // distance function unless there's something egregiously broken there.
  let subset_size: u32 = 5_000_000;
  let base_path = Path::new("/mnt/970pro/anns/bigann_base.bvecs_array");
  let base_vecs = texmex::Vecs::<u8>::new(base_path, 128).unwrap();
  let query_path = Path::new("/mnt/970pro/anns/bigann_query.bvecs_array");
  let query_vecs = texmex::Vecs::<u8>::new(query_path, 128).unwrap();
  let gnd_path = Path::new("/mnt/970pro/anns/gnd/idx_1000M.ivecs_array");
  let ground_truth = texmex::Vecs::<i32>::new(gnd_path, 1000).unwrap();

  let dist_fn = make_dist_fn(base_vecs, query_vecs);
  let g = load_texmex_to_dense(subset_size, &dist_fn);

  let mut prng = Xoshiro256StarStar::seed_from_u64(1);
  let recall = recall_at_r(&g, &query_vecs, &ground_truth, 10, &mut prng);
  println!("Recall@10: {}", recall);
}

// TODO: we need to parallelize insertion because it gets slower as more stuff
// is inserted. Let's start by implementing parallelization here, then move
// it into the library once we know what we are doing. It would be nice if we
// can keep it flexible -- maybe no dependency on rayon or queues and just
// focus on providing a helper that maintains a set of graphs behind rwlocks
// and basic tools to merge search results. Then users can decide how to put
// them together into a real application. We can also provide a basic helper
// that works for most use cases.
//
// Step 1: merge function for search results.
// Step 2: struct that contains a vec of rwlock<graph>
// Step 3: use rayon parallel iterator over enum(rows_to_insert), insert
// into i%n graph, where n is the number of graphs.
// Step 4: search in parallel across all graphs, merge results.

/// A simple wrapper around a Vec of RwLocks of graphs to provide a basic
/// parallel insert/search interface.
/// The capacity of each graph is set to 1/n of the total capacity in the input
/// config, so be
/// careful when inserting -- if you don't balance your inserts, you could hit
/// problems with the graphs filling up.
/// insert() can be called from multiple threads and write locks a random graph.
/// search() can be called from multiple threads and read locks all graphs. All
/// searches within are performed sequentially and the results are
/// merged.
/// delete() can be called from multiple threads and write locks all graphs sequentially (because
/// we don't know which graph an item was originally inserted into). Only one graph is locked at any time.
/// All deletes within are performed in sequentially.
///
/// This can obviously be improved upon -- concurrent search and delete could experience races
/// since internally, graphs are locked sequentially. The user can also directly access
/// individual graphs for better performance -- for example, you can parallelize
/// insertion more efficiently by inserting into graph i%n instead of calling
/// a prng. The insert() function can't do that because it doesn't have access
/// to i.
pub struct ManyKNN<'a, ID, S: BuildHasher + Clone> {
  pub graphs: Vec<RwLock<KNN<'a, ID, S>>>,
}

impl<ID: Copy + Ord + std::hash::Hash, S: BuildHasher + Clone>
  ManyKNN<'_, ID, S>
{
  pub fn new(n: u32, config: KNNGraphConfig<ID, S>) -> ManyKNN<ID, S> {
    let total_capacity = config.capacity;
    let individual_capacity = total_capacity / n + total_capacity % n as u32;
    let mut graphs = Vec::with_capacity(n as usize);
    for _ in 0..n {
      let mut config = config.clone();
      config.capacity = individual_capacity;
      graphs.push(RwLock::new(KNN::new(config)));
    }
    ManyKNN { graphs }
  }

  pub fn debug_size_stats(&self) -> SpaceReport {
    self
      .graphs
      .iter()
      .map(|g| g.read().unwrap().debug_size_stats())
      .reduce(|acc, e| acc.merge(&e))
      .unwrap()
  }
}

impl<
    'a,
    T: Copy + Ord + Eq + std::hash::Hash + Send + Sync,
    S: BuildHasher + Clone + Send + Sync,
  > NN<T> for ManyKNN<'a, T, S>
{
  fn insert<R: RngCore>(&mut self, x: T, prng: &mut R) -> () {
    let i = prng.next_u32() % self.graphs.len() as u32;
    let mut g = self.graphs[i as usize].write().unwrap();
    g.insert(x, prng);
  }

  fn delete(&mut self, x: T) -> () {
    for g in &self.graphs {
      let mut g = g.write().unwrap();
      g.delete(x);
    }
  }

  fn query<R: RngCore>(
    &self,
    q: &T,
    max_results: usize,
    _prng: &mut R,
  ) -> SearchResults<T> {
    let mut results = (0..self.graphs.len())
      .into_par_iter()
      .map(|i| {
        self.graphs[i].read().unwrap().query(
          q,
          max_results,
          &mut rand::thread_rng(),
        )
      })
      .reduce(|| SearchResults::<T>::default(), |acc, e| acc.merge(&e));

    results.approximate_nearest_neighbors.truncate(max_results);
    results
  }
}

#[repr(C, packed)]
struct Test {
  a: u32,
  b: f32,
  d: u8,
}

fn load_texmex_to_dense_par<'a>(
  subset_size: u32,
  num_graphs: u32,
  dist_fn: &'a (dyn Fn(&ID, &ID) -> f32 + Sync),
) -> ManyKNN<'a, ID, RandomState> {
  let start_allocate_graph = Instant::now();

  let build_hasher = RandomState::new();

  let out_degree = 7u8;
  println!(
    "Size of edges array will be {}",
    out_degree as usize
      * subset_size as usize
      * mem::size_of::<(u32, f32, u8)>() as usize
  );

  println!(
    "If we used a custom struct with packed, it would be {}",
    out_degree as usize
      * subset_size as usize
      * mem::size_of::<Test>() as usize
  );

  println!(
    "If we switched to Vec<(u32, f32)>, Vec<u8>, it would be {}",
    out_degree as usize
      * subset_size as usize
      * mem::size_of::<(u32, f32)>() as usize
      + out_degree as usize
        * subset_size as usize
        * mem::size_of::<u8>() as usize
  );

  println!(
    "If we switched to Vec<(u32, f16)>, Vec<u8>, it would be {}",
    out_degree as usize
      * subset_size as usize
      * mem::size_of::<(u32, u16)>() as usize
      + out_degree as usize
        * subset_size as usize
        * mem::size_of::<u8>() as usize
  );

  println!(
    "If we switched to struct of arrays, it would be {}",
    out_degree as usize * subset_size as usize * mem::size_of::<u32>() as usize
      + out_degree as usize
        * subset_size as usize
        * mem::size_of::<f32>() as usize
      + out_degree as usize
        * subset_size as usize
        * mem::size_of::<u8>() as usize
  );

  println!(
    "If we switched to f16 stored distances, it would be {}",
    out_degree as usize * subset_size as usize * mem::size_of::<u32>() as usize
      + out_degree as usize
        * subset_size as usize
        * mem::size_of::<u16>() as usize
      + out_degree as usize
        * subset_size as usize
        * mem::size_of::<u8>() as usize
  );

  println!(
    "If we always recomputed distances and didn't store them, it would be {}",
    out_degree as usize * subset_size as usize * mem::size_of::<u32>() as usize
      + out_degree as usize
        * subset_size as usize
        * mem::size_of::<u8>() as usize
  );

  let config = KNNGraphConfig::new(
    subset_size as u32,
    out_degree,
    7,
    dist_fn,
    build_hasher,
    true,
    2,
    true,
  );

  let g = ManyKNN::new(num_graphs, config);

  // oops, this is irrelevant because we start out with the brute force
  // implementation. So this is 19 microseconds. All of the real allocation is
  // going to happen invisibly on one of the inserts. We should log the insert
  // time of each insert to a CSV. We're going to see a *crazy* latency spike
  // for one of the insertions.
  println!("Allocated graph in {:?}", start_allocate_graph.elapsed());

  let start_inserting = Instant::now();

  let style = ProgressStyle::with_template(
    "[{elapsed_precise}] {bar:40.cyan/blue} {human_pos:>7}/{human_len:7} ETA: {eta_precise} Insertions/sec: {per_sec}",
  )
  .unwrap();

  (0..subset_size)
    .into_par_iter()
    .progress_with_style(style)
    .for_each(|i| {
      let mut g = g.graphs[i as usize % g.graphs.len()].write().unwrap();
      g.insert(ID::Base(i as u32), &mut rand::thread_rng());
    });
  println!(
    "Finished building the nearest neighbors graph in {:?}",
    start_inserting.elapsed()
  );

  println!("Graph size: {:?}", g.debug_size_stats());
  g
}

fn main() {
  // NOTE: change gnd_path when you change this
  let subset_size: u32 = 1_000_000_000;
  let num_graphs: u32 = 32;
  let base_path = Path::new("/mnt/970pro/anns/bigann_base.bvecs_array");
  let base_vecs = texmex::Vecs::<u8>::new(base_path, 128).unwrap();
  let query_path = Path::new("/mnt/970pro/anns/bigann_query.bvecs_array");
  let query_vecs = texmex::Vecs::<u8>::new(query_path, 128).unwrap();
  // NOTE: change this path depending on subset_size
  let gnd_path = Path::new("/mnt/970pro/anns/gnd/idx_1000M.ivecs_array");
  let ground_truth = texmex::Vecs::<i32>::new(gnd_path, 1000).unwrap();

  let dist_fn = make_dist_fn(base_vecs, query_vecs);
  let g = load_texmex_to_dense_par(subset_size, num_graphs, &dist_fn);

  let mut prng = Xoshiro256StarStar::seed_from_u64(1);
  let recall = recall_at_r(&g, &query_vecs, &ground_truth, 10, &mut prng);
  println!("Recall@10: {}", recall);
  pause()
}

// TODO: parallel version appears to use more memory than single-threaded version
// and I am not sure why.
// For subset_size = 5M and num_graphs = 32,
// parallel version is: 83s to build graph, 1.9G res, 122.0 virt
// single threaded version is: 1791s to build graph, 1.9G res, 122.0 virt
// never mind; I guess I imagined it.

// TODO: fine-grained parallel insertion? Would this be helpful in some way?
// We could make an insert_batch() API that runs all the searches for the batch
// in parallel, then inserts each one serially. The only benefit I can see for
// that is that we could be parallel within a single partition of the data
// structure, which could allow us to keep memory usage low by writing partitions
// to disk after they have reached a certain size. However, we would still need
// them all in memory at query time, so I don't see a big advantage to this.
// Parallelism would also be limited by the sequential insertion, which is
// able to be parallelized across partitions with our current approach.

// TODO: NVMe works best with high queue depths. Should we expose a prefetch
// callback to the user, which we call as soon as we know we are going to
// call the distance function on an item?

// TODO: other optimization ideas:
// - try BTreeMap for external and internal mappings -- might be more space
// efficient, might be faster in time, too. DONE, much slower, no memory savings.
// - Try to find a way to avoid storing two copies of the external id for the
// internal/external maps. Is there a bidi map that can avoid extra copies of
// the k/v?
// - Eliminate the temporary SetU32 in query_internal (r_nbrs). DONE, 10% speedup.
// - Use a vec for the internal to external mapping. DONE, good speedup, 10% memory savings.
// - Replace random selection of starting points in each query with
// reservoir-sampled selection of starting points. Eventually this could be
// replaced with smarter pivot selection algorithms. DONE as part of switch to vec
// for int_to_ext mapping. Unclear if this had a performance impact outside of
// the 10% speedup from using a vec for int_to_ext.
// - Analyze the behavior of the algorithm itself and devise improvements and
// novel extensions. Open questions:
//   - Does including backpointers in the search improve search performance?
//   - How much does distance to the best-so-far decrease on each iteration?
//   - Do long-distance jumps (or random restarts) help performance?
//   - How many nodes do we visit per query?
//   - Can we learn better starting points based on our query patterns? For
//     example, does starting at recent query points help? Or can we find
//     frequently-visited points and use them as starting points?
//   - Can we avoid tracking already-traversed nodes by ensuring our algorithm
//     always moves in the direction of decreasing distance?
//   - Does a greater number of starting points even help performance (recall@n)? Early
//     evidence suggests the answer is no, so does reducing the number of starting points
//     reduce query latency? Note that this may be dataset-specific -- I can imagine that
//     having one starting point per cluster could really help performance.
//   - Can we use the triangle inequality to reduce distance calls?
//   - Can we find a more clever stopping condition that allows us to stop earlier?
//   - Why does increasing out_degree improve recall@n so much, anyway? I thought
//     it would simply affect query latency. Find the answer.
//   - store fp16 floats. Recompute distances in cases where the precision loss
//     was so large that two floats can't be properly compared. Or maybe even
//     don't do that; this is approximate nearest neighbors, after all.

// TODO: memory optimization. The main candidate for optimization is the edges
// // vec. For 1B points with out_degree 7, we have:
// Size of edges array will be 84000000000
// If we used a custom struct with packed, it would be 63000000000
// If we switched to Vec<(u32, f32)>, Vec<u8>, it would be 63000000000
// If we switched to Vec<(u32, f16)>, Vec<u8>, it would be 63000000000
// If we switched to struct of arrays, it would be 63000000000
// If we switched to f16 stored distances, it would be 49000000000
// If we always recomputed distances and didn't store them, it would be 35000000000
//
// There are performance tradeoffs for all of these: cache locality is better if
// we keep it as-is (index, distance, crowding factor stored together). If we
// use packed, we have the performance penalty of unaligned accesses. However,
// we need to consider how often we actually use all of the elements of the tuples
// we have today... maybe the cache "locality" we think we are getting is
// actually a waste, because we don't always use all the tuple elements.
//
// We should also consider the performance benefit of using less memory: this
// means we can increase out_degree, which improves recall. of course, the
// better way to do that is to buy more RAM, so that shouldn't be an overwhelming
// consideration.

// test to make sure I understand how to share a vec of atomics between threads.

// TODO: https://travisdowns.github.io/blog/2020/07/06/concurrency-costs.html
// Read this a second time. It's great. See especially "Level 0" section.

// use std::sync::atomic::AtomicU32;
// use std::sync::Arc;
// fn main() {
//   let atomics = Arc::new(vec![AtomicU32::new(2), AtomicU32::new(1)]);

//   let num_threads = 2;

//   let mut handles = Vec::new();

//   for i in 0..num_threads {
//     let thread_num = i.clone();
//     let thread_atomics = Arc::clone(&atomics);
//     handles.push(thread::spawn(move || {
//       let my_atomic = &thread_atomics[thread_num];
//       let other_atomic = &thread_atomics[(thread_num + 1) % num_threads];
//       let other_atomic_value = other_atomic.load(std::sync::atomic::Ordering::Relaxed);
//       my_atomic.fetch_add(other_atomic_value, std::sync::atomic::Ordering::Relaxed)
//     }));
//   }

//   for handle in handles {
//     handle.join().unwrap();
//   }
//   println!("{:?}", atomics)
// }
