#![feature(total_cmp)]
#![feature(is_sorted)]
#![feature(portable_simd)]
extern crate atomic_float;
extern crate graph_anns;
extern crate nix;
extern crate nohash_hasher;
extern crate parking_lot;
extern crate rand;
extern crate rand_core;
extern crate rand_xoshiro;
extern crate rayon;
extern crate tinyset;
extern crate unroll;

use graph_anns::*;
use nohash_hasher::NoHashHasher;
use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::binary_heap::BinaryHeap;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, BuildHasherDefault, Hasher};
use std::path::Path;
use std::time::Instant;
use std::{io, thread};
use texmex::ID;

use std::io::prelude::*;

mod texmex;

#[derive(Debug)]
struct SearchResult {
  pub vec_index: usize,
  pub dist: f32,
}

impl SearchResult {
  pub fn new(vec_index: usize, dist: f32) -> SearchResult {
    Self { vec_index, dist }
  }
}

impl PartialEq for SearchResult {
  fn eq(&self, other: &Self) -> bool {
    self.vec_index == other.vec_index
  }
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    self.dist.partial_cmp(&other.dist)
  }
}

impl Ord for SearchResult {
  fn cmp(&self, other: &Self) -> Ordering {
    self.dist.total_cmp(&other.dist)
  }
}

fn _search_range<'a, T: ?Sized, C: std::ops::Index<usize, Output = T>>(
  query_set: &C,
  query_set_size: usize,
  db: &C,
  k: usize,
  lower_bound_incl: usize,
  upper_bound_excl: usize,
  // TODO: parametrize the type of the distances so we can use much faster
  // i32 if possible.
  dist_fn: fn(&T, &T) -> f32,
) -> Vec<BinaryHeap<SearchResult>> {
  let mut nearest_neighbors = Vec::new();
  for _ in 0..query_set_size {
    nearest_neighbors.push(BinaryHeap::new());
  }
  for i in lower_bound_incl..upper_bound_excl {
    for q in 0..query_set_size {
      let dist = dist_fn(&db[i], &query_set[q]);
      let heap: &mut BinaryHeap<SearchResult> =
        nearest_neighbors.get_mut(q).unwrap();
      heap.push(SearchResult { vec_index: i, dist });
      while heap.len() > k {
        heap.pop().unwrap();
      }
    }
  }
  nearest_neighbors
}

fn _search_identity_elem(
  query_set_size: usize,
) -> Vec<BinaryHeap<SearchResult>> {
  let mut nearest_neighbors = Vec::new();
  for _ in 0..query_set_size {
    nearest_neighbors.push(BinaryHeap::new());
  }
  nearest_neighbors
}

fn _search_sum(
  k: usize,
  mut v1: Vec<BinaryHeap<SearchResult>>,
  mut v2: Vec<BinaryHeap<SearchResult>>,
) -> Vec<BinaryHeap<SearchResult>> {
  for (x, y) in v1.iter_mut().zip(v2.iter_mut()) {
    x.append(y);
    while x.len() > k {
      x.pop().unwrap();
    }
  }
  v1
}

fn _search_inject<'a, T: ?Sized, C: std::ops::Index<usize, Output = T>>(
  query_set: &C,
  query_set_size: usize,
  db: &C,
  i: usize,
  k: usize,
  dist_fn: fn(&T, &T) -> f32,
  mut nearest_neighbors: Vec<BinaryHeap<SearchResult>>,
) -> Vec<BinaryHeap<SearchResult>> {
  for q in 0..query_set_size {
    let dist = dist_fn(&db[i], &query_set[q]);
    let heap: &mut BinaryHeap<SearchResult> =
      nearest_neighbors.get_mut(q).unwrap();
    heap.push(SearchResult::new(i, dist));
    while heap.len() > k {
      heap.pop().unwrap();
    }
  }
  nearest_neighbors
}

fn _search_rayon<
  'a,
  T: ?Sized,
  C: std::ops::Index<usize, Output = T> + Sync,
>(
  query_set: &C,
  query_set_size: usize,
  db: &C,
  db_size: usize,
  k: usize,
  // TODO: parametrize the type of the distances so we can use much faster
  // i32 if possible.
  dist_fn: fn(&T, &T) -> f32,
) -> Vec<BinaryHeap<SearchResult>> {
  let nearest_neighbors = (0..db_size)
    .into_par_iter()
    // NOTE: fold + reduce ensures that we don't get swamped with the overhead
    // of allocating one billion BinaryHeaps.
    .fold(
      || _search_identity_elem(query_set_size),
      |nns: Vec<BinaryHeap<SearchResult>>, i: usize| {
        _search_inject(query_set, query_set_size, db, i, k, dist_fn, nns)
      },
    )
    .reduce(
      || _search_identity_elem(query_set_size),
      |v1: Vec<BinaryHeap<SearchResult>>, v2: Vec<BinaryHeap<SearchResult>>| {
        _search_sum(k, v1, v2)
      },
    );

  nearest_neighbors
}

fn _main_old() {
  let start = Instant::now();
  let base_vecs =
    texmex::Vecs::<u8>::new("/mnt/970pro/anns/bigann_base.bvecs_array", 128)
      .unwrap();
  let query_vecs = texmex::Vecs::<u8>::new(
    "/mnt/970pro/anns/bigann_query.bvecs_array_one_point",
    128,
  )
  .unwrap();
  println!("Loaded dataset in {:?}", start.elapsed());

  let mut handles = Vec::new();
  let num_threads = 32;
  let k = 1000;

  let mut lower_bound = 0;
  let rows_per_thread = base_vecs.num_rows / num_threads;
  for i in 0..num_threads {
    let lower_bound_clone = lower_bound.clone();
    let upper_bound = if i == num_threads - 1 {
      base_vecs.num_rows
    } else {
      lower_bound + rows_per_thread
    };
    println!(
      "Launching thread {} to search from {} to {}",
      i, lower_bound, upper_bound
    );
    handles.push(thread::spawn(move || {
      _search_range::<[u8], texmex::Vecs<u8>>(
        &query_vecs,
        query_vecs.num_rows,
        &base_vecs,
        k,
        lower_bound_clone,
        upper_bound,
        texmex::sq_euclidean_faster,
      );
    }));
    lower_bound += rows_per_thread;
  }

  for handle in handles {
    handle.join().unwrap();
  }
}

// version using rayon
fn _main_rayon() {
  let start = Instant::now();
  let base_vecs =
    texmex::Vecs::<u8>::new("/mnt/970pro/anns/bigann_base.bvecs_array", 128)
      .unwrap();
  let query_vecs = texmex::Vecs::<u8>::new(
    "/mnt/970pro/anns/bigann_query.bvecs_array_one_point",
    128,
  )
  .unwrap();
  println!("Loaded dataset in {:?}", start.elapsed());

  _search_rayon(
    &query_vecs,
    query_vecs.num_rows,
    &base_vecs,
    base_vecs.num_rows,
    1000,
    texmex::sq_euclidean_faster,
  );
}

// fn main() {
//   let mmap_start = Instant::now();
//   let base_vecs =
//     texmex::Vecs::<u8>::new("/mnt/970pro/anns/bigann_base.bvecs_array", 128)
//       .unwrap();
//   println!("mmaped dataset in {:?}", mmap_start.elapsed());

//   let rand_init_graph_start = Instant::now();
//   let mut prng = Xoshiro256StarStar::seed_from_u64(1);
//   println!("TODO: fix this");
// }

fn pause() {
  let mut stdin = io::stdin();
  let mut stdout = io::stdout();

  // We want the cursor to stay at the end of the line, so we print without a newline and flush manually.
  write!(stdout, "Press any key to continue...").unwrap();
  stdout.flush().unwrap();

  // Read a single byte and discard
  let _ = stdin.read(&mut [0u8]).unwrap();
}

// fn main() {
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

fn allocate_1_billion() {
  println!(
    "let's see how much memory a 1-billion element DenseKNNGraph takes."
  );

  // answer: 67.3G resident, took 211 seconds to allocate.

  fn dist(x: &u32, y: &u32) -> f32 {
    return 1.1f32;
  }

  let n = 1000_000_000;
  let start = Instant::now();
  let build_hasher: BuildHasherDefault<NoHashHasher<u32>> =
    nohash_hasher::BuildNoHashHasher::default();
  let config =
    KNNGraphConfig::new(n, 5, 5, &dist, build_hasher, false, 2, false);
  let g = DenseKNNGraph::empty(config);

  let duration = start.elapsed();

  println!("Time elapsed: {:?}", duration);
  pause();
}

fn make_dist_fn<'a>(
  base_vecs: texmex::Vecs<'a, u8>,
  query_vecs: texmex::Vecs<'a, u8>,
) -> impl Fn(&ID, &ID) -> f32 + 'a {
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
  dist_fn: &'a dyn Fn(&ID, &ID) -> f32,
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
  g
}

fn recall_at_r<R: RngCore>(
  g: &KNN<ID, RandomState>,
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

fn main() {
  // TODO: this is absurdly slow to build a graph, even for just 1M elements.
  // Optimize it. Focus on the stuff in lib; don't spend time optimizing the
  // distance function unless there's something egregiously broken there.
  let subset_size: u32 = 1000_000_000;
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
