#![feature(is_sorted)]
#![feature(portable_simd)]
extern crate atomic_float;
extern crate clap;
extern crate graph_anns;
extern crate indicatif;
extern crate nix;
extern crate nohash_hasher;
extern crate parking_lot;
extern crate rand;
extern crate rand_core;
extern crate rand_xoshiro;
extern crate rayon;
extern crate serde;
extern crate tinyset;

use clap::Parser;
use graph_anns::Error;
use graph_anns::*;
use indicatif::{ParallelProgressIterator, ProgressIterator, ProgressStyle};
use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::RandomState;
use std::fs::{File, OpenOptions};
use std::hash::{BuildHasher, Hash};
use std::io::{prelude::*, LineWriter};
use std::path::Path;
use std::sync::RwLock;
use std::time::Instant;
use std::{io, mem};
use texmex::ID;

mod texmex;

fn pause() {
  let mut stdin = io::stdin();
  let mut stdout = io::stdout();

  // We want the cursor to stay at the end of the line, so we print without a newline and flush manually.
  write!(stdout, "Press enter to continue...").unwrap();
  stdout.flush().unwrap();

  // Read a single byte and discard
  let _ = stdin.read(&mut [0u8]).unwrap();
}

fn make_dist_fn<'a>(
  base_vecs: texmex::Vecs<'a, u8>,
  query_vecs: texmex::Vecs<'a, u8>,
) -> impl Fn(&ID, &ID) -> f32 + 'a + Clone + Sync + Send {
  Box::new(move |x: &ID, y: &ID| -> f32 {
    let x_slice = match x {
      ID::Base(i) => &base_vecs[*i as usize],
      ID::Query(i) => &query_vecs[*i as usize],
    };
    let y_slice = match y {
      ID::Base(i) => &base_vecs[*i as usize],
      ID::Query(i) => &query_vecs[*i as usize],
    };
    texmex::sq_euclidean_iter(x_slice, y_slice)
  })
}

fn load_texmex_to_dense(
  subset_size: u32,
  dist_fn: &(dyn Fn(&ID, &ID) -> f32 + Sync),
  args: Args,
) -> Knn<ID, RandomState> {
  let start_allocate_graph = Instant::now();
  let out_degree = args.out_degree;
  println!(
    "struct of arrays, will be {}",
    out_degree as usize * subset_size as usize * mem::size_of::<u32>()
      + out_degree as usize * subset_size as usize * mem::size_of::<f32>()
      + out_degree as usize * subset_size as usize * mem::size_of::<u8>()
  );

  println!(
    "The internal_to_external_ids mapping should take {}",
    subset_size as usize * mem::size_of::<Option<ID>>()
  );

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
  let config = KnnGraphConfigBuilder::new(
    subset_size,
    out_degree,
    args.num_searchers,
    build_hasher,
  )
  .use_rrnp(args.rrnp)
  .rrnp_max_depth(args.rrnp_depth)
  .use_lgd(args.lgd)
  .build();

  let mut prng = Xoshiro256StarStar::seed_from_u64(1);
  let mut g = Knn::new(config, dist_fn);

  // oops, this is irrelevant because we start out with the brute force
  // implementation. So this is 19 microseconds. All of the real allocation is
  // going to happen invisibly on one of the inserts. We should log the insert
  // time of each insert to a CSV. We're going to see a *crazy* latency spike
  // for one of the insertions.
  println!("Allocated graph in {:?}", start_allocate_graph.elapsed());

  let start_inserting = Instant::now();

  for i in 0..args.early_exit_after_num_insertions.unwrap_or(subset_size) {
    if i % 100000 == 0 {
      println!("inserting {i}");
      println!("elapsed: {:?}", start_inserting.elapsed());
    }
    g.insert(ID::Base(i), &mut prng).unwrap();
  }
  println!(
    "Finished building the nearest neighbors graph in {:?}",
    start_inserting.elapsed()
  );
  g
}

fn write_search_stats(
  search_stats: &Option<SearchStats>,
  found_closest: bool,
  line_writer: &mut LineWriter<File>,
  args: Args,
) {
  match search_stats {
    Some(s) => {
      writeln!(
        line_writer,
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        found_closest as u8,
        s.nearest_neighbor_distance,
        s.num_distance_computations,
        s.distance_from_nearest_starting_point,
        s.distance_from_farthest_starting_point,
        s.search_duration.as_micros(),
        s.largest_distance_improvement_single_hop,
        s.smallest_distance_improvement_single_hop,
        s.nearest_neighbor_path_length,
        s.num_visited,
        args.parallel,
        args.rrnp,
        args.lgd,
        args.rrnp_depth,
        args.out_degree,
        args.num_searchers,
        args.experiment_name,
      )
      .unwrap();
    }
    None => {}
  }
}

fn recall_at_r<G: NN<ID>, R: RngCore>(
  g: &G,
  query_vecs: &texmex::Vecs<u8>,
  ground_truth: &texmex::Vecs<i32>,
  r: usize,
  prng: &mut R,
  line_writer: &mut LineWriter<File>,
  args: Args,
) -> f32 {
  println!("Starting recall@{}...", r);
  let start_recall = Instant::now();
  let mut num_correct = 0;
  for i in (0..query_vecs.num_rows).progress() {
    let query = ID::Query(i as u32);
    let SearchResults {
      approximate_nearest_neighbors,
      search_stats,
      ..
    } = g.query(&query, r, prng).unwrap();

    let mut found_closest: bool = false;
    // TODO: passing a PRNG into every query? Just store it in the graph.
    for nbr in approximate_nearest_neighbors {
      let nbr_id = match nbr.item {
        ID::Base(i) => i,
        ID::Query(i) => i,
      };
      if ground_truth[i][0] == nbr_id as i32 {
        num_correct += 1;
        found_closest = true;
      }
    }
    write_search_stats(&search_stats, found_closest, line_writer, args.clone());
  }

  let qps = query_vecs.num_rows as f32 / start_recall.elapsed().as_secs_f32();

  println!(
    "Finished recall@{} for {} queries in {:?} for {} qps",
    r,
    query_vecs.num_rows,
    start_recall.elapsed(),
    qps
  );

  num_correct as f32 / query_vecs.num_rows as f32
}

fn open_csv(args: Args) -> LineWriter<File> {
  let fp = args.output_csv;
  let existed = std::path::Path::new(&fp).exists();
  let file = OpenOptions::new()
    .write(true)
    .create(true)
    .append(true)
    .open(fp)
    .unwrap();
  let mut line_writer = LineWriter::new(file);
  if !existed {
    line_writer.write_all(b"found_closest,nearest_neighbor_distance,num_distance_computations,distance_from_nearest_starting_point,distance_from_farthest_starting_point,search_duration_microseconds,largest_distance_improvement_single_hop,smallest_distance_improvement_single_hop,nearest_neighbor_path_length,num_visited,parallelism,rrnp,lgd,rrnp_depth,out_degree,num_searchers,experiment_name\n").unwrap();
  }
  line_writer
}

fn main_single_threaded(args: Args) {
  // TODO: this is absurdly slow to build a graph, even for just 1M elements.
  // Optimize it. Focus on the stuff in lib; don't spend time optimizing the
  // distance function unless there's something egregiously broken there.
  let subset_size: u32 = args.subset_size;
  let base_path = Path::new(&args.texmex_path);
  let base_vecs = texmex::Vecs::<u8>::new(base_path, 128).unwrap();
  let query_path = Path::new(&args.query_path);
  let query_vecs = texmex::Vecs::<u8>::new(query_path, 128).unwrap();
  let gnd_path = Path::new(&args.gnd_path);
  let ground_truth = texmex::Vecs::<i32>::new(gnd_path, 1000).unwrap();

  let dist_fn = make_dist_fn(base_vecs, query_vecs);
  let g = if let Some(fp) = args.clone().existing_graph {
    println!("Loading graph from {}", fp);
    let start_loading = Instant::now();
    let g = Knn::<ID, RandomState>::load(&fp, &dist_fn).unwrap();
    println!("Finished loading graph in {:?}", start_loading.elapsed());
    g
  } else {
    load_texmex_to_dense(subset_size, &dist_fn, args.clone())
  };

  let mut prng = Xoshiro256StarStar::seed_from_u64(1);
  let mut line_writer = open_csv(args.clone());

  let recall = recall_at_r(
    &g,
    &query_vecs,
    &ground_truth,
    10,
    &mut prng,
    &mut line_writer,
    args.clone(),
  );
  println!("Recall@10: {recall}");
  if let Some(fp) = args.save_graph_to {
    println!("Saving graph to {}", fp);
    let start_saving = Instant::now();
    g.save(&fp).unwrap();
    println!("Finished saving graph in {:?}", start_saving.elapsed());
  }
  pause();
}

// /// Just allocate the core data structures directly, fill them with junk, and
// /// see what our RES is by pausing at the end. By process of elimination, we
// /// can figure out what is going on here.
// fn memory_usage_experiment(subset_size: usize, out_degree: usize) {
//   let mut prng = Xoshiro256StarStar::seed_from_u64(1);

//   let to = vec![0; subset_size * out_degree];

//   let distance = vec![0.0; subset_size * out_degree];

//   let crowding_factor = vec![0; subset_size * out_degree];

//   let mut edges = EdgeVec {
//     to,
//     distance,
//     crowding_factor,
//   };

//   for i in 0..subset_size * out_degree {
//     let edge = edges.get_mut(i).unwrap();
//     *edge.crowding_factor = 1;
//     *edge.to = prng.next_u32() % subset_size as u32;
//     *edge.distance = (prng.next_u32() % (subset_size as u32)) as f32;
//   }

//   let mut internal_to_external_ids = Vec::with_capacity(subset_size);
//   for _ in 0..subset_size {
//     internal_to_external_ids
//       .push(Some(ID::Base(prng.next_u32() % subset_size as u32)));
//   }

//   let mut backpointers = Vec::with_capacity(subset_size);

//   for i in 0..subset_size {
//     backpointers.push(SetU32::new());
//     // NOTE: nothing guarantees out_degree elements in backpointers. Could be
//     // larger or smaller. This is a simplifying assumption.
//     for _ in 0..out_degree {
//       backpointers[i].insert(prng.next_u32() % subset_size as u32);
//     }
//   }

//   //TODO: not sure if backpointers.mem_used already includes this. Try it both ways.
//   let mut backpointers_total_mem = subset_size * mem::size_of::<SetU32>();

//   for i in 0..subset_size {
//     backpointers_total_mem += backpointers[i].mem_used();
//   }

//   let struct_of_arrays_expected_size =
//     out_degree * subset_size * mem::size_of::<u32>()
//       + out_degree * subset_size * mem::size_of::<f32>()
//       + out_degree * subset_size * mem::size_of::<u8>();

//   let internal_to_external_ids_expected_size =
//     subset_size as usize * mem::size_of::<Option<ID>>() as usize;

//   println!(
//     "struct of arrays should be {}",
//     struct_of_arrays_expected_size
//   );

//   println!(
//     "The internal_to_external_ids mapping should take {}",
//     internal_to_external_ids_expected_size,
//   );

//   // NOTE: this is an extreme underestimation.
//   println!(
//     "backpointers expected size (SIGNIFICANTLY UNDERESTIMATED): {}",
//     backpointers_total_mem
//   );

//   println!(
//     "total expected size: {}",
//     struct_of_arrays_expected_size
//       + internal_to_external_ids_expected_size
//       + backpointers_total_mem
//   );

//   pause();
// }

// fn main_mem() {
//   memory_usage_experiment(5_000_000, 7);
// }

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
pub struct ManyKnn<
  'a,
  ID: Eq + std::hash::Hash + Clone,
  S: BuildHasher + Clone + Default,
> {
  pub graphs: Vec<RwLock<Knn<'a, ID, S>>>,
}

impl<
    'a,
    ID: Copy + Ord + std::hash::Hash + Serialize,
    S: BuildHasher + Clone + Default,
  > ManyKnn<'a, ID, S>
{
  pub fn new(
    n: u32,
    config: KnnGraphConfig<S>,
    dist_fn: &'a (dyn Fn(&ID, &ID) -> f32 + Sync),
  ) -> ManyKnn<'a, ID, S> {
    let total_capacity = config.capacity();
    let individual_capacity = total_capacity / n + total_capacity % n;
    let mut graphs = Vec::with_capacity(n as usize);
    for _ in 0..n {
      let config = KnnGraphConfigBuilder::from(config.clone())
        .capacity(individual_capacity)
        .build();
      graphs.push(RwLock::new(Knn::new(config, dist_fn)));
    }
    ManyKnn { graphs }
  }

  pub fn save(&self, path: &str) {
    for (i, g) in &mut self.graphs.iter().enumerate() {
      let i_path = format!("{}_{}", path, i);
      let g = g.read().unwrap();
      g.save(&i_path).unwrap();
    }
  }

  pub fn load<
    U: Serialize
      + Clone
      + Ord
      + PartialOrd
      + Eq
      + Hash
      + for<'de> Deserialize<'de>,
  >(
    path: &str,
    n: u32,
    dist_fn: &'a (dyn Fn(&U, &U) -> f32 + Sync),
  ) -> ManyKnn<'a, U, S> {
    let mut graphs = Vec::new();
    for i in 0..n {
      let i_path = format!("{}_{}", path, i);
      let g = Knn::<U, S>::load(&i_path, dist_fn).unwrap();
      graphs.push(RwLock::new(g));
    }
    ManyKnn { graphs }
  }
}

impl<
    'a,
    T: Copy + Ord + Eq + std::hash::Hash + Send + Sync,
    S: BuildHasher + Clone + Send + Sync + Default,
  > NN<T> for ManyKnn<'a, T, S>
{
  fn insert<R: RngCore>(&mut self, x: T, prng: &mut R) -> Result<(), Error> {
    let i = prng.next_u32() % self.graphs.len() as u32;
    let mut g = self.graphs[i as usize].write().unwrap();
    g.insert(x, prng)
  }

  fn delete<R: RngCore>(&mut self, x: T, prng: &mut R) -> Result<(), Error> {
    for g in &self.graphs {
      let mut g = g.write().unwrap();
      g.delete(x, prng)?
    }
    Ok(())
  }

  fn query<R: RngCore>(
    &self,
    q: &T,
    max_results: usize,
    _prng: &mut R,
  ) -> Result<SearchResults<T>, Error> {
    let mut results = (0..self.graphs.len())
      .into_par_iter()
      .map(|i| {
        self.graphs[i].read().unwrap().query(
          q,
          max_results,
          &mut rand::thread_rng(),
        )
      })
      .try_reduce(SearchResults::<T>::default, |acc, e| Ok(acc.merge(&e)))?;

    results.approximate_nearest_neighbors.truncate(max_results);
    Ok(results)
  }
}

#[repr(C, packed)]
struct Test {
  a: u32,
  b: f32,
  d: u8,
}

fn load_texmex_to_dense_par(
  subset_size: u32,
  num_graphs: u32,
  dist_fn: &(dyn Fn(&ID, &ID) -> f32 + Sync),
  args: Args,
) -> ManyKnn<ID, RandomState> {
  let start_allocate_graph = Instant::now();

  let build_hasher = RandomState::new();

  let out_degree = args.out_degree;
  println!(
    "Size of edges array will be {}",
    out_degree as usize
      * subset_size as usize
      * mem::size_of::<(u32, f32, u8)>()
  );

  println!(
    "If we used a custom struct with packed, it would be {}",
    out_degree as usize * subset_size as usize * mem::size_of::<Test>()
  );

  println!(
    "If we switched to Vec<(u32, f32)>, Vec<u8>, it would be {}",
    out_degree as usize * subset_size as usize * mem::size_of::<(u32, f32)>()
      + out_degree as usize * subset_size as usize * mem::size_of::<u8>()
  );

  println!(
    "If we switched to Vec<(u32, f16)>, Vec<u8>, it would be {}",
    out_degree as usize * subset_size as usize * mem::size_of::<(u32, u16)>()
      + out_degree as usize * subset_size as usize * mem::size_of::<u8>()
  );

  println!(
    "If we switched to struct of arrays, it would be {}",
    out_degree as usize * subset_size as usize * mem::size_of::<u32>()
      + out_degree as usize * subset_size as usize * mem::size_of::<f32>()
      + out_degree as usize * subset_size as usize * mem::size_of::<u8>()
  );

  println!(
    "If we switched to f16 stored distances, it would be {}",
    out_degree as usize * subset_size as usize * mem::size_of::<u32>()
      + out_degree as usize * subset_size as usize * mem::size_of::<u16>()
      + out_degree as usize * subset_size as usize * mem::size_of::<u8>()
  );

  println!(
    "If we always recomputed distances and didn't store them, it would be {}",
    out_degree as usize * subset_size as usize * mem::size_of::<u32>()
      + out_degree as usize * subset_size as usize * mem::size_of::<u8>()
  );

  println!(
    "The internal_to_external_ids mapping should take {}",
    subset_size as usize * mem::size_of::<Option<ID>>()
  );

  let config = KnnGraphConfigBuilder::new(
    subset_size,
    out_degree,
    args.num_searchers,
    build_hasher,
  )
  .use_rrnp(args.rrnp)
  .rrnp_max_depth(args.rrnp_depth)
  .use_lgd(args.lgd)
  .build();

  let g = ManyKnn::new(num_graphs, config, dist_fn);

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

  (0..args.early_exit_after_num_insertions.unwrap_or(subset_size))
    .into_par_iter()
    .progress_with_style(style)
    .for_each(|i| {
      let mut g = g.graphs[i as usize % g.graphs.len()].write().unwrap();
      g.insert(ID::Base(i), &mut rand::thread_rng()).unwrap();
    });
  println!(
    "Finished building the nearest neighbors graph in {:?}",
    start_inserting.elapsed()
  );

  //println!("Graph size: {:?}", g.debug_size_stats());
  g
}

fn main_parallel(args: Args) {
  // NOTE: change gnd_path when you change this
  let subset_size: u32 = args.subset_size;
  let num_graphs: u32 = args.parallel as u32;
  let base_path = Path::new(&args.texmex_path);
  let base_vecs = texmex::Vecs::<u8>::new(base_path, 128).unwrap();
  let query_path = Path::new(&args.query_path);
  let query_vecs = texmex::Vecs::<u8>::new(query_path, 128).unwrap();
  // NOTE: change this path depending on subset_size
  let gnd_path = Path::new(&args.gnd_path);
  let ground_truth = texmex::Vecs::<i32>::new(gnd_path, 1000).unwrap();

  let dist_fn = make_dist_fn(base_vecs, query_vecs);
  let g = if let Some(fp) = args.clone().existing_graph {
    println!("Loading graph from {}", fp);
    let start_load = Instant::now();
    let g = ManyKnn::<ID, RandomState>::load(&fp, num_graphs, &dist_fn);
    println!("Loaded graph in {:?}", start_load.elapsed());
    g
  } else {
    load_texmex_to_dense_par(subset_size, num_graphs, &dist_fn, args.clone())
  };

  let mut prng = Xoshiro256StarStar::seed_from_u64(1);
  let mut line_writer = open_csv(args.clone());
  let recall = recall_at_r(
    &g,
    &query_vecs,
    &ground_truth,
    10,
    &mut prng,
    &mut line_writer,
    args.clone(),
  );
  println!("Recall@10: {recall}");
  if let Some(fp) = args.save_graph_to {
    let start_saving = Instant::now();
    println!("Saving graph to {}", fp);
    g.save(&fp);
    println!("Saved graph in {:?}", start_saving.elapsed());
  }
  pause()
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
  #[arg(long)]
  parallel: u8,
  #[arg(long)]
  rrnp: bool,
  #[arg(long)]
  lgd: bool,
  #[arg(long, default_value_t = 2)]
  rrnp_depth: u32,
  #[arg(long, default_value_t = 7)]
  out_degree: u8,
  #[arg(long, default_value_t = 7)]
  num_searchers: u32,
  #[arg(long, default_value = "search_stats.csv")]
  output_csv: String,
  #[arg(long, default_value = "default")]
  experiment_name: String,
  #[arg(long, default_value = "/mnt/970_pro/anns/bigann_base.bvecs_array")]
  texmex_path: String,
  #[arg(long, default_value_t = 5_000_000)]
  subset_size: u32,
  #[arg(long, default_value = "/mnt/970_pro/anns/bigann_query.bvecs_array")]
  query_path: String,
  #[arg(long, default_value = "/mnt/970_pro/anns/gnd/idx_5M.ivecs_array")]
  gnd_path: String,
  #[arg(long)]
  early_exit_after_num_insertions: Option<u32>,
  #[arg(long)]
  existing_graph: Option<String>,
  #[arg(long)]
  save_graph_to: Option<String>,
}

fn main() {
  let args = Args::parse();

  if args.parallel > 1 {
    main_parallel(args);
  } else {
    main_single_threaded(args);
  }
}
