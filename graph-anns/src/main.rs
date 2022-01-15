#![feature(total_cmp)]
extern crate atomic_float;
extern crate nix;
extern crate parking_lot;
extern crate tinyset;
extern crate rand;
extern crate rand_core;
extern crate rand_xoshiro;
extern crate rayon;

use std::cmp::Ordering;
use std::collections::binary_heap::BinaryHeap;
use std::thread;
use std::time::Instant;
use rand_xoshiro::Xoshiro256StarStar;
use rand_core::SeedableRng;
use rayon::prelude::*;

mod knn_graph;
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
      let heap: &mut BinaryHeap<SearchResult> = nearest_neighbors.get_mut(q).unwrap();
      heap.push(SearchResult::new(i, dist));
      while heap.len() > k {
        heap.pop().unwrap();
      }
    }
  }
  nearest_neighbors
}

fn _search_identity_elem(query_set_size: usize) -> Vec<BinaryHeap<SearchResult>> {
  let mut nearest_neighbors = Vec::new();
  for _ in 0..query_set_size {
    nearest_neighbors.push(BinaryHeap::new());
  }
  nearest_neighbors
}

fn _search_sum(k: usize, mut v1: Vec<BinaryHeap<SearchResult>>, mut v2: Vec<BinaryHeap<SearchResult>>) -> Vec<BinaryHeap<SearchResult>> {
  for (x,y) in v1.iter_mut().zip(v2.iter_mut()) {
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
  mut nearest_neighbors: Vec<BinaryHeap<SearchResult>>
) -> Vec<BinaryHeap<SearchResult>> {

  for q in 0..query_set_size {
    let dist = dist_fn(&db[i], &query_set[q]);
    let heap: &mut BinaryHeap<SearchResult> = nearest_neighbors.get_mut(q).unwrap();
    heap.push(SearchResult::new(i, dist));
    while heap.len() > k {
      heap.pop().unwrap();
    }
  }
  nearest_neighbors
  }

fn _search_rayon<'a, T: ?Sized, C: std::ops::Index<usize, Output = T> + Sync>(
  query_set: &C,
  query_set_size: usize,
  db: &C,
  db_size: usize,
  k: usize,
  // TODO: parametrize the type of the distances so we can use much faster
  // i32 if possible.
  dist_fn: fn(&T, &T) -> f32,
) -> Vec<BinaryHeap<SearchResult>> {
  let nearest_neighbors =
      (0..db_size)
      .into_par_iter()
      // NOTE: fold + reduce ensures that we don't get swamped with the overhead
      // of allocating one billion BinaryHeaps.
      .fold(|| _search_identity_elem(query_set_size),
            |mut nns: Vec<BinaryHeap<SearchResult>>, i: usize| {
              _search_inject(query_set, query_set_size, db, i, k, dist_fn, nns)
            })
      .reduce(|| _search_identity_elem(query_set_size),
              |mut v1: Vec<BinaryHeap<SearchResult>>, v2: Vec<BinaryHeap<SearchResult>>| {
                _search_sum(k, v1,v2)
              });

  nearest_neighbors
}

fn _main_old() {
  let start = Instant::now();
  let base_vecs = texmex::Vecs::<u8>::new("/mnt/970pro/anns/bigann_base.bvecs_array", 128).unwrap();
  let query_vecs =
    texmex::Vecs::<u8>::new("/mnt/970pro/anns/bigann_query.bvecs_array_one_point", 128).unwrap();
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
  let base_vecs = texmex::Vecs::<u8>::new("/mnt/970pro/anns/bigann_base.bvecs_array", 128).unwrap();
  let query_vecs =
    texmex::Vecs::<u8>::new("/mnt/970pro/anns/bigann_query.bvecs_array_one_point", 128).unwrap();
  println!("Loaded dataset in {:?}", start.elapsed());

  _search_rayon(&query_vecs, query_vecs.num_rows, &base_vecs, base_vecs.num_rows, 1000, texmex::sq_euclidean_faster);
}

fn main() {
  let mmap_start = Instant::now();
  let base_vecs = texmex::Vecs::<u8>::new("/mnt/970pro/anns/bigann_base.bvecs_array", 128).unwrap();
  println!("mmaped dataset in {:?}", mmap_start.elapsed());

  let rand_init_graph_start = Instant::now();
  let mut prng = Xoshiro256StarStar::seed_from_u64(1);
  let (g, bp) = knn_graph::random_init(base_vecs.num_rows as u32, 5, &mut prng, &base_vecs,
    texmex::sq_euclidean_faster, 2);
  println!("Initialized knn_graph in {:?}", rand_init_graph_start.elapsed());
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
