#![feature(total_cmp)]
extern crate memmap;
use std::cmp::Ordering;
use std::collections::binary_heap::BinaryHeap;
use std::convert::TryInto;
use std::fs::File;

// A handle to a file in the bvecs_array, ivecs_array, or fvecs_array format on
// disk. The file is mmaped. The associated functions are used to read vectors
// from the file.
struct Vecs<'a, T> {
  _mmap: memmap::Mmap, // segfaults unless we carry this around.
  pub num_rows: usize,
  pub num_dim: usize,
  buffer: &'a [T],
}

impl<'a, T> Vecs<'_, T> {
  pub fn new<P>(path: P, num_dim: usize) -> Result<Self, String>
  where
    P: AsRef<std::path::Path>,
  {
    let f = File::open(path).expect("Failed to open file");
    let filesize: usize = f
      .metadata()
      .expect("Failed to read file size")
      .len()
      .try_into()
      .unwrap();
    let num_rows = filesize / (num_dim * std::mem::size_of::<T>());
    let _mmap = unsafe { memmap::MmapOptions::new().map(&f) }.expect("mmap failed");
    let buffer =
      unsafe { std::slice::from_raw_parts(_mmap.as_ptr() as *const T, num_rows * num_dim) };
    Ok(Self {
      _mmap,
      num_rows,
      num_dim,
      buffer,
    })
  }

  pub fn get(&'a self, i: usize) -> &'a [T] {
    &self.buffer[i * self.num_dim..i * self.num_dim + self.num_dim]
  }
}

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

fn search_range<'a, T>(
  query_set: Vecs<'a, T>,
  db: Vecs<'a, T>,
  k: usize,
  lower_bound_incl: usize,
  upper_bound_excl: usize,
  dist_fn: fn(&[T], &[T]) -> f32,
) -> Vec<BinaryHeap<SearchResult>> {
  let mut nearest_neighbors = Vec::new();
  for _ in 0..query_set.num_rows {
    nearest_neighbors.push(BinaryHeap::new());
  }
  for i in lower_bound_incl..upper_bound_excl {
    for q in 0..query_set.num_rows {
      let dist = dist_fn(db.get(i), query_set.get(q));
      let heap: &mut BinaryHeap<SearchResult> = nearest_neighbors.get_mut(q).unwrap();
      heap.push(SearchResult::new(i, dist));
      while heap.len() > k {
        heap.pop().unwrap();
      }
    }
  }
  nearest_neighbors
}

fn sq_euclidean<T: Into<f32> + Copy>(v1: &[T], v2: &[T]) -> f32 {
  let mut result = 0.0;
  let n = v1.len();
  for i in 0..n {
    let diff = v2[i].into() - v1[i].into();
    result += diff * diff;
  }
  result
}

fn main() {
  let base_vecs = Vecs::<u8>::new("/mnt/970pro/anns/bigann_base.bvecs_array", 128).unwrap();
  let query_vecs =
    Vecs::<u8>::new("/mnt/970pro/anns/bigann_query.bvecs_array_one_point", 128).unwrap();
  search_range(query_vecs, base_vecs, 1000, 0, 999999999, sq_euclidean);
}
