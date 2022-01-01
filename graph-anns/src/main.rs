#![feature(total_cmp)]
extern crate memmap;
extern crate nix;
use nix::sys::mman::{mmap, MapFlags, ProtFlags};
use std::cmp::Ordering;
use std::collections::binary_heap::BinaryHeap;
use std::convert::TryInto;
use std::fs::File;
use std::os::unix::io::IntoRawFd;
use std::thread;
use std::time::Instant;

// A handle to a file in the bvecs_array, ivecs_array, or fvecs_array format on
// disk. The file is mmaped. The associated functions are used to read vectors
// from the file.
#[derive(Debug, Clone, Copy)]
struct Vecs<'a, T> {
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
    let buffer = unsafe {
      let mmap = mmap(
        std::ptr::null_mut(),
        filesize,
        ProtFlags::PROT_READ,
        MapFlags::MAP_PRIVATE | MapFlags::MAP_POPULATE,
        f.into_raw_fd(),
        0,
      )
      .expect("mmap failed");
      std::slice::from_raw_parts(mmap as *const T, num_rows * num_dim)
    };
    Ok(Self {
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
  query_set: &Vecs<'a, T>,
  db: &Vecs<'a, T>,
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

trait PrimitiveToF32 {
  fn tof32(self) -> f32;
}

impl PrimitiveToF32 for u8 {
  fn tof32(self) -> f32 {
    self as f32
  }
}

impl PrimitiveToF32 for i32 {
  fn tof32(self) -> f32 {
    self as f32
  }
}

impl PrimitiveToF32 for f32 {
  fn tof32(self) -> f32 {
    self
  }
}

fn sq_euclidean_faster<T: PrimitiveToF32 + Copy>(v1: &[T], v2: &[T]) -> f32 {
  let mut result = 0.0;
  let n = v1.len();
  for i in 0..n {
    let diff = v2[i].tof32() - v1[i].tof32();
    result += diff * diff;
  }
  result
}

fn main() {
  let start = Instant::now();
  let base_vecs = Vecs::<u8>::new("/mnt/970pro/anns/bigann_base.bvecs_array", 128).unwrap();
  let query_vecs =
    Vecs::<u8>::new("/mnt/970pro/anns/bigann_query.bvecs_array_one_point", 128).unwrap();
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
      search_range(
        &query_vecs,
        &base_vecs,
        k,
        lower_bound_clone,
        upper_bound,
        sq_euclidean_faster,
      );
    }));
    lower_bound += rows_per_thread;
  }

  for handle in handles {
    handle.join().unwrap();
  }
}
