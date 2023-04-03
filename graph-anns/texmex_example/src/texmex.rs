use nix::sys::mman::{mmap, MapFlags, ProtFlags};
use serde::{Deserialize, Serialize};
use std::convert::TryInto;
use std::fs::File;
use std::os::unix::io::IntoRawFd;

// A handle to a file in the bvecs_array, ivecs_array, or fvecs_array format on
// disk. The file is mmaped. The associated functions are used to read vectors
// from the file.
//
// The files referenced are modified versions of the k-nn benchmarks
// available at http://corpus-texmex.irisa.fr/index.html
// See README.md for more information about how this file format works.
#[derive(Debug, Clone, Copy)]
pub struct Vecs<'a, T> {
  pub num_rows: usize,
  pub num_dim: usize,
  buffer: &'a [T],
}

impl<T> Vecs<'_, T> {
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
        // NOTE: because our dataset is close to our max RAM size,
        // parts of our algorithm try to avoid random access across the
        // entire thing, so MAP_POPULATE is counter-productive -- it loads
        // parts of the file that will just get unloaded again.
        MapFlags::MAP_PRIVATE,
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
}

impl<'a, T> std::ops::Index<usize> for Vecs<'a, T> {
  type Output = [T];

  fn index(self: &Vecs<'a, T>, i: usize) -> &'a Self::Output {
    &self.buffer[i * self.num_dim..i * self.num_dim + self.num_dim]
  }
}

// TODO: see commit 5a2b1254fe058c1d23a52d6f16f02158e31744e2 for why this
// PrimitiveToF32 mess is necessary. .into() is much slower.

pub trait PrimitiveToF32 {
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

// NOTE: this gets auto-vectorized at opt-level 3 and is fastest
pub fn sq_euclidean_iter(v1: &[u8], v2: &[u8]) -> f32 {
  v1.iter().zip(v2.iter()).fold(0i32, |acc, (x, y)| {
    acc + (*x as i32 - *y as i32) * (*x as i32 - *y as i32)
  }) as f32
}

/// ID for a vector in a texmex benchmark problem. Because these datasets push
/// the limit of my machine's memory, we mmap the vectors to a file, and only
/// refer to them by these small IDs. A vector can occur in either the base_vecs
/// or the query_vecs, but not both. The ID is the index into the appropriate
/// Vecs struct.
#[derive(
  Clone,
  Copy,
  Debug,
  PartialEq,
  Eq,
  PartialOrd,
  Ord,
  Hash,
  Serialize,
  Deserialize,
)]
pub enum ID {
  Base(u32),
  Query(u32),
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_sq_euclidean_iter() {
    let v1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let v2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(sq_euclidean_iter(&v1, &v2), 0.0);
    let v1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let v2 = [2, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(sq_euclidean_iter(&v1, &v2), 1.0);
    let v1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let v2 = [2, 2, 3, 4, 5, 6, 7, 8, 9, 11];
    assert_eq!(sq_euclidean_iter(&v1, &v2), 2.0);
  }
}
