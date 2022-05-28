extern crate criterion;
use std::collections::HashSet;
use std::hash::BuildHasherDefault;
use std::iter::FromIterator;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use criterion::{BatchSize, BenchmarkId};

extern crate graph_anns;
extern crate nohash_hasher;
extern crate rand_core;
extern crate rand_xoshiro;
use graph_anns::*;
use nohash_hasher::NoHashHasher;
use rand_core::RngCore;
use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

type NHH = BuildHasherDefault<NoHashHasher<u32>>;

/// Hamming distance between two random 640-bit vectors which are uniquely
/// identified by x and y.
fn hamming_dist(x: &u32, y: &u32) -> f32 {
  let mut x_prng = Xoshiro256StarStar::seed_from_u64(*x as u64);
  let mut y_prng = Xoshiro256StarStar::seed_from_u64(*y as u64);
  let mut result = 0;
  for _ in 0..10 {
    let x_rand = x_prng.next_u64();
    let y_rand = y_prng.next_u64();
    result += (x_rand ^ y_rand).count_ones();
  }
  return result as f32;
}

fn construct_graph(
  n: u32,
  capacity: u32,
) -> DenseKNNGraph<u32, nohash_hasher::BuildNoHashHasher<u32>> {
  let ids = Vec::<u32>::from_iter(0..n);
  let g = exhaustive_knn_graph(
    ids.iter().collect(),
    capacity,
    5,
    &|x, y| hamming_dist(x, y),
    nohash_hasher::BuildNoHashHasher::default(),
  );
  return g;
}

fn bench_construct_exhaustive_graph(c: &mut Criterion) {
  let mut group = c.benchmark_group("exhaustive_knn_graph");
  for n in [100, 500, 1000].iter() {
    group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
      b.iter(|| construct_graph(n, n))
    });
  }
}

fn bench_insert_one(c: &mut Criterion) {
  // NOTE: intentionally not creating a new prng for each iteration, because I
  // want to include search randomness in our sampling.
  let mut prng = Xoshiro256StarStar::seed_from_u64(12);

  let mut group = c.benchmark_group("insert_one");
  group.measurement_time(Duration::from_secs(60));

  for n in [500, 1000, 10000].iter() {
    group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
      let mk_g = || construct_graph(n, n + 1);
      b.iter_batched(
        mk_g,
        |mut g| {
          insert_approx(&mut g, n, &|i| hamming_dist(&(n + 1), i), 5, &mut prng)
        },
        BatchSize::SmallInput,
      )
    });
  }
}

fn construct_graph_approx_iterative(n: u32) {
  let mut prng = Xoshiro256StarStar::seed_from_u64(12);
  let ids = Vec::<u32>::from_iter(0..50);
  let mut g: DenseKNNGraph<u32, NHH> = exhaustive_knn_graph(
    ids.iter().collect(),
    n,
    5,
    &hamming_dist,
    nohash_hasher::BuildNoHashHasher::default(),
  );
  for q in 50..n {
    insert_approx(&mut g, q, &|i| hamming_dist(&q, i), 5, &mut prng);
  }
}

fn bench_construct_graph_approx_iterative(c: &mut Criterion) {
  // NOTE: intentionally not creating a new prng for each iteration, because I
  // want to include search randomness in our sampling.

  let mut group = c.benchmark_group("construct_graph_approx_iterative");
  group.measurement_time(Duration::from_secs(60));

  for n in [500, 1000, 10000, 1000_000].iter() {
    group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
      b.iter(|| construct_graph_approx_iterative(n))
    });
  }
}

criterion_group!(
  benches,
  // bench_construct_exhaustive_graph,
  // bench_insert_one,
  bench_construct_graph_approx_iterative
);
criterion_main!(benches);
