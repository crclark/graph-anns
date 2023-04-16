#![allow(clippy::unwrap_used)]

extern crate criterion;
use std::hash::BuildHasherDefault;
use std::iter::FromIterator;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, SamplingMode};
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

type Nhh = BuildHasherDefault<NoHashHasher<u32>>;

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
  result as f32
}

fn mk_config(capacity: u32) -> KnnGraphConfig<Nhh> {
  let out_degree = 5;
  let num_searchers = 5;
  let use_rrnp = false;
  let rrnp_max_depth = 2;
  let use_lgd = false;
  let build_hasher = nohash_hasher::BuildNoHashHasher::default();
  KnnGraphConfigBuilder::new(capacity, out_degree, num_searchers, build_hasher)
    .use_rrnp(use_rrnp)
    .rrnp_max_depth(rrnp_max_depth)
    .use_lgd(use_lgd)
    .build()
}

fn construct_graph<'a>(
  n: u32,
  capacity: u32,
) -> Knn<'a, u32, nohash_hasher::BuildNoHashHasher<u32>> {
  let ids = Vec::<u32>::from_iter(0..n);
  let mut prng = Xoshiro256StarStar::seed_from_u64(1);
  exhaustive_knn_graph(
    ids.iter().collect(),
    mk_config(capacity),
    &hamming_dist,
    &mut prng,
  )
  .unwrap()
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

  for n in [500, 1000].iter() {
    group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
      let mk_g = || construct_graph(n, n + 1);
      b.iter_batched(
        mk_g,
        |mut g| g.insert(n, &mut prng),
        BatchSize::SmallInput,
      )
    });
  }
}

fn construct_graph_approx_iterative(n: u32) {
  let mut prng = Xoshiro256StarStar::seed_from_u64(12);
  let ids = Vec::<u32>::from_iter(0..50);
  let mut g: Knn<u32, Nhh> = exhaustive_knn_graph(
    ids.iter().collect(),
    mk_config(n),
    &hamming_dist,
    &mut prng,
  )
  .unwrap();
  for q in 50..n {
    g.insert(q, &mut prng).unwrap();
  }
}

fn bench_construct_graph_approx_iterative(c: &mut Criterion) {
  // NOTE: intentionally not creating a new prng for each iteration, because I
  // want to include search randomness in our sampling.

  let mut group = c.benchmark_group("construct_graph_approx_iterative");
  group.measurement_time(Duration::from_secs(60));
  group.sampling_mode(SamplingMode::Flat);

  let n = &500;
  group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
    b.iter(|| construct_graph_approx_iterative(n))
  });
}

criterion_group!(
  benches,
  bench_construct_exhaustive_graph,
  bench_insert_one,
  bench_construct_graph_approx_iterative
);
criterion_main!(benches);
