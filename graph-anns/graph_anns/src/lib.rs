#![deny(missing_docs)]
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]
#![warn(clippy::panic_in_result_fn)]
#![warn(non_fmt_panics)]

//! An implementation of the approximate nearest-neighbor search data structure
//! and algorithms
//! described in W. -L. Zhao, H. Wang and C. -W. Ngo, "Approximate k-NN Graph
//! Construction: A Generic Online Approach," in IEEE Transactions on Multimedia,
//! vol. 24, pp. 1909-1921, 2022, doi: 10.1109/TMM.2021.3073811.
//!
//! This library is intended to be a flexible component in a system that
//! uses approximate nearest neighbor search. It is not a complete system
//! on its own, and requires a bit of glue code to be useful.
//!
//! Interesting features include:
//! - A generic interface for nearest neighbor search on arbitrary distance
//! functions.
//! - Incremental insertion and deletion of elements.
//! - Relatively fast.
//!
//! To get started, use the [KnnGraphConfigBuilder] to create a new nearest
//! neighbor search structure with [Knn::new]. For vectors of floats, you can use
//! the ordered_float crate to ensure that your floats are not NaN and to
//! satisfy the trait requirements of this library.
//!
//! Minimal example:
//! ```
//! extern crate graph_anns;
//! extern crate ordered_float;
//! extern crate rand_xoshiro;
//!
//! use graph_anns::{Knn, KnnGraphConfigBuilder, NN, SearchResults};
//! use ordered_float::NotNan;
//! use rand_xoshiro::rand_core::SeedableRng;
//! use rand_xoshiro::Xoshiro256StarStar;
//!
//! use std::collections::hash_map::RandomState;
//!
//! let dist_fn = &|x: &Vec<NotNan<f32>>, y: &Vec<NotNan<f32>>| {
//! let mut sum = 0.0;
//! for i in 0..x.len() {
//!   sum += (x[i] - y[i]).powi(2);
//! }
//! sum
//! };
//!
//! let conf = KnnGraphConfigBuilder::new(5, 3, 1, Default::default())
//! .use_rrnp(true)
//! .rrnp_max_depth(2)
//! .use_lgd(true)
//! .build();
//!
//! let mut g: Knn<Vec<NotNan<f32>>, RandomState> = Knn::new(conf, dist_fn);
//!
//! let mut prng = Xoshiro256StarStar::seed_from_u64(1);
//!
//! let example_vec = vec![
//! NotNan::new(1f32).unwrap(),
//! NotNan::new(2f32).unwrap(),
//! NotNan::new(3f32).unwrap(),
//! NotNan::new(4f32).unwrap(),
//! ];
//!
//! g.insert(example_vec.clone(), &mut prng).unwrap();
//!
//! let SearchResults {
//! approximate_nearest_neighbors: nearest_neighbors,
//! ..
//! } = g.query(
//! &vec![
//!   NotNan::new(34f32).unwrap(),
//!   NotNan::new(5f32).unwrap(),
//!   NotNan::new(53f32).unwrap(),
//!   NotNan::new(312f32).unwrap(),
//! ],
//! 1,
//! &mut prng,
//! ).unwrap();
//! assert_eq!(nearest_neighbors[0].item, example_vec);
//! ```
extern crate bincode;
extern crate is_sorted;
extern crate rand;
extern crate rand_xoshiro;
extern crate serde;
extern crate soa_derive;
extern crate thiserror;

use rand::RngCore;
use serde::Deserialize;
use serde::Serialize;
use std::cmp::max;
use std::collections::binary_heap::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::hash::BuildHasher;
use std::io::BufWriter;

mod edge;

mod error;
pub use error::*;

#[allow(unused_imports)]
use edge::EdgeSlice;

mod graph_knn;
pub use graph_knn::*;

mod search_results;
pub use search_results::*;

mod id_mapping;
mod space_report;
pub use space_report::*;

// TODO: revisit every use of `pub` in this file.

// TODO: many benchmarking utilities. Recall@n  metrics and so forth.

// NOTE: can't do a Default because we can't guess a reasonable capacity.
// TODO: auto-resize like a Vec?

/// A trait for a nearest neighbor search data structure.
pub trait NN<T> {
  // TODO: more functions in this interface.

  /// Insert a new element into the nearest neighbor data structure.
  fn insert<R: RngCore>(&mut self, x: T, prng: &mut R) -> Result<(), Error>;
  /// Delete an element from the nearest neighbor data structure.
  fn delete<R: RngCore>(&mut self, x: T, prng: &mut R) -> Result<(), Error>;
  /// Query the nearest neighbor data structure for the approximate
  /// nearest neighbors of the given element.
  fn query<R: RngCore>(
    &self,
    q: &T,
    max_results: usize,
    prng: &mut R,
  ) -> Result<SearchResults<T>, Error>;
}

/// A nearest neighbor search data structure that uses exhaustive search.
#[derive(Deserialize, Serialize)]
struct ExhaustiveKnn<T: Eq + std::hash::Hash> {
  /// All inserted elements.
  pub contents: HashSet<T>,
  // TODO: made this Sync so that I could share a single closure across threads
  // when splitting up a graph into multiple pieces. Is this going to be onerous
  // to users?
}

impl<T: Clone + Eq + std::hash::Hash> ExhaustiveKnn<T> {
  fn new() -> ExhaustiveKnn<T> {
    ExhaustiveKnn {
      contents: HashSet::new(),
    }
  }

  fn debug_size_stats(&self) -> SpaceReport {
    SpaceReport::default()
  }

  fn insert(&mut self, x: T) {
    self.contents.insert(x);
  }

  fn delete(&mut self, x: T) {
    self.contents.remove(&x);
  }

  fn query(
    &self,
    q: &T,
    max_results: usize,
    distance: &(dyn Fn(&T, &T) -> f32 + Sync),
  ) -> SearchResults<T> {
    let mut nearest_neighbors_max_dist_heap: BinaryHeap<SearchResult<T>> =
      BinaryHeap::new();
    let mut visited_nodes = Vec::new();
    for x in self.contents.iter() {
      let dist = distance(x, q);
      let search_result: SearchResult<T> =
        SearchResult::new(x.clone(), None, dist, 0, 0);
      visited_nodes.push(search_result.clone());
      nearest_neighbors_max_dist_heap.push(search_result);
      if nearest_neighbors_max_dist_heap.len() > max_results {
        nearest_neighbors_max_dist_heap.pop();
      }
    }
    SearchResults {
      approximate_nearest_neighbors: nearest_neighbors_max_dist_heap
        .into_sorted_vec(),
      visited_nodes,
      visited_nodes_distances_to_q: HashMap::new(),
      search_stats: None,
    }
  }
}

fn convert_bruteforce_to_dense<
  'a,
  T: Clone + Eq + std::hash::Hash,
  S: BuildHasher + Clone + Default,
  R: RngCore,
>(
  bf: &mut ExhaustiveKnn<T>,
  config: KnnGraphConfig<S>,
  dist_fn: &'a (dyn Fn(&T, &T) -> f32 + Sync),
  prng: &mut R,
) -> Result<DenseKnnGraph<T, S>, Error> {
  let ids = bf.contents.iter().collect();
  let KnnInner::Large(g) = exhaustive_knn_graph(ids, config, dist_fn, prng)?.inner
  else {
    return Err(Error::InternalError(
      "internal error: exhaustive_knn_graph returned a small graph".to_string(),
    ));
  };
  Ok(g)
}

fn convert_dense_to_bruteforce<
  T: Clone + Eq + std::hash::Hash,
  S: BuildHasher + Clone + Default,
>(
  g: &mut DenseKnnGraph<T, S>,
) -> ExhaustiveKnn<T> {
  let mut contents = HashSet::new();
  for (ext_id, _) in g.mapping.external_to_internal_ids.drain() {
    contents.insert(ext_id);
  }
  ExhaustiveKnn { contents }
}

/// The primary data structure for approximate nearest neighbor search exposed
/// by this library.
/// This is a wrapper around either a brute-force exhaustive search
/// or a graph-based search, depending on the number of elements inserted. The
/// switch to a graph-based search is triggered when at least max(out_degree,
/// num_searchers) elements are inserted.
pub struct Knn<
  'a,
  T: Clone + Eq + std::hash::Hash,
  S: BuildHasher + Clone + Default,
> {
  inner: KnnInner<T, S>,
  dist_fn: &'a (dyn Fn(&T, &T) -> f32 + Sync),
}

impl<
    'a,
    T: Clone + Ord + Eq + std::hash::Hash + Serialize,
    S: BuildHasher + Clone + Default,
  > Knn<'a, T, S>
{
  /// Save the graph to a file.
  pub fn save(&self, path: &str) -> Result<(), bincode::Error> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(&file);
    bincode::serialize_into(&mut writer, &self.inner)?;
    Ok(())
  }

  /// Serialize the graph to a byte vector.
  pub fn serialize(&self) -> Result<Vec<u8>, bincode::Error> {
    bincode::serialize(&self.inner)
  }

  /// Serialize the graph to a writer.
  pub fn serialize_into(
    &self,
    writer: &mut dyn std::io::Write,
  ) -> Result<(), bincode::Error> {
    bincode::serialize_into(writer, &self.inner)
  }

  /// Load a graph from a file.
  pub fn load<U: Clone + Eq + std::hash::Hash + for<'de> Deserialize<'de>>(
    path: &str,
    dist_fn: &'a (dyn Fn(&U, &U) -> f32 + Sync),
  ) -> Result<Knn<'a, U, S>, bincode::Error> {
    let file = File::open(path)?;
    let mut reader = std::io::BufReader::new(&file);
    let inner = bincode::deserialize_from(&mut reader)?;
    Ok(Knn { inner, dist_fn })
  }

  /// Deserialize a graph from a byte slice.
  pub fn deserialize<
    U: Clone + Eq + std::hash::Hash + for<'de> Deserialize<'de>,
  >(
    bytes: &[u8],
    dist_fn: &'a (dyn Fn(&U, &U) -> f32 + Sync),
  ) -> Result<Knn<'a, U, S>, bincode::Error> {
    let inner = bincode::deserialize(bytes)?;
    Ok(Knn { inner, dist_fn })
  }
}

impl<
    'a,
    T: Clone + Ord + Eq + std::hash::Hash,
    S: BuildHasher + Clone + Default,
  > NN<T> for Knn<'a, T, S>
{
  fn insert<R: RngCore>(&mut self, x: T, prng: &mut R) -> Result<(), Error> {
    self.inner.insert(x, self.dist_fn, prng)
  }

  fn delete<R: RngCore>(&mut self, x: T, prng: &mut R) -> Result<(), Error> {
    self.inner.delete(x, self.dist_fn, prng)
  }

  fn query<R: RngCore>(
    &self,
    q: &T,
    max_results: usize,
    prng: &mut R,
  ) -> Result<SearchResults<T>, Error> {
    self.inner.query(q, max_results, self.dist_fn, prng)
  }
}

impl<
    'a,
    T: Clone + Ord + Eq + std::hash::Hash,
    S: BuildHasher + Clone + Default,
  > Knn<'a, T, S>
{
  /// Initialize a new graph with the given configuration.
  pub fn new(
    config: KnnGraphConfig<S>,
    dist_fn: &'a (dyn Fn(&T, &T) -> f32 + Sync),
  ) -> Knn<'a, T, S> {
    Knn {
      inner: KnnInner::new(config),
      dist_fn,
    }
  }

  /// Returns error if graph structure is internally inconsistent. Used for testing.
  pub fn debug_consistency_check(&self) -> Result<(), Error> {
    self.inner.debug_consistency_check()
  }

  /// Returns information about the length and capacity of all data structures
  /// in the graph.
  pub fn debug_size_stats(&self) -> Result<SpaceReport, Error> {
    self.inner.debug_size_stats()
  }
}

#[derive(Deserialize, Serialize)]
enum KnnInner<T: Clone + Eq + std::hash::Hash, S: BuildHasher + Clone + Default>
{
  Small {
    g: ExhaustiveKnn<T>,
    #[serde(bound(
      serialize = "KnnGraphConfig<S>: Serialize",
      deserialize = "KnnGraphConfig<S>: Deserialize<'de>"
    ))]
    config: KnnGraphConfig<S>,
  },

  #[serde(bound(
    serialize = "DenseKnnGraph<T, S>: Serialize",
    deserialize = "DenseKnnGraph<T, S>: Deserialize<'de>"
  ))]
  Large(DenseKnnGraph<T, S>),
}

impl<
    'a,
    T: Clone + Ord + Eq + std::hash::Hash,
    S: BuildHasher + Clone + Default,
  > KnnInner<T, S>
{
  fn insert<R: RngCore>(
    &mut self,
    x: T,
    dist_fn: &'a (dyn Fn(&T, &T) -> f32 + Sync),
    prng: &mut R,
  ) -> Result<(), error::Error> {
    match self {
      KnnInner::Small { g, config } => {
        if config.capacity as usize == g.contents.len() {
          Err(Error::CapacityExceeded)
        } else if g.contents.len()
          > max(config.out_degree as usize, config.num_searchers as usize)
        {
          *self = KnnInner::Large(convert_bruteforce_to_dense(
            g,
            config.clone(),
            #[allow(clippy::clone_double_ref)]
            dist_fn.clone(),
            prng,
          )?);
          self.insert(x, dist_fn, prng)
        } else {
          g.insert(x);
          Ok(())
        }
      }
      KnnInner::Large(g) => g.insert(x, dist_fn, prng),
    }
  }

  fn delete<R: RngCore>(
    &mut self,
    x: T,
    dist_fn: &'a (dyn Fn(&T, &T) -> f32 + Sync),
    prng: &mut R,
  ) -> Result<(), Error> {
    match self {
      KnnInner::Small { g, .. } => {
        g.delete(x);
        Ok(())
      }
      KnnInner::Large(g) => {
        if g.mapping.external_to_internal_ids.len()
          == g.config.out_degree as usize + 1
        {
          let config = g.config.clone();
          let mut small_g = convert_dense_to_bruteforce(g);
          small_g.delete(x);
          *self = KnnInner::Small { g: small_g, config };
          Ok(())
        } else {
          g.delete(x, dist_fn, prng)
        }
      }
    }
  }

  fn query<R: RngCore>(
    &self,
    q: &T,
    max_results: usize,
    dist_fn: &'a (dyn Fn(&T, &T) -> f32 + Sync),
    prng: &mut R,
  ) -> Result<SearchResults<T>, error::Error> {
    match self {
      KnnInner::Small { g, .. } => Ok(g.query(q, max_results, dist_fn)),
      KnnInner::Large(g) => g.query(q, max_results, dist_fn, prng),
    }
  }
}

impl<T: Clone + Ord + Eq + std::hash::Hash, S: BuildHasher + Clone + Default>
  KnnInner<T, S>
{
  /// Initialize a new graph with the given configuration.
  pub fn new(config: KnnGraphConfig<S>) -> KnnInner<T, S> {
    KnnInner::Small {
      g: ExhaustiveKnn::new(),
      config,
    }
  }

  /// Returns error if graph structure is internally inconsistent. Used for testing.
  pub fn debug_consistency_check(&self) -> Result<(), Error> {
    match self {
      KnnInner::Small { .. } => Ok(()),
      KnnInner::Large(g) => g.consistency_check(),
    }
  }

  /// Returns information about the length and capacity of all data structures
  /// in the graph.
  pub fn debug_size_stats(&self) -> Result<SpaceReport, Error> {
    match self {
      KnnInner::Small { g, .. } => Ok(g.debug_size_stats()),
      KnnInner::Large(g) => g.debug_size_stats(),
    }
  }
}

/// Constructs an exact k-nn graph on the given IDs. O(n^2). This can improve
/// performance if you have a small number of IDs and a large number of queries,
/// but it is not recommended for large numbers of IDs.
/// The capacity set in your config must be greater than the length of the
/// ids vector.
pub fn exhaustive_knn_graph<
  'a,
  T: Clone + Eq + std::hash::Hash,
  S: BuildHasher + Clone + Default,
  R: RngCore,
>(
  ids: Vec<&T>,
  config: KnnGraphConfig<S>,
  dist_fn: &'a (dyn Fn(&T, &T) -> f32 + Sync),
  prng: &mut R,
) -> Result<Knn<'a, T, S>, Error> {
  Ok(Knn {
    inner: KnnInner::Large(exhaustive_knn_graph_internal(
      ids, config, dist_fn, prng,
    )?),
    dist_fn,
  })
}

#[cfg(test)]
mod tests {
  #![allow(clippy::unwrap_used)]
  extern crate nohash_hasher;
  extern crate ordered_float;
  use std::{collections::hash_map::RandomState, hash::BuildHasherDefault};

  use self::nohash_hasher::NoHashHasher;
  use self::ordered_float::NotNan;
  use super::*;
  use rand::SeedableRng;
  use rand_xoshiro::Xoshiro256StarStar;

  type Nhh = BuildHasherDefault<NoHashHasher<u32>>;

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

  pub fn sq_euclidean<T: PrimitiveToF32 + Copy>(v1: &[T], v2: &[T]) -> f32 {
    let mut result = 0.0;
    let n = v1.len();
    for i in 0..n {
      let diff = v2[i].tof32() - v1[i].tof32();
      result += diff * diff;
    }
    result
  }

  fn mk_config(capacity: u32) -> KnnGraphConfig<Nhh> {
    let out_degree = 5;
    let num_searchers = 5;
    let use_rrnp = false;
    let rrnp_max_depth = 2;
    let use_lgd = false;
    let build_hasher = nohash_hasher::BuildNoHashHasher::default();
    KnnGraphConfig::<Nhh> {
      capacity,
      out_degree,
      num_searchers,
      build_hasher,
      use_rrnp,
      rrnp_max_depth,
      use_lgd,
    }
  }

  #[test]
  fn test_beam_search_fully_connected_graph() {
    let db = vec![[1.1f32], [2f32], [3f32], [10f32], [11f32], [12f32]];
    let dist_fn =
      &|x: &u32, y: &u32| sq_euclidean(&db[*x as usize], &db[*y as usize]);
    let mut config = mk_config(10);
    config.out_degree = 2;
    let mut prng = Xoshiro256StarStar::seed_from_u64(1);
    let g: DenseKnnGraph<u32, Nhh> = exhaustive_knn_graph_internal(
      vec![&0, &1, &2, &3, &4, &5],
      config,
      dist_fn,
      &mut prng,
    )
    .unwrap();
    g.consistency_check().unwrap();
    let SearchResults {
      approximate_nearest_neighbors: nearest_neighbors,
      ..
    } = g.query(&1, 2, dist_fn, &mut prng).unwrap();
    assert_eq!(
      nearest_neighbors
        .iter()
        .map(|x| (x.item, x.dist))
        .collect::<Vec<(u32, f32)>>(),
      vec![(1u32, 0.0), (0, 0.9 * 0.9)] //TODO: make test not depend on fp equality
    );
  }

  #[test]
  fn test_insert_from_empty() {
    let db = vec![
      [1f32],
      [2f32],
      [3f32],
      [10f32],
      [11f32],
      [12f32],
      [18f32],
      [19f32],
      [20f32],
      [21f32],
      [22f32],
    ];
    let dist_fn =
      &|x: &u32, y: &u32| sq_euclidean(&db[*x as usize], &db[*y as usize]);
    let mut g: KnnInner<u32, Nhh> = KnnInner::new(mk_config(10));
    let mut prng = Xoshiro256StarStar::seed_from_u64(1);
    g.insert(0, dist_fn, &mut prng).unwrap();

    g.debug_consistency_check().unwrap();
  }

  // An example of inserting vecs of floats directly into the graph, rather than
  // identifiers for the vectors. The downside of this is that the graph
  // internally stores the vectors, while the user most likely is already
  // storing them somewhere else, anyway. The benefit, though, is that querying
  // for vectors not present in the input data set is easier -- you simply pass
  // it in, and then it gets passed directly to your callback. With identifiers,
  // you have to look up the vectors in your distance callback, and maintain
  // some kind of special identifier for the query vector.
  #[test]
  fn test_realistic_usage() {
    let dist_fn = &|x: &Vec<NotNan<f32>>, y: &Vec<NotNan<f32>>| {
      sq_euclidean(
        &x.iter().map(|x| x.into_inner()).collect::<Vec<f32>>(),
        &y.iter().map(|x| x.into_inner()).collect::<Vec<f32>>(),
      )
    };
    for use_rrnp in [false, true] {
      for use_lgd in [false, true] {
        let s = RandomState::new();
        let config = KnnGraphConfig {
          capacity: 50,
          out_degree: 5,
          num_searchers: 5,
          build_hasher: s,
          use_rrnp,
          rrnp_max_depth: 2,
          use_lgd,
        };

        let mut prng = Xoshiro256StarStar::seed_from_u64(1);

        let mut g = Knn::new(config, dist_fn);
        g.insert(
          vec![
            NotNan::new(1f32).unwrap(),
            NotNan::new(2f32).unwrap(),
            NotNan::new(3f32).unwrap(),
            NotNan::new(4f32).unwrap(),
          ],
          &mut prng,
        )
        .unwrap();
        g.insert(
          vec![
            NotNan::new(2f32).unwrap(),
            NotNan::new(4f32).unwrap(),
            NotNan::new(5f32).unwrap(),
            NotNan::new(6f32).unwrap(),
          ],
          &mut prng,
        )
        .unwrap();
        g.insert(
          vec![
            NotNan::new(3f32).unwrap(),
            NotNan::new(4f32).unwrap(),
            NotNan::new(5f32).unwrap(),
            NotNan::new(12f32).unwrap(),
          ],
          &mut prng,
        )
        .unwrap();
        g.insert(
          vec![
            NotNan::new(23f32).unwrap(),
            NotNan::new(14f32).unwrap(),
            NotNan::new(45f32).unwrap(),
            NotNan::new(142f32).unwrap(),
          ],
          &mut prng,
        )
        .unwrap();
        g.insert(
          vec![
            NotNan::new(37f32).unwrap(),
            NotNan::new(45f32).unwrap(),
            NotNan::new(53f32).unwrap(),
            NotNan::new(122f32).unwrap(),
          ],
          &mut prng,
        )
        .unwrap();
        g.insert(
          vec![
            NotNan::new(13f32).unwrap(),
            NotNan::new(14f32).unwrap(),
            NotNan::new(555f32).unwrap(),
            NotNan::new(125f32).unwrap(),
          ],
          &mut prng,
        )
        .unwrap();
        g.insert(
          vec![
            NotNan::new(13f32).unwrap(),
            NotNan::new(4f32).unwrap(),
            NotNan::new(53f32).unwrap(),
            NotNan::new(12f32).unwrap(),
          ],
          &mut prng,
        )
        .unwrap();
        g.insert(
          vec![
            NotNan::new(33f32).unwrap(),
            NotNan::new(4f32).unwrap(),
            NotNan::new(53f32).unwrap(),
            NotNan::new(312f32).unwrap(),
          ],
          &mut prng,
        )
        .unwrap();

        let SearchResults {
          approximate_nearest_neighbors: nearest_neighbors,
          ..
        } = g
          .query(
            &vec![
              NotNan::new(34f32).unwrap(),
              NotNan::new(5f32).unwrap(),
              NotNan::new(53f32).unwrap(),
              NotNan::new(312f32).unwrap(),
            ],
            1,
            &mut prng,
          )
          .unwrap();
        assert_eq!(
          nearest_neighbors
            .iter()
            .map(|x| (x.item.clone(), x.dist))
            .collect::<Vec<(Vec<NotNan<f32>>, f32)>>()[0]
            .0,
          vec![
            NotNan::new(33f32).unwrap(),
            NotNan::new(4f32).unwrap(),
            NotNan::new(53f32).unwrap(),
            NotNan::new(312f32).unwrap(),
          ]
        );
      }
    }
  }

  #[test]
  fn test_search_results_merge() {
    let sr1: SearchResults<u32> = SearchResults {
      approximate_nearest_neighbors: vec![
        SearchResult {
          item: 1,
          internal_id: Some(1),
          dist: 1.0,
          search_root_ancestor: 0,
          search_depth: 0,
        },
        SearchResult {
          item: 2,
          internal_id: Some(2),
          dist: 2.0,
          search_root_ancestor: 0,
          search_depth: 0,
        },
      ],
      visited_nodes: vec![
        SearchResult {
          item: 1,
          internal_id: Some(1),
          dist: 1.0,
          search_root_ancestor: 0,
          search_depth: 0,
        },
        SearchResult {
          item: 2,
          internal_id: Some(2),
          dist: 2.0,
          search_root_ancestor: 0,
          search_depth: 0,
        },
        SearchResult {
          item: 3,
          internal_id: Some(3),
          dist: 3.0,
          search_root_ancestor: 0,
          search_depth: 0,
        },
      ],
      visited_nodes_distances_to_q: HashMap::from([
        (1, (1, 1.0)),
        (2, (2, 2.0)),
        (3, (3, 3.0)),
      ]),
      search_stats: None,
    };
    let sr2: SearchResults<u32> = SearchResults {
      approximate_nearest_neighbors: vec![
        SearchResult {
          item: 3,
          internal_id: Some(3),
          dist: 3.0,
          search_root_ancestor: 0,
          search_depth: 0,
        },
        SearchResult {
          item: 4,
          internal_id: Some(4),
          dist: 4.0,
          search_root_ancestor: 0,
          search_depth: 0,
        },
      ],
      visited_nodes: vec![
        SearchResult {
          item: 1,
          internal_id: Some(1),
          dist: 1.0,
          search_root_ancestor: 0,
          search_depth: 0,
        },
        SearchResult {
          item: 3,
          internal_id: Some(3),
          dist: 3.0,
          search_root_ancestor: 0,
          search_depth: 0,
        },
        SearchResult {
          item: 4,
          internal_id: Some(4),
          dist: 4.0,
          search_root_ancestor: 0,
          search_depth: 0,
        },
      ],
      visited_nodes_distances_to_q: HashMap::from([
        (1, (1, 1.0)),
        (3, (3, 3.0)),
        (4, (4, 4.0)),
      ]),
      search_stats: None,
    };

    let sr3 = sr1.merge(&sr2);
    assert_eq!(
      sr3.approximate_nearest_neighbors,
      vec![
        SearchResult {
          item: 1,
          internal_id: Some(1),
          dist: 1.0,
          search_root_ancestor: 0,
          search_depth: 0
        },
        SearchResult {
          item: 2,
          internal_id: Some(2),
          dist: 2.0,
          search_root_ancestor: 0,
          search_depth: 0
        },
        SearchResult {
          item: 3,
          internal_id: Some(3),
          dist: 3.0,
          search_root_ancestor: 0,
          search_depth: 0
        },
        SearchResult {
          item: 4,
          internal_id: Some(4),
          dist: 4.0,
          search_root_ancestor: 0,
          search_depth: 0
        },
      ]
    );
    assert_eq!(
      sr3.visited_nodes,
      vec![
        SearchResult {
          item: 1,
          internal_id: Some(1),
          dist: 1.0,
          search_root_ancestor: 0,
          search_depth: 0
        },
        SearchResult {
          item: 2,
          internal_id: Some(2),
          dist: 2.0,
          search_root_ancestor: 0,
          search_depth: 0
        },
        SearchResult {
          item: 3,
          internal_id: Some(3),
          dist: 3.0,
          search_root_ancestor: 0,
          search_depth: 0
        },
        SearchResult {
          item: 4,
          internal_id: Some(4),
          dist: 4.0,
          search_root_ancestor: 0,
          search_depth: 0
        },
      ],
    );
    assert_eq!(
      sr3.visited_nodes_distances_to_q,
      HashMap::from([
        (1, (1, 1.0)),
        (2, (2, 2.0)),
        (3, (3, 3.0)),
        (4, (4, 4.0)),
      ]),
    );
  }
}
