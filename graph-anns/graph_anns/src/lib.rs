#![feature(is_sorted)]
extern crate indexmap;
extern crate nohash_hasher;
extern crate rand;
extern crate rand_xoshiro;
extern crate tinyset;

use indexmap::map::IndexMap;
use nohash_hasher::BuildNoHashHasher;
use rand::seq::index::sample;
use rand::thread_rng;
use rand::RngCore;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::binary_heap::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::hash::BuildHasher;
use tinyset::SetU32;

const DEFAULT_LAMBDA: u8 = 0;

// TODO: revisit every use of `pub` in this file.

// TODO: many benchmarking utilities. Recall@n  metrics and so forth.

// TODO: I see two possible problems in the paper:
// 1. Nothing in the paper guarantees the connectivity of the graph.
// 2. Intuitively, it seems like being able to jump long distances if we are in
//    a really irrelevant part of the graph would be a good idea.
//
// To fix both of these problems, I have an idea. Each node has one extra edge
// that is used normally for searching, but can't be removed by the insertion
// algorithm. These edges, taken together, form a permutation of the nodes of
// the graph, ensuring that we have a route passing through all the nodes. We
// try to optimize this permutation by swapping, in order to decrease the sum
// of the distances between successive nodes. Also, the node linked to in this
// special edge cannot be a node that is already linked to by the normal edges.
// We could use idle background CPU to optimize this permutation while we wait
// for requests.
//
// We'd need to make this optional and benchmark carefully to verify that it
// improves performance -- the permutation and optimization of the permutation
// should be two separate flags that are evaluated separately. Maybe a random
// permutation is better, because it's essentially giving each searcher the
// option to do something like a random restart at each iteration, and random
// restarts are a well-known trick for improving local search algorithms such
// as this one.
//
// test_delete below actually hits this edge case -- the graph is split into two
// connected components, and so a search that starts in one component can
// never reach the other. We are doing beam search from multiple starting points
// and that should mitigate the issue, but it doesn't guarantee anything.
// Ideally, we would ensure that the graph is always connected.
//
// One way to do this would be to maintain one special edge per node that points
// to a nearby-ish neighbor with an additional property: all of these special
// edges together form a path that touches each node in the graph once. In other
// words, we maintain a permutation of the nodes to ensure that we always have
// a path through all of the nodes.
//
// Problems with this idea:
// 1. We would need to optimize the path to ensure that consecutive nodes are
//   close to one another.
// 2. Some consecutive nodes simply won't be very close no matter how we try.
// 3. It may have unintended cosequences on the search performance.
// 4. Internal implementation becomes even more complicated.
//
// What about other solutions? We could simply trust the beam search, and
// recommend using a higher number of searchers. However, I suspect that this
// permutation idea may even help search performance -- see above note about
// random restarts.

#[derive(Clone, Copy)]
pub struct KNNGraphConfig<'a, T, S: BuildHasher + Clone> {
  /// The max number of vertices that can be inserted into the graph. Constant.
  pub capacity: u32,
  /// The number of approximate nearest neighbors to store for each inserted
  /// element. This is a constant. Each graph node is guaranteed to always have
  /// exactly this out-degree. It's recommended to set this equal to the
  /// intrinsic dimensionality of the data. Lower values hinder search performance
  /// by preventing the search from moving in useful directions, higher values
  /// hinder performance by incurring needless distance calls.
  pub out_degree: u8,
  /// Number of simultaneous searchers in the beam search.
  pub num_searchers: u32,
  /// distance function. Must satisfy the criteria of a metric:
  /// https://en.wikipedia.org/wiki/Metric_(mathematics). Several internal
  /// optimizations assume the triangle inequality holds.
  pub dist_fn: &'a (dyn Fn(&T, &T) -> f32 + Sync),
  pub build_hasher: S,
  /// Whether to use restricted recursive neighborhood propagation. This improves
  /// search speed by about 10%, but decreases insertion throughput by about 10%.
  pub use_rrnp: bool,
  /// Maximum recursion depth for RRNP. 2 is a good default.
  pub rrnp_max_depth: u32,
  /// Whether to use lazy graph diversification. This improves search speed.
  /// TODO: parametrize the graph type so that the LGD vector is never allocated/
  /// takes no memory if this is set to false.
  pub use_lgd: bool,
}

// NOTE: can't do a Default because we can't guess a reasonable capacity.
// TODO: auto-resize like a Vec?

impl<'a, T, S: BuildHasher + Clone> KNNGraphConfig<'a, T, S> {
  /// Create a new KNNGraphConfig.
  pub fn new(
    capacity: u32,
    out_degree: u8,
    num_searchers: u32,
    dist_fn: &'a (dyn Fn(&T, &T) -> f32 + Sync),
    build_hasher: S,
    use_rrnp: bool,
    rrnp_max_depth: u32,
    use_lgd: bool,
  ) -> KNNGraphConfig<'a, T, S> {
    KNNGraphConfig::<'a, T, S> {
      capacity,
      out_degree,
      num_searchers,
      dist_fn,
      build_hasher,
      use_rrnp,
      rrnp_max_depth,
      use_lgd,
    }
  }
}

// TODO: test that query can find already-inserted items

pub trait NN<T> {
  // TODO: return types with error sums, more informative delete (did it exist?)
  // etc. Eliminate all panics that are not internal errors.

  // TODO: more functions in this interface.

  fn insert<R: RngCore>(&mut self, x: T, prng: &mut R) -> ();
  fn delete(&mut self, x: T) -> ();
  fn query<R: RngCore>(
    &self,
    q: &T,
    max_results: usize,
    prng: &mut R,
  ) -> SearchResults<T>;
}

pub struct BruteForceKNN<'a, T> {
  pub contents: HashSet<T>,
  // TODO: made this Sync so that I could share a single closure across threads
  // when splitting up a graph into multiple pieces. Is this going to be onerous
  // to users?
  pub distance: &'a (dyn Fn(&T, &T) -> f32 + Sync),
}

impl<'a, T> BruteForceKNN<'a, T> {
  pub fn new(
    distance: &'a (dyn Fn(&T, &T) -> f32 + Sync),
  ) -> BruteForceKNN<'a, T> {
    BruteForceKNN {
      contents: HashSet::new(),
      distance,
    }
  }

  pub fn debug_size_stats(&self) -> SpaceReport {
    SpaceReport::default()
  }
}

impl<'a, T: Copy + Ord + Eq + std::hash::Hash> NN<T> for BruteForceKNN<'a, T> {
  fn insert<R: RngCore>(&mut self, x: T, _prng: &mut R) -> () {
    self.contents.insert(x);
  }

  fn delete(&mut self, x: T) -> () {
    self.contents.remove(&x);
  }

  fn query<R: RngCore>(
    &self,
    q: &T,
    max_results: usize,
    _prng: &mut R,
  ) -> SearchResults<T> {
    let mut nearest_neighbors_max_dist_heap: BinaryHeap<SearchResult<T>> =
      BinaryHeap::new();
    let mut visited_nodes: HashSet<T> = HashSet::new();
    let mut visited_node_distances_to_q: HashMap<T, f32> = HashMap::default();
    for x in self.contents.iter() {
      let dist = (self.distance)(x, &q);
      visited_nodes.insert(*x);
      nearest_neighbors_max_dist_heap.push(SearchResult::new(*x, dist));
      if nearest_neighbors_max_dist_heap.len() >= max_results {
        nearest_neighbors_max_dist_heap.pop();
      }
      visited_node_distances_to_q.insert(*x, dist);
    }
    SearchResults {
      approximate_nearest_neighbors: nearest_neighbors_max_dist_heap
        .into_sorted_vec(),
      visited_nodes,
      visited_node_distances_to_q,
    }
  }
}

/// Maps from the user's chosen ID type to our internal u32 ids that are used
/// within the core search functions to keep things fast and compact.
/// Ideally, we should translate to and from user ids at the edges of
/// performance-critical code. In practice, doing so may be difficult, since the
/// user's distance callback is passed external ids (the user's IDs).
#[derive(Debug)]
pub struct InternalExternalIDMapping<T, S: BuildHasher> {
  pub capacity: u32,
  // TODO: using a vec might be faster.
  pub internal_to_external_ids: IndexMap<u32, T, BuildNoHashHasher<u32>>,
  pub external_to_internal_ids: HashMap<T, u32, S>,
  pub deleted: Vec<u32>,
}

impl<T: Clone + Eq + std::hash::Hash, S: BuildHasher>
  InternalExternalIDMapping<T, S>
{
  fn with_capacity_and_hasher(capacity: u32, hash_builder: S) -> Self {
    let internal_to_external_ids =
      IndexMap::<u32, T, BuildNoHashHasher<u32>>::with_capacity_and_hasher(
        capacity as usize,
        nohash_hasher::BuildNoHashHasher::default(),
      );
    let external_to_internal_ids =
      HashMap::with_capacity_and_hasher(capacity as usize, hash_builder);

    let deleted = Vec::<u32>::new();
    InternalExternalIDMapping {
      capacity,
      internal_to_external_ids,
      external_to_internal_ids,
      deleted,
    }
  }

  fn insert(self: &mut Self, x: &T) -> u32 {
    match self.external_to_internal_ids.get(&x) {
      Some(id) => {
        return (*id).clone();
      }
      None => {
        let x_int = match self.deleted.pop() {
          None => self.internal_to_external_ids.len() as u32,
          Some(i) => i,
        };
        if x_int > self.capacity {
          panic!("exceeded capacity TODO: bubble up error");
        }

        // TODO: we are storing two clones of x. Replace with a bidirectional
        // map or something to reduce memory usage.
        self.internal_to_external_ids.insert(x_int, x.clone());
        self.external_to_internal_ids.insert(x.clone(), x_int);
        return x_int;
      }
    }
  }

  fn int_to_ext(self: &Self, x: u32) -> &T {
    match self.internal_to_external_ids.get(&x) {
      None => panic!("internal error: unknown internal id: {}", x),
      Some(i) => i,
    }
  }

  fn ext_to_int(self: &Self, x: &T) -> &u32 {
    match self.external_to_internal_ids.get(x) {
      None => panic!("external error: unknown external id"),
      Some(i) => i,
    }
  }

  fn delete(self: &mut Self, x: &T) -> u32 {
    let x_int = (*self.ext_to_int(x)).clone();
    self.deleted.push(x_int);
    self.internal_to_external_ids.remove(&x_int);
    self.external_to_internal_ids.remove(&x);
    return x_int;
  }
}

fn convert_bruteforce_to_dense<
  'a,
  T: Copy + Eq + std::hash::Hash,
  S: BuildHasher + Clone,
>(
  bf: &mut BruteForceKNN<'a, T>,
  config: KNNGraphConfig<'a, T, S>,
) -> DenseKNNGraph<'a, T, S> {
  let ids = bf.contents.iter().collect();
  exhaustive_knn_graph(ids, config)
}

fn convert_dense_to_bruteforce<
  'a,
  T: Copy + Eq + std::hash::Hash,
  S: BuildHasher + Clone,
>(
  g: &mut DenseKNNGraph<'a, T, S>,
) -> BruteForceKNN<'a, T> {
  let mut contents = HashSet::new();
  let distance = g.config.dist_fn;
  for (ext_id, _) in g.mapping.external_to_internal_ids.drain() {
    contents.insert(ext_id);
  }
  BruteForceKNN { contents, distance }
}

/// Switches from brute-force to approximate nearest neighbor search based on
/// the number of inserted elements.
pub enum KNN<'a, T, S: BuildHasher + Clone> {
  Small {
    g: BruteForceKNN<'a, T>,
    config: KNNGraphConfig<'a, T, S>,
  },

  Large(DenseKNNGraph<'a, T, S>),
}

impl<'a, T: Copy + Ord + Eq + std::hash::Hash, S: BuildHasher + Clone> NN<T>
  for KNN<'a, T, S>
{
  fn insert<R: RngCore>(&mut self, x: T, prng: &mut R) -> () {
    match self {
      KNN::Small { g, config } => {
        if config.capacity as usize == g.contents.len() {
          panic!("TODO create error type etc.");
        } else if g.contents.len() == 20 {
          *self = KNN::Large(convert_bruteforce_to_dense(g, config.clone()));
          self.insert(x, prng);
        } else {
          g.insert(x, prng);
        }
      }
      KNN::Large(g) => g.insert(x, prng),
    }
  }

  fn delete(&mut self, x: T) -> () {
    match self {
      KNN::Small { g, .. } => {
        g.delete(x);
      }
      KNN::Large(g) => {
        if g.mapping.external_to_internal_ids.len()
          == g.config.out_degree as usize + 1
        {
          let config = g.config.clone();
          let mut small_g = convert_dense_to_bruteforce(g);
          small_g.delete(x);
          *self = KNN::Small {
            g: small_g,
            config: config,
          };
        }
      }
    }
  }

  fn query<R: RngCore>(
    &self,
    q: &T,
    max_results: usize,
    prng: &mut R,
  ) -> SearchResults<T> {
    match self {
      KNN::Small { g, .. } => g.query(&q, max_results, prng),
      KNN::Large(g) => g.query(&q, max_results, prng),
    }
  }
}

impl<'a, T: Copy + Ord + Eq + std::hash::Hash, S: BuildHasher + Clone>
  KNN<'a, T, S>
{
  pub fn new(config: KNNGraphConfig<'a, T, S>) -> KNN<'a, T, S> {
    KNN::Small {
      g: BruteForceKNN::new(config.dist_fn),
      config,
    }
  }

  /// Panics if graph structure is internally inconsistent. Used for testing.
  pub fn debug_consistency_check(&self) -> () {
    match self {
      KNN::Small { .. } => {
        return;
      }
      KNN::Large(g) => {
        g.consistency_check();
      }
    }
  }

  /// Returns information about the length and capacity of all data structures
  /// in the graph.
  pub fn debug_size_stats(&self) -> SpaceReport {
    match self {
      KNN::Small { g, .. } => g.debug_size_stats(),
      KNN::Large(g) => g.debug_size_stats(),
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpaceReport {
  pub mapping_int_ext_len: usize,
  pub mapping_int_ext_capacity: usize,
  pub mapping_ext_int_len: usize,
  pub mapping_ext_int_capacity: usize,
  pub mapping_deleted_len: usize,
  pub mapping_deleted_capacity: usize,
  pub edges_vec_len: usize,
  pub edges_vec_capacity: usize,
  pub backpointers_len: usize,
  pub backpointers_capacity: usize,
  pub backpointers_sets_sum_len: usize,
  pub backpointers_sets_sum_capacity: usize,
  backpointers_sets_mem_used: usize,
}

impl SpaceReport {
  pub fn merge(&self, other: &Self) -> SpaceReport {
    SpaceReport {
      mapping_int_ext_len: self.mapping_int_ext_len + other.mapping_int_ext_len,
      mapping_int_ext_capacity: self.mapping_int_ext_capacity
        + other.mapping_int_ext_capacity,
      mapping_ext_int_len: self.mapping_ext_int_len + other.mapping_ext_int_len,
      mapping_ext_int_capacity: self.mapping_ext_int_capacity
        + other.mapping_ext_int_capacity,
      mapping_deleted_len: self.mapping_deleted_len + other.mapping_deleted_len,
      mapping_deleted_capacity: self.mapping_deleted_capacity
        + other.mapping_deleted_capacity,
      edges_vec_len: self.edges_vec_len + other.edges_vec_len,
      edges_vec_capacity: self.edges_vec_capacity + other.edges_vec_capacity,
      backpointers_len: self.backpointers_len + other.backpointers_len,
      backpointers_capacity: self.backpointers_capacity
        + other.backpointers_capacity,
      backpointers_sets_sum_len: self.backpointers_sets_sum_len
        + other.backpointers_sets_sum_len,
      backpointers_sets_sum_capacity: self.backpointers_sets_sum_capacity
        + other.backpointers_sets_sum_capacity,
      backpointers_sets_mem_used: self.backpointers_sets_mem_used
        + other.backpointers_sets_mem_used,
    }
  }
}

impl Default for SpaceReport {
  fn default() -> Self {
    SpaceReport {
      mapping_int_ext_len: 0,
      mapping_int_ext_capacity: 0,
      mapping_ext_int_len: 0,
      mapping_ext_int_capacity: 0,
      mapping_deleted_len: 0,
      mapping_deleted_capacity: 0,
      edges_vec_len: 0,
      edges_vec_capacity: 0,
      backpointers_len: 0,
      backpointers_capacity: 0,
      backpointers_sets_sum_len: 0,
      backpointers_sets_sum_capacity: 0,
      backpointers_sets_mem_used: 0,
    }
  }
}

/// A directed graph stored contiguously in memory as an adjacency list.
/// All vertices are guaranteed to have the same out-degree.
/// Nodes are u32s numbered from 0 to n-1.
pub struct DenseKNNGraph<'a, T, S: BuildHasher + Clone> {
  /// A mapping from the user's ID type, T, to our internal ids, which are u32.
  /// TODO: now that we have to store a hashmap anyway, are we gaining anything
  /// by storing edges as a vector? We could potentially simplify by eliminating
  /// it and the external/internal id distinction. I think all we gain is a
  /// more memory-efficient adjacency list... which may be a decisive factor.
  pub mapping: InternalExternalIDMapping<T, S>,
  /// n, the current number of vertices in the graph. The valid indices of the
  /// graph are given by self.mapping.internal_to_external_ids.keys().
  pub num_vertices: u32,
  /// The underlying buffer of capacity*out_degree neighbor information.
  /// An adjacency list of (node id, distance, lambda crowding factor).
  /// Use with caution.
  /// Prefer to use get_edges to access the neighbors of a vertex.
  pub edges: Vec<(u32, f32, u8)>,
  /// Maintains an association between vertices and the vertices that link out to
  /// them. In other words, each backpointers[i] is the set of vertices S s.t.
  /// for all x in S, a directed edge exists pointing from x to i.
  pub backpointers: Vec<SetU32>,
  pub config: KNNGraphConfig<'a, T, S>,
}

impl<'a, T: Clone + Eq + std::hash::Hash, S: BuildHasher + Clone>
  DenseKNNGraph<'a, T, S>
{
  /// Returns information about the length and capacity of all data structures
  /// in the graph.
  pub fn debug_size_stats(&self) -> SpaceReport {
    SpaceReport {
      mapping_int_ext_len: self.mapping.internal_to_external_ids.len(),
      mapping_int_ext_capacity: self
        .mapping
        .internal_to_external_ids
        .capacity(),
      mapping_ext_int_len: self.mapping.external_to_internal_ids.len(),
      mapping_ext_int_capacity: self
        .mapping
        .external_to_internal_ids
        .capacity(),
      mapping_deleted_len: self.mapping.deleted.len(),
      mapping_deleted_capacity: self.mapping.deleted.capacity(),
      edges_vec_len: self.edges.len(),
      edges_vec_capacity: self.edges.capacity(),
      backpointers_len: self.backpointers.len(),
      backpointers_capacity: self.backpointers.capacity(),
      backpointers_sets_sum_len: self
        .backpointers
        .iter()
        .map(|s| s.len())
        .sum(),
      backpointers_sets_sum_capacity: self
        .backpointers
        .iter()
        .map(|s| s.capacity())
        .sum(),
      backpointers_sets_mem_used: self
        .backpointers
        .iter()
        .map(|s| s.mem_used())
        .sum(),
    }
  }

  /// Allocates a graph of the specified size and out_degree, but
  /// doesn't populate the edges.
  // TODO: no pub, not usable
  pub fn empty(config: KNNGraphConfig<'a, T, S>) -> DenseKNNGraph<'a, T, S> {
    let mapping = InternalExternalIDMapping::<T, S>::with_capacity_and_hasher(
      config.capacity,
      config.build_hasher.clone(),
    );

    // TODO: splitting this into three vecs would allow at least the first one
    // to specialize to calloc and potentially be a lot faster.
    let edges = vec![
      (0, 0.0, DEFAULT_LAMBDA);
      config.capacity as usize * config.out_degree as usize
    ];

    let mut backpointers = Vec::with_capacity(config.capacity as usize);

    for _ in 0..config.capacity {
      backpointers.push(SetU32::new());
    }

    let num_vertices = 0;

    DenseKNNGraph {
      mapping,
      num_vertices,
      edges,
      backpointers,
      config: config.clone(),
    }
  }

  /// Get the neighbors of u and their distances. Panics if index
  /// >= capacity or does not exist.
  pub fn get_edges(&self, index: u32) -> &[(u32, f32, u8)] {
    assert!(
      index < self.config.capacity,
      "index {} out of bounds",
      index
    );
    assert!(
      self.mapping.internal_to_external_ids.contains_key(&index),
      "index {} does not exist in internal ids",
      index
    );

    let i = index * self.config.out_degree as u32;
    let j = i + self.config.out_degree as u32;
    &self.edges[i as usize..j as usize]
  }

  fn debug_get_neighbor_indices(&self, index: u32) -> Vec<u32> {
    self.get_edges(index).iter().map(|e| e.0).collect()
  }

  /// Get the neighbors of u and their distances. Panics if index
  /// >= capacity or does not exist.
  fn get_edges_mut(&mut self, index: u32) -> &mut [(u32, f32, u8)] {
    assert!(index < self.config.capacity);
    assert!(self.mapping.internal_to_external_ids.contains_key(&index));

    let i = index * self.config.out_degree as u32;
    let j = i + self.config.out_degree as u32;
    &mut self.edges[i as usize..j as usize]
  }

  fn sort_edges(&mut self, index: u32) {
    let edges = self.get_edges_mut(index);
    edges.sort_by(|a, b| a.1.total_cmp(&b.1));
  }

  /// Get distance in terms of two internal ids. Panics if internal ids do not
  /// exist in the mapping.
  fn dist_int(&self, int_id1: u32, int_id2: u32) -> f32 {
    let ext_id1 = self.mapping.internal_to_external_ids.get(&int_id1).unwrap();
    let ext_id2 = self.mapping.internal_to_external_ids.get(&int_id2).unwrap();
    (self.config.dist_fn)(ext_id1, ext_id2)
  }

  /// Replace the edge (u,v) with a new edge (u,w)
  /// DOES NOT update v or w's backpointers
  fn replace_edge_no_backptr_update(
    &mut self,
    u: u32,
    v: u32,
    w: u32,
    u_w_dist: f32,
  ) {
    let u_edges = self.get_edges_mut(u);
    for (edge_ix, edge_dist, edge_lambda) in u_edges.iter_mut() {
      if *edge_ix == v {
        *edge_ix = w;
        *edge_dist = u_w_dist;
        *edge_lambda = DEFAULT_LAMBDA;
      }
    }

    // TODO: the list is all sorted except for the new element; we could optimize
    // this.
    self.sort_edges(u);
  }

  /// Creates an edge from `from` to `to` if the distance `dist` between them is
  /// less than the distance from `from` to one of its existing neighbors `u`.
  /// If so, removes the edge (`from`, `u`).
  ///
  /// `to` must have already been added to the graph by insert_vertex etc.
  ///
  /// Returns `true` if the new edge was added.
  fn insert_edge_if_closer(&mut self, from: u32, to: u32, dist: f32) -> bool {
    let most_distant_ix = (self.config.out_degree - 1) as usize;
    let edges = self.get_edges_mut(from);

    if dist < edges[most_distant_ix].1 {
      let old = edges[most_distant_ix].0;
      edges[most_distant_ix].0 = to;
      edges[most_distant_ix].1 = dist;
      edges[most_distant_ix].2 = DEFAULT_LAMBDA;
      self.sort_edges(from);
      self.backpointers[old as usize].remove(from);
      self.backpointers[to as usize].insert(from);
      return true;
    }

    return false;
  }

  // TODO: can we run k-nn descent on the graph whenever we have free cycles?

  /// Insert a new vertex into the graph, with the given k neighbors and their
  /// distances. Does not connect other vertices to this vertex.
  /// Panics if the graph is already full (num_vertices == capacity).
  /// nbrs and dists must be equal to the out_degree of the graph.
  fn insert_vertex(&mut self, u: u32, nbrs: Vec<u32>, dists: Vec<f32>) {
    //TODO: replace all asserts with Either return values
    assert!(self.num_vertices < self.config.capacity);

    let od = self.config.out_degree as usize;
    assert!(
      nbrs.len() == od && dists.len() == od,
      "nbrs.len() {}, out_degree {} dists.len() {}",
      nbrs.len(),
      od,
      dists.len()
    );

    for ((edge_ix, nbr), dist) in nbrs.iter().enumerate().zip(dists) {
      self.edges[u as usize * od + edge_ix] = (*nbr, dist, DEFAULT_LAMBDA);
      let s = &mut self.backpointers[*nbr as usize];
      s.insert(u);
    }

    self.num_vertices += 1;
    self.sort_edges(u);
  }

  pub fn debug_print(&self) {
    println!("### Adjacency list (index, distance, lambda)");
    for i in self.mapping.internal_to_external_ids.keys() {
      println!("Node {}", i);
      println!("{:#?}", self.get_edges(*i));
    }
    println!("### Backpointers");
    for i in self.mapping.internal_to_external_ids.keys() {
      println!("Node {} {:#?}", i, self.backpointers[*i as usize]);
    }
  }

  /// Panics if graph is internally inconsistent.
  fn consistency_check(&self) {
    if self.edges.len()
      != self.config.capacity as usize * self.config.out_degree as usize
    {
      panic!(
        "edges.len() is not equal to capacity * out_degree. {} != {} * {}",
        self.edges.len(),
        self.config.capacity,
        self.config.out_degree
      );
    }

    for i in self.mapping.internal_to_external_ids.keys() {
      for (nbr, _, _) in self.get_edges(*i).iter() {
        if *nbr == *i {
          panic!("Self loop at node {}", i);
        }
        let nbr_backptrs = &self.backpointers[*nbr as usize];
        if !nbr_backptrs.contains(*i) {
          panic!(
            "Vertex {} links to {} but {}'s backpointers don't include {}!",
            i, nbr, nbr, i
          );
        }
      }

      assert!(self.get_edges(*i).is_sorted_by_key(|e| e.1));

      for referrer in self.backpointers[*i as usize].iter() {
        assert!(self.debug_get_neighbor_indices(referrer).contains(&i));
      }
    }
  }

  fn exists_edge(&self, u: u32, v: u32) -> bool {
    for (w, _, _) in self.get_edges(u).iter() {
      if v == *w {
        return true;
      }
    }
    return false;
  }

  // TODO: expose an interface where the user provides one of their ids already
  // in the database and we
  // return n approximate nearest neighbors very fast by using upper bounds on
  // distance in the same way as two_hop_neighbors_and_dist_upper_bounds. Of
  // course, for best results we should be unsharded.

  // TODO: does our new API even support out-of-db queries effectively? The
  // distance function the user provides takes two ids as input. Make an example
  // of inserting a few vectors and then querying for neigbors to a novel vector
  // that doesn't exist in the database.

  // TODO: k-nn descent helper so we can optimize the graph with idle CPU.

  /// Returns up to limit neighbors that are at 2 hops away from int_id, and an
  /// upper bound on the distance of each (computed as dist(u,v) + dist(v,w)).
  /// Computing this upper bound is very cheap, so it is returned directly.
  /// Up to k*k ids will be returned.
  /// If there exists a path of length 1 and a path of length 2 to the neighbor,
  /// it will *not* be returned -- the goal is to return neighbors of neighbors.
  fn two_hop_neighbors_and_dist_upper_bounds(
    &self,
    int_id: u32,
  ) -> BinaryHeap<Reverse<SearchResult<u32>>> {
    let mut ret: BinaryHeap<Reverse<SearchResult<u32>>> = BinaryHeap::new();
    for (v, v_dist, _) in self.get_edges(int_id).iter() {
      for (w, w_dist, _) in self.get_edges(*v).iter() {
        if !self.exists_edge(int_id, *w) {
          ret.push(Reverse(SearchResult {
            item: *w,
            dist: v_dist + w_dist,
          }));
        }
      }
    }
    return ret;
  }

  fn get_farthest_neighbor(
    self: &DenseKNNGraph<'a, T, S>,
    int_id: u32,
  ) -> (u32, f32) {
    let last_ix = self.config.out_degree - 1;
    let (f, dist, _) = self.get_edges(int_id)[last_ix as usize];
    (f, dist).clone()
  }

  fn in_and_out_neighbors(
    self: &DenseKNNGraph<'a, T, S>,
    int_id: u32,
  ) -> Vec<(u32, f32)> {
    let mut ret = Vec::new();
    for (w, dist, _) in self.get_edges(int_id).iter() {
      ret.push((*w, dist.clone()));
    }
    for w in self.backpointers[int_id as usize].iter() {
      let dist = self
        .get_edges(w)
        .iter()
        .filter(|e| e.0 == int_id)
        .next()
        .unwrap()
        .1;
      ret.push((w, dist.clone()));
    }
    ret
  }

  /// Perform RRNP w.r.t. a newly-inserted element q, given the results of
  /// searching for q in the graph. This implements lines 10 to 24 of algorithm 3
  /// in the paper. A small difference is that I pulled the loop `while |W| > 0`
  /// outside of the `for each r in V` loop. That should have no effect on the
  /// behavior, but it makes things more readable.
  fn rrnp(
    self: &mut DenseKNNGraph<'a, T, S>,
    q: T,
    q_int: u32,
    visited_nodes: &HashSet<T>,
    visited_nodes_distance_to_q: &HashMap<T, f32>,
  ) -> () {
    let mut w = VecDeque::new();
    for r in visited_nodes {
      w.push_back((
        self.mapping.ext_to_int(r).clone(),
        0,
        visited_nodes_distance_to_q.get(r).unwrap(),
      ));
    }

    let mut already_rrnped = HashSet::new();

    while w.len() > 0 {
      let (s_int, depth, dist_s_q) = w.pop_front().unwrap();
      let (_, dist_s_f) = self.get_farthest_neighbor(s_int);
      if depth < self.config.rrnp_max_depth && dist_s_q < &dist_s_f {
        for (e, _) in self.in_and_out_neighbors(s_int) {
          let e_ext = self.mapping.int_to_ext(e);
          if !already_rrnped.contains(&e) && !visited_nodes.contains(&e_ext) {
            already_rrnped.insert(e);
            let dist_e_q = (self.config.dist_fn)(&e_ext, &q);
            self.insert_edge_if_closer(e, q_int, dist_e_q);
          }
        }
      }
    }
  }

  /// Return the average occlusion (lambda values) of the neighbors of the given
  /// node. This is part of the lazy graph diversification algorithm. See the
  /// paper for details. Returns infinity if LGD is disabled.
  fn average_occlusion(&self, int_id: u32) -> f32 {
    if !self.config.use_lgd {
      return f32::INFINITY;
    }
    let mut sum = 0.0;
    for (_, _, lambda) in self.get_edges(int_id).iter() {
      sum += *lambda as f32;
    }
    sum / self.config.out_degree as f32
  }

  fn query_internal<R: RngCore>(
    &self,
    q: &T,
    max_results: usize,
    prng: &mut R,
    ignore_occluded: bool,
  ) -> SearchResults<T> {
    assert!(self.config.num_searchers <= self.num_vertices);
    let mut q_max_heap: BinaryHeap<SearchResult<T>> = BinaryHeap::new();
    let mut r_min_heap: BinaryHeap<Reverse<SearchResult<T>>> =
      BinaryHeap::new();
    let mut visited = HashSet::<T>::new();
    let mut visited_distances: HashMap<T, f32> = HashMap::default();

    // Initialize our search with num_searchers initial points.
    // lines 2 to 10 of the pseudocode
    for r_index_into_hashmap in sample(
      prng,
      self.mapping.internal_to_external_ids.len(),
      self.config.num_searchers as usize,
    ) {
      let (_r, r_ext) = self
        .mapping
        .internal_to_external_ids
        .get_index(r_index_into_hashmap)
        .unwrap();
      let r_dist = (self.config.dist_fn)(&q, r_ext);
      visited.insert(r_ext.clone());
      visited_distances.insert(r_ext.clone(), r_dist);
      r_min_heap.push(Reverse(SearchResult::new(r_ext.clone(), r_dist)));
      match q_max_heap.peek() {
        None => {
          q_max_heap.push(SearchResult::new(r_ext.clone(), r_dist));
          // NOTE: pseudocode has a bug: R.insert(r) at both line 2 and line 8
          // We are skipping it here since we did it above.
        }
        Some(f) => {
          if r_dist < f.dist || q_max_heap.len() < max_results {
            q_max_heap.push(SearchResult::new(r_ext.clone(), r_dist));
          }
        }
      }
    }

    // The main search loop. While unvisited nodes exist in r_min_heap, keep
    // searching.
    // lines 11 to 27 of the pseudocode
    while r_min_heap.len() > 0 {
      while q_max_heap.len() > max_results {
        q_max_heap.pop();
      }

      let Reverse(sr) = r_min_heap.pop().unwrap();
      let f = { q_max_heap.peek().unwrap().clone() };
      let sr_int = self.mapping.ext_to_int(&sr.item);
      if sr.dist > f.dist {
        break;
      }

      let average_lambda = self.average_occlusion(*sr_int);
      let r_nbrs_iter = self.backpointers[*sr_int as usize].iter().chain(
        self
          .get_edges(*sr_int)
          .iter()
          .filter(|(_, _, lambda)| {
            !(ignore_occluded && *lambda as f32 >= average_lambda)
          })
          .map(|(nbr, _, _)| *nbr),
      );

      for e in r_nbrs_iter {
        let e_ext = self.mapping.int_to_ext(e);
        if !visited.contains(&e_ext) {
          visited.insert(e_ext.clone());
          let e_dist = (self.config.dist_fn)(&q, e_ext);
          visited_distances.insert(e_ext.clone(), e_dist);
          if e_dist < f.dist || q_max_heap.len() < max_results {
            q_max_heap.push(SearchResult::new(e_ext.clone(), e_dist));
            r_min_heap.push(Reverse(SearchResult::new(e_ext.clone(), e_dist)));
          }
        }
      }
    }

    SearchResults {
      approximate_nearest_neighbors: q_max_heap.into_sorted_vec(),
      visited_nodes: visited,
      visited_node_distances_to_q: visited_distances,
    }
  }
}

// NOTE: this abomination is necessary because I want helper functions that
// return parts of the graph data structure, but the borrow checker is dumb.
// This macro simply inlines get_edges_mut.
// TODO: find a better solution.
macro_rules! get_edges_mut_macro {
  ($self:ident, $index:ident) => {
    &mut $self.edges[($index as usize * $self.config.out_degree as usize)
      ..$index as usize * $self.config.out_degree as usize
        + $self.config.out_degree as usize]
  };
}

fn apply_lgd<'a, T: Clone + Eq + std::hash::Hash, S: BuildHasher + Clone>(
  r_edges: &mut [(u32, f32, u8)],
  mapping: &InternalExternalIDMapping<T, S>,
  q_int: u32,
  visited_node_distances_to_q: &HashMap<T, f32>,
) {
  let q_ix = r_edges.iter().position(|e| e.0 == q_int).unwrap();
  let q_r_dist = r_edges[q_ix].1;

  r_edges[q_ix].2 = DEFAULT_LAMBDA as u8;

  for s_ix in 0..r_edges.len() {
    let (s_int, s_r_dist, _) = r_edges[s_ix];
    let s_ext = mapping.int_to_ext(s_int);
    let s_q_dist = visited_node_distances_to_q.get(&s_ext);

    match s_q_dist {
      Some(s_q_dist) => {
        // rule 1 from the paper
        if s_r_dist < q_r_dist && s_q_dist >= &q_r_dist {
          continue;
        }
        // rule 2 from the paper
        else if s_r_dist < q_r_dist && s_q_dist < &q_r_dist {
          r_edges[q_ix].2 += 1;
        }
        // rule 3 from the paper: s_r_dist > q_r_dist, q occludes s
        else if s_q_dist < &s_r_dist {
          r_edges[s_ix].2 += 1;
        }
      }

      None => {
        continue;
      }
    }
  }
}

impl<'a, T: Clone + Eq + std::hash::Hash, S: BuildHasher + Clone> NN<T>
  for DenseKNNGraph<'a, T, S>
{
  /// Inserts a new data point into the graph. The graph must not be full.
  /// Optionally performs restricted recursive neighborhood propagation and
  /// lazy graph diversification. These options are set when the graph is
  /// constructed.
  ///
  /// This is equivalent to one iteration of Algorithm 3 in
  /// "Approximate k-NN Graph Construction: A Generic Online Approach".
  fn insert<R: RngCore>(&mut self, q: T, prng: &mut R) -> () {
    //TODO: return the index of the new data point
    let SearchResults {
      approximate_nearest_neighbors: nearest_neighbors,
      visited_nodes,
      visited_node_distances_to_q,
    } = self.query_internal(&q, self.config.out_degree as usize, prng, false);

    let (neighbors, dists) = nearest_neighbors
      .iter()
      .map(|sr| (self.mapping.ext_to_int(&sr.item), sr.dist))
      .unzip();
    let q_int = self.mapping.insert(&q);
    self.insert_vertex(q_int, neighbors, dists);

    for r in &visited_nodes {
      let r_int = self.mapping.ext_to_int(&r).clone();
      let is_inserted = self.insert_edge_if_closer(
        r_int,
        q_int,
        visited_node_distances_to_q[&r],
      );
      let r_edges = get_edges_mut_macro!(self, r_int);
      if is_inserted && self.config.use_lgd {
        apply_lgd(r_edges, &self.mapping, q_int, &visited_node_distances_to_q);
      }
    }
    if self.config.use_rrnp {
      self.rrnp(q, q_int, &visited_nodes, &visited_node_distances_to_q);
    }
  }

  fn delete(&mut self, ext_id: T) -> () {
    assert!(self.num_vertices + 1 > self.config.out_degree as u32);
    match self.mapping.external_to_internal_ids.get(&ext_id) {
      None => {
        // TODO: return a warning?
        return;
      }
      Some(&int_id) => {
        for referrer in self
          .backpointers
          .get(int_id as usize)
          .unwrap()
          .clone() // TODO: remove clone
          .iter()
        {
          // For each node currently pointing to int_id, we need to find a
          // new nearest neighbor to
          // replace int_id. We can be inspired by  k-nn descent: neighbors of
          // neighbors are also close to us, so let's use them as candidates.
          // For speed, we will insert the neighbor of neighbor with the least
          // upper bound distance, computed as dist(u,v) + dist(v,w). This
          // requires the triangle inequality to hold for the user's metric
          let referrer_nbrs: HashSet<u32> =
            self.get_edges(referrer).iter().map(|x| x.0).collect();
          let mut referrer_nbrs_of_nbrs =
            self.two_hop_neighbors_and_dist_upper_bounds(referrer);

          let (new_referent, dist_referrer_new_referent) = loop {
            match referrer_nbrs_of_nbrs.pop() {
              None => {
                // If we reach this case, we're in a small connected component
                // and we can't find any neighbors of neighbors who aren't
                // already neighbors or ourself.
                // Instead of looking for neighbors of neighbors, we simply
                // do a search.
                // TODO: after we have implemented the permutation idea, we are
                // guaranteed to have only one connected component and we can drop
                // this case.
                let SearchResults {
                  approximate_nearest_neighbors: nearest_neighbors,
                  ..
                } = self.query_internal(
                  self.mapping.int_to_ext(referrer),
                  // we need to find one new node who isn't a neighbor, the
                  // node being deleted, or referrer itself, so we search for
                  // one more than that number.
                  self.config.out_degree as usize + 3,
                  &mut thread_rng(),
                  false, // we need all the results we can find, so don't ignore occluded nodes
                );

                let nearest_neighbor = nearest_neighbors
                  .iter()
                  .map(|sr| (self.mapping.ext_to_int(&sr.item), sr.dist))
                  .filter(|(res_int_id, _dist)| {
                    **res_int_id != int_id
                      && **res_int_id != referrer
                      && !referrer_nbrs.contains(&res_int_id)
                  })
                  .next();

                match nearest_neighbor {
                  None => panic!(
                    "No replacement neighbors found -- is the graph too small?"
                  ),
                  Some((id, dist)) => {
                    break (*id, dist);
                  }
                }
              }
              Some(Reverse(SearchResult { item: id, .. })) => {
                if id != int_id
                  && id != referrer
                  && !referrer_nbrs.contains(&id)
                {
                  let dist = self.dist_int(referrer, id);
                  break (id, dist);
                }
              }
            }
          };

          self.replace_edge_no_backptr_update(
            referrer,
            int_id,
            new_referent,
            dist_referrer_new_referent,
          );

          // Insert referrer into the backpointers of the new referent.
          let new_referent_backpointers =
            self.backpointers.get_mut(new_referent as usize).unwrap();
          new_referent_backpointers.insert(referrer);
        }

        // Remove backpointers for all nodes the deleted node was pointing to.
        let nbrs: Vec<u32> =
          self.get_edges(int_id).iter().map(|x| x.0).collect();
        for nbr in nbrs {
          let nbr_backpointers =
            self.backpointers.get_mut(nbr as usize).unwrap();

          nbr_backpointers.remove(int_id);
        }

        // Reset backpointers for the deleted node.
        let backpointers = self.backpointers.get_mut(int_id as usize).unwrap();
        *backpointers = SetU32::default();
        self.mapping.delete(&ext_id);
        self.num_vertices -= 1;
        return;
      }
    }
  }

  // TODO: return fewer than expected results if num_searchers or k is
  // higher than num_vertices.

  // TODO: tests for graph where n is not divisible by k, where n is very small,
  // etc.

  // TODO: add summary statistics to SearchResults? min, max, mean distance of
  // visited nodes. Or just compute them in a more complex benchmark program.

  // TODO: test that nearest neighbor search actually returns nearby neighbor
  // on small graphs.

  // TODO: wouldn't returning a min-dist heap be more useful?

  /// Perform beam search for the k nearest neighbors of a point q. Returns
  /// (nearest_neighbors_max_dist_heap, visited_nodes, visited_node_distances_to_q)
  /// This is a translation of the pseudocode of algorithm 1 from
  /// Approximate k-NN Graph Construction: A Generic Online Approach
  fn query<R: RngCore>(
    &self,
    q: &T,
    max_results: usize,
    prng: &mut R,
  ) -> SearchResults<T> {
    self.query_internal(q, max_results, prng, true)
  }
}

#[derive(Debug, Clone, Copy)]
pub struct SearchResult<T> {
  pub item: T,
  pub dist: f32,
}

impl<T> SearchResult<T> {
  pub fn new(item: T, dist: f32) -> SearchResult<T> {
    Self { item, dist }
  }
}

impl<T> PartialEq for SearchResult<T> {
  fn eq(&self, other: &Self) -> bool {
    self.dist == other.dist
  }
}

impl<T> Eq for SearchResult<T> {}

impl<T> PartialOrd for SearchResult<T> {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    self.dist.partial_cmp(&other.dist)
  }
}

impl<T> Ord for SearchResult<T> {
  fn cmp(&self, other: &Self) -> Ordering {
    self.dist.total_cmp(&other.dist)
  }
}

#[derive(Debug, Clone)]
pub struct SearchResults<T> {
  /// Results of the search, in order of increasing distance from the query.
  pub approximate_nearest_neighbors: Vec<SearchResult<T>>,
  // TODO: allow user to optimize this with nohash_hasher somehow.
  // either allow them to pass in a hasher, or use specializations.
  // The former complicates the API but is more extensible.
  // TODO: consolidate visited_nodes and visited_nodes_distance_to_q into one
  // vec of SearchResults.
  /// Set of nodes visited during the search. Most users won't need this info.
  pub visited_nodes: HashSet<T>,
  /// Distances of the nodes in visited_nodes to the query point.
  pub visited_node_distances_to_q: HashMap<T, f32>,
}

impl<T: Clone + Eq + std::hash::Hash> SearchResults<T> {
  pub fn merge(&self, other: &Self) -> SearchResults<T> {
    let mut merged = self.clone();
    let self_nearest_set = self
      .approximate_nearest_neighbors
      .iter()
      .map(|r| r.item.clone())
      .collect::<HashSet<T>>();
    for result in other.approximate_nearest_neighbors.iter() {
      if !self_nearest_set.contains(&result.item) {
        merged.approximate_nearest_neighbors.push(result.clone());
      }
    }

    merged
      .approximate_nearest_neighbors
      .sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
    merged.visited_nodes.extend(other.visited_nodes.clone());
    merged
      .visited_node_distances_to_q
      .extend(other.visited_node_distances_to_q.clone());
    merged
  }
}

impl<T> Default for SearchResults<T> {
  fn default() -> Self {
    Self {
      approximate_nearest_neighbors: Vec::new(),
      visited_nodes: HashSet::new(),
      visited_node_distances_to_q: HashMap::new(),
    }
  }
}

/// Constructs an exact k-nn graph on the given IDs. O(n^2).
/// `capacity` is max capacity of the returned graph (for future inserts).
/// Must be >= n.
pub fn exhaustive_knn_graph<
  'a,
  T: Clone + Eq + std::hash::Hash,
  S: BuildHasher + Clone,
>(
  ids: Vec<&T>,
  config: KNNGraphConfig<'a, T, S>,
) -> DenseKNNGraph<'a, T, S> {
  let n = ids.len();
  // TODO: return Either

  assert!(
    config.out_degree as usize <= n,
    "out_degree ({}) must be <= ids.len() ({})",
    config.out_degree,
    n,
  );

  let mut g: DenseKNNGraph<T, S> = DenseKNNGraph::empty(config.clone());

  for i_ext in ids.iter() {
    let mut knn = BinaryHeap::new();
    let i = g.mapping.insert(i_ext.clone());
    for j_ext in ids.iter() {
      if i_ext == j_ext {
        continue;
      }
      let j = g.mapping.insert(*j_ext);
      let dist = (config.dist_fn)(i_ext, j_ext);
      knn.push(SearchResult::new(j, dist));

      while knn.len() > config.out_degree as usize {
        knn.pop();
      }
    }
    for edge_ix in 0..config.out_degree as usize {
      let SearchResult { item: id, dist } = knn.pop().unwrap();
      g.edges[(i as usize * config.out_degree as usize + edge_ix)] =
        (id, dist, DEFAULT_LAMBDA);
      let s = &mut g.backpointers[id as usize];
      s.insert(i);
    }
  }

  g.num_vertices = ids.len() as u32;

  for i in 0..n {
    g.sort_edges(i as u32)
  }

  g
}

#[cfg(test)]
mod tests {
  extern crate ordered_float;
  use std::{collections::hash_map::RandomState, hash::BuildHasherDefault};

  use self::ordered_float::OrderedFloat;
  use super::*;
  use nohash_hasher::NoHashHasher;
  use rand::SeedableRng;
  use rand_xoshiro::Xoshiro256StarStar;

  type NHH = BuildHasherDefault<NoHashHasher<u32>>;

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

  pub fn sq_euclidean_faster<T: PrimitiveToF32 + Copy>(
    v1: &[T],
    v2: &[T],
  ) -> f32 {
    let mut result = 0.0;
    let n = v1.len();
    for i in 0..n {
      let diff = v2[i].tof32() - v1[i].tof32();
      result += diff * diff;
    }
    result
  }

  fn mk_config<'a, T>(
    capacity: u32,
    dist_fn: &'a (dyn Fn(&T, &T) -> f32 + Sync),
  ) -> KNNGraphConfig<'a, T, NHH> {
    let out_degree = 5;
    let num_searchers = 5;
    let use_rrnp = false;
    let rrnp_max_depth = 2;
    let use_lgd = false;
    let build_hasher = nohash_hasher::BuildNoHashHasher::default();
    KNNGraphConfig::<'a, T, NHH> {
      capacity,
      out_degree,
      num_searchers,
      dist_fn,
      build_hasher,
      use_rrnp,
      rrnp_max_depth,
      use_lgd,
    }
  }

  #[test]
  fn test_exhaustive_knn_graph() {
    let db: Vec<[i32; 1]> = vec![[1], [2], [3], [10], [11], [12]];
    let dist_fn = &|x: &u32, y: &u32| {
      sq_euclidean_faster(&db[*x as usize], &db[*y as usize])
    };
    let mut config = mk_config(10, dist_fn);
    config.out_degree = 2;
    let g: DenseKNNGraph<u32, NHH> =
      exhaustive_knn_graph(vec![&0u32, &1, &2, &3, &4, &5], config);
    g.consistency_check();
    assert_eq!(g.debug_get_neighbor_indices(0), vec![1, 2]);
    assert_eq!(g.debug_get_neighbor_indices(1), vec![2, 0]);
    assert_eq!(g.debug_get_neighbor_indices(2), vec![1, 0]);
    assert_eq!(g.debug_get_neighbor_indices(3), vec![4, 5]);
    assert_eq!(g.debug_get_neighbor_indices(4), vec![3, 5]);
    assert_eq!(g.debug_get_neighbor_indices(5), vec![4, 3]);
  }

  #[test]
  fn test_beam_search_fully_connected_graph() {
    let db = vec![[1f32], [2f32], [3f32], [10f32], [11f32], [12f32]];
    let dist_fn = &|x: &u32, y: &u32| {
      sq_euclidean_faster(&db[*x as usize], &db[*y as usize])
    };
    let mut config = mk_config(10, dist_fn);
    config.out_degree = 2;
    let g: DenseKNNGraph<u32, NHH> =
      exhaustive_knn_graph(vec![&0, &1, &2, &3, &4, &5], config);
    g.consistency_check();
    let mut prng = Xoshiro256StarStar::seed_from_u64(1);
    let SearchResults {
      approximate_nearest_neighbors: nearest_neighbors,
      ..
    } = g.query(&1, 2, &mut prng);
    assert_eq!(
      nearest_neighbors
        .iter()
        .map(|x| (x.item, x.dist))
        .collect::<Vec<(u32, f32)>>(),
      vec![(1u32, 0.0), (0, 1.0)]
    );
  }

  #[test]
  fn test_insert_vertex() {
    let mut config = mk_config(3, &|_x, _y| 1.0);
    config.out_degree = 2;
    let mut g: DenseKNNGraph<u32, BuildHasherDefault<NoHashHasher<u32>>> =
      DenseKNNGraph::empty(config);
    // NOTE:: g.insert_vertex() calls get_edges_mut which asserts that the
    // internal id is in the mapping. g.mapping.insert() assigns internal ids in
    // order starting from 0, so we are inserting 3 unused u32 external ids,
    // 0,1,2 which get mapped to internal ids 0,1,2. Annoying and unclear, but
    // necessary at the moment.
    g.mapping.insert(&0);
    g.mapping.insert(&1);
    g.mapping.insert(&2);
    g.insert_vertex(0, vec![2, 1], vec![2.0, 1.0]);
    g.insert_vertex(1, vec![2, 0], vec![1.0, 1.0]);
    g.insert_vertex(2, vec![0, 1], vec![2.0, 1.0]);

    assert_eq!(g.get_edges(0), [(1, 1.0, 0), (2, 2.0, 0)].as_slice());
    assert_eq!(g.get_edges(1), [(2, 1.0, 0), (0, 1.0, 0)].as_slice());
    assert_eq!(g.get_edges(2), [(1, 1.0, 0), (0, 2.0, 0)].as_slice());
  }

  #[test]
  #[should_panic]
  fn test_insert_vertex_panic_too_many_vertex() {
    let mut g: DenseKNNGraph<u32, NHH> =
      DenseKNNGraph::empty(mk_config(2, &|_x, _y| 1.0));
    g.insert_vertex(0, vec![2, 1], vec![2.0, 1.0]);
    g.insert_vertex(1, vec![2, 0], vec![1.0, 1.0]);
    g.insert_vertex(2, vec![0, 1], vec![2.0, 1.0]);
  }

  #[test]
  #[should_panic]
  fn test_insert_vertex_panic_wrong_neighbor_length() {
    let mut g: DenseKNNGraph<u32, NHH> =
      DenseKNNGraph::empty(mk_config(2, &|_x, _y| 1.0));
    g.insert_vertex(0, vec![2, 1, 0], vec![2.0, 1.0, 10.1]);
    g.insert_vertex(1, vec![2, 0], vec![1.0, 1.0]);
  }

  #[test]
  fn test_insert_edge_if_closer() {
    let mut config = mk_config(3, &|&_x, &_y| 1.0);
    config.out_degree = 1;
    let mut g: DenseKNNGraph<u32, NHH> = DenseKNNGraph::empty(config);

    // see note in test_insert_vertex
    g.mapping.insert(&0);
    g.mapping.insert(&1);
    g.mapping.insert(&2);

    g.insert_vertex(0, vec![2], vec![2.0]);
    g.insert_vertex(1, vec![2], vec![1.0]);
    g.insert_vertex(2, vec![1], vec![1.0]);

    assert_eq!(g.get_edges(0), [(2, 2.0, 0)].as_slice());

    assert!(g.insert_edge_if_closer(0, 1, 1.0));

    assert_eq!(g.get_edges(0), [(1, 1.0, 0)].as_slice());
    assert_eq!(g.get_edges(1), [(2, 1.0, 0)].as_slice());
    assert_eq!(g.get_edges(2), [(1, 1.0, 0)].as_slice());
  }

  #[test]
  fn test_insert() {
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
    let dist_fn = &|x: &u32, y: &u32| {
      sq_euclidean_faster(&db[*x as usize], &db[*y as usize])
    };
    let mut g: DenseKNNGraph<u32, NHH> = exhaustive_knn_graph(
      vec![&0, &1, &2, &3, &4, &5],
      mk_config(11, dist_fn),
    );
    let mut prng = Xoshiro256StarStar::seed_from_u64(1);
    for i in 6..11 {
      println!("doing {}", i);
      g.insert(i, &mut prng);
    }
    g.consistency_check();
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
    let dist_fn = &|x: &u32, y: &u32| {
      sq_euclidean_faster(&db[*x as usize], &db[*y as usize])
    };
    let mut g: KNN<u32, NHH> = KNN::new(mk_config(10, dist_fn));
    let mut prng = Xoshiro256StarStar::seed_from_u64(1);
    g.insert(0, &mut prng);

    g.debug_consistency_check();
  }

  #[test]
  fn test_delete() {
    let db: Vec<[i32; 1]> = vec![[1], [2], [3], [10], [11], [12]];
    let dist_fn = &|x: &u32, y: &u32| {
      sq_euclidean_faster(&db[*x as usize], &db[*y as usize])
    };
    let mut config = mk_config(10, dist_fn);
    config.out_degree = 2;
    let mut g: DenseKNNGraph<u32, NHH> =
      exhaustive_knn_graph(vec![&0u32, &1, &2, &3, &4, &5], config);
    g.delete(3);
    g.consistency_check();
  }

  #[test]
  fn test_delete_large() {
    let db: Vec<[i32; 1]> = vec![
      [1],
      [2],
      [3],
      [4],
      [5],
      [6],
      [7],
      [8],
      [9],
      [10],
      [11],
      [12],
      [13],
      [14],
      [15],
      [16],
    ];
    let dist_fn = &|x: &u32, y: &u32| {
      sq_euclidean_faster(&db[*x as usize], &db[*y as usize])
    };
    let mut config = mk_config(30, dist_fn);
    config.out_degree = 5;
    let ids = (0u32..16).collect::<Vec<_>>();
    let mut g: DenseKNNGraph<u32, NHH> =
      exhaustive_knn_graph(ids.iter().collect(), config);
    g.consistency_check();
    g.delete(7);
    g.delete(15);
    g.delete(0);
    g.consistency_check();
  }

  #[test]
  fn test_use_rrnp() {
    let db: Vec<[i32; 1]> = vec![[1], [2], [3], [10], [11], [12], [6]];
    let dist_fn = &|x: &u32, y: &u32| {
      sq_euclidean_faster(&db[*x as usize], &db[*y as usize])
    };
    let mut config = mk_config(10, dist_fn);
    config.out_degree = 2;
    config.use_rrnp = true;
    let mut g: DenseKNNGraph<u32, NHH> =
      exhaustive_knn_graph(vec![&0u32, &1, &2, &3, &4, &5], config);
    g.consistency_check();
    let mut prng = Xoshiro256StarStar::seed_from_u64(1);
    g.insert(6, &mut prng);
    g.consistency_check();
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
    let dist_fn = &|x: &Vec<OrderedFloat<f32>>, y: &Vec<OrderedFloat<f32>>| {
      sq_euclidean_faster(
        &x.iter().map(|x| x.into_inner()).collect::<Vec<f32>>(),
        &y.iter().map(|x| x.into_inner()).collect::<Vec<f32>>(),
      )
    };
    for use_rrnp in [false, true] {
      for use_lgd in [false, true] {
        let s = RandomState::new();
        let config =
          KNNGraphConfig::new(50, 5, 5, dist_fn, s, use_rrnp, 2, use_lgd);

        let g = exhaustive_knn_graph(
          vec![
            &vec![
              OrderedFloat(1f32),
              OrderedFloat(2f32),
              OrderedFloat(3f32),
              OrderedFloat(4f32),
            ],
            &vec![
              OrderedFloat(2f32),
              OrderedFloat(4f32),
              OrderedFloat(5f32),
              OrderedFloat(6f32),
            ],
            &vec![
              OrderedFloat(3f32),
              OrderedFloat(4f32),
              OrderedFloat(5f32),
              OrderedFloat(12f32),
            ],
            &vec![
              OrderedFloat(23f32),
              OrderedFloat(14f32),
              OrderedFloat(45f32),
              OrderedFloat(142f32),
            ],
            &vec![
              OrderedFloat(37f32),
              OrderedFloat(45f32),
              OrderedFloat(53f32),
              OrderedFloat(122f32),
            ],
            &vec![
              OrderedFloat(13f32),
              OrderedFloat(14f32),
              OrderedFloat(555f32),
              OrderedFloat(125f32),
            ],
            &vec![
              OrderedFloat(13f32),
              OrderedFloat(4f32),
              OrderedFloat(53f32),
              OrderedFloat(12f32),
            ],
            &vec![
              OrderedFloat(33f32),
              OrderedFloat(4f32),
              OrderedFloat(53f32),
              OrderedFloat(312f32),
            ],
          ],
          config,
        );

        let mut prng = Xoshiro256StarStar::seed_from_u64(1);

        let SearchResults {
          approximate_nearest_neighbors: nearest_neighbors,
          ..
        } = g.query(
          &vec![
            OrderedFloat(34f32),
            OrderedFloat(5f32),
            OrderedFloat(53f32),
            OrderedFloat(312f32),
          ],
          1,
          &mut prng,
        );
        assert_eq!(
          nearest_neighbors
            .iter()
            .map(|x| (x.item.clone(), x.dist))
            .collect::<Vec<(Vec<OrderedFloat<f32>>, f32)>>()[0]
            .0,
          vec![
            OrderedFloat(33f32),
            OrderedFloat(4f32),
            OrderedFloat(53f32),
            OrderedFloat(312f32),
          ]
        );
      }
    }
  }

  #[test]
  fn test_search_results_merge() {
    let sr1: SearchResults<u32> = SearchResults {
      approximate_nearest_neighbors: vec![
        SearchResult { item: 1, dist: 1.0 },
        SearchResult { item: 2, dist: 2.0 },
      ],
      visited_nodes: HashSet::from([1, 2, 3]),
      visited_node_distances_to_q: HashMap::from([
        (1, 1.0),
        (2, 2.0),
        (3, 3.0),
      ]),
    };
    let sr2: SearchResults<u32> = SearchResults {
      approximate_nearest_neighbors: vec![
        SearchResult { item: 3, dist: 3.0 },
        SearchResult { item: 4, dist: 4.0 },
      ],
      visited_nodes: HashSet::from([1, 3, 4]),
      visited_node_distances_to_q: HashMap::from([
        (1, 1.0),
        (3, 3.0),
        (4, 4.0),
      ]),
    };

    let sr3 = sr1.merge(&sr2);
    assert_eq!(
      sr3.approximate_nearest_neighbors,
      vec![
        SearchResult { item: 1, dist: 1.0 },
        SearchResult { item: 2, dist: 2.0 },
        SearchResult { item: 3, dist: 3.0 },
        SearchResult { item: 4, dist: 4.0 },
      ]
    );
    assert_eq!(sr3.visited_nodes, HashSet::from([1, 2, 3, 4]),);
    assert_eq!(
      sr3.visited_node_distances_to_q,
      HashMap::from([(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0),]),
    );
  }
}
