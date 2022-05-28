#![feature(total_cmp)]
#![feature(is_sorted)]
extern crate nohash_hasher;
extern crate rand;
extern crate rand_xoshiro;
extern crate tinyset;

use nohash_hasher::IntMap;
use rand::seq::index::sample;
use rand::RngCore;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::binary_heap::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
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
// improves performance.

// TODO: deletion. I think this is achievable. Tricky parts are:
// 1. Backpointers. We need to update an unbounded number of referrers to the
//    deleted node.
// 2. Edges. Once we have the referrers, we need to delete one of their edges
//    that point to the deleted node. We can either replace it with a new edge,
//    or allow the number of edges to drop below the configured out_degree.
//    I am leaning toward replacement. We can use the set of neighbors of the
//    deleted node as candidates for replacement.
//    However, if we do want to allow the set of out-edges to shrink, we have
//    some options.
//    Because we're storing adjacency in a flat Vec, this requires us to
//    1. Either wrap each element in `Option` or store the length of the edges
//       subvector for each node (moving invalid items to the end). I am
//       leaning toward the latter because it should optimize better -- if
//       the user doesn't want deletion, we don't need to store the lengths at
//       all, saving memory and reducing calculations.
//    2. If the out-degree of the node has dropped to zero because of deletions,
//       we need to fire off a search to find it a new set of `out_degree`
//       neighbors. If we allow it to stay at zero, it will be unsearchable.
// 3. Keeping track of which indices are currently deleted. This could be stored
//    in a vector (pre-allocated to the length `capacity`). Respecting its
//    contents is the hard part:
//    1. On insertion of a new node, we need to reuse a deleted index if there
//       are any.
//    2. When searching, we can only start from non-deleted indices. If we have
//       made a huge number of  deletions, simply finding non-deleted indices
//       could become a slow linear search. Possible mitigation: predetermined,
//       fixed starting points chosen with a pivot selection algorithm. But we
//       would need to be careful about bounding the amount of time needed to
//       update the set of pivots when deletions occur, which may just leave us
//       back where we started, with the same problem of tracking which indices
//       still exist.
//       Another idea: in the basic algorithm, we need a random starting set,
//       but each query could share the same random starting set... that
//       suggests that we choose the random starting set once (or infrequently),
//       then if an element of that set gets deleted, we search for a
//       replacement. If we assume that element was chosen randomly, certainly
//       one of its nearest neighbors would be a suitable replacement, right?
// 4. Edge cases I can think of:
//    1. An individual node has no neighbors left. Fix: search for new ones on
//       deletion of its last neighbor.
//    2. Graph gets broken into multiple components. Mitigation: multiple start
//       points means by chance one should be in each component, unless some
//       components are really small. Possible fix: the permutation path idea.
//       See above.
//    3. The number of nodes drops to 2, 1, or 0. I don't think 2 is an edge
//       case, but it might be for some reason. 1 is definitely, because it hits
//       edge case 1 above -- it will have no neighbors. 0 is a big edge case,
//       because we need to make searches immediately fail with no results.
// 5. The occlusion stuff needs to be updated based on deletion. This could be
//    quite difficult, but I think the paper mentions some ideas.
// 6. We could allow the user to enable deletion after the graph is constructed.
//    It wouldn't be too hard -- just need to flip a bool and allocate some
//    memory.

#[derive(Clone, Copy)]
pub struct KNNGraphConfig<'a, T> {
  /// The max number of vertices that can be inserted into the graph. Constant.
  pub capacity: u32,
  /// The number of approximate nearest neighbors to store for each inserted
  /// element. This is a constant.
  pub out_degree: u32,
  /// Number of simultaneous searchers in the beam search.
  pub num_searchers: u32,
  /// distance function. Must satisfy the criteria of a metric:
  /// https://en.wikipedia.org/wiki/Metric_(mathematics)
  pub dist_fn: &'a dyn Fn(&T, &T) -> f32,
  /// Whether to use restricted recursive neighborhood propagation. This improves
  /// search speed, but decreases insertion throughput. TODO: verify that's
  /// true.
  pub use_rrnp: bool,
  /// Whether to use lazy graph diversification. This improves search speed.
  /// TODO: parametrize this type so that the LGD vector is never allocated/
  /// takes no memory if this is set to false.
  pub use_lgd: bool,
}

// NOTE: can't do a Default because we can't guess a reasonable capacity.
// TODO: auto-resize like a Vec?

impl<'a, T> KNNGraphConfig<'a, T> {
  /// Create a new KNNGraphConfig.
  pub fn new(
    capacity: u32,
    out_degree: u32,
    num_searchers: u32,
    dist_fn: &'a dyn Fn(&T, &T) -> f32,
    use_rrnp: bool,
    use_lgd: bool,
  ) -> KNNGraphConfig<'a, T> {
    KNNGraphConfig::<'a, T> {
      capacity,
      out_degree,
      num_searchers,
      dist_fn,
      use_rrnp,
      use_lgd,
    }
  }
}

// TODO: test that query can find already-inserted items

trait NN<T> {
  // TODO: return types with error sums, more informative delete (did it exist?)
  // etc.

  // TODO: more functions in this interface.

  fn insert<R: RngCore>(&mut self, x: T, prng: &mut R) -> ();
  fn delete(&mut self, x: T) -> ();
  fn query<R: RngCore>(
    &self,
    max_results: usize,
    prng: &mut R,
    q: T,
  ) -> SearchResults<T>;
}

pub struct BruteForceKNN<'a, T> {
  pub contents: HashSet<T>,
  pub distance: &'a dyn Fn(&T, &T) -> f32,
}

impl<'a, T> BruteForceKNN<'a, T> {
  pub fn new(distance: &'a dyn Fn(&T, &T) -> f32) -> BruteForceKNN<'a, T> {
    BruteForceKNN {
      contents: HashSet::new(),
      distance,
    }
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
    max_results: usize,
    _prng: &mut R,
    q: T,
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
      nearest_neighbors_max_dist_heap,
      visited_nodes,
      visited_node_distances_to_q,
    }
  }
}

/// Maps from the user's chosen ID type to our internal u32 ids that are used
/// within the core search functions to keep things fast and compact.
/// Ideally, we should translate to and from user ids at the edges of
/// performance-critical code. In practice, doing so may be difficult, since the
/// user's distance callback is passed the user's IDs.
#[derive(Debug)]
pub struct InternalExternalIDMapping<T, S: BuildHasher> {
  pub capacity: u32,
  // TODO: using a vec might be faster.
  pub internal_to_external_ids: IntMap<u32, T>,
  pub external_to_internal_ids: HashMap<T, u32, S>,
  // TODO: we may also need a fast is_deleted check for internal ids.
  pub deleted: Vec<u32>,
}

impl<T: Copy + Eq + std::hash::Hash, S: BuildHasher>
  InternalExternalIDMapping<T, S>
{
  fn with_capacity_and_hasher(capacity: u32, hash_builder: S) -> Self {
    let internal_to_external_ids = IntMap::<u32, T>::with_capacity_and_hasher(
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

  fn insert(self: &mut Self, x: T) -> u32 {
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

        self.internal_to_external_ids.insert(x_int, x);
        self.external_to_internal_ids.insert(x, x_int);
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
  S: BuildHasher,
>(
  bf: &mut BruteForceKNN<'a, T>,
  config: KNNGraphConfig<'a, T>,
  build_hasher: S, // TODO: move build_hasher into config
) -> DenseKNNGraph<'a, T, S> {
  let ids = bf.contents.iter().collect();
  exhaustive_knn_graph(ids, config, build_hasher)
}

/// Switches from brute-force to approximate nearest neighbor search based on
/// the number of inserted elements.
pub enum KNN<'a, T, S: BuildHasher> {
  Small {
    g: BruteForceKNN<'a, T>,
    config: KNNGraphConfig<'a, T>,
    build_hasher: S,
  },

  Large(DenseKNNGraph<'a, T, S>),
}

impl<'a, T: Copy + Ord + Eq + std::hash::Hash, S: BuildHasher + Clone> NN<T>
  for KNN<'a, T, S>
{
  fn insert<R: RngCore>(&mut self, x: T, prng: &mut R) -> () {
    match self {
      KNN::Small {
        g,
        config,
        build_hasher,
      } => {
        if config.capacity as usize == g.contents.len() {
          panic!("TODO create error type etc.");
        } else if g.contents.len() == 100 {
          *self = KNN::Large(convert_bruteforce_to_dense(
            g,
            config.clone(),
            build_hasher.clone(),
          ));
          self.insert(x, prng);
        } else {
          g.insert(x, prng);
        }
      }
      KNN::Large(g) => g.insert(x, prng),
    }
  }

  fn delete(&mut self, x: T) -> () {
    unimplemented!()
  }

  fn query<R: RngCore>(
    &self,
    max_results: usize,
    prng: &mut R,
    q: T,
  ) -> SearchResults<T> {
    unimplemented!()
  }
}

/// A directed graph stored contiguously in memory as an adjacency list.
/// All vertices are guaranteed to have the same out-degree.
/// Nodes are u32s numbered from 0 to n-1.
pub struct DenseKNNGraph<'a, T, S: BuildHasher> {
  /// A mapping from the user's ID type, T, to our internal ids, which are u32.
  /// TODO: now that we have to store a hashmap anyway, are we gaining anything
  /// by storing edges as a vector? We could potentially simplify by eliminating
  /// it and the external/internal id distinction. I think all we gain is a
  /// more memory-efficient adjacency list... which may be a decisive factor.
  pub mapping: InternalExternalIDMapping<T, S>,
  /// n, the current number of vertices in the graph. The valid indices of the
  /// graph are 0 to n-1.
  pub num_vertices: u32,
  /// The underlying buffer of num_vertices*out_degree neighbor information.
  /// An adjacency list of (node id, distance, lambda crowding factor
  /// (not yet implemented, always zero)).
  /// Use with caution.
  /// Prefer to use indexing to access the neighbors of a vertex.
  /// `g[i]` returns a slice of length `out_degree` of the neighbors of `i` along
  /// with their distances from i.
  pub edges: Vec<(u32, f32, u8)>,
  /// Maintains an association between vertices and the vertices that link out to
  /// them. In other words, each backpointers[i] is the set of vertices S s.t.
  /// for all x in S, a directed edge exists pointing from x to i.
  pub backpointers: Vec<SetU32>,
  pub config: KNNGraphConfig<'a, T>,
}

impl<'a, T: Copy + Eq + std::hash::Hash, S: BuildHasher>
  DenseKNNGraph<'a, T, S>
{
  /// Allocates a graph of the specified size and out_degree, but
  /// doesn't populate the edges.
  fn empty(
    config: KNNGraphConfig<'a, T>,
    build_hasher: S,
  ) -> DenseKNNGraph<'a, T, S> {
    let mapping = InternalExternalIDMapping::<T, S>::with_capacity_and_hasher(
      config.capacity,
      build_hasher,
    );

    let edges =
      Vec::with_capacity(config.capacity as usize * config.out_degree as usize);

    let mut backpointers = Vec::with_capacity(config.capacity as usize);

    for _ in 0..config.capacity {
      backpointers.push(SetU32::new());
    }

    let num_vertices = 0;

    // TODO: expose as params once supported.
    let use_rrnp = false;
    let use_lgd = false;

    DenseKNNGraph {
      mapping,
      num_vertices,
      edges,
      backpointers,
      config,
    }
  }

  /// Get the neighbors of u and their distances. Panics if index
  /// >= num_vertices.
  pub fn get_edges(&self, index: u32) -> &[(u32, f32, u8)] {
    assert!(index < self.num_vertices);

    let i = index * self.config.out_degree;
    let j = i + self.config.out_degree;
    &self.edges[i as usize..j as usize]
  }

  fn debug_get_neighbor_indices(&self, index: u32) -> Vec<u32> {
    self.get_edges(index).iter().map(|e| e.0).collect()
  }

  /// Get the neighbors of u and their distances. Panics if index >= num_vertices.
  fn get_edges_mut(&mut self, index: u32) -> &mut [(u32, f32, u8)] {
    assert!(index < self.num_vertices);

    let i = index * self.config.out_degree;
    let j = i + self.config.out_degree;
    &mut self.edges[i as usize..j as usize]
  }

  fn sort_edges(&mut self, index: u32) {
    let edges = self.get_edges_mut(index);
    edges.sort_by(|a, b| a.1.total_cmp(&b.1));
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

  /// Insert a new vertex into the graph, given its k neighbors and their
  /// distances. Panics if the graph is already full (num_vertices == capacity).
  /// nbrs and dists must be equal to the out_degree of the graph.
  fn insert_vertex(&mut self, u: u32, nbrs: Vec<u32>, dists: Vec<f32>) {
    //TODO: replace all asserts with Either return values
    assert!(self.num_vertices < self.config.capacity);

    let od = self.config.out_degree as usize;
    assert!(nbrs.len() == od && dists.len() == od);

    for (nbr, dist) in nbrs.iter().zip(dists) {
      self.edges.push((*nbr, dist, DEFAULT_LAMBDA));
      let s = &mut self.backpointers[*nbr as usize];
      s.insert(u);
    }

    self.num_vertices += 1;
    self.sort_edges(u);
  }

  pub fn debug_print(&self) {
    println!("### Adjacency list (index, distance, lambda)");
    for i in 0..self.num_vertices {
      println!("Node {}", i);
      println!("{:#?}", self.get_edges(i));
    }
    println!("### Backpointers");
    println!("{:#?}", self.backpointers);
  }

  /// Panics if graph is internally inconsistent.
  fn consistency_check(&self) {
    if self.edges.len()
      != self.num_vertices as usize * self.config.out_degree as usize
    {
      panic!(
        "edges.len() is not equal to num_vertices * out_degree. {} != {} * {}",
        self.edges.len(),
        self.num_vertices,
        self.config.out_degree
      );
    }

    for i in 0..self.num_vertices {
      for (nbr, _, _) in self.get_edges(i).iter() {
        if *nbr == i {
          panic!("Self loop at node {}", i);
        }
        let nbr_backptrs = &self.backpointers[*nbr as usize];
        if !nbr_backptrs.contains(i) {
          panic!(
            "Vertex {} links to {} but {}'s backpointers don't include {}!",
            i, nbr, nbr, i
          );
        }
      }

      assert!(self.get_edges(i).is_sorted_by_key(|e| e.1));

      for referrer in self.backpointers[i as usize].iter() {
        assert!(self.debug_get_neighbor_indices(referrer).contains(&i));
      }
    }
  }
}

impl<'a, T: Copy + Ord + Eq + std::hash::Hash, S: BuildHasher> NN<T>
  for DenseKNNGraph<'a, T, S>
{
  fn insert<R: RngCore>(&mut self, x: T, prng: &mut R) -> () {
    todo!()
  }

  fn delete(&mut self, x: T) -> () {
    todo!()
  }

  fn query<R: RngCore>(
    &self,
    max_results: usize,
    prng: &mut R,
    q: T,
  ) -> SearchResults<T> {
    todo!()
  }
}

#[derive(Debug, Clone, Copy)]
pub struct SearchResult<T> {
  pub id: T,
  pub dist: f32,
}

impl<T> SearchResult<T> {
  pub fn new(id: T, dist: f32) -> SearchResult<T> {
    Self { id, dist }
  }
}

impl<T: std::cmp::PartialOrd + std::cmp::PartialEq> PartialEq
  for SearchResult<T>
{
  fn eq(&self, other: &Self) -> bool {
    self.id == other.id
  }
}

impl<T: std::cmp::PartialOrd + std::cmp::PartialEq> Eq for SearchResult<T> {}

impl<T: std::cmp::PartialOrd + std::cmp::PartialEq> PartialOrd
  for SearchResult<T>
{
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    self.dist.partial_cmp(&other.dist)
  }
}

impl<T: Ord> Ord for SearchResult<T> {
  fn cmp(&self, other: &Self) -> Ordering {
    self.dist.total_cmp(&other.dist)
  }
}

pub struct SearchResults<T> {
  pub nearest_neighbors_max_dist_heap: BinaryHeap<SearchResult<T>>,
  // TODO: allow user to optimize this with nohash_hasher somehow.
  // either allow them to pass in a hasher, or use specializations.
  // The former complicates the API but is more extensible.
  pub visited_nodes: HashSet<T>,
  pub visited_node_distances_to_q: HashMap<T, f32>,
}

// TODO: return fewer than expected results if num_searchers or k is
// higher than num_vertices.

// TODO: tests for graph where n is not divisible by k, where n is very small,
// etc.

// TODO: add summary statistics to SearchResults? min, max, mean distance of
// visited nodes. Or just compute them in a more complex benchmark program.

/// Perform beam search for the k nearest neighbors of a point q. Returns
/// (nearest_neighbors_max_dist_heap, visited_nodes, visited_node_distances_to_q)
/// This is a translation of the pseudocode of algorithm 1 from
/// Approximate k-NN Graph Construction: A Generic Online Approach
pub fn knn_beam_search<
  T: Copy + Ord + Eq + std::hash::Hash,
  S: BuildHasher,
  R: RngCore,
>(
  g: &DenseKNNGraph<T, S>,
  q: T,
  k: usize,
  prng: &mut R,
) -> SearchResults<T> {
  assert!(g.config.num_searchers <= g.num_vertices);
  assert!(k <= g.num_vertices as usize);
  let mut q_max_heap: BinaryHeap<SearchResult<T>> = BinaryHeap::new();
  let mut r_min_heap: BinaryHeap<Reverse<SearchResult<T>>> = BinaryHeap::new();
  let mut visited = HashSet::<T>::new();
  let mut visited_distances: HashMap<T, f32> = HashMap::default();

  // lines 2 to 10 of the pseudocode
  for r in sample(
    prng,
    g.num_vertices as usize,
    g.config.num_searchers as usize,
  ) {
    let r = r as u32;
    let r_ext = g.mapping.int_to_ext(r);
    let r_dist = (g.config.dist_fn)(&q, r_ext);
    visited.insert(*r_ext);
    visited_distances.insert(*r_ext, r_dist);
    r_min_heap.push(Reverse(SearchResult::new(*r_ext, r_dist)));
    match q_max_heap.peek() {
      None => {
        q_max_heap.push(SearchResult::new(*r_ext, r_dist));
        // NOTE: pseudocode has a bug: R.insert(r) at both line 2 and line 8
        // We are skipping it here since we did it above.
      }
      Some(f) => {
        if r_dist < f.dist || q_max_heap.len() < k {
          q_max_heap.push(SearchResult::new(*r_ext, r_dist));
        }
      }
    }
  }

  // lines 11 to 27 of the pseudocode
  while r_min_heap.len() > 0 {
    while q_max_heap.len() > k {
      q_max_heap.pop();
    }

    let Reverse(sr) = r_min_heap.pop().unwrap();
    let &f = { q_max_heap.peek().unwrap() };
    let sr_int = g.mapping.ext_to_int(&sr.id);
    if sr.dist > f.dist {
      break;
    }

    let mut r_nbrs = g.backpointers[*sr_int as usize].clone();
    for (nbr, _, _) in g.get_edges(*sr_int).iter() {
      r_nbrs.insert(*nbr);
    }

    for e in r_nbrs.iter() {
      let e_ext = g.mapping.int_to_ext(e);
      if !visited.contains(&e_ext) {
        visited.insert(*e_ext);
        let e_dist = (g.config.dist_fn)(&q, e_ext);
        let f_dist = (g.config.dist_fn)(&q, &f.id);
        visited_distances.insert(*e_ext, e_dist);
        if e_dist < f_dist || q_max_heap.len() < k {
          q_max_heap.push(SearchResult::new(*e_ext, e_dist));
          r_min_heap.push(Reverse(SearchResult::new(*e_ext, e_dist)));
        }
      }
    }
  }

  SearchResults {
    nearest_neighbors_max_dist_heap: q_max_heap,
    visited_nodes: visited,
    visited_node_distances_to_q: visited_distances,
  }
}

/// Constructs an exact k-nn graph on the given IDs. O(n^2).
/// `capacity` is max capacity of the returned graph (for future inserts).
/// Must be >= n.
pub fn exhaustive_knn_graph<
  'a,
  T: Copy + Eq + std::hash::Hash,
  S: BuildHasher,
>(
  ids: Vec<&T>,
  config: KNNGraphConfig<'a, T>,
  build_hasher: S,
) -> DenseKNNGraph<'a, T, S> {
  let n = ids.len();
  // TODO: return Either
  if config.out_degree >= n as u32 {
    panic!("k must be less than n");
  }

  let mut g: DenseKNNGraph<T, S> = DenseKNNGraph::empty(config, build_hasher);

  for i_ext in ids.iter() {
    let mut knn = BinaryHeap::new();
    let i = g.mapping.insert(**i_ext);
    for j_ext in ids.iter() {
      if i_ext == j_ext {
        continue;
      }
      let j = g.mapping.insert(**j_ext);
      let dist = (config.dist_fn)(i_ext, j_ext);
      knn.push(SearchResult::new(j, dist));

      while knn.len() > config.out_degree as usize {
        knn.pop();
      }
    }
    while knn.len() > 0 {
      let SearchResult { id, dist } = knn.pop().unwrap();
      g.edges.push((id, dist, DEFAULT_LAMBDA));
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

fn rrnp<T, S: BuildHasher>(
  _g: &mut DenseKNNGraph<T, S>,
  _nearest_neighbors_max_dist_heap: &BinaryHeap<SearchResult<T>>,
  _visited_nodes: &HashSet<T>,
  _visited_nodes_distance_to_q: &HashMap<T, f32>,
) -> () {
  unimplemented!()
}

fn apply_lgd<T, S: BuildHasher>(
  _g: &mut DenseKNNGraph<T, S>,
  _nearest_neighbors_max_dist_heap: &BinaryHeap<SearchResult<T>>,
  _visited_nodes: &HashSet<T>,
  _visited_nodes_distance_to_q: &HashMap<T, f32>,
) -> () {
  unimplemented!()
}

// TODO: just rename insert_approx to insert?

/// Inserts a new data point into the graph. The graph must not be full.
/// Optionally performs restricted recursive neighborhood propagation and
/// lazy graph diversification. These options are set when the graph is
/// constructed.
///
/// This is equivalent to one iteration of Algorithm 3 in
/// "Approximate k-NN Graph Construction: A Generic Online Approach".
pub fn insert_approx<
  T: Copy + Eq + Ord + std::hash::Hash,
  S: BuildHasher,
  R: RngCore,
>(
  g: &mut DenseKNNGraph<T, S>,
  q: T,
  prng: &mut R,
) -> () {
  //TODO: return the index of the new data point
  let SearchResults {
    nearest_neighbors_max_dist_heap,
    visited_nodes,
    visited_node_distances_to_q,
  } = knn_beam_search(g, q, g.config.num_searchers as usize, prng);

  if g.config.use_rrnp {
    rrnp(
      g,
      &nearest_neighbors_max_dist_heap,
      &visited_nodes,
      &visited_node_distances_to_q,
    );
  } else {
    let (neighbors, dists) = nearest_neighbors_max_dist_heap
      .iter()
      .map(|sr| (g.mapping.ext_to_int(&sr.id), sr.dist))
      .unzip();
    let q_int = g.mapping.insert(q);
    g.insert_vertex(q_int, neighbors, dists);

    for r in &visited_nodes {
      g.insert_edge_if_closer(
        *g.mapping.ext_to_int(&r),
        q_int,
        visited_node_distances_to_q[&r],
      );
    }
  }

  if g.config.use_lgd {
    apply_lgd(
      g,
      &nearest_neighbors_max_dist_heap,
      &visited_nodes,
      &visited_node_distances_to_q,
    );
  }
}

// TODO: wrap db and dist_fn in a struct and make this a method on it.
// TODO: avoid passing db at all? We need to support incremental data.
// TODO: really need a solution here. The std::ops::Index thing doesn't work
// because it forces the user to return references to objects, which the
// user may not have (what if the objects are ephemeral and the user wants to
// avoid allocating them? What if you can compute distance as the object is
// streamed over the network, so you never materialize the whole thing in
// memory?) Requirements:
// - Don't require the objects to be stored in memory. The user decides how to
// fetch them to compute distance on them. Read from disk, network call, memory,
// whatever the user wants.
// - Don't force user to make a mapping from 0..n to the objects. The user's
// objects might be identified by UUIDs, strings, whatever, on the user's side.
// However, for memory efficiency, allow user to opt into using u32 ids.
// - TODO: Make sure that this all works with deletion! Before deletion, we were
//   assuming that our internal u32 ids pointed to the same object forever.
//   Now we are assuming it could change, so if we pass the internal id to the
//   user's callback, the user needs to make sure that they can handle it. BUT
//   the catch is that *our code* will be assigning the internal id to the
//   user's object. So we need to return the assigned id to the user, and the
//   user needs to keep track of the internal id -> object mapping on their
//   side. This is really tricky for the user and a potential footgun, so we
//   desperately need to expose a utility that handles this for the user.
//   TODO: make sure that the API's use of internal ids makes sense given
//   deletion. Some parts assume the user is assigning internal ids, other parts
//   assume the library is. This is very very bad.
//
// Maybe the right solution is to generalize the graph type itself instead of
// being hardcoded to u32s.
//
// Possible solution:
// - Create an "easy mode" wrapper around DenseKNNGraph that allows the user to
// insert any type they want, and we maintain the mapping from that type to u32.
// As with any container, whether the user stores IDs or objects themselves is
// none of our business as the implementor of the container. We can defer this
// one until later with no problems.
// - Eliminate the db type. If the user needs to carry around more info, their
// distance fn can be a closure. "Easy mode" wrapper would have the user create
// a distance function that takes their type as input, we do the mapping to/from
// u32 behind-the-scenes.
// - For exhaustive_knn_graph, the contract is that dist_fn will be called with
// indices 0..n so dist_fn must be able to give answers over that range. Again,
// an easy mode wrapper can hide this from the user by mapping their stuff to
// 0..n.

#[cfg(test)]
mod tests {
  use std::hash::BuildHasherDefault;

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
    dist_fn: &'a dyn Fn(&T, &T) -> f32,
  ) -> KNNGraphConfig<'a, T> {
    let out_degree = 5;
    let num_searchers = 5;
    let use_rrnp = false;
    let use_lgd = false;
    KNNGraphConfig::<'a, T> {
      capacity,
      out_degree,
      num_searchers,
      dist_fn,
      use_rrnp,
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
    let g: DenseKNNGraph<u32, NHH> = exhaustive_knn_graph(
      vec![&0u32, &1, &2, &3, &4, &5],
      config,
      nohash_hasher::BuildNoHashHasher::default(),
    );
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
    let g: DenseKNNGraph<u32, NHH> = exhaustive_knn_graph(
      vec![&0, &1, &2, &3, &4, &5],
      config,
      nohash_hasher::BuildNoHashHasher::default(),
    );
    g.consistency_check();
    let mut prng = Xoshiro256StarStar::seed_from_u64(1);
    let q = [1.2f32];
    let SearchResults {
      nearest_neighbors_max_dist_heap,
      ..
    } = knn_beam_search(&g, 1, 2, &mut prng);
    assert_eq!(
      nearest_neighbors_max_dist_heap
        .iter()
        .map(|x| (x.id, x.dist))
        .collect::<Vec<(u32, f32)>>(),
      vec![(0, 1.0), (1u32, 0.0)]
    );
  }

  #[test]
  fn test_insert_vertex() {
    let mut config = mk_config(3, &|x, y| 1.0);
    config.out_degree = 2;
    let mut g: DenseKNNGraph<u32, BuildHasherDefault<NoHashHasher<u32>>> =
      DenseKNNGraph::empty(config, nohash_hasher::BuildNoHashHasher::default());
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
    let mut g: DenseKNNGraph<u32, NHH> = DenseKNNGraph::empty(
      mk_config(2, &|x, y| 1.0),
      nohash_hasher::BuildNoHashHasher::default(),
    );
    g.insert_vertex(0, vec![2, 1], vec![2.0, 1.0]);
    g.insert_vertex(1, vec![2, 0], vec![1.0, 1.0]);
    g.insert_vertex(2, vec![0, 1], vec![2.0, 1.0]);
  }

  #[test]
  #[should_panic]
  fn test_insert_vertex_panic_wrong_neighbor_length() {
    let mut g: DenseKNNGraph<u32, NHH> = DenseKNNGraph::empty(
      mk_config(2, &|x, y| 1.0),
      nohash_hasher::BuildNoHashHasher::default(),
    );
    g.insert_vertex(0, vec![2, 1, 0], vec![2.0, 1.0, 10.1]);
    g.insert_vertex(1, vec![2, 0], vec![1.0, 1.0]);
  }

  #[test]
  fn test_insert_edge_if_closer() {
    let mut config = mk_config(3, &|&x, &y| 1.0);
    config.out_degree = 1;
    let mut g: DenseKNNGraph<u32, NHH> =
      DenseKNNGraph::empty(config, nohash_hasher::BuildNoHashHasher::default());
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
  fn test_insert_approx() {
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
      nohash_hasher::BuildNoHashHasher::default(),
    );
    let mut prng = Xoshiro256StarStar::seed_from_u64(1);
    for i in 6..11 {
      println!("doing {}", i);
      insert_approx(&mut g, i, &mut prng);
    }
    g.consistency_check();
  }

  // #[test]
  // fn test_insert_approx_from_empty() {
  //   let db = vec![
  //     [1f32],
  //     [2f32],
  //     [3f32],
  //     [10f32],
  //     [11f32],
  //     [12f32],
  //     [18f32],
  //     [19f32],
  //     [20f32],
  //     [21f32],
  //     [22f32],
  //   ];
  //   let dist_fn = &|x: &u32, y: &u32| {
  //     sq_euclidean_faster(&db[*x as usize], &db[*y as usize])
  //   };
  //   let mut g: DenseKNNGraph<u32, NHH> = DenseKNNGraph::empty(
  //     mk_config(10, dist_fn),
  //     nohash_hasher::BuildNoHashHasher::default(),
  //   );
  //   let mut prng = Xoshiro256StarStar::seed_from_u64(1);
  //   insert_approx(&mut g, 0, &mut prng);

  //   g.consistency_check();
  // }
}
