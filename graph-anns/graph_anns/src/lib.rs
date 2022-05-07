#![feature(total_cmp)]
#![feature(is_sorted)]
extern crate nohash_hasher;
extern crate rand;
extern crate rand_xoshiro;
extern crate tinyset;

use nohash_hasher::IntMap;
use rand::distributions::{Distribution, Uniform};
use rand::seq::index::sample;
use rand::RngCore;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::binary_heap::BinaryHeap;
use std::time::Instant;
use tinyset::SetU32;

const DEFAULT_LAMBDA: u8 = 0;

// TODO: revisit every use of `pub` in this file.

// TODO: many benchmarking utilities. Recall@n  metrics and so forth.

// TODO: benchmark with random data that has lazily-generated distances/vectors.
// We could use random integers and the bitwise hamming distance between them
// for speed.

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
//    that point to the deleted node.
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

/// A directed graph stored contiguously in memory as an adjacency list.
/// All vertices are guaranteed to have the same out-degree.
/// Nodes are u32s numbered from 0 to n-1.
#[derive(Debug)]
pub struct DenseKNNGraph {
  /// n, the number of vertices in the graph. The valid indices of the graph
  /// are 0 to n-1.
  pub num_vertices: u32,
  /// The max number of vertices that can be inserted into the graph. Constant.
  pub capacity: u32,
  /// The number of neighbors of each vertex. This is a constant.
  pub out_degree: u32,
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
  /// Whether to use restricted recursive neighborhood propagation. This improves
  /// search speed, but decreases insertion throughput. TODO: verify that's
  /// true.
  use_rrnp: bool,
  /// Whether to use lazy graph diversification. This improves search speed.
  /// TODO: parametrize this type so that the LGD vector is never allocated/
  /// takes no memory if this is set to false.
  use_lgd: bool,
}

impl DenseKNNGraph {
  /// Allocates a graph of the specified size and out_degree, but
  /// doesn't populate the edges.
  fn empty(capacity: u32, out_degree: u32) -> DenseKNNGraph {
    let edges = Vec::with_capacity(capacity as usize * out_degree as usize);

    let mut backpointers = Vec::with_capacity(capacity as usize);

    for _ in 0..capacity {
      backpointers.push(SetU32::new());
    }

    let num_vertices = 0;

    // TODO: expose as params once supported.
    let use_rrnp = false;
    let use_lgd = false;

    DenseKNNGraph {
      num_vertices,
      capacity,
      out_degree,
      edges,
      backpointers,
      use_rrnp,
      use_lgd,
    }
  }

  /// Get the neighbors of u and their distances. Panics if index
  /// >= num_vertices.
  pub fn get_edges(&self, index: u32) -> &[(u32, f32, u8)] {
    assert!(index < self.num_vertices);

    let i = index * self.out_degree;
    let j = i + self.out_degree;
    &self.edges[i as usize..j as usize]
  }

  fn debug_get_neighbor_indices(&self, index: u32) -> Vec<u32> {
    self.get_edges(index).iter().map(|e| e.0).collect()
  }

  /// Get the neighbors of u and their distances. Panics if index >= num_vertices.
  fn get_edges_mut(&mut self, index: u32) -> &mut [(u32, f32, u8)] {
    assert!(index < self.num_vertices);

    let i = index * self.out_degree;
    let j = i + self.out_degree;
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
    let most_distant_ix = (self.out_degree - 1) as usize;
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

  /// Insert a new vertex into the graph, given its k neighbors and their
  /// distances. Panics if the graph is already full (num_vertices == capacity).
  /// nbrs and dists must be equal to the out_degree of the graph.
  fn insert_vertex(&mut self, u: u32, nbrs: Vec<u32>, dists: Vec<f32>) {
    assert!(self.num_vertices < self.capacity);

    let od = self.out_degree as usize;
    assert!(nbrs.len() == od && dists.len() == od);

    for (nbr, dist) in nbrs.iter().zip(dists) {
      println!("Connecting {} to {}", u, nbr);
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
    if self.edges.len() != self.num_vertices as usize * self.out_degree as usize {
      panic!(
        "edges.len() is not equal to num_vertices * out_degree. {} != {} * {}",
        self.edges.len(),
        self.num_vertices,
        self.out_degree
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

#[derive(Debug, Clone, Copy)]
pub struct SearchResult {
  pub vec_index: u32,
  pub dist: f32,
}

impl SearchResult {
  pub fn new(vec_index: u32, dist: f32) -> SearchResult {
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

// TODO: return fewer than expected results if num_searchers or k is
// higher than num_vertices.

/// Perform beam search for the k nearest neighbors of a point q. Returns
/// (nearest_neighbors_max_dist_heap, visited_nodes, visited_node_distances_to_q)
/// This is a translation of the pseudocode of algorithm 1 from
/// Approximate k-NN Graph Construction: A Generic Online Approach
pub fn knn_beam_search<R: RngCore>(
  g: &DenseKNNGraph,
  dist_to_q: impl Fn(u32) -> f32,
  num_searchers: usize,
  k: usize,
  prng: &mut R,
) -> (BinaryHeap<SearchResult>, SetU32, IntMap<u32, f32>) {
  assert!(num_searchers <= g.num_vertices as usize);
  assert!(k <= g.num_vertices as usize);
  let mut q_max_heap: BinaryHeap<SearchResult> = BinaryHeap::new();
  let mut r_min_heap: BinaryHeap<Reverse<SearchResult>> = BinaryHeap::new();
  let mut visited = SetU32::new();
  let mut visited_distances: IntMap<u32, f32> = IntMap::default();

  // lines 2 to 10 of the pseudocode
  for r in sample(prng, g.num_vertices as usize, num_searchers) {
    let r = r as u32;
    let r_dist = dist_to_q(r);
    visited.insert(r);
    visited_distances.insert(r, r_dist);
    r_min_heap.push(Reverse(SearchResult::new(r, r_dist)));
    match q_max_heap.peek() {
      None => {
        q_max_heap.push(SearchResult::new(r, r_dist));
        // NOTE: pseudocode has a bug: R.insert(r) at both line 2 and line 8
        // We are skipping it here since we did it above.
      }
      Some(f) => {
        let f_dist = dist_to_q(f.vec_index);
        if r_dist < f_dist || q_max_heap.len() < k {
          q_max_heap.push(SearchResult::new(r, r_dist));
        }
      }
    }
  }

  // lines 11 to 27 of the pseudocode
  while r_min_heap.len() > 0 {
    while q_max_heap.len() > k {
      q_max_heap.pop();
    }

    let Reverse(r) = r_min_heap.pop().unwrap();
    let &f = { q_max_heap.peek().unwrap() };
    if r.dist > f.dist {
      break;
    }

    let mut r_nbrs = g.backpointers[r.vec_index as usize].clone();
    for (nbr, _, _) in g.get_edges(r.vec_index).iter() {
      r_nbrs.insert(*nbr);
    }

    for e in r_nbrs.iter() {
      if !visited.contains(e) {
        visited.insert(e);
        let e_dist = dist_to_q(e);
        let f_dist = dist_to_q(f.vec_index);
        visited_distances.insert(e, e_dist);
        if e_dist < f_dist || q_max_heap.len() < k {
          q_max_heap.push(SearchResult::new(e, e_dist));
          r_min_heap.push(Reverse(SearchResult::new(e, e_dist)));
        }
      }
    }
  }

  (q_max_heap, visited, visited_distances)
}

/// Constructs an exact k-nn graph on the first n items in db. O(n^2).
/// `capacity` is max capacity of the returned graph (for future inserts).
/// Must be >= n.
pub fn exhaustive_knn_graph<T: ?Sized, C: std::ops::Index<usize, Output = T>>(
  n: u32,
  capacity: u32,
  k: u32,
  db: &C,
  dist_fn: fn(&T, &T) -> f32,
) -> DenseKNNGraph {
  if k >= n {
    panic!("k must be less than n");
  }

  let mut g = DenseKNNGraph::empty(capacity, k);

  for i in 0..n {
    let mut knn = BinaryHeap::new();
    for j in 0..n {
      if i == j {
        continue;
      }
      let dist = get_dist(i, j, db, dist_fn);
      knn.push(SearchResult::new(j, dist));

      while knn.len() > k as usize {
        knn.pop();
      }
    }
    while knn.len() > 0 {
      let SearchResult { vec_index, dist } = knn.pop().unwrap();
      g.edges.push((vec_index, dist, DEFAULT_LAMBDA));
      let s = &mut g.backpointers[vec_index as usize];
      s.insert(i);
    }
  }

  g.num_vertices = n;

  for i in 0..n {
    g.sort_edges(i)
  }

  g
}

fn rrnp(
  _g: &mut DenseKNNGraph,
  _nearest_neighbors_max_dist_heap: &BinaryHeap<SearchResult>,
  _visited_nodes: &SetU32,
  _visited_nodes_distance_to_q: &IntMap<u32, f32>,
) -> () {
  unimplemented!()
}

fn apply_lgd(
  _g: &mut DenseKNNGraph,
  _nearest_neighbors_max_dist_heap: &BinaryHeap<SearchResult>,
  _visited_nodes: &SetU32,
  _visited_nodes_distance_to_q: &IntMap<u32, f32>,
) -> () {
  unimplemented!()
}

/// Inserts a new data point into the graph. The graph must not be full.
/// Optionally performs restricted recursive neighborhood propagation and
/// lazy graph diversification. These options are set when the graph is
/// constructed.
///
/// This is equivalent to one iteration of Algorithm 3 in
/// "Approximate k-NN Graph Construction: A Generic Online Approach".
pub fn insert_approx<R: RngCore>(
  g: &mut DenseKNNGraph,
  q: u32,
  dist_to_q: impl Fn(u32) -> f32,
  num_searchers: usize,
  prng: &mut R,
) -> () {
  //TODO: return the index of the new data point
  let (nearest_neighbors_max_dist_heap, visited_nodes, visited_nodes_distance_to_q) =
    knn_beam_search(g, dist_to_q, num_searchers, g.out_degree as usize, prng);

  if g.use_rrnp {
    rrnp(
      g,
      &nearest_neighbors_max_dist_heap,
      &visited_nodes,
      &visited_nodes_distance_to_q,
    );
  } else {
    let (neighbors, dists) = nearest_neighbors_max_dist_heap
      .iter()
      .map(|sr| (sr.vec_index, sr.dist))
      .unzip();
    g.insert_vertex(q, neighbors, dists);

    //TODO: try to avoid clone
    for r in visited_nodes.clone() {
      g.insert_edge_if_closer(r, q, visited_nodes_distance_to_q[&r]);
    }
  }

  if g.use_lgd {
    apply_lgd(
      g,
      &nearest_neighbors_max_dist_heap,
      &visited_nodes,
      &visited_nodes_distance_to_q,
    );
  }
}

// TODO: wrap db and dist_fn in a struct and make this a method on it.
// TODO: avoid passing db at all? We need to support incremental data.
fn get_dist<T: ?Sized, C: std::ops::Index<usize, Output = T>>(
  i: u32,
  j: u32,
  db: &C,
  dist_fn: fn(&T, &T) -> f32,
) -> f32 {
  dist_fn(&db[i as usize], &db[j as usize])
}

/// Computes a sliding window of num_vertices/num_partitions vertices around i.
/// Examples:
///
/// ```
/// use graph_anns::chunk_range;
/// assert_eq!(chunk_range(10, 2, 0), (0,5));
/// ```
pub fn chunk_range(num_vertices: u32, num_partitions: usize, i: u32) -> (u32, u32) {
  let w = num_vertices / (num_partitions as u32);
  let r = w / 2;
  let rem = w % 2;

  if i < r {
    (0, w)
  } else if i > num_vertices - r {
    (num_vertices - w, num_vertices)
  } else if i > r + rem {
    (i - r - rem, i + r)
  } else {
    (i - r, i + r + rem)
  }
}

// TODO: not pub
/// Randomly initializes a K-NN Graph. This graph can then be optimized with
/// nn-descent.
pub fn random_init<R: RngCore, T: ?Sized, C: std::ops::Index<usize, Output = T>>(
  num_vertices: u32,
  out_degree: u32,
  prng: &mut R,
  db: &C,
  dist_fn: fn(&T, &T) -> f32,
  // An optimization for large datasets to speed up initialization, probably at
  // the expense of slower convergence. The dataset is split into num_partitions
  // chunks, and nodes in each chunk will only be connected to other nodes in
  // the same chunk. This improves cache locality -- only 1/num_partitions of
  // the dataset will need to be in active use at any time during initialization.
  num_partitions: usize,
) -> DenseKNNGraph {
  if num_vertices == u32::max_value() {
    panic!("Max number of vertices is u32::max_value() - 1")
  }

  let start = Instant::now();

  let mut g = DenseKNNGraph::empty(num_vertices, out_degree);

  println!(
    "Allocated vecs and created mutexes in {:?}",
    start.elapsed()
  );

  let start_loop = Instant::now();
  for u in 0..num_vertices {
    let (ix_range_low, ix_range_high) = chunk_range(num_vertices, num_partitions, u);
    let rand_vertex = Uniform::from(ix_range_low..ix_range_high);

    for _ in 0..out_degree {
      let v = rand_vertex.sample(prng);
      let distance = get_dist(u, v, db, dist_fn);
      g.edges.push((v, distance, DEFAULT_LAMBDA));
      let s = &mut g.backpointers[v as usize];
      s.insert(u);
    }

    if u % 1000_000 == 0 {
      println!("finished u = {:?}, elapsed = {:?}", u, start_loop.elapsed());
    }
  }
  g
}

// Performs the NN-descent algorithm on a subset of the vertices of the graph.
// Early stopping occurs when the number of successful updates in an iteration
// is less than `delta*k*(j-i)`.
// fn nn_descent_thread<R: RngCore>(
//   // Lower bound (inclusive) node that this thread is responsible for
//   i: u32,
//   // Upper bound (exclusive) node that this thread is responsible for
//   j: u32,
//   // Early stopping parameter. Suggested default from the paper: 0.001.
//   delta: f64,
//   g: Arc<DenseKNNGraph>,
//   prng: R,
// ) -> () {
//   unimplemented!()
// }

// TODO: use the triangle inequality to short-circuit the comparisons. Say we have
// a -> b -> c and we are working on a. We have already stored d(a,b) and d(b,c).
// If d(a,b) + d(b,c) is less than another neighbor of a, x, we can replace x
// with c without ever needing to compute d(a,c)...
// Oops, except we need to record d(a,c) in the list of a's neighbors so that we
// can do future iterations... hmm. Or do we? Is storing an upper bound on the
// distance enough? If we only know upper_bound_dist(a,b) and upper_bound_dist(b,c),
// we can say that we have an upper bound on d(a,c) by adding them, but it could
// be a rather bad upper bound. And our upper bounds would be getting looser
// each iteration, as we keep adding upper bounds together again and again.
// OTOH, if we only computed d(a,c) if the upper bound on its distance is less
// than a known distance, we would still be winning...
// ..
// To summarize:
// Use d(a,b) + d(b,c) as an approximation of d(a,c). If approx_d(a,c) < d(a,x) for
// some x, replace x with c and compute d(a,c). If approx_d(a,c) > d(a,x) for
// all x, compute d(a,c) and iterate again. That's a dumb strategy, because
// either way, you are computing d(a,c). The only difference is that you now
// have a worst case where you do 2*k iterations through the list of a's
// neighbors. That's bad.
//
// Another idea: store Either TrueDist UpperBoundDist and be lazy -- compute
// TrueDist only when UpperBoundDist is not precise enough to know whether
// candidate is closer.

#[cfg(test)]
mod tests {
  use super::*;
  use rand::SeedableRng;
  use rand_xoshiro::Xoshiro256StarStar;

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

  pub fn sq_euclidean_faster<T: PrimitiveToF32 + Copy>(v1: &[T], v2: &[T]) -> f32 {
    let mut result = 0.0;
    let n = v1.len();
    for i in 0..n {
      let diff = v2[i].tof32() - v1[i].tof32();
      result += diff * diff;
    }
    result
  }

  #[test]
  fn test_chunk_range() {
    assert_eq!(chunk_range(10, 2, 0), (0, 5));
    assert_eq!(chunk_range(10, 2, 1), (0, 5));
    assert_eq!(chunk_range(10, 2, 2), (0, 5));
    assert_eq!(chunk_range(10, 2, 3), (1, 6));
    assert_eq!(chunk_range(10, 2, 4), (1, 6));
    assert_eq!(chunk_range(10, 2, 5), (2, 7));
    assert_eq!(chunk_range(10, 2, 6), (3, 8));
    assert_eq!(chunk_range(10, 2, 7), (4, 9));
    assert_eq!(chunk_range(10, 2, 8), (5, 10));
    assert_eq!(chunk_range(10, 2, 9), (5, 10));

    assert_eq!(chunk_range(11, 2, 0), (0, 5));
    assert_eq!(chunk_range(11, 2, 1), (0, 5));
    assert_eq!(chunk_range(11, 2, 2), (0, 5));
    assert_eq!(chunk_range(11, 2, 3), (1, 6));
    assert_eq!(chunk_range(11, 2, 4), (1, 6));
    assert_eq!(chunk_range(11, 2, 5), (2, 7));
    assert_eq!(chunk_range(11, 2, 6), (3, 8));
    assert_eq!(chunk_range(11, 2, 7), (4, 9));
    assert_eq!(chunk_range(11, 2, 8), (5, 10));
    assert_eq!(chunk_range(11, 2, 9), (6, 11));

    assert_eq!(chunk_range(10, 3, 0), (0, 3));
    assert_eq!(chunk_range(10, 3, 1), (0, 3));
    assert_eq!(chunk_range(10, 3, 2), (1, 4));
    assert_eq!(chunk_range(10, 3, 3), (1, 4));
    assert_eq!(chunk_range(10, 3, 4), (2, 5));
    assert_eq!(chunk_range(10, 3, 5), (3, 6));
    assert_eq!(chunk_range(10, 3, 6), (4, 7));
    assert_eq!(chunk_range(10, 3, 7), (5, 8));
    assert_eq!(chunk_range(10, 3, 8), (6, 9));
    assert_eq!(chunk_range(10, 3, 9), (7, 10));

    assert_eq!(chunk_range(10, 1, 0), (0, 10));
    assert_eq!(chunk_range(10, 1, 1), (0, 10));
    assert_eq!(chunk_range(10, 1, 2), (0, 10));
    assert_eq!(chunk_range(10, 1, 3), (0, 10));
    assert_eq!(chunk_range(10, 1, 4), (0, 10));
    assert_eq!(chunk_range(10, 1, 5), (0, 10));
    assert_eq!(chunk_range(10, 1, 6), (0, 10));
    assert_eq!(chunk_range(10, 1, 7), (0, 10));
    assert_eq!(chunk_range(10, 1, 8), (0, 10));
    assert_eq!(chunk_range(10, 1, 9), (0, 10));
  }

  #[test]
  fn test_exhaustive_knn_graph() {
    let db = vec![[1], [2], [3], [10], [11], [12]];
    let g = exhaustive_knn_graph(6, 10, 2, &db, |&x, &y| sq_euclidean_faster(&x, &y));
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
    let g = exhaustive_knn_graph(6, 10, 5, &db, |&x, &y| sq_euclidean_faster(&x, &y));
    g.consistency_check();
    let mut prng = Xoshiro256StarStar::seed_from_u64(1);
    let q = [1.2f32];
    let (nearest, _, _) = knn_beam_search(
      &g,
      |i| sq_euclidean_faster(&db[i as usize], &q),
      1,
      2,
      &mut prng,
    );
    assert_eq!(
      nearest.iter().map(|x| x.vec_index).collect::<Vec<u32>>(),
      vec![1u32, 0]
    );
  }

  #[test]
  fn test_insert_vertex() {
    let mut g = DenseKNNGraph::empty(3, 2);
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
    let mut g = DenseKNNGraph::empty(2, 2);
    g.insert_vertex(0, vec![2, 1], vec![2.0, 1.0]);
    g.insert_vertex(1, vec![2, 0], vec![1.0, 1.0]);
    g.insert_vertex(2, vec![0, 1], vec![2.0, 1.0]);
  }

  #[test]
  #[should_panic]
  fn test_insert_vertex_panic_wrong_neighbor_length() {
    let mut g = DenseKNNGraph::empty(2, 2);
    g.insert_vertex(0, vec![2, 1, 0], vec![2.0, 1.0, 10.1]);
    g.insert_vertex(1, vec![2, 0], vec![1.0, 1.0]);
  }

  #[test]
  fn test_insert_edge_if_closer() {
    let mut g = DenseKNNGraph::empty(3, 1);
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
    let mut g = exhaustive_knn_graph(6, 11, 5, &db, |&x, &y| sq_euclidean_faster(&x, &y));
    let mut prng = Xoshiro256StarStar::seed_from_u64(1);
    for i in 6..11 {
      insert_approx(
        &mut g,
        i,
        |j| sq_euclidean_faster(&db[i as usize], &db[j as usize]),
        5,
        &mut prng,
      );
    }
    g.consistency_check();
  }
}
