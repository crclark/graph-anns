use nohash_hasher::IntMap;
use rand::distributions::{Distribution, Uniform};
use rand::RngCore;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::cmp::Reverse;
use std::collections::binary_heap::BinaryHeap;
use std::sync::Arc;
use std::time::Instant;
use tinyset::SetU32;
use Ordering;

// NOTE: we need to be able to insert nodes. That means we need to be
// able to (virtually or actually) resize the vector of edges, AND keep track of
// the number of vertices in the graph in total. That requires coordinated
// updates to multiple fields in the struct, which means we might as well remove
// all the internal atomics/mutexes and forget about fine-grained parallel
// access to this data structure. Instead, just wrap the whole thing in a
// read/write lock. Revisit if Rust ever gains an STM library.

/// A directed graph stored contiguously in memory as an adjacency list.
/// All vertices are guaranteed to have the same out-degree.
/// Nodes are u32s numbered from 0 to n-1.
pub struct DenseKNNGraph {
  /// n, the number of vertices in the graph. The valid indices of the graph
  /// are 0 to n-1.
  pub num_vertices: u32,
  /// The max number of vertices that can be inserted into the graph. Constant.
  pub capacity: u32,
  /// The number of neighbors of each vertex. This is a constant.
  pub out_degree: u32,
  /// The underlying buffer of n*num_vertices AtomicU32s. Use with caution.
  /// Prefer to use indexing to access the neighbors of a vertex.
  /// `g[i]` returns a slice of length `out_degree` of the neighbors of `i` along
  /// with their distances from i.
  pub edges: Vec<u32>,
  pub edge_distances: Vec<f32>,
  /// Maintains an association between vertices and the vertices that link out to
  /// them. In other words, each backpointers[i] is the set of vertices S s.t.
  /// for all x in S, a directed edge exists pointing from x to i.
  pub backpointers: Vec<SetU32>,
}

impl DenseKNNGraph {
  /// Allocates a graph of the specified size and out_degree, but
  /// doesn't populate the edges.
  fn empty(capacity: u32, out_degree: u32) -> DenseKNNGraph {
    let edges = Vec::with_capacity(capacity as usize * out_degree as usize);
    let edge_distances = Vec::with_capacity(capacity as usize * out_degree as usize);

    let mut backpointers = Vec::with_capacity(capacity as usize);

    for u in 0..capacity {
      backpointers.push(SetU32::new());
    }

    let num_vertices = 0;

    DenseKNNGraph {
      num_vertices,
      capacity,
      out_degree,
      edges,
      edge_distances,
      backpointers,
    }
  }

  /// Get the neighbors of u and their distances. Panics if index
  /// >= num_vertices.
  pub fn get_edges(&self, index: u32) -> (&[u32], &[f32]) {
    assert!(index < self.num_vertices);

    let i = index * self.out_degree;
    let j = i + self.out_degree;
    (
      &self.edges[i as usize..j as usize],
      &self.edge_distances[i as usize..j as usize],
    )
  }

  /// Get the neighbors of u and their distances. Panics if index >= num_vertices.
  fn get_edges_mut(&mut self, index: u32) -> (&mut [u32], &mut [f32]) {
    assert!(index < self.num_vertices);

    let i = index * self.out_degree;
    let j = i + self.out_degree;
    (
      &mut self.edges[i as usize..j as usize],
      &mut self.edge_distances[i as usize..j as usize],
    )
  }

  /// Creates an edge from `from` to `to` if the distance `dist` between them is
  /// less than the distance from `from` to one of its existing neighbors `u`.
  /// If so, removes the edge (`from`, `u`).
  ///
  /// `to` must have already been added to the graph by insert_vertex etc.
  ///
  /// Returns `true` if the new edge was added.
  fn insert_edge_if_closer(&mut self, from: u32, to: u32, dist: f32) -> bool {
    let (mut nbrs, mut dists) = self.get_edges_mut(from);

    for (nbr, nbr_dist) in nbrs.iter_mut().zip(dists) {
      if dist < *nbr_dist {
        *nbr = to;
        *nbr_dist = dist;
        return true;
      }
    }

    return false;
  }

  /// Insert a new vertex into the graph, given its k neighbors and their
  /// distances. Panics if num_vertices == capacity.
  fn insert_vertex(&mut self, u: u32, nbrs: Vec<u32>, dists: Vec<f32>) {
    assert!(self.num_vertices < self.capacity);

    let od = self.out_degree as usize;
    assert!(nbrs.len() == od && dists.len() == od);

    for (nbr, dist) in nbrs.iter().zip(dists) {
      self.edges.push(*nbr);
      self.edge_distances.push(dist);
      let s = &mut self.backpointers[*nbr as usize];
      s.insert(u);
    }

    self.num_vertices += 1;
  }

  /// Panics if graph is internally inconsistent.
  fn consistency_check(&self) {
    if self.edges.len() != self.edge_distances.len() {
      panic!("edges and edge_distances are inconsistent lengths");
    }

    if self.edges.len() != self.num_vertices as usize * self.out_degree as usize {
      panic!(
        "edges.len() is not equal to num_vertices * out_degree. {} != {} * {}",
        self.edges.len(),
        self.num_vertices,
        self.out_degree
      );
    }

    for i in 0..self.num_vertices {
      let (nbrs, _) = self.get_edges(i);
      for nbr in nbrs.iter() {
        if *nbr == i {
          panic!("Self loop at node {}", i);
        }
        let nbr_backptrs = &self.backpointers[*nbr as usize];
        if !nbr_backptrs.contains(i) {
          panic!("backpointer missing for vertex {} on nbr {}", i, nbr);
        }
      }
    }
  }
}

#[derive(Debug, Clone, Copy)]
struct SearchResult {
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

/// Perform beam search for the k nearest neighbors of a point q. Returns
/// (nearest_neighbors_max_dist_heap, visited_nodes, visited_node_distances_to_q)
/// This is a translation of the pseudocode of algorithm 1 from
/// Approximate k-NN Graph Construction: A Generic Online Approach
fn knn_beam_search<R: RngCore>(
  g: DenseKNNGraph,
  dist_to_q: impl Fn(u32) -> f32,
  num_searchers: usize,
  k: usize,
  prng: &mut R,
) -> (BinaryHeap<SearchResult>, SetU32, IntMap<u32, f32>) {
  let mut q_max_heap: BinaryHeap<SearchResult> = BinaryHeap::new();
  let mut r_min_heap: BinaryHeap<Reverse<SearchResult>> = BinaryHeap::new();
  let mut visited = SetU32::new();
  let mut visited_distances: IntMap<u32, f32> = IntMap::default();

  let rand_vertex = Uniform::from(0..g.num_vertices);

  // lines 2 to 10 of the pseudocode
  for i in 0..num_searchers {
    let r = rand_vertex.sample(prng);
    let r_dist = dist_to_q(r);
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

    let (r_out, _) = g.get_edges(r.vec_index);
    let mut r_nbrs = g.backpointers[r.vec_index as usize].clone();
    for nbr in r_out.iter() {
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
fn exhaustive_knn_graph<T: ?Sized, C: std::ops::Index<usize, Output = T>>(
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
      g.edges.push(vec_index);
      g.edge_distances.push(dist);
      let s = &mut g.backpointers[vec_index as usize];
      s.insert(i);
    }
  }

  g.num_vertices = n;

  g
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

    for nbr_ix in 0..out_degree {
      let v = rand_vertex.sample(prng);
      let distance = get_dist(u, v, db, dist_fn);
      g.edges.push(v);
      g.edge_distances.push(distance);
      let s = &mut g.backpointers[v as usize];
      s.insert(u);
    }

    if u % 1000_000 == 0 {
      println!("finished u = {:?}, elapsed = {:?}", u, start_loop.elapsed());
    }
  }
  g
}

// TODO: consider doing this when we want to overlay the permutation-based graph
// with the K-NN graph. It is trivial to implement Fits64 for
// (AtomicU32, AtomicF32). It will allow us to implement variable out-degree
// per vertex while keeping memory consumption in check. Oops, wait, would this
// actually work? Would it be creating new atomics each time? Instead, if we
// need variable out-degree, let's reserve u32::max as the empty value.

// pub struct Foo {
//   pub bar: Set64<(AtomicU32, AtomicF32)>,
// }

// Performs the NN-descent algorithm on a subset of the vertices of the graph.
// Early stopping occurs when the number of successful updates in an iteration
// is less than `delta*k*(j-i)`.
fn nn_descent_thread<R: RngCore>(
  // Lower bound (inclusive) node that this thread is responsible for
  i: u32,
  // Upper bound (exclusive) node that this thread is responsible for
  j: u32,
  // Early stopping parameter. Suggested default from the paper: 0.001.
  delta: f64,
  g: Arc<DenseKNNGraph>,
  prng: R,
) -> () {
  unimplemented!()
}

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

// A parallel implementation of the NN-Descent algorithm from the paper
// "Efficient K-Nearest Neighbor Graph Construction for Generic Similarity
// Measures" by Dong et al.
// https://dl.acm.org/doi/10.1145/1963405.1963487
//
// This has some differences from the version presented in the paper:
// - it is parallelized by splitting the set of vertices into equally-sized
//   partitions and making each thread responsible for a partition. There are
//   some small race conditions in this implementation, but they only cause
//   the algorithm to sometimes see an outdated set of neighbors for a vertex,
//   which won't impact convergence significantly.
// - We don't do the "local join" trick described in Section 2.3. There's no
//   reason why we couldn't, but it's extra code complexity that doesn't yet
//   seem to be warranted.
pub fn nn_descent_parallel<'a, T: ?Sized, C: std::ops::Index<usize, Output = T>>(
  num_threads: usize,
  k: usize,
  db: &C,
  // TODO: parametrize the type of the distances so we can use much faster
  // i32 if possible.
  dist_fn: fn(&T, &T) -> f32,
) -> DenseKNNGraph {
  unimplemented!()
}

#[cfg(test)]
mod tests {
  use super::*;
  use texmex::sq_euclidean_faster;

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
    assert_eq!(g.get_edges(0).0, vec![2, 1]);
    assert_eq!(g.get_edges(1).0, vec![2, 0]);
    assert_eq!(g.get_edges(2).0, vec![0, 1]);
    assert_eq!(g.get_edges(3).0, vec![5, 4]);
    assert_eq!(g.get_edges(4).0, vec![3, 5]);
    assert_eq!(g.get_edges(5).0, vec![3, 4]);
  }

  #[test]
  fn test_beam_search_fully_connected_graph() {
    let db = vec![[1f32], [2f32], [3f32], [10f32], [11f32], [12f32]];
    let g = exhaustive_knn_graph(6, 10, 5, &db, |&x, &y| sq_euclidean_faster(&x, &y));
    g.consistency_check();
    let mut prng = Xoshiro256StarStar::seed_from_u64(1);
    let q = [1.2f32];
    let (nearest, _, _) = knn_beam_search(
      g,
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
    g.insert_vertex(0, vec![2,1], vec![2.0, 1.0]);
    g.insert_vertex(1, vec![2,0], vec![1.0, 1.0]);
    g.insert_vertex(2, vec![0,1], vec![2.0, 1.0]);

    assert_eq!(g.get_edges(0), ([2, 1].as_slice(), [2.0, 1.0].as_slice()));
    assert_eq!(g.get_edges(1), ([2, 0].as_slice(), [1.0, 1.0].as_slice()));
    assert_eq!(g.get_edges(2), ([0, 1].as_slice(), [2.0, 1.0].as_slice()));
  }

  #[test]
  #[should_panic]
  fn test_insert_vertex_panic_too_many_vertex() {
    let mut g = DenseKNNGraph::empty(2, 2);
    g.insert_vertex(0, vec![2,1], vec![2.0, 1.0]);
    g.insert_vertex(1, vec![2,0], vec![1.0, 1.0]);
    g.insert_vertex(2, vec![0,1], vec![2.0, 1.0]);
  }

  #[test]
  #[should_panic]
  fn test_insert_vertex_panic_wrong_neighbor_length() {
    let mut g = DenseKNNGraph::empty(2, 2);
    g.insert_vertex(0, vec![2,1,0], vec![2.0, 1.0, 10.1]);
    g.insert_vertex(1, vec![2,0], vec![1.0, 1.0]);
  }

  #[test]
  fn test_insert_edge_if_closer() {
    let mut g = DenseKNNGraph::empty(3, 1);
    g.insert_vertex(0, vec![2], vec![2.0]);
    g.insert_vertex(1, vec![2], vec![1.0]);
    g.insert_vertex(2, vec![1], vec![1.0]);

    assert_eq!(g.get_edges(0), ([2].as_slice(), [2.0].as_slice()));

    assert!(g.insert_edge_if_closer(0, 1, 1.0));

    assert_eq!(g.get_edges(0), ([1].as_slice(), [1.0].as_slice()));
    assert_eq!(g.get_edges(1), ([2].as_slice(), [1.0].as_slice()));
    assert_eq!(g.get_edges(2), ([1].as_slice(), [1.0].as_slice()));
  }
}
