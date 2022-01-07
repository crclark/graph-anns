use atomic_float::AtomicF32;
use parking_lot::Mutex;
use std::sync::atomic::AtomicU32;
use tinyset::{Set64, SetU32};
use rand::{RngCore};
use rand::distributions::{Distribution, Uniform};
use std::sync::Arc;
use std::time::Instant;

// A directed graph stored contiguously in memory as an adjacency list.
// All vertices are guaranteed to have the same out-degree.
// Nodes are u32s numbered from 0 to n-1. Each element of each node's adjacency
// list is an AtomicU32, to enable the implementation of parallelized
// heuristic algorithms.
pub struct DenseKNNGraph {
  // n, the number of vertices in the graph. The valid indices of the graph
  // are 0 to n-1.
  pub num_vertices: u32,
  // The number of neighbors of each vertex. This is a constant.
  pub out_degree: u32,
  // The underlying buffer of n*num_vertices AtomicU32s. Use with caution.
  // Prefer to use indexing to access the neighbors of a vertex.
  // `g[i]` returns a slice of length `out_degree` of the neighbors of `i` along
  // with their distances from i.
  //
  // TODO: split this into two separate vecs so that we can discard the
  // distances when we move to the permutation phase.
  pub edges: Vec<(AtomicU32, AtomicF32)>,
}

impl std::ops::Index<u32> for DenseKNNGraph {
  type Output = [(AtomicU32, AtomicF32)];

  fn index(&self, index: u32) -> &[(AtomicU32, AtomicF32)] {
    let i = index * self.out_degree;
    let j = i + self.out_degree;
    &self.edges[i as usize..j as usize]
  }
}

// Maintains an association between vertices and the vertices that link out to
// them. In other words, each backpointers[i] is the set of vertices S s.t.
// for all x in S, a directed edge exists pointing from x to i.
//
// This is an internal detail of NN-descent and
// doesn't need to be returned to the caller.
pub struct DenseKNNGraphBackpointers {
  pub backpointers: Vec<Mutex<SetU32>>,
}

// TODO: wrap db and dist_fn in a struct and make this a method on it.
fn get_dist<T: ?Sized, C: std::ops::Index<usize, Output = T>>(
  i: u32,
  j: u32,
  db: &C,
  dist_fn: fn(&T, &T) -> f32
) -> f32 {
  dist_fn(&db[i as usize], &db[j as usize])
}

// TODO: not pub
pub fn random_init<R : RngCore, T: ?Sized, C: std::ops::Index<usize, Output = T>>(
  num_vertices: u32,
  out_degree: u32,
  prng: &mut R,
  db: &C,
  dist_fn: fn(&T, &T) -> f32) -> (DenseKNNGraph, DenseKNNGraphBackpointers) {
  if num_vertices == u32::max_value() {
    panic!("Max number of vertices is u32::max_value() - 1")
  }

  let start = Instant::now();
  let mut edges = Vec::with_capacity(num_vertices as usize * out_degree as usize);

  let mut g = DenseKNNGraph{num_vertices, out_degree, edges};

  let mut backpointers = Vec::with_capacity(num_vertices as usize);

  for u in 0..num_vertices {
    backpointers.push(Mutex::new(SetU32::new()));
  }
  let mut bp = DenseKNNGraphBackpointers{backpointers};

  let rand_vertex = Uniform::from(0..num_vertices);

  println!("Allocated vecs and created mutexes in {:?}", start.elapsed());

  // TODO: this becomes slow with the 120GiB dataset because we don't have
  // enough memory to cache the whole thing. We end up taking about 70 seconds
  // per million nodes initialized vs 3 seconds per million with the 12GiB file.
  // IDEA: dramatically improve cache hits by limiting rand_vertex distribution
  // to return nodes from a limited contiguous range of the dataset. Say we
  // split it into 3 chunks.
  // Then we have divs = [333_000_000, 666_000_000, 1_000_000_000]. Find the
  // div of node u by iterating through divs and finding first index where u <
  // divs[index]. Then use dists[index] to select its random neighbors.
  // This would allow us to have all cache hits, without greatly compromising
  // the randomness of the initial state.
  // Parallelizing this would also enable us to saturate the NVMe disk better.
  let start_loop = Instant::now();
  for u in 0..num_vertices {

    for nbr_ix in 0..out_degree {
      let v = rand_vertex.sample(prng);
      let distance = get_dist(u, v, db, dist_fn);
      g.edges.push((AtomicU32::new(v), AtomicF32::new(distance)));
      let mut s = bp.backpointers[v as usize].lock();
      s.insert(u);
    }

    if u % 1000_000 == 0 {
      println!("finished u = {:?}, elapsed = {:?}", u, start_loop.elapsed());
    }

  }
  (g, bp)
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
fn nn_descent_thread<R : RngCore>(
  // Lower bound (inclusive) node that this thread is responsible for
  i: u32,
  // Upper bound (exclusive) node that this thread is responsible for
  j: u32,
  // Early stopping parameter. Suggested default from the paper: 0.001.
  delta: f64,
  g: Arc<DenseKNNGraph>,
  bp: Arc<DenseKNNGraphBackpointers>,
  prng: R) -> () {
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
  dist_fn: fn(&T, &T) -> f32
) -> DenseKNNGraph {
  unimplemented!()
}
