use is_sorted::IsSorted;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::binary_heap::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::hash::BuildHasher;
use std::time::Instant;

pub use search_results::*;

use edge::*;

use id_mapping::*;

use error::*;

const DEFAULT_LAMBDA: u8 = 0;
const DEFAULT_NUM_SEARCHERS: u32 = 5;
const DEFAULT_OUT_DEGREE: u8 = 5;

/// Stores search result info. Used internally in query_internal, and is
/// transformed into a SearchResult before being returned to the user.
#[derive(Debug, Clone, Copy)]
struct IntSearchResult {
  /// The internal identifier for this item within the graph.
  pub internal_id: u32,
  /// The distance from the query point to this item.
  pub dist: f32,

  /// The internal identifier of the node that was the root of the search for
  /// this item.
  pub(crate) search_root_ancestor: u32,
  /// The depth of the search for this item.
  pub(crate) search_depth: u32,
}

impl IntSearchResult {
  pub(crate) fn new(
    internal_id: u32,
    dist: f32,
    search_root_ancestor: u32,
    search_depth: u32,
  ) -> IntSearchResult {
    Self {
      internal_id,
      dist,
      search_root_ancestor,
      search_depth,
    }
  }
}

impl PartialEq for IntSearchResult {
  fn eq(&self, other: &Self) -> bool {
    self.dist == other.dist
  }
}

impl Eq for IntSearchResult {}

impl PartialOrd for IntSearchResult {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    self.dist.partial_cmp(&other.dist)
  }
}

impl Ord for IntSearchResult {
  fn cmp(&self, other: &Self) -> Ordering {
    self.dist.total_cmp(&other.dist)
  }
}

/// Configuration for the graph. Use [KnnGraphConfigBuilder] to construct.
#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct KnnGraphConfig<S: BuildHasher + Clone + Default> {
  /// The max number of vertices that can be inserted into the graph. Constant.
  pub(crate) capacity: u32,
  /// The number of approximate nearest neighbors to store for each inserted
  /// element. This is a constant. Each graph node is guaranteed to always have
  /// exactly this out-degree. It's recommended to set this equal to the
  /// intrinsic dimensionality of the data. Lower values hinder search performance
  /// by preventing the search from moving in useful directions, higher values
  /// hinder performance by incurring needless distance calls.
  pub(crate) out_degree: u8,
  /// Number of simultaneous searchers in the beam search.
  pub(crate) num_searchers: u32,
  /// The [std::hash::BuildHasher] to use for the internal hash table.
  #[serde(skip)]
  pub(crate) build_hasher: S,
  /// Whether to use restricted recursive neighborhood propagation. This improves
  /// search speed by about 10%, but decreases insertion throughput by about 10%.
  pub(crate) use_rrnp: bool,
  /// Maximum recursion depth for RRNP. 2 is a good default.
  pub(crate) rrnp_max_depth: u32,

  // TODO: parametrize the graph type so that the LGD vector is never allocated/
  // takes no memory if this is set to false.
  /// Whether to use lazy graph diversification. This improves search speed.
  pub(crate) use_lgd: bool,

  /// If true, optimize element storage for speed. If true, optimize for memory.
  /// Internally, this controls whether IdMapping uses Arc to reduce copies, or
  /// just stores T inline in each data structure.
  pub(crate) optimize_for_small_type: bool,
}

impl<S: BuildHasher + Clone + Default> KnnGraphConfig<S> {
  /// Get the capacity of the config.
  /// The max number of vertices that can be inserted into the graph. Constant.
  pub fn capacity(&self) -> u32 {
    self.capacity
  }

  /// Get the out_degree of the config.
  /// The number of approximate nearest neighbors to store for each inserted
  /// element. This is a constant. Each graph node is guaranteed to always have
  /// exactly this out-degree. It's recommended to set this equal to the
  /// intrinsic dimensionality of the data. Lower values hinder search performance
  /// by preventing the search from moving in useful directions, higher values
  /// hinder performance by incurring needless distance calls.
  pub fn out_degree(&self) -> u8 {
    self.out_degree
  }

  /// Get the num_searchers of the config.
  /// Number of simultaneous searchers in the beam search.
  /// This has a minor effect on search speed.
  pub fn num_searchers(&self) -> u32 {
    self.num_searchers
  }

  /// Get the build_hasher of the config.
  /// This is the [std::hash::BuildHasher] to use for the internal hash table.
  pub fn build_hasher(&self) -> S {
    self.build_hasher.clone()
  }

  /// Get the use_rrnp of the config.
  /// Whether to use restricted recursive neighborhood propagation. In my tests,
  /// This improves search speed by about 10%, but decreases insertion
  /// throughput by about 10%.
  pub fn use_rrnp(&self) -> bool {
    self.use_rrnp
  }

  /// Get the rrnp_max_depth of the config.
  /// Maximum recursion depth for RRNP. 2 is a good default.
  pub fn rrnp_max_depth(&self) -> u32 {
    self.rrnp_max_depth
  }

  /// Get the use_lgd of the config.
  /// Whether to use lazy graph diversification. This improves search speed.
  pub fn use_lgd(&self) -> bool {
    self.use_lgd
  }

  /// If true, optimize for speed by reducing pointer indirections, but stores each
  /// T twice. If false, optimize for memory by storing each T only once, but
  /// increasing pointer indirections. Rule of thumb: set to true if T is 64 bits
  /// or smaller, or if memory usage is not a concern.
  pub fn optimize_for_small_type(&self) -> bool {
    self.optimize_for_small_type
  }
}

/// Builder for [KnnGraphConfig].
pub struct KnnGraphConfigBuilder<S: BuildHasher + Clone + Default> {
  config: KnnGraphConfig<S>,
}

impl<S: BuildHasher + Clone + Default> KnnGraphConfigBuilder<S> {
  /// Create a new builder with the given capacity, out_degree, num_searchers,
  /// distance function, and hash builder.
  /// * capacity: The max number of vertices that can be inserted into the graph.
  /// * out_degree: The number of approximate nearest neighbors to store for each
  /// inserted element. This is a constant. Each graph node is guaranteed to
  /// always have exactly this out-degree. It's recommended to set this equal to
  /// the intrinsic dimensionality of the data. Setting it lower than that
  ///  hinder search performance by preventing the search from moving in useful
  /// directions, setting it higher than that hinders performance by incurring
  /// needless distance calls.
  /// * num_searchers: Number of simultaneous searchers in the beam search.
  ///   This has a minor effect on search speed.
  /// * dist_fn: distance function. Must satisfy the criteria of a metric:
  ///  <https://en.wikipedia.org/wiki/Metric_(mathematics)>. Several internal
  /// optimizations assume the triangle inequality holds.
  /// * build_hasher: The [std::hash::BuildHasher] to use for the internal hash
  /// table.
  pub fn new(
    capacity: u32,
    out_degree: u8,
    num_searchers: u32,
    build_hasher: S,
  ) -> KnnGraphConfigBuilder<S> {
    KnnGraphConfigBuilder {
      config: KnnGraphConfig {
        capacity,
        out_degree,
        num_searchers,
        build_hasher,
        use_rrnp: true,
        rrnp_max_depth: 2,
        use_lgd: true,
        optimize_for_small_type: false,
      },
    }
  }

  /// Initialize the builder from a [KnnGraphConfig].
  pub fn from(config: KnnGraphConfig<S>) -> KnnGraphConfigBuilder<S> {
    KnnGraphConfigBuilder { config }
  }

  /// Create a new builder with the given capacity, and hash
  /// builder. The out_degree and num_searchers are set to the default values of
  /// 5 and 5, respectively.
  pub fn with_capacity_and_hasher(
    capacity: u32,
    build_hasher: S,
  ) -> KnnGraphConfigBuilder<S> {
    KnnGraphConfigBuilder {
      config: KnnGraphConfig {
        capacity,
        out_degree: DEFAULT_OUT_DEGREE,
        num_searchers: DEFAULT_NUM_SEARCHERS,
        build_hasher,
        use_rrnp: true,
        rrnp_max_depth: 2,
        use_lgd: true,
        optimize_for_small_type: false,
      },
    }
  }

  /// Set the capacity, the maximum number of items that can be inserted into
  /// the graph.
  pub fn capacity(mut self, capacity: u32) -> KnnGraphConfigBuilder<S> {
    self.config.capacity = capacity;
    self
  }

  /// Set the out_degree, the number of approximate nearest neighbors to store
  /// for each inserted element. This is a constant. Each graph node is
  /// guaranteed to always have exactly this out-degree. It's recommended to set
  /// this equal to the intrinsic dimensionality of the data. Setting it lower
  /// than that hinder search performance by preventing the search from moving
  /// in useful directions, setting it higher than that hinders performance by
  /// incurring needless distance calls.
  pub fn out_degree(mut self, out_degree: u8) -> KnnGraphConfigBuilder<S> {
    self.config.out_degree = out_degree;
    self
  }

  /// Set the num_searchers, the number of simultaneous searchers in the beam
  /// search. This has a minor effect on search speed. Try tuning this for your
  /// use case.
  pub fn num_searchers(
    mut self,
    num_searchers: u32,
  ) -> KnnGraphConfigBuilder<S> {
    self.config.num_searchers = num_searchers;
    self
  }

  /// Whether to use restricted recursive neighborhood propagation. This improves
  /// search speed by about 10%, but decreases insertion throughput by about 10%.
  pub fn use_rrnp(mut self, use_rrnp: bool) -> KnnGraphConfigBuilder<S> {
    self.config.use_rrnp = use_rrnp;
    self
  }

  /// Maximum recursion depth for RRNP. 2 is a good default.
  pub fn rrnp_max_depth(
    mut self,
    rrnp_max_depth: u32,
  ) -> KnnGraphConfigBuilder<S> {
    self.config.rrnp_max_depth = rrnp_max_depth;
    self
  }

  /// Whether to use lazy graph diversification. This improves search speed.
  pub fn use_lgd(mut self, use_lgd: bool) -> KnnGraphConfigBuilder<S> {
    self.config.use_lgd = use_lgd;
    self
  }

  /// If true, optimize for speed by reducing pointer indirections, but stores each
  // T twice. If false, optimize for memory by storing each T only once, but
  // increase pointer indirections. Rule of thumb: set to true if T is 64 bits
  // or smaller, or if memory usage is not a concern.
  pub fn optimize_for_small_type(
    mut self,
    optimize_for_small_type: bool,
  ) -> KnnGraphConfigBuilder<S> {
    self.config.optimize_for_small_type = optimize_for_small_type;
    self
  }

  /// Build the config.
  pub fn build(self) -> KnnGraphConfig<S> {
    self.config
  }
}

/// A node in the graph that is used as the starting point for a nearest
/// neighbor search.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct StartPoint {
  id: u32,
  priority: u32,
}

impl PartialOrd for StartPoint {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl Ord for StartPoint {
  fn cmp(&self, other: &Self) -> Ordering {
    other.priority.cmp(&self.priority)
  }
}

// NOTE: this abomination is necessary because I want helper functions that
// return parts of the graph data structure, but the borrow checker is dumb.
// This macro simply inlines get_edges_mut.
// TODO: find a better solution.
macro_rules! get_edges_mut_macro {
  ($self:ident, $index:ident) => {
    &mut $self.edges.slice_mut(
      ($index as usize * $self.config.out_degree as usize)
        ..$index as usize * $self.config.out_degree as usize
          + $self.config.out_degree as usize,
    )
  };
}

macro_rules! get_edges_macro {
  ($self:ident, $index:ident) => {
    &$self.edges.slice(
      ($index as usize * $self.config.out_degree as usize)
        ..$index as usize * $self.config.out_degree as usize
          + $self.config.out_degree as usize,
    )
  };
}

// NOTE: This is separate from KnnGraph simply because it is serializable and
// doesn't contain the dist_fn. But we actually need to push dist_fn all the way
// up to the top-level type so the user-facing type can be saved/loaded.
/// A directed graph stored contiguously in memory as an adjacency list.
/// All vertices are guaranteed to have the same out-degree.
/// Nodes are u32s numbered from 0 to n-1.
#[derive(Deserialize, Serialize)]
pub(crate) struct DenseKnnGraph<
  T: Clone + Eq + std::hash::Hash,
  S: BuildHasher + Clone + Default,
> {
  /// A mapping from the user's ID type, T, to our internal ids, which are u32.
  /// We use u32 internally for memory efficiency (which also makes us faster).
  #[serde(bound(
    serialize = "IdMapping<T, S>: Serialize",
    deserialize = "IdMapping<T, S>: Deserialize<'de>"
  ))]
  pub(crate) mapping: IdMapping<T, S>,
  /// The underlying buffer of capacity*out_degree neighbor information.
  /// An adjacency list of (node id, distance, lambda crowding factor).
  /// Use with caution.
  /// Prefer to use get_edges to access the neighbors of a vertex.
  edges: EdgeVec,
  /// Maintains an association between vertices and the vertices that link out to
  /// them. In other words, each backpointers\[i\] is the set of vertices S s.t.
  /// for all x in S, a directed edge exists pointing from x to i.
  backpointers: Vec<Vec<u32>>,
  /// The configuration of the graph that was passed in at construction time.
  #[serde(bound(
    serialize = "KnnGraphConfig<S>: Serialize",
    deserialize = "KnnGraphConfig<S>: Deserialize<'de>"
  ))]
  pub(crate) config: KnnGraphConfig<S>,
  /// The set of internal ids to start searches from. These are randomly
  /// selected using reservoir sampling as points are inserted into the graph.
  /// The size of this heap is equal to config.num_searchers.
  starting_points_reservoir_sample: BinaryHeap<StartPoint>,
}

impl<'a, T: Clone + Eq + std::hash::Hash, S: BuildHasher + Clone + Default>
  DenseKnnGraph<T, S>
{
  fn num_vertices(&self) -> usize {
    self.mapping.len()
  }

  /// Allocates a graph of the specified size and out_degree, but
  /// doesn't populate the edges.
  pub(crate) fn empty(config: KnnGraphConfig<S>) -> DenseKnnGraph<T, S> {
    let mapping = if config.optimize_for_small_type() {
      IdMapping::with_capacity_and_hasher_copy(
        config.capacity,
        config.build_hasher.clone(),
      )
    } else {
      IdMapping::with_capacity_and_hasher_arc(
        config.capacity,
        config.build_hasher.clone(),
      )
    };

    let to = vec![0; config.capacity as usize * config.out_degree as usize];

    let distance =
      vec![0.0; config.capacity as usize * config.out_degree as usize];

    let crowding_factor =
      vec![0; config.capacity as usize * config.out_degree as usize];

    let edges = EdgeVec {
      to,
      distance,
      crowding_factor,
    };

    let mut backpointers = Vec::with_capacity(config.capacity as usize);

    for _ in 0..config.capacity {
      backpointers.push(Vec::<u32>::new());
    }

    DenseKnnGraph {
      mapping,
      edges,
      backpointers,
      config: config.clone(),
      starting_points_reservoir_sample: BinaryHeap::with_capacity(
        config.num_searchers as usize,
      ),
    }
  }

  /// Get the neighbors of u and their distances. Errors if index
  /// >= capacity or does not exist.
  pub fn get_edges(&self, index: u32) -> Result<EdgeSlice, Error> {
    if index >= self.config.capacity {
      return Err(Error::InternalError(format!(
        "index {} does not exist in internal ids",
        index
      )));
    };

    let i = index * self.config.out_degree as u32;
    let j = i + self.config.out_degree as u32;
    Ok(self.edges.slice(i as usize..j as usize))
  }

  pub(crate) fn debug_get_neighbor_indices(
    &self,
    index: u32,
  ) -> Result<Vec<u32>, Error> {
    Ok(
      self
        .get_edges(index)?
        .iter()
        .map(|e| *e.to)
        .collect::<Vec<_>>(),
    )
  }

  /// Get the neighbors of u and their distances. Errors if index
  /// >= capacity or does not exist.
  fn get_edges_mut(&mut self, index: u32) -> Result<EdgeSliceMut, Error> {
    if index >= self.config.capacity {
      return Err(Error::InternalError(format!(
        "index {} does not exist in internal ids",
        index
      )));
    };

    let i = index * self.config.out_degree as u32;
    let j = i + self.config.out_degree as u32;
    Ok(self.edges.slice_mut(i as usize..j as usize))
  }

  fn sort_edges(&mut self, index: u32) -> Result<(), Error> {
    let mut edges = self.get_edges_mut(index)?;
    //TODO: allocating a new vec and edges is absolutely horrible for performance.
    // Implement sort manually
    // if necessary, but try to find a library that has a generic enough sort
    // implementation.
    let mut tmp_edges_vec: Vec<Edge> = Vec::<Edge>::new();
    for e in edges.iter() {
      tmp_edges_vec.push(Edge {
        to: *e.to,
        distance: *e.distance,
        crowding_factor: *e.crowding_factor,
      })
    }
    tmp_edges_vec.sort_by(|a, b| {
      let b_dist = b.distance;
      a.distance.total_cmp(&b_dist)
    });

    for (i, edge) in tmp_edges_vec.iter().enumerate() {
      let e = edges.get_mut(i).ok_or(Error::InternalError(
        "Failed to get edge in sort_edges".to_string(),
      ))?;
      *e.to = edge.to;
      *e.distance = edge.distance;
      *e.crowding_factor = edge.crowding_factor;
    }

    Ok(())
  }

  /// Get distance in terms of two internal ids.
  fn dist_int(
    &self,
    int_id1: u32,
    int_id2: u32,
    dist_fn: &dyn Fn(&T, &T) -> f32,
  ) -> Result<f32, Error> {
    let ext_id1 = self.mapping.int_to_ext(int_id1)?;
    let ext_id2 = self.mapping.int_to_ext(int_id2)?;
    Ok(dist_fn(ext_id1, ext_id2))
  }

  /// Replace the edge (u,v) with a new edge (u,w)
  /// DOES NOT update v or w's backpointers
  fn replace_edge_no_backptr_update(
    &mut self,
    u: u32,
    v: u32,
    w: u32,
    u_w_dist: f32,
  ) -> Result<(), Error> {
    let mut u_edges = self.get_edges_mut(u)?;
    for EdgeRefMut {
      to,
      distance,
      crowding_factor,
    } in u_edges.iter_mut()
    {
      if *to == v {
        *to = w;
        *distance = u_w_dist;
        *crowding_factor = DEFAULT_LAMBDA;
      }
    }

    // TODO: the list is all sorted except for the new element; we could optimize
    // this.
    self.sort_edges(u)
  }

  /// Creates an edge from `from` to `to` if the distance `dist` between them is
  /// less than the distance from `from` to one of its existing neighbors `u`.
  /// If so, removes the edge (`from`, `u`).
  ///
  /// `to` must have already been added to the graph by insert_vertex etc.
  ///
  /// Returns `true` if the new edge was added.
  fn insert_edge_if_closer(
    &mut self,
    from: u32,
    to: u32,
    dist: f32,
  ) -> Result<bool, Error> {
    let mut edges = self.get_edges_mut(from)?;
    let most_distant_edge = edges.last_mut().ok_or(Error::InternalError(
      "Failed to get last edge in insert_edge_if_closer".to_string(),
    ))?;

    if dist < *most_distant_edge.distance {
      let old = *most_distant_edge.to;
      *most_distant_edge.to = to;
      *most_distant_edge.distance = dist;
      *most_distant_edge.crowding_factor = DEFAULT_LAMBDA;
      self.sort_edges(from)?;
      self.backpointers[old as usize].retain(|b| *b != from);
      self.backpointers[to as usize].push(from);
      return Ok(true);
    }

    Ok(false)
  }

  /// Insert a new vertex into the graph, with the given k neighbors and their
  /// distances. Does not connect other vertices to this vertex.
  /// Throws if the graph is already full (num_vertices == capacity).
  /// nbrs and dists must be equal to the out_degree of the graph.
  fn insert_vertex(
    &mut self,
    u: u32,
    nbrs: Vec<u32>,
    dists: Vec<f32>,
  ) -> Result<(), Error> {
    let od = self.config.out_degree as usize;

    if !(nbrs.len() == od && dists.len() == od) {
      return Err(Error::InternalError(
        "nbrs and dists must be equal to the out_degree of the graph"
          .to_string(),
      ));
    }

    for ((edge_ix, nbr), dist) in nbrs.iter().enumerate().zip(dists) {
      let mut u_out_edges = self.get_edges_mut(u)?;
      let edge = u_out_edges.get_mut(edge_ix).ok_or(Error::InternalError(
        "Failed to get edge in insert_vertex".to_string(),
      ))?;
      *edge.to = *nbr;
      *edge.distance = dist;
      *edge.crowding_factor = DEFAULT_LAMBDA;
      let s = &mut self.backpointers[*nbr as usize];
      s.push(u);
    }

    self.sort_edges(u)
  }

  /// Print the graph, for debugging.
  #[allow(dead_code)]
  pub fn debug_print(&self) -> Result<(), Error> {
    println!("### Adjacency list (index, distance, lambda)");
    for (i, _) in self.mapping.int_ext_iter() {
      println!("Node {i}");
      println!(
        "{:#?}",
        self
          .get_edges(i)?
          .iter()
          .map(|e| (e.to, e.distance, e.crowding_factor))
          .collect::<Vec<_>>()
      );
    }
    println!("### Backpointers");
    for (i, _) in self.mapping.int_ext_iter() {
      println!("Node {} {:#?}", i, self.backpointers[i as usize]);
    }
    Ok(())
  }

  /// Returns error result if graph is internally inconsistent.
  pub fn consistency_check(&self) -> Result<(), Error> {
    if self.edges.len()
      != self.config.capacity as usize * self.config.out_degree as usize
    {
      return Err(Error::InternalError(format!(
        "edges.len() is not equal to capacity * out_degree. {} != {} * {}",
        self.edges.len(),
        self.config.capacity,
        self.config.out_degree
      )));
    }

    for (i, _) in self.mapping.int_ext_iter() {
      for e in self.get_edges(i)? {
        if *e.to == i {
          return Err(Error::InternalError(format!("Self loop at node {}", i)));
        }
        let nbr_backptrs = &self.backpointers[*e.to as usize];
        if !nbr_backptrs.contains(&i) {
          return Err(Error::InternalError(format!(
            "Vertex {} links to {} but {}'s backpointers don't include {}!",
            i, *e.to, *e.to, i
          )));
        }
      }

      if !(IsSorted::is_sorted_by_key(
        &mut self
          .get_edges(i)?
          .iter()
          .map(|e| (*e.to, *e.distance, *e.crowding_factor)),
        |e| e.1,
      )) {
        return Err(Error::InternalError(format!(
          "Edges of node {} are not sorted by distance",
          i
        )));
      };

      for referrer in self.backpointers[i as usize].iter() {
        if !(self.debug_get_neighbor_indices(*referrer)?.contains(&i)) {
          return Err(Error::InternalError(format!(
            "Vertex {} has a backpointer to {} but {}'s neighbors don't include {}!",
            i, *referrer, *referrer, i
          )));
        };
      }
    }
    Ok(())
  }

  fn exists_edge(&self, u: u32, v: u32) -> Result<bool, Error> {
    for e in self.get_edges(u)? {
      if v == *e.to {
        return Ok(true);
      }
    }
    Ok(false)
  }

  // TODO: expose an interface where the user provides one of their ids already
  // in the database and we
  // return n approximate nearest neighbors very fast by using upper bounds on
  // distance in the same way as two_hop_neighbors_and_dist_upper_bounds. Of
  // course, for best results we should be unsharded.

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
  ) -> Result<BinaryHeap<Reverse<SearchResult<u32>>>, Error> {
    let mut ret: BinaryHeap<Reverse<SearchResult<u32>>> = BinaryHeap::new();
    for v in self.get_edges(int_id)? {
      for w in self.get_edges(*v.to)? {
        if !self.exists_edge(int_id, *w.to)? {
          ret.push(Reverse(SearchResult {
            item: *w.to,
            internal_id: Some(*w.to),
            dist: *v.distance + *w.distance,
            // NOTE: these fields are nonsense in this context, but it's
            // convenient to use SearchResult as a container for the
            // information we need.
            search_root_ancestor: *w.to,
            search_depth: 2,
          }));
        }
      }
    }
    Ok(ret)
  }

  fn get_farthest_neighbor(
    self: &DenseKnnGraph<T, S>,
    int_id: u32,
  ) -> Result<(u32, f32), Error> {
    let e = self.get_edges(int_id)?.last().ok_or(Error::InternalError(
      "Failed to get farthest neighbor".to_string(),
    ))?;
    Ok((*e.to, *e.distance))
  }

  fn in_and_out_neighbors(
    self: &DenseKnnGraph<T, S>,
    int_id: u32,
  ) -> Result<Vec<(u32, f32)>, Error> {
    let mut ret = Vec::new();
    for w in self.get_edges(int_id)? {
      ret.push((*w.to, *w.distance));
    }
    for w in self.backpointers[int_id as usize].iter() {
      let w_edges = self.get_edges(*w)?;
      let dist = w_edges
        .iter()
        .find(|e| *e.to == int_id)
        .ok_or(Error::InternalError(
          "Backpointers and edges are inconsistent".to_string(),
        ))?
        .distance;
      ret.push((*w, *dist));
    }
    Ok(ret)
  }

  /// Perform RRNP w.r.t. a newly-inserted element q, given the results of
  /// searching for q in the graph. This implements lines 10 to 24 of algorithm 3
  /// in the paper. A small difference is that I pulled the loop `while |W| > 0`
  /// outside of the `for each r in V` loop. That should have no effect on the
  /// behavior, but it makes things more readable.
  fn rrnp(
    self: &mut DenseKnnGraph<T, S>,
    q_int: u32,
    visited_nodes_distances_to_q: &HashMap<u32, f32>,
    dist_fn: &dyn Fn(&T, &T) -> f32,
  ) -> Result<(), Error> {
    let mut w = VecDeque::new();
    let q = self.mapping.int_to_ext(q_int)?.clone(); //TODO: avoid clone
    for (internal_id, dist) in visited_nodes_distances_to_q {
      w.push_back((*internal_id, 0, dist));
    }

    let mut already_rrnped = HashSet::new();

    while !w.is_empty() {
      let (s_int, depth, dist_s_q) = w.pop_front().ok_or(
        Error::InternalError("Impossible happened: w is empty".to_string()),
      )?;
      let (_, dist_s_f) = self.get_farthest_neighbor(s_int)?;
      if depth < self.config.rrnp_max_depth && dist_s_q < &dist_s_f {
        for (e, _) in self.in_and_out_neighbors(s_int)? {
          let e_ext = self.mapping.int_to_ext(e)?;
          if !already_rrnped.contains(&e)
            && !visited_nodes_distances_to_q.contains_key(&e)
          {
            already_rrnped.insert(e);
            let dist_e_q = dist_fn(e_ext, &q);
            self.insert_edge_if_closer(e, q_int, dist_e_q)?;
          }
        }
      }
    }
    Ok(())
  }

  /// Return the average occlusion (lambda values) of the neighbors of the given
  /// node. This is part of the lazy graph diversification algorithm. See the
  /// paper for details. Returns infinity if LGD is disabled.
  fn average_occlusion(&self, int_id: u32) -> Result<f32, Error> {
    if !self.config.use_lgd {
      return Ok(f32::INFINITY);
    }
    let mut sum = 0.0;
    for e in self.get_edges(int_id)? {
      sum += *e.crowding_factor as f32;
    }
    Ok(sum / self.config.out_degree as f32)
  }

  fn to_search_result(
    &self,
    sr: IntSearchResult,
  ) -> Result<SearchResult<T>, Error> {
    let item = self.mapping.int_to_ext(sr.internal_id)?;
    Ok(SearchResult {
      item: item.clone(),
      internal_id: Some(sr.internal_id),
      dist: sr.dist,
      search_root_ancestor: sr.search_root_ancestor,
      search_depth: sr.search_depth,
    })
  }

  fn query_internal(
    &self,
    q: &T,
    max_results: usize,
    dist_fn: &dyn Fn(&T, &T) -> f32,
    ignore_occluded: bool,
  ) -> Result<SearchResults<T>, Error> {
    let query_start = Instant::now();
    let mut num_distance_computations = 0;
    let mut compute_distance = |x, y| {
      let dist = dist_fn(x, y);
      num_distance_computations += 1;
      dist
    };
    let mut q_max_heap: BinaryHeap<IntSearchResult> = BinaryHeap::new();
    let mut r_min_heap: BinaryHeap<Reverse<IntSearchResult>> =
      BinaryHeap::new();
    let mut visited_distances: HashMap<u32, f32> = HashMap::default();
    // TODO: disable stat tracking on insertion, make it optional elsewhere.
    // tracks the starting node of the search path for each node traversed.
    let mut largest_distance_improvement_single_hop = f32::NEG_INFINITY;
    let mut smallest_distance_improvement_single_hop = f32::INFINITY;

    // Initialize our search with num_searchers initial points.
    // lines 2 to 10 of the pseudocode
    let mut min_r_dist = f32::INFINITY;
    let mut max_r_dist = f32::NEG_INFINITY;
    for StartPoint { id: r_int, .. } in &self.starting_points_reservoir_sample {
      let r_ext = self.mapping.int_to_ext(*r_int)?;
      let r_dist = compute_distance(q, r_ext);
      if r_dist < min_r_dist {
        min_r_dist = r_dist;
      }
      if r_dist > max_r_dist {
        max_r_dist = r_dist;
      }
      visited_distances.insert(*r_int, r_dist);
      r_min_heap.push(Reverse(IntSearchResult::new(*r_int, r_dist, *r_int, 0)));
      match q_max_heap.peek() {
        None => {
          q_max_heap.push(IntSearchResult::new(*r_int, r_dist, *r_int, 0));
          // NOTE: pseudocode has a bug: R.insert(r) at both line 2 and line 8
          // We are skipping it here since we did it above.
        }
        Some(f) => {
          if r_dist < f.dist || q_max_heap.len() < max_results {
            q_max_heap.push(IntSearchResult::new(*r_int, r_dist, *r_int, 0));
          }
        }
      }
    }

    // The main search loop. While unvisited nodes exist in r_min_heap, keep
    // searching.
    // lines 11 to 27 of the pseudocode
    while !r_min_heap.is_empty() {
      while q_max_heap.len() > max_results {
        q_max_heap.pop();
      }

      let Reverse(sr) = r_min_heap
        .pop()
        .ok_or(Error::InternalError("r_min_heap is empty".to_string()))?;
      let f_dist = {
        q_max_heap
          .peek()
          .ok_or(Error::InternalError("q_max_heap is empty".to_string()))?
          .dist
      };
      let sr_int = &sr.internal_id;
      if sr.dist > f_dist {
        break;
      }

      let average_lambda = self.average_occlusion(*sr_int)?;
      let sr_edges = self.get_edges(*sr_int)?;
      let r_nbrs_iter = self.backpointers[*sr_int as usize].iter().chain(
        sr_edges
          .iter()
          .filter(|e| {
            !(ignore_occluded && *e.crowding_factor as f32 >= average_lambda)
          })
          .map(|e| e.to),
      );

      for e in r_nbrs_iter {
        let e_ext = self.mapping.int_to_ext(*e)?;
        if !visited_distances.contains_key(e) {
          let e_dist = compute_distance(q, e_ext);
          visited_distances.insert(*e, e_dist);

          if e_dist < f_dist || q_max_heap.len() < max_results {
            let hop_distance_improvement = -(e_dist - sr.dist);
            if hop_distance_improvement
              > largest_distance_improvement_single_hop
            {
              largest_distance_improvement_single_hop =
                hop_distance_improvement;
            }
            if hop_distance_improvement
              < smallest_distance_improvement_single_hop
            {
              smallest_distance_improvement_single_hop =
                hop_distance_improvement;
            }

            q_max_heap.push(IntSearchResult::new(
              *e,
              e_dist,
              sr.search_root_ancestor,
              sr.search_depth + 1,
            ));
            r_min_heap.push(Reverse(IntSearchResult::new(
              *e,
              e_dist,
              sr.search_root_ancestor,
              sr.search_depth + 1,
            )));
          }
        }
      }
    }

    let mut approximate_nearest_neighbors = Vec::new();
    for isr in q_max_heap.into_sorted_vec() {
      let sr = self.to_search_result(isr)?;
      approximate_nearest_neighbors.push(sr);
    }

    let nearest_neighbor =
      approximate_nearest_neighbors
        .last()
        .ok_or(Error::InternalError(
          "no nearest neighbors found in non-empty graph".to_string(),
        ))?;
    let nearest_neighbor_distance = nearest_neighbor.dist;

    let nearest_neighbor_path_length = nearest_neighbor.search_depth as usize;

    let num_visited = visited_distances.len();

    Ok(SearchResults {
      approximate_nearest_neighbors,
      visited_nodes_distances_to_q: Some(visited_distances),
      search_stats: Some(SearchStats {
        num_distance_computations,
        distance_from_nearest_starting_point: min_r_dist,
        distance_from_farthest_starting_point: max_r_dist,
        search_duration: Instant::now() - query_start,
        largest_distance_improvement_single_hop,
        smallest_distance_improvement_single_hop,
        nearest_neighbor_path_length,
        nearest_neighbor_distance,
        num_visited,
        // distance_from_nearest_neighbor_to_its_starting_point: todo!(),
      }),
    })
  }

  /// Use reservoir sampling to (possibly) insert a new starting point into
  /// our set of search starting points. Called when new nodes are inserted.
  fn maybe_insert_starting_point<R: RngCore>(&mut self, q: u32, prng: &mut R) {
    let priority = prng.next_u32();
    let start_point = StartPoint { id: q, priority };
    self.starting_points_reservoir_sample.push(start_point);
    while self.starting_points_reservoir_sample.len()
      > self.config.num_searchers as usize
    {
      self.starting_points_reservoir_sample.pop();
    }
  }

  /// Delete the point q from the starting points
  fn maybe_replace_starting_point<R: RngCore>(
    &mut self,
    q: u32,
    candidate_replacements: Vec<u32>,
    prng: &mut R,
  ) {
    // TODO: lots of intermediate data structures allocated here. Optimize.
    let to_keep = self
      .starting_points_reservoir_sample
      .drain()
      .filter(|s| s.id != q)
      .collect::<Vec<_>>();
    for x in to_keep.iter() {
      self.starting_points_reservoir_sample.push(*x);
    }
    if self.starting_points_reservoir_sample.len()
      < self.config.num_searchers as usize
    {
      let to_keep_ids = to_keep.iter().map(|s| s.id).collect::<Vec<_>>();
      for replacement in candidate_replacements
        .iter()
        .filter(|s| !to_keep_ids.contains(s))
      {
        if self.starting_points_reservoir_sample.len()
          < self.config.num_searchers as usize
        {
          self.starting_points_reservoir_sample.push(StartPoint {
            id: *replacement,
            priority: prng.next_u32(),
          });
        }
      }
    }
  }

  /// Inserts a new data point into the graph. The graph must not be full.
  /// Optionally performs restricted recursive neighborhood propagation and
  /// lazy graph diversification. These options are set when the graph is
  /// constructed.
  ///
  /// This is equivalent to one iteration of Algorithm 3 in
  /// "Approximate k-NN Graph Construction: A Generic Online Approach".
  pub(crate) fn insert<R: RngCore>(
    &mut self,
    q: T,
    dist_fn: &dyn Fn(&T, &T) -> f32,
    prng: &mut R,
  ) -> Result<(), Error> {
    //TODO: return the index of the new data point
    //TODO: API for using internal ids, for improved speed.
    let SearchResults {
      approximate_nearest_neighbors: nearest_neighbors,
      visited_nodes_distances_to_q,
      search_stats: _,
    } = self.query_internal(
      &q,
      self.config.out_degree as usize,
      dist_fn,
      false,
    )?;

    let visited_nodes_distances_to_q = visited_nodes_distances_to_q.ok_or(
      Error::InternalError("missing visited nodes in insert".to_string()),
    )?;

    let (neighbors, dists): (Vec<Result<u32, Error>>, Vec<f32>) =
      nearest_neighbors
        .iter()
        .map(|sr| {
          (
            sr.internal_id
              .ok_or(Error::InternalError("missing internal id".to_string())),
            sr.dist,
          )
        })
        .unzip();
    let neighbors = neighbors.into_iter().collect::<Result<Vec<_>, _>>()?;
    let q_int = self.mapping.insert(q)?;
    self.insert_vertex(q_int, neighbors, dists)?;

    for (r_int, r_dist) in &visited_nodes_distances_to_q {
      let r_int = *r_int;
      let is_inserted = self.insert_edge_if_closer(r_int, q_int, *r_dist)?;
      let r_edges = get_edges_mut_macro!(self, r_int);
      if is_inserted && self.config.use_lgd {
        apply_lgd(r_edges, q_int, &visited_nodes_distances_to_q)?;
      }
    }
    if self.config.use_rrnp {
      self.rrnp(q_int, &visited_nodes_distances_to_q, dist_fn)?;
    }
    self.maybe_insert_starting_point(q_int, prng);
    Ok(())
  }

  pub(crate) fn delete<R: RngCore>(
    &'a mut self,
    ext_id: &T,
    dist_fn: &dyn Fn(&T, &T) -> f32,
    prng: &mut R,
  ) -> Result<(), Error> {
    if self.num_vertices() < self.config.out_degree as usize {
      return Err(Error::InternalError("graph is too small to delete a node while maintaining internal invariants".to_string()));
    };
    let int_id = self.mapping.ext_to_int(ext_id)?;
    let nbrs = get_edges_mut_macro!(self, int_id)
      .iter()
      .map(|e| *e.to)
      .collect::<Vec<_>>();
    self.maybe_replace_starting_point(int_id, nbrs, prng);
    for referrer in self
      .backpointers
      .get(int_id as usize)
      .ok_or(Error::InternalError("missing backpointers".to_string()))?
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
        self.get_edges(*referrer)?.iter().map(|x| *x.to).collect();
      let mut referrer_nbrs_of_nbrs =
        self.two_hop_neighbors_and_dist_upper_bounds(*referrer)?;

      let (new_referent, dist_referrer_new_referent) = loop {
        match referrer_nbrs_of_nbrs.pop() {
          None => {
            // If we reach this case, we're in a small connected component
            // and we can't find any neighbors of neighbors who aren't
            // already neighbors or ourself.
            // Instead of looking for neighbors of neighbors, we simply
            // do a search.
            let SearchResults {
              approximate_nearest_neighbors: nearest_neighbors,
              ..
            } = self.query_internal(
              self.mapping.int_to_ext(*referrer)?,
              // we need to find one new node who isn't a neighbor, the
              // node being deleted, or referrer itself, so we search for
              // one more than that number.
              self.config.out_degree as usize + 3,
              dist_fn,
              false, // we need all the results we can find, so don't ignore occluded nodes
            )?;

            let nearest_neighbor = nearest_neighbors
              .iter()
              .map(|sr| {
                (
                  sr.internal_id.ok_or(Error::InternalError(
                    "Internal id not found".to_string(),
                  )),
                  sr.dist,
                )
              })
              .find_map(|res| match res {
                (Ok(res_int_id), dist) => {
                  if res_int_id != int_id
                    && res_int_id != *referrer
                    && !referrer_nbrs.contains(&res_int_id)
                  {
                    Some(Ok((res_int_id, dist)))
                  } else {
                    None
                  }
                }
                (Err(e), _) => Some(Err(e)),
              });

            break nearest_neighbor.ok_or(Error::InternalError(
              "no replacement neighbors found -- is the graph too small?"
                .to_string(),
            ))?;
          }

          Some(Reverse(SearchResult { item: id, .. })) => {
            if id != int_id && id != *referrer && !referrer_nbrs.contains(&id) {
              let dist = self.dist_int(*referrer, id, dist_fn)?;
              break Ok((id, dist));
            }
          }
        }
      }?;

      self.replace_edge_no_backptr_update(
        *referrer,
        int_id,
        new_referent,
        dist_referrer_new_referent,
      )?;

      // Insert referrer into the backpointers of the new referent.
      let new_referent_backpointers = self
        .backpointers
        .get_mut(new_referent as usize)
        .ok_or(Error::InternalError("missing backpointers".to_string()))?;
      new_referent_backpointers.push(*referrer);
    }

    // Remove backpointers for all nodes the deleted node was pointing to.
    for nbr in get_edges_macro!(self, int_id).iter().map(|x| *x.to) {
      let nbr_backpointers = self
        .backpointers
        .get_mut(nbr as usize)
        .ok_or(Error::InternalError("missing backpointers".to_string()))?;

      let ix_to_remove = nbr_backpointers
        .iter()
        .position(|x| *x == int_id)
        .ok_or(Error::InternalError("missing backpointers".to_string()))?;
      nbr_backpointers.swap_remove(ix_to_remove);
    }

    // Reset backpointers for the deleted node.
    let backpointers = self
      .backpointers
      .get_mut(int_id as usize)
      .ok_or(Error::InternalError("missing backpointers".to_string()))?;
    *backpointers = Vec::<u32>::with_capacity(self.config.out_degree as usize);
    self.mapping.delete(int_id)?;

    Ok(())
  }

  // TODO: wouldn't returning a min-dist heap be more useful?

  /// Perform beam search for the k nearest neighbors of a point q. Returns
  /// (nearest_neighbors_max_dist_heap, visited_nodes, visited_node_distances_to_q)
  /// This is a translation of the pseudocode of algorithm 1 from
  /// Approximate k-NN Graph Construction: A Generic Online Approach
  pub(crate) fn query<R: RngCore>(
    &self,
    q: &T,
    max_results: usize,
    dist_fn: &dyn Fn(&T, &T) -> f32,
    _prng: &mut R,
  ) -> Result<SearchResults<T>, Error> {
    self.query_internal(q, max_results, dist_fn, true)
  }
}

fn apply_lgd(
  r_edges: &mut EdgeSliceMut,
  q_int: u32,
  visited_node_distances_to_q: &HashMap<u32, f32>,
) -> Result<(), Error> {
  let q_ix =
    r_edges
      .iter()
      .position(|e| *e.to == q_int)
      .ok_or(Error::InternalError(
        "missing edge in apply_lgd".to_string(),
      ))?;
  let q_r_dist = *r_edges
    .get(q_ix)
    .ok_or(Error::InternalError(
      "missing edge in apply_lgd".to_string(),
    ))?
    .distance;

  *r_edges
    .get_mut(q_ix)
    .ok_or(Error::InternalError(
      "missing edge in apply_lgd".to_string(),
    ))?
    .crowding_factor = DEFAULT_LAMBDA;

  for s_ix in 0..r_edges.len() {
    let s_int = *r_edges
      .get(s_ix)
      .ok_or(Error::InternalError(
        "missing edge in apply_lgd".to_string(),
      ))?
      .to;
    let s_r_dist = *r_edges
      .get(s_ix)
      .ok_or(Error::InternalError(
        "missing edge in apply_lgd".to_string(),
      ))?
      .distance;
    let s_q_dist = visited_node_distances_to_q.get(&s_int);

    match s_q_dist {
      Some(s_q_dist) => {
        // rule 1 from the paper
        if s_r_dist < q_r_dist && s_q_dist >= &q_r_dist {
          continue;
        }
        // rule 2 from the paper
        else if s_r_dist < q_r_dist && s_q_dist < &q_r_dist {
          *r_edges
            .get_mut(q_ix)
            .ok_or(Error::InternalError(
              "missing edge in apply_lgd".to_string(),
            ))?
            .crowding_factor += 1;
        }
        // rule 3 from the paper: s_r_dist > q_r_dist, q occludes s
        else if s_q_dist < &s_r_dist {
          *r_edges
            .get_mut(s_ix)
            .ok_or(Error::InternalError(
              "missing edge in apply_lgd".to_string(),
            ))?
            .crowding_factor += 1;
        }
      }

      None => {
        continue;
      }
    }
  }
  Ok(())
}

/// Constructs an exact k-nn graph on the given IDs. O(n^2).
/// `capacity` is max capacity of the returned graph (for future inserts).
/// Must be >= n.
pub(crate) fn exhaustive_knn_graph_internal<
  'a,
  T: Clone + Eq + std::hash::Hash,
  S: BuildHasher + Clone + Default,
  R: RngCore,
>(
  mut ids: Vec<T>,
  config: KnnGraphConfig<S>,
  dist_fn: &'a (dyn Fn(&T, &T) -> f32 + Sync),
  prng: &mut R,
) -> Result<DenseKnnGraph<T, S>, Error> {
  let n = ids.len();

  if config.out_degree as usize >= n {
    return Err(Error::OutDegreeTooLarge(config.out_degree, n));
  }

  let mut g: DenseKnnGraph<T, S> = DenseKnnGraph::empty(config.clone());

  for i_ext in ids.drain(..) {
    let _i = g.mapping.insert(i_ext)?;
  }

  let ext_ints: Vec<(T, u32)> = g
    .mapping
    .ext_int_iter()
    .map(|x| (x.0.clone(), x.1))
    .collect::<Vec<_>>();

  for (i_ext, i) in ext_ints.iter() {
    let mut knn = BinaryHeap::new();
    // TODO: prng in config.
    g.maybe_insert_starting_point(*i, prng);
    for (j_ext, j) in ext_ints.iter() {
      if i_ext == j_ext {
        continue;
      }
      let dist = dist_fn(i_ext, j_ext);
      knn.push(SearchResult::new(j, Some(*j), dist, 0, 0));

      while knn.len() > config.out_degree as usize {
        knn.pop();
      }
    }
    for edge_ix in 0..config.out_degree as usize {
      let SearchResult { item: id, dist, .. } = knn
        .pop()
        .ok_or(Error::InternalError("heap unexpectedly empty".to_string()))?;
      let mut i_edges_mut = g.get_edges_mut(*i)?;
      let e = i_edges_mut.get_mut(edge_ix).ok_or(Error::InternalError(
        "i_edges_mut unexpectedly empty".to_string(),
      ))?;

      *e.to = *id;
      *e.distance = dist;
      *e.crowding_factor = DEFAULT_LAMBDA;
      let s = &mut g.backpointers[*id as usize];
      s.push(*i);
    }
  }

  for i in 0..n {
    g.sort_edges(i as u32)?;
  }

  Ok(g)
}

#[cfg(test)]
mod tests {
  #![allow(clippy::unwrap_used)]
  extern crate nohash_hasher;
  extern crate ordered_float;
  extern crate serde_json;
  use self::nohash_hasher::NoHashHasher;
  use super::*;
  use rand::SeedableRng;
  use rand_xoshiro::Xoshiro256StarStar;
  use std::hash::BuildHasherDefault;

  type Nhh = BuildHasherDefault<NoHashHasher<u32>>;

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

  fn edge_slice_to_vec(e: EdgeSlice) -> Vec<(u32, f32, u8)> {
    e.iter()
      .map(|e| (*e.to, *e.distance, *e.crowding_factor))
      .collect()
  }

  fn mk_config(capacity: u32) -> KnnGraphConfig<Nhh> {
    let out_degree = 5;
    let num_searchers = 5;
    let use_rrnp = false;
    let rrnp_max_depth = 2;
    let use_lgd = false;
    let build_hasher = nohash_hasher::BuildNoHashHasher::default();
    let optimize_for_small_type = false;
    KnnGraphConfig::<Nhh> {
      capacity,
      out_degree,
      num_searchers,
      build_hasher,
      use_rrnp,
      rrnp_max_depth,
      use_lgd,
      optimize_for_small_type,
    }
  }

  #[test]
  fn test_exhaustive_knn_graph() {
    let db: Vec<[i32; 1]> = vec![[1], [2], [3], [10], [11], [12]];
    let dist_fn = &|x: &u32, y: &u32| {
      sq_euclidean_faster(&db[*x as usize], &db[*y as usize])
    };
    let mut config = mk_config(10);
    let mut prng = Xoshiro256StarStar::seed_from_u64(1);
    config.out_degree = 2;
    let g: DenseKnnGraph<u32, Nhh> = exhaustive_knn_graph_internal(
      vec![0u32, 1, 2, 3, 4, 5],
      config,
      dist_fn,
      &mut prng,
    )
    .unwrap();
    g.consistency_check().unwrap();
    assert_eq!(g.debug_get_neighbor_indices(0).unwrap(), vec![1, 2]);
    assert_eq!(g.debug_get_neighbor_indices(1).unwrap(), vec![2, 0]);
    assert_eq!(g.debug_get_neighbor_indices(2).unwrap(), vec![1, 0]);
    assert_eq!(g.debug_get_neighbor_indices(3).unwrap(), vec![4, 5]);
    assert_eq!(g.debug_get_neighbor_indices(4).unwrap(), vec![3, 5]);
    assert_eq!(g.debug_get_neighbor_indices(5).unwrap(), vec![4, 3]);
  }

  #[test]
  fn test_insert_vertex() {
    let mut config = mk_config(3);
    config.out_degree = 2;
    let mut g: DenseKnnGraph<u32, BuildHasherDefault<NoHashHasher<u32>>> =
      DenseKnnGraph::empty(config);
    // NOTE:: g.insert_vertex() calls get_edges_mut which asserts that the
    // internal id is in the mapping. g.mapping.insert() assigns internal ids in
    // order starting from 0, so we are inserting 3 unused u32 external ids,
    // 0,1,2 which get mapped to internal ids 0,1,2. Annoying and unclear, but
    // necessary at the moment.
    g.mapping.insert(0).unwrap();
    g.mapping.insert(1).unwrap();
    g.mapping.insert(2).unwrap();
    g.insert_vertex(0, vec![2, 1], vec![2.0, 1.0]).unwrap();
    g.insert_vertex(1, vec![2, 0], vec![1.0, 1.0]).unwrap();
    g.insert_vertex(2, vec![0, 1], vec![2.0, 1.0]).unwrap();

    assert_eq!(
      edge_slice_to_vec(g.get_edges(0).unwrap()),
      [(1, 1.0, 0), (2, 2.0, 0)]
    );
    assert_eq!(
      edge_slice_to_vec(g.get_edges(1).unwrap()),
      [(2, 1.0, 0), (0, 1.0, 0)].as_slice()
    );
    assert_eq!(
      edge_slice_to_vec(g.get_edges(2).unwrap()),
      [(1, 1.0, 0), (0, 2.0, 0)].as_slice()
    );
  }

  #[test]
  #[should_panic]
  fn test_insert_vertex_panic_too_many_vertex() {
    let mut g: DenseKnnGraph<u32, Nhh> = DenseKnnGraph::empty(mk_config(2));
    g.insert_vertex(0, vec![2, 1], vec![2.0, 1.0]).unwrap();
    g.insert_vertex(1, vec![2, 0], vec![1.0, 1.0]).unwrap();
    g.insert_vertex(2, vec![0, 1], vec![2.0, 1.0]).unwrap();
  }

  #[test]
  #[should_panic]
  fn test_insert_vertex_panic_wrong_neighbor_length() {
    let mut g: DenseKnnGraph<u32, Nhh> = DenseKnnGraph::empty(mk_config(2));
    g.insert_vertex(0, vec![2, 1, 0], vec![2.0, 1.0, 10.1])
      .unwrap();
    g.insert_vertex(1, vec![2, 0], vec![1.0, 1.0]).unwrap();
  }

  #[test]
  fn test_insert_edge_if_closer() {
    let mut config = mk_config(3);
    config.out_degree = 1;
    let mut g: DenseKnnGraph<u32, Nhh> = DenseKnnGraph::empty(config);

    // see note in test_insert_vertex
    g.mapping.insert(0).unwrap();
    g.mapping.insert(1).unwrap();
    g.mapping.insert(2).unwrap();

    g.insert_vertex(0, vec![2], vec![2.0]).unwrap();
    g.insert_vertex(1, vec![2], vec![1.0]).unwrap();
    g.insert_vertex(2, vec![1], vec![1.0]).unwrap();

    assert_eq!(
      edge_slice_to_vec(g.get_edges(0).unwrap()),
      [(2, 2.0, 0)].as_slice()
    );

    assert!(g.insert_edge_if_closer(0, 1, 1.0).unwrap());

    assert_eq!(
      edge_slice_to_vec(g.get_edges(0).unwrap()),
      [(1, 1.0, 0)].as_slice()
    );
    assert_eq!(
      edge_slice_to_vec(g.get_edges(1).unwrap()),
      [(2, 1.0, 0)].as_slice()
    );
    assert_eq!(
      edge_slice_to_vec(g.get_edges(2).unwrap()),
      [(1, 1.0, 0)].as_slice()
    );
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
    let mut prng = Xoshiro256StarStar::seed_from_u64(1);
    let mut g: DenseKnnGraph<u32, Nhh> = exhaustive_knn_graph_internal(
      vec![0, 1, 2, 3, 4, 5],
      mk_config(11),
      dist_fn,
      &mut prng,
    )
    .unwrap();
    for i in 6..11 {
      println!("doing {i}");
      g.insert(i, dist_fn, &mut prng).unwrap();
    }
    g.consistency_check().unwrap();
  }

  #[test]
  fn test_delete() {
    let db: Vec<[i32; 1]> = vec![[1], [2], [3], [10], [11], [12]];
    let dist_fn = &|x: &u32, y: &u32| {
      sq_euclidean_faster(&db[*x as usize], &db[*y as usize])
    };
    let mut config = mk_config(10);
    config.out_degree = 2;
    let mut prng = Xoshiro256StarStar::seed_from_u64(1);
    let mut g: DenseKnnGraph<u32, Nhh> = exhaustive_knn_graph_internal(
      vec![0u32, 1, 2, 3, 4, 5],
      config,
      dist_fn,
      &mut prng,
    )
    .unwrap();
    g.delete(&3, dist_fn, &mut prng).unwrap();
    g.consistency_check().unwrap();
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
    let mut config = mk_config(30);
    config.out_degree = 5;
    let mut prng = Xoshiro256StarStar::seed_from_u64(1);
    let ids = (0u32..16).collect::<Vec<_>>();
    let mut g: DenseKnnGraph<u32, Nhh> =
      exhaustive_knn_graph_internal(ids, config, dist_fn, &mut prng).unwrap();
    g.consistency_check().unwrap();
    g.delete(&7, dist_fn, &mut prng).unwrap();
    g.delete(&15, dist_fn, &mut prng).unwrap();
    g.delete(&0, dist_fn, &mut prng).unwrap();
    g.delete(&1, dist_fn, &mut prng).unwrap();
    g.delete(&2, dist_fn, &mut prng).unwrap();
    g.delete(&3, dist_fn, &mut prng).unwrap();
    g.delete(&4, dist_fn, &mut prng).unwrap();
    g.delete(&5, dist_fn, &mut prng).unwrap();
    g.delete(&6, dist_fn, &mut prng).unwrap();
    g.delete(&8, dist_fn, &mut prng).unwrap();
    g.consistency_check().unwrap();
    let mut prng = Xoshiro256StarStar::seed_from_u64(1);
    g.query(&1, 2, dist_fn, &mut prng).unwrap();
  }

  #[test]
  fn test_use_rrnp() {
    let db: Vec<[i32; 1]> = vec![[1], [2], [3], [10], [11], [12], [6]];
    let dist_fn = &|x: &u32, y: &u32| {
      sq_euclidean_faster(&db[*x as usize], &db[*y as usize])
    };
    let mut config = mk_config(10);
    config.out_degree = 2;
    config.use_rrnp = true;
    let mut prng = Xoshiro256StarStar::seed_from_u64(1);
    let mut g: DenseKnnGraph<u32, Nhh> = exhaustive_knn_graph_internal(
      vec![0u32, 1, 2, 3, 4, 5],
      config,
      dist_fn,
      &mut prng,
    )
    .unwrap();
    g.consistency_check().unwrap();
    g.insert(6, dist_fn, &mut prng).unwrap();
    g.consistency_check().unwrap();
  }
}
