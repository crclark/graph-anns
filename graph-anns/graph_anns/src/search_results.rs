use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::HashSet;
use std::time::Duration;

/// A single search result.
#[derive(Debug, Clone, Copy)]
pub struct SearchResult<T> {
  /// The user-provided item.
  pub item: T,
  /// The internal identifier for this item within the graph. This will be None
  /// if the graph is small and using the brute force algorithm (internally, we
  /// convert from brute force to a nearest-neighbor graph after a small number
  /// of items have been inserted).
  pub internal_id: Option<u32>,
  /// The distance from the query point to this item.
  pub dist: f32,

  pub(crate) search_root_ancestor: u32,
  pub(crate) search_depth: u32,
}

impl<T> SearchResult<T> {
  pub(crate) fn new(
    item: T,
    internal_id: Option<u32>,
    dist: f32,
    search_root_ancestor: u32,
    search_depth: u32,
  ) -> SearchResult<T> {
    Self {
      item,
      internal_id,
      dist,
      search_root_ancestor,
      search_depth,
    }
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

/// Statistics about a single search call. These are mostly used for debugging,
/// testing, and benchmarking, but may be of interest to advanced users.
#[derive(Debug, Clone, Copy)]
pub struct SearchStats {
  /// Distance between query point and nearest neighbor found
  pub nearest_neighbor_distance: f32,
  /// Number of times the user's distance function was called.
  pub num_distance_computations: usize,
  /// The distance from the starting search point that was nearest to the query
  /// point to the nearest neighbor found.
  pub distance_from_nearest_starting_point: f32,
  /// The distance from the starting search point that was farthest from the query
  /// point to the nearest neighbor found.
  pub distance_from_farthest_starting_point: f32,
  /// Total duration of the search call.
  pub search_duration: Duration,
  /// The largest distance moved towards the target point by a single hop from
  /// one node to another node adjacent to it.
  pub largest_distance_improvement_single_hop: f32,
  /// The smallest distance moved towards the target point by a single hop from
  /// one node to another node adjacent to it. If this is negative, it means
  /// we moved away from the target point!
  pub smallest_distance_improvement_single_hop: f32,
  /// The total number of hops from the starting point to the nearest neighbor
  /// that was found.
  pub nearest_neighbor_path_length: usize,
  /// The total number of nodes visited during the search.
  pub num_visited: usize,
  // TODO
  // /// The distance from the nearest neighbor to the starting point that the searcher that found it started from.
  // pub distance_from_nearest_neighbor_to_its_starting_point: f32,
}

impl SearchStats {
  fn merge(&self, other: &SearchStats) -> SearchStats {
    SearchStats {
      nearest_neighbor_distance: self
        .nearest_neighbor_distance
        .min(other.nearest_neighbor_distance),
      num_distance_computations: self.num_distance_computations
        + other.num_distance_computations,
      distance_from_nearest_starting_point: self
        .distance_from_nearest_starting_point
        .min(other.distance_from_nearest_starting_point),
      distance_from_farthest_starting_point: self
        .distance_from_farthest_starting_point
        .max(other.distance_from_farthest_starting_point),
      search_duration: self.search_duration + other.search_duration,
      largest_distance_improvement_single_hop: self
        .largest_distance_improvement_single_hop
        .max(other.largest_distance_improvement_single_hop),
      smallest_distance_improvement_single_hop: self
        .smallest_distance_improvement_single_hop
        .min(other.smallest_distance_improvement_single_hop),
      nearest_neighbor_path_length: if self.nearest_neighbor_distance
        < other.nearest_neighbor_distance
      {
        self.nearest_neighbor_path_length
      } else {
        other.nearest_neighbor_path_length
      },
      num_visited: self.num_visited + other.num_visited,
    }
  }
}

/// The results of a search.
#[derive(Debug, Clone)]
pub struct SearchResults<T> {
  /// Results of the search, in order of increasing distance from the query.
  pub approximate_nearest_neighbors: Vec<SearchResult<T>>,
  /// All nodes visited during the search. Most users won't need this info.
  pub visited_nodes: Vec<SearchResult<T>>,
  /// The same info as visited_nodes, but stored as a map from internal id.
  /// Used for some internal operations; probably not useful for users. This
  /// will be empty if the data structure is small (using the brute force
  /// algorithm), because the brute force algorithm doesn't assign internal ids.
  pub visited_nodes_distances_to_q: HashMap<u32, (T, f32)>,
  /// Statistics about the execution of the search.
  /// This will be None for small graphs, because the brute force algorithm
  /// doesn't collect statistics.
  pub search_stats: Option<SearchStats>,
}

impl<T: Clone + Eq + std::hash::Hash> SearchResults<T> {
  /// Merge two SearchResults objects. This is useful if you want to search
  /// multiple graphs in parallel, and then merge the results.
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
      .sort_by(|a, b| a.dist.total_cmp(&b.dist));
    merged.visited_nodes.extend(other.visited_nodes.clone());
    merged
      .visited_nodes
      .sort_by(|a, b| a.internal_id.cmp(&b.internal_id));
    merged.visited_nodes.dedup_by_key(|x| x.internal_id);
    merged
      .visited_nodes_distances_to_q
      .extend(other.visited_nodes_distances_to_q.clone());
    merged.search_stats = match (&self.search_stats, &other.search_stats) {
      (Some(s1), Some(s2)) => Some(s1.merge(s2)),
      (Some(s1), None) => Some(*s1),
      (None, Some(s2)) => Some(*s2),
      (None, None) => None,
    };
    merged
  }
}

impl<T> Default for SearchResults<T> {
  fn default() -> Self {
    Self {
      approximate_nearest_neighbors: Vec::new(),
      visited_nodes: Vec::new(),
      visited_nodes_distances_to_q: HashMap::new(),
      search_stats: None,
    }
  }
}
