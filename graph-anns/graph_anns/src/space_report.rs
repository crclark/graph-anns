/// Report on the size of the internal data structures.
/// This is useful for debugging and testing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpaceReport {
  /// The number of items in the internal to external id mapping.
  pub mapping_int_ext_len: usize,
  /// The capacity of the internal to external id mapping.
  pub mapping_int_ext_capacity: usize,
  /// The number of items in the external to internal id mapping.
  pub mapping_ext_int_len: usize,
  /// The capacity of the external to internal id mapping.
  pub mapping_ext_int_capacity: usize,
  /// The number of items in the deleted id mapping.
  pub mapping_deleted_len: usize,
  /// The capacity of the deleted id mapping.
  pub mapping_deleted_capacity: usize,
  /// The number of items in the edges vector.
  pub edges_vec_len: usize,
  /// The capacity of the edges vector.
  pub edges_vec_capacity: usize,
  /// The number of items in the backpointers vector.
  pub backpointers_len: usize,
  /// The capacity of the backpointers vector.
  pub backpointers_capacity: usize,
  /// The total number of backpointers.
  pub backpointers_sets_sum_len: usize,
  /// The total capacity of the backpointer sets.
  pub backpointers_sets_sum_capacity: usize,
  /// The smallest number of backpointers for a single node.
  pub backpointers_smallest_set_len: usize,
  /// The largest number of backpointers for a single node.
  pub backpointers_largest_set_len: usize,
  /// The total memory used by the backpointer sets.
  pub backpointers_sets_mem_used: usize,
  /// The number of reciprocal edges in the directed graph.
  pub num_reciprocated_edges: usize,
}

impl SpaceReport {
  /// Combine the SpaceReports of two graphs to report the total space used
  /// by both.
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
      backpointers_smallest_set_len: std::cmp::min(
        self.backpointers_smallest_set_len,
        other.backpointers_smallest_set_len,
      ),
      backpointers_largest_set_len: std::cmp::max(
        self.backpointers_largest_set_len,
        other.backpointers_largest_set_len,
      ),
      num_reciprocated_edges: self.num_reciprocated_edges
        + other.num_reciprocated_edges,
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
      backpointers_smallest_set_len: usize::MAX,
      backpointers_largest_set_len: 0,
      num_reciprocated_edges: 0,
    }
  }
}
