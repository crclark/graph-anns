use serde::{Deserialize, Serialize};
use soa_derive::StructOfArray;

/// Represents a directed edge in a nearest neighbor graph.
#[derive(StructOfArray, Deserialize, Serialize)]
#[soa_derive(Deserialize, Serialize)]
pub struct Edge {
  pub to: u32,
  pub distance: f32,
  pub crowding_factor: u8,
}
