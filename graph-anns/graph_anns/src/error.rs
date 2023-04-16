//! Defines the error type for this crate.

use thiserror::Error;

/// The error type for this crate.
#[derive(Error, Debug)]
pub enum Error {
  /// The user tried to insert an item with an ID that is already in the index.
  #[error("The item with ID {0} is already in the index.")]
  DuplicateId(u32),

  /// The user tried to insert an item but the index is full.
  #[error("The graph is full.")]
  CapacityExceeded,

  /// The user tried to create an exhaustive k-nn graph, but the requested out
  /// degree is too large for the size of the input.
  #[error("The requested out degree {0} is too large for the input size {1}.")]
  OutDegreeTooLarge(u8, usize),

  /// Internal invariant violation, which is most likely a bug in this library.
  #[error("An internal error occurred: {0}. This is probably a bug. Please report it at TODO")]
  InternalError(String),
  // Add more variants here as needed
}
