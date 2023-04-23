use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::BuildHasher;

use crate::error::Error;

// TODO: don't store T twice. It's fine in the tests we've been doing where
// T is a u32, but it's not great in general. Store a reference to T in one
// of the two data structures (probably the hashmap?).

/// Maps from the user's chosen ID type to our internal u32 ids that are used
/// within the core search functions to keep things fast and compact.
/// Ideally, we should translate to and from user ids at the edges of
/// performance-critical code. In practice, doing so may be difficult, since the
/// user's distance callback is passed external ids (the user's IDs).
#[derive(Serialize, Deserialize, Debug)]
pub struct IdMapping<T: Eq + std::hash::Hash, S: BuildHasher + Default> {
  capacity: u32,
  next_int_id: u32,
  pub internal_to_external_ids: Vec<Option<T>>,
  #[serde(bound(
    serialize = "HashMap<T, u32, S>: Serialize",
    deserialize = "HashMap<T, u32, S>: Deserialize<'de>"
  ))]
  pub external_to_internal_ids: HashMap<T, u32, S>,
  pub deleted: Vec<u32>,
}

impl<T: Clone + Eq + std::hash::Hash, S: BuildHasher + Default>
  IdMapping<T, S>
{
  pub fn with_capacity_and_hasher(capacity: u32, hash_builder: S) -> Self {
    let mut internal_to_external_ids = Vec::with_capacity(capacity as usize);
    for _ in 0..capacity {
      internal_to_external_ids.push(None);
    }
    let external_to_internal_ids =
      HashMap::with_capacity_and_hasher(capacity as usize, hash_builder);

    let deleted = Vec::<u32>::new();
    IdMapping {
      capacity,
      next_int_id: 0,
      internal_to_external_ids,
      external_to_internal_ids,
      deleted,
    }
  }

  pub fn insert(&mut self, x: &T) -> Result<u32, Error> {
    match self.external_to_internal_ids.get(x) {
      Some(id) => Ok(*id),
      None => {
        let x_int = match self.deleted.pop() {
          None => self.next_int_id,
          Some(i) => i,
        };
        if x_int > self.capacity {
          return Err(Error::CapacityExceeded);
        }
        self.next_int_id += 1;

        // TODO: we are storing two clones of x. Replace with a bidirectional
        // map or something to reduce memory usage.
        self.internal_to_external_ids[x_int as usize] = Some(x.clone());
        self.external_to_internal_ids.insert(x.clone(), x_int);
        Ok(x_int)
      }
    }
  }

  pub fn int_to_ext(&self, x: u32) -> Result<&T, Error> {
    match self.internal_to_external_ids.get(x as usize) {
      None => Err(Error::InternalError(format!("unknown internal id: {}", x))),
      Some(None) => {
        Err(Error::InternalError(format!("unknown external id: {}", x)))
      }
      Some(Some(i)) => Ok(i),
    }
  }

  pub fn ext_to_int(&self, x: &T) -> Result<&u32, Error> {
    match self.external_to_internal_ids.get(x) {
      None => Err(Error::InternalError("unknown external id".to_string())),
      Some(i) => Ok(i),
    }
  }

  pub fn delete(&mut self, x: &T) -> Result<u32, Error> {
    let x_int = *self.ext_to_int(x)?;
    self.deleted.push(x_int);
    self.internal_to_external_ids[x_int as usize] = None;
    self.external_to_internal_ids.remove(x);
    Ok(x_int)
  }
}
