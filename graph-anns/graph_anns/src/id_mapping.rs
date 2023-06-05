use serde::de::{MapAccess, Visitor};
use serde::ser::{SerializeSeq, SerializeStruct};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};
use std::sync::Arc;

use crate::error::Error;

use std::marker::PhantomData;
use std::{fmt, mem};

// TODO: make IdMapping an enum, so that users can choose whether to use
// the space-optimized version that uses Arc to store only one copy of
// each external ID, or the version that uses a copy of each external ID
// for each internal ID, which is faster because it avoids one extra pointer
// dereference in the inner loop of the graph algorithm.

#[derive(Clone)]
pub struct IdMapping<T: Eq + std::hash::Hash, S: BuildHasher + Default> {
  capacity: usize,
  next_int_id: u32,
  // TODO: make these private with proper abstractions
  pub internal_to_external_ids: Vec<Option<Arc<T>>>,
  pub external_to_internal_ids: HashMap<Arc<T>, u32, S>,
  pub deleted: Vec<u32>,
}

impl<T: Clone + Eq + std::hash::Hash, S: BuildHasher + Default>
  IdMapping<T, S>
{
  pub fn ext_int_iter(&self) -> impl Iterator<Item = (&T, u32)> {
    self
      .external_to_internal_ids
      .iter()
      .map(|(x, i)| (x.as_ref(), *i))
  }

  pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
    let mut internal_to_external_ids = Vec::with_capacity(capacity);
    for _ in 0..capacity {
      internal_to_external_ids.push(None);
    }
    let external_to_internal_ids =
      HashMap::with_capacity_and_hasher(capacity, hash_builder);

    let deleted = Vec::new();
    IdMapping {
      capacity,
      next_int_id: 0,
      internal_to_external_ids,
      external_to_internal_ids,
      deleted,
    }
  }

  pub fn insert(&mut self, x: T) -> Result<u32, Error> {
    let x = Arc::new(x);
    match self.external_to_internal_ids.get(&x) {
      Some(id) => Ok(*id),
      None => {
        let x_int = match self.deleted.pop() {
          None => self.next_int_id,
          Some(i) => i,
        };
        if x_int as usize >= self.capacity {
          return Err(Error::CapacityExceeded);
        }
        self.next_int_id += 1;

        self.external_to_internal_ids.insert(x.clone(), x_int);
        self.internal_to_external_ids[x_int as usize] = Some(x);
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

  pub fn ext_to_int(&self, x: &T) -> Result<u32, Error> {
    match self.external_to_internal_ids.get(x) {
      None => Err(Error::InternalError("unknown external id".to_string())),
      Some(i) => Ok(*i),
    }
  }

  pub fn delete(&mut self, x_int: u32) -> Result<u32, Error> {
    let x =
      mem::replace(&mut self.internal_to_external_ids[x_int as usize], None);
    let x = x.ok_or_else(|| {
      Error::InternalError(format!("unknown external id: {}", x_int))
    })?;
    self.deleted.push(x_int);
    self.external_to_internal_ids.remove(x.as_ref());
    Ok(x_int)
  }

  pub fn consume_and_get_exts(mut self) -> Result<Vec<T>, Error> {
    self.external_to_internal_ids.clear();

    self
      .internal_to_external_ids
      .drain(..)
      .flatten()
      .map(|x| match Arc::<T>::try_unwrap(x) {
        Ok(x) => Ok(x),
        Err(_) => Err(Error::InternalError(
          "internal_to_external_ids still has references".to_string(),
        )),
      })
      .collect()
  }

  pub fn len(&self) -> usize {
    self.external_to_internal_ids.len()
  }
}

impl<T, S> Serialize for IdMapping<T, S>
where
  T: Clone + Eq + Hash + Serialize,
  S: BuildHasher + Default,
{
  fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
  where
    Ser: serde::Serializer,
  {
    let mut state = serializer.serialize_struct("IdMap", 3)?;
    state.serialize_field("capacity", &self.capacity)?;
    state.serialize_field("next_int_id", &self.next_int_id)?;
    state.serialize_field(
      "internal_to_external_ids",
      &SerializeSeqAsPairs(&self.internal_to_external_ids),
    )?;
    state.end()
  }
}

struct SerializeSeqAsPairs<'a, T>(&'a Vec<Option<Arc<T>>>);

impl<'a, T: Serialize> Serialize for SerializeSeqAsPairs<'a, T> {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    let mut seq = serializer.serialize_seq(None)?;
    for (i, opt) in self.0.iter().enumerate() {
      if let Some(rc) = opt {
        seq.serialize_element(&(i as u32, &**rc))?;
      }
    }
    seq.end()
  }
}

impl<'de, T, S> Deserialize<'de> for IdMapping<T, S>
where
  T: Clone + Eq + Hash + Deserialize<'de>,
  S: BuildHasher + Default,
{
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    deserializer.deserialize_struct(
      "IdMap",
      &["capacity", "next_int_id", "internal_to_external_ids"],
      IdMapVisitor(PhantomData),
    )
  }
}

struct IdMapVisitor<T: Clone + Eq + Hash, S: BuildHasher + Default>(
  PhantomData<fn() -> IdMapping<T, S>>,
);

impl<'de, T, S> Visitor<'de> for IdMapVisitor<T, S>
where
  T: Clone + Eq + Hash + Deserialize<'de>,
  S: BuildHasher + Default,
{
  type Value = IdMapping<T, S>;

  fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
    formatter.write_str("struct IdMap")
  }

  fn visit_map<V>(self, mut map: V) -> Result<IdMapping<T, S>, V::Error>
  where
    V: MapAccess<'de>,
  {
    #![allow(clippy::type_complexity)]
    let mut capacity = None;
    let mut next_int_id = None;
    let mut i_to_e_and_e_to_i_and_deleted: Option<(
      Vec<Option<Arc<T>>>,
      HashMap<Arc<T>, u32, _>,
      Vec<u32>,
    )> = None;
    while let Some(key) = map.next_key()? {
      match key {
        "capacity" => {
          if capacity.is_some() {
            return Err(serde::de::Error::duplicate_field("capacity"));
          }
          capacity = Some(map.next_value()?);
        }
        "next_int_id" => {
          if next_int_id.is_some() {
            return Err(serde::de::Error::duplicate_field("next_int_id"));
          }
          next_int_id = Some(map.next_value()?);
        }
        "internal_to_external_ids" => {
          if i_to_e_and_e_to_i_and_deleted.is_some() {
            return Err(serde::de::Error::duplicate_field(
              "internal_to_external_ids",
            ));
          }

          let pairs: Vec<(u32, T)> = map.next_value()?;
          let mut i_to_e = Vec::with_capacity(pairs.len());
          let mut e_to_i = HashMap::default();

          let mut deleted = Vec::new();
          for (i, t) in pairs.into_iter() {
            let rc = Arc::new(t);
            while i_to_e.len() < (i as usize) {
              deleted.push(i);
              i_to_e.push(None);
            }
            i_to_e.push(Some(Arc::clone(&rc)));
            e_to_i.insert(rc, i);
          }

          i_to_e_and_e_to_i_and_deleted = Some((i_to_e, e_to_i, deleted));
        }
        _ => {
          return Err(serde::de::Error::unknown_field(
            key,
            &["capacity", "next_int_id", "internal_to_external_ids"],
          ))
        }
      }
    }

    let capacity =
      capacity.ok_or_else(|| serde::de::Error::missing_field("capacity"))?;
    let next_int_id = next_int_id
      .ok_or_else(|| serde::de::Error::missing_field("next_int_id"))?;
    let (internal_to_external_ids, external_to_internal_ids, deleted) =
      i_to_e_and_e_to_i_and_deleted.ok_or_else(|| {
        serde::de::Error::missing_field("internal_to_external_ids")
      })?;

    Ok(IdMapping {
      capacity,
      next_int_id,
      internal_to_external_ids,
      external_to_internal_ids,
      deleted,
    })
  }
}

#[cfg(test)]
mod tests {
  #![allow(clippy::unwrap_used)]
  extern crate serde_json;

  use super::*;
  use std::collections::hash_map::RandomState;

  #[test]
  fn test_insert() {
    let mut map: IdMapping<String, RandomState> =
      IdMapping::with_capacity_and_hasher(10, RandomState::new());
    map.insert("test1".to_string()).unwrap();
    assert!(map.ext_to_int(&"test1".to_string()).is_ok());
  }

  #[test]
  fn repeated_insert() {
    let mut map: IdMapping<String, RandomState> =
      IdMapping::with_capacity_and_hasher(10, RandomState::new());
    let result1 = map.insert("test1".to_string()).unwrap();
    let result2 = map.insert("test1".to_string()).unwrap();
    assert_eq!(result1, result2);
  }

  #[test]
  fn insert_delete_reuse_id() {
    let mut map: IdMapping<String, RandomState> =
      IdMapping::with_capacity_and_hasher(10, RandomState::new());
    let result1 = map.insert("test1".to_string()).unwrap();
    map.delete(result1).unwrap();
    let result2 = map.insert("test1".to_string()).unwrap();
    assert_eq!(result1, result2);
  }

  #[test]
  fn serde_round_trip() {
    // Create IdMap and insert some values
    let mut map: IdMapping<String, RandomState> =
      IdMapping::with_capacity_and_hasher(10, RandomState::new());
    map.insert("test1".to_string()).unwrap();
    map.insert("test2".to_string()).unwrap();
    map.insert("test3".to_string()).unwrap();

    // Serialize to a JSON string
    let serialized = serde_json::to_string(&map).unwrap();

    // Deserialize back into an IdMap
    let deserialized: IdMapping<String, RandomState> =
      serde_json::from_str(&serialized).unwrap();

    // Check that the deserialized map contains the values
    assert_eq!(
      map.ext_to_int(&"test1".to_string()).unwrap(),
      deserialized.ext_to_int(&"test1".to_string()).unwrap()
    );
    assert_eq!(
      map.ext_to_int(&"test2".to_string()).unwrap(),
      deserialized.ext_to_int(&"test2".to_string()).unwrap()
    );
    assert_eq!(
      map.ext_to_int(&"test3".to_string()).unwrap(),
      deserialized.ext_to_int(&"test3".to_string()).unwrap()
    );
  }
}
