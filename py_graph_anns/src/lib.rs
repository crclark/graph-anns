extern crate graph_anns;
extern crate nohash_hasher;
extern crate numpy;
extern crate pyo3;
extern crate rand;
extern crate rayon;

use graph_anns::{Error, Knn, KnnGraphConfigBuilder, SearchResults, NN};
use nohash_hasher::NoHashHasher;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use rand::RngCore;
use rayon::result;
use std::hash::BuildHasherDefault;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::RwLock;

use rayon::prelude::*;

#[pyclass(get_all, frozen)]
struct NdArrayWithId {
  id: usize,
  array: Py<PyArray1<f32>>,
}

// TODO: most users are actually going to want this interface, but
// but the docs and signatures we have today don't make it obvious
// that this is even safe! We should probably modify the API so that
// the user must provide their own id for each vector, even if that
// makes certain use cases slower.
// implement Clone, ord, eq, hash using only the id field
#[pymethods]
impl NdArrayWithId {
  #[new]
  fn new(id: usize, array: Py<PyArray1<f32>>) -> Self {
    NdArrayWithId { id, array }
  }
}

impl NdArrayWithId {
  /// Converts the Python NumPy array to a Rust Vec<f32> without using `.extract()`
  fn to_internal(&self, py: Python) -> NdArrayWithIdInternal {
    // Access the NumPy array directly
    let array = self.array.as_ref(py);
    let array = array.readonly();

    // Convert the NumPy array to a Vec<f32>
    let rust_vec: Vec<f32> = array.as_array().to_vec();

    NdArrayWithIdInternal {
      id: self.id,
      array: rust_vec,
    }
  }
}

impl Clone for NdArrayWithId {
  fn clone(&self) -> Self {
    NdArrayWithId {
      id: self.id,
      array: self.array.clone(),
    }
  }
}

/// Internal version of NdArrayWithId that is unwrapped from the Py<> to
/// avoid extra pointer indirection in the graph.
#[derive(Debug)]
struct NdArrayWithIdInternal {
  id: usize,
  array: Vec<f32>,
}

impl Clone for NdArrayWithIdInternal {
  fn clone(&self) -> Self {
    NdArrayWithIdInternal {
      id: self.id,
      array: self.array.clone(),
    }
  }
}

impl PartialEq for NdArrayWithIdInternal {
  fn eq(&self, other: &Self) -> bool {
    self.id == other.id
  }
}

impl Eq for NdArrayWithIdInternal {}

impl Hash for NdArrayWithIdInternal {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    self.id.hash(state);
  }
}

impl PartialOrd for NdArrayWithIdInternal {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    Some(self.cmp(other))
  }
}

impl Ord for NdArrayWithIdInternal {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    self.id.cmp(&other.id)
  }
}

type Nhh = BuildHasherDefault<NoHashHasher<u32>>;

#[pyclass]
struct PyKnnGraph {
  inner: Arc<Knn<'static, NdArrayWithIdInternal, Nhh>>,
  prng: rand_xoshiro::Xoshiro256StarStar,
}

#[pymethods]
impl PyKnnGraph {
  #[new]
  fn new(
    capacity: u32,
    out_degree: u8,
    num_searchers: u32,
    use_rrnp: bool,
    rrnp_max_depth: u32,
    use_lgd: bool,
    rand_seed: u64,
  ) -> Self {
    let config = KnnGraphConfigBuilder::new(
      capacity,
      out_degree,
      num_searchers,
      Default::default(),
    )
    .use_rrnp(use_rrnp)
    .rrnp_max_depth(rrnp_max_depth)
    .use_lgd(use_lgd)
    .optimize_for_small_type(false)
    .build();
    let dist_fn = &|x: &NdArrayWithIdInternal, y: &NdArrayWithIdInternal| {
      let mut sum: f32 = 0.0;
      for i in 0..x.array.len() {
        sum += (x.array[i] - y.array[i]).powi(2);
      }
      sum
    };
    let mut prng =
      <rand_xoshiro::Xoshiro256StarStar as rand::SeedableRng>::seed_from_u64(
        rand_seed,
      );
    let knn = Knn::new(config, dist_fn);
    PyKnnGraph {
      inner: Arc::new(knn),
      prng,
    }
  }

  fn insert(&mut self, py: Python, x: NdArrayWithId) -> PyResult<()> {
    Arc::get_mut(&mut self.inner)
      .unwrap()
      .insert(x.to_internal(py), &mut self.prng)
      .unwrap();
    Ok(())
  }

  fn query(
    &mut self,
    py: Python,
    q: NdArrayWithId,
    max_results: usize,
  ) -> PyResult<Vec<NdArrayWithId>> {
    let results = self
      .inner
      .query(&q.to_internal(py), max_results, &mut self.prng)
      // TODO: delete all these unwraps
      .unwrap();
    let mut nearest_neighbors = Vec::new();
    for x in results.approximate_nearest_neighbors {
      nearest_neighbors.push(NdArrayWithId {
        id: x.item.id,
        array: x.item.array.into_pyarray(py).into(),
      });
    }
    println!(
      "consistency check: {:?}",
      self.inner.debug_consistency_check()
    );
    self.inner.debug_print();
    Ok(nearest_neighbors)
  }

  fn delete(&mut self, py: Python, id: usize) -> PyResult<()> {
    Arc::get_mut(&mut self.inner)
      .unwrap()
      .delete(
        NdArrayWithIdInternal {
          id,
          array: Vec::new(),
        },
        &mut self.prng.clone(),
      )
      .unwrap();
    self.inner.debug_print();
    Ok(())
  }
}

#[pyclass]
struct PyKnnGraphMultithreaded {
  inners: Vec<RwLock<Knn<'static, NdArrayWithIdInternal, Nhh>>>,
  prng: rand_xoshiro::Xoshiro256StarStar,
}

#[pymethods]
impl PyKnnGraphMultithreaded {
  #[new]
  fn new(
    capacity: u32,
    out_degree: u8,
    num_searchers: u32,
    use_rrnp: bool,
    rrnp_max_depth: u32,
    use_lgd: bool,
    rand_seed: u64,
    num_threads: u32,
  ) -> Self {
    let config = KnnGraphConfigBuilder::new(
      capacity,
      out_degree,
      num_searchers,
      Default::default(),
    )
    .use_rrnp(use_rrnp)
    .rrnp_max_depth(rrnp_max_depth)
    .use_lgd(use_lgd)
    .optimize_for_small_type(false)
    .build();
    let dist_fn = &|x: &NdArrayWithIdInternal, y: &NdArrayWithIdInternal| {
      let mut sum: f32 = 0.0;
      for i in 0..x.array.len() {
        sum += (x.array[i] - y.array[i]).powi(2);
      }
      sum
    };
    let prng =
      <rand_xoshiro::Xoshiro256StarStar as rand::SeedableRng>::seed_from_u64(
        rand_seed,
      );

    let individual_capacity = capacity / num_threads + capacity % num_threads;
    let mut inners = Vec::new();
    for _ in 0..num_threads {
      let config = KnnGraphConfigBuilder::from(config.clone())
        .capacity(individual_capacity)
        .build();
      inners.push(RwLock::new(Knn::new(config, dist_fn)));
    }
    PyKnnGraphMultithreaded { inners, prng }
  }

  fn insert(&mut self, py: Python, x: NdArrayWithId) -> PyResult<()> {
    let i = self.prng.next_u32() % self.inners.len() as u32;
    let mut g = self.inners[i as usize].write().unwrap();
    g.insert(x.to_internal(py), &mut self.prng).unwrap();
    Ok(())
  }

  fn query(
    &mut self,
    py: Python,
    q: NdArrayWithId,
    max_results: usize,
  ) -> PyResult<Vec<NdArrayWithId>> {
    let q_internal = q.to_internal(py);
    let results = (0..self.inners.len())
      .into_par_iter()
      .map(|i| {
        self.inners[i].read().unwrap().query(
          &q_internal,
          max_results,
          &mut rand::thread_rng(),
        )
      })
      .try_reduce(SearchResults::<NdArrayWithIdInternal>::default, |acc, e| {
        Ok(acc.merge(&e))
      });

    match results {
      Ok(results) => {
        let mut nearest_neighbors = Vec::new();
        for x in results.approximate_nearest_neighbors {
          nearest_neighbors.push(NdArrayWithId {
            id: x.item.id,
            array: x.item.array.into_pyarray(py).into(),
          });
        }
        Ok(nearest_neighbors)
      }
      Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
        "Error: {:?}",
        e
      ))),
    }
  }

  fn delete(&mut self, id: usize) -> PyResult<()> {
    let result = (0..self.inners.len())
      .into_par_iter()
      .map(|i| {
        // TODO: don't ignore errors
        self.inners[i].write().unwrap().delete(
          NdArrayWithIdInternal {
            id,
            array: Vec::new(),
          },
          &mut self.prng.clone(),
        )
      })
      .find_any(|r| r.is_err());
    // TODO: even better if we report back all errors instead of one of them
    match result {
      Some(Err(e)) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
        format!("Error: {:?}", e),
      )),
      _ => Ok(()),
    }
  }
}

#[pymodule]
fn py_graph_anns(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
  m.add_class::<NdArrayWithId>()?;
  m.add_class::<PyKnnGraph>()?;
  m.add_class::<PyKnnGraphMultithreaded>()?;
  Ok(())
}
