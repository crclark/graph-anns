# run `maturin develop` to build the module and make it available to import
import graph_anns
import numpy as np

CAPACITY=10
OUT_DEGREE=5
NUM_SEARCHERS=3
USE_RRNP=True
RRNP_MAX_DEPTH=2
USE_LGD=True
RAND_SEED=1

if __name__ == '__main__':
  x = graph_anns.NdArrayWithId(0, np.array([0.0,0.0,0.0], dtype=np.float32))
  y = graph_anns.NdArrayWithId(1, np.array([0.1,0.1,0.1], dtype=np.float32))
  z = graph_anns.NdArrayWithId(2, np.array([0.5,0.5,0.5], dtype=np.float32))


  g = graph_anns.PyKnnGraph(CAPACITY, OUT_DEGREE, NUM_SEARCHERS, USE_RRNP, RRNP_MAX_DEPTH, USE_LGD, RAND_SEED)

  g.insert(x)
  g.insert(y)
  g.insert(z)

  results = g.query(x, 3)

  for r in results:
    print(r.id, r.array)

  results = g.query(y, 3)

  for r in results:
    print(r.id, r.array)

  results = g.query(z, 3)

  for r in results:
    print(r.id, r.array)
