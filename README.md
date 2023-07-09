An implementation of the approximate nearest-neighbor search data structure
and algorithms
described in

> W. -L. Zhao, H. Wang and C. -W. Ngo, "Approximate k-NN Graph
Construction: A Generic Online Approach," in IEEE Transactions on Multimedia,
vol. 24, pp. 1909-1921, 2022, doi: 10.1109/TMM.2021.3073811.

This library is intended to be a flexible foundation for a system that
uses approximate nearest neighbor search.

Interesting features include:
- A generic interface for nearest neighbor search on arbitrary distance
functions.
- Not trait-based. Distance function is a closure, for maximum flexibility.
- Incremental insertion and deletion of elements.
- Memory-efficient. Scales to hundreds of millions or billions of elements on one machine.
- Relatively fast (benchmarks coming soon).

To get started, use the `KnnGraphConfigBuilder` to create a new nearest
neighbor search structure with `Knn::new`. For vectors of floats, you can use
the ordered_float crate to ensure that your floats are not NaN and to
satisfy the trait requirements of this library.

Minimal example:

```rust
extern crate graph_anns;
extern crate ordered_float;
extern crate rand_xoshiro;

use graph_anns::{Knn, KnnGraphConfigBuilder, NN, SearchResults};
use ordered_float::NotNan;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

use std::collections::hash_map::RandomState;

let dist_fn = &|x: &&Vec<NotNan<f32>>, y: &&Vec<NotNan<f32>>| {
let mut sum = 0.0;
for i in 0..x.len() {
  sum += (x[i] - y[i]).powi(2);
}
sum
};

let conf = KnnGraphConfigBuilder::new(5, 3, 1, Default::default())
.use_rrnp(true)
.rrnp_max_depth(2)
.use_lgd(true)
.build();

let mut g: Knn<&Vec<NotNan<f32>>, RandomState> = Knn::new(conf, dist_fn);

let mut prng = Xoshiro256StarStar::seed_from_u64(1);

let example_vec = vec![
NotNan::new(1f32).unwrap(),
NotNan::new(2f32).unwrap(),
NotNan::new(3f32).unwrap(),
NotNan::new(4f32).unwrap(),
];

let query_vec = vec![
  NotNan::new(34f32).unwrap(),
  NotNan::new(5f32).unwrap(),
  NotNan::new(53f32).unwrap(),
  NotNan::new(312f32).unwrap(),
];

g.insert(&example_vec, &mut prng).unwrap();

let SearchResults {
approximate_nearest_neighbors: nearest_neighbors,
..
} = g.query(
&&query_vec,
1,
&mut prng,
).unwrap();
assert_eq!(*nearest_neighbors[0].item, example_vec);
```
