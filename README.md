# metric-indexing

I want to take arbitrary objects with arbitrary metrics defined on them and
store them in FDB in a way that allows for efficient nearest-neighbor queries.

There are, of course, many metric trees, such as cover trees, vantage point trees, slim-trees, split-trees, etc., but they are designed (mostly) on the assumption that the tree will be stored in memory. I need something that is amenable to storage in FDB.

## Current status

The Haskell code in the root of this directory is abandoned. I am currently focusing on the `graph-anns` directory, which contains Rust code.

Projects:

1. Root dir, Haskell -- Hilbert curve-based code. Abandoned because Hilbert curves are a huge pain.
2. `bruteforce/` dir, C++ -- brute force search through entire datasets. *Also includes a utility to convert a [test corpus](http://corpus-texmex.irisa.fr/) into a plain old C array which can be mmaped*.
3. `graph-anns/` dir, Rust -- an implementation of something similar to [Approximate k-NN graph construction: a generic online approach](https://arxiv.org/pdf/1804.03032), with several extensions. This is the actively developed project. It also contains utilities to read the mmaped format created by `bruteforce/`.

## Requirements

Let's be absurdly ambitious. What fun is a side project if it's not?

1. Scale to at least 100M points. Ideally billions to compete with state-of-the-art (Faiss and navigating spreading-out graph).
2. Support arbitrary metric spaces.
3. Incremental construction: insertion and deletion operations.
4. Approximate or exact k-nearest neighbor queries. Nice to have: radius limits.

### Maybe nice to have; unclear if desirable

1. Indexes stored in secondary memory. State-of-the-art uses in-memory representations, with the dataset stored externally and read once at index rebuild time.

## Open questions

1. The papers I am reading give performance metrics with a "precision" x-axis. How do they adjust precision to produce those graphs? PARTIAL ANSWER: See http://corpus-texmex.irisa.fr/ for a definition of recall@n. Presumably precision is defined the same way.
2. What datasets should we benchmark on? ANSWER: Let's start with http://corpus-texmex.irisa.fr/ .
3. With the ubiquity of vector space embeddings via neural net, do we even care about arbitrary metric spaces these days?
4. If we drop the updatability requirement, could we play with GHC's compact regions feature?
5. Most metric indexing techniques use "pivots" -- special points that are used with the triangle inequality to prune the search space. What pivot selection algo should we use?
6. Is using FDB with billion-scale data feasible for a side project? Yes, only a few hours to insert a billion K/Vs, and reading is faster.

## Brute-force baseline experiment on TexMex corpus

To build intuition for nearest-neighbor search and the speed of modern NVMe disks,
I am doing a baseline experiment with the corpus from http://corpus-texmex.irisa.fr/.

The data format of each file is almost a plain old array, except that it is interleaved with an arguably useless integer telling us how many elements are in the next vector. However, every vector is the same length, so we can just pass the vector length as an arg to our program. To improve performance and shrink the file a little, we wrote a [little program](./convert_texmex_to_serialized_array.c) that converts the file into a serialized array that can be directly `mmap`ed for optimal access speed. **Note that this requires a little-endian architecture**.

Machine stats: AMD 1950X Threadripper (16 cores, 32 threads), 128GiB ECC RAM, Samsung 970 Pro NVMe.

There are a few performance considerations for this experiment:

1. The distance comparison function is quite inexpensive. There are also cheaper approximations we can use, too. This might not be true in the more general case we want to tackle later.
2. If we ignore IO and memory bandwidth, doing exhaustive search for the nearest neighbor of a single query vector will take `n*(distance_compute + compare_distance)` = `n*(3*d + 1)` (euclidean distance is one subtract, one multiply, one addition for each dimension) operations. For a billion 128-d query vectors, that's 385 billion operations. The 1950X can do about 1.2 TFLOPs, which means that it can theoretically do an exhaustive search in about 300ms. We can see why Faiss's design emphasizes compact binary representations of vectors that are amenable to fast intrinsics -- if you can keep a lot of the dataset as close to the CPU as possible by using a more compact encoding, you can increase throughput even further.
3. Given the above thoughts and considerations, we can see that if we use brute force, the best throughput for nearest-neighbor queries we could hope for is maybe 3 QPS.

### First results

On commit dc0c7bcff7bd81b64cab9fde61ae4f8bf48f60f3

```
time ./search b 1 128 2 /mnt/970pro/anns/bigann_base.bvecs_array /mnt/970pro/anns/bigann_query.bvecs_array_one_point
```

takes about 30 seconds to mmap the file, then about 100 seconds to search for the 2 nearest neighbors of one point.

Running it repeatedly, it gets faster (as expected since the file is still in the page cache). The fastest output I get from `time` is 23 seconds. Note that this is single-threaded at this point. If we naively assume perfect parallelism, dividing by 32 is about 720ms to search the entire thing... not too far from our back-of-the-napkin estimate! Or going by my estimate above of 385 billion integer ops, that's 16.7 billion ops/second on one core.

... And then I realized that constant overflows in the arithmetic were giving us
bad results, and fixed it by casting to float first, which caused our speed to degrade to 158 seconds, meaning that my estimate was off by an order of magnitude. Drat.

New command:

```
time ./search b 1 128 1000 /mnt/970pro/anns/bigann_base.bvecs_array /mnt/970pro/anns/bigann_query.bvecs_array_one_point /mnt/970pro/anns/gnd/idx_1000M.ivecs_array /mnt/970pro/anns/gnd/dis_1000M.fvecs_array
```

#### Rust version

See commit d15eef7f5d32a0869e80596c1b90018c1864d352 and graph-anns/ to see the Rust translation of the above brute force code. It's equal in speed to C++. Interestingly, I discovered that MAP_POPULATE is significantly slower than omitting it. That makes no sense to me at all... wouldn't we avoid pipeline stalls if we pre-populate all the pages?

#### Rust version no MAP_POPULATE

`/usr/bin/time -v cargo run --release`

```
	Command being timed: "cargo run --release"
	User time (seconds): 292.56
	System time (seconds): 59.18
	Percent of CPU this job got: 1320%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:26.63
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 115413560
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 63
	Minor (reclaiming a frame) page faults: 2176874
	Voluntary context switches: 206329
	Involuntary context switches: 45831
	Swaps: 0
	File system inputs: 60900952
	File system outputs: 0
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0
```

#### Rust version with MAP_POPULATE

```
	Command being timed: "cargo run --release"
	User time (seconds): 298.30
	System time (seconds): 69.01
	Percent of CPU this job got: 509%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 1:12.04
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 115583888
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 254
	Minor (reclaiming a frame) page faults: 3591142
	Voluntary context switches: 453967
	Involuntary context switches: 38958
	Swaps: 0
	File system inputs: 176273032
	File system outputs: 0
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0
```

#### Rust version with NO MAP_POPULATE, returning i32 distances

```
	Command being timed: "cargo run --release"
	User time (seconds): 99.02
	System time (seconds): 9.46
	Percent of CPU this job got: 1797%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:06.03
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 125002900
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 0
	Minor (reclaiming a frame) page faults: 1957946
	Voluntary context switches: 272
	Involuntary context switches: 11118
	Swaps: 0
	File system inputs: 0
	File system outputs: 0
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0
```

#### Rust version with MAP_POPULATE, f32 distances, using rayon

```
	Command being timed: "cargo run --release"
	User time (seconds): 334.86
	System time (seconds): 11.36
	Percent of CPU this job got: 1724%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:20.07
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 124980652
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 6332
	Minor (reclaiming a frame) page faults: 3298738
	Voluntary context switches: 36411
	Involuntary context switches: 64312
	Swaps: 0
	File system inputs: 14619456
	File system outputs: 0
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0

```

#### Rust version with NO MAP_POPULATE, f32 distances, using rayon

```
	Command being timed: "cargo run --release"
	User time (seconds): 327.52
	System time (seconds): 64.29
	Percent of CPU this job got: 2556%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:15.32
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 125117508
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 1354
	Minor (reclaiming a frame) page faults: 2875502
	Voluntary context switches: 8809
	Involuntary context switches: 101449
	Swaps: 0
	File system inputs: 6957008
	File system outputs: 0
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0

```

#### Rust version with NO MAP_POPULATE, i32 distances, using rayon

```
	Command being timed: "numactl --interleave=all ./target/release/graph-anns"
	User time (seconds): 112.84
	System time (seconds): 6.81
	Percent of CPU this job got: 2252%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:05.31
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 125031892
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 0
	Minor (reclaiming a frame) page faults: 1965918
	Voluntary context switches: 3091
	Involuntary context switches: 15299
	Swaps: 0
	File system inputs: 0
	File system outputs: 0
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0
```

Conclusions:

1. MAP_POPULATE gives MANY more major and minor page faults. However, would it be faster if we are doing multiple passes over the data (which we will be in the future)?
2. Rayon is almost as fast as the handwritten thread code (I forgot to record it, but I was able to hit 4.8 seconds with i32 distances and numactl --interleave=all). This means a few things:
    1. The reason we don't get full 3200% CPU utilization is almost certainly because of
       memory bandwidth limits on the Threadripper, not because some threads are finishing early and then sitting idle (since rayon is work-stealing and would have fixed that). Should have bought the real server CPU!
    2. We can safely adopt rayon and get more flexible and easy-to-modify parallel code, at the cost of added verbosity for reduce operations.
    3. We can safely ignore rayon and get a small speedup for a reduce operation, for this toy use case.

#### MAP_POPULATE vs no MAP_POPULATE on random, repeated access pattern

Randomly initializing a NN-Descent data structure, with no MAP_POPULATE and one thread, it takes 245 seconds to initialize the first 3 million nodes. With MAP_POPULATE, it takes 198 seconds + 53 seconds to mmap = pretty much exactly the same amount of time. If we were running a server process, though, and we wanted to ensure low latency on the first request we serve, MAP_POPULATE would be useful to ensure that our cache was ready to go before we register ourself with the load balancer.

## Slightly original idea: search graph

After reading many good papers about monotonic search networks, I became reasonably certain that I could write something relatively performant by taking that approach.

There is an excellent recent survey paper [here](https://arxiv.org/pdf/2101.12631). Also of use are the [navigating spreading out graph](https://arxiv.org/pdf/1707.00143) and [satellite system graph](https://deepai.org/publication/satellite-system-graph-towards-the-efficiency-up-boundary-of-graph-based-approximate-nearest-neighbor-search) papers.

To summarize, these approaches use a best-first search through a graph to move in (roughly) the direction of the query node in euclidean space. Ideally, the graphs would have the property that there always exists a path with monotonically decreasing distance from a root node to all nodes in the graph, but in practice, it's obviously difficult to create a graph that satisfies that constraint. Instead, they use various tricks to approximate it.

The first step of the approximation is to build some simple starting graph. The three choices used in the literature are the Delaunay graph, k-nn graph, and minimum spanning tree. The k-nn graph is cheapest to approximate (using [a simple local search heuristic](https://www.ambuehler.ethz.ch/CDstore/www2011/proceedings/p577.pdf)), and the survey paper even shows that a worse approximation actually gives better search performance (possibly because it left some long range shortcuts, like contraction hierarchies builds?).

The satellite system graph is interesting because for each node, it tries to select a set of neighbors that are in different directions from each other. It does so by saying that each selected neighbor is the only node within a cone centered on the edge connecting the two nodes in euclidean space. The angle of the cone is `2*alpha`, where `alpha` is a parameter of the algorithm. This ensures that the `m` neighbors selected for a given node are, ideally, in different directions so that it's more likely that we can navigate towards the query point from that node. What's interesting is that, thanks to [the law of cosines](https://www.geeksforgeeks.org/find-angles-given-triangle/), we can find the "angle" between points in any metric space, even if that metric space has no concept of coordinates. I have no idea if this is mathematically sane, but it seems like it could help us generalize the search graph concept to arbitrary metric spaces. That is, we could impose the same constraint on neighboring nodes being "in different directions" by computing the angle between two candidate neighbors.

The survey paper complains that the papers in this area tend to follow the same general schema, but swap in multiple novel steps, fail to describe which of the novel steps actually improved performance, benchmark on a small number of datasets, and proclaim victory. In the spirit of continuing the tradition the survey paper complains of, I propose trying all these new things simultaneously:

1. Generalize to arbitrary metric spaces and maybe optimize the neighbor set by computing angles between neighbors (see above).
2. Use a k-nn graph, but obtain global connectivity by building a TSP-like permutation of the nodes as an additional set of edges to add to the graph. The local search will proceed by randomly swapping nodes in the permutation to try to minimize the distance between successive nodes.
3. Other random stuff thrown in, as I think of it. Start the search from random nodes? Random restarts? Starting new requests at the terminal point of recent requests, to handle spikes in similar queries (kind of like a fuzzy form of caching)? Online insertion and deletion? Background index optimization?

This will be implemented in Rust because I've been putting off learning it for too long.

### More papers discovered

- [Graph-based Nearest Neighbor Search: From Practice to Theory](https://arxiv.org/abs/1907.00845) has several interesting theorems. Unfortunately, they only apply to uniformly distributed data, but it's still helpful. *Beam search on k-nn graphs* is found to be extremely effective. Beam search allows you to keep your degree low even when d is high.
- In the same vein, [Approximate k-NN graph construction: a generic online approach](https://arxiv.org/pdf/1804.03032) also shows great performance with beam search on a k-nn-graph. Plus, it's *incremental and generalized to arbitrary metric spaces*.

## Abandoned attempt 1: hilbert-curve based method

I abandoned this one because it's really difficult to work with space-filling curves -- no good implementations of n-dimensional Hilbert curves *with bounding box support* can be found. Without a function to map a bounding box to segments of the Hilbert curve, I can't be sure when I have exhaustively searched all points within a given radius.

["Efficient Metric Indexing for Similarity Search"](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7113317&casa_token=sm3joK9gxH4AAAAA:F9yNtC9C2SgVbfeVIfs7Ky92r2OssX3gCPcNsfbh7Soi8O4b1MGFOLxkR8buoYKF7rAp&tag=1) looks like a good paper to imitate. It's incredibly simple:

1. Choose a set of p pivot objects.
2. Project each object o into p-dimensional space by computing [d(o, p_1), d(o, p_2) ... d(o, p_p)]. Amazingly, this projection approximately preserves relative distances between objects in the original space when you use the l-infinity norm on the resulting vectors.
3. Map the vectors to a space-filling curve.
4. Store the curve in a B+-tree. In our case, the B+-tree is FoundationDB.

It even supports incremental insertion and deletion, since it's mapped to a space-filling curve.

Questions I want to answer with this prototype:

1. Instead of choosing pivot objects from our data set, can I generate random pivot objects? How does that impact performance in real-world applications?
2. How does performance compare to brute force search?
3. How does performance compare to a vantage point tree (to the extent they are comparable)?

For now, I will use an in-memory Data.Map instead of FDB.

Because it doesn't build in the latest stackage, I moved the `fractals` Hackage package directly into the `src/` directory of this project. The package is BSD3 licensed, (c) 2015 Stephen Dekker.
