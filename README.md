# metric-indexing

I want to take arbitrary objects with arbitrary metrics defined on them and
store them in FDB in a way that allows for efficient nearest-neighbor queries.

There are, of course, many metric trees, such as cover trees, vantage point trees, slim-trees, split-trees, etc., but they are designed (mostly) on the assumption that the tree will be stored in memory. I need something that is amenable to storage in FDB.

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
