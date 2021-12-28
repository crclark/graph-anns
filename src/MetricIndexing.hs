{-# LANGUAGE NamedFieldPuns #-}

module MetricIndexing where

import Data.Bits
import Data.List (minimumBy, foldl')
import qualified Data.Map.Strict as M
import Data.Ord (comparing)
import Data.SpaceFillingCurve.Hilbert.Integer (hilbert, hilbertInverse)
import Data.Word (Word64)

-- | A metric on values of type @a@.
data Metric a =
  Metric
  -- | The maximum distance between any two points in the metric space.
  -- Unbounded distances are not supported because we need to be able to
  -- discretize the distances in order to encode them in a Hilbert curve.
    { maxDistance :: Double
  -- | A distance function. Must be a metric in the mathematical sense.
    , distance :: a -> a -> Double
    }

{-# INLINABLE discretizationDelta #-}
discretizationDelta :: Metric a -> Double
discretizationDelta Metric {maxDistance} =
  maxDistance / fromIntegral (maxBound :: Word64)

discretize :: Metric a -> Double -> Word64
discretize m@Metric {maxDistance} d = floor (d / discretizationDelta m)

undiscretize :: Metric a -> Word64 -> Double
undiscretize m@Metric {maxDistance} discretized =
  fromIntegral discretized * discretizationDelta m

-- | Discretize a real-valued vector where each value of the vector is an
-- output of the distance function of the given metric, then map it to the
-- Hilbert curve.
toHilbert :: Metric a -> [Double] -> Word64
toHilbert m xs = hilbert 64 (map (discretize m) xs)

-- | Translates an object in the metric space into a point in the vector space
-- (R^n, L_inf) by using n pivot objects. If the pivot objects are held
-- constant, the Chebyshev distance between points in the vector space R^n is a
-- lower bound on the distance between the corresponding points in the original
-- metric space. See the paper for more details.
--
-- Obviously if this weren't a prototype we would probably not be using lists.
phi :: Metric a -> [a] -> a -> [Double]
phi Metric {distance} pivots x = map (distance x) pivots

-- | Chebyshev distance between two vectors.
lInf :: [Double] -> [Double] -> Double
lInf xs ys = maximum (zipWith (\x y -> abs (x - y)) xs ys)

-- | Given a metric, a set of pivot points, a query object q, and a radius r
-- around the query object, return the set of vertices in phi-space that bound
-- the region of radius r around the query object in the original metric space.
-- rangeRegion :: Metric a -> [a] -> a -> Double -> [[Double]]
-- rangeRegion m pivots q r =
--   [ undefined
--   | i <- [1..pivotCount]
--   ,
--   ]
--   where
--     pivotCount = length pivots
{-
Notes about the range region and MBB stuff in the paper, because it's complex.

The tricky part of the whole paper is that we are mapping an arbitrary metric
space into a vector space by using a pivot set.

The only connection we have between the metric space and the vector space is
that the Chebyshev distance in the vector space is a lower bound on distance in
the metric space. That is sufficient to speed up our searches.

Next, we need to map the vector space to a hilbert curve so that we can store
keys in a B-tree.

So we have two layers of transformation:

Metric space -> p-dimensional vector space -> 1-d hilbert curve

Now the question is, how do we do a radius query in metric space when we have
stored the data in hilbert curve space?

We need to translate distance in metric space to distance along the hilbert
curve to ensure that we scan the correct part of the B-tree.

Step 1: Convert to p-dimensional vector space. The two corners of the bounding
box for the query are `map (\p -> d p q - r) pivots` and
`map (\p -> d p q + r) pivots`. In other words, distance between q and pivots,
plus or minus the radius. We know that Chebyshev distance in this vector space
is a lower bound on distance, so the bounding box contains all objects _at least_
that far away, and possibly objects even more distant (we can discard those later).

Step 2: This is where the paper seems to make things more complex than they
need to be. In addition to storing hilbert curve keys in the B-tree, internal
nodes of the B-tree store minimum bounding boxes of their subtree. This seems to
be why they needed to implement their own custom B-tree, too. Instead, we can use
[this code](https://github.com/adishavit/hilbert/blob/master/hilbert.h#L92) to
find exactly the range we need to query along the hilbert curve. Then we don't
need a minimum bounding box stored in our B-tree, and we can use an off-the-shelf
ordered key-value store instead of writing our own B-tree code.

To use that code, we call it twice to get the first and last points of the
bounding box that lie on the hilbert curve, hilbert-encode those points, and then
do a range query. The results will be a collection of points that are mostly within
the desired radius of our query object. Unfortunately, we need to filter out the
out-of-range ones by computing their distance to the query object again... this
adds up to a lot of distance calculations per query (though obviously it's still
a lot better than brute force).

Next step: write a Haskell wrapper around that C code.

Calamity! The library can only handle small dimensionality
vectors, because its output is limited to 64-bit words. We need big
integers to map big, high-resolution dimensions onto the Hilbert curve.
Specifically, the above C code requires nDims * nBitsPerAxis <= 64.
In other words, if we have eight dimensions, we can only have 8-bits per
axis, which is quite small.

Instead, let's try to understand this proof https://github.com/davidmoten/hilbert-curve#querying-n-dimensional-space
and implement it ourselves.

It would probably be easiest to translate this repo into Haskell
https://github.com/davidmoten/hilbert-curve


On third thought, maybe we can get away with a small number of pivots and keep
the number of dimensions small enough to bind the C library after all.
How many pivots did the paper use?

Even simpler idea: start two streaming range queries searching
from the query point outward. The user supplies a query
time limit. We return all matches we found within that time
limit. We also return a token that allows the user to page
deeper into the results. No wait, that won't work because we
don't know if we are really returning the closest point at
any given page. We also wouldn't be traversing points in order.

Anyway, I seem to have discovered my own bounding box solution.
The recursive partitioning of space by a Hilbert curve can
be represented as a tree. See Lawder's papers. Lawder gives a
complex binary search algorithm for the tree, but it seems
unnecessary (and it's patented). Instead, you can simply
compute the bounding box
of each tree node, and descend the tree, collecting all nodes
that fall completely inside the bounding box. You are left with
a collection of nodes from various levels of the tree. Each of
these nodes has a corresponding hilbert index range. Merge
all of the ranges and you have the full set of ranges you
need to scan to find everything within the box. BONUS: our bounding box is
guaranteed to be a square because the lower left and upper right corners are
formed by subtracting/adding r from/to a constant point. Can we use this fact to
improve our algorithm?

"The appropriate number of pivots needed to achieve high query efficiency
is related to the intrinsic dimensionality of the dataset. The intrinsic dimensionality
of the dataset can be calculated as rho = mu^2 / (2*sigma^2)), where
mu and sigma are the mean and variance of the of the pairwise distances in the
dataset."

Googling "intrinsic dimensionality of a metric space" gives promising
results.

-}
-- | Given two opposite vertices of a bounding box,
--   return all points on the perimeter of the bounding
--   box.
perimeter :: [Int] -> [Int] -> [[Int]]
perimeter m n = undefined

-- | Given two opposite vertices of a bounding box,
--   return all 2^d vertices of the bounding box.
allVertices :: [Int] -> [Int] -> [[Int]]
allVertices [] [] = error "0 dimensions unsupported"
allVertices [] _ = error "unequal dims"
allVertices _ [] = error "unequal dims"
allVertices [x] [y] = [[x], [y]]
allVertices m n = do
  x <- [head m, head n]
  sub <- allVertices (tail m) (tail n)
  return (x : sub)

-- Idea: to list the adjacent nodes in a hypercube,
-- We enumerate all the one-bit flips of a vertex,
-- where zero = the coordinate c_i comes from the
-- corner m, and
-- one = the coordinate c_i comes from the opposite
-- corner n.
-- TODO: put this algorithm on the code golf
-- page, or something. I couldn't find this algo
-- anywhere by googling, and it's much faster
-- than the dumb code golf solutions already up
-- there (verify the speed claim).
allEdges :: [Int] -> [Int] -> [([Int], [Int])]
allEdges m n = [(vertex e, vertex f) | (e, f) <- allEdgeCombos d]
  where
    d = length m
    vertex = comboVertex' m n

-- TODO: rename combo to bitmask everywhere. Also
-- introduce a type synonym before I go insane.
allEdgeCombos :: Int -> [(Int, Int)]
allEdgeCombos d = [(e, f) | e <- allVertices, f <- greaterVertices e]
  where
    allVertices = allCombos' d
    greaterVertices e = [setBit e i | i <- [0 .. d - 1], not (testBit e i)]

-- | Returns bool vectors of all combinations of
-- the opposite corners of a d-dimensional bounding
-- box. Interpret the returned integers as bit vectors.
allCombos' :: Int -> [Int]
allCombos' d = [0 .. 2 ^ d - 1]

-- | Given opposite corners m and n and a specified
-- combo of the two, return
-- a vertex of the bounding box.
comboVertex' :: [Int] -> [Int] -> Int -> [Int]
comboVertex' m n combo = go m n combo (d - 1)
  where
    go [] [] _ (-1) = []
    go (m:ms) (n:ns) combo i =
      let x =
            if testBit combo i
              then n
              else m
       in x : go ms ns combo (i - 1)
    d = length m

class NearestNeighbor t where
  initNN :: [a] -> (a -> a -> Double) -> t a
  nearest :: t a -> (a -> a -> Double) -> a -> Maybe a

newtype BruteForce a =
  BruteForce [a]
  deriving (Eq, Ord, Show)

instance NearestNeighbor BruteForce where
  initNN xs _ = BruteForce xs
  nearest (BruteForce []) _ _ = Nothing
  nearest (BruteForce xs) d q = Just $ minimumBy (comparing (d q)) xs

newtype SFCIndex a =
  SFCIndex (M.Map Integer a)
  deriving (Eq, Ord, Show)

----------------------------------------------------------------
-- Generating the implicit tree that maps an n-dimensional grid to the Hilbert
-- curve. NOTE: I am using the word "grid" because we are mapping from a
-- discrete n-dimensional grid to a Hilbert curve. However, we still assume that
-- 0,0 is in the lower left corner. The upper right corners used below are
-- INCLUSIVE. That is, the box ([0,0],[3,3]) is a 4x4 grid and its cells are
-- include (0,0) and (3,3).


-- | A node in the implicit Hilbert tree. Keep this lazy or you're going to have
-- a bad time.
data HilbertNode =
  HilbertNode
    {
      -- | The number of dimensions in the input grid.
      hilbertNodeNumDimensions :: Int
      -- | The number of bits per dimension.
    , hilbertNodeNumBits :: Int
     -- | The lower left and upper right corners of this node's
     -- coverage in the n-dimensional grid. Guaranteed
     -- to be square.
    , hilbertNodeBoundingBox :: ([Int], [Int])
     -- | The minimum and maximum values of this node in the
     -- Hilbert curve output.
    , hilbertNodeCurveRange :: (Int, Int)
     -- | The 2^d children of this node.
    , hilbertNodeChildren :: [HilbertNode]
    }
  deriving (Eq, Ord, Show)

hilbertTree ::
  -- | The number of dimensions in the input space.
  Int ->
  -- | The number of bits per dimension.
  Int ->
  -- | The lower left corner of the input space.
  [Int] ->
  -- | The upper right corner of the input space.
  [Int] ->
  HilbertNode
hilbertTree d k lower upper
  | lower /= upper =
      HilbertNode
      { hilbertNodeNumDimensions = d
      , hilbertNodeNumBits = k
      , hilbertNodeBoundingBox = (lower, upper)
      , hilbertNodeCurveRange = (0, 2 ^ k - 1)
      , hilbertNodeChildren = [ hilbertTree d k lower' upper'
                              | (lower', upper') <- quadrants lower upper]
      }


-- | https://oeis.org/A006068
-- Maps from the top k bits of the coordinates of a point in our
-- n-dimensional grid to the next subtree in our recursive descent to find
-- our Hilbert encoding. TODO: optimize once we know it works.
a006068 :: (Data.Bits.Bits b, Integral b) => b -> b
a006068 n =
  foldl xor 0 $
  map (div n) $
  takeWhile (<= n) $
  iterate (* 2) 1

-- | Return the midpoint of an n-dimensional cube.
midpoint :: [Int] -> [Int] -> [Int]
midpoint = zipWith (\ l u -> (l + u) `div` 2)

-- | Given the lower left and upper right corners of a discrete grid in n dimensions,
-- return the lower left and upper right corners of each quadrant of the
-- discrete grid. Note that this is slightly different than the same answer for
-- a box in cartesian space.
--
-- These quadrants are returned in gray code order!
-- TODO: the order is wrong. It's dependent upon what level of the tree and
-- quadrant we are in. Look at a picture.
quadrants :: [Int] -> [Int] -> [([Int], [Int])]
quadrants lower upper =
  let d = length lower
      mid = midpoint lower upper
      leg = 1 + head mid - head lower
      addLeg v combo = zipWith (\ ix i -> i + (fromEnum (testBit combo ix) * leg)) [0..] v
      in [ (addLeg lower gray, addLeg mid gray)
         | combo <- allCombos' d
         , let gray = a006068 combo
         ]

hilbertEncodeNaive :: Int -> [Int] -> Int
hilbertEncodeNaive k xs =
  let
    d = length xs
    toBits k = foldl' (\ acc b -> acc `shiftL` 1 + fromEnum b) 0
    kBitsAtATime = map (\i -> a006068 (toBits k (map (`testBit` i) xs))) [k-1, k-2..0]
    in foldl' (\ acc b -> (acc `shiftL` d) .|. b) 0 kBitsAtATime
