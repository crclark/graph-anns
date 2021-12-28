-- |
-- Module      : Data.SpaceFillingCurve.Hilbert.Integer
-- Copyright   : (c) 2015 Stephen Dekker <steve.dekk@gmail.com>
-- License     : BSD3
--
-- Maintainer  : steve.dekk@gmail.com
-- Stability   : experimental
-- Portability : portable
--
-- An implementation of Butz's classic (and rather beautiful) algorithm for
-- computing the discrete Hilbert index of an N-dimensional point in
-- Cartesian space (A. R. Butz.  "Alternative algorithm for Hilbert’s
-- space-filling curve.  IEEE Transactions on Computers", pages 424–426,
-- April 1971).
--
-- This particular implementation relies upon the 'Integer' numeric type in
-- order to handle unbounded input point coordinates. A version not built
-- around the 'Integer' type could offer improved performance, as the
-- algorithm essentially boils down to the repeated application of bitwise
-- operations.
--
-- The specific algorithm used is the uncompact Hilbert indexing algorithm
-- described by Chris Hamilton (Hamilton, C. "Compact Hilbert Indices",
-- Dalhousie University, Faculty of Computer Science, Technical Report
-- CS-2006-07, July 2006).
--
-- Hamilton's paper provides a thorough overview of the mathematics behind
-- the algorithm and also extends it to handle variable encoding widths for
-- the different Cartesian axes. The compact Hilbert indexing scheme
-- described in the technical report is not implemented in this module.
--
-- The encoding function is written to accept a list of 'Bits' instances
-- for the input point and to produce a 'Num' instance for the output index
-- and is capable of handling unbounded 'Bits' instances such as 'Integer'.
--
-- Similarly, the decoding function will take an unbounded 'Num' instance
-- and produce a point consisting of unbounded 'Bits' components with the
-- desired dimensionality.
--
-- Lastly, the functions exported by this module will accept negative
-- inputs, but the behaviour of the functions for negative Hilbert indices
-- or point coordinates is undefined. These assumptions are made explicit
-- in the included QuickCheck property tests.

module Data.SpaceFillingCurve.Hilbert.Integer (
        -- * Encoding and decoding the Hilbert curve
        hilbert,                -- :: (Bits a, Bits b, Num b) => Int -> [a] -> b
        hilbertInverse,         -- :: (Bits a, Bits b) => Int -> Int -> a -> [b]
  ) where

import           Data.SpaceFillingCurve.Hilbert.Integer.Internal (hilbert, hilbertInverse)
