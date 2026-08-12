// Copyright 2024 Sebastian Eibl
// Copyright 2026 Julian Friedrich Hille
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>

#include "assert/assert.hpp"
#include "datatypes.hpp"
#include "functional.hpp"

namespace mrmd
{
namespace data
{

struct MultiHistogram
{
    MultiHistogram(const std::string& label,
                   const real_t minArg,
                   const real_t maxArg,
                   idx_t numBinsArg,
                   idx_t numHistogramsArg)
        : MultiHistogram(label, minArg, maxArg, numBinsArg, numHistogramsArg, 1)
        {
        }

    MultiHistogram(const std::string& label,
                   const real_t minArg,
                   const real_t maxArg,
                   idx_t numBinsArg,
                   idx_t numHistogramsArg,
                   idx_t numDimensionsArg)
        : min(minArg),
          max(maxArg),
          numBins(numBinsArg),
          numHistograms(numHistogramsArg),
          numDimensions(numDimensionsArg),
          binSize((max - min) / real_c(numBins)),
          inverseBinSize(1_r / binSize),
          data(label, numBins, numHistograms, numDimensions)
    {
        MRMD_HOST_CHECK_GREATER(maxArg, minArg);
        MRMD_HOST_CHECK_GREATEREQUAL(numBinsArg, 0);
        MRMD_HOST_CHECK_GREATEREQUAL(numHistogramsArg, 0);
        MRMD_HOST_CHECK_GREATER(numDimensionsArg, 0);
        MRMD_HOST_CHECK_GREATEREQUAL(3, numDimensionsArg);
    }

    MultiHistogram(const std::string& label, const MultiHistogram& histogram)
        : MultiHistogram(
              label, histogram.min, histogram.max, histogram.numBins, histogram.numHistograms, histogram.numDimensions)
    {
        Kokkos::deep_copy(data, histogram.data);
    }

    idx_t getNumDimensions() const
    {
        return numDimensions;
    }

    MultiHistogram getHighDimensionalHistogramWithFilledValues(const idx_t newNumDimensions) const
    {
        MRMD_HOST_CHECK_EQUAL(numDimensions, 1);
        MRMD_HOST_CHECK_GREATEREQUAL(newNumDimensions, 1);
        MRMD_HOST_CHECK_GREATEREQUAL(3, newNumDimensions);

        auto dataView = data; // avoid capturing this pointer

        MultiHistogram newHistogram("newHistogram", min, max, numBins, numHistograms, newNumDimensions);
        
        auto range = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {numBins, numHistograms});
        auto policy = KOKKOS_LAMBDA(const idx_t binIdx, const idx_t histIdx) {
            auto value = dataView(binIdx, histIdx, 0);
            for (idx_t dimIdx = 0; dimIdx < newNumDimensions; ++dimIdx)
            {
                newHistogram.data(binIdx, histIdx, dimIdx) = value;
            }
        };

        Kokkos::parallel_for("vectorParticleNumberProfile", range, policy);
        return newHistogram;
    }

    /**
     * @param val input value
     * @return corresponding bin or -1 if outside of range
     */
    KOKKOS_INLINE_FUNCTION idx_t getBin(const real_t& val) const
    {
        auto bin = idx_c(std::floor((val - min) * inverseBinSize));
        if (bin < 0) bin = -1;
        if (bin >= numBins) bin = -1;
        return bin;
    }

    KOKKOS_INLINE_FUNCTION real_t getBinPosition(idx_t binIdx) const
    {
        assert(binIdx >= 0);
        assert(binIdx < numBins);
        auto binPosition = min + (real_c(binIdx) + 0.5_r) * binSize;
        return binPosition;
    }

    const real_t min;
    const real_t max;
    const idx_t numBins;
    const idx_t numHistograms;
    const idx_t numDimensions;
    const real_t binSize;
    const real_t inverseBinSize;
    MultiVectorView data;

    MultiHistogram& operator+=(const MultiHistogram& rhs);
    MultiHistogram& operator-=(const MultiHistogram& rhs);
    MultiHistogram& operator*=(const MultiHistogram& rhs);
    MultiHistogram& operator/=(const MultiHistogram& rhs);

    void scale(const real_t& scalingFactor);
    void scale(const ScalarView& scalingFactor);
    void makeSymmetric();
};

/**
 * Add new item to the cumulative moving average.
 *
 * @param average averaged histogram (output)
 * @param current current value
 * @param movingAverageFactor new value = (n * average + current) / (n + 1)
 */
void cumulativeMovingAverage(data::MultiHistogram& average,
                             const data::MultiHistogram& current,
                             const real_t movingAverageFactor = 10_r);

/**
 * Calculates the gradient of the histogram.
 * Uses central difference for all inner values and one-sided/central difference for
 * boundary values depending on periodicity.
 *
 * @param input input histogram
 * @param periodic treat boundaries as periodic
 * @return gradient of the histogram
 */
data::MultiHistogram gradient(const data::MultiHistogram& input, const bool periodic = false);

/**
 * Smoothen a histogram using gaussian convolution.
 *
 * @param input input histogram
 * @param sigma sigma of the gaussian convolution
 * @param range calculation range in multiples of sigma
 * @param periodic treat boundaries as periodic
 * @return smoothened histogram
 */
data::MultiHistogram smoothen(const data::MultiHistogram& input,
                              const real_t& sigma,
                              const real_t& range,
                              const bool periodic = false);

ScalarView createGrid(const MultiHistogram& input);

/**
 * Applies a binary operation to corresponding elements of two input MultiHistograms and stores
 * the result in an output MultiHistogram.
 *
 * @tparam BinaryOp The type of the binary operation to be applied.
 * @param input1 The first input MultiHistogram.
 * @param input2 The second input MultiHistogram.
 * @param output The MultiHistogram where the result of the binary operation will be stored.
 * @param binary_op The binary operation to apply to the elements of input1 and input2.
 *
 * @pre The dimensions (numBins and numHistograms) of input1, input2, and output must match.
 */

template <class BinaryOp>
void transform(const MultiHistogram& input1,
               const MultiHistogram& input2,
               MultiHistogram& output,
               const BinaryOp& binary_op)
{
    MRMD_HOST_CHECK_EQUAL(input1.numHistograms, input2.numHistograms);
    MRMD_HOST_CHECK_EQUAL(input1.numHistograms, output.numHistograms);
    MRMD_HOST_CHECK_EQUAL(input1.numDimensions, input2.numDimensions);
    MRMD_HOST_CHECK_EQUAL(input1.numDimensions, output.numDimensions);
    MRMD_HOST_CHECK_EQUAL(input1.numBins, input2.numBins);
    MRMD_HOST_CHECK_EQUAL(input1.numBins, output.numBins);

    auto input1Data = input1.data;
    auto input2Data = input2.data;
    auto outputData = output.data;

    auto policy =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {input1.numBins, input1.numHistograms, input1.numDimensions});
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, const idx_t jdx, const idx_t kdx)
    {
        outputData(idx, jdx, kdx) = binary_op(input1Data(idx, jdx, kdx), input2Data(idx, jdx, kdx));
    };
    Kokkos::parallel_for("MultiHistogram::transform", policy, kernel);
    Kokkos::fence();
}

/**
 * Replaces histogram values with a new value if the bin position satisfies a predicate.
 *
 * @param hist The MultiHistogram object whose values will be conditionally replaced.
 * @param pred A unary predicate function that is evaluated for each bin position.
 *             If it returns true for a bin position, all histogram values at that bin
 *             are replaced with newValue.
 * @param newValue The value to assign to histogram entries whose bin position satisfies
 *                 the predicate.
 */
template <OneCoordinatePredicate Pred>
void replace_if_bin_position(MultiHistogram& hist, const Pred& pred, real_t newValue)
{
    auto policy =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {hist.numBins, hist.numHistograms, hist.numDimensions});
    auto kernel = KOKKOS_LAMBDA(const idx_t binIdx, const idx_t histIdx, const idx_t dimIdx)
    {
        if (pred(hist.getBinPosition(binIdx)))
        {
            hist.data(binIdx, histIdx, dimIdx) = newValue;
        }
    };
    Kokkos::parallel_for("replace_if_bin_position", policy, kernel);
    Kokkos::fence();
}

}  // namespace data
}  // namespace mrmd