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

#include "MultiHistogram.hpp"

#include <cassert>

#include "util/math.hpp"

namespace mrmd
{
namespace data
{

MultiHistogram& MultiHistogram::operator+=(const MultiHistogram& rhs)
{
    transform(*this, rhs, *this, bin_op::Add());
    return *this;
}
MultiHistogram& MultiHistogram::operator-=(const MultiHistogram& rhs)
{
    transform(*this, rhs, *this, bin_op::Sub());
    return *this;
}
MultiHistogram& MultiHistogram::operator*=(const MultiHistogram& rhs)
{
    transform(*this, rhs, *this, bin_op::Mul());
    return *this;
}
MultiHistogram& MultiHistogram::operator/=(const MultiHistogram& rhs)
{
    transform(*this, rhs, *this, bin_op::Div());
    return *this;
}

void MultiHistogram::scale(const real_t& scalingFactor)
{
    auto hist = data;  // avoid capturing this pointer
    auto policy =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({idx_t(0), idx_t(0), idx_t(0)}, {numBins, numHistograms, numDimensions});
    auto normalizeSampleKernel = KOKKOS_LAMBDA(const idx_t idx, const idx_t jdx, const idx_t kdx)
    {
        hist(idx, jdx, kdx) *= scalingFactor;
    };
    Kokkos::parallel_for("MultiHistogram::scale", policy, normalizeSampleKernel);
    Kokkos::fence();
}

void MultiHistogram::scale(const ScalarView& scalingFactor)
{
    MRMD_HOST_CHECK_GREATEREQUAL(idx_c(scalingFactor.extent(0)), numHistograms);

    auto hist = data;  // avoid capturing this pointer
    auto policy =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({idx_t(0), idx_t(0), idx_t(0)}, {numBins, numHistograms, numDimensions});
    auto normalizeSampleKernel = KOKKOS_LAMBDA(const idx_t idx, const idx_t jdx, const idx_t kdx)
    {
        hist(idx, jdx, kdx) *= scalingFactor(jdx);
    };
    Kokkos::parallel_for("MultiHistogram::scale", policy, normalizeSampleKernel);
    Kokkos::fence();
}

void MultiHistogram::makeSymmetric()
{
    auto maxIdx = numBins - 1;
    auto hist = data;  // avoid capturing this pointer
    auto policy =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({idx_t(0), idx_t(0), idx_t(0)}, {numBins / 2, numHistograms, numDimensions});
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, const idx_t jdx, const idx_t kdx)
    {
        auto val = 0.5_r * (hist(idx, jdx, kdx) + hist(maxIdx - idx, jdx, kdx));
        hist(idx, jdx, kdx) = val;
        hist(maxIdx - idx, jdx, kdx) = val;
    };
    Kokkos::parallel_for("MultiHistogram::makeSymmetric", policy, kernel);
    Kokkos::fence();
}

void cumulativeMovingAverage(data::MultiHistogram& average,
                             const data::MultiHistogram& current,
                             const real_t movingAverageFactor)
{
    MRMD_HOST_CHECK_EQUAL(average.numBins, current.numBins);
    MRMD_HOST_CHECK_EQUAL(average.numHistograms, current.numHistograms);

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({idx_t(0), idx_t(0), idx_t(0)},
                                                         {average.numBins, average.numHistograms, average.numDimensions});
    auto kernel = KOKKOS_LAMBDA(const idx_t binIdx, const idx_t histogramIdx, const idx_t dimIdx)
    {
        // use running average to calculate new mean compensation energy
        average.data(binIdx, histogramIdx, dimIdx) =
            (movingAverageFactor * average.data(binIdx, histogramIdx, dimIdx) +
             current.data(binIdx, histogramIdx, dimIdx)) /
            (movingAverageFactor + 1_r);
    };
    Kokkos::parallel_for("MultiHistogram::cumulativeMovingAverage", policy, kernel);
    Kokkos::fence();
}

data::MultiHistogram gradient(const data::MultiHistogram& input, const bool periodic)
{
    const auto inverseSpacing = input.inverseBinSize;
    const auto inverseDoubleSpacing = 0.5_r * input.inverseBinSize;

    data::MultiHistogram grad("gradient", input.min, input.max, input.numBins, input.numHistograms, input.numDimensions);
    auto policy =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {input.numBins, input.numHistograms, input.numDimensions});
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, const idx_t jdx, const idx_t kdx)
    {
        if (idx == 0)
        {
            if (periodic)
            {
                grad.data(idx, jdx, kdx) =
                    (input.data(idx + 1, jdx, kdx) - input.data(input.numBins - 1, jdx, kdx)) *
                    inverseDoubleSpacing;
            }
            else
            {
                grad.data(idx, jdx, kdx) =
                    (input.data(idx + 1, jdx, kdx) - input.data(idx, jdx, kdx)) * inverseSpacing;
            }
            return;
        }

        if (idx == input.numBins - 1)
        {
            if (periodic)
            {
                grad.data(idx, jdx, kdx) =
                    (input.data(0, jdx, kdx) - input.data(idx - 1, jdx, kdx)) * inverseDoubleSpacing;
            }
            else
            {
                grad.data(idx, jdx, kdx) =
                    (input.data(idx, jdx, kdx) - input.data(idx - 1, jdx, kdx)) * inverseSpacing;
            }
            return;
        }

        grad.data(idx, jdx, kdx) =
            (input.data(idx + 1, jdx, kdx) - input.data(idx - 1, jdx, kdx)) * inverseDoubleSpacing;
    };
    Kokkos::parallel_for("MultiHistogram::gradient", policy, kernel);
    Kokkos::fence();

    return grad;
}

data::MultiHistogram smoothen(const data::MultiHistogram& input,
                              const real_t& sigma,
                              const real_t& range,
                              const bool periodic)
{
    const auto inverseSigma = 1_r / sigma;
    /// how many neighboring bins are affected
    const idx_t delta = int_c(range * sigma * input.inverseBinSize);

    data::MultiHistogram smoothenedDensityProfile(
        "smooth-input", input.min, input.max, input.numBins, input.numHistograms, input.numDimensions);

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({idx_t(0), idx_t(0), idx_t(0)},
                                                         {input.numBins, input.numHistograms, input.numDimensions});
    auto kernel = KOKKOS_LAMBDA(const idx_t binIdx, const idx_t histogramIdx, const idx_t dimIdx)
    {
        auto normalization = 0_r;

        idx_t jdxMin = binIdx - delta;
        idx_t jdxMax = binIdx + delta;

        if (!periodic)
        {
            jdxMin = Kokkos::max(idx_t(0), jdxMin);
            jdxMax = Kokkos::min(input.numBins - 1, jdxMax);
        }
        assert(jdxMin <= jdxMax);

        for (auto jdx = jdxMin; jdx <= jdxMax; ++jdx)
        {
            auto possiblyMappedIdx = jdx;
            if (periodic)
            {
                if (possiblyMappedIdx < 0) possiblyMappedIdx += input.numBins;
                if (possiblyMappedIdx >= input.numBins) possiblyMappedIdx -= input.numBins;
            }
            const auto eFunc =
                Kokkos::exp(-util::sqr(real_c(binIdx - jdx) * input.binSize * inverseSigma));
            normalization += eFunc;
            for (idx_t dimIdx = 0; dimIdx < input.numDimensions; ++dimIdx)
            {
                smoothenedDensityProfile.data(binIdx, histogramIdx, dimIdx) +=
                    input.data(possiblyMappedIdx, histogramIdx, dimIdx) * eFunc;
            }
        }

        for (idx_t dimIdx = 0; dimIdx < input.numDimensions; ++dimIdx)
        {
            smoothenedDensityProfile.data(binIdx, histogramIdx, dimIdx) /= normalization;
        }
    };
    Kokkos::parallel_for("MultiHistogram::smoothen", policy, kernel);
    Kokkos::fence();

    return smoothenedDensityProfile;
}

ScalarView createGrid(const MultiHistogram& input)
{
    const idx_t numBins = input.numBins;

    MRMD_HOST_CHECK_GREATEREQUAL(numBins, 0);

    ScalarView grid("grid", numBins);
    auto policy = Kokkos::RangePolicy<>(0, numBins);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx) { grid[idx] = input.getBinPosition(idx); };
    Kokkos::parallel_for("MultiHistogram::createGrid", policy, kernel);
    Kokkos::fence();

    return grid;
}
}  // namespace data
}  // namespace mrmd