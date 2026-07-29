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

#include "AxialDensityProfile.hpp"

namespace mrmd
{
namespace analysis
{
data::MultiHistogram getAxialParticleNumberProfile(const idx_t numAtoms,
                                                   const data::Atoms::pos_t& positions,
                                                   const data::Atoms::type_t& types,
                                                   const int64_t numTypes,
                                                   const real_t min,
                                                   const real_t max,
                                                   const int64_t numBins,
                                                   const AXIS axis)
{
    MRMD_HOST_CHECK_GREATEREQUAL(max, min);
    MRMD_HOST_CHECK_GREATER(numTypes, 0);

    data::MultiHistogram histogram("density-profile", min, max, numBins, numTypes);
    MultiScatterView scatter(histogram.data);

    auto policy = Kokkos::RangePolicy<>(0, numAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        MRMD_DEVICE_ASSERT_GREATEREQUAL(types(idx), 0);
        MRMD_DEVICE_ASSERT_LESS(types(idx), numTypes);
        auto bin = histogram.getBin(positions(idx, to_underlying(axis)));
        if (bin == -1) return;
        auto access = scatter.access();
        access(bin, types(idx)) += 1_r;
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::Experimental::contribute(histogram.data, scatter);
    Kokkos::fence();

    return histogram;
}

AxialDensityProfile::AxialDensityProfile(const real_t min,
                                         const real_t max,
                                         const idx_t numBins,
                                         const real_t binVolume,
                                         const idx_t numTypes,
                                         const AXIS axis)
    : averageDensityProfile_("cumulative-average-density-profile", min, max, numBins, numTypes),
      binVolume_(binVolume),
      numTypes_(numTypes),
      axis_(axis)
{
    MRMD_HOST_CHECK_GREATEREQUAL(max, min);
    MRMD_HOST_CHECK_GREATER(numTypes, 0);
}

void AxialDensityProfile::reset()
{
    MRMD_HOST_CHECK_GREATER(
        numberOfDensityProfileSamples_,
        0,
        "Cannot reset AxialDensityProfile because no samples have been taken yet.");

    Kokkos::deep_copy(averageDensityProfile_.data, 0_r);
    numberOfDensityProfileSamples_ = 0;
}

void AxialDensityProfile::sample(const data::Atoms& atoms)
{
    auto instantaneousDensityProfile = getAxialParticleNumberProfile(atoms.numLocalAtoms,
                                                                     atoms.getPos(),
                                                                     atoms.getType(),
                                                                     numTypes_,
                                                                     averageDensityProfile_.min,
                                                                     averageDensityProfile_.max,
                                                                     averageDensityProfile_.numBins,
                                                                     axis_);

    instantaneousDensityProfile.scale(1_r / binVolume_);

    cumulativeMovingAverage(averageDensityProfile_,
                            instantaneousDensityProfile,
                            real_c(numberOfDensityProfileSamples_));
    numberOfDensityProfileSamples_++;
}

}  // namespace analysis
}  // namespace mrmd