// Copyright 2024 Sebastian Eibl
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

#include <vector>

#include "data/Atoms.hpp"
#include "data/MultiHistogram.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace analysis
{
/**
 * Calculate a discretized particle number profile along an axis.
 * Out-of-bounds values are discarded.
 */
data::MultiHistogram getAxialParticleNumberProfile(const idx_t numAtoms,
                                            const data::Atoms::pos_t& positions,
                                            const data::Atoms::type_t& types,
                                            const int64_t numTypes,
                                            const real_t min,
                                            const real_t max,
                                            const int64_t numBins,
                                            const AXIS axis);

class DensityProfile
{
private:
    data::MultiHistogram cumulativeParticleNumberProfile_;
    idx_t particleNumberProfileSamples_ = 0;
    data::MultiHistogram densityProfile_;
    real_t binVolume_;
    idx_t numTypes_;
    AXIS axis_;

public:
    void sampleParticleNumberProfile(const data::Atoms& atoms)
    {
        cumulativeParticleNumberProfile_ += getAxialParticleNumberProfile(atoms.numLocalAtoms,
                                                  atoms.getPos(),
                                                  atoms.getType(),
                                                  numTypes_,
                                                  cumulativeParticleNumberProfile_.min,
                                                  cumulativeParticleNumberProfile_.max,
                                                  cumulativeParticleNumberProfile_.numBins,
                                                  axis_);
        particleNumberProfileSamples_++;
    }

    void calcAverageDensityProfile()
    {
        auto normalizationFactor = 1_r / (binVolume_ * real_c(particleNumberProfileSamples_));
        Kokkos::deep_copy(densityProfile_.data, cumulativeParticleNumberProfile_.data);
        densityProfile_.scale(normalizationFactor);
        Kokkos::deep_copy(cumulativeParticleNumberProfile_.data, 0_r);
        particleNumberProfileSamples_ = 0;
    }

    inline auto getDensityProfile() const { return densityProfile_; }
    inline auto getDensityProfile(const idx_t& typeId) const
    {
        assert(typeId < numTypes_);
        assert(typeId >= 0);
        return Kokkos::subview(densityProfile_.data, Kokkos::ALL(), typeId);
    }

    DensityProfile(const real_t min,
                   const real_t max,
                   const idx_t numBins,
                   const real_t binVolume,
                   const idx_t numTypes,
                   const AXIS axis)
        : cumulativeParticleNumberProfile_("cumulative-particle-number-profile",
                                            min,
                                            max,
                                            numBins,
                                            numTypes),
          densityProfile_("density-profile", cumulativeParticleNumberProfile_),
          binVolume_(binVolume),
          numTypes_(numTypes),
          axis_(axis)
    {
        MRMD_HOST_CHECK_GREATEREQUAL(max, min);
        MRMD_HOST_CHECK_GREATER(numTypes, 0);
    }
};
}  // namespace analysis
}  // namespace mrmd