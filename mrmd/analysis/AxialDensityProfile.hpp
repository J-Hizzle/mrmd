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

class AxialDensityProfile
{
private:
    data::MultiHistogram averageDensityProfile_;
    idx_t numberOfDensityProfileSamples_ = 0;
    real_t binVolume_;
    idx_t numTypes_;
    AXIS axis_;

public:
    void sample(const data::Atoms& atoms);

    inline auto getAverageDensityProfile() const { return averageDensityProfile_; }
    inline auto getAverageDensityProfile(const idx_t& typeId) const
    {
        assert(typeId < numTypes_);
        assert(typeId >= 0);
        return Kokkos::subview(averageDensityProfile_.data, Kokkos::ALL(), typeId);
    }

    void reset();

    AxialDensityProfile(const real_t min,
                        const real_t max,
                        const idx_t numBins,
                        const real_t binVolume,
                        const idx_t numTypes,
                        const AXIS axis);
};
}  // namespace analysis
}  // namespace mrmd