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

#include <concepts>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

#include "data/Atoms.hpp"
#include "data/MultiHistogram.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace analysis
{
template <typename F>
concept ProfileSampler =
    std::invocable<F, const data::Atoms&, const real_t, const real_t, const idx_t, const AXIS> &&
    std::same_as<std::invoke_result_t<F,
                                      const data::Atoms&,
                                      const real_t,
                                      const real_t,
                                      const idx_t,
                                      const AXIS>,
                 data::MultiHistogram>;

class AxialAverageProfile
{
private:
    data::MultiHistogram averageProfile_;
    idx_t numberOfSamples_ = 0;
    real_t binVolume_;
    idx_t numTypes_;
    AXIS axis_;

public:
    template <typename Sampler>
        requires ProfileSampler<Sampler&&>
    void sample(const data::Atoms& atoms, Sampler&& sampler)
    {
        auto instantaneousProfile = std::invoke(std::forward<Sampler>(sampler),
                                                atoms,
                                                averageProfile_.min,
                                                averageProfile_.max,
                                                averageProfile_.numBins,
                                                axis_);

        instantaneousProfile.scale(1_r / binVolume_);

        cumulativeMovingAverage(averageProfile_, instantaneousProfile, real_c(numberOfSamples_));
        numberOfSamples_++;
    }

    inline auto getAverageDensityProfile() const { return averageProfile_; }
    inline auto getAverageDensityProfile(const idx_t& typeId) const
    {
        assert(typeId < numTypes_);
        assert(typeId >= 0);
        return Kokkos::subview(averageProfile_.data, Kokkos::ALL(), typeId);
    }

    void reset();

    AxialAverageProfile(const data::Subdomain& subdomain,
                        const real_t binWidth,
                        const idx_t numTypes,
                        const AXIS& axis);
};
}  // namespace analysis
}  // namespace mrmd