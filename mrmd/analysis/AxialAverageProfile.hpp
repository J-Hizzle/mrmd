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
#include <type_traits>
#include <vector>

#include "data/Atoms.hpp"
#include "data/MultiHistogram.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace analysis
{
template <typename Sampler>
concept AxialProfileSampler =
    std::invocable<Sampler, const data::Atoms&, real_t, real_t, idx_t, AXIS> &&
    std::same_as<std::invoke_result_t<Sampler, const data::Atoms&, real_t, real_t, idx_t, AXIS>,
                 data::MultiHistogram>;

class AxialAverageProfile
{
private:
    data::MultiHistogram averageProfile_;
    data::MultiHistogram sampledProfile_;
    idx_t numberOfSamples_ = 0;
    real_t normalizationFactor_;
    idx_t numTypes_;
    AXIS axis_;

public:
    template <AxialProfileSampler Sampler>
    void sample(const data::Atoms& atoms, const Sampler& sampler)
    {
        sampledProfile_ += sampler(
            atoms, averageProfile_.min, averageProfile_.max, averageProfile_.numBins, axis_);

        numberOfSamples_++;
    }

    void update()
    {
        Kokkos::deep_copy(averageProfile_.data, sampledProfile_.data);
        averageProfile_.scale(1_r / (real_c(numberOfSamples_) * normalizationFactor_));
        Kokkos::deep_copy(sampledProfile_.data, 0_r);
        numberOfSamples_ = 0;
    }

    void reweight(const data::MultiHistogram& reweightingHistogram)
    {
        MRMD_HOST_CHECK_EQUAL(
            reweightingHistogram.numBins,
            averageProfile_.numBins,
            "reweighting histogram has different number of bins than average profile");
        MRMD_HOST_CHECK_EQUAL(
            reweightingHistogram.numHistograms,
            averageProfile_.numHistograms,
            "reweighting histogram has different number of histograms than average profile");

        averageProfile_ /= reweightingHistogram;
    }

    inline auto getAverageProfile() const { return averageProfile_; }
    inline auto getAverageProfile(const idx_t& typeId) const
    {
        assert(typeId < numTypes_);
        assert(typeId >= 0);
        return Kokkos::subview(averageProfile_.data, Kokkos::ALL(), typeId);
    }

    inline auto getSampledProfile() const
    {
        data::MultiHistogram histogram("histogram", sampledProfile_);
        histogram.scale(1_r / real_c(numberOfSamples_));
        return histogram;
    }

    inline auto getSampledProfile(const idx_t& typeId) const
    {
        assert(typeId < numTypes_);
        assert(typeId >= 0);

        data::MultiHistogram histogram("histogram", sampledProfile_);
        histogram.scale(1_r / real_c(numberOfSamples_));
        return Kokkos::subview(histogram.data, Kokkos::ALL(), typeId);
    }

    AxialAverageProfile(const data::Subdomain& subdomain,
                        const real_t binWidth,
                        const real_t normalizationFactor,
                        const idx_t numTypes,
                        const AXIS& axis);
};
}  // namespace analysis
}  // namespace mrmd