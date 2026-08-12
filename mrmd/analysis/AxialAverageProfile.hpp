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
#include <algorithm>

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
    const real_t normalizationFactor_;
    const idx_t numTypes_;
    const AXIS gridAxis_;
    const std::vector<AXIS> dataAxes_;

    inline auto getDimensionIdentificatorFromAxis(const AXIS& dataAxis) const
    {
        auto iterator = std::find(dataAxes_.begin(), dataAxes_.end(), dataAxis);
        assert(iterator != dataAxes_.end() && "data axis is not part of the data axes");
        return idx_c(std::distance(dataAxes_.begin(), iterator));
    }

public:
    template <AxialProfileSampler Sampler>
    void sample(const data::Atoms& atoms, const Sampler& sampler)
    {
        sampledProfile_ += sampler(
            atoms, averageProfile_.min, averageProfile_.max, averageProfile_.numBins, gridAxis_);

        numberOfSamples_++;
    }

    void update();

    void reweight(const data::MultiHistogram& reweightingHistogram);

    inline auto getAverageProfile() const { return averageProfile_; }
    inline auto getAverageProfile(const idx_t& typeId) const
    {
        assert(typeId < numTypes_);
        assert(typeId >= 0);
        assert(dataAxes_.size() == 1);
        return Kokkos::subview(averageProfile_.data, Kokkos::ALL(), typeId, 0);
    }

    inline auto getAverageProfile(const idx_t& typeId, const AXIS& dataAxis) const
    {
        assert(typeId < numTypes_);
        assert(typeId >= 0);

        auto dimId = getDimensionIdentificatorFromAxis(dataAxis);

        return Kokkos::subview(averageProfile_.data, Kokkos::ALL(), typeId, dimId);
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
        assert(dataAxes_.size() == 1);

        data::MultiHistogram histogram("histogram", sampledProfile_);
        histogram.scale(1_r / real_c(numberOfSamples_));
        return Kokkos::subview(averageProfile_.data, Kokkos::ALL(), typeId, 0);
    }

    inline auto getSampledProfile(const idx_t& typeId, const AXIS& dataAxis) const
    {
        assert(typeId < numTypes_);
        assert(typeId >= 0);

        data::MultiHistogram histogram("histogram", sampledProfile_);
        histogram.scale(1_r / real_c(numberOfSamples_));
        auto dimId = getDimensionIdentificatorFromAxis(dataAxis);
        return Kokkos::subview(histogram.data, Kokkos::ALL(), typeId, dimId);
    }

    AxialAverageProfile(const data::Subdomain& subdomain,
                        const real_t binWidth,
                        const real_t normalizationFactor,
                        const idx_t numTypes,
                        const AXIS& gridAxis)
        : AxialAverageProfile(subdomain, binWidth, normalizationFactor, numTypes, gridAxis, {gridAxis})
    {
    }

    AxialAverageProfile(const data::Subdomain& subdomain,
                        const real_t binWidth,
                        const real_t normalizationFactor,
                        const idx_t numTypes,
                        const AXIS& gridAxis,
                        const std::vector<AXIS>& dataAxes);
};
}  // namespace analysis
}  // namespace mrmd