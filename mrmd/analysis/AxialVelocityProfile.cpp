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

#include "AxialVelocityProfile.hpp"

namespace mrmd
{
namespace analysis
{
data::MultiHistogram getAxialStreamingVelocityProfile(const data::Atoms& atoms,
                                                      const real_t min,
                                                      const real_t max,
                                                      const idx_t numBins,
                                                      const AXIS gridAxis)
{
    MRMD_HOST_CHECK_GREATEREQUAL(max, min);

    const auto numAtoms = atoms.numLocalAtoms + atoms.numGhostAtoms;
    const auto numTypes = atoms.getNumTypes();
    const auto positions = atoms.getPos();
    const auto type = atoms.getType();
    const auto velocities = atoms.getVel();

    data::MultiHistogram histogram("velocity-profile", min, max, numBins, numTypes, DIMENSIONS);
    MultiVectorView velocityAccum("velocity-accum", numBins, numTypes, DIMENSIONS);
    MultiView particleNumber("particle-number", numBins, numTypes);

    auto velocityAccumView = velocityAccum;
    auto particleNumberView = particleNumber;

    auto accumulationPolicy =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {numAtoms, DIMENSIONS});
    auto accumulationKernel = KOKKOS_LAMBDA(const idx_t idx, const idx_t dimId)
    {
        MRMD_DEVICE_ASSERT_GREATEREQUAL(type(idx), 0);
        MRMD_DEVICE_ASSERT_LESS(type(idx), numTypes);

        const auto bin = histogram.getBin(positions(idx, to_underlying(gridAxis)));
        if (bin == -1) return;

        Kokkos::atomic_add(&velocityAccumView(bin, type(idx), dimId), velocities(idx, dimId));
        Kokkos::atomic_add(&particleNumberView(bin, type(idx)), 1_r);
    };
    Kokkos::parallel_for(
        "AxialVelocityProfile::accumulate", accumulationPolicy, accumulationKernel);
    Kokkos::fence();

    auto normalizationPolicy =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {numBins, numTypes, DIMENSIONS});
    auto normalizationKernel = KOKKOS_LAMBDA(const idx_t bin, const idx_t typeId, const idx_t dimId)
    {
        const auto count = particleNumberView(bin, typeId);
        if (count > 0_r)
        {
            histogram.data(bin, typeId, dimId) = velocityAccumView(bin, typeId, dimId) / count;
        }
        else
        {
            MRMD_DEVICE_ASSERT_EQUAL(velocityAccumView(bin, typeId, dimId), 0_r);
            histogram.data(bin, typeId, dimId) = 0_r;
        }
    };
    Kokkos::parallel_for(
        "AxialVelocityProfile::normalize", normalizationPolicy, normalizationKernel);
    Kokkos::fence();

    return histogram;
}

data::MultiHistogram getAxialTotalVelocityVectorProfile(const data::Atoms& atoms,
                                                        const real_t min,
                                                        const real_t max,
                                                        const idx_t numBins,
                                                        const AXIS gridAxis)
{
    MRMD_HOST_CHECK_GREATEREQUAL(max, min);

    auto numAtoms = atoms.numLocalAtoms + atoms.numGhostAtoms;
    auto numTypes = atoms.getNumTypes();
    auto positions = atoms.getPos();
    auto type = atoms.getType();
    auto velocities = atoms.getVel();

    data::MultiHistogram histogram("velocity-profile", min, max, numBins, numTypes, DIMENSIONS);
    MultiVectorScatterView scatter(histogram.data);

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {numAtoms, DIMENSIONS});
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, const idx_t dimId)
    {
        MRMD_DEVICE_ASSERT_GREATEREQUAL(type(idx), 0);
        MRMD_DEVICE_ASSERT_LESS(type(idx), numTypes);
        auto bin = histogram.getBin(positions(idx, to_underlying(gridAxis)));
        if (bin == -1) return;
        auto access = scatter.access();
        access(bin, type(idx), dimId) += velocities(idx, dimId);
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::Experimental::contribute(histogram.data, scatter);
    Kokkos::fence();

    return histogram;
}
}  // namespace analysis
}  // namespace mrmd