// Copyright 2025 Sebastian Eibl
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

#include "AxialMassFluxProfile.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace analysis
{

data::Atoms initAtoms()
{
    data::Atoms atoms(100 * 2);
    atoms.numLocalAtoms = 200;

    auto policy = Kokkos::RangePolicy<>(0, 1);
    auto kernel = KOKKOS_LAMBDA(const idx_t& /*tmp*/, idx_t& sum)
    {
        idx_t idx = 0;
        for (auto i = 0; i < 10; ++i)
        {
            for (auto j = 0; j < i + 1; ++j)
            {
                atoms.getPos()(idx, 0) = real_c(i) + 0.5_r;
                atoms.getVel()(idx, 0) = 1_r;
                atoms.getMass()(idx) = 1_r;
                atoms.getType()(idx) = 0;
                ++idx;

                atoms.getPos()(idx, 0) = 10_r - (real_c(i) + 0.5_r);
                atoms.getVel()(idx, 0) = -1_r;
                atoms.getMass()(idx) = 2_r;
                atoms.getType()(idx) = 1;
                ++idx;
            }
        }
        sum += idx;
    };
    idx_t numAtoms = 0;
    Kokkos::parallel_reduce("LinearDensityProfile::histogram", policy, kernel, numAtoms);
    Kokkos::fence();
    atoms.numLocalAtoms = numAtoms;
    atoms.numGhostAtoms = 0;

    return atoms;
}

void update(data::Atoms& atoms)
{
    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        atoms.getPos()(idx, 0) += atoms.getVel()(idx, 0);
    };
    Kokkos::parallel_for("update", policy, kernel);
    Kokkos::fence();
}

TEST(AxialMassFluxProfile, histogram)
{
    auto atoms = initAtoms();

    ScalarView planeGrid("planeGrid", 11);
    auto h_planeGrid = Kokkos::create_mirror_view(planeGrid);
    for (idx_t idx = 0; idx < 11; ++idx)
    {
        h_planeGrid(idx) = real_c(idx);
    }
    Kokkos::deep_copy(planeGrid, h_planeGrid);

    AxialMassFluxProfile axialMassFluxProfile(planeGrid, AXIS::X);

    axialMassFluxProfile.startCounting(atoms);
    update(atoms);
    auto histogram = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                         axialMassFluxProfile.stopCounting(atoms));

    for (idx_t idx = 0; idx < 11; ++idx)
    {
        EXPECT_FLOAT_EQ(histogram(idx), real_c(10 - idx) * 2_r - real_c(idx));
    }
}
}  // namespace analysis
}  // namespace mrmd