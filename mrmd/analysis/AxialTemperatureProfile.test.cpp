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

#include "AxialTemperatureProfile.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace analysis
{
data::Atoms initAtoms()
{
    data::Atoms atoms(100 * 3);
    atoms.numLocalAtoms = 300;

    auto policy = Kokkos::RangePolicy<>(0, 1);
    auto kernel = KOKKOS_LAMBDA(const idx_t& /*tmp*/, idx_t& sum)
    {
        idx_t idx = 0;
        for (auto i = 0; i < 10; ++i)
        {
            atoms.getPos()(idx, 0) = real_c(i) + 0.5_r;
            atoms.getVel()(idx, 0) = 1_r;
            atoms.getVel()(idx, 1) = 0_r;
            atoms.getVel()(idx, 2) = 0_r;
            atoms.getMass()(idx) = 1_r;
            atoms.getType()(idx) = 0;
            ++idx;

            for (auto j = 0; j < i + 1; ++j)
            {
                atoms.getPos()(idx, 0) = real_c(i) + 0.5_r;
                atoms.getVel()(idx, 0) = 1_r;
                atoms.getVel()(idx, 1) = 1_r;
                atoms.getVel()(idx, 2) = 1_r;
                atoms.getMass()(idx) = 2_r;
                atoms.getType()(idx) = 1;
                ++idx;

                atoms.getPos()(idx, 0) = 10_r - (real_c(i) + 0.5_r);
                atoms.getVel()(idx, 0) = 1_r;
                atoms.getVel()(idx, 1) = 2_r;
                atoms.getVel()(idx, 2) = 3_r;
                atoms.getMass()(idx) = 3_r;
                atoms.getType()(idx) = 2;
                ++idx;
            }
        }
        sum += idx;
    };
    idx_t numAtoms = 0;
    Kokkos::parallel_reduce("init-atoms", policy, kernel, numAtoms);
    Kokkos::fence();
    atoms.numLocalAtoms = numAtoms;

    return atoms;
}

TEST(AxialTemperatureProfile, histogram)
{
    auto atoms = initAtoms();

    auto histogram = getAxialTotalSquaredVelocityProfile(atoms.numLocalAtoms,
                                                         atoms.getPos(),
                                                         atoms.getVel(),
                                                         atoms.getMass(),
                                                         atoms.getType(),
                                                         3,
                                                         0_r,
                                                         10_r,
                                                         10,
                                                         AXIS::X);
    auto h_data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), histogram.data);

    for (auto i = 0; i < 10; ++i)
    {
        EXPECT_FLOAT_EQ(h_data(i, 0), real_c(1));
        EXPECT_FLOAT_EQ(h_data(i, 1), real_c(i + 1));
        EXPECT_FLOAT_EQ(h_data(i, 2), real_c(11 - (i + 1)));
    }
}
}  // namespace analysis
}  // namespace mrmd