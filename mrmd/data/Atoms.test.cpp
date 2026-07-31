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

#include "Atoms.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace data
{
TEST(Atoms, getNumTypes)
{
    Atoms atoms(100);
    atoms.numLocalAtoms = 100;

    auto type = atoms.getType();
    auto policy = Kokkos::RangePolicy<>(0, atoms.size());
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, idx_t& maxType, idx_t& minType)
    {
        type(idx) = idx % 10 - 1;  // types from -1 to 8
    };
    Kokkos::parallel_for("getNumTypes", policy, kernel);
    Kokkos::fence();
    EXPECT_EQ(atoms.getNumTypes(), 10);
}
}  // namespace data
}  // namespace mrmd