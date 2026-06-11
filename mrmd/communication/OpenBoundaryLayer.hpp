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

#include "communication/AccumulateForce.hpp"
#include "communication/GhostExchange.hpp"
#include "communication/PeriodicMapping.hpp"
#include "communication/UpdateGhostAtoms.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
class OpenBoundaryLayer
{
public:
    void removeOpenBoundaryAtoms(data::Atoms& atoms, const data::Subdomain& subdomain)
    {
        auto pos = atoms.getPos();
        auto type = atoms.getType();
        auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
        auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
        {
            for (auto dim = 0; dim < DIMENSIONS; ++dim)
            {
                if (subdomain.boundaryConditions[dim] == data::Subdomain::BoundaryCondition::OPEN)
                {
                    auto x = pos(idx, dim);
                    if (x > subdomain.maxCorner[dim] || x < subdomain.minCorner[dim])
                    {
                        type(idx) = -1;  // mark atom for deletion by setting type to -1
                        break;
                    }
                }
            }
        };
        Kokkos::parallel_for("OpenBoundaryLayer::removeOpenBoundaryAtoms", policy, kernel);
        Kokkos::fence();

        // remove atoms marked for deletion by copying the remaining atoms to the front of the array
        idx_t newNumLocalAtoms = 0;
        for (idx_t idx = 0; idx < atoms.numLocalAtoms; ++idx)
        {
            if (type(idx) != -1)
            {
                if (idx != newNumLocalAtoms)
                {
                    atoms.copy(newNumLocalAtoms, idx);
                }
                ++newNumLocalAtoms;
            }
        }
        atoms.numLocalAtoms = newNumLocalAtoms;
        atoms.resize(atoms.numLocalAtoms + atoms.numGhostAtoms);
    }
};
}  // namespace communication
}  // namespace mrmd