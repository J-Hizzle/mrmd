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

    void insertOpenBoundaryAtoms(data::Atoms& atoms, const data::Subdomain& subdomain)
    {
        idx_t numInsertedAtoms = 0;

        // sample how many atoms are supposed to be inserted

        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            if (subdomain.boundaryConditions[dim] == data::Subdomain::BoundaryCondition::OPEN)
            {
                numInsertedAtoms += 1;  // TODO: sample number of atoms to be inserted according to density distribution
            }
        }

        std::cout << "numInsertedAtoms: " << numInsertedAtoms << std::endl;

        atoms.numLocalAtoms += numInsertedAtoms;
        atoms.resize(atoms.numLocalAtoms + atoms.numGhostAtoms);

        // insert new atoms at the end of the array
        auto pos = atoms.getPos();
        auto vel = atoms.getVel();
        auto force = atoms.getForce();
        auto type = atoms.getType();
        auto mass = atoms.getMass();
        auto charge = atoms.getCharge();
        auto relativeMass = atoms.getRelativeMass();

        auto policy = Kokkos::RangePolicy<>(atoms.numLocalAtoms - numInsertedAtoms, atoms.numLocalAtoms);
        auto kernel = KOKKOS_LAMBDA(const idx_t idx)
        {
            atoms.copy(idx + numInsertedAtoms, idx); // shift existing ghost atoms to make space for new atoms at the end of the array
            for (auto dim = 0; dim < DIMENSIONS; ++dim)
            {
                pos(idx, dim) = 0_r;  // TODO: sample position of new atom according to density distribution
                vel(idx, dim) = 0_r;  // TODO: sample velocity of new atom according to velocity distribution
                force(idx, dim) = 0_r;
            }
            type(idx) = 0;          // TODO: set type of new atom according to type distribution
            mass(idx) = 1_r;        // TODO: set mass of new atom according to mass distribution
            charge(idx) = 0_r;      // TODO: set charge of new atom according to charge distribution
            relativeMass(idx) = 1_r / mass(idx);  // TODO: set relative mass of new atom according to relative mass distribution
        };
        Kokkos::parallel_for("fillDomainWithAtoms", policy, kernel);
    }
};
}  // namespace communication
}  // namespace mrmd