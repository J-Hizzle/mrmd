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

#include "PositiveNegativeCounter.hpp"
#include "communication/GhostLayer.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "util/concatenation.hpp"

namespace mrmd
{
namespace communication
{
class OpenBoundaryLayer
{
public:
    data::Atoms createBoundaryAtoms(const data::Subdomain& subdomain,
                                    const AXIS& axis,
                                    const idx_t numAtoms,
                                    const bool positive);

    void insertBoundaryAtoms(data::Atoms& atoms,
                             const data::Subdomain& subdomain,
                             const AXIS& axis);

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
        for (auto boundaryAxis = 0; boundaryAxis < DIMENSIONS; ++boundaryAxis)
        {
            if (subdomain.boundaryConditions[boundaryAxis] ==
                data::Subdomain::BoundaryCondition::OPEN)
            {
                insertBoundaryAtoms(atoms, subdomain, static_cast<AXIS>(boundaryAxis));
            }
        }
        // reconstruct ghost layer for the new local atoms
        communication::GhostLayer ghostLayer;
        ghostLayer.exchangeRealAtoms(atoms, subdomain);
        ghostLayer.createGhostAtoms(atoms, subdomain);
    }

    OpenBoundaryLayer() : numberOfAtomsToInsert_("numberOfAtomsToInsert") {}

private:
    /// number of atoms to insert in each direction
    Kokkos::View<idx_t[2]> numberOfAtomsToInsert_;
};

void OpenBoundaryLayer::insertBoundaryAtoms(data::Atoms& atoms,
                                            const data::Subdomain& subdomain,
                                            const AXIS& axis)
{
    auto numberOfAtomsToInsertNegative =
        1000;  // TODO: sample number of atoms to be inserted according to density distribution
    auto numberOfAtomsToInsertPositive =
        1000;  // TODO: sample number of atoms to be inserted according to density distribution

    // create atom buffers and copy atoms to be inserted into them
    auto atomsToInsertNegative =
        createBoundaryAtoms(subdomain, axis, numberOfAtomsToInsertNegative, false);
    auto atomsToInsertPositive =
        createBoundaryAtoms(subdomain, axis, numberOfAtomsToInsertPositive, true);

    // concatenate the new atoms with the existing ones
    util::concatenateRealAtoms(atoms, atomsToInsertNegative);
    util::concatenateRealAtoms(atoms, atomsToInsertPositive);
}

data::Atoms OpenBoundaryLayer::createBoundaryAtoms(const data::Subdomain& subdomain,
                                                   const AXIS& axis,
                                                   const idx_t numAtoms,
                                                   const bool positive)
{
    auto RNG = Kokkos::Random_XorShift1024_Pool<>(1234);

    data::Atoms boundaryAtoms(numAtoms);

    auto pos = boundaryAtoms.getPos();
    auto vel = boundaryAtoms.getVel();
    auto force = boundaryAtoms.getForce();
    auto type = boundaryAtoms.getType();
    auto mass = boundaryAtoms.getMass();
    auto charge = boundaryAtoms.getCharge();
    auto relativeMass = boundaryAtoms.getRelativeMass();

    auto policy = Kokkos::RangePolicy<>(0, numAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        // set position of new atom
        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            if (dim == to_underlying(axis))
            {
                if (positive)
                {
                    pos(idx, dim) =
                        subdomain.maxCorner[dim] -
                        1e-5_r;  // TODO: sample position according to density distribution
                }
                else
                {
                    pos(idx, dim) =
                        subdomain.minCorner[dim] +
                        1e-5_r;  // TODO: sample position according to density distribution
                }
            }
            else
            {
                auto randGen = RNG.get_state();
                pos(idx, dim) =
                    randGen.drand() * subdomain.diameter[dim] + subdomain.minCorner[dim];
                RNG.free_state(randGen);
            }
        }

        // set other properties of new atom
        vel(idx, 0) = 0_r;
        vel(idx, 1) = 0_r;
        vel(idx, 2) = 0_r;
        force(idx, 0) = 0_r;
        force(idx, 1) = 0_r;
        force(idx, 2) = 0_r;
        type(idx) = 0;                  // TODO: set type according to simulation setup
        mass(idx) = 1_r;                // TODO: set mass according to simulation setup
        charge(idx) = 0_r;              // TODO: set charge according to simulation setup
        relativeMass(idx) = mass(idx);  // TODO: set relative mass according to simulation setup
    };
    Kokkos::parallel_for("OpenBoundaryLayer::createBoundaryAtoms", policy, kernel);

    boundaryAtoms.numLocalAtoms = numAtoms;
    boundaryAtoms.numGhostAtoms = 0;
    return boundaryAtoms;
}
}  // namespace communication
}  // namespace mrmd