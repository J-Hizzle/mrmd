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

#include <random>

#include "PositiveNegativeCounter.hpp"
#include "communication/GhostLayer.hpp"
#include "constants.hpp"
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
                                    const bool positive,
                                    const real_t reservoirTemperature,
                                    const real_t reservoirMass,
                                    const real_t dt,
                                    const idx_t typeArg = 0,
                                    const real_t chargeArg = 0_r,
                                    const real_t relativeMassArg = 1_r);

    void insertBoundaryAtoms(data::Atoms& atoms,
                             const data::Subdomain& subdomain,
                             const AXIS& axis,
                             const real_t temperature,
                             const real_t density,
                             const real_t mass,
                             const real_t dt);

    void removeOpenBoundaryAtoms(data::Atoms& atoms, const data::Subdomain& subdomain)
    {
        // capture all slices upfront so both kernels share the same views
        auto pos = atoms.getPos();
        auto vel = atoms.getVel();
        auto force = atoms.getForce();
        auto type = atoms.getType();
        auto mass = atoms.getMass();
        auto charge = atoms.getCharge();
        auto relativeMass = atoms.getRelativeMass();

        // single-pass: inline the boundary check directly in the scan predicate,
        // eliminating a separate mark kernel + fence + full data pass
        Kokkos::View<idx_t*> survivorSrcIndices("survivorSrcIndices", atoms.numLocalAtoms);
        idx_t newNumLocalAtoms = 0;
        Kokkos::parallel_scan(
            "OpenBoundaryLayer::removeOpenBoundaryAtoms",
            Kokkos::RangePolicy<>(0, atoms.numLocalAtoms),
            KOKKOS_LAMBDA(const idx_t idx, idx_t& update, const bool finalPass)
            {
                bool keep = true;
                for (auto dim = 0; dim < DIMENSIONS; ++dim)
                {
                    if (subdomain.boundaryConditions[dim] == data::Subdomain::BoundaryCondition::OPEN)
                    {
                        const auto x = pos(idx, dim);
                        if (x > subdomain.maxCorner[dim] || x < subdomain.minCorner[dim])
                        {
                            keep = false;
                            break;
                        }
                    }
                }
                if (finalPass && keep)
                {
                    survivorSrcIndices(update) = idx;
                }
                if (keep) ++update;
            },
            newNumLocalAtoms);
        Kokkos::fence();

        // scatter surviving atoms into a compact temporary array
        data::Atoms tempAtoms(newNumLocalAtoms);
        auto tempPos = tempAtoms.getPos();
        auto tempVel = tempAtoms.getVel();
        auto tempForce = tempAtoms.getForce();
        auto tempType = tempAtoms.getType();
        auto tempMass = tempAtoms.getMass();
        auto tempCharge = tempAtoms.getCharge();
        auto tempRelativeMass = tempAtoms.getRelativeMass();

        Kokkos::parallel_for(
            "OpenBoundaryLayer::scatterSurvivors",
            Kokkos::RangePolicy<>(0, newNumLocalAtoms),
            KOKKOS_LAMBDA(const idx_t newIdx)
            {
                const idx_t srcIdx = survivorSrcIndices(newIdx);
                for (auto dim = 0; dim < DIMENSIONS; ++dim)
                {
                    tempPos(newIdx, dim) = pos(srcIdx, dim);
                    tempVel(newIdx, dim) = vel(srcIdx, dim);
                    tempForce(newIdx, dim) = force(srcIdx, dim);
                }
                tempType(newIdx) = type(srcIdx);
                tempMass(newIdx) = mass(srcIdx);
                tempCharge(newIdx) = charge(srcIdx);
                tempRelativeMass(newIdx) = relativeMass(srcIdx);
            });
        Kokkos::fence();

        tempAtoms.numLocalAtoms = newNumLocalAtoms;
        tempAtoms.numGhostAtoms = 0;
        data::deep_copy(atoms, tempAtoms);
    }

    void insertOpenBoundaryAtoms(data::Atoms& atoms,
                                 const data::Subdomain& subdomain,
                                 const real_t reservoirTemperature,
                                 const real_t reservoirDensity,
                                 const real_t reservoirMass,
                                 const real_t dt)
    {
        for (auto boundaryAxis = 0; boundaryAxis < DIMENSIONS; ++boundaryAxis)
        {
            if (subdomain.boundaryConditions[boundaryAxis] ==
                data::Subdomain::BoundaryCondition::OPEN)
            {
                insertBoundaryAtoms(atoms,
                                    subdomain,
                                    static_cast<AXIS>(boundaryAxis),
                                    reservoirTemperature,
                                    reservoirDensity,
                                    reservoirMass,
                                    dt);
            }
        }
    }

    idx_t sampleHalfNumberOfAtomsToInsert(const data::Subdomain& subdomain,
                                      const AXIS& axis,
                                      const real_t reservoirTemperature,
                                      const real_t reservoirDensity,
                                      const real_t reservoirMass,
                                      const real_t dt);

    OpenBoundaryLayer(idx_t seed) : randPool_(seed), hostRng_(static_cast<std::mt19937_64::result_type>(seed)) {};

private:
    Kokkos::Random_XorShift1024_Pool<> randPool_ = Kokkos::Random_XorShift1024_Pool<>(1234);
    std::mt19937_64 hostRng_{1234};
};

void OpenBoundaryLayer::insertBoundaryAtoms(data::Atoms& atoms,
                                            const data::Subdomain& subdomain,
                                            const AXIS& axis,
                                            const real_t reservoirTemperature,
                                            const real_t reservoirDensity,
                                            const real_t reservoirMass,
                                            const real_t dt)
{
    // sample number of atoms to be inserted according to density distribution
    auto numberOfAtomsToInsertNegative = sampleHalfNumberOfAtomsToInsert(
        subdomain, axis, reservoirTemperature, reservoirDensity, reservoirMass, dt);
    auto numberOfAtomsToInsertPositive = sampleHalfNumberOfAtomsToInsert(
        subdomain, axis, reservoirTemperature, reservoirDensity, reservoirMass, dt);

    // create atom buffers and copy atoms to be inserted into them
    auto atomsToInsertNegative =
        createBoundaryAtoms(subdomain, axis, numberOfAtomsToInsertNegative, false, reservoirTemperature, reservoirMass, dt);
    auto atomsToInsertPositive =
        createBoundaryAtoms(subdomain, axis, numberOfAtomsToInsertPositive, true, reservoirTemperature, reservoirMass, dt);

    // concatenate the new atoms with the existing ones
    util::concatenateRealAtoms(atoms, atomsToInsertNegative);
    util::concatenateRealAtoms(atoms, atomsToInsertPositive);
}

data::Atoms OpenBoundaryLayer::createBoundaryAtoms(const data::Subdomain& subdomain,
                                                   const AXIS& axis,
                                                   const idx_t numAtoms,
                                                   const bool positive,
                                                   const real_t reservoirTemperature,
                                                   const real_t reservoirMass,
                                                   const real_t dt,
                                                   const idx_t typeArg,
                                                   const real_t chargeArg,
                                                   const real_t relativeMassArg)
{
    data::Atoms boundaryAtoms(numAtoms);

    auto RNG = randPool_;
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
        auto randGen = RNG.get_state();
        auto sigma = std::sqrt(reservoirTemperature / reservoirMass);
        auto sign = positive ? -1_r : 1_r;

        // set position of new atom
        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            if (dim == to_underlying(axis))
            {
                auto velocity = sign * sigma * std::sqrt(-2_r * std::log(randGen.drand()));  // sample velocity from Rayleigh distribution and directed towards the interior of the domain

                vel(idx, dim) = velocity;

                auto corner = positive ? subdomain.maxCorner[dim] : subdomain.minCorner[dim];
                auto offset = randGen.drand() * velocity * dt;  // random position offset
                pos(idx, dim) = corner + offset;  // set position of new atom at the boundary with random offset
            }
            else
            {
                pos(idx, dim) =
                    randGen.drand() * subdomain.diameter[dim] + subdomain.minCorner[dim];
                
                vel(idx, dim) = sigma * randGen.normal();  // set velocity according to Maxwell-Boltzmann distribution
            }
        }

        RNG.free_state(randGen);

        // set other properties of new atom
        force(idx, 0) = 0_r;
        force(idx, 1) = 0_r;
        force(idx, 2) = 0_r;
        type(idx) = typeArg;
        mass(idx) = reservoirMass;
        charge(idx) = chargeArg;
        relativeMass(idx) = relativeMassArg;
    };
    Kokkos::parallel_for("OpenBoundaryLayer::createBoundaryAtoms", policy, kernel);

    boundaryAtoms.numLocalAtoms = numAtoms;
    boundaryAtoms.numGhostAtoms = 0;
    return boundaryAtoms;
}

idx_t OpenBoundaryLayer::sampleHalfNumberOfAtomsToInsert(const data::Subdomain& subdomain,
                                                     const AXIS& axis,
                                                     const real_t reservoirTemperature,
                                                     const real_t reservoirDensity,
                                                     const real_t reservoirMass,
                                                     const real_t dt)
{
    real_t fractionalNumberOfAtomsToInsert = 0_r;
    fractionalNumberOfAtomsToInsert = reservoirDensity * subdomain.getAreaNormalToAxis(axis) * dt *
         std::sqrt(reservoirTemperature / (2 * pi * reservoirMass));
    idx_t integerNumberOfAtomsToInsert = std::floor(fractionalNumberOfAtomsToInsert);
    std::uniform_real_distribution<real_t> dist(0_r, 1_r);
    auto randnum = dist(hostRng_);
    if (randnum < fractionalNumberOfAtomsToInsert - integerNumberOfAtomsToInsert)
    {
        ++integerNumberOfAtomsToInsert;
    }
    return integerNumberOfAtomsToInsert;
}
}  // namespace communication
}  // namespace mrmd