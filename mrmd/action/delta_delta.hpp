// Copyright 2024 Sebastian Eibl
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

#include "LennardJones.hpp"
#include "assert/assert.hpp"
#include "data/Atoms.hpp"
#include "data/Molecules.hpp"
#include "data/MultiHistogram.hpp"
#include "datatypes.hpp"
#include "weighting_function/CheckRegion.hpp"

namespace mrmd
{
namespace action
{
class LJ_IdealGas
{
private:
    impl::CappedLennardJonesPotential LJ_;
    real_t rcSqr_ = 0_r;

    data::Molecules::pos_t moleculesPos_;
    data::Molecules::force_t::atomic_access_slice moleculesForce_;
    data::Molecules::lambda_t moleculesLambda_;
    data::Molecules::modulated_lambda_t moleculesModulatedLambda_;
    data::Molecules::grad_lambda_t moleculesGradLambda_;
    data::Molecules::atoms_offset_t moleculesAtomsOffset_;
    data::Molecules::num_atoms_t moleculesNumAtoms_;

    data::Atoms::pos_t atomsPos_;
    data::Atoms::force_t::atomic_access_slice atomsForce_;
    data::Atoms::type_t atomsType_;

    HalfVerletList verletList_;

    idx_t runCounter_ = 0;
    idx_t numTypes_;

public:
    /**
     * Loop over molecules
     *
     * @param alpha first molecule index
     */
    KOKKOS_INLINE_FUNCTION void operator()(const idx_t& alpha, real_t& sumEnergy) const
    {
        // avoid atomic force contributions to idx in innermost loop
        real_t forceTmpAlpha[3] = {0_r, 0_r, 0_r};

        /// weighting for molecule alpha
        const auto modulatedLambdaAlpha = moleculesModulatedLambda_(alpha);
        assert(0_r <= modulatedLambdaAlpha);
        assert(modulatedLambdaAlpha <= 1_r);

        idx_t binAlpha = -1;
        const real_t gradLambdaAlpha[3] = {moleculesGradLambda_(alpha, 0),
                                           moleculesGradLambda_(alpha, 1),
                                           moleculesGradLambda_(alpha, 2)};

        /// inclusive start index of atoms belonging to alpha
        const auto startAtomsAlpha = moleculesAtomsOffset_(alpha);
        /// exclusive end index of atoms belonging to alpha
        const auto endAtomsAlpha = startAtomsAlpha + moleculesNumAtoms_(alpha);
        assert(0 <= startAtomsAlpha);
        assert(startAtomsAlpha < endAtomsAlpha);
        //            assert(endAtomsAlpha <= atoms_.numLocalAtoms +
        //            atoms_.numGhostAtoms);

        const auto numNeighbors = idx_c(HalfNeighborList::numNeighbor(verletList_, alpha));
        for (idx_t n = 0; n < numNeighbors; ++n)
        {
            /// second molecule index
            const idx_t beta = idx_c(HalfNeighborList::getNeighbor(verletList_, alpha, n));
            assert(0 <= beta);

            // avoid atomic force contributions to idx in innermost loop
            real_t forceTmpBeta[3] = {0_r, 0_r, 0_r};

            /// weighting for molecule beta
            const auto modulatedLambdaBeta = moleculesModulatedLambda_(beta);
            assert(0_r <= modulatedLambdaBeta);
            assert(modulatedLambdaBeta <= 1_r);

            if (((weighting_function::isInATRegion(modulatedLambdaAlpha)) && (weighting_function::isInCGRegion(modulatedLambdaBeta))) || ((weighting_function::isInATRegion(modulatedLambdaBeta)) && (weighting_function::isInCGRegion(modulatedLambdaAlpha))))
            {
                // interaction through delta/delta interface => no interaction
                continue;
            }

            const real_t gradLambdaBeta[3] = {moleculesGradLambda_(beta, 0),
                                              moleculesGradLambda_(beta, 1),
                                              moleculesGradLambda_(beta, 2)};

            /// inclusive start index of atoms belonging to beta
            const auto startAtomsBeta = moleculesAtomsOffset_(beta);
            /// exclusive end index of atoms belonging to beta
            const auto endAtomsBeta = startAtomsBeta + moleculesNumAtoms_(beta);
            assert(0 <= startAtomsBeta);
            assert(startAtomsBeta < endAtomsBeta);
            //            assert(endAtomsBeta <= atoms_.numLocalAtoms +
            //            atoms_.numGhostAtoms);

            /// loop over atoms
            for (idx_t idx = startAtomsAlpha; idx < endAtomsAlpha; ++idx)
            {
                real_t posTmp[3];
                posTmp[0] = atomsPos_(idx, 0);
                posTmp[1] = atomsPos_(idx, 1);
                posTmp[2] = atomsPos_(idx, 2);

                // avoid atomic force contributions to idx in innermost loop
                real_t forceTmpIdx[3] = {0_r, 0_r, 0_r};

                for (idx_t jdx = startAtomsBeta; jdx < endAtomsBeta; ++jdx)
                {
                    const auto dx = posTmp[0] - atomsPos_(jdx, 0);
                    const auto dy = posTmp[1] - atomsPos_(jdx, 1);
                    const auto dz = posTmp[2] - atomsPos_(jdx, 2);

                    const auto distSqr = dx * dx + dy * dy + dz * dz;

                    if (distSqr > rcSqr_) continue;

                    auto typeIdx = atomsType_(idx) * numTypes_ + atomsType_(jdx);
                    MRMD_DEVICE_ASSERT_GREATEREQUAL(typeIdx, 0);
                    MRMD_DEVICE_ASSERT_LESS(typeIdx, numTypes_ * numTypes_);
                    MRMD_DEVICE_ASSERT(!std::isnan(distSqr));
                    auto forceAndEnergy = LJ_.computeForceAndEnergy(distSqr, typeIdx);
                    auto ffactor = forceAndEnergy.forceFactor;
                    MRMD_DEVICE_ASSERT(!std::isnan(ffactor));

                    forceTmpIdx[0] += dx * ffactor;
                    forceTmpIdx[1] += dy * ffactor;
                    forceTmpIdx[2] += dz * ffactor;

                    atomsForce_(jdx, 0) -= dx * ffactor;
                    atomsForce_(jdx, 1) -= dy * ffactor;
                    atomsForce_(jdx, 2) -= dz * ffactor;

                    MRMD_DEVICE_ASSERT(!std::isnan(forceAndEnergy.energy));
                    sumEnergy += forceAndEnergy.energy;
                    auto Vij = 0.5_r * forceAndEnergy.energy;
                }

                atomsForce_(idx, 0) += forceTmpIdx[0];
                atomsForce_(idx, 1) += forceTmpIdx[1];
                atomsForce_(idx, 2) += forceTmpIdx[2];
            }

            moleculesForce_(beta, 0) += forceTmpBeta[0];
            moleculesForce_(beta, 1) += forceTmpBeta[1];
            moleculesForce_(beta, 2) += forceTmpBeta[2];
        }

        moleculesForce_(alpha, 0) += forceTmpAlpha[0];
        moleculesForce_(alpha, 1) += forceTmpAlpha[1];
        moleculesForce_(alpha, 2) += forceTmpAlpha[2];
    }

    real_t run(data::Molecules& molecules, HalfVerletList& verletList, data::Atoms& atoms)
    {
        moleculesPos_ = molecules.getPos();
        moleculesForce_ = molecules.getForce();
        moleculesLambda_ = molecules.getLambda();
        moleculesModulatedLambda_ = molecules.getModulatedLambda();
        moleculesGradLambda_ = molecules.getGradLambda();
        moleculesAtomsOffset_ = molecules.getAtomsOffset();
        moleculesNumAtoms_ = molecules.getNumAtoms();
        atomsPos_ = atoms.getPos();
        atomsForce_ = atoms.getForce();
        atomsType_ = atoms.getType();
        verletList_ = verletList;

        real_t energy = 0_r;
        auto policy = Kokkos::RangePolicy<>(0, molecules.numLocalMolecules);
        Kokkos::parallel_reduce("LJ_IdealGas::applyForces", policy, *this, energy);

        Kokkos::fence();

        return energy;
    }

    LJ_IdealGas(const real_t& cappingDistance,
                const real_t& rc,
                const real_t& sigma,
                const real_t& epsilon,
                const bool doShift)
        : LJ_IdealGas({cappingDistance}, {rc}, {sigma}, {epsilon}, 1, doShift)
    {
    }

    LJ_IdealGas(const std::vector<real_t>& cappingDistance,
                const std::vector<real_t>& rc,
                const std::vector<real_t>& sigma,
                const std::vector<real_t>& epsilon,
                const idx_t numTypes,
                const bool doShift)
        : LJ_(cappingDistance, rc, sigma, epsilon, numTypes, doShift), numTypes_(numTypes)
    {
        MRMD_HOST_ASSERT_EQUAL(cappingDistance.size(), numTypes * numTypes);
        MRMD_HOST_ASSERT_EQUAL(rc.size(), numTypes * numTypes);
        MRMD_HOST_ASSERT_EQUAL(sigma.size(), numTypes * numTypes);
        MRMD_HOST_ASSERT_EQUAL(epsilon.size(), numTypes * numTypes);

        auto maxRC = *std::max_element(rc.begin(), rc.end());
        rcSqr_ = maxRC * maxRC;
    }
};

}  // namespace action
}  // namespace mrmd