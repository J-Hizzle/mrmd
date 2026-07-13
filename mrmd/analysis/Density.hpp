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

#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace analysis
{
    idx_t getNumberOfParticles(const data::Atoms& atoms)
    {
        return atoms.numLocalAtoms;
    }

    template <OnePositionPredicate Pred>
    idx_t getNumberOfParticles_if(const data::Atoms& atoms, const Pred& pred)
    {
        auto pos = atoms.getPos();
        auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
        auto kernel = KOKKOS_LAMBDA(const idx_t& idx, real_t& count)
        {
            if (pred(pos(idx, 0), pos(idx, 1), pos(idx, 2)))
            {
                count += 1_r;
            }
        };

        real_t count = 0_r;
        Kokkos::parallel_reduce("getNumberOfParticles_if", policy, kernel, count);

        return count;
    }

    real_t getDensity(const data::Atoms& atoms, const data::Subdomain& subdomain)
    {
        auto volume = subdomain.getVolume();
        return getNumberOfParticles(atoms) / volume;
    }

    template <OnePositionPredicate Pred>
    real_t getDensity_if(const data::Atoms& atoms, const data::Subdomain& subdomain, const Pred& pred)
    {
        auto volume = pred.getVolume(subdomain);
        return getNumberOfParticles_if(atoms, pred) / volume;
    }
}  // namespace analysis
}  // namespace mrmd