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

#include "assert/assert.hpp"
#include "util/Kokkos_grow.hpp"

namespace mrmd::analysis
{
AxialMassFluxProfile::AxialMassFluxProfile(const ScalarView& planeGrid, const AXIS axis)
    : planeGrid_(planeGrid), axis_(axis), distancesToPlane_("distancesToPlane", 0, 0)
{
}

void AxialMassFluxProfile::startCounting(data::Atoms& atoms)
{
    auto planeGrid = planeGrid_;
    auto axis = axis_;

    auto pos = atoms.getPos();
    util::grow(distancesToPlane_, idx_c(atoms.size()));
    auto distancesToPlane = distancesToPlane_;
    Kokkos::parallel_for(
        "ComputeDistancesToPlanes",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {atoms.size(), planeGrid.size()}),
        KOKKOS_LAMBDA(const idx_t idx, const idx_t jdx) {
            distancesToPlane(idx, jdx) = (planeGrid(jdx) - pos(idx, to_underlying(axis)));
        });
}

IndexView AxialMassFluxProfile::stopCounting(data::Atoms& atoms)
{
    auto planeGrid = planeGrid_;
    auto axis = axis_;

    auto pos = atoms.getPos();
    MRMD_HOST_CHECK_GREATEREQUAL(distancesToPlane_.size(),
                                 pos.size(),
                                 "You must call startCounting before stopCounting. The number of "
                                 "particles is not allowed to change!");
    auto distancesToPlane = distancesToPlane_;
    IndexView counts("counts", planeGrid.size());
    Kokkos::parallel_for(
        "CountCrossings",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {atoms.size(), planeGrid.size()}),
        KOKKOS_LAMBDA(const idx_t idx, const idx_t jdx) {
            auto dist = (planeGrid(jdx) - pos(idx, to_underlying(axis)));
            if (dist * distancesToPlane(idx, jdx) < 0)
            {
                Kokkos::atomic_add(&counts(jdx), (dist > 0) ? 1 : -1);
            }
        });
    return counts;
}

}  // namespace mrmd::analysis