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
    : planeGrid_(planeGrid),
      axis_(axis),
      distancesToPlanes_("distancesToPlanes", 0, planeGrid.extent(0))
{
}

void AxialMassFluxProfile::startCounting(data::Atoms& atoms)
{
    auto planeGrid = planeGrid_;
    auto axis = axis_;
    auto numAtoms = atoms.numLocalAtoms + atoms.numGhostAtoms;

    auto pos = atoms.getPos();

    Kokkos::resize(distancesToPlanes_, idx_c(numAtoms * 1.1_r), distancesToPlanes_.extent(1));
    auto distancesToPlanes = distancesToPlanes_;

    MRMD_DEVICE_CHECK_GREATEREQUAL(idx_c(distancesToPlanes.extent(0)), numAtoms);
    MRMD_DEVICE_CHECK_EQUAL(idx_c(distancesToPlanes.extent(1)), idx_c(planeGrid.extent(0)));

    Kokkos::parallel_for(
        "ComputeDistancesToPlanes",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {numAtoms, idx_c(planeGrid.extent(0))}),
        KOKKOS_LAMBDA(const idx_t idx, const idx_t jdx) {
            distancesToPlanes(idx, jdx) = (planeGrid(jdx) - pos(idx, to_underlying(axis)));
        });
}

ScalarView AxialMassFluxProfile::stopCounting(data::Atoms& atoms)
{
    auto planeGrid = planeGrid_;
    auto axis = axis_;
    auto numAtoms = atoms.numLocalAtoms + atoms.numGhostAtoms;

    auto pos = atoms.getPos();
    auto mass = atoms.getMass();

    MRMD_HOST_CHECK_GREATEREQUAL(idx_c(distancesToPlanes_.extent(0)),
                                 numAtoms,
                                 "You must call startCounting before stopCounting. The number of "
                                 "particles is not allowed to change!");
    auto distancesToPlanes = distancesToPlanes_;
    ScalarView massFlux("massFlux", idx_c(planeGrid.extent(0)));
    Kokkos::parallel_for(
        "CalculateMassFlux",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {numAtoms, idx_c(planeGrid.extent(0))}),
        KOKKOS_LAMBDA(const idx_t idx, const idx_t jdx) {
            auto dist = (planeGrid(jdx) - pos(idx, to_underlying(axis)));
            if (dist * distancesToPlanes(idx, jdx) < 0)
            {
                Kokkos::atomic_add(&massFlux(jdx), (dist > 0) ? mass(idx) : -mass(idx));
            }
        });
    return massFlux;
}

}  // namespace mrmd::analysis