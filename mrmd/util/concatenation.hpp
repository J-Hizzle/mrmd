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

#include "data/Atoms.hpp"

namespace mrmd::util
{
void concatenateRealAtoms(data::Atoms& dst, const data::Atoms& src)
{
    const idx_t oldNumLocalAtoms = dst.numLocalAtoms;
    const idx_t srcNumLocalAtoms = src.numLocalAtoms;

    dst.resize(oldNumLocalAtoms + srcNumLocalAtoms);

    auto dstPos = dst.getPos();
    auto dstVel = dst.getVel();
    auto dstForce = dst.getForce();
    auto dstMass = dst.getMass();
    auto dstType = dst.getType();
    auto dstCharge = dst.getCharge();
    auto dstRelativeMass = dst.getRelativeMass();

    auto srcPos = src.getPos();
    auto srcVel = src.getVel();
    auto srcForce = src.getForce();
    auto srcMass = src.getMass();
    auto srcType = src.getType();
    auto srcCharge = src.getCharge();
    auto srcRelativeMass = src.getRelativeMass();

    auto policy = Kokkos::RangePolicy<>(0, srcNumLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        dstPos(oldNumLocalAtoms + idx, 0) = srcPos(idx, 0);
        dstPos(oldNumLocalAtoms + idx, 1) = srcPos(idx, 1);
        dstPos(oldNumLocalAtoms + idx, 2) = srcPos(idx, 2);
        dstVel(oldNumLocalAtoms + idx, 0) = srcVel(idx, 0);
        dstVel(oldNumLocalAtoms + idx, 1) = srcVel(idx, 1);
        dstVel(oldNumLocalAtoms + idx, 2) = srcVel(idx, 2);
        dstForce(oldNumLocalAtoms + idx, 0) = srcForce(idx, 0);
        dstForce(oldNumLocalAtoms + idx, 1) = srcForce(idx, 1);
        dstForce(oldNumLocalAtoms + idx, 2) = srcForce(idx, 2);
        dstMass(oldNumLocalAtoms + idx) = srcMass(idx);
        dstType(oldNumLocalAtoms + idx) = srcType(idx);
        dstCharge(oldNumLocalAtoms + idx) = srcCharge(idx);
        dstRelativeMass(oldNumLocalAtoms + idx) = srcRelativeMass(idx);
    };
    Kokkos::parallel_for("concatenate", policy, kernel);
    Kokkos::fence();

    dst.numLocalAtoms += srcNumLocalAtoms;
    dst.numGhostAtoms = 0;
}
}  // namespace mrmd::util