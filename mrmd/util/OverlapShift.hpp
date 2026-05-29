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
#include "datatypes.hpp"
#include "util/math.hpp"

namespace mrmd
{
namespace util
{
class OverlapShift
{
private:
    real_t shiftPosition_;

public:
    OverlapShift(const real_t shiftPosition)
        : shiftPosition_(shiftPosition)
    {}

    template <OnePositionPredicate Pred>
    void shift_if(data::Atoms& atoms, const Pred& pred);
};

template <OnePositionPredicate Pred>
void OverlapShift::shift_if(data::Atoms& atoms, const Pred& pred)
{
    auto pos = atoms.getPos();
    auto overlap = atoms.getOverlap();
    auto shiftPosition = shiftPosition_;

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        if (overlap(idx))
        {
            if (pred(pos(idx, 0), pos(idx, 1), pos(idx, 2)))
            {
                pos(idx, 0) = shiftPosition;
            }
        }
        overlap(idx) = false;  // reset overlap flag for next iteration
    };
    Kokkos::parallel_for("OverlapShift::shift_if",
                            policy,
                            kernel);
    Kokkos::fence();
}

}  // namespace util
}  // namespace mrmd