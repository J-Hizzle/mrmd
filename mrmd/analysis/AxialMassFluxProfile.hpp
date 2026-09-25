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

#pragma once

#include "data/Atoms.hpp"
#include "datatypes.hpp"

namespace mrmd::analysis
{
/// Computes the axial mass flux profile across a set of parallel planes.
class AxialMassFluxProfile
{
    ScalarView planeGrid_;
    AXIS axis_;
    MultiView distancesToPlane_;

public:
    AxialMassFluxProfile(const ScalarView &planeGrid, const AXIS axis);

    /**
     * @brief Records the current positions of all particles relative to the planes.
     *
     * This method should be called before tracking particle crossings.
     */
    void startCounting(data::Atoms &atoms);

    /**
     * @brief Counts how many particles have crossed the plane since startCounting() was called.
     *
     * This method compares the current positions of the particles to their positions
     * recorded by startCounting() and determines how many have crossed the plane.
     */
    IndexView stopCounting(data::Atoms &atoms);
};
}  // namespace mrmd::analysis