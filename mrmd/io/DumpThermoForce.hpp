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

#include <fstream>

#include "DumpProfile.hpp"
#include "action/ThermodynamicForce.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace io
{
void dumpThermoForce(const std::string& filename,
                     const action::ThermodynamicForce& thermodynamicForce,
                     const idx_t& typeId)
{
    DumpProfile dumpThermoForce;
    auto numBins = thermodynamicForce.getForce().createGrid().size();
    ScalarView::HostMirror forceView("forceView", numBins);
    auto thermoForce = Kokkos::create_mirror_view(Kokkos::HostSpace(), thermodynamicForce.getForce(typeId));
    for (idx_t idx = 0; idx < numBins; ++idx)
    {
        forceView(idx) = thermoForce(idx);
    }

    dumpThermoForce.dump(filename, thermodynamicForce.getForce().createGrid(), forceView);
}
//void dumpThermoForce(const std::string& filename, const action::ThermodynamicForce& thermoForce)
//{
//    DumpProfile dumpThermoForce;
//    dumpThermoForce.open(filename, thermoForce.getForce().createGrid());
//    auto numBins = thermoForce.getForce().createGrid().size();
//
//    for (idx_t typeId = 0; typeId < thermoForce.getForce().numHistograms; typeId++)
//    {
//        ScalarView forceView("forceTest", numBins);
//
//        for (idx_t idx = 0; idx < numBins; ++idx)
//        {
//            forceView(idx) = thermoForce.getForce(typeId)(idx);
//        }
//
//        dumpThermoForce.dumpStep(forceView);
//    }
//    dumpThermoForce.close();
//}
}  // namespace io
}  // namespace mrmd