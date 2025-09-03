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

#include "RestoreThermoForce.hpp"

#include <filesystem>
#include <fstream>

namespace mrmd
{
namespace io
{
action::ThermodynamicForce restoreThermoForce(
    const std::string& filenameForce,
    const data::Subdomain& subdomain,
    const std::vector<real_t>& targetDensities,
    const std::vector<real_t>& thermodynamicForceModulations,
    const bool enforceSymmetry,
    const bool usePeriodicity,
    const idx_t maxNumForces,
    const idx_t& requestedDensityBinNumber)
{
    std::string line;
    std::string word;
    int binNumForce = 0;
    int histNumForce = 0;

    MRMD_HOST_ASSERT_EQUAL(std::filesystem::exists(filenameForce),
                           true,
                           "Thermodynamic force input file does not exist.");

    // Read the thermodynamic force file
    std::ifstream fileThermoForce(filenameForce);
    std::getline(fileThermoForce, line);
    std::stringstream gridLineStream(line);
    while (gridLineStream >> word)
    {
        binNumForce++;
    }
    MRMD_HOST_ASSERT_GREATER(binNumForce, 1);

    MultiView::HostMirror h_forcesRead("h_forcesRead", binNumForce, maxNumForces);

    while (std::getline(fileThermoForce, line))
    {
        binNumForce = 0;
        std::stringstream forceLineStream(line);
        while (forceLineStream >> word)
        {
            h_forcesRead(binNumForce, histNumForce) = std::stod(word);
            binNumForce++;
        }
        histNumForce++;

        MRMD_HOST_ASSERT_LESSEQUAL(histNumForce, maxNumForces);
    }
    fileThermoForce.close();

    auto h_forces = Kokkos::subview(
        h_forcesRead, Kokkos::make_pair(0, binNumForce), Kokkos::make_pair(0, histNumForce));
    MultiView d_forces("d_forces", binNumForce, histNumForce);
    Kokkos::deep_copy(d_forces, h_forces);

    auto forceBinNumber = idx_c(binNumForce);
    idx_t densityBinNumber;

    if (requestedDensityBinNumber == -1)
    {
        densityBinNumber = forceBinNumber;
    }
    else
    {
        densityBinNumber = requestedDensityBinNumber;
    }

    action::ThermodynamicForce thermodynamicForce(targetDensities,
                                                  subdomain,
                                                  forceBinNumber,
                                                  densityBinNumber,
                                                  thermodynamicForceModulations,
                                                  enforceSymmetry,
                                                  usePeriodicity);

    thermodynamicForce.setForce(d_forces);

    return thermodynamicForce;
}
}  // namespace io
}  // namespace mrmd