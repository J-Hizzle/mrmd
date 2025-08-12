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

#include "util/interpolation.hpp"

namespace mrmd
{
namespace util
{
data::MultiHistogram interpolate(const data::MultiHistogram& input, const ScalarView& grid)
{
    data::MultiHistogram output(
        "interpolated-profile", input.min, input.max, idx_c(grid.extent(0)), input.numHistograms);
    const ScalarView& inputGrid = input.createGrid_d();

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {idx_t(0), idx_t(0)}, {idx_c(grid.extent(0)), input.numHistograms});
    auto kernel = KOKKOS_LAMBDA(const idx_t binIdx, const idx_t histogramIdx)
    {
        // find the two enclosing bins in the input histogram
        auto rightBin = findRightBin(inputGrid, grid(binIdx));
        auto leftBin = rightBin - 1;
        output.data(binIdx, histogramIdx) =
            lerp(input.data(leftBin, histogramIdx),
                 input.data(rightBin, histogramIdx),
                 (grid(binIdx) - inputGrid(leftBin)) * input.inverseBinSize);
    };
    Kokkos::parallel_for("MultiHistogram::interpolate", policy, kernel);
    Kokkos::fence();

    return output;
}
}  // namespace util
}  // namespace mrmd