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

#include <iomanip>
#include <iostream>

#include "datatypes.hpp"

namespace mrmd
{
namespace util
{
template <int PRECISION = 2, typename HEAD>
void printTable(HEAD head)
{
    std::cout << " │ " << std::setw(8 + PRECISION) << std::setprecision(PRECISION) << std::fixed
              << head << " │ " << std::endl;
}
template <int PRECISION = 2, typename HEAD, typename... TAIL>
void printTable(HEAD head, TAIL... tail)
{
    std::cout << " │ " << std::setw(8 + PRECISION) << std::setprecision(PRECISION) << std::fixed
              << head;
    printTable<PRECISION>(tail...);
}

template <int PRECISION = 2, typename HEAD>
void printTableSep(HEAD /*head*/)
{
    std::string sep;
    for (int i = 0; i < 8 + PRECISION; ++i) sep += "─";
    std::cout << "─┼─" << sep << "─┼─" << std::endl;
}
template <int PRECISION = 2, typename HEAD, typename... TAIL>
void printTableSep(HEAD /*head*/, TAIL... tail)
{
    std::string sep;
    for (int i = 0; i < 8 + PRECISION; ++i) sep += "─";
    std::cout << "─┼─" << sep;
    printTableSep<PRECISION>(tail...);
}

}  // namespace util
}  // namespace mrmd
