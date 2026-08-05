# Copyright 2024 Sebastian Eibl
# Copyright 2026 Julian Friedrich Hille
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_executable(ScaleQuenching_EquilibrateBerendsen
        ScaleQuenching_EquilibrateBerendsen.cpp
        )
target_link_libraries(ScaleQuenching_EquilibrateBerendsen
        PRIVATE mrmd CLI11::CLI11
        )
add_test(NAME Example.ScaleQuenching_EquilibrateBerendsen COMMAND ./ScaleQuenching_EquilibrateBerendsen --nsteps 10)

add_executable(ScaleQuenching_EquilibrateLangevin
        ScaleQuenching_EquilibrateLangevin.cpp
        )
target_link_libraries(ScaleQuenching_EquilibrateLangevin
        PRIVATE mrmd CLI11::CLI11
        )