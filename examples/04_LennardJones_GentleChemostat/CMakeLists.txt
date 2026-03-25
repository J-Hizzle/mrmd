# Copyright 2024 Sebastian Eibl
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

add_executable(04_LennardJones_GentleChemostat
        04_LennardJones_GentleChemostat.cpp
        )
target_link_libraries(04_LennardJones_GentleChemostat
        PRIVATE mrmd CLI11::CLI11
        )
add_test(NAME Example.04_LennardJones_GentleChemostat COMMAND ./04_LennardJones_GentleChemostat --nsteps 10)

add_executable(04_LennardJones_EquilibrateBerendsen
        04_LennardJones_EquilibrateBerendsen.cpp
        )
target_link_libraries(04_LennardJones_EquilibrateBerendsen
        PRIVATE mrmd CLI11::CLI11
        )
add_executable(04_LennardJones_EquilibrateLangevin
        04_LennardJones_EquilibrateLangevin.cpp
        )
target_link_libraries(04_LennardJones_EquilibrateLangevin
        PRIVATE mrmd CLI11::CLI11
        )