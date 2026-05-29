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

add_executable(04_LennardJones_GentleChemostat_update
        04_LennardJones_GentleChemostat_update.cpp
        )
target_link_libraries(04_LennardJones_GentleChemostat_update
        PRIVATE mrmd CLI11::CLI11
        )
add_executable(04_LennardJones_GentleChemostat_updateIf
        04_LennardJones_GentleChemostat_updateIf.cpp
        )
target_link_libraries(04_LennardJones_GentleChemostat_updateIf
        PRIVATE mrmd CLI11::CLI11
        )
add_executable(04_LennardJones_GentleChemostat_applyInterpolated
        04_LennardJones_GentleChemostat_applyInterpolated.cpp
        )
target_link_libraries(04_LennardJones_GentleChemostat_applyInterpolated
        PRIVATE mrmd CLI11::CLI11
        )

add_executable(04_GentleChemostat_applyInterpolated_OneRegion
        04_GentleChemostat_applyInterpolated_OneRegion.cpp
        )
target_link_libraries(04_GentleChemostat_applyInterpolated_OneRegion
        PRIVATE mrmd CLI11::CLI11
        )

add_executable(04_LennardJones_ThermodynamicForce
        04_LennardJones_ThermodynamicForce.cpp
        )
target_link_libraries(04_LennardJones_ThermodynamicForce
        PRIVATE mrmd CLI11::CLI11
        )

add_executable(04_CancelledLennardJones_ThermodynamicForce
        04_CancelledLennardJones_ThermodynamicForce.cpp
        )
target_link_libraries(04_CancelledLennardJones_ThermodynamicForce
        PRIVATE mrmd CLI11::CLI11
        )

add_executable(04_CancelledLennardJones_InterfaceIntegrator_ThermodynamicForce
        04_CancelledLennardJones_InterfaceIntegrator_ThermodynamicForce.cpp
        )
target_link_libraries(04_CancelledLennardJones_InterfaceIntegrator_ThermodynamicForce
        PRIVATE mrmd CLI11::CLI11
        )