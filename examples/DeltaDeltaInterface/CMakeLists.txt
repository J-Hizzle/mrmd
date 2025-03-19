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

add_executable(EquilibrationBerendsenNVT
        EquilibrationBerendsenNVT.cpp
        )
target_link_libraries(EquilibrationBerendsenNVT
        PRIVATE mrmd CLI11::CLI11
        )
add_executable(EquilibrationLangevinNVT
        EquilibrationLangevinNVT.cpp
        )
target_link_libraries(EquilibrationLangevinNVT
        PRIVATE mrmd CLI11::CLI11
        )
add_executable(TFIterationsDeltaDelta
        TFIterationsDeltaDelta.cpp
        )
target_link_libraries(TFIterationsDeltaDelta
        PRIVATE mrmd CLI11::CLI11 fmt::fmt
        )
add_executable(DeltaDeltaProductionNVT
        DeltaDeltaProductionNVT.cpp
        )
target_link_libraries(DeltaDeltaProductionNVT
        PRIVATE mrmd CLI11::CLI11 fmt::fmt
        )