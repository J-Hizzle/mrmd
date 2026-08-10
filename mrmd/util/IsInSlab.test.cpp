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

#include "IsInSlab.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace util
{
TEST(IsInSlab, testDefaultAxis)
{
    const auto slabMin = 2_r;
    const auto slabMax = 4_r;
    auto isInSlab = IsInSlab(slabMin, slabMax);

    EXPECT_TRUE(isInSlab(3.99_r, 10_r, 10_r));
    EXPECT_TRUE(isInSlab(2.01_r, 10_r, 10_r));
    EXPECT_TRUE(isInSlab(3.1_r, 10_r, 10_r));

    EXPECT_FALSE(isInSlab(5.7_r, 10_r, 10_r));
    EXPECT_FALSE(isInSlab(6_r, 10_r, 10_r));
    EXPECT_FALSE(isInSlab(-2_r, 10_r, 10_r));
    EXPECT_FALSE(isInSlab(-1.2_r, 10_r, 10_r));
    EXPECT_FALSE(isInSlab(0_r, 10_r, 10_r));
    EXPECT_FALSE(isInSlab(-2.1_r, 10_r, 10_r));
}

TEST(IsInSlab, testCustomAxisAndTolerance)
{
    const auto slabMin = 2_r;
    const auto slabMax = 4_r;
    const auto tolerance = 0.1_r;
    const auto axis = AXIS::Y;
    auto isInSlab = IsInSlab(slabMin, slabMax, axis, tolerance);

    EXPECT_TRUE(isInSlab(10_r, 2_r, 10_r));
    EXPECT_TRUE(isInSlab(10_r, 4_r, 10_r));
    EXPECT_TRUE(isInSlab(10_r, 2.7_r, 10_r));
    EXPECT_TRUE(isInSlab(10_r, 3_r, 10_r));
    EXPECT_TRUE(isInSlab(10_r, 2.1_r, 10_r));

    EXPECT_FALSE(isInSlab(10_r, -1.2_r, 10_r));
    EXPECT_FALSE(isInSlab(10_r, -2_r, 10_r));
    EXPECT_FALSE(isInSlab(10_r, 4.2_r, 10_r));
    EXPECT_FALSE(isInSlab(3.1_r, 10_r, 10_r));
    EXPECT_FALSE(isInSlab(-2.1_r, 10_r, 10_r));
}

TEST(IsInSlab, testInterval)
{
    const auto intervalMin = 2_r;
    const auto intervalMax = 4_r;
    auto isInInterval = IsInSlab(intervalMin, intervalMax);

    EXPECT_TRUE(isInInterval(3.99_r));
    EXPECT_TRUE(isInInterval(2.01_r));
    EXPECT_TRUE(isInInterval(3.1_r));

    EXPECT_FALSE(isInInterval(5.7_r));
    EXPECT_FALSE(isInInterval(6_r));
    EXPECT_FALSE(isInInterval(-2_r));
    EXPECT_FALSE(isInInterval(-1.2_r));
    EXPECT_FALSE(isInInterval(0_r));
    EXPECT_FALSE(isInInterval(-2.1_r));
}

TEST(IsInSlab, testGetVolume)
{
    const auto slabMin = 2_r;
    const auto slabMax = 4_r;
    auto isInSlab = IsInSlab(slabMin, slabMax);

    data::Subdomain subdomain({0_r, 0_r, 0_r}, {10_r, 10_r, 10_r}, {1_r, 1_r, 1_r});

    EXPECT_FLOAT_EQ(isInSlab.getVolume(subdomain), 200.0);
}
}  // namespace util
}  // namespace mrmd