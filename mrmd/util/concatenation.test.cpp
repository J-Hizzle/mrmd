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

#include "util/concatenation.hpp"

#include <gtest/gtest.h>

#include "data/Atoms.hpp"

namespace mrmd::util
{
TEST(concatenate, simple)
{
    data::HostAtoms h_atoms(2);
    h_atoms.numLocalAtoms = 1;
    h_atoms.numGhostAtoms = 1;

    h_atoms.getPos()(0, 0) = 1_r;
    h_atoms.getPos()(0, 1) = 2_r;
    h_atoms.getPos()(0, 2) = 3_r;
    h_atoms.getVel()(0, 0) = 4_r;
    h_atoms.getVel()(0, 1) = 5_r;
    h_atoms.getVel()(0, 2) = 6_r;
    h_atoms.getForce()(0, 0) = 7_r;
    h_atoms.getForce()(0, 1) = 8_r;
    h_atoms.getForce()(0, 2) = 9_r;
    h_atoms.getMass()(0) = 10_r;
    h_atoms.getType()(0) = 11;
    h_atoms.getCharge()(0) = 12_r;
    h_atoms.getRelativeMass()(0) = 13_r;

    data::Atoms atoms(h_atoms);
    data::Atoms atomsRef(h_atoms);

    data::HostAtoms h_other(3);
    h_other.numLocalAtoms = 1;
    h_other.numGhostAtoms = 2;

    h_other.getPos()(0, 0) = 27_r;
    h_other.getPos()(0, 1) = 28_r;
    h_other.getPos()(0, 2) = 29_r;
    h_other.getVel()(0, 0) = 30_r;
    h_other.getVel()(0, 1) = 31_r;
    h_other.getVel()(0, 2) = 32_r;
    h_other.getForce()(0, 0) = 33_r;
    h_other.getForce()(0, 1) = 34_r;
    h_other.getForce()(0, 2) = 35_r;
    h_other.getMass()(0) = 36_r;
    h_other.getType()(0) = 37;
    h_other.getCharge()(0) = 38_r;
    h_other.getRelativeMass()(0) = 39_r;

    data::Atoms other(h_other);

    concatenateRealAtoms(atoms, other);
    EXPECT_EQ(atoms.numLocalAtoms, 2);

    EXPECT_FLOAT_EQ(atoms.getPos()(0, 0), atomsRef.getPos()(0, 0));
    EXPECT_FLOAT_EQ(atoms.getPos()(0, 1), atomsRef.getPos()(0, 1));
    EXPECT_FLOAT_EQ(atoms.getPos()(0, 2), atomsRef.getPos()(0, 2));
    EXPECT_FLOAT_EQ(atoms.getPos()(1, 0), other.getPos()(0, 0));
    EXPECT_FLOAT_EQ(atoms.getPos()(1, 1), other.getPos()(0, 1));
    EXPECT_FLOAT_EQ(atoms.getPos()(1, 2), other.getPos()(0, 2));

    EXPECT_FLOAT_EQ(atoms.getVel()(0, 0), atomsRef.getVel()(0, 0));
    EXPECT_FLOAT_EQ(atoms.getVel()(0, 1), atomsRef.getVel()(0, 1));
    EXPECT_FLOAT_EQ(atoms.getVel()(0, 2), atomsRef.getVel()(0, 2));
    EXPECT_FLOAT_EQ(atoms.getVel()(1, 0), other.getVel()(0, 0));
    EXPECT_FLOAT_EQ(atoms.getVel()(1, 1), other.getVel()(0, 1));
    EXPECT_FLOAT_EQ(atoms.getVel()(1, 2), other.getVel()(0, 2));

    EXPECT_FLOAT_EQ(atoms.getForce()(0, 0), atomsRef.getForce()(0, 0));
    EXPECT_FLOAT_EQ(atoms.getForce()(0, 1), atomsRef.getForce()(0, 1));
    EXPECT_FLOAT_EQ(atoms.getForce()(0, 2), atomsRef.getForce()(0, 2));
    EXPECT_FLOAT_EQ(atoms.getForce()(1, 0), other.getForce()(0, 0));
    EXPECT_FLOAT_EQ(atoms.getForce()(1, 1), other.getForce()(0, 1));
    EXPECT_FLOAT_EQ(atoms.getForce()(1, 2), other.getForce()(0, 2));

    EXPECT_FLOAT_EQ(atoms.getMass()(0), atomsRef.getMass()(0));
    EXPECT_FLOAT_EQ(atoms.getMass()(1), other.getMass()(0));

    EXPECT_EQ(atoms.getType()(0), atomsRef.getType()(0));
    EXPECT_EQ(atoms.getType()(1), other.getType()(0));

    EXPECT_EQ(atoms.getCharge()(0), atomsRef.getCharge()(0));
    EXPECT_EQ(atoms.getCharge()(1), other.getCharge()(0));

    EXPECT_FLOAT_EQ(atoms.getRelativeMass()(0), atomsRef.getRelativeMass()(0));
    EXPECT_FLOAT_EQ(atoms.getRelativeMass()(1), other.getRelativeMass()(0));
}
}  // namespace mrmd::util