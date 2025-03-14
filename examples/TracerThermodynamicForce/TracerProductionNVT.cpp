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

#include "datatypes.hpp"
#include "initialization.hpp"
#include "io/DumpThermoForce.hpp"
#include "io/RestoreThermoForce.hpp"
#include "io/RestoreH5MDParallel.hpp"
#include "data/Subdomain.hpp"
#include "data/Atoms.hpp"
#include "data/Molecules.hpp"
#include "data/MoleculesFromAtoms.hpp"

using namespace mrmd;

int main(int argc, char* argv[])  // NOLINT
{
    mrmd::initialize();
    {
    data::Subdomain subdomain;
    auto atoms = data::Atoms(0);

    // load data from file
    auto mpiInfo = std::make_shared<data::MPIInfo>();
    auto inpH5MD = io::RestoreH5MDParallel(mpiInfo);
    inpH5MD.restore("equilibrateLangevin.h5md", subdomain, atoms);
    auto molecules = data::createMoleculeForEachAtom(atoms);

    const auto volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];
    auto rho = real_c(atoms.numLocalAtoms) / volume;
    std::cout << "rho: " << rho << std::endl;
    auto thermodynamicForce = io::restoreThermoForce("test_final_tf.txt", subdomain);
    
    io::dumpThermoForce("test_final_tf_out.txt", thermodynamicForce, 0);
    }
    mrmd::finalize();
    return EXIT_SUCCESS;
}