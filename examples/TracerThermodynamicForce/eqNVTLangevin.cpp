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

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <Kokkos_Core.hpp>

#include "action/LangevinThermostat.hpp"
#include "action/LennardJones.hpp"
#include "action/VelocityVerlet.hpp"
#include "analysis/KineticEnergy.hpp"
#include "analysis/Pressure.hpp"
#include "analysis/SystemMomentum.hpp"
#include "communication/GhostLayer.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "initialization.hpp"
#include "io/DumpGRO.hpp"
#include "io/DumpH5MDParallel.hpp"
#include "io/RestoreH5MDParallel.hpp"
#include "util/EnvironmentVariables.hpp"
#include "util/ExponentialMovingAverage.hpp"
#include "util/PrintTable.hpp"

using namespace mrmd;

struct Config
{
    // output parameters
    bool bOutput = true;
    std::string fileOutH5MD = "equilibrateLangevin.h5md";
    std::string fileOutGro = "equilibrateLangevin.gro";
    idx_t outputInterval = 1000;

    // time parameters
    idx_t nsteps = 100001;
    real_t dt = 0.002;

    // input file parameters
    std::string fileRestoreH5MD = "equilibrateBerendsen.h5md";

    // system parameters
    const std::string resName = "Argon";
    const std::vector<std::string> typeNames = {"Ar"};

    // interaction parameters
    real_t sigma = 1_r;
    real_t epsilon = 1_r;
    real_t rCut = 2.5_r;
    real_t rCap = 0.0_r;
    bool doShift = true;

    // pressure parameters
    real_t pressure_averaging_coefficient = 0.02;

    // thermostatting parameters
    real_t target_temperature = 1.5_r;
    real_t temperature_relaxation_coefficient = 1.0_r;
    real_t temperature_averaging_coefficient = 0.2_r;
    idx_t thermostat_interval = 1;

    // neighbor-list parameters
    real_t cell_ratio = 1.0_r;
    real_t skin = 0.3;
    real_t neighborCutoff = rCut + skin;
    idx_t estimatedMaxNeighbors = 60;
};

void equilibrateLangevin(Config& config)
{
    // initialize
    data::Subdomain subdomain;
    auto atoms = data::Atoms(0);

    // load data from file
    auto mpiInfo = std::make_shared<data::MPIInfo>();
    auto io = io::RestoreH5MDParallel(mpiInfo);
    io.restore(config.fileRestoreH5MD, subdomain, atoms);

    const auto volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];
    auto rho = real_c(atoms.numLocalAtoms) / volume;
    std::cout << "rho: " << rho << std::endl;

    // output management
    auto dump = io::DumpH5MDParallel(mpiInfo, "J-Hizzle");

    // technical setup
    communication::GhostLayer ghostLayer;
    action::LennardJones LJ(config.rCut, config.sigma, config.epsilon, 0.5_r * config.sigma);
    HalfVerletList verletList;
    action::LangevinThermostat thermostat(
        config.temperature_relaxation_coefficient, config.target_temperature, config.dt);
    Kokkos::Timer timer;
    real_t maxAtomDisplacement = std::numeric_limits<real_t>::max();
    idx_t rebuildCounter = 0;
    util::ExponentialMovingAverage currentPressure(config.pressure_averaging_coefficient);
    util::ExponentialMovingAverage currentTemperature(config.temperature_averaging_coefficient);

    // thermodynamic observables table
    if (config.bOutput)
    {
        util::printTable(
            "step", "wall time", "T", "p", "V", "E_kin", "E_LJ", "E_total", "Nlocal", "Nghost");
        util::printTableSep(
            "step", "wall time", "T", "p", "V", "E_kin", "E_LJ", "E_total", "Nlocal", "Nghost");
    }

    // main integration loop
    for (auto step = 0; step < config.nsteps; ++step)
    {
        maxAtomDisplacement += action::VelocityVerlet::preForceIntegrate(atoms, config.dt);

        if (step % config.thermostat_interval == 0)
        {
            thermostat.apply(atoms);
        }

        if (maxAtomDisplacement >= config.skin * 0.5_r)
        {
            // reset displacement
            maxAtomDisplacement = 0_r;

            ghostLayer.exchangeRealAtoms(atoms, subdomain);

            real_t gridDelta[3] = {
                config.neighborCutoff, config.neighborCutoff, config.neighborCutoff};
            LinkedCellList linkedCellList(atoms.getPos(),
                                          0,
                                          atoms.numLocalAtoms,
                                          gridDelta,
                                          subdomain.minCorner.data(),
                                          subdomain.maxCorner.data());
            //            atoms.permute(linkedCellList);

            ghostLayer.createGhostAtoms(atoms, subdomain);
            verletList.build(atoms.getPos(),
                             0,
                             atoms.numLocalAtoms,
                             config.neighborCutoff,
                             config.cell_ratio,
                             subdomain.minGhostCorner.data(),
                             subdomain.maxGhostCorner.data(),
                             config.estimatedMaxNeighbors);
            ++rebuildCounter;
        }
        else
        {
            ghostLayer.updateGhostAtoms(atoms, subdomain);
        }

        auto force = atoms.getForce();
        Cabana::deep_copy(force, 0_r);

        LJ.apply(atoms, verletList);

        auto Ek = analysis::getKineticEnergy(atoms);
        currentPressure << 2_r * (Ek - LJ.getVirial()) / (3_r * volume);
        Ek /= real_c(atoms.numLocalAtoms);
        currentTemperature << (2_r / 3_r) * Ek;

        ghostLayer.contributeBackGhostToReal(atoms);
        action::VelocityVerlet::postForceIntegrate(atoms, config.dt);

        if (config.bOutput && (step % config.outputInterval == 0))
        {
            util::printTable(step,
                             timer.seconds(),
                             currentTemperature,
                             currentPressure,
                             volume,
                             Ek,
                             LJ.getEnergy() / real_c(atoms.numLocalAtoms),
                             Ek + LJ.getEnergy() / real_c(atoms.numLocalAtoms),
                             atoms.numLocalAtoms,
                             atoms.numGhostAtoms);
        }
    }
    auto time = timer.seconds();
    std::cout << time << std::endl;

    auto cores = util::getEnvironmentVariable("OMP_NUM_THREADS");

    std::ofstream fout("ecab.perf", std::ofstream::app);
    fout << cores << ", " << time << ", " << atoms.numLocalAtoms << ", " << config.nsteps
         << std::endl;
    fout.close();

    dump.dump(config.fileOutH5MD, subdomain, atoms);
    io::dumpGRO(config.fileOutGro,
                atoms,
                subdomain,
                0,
                config.resName,
                config.resName,
                config.typeNames,
                false,
                true);
}

int main(int argc, char* argv[])
{
    mrmd::initialize();

    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    Config config;
    CLI::App app{"Lennard Jones Fluid benchmark application"};
    app.add_option("-n,--nsteps", config.nsteps, "number of simulation steps");
    app.add_option("-o,--output", config.outputInterval, "output interval");
    CLI11_PARSE(app, argc, argv);
    if (config.outputInterval < 0) config.bOutput = false;
    equilibrateLangevin(config);

    mrmd::finalize();

    return EXIT_SUCCESS;
}