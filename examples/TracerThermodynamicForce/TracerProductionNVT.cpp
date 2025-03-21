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

#include <fmt/format.h>

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>

#include "action/LJ_IdealGas_FAdResS.hpp"
#include "action/LangevinThermostat.hpp"
#include "action/UpdateMolecules.hpp"
#include "action/VelocityVerlet.hpp"
#include "analysis/KineticEnergy.hpp"
#include "analysis/SystemMomentum.hpp"
#include "communication/MultiResGhostLayer.hpp"
#include "data/Atoms.hpp"
#include "data/Molecules.hpp"
#include "data/MoleculesFromAtoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "initialization.hpp"
#include "io/DumpGRO.hpp"
#include "io/DumpH5MDParallel.hpp"
#include "io/RestoreH5MDParallel.hpp"
#include "io/RestoreThermoForce.hpp"
#include "util/EnvironmentVariables.hpp"
#include "util/PrintTable.hpp"

using namespace mrmd;

struct Config
{
    // time parameters
    idx_t nsteps = 40000001;
    real_t dt = 0.002;

    // input file parameters
    std::string fileRestore = "thermoForce";
    std::string fileRestoreH5MD = fmt::format("{0}_final.h5md", fileRestore);
    std::string fileRestoreTF = fmt::format("{0}_final_tf.txt", fileRestore);

    // system parameters
    const std::string resName = "Argon";
    const std::vector<std::string> typeNames = {"Ar"};

    // interaction parameters
    real_t sigma = 1_r;
    real_t epsilon = 1_r;
    real_t rCut = 2.5_r;
    real_t rCap = 0.82417464_r;
    bool doShift = true;

    // neighborlist parameters
    real_t skin = 0.3_r;
    real_t neighborCutoff = rCut + skin;
    real_t cell_ratio = 0.5_r;
    idx_t estimatedMaxNeighbors = 60;

    // pressure parameters
    real_t pressure_averaging_coefficient = 0.02;

    // thermostatting parameters
    real_t target_temperature = 1.5_r;
    real_t temperature_relaxation_coefficient = 20.0_r;
    real_t temperature_averaging_coefficient = 0.2_r;
    idx_t thermostat_interval = 1;

    // AdResS parameters
    weighting_function::Slab::InterfaceType interfaceType =
        weighting_function::Slab::InterfaceType::ABRUPT;
    real_t atomisticRegionDiameter = 6_r;
    real_t hybridRegionDiameter = 2.5_r;

    // thermodynamic force parameters
    real_t applicationRegionMin = 0.5_r * atomisticRegionDiameter;
    real_t applicationRegionMax = 0.5_r * atomisticRegionDiameter + 2_r * hybridRegionDiameter;

    // output parameters
    bool bOutput = true;
    idx_t outputInterval = 1000;
    std::string fileOut = "tracerProductionNVT";
    std::string fileOutH5md = fmt::format("{0}.h5md", fileOut);
    std::string fileOutGro = fmt::format("{0}.gro", fileOut);
    std::string fileOutTF = fmt::format("{0}_tf.txt", fileOut);
    std::string fileOutFinalH5MD = fmt::format("{0}_final.h5md", fileOut);
};

void LJ(Config& config)
{
    // initialize
    data::Subdomain subdomain;
    auto atoms = data::Atoms(0);
    auto mpiInfo = std::make_shared<data::MPIInfo>();

    // load data from file
    auto restoreH5MD = io::RestoreH5MDParallel(mpiInfo);
    restoreH5MD.restore(config.fileRestoreH5MD, subdomain, atoms);
    auto thermodynamicForce = io::restoreThermoForce(config.fileRestoreTF, subdomain);

    auto molecules = data::createMoleculeForEachAtom(atoms);

    const auto volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];
    auto rho = real_c(atoms.numLocalAtoms) / volume;
    std::cout << "rho: " << rho << std::endl;

    // data allocations
    HalfVerletList moleculesVerletList;
    idx_t verletlistRebuildCounter = 0;

    Kokkos::Timer timer;
    real_t maxAtomDisplacement = std::numeric_limits<real_t>::max();

    real_t boxCenterX = 0.5_r * (subdomain.maxCorner[0] + subdomain.minCorner[0]);
    real_t boxCenterY = 0.5_r * (subdomain.maxCorner[1] + subdomain.minCorner[1]);
    real_t boxCenterZ = 0.5_r * (subdomain.maxCorner[2] + subdomain.minCorner[2]);

    std::cout << "x center: " << boxCenterX << std::endl;
    std::cout << "y center: " << boxCenterY << std::endl;
    std::cout << "z center: " << boxCenterZ << std::endl;

    auto weightingFunction =
        weighting_function::Slab({boxCenterX, boxCenterY, boxCenterZ},
                                 config.atomisticRegionDiameter,
                                 config.hybridRegionDiameter,
                                 0,  // here would be the exponent, but not necessary for abrupt
                                     // interface - maybe redesign in the future?
                                 config.interfaceType);
    auto applicationRegion = util::ApplicationRegion({boxCenterX, boxCenterY, boxCenterZ},
                                                     config.applicationRegionMin,
                                                     config.applicationRegionMax);

    // actions
    action::LJ_IdealGas LJ(config.rCap, config.rCut, config.sigma, config.epsilon, config.doShift);
    action::LangevinThermostat langevinThermostat(
        config.temperature_relaxation_coefficient, config.target_temperature, config.dt);
    communication::MultiResGhostLayer ghostLayer;

    // output management
    auto dumpH5MD = io::DumpH5MDParallel(mpiInfo, "J-Hizzle");
    if (config.bOutput)
    {
        util::printTable(
            "step", "time", "T", "Ek", "E0", "E", "mu_left", "mu_right", "Nlocal", "Nghost");
        util::printTableSep(
            "step", "time", "T", "Ek", "E0", "E", "mu_left", "mu_right", "Nlocal", "Nghost");
        // density profile
        auto densityGrid = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), thermodynamicForce.getDensityProfile().createGrid());
        dumpH5MD.open(config.fileOutH5md, atoms);
    }
    for (auto step = 0; step < config.nsteps; ++step)
    {
        assert(atoms.numLocalAtoms == molecules.numLocalMolecules);
        assert(atoms.numGhostAtoms == molecules.numGhostMolecules);
        maxAtomDisplacement += action::VelocityVerlet::preForceIntegrate(atoms, config.dt);

        // update molecule positions
        action::UpdateMolecules::update(molecules, atoms, weightingFunction);

        if (maxAtomDisplacement >= config.skin * 0.5_r)
        {
            // reset displacement
            maxAtomDisplacement = 0_r;

            ghostLayer.exchangeRealAtoms(molecules, atoms, subdomain);

            ghostLayer.createGhostAtoms(molecules, atoms, subdomain);
            moleculesVerletList.build(molecules.getPos(),
                                      0,
                                      molecules.numLocalMolecules,
                                      config.neighborCutoff,
                                      config.cell_ratio,
                                      subdomain.minGhostCorner.data(),
                                      subdomain.maxGhostCorner.data(),
                                      config.estimatedMaxNeighbors);
            ++verletlistRebuildCounter;
        }
        else
        {
            ghostLayer.updateGhostAtoms(atoms, subdomain);
        }

        action::UpdateMolecules::update(molecules, atoms, weightingFunction);

        auto atomsForce = atoms.getForce();
        Cabana::deep_copy(atomsForce, 0_r);
        auto moleculesForce = molecules.getForce();
        Cabana::deep_copy(moleculesForce, 0_r);

        thermodynamicForce.apply(atoms, applicationRegion);
        auto E0 = LJ.run(molecules, moleculesVerletList, atoms);

        if (config.target_temperature >= 0)
        {
            langevinThermostat.apply(atoms);
        }
        ghostLayer.contributeBackGhostToReal(atoms);

        action::VelocityVerlet::postForceIntegrate(atoms, config.dt);

        if (config.bOutput && (step % config.outputInterval == 0))
        {
            auto Ek = analysis::getMeanKineticEnergy(atoms);
            auto systemMomentum = analysis::getSystemMomentum(atoms);
            auto T = (2_r / 3_r) * Ek;
            E0 /= real_c(atoms.numLocalAtoms);

            // calc chemical potential
            auto muLeft = thermodynamicForce.getMuLeft()[0];
            auto muRight = thermodynamicForce.getMuRight()[0];

            util::printTable(step,
                             timer.seconds(),
                             T,
                             Ek,
                             E0,
                             E0 + Ek,
                             muLeft,
                             muRight,
                             atoms.numLocalAtoms,
                             atoms.numGhostAtoms);

            // microstate output
            dumpH5MD.dumpStep(subdomain, atoms, step, config.dt);
        }
        if (config.bOutput)
        {
            dumpH5MD.close();

            // final microstates output
            dumpH5MD.dump(config.fileOutFinalH5MD, subdomain, atoms);

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
        auto cores = util::getEnvironmentVariable("OMP_NUM_THREADS");

        auto time = timer.seconds();
        std::cout << time << std::endl;

        std::ofstream fout("ecab.perf", std::ofstream::app);
        fout << cores << ", " << time << ", " << atoms.numLocalAtoms << ", " << config.nsteps
             << std::endl;
        fout.close();
    }
}

int main(int argc, char* argv[])  // NOLINT
{
    mrmd::initialize();

    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    Config config;

    CLI::App app{"AdResS tracer thermodynamic force simulation"};
    app.add_option("-n,--nsteps", config.nsteps, "number of simulation steps");
    app.add_option("-o,--outint", config.outputInterval, "output interval");
    app.add_option("-i,--inpfile", config.fileRestore, "input file name");
    app.add_option("-f,--outfile", config.fileOut, "output file name");
    app.add_option("--appmin", config.applicationRegionMin, "application region minimum");
    app.add_option("--appmax", config.applicationRegionMax, "application region maximum");

    CLI11_PARSE(app, argc, argv);

    config.fileRestoreH5MD = fmt::format("{0}.h5md", config.fileRestore);
    config.fileRestoreTF = fmt::format("{0}_tf.txt", config.fileRestore);
    config.fileOutH5md = fmt::format("{0}.h5md", config.fileOut);
    config.fileOutGro = fmt::format("{0}.gro", config.fileOut);
    config.fileOutTF = fmt::format("{0}_tf.txt", config.fileOut);
    config.fileOutFinalH5MD = fmt::format("{0}_final.h5md", config.fileOut);

    LJ(config);

    mrmd::finalize();
    return EXIT_SUCCESS;
}