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
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "action/ContributeMoleculeForceToAtoms.hpp"
#include "action/LJ_IdealGas_FAdResS.hpp"
#include "action/LennardJones.hpp"
#include "action/ThermodynamicForce.hpp"
#include "action/UpdateMolecules.hpp"
#include "action/VelocityVerletLangevinThermostat.hpp"
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
#include "io/DumpProfile.hpp"
#include "io/DumpThermoForce.hpp"
#include "io/RestoreH5MDParallel.hpp"
#include "util/ApplicationRegion.hpp"
#include "util/EnvironmentVariables.hpp"
#include "util/PrintTable.hpp"
#include "util/Random.hpp"
#include "weighting_function/Slab.hpp"
#include "io/RestoreThermoForce.hpp"

using namespace mrmd;

struct Config
{
    // time parameters
    idx_t nsteps = 40000001;
    real_t dt = 0.002;

    // input file parameters
    std::string fileRestoreH5MD = "equilibrateLangevin.h5md";
    std::string fileRestoreTF;

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
    idx_t densitySamplingInterval = 200;
    idx_t densityUpdateInterval = 1000000;
    real_t densityBinWidth = -1_r;
    real_t smoothingDamping = 1_r;
    real_t smoothingInverseDamping = 1_r / smoothingDamping;
    idx_t smoothingNeighbors = 10;
    real_t smoothingRange = real_c(smoothingNeighbors) * densityBinWidth * smoothingDamping;
    real_t thermodynamicForceModulation = 2_r;
    real_t applicationRegionMin = 0.5_r * atomisticRegionDiameter;
    real_t applicationRegionMax = 0.5_r * atomisticRegionDiameter + 2_r * hybridRegionDiameter;
    bool enforceSymmetry = true;

    // output parameters
    bool bOutput = true;
    idx_t outputInterval = densityUpdateInterval;
    std::string fileOut = "thermoForce";
    std::string fileOutH5MD = fmt::format("{0}.h5md", fileOut);
    std::string fileOutGro = fmt::format("{0}.gro", fileOut);
    std::string fileOutTF = fmt::format("{0}_tf.txt", fileOut);
    std::string fileOutDens = fmt::format("{0}_dens.txt", fileOut);
    std::string fileOutFinalH5MD = fmt::format("{0}_final.h5md", fileOut);
    std::string fileOutFinalTF = fmt::format("{0}_final_tf.txt", fileOut);
};

void LJ(Config& config)
{
    // initialize
    data::Subdomain initialSubdomain;
    auto atoms = data::Atoms(0);

    // load data from file
    auto mpiInfo = std::make_shared<data::MPIInfo>();
    auto io = io::RestoreH5MDParallel(mpiInfo);
    io.restore(config.fileRestoreH5MD, initialSubdomain, atoms);

    auto molecules = data::createMoleculeForEachAtom(atoms);

    // reinitialize subdomain with no ghost layer in x-direction
    auto subdomain = data::Subdomain({
        initialSubdomain.minCorner[0],
        initialSubdomain.minCorner[1],
        initialSubdomain.minCorner[2],
    },
    {
        initialSubdomain.maxCorner[0],
        initialSubdomain.maxCorner[1],
        initialSubdomain.maxCorner[2],
    },
    {
        0_r, 
        initialSubdomain.ghostLayerThickness[1], 
        initialSubdomain.ghostLayerThickness[1]
    });

    const auto volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];
    auto rho = real_c(atoms.numLocalAtoms) / volume;
    std::cout << "rho: " << rho << std::endl;
    std::cout << "restoring thermodynamic force from file" << std::endl;

    auto thermodynamicForce = io::restoreThermoForce(config.fileRestoreTF, subdomain, {rho}, 
                                                {config.thermodynamicForceModulation},
                                                config.enforceSymmetry,
                                                false,
                                                1,
                                                config.densityBinWidth);

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

    std::tuple<real_t, idx_t, idx_t> integratorResult;
    real_t fluxBoundaryLeft = boxCenterX - 0.5_r * config.atomisticRegionDiameter;
    real_t fluxBoundaryRight = boxCenterX + 0.5_r * config.atomisticRegionDiameter;
    idx_t fluxLeft = 0;
    idx_t fluxRight = 0;

    // actions
    action::LJ_IdealGas LJ(config.rCap, config.rCut, config.sigma, config.epsilon, config.doShift);
    action::VelocityVerletLangevinThermostat integrator(
        config.temperature_relaxation_coefficient, config.target_temperature);
    communication::MultiResGhostLayer ghostLayer;

    // output management
    io::DumpProfile dumpDens;
    io::DumpProfile dumpThermoForce;
    real_t densityBinVolume =
        subdomain.diameter[1] * subdomain.diameter[2] * config.densityBinWidth;
    auto dumpH5MD = io::DumpH5MDParallel(mpiInfo, "J-Hizzle");
    if (config.bOutput)
    {
        util::printTable(
            "step", "time", "T", "Ek", "E0", "E", "mu_left", "mu_right", "flux left", "flux right", "Nlocal", "Nghost");
        util::printTableSep(
            "step", "time", "T", "Ek", "E0", "E", "mu_left", "mu_right", "flux left", "flux right", "Nlocal", "Nghost");
        // density profile
        dumpDens.open(config.fileOutDens);
        dumpDens.dumpScalarView(thermodynamicForce.getDensityProfile().createGrid());
        // thermodynamic force
        dumpThermoForce.open(config.fileOutTF);
        dumpThermoForce.dumpScalarView(thermodynamicForce.getForce().createGrid());
        // microstate
        dumpH5MD.open(config.fileOutH5MD, subdomain, atoms);
    }

    for (auto step = 0; step < config.nsteps; ++step)
    {
        assert(atoms.numLocalAtoms == molecules.numLocalMolecules);
        assert(atoms.numGhostAtoms == molecules.numGhostMolecules);
        integratorResult = integrator.preForceIntegrate(atoms, config.dt, fluxBoundaryLeft, fluxBoundaryRight);
        maxAtomDisplacement += std::get<0>(integratorResult);
        fluxLeft += std::get<1>(integratorResult);
        fluxRight += std::get<2>(integratorResult);

        if (maxAtomDisplacement >= config.skin * 0.5_r)
        {
            // update molecule positions
            action::UpdateMolecules::update(molecules, atoms, weightingFunction);

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

        if (step % config.densitySamplingInterval == 0)
        {
            thermodynamicForce.sample(atoms);
        }

        if (config.bOutput && (step % config.outputInterval == 0))
        {
            // density profile output
            auto numberOfDensityProfileSamples =
                thermodynamicForce.getNumberOfDensityProfileSamples();

            real_t normalizationFactor = 1_r / densityBinVolume;
            if (numberOfDensityProfileSamples > 0)
            {
                normalizationFactor =
                    1_r / (densityBinVolume * real_c(numberOfDensityProfileSamples));
            }
            auto densityProfile = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), thermodynamicForce.getDensityProfile(0));
            dumpDens.dumpScalarView(densityProfile, normalizationFactor);
        }

        if (step % config.densityUpdateInterval == 0 && step > 0)
        {
            thermodynamicForce.update(config.smoothingInverseDamping, config.smoothingRange);
        }

        thermodynamicForce.apply(atoms, applicationRegion);
        auto E0 = LJ.run(molecules, moleculesVerletList, atoms);

        ghostLayer.contributeBackGhostToReal(atoms);

        integrator.postForceIntegrate(atoms, config.dt);

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
                             fluxLeft,
                             fluxRight,
                             atoms.numLocalAtoms,
                             atoms.numGhostAtoms);
            // reset flux counters 
            fluxLeft = 0;
            fluxRight = 0;

            // thermodynamic force output
            auto thermoForce = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                                   thermodynamicForce.getForce(0));
            dumpThermoForce.dumpScalarView(thermoForce);

            // microstate output
            dumpH5MD.dumpStep(subdomain, atoms, step, config.dt);
        }
    }
    if (config.bOutput)
    {
        dumpDens.close();
        dumpThermoForce.close();
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

        // final thermodynamic force output
        io::dumpThermoForce(config.fileOutFinalTF, thermodynamicForce, 0);
    }

    auto cores = util::getEnvironmentVariable("OMP_NUM_THREADS");

    auto time = timer.seconds();
    std::cout << time << std::endl;

    std::ofstream fout("ecab.perf", std::ofstream::app);
    fout << cores << ", " << time << ", " << atoms.numLocalAtoms << ", " << config.nsteps
         << std::endl;
    fout.close();
}

int main(int argc, char* argv[])  // NOLINT
{
    mrmd::initialize();

    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    Config config;
    CLI::App app{"AdResS tracer thermodynamic force simulation"};
    app.add_option("-n,--nsteps", config.nsteps, "number of simulation steps");
    app.add_option("-d,--tstep", config.dt, "time step");
    app.add_option("-o,--outint", config.outputInterval, "output interval");
    app.add_option("-i,--inpfile", config.fileRestoreH5MD, "input file name");
    app.add_option("-f,--outfile", config.fileOut, "output file name");

    app.add_option("--forceguess", config.fileRestoreTF, "initial guess for the thermodynamics force");
    app.add_option("--sampling", config.densitySamplingInterval, "density sampling interval");
    app.add_option("--update", config.densityUpdateInterval, "density update interval");
    app.add_option("--densbinwidth", config.densityBinWidth, "density bin width");
    app.add_option("--damping", config.smoothingDamping, "density smoothing damping factor");
    app.add_option("--neighbors", config.smoothingNeighbors, "density smoothing neighbors");
    app.add_option(
        "--forcemod", config.thermodynamicForceModulation, "thermodynamic force modulation");

    app.add_option("--appmin", config.applicationRegionMin, "application region minimum");
    app.add_option("--appmax", config.applicationRegionMax, "application region maximum");
    app.add_option("--atdiameter", config.atomisticRegionDiameter, "atomistic region diameter");
    app.add_option("--hydiameter", config.hybridRegionDiameter, "hybrid region diameter");

    CLI11_PARSE(app, argc, argv);

    config.fileOutH5MD = fmt::format("{0}.h5md", config.fileOut);
    config.fileOutGro = fmt::format("{0}.gro", config.fileOut);
    config.fileOutTF = fmt::format("{0}_tf.txt", config.fileOut);
    config.fileOutDens = fmt::format("{0}_dens.txt", config.fileOut);
    config.fileOutFinalH5MD = fmt::format("{0}_final.h5md", config.fileOut);
    config.fileOutFinalTF = fmt::format("{0}_final_tf.txt", config.fileOut);

    config.smoothingRange =
        real_c(config.smoothingNeighbors) * config.densityBinWidth * config.smoothingDamping;

    if (config.outputInterval < 0) config.bOutput = false;
    LJ(config);

    mrmd::finalize();
    return EXIT_SUCCESS;
}