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

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "Cabana_NeighborList.hpp"
#include "action/LennardJones.hpp"
#include "action/LimitAcceleration.hpp"
#include "action/LimitVelocity.hpp"
#include "action/ThermodynamicForce.hpp"
#include "action/VelocityVerletLangevinThermostat.hpp"
#include "analysis/KineticEnergy.hpp"
#include "analysis/MeanSquareDisplacement.hpp"
#include "analysis/Pressure.hpp"
#include "analysis/SystemMomentum.hpp"
#include "analysis/Density.hpp"
#include "communication/GhostLayer.hpp"
#include "communication/OpenBoundaryLayer.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "initialization.hpp"
#include "io/DumpGRO.hpp"
#include "io/DumpProfile.hpp"
#include "io/DumpThermoForce.hpp"
#include "io/RestoreGRO.hpp"
#include "util/EnvironmentVariables.hpp"
#include "util/IsInSymmetricSlab.hpp"
#include "util/PrintTable.hpp"
#include "util/IsInSymmetricSlab.hpp"
#include "util/IsInSlab.hpp"

using namespace mrmd;

/**
 * Configuration for the flux boundary example simulation.
 */
struct Config
{
    // input file parameters
    std::string fileRestoreGRO = "equilibrateLangevin_rho370_T15_x30_yz30_2026_07_12.gro";

    // simulation time parameters
    idx_t nsteps = 400001;               ///< number of steps to simulate
    real_t dt = 0.002_r;  ///< time step size in reduced units

    // interaction parameters
    static constexpr real_t sigma =
        1_r;  ///< distance at which LJ potential is zero in reduced units
    static constexpr real_t epsilon = 1_r;  ///< energy well depth of LJ potential in reduced units
    static constexpr real_t mass = 1_r;     ///< mass of one atom in reduced units
    static constexpr real_t maxVelocity =
        1_r;  ///< maximum initial velocity component in reduced units
    static constexpr real_t r_cut = 2.5_r * sigma;  ///< cutoff radius for LJ potential
    real_t r_cap = 0.82417464_r * sigma;  ///< capping radius for LJ potential

    // neighbor list parameters
    static constexpr real_t skin = 0.1_r * sigma;           ///< skin thickness for neighbor list
    static constexpr real_t neighborCutoff = r_cut + skin;  ///< cutoff radius for neighbor list
    static constexpr real_t cell_ratio =
        1_r;  ///< ratio of cell size on Cartesian grid to cutoff radius for neighbor list
    static constexpr idx_t estimatedMaxNeighbors =
        60;  ///< estimated maximum number of neighbors per atom

    // thermostat parameters
    communication::OpenBoundaryLayer::BoundaryValues reservoirTemperature{{
        {{1.5_r, 1.6_r}},
        {{-1_r, -1_r}},
        {{-1_r, -1_r}}
    }};
    real_t gamma = 100_r;  ///< friction coefficient for Langevin thermostat

    // chemostat parameters
    communication::OpenBoundaryLayer::BoundaryValues reservoirDensity{{
        {{0.370_r, 0.360_r}}, // x: left, right
        {{-1_r,   -1_r   }},  // y
        {{-1_r,   -1_r   }}   // z
    }};
    real_t reservoirDensityFeedback =
        10_r * dt;  ///< feedback gain for reservoir density control (0 disables control)

    // thermodynamic force parameters
    idx_t densitySamplingInterval = 200;
    idx_t densityUpdateInterval = 10000;
    real_t densityBinWidth = 0.2_r * sigma;
    real_t smoothingDamping = 1_r;
    real_t smoothingInverseDamping = 1_r / smoothingDamping;
    idx_t smoothingNeighbors = 0;
    real_t smoothingRange = real_c(smoothingNeighbors) * densityBinWidth * smoothingDamping;
    real_t thermodynamicForceModulation = 1_r;

    // application regions
    real_t thermostatRegionMinLeft = 0_r * sigma;
    real_t thermostatRegionMaxLeft = 1.5_r * sigma;
    real_t thermostatRegionMinRight = 28.5_r * sigma;
    real_t thermostatRegionMaxRight = 30_r * sigma;
    real_t thermoForceRegionMin = 12.5_r * sigma;
    real_t thermoForceRegionMax = 15_r * sigma;

    // output parameters
    bool bOutput = true;                  ///< whether to output data files
    idx_t outputInterval = -1;            ///< interval for data file output (-1: no output)
    const std::string resName = "Argon";  ///< residue name for output files
    const std::vector<std::string> typeNames = {"Ar"};  ///< atom type names for output files

    std::string fileOut = "openBoundary";  ///< base name for output files
    std::string fileOutFinalGro = format("{0}_final.gro", fileOut);
    std::string fileOutTF = format("{0}_tf.txt", fileOut);
    std::string fileOutDens = format("{0}_dens.txt", fileOut);
    std::string fileOutFinalTF = format("{0}_final_tf.txt", fileOut);
};

void runLennardJones_idealGas_localCap(Config& config)
{
    // initialize simulation domain
    auto subdomainTmp = data::Subdomain({0_r, 0_r, 0_r}, {0_r, 0_r, 0_r}, config.neighborCutoff);

    // initialize atoms randomly in the domain
    auto atoms = data::Atoms(0);

    // restore initial phase point from file
    io::restoreGRO(config.fileRestoreGRO, subdomainTmp, atoms);

    data::Subdomain subdomain(subdomainTmp.minCorner,
                              subdomainTmp.maxCorner,
                              Kokkos::Array<data::Subdomain::BoundaryCondition, 3>{
                                  data::Subdomain::BoundaryCondition::OPEN,
                                  data::Subdomain::BoundaryCondition::PERIODIC,
                                  data::Subdomain::BoundaryCondition::PERIODIC},
                              config.neighborCutoff);

    // calculate volume of the simulation domain
    const auto volume = subdomain.getVolume();

    std::cout
        << "boundaryCondition x: "
        << static_cast<typename std::underlying_type<data::Subdomain::BoundaryCondition>::type>(
               subdomain.boundaryConditions[0])
        << std::endl;
    std::cout
        << "boundaryCondition y: "
        << static_cast<typename std::underlying_type<data::Subdomain::BoundaryCondition>::type>(
               subdomain.boundaryConditions[1])
        << std::endl;
    std::cout
        << "boundaryCondition z: "
        << static_cast<typename std::underlying_type<data::Subdomain::BoundaryCondition>::type>(
               subdomain.boundaryConditions[2])
        << std::endl;

    std::cout << "maxGhostCorner x: " << subdomain.maxGhostCorner[0] << std::endl;
    std::cout << "maxGhostCorner y: " << subdomain.maxGhostCorner[1] << std::endl;
    std::cout << "maxGhostCorner z: " << subdomain.maxGhostCorner[2] << std::endl;

    std::cout << "ghostLayerThickness x: " << subdomain.ghostLayerThickness[0] << std::endl;
    std::cout << "ghostLayerThickness y: " << subdomain.ghostLayerThickness[1] << std::endl;
    std::cout << "ghostLayerThickness z: " << subdomain.ghostLayerThickness[2] << std::endl;

    // calculate and print initial density
    auto rhoInit = real_c(atoms.numLocalAtoms) / volume;
    std::cout << "rho initial: " << rhoInit << std::endl;
    communication::OpenBoundaryLayer::BoundaryValues rhoReservoir = config.reservoirDensity;

    // set up ghost layer for periodic boundary conditions
    communication::GhostLayer ghostLayer;

    // set up open boundary layer for open boundary conditions
    communication::OpenBoundaryLayer openBoundaryLayer(1234);

    // set up neighbor list
    HalfVerletList verletList;
    real_t maxAtomDisplacement = std::numeric_limits<real_t>::max();
    idx_t rebuildCounter = 0;

    // set up interaction potential and force calculation and application
    action::LennardJones lennardJones(config.r_cut, config.sigma, config.epsilon, config.r_cap);

    // calculate and print box center coordinates
    const auto boxCenter = subdomain.getCenter();

    std::cout << "x center: " << boxCenter[0] << std::endl;
    std::cout << "y center: " << boxCenter[1] << std::endl;
    std::cout << "z center: " << boxCenter[2] << std::endl;

    // set up thermostat for temperature control during equilibration
    action::VelocityVerletLangevinThermostat langevinIntegrator(config.gamma, 1.5_r);

    // set up thermodynamic force for density control
    action::ThermodynamicForce thermodynamicForce({1_r},
                                                  subdomain,
                                                  config.densityBinWidth,
                                                  {config.thermodynamicForceModulation},
                                                  false,
                                                  false);

    // set up application regions
    util::IsInSlab isInThermostatRegionLeft(config.thermostatRegionMinLeft,
                                                 config.thermostatRegionMaxLeft);
    util::IsInSlab isInThermostatRegionRight(config.thermostatRegionMinRight,
                                                 config.thermostatRegionMaxRight);
    util::IsInSymmetricSlab isInThermoForceRegion({boxCenter[0], boxCenter[1], boxCenter[2]},
                                                  config.thermoForceRegionMin,
                                                  config.thermoForceRegionMax,
                                                  AXIS::X,
                                                  std::numeric_limits<real_t>::epsilon());
    util::IsInSlab isInDensitySamplingRegionRight(config.thermostatRegionMinRight - 1_r,
                                                 config.thermostatRegionMinRight);

    // set up timer for runtime measurement
    Kokkos::Timer timer;

    // print table header for simulation statistics
    util::printTable<4>("step", "time", "T", "Ek", "E0", "E", "p", "Nlocal", "Nghost", "rhoInstant", "rhoInstLeft", "rhoInstRight", "rhoResLeft", "rhoResRight");
    util::printTableSep<4>("step", "time", "T", "Ek", "E0", "E", "p", "Nlocal", "Nghost", "rhoInstant", "rhoInstLeft", "rhoInstRight", "rhoResLeft", "rhoResRight");

    // open statistics file for writing simulation statistics
    std::ofstream fStat("statistics.txt");

    real_t densityBinVolume =
        subdomain.diameter[1] * subdomain.diameter[2] * config.densityBinWidth;
    io::DumpProfile dumpDens;
    io::DumpProfile dumpThermoForce;
    dumpDens.open(config.fileOutDens);
    dumpDens.dumpScalarView(Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), data::createGrid(thermodynamicForce.getDensityProfile())));
    dumpThermoForce.open(config.fileOutTF);
    dumpThermoForce.dumpScalarView(Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), data::createGrid(thermodynamicForce.getForce())));

    // main simulation loop
    for (auto step = 0; step < config.nsteps; ++step)
    {
        // integrate equations of motion before force calculation
        maxAtomDisplacement += langevinIntegrator.preForceIntegrate_applyAsymmetric_if(atoms, config.dt, isInThermostatRegionLeft, isInThermostatRegionRight, config.reservoirTemperature[0][0], config.reservoirTemperature[0][1]);

        // remove atoms that left the domain through the open boundary
        openBoundaryLayer.removeOpenBoundaryAtoms(atoms, subdomain);

        // calculate instantaneous densities
        const auto rhoInstantLeft = analysis::getDensity_if(atoms, subdomain, isInThermostatRegionLeft);
        const auto rhoInstantRight = analysis::getDensity_if(atoms, subdomain, isInThermostatRegionRight);
        const auto rhoTargetRight = analysis::getDensity_if(atoms, subdomain, isInDensitySamplingRegionRight);

        rhoReservoir[0][0] += config.reservoirDensityFeedback * (config.reservoirDensity[0][0] - rhoInstantLeft);
        rhoReservoir[0][0] = std::max(0_r, rhoReservoir[0][0]);

        rhoReservoir[0][1] += config.reservoirDensityFeedback * (rhoTargetRight - rhoInstantRight);
        rhoReservoir[0][1] = std::max(0_r, rhoReservoir[0][1]);

        // insert atoms that entered the domain through the open boundary
        openBoundaryLayer.insertOpenBoundaryAtoms(
            atoms, subdomain, config.reservoirTemperature, rhoReservoir, config.mass, config.dt);

        // reset displacement
        maxAtomDisplacement = 0_r;

        // reinsert atoms that left the domain according to periodic boundary conditions
        ghostLayer.exchangeRealAtoms(atoms, subdomain);

        // create ghost atoms in the ghost layer beyond the periodic boundaries
        ghostLayer.createGhostAtoms(atoms, subdomain);

        // rebuild neighbor list
        verletList.build(atoms.getPos(),
                            0,
                            atoms.numLocalAtoms,
                            config.neighborCutoff,
                            config.cell_ratio,
                            subdomain.minGhostCorner.data(),
                            subdomain.maxGhostCorner.data(),
                            config.estimatedMaxNeighbors);
        ++rebuildCounter;

        if (step % config.densitySamplingInterval == 0)
        {
            thermodynamicForce.sample(atoms);
        }

        if (step % config.densityUpdateInterval == 0 && step > 0)
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

            thermodynamicForce.update_if(
                config.smoothingInverseDamping, config.smoothingRange, isInThermoForceRegion);
            
            // thermodynamic force output
            auto thermoForce = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                                   thermodynamicForce.getForce(0));
            dumpThermoForce.dumpScalarView(thermoForce);

            io::dumpThermoForce(format("{0}_i{1:02}_tf.txt",
                                       config.fileOut,
                                       idx_c(std::floor(step / config.densityUpdateInterval))),
                                thermodynamicForce,
                                0);        
        }

        // reset forces to zero
        auto force = atoms.getForce();
        Cabana::deep_copy(force, 0_r);

        thermodynamicForce.applyInterpolated_if(atoms, isInThermoForceRegion);

        // calculate and apply forces
        lennardJones.apply(atoms, verletList);

        // contribute forces calculated on ghost atoms back to real atoms
        ghostLayer.contributeBackGhostToReal(atoms);

        // integrate equations of motion after force calculation
        langevinIntegrator.postForceIntegrate(atoms, config.dt);

        // handle output and statistics
        if (config.bOutput && (step % config.outputInterval == 0))
        {
            // calculate statistics
            auto E0 = (lennardJones.getEnergy()) /
                      real_c(atoms.numLocalAtoms);
            auto Ek = analysis::getMeanKineticEnergy(atoms);
            auto systemMomentum = analysis::getSystemMomentum(atoms);
            auto T = (2_r / 3_r) * Ek;
            auto p = analysis::getPressure(atoms, subdomain);
            auto rhoInstant = analysis::getDensity(atoms, subdomain);

            // print statistics to console
            util::printTable<4>(step,
                             timer.seconds(),
                             T,
                             Ek,
                             E0,
                             E0 + Ek,
                             p,
                             atoms.numLocalAtoms,
                             atoms.numGhostAtoms,
                             rhoInstant,
                             rhoInstantLeft,
                             rhoInstantRight,
                             rhoReservoir[0][0],
                             rhoReservoir[0][1]);

            // dump statistics to file
            fStat << step << " " << timer.seconds() << " " << T << " " << Ek << " " << E0 << " "
                  << E0 + Ek << " " << p << " " << atoms.numLocalAtoms << " " << atoms.numGhostAtoms
                  << " " << rhoInstant << " " << rhoInstantLeft << " " << rhoInstantRight << " " << rhoReservoir[0][0] << " " << rhoReservoir[0][1] << std::endl;
        }
    }
    if (config.bOutput)
    {
        // close statistics file
        fStat.close();
        auto time = timer.seconds();
        std::cout << time << std::endl;

        io::dumpGRO(config.fileOutFinalGro,
                    atoms,
                    subdomain,
                    0,
                    config.resName,
                    config.resName,
                    config.typeNames,
                    false,
                    true);
    }

    dumpDens.close();
    dumpThermoForce.close();

    // final thermodynamic force output
    io::dumpThermoForce(config.fileOutFinalTF, thermodynamicForce, 0);

    // write performance data to file
    auto cores = util::getEnvironmentVariable("OMP_NUM_THREADS");
    std::ofstream fout("ecab.perf", std::ofstream::app);
    fout << cores << ", " << time << ", " << atoms.numLocalAtoms << ", " << config.nsteps
         << std::endl;
    fout.close();
}

int main(int argc, char* argv[])  // NOLINT
{
    // initialize Kokkos
    Kokkos::ScopeGuard scope_guard(argc, argv);

    // print Kokkos execution space
    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    // initialize simulation configuration with command line interface
    Config config;
    CLI::App app{"Lennard Jones Fluid benchmark application"};
    app.add_option("-n,--nsteps", config.nsteps, "number of simulation steps");
    app.add_option("-d,--tstep", config.dt, "time step");
    app.add_option("-o,--outint", config.outputInterval, "output interval");
    app.add_option("-f,--outfile", config.fileOut, "output file name");

    //app.add_option("--temp", config.temperature, "target temperature");
    app.add_option("--friction", config.gamma, "friction coefficient for langevin thermostat");
    app.add_option("--density-feedback",
                   config.reservoirDensityFeedback,
                   "feedback gain for reservoir density control (0 disables)");

    app.add_option("--sampling", config.densitySamplingInterval, "density sampling interval");
    app.add_option("--update", config.densityUpdateInterval, "density update interval");
    app.add_option("--densbinwidth", config.densityBinWidth, "density bin width");
    app.add_option("--damping", config.smoothingDamping, "density smoothing damping factor");
    app.add_option("--neighbors", config.smoothingNeighbors, "density smoothing neighbors");
    app.add_option(
        "--forcemod", config.thermodynamicForceModulation, "thermodynamic force modulation");
    app.add_option(
        "--rcap", config.r_cap, "capping radius for inner Lennard-Jones potential");

    //app.add_option(
    //    "--thermostatmin", config.thermostatRegionMin, "thermostat region minimum coordinate");
    //app.add_option(
    //    "--thermostatmax", config.thermostatRegionMax, "thermostat region maximum coordinate");
    app.add_option("--thermoforcemin",
                   config.thermoForceRegionMin,
                   "thermodynamic force region minimum coordinate");
    app.add_option("--thermoforcemax",
                   config.thermoForceRegionMax,
                   "thermodynamic force region maximum coordinate");

    CLI11_PARSE(app, argc, argv);

    config.fileOutTF = format("{0}_tf.txt", config.fileOut);
    config.fileOutDens = format("{0}_dens.txt", config.fileOut);
    config.fileOutFinalGro = format("{0}_final.gro", config.fileOut);
    config.fileOutFinalTF = format("{0}_final_tf.txt", config.fileOut);

    config.smoothingRange =
        real_c(config.smoothingNeighbors) * config.densityBinWidth * config.smoothingDamping;

    // reset output parameter if output interval is negative
    if (config.outputInterval < 0) config.bOutput = false;


    // set up run simulation
    runLennardJones_idealGas_localCap(config);

    return EXIT_SUCCESS;
}