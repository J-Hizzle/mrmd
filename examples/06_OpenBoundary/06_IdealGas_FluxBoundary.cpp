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
#include "action/LimitAcceleration.hpp"
#include "action/LimitVelocity.hpp"
#include "action/VelocityVerletLangevinThermostat.hpp"
#include "analysis/KineticEnergy.hpp"
#include "analysis/MeanSquareDisplacement.hpp"
#include "analysis/Pressure.hpp"
#include "analysis/SystemMomentum.hpp"
#include "communication/GhostLayer.hpp"
#include "communication/OpenBoundaryLayer.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "initialization.hpp"
#include "io/DumpGRO.hpp"
#include "util/EnvironmentVariables.hpp"
#include "util/IsInSymmetricSlab.hpp"
#include "util/PrintTable.hpp"
#include "util/simulationSetup.hpp"

using namespace mrmd;

/**
 * Configuration for the Argon NVE example simulation.
 */
struct Config
{
    // simulation time parameters
    idx_t nsteps = 400001;  ///< number of steps to simulate
    real_t dt = 0.002_r;    ///< time step size in reduced units

    // interaction parameters
    static constexpr real_t sigma =
        1_r;  ///< distance at which LJ potential is zero in reduced units
    static constexpr real_t epsilon = 1_r;  ///< energy well depth of LJ potential in reduced units
    static constexpr real_t mass = 1_r;     ///< mass of one atom in reduced units
    static constexpr real_t maxVelocity =
        1_r;  ///< maximum initial velocity component in reduced units

    // system parameters
    static constexpr idx_t numAtoms = 16 * 16 * 16;  ///< number of atoms in the simulation
    real_t Lx = 30_r * sigma;                        ///< box edge length in x-direction

    // thermostat parameters
    real_t temperature =
        1.5_r;  ///< target temperature during equilibration for thermostat in reduced units
    real_t gamma = 0.04_r / dt;  ///< friction coefficient for Langevin thermostat
    real_t reservoirDensityFeedback =
        0.002_r;  ///< feedback gain for reservoir density control (0 disables control)

    // output parameters
    bool bOutput = true;                  ///< whether to output data files
    idx_t outputInterval = -1;            ///< interval for data file output (-1: no output)
    const std::string resName = "Argon";  ///< residue name for output files
    const std::vector<std::string> typeNames = {"Ar"};  ///< atom type names for output files

    std::string fileOut = "idealGas_fluxBoundary";  ///< base name for output files
    std::string fileOutFinalGro = format("{0}_final.gro", fileOut);
};

void runLennardJones_idealGas_localCap(Config& config)
{
    // initialize simulation domain
    data::Subdomain subdomain({0_r, 0_r, 0_r},
                              {config.Lx, config.Lx, config.Lx},
                              Kokkos::Array<data::Subdomain::BoundaryCondition, 3>{
                                  data::Subdomain::BoundaryCondition::OPEN,
                                  data::Subdomain::BoundaryCondition::PERIODIC,
                                  data::Subdomain::BoundaryCondition::PERIODIC},
                              0_r);

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

    // calculate volume of the simulation domain
    const auto volume = subdomain.getVolume();

    // initialize atoms randomly in the domain
    auto atoms =
        util::fillDomainWithAtoms(subdomain, config.numAtoms, config.maxVelocity, config.mass);

    // calculate and print initial density
    auto rhoTarget = real_c(atoms.numLocalAtoms) / volume;
    auto rhoReservoir = rhoTarget;
    std::cout << "rho target: " << rhoTarget << std::endl;

    // set up ghost layer for periodic boundary conditions
    communication::GhostLayer ghostLayer;

    // set up open boundary layer for open boundary conditions
    communication::OpenBoundaryLayer openBoundaryLayer(1234);

    // set up neighbor list
    HalfVerletList verletList;
    real_t maxAtomDisplacement = std::numeric_limits<real_t>::max();
    idx_t rebuildCounter = 0;

    // calculate and print box center coordinates
    const auto boxCenter = subdomain.getCenter();

    std::cout << "x center: " << boxCenter[0] << std::endl;
    std::cout << "y center: " << boxCenter[1] << std::endl;
    std::cout << "z center: " << boxCenter[2] << std::endl;

    // set up thermostat for temperature control during equilibration
    action::VelocityVerletLangevinThermostat langevinIntegrator(config.gamma, config.temperature);

    // set up timer for runtime measurement
    Kokkos::Timer timer;

    // print table header for simulation statistics
    util::printTable("step", "time", "T", "Ek", "E0", "E", "p", "Nlocal", "Nghost");
    util::printTableSep("step", "time", "T", "Ek", "E0", "E", "p", "Nlocal", "Nghost");

    // open statistics file for writing simulation statistics
    std::ofstream fStat("statistics.txt");

    VectorView previousPos("previousPos", atoms.size());

    // main simulation loop
    for (auto step = 0; step < config.nsteps; ++step)
    {
        if (previousPos.extent(0) < atoms.size())
        {
            previousPos = VectorView("previousPos", atoms.size());
        }

        {
            auto pos = atoms.getPos();
            auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
            auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
            {
                previousPos(idx, 0) = pos(idx, 0);
                previousPos(idx, 1) = pos(idx, 1);
                previousPos(idx, 2) = pos(idx, 2);
            };
            Kokkos::parallel_for("snapshotPreviousPositions", policy, kernel);
            Kokkos::fence();
        }

        // integrate equations of motion before force calculation
        maxAtomDisplacement += langevinIntegrator.preForceIntegrate(atoms, config.dt);

        // remove atoms that left the domain through the open boundary
        openBoundaryLayer.removeOpenBoundaryAtoms(atoms, subdomain, previousPos, config.dt);

        const auto rhoInstant = real_c(atoms.numLocalAtoms) / volume;
        rhoReservoir += config.reservoirDensityFeedback * (rhoTarget - rhoInstant);
        rhoReservoir = std::max(0_r, rhoReservoir);

        // insert atoms that entered the domain through the open boundary
        openBoundaryLayer.insertOpenBoundaryAtoms(
            atoms, subdomain, config.temperature, rhoReservoir, config.mass, config.dt);

        // reinsert atoms that left the domain according to periodic boundary conditions
        ghostLayer.exchangeRealAtoms(atoms, subdomain);

        // create ghost atoms in the ghost layer beyond the periodic boundaries
        ghostLayer.createGhostAtoms(atoms, subdomain);

        // integrate equations of motion after force calculation
        langevinIntegrator.postForceIntegrate(atoms, config.dt);

        // handle output and statistics
        if (config.bOutput && (step % config.outputInterval == 0))
        {
            // calculate statistics
            auto E0 = 0_r;  // potential energy is zero for ideal gas
            auto Ek = analysis::getMeanKineticEnergy(atoms);
            auto systemMomentum = analysis::getSystemMomentum(atoms);
            auto T = (2_r / 3_r) * Ek;
            auto p = analysis::getPressure(atoms, subdomain);

            // print statistics to console
            util::printTable(step,
                             timer.seconds(),
                             T,
                             Ek,
                             E0,
                             E0 + Ek,
                             p,
                             atoms.numLocalAtoms,
                             atoms.numGhostAtoms);

            // dump statistics to file
            fStat << step << " " << timer.seconds() << " " << T << " " << Ek << " " << E0 << " "
                  << E0 + Ek << " " << p << " " << atoms.numLocalAtoms << " " << atoms.numGhostAtoms
                  << " " << std::endl;
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

    // write performance data to file
    auto cores = util::getEnvironmentVariable("OMP_NUM_THREADS");
    std::ofstream fout("ecab.perf", std::ofstream::app);
    fout << cores << ", " << time << ", " << atoms.numLocalAtoms << ", " << config.nsteps
         << std::endl;
    fout.close();
}

int main(int argc, char* argv[])  // NOLINT
{
    // initialize
    Kokkos::initialize(argc, argv);

    // print Kokkos execution space
    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    // initialize simulation configuration with command line interface
    Config config;
    CLI::App app{"Lennard Jones Fluid benchmark application"};
    app.add_option("-n,--nsteps", config.nsteps, "number of simulation steps");
    app.add_option("-d,--tstep", config.dt, "time step");
    app.add_option("-o,--outint", config.outputInterval, "output interval");
    app.add_option("-f,--outfile", config.fileOut, "output file name");

    app.add_option("--temp", config.temperature, "target temperature");
    app.add_option("--friction", config.gamma, "friction coefficient for langevin thermostat");
    app.add_option("--rho-feedback",
                   config.reservoirDensityFeedback,
                   "feedback gain for reservoir density control (0 disables)");

    CLI11_PARSE(app, argc, argv);

    // reset output parameter if output interval is negative
    if (config.outputInterval < 0) config.bOutput = false;

    // set up run simulation
    runLennardJones_idealGas_localCap(config);

    // finalize
    Kokkos::finalize();

    return EXIT_SUCCESS;
}