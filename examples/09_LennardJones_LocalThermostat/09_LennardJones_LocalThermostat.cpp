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
#include "action/VelocityVerletLangevinThermostat.hpp"
#include "analysis/AxialAverageProfile.hpp"
#include "analysis/AxialDensityProfile.hpp"
#include "analysis/AxialTemperatureProfile.hpp"
#include "analysis/KineticEnergy.hpp"
#include "analysis/MeanSquareDisplacement.hpp"
#include "analysis/Pressure.hpp"
#include "analysis/SystemMomentum.hpp"
#include "communication/GhostLayer.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "initialization.hpp"
#include "io/DumpProfile.hpp"
#include "io/RestoreH5MD.hpp"
#include "util/EnvironmentVariables.hpp"
#include "util/IsInSymmetricSlab.hpp"
#include "util/PrintTable.hpp"
#include "util/simulationSetup.hpp"

using namespace mrmd;

/**
 * Configuration for the Lennard-Jones local thermostat example simulation.
 */
struct Config
{
    // simulation time parameters
    idx_t nsteps = 400001;               ///< number of steps to simulate
    static constexpr real_t dt = 0.002;  ///< time step size in reduced units

    // input file parameters
    std::string fileRestoreH5MD =
        "equilibrateLangevin_final.h5md";  ///< name of the file to restore the initial phase point
                                           ///< from

    // interaction parameters
    static constexpr real_t sigma =
        1_r;  ///< distance at which LJ potential is zero in reduced units
    static constexpr real_t epsilon = 1_r;  ///< energy well depth of LJ potential in reduced units
    static constexpr real_t mass = 1_r;     ///< mass of one atom in reduced units
    static constexpr real_t r_cut = 2.5_r * sigma;  ///< cutoff radius for LJ potential
    real_t r_cap = 0_r;                             ///< capping radius for LJ potential

    // neighbor list parameters
    static constexpr real_t skin = 0.1_r * sigma;           ///< skin thickness for neighbor list
    static constexpr real_t neighborCutoff = r_cut + skin;  ///< cutoff radius for neighbor list
    static constexpr real_t cell_ratio =
        1_r;  ///< ratio of cell size on Cartesian grid to cutoff radius for neighbor list
    static constexpr idx_t estimatedMaxNeighbors =
        60;  ///< estimated maximum number of neighbors per atom

    // thermostat parameters
    real_t temperatureLeft =
        1.5_r;  ///< target temperature during equilibration for thermostat in reduced units
    real_t temperatureRight =
        2_r;  ///< target temperature during equilibration for thermostat in reduced units
    static constexpr real_t friction =
        0.04_r / dt;  ///< friction coefficient for Langevin thermostat

    real_t thermostatRegionMin =
        0_r * sigma;  ///< minimum x-coordinate of the thermodynamic force region
    real_t thermostatRegionMax =
        15_r * sigma;  ///< maximum x-coordinate of the thermodynamic force region

    // profile sampling parameters
    idx_t profileSamplingInterval = 200;     ///< interval for sampling profiles
    real_t profileBinWidth = 0.2_r * sigma;  ///< bin width for profiles

    // output parameters
    bool bOutput = true;                  ///< whether to output data files
    idx_t outputInterval = -1;            ///< interval for data file output (-1: no output)
    const std::string resName = "Argon";  ///< residue name for output files
    const std::vector<std::string> typeNames = {"Ar"};  ///< atom type names for output files

    std::string fileOut = "localThermostat";  ///< base name for output files
    std::string fileOutDens;
    std::string fileOutTemp;
};

class LeftRightEvaluator
{
private:
    const real_t leftValue_;
    const real_t rightValue_;
    const real_t center_;
    const AXIS axis_;

public:
    LeftRightEvaluator(real_t leftValue, real_t rightValue, real_t center, AXIS axis)
        : leftValue_(leftValue), rightValue_(rightValue), center_(center), axis_(axis)
    {
    }

    KOKKOS_INLINE_FUNCTION
    real_t operator()(const real_t x, const real_t y, const real_t z) const
    {
        real_t coord = 0_r;
        switch (axis_)
        {
            case AXIS::X:
                coord = x;
                break;
            case AXIS::Y:
                coord = y;
                break;
            case AXIS::Z:
                coord = z;
                break;
        }

        if (coord < center_)
            return leftValue_;
        else
            return rightValue_;
    }
};

void lennardJones_localThermostat(Config& config)
{
    // initialize
    data::Subdomain subdomain;
    auto atoms = data::Atoms(0);

    // load data from file
    auto io = io::RestoreH5MD();
    io.restore(config.fileRestoreH5MD, subdomain, atoms);

    // calculate volume of the simulation domain
    const auto volume = subdomain.getVolume();

    // calculate and print initial density
    auto rho = real_c(atoms.numLocalAtoms) / volume;
    std::cout << "rho: " << rho << std::endl;

    // set up ghost layer for periodic boundary conditions
    communication::GhostLayer ghostLayer;

    // set up neighbor list
    HalfVerletList verletList;
    real_t maxAtomDisplacement = std::numeric_limits<real_t>::max();
    idx_t rebuildCounter = 0;

    // set up interaction potential and force calculation and application
    action::LennardJones lennardJones(config.r_cut, config.sigma, config.epsilon, 0_r);
    action::LennardJones lennardJonesCap(config.r_cut, config.sigma, config.epsilon, config.r_cap);

    // calculate and print box center coordinates
    const auto boxCenter = subdomain.getCenter();

    std::cout << "x center: " << boxCenter[0] << std::endl;
    std::cout << "y center: " << boxCenter[1] << std::endl;
    std::cout << "z center: " << boxCenter[2] << std::endl;

    // set up regions
    util::IsInSymmetricSlab isInThermostatRegion(
        boxCenter, config.thermostatRegionMin, config.thermostatRegionMax);

    LeftRightEvaluator evalTemperature(
        config.temperatureLeft, config.temperatureRight, boxCenter[0], AXIS::X);

    // set up thermostat for temperature control during equilibration
    action::VelocityVerletLangevinThermostat langevinIntegrator;

    // set up profile sampling
    analysis::AxialAverageProfile densityProfile(
        subdomain,
        config.profileBinWidth,
        config.profileBinWidth * subdomain.getAreaNormalToAxis(AXIS::X),
        atoms.getNumTypes(),
        AXIS::X);

    analysis::AxialAverageProfile temperatureProfile(
        subdomain,
        config.profileBinWidth,
        2_r/3_r,  // normalization factor for kinetic energy to temperature
        atoms.getNumTypes(),
        AXIS::X);

    // set up timer for runtime measurement
    Kokkos::Timer timer;

    // set up mean square displacement analysis
    analysis::MeanSquareDisplacement meanSquareDisplacement;
    meanSquareDisplacement.reset(atoms);
    auto msd = 0_r;

    // output management
    io::DumpProfile dumpDens;
    io::DumpProfile dumpTemp;
    std::ofstream fStat("statistics.txt");
    if (config.bOutput)
    {
        // print table header for simulation statistics
        util::printTable("step", "time", "T", "Ek", "E0", "E", "p", "msd", "Nlocal", "Nghost");
        util::printTableSep("step", "time", "T", "Ek", "E0", "E", "p", "msd", "Nlocal", "Nghost");
        dumpDens.open(config.fileOutDens);
        dumpDens.dumpScalarView(Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), data::createGrid(densityProfile.getAverageProfile())));
        dumpTemp.open(config.fileOutTemp);
        dumpTemp.dumpScalarView(Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), data::createGrid(temperatureProfile.getAverageProfile())));
    }

    // main simulation loop
    for (auto step = 0; step < config.nsteps; ++step)
    {
        // integrate equations of motion before force calculation
        maxAtomDisplacement += langevinIntegrator.preForceIntegrate_apply_if_as(
            atoms,
            config.dt,
            isInThermostatRegion,
            evalTemperature,
            KOKKOS_LAMBDA(const real_t, const real_t, const real_t) { return config.friction; });

        // check if neighbor list needs to be rebuilt
        if (maxAtomDisplacement >=
            config.skin *
                0.5_r)  // the condition is on half the skin thickness because in principle two
                        // atoms may both move half the skin thickness towards each other
        {
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
        }
        else
        {
            // update ghost atom positions in the ghost layer according to periodic boundary
            // conditions
            ghostLayer.updateGhostAtoms(atoms, subdomain);
        }

        if (step % config.profileSamplingInterval == 0)
        {
            densityProfile.sample(atoms, analysis::getAxialParticleNumberProfile);

            temperatureProfile.sample(atoms, analysis::getAxialMeanKineticEnergyProfile);
        }

        if (config.bOutput && (step % config.outputInterval == 0))
        {
            // profile output
            auto densityProfileView = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), densityProfile.getAverageProfile(0));
            dumpDens.dumpScalarView(densityProfileView);
            auto temperatureProfileView = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), temperatureProfile.getAverageProfile(0));
            dumpTemp.dumpScalarView(temperatureProfileView);
        }

        // reset forces to zero
        auto force = atoms.getForce();
        Cabana::deep_copy(force, 0_r);

        // compute and apply forces
        lennardJones.apply(atoms, verletList);

        // contribute forces calculated on ghost atoms back to real atoms
        ghostLayer.contributeBackGhostToReal(atoms);

        // integrate equations of motion after force calculation
        langevinIntegrator.postForceIntegrate(atoms, config.dt);

        // handle output and statistics
        if (config.bOutput && (step % config.outputInterval == 0))
        {
            // calculate statistics
            auto E0 = (lennardJones.getEnergy() + lennardJonesCap.getEnergy()) /
                      real_c(atoms.numLocalAtoms);
            auto Ek = analysis::getMeanKineticEnergy(atoms);
            auto systemMomentum = analysis::getSystemMomentum(atoms);
            auto T = (2_r / 3_r) * Ek;
            auto p = analysis::getPressure(atoms, subdomain);
            msd = meanSquareDisplacement.calc(atoms, subdomain) /
                  (real_c(config.outputInterval) * config.dt);
            meanSquareDisplacement.reset(atoms);

            // print statistics to console
            util::printTable(step,
                             timer.seconds(),
                             T,
                             Ek,
                             E0,
                             E0 + Ek,
                             p,
                             msd,
                             atoms.numLocalAtoms,
                             atoms.numGhostAtoms);

            // dump statistics to file
            fStat << step << " " << timer.seconds() << " " << T << " " << Ek << " " << E0 << " "
                  << E0 + Ek << " " << p << " " << msd << " " << atoms.numLocalAtoms << " "
                  << atoms.numGhostAtoms << " " << std::endl;
        }
    }

    if (config.bOutput)
    {
        dumpDens.close();
        dumpTemp.close();

        // close statistics file
        fStat.close();
    }

    // write performance data to file
    auto time = timer.seconds();
    std::cout << time << std::endl;
    auto cores = util::getEnvironmentVariable("OMP_NUM_THREADS");
    std::ofstream fout("ecab.perf", std::ofstream::app);
    fout << cores << ", " << time << ", " << atoms.numLocalAtoms << ", " << config.nsteps
         << std::endl;
    fout.close();
}

int main(int argc, char* argv[])
{
    // initialize Kokkos
    Kokkos::ScopeGuard scope_guard(argc, argv);

    // print Kokkos execution space
    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    // initialize simulation configuration with command line interface
    Config config;
    CLI::App app{"Lennard Jones Fluid benchmark application"};
    app.add_option("-n,--nsteps", config.nsteps, "total number of simulation steps");
    app.add_option("-o,--outint", config.outputInterval, "output interval");
    app.add_option("-i,--inpfile", config.fileRestoreH5MD, "input file name");
    app.add_option("-f,--outfile", config.fileOut, "output file name");

    app.add_option("--temperature-left",
                   config.temperatureLeft,
                   "temperature of the left-hand Langevin thermostat (negative numbers deactivate "
                   "the thermostat)");
    app.add_option("--temperature-right",
                   config.temperatureRight,
                   "temperature of the right-hand Langevin thermostat (negative numbers deactivate "
                   "the thermostat)");

    CLI11_PARSE(app, argc, argv);

    config.fileOutDens = format("{0}_dens.txt", config.fileOut);
    config.fileOutTemp = format("{0}_temp.txt", config.fileOut);

    // reset output parameter if output interval is negative
    if (config.outputInterval < 0) config.bOutput = false;

    // set up run simulation
    lennardJones_localThermostat(config);

    return EXIT_SUCCESS;
}