#include "NPT.hpp"

#include "action/BerendsenBarostat.hpp"
#include "action/BerendsenThermostat.hpp"
#include "action/LennardJones.hpp"
#include "action/VelocityVerlet.hpp"
#include "analysis/KineticEnergy.hpp"
#include "communication/GhostLayer.hpp"
#include "util/ExponentialMovingAverage.hpp"
#include "util/PrintTable.hpp"

namespace mrmd
{
void npt(YAML::Node& config, data::Atoms& atoms, data::Subdomain& subdomain)
{
    constexpr int64_t estimatedMaxNeighbors = 60;
    constexpr real_t cellRatio = 0.5_r;
    const real_t skin = config["LJ"]["skin"].as<real_t>();
    auto rcVec = config["LJ"]["cutoff"].as<std::vector<real_t>>();
    const real_t rc = *std::max_element(rcVec.begin(), rcVec.end());
    const real_t neighborCutoff = rc + skin;
    auto volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];

    communication::GhostLayer ghostLayer;
    auto LJ = action::LennardJones(config["LJ"]["capping"].as<std::vector<real_t>>(),
                                   config["LJ"]["cutoff"].as<std::vector<real_t>>(),
                                   config["LJ"]["sigma"].as<std::vector<real_t>>(),
                                   config["LJ"]["epsilon"].as<std::vector<real_t>>(),
                                   2,
                                   true);
    VerletList verletList;

    real_t maxAtomDisplacement = std::numeric_limits<real_t>::max();
    util::ExponentialMovingAverage currentPressure(
        config["pressure_averaging_coefficient"].as<real_t>());
    util::ExponentialMovingAverage currentTemperature(
        config["temperature_averaging_coefficient"].as<real_t>());
    currentTemperature << analysis::getMeanKineticEnergy(atoms) * 2_r / 3_r;

    if (config["enable_output"].as<bool>())
        util::printTable("step", "T", "p", "V", "Nlocal", "Nghost");
    if (config["enable_output"].as<bool>())
        util::printTableSep("step", "T", "p", "V", "Nlocal", "Nghost");
    for (auto step = 0; step < config["time_steps"].as<int64_t>(); ++step)
    {
        maxAtomDisplacement +=
            action::VelocityVerlet::preForceIntegrate(atoms, config["dt"].as<real_t>());

        if ((step > 200) && (step % config["barostat_interval"].as<int64_t>() == 0))
        {
            action::BerendsenBarostat::apply<false, true, false>(
                atoms,
                currentPressure,
                config["target_pressure"].as<real_t>(),
                config["pressure_relaxation_coefficient"].as<real_t>(),
                subdomain);
            volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];
            maxAtomDisplacement = std::numeric_limits<real_t>::max();
        }

        if (step % config["thermostat_interval"].as<int64_t>() == 0)
        {
            action::BerendsenThermostat::apply(
                atoms,
                currentTemperature,
                config["target_temperature"].as<real_t>(),
                config["temperature_relaxation_coefficient"].as<real_t>());
        }

        if (maxAtomDisplacement >= skin * 0.5_r)
        {
            // reset displacement
            maxAtomDisplacement = 0_r;

            ghostLayer.exchangeRealAtoms(atoms, subdomain);

            real_t gridDelta[3] = {neighborCutoff, neighborCutoff, neighborCutoff};
            LinkedCellList linkedCellList(atoms.getPos(),
                                          0,
                                          atoms.numLocalAtoms,
                                          gridDelta,
                                          subdomain.minCorner.data(),
                                          subdomain.maxCorner.data());
            atoms.permute(linkedCellList);

            ghostLayer.createGhostAtoms(atoms, subdomain);
            verletList.build(atoms.getPos(),
                             0,
                             atoms.numLocalAtoms,
                             neighborCutoff,
                             cellRatio,
                             subdomain.minGhostCorner.data(),
                             subdomain.maxGhostCorner.data(),
                             estimatedMaxNeighbors);
        }
        else
        {
            ghostLayer.updateGhostAtoms(atoms, subdomain);
        }

        auto force = atoms.getForce();
        Cabana::deep_copy(force, 0_r);

        LJ.applyForces(atoms, verletList);

        if (step < 201)
        {
            currentPressure = util::ExponentialMovingAverage(
                config["pressure_averaging_coefficient"].as<real_t>());
            currentTemperature = util::ExponentialMovingAverage(
                config["temperature_averaging_coefficient"].as<real_t>());
        }
        auto Ek = analysis::getKineticEnergy(atoms);
        currentPressure << 2_r * (Ek - LJ.getVirial()) / (3_r * volume);
        Ek /= real_c(atoms.numLocalAtoms);
        currentTemperature << (2_r / 3_r) * Ek;

        ghostLayer.contributeBackGhostToReal(atoms);
        action::VelocityVerlet::postForceIntegrate(atoms, config["dt"].as<real_t>());

        if ((config["enable_output"].as<bool>()) &&
            (step % config["output_interval"].as<int64_t>() == 0))
        {
            util::printTable(step,
                             currentTemperature,
                             currentPressure,
                             volume,
                             atoms.numLocalAtoms,
                             atoms.numGhostAtoms);
        }
    }
    if (config["enable_output"].as<bool>())
        util::printTableSep("step", "T", "p", "V", "Nlocal", "Nghost");
}
}  // namespace mrmd