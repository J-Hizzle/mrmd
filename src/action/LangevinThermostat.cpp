#include "LangevinThermostat.hpp"

#include <Kokkos_Core.hpp>

namespace mrmd
{
namespace action
{
void LangevinThermostat::apply(data::Particles& particles)
{
    auto RNG = randPool_;
    auto vel = particles.getVel();
    auto force = particles.getForce();

    auto policy = Kokkos::RangePolicy<>(0, particles.numLocalParticles);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        constexpr real_t mass = 1_r;
        constexpr real_t massSqrt = 1_r;  // std::sqrt(mass);

        // Get a random number state from the pool for the active thread
        auto randGen = RNG.get_state();

        force(idx, 0) += pref1 * vel(idx, 0) * mass + pref2 * (randGen.drand() - 0.5_r) * massSqrt;
        force(idx, 1) += pref1 * vel(idx, 1) * mass + pref2 * (randGen.drand() - 0.5_r) * massSqrt;
        force(idx, 2) += pref1 * vel(idx, 2) * mass + pref2 * (randGen.drand() - 0.5_r) * massSqrt;

        // Give the state back, which will allow another thread to acquire it
        RNG.free_state(randGen);
    };
    Kokkos::parallel_for(policy, kernel, "LangevinThermostat::applyThermostat");

    Kokkos::fence();
}

}  // namespace action
}  // namespace mrmd