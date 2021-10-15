#include "UpdateGhostAtoms.hpp"

#include <Kokkos_Core.hpp>

#include <cassert>

#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
namespace UpdateGhostAtoms
{
void updateOnlyPos(data::Atoms& atoms,
                   const IndexView& correspondingRealAtom,
                   const data::Subdomain& subdomain)
{
    auto pos = atoms.getPos();

    auto policy =
        Kokkos::RangePolicy<>(atoms.numLocalAtoms, atoms.numLocalAtoms + atoms.numGhostAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        auto realIdx = correspondingRealAtom(idx);

        assert(realIdx != idx);
        assert(correspondingRealAtom_(realIdx) == -1);

        real_t dx[3];
        dx[0] = pos(idx, 0) - pos(realIdx, 0);
        dx[1] = pos(idx, 1) - pos(realIdx, 1);
        dx[2] = pos(idx, 2) - pos(realIdx, 2);

        pos(idx, 0) = pos(realIdx, 0);
        pos(idx, 1) = pos(realIdx, 1);
        pos(idx, 2) = pos(realIdx, 2);

        real_t delta[3];
        delta[0] = 0.1_r * subdomain.diameter[0];
        delta[1] = 0.1_r * subdomain.diameter[1];
        delta[2] = 0.1_r * subdomain.diameter[2];
        if (dx[0] > +delta[0]) pos(idx, 0) += subdomain.diameter[0];
        if (dx[1] > +delta[1]) pos(idx, 1) += subdomain.diameter[1];
        if (dx[2] > +delta[2]) pos(idx, 2) += subdomain.diameter[2];
        if (dx[0] < -delta[0]) pos(idx, 0) -= subdomain.diameter[0];
        if (dx[1] < -delta[1]) pos(idx, 1) -= subdomain.diameter[1];
        if (dx[2] < -delta[2]) pos(idx, 2) -= subdomain.diameter[2];
    };
    Kokkos::parallel_for(policy, kernel, "UpdateGhostAtoms::updateOnlyPos");
    Kokkos::fence();
}

}  // namespace UpdateGhostAtoms
}  // namespace impl
}  // namespace communication
}  // namespace mrmd