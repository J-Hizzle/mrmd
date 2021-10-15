#include "MultiResRealAtomsExchange.hpp"

namespace mrmd
{
namespace communication
{
void realAtomsExchange(const data::Subdomain& subdomain,
                       const data::Molecules& molecules,
                       const data::Atoms& atoms)
{
    auto moleculesPos = molecules.getPos();
    auto atomsOffset = molecules.getAtomsOffset();
    auto numAtoms = molecules.getNumAtoms();

    auto atomsPos = atoms.getPos();

    auto policy = Kokkos::RangePolicy<>(0, molecules.numLocalMolecules);
    auto kernel = KOKKOS_LAMBDA(const idx_t& moleculeIdx)
    {
        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            auto& moleculeX = moleculesPos(moleculeIdx, dim);
            if (subdomain.maxCorner[dim] <= moleculeX)
            {
                moleculeX -= subdomain.diameter[dim];

                auto atomsStart = atomsOffset(moleculeIdx);
                auto atomsEnd = atomsStart + numAtoms(moleculeIdx);
                for (idx_t atomIdx = atomsStart; atomIdx < atomsEnd; ++atomIdx)
                {
                    auto& atomX = atomsPos(atomIdx, dim);
                    atomX -= subdomain.diameter[dim];
                }
            }
            if (moleculeX < subdomain.minCorner[dim])
            {
                moleculeX += subdomain.diameter[dim];

                auto atomsStart = atomsOffset(moleculeIdx);
                auto atomsEnd = atomsStart + numAtoms(moleculeIdx);
                for (idx_t atomIdx = atomsStart; atomIdx < atomsEnd; ++atomIdx)
                {
                    auto& atomX = atomsPos(atomIdx, dim);
                    atomX += subdomain.diameter[dim];
                }
            }
            assert(moleculeX < subdomain.maxCorner[dim]);
            assert(subdomain.minCorner[dim] <= moleculeX);
        }
    };
    Kokkos::parallel_for(policy, kernel, "realAtomsExchange::periodicMapping");
    Kokkos::fence();
}

}  // namespace communication
}  // namespace mrmd