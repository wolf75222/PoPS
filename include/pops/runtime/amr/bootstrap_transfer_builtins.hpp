#pragma once

#include <pops/runtime/amr/bootstrap_transfer_registry.hpp>

#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/time/amr/levels/amr_subcycling.hpp>

#include <stdexcept>
#include <vector>

namespace pops::runtime::amr {

inline PreparedTransferKernel prepare_volume_average() {
  PreparedTransferKernel kernel;
  kernel.spatial = [](const MultiFab& fine, MultiFab& coarse,
                      const SpatialTransferContext& context) {
    if (context.index.refinement_ratio != std::vector<int>{2, 2})
      throw std::runtime_error("volume-average restriction ratio mismatch");
    if (fine.ncomp() != coarse.ncomp() || context.components != fine.ncomp())
      throw std::runtime_error("volume-average restriction component mismatch");
    const Box2D coarse_box = coarse.box_array().bounding_box();
    if (context.index.coarse_origin != std::vector<int>{coarse_box.lo[0], coarse_box.lo[1]} ||
        context.index.fine_origin != std::vector<int>{2 * coarse_box.lo[0], 2 * coarse_box.lo[1]})
      throw std::runtime_error("volume-average restriction origin mismatch");

    // A replicated parent deliberately has rank-local ownership metadata: every rank owns its
    // local parent copy.  Generic parallel_copy/average_down cannot build a symmetric MPI schedule
    // from such metadata.  This hierarchy-aware primitive deposits local child averages into a
    // global-indexed buffer, performs one symmetric reduction, then updates every local parent
    // owner/copy.  The same prepared route therefore covers both replicated and distributed
    // parents without an ownership-policy branch.
    mf_average_down_mb(fine, coarse);
  };
  return kernel;
}

inline PreparedTransferKernel prepare_conservative_coarse_fine() {
  PreparedTransferKernel kernel;
  kernel.coarse_fine = [](const MultiFab& parent, MultiFab& fine,
                          const SpatialTransferContext& context) {
    if (context.index.refinement_ratio != std::vector<int>{2, 2})
      throw std::runtime_error("coarse/fine spatial transfer ratio mismatch");
    const Box2D coarse = parent.box_array().bounding_box();
    if (context.index.coarse_origin != std::vector<int>{coarse.lo[0], coarse.lo[1]} ||
        context.index.fine_origin != std::vector<int>{2 * coarse.lo[0], 2 * coarse.lo[1]})
      throw std::runtime_error("coarse/fine spatial transfer origin mismatch");
    mf_fill_fine_ghosts_spatial_mb(fine, parent, context.replicated_parent);
  };
  return kernel;
}

inline PreparedTransferKernel prepare_linear_time_interpolation() {
  PreparedTransferKernel kernel;
  kernel.temporal = [](const MultiFab& old_value, const MultiFab& new_value, MultiFab& destination,
                       const TemporalTransferContext& time) {
    if (&old_value == &new_value)
      throw std::runtime_error("temporal interpolation requires two distinct physical snapshots");
    const double alpha = time.alpha();
    lincomb(destination, Real(1 - alpha), old_value, Real(alpha), new_value);
  };
  return kernel;
}

}  // namespace pops::runtime::amr
