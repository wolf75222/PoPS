#pragma once

#include <pops/coupling/amr/amr_coupler_mp.hpp>
#include <pops/runtime/amr/bootstrap_transfer_registry.hpp>

#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/time/amr/levels/amr_subcycling.hpp>

#include <stdexcept>
#include <vector>

namespace pops::runtime::amr {

inline PreparedTransferKernel prepare_conservative_linear() {
  PreparedTransferKernel kernel;
  kernel.capabilities = PreparedTransferCapabilities{2, {1}};
  kernel.spatial = [](const MultiFab& coarse, MultiFab& fine,
                      const SpatialTransferContext& context) {
    if (context.index.refinement_ratio != std::vector<int>{2, 2})
      throw std::runtime_error("conservative-linear prolongation ratio mismatch");
    if (fine.ncomp() != coarse.ncomp() || context.components != fine.ncomp())
      throw std::runtime_error("conservative-linear prolongation component mismatch");
    if (context.logical_fine_domain != context.logical_coarse_domain.refine(2))
      throw std::runtime_error("conservative-linear prolongation logical-domain mismatch");
    detail::coupler_conservative_linear_to_fine_mb(
        coarse, fine, context.logical_coarse_domain, context.logical_fine_domain,
        context.index.coarse_origin, context.index.fine_origin,
        context.index.refinement_ratio, context.replicated_parent, context.periodicity);
  };
  return kernel;
}

inline PreparedTransferKernel prepare_volume_average() {
  PreparedTransferKernel kernel;
  kernel.capabilities = PreparedTransferCapabilities{1, {0}};
  kernel.spatial = [](const MultiFab& fine, MultiFab& coarse,
                      const SpatialTransferContext& context) {
    if (context.index.refinement_ratio != std::vector<int>{2, 2})
      throw std::runtime_error("volume-average restriction ratio mismatch");
    if (fine.ncomp() != coarse.ncomp() || context.components != fine.ncomp())
      throw std::runtime_error("volume-average restriction component mismatch");
    const Box2D coarse_box = context.logical_coarse_domain;
    if (context.logical_fine_domain != coarse_box.refine(2))
      throw std::runtime_error("volume-average restriction logical-domain mismatch");
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
  kernel.capabilities = PreparedTransferCapabilities{2, {2}};
  kernel.prepared_coarse_fine = std::make_shared<const PreparedCoarseFineOperator>(
      prepare_limited_linear_coarse_fine_operator());
  kernel.coarse_fine = [](const MultiFab& parent, MultiFab& fine,
                          const SpatialTransferContext& context) {
    if (context.index.refinement_ratio != std::vector<int>{2, 2})
      throw std::runtime_error("coarse/fine spatial transfer ratio mismatch");
    const Box2D coarse = context.logical_coarse_domain;
    if (context.logical_fine_domain != coarse.refine(2))
      throw std::runtime_error("coarse/fine spatial transfer logical-domain mismatch");
    if (context.index.coarse_origin != std::vector<int>{coarse.lo[0], coarse.lo[1]} ||
        context.index.fine_origin != std::vector<int>{2 * coarse.lo[0], 2 * coarse.lo[1]})
      throw std::runtime_error("coarse/fine spatial transfer origin mismatch");
    detail::coupler_conservative_linear_fill_ghosts_mb(
        parent, fine, context.logical_coarse_domain, context.logical_fine_domain,
        context.replicated_parent, context.periodicity);
  };
  return kernel;
}

inline PreparedTransferKernel prepare_conservative_polynomial5_coarse_fine() {
  PreparedTransferKernel kernel;
  kernel.capabilities = PreparedTransferCapabilities{5, {3}};
  kernel.prepared_coarse_fine = std::make_shared<const PreparedCoarseFineOperator>(
      prepare_polynomial5_coarse_fine_operator());
  kernel.coarse_fine = [](const MultiFab& parent, MultiFab& fine,
                          const SpatialTransferContext& context) {
    if (context.index.refinement_ratio != std::vector<int>{2, 2})
      throw std::runtime_error("degree-four coarse/fine spatial transfer ratio mismatch");
    const Box2D coarse = context.logical_coarse_domain;
    if (context.logical_fine_domain != coarse.refine(2))
      throw std::runtime_error(
          "degree-four coarse/fine spatial transfer logical-domain mismatch");
    if (context.index.coarse_origin != std::vector<int>{coarse.lo[0], coarse.lo[1]} ||
        context.index.fine_origin != std::vector<int>{2 * coarse.lo[0], 2 * coarse.lo[1]})
      throw std::runtime_error("degree-four coarse/fine spatial transfer origin mismatch");
    detail::coupler_conservative_polynomial5_fill_ghosts_mb(
        parent, fine, context.logical_coarse_domain, context.logical_fine_domain,
        context.replicated_parent, context.periodicity);
  };
  return kernel;
}

inline PreparedTransferKernel prepare_linear_time_interpolation() {
  PreparedTransferKernel kernel;
  kernel.capabilities = PreparedTransferCapabilities{2, {0}};
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
