#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/amr/hierarchy/refinement_ratio.hpp>
#include <pops/coupling/amr/amr_diagnostics.hpp>     // amr_mass, amr_max_drift_speed
#include <pops/coupling/amr/amr_level_storage.hpp>   // AmrLevelStack
#include <pops/coupling/amr/amr_regrid_coupler.hpp>  // amr_regrid_finest (Berger-Rigoutsos)
#include <pops/coupling/single/coupler.hpp>  // detail::coupler_eval_rhs (f = model.elliptic_rhs(U))
#include <pops/coupling/base/aux_fill.hpp>
#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>  // COMPOSITE FAC 2-level Poisson solver (opt-in)
#include <pops/numerics/elliptic/interface/elliptic_solver.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/time/amr/reflux/amr_reflux_mf.hpp>  // AmrLevelMP, amr_step_multilevel_multipatch, mf_*_mb
#include <pops/numerics/spatial/primitives/wave_speed.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/layout/refinement.hpp>  // coarsen_index
#include <pops/parallel/comm.hpp>  // all_reduce_sum / all_reduce_max (distributed mass/drift)
#include <pops/parallel/prepared_load_balance.hpp>
#include <pops/runtime/config/runtime_params.hpp>  // RuntimeParams: NATIVE per-block runtime params (ADC-514)
#include <pops/runtime/output_piece.hpp>

#include <algorithm>   // std::max
#include <cstddef>     // std::size_t
#include <functional>  // std::function (conducting-wall predicate passed to the MG)
#include <map>  // named_aux_: model-named aux fields (comp -> coarse field), re-applied by compute_aux
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>  // std::runtime_error (density size guard)
#include <type_traits>  // std::void_t / if constexpr detection of a brick's runtime-param member (ADC-514)
#include <utility>  // std::pair, std::move
#include <vector>

/// @file
/// @brief AmrCouplerMP: MULTI-PATCH E x B AMR coupler (coarse Poisson -> aux = grad phi ->
///        fine injection -> conservative AMR step), multi-box per-level hierarchy.
///
/// Same role as AmrCoupler but each level is multi-box (std::vector<AmrLevelMP> held by an
/// AmrLevelStack) and integration goes through amr_step_multilevel_multipatch (coverage-aware reflux).
/// regrid() rebuilds the fine level on the fly via Berger-Rigoutsos. Level 0 = single box for the
/// Poisson. The class only ORDERS the operations (hierarchy stored in AmrLevelStack,
/// regrid in amr_regrid_coupler.hpp, diagnostics in amr_diagnostics.hpp). INVARIANT: reduces
/// BIT FOR BIT to AmrCoupler when each level has a single box (validation guard). Level-0
/// ownership policy via replicated_coarse (replicated vs distributed, equivalence proven bit for bit).
/// The detail:: are the DISTRIBUTED primitives (aux injection, density write/read, layout).

namespace pops {

namespace detail {
inline void coupler_conservative_linear_to_fine_mb(
    const MultiFab& coarse, MultiFab& fine, const Box2D& logical_coarse_domain,
    const Box2D& logical_fine_domain, const std::vector<int>& coarse_origin,
    const std::vector<int>& fine_origin, const std::vector<int>& refinement_ratio,
    bool replicated_parent = false, Periodicity periodicity = {});
inline void coupler_conservative_linear_fill_ghosts_mb(const MultiFab& coarse, MultiFab& fine,
                                                       const Box2D& logical_coarse_domain,
                                                       const Box2D& logical_fine_domain,
                                                       bool replicated_parent,
                                                       Periodicity periodicity = {});
inline void coupler_conservative_linear_fill_all_mb(const MultiFab& coarse, MultiFab& fine,
                                                     const Box2D& logical_coarse_domain,
                                                     const Box2D& logical_fine_domain,
                                                     bool replicated_parent,
                                                     Periodicity periodicity = {});
inline void coupler_conservative_polynomial5_to_fine_mb(
    const MultiFab& coarse, MultiFab& fine, const Box2D& logical_coarse_domain,
    const Box2D& logical_fine_domain, const std::vector<int>& coarse_origin,
    const std::vector<int>& fine_origin, const std::vector<int>& refinement_ratio,
    bool replicated_parent = false, Periodicity periodicity = {});
inline void coupler_conservative_polynomial5_fill_ghosts_mb(
    const MultiFab& coarse, MultiFab& fine, const Box2D& logical_coarse_domain,
    const Box2D& logical_fine_domain, bool replicated_parent,
    Periodicity periodicity = {});

// Cell-centred aux publication uses the same certified conservative-linear spatial provider as
// state prolongation.  This covers phi, gradients and named aux uniformly; valid cells and every
// parent-supported ghost are materialized, with no piecewise-constant lower route.
inline void coupler_inject_aux_mb(const MultiFab& parent, MultiFab& child,
                                  const Box2D& logical_parent_domain,
                                  const Box2D& logical_child_domain,
                                  bool replicated_parent = true, Periodicity periodicity = {}) {
  if (parent.ncomp() != child.ncomp())
    throw std::runtime_error("aux conservative-linear prolongation component mismatch");
  coupler_conservative_linear_fill_all_mb(parent, child, logical_parent_domain,
                                           logical_child_domain, replicated_parent, periodicity);
}
// Writes an initial density (component 0, ny*nx row-major in GLOBAL indices) on the coarse
// level, MULTI-BOX and DISTRIBUTION-AWARE: each rank touches only its LOCAL fabs and reads
// rho at the cell GLOBAL index (i,j). For Euler (ncomp 4) it also sets zero momentum
// + thermal energy r/(gamma-1); ncomp 1 touches only density. Replicated mono-box:
// a single fab covering the domain, global == local indices -> bit-identical to the historical
// direct write. Distributed multi-box: each local box reads its window of rho.
inline void coupler_write_coarse(MultiFab& U, const std::vector<double>& rho, int nx, int ny,
                                 int ncomp, double gamma) {
  const std::size_t cells = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
  if (rho.size() != cells)
    throw std::runtime_error("AMR coupler: initial density of size != ny*nx");
  const Box2D logical_domain = U.box_array().bounding_box();
  if (logical_domain.nx() != nx || logical_domain.ny() != ny)
    throw std::runtime_error("AMR coupler: density shape disagrees with coarse logical domain");
  const Real gm1 = Real(gamma) - Real(1);
  // One-time host initialization from caller-owned contiguous storage. This is an explicit
  // host/device data boundary, not a time-stepping kernel.
  U.sync_host();
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 u = U.fab(li).array();
    const Box2D v = U.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const Real r = rho[static_cast<std::size_t>(j - logical_domain.lo[1]) * nx +
                           static_cast<std::size_t>(i - logical_domain.lo[0])];
        u(i, j, 0) = r;
        if (ncomp >= 3) {
          u(i, j, 1) = 0;
          u(i, j, 2) = 0;
        }
        if (ncomp == 4)
          u(i, j, 3) = r / gm1;
      }
  }
  U.sync_device();
}

inline void coupler_write_coarse(MultiFab& U, const std::vector<double>& rho, int n, int ncomp,
                                 double gamma) {
  coupler_write_coarse(U, rho, n, n, ncomp, gamma);
}

// Writes the FULL INITIAL CONSERVATIVE STATE (all components) on the coarse level from a
// flat component-major field @p state (c*ny*nx + j*nx + i), of size ncomp*ny*nx. Counterpart of
// coupler_write_coarse for the multi-component seed: same box traversal (replicated mono-box
// AND distributed multi-box, GLOBAL indices (i,j)), only the per-cell write differs -- here we copy
// the ncomp components positionally (no density/momentum/energy wiring; the caller already provides
// the conservative, e.g. [rho, rho*u, rho*v]). gamma omitted (no energy derived). Index computed
// in std::size_t (no int overflow at large n, unlike the int validation of
// coupler_write_coarse). Used for the drift seed (set_conservative_state).
inline void coupler_write_coarse_state(MultiFab& U, const std::vector<double>& state, int nx,
                                       int ny, int ncomp) {
  const std::size_t cells = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
  if (state.size() != cells * static_cast<std::size_t>(ncomp))
    throw std::runtime_error(
        "AMR coupler: initial state of size != ncomp*ny*nx (full conservative "
        "state; ncomp == model n_vars)");
  const Box2D logical_domain = U.box_array().bounding_box();
  if (logical_domain.nx() != nx || logical_domain.ny() != ny)
    throw std::runtime_error("AMR coupler: state shape disagrees with coarse logical domain");
  // One-time host initialization from caller-owned contiguous storage.
  U.sync_host();
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 u = U.fab(li).array();
    const Box2D v = U.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        for (int c = 0; c < ncomp; ++c)
          u(i, j, c) = state[static_cast<std::size_t>(c) * cells +
                             static_cast<std::size_t>(j - logical_domain.lo[1]) *
                                 static_cast<std::size_t>(nx) +
                             static_cast<std::size_t>(i - logical_domain.lo[0])];
  }
  U.sync_device();
}


inline void coupler_write_coarse_state(MultiFab& U, const std::vector<double>& state, int n,
                                       int ncomp) {
  coupler_write_coarse_state(U, state, n, n, ncomp);
}

// Reads the coarse density (component 0) into a GLOBAL ny*nx row-major field, MULTI-BOX and
// DISTRIBUTION-AWARE. Each rank writes its local cells into a ny*nx buffer initialized to 0
// then, if distributed, all_reduce_sum_inplace recomposes the full field on ALL ranks (the
// boxes are disjoint -> the cross-rank sum reconstructs the field exactly). Replicated mono-box:
// a single fab covers everything, the buffer is already complete, all_reduce would be the identity
// -> we avoid it (bit-identical to the historical direct read fab(0)).
inline std::vector<double> coupler_read_coarse(const MultiFab& U, int nx, int ny,
                                               bool replicated) {
  // Explicit output packing: make the device result host-visible once before the linear pack.
  U.sync_host();
  const Box2D logical_domain = U.box_array().bounding_box();
  if (logical_domain.nx() != nx || logical_domain.ny() != ny)
    throw std::runtime_error("AMR coarse read shape disagrees with logical domain");
  std::vector<double> out(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny), 0.0);
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const Box2D v = U.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        out[static_cast<std::size_t>(j - logical_domain.lo[1]) * nx +
            static_cast<std::size_t>(i - logical_domain.lo[0])] = u(i, j, 0);
  }
  if (!replicated)
    all_reduce_sum_inplace(out.data(), out.size());
  return out;
}

inline std::vector<double> coupler_read_coarse(const MultiFab& U, int n, bool replicated) {
  return coupler_read_coarse(U, n, n, replicated);
}

// Reads the coarse-level potential phi (component 0 of aux(0), written by compute_aux after the
// Poisson solve) into a GLOBAL ny*nx row-major field, MULTI-BOX and DISTRIBUTION-AWARE. aux(0) shares
// EXACTLY the layout of the coarse U (same BoxArray + DistributionMapping, cf. amr_level_storage:
// aux_[0] is built on U.box_array()/U.dmap()), so the recomposition is identical to
// coupler_read_coarse: local ny*nx buffer, all_reduce_sum if distributed (disjoint boxes -> exact
// sum), avoided in replicated mono-box (field already complete). PRECONDITION: update()/compute_aux
// has run at least once (otherwise aux(0) is 0). Strict counterpart of coupler_read_coarse for phi.
inline std::vector<double> coupler_read_coarse_phi(const MultiFab& aux0, int nx, int ny,
                                                   bool replicated) {
  // Explicit output packing, outside the numerical hot path.
  aux0.sync_host();
  const Box2D logical_domain = aux0.box_array().bounding_box();
  if (logical_domain.nx() != nx || logical_domain.ny() != ny)
    throw std::runtime_error("AMR coarse potential shape disagrees with logical domain");
  std::vector<double> out(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny), 0.0);
  for (int li = 0; li < aux0.local_size(); ++li) {
    const ConstArray4 a = aux0.fab(li).const_array();
    const Box2D v = aux0.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        out[static_cast<std::size_t>(j - logical_domain.lo[1]) * nx +
            static_cast<std::size_t>(i - logical_domain.lo[0])] = a(i, j, 0);
  }
  if (!replicated)
    all_reduce_sum_inplace(out.data(), out.size());
  return out;
}

inline std::vector<double> coupler_read_coarse_phi(const MultiFab& aux0, int n,
                                                   bool replicated) {
  return coupler_read_coarse_phi(aux0, n, n, replicated);
}

// Prolongs the coarse state into the valid cells of a fine patch (conservative linear, ratio 2),
// MULTI-BOX and DISTRIBUTION-AWARE. Makes the hierarchy consistent before the first sync_down (the
// seed patch is at 0). Replicated mono-box: coarse fully local, direct read via
// mf_find_box (always found); no collective -> bit-identical to the historical fab(0).
// Distributed multi-box: we bring the needed coarse regions onto a LOCAL child-coarsen grid
// via parallel_copy (same scheme as coupler_inject_aux_mb), then reconstruct.
inline void coupler_inject_coarse_to_fine_mb(const MultiFab& Uc, MultiFab& Uf,
                                             const Box2D& coarse_domain,
                                             const Box2D& fine_domain, bool replicated,
                                             Periodicity periodicity = {}) {
  if (Uc.ncomp() != Uf.ncomp())
    throw std::runtime_error("initial conservative-linear prolongation component mismatch");
  coupler_conservative_linear_to_fine_mb(
      Uc, Uf, coarse_domain, fine_domain, {coarse_domain.lo[0], coarse_domain.lo[1]},
      {fine_domain.lo[0], fine_domain.lo[1]}, {kAmrRefRatio, kAmrRefRatio}, replicated,
      periodicity);
}

enum class ConservativeCellFillRegion : unsigned char { Valid, Ghost, ValidAndGhost };

/// Persistent parent carrier for one exact conservative-linear transfer topology. Preparation
/// allocates the carrier and periodic image catalogue once and warms the redistribution schedule;
/// stable apply() calls only refresh values and launch reconstruction kernels. The workspace is
/// owned by the coupler/runtime that owns the hierarchy, never by a process-global cache.
class PreparedConservativeCellTransferWorkspace {
 public:
  PreparedConservativeCellTransferWorkspace(
      const PreparedConservativeCellTransferWorkspace&) = delete;
  PreparedConservativeCellTransferWorkspace& operator=(
      const PreparedConservativeCellTransferWorkspace&) = delete;
  PreparedConservativeCellTransferWorkspace(
      PreparedConservativeCellTransferWorkspace&&) noexcept = default;
  PreparedConservativeCellTransferWorkspace& operator=(
      PreparedConservativeCellTransferWorkspace&&) noexcept = default;

  static PreparedConservativeCellTransferWorkspace prepare(
      const MultiFab& coarse, const MultiFab& fine, const Box2D& coarse_domain,
      const Box2D& fine_domain, bool replicated_parent, ConservativeCellFillRegion region,
      Periodicity periodicity, std::uint64_t topology_generation,
      const CommunicatorView& communicator) {
    return prepare(coarse, fine, coarse_domain, fine_domain, replicated_parent, region,
                   periodicity, topology_generation, communicator,
                   std::make_shared<const PreparedCoarseFineOperator>(
                       prepare_limited_linear_coarse_fine_operator()));
  }

  static PreparedConservativeCellTransferWorkspace prepare(
      const MultiFab& coarse, const MultiFab& fine, const Box2D& coarse_domain,
      const Box2D& fine_domain, bool replicated_parent, ConservativeCellFillRegion region,
      Periodicity periodicity, std::uint64_t topology_generation,
      const CommunicatorView& communicator,
      std::shared_ptr<const PreparedCoarseFineOperator> prepared_operator) {
    if (fine.ncomp() != coarse.ncomp())
      throw std::invalid_argument(
          "prepared conservative cell transfer component mismatch");
    if (coarse_domain.empty() || fine_domain != coarse_domain.refine(kAmrRefRatio))
      throw std::invalid_argument(
          "prepared conservative cell transfer has inconsistent logical domains");
    if (!prepared_operator)
      throw std::invalid_argument("prepared conservative cell transfer lacks its operator");
    prepared_operator->validate_domain(coarse_domain);
    if (replicated_parent && communicator.active())
      throw std::invalid_argument(
          "replicated conservative parent requires a rank-local communicator");

    const bool includes_ghosts = region != ConservativeCellFillRegion::Valid;
    // BoxArray growth is currently isotropic.  Grow by the larger directional reach (safe
    // over-allocation for an anisotropic provider), while retaining directional validation below.
    const int reach =
        std::max(prepared_operator->parent_reach_x, prepared_operator->parent_reach_y);
    const int fine_growth = checked_coarse_fine_carrier_growth(
        includes_ghosts ? fine.n_grow() : 0, kAmrRefRatio, reach);
    const BoxArray parent_boxes =
        coarsen_grown(fine.box_array(), fine_growth, kAmrRefRatio);
    for (const Box2D& box : parent_boxes.boxes())
      if (box.nx() < prepared_operator->minimum_axis_cells_x ||
          box.ny() < prepared_operator->minimum_axis_cells_y)
        throw std::invalid_argument(
            "prepared conservative carrier cannot hold the selected directional stencil");
    DistributionMapping parent_mapping = fine.dmap();
    if (replicated_parent)
      parent_mapping = DistributionMapping(
          std::vector<int>(static_cast<std::size_t>(parent_boxes.size()), my_rank()));

    PreparedConservativeCellTransferWorkspace workspace(
        MultiFab(parent_boxes, parent_mapping, coarse.ncomp(), 0), fine, coarse_domain,
        fine_domain, replicated_parent, region, periodicity, topology_generation,
        std::move(prepared_operator));
    workspace.copy_plan_.emplace(PreparedPeriodicCopyPlan::prepare(
        workspace.parent_carrier_, coarse, coarse_domain, periodicity, topology_generation,
        communicator));
    return workspace;
  }

  void apply(const MultiFab& coarse, MultiFab& fine, std::uint64_t topology_generation,
             const CommunicatorView& communicator) {
    validate_(coarse, fine, topology_generation);
    copy_plan_->apply(parent_carrier_, coarse, topology_generation, communicator);
    fill_components_(fine, {});
  }

  void apply(const MultiFab& coarse, MultiFab& fine, std::span<const int> components,
             std::uint64_t topology_generation, const CommunicatorView& communicator) {
    validate_(coarse, fine, topology_generation);
    for (const int component : components)
      if (component < 0 || component >= fine.ncomp())
        throw std::out_of_range(
            "prepared conservative-linear selected component is out of range");
    copy_plan_->apply(parent_carrier_, coarse, topology_generation, communicator);
    fill_components_(fine, components);
  }

  /// Publishes the source snapshot already materialized by prepare(). Intended for one-shot setup
  /// routes; persistent owners call apply() for every later state.
  void publish_prepared(MultiFab& fine) {
    validate_fine_(fine, topology_generation_);
    fill_components_(fine, {});
  }

  [[nodiscard]] const std::shared_ptr<const PreparedCoarseFineOperator>& prepared_operator()
      const noexcept {
    return prepared_operator_;
  }

 private:
  PreparedConservativeCellTransferWorkspace(
      MultiFab parent_carrier, const MultiFab& fine, Box2D coarse_domain, Box2D fine_domain,
      bool replicated_parent, ConservativeCellFillRegion region, Periodicity periodicity,
      std::uint64_t topology_generation,
      std::shared_ptr<const PreparedCoarseFineOperator> prepared_operator)
      : parent_carrier_(std::move(parent_carrier)),
        fine_boxes_(fine.box_array().boxes()),
        fine_ranks_(fine.dmap().ranks()),
        fine_ncomp_(fine.ncomp()),
        fine_ngrow_(fine.n_grow()),
        coarse_domain_(coarse_domain),
        fine_domain_(fine_domain),
        transform_{coarse_domain.lo[0], coarse_domain.lo[1], fine_domain.lo[0],
                   fine_domain.lo[1], kAmrRefRatio, kAmrRefRatio},
        replicated_parent_(replicated_parent),
        region_(region),
        periodicity_(periodicity),
        topology_generation_(topology_generation),
        prepared_operator_(std::move(prepared_operator)) {}

  void validate_fine_(const MultiFab& fine, std::uint64_t topology_generation) const {
    if (fine.box_array().boxes() != fine_boxes_ || fine.dmap().ranks() != fine_ranks_ ||
        fine.ncomp() != fine_ncomp_ || fine.n_grow() != fine_ngrow_)
      throw std::invalid_argument(
          "prepared conservative cell transfer crossed an exact fine layout");
    if (topology_generation != topology_generation_)
      throw std::invalid_argument(
          "prepared conservative cell transfer crossed a topology generation");
  }

  void validate_(const MultiFab& coarse, const MultiFab& fine,
                 std::uint64_t topology_generation) const {
    validate_fine_(fine, topology_generation);
    if (!copy_plan_ || coarse.ncomp() != fine_ncomp_)
      throw std::invalid_argument(
          "prepared conservative cell transfer has incompatible parent storage");
  }

  void fill_components_(MultiFab& fine, std::span<const int> selected) {
    const bool includes_ghosts = region_ != ConservativeCellFillRegion::Valid;
    const bool fill_valid = region_ != ConservativeCellFillRegion::Ghost;
    const bool fill_ghost = region_ != ConservativeCellFillRegion::Valid;
    for (int local_fine = 0; local_fine < fine.local_size(); ++local_fine) {
      const int local_parent_index =
          replicated_parent_ ? fine.global_index(local_fine) : local_fine;
      Array4 destination = fine.fab(local_fine).array();
      const ConstArray4 source = parent_carrier_.fab(local_parent_index).const_array();
      const Box2D valid = fine.box(local_fine);
      const Box2D target = includes_ghosts ? fine.fab(local_fine).grown_box() : valid;
      const auto launch = [&](int component) {
        prepared_operator_->launch_spatial(destination, source, target, valid, coarse_domain_,
                                           fine_domain_, transform_, component, fill_valid,
                                           fill_ghost, periodicity_);
      };
      if (selected.empty()) {
        for (int component = 0; component < fine.ncomp(); ++component)
          launch(component);
      } else {
        for (const int component : selected)
          launch(component);
      }
    }
    // The persistent carrier may be refreshed by the next apply, so all reconstruction reads must
    // be complete before returning it to the owner.
    device_fence();
  }

  MultiFab parent_carrier_;
  std::optional<PreparedPeriodicCopyPlan> copy_plan_;
  std::vector<Box2D> fine_boxes_;
  std::vector<int> fine_ranks_;
  int fine_ncomp_ = 0;
  int fine_ngrow_ = 0;
  Box2D coarse_domain_{};
  Box2D fine_domain_{};
  PreparedCoarseFineTransform2D transform_{};
  bool replicated_parent_ = false;
  ConservativeCellFillRegion region_ = ConservativeCellFillRegion::Valid;
  Periodicity periodicity_{};
  std::uint64_t topology_generation_ = 0;
  std::shared_ptr<const PreparedCoarseFineOperator> prepared_operator_;
};

using PreparedConservativeLinearTransferWorkspace =
    PreparedConservativeCellTransferWorkspace;

/// Materialize a selected region through one migrated parent carrier.  Valid+ghost publication is
/// intentionally a single pass: aux propagation runs every field update and must not pay for two
/// parallel_copy schedules, two carrier allocations and four device fences per level.
inline void coupler_conservative_linear_fill_region_mb(
    const MultiFab& coarse, MultiFab& fine, const Box2D& coarse_domain,
    const Box2D& fine_domain, bool replicated_parent,
    ConservativeCellFillRegion region, Periodicity periodicity) {
  const CommunicatorView communicator =
      replicated_parent ? CommunicatorView{} : world_communicator_view();
  auto workspace = PreparedConservativeCellTransferWorkspace::prepare(
      coarse, fine, coarse_domain, fine_domain, replicated_parent, region, periodicity,
      /*topology_generation=*/0, communicator);
  workspace.publish_prepared(fine);
}

/// Ratio-2 conservative piecewise-linear prolongation. The four fine children average exactly to
/// the parent value; minmod-limited slopes make the operator monotone. Parent regions are first
/// migrated onto the child DistributionMapping, so the per-patch kernel is MPI/GPU-safe.
inline void coupler_conservative_linear_to_fine_mb(const MultiFab& coarse, MultiFab& fine,
                                                   const Box2D& coarse_domain,
                                                   const Box2D& fine_domain,
                                                   const std::vector<int>& coarse_origin,
                                                   const std::vector<int>& fine_origin,
                                                   const std::vector<int>& refinement_ratio,
                                                   bool replicated_parent,
                                                   Periodicity periodicity) {
  if (fine.ncomp() != coarse.ncomp())
    throw std::runtime_error("conservative-linear prolongation component mismatch");
  if (coarse_origin.size() != 2 || fine_origin.size() != 2 ||
      refinement_ratio != std::vector<int>{2, 2})
    throw std::runtime_error(
        "conservative-linear prolongation received an invalid index transform");
  if (coarse_domain.empty() || fine_domain != coarse_domain.refine(refinement_ratio[0]))
    throw std::runtime_error("conservative-linear prolongation logical-domain mismatch");
  if (coarse_domain.lo[0] != coarse_origin[0] || coarse_domain.lo[1] != coarse_origin[1] ||
      fine_domain.lo[0] != fine_origin[0] || fine_domain.lo[1] != fine_origin[1])
    throw std::runtime_error("conservative-linear prolongation index origin mismatch");
  coupler_conservative_linear_fill_region_mb(
      coarse, fine, coarse_domain, fine_domain, replicated_parent,
      ConservativeCellFillRegion::Valid, periodicity);
}

/// Ratio-2 conservative piecewise-linear coarse/fine ghost production. The provider fills every
/// requested fine ghost layer that is supported by the parent level and leaves physical ghosts to
/// the boundary authority. Parent data is migrated onto one grown carrier per fine patch; the
/// replicated-parent branch deliberately makes that carrier rank-local to avoid asymmetric MPI
/// schedules from replicated coarse ownership metadata.
inline void coupler_conservative_linear_fill_ghosts_mb(const MultiFab& coarse, MultiFab& fine,
                                                       const Box2D& coarse_domain,
                                                       const Box2D& fine_domain,
                                                       bool replicated_parent,
                                                       Periodicity periodicity) {
  coupler_conservative_linear_fill_region_mb(
      coarse, fine, coarse_domain, fine_domain, replicated_parent,
      ConservativeCellFillRegion::Ghost, periodicity);
}

inline void coupler_conservative_linear_fill_all_mb(const MultiFab& coarse, MultiFab& fine,
                                                     const Box2D& coarse_domain,
                                                     const Box2D& fine_domain,
                                                     bool replicated_parent,
                                                     Periodicity periodicity) {
  coupler_conservative_linear_fill_region_mb(
      coarse, fine, coarse_domain, fine_domain, replicated_parent,
      ConservativeCellFillRegion::ValidAndGhost, periodicity);
}

inline void coupler_conservative_polynomial5_fill_region_mb(
    const MultiFab& coarse, MultiFab& fine, const Box2D& coarse_domain,
    const Box2D& fine_domain, bool replicated_parent,
    ConservativeCellFillRegion region, Periodicity periodicity) {
  const CommunicatorView communicator =
      replicated_parent ? CommunicatorView{} : world_communicator_view();
  auto workspace = PreparedConservativeCellTransferWorkspace::prepare(
      coarse, fine, coarse_domain, fine_domain, replicated_parent, region, periodicity,
      /*topology_generation=*/0, communicator,
      std::make_shared<const PreparedCoarseFineOperator>(
          prepare_polynomial5_coarse_fine_operator()));
  workspace.publish_prepared(fine);
}

inline void coupler_conservative_polynomial5_to_fine_mb(
    const MultiFab& coarse, MultiFab& fine, const Box2D& coarse_domain,
    const Box2D& fine_domain, const std::vector<int>& coarse_origin,
    const std::vector<int>& fine_origin, const std::vector<int>& refinement_ratio,
    bool replicated_parent, Periodicity periodicity) {
  if (fine.ncomp() != coarse.ncomp())
    throw std::runtime_error("degree-four conservative prolongation component mismatch");
  if (coarse_origin.size() != 2 || fine_origin.size() != 2 ||
      refinement_ratio != std::vector<int>{2, 2})
    throw std::runtime_error(
        "degree-four conservative prolongation received an invalid index transform");
  if (coarse_domain.empty() || fine_domain != coarse_domain.refine(2) ||
      coarse_origin != std::vector<int>{coarse_domain.lo[0], coarse_domain.lo[1]} ||
      fine_origin != std::vector<int>{fine_domain.lo[0], fine_domain.lo[1]})
    throw std::runtime_error("degree-four conservative prolongation logical-domain mismatch");
  coupler_conservative_polynomial5_fill_region_mb(
      coarse, fine, coarse_domain, fine_domain, replicated_parent,
      ConservativeCellFillRegion::Valid, periodicity);
}

inline void coupler_conservative_polynomial5_fill_ghosts_mb(
    const MultiFab& coarse, MultiFab& fine, const Box2D& coarse_domain,
    const Box2D& fine_domain, bool replicated_parent, Periodicity periodicity) {
  coupler_conservative_polynomial5_fill_region_mb(
      coarse, fine, coarse_domain, fine_domain, replicated_parent,
      ConservativeCellFillRegion::Ghost, periodicity);
}

// Builds the coarse level (BoxArray + DistributionMapping) of the AmrSystem path according to the
// ownership policy, in a SINGLE point for both build paths (native + compiled):
//  - replicated (distribute=false, DEFAULT): mono-box covering the domain, dmap = my_rank() everywhere
//    (the box lives on each rank). In serial my_rank()=0 -> identical to round-robin, bit for bit.
//    This is the layout GeometricMG(replicated=true) and the historical one expect.
//  - distributed (distribute=true): multi-box BoxArray::from_domain(dom, max_grid) assigned by the
//    same prepared ownership authority used for fine seeds and every regrid. Each rank carries only
//    its tiles -> the coarse Poisson and coarse transport distribute (strong-scaling).
inline std::pair<BoxArray, DistributionMapping> coupler_make_coarse_layout(
    int nx, int ny, bool distribute, int max_grid,
    const PreparedLoadBalanceAuthority& load_balance) {
  const Box2D dom = Box2D::from_extents(nx, ny);
  if (!distribute) {
    BoxArray ba(std::vector<Box2D>{dom});
    return {ba, DistributionMapping(std::vector<int>{my_rank()})};
  }
  const int mgx = (max_grid > 0) ? max_grid : std::max(1, nx / 2);
  const int mgy = (max_grid > 0) ? max_grid : std::max(1, ny / 2);
  BoxArray ba = BoxArray::from_domain(dom, mgx, mgy);
  return {ba, load_balance.distribute(ba, n_ranks())};
}

inline std::pair<BoxArray, DistributionMapping> coupler_make_coarse_layout(
    int n, bool distribute, int max_grid,
    const PreparedLoadBalanceAuthority& load_balance) {
  return coupler_make_coarse_layout(n, n, distribute, max_grid, load_balance);
}

inline DistributionMapping coupler_authoritative_coarse_mapping(
    const BoxArray& coarse_boxes, const std::vector<AmrLevelMP>& levels) {
  if (levels.empty())
    throw std::invalid_argument("AmrCouplerMP requires a coarse level");
  const MultiFab& coarse = levels.front().U;
  if (coarse.box_array().boxes() != coarse_boxes.boxes())
    throw std::invalid_argument(
        "AmrCouplerMP coarse BoxArray disagrees with the authoritative level field");
  return coarse.dmap();
}

inline Geometry coupler_validated_geometry(Geometry geometry) {
  require_positive_finite_amr_spacing(geometry.dx(), geometry.dy());
  return geometry;
}

}  // namespace detail

/// Multi-patch E x B AMR coupler. @tparam Model: PhysicalModel (flux, source, elliptic_rhs,
/// max_wave_speed). @tparam Elliptic: elliptic backend (EllipticSolver concept, default GeometricMG).
/// ORCHESTRATES only: the hierarchy lives in an AmrLevelStack<AmrLevelMP>, the Poisson solve in
/// mg_, the regrid in amr_regrid_finest. Reduces bit for bit to AmrCoupler in mono-box per level.
template <class Model, class Elliptic = GeometricMG>
class AmrCouplerMP {
  static_assert(EllipticSolver<Elliptic>, "Elliptic must model EllipticSolver");

 public:
  // active: optional "active cell" predicate (interior of the conductor), for the circular
  // conducting wall of the column instability (passed as-is to the multigrid). Empty
  // by default -> no wall (historical behavior unchanged). Only the coarse carries the
  // wall: the fine patches refine the ring edge, strictly inside the wall.
  // replicated_coarse: level-0 (coarse) OWNERSHIP POLICY. BOTH modes are
  // stable and their equivalence is proven bit for bit (test_mpi_decoarse, maxdiff=0):
  //   true  (performant DEFAULT): coarse mono-box REPLICATED on all ranks. Best coarse
  //          MG solve (no multigrid degeneration), zero communication for the
  //          coarse Poisson, robust reference -> the right default for small/medium cases.
  //   false (EXPLICIT scalable mode): coarse multi-box assigned by the prepared authority. Lifts the
  //          O(NX*NY*nranks) memory lock of level 0, required at very large scale. But the
  //          geometric MG degenerates for a finely-split coarse (>2x2 boxes do not tile the
  //          coarsest grid): reserve for cases where the level-0 memory is the lock.
  // Criterion: set false ONLY when memory scalability requires it; otherwise keep true.
  // Removing the replicated path is DEFERRED as long as the distributed one is not strictly
  // superior. mg_ receives the same flag (otherwise, under replicated MPI, the coarse would fall on
  // the single rank 0 and compute_aux would read a phi absent elsewhere). In serial, both coincide.
  template <class FactoryT = DefaultEllipticFactory<Elliptic>>
    requires pops::EllipticFactory<FactoryT, Elliptic>
  AmrCouplerMP(const Model& model, const Geometry& geom, const BoxArray& ba_coarse, const BCRec& bc,
               std::vector<AmrLevelMP> levels, ActiveRegionProvider2D active,
               bool replicated_coarse,
               std::shared_ptr<const PreparedLoadBalanceAuthority> load_balance,
               FactoryT elliptic_factory = {})
      : model_(model),
        geom_(detail::coupler_validated_geometry(geom)),
        coarse_boxes_(ba_coarse),
        coarse_mapping_(detail::coupler_authoritative_coarse_mapping(ba_coarse, levels)),
        elliptic_bc_(bc),
        mg_(make_elliptic_solver<Elliptic>(
            {geom_, coarse_boxes_, coarse_mapping_, elliptic_bc_, std::move(active),
             replicated_coarse ? FieldDistribution::Replicated : FieldDistribution::Distributed},
            std::move(elliptic_factory))),
        stack_(geom_.domain, std::move(levels), aux_comps<Model>()),
        replicated_coarse_(replicated_coarse),
        load_balance_authority_(std::move(load_balance)) {
    if (!load_balance_authority_)
      throw std::invalid_argument("AmrCouplerMP requires a prepared load-balance authority");
    for (const AmrLevelMP& level : stack_.levels())
      detail::require_positive_finite_amr_spacing(level.dx, level.dy);
    prepare_aux_transfer_workspaces_();
  }

  std::vector<AmrLevelMP>& levels() { return stack_.levels(); }
  MultiFab& coarse() { return stack_.coarse(); }
  const MultiFab& coarse() const { return stack_.coarse(); }
  // coarse-level aux: (phi, dphi/dx, dphi/dy), component 0 = phi (cf. compute_aux). Same
  // layout as coarse(). Read by the AmrSystem potential hook (coupler_read_coarse_phi).
  MultiFab& aux0() { return stack_.aux(0); }
  const MultiFab& aux0() const { return stack_.aux(0); }

  /// Registers a model-NAMED aux field (ADC-291) at shared-channel component @p comp (>= kAuxNamedBase),
  /// as a coarse base-level field @p field (ny*nx row-major, global cell index j*nx+i). STATIC user field
  /// installed once on the coarse authority and injected coarse->fine on every update/regrid.
  /// Single-block AMR counterpart of System::set_aux_field_component. The facade validates comp/size and
  /// resolves the name. No-op default (no named field -> empty map -> bit-identical).
  void set_named_aux(int comp, std::vector<Real> field) {
    if (comp < kAuxNamedBase || comp >= stack_.aux(0).ncomp())
      throw std::out_of_range("AmrCouplerMP named aux component is outside the model channel");
    const Box2D logical_domain = stack_.domain();
    const std::size_t expected = static_cast<std::size_t>(logical_domain.nx()) *
                                 static_cast<std::size_t>(logical_domain.ny());
    if (field.size() != expected)
      throw std::invalid_argument("AmrCouplerMP named aux field shape disagrees with the domain");
    if (named_aux_.find(comp) == named_aux_.end()) {
      named_aux_components_.push_back(comp);
      std::sort(named_aux_components_.begin(), named_aux_components_.end());
    }
    named_aux_[comp] = std::move(field);
    apply_named_aux();  // stack_ exists at ctor: reflect onto the coarse aux right away
  }
  /// Registers a per-field aux HALO policy (ADC-369) for the named component @p comp. compute_aux
  /// applies it onto the COARSE aux AFTER the shared fill, overriding only that component's
  /// physical-face ghosts (periodic faces stay periodic). Single-block AMR counterpart of
  /// System::set_aux_field_halo_component. No-op default.
  void set_named_aux_bc(int comp, AuxHaloPolicy policy) {
    if (comp < kAuxNamedBase || comp >= stack_.aux(0).ncomp())
      throw std::out_of_range("AmrCouplerMP named aux halo component is outside the model channel");
    named_aux_bc_[comp] = policy;
  }
  const Box2D& domain() const { return stack_.domain(); }
  int nlev() const { return stack_.nlev(); }
  void set_transport_boundary_fill(AmrBoundaryFillAuthority authority) {
    validate_amr_boundary_fill_authority(authority.periodicity, &authority, stack_.L());
    transport_periodicity_ = authority.periodicity;
    transport_boundary_fill_ = std::move(authority);
    prepare_aux_transfer_workspaces_(next_transfer_topology_generation_());
  }

  // ----------------------------------------------------------------------------------------------
  // AMR ACCEPTED-STATE CHECKPOINT / RESTART. The mono-block coupler carries the FULL conservative
  // state per level (all components) plus phi (multigrid warm-start), and can impose a saved fine
  // hierarchy instead of reclustering tags. Local accessors preserve native patch ownership; their
  // explicit global counterparts perform the MPI gather used by the strict v3 checkpoint provider.
  // ----------------------------------------------------------------------------------------------

  // Reads the FULL conservative state (all components) of level @p k into a flat
  // component-major field over the exact rectangular level domain. The cells OUTSIDE
  // patches (uncovered fine level) stay at 0: a fine level is only defined within its patches
  // (at restart we rewrite ONLY the patch cells, cf. set_level_state).
  std::vector<double> level_state(int k) {
    std::vector<AmrLevelMP>& L = stack_.L();
    if (k < 0 || k >= static_cast<int>(L.size()))
      throw std::runtime_error("AmrCouplerMP::level_state: level out of bounds");
    MultiFab& U = L[k].U;
    const int nc = U.ncomp();
    const Box2D level_domain = amr_level_index_domain(stack_.domain(), k);
    const std::size_t nx = static_cast<std::size_t>(level_domain.nx());
    const std::size_t cells = nx * static_cast<std::size_t>(level_domain.ny());
    std::vector<double> out(static_cast<std::size_t>(nc) * cells, 0.0);
    // Checkpoint/output serialization is an explicit host packing boundary.
    U.sync_host();
    for (int li = 0; li < U.local_size(); ++li) {
      const ConstArray4 u = U.fab(li).const_array();
      const Box2D v = U.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          for (int c = 0; c < nc; ++c)
            out[static_cast<std::size_t>(c) * cells +
                static_cast<std::size_t>(j - level_domain.lo[1]) * nx +
                static_cast<std::size_t>(i - level_domain.lo[0])] = u(i, j, c);
    }
    return out;
  }

  // Restores the full conservative state of level @p k from @p s (same layout as level_state).
  // Writes ONLY the VALID cells of the local fabs (the patches): the ghosts are redone at the
  // next update()/advance (exactly like after a regrid), and a fine cell outside a patch
  // does not exist. NO RE-PROLONGATION: the state is restored AS-IS (no coarse->fine injection).
  void set_level_state(int k, const std::vector<double>& s) {
    std::vector<AmrLevelMP>& L = stack_.L();
    if (k < 0 || k >= static_cast<int>(L.size()))
      throw std::runtime_error("AmrCouplerMP::set_level_state: level out of bounds");
    MultiFab& U = L[k].U;
    const int nc = U.ncomp();
    const Box2D level_domain = amr_level_index_domain(stack_.domain(), k);
    const std::size_t nx = static_cast<std::size_t>(level_domain.nx());
    const std::size_t cells = nx * static_cast<std::size_t>(level_domain.ny());
    if (s.size() != static_cast<std::size_t>(nc) * cells)
      throw std::runtime_error(
          "AmrCouplerMP::set_level_state: state size differs from ncomp*level_cells");
    // Checkpoint restore is an explicit host unpacking boundary.
    U.sync_host();
    for (int li = 0; li < U.local_size(); ++li) {
      Array4 u = U.fab(li).array();
      const Box2D v = U.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          for (int c = 0; c < nc; ++c)
            u(i, j, c) = s[static_cast<std::size_t>(c) * cells +
                           static_cast<std::size_t>(j - level_domain.lo[1]) * nx +
                           static_cast<std::size_t>(i - level_domain.lo[0])];
    }
    U.sync_device();
  }

  // Reads the potential phi of level @p k, flat exact-domain row-major field, zeros outside patches.
  // Level 0: the multigrid WARM-START -- mg_.phi() (VALID cells), the state actually
  // reused by the NEXT solve (GeometricMG::solve keeps phi between calls). Level >= 1:
  // aux(k) component 0 (informational; recomputed at update). It is mg_.phi() level 0 that makes the
  // restart BIT-IDENTICAL (the 1st post-restart solve starts from the same guess as the continuous run).
  std::vector<double> level_potential(int k) {
    if (k < 0 || k >= stack_.nlev())
      throw std::runtime_error("AmrCouplerMP::level_potential: level out of bounds");
    const Box2D level_domain = amr_level_index_domain(stack_.domain(), k);
    const std::size_t nx = static_cast<std::size_t>(level_domain.nx());
    std::vector<double> out(nx * static_cast<std::size_t>(level_domain.ny()), 0.0);
    const MultiFab& P = (k == 0) ? mg_.phi() : stack_.aux(k);
    // Checkpoint/output serialization is an explicit host packing boundary.
    P.sync_host();
    for (int li = 0; li < P.local_size(); ++li) {
      const ConstArray4 p = P.fab(li).const_array();
      const Box2D v = P.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          out[static_cast<std::size_t>(j - level_domain.lo[1]) * nx +
              static_cast<std::size_t>(i - level_domain.lo[0])] = p(i, j, 0);
    }
    return out;
  }

  // Restores the potential of level @p k. Level 0: warm-start mg_.phi() (valid cells) -> the
  // multigrid restart is BIT-IDENTICAL (the 1st post-restart solve starts from the same guess). Level
  // >= 1: aux(k) comp 0 (recomputed at update; idempotent restore, no effect on the dynamics).
  void set_level_potential(int k, const std::vector<double>& p) {
    if (k < 0 || k >= stack_.nlev())
      throw std::runtime_error("AmrCouplerMP::set_level_potential: level out of bounds");
    const Box2D level_domain = amr_level_index_domain(stack_.domain(), k);
    const std::size_t nx = static_cast<std::size_t>(level_domain.nx());
    if (p.size() != nx * static_cast<std::size_t>(level_domain.ny()))
      throw std::runtime_error(
          "AmrCouplerMP::set_level_potential: phi size differs from level cell count");
    MultiFab& P = (k == 0) ? mg_.phi() : stack_.aux(k);
    // Warm-start restore is an explicit host unpacking boundary.
    P.sync_host();
    for (int li = 0; li < P.local_size(); ++li) {
      Array4 q = P.fab(li).array();
      const Box2D v = P.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          q(i, j, 0) = p[static_cast<std::size_t>(j - level_domain.lo[1]) * nx +
                         static_cast<std::size_t>(i - level_domain.lo[0])];
    }
    P.sync_device();
  }

  // GLOBAL variants of level_state / level_potential (ADC-509). Ownership-distributed levels contain
  // zeros outside the local fabs, so all_reduce_sum_inplace gathers the complete field on every rank
  // (AMR reflux pattern, comm.hpp; MIRROR of System::state_global / gather_global).  Replicated level
  // 0 is already complete on every rank and must not be reduced, which would multiply it by n_ranks.
  std::vector<double> level_state_global(int k) {
    std::vector<double> out = level_state(k);
    if (k > 0 || !replicated_coarse_)
      all_reduce_sum_inplace(out.data(), out.size());
    return out;
  }

  /// Exact valid-cell pieces allocated on this rank, including the complete replicated coarse fab
  /// when that ownership policy is active.  No level-sized staging buffer and no MPI collective.
  std::vector<PatchBox> output_geometry_boxes() const {
    std::vector<PatchBox> result;
    const std::vector<AmrLevelMP>& levels = stack_.levels();
    for (int level = 0; level < static_cast<int>(levels.size()); ++level) {
      const auto& boxes = levels[static_cast<std::size_t>(level)].U.box_array().boxes();
      for (const Box2D& box : boxes)
        result.push_back(PatchBox{level, box.lo[0], box.lo[1], box.hi[0], box.hi[1]});
    }
    return result;
  }

  std::vector<OutputPiece> output_state_local_pieces(int k) {
    std::vector<AmrLevelMP>& levels = stack_.L();
    if (k < 0 || k >= static_cast<int>(levels.size()))
      throw std::runtime_error("AmrCouplerMP::output_state_local_pieces: level out of bounds");
    return output_local_pieces(levels[static_cast<std::size_t>(k)].U, k,
                               k == 0 && replicated_coarse_);
  }

  std::vector<double> level_potential_global(int k) {
    std::vector<double> out = level_potential(k);
    if (k > 0 || !replicated_coarse_)
      all_reduce_sum_inplace(out.data(), out.size());
    return out;
  }

  // Imposes the fine-level hierarchy (restart): rebuilds level 1 on the SAVED @p
  // fine_boxes BoxArray (instead of Berger-Rigoutsos clustering on tags), via the SAME mechanism as
  // regrid (prepared parent interpolation + fine carry-over), then reattaches the level-1 aux.
  // The rebuilt valid content is OVERWRITTEN afterwards by set_level_state (restore as-is): here we
  // rely only on the IMPOSED LAYOUT. SINGLE-RANK, 2-level mono-block hierarchy (so we impose
  // ONLY level 1). Clear rejection if the hierarchy has no fine level or if no box was saved.
  void set_hierarchy(const std::vector<Box2D>& fine_boxes) {
    std::vector<AmrLevelMP>& L = stack_.L();
    if (L.size() < 2)
      throw std::runtime_error(
          "AmrCouplerMP::set_hierarchy: mono-level hierarchy (no fine patch "
          "to impose)");
    if (fine_boxes.empty())
      throw std::runtime_error(
          "AmrCouplerMP::set_hierarchy: no saved fine box (restart of a "
          "fine-patch hierarchy required)");
    const int ngf = L[1].U.n_grow();  // inherit the ghost width of the current fine (scheme parity)
    BoxArray fb(fine_boxes);
    DistributionMapping dmap = load_balance_authority_->distribute(fb, n_ranks());
    const RegridProlongation prolong = [base_domain = stack_.domain(),
                                        periodicity = transport_periodicity_](
                                          const MultiFab& parent, MultiFab& fine,
                                          int parent_level, int ratio,
                                          bool parent_replicated,
                                          const CommunicatorView&) {
      const Box2D coarse_domain = amr_level_index_domain(base_domain, parent_level);
      const Box2D fine_domain = coarse_domain.refine(ratio);
      detail::coupler_conservative_linear_to_fine_mb(
          parent, fine, coarse_domain, fine_domain,
          {coarse_domain.lo[0], coarse_domain.lo[1]},
          {fine_domain.lo[0], fine_domain.lo[1]}, {ratio, ratio}, parent_replicated,
          periodicity);
      (void)parent_level;
    };
    L[1].U = regrid_field_on_layout_with_provider(
        fb, dmap, L[0].U, L[1].U, /*pk=*/0, ngf, prolong, world_communicator_view(),
        replicated_coarse_);
    stack_.reattach_aux(1);  // realloc aux[1] on the new layout + rewire L[1].aux
    prepare_aux_transfer_workspaces_(next_transfer_topology_generation_());
  }

  void sync_down() {  // average fine -> coarse over the whole hierarchy (multi-box)
    auto& L = stack_.L();
    if (!average_down_plan_)
      throw std::logic_error("AmrCouplerMP average-down plan was not prepared");
    for (int k = stack_.nlev() - 1; k >= 1; --k)
      mf_average_down_mb(L[k].U, L[k - 1].U,
                         average_down_plan_->transition_for_child(k),
                         average_down_plan_->topology_generation(),
                         world_communicator_view());
  }

  /// OPT-IN: replaces the coarse-only AMR Poisson with a COMPOSITE FAC elliptic solve (the fine
  /// patch REFINES the elliptic). Cf. CompositeFacPoisson.
  /// Current certified scope: 2 levels, ONE interior mono-box fine patch and replicated coarse.
  /// Requests outside that scope are rejected; composite=true never falls back to a coarse-only solve.
  void set_composite_poisson(bool v) { composite_poisson_ = v; }
  bool composite_poisson() const { return composite_poisson_; }

  /// ADC-614: install the composite-FAC knobs applied to fac_ when the composite path builds it
  /// (compute_aux_composite). Defaults = kFAC* -> bit-identical. Applied to an already-built solver.
  void set_fac_options(const CompositeFacOptions& o) {
    fac_options_ = o;
    if (fac_)
      fac_->set_options(o);
  }

  void compute_aux() {  // coarse Poisson + grad phi + injection to the fine levels
    auto& L = stack_.L();
    const Box2D& dom = stack_.domain();
    detail::require_positive_finite_amr_spacing(geom_.dx(), geom_.dy());
    for (const AmrLevelMP& level : L)
      detail::require_positive_finite_amr_spacing(level.dx, level.dy);
    // COMPOSITE path (opt-in): the fine patch truly refines the elliptic. Never replace an unsupported
    // composite request with the physically different coarse-only solve below.
    if (composite_poisson_) {
      if (!replicated_coarse_ || stack_.nlev() != 2 || L[1].U.box_array().size() != 1)
        throw std::runtime_error(
            "AmrCouplerMP composite Poisson currently requires exactly two levels, one fine patch "
            "and a replicated coarse level; coarse-only fallback is forbidden");
      compute_aux_composite();
      return;
    }
    // right-hand side via the model (no copied formula): f = elliptic_rhs(U)
    detail::coupler_eval_rhs(L[0].U, mg_.rhs(), model_);
    mg_.solve();  // leaves phi with its ghosts filled (last gs_rb_sweep -> fill_ghosts)
    // Use the same named device kernel as the single-block and composite-FAC paths. This is the
    // exact centered operation order, handles every local box (including non-zero origins), and
    // leaves model-named aux components untouched.
    detail::coupler_grad_phi(mg_.phi(), stack_.aux(0),
                             Real(1) / (Real(2) * geom_.dx()),
                             Real(1) / (Real(2) * geom_.dy()));
    // Named aux components are installed once by set_named_aux. The gradient kernel writes only
    // comps 0..2, so replaying a host field on every update was redundant. The accepted coarse
    // values are propagated below and survive fine-level regrids through the same provider.
    BCRec level_aux_bc = detail::derive_aux_bc(elliptic_bc_);
    const Periodicity aux_periodicity{
        level_aux_bc.xlo == BCType::Periodic && level_aux_bc.xhi == BCType::Periodic,
        level_aux_bc.ylo == BCType::Periodic && level_aux_bc.yhi == BCType::Periodic};
    fill_ghosts(stack_.aux(0), dom, level_aux_bc);
    apply_named_aux_bc();  // ADC-369: per-field halo override on the coarse physical ghosts (after the
                           // shared fill); no-op on a periodic domain / without a policy.
    // parent aux(k-1) replicated only if level 0 is: otherwise it is DISTRIBUTED (multi-box)
    // and the injection goes through parallel_copy. Beyond level 1, the parent is always distributed.
    for (int k = 1; k < stack_.nlev(); ++k) {
      auto& workspace = aux_transfer_workspaces_.at(static_cast<std::size_t>(k - 1));
      if (!workspace)
        throw std::logic_error("AMR aux transfer workspace was not prepared");
      const bool replicated_parent = (k == 1) && replicated_coarse_;
      const CommunicatorView communicator =
          replicated_parent ? CommunicatorView{} : world_communicator_view();
      workspace->apply(stack_.aux(k - 1), stack_.aux(k), transfer_topology_generation_,
                       communicator);
      // The C/F provider owns parent-supported cells only. Same-level exchange then overwrites
      // overlaps and wraps domain-edge ghosts from the accepted fine valid state; leaving these
      // cells at allocation sentinels would corrupt a patch that touches a periodic seam.
      const Box2D level_domain = amr_level_index_domain(dom, k);
      level_aux_bc.dx /= Real(kAmrRefRatio);
      level_aux_bc.dy /= Real(kAmrRefRatio);
      fill_ghosts(stack_.aux(k), level_domain, level_aux_bc);
      apply_named_aux_bc(k);
    }
  }

  /// Updates the hierarchy before a step: sync_down (fine -> coarse) then compute_aux (coarse
  /// Poisson + grad phi + injection to the fine levels).
  void update() {
    sync_down();
    compute_aux();
  }

  // Selectable spatial discretization (default FirstOrder = NoSlope + Rusanov,
  // strictly identical to the old step()). recon_prim selects the primitive
  // reconstruction (same parameter as assemble_rhs / System); false (default) -> conservative.
  // imex: treats the stiff source IMPLICITLY (backward_euler) rather than forward Euler;
  // false (default) -> historical explicit treatment, bit-identical. The source being
  // cell-local (outside reflux registers), the implicit split preserves conservation.
  /// Advances the hierarchy by one step dt: update() then advance_amr (Berger-Oliger subcycling +
  /// reflux + conservative average_down). @tparam Disc: spatial discretization (limiter + flux,
  /// default FirstOrder bit-identical to the historical one). recon_prim: primitive reconstruction; imex:
  /// stiff source implicit (backward_euler). Defaults (false) -> historical explicit path.
  /// @p nopts: OPTIONS of the IMEX implicit-source Newton (iteration budget, tolerances,
  /// fd_eps, damping, fail_policy), threaded down to backward_euler_source by advance_amr ->
  /// subcycle_level_mp -> mf_apply_source_treatment. DEFAULT {} = historical constants (2 iters,
  /// 1e-7, ...) -> path (2a) BIT-IDENTICAL to the old call. No effect if imex==false. The
  /// partial IMEX mask is NOT carried by this mono-block path (full backward-Euler), only the OPTIONS
  /// are (the mono-block AmrSystem wires the Newton options but not the mask or the diagnostics).
  /// @p tmethod: time method (kEuler = forward Euler; kSsprk2 = order-2 SSPRK2/Heun;
  /// kSsprk3 = order-3 SSPRK3). SSP methods expose a stage-weighted effective reflux flux and require
  /// imex == false (rejected otherwise).
  template <class Disc = FirstOrder>
  void step(Real dt, bool recon_prim = false, bool imex = false, const NewtonOptions& nopts = {},
            AmrTimeMethod tmethod = AmrTimeMethod::kEuler, Real pos_floor = Real(0),
            Real weno_eps = kWenoEpsilon, bool wave_speed_cache = false) {
    update();
    advance_amr<typename Disc::Limiter, typename Disc::NumericalFlux>(
        model_, stack_.L(), stack_.domain(), dt, transport_periodicity_, replicated_coarse_,
        recon_prim, imex, nopts, tmethod, pos_floor, weno_eps, wave_speed_cache,
        transport_boundary_fill_ ? &*transport_boundary_fill_ : nullptr,
        fill_patch_plan_ ? &*fill_patch_plan_ : nullptr,
        average_down_plan_ ? &*average_down_plan_ : nullptr);
  }

  /// TRANSPORT-ONLY ADVANCE (hyperbolic), WITHOUT update() or source. Counterpart of step() stripped
  /// of its field solve and with imex==false: this is the PURE HYPERBOLIC advance (-div F) of the
  /// generated-Program path, where the field solve and source update are authored explicitly. The model must be
  /// SOURCE-FREE (NoSource source brick) so that the source is not counted twice (once
  /// here in forward Euler, once by the Program): this is the transport-only contract.
  template <class Disc = FirstOrder>
  void advance_transport(Real dt, bool recon_prim = false, Real pos_floor = Real(0),
                         Real weno_eps = kWenoEpsilon, bool wave_speed_cache = false) {
    advance_amr<typename Disc::Limiter, typename Disc::NumericalFlux>(
        model_, stack_.L(), stack_.domain(), dt, transport_periodicity_, replicated_coarse_,
        recon_prim, /*imex=*/false, NewtonOptions{}, AmrTimeMethod::kEuler, pos_floor, weno_eps,
        wave_speed_cache, transport_boundary_fill_ ? &*transport_boundary_fill_ : nullptr,
        fill_patch_plan_ ? &*fill_patch_plan_ : nullptr,
        average_down_plan_ ? &*average_down_plan_ : nullptr);
  }

  /// Injects the CURRENT native runtime-param values @p rp into the model's bricks (ADC-514): every
  /// brick (hyp / src / ell) carrying a `pops::RuntimeParams params` member takes @p rp in place of its
  /// declaration defaults, so the NEXT update() / advance reads the new values -- no recompile. A brick
  /// without such a member is a no-op (the SAME apply_runtime_params contract as the AOT ABI). Called at
  /// the top of each macro-step by the build_amr_compiled closure when the block declares a runtime param.
  void set_params(const RuntimeParams& rp) {
    apply_params_to_brick(model_.hyp, rp);
    apply_params_to_brick(model_.src, rp);
    apply_params_to_brick(model_.ell, rp);
  }

  // Regrid of the FINE level by Berger-Rigoutsos (delegated to amr_regrid_finest):
  // rebuilds the patches (carry over fine data, otherwise parent interp) + the aux.
  // margin = nesting. The coupler only orders the call.
  template <class Crit>
  void regrid(Crit crit, int grow = 2, int margin = 2) {
    const RegridProlongation prolong = [base_domain = stack_.domain(),
                                        periodicity = transport_periodicity_](
                                          const MultiFab& parent, MultiFab& fine,
                                          int parent_level, int ratio,
                                          bool parent_replicated,
                                          const CommunicatorView&) {
      const Box2D coarse_domain = amr_level_index_domain(base_domain, parent_level);
      const Box2D fine_domain = coarse_domain.refine(ratio);
      detail::coupler_conservative_linear_to_fine_mb(
          parent, fine, coarse_domain, fine_domain,
          {coarse_domain.lo[0], coarse_domain.lo[1]},
          {fine_domain.lo[0], fine_domain.lo[1]}, {ratio, ratio}, parent_replicated,
          periodicity);
      (void)parent_level;
    };
    std::optional<RegridPhysicalGhostSupport> physical_support;
    if (transport_boundary_fill_)
      physical_support = RegridPhysicalGhostSupport{
          transport_boundary_fill_->provided_depth,
          transport_boundary_fill_->fills_all_allocated_ghosts};
    amr_regrid_finest(
        stack_.L(), stack_.aux(), stack_.domain(), crit, grow, margin, prolong,
        aux_comps<Model>(), replicated_coarse_, *load_balance_authority_,
        RegridPeriodicity{transport_periodicity_.x, transport_periodicity_.y},
        world_communicator_view(),
        physical_support ? &*physical_support : nullptr);
    prepare_aux_transfer_workspaces_(next_transfer_topology_generation_());
  }

  // coarse mass via the shared diagnostic amr_mass_mb (replicated mono-box as well as
  // distributed multi-box). Replicated coarse: the local sum IS already the total mass
  // (each rank holds everything) -> no all_reduce. Distributed: local part -> all_reduce_sum.
  Real mass() const {
    const Real M = amr_mass_mb(stack_.coarse(), geom_.dx(), geom_.dy());
    return replicated_coarse_ ? M : all_reduce_sum(M);
  }

  // max drift speed via amr_max_drift_speed_mb + floor. all_reduce_max correct
  // in BOTH cases: under replication the local max is already global (idempotent);
  // distributed, we take the max of the parts.
  Real max_drift_speed() const {
    const Real local = amr_max_drift_speed_mb(stack_.aux(0), model_.B0);
    const Real global = all_reduce_max(local);
    detail::require_finite_amr_drift_speed(global);
    return std::max(global, kAmrDriftSpeedFloor);
  }

  /// @brief Max wave speed on the coarse level via `model.max_wave_speed`.
  ///
  /// Model-generic CFL speed (any `PhysicalModel`), unlike `max_drift_speed`
  /// which is specific to the E x B drift (`model.B0`). For a pure E x B transport, it equals
  /// the drift speed.
  ///
  /// @return the max over the coarse cells and the two directions, reduced over the ranks.
  /// @note `update()` must have run so that `aux(0)` carries the current `grad phi`.
  Real max_wave_speed() {
    MultiFab& U = stack_.coarse();
    MultiFab& A = stack_.aux(0);
    // max_wave_speed_mf is the generic, cross-TU device kernel used by every Cartesian block. It
    // performs the exact Kokkos max over both directions, validates finite non-negative model
    // outputs collectively, then executes one MPI max. Do not duplicate that contract here.
    return std::max(max_wave_speed_mf(model_, U, A), kAmrDriftSpeedFloor);
  }

 private:
  Periodicity aux_periodicity_() const {
    const BCRec bc = detail::derive_aux_bc(elliptic_bc_);
    return Periodicity{bc.xlo == BCType::Periodic && bc.xhi == BCType::Periodic,
                       bc.ylo == BCType::Periodic && bc.yhi == BCType::Periodic};
  }

  void prepare_aux_transfer_workspaces_() {
    prepare_aux_transfer_workspaces_(transfer_topology_generation_);
  }

  void prepare_aux_transfer_workspaces_(std::uint64_t topology_generation) {
    std::vector<std::optional<detail::PreparedConservativeLinearTransferWorkspace>> prepared(
        static_cast<std::size_t>(std::max(0, stack_.nlev() - 1)));
    const Periodicity periodicity = aux_periodicity_();
    for (int level = 1; level < stack_.nlev(); ++level) {
      const bool replicated_parent = level == 1 && replicated_coarse_;
      const CommunicatorView communicator =
          replicated_parent ? CommunicatorView{} : world_communicator_view();
      prepared[static_cast<std::size_t>(level - 1)].emplace(
          detail::PreparedConservativeLinearTransferWorkspace::prepare(
              stack_.aux(level - 1), stack_.aux(level),
              amr_level_index_domain(stack_.domain(), level - 1),
              amr_level_index_domain(stack_.domain(), level), replicated_parent,
              detail::ConservativeCellFillRegion::ValidAndGhost, periodicity,
              topology_generation, communicator));
    }
    auto fill_patch = PreparedAmrFillPatchPlan::prepare(
        stack_.L(), stack_.domain(), transport_periodicity_, replicated_coarse_,
        topology_generation);
    auto average_down =
        PreparedAmrAverageDownPlan::prepare(stack_.L(), topology_generation);
    aux_transfer_workspaces_.swap(prepared);
    fill_patch_plan_ = std::move(fill_patch);
    average_down_plan_ = std::move(average_down);
    transfer_topology_generation_ = topology_generation;
  }

  std::uint64_t next_transfer_topology_generation_() const noexcept {
    std::uint64_t next = transfer_topology_generation_ + 1;
    if (next == 0)
      ++next;
    return next;
  }

  /// COMPOSITE FAC Poisson step (opt-in path). Solves the elliptic on coarse + fine patch coupled by
  /// FAC, then sets aux PER LEVEL from the phi OF EACH LEVEL: fine aux = (phi_f, fine grad) where fine
  /// grad = centered diff on phi_f (solved at fine resolution), NOT the constant coarse-grad injection of Option A.
  void compute_aux_composite() {
    auto& L = stack_.L();
    const Box2D& dom = stack_.domain();
    const Box2D fine_box = L[1].U.box_array()[0];
    if (!fac_built_ || !same_box(fac_fine_box_, fine_box)) {
      fac_ = std::make_shared<CompositeFacPoisson>(geom_, coarse_boxes_, elliptic_bc_, fine_box, 2);
      fac_->set_options(fac_options_);  // ADC-614: apply the installed FAC knobs (default = kFAC*).
      fac_fine_box_ = fine_box;
      fac_built_ = true;
    }
    // f = elliptic_rhs(U) PER LEVEL: the fine has its OWN refined right-hand side (not an injection).
    detail::coupler_eval_rhs(L[0].U, fac_->rhs_coarse(), model_);
    detail::coupler_eval_rhs(L[1].U, fac_->rhs_fine(), model_);
    fac_->solve();
    device_fence();
    // level-0 aux (coarse): phi + grad from phi_coarse (same centered stencils as the Option A path).
    fill_ghosts(fac_->phi_coarse(), dom, elliptic_bc_);
    detail::coupler_grad_phi(fac_->phi_coarse(), stack_.aux(0), Real(1) / (Real(2) * geom_.dx()),
                             Real(1) / (Real(2) * geom_.dy()));
    BCRec coarse_aux_bc = detail::derive_aux_bc(elliptic_bc_);
    fill_ghosts(stack_.aux(0), dom, coarse_aux_bc);
    apply_named_aux_bc(0);
    // level-1 aux (fine): phi + grad from phi_fine -> FINE grad (fine centered diff, reads the C-F
    // bilinear ghosts) = the fidelity gain vs the constant coarse grad injected by Option A.
    detail::coupler_grad_phi(fac_->phi_fine(), stack_.aux(1), Real(1) / (Real(2) * L[1].dx),
                             Real(1) / (Real(2) * L[1].dy));
    if (!named_aux_components_.empty()) {
      auto& workspace = aux_transfer_workspaces_.at(0);
      if (!workspace)
        throw std::logic_error("composite AMR named-aux transfer workspace was not prepared");
      const CommunicatorView communicator =
          replicated_coarse_ ? CommunicatorView{} : world_communicator_view();
      workspace->apply(stack_.aux(0), stack_.aux(1),
                       std::span<const int>(named_aux_components_),
                       transfer_topology_generation_, communicator);
    }
    BCRec fine_aux_bc = coarse_aux_bc;
    fine_aux_bc.dx /= Real(kAmrRefRatio);
    fine_aux_bc.dy /= Real(kAmrRefRatio);
    fill_ghosts(stack_.aux(1), amr_level_index_domain(dom, 1), fine_aux_bc);
    apply_named_aux_bc(1);
  }

  static bool same_box(const Box2D& a, const Box2D& b) {
    return a.lo[0] == b.lo[0] && a.lo[1] == b.lo[1] && a.hi[0] == b.hi[0] && a.hi[1] == b.hi[1];
  }

  // Detect whether brick @p B carries a `pops::RuntimeParams params` member (ADC-514). A native brick or
  // a brick without a runtime param has none -> apply_params_to_brick is then a no-op (bit-identity). SAME
  // shape as compiled_block::HasRuntimeParams, reproduced here to keep this header off the heavy AOT ABI.
  template <class B, class = void>
  struct HasParamsMember : std::false_type {};
  template <class B>
  struct HasParamsMember<B, std::void_t<decltype(std::declval<B&>().params)>>
      : std::is_same<std::decay_t<decltype(std::declval<B&>().params)>, RuntimeParams> {};

  template <class B>
  static void apply_params_to_brick(B& b, const RuntimeParams& rp) {
    if constexpr (HasParamsMember<B>::value)
      b.params = rp;
  }

  // NATIVE per-block runtime params (ADC-514) mutate the model between macro-steps (set_params), so model_
  // is mutable. For a param-free model set_params is never called -> no mutation, bit-identical.
  mutable Model model_;
  Geometry geom_;
  BoxArray coarse_boxes_;
  DistributionMapping coarse_mapping_;
  BCRec elliptic_bc_;
  Elliptic mg_;
  AmrLevelStack<AmrLevelMP> stack_;
  bool
      replicated_coarse_;  // level 0 replicated (true) or distributed multi-box (false, de-replication)
  std::shared_ptr<const PreparedLoadBalanceAuthority> load_balance_authority_;
  Periodicity transport_periodicity_{true, true};
  std::optional<AmrBoundaryFillAuthority> transport_boundary_fill_;
  // COMPOSITE FAC Poisson path (opt-in, set_composite_poisson). fac_ built lazily on the
  // current fine patch (rebuilt if the patch changes after regrid). Default OFF -> Option A bit-identical.
  bool composite_poisson_ = false;
  bool fac_built_ = false;
  std::shared_ptr<CompositeFacPoisson> fac_;
  CompositeFacOptions fac_options_;  ///< ADC-614: FAC knobs applied at build (default = kFAC*).
  Box2D fac_fine_box_{};
  std::uint64_t transfer_topology_generation_ = 1;
  std::vector<std::optional<detail::PreparedConservativeLinearTransferWorkspace>>
      aux_transfer_workspaces_;
  std::optional<PreparedAmrFillPatchPlan> fill_patch_plan_;
  std::optional<PreparedAmrAverageDownPlan> average_down_plan_;
  // Model-NAMED aux fields (ADC-291): component (>= kAuxNamedBase) -> coarse base-level field
  // (ny*nx row-major). STATIC user fields are installed once; compute_aux writes only comps 0..2 and
  // every fine regrid is repopulated from the coarse authority. Empty by default -> bit-identical.
  std::map<int, std::vector<Real>> named_aux_;
  std::vector<int> named_aux_components_;
  // Per-field aux HALO policy (ADC-369): component -> uniform boundary policy, applied to the coarse aux
  // after the shared fill (apply_named_aux_bc). Empty by default -> bit-identical.
  std::map<int, AuxHaloPolicy> named_aux_bc_;

  // Re-applies the model-NAMED aux fields onto the COARSE shared aux valid cells. Mirror of
  // SystemFieldSolver::apply_named_aux_one and AmrRuntime::apply_named_aux: per local fab (MPI-safe),
  // valid cells only, global flat index j*nx+i. compute_aux runs the coarse->fine injection right
  // after, carrying the named comps to the fine levels. No-op without a named field.
  void apply_named_aux() {
    if (named_aux_.empty())
      return;
    const Box2D logical_domain = stack_.domain();
    const int row = logical_domain.nx();
    const std::size_t expected_size =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(logical_domain.ny());
    // User-field installation is an explicit host unpacking boundary. It is called by the setter,
    // never by the time-step/update hot path.
    stack_.aux(0).sync_host();
    for (const auto& [comp, field] : named_aux_) {
      if (field.empty() || comp >= stack_.aux(0).ncomp())
        continue;
      if (field.size() != expected_size)
        throw std::runtime_error("named AMR aux field shape disagrees with the logical domain");
      for (int li = 0; li < stack_.aux(0).local_size(); ++li) {
        Array4 a = stack_.aux(0).fab(li).array();
        const Box2D v = stack_.aux(0).box(li);
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i)
            a(i, j, comp) =
                field[static_cast<std::size_t>(j - logical_domain.lo[1]) * row +
                      static_cast<std::size_t>(i - logical_domain.lo[0])];
      }
    }
    stack_.aux(0).sync_device();
  }

  // Per-field aux HALO override (ADC-369) on the COARSE aux, AFTER the shared fill. Overrides only each
  // declared component's physical-face ghosts; aux_halo_override(elliptic_bc_, policy) keeps periodic faces
  // periodic (so on a periodic domain this is a no-op). Mirror of SystemFieldSolver::apply_named_aux_bc.
  void apply_named_aux_bc(int level = 0) {
    if (named_aux_bc_.empty())
      return;
    if (level < 0 || level >= stack_.nlev())
      throw std::out_of_range("AmrCouplerMP named aux boundary level is out of range");
    const Box2D level_domain = amr_level_index_domain(stack_.domain(), level);
    BCRec level_bc = detail::derive_aux_bc(elliptic_bc_);
    for (int transition = 0; transition < level; ++transition) {
      level_bc.dx /= Real(kAmrRefRatio);
      level_bc.dy /= Real(kAmrRefRatio);
    }
    for (const auto& [comp, policy] : named_aux_bc_) {
      if (comp >= stack_.aux(level).ncomp())
        continue;
      fill_physical_bc(stack_.aux(level), level_domain, aux_halo_override(level_bc, policy), comp);
    }
  }
};

}  // namespace pops
