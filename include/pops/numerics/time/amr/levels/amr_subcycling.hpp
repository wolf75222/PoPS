#pragma once
#include <pops/core/foundation/validation.hpp>
#include <pops/amr/hierarchy/refinement_ratio.hpp>
#include <pops/mesh/layout/refinement.hpp>  // coarsen, parallel_copy
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/time/amr/reflux/amr_flux_helpers.hpp>
#include <pops/numerics/time/amr/levels/amr_patch_range.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

/// @file
/// @brief AMR multi-patch hierarchy storage, prepared transfer/reflux workspaces and spatial helpers
///        (mf_fill_fine_ghosts_mb, mf_average_down_mb, mf_find_box, coarsen_grown) around
///        AmrLevelMP. ProgramGraph composes these objects in time.
///
/// Layer: `include/pops/numerics/time`.
/// Role: COVERAGE-AWARE reflux in the style of AMReX FluxRegister -- a coarse cell adjacent to a
///        fine patch is corrected ONLY if it is not covered by another patch (fine-fine
///        interfaces are handled by fill_boundary).
///
/// Invariants:
/// - distributed (MPI) with COARSE REPLICATION: the single-box coarse level is replicated on each
///   rank (local periodic fill), the fine patches distributed; reflux and average_down gather up
///   through GLOBAL-indexed coarse buffers + all_reduce_sum_inplace, then each rank applies to
///   its copy -> all stay identical. In serial this is bit-for-bit identical to the direct path;
/// - validation: test_mpi_amr_multipatch (np=1/2/4 bit-identical);
/// - state transfer and reflux helper kernels are device-clean (named functors).

namespace pops {

static_assert(kAmrRefRatio == 2, "ratio-2-structural kernels below assume kAmrRefRatio == 2");

inline Box2D amr_level_index_domain(Box2D base_domain, int level) {
  if (level < 0)
    throw std::invalid_argument("AMR level index must be non-negative");
  for (int transition = 0; transition < level; ++transition)
    base_domain = base_domain.refine(kAmrRefRatio);
  return base_domain;
}

struct AmrBoundaryFillContext {
  Box2D domain;
  int level = 0;
  Real dx = Real(1);
  Real dy = Real(1);
};

using AmrPhysicalBoundaryFill = std::function<void(MultiFab&, const AmrBoundaryFillContext&)>;

/// Exact host-side authority for physical AMR ghosts.  Same-level and periodic exchange remain
/// native runtime responsibilities; this callback owns only faces where periodicity is false.
/// A bounded external provider certifies provided_depth; a provider whose algorithm explicitly
/// handles arbitrary allocated depth certifies fills_all_allocated_ghosts instead.  Neither value
/// is inferred from a BC enum or a reconstruction name.
struct AmrBoundaryFillAuthority {
  Periodicity periodicity{};
  int provided_depth = 0;
  bool fills_all_allocated_ghosts = false;
  AmrPhysicalBoundaryFill fill_physical{};
};

inline AmrBoundaryFillAuthority make_amr_boundary_fill_authority(const BCRec& boundary) {
  detail::validate_periodic_pairs(boundary);
  BCRec prepared = boundary;
  return AmrBoundaryFillAuthority{
      Periodicity{boundary.xlo == BCType::Periodic, boundary.ylo == BCType::Periodic}, 0, true,
      [prepared](MultiFab& state, const AmrBoundaryFillContext& context) mutable {
        prepared.dx = context.dx;
        prepared.dy = context.dy;
        fill_physical_bc(state, context.domain, prepared);
      }};
}

inline void validate_amr_boundary_fill_authority(Periodicity periodicity,
                                                 const AmrBoundaryFillAuthority* authority) {
  const bool has_physical_face = !periodicity.x || !periodicity.y;
  if (authority == nullptr) {
    if (has_physical_face)
      throw std::runtime_error(
          "non-periodic AMR advance requires an explicit physical boundary-fill authority");
    return;
  }
  if (!same_periodicity(periodicity, authority->periodicity))
    throw std::runtime_error(
        "AMR boundary-fill authority periodicity disagrees with the hierarchy");
  if (authority->provided_depth < 0 || (has_physical_face && !authority->fill_physical))
    throw std::runtime_error("AMR boundary-fill authority is incomplete");
}

template <class Levels>
inline void validate_amr_boundary_fill_authority(Periodicity periodicity,
                                                 const AmrBoundaryFillAuthority* authority,
                                                 const Levels& levels) {
  validate_amr_boundary_fill_authority(periodicity, authority);
  if (authority == nullptr)
    return;
  for (const auto& level : levels)
    if (!authority->fills_all_allocated_ghosts && authority->provided_depth < level.U.n_grow())
      throw std::runtime_error("AMR boundary-fill authority does not cover all state ghosts");
}

inline void fill_amr_same_level_and_physical(MultiFab& state, const Box2D& domain, int level,
                                             Real dx, Real dy, Periodicity periodicity,
                                             const AmrBoundaryFillAuthority* authority) {
  fill_boundary(state, domain, periodicity);
  if ((!periodicity.x || !periodicity.y) && authority != nullptr) {
    std::string local_error;
    try {
      authority->fill_physical(state, AmrBoundaryFillContext{domain, level, dx, dy});
    } catch (const std::exception& error) {
      local_error = error.what();
    } catch (...) {
      local_error = "physical boundary callback raised a non-standard exception";
    }
    if (all_reduce_max(local_error.empty() ? 0L : 1L) != 0) {
      if (n_ranks() == 1)
        throw std::runtime_error(local_error);
      throw std::runtime_error("physical AMR boundary callback failed on at least one MPI rank");
    }
  }
}

// --- MULTI-PATCH (several fine boxes per level) ---
// The fine level is a MultiFab with N boxes. Reflux is COVERAGE-AWARE: it corrects a coarse
// cell adjacent to a fine box only if it is NOT covered by another fine box (real fine-coarse
// interface; fine-fine interfaces are handled by fill_boundary). This is AMReX FluxRegister
// logic.

// --- N-LEVEL MULTI-PATCH SPATIAL SERVICES (multi-box at EACH level) ---
// Each level is a multi-box MultiFab. The Program-owned conservative ledger feeds the prepared
// coverage-aware reflux service, which routes each correction to the PARENT box containing the
// adjacent coarse cell. With one box per level, the same spatial transfer and correction contracts
// apply without a separate single-box time path.
//
// Distributed state (MPI): DISTRIBUTED and tested bit-for-bit identical np=1/2/4
// (test_mpi_amr_multipatch3, 3 levels with a distributed multi-box intermediate level whose fine
// patch PARENT falls on another rank). Level 0 (coarse) is REPLICATED as in the 2-level case;
// levels >0 are distributed and play the role of both child and parent simultaneously. The five
// points assuming a local parent (via mf_find_box) are resolved:
//   1. mf_fill_fine_ghosts_mb: REPLICATED parent (lev==1) read locally; DISTRIBUTED parent
//      (lev>=2) brought in by parallel_copy (parent -> fine-coarsen) then interpolated;
//   2. coarse register sampling: REPLICATED parent read locally, DISTRIBUTED parent brought in by
//      parallel_copy onto a child-coarsen FACE grid;
//   3. mf_average_down_mb: average deposited in a GLOBAL-indexed coarse buffer + all_reduce_sum,
//      applied to the local parent boxes (replicated: all; distributed: the owner);
//   4. reflux: same global buffer + all_reduce, application guarded by local ownership of the
//      parent box (no double counting since the distributed parent has a single owner);
//   5. coverage: already built on the global box_array() (MPI-safe).
// In serial all_reduce is the identity and parallel_copy reduces to memory copies: the
// distributed path runs the same floating-point operations as the single-rank one -> bit-
// identical.
// AmrCouplerMP uses the same distributed contract for aux publication: one prepared
// conservative-linear workspace per transition migrates every distributed parent through
// PreparedPeriodicCopyPlan before reconstruction. No level-count-specific local-parent shortcut
// remains in either the state or aux path.

// LOCAL (valid) box containing cell (I,J), or -1.
inline int mf_find_box(const MultiFab& mf, int I, int J) {
  for (int li = 0; li < mf.local_size(); ++li)
    if (mf.box(li).contains(I, J))
      return li;
  return -1;
}

// Sparse cell -> LOCAL-box-index lookup over a MultiFab's LOCAL valid boxes.  Construction and
// storage are O(sum of local box areas), independent of holes between patches; the open-addressed
// view is also device-addressable for prepared kernels.  Valid boxes are required to be disjoint,
// so each populated key has one deterministic owner.  Missing cells map to -1, matching
// mf_find_box.  Per-rank local: no collective, MPI-safe.
struct MfBoxLookup {
  SparseCellLookup cells;

  explicit MfBoxLookup(const MultiFab& mf) {
    const int n = mf.local_size();
    if (n == 0)
      return;
    std::size_t covered_cells = 0;
    for (int li = 0; li < n; ++li) {
      const std::int64_t box_cells = mf.box(li).num_cells();
      if (box_cells <= 0 || static_cast<std::uint64_t>(box_cells) >
                                std::numeric_limits<std::size_t>::max() - covered_cells)
        throw std::overflow_error("MultiFab local box lookup size overflow");
      covered_cells += static_cast<std::size_t>(box_cells);
    }
    cells.reserve(covered_cells);
    for (int li = 0; li < n; ++li) {
      const Box2D b = mf.box(li);
      for (int J = b.lo[1];;) {
        for (int I = b.lo[0];;) {
          cells.insert(I, J, static_cast<std::size_t>(li), /*reject_duplicate=*/true);
          if (I == b.hi[0])
            break;
          ++I;
        }
        if (J == b.hi[1])
          break;
        ++J;
      }
    }
  }

  // Local box index containing (I,J), or -1. Identical to mf_find_box(mf, I, J).
  int find(int I, int J) const {
    std::size_t result = 0;
    if (!cells.view().locate(I, J, result))
      return -1;
    if (result > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("MultiFab local box index exceeds native range");
    return static_cast<int>(result);
  }

  SparseCellLookupView view() const { return cells.view(); }
  [[nodiscard]] std::size_t lookup_capacity() const noexcept { return cells.capacity(); }
};

// BoxArray of the child boxes grown by ngrow then coarsened (ratio 2). Each box covers all the
// coarse cells the child needs, ghosts included: this is the FillPatch fine-coarsen grid (cf.
// refinement.hpp::interpolate).
inline BoxArray coarsen_grown(const BoxArray& ba, int ngrow, int r) {
  std::vector<Box2D> b;
  b.reserve(ba.size());
  for (int i = 0; i < ba.size(); ++i)
    b.push_back(ba[i].grow(ngrow).coarsen(r));
  return BoxArray{std::move(b)};
}

/// Persistent FillPatch data plane for one exact parent/child topology.  Preparation allocates the
/// old/new parent carriers, materializes periodic image catalogues, and warms both communication
/// schedules.  Stable apply() calls only refresh the two snapshots and launch the temporal/spatial
/// interpolation kernels.  A topology generation is authenticated even when a regrid happens to
/// reproduce byte-identical boxes.
class PreparedFillPatchWorkspace {
 public:
  PreparedFillPatchWorkspace(const PreparedFillPatchWorkspace&) = delete;
  PreparedFillPatchWorkspace& operator=(const PreparedFillPatchWorkspace&) = delete;
  PreparedFillPatchWorkspace(PreparedFillPatchWorkspace&&) noexcept = default;
  PreparedFillPatchWorkspace& operator=(PreparedFillPatchWorkspace&&) noexcept = default;

  static PreparedFillPatchWorkspace prepare(const MultiFab& fine, const MultiFab& old_parent,
                                            const MultiFab& new_parent, const Box2D& coarse_domain,
                                            bool replicated_parent, Periodicity periodicity,
                                            std::uint64_t topology_generation,
                                            const CommunicatorView& communicator) {
    return prepare(fine, old_parent, new_parent, coarse_domain, replicated_parent, periodicity,
                   topology_generation, communicator,
                   std::make_shared<const PreparedCoarseFineOperator>(
                       prepare_limited_linear_coarse_fine_operator()));
  }

  static PreparedFillPatchWorkspace prepare(
      const MultiFab& fine, const MultiFab& old_parent, const MultiFab& new_parent,
      const Box2D& coarse_domain, bool replicated_parent, Periodicity periodicity,
      std::uint64_t topology_generation, const CommunicatorView& communicator,
      std::shared_ptr<const PreparedCoarseFineOperator> prepared_operator) {
    validate_temporal_window_(fine, old_parent, new_parent, coarse_domain);
    if (!prepared_operator)
      throw std::invalid_argument("prepared FillPatch lacks its coarse/fine operator");
    prepared_operator->validate_domain(coarse_domain);
    if (replicated_parent && communicator.active())
      throw std::invalid_argument("replicated FillPatch parent requires a rank-local communicator");

    // The carrier contains every source cell used by the selected route, including the furthest
    // one-sided stencil next to a non-periodic boundary.  This is prepared once per topology.
    const int reach =
        std::max(prepared_operator->parent_reach_x, prepared_operator->parent_reach_y);
    const int fine_growth =
        detail::checked_coarse_fine_carrier_growth(fine.n_grow(), kAmrRefRatio, reach);
    const BoxArray carrier_boxes = coarsen_grown(fine.box_array(), fine_growth, kAmrRefRatio);
    for (const Box2D& box : carrier_boxes.boxes())
      if (box.nx() < prepared_operator->minimum_axis_cells_x ||
          box.ny() < prepared_operator->minimum_axis_cells_y)
        throw std::invalid_argument(
            "prepared FillPatch carrier cannot hold the selected directional stencil");
    DistributionMapping carrier_mapping = fine.dmap();
    if (replicated_parent)
      carrier_mapping = DistributionMapping(
          std::vector<int>(static_cast<std::size_t>(carrier_boxes.size()), my_rank()));

    PreparedFillPatchWorkspace workspace(MultiFab(carrier_boxes, carrier_mapping, fine.ncomp(), 0),
                                         MultiFab(carrier_boxes, carrier_mapping, fine.ncomp(), 0),
                                         fine, old_parent, new_parent, coarse_domain,
                                         replicated_parent, periodicity, topology_generation,
                                         std::move(prepared_operator));
    workspace.old_copy_plan_.emplace(
        PreparedPeriodicCopyPlan::prepare(workspace.old_parent_carrier_, old_parent, coarse_domain,
                                          periodicity, topology_generation, communicator));
    workspace.new_copy_plan_.emplace(
        PreparedPeriodicCopyPlan::prepare(workspace.new_parent_carrier_, new_parent, coarse_domain,
                                          periodicity, topology_generation, communicator));
    workspace.validate_carrier_ownership_(fine);
    return workspace;
  }

  void apply(MultiFab& fine, const MultiFab& old_parent, const MultiFab& new_parent, Real fraction,
             Real positivity_floor, int positivity_component, std::uint64_t topology_generation,
             const CommunicatorView& communicator) {
    validate_replay_(fine, old_parent, new_parent, fraction, positivity_floor, positivity_component,
                     topology_generation);
    // Validate both sources before posting either transfer: an invalid new snapshot cannot leave
    // only the old carrier refreshed.  PreparedPeriodicCopyPlan repeats the same exact check at the
    // collective boundary and authenticates the communicator.
    old_copy_plan_->apply(old_parent_carrier_, old_parent, topology_generation, communicator);
    new_copy_plan_->apply(new_parent_carrier_, new_parent, topology_generation, communicator);
    fill_from_prepared_carriers_(fine, fraction, positivity_floor, positivity_component);
  }

  /// Publish the two source snapshots already copied by prepare().  This is the one-shot setup/test
  /// route; a persistent runtime calls apply() for each later substep and stage.
  void publish_prepared(MultiFab& fine, Real fraction, Real positivity_floor = Real(0),
                        int positivity_component = 0) {
    validate_fine_(fine, topology_generation_);
    validate_numerical_inputs_(fine.ncomp(), fraction, positivity_floor, positivity_component);
    fill_from_prepared_carriers_(fine, fraction, positivity_floor, positivity_component);
  }

  [[nodiscard]] std::uint64_t topology_generation() const noexcept { return topology_generation_; }
  [[nodiscard]] const std::shared_ptr<const PreparedCoarseFineOperator>& prepared_operator()
      const noexcept {
    return prepared_operator_;
  }

 private:
  PreparedFillPatchWorkspace(MultiFab old_parent_carrier, MultiFab new_parent_carrier,
                             const MultiFab& fine, const MultiFab& old_parent,
                             const MultiFab& new_parent, Box2D coarse_domain,
                             bool replicated_parent, Periodicity periodicity,
                             std::uint64_t topology_generation,
                             std::shared_ptr<const PreparedCoarseFineOperator> prepared_operator)
      : old_parent_carrier_(std::move(old_parent_carrier)),
        new_parent_carrier_(std::move(new_parent_carrier)),
        fine_boxes_(fine.box_array().boxes()),
        fine_ranks_(fine.dmap().ranks()),
        old_parent_boxes_(old_parent.box_array().boxes()),
        old_parent_ranks_(old_parent.dmap().ranks()),
        new_parent_boxes_(new_parent.box_array().boxes()),
        new_parent_ranks_(new_parent.dmap().ranks()),
        fine_ncomp_(fine.ncomp()),
        fine_ngrow_(fine.n_grow()),
        old_parent_ngrow_(old_parent.n_grow()),
        new_parent_ngrow_(new_parent.n_grow()),
        coarse_domain_(coarse_domain),
        fine_domain_(coarse_domain.refine(kAmrRefRatio)),
        transform_{coarse_domain.lo[0], coarse_domain.lo[1], fine_domain_.lo[0],
                   fine_domain_.lo[1],  kAmrRefRatio,        kAmrRefRatio},
        replicated_parent_(replicated_parent),
        periodicity_(periodicity),
        topology_generation_(topology_generation),
        prepared_operator_(std::move(prepared_operator)) {}

  static void validate_temporal_window_(const MultiFab& fine, const MultiFab& old_parent,
                                        const MultiFab& new_parent, const Box2D& coarse_domain) {
    if (coarse_domain.empty())
      throw std::invalid_argument("FillPatch requires a non-empty coarse domain");
    if (fine.ncomp() <= 0 || old_parent.ncomp() != fine.ncomp() ||
        new_parent.ncomp() != fine.ncomp())
      throw std::invalid_argument("FillPatch parent/child component mismatch");
    if (old_parent.box_array().boxes() != new_parent.box_array().boxes() ||
        old_parent.dmap().ranks() != new_parent.dmap().ranks() ||
        old_parent.n_grow() != new_parent.n_grow())
      throw std::invalid_argument("FillPatch old/new parent snapshots require one exact layout");
    validate_ratio_aligned_disjoint_fine_layout(fine.box_array(), &coarse_domain);
  }

  static void validate_numerical_inputs_(int components, Real fraction, Real positivity_floor,
                                         int positivity_component) {
    if (!std::isfinite(fraction) || fraction < Real(0) || fraction > Real(1))
      throw std::invalid_argument("FillPatch temporal fraction must lie in [0, 1]");
    if (!std::isfinite(positivity_floor))
      throw std::invalid_argument("FillPatch positivity floor must be finite");
    if (positivity_floor > Real(0) &&
        (positivity_component < 0 || positivity_component >= components))
      throw std::out_of_range("FillPatch positivity component is out of range");
  }

  void validate_fine_(const MultiFab& fine, std::uint64_t topology_generation) const {
    if (fine.box_array().boxes() != fine_boxes_ || fine.dmap().ranks() != fine_ranks_ ||
        fine.ncomp() != fine_ncomp_ || fine.n_grow() != fine_ngrow_)
      throw std::invalid_argument("prepared FillPatch crossed an exact child layout");
    if (topology_generation != topology_generation_)
      throw std::invalid_argument("prepared FillPatch crossed a topology generation");
  }

  void validate_replay_(const MultiFab& fine, const MultiFab& old_parent,
                        const MultiFab& new_parent, Real fraction, Real positivity_floor,
                        int positivity_component, std::uint64_t topology_generation) const {
    validate_fine_(fine, topology_generation);
    validate_numerical_inputs_(fine_ncomp_, fraction, positivity_floor, positivity_component);
    if (!old_copy_plan_ || !new_copy_plan_ || old_parent.box_array().boxes() != old_parent_boxes_ ||
        old_parent.dmap().ranks() != old_parent_ranks_ || old_parent.ncomp() != fine_ncomp_ ||
        old_parent.n_grow() != old_parent_ngrow_ ||
        new_parent.box_array().boxes() != new_parent_boxes_ ||
        new_parent.dmap().ranks() != new_parent_ranks_ || new_parent.ncomp() != fine_ncomp_ ||
        new_parent.n_grow() != new_parent_ngrow_)
      throw std::invalid_argument("prepared FillPatch crossed an exact parent layout");
  }

  void validate_carrier_ownership_(const MultiFab& fine) const {
    for (int local_fine = 0; local_fine < fine.local_size(); ++local_fine) {
      const int global = fine.global_index(local_fine);
      if (old_parent_carrier_.local_index_of(global) < 0 ||
          new_parent_carrier_.local_index_of(global) < 0)
        throw std::logic_error("prepared FillPatch carrier does not follow child ownership");
    }
  }

  void fill_from_prepared_carriers_(MultiFab& fine, Real fraction, Real positivity_floor,
                                    int positivity_component) {
    validate_carrier_ownership_(fine);  // complete host preflight before the first kernel launch
    for (int local_fine = 0; local_fine < fine.local_size(); ++local_fine) {
      const int global = fine.global_index(local_fine);
      const int old_carrier_local = old_parent_carrier_.local_index_of(global);
      const int new_carrier_local = new_parent_carrier_.local_index_of(global);
      const Box2D valid = fine.box(local_fine);
      prepared_operator_->launch_space_time(
          fine.fab(local_fine).array(), old_parent_carrier_.fab(old_carrier_local).const_array(),
          new_parent_carrier_.fab(new_carrier_local).const_array(),
          fine.fab(local_fine).grown_box(), valid, coarse_domain_, fine_domain_, transform_,
          fine_ncomp_, fraction, positivity_floor, positivity_component, periodicity_);
    }
    // Carriers are persistent and may be refreshed by the next apply.  Complete all device reads
    // before returning ownership to the runtime.
    device_fence();
  }

  MultiFab old_parent_carrier_;
  MultiFab new_parent_carrier_;
  std::optional<PreparedPeriodicCopyPlan> old_copy_plan_;
  std::optional<PreparedPeriodicCopyPlan> new_copy_plan_;
  std::vector<Box2D> fine_boxes_;
  std::vector<int> fine_ranks_;
  std::vector<Box2D> old_parent_boxes_;
  std::vector<int> old_parent_ranks_;
  std::vector<Box2D> new_parent_boxes_;
  std::vector<int> new_parent_ranks_;
  int fine_ncomp_ = 0;
  int fine_ngrow_ = 0;
  int old_parent_ngrow_ = 0;
  int new_parent_ngrow_ = 0;
  Box2D coarse_domain_{};
  Box2D fine_domain_{};
  PreparedCoarseFineTransform2D transform_{};
  bool replicated_parent_ = false;
  Periodicity periodicity_{};
  std::uint64_t topology_generation_ = 0;
  std::shared_ptr<const PreparedCoarseFineOperator> prepared_operator_;
};

// multi-box fine ghosts from a MULTI-BOX parent (conservative linear space + linear time),
// DISTRIBUTED. Two parent cases:
//  - REPLICATED (level 0, replicated_parent=true): the parent is fully local on each rank, read
//    directly via mf_find_box (always found); no collective. This is the replicated-coarse path,
//    like the 2-level case (parallel_copy would violate the replicated-metadata assumption of the
//    parent, per-rank dmap).
//  - DISTRIBUTED (intermediate): the parent may be on another rank; its valid regions are brought
//    onto a LOCAL child-coarsen grid by parallel_copy (MPI routing handled there), then
//    interpolated. No more silent remote failures.
// In serial both paths are identical (parent local everywhere, parallel_copy = memory copy).
inline void mf_fill_fine_ghosts_mb(MultiFab& Uf, const MultiFab& Po, const MultiFab& Pn,
                                   const Box2D& coarse_domain, Real frac, bool replicated_parent,
                                   Real pos_floor, int pos_comp, Periodicity periodicity) {
  const CommunicatorView communicator =
      replicated_parent ? CommunicatorView{} : world_communicator_view();
  auto workspace =
      PreparedFillPatchWorkspace::prepare(Uf, Po, Pn, coarse_domain, replicated_parent, periodicity,
                                          /*topology_generation=*/0, communicator);
  workspace.publish_prepared(Uf, frac, pos_floor, pos_comp);
}

/// Allocation-free FillPatch replay used by prepared runtimes.  The owning hierarchy keeps one
/// workspace per parent/child transition and replaces it only after a topology-generation change.
inline void mf_fill_fine_ghosts_mb(MultiFab& fine, const MultiFab& old_parent,
                                   const MultiFab& new_parent,
                                   PreparedFillPatchWorkspace& workspace, Real fraction,
                                   Real positivity_floor, int positivity_component,
                                   std::uint64_t topology_generation,
                                   const CommunicatorView& communicator) {
  workspace.apply(fine, old_parent, new_parent, fraction, positivity_floor, positivity_component,
                  topology_generation, communicator);
}

// Prepared coarse/fine spatial transfer. Unlike mf_fill_fine_ghosts_mb this operation has exactly
// one physical parent snapshot: time interpolation is a separate prepared route with explicit
// TimePoints. Keeping the two protocols separate prevents callers from manufacturing a fake
// `(parent, parent)` temporal window merely to request conservative spatial ghost materialization.
inline void mf_fill_fine_ghosts_spatial_mb(MultiFab& Uf, const MultiFab& parent,
                                           const Box2D& coarse_domain, bool replicated_parent,
                                           Periodicity periodicity) {
  if (parent.ncomp() != Uf.ncomp())
    throw std::runtime_error("coarse/fine spatial transfer component mismatch");
  mf_fill_fine_ghosts_mb(Uf, parent, parent, coarse_domain, Real(0), replicated_parent, Real(0), 0,
                         periodicity);
}

/// Persistent fine-to-parent synchronization for one exact transition topology.  The collective
/// register, device lookup, and coverage mask are allocated once; apply() clears and reuses them.
/// Exact layout and topology validation happens before the first device write.
class PreparedAverageDownWorkspace {
 public:
  PreparedAverageDownWorkspace(const PreparedAverageDownWorkspace&) = delete;
  PreparedAverageDownWorkspace& operator=(const PreparedAverageDownWorkspace&) = delete;
  PreparedAverageDownWorkspace(PreparedAverageDownWorkspace&&) noexcept = default;
  PreparedAverageDownWorkspace& operator=(PreparedAverageDownWorkspace&&) noexcept = default;

  static PreparedAverageDownWorkspace prepare(const MultiFab& fine, const MultiFab& coarse,
                                              std::uint64_t topology_generation) {
    if (fine.ncomp() <= 0 || fine.ncomp() != coarse.ncomp())
      throw std::invalid_argument("average-down parent/child component mismatch");
    if (fine.box_array().size() == 0 || coarse.box_array().size() == 0)
      throw std::invalid_argument("average-down requires non-empty parent and child layouts");
    validate_ratio_aligned_disjoint_fine_layout(fine.box_array());
    for (int current = 0; current < coarse.box_array().size(); ++current)
      for (int previous = 0; previous < current; ++previous)
        if (!coarse.box_array()[current].intersect(coarse.box_array()[previous]).empty())
          throw std::invalid_argument("average-down requires disjoint parent boxes");

    const BoxArray parent_footprints = coarsen(fine.box_array(), kAmrRefRatio);
    for (int footprint_index = 0; footprint_index < parent_footprints.size(); ++footprint_index) {
      const Box2D footprint = parent_footprints[footprint_index];
      std::int64_t covered_cells = 0;
      for (const Box2D& parent_box : coarse.box_array().boxes())
        covered_cells += footprint.intersect(parent_box).num_cells();
      if (covered_cells != footprint.num_cells())
        throw std::invalid_argument(
            "average-down child footprint is not exactly covered by the parent layout");
    }

    const Box2D bounds = parent_footprints.bounding_box();
    PreparedAverageDownWorkspace workspace(fine, coarse, parent_footprints.boxes(), bounds,
                                           topology_generation);
    for (const Box2D& footprint : parent_footprints.boxes())
      workspace.coverage_.mark(footprint);
    return workspace;
  }

  void apply(const MultiFab& fine, MultiFab& coarse, std::uint64_t topology_generation,
             const CommunicatorView& communicator) {
    validate_replay_(fine, coarse, topology_generation);
    average_.clear_on_device();
    for (int local_fine = 0; local_fine < fine.local_size(); ++local_fine) {
      const PatchRange range(fine.box(local_fine));
      for_each_cell(range.box(), detail::AverageDownRegisterKernel{
                                     fine.fab(local_fine).const_array(), average_.view(), ncomp_});
    }
    average_.gather(communicator);
    for (int local_coarse = 0; local_coarse < coarse.local_size(); ++local_coarse) {
      const Box2D target = coarse.box(local_coarse).intersect(bounds_);
      if (!target.empty())
        for_each_cell(target, detail::ApplyAverageDownRegisterKernel{
                                  coarse.fab(local_coarse).array(), average_.view(),
                                  coverage_.view(), ncomp_});
    }
    device_fence();
  }

 private:
  PreparedAverageDownWorkspace(const MultiFab& fine, const MultiFab& coarse,
                               std::vector<Box2D> parent_footprints, Box2D bounds,
                               std::uint64_t topology_generation)
      : fine_boxes_(fine.box_array().boxes()),
        fine_ranks_(fine.dmap().ranks()),
        coarse_boxes_(coarse.box_array().boxes()),
        coarse_ranks_(coarse.dmap().ranks()),
        fine_ngrow_(fine.n_grow()),
        coarse_ngrow_(coarse.n_grow()),
        ncomp_(fine.ncomp()),
        bounds_(bounds),
        topology_generation_(topology_generation),
        average_(std::move(parent_footprints), ncomp_),
        coverage_(bounds) {}

  void validate_replay_(const MultiFab& fine, const MultiFab& coarse,
                        std::uint64_t topology_generation) const {
    if (fine.box_array().boxes() != fine_boxes_ || fine.dmap().ranks() != fine_ranks_ ||
        fine.n_grow() != fine_ngrow_ || fine.ncomp() != ncomp_ ||
        coarse.box_array().boxes() != coarse_boxes_ || coarse.dmap().ranks() != coarse_ranks_ ||
        coarse.n_grow() != coarse_ngrow_ || coarse.ncomp() != ncomp_)
      throw std::invalid_argument("prepared average-down crossed an exact layout");
    if (topology_generation != topology_generation_)
      throw std::invalid_argument("prepared average-down crossed a topology generation");
  }

  std::vector<Box2D> fine_boxes_;
  std::vector<int> fine_ranks_;
  std::vector<Box2D> coarse_boxes_;
  std::vector<int> coarse_ranks_;
  int fine_ngrow_ = 0;
  int coarse_ngrow_ = 0;
  int ncomp_ = 0;
  Box2D bounds_{};
  std::uint64_t topology_generation_ = 0;
  FluxRegister average_;
  CoverageMask coverage_;
};

// multi-box fine average -> multi-box parent (each cell routed to its parent box), DISTRIBUTED.
// The parent box of a coarse cell may be on another rank, and the parent may be either REPLICATED
// (level 0, each rank has a copy) or DISTRIBUTED (intermediate, a single owner). Both are covered
// by a GLOBAL-indexed coarse buffer: each rank deposits the 2x2 average of ITS local fine patches
// (0 elsewhere; disjoint patches so a single contribution per covered cell), all_reduce_sum ->
// each rank has the total, then applies to ITS local parent boxes (overwrite). Replicated: all
// apply the same value to their copy. Distributed: only the owner applies. In serial all_reduce
// is the identity (0 + average = average) -> bit-for-bit identical to the direct routing.
inline void mf_average_down_mb(const MultiFab& Uf, MultiFab& Uc) {
  auto workspace = PreparedAverageDownWorkspace::prepare(Uf, Uc, /*topology_generation=*/0);
  workspace.apply(Uf, Uc, /*topology_generation=*/0, world_communicator_view());
}

inline void mf_average_down_mb(const MultiFab& fine, MultiFab& coarse,
                               PreparedAverageDownWorkspace& workspace,
                               std::uint64_t topology_generation,
                               const CommunicatorView& communicator) {
  workspace.apply(fine, coarse, topology_generation, communicator);
}

// one level of the multi-patch hierarchy (U + multi-box aux, same BoxArray).
struct AmrLevelMP {
  MultiFab U;
  const MultiFab* aux;
  Real dx, dy;
};

/// Prepared FillPatch workspaces for an exact hierarchy topology.  Runtime/coupler owners build one
/// plan after hierarchy installation or regrid and replay one workspace per parent/child transition
/// throughout every stage and substep.  No process-global cache is involved.
class PreparedAmrFillPatchPlan {
 public:
  PreparedAmrFillPatchPlan(const PreparedAmrFillPatchPlan&) = delete;
  PreparedAmrFillPatchPlan& operator=(const PreparedAmrFillPatchPlan&) = delete;
  PreparedAmrFillPatchPlan(PreparedAmrFillPatchPlan&&) noexcept = default;
  PreparedAmrFillPatchPlan& operator=(PreparedAmrFillPatchPlan&&) noexcept = default;

  static PreparedAmrFillPatchPlan prepare(const std::vector<AmrLevelMP>& levels,
                                          const Box2D& base_domain, Periodicity periodicity,
                                          bool coarse_replicated,
                                          std::uint64_t topology_generation) {
    return prepare(levels, base_domain, periodicity, coarse_replicated, topology_generation,
                   std::make_shared<const PreparedCoarseFineOperator>(
                       prepare_limited_linear_coarse_fine_operator()));
  }

  static PreparedAmrFillPatchPlan prepare(
      const std::vector<AmrLevelMP>& levels, const Box2D& base_domain, Periodicity periodicity,
      bool coarse_replicated, std::uint64_t topology_generation,
      std::shared_ptr<const PreparedCoarseFineOperator> prepared_operator) {
    if (levels.empty() || base_domain.empty())
      throw std::invalid_argument(
          "prepared AMR FillPatch plan requires a non-empty hierarchy and base domain");
    std::vector<PreparedFillPatchWorkspace> transitions;
    transitions.reserve(levels.size() - 1);
    for (std::size_t child = 1; child < levels.size(); ++child) {
      const bool replicated_parent = child == 1 && coarse_replicated;
      const CommunicatorView communicator =
          replicated_parent ? CommunicatorView{} : world_communicator_view();
      const MultiFab& parent = levels[child - 1].U;
      transitions.push_back(PreparedFillPatchWorkspace::prepare(
          levels[child].U, parent, parent,
          amr_level_index_domain(base_domain, static_cast<int>(child - 1)), replicated_parent,
          periodicity, topology_generation, communicator, prepared_operator));
    }
    return PreparedAmrFillPatchPlan(static_cast<int>(levels.size()), base_domain, periodicity,
                                    coarse_replicated, topology_generation, std::move(transitions),
                                    std::move(prepared_operator));
  }

  PreparedFillPatchWorkspace& transition_for_child(int child_level) {
    if (child_level <= 0 || child_level >= nlevels_)
      throw std::out_of_range("prepared AMR FillPatch child level is out of range");
    return transitions_.at(static_cast<std::size_t>(child_level - 1));
  }

  [[nodiscard]] int nlevels() const noexcept { return nlevels_; }
  [[nodiscard]] std::uint64_t topology_generation() const noexcept { return topology_generation_; }
  [[nodiscard]] const std::shared_ptr<const PreparedCoarseFineOperator>& prepared_operator()
      const noexcept {
    return prepared_operator_;
  }

  void validate_hierarchy_contract(
      int nlevels, const Box2D& base_domain, Periodicity periodicity, bool coarse_replicated,
      const std::shared_ptr<const PreparedCoarseFineOperator>& prepared_operator) const {
    if (nlevels != nlevels_ || base_domain != base_domain_ || periodicity.x != periodicity_.x ||
        periodicity.y != periodicity_.y || coarse_replicated != coarse_replicated_ ||
        prepared_operator.get() != prepared_operator_.get())
      throw std::invalid_argument(
          "prepared AMR FillPatch plan does not match the hierarchy contract");
  }

 private:
  PreparedAmrFillPatchPlan(int nlevels, Box2D base_domain, Periodicity periodicity,
                           bool coarse_replicated, std::uint64_t topology_generation,
                           std::vector<PreparedFillPatchWorkspace> transitions,
                           std::shared_ptr<const PreparedCoarseFineOperator> prepared_operator)
      : nlevels_(nlevels),
        base_domain_(base_domain),
        periodicity_(periodicity),
        coarse_replicated_(coarse_replicated),
        topology_generation_(topology_generation),
        transitions_(std::move(transitions)),
        prepared_operator_(std::move(prepared_operator)) {}

  int nlevels_ = 0;
  Box2D base_domain_{};
  Periodicity periodicity_{};
  bool coarse_replicated_ = false;
  std::uint64_t topology_generation_ = 0;
  std::vector<PreparedFillPatchWorkspace> transitions_;
  std::shared_ptr<const PreparedCoarseFineOperator> prepared_operator_;
};

class PreparedAmrAverageDownPlan {
 public:
  PreparedAmrAverageDownPlan(const PreparedAmrAverageDownPlan&) = delete;
  PreparedAmrAverageDownPlan& operator=(const PreparedAmrAverageDownPlan&) = delete;
  PreparedAmrAverageDownPlan(PreparedAmrAverageDownPlan&&) noexcept = default;
  PreparedAmrAverageDownPlan& operator=(PreparedAmrAverageDownPlan&&) noexcept = default;

  static PreparedAmrAverageDownPlan prepare(const std::vector<AmrLevelMP>& levels,
                                            std::uint64_t topology_generation) {
    if (levels.empty())
      throw std::invalid_argument("prepared AMR average-down plan requires a non-empty hierarchy");
    std::vector<PreparedAverageDownWorkspace> transitions;
    transitions.reserve(levels.size() - 1);
    for (std::size_t child = 1; child < levels.size(); ++child)
      transitions.push_back(PreparedAverageDownWorkspace::prepare(
          levels[child].U, levels[child - 1].U, topology_generation));
    return PreparedAmrAverageDownPlan(static_cast<int>(levels.size()), topology_generation,
                                      std::move(transitions));
  }

  PreparedAverageDownWorkspace& transition_for_child(int child_level) {
    if (child_level <= 0 || child_level >= nlevels_)
      throw std::out_of_range("prepared AMR average-down child level is out of range");
    return transitions_.at(static_cast<std::size_t>(child_level - 1));
  }

  [[nodiscard]] int nlevels() const noexcept { return nlevels_; }
  [[nodiscard]] std::uint64_t topology_generation() const noexcept { return topology_generation_; }

 private:
  PreparedAmrAverageDownPlan(int nlevels, std::uint64_t topology_generation,
                             std::vector<PreparedAverageDownWorkspace> transitions)
      : nlevels_(nlevels),
        topology_generation_(topology_generation),
        transitions_(std::move(transitions)) {}

  int nlevels_ = 0;
  std::uint64_t topology_generation_ = 0;
  std::vector<PreparedAverageDownWorkspace> transitions_;
};

namespace detail {

struct CopyAmrStorageKernel {
  Array4 destination;
  ConstArray4 source;
  int components = 0;

  POPS_HD void operator()(int i, int j) const {
    for (int component = 0; component < components; ++component)
      destination(i, j, component) = source(i, j, component);
  }
};

inline void copy_amr_storage(MultiFab& destination, const MultiFab& source) {
  if (destination.box_array().boxes() != source.box_array().boxes() ||
      destination.dmap().ranks() != source.dmap().ranks() ||
      destination.ncomp() != source.ncomp() || destination.n_grow() != source.n_grow())
    throw std::invalid_argument("AMR storage copy requires identical exact layouts");
  for (int local = 0; local < destination.local_size(); ++local)
    for_each_cell(destination.fab(local).grown_box(),
                  CopyAmrStorageKernel{destination.fab(local).array(),
                                       source.fab(local).const_array(), source.ncomp()});
}

inline void clear_reflux_storage_on_device(RefluxStorage<Real>& values) {
  if (values.empty())
    return;
  detail::ensure_kokkos_initialized();
  Kokkos::parallel_for(
      "pops_clear_reflux_strip",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<std::int64_t>>(
          0, static_cast<std::int64_t>(values.size())),
      ClearRefluxStorageKernel{values.data()});
}

}  // namespace detail

/// Persistent, patch-local output storage for one external Reflux invocation. It is allocated with
/// the topology plan, poisoned before every callback and consumed by PoPS only after all entries are
/// finite. Non-owning ABI views never outlive this workspace.
struct PreparedAmrRefluxFaceWorkspace {
  int I0 = 0, I1 = -1, J0 = 0, J1 = -1;
  int components = 0;
  RefluxStorage<Real> x_low;
  RefluxStorage<Real> x_high;
  RefluxStorage<Real> y_low;
  RefluxStorage<Real> y_high;
  std::string patch_identity;
  std::array<std::string, 4> interface_identities;

  static PreparedAmrRefluxFaceWorkspace prepare(const Box2D& footprint, int ncomp,
                                                std::string transition_identity,
                                                std::size_t global_child) {
    if (footprint.empty() || ncomp <= 0 || transition_identity.empty())
      throw std::invalid_argument("prepared external Reflux workspace is incomplete");
    const auto checked_size = [ncomp](std::int64_t extent) {
      const std::size_t components = static_cast<std::size_t>(ncomp);
      if (extent <= 0 ||
          static_cast<std::uint64_t>(extent) > std::numeric_limits<std::size_t>::max() / components)
        throw std::overflow_error("prepared external Reflux face size overflow");
      return static_cast<std::size_t>(extent) * components;
    };
    PreparedAmrRefluxFaceWorkspace result;
    result.I0 = footprint.lo[0];
    result.I1 = footprint.hi[0];
    result.J0 = footprint.lo[1];
    result.J1 = footprint.hi[1];
    result.components = ncomp;
    result.x_low.resize(checked_size(footprint.ny()));
    result.x_high.resize(result.x_low.size());
    result.y_low.resize(checked_size(footprint.nx()));
    result.y_high.resize(result.y_low.size());
    result.patch_identity = transition_identity + "/patch=" + std::to_string(global_child);
    result.interface_identities = {
        result.patch_identity + "/x-low", result.patch_identity + "/x-high",
        result.patch_identity + "/y-low", result.patch_identity + "/y-high"};
    return result;
  }

  void poison() {
    const Real sentinel = std::numeric_limits<Real>::quiet_NaN();
    for (auto* values : {&x_low, &x_high, &y_low, &y_high})
      std::fill(values->begin(), values->end(), sentinel);
  }

  [[nodiscard]] bool all_finite() const {
    for (const auto* values : {&x_low, &x_high, &y_low, &y_high})
      if (std::any_of(values->begin(), values->end(),
                      [](Real value) { return !std::isfinite(static_cast<double>(value)); }))
        return false;
    return true;
  }

  [[nodiscard]] RefluxFaceCorrectionView view() {
    return {I0, I1, J0, J1, x_low.data(), x_high.data(), y_low.data(), y_high.data(), components};
  }

  [[nodiscard]] RefluxFaceCorrectionConstView view() const {
    return {I0, I1, J0, J1, x_low.data(), x_high.data(), y_low.data(), y_high.data(), components};
  }
};

/// Complete local/noncollective invocation data. Flux strips are already integrated in time and
/// averaged onto coarse faces; the callback may only fill `correction`.
struct PreparedAmrRefluxLocalRequest {
  const std::string* transition_identity = nullptr;
  const std::string* patch_identity = nullptr;
  std::array<const std::string*, 4> interface_identities{};
  int parent_level = -1;
  int child_level = -1;
  std::size_t global_child = 0;
  RefluxStripConstView coarse;
  RefluxStripConstView fine;
  RefluxFaceCorrectionView correction;
  amr::ClockStamp logical_time;
  Real dx = Real(0);
  Real dy = Real(0);
};

using PreparedAmrRefluxLocalKernel = std::function<void(const PreparedAmrRefluxLocalRequest&)>;

/// Prepared spatial reflux storage for one exact Program-owned parent/child transition.  It owns
/// only the interface topology and collective correction register; ProgramGraph supplies the
/// already time-integrated coarse/fine flux strips.
class PreparedAmrProgramRefluxTransition {
 public:
  PreparedAmrProgramRefluxTransition(const PreparedAmrProgramRefluxTransition&) = delete;
  PreparedAmrProgramRefluxTransition& operator=(const PreparedAmrProgramRefluxTransition&) = delete;
  PreparedAmrProgramRefluxTransition(PreparedAmrProgramRefluxTransition&&) noexcept = default;
  PreparedAmrProgramRefluxTransition& operator=(PreparedAmrProgramRefluxTransition&&) noexcept =
      default;

  static PreparedAmrProgramRefluxTransition prepare(const AmrLevelMP& parent,
                                                    const AmrLevelMP& child,
                                                    const Box2D& parent_domain,
                                                    Periodicity periodicity,
                                                    const CommunicatorView& communicator) {
    return prepare_with_local_kernel(parent, child, parent_domain, periodicity, 0,
                                     "pops://runtime/amr/program-reflux/parent=0/child=1", {},
                                     communicator);
  }

  static PreparedAmrProgramRefluxTransition prepare_with_local_kernel(
      const AmrLevelMP& parent, const AmrLevelMP& child, const Box2D& parent_domain,
      Periodicity periodicity, int parent_level, std::string transition_identity,
      PreparedAmrRefluxLocalKernel local_kernel, const CommunicatorView& communicator) {
    if (parent.U.ncomp() != child.U.ncomp())
      throw std::invalid_argument("prepared AMR Program reflux transition component mismatch");
    if (parent_level < 0 || transition_identity.empty())
      throw std::invalid_argument("prepared AMR Program reflux transition identity is incomplete");
    validate_ratio_aligned_disjoint_fine_layout(child.U.box_array(), &parent_domain);
    CoarseFineInterface interface(parent_domain, child.U.box_array(), periodicity);
    std::vector<Box2D> correction_regions = interface.reflux_register_regions(child.U.box_array());
    std::vector<PreparedAmrRefluxFaceWorkspace> local_workspaces(
        static_cast<std::size_t>(child.U.box_array().size()));
    if (local_kernel)
      for (int global_child = 0; global_child < child.U.box_array().size(); ++global_child)
        if (child.U.dmap()[global_child] == communicator.rank())
          local_workspaces[static_cast<std::size_t>(global_child)] =
              PreparedAmrRefluxFaceWorkspace::prepare(
                  PatchRange(child.U.box_array()[global_child]).box(), parent.U.ncomp(),
                  transition_identity, static_cast<std::size_t>(global_child));
    return PreparedAmrProgramRefluxTransition(parent, child, communicator, parent_level,
                                              std::move(transition_identity),
                                              std::move(local_kernel), std::move(local_workspaces),
                                              std::move(interface), std::move(correction_regions));
  }

  template <class CoarseStripRange, class FineStripRange>
  void synchronize_integrated(MultiFab& parent_state, Real dx, Real dy,
                              const CoarseStripRange& coarse_role, const FineStripRange& fine_role,
                              const CommunicatorView& communicator,
                              const amr::ClockStamp* logical_time = nullptr) {
    validate_communicator_(communicator);
    using CoarseStrip = typename CoarseStripRange::value_type;
    using FineStrip = typename FineStripRange::value_type;
    static_assert(std::is_same_v<CoarseStrip, FineStrip>,
                  "AMR Program coarse/fine reflux roles require one strip carrier type");

    // Complete every rank-local metadata check before touching pinned storage.  Owner-only strip
    // corruption is then converted into one communicator-wide decision, so no healthy rank can
    // enter the correction Allreduce while its peer unwinds.
    std::exception_ptr local_failure;
    try {
      if (local_kernel_ &&
          (logical_time == nullptr || logical_time->level != parent_level_ ||
           logical_time->macro_step < 0 || !std::isfinite(logical_time->physical_time)))
        throw std::invalid_argument(
            "prepared external Reflux requires the exact parent logical time");
      validate_parent_state_(parent_state);
      if (coarse_role.size() != child_global_size_ || fine_role.size() != child_global_size_)
        throw std::runtime_error(
            "AMR Program reflux ledger size differs from the prepared child layout "
            "(coarse=" +
            std::to_string(coarse_role.size()) + ", fine=" + std::to_string(fine_role.size()) +
            ", child=" + std::to_string(child_global_size_) + ")");
      for (std::size_t global_child = 0; global_child < child_global_size_; ++global_child) {
        const CoarseStrip& coarse = coarse_role[global_child];
        const FineStrip& fine = fine_role[global_child];
        const bool coarse_present = coarse_role_present_(coarse);
        const bool fine_present = fine_role_present_(fine);
        const bool local_owner = child_ranks_[global_child] == communicator_rank_;
        if (coarse_present != local_owner || fine_present != local_owner)
          throw std::runtime_error(
              "AMR Program reflux roles are not materialized exactly on the child owner");
        if (!local_owner)
          continue;
        const Box2D& expected = child_footprints_[global_child];
        if (!same_strip_footprint_(coarse, expected) || !same_strip_footprint_(fine, expected))
          throw std::runtime_error(
              "AMR Program reflux strip footprint differs from its prepared global child");
        interface_.preflight_reflux_integrated_pair(coarse, fine, dx, dy, correction_, ncomp_);
      }
    } catch (...) {
      local_failure = std::current_exception();
    }
    // Presence, rank-local preflight failure and the later execution branch are decided by one
    // collective bitmask. A rank can therefore never enter the builtin gather while a peer invokes
    // an external callback.
    constexpr char kExternalSelected = char{1};
    constexpr char kBuiltinSelected = char{2};
    constexpr char kPreflightFailed = char{4};
    char preflight_consensus = local_kernel_ ? kExternalSelected : kBuiltinSelected;
    if (local_failure)
      preflight_consensus |= kPreflightFailed;
    all_reduce_or_inplace(&preflight_consensus, std::size_t{1}, communicator);
    const bool provider_mismatch = (preflight_consensus & kExternalSelected) != 0 &&
                                   (preflight_consensus & kBuiltinSelected) != 0;
    if ((preflight_consensus & kPreflightFailed) != 0 || provider_mismatch) {
      if (local_failure)
        std::rethrow_exception(local_failure);
      throw std::runtime_error(provider_mismatch
                                   ? "prepared Reflux provider differs between communicator ranks"
                                   : "AMR Program reflux preflight failed on another "
                                     "communicator rank");
    }
    const bool use_external = (preflight_consensus & kExternalSelected) != 0;

    if (use_external) {
      std::exception_ptr local_failure;
      try {
        correction_.clear_on_device();
        device_fence();
        for (std::size_t global_child = 0; global_child < child_global_size_; ++global_child) {
          const CoarseStrip& coarse = coarse_role[global_child];
          const FineStrip& fine = fine_role[global_child];
          if (!coarse_role_present_(coarse))
            continue;
          PreparedAmrRefluxFaceWorkspace& workspace = local_workspaces_[global_child];
          workspace.poison();
          std::array<const std::string*, 4> interface_identities;
          for (std::size_t face = 0; face < interface_identities.size(); ++face)
            interface_identities[face] = &workspace.interface_identities[face];
          local_kernel_(PreparedAmrRefluxLocalRequest{
              &transition_identity_, &workspace.patch_identity, interface_identities, parent_level_,
              parent_level_ + 1, global_child, reflux_strip_const_view(coarse, ncomp_),
              reflux_strip_const_view(fine, ncomp_), workspace.view(), *logical_time, dx, dy});
          if (!workspace.all_finite())
            throw std::runtime_error(
                "native Reflux component left a non-finite or unwritten correction");
          const PreparedAmrRefluxFaceWorkspace& completed = workspace;
          interface_.route_prepared_reflux_correction_(completed.view(), correction_, ncomp_);
        }
        device_fence();
      } catch (...) {
        local_failure = std::current_exception();
        try {
          device_fence();
        } catch (...) {
        }
      }
      const std::uint64_t rejected =
          all_reduce_max(local_failure ? std::uint64_t(1) : std::uint64_t(0), communicator);
      if (rejected != 0) {
        if (local_failure)
          std::rethrow_exception(local_failure);
        throw std::runtime_error("native Reflux component failed on another communicator rank");
      }
    } else {
      correction_.clear_on_device();
      for (std::size_t global_child = 0; global_child < child_global_size_; ++global_child) {
        const CoarseStrip& coarse = coarse_role[global_child];
        const FineStrip& fine = fine_role[global_child];
        if (!coarse_role_present_(coarse))
          continue;
        interface_.route_reflux_integrated_pair_prevalidated_(coarse, fine, dx, dy, correction_,
                                                              ncomp_);
      }
    }
    try {
      correction_.gather(communicator);
      for (int local_parent = 0; local_parent < parent_state.local_size(); ++local_parent)
        for_each_cell(parent_state.box(local_parent),
                      detail::ApplyRefluxRegisterKernel{parent_state.fab(local_parent).array(),
                                                        correction_.view(), ncomp_});
      device_fence();
    } catch (...) {
      // Pinned reflux storage has no deferred-free arena.  Drain any clear/route/apply kernel before
      // an exceptional caller can destroy the prepared transition.
      try {
        device_fence();
      } catch (...) {
      }
      throw;
    }
  }

 private:
  PreparedAmrProgramRefluxTransition(const AmrLevelMP& parent, const AmrLevelMP& child,
                                     const CommunicatorView& communicator, int parent_level,
                                     std::string transition_identity,
                                     PreparedAmrRefluxLocalKernel local_kernel,
                                     std::vector<PreparedAmrRefluxFaceWorkspace> local_workspaces,
                                     CoarseFineInterface interface,
                                     std::vector<Box2D> correction_regions)
      : parent_boxes_(parent.U.box_array().boxes()),
        parent_ranks_(parent.U.dmap().ranks()),
        child_footprints_(make_child_footprints_(child.U.box_array())),
        child_ranks_(child.U.dmap().ranks()),
        parent_ngrow_(parent.U.n_grow()),
        ncomp_(parent.U.ncomp()),
        child_global_size_(static_cast<std::size_t>(child.U.box_array().size())),
        communicator_size_(communicator.size()),
        communicator_rank_(communicator.rank()),
        communicator_identity_(detail::parallel_copy_communicator_identity(communicator)),
        parent_level_(parent_level),
        transition_identity_(std::move(transition_identity)),
        local_kernel_(std::move(local_kernel)),
        local_workspaces_(std::move(local_workspaces)),
        interface_(std::move(interface)),
        correction_(std::move(correction_regions), ncomp_) {
    if (child_footprints_.size() != child_global_size_ || child_ranks_.size() != child_global_size_)
      throw std::invalid_argument(
          "prepared AMR Program reflux child layout metadata is inconsistent");
    for (int owner : child_ranks_)
      if (owner < 0 || owner >= communicator_size_)
        throw std::invalid_argument(
            "prepared AMR Program reflux child owner lies outside the communicator");
    if (parent_level_ < 0 || transition_identity_.empty() ||
        local_workspaces_.size() != child_global_size_)
      throw std::invalid_argument(
          "prepared AMR Program reflux local-provider metadata is inconsistent");
  }

  static std::vector<Box2D> make_child_footprints_(const BoxArray& child_boxes) {
    std::vector<Box2D> footprints;
    footprints.reserve(static_cast<std::size_t>(child_boxes.size()));
    for (const Box2D& child : child_boxes.boxes())
      footprints.push_back(PatchRange(child).box());
    return footprints;
  }

  template <class Strip>
  static bool coarse_role_present_(const Strip& strip) noexcept {
    return !strip.cL.empty() || !strip.cR.empty() || !strip.cB.empty() || !strip.cT.empty();
  }

  template <class Strip>
  static bool fine_role_present_(const Strip& strip) noexcept {
    return !strip.fL.empty() || !strip.fR.empty() || !strip.fB.empty() || !strip.fT.empty();
  }

  template <class Strip>
  static bool same_strip_footprint_(const Strip& strip, const Box2D& expected) noexcept {
    return strip.I0 == expected.lo[0] && strip.I1 == expected.hi[0] && strip.J0 == expected.lo[1] &&
           strip.J1 == expected.hi[1];
  }

  void validate_communicator_(const CommunicatorView& communicator) const {
    if (communicator.size() != communicator_size_ || communicator.rank() != communicator_rank_ ||
        detail::parallel_copy_communicator_identity(communicator) != communicator_identity_)
      throw std::invalid_argument(
          "prepared AMR Program reflux transition changed execution communicator");
  }

  void validate_parent_state_(const MultiFab& parent) const {
    if (parent.box_array().boxes() != parent_boxes_ || parent.dmap().ranks() != parent_ranks_ ||
        parent.n_grow() != parent_ngrow_ || parent.ncomp() != ncomp_)
      throw std::invalid_argument(
          "prepared AMR Program reflux transition crossed an exact parent layout");
  }

  std::vector<Box2D> parent_boxes_;
  std::vector<int> parent_ranks_;
  std::vector<Box2D> child_footprints_;
  std::vector<int> child_ranks_;
  int parent_ngrow_ = 0;
  int ncomp_ = 0;
  std::size_t child_global_size_ = 0;
  int communicator_size_ = 1;
  int communicator_rank_ = 0;
  std::int64_t communicator_identity_ = 0;
  int parent_level_ = 0;
  std::string transition_identity_;
  PreparedAmrRefluxLocalKernel local_kernel_;
  std::vector<PreparedAmrRefluxFaceWorkspace> local_workspaces_;
  CoarseFineInterface interface_;
  FluxRegister correction_;
};

/// Whole-hierarchy spatial reflux plan consumed only by ProgramGraph's conservative ledger commit.
/// It owns no candidate hierarchy, stage state, time method or attempt/publish lifecycle.
class PreparedAmrProgramRefluxPlan {
 public:
  PreparedAmrProgramRefluxPlan(const PreparedAmrProgramRefluxPlan&) = delete;
  PreparedAmrProgramRefluxPlan& operator=(const PreparedAmrProgramRefluxPlan&) = delete;
  PreparedAmrProgramRefluxPlan(PreparedAmrProgramRefluxPlan&&) noexcept = default;
  PreparedAmrProgramRefluxPlan& operator=(PreparedAmrProgramRefluxPlan&&) noexcept = default;

  static PreparedAmrProgramRefluxPlan prepare(
      const std::vector<AmrLevelMP>& levels, const Box2D& base_domain, Periodicity periodicity,
      std::uint64_t topology_generation,
      const CommunicatorView& communicator = world_communicator_view(),
      PreparedAmrRefluxLocalKernel local_kernel = {}, std::string block_identity = {}) {
    if (levels.empty() || base_domain.empty())
      throw std::invalid_argument("prepared AMR Program reflux requires a non-empty hierarchy");
    if (local_kernel && block_identity.empty())
      throw std::invalid_argument("prepared external Reflux requires one qualified block identity");
    std::vector<PreparedAmrProgramRefluxTransition> transitions;
    transitions.reserve(levels.size() - 1);
    for (std::size_t parent = 0; parent + 1 < levels.size(); ++parent) {
      const std::string transition_identity =
          (block_identity.empty() ? "pops://runtime/amr/program-reflux" : block_identity) +
          "/parent=" + std::to_string(parent) + "/child=" + std::to_string(parent + 1);
      transitions.push_back(PreparedAmrProgramRefluxTransition::prepare_with_local_kernel(
          levels[parent], levels[parent + 1],
          amr_level_index_domain(base_domain, static_cast<int>(parent)), periodicity,
          static_cast<int>(parent), transition_identity, local_kernel, communicator));
    }
    return PreparedAmrProgramRefluxPlan(static_cast<int>(levels.size()), topology_generation,
                                        std::move(transitions));
  }

  PreparedAmrProgramRefluxTransition& transition_for_child(int child_level,
                                                           std::uint64_t topology_generation) {
    if (topology_generation != topology_generation_)
      throw std::invalid_argument("prepared AMR Program reflux crossed a topology generation");
    if (child_level <= 0 || child_level >= nlevels_)
      throw std::out_of_range("prepared AMR Program reflux child level is out of range");
    return transitions_.at(static_cast<std::size_t>(child_level - 1));
  }
  [[nodiscard]] std::uint64_t topology_generation() const noexcept { return topology_generation_; }

 private:
  PreparedAmrProgramRefluxPlan(int nlevels, std::uint64_t topology_generation,
                               std::vector<PreparedAmrProgramRefluxTransition> transitions)
      : nlevels_(nlevels),
        topology_generation_(topology_generation),
        transitions_(std::move(transitions)) {}

  int nlevels_ = 0;
  std::uint64_t topology_generation_ = 0;
  std::vector<PreparedAmrProgramRefluxTransition> transitions_;
};

}  // namespace pops
