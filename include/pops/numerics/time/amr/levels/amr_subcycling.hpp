#pragma once
#include <pops/core/foundation/validation.hpp>
#include <pops/mesh/storage/mf_arith.hpp>  // saxpy, lincomb (SSPRK stages, named device-clean functors)
#include <pops/amr/hierarchy/refinement_ratio.hpp>
#include <pops/mesh/layout/refinement.hpp>  // coarsen, parallel_copy
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/time/amr/reflux/amr_flux_helpers.hpp>
#include <pops/numerics/time/amr/levels/amr_patch_range.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>

/// @file
/// @brief AMR multi-patch subcycling engine (several fine boxes per level): 2-level step
///        (amr_step_2level_multipatch), N-level recursion (detail::subcycle_level_mp,
///        detail::amr_step_multilevel_multipatch), SSPRK2/SSPRK3 per-stage advance, multi-box helpers
///        (mf_fill_fine_ghosts_mb, mf_average_down_mb, mf_find_box, coarsen_grown) and types
///        AmrLevelMP / RegMP. This is the engine behind advance_amr.
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
/// - SSPRK2/SSPRK3 refill ghosts BEFORE each stage flux evaluation
///   (ssprk_refill_level_ghosts), and require imex == false;
/// - saxpy/lincomb and the helper kernels are device-clean (named functors).

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

using AmrPhysicalBoundaryFill =
    std::function<void(MultiFab&, const AmrBoundaryFillContext&)>;

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
      Periodicity{boundary.xlo == BCType::Periodic, boundary.ylo == BCType::Periodic},
      0,
      true,
      [prepared](MultiFab& state, const AmrBoundaryFillContext& context) mutable {
        prepared.dx = context.dx;
        prepared.dy = context.dy;
        fill_physical_bc(state, context.domain, prepared);
      }};
}

inline void validate_amr_boundary_fill_authority(
    Periodicity periodicity, const AmrBoundaryFillAuthority* authority) {
  const bool has_physical_face = !periodicity.x || !periodicity.y;
  if (authority == nullptr) {
    if (has_physical_face)
      throw std::runtime_error(
          "non-periodic AMR advance requires an explicit physical boundary-fill authority");
    return;
  }
  if (!same_periodicity(periodicity, authority->periodicity))
    throw std::runtime_error("AMR boundary-fill authority periodicity disagrees with the hierarchy");
  if (authority->provided_depth < 0 || (has_physical_face && !authority->fill_physical))
    throw std::runtime_error("AMR boundary-fill authority is incomplete");
}

template <class Levels>
inline void validate_amr_boundary_fill_authority(
    Periodicity periodicity, const AmrBoundaryFillAuthority* authority,
    const Levels& levels) {
  validate_amr_boundary_fill_authority(periodicity, authority);
  if (authority == nullptr)
    return;
  for (const auto& level : levels)
    if (!authority->fills_all_allocated_ghosts &&
        authority->provided_depth < level.U.n_grow())
      throw std::runtime_error("AMR boundary-fill authority does not cover all state ghosts");
}

inline void fill_amr_same_level_and_physical(
    MultiFab& state, const Box2D& domain, int level, Real dx, Real dy,
    Periodicity periodicity, const AmrBoundaryFillAuthority* authority) {
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
      throw std::runtime_error(
          "physical AMR boundary callback failed on at least one MPI rank");
    }
  }
}

// --- MULTI-PATCH (several fine boxes per level) ---
// The fine level is a MultiFab with N boxes. Reflux is COVERAGE-AWARE: it corrects a coarse
// cell adjacent to a fine box only if it is NOT covered by another fine box (real fine-coarse
// interface; fine-fine interfaces are handled by fill_boundary). This is AMReX FluxRegister
// logic.

// Conservative 2-level step, MULTI-BOX fine level. Uc: single-box coarse (periodic).
// Uf: MultiFab with N fine boxes (ratio 2, strictly interior, coarse-aligned).
//
// Distributed (MPI) with COARSE REPLICATION. The single-box coarse level is replicated: each
// rank holds an identical copy (per-rank DistributionMapping, or deterministic init). The coarse
// advance (self-periodic fill_boundary, flux, advance) runs identically on each copy; the fine
// patches are distributed. reflux (addition to bordering cells) and average_down (overwrite of covered
// bordering cells) gather up through two global-indexed coarse buffers + all_reduce_sum_inplace,
// then each rank applies to its copy -> all stay identical. In serial this is bit-for-bit
// identical to the direct path (see the final block). Validation: test_mpi_amr_multipatch
// (np=1/2/4 bit-identical). The coarse level is small (base level), so the replication is
// accepted; the recursive N-level path (subcycle_level_mp) still has to be generalized the same
// way (ROADMAP).
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void amr_step_2level_multipatch(const Model& m, MultiFab& Uc, const Box2D& dom, Real dxc, Real dyc,
                                MultiFab& Uf, const MultiFab& auxc, const MultiFab& auxf, Real dt,
                                Periodicity base_periodicity,
                                const AmrBoundaryFillAuthority* boundary_fill = nullptr) {
  validate_amr_boundary_fill_authority(base_periodicity, boundary_fill);
  if (boundary_fill != nullptr && !boundary_fill->fills_all_allocated_ghosts &&
      (boundary_fill->provided_depth < Uc.n_grow() ||
       boundary_fill->provided_depth < Uf.n_grow()))
    throw std::runtime_error("AMR boundary-fill authority does not cover all state ghosts");
  // The complete coarse/fine attempt is private.  A failed fine substep must not leave the
  // coarse level advanced, and a failed coarse source must not even refresh the public ghosts.
  // MultiFab copies own their storage; the final moves below are the sole publication point.
  MultiFab coarse = Uc;
  MultiFab fine = Uf;
  const SubcyclingSchedule sched(0, 1, amr::Rational(kAmrRefRatio, 1),
                                 amr::RemainderPolicy::IntegralOnly);
  const int nc = coarse.ncomp();
  const Real dxf = dxc / kAmrRefRatio, dyf = dyc / kAmrRefRatio, dtf = sched.dt_sub(dt);

  // coarse-fine interface: coverage (coarse cells shadowed by a fine patch) + bordering reflux
  // routing. Coverage built on the GLOBAL BoxArray (all boxes, known to all ranks) -> correct
  // under MPI.
  const CoarseFineInterface cfi(dom, fine.box_array(), base_periodicity);

  MultiFab Uc_old = coarse;
  fill_periodic_local(coarse, dom, base_periodicity);  // replicated coarse -> local periodic fill
  if ((!base_periodicity.x || !base_periodicity.y) && boundary_fill != nullptr)
    boundary_fill->fill_physical(
        coarse, AmrBoundaryFillContext{dom, 0, dxc, dyc});
  MultiFab fxc(BoxArray(std::vector<Box2D>{xface_box(coarse.box(0))}), coarse.dmap(), nc, 0);
  MultiFab fyc(BoxArray(std::vector<Box2D>{yface_box(coarse.box(0))}), coarse.dmap(), nc, 0);
  compute_face_fluxes<Limiter, NumericalFlux>(m, coarse, auxc, fxc, fyc, dxc, dyc);
  mf_advance_faces(coarse, fxc, fyc, dxc, dyc, dt);
  mf_apply_source(m, coarse, auxc, dt);  // source S(U,aux) at the substep
  // One collective covers every value the accepted coarse attempt would publish or record.
  detail::reject_nonfinite_finite_volume_data("amr_step_2level_multipatch(coarse)", coarse, fxc,
                                               fyc);

  // per fine-box register: coarse flux (without dt) saved at the 4 faces.
  struct Reg {
    int I0, I1, J0, J1;
    RefluxStorage<Real> cL, cR, cB, cT, fL, fR, fB, fT;
  };
  std::vector<Reg> regs(fine.local_size());
  {
    const ConstArray4 FX = fxc.fab(0).const_array(), FY = fyc.fab(0).const_array();
    for (int li = 0; li < fine.local_size(); ++li) {
      const PatchRange pr(fine.box(li));
      Reg& g = regs[li];
      g.I0 = pr.I0;
      g.I1 = pr.I1;
      g.J0 = pr.J0;
      g.J1 = pr.J1;
      const int nJ = g.J1 - g.J0 + 1, nI = g.I1 - g.I0 + 1;
      g.cL.assign(nJ * nc, 0);
      g.cR.assign(nJ * nc, 0);
      g.cB.assign(nI * nc, 0);
      g.cT.assign(nI * nc, 0);
      g.fL.assign(nJ * nc, 0);
      g.fR.assign(nJ * nc, 0);
      g.fB.assign(nI * nc, 0);
      g.fT.assign(nI * nc, 0);
      sample_coarse_strip(FX, FX, FY, FY, reflux_strip_view(g, nc));
    }
  }
  // multi-box fine fluxes: one face-box per GLOBAL fine box, same dmap as Uf. Built on the global
  // box_array() (not the local boxes) so that BoxArray and DistributionMapping have the same size
  // under MPI: fxf.fab(li) then corresponds to Uf.fab(li) (same dmap, same global order). In
  // serial it is identical (local == global).
  std::vector<Box2D> fxb, fyb;
  for (int g = 0; g < fine.box_array().size(); ++g) {
    fxb.push_back(xface_box(fine.box_array()[g]));
    fyb.push_back(yface_box(fine.box_array()[g]));
  }
  MultiFab fxf(BoxArray(std::move(fxb)), fine.dmap(), nc, 0);
  MultiFab fyf(BoxArray(std::move(fyb)), fine.dmap(), nc, 0);
  const Box2D fdom = dom.refine(kAmrRefRatio);

  for (int s = 0; s < sched.count(); ++s) {
    mf_fill_fine_ghosts_multi(fine, Uc_old, coarse, dom, sched.frac(s));
    fill_amr_same_level_and_physical(fine, fdom, 1, dxf, dyf, base_periodicity, boundary_fill);
    compute_face_fluxes<Limiter, NumericalFlux>(m, fine, auxf, fxf, fyf, dxf, dyf);
    mf_advance_faces(fine, fxf, fyf, dxf, dyf, dtf);
    mf_apply_source(m, fine, auxf, dtf);  // source S(U,aux) at the substep
    // Validate before this substep contributes to the time-integrated fine register.
    detail::reject_nonfinite_finite_volume_data("amr_step_2level_multipatch(fine)", fine, fxf,
                                                 fyf);
    for (int li = 0; li < fine.local_size(); ++li) {
      Reg& g = regs[li];
      const ConstArray4 FX = fxf.fab(li).const_array(), FY = fyf.fab(li).const_array();
      accumulate_fine_strip(FX, FY, reflux_strip_view(g, nc), dtf);
    }
  }

  // DISTRIBUTED reflux THEN average_down, the coarse level being REPLICATED (each rank holds an
  // identical copy after the deterministic coarse advance). Each rank deposits, for its LOCAL
  // fine patches, into two global-indexed coarse buffers:
  //   avg: the average-down over the COVERED cells (overwrite semantics; a single contribution
  //        per cell since the patches are disjoint);
  //   ref: the reflux correction on the uncovered BORDERING cells (addition).
  // all_reduce_sum -> each rank has the total, then applies to ITS copy: bordering += ref,
  // then covered = avg. All copies stay identical. In serial (np=1) the all-reduce is the identity
  // and it is bit-for-bit identical to the direct path (0 + average = average exactly; advance +
  // correction). Cost follows the exact covered cells plus their one-cell reflux halo.
  device_fence();
  // Registers use the exact sparse coarse-fine interface (covered footprints plus their one-cell
  // reflux halo, clamped/canonicalized by the interface).  Far-apart patches therefore do not
  // materialize the holes of a bounding box.  Cells outside the interface remain implicit zero.
  const std::vector<Box2D> register_regions = cfi.reflux_register_regions(fine.box_array());
  FluxRegister avg(register_regions, nc);  // average-down (overwrite of covered cells)
  FluxRegister ref(register_regions, nc);  // reflux (addition to bordering cells)
  for (int li = 0; li < fine.local_size(); ++li) {
    const ConstArray4 f = fine.fab(li).const_array();
    Reg& g = regs[li];
    for_each_cell(g.I0 <= g.I1 && g.J0 <= g.J1 ? Box2D{{g.I0, g.J0}, {g.I1, g.J1}}
                                                : Box2D{},
                  detail::AverageDownRegisterKernel{f, avg.view(), nc});
  }
  // The edge-strip kernels write pinned device-visible buffers.  Reflux routing is a sequence of
  // named device kernels in stable patch/face order; finish strip production before enqueuing it.
  device_fence();
  for (int li = 0; li < fine.local_size(); ++li) {
    Reg& g = regs[li];
    cfi.route_reflux(g, dxc, dyc, dt, ref, nc);  // coverage-aware bordering reflux
  }
  avg.gather();
  ref.gather();
  if (coarse.local_size() > 0) {  // each rank holding a copy of the coarse level applies it
    const Box2D cb = coarse.box(0);
    for_each_cell(cb, detail::ApplyRefluxThenAverageKernel{
                          coarse.fab(0).array(), ref.view(), avg.view(), cfi.cmask.view(), nc});
  }
  // Reflux and average-down are arithmetic too: finite stage data may still overflow while being
  // accumulated or synchronized.  Reject the complete candidate before either public state moves.
  detail::reject_nonfinite_finite_volume_data(
      "amr_step_2level_multipatch(synchronization)", coarse, fine);
  device_fence();
  Uc = std::move(coarse);
  Uf = std::move(fine);
}

// --- N-LEVEL MULTI-PATCH (multi-box at EACH level) ---
// Generalizes subcycle_level_mf: each level is a multi-box MultiFab. Reflux (FluxRegister) is
// coverage-aware AND routes the correction to the PARENT box containing the adjacent coarse cell.
// Reduces BIT-FOR-BIT to the single-box path when each level has only one box (validation guard).
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

  static PreparedFillPatchWorkspace prepare(
      const MultiFab& fine, const MultiFab& old_parent, const MultiFab& new_parent,
      const Box2D& coarse_domain, bool replicated_parent, Periodicity periodicity,
      std::uint64_t topology_generation, const CommunicatorView& communicator) {
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
      throw std::invalid_argument(
          "replicated FillPatch parent requires a rank-local communicator");

    // The carrier contains every source cell used by the selected route, including the furthest
    // one-sided stencil next to a non-periodic boundary.  This is prepared once per topology.
    const int reach =
        std::max(prepared_operator->parent_reach_x, prepared_operator->parent_reach_y);
    const int fine_growth = detail::checked_coarse_fine_carrier_growth(
        fine.n_grow(), kAmrRefRatio, reach);
    const BoxArray carrier_boxes =
        coarsen_grown(fine.box_array(), fine_growth, kAmrRefRatio);
    for (const Box2D& box : carrier_boxes.boxes())
      if (box.nx() < prepared_operator->minimum_axis_cells_x ||
          box.ny() < prepared_operator->minimum_axis_cells_y)
        throw std::invalid_argument(
            "prepared FillPatch carrier cannot hold the selected directional stencil");
    DistributionMapping carrier_mapping = fine.dmap();
    if (replicated_parent)
      carrier_mapping = DistributionMapping(
          std::vector<int>(static_cast<std::size_t>(carrier_boxes.size()), my_rank()));

    PreparedFillPatchWorkspace workspace(
        MultiFab(carrier_boxes, carrier_mapping, fine.ncomp(), 0),
        MultiFab(carrier_boxes, carrier_mapping, fine.ncomp(), 0), fine, old_parent, new_parent,
        coarse_domain, replicated_parent, periodicity, topology_generation,
        std::move(prepared_operator));
    workspace.old_copy_plan_.emplace(PreparedPeriodicCopyPlan::prepare(
        workspace.old_parent_carrier_, old_parent, coarse_domain, periodicity,
        topology_generation, communicator));
    workspace.new_copy_plan_.emplace(PreparedPeriodicCopyPlan::prepare(
        workspace.new_parent_carrier_, new_parent, coarse_domain, periodicity,
        topology_generation, communicator));
    workspace.validate_carrier_ownership_(fine);
    return workspace;
  }

  void apply(MultiFab& fine, const MultiFab& old_parent, const MultiFab& new_parent, Real fraction,
             Real positivity_floor, int positivity_component,
             std::uint64_t topology_generation, const CommunicatorView& communicator) {
    validate_replay_(fine, old_parent, new_parent, fraction, positivity_floor,
                     positivity_component, topology_generation);
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

  [[nodiscard]] std::uint64_t topology_generation() const noexcept {
    return topology_generation_;
  }
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
                   fine_domain_.lo[1], kAmrRefRatio, kAmrRefRatio},
        replicated_parent_(replicated_parent),
        periodicity_(periodicity),
        topology_generation_(topology_generation),
        prepared_operator_(std::move(prepared_operator)) {}

  static void validate_temporal_window_(const MultiFab& fine, const MultiFab& old_parent,
                                        const MultiFab& new_parent,
                                        const Box2D& coarse_domain) {
    if (coarse_domain.empty())
      throw std::invalid_argument("FillPatch requires a non-empty coarse domain");
    if (fine.ncomp() <= 0 || old_parent.ncomp() != fine.ncomp() ||
        new_parent.ncomp() != fine.ncomp())
      throw std::invalid_argument("FillPatch parent/child component mismatch");
    if (old_parent.box_array().boxes() != new_parent.box_array().boxes() ||
        old_parent.dmap().ranks() != new_parent.dmap().ranks() ||
        old_parent.n_grow() != new_parent.n_grow())
      throw std::invalid_argument(
          "FillPatch old/new parent snapshots require one exact layout");
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
    if (!old_copy_plan_ || !new_copy_plan_ ||
        old_parent.box_array().boxes() != old_parent_boxes_ ||
        old_parent.dmap().ranks() != old_parent_ranks_ ||
        old_parent.ncomp() != fine_ncomp_ || old_parent.n_grow() != old_parent_ngrow_ ||
        new_parent.box_array().boxes() != new_parent_boxes_ ||
        new_parent.dmap().ranks() != new_parent_ranks_ ||
        new_parent.ncomp() != fine_ncomp_ || new_parent.n_grow() != new_parent_ngrow_)
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
          fine.fab(local_fine).array(),
          old_parent_carrier_.fab(old_carrier_local).const_array(),
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
                                   const Box2D& coarse_domain, Real frac,
                                   bool replicated_parent, Real pos_floor, int pos_comp,
                                   Periodicity periodicity) {
  const CommunicatorView communicator =
      replicated_parent ? CommunicatorView{} : world_communicator_view();
  auto workspace = PreparedFillPatchWorkspace::prepare(
      Uf, Po, Pn, coarse_domain, replicated_parent, periodicity,
      /*topology_generation=*/0, communicator);
  workspace.publish_prepared(Uf, frac, pos_floor, pos_comp);
}

/// Allocation-free FillPatch replay used by prepared runtimes.  The owning hierarchy keeps one
/// workspace per parent/child transition and replaces it only after a topology-generation change.
inline void mf_fill_fine_ghosts_mb(
    MultiFab& fine, const MultiFab& old_parent, const MultiFab& new_parent,
    PreparedFillPatchWorkspace& workspace, Real fraction, Real positivity_floor,
    int positivity_component, std::uint64_t topology_generation,
    const CommunicatorView& communicator) {
  workspace.apply(fine, old_parent, new_parent, fraction, positivity_floor,
                  positivity_component, topology_generation, communicator);
}

// Prepared coarse/fine spatial transfer. Unlike mf_fill_fine_ghosts_mb this operation has exactly
// one physical parent snapshot: time interpolation is a separate prepared route with explicit
// TimePoints. Keeping the two protocols separate prevents callers from manufacturing a fake
// `(parent, parent)` temporal window merely to request conservative spatial ghost materialization.
inline void mf_fill_fine_ghosts_spatial_mb(MultiFab& Uf, const MultiFab& parent,
                                           const Box2D& coarse_domain,
                                           bool replicated_parent, Periodicity periodicity) {
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
        coarse.box_array().boxes() != coarse_boxes_ ||
        coarse.dmap().ranks() != coarse_ranks_ || coarse.n_grow() != coarse_ngrow_ ||
        coarse.ncomp() != ncomp_)
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
  auto workspace = PreparedAverageDownWorkspace::prepare(
      Uf, Uc, /*topology_generation=*/0);
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
                                          const Box2D& base_domain,
                                          Periodicity periodicity, bool coarse_replicated,
                                          std::uint64_t topology_generation) {
    return prepare(levels, base_domain, periodicity, coarse_replicated, topology_generation,
                   std::make_shared<const PreparedCoarseFineOperator>(
                       prepare_limited_linear_coarse_fine_operator()));
  }

  static PreparedAmrFillPatchPlan prepare(
      const std::vector<AmrLevelMP>& levels, const Box2D& base_domain,
      Periodicity periodicity, bool coarse_replicated, std::uint64_t topology_generation,
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
          amr_level_index_domain(base_domain, static_cast<int>(child - 1)),
          replicated_parent, periodicity, topology_generation, communicator,
          prepared_operator));
    }
    return PreparedAmrFillPatchPlan(static_cast<int>(levels.size()), base_domain, periodicity,
                                    coarse_replicated, topology_generation,
                                    std::move(transitions), std::move(prepared_operator));
  }

  PreparedFillPatchWorkspace& transition_for_child(int child_level) {
    if (child_level <= 0 || child_level >= nlevels_)
      throw std::out_of_range("prepared AMR FillPatch child level is out of range");
    return transitions_.at(static_cast<std::size_t>(child_level - 1));
  }

  [[nodiscard]] int nlevels() const noexcept { return nlevels_; }
  [[nodiscard]] std::uint64_t topology_generation() const noexcept {
    return topology_generation_;
  }
  [[nodiscard]] const std::shared_ptr<const PreparedCoarseFineOperator>& prepared_operator()
      const noexcept {
    return prepared_operator_;
  }

  void validate_hierarchy_contract(int nlevels, const Box2D& base_domain,
                                   Periodicity periodicity, bool coarse_replicated,
                                   const std::shared_ptr<const PreparedCoarseFineOperator>&
                                       prepared_operator) const {
    if (nlevels != nlevels_ || base_domain != base_domain_ ||
        periodicity.x != periodicity_.x || periodicity.y != periodicity_.y ||
        coarse_replicated != coarse_replicated_ ||
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
      throw std::invalid_argument(
          "prepared AMR average-down plan requires a non-empty hierarchy");
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
  [[nodiscard]] std::uint64_t topology_generation() const noexcept {
    return topology_generation_;
  }

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

// per child-patch register (PARENT coords I0..J1). c* = coarse flux (without dt);
// f* = time-integrated fine flux accumulated by the child during subcycling.
struct RegMP {
  int I0, I1, J0, J1;
  RefluxStorage<Real> cL, cR, cB, cT, fL, fR, fB, fT;
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

inline BoxArray amr_face_boxes(const BoxArray& cells, bool x_direction) {
  std::vector<Box2D> faces;
  faces.reserve(static_cast<std::size_t>(cells.size()));
  for (const Box2D& cell_box : cells.boxes())
    faces.push_back(x_direction ? xface_box(cell_box) : yface_box(cell_box));
  return BoxArray(std::move(faces));
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

inline void clear_reflux_strip_on_device(RegMP& strip) {
  clear_reflux_storage_on_device(strip.cL);
  clear_reflux_storage_on_device(strip.cR);
  clear_reflux_storage_on_device(strip.cB);
  clear_reflux_storage_on_device(strip.cT);
  clear_reflux_storage_on_device(strip.fL);
  clear_reflux_storage_on_device(strip.fR);
  clear_reflux_storage_on_device(strip.fB);
  clear_reflux_storage_on_device(strip.fT);
}

}  // namespace detail

/// Persistent numerical storage for one exact AMR level.  All buffers used by Euler, SSPRK2,
/// SSPRK3 and the optional HLL speed cache are materialized at preparation; replay only overwrites
/// them through device kernels.
class PreparedAmrLevelAdvanceScratch {
 public:
  PreparedAmrLevelAdvanceScratch(const PreparedAmrLevelAdvanceScratch&) = delete;
  PreparedAmrLevelAdvanceScratch& operator=(const PreparedAmrLevelAdvanceScratch&) = delete;
  PreparedAmrLevelAdvanceScratch(PreparedAmrLevelAdvanceScratch&&) noexcept = default;
  PreparedAmrLevelAdvanceScratch& operator=(PreparedAmrLevelAdvanceScratch&&) noexcept = default;

  static PreparedAmrLevelAdvanceScratch prepare(const AmrLevelMP& level,
                                                bool wave_speed_cache) {
    if (level.aux == nullptr)
      throw std::invalid_argument("prepared AMR advance requires a level auxiliary field");
    if (level.U.ncomp() <= 0 || level.U.box_array().size() == 0)
      throw std::invalid_argument("prepared AMR advance requires a non-empty state layout");
    return PreparedAmrLevelAdvanceScratch(level, wave_speed_cache);
  }

  void validate(const AmrLevelMP& level, bool wave_speed_cache) const {
    if (level.aux == nullptr || level.U.box_array().boxes() != state_boxes_ ||
        level.U.dmap().ranks() != state_ranks_ || level.U.ncomp() != ncomp_ ||
        level.U.n_grow() != state_ngrow_ || level.aux->box_array().boxes() != aux_boxes_ ||
        level.aux->dmap().ranks() != aux_ranks_ || level.aux->ncomp() != aux_ncomp_ ||
        level.aux->n_grow() != aux_ngrow_ || level.dx != dx_ || level.dy != dy_)
      throw std::invalid_argument("prepared AMR level scratch crossed an exact layout");
    if (wave_speed_cache != wave_speed_cache_enabled_)
      throw std::invalid_argument("prepared AMR level scratch changed HLL cache policy");
  }

  void capture_start(const MultiFab& state) { detail::copy_amr_storage(start_state_, state); }

  MultiFab& start_state() noexcept { return start_state_; }
  MultiFab& rhs() noexcept { return rhs_; }
  MultiFab& flux_x() noexcept { return flux_x_; }
  MultiFab& flux_y() noexcept { return flux_y_; }
  MultiFab& stage_flux_x() noexcept { return stage_flux_x_; }
  MultiFab& stage_flux_y() noexcept { return stage_flux_y_; }
  MultiFab* wave_speed_cache() noexcept {
    return wave_speed_cache_enabled_ ? &wave_speed_cache_ : nullptr;
  }

 private:
  PreparedAmrLevelAdvanceScratch(const AmrLevelMP& level, bool wave_speed_cache)
      : state_boxes_(level.U.box_array().boxes()),
        state_ranks_(level.U.dmap().ranks()),
        aux_boxes_(level.aux->box_array().boxes()),
        aux_ranks_(level.aux->dmap().ranks()),
        state_ngrow_(level.U.n_grow()),
        aux_ngrow_(level.aux->n_grow()),
        ncomp_(level.U.ncomp()),
        aux_ncomp_(level.aux->ncomp()),
        dx_(level.dx),
        dy_(level.dy),
        wave_speed_cache_enabled_(wave_speed_cache),
        start_state_(level.U.box_array(), level.U.dmap(), ncomp_, state_ngrow_),
        rhs_(level.U.box_array(), level.U.dmap(), ncomp_, 0),
        flux_x_(detail::amr_face_boxes(level.U.box_array(), true), level.U.dmap(), ncomp_, 0),
        flux_y_(detail::amr_face_boxes(level.U.box_array(), false), level.U.dmap(), ncomp_, 0),
        stage_flux_x_(detail::amr_face_boxes(level.U.box_array(), true), level.U.dmap(), ncomp_, 0),
        stage_flux_y_(detail::amr_face_boxes(level.U.box_array(), false), level.U.dmap(), ncomp_, 0),
        wave_speed_cache_(wave_speed_cache
                              ? MultiFab(level.U.box_array(), level.U.dmap(), 4, 1)
                              : MultiFab()) {
    // Transition preparation deliberately warms distributed face-copy schedules.  Give every
    // persistent carrier a defined device value before that cold-path copy so neither sanitizers nor
    // a non-host execution space can observe uninitialised storage during materialisation.
    start_state_.set_val(Real(0));
    rhs_.set_val(Real(0));
    flux_x_.set_val(Real(0));
    flux_y_.set_val(Real(0));
    stage_flux_x_.set_val(Real(0));
    stage_flux_y_.set_val(Real(0));
    if (wave_speed_cache_enabled_)
      wave_speed_cache_.set_val(Real(0));
  }

  std::vector<Box2D> state_boxes_;
  std::vector<int> state_ranks_;
  std::vector<Box2D> aux_boxes_;
  std::vector<int> aux_ranks_;
  int state_ngrow_ = 0;
  int aux_ngrow_ = 0;
  int ncomp_ = 0;
  int aux_ncomp_ = 0;
  Real dx_ = Real(0);
  Real dy_ = Real(0);
  bool wave_speed_cache_enabled_ = false;
  MultiFab start_state_;
  MultiFab rhs_;
  MultiFab flux_x_;
  MultiFab flux_y_;
  MultiFab stage_flux_x_;
  MultiFab stage_flux_y_;
  MultiFab wave_speed_cache_;
};

/// Persistent coarse/fine storage for one exact parent/child transition.  It owns the routed
/// coarse face carriers, per-local-child strips, sparse parent lookup, interface coverage and the
/// collective correction register.  Preparation warms distributed copies; replay clears and
/// reuses all storage without rebuilding topology-dependent data.
class PreparedAmrTransitionAdvanceScratch {
 public:
  PreparedAmrTransitionAdvanceScratch(const PreparedAmrTransitionAdvanceScratch&) = delete;
  PreparedAmrTransitionAdvanceScratch& operator=(const PreparedAmrTransitionAdvanceScratch&) =
      delete;
  PreparedAmrTransitionAdvanceScratch(PreparedAmrTransitionAdvanceScratch&&) noexcept = default;
  PreparedAmrTransitionAdvanceScratch& operator=(PreparedAmrTransitionAdvanceScratch&&) noexcept =
      default;

  static PreparedAmrTransitionAdvanceScratch prepare(
      const AmrLevelMP& parent, const AmrLevelMP& child, const MultiFab& parent_flux_x,
      const MultiFab& parent_flux_y, const Box2D& parent_domain, Periodicity periodicity,
      bool replicated_parent, std::uint64_t topology_generation,
      const CommunicatorView& communicator) {
    if (parent.U.ncomp() != child.U.ncomp())
      throw std::invalid_argument("prepared AMR transition component mismatch");
    validate_ratio_aligned_disjoint_fine_layout(child.U.box_array(), &parent_domain);
    const BoxArray child_parent_boxes = coarsen(child.U.box_array(), kAmrRefRatio);
    MultiFab coarse_flux_x;
    MultiFab coarse_flux_y;
    if (!replicated_parent) {
      coarse_flux_x = MultiFab(detail::amr_face_boxes(child_parent_boxes, true), child.U.dmap(),
                              parent.U.ncomp(), 0);
      coarse_flux_y = MultiFab(detail::amr_face_boxes(child_parent_boxes, false), child.U.dmap(),
                              parent.U.ncomp(), 0);
    }
    CoarseFineInterface interface(parent_domain, child.U.box_array(), periodicity);
    std::vector<Box2D> correction_regions =
        interface.reflux_register_regions(child.U.box_array());
    PreparedAmrTransitionAdvanceScratch result(
        parent, child, parent_flux_x, parent_flux_y, parent_domain, periodicity,
        replicated_parent, topology_generation, communicator, std::move(coarse_flux_x),
        std::move(coarse_flux_y), std::move(interface), std::move(correction_regions));
    result.prepare_strips_(child);
    if (!replicated_parent) {
      detail::parallel_copy_on(result.coarse_flux_x_, parent_flux_x, communicator,
                               ExecutionLane::parallel_copy_message_tag);
      detail::parallel_copy_on(result.coarse_flux_y_, parent_flux_y, communicator,
                               ExecutionLane::parallel_copy_message_tag);
      device_fence();
    }
    result.validate_replicated_parent_interfaces_(parent);
    return result;
  }

  void begin_replay(const AmrLevelMP& parent, const AmrLevelMP& child,
                    const MultiFab& parent_flux_x, const MultiFab& parent_flux_y,
                    std::uint64_t topology_generation,
                    const CommunicatorView& communicator) {
    validate_replay_(parent, child, parent_flux_x, parent_flux_y, topology_generation,
                     communicator);
    correction_.clear_on_device();
    for (RegMP& strip : strips_)
      detail::clear_reflux_strip_on_device(strip);
    if (!replicated_parent_) {
      detail::parallel_copy_on(coarse_flux_x_, parent_flux_x, communicator,
                               ExecutionLane::parallel_copy_message_tag);
      detail::parallel_copy_on(coarse_flux_y_, parent_flux_y, communicator,
                               ExecutionLane::parallel_copy_message_tag);
    }
    device_fence();
    sample_coarse_fluxes_(parent_flux_x, parent_flux_y);
  }

  std::vector<RegMP>& strips() noexcept { return strips_; }

  void synchronize(MultiFab& parent_state, Real dx, Real dy, Real dt,
                   const CommunicatorView& communicator) {
    validate_communicator_(communicator);
    for (RegMP& strip : strips_)
      interface_.route_reflux(strip, dx, dy, dt, correction_, ncomp_);
    correction_.gather(communicator);
    for (int local_parent = 0; local_parent < parent_state.local_size(); ++local_parent)
      for_each_cell(parent_state.box(local_parent),
                    detail::ApplyRefluxRegisterKernel{parent_state.fab(local_parent).array(),
                                                      correction_.view(), ncomp_});
    device_fence();
  }

  template <class CoarseStripRange, class FineStripRange>
  void synchronize_integrated(MultiFab& parent_state, Real dx, Real dy,
                              const CoarseStripRange& coarse_role,
                              const FineStripRange& fine_role,
                              const CommunicatorView& communicator) {
    validate_communicator_(communicator);
    using CoarseStrip = typename CoarseStripRange::value_type;
    using FineStrip = typename FineStripRange::value_type;
    const CoarseStrip empty_coarse{};
    const FineStrip empty_fine{};
    correction_.clear_on_device();
    for (std::size_t global_child = 0; global_child < child_global_size_; ++global_child) {
      const CoarseStrip& coarse = global_child < coarse_role.size()
                                      ? coarse_role[global_child]
                                      : empty_coarse;
      const FineStrip& fine =
          global_child < fine_role.size() ? fine_role[global_child] : empty_fine;
      if (coarse.cL.empty() && coarse.cB.empty() && fine.fL.empty() && fine.fB.empty())
        continue;
      interface_.route_reflux_integrated_pair(coarse, fine, dx, dy, correction_, ncomp_);
    }
    correction_.gather(communicator);
    for (int local_parent = 0; local_parent < parent_state.local_size(); ++local_parent)
      for_each_cell(parent_state.box(local_parent),
                    detail::ApplyRefluxRegisterKernel{parent_state.fab(local_parent).array(),
                                                      correction_.view(), ncomp_});
    device_fence();
  }

  [[nodiscard]] std::size_t correction_lookup_capacity() const noexcept {
    return correction_.lookup_capacity();
  }

 private:
  PreparedAmrTransitionAdvanceScratch(
      const AmrLevelMP& parent, const AmrLevelMP& child, const MultiFab& parent_flux_x,
      const MultiFab& parent_flux_y, Box2D parent_domain, Periodicity periodicity,
      bool replicated_parent, std::uint64_t topology_generation,
      const CommunicatorView& communicator, MultiFab coarse_flux_x, MultiFab coarse_flux_y,
      CoarseFineInterface interface, std::vector<Box2D> correction_regions)
      : parent_boxes_(parent.U.box_array().boxes()),
        parent_ranks_(parent.U.dmap().ranks()),
        child_boxes_(child.U.box_array().boxes()),
        child_ranks_(child.U.dmap().ranks()),
        parent_flux_x_boxes_(parent_flux_x.box_array().boxes()),
        parent_flux_y_boxes_(parent_flux_y.box_array().boxes()),
        parent_ngrow_(parent.U.n_grow()),
        child_ngrow_(child.U.n_grow()),
        ncomp_(parent.U.ncomp()),
        child_global_size_(static_cast<std::size_t>(child.U.box_array().size())),
        parent_domain_(parent_domain),
        periodicity_(periodicity),
        replicated_parent_(replicated_parent),
        topology_generation_(topology_generation),
        communicator_size_(communicator.size()),
        communicator_rank_(communicator.rank()),
        communicator_identity_(detail::parallel_copy_communicator_identity(communicator)),
        coarse_flux_x_(std::move(coarse_flux_x)),
        coarse_flux_y_(std::move(coarse_flux_y)),
        replicated_parent_lookup_(replicated_parent
                                      ? std::optional<MfBoxLookup>(std::in_place, parent.U)
                                      : std::nullopt),
        interface_(std::move(interface)),
        correction_(std::move(correction_regions), ncomp_) {}

  void prepare_strips_(const AmrLevelMP& child) {
    strips_.resize(static_cast<std::size_t>(child.U.local_size()));
    for (int local_child = 0; local_child < child.U.local_size(); ++local_child) {
      const PatchRange range(child.U.box(local_child));
      RegMP& strip = strips_[static_cast<std::size_t>(local_child)];
      strip.I0 = range.I0;
      strip.I1 = range.I1;
      strip.J0 = range.J0;
      strip.J1 = range.J1;
      const std::size_t ni = static_cast<std::size_t>(strip.I1 - strip.I0 + 1);
      const std::size_t nj = static_cast<std::size_t>(strip.J1 - strip.J0 + 1);
      if (ni > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(ncomp_) ||
          nj > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(ncomp_))
        throw std::overflow_error("prepared AMR reflux strip size overflow");
      strip.cL.assign(nj * static_cast<std::size_t>(ncomp_), Real(0));
      strip.cR.assign(nj * static_cast<std::size_t>(ncomp_), Real(0));
      strip.cB.assign(ni * static_cast<std::size_t>(ncomp_), Real(0));
      strip.cT.assign(ni * static_cast<std::size_t>(ncomp_), Real(0));
      strip.fL.assign(nj * static_cast<std::size_t>(ncomp_), Real(0));
      strip.fR.assign(nj * static_cast<std::size_t>(ncomp_), Real(0));
      strip.fB.assign(ni * static_cast<std::size_t>(ncomp_), Real(0));
      strip.fT.assign(ni * static_cast<std::size_t>(ncomp_), Real(0));
    }
  }

  void validate_replicated_parent_interfaces_(const AmrLevelMP& parent) const {
    if (!replicated_parent_)
      return;
    if (!replicated_parent_lookup_)
      throw std::logic_error("prepared replicated AMR transition has no parent lookup");
    for (const RegMP& strip : strips_) {
      for (int j = strip.J0;;) {
        if (replicated_parent_lookup_->find(strip.I0, j) < 0 ||
            replicated_parent_lookup_->find(strip.I1, j) < 0)
          throw std::invalid_argument(
              "prepared replicated parent does not cover a child x interface");
        if (j == strip.J1)
          break;
        ++j;
      }
      for (int i = strip.I0;;) {
        if (replicated_parent_lookup_->find(i, strip.J0) < 0 ||
            replicated_parent_lookup_->find(i, strip.J1) < 0)
          throw std::invalid_argument(
              "prepared replicated parent does not cover a child y interface");
        if (i == strip.I1)
          break;
        ++i;
      }
    }
    (void)parent;
  }

  void sample_coarse_fluxes_(const MultiFab& parent_flux_x,
                             const MultiFab& parent_flux_y) {
    for (std::size_t local_child = 0; local_child < strips_.size(); ++local_child) {
      RegMP& strip = strips_[local_child];
      const RefluxStripView view = reflux_strip_view(strip, ncomp_);
      if (!replicated_parent_) {
        const int local = static_cast<int>(local_child);
        sample_coarse_strip(coarse_flux_x_.fab(local).const_array(),
                            coarse_flux_x_.fab(local).const_array(),
                            coarse_flux_y_.fab(local).const_array(),
                            coarse_flux_y_.fab(local).const_array(), view);
        continue;
      }
      for (int j = strip.J0; j <= strip.J1;) {
        const int left = replicated_parent_lookup_->find(strip.I0, j);
        const int right = replicated_parent_lookup_->find(strip.I1, j);
        int end = j;
        while (end < strip.J1 &&
               replicated_parent_lookup_->find(strip.I0, end + 1) == left &&
               replicated_parent_lookup_->find(strip.I1, end + 1) == right)
          ++end;
        sample_coarse_x_strip(parent_flux_x.fab(left).const_array(),
                              parent_flux_x.fab(right).const_array(), view, j, end);
        j = end + 1;
      }
      for (int i = strip.I0; i <= strip.I1;) {
        const int bottom = replicated_parent_lookup_->find(i, strip.J0);
        const int top = replicated_parent_lookup_->find(i, strip.J1);
        int end = i;
        while (end < strip.I1 &&
               replicated_parent_lookup_->find(end + 1, strip.J0) == bottom &&
               replicated_parent_lookup_->find(end + 1, strip.J1) == top)
          ++end;
        sample_coarse_y_strip(parent_flux_y.fab(bottom).const_array(),
                              parent_flux_y.fab(top).const_array(), view, i, end);
        i = end + 1;
      }
    }
  }

  void validate_communicator_(const CommunicatorView& communicator) const {
    if (communicator.size() != communicator_size_ || communicator.rank() != communicator_rank_ ||
        detail::parallel_copy_communicator_identity(communicator) != communicator_identity_)
      throw std::invalid_argument("prepared AMR transition changed execution communicator");
  }

  void validate_replay_(const AmrLevelMP& parent, const AmrLevelMP& child,
                        const MultiFab& parent_flux_x, const MultiFab& parent_flux_y,
                        std::uint64_t topology_generation,
                        const CommunicatorView& communicator) const {
    if (parent.U.box_array().boxes() != parent_boxes_ ||
        parent.U.dmap().ranks() != parent_ranks_ || parent.U.n_grow() != parent_ngrow_ ||
        parent.U.ncomp() != ncomp_ || child.U.box_array().boxes() != child_boxes_ ||
        child.U.dmap().ranks() != child_ranks_ || child.U.n_grow() != child_ngrow_ ||
        child.U.ncomp() != ncomp_ ||
        parent_flux_x.box_array().boxes() != parent_flux_x_boxes_ ||
        parent_flux_y.box_array().boxes() != parent_flux_y_boxes_ ||
        parent_flux_x.dmap().ranks() != parent_ranks_ ||
        parent_flux_y.dmap().ranks() != parent_ranks_ || parent_flux_x.ncomp() != ncomp_ ||
        parent_flux_y.ncomp() != ncomp_)
      throw std::invalid_argument("prepared AMR transition crossed an exact layout");
    if (topology_generation != topology_generation_)
      throw std::invalid_argument("prepared AMR transition crossed a topology generation");
    validate_communicator_(communicator);
  }

  std::vector<Box2D> parent_boxes_;
  std::vector<int> parent_ranks_;
  std::vector<Box2D> child_boxes_;
  std::vector<int> child_ranks_;
  std::vector<Box2D> parent_flux_x_boxes_;
  std::vector<Box2D> parent_flux_y_boxes_;
  int parent_ngrow_ = 0;
  int child_ngrow_ = 0;
  int ncomp_ = 0;
  std::size_t child_global_size_ = 0;
  Box2D parent_domain_{};
  Periodicity periodicity_{};
  bool replicated_parent_ = false;
  std::uint64_t topology_generation_ = 0;
  int communicator_size_ = 1;
  int communicator_rank_ = 0;
  std::int64_t communicator_identity_ = 0;
  MultiFab coarse_flux_x_;
  MultiFab coarse_flux_y_;
  std::optional<MfBoxLookup> replicated_parent_lookup_;
  CoarseFineInterface interface_;
  FluxRegister correction_;
  std::vector<RegMP> strips_;
};

/// Whole-hierarchy prepared scratch, including a persistent private attempt used to preserve the
/// all-or-nothing publish contract without allocating a copied hierarchy at every step.
class PreparedAmrAdvanceScratchPlan {
 public:
  PreparedAmrAdvanceScratchPlan(const PreparedAmrAdvanceScratchPlan&) = delete;
  PreparedAmrAdvanceScratchPlan& operator=(const PreparedAmrAdvanceScratchPlan&) = delete;
  PreparedAmrAdvanceScratchPlan(PreparedAmrAdvanceScratchPlan&&) noexcept = default;
  PreparedAmrAdvanceScratchPlan& operator=(PreparedAmrAdvanceScratchPlan&&) noexcept = default;

  static PreparedAmrAdvanceScratchPlan prepare(
      const std::vector<AmrLevelMP>& levels, const Box2D& base_domain, Periodicity periodicity,
      bool coarse_replicated, bool wave_speed_cache, std::uint64_t topology_generation,
      const CommunicatorView& communicator = world_communicator_view()) {
    if (levels.empty() || base_domain.empty())
      throw std::invalid_argument("prepared AMR advance requires a non-empty hierarchy");
    std::vector<PreparedAmrLevelAdvanceScratch> level_scratch;
    std::vector<AmrLevelMP> attempt;
    level_scratch.reserve(levels.size());
    attempt.reserve(levels.size());
    for (const AmrLevelMP& level : levels) {
      level_scratch.push_back(
          PreparedAmrLevelAdvanceScratch::prepare(level, wave_speed_cache));
      attempt.push_back({MultiFab(level.U.box_array(), level.U.dmap(), level.U.ncomp(),
                                  level.U.n_grow()),
                         level.aux, level.dx, level.dy});
    }
    std::vector<PreparedAmrTransitionAdvanceScratch> transitions;
    transitions.reserve(levels.size() - 1);
    for (std::size_t parent = 0; parent + 1 < levels.size(); ++parent)
      transitions.push_back(PreparedAmrTransitionAdvanceScratch::prepare(
          levels[parent], levels[parent + 1], level_scratch[parent].flux_x(),
          level_scratch[parent].flux_y(), amr_level_index_domain(base_domain,
                                                                 static_cast<int>(parent)),
          periodicity, parent == 0 && coarse_replicated, topology_generation, communicator));
    return PreparedAmrAdvanceScratchPlan(
        levels, base_domain, periodicity, coarse_replicated, wave_speed_cache,
        topology_generation, communicator, std::move(level_scratch), std::move(transitions),
        std::move(attempt));
  }

  void validate_hierarchy_contract(const std::vector<AmrLevelMP>& levels,
                                   const Box2D& base_domain, Periodicity periodicity,
                                   bool coarse_replicated, bool wave_speed_cache,
                                   std::uint64_t topology_generation,
                                   const CommunicatorView& communicator) const {
    if (levels.size() != levels_.size() || base_domain != base_domain_ ||
        !same_periodicity(periodicity, periodicity_) ||
        coarse_replicated != coarse_replicated_ || wave_speed_cache != wave_speed_cache_ ||
        topology_generation != topology_generation_)
      throw std::invalid_argument("prepared AMR advance changed its hierarchy contract");
    if (communicator.size() != communicator_size_ || communicator.rank() != communicator_rank_ ||
        detail::parallel_copy_communicator_identity(communicator) != communicator_identity_)
      throw std::invalid_argument("prepared AMR advance changed execution communicator");
    for (std::size_t level = 0; level < levels.size(); ++level)
      level_scratch_.at(level).validate(levels[level], wave_speed_cache);
  }

  std::vector<AmrLevelMP>& begin_attempt(const std::vector<AmrLevelMP>& live) {
    if (attempt_active_)
      throw std::logic_error("prepared AMR advance attempt is already active");
    validate_live_layouts_(live);
    for (std::size_t level = 0; level < live.size(); ++level) {
      detail::copy_amr_storage(attempt_[level].U, live[level].U);
      attempt_[level].aux = live[level].aux;
      attempt_[level].dx = live[level].dx;
      attempt_[level].dy = live[level].dy;
    }
    attempt_active_ = true;
    return attempt_;
  }

  void publish_attempt(std::vector<AmrLevelMP>& live) {
    if (!attempt_active_)
      throw std::logic_error("prepared AMR advance has no active attempt to publish");
    validate_live_layouts_(live);
    detail::reject_nonfinite_finite_volume_hierarchy(
        "PreparedAmrAdvanceScratchPlan::publish_attempt", attempt_);
    device_fence();
    for (std::size_t level = 0; level < live.size(); ++level)
      detail::copy_amr_storage(live[level].U, attempt_[level].U);
    device_fence();
    attempt_active_ = false;
  }

  void abort_attempt() noexcept { attempt_active_ = false; }

  PreparedAmrLevelAdvanceScratch& level(int index) {
    return level_scratch_.at(static_cast<std::size_t>(index));
  }
  PreparedAmrTransitionAdvanceScratch& transition_for_child(int child_level) {
    if (child_level <= 0 || child_level >= static_cast<int>(levels_.size()))
      throw std::out_of_range("prepared AMR advance child level is out of range");
    return transitions_.at(static_cast<std::size_t>(child_level - 1));
  }
  [[nodiscard]] std::uint64_t topology_generation() const noexcept {
    return topology_generation_;
  }

 private:
  PreparedAmrAdvanceScratchPlan(
      const std::vector<AmrLevelMP>& levels, Box2D base_domain, Periodicity periodicity,
      bool coarse_replicated, bool wave_speed_cache, std::uint64_t topology_generation,
      const CommunicatorView& communicator,
      std::vector<PreparedAmrLevelAdvanceScratch> level_scratch,
      std::vector<PreparedAmrTransitionAdvanceScratch> transitions,
      std::vector<AmrLevelMP> attempt)
      : base_domain_(base_domain),
        periodicity_(periodicity),
        coarse_replicated_(coarse_replicated),
        wave_speed_cache_(wave_speed_cache),
        topology_generation_(topology_generation),
        communicator_size_(communicator.size()),
        communicator_rank_(communicator.rank()),
        communicator_identity_(detail::parallel_copy_communicator_identity(communicator)),
        level_scratch_(std::move(level_scratch)),
        transitions_(std::move(transitions)),
        attempt_(std::move(attempt)) {
    levels_.reserve(levels.size());
    for (const AmrLevelMP& level : levels)
      levels_.push_back({level.U.box_array().boxes(), level.U.dmap().ranks(), level.U.ncomp(),
                         level.U.n_grow(), level.aux});
  }

  struct LevelIdentity {
    std::vector<Box2D> boxes;
    std::vector<int> ranks;
    int ncomp = 0;
    int ngrow = 0;
    const MultiFab* aux = nullptr;
  };

  void validate_live_layouts_(const std::vector<AmrLevelMP>& live) const {
    if (live.size() != levels_.size())
      throw std::invalid_argument("prepared AMR advance changed hierarchy level count");
    for (std::size_t level = 0; level < live.size(); ++level) {
      const LevelIdentity& expected = levels_[level];
      if (live[level].U.box_array().boxes() != expected.boxes ||
          live[level].U.dmap().ranks() != expected.ranks ||
          live[level].U.ncomp() != expected.ncomp || live[level].U.n_grow() != expected.ngrow ||
          live[level].aux != expected.aux)
        throw std::invalid_argument("prepared AMR advance crossed an exact live layout");
    }
  }

  std::vector<LevelIdentity> levels_;
  Box2D base_domain_{};
  Periodicity periodicity_{};
  bool coarse_replicated_ = false;
  bool wave_speed_cache_ = false;
  std::uint64_t topology_generation_ = 0;
  int communicator_size_ = 1;
  int communicator_rank_ = 0;
  std::int64_t communicator_identity_ = 0;
  std::vector<PreparedAmrLevelAdvanceScratch> level_scratch_;
  std::vector<PreparedAmrTransitionAdvanceScratch> transitions_;
  std::vector<AmrLevelMP> attempt_;
  bool attempt_active_ = false;
};

namespace detail {  // INTERNAL N-level multi-patch engine; the public facade is advance_amr

/// One child interval of an explicitly authored parent/child clock relation.  The fractions are
/// exact until this final numerical lowering: @c parent_begin is the interpolation point in the
/// parent window and @c parent_span is the fraction of the parent dt advanced by this child.
struct ExplicitTemporalSubstep {
  Real parent_begin = Real(0);
  Real parent_span = Real(0);
};

/// Lowers an authored clock relation to the scalar intervals consumed by the native Berger-Oliger
/// engine.  In particular, a rational ratio such as 5/2 becomes {2/5, 2/5, 1/5}; the final 1/5
/// interval exists only when ExplicitFinalSubstep was declared.  Calling ParentChildClockRelation::
/// partition here preserves the same validation and exact Rational authority as the Program path.
inline std::vector<ExplicitTemporalSubstep> explicit_temporal_partition(
    const amr::ParentChildClockRelation& relation) {
  const amr::ClockWindow unit_parent{{relation.parent_level(), 0, amr::Rational(0, 1), 0.0},
                                     {relation.parent_level(), 0, amr::Rational(1, 1), 1.0}};
  const auto authored = relation.partition(unit_parent);
  std::vector<ExplicitTemporalSubstep> result;
  result.reserve(authored.size());
  for (const auto& child : authored) {
    const amr::Rational begin = child.window.begin.phase - unit_parent.begin.phase;
    const amr::Rational span = child.window.end.phase - child.window.begin.phase;
    result.push_back({static_cast<Real>(begin.value()), static_cast<Real>(span.value())});
  }
  return result;
}

/// Immutable execution plan prepared once from the authored clock chain.  Runtime blocks share this
/// object across every block/substep, so the hot recursive path performs no clock partition allocation
/// or Rational lowering.
class PreparedAmrTemporalPlan {
 public:
  static PreparedAmrTemporalPlan prepare(
      const std::vector<amr::ParentChildClockRelation>& relations, int nlevels) {
    const std::size_t expected = nlevels > 0 ? static_cast<std::size_t>(nlevels - 1) : 0u;
    if (relations.size() != expected)
      throw std::runtime_error(
          "explicit AMR temporal relations must contain exactly one row per level transition");
    std::vector<std::vector<ExplicitTemporalSubstep>> transitions;
    transitions.reserve(relations.size());
    for (std::size_t transition = 0; transition < relations.size(); ++transition) {
      const auto& relation = relations[transition];
      if (relation.parent_level() != static_cast<int>(transition) ||
          relation.child_level() != static_cast<int>(transition + 1))
        throw std::runtime_error(
            "explicit AMR temporal relations must form the contiguous level chain");
      transitions.push_back(explicit_temporal_partition(relation));
    }
    return PreparedAmrTemporalPlan(nlevels, std::move(transitions));
  }

  int nlevels() const { return nlevels_; }
  const std::vector<ExplicitTemporalSubstep>& transition(int parent_level) const {
    return transitions_.at(static_cast<std::size_t>(parent_level));
  }

 private:
  PreparedAmrTemporalPlan(int nlevels,
                          std::vector<std::vector<ExplicitTemporalSubstep>> transitions)
      : nlevels_(nlevels), transitions_(std::move(transitions)) {}

  int nlevels_ = 0;
  std::vector<std::vector<ExplicitTemporalSubstep>> transitions_;
};

inline void fill_amr_coarse_fine_temporal_ghosts(
    MultiFab& fine, int child_level, const Box2D& base_domain, Periodicity periodicity,
    const MultiFab& old_parent, const MultiFab& new_parent, Real fraction,
    bool coarse_replicated, Real positivity_floor, int positivity_component,
    PreparedAmrFillPatchPlan* fill_patch_plan) {
  if (child_level <= 0)
    throw std::invalid_argument("coarse/fine temporal ghost fill requires a fine level");
  const bool replicated_parent = child_level == 1 && coarse_replicated;
  if (fill_patch_plan == nullptr) {
    mf_fill_fine_ghosts_mb(
        fine, old_parent, new_parent, amr_level_index_domain(base_domain, child_level - 1),
        fraction, replicated_parent, positivity_floor, positivity_component, periodicity);
    return;
  }
  const CommunicatorView communicator =
      replicated_parent ? CommunicatorView{} : world_communicator_view();
  mf_fill_fine_ghosts_mb(
      fine, old_parent, new_parent, fill_patch_plan->transition_for_child(child_level), fraction,
      positivity_floor, positivity_component, fill_patch_plan->topology_generation(), communicator);
}

// Fills the ghosts of an AMR level: level 0 = base-domain BC (fill_boundary); level > 0 =
// time-interpolated coarse-fine ghosts from the parent at position frac (mf_fill_fine_ghosts_mb)
// THEN fine-fine halos (fill_boundary). Factored out of the head of subcycle_level_mp, REUSED by
// either SSPRK advance, which must refill the ghosts BEFORE each stage flux evaluation. The parent
// is REPLICATED only for lev == 1 (replicated level 0), otherwise distributed (internal
// parallel_copy).
inline void ssprk_refill_level_ghosts(MultiFab& U, int lev, const Box2D& base_dom,
                                      Periodicity base_per, const MultiFab* pOld,
                                      const MultiFab* pNew, Real frac, bool coarse_replicated,
                                      Real pos_floor = Real(0), int pos_comp = 0,
                                      const AmrBoundaryFillAuthority* boundary_fill = nullptr,
                                      Real dx = Real(1), Real dy = Real(1),
                                      PreparedAmrFillPatchPlan* fill_patch_plan = nullptr) {
  if (lev == 0) {
    fill_amr_same_level_and_physical(U, base_dom, lev, dx, dy, base_per, boundary_fill);
  } else {
    fill_amr_coarse_fine_temporal_ghosts(U, lev, base_dom, base_per, *pOld, *pNew, frac,
                                         coarse_replicated, pos_floor, pos_comp,
                                         fill_patch_plan);
    const Box2D fdom = amr_level_index_domain(base_dom, lev);
    fill_amr_same_level_and_physical(U, fdom, lev, dx, dy, base_per, boundary_fill);
  }
}

/// Maps an SSP tableau abscissa into the parent temporal window of a fine-level substep.  The root
/// has no parent and ignores the value.  Fine levels must carry a positive parent span; clamp only
/// round-off at the endpoints and reject an inconsistent temporal schedule rather than extrapolate.
inline Real ssprk_parent_stage_fraction(int lev, Real parent_begin, Real parent_span,
                                        Real stage_abscissa) {
  if (lev == 0)
    return parent_begin;
  if (!(parent_span > Real(0)))
    throw std::runtime_error("SSPRK fine-level advance requires a positive parent temporal span");
  const Real fraction = parent_begin + stage_abscissa * parent_span;
  const Real tolerance = Real(32) * std::numeric_limits<Real>::epsilon();
  if (fraction < -tolerance || fraction > Real(1) + tolerance)
    throw std::runtime_error("SSPRK stage lies outside the parent temporal window");
  return std::clamp(fraction, Real(0), Real(1));
}

// SSPRK2 / Heun on ONE AMR level:
//   U1    = U0 + dt L(U0)
//   U_new = 1/2 U0 + 1/2 (U1 + dt L(U1)).
// L(U) = -div F(U) + S(U) is evaluated consistently at each stage.  On return (fx, fy) contain
// Feff = 1/2 F(U0) + 1/2 F(U1), exactly the face flux whose divergence appears in U_new.  The
// parent/fine reflux machinery can therefore consume this method through its existing flux slots
// without a scheme-specific correction.  IMEX is deliberately absent and rejected by the caller.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void ssprk2_advance_level(const Model& m, AmrLevelMP& lv, Real dt, MultiFab& fx, MultiFab& fy,
                          bool recon_prim, int lev, const Box2D& base_dom, Periodicity base_per,
                          const MultiFab* pOld, const MultiFab* pNew, Real frac, Real parent_span,
                          bool coarse_replicated, Real pos_floor = Real(0),
                          Real weno_eps = kWenoEpsilon, MultiFab* wave_speed_cache = nullptr,
                          const AmrBoundaryFillAuthority* boundary_fill = nullptr,
                          PreparedAmrFillPatchPlan* fill_patch_plan = nullptr,
                          PreparedAmrLevelAdvanceScratch* prepared_scratch = nullptr) {
  const int nc = lv.U.ncomp();
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  std::optional<MultiFab> owned_U0;
  std::optional<MultiFab> owned_R;
  std::optional<MultiFab> owned_Fxs;
  std::optional<MultiFab> owned_Fys;
  MultiFab* U0 = nullptr;
  MultiFab* R = nullptr;
  MultiFab* Fxs = nullptr;
  MultiFab* Fys = nullptr;
  if (prepared_scratch != nullptr) {
    prepared_scratch->capture_start(lv.U);
    U0 = &prepared_scratch->start_state();
    R = &prepared_scratch->rhs();
    Fxs = &prepared_scratch->stage_flux_x();
    Fys = &prepared_scratch->stage_flux_y();
  } else {
    owned_U0.emplace(lv.U);
    owned_R.emplace(lv.U.box_array(), lv.U.dmap(), nc, 0);
    owned_Fxs.emplace(fx.box_array(), fx.dmap(), nc, 0);
    owned_Fys.emplace(fy.box_array(), fy.dmap(), nc, 0);
    U0 = &*owned_U0;
    R = &*owned_R;
    Fxs = &*owned_Fxs;
    Fys = &*owned_Fys;
  }

  // Stage 0: the caller already materialized F(U0).
  detail::mf_eval_rhs_unchecked(m, lv.U, *lv.aux, fx, fy, lv.dx, lv.dy, *R);
  saxpy(lv.U, dt, *R);                        // U1 = U0 + dt L(U0)
  lincomb(fx, Real(1) / 2, fx, Real(0), fx);  // Feff <- 1/2 F(U0)
  lincomb(fy, Real(1) / 2, fy, Real(0), fy);
  // Validate after the arithmetic: finite R and F0 can still overflow in dt*R or a weighted
  // flux.  One collective covers both the stage state and everything already destined for reflux.
  detail::reject_nonfinite_finite_volume_data("ssprk2_advance_level(stage 0)", lv.U, fx, fy);

  // Stage 1: refresh the stage state before evaluating F(U1) and L(U1).
  const Real stage1_fraction = ssprk_parent_stage_fraction(lev, frac, parent_span, Real(1));
  ssprk_refill_level_ghosts(lv.U, lev, base_dom, base_per, pOld, pNew, stage1_fraction,
                            coarse_replicated, pos_floor, pos_comp, boundary_fill, lv.dx, lv.dy,
                            fill_patch_plan);
  compute_face_fluxes_with_optional_hll_cache<Limiter, NumericalFlux>(
      m, lv.U, *lv.aux, *Fxs, *Fys, wave_speed_cache, lv.dx, lv.dy, recon_prim, pos_floor,
      weno_eps);
  device_fence();
  detail::mf_eval_rhs_unchecked(m, lv.U, *lv.aux, *Fxs, *Fys, lv.dx, lv.dy, *R);
  saxpy(lv.U, dt, *R);                                  // U1 + dt L(U1)
  lincomb(lv.U, Real(1) / 2, *U0, Real(1) / 2, lv.U);  // Shu-Osher U_new
  saxpy(fx, Real(1) / 2, *Fxs);                        // Feff += 1/2 F(U1)
  saxpy(fy, Real(1) / 2, *Fys);
  detail::reject_nonfinite_finite_volume_data("ssprk2_advance_level(stage 1)", lv.U, fx, fy);
}

// SSPRK3 (Shu-Osher, 3 stages, order 3) on ONE AMR level. (1) Advance lv.U from t to t+dt:
//   U1 = U0 + dt L(U0); U2 = 3/4 U0 + 1/4 (U1 + dt L(U1)); U_new = 1/3 U0 + 2/3 (U2 + dt L(U2))
// with L(U) = -div F(U) + S(U) (EXPLICIT source per stage, evaluated at the same state as the
// flux: true SSPRK method-of-lines, cf. mf_eval_rhs -- IMEX is NOT supported, rejected upstream).
// (2) Fills (fx, fy) with the EFFECTIVE FLUX of the step    Feff = 1/6 F(U0) + 1/6 F(U1) + 2/3 F(U2)
// which is EXACTLY the transport flux seen by the final state (U_new = U0 - dt div Feff + dt Seff).
// This is the flux the conservative reflux must record (coarse side g.c* and fine side g.f*), hence
// its write into (fx, fy) where the Euler path leaves the single flux F(U0). On INPUT (fx, fy)
// already contain F(U0) (stage 0, computed by the caller before the call). Between stages, the
// ghosts are refreshed at the SSPRK3 tableau abscissae in the parent temporal window: c1=1 for U1
// and c2=1/2 for U2. saxpy/lincomb and the RHS functor are device-clean kernels (named functors),
// with no extended lambda.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void ssprk3_advance_level(const Model& m, AmrLevelMP& lv, Real dt, MultiFab& fx, MultiFab& fy,
                          bool recon_prim, int lev, const Box2D& base_dom, Periodicity base_per,
                          const MultiFab* pOld, const MultiFab* pNew, Real frac, Real parent_span,
                          bool coarse_replicated, Real pos_floor = Real(0),
                          Real weno_eps = kWenoEpsilon, MultiFab* wave_speed_cache = nullptr,
                          const AmrBoundaryFillAuthority* boundary_fill = nullptr,
                          PreparedAmrFillPatchPlan* fill_patch_plan = nullptr,
                          PreparedAmrLevelAdvanceScratch* prepared_scratch = nullptr) {
  const int nc = lv.U.ncomp();
  // Density-role component for the C/F ghost floor (ADC-259), resolved ONCE on the host. pos_floor<=0
  // -> 0 without model introspection (positivity_comp short-circuit) -> bit-identical historical path.
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  std::optional<MultiFab> owned_U0;
  std::optional<MultiFab> owned_R;
  std::optional<MultiFab> owned_Fxs;
  std::optional<MultiFab> owned_Fys;
  MultiFab* U0 = nullptr;
  MultiFab* R = nullptr;
  MultiFab* Fxs = nullptr;
  MultiFab* Fys = nullptr;
  if (prepared_scratch != nullptr) {
    prepared_scratch->capture_start(lv.U);
    U0 = &prepared_scratch->start_state();
    R = &prepared_scratch->rhs();
    Fxs = &prepared_scratch->stage_flux_x();
    Fys = &prepared_scratch->stage_flux_y();
  } else {
    owned_U0.emplace(lv.U);
    owned_R.emplace(lv.U.box_array(), lv.U.dmap(), nc, 0);
    owned_Fxs.emplace(fx.box_array(), fx.dmap(), nc, 0);
    owned_Fys.emplace(fy.box_array(), fy.dmap(), nc, 0);
    U0 = &*owned_U0;
    R = &*owned_R;
    Fxs = &*owned_Fxs;
    Fys = &*owned_Fys;
  }

  // --- stage 0: F(U0) already in (fx, fy), R0 = -div F0 + S(U0) ---
  detail::mf_eval_rhs_unchecked(m, lv.U, *lv.aux, fx, fy, lv.dx, lv.dy, *R);
  saxpy(lv.U, dt, *R);                        // lv.U = U1 = U0 + dt R0
  lincomb(fx, Real(1) / 6, fx, Real(0), fx);  // Feff <- 1/6 F0 (pointwise aliasing, safe)
  lincomb(fy, Real(1) / 6, fy, Real(0), fy);
  detail::reject_nonfinite_finite_volume_data("ssprk3_advance_level(stage 0)", lv.U, fx, fy);

  // --- stage 1: F(U1) ---
  const Real stage1_fraction = ssprk_parent_stage_fraction(lev, frac, parent_span, Real(1));
  ssprk_refill_level_ghosts(lv.U, lev, base_dom, base_per, pOld, pNew, stage1_fraction,
                            coarse_replicated, pos_floor, pos_comp, boundary_fill, lv.dx, lv.dy,
                            fill_patch_plan);
  compute_face_fluxes_with_optional_hll_cache<Limiter, NumericalFlux>(
      m, lv.U, *lv.aux, *Fxs, *Fys, wave_speed_cache, lv.dx, lv.dy, recon_prim, pos_floor,
      weno_eps);
  device_fence();
  detail::mf_eval_rhs_unchecked(m, lv.U, *lv.aux, *Fxs, *Fys, lv.dx, lv.dy,
                                *R);  // R1 = -div F1 + S(U1)
  saxpy(lv.U, dt, *R);                                       // lv.U = U1 + dt R1
  lincomb(lv.U, Real(3) / 4, *U0, Real(1) / 4, lv.U);        // lv.U = U2
  saxpy(fx, Real(1) / 6, *Fxs);                              // Feff += 1/6 F1
  saxpy(fy, Real(1) / 6, *Fys);
  detail::reject_nonfinite_finite_volume_data("ssprk3_advance_level(stage 1)", lv.U, fx, fy);

  // --- stage 2: F(U2) ---
  const Real stage2_fraction = ssprk_parent_stage_fraction(lev, frac, parent_span, Real(1) / 2);
  ssprk_refill_level_ghosts(lv.U, lev, base_dom, base_per, pOld, pNew, stage2_fraction,
                            coarse_replicated, pos_floor, pos_comp, boundary_fill, lv.dx, lv.dy,
                            fill_patch_plan);
  compute_face_fluxes_with_optional_hll_cache<Limiter, NumericalFlux>(
      m, lv.U, *lv.aux, *Fxs, *Fys, wave_speed_cache, lv.dx, lv.dy, recon_prim, pos_floor,
      weno_eps);
  device_fence();
  detail::mf_eval_rhs_unchecked(m, lv.U, *lv.aux, *Fxs, *Fys, lv.dx, lv.dy,
                                *R);  // R2 = -div F2 + S(U2)
  saxpy(lv.U, dt, *R);                                       // lv.U = U2 + dt R2
  lincomb(lv.U, Real(1) / 3, *U0, Real(2) / 3, lv.U);        // lv.U = U_new (t + dt)
  saxpy(fx, Real(2) / 3, *Fxs);                              // Feff += 2/3 F2
  saxpy(fy, Real(2) / 3, *Fys);
  detail::reject_nonfinite_finite_volume_data("ssprk3_advance_level(stage 2)", lv.U, fx, fy);
}

template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void subcycle_level_mp(const Model& m, std::vector<AmrLevelMP>& L, int lev, Real dt,
                       const Box2D& base_dom, Periodicity base_per, const MultiFab* pOld,
                       const MultiFab* pNew, Real frac, Real parent_span,
                       std::vector<RegMP>* parentRegs, bool coarse_replicated = true,
                       bool recon_prim = false, bool imex = false, const NewtonOptions& nopts = {},
                       AmrTimeMethod tmethod = AmrTimeMethod::kEuler, Real pos_floor = Real(0),
                       Real weno_eps = kWenoEpsilon,
                       bool wave_speed_cache = false,
                       const PreparedAmrTemporalPlan* temporal_plan = nullptr,
                       const AmrBoundaryFillAuthority* boundary_fill = nullptr,
                       PreparedAmrFillPatchPlan* fill_patch_plan = nullptr,
                       PreparedAmrAverageDownPlan* average_down_plan = nullptr,
                       PreparedAmrAdvanceScratchPlan* advance_scratch_plan = nullptr) {
  // An unknown enum value must never fall through to the Euler branch.  This also protects direct
  // low-level callers that bypass the strict integer-wire lowering used by the runtime seams.
  switch (tmethod) {
    case AmrTimeMethod::kEuler:
    case AmrTimeMethod::kSsprk2:
    case AmrTimeMethod::kSsprk3:
      break;
    default:
      throw std::runtime_error("subcycle_level_mp: unknown AMR time method");
  }
  // SSPRK + IMEX is not a defined composition: both SSP methods evaluate an explicit source at
  // every stage, whereas the AMR IMEX route is a separate Euler-transport/backward-Euler-source
  // split.  Reject rather than silently replacing either temporal contract.
  if (tmethod != AmrTimeMethod::kEuler && imex)
    throw std::runtime_error(
        "subcycle_level_mp: SSPRK2/SSPRK3 + IMEX unsupported; use an explicit SSP method "
        "or time='imex' (forward Euler transport + implicit source)");
  const int nc = L[lev].U.ncomp();
  // Density-role component for the C/F ghost floor (ADC-259), resolved ONCE on the host. pos_floor<=0
  // -> 0 without model introspection (positivity_comp short-circuit) -> bit-identical historical path.
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  AmrLevelMP& lv = L[lev];
  const int np = lv.U.local_size();
  const bool ssprk2 = (tmethod == AmrTimeMethod::kSsprk2);
  const bool ssprk3 = (tmethod == AmrTimeMethod::kSsprk3);
  const bool ssprk = ssprk2 || ssprk3;
  const bool is_leaf = (lev + 1 >= static_cast<int>(L.size()));
  PreparedAmrLevelAdvanceScratch* level_scratch =
      advance_scratch_plan == nullptr ? nullptr : &advance_scratch_plan->level(lev);
  if (level_scratch != nullptr)
    level_scratch->validate(lv, wave_speed_cache);

  if (lev == 0) {
    fill_amr_same_level_and_physical(lv.U, base_dom, lev, lv.dx, lv.dy, base_per,
                                     boundary_fill);
  } else {
    // parent (level lev-1) REPLICATED only if it is level 0 (lev == 1); otherwise distributed ->
    // FillPatch by parallel_copy.
    fill_amr_coarse_fine_temporal_ghosts(lv.U, lev, base_dom, base_per, *pOld, *pNew, frac,
                                         coarse_replicated, pos_floor, pos_comp,
                                         fill_patch_plan);
    const Box2D fdom = amr_level_index_domain(base_dom, lev);
    fill_amr_same_level_and_physical(lv.U, fdom, lev, lv.dx, lv.dy, base_per, boundary_fill);
  }

  // face-box per GLOBAL box + same dmap (cf. amr_step_2level_multipatch): BoxArray and
  // DistributionMapping of the same size under MPI, fx.fab(li) <-> lv.U.fab(li). Identical in
  // serial (local == global).
  std::optional<MultiFab> owned_fx;
  std::optional<MultiFab> owned_fy;
  if (level_scratch == nullptr) {
    owned_fx.emplace(detail::amr_face_boxes(lv.U.box_array(), true), lv.U.dmap(), nc, 0);
    owned_fy.emplace(detail::amr_face_boxes(lv.U.box_array(), false), lv.U.dmap(), nc, 0);
  }
  MultiFab& fx = level_scratch == nullptr ? *owned_fx : level_scratch->flux_x();
  MultiFab& fy = level_scratch == nullptr ? *owned_fy : level_scratch->flux_y();
  MultiFab owned_wave_cache;
  MultiFab* wave_cache_ptr = nullptr;
  if constexpr (std::is_same_v<NumericalFlux, HLLFlux>) {
    if (wave_speed_cache)
      wave_cache_ptr = level_scratch == nullptr ? &owned_wave_cache
                                                : level_scratch->wave_speed_cache();
  }
  compute_face_fluxes_with_optional_hll_cache<Limiter, NumericalFlux>(
      m, lv.U, *lv.aux, fx, fy, wave_cache_ptr, lv.dx, lv.dy, recon_prim, pos_floor, weno_eps);
  device_fence();

  // SSPRK: advance lv.U before the shared register/reflux path and replace (fx, fy), initially F(U0),
  // by the method's effective flux (1/2 F0 + 1/2 F1 for SSPRK2; 1/6 F0 + 1/6 F1 + 2/3 F2 for
  // SSPRK3).  The existing parent/fine register logic then remains scheme-independent and exactly
  // conservative for the selected time method.  Save the starting state for child interpolation.
  // Euler skips this block and retains the historical in-place path below.
  MultiFab ssp_U_old;  // state t (pre-advance capture); filled only for an SSP coarse role
  if (ssprk) {
    if (!is_leaf && level_scratch == nullptr)
      ssp_U_old = lv.U;  // the children interpolate between this state (t) and advanced lv.U (t+dt)
    if (ssprk2)
      ssprk2_advance_level<Limiter, NumericalFlux>(m, lv, dt, fx, fy, recon_prim, lev, base_dom,
                                                   base_per, pOld, pNew, frac, parent_span,
                                                   coarse_replicated, pos_floor, weno_eps,
                                                   wave_cache_ptr, boundary_fill,
                                                   fill_patch_plan, level_scratch);
    else
      ssprk3_advance_level<Limiter, NumericalFlux>(m, lv, dt, fx, fy, recon_prim, lev, base_dom,
                                                   base_per, pOld, pNew, frac, parent_span,
                                                   coarse_replicated, pos_floor, weno_eps,
                                                   wave_cache_ptr, boundary_fill,
                                                   fill_patch_plan, level_scratch);
  }

  // Euler owns one private hierarchy transaction (created by the public driver below).  Advance
  // the candidate before any parent/fine register sees its flux, then validate the candidate and
  // both face fields with exactly one collective.  A failed explicit or implicit source therefore
  // cannot publish a level or contaminate a conservation register.
  MultiFab euler_U_old;
  if (!ssprk) {
    if (!is_leaf) {
      if (level_scratch != nullptr)
        level_scratch->capture_start(lv.U);
      else
        euler_U_old = lv.U;
    }
    mf_advance_faces(lv.U, fx, fy, lv.dx, lv.dy, dt);
    mf_apply_source_treatment(m, lv.U, *lv.aux, dt, imex, nopts);
    detail::reject_nonfinite_finite_volume_data("subcycle_level_mp(Euler)", lv.U, fx, fy);
  }

  if (parentRegs) {  // FINE role: fine fluxes of THIS level into the parent register
    for (int li = 0; li < np; ++li) {
      RegMP& g = (*parentRegs)[li];
      const ConstArray4 FX = fx.fab(li).const_array(), FY = fy.fab(li).const_array();
      accumulate_fine_strip(FX, FY, reflux_strip_view(g, nc), dt);
    }
  }

  if (is_leaf) {  // leaf: both Euler and SSPRK candidates were validated before register use
    return;
  }

  // COARSE role for lev+1. The prepared route owns every topology-dependent carrier, strip,
  // lookup and correction register. A direct low-level caller gets the same implementation through
  // a one-shot owner whose lifetime spans the complete child recursion.
  const Box2D level_domain = amr_level_index_domain(base_dom, lev);
  const bool replicated_parent = (lev == 0) && coarse_replicated;
  const CommunicatorView communicator = world_communicator_view();
  std::optional<PreparedAmrTransitionAdvanceScratch> owned_transition;
  PreparedAmrTransitionAdvanceScratch* transition = nullptr;
  if (advance_scratch_plan != nullptr) {
    transition = &advance_scratch_plan->transition_for_child(lev + 1);
  } else {
    owned_transition.emplace(PreparedAmrTransitionAdvanceScratch::prepare(
        lv, L[lev + 1], fx, fy, level_domain, base_per, replicated_parent,
        /*topology_generation=*/0, communicator));
    transition = &*owned_transition;
  }
  const std::uint64_t transition_generation =
      advance_scratch_plan == nullptr ? 0 : advance_scratch_plan->topology_generation();
  transition->begin_replay(lv, L[lev + 1], fx, fy, transition_generation, communicator);
  std::vector<RegMP>& regs = transition->strips();

  // State t for temporal interpolation of the children.  Both methods are already advanced; the
  // method-specific snapshot was captured immediately before its first arithmetic update.
  const MultiFab& U_old = level_scratch != nullptr
                              ? level_scratch->start_state()
                              : (ssprk ? ssp_U_old : euler_U_old);
  if (temporal_plan != nullptr) {
    for (const ExplicitTemporalSubstep& child : temporal_plan->transition(lev))
      subcycle_level_mp<Limiter, NumericalFlux>(
          m, L, lev + 1, dt * child.parent_span, base_dom, base_per, &U_old, &lv.U,
          child.parent_begin, child.parent_span, &regs, coarse_replicated, recon_prim, imex, nopts,
          tmethod, pos_floor, weno_eps, wave_speed_cache, temporal_plan, boundary_fill,
          fill_patch_plan, average_down_plan, advance_scratch_plan);
  } else {
    // Clearly separated low-level compatibility route.  Callers that have not authored temporal
    // relations retain the historical ratio-two cadence; an installed explicit relation never
    // reaches this branch.
    const SubcyclingSchedule legacy_schedule(lev, lev + 1, amr::Rational(kAmrRefRatio, 1),
                                             amr::RemainderPolicy::IntegralOnly);
    for (int s = 0; s < legacy_schedule.count(); ++s)
      subcycle_level_mp<Limiter, NumericalFlux>(
          m, L, lev + 1, legacy_schedule.dt_sub(dt), base_dom, base_per, &U_old, &lv.U,
          legacy_schedule.frac(s), Real(1) / Real(legacy_schedule.count()), &regs,
          coarse_replicated, recon_prim, imex, nopts, tmethod, pos_floor, weno_eps,
          wave_speed_cache, nullptr, boundary_fill, fill_patch_plan, average_down_plan,
          advance_scratch_plan);
  }
  transition->synchronize(lv.U, lv.dx, lv.dy, dt, communicator);
  // The synchronization contract is explicit and ordered: first route/apply the conservative
  // interface correction to uncovered parent cells, then replace covered cells by the child average.
  // The two regions are disjoint, but preserving this order makes the phase observable and extensible.
  if (average_down_plan != nullptr) {
    mf_average_down_mb(
        L[lev + 1].U, lv.U, average_down_plan->transition_for_child(lev + 1),
        average_down_plan->topology_generation(), world_communicator_view());
  } else {
    mf_average_down_mb(L[lev + 1].U, lv.U);  // one-shot compatibility route
  }
}

// Publish only state storage; aux pointers and geometry belong to the prepared live hierarchy.
// Every numerical operation, flux register and coarse/fine synchronization has already succeeded
// on the private attempt when this function is reached.
inline void publish_amr_state_transaction(std::vector<AmrLevelMP>& live,
                                          std::vector<AmrLevelMP>& attempt) {
  reject_nonfinite_finite_volume_hierarchy(
      "publish_amr_state_transaction(synchronization)", attempt);
  device_fence();
  for (std::size_t level = 0; level < live.size(); ++level)
    live[level].U = std::move(attempt[level].U);
}

// Driver: one dt step of the N-level multi-patch hierarchy (level 0 = coarse).
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void amr_step_multilevel_multipatch(const Model& m, std::vector<AmrLevelMP>& L, const Box2D& dom,
                                    Real dt, Periodicity per,
                                    bool coarse_replicated = true, bool recon_prim = false,
                                    bool imex = false, const NewtonOptions& nopts = {},
                                    AmrTimeMethod tmethod = AmrTimeMethod::kEuler,
                                    Real pos_floor = Real(0),
                                    Real weno_eps = kWenoEpsilon,
                                    bool wave_speed_cache = false,
                                    const AmrBoundaryFillAuthority* boundary_fill = nullptr,
                                    PreparedAmrFillPatchPlan* fill_patch_plan = nullptr,
                                    PreparedAmrAverageDownPlan* average_down_plan = nullptr,
                                    PreparedAmrAdvanceScratchPlan* advance_scratch_plan = nullptr) {
  validate_amr_boundary_fill_authority(per, boundary_fill, L);
  if (fill_patch_plan != nullptr)
    fill_patch_plan->validate_hierarchy_contract(static_cast<int>(L.size()), dom, per,
                                                 coarse_replicated,
                                                 fill_patch_plan->prepared_operator());
  if (average_down_plan != nullptr && average_down_plan->nlevels() != static_cast<int>(L.size()))
    throw std::invalid_argument(
        "prepared AMR average-down plan does not match the hierarchy level count");
  const auto execute = [&](std::vector<AmrLevelMP>& attempt) {
    subcycle_level_mp<Limiter, NumericalFlux>(
        m, attempt, 0, dt, dom, per, nullptr, nullptr, Real(0), Real(0), nullptr,
        coarse_replicated, recon_prim, imex, nopts, tmethod, pos_floor, weno_eps,
        wave_speed_cache, nullptr, boundary_fill, fill_patch_plan, average_down_plan,
        advance_scratch_plan);
  };
  if (advance_scratch_plan == nullptr) {
    std::vector<AmrLevelMP> attempt = L;
    execute(attempt);
    publish_amr_state_transaction(L, attempt);
    return;
  }
  advance_scratch_plan->validate_hierarchy_contract(
      L, dom, per, coarse_replicated, wave_speed_cache,
      advance_scratch_plan->topology_generation(), world_communicator_view());
  std::vector<AmrLevelMP>& attempt = advance_scratch_plan->begin_attempt(L);
  try {
    execute(attempt);
    advance_scratch_plan->publish_attempt(L);
  } catch (...) {
    advance_scratch_plan->abort_attempt();
    throw;
  }
}

/// Explicit-clock production route.  Validation of the full chain is completed before level zero
/// is touched, then every recursive transition consumes the authored relation (never the spatial
/// refinement ratio).  The legacy entry above remains available only for direct low-level callers
/// that have no temporal contract.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void amr_step_multilevel_multipatch_with_temporal_relations(
    const Model& m, std::vector<AmrLevelMP>& L, const Box2D& dom, Real dt,
    const std::vector<amr::ParentChildClockRelation>& temporal_relations,
    Periodicity per, bool coarse_replicated = true,
    bool recon_prim = false, bool imex = false, const NewtonOptions& nopts = {},
    AmrTimeMethod tmethod = AmrTimeMethod::kEuler, Real pos_floor = Real(0),
    Real weno_eps = kWenoEpsilon, bool wave_speed_cache = false,
    const AmrBoundaryFillAuthority* boundary_fill = nullptr,
    PreparedAmrFillPatchPlan* fill_patch_plan = nullptr,
    PreparedAmrAverageDownPlan* average_down_plan = nullptr,
    PreparedAmrAdvanceScratchPlan* advance_scratch_plan = nullptr) {
  const PreparedAmrTemporalPlan plan =
      PreparedAmrTemporalPlan::prepare(temporal_relations, static_cast<int>(L.size()));
  validate_amr_boundary_fill_authority(per, boundary_fill, L);
  if (fill_patch_plan != nullptr)
    fill_patch_plan->validate_hierarchy_contract(static_cast<int>(L.size()), dom, per,
                                                 coarse_replicated,
                                                 fill_patch_plan->prepared_operator());
  if (average_down_plan != nullptr && average_down_plan->nlevels() != static_cast<int>(L.size()))
    throw std::invalid_argument(
        "prepared AMR average-down plan does not match the hierarchy level count");
  const auto execute = [&](std::vector<AmrLevelMP>& attempt) {
    subcycle_level_mp<Limiter, NumericalFlux>(
        m, attempt, 0, dt, dom, per, nullptr, nullptr, Real(0), Real(0), nullptr,
        coarse_replicated, recon_prim, imex, nopts, tmethod, pos_floor, weno_eps,
        wave_speed_cache, &plan, boundary_fill, fill_patch_plan, average_down_plan,
        advance_scratch_plan);
  };
  if (advance_scratch_plan == nullptr) {
    std::vector<AmrLevelMP> attempt = L;
    execute(attempt);
    publish_amr_state_transaction(L, attempt);
    return;
  }
  advance_scratch_plan->validate_hierarchy_contract(
      L, dom, per, coarse_replicated, wave_speed_cache,
      advance_scratch_plan->topology_generation(), world_communicator_view());
  std::vector<AmrLevelMP>& attempt = advance_scratch_plan->begin_attempt(L);
  try {
    execute(attempt);
    advance_scratch_plan->publish_attempt(L);
  } catch (...) {
    advance_scratch_plan->abort_attempt();
    throw;
  }
}

/// Relation-allocation-free execution of a plan prepared by the owning runtime at installation.
/// The numerical step still owns the same private state transaction as every other production
/// route, so failure cannot partially publish a hierarchy.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void amr_step_multilevel_multipatch_with_temporal_plan(
    const Model& m, std::vector<AmrLevelMP>& L, const Box2D& dom, Real dt,
    const PreparedAmrTemporalPlan& temporal_plan, Periodicity per,
    bool coarse_replicated = true, bool recon_prim = false, bool imex = false,
    const NewtonOptions& nopts = {}, AmrTimeMethod tmethod = AmrTimeMethod::kEuler,
    Real pos_floor = Real(0), Real weno_eps = kWenoEpsilon,
    bool wave_speed_cache = false,
    const AmrBoundaryFillAuthority* boundary_fill = nullptr,
    PreparedAmrFillPatchPlan* fill_patch_plan = nullptr,
    PreparedAmrAverageDownPlan* average_down_plan = nullptr,
    PreparedAmrAdvanceScratchPlan* advance_scratch_plan = nullptr) {
  if (temporal_plan.nlevels() != static_cast<int>(L.size()))
    throw std::runtime_error("prepared AMR temporal plan does not match the level hierarchy");
  validate_amr_boundary_fill_authority(per, boundary_fill, L);
  if (fill_patch_plan != nullptr)
    fill_patch_plan->validate_hierarchy_contract(static_cast<int>(L.size()), dom, per,
                                                 coarse_replicated,
                                                 fill_patch_plan->prepared_operator());
  if (average_down_plan != nullptr && average_down_plan->nlevels() != static_cast<int>(L.size()))
    throw std::invalid_argument(
        "prepared AMR average-down plan does not match the hierarchy level count");
  const auto execute = [&](std::vector<AmrLevelMP>& attempt) {
    subcycle_level_mp<Limiter, NumericalFlux>(
        m, attempt, 0, dt, dom, per, nullptr, nullptr, Real(0), Real(0), nullptr,
        coarse_replicated, recon_prim, imex, nopts, tmethod, pos_floor, weno_eps,
        wave_speed_cache, &temporal_plan, boundary_fill, fill_patch_plan, average_down_plan,
        advance_scratch_plan);
  };
  if (advance_scratch_plan == nullptr) {
    std::vector<AmrLevelMP> attempt = L;
    execute(attempt);
    publish_amr_state_transaction(L, attempt);
    return;
  }
  advance_scratch_plan->validate_hierarchy_contract(
      L, dom, per, coarse_replicated, wave_speed_cache,
      advance_scratch_plan->topology_generation(), world_communicator_view());
  std::vector<AmrLevelMP>& attempt = advance_scratch_plan->begin_attempt(L);
  try {
    execute(attempt);
    advance_scratch_plan->publish_attempt(L);
  } catch (...) {
    advance_scratch_plan->abort_attempt();
    throw;
  }
}

}  // namespace detail

}  // namespace pops
