#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <pops/core/foundation/types.hpp>      // Real
#include <pops/mesh/index/box2d.hpp>           // Box2D
#include <pops/mesh/layout/box_array.hpp>      // BoxArray
#include <pops/mesh/storage/multifab.hpp>      // MultiFab, ConstArray4
#include <pops/numerics/spatial_operator.hpp>  // xface_box / yface_box (Fx/Fy sizing)
#include <pops/runtime/amr/amr_runtime.hpp>    // the class this file defines members of

/// @file
/// @brief ADC-639 conservative reflux for a whole-system compiled Program on AMR: the flux-materialising
/// per-level residual seams, the effective-flux interface-strip samplers, and the route_reflux driver
/// the recursively subcycled Program driver runs at level sync.
///
/// Included at the END of amr_runtime.hpp (so the full AmrRuntime class AND the reflux types from
/// amr_reflux_mf.hpp -- PatchRange / FluxRegister / CoverageMask / CoarseFineInterface / RegMP -- are
/// visible). It defines AmrRuntime members out-of-line; it is NOT a standalone header.
///
/// The apparatus reuses the native Berger-Oliger reflux. Round-off C/F conservation comes from capturing
/// every substep's per-level effective flux at the interface (through the Program's own
/// linear-combination weights, the flux ledger in amr_program_context.hpp) and routing it through the
/// native coverage-aware correction at level sync -- reflux THEN average-down, finest first. The
/// coarse-only / flat Program (nlev==1) never reaches any of this: the trajectory stays
/// bit-identical (the load-bearing parity gate).

namespace pops {

/// One patch's effective-flux interface strip (the native Reg / RegMP field layout: c* = the coarse-side
/// flux at the 4 C/F faces, f* = the fine-side flux, both already dt-integrated in the ledger). The strip
/// is over a coarse (PARENT-coordinate) footprint [I0..I1] x [J0..J1]; the x strips hold nJ*nc entries
/// (indexed (J - J0)*nc + k), the y strips nI*nc. Reused by CoarseFineInterface::route_reflux_integrated.
struct EdgeStrip {
  int I0 = 0, I1 = -1, J0 = 0, J1 = -1;
  std::vector<Real> cL, cR, cB, cT;  // coarse-side flux at the L/R/B/T C/F faces
  std::vector<Real> fL, fR, fB, fT;  // fine-side (time-integrated) flux at the same faces

  void alloc(const Box2D& coarse_footprint, int nc) {
    I0 = coarse_footprint.lo[0];
    I1 = coarse_footprint.hi[0];
    J0 = coarse_footprint.lo[1];
    J1 = coarse_footprint.hi[1];
    const int nJ = (J1 - J0 + 1) * nc, nI = (I1 - I0 + 1) * nc;
    cL.assign(nJ, Real(0));
    cR.assign(nJ, Real(0));
    fL.assign(nJ, Real(0));
    fR.assign(nJ, Real(0));
    cB.assign(nI, Real(0));
    cT.assign(nI, Real(0));
    fB.assign(nI, Real(0));
    fT.assign(nI, Real(0));
  }
};

/// The effective-flux ledger entry of ONE tracked buffer at ONE level: a COARSE-ROLE strip per child
/// (level-(k+1)) patch (the level-k coarse flux at the faces bordering each fine patch, present only if a
/// child level exists) and a FINE-ROLE strip per THIS-level (level-k) patch (the coarse-face-averaged
/// level-k flux at the level-k patch edges, present only if k >= 1). The Program's linear combination is
/// shadowed on these strips (component-wise axpy / lincomb): sampling is linear, so it commutes with the
/// combine, and the strip a commit lands on holds dt * Feff = dt * sum_i w_i F_i -- exactly the native
/// effective flux, reproduced from the Program text with no scheme dispatch.
struct EdgeFlux {
  std::vector<EdgeStrip> coarse;  // indexed by level-(k+1) patch (coarse role)
  std::vector<EdgeStrip> fine;    // indexed by level-k patch (fine role)
  bool empty() const { return coarse.empty() && fine.empty(); }
};

namespace detail {

inline void require_program_reflux(bool condition, const std::string& detail) {
  if (!condition)
    throw std::runtime_error("Program reflux: " + detail);
}

/// Validates that @p Fx / @p Fy are the face-centred views of @p state with the same global box
/// identity and ownership.  The state MultiFab is the sole layout authority: a face field that merely
/// has the same local_size is not accepted because local indices are rank-relative and can alias a
/// different global patch after redistribution.
inline void validate_program_reflux_faces(const MultiFab& state, const MultiFab& Fx,
                                          const MultiFab& Fy, int nc, const char* role) {
  const std::string prefix = std::string(role) + " face layout ";
  require_program_reflux(nc > 0, prefix + "requires nc > 0");
  require_program_reflux(state.ncomp() == nc, prefix + "state component count differs from nc");
  require_program_reflux(Fx.ncomp() == nc && Fy.ncomp() == nc,
                         prefix + "flux component count differs from nc");
  require_program_reflux(Fx.box_array().size() == state.box_array().size() &&
                             Fy.box_array().size() == state.box_array().size(),
                         prefix + "global box count differs from the state authority");
  require_program_reflux(
      Fx.dmap().ranks() == state.dmap().ranks() && Fy.dmap().ranks() == state.dmap().ranks(),
      prefix + "ownership differs from the state authority");
  for (int g = 0; g < state.box_array().size(); ++g) {
    require_program_reflux(Fx.box_array()[g] == xface_box(state.box_array()[g]),
                           prefix + "x-face global order differs at box " + std::to_string(g));
    require_program_reflux(Fy.box_array()[g] == yface_box(state.box_array()[g]),
                           prefix + "y-face global order differs at box " + std::to_string(g));
  }
  require_program_reflux(
      Fx.local_size() == state.local_size() && Fy.local_size() == state.local_size(),
      prefix + "local box count differs from the state authority");
  for (int li = 0; li < state.local_size(); ++li) {
    const int g = state.global_index(li);
    require_program_reflux(
        Fx.global_index(li) == g && Fy.global_index(li) == g,
        prefix + "local/global patch identity differs at local box " + std::to_string(li));
  }
}

/// Persistent redistribution storage for one (Program block, parent level).  The destination
/// MultiFabs own CopyScheduleCache instances, so retaining them across RHS evaluations also retains
/// the MPI pack/unpack schedule.  An AMR topology epoch change, component-width change, or exact child
/// layout/ownership change rebuilds both fields; no schedule can survive a regrid under a stale layout.
struct CoarseRoleScratch {
  std::uint64_t topology_epoch = std::numeric_limits<std::uint64_t>::max();
  int ncomp = 0;
  std::vector<Box2D> child_boxes;
  std::vector<int> child_ranks;
  MultiFab Fx;
  MultiFab Fy;

  bool matches(const MultiFab& child, std::uint64_t epoch, int nc) const {
    return topology_epoch == epoch && ncomp == nc && child_boxes == child.box_array().boxes() &&
           child_ranks == child.dmap().ranks();
  }

  void prepare(const MultiFab& child, std::uint64_t epoch, int nc) {
    if (matches(child, epoch, nc))
      return;
    const BoxArray coarse_child = coarsen(child.box_array(), kAmrRefRatio);
    std::vector<Box2D> xfaces, yfaces;
    xfaces.reserve(static_cast<std::size_t>(coarse_child.size()));
    yfaces.reserve(static_cast<std::size_t>(coarse_child.size()));
    for (int g = 0; g < coarse_child.size(); ++g) {
      xfaces.push_back(xface_box(coarse_child[g]));
      yfaces.push_back(yface_box(coarse_child[g]));
    }
    Fx = MultiFab(BoxArray(std::move(xfaces)), child.dmap(), nc, 0);
    Fy = MultiFab(BoxArray(std::move(yfaces)), child.dmap(), nc, 0);
    topology_epoch = epoch;
    ncomp = nc;
    child_boxes = child.box_array().boxes();
    child_ranks = child.dmap().ranks();
  }
};

/// Add a * src into dst component-wise (both strips share the SAME patch footprint within a frozen macro-
/// step layout, so the flat arrays align by index). Missing src (empty) contributes 0. Used by the ledger
/// axpy mirror.
inline void edge_axpy(std::vector<Real>& dst, Real a, const std::vector<Real>& src) {
  if (src.empty())
    return;
  if (dst.size() != src.size())
    dst.assign(src.size(), Real(0));
  for (std::size_t i = 0; i < src.size(); ++i)
    dst[i] += a * src[i];
}

inline void edge_strip_axpy(EdgeStrip& d, Real a, const EdgeStrip& s) {
  d.I0 = s.I0;
  d.I1 = s.I1;
  d.J0 = s.J0;
  d.J1 = s.J1;
  edge_axpy(d.cL, a, s.cL);
  edge_axpy(d.cR, a, s.cR);
  edge_axpy(d.cB, a, s.cB);
  edge_axpy(d.cT, a, s.cT);
  edge_axpy(d.fL, a, s.fL);
  edge_axpy(d.fR, a, s.fR);
  edge_axpy(d.fB, a, s.fB);
  edge_axpy(d.fT, a, s.fT);
}

/// dst += a * src over EVERY strip (coarse + fine roles), growing dst to src's patch count on first touch.
/// This is the whole engine of the ledger's saxpy / lincomb mirror: an arbitrary DSL scheme's stage
/// weights reach the reflux register through exactly these calls.
inline void edge_flux_axpy(EdgeFlux& dst, Real a, const EdgeFlux& src) {
  if (src.coarse.size() > dst.coarse.size())
    dst.coarse.resize(src.coarse.size());
  if (src.fine.size() > dst.fine.size())
    dst.fine.resize(src.fine.size());
  for (std::size_t i = 0; i < src.coarse.size(); ++i)
    edge_strip_axpy(dst.coarse[i], a, src.coarse[i]);
  for (std::size_t i = 0; i < src.fine.size(); ++i)
    edge_strip_axpy(dst.fine[i], a, src.fine[i]);
}

/// Sample the COARSE-ROLE strip of a level-k flux field (Fx/Fy on the level-k grid, PARENT of level k+1):
/// for each child (level-(k+1)) patch, read the level-k coarse flux at the 4 C/F faces of the patch's
/// coarse footprint.  The child layout is the ownership authority: exactly its owner samples each
/// strip.  A replicated parent is read locally; a distributed parent is first redistributed onto
/// child-owned coarse face grids.  This is the native algorithm in amr_subcycling.hpp, shared here
/// without assuming a mono-box parent.  @p parent identifies the face-Fab corresponding to each
/// parent cell box. @p child is the GLOBAL level-(k+1) layout and its rank ownership.
inline void sample_coarse_role_strip(const MultiFab& parent, const MultiFab& Fx, const MultiFab& Fy,
                                     const MultiFab& child, bool replicated_parent,
                                     std::uint64_t topology_epoch, int nc,
                                     CoarseRoleScratch& scratch, EdgeFlux& out) {
  device_fence();
  validate_program_reflux_faces(parent, Fx, Fy, nc, "coarse-role");
  require_program_reflux(child.ncomp() == nc, "coarse-role child component count differs from nc");
  const BoxArray& child_ba = child.box_array();
  out.coarse.assign(static_cast<std::size_t>(child_ba.size()), EdgeStrip{});

  if (!replicated_parent) {
    scratch.prepare(child, topology_epoch, nc);
    parallel_copy(scratch.Fx, Fx);
    parallel_copy(scratch.Fy, Fy);
    device_fence();
  } else {
    require_program_reflux(parent.local_size() == parent.box_array().size(),
                           "replicated parent is not fully resident on this rank");
  }

  for (int lc = 0; lc < child.local_size(); ++lc) {
    const int g = child.global_index(lc);
    const PatchRange pr(child.box(lc));
    EdgeStrip& s = out.coarse[static_cast<std::size_t>(g)];
    s.alloc(pr.box(), nc);

    if (replicated_parent) {
      for (int J = s.J0; J <= s.J1; ++J) {
        const int left = mf_find_box(parent, s.I0, J);
        const int right = mf_find_box(parent, s.I1, J);
        if (left < 0 || right < 0)
          throw std::runtime_error(
              "Program reflux: replicated parent does not cover a child x interface");
        const ConstArray4 FXL = Fx.fab(left).const_array();
        const ConstArray4 FXR = Fx.fab(right).const_array();
        for (int k = 0; k < nc; ++k) {
          s.cL[static_cast<std::size_t>((J - s.J0) * nc + k)] = FXL(s.I0, J, k);
          s.cR[static_cast<std::size_t>((J - s.J0) * nc + k)] = FXR(s.I1 + 1, J, k);
        }
      }
      for (int I = s.I0; I <= s.I1; ++I) {
        const int bottom = mf_find_box(parent, I, s.J0);
        const int top = mf_find_box(parent, I, s.J1);
        if (bottom < 0 || top < 0)
          throw std::runtime_error(
              "Program reflux: replicated parent does not cover a child y interface");
        const ConstArray4 FYB = Fy.fab(bottom).const_array();
        const ConstArray4 FYT = Fy.fab(top).const_array();
        for (int k = 0; k < nc; ++k) {
          s.cB[static_cast<std::size_t>((I - s.I0) * nc + k)] = FYB(I, s.J0, k);
          s.cT[static_cast<std::size_t>((I - s.I0) * nc + k)] = FYT(I, s.J1 + 1, k);
        }
      }
      continue;
    }

    const ConstArray4 FX = scratch.Fx.fab(lc).const_array();
    const ConstArray4 FY = scratch.Fy.fab(lc).const_array();
    for (int J = s.J0; J <= s.J1; ++J)
      for (int k = 0; k < nc; ++k) {
        s.cL[static_cast<std::size_t>((J - s.J0) * nc + k)] = FX(s.I0, J, k);
        s.cR[static_cast<std::size_t>((J - s.J0) * nc + k)] = FX(s.I1 + 1, J, k);
      }
    for (int I = s.I0; I <= s.I1; ++I)
      for (int k = 0; k < nc; ++k) {
        s.cB[static_cast<std::size_t>((I - s.I0) * nc + k)] = FY(I, s.J0, k);
        s.cT[static_cast<std::size_t>((I - s.I0) * nc + k)] = FY(I, s.J1 + 1, k);
      }
  }
}

/// Sample the FINE-ROLE strip of a level-k flux field (Fx/Fy on the level-k grid): for each LOCAL level-k
/// patch, accumulate the coarse-face-averaged level-k flux 0.5*(F(2I,2J)+F(2I,2J+1)) at the patch edges.
/// This is the native fine-flux accumulation of amr_subcycling.hpp:135-148 (mono) / :567-580 (multi),
/// lifted verbatim (WITHOUT the *dtf: the ledger carries dt through the Program combine, so the strip is a
/// pure flux and the commit's dt-weighting lands it as dt*Feff). @p Fx / @p Fy: the level-k face fluxes,
/// one face-box per LOCAL level-k box, co-distributed with the level state. @p state is the authoritative
/// level-k cell layout (the fine patches at this level, seen from level k-1). Fills @p out.fine.
inline void sample_fine_role_strip(const MultiFab& state, const MultiFab& Fx, const MultiFab& Fy,
                                   int nc, EdgeFlux& out) {
  device_fence();
  validate_program_reflux_faces(state, Fx, Fy, nc, "fine-role");
  const BoxArray& patch_ba = state.box_array();
  out.fine.assign(static_cast<std::size_t>(patch_ba.size()), EdgeStrip{});
  // The validated state authority supplies both local iteration and stable global patch identity.
  // Only the owning rank fills that global slot (the reflux router later reduces disjoint slots).
  for (int li = 0; li < state.local_size(); ++li) {
    const int g = state.global_index(li);
    const PatchRange pr(state.box(li));
    EdgeStrip& s = out.fine[static_cast<std::size_t>(g)];
    s.alloc(pr.box(), nc);
    const ConstArray4 FX = Fx.fab(li).const_array();
    const ConstArray4 FY = Fy.fab(li).const_array();
    for (int J = s.J0; J <= s.J1; ++J)
      for (int k = 0; k < nc; ++k) {
        s.fL[static_cast<std::size_t>((J - s.J0) * nc + k)] =
            Real(0.5) * (FX(2 * s.I0, 2 * J, k) + FX(2 * s.I0, 2 * J + 1, k));
        s.fR[static_cast<std::size_t>((J - s.J0) * nc + k)] =
            Real(0.5) * (FX(2 * s.I1 + 2, 2 * J, k) + FX(2 * s.I1 + 2, 2 * J + 1, k));
      }
    for (int I = s.I0; I <= s.I1; ++I)
      for (int k = 0; k < nc; ++k) {
        s.fB[static_cast<std::size_t>((I - s.I0) * nc + k)] =
            Real(0.5) * (FY(2 * I, 2 * s.J0, k) + FY(2 * I + 1, 2 * s.J0, k));
        s.fT[static_cast<std::size_t>((I - s.I0) * nc + k)] =
            Real(0.5) * (FY(2 * I, 2 * s.J1 + 2, k) + FY(2 * I + 1, 2 * s.J1 + 2, k));
      }
  }
}

/// Combine a coarse-role EdgeStrip (c* = the level-(k-1) coarse flux at a patch's C/F faces) and the
/// matching fine-role EdgeStrip (f* = the coarse-face-averaged level-k flux at the same faces) into one
/// route_reflux-shaped register. Both are dt-integrated (the ledger carried dt through the Program), keyed
/// by the SAME level-k patch footprint. A missing role (empty strip) contributes 0.
inline EdgeStrip merge_reflux_strip(const EdgeStrip& coarse, const EdgeStrip& fine, int nc) {
  // Take the footprint from whichever strip is populated (they coincide when both exist).
  const EdgeStrip& shape =
      (!fine.cL.empty() || !fine.fL.empty() || fine.I1 >= fine.I0) ? fine : coarse;
  EdgeStrip g;
  g.I0 = shape.I0;
  g.I1 = shape.I1;
  g.J0 = shape.J0;
  g.J1 = shape.J1;
  const int nJ = (g.J1 - g.J0 + 1) * nc, nI = (g.I1 - g.I0 + 1) * nc;
  auto take = [](const std::vector<Real>& v, int n) {
    return v.size() == static_cast<std::size_t>(n)
               ? v
               : std::vector<Real>(static_cast<std::size_t>(n), Real(0));
  };
  g.cL = take(coarse.cL, nJ);
  g.cR = take(coarse.cR, nJ);
  g.cB = take(coarse.cB, nI);
  g.cT = take(coarse.cT, nI);
  g.fL = take(fine.fL, nJ);
  g.fR = take(fine.fR, nJ);
  g.fB = take(fine.fB, nI);
  g.fT = take(fine.fT, nI);
  return g;
}

/// Route the conservative reflux of the coarse-fine interface between coarse level @p k-1 (PARENT) and
/// fine level @p k for block @p b: for each level-k patch, merge the coarse-role strip (from the level-
/// (k-1) buffer's ledger) with the fine-role strip (from the level-k buffer's ledger) into a route_reflux
/// register, deposit the coverage-aware correction into a GLOBAL-indexed FluxRegister restricted to the
/// interface bounding box, gather (all_reduce; identity in serial), and apply to the coarse live state
/// under the coverage guard. Both sides are dt-integrated, so route_reflux_integrated (NO *dt) keeps the
/// cancellation exact. REUSES CoarseFineInterface / FluxRegister / CoverageMask verbatim. MPI single-writer
/// per (cell,direction) (ADC-636 ownership: each C/F face is owned by the rank holding the covering fine
/// patch), so the gather is associativity-free -> distributed == replicated bit-for-bit.
inline void route_reflux_program(AmrRuntime& eng, std::size_t b, int k, const EdgeFlux& coarse_role,
                                 const EdgeFlux& fine_role) {
  MultiFab& Uc = eng.level_state(b, k - 1);  // the PARENT (coarse) live state we correct
  const int nc = Uc.ncomp();
  const BoxArray child_ba = eng.level_state(b, k).box_array();  // GLOBAL level-k patches
  if (child_ba.size() == 0)
    return;
  const Geometry gc = eng.level_geom(k - 1);
  const int NX = gc.domain.nx(), NY = gc.domain.ny();
  const CoarseFineInterface cfi(Box2D{{0, 0}, {NX - 1, NY - 1}}, child_ba);
  // Interface register restricted to the coarse footprint of the fine patches, grown by 1 for the
  // bordering reflux cells, clamped to the coarse domain (the native rbox, amr_subcycling.hpp:169-171).
  const Box2D fpc = coarsen(child_ba, kAmrRefRatio).bounding_box();
  const Box2D rbox{{std::max(fpc.lo[0] - 1, 0), std::max(fpc.lo[1] - 1, 0)},
                   {std::min(fpc.hi[0] + 1, NX - 1), std::min(fpc.hi[1] + 1, NY - 1)}};
  FluxRegister ref(rbox, nc);
  const std::size_t np = static_cast<std::size_t>(child_ba.size());
  for (std::size_t g = 0; g < np; ++g) {
    const EdgeStrip c = (g < coarse_role.coarse.size()) ? coarse_role.coarse[g] : EdgeStrip{};
    const EdgeStrip f = (g < fine_role.fine.size()) ? fine_role.fine[g] : EdgeStrip{};
    if (c.cL.empty() && f.fL.empty() && c.cB.empty() && f.fB.empty())
      continue;  // this rank owns neither role for this patch -> 0 (single-writer MPI rule)
    const EdgeStrip merged = merge_reflux_strip(c, f, nc);
    cfi.route_reflux_integrated(merged, gc.dx(), gc.dy(), ref, nc);
  }
  ref.gather();  // all_reduce (identity in serial); single-writer per slot -> associativity-free
  device_fence();
  for (int pb = 0; pb < Uc.local_size();
       ++pb) {  // apply to the local coarse boxes under coverage guard
    Array4 c = Uc.fab(pb).array();
    const Box2D pbx = Uc.box(pb);
    for (int J = pbx.lo[1]; J <= pbx.hi[1]; ++J)
      for (int I = pbx.lo[0]; I <= pbx.hi[0]; ++I) {
        if (!ref.in(I, J))
          continue;
        for (int kk = 0; kk < nc; ++kk)
          c(I, J, kk) +=
              ref.at(I, J, kk);  // reflux (0 if no face); covered cells were average_down'd
      }
  }
}

}  // namespace detail

// --- AmrRuntime member definitions (ADC-639 capture seams) -------------------------------------------

inline void AmrRuntime::level_rhs_capture_into(
    std::size_t b, int k, const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
    MultiFab& R, MultiFab& Fx, MultiFab& Fy) {
  if (!blocks_[b].level_flux_capture)
    throw std::runtime_error(
        "AmrRuntime::level_rhs_capture_into: block '" + blocks_[b].name +
        "' has no flux-materialising per-level residual closure (rebuild the AMR block via the "
        "production DSL target='amr_system')");
  if (point.level != k)
    throw std::invalid_argument("AMR reflux boundary point belongs to another level");
  fill_level_state_cf_ghosts(b, k, U);  // fine-level C/F ghost refresh, identical to level_rhs_into
  if (static_cast<std::size_t>(k) < blocks_[b].boundary_sessions.size() &&
      blocks_[b].boundary_sessions[static_cast<std::size_t>(k)]) {
    if (!blocks_[b].level_flux_capture_prepared)
      throw std::runtime_error("AMR reflux block lacks its prepared flux capture closure");
    blocks_[b].level_flux_capture_prepared(
        point, U, aux_[k], level_geom(k), Fx, Fy, R,
        *blocks_[b].boundary_sessions[static_cast<std::size_t>(k)]);
    return;
  }
  if (blocks_[b].boundary_plan)
    throw std::runtime_error("AMR reflux boundary plan has no persistent prepared session");
  blocks_[b].level_flux_capture(U, aux_[k], level_geom(k), Fx, Fy, R);
}

inline void AmrRuntime::level_neg_div_flux_capture_into(
    std::size_t b, int k, const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
    MultiFab& R, MultiFab& Fx, MultiFab& Fy) {
  if (!blocks_[b].level_flux_capture_neg_div)
    throw std::runtime_error("AmrRuntime::level_neg_div_flux_capture_into: block '" +
                             blocks_[b].name +
                             "' has no flux-only flux-materialising per-level residual closure");
  if (point.level != k)
    throw std::invalid_argument("AMR reflux boundary point belongs to another level");
  fill_level_state_cf_ghosts(b, k, U);
  if (static_cast<std::size_t>(k) < blocks_[b].boundary_sessions.size() &&
      blocks_[b].boundary_sessions[static_cast<std::size_t>(k)]) {
    if (!blocks_[b].level_flux_capture_neg_div_prepared)
      throw std::runtime_error("AMR reflux block lacks its prepared flux-only capture closure");
    blocks_[b].level_flux_capture_neg_div_prepared(
        point, U, aux_[k], level_geom(k), Fx, Fy, R,
        *blocks_[b].boundary_sessions[static_cast<std::size_t>(k)]);
    return;
  }
  if (blocks_[b].boundary_plan)
    throw std::runtime_error("AMR reflux boundary plan has no persistent prepared session");
  blocks_[b].level_flux_capture_neg_div(U, aux_[k], level_geom(k), Fx, Fy, R);
}

inline void AmrRuntime::level_rhs_capture_into_temporal(
    std::size_t b, int k, const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
    MultiFab& R, MultiFab& Fx, MultiFab& Fy, const MultiFab& parent_old, const MultiFab& parent_new,
    const runtime::amr::TemporalTransferContext& target_time) {
  if (!blocks_[b].level_flux_capture)
    throw std::runtime_error("AmrRuntime::level_rhs_capture_into_temporal: block has no capture");
  if (point.level != k)
    throw std::invalid_argument("AMR reflux boundary point belongs to another level");
  fill_level_state_cf_ghosts_temporal(b, k, U, parent_old, parent_new, target_time);
  if (static_cast<std::size_t>(k) < blocks_[b].boundary_sessions.size() &&
      blocks_[b].boundary_sessions[static_cast<std::size_t>(k)]) {
    if (!blocks_[b].level_flux_capture_prepared)
      throw std::runtime_error("AMR reflux block lacks its prepared flux capture closure");
    blocks_[b].level_flux_capture_prepared(
        point, U, aux_[k], level_geom(k), Fx, Fy, R,
        *blocks_[b].boundary_sessions[static_cast<std::size_t>(k)]);
    return;
  }
  if (blocks_[b].boundary_plan)
    throw std::runtime_error("AMR reflux boundary plan has no persistent prepared session");
  blocks_[b].level_flux_capture(U, aux_[k], level_geom(k), Fx, Fy, R);
}

inline void AmrRuntime::level_neg_div_flux_capture_into_temporal(
    std::size_t b, int k, const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
    MultiFab& R, MultiFab& Fx, MultiFab& Fy, const MultiFab& parent_old, const MultiFab& parent_new,
    const runtime::amr::TemporalTransferContext& target_time) {
  if (!blocks_[b].level_flux_capture_neg_div)
    throw std::runtime_error(
        "AmrRuntime::level_neg_div_flux_capture_into_temporal: block has no capture");
  if (point.level != k)
    throw std::invalid_argument("AMR reflux boundary point belongs to another level");
  fill_level_state_cf_ghosts_temporal(b, k, U, parent_old, parent_new, target_time);
  if (static_cast<std::size_t>(k) < blocks_[b].boundary_sessions.size() &&
      blocks_[b].boundary_sessions[static_cast<std::size_t>(k)]) {
    if (!blocks_[b].level_flux_capture_neg_div_prepared)
      throw std::runtime_error("AMR reflux block lacks its prepared flux-only capture closure");
    blocks_[b].level_flux_capture_neg_div_prepared(
        point, U, aux_[k], level_geom(k), Fx, Fy, R,
        *blocks_[b].boundary_sessions[static_cast<std::size_t>(k)]);
    return;
  }
  if (blocks_[b].boundary_plan)
    throw std::runtime_error("AMR reflux boundary plan has no persistent prepared session");
  blocks_[b].level_flux_capture_neg_div(U, aux_[k], level_geom(k), Fx, Fy, R);
}

}  // namespace pops
