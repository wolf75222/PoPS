/// @file
/// @brief PHYSICAL boundary conditions at the domain edge (BCType, BCRec, fill_physical_bc,
///        fill_ghosts).
///
/// fill_boundary already fills the INTERIOR and periodic ghosts; here we fill the ghosts that
/// fall OUTSIDE the domain on non-periodic faces. Foextrap: zero-order extrapolation
/// (zero gradient), ghost = mirror interior cell (outflow / order-0 wall). Dirichlet: value
/// imposed at the FACE, ghost = 2 v - mirror interior (the ghost/interior average equals v at the face).
/// fill_ghosts composes both in the right order (interior/periodic THEN physical edge) and
/// fills the corners via x-faces then y-faces over the full extension. The edge kernels are
/// device-clean NAMED FUNCTORS (nvcc limitation).

#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/multifab.hpp>

namespace adc {

/// Boundary condition type for a face: Periodic (handled by fill_boundary), Foextrap (zero gradient,
/// outflow/order-0 wall), Dirichlet (value imposed at the face by reflection).
enum class BCType { Periodic, Foextrap, Dirichlet };

/// Boundary conditions for the FOUR faces of the domain (type + associated Dirichlet value). Default
/// is all periodic (xlo_val... ignored for non-Dirichlet faces).
struct BCRec {
  BCType xlo = BCType::Periodic, xhi = BCType::Periodic;
  BCType ylo = BCType::Periodic, yhi = BCType::Periodic;
  Real xlo_val = 0, xhi_val = 0, ylo_val = 0, yhi_val = 0;
};

namespace detail {
// NAMED FUNCTORS (not ADC_HD lambdas) for the physical boundary conditions. Same reasons as the
// rest of the elliptic/mesh path (#93, recipe #64): fill_physical_bc is called from fill_ghosts,
// itself pulled from the MG V-cycle first-instantiated from an external TU; an extended lambda there
// trips up device kernel emission under nvcc. Body identical to the old lambdas (Foextrap: copy of
// the mirror interior cell; Dirichlet: 2 v - reflection) -> bit-identical.
// x low face: i = lo - k (k = lo - i), Dirichlet mirror at 2 lo - i - 1.
struct BCFaceXLoKernel {
  Array4 a;
  int nc, lo;
  bool foe;
  Real val;
  ADC_HD void operator()(int i, int j) const {
    for (int c = 0; c < nc; ++c)
      a(i, j, c) = foe ? a(lo, j, c) : 2 * val - a(2 * lo - i - 1, j, c);
  }
};
struct BCFaceXHiKernel {
  Array4 a;
  int nc, hi;
  bool foe;
  Real val;
  ADC_HD void operator()(int i, int j) const {
    for (int c = 0; c < nc; ++c)
      a(i, j, c) = foe ? a(hi, j, c) : 2 * val - a(2 * hi - i + 1, j, c);
  }
};
struct BCFaceYLoKernel {
  Array4 a;
  int nc, lo;
  bool foe;
  Real val;
  ADC_HD void operator()(int i, int j) const {
    for (int c = 0; c < nc; ++c)
      a(i, j, c) = foe ? a(i, lo, c) : 2 * val - a(i, 2 * lo - j - 1, c);
  }
};
struct BCFaceYHiKernel {
  Array4 a;
  int nc, hi;
  bool foe;
  Real val;
  ADC_HD void operator()(int i, int j) const {
    for (int c = 0; c < nc; ++c)
      a(i, j, c) = foe ? a(i, hi, c) : 2 * val - a(i, 2 * hi - j + 1, c);
  }
};
}  // namespace detail

/// Fills the OUT-OF-domain ghosts of the NON-periodic faces of @p mf according to @p bc (Foextrap or
/// Dirichlet), over all components. No-op if there is no ghost or everything is periodic. PRECONDITION:
/// fill_boundary has already filled the interior/periodic (the x-faces read the y/theta ghosts already
/// filled to extend the radial BC into the halo, and the y-faces read the x ghosts for the corners).
/// CORNERS of the 9-point stencil: the x-face BC is extended to the EXTENDED j range (y/theta ghosts
/// included), so that the corner (x-physical CROSSED with y-periodic/neighbor) -- read by the cross
/// terms of a 9-point operator (e.g. PolarTensorKrylovSolver) -- is correct even in MULTI-BOX.
inline void fill_physical_bc(MultiFab& mf, const Box2D& domain,
                             const BCRec& bc) {
  const int ng = mf.n_grow();
  if (ng == 0) return;
  // All periodic: fill_boundary has already done everything, nothing to read/write here (and we
  // avoid a useless barrier on the hot path of the periodic multigrid).
  if (bc.xlo == BCType::Periodic && bc.xhi == BCType::Periodic &&
      bc.ylo == BCType::Periodic && bc.yhi == BCType::Periodic)
    return;
  // Physical edges on DEVICE (for_each_cell -> kernel): ghost = mirror cell (Foextrap: copy of the
  // 1st interior; Dirichlet: 2 v - reflection). Ghost index <-> layer: for x low, i = lo-k so the
  // Dirichlet mirror is 2 lo - i - 1 (k = lo - i). No more device_fence nor host access: these
  // kernels order after copy_shifted (same execution space), and the y-faces (i EXTENDED for the
  // corners) order after the x-faces on the same stream.
  const int nc = mf.ncomp();
  for (int li = 0; li < mf.local_size(); ++li) {
    Fab2D& F = mf.fab(li);
    const Box2D v = F.box();
    Array4 a = F.array();

    // --- x-faces, over the EXTENDED j range (j-ghosts included) ---
    // We extend the j range to the y/theta GHOSTS (j from v.lo[1]-ng to v.hi[1]+ng) instead of the
    // VALID range alone. Reason (9-point stencil corner, multi-box): a CROSS term (a_rt/a_tr of the
    // polar operator) reads the DIAGONAL neighbors p(i+-1, j+-1) -> the CORNER ghost (x-physical
    // CROSSED with y-ghost) must be filled. When y/theta is PERIODIC or borders a NEIGHBOR box,
    // fill_boundary has already filled the j-ghost row for the INTERIOR x columns; the x-physical
    // reflection (which reads a(lo, j) / a(2 lo - i - 1, j) at the SAME j) thus correctly extends the
    // radial edge into the y halo. Without this extension, the corner (x-ghost, y-ghost) stays at 0
    // and the cross term is WRONG at the box edge (multi-box divergence, cf. test_polar_schur_multibox).
    // The VALID range alone was enough in 5-point (no diagonal read); that corner was never read.
    // NOTE: a DOUBLE-physical corner (x AND y non-periodic) is then OVERWRITTEN by the y pass (i
    // extended, below, which runs AFTER) -> Cartesian behavior unchanged (y wins). In y-physical we
    // read a(lo, j-ghost) here, possibly not filled, but the result is overwritten: no effect on the
    // final corner value. Mono-box theta periodic: only the corners (previously at 0, never read in
    // 5-point) change -> bit-identical for any stencil <= 9-point including the 5-point Cartesian
    // (the new corner value is only read by a 9-point).
    const int jglo = v.lo[1] - ng, jghi = v.hi[1] + ng;
    if (bc.xlo != BCType::Periodic && v.lo[0] == domain.lo[0]) {
      const int lo = domain.lo[0];
      const bool foe = bc.xlo == BCType::Foextrap;
      const Real val = bc.xlo_val;
      for_each_cell(Box2D{{lo - ng, jglo}, {lo - 1, jghi}},
                    detail::BCFaceXLoKernel{a, nc, lo, foe, val});
    }
    if (bc.xhi != BCType::Periodic && v.hi[0] == domain.hi[0]) {
      const int hi = domain.hi[0];
      const bool foe = bc.xhi == BCType::Foextrap;
      const Real val = bc.xhi_val;
      for_each_cell(Box2D{{hi + 1, jglo}, {hi + ng, jghi}},
                    detail::BCFaceXHiKernel{a, nc, hi, foe, val});
    }

    // --- y-faces, over the EXTENDED i range (corners via the already-filled x-ghosts) ---
    const int iglo = v.lo[0] - ng, ighi = v.hi[0] + ng;
    if (bc.ylo != BCType::Periodic && v.lo[1] == domain.lo[1]) {
      const int lo = domain.lo[1];
      const bool foe = bc.ylo == BCType::Foextrap;
      const Real val = bc.ylo_val;
      for_each_cell(Box2D{{iglo, lo - ng}, {ighi, lo - 1}},
                    detail::BCFaceYLoKernel{a, nc, lo, foe, val});
    }
    if (bc.yhi != BCType::Periodic && v.hi[1] == domain.hi[1]) {
      const int hi = domain.hi[1];
      const bool foe = bc.yhi == BCType::Foextrap;
      const Real val = bc.yhi_val;
      for_each_cell(Box2D{{iglo, hi + 1}, {ighi, hi + ng}},
                    detail::BCFaceYHiKernel{a, nc, hi, foe, val});
    }
  }
}

/// COMPLETE ghost filling: fill_boundary (interior + periodic, periodicity deduced from
/// @p bc) THEN fill_physical_bc (physical edges). Usual entry point before assembling a residual.
inline void fill_ghosts(MultiFab& mf, const Box2D& domain, const BCRec& bc) {
  Periodicity per{bc.xlo == BCType::Periodic, bc.ylo == BCType::Periodic};
  fill_boundary(mf, domain, per);
  fill_physical_bc(mf, domain, bc);
}

}  // namespace adc
