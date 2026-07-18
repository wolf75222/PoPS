// ADC-636: adjacent fine patches in the composite FAC. Two level-1 patches that SHARE A FACE (their
// coarse footprints are edge-adjacent). The fine-fine join is handled by fill_boundary before the C/F
// bilerp (a shared ghost takes the sibling's valid data), and the two-way flux correction is
// enumerated from the uncovered coarse side so the shared interior face gets NO correction (it is not
// a coarse-fine face). We check:
//   (i)   ctor ACCEPTS adjacent patches (previously refused) and refuses overlapping ones;
//   (ii)  finiteness + FAC convergence;
//   (iii) POTENTIAL CONTINUITY across the shared face: the two patches agree at the shared boundary
//         (each patch's edge cell ~ the sibling's adjacent edge cell) far below the coarse error;
//   (iv)  MMS: the composite over the adjacent union beats coarse-only in the refined region.
//
// Serial (Kokkos OFF); coarse mono-box replicated.

#include <gtest/gtest.h>

#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace pops;
static constexpr double kPi = 3.14159265358979323846;

static double u_exact(double x, double y) {
  return std::sin(3.0 * kPi * x) * std::sin(3.0 * kPi * y);
}
static double f_rhs(double x, double y) {
  return -18.0 * kPi * kPi * u_exact(x, y);
}

static void fill_f(MultiFab& f, const Geometry& g) {
  for (int li = 0; li < f.local_size(); ++li) {
    Array4 a = f.fab(li).array();
    const Box2D b = f.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        a(i, j, 0) = f_rhs(g.x_cell(i), g.y_cell(j));
  }
}

// ------------------------------------------------------------------------------------------------
// (i) the ctor accepts adjacent patches and refuses overlapping ones.
// ------------------------------------------------------------------------------------------------
TEST(CompositeFacAdjacentTest, ctor_accepts_adjacent_refuses_overlap) {
  const int n = 32, r = 2;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom_c{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba_c = BoxArray::from_domain(dom, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

  // A: coarse footprint [8,15]x[8,23] -> fine [16,31]x[16,47].
  // B: coarse footprint [16,23]x[8,23] -> fine [32,47]x[16,47] : shares the x-face with A (adjacent).
  Box2D fA{{16, 16}, {31, 47}};
  Box2D fB{{32, 16}, {47, 47}};
  BoxArray adj(std::vector<Box2D>{fA, fB});
  EXPECT_NO_THROW({ CompositeFacPoisson fac(geom_c, ba_c, bc, adj, r); });

  // overlapping footprints must still refuse.
  Box2D oA{{16, 16}, {31, 47}};
  Box2D oB{{30, 16}, {47, 47}};  // footprint [15,23] overlaps A's [8,15]
  BoxArray ovl(std::vector<Box2D>{oA, oB});
  EXPECT_THROW({ CompositeFacPoisson fac(geom_c, ba_c, bc, ovl, r); }, std::runtime_error);
}

// ------------------------------------------------------------------------------------------------
// (ii)-(iv) continuity + MMS on the adjacent union.
// ------------------------------------------------------------------------------------------------
TEST(CompositeFacAdjacentTest, adjacent_patch_continuity_and_mms) {
  comm_init();
  const int n = 48, r = 2;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom_c{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba_c = BoxArray::from_domain(dom, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
  const Geometry geom_f = geom_c.refine(r);

  // Adjacent patches spanning the central half, split at the x-midline into A (left) and B (right).
  // A: coarse [n/4, n/2 - 1] x [n/4, 3n/4 - 1] ; B: coarse [n/2, 3n/4 - 1] x [n/4, 3n/4 - 1].
  const int Ax0 = n / 4, Ax1 = n / 2 - 1, Bx0 = n / 2, Bx1 = 3 * n / 4 - 1;
  const int Jy0 = n / 4, Jy1 = 3 * n / 4 - 1;
  Box2D fA{{r * Ax0, r * Jy0}, {r * Ax1 + r - 1, r * Jy1 + r - 1}};
  Box2D fB{{r * Bx0, r * Jy0}, {r * Bx1 + r - 1, r * Jy1 + r - 1}};
  BoxArray adj(std::vector<Box2D>{fA, fB});

  CompositeFacPoisson fac(geom_c, ba_c, bc, adj, r);
  fill_f(fac.rhs_coarse(), geom_c);
  fill_f(fac.rhs_fine(), geom_f);

  // coarse-only reference.
  GeometricMG mg0(geom_c, ba_c, bc, {}, FieldDistribution::Replicated);
  fill_f(mg0.rhs(), geom_c);
  mg0.phi().set_val(0.0);
  mg0.solve(1e-12, 100);
  device_fence();

  const Real rfac =
      fac.solve(/*max_iters=*/60, /*fine_sweeps=*/100, /*rel_tol=*/1e-9, /*abs_tol=*/0.0);
  device_fence();

  EXPECT_TRUE(std::isfinite(rfac));
  EXPECT_TRUE(rfac < 1e-2) << "adjacent-patch FAC converges: rfac=" << rfac;

  // (iii) CONTINUITY across the shared x-face (fine i = r*Bx0 = r*(n/2)). Patch A's rightmost valid
  // column is i = r*Ax1 + r - 1 = r*Bx0 - 1; patch B's leftmost is i = r*Bx0. fill_boundary makes each
  // patch's ghost the sibling's valid cell, so at convergence phi is continuous: A's edge cell ~ B's
  // adjacent edge cell (both approximate u at neighboring fine centers). We compare A's last column to
  // B's first column at the same rows; the jump must be far below the coarse discretization error.
  const ConstArray4 PA = fac.phi_fine().fab(0).const_array();  // patch A (first box)
  const ConstArray4 PB = fac.phi_fine().fab(1).const_array();  // patch B (second box)
  const int iA = r * Ax1 + r - 1;                              // A rightmost valid column
  const int iB = r * Bx0;  // B leftmost valid column (== iA + 1)
  double jump = 0, uscale = 0;
  for (int j = r * Jy0; j <= r * Jy1 + r - 1; ++j) {
    // exact continuity reference: |u at the two adjacent fine centers| difference is O(h); the SOLVED
    // jump must be comparable, i.e. the discrete potential is continuous across the join (no seam).
    const double xa = geom_f.x_cell(iA), xb = geom_f.x_cell(iB), yy = geom_f.y_cell(j);
    const double ua = u_exact(xa, yy), ub = u_exact(xb, yy);
    jump = std::fmax(jump, std::fabs((PA(iA, j, 0) - PB(iB, j, 0)) - (ua - ub)));
    uscale = std::fmax(uscale, std::fabs(ua));
  }
  jump = all_reduce_max(jump);
  uscale = all_reduce_max(uscale);
  if (my_rank() == 0)
    std::printf("  [adjacent] continuity jump (solved - exact)=%.3e (uscale=%.3e) rfac=%.2e\n",
                jump, uscale, rfac);
  // the seam introduces no discontinuity beyond the discretization error (~1e-2 of the solution).
  EXPECT_TRUE(jump < 1e-2 * std::fmax(uscale, 1e-30))
      << "(continuity) potential continuous across the shared face: jump=" << jump;

  // (iv) MMS: composite beats coarse-only in the refined interior (patch A interior, guarded).
  const int guard = 3;
  const ConstArray4 PC0 = mg0.phi().fab(0).const_array();
  double e_coarse = 0, e_comp = 0;
  for (int J = Ax0 + guard; J <= Ax1; ++J)
    for (int I = Jy0 + guard; I <= Jy1 - guard; ++I)
      for (int tj = 0; tj < r; ++tj)
        for (int ti = 0; ti < r; ++ti) {
          const int iff = r * J + ti, jff = r * I + tj;
          const double ue = u_exact(geom_f.x_cell(iff), geom_f.y_cell(jff));
          e_comp = std::fmax(e_comp, std::fabs(PA(iff, jff, 0) - ue));
          e_coarse =
              std::fmax(e_coarse, std::fabs(detail::fac_bilerp_coarse(PC0, iff, jff, r) - ue));
        }
  e_coarse = all_reduce_max(e_coarse);
  e_comp = all_reduce_max(e_comp);
  if (my_rank() == 0)
    std::printf("  [adjacent] mms e_coarse=%.3e e_composite=%.3e (x%.2f)\n", e_coarse, e_comp,
                e_coarse / std::fmax(e_comp, 1e-30));
  EXPECT_TRUE(e_comp < 0.7 * e_coarse)
      << "(mms) adjacent composite beats coarse-only: e_comp=" << e_comp
      << " e_coarse=" << e_coarse;

  if (my_rank() == 0)
    std::printf("OK test_composite_fac_adjacent\n");
  comm_finalize();
}
