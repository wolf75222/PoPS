// ADC-636: N-level composite FAC (CompositeFacPoisson general path). Two properties:
//
//   (A) CROSS-CHECK (algebraic reduction). A 2-level, NON adjacent, MONO-RANK input solved through the
//       GENERAL path (force_general_path_for_test) equals the legacy 2-level path bit-for-bit
//       (array_equal on phi_coarse + phi_fine). This documents that the general driver reduces to the
//       historical loop at L == 2 -- the shipping 2-level path stays the verbatim legacy body, so
//       bit-identity there does not depend on this test, but the reduction is proven here.
//
//   (B) 3-LEVEL NESTED MMS. Manufactured u = sin(3 pi x) sin(3 pi y) on [0,1]^2, Dirichlet 0,
//       f = Lap u = -18 pi^2 u. Hierarchy: coarse n x n, level-1 patch on the central half, level-2
//       patch nested in the central quarter of level 1. We check:
//         (i)  finiteness + FAC convergence (composite residual small);
//         (ii) the deepest refinement IMPROVES accuracy: e(level 2) < e(level 1) < e(coarse-only)
//              in the innermost region (the composite discretization is order 2, refined twice);
//         (iii) C/F CONSERVATION to ulp at BOTH interfaces: the covered coarse cells equal the 2x2
//              average of the child (average_down consistency), |cov - avg| at ulp, at the 0-1 AND the
//              1-2 interfaces (two-way at every interface).
//
// Serial (Kokkos OFF); the coarse is mono-box replicated, the FAC mono-rank is validated here.

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
static double f_rhs(double x, double y) {  // Lap u = -(9+9) pi^2 u
  return -18.0 * kPi * kPi * u_exact(x, y);
}

// fill a MultiFab's valid cells with f_rhs at that level's geometry.
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
// (A) cross-check: general path (2-level) == legacy path (array_equal).
// ------------------------------------------------------------------------------------------------
TEST(CompositeFacNlevelTest, general_two_level_equals_legacy) {
  comm_init();
  const int n = 32, r = 2;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom_c{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba_c = BoxArray::from_domain(dom, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
  const int Ic0 = n / 4, Ic1 = 3 * n / 4 - 1;
  Box2D fine_box{{r * Ic0, r * Ic0}, {r * Ic1 + r - 1, r * Ic1 + r - 1}};
  Geometry geom_f = geom_c.refine(r);

  auto fill = [&](CompositeFacPoisson& fac) {
    fill_f(fac.rhs_coarse(), geom_c);
    fill_f(fac.rhs_fine(), geom_f);
  };

  CompositeFacPoisson legacy(geom_c, ba_c, bc, fine_box, r);
  fill(legacy);
  const Real r_legacy = legacy.solve(40, 80, 1e-10, 0.0);

  CompositeFacPoisson general(geom_c, ba_c, bc, fine_box, r);
  fill(general);
  general.force_general_path_for_test(true);  // route the SAME input through the general driver
  const Real r_general = general.solve(40, 80, 1e-10, 0.0);

  EXPECT_EQ(r_legacy, r_general) << "general 2-level residual must equal the legacy residual";

  // array_equal on both levels: the general driver reduces algebraically to the legacy loop.
  double max_c = 0, max_f = 0;
  {
    const ConstArray4 A = legacy.phi_coarse().fab(0).const_array();
    const ConstArray4 B = general.phi_coarse().fab(0).const_array();
    const Box2D b = legacy.phi_coarse().box(0);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        max_c = std::fmax(max_c, std::fabs(A(i, j, 0) - B(i, j, 0)));
  }
  {
    const ConstArray4 A = legacy.phi_fine().fab(0).const_array();
    const ConstArray4 B = general.phi_fine().fab(0).const_array();
    const Box2D b = legacy.phi_fine().box(0);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        max_f = std::fmax(max_f, std::fabs(A(i, j, 0) - B(i, j, 0)));
  }
  if (my_rank() == 0)
    std::printf("  [cross-check] max|phi_c general-legacy|=%.3e max|phi_f|=%.3e\n", max_c, max_f);
  EXPECT_EQ(max_c, 0.0) << "coarse potential: general == legacy bit-identical";
  EXPECT_EQ(max_f, 0.0) << "fine potential: general == legacy bit-identical";
  comm_finalize();
}

// ------------------------------------------------------------------------------------------------
// (B) 3-level nested MMS: accuracy improves per refinement + C/F conservation to ulp.
// ------------------------------------------------------------------------------------------------
TEST(CompositeFacNlevelTest, three_level_nested_mms_order2_and_conservation) {
  comm_init();
  const int n = 48, r = 2;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom_c{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba_c = BoxArray::from_domain(dom, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
  const Geometry geom_1 = geom_c.refine(r);      // level 1: refined once
  const Geometry geom_2 = geom_c.refine(r * r);  // level 2: refined twice

  // level-1 patch = central half [n/4, 3n/4) in coarse -> fine box.
  const int A0 = n / 4, A1 = 3 * n / 4 - 1;  // coarse footprint of level 1
  Box2D box1{{r * A0, r * A0}, {r * A1 + r - 1, r * A1 + r - 1}};
  // level-2 patch = central quarter of the coarse [3n/8, 5n/8) -> level-2 index space (refined x4).
  const int B0 = 3 * n / 8, B1 = 5 * n / 8 - 1;  // coarse footprint of level 2
  Box2D box2{{r * r * B0, r * r * B0}, {r * r * B1 + r * r - 1, r * r * B1 + r * r - 1}};

  std::vector<BoxArray> level_boxes = {BoxArray(std::vector<Box2D>{box1}),
                                       BoxArray(std::vector<Box2D>{box2})};
  CompositeFacPoisson fac(geom_c, ba_c, bc, level_boxes, r);
  ASSERT_EQ(fac.n_levels(), 3);

  fill_f(fac.rhs_level(0), geom_c);
  fill_f(fac.rhs_level(1), geom_1);
  fill_f(fac.rhs_level(2), geom_2);

  // coarse-only reference (a single coarse solve).
  GeometricMG mg0(geom_c, ba_c, bc, {}, /*replicated=*/true);
  fill_f(mg0.rhs(), geom_c);
  mg0.phi().set_val(0.0);
  mg0.solve(1e-12, 100);
  device_fence();

  const Real rfac =
      fac.solve(/*max_iters=*/60, /*fine_sweeps=*/100, /*rel_tol=*/1e-9, /*abs_tol=*/0.0);
  device_fence();

  // (i) finiteness + convergence.
  EXPECT_TRUE(std::isfinite(rfac));
  EXPECT_TRUE(rfac < 1e-2) << "3-level FAC converges (composite residual small): rfac=" << rfac;

  // (ii) accuracy IMPROVES per level in the innermost region (level-2 footprint interior, guarded).
  const int guard = 2;  // coarse cells of margin from the C-F border
  const int gi0 = B0 + guard, gi1 = B1 - guard;
  const ConstArray4 PC0 = mg0.phi().fab(0).const_array();
  const ConstArray4 P1 = fac.phi_level(1).fab(0).const_array();
  const ConstArray4 P2 = fac.phi_level(2).fab(0).const_array();
  double e_coarse = 0, e_l1 = 0, e_l2 = 0;
  for (int J = gi0; J <= gi1; ++J)
    for (int I = gi0; I <= gi1; ++I) {
      // sample at the level-2 cell centers covering coarse cell (I,J): 4x4 fine cells.
      for (int tj = 0; tj < r * r; ++tj)
        for (int ti = 0; ti < r * r; ++ti) {
          const int i2 = r * r * I + ti, j2 = r * r * J + tj;
          const double xf = geom_2.x_cell(i2), yf = geom_2.y_cell(j2);
          const double ue = u_exact(xf, yf);
          e_l2 = std::fmax(e_l2, std::fabs(P2(i2, j2, 0) - ue));
          e_coarse =
              std::fmax(e_coarse, std::fabs(detail::fac_bilerp_coarse(PC0, i2, j2, r * r) - ue));
        }
      // level-1 error sampled at the level-1 cells covering (I,J): 2x2 fine cells.
      for (int tj = 0; tj < r; ++tj)
        for (int ti = 0; ti < r; ++ti) {
          const int i1 = r * I + ti, j1 = r * J + tj;
          const double ue = u_exact(geom_1.x_cell(i1), geom_1.y_cell(j1));
          e_l1 = std::fmax(e_l1, std::fabs(P1(i1, j1, 0) - ue));
        }
    }
  e_coarse = all_reduce_max(e_coarse);
  e_l1 = all_reduce_max(e_l1);
  e_l2 = all_reduce_max(e_l2);
  if (my_rank() == 0)
    std::printf(
        "  [3-level] e_coarse=%.3e e_level1=%.3e e_level2=%.3e (l1/l2=%.2f l2/coarse=%.2f) "
        "rfac=%.2e\n",
        e_coarse, e_l1, e_l2, e_l1 / std::fmax(e_l2, 1e-30), e_l2 / std::fmax(e_coarse, 1e-30),
        rfac);
  EXPECT_TRUE(std::isfinite(e_l1) && std::isfinite(e_l2) && std::isfinite(e_coarse));
  // each refinement reduces the error; the deepest level is the most accurate.
  EXPECT_TRUE(e_l1 < 0.7 * e_coarse)
      << "(order 2) level 1 improves over coarse-only: e_l1=" << e_l1 << " e_coarse=" << e_coarse;
  EXPECT_TRUE(e_l2 < 0.7 * e_l1) << "(order 2) level 2 improves over level 1: e_l2=" << e_l2
                                 << " e_l1=" << e_l1;

  // (iii) C/F CONSERVATION to ulp at BOTH interfaces (covered coarse = 2x2 average of the child).
  auto avgdown_defect = [&](const MultiFab& parent, const MultiFab& child, const Box2D& foot) {
    const ConstArray4 PP = parent.fab(0).const_array();
    const ConstArray4 CF = child.fab(0).const_array();
    double d = 0;
    for (int J = foot.lo[1]; J <= foot.hi[1]; ++J)
      for (int I = foot.lo[0]; I <= foot.hi[0]; ++I) {
        const double avg = 0.25 * (CF(2 * I, 2 * J, 0) + CF(2 * I + 1, 2 * J, 0) +
                                   CF(2 * I, 2 * J + 1, 0) + CF(2 * I + 1, 2 * J + 1, 0));
        d = std::fmax(d, std::fabs(PP(I, J, 0) - avg));
      }
    return d;
  };
  // 0-1 interface: coarse covered = 2x2 average of level 1 over [A0,A1]^2.
  const double d01 =
      all_reduce_max(avgdown_defect(fac.phi_level(0), fac.phi_level(1), Box2D{{A0, A0}, {A1, A1}}));
  // 1-2 interface: level-1 covered = 2x2 average of level 2 over the level-1 footprint of box2
  // (level-1 index space: [2*B0, 2*B1+1]).
  const double d12 =
      all_reduce_max(avgdown_defect(fac.phi_level(1), fac.phi_level(2),
                                    Box2D{{r * B0, r * B0}, {r * B1 + r - 1, r * B1 + r - 1}}));
  if (my_rank() == 0)
    std::printf("  [3-level] avgdown defect: 0-1=%.3e  1-2=%.3e (ulp conservation)\n", d01, d12);
  EXPECT_TRUE(d01 < 1e-12) << "(conservation) 0-1 covered coarse = fine average to ulp: d01="
                           << d01;
  EXPECT_TRUE(d12 < 1e-12) << "(conservation) 1-2 covered level-1 = level-2 average to ulp: d12="
                           << d12;

  if (my_rank() == 0)
    std::printf("OK test_composite_fac_nlevel\n");
  comm_finalize();
}
