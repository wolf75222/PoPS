// ADC-636 MPI parity of the composite FAC (CompositeFacPoisson general path). The coarse is
// REPLICATED on every rank and the fine patches are DISTRIBUTED; the composite solve is bit-identical
// distributed-vs-replicated BY CONSTRUCTION (single-writer per-face FluxRegister + collective norms),
// not by all_reduce luck. We solve a manufactured Poisson on 2-level, 3-level and adjacent-patch
// hierarchies and check:
//   (1) CROSS-RANK CONSISTENCY: the coarse potential is replicated, so its checksum + the fine-patch
//       checksums (each patch owned by exactly one rank, gathered) are GLOBAL quantities identical on
//       every rank -- cross-rank spread == 0 exactly. A halo / gather / flux-register bug breaks it.
//   (2) PARITY AT NP: the printed checksums let the build script re-run the SAME binary at np=1/2/4
//       and diff (np=1 = oracle; np=2/4 bit-identical). Serial all_reduce/gather is the identity, so
//       np=1 is also the legacy-equivalent output.
//
// Independent of the backend (Kokkos Serial CI, Cuda GH200 on ROMEO for the device+MPI parity gate).

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;
static constexpr double kPi = 3.14159265358979323846;
static double u_exact(double x, double y) {
  return std::sin(3.0 * kPi * x) * std::sin(3.0 * kPi * y);
}
static double f_rhs(double x, double y) {
  return -18.0 * kPi * kPi * u_exact(x, y);
}

static double periodic_exact(double x, double y) {
  return std::sin(2.0 * kPi * x) * std::sin(2.0 * kPi * y);
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

static void fill_screened_f(MultiFab& f, const Geometry& g, Real reaction) {
  for (int li = 0; li < f.local_size(); ++li) {
    Array4 a = f.fab(li).array();
    const Box2D b = f.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        a(i, j, 0) = -(Real(18) * Real(kPi) * Real(kPi) + reaction) *
                     Real(u_exact(g.x_cell(i), g.y_cell(j)));
  }
}

// checksum of the REPLICATED coarse (identical on every rank) = sum of phi_coarse^2 over the domain.
static double coarse_checksum(CompositeFacPoisson& fac) {
  const ConstArray4 P = fac.phi_coarse().fab(0).const_array();
  const Box2D b = fac.phi_coarse().box(0);
  double s = 0;
  for (int j = b.lo[1]; j <= b.hi[1]; ++j)
    for (int i = b.lo[0]; i <= b.hi[0]; ++i)
      s += P(i, j, 0) * P(i, j, 0);
  return s;
}

// checksum of a DISTRIBUTED fine level, DETERMINISTIC across np. Each patch is owned by exactly one
// rank (single writer); we sum each patch's cells locally into a per-GLOBAL-patch slot, all_reduce_sum
// the slot vector (each slot summed as owner-value + 0 = exact), then fold the slots in a FIXED global
// patch-index order. This removes the rank-dependent accumulation order of a naive per-rank partial +
// all_reduce, so the printed checksum is bit-identical np=1/2/4 (the parity DIFF the build script runs).
static double fine_checksum(MultiFab& phi) {
  const int npatch = phi.box_array().size();
  std::vector<double> slot(npatch, 0.0);
  for (int li = 0; li < phi.local_size(); ++li) {
    const int g = phi.global_index(li);
    const ConstArray4 P = phi.fab(li).const_array();
    const Box2D b = phi.box(li);
    double s = 0;
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        s += P(i, j, 0) * P(i, j, 0);
    slot[g] = s;
  }
  all_reduce_sum_inplace(slot.data(), npatch);
  double total = 0;
  for (int g = 0; g < npatch; ++g)
    total += slot[g];  // fixed global order
  return total;
}

static double spread(double x) {
  return all_reduce_max(x) - (-all_reduce_max(-x));
}

static int pops_run_test_mpi_composite_fac(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int me = my_rank(), np = n_ranks();
  const int n = 48, r = 2;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom_c{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba_c = BoxArray::from_domain(dom, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
  const Geometry g1 = geom_c.refine(r), g2 = geom_c.refine(r * r);

  int fails = 0;

  // --- (A) 2-level, non-adjacent (routed to the general path via the MPI dispatch at np>1) ---
  {
    const int Ic0 = n / 4, Ic1 = 3 * n / 4 - 1;
    Box2D fb{{r * Ic0, r * Ic0}, {r * Ic1 + r - 1, r * Ic1 + r - 1}};
    CompositeFacPoisson fac(geom_c, ba_c, bc, fb, r);
    fill_f(fac.rhs_coarse(), geom_c);
    fill_f(fac.rhs_fine(), g1);
    const Real rf = fac.solve(40, 80, 1e-10, 0.0);
    const double cc = coarse_checksum(fac), fc = fine_checksum(fac.phi_fine());
    const double sp = std::fmax(spread(cc), spread(rf));
    if (me == 0)
      std::printf("FAC2 np=%d rfac=%.17e csum_c=%.17e csum_f=%.17e spread=%.3e\n", np, rf, cc, fc,
                  sp);
    if (!(std::isfinite(rf) && rf < 1e-2)) {
      if (me == 0)
        std::printf("FAIL 2-level not converged\n");
      ++fails;
    }
    if (!(sp == 0.0)) {
      if (me == 0)
        std::printf("FAIL 2-level not bit-identical cross-rank\n");
      ++fails;
    }
  }

  // --- (B) 3-level nested ---
  {
    const int A0 = n / 4, A1 = 3 * n / 4 - 1;
    Box2D b1{{r * A0, r * A0}, {r * A1 + r - 1, r * A1 + r - 1}};
    const int B0 = 3 * n / 8, B1 = 5 * n / 8 - 1;
    Box2D b2{{r * r * B0, r * r * B0}, {r * r * B1 + r * r - 1, r * r * B1 + r * r - 1}};
    std::vector<BoxArray> lb{BoxArray(std::vector<Box2D>{b1}), BoxArray(std::vector<Box2D>{b2})};
    CompositeFacPoisson fac(geom_c, ba_c, bc, lb, r);
    fill_f(fac.rhs_level(0), geom_c);
    fill_f(fac.rhs_level(1), g1);
    fill_f(fac.rhs_level(2), g2);
    const Real rf = fac.solve(60, 100, 1e-9, 0.0);
    const double cc = coarse_checksum(fac);
    const double f1 = fine_checksum(fac.phi_level(1)), f2 = fine_checksum(fac.phi_level(2));
    const double sp = std::fmax(spread(cc), spread(rf));
    if (me == 0)
      std::printf("FAC3 np=%d rfac=%.17e csum_c=%.17e csum_1=%.17e csum_2=%.17e spread=%.3e\n", np,
                  rf, cc, f1, f2, sp);
    if (!(std::isfinite(rf) && rf < 1e-2)) {
      if (me == 0)
        std::printf("FAIL 3-level not converged\n");
      ++fails;
    }
    if (!(sp == 0.0)) {
      if (me == 0)
        std::printf("FAIL 3-level not bit-identical cross-rank\n");
      ++fails;
    }
  }

  // --- (C) adjacent patches (two touching level-1 patches, distributed) ---
  {
    const int Ax0 = n / 4, Ax1 = n / 2 - 1, Bx0 = n / 2, Bx1 = 3 * n / 4 - 1;
    const int Jy0 = n / 4, Jy1 = 3 * n / 4 - 1;
    Box2D fA{{r * Ax0, r * Jy0}, {r * Ax1 + r - 1, r * Jy1 + r - 1}};
    Box2D fB{{r * Bx0, r * Jy0}, {r * Bx1 + r - 1, r * Jy1 + r - 1}};
    BoxArray adj(std::vector<Box2D>{fA, fB});
    CompositeFacPoisson fac(geom_c, ba_c, bc, adj, r);
    fill_f(fac.rhs_coarse(), geom_c);
    fill_f(fac.rhs_fine(), g1);
    const Real rf = fac.solve(60, 100, 1e-9, 0.0);
    const double cc = coarse_checksum(fac), fc = fine_checksum(fac.phi_fine());
    const double sp = std::fmax(spread(cc), spread(rf));
    if (me == 0)
      std::printf("FACADJ np=%d rfac=%.17e csum_c=%.17e csum_f=%.17e spread=%.3e\n", np, rf, cc, fc,
                  sp);
    if (!(std::isfinite(rf) && rf < 1e-2)) {
      if (me == 0)
        std::printf("FAIL adjacent not converged\n");
      ++fails;
    }
    if (!(sp == 0.0)) {
      if (me == 0)
        std::printf("FAIL adjacent not bit-identical cross-rank\n");
      ++fails;
    }
  }

  // --- (D) screened composite: scalar kappa is identical on every rank and every FAC level ---
  {
    const Real reaction = Real(40);
    const int Ic0 = n / 4, Ic1 = 3 * n / 4 - 1;
    Box2D fb{{r * Ic0, r * Ic0}, {r * Ic1 + r - 1, r * Ic1 + r - 1}};
    CompositeFacPoisson fac(geom_c, ba_c, bc, fb, r);
    fac.set_reaction(reaction);
    fill_screened_f(fac.rhs_coarse(), geom_c, reaction);
    fill_screened_f(fac.rhs_fine(), g1, reaction);
    const Real rf = fac.solve(40, 80, 1e-10, 0.0);
    const double cc = coarse_checksum(fac), fc = fine_checksum(fac.phi_fine());
    const double sp = std::fmax(spread(cc), spread(rf));
    if (me == 0)
      std::printf("FACSCREEN np=%d rfac=%.17e csum_c=%.17e csum_f=%.17e spread=%.3e\n", np, rf, cc,
                  fc, sp);
    if (!(std::isfinite(rf) && rf < 1e-2)) {
      if (me == 0)
        std::printf("FAIL screened 2-level not converged\n");
      ++fails;
    }
    if (!(sp == 0.0)) {
      if (me == 0)
        std::printf("FAIL screened 2-level not bit-identical cross-rank\n");
      ++fails;
    }
  }

  // --- (E) singular periodic Poisson, 3 levels, full constant tensor ---
  //
  // The periodic base correction has a constant nullspace.  Independently rounded volume and
  // interface-flux contributions leave a tiny constant component in the internal FAC residual;
  // every rank must project that replicated correction onto the compatible mean-zero range before
  // the coarse multigrid solve.  This case used to grow geometrically to infinity at three levels.
  {
    constexpr Real eps_x = Real(1.2);
    constexpr Real eps_y = Real(0.8);
    constexpr Real a_xy = Real(0.2);
    constexpr Real a_yx = Real(-0.2);
    const Real wave = Real(2) * Real(kPi);
    BCRec periodic;
    periodic.xlo = periodic.xhi = periodic.ylo = periodic.yhi = BCType::Periodic;
    const int A0 = n / 4, A1 = 3 * n / 4 - 1;
    Box2D b1{{r * A0, r * A0}, {r * A1 + r - 1, r * A1 + r - 1}};
    const int B0 = 3 * n / 8, B1 = 5 * n / 8 - 1;
    Box2D b2{{r * r * B0, r * r * B0}, {r * r * B1 + r * r - 1,
                                         r * r * B1 + r * r - 1}};
    std::vector<BoxArray> lb{BoxArray(std::vector<Box2D>{b1}),
                             BoxArray(std::vector<Box2D>{b2})};
    CompositeFacPoisson fac(geom_c, ba_c, periodic, lb, r);
    for (int level = 0; level < fac.n_levels(); ++level) {
      const Geometry& geometry = fac.geom_level(level);
      fac.eps_level(level).set_val(eps_x);
      fac.eps_y_level(level).set_val(eps_y);
      fac.a_xy_level(level).set_val(a_xy);
      fac.a_yx_level(level).set_val(a_yx);
      MultiFab& rhs = fac.rhs_level(level);
      for (int local = 0; local < rhs.local_size(); ++local) {
        Array4 values = rhs.fab(local).array();
        const Box2D valid = rhs.box(local);
        for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
          for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
            values(i, j, 0) =
                -(eps_x + eps_y) * wave * wave *
                periodic_exact(static_cast<Real>(geometry.x_cell(i)),
                               static_cast<Real>(geometry.y_cell(j)));
      }
    }
    fac.use_variable_coefficient(true);
    fac.use_anisotropic_coefficient(true);
    fac.use_cross_terms(true);
    const Real rf = fac.solve(60, 100, 1e-9, 1e-12);
    const double cc = coarse_checksum(fac);
    const double f1 = fine_checksum(fac.phi_level(1));
    const double f2 = fine_checksum(fac.phi_level(2));
    const double sp = std::fmax(spread(cc), spread(rf));
    if (me == 0)
      std::printf(
          "FACPERIODIC3 np=%d rfac=%.17e csum_c=%.17e csum_1=%.17e csum_2=%.17e "
          "spread=%.3e\n",
          np, rf, cc, f1, f2, sp);
    if (!(std::isfinite(rf) && fac.last_solve_report().solved() && rf < 1e-5)) {
      if (me == 0)
        std::printf("FAIL periodic 3-level tensor FAC not converged\n");
      ++fails;
    }
    if (!(sp == 0.0)) {
      if (me == 0)
        std::printf("FAIL periodic 3-level tensor FAC not bit-identical cross-rank\n");
      ++fails;
    }
  }

  if (me == 0 && fails == 0)
    std::printf(
        "OK test_mpi_composite_fac np=%d (replicated coarse + distributed fine, "
        "bit-identical cross-rank)\n",
        np);
  comm_finalize();
  return fails ? 1 : 0;
}

TEST(test_mpi_composite_fac, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_mpi_composite_fac, "test_mpi_composite_fac"), 0);
}
