// WENO5-Z spatial closures composed with the core SSPRK3 Program primitive.
//
// Verifications :
//  (1) PARITE SCHEMA : make_block("weno5", "rusanov").rhs_into == assemble_rhs<Weno5, RusanovFlux>
//      direct (le dispatch route bien vers Weno5 ; spatial_operator branche weno5z a n_ghost >= 3).
//  (2) SSPRK3 : le stepper coeur consomme directement le residu spatial, avec scratch reutilisable.
//  (3) ORDRE / PRECISION : advection lineaire d'un sinus lisse periodique sur une periode complete.
//      WENO5+SSPRK3 a une erreur < Minmod+SSPRK2 a meme resolution, et une pente de convergence > 2
//      (au-dela de l'ordre 2 du MUSCL). Test court (n <= 64), CI-friendly.
#include <gtest/gtest.h>

#include <pops/validation/physics/advection_diffusion.hpp>  // pops::validation::AdvectionDiffusion : transport scalaire (nu=0 = advection pure)
#include <pops/runtime/builders/block/block_builder.hpp>

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/spatial_operator.hpp>
#include <pops/numerics/time/integrators/time_steppers.hpp>

#include <cmath>
#include <cstdio>
#include <string>

using namespace pops;

namespace {
constexpr double kPi = 3.14159265358979323846;

// Erreur L1 (cellules valides) entre u(.,0) et la solution exacte advectee d'une periode (== u0).
double advect_error(int n, const std::string& limiter, const std::string& method) {
  const double L = 1.0;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;  // periodique
  MultiFab aux(ba, dm, 3, 1);
  aux.set_val(0.0);

  pops::validation::AdvectionDiffusion model{/*ax=*/1.0, /*ay=*/0.0,
                                             /*nu=*/0.0};  // advection pure selon x
  const int ng = block_n_ghost(limiter);
  MultiFab U(ba, dm, 1, ng);
  auto init = [&](MultiFab& mf) {
    Array4 a = mf.fab(0).array();
    for_each_cell(dom, [a, geom](int i, int j) {
      a(i, j, 0) = std::sin(2 * kPi * geom.x_cell(i));  // sinus lisse, une periode en x
    });
  };
  init(U);

  const GridContext ctx{dom, bc, geom, &aux};
  BlockClosures clo = make_block(model, limiter, "rusanov", ctx, /*recon_prim=*/false);
  // Avance d'une PERIODE (t = L/ax = 1) a CFL fixe : u(t=1) == u0 exactement (advection periodique).
  const double dx = geom.dx();
  const double cfl = 0.4;
  const double dt = cfl * dx / 1.0;  // |a| = 1
  const int nsteps = static_cast<int>(std::ceil(1.0 / dt));
  const double dt_exact = 1.0 / nsteps;  // ajuste pour terminer pile a t=1
  auto rhs = [&](MultiFab& state, MultiFab& residual) { clo.rhs_into(state, residual); };
  if (method == "ssprk3")
    run_explicit_substeps<SSPRK3Step>(rhs, U, dt_exact, nsteps);
  else
    run_explicit_substeps<SSPRK2Step>(rhs, U, dt_exact, nsteps);

  double err = 0;
  const ConstArray4 u = U.fab(0).const_array();
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      err += std::fabs(u(i, j, 0) - std::sin(2 * kPi * geom.x_cell(i)));
  return err / (static_cast<double>(n) * n);  // L1 moyenne
}

}  // namespace

// (1) PARITE SCHEMA : weno5 alloue 3 ghosts ; make_block route vers Weno5 -> rhs_into ==
// assemble_rhs<Weno5, RusanovFlux> direct (meme reconstruction weno5z).
TEST(test_weno5_ssprk3, weno5_rhs_matches_direct_assemble_rhs) {
  const int n = 48;
  const double L = 1.0;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;  // periodique
  MultiFab aux(ba, dm, 3, 1);
  aux.set_val(0.0);

  pops::validation::AdvectionDiffusion model{/*ax=*/1.0, /*ay=*/0.3,
                                             /*nu=*/0.0};  // advection 2D oblique, pure
  const GridContext ctx{dom, bc, geom, &aux};

  auto init = [&](MultiFab& mf) {
    Array4 a = mf.fab(0).array();
    for_each_cell(dom, [a, geom](int i, int j) {
      a(i, j, 0) = std::sin(2 * kPi * geom.x_cell(i)) * std::cos(2 * kPi * geom.y_cell(j));
    });
  };

  EXPECT_EQ(block_n_ghost("weno5"), 3) << "block_n_ghost(weno5) == 3";

  MultiFab U(ba, dm, 1, 3);
  init(U);
  BlockClosures clo = make_block(model, "weno5", "rusanov", ctx, false);
  MultiFab R1(ba, dm, 1, 0), R2(ba, dm, 1, 0);
  clo.rhs_into(U, R1);
  fill_ghosts(U, dom, bc);
  assemble_rhs<Weno5, RusanovFlux>(model, U, aux, geom, R2, false);
  double dres = 0, nrm = 0;
  const ConstArray4 r1 = R1.fab(0).const_array(), r2 = R2.fab(0).const_array();
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      dres = std::fmax(dres, std::fabs(r1(i, j, 0) - r2(i, j, 0)));
      nrm = std::fmax(nrm, std::fabs(r2(i, j, 0)));
    }
  EXPECT_LT(dres, 1e-14) << "make_block(weno5).rhs_into == assemble_rhs<Weno5> direct";
  EXPECT_GT(nrm, 1e-6) << "residu WENO5 non trivial";
}

// (2) SSPRK3 : le helper de sous-pas du coeur et un step one-shot consomment exactement la meme
// primitive spatiale weno5/rusanov.
TEST(test_weno5_ssprk3, spatial_residual_composes_with_core_ssprk3) {
  const int n = 48;
  const double L = 1.0;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;  // periodique
  MultiFab aux(ba, dm, 3, 1);
  aux.set_val(0.0);

  pops::validation::AdvectionDiffusion model{/*ax=*/1.0, /*ay=*/0.3, /*nu=*/0.0};
  const GridContext ctx{dom, bc, geom, &aux};
  auto init = [&](MultiFab& mf) {
    Array4 a = mf.fab(0).array();
    for_each_cell(dom, [a, geom](int i, int j) {
      a(i, j, 0) = std::sin(2 * kPi * geom.x_cell(i)) * std::cos(2 * kPi * geom.y_cell(j));
    });
  };

  MultiFab U(ba, dm, 1, 3), Uref(ba, dm, 1, 3);
  init(U);
  init(Uref);
  BlockClosures clo = make_block(model, "weno5", "rusanov", ctx, false);
  const double dt = 1e-3;
  auto rhs = [&](MultiFab& state, MultiFab& residual) { clo.rhs_into(state, residual); };
  run_explicit_substeps<SSPRK3Step>(rhs, U, dt, 1);
  SSPRK3Step{}.take_step(rhs, Uref, dt);
  double d = 0;
  const ConstArray4 u = U.fab(0).const_array(), ur = Uref.fab(0).const_array();
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      d = std::fmax(d, std::fabs(u(i, j, 0) - ur(i, j, 0)));
  EXPECT_LT(d, 1e-14) << "Program SSPRK3 helper == one-shot SSPRK3 on the same spatial residual";
}

// (2b) REUSE DU SCRATCH (ADC-261) : une avance a n>1 sous-pas reutilise UN seul Scratch hoiste a
// travers les sous-pas (run_explicit_substeps) ; elle doit egaler n take_step one-shot SSPRK3Step
// separes (scratch frais a chaque appel, h=dt/n). Verrouille le chemin de REUTILISATION, pas couvert
// par le cas n=1 ci-dessus.
TEST(test_weno5_ssprk3, substep_scratch_reuse_matches_oneshot_steps) {
  const int n = 48;
  const double L = 1.0;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;
  MultiFab aux(ba, dm, 3, 1);
  aux.set_val(0.0);

  pops::validation::AdvectionDiffusion model{/*ax=*/1.0, /*ay=*/0.3, /*nu=*/0.0};
  const GridContext ctx{dom, bc, geom, &aux};
  auto init = [&](MultiFab& mf) {
    Array4 a = mf.fab(0).array();
    for_each_cell(dom, [a, geom](int i, int j) {
      a(i, j, 0) = std::sin(2 * kPi * geom.x_cell(i)) * std::cos(2 * kPi * geom.y_cell(j));
    });
  };

  MultiFab U(ba, dm, 1, 3), Uref(ba, dm, 1, 3);
  init(U);
  init(Uref);
  BlockClosures clo = make_block(model, "weno5", "rusanov", ctx, false);
  const double dt = 4e-3;
  const int nsub = 4;
  auto rhs = [&](MultiFab& state, MultiFab& residual) { clo.rhs_into(state, residual); };
  const double h = dt / nsub;
  run_explicit_substeps<SSPRK3Step>(rhs, U, h, nsub);
  for (int s = 0; s < nsub; ++s)
    SSPRK3Step{}.take_step(rhs, Uref, h);
  double d = 0;
  const ConstArray4 u = U.fab(0).const_array(), ur = Uref.fab(0).const_array();
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      d = std::fmax(d, std::fabs(u(i, j, 0) - ur(i, j, 0)));
  EXPECT_LT(d, 1e-14)
      << "run_explicit_substeps(n=4) == 4x take_step one-shot (reuse du scratch bit-identique)";
}

// (3) ORDRE / PRECISION : advection 1D d'un sinus sur une periode. WENO5+SSPRK3 plus precis que
// Minmod+SSPRK2 a meme n, et pente de convergence > 2 (au-dela de l'ordre 2 du MUSCL).
TEST(test_weno5_ssprk3, weno5_ssprk3_more_accurate_and_higher_order_than_minmod) {
  const double e_minmod_64 = advect_error(64, "minmod", "explicit");
  const double e_weno_64 = advect_error(64, "weno5", "ssprk3");
  EXPECT_LT(e_weno_64, e_minmod_64) << "WENO5+SSPRK3 plus precis que Minmod+SSPRK2 a n=64";
  const double e_weno_32 = advect_error(32, "weno5", "ssprk3");
  const double slope = std::log(e_weno_32 / e_weno_64) / std::log(2.0);
  std::printf("  WENO5 L1 : n=32 %.3e  n=64 %.3e  pente %.2f (Minmod n=64 %.3e)\n", e_weno_32,
              e_weno_64, slope, e_minmod_64);
  EXPECT_GT(slope, 2.0) << "WENO5 : pente de convergence > 2 (ordre superieur a O2)";
}
