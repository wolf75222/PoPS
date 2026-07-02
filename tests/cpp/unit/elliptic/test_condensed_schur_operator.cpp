// ADC-587 -- native parity of the EXTRACTED condensed-Schur / Lorentz program kernels.
//
// The Phase-4 refactor moved the aux-component-aware Schur functors out of
// include/pops/runtime/program/program_context.hpp into
// include/pops/coupling/schur/program/schur_program_kernels.hpp VERBATIM. This test pins that the move
// changed NO numerics: the extracted pops::coupling::schur::program::detail::Schur*KernelC functors
// produce BIT-IDENTICAL output to the native pops::detail::Schur*Kernel they mirror when fed the same
// state and the same B_z field (the aux-aware variant reads B_z from the aux at c_bz; with c_bz = 0 and
// the same field the two are the same computation). It also pins that the extracted schur_coeff_bc
// helper equals the native coefficient boundary policy (periodic preserved, else Foextrap).
//
// This is a PURE-FUNCTOR test (no System / ProgramContext): the same for_each_cell + MultiFab pattern
// as tests/cpp/unit/elliptic/test_schur_condensation.cpp, so it needs no runtime facade to exercise the
// moved kernels directly.

#include <gtest/gtest.h>

#include <pops/coupling/schur/core/schur_condensation.hpp>  // native detail::Schur*Kernel
#include <pops/coupling/schur/program/schur_program_kernels.hpp>  // extracted detail::Schur*KernelC + schur_coeff_bc

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/linalg/lorentz_eliminator.hpp>
#include <pops/parallel/comm.hpp>

#include <cmath>
#include <cstdio>

using namespace pops;
namespace csp = pops::coupling::schur::program;

static double dabs(double x) {
  return x < 0 ? -x : x;
}

// Unit-square, mono-box grid (round-robin MPI distribution), mirroring test_schur_condensation.cpp.
// Named GridSetup (not "Setup"): ::testing::Test declares a virtual Setup() that would shadow a free
// type "Setup" inside a TEST() body.
struct GridSetup {
  Box2D dom;
  Geometry geom;
  BoxArray ba;
  DistributionMapping dm;
  BCRec bc;
  explicit GridSetup(int n)
      : dom(Box2D::from_extents(n, n)),
        geom{dom, 0.0, 1.0, 0.0, 1.0},
        ba(BoxArray::from_domain(dom, n)),
        dm(ba.size(), n_ranks()) {
    bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
  }
};

// A varying, non-degenerate field so bit-identity is a real check (not two constants agreeing):
// rho / mx / my / B_z are smooth functions of (i, j). Written on the VALID cells.
static void fill_state_and_bz(MultiFab& state, MultiFab& bz) {
  for (int li = 0; li < state.local_size(); ++li) {
    Array4 u = state.fab(li).array();
    Array4 b = bz.fab(li).array();
    const Box2D vb = state.box(li);
    for (int j = vb.lo[1]; j <= vb.hi[1]; ++j)
      for (int i = vb.lo[0]; i <= vb.hi[0]; ++i) {
        u(i, j, 0) = 1.3 + 0.1 * i - 0.05 * j;   // rho (strictly positive over the tested range)
        u(i, j, 1) = 0.4 * i - 0.2 * j;          // mx
        u(i, j, 2) = -0.3 * i + 0.6 * j;         // my
        b(i, j, 0) = 0.7 + 0.02 * i + 0.03 * j;  // B_z
      }
  }
}

// MAX absolute gap between component 0 of two MultiFabs over the local valid cells, reduced across MPI.
static double max_diff(const MultiFab& a, const MultiFab& b) {
  sync_host();  // device_fence under Kokkos::Cuda (no-op serial/OpenMP): make host residence valid.
  double d = 0;
  for (int li = 0; li < a.local_size(); ++li) {
    const ConstArray4 pa = a.fab(li).const_array();
    const ConstArray4 pb = b.fab(li).const_array();
    const Box2D box = a.box(li);
    for (int j = box.lo[1]; j <= box.hi[1]; ++j)
      for (int i = box.lo[0]; i <= box.hi[0]; ++i)
        d = std::fmax(d, dabs(pa(i, j, 0) - pb(i, j, 0)));
  }
  return all_reduce_max(d);
}

// ----------------------------------------------------------------------------------------------
// (A) A_op = I + c rho B^{-1} coefficients: extracted SchurOperatorCoeffKernelC (aux at c_bz=0) vs
//     native SchurOperatorCoeffKernel (dedicated B_z field). Must be BIT-IDENTICAL.
// ----------------------------------------------------------------------------------------------
TEST(test_condensed_schur_operator, coeff_kernel_matches_native_bit_identical) {
  const int n = 24;
  GridSetup S(n);
  const Real alpha = 0.8, theta = 0.5, dt = 0.3;
  const Real c = theta * theta * dt * dt * alpha;  // theta^2 dt^2 alpha
  const Real th_dt = theta * dt;

  MultiFab state(S.ba, S.dm, 3, 0), bz(S.ba, S.dm, 1, 0);
  fill_state_and_bz(state, bz);

  MultiFab exN(S.ba, S.dm, 1, 1), eyN(S.ba, S.dm, 1, 1), axyN(S.ba, S.dm, 1, 1),
      ayxN(S.ba, S.dm, 1, 1);
  MultiFab exC(S.ba, S.dm, 1, 1), eyC(S.ba, S.dm, 1, 1), axyC(S.ba, S.dm, 1, 1),
      ayxC(S.ba, S.dm, 1, 1);
  for (int li = 0; li < state.local_size(); ++li) {
    const ConstArray4 s = state.fab(li).const_array();
    const ConstArray4 b = bz.fab(li).const_array();
    // native: reads B_z from the dedicated field at comp 0.
    for_each_cell(exN.box(li),
                  detail::SchurOperatorCoeffKernel{s, b, exN.fab(li).array(), eyN.fab(li).array(),
                                                   axyN.fab(li).array(), ayxN.fab(li).array(), c,
                                                   th_dt, /*c_rho=*/0});
    // extracted: reads B_z from the "aux" at c_bz=0 (here the same field), otherwise verbatim.
    for_each_cell(exC.box(li), csp::detail::SchurOperatorCoeffKernelC{
                                   s, b, exC.fab(li).array(), eyC.fab(li).array(),
                                   axyC.fab(li).array(), ayxC.fab(li).array(), c, th_dt,
                                   /*c_rho=*/0, /*c_bz=*/0});
  }
  EXPECT_EQ(max_diff(exN, exC), 0.0) << "eps_x extracted != native";
  EXPECT_EQ(max_diff(eyN, eyC), 0.0) << "eps_y extracted != native";
  EXPECT_EQ(max_diff(axyN, axyC), 0.0) << "a_xy extracted != native";
  EXPECT_EQ(max_diff(ayxN, ayxC), 0.0) << "a_yx extracted != native";
  std::printf("(A) coeff kernel: extracted == native bit-identical (n=%d)\n", n);
}

// ----------------------------------------------------------------------------------------------
// (B) explicit flux F = B^{-1}(mx, my): extracted SchurExplicitFluxKernelC (2-comp packed buffer,
//     Fx comp 0 / Fy comp 1, aux at c_bz=0) vs native SchurExplicitFluxKernel (two 1-comp fx/fy).
//     The packed and split layouts hold the SAME values.
// ----------------------------------------------------------------------------------------------
TEST(test_condensed_schur_operator, explicit_flux_matches_native_bit_identical) {
  const int n = 24;
  GridSetup S(n);
  const Real theta = 0.5, dt = 0.3;
  const Real th_dt = theta * dt;

  MultiFab state(S.ba, S.dm, 3, 0), bz(S.ba, S.dm, 1, 0);
  fill_state_and_bz(state, bz);

  MultiFab fxN(S.ba, S.dm, 1, 1), fyN(S.ba, S.dm, 1, 1);  // native: two 1-comp fields
  MultiFab fC(S.ba, S.dm, 2, 1);                          // extracted: one 2-comp packed field
  for (int li = 0; li < state.local_size(); ++li) {
    const ConstArray4 s = state.fab(li).const_array();
    const ConstArray4 b = bz.fab(li).const_array();
    for_each_cell(fxN.box(li), detail::SchurExplicitFluxKernel{s, b, fxN.fab(li).array(),
                                                               fyN.fab(li).array(), th_dt,
                                                               /*c_mx=*/1, /*c_my=*/2});
    for_each_cell(fC.box(li), csp::detail::SchurExplicitFluxKernelC{s, b, fC.fab(li).array(), th_dt,
                                                                    /*c_mx=*/1, /*c_my=*/2,
                                                                    /*c_bz=*/0});
  }
  // Compare Fx (native fxN comp 0 vs packed fC comp 0) and Fy (native fyN comp 0 vs packed fC comp 1).
  sync_host();
  double dfx = 0, dfy = 0;
  for (int li = 0; li < fC.local_size(); ++li) {
    const ConstArray4 pxn = fxN.fab(li).const_array();
    const ConstArray4 pyn = fyN.fab(li).const_array();
    const ConstArray4 pc = fC.fab(li).const_array();
    const Box2D box = fC.box(li);
    for (int j = box.lo[1]; j <= box.hi[1]; ++j)
      for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
        dfx = std::fmax(dfx, dabs(pxn(i, j, 0) - pc(i, j, 0)));
        dfy = std::fmax(dfy, dabs(pyn(i, j, 0) - pc(i, j, 1)));
      }
  }
  EXPECT_EQ(all_reduce_max(dfx), 0.0) << "Fx extracted != native";
  EXPECT_EQ(all_reduce_max(dfy), 0.0) << "Fy extracted != native";
  std::printf("(B) explicit flux: extracted == native bit-identical (n=%d)\n", n);
}

// ----------------------------------------------------------------------------------------------
// (C) schur_coeff_bc: the coefficient / flux boundary policy (periodic preserved, else Foextrap).
//     Extracted csp::schur_coeff_bc must equal the native GeometricMG / Schur coefficient policy.
// ----------------------------------------------------------------------------------------------
TEST(test_condensed_schur_operator, coeff_bc_periodic_preserved_else_foextrap) {
  // Mixed BC: one periodic axis, one Dirichlet axis -> periodic stays, Dirichlet becomes Foextrap.
  BCRec in;
  in.xlo = BCType::Periodic;
  in.xhi = BCType::Periodic;
  in.ylo = BCType::Dirichlet;
  in.yhi = BCType::Dirichlet;
  const BCRec out = csp::schur_coeff_bc(in);
  EXPECT_EQ(out.xlo, BCType::Periodic) << "periodic x-lo must be preserved";
  EXPECT_EQ(out.xhi, BCType::Periodic) << "periodic x-hi must be preserved";
  EXPECT_EQ(out.ylo, BCType::Foextrap) << "physical y-lo must become Foextrap";
  EXPECT_EQ(out.yhi, BCType::Foextrap) << "physical y-hi must become Foextrap";
  std::printf("(C) coeff_bc: periodic preserved, physical -> Foextrap\n");
}
