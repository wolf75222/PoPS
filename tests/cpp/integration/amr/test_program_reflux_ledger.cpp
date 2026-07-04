// ADC-639: the conservative-reflux effective-flux LEDGER + the dt-integrated route_reflux variant for a
// whole-system compiled Program on AMR. Two contracts are frozen here, WITHOUT an engine:
//   - CoarseFineInterface::route_reflux_integrated: the /dt-free correction -(fL - cL)/dx (both sides
//     already dt-integrated in the ledger), bit-identical to route_reflux with dt == 1;
//   - the EdgeFlux ledger arithmetic (detail::edge_flux_axpy) reproduces Feff = sum_i w_i F_i through the
//     Program's linear combination -- the SSPRK2 and midpoint worked examples of design-639 section 2b.
// The full engine path (capture closures + route_reflux_program) is covered by the Python acceptances
// (test_amr_program_reflux.py, test_amr_history_regrid.py); this pins the arithmetic the driver relies on.

#include <gtest/gtest.h>

#include <pops/numerics/time/amr/reflux/amr_reflux_mf.hpp>  // CoarseFineInterface / FluxRegister
#include <pops/runtime/amr/amr_program_reflux.hpp>          // EdgeFlux / EdgeStrip / detail::edge_flux_axpy
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>

#include <vector>

using namespace pops;

namespace {
// One fine patch [4..11]^2 -> coarse footprint [2..5]^2 (PatchRange). One component.
EdgeStrip make_strip(int I0, int I1, int J0, int J1, int nc) {
  EdgeStrip s;
  s.alloc(Box2D{{I0, J0}, {I1, J1}}, nc);
  return s;
}
}  // namespace

TEST(test_program_reflux_ledger, route_reflux_integrated_drops_dt) {
  // Same fixture as test_cf_interface but the strips already carry dt*Feff, so the correction is
  // -(fL - cL)/dx (NO *dt). We check the integrated variant equals route_reflux with dt == 1.
  const int nc = 1;
  BoxArray fine(std::vector<Box2D>{Box2D{{4, 4}, {11, 11}}});
  CoarseFineInterface cfi(Box2D{{0, 0}, {7, 7}}, fine);
  EdgeStrip g = make_strip(2, 5, 2, 5, nc);
  const int nJ = 4 * nc, nI = 4 * nc;
  g.cL.assign(nJ, Real(1));
  g.cR.assign(nJ, Real(2));
  g.cB.assign(nI, Real(3));
  g.cT.assign(nI, Real(4));
  g.fL.assign(nJ, Real(10));
  g.fR.assign(nJ, Real(20));
  g.fB.assign(nI, Real(30));
  g.fT.assign(nI, Real(40));
  const Real dx = Real(0.5), dy = Real(0.25);
  FluxRegister ref(Box2D{{1, 1}, {6, 6}}, nc);
  cfi.route_reflux_integrated(g, dx, dy, ref, nc);
  // -(fL - cL)/dx = -(10 - 1)/0.5 = -18; +(fR - cR)/dx = (20 - 2)/0.5 = 36.
  EXPECT_EQ(ref.at(1, 2, 0), -(Real(10) - Real(1)) / dx) << "integrated_left";
  EXPECT_EQ(ref.at(6, 2, 0), +(Real(20) - Real(2)) / dx) << "integrated_right";
  EXPECT_EQ(ref.at(2, 1, 0), -(Real(30) - Real(3)) / dy) << "integrated_bottom";
  EXPECT_EQ(ref.at(2, 6, 0), +(Real(40) - Real(4)) / dy) << "integrated_top";

  // Equivalence to route_reflux at dt == 1 (both formulas coincide when the coarse side is not re-scaled).
  FluxRegister ref_dt1(Box2D{{1, 1}, {6, 6}}, nc);
  cfi.route_reflux(g, dx, dy, Real(1), ref_dt1, nc);
  EXPECT_EQ(ref.at(1, 2, 0), ref_dt1.at(1, 2, 0)) << "integrated_eq_dt1";
}

// A helper building a single-patch EdgeFlux carrying the FINE-role strip with a scalar flux value F, so
// the ledger arithmetic can be checked on a representative face. The x-left face carries F on the whole
// strip; the other faces mirror it (a uniform flux -- enough to prove the linear-combination weights).
static EdgeFlux fine_flux(Real F, int nc = 1) {
  EdgeStrip s = make_strip(2, 5, 2, 5, nc);
  const std::size_t nJ = s.fL.size(), nI = s.fB.size();
  s.fL.assign(nJ, F);
  s.fR.assign(nJ, F);
  s.fB.assign(nI, F);
  s.fT.assign(nI, F);
  EdgeFlux ef;
  ef.fine.push_back(s);
  return ef;
}

TEST(test_program_reflux_ledger, ssprk2_effective_flux) {
  // SSPRK2 (design 2b): the terminal commit strip must hold dt*(1/2 F0 + 1/2 F1).
  // Emulate the codegen lowering:
  //   k0.ledger = F0;  k1.ledger = F1;
  //   U1 = axpy(0, 1*U0) + axpy(dt*k0)          -> u_U1.ledger = dt*F0
  //   acc = axpy(0.5*U1) + axpy(0.5*dt*k1)      -> acc.ledger  = 0.5*dt*F0 + 0.5*dt*F1
  //   commit: lincomb(U, 0.5, U, 1, acc)        -> U.ledger    = 0.5*0 + acc = dt*(1/2 F0 + 1/2 F1)
  const Real dt = Real(0.1), F0 = Real(3), F1 = Real(5);
  const EdgeFlux led_k0 = fine_flux(F0), led_k1 = fine_flux(F1);

  EdgeFlux u_U1;                                   // U1 = U0 + dt*k0 (U0 has no flux)
  detail::edge_flux_axpy(u_U1, dt, led_k0);        // ledger[U1] += dt*F0

  EdgeFlux acc;                                    // acc = 0.5*U1 + 0.5*dt*k1
  detail::edge_flux_axpy(acc, Real(0.5), u_U1);    // += 0.5*dt*F0
  detail::edge_flux_axpy(acc, Real(0.5) * dt, led_k1);  // += 0.5*dt*F1

  EdgeFlux commit;                                 // lincomb(U, 0.5, U(=0), 1, acc)
  detail::edge_flux_axpy(commit, Real(1), acc);    // U.ledger = acc

  // Same grouping as the ledger accumulation (0.5*(dt*F0) then += (0.5*dt)*F1): the exact reflux value.
  const Real expected = Real(0.5) * (dt * F0) + (Real(0.5) * dt) * F1;
  ASSERT_FALSE(commit.fine.empty());
  EXPECT_EQ(commit.fine[0].fL[0], expected) << "ssprk2_Feff";
  EXPECT_EQ(commit.fine[0].fT[0], expected) << "ssprk2_Feff_top";
}

TEST(test_program_reflux_ledger, midpoint_effective_flux_is_F1_only) {
  // Midpoint (design 2b): U1 = U0 + 0.5 dt k0; U <<= U0 + dt k1. The terminal strip must hold dt*F1
  // (the second-stage flux ONLY) -- proving the Program text, not a hard-coded RK, drives Feff.
  const Real dt = Real(0.1), F0 = Real(3), F1 = Real(5);
  const EdgeFlux led_k0 = fine_flux(F0), led_k1 = fine_flux(F1);

  EdgeFlux u_U1;                                   // U1 = U0 + 0.5 dt k0
  detail::edge_flux_axpy(u_U1, Real(0.5) * dt, led_k0);  // ledger[U1] = 0.5 dt F0 (not read by the commit)

  EdgeFlux acc;                                    // acc = dt*k1  (U0 is the base, coeff on k1 is dt)
  detail::edge_flux_axpy(acc, dt, led_k1);         // += dt*F1

  EdgeFlux commit;                                 // lincomb(U, 1, U(=0), 1, acc)
  detail::edge_flux_axpy(commit, Real(1), acc);

  ASSERT_FALSE(commit.fine.empty());
  EXPECT_EQ(commit.fine[0].fL[0], dt * F1) << "midpoint_Feff_is_F1";
  // and NOT the SSPRK2 answer (proves the weights come from the Program combine).
  EXPECT_NE(commit.fine[0].fL[0], dt * (Real(0.5) * F0 + Real(0.5) * F1)) << "midpoint_differs";
}

TEST(test_program_reflux_ledger, ab2_effective_flux_uses_lagged_flux) {
  // AB2 (design 2b / acceptance e): U <<= U + dt*(1.5 R_n - 0.5 R_{n-1}). The lagged R_{n-1} flux rides
  // with the history ring (ring_flux_ in the context); here we emulate the commit combine given that both
  // R_n and R_{n-1} strips are present in the ledger. The terminal strip must hold dt*(1.5 F_n - 0.5 F_nm1).
  const Real dt = Real(0.1), Fn = Real(7), Fnm1 = Real(2);
  const EdgeFlux led_Rn = fine_flux(Fn), led_Rnm1 = fine_flux(Fnm1);

  EdgeFlux acc;                                    // acc = 1.5 dt R_n + (-0.5) dt R_{n-1}
  detail::edge_flux_axpy(acc, Real(1.5) * dt, led_Rn);
  detail::edge_flux_axpy(acc, Real(-0.5) * dt, led_Rnm1);

  EdgeFlux commit;                                 // lincomb(U, 1, U(=0), 1, acc)
  detail::edge_flux_axpy(commit, Real(1), acc);

  // The ledger accumulates the two weighted terms in SEQUENCE (mirror of the codegen axpy chain), so the
  // expected value must use the SAME grouping / rounding order -- 1.5*dt*Fn then += (-0.5*dt)*Fnm1 -- not a
  // single fused dt*(1.5 Fn - 0.5 Fnm1) (a different float rounding). This IS the reflux the driver applies.
  const Real expected = Real(1.5) * dt * Fn + (Real(-0.5) * dt) * Fnm1;
  ASSERT_FALSE(commit.fine.empty());
  EXPECT_EQ(commit.fine[0].fL[0], expected) << "ab2_Feff_uses_lagged";
}
