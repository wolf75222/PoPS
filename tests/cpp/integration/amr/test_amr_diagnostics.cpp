// Diagnostics AMR (coupling/amr_diagnostics.hpp) contre des valeurs analytiques.
// amr_mass (somme u*dV, routee par le seam reducteur for_each_cell_reduce_sum) et
// amr_max_drift_speed (max |grad phi| / B0, reduction Kokkos). Pin les deux independamment
// des coupleurs, puis exerce le chemin complet AmrCouplerMP sur un domaine d'origine non nulle
// et plusieurs boites : gradient, vitesse de derive et vitesse d'onde restent sur le device.

#include <gtest/gtest.h>

#include "load_balance_test_authority.hpp"

#include <pops/coupling/amr/amr_diagnostics.hpp>
#include <pops/coupling/amr/amr_coupler_mp.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace pops;

namespace {

struct AffinePotentialFill {
  Array4 phi;
  POPS_HD void operator()(int i, int j) const {
    phi(i, j, 0) = Real(2) * i - Real(3) * j + Real(7);
  }
};

struct GradientParityError {
  ConstArray4 aux;
  Real extra_sentinel;
  POPS_HD Real operator()(int i, int j) const {
    const Real e_phi = aux(i, j, 0) - (Real(2) * i - Real(3) * j + Real(7));
    const Real e_gx = aux(i, j, 1) - Real(2);
    const Real e_gy = aux(i, j, 2) + Real(3);
    const Real e_extra = aux(i, j, 3) - extra_sentinel;
    const Real a_phi = e_phi < Real(0) ? -e_phi : e_phi;
    const Real a_gx = e_gx < Real(0) ? -e_gx : e_gx;
    const Real a_gy = e_gy < Real(0) ? -e_gy : e_gy;
    const Real a_extra = e_extra < Real(0) ? -e_extra : e_extra;
    const Real gradient = a_gx > a_gy ? a_gx : a_gy;
    const Real field = a_phi > a_extra ? a_phi : a_extra;
    return gradient > field ? gradient : field;
  }
};

struct DiagnosticWaveModel {
  using State = StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;
  Real B0 = Real(2);

  POPS_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  POPS_HD State source(const State&, const Aux&) const { return State{}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
  POPS_HD Real max_wave_speed(const State& state, const Aux& aux, int direction) const {
    const Real state_magnitude = state[0] < Real(0) ? -state[0] : state[0];
    const Real gradient = direction == 0 ? aux.grad_x : aux.grad_y;
    const Real gradient_magnitude = gradient < Real(0) ? -gradient : gradient;
    return Real(direction + 1) * state_magnitude + gradient_magnitude;
  }
};

struct WaveDiagnosticFill {
  Array4 state;
  Array4 aux;
  int ilo;
  int jlo;

  POPS_HD void operator()(int i, int j) const {
    state(i, j, 0) = Real(i - ilo + 1);
    aux(i, j, 0) = Real(0);
    aux(i, j, 1) = Real(0.5) * Real(j - jlo);
    aux(i, j, 2) = Real(0.25) * Real(i - ilo);
  }
};

struct NonFiniteGradientFill {
  Array4 aux;
  int bad_i;
  int bad_j;

  POPS_HD void operator()(int i, int j) const {
    aux(i, j, 0) = Real(0);
    aux(i, j, 1) =
        (i == bad_i && j == bad_j) ? std::numeric_limits<Real>::quiet_NaN() : Real(0);
    aux(i, j, 2) = Real(0);
  }
};

}  // namespace

TEST(test_amr_diagnostics, Runs) {
  const int nc = 16;
  Box2D dom = Box2D::from_extents(nc, nc);
  const Real dx = Real(1) / nc, dy = Real(1) / nc;
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());

  // amr_mass, champ constant u = 2.5 : masse = 2.5 * Lx * Ly = 2.5 (Lx = Ly = 1).
  {
    MultiFab U(ba, dm, 1, 1);
    Array4 u = U.fab(0).array();
    const Box2D g = U.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        u(i, j, 0) = Real(2.5);
    const Real M = amr_mass(U, dom, dx, dy);
    EXPECT_TRUE(std::fabs(M - Real(2.5)) < 1e-13)
        << "amr_mass_constant: M=" << M << " (attendu 2.5)";
  }

  // amr_mass, champ varie : EXACTEMENT la boucle hote de reference en serie (meme dV
  // multiplie dans le noyau, meme ordre lexicographique j-externe/i-interne).
  {
    MultiFab U(ba, dm, 1, 1);
    Array4 u = U.fab(0).array();
    const Real dV = dx * dy;
    Real ref = 0;
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) {
        const Real v = i + 2 * j - Real(0.3) * i * j;
        u(i, j, 0) = v;
        ref += v * dV;
      }
    const Real M = amr_mass(U, dom, dx, dy);
    EXPECT_EQ(M, ref) << "amr_mass_bit_identique_hote: diff=" << std::fabs(M - ref);
  }

  // amr_max_drift_speed, aux a gradient constant (gx, gy) : max = hypot(gx,gy) / B0.
  {
    const Real gx = Real(0.3), gy = Real(-0.4), B0 = Real(2);  // hypot = 0.5 -> 0.25
    MultiFab aux(ba, dm, 3, 1);
    Array4 a = aux.fab(0).array();
    const Box2D g = aux.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        a(i, j, 0) = 0;
        a(i, j, 1) = gx;
        a(i, j, 2) = gy;
      }
    const Real v = amr_max_drift_speed(aux, dom, B0);
    const Real exp = std::hypot(gx, gy) / B0;
    EXPECT_TRUE(std::fabs(v - exp) < 1e-13)
        << "amr_max_drift_speed_const: v=" << v << " (attendu " << exp << ")";
  }

  // amr_max_drift_speed, plancher 1e-12 quand le champ est nul (garde-fou CFL).
  {
    MultiFab aux(ba, dm, 3, 1);
    Array4 a = aux.fab(0).array();
    const Box2D g = aux.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        a(i, j, 0) = 0;
        a(i, j, 1) = 0;
        a(i, j, 2) = 0;
      }
    const Real v = amr_max_drift_speed(aux, dom, Real(1));
    EXPECT_EQ(v, Real(1e-12)) << "amr_max_drift_speed_plancher";
  }
}

TEST(test_amr_diagnostics, DeviceMultiboxNonzeroOriginParity) {
  const Box2D domain{{11, -7}, {18, 0}};
  const BoxArray boxes(
      std::vector<Box2D>{Box2D{{11, -7}, {14, 0}}, Box2D{{15, -7}, {18, 0}}});
  const DistributionMapping mapping(boxes.size(), n_ranks());

  // The exact shared gradient provider must cover every local box and respect global indices.
  MultiFab phi(boxes, mapping, 1, 1);
  MultiFab derived(boxes, mapping, 4, 1);
  constexpr Real extra_sentinel = Real(91);
  derived.set_val(extra_sentinel);
  for (int local = 0; local < phi.local_size(); ++local)
    for_each_cell(phi.fab(local).grown_box(), AffinePotentialFill{phi.fab(local).array()});
  detail::coupler_grad_phi(phi, derived, Real(0.5), Real(0.5));

  Real local_error = Real(0);
  for (int local = 0; local < derived.local_size(); ++local)
    local_error = std::max(
        local_error,
        for_each_cell_reduce_max(
            derived.box(local),
            GradientParityError{derived.fab(local).const_array(), extra_sentinel}));
  EXPECT_EQ(all_reduce_max(local_error), Real(0));

  // Exercise both public AMR CFL diagnostics on the same non-zero-origin, multi-box layout.
  MultiFab state(boxes, mapping, 1, 1);
  std::vector<AmrLevelMP> levels;
  levels.push_back(AmrLevelMP{std::move(state), nullptr, Real(1) / 8, Real(1) / 8});
  const Geometry geometry{domain, Real(0), Real(1), Real(0), Real(1)};
  const auto load_balance = test::prepare_test_space_filling_curve_load_balance();
  AmrCouplerMP<DiagnosticWaveModel> coupler(DiagnosticWaveModel{}, geometry, boxes, BCRec{},
                                            std::move(levels), {},
                                            /*replicated_coarse=*/false, load_balance);
  for (int local = 0; local < coupler.coarse().local_size(); ++local)
    for_each_cell(coupler.coarse().box(local),
                  WaveDiagnosticFill{coupler.coarse().fab(local).array(),
                                     coupler.aux0().fab(local).array(), domain.lo[0],
                                     domain.lo[1]});

  const Real expected_wave = Real(2) * Real(8) + Real(0.25) * Real(7);
  const Real expected_drift = std::hypot(Real(0.5) * Real(7), Real(0.25) * Real(7)) / Real(2);
  const Real wave = coupler.max_wave_speed();
  const Real drift = coupler.max_drift_speed();
  EXPECT_TRUE(std::isfinite(static_cast<double>(wave)) &&
              std::isfinite(static_cast<double>(drift)));
  EXPECT_EQ(wave, expected_wave);
  EXPECT_NEAR(drift, expected_drift, Real(1e-14));
}

TEST(test_amr_diagnostics, RejectsNonFiniteDriftInputs) {
  const Box2D domain{{9, -4}, {12, -1}};
  const BoxArray boxes(std::vector<Box2D>{domain});
  const DistributionMapping mapping(boxes.size(), n_ranks());
  MultiFab aux(boxes, mapping, 3, 0);
  for (int local = 0; local < aux.local_size(); ++local)
    for_each_cell(aux.box(local), NonFiniteGradientFill{aux.fab(local).array(), domain.lo[0],
                                                        domain.lo[1]});

  EXPECT_THROW((void)amr_max_drift_speed(aux, domain, Real(1)), std::domain_error);
  EXPECT_THROW((void)amr_max_drift_speed(aux, domain, Real(0)), std::domain_error);
  EXPECT_THROW((void)amr_max_drift_speed(aux, domain,
                                         std::numeric_limits<Real>::infinity()),
               std::domain_error);

  MultiFab too_narrow(boxes, mapping, 2, 0);
  EXPECT_THROW((void)amr_max_drift_speed_mb(too_narrow, Real(1)), std::invalid_argument);
}

TEST(test_amr_diagnostics, RejectsInvalidSpacingBeforeFieldKernels) {
  const Box2D domain{{4, -8}, {11, -1}};
  const BoxArray boxes(std::vector<Box2D>{domain});
  const DistributionMapping mapping(boxes.size(), n_ranks());
  const auto load_balance = test::prepare_test_space_filling_curve_load_balance();

  {
    MultiFab state(boxes, mapping, 1, 1);
    std::vector<AmrLevelMP> levels;
    levels.push_back(AmrLevelMP{std::move(state), nullptr, Real(1) / 8, Real(1) / 8});
    const Geometry zero_width{domain, Real(0), Real(0), Real(0), Real(1)};
    EXPECT_THROW(
        {
          AmrCouplerMP<DiagnosticWaveModel> invalid(
              DiagnosticWaveModel{}, zero_width, boxes, BCRec{}, std::move(levels), {}, false,
              load_balance);
        },
        std::invalid_argument);
  }

  {
    MultiFab state(boxes, mapping, 1, 1);
    std::vector<AmrLevelMP> levels;
    levels.push_back(AmrLevelMP{std::move(state), nullptr, Real(0), Real(1) / 8});
    const Geometry geometry{domain, Real(0), Real(1), Real(0), Real(1)};
    EXPECT_THROW(
        {
          AmrCouplerMP<DiagnosticWaveModel> invalid(
              DiagnosticWaveModel{}, geometry, boxes, BCRec{}, std::move(levels), {}, false,
              load_balance);
        },
        std::invalid_argument);
  }
}
