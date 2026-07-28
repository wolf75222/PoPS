// CAPABILITIES Riemann (audit 2026-06 + ADC-590) : HLLC et Roe sont des algorithmes GENERIQUES dont
// la structure physique (onde de contact, dissipation de Roe) est fournie par le MODELE via les
// traits HasHLLCStructure / HasRoeDissipation. La brique pops::Euler porte ces capabilites, donc
// HLLCFlux / RoeFlux l'acceptent par exactement le meme contrat qu'un modele non canonique.
//
// Verifications :
//  (1) DETECTION : pops::Euler satisfait maintenant HasHLLCStructure / HasRoeDissipation (ADC-590) ;
//      HookedEuler (memes hooks ecrits a la main) aussi.
//  (2) ORACLE : pops::Euler et un provider indepedant portant les memes fermetures produisent le
//      meme flux par l'unique pipeline HLLCFlux / RoeFlux, en subsonique/supersonique et x/y.
//  (3) NON-EULER : un modele ISOTHERME 5 VARIABLES avec deux passifs fournit ses hooks HLLC :
//      consistance F*(U,U) == flux(U), cisaillement et passifs stationnaires preserves exactement.
#include <gtest/gtest.h>

#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/physics/fluids/euler.hpp>

#include <cmath>

using pops::Aux;
using pops::Real;

namespace {

// ---------------------------------------------------------------------------------------------
// HookedEuler : pops::Euler + capabilities HLLC/Roe reproduisant EXACTEMENT les formules du chemin
// canonique (Toro 10.37 pour le contact, moyenne de Roe + Harten pour la dissipation).
// ---------------------------------------------------------------------------------------------
struct HookedEuler : pops::Euler {
  POPS_HD Real contact_speed(const State& UL, const State& UR, Real pL, Real pR, Real sL, Real sR,
                             int dir) const {
    const int in = (dir == 0) ? 1 : 2;
    const Real rL = UL[0], rR = UR[0];
    const Real unL = UL[in] / rL, unR = UR[in] / rR;
    return (pR - pL + rL * unL * (sL - unL) - rR * unR * (sR - unR)) /
           (rL * (sL - unL) - rR * (sR - unR));
  }
  POPS_HD State hllc_star_state(const State& U, Real p, Real s, Real sStar, int dir) const {
    const int in = (dir == 0) ? 1 : 2;
    const int it = (dir == 0) ? 2 : 1;
    const Real r = U[0];
    const Real un = U[in] / r;
    const Real fac = r * (s - un) / (s - sStar);
    State Us{};
    Us[0] = fac;
    Us[in] = fac * sStar;
    Us[it] = fac * (U[it] / r);
    Us[3] = fac * (U[3] / r + (sStar - un) * (sStar + p / (r * (s - un))));
    return Us;
  }
  POPS_HD State roe_dissipation(const State& UL, const Aux&, const State& UR, const Aux&,
                                int dir) const {
    const int in = (dir == 0) ? 1 : 2;
    const int it = (dir == 0) ? 2 : 1;
    const Real rL = UL[0], rR = UR[0];
    const Real unL = UL[in] / rL, unR = UR[in] / rR;
    const Real utL = UL[it] / rL, utR = UR[it] / rR;
    const Real pL = pressure(UL), pR = pressure(UR);
    const Real HL = (UL[3] + pL) / rL, HR = (UR[3] + pR) / rR;
    const Real sqL = std::sqrt(rL), sqR = std::sqrt(rR), den = sqL + sqR;
    const Real un = (sqL * unL + sqR * unR) / den;
    const Real ut = (sqL * utL + sqR * utR) / den;
    const Real H = (sqL * HL + sqR * HR) / den;
    const Real rho = sqL * sqR;
    const Real q2 = un * un + ut * ut;
    const Real gm1 = pL / (UL[3] - Real(0.5) * rL * (unL * unL + utL * utL));
    const Real c2 = gm1 * (H - Real(0.5) * q2);
    const Real c = std::sqrt(c2);
    const Real dr = rR - rL, dp = pR - pL, dun = unR - unL, dut = utR - utL;
    const Real a1 = (dp - rho * c * dun) / (Real(2) * c2);
    const Real a2 = dr - dp / c2;
    const Real a3 = rho * dut;
    const Real a5 = (dp + rho * c * dun) / (Real(2) * c2);
    const pops::HartenEntropyFix entropy_fix{Real(0.1)};
    const Real al1 = entropy_fix(un - c, c);
    const Real al2 = un < Real(0) ? -un : un;
    const Real al5 = entropy_fix(un + c, c);
    State d{};
    d[0] = al1 * a1 + al2 * a2 + al5 * a5;
    d[in] = al1 * a1 * (un - c) + al2 * a2 * un + al5 * a5 * (un + c);
    d[it] = al1 * a1 * ut + al2 * (a2 * ut + a3) + al5 * a5 * ut;
    d[3] =
        al1 * a1 * (H - un * c) + al2 * (a2 * Real(0.5) * q2 + a3 * ut) + al5 * a5 * (H + un * c);
    return d;
  }
};

// Same Euler physics with state layout (E, m_y, rho, m_x). The numerical policies must not know
// this permutation: the provider alone maps its declared representation.
struct PermutedEuler {
  using State = pops::StateVec<4>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 4;

  pops::Euler canonical{1.4};

  POPS_HD static pops::Euler::State unpack(const State& value) {
    return pops::Euler::State{value[2], value[3], value[1], value[0]};
  }
  POPS_HD static State pack(const pops::Euler::State& value) {
    return State{value[3], value[2], value[0], value[1]};
  }
  POPS_HD Real pressure(const State& value) const { return canonical.pressure(unpack(value)); }
  POPS_HD State flux(const State& value, const Aux& aux, int axis) const {
    return pack(canonical.flux(unpack(value), aux, axis));
  }
  POPS_HD Real max_wave_speed(const State& value, const Aux& aux, int axis) const {
    return canonical.max_wave_speed(unpack(value), aux, axis);
  }
  POPS_HD void wave_speeds(const State& value, const Aux& aux, int axis, Real& lower,
                           Real& upper) const {
    canonical.wave_speeds(unpack(value), aux, axis, lower, upper);
  }
  POPS_HD Real contact_speed(const State& left, const State& right, Real pressure_left,
                             Real pressure_right, Real lower, Real upper, int axis) const {
    return canonical.contact_speed(unpack(left), unpack(right), pressure_left, pressure_right,
                                   lower, upper, axis);
  }
  POPS_HD State hllc_star_state(const State& value, Real pressure_value, Real speed, Real contact,
                                int axis) const {
    return pack(canonical.hllc_star_state(unpack(value), pressure_value, speed, contact, axis));
  }
  POPS_HD State roe_dissipation(const State& left, const Aux& left_aux, const State& right,
                                const Aux& right_aux, int axis) const {
    return pack(canonical.roe_dissipation(unpack(left), left_aux, unpack(right), right_aux, axis));
  }
};

// ---------------------------------------------------------------------------------------------
// IsoHLLC : fluide isotherme avec deux scalaires passifs (rho, m_x, m_y, q0, q1). La fermeture
// HLLC et le layout complet appartiennent au provider, pas au flux numerique.
// ---------------------------------------------------------------------------------------------
struct IsoHLLC {
  using State = pops::StateVec<5>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 5;
  Real cs2 = 0.5;

  POPS_HD State flux(const State& u, const Aux&, int dir) const {
    const int in = (dir == 0) ? 1 : 2;
    const int it = (dir == 0) ? 2 : 1;
    const Real un = u[in] / u[0];
    State F{};
    F[0] = u[in];
    F[in] = u[in] * un + cs2 * u[0];
    F[it] = u[it] * un;
    F[3] = u[3] * un;
    F[4] = u[4] * un;
    return F;
  }
  POPS_HD Real max_wave_speed(const State& u, const Aux&, int dir) const {
    const int in = (dir == 0) ? 1 : 2;
    const Real un = u[in] / u[0];
    const Real c = std::sqrt(cs2);
    const Real a = un < 0 ? -un : un;
    return a + c;
  }
  POPS_HD void wave_speeds(const State& u, const Aux&, int dir, Real& smin, Real& smax) const {
    const int in = (dir == 0) ? 1 : 2;
    const Real un = u[in] / u[0];
    const Real c = std::sqrt(cs2);
    smin = un - c;
    smax = un + c;
  }
  POPS_HD Real pressure(const State& u) const { return cs2 * u[0]; }
  POPS_HD Real contact_speed(const State& UL, const State& UR, Real pL, Real pR, Real sL, Real sR,
                             int dir) const {
    const int in = (dir == 0) ? 1 : 2;
    const Real rL = UL[0], rR = UR[0];
    const Real unL = UL[in] / rL, unR = UR[in] / rR;
    return (pR - pL + rL * unL * (sL - unL) - rR * unR * (sR - unR)) /
           (rL * (sL - unL) - rR * (sR - unR));
  }
  POPS_HD State hllc_star_state(const State& U, Real /*p*/, Real s, Real sStar, int dir) const {
    const int in = (dir == 0) ? 1 : 2;
    const int it = (dir == 0) ? 2 : 1;
    const Real r = U[0];
    const Real un = U[in] / r;
    const Real fac = r * (s - un) / (s - sStar);
    State Us{};
    Us[0] = fac;
    Us[in] = fac * sStar;
    Us[it] = fac * (U[it] / r);
    Us[3] = fac * (U[3] / r);
    Us[4] = fac * (U[4] / r);
    return Us;
  }
};

using State4 = pops::StateVec<4>;
using StateIso = pops::StateVec<5>;

State4 cons(double rho, double u, double v, double p, double gamma) {
  State4 U{};
  U[0] = rho;
  U[1] = rho * u;
  U[2] = rho * v;
  U[3] = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
  return U;
}

template <int N>
double maxdiff(const pops::StateVec<N>& a, const pops::StateVec<N>& b) {
  double m = 0;
  for (int c = 0; c < N; ++c)
    m = std::fmax(m, std::fabs(a[c] - b[c]));
  return m;
}

template <class Policy, class Model>
typename Model::State face_density(const Policy& policy, const Model& model,
                                   const typename Model::State& left, const Aux& left_providers,
                                   const typename Model::State& right, const Aux& right_providers,
                                   int axis) {
  pops::FluxProviderValues<Model> left_values{}, right_values{};
  left_values[0] = left_providers.phi;
  left_values[1] = left_providers.grad_x;
  left_values[2] = left_providers.grad_y;
  right_values[0] = right_providers.phi;
  right_values[1] = right_providers.grad_x;
  right_values[2] = right_providers.grad_y;
  return pops::evaluate_numerical_flux(
             policy, model, left, pops::bind_flux_providers<Model>(left_values), right,
             pops::bind_flux_providers<Model>(right_values), pops::FaceContext::axis_aligned(axis))
      .checked_density()
      .value;
}

}  // namespace

TEST(test_riemann_capabilities, compile_time_detection) {
  // (1) DETECTION compile-time des capabilities. ADC-590 : pops::Euler PORTE desormais les hooks.
  static_assert(pops::HasHLLCStructure<pops::Euler>,
                "Euler doit satisfaire HasHLLCStructure (ADC-590 : brique a-capabilites)");
  static_assert(pops::HasRoeDissipation<pops::Euler>,
                "Euler doit satisfaire HasRoeDissipation (ADC-590 : brique a-capabilites)");
  static_assert(pops::HasHLLCStructure<HookedEuler>,
                "HookedEuler doit satisfaire HasHLLCStructure");
  static_assert(pops::HasRoeDissipation<HookedEuler>,
                "HookedEuler doit satisfaire HasRoeDissipation");
  static_assert(pops::HasHLLCStructure<PermutedEuler>);
  static_assert(pops::HasRoeDissipation<PermutedEuler>);
  static_assert(pops::HasHLLCStructure<IsoHLLC>, "IsoHLLC doit satisfaire HasHLLCStructure");
  SUCCEED() << "detection des capabilities (Euler a-capabilites, Hooked/Iso capability)";
}

TEST(test_riemann_capabilities, euler_preserves_provider_oracle_through_generic_pipeline) {
  pops::Euler e;
  e.gamma = 1.4;
  HookedEuler oracle;
  oracle.gamma = 1.4;
  pops::HLLCFlux hllc;
  pops::RoeFlux roe;
  Aux a{};

  // The independent provider body is the accepted pre-cutover oracle. Both values travel through
  // the same layout-blind numerical policy; only their constitutive capability differs.
  const State4 pairs[][2] = {
      {cons(1.2, 0.3, -0.1, 1.5, 1.4), cons(0.7, -0.2, 0.4, 0.9, 1.4)},   // subsonique
      {cons(1.0, 8.0, 0.5, 1.0, 1.4), cons(1.5, 12.0, -0.3, 1.3, 1.4)},   // supersonique +
      {cons(0.9, -7.0, 0.2, 1.1, 1.4), cons(1.1, -9.0, -0.4, 0.8, 1.4)},  // supersonique -
  };
  for (const auto& pr : pairs)
    for (int dir = 0; dir < 2; ++dir) {
      const double dh = maxdiff(face_density(hllc, e, pr[0], a, pr[1], a, dir),
                                face_density(hllc, oracle, pr[0], a, pr[1], a, dir));
      EXPECT_EQ(dh, 0.0) << "HLLC provider oracle changed (dir " << dir << ")";
      const double dr = maxdiff(face_density(roe, e, pr[0], a, pr[1], a, dir),
                                face_density(roe, oracle, pr[0], a, pr[1], a, dir));
      EXPECT_EQ(dr, 0.0) << "Roe provider oracle changed (dir " << dir << ")";
    }
}

TEST(test_riemann_capabilities, state_layout_permutation_is_provider_owned) {
  pops::Euler canonical{1.4};
  PermutedEuler permuted;
  pops::HLLCFlux hllc;
  pops::RoeFlux roe;
  Aux aux{};
  const State4 left = cons(1.2, 0.3, -0.1, 1.5, 1.4);
  const State4 right = cons(0.7, -0.2, 0.4, 0.9, 1.4);
  const PermutedEuler::State permuted_left = PermutedEuler::pack(left);
  const PermutedEuler::State permuted_right = PermutedEuler::pack(right);

  for (int axis = 0; axis < 2; ++axis) {
    const auto hllc_reference = face_density(hllc, canonical, left, aux, right, aux, axis);
    const auto hllc_permuted = PermutedEuler::unpack(
        face_density(hllc, permuted, permuted_left, aux, permuted_right, aux, axis));
    const auto roe_reference = face_density(roe, canonical, left, aux, right, aux, axis);
    const auto roe_permuted = PermutedEuler::unpack(
        face_density(roe, permuted, permuted_left, aux, permuted_right, aux, axis));
    EXPECT_EQ(maxdiff(hllc_reference, hllc_permuted), 0.0);
    EXPECT_EQ(maxdiff(roe_reference, roe_permuted), 0.0);
  }
}

TEST(test_riemann_capabilities, orientation_reversal_is_common_to_hllc_and_roe) {
  pops::Euler model{1.4};
  const State4 left = cons(1.2, 0.3, -0.1, 1.5, 1.4);
  const State4 right = cons(0.7, -0.2, 0.4, 0.9, 1.4);
  pops::FluxProviderValues<pops::Euler> values{};
  const auto providers = pops::bind_flux_providers<pops::Euler>(values);
  for (int axis = 0; axis < 2; ++axis) {
    const auto positive_face =
        pops::FaceContext::axis_aligned(axis, Real(1), pops::FaceOrientation::kPositive);
    const auto negative_face =
        pops::FaceContext::axis_aligned(axis, Real(1), pops::FaceOrientation::kNegative);
    const auto assert_reversal = [&](const auto& policy) {
      const auto positive = pops::evaluate_numerical_flux(policy, model, left, providers, right,
                                                          providers, positive_face)
                                .checked_density()
                                .value;
      const auto reversed = pops::evaluate_numerical_flux(policy, model, right, providers, left,
                                                          providers, negative_face)
                                .checked_density()
                                .value;
      for (int component = 0; component < 4; ++component)
        EXPECT_EQ(reversed[component], -positive[component]);
    };
    assert_reversal(pops::HLLCFlux{});
    assert_reversal(pops::RoeFlux{});
  }
}

TEST(test_riemann_capabilities, non_euler_isothermal_hllc_consistency) {
  // (3a) consistance : F*(U, U) == flux(U).
  IsoHLLC iso;
  pops::HLLCFlux hllc;
  Aux a{};
  StateIso U{};
  U[0] = 1.3;
  U[1] = 0.4;
  U[2] = -0.7;
  U[3] = 2.0;
  U[4] = -0.3;
  for (int dir = 0; dir < 2; ++dir) {
    const double d = maxdiff(face_density(hllc, iso, U, a, U, a, dir), iso.flux(U, a, dir));
    EXPECT_LE(d, 1e-13) << "consistance HLLC isotherme (dir " << dir << ")";
  }
}

TEST(test_riemann_capabilities, non_euler_isothermal_preserves_stationary_shear) {
  // (3b) cisaillement stationnaire (un = 0, rho egal, saut tangentiel) : l'onde intermediaire est
  // resolue -> flux tangentiel EXACTEMENT nul (HLLC), la ou HLL le diffuse (terme sL sR dU != 0).
  IsoHLLC iso;
  pops::HLLCFlux hllc;
  pops::HLLFlux hll;
  Aux a{};
  StateIso UL{}, UR{};
  UL[0] = 1.0;
  UL[1] = 0.0;
  UL[2] = 2.0;  // u_t = +2
  UR[0] = 1.0;
  UR[1] = 0.0;
  UR[2] = -3.0;  // u_t = -3
  UL[3] = 2.5;
  UL[4] = -1.0;
  UR[3] = -0.5;
  UR[4] = 4.0;
  const StateIso Fc = face_density(hllc, iso, UL, a, UR, a, 0);
  const StateIso Fh = face_density(hll, iso, UL, a, UR, a, 0);
  EXPECT_LE(std::fabs(Fc[2]), 1e-14) << "HLLC isotherme : cisaillement stationnaire diffuse";
  EXPECT_GE(std::fabs(Fh[2]), 1e-2) << "temoin HLL : le cisaillement devrait etre diffuse";
  EXPECT_LE(std::fabs(Fc[3]), 1e-14) << "HLLC isotherme : scalaire passif diffuse";
  EXPECT_LE(std::fabs(Fc[4]), 1e-14) << "HLLC isotherme : scalaire passif diffuse";
}
