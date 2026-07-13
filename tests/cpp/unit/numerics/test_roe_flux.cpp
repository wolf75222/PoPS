// Test du flux de Roe (Euler 2D) + eigenvalues(). Deux verifications rigoureuses :
//  (1) consistance : F*(U, U) == flux(U) (dissipation nulle a etat constant) ;
//  (2) propriete de Roe via l'amont SUPERSONIQUE : si toutes les valeurs propres sont de meme signe
//      (ecoulement supersonique), F* doit valoir EXACTEMENT le flux amont. C'est equivalent a
//      F_R - F_L = A_roe (U_R - U_L) : ca ne passe que si la decomposition en ondes est correcte.
#include <gtest/gtest.h>

#include <pops/physics/fluids/euler.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>

#include <cmath>

using State = pops::StateVec<4>;

namespace {

State cons(double rho, double u, double v, double p, double gamma) {
  State U{};
  U[0] = rho;
  U[1] = rho * u;
  U[2] = rho * v;
  U[3] = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
  return U;
}

double maxdiff(const State& a, const State& b) {
  double m = 0;
  for (int c = 0; c < 4; ++c)
    m = std::fmax(m, std::fabs(a[c] - b[c]));
  return m;
}

template <class Policy>
State face_density(const Policy& policy, const pops::Euler& model, const State& left,
                   const pops::Aux& left_providers, const State& right,
                   const pops::Aux& right_providers, int axis) {
  pops::FluxProviderValues<pops::Euler> left_values{}, right_values{};
  left_values[0] = left_providers.phi;
  left_values[1] = left_providers.grad_x;
  left_values[2] = left_providers.grad_y;
  right_values[0] = right_providers.phi;
  right_values[1] = right_providers.grad_x;
  right_values[2] = right_providers.grad_y;
  return pops::evaluate_numerical_flux(
             policy, model, left, pops::bind_flux_providers<pops::Euler>(left_values), right,
             pops::bind_flux_providers<pops::Euler>(right_values),
             pops::FaceContext::axis_aligned(axis))
      .checked_density()
      .value;
}

}  // namespace

TEST(test_roe_flux, consistent_at_constant_state) {
  pops::Euler e;
  e.gamma = 1.4;
  pops::RoeFlux roe;
  pops::Aux a{};

  // (1) consistance a etat constant, deux etats subsoniques, x et y
  for (const State U : {cons(1.2, 0.3, -0.1, 1.5, 1.4), cons(0.7, -0.2, 0.4, 0.9, 1.4)})
    for (int dir = 0; dir < 2; ++dir) {
      const double d = maxdiff(face_density(roe, e, U, a, U, a, dir), e.flux(U, a, dir));
      EXPECT_LE(d, 1e-12) << "consistance Roe (dir " << dir << ") : " << d;
    }
}

TEST(test_roe_flux, supersonic_upwind_property) {
  pops::Euler e;
  e.gamma = 1.4;
  pops::RoeFlux roe;
  pops::Aux a{};

  // (2) supersonique +x : un >> c pour L ET R -> F* == flux amont (gauche), exact
  {
    const State UL = cons(1.0, 8.0, 0.5, 1.0, 1.4);
    const State UR = cons(1.5, 12.0, -0.3, 1.3, 1.4);  // un ~ 10 >> c ~ 1.2
    const double d = maxdiff(face_density(roe, e, UL, a, UR, a, 0), e.flux(UL, a, 0));
    EXPECT_LE(d, 1e-9) << "Roe supersonique +x (devrait valoir le flux amont gauche) : " << d;
  }
  // supersonique -x : un << -c -> F* == flux amont (droit)
  {
    const State UL = cons(1.2, -12.0, 0.0, 1.1, 1.4);
    const State UR = cons(0.9, -9.0, 0.4, 0.8, 1.4);
    const double d = maxdiff(face_density(roe, e, UL, a, UR, a, 0), e.flux(UR, a, 0));
    EXPECT_LE(d, 1e-9) << "Roe supersonique -x (devrait valoir le flux amont droit) : " << d;
  }
  // supersonique +y
  {
    const State UL = cons(1.0, 0.2, 9.0, 1.0, 1.4);
    const State UR = cons(1.4, -0.1, 13.0, 1.2, 1.4);
    const double d = maxdiff(face_density(roe, e, UL, a, UR, a, 1), e.flux(UL, a, 1));
    EXPECT_LE(d, 1e-9) << "Roe supersonique +y : " << d;
  }
}

TEST(test_roe_flux, eigenvalues_are_vn_minus_c_vn_vn_vn_plus_c) {
  pops::Euler e;
  e.gamma = 1.4;
  pops::Aux a{};

  const State U = cons(1.0, 0.5, -0.2, 1.0, 1.4);
  const double c = std::sqrt(1.4 * 1.0 / 1.0);
  const State ev = e.eigenvalues(U, a, 0);
  EXPECT_LE(std::fabs(ev[0] - (0.5 - c)), 1e-12) << "eigenvalues Euler (0)";
  EXPECT_LE(std::fabs(ev[1] - 0.5), 1e-12) << "eigenvalues Euler (1)";
  EXPECT_LE(std::fabs(ev[2] - 0.5), 1e-12) << "eigenvalues Euler (2)";
  EXPECT_LE(std::fabs(ev[3] - (0.5 + c)), 1e-12) << "eigenvalues Euler (3)";
}
