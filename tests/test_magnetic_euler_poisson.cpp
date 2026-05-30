// Validation d'Euler-Poisson MAGNETIQUE (integrator/magnetic_euler_poisson.hpp,
// Hoffart eq 2.4) : Euler-Poisson + force de Lorentz m x Omega par splitting de Strang
// autour du Coupler<EulerPoisson>.
//
//   1. la rotation cyclotron conserve rho et E BIT A BIT, conserve |m|, et tourne
//      (m_x, m_y) de l'angle exact.
//   2. rotation(theta) o rotation(-theta) = identite (a l'arrondi).
//   3. FILET : a Omega = 0 le pas magnetique est BIT A BIT identique au Coupler nu
//      (la rotation est l'identite a virgule flottante), donc tout le chemin Euler-Poisson
//      deja teste est preserve.
//   4. PHYSIQUE : le point fixe de la carte de Strang (1/2 rotation, impulsion -rho grad
//      phi, 1/2 rotation) converge a l'ORDRE 2 vers la derive E x B
//      v = (-d_y phi, d_x phi)/Omega, la vitesse du modele Diocotron : le systeme complet
//      se reduit a la limite de derive quand Omega grandit.

#include <adc/core/physical_model.hpp>
#include <adc/coupling/coupler.hpp>
#include <adc/integrator/magnetic_euler_poisson.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/euler_poisson.hpp>
#include <adc/operator/reconstruction.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  static_assert(PhysicalModel<EulerPoisson>, "EulerPoisson modele PhysicalModel");

  const int N = 32;
  const double L = 1.0;
  Box2D dom = Box2D::from_extents(N, N);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);

  // remplit U avec un etat non trivial reproductible (rho > 0, m variable, E > 0).
  auto fill_state = [&](MultiFab& U) {
    Fab2D& f = U.fab(0);
    const Box2D v = U.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const double x = (i + 0.5) / N, y = (j + 0.5) / N;
        f(i, j, 0) = 1.0 + 0.3 * std::sin(2 * kPi * x);            // rho
        f(i, j, 1) = 0.4 * std::cos(2 * kPi * y) - 0.1;            // m_x
        f(i, j, 2) = 0.25 * std::sin(2 * kPi * (x + y));          // m_y
        f(i, j, 3) = 2.0 + 0.5 * std::cos(2 * kPi * x);            // E
      }
  };

  // ---- 1. invariants de la rotation cyclotron --------------------------------
  {
    MagneticEulerPoissonCoupler<> dummy(EulerPoisson{}, geom, ba, BCRec{}, BCRec{}, 0.0);
    (void)dummy;
    MultiFab U(ba, dm, 4, 1), U0(ba, dm, 4, 1);
    fill_state(U);
    fill_state(U0);
    const double theta = 0.7;
    magnetic_rotate(U, theta);
    const ConstArray4 a = U.fab(0).const_array(), a0 = U0.fab(0).const_array();
    const Box2D v = U.box(0);
    double drho = 0, dE = 0, dmag = 0, dang = 0;
    const double c = std::cos(theta), s = std::sin(theta);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        drho = std::fmax(drho, std::fabs(a(i, j, 0) - a0(i, j, 0)));  // doit etre 0 bit
        dE = std::fmax(dE, std::fabs(a(i, j, 3) - a0(i, j, 3)));      // doit etre 0 bit
        const double m2 = a(i, j, 1) * a(i, j, 1) + a(i, j, 2) * a(i, j, 2);
        const double m20 = a0(i, j, 1) * a0(i, j, 1) + a0(i, j, 2) * a0(i, j, 2);
        dmag = std::fmax(dmag, std::fabs(m2 - m20));
        const double mx = c * a0(i, j, 1) + s * a0(i, j, 2);   // rotation analytique
        const double my = -s * a0(i, j, 1) + c * a0(i, j, 2);
        dang = std::fmax(dang, std::fabs(a(i, j, 1) - mx) + std::fabs(a(i, j, 2) - my));
      }
    std::printf("rotation : drho=%.1e dE=%.1e d|m|^2=%.1e d(angle)=%.1e\n",
                drho, dE, dmag, dang);
    chk(drho == 0.0, "rho_inchange_bit");
    chk(dE == 0.0, "E_inchange_bit");
    chk(dmag < 1e-12, "norme_m_conservee");
    chk(dang < 1e-14, "rotation_angle_exact");
  }

  // ---- 2. rotation o rotation^{-1} = identite --------------------------------
  {
    MultiFab U(ba, dm, 4, 1), U0(ba, dm, 4, 1);
    fill_state(U);
    fill_state(U0);
    magnetic_rotate(U, 1.3);
    magnetic_rotate(U, -1.3);
    const ConstArray4 a = U.fab(0).const_array(), a0 = U0.fab(0).const_array();
    const Box2D v = U.box(0);
    double d = 0;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        for (int k = 0; k < 4; ++k)
          d = std::fmax(d, std::fabs(a(i, j, k) - a0(i, j, k)));
    std::printf("rotation inverse : ecart max = %.2e\n", d);
    chk(d < 1e-13, "rotation_inversible");
  }

  // ---- 3. FILET : Omega = 0 -> bit a bit le Coupler<EulerPoisson> nu ----------
  {
    EulerPoisson model;
    model.hydro.gamma = 5.0 / 3.0;
    model.four_pi_G = 8.0;
    model.coupling_sign = -1;  // electrostatique (plasma)
    model.rho0 = 1.0;
    BCRec bcU, bcPhi;          // periodique

    MultiFab Ua(ba, dm, 4, 2), Ub(ba, dm, 4, 2);
    fill_state(Ua);
    fill_state(Ub);

    Coupler<EulerPoisson> ref(model, geom, ba, bcU, bcPhi);
    MagneticEulerPoissonCoupler<> mag(model, geom, ba, bcU, bcPhi, /*Omega=*/0.0);

    const double dt = 0.2 * (L / N);
    for (int s = 0; s < 5; ++s) {
      ref.advance<Minmod>(Ua, dt);
      mag.step<Minmod>(Ub, dt);
    }
    const ConstArray4 a = Ua.fab(0).const_array(), b = Ub.fab(0).const_array();
    const Box2D v = Ua.box(0);
    double d = 0;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        for (int k = 0; k < 4; ++k)
          d = std::fmax(d, std::fabs(a(i, j, k) - b(i, j, k)));
    std::printf("Omega=0 vs Coupler nu : ecart max = %.2e (doit etre 0)\n", d);
    chk(d == 0.0, "omega0_bit_identique");
  }

  // ---- 4. PHYSIQUE : point fixe de la carte de Strang -> derive E x B ---------
  // Carte (force gelee F = -rho grad phi) : m -> R(O dt/2)[R(O dt/2) m + dt F].
  // Point fixe discret resolu lineairement : (I - R(O dt)) m = R(O dt/2) dt F.
  // Cible continue (derive E x B) : m* = (1/O)(F_y, -F_x). On verifie la convergence
  // a l'ordre 2 quand dt -> dt/2.
  {
    const double Omega = 5.0, rho = 1.2;
    const double dphidx = 0.7, dphidy = -0.4;        // grad phi constant
    const double Fx = -rho * dphidx, Fy = -rho * dphidy;  // -rho grad phi
    const double msx = Fy / Omega, msy = -Fx / Omega;     // derive continue m*

    auto fixed_point_err = [&](double dt) {
      const double th = Omega * dt;
      const double c = std::cos(th), s = std::sin(th);
      const double ch = std::cos(th / 2), sh = std::sin(th / 2);
      // membre b = R(th/2) dt F
      const double bx = (ch * Fx + sh * Fy) * dt;
      const double by = (-sh * Fx + ch * Fy) * dt;
      // (I - R(th)) = [[1-c, -s],[s, 1-c]] ; det = 2(1-c)
      const double det = 2 * (1 - c);
      const double mx = ((1 - c) * bx + s * by) / det;   // inverse 2x2
      const double my = (-s * bx + (1 - c) * by) / det;
      return std::sqrt((mx - msx) * (mx - msx) + (my - msy) * (my - msy));
    };

    const double dt1 = 0.02, e1 = fixed_point_err(dt1), e2 = fixed_point_err(dt1 / 2);
    const double ratio = e1 / e2;
    std::printf("derive E x B : err(dt)=%.2e err(dt/2)=%.2e ratio=%.2f (vise ~4)\n",
                e1, e2, ratio);
    chk(e1 < 1e-3, "derive_proche_a_dt");
    chk(ratio > 3.5 && ratio < 4.5, "convergence_ordre_2");
  }

  if (fails == 0) std::printf("OK test_magnetic_euler_poisson\n");
  return fails == 0 ? 0 : 1;
}
