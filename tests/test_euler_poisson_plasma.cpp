// Validation du couplage REPULSIF (plasma electrostatique) d'Euler-Poisson : meme
// modele que test_euler_poisson, signe de la source elliptique retourne (coupling_sign
// = -1). On verifie que ce seul changement de signe transforme la gravite en
// electrostatique mono-espece.
//
//   A. dispersion de Langmuir / Bohm-Gross : une perturbation acoustique au repos
//      oscille a omega = sqrt(c_s^2 k^2 + omega_p^2), omega_p^2 = 4 pi G rho0. Noter le
//      signe + (contre le - de Jeans) : le plasma est TOUJOURS stable, jamais d'instabilite.
//   B. explosion de Coulomb vs effondrement de Jeans : un meme grumeau de densite au
//      repos, sous pression uniforme, voit son pic CROITRE en gravite (attraction) et
//      DECROITRE en plasma (repulsion). Le signe oppose du couplage suffit.

#include <adc/core/physical_model.hpp>
#include <adc/coupling/coupler.hpp>
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

static_assert(PhysicalModel<EulerPoisson>, "EulerPoisson modele PhysicalModel");

// Construit un modele Euler-Poisson avec le signe de couplage voulu.
static EulerPoisson make_model(double gamma, double four_pi_G, double rho0, double sign) {
  EulerPoisson m;
  m.hydro.gamma = gamma;
  m.four_pi_G = four_pi_G;
  m.rho0 = rho0;
  m.coupling_sign = sign;
  return m;
}

// ---- A. oscillation de Langmuir : omega^2 = c_s^2 k^2 + omega_p^2 (signe +) ----
static int test_langmuir(int& fails) {
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const int N = 64;
  const double L = 1.0, rho0 = 1.0, p0 = 1.0, gamma = 5.0 / 3.0;
  const double four_pi_G = 20.0, eps = 1e-3;
  const double k = 2 * kPi / L;
  const double cs2 = gamma * p0 / rho0;
  const double omega_p2 = four_pi_G * rho0;                  // omega_p^2 = 4 pi G rho0
  const double omega_th = std::sqrt(cs2 * k * k + omega_p2);  // Bohm-Gross : signe +

  Box2D dom = Box2D::from_extents(N, N);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);

  EulerPoisson model = make_model(gamma, four_pi_G, rho0, -1.0);  // REPULSIF
  BCRec bcU, bcPhi;
  Coupler<EulerPoisson> cpl(model, geom, ba, bcU, bcPhi);

  MultiFab U(ba, dm, 4, 2);
  {
    Fab2D& f = U.fab(0);
    const Box2D v = U.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const double drho = eps * rho0 * std::cos(k * geom.x_cell(i));
        const double rho = rho0 + drho, p = p0 + cs2 * drho;
        f(i, j, 0) = rho;
        f(i, j, 1) = 0.0;
        f(i, j, 2) = 0.0;
        f(i, j, 3) = p / (gamma - 1);
      }
  }

  auto mode_amp = [&]() {
    const ConstArray4 u = U.fab(0).const_array();
    const Box2D v = U.box(0);
    double m = 0;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        m += (u(i, j, 0) - rho0) * std::cos(k * geom.x_cell(i));
    return 2.0 * m / (static_cast<double>(N) * N);
  };

  const double T = 0.25, cfl = 0.35;  // T > quart de periode (~0.17, plasma oscille vite)
  const double dt = cfl * (L / N) / (std::sqrt(cs2) + 0.1);

  double t = 0, mprev = mode_amp(), tprev = 0, tzero = -1;
  while (t < T) {
    cpl.advance<Minmod>(U, dt);
    t += dt;
    const double m = mode_amp();
    if (tzero < 0 && m < 0 && mprev > 0)
      tzero = tprev + dt * mprev / (mprev - m);
    mprev = m;
    tprev = t;
  }

  const double omega_meas = (tzero > 0) ? kPi / (2 * tzero) : 0.0;
  const double rel = std::fabs(omega_meas - omega_th) / omega_th;
  std::printf("Langmuir : omega_th=%.4f omega_mesure=%.4f (ecart %.1f%%)\n", omega_th,
              omega_meas, 100 * rel);
  chk(tzero > 0, "oscillation_detectee");
  chk(rel < 0.08, "frequence_de_Bohm_Gross");
  // Le plasma est inconditionnellement stable : omega^2 > 0 toujours, contre Jeans.
  chk(omega_th > std::sqrt(cs2) * k, "plasma_durcit_l_acoustique");
  return fails;
}

// ---- B. explosion de Coulomb (plasma) vs effondrement de Jeans (gravite) ----
// Un grumeau gaussien au repos, pression uniforme : la seule force initiale est le
// champ. En gravite il se contracte (pic croit), en plasma il se disperse (pic decroit).
static double peak_trend(double sign) {
  const int N = 64;
  const double L = 1.0, p0 = 1.0, gamma = 5.0 / 3.0, four_pi_G = 50.0;
  const double rho_bg = 1.0, amp = 0.5, w = 0.08 * L;  // grumeau marque

  Box2D dom = Box2D::from_extents(N, N);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);

  MultiFab U(ba, dm, 4, 2);
  double mean = 0;
  {
    Fab2D& f = U.fab(0);
    const Box2D v = U.box(0);
    const double xc = 0.5 * L, yc = 0.5 * L;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const double dx = geom.x_cell(i) - xc, dy = geom.y_cell(j) - yc;
        const double rho = rho_bg + amp * std::exp(-(dx * dx + dy * dy) / (w * w));
        f(i, j, 0) = rho;
        f(i, j, 1) = 0.0;
        f(i, j, 2) = 0.0;
        f(i, j, 3) = p0 / (gamma - 1);  // pression UNIFORME : grad p = 0 a t=0
        mean += rho;
      }
    mean /= static_cast<double>(N) * N;
  }

  // rho0 = <rho> : second membre periodique a moyenne nulle (solvabilite de Poisson).
  EulerPoisson model = make_model(gamma, four_pi_G, mean, sign);
  BCRec bcU, bcPhi;
  Coupler<EulerPoisson> cpl(model, geom, ba, bcU, bcPhi);

  auto peak = [&]() {
    const ConstArray4 u = U.fab(0).const_array();
    const Box2D v = U.box(0);
    double p = 0;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) p = std::max(p, u(i, j, 0));
    return p;
  };

  const double peak0 = peak();
  const double cs2 = gamma * p0 / rho_bg, dt = 0.3 * (L / N) / (std::sqrt(cs2) + 0.1);
  for (int s = 0; s < 40; ++s) cpl.advance<Minmod>(U, dt);
  return peak() - peak0;  // > 0 : contraction ; < 0 : dispersion
}

static int test_coulomb_vs_jeans(int& fails) {
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };
  const double trend_grav = peak_trend(+1.0);   // gravite : effondrement
  const double trend_plasma = peak_trend(-1.0);  // plasma : explosion
  std::printf("pic du grumeau : gravite %+.3e  plasma %+.3e\n", trend_grav, trend_plasma);
  chk(trend_grav > 0, "gravite_contracte_le_grumeau");
  chk(trend_plasma < 0, "plasma_disperse_le_grumeau");
  chk(trend_grav * trend_plasma < 0, "signes_opposes");
  return fails;
}

int main() {
  int fails = 0;
  test_langmuir(fails);
  test_coulomb_vs_jeans(fails);
  if (fails == 0) std::printf("OK test_euler_poisson_plasma\n");
  return fails == 0 ? 0 : 1;
}
