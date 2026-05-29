// Enveloppe de robustesse du schema deux-fluides AP en fonction de l'amplitude
// initiale. Le transport de quantite de mouvement (tfap_mstar) est Rusanov ordre 1
// et la continuite (tfap_div_update) est CENTREE (non upwindee) : adapte au regime
// lineaire AP, mais la continuite centree est dispersive et peut perdre la positivite
// a grande amplitude. Ce test CARTOGRAPHIE ou le schema tient, pour decider si une
// reconstruction MUSCL/limitee est reellement necessaire (et la valider plus tard).
//
// On balaye eps croissant a regime raide (dt*omega_pe = 5, stabilise) et on mesure :
// densite min (positivite), max|charge| (quasi-neutralite), derive de masse, finitude.

#include <adc/elliptic/poisson_fft_solver.hpp>
#include <adc/integrator/two_fluid_ap.hpp>
#include <adc/model/two_fluid_isothermal.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;
using Driver = TwoFluidAP2D<PoissonFFTSolver>;

static double mindens(const Driver& d) {
  double m = 1e300;
  const Fab2D& f = d.e.fab(0);
  for (int j = d.dom.lo[1]; j <= d.dom.hi[1]; ++j)
    for (int i = d.dom.lo[0]; i <= d.dom.hi[0]; ++i) m = std::fmin(m, f(i, j, 0));
  return m;
}
static double maxdev(const Driver& d) {
  double m = 0;
  const Fab2D& f = d.e.fab(0);
  for (int j = d.dom.lo[1]; j <= d.dom.hi[1]; ++j)
    for (int i = d.dom.lo[0]; i <= d.dom.hi[0]; ++i)
      m = std::fmax(m, std::fabs(f(i, j, 0) - 1.0));
  return m;
}
static double maxcharge(const Driver& d) {
  double m = 0;
  const Fab2D& fe = d.e.fab(0); const Fab2D& fi = d.ion.fab(0);
  for (int j = d.dom.lo[1]; j <= d.dom.hi[1]; ++j)
    for (int i = d.dom.lo[0]; i <= d.dom.hi[0]; ++i)
      m = std::fmax(m, std::fabs(fi(i, j, 0) - fe(i, j, 0)));
  return m;
}
static bool allfinite(const Driver& d) {
  const Fab2D& f = d.e.fab(0);
  for (int j = d.dom.lo[1]; j <= d.dom.hi[1]; ++j)
    for (int i = d.dom.lo[0]; i <= d.dom.hi[0]; ++i)
      if (!std::isfinite(f(i, j, 0))) return false;
  return true;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const double eps_list[] = {1e-3, 1e-2, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8};
  const double dt = 5.0 / 1e3;
  std::printf("enveloppe AP 2D (n=64, dt*omega_pe=5, 300 pas, stabilise) :\n");
  std::printf("   eps    min(n_e)   max|dne|  max|charge|   d masse_e   etat\n");
  double last_positive_eps = 0;
  for (double eps : eps_list) {
    Driver d(64, 2 * kPi, 1.0, 0.04, 1e3, 20.0);
    d.init(eps);
    const double m0 = sum(d.e, 0);
    for (int t = 0; t < 300; ++t) d.step(dt, true);
    const bool fin = allfinite(d);
    const double mn = fin ? mindens(d) : -1.0;
    const double dev = fin ? maxdev(d) : INFINITY;
    const double chg = fin ? maxcharge(d) : INFINITY;
    const double dm = std::fabs(sum(d.e, 0) - m0);
    const char* etat = !fin ? "NON-FINI" : (mn <= 0 ? "n<=0" : "ok");
    std::printf("  %5.3f  %9.3e  %9.3e  %10.3e  %9.2e   %s\n", eps, mn, dev, chg, dm, etat);
    if (fin && mn > 0) last_positive_eps = eps;
    // garde de regression : tant que la solution reste finie, la masse est conservee
    // (le schema est conservatif par construction : div centree + flux Rusanov).
    if (fin) chk(dm < 1e-7, "masse_conservee_si_fini");
  }
  std::printf("amplitude max gardant n_e > 0 : eps = %.3f\n", last_positive_eps);
  // garde minimale : le regime lineaire/faiblement non-lineaire reste positif et borne.
  chk(last_positive_eps >= 0.1, "positif_jusqua_eps_0.1");

  // --- scenario 2 : front raide en transport (couplage faible ~= Euler isotherme) ---
  // Une bosse gaussienne ETROITE quasi-neutre (e == ion -> E ~= 0) lance des ondes
  // acoustiques a fronts raides. La continuite CENTREE est dispersive : on mesure le
  // sous-depassement (min(n_e) sous le fond) revelateur d'oscillations de Gibbs. C'est
  // ce regime, pas la perturbation lisse ci-dessus, qui motiverait une reconstruction
  // MUSCL/limitee. Le schema reste conservatif et borne ; seul l'overshoot est en jeu.
  {
    Driver d(96, 2 * kPi, 1.0, 1.0, 2.0, 0.4);  // couplage faible
    d.e.set_val(0); d.ion.set_val(0);
    Array4 ae = d.e.fab(0).array(), ai = d.ion.fab(0).array();
    const double xc = kPi, yc = kPi, w = 6.0 * (2 * kPi / 96);  // demi-largeur ~6 mailles
    for (int j = d.dom.lo[1]; j <= d.dom.hi[1]; ++j)
      for (int i = d.dom.lo[0]; i <= d.dom.hi[0]; ++i) {
        const double x = d.geom.x_cell(i), y = d.geom.y_cell(j);
        const double r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);
        const double bump = 1.0 + 1.0 * std::exp(-r2 / (w * w));  // pic n = 2
        ae(i, j, 0) = bump; ai(i, j, 0) = bump;
      }
    const double m0 = sum(d.e, 0);
    const double dt = 0.3 * (2 * kPi / 96) / std::sqrt(1.0);  // CFL acoustique
    double minover = 1e300;
    for (int t = 0; t < 200; ++t) {
      d.step(dt, true);
      minover = std::fmin(minover, mindens(d));
    }
    const double dm = std::fabs(sum(d.e, 0) - m0);
    const bool fin = allfinite(d);
    std::printf("front raide (bosse etroite, couplage faible) : min(n_e) sur le run=%.4f "
                "(fond=1.0, sous-depassement=%.2e) d masse=%.2e %s\n",
                minover, std::fabs(1.0 - minover) > 0 ? (1.0 - minover) : 0.0, dm,
                fin ? "fini" : "NON-FINI");
    chk(fin, "front_raide_fini");
    chk(dm < 1e-7, "front_raide_masse_conservee");
  }

  if (fails == 0) std::printf("OK test_two_fluid_ap_amplitude\n");
  return fails == 0 ? 0 : 1;
}
