// AMR dans le temps : advance 2-niveaux sous-cycle (Berger-Oliger) avec reflux.
// Un blob est advecte a vitesse constante A TRAVERS l'interface coarse-fine.
// Le sous-cyclage (le fin fait r=2 sous-pas de dt/2) + l'accumulation des flux
// fins dans le registre + le FillPatch interpole en temps rendent le schema
// a la fois CONSERVATIF (masse a l'arrondi) et STABLE (solution bornee, pas
// d'instabilite a l'interface).

#include <adc/integrator/amr_reflux.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/model/diocotron.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int nc = 32;
  Box2D dom = Box2D::from_extents(nc, nc);
  const double dxc = 1.0 / nc, dyc = 1.0 / nc;
  const int CI0 = 8, CI1 = 23, CJ0 = 8, CJ1 = 23;  // region fine (coarse idx)
  Box2D fbox{{2 * CI0, 2 * CJ0}, {2 * CI1 + 1, 2 * CJ1 + 1}};

  Diocotron m;
  m.B0 = 1.0;
  // champ de vitesse E x B variable en espace et divergence-libre :
  // vx = -gy/B0 = 1, vy = gx/B0 = 0.2 sin(2 pi x). aux = (phi, gx, gy).
  constexpr double kPi = 3.14159265358979323846;
  auto fill_aux = [&](Fab2D& aux, double dx, double dy) {
    const Box2D g = aux.grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        aux(i, j, 0) = 0.0;                                  // phi (inutilise)
        aux(i, j, 1) = 0.2 * std::sin(2 * kPi * (i + 0.5) * dx);  // gx
        aux(i, j, 2) = -1.0;                                 // gy
      }
  };

  Fab2D Uc(dom, 1, 1), Uf(fbox, 1, 1);
  Fab2D auxc(dom, 3, 1), auxf(fbox, 3, 1);
  fill_aux(auxc, dxc, dyc);
  fill_aux(auxf, dxc / 2, dyc / 2);
  auto blob = [](double x, double y) {
    const double r2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
    return 1.0 + 0.5 * std::exp(-r2 / 0.02);
  };
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      Uc(i, j) = blob((i + 0.5) * dxc, (j + 0.5) * dyc);
  const double dxf = dxc / 2, dyf = dyc / 2;
  for (int j = fbox.lo[1]; j <= fbox.hi[1]; ++j)
    for (int i = fbox.lo[0]; i <= fbox.hi[0]; ++i)
      Uf(i, j) = blob((i + 0.5) * dxf, (j + 0.5) * dyf);
  average_down_fab(Uf, Uc, CI0, CI1, CJ0, CJ1);  // sync initial

  auto mass = [&]() {
    double M = 0;
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) M += Uc(i, j) * dxc * dyc;
    return M;
  };
  const double M0 = mass();

  const double dt = 0.4 * dxc;  // CFL grossier ; CFL fin = 0.4 (sous-cycle)
  for (int s = 0; s < 60; ++s)
    amr_step_2level(m, Uc, dom, dxc, dyc, Uf, CI0, CI1, CJ0, CJ1, auxc, auxf, dt);

  const double M1 = mass();
  std::printf("masse : M0=%.10f M1=%.10f  drift=%.3e\n", M0, M1,
              std::fabs(M1 - M0));
  chk(std::fabs(M1 - M0) < 1e-12, "mass_conserved_with_reflux");

  // solution bornee et propre (le blob reste dans [1, 1.5], pas d'instabilite)
  double mn = 1e300, mx = -1e300;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      mn = std::min(mn, Uc(i, j));
      mx = std::max(mx, Uc(i, j));
    }
  chk(mn > 0.999 && mx < 1.5, "stable_and_bounded");

  if (fails == 0) std::printf("OK test_amr_reflux\n");
  return fails == 0 ? 0 : 1;
}
