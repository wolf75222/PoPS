// Test de caracterisation du AmrCoupler (le coupleur AMR de production, ce qui
// tourne dans examples/diocotron_amr.cpp) : jusqu'ici AUCUN test ne le couvrait,
// seules ses briques (amr_step_multilevel, average_down_fab) l'etaient. Ce test
// fige son comportement AVANT la refonte AMR a venir : conservation de la masse sur
// la hierarchie 2 niveaux (le reflux coarse-fine doit conserver a l'arrondi),
// finitude, et couplage Poisson effectif (derive E x B non nulle).
//
// Hierarchie : niveau 0 = domaine periodique nc x nc, niveau 1 = box fine (ratio 2)
// au centre. CI lisse sin*sin (moyenne 1 -> n_i0 = 1, RHS de Poisson a moyenne nulle).

#include <adc/coupling/amr_coupler.hpp>
#include <adc/integrator/amr_multilevel.hpp>  // AmrLevel
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/diocotron.hpp>

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

  const int nc = 32;
  Box2D dom = Box2D::from_extents(nc, nc);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const double dxc = geom.dx(), dyc = geom.dy(), dxf = dxc / 2, dyf = dyc / 2;
  BoxArray ba(std::vector<Box2D>{dom});

  Diocotron model;
  model.B0 = 1.0;
  model.alpha = 1.0;
  model.n_i0 = 1.0;  // CI sin*sin de moyenne 1
  auto ne0 = [&](double x, double y) {
    return 1.0 + 0.3 * std::sin(2 * kPi * x) * std::sin(2 * kPi * y);
  };

  // region raffinee : quart central du niveau grossier
  const int CI0 = nc / 4, CI1 = 3 * nc / 4 - 1, CJ0 = nc / 4, CJ1 = 3 * nc / 4 - 1;
  Box2D fbox{{2 * CI0, 2 * CJ0}, {2 * CI1 + 1, 2 * CJ1 + 1}};

  Fab2D Uc(dom, 1, 1), Uf(fbox, 1, 1);
  const Box2D gc = Uc.grown_box();
  for (int j = gc.lo[1]; j <= gc.hi[1]; ++j)
    for (int i = gc.lo[0]; i <= gc.hi[0]; ++i)
      Uc(i, j) = ne0((i + 0.5) * dxc, (j + 0.5) * dyc);
  for (int j = fbox.lo[1]; j <= fbox.hi[1]; ++j)
    for (int i = fbox.lo[0]; i <= fbox.hi[0]; ++i)
      Uf(i, j) = ne0((i + 0.5) * dxf, (j + 0.5) * dyf);
  average_down_fab(Uf, Uc, CI0, CI1, CJ0, CJ1);

  std::vector<AmrLevel> L0;
  L0.push_back({std::move(Uc), nullptr, dxc, dyc, CI0, CI1, CJ0, CJ1, true});
  L0.push_back({std::move(Uf), nullptr, dxf, dyf, 0, 0, 0, 0, false});

  BCRec bc;  // periodique
  AmrCoupler<Diocotron> sim(model, geom, ba, bc, std::move(L0));

  sim.update();  // resout les champs pour le premier max_drift_speed
  const double v0 = sim.max_drift_speed();
  const double m0 = sim.mass();
  const double dt = 0.4 * dxc / v0;

  bool finite = true;
  for (int s = 0; s < 20; ++s) {
    sim.step(dt);
    if (!std::isfinite(sim.mass())) finite = false;
  }
  const double m1 = sim.mass();
  std::printf("AmrCoupler 2 niveaux : v_derive=%.3e masse0=%.10e masse=%.10e "
              "dmasse=%.2e\n", v0, m0, m1, std::fabs(m1 - m0));

  chk(v0 > 1e-6, "couplage_poisson_effectif");   // la derive E x B est non nulle
  chk(finite && std::isfinite(m1), "stable_fini");
  chk(std::fabs(m1 - m0) < 1e-9, "masse_conservee_hierarchie");  // reflux conservatif

  if (fails == 0) std::printf("OK test_amr_coupler\n");
  return fails == 0 ? 0 : 1;
}
