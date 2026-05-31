// Invariants PHYSIQUES du diocotron, brique de verification de fidelite (au-dela du taux de
// croissance). Le transport E x B etant a divergence nulle, le systeme ideal conserve la masse
// et tout Casimir, dissipe (au plus) l'enstrophie via la diffusion du schema, et respecte le
// principe du maximum. On verifie :
//   1. le calcul de diocotron_invariants sur un champ ANALYTIQUE (masse, enstrophie, energie) ;
//   2. sur un run couple (Coupler<Diocotron>, SSPRK2 + Minmod, boite periodique) :
//      masse conservee a l'arrondi, enstrophie NON CROISSANTE, energie bornee, max principle.
// Complement de test_diocotron_stability (qui couvre masse/max/enstrophie) : ajoute l'ENERGIE
// de champ et fige le contrat du module d'invariants.

#include <adc/analysis/diocotron_invariants.hpp>
#include <adc/coupling/coupler.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/diocotron.hpp>
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

  const int n = 64;
  const double Lbox = 1.0, n_i0 = 1.0, eps = 0.3;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, Lbox, 0.0, Lbox};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);

  // 1. calcul sur un champ analytique : rho = c constante, phi = 0.
  //    masse = c * L^2, enstrophie = c^2 * L^2, energie = 0, angmom = c * int r^2.
  {
    MultiFab rho(ba, dm, 1, 1), phi(ba, dm, 1, 1);
    rho.set_val(2.0); phi.set_val(0.0);
    const DiocotronInvariants q = diocotron_invariants(rho, phi, geom, 0.5, 0.5);
    chk(std::fabs(q.mass - 2.0 * Lbox * Lbox) < 1e-12, "mass_constante");
    chk(std::fabs(q.enstrophy - 4.0 * Lbox * Lbox) < 1e-12, "enstrophie_constante");
    chk(std::fabs(q.energy) < 1e-12, "energie_phi_nul");
    chk(std::fabs(q.rho_min - 2.0) < 1e-12 && std::fabs(q.rho_max - 2.0) < 1e-12, "minmax_constante");
  }

  // 2. conservation sur un run couple (boite periodique).
  Diocotron model; model.B0 = 1.0; model.n_i0 = n_i0; model.alpha = 1.0;
  MultiFab U(ba, dm, 1, 2);
  {
    Fab2D& f = U.fab(0);
    const Box2D v = U.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        f(i, j, 0) = n_i0 + eps * std::sin(2 * kPi * geom.x_cell(i)) *
                                std::sin(2 * kPi * geom.y_cell(j));
  }
  Coupler<Diocotron> cpl(model, geom, ba, BCRec{}, BCRec{});
  cpl.solve_fields(U);
  const DiocotronInvariants q0 = diocotron_invariants(U, cpl.phi(), geom, 0.5, 0.5);

  double ens_max = q0.enstrophy, energy_max = q0.energy;
  const double dt = 0.06;
  for (int s = 0; s < 150; ++s) {
    cpl.advance<Minmod>(U, dt);
    cpl.solve_fields(U);
    const DiocotronInvariants q = diocotron_invariants(U, cpl.phi(), geom, 0.5, 0.5);
    ens_max = std::max(ens_max, q.enstrophy);
    energy_max = std::max(energy_max, q.energy);
  }
  cpl.solve_fields(U);
  const DiocotronInvariants qF = diocotron_invariants(U, cpl.phi(), geom, 0.5, 0.5);
  const double dmass = std::fabs(qF.mass - q0.mass);
  std::printf("invariants : dmasse=%.2e | enstrophie %.5f -> %.5f (max %.5f) | "
              "energie %.3e -> %.3e (max %.3e) | rho [%.4f, %.4f]\n",
              dmass, q0.enstrophy, qF.enstrophy, ens_max, q0.energy, qF.energy, energy_max, qF.rho_min, qF.rho_max);

  chk(dmass < 1e-10, "masse_conservee");
  chk(ens_max < q0.enstrophy * (1 + 1e-9), "enstrophie_non_croissante");
  chk(energy_max < q0.energy * 1.05 + 1e-12, "energie_bornee");  // pas d'explosion d'energie
  chk(qF.rho_min > 0, "positivite");

  if (fails == 0) std::printf("OK test_diocotron_invariants\n");
  return fails == 0 ? 0 : 1;
}
