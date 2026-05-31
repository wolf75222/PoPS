#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>

// Invariants PHYSIQUES du transport diocotron (derive E x B, div v = 0, equivalent a l'Euler
// incompressible 2D en formulation vorticite). Le champ rho est SIMPLEMENT advecte par un flot
// a divergence nulle, donc le systeme continu ideal conserve :
//   - la masse              M   = int rho dA                      (exacte au schema flux) ;
//   - tout Casimir          int f(rho) dA, en particulier l'ENSTROPHIE Z = int rho^2 dA
//     (un schema upwind/limite la DISSIPE : sa decroissance MESURE la diffusion numerique) ;
//   - le principe du maximum : rho reste dans [min0, max0] (pas de creation de valeurs) ;
//   - le moment angulaire   P   = int rho r^2 dA  (invariant diocotron en cavite, Davidson) ;
//   - l'energie de champ    W   = 1/2 int |grad phi|^2 dA  (conservee par le systeme ideal ;
//     l'instabilite REDISTRIBUE l'energie entre moyen et perturbation, le total reste borne).
// Ces quantites sont de VRAIS indicateurs de fidelite physique, au-dela du taux de croissance :
// une masse derivante, une enstrophie qui CROIT, un max principle viole ou un moment angulaire
// qui derive signalent un schema non conservatif ou instable. Host-only (hors hot path).

namespace adc {

struct DiocotronInvariants {
  double mass = 0;       // int rho dA
  double energy = 0;     // 1/2 int |grad phi|^2 dA
  double enstrophy = 0;  // int rho^2 dA  (Casimir ; decroit avec la diffusion numerique)
  double angmom = 0;     // int rho ((x-cx)^2 + (y-cy)^2) dA
  double rho_min = 0, rho_max = 0;
};

// Calcule les invariants sur les cellules valides (mono-box ; rho comp 0, phi avec >=1 ghost
// pour le gradient centre). cx, cy = centre de la cavite pour le moment angulaire.
inline DiocotronInvariants diocotron_invariants(const MultiFab& rho, const MultiFab& phi,
                                                const Geometry& geom, double cx, double cy) {
  const ConstArray4 r = rho.fab(0).const_array();
  const ConstArray4 p = phi.fab(0).const_array();
  const Box2D v = rho.box(0);
  const double dx = geom.dx(), dy = geom.dy(), dA = dx * dy;
  DiocotronInvariants q;
  q.rho_min = 1e300;
  q.rho_max = -1e300;
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
      const double rho_ij = r(i, j, 0);
      q.mass += rho_ij * dA;
      q.enstrophy += rho_ij * rho_ij * dA;
      const double x = geom.x_cell(i) - cx, y = geom.y_cell(j) - cy;
      q.angmom += rho_ij * (x * x + y * y) * dA;
      q.rho_min = std::min(q.rho_min, rho_ij);
      q.rho_max = std::max(q.rho_max, rho_ij);
      const double ex = (p(i + 1, j) - p(i - 1, j)) / (2 * dx);
      const double ey = (p(i, j + 1) - p(i, j - 1)) / (2 * dy);
      q.energy += 0.5 * (ex * ex + ey * ey) * dA;
    }
  return q;
}

}  // namespace adc
