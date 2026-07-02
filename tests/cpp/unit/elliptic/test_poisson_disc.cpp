// Paroi conductrice circulaire (embedded boundary) dans la multigrille.
// Solution manufacturee dans un disque de rayon R centre en (0.5,0.5) :
//   phi = R^2 - r^2  satisfait  lap(phi) = -4  et  phi = 0  sur le cercle.
// On resout lap(phi) = -4 avec phi=0 hors du disque (masque) et on compare a
// l'interieur (loin de la frontiere en escalier, O(dx) la-bas).

#include <gtest/gtest.h>

#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cmath>
#include <cstdio>

using namespace pops;

namespace {

const double kCx = 0.5, kCy = 0.5, kR = 0.4;

void RunDisc(int n, int& cycles, double& err) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
  auto active = [=](Real x, Real y) { return std::hypot(x - kCx, y - kCy) < kR; };
  GeometricMG mg(geom, ba, bc, active);
  mg.rhs().set_val(-4.0);
  mg.phi().set_val(0.0);

  // convergence degradee par la frontiere en escalier (le grossier ne la
  // represente pas) : on vise une reduction 1e-6 du residu, suffisante pour
  // la precision de troncature. En usage couple, le warm start ramene a
  // 1-2 cycles par pas.
  const Real r0 = mg.current_residual();
  Real rn = r0;
  cycles = 0;
  while (rn > 1e-6 * r0 && cycles < 200) {
    mg.vcycle();
    rn = mg.current_residual();
    ++cycles;
  }

  const Fab2D& p = mg.phi().fab(0);
  err = 0;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      const double x = geom.x_cell(i) - kCx, y = geom.y_cell(j) - kCy;
      const double r = std::hypot(x, y);
      if (r < 0.8 * kR)  // interieur, loin de l'escalier
        err = std::max(err, std::fabs(p(i, j, 0) - (kR * kR - r * r)));
    }
}

}  // namespace

TEST(test_poisson_disc, converges_and_matches_manufactured_solution) {
  int c128 = 0, c256 = 0;
  double e128 = 0, e256 = 0;
  RunDisc(128, c128, e128);
  RunDisc(256, c256, e256);
  std::printf("disc : n=128 cycles=%d err=%.3e | n=256 cycles=%d err=%.3e\n", c128, e128, c256,
              e256);

  EXPECT_TRUE(c128 < 200) << "converged_128";
  EXPECT_TRUE(c256 < 200) << "converged_256";
  EXPECT_TRUE(e256 < 5e-3) << "accurate";
  EXPECT_TRUE(e256 < e128) << "converges_with_resolution";
}
