// Validation NUMERIQUE (pas seulement bit-identique) du Laplacien 5 points : ordre de
// convergence quantitatif sur solutions manufacturees. On raffine n = 32 -> 64 -> 128 et
// on mesure les erreurs L2 ET Linf vs la solution exacte ; l'ordre observe
// log(e_n / e_2n)/log 2 doit tendre vers 2 (precision O(dx^2) du stencil). Couvre aussi le
// NULLSPACE periodique : le second membre a moyenne nulle (solvabilite), la solution est
// fixee a moyenne nulle (jauge) puis comparee.

#include <gtest/gtest.h>

#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cmath>
#include <cstdio>

using namespace pops;
static constexpr double kPi = 3.14159265358979323846;

namespace {

// Resout lap(phi) = f, renvoie erreurs L2 et Linf (jauge moyenne-nulle si periodique) +
// la moyenne du second membre (controle de solvabilite du nullspace).
template <class PhiEx, class RhsF>
void Solve(int n, const BCRec& bc, bool periodic, PhiEx phi_ex, RhsF rhs_f, double& eL2,
           double& eInf, double& rhs_mean) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);

  GeometricMG mg(geom, ba, bc);
  Array4 af = mg.rhs().fab(0).array();
  for_each_cell(dom, [af, geom, rhs_f](int i, int j) {
    af(i, j, 0) = rhs_f(geom.x_cell(i), geom.y_cell(j));
  });
  rhs_mean = sum(mg.rhs()) / static_cast<double>(dom.num_cells());
  mg.phi().set_val(0.0);

  const Real r0 = mg.current_residual();
  Real rn = r0;
  for (int c = 0; c < 60 && rn > 1e-11 * r0; ++c) {
    mg.vcycle();
    rn = mg.current_residual();
  }

  Fab2D& p = mg.phi().fab(0);
  if (periodic) {  // solution definie a une constante pres : jauge moyenne nulle
    const Real mean = sum(mg.phi()) / static_cast<Real>(dom.num_cells());
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        p(i, j, 0) -= mean;
  }
  double s2 = 0;
  eInf = 0;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      const double e = p(i, j, 0) - phi_ex(geom.x_cell(i), geom.y_cell(j));
      s2 += e * e;
      eInf = std::max(eInf, std::fabs(e));
    }
  eL2 = std::sqrt(s2 / static_cast<double>(dom.num_cells()));
}

double Order(double ec, double ef) {
  return std::log(ec / ef) / std::log(2.0);
}

}  // namespace

TEST(test_poisson_convergence, dirichlet_order2) {
  // phi = sin(pi x) sin(pi y), lap phi = -2 pi^2 phi
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
  auto pe = [](double x, double y) { return std::sin(kPi * x) * std::sin(kPi * y); };
  auto fr = [&](double x, double y) { return -2 * kPi * kPi * pe(x, y); };
  double l2_32, li_32, l2_64, li_64, l2_128, li_128, mm;
  Solve(32, bc, false, pe, fr, l2_32, li_32, mm);
  Solve(64, bc, false, pe, fr, l2_64, li_64, mm);
  Solve(128, bc, false, pe, fr, l2_128, li_128, mm);
  const double oL2 = Order(l2_64, l2_128), oInf = Order(li_64, li_128);
  std::printf("Dirichlet : L2 ordre %.2f (%.2e->%.2e) | Linf ordre %.2f (%.2e->%.2e)\n", oL2, l2_64,
              l2_128, oInf, li_64, li_128);
  EXPECT_TRUE(oL2 > 1.85 && oL2 < 2.15) << "dirichlet_L2_ordre2";
  EXPECT_TRUE(oInf > 1.85 && oInf < 2.15) << "dirichlet_Linf_ordre2";
}

TEST(test_poisson_convergence, periodic_order2_and_nullspace) {
  // periodique : phi = sin(2 pi x) sin(2 pi y), lap phi = -8 pi^2 phi (nullspace)
  BCRec bc;  // periodique par defaut
  auto pe = [](double x, double y) { return std::sin(2 * kPi * x) * std::sin(2 * kPi * y); };
  auto fr = [&](double x, double y) { return -8 * kPi * kPi * pe(x, y); };
  double l2_32, li_32, l2_64, li_64, l2_128, li_128, mm32, mm64, mm128;
  Solve(32, bc, true, pe, fr, l2_32, li_32, mm32);
  Solve(64, bc, true, pe, fr, l2_64, li_64, mm64);
  Solve(128, bc, true, pe, fr, l2_128, li_128, mm128);
  const double oL2 = Order(l2_64, l2_128), oInf = Order(li_64, li_128);
  std::printf("Periodique : L2 ordre %.2f (%.2e->%.2e) | Linf ordre %.2f | <f>=%.1e\n", oL2, l2_64,
              l2_128, oInf, mm128);
  EXPECT_TRUE(oL2 > 1.85 && oL2 < 2.15) << "periodique_L2_ordre2";
  EXPECT_TRUE(oInf > 1.85 && oInf < 2.15) << "periodique_Linf_ordre2";
  // nullspace : second membre a moyenne nulle (solvabilite periodique)
  EXPECT_TRUE(std::fabs(mm128) < 1e-10) << "nullspace_rhs_moyenne_nulle";
}
