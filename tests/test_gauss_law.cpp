// Validation NUMERIQUE du COUPLAGE elliptique-hyperbolique (le coeur de la lib) : la loi
// de Gauss discrete. Le coupleur resout lap(phi) = f = elliptic_rhs(U) puis donne au modele
// aux = grad(phi) (differences centrees). On verifie que la divergence discrete de ce champ
// redonne le second membre : div(grad phi) -> f. Comme grad et div sont centres, le stencil
// est large mais reste O(dx^2) : l'erreur L2 decroit a l'ordre 2 en raffinant. C'est la
// coherence du chemin elliptique -> hyperbolique (aux = grad phi nourrit la source), pas
// seulement la justesse du solveur de Poisson seul.

#include <adc/coupling/coupler.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/euler_poisson.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

// Resout le champ pour rho = rho0 + eps cos(2 pi x) cos(2 pi y), renvoie l'erreur L2 de
// div(grad phi) vs f = elliptic_rhs (= four_pi_G (rho - rho0) ici, signe +).
static double gauss_error(int n) {
  const double L = 1.0, rho0 = 1.0, p0 = 1.0, gamma = 5.0 / 3.0, eps = 0.1;
  const double k = 2 * kPi / L;

  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, L, 0.0, L};
  const double dx = geom.dx(), dy = geom.dy();
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);

  EulerPoisson model;
  model.hydro.gamma = gamma;
  model.four_pi_G = 1.0;
  model.rho0 = rho0;  // coupling_sign = +1 par defaut

  MultiFab U(ba, dm, 4, 2);
  {
    Fab2D& f = U.fab(0);
    const Box2D g = f.grown_box();
    auto w = [&](int x) { return (x % n + n) % n; };
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        const double x = (w(i) + 0.5) / n * L, y = (w(j) + 0.5) / n * L;
        const double rho = rho0 + eps * std::cos(k * x) * std::cos(k * y);
        f(i, j, 0) = rho;
        f(i, j, 1) = 0.0;
        f(i, j, 2) = 0.0;
        f(i, j, 3) = p0 / (gamma - 1);
      }
  }

  BCRec bcU, bcPhi;  // periodique
  Coupler<EulerPoisson> cpl(model, geom, ba, bcU, bcPhi);
  cpl.solve_fields(U);  // resout Poisson + aux = grad phi

  const ConstArray4 a = cpl.aux().fab(0).const_array();  // (phi, grad_x, grad_y)
  auto w = [&](int x) { return (x % n + n) % n; };
  double s2 = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      // divergence centree du champ grad phi (acces periodique aux cellules valides)
      const double div = (a(w(i + 1), j, 1) - a(w(i - 1), j, 1)) / (2 * dx) +
                         (a(i, w(j + 1), 2) - a(i, w(j - 1), 2)) / (2 * dy);
      const double x = (i + 0.5) / n * L, y = (j + 0.5) / n * L;
      const double f = model.four_pi_G * (eps * std::cos(k * x) * std::cos(k * y));
      const double e = div - f;
      s2 += e * e;
    }
  return std::sqrt(s2 / (static_cast<double>(n) * n));
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const double e32 = gauss_error(32), e64 = gauss_error(64), e128 = gauss_error(128);
  const double ord = std::log(e64 / e128) / std::log(2.0);
  std::printf("loi de Gauss div(grad phi) vs f : e32=%.3e e64=%.3e e128=%.3e ordre=%.2f\n",
              e32, e64, e128, ord);

  chk(ord > 1.8 && ord < 2.2, "gauss_ordre2");      // div(E) = rho au 2e ordre
  chk(e128 < 5e-2, "gauss_precis");                  // identite discrete a O(dx^2)
  chk(e128 < e64 && e64 < e32, "gauss_converge");

  if (fails == 0) std::printf("OK test_gauss_law\n");
  return fails == 0 ? 0 : 1;
}
