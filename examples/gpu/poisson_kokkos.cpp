// Etape B du solveur complet GPU : les briques de la multigrille geometrique
// (operateur de Poisson 5 points + lisseur Gauss-Seidel red-black) tournent sur
// GPU. apply_laplacian / poisson_residual / gs_color utilisent DEJA for_each_cell ;
// il suffit que leurs lambdas soient device-callable (ADC_HD). Subtilite GPU :
// gs_rb_sweep et poisson_residual appellent fill_ghosts (boucle HOTE sur la
// memoire unifiee) entre deux kernels ; on insere donc un device_fence()
// (Kokkos::fence) avant chaque lecture hote, sinon l'hote lirait du phi encore
// en cours d'ecriture par le kernel precedent.
//
// Validation (probleme manufacture Dirichlet, phi = sin(pi x) sin(pi y)) :
//   1. poisson_residual sur GPU == reference SERIE (memes fonctions, bit a ~1e-13)
//   2. gs_smooth sur GPU fait chuter le residu (lisseur + barrieres OK de bout en bout)

#include <adc/elliptic/poisson_operator.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

static constexpr double kPi = 3.14159265358979323846;
static double phi_exact(double x, double y) {
  return std::sin(kPi * x) * std::sin(kPi * y);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const int n = 128;
    Box2D dom = Box2D::from_extents(n, n);
    Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
    BoxArray ba(std::vector<Box2D>{dom});
    DistributionMapping dm(1, 1);

    MultiFab phi(ba, dm, 1, 1), f(ba, dm, 1, 0);
    MultiFab res(ba, dm, 1, 0), resref(ba, dm, 1, 0);

    // second membre f = -2 pi^2 sin(pi x) sin(pi y) (Dirichlet 0 au bord)
    Array4 af = f.fab(0).array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        af(i, j, 0) =
            -2 * kPi * kPi * phi_exact(geom.x_cell(i), geom.y_cell(j));

    BCRec bc;
    bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

    // ------- Partie 1 : operateur residu sur GPU vs reference serie -------
    // phi non trivial (0.5 * solution exacte) -> residu = f - lap(phi) non nul.
    {
      Array4 ap = phi.fab(0).array();
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
          ap(i, j, 0) = 0.5 * phi_exact(geom.x_cell(i), geom.y_cell(j));
    }
    poisson_residual(phi, f, geom, bc, res);  // for_each_cell -> Cuda
    Kokkos::fence();

    // reference serie : memes formules, iteration hote (phi a deja ses ghosts
    // Dirichlet remplis par le fill_ghosts hote dans poisson_residual).
    const double idx2 = 1.0 / (geom.dx() * geom.dx());
    const double idy2 = 1.0 / (geom.dy() * geom.dy());
    {
      const ConstArray4 p = phi.fab(0).const_array();
      const ConstArray4 ff = f.fab(0).const_array();
      Array4 r = resref.fab(0).array();
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
          const double lap = (p(i + 1, j) - 2 * p(i, j) + p(i - 1, j)) * idx2 +
                             (p(i, j + 1) - 2 * p(i, j) + p(i, j - 1)) * idy2;
          r(i, j) = ff(i, j) - lap;
        }
    }
    double maxdiff = 0;
    const ConstArray4 rg = res.fab(0).const_array();
    const ConstArray4 rr = resref.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        maxdiff = std::fmax(maxdiff, std::fabs(rg(i, j) - rr(i, j)));

    // ------- Partie 2 : lisseur GS red-black sur GPU fait chuter le residu -------
    phi.set_val(0.0);
    poisson_residual(phi, f, geom, bc, res);
    Kokkos::fence();
    auto norm_inf_res = [&]() {
      Kokkos::fence();  // res ecrit par un kernel : barriere avant lecture hote
      const ConstArray4 r = res.fab(0).const_array();
      double m = 0;
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
          m = std::fmax(m, std::fabs(r(i, j)));
      return m;
    };
    const double r0 = norm_inf_res();

    const int nsweeps = 4000;
    gs_smooth(phi, f, geom, bc, nsweeps);  // chaque balayage : 2 kernels + 2 fences
    poisson_residual(phi, f, geom, bc, res);
    const double rN = norm_inf_res();

    std::printf(
        "exec=%s  n=%d\n  P1 poisson_residual GPU vs serie  maxdiff=%.3e\n"
        "  P2 gs_smooth(%d) GPU  r0=%.3e rN=%.3e ratio=%.3e\n",
        Kokkos::DefaultExecutionSpace::name(), n, maxdiff, nsweeps, r0, rN,
        rN / r0);

    const bool ok_op = maxdiff < 1e-11;
    const bool ok_smooth = rN < r0 * 0.2;  // le lisseur amortit fortement le residu
    if (ok_op && ok_smooth)
      std::printf("OK poisson_kokkos\n");
    else {
      std::printf("FAIL poisson_kokkos (op=%d smooth=%d)\n", ok_op, ok_smooth);
      rc = 1;
    }
  }
  Kokkos::finalize();
  return rc;
}
