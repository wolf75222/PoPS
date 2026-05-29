// Etape B (suite) : la multigrille geometrique ENTIERE (V-cycle complet) tourne
// sur GPU. Au dela du residu et du lisseur (poisson_kokkos), on route aussi les
// operateurs de transfert (average_down / interpolate via for_each_cell + ADC_HD,
// coarsen_index ADC_HD) et l'arithmetique de MultiFab (saxpy / lincomb sur
// for_each_cell). Les seuls points hote restants sont fences : remplissage de bord
// (fill_ghosts dans gs/residu), reduction norm_inf et set_val portent un
// device_fence() (= Kokkos::fence, no-op hors Kokkos). Le V-cycle devient donc une
// chaine de kernels device, ordonnancee sur le stream par defaut.
//
// Validation : probleme manufacture Dirichlet phi = sin(pi x) sin(pi y),
// lap(phi) = -2 pi^2 phi. On verifie que GeometricMG::solve converge (residu
// relatif < 1e-9 en peu de cycles) et que la solution est exacte a O(dx^2),
// exactement comme le test CPU test_geometric_mg.

#include <adc/elliptic/geometric_mg.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
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
    BoxArray ba = BoxArray::from_domain(dom, n);

    BCRec bc;
    bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

    GeometricMG mg(geom, ba, bc);  // hierarchie 128 -> 64 -> ... -> 2

    // second membre f = -2 pi^2 sin(pi x) sin(pi y) (remplissage hote, avant kernels)
    {
      Array4 af = mg.rhs().fab(0).array();
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
          af(i, j, 0) =
              -2 * kPi * kPi * phi_exact(geom.x_cell(i), geom.y_cell(j));
    }
    mg.phi().set_val(0.0);

    const double r0 = mg.current_residual();
    const int cycles = mg.solve(1e-9, 50);  // V-cycles complets sur GPU
    const double rN = mg.current_residual();

    // erreur vs solution exacte (lecture hote -> barriere)
    Kokkos::fence();
    double err = 0;
    const Fab2D& p = mg.phi().fab(0);
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        err = std::fmax(err, std::fabs(p(i, j, 0) -
                                       phi_exact(geom.x_cell(i), geom.y_cell(j))));

    std::printf(
        "exec=%s  n=%d  niveaux=%d\n  GeometricMG::solve GPU : cycles=%d "
        "r0=%.3e rN=%.3e (ratio=%.2e)  err_inf=%.3e\n",
        Kokkos::DefaultExecutionSpace::name(), n, mg.num_levels(), cycles, r0,
        rN, rN / r0, err);

    const bool ok_conv = cycles < 50 && rN <= 1e-9 * r0;
    const bool ok_acc = err < 1e-2;  // O(dx^2) a n=128
    if (ok_conv && ok_acc)
      std::printf("OK mg_kokkos\n");
    else {
      std::printf("FAIL mg_kokkos (conv=%d acc=%d)\n", ok_conv, ok_acc);
      rc = 1;
    }
  }
  Kokkos::finalize();
  return rc;
}
