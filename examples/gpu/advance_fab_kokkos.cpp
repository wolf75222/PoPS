// Etape 3 du portage GPU : le VRAI operateur de transport advance_fab_1c
// (flux de Rusanov via le modele Diocotron, integrateur Euler explicite) tourne
// ENTIEREMENT sur GPU, sans une ligne de code GPU dans l'operateur. Tout passe
// par le seam for_each_cell (backend Kokkos -> Cuda) + un Fab2D en memoire
// unifiee. La chaine flux (StateVec, load_aux, rusanov_flux, Diocotron) est
// annotee ADC_HD, donc device-callable. On valide contre une reference CPU.
//
// C'est la preuve que les operateurs existants tournent sur GPU sans reecriture :
// seuls le backend (for_each_cell) et l'allocateur (Fab2D unifie) changent.

#include <adc/integrator/amr_reflux.hpp>  // advance_fab_1c, xface_box, yface_box
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/model/diocotron.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

static inline double rus(double uL, double uR, double v) {
  const double a = v < 0 ? -v : v;
  return 0.5 * v * (uL + uR) - 0.5 * a * (uR - uL);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const int N = 256, ng = 1;
    const double dx = 1.0 / N, dy = 1.0 / N, dt = 0.3 * dx;
    const double vx = 1.0, vy = 0.3;  // derive E x B constante
    Box2D dom = Box2D::from_extents(N, N);

    Diocotron model;
    model.B0 = 1.0;
    // Fab2D en memoire unifiee (vrai stockage de la lib).
    Fab2D U(dom, 1, ng), aux(dom, 3, ng);
    Fab2D fx(xface_box(dom), 1, 0), fy(yface_box(dom), 1, 0);

    // aux constant : gx=0.3 -> vy, gy=-1 -> vx. Rempli sur la box etendue.
    {
      const Box2D g = aux.grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
          aux(i, j, 0) = 0.0;
          aux(i, j, 1) = 0.3;
          aux(i, j, 2) = -1.0;
        }
    }
    auto u0 = [&](int i, int j) {
      constexpr double pi = 3.14159265358979323846;
      return 1.0 + 0.5 * std::sin(2 * pi * (i + 0.5) / N) *
                       std::sin(2 * pi * (j + 0.5) / N);
    };
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < N; ++i) U(i, j) = u0(i, j);
    auto wrap = [&](int x) { return (x % N + N) % N; };
    for (int j = -ng; j < N + ng; ++j)
      for (int i = -ng; i < N + ng; ++i)
        if (i < 0 || i >= N || j < 0 || j >= N) U(i, j) = u0(wrap(i), wrap(j));

    // reference CPU (depuis l'etat initial, avant que le GPU n'ecrase U).
    std::vector<double> ref(static_cast<std::size_t>(N) * N);
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < N; ++i) {
        const double fxL = rus(U(i - 1, j), U(i, j), vx);
        const double fxR = rus(U(i, j), U(i + 1, j), vx);
        const double fyB = rus(U(i, j - 1), U(i, j), vy);
        const double fyT = rus(U(i, j), U(i, j + 1), vy);
        ref[j * N + i] = U(i, j) - dt * ((fxR - fxL) / dx + (fyT - fyB) / dy);
      }

    // LE pas de transport de la lib, execute sur GPU (for_each_cell -> Cuda).
    advance_fab_1c(model, U, aux, dx, dy, dt, fx, fy);
    Kokkos::fence();

    double maxdiff = 0;
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < N; ++i)
        maxdiff = std::fmax(maxdiff, std::fabs(U(i, j) - ref[j * N + i]));

    std::printf("exec=%s  N=%d  advance_fab_1c sur GPU  maxdiff(GPU vs CPU)=%.3e\n",
                Kokkos::DefaultExecutionSpace::name(), N, maxdiff);
    if (maxdiff < 1e-12)
      std::printf("OK advance_fab_kokkos\n");
    else {
      std::printf("FAIL advance_fab_kokkos\n");
      rc = 1;
    }
  }
  Kokkos::finalize();
  return rc;
}
