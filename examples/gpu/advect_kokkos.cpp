// Preuve de portabilite Kokkos : le meme pas d'advection E x B (Rusanov) execute
// via Kokkos::parallel_for (MDRangePolicy 2D), en reutilisant TELLE QUELLE la vue
// Array4 de la couche maillage DANS le kernel device. C'est le vrai pendant du
// seam for_each_cell : un parallel_for Kokkos peut le porter sans changer ni le
// layout (SoA composante-lente) ni l'indexation a(i, j, c) (annotee ADC_HD).
// On valide le resultat (espace d'execution = Cuda sur GH200) contre le CPU.
//
// Build : via examples/gpu/CMakeLists.txt (find_package(Kokkos)), avec
// CMAKE_CXX_COMPILER = <kokkos>/bin/nvcc_wrapper. Cf. README section GPU.

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>  // le seam : backend Kokkos sous ADC_HAS_KOKKOS

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

ADC_HD inline double rus(double uL, double uR, double v) {
  const double a = v < 0 ? -v : v;
  return 0.5 * v * (uL + uR) - 0.5 * a * (uR - uL);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const int N = 256, ng = 1;
    const double dx = 1.0 / N, dy = 1.0 / N, dt = 0.3 * dx, vx = 1.0, vy = 0.3;
    Box2D dom = Box2D::from_extents(N, N);

    // champ initial + ghosts periodiques + reference CPU (sur l'hote).
    Fab2D U(dom, 1, ng);
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
    std::vector<double> ref(static_cast<std::size_t>(N) * N);
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < N; ++i) {
        const double fxL = rus(U(i - 1, j), U(i, j), vx);
        const double fxR = rus(U(i, j), U(i + 1, j), vx);
        const double fyB = rus(U(i, j - 1), U(i, j), vy);
        const double fyT = rus(U(i, j), U(i, j + 1), vy);
        ref[j * N + i] = U(i, j) - dt * ((fxR - fxL) / dx + (fyT - fyB) / dy);
      }

    // buffers Kokkos (taille = layout Fab2D). View device + mirror hote.
    const ConstArray4 hv = U.const_array();
    const long sz = U.size();
    Kokkos::View<Real*> d_u("u", sz), d_un("un", sz);
    auto h_u = Kokkos::create_mirror_view(d_u);
    for (long t = 0; t < sz; ++t) h_u(t) = U.data()[t];
    Kokkos::deep_copy(d_u, h_u);
    Kokkos::deep_copy(d_un, 0.0);

    // vues Array4 sur la memoire Kokkos device : memes strides/offsets.
    ConstArray4 u_dev{d_u.data(), hv.nx_tot, hv.comp_stride, hv.ig0, hv.jg0};
    Array4 un_dev{d_un.data(), hv.nx_tot, hv.comp_stride, hv.ig0, hv.jg0};

    // Le MEME for_each_cell que le code CPU. Sous ADC_HAS_KOKKOS il dispatche
    // vers Kokkos::parallel_for (espace Cuda) ; le fonctor est device-callable
    // (ADC_HD) et opere sur la donnee device-residente (vues sur d_u / d_un).
    for_each_cell(Box2D{{0, 0}, {N - 1, N - 1}}, [=] ADC_HD(int i, int j) {
      const double fxL = rus(u_dev(i - 1, j), u_dev(i, j), vx);
      const double fxR = rus(u_dev(i, j), u_dev(i + 1, j), vx);
      const double fyB = rus(u_dev(i, j - 1), u_dev(i, j), vy);
      const double fyT = rus(u_dev(i, j), u_dev(i, j + 1), vy);
      un_dev(i, j) = u_dev(i, j) - dt * ((fxR - fxL) / dx + (fyT - fyB) / dy);
    });
    Kokkos::fence();

    auto h_un = Kokkos::create_mirror_view(d_un);
    Kokkos::deep_copy(h_un, d_un);
    Array4 hview{h_un.data(), hv.nx_tot, hv.comp_stride, hv.ig0, hv.jg0};
    double maxdiff = 0;
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < N; ++i)
        maxdiff = std::fmax(maxdiff, std::fabs(hview(i, j) - ref[j * N + i]));

    std::printf("Kokkos exec=%s  N=%d  maxdiff(Kokkos vs CPU)=%.3e\n",
                Kokkos::DefaultExecutionSpace::name(), N, maxdiff);
    if (maxdiff < 1e-12)
      std::printf("OK advect_kokkos\n");
    else {
      std::printf("FAIL advect_kokkos\n");
      rc = 1;
    }
  }
  Kokkos::finalize();
  return rc;
}
