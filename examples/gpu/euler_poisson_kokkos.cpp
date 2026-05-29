// Euler-Poisson auto-gravitant (4 variables) : un pas couple entier sur GPU.
// Generalise coupled_kokkos (diocotron, 1 variable) au modele EulerPoisson : meme
// Coupler<Model>, meme for_each_cell, meme multigrille ; seul le modele change. On
// verifie que le pas couple a 4 variables donne des sommes de controle BIT A BIT
// identiques entre CPU (serie) et GH200 (Cuda). Meme source compilee dans les deux.

#include <adc/coupling/coupler.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/euler_poisson.hpp>
#include <adc/operator/reconstruction.hpp>

#ifdef ADC_HAS_KOKKOS
#include <Kokkos_Core.hpp>
#endif

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main(int argc, char** argv) {
#ifdef ADC_HAS_KOKKOS
  Kokkos::initialize(argc, argv);
  const char* exec = Kokkos::DefaultExecutionSpace::name();
#else
  (void)argc;
  (void)argv;
  const char* exec = "serial-cpu";
#endif
  int rc = 0;
  {
    const int N = 128, ng = 2;
    const double L = 1.0, rho0 = 1.0, p0 = 1.0, gamma = 5.0 / 3.0;
    const double four_pi_G = 20.0, eps = 1e-3, k = 2 * kPi / L;
    const double cs2 = gamma * p0 / rho0;
    Box2D dom = Box2D::from_extents(N, N);
    Geometry geom{dom, 0.0, L, 0.0, L};
    BoxArray ba(std::vector<Box2D>{dom});
    DistributionMapping dm(1, 1);

    EulerPoisson model;
    model.hydro.gamma = gamma;
    model.four_pi_G = four_pi_G;
    model.rho0 = rho0;

    BCRec bcU, bcPhi;  // periodique
    Coupler<EulerPoisson> cpl(model, geom, ba, bcU, bcPhi);

    // perturbation acoustique-gravitationnelle au repos (Jeans, regime stable)
    MultiFab U(ba, dm, 4, ng);
    {
      Array4 u = U.fab(0).array();
      const Box2D g = U.fab(0).grown_box();
      auto wrap = [&](int x) { return (x % N + N) % N; };
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
          const double x = (wrap(i) + 0.5) / N;
          const double drho = eps * rho0 * std::cos(k * x);
          const double rho = rho0 + drho, p = p0 + cs2 * drho;
          u(i, j, 0) = rho;
          u(i, j, 1) = 0.0;
          u(i, j, 2) = 0.0;
          u(i, j, 3) = p / (gamma - 1);
        }
    }

    const double dt = 0.4 * (L / N) / (std::sqrt(cs2) + 0.1);
    const int nsteps = 10;
    for (int s = 0; s < nsteps; ++s) cpl.advance<Minmod>(U, dt);

#ifdef ADC_HAS_KOKKOS
    Kokkos::fence();
#endif
    const double mass = sum(U, 0), energy = sum(U, 3);
    const double px = sum(U, 1), py = sum(U, 2);
    double sumsq = 0, maxmom = 0;
    const ConstArray4 u = U.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
        for (int c = 0; c < 4; ++c) sumsq += u(i, j, c) * u(i, j, c);
        maxmom = std::fmax(maxmom, std::fabs(u(i, j, 1)));
      }

    std::printf(
        "exec=%s  N=%d  pas couples=%d (Euler-Poisson, 4 var)\n"
        "  masse=%.10e  energie=%.10e\n"
        "  p_tot=(%.3e, %.3e)  max|px|=%.10e\n"
        "  sum(U^2)=%.10e\n",
        exec, N, nsteps, mass, energy, px, py, maxmom, sumsq);

    const bool finite = std::isfinite(mass) && std::isfinite(sumsq);
    const bool conserv = std::fabs(px) < 1e-9 && std::fabs(py) < 1e-9;
    const bool active = maxmom > 0;  // la dynamique a bien fait quelque chose
    if (finite && conserv && active)
      std::printf("OK euler_poisson_kokkos (checksums a comparer CPU vs GPU)\n");
    else {
      std::printf("FAIL euler_poisson_kokkos (finite=%d conserv=%d active=%d)\n",
                  finite, conserv, active);
      rc = 1;
    }
  }
#ifdef ADC_HAS_KOKKOS
  Kokkos::finalize();
#endif
  return rc;
}
