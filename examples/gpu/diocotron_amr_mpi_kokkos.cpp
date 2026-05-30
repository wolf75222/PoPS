// Diocotron AMR distribue sur PLUSIEURS GPU : MPI + Kokkos/CUDA, 1 rang par GH200. Le
// coupleur AmrCouplerMP (Poisson grossier replique + injection d'aux + reflux multi-patch +
// regrid) tourne, ses patchs fins repartis sur les rangs, chaque rang sur son GPU. C'est le
// test multi-GPU de l'etape 1 du hero-run AMR (cf. docs/HERO_RUN_AMR.md) : il exerce le
// parallel_copy / all_reduce entre fabs DEVICE de rangs differents (MPI CUDA-aware).
//
// Gate d'invariance a la distribution : meme probleme, patchs fins repartis round-robin (DIST)
// vs tous sur le rang 0 (REF = mono-rang), executes par TOUS les rangs. Le grossier etant
// replique, chaque rang detient les deux resultats -> comparaison bit a bit. DIST == REF
// prouve que la repartition multi-GPU ne change pas le resultat.

#include <adc/coupling/amr_coupler_mp.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#ifdef ADC_HAS_KOKKOS
#include <Kokkos_Core.hpp>
#endif

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
#ifdef ADC_HAS_KOKKOS
  Kokkos::initialize(argc, argv);
  const char* exec = Kokkos::DefaultExecutionSpace::name();
#else
  const char* exec = "serie-cpu";
#endif
  int rc = 0;
  {
    const int me = my_rank();
    const int nc = 64;
    Box2D dom = Box2D::from_extents(nc, nc);
    Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
    const double dxc = geom.dx(), dyc = geom.dy(), dxf = dxc / 2, dyf = dyc / 2;
    BoxArray bac(std::vector<Box2D>{dom});
    DistributionMapping dmc(std::vector<int>(1, me));  // grossier REPLIQUE (box 0 / rang)
    BCRec bc;

    Diocotron model;
    model.B0 = 1.0; model.alpha = 1.0; model.n_i0 = 1.0;

    auto ne0 = [&](double x, double y) {
      return 1.0 + 0.3 * std::sin(2 * kPi * x) * std::sin(2 * kPi * y);
    };
    auto initc = [&](MultiFab& U) {
      Array4 u = U.fab(0).array();
      const Box2D g = U.fab(0).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) u(i, j, 0) = ne0((i + 0.5) * dxc, (j + 0.5) * dyc);
    };
    auto initf = [&](MultiFab& U) {
      for (int li = 0; li < U.local_size(); ++li) {
        Array4 u = U.fab(li).array();
        const Box2D b = U.box(li);
        for (int j = b.lo[1]; j <= b.hi[1]; ++j)
          for (int i = b.lo[0]; i <= b.hi[0]; ++i) u(i, j, 0) = ne0((i + 0.5) * dxf, (j + 0.5) * dyf);
      }
    };
    // region fine [16..47]^2 en 2x2 quadrants (4 patchs) : un patch par GPU a np=4.
    const int I0 = 16, I1 = 47, J0 = 16, J1 = 47, MI = 31, MJ = 31;
    std::vector<Box2D> faces = {
        {{2 * I0, 2 * J0}, {2 * MI + 1, 2 * MJ + 1}},
        {{2 * (MI + 1), 2 * J0}, {2 * I1 + 1, 2 * MJ + 1}},
        {{2 * I0, 2 * (MJ + 1)}, {2 * MI + 1, 2 * J1 + 1}},
        {{2 * (MI + 1), 2 * (MJ + 1)}, {2 * I1 + 1, 2 * J1 + 1}}};
    BoxArray baf(faces);

    auto run = [&](const DistributionMapping& dmf) {
      MultiFab Uc(bac, dmc, 1, 1), Uf(baf, dmf, 1, 1);
      initc(Uc); initf(Uf); mf_average_down_mb(Uf, Uc);
      std::vector<AmrLevelMP> LP;
      LP.push_back({std::move(Uc), nullptr, dxc, dyc});
      LP.push_back({std::move(Uf), nullptr, dxf, dyf});
      AmrCouplerMP<Diocotron> sim(model, geom, bac, bc, std::move(LP));
      sim.update();
      const double dt = 0.4 * dxc / sim.max_drift_speed();
      for (int s = 0; s < 20; ++s) sim.step(dt);
      return MultiFab(sim.coarse());
    };

    MultiFab UcDist = run(DistributionMapping(static_cast<int>(faces.size()), n_ranks()));
    MultiFab UcRef = run(DistributionMapping(std::vector<int>(faces.size(), 0)));

    device_fence();
    double maxdiff = 0;
    const ConstArray4 ud = UcDist.fab(0).const_array(), ur = UcRef.fab(0).const_array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i)
        maxdiff = std::fmax(maxdiff, std::fabs(ud(i, j, 0) - ur(i, j, 0)));
    maxdiff = all_reduce_max(maxdiff);

    if (me == 0) {
      std::printf("diocotron AMR multi-GPU (exec=%s, np=%d) : max|Uc_dist - Uc_ref| = %.3e\n",
                  exec, n_ranks(), maxdiff);
      std::printf(maxdiff <= 1e-14 ? "OK diocotron_amr_mpi_kokkos\n"
                                   : "FAIL distribution_invariante\n");
    }
    rc = (maxdiff <= 1e-14) ? 0 : 1;
  }
#ifdef ADC_HAS_KOKKOS
  Kokkos::finalize();
#endif
  comm_finalize();
  return rc;
}
