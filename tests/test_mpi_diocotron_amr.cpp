// Diocotron AMR 2-niveaux DISTRIBUE de bout en bout via le coupleur AmrCouplerMP
// (Poisson grossier + injection d'aux + pas multi-patch + regrid), lance par mpirun -np N.
// C'est l'etape 0 du hero-run AMR distribue (cf. docs/HERO_RUN_AMR.md).
//
// Le niveau 0 (grossier mono-box) est REPLIQUE sur chaque rang : le multigrille de Poisson
// l'est aussi (AmrCouplerMP replicated_coarse=true -> GeometricMG replicated=true), donc chaque
// rang resout le meme Poisson grossier et chaque patch fin, ou qu'il tombe, lit son aux parent
// localement. Les patchs fins sont repartis. Deux gardes :
//   1. INVARIANCE A LA DISTRIBUTION : meme probleme, patchs fins repartis round-robin (DIST) vs
//      tous sur le rang 0 (REF = np=1), executes par TOUS les rangs (collectives appariees).
//      Le grossier etant replique, chaque rang detient les deux resultats -> bit a bit.
//   2. REGRID DYNAMIQUE DISTRIBUE : regrid Berger-Rigoutsos periodique sous MPI, masse grossiere
//      conservee a l'arrondi, etat fini.

#include <adc/coupling/amr_coupler_mp.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank();
  long fails = 0;

  const int nc = 32;
  Box2D dom = Box2D::from_extents(nc, nc);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const double dxc = geom.dx(), dyc = geom.dy(), dxf = dxc / 2, dyf = dyc / 2;
  BoxArray bac(std::vector<Box2D>{dom});
  DistributionMapping dmc(std::vector<int>(1, me));  // grossier REPLIQUE (box 0 sur chaque rang)
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

  // region fine [8..23]^2 en 2x2 quadrants (4 patchs, ratio 2) : meme decoupe que le test reflux.
  const int I0 = 8, I1 = 23, J0 = 8, J1 = 23, MI = 15, MJ = 15;
  std::vector<Box2D> faces = {
      {{2 * I0, 2 * J0}, {2 * MI + 1, 2 * MJ + 1}},
      {{2 * (MI + 1), 2 * J0}, {2 * I1 + 1, 2 * MJ + 1}},
      {{2 * I0, 2 * (MJ + 1)}, {2 * MI + 1, 2 * J1 + 1}},
      {{2 * (MI + 1), 2 * (MJ + 1)}, {2 * I1 + 1, 2 * J1 + 1}}};
  BoxArray baf(faces);

  // ===== Garde 1 : invariance a la distribution des patchs fins =====
  auto run = [&](const DistributionMapping& dmf) {
    MultiFab Uc(bac, dmc, 1, 1), Uf(baf, dmf, 1, 1);
    initc(Uc); initf(Uf); mf_average_down_mb(Uf, Uc);
    std::vector<AmrLevelMP> LP;
    LP.push_back({std::move(Uc), nullptr, dxc, dyc});
    LP.push_back({std::move(Uf), nullptr, dxf, dyf});
    AmrCouplerMP<Diocotron> sim(model, geom, bac, bc, std::move(LP));  // replicated_coarse=true
    sim.update();
    const double dt = 0.4 * dxc / sim.max_drift_speed();
    for (int s = 0; s < 20; ++s) sim.step(dt);
    return MultiFab(sim.coarse());  // copie du grossier replique
  };

  MultiFab UcDist = run(DistributionMapping(static_cast<int>(faces.size()), n_ranks()));
  MultiFab UcRef = run(DistributionMapping(std::vector<int>(faces.size(), 0)));

  double maxdiff = 0;
  const ConstArray4 ud = UcDist.fab(0).const_array(), ur = UcRef.fab(0).const_array();
  for (int j = 0; j < nc; ++j)
    for (int i = 0; i < nc; ++i)
      maxdiff = std::fmax(maxdiff, std::fabs(ud(i, j, 0) - ur(i, j, 0)));
  maxdiff = all_reduce_max(maxdiff);
  if (maxdiff > 1e-14) { if (me == 0) std::printf("FAIL distribution_invariante (%.3e)\n", maxdiff); ++fails; }

  // ===== Garde 2 : regrid Berger-Rigoutsos dynamique distribue + conservation =====
  {
    auto blob = [&](double x, double y) {
      return 1.0 + 0.6 * std::exp(-((x - 0.32) * (x - 0.32) + (y - 0.5) * (y - 0.5)) / 0.004) +
             0.6 * std::exp(-((x - 0.68) * (x - 0.68) + (y - 0.5) * (y - 0.5)) / 0.004);
    };
    auto crit = [&](const ConstArray4& a, int i, int j) { return a(i, j, 0) > model.n_i0 + 0.05; };
    Box2D seed{{2 * (nc / 4), 2 * (nc / 4)}, {2 * (3 * nc / 4) - 1, 2 * (3 * nc / 4) - 1}};
    MultiFab Uc(bac, dmc, 1, 1);
    MultiFab Uf(BoxArray(std::vector<Box2D>{seed}),
                DistributionMapping(1, n_ranks()), 1, 1);
    {
      Array4 u = Uc.fab(0).array();
      const Box2D g = Uc.fab(0).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) u(i, j, 0) = blob((i + 0.5) * dxc, (j + 0.5) * dyc);
      for (int li = 0; li < Uf.local_size(); ++li) {
        Array4 uf = Uf.fab(li).array();
        const Box2D b = Uf.box(li);
        for (int j = b.lo[1]; j <= b.hi[1]; ++j)
          for (int i = b.lo[0]; i <= b.hi[0]; ++i) uf(i, j, 0) = blob((i + 0.5) * dxf, (j + 0.5) * dyf);
      }
    }
    std::vector<AmrLevelMP> LP;
    LP.push_back({std::move(Uc), nullptr, dxc, dyc});
    LP.push_back({std::move(Uf), nullptr, dxf, dyf});
    AmrCouplerMP<Diocotron> sim(model, geom, bac, bc, std::move(LP));

    sim.regrid(crit);
    sim.update();
    const double m0 = sim.mass();
    const double dt = 0.4 * dxc / sim.max_drift_speed();
    bool finite = true;
    for (int s = 0; s < 60; ++s) {
      if (s % 10 == 0) sim.regrid(crit);
      sim.step(dt);
      if (!std::isfinite(sim.mass())) finite = false;
    }
    const double drift = all_reduce_max(std::fabs(sim.mass() - m0));
    finite = all_reduce_max(finite ? 0.0 : 1.0) == 0.0;
    if (me == 0)
      std::printf("regrid distribue (np=%d) : derive_masse=%.3e %s\n", n_ranks(), drift,
                  finite ? "fini" : "NON-FINI");
    if (!finite) { if (me == 0) std::printf("FAIL regrid_fini\n"); ++fails; }
    if (drift > 1e-9) { if (me == 0) std::printf("FAIL regrid_conservation\n"); ++fails; }
  }

  if (me == 0)
    std::printf("diocotron AMR distribue (np=%d) : max|Uc_dist - Uc_ref| = %.3e\n",
                n_ranks(), maxdiff);
  fails = all_reduce_sum(fails);
  if (fails == 0 && me == 0) std::printf("OK test_mpi_diocotron_amr\n");
  comm_finalize();
  return fails == 0 ? 0 : 1;
}
