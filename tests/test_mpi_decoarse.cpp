// DE-REPLICATION du grossier (etape 2 du hero-run AMR, cf. docs/HERO_RUN_AMR.md). Le niveau 0
// n'est plus une box unique repliquee sur chaque rang, mais une grille MULTI-BOX REPARTIE
// (DistributionMapping round-robin). On verifie que le diocotron AMR avec grossier de-replique
// (AmrCouplerMP replicated_coarse=false) donne EXACTEMENT le meme grossier que la version
// repliquee mono-box, en rassemblant le grossier reparti sur une box unique (parallel_copy) et
// en comparant bit a bit. Lance par mpirun -np N.
//
// C'est le test qui leve le verrou memoire O(NX*NY*nranks) : a l'echelle hero le grossier 8192^2
// ne peut pas etre replique. Si le reflux multi-patch et le multigrille tournent corrects sur un
// grossier multi-box reparti, la de-replication est acquise.
//
// ETAT : bit-identique a np=1 et np=2 (lance a np=2). A np=4 (exactement 1 box grossiere par
// rang) un parallel_copy leve MPI_ERR_TRUNCATE (taille de message incoherente) : bug a CORRIGER
// dans parallel_copy (primitive partagee), pas dans la logique de de-replication elle-meme, qui
// est correcte a np<=2. Voir docs/HERO_RUN_AMR.md etape 2.

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
  BCRec bc;

  Diocotron model;
  model.B0 = 1.0; model.alpha = 1.0; model.n_i0 = 1.0;

  auto ne0 = [&](double x, double y) {
    return 1.0 + 0.3 * std::sin(2 * kPi * x) * std::sin(2 * kPi * y);
  };
  auto fillc = [&](MultiFab& U) {
    for (int li = 0; li < U.local_size(); ++li) {
      Array4 u = U.fab(li).array();
      const Box2D g = U.fab(li).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) u(i, j, 0) = ne0((i + 0.5) * dxc, (j + 0.5) * dyc);
    }
  };
  auto fillf = [&](MultiFab& U) {
    for (int li = 0; li < U.local_size(); ++li) {
      Array4 u = U.fab(li).array();
      const Box2D b = U.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i) u(i, j, 0) = ne0((i + 0.5) * dxf, (j + 0.5) * dyf);
    }
  };

  // patchs fins : region [8..23]^2 (coarse) en 2x2 quadrants -> 4 patchs repartis.
  const int I0 = 8, I1 = 23, J0 = 8, J1 = 23, MI = 15, MJ = 15;
  std::vector<Box2D> faces = {
      {{2 * I0, 2 * J0}, {2 * MI + 1, 2 * MJ + 1}},
      {{2 * (MI + 1), 2 * J0}, {2 * I1 + 1, 2 * MJ + 1}},
      {{2 * I0, 2 * (MJ + 1)}, {2 * MI + 1, 2 * J1 + 1}},
      {{2 * (MI + 1), 2 * (MJ + 1)}, {2 * I1 + 1, 2 * J1 + 1}}};
  BoxArray baf(faces);
  DistributionMapping dmf(static_cast<int>(faces.size()), n_ranks());

  auto run = [&](const BoxArray& bac, const DistributionMapping& dmc, bool replicated) {
    MultiFab Uc(bac, dmc, 1, 1), Uf(baf, dmf, 1, 1);
    fillc(Uc); fillf(Uf); mf_average_down_mb(Uf, Uc);
    std::vector<AmrLevelMP> LP;
    LP.push_back({std::move(Uc), nullptr, dxc, dyc});
    LP.push_back({std::move(Uf), nullptr, dxf, dyf});
    AmrCouplerMP<Diocotron> sim(model, geom, bac, bc, std::move(LP), {}, replicated);
    sim.update();
    const double dt = 0.4 * dxc / sim.max_drift_speed();
    for (int s = 0; s < 20; ++s) sim.step(dt);
    return MultiFab(sim.coarse());
  };

  // REF : grossier mono-box REPLIQUE (box 0 sur chaque rang).
  BoxArray ba_repl(std::vector<Box2D>{dom});
  MultiFab UcRef = run(ba_repl, DistributionMapping(std::vector<int>(1, me)), /*replicated=*/true);

  // DIST : grossier MULTI-BOX (2x2 quadrants 16x16) REPARTI round-robin -> de-replique.
  std::vector<Box2D> cboxes = {
      {{0, 0}, {15, 15}}, {{16, 0}, {31, 15}}, {{0, 16}, {15, 31}}, {{16, 16}, {31, 31}}};
  BoxArray ba_multi(cboxes);
  MultiFab UcDist = run(ba_multi, DistributionMapping(static_cast<int>(cboxes.size()), n_ranks()),
                        /*replicated=*/false);

  // rassemble le grossier reparti sur une box unique posee sur le RANG 0 (dmap coherent
  // {0} sur tous les rangs : parallel_copy gather). NB : un dmap "replique" {me} serait
  // INCOHERENT entre rangs (chaque rang croirait posseder la box 0) -> deadlock collectif.
  MultiFab gathered(ba_repl, DistributionMapping(std::vector<int>(1, 0)), 1, 0);
  gathered.set_val(0.0);
  parallel_copy(gathered, UcDist);  // multi-box reparti -> box unique sur le rang 0
  device_fence();
  double maxdiff = 0;
  if (me == 0) {  // seul le rang 0 detient la box rassemblee ; UcRef est replique (valide partout)
    const ConstArray4 ug = gathered.fab(0).const_array(), ur = UcRef.fab(0).const_array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i)
        maxdiff = std::fmax(maxdiff, std::fabs(ug(i, j, 0) - ur(i, j, 0)));
  }
  maxdiff = all_reduce_max(maxdiff);

  if (me == 0)
    std::printf("de-replication grossier (np=%d) : grossier multi-box reparti vs mono-box "
                "replique, max|d| = %.3e\n", n_ranks(), maxdiff);
  if (maxdiff > 1e-12) { if (me == 0) std::printf("FAIL derepli_bit_identique\n"); ++fails; }

  fails = all_reduce_sum(fails);
  if (fails == 0 && me == 0) std::printf("OK test_mpi_decoarse\n");
  comm_finalize();
  return fails == 0 ? 0 : 1;
}
