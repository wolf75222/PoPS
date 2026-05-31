// DE-REPLICATION du grossier (etape 2 du hero-run AMR, cf. docs/HERO_RUN_AMR.md). Le niveau 0
// n'est plus une box unique repliquee sur chaque rang, mais une grille MULTI-BOX REPARTIE
// (DistributionMapping round-robin). On verifie que le diocotron AMR avec grossier de-replique
// (AmrCouplerMP replicated_coarse=false) donne EXACTEMENT le meme grossier que la version
// repliquee mono-box, en rassemblant le grossier reparti sur une box unique (parallel_copy) et
// en comparant bit a bit. Lance par mpirun -np N.
//
// C'est le test qui leve le verrou memoire O(NX*NY*nranks) : a l'echelle hero le grossier 8192^2
// ne peut pas etre replique. Le reflux multi-patch et le multigrille tournent corrects sur un
// grossier multi-box reparti, donc la de-replication est acquise : bit-identique np=1/2/4.
//
// Note : ce test a revele que GeometricMG::current_residual ne reduisait pas norm_inf entre rangs
// (max LOCAL), d'ou un nombre de V-cycles different par rang sur un grossier multi-box reparti,
// donc des sequences fill_boundary desynchronisees -> MPI_ERR_TRUNCATE a np=4. Corrige par
// all_reduce_max sur le residu (idempotent sous replication).

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

  // un run AMR diocotron complet (20 pas) sur un grossier donne, renvoie le grossier final. Les
  // patchs fins (baf/dmf) sont parametres pour pouvoir exercer differents motifs de couverture.
  auto run = [&](const BoxArray& bac, const DistributionMapping& dmc, bool replicated,
                 const BoxArray& baf, const DistributionMapping& dmf) {
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

  // grossier mono-box REPLIQUE (box 0 sur chaque rang) : la reference.
  BoxArray ba_repl(std::vector<Box2D>{dom});
  // grossier 2x2 (4 boxes 16x16) REPARTI round-robin -> de-replique. La hierarchie multigrille
  // coarsen 16->8->4->2->1 : 4 boxes 1x1 pavent EXACTEMENT le fond MG 2x2, donc le MG multi-box
  // est bit-identique au mono-box. (Un decoupage plus fin, p.ex. 4x4, ne pave PAS le fond 2x2 :
  // la hierarchie MG degenere et converge a un point distinct a la tolerance pres, non
  // bit-identique et non deterministe -> inutilisable comme oracle. On garde donc 2x2.)
  BoxArray ba_dist(std::vector<Box2D>{
      {{0, 0}, {15, 15}}, {{16, 0}, {31, 15}}, {{0, 16}, {15, 31}}, {{16, 16}, {31, 31}}});

  // Compare le grossier de-replique (4-box) au grossier mono-box replique, pour un motif de
  // patchs fins donne. Rassemble le reparti sur une box unique posee sur le RANG 0 (dmap coherent
  // {0} sur tous les rangs : parallel_copy gather ; un dmap "replique" {me} serait INCOHERENT
  // entre rangs -> deadlock collectif), compare bit a bit sur le rang 0. Doit etre BIT IDENTIQUE.
  auto check = [&](const BoxArray& baf, const DistributionMapping& dmf, const char* label) {
    MultiFab UcRef = run(ba_repl, DistributionMapping(std::vector<int>(1, me)), true, baf, dmf);
    MultiFab UcDist = run(ba_dist, DistributionMapping(4, n_ranks()), false, baf, dmf);
    MultiFab gathered(ba_repl, DistributionMapping(std::vector<int>(1, 0)), 1, 0);
    gathered.set_val(0.0);
    parallel_copy(gathered, UcDist);  // multi-box reparti -> box unique sur le rang 0
    device_fence();
    double maxdiff = 0;
    if (me == 0) {  // seul le rang 0 detient la box rassemblee ; UcRef replique (valide partout)
      const ConstArray4 ug = gathered.fab(0).const_array(), ur = UcRef.fab(0).const_array();
      for (int j = 0; j < nc; ++j)
        for (int i = 0; i < nc; ++i)
          maxdiff = std::fmax(maxdiff, std::fabs(ug(i, j, 0) - ur(i, j, 0)));
    }
    maxdiff = all_reduce_max(maxdiff);
    if (me == 0)
      std::printf("de-replication grossier (np=%d) : %s, max|d| = %.3e\n", n_ranks(), label,
                  maxdiff);
    if (maxdiff > 1e-12) { if (me == 0) std::printf("FAIL %s\n", label); ++fails; }
  };

  // motif A : 4 patchs fins en 2x2 quadrants sur [8..23]^2 (coarse). A np=4 chaque empreinte
  // grossiere fine tombe dans la box grossiere du MEME rang (alignement round-robin) : ce motif
  // n'exerce PAS le chemin "box parente distante".
  const int I0 = 8, I1 = 23, J0 = 8, J1 = 23, MI = 15, MJ = 15;
  std::vector<Box2D> quad = {
      {{2 * I0, 2 * J0}, {2 * MI + 1, 2 * MJ + 1}},
      {{2 * (MI + 1), 2 * J0}, {2 * I1 + 1, 2 * MJ + 1}},
      {{2 * I0, 2 * (MJ + 1)}, {2 * MI + 1, 2 * J1 + 1}},
      {{2 * (MI + 1), 2 * (MJ + 1)}, {2 * I1 + 1, 2 * J1 + 1}}};
  check(BoxArray(quad), DistributionMapping(4, n_ranks()), "4 patchs quadrants (aligne)");

  // motif B : UN seul patch fin CENTRE sur la jonction des 4 boxes grossieres. Fine [24..39]^2 ->
  // empreinte grossiere [12..19]^2 qui CHEVAUCHE la frontiere coarse 16 en x ET y, donc touche les
  // 4 boxes grossieres. Le patch unique vit sur le rang 0 (dmf {0}), mais a np=4 il lit le flux
  // grossier des boxes 1,2,3 possedees par des rangs DISTANTS : c'est exactement le chemin ou
  // mf_find_box renvoyait -1 -> fx.fab(-1) -> segfault (corrige : route par parallel_copy). MG
  // 4-box propre -> BIT IDENTIQUE et deterministe.
  std::vector<Box2D> center = {{{24, 24}, {39, 39}}};
  check(BoxArray(center), DistributionMapping(1, n_ranks()), "1 patch centre (box parente distante)");

  fails = all_reduce_sum(fails);
  if (fails == 0 && me == 0) std::printf("OK test_mpi_decoarse\n");
  comm_finalize();
  return fails == 0 ? 0 : 1;
}
