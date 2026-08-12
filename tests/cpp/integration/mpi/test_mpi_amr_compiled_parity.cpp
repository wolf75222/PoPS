// VALIDATION INTEGREE AmrSystem + MPI + GPU (deliverable C). Un SEUL run combine, pour la premiere
// fois, les trois axes qui n'avaient ete valides que SEPAREMENT sur GH200 (cf docs/GPU_RUNTIME_PORT.md,
// phases 5/6/9) :
//   - une HIERARCHIE AMR reelle (AmrSystem : grossier replique + niveau fin multi-patch suivi par
//     regrid Berger-Rigoutsos, reflux conservatif, Poisson grossier a chaque pas) ;
//   - un MODELE COMPILE branche par add_compiled_model(AmrSystem, ...) (CompositeModel connu a la
//     compilation, chemin amr_dsl_block.hpp, PR #45) ;
//   - une DISTRIBUTION MPI : les patchs fins sont repartis sur n_ranks() GPU (un par rang), halos
//     cross-rang via fill_boundary, reflux et masse reduits par all_reduce.
//
// Propriete verifiee : l'evolution de la hierarchie est INVARIANTE AU NOMBRE DE RANGS. Le grossier
// etant REPLIQUE (defaut AmrCouplerMP), density(), mass() et les vues de checkpoint
// level_{state,potential}_global(0) sont des grandeurs GLOBALES identiques sur chaque rang ; le
// decoupage du niveau fin entre rangs (et donc le chemin MPI : halos distants,
// injection parallel_copy, reflux route vers la box parente distante) ne doit RIEN changer au
// resultat bit a bit. On le controle de deux manieres complementaires :
//   (1) CONSISTANCE CROSS-RANG dans le run : tous les rangs voient la MEME densite grossiere et la
//       MEME masse (diff max reduite sur les rangs == 0). Sans cela, un bug de halo/reflux distant
//       casserait silencieusement la replication.
//   (2) PARITE AU NB DE RANGS : une ligne de signature canonique contient le checksum de la densite
//       et la masse. Le CTest rank-parity relance ce MEME binaire en np=1/2/4 et compare exactement
//       les signatures. np=1 est l'oracle mono-rang ; np=2/4 doivent etre BIT-IDENTIQUES (dmax=0).
//
// Independant du backend : vert sous Kokkos Serial (CI, CPU) ET sous Kokkos Cuda (ROMEO GH200,
// multi-GPU). Sous Cuda, for_each_cell ne fence pas (async) : density()/mass() de l'AmrSystem font
// deja un device_fence() interne avant la lecture hote (read_coarse / amr_read_coarse), donc la
// lecture hote ici est sure. On insere malgre tout un Kokkos::fence() de ceinture avant les diffs.
#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include "gtest_compat.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/physics/bricks/bricks.hpp>  // CompositeModel, GravityForce, GravityCoupling
#include <pops/physics/fluids/euler.hpp>   // EulerND (transport compressible)
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>  // add_compiled_model(AmrSystem, ...)
#include <pops/runtime/amr_system.hpp>

#include "amr_tagging_test_authority.hpp"
#include <pops/parallel/comm.hpp>  // comm_init, my_rank, n_ranks, all_reduce_*

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

// QUATRE bulles de densite lisses, periodiques, bien separees : chacune depasse le seuil de
// raffinement -> Berger-Rigoutsos produit PLUSIEURS patchs fins disjoints, que le regrid REPARTIT
// sur les rangs (round-robin DistributionMapping(nfine, n_ranks())). C'est ce qui distribue
// reellement le niveau fin sur plusieurs GPU (et non un seul patch central sur un seul rang).
template <int Dim>
std::size_t cell_count(const Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

template <int Dim>
static std::vector<double> four_bubbles(const Extent<Dim>& shape) {
  std::vector<double> rho(cell_count(shape));
  for (std::size_t ordinal = 0; ordinal < rho.size(); ++ordinal) {
    std::size_t remainder = ordinal;
    double radius_squared[4]{};
    for (int axis = 0; axis < Dim; ++axis) {
      const auto extent = static_cast<std::size_t>(shape[axis]);
      const double coordinate =
          (static_cast<double>(remainder % extent) + 0.5) / static_cast<double>(extent);
      remainder /= extent;
      for (int bubble = 0; bubble < 4; ++bubble) {
        const double center = ((bubble >> (axis % 2)) & 1) == 0 ? 0.25 : 0.75;
        const double delta = coordinate - center;
        radius_squared[bubble] += delta * delta;
      }
    }
    double value = 1.0;
    for (double radius : radius_squared)
      value += 0.5 * std::exp(-radius / 0.004);
    rho[ordinal] = value;
  }
  // Periodic self-gravity requires an RHS orthogonal to the constant nullspace. Preserve the four
  // non-trivial peaks but encode their neutralizing background in the fixture; no solver-side
  // projection is permitted.
  double mean = 0.0;
  for (double value : rho)
    mean += value;
  mean /= static_cast<double>(rho.size());
  for (double& value : rho)
    value += 1.0 - mean;
  return rho;
}

template <int Dim>
using Model = CompositeModel<EulerND<Dim>, GravityForceND<Dim>, GravityCoupling>;

template <int Dim>
Model<Dim> gravity_model() {
  Model<Dim> model{};
  model.hyp = EulerND<Dim>::prepare(Real(1.4));
  model.src = GravityForceND<Dim>{};
  model.ell = GravityCoupling{Real(-1.0), Real(1.0), Real(1.0)};
  return model;
}

static double max_abs_difference(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size())
    return std::numeric_limits<double>::infinity();
  double dmax = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    dmax = std::fmax(dmax, std::fabs(a[i] - b[i]));
  return dmax;
}

static int pops_run_test_mpi_amr_compiled_parity(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int me = my_rank(), np = n_ranks();
  const int n = 64;
  constexpr int Dim = kNativeDimension;

  AmrSystemConfig<Dim> cfg;
  for (int axis = 0; axis < Dim; ++axis) {
    cfg.shape[axis] = n;
    cfg.periodicity[axis] = true;
  }
  const std::vector<double> rho = four_bubbles(cfg.shape);
  cfg.regrid_every = 4;  // re-raffinement periodique : exerce le regrid distribue plusieurs fois

  // Modele euler_poisson COMPILE branche sur la hierarchie AMR (chemin de production add_compiled_model).
  AmrSystem<Dim> sys(cfg);
  sys.set_temporal_relations({2}, {1}, {"integral_only"});
  add_compiled_model<Dim>(sys, "gas", gravity_model<Dim>(), "minmod", "rusanov", "conservative",
                          "explicit", /*gamma=*/1.4);
  sys.set_poisson("charge_density", "geometric_mg");
  test::install_prepared_threshold_union(sys, {{"gas", "rho", 1.2}});
  sys.set_density("gas", rho);
  test::install_forward_euler_program(sys);

  const double m0 = sys.mass();  // declenche le build paresseux (regrid initial distribue)
  const int np0 = sys.n_patches();

  // Plusieurs macro-pas AMR : chaque pas = Poisson grossier + injection vers les fins + transport
  // multi-niveaux + reflux conservatif ; tous les 4 pas, regrid Berger-Rigoutsos (redistribue les
  // patchs sur les rangs). C'est la totalite du chemin AMR + MPI exercee ensemble.
  const double dt = 1e-3;
  const int nsteps = 16;
  for (int s = 0; s < nsteps; ++s)
    sys.step(dt);

#if defined(POPS_HAS_KOKKOS)
  Kokkos::fence();  // ceinture avant la lecture hote (density()/mass() fencent deja en interne)
#endif
  const std::vector<double> dens = sys.density();  // grossier REPLIQUE : identique sur chaque rang
  const double mass = sys.mass();
  const int npf = sys.n_patches();
  // Contract checkpoint MPI: level_state_global / level_potential_global gather owned fields, but
  // level 0 is replicated. Reducing that level would multiply it by np. density()/potential() are
  // independent production diagnostics for component 0 and phi, respectively.
  const std::vector<double> state_global = sys.level_state_global(0);
  const std::vector<double> phi = sys.potential();
  const std::vector<double> phi_global = sys.level_potential_global(0);
  const std::size_t nn = cell_count(cfg.shape);
  const double state_density_dmax =
      state_global.size() >= nn
          ? max_abs_difference(std::vector<double>(state_global.begin(), state_global.begin() + nn),
                               dens)
          : std::numeric_limits<double>::infinity();
  const double phi_dmax = max_abs_difference(phi_global, phi);

  // Checksum de la densite grossiere (somme + somme des carres + max) : signature bit-sensible du
  // champ final, comparable entre nombres de rangs par le script de build.
  double csum = 0, csumsq = 0, cmax = 0;
  for (double v : dens) {
    csum += v;
    csumsq += v * v;
    const double a = std::fabs(v);
    if (a > cmax)
      cmax = a;
  }

  // (1) CONSISTANCE CROSS-RANG : le grossier replique impose que chaque rang ait EXACTEMENT le meme
  // champ. On reduit l'ecart max entre les checksums locaux et ceux du rang 0 (max - min == 0 ssi
  // tous egaux). On compare via all_reduce_max/min des memes quantites.
  const double smax = all_reduce_max(csum), smin = -all_reduce_max(-csum);
  const double qmax = all_reduce_max(csumsq), qmin = -all_reduce_max(-csumsq);
  const double mmax = all_reduce_max(mass), mmin = -all_reduce_max(-mass);
  const double xmax = all_reduce_max(cmax), xmin = -all_reduce_max(-cmax);
  const double spread =
      std::fmax(std::fmax(smax - smin, qmax - qmin), std::fmax(mmax - mmin, xmax - xmin));

  int fails = 0;
  if (me == 0) {
    // Signature sans np : l'orchestrateur CTest generique exige exactement la meme ligne pour
    // chaque nombre de rangs declare dans tests/test_manifest.toml.
    std::printf(
        "POPS_MPI_PARITY_SIGNATURE_test_mpi_amr_compiled_parity "
        "patches0=%d patchesF=%d mass=%.17e csum=%.17e csumsq=%.17e cmax=%.17e "
        "state0_vs_density=%.17e phi_vs_global=%.17e\n",
        np0, npf, mass, csum, csumsq, cmax, state_density_dmax, phi_dmax);
    std::printf(
        "AMRMPI np=%d patches0=%d patchesF=%d | mass=%.17e | csum=%.17e csumsq=%.17e "
        "cmax=%.17e | crossrank_spread=%.3e | state0_vs_density=%.3e phi_vs_global=%.3e\n",
        np, np0, npf, mass, csum, csumsq, cmax, spread, state_density_dmax, phi_dmax);
#if defined(POPS_HAS_KOKKOS)
    const char* space = Kokkos::DefaultExecutionSpace::name();
#else
    const char* space = "Serial(host)";
#endif
    std::printf("AMRMPI exec=%s m0=%.17e (conservation: dm=%.3e)\n", space, m0,
                std::fabs(mass - m0));

    if (!(dens.size() == nn)) {
      std::printf("FAIL densite grossiere de mauvaise taille\n");
      ++fails;
    }
    if (!(state_global.size() >= nn && state_density_dmax == 0.0)) {
      std::printf("FAIL level_state_global(0) replique != densite (dmax=%.3e)\n",
                  state_density_dmax);
      ++fails;
    }
    if (!(phi_global.size() == phi.size() && phi_dmax == 0.0)) {
      std::printf("FAIL level_potential_global(0) replique != potential (dmax=%.3e)\n", phi_dmax);
      ++fails;
    }
    if (!(cmax > 1e-6)) {
      std::printf("FAIL densite triviale (pas de signal)\n");
      ++fails;
    }
    // >= 2 patchs fins : sous np>=2 ils se repartissent sur plusieurs rangs/GPU (round-robin),
    // exercant le chemin fin DISTRIBUE (halos cross-rang, reflux route vers la box parente distante)
    // et pas seulement le grossier replique. Les 4 bulles produisent typiquement 4 patchs.
    if (!(npf >= 2)) {
      std::printf("FAIL < 2 patchs fins (niveau fin non distribuable)\n");
      ++fails;
    }
    // Le grossier replique DOIT etre bit-identique sur tous les rangs (spread exactement 0).
    if (!(spread == 0.0)) {
      std::printf("FAIL grossier non bit-identique entre rangs\n");
      ++fails;
    }
    if (fails == 0)
      std::printf(
          "OK test_mpi_amr_compiled_parity np=%d (AmrSystem+MPI+compile : grossier "
          "bit-identique cross-rang)\n",
          np);
  }
  comm_finalize();
  return fails ? 1 : 0;
}

TEST(test_mpi_amr_compiled_parity, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_mpi_amr_compiled_parity,
                                    "test_mpi_amr_compiled_parity"),
            0);
}
