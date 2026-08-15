// Historical design context follows.  The initialized exact-ranked hierarchy is qualified; positive
// multi-level advance is intentionally fail-closed until the public synchronization provider exists.
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
// multi-GPU). Sous Cuda, for_each_cell ne fence pas (async) : les diagnostics du runtime font
// deja un device_fence() interne avant la lecture hote (read_coarse / amr_read_coarse), donc la
// lecture hote ici est sure. On insere malgre tout un Kokkos::fence() de ceinture avant les diffs.
#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include "gtest_compat.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>  // add_compiled_model(AmrSystem, ...)
#include <pops/runtime/amr_system.hpp>

#include "amr_tagging_test_authority.hpp"
#include <pops/parallel/comm.hpp>  // comm_init, my_rank, n_ranks, all_reduce_*

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

template <int Dim>
struct AdvectionModel {
  using Law = nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;
  Law law{};

  static PreparedProviderIdentity provider_identity() noexcept {
    return {"test.mpi-amr-compiled-parity.scalar-advection", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(law.velocity()[axis]);
  }
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"u"}, 1, {VariableRole::Scalar}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"u"}, 1, {VariableRole::Scalar}};
  }
  POPS_HD nd::StateConversion<Primitive> recover(const State& state) const { return law.recover(state); }
  POPS_HD nd::StateConversion<State> make_conservative(const Primitive& primitive) const {
    return law.make_conservative(primitive);
  }
  POPS_HD nd::StateConversionStatus admissibility(const State& state) const {
    return law.admissibility(state);
  }
  template <int Axis>
  POPS_HD State flux(const State& state) const { return law.template flux<Axis>(state); }
  template <int Axis>
  POPS_HD Real max_wave_speed(const State& state) const { return law.template max_wave_speed<Axis>(state); }
  template <int Axis>
  POPS_HD void wave_speeds(const State& state, Real& lower, Real& upper) const {
    law.template wave_speeds<Axis>(state, lower, upper);
  }
  POPS_HD State source(const State&, const ProviderValues<0>&) const { return {}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

template <int Dim>
AdvectionModel<Dim> advection_model() {
  return {nd::ScalarAdvection<Dim>::prepare(RealVector<Dim>{})};
}

// QUATRE bulles de densite lisses, periodiques, bien separees : chacune depasse le seuil de
// raffinement -> Berger-Rigoutsos produit PLUSIEURS patchs fins disjoints, que le regrid REPARTIT
// sur les rangs (round-robin DistributionMapping(nfine, n_ranks())). C'est ce qui distribue
// reellement le niveau fin sur plusieurs GPU (et non un seul patch central sur un seul rang).
template <int Dim>
static std::size_t cell_count(const Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

template <int Dim>
static std::vector<double> four_bubbles(const Extent<Dim>& shape) {
  std::vector<double> rho(cell_count(shape));
  const std::size_t bubble_count = std::size_t{1} << Dim;
  for (std::size_t linear = 0; linear < rho.size(); ++linear) {
    std::size_t remainder = linear;
    std::array<double, Dim> coordinates{};
    for (int axis = 0; axis < Dim; ++axis) {
      const auto extent = static_cast<std::size_t>(shape[axis]);
      coordinates[axis] =
          (static_cast<double>(remainder % extent) + 0.5) / static_cast<double>(extent);
      remainder /= extent;
    }
    double value = 1.0;
    for (std::size_t bubble = 0; bubble < bubble_count; ++bubble) {
      double radius_squared = 0.0;
      for (int axis = 0; axis < Dim; ++axis) {
        const double center = (bubble & (std::size_t{1} << axis)) == 0 ? 0.25 : 0.75;
        const double displacement = coordinates[axis] - center;
        radius_squared += displacement * displacement;
      }
      value += 0.5 * std::exp(-radius_squared / 0.004);
    }
    rho[linear] = value;
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
  constexpr int Dim = kNativeDimension;
  AmrSystemConfig<Dim> cfg;
  for (int axis = 0; axis < Dim; ++axis)
    cfg.shape[axis] = 64;
  cfg.regrid_every = 4;  // re-raffinement periodique : exerce le regrid distribue plusieurs fois
  const std::vector<double> rho = four_bubbles(cfg.shape);

  // Modele euler_poisson COMPILE branche sur la hierarchie AMR (chemin de production add_compiled_model).
  AmrSystem<Dim> sys(cfg);
  sys.set_temporal_relations({2}, {1}, {"integral_only"});
  sys.install_block_state_route("gas", "test.mpi-amr-compiled-parity/state/gas");
  add_compiled_model<Dim>(sys, "gas", advection_model<Dim>(), "minmod", "rusanov",
                          "conservative", "explicit", /*gamma=*/1.4, /*substeps=*/1,
                          /*stride=*/1, {}, {}, /*positivity_floor=*/0.0,
                          static_cast<double>(kWenoEpsilon), /*wave_speed_cache=*/false,
                          "test.mpi-amr-compiled-parity.gas");
  test::install_prepared_threshold_union(sys, {{"gas", "u", 1.2}});
  sys.set_density("gas", rho);
  test::install_forward_euler_program(sys);

  const double m0 = sys.mass();  // declenche le build paresseux (regrid initial distribue)
  const int np0 = sys.n_patches();
  const std::vector<double> before_density = sys.density();
  const std::vector<AmrPatch<Dim>> before_patches = sys.patch_boxes();
  const int before_levels = sys.n_levels();

  bool advance_refused = false;
  try {
    sys.step(1e-3);
  } catch (const std::exception& error) {
    advance_refused = std::string(error.what()).find(
                          "requires a prepared exact-ranked conservative multi-level synchronization provider") !=
                      std::string::npos;
  }

#if defined(POPS_HAS_KOKKOS)
  Kokkos::fence();  // ceinture avant la lecture hote (density()/mass() fencent deja en interne)
#endif
  const std::vector<double> dens = sys.density();  // grossier REPLIQUE : identique sur chaque rang
  const double mass = sys.mass();
  const int npf = sys.n_patches();
  const bool no_mutation = dens == before_density && sys.patch_boxes() == before_patches &&
                           sys.n_levels() == before_levels && mass == m0;
  // Contract checkpoint MPI: level_state_global / level_potential_global gather owned fields, but
  // level 0 is replicated. Reducing that level would multiply it by np. density()/potential() are
  // independent production diagnostics for component 0 and phi, respectively.
  const std::vector<double> state_global = sys.level_state_global(0);
  const std::size_t nn = cell_count(cfg.shape);
  const double state_density_dmax =
      state_global.size() >= nn
          ? max_abs_difference(std::vector<double>(state_global.begin(), state_global.begin() + nn),
                               dens)
          : std::numeric_limits<double>::infinity();

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
        "patches0=%d patchesF=%d csum=%.17e csumsq=%.17e cmax=%.17e "
        "state0_vs_density=%.17e\n",
        np0, npf, csum, csumsq, cmax, state_density_dmax);
    std::printf(
        "AMRMPI np=%d patches0=%d patchesF=%d | mass=%.17e | csum=%.17e csumsq=%.17e "
        "cmax=%.17e | crossrank_spread=%.3e | state0_vs_density=%.3e\n",
        np, np0, npf, mass, csum, csumsq, cmax, spread, state_density_dmax);
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
    if (!advance_refused || !no_mutation) {
      std::printf("FAIL avance multi-niveau non refusee ou etat mute sans provider public\n");
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
          "bit-identique cross-rang, avance multi-niveau fail-closed)\n",
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
