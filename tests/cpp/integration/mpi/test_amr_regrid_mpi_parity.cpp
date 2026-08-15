// Historical design context follows.  The exact-ranked runtime currently accepts one prepared
// compiled block; this test retains distributed regrid parity and verifies a fail-closed advance.
// PARITE MPI du REGRID D'UNION DES TAGS multi-blocs (T4 du design
// docs/AMR_REGRID_UNION_TAGS_DESIGN.md, suivi #199). C'est le verrou de parite cross-rang manquant :
// le regrid d'union reduit les tags cross-rang par all_reduce_or_inplace (etape R4) AVANT le
// clustering Berger-Rigoutsos, de sorte que TOUS les rangs partent de la MEME grille de tags et
// produisent EXACTEMENT le meme BoxArray fin -> meme DistributionMapping -> hierarchie IDENTIQUE quel
// que soit le nombre de rangs. Si la reduction (R4) etait omise ou buguee, deux rangs partiraient de
// grilles de tags differentes, le clustering divergerait par rang et MPI desynchroniserait (risques
// X1/X2 du design).
//
// SCENARIO (le MEME a np=1/2/4) : deux blocs ExB a charges opposees (Poisson de systeme somme), un
// blob a gauche et un a droite, sur une hierarchie 2 niveaux. GROSSIER REPARTI (distribute_coarse=true,
// BoxArray multi-box round-robin) : c'est le seul chemin ou (R4) est active (en grossier REPLIQUE,
// chaque rang a deja la grille de tags complete, all_reduce_or serait l'identite). regrid_every=2 :
// la grille se re-grille effectivement pendant la sequence, en suivant l'union des tags densite par
// bloc + le tag de phi sur |grad phi|, tous installes dans un seul graphe prepare. On avance
// plusieurs macro-pas (donc plusieurs regrids), puis on observe la hierarchie finale.
//
// ASSERTIONS :
//   (1) CONSISTANCE CROSS-RANG (dans CHAQUE run) : la densite grossiere de chaque bloc est reconstruite
//       GLOBALEMENT (all_reduce des boites disjointes du grossier reparti), donc n*n sur chaque rang ;
//       son checksum, le potentiel de systeme et n_patches sont des grandeurs GLOBALES -> spread max
//       cross-rang == 0 (insensible a l'ordre via all_reduce_max). Un bug de halo / Poisson somme /
//       layout fin divergent le casserait.
//   (2) PARITE AU NB DE RANGS : on imprime n_patches + des checksums (densite par bloc + potentiel) ;
//       la CI relance le MEME binaire en np=1/2/4 et DIFFE la ligne AMRREGRID (np=1 = oracle ;
//       np=2/4 doivent etre BIT-IDENTIQUES). Le n_patches identique cross-np = layout fin identique
//       cross-np (LE point du regrid d'union : un seul fb/dmap pour tous les rangs).
//   (3) CONSERVATION PAR BLOC a travers les regrids : la masse de chaque bloc est conservee (reflux +
//       report fin exact + interp parent piecewise-constant conservatif au sens integral).
//
// Independant du backend (Kokkos Serial CI, Kokkos Cuda GH200). Compile le runtime exact-ranke comme
// test_mpi_amr_twoblock_parity (avec python/amr_system.cpp).
#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include "gtest_compat.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/parallel/comm.hpp>  // comm_init, my_rank, n_ranks, all_reduce_*

#include "test_harness.hpp"  // pops::test::checksum (somme des carres partagee)
#include "amr_tagging_test_authority.hpp"

#include <cmath>
#include <cstdio>
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
    return {"test.amr-regrid-mpi-parity.scalar-advection", 1};
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

// Bump isotrope sur le domaine unitaire exact-ranke. Son maximum active le regrid prepare.
template <int Dim>
static std::size_t cell_count(const Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

template <int Dim>
static std::vector<double> blob(const Extent<Dim>& shape, double center, double amp, double base,
                                double width) {
  std::vector<double> rho(cell_count(shape), base);
  for (std::size_t linear = 0; linear < rho.size(); ++linear) {
    std::size_t remainder = linear;
    double radius_squared = 0.0;
    for (int axis = 0; axis < Dim; ++axis) {
      const auto extent = static_cast<std::size_t>(shape[axis]);
      const double coordinate =
          (static_cast<double>(remainder % extent) + 0.5) / static_cast<double>(extent);
      remainder /= extent;
      const double displacement = coordinate - center;
      radius_squared += displacement * displacement;
    }
    rho[linear] = base + amp * std::exp(-radius_squared / (width * width));
  }
  return rho;
}

static int pops_run_test_amr_regrid_mpi_parity(int argc, char** argv) {
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
  for (int axis = 0; axis < Dim; ++axis) {
    cfg.shape[axis] = 32;
    cfg.coarse_max_grid[axis] = 16;
  }
  cfg.regrid_every = 2;          // REGRID ACTIF : la hierarchie se re-grille pendant la sequence
  cfg.distribute_coarse = true;  // GROSSIER REPARTI : active la reduction collective des tags (R4)
  const std::vector<double> rho = blob(cfg.shape, 0.30, 1.0, 1.0, 0.07);

  AmrSystem<Dim> sys(cfg);
  sys.set_temporal_relations({2}, {1}, {"integral_only"});
  sys.install_block_state_route("tracer", "test.amr-regrid-mpi-parity/state/tracer");
  add_compiled_model<Dim>(sys, "tracer", advection_model<Dim>(), "minmod", "rusanov",
                          "conservative", "explicit", /*gamma=*/1.4, /*substeps=*/1,
                          /*stride=*/1, {}, {}, /*positivity_floor=*/0.0,
                          static_cast<double>(kWenoEpsilon), /*wave_speed_cache=*/false,
                          "test.amr-regrid-mpi-parity.tracer");
  test::install_prepared_threshold_union(sys, {{"tracer", "u", 1.5}});
  sys.set_density("tracer", rho);
  test::install_forward_euler_program(sys);

  const double m0 = sys.mass("tracer");  // declenche le build paresseux
  EXPECT_NE(sys.engine(), nullptr);

  const std::vector<double> before = sys.density("tracer");
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
  Kokkos::fence();
#endif
  const std::vector<double> density = sys.density("tracer");
  const double mass = sys.mass("tracer");
  const int npatch = sys.n_patches();  // nombre de patchs fins = signature du layout fin d'union
  const bool no_mutation = density == before && sys.patch_boxes() == before_patches &&
                           sys.n_levels() == before_levels && mass == m0;

  using pops::test::checksum;  // somme des carres partagee (signature deterministe d'un champ)
  const double csum = checksum(density);

  // (1) CONSISTANCE CROSS-RANG : densite reconstruite globalement + potentiel + n_patches sont des
  // grandeurs GLOBALES identiques sur tout rang. spread = max - min cross-rang (insensible a l'ordre).
  auto spread = [](double x) { return all_reduce_max(x) - (-all_reduce_max(-x)); };
  const double sp = std::fmax(std::fmax(spread(csum), spread(mass)),
                              spread(static_cast<double>(npatch)));

  int fails = 0;
  if (me == 0) {
    // Ligne PARITE (diffee cross-np par la CI) : n_patches + checksums imprimes en %.17e bit-exact.
    std::printf(
        "AMRREGRID np=%d | n_patches=%d | csum=%.17e mass=%.17e | "
        "crossrank_spread=%.3e\n",
        np, npatch, csum, mass, sp);
    std::printf("AMRREGRID fail-closed: dm=%.3e | mass=%.17e\n", std::fabs(mass - m0), mass);

    if (!(density.size() == cell_count(cfg.shape))) {
      std::printf("FAIL taille densite (%zu != %zu)\n", density.size(), cell_count(cfg.shape));
      ++fails;
    }
    if (!(npatch >= 1)) {
      std::printf("FAIL aucun patch fin (le regrid n'a pas raffine)\n");
      ++fails;
    }
    if (!std::isfinite(csum) || !std::isfinite(mass)) {
      std::printf("FAIL champ non fini (regrid casse ?)\n");
      ++fails;
    }
    if (!advance_refused || !no_mutation) {
      std::printf("FAIL avance multi-niveau non refusee ou etat mute sans provider public\n");
      ++fails;
    }
    // (1) grossier reparti reconstruit GLOBALEMENT + layout fin d'union UNIQUE -> tout bit-identique
    // cross-rang (spread exactement 0). Le n_patches dans le spread = meme layout fin sur tous les rangs.
    if (!(sp == 0.0)) {
      std::printf(
          "FAIL grandeurs non bit-identiques entre rangs (spread=%.3e) : la reduction des tags "
          "(R4) ou le layout fin d'union diverge par rang\n",
          sp);
      ++fails;
    }
    if (fails == 0)
      std::printf(
          "OK test_amr_regrid_mpi_parity np=%d (regrid d'union : layout fin IDENTIQUE "
          "cross-rang, avance multi-niveau fail-closed sans mutation ; CI diffe np=1/2/4)\n",
          np);
  } else {
    (void)sp;
  }
  comm_finalize();
  return fails ? 1 : 0;
}

TEST(test_amr_regrid_mpi_parity, Runs) {
  EXPECT_EQ(
      pops::test::RunTestBody(&pops_run_test_amr_regrid_mpi_parity, "test_amr_regrid_mpi_parity"),
      0);
}
