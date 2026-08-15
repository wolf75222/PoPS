// PARITE MPI du REGRID D'UNION DES TAGS multi-blocs (T4 du design
// docs/AMR_REGRID_UNION_TAGS_DESIGN.md, suivi #199). C'est le verrou de parite cross-rang manquant :
// le regrid d'union reduit les tags cross-rang par all_reduce_or_inplace (etape R4) AVANT le
// clustering Berger-Rigoutsos, de sorte que TOUS les rangs partent de la MEME grille de tags et
// produisent EXACTEMENT le meme BoxArray fin -> meme DistributionMapping -> hierarchie IDENTIQUE quel
// que soit le nombre de rangs. Si la reduction (R4) etait omise ou buguee, deux rangs partiraient de
// grilles de tags differentes, le clustering divergerait par rang et MPI desynchroniserait (risques
// X1/X2 du design).
//
// SCENARIO (le MEME a np=1/2/4) : deux blocs ExB alimentes par un champ electrique et magnetique
// explicitement prepare, un blob a gauche et un a droite, sur une hierarchie 2 niveaux. GROSSIER
// REPARTI (distribute_coarse=true,
// BoxArray multi-box round-robin) : c'est le seul chemin ou (R4) est active (en grossier REPLIQUE,
// chaque rang a deja la grille de tags complete, all_reduce_or serait l'identite). regrid_every=2 :
// la grille se re-grille effectivement pendant la sequence, en suivant l'union des tags densite par
// bloc, installes dans un seul graphe prepare. On avance
// plusieurs macro-pas (donc plusieurs regrids), puis on observe la hierarchie finale.
//
// ASSERTIONS :
//   (1) CONSISTANCE CROSS-RANG (dans CHAQUE run) : la densite grossiere de chaque bloc est reconstruite
//       GLOBALEMENT (all_reduce des boites disjointes du grossier reparti), donc n*n sur chaque rang ;
//       son checksum et n_patches sont des grandeurs GLOBALES -> spread max cross-rang == 0
//       (insensible a l'ordre via all_reduce_max). Un bug de halo / layout fin divergent le casserait.
//   (2) PARITE AU NB DE RANGS : on imprime n_patches + les checksums des deux densites ;
//       la CI relance le MEME binaire en np=1/2/4 et DIFFE la ligne AMRREGRID (np=1 = oracle ;
//       np=2/4 doivent etre BIT-IDENTIQUES). Le n_patches identique cross-np = layout fin identique
//       cross-np (LE point du regrid d'union : un seul fb/dmap pour tous les rangs).
//   (3) CONSERVATION PAR BLOC a travers les regrids : la masse de chaque bloc est conservee (reflux +
//       report fin exact + interp parent piecewise-constant conservatif au sens integral).
//
// Independant du backend (Kokkos Serial CI, Kokkos Cuda GH200). Compile le runtime AmrSystem comme
// test_mpi_amr_twoblock_parity (avec python/amr_system.cpp).
#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include "gtest_compat.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/physics/bricks/bricks.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/parallel/comm.hpp>  // comm_init, my_rank, n_ranks, all_reduce_*

#include "test_harness.hpp"  // pops::test::checksum (somme des carres partagee)
#include "amr_tagging_test_authority.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

constexpr std::array<const char*, 2> kStateRoutes{"tests.amr-regrid-mpi-parity/a/state@1",
                                                  "tests.amr-regrid-mpi-parity/b/state@1"};
constexpr std::array<const char*, 2> kConsumerQids{"tests.amr-regrid-mpi-parity/a/physical-flux@1",
                                                   "tests.amr-regrid-mpi-parity/b/physical-flux@1"};

template <int Dim>
using ExbTransportModel = CompositeModel<CartesianExBDriftND<Dim>, NoSource, NoElliptic>;

template <int Dim>
ExbTransportModel<Dim> exb_transport() {
  return {};
}

template <int Dim>
std::vector<runtime::system::AuxiliaryComponentKey> install_exb_auxiliary_authority(
    AmrSystem<Dim>& system) {
  using namespace runtime::system;
  const AuxiliaryComponentContract electric_contract{"cell-average", "cell", "unitless",
                                                     "tests-exb-electric-input", "scalar"};
  const AuxiliaryComponentContract magnetic_contract{"cell-average", "cell", "tesla",
                                                     "tests-exb-magnetic-input", "scalar"};
  AuxiliaryStorageShape<Dim> shape;
  for (int axis = 0; axis < Dim; ++axis)
    shape.halo[axis] = 2;

  std::vector<AuxiliaryComponentKey> keys;
  std::vector<AuxiliaryOutput<Dim>> outputs;
  keys.reserve(static_cast<std::size_t>(Dim + 3));
  outputs.reserve(static_cast<std::size_t>(Dim + 3));
  for (int axis = 0; axis < Dim; ++axis) {
    AuxiliaryComponentKey key{"tests.amr-regrid-mpi-parity", "input", "electric",
                              "gradient-" + std::to_string(axis)};
    keys.push_back(key);
    outputs.push_back({std::move(key), electric_contract, shape});
  }
  for (int component = 0; component < 3; ++component) {
    AuxiliaryComponentKey key{"tests.amr-regrid-mpi-parity", "input", "magnetic",
                              "B-" + std::to_string(component)};
    keys.push_back(key);
    outputs.push_back({std::move(key), magnetic_contract, shape});
  }
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<Dim>{
      "tests.amr-regrid-mpi-parity/exb-input@1",
      AuxiliaryProviderKind::input,
      {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
      std::move(outputs),
      {}});

  for (const char* consumer_qid : kConsumerQids) {
    AuxiliaryConsumerProviderPlan<Dim> plan;
    plan.consumer_qid = consumer_qid;
    plan.values.reserve(static_cast<std::size_t>(Dim + 3));
    for (int axis = 0; axis < Dim; ++axis)
      plan.values.push_back({{keys[static_cast<std::size_t>(axis)], electric_contract, shape},
                             static_cast<std::size_t>(axis)});
    for (std::size_t component = 0; component < 3; ++component)
      plan.values.push_back(
          {{keys[static_cast<std::size_t>(Dim) + component], magnetic_contract, shape},
           static_cast<std::size_t>(Dim) + component});
    system.install_auxiliary_consumer_plan(std::move(plan));
  }
  system.seal_auxiliary_providers();
  return keys;
}

}  // namespace

// Bulle gaussienne centree sur l'axe 0 (et au milieu des autres axes), aplatie dans l'ordre natif.
// Le maximum (base + amp) depasse le seuil de raffinement -> la region taguee suit le blob (regrid).
template <int Dim>
std::size_t cell_count(const Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

template <int Dim>
static std::vector<double> blob(const Extent<Dim>& shape, double x_center, double amp, double base,
                                double width) {
  std::vector<double> rho(cell_count(shape), base);
  for (std::size_t ordinal = 0; ordinal < rho.size(); ++ordinal) {
    std::size_t remainder = ordinal;
    double radius_squared = 0.0;
    for (int axis = 0; axis < Dim; ++axis) {
      const auto extent = static_cast<std::size_t>(shape[axis]);
      const double coordinate =
          (static_cast<double>(remainder % extent) + 0.5) / static_cast<double>(extent);
      remainder /= extent;
      const double center = axis == 0 ? x_center : 0.5;
      const double delta = coordinate - center;
      radius_squared += delta * delta;
    }
    rho[ordinal] = base + amp * std::exp(-radius_squared / (width * width));
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
  const int n = 32;
  constexpr int Dim = kNativeDimension;
  const double B0 = 1.0;

  AmrSystemConfig<Dim> cfg;
  for (int axis = 0; axis < Dim; ++axis) {
    cfg.shape[axis] = n;
    cfg.periodicity[axis] = true;
    cfg.coarse_max_grid[axis] = n / 2;
  }
  const mesh::BoxArray<Dim> coarse_layout =
      mesh::BoxArray<Dim>::from_domain(cfg.index_domain(), cfg.coarse_max_grid);
  cfg.boxes = coarse_layout.boxes();
  const std::vector<double> rho0 = blob(cfg.shape, 0.30, 1.0, 1.0, 0.07);  // bloc a gauche
  const std::vector<double> rho1 = blob(cfg.shape, 0.70, 1.0, 1.0, 0.07);  // bloc a droite
  cfg.regrid_every = 2;          // REGRID ACTIF : la hierarchie se re-grille pendant la sequence
  cfg.distribute_coarse = true;  // GROSSIER REPARTI : active la reduction collective des tags (R4)
  // n/2 par axe -> 2^Dim boites grossieres reparties round-robin entre les rangs.

  AmrSystem<Dim> sys(cfg);
  test::install_amr_runtime_authority(sys, "tests.amr-regrid-mpi-parity/runtime-instance@1");
  sys.set_temporal_relations({2}, {1}, {"integral_only"});
  sys.install_block_state_route("a", kStateRoutes[0]);
  sys.install_block_state_route("b", kStateRoutes[1]);
  add_compiled_model<Dim>(sys, "a", exb_transport<Dim>(), "minmod", "rusanov", "conservative",
                          "explicit", static_cast<double>(kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
                          static_cast<double>(kWenoEpsilon), false, kConsumerQids[0]);
  add_compiled_model<Dim>(sys, "b", exb_transport<Dim>(), "minmod", "rusanov", "conservative",
                          "explicit", static_cast<double>(kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
                          static_cast<double>(kWenoEpsilon), false, kConsumerQids[1]);
  const auto auxiliary_keys = install_exb_auxiliary_authority(sys);
  const std::vector<double> zero_input(cell_count(cfg.shape), 0.0);
  const std::vector<double> unit_input(cell_count(cfg.shape), 1.0);
  for (int axis = 0; axis < Dim; ++axis)
    sys.stage_auxiliary_input(auxiliary_keys[static_cast<std::size_t>(axis)],
                              axis == 0 ? unit_input : zero_input);
  for (int component = 0; component < 3; ++component)
    sys.stage_auxiliary_input(
        auxiliary_keys[static_cast<std::size_t>(Dim + component)],
        component == 2 ? std::vector<double>(cell_count(cfg.shape), B0) : zero_input);
  test::install_prepared_threshold_union(
      sys, {{"a", "n", 1.5, test::PreparedThresholdRelation::Above, kStateRoutes[0]},
            {"b", "n", 1.5, test::PreparedThresholdRelation::Above, kStateRoutes[1]}});
  sys.set_density("a", rho0);
  sys.set_density("b", rho1);
  test::install_forward_euler_program(sys, false);

  const double m0a = sys.mass("a");  // declenche le build paresseux
  const double m0b = sys.mass("b");
  EXPECT_NE(sys.engine(), nullptr);
  const int coarse_local = sys.coarse_local_boxes();
  const int coarse_total = sys.coarse_total_boxes();

  const double dt = 1e-3;
  for (int s = 0; s < 16; ++s)
    sys.step(dt);  // 16 macro-pas, regrid tous les 2 -> plusieurs regrids

#if defined(POPS_HAS_KOKKOS)
  Kokkos::fence();
#endif
  const std::vector<double> da = sys.density("a");
  const std::vector<double> db = sys.density("b");
  const double ma = sys.mass("a"), mb = sys.mass("b");
  const int npatch = sys.n_patches();  // nombre de patchs fins = signature du layout fin d'union

  using pops::test::checksum;  // somme des carres partagee (signature deterministe d'un champ)
  const double ca = checksum(da), cb = checksum(db);

  // (1) CONSISTANCE CROSS-RANG : densite reconstruite globalement + n_patches sont des
  // grandeurs GLOBALES identiques sur tout rang. spread = max - min cross-rang (insensible a l'ordre).
  auto spread = [](double x) { return all_reduce_max(x) - (-all_reduce_max(-x)); };
  const double sp = std::fmax(
      std::fmax(spread(ca), spread(cb)),
      std::fmax(spread(ma),
                std::fmax(spread(mb), std::fmax(spread(static_cast<double>(npatch)),
                                                spread(static_cast<double>(coarse_total))))));
  const double max_coarse_local = all_reduce_max(static_cast<double>(coarse_local));

  int fails = 0;
  if (me == 0) {
    // Ligne PARITE (diffee cross-np par la CI) : n_patches + checksums imprimes en %.17e bit-exact.
    std::printf(
        "AMRREGRID np=%d | coarse_boxes=%d n_patches=%d | csum_a=%.17e csum_b=%.17e | "
        "crossrank_spread=%.3e\n",
        np, coarse_total, npatch, ca, cb, sp);
    std::printf("AMRREGRID conservation: dm_a=%.3e dm_b=%.3e | mass_a=%.17e mass_b=%.17e\n",
                std::fabs(ma - m0a), std::fabs(mb - m0b), ma, mb);

    if (!(da.size() == cell_count(cfg.shape))) {
      std::printf("FAIL taille densite (%zu != %zu)\n", da.size(), cell_count(cfg.shape));
      ++fails;
    }
    if (!(npatch >= 1)) {
      std::printf("FAIL aucun patch fin (le regrid n'a pas raffine)\n");
      ++fails;
    }
    const int expected_coarse_boxes = 1 << Dim;
    if (!(coarse_total == expected_coarse_boxes &&
          (np == 1 ? coarse_local == coarse_total : max_coarse_local < coarse_total))) {
      std::printf(
          "FAIL grossier non distribue (local=%d max_local=%.0f total=%d expected=%d np=%d)\n",
          coarse_local, max_coarse_local, coarse_total, expected_coarse_boxes, np);
      ++fails;
    }
    if (!std::isfinite(ca) || !std::isfinite(cb)) {
      std::printf("FAIL champ non fini (transport / regrid casse ?)\n");
      ++fails;
    }
    // (3) masse de CHAQUE bloc conservee a travers les regrids (report fin exact + interp parent
    // piecewise-constant conservatif au sens integral + reflux conservatif).
    if (!(std::fabs(ma - m0a) < 1e-9)) {
      std::printf("FAIL masse bloc a non conservee a travers le regrid\n");
      ++fails;
    }
    if (!(std::fabs(mb - m0b) < 1e-9)) {
      std::printf("FAIL masse bloc b non conservee a travers le regrid\n");
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
          "cross-rang, masse par bloc conservee ; CI diffe np=1/2/4)\n",
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
