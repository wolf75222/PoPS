// VALIDATION INTEGREE AmrSystem + MPI + GPU du transport compile :
//   - une HIERARCHIE AMR reelle (grossier replique + niveau fin multi-patch, regrid
//     Berger-Rigoutsos et reflux conservatif) ;
//   - un MODELE EULER PUR compile par add_compiled_model(AmrSystem, ...), sans source ni elliptique ;
//   - une DISTRIBUTION MPI : les patchs fins sont repartis sur n_ranks() GPU, avec halos cross-rang
//     via fill_boundary et reflux route vers le parent distant.
//
// Propriete verifiee : l'evolution de la hierarchie est INVARIANTE AU NOMBRE DE RANGS. Le grossier
// etant REPLIQUE (defaut AmrCouplerMP), density(), mass() et level_state_global(0) sont des
// grandeurs GLOBALES identiques sur chaque rang. Le decoupage du niveau fin entre rangs ne doit RIEN
// changer au resultat bit a bit. On controle la conservation, le layout et l'etat complet :
//   (1) CONSISTANCE CROSS-RANG dans le run : checksums de l'etat conservatif, densite, masse et
//       nombres de patchs ont une dispersion exactement nulle ;
//   (2) PARITE AU NB DE RANGS : le CTest rank-parity relance ce MEME binaire en np=1/2/4 et compare
//       exactement la signature canonique de l'etat et du layout. np=1 est l'oracle mono-rang.
//
// Ce test ne pretend pas qualifier un FAC composite distribue. Une preflight directe du provider
// builtin verifie au contraire qu'une requete composite partitionnee, valide en mono-rang, est
// refusee en MPI avec le diagnostic exact publie par le provider, sans registre ni fallback cache.
//
// Independant du backend : vert sous Kokkos Serial (CI, CPU) ET sous Kokkos Cuda (ROMEO GH200,
// multi-GPU). Sous Cuda, for_each_cell ne fence pas (async) : density()/mass() de l'AmrSystem font
// deja un device_fence() interne avant la lecture hote (read_coarse / amr_read_coarse), donc la
// lecture hote ici est sure. On insere malgre tout un Kokkos::fence() de ceinture avant les diffs.
#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include "gtest_compat.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/physics/bricks/bricks.hpp>  // CompositeModel, NoSource, NoElliptic
#include <pops/physics/fluids/euler.hpp>   // EulerND (transport compressible)
#include <pops/runtime/amr/exact_field_solver_provider.hpp>
#include <pops/runtime/amr/field_solver_options.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>  // add_compiled_model(AmrSystem, ...)
#include <pops/runtime/amr_system.hpp>

#include "amr_tagging_test_authority.hpp"
#include <pops/parallel/comm.hpp>  // comm_init, my_rank, n_ranks, all_reduce_*

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <string_view>
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
  return rho;
}

template <int Dim>
using Model = CompositeModel<EulerND<Dim>, NoSource, NoElliptic>;

constexpr double kGamma = 1.4;
constexpr const char* kStateRoute = "tests.mpi-amr-compiled-parity/gas/state@1";
constexpr const char* kConsumerQid = "tests.mpi-amr-compiled-parity/gas/physical-flux@1";
constexpr std::string_view kDistributedFacReason =
    "composite FAC distributed inter-level transfers are unavailable";

template <int Dim>
Model<Dim> transport_model() {
  return Model<Dim>{{}, EulerND<Dim>::prepare(Real(kGamma)), NoSource{}, NoElliptic{}};
}

template <int Dim>
std::vector<double> transport_state(const std::vector<double>& density) {
  using Gas = EulerND<Dim>;
  const std::size_t cells = density.size();
  std::vector<double> state(static_cast<std::size_t>(Gas::n_vars) * cells, 0.0);

  std::array<double, Dim> velocity{};
  double speed_squared = 0.0;
  for (int axis = 0; axis < Dim; ++axis) {
    velocity[static_cast<std::size_t>(axis)] = 0.08 * static_cast<double>(axis + 1);
    speed_squared +=
        velocity[static_cast<std::size_t>(axis)] * velocity[static_cast<std::size_t>(axis)];
  }

  for (std::size_t cell = 0; cell < cells; ++cell) {
    const double rho = density[cell];
    state[static_cast<std::size_t>(Gas::density_component) * cells + cell] = rho;
    for (int axis = 0; axis < Dim; ++axis)
      state[static_cast<std::size_t>(Gas::momentum_component(axis)) * cells + cell] =
          rho * velocity[static_cast<std::size_t>(axis)];
    state[static_cast<std::size_t>(Gas::energy_component) * cells + cell] =
        1.0 / (kGamma - 1.0) + 0.5 * rho * speed_squared;
  }
  return state;
}

template <int Dim>
EllipticBuildRequest<Dim> partitioned_fac_level(int cells, const mesh::RankSpace<Dim>& rank_space,
                                                const Index<Dim>& local_rank) {
  Extent<Dim> shape{};
  RealVector<Dim> lower{};
  RealVector<Dim> upper{};
  std::array<bool, Dim> periodic{};
  Extent<Dim> phi_ghosts{};
  for (int axis = 0; axis < Dim; ++axis) {
    shape[axis] = cells;
    upper[axis] = Real(1);
    periodic[static_cast<std::size_t>(axis)] = true;
    phi_ghosts[axis] = 1;
  }

  const Box<Dim> domain = Box<Dim>::from_extents(shape);
  const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(domain, lower, upper);
  const mesh::BoxArray<Dim> boxes{std::vector<Box<Dim>>{domain}};
  const mesh::Distribution<Dim> distribution =
      mesh::Distribution<Dim>::partitioned(boxes, rank_space, {rank_space.origin()});
  RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  return {geometry,
          boxes,
          distribution,
          local_rank,
          PhysicalBoundaryConditions<Dim>{BoundaryTopology<Dim>::axis_periodic(periodic), faces,
                                          spacing},
          Extent<Dim>{},
          phi_ghosts,
          {boxes.size(), 0}};
}

template <int Dim>
PreparedProviderSupport distributed_composite_fac_support(const ExecutionLane& lane) {
  Extent<Dim> rank_extent{};
  for (int axis = 0; axis < Dim; ++axis)
    rank_extent[axis] = 1;
  rank_extent[0] = lane.size();
  const mesh::RankSpace<Dim> rank_space{Index<Dim>{}, rank_extent};

  runtime::amr::ExactAmrFieldSolverBuildRequest<Dim> request;
  request.hierarchy.levels = {
      partitioned_fac_level<Dim>(4, rank_space, rank_space.coordinate(lane.rank())),
      partitioned_fac_level<Dim>(8, rank_space, rank_space.coordinate(lane.rank()))};
  std::array<int, Dim> ratio{};
  ratio.fill(2);
  request.hierarchy.ratios.emplace_back(ratio);
  request.mode = runtime::amr::ExactFieldHierarchyMode::composite;
  request.provider_options =
      geometric_mg_amr_field_solver_options(GeometricMgOptions{}, CompositeFacOptions{});
  request.use_contract = "tests.mpi-amr-compiled-parity/distributed-fac-preflight@1";
  request.spatial_contract = "tests.mpi-amr-compiled-parity/two-level-partitioned-layout@1";

  const auto provider = runtime::amr::make_builtin_exact_amr_field_solver_provider<Dim>();
  return provider->supports(request, lane);
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
  const int n = Dim >= 3 ? 32 : 64;

  AmrSystemConfig<Dim> cfg;
  for (int axis = 0; axis < Dim; ++axis) {
    cfg.shape[axis] = n;
    cfg.periodicity[axis] = true;
  }
  const std::vector<double> rho = four_bubbles(cfg.shape);
  const std::vector<double> initial_state = transport_state<Dim>(rho);
  cfg.regrid_every = 4;  // re-raffinement periodique : exerce le regrid distribue plusieurs fois

  // Preflight directe du provider exact-rank : la meme requete partitionnee est valide en serie,
  // puis refusee avant tout build en MPI. Aucun registre de providers ne peut choisir un fallback.
  const ExecutionLane fac_lane =
      ExecutionLane::world("tests.mpi-amr-compiled-parity/distributed-fac-preflight@1");
  const PreparedProviderSupport fac_support = distributed_composite_fac_support<Dim>(fac_lane);
  const bool fac_support_matches_contract =
      fac_support.well_formed() && (np == 1 ? fac_support.accepted()
                                            : (!fac_support.accepted() && fac_support.code == 4 &&
                                               fac_support.reason == kDistributedFacReason));
  const long fac_preflight_failed = all_reduce_max(fac_support_matches_contract ? 0L : 1L);

  // Modele Euler pur COMPILE branche sur la hierarchie AMR. Le transport recoit des identites
  // explicites et le Program n'appelle aucun solve de champ par defaut.
  AmrSystem<Dim> sys(cfg);
  test::install_amr_runtime_authority(sys, "test.mpi-amr-compiled-parity.runtime");
  sys.set_temporal_relations({2}, {1}, {"integral_only"});
  sys.install_block_state_route("gas", kStateRoute);
  add_compiled_model<Dim>(sys, "gas", transport_model<Dim>(), "minmod", "rusanov", "conservative",
                          "explicit", kGamma, 1, 1, {}, {}, 0.0, static_cast<double>(kWenoEpsilon),
                          false, kConsumerQid);
  test::install_prepared_threshold_union(
      sys, {{"gas", "rho", 1.2, test::PreparedThresholdRelation::Above, kStateRoute}});
  sys.set_conservative_state("gas", initial_state);
  test::install_forward_euler_program(sys, false);

  const double m0 = sys.mass("gas");  // declenche le build paresseux (regrid initial distribue)
  const int np0 = sys.n_patches();

  // Plusieurs macro-pas de transport AMR multi-niveaux avec reflux conservatif ; tous les 4 pas,
  // Berger-Rigoutsos redistribue les patchs. Aucun champ auxiliaire ou elliptique n'est installe.
  const double dt = 1e-3;
  const int nsteps = Dim >= 3 ? 2 : 16;
  for (int s = 0; s < nsteps; ++s)
    sys.step(dt);

#if defined(POPS_HAS_KOKKOS)
  Kokkos::fence();  // ceinture avant la lecture hote (density()/mass() fencent deja en interne)
#endif
  const std::vector<double> dens =
      sys.density("gas");  // grossier REPLIQUE : identique sur chaque rang
  const double mass = sys.mass("gas");
  const int npf = sys.n_patches();
  // Contract checkpoint MPI : le niveau 0 est replique. La vue d'etat nommee doit conserver le
  // layout composant-major et sa composante rho doit etre exactement la vue density("gas").
  const std::vector<double> state_global = sys.block_level_state_global("gas", 0);
  const std::size_t nn = cell_count(cfg.shape);
  const std::size_t expected_state_size = static_cast<std::size_t>(Model<Dim>::n_vars) * nn;
  double state_density_dmax = std::numeric_limits<double>::infinity();
  if (state_global.size() == expected_state_size && dens.size() == nn) {
    state_density_dmax = 0.0;
    for (std::size_t cell = 0; cell < nn; ++cell)
      state_density_dmax = std::fmax(
          state_density_dmax,
          std::fabs(
              state_global[static_cast<std::size_t>(EulerND<Dim>::density_component) * nn + cell] -
              dens[cell]));
  }
  const double density_evolution_dmax = max_abs_difference(dens, rho);
  const double state_evolution_dmax = max_abs_difference(state_global, initial_state);

  // Checksums de la densite ET de l'etat conservatif complet : signature bit-sensible du resultat
  // final, comparable entre nombres de rangs par l'orchestrateur rank-parity.
  double csum = 0, csumsq = 0, cmax = 0;
  double density_min = std::numeric_limits<double>::infinity();
  for (double v : dens) {
    csum += v;
    csumsq += v * v;
    density_min = std::fmin(density_min, v);
    const double a = std::fabs(v);
    if (a > cmax)
      cmax = a;
  }
  double state_sum = 0, state_sumsq = 0, state_absmax = 0;
  bool state_finite = true;
  for (double value : state_global) {
    state_sum += value;
    state_sumsq += value * value;
    state_absmax = std::fmax(state_absmax, std::fabs(value));
    state_finite = state_finite && std::isfinite(value);
  }
  double energy_min = std::numeric_limits<double>::infinity();
  if (state_global.size() == expected_state_size)
    for (std::size_t cell = 0; cell < nn; ++cell)
      energy_min = std::fmin(
          energy_min,
          state_global[static_cast<std::size_t>(EulerND<Dim>::energy_component) * nn + cell]);

  // (1) CONSISTANCE CROSS-RANG : les signatures de l'etat et de la densite, la masse, le layout et
  // les oracles de route/evolution doivent tous avoir une dispersion exactement nulle.
  const std::array<double, 12> parity_values{csum,
                                             csumsq,
                                             cmax,
                                             state_sum,
                                             state_sumsq,
                                             state_absmax,
                                             mass,
                                             static_cast<double>(np0),
                                             static_cast<double>(npf),
                                             state_density_dmax,
                                             density_evolution_dmax,
                                             state_evolution_dmax};
  double spread = 0.0;
  for (double value : parity_values)
    spread = std::fmax(spread, all_reduce_max(value) - (-all_reduce_max(-value)));

  int fails = 0;
  if (me == 0) {
    // Signature sans np : l'orchestrateur CTest generique exige exactement la meme ligne pour
    // chaque nombre de rangs declare dans tests/test_manifest.toml.
    std::printf(
        "POPS_MPI_PARITY_SIGNATURE_test_mpi_amr_compiled_parity "
        "patches0=%d patchesF=%d rho_sum=%.17e rho_sumsq=%.17e "
        "rho_absmax=%.17e state_sum=%.17e state_sumsq=%.17e state_absmax=%.17e "
        "state0_vs_density=%.17e\n",
        np0, npf, csum, csumsq, cmax, state_sum, state_sumsq, state_absmax, state_density_dmax);
    std::printf(
        "AMRMPI np=%d patches0=%d patchesF=%d | mass=%.17e | rho_sum=%.17e "
        "rho_sumsq=%.17e rho_absmax=%.17e | state_sum=%.17e state_sumsq=%.17e "
        "state_absmax=%.17e | crossrank_spread=%.3e state0_vs_density=%.3e\n",
        np, np0, npf, mass, csum, csumsq, cmax, state_sum, state_sumsq, state_absmax, spread,
        state_density_dmax);
    std::printf("AMRMPI FAC preflight np=%d accepted=%d code=%u reason=%.*s\n", np,
                fac_support.accepted() ? 1 : 0, static_cast<unsigned int>(fac_support.code),
                static_cast<int>(fac_support.reason.size()),
                fac_support.reason.empty() ? "" : fac_support.reason.data());
#if defined(POPS_HAS_KOKKOS)
    const char* space = Kokkos::DefaultExecutionSpace::name();
#else
    const char* space = "Serial(host)";
#endif
    std::printf("AMRMPI exec=%s m0=%.17e (conservation: dm=%.3e)\n", space, m0,
                std::fabs(mass - m0));

    if (fac_preflight_failed != 0) {
      std::printf(
          "FAIL preflight FAC distribuee: np1 doit accepter; np>1 doit refuser code=4, "
          "raison=%.*s\n",
          static_cast<int>(kDistributedFacReason.size()), kDistributedFacReason.data());
      ++fails;
    }
    if (!(dens.size() == nn)) {
      std::printf("FAIL densite grossiere de mauvaise taille\n");
      ++fails;
    }
    if (!(state_global.size() == expected_state_size && state_density_dmax == 0.0)) {
      std::printf("FAIL block_level_state_global(gas, 0) incoherent avec density(gas)\n");
      ++fails;
    }
    if (!(state_finite && std::isfinite(csum) && std::isfinite(csumsq))) {
      std::printf("FAIL etat conservatif non fini\n");
      ++fails;
    }
    if (!(density_min > 0.0 && energy_min > 0.0)) {
      std::printf("FAIL positivite rho/E perdue (rho_min=%.3e E_min=%.3e)\n", density_min,
                  energy_min);
      ++fails;
    }
    if (!(std::fabs(mass - m0) < 1e-9)) {
      std::printf("FAIL masse non conservee a travers transport/regrid/reflux\n");
      ++fails;
    }
    if (!(density_evolution_dmax > 1e-8 && state_evolution_dmax > 1e-8)) {
      std::printf("FAIL transport trivial (rho_dmax=%.3e state_dmax=%.3e)\n",
                  density_evolution_dmax, state_evolution_dmax);
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
    // Etat, masse et layout globaux DOIVENT etre bit-identiques sur tous les rangs.
    if (!(spread == 0.0)) {
      std::printf("FAIL etat/layout non bit-identique entre rangs\n");
      ++fails;
    }
    if (fails == 0)
      std::printf(
          "OK test_mpi_amr_compiled_parity np=%d (transport compile + regrid/reflux : etat, "
          "layout et masse bit-identiques cross-rang)\n",
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
