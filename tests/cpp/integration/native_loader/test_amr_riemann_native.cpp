// HLLC/Roe + reconstruction primitive sur le chemin "production" NATIF cote AMR (Gap 1 parite).
// Pendant de tests/cpp/integration/native_loader/test_amr_weno5_native.cpp : prouve que riemann=hllc/roe ET recon=primitive
// tournent la MEME hierarchie AMR (AmrCouplerMP<Model> + reflux + regrid), BIT-IDENTIQUE, sur les
// trois chemins, et que hllc/roe produisent un resultat DIFFERENT de rusanov (flux actif, non muet).
//
//   prepared exact-rank block == add_native_block(loader.so) -- deux routes d'installation
//   supportees et authentifiees, sur 4 combos (hllc/roe) x (conservative/primitive), dmax==0.
//   + rusanov conservatif (oracle) -> hllc != rusanov et roe != rusanov (flux actif).
//
// Le modele est un Euler PUR (CompositeModel<EulerND<Dim>, NoSource, BackgroundDensity{alpha=0}>) : la
// brique elliptique vaut 0, phi=0 (zero bruit FP), parite STRICTE. La pression est requise pour
// hllc/roe (cf. Euler::pressure()), d'ou le choix d'Euler.
//
// CMake injecte POPS_TEST_CXX, POPS_TEST_INCLUDE, POPS_TEST_CXX_STD, POPS_TEST_TMPDIR (meme pattern
// que test_amr_weno5_native).
#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "native_dso_compiler.hpp"
#include "explicit_amr_program.hpp"
#include "component_abi_test_helpers.hpp"
#include <pops/physics/bricks/bricks.hpp>  // CompositeModel, Euler, NoSource, BackgroundDensity
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/dynamic/authenticated_native_file.hpp>
#include <pops/runtime/dynamic/dynlib.hpp>
#include <pops/runtime/dynamic/prepared_execution_context.hpp>

#include "amr_tagging_test_authority.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

constexpr int kDim = kNativeDimension;
using NativeEuler = EulerND<kDim>;
using ProdModel = CompositeModel<NativeEuler, NoSource, BackgroundDensity>;
constexpr double kGamma = 1.4;
constexpr const char* kProviderConsumerQid = "tests.amr-riemann-native/physical-flux";
constexpr const char* kStateIdentity = "tests.amr-riemann-native/gas/state@1";
constexpr const char* kPrimaryClock = "test.clock.macro";
constexpr const char* kFineClock = "tests.amr-riemann-native/clock/level-1";
constexpr const char* kDsoRouteIdentity =
    "tests.amr-riemann-native/route/source-built-add-native-block@1";
constexpr const char* kModelIdentity =
    "1111111111111111111111111111111111111111111111111111111111111111";

std::shared_ptr<const component::PreparedExecutionContextV1> prepared_execution() {
  const PopsExecutionContextV1 execution = component::test_support::host_execution_context();
  return std::make_shared<const component::PreparedExecutionContextV1>(
      execution.execution_identity, execution.context_version, execution.memory_space,
      execution.backend_identity, execution.device_identity, execution.scalar_type,
      execution.storage_precision, execution.compute_precision, execution.accumulation_precision,
      execution.reduction_precision, execution.stream_handle, execution.stream_identity,
      execution.communicator_f_handle, execution.communicator_datatype_f_handle,
      execution.communicator_identity, execution.communicator_datatype_identity);
}

ProdModel make_model() {
  // alpha=0 : elliptic_rhs nul -> phi=0, parite stricte.
  return ProdModel{
      {}, NativeEuler{static_cast<Real>(kGamma)}, NoSource{}, BackgroundDensity{Real(0), Real(0)}};
}

template <int Dim>
std::size_t cell_count(int n) {
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    cells *= static_cast<std::size_t>(n);
  return cells;
}

template <int Dim>
std::vector<double> bubble_state(int n) {
  const std::size_t cells = cell_count<Dim>(n);
  std::vector<double> state(static_cast<std::size_t>(EulerND<Dim>::n_vars) * cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    std::size_t remaining = cell;
    double radius_squared = 0.0;
    for (int axis = 0; axis < Dim; ++axis) {
      const int coordinate = static_cast<int>(remaining % static_cast<std::size_t>(n));
      remaining /= static_cast<std::size_t>(n);
      const double position = (static_cast<double>(coordinate) + 0.5) / n - 0.5;
      radius_squared += position * position;
    }
    state[cell] = 1.0 + 0.5 * std::exp(-radius_squared / 0.02);
    state[static_cast<std::size_t>(EulerND<Dim>::energy_component) * cells + cell] = 2.5;
  }
  return state;
}

template <int Dim>
AmrSystemConfig<Dim> make_cfg(int n) {
  AmrSystemConfig<Dim> cfg;
  for (int axis = 0; axis < Dim; ++axis) {
    cfg.shape[axis] = n;
    cfg.lower[axis] = Real(0);
    cfg.upper[axis] = Real(1);
    cfg.periodicity[axis] = true;
    cfg.transition_buffers.front()[axis] = 1;
    cfg.transition_lookaheads.front()[axis] = 0;
  }
  cfg.level_count = 2;
  cfg.regrid_every = 4;
  return cfg;
}

template <int Dim>
void install_compiled_model(AmrSystem<Dim>& system, const char* riemann,
                            const char* reconstruction) {
  auto lane = std::make_shared<pops::ExecutionLane>(
      pops::ExecutionLane::duplicate_world_collectively("test.amr-riemann.direct-package"));
  auto execution = std::make_shared<const pops::component::PreparedExecutionContextV1>(
      prepared_execution()->for_lane(*lane));
  system.install_prepared_boundary_execution_context(std::move(lane), std::move(execution));
  system.install_block_state_route("gas", kStateIdentity);
  system.install_prepared_amr_block(prepare_compiled_amr_system_block<Dim>(
      "gas", make_model(), "minmod", riemann, reconstruction, "explicit", kGamma, 1, 1, 0.0,
      static_cast<double>(kWenoEpsilon), false, kProviderConsumerQid));
}

struct Snap {
  std::vector<double> density;
  double mass = 0;
  double initial_mass = 0;
  int n_patches = 0;
  int n_levels = 0;
  std::int64_t refined_cells = 0;
  std::int64_t fine_domain_cells = 0;
  int bootstrap_regrid_count = 0;
  int regrid_count = 0;
  int cadence_regrid_count = 0;
  std::int64_t accepted_primary_ticks = -1;
  std::int64_t accepted_fine_ticks = -1;
  std::int64_t accepted_coarse_steps = -1;
  std::int64_t accepted_fine_steps = -1;
  int coarse_flux_fragments = 0;
  int fine_flux_fragments = 0;
  int fine_phase_mask = 0;
  int reflux_syncs = 0;
  int average_down_syncs = 0;
  bool saw_coarse_dt = false;
  bool saw_fine_dt = false;
  bool temporal_ratio_is_two = false;
};

template <int Dim>
Snap run(AmrSystem<Dim>& s, int nsteps) {
  // Time subcycling is an independent execution contract. Keep it explicit even though this test
  // deliberately chooses the same value as the spatial refinement ratio.
  s.set_temporal_relations({2}, {1}, {"integral_only"});
  s.set_poisson("charge_density", "geometric_mg");
  test::install_prepared_threshold_union(s, {{"gas", "rho", 1.2}});
  auto context = test::install_forward_euler_program_context(s, true);
  context->declare_clock_relation(kPrimaryClock, kFineClock, 2);

  // Installing the explicit Program materializes the engine and performs the automatic initial
  // hierarchy bootstrap.  The cadence witness must therefore advance strictly after this baseline,
  // not merely be non-zero.
  const double initial_mass = s.mass();
  const int bootstrap_regrid_count = s.checkpoint_regrid_count();
  const double dt = 2e-4;
  for (int k = 0; k < nsteps; ++k)
    s.step(dt);

  Snap snap;
  snap.density = s.density();
  snap.mass = s.mass();
  snap.initial_mass = initial_mass;
  snap.n_patches = s.n_patches();
  snap.n_levels = s.n_levels();
  if (snap.n_levels >= 2) {
    for (const pops::Box<Dim>& patch : s.prepared_amr_block_state(0, 1).layout().boxes())
      snap.refined_cells += patch.numPts();
    snap.fine_domain_cells = s.prepared_amr_level_geometry(1).domain().numPts();
  }
  snap.bootstrap_regrid_count = bootstrap_regrid_count;
  snap.regrid_count = s.checkpoint_regrid_count();
  snap.cadence_regrid_count = snap.regrid_count - bootstrap_regrid_count;

  for (const auto& row : s.program_clock_manifest()) {
    if (row.size() == 3 && row[0] == "logical") {
      if (row[1] == kPrimaryClock)
        snap.accepted_primary_ticks = std::stoll(row[2]);
      else if (row[1] == kFineClock)
        snap.accepted_fine_ticks = std::stoll(row[2]);
    } else if (row.size() == 6 && row[0] == "level") {
      if (row[1] == "0")
        snap.accepted_coarse_steps = std::stoll(row[2]);
      else if (row[1] == "1")
        snap.accepted_fine_steps = std::stoll(row[2]);
    }
  }
  for (const auto& row : s.program_flux_ledger_manifest()) {
    if (row.size() != 13)
      continue;
    const bool coarse = row[10].ends_with("_coarse");
    const double duration = std::stod(row[12]);
    if (coarse) {
      ++snap.coarse_flux_fragments;
      snap.saw_coarse_dt = snap.saw_coarse_dt || std::fabs(duration - dt) < 1e-12;
    } else if (row[10].ends_with("_fine")) {
      ++snap.fine_flux_fragments;
      snap.saw_fine_dt = snap.saw_fine_dt || std::fabs(duration - dt / 2.0) < 1e-12;
      if (row[7] == "1" && row[6] == "0")
        snap.fine_phase_mask |= 1;
      if (row[7] == "2" && row[6] == "1")
        snap.fine_phase_mask |= 2;
    }
  }
  for (const auto& row : s.program_sync_manifest()) {
    if (row.size() != 7)
      continue;
    if (row[3] == "reflux")
      ++snap.reflux_syncs;
    else if (row[3] == "average_down")
      ++snap.average_down_syncs;
  }
  const auto temporal = s.checkpoint_temporal_relations();
  snap.temporal_ratio_is_two =
      temporal.size() == 1 &&
      temporal[0] == std::vector<std::string>({"0", "1", "2", "1", "integral_only"});
  return snap;
}

double maxdiff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0;
  for (std::size_t k = 0; k < a.size() && k < b.size(); ++k)
    d = std::fmax(d, std::fabs(a[k] - b[k]));
  return d;
}
double maxabs(const std::vector<double>& a) {
  double m = 0;
  for (double v : a)
    m = std::fmax(m, std::fabs(v));
  return m;
}

// Source du loader AMR : MEME forme que dsl.emit_cpp_native_loader(target="amr_system"), modele en dur.
std::string loader_source() {
  // Generated C++ source raw string: clang-format would reindent (or, with the
  // interleaved R"CPP( delimiters, runaway-indent) the inner content. Fence it to keep the
  // emitted source verbatim.
  // clang-format off
  return R"CPP(
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/config/route_ids.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/physics/bricks/bricks.hpp>
#include <stdexcept>
#include <string>
#include <utility>
namespace pops_generated {
constexpr int Dim = pops::kNativeDimension;
using ProdModel =
    pops::CompositeModel<pops::EulerND<Dim>, pops::NoSource, pops::BackgroundDensity>;
}
// LITTERAL preprocesseur (PAS abi_key_string() : une inline serait interposee, ELF/RTLD_GLOBAL,
// vers la copie du module deja charge -> cle du module renvoyee -> garde d'ABI tautologique).
extern "C" const char* pops_native_abi_key() { return POPS_ABI_KEY_LITERAL; }
extern "C" const char* pops_compiled_model_identity() {
  return "1111111111111111111111111111111111111111111111111111111111111111";
}
extern "C" const char* pops_compiled_route_manifest() { return pops::kRouteRegistrySignature; }
extern "C" int pops_compiled_nparams() { return 0; }
extern "C" const char* pops_compiled_param_names() { return ""; }
extern "C" const char* pops_test_amr_riemann_route_identity() {
  return "tests.amr-riemann-native/route/source-built-add-native-block@1";
}
extern "C" void pops_register_provider_routes_amr(
    pops::AmrSystem<pops::kNativeDimension>* system) {
  if (system == nullptr)
    throw std::invalid_argument("AMR provider route installer received null exact runtime");
}
extern "C" void pops_install_native_amr(void* sys, const char* name, const char* limiter,
                                       const char* riemann, const char* recon, const char* time,
                                       double gamma, int substeps, const double*, int,
                                       double pos_floor, double weno_epsilon,
                                       bool wave_speed_cache) {
  auto* s = reinterpret_cast<pops::AmrSystem<pops_generated::Dim>*>(sys);
  pops::PreparedNativeAmrPackage<pops_generated::Dim> package;
  package.block = pops::prepare_compiled_amr_system_block<pops_generated::Dim>(
      name,
      pops_generated::ProdModel{
          {},
          pops::EulerND<pops_generated::Dim>{static_cast<pops::Real>(gamma)}, pops::NoSource{},
          pops::BackgroundDensity{pops::Real(0), pops::Real(0)}},
      limiter, riemann, recon, time, gamma, substeps, 1, pos_floor, weno_epsilon,
      wave_speed_cache, "tests.amr-riemann-native/physical-flux");
  s->install_prepared_native_amr_package(std::move(package));
}
)CPP";
  // clang-format on
}

}  // namespace

static int pops_run_test_amr_riemann_native(int argc, char** argv) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int n = 64;
  // Exercise two cadence regrids (steps 4 and 8), then retain one complete accepted subcycling
  // witness on the live topology instead of sampling immediately after the step-12 invalidation.
  const int nsteps = 10;
  const std::vector<double> initial_state = bubble_state<kDim>(n);

  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // Calcule l'oracle rusanov conservatif (pour le NO-SILENT-FALLBACK hllc/roe != rusanov).
  std::vector<double> d_rusanov;
  {
    AmrSystem<kDim> Ref(make_cfg<kDim>(n));
    install_compiled_model(Ref, "rusanov", "conservative");
    Ref.set_conservative_state("gas", initial_state);
    d_rusanov = run(Ref, nsteps).density;
  }

  // Le second chemin est un vrai DSO source-built charge par add_native_block. Son symbole de test
  // authentifie l'origine; l'identite binaire lie ensuite cette image exacte a l'unique bloc publie
  // par la transaction native privee.
  const std::string tmp = std::string(POPS_TEST_TMPDIR) + "/amr_riemann_native_" +
                          std::to_string(static_cast<long>(std::clock()));
  const std::string src = tmp + ".cpp";
  const std::string so = tmp + ".so";
  {
    std::ofstream f(src);
    f << loader_source();
  }
  const auto package = pops::test::native_dso::compile_shared(src, so);
  if (!package.ok) {
    pops::test::native_dso::report_compile_failure("test_amr_riemann_native", package);
    return 1;
  }
  using RouteIdentityFn = const char* (*)();
  pops::dynlib::handle inspection{};
  RouteIdentityFn route_identity = nullptr;

  auto check_execution_evidence = [&](const Snap& snap, const char* riem, const char* recon,
                                      const char* route) {
    char w[200];
    std::snprintf(w, sizeof w, "[%s/%s/%s] hierarchie AMR et patches fins reels", riem, recon,
                  route);
    chk(snap.n_levels >= 2 && snap.n_patches > 0, w);
    std::snprintf(w, sizeof w, "[%s/%s/%s] interface coarse-fine strictement partielle", riem,
                  recon, route);
    chk(snap.refined_cells > 0 && snap.refined_cells < snap.fine_domain_cells, w);
    std::snprintf(w, sizeof w, "[%s/%s/%s] regrid cadence apres bootstrap", riem, recon, route);
    chk(snap.cadence_regrid_count > 0 && snap.regrid_count > snap.bootstrap_regrid_count, w);
    std::snprintf(w, sizeof w, "[%s/%s/%s] horloges acceptees 1:2", riem, recon, route);
    chk(snap.temporal_ratio_is_two && snap.accepted_primary_ticks == nsteps &&
            snap.accepted_fine_ticks == 2 * nsteps && snap.accepted_coarse_steps == nsteps &&
            snap.accepted_fine_steps == nsteps,
        w);
    std::snprintf(w, sizeof w, "[%s/%s/%s] deux sous-pas fins publies dans le ledger", riem, recon,
                  route);
    chk(snap.coarse_flux_fragments > 0 && snap.fine_flux_fragments > 0 &&
            snap.fine_phase_mask == 3 && snap.saw_coarse_dt && snap.saw_fine_dt,
        w);
    std::snprintf(w, sizeof w, "[%s/%s/%s] ledger consomme par reflux puis average-down", riem,
                  recon, route);
    chk(snap.reflux_syncs > 0 && snap.average_down_syncs > 0, w);
    std::snprintf(w, sizeof w, "[%s/%s/%s] conservation apres reflux/subcycling", riem, recon,
                  route);
    chk(std::fabs(snap.mass - snap.initial_mass) < 1e-11 * (std::fabs(snap.initial_mass) + 1.0), w);
  };

  auto parity_loader = [&](const char* riem, const char* recon) -> std::vector<double> {
    AmrSystem<kDim> loaded(make_cfg<kDim>(n));
    auto lane = std::make_shared<pops::ExecutionLane>(
        pops::ExecutionLane::duplicate_world_collectively("test.amr-riemann.package"));
    auto execution = std::make_shared<const pops::component::PreparedExecutionContextV1>(
        prepared_execution()->for_lane(*lane));
    loaded.install_prepared_boundary_execution_context(std::move(lane), std::move(execution));
    loaded.install_block_state_route("gas", kStateIdentity);
    const pops::dynlib::AuthenticatedNativeFile authenticated(so);
    loaded.add_native_block("gas", so, kModelIdentity, authenticated.binary_identity(), "minmod",
                            riem, recon, "explicit", kGamma, 1);
    if (!pops::dynlib::valid(inspection)) {
      inspection = pops::dynlib::open(so);
      route_identity = reinterpret_cast<RouteIdentityFn>(
          pops::dynlib::sym(inspection, "pops_test_amr_riemann_route_identity"));
    }
    char w[200];
    std::snprintf(w, sizeof w, "[%s/%s] DSO route identity authentifiee", riem, recon);
    chk(pops::dynlib::valid(inspection) && route_identity != nullptr &&
            std::strcmp(route_identity(), kDsoRouteIdentity) == 0,
        w);
    std::snprintf(w, sizeof w, "[%s/%s] DSO publie exactement un bloc prepare", riem, recon);
    chk(loaded.n_blocks() == 1, w);
    loaded.set_conservative_state("gas", initial_state);
    const Snap from_dso = run(loaded, nsteps);

    AmrSystem<kDim> direct(make_cfg<kDim>(n));
    install_compiled_model(direct, riem, recon);
    direct.set_conservative_state("gas", initial_state);
    const Snap from_direct = run(direct, nsteps);
    const double dmax = maxdiff(from_dso.density, from_direct.density);
    std::snprintf(w, sizeof w, "[%s/%s] add_native_block == prepare/install direct (dmax==0)", riem,
                  recon);
    chk(dmax == 0.0, w);
    std::snprintf(w, sizeof w, "[%s/%s] densite non triviale", riem, recon);
    chk(maxabs(from_direct.density) > 1e-6, w);
    std::snprintf(w, sizeof w, "[%s/%s] masse et topologie identiques", riem, recon);
    chk(std::fabs(from_dso.mass - from_direct.mass) < 1e-12 * (std::fabs(from_direct.mass) + 1.0) &&
            from_dso.n_patches == from_direct.n_patches &&
            from_dso.n_levels == from_direct.n_levels &&
            from_dso.regrid_count == from_direct.regrid_count &&
            from_dso.cadence_regrid_count == from_direct.cadence_regrid_count,
        w);
    std::snprintf(w, sizeof w, "[%s/%s] preuves acceptees de subcycling/reflux identiques", riem,
                  recon);
    chk(from_dso.accepted_primary_ticks == from_direct.accepted_primary_ticks &&
            from_dso.accepted_fine_ticks == from_direct.accepted_fine_ticks &&
            from_dso.coarse_flux_fragments == from_direct.coarse_flux_fragments &&
            from_dso.fine_flux_fragments == from_direct.fine_flux_fragments &&
            from_dso.fine_phase_mask == from_direct.fine_phase_mask &&
            from_dso.reflux_syncs == from_direct.reflux_syncs &&
            from_dso.average_down_syncs == from_direct.average_down_syncs,
        w);
    check_execution_evidence(from_dso, riem, recon, "dso");
    check_execution_evidence(from_direct, riem, recon, "direct");
    std::printf("OK  [%s/%s] dmax=%.0f\n", riem, recon, dmax);
    return from_direct.density;
  };

  const std::vector<double> d_hllc_cons = parity_loader("hllc", "conservative");
  (void)parity_loader("hllc", "primitive");
  const std::vector<double> d_roe_cons = parity_loader("roe", "conservative");
  (void)parity_loader("roe", "primitive");

  // NO-SILENT-FALLBACK : hllc et roe doivent differer de rusanov sur ce meme etat.
  chk(maxdiff(d_hllc_cons, d_rusanov) > 1e-12,
      "hllc != rusanov (le flux hllc est actif, non silencieux)");
  chk(maxdiff(d_roe_cons, d_rusanov) > 1e-12,
      "roe != rusanov (le flux roe est actif, non silencieux)");

  pops::dynlib::close(inspection);
  std::remove(src.c_str());
  std::remove(so.c_str());

  if (fails == 0)
    std::printf(
        "OK test_amr_riemann_native (hllc/roe x conservative/primitive : "
        "DSO add_native_block == prepare/install direct, routes authentifiees, bit-identique ; "
        "AMR 2 niveaux, regrid post-bootstrap, subcycling/reflux acceptes ; hllc/roe actifs vs "
        "rusanov)\n");
  return fails ? 1 : 0;
}

TEST(test_amr_riemann_native, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_amr_riemann_native, "test_amr_riemann_native"),
            0);
}
