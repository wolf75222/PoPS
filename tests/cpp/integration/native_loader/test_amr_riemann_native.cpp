// HLLC/Roe + reconstruction primitive sur le chemin "production" NATIF cote AMR (Gap 1 parite).
// Pendant de tests/cpp/integration/native_loader/test_amr_weno5_native.cpp : prouve que riemann=hllc/roe ET recon=primitive
// tournent le MEME niveau AMR exact-rank, BIT-IDENTIQUE, sur les
// trois chemins, et que hllc/roe produisent un resultat DIFFERENT de rusanov (flux actif, non muet).
//
//   (A) add_compiled_model(AmrSystem&) == prepare+install explicite -- direct, sans .so : tourne sous
//       TOUS les backends (hote, Kokkos Serial), c'est la parite decisive qui ne casse pas nvcc.
//       4 combos : (hllc/roe) x (conservative/primitive), chacun dmax==0.
//       + rusanov conservatif (oracle) -> hllc != rusanov et roe != rusanov (flux actif).
//
//   (B) add_native_block(loader.so) == add_compiled_model(AmrSystem&) -- chemin .so, autocompile a
//       l'execution (source du loader AMR, cf. dsl.emit_cpp_native_loader(target="amr_system")).
//       Le DSO rejoue le contrat exact du backend hote, Kokkos inclus ; aucun auto-skip.
//
// Le modele est un Euler PUR (CompositeModel<Euler, NoSource, BackgroundDensity{alpha=0}>) : la
// brique elliptique vaut 0, phi=0 (zero bruit FP), parite STRICTE. La pression est requise pour
// hllc/roe (cf. Euler::pressure()), d'ou le choix d'Euler.
//
// CMake injecte POPS_TEST_CXX, POPS_TEST_INCLUDE, POPS_TEST_CXX_STD, POPS_TEST_TMPDIR (meme pattern
// que test_amr_weno5_native).
#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "native_dso_compiler.hpp"
#include "explicit_amr_program.hpp"
#include <pops/physics/bricks/bricks.hpp>  // CompositeModel, EulerND, NoSource, BackgroundDensity
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/amr_system.hpp>

#include <cmath>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

constexpr int Dim = kNativeDimension;
constexpr const char* kProviderConsumerQid = "tests.amr-riemann.providers/gas";
using NativeAmrSystem = AmrSystem<Dim>;
using NativeAmrSystemConfig = AmrSystemConfig<Dim>;
using ProdModel = CompositeModel<EulerND<Dim>, NoSource, BackgroundDensity>;
using ProdProviderValues = ProviderValues<provider_count<ProdModel>()>;
constexpr double kGamma = 1.4;

static_assert(ProdProviderValues::size == provider_count<ProdModel>());

ProdModel make_model() {
  // alpha=0 : elliptic_rhs nul -> phi=0, parite stricte.
  return ProdModel{
      {}, EulerND<Dim>{static_cast<Real>(kGamma)}, NoSource{}, BackgroundDensity{Real(0), Real(0)}};
}

std::vector<double> bubble(int n) {
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    cells *= static_cast<std::size_t>(n);
  std::vector<double> rho(cells);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    std::size_t quotient = cell;
    double radius_squared = 0.0;
    for (int axis = 0; axis < Dim; ++axis) {
      const int coordinate = static_cast<int>(quotient % static_cast<std::size_t>(n));
      quotient /= static_cast<std::size_t>(n);
      const double x = (static_cast<double>(coordinate) + 0.5) / n - 0.5;
      radius_squared += x * x;
    }
    rho[cell] = 1.0 + 0.5 * std::exp(-radius_squared / 0.02);
  }
  return rho;
}

std::vector<double> conservative_state(const std::vector<double>& density) {
  const std::size_t cells = density.size();
  std::vector<double> state(static_cast<std::size_t>(ProdModel::n_vars) * cells, 0.0);
  double velocity_squared = 0.0;
  for (int axis = 0; axis < Dim; ++axis) {
    const double velocity = 0.05 * static_cast<double>(axis + 1);
    velocity_squared += velocity * velocity;
    for (std::size_t cell = 0; cell < cells; ++cell)
      state[static_cast<std::size_t>(axis + 1) * cells + cell] = density[cell] * velocity;
  }
  for (std::size_t cell = 0; cell < cells; ++cell) {
    state[cell] = density[cell];
    state[static_cast<std::size_t>(Dim + 1) * cells + cell] =
        1.0 / (kGamma - 1.0) + 0.5 * density[cell] * velocity_squared;
  }
  return state;
}

NativeAmrSystemConfig make_cfg(int n) {
  NativeAmrSystemConfig cfg;
  for (int axis = 0; axis < Dim; ++axis) {
    cfg.shape[static_cast<std::size_t>(axis)] = n;
    cfg.lower[static_cast<std::size_t>(axis)] = Real(0);
    cfg.upper[static_cast<std::size_t>(axis)] = Real(1);
    cfg.periodicity[static_cast<std::size_t>(axis)] = true;
  }
  cfg.regrid_every = 0;
  cfg.level_count = 1;
  cfg.transition_ratios.clear();
  cfg.transition_buffers.clear();
  cfg.transition_lookaheads.clear();
  return cfg;
}

void install_gas_state_route(NativeAmrSystem& system) {
  system.install_block_state_route("gas", "tests.amr-riemann.state/gas");
}

struct Snap {
  std::vector<double> density;
  double mass = 0;
  int n_patches = 0;
};

Snap run(NativeAmrSystem& s, int nsteps) {
  s.set_poisson("charge_density", "geometric_mg");
  test::install_forward_euler_program(s);
  const double dt = 2e-4;
  for (int k = 0; k < nsteps; ++k)
    s.step(dt);
  return Snap{s.density(), s.mass(), s.n_patches()};
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
#include <string>
namespace pops_generated {
using ProdModel = pops::CompositeModel<pops::EulerND<pops::kNativeDimension>, pops::NoSource,
                                       pops::BackgroundDensity>;
}
// LITTERAL preprocesseur (PAS abi_key_string() : une inline serait interposee, ELF/RTLD_GLOBAL,
// vers la copie du module deja charge -> cle du module renvoyee -> garde d'ABI tautologique).
extern "C" const char* pops_native_abi_key() { return POPS_ABI_KEY_LITERAL; }
extern "C" const char* pops_compiled_route_manifest() { return pops::kRouteRegistrySignature; }
extern "C" int pops_compiled_nparams() { return 0; }
extern "C" const char* pops_compiled_param_names() { return ""; }
extern "C" void pops_install_native_amr(void* sys, const char* name, const char* limiter,
                                       const char* riemann, const char* recon, const char* time,
                                       double gamma, int substeps, const double*, int,
                                       double pos_floor, double weno_epsilon,
                                       bool wave_speed_cache) {
  pops::AmrSystem<pops::kNativeDimension>* s =
      reinterpret_cast<pops::AmrSystem<pops::kNativeDimension>*>(sys);
  pops::add_compiled_model<pops::kNativeDimension>(
      *s, name,
      pops_generated::ProdModel{{}, pops::EulerND<pops::kNativeDimension>{
                                      static_cast<pops::Real>(gamma)},
                                pops::NoSource{},
                               pops::BackgroundDensity{pops::Real(0), pops::Real(0)}},
      limiter, riemann, recon, time, gamma, substeps, 1, {}, {}, pos_floor, weno_epsilon,
      wave_speed_cache, "tests.amr-riemann.providers/gas");
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
  const int n = Dim == 3 ? 16 : 64;
  const int nsteps = Dim == 3 ? 4 : 12;
  const std::vector<double> rho = bubble(n);
  const std::vector<double> initial_state = conservative_state(rho);

  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // A rejected built-in route must leave the facade empty and eligible for a later valid install.
  {
    NativeAmrSystem transactional(make_cfg(n));
    install_gas_state_route(transactional);
    bool rejected = false;
    try {
      add_compiled_model(transactional, "gas", make_model(), "minmod", "missing-riemann",
                         "conservative", "explicit", kGamma, 1, 1, {}, {}, 0.0,
                         static_cast<double>(kWenoEpsilon), false, kProviderConsumerQid);
    } catch (const std::exception&) {
      rejected = true;
    }
    chk(rejected, "unknown built-in Riemann route rejected");
    chk(transactional.n_blocks() == 0, "failed built-in Riemann install is transactional");
    add_compiled_model(transactional, "gas", make_model(), "minmod", "rusanov", "conservative",
                       "explicit", kGamma, 1, 1, {}, {}, 0.0, static_cast<double>(kWenoEpsilon),
                       false, kProviderConsumerQid);
    chk(transactional.n_blocks() == 1, "valid built-in Riemann installs after rollback");
  }

  // ============================================================================================
  // (A) PARITE DECISIVE (sans .so) : convenience == preparation/publication explicite
  //     pour riemann=hllc/roe et recon=conservative/primitive (4 combos).
  //     Tourne sous TOUS les backends (hote, Kokkos Serial) -> ne casse pas nvcc.
  // ============================================================================================

  // Calcule l'oracle rusanov conservatif (pour le NO-SILENT-FALLBACK hllc/roe != rusanov).
  std::vector<double> d_rusanov;
  {
    NativeAmrSystem Ref(make_cfg(n));
    install_gas_state_route(Ref);
    add_compiled_model(Ref, "gas", make_model(), "minmod", "rusanov", "conservative", "explicit",
                       kGamma, 1, 1, {}, {}, 0.0, static_cast<double>(kWenoEpsilon), false,
                       kProviderConsumerQid);
    Ref.set_conservative_state("gas", initial_state);
    d_rusanov = run(Ref, nsteps).density;
  }

  // Parite add_compiled_model == prepare+install pour les 4 combos.
  // Renvoie la densite add_compiled_model (pour le test rusanov != riemann).
  auto parity_direct = [&](const char* riem, const char* recon) -> std::vector<double> {
    NativeAmrSystem A(make_cfg(n));  // bloc COMPILE
    install_gas_state_route(A);
    add_compiled_model(A, "gas", make_model(), "minmod", riem, recon, "explicit", kGamma, 1, 1, {},
                       {}, 0.0, static_cast<double>(kWenoEpsilon), false, kProviderConsumerQid);
    A.set_conservative_state("gas", initial_state);
    const Snap sa = run(A, nsteps);

    NativeAmrSystem B(make_cfg(n));  // publication preparee explicite
    install_gas_state_route(B);
    install_prepared_amr_block(
        B, prepare_compiled_amr_system_block<Dim>(
               "gas", make_model(), "minmod", riem, recon, "explicit", kGamma, 1, 1, 0.0,
               static_cast<double>(kWenoEpsilon), false, kProviderConsumerQid));
    B.set_conservative_state("gas", initial_state);
    const Snap sb = run(B, nsteps);

    const double nrm = maxabs(sb.density), dmax = maxdiff(sa.density, sb.density);
    char w[200];
    std::snprintf(w, sizeof w, "[%s/%s] densite non triviale", riem, recon);
    chk(nrm > 1e-6, w);
    std::snprintf(w, sizeof w, "[%s/%s] add_compiled_model == prepare+install (dmax==0)", riem,
                  recon);
    chk(dmax == 0.0, w);
    std::snprintf(w, sizeof w, "[%s/%s] masse add_compiled_model == prepare+install", riem, recon);
    chk(std::fabs(sa.mass - sb.mass) < 1e-12 * (std::fabs(sb.mass) + 1.0), w);
    std::snprintf(w, sizeof w, "[%s/%s] n_patches identique", riem, recon);
    chk(sa.n_patches == sb.n_patches, w);
    std::printf("OK  [%s/%s] dmax=%.0f\n", riem, recon, dmax);
    return sa.density;
  };

  const std::vector<double> d_hllc_cons = parity_direct("hllc", "conservative");
  const std::vector<double> d_hllc_prim = parity_direct("hllc", "primitive");
  const std::vector<double> d_roe_cons = parity_direct("roe", "conservative");
  const std::vector<double> d_roe_prim = parity_direct("roe", "primitive");

  // NO-SILENT-FALLBACK : hllc et roe doivent differer de rusanov sur ce meme etat (la bulle
  // est un ecoulement compressible non-trivial : hllc/roe ne se reduisent PAS a rusanov).
  chk(maxdiff(d_hllc_cons, d_rusanov) > 1e-12,
      "hllc != rusanov (le flux hllc est actif, non silencieux)");
  chk(maxdiff(d_roe_cons, d_rusanov) > 1e-12,
      "roe != rusanov (le flux roe est actif, non silencieux)");

  std::printf(
      "OK  (A) 4 combos hllc/roe x conservative/primitive BIT-IDENTIQUES (dmax==0) ; "
      "hllc != rusanov, roe != rusanov (flux actifs)\n");

  // ============================================================================================
  // (B) CHEMIN .so : add_native_block(loader) == add_compiled_model(AmrSystem&), hllc ET roe.
  //     Le loader est compile avec le meme compilateur et le meme contrat Kokkos que l'hote.
  // ============================================================================================
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
  auto parity_loader = [&](const char* riem, const char* recon) {
    NativeAmrSystem A(make_cfg(n));  // chemin "production" : loader .so -> add_native_block
    install_gas_state_route(A);
    A.add_native_block("gas", so, "minmod", riem, recon, "explicit", kGamma, 1);
    A.set_conservative_state("gas", initial_state);
    const Snap sa = run(A, nsteps);

    NativeAmrSystem B(make_cfg(n));  // MEME modele installe EN DIRECT
    install_gas_state_route(B);
    add_compiled_model(B, "gas", make_model(), "minmod", riem, recon, "explicit", kGamma, 1, 1, {},
                       {}, 0.0, static_cast<double>(kWenoEpsilon), false, kProviderConsumerQid);
    B.set_conservative_state("gas", initial_state);
    const Snap sb = run(B, nsteps);

    const double dmax = maxdiff(sa.density, sb.density);
    char w[200];
    std::snprintf(w, sizeof w, "[%s/%s] add_native_block == add_compiled_model (dmax==0)", riem,
                  recon);
    chk(dmax == 0.0, w);
    std::snprintf(w, sizeof w, "[%s/%s] n_patches loader == direct", riem, recon);
    chk(sa.n_patches == sb.n_patches, w);
  };
  parity_loader("hllc", "conservative");
  parity_loader("hllc", "primitive");
  parity_loader("roe", "conservative");
  parity_loader("roe", "primitive");
  std::printf("OK (B) add_native_block(hllc/roe, cons/prim) == add_compiled_model (dmax==0)\n");

  if (fails == 0)
    std::printf(
        "OK test_amr_riemann_native (hllc/roe x conservative/primitive : "
        "add_compiled_model == prepare+install, bit-identique ; hllc/roe actifs vs rusanov)\n");
  return fails ? 1 : 0;
}

TEST(test_amr_riemann_native, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_amr_riemann_native, "test_amr_riemann_native"),
            0);
}
