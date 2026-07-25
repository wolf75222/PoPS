// Final production-package coverage for a model-named auxiliary field. The test compiles an
// authenticated package, installs it through System::add_native_block, writes the named channel and
// executes the real native residual. No host callback or flat-array model path is involved.
#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "native_dso_compiler.hpp"

#include <pops/core/state/state.hpp>
#include <pops/runtime/system.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

using namespace pops;

namespace {

std::string package_source() {
  return R"CPP(
#include <pops/core/state/state.hpp>
#include <pops/core/state/variables.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/config/route_ids.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>

    struct NamedAuxModel {
      using State = pops::StateVec<1>;
      using Prim = pops::StateVec<1>;
      using Aux = pops::Aux;
      static constexpr int n_vars = 1;
      static constexpr int n_aux = pops::kAuxNamedBase + 1;
      POPS_HD State flux(const State&, const Aux&, int) const { return State{}; }
      POPS_HD pops::Real max_wave_speed(const State&, const Aux&, int) const { return pops::Real(0); }
      POPS_HD State source(const State& u, const Aux& aux) const {
        return State{aux.extra_field(0) * u[0]};
      }
      POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
      POPS_HD Prim to_primitive(const State& u) const { return u; }
      POPS_HD State to_conservative(const Prim& p) const { return p; }
      static pops::VariableSet conservative_vars() {
        return {pops::VariableKind::Conservative, {"u"}, 1, {pops::VariableRole::Custom}};
      }
      static pops::VariableSet primitive_vars() {
        return {pops::VariableKind::Primitive, {"u"}, 1, {pops::VariableRole::Custom}};
      }
    };

    extern "C" const char* pops_native_abi_key() {
      return POPS_ABI_KEY_LITERAL;
    }
    extern "C" const char* pops_compiled_route_manifest() {
      return pops::kRouteRegistrySignature;
    }
    extern "C" int pops_compiled_nparams() {
      return 0;
    }
    extern "C" const char* pops_compiled_param_names() {
      return "";
    }
    extern "C" void pops_install_native(void* raw, const char* name, const char* limiter,
                                        const char* riemann, const char* recon, const char* time,
                                        double gamma, int substeps, int evolve, int stride,
                                        const double*, int, double pos_floor) {
      auto* system = reinterpret_cast<pops::System*>(raw);
      pops::add_compiled_model(*system, name, NamedAuxModel{}, limiter, riemann, recon, time, gamma,
                               substeps, evolve != 0, stride, pos_floor);
    }
  )CPP";
}

}  // namespace

static int pops_run_test_native_aux_named(int argc, char** argv) {
  (void)argc;
  (void)argv;
  const std::string stem = std::string(POPS_TEST_TMPDIR) + "/native_named_aux_" +
                           std::to_string(static_cast<long>(std::clock()));
  const std::string source = stem + ".cpp";
  const std::string library = stem + ".so";
  {
    std::ofstream output(source);
    output << package_source();
  }
  const auto package = pops::test::native_dso::compile_shared(source, library);
  if (!package.ok) {
    pops::test::native_dso::report_compile_failure("test_native_aux_named", package);
    return 1;
  }

  constexpr int n = 8;
  constexpr double kappa = 0.7;
  const std::size_t cells = static_cast<std::size_t>(n) * n;
  SystemConfig config;
  config.n = n;
  config.L = 1.0;
  config.periodicity = {true, true};

  System system(config);
  system.add_native_block("scalar", library, "none", "rusanov", "conservative", "euler");
  system.set_state("scalar", std::vector<double>(cells, 1.0));
  system.set_aux_field_component(kAuxNamedBase, std::vector<double>(cells, kappa));
  const std::vector<double> residual = system.eval_rhs("scalar");
  double error = 0.0;
  for (double value : residual)
    error = std::fmax(error, std::fabs(value - kappa));

  bool missing_provider_rejected = false;
  try {
    System empty(config);
    empty.set_aux_field_component(kAuxNamedBase, std::vector<double>(cells, kappa));
  } catch (const std::runtime_error&) {
    missing_provider_rejected = true;
  }

  if (error > 1e-14 || !missing_provider_rejected) {
    std::printf("FAIL native named aux: error=%.3e missing_provider_rejected=%d\n", error,
                missing_provider_rejected ? 1 : 0);
    return 1;
  }
  std::printf("OK test_native_aux_named (authenticated native package, error=%.1e)\n", error);
  return 0;
}

TEST(test_native_aux_named, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_native_aux_named, "test_native_aux_named"), 0);
}
