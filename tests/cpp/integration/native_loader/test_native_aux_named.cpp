// Final production-package coverage for a model-named auxiliary field. The test compiles an
// authenticated package, registers and finalizes it through System's native-package transaction,
// writes the named channel and executes the real native residual. No host callback or flat-array
// model path is involved.
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
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/config/route_ids.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>

#include <utility>

    namespace pops {

    template <int Dim, class Model>
    PreparedSystemBlock<Dim> prepare_exact_system_block(
        CompiledSystemBlockPreparation<Dim, Model> request) {
      return prepare_generated_system_block(std::move(request));
    }

    }  // namespace pops

    struct NamedAuxModel {
      using Law = pops::nd::ScalarAdvection<pops::kNativeDimension>;
      using Schema = typename Law::Schema;
      using State = typename Law::State;
      using Primitive = typename Law::Primitive;
      static constexpr int dimension = pops::kNativeDimension;
      static constexpr int n_vars = Law::n_vars;
      static constexpr int n_providers = 1;
      Law law{};
      static pops::PreparedProviderIdentity provider_identity() noexcept {
        return {"test.native-aux.named-scalar", 1};
      }
      void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
        law.serialize_exact_parameters(contract);
      }
      POPS_HD pops::nd::StateConversion<Primitive> recover(const State& state) const {
        return law.recover(state);
      }
      POPS_HD pops::nd::StateConversion<State> make_conservative(const Primitive& primitive) const {
        return law.make_conservative(primitive);
      }
      POPS_HD pops::nd::StateConversionStatus admissibility(const State& state) const {
        return law.admissibility(state);
      }
      template <int Axis>
      POPS_HD State flux(const State& state) const {
        return law.template flux<Axis>(state);
      }
      template <int Axis>
      POPS_HD pops::Real max_wave_speed(const State& state) const {
        return law.template max_wave_speed<Axis>(state);
      }
      template <int Axis>
      POPS_HD void wave_speeds(const State& state, pops::Real& lower, pops::Real& upper) const {
        law.template wave_speeds<Axis>(state, lower, upper);
      }
      POPS_HD State source(const State& u, const pops::ProviderValues<1>& providers) const {
        return State{providers[0] * u[0]};
      }
      POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
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
      auto* system = reinterpret_cast<pops::System<pops::kNativeDimension>*>(raw);
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
  std::size_t cells = 1;
  SystemConfig<kNativeDimension> config;
  for (int axis = 0; axis < kNativeDimension; ++axis) {
    cells *= static_cast<std::size_t>(n);
    config.shape[axis] = n;
    config.lower[axis] = Real(0);
    config.upper[axis] = Real(1);
    config.periodicity[axis] = true;
  }

  System<kNativeDimension> system(config);
  using namespace runtime::system;
  AuxiliaryStorageShape<kNativeDimension> shape;
  for (int axis = 0; axis < kNativeDimension; ++axis)
    shape.halo[axis] = 1;
  AuxiliaryComponentKey input_key{"test.native-aux", "input", "coefficient", "kappa"};
  AuxiliaryComponentContract contract{"cell-average", "cell", "unitless", "test-input", "scalar"};
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<kNativeDimension>{
      "test.native-aux.input",
      AuxiliaryProviderKind::input,
      {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
      {{input_key, contract, shape}},
      {}});
  system.install_auxiliary_consumer_plan(AuxiliaryConsumerProviderPlan<kNativeDimension>{
      "scalar", {{{input_key, contract, shape}, 0}}});
  system.install_block_state_route("scalar", "test.native-aux/scalar/state@1");
  system.register_native_package("scalar", library, "none", "rusanov", "conservative", "euler");
  system.finalize_native_packages();
  system.set_state("scalar", std::vector<double>(cells, 1.0));
  system.stage_auxiliary_input(input_key, std::vector<double>(cells, kappa));
  system.refresh_auxiliary(AuxiliaryEvaluationPoint{"test.native-aux", 0, 0, 0, 0, 0, 0,
                                                    AuxiliaryEvaluationEvent::initialization});
  const std::vector<double> residual = system.eval_rhs("scalar");
  bool residual_finite = true;
  for (double value : residual)
    residual_finite = std::isfinite(value) && residual_finite;
  const bool residual_valid = residual.size() == cells && residual_finite;
  double error = 0.0;
  if (residual_valid)
    for (double value : residual)
      error = std::fmax(error, std::fabs(value - kappa));

  bool missing_provider_rejected = false;
  try {
    System<kNativeDimension> empty(config);
    empty.seal_auxiliary_providers();
    empty.stage_auxiliary_input(input_key, std::vector<double>(cells, kappa));
  } catch (const std::out_of_range&) {
    missing_provider_rejected = true;
  }

  if (!residual_valid || error > 1e-14 || !missing_provider_rejected) {
    std::printf(
        "FAIL native named aux: residual_valid=%d residual_size=%zu error=%.3e "
        "missing_provider_rejected=%d\n",
        residual_valid ? 1 : 0, residual.size(), error, missing_provider_rejected ? 1 : 0);
    return 1;
  }
  std::printf("OK test_native_aux_named (authenticated native package, error=%.1e)\n", error);
  return 0;
}

TEST(test_native_aux_named, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_native_aux_named, "test_native_aux_named"), 0);
}
