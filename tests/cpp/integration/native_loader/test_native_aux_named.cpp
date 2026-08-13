// Final production-package coverage for a model-named auxiliary field. The test compiles an
// authenticated package, registers and finalizes it through System's native-package transaction,
// writes the named channel and executes the real native residual. No host callback or flat-array
// model path is involved.
#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "native_dso_compiler.hpp"

#include <pops/core/state/state.hpp>
#include <pops/runtime/dynamic/authenticated_native_file.hpp>
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
    extern "C" const char* pops_compiled_model_identity() {
      return "0000000000000000000000000000000000000000000000000000000000000000";
    }
    extern "C" int pops_native_system_package_abi_version() {
      return 2;
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
    extern "C" void pops_register_provider_routes(void* raw) {
      auto* system =
          static_cast<pops::runtime::system::PreparedNativeRouteRegistrar<pops::kNativeDimension>*>(
              raw);
      if (system == nullptr)
        throw std::invalid_argument("auxiliary route installer received null exact runtime");
      using namespace pops::runtime::system;
      AuxiliaryStorageShape<pops::kNativeDimension> shape;
      for (int axis = 0; axis < pops::kNativeDimension; ++axis)
        shape.halo[axis] = 1;
      const AuxiliaryComponentKey input_key{"test.native-aux", "input", "coefficient", "kappa"};
      const AuxiliaryComponentKey derived_key{"test.native-aux", "derived", "coefficient",
                                              "effective-kappa"};
      const AuxiliaryComponentContract contract{"cell-average", "cell", "unitless", "test-input",
                                                "scalar"};
      system->install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<pops::kNativeDimension>{
          "test.native-aux.input",
          AuxiliaryProviderKind::input,
          {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
          {{input_key, contract, shape}},
          {}});
      system->install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<pops::kNativeDimension>{
          "test.native-aux.derived",
          AuxiliaryProviderKind::derived,
          {AuxiliaryEvaluationEvent::before_residual, AuxiliaryFreshness::evaluation},
          {{derived_key, contract, shape}},
          {{input_key, contract, shape}},
          PreparedAuxiliaryProvider<pops::kNativeDimension>::launcher_type::trusted_extension(
              {"test.native-aux.derived-launcher", 1}, "test.native-aux.derived",
              [](const AuxiliaryKernelLaunchContext<pops::kNativeDimension>& context) {
                if (context.outputs.size() != 1 || context.dependencies.size() != 1 ||
                    context.storage.candidate == nullptr)
                  throw std::logic_error("native named auxiliary launcher contract differs");
                auto* const output_group =
                    context.storage.candidate->find(context.outputs[0].address.group);
                const auto* const input_group =
                    context.storage.candidate->find(context.dependencies[0].address.group);
                if (output_group == nullptr || input_group == nullptr ||
                    output_group->layout() != input_group->layout() ||
                    output_group->distribution() != input_group->distribution() ||
                    output_group->local_rank() != input_group->local_rank())
                  throw std::logic_error("native named auxiliary storage differs");
                const auto output_component = context.outputs[0].address.component;
                const auto input_component = context.dependencies[0].address.component;
                for (std::size_t local = 0; local < output_group->local_size(); ++local) {
                  const auto output = output_group->fab(local).view();
                  const auto input = input_group->fab(local).view();
                  std::size_t cells = 1;
                  for (int axis = 0; axis < pops::kNativeDimension; ++axis)
                    cells *= static_cast<std::size_t>(output.extents[axis]);
                  Kokkos::parallel_for(
                      "test_native_aux_named_derived", Kokkos::RangePolicy<>(0, cells),
                      KOKKOS_LAMBDA(const std::size_t linear) {
                        std::size_t remainder = linear;
                        pops::Index<pops::kNativeDimension> index{};
                        for (int axis = 0; axis < pops::kNativeDimension; ++axis) {
                          index[axis] = output.origin[axis] +
                                        static_cast<int>(remainder % static_cast<std::size_t>(
                                                                         output.extents[axis]));
                          remainder /= static_cast<std::size_t>(output.extents[axis]);
                        }
                        output(index, output_component) = input(index, input_component);
                      });
                }
              })});
      system->install_auxiliary_consumer_plan(AuxiliaryConsumerProviderPlan<pops::kNativeDimension>{
          "scalar", {{{derived_key, contract, shape}, 0}}});
    }
    extern "C" void pops_install_native(void* raw, const char* name, const char* limiter,
                                        const char* riemann, const char* recon, const char* time,
                                        double gamma, int substeps, int evolve, int stride,
                                        const double*, int, double pos_floor) {
      auto* system =
          static_cast<pops::runtime::system::PreparedNativeBlockInstaller<pops::kNativeDimension>*>(
              raw);
      pops::add_compiled_model(*system, name, "scalar", NamedAuxModel{}, limiter, riemann, recon,
                               time, gamma, substeps, evolve != 0, stride, pos_floor);
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
  const std::string binary_identity = dynlib::AuthenticatedNativeFile(library).binary_identity();

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
  AuxiliaryComponentKey input_key{"test.native-aux", "input", "coefficient", "kappa"};
  AuxiliaryComponentKey derived_key{"test.native-aux", "derived", "coefficient", "effective-kappa"};
  system.install_block_state_route("scalar", "test.native-aux/scalar/state@1");
  system.register_native_package("scalar", library,
                                 "0000000000000000000000000000000000000000000000000000000000000000",
                                 binary_identity, "none", "rusanov", "conservative", "euler");
  system.finalize_native_packages();
  const auto& consumer = system.prepared_auxiliary_consumer_plan("scalar");
  const auto address = system.auxiliary_address(derived_key);
  bool route_valid = consumer.consumer_qid == "scalar" && consumer.values.size() == 1 &&
                     consumer.values[0].key.exact_key() == derived_key.exact_key() &&
                     consumer.values[0].address.group == address.group &&
                     consumer.values[0].address.component == address.component;
  if (route_valid)
    for (int axis = 0; axis < kNativeDimension; ++axis)
      route_valid = consumer.values[0].shape.halo[axis] == 1 && route_valid;
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

  bool retryable_failure_exact = false;
  bool rollback_missing_key_exact = false;
  bool retry_valid = false;
  System<kNativeDimension> rejected(config);
  rejected.register_native_package(
      "scalar", library, "0000000000000000000000000000000000000000000000000000000000000000",
      binary_identity, "none", "rusanov", "conservative", "euler");
  try {
    rejected.finalize_native_packages();
  } catch (const std::runtime_error& error) {
    retryable_failure_exact = std::string(error.what()) ==
                              "System prepared block lacks its exact pre-installed state identity";
  }
  System<kNativeDimension> sealed_empty(config);
  sealed_empty.seal_auxiliary_providers();
  try {
    sealed_empty.stage_auxiliary_input(input_key, std::vector<double>(cells, kappa));
  } catch (const std::out_of_range& error) {
    rollback_missing_key_exact =
        std::string(error.what()) == "System auxiliary input key is not produced by this registry";
  }
  if (retryable_failure_exact) {
    rejected.install_block_state_route("scalar", "test.native-aux/scalar/state@1");
    rejected.finalize_native_packages();
    rejected.set_state("scalar", std::vector<double>(cells, 1.0));
    rejected.stage_auxiliary_input(input_key, std::vector<double>(cells, kappa));
    rejected.refresh_auxiliary(AuxiliaryEvaluationPoint{"test.native-aux-retry", 0, 0, 0, 0, 0, 0,
                                                        AuxiliaryEvaluationEvent::initialization});
    const std::vector<double> retry_residual = rejected.eval_rhs("scalar");
    retry_valid = retry_residual.size() == cells;
    for (double value : retry_residual)
      retry_valid = std::isfinite(value) && std::fabs(value - kappa) <= 1e-14 && retry_valid;
  }

  if (!route_valid || !residual_valid || error > 1e-14 || !retryable_failure_exact ||
      !rollback_missing_key_exact || !retry_valid) {
    std::printf(
        "FAIL native named aux: route_valid=%d residual_valid=%d residual_size=%zu error=%.3e "
        "retryable_failure_exact=%d rollback_missing_key_exact=%d retry_valid=%d\n",
        route_valid ? 1 : 0, residual_valid ? 1 : 0, residual.size(), error,
        retryable_failure_exact ? 1 : 0, rollback_missing_key_exact ? 1 : 0, retry_valid ? 1 : 0);
    return 1;
  }
  std::printf("OK test_native_aux_named (authenticated native package, error=%.1e)\n", error);
  return 0;
}

TEST(test_native_aux_named, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_native_aux_named, "test_native_aux_named"), 0);
}
