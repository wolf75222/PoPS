/// @file
/// @brief Source-built exact-ranked AMR IMEX trajectory through PreparedAmrSystemBlock.

#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "gtest_compat.hpp"
#include "native_dso_compiler.hpp"

#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/runtime/program/step_transaction.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace {

constexpr int Dim = pops::kNativeDimension;
constexpr std::uint32_t kInjectedRetryReason = 0x494d4558u;
constexpr const char* kBlock = "tracer";
constexpr const char* kStateRoute = "tests.amr-imex/state/tracer";
constexpr const char* kProviderConsumer = "tests.amr-imex/providers/tracer";
constexpr const char* kImexProgramHash = "tests.amr-imex/program/imex-v1";
constexpr const char* kTransportProgramHash = "tests.amr-imex/program/transport-v1";
constexpr const char* kExplicitProgramHash = "tests.amr-imex/program/explicit-source-v1";

std::size_t cell_count(const pops::Extent<Dim>& shape) {
  std::size_t count = 1;
  for (int axis = 0; axis < Dim; ++axis)
    count *= static_cast<std::size_t>(shape[axis]);
  return count;
}

pops::AmrSystemConfig<Dim> config() {
  pops::AmrSystemConfig<Dim> result;
  const int width = Dim == 3 ? 12 : 24;
  result.level_count = 2;
  result.transition_ratios.resize(1);
  result.transition_buffers.resize(1);
  result.transition_lookaheads.resize(1);
  result.regrid_every = 0;
  result.explicit_bootstrap = true;
  result.distribute_coarse = true;
  for (int axis = 0; axis < Dim; ++axis) {
    result.shape[axis] = width;
    result.lower[axis] = pops::Real(0);
    result.upper[axis] = pops::Real(1);
    result.periodicity[axis] = true;
    result.coarse_max_grid[axis] = width / 2;
    result.transition_ratios[0][axis] = 2;
    result.transition_buffers[0][axis] = 1;
    result.transition_lookaheads[0][axis] = 1;
  }
  return result;
}

std::vector<double> initial_state(const pops::Extent<Dim>& shape) {
  const std::size_t cells = cell_count(shape);
  std::vector<double> result(cells, 1.0);
  const double pi = std::acos(-1.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    std::size_t quotient = cell;
    double wave = 0.08;
    for (int axis = 0; axis < Dim; ++axis) {
      const int coordinate = static_cast<int>(quotient % static_cast<std::size_t>(shape[axis]));
      quotient /= static_cast<std::size_t>(shape[axis]);
      const double x = (static_cast<double>(coordinate) + 0.5) / shape[axis];
      const double envelope = std::sin(pi * x);
      wave *= envelope * envelope;
    }
    result[cell] += wave;
  }
  return result;
}

std::string loader_source() {
  // clang-format off
  return R"CPP(
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/time/integrators/implicit_stepper.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/config/route_ids.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
namespace pops_generated {
template <int Dim>
struct RelaxingAdvection {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;
  Law law{};
  pops::Real decay = pops::Real(0);
  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.amr-imex.relaxing-advection", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    for (int axis = 0; axis < Dim; ++axis) contract.scalar(law.velocity()[axis]);
    contract.scalar(decay);
  }
  static pops::VariableSet conservative_vars() {
    return {pops::VariableKind::Conservative, {"u"}, 1, {pops::VariableRole::Scalar}};
  }
  static pops::VariableSet primitive_vars() {
    return {pops::VariableKind::Primitive, {"u"}, 1, {pops::VariableRole::Scalar}};
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
  template <int Axis> POPS_HD State flux(const State& state) const {
    return law.template flux<Axis>(state);
  }
  template <int Axis> POPS_HD pops::Real max_wave_speed(const State& state) const {
    return law.template max_wave_speed<Axis>(state);
  }
  template <int Axis>
  POPS_HD void wave_speeds(const State& state, pops::Real& lower, pops::Real& upper) const {
    law.template wave_speeds<Axis>(state, lower, upper);
  }
  POPS_HD State source(const State& state, const pops::ProviderValues<0>&) const {
    const pops::Real departure = state[0] - pops::Real(1);
    return State{-decay * (departure + departure * departure * departure)};
  }
  POPS_HD void source_jacobian(const State& state, const pops::ProviderValues<0>&,
                               pops::Real (&jacobian)[1][1]) const {
    const pops::Real departure = state[0] - pops::Real(1);
    jacobian[0][0] = -decay * (pops::Real(1) + pops::Real(3) * departure * departure);
  }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};
using Model = RelaxingAdvection<pops::kNativeDimension>;
}
extern "C" const char* pops_native_abi_key() { return POPS_ABI_KEY_LITERAL; }
extern "C" const char* pops_compiled_route_manifest() { return pops::kRouteRegistrySignature; }
extern "C" int pops_compiled_nparams() { return 0; }
extern "C" const char* pops_compiled_param_names() { return ""; }
extern "C" void pops_install_native_amr(void* sys, const char* name, const char* limiter,
                                        const char* riemann, const char* recon, const char* time,
                                        double gamma, int substeps, const double*, int,
                                        double pos_floor, double weno_epsilon,
                                        bool wave_speed_cache) {
  pops::RealVector<pops::kNativeDimension> velocity{};
  for (int axis = 0; axis < pops::kNativeDimension; ++axis)
    velocity[axis] = pops::Real(0.2) / pops::Real(axis + 1);
  pops_generated::Model model{
      pops::nd::ScalarAdvection<pops::kNativeDimension>::prepare(velocity), pops::Real(80)};
  auto* system = static_cast<pops::AmrSystem<pops::kNativeDimension>*>(sys);
  pops::add_compiled_model<pops::kNativeDimension>(
      *system, name, model, limiter, riemann, recon, time, gamma, substeps, 1, {}, {}, pos_floor,
      weno_epsilon, wave_speed_cache, "tests.amr-imex/providers/tracer");
}
)CPP";
  // clang-format on
}

struct ProgramEvidence {
  bool inject_retry = false;
  int rejected_solves = 0;
  int accepted_solves = 0;
  int coarse_transport_evaluations = 0;
  int fine_transport_evaluations = 0;
};

std::shared_ptr<ProgramEvidence> install_imex_program(pops::AmrSystem<Dim>& system,
                                                      bool inject_retry) {
  auto context = pops::runtime::program::make_program_execution_provider(&system);
  auto evidence = std::make_shared<ProgramEvidence>();
  evidence->inject_retry = inject_retry;
  context->configure_primary_clock("tests.amr-imex.clock");
  context->install([context, evidence](double macro_dt) {
    context->advance_hierarchy(macro_dt, [context, evidence](double level_dt) {
      context->set_stage_time(0, 1);
      pops::MultiFab<Dim>& accepted = context->state(0);
      pops::MultiFab<Dim>& transport = context->scratch_state(1000, 0, accepted);
      pops::MultiFab<Dim>& explicit_rate = context->rhs_scratch(2000, 0, accepted);
      context->neg_div_flux_default_into(0, accepted, explicit_rate, 3000);
      if (context->level() == 0)
        ++evidence->coarse_transport_evaluations;
      else
        ++evidence->fine_transport_evaluations;
      context->lincomb(transport, pops::Real(1), accepted, pops::Real(0), accepted);
      context->axpy(transport, pops::Real(level_dt), explicit_rate);

      pops::NewtonOptions options;
      if (evidence->inject_retry)
        options.max_iters = 1;
      evidence->inject_retry = false;
      pops::SolveOutcome implicit =
          context->solve_source_default(0, transport, pops::Real(level_dt), options);
      if (!implicit.report().solved_value_available()) {
        const pops::SolveReport rejected = implicit.consume(pops::SolveConsumption::kRejectAttempt);
        ++evidence->rejected_solves;
        throw pops::runtime::program::StepAttemptRejected(
            rejected.status, pops::runtime::program::StepAttemptDisposition::kRetry,
            kInjectedRetryReason, "implicit-source", rejected.reason);
      }
      const pops::SolveReport accepted_solve = implicit.consume(pops::SolveConsumption::kAccept);
      if (!accepted_solve.solved())
        throw std::logic_error("accepted IMEX source solve did not publish a solved value");
      ++evidence->accepted_solves;
      context->commit_many({{&accepted, &transport}});
    });
  });
  system.set_program_block_map({0});
  using FluxBudget = typename pops::AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBlockBudget;
  system.install_prepared_amr_program_flux_expression_budget(kImexProgramHash,
                                                             std::vector<FluxBudget>{{1, 1}});
  return evidence;
}

void install_transport_program(pops::AmrSystem<Dim>& system, bool explicit_source) {
  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock(explicit_source ? "tests.amr-imex.explicit-clock"
                                                   : "tests.amr-imex.transport-clock");
  context->install([context, explicit_source](double macro_dt) {
    context->advance_hierarchy(macro_dt, [context, explicit_source](double level_dt) {
      context->set_stage_time(0, 1);
      pops::MultiFab<Dim>& accepted = context->state(0);
      pops::MultiFab<Dim>& candidate = context->scratch_state(4000, 0, accepted);
      pops::MultiFab<Dim>& rate = context->rhs_scratch(5000, 0, accepted);
      context->neg_div_flux_default_into(0, accepted, rate, 6000);
      context->lincomb(candidate, pops::Real(1), accepted, pops::Real(0), accepted);
      context->axpy(candidate, pops::Real(level_dt), rate);
      if (explicit_source) {
        context->source_default_into(0, candidate, rate);
        context->axpy(candidate, pops::Real(level_dt), rate);
      }
      context->commit_many({{&accepted, &candidate}});
    });
  });
  system.set_program_block_map({0});
  using FluxBudget = typename pops::AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBlockBudget;
  system.install_prepared_amr_program_flux_expression_budget(
      explicit_source ? kExplicitProgramHash : kTransportProgramHash,
      std::vector<FluxBudget>{{1, 1}});
}

void build_refined_system(pops::AmrSystem<Dim>& system, const std::string& shared_object,
                          const std::vector<double>& state) {
  system.install_block_state_route(kBlock, kStateRoute);
  system.add_native_block(kBlock, shared_object, "minmod", "rusanov", "conservative", "imex", 1.4,
                          1);
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  system.set_conservative_state(kBlock, state);
  pops::test::install_prepared_threshold_union(system, {{kBlock, "u", 1.03}},
                                               "tests.amr-imex.tagging@1");
  system.begin_bootstrap_plan();
  if (!system.bootstrap_next_level()) {
    system.rollback_bootstrap_level();
    throw std::runtime_error("native IMEX fixture did not create its refined level");
  }
  system.commit_bootstrap_level();
}

double max_difference(const std::vector<double>& left, const std::vector<double>& right) {
  if (left.size() != right.size())
    return std::numeric_limits<double>::infinity();
  double result = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index)
    result = std::max(result, std::fabs(left[index] - right[index]));
  return result;
}

bool all_finite(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

double max_departure_from_equilibrium(const std::vector<double>& values) {
  double result = 0.0;
  for (const double value : values)
    result = std::max(result, std::fabs(value - 1.0));
  return result;
}

}  // namespace

TEST(test_amr_imex_native, SourceBuiltRefinedTrajectoryRollsBackRetriesAndRestartsExactly) {
#if defined(POPS_HAS_KOKKOS)
  int argc = 0;
  char** argv = nullptr;
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const std::string stem = std::string(POPS_TEST_TMPDIR) + "/amr_imex_native_" +
                           std::to_string(pops::my_rank()) + "_" +
                           std::to_string(static_cast<long>(std::clock()));
  const std::string source_path = stem + ".cpp";
  const std::string shared_object = stem + ".so";
  {
    std::ofstream source(source_path);
    source << loader_source();
  }
  const auto package = pops::test::native_dso::compile_shared(source_path, shared_object);
  if (!package.ok) {
    pops::test::native_dso::report_compile_failure("test_amr_imex_native", package);
    FAIL() << "source-built AMR IMEX artifact did not compile";
  }

  const auto system_config = config();
  const std::vector<double> initial = initial_state(system_config.shape);
  constexpr double dt = 3.0e-2;

  pops::AmrSystem<Dim> synchronization_control(system_config);
  build_refined_system(synchronization_control, shared_object, initial);
  install_transport_program(synchronization_control, false);
  synchronization_control.step(1.0e-12);

  pops::AmrSystem<Dim> transport_control(system_config);
  build_refined_system(transport_control, shared_object, initial);
  install_transport_program(transport_control, false);
  transport_control.step(dt);
  const double synchronized_mass = synchronization_control.mass(kBlock);
  EXPECT_NEAR(transport_control.mass(kBlock), synchronized_mass,
              2.0e-10 * (std::fabs(synchronized_mass) + 1.0));

  pops::AmrSystem<Dim> explicit_control(system_config);
  build_refined_system(explicit_control, shared_object, initial);
  install_transport_program(explicit_control, true);
  for (int step = 0; step < 3; ++step)
    explicit_control.step(dt);
  const std::vector<double> coarse_explicit = explicit_control.block_level_state_global(kBlock, 0);
  ASSERT_TRUE(all_finite(coarse_explicit));

  pops::AmrSystem<Dim> continuous(system_config);
  build_refined_system(continuous, shared_object, initial);
  ASSERT_EQ(continuous.n_levels(), 2);
  ASSERT_GT(continuous.n_patches(), 0);
  const std::vector<double> coarse_before = continuous.block_level_state_global(kBlock, 0);
  const std::vector<double> fine_before = continuous.block_level_state_global(kBlock, 1);
  const auto evidence = install_imex_program(continuous, true);

  try {
    continuous.step(dt);
    FAIL() << "the injected implicit-source retry was not surfaced";
  } catch (const pops::runtime::program::StepAttemptRejected& rejected) {
    EXPECT_EQ(rejected.disposition(), pops::runtime::program::StepAttemptDisposition::kRetry);
    EXPECT_EQ(rejected.reason_code(), kInjectedRetryReason);
    EXPECT_EQ(rejected.phase(), "implicit-source");
  }
  EXPECT_EQ(evidence->rejected_solves, 1);
  EXPECT_EQ(continuous.macro_step(), 0);
  EXPECT_DOUBLE_EQ(continuous.time(), 0.0);
  EXPECT_EQ(continuous.block_level_state_global(kBlock, 0), coarse_before);
  EXPECT_EQ(continuous.block_level_state_global(kBlock, 1), fine_before);

  continuous.step(dt);
  const std::vector<double> coarse_first = continuous.block_level_state_global(kBlock, 0);
  const std::vector<double> fine_first = continuous.block_level_state_global(kBlock, 1);
  ASSERT_TRUE(all_finite(coarse_first));
  ASSERT_TRUE(all_finite(fine_first));
  EXPECT_GT(max_difference(coarse_first, coarse_before), 0.0);
  EXPECT_GT(max_difference(fine_first, fine_before), 0.0);
  EXPECT_GT(evidence->coarse_transport_evaluations, 0);
  EXPECT_GT(evidence->fine_transport_evaluations, 0);
  EXPECT_GT(evidence->accepted_solves, 0);

  continuous.step(dt);
  continuous.step(dt);
  const std::vector<double> coarse_accepted = continuous.block_level_state_global(kBlock, 0);
  const std::vector<double> fine_accepted = continuous.block_level_state_global(kBlock, 1);
  EXPECT_LT(max_departure_from_equilibrium(coarse_accepted),
            max_departure_from_equilibrium(coarse_explicit));

  const std::vector<std::uint8_t> accepted_program = continuous.program_accepted_state();
  const double accepted_time = continuous.time();
  const int accepted_macro_step = continuous.macro_step();

  pops::AmrSystem<Dim> restarted(system_config);
  build_refined_system(restarted, shared_object, initial);
  static_cast<void>(install_imex_program(restarted, false));
  restarted.set_block_level_state(kBlock, 0, coarse_accepted);
  restarted.set_block_level_state(kBlock, 1, fine_accepted);
  restarted.restore_checkpoint_accepted_state(accepted_program);
  restarted.set_clock(accepted_time, accepted_macro_step);

  continuous.step(dt);
  restarted.step(dt);
  EXPECT_EQ(restarted.macro_step(), continuous.macro_step());
  EXPECT_DOUBLE_EQ(restarted.time(), continuous.time());
  EXPECT_EQ(restarted.block_level_state_global(kBlock, 0),
            continuous.block_level_state_global(kBlock, 0));
  EXPECT_EQ(restarted.block_level_state_global(kBlock, 1),
            continuous.block_level_state_global(kBlock, 1));
  EXPECT_DOUBLE_EQ(restarted.mass(kBlock), continuous.mass(kBlock));
}
