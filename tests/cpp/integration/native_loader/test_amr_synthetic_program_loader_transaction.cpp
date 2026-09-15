/// @file
/// @brief Synthetic source-built Program loader and AMR transaction artifact.
///
/// Its DSO body is deliberately authored in this C++ fixture to exercise loader ABI/hash/budget
/// authentication and hierarchy retry/rollback/publication.  It is not evidence for a user-authored
/// Python Program, its tableau, or temporal composition semantics.

#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "component_abi_test_helpers.hpp"
#include "gtest_compat.hpp"
#include "native_dso_compiler.hpp"

#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/dynamic/authenticated_native_file.hpp>
#include <pops/runtime/dynamic/dynlib.hpp>
#include <pops/runtime/dynamic/prepared_execution_context.hpp>
#include <pops/runtime/multiblock/interface_flux_scheduler.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/runtime/program/step_transaction.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace {

constexpr int Dim = pops::kNativeDimension;
constexpr std::uint32_t kSyntheticLoaderRetryReason = 0x534C5452u;
constexpr const char* kBlock = "tracer";
constexpr const char* kStateRoute = "tests.synthetic-loader/state/tracer";
constexpr const char* kProviderConsumer = "tests.synthetic-loader/providers/tracer";
constexpr const char* kSyntheticLoaderProgramHash =
    "tests.synthetic-loader/program/loader-transaction-v1";

std::shared_ptr<const pops::component::PreparedExecutionContextV1> prepared_execution() {
  const PopsExecutionContextV1 execution = pops::component::test_support::host_execution_context();
  return std::make_shared<const pops::component::PreparedExecutionContextV1>(
      execution.execution_identity, execution.context_version, execution.memory_space,
      execution.backend_identity, execution.device_identity, execution.scalar_type,
      execution.storage_precision, execution.compute_precision, execution.accumulation_precision,
      execution.reduction_precision, execution.stream_handle, execution.stream_identity,
      execution.communicator_f_handle, execution.communicator_datatype_f_handle,
      execution.communicator_identity, execution.communicator_datatype_identity);
}

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

std::string loader_source(bool interface_blocks = false, bool histories = false,
                          bool attempt_cursor = false, bool with_flux = true,
                          bool mapping = false) {
  // clang-format off
  return std::string("#define POPS_TEST_INTERFACE_BLOCKS ") +
      (interface_blocks ? "1\n" : "0\n") + "#define POPS_TEST_HISTORIES " +
      (histories ? "1\n" : "0\n") + "#define POPS_TEST_ATTEMPT_CURSOR " +
      (attempt_cursor ? "1\n" : "0\n") + "#define POPS_TEST_ATTEMPT_FLUX " +
      (with_flux ? "1\n" : "0\n") + "#define POPS_TEST_ATTEMPT_MAPPING " +
      (mapping ? "1\n" : "0\n") + R"CPP(
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/time/integrators/implicit_stepper.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/system/native_package_capability.hpp>
#include <pops/runtime/config/route_ids.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/runtime/program/step_transaction.hpp>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#if !defined(POPS_RUNTIME_SHARED_EXCEPTION_ABI)
#error "synthetic Program loader requires the shared runtime exception ABI"
#endif
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
    return {"test.synthetic-loader.relaxing-advection", 1};
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
#if POPS_TEST_ATTEMPT_CURSOR
    return {};
#else
    const pops::Real departure = state[0] - pops::Real(1);
    return State{-decay * (departure + departure * departure * departure)};
#endif
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
extern "C" int pops_native_system_package_abi_version() {
  return pops::runtime::system::kNativeSystemPackageAbiVersion;
}
extern "C" const char* pops_compiled_model_identity() {
  return "2222222222222222222222222222222222222222222222222222222222222222";
}
extern "C" const char* pops_compiled_route_manifest() { return pops::kRouteRegistrySignature; }
extern "C" int pops_compiled_nparams() { return 0; }
extern "C" const char* pops_compiled_param_names() { return ""; }
extern "C" void pops_register_provider_routes_amr(
    pops::AmrSystem<pops::kNativeDimension>* system) {
  if (system == nullptr)
    throw std::invalid_argument("AMR provider route installer received null exact runtime");
}
extern "C" void pops_install_native_amr(void* sys, const char* name, const char* limiter,
                                        const char* riemann, const char* recon, const char* time,
                                        double gamma, int substeps, int stride, const double*, int,
                                        double pos_floor, double weno_epsilon,
                                        bool wave_speed_cache, int newton_max_iters,
                                        double newton_rel_tol, double newton_abs_tol,
                                        double newton_fd_eps, double newton_damping,
                                        int newton_diagnostics) {
  pops::RealVector<pops::kNativeDimension> velocity{};
  for (int axis = 0; axis < pops::kNativeDimension; ++axis)
    velocity[axis] = POPS_TEST_ATTEMPT_CURSOR ? pops::Real(axis + 1)
                                             : pops::Real(0.2) / pops::Real(axis + 1);
  pops_generated::Model model{
      pops::nd::ScalarAdvection<pops::kNativeDimension>::prepare(velocity),
      POPS_TEST_ATTEMPT_CURSOR ? pops::Real(0) : pops::Real(80)};
  auto* system = static_cast<pops::AmrSystem<pops::kNativeDimension>*>(sys);
  const pops::NewtonOptions newton = pops::newton_options_from_abi(
      newton_max_iters, newton_rel_tol, newton_abs_tol, newton_fd_eps, newton_damping);
  pops::PreparedNativeAmrPackage<pops::kNativeDimension> package;
  package.block = pops::prepare_compiled_amr_system_block<pops::kNativeDimension>(
      name, std::move(model), limiter, riemann, recon, time, gamma, substeps, stride, pos_floor,
      weno_epsilon, wave_speed_cache, "tests.synthetic-loader/providers/tracer", newton,
      newton_diagnostics != 0);
  system->install_prepared_native_amr_package(std::move(package));
}

extern "C" const char* pops_program_abi_key() { return POPS_ABI_KEY_LITERAL; }
extern "C" const char* pops_program_route_manifest() { return pops::kRouteRegistrySignature; }
extern "C" const char* pops_program_name() { return "source-built-synthetic-loader-transaction"; }
extern "C" const char* pops_program_hash() {
#if POPS_TEST_ATTEMPT_CURSOR
  return POPS_TEST_ATTEMPT_MAPPING
      ? (POPS_TEST_ATTEMPT_FLUX ? "tests.attempt-cursor/mapping-flux@1"
                                : "tests.attempt-cursor/mapping-empty@1")
      : (POPS_TEST_ATTEMPT_FLUX ? "tests.attempt-cursor/hierarchy-flux@1"
                                : "tests.attempt-cursor/hierarchy-empty@1");
#elif POPS_TEST_HISTORIES
  return "tests.synthetic-loader/program/history-restart-v1";
#else
  return POPS_TEST_INTERFACE_BLOCKS
      ? "tests.synthetic-loader/program/interface-publication-v1"
      : "tests.synthetic-loader/program/loader-transaction-v1";
#endif
}
extern "C" int pops_program_operator_authority_count() { return 0; }
extern "C" std::uint64_t pops_program_operator_authority_word(int, int) { return 0; }
extern "C" int pops_program_block_count() { return POPS_TEST_INTERFACE_BLOCKS ? 2 : 1; }
extern "C" const char* pops_program_block_name(int block) {
  return block == 0 ? "tracer" : (POPS_TEST_INTERFACE_BLOCKS && block == 1 ? "tracer2" : "");
}
extern "C" bool pops_program_has_flux_expression() {
  return !POPS_TEST_ATTEMPT_CURSOR || POPS_TEST_ATTEMPT_FLUX;
}
extern "C" int pops_program_flux_expression_budget_count() { return pops_program_block_count(); }
extern "C" std::uint64_t pops_program_interface_coupling_application_bound() {
  return POPS_TEST_INTERFACE_BLOCKS ? UINT64_C(1) : UINT64_C(0);
}
extern "C" std::uint64_t pops_program_interface_coupling_identity_character_bound() {
  return POPS_TEST_INTERFACE_BLOCKS ? UINT64_C(128) : UINT64_C(0);
}
extern "C" std::uint64_t pops_program_flux_rhs_basis_bound(int block) {
  return block == 0 ? (POPS_TEST_ATTEMPT_CURSOR ? (POPS_TEST_ATTEMPT_FLUX ? UINT64_C(1) : UINT64_C(0))
                                              : UINT64_C(10)) : UINT64_C(0);
}
extern "C" std::uint64_t pops_program_flux_coefficient_term_bound(int block) {
  return block == 0 && (!POPS_TEST_ATTEMPT_CURSOR || POPS_TEST_ATTEMPT_FLUX)
      ? UINT64_C(1) : UINT64_C(0);
}
extern "C" int pops_program_checkpoint_history_count() { return POPS_TEST_HISTORIES ? 2 : 0; }
extern "C" const char* pops_program_checkpoint_history_name(int history) {
  return POPS_TEST_HISTORIES ? (history == 0 ? "tracer.first" : "tracer.second") : "";
}
extern "C" int pops_program_checkpoint_history_owner(int) { return 0; }
extern "C" const char* pops_program_checkpoint_history_state_identity(int) {
  return POPS_TEST_HISTORIES ? ("tests.synthetic-loader/state/tracer") : "";
}
extern "C" const char* pops_program_checkpoint_history_space_identity(int) {
  return POPS_TEST_HISTORIES ? ("cell.conservative") : "";
}
extern "C" const char* pops_program_checkpoint_history_clock_identity(int) {
  return POPS_TEST_HISTORIES ? ("tests.synthetic-loader.clock") : "";
}
extern "C" const char* pops_program_checkpoint_history_interpolation_identity(int) {
  return POPS_TEST_HISTORIES ? ("none") : "";
}
extern "C" int pops_program_checkpoint_history_depth(int) { return POPS_TEST_HISTORIES ? 2 : 0; }
extern "C" int pops_program_checkpoint_history_components(int) { return POPS_TEST_HISTORIES ? 1 : 0; }
extern "C" int pops_program_checkpoint_logical_clock_count() {
  return POPS_TEST_ATTEMPT_CURSOR ? 2 : 1;
}
extern "C" const char* pops_program_checkpoint_logical_clock_identity(int clock) {
  if (POPS_TEST_ATTEMPT_CURSOR)
    return clock == 0 ? "clock.macro" : (clock == 1 ? "clock.level.1" : "");
  return clock == 0 ? "tests.synthetic-loader.clock" : "";
}
extern "C" const char* pops_program_checkpoint_temporal_provider_identity() {
  return "pops.temporal-partition.global@1";
}
extern "C" std::uint64_t pops_program_checkpoint_temporal_cell_capacity() {
  return UINT64_C(0);
}
extern "C" std::uint64_t pops_program_checkpoint_temporal_cells_per_topology_cell() {
  return UINT64_C(0);
}
extern "C" int pops_module_operator_count() { return 0; }
extern "C" const char* pops_module_operator_owner(int) { return ""; }
extern "C" const char* pops_module_operator_name(int) { return ""; }
extern "C" const char* pops_module_operator_kind(int) { return ""; }
extern "C" const char* pops_module_operator_signature(int) { return ""; }
extern "C" const char* pops_module_operator_requirements(int) { return ""; }
extern "C" int pops_module_state_space_count() { return pops_program_block_count(); }
extern "C" const char* pops_module_state_space_name(int space) {
  return space >= 0 && space < pops_program_block_count() ? "U" : "";
}
extern "C" const char* pops_module_state_space_owner(int space) {
  return pops_program_block_name(space);
}
extern "C" int pops_module_field_space_count() { return 0; }
extern "C" const char* pops_module_field_space_name(int) { return ""; }
extern "C" const char* pops_module_field_space_owner(int) { return ""; }

#if POPS_TEST_ATTEMPT_CURSOR
namespace pops::runtime::program::detail {
template <int Dim>
struct AmrProgramHistoryRemapCollectiveTestAccess {
  using Context = AmrProgramContext<Dim>;
  static bool owns(const Context& context, const pops::AmrSystem<Dim>* system) {
    return context.facade_ == system;
  }
  static std::uint64_t generation(const Context& context) {
    return context.runtime_state().step_install_generation_;
  }
  static bool artifact_backed(const Context& context) {
    return context.runtime_state().artifact_backed_;
  }
  static std::uint64_t accepted(const Context& context) {
    return context.accepted_subcycling_attempt_;
  }
  static std::uint64_t allocated(const Context& context) {
    return context.allocated_subcycling_attempt_;
  }
  static void require_active_import_refusal(Context& context) {
    try {
      context.import_accepted_state_(true);
    } catch (const std::logic_error& error) {
      if (std::string(error.what()) ==
          "AMR accepted-state import crossed an active Program attempt")
        return;
      throw;
    }
    throw std::runtime_error("attempt-cursor active import unexpectedly succeeded");
  }
};
}
using AttemptContext = pops::runtime::program::AmrProgramContext<pops::kNativeDimension>;
using AttemptAccess = pops::runtime::program::detail::AmrProgramHistoryRemapCollectiveTestAccess<
    pops::kNativeDimension>;
struct AttemptControl {
  std::weak_ptr<AttemptContext> context;
  std::weak_ptr<int> failure;
  std::uint64_t generation;
};
static std::unordered_map<pops::AmrSystem<pops::kNativeDimension>*, AttemptControl>
    attempt_controls;
static std::shared_ptr<AttemptContext> require_attempt_context(
    pops::AmrSystem<pops::kNativeDimension>* system) {
  const auto found = attempt_controls.find(system);
  if (found == attempt_controls.end())
    throw std::logic_error("attempt-cursor control lacks its exact installed system");
  auto context = found->second.context.lock();
  if (!context || !AttemptAccess::owns(*context, system) ||
      AttemptAccess::generation(*context) != found->second.generation ||
      system->installed_program_hash() != pops_program_hash() ||
      !AttemptAccess::artifact_backed(*context))
    throw std::logic_error("attempt-cursor control lost its authenticated installed context");
  return context;
}
extern "C" std::uint64_t pops_test_attempt_observe(
    pops::AmrSystem<pops::kNativeDimension>* system, int observation) {
  auto context = require_attempt_context(system);
  if (observation == 0) return AttemptAccess::accepted(*context);
  if (observation == 1) return AttemptAccess::allocated(*context);
  if (observation == 2) return AttemptAccess::artifact_backed(*context) ? 1 : 0;
  throw std::invalid_argument("attempt-cursor observation is not declared");
}
extern "C" void pops_test_attempt_failure(
    pops::AmrSystem<pops::kNativeDimension>* system, int failure) {
  (void)require_attempt_context(system);
  auto control = attempt_controls.at(system).failure.lock();
  if (!control || failure < 0 || failure > 5)
    throw std::invalid_argument("attempt-cursor failure control is not declared");
  *control = failure;
}
#endif

static bool reject_interface_refresh = false;
extern "C" void pops_test_reject_interface_refresh(bool reject) {
  reject_interface_refresh = reject;
}
static bool reject_history_resource_refresh = false;
extern "C" void pops_test_reject_history_resource_refresh(bool reject) {
  reject_history_resource_refresh = reject;
}
extern "C" void pops_install_program_amr(
    pops::AmrSystem<pops::kNativeDimension>* system) {
  auto context = pops::runtime::program::make_program_execution_provider(system);
  auto inject_retry = std::make_shared<bool>(true);
#if POPS_TEST_ATTEMPT_CURSOR
  context->configure_primary_clock("clock.macro");
  context->declare_clock_relation("clock.macro", "clock.level.1", 1);
  auto failure = std::make_shared<int>(0);
  context->install([context, failure](double dt) {
    const auto body = [context, failure](double level_dt) {
      auto& state = context->state(0);
      if (*failure == 3)
        AttemptAccess::require_active_import_refusal(*context);
      if (POPS_TEST_ATTEMPT_FLUX) {
        context->set_stage_time(0, 1);
        auto& rhs = context->rhs_scratch(700, 0, state);
        context->rhs_into(0, state, rhs, 701);
        context->axpy(state, pops::Real(level_dt), rhs);
      }
      if (*failure == 1) {
        state.set_val(pops::Real(9));
        if (context->prepared_execution_lane().rank() == 0)
          throw std::runtime_error("attempt-cursor candidate rejection");
      }
      if (*failure == 4 || *failure == 5) {
        state.set_val(pops::Real(9));
        if (context->prepared_execution_lane().rank() == 0) {
          const bool retry = *failure == 4;
          throw pops::runtime::program::StepAttemptRejected(
              retry ? pops::SolveStatus::kIterationLimit : pops::SolveStatus::kInvalidEvaluation,
              retry ? pops::runtime::program::StepAttemptDisposition::kRetry
                    : pops::runtime::program::StepAttemptDisposition::kReject,
              retry ? UINT32_C(0x41544352) : UINT32_C(0x41544354),
              "attempt-cursor-body", retry ? "rank-zero-retry" : "rank-zero-terminal");
        }
      }
    };
    const auto completed = [failure] {
      if (*failure == 2)
        throw std::runtime_error("attempt-cursor outer publication rejection");
    };
    if (POPS_TEST_ATTEMPT_MAPPING)
      context->advance_mapping_hierarchy(dt, body, context, completed);
    else {
      context->advance_hierarchy(dt, body);
      completed();
    }
  }, context);
  // Keep context->install's real snapshot/preflight/regrid/resync callbacks. These source-built
  // tests exercise facade authority and transactions, not Python lowering or scientific accuracy.
  auto found = attempt_controls.find(system);
  if (found != attempt_controls.end() && !found->second.context.expired())
    throw std::logic_error("attempt-cursor context was installed twice for the same system");
  attempt_controls.insert_or_assign(
      system, AttemptControl{context, failure, AttemptAccess::generation(*context)});
#else
  context->configure_primary_clock("tests.synthetic-loader.clock");
#if POPS_TEST_HISTORIES
  // Cache actual scratch borrows per level, like the generated CPS installer. A restart must
  // rebuild them through the resource hook; the step intentionally never refreshes this cache.
  using LevelBody = std::function<void()>;
  auto level_bodies = std::make_shared<std::vector<LevelBody>>();
  auto epoch = std::make_shared<std::uint64_t>(std::numeric_limits<std::uint64_t>::max());
  auto generation = std::make_shared<std::uint64_t>(std::numeric_limits<std::uint64_t>::max());
  const auto refresh_resources = [context, level_bodies, epoch, generation](bool force = false) {
    if (force)
      *epoch = *generation = std::numeric_limits<std::uint64_t>::max();
    const auto topology = context->program_resource_topology();
    const auto& lane = context->prepared_execution_lane();
    const bool stale = force || *epoch != topology.epoch || *generation != topology.generation ||
                       level_bodies->size() != static_cast<std::size_t>(topology.levels);
    if (pops::all_reduce_max(stale ? 1L : 0L, lane) == 0)
      return;
    *epoch = *generation = std::numeric_limits<std::uint64_t>::max();
    std::vector<LevelBody> next;
    std::exception_ptr error;
    try { next.reserve(static_cast<std::size_t>(topology.levels)); }
    catch (...) { error = std::current_exception(); }
    pops::collectively_rethrow_exception(error, lane, "fixture resource allocation failed");
    context->for_each_program_resource_level([&](int) {
      error = {};
      try {
        for (const char* name : {"tracer.first", "tracer.second"})
          context->register_history(name, 1, 1, 0, "tests.synthetic-loader/state/tracer",
                                    "cell.conservative", "tests.synthetic-loader.clock", "none");
        auto* candidate = &context->scratch_state(1000, 0, context->state(0));
        next.emplace_back([context, candidate] {
          auto& accepted = context->state(0);
          context->lincomb(*candidate, pops::Real(2), accepted, pops::Real(0), accepted);
          context->store_history("tracer.first", accepted, 0);
          context->store_history("tracer.second", *candidate, 0);
          context->rotate_histories("tests.synthetic-loader.clock");
          context->commit_many({{&accepted, candidate}});
        });
      } catch (...) { error = std::current_exception(); }
      pops::collectively_rethrow_exception(error, lane, "fixture resource capture failed");
    });
    error = {};
    try {
      if (std::exchange(reject_history_resource_refresh, false))
        throw std::runtime_error("injected history resource refresh");
    } catch (...) { error = std::current_exception(); }
    pops::collectively_rethrow_exception(error, lane, "fixture resource publication failed");
    level_bodies->swap(next);
    *epoch = topology.epoch;
    *generation = topology.generation;
  };
  refresh_resources();
  context->install([context, level_bodies, epoch, generation](double dt) {
    context->advance_mapping_hierarchy(dt, [=](double) {
      const auto topology = context->program_resource_topology();
      if (*epoch != topology.epoch || *generation != topology.generation ||
          level_bodies->size() != static_cast<std::size_t>(topology.levels))
        throw std::logic_error("fixture continuation resources lost their exact hierarchy generation");
      level_bodies->at(static_cast<std::size_t>(context->level()))();
    }, context, [] {});
  }, context, [=] { refresh_resources(); }, [=] { refresh_resources(true); });

#else
  context->install(
      [context, inject_retry](double macro_dt) {
        context->advance_hierarchy(macro_dt, [context, inject_retry](double level_dt) {
          context->set_stage_time(0, 1);
          auto& accepted = context->state(0);
          auto& candidate = context->scratch_state(1000, 0, accepted);
          auto& explicit_rate = context->rhs_scratch(2000, 0, accepted);
          const bool source_only = context->macro_step() >= 3;
          if (!source_only)
            context->neg_div_flux_default_into(0, accepted, explicit_rate, 3000);
          context->lincomb(candidate, pops::Real(1), accepted, pops::Real(0), accepted);
          // Materialize ten independent, authenticated default-flux bases. The dyadic weights
          // sum exactly to one, so this decimal-boundary capacity witness preserves the fixture's
          // physical update while forcing identities 1 through 10 into the live expression.
          for (int basis = 0; !source_only && basis < 10; ++basis) {
            auto& rate = basis == 0 ? explicit_rate
                                    : context->rhs_scratch(2000 + basis, 0, accepted);
            if (basis != 0)
              context->neg_div_flux_default_into(0, accepted, rate, 3000 + basis);
            const int exponent = basis == 9 ? 9 : basis + 1;
            context->axpy(candidate, pops::Real(level_dt / static_cast<double>(1 << exponent)),
                          rate);
          }
          pops::SolveOutcome implicit = context->solve_source_default(
              0, candidate, pops::Real(level_dt), pops::NewtonOptions{});
          const pops::SolveReport solved = implicit.consume(pops::SolveConsumption::kAccept);
          if (!solved.solved())
            throw std::runtime_error("source-built AMR implicit source did not converge");
          if (*inject_retry) {
            *inject_retry = false;
            throw pops::runtime::program::StepAttemptRejected(
                pops::SolveStatus::kIterationLimit,
                pops::runtime::program::StepAttemptDisposition::kRetry, UINT32_C(0x534C5452),
                "implicit-source", "injected-synthetic-loader-transaction-retry");
          }
          context->commit_many({{&accepted, &candidate}});
        });
      },
      context, [context] {
        if (reject_interface_refresh)
          context->declare_clock_relation("tests.synthetic-loader.clock",
                                          "tests.synthetic-loader.undeclared-clock", 1);
      });
  system->install_program_restart_hooks(
      [] {}, [] {}, [] {},
      [context] { return context->accepted_context_snapshot(); });
#endif
#endif
}
)CPP";
  // clang-format on
}

// Native package consensus authenticates the actual binary bytes. Compile exactly once and
// distribute that image; independent links may embed distinct UUIDs or local dylib paths even
// when their C++ inputs are identical.
std::unique_ptr<pops::dynlib::AuthenticatedNativeFile> compile_exact_loader_artifact(
    const std::string& source_path, const std::string& shared_object,
    const pops::ExecutionLane& lane, bool interface_blocks = false, bool histories = false,
    bool attempt_cursor = false, bool with_flux = true, bool mapping = false) {
  const auto broadcast = [&](std::string& payload) {
    const bool length_overflow =
        lane.rank() == 0 &&
        payload.size() > static_cast<std::size_t>(std::numeric_limits<long>::max());
    if (pops::all_reduce_max(length_overflow ? 1L : 0L, lane) != 0)
      throw std::length_error("AMR fixture artifact exceeds the fixture length domain");
    const long count =
        pops::all_reduce_max(lane.rank() == 0 ? static_cast<long>(payload.size()) : 0L, lane);
    long allocation_failed = 0;
    try {
      payload.resize(static_cast<std::size_t>(count));
    } catch (const std::exception&) {
      allocation_failed = 1;
    }
    if (pops::all_reduce_max(allocation_failed, lane) != 0)
      throw std::runtime_error("AMR fixture artifact allocation failed collectively");
    pops::broadcast_bytes_inplace(payload.data(), payload.size(), lane, 0);
  };
  std::string image;
  std::string preparation_error;
  if (lane.rank() == 0) {
    try {
      {
        std::ofstream source(source_path);
        source.exceptions(std::ios::badbit | std::ios::failbit);
        source << loader_source(interface_blocks, histories, attempt_cursor, with_flux, mapping);
      }
      const auto package = pops::test::native_dso::compile_shared(
          source_path, shared_object, "-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI");
      if (!package.ok) {
        pops::test::native_dso::report_compile_failure(
            "test_amr_synthetic_program_loader_transaction", package);
        throw std::runtime_error("authenticated AMR fixture artifact did not compile");
      }
      std::ifstream binary(shared_object, std::ios::binary);
      binary.exceptions(std::ios::badbit);
      if (!binary)
        throw std::runtime_error("cannot read the compiled AMR fixture artifact");
      image.assign(std::istreambuf_iterator<char>(binary), std::istreambuf_iterator<char>());
    } catch (const std::exception& error) {
      preparation_error = error.what();
    }
  }
  broadcast(preparation_error);
  if (!preparation_error.empty())
    throw std::runtime_error("AMR fixture artifact preparation failed: " + preparation_error);
  // Independent links can carry different UUIDs. Every rank authenticates and loads the exact
  // rank-zero binary image, even when its local artifact path differs.
  broadcast(image);
  std::unique_ptr<pops::dynlib::AuthenticatedNativeFile> authenticated;
  try {
    if (lane.rank() != 0) {
      std::ofstream binary(shared_object, std::ios::binary);
      binary.exceptions(std::ios::badbit | std::ios::failbit);
      binary.write(image.data(), static_cast<std::streamsize>(image.size()));
    }
    authenticated = std::make_unique<pops::dynlib::AuthenticatedNativeFile>(shared_object);
  } catch (const std::exception& error) {
    preparation_error = error.what();
  }
  if (pops::all_reduce_max(preparation_error.empty() ? 0L : 1L, lane) != 0)
    throw std::runtime_error("AMR fixture artifact materialization failed collectively: " +
                             preparation_error);
  if (!pops::all_ranks_agree_exact_ordered_byte_pairs(
          {{"AMR fixture artifact", authenticated->content_sha256()}}, lane))
    throw std::runtime_error("AMR fixture artifact bytes differ between ranks");
  return authenticated;
}

void build_refined_system(pops::AmrSystem<Dim>& system, const std::string& shared_object,
                          const std::vector<double>& state, bool synchronous = false) {
  auto lane = std::make_shared<pops::ExecutionLane>(
      pops::ExecutionLane::duplicate_world_collectively("test.synthetic-loader.package"));
  auto execution = std::make_shared<const pops::component::PreparedExecutionContextV1>(
      prepared_execution()->for_lane(*lane));
  system.install_prepared_boundary_execution_context(std::move(lane), std::move(execution));
  system.install_block_state_route(kBlock, kStateRoute);
  const pops::dynlib::AuthenticatedNativeFile authenticated(shared_object);
  system.add_native_block(
      kBlock, shared_object, "2222222222222222222222222222222222222222222222222222222222222222",
      authenticated.binary_identity(), "minmod", "rusanov", "conservative", "explicit", 1.4, 1);
  system.set_temporal_relations({synchronous ? 1 : 2}, {1}, {"integral_only"});
  system.bind_bootstrap_subject(kStateRoute, kBlock, "bound_level_zero");
  system.stage_bootstrap_array(kStateRoute, kBlock, "cell", "cell", 1, system.spatial_shape(),
                               state);
  pops::Extent<Dim> transfer_ghosts{};
  pops::Extent<Dim> refinement_ratio{};
  for (int axis = 0; axis < Dim; ++axis) {
    transfer_ghosts[axis] = 1;
    refinement_ratio[axis] = 2;
  }
  system.register_bootstrap_transfer_route(
      "tests.synthetic-loader/bootstrap/prolongation", {kStateRoute},
      "tests.synthetic-loader/bootstrap/provider", "cell", "cell", "conservative", "dense",
      "prolongation", "conservative_linear", 2, transfer_ghosts, refinement_ratio);
  pops::test::install_prepared_threshold_union(system, {{kBlock, "u", 1.03}},
                                               "tests.synthetic-loader.tagging@1");
  // The real loader authenticates the Program block table, hash and four flux-expression budget
  // symbols before it calls the artifact installer and materializes the hierarchy.
  system.install_program(shared_object);
  // Match the public bind lifecycle: explicit bootstrap remains inside the assembling transaction,
  // and the final bound transition rechecks the Program checkpoint-capacity seal afterward.
  system.begin_bootstrap_plan();
  (void)system.materialize_bootstrap_action(kStateRoute, "initialize_level_zero",
                                            "bound_level_zero", 0);
  if (!system.bootstrap_next_level()) {
    system.rollback_bootstrap_level();
    throw std::runtime_error("synthetic loader fixture did not create its refined level");
  }
  (void)system.materialize_bootstrap_action(kStateRoute, "prolong_from_parent",
                                            "conservative_linear", 1);
  system.commit_bootstrap_level();
  system.mark_bound();
}

struct AttemptCursorArtifact {
  std::string path;
  std::unique_ptr<pops::dynlib::AuthenticatedNativeFile> authenticated;
  std::uint64_t (*observe)(pops::AmrSystem<Dim>*, int) = nullptr;
  void (*failure)(pops::AmrSystem<Dim>*, int) = nullptr;
};

AttemptCursorArtifact make_attempt_cursor_artifact(bool with_flux, bool mapping = false) {
  AttemptCursorArtifact artifact;
  const std::string stem = std::string(POPS_TEST_TMPDIR) + "/amr_attempt_cursor_" +
                           std::to_string(pops::my_rank()) + "_" +
                           std::to_string(static_cast<long>(std::clock()));
  artifact.path = stem + ".so";
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test.attempt-cursor.artifact");
  artifact.authenticated = compile_exact_loader_artifact(stem + ".cpp", artifact.path, lane, false,
                                                         false, true, with_flux, mapping);
  const auto handle = pops::dynlib::open(artifact.path);
  if (handle == nullptr)
    throw std::runtime_error("attempt-cursor artifact could not be opened");
  artifact.observe = reinterpret_cast<std::uint64_t (*)(pops::AmrSystem<Dim>*, int)>(
      pops::dynlib::sym(handle, "pops_test_attempt_observe"));
  artifact.failure = reinterpret_cast<void (*)(pops::AmrSystem<Dim>*, int)>(
      pops::dynlib::sym(handle, "pops_test_attempt_failure"));
  if (artifact.observe == nullptr || artifact.failure == nullptr)
    throw std::runtime_error("attempt-cursor artifact lacks its exact test controls");
  return artifact;
}

struct AttemptCursorFixture {
  std::unique_ptr<pops::AmrSystem<Dim>> system;
  const AttemptCursorArtifact* artifact = nullptr;

  std::uint64_t accepted_attempt() const { return artifact->observe(system.get(), 0); }
  std::uint64_t allocated_attempt() const { return artifact->observe(system.get(), 1); }
  bool program_is_artifact_backed() const { return artifact->observe(system.get(), 2) == 1; }
  void set_failure(int failure) const { artifact->failure(system.get(), failure); }
};

AttemptCursorFixture make_attempt_cursor_fixture(const AttemptCursorArtifact& artifact) {
  AttemptCursorFixture fixture;
  fixture.artifact = &artifact;
  pops::AmrSystemConfig<Dim> system_config;
  system_config.regrid_every = 0;
  system_config.explicit_bootstrap = true;
  for (int axis = 0; axis < Dim; ++axis) {
    system_config.shape[axis] = 16;
    system_config.periodicity[axis] = true;
    system_config.transition_buffers.front()[axis] = 0;
    system_config.transition_lookaheads.front()[axis] = 0;
  }
  fixture.system = std::make_unique<pops::AmrSystem<Dim>>(system_config);
  auto& system = *fixture.system;
  auto lane = std::make_shared<pops::ExecutionLane>(
      pops::ExecutionLane::duplicate_world_collectively("test.attempt-cursor.package"));
  auto execution = std::make_shared<const pops::component::PreparedExecutionContextV1>(
      prepared_execution()->for_lane(*lane));
  system.install_prepared_boundary_execution_context(std::move(lane), std::move(execution));
  system.set_temporal_relations({1}, {1}, {"integral_only"});
  system.install_block_state_route(kBlock, kStateRoute);
  system.add_native_block(kBlock, artifact.path,
                          "2222222222222222222222222222222222222222222222222222222222222222",
                          artifact.authenticated->binary_identity(), "minmod", "rusanov",
                          "conservative", "explicit", 1.4, 1);
  std::vector<double> initial(cell_count(system_config.shape), 1.0);
  for (std::size_t cell = 0; cell < initial.size(); ++cell) {
    std::size_t remainder = cell;
    bool centered = true;
    for (int axis = 0; axis < Dim; ++axis) {
      const auto index = remainder % 16;
      remainder /= 16;
      centered = centered && index >= 4 && index < 12;
    }
    if (centered)
      initial[cell] = 2.0;
  }
  system.bind_bootstrap_subject(kStateRoute, kBlock, "bound_level_zero");
  system.stage_bootstrap_array(kStateRoute, kBlock, "cell", "cell", 1, system.spatial_shape(),
                               initial);
  pops::Extent<Dim> transfer_ghosts{}, refinement_ratio{};
  for (int axis = 0; axis < Dim; ++axis) {
    transfer_ghosts[axis] = 1;
    refinement_ratio[axis] = 2;
  }
  system.register_bootstrap_transfer_route(
      "tests.attempt-cursor/bootstrap/prolongation", {kStateRoute},
      "tests.attempt-cursor/bootstrap/provider", "cell", "cell", "conservative", "dense",
      "prolongation", "conservative_linear", 2, transfer_ghosts, refinement_ratio);
  pops::test::install_prepared_refine_coarsen_threshold(
      system, {kBlock, "u", 1.5, pops::test::PreparedThresholdRelation::Above, kStateRoute},
      {kBlock, "u", 1.5, pops::test::PreparedThresholdRelation::Below, kStateRoute},
      "tests.attempt-cursor/tagging@1");
  // The public loader must grant authority BEFORE explicit materialization. A direct callback
  // install deliberately cannot qualify the facade's snapshot, restore or RegridOnRestart hooks.
  system.install_program(artifact.path);
  if (!fixture.program_is_artifact_backed())
    throw std::logic_error("attempt-cursor witness requires loader-granted artifact authority");
  system.begin_bootstrap_plan();
  (void)system.materialize_bootstrap_action(kStateRoute, "initialize_level_zero",
                                            "bound_level_zero", 0);
  if (!system.bootstrap_next_level()) {
    system.rollback_bootstrap_level();
    throw std::logic_error("attempt-cursor witness requires an actual partial fine level");
  }
  (void)system.materialize_bootstrap_action(kStateRoute, "prolong_from_parent",
                                            "conservative_linear", 1);
  system.commit_bootstrap_level();
  system.mark_bound();
  if (system.engine()->hierarchy().num_levels() != 2)
    throw std::logic_error("attempt-cursor witness requires an actual partial fine level");
  std::int64_t fine_cells = 0;
  for (const auto& box : system.prepared_amr_block_state(0, 1).layout().boxes())
    fine_cells += box.numPts();
  if (fine_cells <= 0 || fine_cells >= system.prepared_amr_level_geometry(1).domain().numPts())
    throw std::logic_error("attempt-cursor witness requires nonempty, partial fine coverage");
  return fixture;
}

double max_difference(const std::vector<double>& left, const std::vector<double>& right) {
  if (left.size() != right.size())
    return std::numeric_limits<double>::infinity();
  double result = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index)
    result = std::max(result, std::fabs(left[index] - right[index]));
  return result;
}

bool byte_exact_equal(const std::vector<double>& left, const std::vector<double>& right) {
  return left.size() == right.size() &&
         (left.empty() ||
          std::memcmp(left.data(), right.data(), left.size() * sizeof(double)) == 0);
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

std::vector<std::size_t> interior_level_indices(pops::AmrSystem<Dim>& system, int level) {
  const pops::Box<Dim> domain = system.prepared_amr_level_geometry(level).domain();
  const auto& boxes = system.prepared_amr_block_state(0, level).layout().boxes();
  std::vector<std::size_t> indices;
  for (const pops::Box<Dim>& patch : boxes) {
    // The fixture authenticates loader publication and rollback, not monotonic transport through
    // the coarse/fine halo band.  Sample only cells beyond that interface influence.
    const pops::Box<Dim> interior = patch.grow(-4);
    if (interior.empty())
      continue;
    indices.reserve(indices.size() + static_cast<std::size_t>(interior.numPts()));
    for (std::int64_t ordinal = 0; ordinal < interior.numPts(); ++ordinal) {
      std::int64_t remaining = ordinal;
      pops::Index<Dim> cell{};
      for (int axis = 0; axis < Dim; ++axis) {
        cell[axis] = interior.lo[axis] + remaining % interior.length(axis);
        remaining /= interior.length(axis);
      }
      std::size_t linear = 0;
      std::size_t stride = 1;
      for (int axis = 0; axis < Dim; ++axis) {
        linear += static_cast<std::size_t>(cell[axis] - domain.lo[axis]) * stride;
        stride *= static_cast<std::size_t>(domain.length(axis));
      }
      indices.push_back(linear);
    }
  }
  return indices;
}

std::vector<double> select_indices(const std::vector<double>& values,
                                   const std::vector<std::size_t>& indices) {
  std::vector<double> selected;
  selected.reserve(indices.size());
  for (const std::size_t index : indices)
    selected.push_back(values.at(index));
  return selected;
}

}  // namespace

TEST(test_amr_synthetic_program_loader_transaction,
     InterfacePublicationPreparesCapacityBeforeBootstrapAndRollsBackFailedRefresh) {
  const std::string stem = std::string(POPS_TEST_TMPDIR) + "/amr_interface_publication_" +
                           std::to_string(pops::my_rank()) + "_" +
                           std::to_string(static_cast<long>(std::clock()));
  const std::string source_path = stem + ".cpp";
  const std::string shared_object = stem + ".so";
  auto lane = std::make_shared<pops::ExecutionLane>(
      pops::ExecutionLane::duplicate_world_collectively("test.interface-publication.package"));
  const auto authenticated = compile_exact_loader_artifact(source_path, shared_object, *lane, true);
  const auto system_config = config();
  const auto initial = initial_state(system_config.shape);
  pops::AmrSystem<Dim> system(system_config);
  auto execution = std::make_shared<const pops::component::PreparedExecutionContextV1>(
      prepared_execution()->for_lane(*lane));
  system.install_prepared_boundary_execution_context(lane, execution);
  const std::vector<std::string> blocks{kBlock, "tracer2"};
  for (const auto& block : blocks)
    system.install_block_state_route(block, "state/" + block);
  for (const auto& block : blocks) {
    system.add_native_block(
        block, shared_object, "2222222222222222222222222222222222222222222222222222222222222222",
        authenticated->binary_identity(), "minmod", "rusanov", "conservative", "explicit", 1.4, 1);
  }
  for (const auto& block : blocks) {
    const std::string route = "state/" + block;
    system.bind_bootstrap_subject(route, block, "bound_level_zero");
    system.stage_bootstrap_array(route, block, "cell", "cell", 1, system.spatial_shape(), initial);
  }
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  pops::test::install_prepared_threshold_union(system, {{kBlock, "u", 1.03}},
                                               "tests.interface-publication.tagging@1");
  system.install_program(shared_object);
  ASSERT_EQ(system.prepared_amr_program_flux_expression_budget().blocks.size(), 2u);
  ASSERT_EQ(
      system.prepared_amr_program_flux_expression_budget().interface_coupling_application_bound,
      1u);
  const auto before = system.program_accepted_state();
  const auto before_revision = system.program_accepted_state_revision();
  const auto before_budget = system.prepared_amr_interface_flux_ledger_budget().exact_contract;
  const auto before_state = system.block_level_state_global(kBlock, 0);
  // This is the real artifact-backed window after install_program and before the first capacity
  // producer/bootstrap. Ordinary public restoration must still refuse the missing byte ceiling.
  EXPECT_THROW(system.restore_program_accepted_state(before), std::exception);
  const auto handle = pops::dynlib::open(shared_object);
  ASSERT_NE(handle, nullptr);
  const auto reject_refresh = reinterpret_cast<void (*)(bool)>(
      pops::dynlib::sym(handle, "pops_test_reject_interface_refresh"));
  ASSERT_NE(reject_refresh, nullptr);
  using namespace pops::runtime::multiblock;
  auto install = [&](const std::string& identity, bool high) {
    system.install_prepared_amr_interface_flux_provider(identity, [&](auto& scheduler) {
      AxisAlignedInterface<Dim> route;
      route.identity = identity;
      route.left_block = 0;
      route.right_block = 1;
      route.left_axis = route.right_axis = 0;
      route.left_side = high ? InterfaceSide::High : InterfaceSide::Low;
      route.right_side = high ? InterfaceSide::Low : InterfaceSide::High;
      route.right_component_for_left = {0};
      route.affine_mapping_identity = identity + ".translation";
      route.right_normal_translation = high ? pops::Real(1) : pops::Real(-1);
      route.left_trace_projection_identity = identity + ".left.trace";
      route.right_trace_projection_identity = identity + ".right.trace";
      route.left_trace_provider_identity = "test.cell-average.left";
      route.right_trace_provider_identity = "test.cell-average.right";
      route.left_trace_operation = route.right_trace_operation =
          InterfaceTraceOperation::CellAverage;
      route.left_trace_required_depth = route.right_trace_required_depth = 1;
      const auto geometry = system.prepared_amr_level_geometry(0);
      scheduler.install(
          route, system.prepared_amr_block_state(0, 0), geometry,
          system.prepared_amr_block_state(1, 0), geometry, execution->view(),
          InterfaceFluxEvaluatorFactory([]() {
            return InterfaceFluxEvaluator(
                [](const BoundaryEvaluationPoint&, const InterfaceFluxBatch&) {
                  throw std::logic_error("publication fixture cannot evaluate numerical flux");
                });
          }));
    });
  };
  // A valid local context mutation produces a rank-divergent checkpoint on MPI2, or a checkpoint
  // outside the frozen logical-clock metadata on rank1. Both fail after scheduler publication.
  reject_refresh(pops::my_rank() == 0);
  EXPECT_THROW(install("tests.interface.first", true), std::exception);
  reject_refresh(false);
  EXPECT_EQ(system.program_accepted_state(), before);
  EXPECT_EQ(system.program_accepted_state_revision(), before_revision);
  EXPECT_EQ(system.prepared_amr_interface_flux_ledger_budget().exact_contract, before_budget);
  EXPECT_THROW(system.restore_program_accepted_state(before), std::exception);
  install("tests.interface.first", true);
  const auto first_bytes = system.program_accepted_state();
  const auto first_revision = system.program_accepted_state_revision();
  const auto first_capacity = system.checkpoint_program_state_capacity();
  const auto first_budget = system.prepared_amr_interface_flux_ledger_budget();
  // The public ledger budget describes the one live level before bootstrap. It has no adjacent
  // coarse/fine window, so the scheduler produces exactly zero oriented fragments and payload.
  // The checkpoint ceiling instead reserves the configured two-level hierarchy; adding another
  // logical interface below must still enlarge that authentic future capacity. This fixture
  // proves publication/rollback and capacity assembly without evaluating flux. The full
  // multilevel execution oracle belongs to test_shared_interface_runtime.py.
  EXPECT_EQ(system.n_levels(), 1);
  EXPECT_EQ(system.configured_n_levels(), 2);
  EXPECT_EQ(first_budget.max_fragments_per_window, 0u);
  EXPECT_EQ(first_budget.max_payload_terms_per_window, 0u);
  EXPECT_EQ(first_budget.max_transaction_depth, 1u);
  EXPECT_EQ(first_budget.max_evaluation_fragments, 0u);
  EXPECT_EQ(first_budget.max_evaluation_payload_terms, 0u);
  EXPECT_NE(first_budget.exact_contract, before_budget);
  EXPECT_LE(first_bytes.size(), first_capacity.first);

  reject_refresh(pops::my_rank() == 0);
  EXPECT_THROW(install("tests.interface.second", false), std::exception);
  reject_refresh(false);
  EXPECT_EQ(system.program_accepted_state(), first_bytes);
  EXPECT_EQ(system.program_accepted_state_revision(), first_revision);
  EXPECT_EQ(system.checkpoint_program_state_capacity(), first_capacity);
  EXPECT_EQ(system.prepared_amr_interface_flux_ledger_budget().exact_contract,
            first_budget.exact_contract);
  install("tests.interface.second", false);
  const auto complete_capacity = system.checkpoint_program_state_capacity();
  EXPECT_GT(complete_capacity.first, first_capacity.first);
  EXPECT_EQ(system.block_level_state_global(kBlock, 0), before_state);
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_DOUBLE_EQ(system.time(), 0.0);

  system.begin_bootstrap_plan();
  for (const auto& block : blocks)
    (void)system.materialize_bootstrap_action("state/" + block, "initialize_level_zero",
                                              "bound_level_zero", 0);
  system.install_prepared_amr_interface_flux_provider("tests.interface.bootstrap-prefix",
                                                      [](auto&) {});
  system.commit_bootstrap_level();
  system.mark_bound();
  EXPECT_EQ(system.checkpoint_program_state_capacity(), complete_capacity);
  for (const auto& block : blocks)
    EXPECT_TRUE(byte_exact_equal(system.block_level_state_global(block, 0), initial));
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_DOUBLE_EQ(system.time(), 0.0);
  pops::dynlib::close(handle);
  std::remove(source_path.c_str());
  std::remove(shared_object.c_str());
}

TEST(test_amr_synthetic_program_loader_transaction,
     SourceBuiltArtifactLoadsBudgetRollsBackAndRetries) {
  const std::string stem = std::string(POPS_TEST_TMPDIR) + "/amr_synthetic_loader_" +
                           std::to_string(pops::my_rank()) + "_" +
                           std::to_string(static_cast<long>(std::clock()));
  const std::string source_path = stem + ".cpp";
  const std::string shared_object = stem + ".so";
  auto artifact_lane =
      pops::ExecutionLane::duplicate_world_collectively("test.synthetic-loader.artifact");
  {
    SCOPED_TRACE("compile and authenticate one exact artifact across ranks");
    ASSERT_NO_THROW((void)compile_exact_loader_artifact(source_path, shared_object, artifact_lane,
                                                        false, false));
  }

  const auto system_config = config();
  const std::vector<double> initial = initial_state(system_config.shape);
  constexpr double dt = 3.0e-2;

  pops::AmrSystem<Dim> continuous(system_config);
  build_refined_system(continuous, shared_object, initial);
  ASSERT_EQ(continuous.n_levels(), 2);
  ASSERT_GT(continuous.n_patches(), 0);
  EXPECT_EQ(continuous.installed_program_hash(), kSyntheticLoaderProgramHash);
  EXPECT_TRUE(continuous.program_flux_ledger_manifest().empty());
  EXPECT_TRUE(continuous.program_sync_manifest().empty());
  const auto& budget = continuous.prepared_amr_program_flux_expression_budget();
  EXPECT_EQ(budget.program_hash, kSyntheticLoaderProgramHash);
  ASSERT_EQ(budget.blocks.size(), 1u);
  ASSERT_EQ(budget.program_block_map.canonical_indices.size(), 1u);
  EXPECT_EQ(budget.program_block_map.canonical_indices[0], 0u);
  EXPECT_EQ(budget.blocks[0].rhs_basis_bound, 10u);
  EXPECT_EQ(budget.blocks[0].coefficient_term_bound, 1u);

  const std::vector<double> coarse_before = continuous.block_level_state_global(kBlock, 0);
  const std::vector<double> fine_before = continuous.block_level_state_global(kBlock, 1);
  const std::vector<std::size_t> fine_interior_indices = interior_level_indices(continuous, 1);
  ASSERT_FALSE(fine_interior_indices.empty());
  const std::vector<double> fine_interior_before =
      select_indices(fine_before, fine_interior_indices);
  EXPECT_GT(max_departure_from_equilibrium(fine_interior_before), 0.0);
  EXPECT_LT(max_departure_from_equilibrium(fine_interior_before), 0.25);

  try {
    continuous.step(dt);
    FAIL() << "the injected implicit-source retry was not surfaced";
  } catch (const pops::runtime::program::StepAttemptRejected& rejected) {
    EXPECT_EQ(rejected.disposition(), pops::runtime::program::StepAttemptDisposition::kRetry);
    EXPECT_EQ(rejected.reason_code(), kSyntheticLoaderRetryReason);
    EXPECT_EQ(rejected.phase(), "implicit-source");
  }
  EXPECT_EQ(continuous.macro_step(), 0);
  EXPECT_DOUBLE_EQ(continuous.time(), 0.0);
  EXPECT_TRUE(byte_exact_equal(continuous.block_level_state_global(kBlock, 0), coarse_before));
  EXPECT_TRUE(byte_exact_equal(continuous.block_level_state_global(kBlock, 1), fine_before));

  continuous.step(dt);
  const std::vector<double> coarse_first = continuous.block_level_state_global(kBlock, 0);
  const std::vector<double> fine_first = continuous.block_level_state_global(kBlock, 1);
  const std::vector<double> fine_interior_first = select_indices(fine_first, fine_interior_indices);
  ASSERT_TRUE(all_finite(coarse_first));
  ASSERT_TRUE(all_finite(fine_first));
  EXPECT_GT(max_difference(coarse_first, coarse_before), 0.0);
  EXPECT_GT(max_difference(fine_first, fine_before), 0.0);
  EXPECT_LT(max_departure_from_equilibrium(coarse_first),
            max_departure_from_equilibrium(coarse_before));
  EXPECT_LT(max_departure_from_equilibrium(fine_interior_first),
            max_departure_from_equilibrium(fine_interior_before));
  EXPECT_EQ(continuous.macro_step(), 1);
  EXPECT_DOUBLE_EQ(continuous.time(), dt);
  const std::vector<std::uint8_t> accepted_bytes = continuous.program_accepted_state();
  const auto accepted =
      pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(accepted_bytes);
  EXPECT_TRUE(std::any_of(accepted.accepted_face_flux.begin(), accepted.accepted_face_flux.end(),
                          [](const auto& fragments) { return !fragments.empty(); }));
  std::set<std::string> materialized_stages;
  for (const auto& fragments : accepted.accepted_face_flux)
    for (const auto& fragment : fragments)
      materialized_stages.insert(fragment.key.stage);
  EXPECT_EQ(materialized_stages.size(), 10u);
  EXPECT_TRUE(std::any_of(
      materialized_stages.begin(), materialized_stages.end(),
      [](const std::string& stage) { return stage.find("/basis/10/") != std::string::npos; }));
  EXPECT_LE(accepted_bytes.size(), continuous.checkpoint_program_state_capacity().first);

  continuous.step(dt);
  continuous.step(dt);
  const std::vector<double> coarse_accepted = continuous.block_level_state_global(kBlock, 0);
  const std::vector<double> fine_accepted = continuous.block_level_state_global(kBlock, 1);
  const std::vector<double> fine_interior_accepted =
      select_indices(fine_accepted, fine_interior_indices);
  ASSERT_TRUE(all_finite(coarse_accepted));
  ASSERT_TRUE(all_finite(fine_accepted));
  EXPECT_LT(max_departure_from_equilibrium(coarse_accepted),
            max_departure_from_equilibrium(coarse_first));
  EXPECT_LT(max_departure_from_equilibrium(fine_interior_accepted),
            max_departure_from_equilibrium(fine_interior_first));
  EXPECT_EQ(continuous.macro_step(), 3);
  EXPECT_DOUBLE_EQ(continuous.time(), 3.0 * dt);

  const auto before_regrid = pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(
      continuous.program_accepted_state());
  ASSERT_TRUE(before_regrid.face_evidence_provenance);
  const auto recorded_boxes = continuous.patch_boxes();
  std::vector<int> recorded_owners(recorded_boxes.size(), -1);
  continuous.rebuild_hierarchy(recorded_boxes, recorded_owners);
  const auto same_geometry = pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(
      continuous.program_accepted_state());
  EXPECT_EQ(same_geometry.face_evidence_provenance, before_regrid.face_evidence_provenance);
  EXPECT_EQ(same_geometry.accepted_face_flux[0].size(), before_regrid.accepted_face_flux[0].size());
  EXPECT_EQ(same_geometry.synchronization_events.size(),
            before_regrid.synchronization_events.size());
  for (int axis = 0; axis < Dim; ++axis) {
    ASSERT_EQ(same_geometry.accepted_face_flux[axis].size(),
              before_regrid.accepted_face_flux[axis].size());
    for (std::size_t index = 0; index < before_regrid.accepted_face_flux[axis].size(); ++index) {
      const auto& actual_key = same_geometry.accepted_face_flux[axis][index].key;
      const auto& expected_key = before_regrid.accepted_face_flux[axis][index].key;
      EXPECT_FALSE(actual_key < expected_key);
      EXPECT_FALSE(expected_key < actual_key);
      EXPECT_EQ(same_geometry.accepted_face_flux[axis][index].payload,
                before_regrid.accepted_face_flux[axis][index].payload);
    }
  }

  continuous.rebuild_hierarchy({}, {});
  ASSERT_EQ(continuous.n_levels(), 1);
  const auto removed_child = pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(
      continuous.program_accepted_state());
  EXPECT_EQ(removed_child.face_evidence_provenance, before_regrid.face_evidence_provenance);
  EXPECT_EQ(removed_child.face_evidence_provenance->level_count, 2u);
  EXPECT_EQ(removed_child.level_clocks.size(), 1u);
  EXPECT_EQ(removed_child.accepted_face_flux[0].size(), before_regrid.accepted_face_flux[0].size());

  // The fourth accepted step evaluates only the genuine implicit source. Historic transport
  // fragments must disappear, rather than being mistaken for this new step's numerical input.
  continuous.step(dt);
  const auto source_only = pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(
      continuous.program_accepted_state());
  for (const auto& fragments : source_only.accepted_face_flux)
    EXPECT_TRUE(fragments.empty());
  EXPECT_TRUE(source_only.synchronization_events.empty());
  EXPECT_FALSE(source_only.face_evidence_provenance);
}

TEST(test_amr_synthetic_program_loader_transaction,
     InitializedHistoryRestartRequiresCompleteRestorationAndRollsBack) {
  const std::string stem = std::string(POPS_TEST_TMPDIR) + "/amr_history_restart_" +
                           std::to_string(pops::my_rank()) + "_" +
                           std::to_string(static_cast<long>(std::clock()));
  const std::string source_path = stem + ".cpp";
  const std::string shared_object = stem + ".so";
  auto artifact_lane =
      pops::ExecutionLane::duplicate_world_collectively("test.synthetic-loader.artifact");
  {
    SCOPED_TRACE("compile and authenticate one exact artifact across ranks");
    ASSERT_NO_THROW((void)compile_exact_loader_artifact(source_path, shared_object, artifact_lane,
                                                        false, true));
  }

  const auto settings = config();
  pops::AmrSystem<Dim> system(settings);
  {
    SCOPED_TRACE("install, bootstrap, and seal frozen history capacity");
    ASSERT_NO_THROW(
        build_refined_system(system, shared_object, initial_state(settings.shape), true));
  }
  ASSERT_EQ(system.installed_program_hash(), "tests.synthetic-loader/program/history-restart-v1");
  const auto handle = pops::dynlib::open(shared_object);
  ASSERT_NE(handle, nullptr);
  using RejectRefresh = void (*)(bool);
  auto reject_refresh = reinterpret_cast<RejectRefresh>(
      pops::dynlib::sym(handle, "pops_test_reject_history_resource_refresh"));
  ASSERT_NE(reject_refresh, nullptr);
  ASSERT_EQ(system.n_levels(), 2);
  const std::vector<std::string> names{"tracer.first", "tracer.second"};
  ASSERT_EQ(system.history_names(), names);
  for (const auto& name : names) {
    ASSERT_EQ(system.history_depth(name), 2);
    ASSERT_EQ(system.history_levels(name), (std::vector<int>{0, 1}));
    for (int level = 0; level < system.n_levels(); ++level) {
      EXPECT_FALSE(system.history_initialized(name, level));
      EXPECT_EQ(system.history_fill_count(name, level), 0);
      for (int slot = 0; slot < 2; ++slot)
        EXPECT_DOUBLE_EQ(system.history_slot_dt(name, level, slot), 0.0);
    }
  }
  struct History {
    std::string name;
    int level;
    bool initialized;
    int fill;
    std::vector<std::vector<double>> values;
    std::vector<double> dt;
    std::vector<std::uint8_t> samples;
    bool operator==(const History&) const = default;
  };
  struct Image {
    std::vector<pops::AmrPatch<Dim>> boxes;
    std::vector<int> owners;
    std::vector<std::vector<double>> states;
    std::vector<History> histories;
    std::vector<std::uint8_t> accepted, exchanges, flux_shard;
    int regrids, step;
    std::uint64_t epoch;
    double time, last_dt;
    bool operator==(const Image&) const = default;
  };
  const auto capture = [&] {
    Image image;
    image.boxes = system.patch_boxes();
    image.owners = system.level_owner_ranks(1);
    if (system.level_distribution_mode(1) == "replicated")
      std::fill(image.owners.begin(), image.owners.end(), -1);
    image.accepted = system.program_accepted_state();
    image.exchanges = system.checkpoint_program_exchanges();
    image.flux_shard = system.program_history_flux_snapshot_shard();
    image.regrids = system.checkpoint_regrid_count();
    image.epoch = system.checkpoint_topology_epoch();
    image.step = system.macro_step();
    image.time = system.time();
    image.last_dt = system.program_last_dt();
    for (int level = 0; level < system.n_levels(); ++level)
      image.states.push_back(system.block_level_state_global(kBlock, level));
    for (const auto& name : names)
      for (int level = 0; level < system.n_levels(); ++level) {
        History history{name,
                        level,
                        system.history_initialized(name, level),
                        system.history_fill_count(name, level),
                        {},
                        {},
                        system.history_sample_identity(name, level)};
        for (int slot = 0; slot < system.history_depth(name); ++slot) {
          history.values.push_back(system.history_global(name, level, slot));
          history.dt.push_back(system.history_slot_dt(name, level, slot));
        }
        image.histories.push_back(std::move(history));
      }
    return image;
  };
  const auto read_wire8 = [](const std::vector<std::uint8_t>& bytes) {
    pops::runtime::program::checkpoint_detail::Reader header(bytes);
    header.expect_raw(pops::runtime::program::checkpoint_detail::kMagic);
    return pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(bytes);
  };
  constexpr double first_dt = 0.125;
  constexpr double second_dt = 0.1875;
  Image checkpoint;
  {
    SCOPED_TRACE("first accepted step and complete checkpoint capture");
    ASSERT_NO_THROW(system.step(first_dt));
    ASSERT_EQ(system.history_names(), names);
    ASSERT_NO_THROW(checkpoint = capture());
  }
  const auto checkpoint_program = read_wire8(checkpoint.accepted);
  ASSERT_EQ(checkpoint_program.accepted_attempt, 1u);
  ASSERT_EQ(checkpoint.histories.size(), 4u);
  for (const auto& history : checkpoint.histories) {
    ASSERT_TRUE(history.initialized);
    ASSERT_EQ(history.fill, 1);
    ASSERT_EQ(history.values.size(), 2u);
    EXPECT_EQ(history.dt, (std::vector<double>{first_dt, first_dt}));
  }
  // These are state histories: the exact native archive is empty, not an omitted RHS payload.
  ASSERT_TRUE(checkpoint.flux_shard.empty());
  Image uninterrupted;
  {
    SCOPED_TRACE("second accepted step and uninterrupted image capture");
    ASSERT_NO_THROW(system.step(second_dt));
    ASSERT_NO_THROW(uninterrupted = capture());
  }
  const auto uninterrupted_program = read_wire8(uninterrupted.accepted);
  ASSERT_EQ(uninterrupted_program.accepted_attempt, 2u);
  ASSERT_NE(uninterrupted.histories, checkpoint.histories);
  for (const auto& history : uninterrupted.histories)
    ASSERT_EQ(history.fill, 2);

  // Arbitrary topology edits still cannot discard initialized history provenance.
  EXPECT_THROW(system.rebuild_hierarchy(checkpoint.boxes, checkpoint.owners), std::exception);
  EXPECT_EQ(capture(), uninterrupted);
  system.begin_restart_transaction();
  ASSERT_NO_THROW(system.rebuild_hierarchy(checkpoint.boxes, checkpoint.owners));
  EXPECT_THROW(system.commit_restart_transaction(), std::exception);
  system.finalize_restart_transaction();  // A rejected commit must retain rollback ownership.
  ASSERT_NO_THROW(system.rollback_restart_transaction());
  EXPECT_EQ(capture(), uninterrupted);

  const auto materialize = [&] {
    system.rebuild_hierarchy(checkpoint.boxes, checkpoint.owners);
    system.restore_checkpoint_counters(checkpoint.regrids, checkpoint.epoch);
    system.materialize_program_restart_histories(checkpoint.accepted, names, {2, 2}, {1, 1});
  };
  const auto restore = [&](bool omit_one_payload) {
    for (int level = 0; level < system.n_levels(); ++level)
      system.set_block_level_state(kBlock, level, checkpoint.states.at(level));
    system.restore_program_cadence_window(0.0, 0, 0.0, checkpoint.last_dt, checkpoint.time,
                                          checkpoint.step);
    system.set_clock(checkpoint.time, checkpoint.step);
    for (const auto& history : checkpoint.histories) {
      for (int slot = 0; slot < static_cast<int>(history.values.size()); ++slot) {
        // Every rank must enter restore_history's collective engine preflight. Reject only the
        // numeric payload on rank zero, after that preflight and before its slot is marked written.
        if (omit_one_payload && history.name == names.back() && history.level == 1 && slot == 1) {
          SCOPED_TRACE("rank-local numeric rejection after collective history preflight");
          const bool reject_payload = pops::my_rank() == 0;
          const std::vector<double> missing_payload;
          std::string refusal;
          try {
            system.restore_history(history.name, history.level, slot,
                                   reject_payload ? missing_payload : history.values.at(slot));
          } catch (const std::invalid_argument& error) {
            refusal = error.what();
          }
          EXPECT_EQ(refusal, reject_payload
                                 ? "AMR history restore payload has the wrong exact-ranked size"
                                 : "");
        } else {
          system.restore_history(history.name, history.level, slot, history.values.at(slot));
        }
      }
      system.restore_history_provenance(history.name, history.level, history.dt,
                                        history.initialized, history.fill);
      system.restore_history_sample_identity(history.name, history.level, history.samples);
    }
    system.restore_checkpoint_program_exchanges(checkpoint.exchanges);
    system.restore_checkpoint_accepted_state(checkpoint.accepted);
  };
  system.begin_restart_transaction();
  ASSERT_NO_THROW(materialize());
  EXPECT_THROW(system.commit_restart_transaction(), std::exception);
  ASSERT_NO_THROW(restore(true));  // Metadata import before selective replay remains legitimate.
  EXPECT_THROW(system.preflight_regrid_on_restart(), std::exception);
  EXPECT_THROW(system.regrid_on_restart(), std::exception);
  EXPECT_THROW(system.commit_restart_transaction(), std::exception);
  system.finalize_restart_transaction();
  ASSERT_NO_THROW(system.rollback_restart_transaction());
  EXPECT_EQ(capture(), uninterrupted);

  std::optional<std::uint64_t> third_accepted_attempt, fourth_accepted_attempt;
  {
    SCOPED_TRACE("failed resource publication restores the previous CPS captures");
    system.begin_restart_transaction();
    ASSERT_NO_THROW(materialize());
    ASSERT_NO_THROW(restore(false));
    reject_refresh(pops::my_rank() == 0);
    EXPECT_THROW(restore(false), std::exception);
    EXPECT_THROW(system.preflight_regrid_on_restart(), std::exception);
    EXPECT_THROW(system.regrid_on_restart(), std::exception);
    EXPECT_THROW(system.commit_restart_transaction(), std::exception);
    // Failure while recapturing the rollback image must retain its snapshot and block execution.
    reject_refresh(pops::my_rank() == 0);
    EXPECT_THROW(system.rollback_restart_transaction(), std::exception);
    EXPECT_THROW(system.step(second_dt), std::logic_error);
    ASSERT_NO_THROW(system.rollback_restart_transaction());
    EXPECT_EQ(capture(), uninterrupted);
    // The outer step transaction restores the same physical generation but destroys scratches.
    // Its rollback must force fresh captures before the following continuation can use them.
    system.begin_step_transaction();
    ASSERT_NO_THROW(system.step(second_dt));
    ASSERT_NO_THROW(system.commit_step_transaction());
    third_accepted_attempt = read_wire8(system.program_accepted_state()).accepted_attempt;
    ASSERT_EQ(third_accepted_attempt, 3u);
    reject_refresh(pops::my_rank() == 0);
    EXPECT_THROW(system.rollback_step_transaction(), std::exception);
    EXPECT_TRUE(system.has_active_step_transaction());
    EXPECT_THROW(system.step(second_dt), std::logic_error);
    EXPECT_THROW(system.commit_step_transaction(), std::exception);
    EXPECT_THROW(system.finalize_step_transaction(), std::exception);
    ASSERT_NO_THROW(system.rollback_step_transaction());
    EXPECT_EQ(capture(), uninterrupted);
    system.begin_step_transaction();
    ASSERT_NO_THROW(system.step(second_dt));
    fourth_accepted_attempt = read_wire8(system.program_accepted_state()).accepted_attempt;
    ASSERT_EQ(fourth_accepted_attempt, 4u);
    ASSERT_NO_THROW(system.rollback_step_transaction());
    EXPECT_EQ(capture(), uninterrupted);
  }

  system.begin_restart_transaction();
  ASSERT_NO_THROW(materialize());
  ASSERT_NO_THROW(restore(false));
  // A second materialization invalidates both the imported authority and numeric completion.
  ASSERT_NO_THROW(
      system.materialize_program_restart_histories(checkpoint.accepted, names, {2, 2}, {1, 1}));
  EXPECT_THROW(system.commit_restart_transaction(), std::exception);
  ASSERT_NO_THROW(restore(false));
  EXPECT_EQ(capture(), checkpoint);
  ASSERT_NO_THROW(system.commit_restart_transaction());
  EXPECT_THROW(system.rebuild_hierarchy(checkpoint.boxes, checkpoint.owners), std::exception);
  EXPECT_THROW(
      system.materialize_program_restart_histories(checkpoint.accepted, names, {2, 2}, {1, 1}),
      std::exception);
  const auto& row = checkpoint.histories.front();
  EXPECT_THROW(system.restore_history(row.name, row.level, 0, row.values.front()), std::exception);
  EXPECT_THROW(system.set_history_initialized(row.name, row.level, row.initialized),
               std::exception);
  EXPECT_THROW(system.restore_history_fill_count(row.name, row.level, row.fill), std::exception);
  EXPECT_THROW(system.restore_history_metadata(row.name, row.level, row.initialized, row.fill),
               std::exception);
  EXPECT_THROW(
      system.restore_history_provenance(row.name, row.level, row.dt, row.initialized, row.fill),
      std::exception);
  EXPECT_THROW(system.restore_history_slot_dt(row.name, row.level, 0, row.dt.front()),
               std::exception);
  EXPECT_THROW(system.restore_history_sample_identity(row.name, row.level, row.samples),
               std::exception);
  EXPECT_THROW(system.rebuild_history_slots(row.name, {0, 1}), std::exception);
  EXPECT_EQ(capture(), checkpoint);
  system.finalize_restart_transaction();
  {
    SCOPED_TRACE("continue the fully restored checkpoint");
    ASSERT_NO_THROW(system.step(second_dt));
    const Image continued = capture();
    const auto continued_program = read_wire8(continued.accepted);
    ASSERT_EQ(continued_program.accepted_attempt, 5u);
    ASSERT_TRUE(fourth_accepted_attempt.has_value());
    EXPECT_EQ(continued_program.accepted_attempt, *fourth_accepted_attempt + 1u);
    // This is an in-place restore after accepted attempts 3 and 4 were rolled back. Their
    // allocator IDs remain consumed; the older reference has C2, while this continuation has C5.
    // Fresh-context whole-image equality is covered separately by the interior restart witness.
    EXPECT_EQ(continued.boxes, uninterrupted.boxes);
    EXPECT_EQ(continued.owners, uninterrupted.owners);
    EXPECT_EQ(continued.states, uninterrupted.states);
    EXPECT_EQ(continued.histories, uninterrupted.histories);
    EXPECT_EQ(continued.exchanges, uninterrupted.exchanges);
    EXPECT_EQ(continued.flux_shard, uninterrupted.flux_shard);
    EXPECT_EQ(continued.regrids, uninterrupted.regrids);
    EXPECT_EQ(continued.step, uninterrupted.step);
    EXPECT_EQ(continued.epoch, uninterrupted.epoch);
    EXPECT_EQ(continued.time, uninterrupted.time);
    EXPECT_EQ(continued.last_dt, uninterrupted.last_dt);

    // Both complete wire8 images were decoded and validated above. Derive the one u64 field's
    // extent from the native prefix schema; compare every other original byte without rewriting.
    ASSERT_EQ(continued_program.spatial_contract, uninterrupted_program.spatial_contract);
    pops::runtime::program::checkpoint_detail::CountingWriter prefix;
    prefix.raw(pops::runtime::program::checkpoint_detail::kMagic);
    prefix.i32(Dim);
    prefix.string(continued_program.spatial_contract);
    prefix.u64(continued_program.topology_epoch);
    prefix.u64(continued_program.materialization_generation);
    const std::size_t attempt_begin = prefix.count();
    prefix.u64(*continued_program.accepted_attempt);
    const std::size_t attempt_end = prefix.count();
    ASSERT_EQ(continued.accepted.size(), uninterrupted.accepted.size());
    ASSERT_LE(attempt_end, continued.accepted.size());
    EXPECT_TRUE(std::equal(continued.accepted.begin(), continued.accepted.begin() + attempt_begin,
                           uninterrupted.accepted.begin()));
    EXPECT_TRUE(std::equal(continued.accepted.begin() + attempt_end, continued.accepted.end(),
                           uninterrupted.accepted.begin() + attempt_end));
  }

  // The declared post-restore regrid consumes a complete incoming image and creates a new
  // authenticated history image. The final commit must validate that transformed authority.
  system.begin_restart_transaction();
  ASSERT_NO_THROW(materialize());
  ASSERT_NO_THROW(restore(false));
  ASSERT_NO_THROW(system.preflight_regrid_on_restart());
  ASSERT_NO_THROW(system.regrid_on_restart());
  EXPECT_GT(system.checkpoint_topology_epoch(), checkpoint.epoch);
  EXPECT_NE(system.patch_boxes(), checkpoint.boxes);
  for (const auto& history : checkpoint.histories) {
    EXPECT_EQ(system.history_fill_count(history.name, history.level), history.fill);
    EXPECT_EQ(system.history_sample_identity(history.name, history.level), history.samples);
  }
  ASSERT_NO_THROW(system.commit_restart_transaction());
  system.finalize_restart_transaction();
}

// Source-built scalar artifacts authenticate native lifecycle structure only. The retained
// direct-install failure did not grant facade restart authority; these four witnesses use the
// actual loader and preserve every C/H, full-image, field, revision, clock and rollback check.
TEST(test_amr_synthetic_program_loader_transaction,
     AttemptCursorFreshInteriorRestartMatchesEntireAcceptedImage) {
  for (const bool with_flux : {false, true}) {
    SCOPED_TRACE(with_flux);
    auto artifact = make_attempt_cursor_artifact(with_flux);
    auto cold = make_attempt_cursor_fixture(artifact);
    ASSERT_TRUE(cold.program_is_artifact_backed());
    EXPECT_THROW(artifact.observe(nullptr, 0), std::logic_error);
    auto& original = *cold.system;
    const auto initial = pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(
        original.program_accepted_state());
    EXPECT_EQ(initial.accepted_attempt, 0u);
    EXPECT_EQ(cold.allocated_attempt(), 0u);
    for (const auto& axis : initial.accepted_face_flux)
      EXPECT_TRUE(axis.empty());
    const auto epoch = original.checkpoint_topology_epoch();
    ASSERT_NO_THROW(original.step(.01));
    ASSERT_NO_THROW(original.step(.01));
    const auto middle = original.program_accepted_state();
    const auto decoded =
        pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(middle);
    EXPECT_EQ(decoded.accepted_attempt, 2u);
    EXPECT_EQ(original.checkpoint_topology_epoch(), epoch);  // Strictly between regrids.
    const bool has_faces =
        std::any_of(decoded.accepted_face_flux.begin(), decoded.accepted_face_flux.end(),
                    [](const auto& axis) { return !axis.empty(); });
    EXPECT_EQ(has_faces, with_flux);

    auto resumed = make_attempt_cursor_fixture(artifact);
    ASSERT_TRUE(resumed.program_is_artifact_backed());
    auto& restored = *resumed.system;
    EXPECT_EQ(resumed.allocated_attempt(), 0u);
    restored.begin_restart_transaction();
    for (int level = 0; level < original.n_levels(); ++level)
      restored.set_block_level_state("tracer", level,
                                     original.block_level_state_global("tracer", level));
    restored.restore_program_cadence_window(
        original.program_cadence_window_dt(), original.program_cadence_window_steps(),
        original.program_cadence_window_start_time(), original.program_last_dt(), original.time(),
        original.macro_step());
    restored.set_clock(original.time(), original.macro_step());
    restored.restore_checkpoint_counters(original.checkpoint_regrid_count(), epoch);
    ASSERT_NO_THROW(restored.restore_checkpoint_accepted_state(middle));
    restored.commit_restart_transaction();
    restored.finalize_restart_transaction();
    EXPECT_EQ(resumed.accepted_attempt(), 2u);
    EXPECT_EQ(resumed.allocated_attempt(), 2u);
    EXPECT_EQ(restored.program_accepted_state(), middle);
    EXPECT_EQ(restored.program_last_dt(), original.program_last_dt());
    ASSERT_NO_THROW(original.step(.01));
    ASSERT_NO_THROW(restored.step(.01));
    EXPECT_EQ(original.program_accepted_state(), restored.program_accepted_state());
    for (int level = 0; level < original.n_levels(); ++level)
      EXPECT_EQ(original.block_level_state_global("tracer", level),
                restored.block_level_state_global("tracer", level));
    EXPECT_EQ(cold.accepted_attempt(), 3u);
    EXPECT_EQ(resumed.accepted_attempt(), 3u);
  }
}

TEST(test_amr_synthetic_program_loader_transaction,
     AttemptCursorRollbackBurnsRejectedIdsAndRestoresCommittedBytes) {
  for (const bool mapping : {false, true}) {
    SCOPED_TRACE(mapping);
    auto artifact = make_attempt_cursor_artifact(true, mapping);
    auto fixture = make_attempt_cursor_fixture(artifact);
    ASSERT_TRUE(fixture.program_is_artifact_backed());
    auto& system = *fixture.system;
    ASSERT_NO_THROW(system.step(.01));
    const auto accepted = system.program_accepted_state();
    const auto revision = system.program_accepted_state_revision();
    const auto coarse = system.block_level_state_global("tracer", 0);
    const auto fine = system.block_level_state_global("tracer", 1);
    for (const int failure : {1, 2}) {
      fixture.set_failure(failure);
      EXPECT_THROW(system.step(.01), std::runtime_error);
      EXPECT_EQ(system.program_accepted_state(), accepted);
      EXPECT_EQ(system.program_accepted_state_revision(), revision);
      EXPECT_EQ(system.block_level_state_global("tracer", 0), coarse);
      EXPECT_EQ(system.block_level_state_global("tracer", 1), fine);
      EXPECT_EQ(system.macro_step(), 1);
      EXPECT_EQ(system.time(), .01);
      EXPECT_EQ(fixture.accepted_attempt(), 1u);
      EXPECT_EQ(fixture.allocated_attempt(), 1u + failure);
    }
    // Restoring an older accepted image in this SAME context must not recycle either failed ID.
    ASSERT_NO_THROW(system.restore_checkpoint_accepted_state(accepted));
    EXPECT_EQ(fixture.allocated_attempt(), 3u);
    fixture.set_failure(3);
    ASSERT_NO_THROW(system.step(.01));
    EXPECT_EQ(fixture.accepted_attempt(), 4u);
    const auto next = pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(
        system.program_accepted_state());
    EXPECT_EQ(next.accepted_attempt, 4u);
    for (const auto& axis : next.accepted_face_flux)
      for (const auto& fragment : axis)
        EXPECT_EQ(fragment.key.attempt, 4u);
  }
}

TEST(test_amr_synthetic_program_loader_transaction,
     AttemptCursorRankLocalTypedRejectionRetainsMetadataAndRollback) {
  auto artifact = make_attempt_cursor_artifact(true);
  auto fixture = make_attempt_cursor_fixture(artifact);
  ASSERT_TRUE(fixture.program_is_artifact_backed());
  auto& system = *fixture.system;
  ASSERT_NO_THROW(system.step(.01));
  const auto accepted = system.program_accepted_state();
  const auto revision = system.program_accepted_state_revision();
  const auto coarse = system.block_level_state_global("tracer", 0);
  const auto fine = system.block_level_state_global("tracer", 1);
  const auto last_dt = system.program_last_dt();
  std::uint64_t allocated = 1;
  for (const int failure : {4, 5}) {
    SCOPED_TRACE(failure);
    const bool retry = failure == 4;
    fixture.set_failure(failure);
    try {
      system.step(.01);
      FAIL() << "the rank-local typed body rejection was not surfaced";
    } catch (const pops::runtime::program::StepAttemptRejected& rejected) {
      EXPECT_EQ(rejected.status(),
                retry ? pops::SolveStatus::kIterationLimit : pops::SolveStatus::kInvalidEvaluation);
      EXPECT_EQ(rejected.disposition(),
                retry ? pops::runtime::program::StepAttemptDisposition::kRetry
                      : pops::runtime::program::StepAttemptDisposition::kReject);
      EXPECT_EQ(rejected.reason_code(), retry ? 0x41544352u : 0x41544354u);
      EXPECT_EQ(rejected.phase(), "attempt-cursor-body");
      EXPECT_EQ(rejected.detail(), retry ? "rank-zero-retry" : "rank-zero-terminal");
    }
    EXPECT_EQ(system.program_accepted_state(), accepted);
    EXPECT_EQ(system.program_accepted_state_revision(), revision);
    EXPECT_EQ(system.block_level_state_global("tracer", 0), coarse);
    EXPECT_EQ(system.block_level_state_global("tracer", 1), fine);
    EXPECT_EQ(system.macro_step(), 1);
    EXPECT_EQ(system.time(), .01);
    EXPECT_EQ(system.program_last_dt(), last_dt);
    EXPECT_EQ(fixture.accepted_attempt(), 1u);
    EXPECT_EQ(fixture.allocated_attempt(), ++allocated);
  }
  fixture.set_failure(0);
  ASSERT_NO_THROW(system.step(.01));
  EXPECT_EQ(fixture.accepted_attempt(), 4u);
  EXPECT_EQ(fixture.allocated_attempt(), 4u);
  EXPECT_EQ(system.macro_step(), 2);
  EXPECT_EQ(system.time(), .02);
  const auto next = pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(
      system.program_accepted_state());
  EXPECT_EQ(next.accepted_attempt, 4u);
  ASSERT_FALSE(next.accepted_face_flux[0].empty());
  for (const auto& axis : next.accepted_face_flux)
    for (const auto& fragment : axis)
      EXPECT_EQ(fragment.key.attempt, 4u);
}

TEST(test_amr_synthetic_program_loader_transaction,
     AttemptCursorRemainsMonotoneThroughRegridAndRegridOnRestart) {
  auto artifact = make_attempt_cursor_artifact(false);
  auto fixture = make_attempt_cursor_fixture(artifact);
  ASSERT_TRUE(fixture.program_is_artifact_backed());
  auto& system = *fixture.system;
  ASSERT_NO_THROW(system.step(.01));
  fixture.set_failure(1);
  EXPECT_THROW(system.step(.01), std::runtime_error);
  fixture.set_failure(0);
  const auto old_epoch = system.checkpoint_topology_epoch();
  // A real coverage expansion forces hierarchy and prepared-engine reconstruction.
  system.set_conservative_state("tracer", std::vector<double>(std::size_t{1} << (4 * Dim), 2.0));
  (void)system.execute_prepared_tagging(0);
  ASSERT_TRUE(system.regrid_from_prepared_tagging(0));
  EXPECT_GT(system.checkpoint_topology_epoch(), old_epoch);
  EXPECT_EQ(fixture.accepted_attempt(), 1u);
  EXPECT_EQ(fixture.allocated_attempt(), 2u);
  EXPECT_EQ(pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(
                system.program_accepted_state())
                .accepted_attempt,
            1u);
  ASSERT_NO_THROW(system.step(.01));
  EXPECT_EQ(fixture.accepted_attempt(), 3u);
  const auto before_restart_regrid = system.program_accepted_state();
  system.begin_restart_transaction();
  ASSERT_NO_THROW(system.restore_checkpoint_accepted_state(before_restart_regrid));
  ASSERT_NO_THROW(system.preflight_regrid_on_restart());
  ASSERT_NO_THROW(system.regrid_on_restart());
  system.commit_restart_transaction();
  system.finalize_restart_transaction();
  const auto cleared = pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(
      system.program_accepted_state());
  EXPECT_EQ(cleared.accepted_attempt, 3u);
  for (const auto& axis : cleared.accepted_face_flux)
    EXPECT_TRUE(axis.empty());
  EXPECT_TRUE(cleared.synchronization_events.empty());
  ASSERT_NO_THROW(system.step(.01));
  EXPECT_EQ(fixture.accepted_attempt(), 4u);
}

TEST(test_amr_synthetic_program_loader_transaction,
     AttemptCursorInvalidRestorePreservesFacadeAndLiveAuthority) {
  auto artifact = make_attempt_cursor_artifact(true);
  auto fixture = make_attempt_cursor_fixture(artifact);
  ASSERT_TRUE(fixture.program_is_artifact_backed());
  auto& system = *fixture.system;
  ASSERT_NO_THROW(system.step(.01));
  const auto accepted = system.program_accepted_state();
  const auto decoded =
      pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(accepted);
  ASSERT_TRUE(std::any_of(decoded.accepted_face_flux.begin(), decoded.accepted_face_flux.end(),
                          [](const auto& axis) { return !axis.empty(); }));
  const auto revision = system.program_accepted_state_revision();
  const auto coarse = system.block_level_state_global("tracer", 0);
  const auto fine = system.block_level_state_global("tracer", 1);
  const auto unchanged = [&] {
    EXPECT_EQ(system.program_accepted_state(), accepted);
    EXPECT_EQ(system.program_accepted_state_revision(), revision);
    EXPECT_EQ(system.block_level_state_global("tracer", 0), coarse);
    EXPECT_EQ(system.block_level_state_global("tracer", 1), fine);
    EXPECT_EQ(system.time(), .01);
    EXPECT_EQ(system.macro_step(), 1);
    EXPECT_EQ(fixture.accepted_attempt(), 1u);
    EXPECT_EQ(fixture.allocated_attempt(), 1u);
  };
  const std::size_t offset = 40u + decoded.spatial_contract.size();
  auto corrupt = accepted;
  std::fill(corrupt.begin() + offset, corrupt.begin() + offset + 8, std::uint8_t{0});
  EXPECT_THROW(system.restore_checkpoint_accepted_state(corrupt), std::invalid_argument);
  unchanged();
  auto legacy = accepted;
  legacy.erase(legacy.begin() + offset, legacy.begin() + offset + 8);
  legacy[7] = '7';
  if (pops::n_ranks() == 1)
    EXPECT_THROW(system.restore_checkpoint_accepted_state(legacy), std::invalid_argument);
  else
    EXPECT_THROW(system.restore_checkpoint_accepted_state(legacy), std::runtime_error);
  unchanged();
  if (pops::n_ranks() > 1) {
    auto divergent = decoded;
    if (pops::my_rank() == 0)
      divergent.accepted_attempt = 2;
    EXPECT_THROW(system.restore_checkpoint_accepted_state(
                     pops::runtime::program::serialize_amr_program_accepted_state(divergent)),
                 std::runtime_error);
    unchanged();
  }
  // This explicit policy erases the populated report, while preserving its committed authority.
  system.begin_restart_transaction();
  system.restore_checkpoint_accepted_state(accepted);
  system.preflight_regrid_on_restart();
  system.regrid_on_restart();
  system.commit_restart_transaction();
  system.finalize_restart_transaction();
  const auto cleared = pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(
      system.program_accepted_state());
  EXPECT_EQ(cleared.accepted_attempt, 1u);
  for (const auto& axis : cleared.accepted_face_flux)
    EXPECT_TRUE(axis.empty());
  EXPECT_EQ(fixture.allocated_attempt(), 1u);
  ASSERT_NO_THROW(system.step(.01));
  EXPECT_EQ(fixture.accepted_attempt(), 2u);
}
