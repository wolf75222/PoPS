#pragma once

#include <stdexcept>
#include <string>

namespace pops::test::synthetic_loader {

// The six compiled programs exercised by the nine independent transaction cases.
inline unsigned variant(bool interface_blocks, bool histories, bool attempt_cursor, bool with_flux,
                        bool mapping) {
  const unsigned result = unsigned(interface_blocks) | (unsigned(histories) << 1) |
                          (unsigned(attempt_cursor) << 2) | (unsigned(with_flux) << 3) |
                          (unsigned(mapping) << 4);
  switch (result) {
    case 4:
    case 8:
    case 9:
    case 10:
    case 12:
    case 28:
      return result;
    default:
      throw std::invalid_argument("unprepared synthetic loader fixture variant");
  }
}

inline std::string source(bool interface_blocks = false, bool histories = false,
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

}  // namespace pops::test::synthetic_loader
