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

std::string loader_source(bool interface_blocks = false, bool histories = false) {
  // clang-format off
  return std::string("#define POPS_TEST_INTERFACE_BLOCKS ") +
      (interface_blocks ? "1\n" : "0\n") + "#define POPS_TEST_HISTORIES " +
      (histories ? "1\n" : "0\n") + R"CPP(
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
    velocity[axis] = pops::Real(0.2) / pops::Real(axis + 1);
  pops_generated::Model model{
      pops::nd::ScalarAdvection<pops::kNativeDimension>::prepare(velocity), pops::Real(80)};
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
#if POPS_TEST_HISTORIES
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
extern "C" bool pops_program_has_flux_expression() { return true; }
extern "C" int pops_program_flux_expression_budget_count() { return pops_program_block_count(); }
extern "C" std::uint64_t pops_program_interface_coupling_application_bound() {
  return POPS_TEST_INTERFACE_BLOCKS ? UINT64_C(1) : UINT64_C(0);
}
extern "C" std::uint64_t pops_program_interface_coupling_identity_character_bound() {
  return POPS_TEST_INTERFACE_BLOCKS ? UINT64_C(128) : UINT64_C(0);
}
extern "C" std::uint64_t pops_program_flux_rhs_basis_bound(int block) {
  return block == 0 ? UINT64_C(10) : UINT64_C(0);
}
extern "C" std::uint64_t pops_program_flux_coefficient_term_bound(int block) {
  return block == 0 ? UINT64_C(1) : UINT64_C(0);
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
extern "C" int pops_program_checkpoint_logical_clock_count() { return 1; }
extern "C" const char* pops_program_checkpoint_logical_clock_identity(int clock) {
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

static bool reject_interface_refresh = false;
extern "C" void pops_test_reject_interface_refresh(bool reject) {
  reject_interface_refresh = reject;
}
extern "C" void pops_install_program_amr(
    pops::AmrSystem<pops::kNativeDimension>* system) {
  auto context = pops::runtime::program::make_program_execution_provider(system);
  auto inject_retry = std::make_shared<bool>(true);
  context->configure_primary_clock("tests.synthetic-loader.clock");
#if POPS_TEST_HISTORIES
  // Match the generated installer: materialize the frozen history descriptors before bind seals
  // checkpoint capacity, and refresh all level-qualified rings whenever the hierarchy changes.
  const auto register_histories = [context] {
    context->for_each_program_resource_level([&](int) {
      for (const char* name : {"tracer.first", "tracer.second"})
        context->register_history(name, 1, 1, 0, "tests.synthetic-loader/state/tracer",
                                  "cell.conservative", "tests.synthetic-loader.clock", "none");
    });
  };
  register_histories();
  context->install([context, register_histories](double dt) {
    register_histories();
    context->advance_hierarchy(dt, [context](double) {
      auto& accepted = context->state(0);
      auto& candidate = context->scratch_state(1000, 0, accepted);
      context->lincomb(candidate, pops::Real(2), accepted, pops::Real(0), accepted);
      context->store_history("tracer.first", accepted, 0);
      context->store_history("tracer.second", candidate, 0);
      context->rotate_histories("tests.synthetic-loader.clock");
      context->commit_many({{&accepted, &candidate}});
    });
  }, context, register_histories);
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
}
)CPP";
  // clang-format on
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
  const auto broadcast = [&](std::string& payload) {
    const bool length_overflow =
        lane->rank() == 0 &&
        payload.size() > static_cast<std::size_t>(std::numeric_limits<long>::max());
    if (pops::all_reduce_max(length_overflow ? 1L : 0L, *lane) != 0)
      throw std::length_error("AMR publication artifact exceeds the fixture length domain");
    const long count =
        pops::all_reduce_max(lane->rank() == 0 ? static_cast<long>(payload.size()) : 0L, *lane);
    long allocation_failed = 0;
    try {
      payload.resize(static_cast<std::size_t>(count));
    } catch (const std::exception&) {
      allocation_failed = 1;
    }
    if (pops::all_reduce_max(allocation_failed, *lane) != 0)
      throw std::runtime_error("AMR publication artifact allocation failed collectively");
    pops::broadcast_bytes_inplace(payload.data(), payload.size(), *lane, 0);
  };
  std::string image;
  std::string preparation_error;
  if (lane->rank() == 0) {
    try {
      {
        std::ofstream source(source_path);
        source.exceptions(std::ios::badbit | std::ios::failbit);
        source << loader_source(true);
      }
      const auto package = pops::test::native_dso::compile_shared(
          source_path, shared_object, "-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI");
      if (!package.ok) {
        pops::test::native_dso::report_compile_failure("test_amr_interface_publication", package);
        throw std::runtime_error(
            "authenticated two-block AMR publication artifact did not compile");
      }
      std::ifstream binary(shared_object, std::ios::binary);
      binary.exceptions(std::ios::badbit);
      if (!binary)
        throw std::runtime_error("cannot read the compiled AMR publication artifact");
      image.assign(std::istreambuf_iterator<char>(binary), std::istreambuf_iterator<char>());
    } catch (const std::exception& error) {
      preparation_error = error.what();
    }
  }
  broadcast(preparation_error);
  ASSERT_TRUE(preparation_error.empty()) << preparation_error;
  // Independent links can carry different UUIDs. Every rank authenticates and loads the exact
  // rank-zero binary image, even when its local artifact path differs.
  broadcast(image);
  std::unique_ptr<pops::dynlib::AuthenticatedNativeFile> authenticated;
  try {
    if (lane->rank() != 0) {
      std::ofstream binary(shared_object, std::ios::binary);
      binary.exceptions(std::ios::badbit | std::ios::failbit);
      binary.write(image.data(), static_cast<std::streamsize>(image.size()));
    }
    authenticated = std::make_unique<pops::dynlib::AuthenticatedNativeFile>(shared_object);
  } catch (const std::exception& error) {
    preparation_error = error.what();
  }
  ASSERT_EQ(pops::all_reduce_max(preparation_error.empty() ? 0L : 1L, *lane), 0L)
      << preparation_error;
  ASSERT_TRUE(pops::all_ranks_agree_exact_ordered_byte_pairs(
      {{"AMR publication artifact", authenticated->content_sha256()}}, *lane));
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
  {
    std::ofstream source(source_path);
    source << loader_source();
  }
  const auto package = pops::test::native_dso::compile_shared(
      source_path, shared_object, "-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI");
  if (!package.ok) {
    pops::test::native_dso::report_compile_failure("test_amr_synthetic_program_loader_transaction",
                                                   package);
    FAIL() << "synthetic source-built AMR loader transaction artifact did not compile";
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
  {
    std::ofstream source(source_path);
    source << loader_source(false, true);
  }
  const auto package = pops::test::native_dso::compile_shared(
      source_path, shared_object, "-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI");
  if (!package.ok) {
    pops::test::native_dso::report_compile_failure("test_amr_history_restart", package);
    FAIL() << "history restart fixture artifact did not compile";
  }
  const auto settings = config();
  pops::AmrSystem<Dim> system(settings);
  {
    SCOPED_TRACE("install, bootstrap, and seal frozen history capacity");
    ASSERT_NO_THROW(
        build_refined_system(system, shared_object, initial_state(settings.shape), true));
  }
  ASSERT_EQ(system.installed_program_hash(), "tests.synthetic-loader/program/history-restart-v1");
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
  constexpr double first_dt = 0.125;
  constexpr double second_dt = 0.1875;
  Image checkpoint;
  {
    SCOPED_TRACE("first accepted step and complete checkpoint capture");
    ASSERT_NO_THROW(system.step(first_dt));
    ASSERT_EQ(system.history_names(), names);
    ASSERT_NO_THROW(checkpoint = capture());
  }
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
        // All metadata is authentic. One rank omits one numeric write in the final ring, which
        // must not become accepted just because its freshly allocated values happen to be finite.
        if (omit_one_payload && pops::my_rank() == 0 && history.name == names.back() &&
            history.level == 1 && slot == 1)
          continue;
        system.restore_history(history.name, history.level, slot, history.values.at(slot));
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
    EXPECT_EQ(capture(), uninterrupted);
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
