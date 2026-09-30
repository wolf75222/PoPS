// Compiled time-program runtime seam (epic ADC-399 / ADC-401 Phase 2b): a Forward-Euler Program,
// installed as a macro-step closure via pops::runtime::program::ProgramContext, runs C++-side during
// sim.step(dt). This test proves the seam end-to-end WITHOUT codegen or a .so: it builds the closure
// in C++ (the role the generated problem.so will later fill) and checks bit-parity against a reference
// Forward-Euler step computed from the SAME existing primitives (solve_fields + eval_rhs + U + dt*R).
//
// Model: a compressible Euler gas with a NON-UNIFORM pressure IC (u = v = 0), so -div F has a non-zero
// momentum component -> the step actually changes the state (parity is not vacuous). No source, no
// charge (NoEll), so the result is pure gas dynamics and deterministic across two System instances.

#include <gtest/gtest.h>

#include <pops/mesh/storage/multifab.hpp>
#include <pops/physics/bricks/source.hpp>                // NoSource
#include <pops/physics/composition/composite.hpp>        // CompositeModel
#include <pops/physics/fluids/euler.hpp>                 // Euler
#include <pops/runtime/builders/compiled/dsl_block.hpp>  // add_compiled_model
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/program/program_context.hpp>  // ProgramContext (the seam under test)
#include <pops/runtime/program/program_runtime_state.hpp>
#include <pops/runtime/program/step_transaction.hpp>
#include <pops/runtime/system.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// M28 fixtures below use the existing native Program/System transaction; no
// alternate solver or host state dictionary substitutes for these buffers.

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace pops {

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

#if defined(POPS_HAS_KOKKOS)
static void ensure_kokkos() {
  static Kokkos::ScopeGuard guard;
  (void)guard;
}
#endif

static void install_execution_lane(System<kNativeDimension>& system, std::string identity) {
  system.install_prepared_boundary_execution_lane(std::make_shared<ExecutionLane>(
      ExecutionLane::duplicate_world_collectively(std::move(identity))));
}

static constexpr int kGasComponents = kNativeDimension + 2;

template <int Dim>
static std::size_t exact_cell_count(int cells_per_axis) {
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    cells *= static_cast<std::size_t>(cells_per_axis);
  return cells;
}

// Elliptic brick that contributes nothing (no charge): the Poisson RHS stays zero, phi = 0, and the
// Euler flux has no provider requirements here, so the residual is pure gas dynamics.
struct NoEll {
  template <class State>
  POPS_HD Real rhs(const State&) const {
    return Real(0);
  }
};
using GasModel = CompositeModel<EulerND<kNativeDimension>, NoSource, NoEll>;

// Deliberately select storage only even though the reusable physical state type
// also has gas conversions. No spatial or elliptic law is selected by this route.
struct ProgramStorageGas : GasModel {
  static constexpr bool program_only_storage = true;
  static constexpr int program_state_ghost_depth = 1;
};

struct UnitDensitySource {
  template <class State, class Providers>
  POPS_HD State apply(const State&, const Providers&) const {
    State source{};
    source[0] = Real(1);
    return source;
  }
};
using SourcedGasModel = CompositeModel<EulerND<kNativeDimension>, UnitDensitySource, NoEll>;

struct DrainingDensitySource {
  template <class State, class Providers>
  POPS_HD State apply(const State&, const Providers&) const {
    State source{};
    source[0] = Real(-1);
    return source;
  }
};
using DrainingGasModel = CompositeModel<EulerND<kNativeDimension>, DrainingDensitySource, NoEll>;

struct ProjectingEuler : EulerND<kNativeDimension> {
  template <class Providers>
  POPS_HD State project(const State& input, const Providers&) const {
    State output = input;
    output[0] = Real(2);
    return output;
  }
};
using ProjectingGasModel = CompositeModel<ProjectingEuler, NoSource, NoEll>;

struct DiffusiveGasModel : GasModel {
  POPS_HD Real diffusivity() const { return Real(0.1); }
};

template <int Dim>
SystemConfig<Dim> unit_domain_config(int cells_per_axis) {
  // SystemConfig snapshots the rank space in its defaults. This helper must
  // also work when its test runs first in an MPI-filtered binary.
  comm_init();
  SystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = cells_per_axis;
    config.lower[axis] = Real(0);
    config.upper[axis] = Real(1);
    config.periodicity[axis] = true;
  }
  return config;
}

template <int Dim>
SystemConfig<Dim> distributed_boundary_domain_config(int cells_per_axis) {
  // Establish the execution world before querying its rank count or constructing
  // a configuration whose defaults depend on it. This fixture can run first in
  // an MPI-filtered binary, without a preceding System having initialized MPI.
  comm_init();
  SystemConfig<Dim> config = unit_domain_config<Dim>(cells_per_axis);
  const int ranks = n_ranks();
  if (ranks == 1)
    return config;
  if (cells_per_axis % ranks != 0)
    throw std::invalid_argument("prepared boundary test cells must partition the MPI rank count");
  config.boxes.clear();
  for (int rank = 0; rank < ranks; ++rank) {
    Index<Dim> lower{};
    Index<Dim> upper{};
    lower[0] = rank * cells_per_axis / ranks;
    upper[0] = (rank + 1) * cells_per_axis / ranks - 1;
    for (int axis = 1; axis < Dim; ++axis)
      upper[axis] = cells_per_axis - 1;
    config.boxes.emplace_back(lower, upper);
  }
  return config;
}

// Exact-rank implicit ball used by every EB runtime proof below.  It consumes each coordinate of
// the native dimension, so the test geometry is a LevelSet authority rather than a legacy 2-D disc.
static void install_centered_ball(System<kNativeDimension>& system, double radius,
                                  const std::string& mode, double kappa_min = 0.0,
                                  double face_open_eps = 0.0, double cut_theta_min = 0.0) {
  static constexpr const char* coordinates[] = {"x", "y", "z"};
  std::vector<std::string> opcodes;
  std::vector<double> literals;
  for (int axis = 0; axis < kNativeDimension; ++axis) {
    opcodes.emplace_back(coordinates[axis]);
    literals.push_back(0.0);
    opcodes.emplace_back("constant");
    literals.push_back(0.5);
    opcodes.emplace_back("sub");
    literals.push_back(0.0);
    if (axis > 0) {
      opcodes.emplace_back("hypot");
      literals.push_back(0.0);
    }
  }
  opcodes.emplace_back("constant");
  literals.push_back(radius);
  opcodes.emplace_back("sub");
  literals.push_back(0.0);
  system.set_analytic_level_set(opcodes, literals, mode, kappa_min, face_open_eps, cut_theta_min);
}

template <int Dim>
struct SetProjectedDensity {
  FieldView<Real, Dim> state{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    state(index, 0) = Real(2);
    if (!(state(index, Dim + 1) > Real(0)))
      state(index, Dim + 1) = Real(5);
  }
};

template <int Dim>
struct SetBoundaryJvpDirection {
  FieldView<Real, Dim> field{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    field(index, 0) = Real(0.125);
    field(index, Dim + 1) = Real(0.25);
  }
};

template <int Dim>
struct SetProjectedActiveDensity {
  FieldView<Real, Dim> state{};
  FieldView<const Real, Dim> active{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    if (active(index) >= Real(0.5)) {
      state(index, 0) = Real(2);
      if (!(state(index, Dim + 1) > Real(0)))
        state(index, Dim + 1) = Real(5);
    }
  }
};

template <int Dim>
void project_density_exact(MultiFab<Dim>& state) {
  for (std::size_t local = 0; local < state.local_size(); ++local)
    for_each_cell(state.box(local), SetProjectedDensity<Dim>{state.fab(local).view()});
  device_fence();
}

template <int Dim>
void set_boundary_jvp_direction(MultiFab<Dim>& direction) {
  direction.set_val(Real(0));
  for (std::size_t local = 0; local < direction.local_size(); ++local)
    for_each_cell(direction.box(local), SetBoundaryJvpDirection<Dim>{direction.fab(local).view()});
  device_fence();
}

template <int Dim>
Real norm_inf_all_components(const MultiFab<Dim>& field) {
  Real maximum = Real(0);
  for (int component = 0; component < field.ncomp(); ++component)
    maximum = std::max(maximum, pops::reduce_norm_inf(field, component));
  return maximum;
}

template <int Dim>
void project_density_exact(MultiFab<Dim>& state,
                           const runtime::system::PreparedEmbeddedBoundaryGeometry<Dim>& embedded) {
  const MultiFab<Dim>& active = embedded.active_mask();
  if (state.layout() != active.layout() || state.distribution() != active.distribution() ||
      state.local_rank() != active.local_rank() || state.local_size() != active.local_size())
    throw std::invalid_argument("test projection provider received a mismatched active mask");
  for (std::size_t local = 0; local < state.local_size(); ++local)
    for_each_cell(state.box(local), SetProjectedActiveDensity<Dim>{state.fab(local).view(),
                                                                   active.fab(local).view()});
  device_fence();
}

template <class Context, class Field>
concept HasUnqualifiedBoundaryLinearization =
    requires(Context& context, const runtime::multiblock::BoundaryEvaluationPoint& point,
             Field& state, Field& output) {
      context.boundary_residual_into_at(point, 0, state, output);
      context.boundary_jvp_into_at(point, 0, state, output, output);
    };

static void fill_ic(std::vector<double>& U, int n, double gamma) {
  const std::size_t cells = exact_cell_count<kNativeDimension>(n);
  const double pi = 3.14159265358979323846;
  U.assign(static_cast<std::size_t>(kGasComponents) * cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    std::size_t coordinate_index = cell;
    double pressure_mode = 1.0;
    for (int axis = 0; axis < kNativeDimension; ++axis) {
      const int coordinate = static_cast<int>(coordinate_index % static_cast<std::size_t>(n));
      coordinate_index /= static_cast<std::size_t>(n);
      pressure_mode *= std::cos(2 * pi * (static_cast<double>(coordinate) + 0.5) / n);
    }
    U[cell] = 1.0;
    U[static_cast<std::size_t>(kNativeDimension + 1) * cells + cell] =
        (3.0 + 0.5 * pressure_mode) / (gamma - 1.0);
  }
}

template <int Dim>
static void fill_boundary_euler_ic(std::vector<double>& state, int cells_per_axis, double gamma) {
  const std::size_t cells = exact_cell_count<Dim>(cells_per_axis);
  state.assign(static_cast<std::size_t>(Dim + 2) * cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const double rho =
        1.0 + 0.01 * static_cast<double>(cell % static_cast<std::size_t>(cells_per_axis));
    state[cell] = rho;
    for (int axis = 0; axis < Dim; ++axis) {
      state[static_cast<std::size_t>(axis + 1) * cells + cell] = 0.0;
    }
    const double pressure = 3.0 + 0.1 * static_cast<double>(cell % 7u);
    state[static_cast<std::size_t>(Dim + 1) * cells + cell] = pressure / (gamma - 1.0);
  }
}

static void add_gas(System<kNativeDimension>& s, double gamma,
                    const std::string& limiter = "minmod") {
  s.install_block_state_route("gas", "test.program-runtime.gas.state@1");
  s.seal_auxiliary_providers();
  GasModel model;
  model.hyp = EulerND<kNativeDimension>{gamma};
  add_compiled_model(s, "gas", model, limiter, "rusanov", "conservative", "explicit", gamma);
  s.set_poisson("charge_density", "cartesian_cg");
}

static void add_boundary_gas(System<kNativeDimension>& system, double gamma) {
  system.install_block_state_route("gas", "test.program-runtime.boundary-gas.state@1");
  std::vector<std::string> face_types(static_cast<std::size_t>(2 * kNativeDimension), "no_flux");
  std::vector<std::string> face_identities;
  face_identities.reserve(static_cast<std::size_t>(2 * kNativeDimension));
  for (int face = 0; face < 2 * kNativeDimension; ++face)
    face_identities.push_back("test:program-runtime/no-flux-face-" + std::to_string(face));
  std::vector<std::string> roles{"density"};
  for (int axis = 0; axis < kNativeDimension; ++axis)
    roles.push_back("momentum:" + std::to_string(axis));
  roles.push_back("energy");
  system.install_hyperbolic_boundary(
      "gas", "test:program-runtime/prepared-boundary@1", 1, face_types,
      std::vector<double>(roles.size() * static_cast<std::size_t>(2 * kNativeDimension), 0.0),
      face_identities, roles, "test.program-runtime.boundary-gas.state@1");
  system.seal_auxiliary_providers();
  GasModel model;
  model.hyp = EulerND<kNativeDimension>{gamma};
  add_compiled_model(system, "gas", model, "none", "rusanov", "conservative", "explicit", gamma);
}

static void add_sourced_gas(System<kNativeDimension>& system, double gamma) {
  system.install_block_state_route("gas", "test.program-runtime.sourced-gas.state@1");
  system.seal_auxiliary_providers();
  SourcedGasModel model;
  model.hyp = EulerND<kNativeDimension>{gamma};
  add_compiled_model(system, "gas", model, "none", "rusanov", "conservative", "explicit", gamma);
}

static void add_draining_gas(System<kNativeDimension>& system, const std::string& name,
                             double gamma) {
  system.install_block_state_route(name, "test.program-runtime." + name + ".state@1");
  system.seal_auxiliary_providers();
  DrainingGasModel model;
  model.hyp = EulerND<kNativeDimension>{gamma};
  add_compiled_model(system, name, model, "none", "rusanov", "conservative", "explicit", gamma);
}

static void add_projecting_gas(System<kNativeDimension>& system, double gamma,
                               int* projection_calls = nullptr) {
  system.install_block_state_route("gas", "test.program-runtime.projecting-gas.state@1");
  system.seal_auxiliary_providers();
  ProjectingEuler transport;
  transport.gamma = gamma;
  ProjectingGasModel model;
  model.hyp = transport;
  auto prepared = prepare_compiled_system_block<kNativeDimension>(
      system, "gas", model, "none", "rusanov", "conservative", "explicit", gamma,
      /*substeps=*/1, /*evolve=*/true, /*stride=*/1);
  prepared.provider_identity += "/test-projection@1";
  prepared.closures.project = [projection_calls](MultiFab<kNativeDimension>& state,
                                                 const ExecutionLane&) {
    if (projection_calls != nullptr)
      ++*projection_calls;
    project_density_exact(state);
  };
  prepared.closures.project_masked = prepared.closures.project;
  const auto embedded_projection =
      [](MultiFab<kNativeDimension>& state,
         const runtime::system::PreparedEmbeddedBoundaryGeometry<kNativeDimension>& embedded,
         const ExecutionLane&) { project_density_exact(state, embedded); };
  prepared.closures.staircase.project = embedded_projection;
  prepared.closures.cut_cell.project = embedded_projection;
  install_prepared_block(system, std::move(prepared));
}

static void add_generated_projecting_gas(System<kNativeDimension>& system, double gamma) {
  system.install_block_state_route("gas", "test.program-runtime.generated-projecting-gas.state@1");
  system.seal_auxiliary_providers();
  ProjectingEuler transport;
  transport.gamma = gamma;
  ProjectingGasModel model;
  model.hyp = transport;
  add_compiled_model(system, "gas", model, "none", "rusanov", "conservative", "explicit", gamma);
}

struct ConditionalFiniteProjectingEuler : EulerND<kNativeDimension> {
  template <class Providers>
  POPS_HD State project(const State& input, const Providers&) const {
    State output = input;
    if (!(input[0] > Real(0)))
      output[0] = std::numeric_limits<Real>::quiet_NaN();
    else
      output[0] = Real(2);
    return output;
  }
};
using ConditionalFiniteProjectingGasModel =
    CompositeModel<ConditionalFiniteProjectingEuler, NoSource, NoEll>;

static void add_generated_conditional_projecting_gas(System<kNativeDimension>& system,
                                                     double gamma) {
  system.install_block_state_route(
      "gas", "test.program-runtime.generated-conditional-projecting-gas.state@1");
  system.seal_auxiliary_providers();
  ConditionalFiniteProjectingEuler transport;
  transport.gamma = gamma;
  ConditionalFiniteProjectingGasModel model;
  model.hyp = transport;
  add_compiled_model(system, "gas", model, "none", "rusanov", "conservative", "explicit", gamma);
}

static void add_diffusive_gas(System<kNativeDimension>& system, double gamma) {
  system.install_block_state_route("gas", "test.program-runtime.diffusive-gas.state@1");
  system.seal_auxiliary_providers();
  DiffusiveGasModel model;
  model.hyp.gamma = gamma;
  add_compiled_model(system, "gas", model, "none", "rusanov", "conservative", "explicit", gamma);
}

static void add_scalar(System<kNativeDimension>& system) {
  system.install_block_state_route("tracer", "test.program-runtime.tracer.state@1");
  system.seal_auxiliary_providers();
  add_compiled_model(system, "tracer", nd::ScalarAdvection<kNativeDimension>{}, "none", "rusanov",
                     "conservative", "explicit");
}

TEST(ProgramRuntime, BalanceDueWindowUsesTheOuterAcceptedStepAndCleansUpOnFailure) {
  runtime::program::ProgramRuntimeState<kNativeDimension> state;
  const std::string contract = "pops.balance-due-contract.v1:sha256:" + std::string(64, '1');
  const std::string route = "pops.balance-ledger-route.v1:sha256:" + std::string(64, '2');

  EXPECT_THROW((void)state.balance_consumer_is_due(contract, route, 3, "test"), std::logic_error);
  state.run_balance_due_window(2, "test", [&] {
    EXPECT_TRUE(state.balance_consumer_is_due(contract, route, 3, "test"));
    EXPECT_FALSE(state.balance_consumer_is_due(contract, route, 2, "test"));
    EXPECT_THROW((void)state.balance_consumer_is_due(contract, route, 0, "test"),
                 std::invalid_argument);
    EXPECT_THROW((void)state.balance_consumer_is_due("forged", route, 3, "test"),
                 std::invalid_argument);
  });
  EXPECT_THROW((void)state.balance_consumer_is_due(contract, route, 3, "test"), std::logic_error);

  EXPECT_THROW(
      state.run_balance_due_window(3, "test", [] { throw std::runtime_error("attempt rejected"); }),
      std::runtime_error);
  EXPECT_THROW((void)state.balance_consumer_is_due(contract, route, 4, "test"), std::logic_error);
}

TEST(ProgramRuntime, AutomaticBalanceDueMarkerIsAttemptLocalMonotoneAndReplaySafe) {
  runtime::program::ProgramRuntimeState<kNativeDimension> state;

  EXPECT_FALSE(state.automatic_balance_capture_due());
  EXPECT_THROW(state.note_automatic_balance_capture_due(true, "test"), std::logic_error);
  state.run_balance_due_window(0, "test", [&] {
    state.note_automatic_balance_capture_due(false, "test");
    EXPECT_FALSE(state.automatic_balance_capture_due());
    state.note_automatic_balance_capture_due(true, "test");
    EXPECT_TRUE(state.automatic_balance_capture_due());
    state.note_automatic_balance_capture_due(false, "test");
    EXPECT_TRUE(state.automatic_balance_capture_due());
  });
  EXPECT_TRUE(state.automatic_balance_capture_due());

  state.begin_step_projection_report();
  EXPECT_FALSE(state.automatic_balance_capture_due());
  state.run_balance_replay("test", [&] {
    state.note_automatic_balance_capture_due(false, "test");
    EXPECT_FALSE(state.automatic_balance_capture_due());
    EXPECT_THROW(state.note_automatic_balance_capture_due(true, "test"), std::logic_error);
  });
  EXPECT_FALSE(state.automatic_balance_capture_due());
}

TEST(ProgramRuntime, SelectedAutomaticBalanceTermsRequireCompleteQualifiedEvidence) {
  runtime::program::ProgramRuntimeState<kNativeDimension> state;
  const std::string route = "pops.balance-ledger-route.v1:sha256:" + std::string(64, '5');
  state.begin_step_projection_report();
  state.run_balance_due_window(0, "test", [&] {
    state.note_automatic_balance_capture_due(true, "test");
    state.record_balance_term(route, "storage_change", 1.0, "test");
    state.record_balance_term(route, "outward_boundary_flux", 2.0, "test");
    state.record_balance_term(route, "sources", 3.0, "test");
    state.record_automatic_balance_term(2, 0, 1, "projection", 0.25, "test");
    state.record_automatic_balance_term(2, 1, 1, "projection", 0.75, "test");
    state.record_automatic_balance_term(2, 0, 1, "reflux", 0.5, "test");
  });
  state.complete_balance_step(true);

  const auto selected =
      state.selected_accepted_balance_terms(route, 2, 1, {0, 1}, {"projection", "reflux"}, "test");
  EXPECT_EQ(selected.at("storage_change"), 1.0);
  EXPECT_EQ(selected.at("outward_boundary_flux"), 2.0);
  EXPECT_EQ(selected.at("sources"), 3.0);
  EXPECT_EQ(selected.at("projection"), 1.0);
  EXPECT_EQ(selected.at("reflux"), 0.5);

  EXPECT_THROW((void)state.selected_accepted_balance_terms(route, 2, 1, {0, 1, 2},
                                                           {"projection", "reflux"}, "test"),
               std::runtime_error);
  EXPECT_THROW((void)state.selected_accepted_balance_terms(route, 2, 1, {0, 2},
                                                           {"projection", "reflux"}, "test"),
               std::invalid_argument);
}

TEST(ProgramRuntime, SelectiveReplayCompilesBalanceOffAndRestoresTheGuard) {
  runtime::program::ProgramRuntimeState<kNativeDimension> state;
  const std::string contract = "pops.balance-due-contract.v1:sha256:" + std::string(64, '3');
  const std::string route = "pops.balance-ledger-route.v1:sha256:" + std::string(64, '4');

  EXPECT_THROW((void)state.balance_consumer_is_due(contract, route, 2, "test"), std::logic_error);
  state.run_balance_replay("test", [&] {
    EXPECT_FALSE(state.balance_consumer_is_due(contract, route, 2, "test"));
    EXPECT_THROW((void)state.balance_consumer_is_due("forged", route, 2, "test"),
                 std::invalid_argument);
    EXPECT_THROW((void)state.balance_consumer_is_due(contract, route, 0, "test"),
                 std::invalid_argument);
    EXPECT_THROW(state.run_balance_replay("nested", [] {}), std::logic_error);
    EXPECT_THROW(state.run_balance_due_window(1, "nested", [] {}), std::logic_error);
  });
  EXPECT_THROW((void)state.balance_consumer_is_due(contract, route, 2, "test"), std::logic_error);
  state.run_balance_due_window(1, "test", [&] {
    EXPECT_THROW(state.run_balance_replay("window", [] {}), std::logic_error);
  });

  EXPECT_THROW(state.run_balance_replay("test", [] { throw std::runtime_error("replay failed"); }),
               std::runtime_error);
  EXPECT_THROW((void)state.balance_consumer_is_due(contract, route, 2, "test"), std::logic_error);
}

TEST(ProgramRuntime, ReplayAuthorityRequiresAnArtifactAndAnExactRingDepthPair) {
  runtime::program::ProgramRuntimeState<kNativeDimension> state;
  state.history_replay_authorities_ = {{"gas.previous", 3}};

  EXPECT_FALSE(state.authorizes_history_replay("gas.previous", 3))
      << "a table without an authenticated artifact marker is not replay authority";

  state.artifact_backed_ = true;
  state.operator_authorities_ = {{{1, 2, 3, 4}}};
  state.installed_hash_ = "old-artifact";
  state.block_map_ = {0};
  state.seed_params(0, {2.0});
  state.dt_bound_ = [](Real cfl) { return cfl; };
  EXPECT_TRUE(state.authorizes_history_replay("gas.previous", 3));
  EXPECT_FALSE(state.authorizes_history_replay("gas.previous", 2));
  EXPECT_FALSE(state.authorizes_history_replay("other.previous", 3));

  state.install_unverified_step([](double) {});
  EXPECT_FALSE(state.artifact_backed_);
  EXPECT_TRUE(state.operator_authorities_.empty());
  EXPECT_TRUE(state.history_replay_authorities_.empty());
  EXPECT_TRUE(state.installed_hash_.empty());
  EXPECT_TRUE(state.block_map_.empty());
  EXPECT_TRUE(state.block_params_.empty());
  EXPECT_FALSE(state.dt_bound_);
  EXPECT_FALSE(state.restart_regrid_preflight_);
  EXPECT_FALSE(state.restart_regrid_);
  EXPECT_FALSE(state.restart_resync_);
  EXPECT_FALSE(state.authorizes_history_replay("gas.previous", 3))
      << "a direct native step must revoke every earlier artifact authority";
}

TEST(ProgramRuntime, ArtifactStepInstallRequiresOneNewStepAndRollsBackExactly) {
  struct AcceptedContextSnapshot final : runtime::program::AcceptedProgramContextSnapshot {
    std::unique_ptr<runtime::program::AcceptedProgramContextSnapshot> prepare_restore()
        const override {
      return std::make_unique<AcceptedContextSnapshot>();
    }
    void publish_restore() noexcept override {}
  };
  runtime::program::ProgramRuntimeState<kNativeDimension> state;
  int old_steps = 0;
  int new_steps = 0;
  int restart_preflights = 0;
  int restart_regrids = 0;
  int restart_resyncs = 0;
  state.install_unverified_step([&](double) { ++old_steps; });
  state.install_restart_hooks([&] { ++restart_preflights; }, [&] { ++restart_regrids; },
                              [&] { ++restart_resyncs; },
                              [] { return std::make_unique<AcceptedContextSnapshot>(); }, "test");
  state.operator_authorities_ = {{{1, 2, 3, 4}}};
  state.history_replay_authorities_ = {{"gas.previous", 3}};
  state.installed_hash_ = "accepted-artifact";
  state.block_map_ = {2};
  state.seed_params(0, {3.0});
  state.dt_bound_ = [](Real cfl) { return Real(0.5) * cfl; };
  state.artifact_backed_ = true;
  runtime::program::MovingIntervalGeometry<kNativeDimension> installed_geometry;
  installed_geometry.physical_frame = "accepted-frame";
  installed_geometry.generation = 17;
  state.moving_interval_geometry_.emplace("accepted-mesh",std::move(installed_geometry));
  const auto accepted_generation = state.step_install_generation_;

  auto interrupted = state.capture_artifact_step_install();
  state.reset_artifact_candidate_state();
  EXPECT_TRUE(state.moving_interval_geometry_.empty());
  state.operator_authorities_ = {{{9, 8, 7, 6}}};
  state.install_unverified_step([&](double) { ++new_steps; });
  state.rollback_artifact_step_install(std::move(interrupted));
  state.step_(0.1);
  EXPECT_EQ(old_steps, 1);
  EXPECT_EQ(new_steps, 0);
  EXPECT_EQ(state.step_install_generation_, accepted_generation);
  EXPECT_EQ(state.operator_authorities_,
            (std::vector<std::array<std::uint64_t, 4>>{{{1, 2, 3, 4}}}));
  EXPECT_EQ(state.installed_hash_, "accepted-artifact");
  EXPECT_EQ(state.block_map_, (std::vector<int>{2}));
  ASSERT_EQ(state.moving_interval_geometry_.size(),1u);
  EXPECT_EQ(state.moving_interval_geometry_.at("accepted-mesh").physical_frame,"accepted-frame");
  EXPECT_EQ(state.moving_interval_geometry_.at("accepted-mesh").generation,17u);
  EXPECT_EQ(state.block_params_.size(), 1u);
  ASSERT_TRUE(state.dt_bound_);
  EXPECT_DOUBLE_EQ(state.dt_bound_(0.4), 0.2);
  EXPECT_TRUE(state.authorizes_history_replay("gas.previous", 3));
  EXPECT_NO_THROW(state.preflight_regrid_on_restart("test"));
  EXPECT_NO_THROW(state.regrid_on_restart("test"));
  EXPECT_NO_THROW(state.resync_after_restart("test"));
  EXPECT_EQ(restart_preflights, 1);
  EXPECT_EQ(restart_regrids, 1);
  EXPECT_EQ(restart_resyncs, 1);

  const auto no_op = state.capture_artifact_step_install();
  EXPECT_THROW(state.require_exact_artifact_step_install(no_op, "test"), std::runtime_error);

  auto exact = state.capture_artifact_step_install();
  state.install_unverified_step([&](double) { ++new_steps; });
  EXPECT_NO_THROW(state.require_exact_artifact_step_install(exact, "test"));
  state.rollback_artifact_step_install(std::move(exact));
  EXPECT_TRUE(state.authorizes_history_replay("gas.previous", 3));

  auto duplicate = state.capture_artifact_step_install();
  state.install_unverified_step([&](double) { ++new_steps; });
  state.install_unverified_step([&](double) { ++new_steps; });
  EXPECT_THROW(state.require_exact_artifact_step_install(duplicate, "test"), std::runtime_error);
  state.rollback_artifact_step_install(std::move(duplicate));
  EXPECT_TRUE(state.authorizes_history_replay("gas.previous", 3));

  const auto generation_before_empty = state.step_install_generation_;
  EXPECT_THROW(state.install_unverified_step({}), std::invalid_argument);
  EXPECT_EQ(state.step_install_generation_, generation_before_empty);
  EXPECT_TRUE(state.authorizes_history_replay("gas.previous", 3));
}

TEST(ProgramRuntime, FacadeTemporalOperationsRequireProgramBeforeMutation) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  System<kNativeDimension> system(unit_domain_config<kNativeDimension>(8));
  install_execution_lane(system, "pops.test.program-runtime.program-required");
  system.enable_profiling();
  const double initial_time = system.time();
  const int initial_step = system.macro_step();
  const std::string initial_profile = system.profile_report();

  const auto expect_program_required = [&](auto&& operation, const char* name) {
    try {
      operation();
      ADD_FAILURE() << name << " accepted a program-less temporal operation";
    } catch (const std::logic_error& error) {
      EXPECT_NE(std::string(error.what()).find(name), std::string::npos);
      EXPECT_NE(std::string(error.what()).find("installed whole-system Program"),
                std::string::npos);
    }
    EXPECT_DOUBLE_EQ(system.time(), initial_time);
    EXPECT_EQ(system.macro_step(), initial_step);
    EXPECT_EQ(system.profile_report(), initial_profile);
  };

  expect_program_required([&] { system.step(0.01); }, "System::step");
  expect_program_required([&] { system.advance(0.01, 0); }, "System::advance");
  expect_program_required([&] { (void)system.step_cfl(0.4); }, "System::step_cfl");
}

TEST(ProgramRuntime, GlobalCadencePublishesExactSubstepAndStrideWindowTimes) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  auto config = unit_domain_config<kNativeDimension>(4);

  System<kNativeDimension> subcycled(config);
  subcycled.set_clock(1.0, 10);
  runtime::program::ProgramContext subcycled_context(&subcycled);
  std::vector<double> subcycled_times;
  std::vector<int> subcycled_macro_steps;
  subcycled_context.install([&](double) {
    subcycled_times.push_back(static_cast<double>(subcycled_context.physical_time()));
    subcycled_macro_steps.push_back(subcycled.macro_step());
  });
  subcycled.set_program_cadence(/*substeps=*/2, /*stride=*/1);
  subcycled.step(0.2);

  ASSERT_EQ(subcycled_times.size(), 2u);
  EXPECT_NEAR(subcycled_times[0], 1.0, 1.0e-14);
  EXPECT_NEAR(subcycled_times[1], 1.1, 1.0e-14);
  EXPECT_EQ(subcycled_macro_steps, (std::vector<int>{10, 10}));
  EXPECT_NEAR(subcycled.time(), 1.2, 1.0e-14);
  EXPECT_EQ(subcycled.macro_step(), 11);

  System<kNativeDimension> catchup(config);
  runtime::program::ProgramContext catchup_context(&catchup);
  catchup_context.configure_primary_clock("macro");
  std::vector<double> catchup_times;
  std::vector<double> catchup_steps;
  std::vector<int> catchup_macro_steps;
  std::vector<bool> catchup_every_one_due;
  catchup_context.install([&](double h) {
    catchup_times.push_back(static_cast<double>(catchup_context.physical_time()));
    catchup_steps.push_back(h);
    catchup_macro_steps.push_back(catchup.macro_step());
    catchup_every_one_due.push_back(catchup_context.schedule_is_due(
        41, 1, runtime::program::ScheduleDomainKind::kAcceptedStep, "macro", "", -1));
  });
  catchup.set_program_cadence(/*substeps=*/3, /*stride=*/2);

  catchup.step(0.1);
  EXPECT_TRUE(catchup_times.empty());
  EXPECT_NEAR(catchup.time(), 0.1, 1.0e-14);
  EXPECT_EQ(catchup.macro_step(), 1);
  EXPECT_DOUBLE_EQ(catchup.program_cadence_window_dt(), 0.1);
  EXPECT_EQ(catchup.program_cadence_window_steps(), 1);
  EXPECT_DOUBLE_EQ(catchup.program_cadence_window_start_time(), 0.0);

  catchup.step(0.2);
  ASSERT_EQ(catchup_times.size(), 3u);
  EXPECT_NEAR(catchup_times[0], 0.0, 1.0e-14);
  EXPECT_NEAR(catchup_times[1], 0.1, 1.0e-14);
  EXPECT_NEAR(catchup_times[2], 0.2, 1.0e-14);
  ASSERT_EQ(catchup_steps.size(), 3u);
  for (const double h : catchup_steps)
    EXPECT_NEAR(h, 0.1, 1.0e-14);
  EXPECT_DOUBLE_EQ(catchup_times.back() + catchup_steps.back(), 0.1 + 0.2);
  EXPECT_EQ(catchup_macro_steps, (std::vector<int>{0, 0, 0}));
  EXPECT_EQ(catchup_every_one_due, (std::vector<bool>{true, true, true}));
  EXPECT_NEAR(catchup.time(), 0.3, 1.0e-14);
  EXPECT_EQ(catchup.macro_step(), 2);
  EXPECT_DOUBLE_EQ(catchup.program_cadence_window_dt(), 0.0);
  EXPECT_EQ(catchup.program_cadence_window_steps(), 0);
  EXPECT_DOUBLE_EQ(catchup.program_cadence_window_start_time(), 0.0);
}

TEST(ProgramRuntime, StrideHeldStepsPublishTheExactZeroBalance) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  auto config = unit_domain_config<kNativeDimension>(4);

  System<kNativeDimension> system(config);
  install_execution_lane(system, "test.program.stride-held-balance");
  runtime::program::ProgramContext context(&system);
  const std::string route = "pops.balance-ledger-route.v1:sha256:" + std::string(64, '7');
  const std::array<std::pair<const char*, double>, 5> records{{
      {"storage_change", 1.0},
      {"outward_boundary_flux", 2.0},
      {"sources", 3.0},
      {"reflux", 4.0},
      {"projection", 5.0},
  }};
  context.install([&](double) {
    for (const auto& [name, value] : records)
      context.record_balance_term(route, name, value);
  });
  system.set_program_cadence(/*substeps=*/1, /*stride=*/3);

  const auto step_and_read = [&]() {
    system.begin_step_transaction();
    system.step(0.1);
    const auto balance = system.accepted_balance_terms(route);
    system.commit_step_transaction();
    system.finalize_step_transaction();
    return balance;
  };

  for (int held = 0; held < 2; ++held) {
    const auto balance = step_and_read();
    ASSERT_EQ(balance.size(), records.size());
    for (const auto& [name, _value] : records)
      EXPECT_DOUBLE_EQ(balance.at(name), 0.0);
  }

  const auto accepted_image = [&]() {
    const auto& state = context.runtime_state();
    return std::tuple{system.time(),
                      system.macro_step(),
                      system.program_cadence_window_dt(),
                      system.program_cadence_window_steps(),
                      system.program_cadence_window_start_time(),
                      state.step_balance_terms_,
                      state.balance_step_completed_,
                      state.balance_program_was_due_};
  };
  const auto held_image = accepted_image();

  system.begin_step_transaction();
  system.step(0.1);
  const auto rejected_due = system.accepted_balance_terms(route);
  for (const auto& [name, value] : records)
    EXPECT_DOUBLE_EQ(rejected_due.at(name), value);
  system.rollback_step_transaction();
  EXPECT_EQ(accepted_image(), held_image);
  system.begin_step_transaction();
  // Rollback restores the held image; a new attempt must still start with an empty mailbox.
  try {
    (void)system.accepted_balance_terms(route);
    FAIL() << "A fresh attempt reused the preceding held-step balance";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(
        std::string(error.what()).find("current native attempt omitted term 'storage_change'"),
        std::string::npos);
  }
  system.rollback_step_transaction();
  EXPECT_EQ(accepted_image(), held_image);

  const auto due = step_and_read();
  ASSERT_EQ(due.size(), records.size());
  for (const auto& [name, value] : records)
    EXPECT_DOUBLE_EQ(due.at(name), value);
}

TEST(ProgramRuntime,
     CadenceUsesThePreparedFacadeEndpointWhenFloatingPointAdditionIsNonAssociative) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  auto config = unit_domain_config<kNativeDimension>(4);

  System<kNativeDimension> system(config);
  system.set_clock(0.1, 0);
  runtime::program::ProgramContext context(&system);
  std::vector<double> starts;
  std::vector<double> steps;
  context.install([&](double h) {
    starts.push_back(static_cast<double>(context.physical_time()));
    steps.push_back(h);
  });
  system.set_program_cadence(/*substeps=*/3, /*stride=*/3);

  const double after_first = 0.1 + 0.1;
  const double after_second = after_first + 0.1;
  const double accepted_endpoint = after_second + 0.3;
  const double effective_dt = (0.1 + 0.1) + 0.3;
  const double reconstructed_endpoint = 0.1 + effective_dt;
  const double numerical_dt = accepted_endpoint - 0.1;
  ASSERT_NE(std::bit_cast<std::uint64_t>(accepted_endpoint),
            std::bit_cast<std::uint64_t>(reconstructed_endpoint))
      << "fixture must exercise floating-point non-associativity";
  ASSERT_NE(std::bit_cast<std::uint64_t>(numerical_dt), std::bit_cast<std::uint64_t>(effective_dt))
      << "fixture must distinguish dt provenance from the representable facade interval";

  system.step(0.1);
  system.step(0.1);
  EXPECT_TRUE(starts.empty());
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.1 + 0.1);
  EXPECT_EQ(system.program_cadence_window_steps(), 2);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_start_time(), 0.1);

  system.step(0.3);
  ASSERT_EQ(starts.size(), 3u);
  ASSERT_EQ(steps.size(), 3u);
  EXPECT_EQ(std::bit_cast<std::uint64_t>(starts.front()), std::bit_cast<std::uint64_t>(0.1));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(starts.back() + steps.back()),
            std::bit_cast<std::uint64_t>(accepted_endpoint));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(system.time()),
            std::bit_cast<std::uint64_t>(accepted_endpoint));
  EXPECT_NE(std::bit_cast<std::uint64_t>(system.time()),
            std::bit_cast<std::uint64_t>(reconstructed_endpoint));
  EXPECT_EQ(system.macro_step(), 3);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.0);
  EXPECT_EQ(system.program_cadence_window_steps(), 0);
}

TEST(ProgramRuntime, CadenceWindowRestartAndRejectedDueStepAreTransactional) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  auto config = unit_domain_config<kNativeDimension>(4);

  System<kNativeDimension> system(config);
  std::vector<double> accepted_steps;
  bool reject = true;
  system.install_program_step([&](double h) {
    if (reject)
      throw runtime::program::StepAttemptRejected(SolveStatus::kIterationLimit, "cadence",
                                                  "fault injection in due stride window");
    accepted_steps.push_back(h);
  });
  system.set_program_cadence(/*substeps=*/1, /*stride=*/2);
  system.step(0.1);

  EXPECT_THROW(system.step(0.2), runtime::program::StepAttemptRejected);
  EXPECT_DOUBLE_EQ(system.time(), 0.1);
  EXPECT_EQ(system.macro_step(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.1);
  EXPECT_EQ(system.program_cadence_window_steps(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_start_time(), 0.0);

  reject = false;
  system.step(0.2);
  ASSERT_EQ(accepted_steps.size(), 1u);
  EXPECT_NEAR(accepted_steps[0], 0.3, 1.0e-14);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.0);
  EXPECT_EQ(system.program_cadence_window_steps(), 0);

  System<kNativeDimension> restarted(config);
  std::vector<double> restarted_times;
  restarted.install_program_step([&](double) { restarted_times.push_back(restarted.time()); });
  restarted.set_program_cadence(/*substeps=*/1, /*stride=*/2);
  restarted.restore_program_cadence_window(/*accumulated_dt=*/0.1, /*held_steps=*/1,
                                           /*window_start_time=*/0.0, /*accepted_last_dt=*/0.07,
                                           /*accepted_time=*/0.1,
                                           /*macro_step=*/1);
  restarted.set_clock(/*t=*/0.1, /*macro_step=*/1);
  EXPECT_DOUBLE_EQ(restarted.program_last_dt(), 0.07);
  restarted.step(0.2);
  EXPECT_EQ(restarted_times, (std::vector<double>{0.0}));
  EXPECT_NEAR(restarted.time(), 0.3, 1.0e-14);
  EXPECT_EQ(restarted.macro_step(), 2);
}

TEST(ProgramRuntime, CadenceRestoreCommitsOnlyForTheExactAcceptedClockPair) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  auto config = unit_domain_config<kNativeDimension>(4);

  System<kNativeDimension> system(config);
  system.install_program_step([](double) {});
  system.set_program_cadence(/*substeps=*/1, /*stride=*/2);

  // The accepted time is part of the native restore preflight. A malformed image cannot touch the
  // accepted window or arm a transaction.
  EXPECT_THROW(system.restore_program_cadence_window(
                   /*accumulated_dt=*/0.1, /*held_steps=*/1, /*window_start_time=*/0.1,
                   /*accepted_last_dt=*/0.075, /*accepted_time=*/0.1, /*macro_step=*/1),
               std::runtime_error);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.0);
  EXPECT_EQ(system.program_cadence_window_steps(), 0);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_start_time(), 0.0);
  EXPECT_THROW(system.restore_program_cadence_window(
                   /*accumulated_dt=*/0.1, /*held_steps=*/1, /*window_start_time=*/0.0,
                   /*accepted_last_dt=*/std::numeric_limits<double>::quiet_NaN(),
                   /*accepted_time=*/0.1, /*macro_step=*/1),
               std::runtime_error);
  EXPECT_DOUBLE_EQ(system.program_last_dt(), 0.0);

  // A valid image is only staged. One-ulp clock drift rejects and discards the transaction while the
  // accepted facade clock and cadence image remain byte-for-byte unchanged.
  system.restore_program_cadence_window(/*accumulated_dt=*/0.1, /*held_steps=*/1,
                                        /*window_start_time=*/0.0, /*accepted_last_dt=*/0.075,
                                        /*accepted_time=*/0.1,
                                        /*macro_step=*/1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.0);
  EXPECT_EQ(system.program_cadence_window_steps(), 0);
  EXPECT_DOUBLE_EQ(system.program_last_dt(), 0.0);
  EXPECT_THROW(system.set_clock(std::nextafter(0.1, 1.0), /*macro_step=*/1), std::runtime_error);
  EXPECT_DOUBLE_EQ(system.time(), 0.0);
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.0);
  EXPECT_EQ(system.program_cadence_window_steps(), 0);

  // The mismatch did not strand a pending token: a direct clean-boundary restore remains usable.
  system.set_clock(/*t=*/0.25, /*macro_step=*/0);
  system.restore_program_cadence_window(/*accumulated_dt=*/0.25, /*held_steps=*/1,
                                        /*window_start_time=*/0.25, /*accepted_last_dt=*/0.075,
                                        /*accepted_time=*/0.5,
                                        /*macro_step=*/1);
  system.set_clock(/*t=*/0.5, /*macro_step=*/1);
  EXPECT_DOUBLE_EQ(system.time(), 0.5);
  EXPECT_EQ(system.macro_step(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.25);
  EXPECT_EQ(system.program_cadence_window_steps(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_start_time(), 0.25);
  EXPECT_DOUBLE_EQ(system.program_last_dt(), 0.075);

  // Replacing a non-empty accepted window is transactional too: a bad replacement leaves both that
  // window and the facade cursor intact, and ordinary stepping remains possible afterwards.
  system.restore_program_cadence_window(/*accumulated_dt=*/0.4, /*held_steps=*/1,
                                        /*window_start_time=*/1.0, /*accepted_last_dt=*/0.2,
                                        /*accepted_time=*/1.4,
                                        /*macro_step=*/3);
  EXPECT_THROW(system.set_clock(std::nextafter(1.4, 2.0), /*macro_step=*/3), std::runtime_error);
  EXPECT_DOUBLE_EQ(system.time(), 0.5);
  EXPECT_EQ(system.macro_step(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.25);
  EXPECT_EQ(system.program_cadence_window_steps(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_start_time(), 0.25);
  EXPECT_DOUBLE_EQ(system.program_last_dt(), 0.075);
  system.step(0.25);
  EXPECT_DOUBLE_EQ(system.time(), 0.75);
  EXPECT_EQ(system.macro_step(), 2);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.0);
  EXPECT_EQ(system.program_cadence_window_steps(), 0);
}

TEST(ProgramRuntime, CadenceRejectsDtAbsorbedByThePhysicalClock) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  auto config = unit_domain_config<kNativeDimension>(4);

  System<kNativeDimension> system(config);
  int calls = 0;
  system.install_program_step([&](double) { ++calls; });
  system.set_clock(1.0e16, 0);

  EXPECT_THROW(system.step(0.5), std::overflow_error);
  EXPECT_DOUBLE_EQ(system.time(), 1.0e16);
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_EQ(calls, 0);
}

TEST(ProgramRuntime, CadenceFailsBeforeMutationWhenSubstepsCollapseTheRepresentableInterval) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  auto config = unit_domain_config<kNativeDimension>(4);

  System<kNativeDimension> system(config);
  int calls = 0;
  system.install_program_step([&](double) { ++calls; });
  system.set_program_cadence(/*substeps=*/3, /*stride=*/1);
  system.set_clock(1.0, 0);
  const double one_ulp = std::nextafter(1.0, 2.0) - 1.0;

  EXPECT_THROW(system.step(one_ulp), std::overflow_error);
  EXPECT_DOUBLE_EQ(system.time(), 1.0);
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_EQ(calls, 0);
}

TEST(ProgramRuntime, ProgramStateStorageDoesNotAdvertiseAnImplicitPoissonSource) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  System<kNativeDimension> system(distributed_boundary_domain_config<kNativeDimension>(8));
  install_execution_lane(system, "pops.test.program-runtime.storage-field-capability");
  system.install_block_state_route("gas", "test.program-runtime.storage.state@1");
  system.seal_auxiliary_providers();
  const auto prepared = prepare_compiled_system_block<kNativeDimension>(
      system, "gas", ProgramStorageGas{}, "state_storage", "unavailable", "conservative",
      "explicit", 1.4, 1, true, 1);
  EXPECT_FALSE(static_cast<bool>(prepared.poisson_rhs));
}

TEST(ProgramRuntime, CflWithoutPoissonProviderPublishesTheProgramStep) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr double dt = 1.e-4;
  System<kNativeDimension> system(distributed_boundary_domain_config<kNativeDimension>(8));
  install_execution_lane(system, "pops.test.program-runtime.cfl-no-field");
  system.install_block_state_route("gas", "test.program-runtime.cfl-no-field.state@1");
  system.seal_auxiliary_providers();
  auto prepared = prepare_compiled_system_block<kNativeDimension>(
      system, "gas", GasModel{}, "none", "rusanov", "conservative", "explicit", 1.4,
      1, true, 1);
  prepared.poisson_rhs = {};
  install_prepared_block(system, std::move(prepared));
  std::vector<double> initial;
  fill_ic(initial, 8, 1.4);
  system.set_state("gas", initial);
  const auto accepted = system.get_state("gas");
  system.set_program_block_map({0});
  auto context = runtime::program::make_program_execution_provider(&system);
  context->install([context](double h) {
    context->begin_step(h);
    auto& state = context->state(0);
    context->axpy(state, Real(.1), state);
  });
  system.set_program_block_map({0});
  EXPECT_TRUE(system.configured_field_provider_slots().empty());
  EXPECT_NO_THROW(system.step_cfl(.25, 1.e-12, dt, 0.));
  const auto result = system.get_state("gas");
  ASSERT_EQ(result.size(), accepted.size());
  for (std::size_t i = 0; i < result.size(); ++i)
    EXPECT_NEAR(result[i], 1.1 * accepted[i], 2.e-14);
  EXPECT_TRUE(system.field_provider_slots().empty());
  EXPECT_DOUBLE_EQ(system.time(), dt);
  EXPECT_EQ(system.macro_step(), 1);
}

TEST(ProgramRuntime, CflExplicitPoissonWithoutProviderRefusesBeforeProgramPublication) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  System<kNativeDimension> system(distributed_boundary_domain_config<kNativeDimension>(8));
  install_execution_lane(system, "pops.test.program-runtime.cfl-missing-field");
  system.install_block_state_route("gas", "test.program-runtime.cfl-missing-field.state@1");
  system.seal_auxiliary_providers();
  auto prepared = prepare_compiled_system_block<kNativeDimension>(
      system, "gas", GasModel{}, "none", "rusanov", "conservative", "explicit", 1.4,
      1, true, 1);
  prepared.poisson_rhs = {};
  install_prepared_block(system, std::move(prepared));
  system.set_poisson("charge_density", "cartesian_cg");
  std::vector<double> initial;
  fill_ic(initial, 8, 1.4);
  system.set_state("gas", initial);
  const auto accepted = system.get_state("gas");
  int program_calls = 0;
  system.install_program_step([&program_calls](double) { ++program_calls; });
  EXPECT_THROW(system.step_cfl(.25, 1.e-12, 1.e-4, 0.), std::runtime_error);
  EXPECT_THROW((void)system.solve_fields(), std::runtime_error);
  EXPECT_EQ(program_calls, 0);
  EXPECT_EQ(system.get_state("gas"), accepted);
  EXPECT_DOUBLE_EQ(system.time(), 0.);
  EXPECT_EQ(system.macro_step(), 0);
}

TEST(ProgramRuntime, CflRetainsImplicitPoissonProviderAndStageStateConsumption) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  System<kNativeDimension> system(distributed_boundary_domain_config<kNativeDimension>(8));
  install_execution_lane(system, "pops.test.program-runtime.cfl-stage-field");
  system.install_block_state_route("gas", "test.program-runtime.cfl-stage-field.state@1");
  system.seal_auxiliary_providers();
  auto prepared = prepare_compiled_system_block<kNativeDimension>(
      system, "gas", GasModel{}, "none", "rusanov", "conservative", "explicit", 1.4,
      1, true, 1);
  std::vector<Real> rhs_densities;
  auto physical_rhs = std::move(prepared.poisson_rhs);
  ASSERT_TRUE(static_cast<bool>(physical_rhs));
  prepared.poisson_rhs = [&rhs_densities, physical_rhs](const auto& state, auto& rhs) {
    rhs_densities.push_back(reduce_min(state, 0));
    physical_rhs(state, rhs);
  };
  install_prepared_block(system, std::move(prepared));
  std::vector<double> initial;
  fill_ic(initial, 8, 1.4);
  system.set_state("gas", initial);
  system.set_program_block_map({0});
  auto context = runtime::program::make_program_execution_provider(&system);
  context->install([context](double h) {
    context->begin_step(h);
    auto& state = context->state(0);
    context->axpy(state, Real(1), state);
    auto outcome = context->solve_fields_from_state(0, state);
    (void)outcome.consume(SolveConsumption::kAccept);
  });
  system.set_program_block_map({0});
  EXPECT_NO_THROW(system.step_cfl(.25, 1.e-12, 1.e-4, 0.));
  ASSERT_EQ(rhs_densities.size(), 2u);
  EXPECT_DOUBLE_EQ(rhs_densities[0], 1.);
  EXPECT_DOUBLE_EQ(rhs_densities[1], 2.);
  EXPECT_TRUE(system.field_provider_materialized("pops.system.default-field"));
  EXPECT_DOUBLE_EQ(system.time(), 1.e-4);
  EXPECT_EQ(system.macro_step(), 1);
}

TEST(ProgramRuntime, CflRejectsRankLocalPoissonRequirementsBeforeBranching) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  if (n_ranks() < 2)
    GTEST_SKIP() << "requires two MPI ranks to disagree about the field requirement";
  for (const bool explicit_request : {false, true}) {
    System<kNativeDimension> system(distributed_boundary_domain_config<kNativeDimension>(8));
    install_execution_lane(system, "pops.test.program-runtime.cfl-rank-field-contract");
    system.install_block_state_route("gas", "test.program-runtime.cfl-rank-field.state@1");
    system.seal_auxiliary_providers();
    auto prepared = prepare_compiled_system_block<kNativeDimension>(
        system, "gas", GasModel{}, "none", "rusanov", "conservative", "explicit", 1.4,
        1, true, 1);
    // First probe differs only in RHS availability; second only in the explicit
    // field request. Both must converge before any field/provider collective.
    if (explicit_request || system.prepared_boundary_execution_lane().rank() == 0)
      prepared.poisson_rhs = {};
    install_prepared_block(system, std::move(prepared));
    if (explicit_request && system.prepared_boundary_execution_lane().rank() == 0)
      system.set_poisson("charge_density", "cartesian_cg");
    std::vector<double> initial;
    fill_ic(initial, 8, 1.4);
    system.set_state("gas", initial);
    const auto accepted = system.get_state("gas");
    int calls = 0;
    system.install_program_step([&calls](double) { ++calls; });
    EXPECT_THROW(system.step_cfl(.25, 1.e-12, 1.e-4, 0.), std::invalid_argument);
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(system.get_state("gas"), accepted);
    EXPECT_DOUBLE_EQ(system.time(), 0.);
    EXPECT_EQ(system.macro_step(), 0);
  }
}

TEST(ProgramRuntime, ForwardEulerProgramContextMatchesEvalRhsReferenceAndCountsKernels) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  const int n = 16;
  const double gamma = 1.4, dt = 1e-3;
  const std::size_t cells = exact_cell_count<kNativeDimension>(n);

  auto cfg = unit_domain_config<kNativeDimension>(n);

  std::vector<double> U0(static_cast<std::size_t>(kGasComponents) * cells);
  fill_ic(U0, n, gamma);

  // Reference: one Forward-Euler step via the existing primitives, combined on the host.
  System<kNativeDimension> ref(cfg);
  install_execution_lane(ref, "pops.test.program-runtime.forward-euler-reference");
  add_gas(ref, gamma);
  ref.set_state("gas", U0);
  (void)pops::consume_solve_outcome(ref.solve_fields());
  const std::vector<double> R0 = ref.eval_rhs("gas");
  std::vector<double> Uref(static_cast<std::size_t>(kGasComponents) * cells);
  for (std::size_t k = 0; k < Uref.size(); ++k)
    Uref[k] = U0[k] + dt * R0[k];

  // Program: the SAME step expressed as a ProgramContext closure and driven by sim.step(dt).
  System<kNativeDimension> sim(cfg);
  install_execution_lane(sim, "pops.test.program-runtime.forward-euler");
  add_gas(sim, gamma);
  sim.set_state("gas", U0);
  sim.set_program_block_map({0});

  auto ctx = runtime::program::make_program_execution_provider(&sim);
  ctx->configure_primary_clock("macro");
  ctx->install([ctx](double h) {
    ctx->begin_step(h);
    ctx->set_stage_time(0, 1);
    auto field_outcome = ctx->solve_fields();
    (void)field_outcome.consume(SolveConsumption::kAccept);
    for (int b = 0; b < ctx->n_blocks(); ++b) {
      MultiFab<kNativeDimension>& U = ctx->state(b);
      MultiFab<kNativeDimension> R = ctx->rhs_scratch_like(U);
      ctx->rhs_into(b, U, R, 0);
      ctx->axpy(U, Real(h), R);  // U <- U + h * R  (Forward Euler)
    }
  });
  sim.set_program_block_map({0});

  // Profiling counters (ADC-459, Spec 3 section 29): the ProgramContext owns the two explicit
  // device algebra dispatches below (rhs_into + axpy). The field SolveOutcome owns its backend
  // diagnostics separately, and an ephemeral value scratch is deliberately absent from the
  // persistent scratch registry.
  sim.enable_profiling();
  const int step0 = sim.macro_step();
  sim.step(dt);
  const std::vector<double> Up = sim.get_state("gas");

  double err = 0, change = 0;
  for (std::size_t k = 0; k < Up.size(); ++k) {
    err = std::fmax(err, std::fabs(Up[k] - Uref[k]));
    change = std::fmax(change, std::fabs(Up[k] - U0[k]));
  }
  EXPECT_TRUE(err < 1e-12) << "parity: max|Up - Uref| = " << err;
  EXPECT_TRUE(sim.macro_step() == step0 + 1)
      << "macro_step not advanced (" << step0 << " -> " << sim.macro_step() << ")";
  EXPECT_TRUE(change > 1e-9) << "program step did not change the state (change = " << change << ")";

  // Pin exact ownership: no double-counting of the independently reported field solve, and no
  // persistent allocation is attributed to rhs_scratch_like.
  const runtime::program::Profiler& prof = sim.profiler();
  EXPECT_TRUE(prof.counter("kernels") == 2)
      << "kernels counter = " << static_cast<long long>(prof.counter("kernels"))
      << ", expected 2 (rhs_into + axpy; solve backend owns its diagnostics)";
  EXPECT_EQ(prof.counter("scratch_allocs"), 0);
  EXPECT_EQ(prof.counter("scratch_peak_bytes"), 0);
  // The cache hit/skip counters never fire on this native ProgramContext step (no held schedule); they
  // exist as counters only after the compiled scheduler emits cache_should_update. Assert they read 0.
  EXPECT_TRUE(prof.counter("cache_hits") == 0 && prof.counter("cache_misses") == 0)
      << "cache counters moved on the native path (hits="
      << static_cast<long long>(prof.counter("cache_hits"))
      << " misses=" << static_cast<long long>(prof.counter("cache_misses")) << ")";
  {
    const std::string report = sim.profile_report();
    EXPECT_TRUE(report.find("kernels=") != std::string::npos)
        << "profile_report omits the kernels counter line";
  }
}

TEST(ProgramRuntime, ForwardEulerProgramContextHonorsEmbeddedBoundaryResidualMetrics) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 16;
  constexpr double gamma = 1.4;
  constexpr double dt = 1e-3;
  const std::size_t cells = exact_cell_count<kNativeDimension>(n);
  auto cfg = unit_domain_config<kNativeDimension>(n);

  std::vector<double> initial(static_cast<std::size_t>(kGasComponents) * cells);
  fill_ic(initial, n, gamma);

  const auto install_forward_euler = [](System<kNativeDimension>& system) {
    system.set_program_block_map({0});
    auto context = runtime::program::make_program_execution_provider(&system);
    context->configure_primary_clock("macro");
    context->install([context](double step) {
      context->begin_step(step);
      context->set_stage_time(0, 1);
      MultiFab<kNativeDimension>& state = context->state(0);
      MultiFab<kNativeDimension> residual = context->rhs_scratch_like(state);
      context->rhs_into(0, state, residual, 0);
      context->axpy(state, Real(step), residual);
    });
    system.set_program_block_map({0});
  };

  System<kNativeDimension> cartesian(cfg);
  install_execution_lane(cartesian, "pops.test.program-runtime.embedded-forward-euler.cartesian");
  add_gas(cartesian, gamma, "none");
  cartesian.set_state("gas", initial);
  install_forward_euler(cartesian);
  cartesian.step(dt);
  const std::vector<double> cartesian_state = cartesian.get_state("gas");

  System<kNativeDimension> staircase(cfg);
  install_execution_lane(staircase, "pops.test.program-runtime.embedded-forward-euler.staircase");
  add_gas(staircase, gamma, "none");
  staircase.set_state("gas", initial);
  install_centered_ball(staircase, 0.34, "staircase");
  const std::vector<double> mask = staircase.embedded_boundary_mask();
  install_forward_euler(staircase);
  staircase.step(dt);
  const std::vector<double> staircase_state = staircase.get_state("gas");

  System<kNativeDimension> cutcell(cfg);
  install_execution_lane(cutcell, "pops.test.program-runtime.embedded-forward-euler.cutcell");
  add_gas(cutcell, gamma, "none");
  cutcell.set_state("gas", initial);
  install_centered_ball(cutcell, 0.34, "cutcell");
  install_forward_euler(cutcell);
  cutcell.step(dt);
  const std::vector<double> cutcell_state = cutcell.get_state("gas");

  double inactive_change = 0.0;
  double cutcell_inactive_change = 0.0;
  double active_change = 0.0;
  double cutcell_active_change = 0.0;
  double cartesian_inactive_change = 0.0;
  double route_difference = 0.0;
  int active_cells = 0;
  int inactive_cells = 0;
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const bool active = mask[cell] >= 0.5;
    active_cells += active ? 1 : 0;
    inactive_cells += active ? 0 : 1;
    for (int component = 0; component < kGasComponents; ++component) {
      const std::size_t index = static_cast<std::size_t>(component) * cells + cell;
      const double change = std::fabs(staircase_state[index] - initial[index]);
      if (active)
        active_change = std::fmax(active_change, change);
      else {
        inactive_change = std::fmax(inactive_change, change);
        cartesian_inactive_change = std::fmax(cartesian_inactive_change,
                                              std::fabs(cartesian_state[index] - initial[index]));
      }
      route_difference =
          std::fmax(route_difference, std::fabs(staircase_state[index] - cartesian_state[index]));
      const double cutcell_change = std::fabs(cutcell_state[index] - initial[index]);
      if (active)
        cutcell_active_change = std::fmax(cutcell_active_change, cutcell_change);
      else
        cutcell_inactive_change = std::fmax(cutcell_inactive_change, cutcell_change);
      route_difference =
          std::fmax(route_difference, std::fabs(cutcell_state[index] - cartesian_state[index]));
    }
  }

  ASSERT_GT(active_cells, 0);
  ASSERT_GT(inactive_cells, 0);
  EXPECT_EQ(inactive_change, 0.0)
      << "the Program wrote a non-zero staircase RHS outside the active set";
  EXPECT_EQ(cutcell_inactive_change, 0.0)
      << "the Program wrote a non-zero cut-cell RHS outside the active set";
  EXPECT_GT(active_change, 1e-10) << "the active Program residual was vacuous";
  EXPECT_GT(cutcell_active_change, 1e-10) << "the active cut-cell Program residual was vacuous";
  EXPECT_GT(cartesian_inactive_change, 1e-10)
      << "the Cartesian oracle did not exercise cells excluded by the staircase";
  EXPECT_GT(route_difference, 1e-10)
      << "the Program silently evaluated the Cartesian residual under staircase geometry";
  for (const double value : cutcell_state)
    EXPECT_TRUE(std::isfinite(value));
}

TEST(ProgramRuntime, SourceOnlyProgramStagePreservesEmbeddedBoundaryInactiveCells) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 12;
  constexpr double gamma = 1.4;
  constexpr double dt = 0.125;
  const std::size_t cells = exact_cell_count<kNativeDimension>(n);
  auto cfg = unit_domain_config<kNativeDimension>(n);
  std::vector<double> initial(static_cast<std::size_t>(kGasComponents) * cells);
  fill_ic(initial, n, gamma);

  const auto install_source_step = [](System<kNativeDimension>& system) {
    system.set_program_block_map({0});
    auto context = runtime::program::make_program_execution_provider(&system);
    context->configure_primary_clock("macro");
    context->install([context](double step) {
      context->begin_step(step);
      MultiFab<kNativeDimension>& state = context->state(0);
      MultiFab<kNativeDimension> source = context->rhs_scratch_like(state);
      context->source_default_into(0, state, source);
      context->axpy(state, Real(step), source);
    });
    system.set_program_block_map({0});
  };

  System<kNativeDimension> cartesian(cfg);
  install_execution_lane(cartesian, "pops.test.program-runtime.embedded-source-step.cartesian");
  add_sourced_gas(cartesian, gamma);
  cartesian.set_state("gas", initial);
  install_source_step(cartesian);
  cartesian.step(dt);
  const auto cartesian_state = cartesian.get_state("gas");

  System<kNativeDimension> staircase(cfg);
  install_execution_lane(staircase, "pops.test.program-runtime.embedded-source-step.staircase");
  add_sourced_gas(staircase, gamma);
  staircase.set_state("gas", initial);
  install_centered_ball(staircase, 0.31, "staircase");
  const auto mask = staircase.embedded_boundary_mask();
  install_source_step(staircase);
  staircase.step(dt);
  const auto staircase_state = staircase.get_state("gas");

  int active_cells = 0;
  int inactive_cells = 0;
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const bool active = mask[cell] >= 0.5;
    active_cells += active ? 1 : 0;
    inactive_cells += active ? 0 : 1;
    const double expected = initial[cell] + (active ? dt : 0.0);
    EXPECT_DOUBLE_EQ(staircase_state[cell], expected);
    EXPECT_DOUBLE_EQ(cartesian_state[cell], initial[cell] + dt);
    for (int component = 1; component < kGasComponents; ++component) {
      const std::size_t index = static_cast<std::size_t>(component) * cells + cell;
      EXPECT_DOUBLE_EQ(staircase_state[index], initial[index]);
    }
  }
  EXPECT_GT(active_cells, 0);
  EXPECT_GT(inactive_cells, 0);
}

TEST(ProgramRuntime, UnqualifiedGeneratedOperatorsRefuseActiveEmbeddedBoundaries) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 12;
  constexpr double gamma = 1.4;
  auto cfg = unit_domain_config<kNativeDimension>(n);

  const auto prepare = [&](System<kNativeDimension>& system, const std::string& identity) {
    install_execution_lane(system, identity);
    add_sourced_gas(system, gamma);
  };
  const auto expect_refusal = [](System<kNativeDimension>& system, const std::string& operation) {
    try {
      system.require_cartesian_generated_operator(0, operation);
      FAIL() << "active embedded geometry accepted unqualified operation '" << operation << "'";
    } catch (const std::runtime_error& error) {
      const std::string message = error.what();
      EXPECT_NE(message.find("block 0"), std::string::npos);
      EXPECT_NE(message.find(operation), std::string::npos);
    }
  };

  System<kNativeDimension> cartesian(cfg);
  prepare(cartesian, "pops.test.program-runtime.generated-preflight.cartesian");
  EXPECT_NO_THROW(cartesian.require_cartesian_generated_operator(0, "named_source"));

  System<kNativeDimension> inactive(cfg);
  prepare(inactive, "pops.test.program-runtime.generated-preflight.inactive");
  install_centered_ball(inactive, 0.31, "none");
  EXPECT_NO_THROW(inactive.require_cartesian_generated_operator(0, "named_source"));

  System<kNativeDimension> staircase(cfg);
  prepare(staircase, "pops.test.program-runtime.generated-preflight.staircase");
  install_centered_ball(staircase, 0.31, "staircase");
  expect_refusal(staircase, "named_source");

  System<kNativeDimension> cutcell(cfg);
  prepare(cutcell, "pops.test.program-runtime.generated-preflight.cutcell");
  install_centered_ball(cutcell, 0.31, "cutcell");
  expect_refusal(cutcell, "solve_local_linear");
}

TEST(ProgramRuntime, TerminalSourcePublicationAcceptsPreparedRecoveryCandidate) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  constexpr double gamma = 1.4;
  const std::size_t cells = exact_cell_count<kNativeDimension>(n);
  auto cfg = unit_domain_config<kNativeDimension>(n);

  System<kNativeDimension> system(cfg);
  install_execution_lane(system, "pops.test.program-runtime.source-recovery");
  add_draining_gas(system, "gas", gamma);
  std::vector<double> initial(static_cast<std::size_t>(kGasComponents) * cells);
  fill_ic(initial, n, gamma);
  system.set_state("gas", initial);
  system.set_program_block_map({0});
  auto context = runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("test.clock.source-recovery");
  context->install([context](double step) {
    context->begin_step(step);
    MultiFab<kNativeDimension>& live = context->state(0);
    MultiFab<kNativeDimension>& source = context->rhs_scratch(920001, 0, live);
    MultiFab<kNativeDimension>& candidate = context->scratch_state(920002, 0, live);
    context->source_default_into(0, live, source);
    context->lincomb(candidate, Real(1), live, Real(0), live);
    context->axpy(candidate, Real(step), source);
    context->commit_many({{&live, &candidate}});
  });
  system.set_program_block_map({0});

  system.step(0.25);
  const std::vector<double> accepted = system.get_state("gas");
  for (std::size_t cell = 0; cell < cells; ++cell)
    EXPECT_DOUBLE_EQ(accepted[cell], 0.75);
  EXPECT_DOUBLE_EQ(system.time(), 0.25);
  EXPECT_EQ(system.macro_step(), 1);
}

TEST(ProgramRuntime, TerminalSourceRecoveryRefusalPreventsPartialMultiBlockCommit) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  constexpr double gamma = 1.4;
  const std::size_t cells = exact_cell_count<kNativeDimension>(n);
  auto cfg = unit_domain_config<kNativeDimension>(n);

  System<kNativeDimension> system(cfg);
  install_execution_lane(system, "pops.test.program-runtime.source-recovery-multiblock");
  add_draining_gas(system, "first", gamma);
  add_draining_gas(system, "second", gamma);
  std::vector<double> initial(static_cast<std::size_t>(kGasComponents) * cells);
  fill_ic(initial, n, gamma);
  system.set_state("first", initial);
  system.set_state("second", initial);
  system.set_program_block_map({0, 1});
  auto context = runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("test.clock.source-recovery-multiblock");
  context->install([context](double step) {
    context->begin_step(step);
    MultiFab<kNativeDimension>& first = context->state(0);
    MultiFab<kNativeDimension>& second = context->state(1);
    MultiFab<kNativeDimension>& first_source = context->rhs_scratch(920011, 0, first);
    MultiFab<kNativeDimension>& second_source = context->rhs_scratch(920012, 1, second);
    MultiFab<kNativeDimension>& first_candidate = context->scratch_state(920013, 0, first);
    MultiFab<kNativeDimension>& second_candidate = context->scratch_state(920014, 1, second);
    context->source_default_into(0, first, first_source);
    context->source_default_into(1, second, second_source);
    context->lincomb(first_candidate, Real(1), first, Real(0), first);
    context->lincomb(second_candidate, Real(1), second, Real(0), second);
    context->axpy(first_candidate, Real(step), first_source);
    context->axpy(second_candidate, Real(2) * Real(step), second_source);
    context->commit_many({{&first, &first_candidate}, {&second, &second_candidate}});
  });
  system.set_program_block_map({0, 1});

  // The first block reaches rho=0.5 and is valid, while the second reaches rho=0.  If commit_many
  // copied as it iterated, the first live state would leak before the second recovery refusal.
  try {
    system.step(0.5);
    FAIL() << "an unrecoverable multi-block model-source endpoint must not publish";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("variable recovery rejected the candidate"),
              std::string::npos);
  }
  EXPECT_EQ(system.get_state("first"), initial);
  EXPECT_EQ(system.get_state("second"), initial);
  EXPECT_DOUBLE_EQ(system.time(), 0.0);
  EXPECT_EQ(system.macro_step(), 0);
}

TEST(ProgramRuntime, ExplicitSourceProgramPreservesEmbeddedBoundaryInactiveCells) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 12;
  constexpr double gamma = 1.4;
  constexpr double dt = 1e-3;
  const std::size_t cells = exact_cell_count<kNativeDimension>(n);
  auto cfg = unit_domain_config<kNativeDimension>(n);
  std::vector<double> initial(static_cast<std::size_t>(kGasComponents) * cells);
  fill_ic(initial, n, gamma);

  System<kNativeDimension> system(cfg);
  install_execution_lane(system, "pops.test.program-runtime.inactive-source");
  add_sourced_gas(system, gamma);
  system.set_state("gas", initial);
  install_centered_ball(system, 0.31, "staircase");
  const auto mask = system.embedded_boundary_mask();
  system.set_program_block_map({0});
  auto context = runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("macro");
  context->install([context](double step) {
    context->begin_step(step);
    MultiFab<kNativeDimension>& state = context->state(0);
    MultiFab<kNativeDimension> source = context->rhs_scratch_like(state);
    context->source_default_into(0, state, source);
    context->axpy(state, Real(step), source);
  });
  system.set_program_block_map({0});
  system.step(dt);
  const auto result = system.get_state("gas");

  double active_change = 0.0;
  int inactive_cells = 0;
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const bool active = mask[cell] >= 0.5;
    inactive_cells += active ? 0 : 1;
    for (int component = 0; component < kGasComponents; ++component) {
      const std::size_t index = static_cast<std::size_t>(component) * cells + cell;
      if (active)
        active_change = std::fmax(active_change, std::fabs(result[index] - initial[index]));
      else
        EXPECT_DOUBLE_EQ(result[index], initial[index]);
    }
  }
  EXPECT_GT(inactive_cells, 0);
  EXPECT_GT(active_change, 1e-10);
}

TEST(ProgramRuntime, PreparedMaximumSpeedProviderIsGeometryIndependent) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 12;
  constexpr double gamma = 1.4;
  const std::size_t cells = exact_cell_count<kNativeDimension>(n);
  auto cfg = unit_domain_config<kNativeDimension>(n);

  System<kNativeDimension> system(cfg);
  install_execution_lane(system, "pops.test.program-runtime.geometry-independent-speed");
  add_gas(system, gamma, "none");
  install_centered_ball(system, 0.31, "staircase");
  const auto mask = system.embedded_boundary_mask();
  std::vector<double> state(static_cast<std::size_t>(kGasComponents) * cells);
  fill_ic(state, n, gamma);
  for (std::size_t cell = 0; cell < cells; ++cell)
    if (mask[cell] < 0.5)
      state[static_cast<std::size_t>(kNativeDimension + 1) * cells + cell] = 1.0e12;
  system.set_state("gas", state);

  const double embedded_speed = system.block_max_speed(0, system.block_state(0));
  system.set_geometry_mode("none");
  const double cartesian_speed = system.block_max_speed(0, system.block_state(0));
  EXPECT_GT(embedded_speed, 0.0);
  EXPECT_DOUBLE_EQ(cartesian_speed, embedded_speed)
      << "geometry mode selected a hidden maximum-speed implementation";
}

TEST(ProgramRuntime, DiffusiveCflAddsPreparedParabolicFrequencyToTransport) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 12;
  constexpr double gamma = 1.4;
  constexpr double cfl = 0.4;
  constexpr double diffusivity = 0.1;
  const std::size_t cells = exact_cell_count<kNativeDimension>(n);
  auto cfg = unit_domain_config<kNativeDimension>(n);

  System<kNativeDimension> system(cfg);
  install_execution_lane(system, "pops.test.program-runtime.diffusive-cfl");
  add_diffusive_gas(system, gamma);
  std::vector<double> state(static_cast<std::size_t>(kGasComponents) * cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    state[cell] = 1.0;
    state[static_cast<std::size_t>(kNativeDimension + 1) * cells + cell] = 2.5;
  }
  system.set_state("gas", state);
  system.install_program_step([](double) {});

  const double transport_frequency = std::sqrt(gamma) * n;
  const double parabolic_frequency =
      2.0 * diffusivity * static_cast<double>(kNativeDimension) * n * n;
  EXPECT_NEAR(system.step_cfl(cfl, 1.0e-12), cfl / (transport_frequency + parabolic_frequency),
              1.0e-12);
  EXPECT_EQ(system.last_dt_bound(), "parabolic_frequency:gas");
}

TEST(ProgramRuntime, CutCellPreparesFractionalMeasureAndSharesTheModelSpeedProvider) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 18;
  constexpr double gamma = 1.4;
  const std::size_t cells = exact_cell_count<kNativeDimension>(n);
  auto cfg = unit_domain_config<kNativeDimension>(n);
  std::vector<double> uniform(static_cast<std::size_t>(kGasComponents) * cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    uniform[cell] = 1.0;
    uniform[static_cast<std::size_t>(kNativeDimension + 1) * cells + cell] = 2.5;
  }

  System<kNativeDimension> staircase(cfg);
  install_execution_lane(staircase, "pops.test.program-runtime.fractional-speed.staircase");
  add_gas(staircase, gamma, "none");
  staircase.set_state("gas", uniform);
  install_centered_ball(staircase, 0.34, "staircase", 0.1);
  const double staircase_speed = staircase.block_max_speed(0, staircase.block_state(0));

  System<kNativeDimension> cutcell(cfg);
  install_execution_lane(cutcell, "pops.test.program-runtime.fractional-speed.cutcell");
  add_gas(cutcell, gamma, "none");
  cutcell.set_state("gas", uniform);
  install_centered_ball(cutcell, 0.34, "cutcell", 0.1);
  const double cutcell_speed = cutcell.block_max_speed(0, cutcell.block_state(0));
  const auto kappa_pieces = cutcell.output_embedded_boundary_local_pieces("pops_kappa", 0);
  int fractional_cells = 0;
  for (const auto& piece : kappa_pieces)
    fractional_cells +=
        static_cast<int>(std::count_if(piece.values.begin(), piece.values.end(),
                                       [](double value) { return value > 0.0 && value < 1.0; }));

  EXPECT_GT(staircase_speed, 0.0);
  EXPECT_DOUBLE_EQ(cutcell_speed, staircase_speed)
      << "cut-cell geometry selected a hidden model-speed provider";
  EXPECT_GT(fractional_cells, 0)
      << "the prepared cut-cell geometry did not publish a fractional cell measure";
}

TEST(ProgramRuntime, PhysicalReductionsUsePreparedEmbeddedBoundaryMeasure) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 16;
  constexpr double gamma = 1.4;
  const std::size_t cells = exact_cell_count<kNativeDimension>(n);
  auto cfg = unit_domain_config<kNativeDimension>(n);

  System<kNativeDimension> staircase(cfg);
  install_execution_lane(staircase, "pops.test.program-runtime.staircase-reductions");
  add_gas(staircase, gamma, "none");
  install_centered_ball(staircase, 0.31, "staircase");
  const std::vector<double> staircase_mask = staircase.embedded_boundary_mask();
  std::vector<double> staircase_state(static_cast<std::size_t>(kGasComponents) * cells, 0.0);
  int staircase_active = 0;
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const bool active = staircase_mask[cell] >= 0.5;
    staircase_active += active ? 1 : 0;
    staircase_state[cell] = active ? 2.0 : 1000.0;
    staircase_state[cells + cell] = active ? 3.0 : -1000.0;
  }
  ASSERT_GT(staircase_active, 0);
  ASSERT_LT(staircase_active, static_cast<int>(cells));
  staircase.set_state("gas", staircase_state);
  staircase.set_program_block_map({0});
  runtime::program::ProgramContext staircase_context(&staircase);
  MultiFab<kNativeDimension>& staircase_field = staircase_context.state(0);
  const int staircase_inactive = static_cast<int>(cells) - staircase_active;
  const Real staircase_raw_sum =
      Real(2) * Real(staircase_active) + Real(1000) * Real(staircase_inactive);
  const Real staircase_raw_dot =
      Real(4) * Real(staircase_active) + Real(1000000) * Real(staircase_inactive);
  const Real staircase_active_sum = Real(2 * staircase_active);
  const Real staircase_active_dot = Real(4 * staircase_active);
  // Ownerless ProgramContext overloads remain raw scratch-field algebra, including inactive
  // sentinels. Owner-qualified overloads exclude inactive cells; System alone applies kappa.
  EXPECT_EQ(staircase_context.sum_component(staircase_field, 0), staircase_raw_sum);
  EXPECT_EQ(staircase_context.abs_sum_component(staircase_field, 0), staircase_raw_sum);
  EXPECT_EQ(pops::reduce_abs_sum(staircase_field, 0), staircase_raw_sum);
  EXPECT_EQ(pops::dot(staircase_field, staircase_field, 0), staircase_raw_dot);
  EXPECT_EQ(staircase_context.max_component(staircase_field, 0), Real(1000));
  EXPECT_EQ(staircase_context.min_component(staircase_field, 1), Real(-1000));
  EXPECT_EQ(staircase_context.sum_component(0, staircase_field, 0), staircase_active_sum);
  EXPECT_EQ(staircase_context.abs_sum_component(0, staircase_field, 0), staircase_active_sum);
  EXPECT_EQ(staircase_context.dot(0, staircase_field, staircase_field), staircase_active_dot);
  EXPECT_EQ(staircase_context.max_component(0, staircase_field, 0), Real(2));
  EXPECT_EQ(staircase_context.min_component(0, staircase_field, 1), Real(3));
  EXPECT_EQ(staircase_context.norm2(0, staircase_field), std::sqrt(staircase_active_dot));
  EXPECT_EQ(staircase_context.norm_inf(0, staircase_field), Real(2));
  EXPECT_EQ(staircase.mass("gas"), static_cast<double>(staircase_active_sum));
  EXPECT_EQ(staircase.reduce_component("gas", "sum", 0), static_cast<double>(staircase_active_sum));
  EXPECT_EQ(staircase.reduce_component("gas", "sum_sq", 0),
            static_cast<double>(Real(2) * staircase_active_sum));
  const int invalid_component = staircase_field.ncomp();
  const auto expect_original_component_error = [&](auto&& reduce, const char* helper) {
    try {
      (void)reduce();
      FAIL() << helper << " must rethrow the original rank-local component error in serial";
    } catch (const std::out_of_range& error) {
      EXPECT_NE(std::string(error.what()).find(helper), std::string::npos);
    }
  };
  expect_original_component_error(
      [&] { return staircase_context.sum_component(0, staircase_field, invalid_component); },
      "pops::reduce_active_sum_local");
  expect_original_component_error(
      [&] { return staircase_context.abs_sum_component(0, staircase_field, invalid_component); },
      "pops::reduce_active_abs_sum_local");
  expect_original_component_error(
      [&] { return staircase_context.max_component(0, staircase_field, invalid_component); },
      "pops::reduce_active_max_local");
  expect_original_component_error(
      [&] { return staircase_context.min_component(0, staircase_field, invalid_component); },
      "pops::reduce_active_min_local");
  MultiFab<kNativeDimension>& staircase_status =
      staircase_context.scalar_scratch(911, 0, staircase_field, 1, 0);
  try {
    (void)staircase_context.dot(0, staircase_field, staircase_status);
    FAIL() << "owner-qualified dot must rethrow the original layout error in serial";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("ProgramContext dot"), std::string::npos);
  }
  EXPECT_EQ(staircase.reduce_component("gas", "max", 0), 2.0);
  EXPECT_EQ(staircase.reduce_component("gas", "min", 1), 3.0);
  EXPECT_EQ(staircase.reduce_component("gas", "abs_max", 0), 2.0);

  System<kNativeDimension> cutcell(cfg);
  install_execution_lane(cutcell, "pops.test.program-runtime.cutcell-reductions");
  add_gas(cutcell, gamma, "none");
  install_centered_ball(cutcell, 0.31, "cutcell");
  const std::vector<double> cutcell_mask = cutcell.embedded_boundary_mask();
  const auto cutcell_kappa_pieces = cutcell.output_embedded_boundary_local_pieces("pops_kappa", 0);
  std::vector<double> cutcell_state(static_cast<std::size_t>(kGasComponents) * cells, 0.0);
  int cutcell_active = 0;
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const bool active = cutcell_mask[cell] >= 0.5;
    cutcell_active += active ? 1 : 0;
    cutcell_state[cell] = active ? 2.0 : 1000.0;
    cutcell_state[cells + cell] = active ? 3.0 : -1000.0;
  }
  ASSERT_GT(cutcell_active, 0);
  ASSERT_LT(cutcell_active, static_cast<int>(cells));
  double cutcell_kappa_sum = 0.0;
  for (const auto& piece : cutcell_kappa_pieces)
    for (const double kappa : piece.values)
      cutcell_kappa_sum += kappa;
  cutcell.set_state("gas", cutcell_state);
  cutcell.set_program_block_map({0});
  runtime::program::ProgramContext cutcell_context(&cutcell);
  MultiFab<kNativeDimension>& cutcell_field = cutcell_context.state(0);
  const int cutcell_inactive = static_cast<int>(cells) - cutcell_active;
  const Real cutcell_raw_sum = Real(2) * Real(cutcell_active) + Real(1000) * Real(cutcell_inactive);
  const Real cutcell_raw_dot =
      Real(4) * Real(cutcell_active) + Real(1000000) * Real(cutcell_inactive);
  const Real cutcell_active_sum = Real(2 * cutcell_active);
  const Real cutcell_active_dot = Real(4 * cutcell_active);
  const double cutcell_physical_sum = 2.0 * cutcell_kappa_sum;
  EXPECT_EQ(cutcell_context.sum_component(cutcell_field, 0), cutcell_raw_sum);
  EXPECT_EQ(cutcell_context.abs_sum_component(cutcell_field, 0), cutcell_raw_sum);
  EXPECT_EQ(pops::reduce_abs_sum(cutcell_field, 0), cutcell_raw_sum);
  EXPECT_EQ(pops::dot(cutcell_field, cutcell_field, 0), cutcell_raw_dot);
  EXPECT_EQ(cutcell_context.max_component(cutcell_field, 0), Real(1000));
  EXPECT_EQ(cutcell_context.min_component(cutcell_field, 1), Real(-1000));
  EXPECT_EQ(cutcell_context.sum_component(0, cutcell_field, 0), cutcell_active_sum);
  EXPECT_EQ(cutcell_context.abs_sum_component(0, cutcell_field, 0), cutcell_active_sum);
  EXPECT_EQ(cutcell_context.dot(0, cutcell_field, cutcell_field), cutcell_active_dot);
  EXPECT_EQ(cutcell_context.max_component(0, cutcell_field, 0), Real(2));
  EXPECT_EQ(cutcell_context.min_component(0, cutcell_field, 1), Real(3));
  EXPECT_EQ(cutcell_context.norm2(0, cutcell_field), std::sqrt(cutcell_active_dot));
  EXPECT_EQ(cutcell_context.norm_inf(0, cutcell_field), Real(2));
  EXPECT_GT(cutcell_physical_sum, 0.0);
  EXPECT_LT(cutcell_physical_sum, static_cast<double>(staircase_active_sum))
      << "the cut-cell integral ignored the prepared relative volume fraction";
  EXPECT_NEAR(cutcell.mass("gas"), cutcell_physical_sum, 1e-12);
  EXPECT_NEAR(cutcell.reduce_component("gas", "sum", 0), cutcell_physical_sum, 1e-12);
  EXPECT_NEAR(cutcell.reduce_component("gas", "sum_sq", 0), 2.0 * cutcell_physical_sum, 1e-10);
  EXPECT_EQ(cutcell.reduce_component("gas", "max", 0), 2.0);
  EXPECT_EQ(cutcell.reduce_component("gas", "min", 1), 3.0);
  EXPECT_EQ(cutcell.reduce_component("gas", "abs_max", 0), 2.0);
}

TEST(ProgramRuntime, PreparedEmbeddedBoundaryMaskSeparatesActiveAndInactiveCells) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 12;
  constexpr double gamma = 1.4;
  const std::size_t cells = exact_cell_count<kNativeDimension>(n);
  auto cfg = unit_domain_config<kNativeDimension>(n);

  for (const std::string mode : {"staircase", "cutcell"}) {
    System<kNativeDimension> system(cfg);
    install_execution_lane(system, "pops.test.program-runtime.mask-separation." + mode);
    add_gas(system, gamma, "none");
    install_centered_ball(system, 0.32, mode);
    const std::vector<double> mask = system.embedded_boundary_mask();
    system.set_state("gas",
                     std::vector<double>(static_cast<std::size_t>(kGasComponents) * cells, 2.0));
    int active = 0;
    int inactive = 0;
    for (const double value : mask) {
      active += value >= 0.5 ? 1 : 0;
      inactive += value < 0.5 ? 1 : 0;
    }
    ASSERT_GT(active, 0) << mode;
    ASSERT_GT(inactive, 0) << mode;
    EXPECT_EQ(active + inactive, static_cast<int>(cells)) << mode;
  }
}

TEST(ProgramRuntime, PointwiseStatusUsesPreparedEmbeddedBoundaryMask) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 12;
  constexpr double gamma = 1.4;
  auto cfg = unit_domain_config<kNativeDimension>(n);

  for (const std::string mode : {"staircase", "cutcell"}) {
    System<kNativeDimension> system(cfg);
    install_execution_lane(system, "pops.test.program-runtime.pointwise-status." + mode);
    add_gas(system, gamma, "none");
    install_centered_ball(system, 0.32, mode);
    const std::vector<double> flattened_mask = system.embedded_boundary_mask();
    ASSERT_TRUE(std::any_of(flattened_mask.begin(), flattened_mask.end(), [](double value) {
      return value >= 0.5;
    })) << mode;
    ASSERT_TRUE(std::any_of(flattened_mask.begin(), flattened_mask.end(), [](double value) {
      return value < 0.5;
    })) << mode;
    system.set_program_block_map({0});
    runtime::program::ProgramContext context(&system);
    MultiFab<kNativeDimension>& state = context.state(0);
    const MultiFab<kNativeDimension>* const active = context.pointwise_active_mask(0, state);
    ASSERT_NE(active, nullptr) << mode;
    EXPECT_EQ(active, context.pointwise_active_mask(0, state)) << mode;

    MultiFab<kNativeDimension>& status = context.scalar_scratch(910, 0, state, 1, 0);
    const auto write_status = [&](Real active_value, Real inactive_value) {
      for (std::size_t local = 0; local < status.local_size(); ++local) {
        const FieldView<Real, kNativeDimension> output = status.fab(local).view();
        const FieldView<const Real, kNativeDimension> mask =
            std::as_const(active->fab(local)).view();
        for_each_cell(status.box(local), [=] POPS_HD(const Index<kNativeDimension>& cell) {
          output(cell, 0) = mask(cell, 0) >= Real{0.5} ? active_value : inactive_value;
        });
      }
      device_fence();
    };

    write_status(Real{0}, Real{2});
    EXPECT_EQ(context.pointwise_status_max(0, status, active, context.prepared_execution_lane()),
              Real{0})
        << mode;
    write_status(Real{0}, std::numeric_limits<Real>::quiet_NaN());
    EXPECT_EQ(context.pointwise_status_max(0, status, active, context.prepared_execution_lane()),
              Real{0})
        << mode;
    write_status(Real{2}, Real{0});
    EXPECT_EQ(context.pointwise_status_max(0, status, active, context.prepared_execution_lane()),
              Real{2})
        << mode;
    write_status(std::numeric_limits<Real>::quiet_NaN(), Real{0});
    EXPECT_EQ(context.pointwise_status_max(0, status, active, context.prepared_execution_lane()),
              Real{3})
        << mode;
    EXPECT_THROW(
        context.pointwise_status_max(0, status, &status, context.prepared_execution_lane()),
        std::invalid_argument)
        << mode;
  }
}

TEST(ProgramRuntime, Ssprk3ProgramAlgebraPreservesInactiveBits) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 14;
  constexpr double gamma = 1.4;
  constexpr double inactive_value = 0.9;
  const std::size_t cells = exact_cell_count<kNativeDimension>(n);
  auto cfg = unit_domain_config<kNativeDimension>(n);
  std::vector<double> initial(static_cast<std::size_t>(kGasComponents) * cells);
  fill_ic(initial, n, gamma);

  System<kNativeDimension> program(cfg);
  install_execution_lane(program, "pops.test.program-runtime.ssprk3");
  add_gas(program, gamma, "none");
  install_centered_ball(program, 0.32, "staircase");
  const auto mask = program.embedded_boundary_mask();
  for (std::size_t cell = 0; cell < cells; ++cell)
    if (mask[cell] < 0.5)
      for (int component = 0; component < kGasComponents; ++component)
        initial[static_cast<std::size_t>(component) * cells + cell] = inactive_value;
  program.set_state("gas", initial);
  program.set_program_block_map({0});
  auto context = runtime::program::make_program_execution_provider(&program);
  context->configure_primary_clock("macro");
  context->install([context](double step) {
    context->begin_step(step);
    MultiFab<kNativeDimension>& state = context->state(0);
    MultiFab<kNativeDimension> initial_state = state;
    MultiFab<kNativeDimension> stage = state;
    MultiFab<kNativeDimension> residual = context->rhs_scratch_like(state);

    context->set_stage_time(0, 1);
    context->rhs_into(0, state, residual, 100);
    context->axpy(stage, Real(step), residual);

    context->set_stage_time(1, 1);
    residual.set_val(Real(0));
    context->rhs_into(0, stage, residual, 101);
    context->axpy(stage, Real(step), residual);
    context->lincomb(stage, Real(3) / Real(4), initial_state, Real(1) / Real(4), stage);

    context->set_stage_time(1, 2);
    residual.set_val(Real(0));
    context->rhs_into(0, stage, residual, 102);
    context->axpy(stage, Real(step), residual);
    context->lincomb(stage, Real(1) / Real(3), initial_state, Real(2) / Real(3), stage);
    context->commit_many({{&state, &stage}});
  });
  program.set_program_block_map({0});
  program.step(1.0e-4);
  const auto program_result = program.get_state("gas");

  int inactive_cells = 0;
  for (std::size_t cell = 0; cell < cells; ++cell) {
    if (mask[cell] >= 0.5)
      continue;
    ++inactive_cells;
    for (int component = 0; component < kGasComponents; ++component) {
      const std::size_t index = static_cast<std::size_t>(component) * cells + cell;
      EXPECT_EQ(std::bit_cast<std::uint64_t>(program_result[index]),
                std::bit_cast<std::uint64_t>(initial[index]));
    }
  }
  EXPECT_GT(inactive_cells, 0);
}

TEST(ProgramRuntime, EmbeddedBoundaryCapabilitiesUsePreparedExactRankProviders) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  auto cfg = unit_domain_config<kNativeDimension>(10);

  System<kNativeDimension> reconstructed(cfg);
  install_execution_lane(reconstructed,
                         "pops.test.program-runtime.embedded-capability.reconstructed");
  add_gas(reconstructed, 1.4, "minmod");
  EXPECT_NO_THROW(install_centered_ball(reconstructed, 0.3, "staircase"));
  const auto reconstructed_mask = reconstructed.embedded_boundary_mask();
  EXPECT_TRUE(std::any_of(reconstructed_mask.begin(), reconstructed_mask.end(),
                          [](double value) { return value == 0.0; }));
  EXPECT_TRUE(std::any_of(reconstructed_mask.begin(), reconstructed_mask.end(),
                          [](double value) { return value == 1.0; }));

  for (const std::string mode : {"staircase", "cutcell"}) {
    System<kNativeDimension> diffusive(cfg);
    install_execution_lane(diffusive,
                           "pops.test.program-runtime.embedded-capability.diffusive." + mode);
    add_diffusive_gas(diffusive, 1.4);
    EXPECT_NO_THROW(install_centered_ball(diffusive, 0.3, mode));
    const auto diffusive_mask = diffusive.embedded_boundary_mask();
    EXPECT_TRUE(std::any_of(diffusive_mask.begin(), diffusive_mask.end(),
                            [](double value) { return value == 0.0; }));
    EXPECT_TRUE(std::any_of(diffusive_mask.begin(), diffusive_mask.end(),
                            [](double value) { return value == 1.0; }));
    EXPECT_THROW((void)diffusive.eval_rhs("gas"), std::invalid_argument);
  }
}

TEST(ProgramRuntime, PointwiseProjectionPreservesEmbeddedBoundaryInactiveCells) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 12;
  constexpr double gamma = 1.4;
  const std::size_t cells = exact_cell_count<kNativeDimension>(n);
  auto cfg = unit_domain_config<kNativeDimension>(n);
  std::vector<double> initial(static_cast<std::size_t>(kGasComponents) * cells);
  fill_ic(initial, n, gamma);

  const auto install_projection_step = [](System<kNativeDimension>& system) {
    system.set_program_block_map({0});
    auto context = runtime::program::make_program_execution_provider(&system);
    context->configure_primary_clock("macro");
    context->install([context](double step) {
      context->begin_step(step);
      context->apply_projection(0, context->state(0));
    });
    system.set_program_block_map({0});
  };

  System<kNativeDimension> cartesian(cfg);
  install_execution_lane(cartesian, "pops.test.program-runtime.projection-step.cartesian");
  add_projecting_gas(cartesian, gamma);
  cartesian.set_state("gas", initial);
  install_projection_step(cartesian);
  cartesian.step(0.1);
  const auto cartesian_state = cartesian.get_state("gas");

  System<kNativeDimension> cutcell(cfg);
  install_execution_lane(cutcell, "pops.test.program-runtime.projection-step.cutcell");
  add_projecting_gas(cutcell, gamma);
  cutcell.set_state("gas", initial);
  install_centered_ball(cutcell, 0.31, "cutcell");
  const auto mask = cutcell.embedded_boundary_mask();
  install_projection_step(cutcell);
  cutcell.step(0.1);
  const auto cutcell_state = cutcell.get_state("gas");

  int active_cells = 0;
  int inactive_cells = 0;
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const bool active = mask[cell] >= 0.5;
    active_cells += active ? 1 : 0;
    inactive_cells += active ? 0 : 1;
    EXPECT_DOUBLE_EQ(cartesian_state[cell], 2.0);
    EXPECT_DOUBLE_EQ(cutcell_state[cell], active ? 2.0 : initial[cell]);
    for (int component = 1; component < kGasComponents; ++component) {
      const std::size_t index = static_cast<std::size_t>(component) * cells + cell;
      EXPECT_DOUBLE_EQ(cutcell_state[index], initial[index]);
    }
  }
  EXPECT_GT(active_cells, 0);
  EXPECT_GT(inactive_cells, 0);
}

TEST(ProgramRuntime, ProjectAndRecheckConsumesSolveAndCommitsProjectedCandidate) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  constexpr double gamma = 1.4;
  auto cfg = unit_domain_config<kNativeDimension>(n);

  System<kNativeDimension> sim(cfg);
  install_execution_lane(sim, "pops.test.program-runtime.project-recheck-accept");
  add_projecting_gas(sim, gamma);
  sim.set_poisson("charge_density", "cartesian_cg");
  std::vector<double> initial;
  fill_ic(initial, n, gamma);
  sim.set_state("gas", initial);
  (void)consume_solve_outcome(sim.solve_fields());
  sim.set_program_block_map({0});

  int consumed_solves = 0;
  auto ctx = runtime::program::make_program_execution_provider(&sim);
  ctx->install([ctx, &consumed_solves](double dt) {
    ctx->begin_step(dt);
    MultiFab<kNativeDimension>& state = ctx->state(0);
    MultiFab<kNativeDimension>& candidate = ctx->scratch_state(666001, 0, state);
    candidate.set_val(Real(-1));

    auto field_outcome = ctx->solve_fields();
    const SolveReport field_report = field_outcome.consume(SolveConsumption::kAccept);
    if (!field_report.solved_value_available())
      throw std::logic_error("ProjectAndRecheck test did not receive a solved field value");
    ++consumed_solves;

    if (ctx->min_component(candidate, 0) <= Real(0)) {
      ctx->apply_projection(0, candidate);
      if (ctx->min_component(candidate, 0) <= Real(0))
        throw runtime::program::StepAttemptRejected(
            SolveStatus::kIterationLimit, "guard recheck",
            "ProjectAndRecheck projection did not repair the candidate");
    }
    ctx->commit_many({{&state, &candidate}});
  });
  sim.set_program_block_map({0});

  sim.step(1e-3);

  EXPECT_EQ(consumed_solves, 1);
  EXPECT_EQ(sim.macro_step(), 1);
  EXPECT_DOUBLE_EQ(sim.time(), 1e-3);
  const std::vector<double> accepted = sim.get_state("gas");
  const std::size_t cells = exact_cell_count<kNativeDimension>(n);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    EXPECT_DOUBLE_EQ(accepted[cell], 2.0);
    for (int component = 1; component < kNativeDimension + 2; ++component) {
      const double expected = component == kNativeDimension + 1 ? 5.0 : -1.0;
      EXPECT_DOUBLE_EQ(accepted[static_cast<std::size_t>(component) * cells + cell], expected);
    }
  }
}

TEST(ProgramRuntime, ProjectAndRecheckFailureConsumesSolveAndRollsBackWithoutPublication) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  constexpr double gamma = 1.4;
  auto cfg = unit_domain_config<kNativeDimension>(n);

  System<kNativeDimension> sim(cfg);
  install_execution_lane(sim, "pops.test.program-runtime.project-recheck-reject");
  add_projecting_gas(sim, gamma);
  sim.set_poisson("charge_density", "cartesian_cg");
  std::vector<double> initial;
  fill_ic(initial, n, gamma);
  sim.set_state("gas", initial);
  (void)consume_solve_outcome(sim.solve_fields());
  sim.register_history("gas.guard_candidate", 2, kGasComponents);
  sim.set_program_block_map({0});

  int consumed_solves = 0;
  auto ctx = runtime::program::make_program_execution_provider(&sim);
  ctx->install([ctx, &consumed_solves](double dt) {
    ctx->begin_step(dt);
    MultiFab<kNativeDimension>& state = ctx->state(0);
    MultiFab<kNativeDimension>& candidate = ctx->scratch_state(666002, 0, state);
    candidate.set_val(Real(-1));

    auto field_outcome = ctx->solve_fields();
    const SolveReport field_report = field_outcome.consume(SolveConsumption::kAccept);
    if (!field_report.solved_value_available())
      throw std::logic_error("ProjectAndRecheck test did not receive a solved field value");
    ++consumed_solves;

    if (ctx->min_component(candidate, 0) < Real(3)) {
      ctx->apply_projection(0, candidate);
      ctx->store_history("gas.guard_candidate", candidate);
      ctx->rotate_histories();
      ctx->cache_store_scratch(666002, candidate);
      ctx->record_scalar("project_and_recheck.provisional", Real(1));
      if (ctx->min_component(candidate, 0) < Real(3))
        throw runtime::program::StepAttemptRejected(
            SolveStatus::kIterationLimit, "guard recheck",
            "ProjectAndRecheck candidate remained inadmissible");
    }
    ctx->commit_many({{&state, &candidate}});
  });
  sim.set_program_block_map({0});

  EXPECT_THROW(sim.step(1e-3), runtime::program::StepAttemptRejected);

  EXPECT_EQ(consumed_solves, 1);
  EXPECT_EQ(sim.macro_step(), 0);
  EXPECT_DOUBLE_EQ(sim.time(), 0.0);
  EXPECT_EQ(sim.get_state("gas"), initial);
  EXPECT_FALSE(sim.history_initialized("gas.guard_candidate"));
  EXPECT_FALSE(sim.program_cache().has(666002));
  EXPECT_TRUE(sim.program_diagnostics().empty());
}

TEST(ProgramRuntime, GeneratedUniformBlockSuppliesProjectionRoutesOnlyForCapableModels) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr double gamma = 1.4;
  System<kNativeDimension> system(unit_domain_config<kNativeDimension>(4));
  install_execution_lane(system, "pops.test.program-runtime.generated-projection-routes");
  system.install_block_state_route("gas",
                                   "test.program-runtime.generated-projection-routes.state@1");
  system.seal_auxiliary_providers();

  ProjectingEuler transport;
  transport.gamma = gamma;
  ProjectingGasModel projecting;
  projecting.hyp = transport;
  auto prepared = prepare_compiled_system_block<kNativeDimension>(
      system, "gas", projecting, "none", "rusanov", "conservative", "explicit", gamma,
      /*substeps=*/1, /*evolve=*/true, /*stride=*/1);
  EXPECT_TRUE(static_cast<bool>(prepared.closures.project));
  EXPECT_TRUE(static_cast<bool>(prepared.closures.project_masked));
  EXPECT_TRUE(static_cast<bool>(prepared.closures.staircase.project));
  EXPECT_TRUE(static_cast<bool>(prepared.closures.cut_cell.project));

  GasModel plain;
  plain.hyp = EulerND<kNativeDimension>{gamma};
  auto incapable = prepare_compiled_system_block<kNativeDimension>(
      system, "gas", plain, "none", "rusanov", "conservative", "explicit", gamma,
      /*substeps=*/1, /*evolve=*/true, /*stride=*/1);
  EXPECT_FALSE(static_cast<bool>(incapable.closures.project));
  EXPECT_FALSE(static_cast<bool>(incapable.closures.project_masked));
  EXPECT_FALSE(static_cast<bool>(incapable.closures.staircase.project));
  EXPECT_FALSE(static_cast<bool>(incapable.closures.cut_cell.project));
}

TEST(ProgramRuntime, GeneratedUniformProjectionPreservesEmbeddedBoundaryInactiveCells) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 12;
  constexpr double gamma = 1.4;
  const std::size_t cells = exact_cell_count<kNativeDimension>(n);
  std::vector<double> initial(static_cast<std::size_t>(kGasComponents) * cells);
  fill_ic(initial, n, gamma);

  const auto install_projection_step = [](System<kNativeDimension>& system) {
    system.set_program_block_map({0});
    auto context = runtime::program::make_program_execution_provider(&system);
    context->configure_primary_clock("macro");
    context->install([context](double step) {
      context->begin_step(step);
      context->apply_projection(0, context->state(0));
    });
    system.set_program_block_map({0});
  };

  System<kNativeDimension> cartesian(unit_domain_config<kNativeDimension>(n));
  install_execution_lane(cartesian, "pops.test.program-runtime.generated-projection.cartesian");
  add_generated_projecting_gas(cartesian, gamma);
  cartesian.set_state("gas", initial);
  install_projection_step(cartesian);
  cartesian.step(0.1);
  const auto cartesian_state = cartesian.get_state("gas");

  System<kNativeDimension> cutcell(unit_domain_config<kNativeDimension>(n));
  install_execution_lane(cutcell, "pops.test.program-runtime.generated-projection.cutcell");
  add_generated_projecting_gas(cutcell, gamma);
  cutcell.set_state("gas", initial);
  install_centered_ball(cutcell, 0.31, "cutcell");
  const auto mask = cutcell.embedded_boundary_mask();
  install_projection_step(cutcell);
  cutcell.step(0.1);
  const auto cutcell_state = cutcell.get_state("gas");

  int active_cells = 0;
  int inactive_cells = 0;
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const bool active = mask[cell] >= 0.5;
    active_cells += active ? 1 : 0;
    inactive_cells += active ? 0 : 1;
    EXPECT_DOUBLE_EQ(cartesian_state[cell], 2.0);
    EXPECT_DOUBLE_EQ(cutcell_state[cell], active ? 2.0 : initial[cell]);
    for (int component = 1; component < kGasComponents; ++component) {
      const std::size_t index = static_cast<std::size_t>(component) * cells + cell;
      EXPECT_DOUBLE_EQ(cartesian_state[index], initial[index]);
      EXPECT_DOUBLE_EQ(cutcell_state[index], initial[index]);
    }
  }
  EXPECT_GT(active_cells, 0);
  EXPECT_GT(inactive_cells, 0);
}

TEST(ProgramRuntime, GeneratedUniformProjectionNonFiniteRefusalIsCollectiveAndTransactional) {
  comm_init();
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  constexpr double gamma = 1.4;
  auto config = distributed_boundary_domain_config<kNativeDimension>(n);
  System<kNativeDimension> system(config);
  install_execution_lane(system, "pops.test.program-runtime.generated-projection.nonfinite");
  add_generated_conditional_projecting_gas(system, gamma);
  std::vector<double> initial;
  fill_ic(initial, n, gamma);
  initial[0] = -1.0;
  system.set_state("gas", initial);
  const std::vector<double> accepted = system.get_state("gas");
  system.set_program_block_map({0});

  runtime::program::ProgramContext context(&system);
  context.configure_primary_clock("macro");
  context.begin_step(1.0e-3);
  MultiFab<kNativeDimension>& candidate = context.state(0);
  EXPECT_THROW(context.apply_projection(0, candidate), std::runtime_error);
  EXPECT_EQ(system.get_state("gas"), accepted);
}

TEST(ProgramRuntime, SystemProjectionRefusesForeignFieldContractsBeforeProviderInvocation) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 4;
  constexpr double gamma = 1.4;
  int projection_calls = 0;
  System<kNativeDimension> system(unit_domain_config<kNativeDimension>(n));
  install_execution_lane(system, "pops.test.program-runtime.projection-field-preflight");
  add_projecting_gas(system, gamma, &projection_calls);
  std::vector<double> initial;
  fill_ic(initial, n, gamma);
  system.set_state("gas", initial);

  MultiFab<kNativeDimension>& accepted = system.block_state(0);
  MultiFab<kNativeDimension> wrong_ncomp(accepted.layout(), accepted.distribution(),
                                         accepted.local_rank(), accepted.ncomp() - 1,
                                         accepted.ghosts());
  EXPECT_THROW(system.block_project(0, wrong_ncomp), std::invalid_argument);
  EXPECT_EQ(projection_calls, 0);

  System<kNativeDimension> foreign_system(unit_domain_config<kNativeDimension>(n + 1));
  install_execution_lane(foreign_system,
                         "pops.test.program-runtime.projection-field-preflight.foreign");
  add_projecting_gas(foreign_system, gamma);
  EXPECT_THROW(system.block_project(0, foreign_system.block_state(0)), std::invalid_argument);
  EXPECT_EQ(projection_calls, 0);
  EXPECT_EQ(system.get_state("gas"), initial);
}

TEST(ProgramRuntime, EmbeddedBoundaryRejectsUnqualifiedBoundaryLinearizationEntryPoints) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  using Context = runtime::program::ProgramContext<kNativeDimension>;
  using Field = MultiFab<kNativeDimension>;
  EXPECT_FALSE((HasUnqualifiedBoundaryLinearization<Context, Field>));
}

TEST(ProgramRuntime, UniformPathPreflightPreservesScratchOnRankLocalRefusal) {
  comm_init();
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  constexpr int rate_id = 820001;
  constexpr double gamma = 1.4;
  System<kNativeDimension> system(distributed_boundary_domain_config<kNativeDimension>(n));
  install_execution_lane(system, "pops.test.program-runtime.uniform-path-preflight");
  // A regular block is sufficient: invalid staging must be rejected before any
  // native path closure is invoked. Path-kernel execution has separate witnesses.
  add_gas(system, gamma);
  std::vector<double> initial;
  fill_ic(initial, n, gamma);
  system.set_state("gas", initial);
  const auto accepted_state = system.get_state("gas");
  system.set_program_block_map({0});
  auto context = runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("uniform-path-preflight");
  auto& state = context->state(0);
  auto& residual = context->rhs_scratch(rate_id, 0, state);
  auto foreign_output = context->rhs_scratch_like(state);
  const ExecutionLane& lane = context->prepared_execution_lane();
  int failure = 0;
  bool entered_path_call = false;
  context->install([&](double dt) {
    context->begin_step(dt);
    const bool inject = lane.rank() == 0;
    const int selected_rate = inject && failure == 1 ? -1 : rate_id;
    const double courant = inject && failure == 0 ? 0.0 : context->path_rhs_courant();
    auto& selected_output = inject && failure == 2 ? state
                            : inject && failure == 3 ? foreign_output : residual;
    entered_path_call = true;
    context->path_rhs_into(0, state, selected_output, selected_rate,
                           "test.uniform-path-preflight", courant);
  });

  for (failure = 0; failure != 4; ++failure) {
    SCOPED_TRACE(failure);
    residual.set_val(Real(7));
    foreign_output.set_val(Real(11));
    entered_path_call = false;
    // Peers carry valid staging arguments. The common preflight must refuse on
    // every rank without clearing the output merely to look up its identity.
    EXPECT_THROW(system.step_cfl(0.25, 1.e-12, 1.e-4, 0.0), std::exception);
    EXPECT_TRUE(entered_path_call);
    for (int component = 0; component < residual.ncomp(); ++component) {
      EXPECT_EQ(reduce_min(residual, component), Real(7));
      EXPECT_EQ(reduce_max(residual, component), Real(7));
      EXPECT_EQ(reduce_min(foreign_output, component), Real(11));
      EXPECT_EQ(reduce_max(foreign_output, component), Real(11));
    }
    EXPECT_EQ(system.get_state("gas"), accepted_state);
    EXPECT_DOUBLE_EQ(system.time(), 0.0);
    EXPECT_EQ(system.macro_step(), 0);
    EXPECT_DOUBLE_EQ(system.active_program_step_courant(), 0.0);
  }
}

TEST(ProgramRuntime, PreparedBoundaryResidualAndJvpUseGeneratedBlockClosuresTransactionally) {
  comm_init();
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  constexpr double gamma = 1.4;
  auto config = distributed_boundary_domain_config<kNativeDimension>(n);
  for (int axis = 0; axis < kNativeDimension; ++axis)
    config.periodicity[static_cast<std::size_t>(axis)] = false;
  System<kNativeDimension> system(config);
  install_execution_lane(system, "pops.test.program-runtime.prepared-boundary");
  add_boundary_gas(system, gamma);
  std::vector<double> initial;
  fill_boundary_euler_ic<kNativeDimension>(initial, n, gamma);
  system.set_state("gas", initial);
  system.set_program_block_map({0});

  runtime::program::ProgramContext<kNativeDimension> context(&system);
  context.configure_primary_clock("test.prepared-boundary");
  context.begin_step(1.0e-3);
  const auto point = context.boundary_evaluation_point(17);
  const ExecutionLane& lane = system.prepared_boundary_execution_lane();
  MultiFab<kNativeDimension>& state = context.state(0);
  const auto session = context.prepare_block_boundary_session(0, state, point, lane);
  ASSERT_TRUE(context.has_boundary_linearization(0));

  MultiFab<kNativeDimension> residual = context.rhs_scratch_like(state);
  residual.set_val(Real(0));
  ASSERT_NO_THROW(context.boundary_residual_into_at(point, 0, state, residual, *session));
  EXPECT_GT(norm_inf_all_components(residual), Real(0));

  MultiFab<kNativeDimension> direction = context.rhs_scratch_like(state);
  set_boundary_jvp_direction(direction);
  MultiFab<kNativeDimension> jvp = context.rhs_scratch_like(state);
  jvp.set_val(Real(0));
  ASSERT_NO_THROW(context.boundary_jvp_into_at(point, 0, state, direction, jvp, *session));
  EXPECT_GT(norm_inf_all_components(jvp), Real(0));

  MultiFab<kNativeDimension> scaled_direction = context.rhs_scratch_like(state);
  lincomb(scaled_direction, Real(3), direction, Real(0), direction);
  MultiFab<kNativeDimension> scaled_jvp = context.rhs_scratch_like(state);
  scaled_jvp.set_val(Real(0));
  ASSERT_NO_THROW(
      context.boundary_jvp_into_at(point, 0, state, scaled_direction, scaled_jvp, *session));
  MultiFab<kNativeDimension> scaling_error = context.rhs_scratch_like(state);
  lincomb(scaling_error, Real(1), scaled_jvp, Real(-3), jvp);
  EXPECT_LT(norm_inf_all_components(scaling_error),
            Real(5.0e-5) * (Real(1) + norm_inf_all_components(scaled_jvp)));

  const std::vector<double> accepted = system.get_state("gas");
  MultiFab<kNativeDimension> non_finite_state = context.rhs_scratch_like(state);
  non_finite_state.set_val(std::numeric_limits<Real>::quiet_NaN());
  residual.set_val(Real(-7));
  EXPECT_THROW(context.boundary_residual_into_at(point, 0, non_finite_state, residual, *session),
               std::runtime_error);
  EXPECT_EQ(system.get_state("gas"), accepted);
  EXPECT_EQ(context.norm_inf(0, residual), Real(7));

  MultiFab<kNativeDimension> non_finite_direction = context.rhs_scratch_like(state);
  non_finite_direction.set_val(std::numeric_limits<Real>::quiet_NaN());
  jvp.set_val(Real(-8));
  EXPECT_THROW(context.boundary_jvp_into_at(point, 0, state, non_finite_direction, jvp, *session),
               std::runtime_error);
  EXPECT_EQ(system.get_state("gas"), accepted);
  EXPECT_EQ(context.norm_inf(0, jvp), Real(8));

  residual.set_val(Real(-3));
  auto mismatched_point = point;
  if (lane.rank() != 0)
    ++mismatched_point.stage;
  if (lane.size() == 1)
    ++mismatched_point.stage;
  if (lane.size() == 1) {
    EXPECT_THROW(context.boundary_residual_into_at(mismatched_point, 0, state, residual, *session),
                 std::invalid_argument);
  } else {
    EXPECT_THROW(context.boundary_residual_into_at(mismatched_point, 0, state, residual, *session),
                 std::runtime_error);
  }
  EXPECT_EQ(system.get_state("gas"), accepted);
  EXPECT_EQ(context.norm_inf(0, residual), Real(3));

  residual.set_val(Real(-4));
  if (lane.size() == 1) {
    EXPECT_THROW(context.rhs_core_into_at(mismatched_point, 0, state, residual, false, *session),
                 std::invalid_argument);
  } else {
    EXPECT_THROW(context.rhs_core_into_at(mismatched_point, 0, state, residual, false, *session),
                 std::runtime_error);
  }
  EXPECT_EQ(system.get_state("gas"), accepted);
  EXPECT_EQ(context.norm_inf(0, residual), Real(4));
  EXPECT_NO_THROW(context.rhs_core_into_at(point, 0, state, residual, false, *session));
  EXPECT_GT(norm_inf_all_components(residual), Real(0));

  residual.set_val(Real(-3));

  System<kNativeDimension> foreign_system(config);
  install_execution_lane(foreign_system, "pops.test.program-runtime.foreign-prepared-boundary");
  add_boundary_gas(foreign_system, gamma);
  foreign_system.set_state("gas", initial);
  foreign_system.set_program_block_map({0});
  runtime::program::ProgramContext<kNativeDimension> foreign_context(&foreign_system);
  foreign_context.configure_primary_clock("test.prepared-boundary");
  foreign_context.begin_step(1.0e-3);
  MultiFab<kNativeDimension>& foreign_state = foreign_context.state(0);
  const auto foreign_session = foreign_context.prepare_block_boundary_session(
      0, foreign_state, point, foreign_system.prepared_boundary_execution_lane());
  if (lane.size() == 1) {
    EXPECT_THROW(context.boundary_residual_into_at(point, 0, state, residual, *foreign_session),
                 std::invalid_argument);
  } else {
    EXPECT_THROW(context.boundary_residual_into_at(point, 0, state, residual, *foreign_session),
                 std::runtime_error);
  }
  EXPECT_EQ(system.get_state("gas"), accepted);
  EXPECT_EQ(context.norm_inf(0, residual), Real(3));

  if (lane.size() > 1) {
    const int divergent_block = lane.rank() == 0 ? 0 : 1;
    EXPECT_THROW(
        context.boundary_residual_into_at(point, divergent_block, state, residual, *session),
        std::runtime_error);
    EXPECT_EQ(system.get_state("gas"), accepted);
    EXPECT_EQ(context.norm_inf(0, residual), Real(3));
  }

  EXPECT_NO_THROW(context.boundary_residual_into_at(point, 0, state, residual, *session));
  EXPECT_GT(norm_inf_all_components(residual), Real(0));
}

TEST(ProgramRuntime, AnalyticMappedInitialFailureDoesNotPublishTheCandidate) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  System<kNativeDimension> system(unit_domain_config<kNativeDimension>(n));
  install_execution_lane(system, "pops.test.program-runtime.analytic-mapped-failure");
  add_scalar(system);

  const std::vector<double> accepted(exact_cell_count<kNativeDimension>(n), 0.25);
  system.set_state("tracer", accepted);
  EXPECT_THROW(
      system.set_analytic_mapped_state(
          "tracer", {{"input", "constant", "add"}}, {{0.0, 1.0, 0.0}},
          {runtime::system::AnalyticMappedInput::provider({"test.owner", "field", "phi", "value"})},
          "test.tracer/initial-map"),
      std::logic_error);
  EXPECT_EQ(system.get_state("tracer"), accepted);

  EXPECT_EQ(system.set_analytic_expression_state(
                "tracer", "cell", "cell", "conservative_cell_average", {{"constant"}}, {{0.5}}),
            static_cast<std::int64_t>(exact_cell_count<kNativeDimension>(n)));
  for (const double value : system.get_state("tracer"))
    EXPECT_NEAR(value, 0.5, 8.0 * std::numeric_limits<double>::epsilon());
}

TEST(ProgramRuntime, AnalyticInitialStatePublishesInTheFinalTypedRuntime) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  System<kNativeDimension> system(unit_domain_config<kNativeDimension>(n));
  install_execution_lane(system, "pops.test.program-runtime.analytic-initial-state");
  add_scalar(system);
  EXPECT_EQ(system.set_analytic_expression_state(
                "tracer", "cell", "cell", "conservative_cell_average", {{"constant"}}, {{0.5}}),
            static_cast<std::int64_t>(exact_cell_count<kNativeDimension>(n)));
  for (const double value : system.get_state("tracer"))
    EXPECT_NEAR(value, 0.5, 8.0 * std::numeric_limits<double>::epsilon());
}

TEST(ProgramRuntime, RejectedAttemptRestoresStateHistoryCacheDiagnosticsAndClock) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  constexpr double gamma = 1.4;
  auto cfg = unit_domain_config<kNativeDimension>(n);

  System<kNativeDimension> sim(cfg);
  install_execution_lane(sim, "pops.test.program-runtime.rejected-attempt-rollback");
  add_gas(sim, gamma);
  std::vector<double> initial;
  fill_ic(initial, n, gamma);
  sim.set_state("gas", initial);
  // Global reads are gathered on the root; non-root ranks retain an empty
  // global projection and must still participate in every collective read.
  const auto accepted_state = sim.get_state("gas");
  sim.set_program_block_map({0});

  auto ctx = runtime::program::make_program_execution_provider(&sim);
  ctx->register_history("gas.U", 2, kGasComponents);
  ctx->install([ctx](double dt) {
    ctx->begin_step(dt);
    MultiFab<kNativeDimension>& state = ctx->state(0);
    MultiFab<kNativeDimension> bump = state;
    bump.set_val(Real(dt));
    ctx->axpy(state, Real(1), bump);
    ctx->store_history("gas.U", state);
    ctx->rotate_histories();
    ctx->cache_store_scratch(17, state);
    ctx->record_scalar("provisional", Real(42));
    throw runtime::program::StepAttemptRejected(SolveStatus::kIterationLimit, "solve",
                                                "fault injection after provisional publications");
  });
  sim.set_program_block_map({0});

  EXPECT_THROW(sim.step(1e-3), runtime::program::StepAttemptRejected);
  EXPECT_EQ(sim.macro_step(), 0);
  EXPECT_DOUBLE_EQ(sim.time(), 0.0);
  EXPECT_EQ(sim.get_state("gas"), accepted_state);
  EXPECT_FALSE(sim.history_initialized("gas.U"));
  EXPECT_FALSE(sim.program_cache().has(17));
  EXPECT_TRUE(sim.program_diagnostics().empty());
}

TEST(ProgramRuntime, NestedChildCommitThenParentRejectRestoresDurationAndExchangeMailbox) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  constexpr int node = 17;
  constexpr double skipped_dt = 0.25;
  System<kNativeDimension> sim(unit_domain_config<kNativeDimension>(n));
  install_execution_lane(sim, "pops.test.program-runtime.nested-duration-rollback");
  add_gas(sim, 1.4);
  std::vector<double> initial;
  fill_ic(initial, n, 1.4);
  sim.set_state("gas", initial);
  const auto accepted_state = sim.get_state("gas");
  sim.set_program_block_map({0});
  auto ctx = runtime::program::make_program_execution_provider(&sim);
  ctx->configure_primary_clock("test.nested-duration-rollback");
  ctx->register_history("gas.U", 2, kGasComponents);
  ctx->cache_store_scratch(node, ctx->state(0));
  ctx->cache_accumulate_dt(node, Real(skipped_dt));
  const auto initial_cache = sim.program_cache_global(node);

  bool fail_after_effective_dt = true;
  std::vector<Real> observed_effective_dt;
  std::vector<std::string> observed_exchange_contexts;
  bool duplicate_rejected = false;
  std::string duplicate_error;
  ctx->install([&](double dt) {
    ctx->begin_step(dt);
    const Real effective_dt = ctx->cache_effective_dt(node, Real(dt));
    observed_effective_dt.push_back(effective_dt);
    MultiFab<kNativeDimension>& state = ctx->state(0);
    MultiFab<kNativeDimension> bump = state;
    // A duration-dependent contribution is deliberately computed from this attempt's
    // accumulated native duration, rather than from an author-side date-keyed value.
    bump.set_val(effective_dt);
    ctx->axpy(state, Real(dt), bump);
    ctx->store_history("gas.U", state);
    ctx->rotate_histories();
    ctx->cache_store_scratch(node, state);
    const runtime::program::ExchangeRecord contribution{
        "transport.face", "face.0", "stage.0", "euler", 1, 1.0,
        static_cast<double>(effective_dt), dt, 1};
    ctx->stage_exchange(contribution);
    const auto staged = sim.program_exchange_records();
    observed_exchange_contexts.push_back(staged.empty() ? std::string{}
                                                        : staged.back().evaluation_context);
    if (dt == 0.3) {
      try {
        ctx->stage_exchange(contribution);
      } catch (const std::exception& error) {
        duplicate_rejected = true;
        duplicate_error = error.what();
      }
    }
    if (fail_after_effective_dt)
      throw runtime::program::StepAttemptRejected(
          SolveStatus::kIterationLimit, "injected",
          "failure after consuming cached duration and staging native outputs");
  });
  sim.set_program_block_map({0});

  const auto expect_initial = [&] {
    EXPECT_EQ(sim.get_state("gas"), accepted_state);
    EXPECT_EQ(sim.time(), 0.0);
    EXPECT_EQ(sim.macro_step(), 0);
    EXPECT_FALSE(sim.history_initialized("gas.U"));
    EXPECT_EQ(sim.history_fill_count("gas.U"), 0);
    EXPECT_TRUE(sim.program_cache().has(node));
    EXPECT_DOUBLE_EQ(sim.program_cache().accumulated_dt_of(node), skipped_dt);
    EXPECT_EQ(sim.program_cache_global(node), initial_cache);
    EXPECT_TRUE(sim.program_exchange_records().empty());
  };
  expect_initial();

  sim.begin_step_transaction();
  sim.begin_nested_step_transaction();
  EXPECT_THROW(sim.step(0.1), runtime::program::StepAttemptRejected);
  EXPECT_EQ(observed_effective_dt.size(), 1u);
  if (!observed_effective_dt.empty())
    EXPECT_DOUBLE_EQ(observed_effective_dt.back(), Real(skipped_dt + 0.1));
  EXPECT_EQ(observed_exchange_contexts.size(), 1u);
  expect_initial();

  fail_after_effective_dt = false;
  sim.step(0.2);
  EXPECT_EQ(observed_effective_dt.size(), 2u);
  if (!observed_effective_dt.empty())
    EXPECT_DOUBLE_EQ(observed_effective_dt.back(), Real(skipped_dt + 0.2));
  EXPECT_EQ(observed_exchange_contexts.size(), 2u);
  if (observed_exchange_contexts.size() == 2u)
    EXPECT_NE(observed_exchange_contexts[0], observed_exchange_contexts[1]);
  EXPECT_EQ(sim.macro_step(), 1);
  EXPECT_DOUBLE_EQ(sim.time(), 0.2);
  EXPECT_TRUE(sim.history_initialized("gas.U"));
  EXPECT_EQ(sim.history_fill_count("gas.U"), 1);
  EXPECT_DOUBLE_EQ(sim.program_cache().accumulated_dt_of(node), 0.0);
  const auto child_records = sim.program_exchange_records();
  EXPECT_EQ(child_records.size(), 1u);
  if (child_records.size() == 1u) {
    if (observed_exchange_contexts.size() == 2u)
      EXPECT_EQ(child_records[0].evaluation_context, observed_exchange_contexts[1]);
    EXPECT_DOUBLE_EQ(child_records[0].numerical_flux, skipped_dt + 0.2);
    EXPECT_DOUBLE_EQ(child_records[0].temporal_weight, 0.2);
    EXPECT_DOUBLE_EQ(child_records[0].integrated_amount(), (skipped_dt + 0.2) * 0.2);
  }
  const auto child_state = sim.get_state("gas");
  EXPECT_EQ(child_state.size(), accepted_state.size());
  for (std::size_t i = 0; i < std::min(child_state.size(), accepted_state.size()); ++i)
    EXPECT_DOUBLE_EQ(child_state[i], accepted_state[i] + Real(0.2) * Real(skipped_dt + 0.2));
  // History/cache snapshots are replicated; get_state returns its global array
  // only on rank zero. Every rank must still enter both snapshot collectives.
  const auto child_history = sim.history_global("gas.U", 1);
  const auto child_cache = sim.program_cache_global(node);
  EXPECT_EQ(child_history, child_cache);
  if (sim.prepared_boundary_execution_lane().rank() == 0) {
    EXPECT_EQ(child_state.size(), initial.size());
    EXPECT_EQ(child_history, child_state);
  }
  sim.commit_step_transaction();
  sim.finalize_step_transaction();
  EXPECT_EQ(sim.step_transaction_depth(), 1u);
  sim.rollback_step_transaction();
  EXPECT_EQ(sim.step_transaction_depth(), 0u);
  expect_initial();

  // The parent rejection revokes the accepted child contribution. A fresh root attempt with
  // another duration must use the restored skipped interval and a distinct runtime exchange key.
  sim.step(0.3);
  EXPECT_EQ(observed_effective_dt.size(), 3u);
  if (!observed_effective_dt.empty())
    EXPECT_DOUBLE_EQ(observed_effective_dt.back(), Real(skipped_dt + 0.3));
  EXPECT_EQ(observed_exchange_contexts.size(), 3u);
  if (observed_exchange_contexts.size() == 3u) {
    const auto encoded_dt = [](double duration) {
      return "/" + std::to_string(std::bit_cast<std::uint64_t>(duration)) + "/";
    };
    EXPECT_NE(observed_exchange_contexts[0].find(encoded_dt(0.1)), std::string::npos);
    EXPECT_NE(observed_exchange_contexts[1].find(encoded_dt(0.2)), std::string::npos);
    EXPECT_NE(observed_exchange_contexts[2].find(encoded_dt(0.3)), std::string::npos);
    EXPECT_NE(observed_exchange_contexts[2], observed_exchange_contexts[0]);
    EXPECT_NE(observed_exchange_contexts[2], observed_exchange_contexts[1]);
  }
  EXPECT_TRUE(duplicate_rejected);
  EXPECT_NE(duplicate_error.find("duplicate accepted exchange occurrence/quadrature contribution"),
            std::string::npos) << duplicate_error;
  EXPECT_EQ(sim.macro_step(), 1);
  EXPECT_DOUBLE_EQ(sim.time(), 0.3);
  EXPECT_TRUE(sim.history_initialized("gas.U"));
  EXPECT_EQ(sim.history_fill_count("gas.U"), 1);
  EXPECT_DOUBLE_EQ(sim.program_cache().accumulated_dt_of(node), 0.0);
  const auto accepted_records = sim.program_exchange_records();
  EXPECT_EQ(accepted_records.size(), 1u);
  if (accepted_records.size() == 1u) {
    if (observed_exchange_contexts.size() == 3u)
      EXPECT_EQ(accepted_records[0].evaluation_context, observed_exchange_contexts[2]);
    EXPECT_DOUBLE_EQ(accepted_records[0].numerical_flux, skipped_dt + 0.3);
    EXPECT_DOUBLE_EQ(accepted_records[0].temporal_weight, 0.3);
    EXPECT_DOUBLE_EQ(accepted_records[0].integrated_amount(), (skipped_dt + 0.3) * 0.3);
  }
  const auto retried_state = sim.get_state("gas");
  EXPECT_EQ(retried_state.size(), accepted_state.size());
  for (std::size_t i = 0; i < std::min(retried_state.size(), accepted_state.size()); ++i)
    EXPECT_DOUBLE_EQ(retried_state[i], accepted_state[i] + Real(0.3) * Real(skipped_dt + 0.3));
  const auto retried_history = sim.history_global("gas.U", 1);
  const auto retried_cache = sim.program_cache_global(node);
  EXPECT_EQ(retried_history, retried_cache);
  if (sim.prepared_boundary_execution_lane().rank() == 0) {
    EXPECT_EQ(retried_state.size(), initial.size());
    EXPECT_EQ(retried_history, retried_state);
  }
}

#include "moving_interval_projection_tests.inc"
#include "moving_interval_codec_independent_review.inc"
#include "integral_candidate_capture_tests.inc"
#include "../../../review/sol61_t5_public_native_capture.inc"

TEST(ProgramRuntime, MovingIntervalsPublishRealGeometryStateAndLedgerThenParentRollback) {
  if constexpr (kNativeDimension != 1) {
    GTEST_SKIP() << "The first moving-interval provider requires a native Dim1 build";
  }
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  for (int n : {8, 24}) for (bool permute : {false, true}) {
    System<kNativeDimension> sim(distributed_boundary_domain_config<kNativeDimension>(n));
    install_execution_lane(sim, "pops.test.moving-interval.native-transaction");
    add_gas(sim, 1.4);
    std::array<Real, kGasComponents> initial{};
    for (int component = 0; component < kGasComponents; ++component)
      initial[component] = component == 0 ? Real(2.3) :
                           component == kGasComponents - 1 ? Real(6) : Real(0);
    if (permute) std::swap(initial[0], initial[kGasComponents - 1]);
    std::vector<double> values(static_cast<std::size_t>(n) * kGasComponents);
    for (int component = 0; component < kGasComponents; ++component)
      for (int i = 0; i < n; ++i) values[static_cast<std::size_t>(component*n + i)] = initial[component];
    sim.set_state("gas", values);
    sim.set_program_block_map({0});
    auto ctx = runtime::program::make_program_execution_provider(&sim);
    ctx->configure_primary_clock("ale-clock");
    ctx->initialize_moving_interval_geometry("mesh", 0, "unit-interval",
        Real(128)*std::numeric_limits<Real>::epsilon());
    const auto initial_geometry_image=sim.checkpoint_program_exchanges();
    ASSERT_GE(initial_geometry_image.size(),8u);
    EXPECT_EQ(std::string(initial_geometry_image.begin(),initial_geometry_image.begin()+8),"POPSEX03");
    EXPECT_NO_THROW(sim.validate_checkpoint_program_exchanges(initial_geometry_image));
    EXPECT_ANY_THROW(ctx->commit_many({{&ctx->state(0), &ctx->state(0)}}));
    const auto original = sim.get_state("gas");
    const auto original_geometry = ctx->moving_interval_geometry("mesh");
    bool reject_after_publication = true;
    bool reuse_rejected_evaluation = false;
    bool partial_interval = false, change_input_after_prepare = false;
    int change_geometry_after_prepare = 0;
    std::optional<runtime::program::RuntimeIntervalEvaluation<kNativeDimension>> cached_evaluation;
    ctx->install([&](double dt) {
      ctx->begin_step(dt);
      if (reuse_rejected_evaluation) {
        ctx->advance_moving_intervals(*cached_evaluation, Real(1e-13));
        return;
      }
      if (partial_interval) ctx->set_stage_time(1, 2);
      auto evaluation = ctx->evaluate_moving_interval("mesh", 0, "unit-interval", "endpoint-swept@1",
          [&](const runtime::multiblock::BoundaryEvaluationPoint& point) {
      const auto& accepted = ctx->moving_interval_geometry("mesh");
      auto proposed = accepted.coordinates;
      auto sweeps = accepted.swept_volumes;
      std::vector<nd::FaceField<kNativeDimension>> physical, density;
      auto source = ctx->scratch_state_like(ctx->state(0));
      source.set_val(Real(0));
      for (std::size_t patch = 0; patch < ctx->state(0).local_size(); ++patch) {
        const auto box = ctx->state(0).box(patch);
        physical.emplace_back(box, kGasComponents);
        density.emplace_back(box, kGasComponents);
        const auto old = accepted.coordinates[patch].template field<0>().view();
        const auto pos = proposed[patch].template field<0>().view();
        const auto sw = sweeps[patch].template field<0>().view();
        const auto flux = physical.back().template field<0>().view();
        const auto trace = density.back().template field<0>().view();
        const auto geom = ctx->geometry();
        const double next_time = point.physical_time + point.dt;
        const auto constants = initial;
        for_each_cell(nd::face_box(box, 0), [=] POPS_HD(const Index<kNativeDimension>& face) {
          const Real reference = geom.face_coordinate(0, face[0]);
          const Real shape = face[0] == geom.domain().lo[0] ||
                             face[0] == geom.domain().hi[0] + 1 ? Real(0) :
                             Kokkos::sin(Real(6.2831853071795864769) * reference);
          pos(face) = reference + Real(.08) * shape * Real(next_time);
          sw(face) = pos(face) - old(face);
          for (int component = 0; component < kGasComponents; ++component) {
            flux(face, component) = Real(0);
            trace(face, component) = constants[component];
          }
        });
      }
      return runtime::program::MovingIntervalInputs<kNativeDimension>{
          std::move(proposed), std::move(sweeps), std::move(physical), std::move(density), std::move(source)};
      });
      cached_evaluation = evaluation;
      auto proposal = ctx->prepare_moving_interval_update(evaluation,
          Real(128) * std::numeric_limits<Real>::epsilon());
      // Preparing an SSA candidate changes neither live geometry nor ledger.
      EXPECT_EQ(ctx->moving_interval_geometry("mesh").generation, 0u);
      EXPECT_TRUE(sim.program_exchange_records().empty());
      if (change_input_after_prepare) ctx->state(0).set_val(Real(100));
      if (change_geometry_after_prepare && ctx->prepared_execution_lane().rank() == 0) {
        auto& mutable_geometry = ctx->runtime_state().moving_interval_geometry_.at("mesh");
        if (change_geometry_after_prepare == 1) mutable_geometry.physical_frame = "forged-frame";
        if (change_geometry_after_prepare == 2) mutable_geometry.measures.set_val(Real(9));
        if (change_geometry_after_prepare == 3 && !mutable_geometry.coordinates.empty())
          mutable_geometry.coordinates.front().set_val(Real(9));
        if (change_geometry_after_prepare == 4) mutable_geometry.last_interval = "forged-interval";
        if (change_geometry_after_prepare == 5) {
          MultiFab<kNativeDimension> changed(mutable_geometry.measures.layout(),
              mutable_geometry.measures.distribution(), mutable_geometry.measures.local_rank(),
              2, mutable_geometry.measures.ghosts());
          changed.set_val(Real(0));
          for (std::size_t patch=0; patch<changed.local_size(); ++patch) {
            const auto old=mutable_geometry.measures.fab(patch).view();
            const auto replacement=changed.fab(patch).view();
            for_each_cell(changed.box(patch),[=] POPS_HD(const Index<kNativeDimension>& cell) {
              replacement(cell,0)=old(cell,0);
            });
          }
          Kokkos::fence(); mutable_geometry.measures=std::move(changed);
        }
      }
      ctx->commit_moving_interval(proposal);
      if (reject_after_publication)
        throw runtime::program::StepAttemptRejected(SolveStatus::kIterationLimit, "ale-post-publication");
    });
    // install resets artifact-owned mappings; bind this explicit map afterwards.
    sim.set_program_block_map({0});
    const auto expect_original = [&] {
      EXPECT_EQ(sim.get_state("gas"), original);
      EXPECT_EQ(sim.time(), 0.);
      const auto& actual = ctx->moving_interval_geometry("mesh");
      EXPECT_EQ(actual.generation, 0u);
      EXPECT_EQ(actual.physical_frame, "unit-interval");
      EXPECT_EQ(actual.measures.ncomp(),1);
      EXPECT_TRUE(actual.last_interval.empty());
      EXPECT_FALSE(actual.last_receipt.has_value());
      EXPECT_TRUE(sim.program_exchange_records().empty());
      for (std::size_t patch = 0; patch < actual.coordinates.size(); ++patch) {
        const auto coords = actual.coordinates[patch].template field<0>().create_host_mirror();
        const auto old = original_geometry.coordinates[patch].template field<0>().create_host_mirror();
        actual.coordinates[patch].template field<0>().copy_to_host(coords);
        original_geometry.coordinates[patch].template field<0>().copy_to_host(old);
        for (std::size_t i = 0; i < coords.size(); ++i) EXPECT_DOUBLE_EQ(coords(i), old(i));
      }
    };
    sim.begin_step_transaction();
    EXPECT_THROW(sim.checkpoint_program_exchanges(true),std::logic_error);
    sim.begin_nested_step_transaction();
    EXPECT_THROW(sim.step(.1), runtime::program::StepAttemptRejected);
    expect_original();
    reject_after_publication = false;
    // Coordinates/sweeps/physical quantities were mutually consistent in .1;
    // their revoked attempt cannot be relabeled for a .15 retry.
    reuse_rejected_evaluation = true;
    EXPECT_ANY_THROW(sim.step(.15));
    expect_original();
    reuse_rejected_evaluation = false;
    partial_interval = true;
    EXPECT_ANY_THROW(sim.step(.17));
    expect_original();
    partial_interval = false;
    change_input_after_prepare = true;
    EXPECT_ANY_THROW(sim.step(.19));
    expect_original();
    change_input_after_prepare = false;
    for (int mutation : {1, 2, 3, 4, 5}) {
      change_geometry_after_prepare = mutation;
      EXPECT_ANY_THROW(sim.step(.19));
      expect_original();
    }
    change_geometry_after_prepare = 0;
    sim.step(.2);
    EXPECT_EQ(ctx->moving_interval_geometry("mesh").generation, 1u);
    EXPECT_THROW(sim.checkpoint_program_exchanges(true),std::logic_error);
    const auto child_interval = ctx->moving_interval_geometry("mesh").last_interval;
    sim.commit_step_transaction();
    sim.finalize_step_transaction();
    EXPECT_THROW(sim.checkpoint_program_exchanges(),std::logic_error);
    const auto parent_staged_geometry_image=sim.checkpoint_program_exchanges(true);
    sim.rollback_step_transaction();
    expect_original();
    EXPECT_EQ(sim.checkpoint_program_exchanges(),initial_geometry_image);
    EXPECT_NE(sim.checkpoint_program_exchanges(),parent_staged_geometry_image);
    sim.begin_step_transaction();
    sim.step(.3);
    const auto& actual = ctx->moving_interval_geometry("mesh");
    EXPECT_EQ(actual.generation, 1u);
    EXPECT_NE(actual.last_interval, child_interval);
    const auto& lane = sim.prepared_boundary_execution_lane();
    const double total_measure = all_reduce_sum(pops::reduce_sum_local(actual.measures, 0), lane);
    EXPECT_NEAR(total_measure, 1., 3e-13);
    // Independently reconstruct Q and V from the real accepted cell/face data.
    for (std::size_t patch = 0; patch < ctx->state(0).local_size(); ++patch) {
      const auto coords = actual.coordinates[patch].template field<0>().create_host_mirror();
      const auto measures = actual.measures.fab(patch).create_host_mirror();
      const auto state = ctx->state(0).fab(patch).create_host_mirror();
      actual.coordinates[patch].template field<0>().copy_to_host(coords);
      actual.measures.fab(patch).copy_to_host(measures);
      ctx->state(0).fab(patch).copy_to_host(state);
      const auto view = ctx->state(0).fab(patch).view();
      const auto box = ctx->state(0).box(patch);
      for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
        const auto local = static_cast<std::size_t>(i - box.lo[0]);
        const Real volume = coords(local + 1) - coords(local);
        EXPECT_DOUBLE_EQ(measures(local), volume);
        for (int component = 0; component < kGasComponents; ++component) {
          const auto offset = static_cast<std::size_t>(i-view.origin[0]) * view.strides[0] +
                              static_cast<std::size_t>(component) * view.component_stride;
          EXPECT_NEAR(state(offset), initial[component], 2e-12);
          EXPECT_NEAR(state(offset)*volume, initial[component]*volume, 2e-12);
        }
      }
    }
    const auto staged_geometry_image=sim.checkpoint_program_exchanges(true);
    EXPECT_THROW(sim.checkpoint_program_exchanges(),std::logic_error);
    const auto records = sim.program_exchange_records();
    EXPECT_EQ(all_reduce_sum(static_cast<long>(records.size()), lane),
              static_cast<long>(n) * (2 + 3*kGasComponents));
    for (int component = 0; component < kGasComponents; ++component) {
      double net = 0;
      for (const auto& record : records)
        if (record.operation_identity == "amount:mesh/component:" + std::to_string(component))
          net += record.integrated_amount();
      EXPECT_NEAR(all_reduce_sum(net, lane), 0., 3e-13);
    }
    sim.commit_step_transaction();
    sim.finalize_step_transaction();
    const auto saved_geometry_image=sim.checkpoint_program_exchanges();
    EXPECT_EQ(saved_geometry_image,staged_geometry_image);
    EXPECT_NO_THROW(sim.validate_checkpoint_program_exchanges(saved_geometry_image));
    EXPECT_NO_THROW(sim.validate_checkpoint_moving_geometry(saved_geometry_image,.3,1));
    EXPECT_ANY_THROW(sim.validate_checkpoint_moving_geometry(saved_geometry_image,.31,1));
    EXPECT_ANY_THROW(sim.validate_checkpoint_moving_geometry(saved_geometry_image,.3,2));
    auto truncated_geometry_image=saved_geometry_image;
    if (sim.prepared_boundary_execution_lane().rank()==0) truncated_geometry_image.pop_back();
    sim.begin_restart_transaction();
    EXPECT_THROW(sim.checkpoint_program_exchanges(true),std::logic_error);
    EXPECT_ANY_THROW(sim.restore_checkpoint_program_exchanges(truncated_geometry_image));
    sim.rollback_restart_transaction();
    EXPECT_EQ(sim.checkpoint_program_exchanges(),saved_geometry_image);
    sim.begin_restart_transaction();
    ctx->state(0).set_val(Real(1));
    EXPECT_ANY_THROW(sim.restore_checkpoint_program_exchanges(saved_geometry_image));
    sim.rollback_restart_transaction();
    EXPECT_EQ(sim.checkpoint_program_exchanges(),saved_geometry_image);
    sim.begin_restart_transaction();
    ctx->runtime_state().moving_interval_geometry_.at("mesh")=original_geometry;
    EXPECT_NO_THROW(sim.restore_checkpoint_program_exchanges(saved_geometry_image));
    EXPECT_EQ(ctx->moving_interval_geometry("mesh").generation,1u);
    ASSERT_TRUE(ctx->moving_interval_geometry("mesh").last_receipt.has_value());
    EXPECT_DOUBLE_EQ(ctx->moving_interval_geometry("mesh").last_receipt->point.dt,.3);
    sim.rollback_restart_transaction();
    EXPECT_EQ(sim.checkpoint_program_exchanges(),saved_geometry_image);
  }
}

TEST(ProgramRuntime, MovingIntervalsRejectStaleDurationSweepsCollectivelyBeforePublication) {
  if constexpr (kNativeDimension != 1) {
    GTEST_SKIP() << "The moving-interval provider requires a native Dim1 build";
  }
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  System<kNativeDimension> sim(distributed_boundary_domain_config<kNativeDimension>(16));
  install_execution_lane(sim, "pops.test.moving-interval.stale-sweep");
  add_gas(sim, 1.4);
  std::vector<double> initial;
  fill_ic(initial, 16, 1.4);
  sim.set_state("gas", initial);
  sim.set_program_block_map({0});
  auto ctx = runtime::program::make_program_execution_provider(&sim);
  ctx->configure_primary_clock("ale-stale-clock");
  ctx->initialize_moving_interval_geometry("mesh", 0, "unit-interval",Real(1e-13));
  const auto before = sim.get_state("gas");
  bool stale = true;
  bool inadmissible = false;
  bool physical_source_evaluated = false;
  ctx->install([&](double dt) {
    ctx->begin_step(dt);
    auto evaluation = ctx->evaluate_moving_interval("mesh", 0, "unit-interval", "endpoint-swept@1",
        [&](const runtime::multiblock::BoundaryEvaluationPoint& point) {
    const double interval_dt = point.dt;
    const auto& accepted = ctx->moving_interval_geometry("mesh");
    auto proposed = accepted.coordinates;
    auto sweep = accepted.swept_volumes;
    std::vector<nd::FaceField<kNativeDimension>> physical, density;
    auto source = ctx->scratch_state_like(ctx->state(0)); source.set_val(0);
    for (std::size_t patch = 0; patch < ctx->state(0).local_size(); ++patch) {
      const auto box = ctx->state(0).box(patch);
      physical.emplace_back(box, kGasComponents); physical.back().set_val(0);
      density.emplace_back(box, kGasComponents); density.back().set_val(1);
      const auto pos = proposed[patch].template field<0>().view();
      const auto sw = sweep[patch].template field<0>().view();
      const auto old = accepted.coordinates[patch].template field<0>().view();
      const auto geom = ctx->geometry();
      const bool use_stale = stale;
      for_each_cell(nd::face_box(box, 0), [=] POPS_HD(const Index<kNativeDimension>& face) {
        const Real x = geom.face_coordinate(0, face[0]);
        const Real shift = face[0] == geom.domain().lo[0] ||
                           face[0] == geom.domain().hi[0] + 1 ? Real(0) :
                           Real(.1) * Real(interval_dt) * Kokkos::sin(Real(6.2831853071795864769)*x);
        pos(face) = old(face) + shift;
        sw(face) = use_stale ? shift/Real(2) : shift;
      });
      if (inadmissible) {
        physical_source_evaluated = true;
        const auto initial_state = ctx->state(0).fab(patch).view();
        const auto volume = accepted.measures.fab(patch).view();
        const auto amount = source.fab(patch).view();
        for_each_cell(box, [=] POPS_HD(const Index<kNativeDimension>& cell) {
          amount(cell, 0) = -(initial_state(cell, 0) + Real(10)) * volume(cell);
        });
      }
    }
    return runtime::program::MovingIntervalInputs<kNativeDimension>{
        std::move(proposed), std::move(sweep), std::move(physical), std::move(density), std::move(source)};
    });
    ctx->advance_moving_intervals(evaluation, Real(1e-13));
  });
  sim.set_program_block_map({0});
  sim.begin_step_transaction();
  EXPECT_THROW(sim.step(.2), runtime::program::StepAttemptRejected);
  EXPECT_EQ(sim.get_state("gas"), before);
  EXPECT_EQ(sim.time(), 0.);
  EXPECT_EQ(ctx->moving_interval_geometry("mesh").generation, 0u);
  EXPECT_TRUE(sim.program_exchange_records().empty());
  stale = false;
  inadmissible = true;
  try {
    sim.step(.25);
    FAIL() << "The negative-density proposal must fail physical recovery";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("System Program terminal state publication"), std::string::npos);
    EXPECT_NE(message.find("variable recovery rejected the candidate"), std::string::npos);
  }
  EXPECT_TRUE(physical_source_evaluated);
  EXPECT_EQ(sim.get_state("gas"), before);
  EXPECT_EQ(ctx->moving_interval_geometry("mesh").generation, 0u);
  EXPECT_TRUE(sim.program_exchange_records().empty());
  inadmissible = false;
  EXPECT_NO_THROW(sim.step(.3));
  EXPECT_EQ(ctx->moving_interval_geometry("mesh").generation, 1u);
  sim.rollback_step_transaction();
  EXPECT_EQ(sim.get_state("gas"), before);
  EXPECT_EQ(ctx->moving_interval_geometry("mesh").generation, 0u);
}

TEST(ProgramRuntime, MovingIntervalsUseProjectedPhysicalFluxAndSpaceTimeSourceExactlyOnce) {
  if constexpr (kNativeDimension != 1) {
    GTEST_SKIP() << "The moving-interval provider requires a native Dim1 build";
  }
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 16;
  constexpr Real growth = Real(.4), divergence = Real(.7), duration = Real(.2);
  auto config = distributed_boundary_domain_config<kNativeDimension>(n);
  config.periodicity[0] = false;  // F=divergence*x is an open-boundary physical flux.
  System<kNativeDimension> sim(config);
  install_execution_lane(sim, "pops.test.moving-interval.source-projection");
  add_gas(sim, 1.4);
  std::vector<double> values(static_cast<std::size_t>(n) * kGasComponents, 0.);
  for (int i = 0; i < n; ++i) { values[i] = 2.3; values[(kGasComponents-1)*n+i] = 6.; }
  sim.set_state("gas", values); sim.set_program_block_map({0});
  auto ctx = runtime::program::make_program_execution_provider(&sim);
  ctx->configure_primary_clock("ale-source-clock");
  ctx->initialize_moving_interval_geometry("mesh", 0, "unit-interval",Real(1e-13));
  ctx->install([&](double dt) {
    ctx->begin_step(dt);
    auto evaluation = ctx->evaluate_moving_interval("mesh", 0, "unit-interval", "endpoint-swept@1",
        [&](const runtime::multiblock::BoundaryEvaluationPoint& point) {
    const double interval_dt = point.dt;
    const auto& accepted = ctx->moving_interval_geometry("mesh");
    auto proposed = accepted.coordinates, sweep = accepted.swept_volumes;
    std::vector<nd::FaceField<kNativeDimension>> physical, density;
    auto source = ctx->scratch_state_like(ctx->state(0)); source.set_val(0);
    for (std::size_t patch = 0; patch < ctx->state(0).local_size(); ++patch) {
      const auto box = ctx->state(0).box(patch);
      physical.emplace_back(box, kGasComponents); physical.back().set_val(0);
      density.emplace_back(box, kGasComponents); density.back().set_val(0);
      const auto old = accepted.coordinates[patch].template field<0>().view();
      const auto pos = proposed[patch].template field<0>().view();
      const auto sw = sweep[patch].template field<0>().view();
      const auto flux = physical.back().template field<0>().view();
      const auto trace = density.back().template field<0>().view();
      const auto source_view = source.fab(patch).view();
      const auto geom = ctx->geometry();
      for_each_cell(nd::face_box(box, 0), [=] POPS_HD(const Index<kNativeDimension>& face) {
        const Real x = old(face);
        const Real shift = face[0] == geom.domain().lo[0] ||
                           face[0] == geom.domain().hi[0] + 1 ? Real(0) :
                           Real(.08) * Real(interval_dt) * Kokkos::sin(Real(6.2831853071795864769)*x);
        pos(face) = x + shift; sw(face) = shift;
        // Manufactured PDE U_t + div F = S, U=2.3+growth*t,
        // F=divergence*x, S=growth+divergence. These linear-in-time
        // face paths use an exact midpoint space-time quadrature.
        flux(face, 0) = divergence * Real(interval_dt) * (old(face)+pos(face))/Real(2);
        trace(face, 0) = Real(2.3) + growth * Real(interval_dt)/Real(2);
        trace(face, kGasComponents-1) = Real(6);
      });
      for_each_cell(box, [=] POPS_HD(const Index<kNativeDimension>& cell) {
        Index<kNativeDimension> right = cell; ++right[0];
        const Real average_volume = ((old(right)-old(cell))+(pos(right)-pos(cell)))/Real(2);
        source_view(cell, 0) = (growth+divergence)*Real(interval_dt)*average_volume;
      });
    }
    return runtime::program::MovingIntervalInputs<kNativeDimension>{
        std::move(proposed), std::move(sweep), std::move(physical), std::move(density), std::move(source)};
    });
    ctx->advance_moving_intervals(evaluation, Real(1e-13));
  });
  sim.set_program_block_map({0});
  sim.begin_step_transaction(); sim.step(duration);
  const auto result = sim.get_state("gas");
  std::size_t local_cells = 0;
  for (std::size_t patch = 0; patch < ctx->state(0).local_size(); ++patch)
    local_cells += static_cast<std::size_t>(ctx->state(0).box(patch).length(0));
  ASSERT_EQ(result.size(), local_cells * kGasComponents);
  for (std::size_t cell = 0; cell < local_cells; ++cell)
    EXPECT_NEAR(result[cell], 2.3+growth*duration, 2e-12);
  const auto& accepted_geometry = ctx->moving_interval_geometry("mesh");
  ASSERT_TRUE(accepted_geometry.last_receipt.has_value());
  const auto& receipt = *accepted_geometry.last_receipt;
  EXPECT_EQ(receipt.point.dt, duration);
  EXPECT_EQ(receipt.physical_frame, "unit-interval");
  double saved_balance_residual = 0, saved_gcl_residual = 0;
  for (std::size_t patch = 0; patch < ctx->state(0).local_size(); ++patch) {
    const auto current = ctx->state(0).fab(patch).create_host_mirror();
    const auto previous = receipt.previous_state.fab(patch).create_host_mirror();
    const auto v0 = receipt.previous_measures.fab(patch).create_host_mirror();
    const auto v1 = accepted_geometry.measures.fab(patch).create_host_mirror();
    const auto source = receipt.integrated_source.fab(patch).create_host_mirror();
    const auto old_faces = receipt.previous_coordinates[patch].template field<0>().create_host_mirror();
    const auto new_faces = accepted_geometry.coordinates[patch].template field<0>().create_host_mirror();
    const auto sweeps = accepted_geometry.swept_volumes[patch].template field<0>().create_host_mirror();
    const auto flux = receipt.physical_flux[patch].template field<0>().create_host_mirror();
    const auto trace = receipt.face_density[patch].template field<0>().create_host_mirror();
    ctx->state(0).fab(patch).copy_to_host(current);
    receipt.previous_state.fab(patch).copy_to_host(previous);
    receipt.previous_measures.fab(patch).copy_to_host(v0);
    accepted_geometry.measures.fab(patch).copy_to_host(v1);
    receipt.integrated_source.fab(patch).copy_to_host(source);
    receipt.previous_coordinates[patch].template field<0>().copy_to_host(old_faces);
    accepted_geometry.coordinates[patch].template field<0>().copy_to_host(new_faces);
    accepted_geometry.swept_volumes[patch].template field<0>().copy_to_host(sweeps);
    receipt.physical_flux[patch].template field<0>().copy_to_host(flux);
    receipt.face_density[patch].template field<0>().copy_to_host(trace);
    const auto box = ctx->state(0).box(patch);
    const auto cells = static_cast<std::size_t>(box.length(0));
    const auto current_view = ctx->state(0).fab(patch).view();
    const auto previous_view = receipt.previous_state.fab(patch).view();
    const auto source_view = receipt.integrated_source.fab(patch).view();
    for (std::size_t cell = 0; cell < cells; ++cell) {
      EXPECT_NEAR(v0(cell), old_faces(cell+1)-old_faces(cell), 1e-13);
      EXPECT_NEAR(v1(cell), new_faces(cell+1)-new_faces(cell), 1e-13);
      saved_gcl_residual += (v1(cell)-v0(cell))-(sweeps(cell+1)-sweeps(cell));
      for (int component = 0; component < kGasComponents; ++component) {
        const auto c = static_cast<std::size_t>(component);
        const int index = box.lo[0]+static_cast<int>(cell);
        const auto current_offset = static_cast<std::size_t>(index-current_view.origin[0]) *
            current_view.strides[0]+c*current_view.component_stride;
        const auto previous_offset = static_cast<std::size_t>(index-previous_view.origin[0]) *
            previous_view.strides[0]+c*previous_view.component_stride;
        const auto source_offset = static_cast<std::size_t>(index-source_view.origin[0]) *
            source_view.strides[0]+c*source_view.component_stride;
        const auto face0 = cell+(cells+1)*c, face1 = face0+1;
        const Real q0 = previous(previous_offset)*v0(cell);
        const Real q1 = current(current_offset)*v1(cell);
        if (component == 0)
          EXPECT_NEAR(current(current_offset), Real(2.3)+growth*duration, 2e-12);
        const Real relative0 = flux(face0)-trace(face0)*sweeps(cell);
        const Real relative1 = flux(face1)-trace(face1)*sweeps(cell+1);
        const Real balance = (q1-q0)+(relative1-relative0)-source(source_offset);
        EXPECT_NEAR(balance, 0., 2e-13);
        saved_balance_residual += balance;
      }
    }
  }
  const auto& receipt_lane = sim.prepared_boundary_execution_lane();
  EXPECT_NEAR(all_reduce_sum(saved_balance_residual, receipt_lane), 0., 4e-13);
  EXPECT_NEAR(all_reduce_sum(saved_gcl_residual, receipt_lane), 0., 4e-13);
  double relative_net = 0, source_total = 0;
  for (const auto& record : sim.program_exchange_records()) {
    if (record.operation_identity == "amount:mesh/component:0") relative_net += record.integrated_amount();
    if (record.operation_identity == "source:mesh/component:0") source_total += record.integrated_amount();
  }
  const auto& lane = sim.prepared_boundary_execution_lane();
  EXPECT_NEAR(all_reduce_sum(source_total, lane), (growth+divergence)*duration, 3e-13);
  EXPECT_NEAR(all_reduce_sum(relative_net, lane), divergence*duration, 3e-13);
  EXPECT_NEAR(all_reduce_sum(source_total-relative_net, lane), growth*duration, 3e-13);
  sim.commit_step_transaction(); sim.finalize_step_transaction();
}

#include "ale_carrier_independent_review.inc"

#include "program_dot_all_contract.inc"
