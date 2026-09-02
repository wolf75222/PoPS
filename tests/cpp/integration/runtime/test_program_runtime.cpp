// Compiled time-program runtime seam (epic ADC-399 / ADC-401 Phase 2b): a Forward-Euler Program,
// installed as a macro-step closure via pops::runtime::program::ProgramExecutionServices, runs C++-side during
// sim.step(dt). This test proves the seam end-to-end WITHOUT codegen or a .so: it builds the closure
// in C++ (the role the generated problem.so will later fill) and checks bit-parity against a reference
// Forward-Euler step computed from the SAME existing primitives (solve_fields + eval_rhs + U + dt*R).
//
// Model: a compressible Euler gas with a NON-UNIFORM pressure IC (u = v = 0), so -div F has a non-zero
// momentum component -> the step actually changes the state (parity is not vacuous). No source, no
// charge (NoEll), so the result is pure gas dynamics and deterministic across two System instances.

#include <gtest/gtest.h>

#include "native_dso_compiler.hpp"
#include "program_v5_fixture.hpp"
#include <pops/mesh/storage/multifab.hpp>
#include <pops/physics/bricks/source.hpp>                // NoSource
#include <pops/physics/composition/composite.hpp>        // CompositeModel
#include <pops/physics/fluids/euler.hpp>                 // Euler
#include <pops/runtime/builders/compiled/dsl_block.hpp>  // add_compiled_model
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/program/program_execution_services.hpp>  // ProgramExecutionServices (the seam under test)
#include <pops/runtime/program/owned_program_installation.hpp>
#include <pops/runtime/program/program_preparation_image.hpp>
#include <pops/runtime/program/program_runtime_state.hpp>
#include <pops/runtime/program/step_transaction.hpp>
#include <pops/runtime/system.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

static void install_v5_authority_program(System<kNativeDimension>& system, std::string_view mode,
                                         std::string_view identity,
                                         const std::vector<std::string>& blocks = {"tracer"},
                                         std::string_view marker = {},
                                         std::string_view release = {}) {
#if !defined(POPS_TEST_TMPDIR)
  throw std::runtime_error("ABI-v5 authority fixture requires POPS_TEST_TMPDIR");
#else
  static std::size_t fixture_index = 0;
  const std::string prefix = std::string(POPS_TEST_TMPDIR) + "/program_runtime_authority_" +
                             std::to_string(++fixture_index);
  const std::string source_path = prefix + ".cpp";
  const std::string library_path = prefix + ".so";
  {
    std::ofstream source(source_path);
    if (!source)
      throw std::runtime_error("cannot create ABI-v5 runtime authority fixture source");
    source << pops::test::program_v5::authority_program_source(mode, identity, blocks, marker,
                                                               release);
  }
  const auto compiled = pops::test::native_dso::compile_shared(source_path, library_path);
  if (!compiled.ok) {
    pops::test::native_dso::report_compile_failure("test_program_runtime", compiled);
    throw std::runtime_error("ABI-v5 runtime authority fixture compilation failed");
  }
  system.install_program(library_path);
#endif
}

// The public System installer is the only lifecycle authority.  These tests still need to
// exercise a broad set of low-level ProgramExecutionServices operations, so the v5 fixture DSO
// dispatches its ordinary step to a host-owned test callback.  The callback receives the detached
// provider created by the DSO; it is not an installation/facade shortcut and is reachable only
// after a validated pops_install_program candidate has been published.
using RuntimeProgramServices = runtime::program::ProgramExecutionServices<kNativeDimension>;
using RuntimeProgramCallback = std::function<void(RuntimeProgramServices&, double)>;

static std::vector<RuntimeProgramCallback>& runtime_program_callbacks() {
  static std::vector<RuntimeProgramCallback> callbacks;
  return callbacks;
}

static std::vector<std::uint64_t>& runtime_program_prepare_markers() {
  static std::vector<std::uint64_t> markers;
  return markers;
}

static std::uint64_t register_runtime_program_callback(RuntimeProgramCallback callback) {
  auto& callbacks = runtime_program_callbacks();
  const auto identifier = static_cast<std::uint64_t>(callbacks.size());
  callbacks.push_back(std::move(callback));
  return identifier;
}

extern "C" void pops_test_program_runtime_callback(std::uint64_t identifier, void* opaque,
                                                   double dt) {
  auto& callbacks = runtime_program_callbacks();
  if (opaque == nullptr || identifier >= callbacks.size())
    throw std::logic_error("ABI-v5 runtime callback received an invalid dispatch token");
  callbacks[static_cast<std::size_t>(identifier)](*static_cast<RuntimeProgramServices*>(opaque),
                                                  dt);
}

extern "C" void pops_test_program_runtime_prepare_marker(std::uint64_t identifier) noexcept {
  runtime_program_prepare_markers().push_back(identifier);
}

static void inject_runtime_prepare_marker(std::string& source, std::uint64_t identifier) {
  constexpr std::string_view callback_declaration =
      "extern \"C\" void pops_test_program_runtime_callback(std::uint64_t, void*, double);\n";
  const std::size_t declaration = source.find(callback_declaration);
  if (declaration == std::string::npos)
    throw std::logic_error("ABI-v5 runtime callback fixture has no callback declaration");
  source.insert(
      declaration + callback_declaration.size(),
      "extern \"C\" void pops_test_program_runtime_prepare_marker(std::uint64_t) noexcept;\n");
  constexpr std::string_view prepare_begin =
      "  try {\n    state.context = pops::runtime::program::make_program_execution_provider";
  const std::size_t prepare = source.find(prepare_begin);
  if (prepare == std::string::npos)
    throw std::logic_error("ABI-v5 runtime callback fixture has no candidate prepare body");
  source.replace(
      prepare, prepare_begin.size(),
      "  try {\n    pops_test_program_runtime_prepare_marker(" + std::to_string(identifier) +
          ");\n    state.context = pops::runtime::program::make_program_execution_provider");
}

static std::string runtime_callback_program_source(std::uint64_t callback_identifier,
                                                   std::string_view identity,
                                                   std::string_view clock,
                                                   const std::vector<std::string>& blocks) {
  return pops::test::program_v5::callback_program_source(
      callback_identifier, identity, clock, blocks,
      std::vector<pops::test::program_v5::CallbackProgramResource>{},
      "pops_test_program_runtime_callback", "uniform");
}

static std::string runtime_callback_program_source(
    std::uint64_t callback_identifier, std::string_view identity, std::string_view clock,
    const std::vector<std::string>& blocks,
    const std::vector<pops::test::program_v5::CallbackProgramResource>& resources,
    const pops::test::program_v5::CallbackProgramTransactionAuthorities& authorities = {},
    const std::vector<pops::test::program_v5::CallbackProgramHistory>& histories = {}) {
  return pops::test::program_v5::callback_program_source(
      callback_identifier, identity, clock, blocks, resources, "pops_test_program_runtime_callback",
      "uniform", {}, authorities, histories);
}

static void install_v5_callback_program(
    System<kNativeDimension>& system, std::string_view identity, std::string_view clock,
    const std::vector<std::string>& blocks,
    const std::vector<pops::test::program_v5::CallbackProgramResource>& resources,
    RuntimeProgramCallback callback,
    const pops::test::program_v5::CallbackProgramTransactionAuthorities& authorities = {},
    bool mark_candidate_prepare = false,
    const std::vector<pops::test::program_v5::CallbackProgramHistory>& histories = {}) {
#if !defined(POPS_TEST_TMPDIR)
  (void)system;
  (void)identity;
  (void)clock;
  (void)blocks;
  (void)callback;
  throw std::runtime_error("ABI-v5 runtime callback fixture requires POPS_TEST_TMPDIR");
#else
  static std::size_t fixture_index = 0;
  const std::uint64_t callback_identifier = register_runtime_program_callback(std::move(callback));
  const std::string prefix = std::string(POPS_TEST_TMPDIR) + "/program_runtime_callback_" +
                             std::to_string(++fixture_index);
  const std::string source_path = prefix + ".cpp";
  const std::string library_path = prefix + ".so";
  {
    std::ofstream source(source_path);
    if (!source)
      throw std::runtime_error("cannot create ABI-v5 runtime callback fixture source");
    auto generated = runtime_callback_program_source(callback_identifier, identity, clock, blocks,
                                                     resources, authorities, histories);
    if (mark_candidate_prepare)
      inject_runtime_prepare_marker(generated, callback_identifier);
    source << generated;
  }
  const auto compiled = pops::test::native_dso::compile_shared(source_path, library_path);
  if (!compiled.ok) {
    pops::test::native_dso::report_compile_failure("test_program_runtime callback", compiled);
    throw std::runtime_error("ABI-v5 runtime callback fixture compilation failed");
  }
  system.install_program(library_path);
#endif
}

static void install_v5_callback_program(
    System<kNativeDimension>& system, std::string_view identity, std::string_view clock,
    const std::vector<std::string>& blocks, RuntimeProgramCallback callback,
    const pops::test::program_v5::CallbackProgramTransactionAuthorities& authorities = {},
    bool mark_candidate_prepare = false,
    const std::vector<pops::test::program_v5::CallbackProgramHistory>& histories = {}) {
  install_v5_callback_program(system, identity, clock, blocks,
                              std::vector<pops::test::program_v5::CallbackProgramResource>{},
                              std::move(callback), authorities, mark_candidate_prepare, histories);
}

namespace {

struct RuntimeSyntheticCandidate final {
  int step_calls = 0;
  int dt_bound_calls = 0;
  int hierarchy_refresh_calls = 0;
  int history_remap_calls = 0;
  int restart_preflight_calls = 0;
  int restart_regrid_calls = 0;
  int restart_resync_calls = 0;
  int snapshot_create_calls = 0;
  int snapshot_destroy_calls = 0;
  int prepare_calls = 0;
  int destroy_calls = 0;
  double last_dt = 0.0;
};

class RuntimeSyntheticPreparationImage final : public runtime::program::ProgramPreparationImage {
 public:
  RuntimeSyntheticPreparationImage(std::uint32_t dimension, std::uint64_t generation,
                                   runtime::program::ProgramExecutionServicesRef services)
      : ProgramPreparationImage(dimension, runtime::program::ProgramRuntimeKind::uniform, services,
                                generation) {
    bind_image_services(services);
  }
};

void runtime_synthetic_step(void* opaque, double dt) {
  auto& candidate = *static_cast<RuntimeSyntheticCandidate*>(opaque);
  ++candidate.step_calls;
  candidate.last_dt = dt;
}

bool runtime_synthetic_prepare(void* opaque, const runtime::program::ProgramHostDescriptor*,
                               runtime::program::ProgramInstallDiagnostic*) noexcept {
  ++static_cast<RuntimeSyntheticCandidate*>(opaque)->prepare_calls;
  return true;
}

double runtime_synthetic_dt_bound(void* opaque, double cfl) {
  auto& candidate = *static_cast<RuntimeSyntheticCandidate*>(opaque);
  ++candidate.dt_bound_calls;
  return 0.5 * cfl;
}

void runtime_synthetic_destroy(void* opaque) noexcept {
  ++static_cast<RuntimeSyntheticCandidate*>(opaque)->destroy_calls;
}

void runtime_synthetic_hierarchy_refresh(void* opaque) {
  ++static_cast<RuntimeSyntheticCandidate*>(opaque)->hierarchy_refresh_calls;
}

void runtime_synthetic_history_remap(void* opaque, const void* descriptor) {
  if (descriptor == nullptr)
    throw std::invalid_argument("runtime synthetic history remap requires a descriptor");
  ++static_cast<RuntimeSyntheticCandidate*>(opaque)->history_remap_calls;
}

void runtime_synthetic_restart_preflight(void* opaque) {
  ++static_cast<RuntimeSyntheticCandidate*>(opaque)->restart_preflight_calls;
}

void runtime_synthetic_restart_regrid(void* opaque) {
  ++static_cast<RuntimeSyntheticCandidate*>(opaque)->restart_regrid_calls;
}

void runtime_synthetic_restart_resync(void* opaque) {
  ++static_cast<RuntimeSyntheticCandidate*>(opaque)->restart_resync_calls;
}

class RuntimeSyntheticAcceptedSnapshot final
    : public runtime::program::AcceptedProgramExecutionServicesSnapshot {
 public:
  explicit RuntimeSyntheticAcceptedSnapshot(RuntimeSyntheticCandidate& candidate)
      : candidate_(&candidate) {}
  ~RuntimeSyntheticAcceptedSnapshot() override { ++candidate_->snapshot_destroy_calls; }

  std::unique_ptr<runtime::program::AcceptedProgramExecutionServicesSnapshot> prepare_restore()
      const override {
    return std::make_unique<RuntimeSyntheticAcceptedSnapshot>(*candidate_);
  }
  void publish_restore() noexcept override {}

 private:
  RuntimeSyntheticCandidate* candidate_ = nullptr;
};

runtime::program::AcceptedProgramExecutionServicesSnapshot* runtime_synthetic_snapshot_create(
    void* opaque) {
  auto& candidate = *static_cast<RuntimeSyntheticCandidate*>(opaque);
  ++candidate.snapshot_create_calls;
  return new RuntimeSyntheticAcceptedSnapshot(candidate);
}

template <int Dim>
runtime::program::ProgramCandidateDescriptor runtime_synthetic_descriptor(
    RuntimeSyntheticCandidate& candidate, bool with_dt_bound, bool with_lifecycle,
    bool with_block_table = false) {
  using namespace runtime::program;
  static constexpr char metadata[] = "synthetic-runtime";
  static constexpr char block_name[] = "synthetic";
  static constexpr ProgramBlockRecord blocks[] = {{{block_name, sizeof(block_name) - 1}}};
  ProgramCandidateDescriptor descriptor{};
  descriptor.struct_size = sizeof(ProgramCandidateDescriptor);
  descriptor.abi_version = kProgramInstallAbiVersion;
  descriptor.native_dimension = Dim;
  descriptor.runtime_kind = with_lifecycle ? ProgramRuntimeKind::amr : ProgramRuntimeKind::uniform;
  descriptor.provided_capability_bits = kKnownProgramCapabilityBits;
  descriptor.program_name = {metadata, sizeof(metadata) - 1};
  descriptor.artifact_identity = {metadata, sizeof(metadata) - 1};
  descriptor.abi_key = {metadata, sizeof(metadata) - 1};
  descriptor.route_manifest = {metadata, sizeof(metadata) - 1};
  descriptor.boundary_manifest = {metadata, sizeof(metadata) - 1};
  descriptor.persistent_resource_manifest = {metadata, sizeof(metadata) - 1};
  descriptor.checkpoint_identity = {metadata, sizeof(metadata) - 1};
  if (with_block_table)
    descriptor.blocks = {blocks, 1, sizeof(ProgramBlockRecord)};
  descriptor.maximum_bytes = 0;
  descriptor.context = &candidate;
  descriptor.prepare = &runtime_synthetic_prepare;
  descriptor.step = &runtime_synthetic_step;
  descriptor.dt_bound = with_dt_bound ? &runtime_synthetic_dt_bound : nullptr;
  descriptor.destroy = &runtime_synthetic_destroy;
  if (with_lifecycle) {
    descriptor.hierarchy_refresh = &runtime_synthetic_hierarchy_refresh;
    descriptor.history_remap_accepted = &runtime_synthetic_history_remap;
    descriptor.restart_regrid_preflight = &runtime_synthetic_restart_preflight;
    descriptor.restart_regrid = &runtime_synthetic_restart_regrid;
    descriptor.restart_resync = &runtime_synthetic_restart_resync;
    descriptor.create_accepted_snapshot = &runtime_synthetic_snapshot_create;
  }
  return descriptor;
}

template <int Dim>
runtime::program::PreparedProgramInstallation runtime_prepared_artifact(
    RuntimeSyntheticCandidate& candidate, std::uint64_t generation, bool with_dt_bound = true,
    bool with_lifecycle = false, bool with_block_table = false) {
  using namespace runtime::program;
  ProgramExecutionServicesRef services{&candidate, &candidate, &candidate, &candidate, &candidate,
                                       &candidate, &candidate, &candidate, &candidate};
  ProgramHostDescriptor host{};
  host.native_dimension = Dim;
  host.runtime_kind = ProgramRuntimeKind::uniform;
  host.capability_bits = kKnownProgramCapabilityBits;
  host.services = services;
  auto image = std::make_shared<RuntimeSyntheticPreparationImage>(Dim, generation, services);
  bind_program_preparation_image(host, image);
  static const std::string resource_manifest = [] {
    ProgramInstallationTables tables;
    const std::string payload = tables.canonical_resource_digest_payload(0);
    const std::string digest =
        pops::identity::sha256_hex(std::vector<std::uint8_t>(payload.begin(), payload.end()));
    return "{\"resource_plan\":" + tables.canonical_resource_manifest(0, digest) +
           ",\"resource_plan_digest\":\"" + digest + "\"}";
  }();
  auto descriptor =
      runtime_synthetic_descriptor<Dim>(candidate, with_dt_bound, with_lifecycle, with_block_table);
  descriptor.persistent_resource_manifest = {resource_manifest.data(), resource_manifest.size()};
  OwnedProgramInstallation owner(
      pops::dynlib::UniqueHandle{nullptr}, descriptor,
      ProgramInstallationMetadata{"synthetic-runtime", "abi", "route", "boundary",
                                  resource_manifest, "checkpoint"});
  owner.set_preparation_image(image);
  owner.prepare(host);
  PreparedProgramInstallation artifact(std::move(owner));
  artifact.seal_resource_plan(std::span<const ProgramInstallationTables::ResourcePrototype>{});
  return artifact;
}

}  // namespace

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
  state.bind_transaction_authorities();

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
  state.declare_balance_route(route);
  state.declare_automatic_balance_term(2, 0, 1, "projection");
  state.declare_automatic_balance_term(2, 1, 1, "projection");
  state.declare_automatic_balance_term(2, 0, 1, "reflux");
  state.bind_transaction_authorities();
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

  RuntimeSyntheticCandidate replacement;
  state.install_prepared_artifact(runtime_prepared_artifact<kNativeDimension>(
      replacement, state.step_install_generation_ + 1, false));
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

TEST(ProgramRuntime, PreparedArtifactPublicationRejectsStaleGenerationAndPublishesExactlyOnce) {
  runtime::program::ProgramRuntimeState<kNativeDimension> state;
  RuntimeSyntheticCandidate old_candidate;
  RuntimeSyntheticCandidate new_candidate;
  state.install_prepared_artifact(runtime_prepared_artifact<kNativeDimension>(
      old_candidate, state.step_install_generation_ + 1, true, true));
  state.operator_authorities_ = {{{1, 2, 3, 4}}};
  state.history_replay_authorities_ = {{"gas.previous", 3}};
  state.installed_hash_ = "accepted-artifact";
  state.block_map_ = {2};
  state.seed_params(0, {3.0});
  state.dt_bound_ = [](Real cfl) { return Real(0.5) * cfl; };
  state.artifact_backed_ = true;
  const auto accepted_generation = state.step_install_generation_;

  // A stale prepared image is rejected before publication; the accepted owner and all of its
  // authority remain untouched.  This is the v5 generation contract, not an install-step snapshot.
  RuntimeSyntheticCandidate stale_candidate;
  auto stale =
      runtime_prepared_artifact<kNativeDimension>(stale_candidate, accepted_generation, true, true);
  EXPECT_THROW((void)runtime::program::ProgramRuntimeState<
                   kNativeDimension>::PreparedArtifactPublication::prepare(std::move(stale),
                                                                           accepted_generation + 1),
               std::invalid_argument);
  EXPECT_EQ(state.step_install_generation_, accepted_generation);
  state.step_(0.1);
  EXPECT_EQ(old_candidate.step_calls, 1);
  EXPECT_EQ(state.operator_authorities_,
            (std::vector<std::array<std::uint64_t, 4>>{{{1, 2, 3, 4}}}));
  EXPECT_EQ(state.installed_hash_, "accepted-artifact");
  EXPECT_EQ(state.block_map_, (std::vector<int>{2}));
  EXPECT_EQ(state.block_params_.size(), 1u);
  ASSERT_TRUE(state.dt_bound_);
  EXPECT_DOUBLE_EQ(state.dt_bound_(0.4), 0.2);
  EXPECT_TRUE(state.authorizes_history_replay("gas.previous", 3));
  EXPECT_NO_THROW(state.preflight_regrid_on_restart("test"));
  EXPECT_NO_THROW(state.regrid_on_restart("test"));
  EXPECT_NO_THROW(state.resync_after_restart("test"));
  EXPECT_EQ(old_candidate.restart_preflight_calls, 1);
  EXPECT_EQ(old_candidate.restart_regrid_calls, 1);
  EXPECT_EQ(old_candidate.restart_resync_calls, 1);

  // Publication consumes one fully prepared artifact exactly once.  It is the sole API that can
  // replace the owner; the old DSO-backed closures are released only after the exchange returns.
  auto replacement = runtime_prepared_artifact<kNativeDimension>(
      new_candidate, state.step_install_generation_ + 1, true, true);
  auto publication =
      runtime::program::ProgramRuntimeState<kNativeDimension>::PreparedArtifactPublication::prepare(
          std::move(replacement), state.step_install_generation_ + 1);
  state.publish_prepared_artifact(std::move(publication));
  EXPECT_EQ(state.step_install_generation_, accepted_generation + 1);
  state.step_(0.2);
  EXPECT_EQ(old_candidate.step_calls, 1);
  EXPECT_EQ(new_candidate.step_calls, 1);
  EXPECT_EQ(new_candidate.last_dt, 0.2);
}

TEST(ProgramRuntime, PreparedArtifactPublicationAuthenticatesStateFreeFlagAgainstTables) {
  using State = runtime::program::ProgramRuntimeState<kNativeDimension>;
  const runtime::program::ProgramCheckpointMetadata empty_checkpoint{};

  RuntimeSyntheticCandidate state_free_candidate;
  auto state_free = State::PreparedArtifactPublication::prepare(
      runtime_prepared_artifact<kNativeDimension>(state_free_candidate, 1), 1);
  EXPECT_NO_THROW(
      state_free.set_resolved_authority("state-free", {}, {}, empty_checkpoint, {}, {}, true));

  RuntimeSyntheticCandidate empty_table_candidate;
  auto empty_table = State::PreparedArtifactPublication::prepare(
      runtime_prepared_artifact<kNativeDimension>(empty_table_candidate, 1), 1);
  EXPECT_THROW(
      empty_table.set_resolved_authority("empty-table", {}, {}, empty_checkpoint, {}, {}, false),
      std::invalid_argument);

  RuntimeSyntheticCandidate block_table_candidate;
  auto block_table = State::PreparedArtifactPublication::prepare(
      runtime_prepared_artifact<kNativeDimension>(block_table_candidate, 1, true, false, true), 1);
  EXPECT_THROW(
      block_table.set_resolved_authority("block-table", {}, {}, empty_checkpoint, {0}, {}, true),
      std::invalid_argument);
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
  install_execution_lane(subcycled, "pops.test.program-runtime.cadence-subcycled");
  subcycled.set_clock(1.0, 10);
  std::vector<double> subcycled_times;
  std::vector<int> subcycled_macro_steps;
  install_v5_callback_program(
      subcycled, "test.program-runtime.cadence-subcycled.v5", "macro", {},
      [&](RuntimeProgramServices& context, double) {
        subcycled_times.push_back(static_cast<double>(context.physical_time()));
        subcycled_macro_steps.push_back(context.macro_step());
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
  install_execution_lane(catchup, "pops.test.program-runtime.cadence-catchup");
  add_scalar(catchup);
  catchup.set_program_block_map({0});
  std::vector<double> catchup_times;
  std::vector<double> catchup_steps;
  std::vector<int> catchup_macro_steps;
  std::vector<bool> catchup_every_one_due;
  install_v5_callback_program(
      catchup, "test.program-runtime.cadence-catchup.v5", "macro", {"tracer"},
      {{pops::test::program_v5::CallbackProgramResource::Kind::cache, 0, 0, 0, -1, 1, 0}},
      [&](RuntimeProgramServices& context, double h) {
        catchup_times.push_back(static_cast<double>(context.physical_time()));
        catchup_steps.push_back(h);
        catchup_macro_steps.push_back(context.macro_step());
        catchup_every_one_due.push_back(context.schedule_is_due(
            0, 1, runtime::program::ScheduleDomainKind::kAcceptedStep, "macro", "", -1));
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

TEST(ProgramRuntime, StateFreeV5ProgramRunsCadenceAndRejectsBlockAccess) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  System<kNativeDimension> system(unit_domain_config<kNativeDimension>(4));
  install_execution_lane(system, "pops.test.program-runtime.state-free");

  std::vector<int> observed_block_counts;
  std::vector<double> observed_times;
  std::vector<bool> rejected_block_access;
  install_v5_callback_program(
      system, "test.program-runtime.state-free.v5", "macro", {},
      [&](RuntimeProgramServices& context, double) {
        observed_block_counts.push_back(context.n_blocks());
        observed_times.push_back(static_cast<double>(context.physical_time()));
        try {
          (void)context.sys_block(0);
          rejected_block_access.push_back(false);
        } catch (const std::runtime_error&) {
          rejected_block_access.push_back(true);
        }
        context.record_scalar("state-free.cadence", static_cast<Real>(context.n_blocks()));
      },
      {.diagnostics = {"state-free.cadence"}});

  system.set_program_cadence(/*substeps=*/2, /*stride=*/1);
  system.step(0.2);

  EXPECT_EQ(observed_block_counts, (std::vector<int>{0, 0}));
  EXPECT_EQ(rejected_block_access, (std::vector<bool>{true, true}));
  ASSERT_EQ(observed_times.size(), 2u);
  EXPECT_NEAR(observed_times[0], 0.0, 1.0e-14);
  EXPECT_NEAR(observed_times[1], 0.1, 1.0e-14);
  EXPECT_DOUBLE_EQ(system.program_diagnostic("state-free.cadence"), 0.0);
  EXPECT_NEAR(system.time(), 0.2, 1.0e-14);
  EXPECT_EQ(system.macro_step(), 1);
}

TEST(ProgramRuntime, StateFreeV5ProgramIsRefusedBeforePrepareOnNonemptySystem) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  System<kNativeDimension> system(unit_domain_config<kNativeDimension>(4));
  install_execution_lane(system, "pops.test.program-runtime.state-free-refusal");
  add_gas(system, 1.4);
  runtime_program_prepare_markers().clear();

  try {
    install_v5_callback_program(
        system, "test.program-runtime.state-free-refusal.v5", "macro", {},
        std::vector<pops::test::program_v5::CallbackProgramResource>{},
        [](RuntimeProgramServices&, double) { ADD_FAILURE() << "state-free callback ran"; }, {},
        true);
    ADD_FAILURE() << "state-free Program installed on a System with accepted blocks";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("state-free Program"), std::string::npos);
    EXPECT_NE(std::string(error.what()).find("accepted blocks"), std::string::npos);
  }
  EXPECT_TRUE(runtime_program_prepare_markers().empty())
      << "state-free refusal must precede the DSO candidate_prepare callback";
}

TEST(ProgramRuntime, StateFreePreparationBindsOnlyExactGlobalResources) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  System<kNativeDimension> system(unit_domain_config<kNativeDimension>(4));
  install_execution_lane(system, "pops.test.program-runtime.state-free-resource-plan");

  runtime::program::ProgramInstallationTables::ResourcePlan global;
  global.slot = 0;
  global.value_id = 0x51;
  global.occurrence_path_id = 0x52;
  global.level = -1;
  global.components = 1;
  global.ghosts = 0;
  global.bytes = sizeof(Real);
  global.maximum_bytes = sizeof(Real);
  global.cells = 1;
  global.itemsize = sizeof(Real);
  global.resource_type = runtime::program::ProgramResourcePlanType::exact;
  global.schema = "program-resource-plan:v1";
  global.plan_digest = std::string(64, 'a');
  global.identity = "test.program-runtime/state-free-global";
  global.occurrence_path = global.identity;
  global.owner = "global";
  global.space = "global";
  global.clock = "macro";
  global.lifetime = "persistent";
  global.centering = "none";
  global.off_policy = "hold";
  global.communication = "none";
  global.transfer_provider = "none";
  global.restart_provider = "none";
  global.component_names = "[\"value\"]";
  global.shape = "[]";

  const auto make_image = [&] {
    return runtime::program::make_program_preparation_image<kNativeDimension>(&system, 1);
  };
  EXPECT_NO_THROW(
      runtime::program::bind_staged_uniform_program_resource_declaration<kNativeDimension>(
          make_image(), {global}, {}));

  auto runtime_sized = global;
  runtime_sized.flags = runtime::program::kProgramResourceRuntimeSized;
  runtime_sized.resource_type = runtime::program::ProgramResourcePlanType::runtime_sized;
  runtime_sized.bytes.reset();
  runtime_sized.maximum_bytes.reset();
  runtime_sized.cells.reset();
  runtime_sized.itemsize.reset();
  EXPECT_THROW(runtime::program::bind_staged_uniform_program_resource_declaration<kNativeDimension>(
                   make_image(), {runtime_sized}, {}),
               std::invalid_argument);

  auto block_owned = global;
  block_owned.owner = "gas";
  EXPECT_THROW(runtime::program::bind_staged_uniform_program_resource_declaration<kNativeDimension>(
                   make_image(), {block_owned}, {}),
               std::invalid_argument);
}

TEST(ProgramRuntime, StrideHeldStepsPublishTheExactZeroBalance) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  auto config = unit_domain_config<kNativeDimension>(4);

  System<kNativeDimension> system(config);
  install_execution_lane(system, "pops.test.program-runtime.balance-ledger");
  const std::string route = "pops.balance-ledger-route.v1:sha256:" + std::string(64, '7');
  const std::array<std::pair<const char*, double>, 5> records{{
      {"storage_change", 1.0},
      {"outward_boundary_flux", 2.0},
      {"sources", 3.0},
      {"reflux", 4.0},
      {"projection", 5.0},
  }};
  install_v5_callback_program(system, "test.program-runtime.balance-ledger.v5", "macro", {},
                              [&](RuntimeProgramServices& context, double) {
                                for (const auto& [name, value] : records)
                                  context.record_balance_term(route, name, value);
                              },
                              {.balance_routes = {route}});
  system.set_program_cadence(/*substeps=*/1, /*stride=*/3);

  const auto step_and_read = [&]() {
    system.begin_step_transaction();
    system.step(0.1);
    std::map<std::string, double> balance;
    {
      auto scope = system._provisional_read_scope();
      balance = system.accepted_balance_terms(route);
    }
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

  system.begin_step_transaction();
  system.step(0.1);
  std::map<std::string, double> rejected_due;
  {
    auto scope = system._provisional_read_scope();
    rejected_due = system.accepted_balance_terms(route);
  }
  for (const auto& [name, value] : records)
    EXPECT_DOUBLE_EQ(rejected_due.at(name), value);
  system.rollback_step_transaction();
  system.begin_step_transaction();
  std::map<std::string, double> restored_held;
  {
    auto scope = system._provisional_read_scope();
    restored_held = system.accepted_balance_terms(route);
  }
  for (const auto& [name, _value] : records)
    EXPECT_DOUBLE_EQ(restored_held.at(name), 0.0);
  system.rollback_step_transaction();

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
  install_execution_lane(system, "pops.test.program-runtime.cadence-nonassociative");
  system.set_clock(0.1, 0);
  std::vector<double> starts;
  std::vector<double> steps;
  install_v5_callback_program(system, "test.program-runtime.cadence-nonassociative.v5", "macro", {},
                              [&](RuntimeProgramServices& context, double h) {
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
  install_execution_lane(system, "pops.test.program-runtime.cadence-retry");
  add_scalar(system);
  install_v5_authority_program(system, "reject_first_dt", "test.program-runtime.cadence-retry.v5");
  system.set_program_cadence(/*substeps=*/1, /*stride=*/2);
  system.step(0.1);

  EXPECT_THROW(system.step(0.2), runtime::program::StepAttemptRejected);
  EXPECT_DOUBLE_EQ(system.time(), 0.1);
  EXPECT_EQ(system.macro_step(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.1);
  EXPECT_EQ(system.program_cadence_window_steps(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_start_time(), 0.0);

  system.step(0.2);
  EXPECT_NEAR(system.program_diagnostic("test.program.v5.authority.last_dt"), 0.3, 1.0e-14);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.0);
  EXPECT_EQ(system.program_cadence_window_steps(), 0);

  System<kNativeDimension> restarted(config);
  install_execution_lane(restarted, "pops.test.program-runtime.cadence-restarted");
  std::vector<double> restarted_times;
  install_v5_callback_program(
      restarted, "test.program-runtime.cadence-restarted.v5", "macro", {},
      [&](RuntimeProgramServices& context, double) {
        restarted_times.push_back(static_cast<double>(context.physical_time()));
      });
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
  install_execution_lane(system, "pops.test.program-runtime.cadence-restore");
  add_scalar(system);
  install_v5_authority_program(system, "noop", "test.program-runtime.cadence-restore.v5");
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
  install_execution_lane(system, "pops.test.program-runtime.cadence-overflow");
  add_scalar(system);
  install_v5_authority_program(system, "noop", "test.program-runtime.cadence-overflow.v5");
  system.set_clock(1.0e16, 0);

  EXPECT_THROW(system.step(0.5), std::overflow_error);
  EXPECT_DOUBLE_EQ(system.time(), 1.0e16);
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_TRUE(system.program_diagnostics().empty());
}

TEST(ProgramRuntime, CadenceFailsBeforeMutationWhenSubstepsCollapseTheRepresentableInterval) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  auto config = unit_domain_config<kNativeDimension>(4);

  System<kNativeDimension> system(config);
  install_execution_lane(system, "pops.test.program-runtime.cadence-collapse");
  add_scalar(system);
  install_v5_authority_program(system, "noop", "test.program-runtime.cadence-collapse.v5");
  system.set_program_cadence(/*substeps=*/3, /*stride=*/1);
  system.set_clock(1.0, 0);
  const double one_ulp = std::nextafter(1.0, 2.0) - 1.0;

  EXPECT_THROW(system.step(one_ulp), std::overflow_error);
  EXPECT_DOUBLE_EQ(system.time(), 1.0);
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_TRUE(system.program_diagnostics().empty());
}

TEST(ProgramRuntime, ForwardEulerProgramExecutionServicesMatchesEvalRhsReferenceAndCountsKernels) {
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

  // Program: the SAME step expressed as a ProgramExecutionServices closure and driven by sim.step(dt).
  System<kNativeDimension> sim(cfg);
  install_execution_lane(sim, "pops.test.program-runtime.forward-euler");
  add_gas(sim, gamma);
  sim.set_state("gas", U0);
  sim.set_program_block_map({0});

  install_v5_callback_program(sim, "test.program-runtime.forward-euler.v5", "macro", {"gas"},
                              [](RuntimeProgramServices& ctx, double h) {
                                ctx.begin_step(h);
                                ctx.set_stage_time(0, 1);
                                auto field_outcome = ctx.solve_fields();
                                (void)field_outcome.consume(SolveConsumption::kAccept);
                                for (int b = 0; b < ctx.n_blocks(); ++b) {
                                  MultiFab<kNativeDimension>& U = ctx.state(b);
                                  MultiFab<kNativeDimension> R = ctx.rhs_scratch_like(U);
                                  ctx.rhs_into(b, U, R, 0);
                                  ctx.axpy(U, Real(h), R);  // U <- U + h * R  (Forward Euler)
                                }
                              });
  sim.mark_bound();

  // Profiling counters (ADC-459, Spec 3 section 29): the ProgramExecutionServices owns the two explicit
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
  const auto prof = sim.profiler();
  EXPECT_TRUE(prof.counter("kernels") == 2)
      << "kernels counter = " << static_cast<long long>(prof.counter("kernels"))
      << ", expected 2 (rhs_into + axpy; solve backend owns its diagnostics)";
  EXPECT_EQ(prof.counter("scratch_allocs"), 0);
  EXPECT_EQ(prof.counter("scratch_peak_bytes"), 0);
  // The cache hit/skip counters never fire on this native ProgramExecutionServices step (no held schedule); they
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

TEST(ProgramRuntime, ForwardEulerProgramExecutionServicesHonorsEmbeddedBoundaryResidualMetrics) {
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
    install_v5_callback_program(system, "test.program-runtime.embedded-forward-euler.v5", "macro",
                                {"gas"}, [](RuntimeProgramServices& context, double step) {
                                  context.begin_step(step);
                                  context.set_stage_time(0, 1);
                                  MultiFab<kNativeDimension>& state = context.state(0);
                                  MultiFab<kNativeDimension> residual =
                                      context.rhs_scratch_like(state);
                                  context.rhs_into(0, state, residual, 0);
                                  context.axpy(state, Real(step), residual);
                                });
    system.mark_bound();
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
    install_v5_callback_program(system, "test.program-runtime.embedded-source-step.v5", "macro",
                                {"gas"}, [](RuntimeProgramServices& context, double step) {
                                  context.begin_step(step);
                                  MultiFab<kNativeDimension>& state = context.state(0);
                                  MultiFab<kNativeDimension> source =
                                      context.rhs_scratch_like(state);
                                  context.source_default_into(0, state, source);
                                  context.axpy(state, Real(step), source);
                                });
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
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const std::vector<Resource> resources{{Resource::Kind::rhs, 0, 0, 0, -1, kGasComponents, 1},
                                        {Resource::Kind::state, 1, 0, 0, -1, kGasComponents, 1}};
  install_v5_callback_program(
      system, "test.program-runtime.source-recovery.v5", "test.clock.source-recovery", {"gas"},
      resources, [](RuntimeProgramServices& context, double step) {
        context.begin_step(step);
        MultiFab<kNativeDimension>& live = context.state(0);
        MultiFab<kNativeDimension>& source = context.rhs_scratch(0, 0, live);
        MultiFab<kNativeDimension>& candidate = context.scratch_state(1, 0, live);
        context.source_default_into(0, live, source);
        context.lincomb(candidate, Real(1), live, Real(0), live);
        context.axpy(candidate, Real(step), source);
        context.commit_many({{&live, &candidate}});
      });

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
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const std::vector<Resource> resources{{Resource::Kind::rhs, 0, 0, 0, -1, kGasComponents, 1},
                                        {Resource::Kind::rhs, 1, 0, 1, -1, kGasComponents, 1},
                                        {Resource::Kind::state, 2, 0, 0, -1, kGasComponents, 1},
                                        {Resource::Kind::state, 3, 0, 1, -1, kGasComponents, 1}};
  install_v5_callback_program(
      system, "test.program-runtime.source-recovery-multiblock.v5",
      "test.clock.source-recovery-multiblock", {"first", "second"}, resources,
      [](RuntimeProgramServices& context, double step) {
        context.begin_step(step);
        MultiFab<kNativeDimension>& first = context.state(0);
        MultiFab<kNativeDimension>& second = context.state(1);
        MultiFab<kNativeDimension>& first_source = context.rhs_scratch(0, 0, first);
        MultiFab<kNativeDimension>& second_source = context.rhs_scratch(1, 0, second);
        MultiFab<kNativeDimension>& first_candidate = context.scratch_state(2, 0, first);
        MultiFab<kNativeDimension>& second_candidate = context.scratch_state(3, 0, second);
        context.source_default_into(0, first, first_source);
        context.source_default_into(1, second, second_source);
        context.lincomb(first_candidate, Real(1), first, Real(0), first);
        context.lincomb(second_candidate, Real(1), second, Real(0), second);
        context.axpy(first_candidate, Real(step), first_source);
        context.axpy(second_candidate, Real(2) * Real(step), second_source);
        context.commit_many({{&first, &first_candidate}, {&second, &second_candidate}});
      });

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
  install_v5_callback_program(system, "test.program-runtime.inactive-source.v5", "macro", {"gas"},
                              [](RuntimeProgramServices& context, double step) {
                                context.begin_step(step);
                                MultiFab<kNativeDimension>& state = context.state(0);
                                MultiFab<kNativeDimension> source = context.rhs_scratch_like(state);
                                context.source_default_into(0, state, source);
                                context.axpy(state, Real(step), source);
                              });
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

  double embedded_speed = 0.0;
  {
    const MultiFab<kNativeDimension> state = [&] {
      const auto state_view = system.block_state(0);
      return MultiFab<kNativeDimension>(*state_view.get());
    }();
    embedded_speed = system.block_max_speed(0, state);
  }
  system.set_geometry_mode("none");
  double cartesian_speed = 0.0;
  {
    const MultiFab<kNativeDimension> state = [&] {
      const auto state_view = system.block_state(0);
      return MultiFab<kNativeDimension>(*state_view.get());
    }();
    cartesian_speed = system.block_max_speed(0, state);
  }
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
  install_v5_authority_program(system, "noop", "test.program-runtime.diffusive-cfl.v5", {"gas"});

  const double transport_frequency = std::sqrt(gamma) * n;
  const double parabolic_frequency =
      2.0 * diffusivity * static_cast<double>(kNativeDimension) * n * n;
  system.mark_bound();
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
  double staircase_speed = 0.0;
  {
    const MultiFab<kNativeDimension> state = [&] {
      const auto state_view = staircase.block_state(0);
      return MultiFab<kNativeDimension>(*state_view.get());
    }();
    staircase_speed = staircase.block_max_speed(0, state);
  }

  System<kNativeDimension> cutcell(cfg);
  install_execution_lane(cutcell, "pops.test.program-runtime.fractional-speed.cutcell");
  add_gas(cutcell, gamma, "none");
  cutcell.set_state("gas", uniform);
  install_centered_ball(cutcell, 0.34, "cutcell", 0.1);
  double cutcell_speed = 0.0;
  {
    const MultiFab<kNativeDimension> state = [&] {
      const auto state_view = cutcell.block_state(0);
      return MultiFab<kNativeDimension>(*state_view.get());
    }();
    cutcell_speed = cutcell.block_max_speed(0, state);
  }
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
  const int staircase_inactive = static_cast<int>(cells) - staircase_active;
  const Real staircase_raw_sum =
      Real(2) * Real(staircase_active) + Real(1000) * Real(staircase_inactive);
  const Real staircase_raw_dot =
      Real(4) * Real(staircase_active) + Real(1000000) * Real(staircase_inactive);
  const Real staircase_active_sum = Real(2 * staircase_active);
  const Real staircase_active_dot = Real(4 * staircase_active);
  const std::vector<pops::test::program_v5::CallbackProgramResource> reduction_resources{
      {pops::test::program_v5::CallbackProgramResource::Kind::scalar, 0, 0, 0, -1, 1, 0}};
  install_v5_callback_program(
      staircase, "pops.test.program-runtime.staircase-reductions.v5", "macro", {"gas"},
      reduction_resources, [&](RuntimeProgramServices& context, double step) {
        context.begin_step(step);
        MultiFab<kNativeDimension>& staircase_field = context.state(0);
        // Ownerless ProgramExecutionServices overloads remain raw scratch-field algebra, including
        // inactive sentinels. Owner-qualified overloads exclude inactive cells; System alone applies
        // kappa.
        EXPECT_EQ(context.sum_component(staircase_field, 0), staircase_raw_sum);
        EXPECT_EQ(context.abs_sum_component(staircase_field, 0), staircase_raw_sum);
        EXPECT_EQ(pops::reduce_abs_sum(staircase_field, 0), staircase_raw_sum);
        EXPECT_EQ(pops::dot(staircase_field, staircase_field, 0), staircase_raw_dot);
        EXPECT_EQ(context.max_component(staircase_field, 0), Real(1000));
        EXPECT_EQ(context.min_component(staircase_field, 1), Real(-1000));
        EXPECT_EQ(context.sum_component(0, staircase_field, 0), staircase_active_sum);
        EXPECT_EQ(context.abs_sum_component(0, staircase_field, 0), staircase_active_sum);
        EXPECT_EQ(context.dot(0, staircase_field, staircase_field), staircase_active_dot);
        EXPECT_EQ(context.max_component(0, staircase_field, 0), Real(2));
        EXPECT_EQ(context.min_component(0, staircase_field, 1), Real(3));
        EXPECT_EQ(context.norm2(0, staircase_field), std::sqrt(staircase_active_dot));
        EXPECT_EQ(context.norm_inf(0, staircase_field), Real(2));
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
            [&] { return context.sum_component(0, staircase_field, invalid_component); },
            "pops::reduce_active_sum_local");
        expect_original_component_error(
            [&] { return context.abs_sum_component(0, staircase_field, invalid_component); },
            "pops::reduce_active_abs_sum_local");
        expect_original_component_error(
            [&] { return context.max_component(0, staircase_field, invalid_component); },
            "pops::reduce_active_max_local");
        expect_original_component_error(
            [&] { return context.min_component(0, staircase_field, invalid_component); },
            "pops::reduce_active_min_local");
        MultiFab<kNativeDimension>& staircase_status =
            context.scalar_scratch(0, 0, staircase_field, 1, 0);
        try {
          (void)context.dot(0, staircase_field, staircase_status);
          FAIL() << "owner-qualified dot must rethrow the original layout error in serial";
        } catch (const std::invalid_argument& error) {
          EXPECT_NE(std::string(error.what()).find("ProgramExecutionServices dot"),
                    std::string::npos);
        }
      });
  staircase.step(1.0e-3);
  EXPECT_EQ(staircase.mass("gas"), static_cast<double>(staircase_active_sum));
  EXPECT_EQ(staircase.reduce_component("gas", "sum", 0), static_cast<double>(staircase_active_sum));
  EXPECT_EQ(staircase.reduce_component("gas", "sum_sq", 0),
            static_cast<double>(Real(2) * staircase_active_sum));
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
  const int cutcell_inactive = static_cast<int>(cells) - cutcell_active;
  const Real cutcell_raw_sum = Real(2) * Real(cutcell_active) + Real(1000) * Real(cutcell_inactive);
  const Real cutcell_raw_dot =
      Real(4) * Real(cutcell_active) + Real(1000000) * Real(cutcell_inactive);
  const Real cutcell_active_sum = Real(2 * cutcell_active);
  const Real cutcell_active_dot = Real(4 * cutcell_active);
  const double cutcell_physical_sum = 2.0 * cutcell_kappa_sum;
  install_v5_callback_program(
      cutcell, "pops.test.program-runtime.cutcell-reductions.v5", "macro", {"gas"},
      reduction_resources, [&](RuntimeProgramServices& context, double step) {
        context.begin_step(step);
        MultiFab<kNativeDimension>& cutcell_field = context.state(0);
        EXPECT_EQ(context.sum_component(cutcell_field, 0), cutcell_raw_sum);
        EXPECT_EQ(context.abs_sum_component(cutcell_field, 0), cutcell_raw_sum);
        EXPECT_EQ(pops::reduce_abs_sum(cutcell_field, 0), cutcell_raw_sum);
        EXPECT_EQ(pops::dot(cutcell_field, cutcell_field, 0), cutcell_raw_dot);
        EXPECT_EQ(context.max_component(cutcell_field, 0), Real(1000));
        EXPECT_EQ(context.min_component(cutcell_field, 1), Real(-1000));
        EXPECT_EQ(context.sum_component(0, cutcell_field, 0), cutcell_active_sum);
        EXPECT_EQ(context.abs_sum_component(0, cutcell_field, 0), cutcell_active_sum);
        EXPECT_EQ(context.dot(0, cutcell_field, cutcell_field), cutcell_active_dot);
        EXPECT_EQ(context.max_component(0, cutcell_field, 0), Real(2));
        EXPECT_EQ(context.min_component(0, cutcell_field, 1), Real(3));
        EXPECT_EQ(context.norm2(0, cutcell_field), std::sqrt(cutcell_active_dot));
        EXPECT_EQ(context.norm_inf(0, cutcell_field), Real(2));
      });
  cutcell.step(1.0e-3);
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
    const std::vector<pops::test::program_v5::CallbackProgramResource> status_resources{
        {pops::test::program_v5::CallbackProgramResource::Kind::scalar, 0, 0, 0, -1, 1, 0}};
    install_v5_callback_program(
        system, "pops.test.program-runtime.pointwise-status." + mode + ".v5", "macro", {"gas"},
        status_resources, [&, mode](RuntimeProgramServices& context, double step) {
          context.begin_step(step);
          MultiFab<kNativeDimension>& state = context.state(0);
          const MultiFab<kNativeDimension>* const active = context.pointwise_active_mask(0, state);
          ASSERT_NE(active, nullptr) << mode;
          EXPECT_EQ(active, context.pointwise_active_mask(0, state)) << mode;

          MultiFab<kNativeDimension>& status = context.scalar_scratch(0, 0, state, 1, 0);
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
          EXPECT_EQ(
              context.pointwise_status_max(0, status, active, context.prepared_execution_lane()),
              Real{0})
              << mode;
          write_status(Real{0}, std::numeric_limits<Real>::quiet_NaN());
          EXPECT_EQ(
              context.pointwise_status_max(0, status, active, context.prepared_execution_lane()),
              Real{0})
              << mode;
          write_status(Real{2}, Real{0});
          EXPECT_EQ(
              context.pointwise_status_max(0, status, active, context.prepared_execution_lane()),
              Real{2})
              << mode;
          write_status(std::numeric_limits<Real>::quiet_NaN(), Real{0});
          EXPECT_EQ(
              context.pointwise_status_max(0, status, active, context.prepared_execution_lane()),
              Real{3})
              << mode;
          EXPECT_THROW(
              context.pointwise_status_max(0, status, &status, context.prepared_execution_lane()),
              std::invalid_argument)
              << mode;
        });
    system.step(1.0e-3);
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
  install_v5_callback_program(
      program, "test.program-runtime.ssprk3.v5", "macro", {"gas"},
      [](RuntimeProgramServices& context, double step) {
        context.begin_step(step);
        MultiFab<kNativeDimension>& state = context.state(0);
        MultiFab<kNativeDimension> initial_state = state;
        MultiFab<kNativeDimension> stage = state;
        MultiFab<kNativeDimension> residual = context.rhs_scratch_like(state);

        context.set_stage_time(0, 1);
        context.rhs_into(0, state, residual, 100);
        context.axpy(stage, Real(step), residual);

        context.set_stage_time(1, 1);
        residual.set_val(Real(0));
        context.rhs_into(0, stage, residual, 101);
        context.axpy(stage, Real(step), residual);
        context.lincomb(stage, Real(3) / Real(4), initial_state, Real(1) / Real(4), stage);

        context.set_stage_time(1, 2);
        residual.set_val(Real(0));
        context.rhs_into(0, stage, residual, 102);
        context.axpy(stage, Real(step), residual);
        context.lincomb(stage, Real(1) / Real(3), initial_state, Real(2) / Real(3), stage);
        context.commit_many({{&state, &stage}});
      });
  program.mark_bound();
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
    install_v5_callback_program(system, "test.program-runtime.projection-step.v5", "macro", {"gas"},
                                [](RuntimeProgramServices& context, double step) {
                                  context.begin_step(step);
                                  context.apply_projection(0, context.state(0));
                                });
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
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const std::vector<Resource> resources{{Resource::Kind::state, 0, 0, 0, -1, kGasComponents, 1}};
  install_v5_callback_program(
      sim, "test.program-runtime.project-recheck-accept.v5", "macro", {"gas"}, resources,
      [&consumed_solves](RuntimeProgramServices& ctx, double dt) {
        ctx.begin_step(dt);
        MultiFab<kNativeDimension>& state = ctx.state(0);
        MultiFab<kNativeDimension>& candidate = ctx.scratch_state(0, 0, state);
        candidate.set_val(Real(-1));

        auto field_outcome = ctx.solve_fields();
        const SolveReport field_report = field_outcome.consume(SolveConsumption::kAccept);
        if (!field_report.solved_value_available())
          throw std::logic_error("ProjectAndRecheck test did not receive a solved field value");
        ++consumed_solves;

        if (ctx.min_component(candidate, 0) <= Real(0)) {
          ctx.apply_projection(0, candidate);
          if (ctx.min_component(candidate, 0) <= Real(0))
            throw runtime::program::StepAttemptRejected(
                SolveStatus::kIterationLimit, "guard recheck",
                "ProjectAndRecheck projection did not repair the candidate");
        }
        ctx.commit_many({{&state, &candidate}});
      });

  sim.mark_bound();
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
  sim.set_program_block_map({0});

  using History = pops::test::program_v5::CallbackProgramHistory;
  int consumed_solves = 0;
  install_v5_callback_program(
      sim, "test.program-runtime.project-recheck-reject.v5", "macro", {"gas"},
      [&consumed_solves](RuntimeProgramServices& ctx, double dt) {
        ctx.begin_step(dt);
        MultiFab<kNativeDimension>& state = ctx.state(0);
        MultiFab<kNativeDimension> candidate = ctx.scratch_state_like(state);
        candidate.set_val(Real(-1));

        auto field_outcome = ctx.solve_fields();
        const SolveReport field_report = field_outcome.consume(SolveConsumption::kAccept);
        if (!field_report.solved_value_available())
          throw std::logic_error("ProjectAndRecheck test did not receive a solved field value");
        ++consumed_solves;

        if (ctx.min_component(candidate, 0) < Real(3)) {
          ctx.apply_projection(0, candidate);
          ctx.store_history("gas.guard_candidate", candidate);
          ctx.rotate_histories();
          ctx.record_scalar("project_and_recheck.provisional", Real(1));
          if (ctx.min_component(candidate, 0) < Real(3))
            throw runtime::program::StepAttemptRejected(
                SolveStatus::kIterationLimit, "guard recheck",
                "ProjectAndRecheck candidate remained inadmissible");
        }
        ctx.commit_many({{&state, &candidate}});
      },
      {.diagnostics = {"project_and_recheck.provisional"}}, false,
      {History{"gas.guard_candidate", 1, kGasComponents, 0, "gas.U", "cell.conservative", "macro",
               "dense.linear"}});

  sim.mark_bound();
  EXPECT_THROW(sim.step(1e-3), runtime::program::StepAttemptRejected);

  EXPECT_EQ(consumed_solves, 1);
  EXPECT_EQ(sim.macro_step(), 0);
  EXPECT_DOUBLE_EQ(sim.time(), 0.0);
  EXPECT_EQ(sim.get_state("gas"), initial);
  EXPECT_FALSE(sim.history_initialized("gas.guard_candidate"));
  EXPECT_TRUE(sim.program_cache_slots().empty());
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
    install_v5_callback_program(system, "test.program-runtime.generated-projection.v5", "macro",
                                {"gas"}, [](RuntimeProgramServices& context, double step) {
                                  context.begin_step(step);
                                  context.apply_projection(0, context.state(0));
                                });
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
  install_v5_callback_program(system, "pops.test.program-runtime.generated-projection.nonfinite.v5",
                              "macro", {"gas"}, [](RuntimeProgramServices& context, double step) {
                                context.begin_step(step);
                                MultiFab<kNativeDimension>& candidate = context.state(0);
                                context.apply_projection(0, candidate);
                              });
  EXPECT_THROW(system.step(1.0e-3), std::runtime_error);
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

  MultiFab<kNativeDimension> wrong_ncomp = [&] {
    const auto accepted_view = system.block_state(0);
    const MultiFab<kNativeDimension>& accepted = *accepted_view.get();
    return MultiFab<kNativeDimension>(accepted.layout(), accepted.distribution(),
                                      accepted.local_rank(), accepted.ncomp() - 1,
                                      accepted.ghosts());
  }();
  EXPECT_THROW(system.block_project(0, wrong_ncomp), std::invalid_argument);
  EXPECT_EQ(projection_calls, 0);

  System<kNativeDimension> foreign_system(unit_domain_config<kNativeDimension>(n + 1));
  install_execution_lane(foreign_system,
                         "pops.test.program-runtime.projection-field-preflight.foreign");
  add_projecting_gas(foreign_system, gamma);
  MultiFab<kNativeDimension> foreign_state = [&] {
    const auto foreign_view = foreign_system.block_state(0);
    return MultiFab<kNativeDimension>(*foreign_view.get());
  }();
  EXPECT_THROW(system.block_project(0, foreign_state), std::invalid_argument);
  EXPECT_EQ(projection_calls, 0);
  EXPECT_EQ(system.get_state("gas"), initial);
}

TEST(ProgramRuntime, EmbeddedBoundaryRejectsUnqualifiedBoundaryLinearizationEntryPoints) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  using Context = runtime::program::ProgramExecutionServices<kNativeDimension>;
  using Field = MultiFab<kNativeDimension>;
  EXPECT_FALSE((HasUnqualifiedBoundaryLinearization<Context, Field>));
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
  const ExecutionLane& lane = system.prepared_boundary_execution_lane();
  const std::vector<double> accepted = system.get_state("gas");

  System<kNativeDimension> foreign_system(config);
  install_execution_lane(foreign_system, "pops.test.program-runtime.foreign-prepared-boundary");
  add_boundary_gas(foreign_system, gamma);
  foreign_system.set_state("gas", initial);
  foreign_system.set_program_block_map({0});
  const ExecutionLane& foreign_lane = foreign_system.prepared_boundary_execution_lane();
  using BoundarySession = RuntimeProgramServices::block_boundary_session_type;
  std::shared_ptr<BoundarySession> foreign_session;
  install_v5_callback_program(
      foreign_system, "pops.test.program-runtime.foreign-prepared-boundary.v5",
      "test.prepared-boundary", {"gas"}, [&](RuntimeProgramServices& context, double step) {
        context.begin_step(step);
        MultiFab<kNativeDimension>& foreign_state = context.state(0);
        const auto foreign_point = context.boundary_evaluation_point(17);
        foreign_session =
            context.prepare_block_boundary_session(0, foreign_state, foreign_point, foreign_lane);
      });
  foreign_system.step(1.0e-3);
  ASSERT_NE(foreign_session, nullptr);

  install_v5_callback_program(
      system, "pops.test.program-runtime.prepared-boundary.v5", "test.prepared-boundary", {"gas"},
      [&](RuntimeProgramServices& context, double step) {
        context.begin_step(step);
        const auto point = context.boundary_evaluation_point(17);
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

        MultiFab<kNativeDimension> non_finite_state = context.rhs_scratch_like(state);
        non_finite_state.set_val(std::numeric_limits<Real>::quiet_NaN());
        residual.set_val(Real(-7));
        EXPECT_THROW(
            context.boundary_residual_into_at(point, 0, non_finite_state, residual, *session),
            std::runtime_error);
        EXPECT_EQ(context.norm_inf(0, residual), Real(7));

        MultiFab<kNativeDimension> non_finite_direction = context.rhs_scratch_like(state);
        non_finite_direction.set_val(std::numeric_limits<Real>::quiet_NaN());
        jvp.set_val(Real(-8));
        EXPECT_THROW(
            context.boundary_jvp_into_at(point, 0, state, non_finite_direction, jvp, *session),
            std::runtime_error);
        EXPECT_EQ(context.norm_inf(0, jvp), Real(8));

        residual.set_val(Real(-3));
        auto mismatched_point = point;
        if (lane.rank() != 0)
          ++mismatched_point.stage;
        if (lane.size() == 1)
          ++mismatched_point.stage;
        if (lane.size() == 1) {
          EXPECT_THROW(
              context.boundary_residual_into_at(mismatched_point, 0, state, residual, *session),
              std::invalid_argument);
        } else {
          EXPECT_THROW(
              context.boundary_residual_into_at(mismatched_point, 0, state, residual, *session),
              std::runtime_error);
        }
        EXPECT_EQ(context.norm_inf(0, residual), Real(3));

        residual.set_val(Real(-4));
        if (lane.size() == 1) {
          EXPECT_THROW(
              context.rhs_core_into_at(mismatched_point, 0, state, residual, false, *session),
              std::invalid_argument);
        } else {
          EXPECT_THROW(
              context.rhs_core_into_at(mismatched_point, 0, state, residual, false, *session),
              std::runtime_error);
        }
        EXPECT_EQ(context.norm_inf(0, residual), Real(4));
        EXPECT_NO_THROW(context.rhs_core_into_at(point, 0, state, residual, false, *session));
        EXPECT_GT(norm_inf_all_components(residual), Real(0));

        residual.set_val(Real(-3));
        if (lane.size() == 1) {
          EXPECT_THROW(
              context.boundary_residual_into_at(point, 0, state, residual, *foreign_session),
              std::invalid_argument);
        } else {
          EXPECT_THROW(
              context.boundary_residual_into_at(point, 0, state, residual, *foreign_session),
              std::runtime_error);
        }
        EXPECT_EQ(context.norm_inf(0, residual), Real(3));

        if (lane.size() > 1) {
          const int divergent_block = lane.rank() == 0 ? 0 : 1;
          EXPECT_THROW(
              context.boundary_residual_into_at(point, divergent_block, state, residual, *session),
              std::runtime_error);
          EXPECT_EQ(context.norm_inf(0, residual), Real(3));
        }

        EXPECT_NO_THROW(context.boundary_residual_into_at(point, 0, state, residual, *session));
        EXPECT_GT(norm_inf_all_components(residual), Real(0));
      });
  system.step(1.0e-3);
  EXPECT_EQ(system.get_state("gas"), accepted);
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
  sim.set_program_block_map({0});

  using History = pops::test::program_v5::CallbackProgramHistory;
  install_v5_callback_program(sim, "test.program-runtime.rejected-attempt-rollback.v5", "macro",
                              {"gas"},
                              [](RuntimeProgramServices& ctx, double dt) {
                                ctx.begin_step(dt);
                                MultiFab<kNativeDimension>& state = ctx.state(0);
                                MultiFab<kNativeDimension> bump = state;
                                bump.set_val(Real(dt));
                                ctx.axpy(state, Real(1), bump);
                                ctx.store_history("gas.U", state);
                                ctx.rotate_histories();
                                ctx.record_scalar("provisional", Real(42));
                                throw runtime::program::StepAttemptRejected(
                                    SolveStatus::kIterationLimit, "solve",
                                    "fault injection after provisional publications");
                              },
                              {.diagnostics = {"provisional"}}, false,
                              {History{"gas.U", 1, kGasComponents, 0, "gas.U", "cell.conservative",
                                       "macro", "dense.linear"}});

  EXPECT_THROW(sim.step(1e-3), runtime::program::StepAttemptRejected);
  EXPECT_EQ(sim.macro_step(), 0);
  EXPECT_DOUBLE_EQ(sim.time(), 0.0);
  EXPECT_EQ(sim.get_state("gas"), initial);
  EXPECT_FALSE(sim.history_initialized("gas.U"));
  EXPECT_TRUE(sim.program_cache_slots().empty());
  EXPECT_TRUE(sim.program_diagnostics().empty());
}
