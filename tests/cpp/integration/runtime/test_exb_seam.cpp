// C++ equivalent of the ExB row of tests/python/unit/runtime/test_seam_combinations.py.
//
// The fixture intentionally uses the generated compiled System path: the background-density
// Poisson solve publishes every electric-field gradient through the sealed auxiliary graph, while
// the three magnetic components remain explicit inputs.  CartesianExBDriftND consumes those exact
// slots, so this covers the real E x B transport seam rather than a uniform-advection surrogate.
#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "native_dso_compiler.hpp"
#include "program_v5_fixture.hpp"
#include "test_harness.hpp"  // pops::test::Checker
#include <pops/parallel/execution_lane.hpp>
#include <pops/physics/bricks/bricks.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/program/program_execution_services.hpp>
#include <pops/runtime/system.hpp>
#include <pops/runtime/system/derived_aux_provider.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
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

namespace {

constexpr const char* kBlock = "blk";
constexpr const char* kField = "electric-field";
constexpr const char* kFieldSlot = "tests.exb-seam/electric-field@1";
constexpr const char* kFieldOutputOwner = "tests.exb-seam/electric-output@1";

using ExbModel = CompositeModel<CartesianExBDriftND<kNativeDimension>, NoSource, BackgroundDensity>;
using ExbProgramExecutionServices = runtime::program::ProgramExecutionServices<kNativeDimension>;
using ExbProgramCallback = std::function<void(ExbProgramExecutionServices&, double)>;

std::vector<ExbProgramCallback>& exb_program_callbacks() {
  static std::vector<ExbProgramCallback> callbacks;
  return callbacks;
}

extern "C" void pops_test_exb_program_callback(std::uint64_t identifier, void* opaque, double dt) {
  auto& callbacks = exb_program_callbacks();
  if (opaque == nullptr || identifier >= callbacks.size())
    throw std::logic_error("ABI-v5 ExB callback received an invalid dispatch token");
  callbacks.at(static_cast<std::size_t>(identifier))(
      *static_cast<ExbProgramExecutionServices*>(opaque), dt);
}

// Smooth periodic density bump, matching the Python seam's perturbation around one.
std::vector<double> seed_density(int n) {
  std::size_t cells = 1;
  for (int axis = 0; axis < kNativeDimension; ++axis)
    cells *= static_cast<std::size_t>(n);
  std::vector<double> rho(cells);
  const double pi = 3.14159265358979323846;
  for (std::size_t cell = 0; cell < cells; ++cell) {
    std::size_t remainder = cell;
    double perturbation = 1.0;
    for (int axis = 0; axis < kNativeDimension; ++axis) {
      const int index = static_cast<int>(remainder % static_cast<std::size_t>(n));
      remainder /= static_cast<std::size_t>(n);
      perturbation *= std::sin(2 * pi * (static_cast<double>(index) + 0.5) / n);
    }
    rho[cell] = 1.0 + 0.1 * perturbation;
  }
  return rho;
}

ExbModel exb_seam_model(double n0) {
  ExbModel model{};
  model.ell = BackgroundDensity{Real(1), static_cast<Real>(n0)};
  return model;
}

void install_execution_lane(System<kNativeDimension>& system) {
  system.install_prepared_boundary_execution_lane(std::make_shared<ExecutionLane>(
      ExecutionLane::duplicate_world_collectively("tests.exb-seam/runtime-instance@1")));
}

struct ExbAuxiliaryAuthority {
  std::vector<runtime::system::AuxiliaryComponentKey> field_outputs;
  std::vector<runtime::system::AuxiliaryComponentKey> magnetic_inputs;
};

ExbAuxiliaryAuthority install_exb_auxiliary_authority(System<kNativeDimension>& system) {
  using namespace runtime::system;
  constexpr int Dim = kNativeDimension;
  constexpr const char* backend = "tests.exb-seam/cartesian-cg@1";

  system.register_configured_field_solver_provider(
      "cartesian_cg", backend,
      PreparedProviderOptions{
          "pops.system.cartesian-cg-options@1",
          {{"abs_tol", 0.0}, {"max_iterations", std::int64_t{200}}, {"rel_tol", 1.0e-8}}});
  system.set_field_solver_plan(kFieldSlot, "tests.exb-seam/electric-plan@1",
                               "tests.exb-seam/electric-provider@1", kFieldOutputOwner, kBlock,
                               kField, {"tests.exb-seam/blk/electric-rhs@1"}, {kBlock}, {kField},
                               {1.0}, backend);
  system.set_field_topology_authority(kFieldSlot, "builtin_rectangular_cell_graph_v1",
                                      "tests.exb-seam/periodic-cartesian@1",
                                      "tests.exb-seam/periodic-cartesian:v1");
  const std::vector<std::string> periodic_faces(static_cast<std::size_t>(2 * Dim), "periodic");
  const std::vector<double> zero_faces(static_cast<std::size_t>(2 * Dim), 0.0);
  system.set_field_boundary_plan(kFieldSlot, periodic_faces, zero_faces, zero_faces, zero_faces);

  AuxiliaryStorageShape<Dim> shape;
  for (int axis = 0; axis < Dim; ++axis)
    shape.halo[axis] = 1;
  const AuxiliaryComponentContract field_contract{"cell-average", "cell", "unitless", "field",
                                                  "scalar"};
  const AuxiliaryComponentContract magnetic_contract{"cell-average", "cell", "tesla", "input",
                                                     "scalar"};

  ExbAuxiliaryAuthority authority;
  std::vector<AuxiliaryOutput<Dim>> field_output_values;
  authority.field_outputs.reserve(static_cast<std::size_t>(Dim + 1));
  field_output_values.reserve(static_cast<std::size_t>(Dim + 1));
  for (int component = 0; component <= Dim; ++component) {
    AuxiliaryComponentKey key{
        kFieldOutputOwner, "field", kField,
        component == 0 ? "potential" : "gradient-" + std::to_string(component - 1)};
    authority.field_outputs.push_back(key);
    field_output_values.push_back({std::move(key), field_contract, shape});
  }
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<Dim>{
      "tests.exb-seam/electric-field-output@1",
      AuxiliaryProviderKind::field_output,
      {AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryFreshness::evaluation},
      std::move(field_output_values),
      {}});

  std::vector<AuxiliaryOutput<Dim>> magnetic_values;
  authority.magnetic_inputs.reserve(3);
  magnetic_values.reserve(3);
  for (int component = 0; component < 3; ++component) {
    AuxiliaryComponentKey key{"tests.exb-seam", "input", "magnetic",
                              "B-" + std::to_string(component)};
    authority.magnetic_inputs.push_back(key);
    magnetic_values.push_back({std::move(key), magnetic_contract, shape});
  }
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<Dim>{
      "tests.exb-seam/magnetic-input@1",
      AuxiliaryProviderKind::input,
      {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
      std::move(magnetic_values),
      {}});

  // The direct System builder resolves this plan by the compiled block name.  Preserve the provider
  // image: grad(phi)[0..Dim-1] then B[Dim..Dim+2].
  AuxiliaryConsumerProviderPlan<Dim> plan{kBlock, {}};
  plan.values.reserve(static_cast<std::size_t>(Dim + 3));
  for (int axis = 0; axis < Dim; ++axis)
    plan.values.push_back(
        {{authority.field_outputs[static_cast<std::size_t>(axis + 1)], field_contract, shape},
         static_cast<std::size_t>(axis)});
  for (int component = 0; component < 3; ++component)
    plan.values.push_back(
        {{authority.magnetic_inputs[static_cast<std::size_t>(component)], magnetic_contract, shape},
         static_cast<std::size_t>(Dim + component)});
  system.install_auxiliary_consumer_plan(std::move(plan));
  system.seal_auxiliary_providers();
  return authority;
}

void install_exb_forward_euler_program(System<kNativeDimension>& system) {
  const auto state = system.block_state(0);
  if (!state)
    throw std::logic_error("ExB Program fixture requires one materialized state");
  using Resource = test::program_v5::CallbackProgramResource;
  const std::vector<Resource> resources{
      {Resource::Kind::rhs, 0, 0, 0, -1, static_cast<std::uint32_t>(state->ncomp()),
       static_cast<std::uint32_t>(state->ghosts()[0])},
      {Resource::Kind::state, 1, 0, 0, -1, static_cast<std::uint32_t>(state->ncomp()),
       static_cast<std::uint32_t>(state->ghosts()[0])},
  };
  auto& callbacks = exb_program_callbacks();
  const auto callback_identifier = static_cast<std::uint64_t>(callbacks.size());
  callbacks.emplace_back([](ExbProgramExecutionServices& context, double dt) {
    context.begin_step(dt);
    context.set_stage_time(0, 1);
    MultiFab<kNativeDimension>& state = context.state(0);
    (void)consume_solve_outcome(context.solve_fields());

    MultiFab<kNativeDimension>& residual = context.rhs_scratch(0, 0, state);
    MultiFab<kNativeDimension>& next = context.scratch_state(1, 0, state);
    context.rhs_into(0, state, residual, 3000);
    context.lincomb(next, Real(1), state, static_cast<Real>(dt), residual);
    context.lincomb(state, Real(0), state, Real(1), next);
  });
#if !defined(POPS_TEST_TMPDIR)
  throw std::runtime_error("ABI-v5 ExB fixture requires POPS_TEST_TMPDIR");
#else
  static std::size_t fixture_index = 0;
  const std::string prefix =
      std::string(POPS_TEST_TMPDIR) + "/exb_program_callback_" + std::to_string(++fixture_index);
  const std::string source_path = prefix + ".cpp";
  const std::string library_path = prefix + ".so";
  {
    std::ofstream source(source_path);
    if (!source)
      throw std::runtime_error("cannot create ABI-v5 ExB fixture source");
    source << test::program_v5::callback_program_source(
        callback_identifier, "tests.exb-seam/forward-euler@1", "test.clock.macro", {kBlock},
        resources, "pops_test_exb_program_callback", "uniform");
  }
  const auto compiled = test::native_dso::compile_shared(source_path, library_path);
  if (!compiled.ok) {
    test::native_dso::report_compile_failure("test_exb_seam", compiled);
    throw std::runtime_error("ABI-v5 ExB fixture compilation failed");
  }
  system.install_program(library_path);
#endif
}

}  // namespace

static int pops_run_test_exb_seam(int argc, char** argv) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  pops::test::Checker chk;

  const int n = 32;
  const std::vector<double> rho = seed_density(n);
  const double n0 = std::accumulate(rho.begin(), rho.end(), 0.0) / static_cast<double>(rho.size());

  SystemConfig<kNativeDimension> cfg;
  for (int axis = 0; axis < kNativeDimension; ++axis) {
    cfg.shape[axis] = n;
    cfg.lower[axis] = Real(0);
    cfg.upper[axis] = Real(1);
    cfg.periodicity[axis] = true;
  }

  System<kNativeDimension> sys(cfg);
  install_execution_lane(sys);
  sys.install_block_state_route(kBlock, "tests.exb-seam/blk/state@1");
  sys.set_poisson("charge_density", "cartesian_cg", "periodic");
  const ExbAuxiliaryAuthority authority = install_exb_auxiliary_authority(sys);
  const ExbModel model = exb_seam_model(n0);
  add_compiled_model(sys, kBlock, model, "minmod", "rusanov", "conservative", "explicit");
  sys.register_elliptic_field(kBlock, kField, authority.field_outputs, /*gradient_sign=*/1);
  sys.set_block_elliptic_field(kBlock, kField, make_poisson_rhs(model));

  const std::vector<double> zero(rho.size(), 0.0);
  const std::vector<double> unit_bz(rho.size(), 1.0);
  for (int component = 0; component < 3; ++component)
    sys.stage_auxiliary_input(authority.magnetic_inputs[static_cast<std::size_t>(component)],
                              component == 2 ? unit_bz : zero);
  sys.set_density(kBlock, rho);
  install_exb_forward_euler_program(sys);

  // step_cfl computes the generated transport speed before entering the Program body.  Publish the
  // staged B input and solve/publish the named electric field first, so that the same accepted
  // grad(phi), B provider image is visible both to this initial CFL query and to the Program RHS.
  sys.refresh_auxiliary({"tests.exb-seam", 0, 0, 0, 0, 0, 0,
                         runtime::system::AuxiliaryEvaluationEvent::initialization});
  const MultiFab<kNativeDimension> state = [&] {
    const auto state_view = sys.block_state(0);
    return MultiFab<kNativeDimension>(*state_view.get());
  }();
  (void)consume_solve_outcome(sys.solve_fields_from_state(kField, 0, state));

  const double m0 = sys.mass(kBlock);
  chk(std::isfinite(m0), "initial mass finite");

  const double dt = sys.step_cfl(0.4);
  chk(std::isfinite(dt), "exb seam: step_cfl returns a finite dt");
  chk(dt > 0.0, "exb seam: step_cfl returns a positive dt");

  const double m1 = sys.mass(kBlock);
  chk(std::isfinite(m1), "post-step mass finite");
  const double dm = std::fabs(m1 - m0);
  chk(dm < 1e-9 * (std::fabs(m0) + 1.0), "exb seam: mass conserved across one CFL step");

  std::printf("EXBSEAM dt=%.3e m0=%.17e m1=%.17e dm=%.3e\n", dt, m0, m1, dm);
  if (chk.fails() == 0)
    std::printf(
        "OK test_exb_seam (compiled System ExB/background seam: field gradients and B "
        "inputs drive one conservative CFL step)\n");
  return chk.failed();
}

TEST(test_exb_seam, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_exb_seam, "test_exb_seam"), 0);
}
