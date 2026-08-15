// Exact-ranked System integration coverage for field-driven Cartesian E x B transport.
//
// The test deliberately owns no model-spec or implicit auxiliary fallback: a named exact field
// publishes phi plus one gradient per native axis into the sealed provider graph, and the three
// physical Cartesian magnetic components are staged through explicit input providers.  The compiled
// block consumes that exact ordered plan, so the same source qualifies the native 1-D, 2-D and 3-D
// specializations without a planar adapter.
#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "test_harness.hpp"
#include <pops/mesh/execution/for_each.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/physics/bricks/elliptic.hpp>
#include <pops/physics/bricks/hyperbolic.hpp>
#include <pops/physics/bricks/source.hpp>
#include <pops/physics/composition/composite.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/program/program_context.hpp>
#include <pops/runtime/system.hpp>
#include <pops/runtime/system/derived_aux_provider.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace pops {

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

using namespace pops;

namespace {

constexpr int Dim = kNativeDimension;
using ExBModel = CompositeModel<CartesianExBDriftND<Dim>, NoSource, BackgroundDensity>;
using NativeField = MultiFab<Dim>;
using NativeSystem = System<Dim>;
using NativeSystemConfig = SystemConfig<Dim>;

using runtime::system::AuxiliaryComponentContract;
using runtime::system::AuxiliaryComponentKey;
using runtime::system::AuxiliaryConsumerProviderPlan;
using runtime::system::AuxiliaryEvaluationEvent;
using runtime::system::AuxiliaryEvaluationPoint;
using runtime::system::AuxiliaryFreshness;
using runtime::system::AuxiliaryOutput;
using runtime::system::AuxiliaryProviderKind;
using runtime::system::AuxiliaryStorageShape;
using runtime::system::PreparedAuxiliaryProvider;

std::size_t native_cell_count(int cells_per_axis) {
  std::size_t count = 1;
  for (int axis = 0; axis < Dim; ++axis)
    count *= static_cast<std::size_t>(cells_per_axis);
  return count;
}

std::vector<double> seed_density(int cells_per_axis, double& mean) {
  constexpr double pi = 3.14159265358979323846;
  std::vector<double> density(native_cell_count(cells_per_axis));
  mean = 0.0;
  for (std::size_t cell = 0; cell < density.size(); ++cell) {
    std::size_t quotient = cell;
    double perturbation = 1.0;
    for (int axis = 0; axis < Dim; ++axis) {
      const int coordinate = static_cast<int>(quotient % static_cast<std::size_t>(cells_per_axis));
      quotient /= static_cast<std::size_t>(cells_per_axis);
      const double location = (static_cast<double>(coordinate) + 0.5) / cells_per_axis;
      perturbation *= std::sin(2.0 * pi * location);
    }
    density[cell] = 1.0 + 0.1 * perturbation;
    mean += density[cell];
  }
  mean /= static_cast<double>(density.size());
  return density;
}

NativeSystemConfig periodic_unit_config(int cells_per_axis) {
  NativeSystemConfig config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = cells_per_axis;
    config.lower[axis] = Real(0);
    config.upper[axis] = Real(1);
    config.periodicity[axis] = true;
  }
  return config;
}

void install_periodic_field_boundary(NativeSystem& system, const std::string& slot) {
  const std::vector<std::string> kinds(static_cast<std::size_t>(2 * Dim), "periodic");
  const std::vector<double> values(static_cast<std::size_t>(2 * Dim), 0.0);
  system.set_field_boundary_plan(slot, kinds, values, values, values);
}

void add_background_rhs(NativeSystem& system, const std::string& field, double background) {
  system.set_block_elliptic_field(
      "exb", field, [background](const NativeField& state, NativeField& rhs) {
        for (std::size_t local = 0; local < state.local_size(); ++local) {
          const auto source = state.fab(local).view();
          const auto destination = rhs.fab(local).view();
          const Box<Dim>& valid = state.box(local);
          for_each_cell(valid, [=] POPS_HD(const Index<Dim>& cell) {
            destination(cell, 0) += source(cell, 0) - static_cast<Real>(background);
          });
        }
        Kokkos::fence();
      });
}

void install_forward_euler_exb_program(NativeSystem& system, const std::string& field_slot) {
  system.set_program_block_map({0});
  runtime::program::ProgramContext<Dim> context(&system);
  context.configure_primary_clock("test.exb-seam.clock");
  context.install([context, field_slot](double dt) {
    context.begin_step(dt);
    context.set_stage_time(0, 1);
    NativeField& state = context.state(0);
    (void)consume_solve_outcome(context.solve_fields_from_state_at(
        context.boundary_evaluation_point(0), field_slot, 0, state));
    NativeField residual = context.rhs_scratch_like(state);
    context.rhs_into(0, state, residual, 0);
    context.axpy(state, static_cast<Real>(dt), residual);
  });
  system.set_program_block_map({0});
}

bool finite_and_nontrivial(const std::vector<double>& values) {
  double maximum = 0.0;
  for (const double value : values) {
    if (!std::isfinite(value))
      return false;
    maximum = std::fmax(maximum, std::fabs(value));
  }
  return maximum > 1.0e-8;
}

}  // namespace

static int pops_run_test_exb_seam(int argc, char** argv) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  test::Checker chk;

  constexpr int cells_per_axis = 32;
  double background = 0.0;
  const std::vector<double> density = seed_density(cells_per_axis, background);
  NativeSystem system(periodic_unit_config(cells_per_axis));

  AuxiliaryStorageShape<Dim> shape;
  for (int axis = 0; axis < Dim; ++axis)
    shape.halo[axis] = 1;
  const AuxiliaryComponentContract field_contract{"cell-average", "cell", "unitless", "field",
                                                  "scalar"};
  const AuxiliaryComponentContract magnetic_contract{"cell-average", "cell", "unitless", "input",
                                                     "scalar"};

  const std::string field_owner = "test.exb-seam.field";
  const std::string field_name = "electrostatic";
  std::vector<AuxiliaryComponentKey> field_keys;
  std::vector<AuxiliaryOutput<Dim>> field_outputs;
  field_keys.reserve(static_cast<std::size_t>(Dim + 1));
  field_outputs.reserve(static_cast<std::size_t>(Dim + 1));
  for (int component = 0; component <= Dim; ++component) {
    AuxiliaryComponentKey key{
        field_owner, "field", field_name,
        component == 0 ? "potential" : "gradient-" + std::to_string(component - 1)};
    field_keys.push_back(key);
    field_outputs.push_back({std::move(key), field_contract, shape});
  }
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<Dim>{
      "test.exb-seam.field-output@1",
      AuxiliaryProviderKind::field_output,
      {AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryFreshness::evaluation},
      std::move(field_outputs),
      {}});

  std::array<AuxiliaryComponentKey, 3> magnetic_keys;
  std::vector<AuxiliaryOutput<Dim>> magnetic_outputs;
  magnetic_outputs.reserve(magnetic_keys.size());
  for (int component = 0; component < static_cast<int>(magnetic_keys.size()); ++component) {
    magnetic_keys[static_cast<std::size_t>(component)] = {"test.exb-seam.magnetic", "input",
                                                          "magnetic-field",
                                                          "component-" + std::to_string(component)};
    magnetic_outputs.push_back(
        {magnetic_keys[static_cast<std::size_t>(component)], magnetic_contract, shape});
  }
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<Dim>{
      "test.exb-seam.magnetic-input@1",
      AuxiliaryProviderKind::input,
      {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
      std::move(magnetic_outputs),
      {}});

  AuxiliaryConsumerProviderPlan<Dim> plan;
  plan.consumer_qid = "exb";
  for (int axis = 0; axis < Dim; ++axis) {
    plan.values.push_back({{field_keys[static_cast<std::size_t>(axis + 1)], field_contract, shape},
                           static_cast<std::size_t>(axis)});
  }
  for (int component = 0; component < static_cast<int>(magnetic_keys.size()); ++component) {
    plan.values.push_back(
        {{magnetic_keys[static_cast<std::size_t>(component)], magnetic_contract, shape},
         static_cast<std::size_t>(Dim + component)});
  }
  system.install_auxiliary_consumer_plan(std::move(plan));
  system.seal_auxiliary_providers();

  system.install_block_state_route("exb", "test.exb-seam.state/exb@1");
  ExBModel model{};
  model.ell = BackgroundDensity{Real(1), static_cast<Real>(background)};
  add_compiled_model<Dim>(system, "exb", std::move(model), "minmod", "rusanov", "conservative",
                          "explicit");

  const std::string field_slot = "test.exb-seam.field-solver";
  const PreparedProviderOptions field_options{
      "pops.system.cartesian-cg-options@1",
      {{"abs_tol", 0.0}, {"max_iterations", std::int64_t{2000}}, {"rel_tol", 1.0e-10}}};
  system.register_configured_field_solver_provider("cartesian_cg", field_slot, field_options);
  system.set_field_solver_plan(
      field_slot, "test.exb-seam.field-plan@1", "test.exb-seam.field-rhs@1", field_owner, "exb",
      field_name, {"test.exb-seam.field-rhs/exb@1"}, {"exb"}, {field_name}, {1.0}, field_slot);
  system.set_field_topology_authority(field_slot, "builtin_rectangular_cell_graph_v1",
                                      "test.exb-seam.periodic-topology@1",
                                      "test.exb-seam.periodic-topology-digest@1");
  install_periodic_field_boundary(system, field_slot);
  system.register_elliptic_field("exb", field_name, field_keys, 1);
  add_background_rhs(system, field_name, background);

  system.set_density("exb", density);
  for (int component = 0; component < static_cast<int>(magnetic_keys.size()); ++component) {
    system.stage_auxiliary_input(
        magnetic_keys[static_cast<std::size_t>(component)],
        std::vector<double>(density.size(), static_cast<double>(component + 1)));
  }
  system.refresh_auxiliary(AuxiliaryEvaluationPoint{"test.exb-seam", 0, 0, 0, 0, 0, 0,
                                                    AuxiliaryEvaluationEvent::initialization});

  const SolveReport initial_field =
      consume_solve_outcome(system.solve_fields_from_state(field_slot, 0, system.block_state(0)));
  chk(initial_field.solved_value_available(), "initial exact field solve accepted its candidate");
  for (int axis = 0; axis < Dim; ++axis) {
    chk(finite_and_nontrivial(
            system.auxiliary_component(field_keys[static_cast<std::size_t>(axis + 1)])),
        "field-driven ExB gradient provider is finite and nontrivial");
  }

  install_forward_euler_exb_program(system, field_slot);
  const double mass_before = system.mass("exb");
  chk(std::isfinite(mass_before), "initial mass finite");

  const double dt = system.step_cfl(0.4);
  chk(std::isfinite(dt), "field-driven ExB step_cfl returns a finite dt");
  chk(dt > 0.0, "field-driven ExB step_cfl returns a positive dt");

  const std::vector<double> state_after = system.get_state("exb");
  bool state_finite = true;
  for (const double value : state_after)
    state_finite = state_finite && std::isfinite(value);
  chk(state_finite, "post-step ExB state is finite");

  const double mass_after = system.mass("exb");
  chk(std::isfinite(mass_after), "post-step mass finite");
  const double mass_change = std::fabs(mass_after - mass_before);
  chk(mass_change < 1.0e-9 * (std::fabs(mass_before) + 1.0),
      "field-driven ExB mass is conserved across one CFL step");

  std::printf("EXBSEAM dim=%d dt=%.3e m0=%.17e m1=%.17e dm=%.3e\n", Dim, dt, mass_before,
              mass_after, mass_change);
  if (chk.fails() == 0)
    std::printf("OK test_exb_seam (exact-ranked System field-driven ExB)\n");
  return chk.failed();
}

TEST(test_exb_seam, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_exb_seam, "test_exb_seam"), 0);
}
