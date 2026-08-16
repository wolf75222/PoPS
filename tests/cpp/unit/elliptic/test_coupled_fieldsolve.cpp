#include <gtest/gtest.h>

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/coupling/base/elliptic_rhs.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/physics/bricks/elliptic.hpp>
#include <pops/physics/bricks/source.hpp>
#include <pops/physics/composition/composite.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/system/prepared_field_solver_component.hpp>
#include <pops/runtime/system.hpp>
#include <pops/runtime/system/derived_aux_provider.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace pops {

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

namespace {

constexpr int Dim = pops::kNativeDimension;
using NativeSystem = pops::System<Dim>;
using NativeField = pops::MultiFab<Dim>;
using ChargeModel =
    pops::CompositeModel<pops::nd::ScalarAdvection<Dim>, pops::NoSource, pops::ChargeDensity>;

static_assert(ChargeModel::n_providers == 0);

void install_execution_lane(NativeSystem& system) {
  system.install_prepared_boundary_execution_lane(
      std::make_shared<pops::ExecutionLane>(pops::ExecutionLane::duplicate_world_collectively(
          "test.coupled-fieldsolve.runtime-instance@1")));
}

pops::SystemConfig<Dim> config(int cells) {
  pops::SystemConfig<Dim> result;
  for (int axis = 0; axis < Dim; ++axis) {
    result.shape[axis] = cells;
    result.lower[axis] = pops::Real(0);
    result.upper[axis] = pops::Real(1);
    result.periodicity[axis] = true;
  }
  return result;
}

std::size_t cell_count(int cells) {
  std::size_t count = 1;
  for (int axis = 0; axis < Dim; ++axis)
    count *= static_cast<std::size_t>(cells);
  return count;
}

std::vector<double> charge_density(int cells, double amplitude, double phase) {
  std::vector<double> density(cell_count(cells));
  for (std::size_t ordinal = 0; ordinal < density.size(); ++ordinal) {
    std::size_t remainder = ordinal;
    double value = amplitude;
    for (int axis = 0; axis < Dim; ++axis) {
      const int coordinate = static_cast<int>(remainder % static_cast<std::size_t>(cells));
      remainder /= static_cast<std::size_t>(cells);
      const double x = (coordinate + 0.5) / cells;
      value *= std::cos(2.0 * std::acos(-1.0) * (x + (axis == 0 ? phase : 0.0)));
    }
    density[ordinal] = value;
  }
  return density;
}

void add_charge_block(NativeSystem& system, const std::string& name) {
  system.install_block_state_route(name, "test.coupled-fieldsolve/" + name + "/state@1");
  pops::RealVector<Dim> velocity{};
  ChargeModel model{};
  model.hyp = pops::nd::ScalarAdvection<Dim>::prepare(velocity);
  model.ell.q = pops::Real(1);
  pops::add_compiled_model(system, name, std::move(model), "minmod", "rusanov", "conservative",
                           "explicit", static_cast<double>(pops::kPhysicalDefaultGamma), 1, true);
}

NativeSystem two_block_system(int cells, const std::vector<double>& first,
                              const std::vector<double>& second) {
  NativeSystem system(config(cells));
  install_execution_lane(system);
  add_charge_block(system, "first");
  add_charge_block(system, "second");
  system.seal_auxiliary_providers();
  system.set_poisson("charge_density", "cartesian_cg");
  system.set_density("first", first);
  system.set_density("second", second);
  return system;
}

double max_difference(const std::vector<double>& first, const std::vector<double>& second) {
  if (first.size() != second.size())
    return std::numeric_limits<double>::infinity();
  double result = 0;
  for (std::size_t index = 0; index < first.size(); ++index)
    result = std::max(result, std::abs(first[index] - second[index]));
  return result;
}

std::vector<pops::runtime::system::AuxiliaryComponentKey> install_field_outputs(
    NativeSystem& system, const std::string& owner, const std::string& field) {
  using namespace pops::runtime::system;
  AuxiliaryStorageShape<Dim> shape;
  for (int axis = 0; axis < Dim; ++axis)
    shape.halo[axis] = 1;
  const AuxiliaryComponentContract contract{"cell-average", "cell", "unitless", "field", "scalar"};
  std::vector<AuxiliaryOutput<Dim>> outputs;
  std::vector<AuxiliaryComponentKey> keys;
  for (int component = 0; component <= Dim; ++component) {
    AuxiliaryComponentKey key{
        owner, "field", field,
        component == 0 ? "potential" : "gradient-" + std::to_string(component - 1)};
    keys.push_back(key);
    outputs.push_back({std::move(key), contract, shape});
  }
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<Dim>{
      "test.field-output/" + owner + "/" + field,
      AuxiliaryProviderKind::field_output,
      {AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryFreshness::evaluation},
      std::move(outputs),
      {}});
  system.seal_auxiliary_providers();
  return keys;
}

std::vector<double> periodic_faces(double value) {
  return std::vector<double>(static_cast<std::size_t>(2 * Dim), value);
}

std::vector<std::string> periodic_kinds() {
  return std::vector<std::string>(static_cast<std::size_t>(2 * Dim), "periodic");
}

}  // namespace

TEST(test_coupled_fieldsolve, simultaneous_stage_rhs_uses_every_qualified_block) {
  constexpr int cells = 24;
  const auto first = charge_density(cells, 1.0, 0.0);
  const auto second = charge_density(cells, 0.6, 0.25);
  NativeSystem system = two_block_system(cells, first, second);

  const pops::SolveReport live_report = pops::consume_solve_outcome(system.solve_fields());
  ASSERT_TRUE(live_report.solved()) << live_report.reason;
  const std::vector<double> all_live = system.potential();
  ASSERT_EQ(all_live.size(), cell_count(cells));

  std::vector<const NativeField*> live_stages{&system.block_state(0), &system.block_state(1)};
  const pops::SolveReport simultaneous_report =
      pops::consume_solve_outcome(system.solve_fields_from_blocks(live_stages));
  ASSERT_TRUE(simultaneous_report.solved()) << simultaneous_report.reason;
  const std::vector<double> simultaneous = system.potential();
  double scale = 0;
  for (double value : all_live)
    scale = std::max(scale, std::abs(value));
  EXPECT_LE(max_difference(simultaneous, all_live), 1e-11 * std::max(1.0, scale));
  EXPECT_GT(scale, 0.0);

  NativeField second_stage = system.block_state(1);
  second_stage.set_val(pops::Real(0));
  std::vector<const NativeField*> override_stages{&system.block_state(0), &second_stage};
  const pops::SolveReport override_report =
      pops::consume_solve_outcome(system.solve_fields_from_blocks(override_stages));
  ASSERT_TRUE(override_report.solved()) << override_report.reason;
  const std::vector<double> override_potential = system.potential();

  NativeSystem first_only =
      two_block_system(cells, first, std::vector<double>(cell_count(cells), 0.0));
  const pops::SolveReport reference_report = pops::consume_solve_outcome(first_only.solve_fields());
  ASSERT_TRUE(reference_report.solved()) << reference_report.reason;
  const std::vector<double> reference = first_only.potential();
  EXPECT_LE(max_difference(override_potential, reference), 1e-11 * std::max(1.0, scale));
  EXPECT_GT(max_difference(override_potential, all_live), 1e-5)
      << "the second block's simultaneous stage contribution must affect the field";
  EXPECT_EQ(system.density("first"), first);
  EXPECT_EQ(system.density("second"), second);

  std::vector<const NativeField*> invalid{&system.block_state(0)};
  EXPECT_THROW((void)system.solve_fields_from_blocks(invalid), std::invalid_argument);
}

TEST(test_coupled_fieldsolve,
     named_prepared_provider_publishes_potential_and_signed_gradient_from_simultaneous_rhs) {
  constexpr int cells = 24;
  const auto first = charge_density(cells, 1.0, 0.0);
  const auto second = charge_density(cells, 0.6, 0.25);
  NativeSystem system(config(cells));
  install_execution_lane(system);
  add_charge_block(system, "first");
  add_charge_block(system, "second");

  const std::string slot = "qualified-coupled-provider";
  const pops::PreparedProviderOptions backend_options{
      "pops.system.cartesian-cg-options@1",
      {{"abs_tol", 0.0}, {"max_iterations", std::int64_t{200}}, {"rel_tol", 1.0e-8}}};
  system.register_configured_field_solver_provider("cartesian_cg", slot, backend_options);
  system.set_field_solver_plan(slot, "test.qualified-coupled-plan",
                               "test.qualified-coupled-provider", "test.qualified-coupled", "first",
                               "potential",
                               {"test.first/potential/rhs", "test.second/potential/rhs"},
                               {"first", "second"}, {"potential", "potential"}, {1.0, 1.0}, slot);
  system.set_field_topology_authority(slot, "builtin_rectangular_cell_graph_v1",
                                      "test.periodic-cartesian", "test.periodic-cartesian.v1");
  system.set_field_boundary_plan(slot, periodic_kinds(), periodic_faces(0.0), periodic_faces(0.0),
                                 periodic_faces(0.0));
  system.set_field_nullspace(
      slot, "pops.field-nullspace.operator-topology-derived",
      pops::PreparedProviderOptions{"pops.field-nullspace.operator-topology-derived.options@1",
                                    {{"gauge.value", 0.0}}});
  const auto outputs = install_field_outputs(system, "test.qualified-coupled", "potential");
  system.register_elliptic_field("first", "potential", outputs, -1);
  system.set_block_elliptic_field("first", "potential",
                                  [](const NativeField& state, NativeField& rhs) {
                                    pops::add_scaled_component(state, pops::Real(1), 0, rhs);
                                  });
  system.set_block_elliptic_field("second", "potential",
                                  [](const NativeField& state, NativeField& rhs) {
                                    pops::add_scaled_component(state, pops::Real(1), 0, rhs);
                                  });
  system.set_density("first", first);
  system.set_density("second", second);

  std::vector<const NativeField*> live_stages{&system.block_state(0), &system.block_state(1)};
  const pops::SolveReport live_report =
      pops::consume_solve_outcome(system.solve_fields_from_blocks(slot, live_stages));
  ASSERT_TRUE(live_report.solved()) << live_report.reason;
  const std::vector<double> all_live = system.field_potential_global(slot);
  ASSERT_EQ(all_live.size(), cell_count(cells));
  ASSERT_EQ(outputs.size(), static_cast<std::size_t>(Dim + 1));
  EXPECT_TRUE(system.field_topology_report(slot).empty())
      << "the builtin Cartesian CG route has no external component-topology report";

  for (int axis = 0; axis < Dim; ++axis) {
    const auto gradient = system.auxiliary_component(outputs[static_cast<std::size_t>(axis + 1)]);
    ASSERT_EQ(gradient.size(), all_live.size());
    const std::size_t stride = [&] {
      std::size_t value = 1;
      for (int lower_axis = 0; lower_axis < axis; ++lower_axis)
        value *= static_cast<std::size_t>(cells);
      return value;
    }();
    double error = 0;
    double reference = 0;
    for (std::size_t cell = 0; cell < all_live.size(); ++cell) {
      const int coordinate = static_cast<int>((cell / stride) % static_cast<std::size_t>(cells));
      const std::size_t minus = coordinate == 0 ? cell + stride * (cells - 1) : cell - stride;
      const std::size_t plus =
          coordinate == cells - 1 ? cell - stride * (cells - 1) : cell + stride;
      const double expected = -0.5 * cells * (all_live[plus] - all_live[minus]);
      error = std::max(error, std::abs(gradient[cell] - expected));
      reference = std::max(reference, std::abs(expected));
    }
    ASSERT_GT(reference, 1024.0 * std::numeric_limits<pops::Real>::epsilon());
    EXPECT_LE(error, 16.0 * std::numeric_limits<pops::Real>::epsilon() * std::max(1.0, reference));
  }

  NativeField second_stage = system.block_state(1);
  second_stage.set_val(pops::Real(0));
  std::vector<const NativeField*> override_stages{&system.block_state(0), &second_stage};
  const pops::SolveReport override_report =
      pops::consume_solve_outcome(system.solve_fields_from_blocks(slot, override_stages));
  ASSERT_TRUE(override_report.solved()) << override_report.reason;
  EXPECT_GT(max_difference(system.field_potential_global(slot), all_live), 1e-5)
      << "the named prepared plan must consume both qualified simultaneous stage slots";
  EXPECT_EQ(system.density("first"), first);
  EXPECT_EQ(system.density("second"), second);
}
