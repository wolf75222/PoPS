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
    NativeSystem& system, const std::string& owner, const std::string& field, bool seal = true) {
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
  if (seal)
    system.seal_auxiliary_providers();
  return keys;
}

std::vector<double> periodic_faces(double value) {
  return std::vector<double>(static_cast<std::size_t>(2 * Dim), value);
}

std::vector<std::string> periodic_kinds() {
  return std::vector<std::string>(static_cast<std::size_t>(2 * Dim), "periodic");
}

struct NativeLoad {
  std::string key;
  pops::Real scale;
  pops::runtime::system::NativeEllipticAttachmentRole role =
      pops::runtime::system::NativeEllipticAttachmentRole::output_and_rhs;
  std::string field_slot;
  std::string binding_identity;
  bool claim_outputs = false;
  int gradient_sign = 1;
};

NativeLoad rhs_only_load(std::string key, pops::Real scale) {
  NativeLoad result{std::move(key), scale};
  result.role = pops::runtime::system::NativeEllipticAttachmentRole::rhs_only;
  result.field_slot = "test.exact-load-plan";
  result.binding_identity = "test.resolved-provider/" + result.key;
  return result;
}

void stage_charge_package(
    NativeSystem& system, const std::string& block,
    const std::vector<pops::runtime::system::AuxiliaryComponentKey>& outputs,
    const std::vector<NativeLoad>& loads) {
  using namespace pops::runtime::system;
  system.install_block_state_route(block, "test.coupled-fieldsolve/" + block + "/state@1");
  auto capability = std::make_shared<NativePackageCapabilityState<Dim>>();
  capability->identity = block;
  auto installer = NativePackageCapabilityFactory<Dim>::block_installer(capability);
  system.stage_prepared_native_package(
      "test.named-load-package/" + block,
      [capability] {
        capability->phase = NativeCapabilityPhase::routes_open;
        capability->close_routes();
      },
      [capability, installer, block, outputs, loads] {
        capability->phase = NativeCapabilityPhase::install_open;
        ChargeModel model{};
        model.hyp = pops::nd::ScalarAdvection<Dim>::prepare(pops::RealVector<Dim>{});
        model.ell.q = pops::Real(1);
        PreparedNativeSystemPackage<Dim> package;
        package.consumer_qid = block;
        package.block = pops::prepare_compiled_system_block<Dim>(
            *installer, block, block, std::move(model), "minmod", "rusanov", "conservative",
            "explicit", static_cast<double>(pops::kPhysicalDefaultGamma), 1, true, 1);
        for (const auto& load : loads) {
          package.elliptic_attachments.push_back({
              load.key, "test.native-rhs/" + block + "/" + load.key,
              load.key == "fields_from_state" ||
                      (load.role == NativeEllipticAttachmentRole::rhs_only && !load.claim_outputs)
                  ? std::vector<AuxiliaryComponentKey>{}
                  : outputs,
              load.gradient_sign, [scale = load.scale](const NativeField& state, NativeField& rhs) {
                pops::add_scaled_component(state, scale, 0, rhs);
              }, load.role, load.field_slot, load.binding_identity});
        }
        installer->commit(std::move(package));
      },
      capability, capability);
}

NativeSystem named_load_system(int cells, const std::vector<std::string>& keys,
                               const std::vector<double>& coefficients,
                               const std::vector<NativeLoad>& attachments,
                               int output_gradient_sign = 1) {
  NativeSystem system(config(cells));
  install_execution_lane(system);
  const auto outputs = install_field_outputs(system, "test.exact-load-output", "potential", false);
  stage_charge_package(system, "first", outputs, attachments);
  const std::string slot = "test.exact-load-plan";
  system.register_configured_field_solver_provider(
      "cartesian_cg", slot,
      {"pops.system.cartesian-cg-options@1",
       {{"abs_tol", 0.0}, {"max_iterations", std::int64_t{200}}, {"rel_tol", 1.0e-8}}});
  std::vector<std::string> identities;
  for (const auto& key : keys)
    identities.push_back("test.resolved-provider/" + key);
  system.set_field_solver_plan(slot, "test.exact-load.plan@1", "test.exact-load.provider@1",
                               "test.exact-load-output", "first", "potential", identities,
                               std::vector<std::string>(keys.size(), "first"), keys,
                               coefficients, slot);
  system.set_field_topology_authority(slot, "builtin_rectangular_cell_graph_v1",
                                      "test.periodic-cartesian", "test.periodic-cartesian.v1");
  system.set_field_boundary_plan(slot, periodic_kinds(), periodic_faces(0.0), periodic_faces(0.0),
                                 periodic_faces(0.0));
  system.set_field_nullspace(
      slot, "pops.field-nullspace.operator-topology-derived",
      {"pops.field-nullspace.operator-topology-derived.options@1", {{"gauge.value", 0.0}}});
  system.register_elliptic_field("first", "potential", outputs, output_gradient_sign);
  return system;
}

void expect_negative_field_gradient(const NativeSystem& system, const std::vector<double>& potential,
                                    int cells) {
  for (int axis = 0; axis < Dim; ++axis) {
    SCOPED_TRACE(axis);
    const auto gradient = system.auxiliary_component(
        {"test.exact-load-output", "field", "potential", "gradient-" + std::to_string(axis)});
    ASSERT_EQ(gradient.size(), potential.size());
    std::size_t stride = 1;
    for (int lower_axis = 0; lower_axis < axis; ++lower_axis)
      stride *= static_cast<std::size_t>(cells);
    double error = 0;
    double magnitude = 0;
    for (std::size_t cell = 0; cell < potential.size(); ++cell) {
      const int coordinate = static_cast<int>((cell / stride) % static_cast<std::size_t>(cells));
      const std::size_t minus = coordinate == 0 ? cell + stride * (cells - 1) : cell - stride;
      const std::size_t plus = coordinate == cells - 1 ? cell - stride * (cells - 1) : cell + stride;
      const double expected = -0.5 * cells * (potential[plus] - potential[minus]);
      error = std::max(error, std::abs(gradient[cell] - expected));
      magnitude = std::max(magnitude, std::abs(expected));
    }
    ASSERT_GT(magnitude, 1024.0 * std::numeric_limits<pops::Real>::epsilon());
    EXPECT_LE(error, 16.0 * std::numeric_limits<pops::Real>::epsilon() * std::max(1.0, magnitude));
  }
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

TEST(test_coupled_fieldsolve, native_load_keys_use_the_case_output_and_each_exact_coefficient) {
  constexpr int cells = 24;
  const auto density = charge_density(cells, 1.0, 0.0);
  auto system = named_load_system(cells, {"load-a", "load-b"}, {2.0, -0.5},
                                  {{"load-a", pops::Real(3)}, {"load-b", pops::Real(5)}});
  ASSERT_NO_THROW(system.finalize_native_packages());
  system.set_density("first", density);
  auto reference = named_load_system(cells, {"combined-load"}, {1.0},
                                     {{"combined-load", pops::Real(3.5)}});
  ASSERT_NO_THROW(reference.finalize_native_packages());
  reference.set_density("first", density);
  const std::string slot = "test.exact-load-plan";
  const auto report = pops::consume_solve_outcome(system.solve_fields_from_blocks(
      slot, std::vector<const NativeField*>{&system.block_state(0)}));
  const auto reference_report = pops::consume_solve_outcome(reference.solve_fields_from_blocks(
      slot, std::vector<const NativeField*>{&reference.block_state(0)}));
  ASSERT_TRUE(report.solved()) << report.reason;
  ASSERT_TRUE(reference_report.solved()) << reference_report.reason;
  const auto potential = system.field_potential_global(slot);
  EXPECT_LE(max_difference(potential, reference.field_potential_global(slot)), 1.0e-11);
  EXPECT_GT(max_difference(potential, std::vector<double>(potential.size(), 0.0)), 1.0e-5);
  EXPECT_THROW((void)system.field_potential_global("load-a"), std::exception);
  EXPECT_THROW((void)system.field_potential_global("load-b"), std::exception);
}

TEST(test_coupled_fieldsolve, native_output_alias_requires_one_exact_provider_on_its_block) {
  auto supported = named_load_system(24, {"electron-load"}, {2.0},
                                     {{"potential", pops::Real(1)}});
  EXPECT_NO_THROW(supported.finalize_native_packages());
  auto ambiguous = named_load_system(24, {"load-a", "load-b"}, {2.0, -0.5},
                                     {{"potential", pops::Real(1)}});
  EXPECT_THROW(ambiguous.finalize_native_packages(), std::exception);
  EXPECT_THROW((void)ambiguous.block_state(0), std::exception);
}

TEST(test_coupled_fieldsolve, native_missing_and_foreign_loads_reject_before_block_publication) {
  auto missing = named_load_system(24, {"load-a", "load-b"}, {2.0, -0.5},
                                   {{"load-a", pops::Real(1)}});
  EXPECT_THROW(missing.finalize_native_packages(), std::exception);
  EXPECT_THROW((void)missing.block_state(0), std::exception);
  auto foreign = named_load_system(24, {"load-a"}, {2.0},
                                   {{"foreign-load", pops::Real(1)}});
  EXPECT_THROW(foreign.finalize_native_packages(), std::exception);
  EXPECT_THROW((void)foreign.block_state(0), std::exception);
}

TEST(test_coupled_fieldsolve, native_default_poisson_attachment_retains_its_prepared_block_rhs) {
  NativeSystem system(config(24));
  install_execution_lane(system);
  stage_charge_package(system, "first", {}, {{"fields_from_state", pops::Real(1)}});
  ASSERT_NO_THROW(system.finalize_native_packages());
  system.set_poisson("charge_density", "cartesian_cg");
  system.set_density("first", charge_density(24, 1.0, 0.0));
  const auto report = pops::consume_solve_outcome(system.solve_fields());
  ASSERT_TRUE(report.solved()) << report.reason;
  const auto potential = system.potential();
  EXPECT_GT(max_difference(potential, std::vector<double>(potential.size(), 0.0)), 1.0e-5);
}

TEST(test_coupled_fieldsolve, native_rhs_only_repeated_signed_coefficients_preserve_case_gradient) {
  constexpr int cells = 24;
  const auto density = charge_density(cells, 1.0, 0.0);
  auto system = named_load_system(cells, {"electron-load", "electron-load"}, {2.0, -0.5},
                                  {rhs_only_load("electron-load", pops::Real(0.5))}, -1);
  ASSERT_NO_THROW(system.finalize_native_packages());
  system.set_density("first", density);
  // Both resolved occurrences use the same density law: (2 - 0.5) * (0.5 rho) = 0.75 rho.
  auto reference = named_load_system(cells, {"combined-load"}, {1.0},
                                     {rhs_only_load("combined-load", pops::Real(0.75))}, -1);
  ASSERT_NO_THROW(reference.finalize_native_packages());
  reference.set_density("first", density);
  const std::string slot = "test.exact-load-plan";
  const auto report = pops::consume_solve_outcome(system.solve_fields_from_blocks(
      slot, std::vector<const NativeField*>{&system.block_state(0)}));
  const auto reference_report = pops::consume_solve_outcome(reference.solve_fields_from_blocks(
      slot, std::vector<const NativeField*>{&reference.block_state(0)}));
  ASSERT_TRUE(report.solved()) << report.reason;
  ASSERT_TRUE(reference_report.solved()) << reference_report.reason;
  const auto potential = system.field_potential_global(slot);
  EXPECT_LE(max_difference(potential, reference.field_potential_global(slot)), 1.0e-11);
  EXPECT_GT(max_difference(potential, std::vector<double>(potential.size(), 0.0)), 1.0e-5);
  expect_negative_field_gradient(system, potential, cells);
  EXPECT_EQ(system.density("first"), density);
  EXPECT_THROW((void)system.field_potential_global("electron-load"), std::exception);
}

TEST(test_coupled_fieldsolve, native_rhs_only_electron_and_ion_blocks_share_one_case_output) {
  constexpr int cells = 24;
  const auto electron_density = charge_density(cells, 1.0, 0.0);
  const auto ion_density = charge_density(cells, 0.6, 0.25);
  NativeSystem system(config(cells));
  install_execution_lane(system);
  const auto outputs = install_field_outputs(system, "test.exact-load-output", "potential", false);
  stage_charge_package(system, "electron", outputs,
                       {rhs_only_load("electron-load", pops::Real(1))});
  stage_charge_package(system, "ion", outputs, {rhs_only_load("ion-load", pops::Real(1))});
  const std::string slot = "test.exact-load-plan";
  system.register_configured_field_solver_provider(
      "cartesian_cg", slot,
      {"pops.system.cartesian-cg-options@1",
       {{"abs_tol", 0.0}, {"max_iterations", std::int64_t{200}}, {"rel_tol", 1.0e-8}}});
  system.set_field_solver_plan(
      slot, "test.exact-load.plan@1", "test.exact-load.provider@1", "test.exact-load-output",
      "electron", "potential",
      {"test.resolved-provider/electron-load", "test.resolved-provider/ion-load"},
      {"electron", "ion"}, {"electron-load", "ion-load"}, {-1.0, 1.0}, slot);
  system.set_field_topology_authority(slot, "builtin_rectangular_cell_graph_v1",
                                      "test.periodic-cartesian", "test.periodic-cartesian.v1");
  system.set_field_boundary_plan(slot, periodic_kinds(), periodic_faces(0.0), periodic_faces(0.0),
                                 periodic_faces(0.0));
  system.set_field_nullspace(
      slot, "pops.field-nullspace.operator-topology-derived",
      {"pops.field-nullspace.operator-topology-derived.options@1", {{"gauge.value", 0.0}}});
  system.register_elliptic_field("electron", "potential", outputs, -1);
  ASSERT_NO_THROW(system.finalize_native_packages());
  system.set_density("electron", electron_density);
  system.set_density("ion", ion_density);

  std::vector<double> combined(electron_density.size());
  for (std::size_t index = 0; index < combined.size(); ++index)
    combined[index] = -electron_density[index] + ion_density[index];
  auto reference = named_load_system(cells, {"combined-load"}, {1.0},
                                     {rhs_only_load("combined-load", pops::Real(1))}, -1);
  ASSERT_NO_THROW(reference.finalize_native_packages());
  reference.set_density("first", combined);
  const auto report = pops::consume_solve_outcome(system.solve_fields_from_blocks(
      slot, std::vector<const NativeField*>{&system.block_state(0), &system.block_state(1)}));
  const auto reference_report = pops::consume_solve_outcome(reference.solve_fields_from_blocks(
      slot, std::vector<const NativeField*>{&reference.block_state(0)}));
  ASSERT_TRUE(report.solved()) << report.reason;
  ASSERT_TRUE(reference_report.solved()) << reference_report.reason;
  const auto potential = system.field_potential_global(slot);
  EXPECT_LE(max_difference(potential, reference.field_potential_global(slot)), 1.0e-11);
  expect_negative_field_gradient(system, potential, cells);
  EXPECT_EQ(system.density("electron"), electron_density);
  EXPECT_EQ(system.density("ion"), ion_density);
}

TEST(test_coupled_fieldsolve, native_rhs_only_forged_authorities_reject_before_block_publication) {
  for (int forgery = 0; forgery != 5; ++forgery) {
    SCOPED_TRACE(forgery);
    auto load = rhs_only_load("electron-load", pops::Real(1));
    switch (forgery) {
      case 0: load.field_slot = "another-field-plan"; break;
      case 1: load.binding_identity = "test.resolved-provider/foreign-load"; break;
      case 2: load.claim_outputs = true; break;
      case 3: load.gradient_sign = -1; break;
      case 4:
        load.role = pops::runtime::system::NativeEllipticAttachmentRole::output_and_rhs;
        break;
    }
    auto system = named_load_system(24, {"electron-load"}, {1.0}, {load}, -1);
    EXPECT_THROW(system.finalize_native_packages(), std::exception);
    EXPECT_THROW((void)system.block_state(0), std::exception);
  }
}

TEST(test_coupled_fieldsolve, native_output_bearing_wrong_gradient_rejects_before_block_publication) {
  auto system = named_load_system(24, {"electron-load"}, {1.0},
                                  {{"electron-load", pops::Real(1)}}, -1);
  EXPECT_THROW(system.finalize_native_packages(), std::exception);
  EXPECT_THROW((void)system.block_state(0), std::exception);
}
