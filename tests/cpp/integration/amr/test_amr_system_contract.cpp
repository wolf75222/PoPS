/// @file
/// @brief Positive exact-ranked AmrSystem facade and prepared-package contracts.

#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "component_abi_test_helpers.hpp"
#include "explicit_amr_program.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/amr/composite_reduction.hpp>
#include <pops/physics/bricks/elliptic.hpp>
#include <pops/physics/bricks/hyperbolic.hpp>
#include <pops/physics/bricks/source.hpp>
#include <pops/physics/composition/composite.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/dynamic/prepared_execution_context.hpp>
#include <pops/runtime/program/amr_program_checkpoint.hpp>
#include <pops/runtime/system/derived_aux_provider.hpp>
#include <pops/runtime/system/exact_field_marshaling.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace {

template <int Dim>
void install_package_lane(pops::AmrSystem<Dim>& system, std::string_view identity) {
  auto lane = std::make_shared<pops::ExecutionLane>(
      pops::ExecutionLane::duplicate_world_collectively(identity));
  const PopsExecutionContextV1 raw = pops::component::test_support::host_execution_context();
  auto parent = std::make_shared<const pops::component::PreparedExecutionContextV1>(
      raw.execution_identity, raw.context_version, raw.memory_space, raw.backend_identity,
      raw.device_identity, raw.scalar_type, raw.storage_precision, raw.compute_precision,
      raw.accumulation_precision, raw.reduction_precision, raw.stream_handle, raw.stream_identity,
      raw.communicator_f_handle, raw.communicator_datatype_f_handle, raw.communicator_identity,
      raw.communicator_datatype_identity);
  auto execution =
      std::make_shared<const pops::component::PreparedExecutionContextV1>(parent->for_lane(*lane));
  system.install_prepared_boundary_execution_context(std::move(lane), std::move(execution));
}

template <int Dim>
class TracerModel : public pops::nd::ScalarAdvection<Dim> {
 public:
  using Base = pops::nd::ScalarAdvection<Dim>;
  using State = typename Base::State;

  explicit TracerModel(Base base) : Base(std::move(base)) {}

  [[nodiscard]] static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"tests.amr.system-contract.tracer", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    Base::serialize_exact_parameters(contract);
  }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};

template <int Dim>
using MagneticModel =
    pops::CompositeModel<pops::EulerND<Dim>, pops::MagneticLorentzForceND<Dim>, pops::NoElliptic>;

template <int Dim>
using GasModel = pops::CompositeModel<pops::EulerND<Dim>, pops::NoSource, pops::NoElliptic>;

template <int Dim>
pops::AmrSystemConfig<Dim> single_level_config(int cells = 8) {
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 1;
  config.regrid_every = 0;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = cells;
    config.lower[axis] = pops::Real(0);
    config.upper[axis] = pops::Real(axis + 1);
    config.periodicity[axis] = true;
  }
  return config;
}

template <int Dim>
std::size_t cell_count(const pops::Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

template <int Dim>
TracerModel<Dim> tracer_model() {
  pops::RealVector<Dim> velocity{};
  velocity[0] = pops::Real(0.25);
  return TracerModel<Dim>{pops::nd::ScalarAdvection<Dim>::prepare(velocity)};
}

template <int Dim>
void install_direct_tracer(pops::AmrSystem<Dim>& system, const std::string& name,
                           const std::string& consumer_qid);

template <int Dim>
void verify_rectangular_geometry_and_independent_periodicity() {
  pops::AmrSystemConfig<Dim> config = single_level_config<Dim>(4);
  std::size_t periodic_axes = 0;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 6 + 2 * axis;
    config.lower[axis] = pops::Real(-1 - axis);
    config.upper[axis] = config.lower[axis] + pops::Real(2 * (axis + 1));
    config.periodicity[axis] = axis % 2 == 0;
    periodic_axes += config.periodicity[axis] ? 1U : 0U;
  }

  constexpr const char* state_route = "tests.amr.system-contract/rectangular/state";
  pops::AmrSystem<Dim> system(config);
  system.install_block_state_route("tracer", state_route);
  std::vector<std::string> face_types;
  std::vector<std::string> face_identities;
  for (int axis = 0; axis < Dim; ++axis) {
    for (int side = 0; side < 2; ++side) {
      face_types.push_back(config.periodicity[axis] ? "periodic" : "foextrap");
      face_identities.push_back("tests.amr.system-contract/rectangular/face-" +
                                std::to_string(2 * axis + side));
    }
  }
  system.install_hyperbolic_boundary("tracer", "tests.amr.system-contract/rectangular/boundary@1",
                                     1, face_types,
                                     std::vector<double>(static_cast<std::size_t>(2 * Dim), 0.0),
                                     face_identities, {"Scalar"}, state_route);
  install_direct_tracer(system, "tracer", "tests.amr.system-contract/rectangular/physical-flux");
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));

  const pops::Geometry<Dim> geometry = system.prepared_amr_level_geometry(0);
  const pops::BoundaryTopology<Dim> topology = system.prepared_amr_boundary_topology();
  EXPECT_EQ(topology.periodic_pair_count(), periodic_axes);
  for (int axis = 0; axis < Dim; ++axis) {
    EXPECT_EQ(geometry.domain().length(axis), config.shape[axis]);
    EXPECT_EQ(geometry.lower()[axis], config.lower[axis]);
    EXPECT_EQ(geometry.upper()[axis], config.upper[axis]);
    EXPECT_EQ(geometry.spacing(axis),
              (config.upper[axis] - config.lower[axis]) / pops::Real(config.shape[axis]));
    EXPECT_EQ(topology.is_periodic(pops::Face<Dim>{axis, pops::BoundarySide::lower}),
              config.periodicity[axis]);
    EXPECT_EQ(topology.is_periodic(pops::Face<Dim>{axis, pops::BoundarySide::upper}),
              config.periodicity[axis]);
  }
}

template <int Dim>
GasModel<Dim> gas_model() {
  return GasModel<Dim>{
      {}, pops::EulerND<Dim>{pops::Real(1.4)}, pops::NoSource{}, pops::NoElliptic{}};
}

template <int Dim>
void install_gas_boundary(pops::AmrSystem<Dim>& system, const std::vector<double>& fixed_state,
                          bool primitive) {
  constexpr const char* state_route = "tests.amr.system-contract/boundary/gas-state";
  system.install_block_state_route("gas", state_route);
  std::vector<std::string> face_types(static_cast<std::size_t>(2 * Dim), "foextrap");
  face_types[1] = "dirichlet";
  std::vector<std::string> face_identities;
  std::vector<std::string> representations(static_cast<std::size_t>(2 * Dim), "conservative");
  std::vector<std::string> converters(static_cast<std::size_t>(2 * Dim));
  if (primitive) {
    representations[1] = "primitive";
    converters[1] = "tests.amr.system-contract/boundary/euler-primitive-to-conservative@1";
  }
  for (int face = 0; face < 2 * Dim; ++face)
    face_identities.push_back("tests.amr.system-contract/boundary/face-" + std::to_string(face));
  std::vector<std::string> roles{"density"};
  for (int axis = 0; axis < Dim; ++axis)
    roles.push_back("momentum:" + std::to_string(axis));
  roles.push_back("energy");
  std::vector<double> values(roles.size() * static_cast<std::size_t>(2 * Dim), 0.0);
  for (std::size_t component = 0; component < fixed_state.size(); ++component)
    values[component * static_cast<std::size_t>(2 * Dim) + 1] = fixed_state[component];
  system.install_hyperbolic_boundary("gas", "tests.amr.system-contract/boundary/prepared@1", 1,
                                     face_types, values, face_identities, roles, state_route,
                                     representations, converters);
  pops::add_compiled_model<Dim>(system, "gas", gas_model<Dim>(), "none", "rusanov", "conservative",
                                "euler", 1.4, 1, 1, {}, {}, 0.0,
                                static_cast<double>(pops::kWenoEpsilon), false,
                                "tests.amr.system-contract/boundary/physical-flux");
}

template <int Dim>
void verify_model_qualified_primitive_boundary_conversion() {
  const pops::AmrSystemConfig<Dim> config = single_level_config<Dim>(6);
  std::vector<double> primitive(static_cast<std::size_t>(Dim + 2), 0.0);
  primitive[0] = 2.0;
  for (int axis = 0; axis < Dim; ++axis)
    primitive[static_cast<std::size_t>(axis + 1)] = 0.2 * static_cast<double>(axis + 1);
  primitive.back() = 1.5;
  const auto converter = pops::prepare_compiled_amr_system_block<Dim>(
      "gas", gas_model<Dim>(), "none", "rusanov", "conservative", "euler", 1.4, 1, 1, 0.0,
      static_cast<double>(pops::kWenoEpsilon), false,
      "tests.amr.system-contract/boundary/physical-flux");
  std::vector<double> conservative(primitive.size(), 0.0);
  converter.primitive_to_conservative(primitive.data(), conservative.data());
  EXPECT_DOUBLE_EQ(conservative[0], primitive[0]);
  double kinetic = 0.0;
  for (int axis = 0; axis < Dim; ++axis) {
    EXPECT_DOUBLE_EQ(conservative[static_cast<std::size_t>(axis + 1)],
                     primitive[0] * primitive[static_cast<std::size_t>(axis + 1)]);
    kinetic += 0.5 * primitive[0] * primitive[static_cast<std::size_t>(axis + 1)] *
               primitive[static_cast<std::size_t>(axis + 1)];
  }
  EXPECT_NEAR(conservative.back(), primitive.back() / 0.4 + kinetic, 1.0e-14);

  pops::AmrSystem<Dim> authored(config);
  pops::AmrSystem<Dim> oracle(config);
  install_gas_boundary(authored, primitive, true);
  install_gas_boundary(oracle, conservative, false);
  const std::size_t cells = cell_count(config.shape);
  std::vector<double> initial(static_cast<std::size_t>(Dim + 2) * cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    initial[cell] = 1.0;
    initial[static_cast<std::size_t>(Dim + 1) * cells + cell] = 2.5;
  }
  authored.set_conservative_state("gas", initial);
  oracle.set_conservative_state("gas", initial);
  const pops::runtime::multiblock::BoundaryEvaluationPoint point{
      "tests.amr.system-contract/boundary-conversion", 0, 0, 0, 0, {0, 1}, 0.0, 0.0};
  const auto& authored_evaluation = authored.evaluate_prepared_amr_level(point);
  const auto& oracle_evaluation = oracle.evaluate_prepared_amr_level(point);
  EXPECT_EQ(pops::difference_sum_sq_all(authored_evaluation.residual, oracle_evaluation.residual),
            pops::Real(0));
  pops::MultiFab<Dim> zero(authored_evaluation.residual);
  zero.set_val(pops::Real(0));
  EXPECT_GT(pops::difference_sum_sq_all(authored_evaluation.residual, zero), pops::Real(0));
}

template <int Dim>
void install_direct_tracer(pops::AmrSystem<Dim>& system, const std::string& name,
                           const std::string& consumer_qid) {
  pops::add_compiled_model<Dim>(system, name, tracer_model<Dim>(), "minmod", "rusanov",
                                "conservative", "explicit",
                                static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
                                static_cast<double>(pops::kWenoEpsilon), false, consumer_qid);
}

template <int Dim>
void verify_prepared_installation_parity() {
  const pops::AmrSystemConfig<Dim> config = single_level_config<Dim>();
  constexpr const char* state_route = "tests.amr.system-contract/parity/state";
  constexpr const char* consumer_qid = "tests.amr.system-contract/parity/physical-flux";
  pops::AmrSystem<Dim> direct(config);
  pops::AmrSystem<Dim> prepared(config);
  install_package_lane(direct, "test.amr-system-contract.direct-package");
  install_package_lane(prepared, "test.amr-system-contract.prepared-package");
  direct.install_block_state_route("tracer", state_route);
  prepared.install_block_state_route("tracer", state_route);

  const auto direct_package = pops::prepare_compiled_amr_system_block<Dim>(
      "tracer", tracer_model<Dim>(), "minmod", "rusanov", "conservative", "explicit",
      static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, 0.0,
      static_cast<double>(pops::kWenoEpsilon), false, consumer_qid);
  auto prepared_package = pops::prepare_compiled_amr_system_block<Dim>(
      "tracer", tracer_model<Dim>(), "minmod", "rusanov", "conservative", "explicit",
      static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, 0.0,
      static_cast<double>(pops::kWenoEpsilon), false, consumer_qid);
  const std::string expected_provider_identity =
      "pops.generated.amr.cartesian.nd/" + std::to_string(Dim) + "/minmod/rusanov/conservative";
  ASSERT_FALSE(std::string_view(state_route).empty());
  EXPECT_EQ(direct_package.name, "tracer");
  EXPECT_EQ(direct_package.provider_consumer_qid, consumer_qid);
  EXPECT_EQ(direct_package.provider_identity, expected_provider_identity);
  ASSERT_FALSE(direct_package.collective_contract.empty());
  EXPECT_EQ(prepared_package.name, direct_package.name);
  EXPECT_EQ(prepared_package.provider_consumer_qid, direct_package.provider_consumer_qid);
  EXPECT_EQ(prepared_package.provider_identity, direct_package.provider_identity);
  EXPECT_EQ(prepared_package.collective_contract, direct_package.collective_contract);
  const int expected_ncomp = prepared_package.ncomp;
  const int expected_ghosts = prepared_package.ghosts[0];
  const double expected_gamma = prepared_package.gamma;
  const int expected_substeps = prepared_package.substeps;
  const int expected_stride = prepared_package.stride;
  const std::string expected_time_route = prepared_package.time_route;
  const std::string expected_package_contract = prepared_package.collective_contract;

  install_direct_tracer(direct, "tracer", consumer_qid);
  pops::install_prepared_amr_block(prepared, std::move(prepared_package));

  const std::size_t cells = cell_count(config.shape);
  std::vector<double> initial(cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell)
    initial[cell] = 1.0 + 0.1 * static_cast<double>(cell % 5);
  direct.set_conservative_state("tracer", initial);
  prepared.set_conservative_state("tracer", initial);
  direct.set_program_block_map({0});
  prepared.set_program_block_map({0});

  const pops::MultiFab<Dim>& direct_state = direct.prepared_amr_block_state(0, 0);
  const pops::MultiFab<Dim>& prepared_state = prepared.prepared_amr_block_state(0, 0);
  ASSERT_EQ(direct_state.layout(), prepared_state.layout());
  ASSERT_EQ(direct_state.distribution(), prepared_state.distribution());
  ASSERT_EQ(direct_state.ncomp(), prepared_state.ncomp());
  EXPECT_EQ(pops::difference_sum_sq_all(direct_state, prepared_state), pops::Real(0));
  ASSERT_NE(direct.engine(), nullptr);
  ASSERT_NE(prepared.engine(), nullptr);
  EXPECT_EQ(direct.engine()->spatial_contract(), prepared.engine()->spatial_contract());
  const auto& direct_map = direct.prepared_amr_program_block_map();
  const auto& prepared_map = prepared.prepared_amr_program_block_map();
  EXPECT_EQ(direct_map.canonical_indices, prepared_map.canonical_indices);
  EXPECT_EQ(direct_map.hierarchy_contract, prepared_map.hierarchy_contract);
  EXPECT_EQ(direct_map.exact_contract, prepared_map.exact_contract);
  pops::ExactContractBuilder expected_map_contract;
  expected_map_contract.text("pops.prepared-multiblock-amr.program-map")
      .scalar(std::uint32_t{1})
      .scalar(std::int32_t{Dim})
      .bytes(prepared_map.hierarchy_contract)
      .scalar(std::uint64_t{1})
      .text("tracer")
      .scalar(std::uint64_t{0});
  EXPECT_EQ(prepared_map.exact_contract, std::move(expected_map_contract).release());

  const pops::EffectiveOptionsReport installed = prepared.effective_options_report();
  ASSERT_EQ(installed.blocks.size(), 1U);
  EXPECT_EQ(installed.blocks.front().name, "tracer");
  EXPECT_EQ(installed.blocks.front().ncomp, expected_ncomp);
  EXPECT_EQ(installed.blocks.front().n_ghost, expected_ghosts);
  EXPECT_DOUBLE_EQ(installed.blocks.front().gamma, expected_gamma);
  EXPECT_EQ(installed.blocks.front().substeps, expected_substeps);
  EXPECT_EQ(installed.blocks.front().stride, expected_stride);
  EXPECT_EQ(installed.blocks.front().time, expected_time_route);
  EXPECT_EQ(expected_package_contract, direct_package.collective_contract);

  const auto direct_geometry = direct.prepared_amr_level_geometry(0);
  const auto prepared_geometry = prepared.prepared_amr_level_geometry(0);
  EXPECT_EQ(direct_geometry, prepared_geometry);
  for (int axis = 0; axis < Dim; ++axis) {
    EXPECT_EQ(direct_geometry.domain().length(axis), config.shape[axis]);
    EXPECT_EQ(direct_geometry.lower()[axis], config.lower[axis]);
    EXPECT_EQ(direct_geometry.upper()[axis], config.upper[axis]);
  }

  const pops::runtime::multiblock::BoundaryEvaluationPoint point{
      "tests.amr.system-contract.installation-parity", 0, 0, 0, 0, {0, 1}, 0.0, 0.0};
  const auto& direct_evaluation = direct.evaluate_prepared_amr_level(point);
  const auto& prepared_evaluation = prepared.evaluate_prepared_amr_level(point);
  EXPECT_EQ(direct_evaluation.spatial_contract, direct.engine()->spatial_contract());
  EXPECT_EQ(prepared_evaluation.spatial_contract, prepared.engine()->spatial_contract());
  EXPECT_EQ(pops::difference_sum_sq_all(direct_evaluation.residual, prepared_evaluation.residual),
            pops::Real(0));
  pops::MultiFab<Dim> zero(direct_evaluation.residual);
  zero.set_val(pops::Real(0));
  EXPECT_GT(pops::difference_sum_sq_all(direct_evaluation.residual, zero), pops::Real(0));
}

template <int Dim>
void verify_program_required_before_temporal_mutation() {
  const pops::AmrSystemConfig<Dim> config = single_level_config<Dim>();
  pops::AmrSystem<Dim> system(config);
  system.install_block_state_route("tracer", "tests.amr.system-contract/program/state");
  install_direct_tracer(system, "tracer", "tests.amr.system-contract/program/physical-flux");

  EXPECT_THROW(system.step(0.01), std::logic_error);
  EXPECT_THROW(system.advance(0.01, 0), std::logic_error);
  EXPECT_THROW((void)system.step_cfl(0.4), std::logic_error);
  EXPECT_DOUBLE_EQ(system.time(), 0.0);
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_FALSE(system.has_active_step_transaction());
}

template <int Dim>
std::array<pops::runtime::system::AuxiliaryComponentKey, 3> install_magnetic_provider(
    pops::AmrSystem<Dim>& system, const std::vector<std::string>& consumer_qids) {
  using namespace pops::runtime::system;
  const AuxiliaryComponentContract contract{"cell-average", "cell", "tesla", "tests-magnetic-input",
                                            "scalar"};
  AuxiliaryStorageShape<Dim> shape;
  for (int axis = 0; axis < Dim; ++axis)
    shape.halo[axis] = 2;
  std::array<AuxiliaryComponentKey, 3> keys{
      AuxiliaryComponentKey{"tests.amr.system-contract", "input", "magnetic", "B-x"},
      AuxiliaryComponentKey{"tests.amr.system-contract", "input", "magnetic", "B-y"},
      AuxiliaryComponentKey{"tests.amr.system-contract", "input", "magnetic", "B-z"}};
  std::vector<AuxiliaryOutput<Dim>> outputs;
  for (std::size_t slot = 0; slot < keys.size(); ++slot) {
    outputs.push_back({keys[slot], contract, shape});
  }
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<Dim>{
      "tests.amr.system-contract/magnetic-provider@1",
      AuxiliaryProviderKind::input,
      {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
      std::move(outputs),
      {}});
  for (const std::string& consumer_qid : consumer_qids) {
    AuxiliaryConsumerProviderPlan<Dim> plan;
    plan.consumer_qid = consumer_qid;
    for (std::size_t slot = 0; slot < keys.size(); ++slot)
      plan.values.push_back({{keys[slot], contract, shape}, slot});
    system.install_auxiliary_consumer_plan(std::move(plan));
  }
  system.seal_auxiliary_providers();
  return keys;
}

template <int Dim>
pops::AmrSystemConfig<Dim> magnetic_config() {
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 2;
  config.regrid_every = 0;
  config.explicit_bootstrap = true;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 8;
    config.lower[axis] = pops::Real(0);
    config.upper[axis] = pops::Real(1);
    config.periodicity[axis] = true;
    config.transition_buffers.front()[axis] = 1;
    config.transition_lookaheads.front()[axis] = 1;
  }
  return config;
}

template <int Dim>
void materialize_magnetic_bootstrap(pops::AmrSystem<Dim>& system,
                                    const pops::AmrSystemConfig<Dim>& config,
                                    const std::vector<double>& state) {
  constexpr const char* state_route = "tests.amr.system-contract/magnetic/state";
  system.bind_bootstrap_subject(state_route, "fluid", "bound_level_zero");
  system.stage_bootstrap_array(state_route, "fluid", "cell", "cell", Dim + 2, config.shape, state);
  pops::Extent<Dim> transfer_ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    transfer_ghosts[axis] = 1;
  system.register_bootstrap_transfer_route(
      "tests.amr.system-contract/magnetic/bootstrap/prolongation", {state_route},
      "tests.amr.system-contract/magnetic/bootstrap/conservative-linear@1", "cell", "cell",
      "conservative", "dense", "prolongation", "conservative_linear", 2, transfer_ghosts,
      config.transition_ratios.front());
  system.begin_bootstrap_plan();
  (void)system.materialize_bootstrap_action(state_route, "initialize_level_zero",
                                            "bound_level_zero", 0);
  if (!system.bootstrap_next_level()) {
    system.rollback_bootstrap_level();
    throw std::runtime_error("magnetic Program proof did not create its refined level");
  }
  (void)system.materialize_bootstrap_action(state_route, "prolong_from_parent",
                                            "conservative_linear", 1);
  system.commit_bootstrap_level();
}

template <int Dim>
std::vector<std::vector<double>> run_magnetic_source(pops::Real bz) {
  static_assert(Dim == 1 || Dim == 2);
  using namespace pops::runtime::system;
  constexpr const char* consumer_qid = "tests.amr.system-contract/magnetic/physical-source";
  const pops::AmrSystemConfig<Dim> config = magnetic_config<Dim>();
  pops::AmrSystem<Dim> system(config);
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  const auto keys = install_magnetic_provider(system, {consumer_qid});
  system.install_block_state_route("fluid", "tests.amr.system-contract/magnetic/state");
  MagneticModel<Dim> model{{},
                           pops::EulerND<Dim>{pops::Real(1.4)},
                           pops::MagneticLorentzForceND<Dim>{pops::Real(1)},
                           pops::NoElliptic{}};
  pops::add_compiled_model<Dim>(system, "fluid", model, "none", "rusanov", "conservative", "euler",
                                static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
                                static_cast<double>(pops::kWenoEpsilon), false, consumer_qid);

  const std::size_t cells = cell_count(config.shape);
  std::vector<double> state(static_cast<std::size_t>(Dim + 2) * cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const int first_axis = static_cast<int>(cell % static_cast<std::size_t>(config.shape[0]));
    const double density = first_axis >= 3 && first_axis <= 5 ? 2.0 : 1.0;
    state[cell] = density;
    state[cells + cell] = density;
    state[static_cast<std::size_t>(Dim + 1) * cells + cell] = 3.0 * density;
  }
  pops::test::install_prepared_threshold_union(
      system,
      {{"fluid", "rho", 1.5, pops::test::PreparedThresholdRelation::Above,
        "tests.amr.system-contract/magnetic/state"}},
      "tests.amr.system-contract/magnetic/tagging@1");
  materialize_magnetic_bootstrap(system, config, state);
  EXPECT_EQ(system.n_levels(), 2);
  const std::vector<double> zero(cells, 0.0);
  system.stage_auxiliary_input(keys[0], zero);
  system.stage_auxiliary_input(keys[1], zero);
  system.stage_auxiliary_input(keys[2], std::vector<double>(cells, static_cast<double>(bz)));
  system.refresh_auxiliary({"tests.amr.system-contract/magnetic-clock", 0, 0, 0, 0, 0, 0,
                            AuxiliaryEvaluationEvent::initialization});
  pops::test::install_forward_euler_program(system);
  system.advance(0.01, 1);

  const auto bz_values = system.auxiliary_component(keys[2]);
  EXPECT_EQ(bz_values.size(), cells);
  for (const double value : bz_values)
    EXPECT_EQ(value, static_cast<double>(bz));
  std::vector<std::vector<double>> result;
  for (int level = 0; level < system.n_levels(); ++level) {
    const pops::MultiFab<Dim>& accepted = system.prepared_amr_block_state(0, level);
    std::int64_t level_cells = 0;
    for (const pops::Box<Dim>& box : accepted.layout().boxes())
      level_cells += box.numPts();
    result.emplace_back();
    for (int component = 0; component < accepted.ncomp(); ++component)
      result.back().push_back(static_cast<double>(pops::reduce_sum(accepted, component)) /
                              static_cast<double>(level_cells));
  }
  return result;
}

struct MultiblockRegridObservation {
  std::array<double, 2> mass_before{};
  std::array<double, 2> mass_after{};
  std::array<double, 2> transverse_momentum{};
  std::uint64_t topology_before = 0;
  std::uint64_t topology_after = 0;
  int levels = 0;
  int patches = 0;
};

template <int Dim>
std::vector<pops::MultiFab<Dim>> prepare_exact_composite_masks(
    pops::AmrSystem<Dim>& system, const std::vector<pops::Extent<Dim>>& transition_ratios) {
  const int level_count = system.n_levels();
  if (transition_ratios.size() + 1 < static_cast<std::size_t>(level_count))
    throw std::invalid_argument("composite proof lacks one exact ratio per live transition");
  std::vector<pops::MultiFab<Dim>> masks;
  masks.reserve(static_cast<std::size_t>(level_count));
  for (int level = 0; level < level_count; ++level) {
    const pops::MultiFab<Dim>& carrier = system.prepared_amr_block_state(0, level);
    masks.emplace_back(carrier.layout(), carrier.distribution(), carrier.local_rank(), 1,
                       pops::Extent<Dim>{});
    masks.back().set_val(pops::Real(1));
  }

  for (int level = 0; level + 1 < level_count; ++level) {
    pops::MultiFab<Dim>& parent_mask = masks[static_cast<std::size_t>(level)];
    const pops::MultiFab<Dim>& child = system.prepared_amr_block_state(0, level + 1);
    const pops::Extent<Dim>& ratio = transition_ratios[static_cast<std::size_t>(level)];
    for (const pops::Box<Dim>& child_patch : child.layout().boxes()) {
      const pops::Box<Dim> covered = pops::coarsen(child_patch, ratio);
      for (std::size_t local = 0; local < parent_mask.local_size(); ++local) {
        const pops::Box<Dim> overlap = parent_mask.box(local).intersect(covered);
        if (overlap.empty())
          continue;
        const pops::FieldView<pops::Real, Dim> active = parent_mask.fab(local).view();
        pops::for_each_cell(overlap,
                            [=] POPS_HD(const pops::Index<Dim>& cell) { active(cell, 0) = 0; });
      }
    }
  }
#if defined(POPS_HAS_KOKKOS)
  Kokkos::fence();
#endif
  return masks;
}

template <int Dim>
pops::runtime::amr::CompositeReductionResult exact_block_composite_sum(
    pops::AmrSystem<Dim>& system, int runtime_block, int component,
    const std::vector<pops::MultiFab<Dim>>& masks) {
  if (masks.size() != static_cast<std::size_t>(system.n_levels()))
    throw std::invalid_argument("composite proof masks differ from the live hierarchy depth");
  std::vector<
      pops::runtime::amr::CompositeLevelView<Dim, typename pops::MultiFab<Dim>::memory_space>>
      levels;
  levels.reserve(masks.size());
  for (std::size_t level = 0; level < masks.size(); ++level) {
    const pops::Geometry<Dim> geometry =
        system.prepared_amr_level_geometry(static_cast<int>(level));
    std::array<pops::Real, Dim> cell_extent{};
    for (int axis = 0; axis < Dim; ++axis)
      cell_extent[static_cast<std::size_t>(axis)] = geometry.spacing(axis);
    levels.push_back({&system.prepared_amr_block_state(runtime_block, static_cast<int>(level)),
                      &masks[level], cell_extent, nullptr});
  }
  return pops::runtime::amr::composite_reduce<Dim, typename pops::MultiFab<Dim>::memory_space>(
      levels, component, pops::runtime::amr::CompositeReductionKind::Sum,
      pops::ExecutionLane::world());
}

template <int Dim>
std::vector<double> explicit_block_state_oracle(pops::AmrSystem<Dim>& system, int runtime_block,
                                                int level, int components) {
  const pops::MultiFab<Dim>& carrier = system.prepared_amr_block_state(runtime_block, level);
  return pops::runtime::system::marshaling::gather_global(
      carrier, system.prepared_amr_level_geometry(level).domain(), components);
}

template <int Dim>
void expect_output_pieces_equal(const std::vector<pops::OutputPiece<Dim>>& actual,
                                const std::vector<pops::OutputPiece<Dim>>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t piece = 0; piece < expected.size(); ++piece) {
    EXPECT_EQ(actual[piece].level, expected[piece].level);
    EXPECT_EQ(actual[piece].box, expected[piece].box);
    EXPECT_EQ(actual[piece].global_box_index, expected[piece].global_box_index);
    EXPECT_EQ(actual[piece].owner_rank, expected[piece].owner_rank);
    EXPECT_EQ(actual[piece].replicated, expected[piece].replicated);
    EXPECT_EQ(actual[piece].ncomp, expected[piece].ncomp);
    EXPECT_EQ(actual[piece].values, expected[piece].values);
  }
}

template <int Dim>
MultiblockRegridObservation run_two_block_regrid_with_bz(pops::Real bz) {
  static_assert(Dim == 2);
  constexpr std::array<const char*, 2> names{"ions", "electrons"};
  constexpr std::array<const char*, 2> state_routes{
      "tests.amr.system-contract/two-block/ions-state",
      "tests.amr.system-contract/two-block/electrons-state"};
  constexpr std::array<const char*, 2> consumer_qids{
      "tests.amr.system-contract/two-block/ions-source",
      "tests.amr.system-contract/two-block/electrons-source"};

  pops::AmrSystemConfig<Dim> config = magnetic_config<Dim>();
  config.regrid_every = 1;
  config.explicit_bootstrap = false;
  pops::AmrSystem<Dim> system(config);
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  const auto keys = install_magnetic_provider(
      system, {std::string(consumer_qids[0]), std::string(consumer_qids[1])});
  for (std::size_t block = 0; block < names.size(); ++block)
    system.install_block_state_route(names[block], state_routes[block]);

  for (std::size_t block = 0; block < names.size(); ++block) {
    MagneticModel<Dim> model{{},
                             pops::EulerND<Dim>{pops::Real(1.4)},
                             pops::MagneticLorentzForceND<Dim>{pops::Real(1)},
                             pops::NoElliptic{}};
    pops::add_compiled_model<Dim>(system, names[block], model, "none", "rusanov", "conservative",
                                  "euler", static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1,
                                  {}, {}, 0.0, static_cast<double>(pops::kWenoEpsilon), false,
                                  consumer_qids[block]);
  }

  const std::size_t cells = cell_count(config.shape);
  for (std::size_t block = 0; block < names.size(); ++block) {
    std::vector<double> state(4 * cells, 0.0);
    for (std::size_t cell = 0; cell < cells; ++cell) {
      const int x = static_cast<int>(cell % static_cast<std::size_t>(config.shape[0]));
      const bool tagged = block == 0 ? (x >= 1 && x <= 3) : (x >= 4 && x <= 6);
      const double density = tagged ? (block == 0 ? 2.0 : 2.5) : 1.0;
      const double longitudinal_velocity = block == 0 ? 1.0 : 0.5;
      state[cell] = density;
      state[cells + cell] = longitudinal_velocity * density;
      state[3 * cells + cell] = 3.0 * density;
    }
    system.set_conservative_state(names[block], state);
  }
  pops::test::install_prepared_threshold_union(
      system,
      {{names[0], "rho", 1.5, pops::test::PreparedThresholdRelation::Above, state_routes[0]},
       {names[1], "rho", 1.5, pops::test::PreparedThresholdRelation::Above, state_routes[1]}},
      "tests.amr.system-contract/two-block/tagging@1");

  const std::vector<double> zero(cells, 0.0);
  system.stage_auxiliary_input(keys[0], zero);
  system.stage_auxiliary_input(keys[1], zero);
  system.stage_auxiliary_input(keys[2], std::vector<double>(cells, static_cast<double>(bz)));
  system.refresh_auxiliary({"tests.amr.system-contract/two-block/magnetic-clock", 0, 0, 0, 0, 0, 0,
                            pops::runtime::system::AuxiliaryEvaluationEvent::initialization});

  if (system.n_blocks() != 2)
    throw std::runtime_error("two-block facade did not materialize both compiled blocks");
  EXPECT_EQ(system.block_names(),
            (std::vector<std::string>{std::string(names[0]), std::string(names[1])}));
  if (system.n_levels() != 2)
    throw std::runtime_error("two-block facade did not materialize its tagged fine level");
  MultiblockRegridObservation result;
  result.levels = system.n_levels();
  result.patches = system.n_patches();
  result.topology_before = system.engine()->topology_epoch();
  EXPECT_NE(&system.prepared_amr_block_state(0, 0), &system.prepared_amr_block_state(1, 0));
  EXPECT_GT(pops::difference_sum_sq_all(system.prepared_amr_block_state(0, 0),
                                        system.prepared_amr_block_state(1, 0)),
            pops::Real(0));
  std::array<std::vector<std::vector<double>>, 2> block_oracles;
  for (std::size_t block = 0; block < names.size(); ++block) {
    block_oracles[block].reserve(static_cast<std::size_t>(system.n_levels()));
    for (int level = 0; level < system.n_levels(); ++level) {
      const pops::MultiFab<Dim>& carrier =
          system.prepared_amr_block_state(static_cast<int>(block), level);
      block_oracles[block].push_back(
          explicit_block_state_oracle(system, static_cast<int>(block), level, carrier.ncomp()));
      EXPECT_EQ(system.block_level_state(names[block], level), block_oracles[block].back());
      EXPECT_EQ(system.block_level_state_global(names[block], level), block_oracles[block].back());
      expect_output_pieces_equal(
          system.output_state_local_pieces(names[block], level),
          pops::output_local_pieces(carrier, level, carrier.distribution().replicated()));
    }
    EXPECT_EQ(system.density(names[block]),
              explicit_block_state_oracle(system, static_cast<int>(block), 0, 1));
  }
  EXPECT_NE(block_oracles[0][0], block_oracles[1][0]);

  const auto masks_before = prepare_exact_composite_masks(system, config.transition_ratios);
  for (std::size_t block = 0; block < names.size(); ++block) {
    const double exact = static_cast<double>(
        exact_block_composite_sum(system, static_cast<int>(block), 0, masks_before).value);
    EXPECT_NEAR(system.composite_reduce(names[block], "sum", 0), exact, 2.0e-12);
    result.mass_before[block] = exact;
  }

  std::vector<double> second_trial = block_oracles[1][0];
  for (double& value : second_trial)
    value += 0.125;
  system.begin_step_transaction();
  system.set_block_level_state(names[1], 0, second_trial);
  EXPECT_EQ(system.block_level_state_global(names[0], 0), block_oracles[0][0]);
  EXPECT_EQ(system.block_level_state_global(names[1], 0), second_trial);
  system.rollback_step_transaction();
  EXPECT_EQ(system.block_level_state_global(names[0], 0), block_oracles[0][0]);
  EXPECT_EQ(system.block_level_state_global(names[1], 0), block_oracles[1][0]);

  pops::test::install_forward_euler_program(system);
  system.step(1.0e-4);

  result.topology_after = system.engine()->topology_epoch();
  result.levels = system.n_levels();
  result.patches = system.n_patches();
  const auto masks_after = prepare_exact_composite_masks(system, config.transition_ratios);
  for (std::size_t block = 0; block < names.size(); ++block) {
    const auto mass = exact_block_composite_sum(system, static_cast<int>(block), 0, masks_after);
    const auto transverse =
        exact_block_composite_sum(system, static_cast<int>(block), 2, masks_after);
    result.mass_after[block] = static_cast<double>(mass.value);
    result.transverse_momentum[block] =
        static_cast<double>(transverse.value / transverse.active_measure);
  }
  return result;
}

template <int Dim>
void verify_stride_window_contract() {
  const pops::AmrSystemConfig<Dim> config = single_level_config<Dim>(4);
  pops::AmrSystem<Dim> system(config);
  system.install_block_state_route("tracer", "tests.amr.system-contract/cadence/state");
  install_direct_tracer(system, "tracer", "tests.amr.system-contract/cadence/physical-flux");
  std::vector<double> times;
  std::vector<double> steps;
  std::vector<int> macro_steps;
  const std::string balance_route = "pops.balance-ledger-route.v1:sha256:" + std::string(64, '9');
  const std::array<std::pair<const char*, double>, 5> balance_records{{
      {"storage_change", 1.0},
      {"outward_boundary_flux", 2.0},
      {"sources", 3.0},
      {"reflux", 4.0},
      {"projection", 5.0},
  }};
  const auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("tests.amr.system-contract/cadence/macro");
  context->install([&, context](double step) {
    times.push_back(system.time());
    steps.push_back(step);
    macro_steps.push_back(system.macro_step());
    for (const auto& [term, value] : balance_records)
      context->record_balance_term(balance_route, term, static_cast<pops::Real>(value));
  });
  system.set_program_block_map({0});
  system.set_program_cadence(3, 2);

  const auto expect_balance = [&](const std::map<std::string, double>& balance, double scale) {
    const std::vector<std::string> expected_order{"outward_boundary_flux", "projection", "reflux",
                                                  "sources", "storage_change"};
    std::vector<std::string> actual_order;
    for (const auto& [term, value] : balance) {
      (void)value;
      actual_order.push_back(term);
    }
    EXPECT_EQ(actual_order, expected_order);
    for (const auto& [term, value] : balance_records)
      EXPECT_DOUBLE_EQ(balance.at(term), scale * value);
  };
  const auto step_and_read_balance = [&](double dt) {
    system.begin_step_transaction();
    system.step(dt);
    const std::map<std::string, double> balance = system.accepted_balance_terms(balance_route);
    system.commit_step_transaction();
    system.finalize_step_transaction();
    return balance;
  };

  system.begin_step_projection_report();
  const auto held_balance = step_and_read_balance(0.1);
  expect_balance(held_balance, 0.0);
  EXPECT_TRUE(times.empty());
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.1);
  EXPECT_EQ(system.program_cadence_window_steps(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_start_time(), 0.0);

  system.begin_step_transaction();
  system.step(0.2);
  const auto rejected_due_balance = system.accepted_balance_terms(balance_route);
  expect_balance(rejected_due_balance, 3.0);
  system.rollback_step_transaction();
  EXPECT_DOUBLE_EQ(system.time(), 0.1);
  EXPECT_EQ(system.macro_step(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.1);
  EXPECT_EQ(system.program_cadence_window_steps(), 1);
  system.begin_step_transaction();
  const auto restored_held_balance = system.accepted_balance_terms(balance_route);
  expect_balance(restored_held_balance, 0.0);
  system.rollback_step_transaction();

  times.clear();
  steps.clear();
  macro_steps.clear();
  const auto retried_due_balance = step_and_read_balance(0.2);
  EXPECT_EQ(retried_due_balance, rejected_due_balance);
  expect_balance(retried_due_balance, 3.0);
  ASSERT_EQ(times.size(), 3U);
  EXPECT_NEAR(times[0], 0.0, 1.0e-14);
  EXPECT_NEAR(times[1], 0.1, 1.0e-14);
  EXPECT_NEAR(times[2], 0.2, 1.0e-14);
  for (const double step : steps)
    EXPECT_NEAR(step, 0.1, 1.0e-14);
  EXPECT_NEAR(std::accumulate(steps.begin(), steps.end(), 0.0), 0.3, 1.0e-14);
  EXPECT_DOUBLE_EQ(times.back() + steps.back(), 0.3);
  EXPECT_EQ(macro_steps, (std::vector<int>{0, 0, 0}));
  EXPECT_NEAR(system.time(), 0.3, 1.0e-14);
  EXPECT_EQ(system.macro_step(), 2);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.0);
  EXPECT_EQ(system.program_cadence_window_steps(), 0);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_start_time(), 0.0);

  system.begin_step_projection_report();
  const auto reset_held_balance = step_and_read_balance(0.05);
  expect_balance(reset_held_balance, 0.0);
  EXPECT_NEAR(system.time(), 0.35, 1.0e-14);
  EXPECT_EQ(system.macro_step(), 3);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.05);
  EXPECT_EQ(system.program_cadence_window_steps(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_start_time(), 0.3);
}

}  // namespace

TEST(test_amr_system_contract, RefusesMappedPeriodicityBeforeRankedFillPatchConstruction) {
  auto boundary = pops::prepare_hyperbolic_boundary<2>(
      {"periodic", "foextrap", "foextrap", "periodic"}, std::vector<double>(4, 0.0),
      {"case::block::tracer::xlo", "case::block::tracer::xhi", "case::block::tracer::ylo",
       "case::block::tracer::yhi"},
      {"Scalar"}, true);
  EXPECT_THROW((void)boundary.periodic_axes(), std::logic_error);
}

TEST(test_amr_system_contract, DirectAndPreparedCompiledInstallationsHaveExactRankedParity) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  verify_prepared_installation_parity<pops::kNativeDimension>();
}

TEST(test_amr_system_contract, RectangularGeometryPreservesIndependentAxisPeriodicity) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  verify_rectangular_geometry_and_independent_periodicity<pops::kNativeDimension>();
}

TEST(test_amr_system_contract, PrimitiveBoundaryMatchesQualifiedConservativeOracle) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  verify_model_qualified_primitive_boundary_conversion<pops::kNativeDimension>();
}

TEST(test_amr_system_contract, TwoBlockFacadeRegridsConservativelyAndSharesPreparedBz) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
#if POPS_NATIVE_DIM == 2
  const MultiblockRegridObservation control = run_two_block_regrid_with_bz<2>(pops::Real(0));
  const MultiblockRegridObservation forced = run_two_block_regrid_with_bz<2>(pops::Real(2));
  ASSERT_EQ(control.levels, 2);
  ASSERT_EQ(forced.levels, 2);
  EXPECT_GT(control.patches, 0);
  EXPECT_GT(forced.patches, 0);
  EXPECT_GT(control.topology_after, control.topology_before);
  EXPECT_GT(forced.topology_after, forced.topology_before);
  EXPECT_GT(std::abs(control.mass_before[0] - control.mass_before[1]), 1.0e-5);
  EXPECT_GT(std::abs(forced.mass_before[0] - forced.mass_before[1]), 1.0e-5);
  for (std::size_t block = 0; block < control.mass_before.size(); ++block) {
    EXPECT_NEAR(control.mass_after[block], control.mass_before[block], 2.0e-12);
    EXPECT_NEAR(forced.mass_after[block], forced.mass_before[block], 2.0e-12);
    EXPECT_LT(forced.transverse_momentum[block], control.transverse_momentum[block] - 1.0e-5)
        << "the shared prepared Bz provider must drive both facade blocks";
  }
  const double ion_effect = forced.transverse_momentum[0] - control.transverse_momentum[0];
  const double electron_effect = forced.transverse_momentum[1] - control.transverse_momentum[1];
  EXPECT_LT(ion_effect, -1.0e-5);
  EXPECT_LT(electron_effect, -1.0e-5);
  EXPECT_GT(std::abs(ion_effect - electron_effect), 1.0e-5)
      << "the two exact runtime-block carriers must retain their distinct longitudinal momenta";
#else
  GTEST_SKIP() << "the transverse two-block Bz trajectory is an explicit Dim=2 proof";
#endif
}

TEST(test_amr_system_contract, TemporalFacadeRequiresInstalledWholeSystemProgram) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  verify_program_required_before_temporal_mutation<pops::kNativeDimension>();
}

TEST(test_amr_system_contract, PreparedBzRotatesTransverseMomentumInDim2) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
#if POPS_NATIVE_DIM == 2
  const auto without_field = run_magnetic_source<2>(pops::Real(0));
  const auto with_field = run_magnetic_source<2>(pops::Real(2));
  ASSERT_EQ(without_field.size(), 2U);
  ASSERT_EQ(with_field.size(), 2U);
  for (std::size_t level = 0; level < with_field.size(); ++level) {
    ASSERT_EQ(without_field[level].size(), 4U);
    ASSERT_EQ(with_field[level].size(), 4U);
    EXPECT_LT(with_field[level][2], without_field[level][2] - 1.0e-3)
        << "positive Bz must rotate positive x momentum toward negative y momentum on level "
        << level;
  }
#else
  GTEST_SKIP() << "the transverse refined Bz trajectory is an explicit Dim=2 proof";
#endif
}

TEST(test_amr_system_contract, PreparedBzHasZeroLongitudinalCrossProductInDim1) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
#if POPS_NATIVE_DIM == 1
  const auto without_field = run_magnetic_source<1>(pops::Real(0));
  const auto with_field = run_magnetic_source<1>(pops::Real(2));
  ASSERT_EQ(without_field.size(), 2U);
  ASSERT_EQ(with_field.size(), 2U);
  for (std::size_t level = 0; level < with_field.size(); ++level) {
    ASSERT_EQ(without_field[level].size(), 3U);
    ASSERT_EQ(with_field[level].size(), 3U);
    for (std::size_t component = 0; component < with_field[level].size(); ++component)
      EXPECT_NEAR(with_field[level][component], without_field[level][component], 1.0e-13)
          << "the 1D longitudinal projection of momentum cross B must vanish on level " << level;
  }
#else
  GTEST_SKIP() << "the longitudinal zero-cross-product trajectory is an explicit Dim=1 proof";
#endif
}

TEST(test_amr_system_contract, VariableDtStrideUsesOneExactPublicWindow) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  verify_stride_window_contract<pops::kNativeDimension>();
}

TEST(test_amr_system_contract, CadenceRestoreRejectsClockDriftWithoutMutation) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystem<Dim> system(single_level_config<Dim>(4));
  system.set_program_cadence(1, 2);
  system.restore_program_cadence_window(0.1, 1, 0.0, 0.075, 0.1, 1);
  EXPECT_THROW(system.set_clock(std::nextafter(0.1, 1.0), 1), std::runtime_error);
  EXPECT_DOUBLE_EQ(system.time(), 0.0);
  EXPECT_EQ(system.macro_step(), 0);
  system.restore_program_cadence_window(0.1, 1, 0.0, 0.075, 0.1, 1);
  system.set_clock(0.1, 1);
  EXPECT_DOUBLE_EQ(system.time(), 0.1);
  EXPECT_EQ(system.macro_step(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.1);
  EXPECT_EQ(system.program_cadence_window_steps(), 1);
  EXPECT_DOUBLE_EQ(system.program_last_dt(), 0.075);
}

TEST(test_amr_system_contract, AcceptedClockSerializationPreservesNonAssociativeEndpoint) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystem<Dim> system(single_level_config<Dim>(4));
  system.install_block_state_route("tracer", "tests.amr.system-contract/clock/state");
  install_direct_tracer(system, "tracer", "tests.amr.system-contract/clock/physical-flux");
  system.set_clock(0.1, 0);
  pops::test::install_forward_euler_program(system);
  system.set_program_cadence(3, 3);

  const double accepted_endpoint = ((0.1 + 0.1) + 0.1) + 0.3;
  const double reconstructed_endpoint = 0.1 + ((0.1 + 0.1) + 0.3);
  ASSERT_NE(std::bit_cast<std::uint64_t>(accepted_endpoint),
            std::bit_cast<std::uint64_t>(reconstructed_endpoint));
  system.step(0.1);
  system.step(0.1);
  system.step(0.3);
  EXPECT_EQ(std::bit_cast<std::uint64_t>(system.time()),
            std::bit_cast<std::uint64_t>(accepted_endpoint));

  pops::runtime::program::AmrProgramAcceptedState<Dim> accepted;
  accepted.spatial_contract = "tests.amr.system-contract/non-associative-clock";
  accepted.level_clocks = {{0, system.macro_step(), pops::amr::Rational(0, 1), system.time()}};
  const auto encoded = pops::runtime::program::serialize_amr_program_accepted_state(accepted);
  const auto decoded = pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(encoded);
  ASSERT_EQ(decoded.level_clocks.size(), 1U);
  EXPECT_EQ(std::bit_cast<std::uint64_t>(decoded.level_clocks.front().physical_time),
            std::bit_cast<std::uint64_t>(system.time()));
  EXPECT_EQ(pops::runtime::program::serialize_amr_program_accepted_state(decoded), encoded);
}
