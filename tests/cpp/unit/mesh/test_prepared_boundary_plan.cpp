#include <gtest/gtest.h>

#include <pops/mesh/boundary/prepared_boundary_plan.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/runtime/context/grid_context.hpp>

#include <type_traits>
#include <vector>

using namespace pops;

namespace {

MultiFab scalar_field(const Box2D& domain, int ncomp = 1, int ngrow = 0) {
  const BoxArray boxes = BoxArray::from_domain(domain, domain.nx());
  return MultiFab(boxes, DistributionMapping(boxes.size(), n_ranks()), ncomp, ngrow);
}

BCRec physical_bc() {
  BCRec bc;
  bc.xlo = BCType::Foextrap;
  bc.xhi = BCType::Dirichlet;
  bc.xhi_val = Real(4);
  bc.ylo = BCType::Foextrap;
  bc.yhi = BCType::Foextrap;
  return bc;
}

PreparedBoundaryComponentSpec linearization_spec(bool jvp, std::string target, std::string output) {
  PreparedBoundaryComponentSpec spec;
  spec.target_identity = std::move(target);
  spec.component_id = "pops://test/field-boundary@1";
  spec.manifest_identity = "component-manifest:test-field-boundary";
  spec.interface_version = 1;
  spec.producer_identity = "case::boundary::producer";
  spec.state_identity = "case::block::state";
  spec.ghost_identity = "case::boundary::left-face";
  spec.layout_identity = "case::layout::cells";
  spec.region.kind = POPS_BOUNDARY_FACE_V1;
  spec.region.dimension = 2;
  spec.region.codimension = 1;
  spec.region.axes = {0};
  spec.region.sides = {-1};
  spec.region.identity = "case::boundary::left-face";
  spec.states = {spec.state_identity};
  spec.directions =
      jvp ? std::vector<std::string>{spec.state_identity} : std::vector<std::string>{};
  spec.fields = {"case::field::frozen"};
  spec.parameter_ids = {"case::param::coefficient"};
  spec.parameter_values = {2.5};
  spec.outputs = {std::move(output)};
  spec.nonlinear_iterate = spec.state_identity;
  spec.parameters_json = "{\"case::param::coefficient\":2.5}";
  return spec;
}

}  // namespace

TEST(test_prepared_boundary_plan, executes_same_level_and_component_physical_producers) {
  const Box2D domain = Box2D::from_extents(4, 4);
  MultiFab state = scalar_field(domain, 2, 1);
  for (int local = 0; local < state.local_size(); ++local) {
    Array4 values = state.fab(local).array();
    for_each_cell(state.box(local), [=](int i, int j) {
      values(i, j, 0) = Real(1);
      values(i, j, 1) = Real(2);
    });
  }
  BCRec first = physical_bc();
  BCRec second = physical_bc();
  second.xhi_val = Real(9);
  PreparedBoundaryPlan plan("case::block::ghost-plan", 1, {first, second});

  plan.fill_same_level_and_physical(state, domain);

  const Fab2D& field = state.fab(0);
  EXPECT_EQ(field(-1, 2, 0), Real(1));
  EXPECT_EQ(field(-1, 2, 1), Real(2));
  EXPECT_EQ(field(4, 2, 0), Real(7));   // 2*4 - interior(1)
  EXPECT_EQ(field(4, 2, 1), Real(16));  // 2*9 - interior(2)
}

TEST(test_prepared_boundary_plan, materializes_move_only_lane_session_before_execution) {
  static_assert(!std::is_copy_constructible_v<PreparedBoundaryPlan::Session>);
  static_assert(!std::is_copy_assignable_v<PreparedBoundaryPlan::Session>);
  static_assert(std::is_nothrow_move_constructible_v<PreparedBoundaryPlan::Session>);

  const Box2D domain = Box2D::from_extents(4, 4);
  MultiFab state = scalar_field(domain, 1, 1);
  for (int local = 0; local < state.local_size(); ++local) {
    Array4 values = state.fab(local).array();
    for_each_cell(state.box(local), [=](int i, int j) { values(i, j, 0) = Real(3); });
  }
  PreparedBoundaryPlan plan("case::block::session-plan", 1, {physical_bc()});
  const auto lane = ExecutionLane::world("case::block::session-lane");
  auto original = plan.make_session(lane);
  auto session = std::move(original);

  EXPECT_THROW(original.fill_same_level_and_physical(state, domain), std::logic_error);
  EXPECT_NO_THROW(session.fill_same_level_and_physical(state, domain));
  EXPECT_EQ(state.fab(0)(-1, 2, 0), Real(3));
  EXPECT_EQ(state.fab(0)(4, 2, 0), Real(5));
}

TEST(test_prepared_boundary_plan, rejects_incomplete_periodic_pairs_and_insufficient_ghosts) {
  BCRec mixed = physical_bc();
  mixed.xlo = BCType::Periodic;
  EXPECT_THROW(PreparedBoundaryPlan("case::bad-periodic::ghost-plan", 1, {mixed}),
               std::runtime_error);

  const Box2D domain = Box2D::from_extents(2, 2);
  MultiFab state = scalar_field(domain, 1, 1);
  PreparedBoundaryPlan deep("case::deep::ghost-plan", 2, {physical_bc()});
  EXPECT_THROW(deep.fill_same_level_and_physical(state, domain), std::runtime_error);
}

TEST(test_prepared_boundary_plan, grid_context_routes_exact_nary_storage_registry) {
  const Box2D domain = Box2D::from_extents(3, 3);
  MultiFab primary = scalar_field(domain, 1, 1);
  MultiFab coupled = scalar_field(domain, 2, 1);
  MultiFab auxiliary = scalar_field(domain, 3, 1);
  MultiFab output = scalar_field(domain, 1, 0);
  auto plan = std::make_shared<PreparedBoundaryPlan>("case::nary::ghost-plan", 1,
                                                     std::vector<BCRec>{physical_bc()});
  GridContext context;
  context.dom = domain;
  context.geom = Geometry(domain, Real(0), Real(1), Real(0), Real(1));
  context.boundary_plan = plan;
  int registry_calls = 0;
  context.boundary_field_registry = [&](const auto&, MultiFab& state, const MultiFab* direction,
                                        MultiFab* destination,
                                        detail::BoundaryFieldRegistry& fields) {
    ++registry_calls;
    EXPECT_EQ(&state, &primary);
    EXPECT_EQ(direction, nullptr);
    EXPECT_EQ(destination, nullptr);
    fields.bind_state("case::state::primary", primary);
    fields.bind_state("case::state::coupled", coupled);
    fields.bind_field("case::field::auxiliary", auxiliary);
    fields.bind_output("case::output::residual", output);
    EXPECT_EQ(&fields.state("case::state::coupled"), &coupled);
    EXPECT_EQ(&fields.field("case::field::auxiliary"), &auxiliary);
    EXPECT_EQ(&fields.output("case::output::residual"), &output);
  };
  const runtime::multiblock::BoundaryEvaluationPoint point{"clock.nary",        0,   0,  0, 0,
                                                           amr::Rational(0, 1), 0.1, 0.0};

  fill_grid_ghosts(primary, context, point);

  EXPECT_EQ(registry_calls, 1);
}

TEST(test_prepared_boundary_plan, authenticates_one_to_one_residual_jvp_contracts) {
  const auto residual =
      linearization_spec(false, "case::boundary::residual", "case::boundary::residual-output");
  const auto jvp = linearization_spec(true, "case::boundary::jvp", "case::boundary::jvp-output");
  EXPECT_NO_THROW(PreparedBoundaryPlan::validate_linearization_bijection({residual}, {jvp}));

  auto changed_component = jvp;
  changed_component.component_id = "pops://test/other-boundary@1";
  EXPECT_THROW(
      PreparedBoundaryPlan::validate_linearization_bijection({residual}, {changed_component}),
      std::runtime_error);
  auto changed_manifest = jvp;
  changed_manifest.manifest_identity = "component-manifest:other";
  EXPECT_THROW(
      PreparedBoundaryPlan::validate_linearization_bijection({residual}, {changed_manifest}),
      std::runtime_error);
  auto changed_parameters = jvp;
  changed_parameters.parameter_values = {3.0};
  EXPECT_THROW(
      PreparedBoundaryPlan::validate_linearization_bijection({residual}, {changed_parameters}),
      std::runtime_error);
  auto changed_target_parameters = jvp;
  changed_target_parameters.target_json = "{\"target\":\"other\"}";
  EXPECT_THROW(PreparedBoundaryPlan::validate_linearization_bijection({residual},
                                                                      {changed_target_parameters}),
               std::runtime_error);
}

TEST(test_prepared_boundary_plan, rejects_duplicate_or_orphan_residual_jvp_endpoints) {
  const auto residual =
      linearization_spec(false, "case::boundary::residual", "case::boundary::residual-output");
  const auto jvp = linearization_spec(true, "case::boundary::jvp", "case::boundary::jvp-output");

  auto duplicate_residual = residual;
  duplicate_residual.target_identity = "case::boundary::residual-duplicate";
  duplicate_residual.outputs = {"case::boundary::residual-output-duplicate"};
  auto orphan_jvp = jvp;
  orphan_jvp.producer_identity = "case::boundary::other-producer";
  EXPECT_THROW(PreparedBoundaryPlan::validate_linearization_bijection(
                   {residual, duplicate_residual}, {jvp, orphan_jvp}),
               std::runtime_error)
      << "one JVP cannot be consumed by two residual endpoints";

  auto duplicate_jvp = jvp;
  duplicate_jvp.target_identity = "case::boundary::jvp-duplicate";
  duplicate_jvp.outputs = {"case::boundary::jvp-output-duplicate"};
  EXPECT_THROW(PreparedBoundaryPlan::validate_linearization_bijection(
                   {residual, duplicate_residual}, {jvp, duplicate_jvp}),
               std::runtime_error)
      << "ambiguous duplicate JVPs fail closed";
}

TEST(test_prepared_boundary_plan, rejects_inexact_jvp_target_direction_and_output_tables) {
  const auto residual =
      linearization_spec(false, "case::boundary::residual", "case::boundary::residual-output");
  auto jvp = linearization_spec(true, "case::boundary::jvp", "case::boundary::jvp-output");

  jvp.directions = {"case::other-block::state"};
  EXPECT_THROW(PreparedBoundaryPlan::validate_linearization_bijection({residual}, {jvp}),
               std::runtime_error);
  jvp = linearization_spec(true, "case::boundary::jvp", "case::boundary::jvp-output");
  jvp.outputs.clear();
  EXPECT_THROW(PreparedBoundaryPlan::validate_linearization_bijection({residual}, {jvp}),
               std::runtime_error);
  jvp = linearization_spec(true, residual.target_identity, "case::boundary::jvp-output");
  EXPECT_THROW(PreparedBoundaryPlan::validate_linearization_bijection({residual}, {jvp}),
               std::runtime_error);
}
