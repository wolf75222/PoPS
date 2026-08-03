#include <gtest/gtest.h>

#include <pops/core/foundation/allocator.hpp>
#include <pops/mesh/boundary/prepared_boundary_plan.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/runtime/context/grid_context.hpp>

#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

using namespace pops;

namespace {

MultiFab scalar_field(const Box2D& domain, int ncomp = 1, int ngrow = 0) {
  const BoxArray boxes = BoxArray::from_domain(domain, domain.nx());
  return MultiFab(boxes, DistributionMapping(boxes.size(), n_ranks()), ncomp, ngrow);
}

PreparedHyperbolicBoundary<2> physical_boundary(std::vector<double> xhi_values = {4.0},
                                                std::vector<std::string> roles = {"Scalar"}) {
  std::vector<double> values;
  values.reserve(4 * xhi_values.size());
  for (double value : xhi_values)
    values.insert(values.end(), {0.0, value, 0.0, 0.0});
  return prepare_hyperbolic_boundary<2>({"foextrap", "dirichlet", "foextrap", "foextrap"}, values,
                                        {"case::xlo", "case::xhi", "case::ylo", "case::yhi"},
                                        roles);
}

PreparedHyperbolicBoundary<2> periodic_boundary(
    std::vector<std::string> face_types = {"periodic", "periodic", "foextrap", "foextrap"},
    bool explicit_identifications = false) {
  if (face_types.empty())
    face_types = {"periodic", "periodic", "foextrap", "foextrap"};
  return prepare_hyperbolic_boundary<2>(
      face_types, std::vector<double>(4, 0.0),
      {"case::periodic::xlo", "case::periodic::xhi", "case::periodic::ylo", "case::periodic::yhi"},
      {"Scalar"}, explicit_identifications);
}

PreparedHyperbolicBoundary<2> analytic_xlo_boundary(
    std::vector<std::string> opcodes = {"x", "y", "add", "input", "add"},
    std::vector<double> literals = {0.0, 0.0, 0.0, 0.0, 0.0}, bool periodic_tangent = false) {
  const bool reads_time = std::find(opcodes.begin(), opcodes.end(), "input") != opcodes.end();
  return prepare_hyperbolic_boundary<2>(
      periodic_tangent ? std::vector<std::string>{"dirichlet", "foextrap", "periodic", "periodic"}
                       : std::vector<std::string>{"dirichlet", "foextrap", "foextrap", "foextrap"},
      std::vector<double>(4, 0.0),
      {"case::analytic::xlo", "case::analytic::xhi", "case::analytic::ylo", "case::analytic::yhi"},
      {"Scalar"}, false, {}, {},
      {std::move(opcodes), std::vector<std::string>{}, std::vector<std::string>{},
       std::vector<std::string>{}},
      {std::move(literals), std::vector<double>{}, std::vector<double>{}, std::vector<double>{}},
      {reads_time ? "clock.analytic" : "", "", "", ""});
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

RecoveryReport recover_positive_scalar(const double* conserved, double* primitive) {
  RecoveryReport report;
  if (std::isfinite(conserved[0]) && conserved[0] > 0.0) {
    primitive[0] = conserved[0];
    report.status = RecoveryStatus::kRecovered;
    report.cause = RecoveryCause::kNone;
  } else {
    report.status = RecoveryStatus::kRejected;
    report.cause = RecoveryCause::kInadmissibleCandidate;
    report.failing_component = 0;
  }
  return report;
}

}  // namespace

TEST(PreparedBoundaryTraceRecovery,
     accepts_admissible_physical_traces_without_hot_path_allocation) {
  const Box2D domain = Box2D::from_extents(4, 4);
  const Geometry geometry(domain, Real(0), Real(1), Real(0), Real(1));
  MultiFab state = scalar_field(domain, 1, 1);
  state.set_val(Real(-99));
  for (int local = 0; local < state.local_size(); ++local) {
    const Array4 values = state.fab(local).array();
    for_each_cell(state.box(local), [=](int i, int j) { values(i, j, 0) = Real(1); });
  }
  device_fence();

  auto plan = std::make_shared<PreparedBoundaryPlan>("case::boundary::recoverable-traces", 1,
                                                     physical_boundary({4.0}, {"Scalar"}));
  plan->prepare_trace_recovery(recover_positive_scalar);
  GridContext context;
  context.dom = domain;
  context.geom = geometry;
  context.boundary_plan = plan;
  const auto lane = ExecutionLane::world("case::boundary::recoverable-traces-lane");
  const runtime::multiblock::BoundaryEvaluationPoint point{"clock.boundary",    0,   0,  0, 0,
                                                           amr::Rational(0, 1), 0.1, 0.0};
  PreparedGridBoundarySession session(context, lane, state, point);

  session.fill(state, point);
  const AllocationEventStats before = allocation_event_stats();
  session.fill(state, point);
  const AllocationEventStats after = allocation_event_stats();

  EXPECT_EQ(after, before);
  if (state.local_size() > 0) {
    state.sync_host();
    EXPECT_EQ(state.fab(0)(domain.hi[0] + 1, 2, 0), Real(7));
  }
}

TEST(PreparedBoundaryTraceRecovery,
     rejects_inadmissible_traces_and_restores_complete_ghost_transaction) {
  const Box2D domain = Box2D::from_extents(4, 4);
  const Geometry geometry(domain, Real(0), Real(1), Real(0), Real(1));
  MultiFab state = scalar_field(domain, 1, 1);
  state.set_val(Real(-99));
  for (int local = 0; local < state.local_size(); ++local) {
    const Array4 values = state.fab(local).array();
    for_each_cell(state.box(local), [=](int i, int j) { values(i, j, 0) = Real(1); });
  }
  device_fence();
  const MultiFab before = state;

  auto plan = std::make_shared<PreparedBoundaryPlan>("case::boundary::rejected-traces", 1,
                                                     physical_boundary({0.0}, {"Scalar"}));
  plan->prepare_trace_recovery(recover_positive_scalar);
  GridContext context;
  context.dom = domain;
  context.geom = geometry;
  context.boundary_plan = plan;
  const auto lane = ExecutionLane::world("case::boundary::rejected-traces-lane");
  const runtime::multiblock::BoundaryEvaluationPoint point{"clock.boundary",    0,   0,  0, 0,
                                                           amr::Rational(0, 1), 0.1, 0.0};
  PreparedGridBoundarySession session(context, lane, state, point);

  EXPECT_THROW(session.fill(state, point), std::runtime_error);
  state.sync_host();
  before.sync_host();
  ASSERT_EQ(state.local_size(), before.local_size());
  for (int local = 0; local < state.local_size(); ++local) {
    const Fab2D& observed = state.fab(local);
    const Fab2D& expected = before.fab(local);
    const Box2D grown = observed.grown_box();
    for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
      for (int i = grown.lo[0]; i <= grown.hi[0]; ++i)
        EXPECT_EQ(observed(i, j, 0), expected(i, j, 0))
            << "rejected trace mutated local fab " << local << " at (" << i << ", " << j << ")";
  }
}

TEST(test_prepared_boundary_plan, explicit_read_dependencies_are_exact_and_strict) {
  PreparedBoundaryPlan plan(
      "case::boundary::read-dependencies", 1, physical_boundary(), {}, "case::state::primary",
      PreparedBoundaryReadDependencies{{"case::state::other"}, {"case::field::potential"}});
  EXPECT_EQ(plan.required_state_identities(), std::vector<std::string>{"case::state::other"});
  EXPECT_EQ(plan.required_field_identities(), std::vector<std::string>{"case::field::potential"});

  EXPECT_THROW(
      PreparedBoundaryPlan(
          "case::boundary::duplicate-state", 1, physical_boundary(), {}, "case::state::primary",
          PreparedBoundaryReadDependencies{{"case::state::other", "case::state::other"}, {}}),
      std::runtime_error);
  EXPECT_THROW(
      PreparedBoundaryPlan("case::boundary::empty-field", 1, physical_boundary(), {},
                           "case::state::primary", PreparedBoundaryReadDependencies{{}, {""}}),
      std::runtime_error);
}

TEST(test_prepared_boundary_plan, prepared_read_tokens_are_owner_bound_and_epoch_checked) {
  const Box2D domain = Box2D::from_extents(3, 3);
  MultiFab primary = scalar_field(domain, 1, 1);
  MultiFab coupled = scalar_field(domain, 1, 1);
  MultiFab auxiliary = scalar_field(domain, 1, 0);
  auto plan = std::make_shared<PreparedBoundaryPlan>(
      "case::boundary::prepared-reads", 1, physical_boundary(), std::vector<int>{},
      "case::state::primary",
      PreparedBoundaryReadDependencies{{"case::state::coupled"}, {"case::field::auxiliary"}});
  auto foreign_plan = std::make_shared<PreparedBoundaryPlan>(
      "case::boundary::foreign-reads", 1, physical_boundary(), std::vector<int>{},
      "case::state::primary", PreparedBoundaryReadDependencies{{"case::state::coupled"}, {}});
  const auto coupled_read = plan->prepare_state_read("case::state::coupled");
  const auto auxiliary_read = plan->prepare_field_read("case::field::auxiliary");
  const auto foreign_read = foreign_plan->prepare_state_read("case::state::coupled");
  EXPECT_THROW((void)plan->prepare_state_read("case::state::missing"), std::invalid_argument);

  GridContext context;
  context.dom = domain;
  context.geom = Geometry(domain, Real(0), Real(1), Real(0), Real(1));
  context.boundary_plan = plan;
  int bindings = 0;
  context.boundary_field_registry = [&](const auto&, MultiFab&, const MultiFab*, MultiFab*,
                                        detail::BoundaryFieldRegistry& registry) {
    ++bindings;
    registry.bind_state_slot(0, coupled);
    registry.bind_field_slot(0, auxiliary);
  };
  const runtime::multiblock::BoundaryEvaluationPoint point{"clock.prepared-reads", 0,   0,  0, 0,
                                                           amr::Rational(0, 1),    0.1, 0.0};
  const auto lane = ExecutionLane::world("case::boundary::prepared-read-lane");
  EXPECT_THROW(PreparedGridBoundarySession(context, lane), std::invalid_argument);
  PreparedGridBoundarySession session(context, lane, primary, point);

  const auto first = session.bind_reads(point, primary);
  EXPECT_EQ(&first.state(coupled_read), &coupled);
  EXPECT_EQ(&first.field(auxiliary_read), &auxiliary);
  const auto second = session.bind_reads(point, primary);
  EXPECT_THROW((void)first.state(coupled_read), std::logic_error);
  EXPECT_THROW((void)second.state(foreign_read), std::invalid_argument);
  EXPECT_EQ(&second.state(coupled_read), &coupled);
  EXPECT_EQ(bindings, 3);
}

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
  PreparedBoundaryPlan plan("case::block::ghost-plan", 1,
                            physical_boundary({4.0, 9.0}, {"Scalar", "Scalar"}));

  plan.fill_same_level_and_physical(state, domain);

  const Fab2D& field = state.fab(0);
  EXPECT_EQ(field(-1, 2, 0), Real(1));
  EXPECT_EQ(field(-1, 2, 1), Real(2));
  EXPECT_EQ(field(4, 2, 0), Real(7));   // 2*4 - interior(1)
  EXPECT_EQ(field(4, 2, 1), Real(16));  // 2*9 - interior(2)
}

TEST(test_prepared_boundary_plan,
     evaluates_prepared_coordinate_time_inflow_on_device_without_hot_path_allocation) {
  const Box2D domain = Box2D::from_extents(4, 3);
  const Geometry geometry(domain, Real(1), Real(5), Real(0), Real(3));
  MultiFab state = scalar_field(domain, 1, 1);
  state.set_val(Real(-99));
  for (int local = 0; local < state.local_size(); ++local) {
    const Array4 values = state.fab(local).array();
    for_each_cell(state.box(local), [=](int i, int j) { values(i, j, 0) = Real(2); });
  }
  PreparedBoundaryPlan plan("case::analytic::plan", 1, analytic_xlo_boundary());
  const auto lane = ExecutionLane::world("case::analytic::lane");
  auto session = plan.make_session(lane);
  const runtime::multiblock::BoundaryEvaluationPoint point{"clock.analytic",    1,   0,   0, 0,
                                                           amr::Rational(0, 1), 0.1, 0.25};

  EXPECT_THROW(session.fill_same_level_and_physical(state, domain), std::logic_error);
  EXPECT_THROW(session.fill_same_level_and_physical(
                   state, geometry,
                   runtime::multiblock::BoundaryEvaluationPoint{"clock.other", 1, 0, 0, 0,
                                                                amr::Rational(0, 1), 0.1, 0.25}),
               std::invalid_argument);
  if (state.local_size() > 0)
    EXPECT_EQ(state.fab(0)(-1, 1, 0), Real(-99));

  session.fill_same_level_and_physical(state, geometry, point);
  if (state.local_size() > 0)
    EXPECT_EQ(state.fab(0)(-1, 1, 0), Real(3.5));
  const AllocationEventStats before = allocation_event_stats();
  session.fill_same_level_and_physical(state, geometry, point);
  const AllocationEventStats after = allocation_event_stats();
  EXPECT_EQ(after, before);
}

TEST(test_prepared_boundary_plan, analytic_inflow_preflights_nonfinite_values_before_any_mutation) {
  const Box2D domain = Box2D::from_extents(4, 3);
  const Geometry geometry(domain, Real(0), Real(4), Real(0), Real(3));
  const BoxArray boxes = BoxArray::from_domain(domain, 2);
  MultiFab state(boxes, DistributionMapping(boxes.size(), n_ranks()), 1, 1);
  state.set_val(Real(-99));
  for (int local = 0; local < state.local_size(); ++local) {
    const Array4 values = state.fab(local).array();
    for_each_cell(state.box(local), [=](int i, int j) { values(i, j, 0) = Real(2 + i + 10 * j); });
  }
  device_fence();
  const MultiFab before = state;
  PreparedBoundaryPlan plan("case::analytic::invalid-plan", 1,
                            analytic_xlo_boundary({"constant", "log"}, {-1.0, 0.0}, true));
  const auto lane = ExecutionLane::world("case::analytic::invalid-lane");
  auto session = plan.make_session(lane);
  const runtime::multiblock::BoundaryEvaluationPoint point{"clock.analytic",    1,   0,   0, 0,
                                                           amr::Rational(0, 1), 0.1, 0.25};

  EXPECT_THROW(session.fill_same_level_and_physical(state, geometry, point), std::runtime_error);
  state.sync_host();
  before.sync_host();
  ASSERT_EQ(state.local_size(), before.local_size());
  for (int local = 0; local < state.local_size(); ++local) {
    const Fab2D& observed = state.fab(local);
    const Fab2D& expected = before.fab(local);
    const Box2D grown = observed.grown_box();
    for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
      for (int i = grown.lo[0]; i <= grown.hi[0]; ++i)
        EXPECT_EQ(observed(i, j, 0), expected(i, j, 0))
            << "analytic refusal mutated local fab " << local << " at (" << i << ", " << j << ")";
  }
}

TEST(test_prepared_boundary_plan, analytic_inflow_authenticates_one_clock_and_time_slot_per_plan) {
  const auto face_types =
      std::vector<std::string>{"dirichlet", "foextrap", "dirichlet", "foextrap"};
  const auto face_values = std::vector<double>(4, 0.0);
  const auto face_identities = std::vector<std::string>{
      "case::analytic::xlo", "case::analytic::xhi", "case::analytic::ylo", "case::analytic::yhi"};
  const auto roles = std::vector<std::string>{"Scalar"};
  const auto opcodes = std::vector<std::vector<std::string>>{{"input"}, {}, {"input"}, {}};
  const auto literals = std::vector<std::vector<double>>{{0.0}, {}, {0.0}, {}};

  EXPECT_THROW(
      prepare_hyperbolic_boundary<2>(face_types, face_values, face_identities, roles, false, {}, {},
                                     opcodes, literals, {"clock.first", "", "clock.second", ""}),
      std::invalid_argument);
  EXPECT_THROW(
      prepare_hyperbolic_boundary<2>({"dirichlet", "foextrap", "foextrap", "foextrap"}, face_values,
                                     face_identities, roles, false, {}, {}, {{"input"}, {}, {}, {}},
                                     {{1.0}, {}, {}, {}}, {"clock.first", "", "", ""}),
      std::invalid_argument);
  auto ambiguous_values = face_values;
  ambiguous_values[0] = 1.0;
  EXPECT_THROW(
      prepare_hyperbolic_boundary<2>({"dirichlet", "foextrap", "foextrap", "foextrap"},
                                     ambiguous_values, face_identities, roles, false, {}, {},
                                     {{"x"}, {}, {}, {}}, {{0.0}, {}, {}, {}}, {"", "", "", ""}),
      std::invalid_argument);
  EXPECT_THROW(prepare_hyperbolic_boundary<2>(face_types, face_values, face_identities, roles,
                                              false, {}, {}, {{}, {}, {}, {}}, {{}, {}, {}, {}},
                                              {"clock.without-program", "", "", ""}),
               std::invalid_argument);
}

TEST(test_prepared_boundary_plan,
     converts_primitive_fixed_state_once_before_conservative_face_execution) {
  const Box2D domain = Box2D::from_extents(4, 4);
  MultiFab state = scalar_field(domain, 4, 1);
  for (int local = 0; local < state.local_size(); ++local) {
    const Array4 values = state.fab(local).array();
    for_each_cell(state.box(local), [=](int i, int j) {
      for (int component = 0; component < 4; ++component)
        values(i, j, component) = Real(1);
    });
  }
  std::vector<double> face_values;
  for (const double primitive : {2.0, 3.0, -1.0, 4.0})
    face_values.insert(face_values.end(), {0.0, primitive, 0.0, 0.0});
  auto boundary = prepare_hyperbolic_boundary<2>(
      {"foextrap", "dirichlet", "foextrap", "foextrap"}, face_values,
      {"case::fluid::xlo", "case::fluid::xhi", "case::fluid::ylo", "case::fluid::yhi"},
      {"Density", "MomentumX", "MomentumY", "Energy"}, false,
      {"conservative", "primitive", "conservative", "conservative"},
      {"", "case::fluid::model-p2c", "", ""});
  PreparedBoundaryPlan plan("case::fluid::primitive-inflow", 1, std::move(boundary));
  const auto lane = ExecutionLane::world("case::fluid::primitive-inflow-lane");
  auto stale_session = plan.make_session(lane);

  EXPECT_TRUE(plan.requires_fixed_state_conversion());
  EXPECT_THROW(plan.fill_same_level_and_physical(state, domain), std::logic_error);
  plan.prepare_fixed_state_conversion([](const double* primitive, double* conservative) {
    constexpr double gamma = 1.4;
    conservative[0] = primitive[0];
    conservative[1] = primitive[0] * primitive[1];
    conservative[2] = primitive[0] * primitive[2];
    conservative[3] =
        primitive[3] / (gamma - 1.0) +
        0.5 * primitive[0] * (primitive[1] * primitive[1] + primitive[2] * primitive[2]);
  });

  EXPECT_FALSE(plan.requires_fixed_state_conversion());
  const auto& prepared_xhi = plan.hyperbolic_boundary().face(0, 1);
  EXPECT_EQ(prepared_xhi.authored_representation, HyperbolicStateRepresentation::Primitive);
  EXPECT_EQ(prepared_xhi.converter_identity, "case::fluid::model-p2c");
  EXPECT_TRUE(prepared_xhi.fixed_state_converted);
  EXPECT_THROW(stale_session.fill_same_level_and_physical(state, domain), std::logic_error);
  plan.fill_same_level_and_physical(state, domain);
  const Fab2D& field = state.fab(0);
  EXPECT_EQ(field(4, 2, 0), Real(3));
  EXPECT_EQ(field(4, 2, 1), Real(11));
  EXPECT_EQ(field(4, 2, 2), Real(-5));
  EXPECT_NEAR(field(4, 2, 3), Real(39), Real(1e-12));
}

TEST(test_prepared_boundary_plan, primitive_fixed_state_conversion_is_transactional_and_finite) {
  auto make_plan = [] {
    return PreparedBoundaryPlan(
        "case::fluid::nonfinite-inflow", 1,
        prepare_hyperbolic_boundary<2>(
            {"foextrap", "dirichlet", "foextrap", "foextrap"}, {0.0, 2.0, 0.0, 0.0},
            {"case::fluid::xlo", "case::fluid::xhi", "case::fluid::ylo", "case::fluid::yhi"},
            {"Scalar"}, false, {"conservative", "primitive", "conservative", "conservative"},
            {"", "case::fluid::model-p2c", "", ""}));
  };
  auto plan = make_plan();
  EXPECT_THROW(plan.prepare_fixed_state_conversion([](const double*, double* conservative) {
    conservative[0] = std::numeric_limits<double>::quiet_NaN();
  }),
               std::runtime_error);
  EXPECT_TRUE(plan.requires_fixed_state_conversion());

  EXPECT_THROW(prepare_hyperbolic_boundary<2>(
                   {"foextrap", "foextrap", "foextrap", "foextrap"}, std::vector<double>(4, 0.0),
                   {"case::xlo", "case::xhi", "case::ylo", "case::yhi"}, {"Scalar"}, false,
                   {"primitive", "conservative", "conservative", "conservative"},
                   {"case::fluid::model-p2c", "", "", ""}),
               std::invalid_argument);
}

TEST(test_prepared_boundary_plan,
     primitive_conversion_preserves_explicit_periodic_identification_validation) {
  auto boundary = prepare_hyperbolic_boundary<2>(
      {"periodic", "dirichlet", "foextrap", "periodic"}, {0.0, 2.0, 0.0, 0.0},
      {"case::fluid::xlo", "case::fluid::xhi", "case::fluid::ylo", "case::fluid::yhi"}, {"Scalar"},
      true, {"conservative", "primitive", "conservative", "conservative"},
      {"", "case::fluid::model-p2c", "", ""});

  EXPECT_TRUE(boundary.requires_fixed_state_conversion());
  const auto converted = boundary.with_converted_fixed_states(
      [](const double* primitive, double* conservative) { conservative[0] = primitive[0]; });
  EXPECT_FALSE(converted.requires_fixed_state_conversion());
}

TEST(test_prepared_boundary_plan,
     model_aware_slip_wall_handles_multiple_normal_and_out_of_plane_components) {
  const Box2D domain = Box2D::from_extents(4, 4);
  MultiFab state = scalar_field(domain, 5, 2);
  for (int local = 0; local < state.local_size(); ++local) {
    const Array4 values = state.fab(local).array();
    for_each_cell(state.box(local), [=](int i, int j) {
      values(i, j, 0) = Real(1);
      values(i, j, 1) = Real(2);
      values(i, j, 2) = Real(5);
      values(i, j, 3) = Real(3);
      values(i, j, 4) = Real(4);
    });
  }
  auto boundary = prepare_hyperbolic_boundary<2>(
      {"slip_wall", "slip_wall", "slip_wall", "slip_wall"}, std::vector<double>(20, 0.0),
      {"case::fluid::xlo", "case::fluid::xhi", "case::fluid::ylo", "case::fluid::yhi"},
      {"Density", "MomentumX", "MomentumX", "MomentumY", "AxialZ"});
  PreparedBoundaryPlan plan("case::fluid::slip-plan", 2, std::move(boundary));

  plan.fill_same_level_and_physical(state, domain);

  const Fab2D& field = state.fab(0);
  EXPECT_EQ(field(-1, 2, 0), Real(1));
  EXPECT_EQ(field(-1, 2, 1), Real(-2));
  EXPECT_EQ(field(-1, 2, 2), Real(-5));
  EXPECT_EQ(field(-1, 2, 3), Real(3));
  EXPECT_EQ(field(-1, 2, 4), Real(-4));
  EXPECT_EQ(field(-2, 2, 1), Real(-2));
  EXPECT_EQ(field(2, -1, 1), Real(2));
  EXPECT_EQ(field(2, -1, 2), Real(5));
  EXPECT_EQ(field(2, -1, 3), Real(-3));
  EXPECT_EQ(field(2, -1, 4), Real(-4));
  EXPECT_EQ(field(2, -2, 3), Real(-3));
}

TEST(test_prepared_boundary_plan, polar_and_axial_reflections_are_distinct_in_1d_2d_3d_frames) {
  const auto polar_1d = HyperbolicComponentTransform<1>::polar_vector(0);
  EXPECT_EQ(polar_1d.reflection_sign(0), Real(-1));

  const auto polar_normal_2d = HyperbolicComponentTransform<2>::polar_vector(0);
  const auto polar_tangent_2d = HyperbolicComponentTransform<2>::polar_vector(1);
  const auto polar_out_of_plane_2d = HyperbolicComponentTransform<2>::polar_vector(2);
  const auto axial_normal_2d = HyperbolicComponentTransform<2>::axial_vector(0);
  const auto axial_tangent_2d = HyperbolicComponentTransform<2>::axial_vector(1);
  const auto axial_out_of_plane_2d = HyperbolicComponentTransform<2>::axial_vector(2);
  EXPECT_EQ(polar_normal_2d.reflection_sign(0), Real(-1));
  EXPECT_EQ(polar_tangent_2d.reflection_sign(0), Real(1));
  EXPECT_EQ(polar_out_of_plane_2d.reflection_sign(0), Real(1));
  EXPECT_EQ(polar_out_of_plane_2d.reflection_sign(1), Real(1));
  EXPECT_EQ(axial_normal_2d.reflection_sign(0), Real(1));
  EXPECT_EQ(axial_tangent_2d.reflection_sign(0), Real(-1));
  EXPECT_EQ(axial_out_of_plane_2d.reflection_sign(0), Real(-1));
  EXPECT_EQ(axial_out_of_plane_2d.reflection_sign(1), Real(-1));

  const auto polar_z_3d = HyperbolicComponentTransform<3>::polar_vector(2);
  const auto axial_z_3d = HyperbolicComponentTransform<3>::axial_vector(2);
  EXPECT_EQ(polar_z_3d.reflection_sign(0), Real(1));
  EXPECT_EQ(axial_z_3d.reflection_sign(0), Real(-1));
  EXPECT_EQ(polar_z_3d.reflection_sign(2), Real(-1));
  EXPECT_EQ(axial_z_3d.reflection_sign(2), Real(1));
}

TEST(test_prepared_boundary_plan, slip_wall_fails_without_declared_normal_polar_role) {
  EXPECT_THROW(prepare_hyperbolic_boundary<2>(
                   {"slip_wall", "slip_wall", "foextrap", "foextrap"}, std::vector<double>(8, 0.0),
                   {"case::xlo", "case::xhi", "case::ylo", "case::yhi"}, {"Density", "MomentumY"}),
               std::invalid_argument);
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
  PreparedBoundaryPlan plan("case::block::session-plan", 1, physical_boundary());
  const auto lane = ExecutionLane::world("case::block::session-lane");
  auto original = plan.make_session(lane);
  auto session = std::move(original);

  EXPECT_THROW(original.fill_same_level_and_physical(state, domain), std::logic_error);
  EXPECT_NO_THROW(session.fill_same_level_and_physical(state, domain));
  EXPECT_EQ(state.fab(0)(-1, 2, 0), Real(3));
  EXPECT_EQ(state.fab(0)(4, 2, 0), Real(5));
}

TEST(test_prepared_boundary_plan, rejects_field_only_robin_as_transport_semantics) {
  EXPECT_THROW(prepare_hyperbolic_boundary<2>(
                   {"robin", "foextrap", "foextrap", "foextrap"}, {0.0, 0.0, 0.0, 0.0},
                   {"case::xlo", "case::xhi", "case::ylo", "case::yhi"}, {"Scalar"}),
               std::invalid_argument);
}

TEST(test_prepared_boundary_plan, rejects_incomplete_periodic_pairs_and_insufficient_ghosts) {
  EXPECT_THROW(prepare_hyperbolic_boundary<2>(
                   {"periodic", "foextrap", "foextrap", "foextrap"}, {0.0, 0.0, 0.0, 0.0},
                   {"case::xlo", "case::xhi", "case::ylo", "case::yhi"}, {"Scalar"}),
               std::invalid_argument);

  const Box2D domain = Box2D::from_extents(2, 2);
  MultiFab state = scalar_field(domain, 1, 1);
  PreparedBoundaryPlan deep("case::deep::ghost-plan", 2, physical_boundary());
  EXPECT_THROW(deep.fill_same_level_and_physical(state, domain), std::runtime_error);
}

TEST(test_prepared_boundary_plan, executes_reflected_periodic_ghosts_on_a_multibox_layout) {
  const Box2D domain = Box2D::from_extents(8, 6);
  const BoxArray boxes = BoxArray::from_domain(domain, 4);
  MultiFab state(boxes, DistributionMapping(boxes.size(), n_ranks()), 1, 1);
  for (int local = 0; local < state.local_size(); ++local) {
    Array4 values = state.fab(local).array();
    for_each_cell(state.box(local), [=](int i, int j) { values(i, j, 0) = Real(i + 100 * j); });
  }
  const PeriodicIdentification2D reflected_x{0, 1, std::array<int, 2>{{0, 1}},
                                             std::array<int, 2>{{1, -1}}};
  PreparedBoundaryPlan plan("case::block::reflected-x", 1, periodic_boundary({}, true), {}, "", {},
                            {reflected_x});

  plan.fill_same_level_and_physical(state, domain);

  for (int local = 0; local < state.local_size(); ++local) {
    const Fab2D& field = state.fab(local);
    const Box2D grown = field.grown_box();
    for (int j = domain.lo[1]; j <= domain.hi[1]; ++j) {
      if (grown.contains(domain.lo[0] - 1, j))
        EXPECT_EQ(field(domain.lo[0] - 1, j, 0),
                  Real(domain.hi[0] + 100 * (domain.lo[1] + domain.hi[1] - j)));
      if (grown.contains(domain.hi[0] + 1, j))
        EXPECT_EQ(field(domain.hi[0] + 1, j, 0),
                  Real(domain.lo[0] + 100 * (domain.lo[1] + domain.hi[1] - j)));
    }
    const Box2D valid = field.box();
    const Box2D interior_ghosts = grown.intersect(domain);
    for (int j = interior_ghosts.lo[1]; j <= interior_ghosts.hi[1]; ++j)
      for (int i = interior_ghosts.lo[0]; i <= interior_ghosts.hi[0]; ++i)
        if (!valid.contains(i, j))
          EXPECT_EQ(field(i, j, 0), Real(i + 100 * j));
  }
}

TEST(test_prepared_boundary_plan, explicit_identity_periodicity_keeps_the_legacy_values) {
  const Box2D domain = Box2D::from_extents(6, 5);
  const BoxArray boxes = BoxArray::from_domain(domain, 3);
  MultiFab legacy(boxes, DistributionMapping(boxes.size(), n_ranks()), 1, 1);
  MultiFab explicit_identity(boxes, DistributionMapping(boxes.size(), n_ranks()), 1, 1);
  for (int local = 0; local < legacy.local_size(); ++local) {
    Array4 legacy_values = legacy.fab(local).array();
    Array4 explicit_values = explicit_identity.fab(local).array();
    for_each_cell(legacy.box(local), [=](int i, int j) {
      const Real value = Real(17 * i - 3 * j);
      legacy_values(i, j, 0) = value;
      explicit_values(i, j, 0) = value;
    });
  }
  const PeriodicIdentification2D identity{0, 1, std::array<int, 2>{{0, 1}},
                                          std::array<int, 2>{{1, 1}}};
  PreparedBoundaryPlan legacy_plan("case::block::legacy-periodic", 1, periodic_boundary());
  PreparedBoundaryPlan explicit_plan("case::block::explicit-identity-periodic", 1,
                                     periodic_boundary({}, true), {}, "", {}, {identity});

  legacy_plan.fill_same_level_and_physical(legacy, domain);
  explicit_plan.fill_same_level_and_physical(explicit_identity, domain);

  for (int local = 0; local < legacy.local_size(); ++local) {
    const Fab2D& expected = legacy.fab(local);
    const Fab2D& actual = explicit_identity.fab(local);
    const Box2D grown = expected.grown_box();
    for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
      for (int i = grown.lo[0]; i <= grown.hi[0]; ++i)
        EXPECT_EQ(actual(i, j, 0), expected(i, j, 0));
  }
}

TEST(test_prepared_boundary_plan, axis_permutation_refuses_incompatible_rectangular_geometry) {
  const PeriodicIdentification2D xlo_to_yhi{0, 3, std::array<int, 2>{{1, 0}},
                                            std::array<int, 2>{{1, 1}}};
  PreparedBoundaryPlan plan(
      "case::block::rotated-periodic", 1,
      periodic_boundary({"periodic", "foextrap", "foextrap", "periodic"}, true), {}, "", {},
      {xlo_to_yhi});
  const Box2D rectangular_domain = Box2D::from_extents(8, 6);
  MultiFab state = scalar_field(rectangular_domain, 1, 1);

  EXPECT_THROW(plan.fill_same_level_and_physical(state, rectangular_domain), std::invalid_argument);
}

TEST(test_prepared_boundary_plan, mapped_periodicity_refuses_unmapped_vector_components) {
  const PeriodicIdentification2D xlo_to_yhi{0, 3, std::array<int, 2>{{1, 0}},
                                            std::array<int, 2>{{1, 1}}};
  auto vector_boundary = prepare_hyperbolic_boundary<2>(
      {"periodic", "foextrap", "foextrap", "periodic"}, std::vector<double>(8, 0.0),
      {"case::vector::xlo", "case::vector::xhi", "case::vector::ylo", "case::vector::yhi"},
      {"Density", "MomentumX"}, true);
  EXPECT_THROW(PreparedBoundaryPlan("case::block::rotated-vector-periodic", 1,
                                    std::move(vector_boundary), {}, "", {}, {xlo_to_yhi}),
               std::runtime_error);
}

TEST(test_prepared_boundary_plan, mapped_periodicity_refuses_unmapped_analytic_coordinates) {
  const PeriodicIdentification2D xlo_to_yhi{0, 3, std::array<int, 2>{{1, 0}},
                                            std::array<int, 2>{{1, 1}}};
  auto analytic_boundary = prepare_hyperbolic_boundary<2>(
      {"periodic", "dirichlet", "foextrap", "periodic"}, std::vector<double>(4, 0.0),
      {"case::analytic::xlo", "case::analytic::xhi", "case::analytic::ylo", "case::analytic::yhi"},
      {"Scalar"}, true, {}, {}, {{}, {"x"}, {}, {}}, {{}, {0.0}, {}, {}}, {"", "", "", ""});

  EXPECT_THROW(PreparedBoundaryPlan("case::block::rotated-analytic-periodic", 1,
                                    std::move(analytic_boundary), {}, "", {}, {xlo_to_yhi}),
               std::runtime_error);
}

TEST(test_prepared_boundary_plan, axis_permutation_executes_on_a_square_domain) {
  const PeriodicIdentification2D xlo_to_yhi{0, 3, std::array<int, 2>{{1, 0}},
                                            std::array<int, 2>{{1, 1}}};
  PreparedBoundaryPlan plan(
      "case::block::rotated-periodic-square", 1,
      periodic_boundary({"periodic", "foextrap", "foextrap", "periodic"}, true), {}, "", {},
      {xlo_to_yhi});
  const Box2D domain = Box2D::from_extents(6, 6);
  MultiFab state = scalar_field(domain, 1, 1);
  for (int local = 0; local < state.local_size(); ++local) {
    const Array4 values = state.fab(local).array();
    for_each_cell(state.box(local), [=](int i, int j) { values(i, j, 0) = Real(i + 100 * j); });
  }

  EXPECT_NO_THROW(plan.fill_same_level_and_physical(state, domain));
  for (int local = 0; local < state.local_size(); ++local) {
    const Fab2D& field = state.fab(local);
    const Box2D grown = field.grown_box();
    for (int j = domain.lo[1]; j <= domain.hi[1]; ++j)
      if (grown.contains(domain.lo[0] - 1, j))
        EXPECT_EQ(field(domain.lo[0] - 1, j, 0), Real(j + 100 * domain.hi[1]));
    for (int i = domain.lo[0]; i <= domain.hi[0]; ++i)
      if (grown.contains(i, domain.hi[1] + 1))
        EXPECT_EQ(field(i, domain.hi[1] + 1, 0), Real(domain.lo[0] + 100 * i));
  }
}

TEST(test_prepared_boundary_plan, grid_context_routes_exact_nary_storage_registry) {
  const Box2D domain = Box2D::from_extents(3, 3);
  MultiFab primary = scalar_field(domain, 1, 1);
  MultiFab coupled = scalar_field(domain, 2, 1);
  MultiFab auxiliary = scalar_field(domain, 3, 1);
  MultiFab output = scalar_field(domain, 1, 0);
  auto plan =
      std::make_shared<PreparedBoundaryPlan>("case::nary::ghost-plan", 1, physical_boundary());
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
