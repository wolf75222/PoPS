#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/runtime/amr/prepared_tagging_execution.hpp>
#include <pops/runtime/amr/persistent_tagging_state.hpp>
#include <pops/runtime/config/component_interfaces.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>
#include <pops/runtime/dynamic/prepared_execution_context.hpp>

#include "component_abi_test_helpers.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace abi = pops::component::test_support;

struct Context {};

struct FluxComponent {
  std::vector<std::string> requirements() const { return {"state", "normal"}; }
  double stability() const { return 1.0; }
  pops::component::EvaluationOutcome<double> evaluate(Context&) const {
    return pops::component::EvaluationOutcome<double>::ok(2.0);
  }
};

struct BoundaryComponent {
  std::vector<std::string> providers() const { return {"state", "logical_time"}; }
  int stencil() const { return 1; }
};

struct TaggerComponent {
  std::vector<std::string> requirements() const { return {"indicator"}; }
  std::string lower(Context&) const { return "tagger-plan"; }
};

struct ClusteringComponent {
  std::string lower(Context&) const { return "cluster-plan"; }
  std::vector<std::string> effects() const { return {"topology"}; }
};

struct TransferComponent {
  int stencil() const { return 2; }
  std::string restart() const { return "stateless"; }
};

struct RefluxComponent {
  int stencil() const { return 1; }
  std::string lower(Context&) const { return "integrated-interface-correction"; }
  std::vector<std::string> effects() const { return {"local-correction"}; }
};

struct SolverComponent {
  pops::component::EvaluationOutcome<int> evaluate(Context&) const {
    return pops::component::EvaluationOutcome<int>::reject("non-converged");
  }
  std::string restart() const { return "warm-start"; }
  std::string report() const { return "solve-report"; }
};

struct WriterComponent {
  std::vector<std::string> effects() const { return {"io"}; }
  std::string format(const double& value) const { return std::to_string(value); }
  std::string report() const { return "writer-report"; }
};

template <int Dim>
struct FillPreparedTaggingFields {
  pops::FieldView<pops::Real, Dim> scalar;
  pops::FieldView<pops::Real, Dim> vector;

  POPS_HD void operator()(const pops::Index<Dim>& index) const {
    scalar(index, 0) = static_cast<pops::Real>(index[0]);
    vector(index, 0) = pops::Real(-17);
    pops::Real sum = pops::Real(0);
    for (int axis = 0; axis < Dim; ++axis)
      sum += static_cast<pops::Real>(index[axis]);
    vector(index, 1) = sum;
  }
};

template <int Dim>
struct SetPreparedTaggingSample {
  pops::FieldView<pops::Real, Dim> field;
  pops::Real value = pops::Real(0);

  POPS_HD void operator()(const pops::Index<Dim>& index) const { field(index, 0) = value; }
};

template <int Dim>
pops::Box<Dim> tagging_domain() {
  pops::Index<Dim> lower{};
  pops::Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] = axis - 3;
    upper[axis] = axis + 2;
  }
  return {lower, upper};
}

template <int Dim>
pops::amr::hierarchy::LevelLayout<Dim> tagging_layout(const pops::Box<Dim>& domain) {
  pops::Box<Dim> left = domain;
  pops::Box<Dim> right = domain;
  left.hi[0] = -1;
  right.lo[0] = 0;
  const pops::mesh::BoxArray<Dim> patches(std::vector<pops::Box<Dim>>{left, right});
  pops::Index<Dim> rank_origin{};
  pops::Extent<Dim> rank_extent{};
  for (int axis = 0; axis < Dim; ++axis) {
    rank_origin[axis] = axis - 2;
    rank_extent[axis] = 1;
  }
  rank_extent[0] = pops::world_communicator_view().size();
  return {0,
          domain,
          patches,
          pops::mesh::Distribution<Dim>::replicated(
              patches, pops::mesh::RankSpace<Dim>{rank_origin, rank_extent}),
          pops::amr::RefinementRatio<Dim>{},
          pops::mesh::BoxArrayValidationBudget{2, 1}};
}

template <int Dim>
pops::amr::hierarchy::LevelLayout<Dim> prescribed_window_layout() {
  pops::Index<Dim> lower{};
  pops::Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = 3;
  pops::Box<Dim> domain{lower, upper};
  pops::Box<Dim> left = domain;
  pops::Box<Dim> right = domain;
  left.hi[0] = 1;
  right.lo[0] = 2;
  const pops::mesh::BoxArray<Dim> patches(std::vector<pops::Box<Dim>>{left, right});
  pops::Index<Dim> rank_origin{};
  pops::Extent<Dim> rank_extent{};
  for (int axis = 0; axis < Dim; ++axis)
    rank_extent[axis] = 1;
  rank_extent[0] = pops::world_communicator_view().size();
  return {0,
          domain,
          patches,
          pops::mesh::Distribution<Dim>::replicated(
              patches, pops::mesh::RankSpace<Dim>{rank_origin, rank_extent}),
          pops::amr::RefinementRatio<Dim>{},
          pops::mesh::BoxArrayValidationBudget{2, 1}};
}

template <int Dim>
pops::runtime::amr::PreparedTaggingExecutionBudget tagging_budget(
    const pops::amr::hierarchy::LevelLayout<Dim>& layout) {
  std::size_t total = 0;
  std::size_t maximum = 0;
  for (const pops::Box<Dim>& patch : layout.patches().boxes()) {
    const std::size_t cells = static_cast<std::size_t>(patch.numPts());
    total += cells;
    maximum = std::max(maximum, cells);
  }
  return {{layout.patches().size(), layout.patches().size(), maximum, total, total, 1U << 20},
          total,
          total * 2u};
}

template <int Dim, class Function>
void for_each_host_cell(const pops::Box<Dim>& box, Function&& function) {
  const std::size_t count = static_cast<std::size_t>(box.numPts());
  for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
    pops::Index<Dim> index{};
    std::size_t quotient = ordinal;
    for (int axis = 0; axis < Dim; ++axis) {
      const std::size_t length = static_cast<std::size_t>(box.length(axis));
      index[axis] = static_cast<int>(static_cast<std::int64_t>(box.lo[axis]) +
                                     static_cast<std::int64_t>(quotient % length));
      quotient /= length;
    }
    function(index);
  }
}

static_assert(pops::component::Requirement<FluxComponent>);
static_assert(pops::component::Stability<FluxComponent>);
static_assert(pops::component::FallibleEvaluation<FluxComponent, Context&>);
static_assert(pops::component::Provider<BoundaryComponent>);
static_assert(pops::component::Stencil<BoundaryComponent>);
static_assert(pops::component::Requirement<TaggerComponent>);
static_assert(pops::component::Lowering<TaggerComponent, Context>);
static_assert(pops::component::Lowering<ClusteringComponent, Context>);
static_assert(pops::component::Effects<ClusteringComponent>);
static_assert(pops::component::Stencil<TransferComponent>);
static_assert(pops::component::Restart<TransferComponent>);
static_assert(pops::component::Stencil<RefluxComponent>);
static_assert(pops::component::Lowering<RefluxComponent, Context>);
static_assert(pops::component::Effects<RefluxComponent>);
static_assert(pops::component::FallibleEvaluation<SolverComponent, Context&>);
static_assert(pops::component::Restart<SolverComponent>);
static_assert(pops::component::Format<WriterComponent, double>);
static_assert(pops::component::Report<WriterComponent>);

pops::component::RegistrationRecord record(std::string id, std::string semantic) {
  return {
      std::move(id),
      "test.external",
      {},
      {"external", "pops://external.test/package", std::move(semantic), "manifest-digest"},
  };
}

TEST(ComponentInterfaces, FallibleOutcomeKeepsTransactionActionExplicit) {
  Context context;
  const auto flux = FluxComponent{}.evaluate(context);
  EXPECT_EQ(flux.status, pops::component::EvaluationStatus::kOk);
  ASSERT_TRUE(flux.value.has_value());
  EXPECT_EQ(*flux.value, 2.0);

  const auto solve = SolverComponent{}.evaluate(context);
  EXPECT_EQ(solve.status, pops::component::EvaluationStatus::kReject);
  EXPECT_EQ(solve.reason, "non-converged");
  EXPECT_THROW(pops::component::EvaluationOutcome<int>::retry(""), std::invalid_argument);
}

template <int Dim>
void verify_persistent_tagging_state() {
  using State = pops::runtime::amr::PersistentTaggingState<Dim>;
  constexpr std::int32_t minimum_cycles = 2;
  pops::Index<Dim> first_cell{};
  pops::Index<Dim> redistributed_cell{};
  for (int axis = 0; axis < Dim; ++axis) {
    first_cell[axis] = axis + 1;
    redistributed_cell[axis] = axis + 4;
  }
  const typename State::CellKey first{0, first_cell};
  const typename State::CellKey redistributed{1, redistributed_cell};

  State state;
  state.begin_cycle(minimum_cycles);
  ASSERT_TRUE(state.transition_allowed(first, minimum_cycles));
  EXPECT_THROW(state.record(first, static_cast<typename State::Decision>(0), minimum_cycles),
               std::invalid_argument);
  state.record(first, State::Decision::Refine, minimum_cycles);
  EXPECT_FALSE(state.transition_allowed(first, minimum_cycles));

  state.begin_cycle(minimum_cycles);
  EXPECT_FALSE(state.transition_allowed(first, minimum_cycles));
  state.record(redistributed, State::Decision::Coarsen, minimum_cycles);
  const auto image = state.encode(minimum_cycles, "test::tagging-graph@1");
  ASSERT_FALSE(image.empty());

  pops::Index<Dim> zero{};
  pops::Index<Dim> coarse_high{};
  pops::Index<Dim> fine_high{};
  for (int axis = 0; axis < Dim; ++axis) {
    coarse_high[axis] = 7;
    fine_high[axis] = 15;
  }
  const std::vector<pops::Box<Dim>> domains{{zero, coarse_high}, {zero, fine_high}};
  State restored = State::decode(image, minimum_cycles, "test::tagging-graph@1", domains);
  EXPECT_EQ(restored.cycle(), state.cycle());
  EXPECT_EQ(restored.active_entry_count(), state.active_entry_count());
  EXPECT_FALSE(restored.transition_allowed(first, minimum_cycles));
  EXPECT_FALSE(restored.transition_allowed(redistributed, minimum_cycles));

  const State accepted = restored;
  restored.begin_cycle(minimum_cycles);
  EXPECT_EQ(accepted.cycle(), state.cycle())
      << "a speculative tagging cycle must detach from the accepted snapshot";
  EXPECT_TRUE(restored.transition_allowed(first, minimum_cycles))
      << "the inclusive min-cycle boundary must allow the next transition";
  EXPECT_FALSE(restored.transition_allowed(redistributed, minimum_cycles));
  EXPECT_THROW((void)State::decode(image, minimum_cycles + 1, "test::tagging-graph@1", domains),
               std::invalid_argument);
  EXPECT_THROW((void)State::decode(image, minimum_cycles, "test::other-graph", domains),
               std::invalid_argument);
  std::vector<std::uint8_t> truncated = image;
  truncated.pop_back();
  EXPECT_THROW((void)State::decode(truncated, minimum_cycles, "test::tagging-graph@1", domains),
               std::invalid_argument);
}

TEST(ComponentInterfaces, PersistentTaggingStateUsesExactRankAndCanonicalRestartImage) {
  verify_persistent_tagging_state<1>();
  verify_persistent_tagging_state<2>();
  verify_persistent_tagging_state<3>();

  using Line = pops::runtime::amr::PersistentTaggingState<1>;
  Line line;
  line.begin_cycle(2);
  line.record({0, pops::Index<1>{1}}, Line::Decision::Refine, 2);
  const auto image = line.encode(2, "test::ranked-tagging@1");
  EXPECT_THROW((void)pops::runtime::amr::PersistentTaggingState<2>::decode(
                   image, 2, "test::ranked-tagging@1",
                   {pops::Box<2>{pops::Index<2>{0, 0}, pops::Index<2>{7, 7}}}),
               std::invalid_argument);
}

TEST(ComponentInterfaces, SolveReportCarriesTypedIncompatibleRhsReason) {
  pops::SolveReport report;
  report.mark_failed(pops::SolveStatus::kIncompatibleRhs, pops::SolveAction::kRejectAttempt,
                     "RHS violates the authenticated nullspace compatibility condition");
  EXPECT_TRUE(report.valid());
  EXPECT_FALSE(report.solved());
  EXPECT_STREQ(report.status_name(), "incompatible_rhs");
  EXPECT_STREQ(report.action_name(), "reject_attempt");
  EXPECT_EQ(report.reason, "RHS violates the authenticated nullspace compatibility condition");
}

TEST(ComponentInterfaces, RegistryIsCollisionSafeIdempotentAndExplicitlyFrozen) {
  pops::component::Registry registry;
  const auto& first = registry.register_component(record("pops://external.test/flux@1.0.0", "s1"));
  EXPECT_EQ(first.component_type, "test.external");
  EXPECT_EQ(registry.revision(), 1u);

  const auto& repeated =
      registry.register_component(record("pops://external.test/flux@1.0.0", "s1"));
  EXPECT_EQ(&first, &repeated);
  EXPECT_EQ(registry.revision(), 1u);

  EXPECT_THROW(registry.register_component(record("pops://external.test/flux@1.0.0", "different")),
               std::invalid_argument);
  EXPECT_EQ(registry.revision(), 1u);

  registry.freeze();
  EXPECT_TRUE(registry.frozen());
  EXPECT_THROW(registry.register_component(record("pops://external.test/writer@1.0.0", "s2")),
               std::logic_error);
}

template <int Dim>
void verify_prepared_tagging_execution() {
  using Program = pops::runtime::amr::PreparedTaggingProgram<Dim>;
  using Plan = pops::runtime::amr::PreparedTaggingExecutionPlan<Dim>;
  using Field = pops::runtime::amr::PreparedTaggingField<
      Dim, typename Kokkos::DefaultExecutionSpace::memory_space>;
  const pops::Box<Dim> domain = tagging_domain<Dim>();
  const auto layout = tagging_layout(domain);
  const pops::Index<Dim> local_rank = layout.distribution().rank_space().coordinate(
      static_cast<std::size_t>(pops::world_communicator_view().rank()));
  pops::Extent<Dim> ghosts{};
  std::array<pops::Real, Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis) {
    ghosts[axis] = 1;
    spacing[axis] = pops::Real(1);
  }
  pops::MultiFab<Dim> scalar(layout.patches(), layout.distribution(), local_rank, 1, ghosts);
  pops::MultiFab<Dim> vector(layout.patches(), layout.distribution(), local_rank, 2, ghosts);
  const auto refill = [&] {
    for (std::size_t local = 0; local < scalar.local_size(); ++local)
      pops::for_each_cell(
          scalar.fab(local).grown_box(),
          FillPreparedTaggingFields<Dim>{scalar.fab(local).view(), vector.fab(local).view()});
    pops::device_fence();
  };
  refill();

  Program program;
  typename Program::Stencil centered;
  centered.identity = "test::centered-gradient";
  centered.route = POPS_TAGGING_STENCIL_ROUTE_LINEAR_AXIS_STENCIL_L2_V1;
  centered.norm = "l2";
  centered.scale = "inverse_cell_size";
  centered.boundary_mode = "ghost_extension";
  for (int axis = 0; axis < Dim; ++axis)
    centered.axes[static_cast<std::size_t>(axis)] =
        typename Program::AxisStencil{axis, 1, 2, 1, 1, {-1, 1}, {-0.5, 0.5}};
  program.stencils = {std::move(centered)};
  program.leaves = {
      typename Program::Leaf{0, 0, POPS_TAGGING_GRADIENT_ABOVE_V1, 0.5, 0},
      typename Program::Leaf{1, 1, POPS_TAGGING_ABOVE_V1, 0.0, POPS_TAGGING_NO_STENCIL_V1},
      typename Program::Leaf{1, 1, POPS_TAGGING_BELOW_V1, 0.0, POPS_TAGGING_NO_STENCIL_V1}};
  program.refine_ops = {POPS_TAGGING_GRADIENT_ABOVE_V1, POPS_TAGGING_ABOVE_V1,
                        POPS_TAGGING_ALL_OF_V1};
  program.refine_args = {0, 1, 2};
  program.coarsen_ops = {POPS_TAGGING_BELOW_V1};
  program.coarsen_args = {2};
  program.clock_identity = "test::clock";
  program.provider_identity = "test::prepared-tagger";
  program.prepared = true;
  const std::vector<std::vector<Field>> fields{
      {{"case::scalar::U", &scalar}, {"case::vector::U", &vector}}};
  const std::vector<pops::amr::hierarchy::LevelLayout<Dim>> layouts{layout};
  const std::vector<pops::runtime::amr::PreparedTaggingExecutionBudget> budgets{
      tagging_budget(layout)};

  auto plan = Plan::prepare(program, fields, layouts, budgets, 7);
  const auto& first = plan.execute(0, layout, spacing, 7);
  for (std::size_t global = 0; global < layout.patches().size(); ++global)
    for_each_host_cell(layout.patches()[global], [&](const pops::Index<Dim>& index) {
      int sum = 0;
      for (int axis = 0; axis < Dim; ++axis)
        sum += index[axis];
      EXPECT_EQ(first.refine.tagged(global, index), sum > 0);
      EXPECT_EQ(first.refine_equalities.tagged(global, index), sum == 0);
      EXPECT_EQ(first.coarsen.tagged(global, index), sum < 0);
      EXPECT_EQ(first.coarsen_equalities.tagged(global, index), sum == 0);
    });

  const auto allocations_before = pops::allocation_event_stats();
  const std::uint64_t consensus_allocations_before = pops::exact_consensus_dynamic_storage_calls();
  const auto& second = plan.execute(0, layout, spacing, 7);
  const auto allocations_after = pops::allocation_event_stats();
  const std::uint64_t consensus_allocations_after = pops::exact_consensus_dynamic_storage_calls();
  EXPECT_EQ(second.refine.count(), first.refine.count());
  EXPECT_EQ(allocations_after.fab_calls, allocations_before.fab_calls)
      << "fixed-topology tagging must not rematerialize device storage";
  EXPECT_EQ(allocations_after.communication_calls, allocations_before.communication_calls);
  EXPECT_EQ(consensus_allocations_after, consensus_allocations_before)
      << "the tagging hot path must use only prepared scalar or fixed-buffer collectives";

  const std::size_t expected_coarsen = second.coarsen.count();
  scalar.set_val(pops::Real(0));
  const auto& without_gradient = plan.execute(0, layout, spacing, 7);
  EXPECT_EQ(without_gradient.refine.count(), 0u);
  EXPECT_EQ(without_gradient.coarsen.count(), expected_coarsen);

  refill();
  vector.set_val(pops::Real(-1));
  const auto& without_positive_value = plan.execute(0, layout, spacing, 7);
  EXPECT_EQ(without_positive_value.refine.count(), 0u);
  EXPECT_EQ(without_positive_value.coarsen.count(), static_cast<std::size_t>(domain.numPts()));
  refill();

  Program magnitude = program;
  magnitude.stencils.clear();
  magnitude.leaves = {typename Program::Leaf{1, 0, POPS_TAGGING_MAGNITUDE_ABOVE_V1, 17.0,
                                             POPS_TAGGING_NO_STENCIL_V1}};
  magnitude.refine_ops = {POPS_TAGGING_MAGNITUDE_ABOVE_V1};
  magnitude.refine_args = {0};
  magnitude.coarsen_ops.clear();
  magnitude.coarsen_args.clear();
  auto magnitude_plan = Plan::prepare(magnitude, fields, layouts, budgets, 8);
  const auto& magnitude_equality = magnitude_plan.execute(0, layout, spacing, 8);
  EXPECT_EQ(magnitude_equality.refine.count(), 0u);
  EXPECT_EQ(magnitude_equality.refine_equalities.count(),
            static_cast<std::size_t>(domain.numPts()));

  magnitude.leaves[0].threshold = 16.0;
  auto magnitude_strict_plan = Plan::prepare(magnitude, fields, layouts, budgets, 9);
  EXPECT_EQ(magnitude_strict_plan.execute(0, layout, spacing, 9).refine.count(),
            static_cast<std::size_t>(domain.numPts()));

  Program gradient_equality = program;
  gradient_equality.leaves = {typename Program::Leaf{0, 0, POPS_TAGGING_GRADIENT_ABOVE_V1, 1.0, 0}};
  gradient_equality.refine_ops = {POPS_TAGGING_GRADIENT_ABOVE_V1};
  gradient_equality.refine_args = {0};
  gradient_equality.coarsen_ops.clear();
  gradient_equality.coarsen_args.clear();
  auto gradient_equality_plan = Plan::prepare(gradient_equality, fields, layouts, budgets, 10);
  const auto& gradient_boundary = gradient_equality_plan.execute(0, layout, spacing, 10);
  EXPECT_EQ(gradient_boundary.refine.count(), 0u);
  EXPECT_EQ(gradient_boundary.refine_equalities.count(), static_cast<std::size_t>(domain.numPts()));

  Program logical = program;
  logical.stencils.clear();
  logical.leaves = {
      typename Program::Leaf{0, 0, POPS_TAGGING_ABOVE_V1, 0.0, POPS_TAGGING_NO_STENCIL_V1},
      typename Program::Leaf{0, 0, POPS_TAGGING_BELOW_V1, 0.0, POPS_TAGGING_NO_STENCIL_V1}};
  logical.coarsen_ops.clear();
  logical.coarsen_args.clear();
  const std::size_t tangent_cells =
      static_cast<std::size_t>(domain.numPts()) / static_cast<std::size_t>(domain.length(0));

  logical.refine_ops = {POPS_TAGGING_ABOVE_V1, POPS_TAGGING_NOT_V1};
  logical.refine_args = {0, 1};
  auto not_plan = Plan::prepare(logical, fields, layouts, budgets, 11);
  const auto& not_result = not_plan.execute(0, layout, spacing, 11);
  EXPECT_EQ(not_result.refine.count(), 3u * tangent_cells);
  EXPECT_EQ(not_result.refine_equalities.count(), tangent_cells);

  logical.refine_ops = {POPS_TAGGING_ABOVE_V1, POPS_TAGGING_BELOW_V1, POPS_TAGGING_ANY_OF_V1};
  logical.refine_args = {0, 1, 2};
  auto any_plan = Plan::prepare(logical, fields, layouts, budgets, 12);
  const auto& any_result = any_plan.execute(0, layout, spacing, 12);
  EXPECT_EQ(any_result.refine.count(), 5u * tangent_cells);
  EXPECT_EQ(any_result.refine_equalities.count(), tangent_cells);

  logical.refine_ops.back() = POPS_TAGGING_ALL_OF_V1;
  auto all_plan = Plan::prepare(logical, fields, layouts, budgets, 13);
  const auto& all_result = all_plan.execute(0, layout, spacing, 13);
  EXPECT_EQ(all_result.refine.count(), 0u);
  EXPECT_EQ(all_result.refine_equalities.count(), tangent_cells);
}

TEST(ComponentInterfaces, PreparedTaggingExecutionIsExactRankedAndAllocationFree) {
  verify_prepared_tagging_execution<1>();
  verify_prepared_tagging_execution<2>();
  verify_prepared_tagging_execution<3>();
}

template <int Dim>
void verify_prescribed_window_execution() {
  using Program = pops::runtime::amr::PreparedTaggingProgram<Dim>;
  using Plan = pops::runtime::amr::PreparedTaggingExecutionPlan<Dim>;
  using Field = pops::runtime::amr::PreparedTaggingField<
      Dim, typename Kokkos::DefaultExecutionSpace::memory_space>;
  const auto layout = prescribed_window_layout<Dim>();
  const pops::Index<Dim> local_rank = layout.distribution().rank_space().coordinate(
      static_cast<std::size_t>(pops::world_communicator_view().rank()));
  pops::Extent<Dim> ghosts{};
  std::array<pops::Real, Dim> spacing{};
  std::array<pops::Real, Dim> lower{};
  std::array<pops::Real, Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    spacing[axis] = pops::Real(1);
    upper[axis] = pops::Real(4);
  }
  pops::MultiFab<Dim> carrier(layout.patches(), layout.distribution(), local_rank, 1, ghosts);
  Program program;
  typename Program::Leaf window;
  window.state_index = 0;
  window.component = 0;
  window.opcode = POPS_TAGGING_PRESCRIBED_WINDOW_V1;
  window.stencil_index = POPS_TAGGING_NO_STENCIL_V1;
  for (int axis = 0; axis < Dim; ++axis) {
    window.window_center[axis] = 0.5;
    window.window_half_width[axis] = 0.49;
  }
  window.window_velocity[0] = 1.0;
  program.leaves = {window};
  program.refine_ops = {POPS_TAGGING_PRESCRIBED_WINDOW_V1};
  program.refine_args = {0};
  program.coarsen_ops = {POPS_TAGGING_PRESCRIBED_WINDOW_V1, POPS_TAGGING_NOT_V1};
  program.coarsen_args = {0, 1};
  program.clock_identity = "test::prescribed-window-clock";
  program.provider_identity = "test::prescribed-window";
  program.prepared = true;
  const std::vector<std::vector<Field>> fields{{{"test::carrier", &carrier}}};
  const std::vector<pops::amr::hierarchy::LevelLayout<Dim>> layouts{layout};
  const std::vector<pops::runtime::amr::PreparedTaggingExecutionBudget> budgets{
      tagging_budget(layout)};
  const auto assert_exact_refine_location = [&layout](const auto& result,
                                                      const pops::Index<Dim>& expected) {
    for (std::size_t global = 0; global < layout.patches().size(); ++global)
      for_each_host_cell(layout.patches()[global], [&](const pops::Index<Dim>& index) {
        EXPECT_EQ(result.refine.tagged(global, index), index == expected)
            << "the prescribed window must tag exactly its physical cell";
      });
  };
  auto plan = Plan::prepare(program, fields, layouts, budgets, 101);
  const auto& at_zero = plan.execute(0, layout, spacing, lower, upper, 0, pops::Real(0), 101);
  EXPECT_EQ(at_zero.refine.count(), 1u);
  pops::Index<Dim> origin{};
  assert_exact_refine_location(at_zero, origin);
  EXPECT_EQ(at_zero.coarsen.count(), static_cast<std::size_t>(layout.domain().numPts() - 1));
  const auto& at_one = plan.execute(0, layout, spacing, lower, upper, 0, pops::Real(1), 101);
  EXPECT_EQ(at_one.refine.count(), 1u);
  pops::Index<Dim> one_step{};
  one_step[0] = 1;
  assert_exact_refine_location(at_one, one_step);

  // Exercise each existing Cartesian axis.  At t=0 the centre lies in the last cell on that
  // axis; one period later it has crossed the upper periodic face and must tag the origin cell.
  for (int axis = 0; axis < Dim; ++axis) {
    Program wrapped = program;
    for (int component = 0; component < Dim; ++component)
      wrapped.leaves[0].window_velocity[component] = 0.0;
    wrapped.leaves[0].window_center[axis] = 3.5;
    wrapped.leaves[0].window_velocity[axis] = 1.0;
    const std::uint32_t periodic_axis = std::uint32_t{1} << static_cast<unsigned>(axis);
    auto wrapped_plan = Plan::prepare(wrapped, fields, layouts, budgets, 102 + axis);
    const auto& before_wrap = wrapped_plan.execute(0, layout, spacing, lower, upper, periodic_axis,
                                                   pops::Real(0), 102 + axis);
    pops::Index<Dim> upper_cell{};
    upper_cell[axis] = 3;
    EXPECT_EQ(before_wrap.refine.count(), 1u);
    assert_exact_refine_location(before_wrap, upper_cell);
    const auto& after_wrap = wrapped_plan.execute(0, layout, spacing, lower, upper, periodic_axis,
                                                  pops::Real(1), 102 + axis);
    EXPECT_EQ(after_wrap.refine.count(), 1u);
    assert_exact_refine_location(after_wrap, origin);
  }
  EXPECT_THROW((void)plan.execute(0, layout, spacing, lower, upper, std::uint32_t{1} << Dim,
                                  pops::Real(1), 101),
               std::runtime_error);

  Program degenerate = program;
  degenerate.leaves[0].window_half_width[0] = 2.0;
  auto degenerate_plan = Plan::prepare(degenerate, fields, layouts, budgets, 103);
  EXPECT_THROW(
      (void)degenerate_plan.execute(0, layout, spacing, lower, upper, 0, pops::Real(0), 103),
      std::runtime_error);

  Program overflowing = program;
  overflowing.leaves[0].window_center[0] = std::numeric_limits<double>::max();
  overflowing.leaves[0].window_velocity[0] = std::numeric_limits<double>::max();
  auto overflowing_plan = Plan::prepare(overflowing, fields, layouts, budgets, 104);
  EXPECT_THROW(
      (void)overflowing_plan.execute(0, layout, spacing, lower, upper, 0, pops::Real(2), 104),
      std::runtime_error);
}

TEST(ComponentInterfaces, PrescribedWindowUsesTimeAndExactPeriodicity) {
  verify_prescribed_window_execution<1>();
  verify_prescribed_window_execution<2>();
  verify_prescribed_window_execution<3>();
}

template <int Dim>
void verify_prepared_tagging_failure_is_transactional() {
  using Program = pops::runtime::amr::PreparedTaggingProgram<Dim>;
  using Plan = pops::runtime::amr::PreparedTaggingExecutionPlan<Dim>;
  using Field = pops::runtime::amr::PreparedTaggingField<
      Dim, typename Kokkos::DefaultExecutionSpace::memory_space>;
  const pops::Box<Dim> domain = tagging_domain<Dim>();
  const auto layout = tagging_layout(domain);
  const pops::Index<Dim> local_rank = layout.distribution().rank_space().coordinate(
      static_cast<std::size_t>(pops::world_communicator_view().rank()));
  pops::Extent<Dim> ghosts{};
  std::array<pops::Real, Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis) {
    ghosts[axis] = 1;
    spacing[axis] = pops::Real(1);
  }
  pops::MultiFab<Dim> state(layout.patches(), layout.distribution(), local_rank, 1, ghosts);
  state.set_val(pops::Real(1));

  Program program;
  program.leaves = {
      typename Program::Leaf{0, 0, POPS_TAGGING_ABOVE_V1, 0.0, POPS_TAGGING_NO_STENCIL_V1}};
  program.refine_ops = {POPS_TAGGING_ABOVE_V1};
  program.refine_args = {0};
  program.clock_identity = "test::clock";
  program.provider_identity = "test::prepared-tagger";
  program.prepared = true;
  const std::vector<std::vector<Field>> fields{{{"case::state::U", &state}}};
  const std::vector<pops::amr::hierarchy::LevelLayout<Dim>> layouts{layout};
  const std::vector<pops::runtime::amr::PreparedTaggingExecutionBudget> budgets{
      tagging_budget(layout)};
  auto plan = Plan::prepare(program, fields, layouts, budgets, 3);
  const auto& accepted = plan.execute(0, layout, spacing, 3);
  EXPECT_EQ(accepted.refine.count(), static_cast<std::size_t>(domain.numPts()));

  const pops::Index<Dim> bad = layout.patches()[0].lo;
  pops::for_each_cell(pops::Box<Dim>{bad, bad},
                      SetPreparedTaggingSample<Dim>{state.fab_global(0).view(),
                                                    std::numeric_limits<pops::Real>::quiet_NaN()});
  pops::device_fence();
  EXPECT_THROW((void)plan.execute(0, layout, spacing, 3), std::runtime_error);
  EXPECT_EQ(accepted.refine.count(), static_cast<std::size_t>(domain.numPts()))
      << "a rejected evaluation must retain the prior accepted mask";
  EXPECT_THROW((void)plan.execute(0, layout, spacing, 4), std::runtime_error);
  state.set_val(pops::Real(1));

  auto malformed = program;
  malformed.refine_args = {7};
  EXPECT_THROW((void)Plan::prepare(malformed, fields, layouts, budgets, 5), std::invalid_argument);

  auto gradient = program;
  typename Program::Stencil centered;
  centered.identity = "test::centered-gradient";
  centered.route = POPS_TAGGING_STENCIL_ROUTE_LINEAR_AXIS_STENCIL_L2_V1;
  centered.norm = "l2";
  centered.scale = "inverse_cell_size";
  centered.boundary_mode = "ghost_extension";
  for (int axis = 0; axis < Dim; ++axis)
    centered.axes[static_cast<std::size_t>(axis)] =
        typename Program::AxisStencil{axis, 1, 2, 1, 1, {-1, 1}, {-0.5, 0.5}};
  gradient.stencils = {std::move(centered)};
  gradient.leaves[0].opcode = POPS_TAGGING_GRADIENT_ABOVE_V1;
  gradient.leaves[0].stencil_index = 0;
  gradient.refine_ops = {POPS_TAGGING_GRADIENT_ABOVE_V1};

  auto false_order = gradient;
  false_order.stencils[0].axes[0] =
      typename Program::AxisStencil{0, 1, 2, 0, 1, {0, 1}, {-1.0, 1.0}};
  EXPECT_THROW((void)Plan::prepare(false_order, fields, layouts, budgets, 6), std::invalid_argument)
      << "a stencil may not claim an order its exact moments do not prove";

  auto repeated_offset = gradient;
  repeated_offset.stencils[0].axes[0] =
      typename Program::AxisStencil{0, 1, 1, 0, 0, {0, 0}, {-1.0, 1.0}};
  EXPECT_THROW((void)Plan::prepare(repeated_offset, fields, layouts, budgets, 6),
               std::invalid_argument);

  auto minimum_offset = gradient;
  minimum_offset.stencils[0].axes[0] = typename Program::AxisStencil{
      0, 1, 1, 0, 0, {std::numeric_limits<std::int32_t>::min(), 0}, {-1.0, 1.0}};
  EXPECT_THROW((void)Plan::prepare(minimum_offset, fields, layouts, budgets, 6),
               std::invalid_argument)
      << "INT_MIN stencil offsets must reject without signed overflow";

  auto overflow = gradient;
  for (int axis = 0; axis < Dim; ++axis)
    overflow.stencils[0].axes[static_cast<std::size_t>(axis)] =
        typename Program::AxisStencil{axis, 1, 1, 0, 1, {0, 1}, {-1.0, 1.0}};
  auto overflow_plan = Plan::prepare(overflow, fields, layouts, budgets, 6);
  const auto& finite_gradient = overflow_plan.execute(0, layout, spacing, 6);
  EXPECT_EQ(finite_gradient.refine.count(), 0u);
  pops::Index<Dim> adjacent = bad;
  ++adjacent[0];
  pops::for_each_cell(pops::Box<Dim>{bad, bad},
                      SetPreparedTaggingSample<Dim>{state.fab_global(0).view(),
                                                    -std::numeric_limits<pops::Real>::max()});
  pops::for_each_cell(pops::Box<Dim>{adjacent, adjacent},
                      SetPreparedTaggingSample<Dim>{state.fab_global(0).view(),
                                                    std::numeric_limits<pops::Real>::max()});
  pops::device_fence();
  EXPECT_THROW((void)overflow_plan.execute(0, layout, spacing, 6), std::runtime_error);
  EXPECT_EQ(finite_gradient.refine.count(), 0u)
      << "a derived non-finite indicator must not publish partial masks";

  for (int axis = 0; axis < Dim; ++axis)
    gradient.stencils[0].axes[static_cast<std::size_t>(axis)] =
        typename Program::AxisStencil{axis, 1, 2, 2, 2, {-2, 2}, {-0.25, 0.25}};
  EXPECT_THROW((void)Plan::prepare(gradient, fields, layouts, budgets, 6), std::invalid_argument)
      << "a prepared stencil must fit every axis of the bound field halo";
}

TEST(ComponentInterfaces, PreparedTaggingRejectsNonFiniteBeforePublishingMasks) {
  verify_prepared_tagging_failure_is_transactional<1>();
  verify_prepared_tagging_failure_is_transactional<2>();
  verify_prepared_tagging_failure_is_transactional<3>();
}

PopsComponentTableHeaderV1 abi_header(std::size_t size, PopsNativeInterfaceIdV1 id,
                                      std::uint32_t version = 1) {
  return {static_cast<std::uint32_t>(size),
          POPS_COMPONENT_PROTOCOL_ABI_V1,
          id,
          version,
          nullptr,
          nullptr};
}

PopsComponentStatusV1 ok_status() {
  return {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
}

const double* values(const PopsConstFieldViewV1& view) {
  return static_cast<const double*>(view.data);
}

double* values(PopsFieldViewV1& view) {
  return static_cast<double*>(view.data);
}

template <int Dim>
PopsFieldPatchMetadataV1 material_patch_metadata(
    const std::array<std::size_t, Dim>& interior_extents) {
  static_assert(Dim >= 1 && Dim <= 3);
  PopsFieldPatchMetadataV1 metadata{sizeof(PopsFieldPatchMetadataV1),
                                    0,
                                    0,
                                    0,
                                    Dim,
                                    {},
                                    {},
                                    {},
                                    {},
                                    POPS_FIELD_CENTERING_CELL_V1,
                                    0,
                                    "test::ranked-material-layout",
                                    "test::ranked-material-patch"};
  for (int axis = 0; axis < Dim; ++axis) {
    metadata.upper[axis] = static_cast<std::int64_t>(interior_extents[axis] - 1);
    metadata.physical_lower[axis] = -0.5 * static_cast<double>(axis + 1);
    metadata.cell_spacing[axis] = 0.125 * static_cast<double>(axis + 1);
  }
  return metadata;
}

template <int Dim>
PopsConstFieldViewV1 strided_fraction_view(const double* data,
                                           const std::array<std::size_t, Dim>& interior_extents,
                                           const std::array<std::size_t, Dim>& ghost_lower,
                                           const std::array<std::size_t, Dim>& ghost_upper,
                                           const std::array<std::ptrdiff_t, Dim>& axis_strides) {
  static_assert(Dim >= 1 && Dim <= 3);
  PopsConstFieldViewV1 view{sizeof(PopsConstFieldViewV1),
                            data,
                            Dim,
                            {1, 1, 1},
                            {},
                            1,
                            1,
                            POPS_FIELD_CENTERING_CELL_V1,
                            0,
                            {},
                            {},
                            POPS_SCALAR_FLOAT64_V1,
                            POPS_MEMORY_SPACE_HOST_V1,
                            "test::ranked-material-layout",
                            "test::ranked-material-patch",
                            POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
  for (int axis = 0; axis < Dim; ++axis) {
    view.extents[axis] = interior_extents[axis] + ghost_lower[axis] + ghost_upper[axis];
    view.axis_strides[axis] = axis_strides[axis];
    view.ghost_lower[axis] = ghost_lower[axis];
    view.ghost_upper[axis] = ghost_upper[axis];
  }
  return view;
}

PopsFieldViewV1 mutable_view_like(double* data, const PopsConstFieldViewV1& source) {
  PopsFieldViewV1 result{sizeof(PopsFieldViewV1),
                         data,
                         source.dimension,
                         {},
                         {},
                         source.component_count,
                         source.component_stride,
                         source.centering,
                         source.centering_axes,
                         {},
                         {},
                         source.scalar_type,
                         source.memory_space,
                         source.layout_identity,
                         source.patch_identity,
                         source.ownership};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    result.extents[axis] = source.extents[axis];
    result.axis_strides[axis] = source.axis_strides[axis];
    result.ghost_lower[axis] = source.ghost_lower[axis];
    result.ghost_upper[axis] = source.ghost_upper[axis];
  }
  return result;
}

template <int Dim>
void verify_ranked_material_fraction_mask(const std::array<std::size_t, Dim>& interior_extents,
                                          const std::array<std::size_t, Dim>& ghost_lower,
                                          const std::array<std::size_t, Dim>& ghost_upper,
                                          const std::array<std::ptrdiff_t, Dim>& axis_strides) {
  const auto metadata = material_patch_metadata<Dim>(interior_extents);
  std::size_t point_count = 1;
  std::size_t maximum_offset = 0;
  for (int axis = 0; axis < Dim; ++axis) {
    point_count *= interior_extents[axis];
    maximum_offset += (ghost_lower[axis] + interior_extents[axis] - 1) *
                      static_cast<std::size_t>(axis_strides[axis]);
  }
  std::vector<double> storage(maximum_offset + 1, 0.0);
  std::vector<std::uint8_t> expected(point_count, 0);
  for (std::size_t point = 0; point < point_count; ++point) {
    std::size_t remaining = point;
    std::size_t offset = 0;
    for (int axis = 0; axis < Dim; ++axis) {
      const std::size_t coordinate = remaining % interior_extents[axis];
      remaining /= interior_extents[axis];
      offset += (coordinate + ghost_lower[axis]) * static_cast<std::size_t>(axis_strides[axis]);
    }
    expected[point] = ((point * 7 + static_cast<std::size_t>(Dim)) % 5) != 0 ? 1u : 0u;
    storage[offset] = expected[point] != 0 ? 0.625 : 0.0;
  }
  const auto view = strided_fraction_view<Dim>(storage.data(), interior_extents, ghost_lower,
                                               ghost_upper, axis_strides);
  const pops::component::FieldTopologyPatchInputV2 input{
      0, POPS_FIELD_MATERIAL_CUT_CELL_FRACTION_V1, {}, view, {}};
  EXPECT_EQ(pops::component::expected_topology_material_mask(input, metadata), expected);
}

TEST(ComponentInterfaces, TopologyMaterialFractionsUseExactRankedStridesAndGhosts) {
  verify_ranked_material_fraction_mask<1>({4}, {2}, {1}, {3});
  verify_ranked_material_fraction_mask<2>({3, 2}, {1, 2}, {2, 1}, {2, 17});
  verify_ranked_material_fraction_mask<3>({2, 3, 2}, {1, 1, 2}, {2, 1, 1}, {3, 19, 113});
}

TEST(ComponentInterfaces, TopologyMaterialFractionsFailClosedOnRankAndStrideOverflow) {
  const std::array<double, 1> storage{0.5};
  auto metadata = material_patch_metadata<1>({2});
  auto view = strided_fraction_view<1>(storage.data(), {2}, {2}, {1},
                                       {std::numeric_limits<std::ptrdiff_t>::max()});
  pops::component::FieldTopologyPatchInputV2 input{
      0, POPS_FIELD_MATERIAL_CUT_CELL_FRACTION_V1, {}, view, {}};
  EXPECT_THROW((void)pops::component::expected_topology_material_mask(input, metadata),
               std::invalid_argument);

  auto invalid_rank = metadata;
  invalid_rank.dimension = 4;
  EXPECT_THROW((void)pops::component::expected_topology_material_mask(input, invalid_rank),
               std::invalid_argument);
  invalid_rank.dimension = 0;
  EXPECT_THROW((void)pops::component::expected_topology_material_mask(input, invalid_rank),
               std::invalid_argument);

  auto hidden_metadata_axis = metadata;
  hidden_metadata_axis.upper[1] = 1;
  EXPECT_THROW((void)pops::component::expected_topology_material_mask(input, hidden_metadata_axis),
               std::invalid_argument);
  auto hidden_view_axis = view;
  hidden_view_axis.axis_strides[1] = 1;
  input.cut_cell_volume_fraction = hidden_view_axis;
  EXPECT_THROW((void)pops::component::expected_topology_material_mask(input, metadata),
               std::invalid_argument);
}

TEST(ComponentInterfaces, BoundaryFluxValidatesCoordinatesAndNormalsAcrossExactRank) {
  constexpr std::array<std::size_t, 3> extents{2, 2, 2};
  constexpr std::array<std::size_t, 3> no_ghosts{};
  constexpr std::array<std::ptrdiff_t, 3> flux_strides{2, 13, 41};
  constexpr std::array<std::ptrdiff_t, 3> coordinate_strides{7, 29, 83};
  constexpr std::array<std::ptrdiff_t, 3> normal_strides{5, 23, 67};
  std::array<double, 57> base_flux{};
  std::array<double, 57> transformed_flux{};
  std::array<double, 124> coordinates{};
  std::array<double, 100> normals{};
  std::array<double, 8> face_measures{};
  std::array<PopsComponentActionV1, 8> actions{};

  auto base_view =
      strided_fraction_view<3>(base_flux.data(), extents, no_ghosts, no_ghosts, flux_strides);
  base_view.centering = POPS_FIELD_CENTERING_FACE_V1;
  base_view.centering_axes = 1u << 2u;
  auto coordinate_view = strided_fraction_view<3>(coordinates.data(), extents, no_ghosts, no_ghosts,
                                                  coordinate_strides);
  coordinate_view.component_count = 3;
  coordinate_view.component_stride = 2;
  auto normal_view =
      strided_fraction_view<3>(normals.data(), extents, no_ghosts, no_ghosts, normal_strides);
  normal_view.component_count = 3;
  normal_view.component_stride = 2;

  for (std::size_t point = 0; point < face_measures.size(); ++point) {
    face_measures[point] = 0.25 + 0.01 * static_cast<double>(point);
    std::size_t remaining = point;
    std::array<std::size_t, 3> coordinate{};
    for (std::size_t axis = 0; axis < coordinate.size(); ++axis) {
      coordinate[axis] = remaining % extents[axis];
      remaining /= extents[axis];
    }
    for (std::size_t component = 0; component < coordinate.size(); ++component) {
      std::size_t coordinate_offset = component * 2;
      std::size_t normal_offset = component * 2;
      for (std::size_t axis = 0; axis < coordinate.size(); ++axis) {
        coordinate_offset += coordinate[axis] * static_cast<std::size_t>(coordinate_strides[axis]);
        normal_offset += coordinate[axis] * static_cast<std::size_t>(normal_strides[axis]);
      }
      coordinates[coordinate_offset] = static_cast<double>(10 * component + coordinate[component]);
      normals[normal_offset] = component == 2 ? 1.0 : 0.0;
    }
  }

  auto transformed_view = mutable_view_like(transformed_flux.data(), base_view);

  const std::array<std::int32_t, 1> axes{2};
  const std::array<std::int32_t, 1> sides{1};
  const PopsBoundaryRegionV1 region{sizeof(PopsBoundaryRegionV1),
                                    POPS_BOUNDARY_FACE_V1,
                                    3,
                                    1,
                                    axes.size(),
                                    axes.data(),
                                    sides.data(),
                                    "z-high"};
  PopsBoundaryFluxApiV1 api{
      abi_header(sizeof(PopsBoundaryFluxApiV1), POPS_NATIVE_INTERFACE_BOUNDARY_FLUX_V1),
      +[](void*, const PopsBoundaryFluxRequestV1* request, PopsBoundaryFluxResultV1* result) {
        const std::size_t points =
            pops::component::field_point_count(request->base_outward_normal_flux);
        std::fill(result->actions, result->actions + points, POPS_COMPONENT_CONTINUE_V1);
        result->status = ok_status();
        return 0;
      }};
  PopsBoundaryFluxRequestV1 request{sizeof(PopsBoundaryFluxRequestV1),
                                    "test::ranked-boundary-flux",
                                    "test::ranked-state",
                                    base_view,
                                    coordinate_view,
                                    normal_view,
                                    face_measures.data(),
                                    region,
                                    0,
                                    nullptr,
                                    0,
                                    nullptr,
                                    abi::logical_time(),
                                    abi::noncollective_host_execution_context()};
  PopsBoundaryFluxResultV1 result{
      sizeof(PopsBoundaryFluxResultV1), transformed_view, actions.data(), {}};
  EXPECT_EQ(pops::component::transform_boundary_flux(api, nullptr, request, result), 0);

  const std::size_t last_point_axis_two_normal_offset =
      static_cast<std::size_t>(normal_strides[0] + normal_strides[1] + normal_strides[2]) + 4;
  normals[last_point_axis_two_normal_offset] = -1.0;
  EXPECT_THROW(pops::component::transform_boundary_flux(api, nullptr, request, result),
               std::invalid_argument)
      << "the final z-slice must not escape ranked coordinate/normal validation";
}

TEST(ComponentInterfaces, RefluxBatchAcceptsAxisTwoAndRejectsAxesOutsideExactRank) {
  constexpr std::array<std::size_t, 3> extents{2, 3, 1};
  constexpr std::array<std::size_t, 3> no_ghosts{};
  constexpr std::array<std::ptrdiff_t, 3> strides{1, 2, 6};
  std::array<double, 6> coarse_values{};
  std::array<double, 6> fine_values{};
  std::array<double, 6> correction_values{};
  auto coarse =
      strided_fraction_view<3>(coarse_values.data(), extents, no_ghosts, no_ghosts, strides);
  coarse.centering = POPS_FIELD_CENTERING_FACE_V1;
  coarse.centering_axes = 1u << 2u;
  coarse.layout_identity = "test::reflux-parent-layout";
  coarse.patch_identity = "test::reflux-parent-patch";
  auto fine = strided_fraction_view<3>(fine_values.data(), extents, no_ghosts, no_ghosts, strides);
  fine.centering = POPS_FIELD_CENTERING_FACE_V1;
  fine.centering_axes = 1u << 2u;
  fine.layout_identity = "test::reflux-child-layout";
  fine.patch_identity = "test::reflux-child-patch";
  auto correction = mutable_view_like(correction_values.data(), coarse);
  correction.centering = POPS_FIELD_CENTERING_CELL_V1;
  correction.centering_axes = 0;

  PopsRefluxFaceV1 face{sizeof(PopsRefluxFaceV1),
                        "test::reflux-z-high",
                        2,
                        POPS_REFLUX_FACE_HIGH_V1,
                        4.0,
                        coarse,
                        fine,
                        correction};
  PopsRefluxRequestV1 request{sizeof(PopsRefluxRequestV1),
                              "test::reflux-transition",
                              0,
                              1,
                              1,
                              &face,
                              abi::logical_time(),
                              abi::noncollective_host_execution_context()};
  PopsRefluxApiV1 api{abi_header(sizeof(PopsRefluxApiV1), POPS_NATIVE_INTERFACE_REFLUX_V1),
                      +[](void*, const PopsRefluxRequestV1*, PopsComponentStatusV1* status) {
                        *status = ok_status();
                        return 0;
                      }};
  auto status = ok_status();
  EXPECT_EQ(pops::component::apply_reflux_interface_batch(api, nullptr, request, status), 0);

  face.axis = 3;
  EXPECT_THROW(pops::component::apply_reflux_interface_batch(api, nullptr, request, status),
               std::invalid_argument);
}

TEST(ComponentInterfaces, ExactAbiConsumersExecuteEveryClosedScientificFamily) {
  const PopsComponentStatusV1 malformed_component_action{sizeof(PopsComponentStatusV1), 0, 17,
                                                         "malformed component action"};
  EXPECT_FALSE(pops::component::component_status_is_well_formed(malformed_component_action));
  std::array<double, 2> left{2.0, 4.0}, right{6.0, 8.0}, normal{1.0, 0.0};
  const auto execution = abi::host_execution_context();
  const auto noncollective_execution = abi::noncollective_host_execution_context();
  EXPECT_NO_THROW(pops::component::validate_execution_context(execution));
  auto anonymous_execution = execution;
  anonymous_execution.execution_identity = "";
  EXPECT_THROW(pops::component::validate_execution_context(anonymous_execution),
               std::invalid_argument);
  auto packed_distributed = execution;
  // MPI_Comm_c2f/MPI_Type_c2f values are implementation-defined; zero is legal for a predefined
  // handle.  The exact identities, not the numeric values, select the distributed route.
  packed_distributed.communicator_f_handle = 0;
  packed_distributed.communicator_datatype_f_handle = 0;
  packed_distributed.communicator_identity = "MPI_COMM_WORLD";
  packed_distributed.communicator_datatype_identity = "MPI_DOUBLE";
  EXPECT_NO_THROW(pops::component::validate_execution_context(packed_distributed));
  auto ambiguous_distributed = packed_distributed;
  ambiguous_distributed.communicator_datatype_identity = "none";
  EXPECT_THROW(pops::component::validate_execution_context(ambiguous_distributed),
               std::invalid_argument);
  auto lane_distributed = packed_distributed;
  lane_distributed.communicator_identity = "case::solver::execution-lane";
  EXPECT_NO_THROW(pops::component::validate_execution_context(lane_distributed))
      << "typed duplicated/subset communicator identities are first-class ABI authorities";
  auto hidden_serial_handle = execution;
  hidden_serial_handle.communicator_f_handle = 7;
  EXPECT_THROW(pops::component::validate_execution_context(hidden_serial_handle),
               std::invalid_argument);
  auto anonymous_device = execution;
  anonymous_device.device_identity = "";
  EXPECT_THROW(pops::component::validate_execution_context(anonymous_device),
               std::invalid_argument);
  std::array<double, 2> flux{};
  double speed = 0.0;
  PopsComponentActionV1 action = POPS_COMPONENT_ABORT_RUN_V1;
  PopsNumericalFluxApiV1 flux_api{
      abi_header(sizeof(PopsNumericalFluxApiV1), POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1),
      +[](void*, const PopsNumericalFluxRequestV1* request, PopsNumericalFluxResultV1* result) {
        const auto* left_values = values(request->left);
        const auto* right_values = values(request->right);
        auto* output_values = values(result->normal_flux);
        for (std::size_t c = 0; c < request->left.component_count; ++c)
          output_values[c] = 0.5 * (left_values[c] + right_values[c]);
        result->stability_bounds[0] = 8.0;
        result->actions[0] = POPS_COMPONENT_CONTINUE_V1;
        result->status = ok_status();
        return 0;
      }};
  PopsNumericalFluxRequestV1 flux_request{sizeof(PopsNumericalFluxRequestV1),
                                          abi::const_field_view(left.data(), 1, 1, 2),
                                          abi::const_field_view(right.data(), 1, 1, 2),
                                          abi::const_field_view(normal.data(), 1, 1, 2),
                                          nullptr,
                                          abi::logical_time(),
                                          execution};
  PopsNumericalFluxResultV1 flux_result{sizeof(PopsNumericalFluxResultV1),
                                        abi::field_view(flux.data(), 1, 1, 2),
                                        &speed,
                                        &action,
                                        {}};
  EXPECT_EQ(pops::component::evaluate_faces(flux_api, nullptr, flux_request, flux_result), 0);
  EXPECT_EQ(flux, (std::array<double, 2>{4.0, 6.0}));
  EXPECT_DOUBLE_EQ(speed, 8.0);
  auto mismatched_patch = flux_request;
  mismatched_patch.right.patch_identity = "test::other-patch";
  EXPECT_THROW(pops::component::evaluate_faces(flux_api, nullptr, mismatched_patch, flux_result),
               std::invalid_argument);
  auto unsupported_dimension = flux_request;
  unsupported_dimension.left.dimension = 3;
  unsupported_dimension.left.extents[2] = 1;
  unsupported_dimension.left.axis_strides[2] = 1;
  EXPECT_THROW(
      pops::component::evaluate_faces(flux_api, nullptr, unsupported_dimension, flux_result),
      std::invalid_argument);
  auto invalid_time = flux_request;
  invalid_time.logical_time.fraction_denominator = 0;
  EXPECT_THROW(pops::component::evaluate_faces(flux_api, nullptr, invalid_time, flux_result),
               std::invalid_argument);

  std::array<double, 2> ghosts{};
  PopsGhostBoundaryApiV1 ghost_api{
      abi_header(sizeof(PopsGhostBoundaryApiV1), POPS_NATIVE_INTERFACE_GHOST_BOUNDARY_V1),
      +[](void*, const PopsGhostBoundaryRequestV1* request, PopsComponentStatusV1* status) {
        if ((request->region.kind != POPS_BOUNDARY_FACE_V1 &&
             request->region.kind != POPS_BOUNDARY_CORNER_V1) ||
            request->dependency_count != 1 || request->parameter_count != 1 ||
            std::strcmp(request->dependencies[0].qualified_id, "case::velocity") != 0 ||
            std::strcmp(request->parameters[0].qualified_id, "case::inlet") != 0)
          return 9;
        auto* ghosts = static_cast<double*>(request->ghosts.data);
        const auto* interior = static_cast<const double*>(request->interior.data);
        for (std::size_t c = 0; c < request->ghosts.component_count; ++c)
          ghosts[c] = -interior[c];
        *status = ok_status();
        return 0;
      }};
  const std::array<std::int32_t, 1> face_axes{0}, face_sides{-1};
  const PopsBoundaryRegionV1 face_region{sizeof(PopsBoundaryRegionV1),
                                         POPS_BOUNDARY_FACE_V1,
                                         2,
                                         1,
                                         face_axes.size(),
                                         face_axes.data(),
                                         face_sides.data(),
                                         "x-low"};
  const PopsQualifiedConstFieldV1 ghost_dependencies[] = {
      {sizeof(PopsQualifiedConstFieldV1), 1, "case::velocity",
       abi::const_field_view(normal.data(), 1, 1, 2)}};
  const PopsQualifiedScalarV1 ghost_parameters[] = {
      {sizeof(PopsQualifiedScalarV1), "case::inlet", 2.5}};
  PopsGhostBoundaryRequestV1 ghost_request{sizeof(PopsGhostBoundaryRequestV1),
                                           "case::ghost-producer",
                                           "case::state",
                                           "case::ghost-output",
                                           abi::const_field_view(left.data(), 1, 1, 2),
                                           abi::field_view(ghosts.data(), 1, 1, 2),
                                           abi::const_field_view(normal.data(), 1, 1, 2),
                                           face_region,
                                           1,
                                           ghost_dependencies,
                                           1,
                                           ghost_parameters,
                                           abi::logical_time(),
                                           noncollective_execution};
  auto status = ok_status();
  EXPECT_EQ(pops::component::apply_ghost_boundary(ghost_api, nullptr, ghost_request, status), 0);
  EXPECT_EQ(ghosts, (std::array<double, 2>{-2.0, -4.0}));
  const std::array<std::int32_t, 2> corner_axes{0, 1}, corner_sides{-1, 1};
  ghost_request.region = {sizeof(PopsBoundaryRegionV1),
                          POPS_BOUNDARY_CORNER_V1,
                          2,
                          2,
                          corner_axes.size(),
                          corner_axes.data(),
                          corner_sides.data(),
                          "x-low-y-high"};
  EXPECT_EQ(pops::component::apply_ghost_boundary(ghost_api, nullptr, ghost_request, status), 0);
  auto invalid_region = ghost_request;
  invalid_region.region.kind = POPS_BOUNDARY_FACE_V1;
  EXPECT_THROW(pops::component::apply_ghost_boundary(ghost_api, nullptr, invalid_region, status),
               std::invalid_argument);

  std::array<double, 2> transformed_outward_flux{};
  const std::array<double, 2> lower_outward_normal{-1.0, 0.0};
  const std::array<double, 1> face_measure{0.5};
  PopsComponentActionV1 boundary_flux_action = POPS_COMPONENT_ABORT_RUN_V1;
  PopsBoundaryFluxApiV1 boundary_flux_api{
      abi_header(sizeof(PopsBoundaryFluxApiV1), POPS_NATIVE_INTERFACE_BOUNDARY_FLUX_V1),
      +[](void*, const PopsBoundaryFluxRequestV1* request, PopsBoundaryFluxResultV1* result) {
        if (request->region.kind != POPS_BOUNDARY_FACE_V1 ||
            request->outward_normals.component_count != 2 ||
            values(request->outward_normals)[0] != -1.0 || request->face_measures[0] != 0.5)
          return 12;
        const auto* base = values(request->base_outward_normal_flux);
        auto* output = values(result->outward_normal_flux);
        for (std::size_t component = 0;
             component < request->base_outward_normal_flux.component_count; ++component)
          output[component] = base[component] + 3.0;
        result->actions[0] = POPS_COMPONENT_CONTINUE_V1;
        result->status = ok_status();
        return 0;
      }};
  auto base_outward_flux_view = abi::const_field_view(left.data(), 1, 1, 2);
  base_outward_flux_view.centering = POPS_FIELD_CENTERING_FACE_V1;
  base_outward_flux_view.centering_axes = 1u;
  auto transformed_outward_flux_view = abi::field_view(transformed_outward_flux.data(), 1, 1, 2);
  transformed_outward_flux_view.centering = POPS_FIELD_CENTERING_FACE_V1;
  transformed_outward_flux_view.centering_axes = 1u;
  PopsBoundaryFluxRequestV1 boundary_flux_request{
      sizeof(PopsBoundaryFluxRequestV1),
      "case::boundary-flux-provider",
      "case::state",
      base_outward_flux_view,
      abi::const_field_view(normal.data(), 1, 1, 2),
      abi::const_field_view(lower_outward_normal.data(), 1, 1, 2),
      face_measure.data(),
      face_region,
      0,
      nullptr,
      0,
      nullptr,
      abi::logical_time(),
      noncollective_execution};
  PopsBoundaryFluxResultV1 boundary_flux_result{
      sizeof(PopsBoundaryFluxResultV1), transformed_outward_flux_view, &boundary_flux_action, {}};
  EXPECT_EQ(pops::component::transform_boundary_flux(boundary_flux_api, nullptr,
                                                     boundary_flux_request, boundary_flux_result),
            0);
  EXPECT_EQ(transformed_outward_flux, (std::array<double, 2>{5.0, 7.0}));
  EXPECT_EQ(boundary_flux_action, POPS_COMPONENT_CONTINUE_V1);
  const std::array<double, 2> wrong_lower_normal{1.0, 0.0};
  auto wrong_orientation = boundary_flux_request;
  wrong_orientation.outward_normals = abi::const_field_view(wrong_lower_normal.data(), 1, 1, 2);
  EXPECT_THROW(pops::component::transform_boundary_flux(boundary_flux_api, nullptr,
                                                        wrong_orientation, boundary_flux_result),
               std::invalid_argument);
  auto mismatched_flux_output = boundary_flux_result;
  mismatched_flux_output.outward_normal_flux.component_count = 1;
  EXPECT_THROW(pops::component::transform_boundary_flux(
                   boundary_flux_api, nullptr, boundary_flux_request, mismatched_flux_output),
               std::invalid_argument);

  std::array<double, 2> direction{1.0, 2.0}, boundary_output{};
  const auto field_eval =
      +[](void*, const PopsFieldBoundaryRequestV1* request, PopsComponentStatusV1* result) {
        if (request->state_count != 1 || request->direction_count > 1 ||
            request->field_count != 1 || request->parameter_count != 1 ||
            request->output_count != 1 || request->level != 2 || request->logical_time.tick != 7 ||
            request->rate.present != 1 || request->nonlinear_iterate.present != 1 ||
            std::strcmp(request->fields[0].qualified_id, "case::coefficient") != 0)
          return 11;
        auto& output = request->outputs[0].values;
        auto* output_values = values(output);
        const auto* state_values = values(request->states[0].values);
        const auto* direction_values =
            request->direction_count == 0 ? nullptr : values(request->directions[0].values);
        for (std::size_t c = 0; c < output.component_count; ++c)
          output_values[c] =
              state_values[c] + (direction_values == nullptr ? 0.0 : direction_values[c]);
        *result = ok_status();
        return 0;
      };
  PopsFieldBoundaryClosureApiV1 field_boundary_api{
      abi_header(sizeof(PopsFieldBoundaryClosureApiV1),
                 POPS_NATIVE_INTERFACE_FIELD_BOUNDARY_CLOSURE_V1),
      field_eval, field_eval};
  const PopsQualifiedConstFieldV1 boundary_states[] = {
      {sizeof(PopsQualifiedConstFieldV1), 1, "case::state",
       abi::const_field_view(left.data(), 1, 1, 2)}};
  const PopsQualifiedConstFieldV1 boundary_directions[] = {
      {sizeof(PopsQualifiedConstFieldV1), 1, "case::normal-direction",
       abi::const_field_view(direction.data(), 1, 1, 2)}};
  const PopsQualifiedConstFieldV1 boundary_fields[] = {
      {sizeof(PopsQualifiedConstFieldV1), 1, "case::coefficient",
       abi::const_field_view(right.data(), 1, 1, 2)}};
  const PopsQualifiedScalarV1 boundary_parameters[] = {
      {sizeof(PopsQualifiedScalarV1), "case::robin-alpha", 0.5}};
  PopsQualifiedFieldV1 boundary_outputs[] = {{sizeof(PopsQualifiedFieldV1), "case::residual",
                                              abi::field_view(boundary_output.data(), 1, 1, 2)}};
  PopsFieldBoundaryRequestV1 field_boundary_request{
      sizeof(PopsFieldBoundaryRequestV1),
      "case::field-boundary",
      ghost_request.region,
      abi::const_field_view(normal.data(), 1, 1, 2),
      1,
      boundary_states,
      1,
      boundary_directions,
      1,
      boundary_fields,
      1,
      boundary_parameters,
      1,
      boundary_outputs,
      {sizeof(PopsQualifiedConstFieldV1), 1, "case::rate",
       abi::const_field_view(right.data(), 1, 1, 2)},
      {sizeof(PopsQualifiedConstFieldV1), 1, "case::nonlinear-iterate",
       abi::const_field_view(left.data(), 1, 1, 2)},
      2,
      abi::logical_time(),
      noncollective_execution};
  EXPECT_EQ(pops::component::evaluate_field_boundary(field_boundary_api, nullptr,
                                                     field_boundary_request, status, true),
            0);
  EXPECT_EQ(boundary_output, (std::array<double, 2>{3.0, 6.0}));
  field_boundary_request.direction_count = 0;
  field_boundary_request.directions = nullptr;
  EXPECT_EQ(pops::component::evaluate_field_boundary(field_boundary_api, nullptr,
                                                     field_boundary_request, status, false),
            0);
  EXPECT_EQ(boundary_output, left);

  std::array<double, 4> tag_values{-1.0, 2.0, 0.0, 3.0};
  std::array<std::uint8_t, 4> tags{};
  PopsTaggerApiV2 tagger_api{
      abi_header(sizeof(PopsTaggerApiV2), POPS_NATIVE_INTERFACE_TAGGER_V2, 2),
      +[](void*, const PopsTaggerRequestV2* request, PopsComponentStatusV1* result) {
        const auto* state = static_cast<const double*>(request->states[0].values.data);
        for (std::size_t i = 0; i < request->refine_candidates.size; ++i)
          request->refine_candidates.data[i] = state[i] > request->program.leaves[0].threshold;
        *result = ok_status();
        return 0;
      }};
  const PopsQualifiedConstFieldV1 tag_states{sizeof(PopsQualifiedConstFieldV1), 1,
                                             "case::tag-state",
                                             abi::const_field_view(tag_values.data(), 2, 2)};
  const PopsTaggingLeafV1 tag_leaf{sizeof(PopsTaggingLeafV1), 0, 0, 1, 0.0,
                                   POPS_TAGGING_NO_STENCIL_V1};
  const std::int32_t tag_op = 1, tag_arg = 0;
  std::array<std::uint8_t, 4> coarsen{}, refine_equalities{}, coarsen_equalities{};
  PopsTaggerRequestV2 tag_request{
      sizeof(PopsTaggerRequestV2),
      POPS_TAGGER_EXECUTION_NATIVE_BACKEND_V2,
      POPS_TAGGER_COLLECTIVE_NONE_V2,
      1,
      &tag_states,
      {sizeof(PopsTaggingProgramV1), "case::tag-program", 0, nullptr, 1, &tag_leaf, 1, &tag_op,
       &tag_arg, 0, nullptr, nullptr, 0, 0, 0, POPS_TAGGING_NON_FINITE_REJECT_V1},
      {0, 0, 0},
      {0, 0, 0},
      {1, 1, 0},
      {1.0, 1.0, 0.0},
      0,
      {sizeof(PopsTaggerMaskViewV2), tags.data(), tags.size(), POPS_MEMORY_SPACE_HOST_V1,
       POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1},
      {sizeof(PopsTaggerMaskViewV2), coarsen.data(), coarsen.size(), POPS_MEMORY_SPACE_HOST_V1,
       POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1},
      {sizeof(PopsTaggerMaskViewV2), refine_equalities.data(), refine_equalities.size(),
       POPS_MEMORY_SPACE_HOST_V1, POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1},
      {sizeof(PopsTaggerMaskViewV2), coarsen_equalities.data(), coarsen_equalities.size(),
       POPS_MEMORY_SPACE_HOST_V1, POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1},
      abi::logical_time(),
      abi::noncollective_host_execution_context()};
  EXPECT_EQ(pops::component::tag_batch(tagger_api, nullptr, tag_request, status), 0);
  EXPECT_EQ(tags, (std::array<std::uint8_t, 4>{0, 1, 0, 1}));
  auto collective_request = tag_request;
  collective_request.execution = execution;
  EXPECT_THROW((void)pops::component::tag_batch(tagger_api, nullptr, collective_request, status),
               std::invalid_argument);
  auto unknown_scope = tag_request;
  unknown_scope.collective_scope = static_cast<PopsTaggerCollectiveScopeV2>(1);
  EXPECT_THROW((void)pops::component::tag_batch(tagger_api, nullptr, unknown_scope, status),
               std::invalid_argument);
  auto mismatched_native_mask = tag_request;
  mismatched_native_mask.refine_candidates.memory_space = POPS_MEMORY_SPACE_MANAGED_V1;
  EXPECT_THROW(
      (void)pops::component::tag_batch(tagger_api, nullptr, mismatched_native_mask, status),
      std::invalid_argument);
  auto explicit_host_request = tag_request;
  explicit_host_request.execution_mode = POPS_TAGGER_EXECUTION_HOST_V2;
  EXPECT_EQ(pops::component::tag_batch(tagger_api, nullptr, explicit_host_request, status), 0);
  auto unsupported_window_request = tag_request;
  auto unsupported_window_leaf = tag_leaf;
  unsupported_window_leaf.opcode = POPS_TAGGING_PRESCRIBED_WINDOW_V1;
  unsupported_window_request.program.leaves = &unsupported_window_leaf;
  unsupported_window_request.program.refine_opcodes = &unsupported_window_leaf.opcode;
  EXPECT_THROW(
      (void)pops::component::tag_batch(tagger_api, nullptr, unsupported_window_request, status),
      std::invalid_argument);

  std::array<std::int64_t, 2> extents{4, 1};
  std::array<std::int64_t, 4> boxes{};
  std::size_t box_count = 0;
  PopsClusteringApiV1 cluster_api{
      abi_header(sizeof(PopsClusteringApiV1), POPS_NATIVE_INTERFACE_CLUSTERING_V1),
      +[](void*, const PopsClusteringRequestV1* request, PopsComponentStatusV1* result) {
        request->boxes[0] = 1;
        request->boxes[1] = 0;
        request->boxes[2] = 3;
        request->boxes[3] = 0;
        *request->box_count = 1;
        *result = ok_status();
        return 0;
      }};
  PopsClusteringRequestV1 cluster_request{sizeof(PopsClusteringRequestV1),
                                          {sizeof(PopsConstByteViewV1), tags.data(), tags.size()},
                                          extents.data(),
                                          2,
                                          boxes.data(),
                                          1,
                                          &box_count,
                                          execution};
  EXPECT_EQ(pops::component::cluster_tags(cluster_api, nullptr, cluster_request, status), 0);
  EXPECT_EQ(box_count, 1u);
  EXPECT_EQ(boxes, (std::array<std::int64_t, 4>{1, 0, 3, 0}));

  PopsClusteringApiV1 excessive_cluster_api{
      abi_header(sizeof(PopsClusteringApiV1), POPS_NATIVE_INTERFACE_CLUSTERING_V1),
      +[](void*, const PopsClusteringRequestV1* request, PopsComponentStatusV1* result) {
        *request->box_count = request->box_capacity + 1;
        *result = ok_status();
        return 0;
      }};
  box_count = 0;
  EXPECT_THROW(
      pops::component::cluster_tags(excessive_cluster_api, nullptr, cluster_request, status),
      std::runtime_error);
  auto missing_cluster_api = cluster_api;
  missing_cluster_api.cluster = nullptr;
  EXPECT_THROW(pops::component::cluster_tags(missing_cluster_api, nullptr, cluster_request, status),
               std::runtime_error);

  std::array<double, 1> transferred{};
  std::array<std::int32_t, 2> ratio{2, 2};
  PopsTransferApiV1 transfer_api{
      abi_header(sizeof(PopsTransferApiV1), POPS_NATIVE_INTERFACE_TRANSFER_V1),
      +[](void*, const PopsTransferRequestV1* request, PopsComponentStatusV1* result) {
        const auto* source = static_cast<const double*>(request->source.data);
        auto* destination = static_cast<double*>(request->destination.data);
        destination[0] = 0.25 * (source[0] + source[1] + source[2] + source[3]);
        *result = ok_status();
        return 0;
      }};
  PopsTransferRequestV1 transfer_request{sizeof(PopsTransferRequestV1),
                                         abi::const_field_view(tag_values.data(), 2, 2),
                                         abi::field_view(transferred.data(), 1, 1),
                                         ratio.data(),
                                         2,
                                         POPS_TRANSFER_OPERATION_CONSERVATIVE_CELL_AVERAGE_V1,
                                         execution};
  EXPECT_EQ(pops::component::apply_transfer(transfer_api, nullptr, transfer_request, status), 0);
  EXPECT_DOUBLE_EQ(transferred[0], 1.0);
  auto wrong_transfer_shape = transfer_request;
  wrong_transfer_shape.destination.extents[1] = 2;
  EXPECT_THROW(pops::component::apply_transfer(transfer_api, nullptr, wrong_transfer_shape, status),
               std::invalid_argument);

  std::array<double, 2> coarse_integrated_flux{1.0, 2.0};
  std::array<double, 2> fine_integrated_flux{3.0, 6.0};
  std::array<double, 2> reflux_correction{};
  PopsRefluxApiV1 reflux_api{
      abi_header(sizeof(PopsRefluxApiV1), POPS_NATIVE_INTERFACE_REFLUX_V1),
      +[](void*, const PopsRefluxRequestV1* request, PopsComponentStatusV1* result) {
        for (std::size_t face_index = 0; face_index < request->face_count; ++face_index) {
          const auto& face = request->faces[face_index];
          const auto* coarse = static_cast<const double*>(face.coarse_integrated_flux.data);
          const auto* fine = static_cast<const double*>(face.fine_integrated_flux.data);
          auto* correction = static_cast<double*>(face.correction.data);
          const std::size_t points =
              pops::component::field_point_count(face.coarse_integrated_flux);
          for (std::size_t point = 0; point < points; ++point)
            correction[point] = static_cast<double>(face.side) * (fine[point] - coarse[point]) *
                                face.inverse_coarse_cell_spacing;
        }
        *result = ok_status();
        return 0;
      }};
  auto coarse_face = abi::const_field_view(coarse_integrated_flux.data(), 1, 2, 1, "parent::layout",
                                           "parent::patch");
  coarse_face.centering = POPS_FIELD_CENTERING_FACE_V1;
  coarse_face.centering_axes = 1u;
  auto fine_face =
      abi::const_field_view(fine_integrated_flux.data(), 1, 2, 1, "child::layout", "child::patch");
  fine_face.centering = POPS_FIELD_CENTERING_FACE_V1;
  fine_face.centering_axes = 1u;
  PopsRefluxFaceV1 reflux_face{
      sizeof(PopsRefluxFaceV1),
      "transition::0-to-1/x-low",
      0,
      POPS_REFLUX_FACE_LOW_V1,
      2.0,
      coarse_face,
      fine_face,
      abi::field_view(reflux_correction.data(), 1, 2, 1, "parent::layout", "parent::patch")};
  PopsRefluxRequestV1 reflux_request{sizeof(PopsRefluxRequestV1),
                                     "transition::0-to-1",
                                     0,
                                     1,
                                     1,
                                     &reflux_face,
                                     abi::logical_time(),
                                     abi::noncollective_host_execution_context()};
  EXPECT_TRUE(pops::component::generated_native_interface_table_is_complete(
      POPS_NATIVE_INTERFACE_REFLUX_V1, &reflux_api, sizeof(reflux_api)));
  EXPECT_EQ(
      pops::component::apply_reflux_interface_batch(reflux_api, nullptr, reflux_request, status),
      0);
  EXPECT_EQ(reflux_correction, (std::array<double, 2>{-4.0, -8.0}));

  auto incomplete_reflux_api = reflux_api;
  incomplete_reflux_api.apply_interface_batch = nullptr;
  EXPECT_FALSE(pops::component::generated_native_interface_table_is_complete(
      POPS_NATIVE_INTERFACE_REFLUX_V1, &incomplete_reflux_api, sizeof(incomplete_reflux_api)));
  EXPECT_THROW(pops::component::apply_reflux_interface_batch(incomplete_reflux_api, nullptr,
                                                             reflux_request, status),
               std::runtime_error);
  auto collective_reflux = reflux_request;
  collective_reflux.execution = execution;
  EXPECT_THROW(
      pops::component::apply_reflux_interface_batch(reflux_api, nullptr, collective_reflux, status),
      std::invalid_argument);
  auto malformed_reflux = reflux_request;
  auto malformed_face = reflux_face;
  malformed_face.correction.layout_identity = "other::parent-layout";
  malformed_reflux.faces = &malformed_face;
  EXPECT_THROW(
      pops::component::apply_reflux_interface_batch(reflux_api, nullptr, malformed_reflux, status),
      std::invalid_argument);

  auto overflowing_ghosts = abi::const_field_view(tag_values.data(), 2, 2);
  overflowing_ghosts.ghost_lower[0] = std::numeric_limits<std::size_t>::max();
  overflowing_ghosts.ghost_upper[0] = 1;
  EXPECT_THROW(
      pops::component::validate_field_view(overflowing_ghosts, "overflowing ghost test view"),
      std::invalid_argument);

  static constexpr PopsTopologyLabelV2 label_vocabulary[] = {
      {sizeof(PopsTopologyLabelV2), 1, "island-a", "test-topology"},
      {sizeof(PopsTopologyLabelV2), 2, "island-b", "test-topology"}};
  struct TopologyCallState {
    int calls = 0;
  } topology_calls;
  PopsFieldTopologyApiV2 topology_api{
      abi_header(sizeof(PopsFieldTopologyApiV2), POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, 2),
      +[](void* raw, const PopsFieldTopologyRequestV2* request, PopsFieldTopologyResultV2* result) {
        auto* state = static_cast<TopologyCallState*>(raw);
        if (++state->calls != 1 || request->topology.patch_count != 2 ||
            request->local_patch_count != 2 || request->topology.periodic_axes != 1 ||
            std::strcmp(request->topology.topology_recipe_identity, "test::topology-recipe") != 0)
          return 8;
        for (std::size_t local = 0; local < request->local_patch_count; ++local) {
          const auto& patch = request->local_patches[local];
          if (patch.material_representation != POPS_FIELD_MATERIAL_FULL_V1 ||
              patch.material_coverage.data != nullptr ||
              patch.cut_cell_volume_fraction.data != nullptr ||
              patch.material_ids.data != nullptr || patch.material_mask.size != 2 ||
              patch.component_labels.size != 2)
            return 9;
          std::fill(patch.material_mask.data, patch.material_mask.data + 2, 1);
          std::fill(patch.component_labels.data, patch.component_labels.data + 2,
                    static_cast<std::int32_t>(local + 1));
        }
        result->label_count = 2;
        result->labels = label_vocabulary;
        result->provenance = "test-topology";
        result->topology_digest = "topology-v2";
        result->status = ok_status();
        return 0;
      }};
  PopsFieldTopologyApiV2 rejecting_topology_api{
      abi_header(sizeof(PopsFieldTopologyApiV2), POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, 2),
      +[](void*, const PopsFieldTopologyRequestV2*, PopsFieldTopologyResultV2* result) {
        result->status = {sizeof(PopsComponentStatusV1), 17, POPS_COMPONENT_REJECT_STEP_V1,
                          "topology rejected by test component"};
        return 0;
      }};
  PopsFieldTopologyApiV2 incomplete_topology_api{
      abi_header(sizeof(PopsFieldTopologyApiV2), POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, 2),
      +[](void*, const PopsFieldTopologyRequestV2*, PopsFieldTopologyResultV2* result) {
        // A successful status is not permission to leave the caller-owned point outputs untouched.
        result->label_count = 2;
        result->labels = label_vocabulary;
        result->provenance = "incomplete-test-topology";
        result->topology_digest = "incomplete-topology-v2";
        result->status = ok_status();
        return 0;
      }};
  static constexpr PopsTopologyLabelV2 undersized_label_vocabulary[] = {
      {0, 1, "island-a", "undersized-label-test-topology"}};
  PopsFieldTopologyApiV2 undersized_label_topology_api{
      abi_header(sizeof(PopsFieldTopologyApiV2), POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, 2),
      +[](void*, const PopsFieldTopologyRequestV2* request, PopsFieldTopologyResultV2* result) {
        for (std::size_t local = 0; local < request->local_patch_count; ++local) {
          const auto& patch = request->local_patches[local];
          std::fill(patch.material_mask.data, patch.material_mask.data + patch.material_mask.size,
                    1);
          std::fill(patch.component_labels.data,
                    patch.component_labels.data + patch.component_labels.size, 1);
        }
        result->label_count = 1;
        result->labels = undersized_label_vocabulary;
        result->provenance = "undersized-label-test-topology";
        result->topology_digest = "undersized-label-topology-v2";
        result->status = ok_status();
        return 0;
      }};
  PopsFieldTopologyApiV2 empty_full_topology_api{
      abi_header(sizeof(PopsFieldTopologyApiV2), POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, 2),
      +[](void*, const PopsFieldTopologyRequestV2* request, PopsFieldTopologyResultV2* result) {
        for (std::size_t local = 0; local < request->local_patch_count; ++local) {
          const auto& patch = request->local_patches[local];
          std::fill(patch.material_mask.data, patch.material_mask.data + patch.material_mask.size,
                    0);
          std::fill(patch.component_labels.data,
                    patch.component_labels.data + patch.component_labels.size, 0);
        }
        result->label_count = 2;
        result->labels = label_vocabulary;
        result->provenance = "empty-full-test-topology";
        result->topology_digest = "empty-full-topology-v2";
        result->status = ok_status();
        return 0;
      }};
  std::string topology_layout =
      "test::owned-topology-layout-identity-longer-than-small-string-storage";
  std::array<std::string, 2> topology_patches{
      "test::owned-topology-patch-zero-longer-than-small-string-storage",
      "test::owned-topology-patch-one-longer-than-small-string-storage"};
  std::array<PopsFieldPatchMetadataV1, 2> metadata{};
  for (std::size_t index = 0; index < metadata.size(); ++index) {
    metadata[index] = {sizeof(PopsFieldPatchMetadataV1),
                       index,
                       0,
                       0,
                       2,
                       {},
                       {},
                       {},
                       {},
                       POPS_FIELD_CENTERING_CELL_V1,
                       0,
                       topology_layout.c_str(),
                       topology_patches[index].c_str()};
    metadata[index].lower[0] = static_cast<std::int64_t>(2 * index);
    metadata[index].upper[0] = static_cast<std::int64_t>(2 * index + 1);
    metadata[index].lower[1] = metadata[index].upper[1] = 0;
    metadata[index].cell_spacing[0] = metadata[index].cell_spacing[1] = 0.25;
  }
  auto unrepresentable_patch = metadata[0];
  unrepresentable_patch.lower[0] = std::numeric_limits<std::int64_t>::min();
  unrepresentable_patch.upper[0] = std::numeric_limits<std::int64_t>::max();
  EXPECT_THROW(pops::component::validate_field_patch_metadata(unrepresentable_patch, 0),
               std::invalid_argument);
  const std::vector<pops::component::FieldTopologyPatchInputV2> topology_inputs{
      {0, POPS_FIELD_MATERIAL_FULL_V1, {}, {}, {}},
      {1, POPS_FIELD_MATERIAL_FULL_V1, {}, {}, {}},
  };
  std::array<std::uint8_t, 2> binary_coverage{1, 0};
  pops::component::FieldTopologyPatchInputV2 binary_input{
      0,
      POPS_FIELD_MATERIAL_BINARY_COVERAGE_V1,
      {sizeof(PopsConstByteViewV1), binary_coverage.data(), binary_coverage.size()},
      {},
      {}};
  EXPECT_EQ(pops::component::expected_topology_material_mask(binary_input, metadata[0]),
            (std::vector<std::uint8_t>{1, 0}));
  binary_coverage[1] = 2;
  EXPECT_THROW((void)pops::component::expected_topology_material_mask(binary_input, metadata[0]),
               std::invalid_argument);
  binary_coverage[1] = 0;

  std::array<double, 2> cut_fractions{1.0, 0.0};
  pops::component::FieldTopologyPatchInputV2 cut_input{
      0,
      POPS_FIELD_MATERIAL_CUT_CELL_FRACTION_V1,
      {},
      abi::const_field_view(cut_fractions.data(), 2, 1, 1, metadata[0].layout_identity,
                            metadata[0].patch_identity),
      {}};
  EXPECT_EQ(pops::component::expected_topology_material_mask(cut_input, metadata[0]),
            (std::vector<std::uint8_t>{1, 0}));
  cut_fractions[1] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW((void)pops::component::expected_topology_material_mask(cut_input, metadata[0]),
               std::invalid_argument);
  cut_fractions[1] = 0.0;
  PopsFieldGlobalTopologyV1 global_topology{sizeof(PopsFieldGlobalTopologyV1),
                                            "test::topology-recipe",
                                            topology_layout.c_str(),
                                            "test::materialized-layout",
                                            2,
                                            {},
                                            {},
                                            1,
                                            metadata.size(),
                                            metadata.data()};
  global_topology.domain_upper[0] = 3;
  const auto topology = [&] {
    EXPECT_THROW(pops::component::prepare_field_topology(
                     rejecting_topology_api, nullptr, global_topology, topology_inputs, execution),
                 std::runtime_error);
    EXPECT_THROW(pops::component::prepare_field_topology(
                     incomplete_topology_api, nullptr, global_topology, topology_inputs, execution),
                 std::runtime_error);
    EXPECT_THROW(
        pops::component::prepare_field_topology(undersized_label_topology_api, nullptr,
                                                global_topology, topology_inputs, execution),
        std::runtime_error);
    EXPECT_THROW(pops::component::prepare_field_topology(
                     empty_full_topology_api, nullptr, global_topology, topology_inputs, execution),
                 std::runtime_error);
    auto prepared = pops::component::prepare_field_topology(
        topology_api, &topology_calls, global_topology, topology_inputs, execution);
    std::fill(topology_layout.begin(), topology_layout.end(), 'x');
    for (auto& patch : topology_patches)
      std::fill(patch.begin(), patch.end(), 'y');
    return prepared;
  }();
  EXPECT_EQ(topology.topology_digest(), "topology-v2");
  ASSERT_EQ(topology.local_patches().size(), 2u);
  EXPECT_EQ(topology.local_patches()[0].component_labels, (std::vector<std::int32_t>{1, 1}));
  EXPECT_EQ(topology.local_patches()[1].component_labels, (std::vector<std::int32_t>{2, 2}));

  std::array<double, 2> rhs_a{1.0, 2.0}, rhs_b{3.0, 4.0};
  std::array<double, 2> solution_a{}, solution_b{};
  struct SolverCallState {
    int calls = 0;
  } solver_calls;
  PopsFieldSolverApiV2 solver_api{
      abi_header(sizeof(PopsFieldSolverApiV2), POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, 2),
      +[](void* raw, const PopsFieldSolverRequestV2* request, PopsSolveReportV2* report) {
        auto* state = static_cast<SolverCallState*>(raw);
        if (++state->calls != 1 || request->topology.patch_count != 2 ||
            request->local_patch_count != 2 || request->topology_label_count != 2 ||
            !request->topology_labels || !request->topology_provenance ||
            std::strcmp(request->topology_provenance, "test-topology") != 0 ||
            std::strcmp(request->topology_digest, "topology-v2") != 0 ||
            std::strcmp(request->topology.topology_recipe_identity, "test::topology-recipe") != 0)
          return 7;
        for (std::size_t index = 0; index < request->topology_label_count; ++index) {
          const auto& label = request->topology_labels[index];
          if (label.struct_size < sizeof(PopsFieldSolverTopologyLabelV2) ||
              label.id != static_cast<std::int32_t>(index + 1) || !label.label ||
              !label.provenance || std::strcmp(label.provenance, "test-topology") != 0)
            return 8;
        }
        for (std::size_t local = 0; local < request->local_patch_count; ++local) {
          const auto& patch = request->local_patches[local];
          if (patch.struct_size < sizeof(PopsFieldSolverPatchV2) || patch.material_mask.size != 2 ||
              patch.component_labels.size != 2)
            return 9;
          const auto* rhs_values = static_cast<const double*>(patch.rhs.data);
          auto* solution_values = static_cast<double*>(patch.solution.data);
          std::copy(rhs_values, rhs_values + 2, solution_values);
        }
        report->status = POPS_SOLVE_SOLVED_V2;
        report->action = POPS_SOLVE_ACTION_NONE_V2;
        report->iterations = 1;
        report->relative_residual = 0.0;
        report->reference_residual_norm = 1.0;
        report->residual_norm = 0.0;
        report->reason = "tolerance reached";
        return 0;
      }};
  const auto& owned_metadata = topology.global_patches();
  const std::vector<pops::component::FieldSolverPatchBindingV2> solver_patches{
      {0,
       abi::const_field_view(rhs_a.data(), 2, 1, 1, owned_metadata[0].layout_identity,
                             owned_metadata[0].patch_identity),
       abi::field_view(solution_a.data(), 2, 1, 1, owned_metadata[0].layout_identity,
                       owned_metadata[0].patch_identity),
       {}},
      {1,
       abi::const_field_view(rhs_b.data(), 2, 1, 1, owned_metadata[1].layout_identity,
                             owned_metadata[1].patch_identity),
       abi::field_view(solution_b.data(), 2, 1, 1, owned_metadata[1].layout_identity,
                       owned_metadata[1].patch_identity),
       {}},
  };
  const auto solver_request = pops::component::bind_field_solver_request(
      topology, solver_patches, execution, "{\"identity\":\"test::boundary\"}", 1e-8, 0.0, 10);
  auto omitted_vocabulary_request = pops::component::bind_field_solver_request(
      topology, solver_patches, execution, "{\"identity\":\"test::boundary\"}", 1e-8, 0.0, 10);
  const_cast<PopsFieldSolverRequestV2&>(omitted_vocabulary_request.request()).topology_labels =
      nullptr;
  PopsSolveReportV2 solve_report{};
  solve_report.struct_size = sizeof(PopsSolveReportV2);
  EXPECT_THROW(pops::component::solve_field(solver_api, &solver_calls, omitted_vocabulary_request,
                                            solve_report),
               std::invalid_argument);
  auto substituted_digest_request = pops::component::bind_field_solver_request(
      topology, solver_patches, execution, "{\"identity\":\"test::boundary\"}", 1e-8, 0.0, 10);
  const_cast<PopsFieldSolverRequestV2&>(substituted_digest_request.request()).topology_digest =
      "substituted-topology";
  EXPECT_THROW(pops::component::solve_field(solver_api, &solver_calls, substituted_digest_request,
                                            solve_report),
               std::invalid_argument);
  EXPECT_EQ(pops::component::solve_field(solver_api, &solver_calls, solver_request, solve_report),
            0);
  EXPECT_EQ(solution_a, rhs_a);
  EXPECT_EQ(solution_b, rhs_b);
  EXPECT_EQ(solver_calls.calls, 1);
  auto topology_mutation_request = pops::component::bind_field_solver_request(
      topology, solver_patches, execution, "{\"identity\":\"test::boundary\"}", 1e-8, 0.0, 10);
  PopsFieldSolverApiV2 topology_mutation_solver_api{
      abi_header(sizeof(PopsFieldSolverApiV2), POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, 2),
      +[](void*, const PopsFieldSolverRequestV2* request, PopsSolveReportV2* report) {
        auto* mask = const_cast<std::uint8_t*>(request->local_patches[0].material_mask.data);
        mask[0] = 0;
        report->status = POPS_SOLVE_SOLVED_V2;
        report->action = POPS_SOLVE_ACTION_NONE_V2;
        report->iterations = 1;
        report->relative_residual = 0.0;
        report->reference_residual_norm = 1.0;
        report->residual_norm = 0.0;
        report->reason = "mutated topology";
        return 0;
      }};
  EXPECT_THROW(pops::component::solve_field(topology_mutation_solver_api, nullptr,
                                            topology_mutation_request, solve_report),
               std::runtime_error);
  EXPECT_EQ(topology.local_patches()[0].material_mask, (std::vector<std::uint8_t>{1, 1}));
  PopsFieldSolverApiV2 false_success_solver_api{
      abi_header(sizeof(PopsFieldSolverApiV2), POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, 2),
      +[](void*, const PopsFieldSolverRequestV2*, PopsSolveReportV2* report) {
        report->status = POPS_SOLVE_SOLVED_V2;
        report->action = POPS_SOLVE_ACTION_NONE_V2;
        report->iterations = 1;
        report->relative_residual = 0.9;
        report->reference_residual_norm = 1.0;
        report->residual_norm = 0.9;
        report->reason = "false success";
        return 0;
      }};
  EXPECT_THROW(
      pops::component::solve_field(false_success_solver_api, nullptr, solver_request, solve_report),
      std::runtime_error);
  PopsFieldSolverApiV2 zero_forcing_false_success_api{
      abi_header(sizeof(PopsFieldSolverApiV2), POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, 2),
      +[](void*, const PopsFieldSolverRequestV2*, PopsSolveReportV2* report) {
        report->status = POPS_SOLVE_SOLVED_V2;
        report->action = POPS_SOLVE_ACTION_NONE_V2;
        report->iterations = 1;
        report->relative_residual = 1.0e-12;
        report->reference_residual_norm = 0.0;
        report->residual_norm = 1.0e-12;
        report->reason = "false zero-reference success";
        return 0;
      }};
  EXPECT_THROW(pops::component::solve_field(zero_forcing_false_success_api, nullptr, solver_request,
                                            solve_report),
               std::runtime_error);
  PopsFieldSolverApiV2 malformed_status_solver_api{
      abi_header(sizeof(PopsFieldSolverApiV2), POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, 2),
      +[](void*, const PopsFieldSolverRequestV2*, PopsSolveReportV2* report) {
        report->status = 17;
        report->action = POPS_SOLVE_ACTION_REJECT_ATTEMPT_V2;
        report->iterations = 1;
        report->relative_residual = 1.0;
        report->reference_residual_norm = 1.0;
        report->residual_norm = 1.0;
        report->reason = "malformed status";
        return 0;
      }};
  EXPECT_THROW(pops::component::solve_field(malformed_status_solver_api, nullptr, solver_request,
                                            solve_report),
               std::runtime_error);
  PopsFieldSolverApiV2 malformed_action_solver_api{
      abi_header(sizeof(PopsFieldSolverApiV2), POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, 2),
      +[](void*, const PopsFieldSolverRequestV2*, PopsSolveReportV2* report) {
        report->status = POPS_SOLVE_ITERATION_LIMIT_V2;
        report->action = 17;
        report->iterations = 1;
        report->relative_residual = 1.0;
        report->reference_residual_norm = 1.0;
        report->residual_norm = 1.0;
        report->reason = "malformed action";
        return 0;
      }};
  EXPECT_THROW(pops::component::solve_field(malformed_action_solver_api, nullptr, solver_request,
                                            solve_report),
               std::runtime_error);
  PopsFieldSolverApiV2 incoherent_ratio_solver_api{
      abi_header(sizeof(PopsFieldSolverApiV2), POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, 2),
      +[](void*, const PopsFieldSolverRequestV2*, PopsSolveReportV2* report) {
        report->status = POPS_SOLVE_ITERATION_LIMIT_V2;
        report->action = POPS_SOLVE_ACTION_FAIL_RUN_V2;
        report->iterations = 10;
        report->relative_residual = 0.5;
        report->reference_residual_norm = 2.0;
        report->residual_norm = 0.2;
        report->reason = "ratio does not authenticate residual norms";
        return 0;
      }};
  EXPECT_THROW(pops::component::solve_field(incoherent_ratio_solver_api, nullptr, solver_request,
                                            solve_report),
               std::runtime_error);
  PopsFieldSolverApiV2 incompatible_rhs_solver_api{
      abi_header(sizeof(PopsFieldSolverApiV2), POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, 2),
      +[](void*, const PopsFieldSolverRequestV2*, PopsSolveReportV2* report) {
        report->status = POPS_SOLVE_INCOMPATIBLE_RHS_V2;
        report->action = POPS_SOLVE_ACTION_FAIL_RUN_V2;
        report->iterations = 0;
        report->relative_residual = 1.0;
        report->reference_residual_norm = 1.0;
        report->residual_norm = 1.0;
        report->reason = "RHS is incompatible with the declared nullspace";
        return 0;
      }};
  EXPECT_EQ(pops::component::solve_field(incompatible_rhs_solver_api, nullptr, solver_request,
                                         solve_report),
            0);
  EXPECT_EQ(solve_report.status, POPS_SOLVE_INCOMPATIBLE_RHS_V2);
  EXPECT_STREQ(solve_report.reason, "RHS is incompatible with the declared nullspace");
  PopsFieldSolverApiV2 transport_failure_solver_api{
      abi_header(sizeof(PopsFieldSolverApiV2), POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, 2),
      +[](void*, const PopsFieldSolverRequestV2*, PopsSolveReportV2*) { return 23; }};
  EXPECT_THROW(pops::component::solve_field(transport_failure_solver_api, nullptr, solver_request,
                                            solve_report),
               std::runtime_error);
  std::array<double, 3> short_solution{};
  auto invalid_patches = solver_patches;
  invalid_patches[0].solution =
      abi::field_view(short_solution.data(), 1, 3, 1, owned_metadata[0].layout_identity,
                      owned_metadata[0].patch_identity);
  EXPECT_THROW(
      pops::component::bind_field_solver_request(
          topology, invalid_patches, execution, "{\"identity\":\"test::boundary\"}", 1e-8, 0.0, 10),
      std::invalid_argument);

  std::array<double, 4> ghosted_coefficients{};
  auto ghosted_patches = solver_patches;
  ghosted_patches[0].coefficients =
      abi::const_field_view(ghosted_coefficients.data(), 4, 1, 1, owned_metadata[0].layout_identity,
                            owned_metadata[0].patch_identity);
  ghosted_patches[0].coefficients.ghost_lower[0] = 1;
  ghosted_patches[0].coefficients.ghost_upper[0] = 1;
  EXPECT_NO_THROW((void)pops::component::bind_field_solver_request(
      topology, ghosted_patches, execution, "{\"identity\":\"test::boundary\"}", 1e-8, 0.0, 10));

  struct WriterCallState {
    int publish_count = 0;
    bool reject_verification = false;
  } writer_state;
  PopsWriterApiV1 writer_api{
      abi_header(sizeof(PopsWriterApiV1), POPS_NATIVE_INTERFACE_WRITER_V1),
      +[](void* state, const PopsWriterRequestV1*, PopsWriterReceiptV1* receipt) {
        receipt->bytes_written = 999;
        receipt->content_digest = "verify-only";
        if (static_cast<WriterCallState*>(state)->reject_verification) {
          receipt->status = {sizeof(PopsComponentStatusV1), 23, POPS_COMPONENT_REJECT_STEP_V1,
                             "writer verification rejected snapshot"};
          return 0;
        }
        receipt->status = ok_status();
        return 0;
      },
      +[](void* state, const PopsWriterRequestV1* request, PopsWriterReceiptV1* receipt) {
        if (request->snapshot_identity == nullptr || request->geometry_count != 1 ||
            request->field_count != 1 || request->fields[0].piece_count != 1 ||
            receipt->struct_size != sizeof(PopsWriterReceiptV1) || receipt->bytes_written != 0 ||
            receipt->content_digest != nullptr ||
            receipt->status.struct_size != sizeof(PopsComponentStatusV1) ||
            receipt->status.code != 0 || receipt->status.action != POPS_COMPONENT_CONTINUE_V1)
          return 3;
        ++static_cast<WriterCallState*>(state)->publish_count;
        receipt->bytes_written =
            pops::component::field_point_count(request->fields[0].pieces[0].values) *
            sizeof(double);
        receipt->content_digest = "writer-v1";
        receipt->status = ok_status();
        return 0;
      },
      +[](void*, const PopsWriterRequestV1*) {}, +[](void*, const PopsWriterRequestV1*) {}};
  const std::array<std::int64_t, 2> writer_lower{0, 0}, writer_upper{2, 2};
  const PopsWriterBoxV1 writer_box{sizeof(PopsWriterBoxV1), 2, writer_lower.data(),
                                   writer_upper.data()};
  const std::array<std::uint8_t, 4> valid_cells{1, 1, 1, 1};
  const std::array<std::uint8_t, 4> covered_cells{0, 0, 0, 0};
  const std::array<double, 4> cell_volumes{1.0, 1.0, 1.0, 1.0};
  const std::array<double, 2> writer_origin{0.0, 0.0}, writer_spacing{1.0, 1.0};
  const std::array<std::size_t, 2> writer_shape{2, 2};
  const PopsWriterGeometryV1 writer_geometry{
      sizeof(PopsWriterGeometryV1),
      "layout-v1",
      "uniform",
      0,
      2,
      writer_origin.data(),
      writer_spacing.data(),
      writer_shape.data(),
      1,
      &writer_box,
      {sizeof(PopsConstByteViewV1), valid_cells.data(), valid_cells.size()},
      {sizeof(PopsConstByteViewV1), covered_cells.data(), covered_cells.size()},
      abi::const_field_view(cell_volumes.data(), 2, 2, 1, "layout-v1", "geometry-patch")};
  const std::array<double, 4> writer_values{1.0, 2.0, 3.0, 4.0};
  const PopsWriterPieceV1 writer_piece{
      sizeof(PopsWriterPieceV1), 2, writer_lower.data(), writer_upper.data(),
      abi::const_field_view(writer_values.data(), 2, 2, 1, "layout-v1", "state-patch")};
  const char* component_names[] = {"u"};
  const PopsWriterFieldV1 writer_field{sizeof(PopsWriterFieldV1),
                                       "field-v1",
                                       "block::u",
                                       "manifest-v1",
                                       "layout-v1",
                                       0,
                                       "accepted",
                                       "cell",
                                       "unspecified",
                                       1,
                                       component_names,
                                       2,
                                       writer_shape.data(),
                                       1,
                                       &writer_piece};
  PopsWriterRequestV1 writer_request{sizeof(PopsWriterRequestV1),
                                     1,
                                     &writer_geometry,
                                     1,
                                     &writer_field,
                                     0,
                                     nullptr,
                                     "{}",
                                     "selection-v1",
                                     "temporary",
                                     "published",
                                     "snapshot-v1",
                                     abi::logical_time(),
                                     execution};
  PopsWriterReceiptV1 receipt{};
  receipt.struct_size = sizeof(PopsWriterReceiptV1);
  EXPECT_EQ(pops::component::publish_output(writer_api, &writer_state, writer_request, receipt), 0);
  EXPECT_EQ(writer_state.publish_count, 1);
  EXPECT_EQ(receipt.bytes_written, 4u * sizeof(double));
  writer_state.reject_verification = true;
  EXPECT_THROW(pops::component::publish_output(writer_api, &writer_state, writer_request, receipt),
               std::runtime_error);
  EXPECT_EQ(writer_state.publish_count, 1);
}

TEST(ComponentInterfaces, FieldSolverV2CarriesOneBinaryCoverageMultilevelBatch) {
  const PopsExecutionContextV1 execution = abi::host_execution_context();
  static constexpr PopsTopologyLabelV2 labels[] = {
      {sizeof(PopsTopologyLabelV2), 1, "composite-material", "multilevel-test"}};
  std::array<std::string, 2> patch_identities{"coarse-patch", "fine-patch"};
  std::array<PopsFieldPatchMetadataV1, 2> metadata{};
  for (std::size_t index = 0; index < metadata.size(); ++index) {
    metadata[index] = {sizeof(PopsFieldPatchMetadataV1),
                       index,
                       0,
                       static_cast<std::int32_t>(index),
                       2,
                       {},
                       {},
                       {},
                       {},
                       POPS_FIELD_CENTERING_CELL_V1,
                       0,
                       "multilevel-layout",
                       patch_identities[index].c_str()};
    metadata[index].lower[0] = static_cast<std::int64_t>(2 * index);
    metadata[index].upper[0] = static_cast<std::int64_t>(2 * index + 1);
    metadata[index].lower[1] = metadata[index].upper[1] = 0;
    metadata[index].cell_spacing[0] = metadata[index].cell_spacing[1] = index == 0 ? 1.0 : 0.5;
  }
  PopsFieldGlobalTopologyV1 global{sizeof(PopsFieldGlobalTopologyV1),
                                   "multilevel-recipe",
                                   "multilevel-layout",
                                   "multilevel-materialization",
                                   2,
                                   {},
                                   {},
                                   0,
                                   metadata.size(),
                                   metadata.data()};
  global.domain_upper[0] = 3;
  std::array<std::uint8_t, 2> coarse_coverage{1, 0};
  std::array<std::uint8_t, 2> fine_coverage{1, 1};
  const std::vector<pops::component::FieldTopologyPatchInputV2> inputs{
      {0,
       POPS_FIELD_MATERIAL_BINARY_COVERAGE_V1,
       {sizeof(PopsConstByteViewV1), coarse_coverage.data(), coarse_coverage.size()},
       {},
       {}},
      {1,
       POPS_FIELD_MATERIAL_BINARY_COVERAGE_V1,
       {sizeof(PopsConstByteViewV1), fine_coverage.data(), fine_coverage.size()},
       {},
       {}},
  };
  struct Calls {
    int topology = 0;
    int solver = 0;
  } calls;
  PopsFieldTopologyApiV2 topology_api{
      abi_header(sizeof(PopsFieldTopologyApiV2), POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, 2),
      +[](void* raw, const PopsFieldTopologyRequestV2* request, PopsFieldTopologyResultV2* result) {
        auto& state = *static_cast<Calls*>(raw);
        ++state.topology;
        if (request->topology.patch_count != 2 || request->local_patch_count != 2 ||
            request->topology.patches[0].level != 0 || request->topology.patches[1].level != 1)
          return 7;
        for (std::size_t index = 0; index < request->local_patch_count; ++index) {
          const auto& patch = request->local_patches[index];
          if (patch.material_representation != POPS_FIELD_MATERIAL_BINARY_COVERAGE_V1 ||
              patch.material_coverage.size != 2)
            return 8;
          std::copy(patch.material_coverage.data,
                    patch.material_coverage.data + patch.material_coverage.size,
                    patch.material_mask.data);
          for (std::size_t point = 0; point < patch.component_labels.size; ++point)
            patch.component_labels.data[point] = patch.material_mask.data[point] == 1 ? 1 : 0;
        }
        result->label_count = 1;
        result->labels = labels;
        result->provenance = "multilevel-test";
        result->topology_digest = "multilevel-topology-digest";
        result->status = ok_status();
        return 0;
      }};
  const auto topology =
      pops::component::prepare_field_topology(topology_api, &calls, global, inputs, execution);
  ASSERT_EQ(topology.local_patches().size(), 2u);
  EXPECT_EQ(topology.local_patches()[0].material_mask, (std::vector<std::uint8_t>{1, 0}));
  EXPECT_EQ(topology.local_patches()[1].material_mask, (std::vector<std::uint8_t>{1, 1}));

  std::array<double, 2> coarse_rhs{2.0, 99.0}, fine_rhs{3.0, 4.0};
  std::array<double, 2> coarse_solution{}, fine_solution{};
  const auto& owned = topology.global_patches();
  const std::vector<pops::component::FieldSolverPatchBindingV2> bindings{
      {0,
       abi::const_field_view(coarse_rhs.data(), 2, 1, 1, owned[0].layout_identity,
                             owned[0].patch_identity),
       abi::field_view(coarse_solution.data(), 2, 1, 1, owned[0].layout_identity,
                       owned[0].patch_identity),
       {}},
      {1,
       abi::const_field_view(fine_rhs.data(), 2, 1, 1, owned[1].layout_identity,
                             owned[1].patch_identity),
       abi::field_view(fine_solution.data(), 2, 1, 1, owned[1].layout_identity,
                       owned[1].patch_identity),
       {}},
  };
  const auto request = pops::component::bind_field_solver_request(
      topology, bindings, execution, "{\"identity\":\"multilevel-boundary\"}", 1e-8, 0.0, 10);
  PopsFieldSolverApiV2 solver_api{
      abi_header(sizeof(PopsFieldSolverApiV2), POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, 2),
      +[](void* raw, const PopsFieldSolverRequestV2* request, PopsSolveReportV2* report) {
        auto& state = *static_cast<Calls*>(raw);
        ++state.solver;
        if (request->topology.patch_count != 2 || request->local_patch_count != 2 ||
            request->topology.patches[0].level != 0 || request->topology.patches[1].level != 1 ||
            request->local_patches[0].material_mask.data[1] != 0 ||
            request->local_patches[1].material_mask.data[1] != 1)
          return 9;
        for (std::size_t patch = 0; patch < request->local_patch_count; ++patch) {
          const auto* rhs = static_cast<const double*>(request->local_patches[patch].rhs.data);
          auto* solution = static_cast<double*>(request->local_patches[patch].solution.data);
          for (std::size_t point = 0; point < 2; ++point)
            if (request->local_patches[patch].material_mask.data[point] == 1)
              solution[point] = rhs[point];
        }
        report->status = POPS_SOLVE_SOLVED_V2;
        report->action = POPS_SOLVE_ACTION_NONE_V2;
        report->iterations = 1;
        report->relative_residual = 0.0;
        report->reference_residual_norm = 1.0;
        report->residual_norm = 0.0;
        report->reason = "multilevel batch solved";
        return 0;
      }};
  PopsSolveReportV2 report{};
  EXPECT_EQ(pops::component::solve_field(solver_api, &calls, request, report), 0);
  EXPECT_EQ(calls.topology, 1);
  EXPECT_EQ(calls.solver, 1);
  EXPECT_EQ(coarse_solution, (std::array<double, 2>{2.0, 0.0}));
  EXPECT_EQ(fine_solution, fine_rhs);
}

TEST(ComponentInterfaces, PreparedExecutionContextBindsExactExecutionLaneAuthority) {
  const PopsExecutionContextV1 execution = abi::host_execution_context();
  const pops::component::PreparedExecutionContextV1 prepared(
      execution.execution_identity, execution.context_version, execution.memory_space,
      execution.backend_identity, execution.device_identity, execution.scalar_type,
      execution.storage_precision, execution.compute_precision, execution.accumulation_precision,
      execution.reduction_precision, execution.stream_handle, execution.stream_identity,
      execution.communicator_f_handle, execution.communicator_datatype_f_handle,
      execution.communicator_identity, execution.communicator_datatype_identity);
  const auto lane =
      pops::ExecutionLane::duplicate_world_collectively("case::runtime::component-lane");
  const auto bound = prepared.for_lane(lane);

  EXPECT_TRUE(bound.matches_lane(lane));
  const PopsExecutionContextV1 view = bound.view();
#ifdef POPS_HAS_MPI
  EXPECT_EQ(std::string_view(view.communicator_identity), lane.identity());
  EXPECT_STREQ(view.communicator_datatype_identity, "MPI_DOUBLE");
#else
  EXPECT_STREQ(view.communicator_identity, "serial");
  EXPECT_STREQ(view.communicator_datatype_identity, "none");
  EXPECT_EQ(view.communicator_f_handle, 0);
  EXPECT_EQ(view.communicator_datatype_f_handle, 0);
#endif
}

}  // namespace
