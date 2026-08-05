#include <gtest/gtest.h>

#include <pops/runtime/program/same_level_cell_temporal_provider.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

using namespace pops;
using namespace pops::runtime::program;

namespace {

void ensure_runtime() {
  static Kokkos::ScopeGuard guard;
}

template <int Dim>
Extent<Dim> test_extents() {
  Extent<Dim> extents{};
  for (int axis = 0; axis < Dim; ++axis)
    extents[axis] = axis + 2;
  return extents;
}

template <int Dim>
Extent<Dim> unit_extent() {
  Extent<Dim> extents{};
  for (int axis = 0; axis < Dim; ++axis)
    extents[axis] = 1;
  return extents;
}

template <int Dim>
RealVector<Dim> physical_lower() {
  RealVector<Dim> lower{};
  for (int axis = 0; axis < Dim; ++axis)
    lower[axis] = Real(axis);
  return lower;
}

template <int Dim>
RealVector<Dim> physical_upper() {
  RealVector<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = Real(axis + 2);
  return upper;
}

template <int Dim>
class ExactRankedTransportRuntime {
 public:
  static constexpr int dimension = Dim;
  using field_type = MultiFab<Dim>;

  ExactRankedTransportRuntime()
      : domain_(Box<Dim>::from_extents(test_extents<Dim>())),
        layout_(std::vector<Box<Dim>>{domain_}),
        rank_space_(Index<Dim>{}, unit_extent<Dim>()),
        distribution_(mesh::Distribution<Dim>::partitioned(layout_, rank_space_,
                                                           std::vector<Index<Dim>>{Index<Dim>{}})),
        state_(layout_, distribution_, Index<Dim>{}, 2, unit_extent<Dim>()),
        geometry_(
            Geometry<Dim>::from_bounds(domain_, physical_lower<Dim>(), physical_upper<Dim>())) {
    periodicity_.fill(true);
    state_.set_val(Real(0));
    const auto values = state_.fab(0).view();
    same_level_cell_temporal_detail::for_each_index(
        domain_, [&](const Index<Dim>& index, std::size_t) {
          for (int component = 0; component < state_.ncomp(); ++component)
            values(index, component) = initial_value(index, component);
        });
  }

  [[nodiscard]] std::uint64_t topology_epoch() const noexcept { return topology_epoch_; }
  [[nodiscard]] std::uint64_t materialization_generation() const noexcept {
    return materialization_generation_;
  }
  [[nodiscard]] std::size_t same_level_cell_block_count() const noexcept { return 1; }
  [[nodiscard]] int same_level_cell_level_count() const noexcept { return 1; }
  [[nodiscard]] field_type& same_level_cell_state() noexcept { return state_; }
  [[nodiscard]] const Geometry<Dim>& same_level_cell_geometry() const noexcept { return geometry_; }
  [[nodiscard]] const std::array<bool, Dim>& same_level_cell_periodicity() const noexcept {
    return periodicity_;
  }
  [[nodiscard]] std::string_view same_level_cell_state_identity() const noexcept {
    return "test://exact-ranked/state";
  }
  [[nodiscard]] std::string_view same_level_cell_flux_provider_identity() const noexcept {
    return "test://exact-ranked/negative-flux-divergence";
  }
  [[nodiscard]] std::string_view same_level_cell_flux_parameter_contract() const noexcept {
    return "test.exact-ranked-spatial-contract.v1";
  }
  [[nodiscard]] bool same_level_cell_has_prepared_boundary_plan() const noexcept { return false; }

  void capture_same_level_negative_flux_divergence(
      const runtime::multiblock::BoundaryEvaluationPoint& point, field_type& state,
      field_type& residual, const std::array<field_type*, Dim>& fluxes) {
    last_point_ = point;
    ++capture_count_;
    residual.set_val(Real(0));
    const auto residual_view = residual.fab(0).view();
    same_level_cell_temporal_detail::for_each_index(
        domain_, [&](const Index<Dim>& index, std::size_t) {
          for (int component = 0; component < state.ncomp(); ++component)
            residual_view(index, component) = residual_value(index, component);
        });
    for (int axis = 0; axis < Dim; ++axis) {
      fluxes[axis]->set_val(Real(0));
      const Box<Dim> faces = fluxes[axis]->box(0);
      const auto flux_view = fluxes[axis]->fab(0).view();
      same_level_cell_temporal_detail::for_each_index(
          faces, [&](const Index<Dim>& index, std::size_t) {
            for (int component = 0; component < state.ncomp(); ++component)
              flux_view(index, component) = flux_value(axis, index, component);
          });
    }
  }

  [[nodiscard]] static Real initial_value(const Index<Dim>& index, int component) {
    Real value = Real(component + 1);
    for (int axis = 0; axis < Dim; ++axis)
      value += Real(axis + 1) * Real(index[axis] + 1) * Real(0.1);
    return value;
  }

  [[nodiscard]] static Real residual_value(const Index<Dim>& index, int component) {
    Real value = Real(component + 1) * Real(0.25);
    for (int axis = 0; axis < Dim; ++axis)
      value += Real(index[axis] + 1) * Real(0.01);
    return value;
  }

  [[nodiscard]] static Real flux_value(int axis, const Index<Dim>& index, int component) {
    Real value = Real(100 * (axis + 1) + 10 * component);
    for (int coordinate = 0; coordinate < Dim; ++coordinate)
      value += Real(coordinate + 1) * Real(index[coordinate]) * Real(0.01);
    return value;
  }

  [[nodiscard]] int capture_count() const { return capture_count_; }
  [[nodiscard]] const runtime::multiblock::BoundaryEvaluationPoint& last_point() const {
    return last_point_;
  }
  void bump_materialization_generation() { ++materialization_generation_; }
  void replace_field_metadata_without_generation() {
    state_ = field_type(layout_, distribution_, Index<Dim>{}, state_.ncomp(), Extent<Dim>{});
  }

 private:
  Box<Dim> domain_;
  mesh::BoxArray<Dim> layout_;
  mesh::RankSpace<Dim> rank_space_;
  mesh::Distribution<Dim> distribution_;
  field_type state_;
  Geometry<Dim> geometry_;
  std::array<bool, Dim> periodicity_{};
  std::uint64_t topology_epoch_ = 17;
  std::uint64_t materialization_generation_ = 31;
  int capture_count_ = 0;
  runtime::multiblock::BoundaryEvaluationPoint last_point_{};
};

template <int Dim>
using ExactProvider =
    PreparedSameLevelTransportEulerStageFluxProvider<Dim, ExactRankedTransportRuntime<Dim>>;

static_assert(SameLevelCellTemporalRuntime<1, ExactRankedTransportRuntime<1>>);
static_assert(SameLevelCellTemporalRuntime<2, ExactRankedTransportRuntime<2>>);
static_assert(SameLevelCellTemporalRuntime<3, ExactRankedTransportRuntime<3>>);
static_assert(CellTemporalStageFluxProvider<ExactProvider<1>>);
static_assert(CellTemporalStageFluxProvider<ExactProvider<2>>);
static_assert(CellTemporalStageFluxProvider<ExactProvider<3>>);
static_assert(CellTemporalRungBatchLifecycle<ExactProvider<1>>);
static_assert(CellTemporalAcceptedBoundaryLifecycle<ExactProvider<3>>);

template <int Dim>
std::string execute_exact_ranked_provider() {
  ExactRankedTransportRuntime<Dim> runtime;
  const CellTemporalPartitionAcceptedState partition =
      prepare_same_level_transport_euler_partition<Dim>(runtime, 0, 100, 0);
  auto ledger = std::make_shared<SameLevelCellIntegratedFluxLedger<Dim>>(
      runtime.topology_epoch(), runtime.materialization_generation(), 0, 0, partition.cells.size(),
      runtime.same_level_cell_state().ncomp());
  ExactProvider<Dim> provider(runtime, partition, ledger, "test.clock.exact-ranked");
  PreparedBatchedCellTemporalExecutor executor(partition, std::move(provider));
  const std::string exact_contract = executor.exact_contract();

  executor.begin_attempt(2);
  executor.advance_to_barrier();
  executor.commit();
  device_fence();

  EXPECT_EQ(runtime.capture_count(), 2);
  EXPECT_EQ(runtime.last_point().level, 0);
  EXPECT_EQ(runtime.last_point().tick, 1);
  EXPECT_DOUBLE_EQ(runtime.last_point().dt, 0.01);
  EXPECT_EQ(ledger->publication_generation(), 1u);
  EXPECT_EQ(ledger->begin_tick(), 0);
  EXPECT_EQ(ledger->end_tick(), 2);
  EXPECT_EQ(ledger->tick_denominator(), 100);

  const MultiFab<Dim>& state = runtime.same_level_cell_state();
  const Box<Dim>& box = state.box(0);
  const auto values = state.fab(0).view();
  same_level_cell_temporal_detail::for_each_index(
      box, [&](const Index<Dim>& index, std::size_t cell) {
        for (int component = 0; component < state.ncomp(); ++component) {
          EXPECT_DOUBLE_EQ(
              values(index, component),
              ExactRankedTransportRuntime<Dim>::initial_value(index, component) +
                  Real(0.02) * ExactRankedTransportRuntime<Dim>::residual_value(index, component));
          for (int axis = 0; axis < Dim; ++axis) {
            Index<Dim> high = index;
            ++high[axis];
            EXPECT_DOUBLE_EQ(
                ledger->integrated_flux(cell, {axis, SameLevelCellFaceSide::Low}, component),
                Real(0.02) * ExactRankedTransportRuntime<Dim>::flux_value(axis, index, component));
            EXPECT_DOUBLE_EQ(
                ledger->integrated_flux(cell, {axis, SameLevelCellFaceSide::High}, component),
                Real(0.02) * ExactRankedTransportRuntime<Dim>::flux_value(axis, high, component));
          }
        }
      });
  return exact_contract;
}

template <int Dim>
void prove_rollback_and_generation_authentication() {
  ExactRankedTransportRuntime<Dim> runtime;
  const CellTemporalPartitionAcceptedState partition =
      prepare_same_level_transport_euler_partition<Dim>(runtime, 0, 100, 0);
  auto ledger = std::make_shared<SameLevelCellIntegratedFluxLedger<Dim>>(
      runtime.topology_epoch(), runtime.materialization_generation(), 0, 0, partition.cells.size(),
      runtime.same_level_cell_state().ncomp());
  MultiFab<Dim> accepted = runtime.same_level_cell_state();
  ExactProvider<Dim> provider(runtime, partition, ledger, "test.clock.exact-ranked");
  PreparedBatchedCellTemporalExecutor executor(partition, std::move(provider));

  executor.begin_attempt(1);
  executor.advance_to_barrier();
  executor.rollback();
  const auto accepted_view = accepted.fab(0).view();
  const auto live_view = runtime.same_level_cell_state().fab(0).view();
  same_level_cell_temporal_detail::for_each_index(
      accepted.box(0), [&](const Index<Dim>& index, std::size_t) {
        for (int component = 0; component < accepted.ncomp(); ++component)
          EXPECT_DOUBLE_EQ(live_view(index, component), accepted_view(index, component));
      });
  EXPECT_EQ(ledger->publication_generation(), 0u);

  executor.begin_attempt(1);
  executor.advance_to_barrier();
  runtime.bump_materialization_generation();
  EXPECT_THROW(executor.commit(), std::runtime_error);
  EXPECT_EQ(ledger->publication_generation(), 0u);

  ExactRankedTransportRuntime<Dim> layout_runtime;
  const CellTemporalPartitionAcceptedState layout_partition =
      prepare_same_level_transport_euler_partition<Dim>(layout_runtime, 0, 100, 0);
  auto layout_ledger = std::make_shared<SameLevelCellIntegratedFluxLedger<Dim>>(
      layout_runtime.topology_epoch(), layout_runtime.materialization_generation(), 0, 0,
      layout_partition.cells.size(), layout_runtime.same_level_cell_state().ncomp());
  ExactProvider<Dim> layout_provider(layout_runtime, layout_partition, layout_ledger,
                                     "test.clock.exact-ranked");
  PreparedBatchedCellTemporalExecutor layout_executor(layout_partition, std::move(layout_provider));
  layout_runtime.replace_field_metadata_without_generation();
  EXPECT_THROW(layout_executor.begin_attempt(1), std::runtime_error);
  EXPECT_EQ(layout_ledger->publication_generation(), 0u);
}

}  // namespace

TEST(test_cell_temporal_program_route,
     executes_one_exact_algorithm_in_one_two_and_three_dimensions) {
  ensure_runtime();
  const std::string one = execute_exact_ranked_provider<1>();
  const std::string two = execute_exact_ranked_provider<2>();
  const std::string three = execute_exact_ranked_provider<3>();
  EXPECT_NE(one, two);
  EXPECT_NE(two, three);
  EXPECT_NE(one, three);
}

TEST(test_cell_temporal_program_route,
     rollback_and_materialization_generation_remain_atomic_in_every_dimension) {
  ensure_runtime();
  prove_rollback_and_generation_authentication<1>();
  prove_rollback_and_generation_authentication<2>();
  prove_rollback_and_generation_authentication<3>();
}
