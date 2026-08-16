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

template <int Dim, class MemorySpace>
Real copied_host_value(const Fab<Dim, MemorySpace>& fab, const Index<Dim>& index, int component) {
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  const Box<Dim>& grown = fab.grown_box();
  std::size_t offset = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    offset += static_cast<std::size_t>(index[axis] - grown.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(grown.length(axis));
  }
  return host(static_cast<std::size_t>(component) * stride + offset);
}

template <int Dim>
class ExactRankedTransportRuntime {
 public:
  static constexpr int dimension = Dim;
  using field_type = MultiFab<Dim>;

  ExactRankedTransportRuntime()
      : domain_(Box<Dim>::from_extents(test_extents<Dim>())),
        layout_(split_layout_(domain_)),
        rank_space_(Index<Dim>{}, unit_extent<Dim>()),
        distribution_(mesh::Distribution<Dim>::partitioned(
            layout_, rank_space_, std::vector<Index<Dim>>(layout_.size(), Index<Dim>{}))),
        state_(layout_, distribution_, Index<Dim>{}, 2, unit_extent<Dim>()),
        geometry_(
            Geometry<Dim>::from_bounds(domain_, physical_lower<Dim>(), physical_upper<Dim>())) {
    periodicity_.fill(true);
    state_.set_val(Real(0));
    for (std::size_t local = 0; local < state_.local_size(); ++local) {
      const auto values = state_.fab(local).view();
      const int components = state_.ncomp();
      for_each_cell(state_.box(local), [=] POPS_HD(const Index<Dim>& index) {
        for (int component = 0; component < components; ++component)
          values(index, component) =
              ExactRankedTransportRuntime<Dim>::initial_value(index, component);
      });
    }
    device_fence();
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
    return "test.exact-ranked-neighbour-flux.v2";
  }
  [[nodiscard]] std::string_view same_level_cell_stage_snapshot_contract() const noexcept {
    return "test.exact-ranked-periodic-halo-snapshot.v1";
  }

  void prepare_same_level_cell_stage_snapshot(const runtime::multiblock::BoundaryEvaluationPoint&,
                                              field_type& snapshot, const ExecutionLane& lane) {
    if (lane.size() != 1)
      throw std::invalid_argument("test stage snapshot received another prepared authority");
    if (snapshot.local_size() != 2)
      throw std::logic_error("test stage snapshot expected its two prepared patches");
    std::array<FieldView<const Real, Dim>, 2> sources{};
    std::array<Box<Dim>, 2> source_boxes{};
    for (std::size_t local = 0; local < sources.size(); ++local) {
      sources[local] = std::as_const(snapshot).fab(local).view();
      source_boxes[local] = snapshot.box(local);
    }
    const int components = snapshot.ncomp();
    for (std::size_t destination = 0; destination < snapshot.local_size(); ++destination) {
      const auto destination_values = snapshot.fab(destination).view();
      for_each_cell(snapshot.fab(destination).grown_box(), [=] POPS_HD(const Index<Dim>& raw) {
        const Index<Dim> wrapped = ExactRankedTransportRuntime<Dim>::wrap_(raw);
        const auto source = source_boxes[0].contains(wrapped) ? sources[0] : sources[1];
        for (int component = 0; component < components; ++component)
          destination_values(raw, component) = source(wrapped, component);
      });
    }
    device_fence();
  }

  void capture_same_level_negative_flux_divergence(
      const runtime::multiblock::BoundaryEvaluationPoint& point, const field_type& state,
      field_type& residual, const std::array<field_type*, Dim>& fluxes) {
    last_point_ = point;
    ++capture_count_;
    for (int axis = 0; axis < Dim; ++axis) {
      fluxes[axis]->set_val(Real(0));
      for (std::size_t local = 0; local < fluxes[axis]->local_size(); ++local) {
        const Box<Dim> faces = fluxes[axis]->box(local);
        const auto flux_view = fluxes[axis]->fab(local).view();
        const auto state_view = state.fab(local).view();
        const int components = state.ncomp();
        for_each_cell(faces, [=] POPS_HD(const Index<Dim>& face) {
          Index<Dim> left = face;
          --left[axis];
          for (int component = 0; component < components; ++component)
            flux_view(face, component) =
                Real(0.5) * (state_view(left, component) + state_view(face, component));
        });
      }
    }
    device_fence();
    residual.set_val(Real(0));
    for (std::size_t local = 0; local < residual.local_size(); ++local) {
      const auto residual_view = residual.fab(local).view();
      std::array<FieldView<const Real, Dim>, Dim> flux_views{};
      for (int axis = 0; axis < Dim; ++axis)
        flux_views[axis] = std::as_const(*fluxes[axis]).fab(local).view();
      const int components = state.ncomp();
      for_each_cell(residual.box(local), [=] POPS_HD(const Index<Dim>& index) {
        for (int component = 0; component < components; ++component) {
          Real value = Real(0);
          for (int axis = 0; axis < Dim; ++axis) {
            Index<Dim> high = index;
            ++high[axis];
            value -= flux_views[axis](high, component) - flux_views[axis](index, component);
          }
          residual_view(index, component) = value;
        }
      });
    }
    device_fence();
  }

  [[nodiscard]] POPS_HD static Real initial_value(const Index<Dim>& index, int component) {
    Real value = Real(component + 1);
    for (int axis = 0; axis < Dim; ++axis)
      value += Real(axis + 1) * Real(index[axis] + 1) * Real(0.1);
    return value;
  }

  [[nodiscard]] static Real first_state(const Index<Dim>& index, int component) {
    return initial_value(index, component) + Real(0.01) * residual_of_(index, component, false);
  }

  [[nodiscard]] static Real expected_value(const Index<Dim>& index, int component) {
    return first_state(index, component) + Real(0.01) * residual_of_(index, component, true);
  }

  [[nodiscard]] static Real integrated_flux_value(int axis, const Index<Dim>& face, int component) {
    return Real(0.01) * face_flux_of_(axis, face, component, false) +
           Real(0.01) * face_flux_of_(axis, face, component, true);
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
  [[nodiscard]] POPS_HD static Index<Dim> wrap_(Index<Dim> index) {
    for (int axis = 0; axis < Dim; ++axis) {
      const int extent = axis + 2;
      index[axis] %= extent;
      if (index[axis] < 0)
        index[axis] += extent;
    }
    return index;
  }

  [[nodiscard]] static Real state_of_(Index<Dim> index, int component, bool first) {
    index = wrap_(index);
    return first ? first_state(index, component) : initial_value(index, component);
  }

  [[nodiscard]] static Real face_flux_of_(int axis, Index<Dim> face, int component, bool first) {
    Index<Dim> left = face;
    --left[axis];
    return Real(0.5) * (state_of_(left, component, first) + state_of_(face, component, first));
  }

  [[nodiscard]] static Real residual_of_(const Index<Dim>& index, int component, bool first) {
    Real result = Real(0);
    for (int axis = 0; axis < Dim; ++axis) {
      Index<Dim> high = index;
      ++high[axis];
      result -= face_flux_of_(axis, high, component, first) -
                face_flux_of_(axis, index, component, first);
    }
    return result;
  }

  static std::vector<Box<Dim>> split_layout_(const Box<Dim>& domain) {
    Box<Dim> lower = domain;
    Box<Dim> upper = domain;
    lower.hi[0] = domain.lo[0];
    upper.lo[0] = domain.lo[0] + 1;
    return {lower, upper};
  }

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
  const ExecutionLane lane =
      ExecutionLane::duplicate_world_collectively("test.cell-temporal.exact-ranked");
  const CellTemporalPartitionAcceptedState partition =
      prepare_same_level_transport_euler_partition<Dim>(runtime, 0, 100, 0, lane);
  auto ledger = std::make_shared<SameLevelCellIntegratedFluxLedger<Dim>>(
      runtime.topology_epoch(), runtime.materialization_generation(), 0, 0, partition.cells.size(),
      runtime.same_level_cell_state().ncomp());
  ExactProvider<Dim> provider(runtime, partition, ledger, "test.clock.exact-ranked", lane);
  PreparedBatchedCellTemporalExecutor executor(partition, std::move(provider), lane);
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
  std::size_t cell = 0;
  for (std::size_t local = 0; local < state.local_size(); ++local) {
    same_level_cell_temporal_detail::for_each_index(
        state.box(local), [&](const Index<Dim>& index, std::size_t) {
          for (int component = 0; component < state.ncomp(); ++component) {
            EXPECT_DOUBLE_EQ(copied_host_value(state.fab(local), index, component),
                             ExactRankedTransportRuntime<Dim>::expected_value(index, component));
            for (int axis = 0; axis < Dim; ++axis) {
              Index<Dim> high = index;
              ++high[axis];
              EXPECT_DOUBLE_EQ(
                  ledger->integrated_flux(cell, {axis, SameLevelCellFaceSide::Low}, component),
                  ExactRankedTransportRuntime<Dim>::integrated_flux_value(axis, index, component));
              EXPECT_DOUBLE_EQ(
                  ledger->integrated_flux(cell, {axis, SameLevelCellFaceSide::High}, component),
                  ExactRankedTransportRuntime<Dim>::integrated_flux_value(axis, high, component));
            }
          }
          ++cell;
        });
  }
  return exact_contract;
}

template <int Dim>
void prove_rollback_and_generation_authentication() {
  ExactRankedTransportRuntime<Dim> runtime;
  const ExecutionLane lane =
      ExecutionLane::duplicate_world_collectively("test.cell-temporal.rollback");
  const CellTemporalPartitionAcceptedState partition =
      prepare_same_level_transport_euler_partition<Dim>(runtime, 0, 100, 0, lane);
  auto ledger = std::make_shared<SameLevelCellIntegratedFluxLedger<Dim>>(
      runtime.topology_epoch(), runtime.materialization_generation(), 0, 0, partition.cells.size(),
      runtime.same_level_cell_state().ncomp());
  MultiFab<Dim> accepted = runtime.same_level_cell_state();
  ExactProvider<Dim> provider(runtime, partition, ledger, "test.clock.exact-ranked", lane);
  PreparedBatchedCellTemporalExecutor executor(partition, std::move(provider), lane);

  executor.begin_attempt(1);
  executor.advance_to_barrier();
  executor.rollback();
  for (std::size_t local = 0; local < accepted.local_size(); ++local) {
    same_level_cell_temporal_detail::for_each_index(
        accepted.box(local), [&](const Index<Dim>& index, std::size_t) {
          for (int component = 0; component < accepted.ncomp(); ++component)
            EXPECT_DOUBLE_EQ(
                copied_host_value(runtime.same_level_cell_state().fab(local), index, component),
                copied_host_value(accepted.fab(local), index, component));
        });
  }
  EXPECT_EQ(ledger->publication_generation(), 0u);

  executor.begin_attempt(1);
  executor.advance_to_barrier();
  runtime.bump_materialization_generation();
  EXPECT_THROW(executor.commit(), std::runtime_error);
  EXPECT_EQ(ledger->publication_generation(), 0u);

  ExactRankedTransportRuntime<Dim> layout_runtime;
  const CellTemporalPartitionAcceptedState layout_partition =
      prepare_same_level_transport_euler_partition<Dim>(layout_runtime, 0, 100, 0, lane);
  auto layout_ledger = std::make_shared<SameLevelCellIntegratedFluxLedger<Dim>>(
      layout_runtime.topology_epoch(), layout_runtime.materialization_generation(), 0, 0,
      layout_partition.cells.size(), layout_runtime.same_level_cell_state().ncomp());
  ExactProvider<Dim> layout_provider(layout_runtime, layout_partition, layout_ledger,
                                     "test.clock.exact-ranked", lane);
  PreparedBatchedCellTemporalExecutor layout_executor(layout_partition, std::move(layout_provider),
                                                      lane);
  layout_runtime.replace_field_metadata_without_generation();
  EXPECT_THROW(layout_executor.begin_attempt(1), std::runtime_error);
  EXPECT_EQ(layout_ledger->publication_generation(), 0u);
}

}  // namespace

TEST(test_cell_temporal_program_route,
     direct_prepared_subengine_executes_one_exact_algorithm_in_every_dimension) {
  ensure_runtime();
  const std::string one = execute_exact_ranked_provider<1>();
  const std::string two = execute_exact_ranked_provider<2>();
  const std::string three = execute_exact_ranked_provider<3>();
  EXPECT_NE(one, two);
  EXPECT_NE(two, three);
  EXPECT_NE(one, three);
}

TEST(test_cell_temporal_program_route,
     direct_prepared_subengine_rollback_and_generation_remain_atomic_in_every_dimension) {
  ensure_runtime();
  prove_rollback_and_generation_authentication<1>();
  prove_rollback_and_generation_authentication<2>();
  prove_rollback_and_generation_authentication<3>();
}
