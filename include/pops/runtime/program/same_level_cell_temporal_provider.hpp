/// @file
/// @brief Exact-ranked same-level finite-volume provider for the cell-local temporal executor.

#pragma once

#include <pops/core/foundation/allocator.hpp>
#include <pops/core/foundation/kokkos_env.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <pops/runtime/program/cell_temporal_partition_executor.hpp>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::runtime::program {

enum class SameLevelCellFaceSide : std::uint8_t { Low = 0, High = 1 };

/// One axis-qualified face of a compile-time-ranked cell.
struct SameLevelCellFace {
  int axis = 0;
  SameLevelCellFaceSide side = SameLevelCellFaceSide::Low;

  friend constexpr bool operator==(const SameLevelCellFace&, const SameLevelCellFace&) = default;
};

template <int Dim>
struct SameLevelCellIntegratedFluxLedgerAcceptedState {
  static_assert(Dim >= 1 && Dim <= 3);
  std::uint64_t topology_epoch = 0;
  std::uint64_t materialization_generation = 0;
  std::size_t block = 0;
  int level = 0;
  std::size_t cell_count = 0;
  int component_count = 0;
  std::vector<Real, fab_allocator<Real>> integrated_flux;
  std::int64_t begin_tick = 0;
  std::int64_t end_tick = 0;
  std::int64_t tick_denominator = 1;
  std::uint64_t publication_generation = 0;
};

template <int Dim, class Runtime>
class PreparedSameLevelTransportEulerStageFluxProvider;

/// Accepted, fixed-shape diagnostic view of time-integrated face fluxes.
///
/// Every cell owns two records per spatial axis. This avoids write races while retaining both
/// copies of every interior face for later conservation audits. This view never owns a second
/// conservation transaction: it is derived from and published with the accepted state candidate.
template <int Dim>
class SameLevelCellIntegratedFluxLedger {
  static_assert(Dim >= 1 && Dim <= 3);

 public:
  SameLevelCellIntegratedFluxLedger(std::uint64_t topology_epoch,
                                    std::uint64_t materialization_generation, std::size_t block,
                                    int level, std::size_t cell_count, int component_count)
      : topology_epoch_(topology_epoch),
        materialization_generation_(materialization_generation),
        block_(block),
        level_(level),
        cell_count_(cell_count),
        component_count_(component_count),
        accepted_(checked_value_count_(cell_count, component_count), Real(0)) {
    if (component_count <= 0)
      throw std::invalid_argument("same-level cell flux ledger requires components > 0");
  }

  [[nodiscard]] std::uint64_t topology_epoch() const noexcept { return topology_epoch_; }
  [[nodiscard]] std::uint64_t materialization_generation() const noexcept {
    return materialization_generation_;
  }
  [[nodiscard]] std::size_t block() const noexcept { return block_; }
  [[nodiscard]] int level() const noexcept { return level_; }
  [[nodiscard]] std::size_t cell_count() const noexcept { return cell_count_; }
  [[nodiscard]] int component_count() const noexcept { return component_count_; }
  [[nodiscard]] std::size_t value_count() const noexcept { return accepted_.size(); }
  [[nodiscard]] std::int64_t begin_tick() const noexcept { return begin_tick_; }
  [[nodiscard]] std::int64_t end_tick() const noexcept { return end_tick_; }
  [[nodiscard]] std::int64_t tick_denominator() const noexcept { return tick_denominator_; }
  [[nodiscard]] std::uint64_t publication_generation() const noexcept {
    return publication_generation_;
  }

  void copy_accepted_state_into(SameLevelCellIntegratedFluxLedgerAcceptedState<Dim>& state) const {
    state.topology_epoch = topology_epoch_;
    state.materialization_generation = materialization_generation_;
    state.block = block_;
    state.level = level_;
    state.cell_count = cell_count_;
    state.component_count = component_count_;
    state.integrated_flux.resize(accepted_.size());
    std::copy(accepted_.begin(), accepted_.end(), state.integrated_flux.begin());
    state.begin_tick = begin_tick_;
    state.end_tick = end_tick_;
    state.tick_denominator = tick_denominator_;
    state.publication_generation = publication_generation_;
  }

  [[nodiscard]] SameLevelCellIntegratedFluxLedgerAcceptedState<Dim> accepted_state() const {
    SameLevelCellIntegratedFluxLedgerAcceptedState<Dim> state;
    copy_accepted_state_into(state);
    return state;
  }

  void restore_accepted_state(const SameLevelCellIntegratedFluxLedgerAcceptedState<Dim>& state) {
    if (state.topology_epoch != topology_epoch_ ||
        state.materialization_generation != materialization_generation_ || state.block != block_ ||
        state.level != level_ || state.cell_count != cell_count_ ||
        state.component_count != component_count_ ||
        state.integrated_flux.size() != accepted_.size())
      throw std::invalid_argument(
          "same-level cell flux ledger rollback image targets another prepared layout");
    if (state.begin_tick < 0 || state.end_tick < state.begin_tick || state.tick_denominator <= 0 ||
        (state.publication_generation == 0 && state.begin_tick != state.end_tick) ||
        (state.publication_generation != 0 && state.end_tick == state.begin_tick))
      throw std::invalid_argument(
          "same-level cell flux ledger rollback image has an invalid accepted clock");

    std::copy(state.integrated_flux.begin(), state.integrated_flux.end(), accepted_.begin());
    begin_tick_ = state.begin_tick;
    end_tick_ = state.end_tick;
    tick_denominator_ = state.tick_denominator;
    publication_generation_ = state.publication_generation;
  }

  void invalidate_accepted_publication(std::int64_t synchronization_tick,
                                       std::int64_t denominator) {
    if (synchronization_tick < 0 || denominator <= 0)
      throw std::invalid_argument(
          "same-level cell flux ledger invalidation requires a valid accepted clock");
    std::fill(accepted_.begin(), accepted_.end(), Real(0));
    begin_tick_ = synchronization_tick;
    end_tick_ = synchronization_tick;
    tick_denominator_ = denominator;
    publication_generation_ = 0;
  }

  [[nodiscard]] Real integrated_flux(std::size_t cell, SameLevelCellFace face,
                                     int component) const {
    if (cell >= cell_count_ || face.axis < 0 || face.axis >= Dim || component < 0 ||
        component >= component_count_)
      throw std::out_of_range("same-level cell flux ledger index is out of range");
    return accepted_.at(storage_offset(cell, face, component, component_count_));
  }

  [[nodiscard]] POPS_HD static constexpr std::size_t storage_offset(std::size_t cell,
                                                                    SameLevelCellFace face,
                                                                    int component,
                                                                    int components) noexcept {
    const std::size_t face_ordinal =
        std::size_t{2} * static_cast<std::size_t>(face.axis) +
        static_cast<std::size_t>(face.side == SameLevelCellFaceSide::High);
    return (cell * (std::size_t{2} * Dim) + face_ordinal) * static_cast<std::size_t>(components) +
           static_cast<std::size_t>(component);
  }

 private:
  template <int, class>
  friend class PreparedSameLevelTransportEulerStageFluxProvider;

  static std::size_t checked_value_count_(std::size_t cells, int components) {
    if (components <= 0)
      return 0;
    constexpr std::size_t faces_per_cell = std::size_t{2} * Dim;
    if (static_cast<std::size_t>(components) >
        std::numeric_limits<std::size_t>::max() / faces_per_cell)
      throw std::overflow_error("same-level cell flux ledger width overflows size_t");
    const std::size_t width = faces_per_cell * static_cast<std::size_t>(components);
    if (cells > std::numeric_limits<std::size_t>::max() / width)
      throw std::overflow_error("same-level cell flux ledger size overflows size_t");
    return cells * width;
  }

  void publish_(std::int64_t begin_tick, std::int64_t end_tick, std::int64_t denominator,
                const Real* values, std::size_t count) noexcept {
    if (count != accepted_.size())
      std::terminate();
    std::copy_n(values, count, accepted_.data());
    begin_tick_ = begin_tick;
    end_tick_ = end_tick;
    tick_denominator_ = denominator;
    ++publication_generation_;
  }

  std::uint64_t topology_epoch_ = 0;
  std::uint64_t materialization_generation_ = 0;
  std::size_t block_ = 0;
  int level_ = 0;
  std::size_t cell_count_ = 0;
  int component_count_ = 0;
  std::vector<Real, fab_allocator<Real>> accepted_;
  std::int64_t begin_tick_ = 0;
  std::int64_t end_tick_ = 0;
  std::int64_t tick_denominator_ = 1;
  std::uint64_t publication_generation_ = 0;
};

inline constexpr std::string_view kSameLevelTransportEulerStageFluxProvider =
    "pops.amr.same-level-transport-euler-stage-flux@2";

/// Exact runtime seam consumed by the same-level provider.
///
/// A runtime implementation owns the live field and all topology/spatial identities. The provider
/// never guesses a dimension, geometry, generation, or flux closure from an unranked facade.
/// Before ``prepare_same_level_cell_stage_snapshot`` is called, every locally owned valid cell has
/// already been copied into ``snapshot`` and that local preparation has reached lane consensus.
/// The runtime hook materializes only the prepared halo/physical image on the supplied exact lane;
/// the following residual/flux hook receives that image as immutable stage state.
template <int Dim, class Runtime>
concept SameLevelCellTemporalRuntime =
    (Dim >= 1 && Dim <= 3) &&
    requires(Runtime& runtime, const Runtime& constant_runtime,
             const multiblock::BoundaryEvaluationPoint& point, MultiFab<Dim>& snapshot,
             const MultiFab<Dim>& immutable_snapshot, MultiFab<Dim>& residual,
             const std::array<MultiFab<Dim>*, Dim>& fluxes, const ExecutionLane& lane) {
      requires(Runtime::dimension == Dim);
      { constant_runtime.topology_epoch() } noexcept -> std::same_as<std::uint64_t>;
      { constant_runtime.materialization_generation() } noexcept -> std::same_as<std::uint64_t>;
      { constant_runtime.same_level_cell_block_count() } noexcept -> std::same_as<std::size_t>;
      { constant_runtime.same_level_cell_level_count() } noexcept -> std::same_as<int>;
      { runtime.same_level_cell_state() } noexcept -> std::same_as<MultiFab<Dim>&>;
      {
        constant_runtime.same_level_cell_geometry()
      } noexcept -> std::same_as<const Geometry<Dim>&>;
      {
        constant_runtime.same_level_cell_periodicity()
      } noexcept -> std::same_as<const std::array<bool, Dim>&>;
      {
        constant_runtime.same_level_cell_state_identity()
      } noexcept -> std::convertible_to<std::string_view>;
      {
        constant_runtime.same_level_cell_flux_provider_identity()
      } noexcept -> std::convertible_to<std::string_view>;
      {
        constant_runtime.same_level_cell_flux_parameter_contract()
      } noexcept -> std::convertible_to<std::string_view>;
      {
        constant_runtime.same_level_cell_stage_snapshot_contract()
      } noexcept -> std::convertible_to<std::string_view>;
      {
        runtime.prepare_same_level_cell_stage_snapshot(point, snapshot, lane)
      } -> std::same_as<void>;
      {
        runtime.capture_same_level_negative_flux_divergence(point, immutable_snapshot, residual,
                                                            fluxes)
      } -> std::same_as<void>;
    };

template <int Dim, class Runtime>
  requires SameLevelCellTemporalRuntime<Dim, Runtime>
CellTemporalPartitionAcceptedState prepare_same_level_transport_euler_partition(
    Runtime& runtime, std::int64_t synchronization_tick, std::int64_t tick_denominator, int rung,
    const ExecutionLane& lane) {
  std::optional<CellTemporalPartitionAcceptedState> candidate;
  std::exception_ptr local_error;
  try {
    if (runtime.same_level_cell_block_count() != 1 || runtime.same_level_cell_level_count() != 1)
      throw std::invalid_argument(
          "same-level transport Euler partition requires one prepared block and one level");
    if (rung < 0 || rung > 30 || synchronization_tick < 0 || tick_denominator <= 0 ||
        synchronization_tick % (std::int64_t{1} << rung) != 0)
      throw std::invalid_argument(
          "same-level transport Euler partition has invalid tick/rung data");
    const MultiFab<Dim>& state = runtime.same_level_cell_state();
    if (state.layout().empty() || (lane.size() > 1 && state.distribution().replicated()) ||
        state.rank_space().size() != static_cast<std::size_t>(lane.size()) ||
        state.rank_space().coordinate(static_cast<std::size_t>(lane.rank())) != state.local_rank())
      throw std::invalid_argument(
          "same-level transport Euler partition rank space differs from its execution lane");

    candidate.emplace();
    candidate->kind = TemporalPartitionKind::CellLocal;
    candidate->provider_identity = std::string(kSameLevelTransportEulerStageFluxProvider);
    candidate->topology_epoch = runtime.topology_epoch();
    candidate->synchronization_tick = synchronization_tick;
    candidate->tick_denominator = tick_denominator;
    std::uint64_t global_cell = 0;
    for (const Box<Dim>& patch : state.layout().boxes()) {
      const std::int64_t count = patch.numPts();
      if (count <= 0 || static_cast<std::uint64_t>(count) >
                            std::numeric_limits<std::uint64_t>::max() - global_cell)
        throw std::overflow_error("same-level transport Euler partition cell count is invalid");
      for (std::int64_t ordinal = 0; ordinal < count; ++ordinal)
        candidate->cells.push_back({0, global_cell++, rung, synchronization_tick});
    }
    validate_cell_temporal_partition_state(*candidate);
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error(
        "same-level transport Euler partition preparation failed on another execution-lane rank");
  }
  return std::move(*candidate);
}

namespace same_level_cell_temporal_detail {

template <int Dim>
mesh::BoxArray<Dim> face_boxes(const mesh::BoxArray<Dim>& cells, int axis) {
  if (axis < 0 || axis >= Dim)
    throw std::invalid_argument("same-level face layout axis is outside the exact rank");
  std::vector<Box<Dim>> boxes;
  boxes.reserve(cells.size());
  for (const Box<Dim>& box : cells.boxes()) {
    Box<Dim> faces = box;
    faces.hi[axis] =
        detail::checked_box_index(static_cast<std::int64_t>(faces.hi[axis]) + 1,
                                  "same-level face layout exceeds the signed index range");
    boxes.push_back(faces);
  }
  return mesh::BoxArray<Dim>(std::move(boxes));
}

template <int Dim>
mesh::Distribution<Dim> distribution_for_layout(const MultiFab<Dim>& source,
                                                const mesh::BoxArray<Dim>& layout) {
  if (source.distribution().replicated())
    return mesh::Distribution<Dim>::replicated(layout, source.rank_space());
  return mesh::Distribution<Dim>::partitioned(layout, source.rank_space(),
                                              source.distribution().owners());
}

template <int Dim>
MultiFab<Dim> field_like(const MultiFab<Dim>& source, mesh::BoxArray<Dim> layout,
                         Extent<Dim> ghosts) {
  auto distribution = distribution_for_layout(source, layout);
  return MultiFab<Dim>(std::move(layout), std::move(distribution), source.local_rank(),
                       source.ncomp(), ghosts);
}

template <int Dim>
Index<Dim> index_from_ordinal(const Box<Dim>& box, std::size_t ordinal) noexcept {
  Index<Dim> index{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t extent = static_cast<std::size_t>(box.length(axis));
    index[axis] = box.lo[axis] + static_cast<int>(ordinal % extent);
    ordinal /= extent;
  }
  return index;
}

template <int Dim, class Function>
void for_each_index(const Box<Dim>& box, Function&& function) {
  const auto count = static_cast<std::size_t>(box.numPts());
  for (std::size_t ordinal = 0; ordinal < count; ++ordinal)
    function(index_from_ordinal(box, ordinal), ordinal);
}

template <int Dim>
void copy_local_storage(const MultiFab<Dim>& source, MultiFab<Dim>& destination) {
  if (source.layout() != destination.layout() ||
      source.distribution() != destination.distribution() ||
      source.local_rank() != destination.local_rank() || source.ncomp() != destination.ncomp() ||
      source.ghosts() != destination.ghosts() || source.local_size() != destination.local_size())
    throw std::invalid_argument("same-level temporal candidate storage contract changed");
  for (std::size_t local = 0; local < source.local_size(); ++local)
    Kokkos::deep_copy(destination.fab(local).storage(), source.fab(local).storage());
}

POPS_HD inline bool finite_device_value(Real value) noexcept {
  return value == value && value <= std::numeric_limits<Real>::max() &&
         value >= -std::numeric_limits<Real>::max();
}

template <int Dim>
struct SameLevelLocalCellAddress {
  std::size_t local_fab = 0;
  Index<Dim> index{};
  std::uint64_t global_cell = 0;
};

template <int Dim>
struct SameLevelTransportEulerDeviceCell {
  Index<Dim> index{};
  std::uint64_t global_cell = 0;
  FieldView<const Real, Dim> stage;
  FieldView<const Real, Dim> residual;
  std::array<FieldView<const Real, Dim>, Dim> fluxes{};
  FieldView<Real, Dim> candidate_a;
  FieldView<Real, Dim> candidate_b;
};

template <int Dim>
struct SameLevelTransportEulerDeviceView {
  const SameLevelTransportEulerDeviceCell<Dim>* cells = nullptr;
  Real* integrated_flux = nullptr;
  Real seconds_per_tick = Real(0);
  std::size_t cell_count = 0;
  int component_count = 0;
  bool current_is_a = true;
  int expected_rung = 0;
  std::int64_t expected_begin_tick = 0;
  std::int64_t expected_end_tick = 0;
  std::int64_t expected_tick_denominator = 1;

  [[nodiscard]] POPS_HD CellTemporalStageOutcome
  evaluate_local_stage_and_record_space_time_flux(CellTemporalStagePoint point) const noexcept {
    if (point.level != 0 || point.rung != expected_rung || point.local_record_index >= cell_count ||
        component_count <= 0 || integrated_flux == nullptr || cells == nullptr ||
        point.begin_tick != expected_begin_tick || point.end_tick != expected_end_tick ||
        point.tick_denominator != expected_tick_denominator || point.end_tick <= point.begin_tick)
      return CellTemporalStageOutcome::failed(0x756001u);
    const SameLevelTransportEulerDeviceCell<Dim>& cell = cells[point.local_record_index];
    if (point.cell != cell.global_cell)
      return CellTemporalStageOutcome::failed(0x756001u);
    const Index<Dim> index = cell.index;
    const FieldView<Real, Dim> candidate = current_is_a ? cell.candidate_b : cell.candidate_a;
    const Real dt = static_cast<Real>(point.end_tick - point.begin_tick) * seconds_per_tick;
    if (!(dt > Real(0)) || !finite_device_value(dt))
      return CellTemporalStageOutcome::failed(0x756002u);

    for (int component = 0; component < component_count; ++component) {
      const Real next = cell.stage(index, component) + dt * cell.residual(index, component);
      std::array<Real, std::size_t{2} * Dim> integrated_faces{};
      bool finite = finite_device_value(next);
      for (int axis = 0; axis < Dim; ++axis) {
        Index<Dim> high = index;
        ++high[axis];
        integrated_faces[std::size_t{2} * axis] = dt * cell.fluxes[axis](index, component);
        integrated_faces[std::size_t{2} * axis + 1] = dt * cell.fluxes[axis](high, component);
        finite = finite && finite_device_value(integrated_faces[std::size_t{2} * axis]) &&
                 finite_device_value(integrated_faces[std::size_t{2} * axis + 1]);
      }
      if (!finite)
        return CellTemporalStageOutcome::rejected(0x756003u);
      candidate(index, component) = next;
      for (int axis = 0; axis < Dim; ++axis) {
        const SameLevelCellFace low{axis, SameLevelCellFaceSide::Low};
        const SameLevelCellFace high{axis, SameLevelCellFaceSide::High};
        integrated_flux[SameLevelCellIntegratedFluxLedger<Dim>::storage_offset(
            point.local_record_index, low, component, component_count)] +=
            integrated_faces[std::size_t{2} * axis];
        integrated_flux[SameLevelCellIntegratedFluxLedger<Dim>::storage_offset(
            point.local_record_index, high, component, component_count)] +=
            integrated_faces[std::size_t{2} * axis + 1];
      }
    }
    return CellTemporalStageOutcome::accepted();
  }
};

static_assert(std::is_trivially_copyable_v<SameLevelTransportEulerDeviceCell<1>>);
static_assert(std::is_trivially_copyable_v<SameLevelTransportEulerDeviceCell<2>>);
static_assert(std::is_trivially_copyable_v<SameLevelTransportEulerDeviceCell<3>>);
static_assert(std::is_trivially_copyable_v<SameLevelTransportEulerDeviceView<1>>);
static_assert(std::is_trivially_copyable_v<SameLevelTransportEulerDeviceView<2>>);
static_assert(std::is_trivially_copyable_v<SameLevelTransportEulerDeviceView<3>>);

}  // namespace same_level_cell_temporal_detail

/// Prepared forward-Euler consumer over one immutable exact-ranked runtime authority.
template <int Dim, class Runtime>
class PreparedSameLevelTransportEulerStageFluxProvider {
  static_assert(SameLevelCellTemporalRuntime<Dim, Runtime>,
                "same-level transport provider requires an exact-ranked runtime authority");

 public:
  using field_type = MultiFab<Dim>;
  using ledger_type = SameLevelCellIntegratedFluxLedger<Dim>;
  using DeviceView = same_level_cell_temporal_detail::SameLevelTransportEulerDeviceView<Dim>;

  PreparedSameLevelTransportEulerStageFluxProvider(
      Runtime& runtime, const CellTemporalPartitionAcceptedState& partition,
      std::shared_ptr<ledger_type> ledger, std::string clock_identity, const ExecutionLane& lane)
      : lane_(&lane),
        lane_borrow_(lane.borrow_immutably()),
        runtime_(&runtime),
        ledger_(std::move(ledger)),
        clock_identity_(std::move(clock_identity)),
        topology_epoch_(runtime.topology_epoch()),
        materialization_generation_(runtime.materialization_generation()) {
    std::exception_ptr local_error;
    try {
      validate_and_materialize_(partition);
      device_fence();
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, *lane_) != 0) {
      if (local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error(
          "same-level transport provider preparation failed on another execution-lane rank");
    }
  }

  PreparedSameLevelTransportEulerStageFluxProvider(
      const PreparedSameLevelTransportEulerStageFluxProvider&) = delete;
  PreparedSameLevelTransportEulerStageFluxProvider& operator=(
      const PreparedSameLevelTransportEulerStageFluxProvider&) = delete;
  PreparedSameLevelTransportEulerStageFluxProvider(
      PreparedSameLevelTransportEulerStageFluxProvider&&) noexcept = default;
  PreparedSameLevelTransportEulerStageFluxProvider& operator=(
      PreparedSameLevelTransportEulerStageFluxProvider&&) noexcept = default;

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.amr.same-level-transport-euler-stage-flux", 2};
  }
  [[nodiscard]] static constexpr bool supports_default_execution_space() noexcept { return true; }
  [[nodiscard]] static constexpr PreparedCellTemporalStageFluxContractV1
  stage_flux_contract() noexcept {
    return {};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.bytes(exact_parameters_);
  }
  [[nodiscard]] const ExecutionLane& execution_lane() const noexcept { return *lane_; }
  [[nodiscard]] std::span<const std::size_t> local_record_indices() const noexcept {
    return local_record_indices_;
  }

  [[nodiscard]] PreparedProviderSupport begin_attempt(
      CellTemporalAttemptDescriptor attempt) noexcept {
    if (active_)
      return PreparedProviderSupport::reject(0x756101u, "provider attempt is already active");
    if (!runtime_storage_is_current_())
      return PreparedProviderSupport::reject(0x756104u,
                                             "provider storage is stale after topology change");
    if (attempt.topology_epoch != topology_epoch_ || attempt.begin_tick != synchronization_tick_ ||
        attempt.target_tick <= attempt.begin_tick ||
        attempt.tick_denominator != tick_denominator_ || attempt.cell_count != global_cell_count_ ||
        attempt.local_cell_count != local_cell_count_)
      return PreparedProviderSupport::reject(0x756105u,
                                             "attempt differs from prepared temporal authority");
    device_fence();
    try {
      same_level_cell_temporal_detail::copy_local_storage(*live_, state_a_);
    } catch (...) {
      return PreparedProviderSupport::reject(0x756109u, "provider candidate copy failed");
    }
    std::fill(attempt_flux_.begin(), attempt_flux_.end(), Real(0));
    current_is_a_ = true;
    attempt_begin_tick_ = attempt.begin_tick;
    attempt_target_tick_ = attempt.target_tick;
    current_tick_ = attempt.begin_tick;
    active_ = true;
    batch_local_prepared_ = false;
    batch_active_ = false;
    return PreparedProviderSupport::accept();
  }

  void prepare_rung_batch_local(CellTemporalRungBatchDescriptor batch) {
    if (!active_ || batch_local_prepared_ || batch_active_ || batch.rung != common_rung_ ||
        batch.begin_tick != current_tick_ ||
        batch.end_tick - batch.begin_tick != (std::int64_t{1} << common_rung_) ||
        batch.end_tick > attempt_target_tick_ || batch.tick_denominator != tick_denominator_ ||
        batch.cell_count != global_cell_count_ || batch.local_cell_count != local_cell_count_)
      throw std::logic_error("same-level transport provider received an unprepared rung batch");
    const Real dt = static_cast<Real>(batch.end_tick - batch.begin_tick) * seconds_per_tick_;
    batch_point_.clock = clock_identity_;
    batch_point_.tick = batch.begin_tick;
    batch_point_.level = 0;
    batch_point_.substep =
        static_cast<int>((batch.begin_tick - attempt_begin_tick_) >> common_rung_);
    batch_point_.stage = 0;
    batch_point_.stage_fraction = amr::Rational(0, 1);
    batch_point_.dt = static_cast<double>(dt);
    batch_point_.physical_time = static_cast<double>(batch.begin_tick) * seconds_per_tick_;
    same_level_cell_temporal_detail::copy_local_storage(current_state_(), candidate_state_());
    same_level_cell_temporal_detail::copy_local_storage(current_state_(), stage_snapshot_);
    device_fence();
    prepared_batch_ = batch;
    batch_end_tick_ = batch.end_tick;
    batch_local_prepared_ = true;
  }

  void materialize_rung_batch_snapshot(CellTemporalRungBatchDescriptor batch) {
    if (!active_ || !batch_local_prepared_ || batch_active_ || batch.rung != prepared_batch_.rung ||
        batch.begin_tick != prepared_batch_.begin_tick ||
        batch.end_tick != prepared_batch_.end_tick ||
        batch.tick_denominator != prepared_batch_.tick_denominator ||
        batch.cell_count != prepared_batch_.cell_count ||
        batch.local_cell_count != prepared_batch_.local_cell_count)
      throw std::logic_error(
          "same-level transport provider snapshot phase differs from local preparation");
    runtime_->prepare_same_level_cell_stage_snapshot(batch_point_, stage_snapshot_, *lane_);
    const auto fluxes = face_flux_pointers_();
    runtime_->capture_same_level_negative_flux_divergence(
        batch_point_, std::as_const(stage_snapshot_), residual_, fluxes);
    device_fence();
    batch_local_prepared_ = false;
    batch_active_ = true;
  }

  void complete_rung_batch(CellTemporalRungBatchDescriptor) noexcept {
    current_is_a_ = !current_is_a_;
    current_tick_ = batch_end_tick_;
    batch_local_prepared_ = false;
    batch_active_ = false;
  }

  [[nodiscard]] DeviceView device_view() const noexcept {
    if (!active_ || !batch_active_)
      return {};
    DeviceView view;
    view.cells = device_cells_.data();
    view.integrated_flux = attempt_flux_.data();
    view.seconds_per_tick = seconds_per_tick_;
    view.cell_count = local_cell_count_;
    view.component_count = component_count_;
    view.current_is_a = current_is_a_;
    view.expected_rung = common_rung_;
    view.expected_begin_tick = current_tick_;
    view.expected_end_tick = batch_end_tick_;
    view.expected_tick_denominator = tick_denominator_;
    return view;
  }

  [[nodiscard]] PreparedProviderSupport prepare_commit_attempt() noexcept {
    device_fence();
    if (!active_ || batch_local_prepared_ || batch_active_ || current_tick_ != attempt_target_tick_)
      return PreparedProviderSupport::reject(
          0x756106u, "provider did not reach its prepared synchronization barrier");
    if (!runtime_storage_is_current_())
      return PreparedProviderSupport::reject(
          0x756107u, "provider storage changed before accepted publication");
    if (!ledger_ || ledger_->topology_epoch() != topology_epoch_ ||
        ledger_->materialization_generation() != materialization_generation_ ||
        ledger_->block() != 0 || ledger_->level() != 0 ||
        ledger_->cell_count() != local_cell_count_ ||
        ledger_->component_count() != component_count_)
      return PreparedProviderSupport::reject(
          0x756108u, "provider flux ledger changed before accepted publication");
    return PreparedProviderSupport::accept();
  }

  void commit_attempt() noexcept {
    const PreparedProviderSupport support = prepare_commit_attempt();
    if (!support.well_formed() || !support.accepted())
      std::terminate();
    try {
      same_level_cell_temporal_detail::copy_local_storage(current_state_(), *live_);
      device_fence();
    } catch (...) {
      std::terminate();
    }
    ledger_->publish_(attempt_begin_tick_, attempt_target_tick_, tick_denominator_,
                      attempt_flux_.data(), attempt_flux_.size());
    synchronization_tick_ = attempt_target_tick_;
    active_ = false;
    batch_local_prepared_ = false;
    batch_active_ = false;
  }

  void rollback_attempt() noexcept {
    if (!active_)
      return;
    device_fence();
    active_ = false;
    batch_local_prepared_ = false;
    batch_active_ = false;
    current_is_a_ = true;
  }

  void restore_accepted_boundary(const CellTemporalPartitionAcceptedState& accepted) noexcept {
    synchronization_tick_ = accepted.synchronization_tick;
    attempt_begin_tick_ = synchronization_tick_;
    attempt_target_tick_ = synchronization_tick_;
    current_tick_ = synchronization_tick_;
    batch_end_tick_ = synchronization_tick_;
    active_ = false;
    batch_local_prepared_ = false;
    batch_active_ = false;
    current_is_a_ = true;
  }

 private:
  [[nodiscard]] bool runtime_storage_is_current_() const noexcept {
    const field_type& current = runtime_->same_level_cell_state();
    return runtime_->topology_epoch() == topology_epoch_ &&
           runtime_->materialization_generation() == materialization_generation_ &&
           runtime_->same_level_cell_block_count() == 1 &&
           runtime_->same_level_cell_level_count() == 1 && &current == live_ &&
           current.layout() == prepared_layout_ &&
           current.distribution() == prepared_distribution_ &&
           current.local_rank() == prepared_local_rank_ && current.ghosts() == prepared_ghosts_ &&
           current.ncomp() == component_count_ && prepared_geometry_.has_value() &&
           runtime_->same_level_cell_geometry() == *prepared_geometry_ &&
           runtime_->same_level_cell_periodicity() == prepared_periodicity_ &&
           std::string_view(runtime_->same_level_cell_state_identity()) ==
               prepared_state_identity_ &&
           std::string_view(runtime_->same_level_cell_flux_provider_identity()) ==
               prepared_flux_provider_identity_ &&
           std::string_view(runtime_->same_level_cell_flux_parameter_contract()) ==
               prepared_flux_parameter_contract_ &&
           std::string_view(runtime_->same_level_cell_stage_snapshot_contract()) ==
               prepared_stage_snapshot_contract_;
  }

  [[nodiscard]] field_type& current_state_() const noexcept {
    return current_is_a_ ? state_a_ : state_b_;
  }

  [[nodiscard]] field_type& candidate_state_() const noexcept {
    return current_is_a_ ? state_b_ : state_a_;
  }

  [[nodiscard]] std::array<field_type*, Dim> face_flux_pointers_() noexcept {
    std::array<field_type*, Dim> result{};
    for (int axis = 0; axis < Dim; ++axis)
      result[axis] = &face_fluxes_[axis];
    return result;
  }

  void validate_and_materialize_(const CellTemporalPartitionAcceptedState& partition) {
    validate_cell_temporal_partition_state(partition);
    if (partition.provider_identity != kSameLevelTransportEulerStageFluxProvider ||
        runtime_->same_level_cell_block_count() != 1 ||
        runtime_->same_level_cell_level_count() != 1)
      throw std::invalid_argument(
          "same-level transport provider requires its exact one-block/one-level partition");
    if (clock_identity_.empty())
      throw std::invalid_argument(
          "same-level transport provider requires a non-empty clock identity");
    live_ = &runtime_->same_level_cell_state();
    if (live_->layout().empty() || (lane_->size() > 1 && live_->distribution().replicated()) ||
        live_->rank_space().size() != static_cast<std::size_t>(lane_->size()) ||
        live_->rank_space().coordinate(static_cast<std::size_t>(lane_->rank())) !=
            live_->local_rank())
      throw std::invalid_argument(
          "same-level transport provider rank space differs from its execution lane");
    if (std::string_view(runtime_->same_level_cell_state_identity()).empty() ||
        std::string_view(runtime_->same_level_cell_flux_provider_identity()).empty() ||
        std::string_view(runtime_->same_level_cell_flux_parameter_contract()).empty() ||
        std::string_view(runtime_->same_level_cell_stage_snapshot_contract()).empty())
      throw std::invalid_argument(
          "same-level transport provider requires exact state and spatial contracts");
    const Geometry<Dim>& geometry = runtime_->same_level_cell_geometry();
    for (const Box<Dim>& box : live_->layout().boxes())
      if (!geometry.domain().contains(box))
        throw std::invalid_argument("same-level transport patch exceeds its exact-ranked geometry");
    component_count_ = live_->ncomp();
    prepared_layout_ = live_->layout();
    prepared_distribution_ = live_->distribution();
    prepared_local_rank_ = live_->local_rank();
    prepared_ghosts_ = live_->ghosts();
    prepared_geometry_ = runtime_->same_level_cell_geometry();
    prepared_periodicity_ = runtime_->same_level_cell_periodicity();
    prepared_state_identity_ = runtime_->same_level_cell_state_identity();
    prepared_flux_provider_identity_ = runtime_->same_level_cell_flux_provider_identity();
    prepared_flux_parameter_contract_ = runtime_->same_level_cell_flux_parameter_contract();
    prepared_stage_snapshot_contract_ = runtime_->same_level_cell_stage_snapshot_contract();
    global_cell_count_ = partition.cells.size();
    if (partition.topology_epoch != topology_epoch_ || global_cell_count_ == 0)
      throw std::invalid_argument("same-level transport partition differs from the live topology");
    common_rung_ = partition.cells.front().rung;
    std::size_t record_index = 0;
    for (std::size_t global_patch = 0; global_patch < live_->layout().size(); ++global_patch) {
      const Box<Dim>& box = live_->layout()[global_patch];
      const std::size_t count = static_cast<std::size_t>(box.numPts());
      const bool owned = live_->contains_local(global_patch);
      const std::size_t local_fab = owned ? live_->local_index_of(global_patch) : 0;
      for (std::size_t ordinal = 0; ordinal < count; ++ordinal, ++record_index) {
        if (record_index >= partition.cells.size())
          throw std::invalid_argument(
              "same-level transport partition is shorter than its topology");
        const CellTemporalPartitionRecord& cell = partition.cells[record_index];
        if (cell.level != 0 || cell.cell != static_cast<std::uint64_t>(record_index) ||
            cell.rung != common_rung_)
          throw std::invalid_argument(
              "same-level transport provider requires topology-derived cells on one common rung");
        if (owned) {
          local_record_indices_.push_back(record_index);
          local_locations_.push_back(
              {local_fab, same_level_cell_temporal_detail::index_from_ordinal(box, ordinal),
               cell.cell});
        }
      }
    }
    if (record_index != partition.cells.size())
      throw std::invalid_argument("same-level transport partition is longer than its topology");
    local_cell_count_ = local_record_indices_.size();
    if (!ledger_ || ledger_->topology_epoch() != topology_epoch_ ||
        ledger_->materialization_generation() != materialization_generation_ ||
        ledger_->block() != 0 || ledger_->level() != 0 ||
        ledger_->cell_count() != local_cell_count_ ||
        ledger_->component_count() != component_count_)
      throw std::invalid_argument("same-level transport provider received the wrong flux ledger");

    synchronization_tick_ = partition.synchronization_tick;
    tick_denominator_ = partition.tick_denominator;
    seconds_per_tick_ = Real(1) / static_cast<Real>(tick_denominator_);
    state_a_ = *live_;
    state_b_ = *live_;
    stage_snapshot_ = *live_;
    residual_ = same_level_cell_temporal_detail::field_like(*live_, live_->layout(), Extent<Dim>{});
    for (int axis = 0; axis < Dim; ++axis)
      face_fluxes_[axis] = same_level_cell_temporal_detail::field_like(
          *live_, same_level_cell_temporal_detail::face_boxes(live_->layout(), axis),
          Extent<Dim>{});
    attempt_flux_.assign(ledger_->value_count(), Real(0));
    materialize_device_tables_();
    current_is_a_ = true;

    const std::array<bool, Dim>& periodicity = runtime_->same_level_cell_periodicity();
    ExactContractBuilder parameters;
    parameters.text("pops.amr.same-level-transport-euler-stage-flux")
        .scalar(std::uint32_t{2})
        .scalar(std::int32_t{Dim})
        .text(runtime_->same_level_cell_state_identity())
        .text(runtime_->same_level_cell_flux_provider_identity())
        .bytes(runtime_->same_level_cell_flux_parameter_contract())
        .bytes(runtime_->same_level_cell_stage_snapshot_contract())
        .text("forward-euler")
        .text("negative-flux-divergence")
        .text("frozen-attempt-auxiliary-fields")
        .text(clock_identity_)
        .text(lane_->identity())
        .scalar(seconds_per_tick_)
        .scalar(topology_epoch_)
        .scalar(materialization_generation_)
        .scalar(static_cast<std::int32_t>(common_rung_))
        .scalar(tick_denominator_)
        .scalar(static_cast<std::int32_t>(component_count_))
        .scalar(static_cast<std::int32_t>(live_->distribution().mode()));
    for (int axis = 0; axis < Dim; ++axis) {
      parameters.scalar(static_cast<std::int32_t>(live_->ghosts()[axis]))
          .scalar(static_cast<std::int32_t>(geometry.domain().lo[axis]))
          .scalar(static_cast<std::int32_t>(geometry.domain().hi[axis]))
          .scalar(geometry.lower()[axis])
          .scalar(geometry.upper()[axis])
          .scalar(periodicity[axis])
          .scalar(static_cast<std::int32_t>(live_->rank_space().origin()[axis]))
          .scalar(static_cast<std::int64_t>(live_->rank_space().extent()[axis]));
    }
    parameters.sequence(live_->layout().boxes(),
                        [](ExactContractBuilder& item, const Box<Dim>& box) {
                          for (int axis = 0; axis < Dim; ++axis)
                            item.scalar(static_cast<std::int32_t>(box.lo[axis]))
                                .scalar(static_cast<std::int32_t>(box.hi[axis]));
                        });
    parameters.sequence(live_->distribution().owners(),
                        [](ExactContractBuilder& item, const Index<Dim>& owner) {
                          for (int axis = 0; axis < Dim; ++axis)
                            item.scalar(static_cast<std::int32_t>(owner[axis]));
                        });
    exact_parameters_ = std::move(parameters).release();
  }

  void materialize_device_tables_() {
    device_cells_.reserve(local_locations_.size());
    for (const auto& address : local_locations_) {
      same_level_cell_temporal_detail::SameLevelTransportEulerDeviceCell<Dim> cell;
      cell.index = address.index;
      cell.global_cell = address.global_cell;
      cell.stage = std::as_const(stage_snapshot_).fab(address.local_fab).view();
      cell.residual = std::as_const(residual_).fab(address.local_fab).view();
      for (int axis = 0; axis < Dim; ++axis)
        cell.fluxes[axis] = std::as_const(face_fluxes_[axis]).fab(address.local_fab).view();
      cell.candidate_a = state_a_.fab(address.local_fab).view();
      cell.candidate_b = state_b_.fab(address.local_fab).view();
      device_cells_.push_back(cell);
    }
  }

  const ExecutionLane* lane_ = nullptr;
  ExecutionLane::ImmutableBorrow lane_borrow_;
  Runtime* runtime_ = nullptr;
  field_type* live_ = nullptr;
  std::shared_ptr<ledger_type> ledger_;
  Real seconds_per_tick_ = Real(0);
  std::string clock_identity_;
  std::uint64_t topology_epoch_ = 0;
  std::uint64_t materialization_generation_ = 0;
  std::string exact_parameters_;
  mesh::BoxArray<Dim> prepared_layout_{};
  mesh::Distribution<Dim> prepared_distribution_{};
  Index<Dim> prepared_local_rank_{};
  Extent<Dim> prepared_ghosts_{};
  std::optional<Geometry<Dim>> prepared_geometry_;
  std::array<bool, Dim> prepared_periodicity_{};
  std::string prepared_state_identity_;
  std::string prepared_flux_provider_identity_;
  std::string prepared_flux_parameter_contract_;
  std::string prepared_stage_snapshot_contract_;
  std::size_t global_cell_count_ = 0;
  std::size_t local_cell_count_ = 0;
  std::vector<std::size_t, fab_allocator<std::size_t>> local_record_indices_;
  std::vector<same_level_cell_temporal_detail::SameLevelLocalCellAddress<Dim>> local_locations_;
  int component_count_ = 0;
  int common_rung_ = 0;
  std::int64_t synchronization_tick_ = 0;
  std::int64_t tick_denominator_ = 1;
  mutable field_type state_a_;
  mutable field_type state_b_;
  field_type stage_snapshot_;
  field_type residual_;
  std::array<field_type, Dim> face_fluxes_{};
  std::vector<
      same_level_cell_temporal_detail::SameLevelTransportEulerDeviceCell<Dim>,
      fab_allocator<same_level_cell_temporal_detail::SameLevelTransportEulerDeviceCell<Dim>>>
      device_cells_;
  mutable std::vector<Real, fab_allocator<Real>> attempt_flux_;
  bool current_is_a_ = true;
  std::int64_t attempt_begin_tick_ = 0;
  std::int64_t attempt_target_tick_ = 0;
  std::int64_t current_tick_ = 0;
  std::int64_t batch_end_tick_ = 0;
  multiblock::BoundaryEvaluationPoint batch_point_{};
  CellTemporalRungBatchDescriptor prepared_batch_{};
  bool active_ = false;
  bool batch_local_prepared_ = false;
  bool batch_active_ = false;
};

}  // namespace pops::runtime::program
