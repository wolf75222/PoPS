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

/// Accepted, fixed-shape time-integrated face-flux publication.
///
/// Every cell owns two records per spatial axis. This avoids write races while retaining both
/// copies of every interior face for later conservation audits. Attempt-local values are published
/// only together with the accepted live-state commit.
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
    if (cell_count == 0 || component_count <= 0)
      throw std::invalid_argument("same-level cell flux ledger requires cells and components > 0");
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
    "pops.amr.same-level-transport-euler-stage-flux@1";

/// Exact runtime seam consumed by the same-level provider.
///
/// A runtime implementation owns the live field and all topology/spatial identities. The provider
/// never guesses a dimension, geometry, generation, or flux closure from an unranked facade.
template <int Dim, class Runtime>
concept SameLevelCellTemporalRuntime =
    (Dim >= 1 && Dim <= 3) &&
    requires(Runtime& runtime, const Runtime& constant_runtime,
             const multiblock::BoundaryEvaluationPoint& point, MultiFab<Dim>& state,
             MultiFab<Dim>& residual, const std::array<MultiFab<Dim>*, Dim>& fluxes) {
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
        constant_runtime.same_level_cell_has_prepared_boundary_plan()
      } noexcept -> std::same_as<bool>;
      {
        runtime.capture_same_level_negative_flux_divergence(point, state, residual, fluxes)
      } -> std::same_as<void>;
    };

template <int Dim, class Runtime>
  requires SameLevelCellTemporalRuntime<Dim, Runtime>
CellTemporalPartitionAcceptedState prepare_same_level_transport_euler_partition(
    Runtime& runtime, std::int64_t synchronization_tick, std::int64_t tick_denominator,
    int rung = 0) {
  if (n_ranks() != 1 || runtime.same_level_cell_block_count() != 1 ||
      runtime.same_level_cell_level_count() != 1)
    throw std::invalid_argument(
        "same-level transport Euler partition requires serial execution, one block and one level");
  if (rung < 0 || rung > 30 || synchronization_tick < 0 || tick_denominator <= 0 ||
      synchronization_tick % (std::int64_t{1} << rung) != 0)
    throw std::invalid_argument("same-level transport Euler partition has invalid tick/rung data");
  const MultiFab<Dim>& state = runtime.same_level_cell_state();
  if (state.layout().size() != 1 || state.local_size() != 1 || state.rank_space().size() != 1 ||
      !state.contains_local(0))
    throw std::invalid_argument(
        "same-level transport Euler partition requires one serial-owned exact-ranked patch");
  const std::int64_t count64 = state.box(0).numPts();
  if (count64 <= 0 || static_cast<std::uint64_t>(count64) >
                          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error("same-level transport Euler partition cell count is invalid");

  CellTemporalPartitionAcceptedState result;
  result.kind = TemporalPartitionKind::CellLocal;
  result.provider_identity = std::string(kSameLevelTransportEulerStageFluxProvider);
  result.topology_epoch = runtime.topology_epoch();
  result.synchronization_tick = synchronization_tick;
  result.tick_denominator = tick_denominator;
  result.cells.reserve(static_cast<std::size_t>(count64));
  for (std::uint64_t cell = 0; cell < static_cast<std::uint64_t>(count64); ++cell)
    result.cells.push_back({0, cell, rung, synchronization_tick});
  validate_cell_temporal_partition_state(result);
  return result;
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
void copy_box(const Fab<Dim>& source, Fab<Dim>& destination, const Box<Dim>& box, int components) {
  const auto source_view = source.view();
  const auto destination_view = destination.view();
  for_each_index(box, [&](const Index<Dim>& index, std::size_t) {
    for (int component = 0; component < components; ++component)
      destination_view(index, component) = source_view(index, component);
  });
}

POPS_HD inline bool finite_device_value(Real value) noexcept {
  return value == value && value <= std::numeric_limits<Real>::max() &&
         value >= -std::numeric_limits<Real>::max();
}

template <int Dim>
struct SameLevelTransportEulerDeviceView {
  FieldView<const Real, Dim> state;
  FieldView<const Real, Dim> residual;
  std::array<FieldView<const Real, Dim>, Dim> fluxes{};
  FieldView<Real, Dim> candidate;
  Real* integrated_flux = nullptr;
  Real seconds_per_tick = Real(0);
  std::size_t cell_count = 0;
  int component_count = 0;
  Box<Dim> valid_box{};
  int expected_rung = 0;
  std::int64_t expected_begin_tick = 0;
  std::int64_t expected_end_tick = 0;
  std::int64_t expected_tick_denominator = 1;

  [[nodiscard]] POPS_HD CellTemporalStageOutcome
  evaluate_local_stage_and_record_space_time_flux(CellTemporalStagePoint point) const noexcept {
    if (point.level != 0 || point.rung != expected_rung || point.record_index >= cell_count ||
        point.cell != static_cast<std::uint64_t>(point.record_index) || component_count <= 0 ||
        integrated_flux == nullptr || point.begin_tick != expected_begin_tick ||
        point.end_tick != expected_end_tick ||
        point.tick_denominator != expected_tick_denominator || point.end_tick <= point.begin_tick)
      return CellTemporalStageOutcome::failed(0x756001u);
    const Index<Dim> index = index_from_ordinal(valid_box, point.record_index);
    const Real dt = static_cast<Real>(point.end_tick - point.begin_tick) * seconds_per_tick;
    if (!(dt > Real(0)) || !finite_device_value(dt))
      return CellTemporalStageOutcome::failed(0x756002u);

    for (int component = 0; component < component_count; ++component) {
      const Real next = state(index, component) + dt * residual(index, component);
      std::array<Real, std::size_t{2} * Dim> integrated_faces{};
      bool finite = finite_device_value(next);
      for (int axis = 0; axis < Dim; ++axis) {
        Index<Dim> high = index;
        ++high[axis];
        integrated_faces[std::size_t{2} * axis] = dt * fluxes[axis](index, component);
        integrated_faces[std::size_t{2} * axis + 1] = dt * fluxes[axis](high, component);
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
            point.record_index, low, component, component_count)] +=
            integrated_faces[std::size_t{2} * axis];
        integrated_flux[SameLevelCellIntegratedFluxLedger<Dim>::storage_offset(
            point.record_index, high, component, component_count)] +=
            integrated_faces[std::size_t{2} * axis + 1];
      }
    }
    return CellTemporalStageOutcome::accepted();
  }
};

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
      std::shared_ptr<ledger_type> ledger, std::string clock_identity)
      : runtime_(&runtime),
        ledger_(std::move(ledger)),
        clock_identity_(std::move(clock_identity)),
        topology_epoch_(runtime.topology_epoch()),
        materialization_generation_(runtime.materialization_generation()) {
    validate_and_materialize_(partition);
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
    return {"pops.amr.same-level-transport-euler-stage-flux", 1};
  }
  [[nodiscard]] static constexpr bool supports_default_execution_space() noexcept {
    return host_execution_();
  }
  [[nodiscard]] static constexpr PreparedCellTemporalStageFluxContractV1
  stage_flux_contract() noexcept {
    return {};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.bytes(exact_parameters_);
  }

  [[nodiscard]] PreparedProviderSupport begin_attempt(
      CellTemporalAttemptDescriptor attempt) noexcept {
    if (active_)
      return PreparedProviderSupport::reject(0x756101u, "provider attempt is already active");
    if (!host_execution_())
      return PreparedProviderSupport::reject(0x756102u, "provider has no GPU execution proof");
    if (n_ranks() != 1)
      return PreparedProviderSupport::reject(0x756103u, "provider has no MPI execution proof");
    if (!runtime_storage_is_current_())
      return PreparedProviderSupport::reject(0x756104u,
                                             "provider storage is stale after topology change");
    if (attempt.topology_epoch != topology_epoch_ || attempt.begin_tick != synchronization_tick_ ||
        attempt.target_tick <= attempt.begin_tick ||
        attempt.tick_denominator != tick_denominator_ || attempt.cell_count != cell_count_)
      return PreparedProviderSupport::reject(0x756105u,
                                             "attempt differs from prepared temporal authority");
    device_fence();
    same_level_cell_temporal_detail::copy_box(live_->fab(0), state_a_.fab(0),
                                              live_->fab(0).grown_box(), component_count_);
    std::fill(attempt_flux_.begin(), attempt_flux_.end(), Real(0));
    current_is_a_ = true;
    attempt_begin_tick_ = attempt.begin_tick;
    attempt_target_tick_ = attempt.target_tick;
    current_tick_ = attempt.begin_tick;
    active_ = true;
    batch_active_ = false;
    return PreparedProviderSupport::accept();
  }

  void begin_rung_batch(CellTemporalRungBatchDescriptor batch) {
    if (!active_ || batch_active_ || batch.rung != common_rung_ ||
        batch.begin_tick != current_tick_ ||
        batch.end_tick - batch.begin_tick != (std::int64_t{1} << common_rung_) ||
        batch.end_tick > attempt_target_tick_ || batch.tick_denominator != tick_denominator_ ||
        batch.cell_count != cell_count_)
      throw std::logic_error("same-level transport provider received an unprepared rung batch");
    const Real dt = static_cast<Real>(batch.end_tick - batch.begin_tick) * seconds_per_tick_;
    multiblock::BoundaryEvaluationPoint point;
    point.clock = clock_identity_;
    point.tick = batch.begin_tick;
    point.level = 0;
    point.substep = static_cast<int>((batch.begin_tick - attempt_begin_tick_) >> common_rung_);
    point.stage = 0;
    point.stage_fraction = amr::Rational(0, 1);
    point.dt = static_cast<double>(dt);
    point.physical_time = static_cast<double>(batch.begin_tick) * seconds_per_tick_;
    const auto fluxes = face_flux_pointers_();
    runtime_->capture_same_level_negative_flux_divergence(point, current_state_(), residual_,
                                                          fluxes);
    same_level_cell_temporal_detail::copy_box(current_state_().fab(0), candidate_state_().fab(0),
                                              current_state_().fab(0).grown_box(),
                                              component_count_);
    batch_end_tick_ = batch.end_tick;
    batch_active_ = true;
  }

  void complete_rung_batch(CellTemporalRungBatchDescriptor) noexcept {
    current_is_a_ = !current_is_a_;
    current_tick_ = batch_end_tick_;
    batch_active_ = false;
  }

  [[nodiscard]] DeviceView device_view() const noexcept {
    if (!active_ || !batch_active_)
      return {};
    DeviceView view;
    view.state = static_cast<const field_type&>(current_state_()).fab(0).view();
    view.residual = residual_.fab(0).view();
    for (int axis = 0; axis < Dim; ++axis)
      view.fluxes[axis] = face_fluxes_[axis].fab(0).view();
    view.candidate = candidate_state_().fab(0).view();
    view.integrated_flux = attempt_flux_.data();
    view.seconds_per_tick = seconds_per_tick_;
    view.cell_count = cell_count_;
    view.component_count = component_count_;
    view.valid_box = valid_box_;
    view.expected_rung = common_rung_;
    view.expected_begin_tick = current_tick_;
    view.expected_end_tick = batch_end_tick_;
    view.expected_tick_denominator = tick_denominator_;
    return view;
  }

  [[nodiscard]] PreparedProviderSupport prepare_commit_attempt() noexcept {
    device_fence();
    if (!active_ || batch_active_ || current_tick_ != attempt_target_tick_)
      return PreparedProviderSupport::reject(
          0x756106u, "provider did not reach its prepared synchronization barrier");
    if (!runtime_storage_is_current_())
      return PreparedProviderSupport::reject(
          0x756107u, "provider storage changed before accepted publication");
    if (!ledger_ || ledger_->topology_epoch() != topology_epoch_ ||
        ledger_->materialization_generation() != materialization_generation_ ||
        ledger_->block() != 0 || ledger_->level() != 0 || ledger_->cell_count() != cell_count_ ||
        ledger_->component_count() != component_count_)
      return PreparedProviderSupport::reject(
          0x756108u, "provider flux ledger changed before accepted publication");
    return PreparedProviderSupport::accept();
  }

  void commit_attempt() noexcept {
    const PreparedProviderSupport support = prepare_commit_attempt();
    if (!support.well_formed() || !support.accepted())
      std::terminate();
    same_level_cell_temporal_detail::copy_box(current_state_().fab(0), live_->fab(0), valid_box_,
                                              component_count_);
    ledger_->publish_(attempt_begin_tick_, attempt_target_tick_, tick_denominator_,
                      attempt_flux_.data(), attempt_flux_.size());
    synchronization_tick_ = attempt_target_tick_;
    active_ = false;
    batch_active_ = false;
  }

  void rollback_attempt() noexcept {
    if (!active_)
      return;
    device_fence();
    active_ = false;
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
    batch_active_ = false;
    current_is_a_ = true;
  }

 private:
  static constexpr bool host_execution_() noexcept {
#if defined(POPS_HAS_KOKKOS)
    return std::is_same_v<typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>;
#else
    return true;
#endif
  }

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
           !runtime_->same_level_cell_has_prepared_boundary_plan();
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
        runtime_->same_level_cell_level_count() != 1 || n_ranks() != 1)
      throw std::invalid_argument(
          "same-level transport provider requires its exact serial one-block/one-level partition");
    if (clock_identity_.empty())
      throw std::invalid_argument(
          "same-level transport provider requires a non-empty clock identity");
    live_ = &runtime_->same_level_cell_state();
    if (live_->layout().size() != 1 || live_->local_size() != 1 ||
        live_->rank_space().size() != 1 || !live_->contains_local(0))
      throw std::invalid_argument(
          "same-level transport provider requires one serial-owned exact-ranked patch");
    if (std::string_view(runtime_->same_level_cell_state_identity()).empty() ||
        std::string_view(runtime_->same_level_cell_flux_provider_identity()).empty() ||
        std::string_view(runtime_->same_level_cell_flux_parameter_contract()).empty())
      throw std::invalid_argument(
          "same-level transport provider requires exact state and spatial contracts");
    if (runtime_->same_level_cell_has_prepared_boundary_plan())
      throw std::invalid_argument(
          "same-level transport provider has no exact prepared-boundary contract proof");
    valid_box_ = live_->box(0);
    if (runtime_->same_level_cell_geometry().domain() != valid_box_)
      throw std::invalid_argument(
          "same-level transport geometry differs from its exact-ranked live patch");
    const std::int64_t count64 = valid_box_.numPts();
    if (count64 <= 0 || static_cast<std::uint64_t>(count64) >
                            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
      throw std::overflow_error("same-level transport provider cell count is invalid");
    cell_count_ = static_cast<std::size_t>(count64);
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
    if (partition.topology_epoch != topology_epoch_ || partition.cells.size() != cell_count_)
      throw std::invalid_argument("same-level transport partition differs from the live topology");
    common_rung_ = partition.cells.front().rung;
    for (std::size_t index = 0; index < partition.cells.size(); ++index) {
      const CellTemporalPartitionRecord& cell = partition.cells[index];
      if (cell.level != 0 || cell.cell != static_cast<std::uint64_t>(index) ||
          cell.rung != common_rung_)
        throw std::invalid_argument(
            "same-level transport provider requires canonical cells on one common rung");
    }
    if (!ledger_ || ledger_->topology_epoch() != topology_epoch_ ||
        ledger_->materialization_generation() != materialization_generation_ ||
        ledger_->block() != 0 || ledger_->level() != 0 || ledger_->cell_count() != cell_count_ ||
        ledger_->component_count() != component_count_)
      throw std::invalid_argument("same-level transport provider received the wrong flux ledger");

    synchronization_tick_ = partition.synchronization_tick;
    tick_denominator_ = partition.tick_denominator;
    seconds_per_tick_ = Real(1) / static_cast<Real>(tick_denominator_);
    state_a_ = *live_;
    state_b_ = *live_;
    residual_ = same_level_cell_temporal_detail::field_like(*live_, live_->layout(), Extent<Dim>{});
    for (int axis = 0; axis < Dim; ++axis)
      face_fluxes_[axis] = same_level_cell_temporal_detail::field_like(
          *live_, same_level_cell_temporal_detail::face_boxes(live_->layout(), axis),
          Extent<Dim>{});
    attempt_flux_.assign(ledger_->value_count(), Real(0));
    current_is_a_ = true;

    const Geometry<Dim>& geometry = runtime_->same_level_cell_geometry();
    const std::array<bool, Dim>& periodicity = runtime_->same_level_cell_periodicity();
    ExactContractBuilder parameters;
    parameters.text("pops.amr.same-level-transport-euler-stage-flux")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(runtime_->same_level_cell_state_identity())
        .text(runtime_->same_level_cell_flux_provider_identity())
        .bytes(runtime_->same_level_cell_flux_parameter_contract())
        .text("forward-euler")
        .text("negative-flux-divergence")
        .text("frozen-attempt-auxiliary-fields")
        .text(clock_identity_)
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
          .scalar(static_cast<std::int64_t>(live_->rank_space().extent()[axis]))
          .scalar(static_cast<std::int32_t>(live_->local_rank()[axis]));
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
  Box<Dim> valid_box_{};
  std::size_t cell_count_ = 0;
  int component_count_ = 0;
  int common_rung_ = 0;
  std::int64_t synchronization_tick_ = 0;
  std::int64_t tick_denominator_ = 1;
  mutable field_type state_a_;
  mutable field_type state_b_;
  field_type residual_;
  std::array<field_type, Dim> face_fluxes_{};
  mutable std::vector<Real, fab_allocator<Real>> attempt_flux_;
  bool current_is_a_ = true;
  std::int64_t attempt_begin_tick_ = 0;
  std::int64_t attempt_target_tick_ = 0;
  std::int64_t current_tick_ = 0;
  std::int64_t batch_end_tick_ = 0;
  bool active_ = false;
  bool batch_active_ = false;
};

}  // namespace pops::runtime::program
