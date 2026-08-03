#pragma once

/// @file
/// @brief Bounded production finite-volume provider for the cell-local temporal executor.

#include <pops/core/foundation/allocator.hpp>
#include <pops/core/foundation/kokkos_env.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial_operator.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/program/cell_temporal_partition_executor.hpp>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::runtime::program {

enum class SameLevelCellFace : std::uint8_t { XLow = 0, XHigh = 1, YLow = 2, YHigh = 3 };

/// Complete accepted image of one same-level integrated-flux publication.
///
/// The static layout qualifiers are retained deliberately: a rollback may restore only the exact
/// ledger that produced the image.  This prevents a stale local-time context from publishing fluxes
/// into a rematerialized hierarchy that happens to have the same number of cells.
struct SameLevelCellIntegratedFluxLedgerAcceptedState {
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

/// Accepted, fixed-shape time-integrated face-flux publication.
///
/// Each cell owns four face records, avoiding device races while retaining both copies of an
/// interior face for later conservation audits. The accepted vector is replaced only together with
/// the provider's live-state commit. Attempt-local values never enter this object.
class SameLevelCellIntegratedFluxLedger {
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
  [[nodiscard]] std::int64_t begin_tick() const noexcept { return begin_tick_; }
  [[nodiscard]] std::int64_t end_tick() const noexcept { return end_tick_; }
  [[nodiscard]] std::int64_t tick_denominator() const noexcept { return tick_denominator_; }
  [[nodiscard]] std::uint64_t publication_generation() const noexcept {
    return publication_generation_;
  }

  /// Copy the accepted publication into caller-owned reusable storage.
  ///
  /// Program attempts keep one resident image and reuse its capacity across steps.  Any allocation
  /// therefore happens at the transaction boundary, never in the prepared rung loop.
  void copy_accepted_state_into(
      SameLevelCellIntegratedFluxLedgerAcceptedState& state) const {
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

  [[nodiscard]] SameLevelCellIntegratedFluxLedgerAcceptedState accepted_state() const {
    SameLevelCellIntegratedFluxLedgerAcceptedState state;
    copy_accepted_state_into(state);
    return state;
  }

  /// Restore one previously captured accepted publication after the hierarchy state rolls back.
  ///
  /// Every qualifier is checked before mutation.  A topology/materialization mismatch is never
  /// interpreted as an empty ledger because that would hide a stale prepared temporal provider.
  void restore_accepted_state(
      const SameLevelCellIntegratedFluxLedgerAcceptedState& state) {
    if (state.topology_epoch != topology_epoch_ ||
        state.materialization_generation != materialization_generation_ ||
        state.block != block_ || state.level != level_ || state.cell_count != cell_count_ ||
        state.component_count != component_count_ || state.integrated_flux.size() != accepted_.size())
      throw std::invalid_argument(
          "same-level cell flux ledger rollback image targets another prepared layout");
    if (state.begin_tick < 0 || state.end_tick < state.begin_tick ||
        state.tick_denominator <= 0 ||
        (state.publication_generation == 0 &&
         (state.begin_tick != 0 || state.end_tick != 0)) ||
        (state.publication_generation != 0 && state.end_tick == state.begin_tick))
      throw std::invalid_argument(
          "same-level cell flux ledger rollback image has an invalid accepted clock");

    std::copy(state.integrated_flux.begin(), state.integrated_flux.end(), accepted_.begin());
    begin_tick_ = state.begin_tick;
    end_tick_ = state.end_tick;
    tick_denominator_ = state.tick_denominator;
    publication_generation_ = state.publication_generation;
  }

  [[nodiscard]] Real integrated_flux(std::size_t cell, SameLevelCellFace face,
                                     int component) const {
    if (cell >= cell_count_ || component < 0 || component >= component_count_)
      throw std::out_of_range("same-level cell flux ledger index is out of range");
    return accepted_.at(storage_offset(cell, face, component, component_count_));
  }

  [[nodiscard]] POPS_HD static std::size_t storage_offset(std::size_t cell, SameLevelCellFace face,
                                                          int component, int components) noexcept {
    return (cell * std::size_t{4} + static_cast<std::size_t>(face)) *
               static_cast<std::size_t>(components) +
           static_cast<std::size_t>(component);
  }

 private:
  friend class PreparedSameLevelTransportEulerStageFluxProvider;

  static std::size_t checked_value_count_(std::size_t cells, int components) {
    if (components <= 0)
      return 0;
    const std::size_t width = std::size_t{4} * static_cast<std::size_t>(components);
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

/// Canonical all-cell partition accepted by the first scientific provider.
///
/// This route is deliberately synchronous within its one level: every valid cell has the same rung.
/// Heterogeneous neighbouring rungs require temporal boundary interpolation and are refused by the
/// provider rather than evaluated from stale data.
inline CellTemporalPartitionAcceptedState prepare_same_level_transport_euler_partition(
    AmrRuntime& runtime, std::int64_t synchronization_tick, std::int64_t tick_denominator,
    int rung = 0) {
  if (n_ranks() != 1 || runtime.n_blocks() != 1 || runtime.nlev() != 1)
    throw std::invalid_argument(
        "same-level transport Euler partition requires serial execution, one block and one level");
  if (rung < 0 || rung > 30 || synchronization_tick < 0 || tick_denominator <= 0 ||
      synchronization_tick % (std::int64_t{1} << rung) != 0)
    throw std::invalid_argument("same-level transport Euler partition has invalid tick/rung data");
  const MultiFab& state = runtime.level_state(0, 0);
  if (state.box_array().size() != 1 || state.local_size() != 1 || state.dmap()[0] != 0)
    throw std::invalid_argument(
        "same-level transport Euler partition requires one serial-owned level box");
  const Box2D box = state.box(0);
  const std::int64_t count64 = box.num_cells();
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

inline BoxArray face_boxes(const BoxArray& cells, bool x_faces) {
  std::vector<Box2D> boxes;
  boxes.reserve(static_cast<std::size_t>(cells.size()));
  for (const Box2D& box : cells.boxes())
    boxes.push_back(x_faces ? xface_box(box) : yface_box(box));
  return BoxArray(std::move(boxes));
}

POPS_HD inline bool finite_device_value(Real value) noexcept {
  return value == value && value <= std::numeric_limits<Real>::max() &&
         value >= -std::numeric_limits<Real>::max();
}

struct SameLevelTransportEulerDeviceView {
  ConstArray4 state;
  ConstArray4 residual;
  ConstArray4 flux_x;
  ConstArray4 flux_y;
  Array4 candidate;
  Real* integrated_flux = nullptr;
  Real seconds_per_tick = Real(0);
  std::size_t cell_count = 0;
  int component_count = 0;
  int ilo = 0;
  int jlo = 0;
  int nx = 0;
  int expected_rung = 0;
  std::int64_t expected_begin_tick = 0;
  std::int64_t expected_end_tick = 0;
  std::int64_t expected_tick_denominator = 1;

  [[nodiscard]] POPS_HD CellTemporalStageOutcome
  evaluate_local_stage_and_record_space_time_flux(CellTemporalStagePoint point) const noexcept {
    if (point.level != 0 || point.rung != expected_rung || point.record_index >= cell_count ||
        point.cell != static_cast<std::uint64_t>(point.record_index) || nx <= 0 ||
        component_count <= 0 || integrated_flux == nullptr ||
        point.begin_tick != expected_begin_tick || point.end_tick != expected_end_tick ||
        point.tick_denominator != expected_tick_denominator || point.end_tick <= point.begin_tick)
      return CellTemporalStageOutcome::failed(0x756001u);
    const std::size_t linear = point.record_index;
    const int i = ilo + static_cast<int>(linear % static_cast<std::size_t>(nx));
    const int j = jlo + static_cast<int>(linear / static_cast<std::size_t>(nx));
    const Real dt = static_cast<Real>(point.end_tick - point.begin_tick) * seconds_per_tick;
    if (!(dt > Real(0)) || !finite_device_value(dt))
      return CellTemporalStageOutcome::failed(0x756002u);

    for (int component = 0; component < component_count; ++component) {
      const Real next = state(i, j, component) + dt * residual(i, j, component);
      const Real xlo = dt * flux_x(i, j, component);
      const Real xhi = dt * flux_x(i + 1, j, component);
      const Real ylo = dt * flux_y(i, j, component);
      const Real yhi = dt * flux_y(i, j + 1, component);
      if (!finite_device_value(next) || !finite_device_value(xlo) || !finite_device_value(xhi) ||
          !finite_device_value(ylo) || !finite_device_value(yhi))
        return CellTemporalStageOutcome::rejected(0x756003u);
      candidate(i, j, component) = next;
      integrated_flux[SameLevelCellIntegratedFluxLedger::storage_offset(
          linear, SameLevelCellFace::XLow, component, component_count)] += xlo;
      integrated_flux[SameLevelCellIntegratedFluxLedger::storage_offset(
          linear, SameLevelCellFace::XHigh, component, component_count)] += xhi;
      integrated_flux[SameLevelCellIntegratedFluxLedger::storage_offset(
          linear, SameLevelCellFace::YLow, component, component_count)] += ylo;
      integrated_flux[SameLevelCellIntegratedFluxLedger::storage_offset(
          linear, SameLevelCellFace::YHigh, component, component_count)] += yhi;
    }
    return CellTemporalStageOutcome::accepted();
  }
};

}  // namespace same_level_cell_temporal_detail

/// First production consumer of ``PreparedBatchedCellTemporalExecutor``.
///
/// It reuses the selected AMR block's real flux-materialising transport closure, updates the real
/// live conservative state with forward Euler, and records the exact four face fluxes used by that
/// divergence. State and ledger remain in fixed attempt-local storage until one barrier commit.
/// The honest first envelope is host/serial, one block, one level, one box and one common rung.
class PreparedSameLevelTransportEulerStageFluxProvider {
 public:
  using DeviceView = same_level_cell_temporal_detail::SameLevelTransportEulerDeviceView;

  PreparedSameLevelTransportEulerStageFluxProvider(
      AmrRuntime& runtime, const CellTemporalPartitionAcceptedState& partition,
      std::shared_ptr<SameLevelCellIntegratedFluxLedger> ledger, std::string clock_identity)
      : runtime_(&runtime),
        ledger_(std::move(ledger)),
        clock_identity_(std::move(clock_identity)),
        topology_epoch_(runtime.topology_epoch()),
        materialization_generation_(runtime.topology_materialization_generation()) {
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
    if (runtime_->topology_epoch() != topology_epoch_ ||
        runtime_->topology_materialization_generation() != materialization_generation_)
      return PreparedProviderSupport::reject(0x756104u,
                                             "provider storage is stale after topology change");
    if (attempt.topology_epoch != topology_epoch_ || attempt.begin_tick != synchronization_tick_ ||
        attempt.target_tick <= attempt.begin_tick ||
        attempt.tick_denominator != tick_denominator_ || attempt.cell_count != cell_count_)
      return PreparedProviderSupport::reject(0x756105u,
                                             "attempt differs from prepared temporal authority");
    device_fence();
    std::copy_n(live_->fab(0).data(), static_cast<std::size_t>(live_->fab(0).size()),
                state_a_.fab(0).data());
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
    ::pops::runtime::multiblock::BoundaryEvaluationPoint point;
    point.clock = clock_identity_;
    point.tick = batch.begin_tick;
    point.level = 0;
    point.substep = static_cast<int>((batch.begin_tick - attempt_begin_tick_) >> common_rung_);
    point.stage = 0;
    point.stage_fraction = ::pops::amr::Rational(0, 1);
    point.dt = static_cast<double>(dt);
    point.physical_time = static_cast<double>(batch.begin_tick) * seconds_per_tick_;
    runtime_->level_neg_div_flux_capture_into(0, 0, point, current_state_(), residual_, flux_x_,
                                              flux_y_);
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
    return {current_state_().fab(0).const_array(),
            residual_.fab(0).const_array(),
            flux_x_.fab(0).const_array(),
            flux_y_.fab(0).const_array(),
            candidate_state_().fab(0).array(),
            attempt_flux_.data(),
            seconds_per_tick_,
            cell_count_,
            component_count_,
            valid_box_.lo[0],
            valid_box_.lo[1],
            valid_box_.nx(),
            common_rung_,
            current_tick_,
            batch_end_tick_,
            tick_denominator_};
  }

  [[nodiscard]] PreparedProviderSupport prepare_commit_attempt() noexcept {
    device_fence();
    if (!active_ || batch_active_ || current_tick_ != attempt_target_tick_)
      return PreparedProviderSupport::reject(
          0x756106u, "provider did not reach its prepared synchronization barrier");
    if (runtime_->topology_epoch() != topology_epoch_ ||
        runtime_->topology_materialization_generation() != materialization_generation_)
      return PreparedProviderSupport::reject(
          0x756107u, "provider storage changed before accepted publication");
    if (!ledger_ || ledger_->topology_epoch() != topology_epoch_ ||
        ledger_->materialization_generation() != materialization_generation_ ||
        ledger_->block() != 0 || ledger_->level() != 0 ||
        ledger_->cell_count() != cell_count_ || ledger_->component_count() != component_count_)
      return PreparedProviderSupport::reject(
          0x756108u, "provider flux ledger changed before accepted publication");
    return PreparedProviderSupport::accept();
  }

  void commit_attempt() noexcept {
    const PreparedProviderSupport support = prepare_commit_attempt();
    if (!support.well_formed() || !support.accepted())
      std::terminate();
    const ConstArray4 source = current_state_().fab(0).const_array();
    const Array4 destination = live_->fab(0).array();
    for (int j = valid_box_.lo[1]; j <= valid_box_.hi[1]; ++j)
      for (int i = valid_box_.lo[0]; i <= valid_box_.hi[0]; ++i)
        for (int component = 0; component < component_count_; ++component)
          destination(i, j, component) = source(i, j, component);
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

  /// Rebind only the accepted clock after the owning Program restored the matching native image.
  /// The executor has already proved that topology, denominator, canonical cells and rungs are the
  /// immutable prepared authority of this provider.
  void restore_accepted_boundary(
      const CellTemporalPartitionAcceptedState& accepted) noexcept {
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

  [[nodiscard]] MultiFab& current_state_() const noexcept {
    return current_is_a_ ? state_a_ : state_b_;
  }

  [[nodiscard]] MultiFab& candidate_state_() const noexcept {
    return current_is_a_ ? state_b_ : state_a_;
  }

  void validate_and_materialize_(const CellTemporalPartitionAcceptedState& partition) {
    validate_cell_temporal_partition_state(partition);
    if (partition.provider_identity != kSameLevelTransportEulerStageFluxProvider ||
        runtime_->n_blocks() != 1 || runtime_->nlev() != 1 || n_ranks() != 1)
      throw std::invalid_argument(
          "same-level transport provider requires its exact serial one-block/one-level partition");
    if (clock_identity_.empty())
      throw std::invalid_argument(
          "same-level transport provider requires a non-empty clock identity");
    live_ = &runtime_->level_state(0, 0);
    if (live_->box_array().size() != 1 || live_->local_size() != 1 || live_->dmap()[0] != 0)
      throw std::invalid_argument("same-level transport provider requires one serial-owned box");
    if (runtime_->block_state_identity(0).empty() ||
        runtime_->block_transport_flux_provider_identity(0).empty() ||
        runtime_->block_transport_flux_parameter_contract(0).empty())
      throw std::invalid_argument(
          "same-level transport provider requires an exact builder-owned state/spatial contract");
    if (runtime_->block_has_prepared_boundary_plan(0))
      throw std::invalid_argument(
          "same-level transport provider has no exact prepared-boundary contract proof");
    valid_box_ = live_->box(0);
    cell_count_ = static_cast<std::size_t>(valid_box_.num_cells());
    component_count_ = live_->ncomp();
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
    state_a_ = MultiFab(live_->box_array(), live_->dmap(), live_->ncomp(), live_->n_grow());
    state_b_ = MultiFab(live_->box_array(), live_->dmap(), live_->ncomp(), live_->n_grow());
    residual_ = MultiFab(live_->box_array(), live_->dmap(), live_->ncomp(), 0);
    flux_x_ = MultiFab(same_level_cell_temporal_detail::face_boxes(live_->box_array(), true),
                       live_->dmap(), live_->ncomp(), 0);
    flux_y_ = MultiFab(same_level_cell_temporal_detail::face_boxes(live_->box_array(), false),
                       live_->dmap(), live_->ncomp(), 0);
    attempt_flux_.assign(cell_count_ * std::size_t{4} * static_cast<std::size_t>(component_count_),
                         Real(0));
    current_is_a_ = true;

    const Geometry geometry = runtime_->level_geom(0);
    const Periodicity periodicity = runtime_->base_periodicity();
    ExactContractBuilder parameters;
    parameters.text("pops.amr.same-level-transport-euler-stage-flux")
        .scalar(std::uint32_t{1})
        .text(runtime_->block_state_identity(0))
        .text(runtime_->block_transport_flux_provider_identity(0))
        .bytes(runtime_->block_transport_flux_parameter_contract(0))
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
        .scalar(static_cast<std::int32_t>(live_->n_grow()))
        .scalar(static_cast<std::int32_t>(geometry.domain.lo[0]))
        .scalar(static_cast<std::int32_t>(geometry.domain.lo[1]))
        .scalar(static_cast<std::int32_t>(geometry.domain.hi[0]))
        .scalar(static_cast<std::int32_t>(geometry.domain.hi[1]))
        .scalar(geometry.xlo)
        .scalar(geometry.xhi)
        .scalar(geometry.ylo)
        .scalar(geometry.yhi)
        .scalar(periodicity.x)
        .scalar(periodicity.y)
        .sequence(live_->box_array().boxes(),
                  [](ExactContractBuilder& item, const Box2D& box) {
                    item.scalar(static_cast<std::int32_t>(box.lo[0]))
                        .scalar(static_cast<std::int32_t>(box.lo[1]))
                        .scalar(static_cast<std::int32_t>(box.hi[0]))
                        .scalar(static_cast<std::int32_t>(box.hi[1]));
                  })
        .sequence(live_->dmap().ranks());
    exact_parameters_ = std::move(parameters).release();
  }

  AmrRuntime* runtime_ = nullptr;
  MultiFab* live_ = nullptr;
  std::shared_ptr<SameLevelCellIntegratedFluxLedger> ledger_;
  Real seconds_per_tick_ = Real(0);
  std::string clock_identity_;
  std::uint64_t topology_epoch_ = 0;
  std::uint64_t materialization_generation_ = 0;
  std::string exact_parameters_;
  Box2D valid_box_{};
  std::size_t cell_count_ = 0;
  int component_count_ = 0;
  int common_rung_ = 0;
  std::int64_t synchronization_tick_ = 0;
  std::int64_t tick_denominator_ = 1;
  mutable MultiFab state_a_;
  mutable MultiFab state_b_;
  MultiFab residual_;
  MultiFab flux_x_;
  MultiFab flux_y_;
  mutable std::vector<Real, fab_allocator<Real>> attempt_flux_;
  bool current_is_a_ = true;
  std::int64_t attempt_begin_tick_ = 0;
  std::int64_t attempt_target_tick_ = 0;
  std::int64_t current_tick_ = 0;
  std::int64_t batch_end_tick_ = 0;
  bool active_ = false;
  bool batch_active_ = false;
};

static_assert(CellTemporalStageFluxProvider<PreparedSameLevelTransportEulerStageFluxProvider>);
static_assert(CellTemporalRungBatchLifecycle<PreparedSameLevelTransportEulerStageFluxProvider>);
static_assert(
    CellTemporalAcceptedBoundaryLifecycle<PreparedSameLevelTransportEulerStageFluxProvider>);

}  // namespace pops::runtime::program
