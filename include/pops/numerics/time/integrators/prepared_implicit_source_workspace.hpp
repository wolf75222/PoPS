#pragma once

/// @file
/// @brief Persistent candidate/publication storage for one prepared local implicit source solve.

#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/nonlinear/local_nonlinear_collective.hpp>
#include <pops/numerics/nonlinear/newton_options.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace pops {

namespace detail {
template <int Dim, class MemorySpace>
struct PreparedImplicitSourceWorkspaceAccess;
}

/// Cold-bound storage for exactly one in-flight implicit source publication.
///
/// `bind` is the sole allocation boundary: it owns the candidate, the 13-component statistics
/// field, the Dim+1 reduction scalars, and a resident staged Newton report.  The bound field is
/// a shape/generation prototype, rather than a publication target: AMR Program solves publish to
/// a detached stage carrier with that exact contract.  A second solve is refused until its
/// preceding SolveOutcome has been accepted, rejected, or failed.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class PreparedImplicitSourceWorkspace final {
 public:
  using field_type = MultiFab<Dim, MemorySpace>;
  using execution_space = Kokkos::DefaultExecutionSpace;

  void bind(field_type& state, std::uint64_t state_generation = 0,
            NewtonReport* diagnostics = nullptr) {
    if (in_flight())
      throw std::logic_error("prepared implicit source workspace is in flight");

    candidate_.emplace(state.layout(), state.distribution(), state.local_rank(), state.ncomp(),
                       state.ghosts());
    statistics_.emplace(state.layout(), state.distribution(), state.local_rank(), 13,
                        Extent<Dim>{});
    failure_buffer_ =
        detail::LocalNonlinearFailureBuffer<Dim>("pops_prepared_implicit_source_failure");
    std::int64_t maximum_points = 1;
    for (std::size_t local = 0; local < state.local_size(); ++local)
      maximum_points = std::max(maximum_points, state.box(local).numPts());
    reduction_.prepare(execution_, maximum_points);
    Kokkos::deep_copy(failure_buffer_, std::numeric_limits<int>::max());
    if constexpr (!std::is_same_v<typename Kokkos::DefaultExecutionSpace::memory_space,
                                  Kokkos::HostSpace>) {
      for (int slot = 0; slot <= Dim; ++slot) {
        const auto minimum = Kokkos::subview(failure_buffer_, slot);
        Kokkos::deep_copy(minimum, std::numeric_limits<int>::max());
        int host_minimum = 0;
        Kokkos::deep_copy(host_minimum, minimum);
      }
    }
    execution_.fence("pops_prepared_implicit_source_bind");
    diagnostics_ = diagnostics;
    state_generation_ = state_generation;
    bind_epoch_ += 1;
    staged_diagnostics_ = diagnostics != nullptr ? *diagnostics : NewtonReport{};
    reason_code_ = 0;
    prepared_ = true;
    publication_active_ = false;
  }

  [[nodiscard]] bool prepared() const noexcept { return prepared_; }
  [[nodiscard]] bool in_flight() const noexcept {
    return reservation_.load(std::memory_order_acquire) == Reservation::kInFlight;
  }
  [[nodiscard]] std::uint64_t state_generation() const noexcept { return state_generation_; }
  [[nodiscard]] std::uint64_t bind_epoch() const noexcept { return bind_epoch_; }
  [[nodiscard]] std::uint32_t reason_code() const noexcept { return reason_code_; }
  [[nodiscard]] std::size_t allocation_count() const noexcept {
    return candidate_.has_value() + statistics_.has_value() +
           (failure_buffer_.is_allocated() ? std::size_t{1} : std::size_t{0}) +
           (reduction_.is_prepared() ? std::size_t{1} : std::size_t{0});
  }

 private:
  friend struct detail::PreparedImplicitSourceWorkspaceAccess<Dim, MemorySpace>;

  enum class Reservation : std::uint8_t { kIdle, kInFlight };

  [[nodiscard]] bool matches(field_type& state, std::uint64_t generation) const noexcept {
    return prepared_ && state_generation_ == generation && candidate_.has_value() &&
           statistics_.has_value() && same_field_contract_(state, *candidate_) &&
           same_statistics_contract_(state, *statistics_);
  }

  [[nodiscard]] bool try_reserve() noexcept {
    Reservation expected = Reservation::kIdle;
    return reservation_.compare_exchange_strong(
        expected, Reservation::kInFlight, std::memory_order_acq_rel, std::memory_order_acquire);
  }

  [[nodiscard]] bool diagnostics_capacity_valid() const noexcept {
    return diagnostics_ == nullptr ||
           (diagnostics_->solve.reason.size() <= staged_diagnostics_.solve.reason.capacity() &&
            diagnostics_->diagnostics.source.size() <=
                staged_diagnostics_.diagnostics.source.capacity() &&
            diagnostics_->diagnostics.events.size() <=
                staged_diagnostics_.diagnostics.events.capacity());
  }

  bool stage_report(const SolveReport& solve, double failed_cells) {
    if (!diagnostics_capacity_valid())
      return false;
    if (diagnostics_ != nullptr) {
      // The exact same report resident storage was provisioned from this diagnostic object at bind.
      // Assignment therefore reuses its cold-established dynamic capacity on every retry.
      staged_diagnostics_ = *diagnostics_;
    } else {
      staged_diagnostics_.enabled = false;
      staged_diagnostics_.converged = true;
      staged_diagnostics_.max_residual = Real(0);
      staged_diagnostics_.max_iters_used = Real(0);
      staged_diagnostics_.n_failed = 0;
      staged_diagnostics_.failure = {};
    }
    staged_diagnostics_.enabled = true;
    staged_diagnostics_.solve = solve;
    staged_diagnostics_.max_residual =
        std::max(staged_diagnostics_.max_residual, solve.residual_norm);
    staged_diagnostics_.max_iters_used =
        std::max(staged_diagnostics_.max_iters_used, static_cast<Real>(solve.iters));
    staged_diagnostics_.n_failed += failed_cells;
    if (!solve.solved()) {
      staged_diagnostics_.converged = false;
      staged_diagnostics_.failure = solve.failure;
    }
    return true;
  }

  void arm_publication(field_type& destination, std::uint64_t generation,
                       std::uint32_t reason_code) noexcept {
    publication_destination_ = &destination;
    publication_generation_ = generation;
    reason_code_ = reason_code;
    publication_active_ = true;
  }

  void validate_publication() const {
    if (!publication_active_ || publication_destination_ == nullptr ||
        publication_generation_ != state_generation_ ||
        !matches(*publication_destination_, publication_generation_))
      throw std::logic_error("prepared implicit source publication is stale");
  }

  void publish() noexcept {
    copy_(*publication_destination_, *candidate_);
    if (diagnostics_ != nullptr)
      *diagnostics_ = staged_diagnostics_;
  }

  void release() noexcept {
    publication_destination_ = nullptr;
    publication_active_ = false;
    reservation_.store(Reservation::kIdle, std::memory_order_release);
  }

  static void copy_(field_type& destination, const field_type& source) noexcept {
    if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>) {
      for (std::size_t local = 0; local < destination.local_size(); ++local) {
        const Box<Dim>& box = destination.box(local);
        const Extent<Dim> extent = box.extent();
        const auto output = destination.fab(local).view();
        const auto input = source.fab(local).view();
        for (int component = 0; component < destination.ncomp(); ++component)
          for (std::int64_t ordinal = 0; ordinal < box.numPts(); ++ordinal) {
            std::int64_t remainder = ordinal;
            Index<Dim> index{};
            for (int axis = 0; axis < Dim; ++axis) {
              index[axis] = box.lo[axis] + static_cast<int>(remainder % extent[axis]);
              remainder /= extent[axis];
            }
            output(index, component) = input(index, component);
          }
      }
    } else {
      lincomb(destination, Real(1), source, Real(0), source);
    }
  }

  static bool same_field_contract_(const field_type& left, const field_type& right) noexcept {
    return left.layout() == right.layout() && left.distribution() == right.distribution() &&
           left.local_rank() == right.local_rank() && left.ncomp() == right.ncomp() &&
           left.ghosts() == right.ghosts() && left.local_size() == right.local_size();
  }

  static bool same_statistics_contract_(const field_type& state,
                                        const field_type& statistics) noexcept {
    return state.layout() == statistics.layout() &&
           state.distribution() == statistics.distribution() &&
           state.local_rank() == statistics.local_rank() && statistics.ncomp() == 13 &&
           statistics.ghosts() == Extent<Dim>{} && state.local_size() == statistics.local_size();
  }

  std::optional<field_type> candidate_{};
  std::optional<field_type> statistics_{};
  detail::LocalNonlinearFailureBuffer<Dim> failure_buffer_{};
  PreparedCellSumReduction<execution_space> reduction_{};
  execution_space execution_{};
  field_type* publication_destination_ = nullptr;
  NewtonReport* diagnostics_ = nullptr;
  NewtonReport staged_diagnostics_{};
  std::uint64_t state_generation_ = 0;
  std::uint64_t publication_generation_ = 0;
  std::uint64_t bind_epoch_ = 0;
  std::uint32_t reason_code_ = 0;
  bool prepared_ = false;
  bool publication_active_ = false;
  std::atomic<Reservation> reservation_{Reservation::kIdle};
};

namespace detail {

template <int Dim, class MemorySpace>
struct PreparedImplicitSourceWorkspaceAccess final {
  using workspace_type = PreparedImplicitSourceWorkspace<Dim, MemorySpace>;
  using field_type = typename workspace_type::field_type;

  static bool matches(const workspace_type& workspace, field_type& state,
                      std::uint64_t generation) noexcept {
    return workspace.matches(state, generation);
  }
  static bool try_reserve(workspace_type& workspace) noexcept { return workspace.try_reserve(); }
  static field_type& candidate(workspace_type& workspace) { return *workspace.candidate_; }
  static field_type& statistics(workspace_type& workspace) { return *workspace.statistics_; }
  static LocalNonlinearFailureBuffer<Dim>& failure_buffer(workspace_type& workspace) {
    return workspace.failure_buffer_;
  }
  static const typename workspace_type::execution_space& execution(
      const workspace_type& workspace) {
    return workspace.execution_;
  }
  static const PreparedCellSumReduction<typename workspace_type::execution_space>& reduction(
      const workspace_type& workspace) {
    return workspace.reduction_;
  }
  static bool diagnostics_capacity_valid(const workspace_type& workspace) noexcept {
    return workspace.diagnostics_capacity_valid();
  }
  static void copy_from_state(workspace_type& workspace, field_type& state) noexcept {
    workspace.copy_(*workspace.candidate_, state);
  }
  static NewtonReport* diagnostics(const workspace_type& workspace) noexcept {
    return workspace.diagnostics_;
  }
  static bool stage_report(workspace_type& workspace, const SolveReport& solve,
                           double failed_cells) {
    return workspace.stage_report(solve, failed_cells);
  }
  static void arm_publication(workspace_type& workspace, field_type& destination,
                              std::uint64_t generation, std::uint32_t reason_code) noexcept {
    workspace.arm_publication(destination, generation, reason_code);
  }
  static void validate_publication(const workspace_type& workspace) {
    workspace.validate_publication();
  }
  static void publish(workspace_type& workspace) noexcept { workspace.publish(); }
  static void release(workspace_type& workspace) noexcept { workspace.release(); }
};

}  // namespace detail
}  // namespace pops
