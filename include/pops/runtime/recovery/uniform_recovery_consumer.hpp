#pragma once

/// @file
/// @brief Generation-qualified primitive materialization for the host Uniform runtime.
///
/// This is the first production consumer of RecoveryWarmStartSlot.  One consumer instance belongs
/// to one runtime block and owns one slot per local Uniform cell.  A slot is reusable only when the
/// cell identity, exact conservative state, topology generation and accepted batch generation all
/// agree.  Candidate primitives and cache entries are staged through
/// RecoveryPublicationTransaction; a failed batch publishes no primitive array and explicitly
/// invalidates every slot touched by that consumer.
///
/// The route is deliberately host/Uniform-only.  AMR patch migration, regrid generations and
/// checkpoint/restart persistence require a hierarchy-owned cache and are not inferred here.

#include <pops/numerics/nonlinear/prepared_variable_recovery.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {

inline constexpr std::size_t kNoRecoveryCell = std::numeric_limits<std::size_t>::max();

/// Result of one all-or-nothing Uniform primitive-materialization batch.
struct UniformRecoveryBatchReport {
  RecoveryReport recovery{};
  std::size_t cell_count = 0;
  std::size_t recovered_cells = 0;
  std::size_t cache_hits = 0;
  std::size_t projection_attempts = 0;
  std::size_t projected_cells = 0;
  std::size_t failed_cell = kNoRecoveryCell;
  std::uint64_t topology_generation = 0;
  std::uint64_t state_generation = 0;
  bool published = false;

  bool publication_permitted() const { return published && failed_cell == kNoRecoveryCell; }
};

/// Type-erased host batch consumed by System::get_primitive_state.
using UniformCellRecovery = std::function<UniformRecoveryBatchReport(
    const std::vector<double>& conserved, std::vector<double>& primitive)>;

namespace recovery_detail {

inline std::uint64_t next_uniform_recovery_generation(std::uint64_t current,
                                                      const char* generation_name) {
  if (current == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error(std::string("Uniform recovery ") + generation_name +
                              " generation exhausted");
  return current + 1;
}

template <int N>
bool exact_uniform_recovery_state(const std::array<double, N>& accepted,
                                  const std::array<double, N>& candidate) {
  return std::memcmp(accepted.data(), candidate.data(), sizeof(double) * N) == 0;
}

/// Explicit legacy-plan overload retained for the separately owned generated AMR consumer.
template <int N, class Admissible, class Methods>
PreparedVariableRecoveryAttempt<N> execute_uniform_recovery(
    const PreparedVariableRecoveryPlan<N, Admissible, Methods>& plan, const Real (&conserved)[N],
    const Real (&initial_guess)[N]) {
  return {recover_prepared_variable(plan, conserved, initial_guess), false, false};
}

/// Uniform's final path consumes the shared prepared inversion authority directly.
template <int N, class Authority>
  requires(Authority::N == N)
PreparedVariableRecoveryAttempt<N> execute_uniform_recovery(std::shared_ptr<Authority>& authority,
                                                            const Real (&conserved)[N],
                                                            const Real (&)[N]) {
  return authority->recover(conserved);
}

}  // namespace recovery_detail

/// Stateful host consumer around one immutable prepared recovery plan.
///
/// Input and output use the System component-major layout: component * n_cells + cell.  The output
/// vector is assigned only after every cell has recovered and every per-cell transaction committed.
/// On any refusal or exception, the caller's output stays byte-exact and all slots are invalidated.
template <int N, class Plan>
class PreparedUniformRecoveryConsumer {
 public:
  static_assert(N > 0, "a Uniform recovery consumer needs at least one variable");

  explicit PreparedUniformRecoveryConsumer(Plan plan) : plan_(std::move(plan)) {}

  UniformRecoveryBatchReport recover(const std::vector<double>& conserved,
                                     std::vector<double>& primitive) {
    if (conserved.size() % static_cast<std::size_t>(N) != 0)
      throw std::invalid_argument(
          "Uniform recovery input size must be divisible by the prepared variable width");

    const std::size_t cells = conserved.size() / static_cast<std::size_t>(N);
    prepare_topology(cells);
    const std::uint64_t candidate_state_generation =
        recovery_detail::next_uniform_recovery_generation(state_generation_, "state");

    UniformRecoveryBatchReport batch;
    batch.cell_count = cells;
    batch.topology_generation = topology_generation_;
    batch.state_generation = state_generation_;

    std::vector<double> candidate(conserved.size());
    std::vector<std::array<double, N>> next_identity(cells);
    try {
      for (std::size_t cell = 0; cell < cells; ++cell) {
        Real cell_conserved[N] = {};
        Real initial_guess[N] = {};
        std::array<double, N> cell_identity{};
        for (int component = 0; component < N; ++component) {
          const double input = conserved[static_cast<std::size_t>(component) * cells + cell];
          const Real value = static_cast<Real>(input);
          cell_conserved[component] = initial_guess[component] = value;
          cell_identity[static_cast<std::size_t>(component)] = input;
          next_identity[cell][static_cast<std::size_t>(component)] = input;
        }

        RecoveryWarmStartSlot<N>& slot = slots_[cell];
        const bool exact_identity =
            identity_valid_[cell] != 0 && recovery_detail::exact_uniform_recovery_state<N>(
                                              accepted_identity_[cell], cell_identity);
        if (exact_identity &&
            slot.load_if_current(topology_generation_, state_generation_, initial_guess))
          ++batch.cache_hits;

        const PreparedVariableRecoveryAttempt<N> recovery =
            recovery_detail::execute_uniform_recovery(plan_, cell_conserved, initial_guess);
        const RecoveryOutcome<N>& outcome = recovery.outcome;
        if (recovery.projection_attempted)
          ++batch.projection_attempts;
        if (recovery.projection_changed)
          ++batch.projected_cells;
        batch.recovery = recovery_report(outcome);
        if (!outcome.publication_permitted()) {
          batch.failed_cell = cell;
          rollback_cache();
          return batch;
        }

        Real accepted_value[N] = {};
        RecoveryPublicationTransaction<N> transaction(accepted_value, slot);
        if (!transaction.publish_tentative(outcome, topology_generation_,
                                           candidate_state_generation) ||
            !transaction.commit())
          throw std::logic_error(
              "Uniform recovery publication transaction refused a recovered "
              "candidate");

        for (int component = 0; component < N; ++component)
          candidate[static_cast<std::size_t>(component) * cells + cell] =
              static_cast<double>(accepted_value[component]);
        ++batch.recovered_cells;
      }
    } catch (...) {
      rollback_cache();
      throw;
    }

    accepted_identity_.swap(next_identity);
    identity_valid_.assign(cells, std::uint8_t{1});
    state_generation_ = candidate_state_generation;
    batch.state_generation = state_generation_;
    batch.published = true;
    primitive = std::move(candidate);
    return batch;
  }

  void invalidate() { rollback_cache(); }

 private:
  void prepare_topology(std::size_t cells) {
    if (topology_initialized_ && slots_.size() == cells)
      return;
    topology_generation_ =
        recovery_detail::next_uniform_recovery_generation(topology_generation_, "topology");
    slots_.assign(cells, RecoveryWarmStartSlot<N>{});
    accepted_identity_.assign(cells, std::array<double, N>{});
    identity_valid_.assign(cells, std::uint8_t{0});
    topology_initialized_ = true;
  }

  void rollback_cache() {
    for (auto& slot : slots_)
      slot.invalidate();
    identity_valid_.assign(identity_valid_.size(), std::uint8_t{0});
  }

  Plan plan_;
  std::vector<RecoveryWarmStartSlot<N>> slots_;
  std::vector<std::array<double, N>> accepted_identity_;
  std::vector<std::uint8_t> identity_valid_;
  std::uint64_t topology_generation_ = 0;
  std::uint64_t state_generation_ = 0;
  bool topology_initialized_ = false;
};

/// Build Uniform's batch consumer over the exact prepared authority already captured by this
/// generated block's pointwise closure.  The shared object owns one reusable inversion workspace;
/// this function deliberately prepares neither a second inversion nor a second admissibility set.
template <class Authority>
UniformCellRecovery make_uniform_variable_inversion_consumer(std::shared_ptr<Authority> authority) {
  constexpr int N = Authority::N;
  if (!authority)
    throw std::invalid_argument("Uniform variable inversion authority must not be null");
  using Consumer = PreparedUniformRecoveryConsumer<N, std::shared_ptr<Authority>>;
  auto consumer = std::make_shared<Consumer>(std::move(authority));
  return [consumer = std::move(consumer)](const std::vector<double>& conserved,
                                          std::vector<double>& primitive) {
    return consumer->recover(conserved, primitive);
  };
}

/// Existing explicit-plan entry point retained solely for the separately owned generated AMR
/// materialization. It is intentionally not selected by generated Uniform materialization.
template <class Model>
UniformCellRecovery make_uniform_recovery_consumer(const Model& model) {
  constexpr int N = Model::n_vars;
  auto plan = prepare_model_variable_recovery(model);
  using Consumer = PreparedUniformRecoveryConsumer<N, decltype(plan)>;
  auto consumer = std::make_shared<Consumer>(std::move(plan));
  return [consumer = std::move(consumer)](const std::vector<double>& conserved,
                                          std::vector<double>& primitive) {
    return consumer->recover(conserved, primitive);
  };
}

}  // namespace pops
