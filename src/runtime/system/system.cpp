// Core exact-ranked System facade. Provider-specific installation and execution seams live in
// sibling translation units; this file owns only layout-independent lifecycle, clock and cadence.
#include "system_impl.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/program/profiler.hpp>
#include <pops/runtime/program/step_transaction.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pops {

POPS_EXPORT std::string abi_key() {
  return detail::abi_key_string();
}

template <int Dim>
std::string System<Dim>::abi_key() {
  return pops::abi_key();
}

template <int Dim>
System<Dim>::System(const SystemConfig<Dim>& config) {
  validate_system_config(config);
  p_ = std::make_unique<Impl>(config);
}

// User-provided: GCC rejects an out-of-line `= default` when the same special members are
// also explicitly instantiated for kNativeDimension.
template <int Dim>
System<Dim>::~System() {}

template <int Dim>
System<Dim>::System(System&& other) noexcept
    : prepared_boundary_execution_lane_(std::move(other.prepared_boundary_execution_lane_)),
      p_(std::move(other.p_)) {}

template <int Dim>
System<Dim>& System<Dim>::operator=(System&& other) noexcept {
  if (this != &other) {
    // Destroy Impl first: installed field solvers and boundary transports may hold
    // ImmutableBorrow pins on the destination lane. Releasing the lane first terminates.
    p_ = std::move(other.p_);
    prepared_boundary_execution_lane_ = std::move(other.prepared_boundary_execution_lane_);
  }
  return *this;
}

template <int Dim>
void System<Dim>::step(double dt) {
  p_->program_.require_step_installed("System::step");
  p_->execute_step_transaction([&] {
    // Profiling is candidate state as well: enter it only after the visibility writer has
    // blocked readers, so a rejected attempt cannot leak a counter or an open scope.
    runtime::program::ProfileScope scope(p_->program_.profiler_, "step");
    p_->program_.profiler_.count("steps");
    p_->program_.dispatch_cadence_step(p_->t, p_->macro_step_, dt, "System");
  });
}

template <int Dim>
void System<Dim>::advance(double dt, int nsteps) {
  p_->program_.require_step_installed("System::advance");
  if (nsteps < 0)
    throw std::invalid_argument("System::advance requires a non-negative step count");
  for (int step_index = 0; step_index < nsteps; ++step_index)
    step(dt);
}

template <int Dim>
void System<Dim>::begin_step_transaction() {
  if (p_->external_program_transaction_)
    throw std::runtime_error("System::begin_step_transaction: transaction already active");
  try {
    // Registry::begin captures the bind-primed carrier, then begin_candidate acquires and retains
    // the visibility writer for the complete outer envelope. This is the one authoritative
    // snapshot used by candidate rollback and by step-change diagnostics.
    p_->external_program_transaction_.emplace(p_->step_transaction_registry_.begin());
    if (!p_->external_program_transaction_->begin_candidate())
      throw std::runtime_error("System transaction candidate phase rejected collectively");
    p_->external_step_transaction_committed_ = false;
  } catch (...) {
    if (p_->external_program_transaction_)
      p_->external_program_transaction_->rollback();
    p_->external_program_transaction_.reset();
    throw;
  }
}

template <int Dim>
void System<Dim>::commit_step_transaction() {
  if (!p_->external_program_transaction_)
    throw std::runtime_error("System::commit_step_transaction: no active transaction");
  if (p_->external_step_transaction_committed_)
    throw std::runtime_error("System::commit_step_transaction: transaction already committed");
  if (p_->external_program_transaction_->phase() !=
      runtime::program::ProgramTransactionPhase::kCandidate)
    throw std::runtime_error(
        "System::commit_step_transaction requires one completed candidate step");
  try {
    if (!p_->external_program_transaction_->begin_solve_guard_effect_prepare())
      throw std::runtime_error(
          "System transaction solve/guard/effect preparation rejected collectively");
    if (!p_->external_program_transaction_->hidden_publish())
      throw std::runtime_error("System transaction hidden publication failed collectively");
    p_->external_step_transaction_committed_ = true;
  } catch (...) {
    p_->external_program_transaction_->rollback();
    p_->external_program_transaction_.reset();
    p_->external_step_transaction_committed_ = false;
    throw;
  }
}

template <int Dim>
double System<Dim>::step_change_l2_for_block(std::string_view name) const {
  p_->step_transaction_carrier_.step_change_last_dispatches = 0;
  // The external transaction deliberately retains the visibility writer from begin through
  // finalize/rollback.  This diagnostic is part of that writer-owned window: taking a shared
  // lease here would self-deadlock, while releasing the writer would expose the provisional
  // candidate to other readers.  The active external snapshot is the authority proving that the
  // caller is inside the protected window, so an explicit provisional scope must authenticate the
  // resident candidate.  `acquire_accepted_read_lease()` recognizes that scope; without it the
  // writer thread is rejected and a foreign reader blocks on the visibility writer.
  // This is the internal Program-writer accessor: the public lane observer takes an accepted
  // read lease and is therefore intentionally unavailable while the candidate writer is held.
  const ExecutionLane& lane = program_prepared_boundary_execution_lane_();
  [[maybe_unused]] auto provisional_read = p_->acquire_accepted_read_lease();
  const MultiFab<Dim>* current = nullptr;
  const MultiFab<Dim>* prior = nullptr;
  double cell_measure = 1.0;
  std::size_t required_terms = 0;
  bool local_preflight_invalid = false;
  try {
    if (!p_->external_program_transaction_ || !p_->step_transaction_carrier_.accepted ||
        !p_->step_transaction_carrier_.snapshot_active)
      local_preflight_invalid = true;
    else {
      const std::vector<MultiFab<Dim>>& previous = p_->step_transaction_carrier_.accepted->states;
      if (previous.size() != p_->sp.size()) {
        local_preflight_invalid = true;
      } else {
        std::size_t matched = 0;
        for (std::size_t block = 0; block < p_->sp.size(); ++block) {
          if (p_->sp[block].name == name) {
            ++matched;
            current = &p_->sp[block].U;
            prior = &previous[block];
          }
        }
        if (matched != 1 || current == nullptr || prior == nullptr) {
          local_preflight_invalid = true;
        } else {
          for (int axis = 0; axis < Dim; ++axis) {
            const double spacing = static_cast<double>(p_->geom.spacing(axis));
            if (!std::isfinite(spacing) || !(spacing > 0.0))
              local_preflight_invalid = true;
            cell_measure *= spacing;
          }
          if (!std::isfinite(cell_measure) || !(cell_measure > 0.0) ||
              current->layout() != prior->layout() ||
              current->distribution() != prior->distribution() ||
              current->local_rank() != prior->local_rank() || current->ncomp() != prior->ncomp() ||
              current->ghosts() != prior->ghosts() ||
              current->local_size() != prior->local_size() || !lane.active() ||
              current->rank_space().size() != static_cast<std::size_t>(lane.size()) ||
              current->rank_space().linear_rank(current->local_rank()) !=
                  static_cast<std::size_t>(lane.rank()) ||
              (current->distribution().replicated() && lane.size() != 1)) {
            local_preflight_invalid = true;
          }
          for (std::size_t local = 0; !local_preflight_invalid && local < current->local_size();
               ++local) {
            const std::int64_t signed_cells = current->box(local).numPts();
            if (signed_cells <= 0 || current->global_index(local) != prior->global_index(local) ||
                current->fab(local).size() != prior->fab(local).size()) {
              local_preflight_invalid = true;
              break;
            }
            const std::size_t cells = static_cast<std::size_t>(signed_cells);
            if (cells > std::numeric_limits<std::size_t>::max() /
                            static_cast<std::size_t>(current->ncomp()) ||
                required_terms > std::numeric_limits<std::size_t>::max() -
                                     cells * static_cast<std::size_t>(current->ncomp())) {
              local_preflight_invalid = true;
              break;
            }
            required_terms += cells * static_cast<std::size_t>(current->ncomp());
          }
        }
      }
    }
  } catch (...) {
    local_preflight_invalid = true;
  }
  // Every validation above is deliberately converted to a lane-wide refusal before any rank
  // launches a reduction kernel or throws.  A rank-local malformed geometry/layout therefore
  // cannot strand its peers in a later collective.
  if (all_reduce_max(local_preflight_invalid ? 1L : 0L, lane) != 0)
    throw std::runtime_error("System::step_change_l2_for_block preflight failed collectively");
  const MultiFab<Dim>& current_field = *current;
  const MultiFab<Dim>& prior_field = *prior;
  {
    auto& terms = p_->step_transaction_carrier_.step_change_terms;
    auto& invalid = p_->step_transaction_carrier_.step_change_invalid;
    const std::size_t capacity = p_->step_transaction_carrier_.step_change_term_capacity;
    const bool local_workspace_invalid =
        terms.data() == nullptr || invalid.data() == nullptr || capacity < required_terms;
    if (all_reduce_max(local_workspace_invalid ? 1L : 0L, lane) != 0)
      throw std::runtime_error(
          "System::step_change_l2_for_block workspace preflight failed collectively");
    // This workspace is exclusively owned by the candidate reader and each prior invocation
    // fences its asynchronous work before returning. Every term below is overwritten exactly
    // once, so the term buffer itself needs no hot reset or pre-launch fence.
    invalid() = 0;
    std::size_t term_offset = 0;
    bool launched_asynchronous_work = false;
    for (std::size_t local = 0; local < current_field.local_size(); ++local) {
      const auto current_view = current_field.fab(local).view();
      const auto previous_view = prior_field.fab(local).view();
      const Box<Dim> box = current_field.box(local);
      const std::int64_t signed_cells = box.numPts();
      const std::size_t cells = static_cast<std::size_t>(signed_cells);
      if constexpr (!std::is_same_v<Kokkos::DefaultExecutionSpace,
                                    Kokkos::DefaultHostExecutionSpace>) {
        launched_asynchronous_work = true;
      } else if (signed_cells >= detail::foreach_serial_threshold()) {
        launched_asynchronous_work = true;
      }
      for (int component = 0; component < current_field.ncomp(); ++component) {
        const std::size_t component_offset = term_offset;
        term_offset += cells;
        for_each_cell(
            box, KOKKOS_LAMBDA(const Index<Dim>& index) {
              std::size_t linear = 0;
              std::size_t stride = 1;
              for (int axis = 0; axis < Dim; ++axis) {
                linear += static_cast<std::size_t>(index[axis] - box.lo[axis]) * stride;
                stride *= static_cast<std::size_t>(box.length(axis));
              }
              const Real difference =
                  current_view(index, component) - previous_view(index, component);
              const Real squared = difference * difference;
              if (!Kokkos::isfinite(difference) || !Kokkos::isfinite(squared)) {
                terms(component_offset + linear) = Real(0);
                Kokkos::atomic_exchange(&invalid(), 1);
              } else {
                terms(component_offset + linear) = squared;
              }
            });
        ++p_->step_transaction_carrier_.step_change_last_dispatches;
      }
    }
    if (term_offset != required_terms)
      throw std::logic_error(
          "System::step_change_l2_for_block reduction workspace accounting drift");
    // SharedSpace host reads follow a fence whenever `for_each_cell` launched a Kokkos kernel.
    // The small-host-box path is an inline synchronous loop and requires no Kokkos fence (whose
    // OpenMP bookkeeping allocates on this otherwise allocation-free diagnostic). Sum the
    // canonical slot order on the host, then reject non-finite terms collectively before
    // reducing a scientific scalar over the prepared lane.
    if (launched_asynchronous_work)
      ::pops::device_fence();
    Real local_sum = Real(0);
    for (std::size_t slot = 0; slot < term_offset; ++slot)
      local_sum += terms(slot);
    const bool local_invalid = invalid() != 0 || !std::isfinite(local_sum);
    if (all_reduce_sum(local_invalid ? 1L : 0L, lane) != 0)
      throw std::runtime_error("System::step_change_l2_for_block encountered a non-finite state");
    const double global_sum = static_cast<double>(all_reduce_sum(local_sum, lane));
    const double result = std::sqrt(cell_measure * global_sum);
    if (all_reduce_sum(std::isfinite(global_sum) && std::isfinite(result) ? 0L : 1L, lane) != 0)
      throw std::runtime_error("System::step_change_l2_for_block produced a non-finite result");
    return result;
  }
}

template <int Dim>
std::uint64_t System<Dim>::_step_change_l2_last_dispatches() const noexcept {
  return p_->step_transaction_carrier_.step_change_last_dispatches;
}

template <int Dim>
std::map<std::string, double> System<Dim>::step_change_l2() const {
  std::map<std::string, double> result;
  for (const typename Impl::Species& block : p_->sp)
    result.emplace(block.name, step_change_l2_for_block(block.name));
  return result;
}

template <int Dim>
void System<Dim>::finalize_step_transaction() {
  if (!p_->external_program_transaction_ || !p_->external_step_transaction_committed_)
    throw std::runtime_error("System::finalize_step_transaction: no committed transaction");
  runtime::program::ProgramFinalizeReceipt finalized;
  try {
    const auto sealed = p_->external_program_transaction_->atomic_seal();
    if (!sealed)
      throw std::runtime_error("System transaction atomic seal failed collectively");
    finalized = p_->external_program_transaction_->irreversible_finalize();
  } catch (...) {
    // A failed seal rolls back inside the transaction. Finalizer failures are represented by the
    // fail-stop receipt and do not roll back the accepted scientific generation.
    p_->external_program_transaction_.reset();
    p_->external_step_transaction_committed_ = false;
    throw;
  }
  p_->step_transaction_carrier_.discard_snapshot();
  p_->external_program_transaction_.reset();
  p_->external_step_transaction_committed_ = false;
  if (!finalized)
    throw std::runtime_error("System transaction entered fail-stop during finalization");
}

template <int Dim>
void System<Dim>::rollback_step_transaction() {
  if (!p_->external_program_transaction_)
    throw std::runtime_error("System::rollback_step_transaction: no active transaction");
  p_->external_program_transaction_->rollback();
  p_->external_program_transaction_.reset();
  p_->external_step_transaction_committed_ = false;
}

template <int Dim>
void System<Dim>::begin_restart_transaction() {
  begin_step_transaction();
}

template <int Dim>
void System<Dim>::commit_restart_transaction() {
  commit_step_transaction();
}

template <int Dim>
void System<Dim>::finalize_restart_transaction() noexcept {
  if (p_->external_program_transaction_) {
    const auto sealed = p_->external_program_transaction_->atomic_seal();
    if (!sealed) {
      p_->external_program_transaction_->rollback();
      std::terminate();
    }
    const auto finalized = p_->external_program_transaction_->irreversible_finalize();
    p_->step_transaction_carrier_.discard_snapshot();
    p_->external_program_transaction_.reset();
    p_->external_step_transaction_committed_ = false;
    if (!finalized)
      std::terminate();
  } else {
    p_->external_step_transaction_committed_ = false;
  }
}

template <int Dim>
void System<Dim>::rollback_restart_transaction() {
  rollback_step_transaction();
}

template <int Dim>
std::uint64_t System<Dim>::accepted_transaction_generation_() const noexcept {
  return static_cast<std::uint64_t>(p_->step_transaction_registry_.accepted_generation());
}

template <int Dim>
bool System<Dim>::accepted_transaction_fail_stop_() const noexcept {
  return p_->step_transaction_registry_.fail_stop();
}

template <int Dim>
runtime::program::ProvisionalReadLease System<Dim>::_provisional_read_scope() const {
  return p_->step_transaction_registry_.acquire_provisional_read();
}

template <int Dim>
double System<Dim>::step_cfl(double cfl, double speed_floor, double max_dt, double min_dt) {
  const ExecutionLane& lane = prepared_boundary_execution_lane();
  std::string request_contract;
  std::exception_ptr request_error;
  try {
    p_->program_.require_step_installed("System::step_cfl");
    if (!std::isfinite(cfl) || !(cfl > 0.0))
      throw std::invalid_argument("System::step_cfl cfl must be finite and positive");
    if (!std::isfinite(speed_floor) || !(speed_floor > 0.0))
      throw std::invalid_argument("System::step_cfl speed_floor must be finite and positive");
    if (std::isnan(max_dt) || max_dt <= 0.0)
      throw std::invalid_argument("System::step_cfl max_dt must be positive or +infinity");
    if (!std::isfinite(min_dt) || min_dt < 0.0)
      throw std::invalid_argument("System::step_cfl min_dt must be finite and non-negative");
    ExactContractBuilder contract;
    contract.text("pops.system.step-cfl-request")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .scalar(cfl)
        .scalar(speed_floor)
        .scalar(max_dt)
        .scalar(min_dt)
        .scalar(static_cast<std::uint64_t>(p_->sp.size()));
    for (const typename Impl::Species& block : p_->sp) {
      contract.text(block.name)
          .scalar(block.evolve)
          .scalar(block.substeps)
          .scalar(block.stride)
          .presence(static_cast<bool>(block.source_frequency))
          .presence(block.parabolic_frequency.has_value());
      if (block.parabolic_frequency) {
        const Real parabolic = *block.parabolic_frequency;
        if (!std::isfinite(parabolic) || parabolic < Real(0))
          throw std::runtime_error("System generated parabolic frequency is invalid");
        contract.scalar(parabolic);
      }
      contract.presence(static_cast<bool>(block.stability_dt));
    }
    contract.scalar(static_cast<std::uint64_t>(p_->coupling_.coupled_freqs.size()));
    for (const runtime::system::CoupledFreq& frequency : p_->coupling_.coupled_freqs)
      contract.text(frequency.label).scalar(frequency.mu);
    contract.scalar(static_cast<std::uint64_t>(p_->coupling_.coupled_frequencies.size()));
    for (const runtime::system::PreparedCoupledFrequency& frequency :
         p_->coupling_.coupled_frequencies)
      contract.text(frequency.label).presence(static_cast<bool>(frequency.maximum_frequency));
    contract.scalar(static_cast<std::uint64_t>(p_->coupling_.dt_bounds.size()));
    for (const runtime::system::GlobalDtBound& bound : p_->coupling_.dt_bounds)
      contract.text(bound.label).presence(static_cast<bool>(bound.fn));
    contract.presence(static_cast<bool>(p_->program_.dt_bound_));
    request_contract = std::move(contract).release();
  } catch (...) {
    request_error = std::current_exception();
  }
  if (all_reduce_max(request_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && request_error)
      std::rethrow_exception(request_error);
    throw std::runtime_error("System::step_cfl request validation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("system-step-cfl-request"), std::string_view(request_contract)}},
          lane))
    throw std::invalid_argument(
        "System::step_cfl inputs or prepared scalar authorities differ between MPI ranks");

  // The field solve participates in the same accepted-state attempt as the CFL decision and
  // Program dispatch.  In particular, an unavailable value is consumed before any candidate
  // can be made visible, and a reject/failure restores every field, clock and auxiliary image
  // captured by execute_step_transaction().
  return p_->execute_step_transaction([&]() -> double {
    SolveOutcome field_outcome = program_solve_fields_();
    const SolveConsumption field_consumption =
        field_outcome.report().solved_value_available()
            ? SolveConsumption::kAccept
            : (field_outcome.report().action == SolveAction::kRejectAttempt
                   ? SolveConsumption::kRejectAttempt
                   : SolveConsumption::kFailRun);
    const SolveReport field_report = field_outcome.consume(field_consumption);
    if (!field_report.solved_value_available()) {
      if (field_consumption == SolveConsumption::kRejectAttempt)
        throw runtime::program::StepAttemptRejected(field_report.status, "CFL field evaluation",
                                                    field_report.reason);
      throw std::runtime_error(std::string("System::step_cfl field evaluation failed: status=") +
                               field_report.status_name() + " action=" +
                               field_report.action_name() + " reason=" + field_report.reason);
    }

    Real minimum_spacing = p_->geom.spacing(0);
    for (int axis = 1; axis < Dim; ++axis)
      minimum_spacing = std::min(minimum_spacing, p_->geom.spacing(axis));

    double selected = std::numeric_limits<double>::infinity();
    std::string reason = "degenerate";
    for (std::size_t block_index = 0; block_index < p_->sp.size(); ++block_index) {
      typename Impl::Species& block = p_->sp[block_index];
      if (!block.evolve)
        continue;
      const Real speed =
          std::max(block_max_speed_prepared_(static_cast<int>(block_index), block.U, lane),
                   static_cast<Real>(speed_floor));
      double block_dt = cfl * static_cast<double>(minimum_spacing) * block.substeps /
                        (static_cast<double>(block.stride) * static_cast<double>(speed));
      const char* block_reason = "transport";
      if (block.parabolic_frequency) {
        const Real parabolic = *block.parabolic_frequency;
        if (parabolic > Real(0)) {
          // Explicit advection--diffusion uses CFL / (speed / h + 2 nu sum_a h_a^-2), rather
          // than min(CFL h / speed, 1 / q): the two spectral radii act in the same stage.
          block_dt = cfl * block.substeps /
                     (static_cast<double>(block.stride) *
                      (static_cast<double>(speed) / static_cast<double>(minimum_spacing) +
                       static_cast<double>(parabolic)));
          block_reason = "parabolic_frequency";
        }
      }
      if (block.source_frequency) {
        const Real frequency = block.source_frequency(block.U);
        if (frequency > Real(0)) {
          const double source_dt =
              cfl * block.substeps /
              (static_cast<double>(block.stride) * static_cast<double>(frequency));
          if (source_dt < block_dt) {
            block_dt = source_dt;
            block_reason = "source_frequency";
          }
        }
      }
      if (block.stability_dt) {
        const Real admissible = block.stability_dt(block.U);
        if (admissible > Real(0)) {
          const double admissible_dt =
              static_cast<double>(admissible) * block.substeps / static_cast<double>(block.stride);
          if (admissible_dt < block_dt) {
            block_dt = admissible_dt;
            block_reason = "stability_dt";
          }
        }
      }
      if (block_dt < selected) {
        selected = block_dt;
        reason = std::string(block_reason) + ":" + block.name;
      }
    }

    for (const runtime::system::CoupledFreq& frequency : p_->coupling_.coupled_freqs) {
      if (!(frequency.mu > 0.0))
        continue;
      const double candidate = cfl / frequency.mu;
      if (candidate < selected) {
        selected = candidate;
        reason = "coupled_source:" + frequency.label;
      }
    }
    for (const runtime::system::PreparedCoupledFrequency& frequency :
         p_->coupling_.coupled_frequencies) {
      if (!frequency.maximum_frequency)
        continue;
      const double maximum_frequency = static_cast<double>(frequency.maximum_frequency());
      if (!std::isfinite(maximum_frequency))
        throw std::runtime_error(
            "System coupled-source frequency provider returned a non-finite "
            "maximum for '" +
            frequency.label + "'");
      if (!(maximum_frequency > 0.0))
        continue;
      const double candidate = cfl / maximum_frequency;
      if (candidate < selected) {
        selected = candidate;
        reason = "coupled_source:" + frequency.label;
      }
    }
    for (const runtime::system::GlobalDtBound& bound : p_->coupling_.dt_bounds) {
      if (!bound.fn)
        continue;
      double candidate = bound.fn();
      if (!(candidate > 0.0) || !std::isfinite(candidate))
        candidate = std::numeric_limits<double>::infinity();
      candidate = all_reduce_min(candidate, lane);
      if (candidate < selected) {
        selected = candidate;
        reason = "global:" + bound.label;
      }
    }

    if (p_->program_.dt_bound_) {
      const double program_dt = static_cast<double>(p_->program_.dt_bound_(static_cast<Real>(cfl)));
      if (std::isfinite(program_dt) && program_dt > 0.0 && program_dt < selected) {
        selected = program_dt;
        reason = "program:dt_bound";
      }
    }
    if (!std::isfinite(selected))
      selected = cfl * static_cast<double>(minimum_spacing) / speed_floor;
    if (max_dt < selected) {
      selected = max_dt;
      reason = "strategy:max_dt";
    }
    if (all_reduce_max(selected < min_dt ? 1L : 0L, lane) != 0)
      throw std::runtime_error("System::step_cfl stability bound is below declared min_dt");

    std::string decision_contract;
    std::exception_ptr decision_error;
    try {
      ExactContractBuilder contract;
      contract.text("pops.system.step-cfl-decision")
          .scalar(std::uint32_t{1})
          .scalar(selected)
          .text(reason);
      decision_contract = std::move(contract).release();
    } catch (...) {
      decision_error = std::current_exception();
    }
    if (all_reduce_max(decision_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && decision_error)
        std::rethrow_exception(decision_error);
      throw std::runtime_error("System::step_cfl decision preparation failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("system-step-cfl-decision"), std::string_view(decision_contract)}},
            lane))
      throw std::runtime_error("System::step_cfl selected different bounds across MPI ranks");

    p_->last_dt_reason_ = std::move(reason);
    p_->program_.dispatch_cadence_step(p_->t, p_->macro_step_, selected, "System");
    return selected;
  });
}

template <int Dim>
int System<Dim>::macro_step() const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  return p_->macro_step_;
}

template <int Dim>
void System<Dim>::mark_bound() {
  const ExecutionLane& lane = prepared_boundary_execution_lane();
  if (p_->program_.artifact_publication_receipt()) {
    // Program installation recorded the cold coupling footprint while the composition was still
    // assembling.  Re-derive its complete witness before sealing providers, freezing lifecycle,
    // or binding the inline arena: a mutation between install_program() and bind must fail closed.
    if (!p_->prepared_coupling_receipt_)
      throw std::logic_error("System::mark_bound: Program coupling footprint receipt is absent");
    const auto current_coupling_receipt =
        p_->prepare_coupling_receipt(lane, p_->program_.block_map());
    if (current_coupling_receipt.contract != p_->prepared_coupling_receipt_->contract ||
        current_coupling_receipt.resident_bytes != p_->prepared_coupling_receipt_->resident_bytes)
      throw std::logic_error(
          "System::mark_bound: Program coupling footprint changed after installation");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"system-prepared-coupling-receipt", current_coupling_receipt.contract}}, lane))
      throw std::runtime_error(
          "System::mark_bound: Program coupling footprint differs between MPI ranks");
  }

  // The provider graph is the only authority for the compact auxiliary carrier.  Seal it before
  // freezing composition so every rank either agrees on one graph or remains fully mutable after a
  // failed collective preflight.
  seal_auxiliary_providers();

  if (p_->lifecycle_.frozen())
    p_->lifecycle_.to_bound();

  // Field-plan setters are deliberately local: one rank may author an extra plan and must not
  // strand peers inside the setter.  Freeze is the single collective commit point for the complete
  // canonical registry and its selected exact-ranked backend authorities.
  p_->require_field_plan_consensus();

  const auto& state_routes = p_->boundary_registry_.state_routes();
  if (!state_routes.empty() && state_routes.size() != p_->sp.size())
    throw std::runtime_error(
        "System::mark_bound: block state routes do not exactly cover materialized blocks");
  for (typename Impl::Species& block : p_->sp) {
    const auto route = state_routes.find(block.name);
    if (!state_routes.empty() && route == state_routes.end())
      throw std::runtime_error(
          "System::mark_bound: materialized block lacks its exact state route");
    if (route != state_routes.end())
      block.state_identity = route->second;
  }

  for (const auto& [name, installed] : p_->boundary_registry_.boundaries()) {
    typename Impl::Species& block = p_->find(name);
    if (installed.authority->ncomp() != block.ncomp)
      throw std::runtime_error("System::mark_bound: boundary component count differs from block '" +
                               name + "'");
    if (installed.state_identity != block.state_identity)
      throw std::runtime_error("System::mark_bound: boundary state identity differs from block '" +
                               name + "'");
    if (installed.authority->periodic_axes() != p_->periodicity)
      throw std::runtime_error(
          "System::mark_bound: boundary periodicity differs from the domain for block '" + name +
          "'");
    for (int axis = 0; axis < Dim; ++axis)
      if (block.U.ghosts()[axis] < installed.required_depth)
        throw std::runtime_error("System::mark_bound: boundary depth exceeds block storage for '" +
                                 name + "'");
    p_->publish_boundary_to_block(name);
  }
  // Freeze the complete Uniform composition before entering the first transaction. The carrier
  // captures all resident fields, provider storage, and Program state once here; subsequent steps
  // recycle this image instead of constructing MultiFabs/maps from the hot path.
  // A transaction can also carry a prepared consumer without a temporal Program. Seal that empty
  // Program authority at the same cold boundary so its accepted image remains copyable.
  std::vector<const MultiFab<Dim>*> coupling_prototypes;
  coupling_prototypes.reserve(p_->sp.size());
  for (const typename Impl::Species& block : p_->sp)
    coupling_prototypes.push_back(&block.U);
  // Coupling has one resident candidate arena, bound only after all structural provider routes
  // have frozen.  Program steps subsequently patch pointer slots and cannot allocate a rollback,
  // conservation image, host mirror, or exact invocation witness.
  p_->coupling_.bind_workspace(p_->coupling_workspace_, coupling_prototypes);
  // Sparse generated RHS groups expand to one slot per System block.  This exact count is part of
  // the bound composition: hot candidate execution must never materialize vectors for it.
  p_->rhs_group_workspace_.bind(p_->sp.size());
  // A Program field solve participates in the accepted transaction.  Its default-field owner
  // therefore has to exist before the carrier captures the accepted image: constructing it from a
  // candidate would make rollback change ownership and violate the sealed transaction contract.
  const long default_field_required_local =
      std::any_of(p_->sp.begin(), p_->sp.end(),
                  [](const typename Impl::Species& block) {
                    return static_cast<bool>(block.add_poisson_rhs);
                  })
          ? 1L
          : 0L;
  const long default_field_required_min = all_reduce_min(default_field_required_local, lane);
  const long default_field_required_max = all_reduce_max(default_field_required_local, lane);
  if (default_field_required_min != default_field_required_max)
    throw std::runtime_error(
        "System::mark_bound: default field requirement differs between communicator ranks");
  if (default_field_required_max != 0)
    prepare_default_field_publication_storage_();
  p_->program_.bind_transaction_authorities();
  p_->prime_step_transaction_image(lane);
  p_->lifecycle_.to_bound();
}

template <int Dim>
int System<Dim>::program_macro_step_() const {
  return p_->macro_step_;
}

template <int Dim>
double System<Dim>::program_time_() const {
  return p_->t;
}

template <int Dim>
std::string System<Dim>::lifecycle_state() const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  return p_->lifecycle_.state(p_->macro_step_);
}

template <int Dim>
AcceptedCacheReadView<Dim> System<Dim>::program_cache() {
  auto accepted_read = p_->acquire_accepted_read_lease();
  return AcceptedCacheReadView<Dim>(std::move(accepted_read), &p_->program_.cache_);
}

template <int Dim>
runtime::program::CacheManager<Dim>& System<Dim>::program_cache_() {
  return p_->program_.cache_;
}

template <int Dim>
Extent<Dim> System<Dim>::spatial_shape() const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  return p_->cfg.shape;
}

template <int Dim>
double System<Dim>::time() const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  return p_->t;
}

template <int Dim>
int System<Dim>::n_species() const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  return p_->blocks_.size();
}

template <int Dim>
std::vector<std::string> System<Dim>::block_names() const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  return p_->blocks_.names();
}

template <int Dim>
EffectiveOptionsReport System<Dim>::effective_options_report() const {
  auto accepted_read = p_->acquire_accepted_read_lease();
  EffectiveOptionsReport report;
  report.runtime = "system";
  report.topology.dimension = Dim;
  report.topology.periodicity.reserve(Dim);
  for (int axis = 0; axis < Dim; ++axis)
    report.topology.periodicity.push_back(p_->periodicity[axis]);
  report.poisson.solver = p_->poisson_solver_;
  report.poisson.solver_option_schema = "pops.system.cartesian-cg-options@1";
  report.poisson.bc = p_->poisson_bc_;
  report.poisson.rel_tol = p_->poisson_rel_tol_;
  report.poisson.abs_tol = p_->poisson_abs_tol_;
  report.poisson.max_iterations = p_->poisson_max_iterations_;
  if (p_->embedded_boundary_) {
    report.eb.enabled = true;
    report.eb.geometry_mode = std::string(
        runtime::system::prepared_embedded_boundary_mode_name(p_->embedded_boundary_->mode()));
    report.eb.kappa_min = static_cast<double>(p_->embedded_boundary_->thresholds().kappa_min);
    report.eb.face_open_eps =
        static_cast<double>(p_->embedded_boundary_->thresholds().face_open_eps);
    report.eb.cut_theta_min =
        static_cast<double>(p_->embedded_boundary_->thresholds().cut_theta_min);
    report.eb.semantic_digest = p_->embedded_boundary_->semantic_digest();
    report.eb.materialization_digest = p_->embedded_boundary_->digest();
    report.eb.generation = p_->embedded_boundary_->generation();
  }

  report.blocks.reserve(p_->sp.size());
  for (const typename Impl::Species& block : p_->sp) {
    EffectiveBlockOptions row;
    row.name = block.name;
    row.ncomp = block.ncomp;
    row.substeps = block.substeps;
    row.stride = block.stride;
    row.newton = effective_newton_options(block.newton, block.newton_diagnostics);
    row.evolve = block.evolve;
    row.gamma = block.gamma;
    row.conservative_vars = block.cons_vars.names;
    row.primitive_vars = block.prim_vars.names;
    const Extent<Dim> ghosts = block.U.ghosts();
    row.n_ghost = ghosts[0];
    for (int axis = 1; axis < Dim; ++axis)
      if (ghosts[axis] != row.n_ghost)
        throw std::runtime_error(
            "System effective-options schema cannot project anisotropic ghost extents");
    report.blocks.push_back(std::move(row));
  }
  return report;
}

template std::string System<kNativeDimension>::abi_key();
template System<kNativeDimension>::System(const SystemConfig<kNativeDimension>&);
template System<kNativeDimension>::~System();
template System<kNativeDimension>::System(System&&) noexcept;
template System<kNativeDimension>& System<kNativeDimension>::operator=(System&&) noexcept;
template void System<kNativeDimension>::step(double);
template void System<kNativeDimension>::advance(double, int);
template void System<kNativeDimension>::begin_step_transaction();
template void System<kNativeDimension>::commit_step_transaction();
template double System<kNativeDimension>::step_change_l2_for_block(std::string_view) const;
template std::uint64_t System<kNativeDimension>::_step_change_l2_last_dispatches() const noexcept;
template std::map<std::string, double> System<kNativeDimension>::step_change_l2() const;
template void System<kNativeDimension>::finalize_step_transaction();
template void System<kNativeDimension>::rollback_step_transaction();
template void System<kNativeDimension>::begin_restart_transaction();
template void System<kNativeDimension>::commit_restart_transaction();
template void System<kNativeDimension>::finalize_restart_transaction() noexcept;
template void System<kNativeDimension>::rollback_restart_transaction();
template std::uint64_t System<kNativeDimension>::accepted_transaction_generation_() const noexcept;
template bool System<kNativeDimension>::accepted_transaction_fail_stop_() const noexcept;
template runtime::program::ProvisionalReadLease System<kNativeDimension>::_provisional_read_scope()
    const;
template double System<kNativeDimension>::step_cfl(double, double, double, double);
template int System<kNativeDimension>::macro_step() const;
template void System<kNativeDimension>::mark_bound();
template int System<kNativeDimension>::program_macro_step_() const;
template double System<kNativeDimension>::program_time_() const;
template std::string System<kNativeDimension>::lifecycle_state() const;
template AcceptedCacheReadView<kNativeDimension> System<kNativeDimension>::program_cache();
template runtime::program::CacheManager<kNativeDimension>&
System<kNativeDimension>::program_cache_();
template Extent<kNativeDimension> System<kNativeDimension>::spatial_shape() const;
template double System<kNativeDimension>::time() const;
template int System<kNativeDimension>::n_species() const;
template std::vector<std::string> System<kNativeDimension>::block_names() const;
template EffectiveOptionsReport System<kNativeDimension>::effective_options_report() const;

}  // namespace pops
