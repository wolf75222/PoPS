// Core exact-ranked System facade. Provider-specific installation and execution seams live in
// sibling translation units; this file owns only layout-independent lifecycle, clock and cadence.
#include "system_impl.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/checkpoint/state_carriers.hpp>
#include <pops/parallel/collective_exception.hpp>
#include <bit>
#include <pops/runtime/program/profiler.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <pops/runtime/program/step_transaction.hpp>
#include <pops/runtime/program/generated_step_bound.hpp>
#include <pops/runtime/program/collective_step_rejection.hpp>
#include <pops/runtime/program/moving_interval_checkpoint.hpp>
#include <pops/runtime/program/uniform_history_exchange_checkpoint.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace pops {

namespace {

/// The fraction that a path rhs reads while the attempts of one step run (step_cfl, or
/// run_advance with a step rule), set for the scope and restored on every exit.
class StepFractionScope {
 public:
  StepFractionScope(double& slot, double fraction) : slot_(slot), prior_(slot) { slot_ = fraction; }
  ~StepFractionScope() { slot_ = prior_; }
  StepFractionScope(const StepFractionScope&) = delete;
  StepFractionScope& operator=(const StepFractionScope&) = delete;

 private:
  double& slot_;
  double prior_;
};

}  // namespace

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
  solve_outcome_authority_ = std::make_shared<SolveOutcomeAttemptAuthority>();
  solve_outcome_authority_->owner = this;
}

// User-provided: GCC rejects an out-of-line `= default` when the same special members are
// also explicitly instantiated for kNativeDimension.
template <int Dim>
System<Dim>::~System() {
  if (solve_outcome_authority_) {
    solve_outcome_authority_->owner = nullptr;
    ++solve_outcome_authority_->incarnation;
  }
}

template <int Dim>
System<Dim>::System(System&& other) noexcept
    : prepared_boundary_execution_lane_(std::move(other.prepared_boundary_execution_lane_)),
      p_(std::move(other.p_)),
      solve_outcome_authority_(std::move(other.solve_outcome_authority_)) {
  if (solve_outcome_authority_)
    solve_outcome_authority_->owner = this;
}

template <int Dim>
System<Dim>& System<Dim>::operator=(System&& other) noexcept {
  if (this != &other) {
    if (solve_outcome_authority_) {
      solve_outcome_authority_->owner = nullptr;
      ++solve_outcome_authority_->incarnation;
    }
    // Destroy Impl first: installed field solvers and boundary transports may hold
    // ImmutableBorrow pins on the destination lane. Releasing the lane first terminates.
    p_ = std::move(other.p_);
    prepared_boundary_execution_lane_ = std::move(other.prepared_boundary_execution_lane_);
    solve_outcome_authority_ = std::move(other.solve_outcome_authority_);
    if (solve_outcome_authority_)
      solve_outcome_authority_->owner = this;
  }
  return *this;
}

template <int Dim>
SolveOutcome System<Dim>::track_solve_outcome(SolveOutcome outcome) const noexcept {
  outcome.bind_native_attempt_(solve_outcome_authority_, prepared_boundary_execution_lane_);
  return outcome;
}

template <int Dim>
void System<Dim>::require_solve_outcome_creation_(long solve_kind) const {
  const ExecutionLane& lane = prepared_boundary_execution_lane();
  runtime::program::require_step_transaction_control(
      lane, 20 + solve_kind, static_cast<long>(step_transaction_depth()),
      !p_->external_step_transaction_committed_,
      "System cannot start a solve result after the parent transaction was committed");
}

template <int Dim>
std::uint64_t System<Dim>::begin_attempt_epoch_() {
  // One invocation of the Program per attempt is what lets a retry find the slot of its first
  // attempt. A cadence with sub-steps or a stride invokes the Program several times in one
  // step(), so the first invocation of a retry would read the slot of the last sub-step of the
  // rejected attempt, a solution of another state (review A1): every invocation then opens its
  // own epoch and no solve is reused.
  const bool single_invocation = p_->program_.substeps_ == 1 && p_->program_.stride_ == 1;
  if (!p_->attempt_retry_mark_ || !single_invocation)
    ++p_->attempt_epoch_;
  p_->attempt_retry_mark_ = false;
  return p_->attempt_epoch_;
}

template <int Dim>
void System<Dim>::count_prepared_solve_(int iterations, double residual_norm) noexcept {
  ++p_->prepared_solves_;
  p_->last_prepared_solve_iterations_ = iterations;
  p_->last_prepared_solve_residual_ = residual_norm;
}

template <int Dim>
void System<Dim>::step(double dt) {
  p_->program_.require_step_installed("System::step");
  const auto communicator = prepared_boundary_execution_lane_
                                ? prepared_boundary_execution_lane_->communicator()
                                : world_communicator_view();
  if (all_reduce_max(solve_outcome_authority_->pending.load(std::memory_order_acquire) == 0 &&
                             !p_->external_step_transaction_committed_
                         ? 0L
                         : 1L,
                     communicator) != 0)
    throw std::runtime_error("System::step has a pending solve or committed parent transaction");
  runtime::program::ProfileScope scope(p_->program_.profiler_, "step");
  // Kokkos Tools sees each step as the region "pops::step": a tool attributes the
  // deep copies of array data to the step that makes them (DM8).
  const Kokkos::Profiling::ScopedRegion kokkos_step_region("pops::step");
  p_->program_.profiler_.count("steps");
  p_->execute_step_transaction(
      communicator,
      [&] {
        p_->program_.dispatch_cadence_step(p_->t, p_->macro_step_, dt, "System", communicator);
        require_continuation_projection();
        if (solve_outcome_authority_->pending.load(std::memory_order_acquire) != 0)
          throw std::logic_error("System Program step left a native solve result unconsumed");
        if (p_->has_pending_field_warm_start())
          throw std::logic_error("System Program step left an unresolved field warm start");
      },
      [&] {
        ++solve_outcome_authority_->attempt;
        rollback_field_publication_transaction();
      });
  ++solve_outcome_authority_->attempt;
}

template <int Dim>
std::string System<Dim>::advance_program_region(double dt) {
  const auto& lane = prepared_boundary_execution_lane();
  runtime::program::require_step_transaction_control(
      lane, 8, static_cast<long>(step_transaction_depth()),
      p_->external_step_transaction_ && !p_->external_step_transaction_committed_,
      "System::advance_program_region");
  runtime::program::require_step_transaction_control(
      lane, 9, static_cast<long>(step_transaction_depth()),
      solve_outcome_authority_->pending.load(std::memory_order_acquire) == 0,
      "System::advance_program_region.solve_results");
  // A Program region of a step that the caller opens and commits: Kokkos Tools sees it
  // as the region "pops::program_region" (DM8).
  const Kokkos::Profiling::ScopedRegion kokkos_program_region("pops::program_region");
  std::string port;
  try {
    runtime::program::collective_step_rejection_phase(
        lane.communicator(),
        {"pops.program-region.rejection.v1", "pops.program-region.rejection", false, false},
        "System Program region failed collectively", [&] {
          port = p_->program_.advance_cadence_region(p_->t, p_->macro_step_, dt, "System",
                                                     lane.communicator());
        });
  } catch (...) {
    p_->program_.cancel_cadence_continuation();
    ++solve_outcome_authority_->attempt;
    rollback_field_publication_transaction();
    throw;
  }
  if (all_reduce_max(
          solve_outcome_authority_->pending.load(std::memory_order_acquire) == 0 ? 0L : 1L, lane) !=
      0) {
    p_->program_.cancel_cadence_continuation();
    ++solve_outcome_authority_->attempt;
    rollback_field_publication_transaction();
    throw std::logic_error("System Program region left a native solve result unconsumed");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("system-program-region-port"), port}}, lane))
    throw std::runtime_error("System Program region reached different map ports between ranks");
  return port;
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
runtime::program::AdvanceRunReport System<Dim>::run_advance(
    double until, double proposal, std::int64_t max_steps,
    const std::vector<runtime::program::AdvanceOutputSchedule>& outputs,
    const runtime::program::AdvanceStepRule* rule) {
  using runtime::program::AdvanceRunStop;
  p_->program_.require_step_installed("System::run_advance");
  const auto communicator = prepared_boundary_execution_lane_
                                ? prepared_boundary_execution_lane_->communicator()
                                : world_communicator_view();
  const int max_attempts = p_->program_.advance_max_attempts_;
  auto& segment = p_->program_.advance_segment_;
  ExactContractBuilder arguments;
  arguments.text("pops.system-run-advance.v3")
      .scalar(until)
      .scalar(proposal)
      .presence(rule != nullptr)
      .scalar(rule ? rule->fraction : 0.0)
      .scalar(rule ? rule->max_dt : 0.0)
      .scalar(rule ? rule->speed_floor : 0.0)
      .scalar(rule ? static_cast<std::uint32_t>(rule->kind) : std::uint32_t{0})
      .text(rule ? rule->bound_identity : std::string())
      .scalar(max_steps)
      .scalar(max_attempts)
      .scalar(p_->t)
      .scalar(static_cast<std::int64_t>(p_->macro_step_))
      .scalar(segment.proposal)
      .scalar(segment.origin)
      .scalar(segment.steps)
      .scalar(segment.time)
      .scalar(segment.macro_step)
      .scalar(static_cast<std::int64_t>(outputs.size()));
  for (const auto& output : outputs) {
    arguments.scalar(output.every_steps).scalar(static_cast<std::int64_t>(output.at.size()));
    for (const double at : output.at)
      arguments.scalar(at);
    arguments.scalar(static_cast<std::int64_t>(output.arrays.size()));
    for (const auto& block : output.arrays)
      arguments.text(block);
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"system-run-advance", std::move(arguments).release()}}, communicator))
    throw std::runtime_error("System::run_advance arguments differ between ranks");
  if (max_attempts < 1)
    throw std::logic_error("System::run_advance requires a Program elaborated from an Advance");
  if (step_transaction_depth() != 0)
    throw std::logic_error("System::run_advance cannot run inside a step transaction");
  // With a step rule the proposal is the rule's selection at each step: the argument is not read.
  if (!std::isfinite(until) ||
      (rule == nullptr && (!std::isfinite(proposal) || !(proposal > 0.0))) || max_steps < 0)
    throw std::invalid_argument(
        "System::run_advance requires a finite frontier, a finite positive proposal and a "
        "non-negative step budget");
  const bool cells = rule != nullptr && rule->kind == runtime::program::AdvanceStepRuleKind::kCells;
  if (cells) {
    // A cells rule (N8 « dt= as a function of cell », D2): the proposal is the generated bound,
    // and the fraction is the Courant budget that the rule declares, in (0, 1] (C2).
    if (!std::isfinite(rule->fraction) || !(rule->fraction > 0.0) || rule->fraction > 1.0)
      throw std::invalid_argument(
          "System::run_advance cells rule requires a Courant budget in (0, 1]");
    if (rule->bound_identity.empty() ||
        rule->bound_identity != p_->program_.advance_step_bound_identity_)
      throw std::invalid_argument(
          "System::run_advance cells rule names a step bound that the installed Program does "
          "not export");
    require_cells_rule_plan_();
  } else if (rule != nullptr) {
    if (rule->kind != runtime::program::AdvanceStepRuleKind::kStability ||
        !rule->bound_identity.empty())
      throw std::invalid_argument("System::run_advance step rule has an unknown kind");
    if (!std::isfinite(rule->fraction) || !(rule->fraction > 0.0) || std::isnan(rule->max_dt) ||
        !(rule->max_dt > 0.0) || !std::isfinite(rule->speed_floor) || !(rule->speed_floor > 0.0))
      throw std::invalid_argument(
          "System::run_advance step rule requires a finite positive fraction, a positive max_dt "
          "or +infinity and a finite positive speed floor");
    // The rule selects once per step, on the accepted state: a bound that reads anything else
    // could give a retry another value (review P2 of the lot note).
    if (!p_->coupling_.dt_bounds.empty())
      throw std::logic_error(
          "System::run_advance with a step rule refuses the global bounds of add_dt_bound, "
          "which read outside the accepted state");
  }
  if (until < p_->t)
    throw std::invalid_argument("System::run_advance frontier precedes the accepted time");
  const auto& output_scalar_counts = p_->program_.advance_output_scalar_counts_;
  if (outputs.size() != output_scalar_counts.size())
    throw std::invalid_argument(
        "System::run_advance needs one schedule per Output of the installed Advance");
  runtime::program::validate_advance_output_schedule(p_->t, outputs);
  for (const auto& output : outputs)
    for (const auto& block : output.arrays)
      (void)p_->find(block);  // an unknown block fails before the first evaluation
  // A run continues the segment that the last compiled run left at this cursor with this
  // proposal; any other run starts one, as master's FixedDt grid does.
  // A run with a step rule has no fixed-proposal segment: each step starts its own below.
  if (rule == nullptr && (segment.macro_step != static_cast<std::int64_t>(p_->macro_step_) ||
                          segment.time != p_->t || segment.proposal != proposal))
    segment = {proposal, p_->t, 0, p_->t, static_cast<std::int64_t>(p_->macro_step_)};
  if (rule != nullptr)
    segment = {0.0, p_->t, 0, p_->t, static_cast<std::int64_t>(p_->macro_step_)};
  runtime::program::AdvanceRunReport report;
  const std::int64_t solves = p_->prepared_solves_;
  const auto finish = [&](AdvanceRunStop stop) {
    report.stop = stop;
    report.segment = segment;
    report.prepared_solves = p_->prepared_solves_ - solves;
    if (report.prepared_solves > 0) {
      report.last_prepared_solve_iterations = p_->last_prepared_solve_iterations_;
      report.last_prepared_solve_residual = p_->last_prepared_solve_residual_;
    }
    return report;
  };
  // An Output is evaluated after publication, outside every step (CC14, PF1): its scalars by the
  // generated entry of the artifact, its arrays by the gather of the accepted state.
  const auto evaluate = [&](std::size_t output) {
    runtime::program::AdvanceOutputSample sample;
    sample.output = static_cast<int>(output);
    sample.time = p_->t;
    sample.macro_step = static_cast<std::int64_t>(p_->macro_step_);
    const int scalars = output_scalar_counts[output];
    if (scalars > 0) {
      std::vector<Real> values(static_cast<std::size_t>(scalars), Real(0));
      p_->program_.advance_output_scalars_(static_cast<int>(output), values.data());
      sample.scalars.assign(values.begin(), values.end());
    }
    for (const auto& block : outputs[output].arrays)
      sample.arrays.push_back(state_global(block));
    report.outputs.push_back(std::move(sample));
  };
  std::vector<std::size_t> cursors(outputs.size(), 0);
  // A run evaluates every Output once at its start, on the state from which it starts.
  for (std::size_t output = 0; output < outputs.size(); ++output)
    evaluate(output);
  while (p_->t < until) {
    if (report.steps == max_steps)
      return finish(AdvanceRunStop::kStepBudget);
    const double frontier = runtime::program::advance_next_frontier(p_->t, until, outputs, cursors);
    double step_proposal = proposal;
    if (cells) {
      // The generated bound on the accepted state, taken once for the attempts of the step; at
      // +infinity on every cell the step goes to the frontier (2.6.9). The step starts its own
      // segment, as a stability rule's does.
      step_proposal = step_bound();
      p_->last_dt_reason_ = "cells:step_bound";
      if (!std::isfinite(step_proposal))
        step_proposal = frontier - p_->t;
      segment = {step_proposal, p_->t, 0, p_->t, static_cast<std::int64_t>(p_->macro_step_)};
    } else if (rule != nullptr) {
      // b6's selection on the accepted state, taken once for the attempts of the step; the step
      // starts its own segment, so the landing rules are those of a single proposal.
      step_proposal = select_step_(rule->fraction, rule->speed_floor, rule->max_dt, 0.0);
      segment = {step_proposal, p_->t, 0, p_->t, static_cast<std::int64_t>(p_->macro_step_)};
    }
    const auto duration = runtime::program::advance_attempt_duration(p_->t, step_proposal, frontier,
                                                                     segment.origin, segment.steps);
    if (rule != nullptr && (duration.lands || duration.dt != step_proposal))
      p_->last_dt_reason_ = "frontier";  // cut or landed by the frontier (review P1)
    // The attempts of a rule run at its fraction, as step_cfl's execution does: a path rhs checks
    // its budget against it.
    std::optional<StepFractionScope> fraction_scope;
    if (rule != nullptr)
      fraction_scope.emplace(p_->active_program_step_courant_, rule->fraction);
    // The epoch before the first attempt of this step: a retry keeps the epoch of the step only
    // when an earlier attempt opened it, so an attempt rejected before its begin_step never
    // lends a later one the epoch of the previous accepted step (review of the code, O1).
    const std::uint64_t epoch_before_step = p_->attempt_epoch_;
    for (std::int64_t attempt = 1;; ++attempt) {
      // A retry computes the same accepted step: its solves whose dependency set excludes the
      // proposal keep their solution (T7 across attempts). The mark is taken by the first
      // begin_step of the attempt and cleared whatever step() does.
      struct RetryMark {
        bool& mark;
        ~RetryMark() { mark = false; }
      } retry{p_->attempt_retry_mark_};
      retry.mark = attempt > 1 && p_->attempt_epoch_ != epoch_before_step;
      try {
        step(duration.dt);
        break;
      } catch (const runtime::program::StepAttemptRejected& rejected) {
        // step() restored the accepted state on every rank before the rejection left it.
        const std::string& phase = rejected.phase();
        if (phase == runtime::program::kTrialFailurePhase) {
          // A fixed failure site (T38): a retry cannot change it, so the step ends here and the
          // attempt is not counted as a rejection.
          report.attempts = attempt;
          report.rejection = rejected.what();
          report.rejection_phase = phase;
          report.rejection_detail = rejected.detail();
          return finish(AdvanceRunStop::kTrialFailure);
        }
        if (phase != runtime::program::kAssessmentRejectionPhase &&
            phase != runtime::program::kTrialRejectionPhase)
          throw;
        ++report.rejections;
        report.rejection = rejected.what();
        report.rejection_phase = phase;
        report.rejection_detail = rejected.detail();
        if (cells) {
          // The face guard refused the attempt with the faces' frequency (ROM-108). The compiled
          // run does not retry at that frequency yet: it stops by name, with the largest frequency
          // over the ranks, on every rank (design note D2, K1-c; review rev 3, P4).
          double frequency = 0.0;
          if (rejected.typed_failure_schema() == runtime::program::kFaceStepFrequencySchema) {
            const std::string& text = rejected.typed_failure_payload();
            char* end = nullptr;
            frequency = std::strtod(text.c_str(), &end);
            if (end == text.c_str() || *end != '\0' || !std::isfinite(frequency) ||
                !(frequency > 0.0))
              frequency = std::numeric_limits<double>::infinity();
          }
          const ExecutionLane& lane = prepared_boundary_execution_lane();
          frequency = static_cast<double>(all_reduce_max(static_cast<Real>(frequency),
                                                         lane.communicator()));
          if (frequency > 0.0) {
            report.attempts = attempt;
            report.face_frequency = frequency;
            return finish(AdvanceRunStop::kFaceFrequency);
          }
        }
        if (attempt >= max_attempts) {
          report.attempts = attempt;
          return finish(AdvanceRunStop::kAttemptBudget);
        }
      }
    }
    ++report.steps;
    if (rule != nullptr) {
      // A rule's step ends its own segment; no fixed-proposal run continues it (proposal 0).
      segment.proposal = 0.0;
      segment.origin = p_->t;
      segment.steps = 0;
    } else if (duration.continuous) {
      ++segment.steps;
    } else {
      segment.origin = p_->t;
      segment.steps = 0;
    }
    segment.time = p_->t;
    segment.macro_step = static_cast<std::int64_t>(p_->macro_step_);
    if (duration.lands && p_->t != frontier)
      throw std::logic_error("System::run_advance did not publish its frontier exactly");
    for (std::size_t output = 0; output < outputs.size(); ++output)
      if (runtime::program::advance_output_due(
              output, p_->t, static_cast<std::int64_t>(p_->macro_step_), outputs, cursors))
        evaluate(output);
  }
  return finish(AdvanceRunStop::kReached);
}

template <int Dim>
int System<Dim>::advance_max_attempts() const noexcept {
  return p_->program_.advance_max_attempts_;
}

template <int Dim>
const std::string& System<Dim>::step_bound_identity() const noexcept {
  return p_->program_.advance_step_bound_identity_;
}

template <int Dim>
void System<Dim>::begin_step_transaction() {
  const auto& lane = prepared_boundary_execution_lane();
  runtime::program::require_step_transaction_control(
      lane, 0, static_cast<long>(step_transaction_depth()),
      !p_->external_step_transaction_ &&
          solve_outcome_authority_->pending.load(std::memory_order_acquire) == 0,
      "System::begin_step_transaction");
  Kokkos::fence();
  std::unique_ptr<typename Impl::AcceptedSnapshot> candidate;
  std::exception_ptr error;
  try {
    candidate = std::make_unique<typename Impl::AcceptedSnapshot>(*p_);
  } catch (...) {
    error = std::current_exception();
  }
  if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && error)
      std::rethrow_exception(error);
    throw std::runtime_error("System step snapshot preparation failed collectively");
  }
  p_->external_step_transaction_ = std::move(candidate);
  p_->external_step_transaction_committed_ = false;
  p_->program_.accepted_exchanges_.clear();
  p_->program_.begin_step_projection_report();
  ++solve_outcome_authority_->attempt;
}

template <int Dim>
void System<Dim>::begin_nested_step_transaction() {
  const auto& lane = prepared_boundary_execution_lane();
  runtime::program::require_step_transaction_control(
      lane, 1, static_cast<long>(step_transaction_depth()),
      p_->external_step_transaction_ && !p_->external_step_transaction_committed_ &&
          !p_->external_restart_transaction_ &&
          solve_outcome_authority_->pending.load(std::memory_order_acquire) == 0,
      "System::begin_nested_step_transaction");
  Kokkos::fence();
  std::unique_ptr<typename Impl::AcceptedSnapshot> candidate;
  std::exception_ptr error;
  try {
    candidate = std::make_unique<typename Impl::AcceptedSnapshot>(*p_);
    p_->parent_step_transactions_.reserve(p_->parent_step_transactions_.size() + 1);
  } catch (...) {
    error = std::current_exception();
  }
  if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && error)
      std::rethrow_exception(error);
    throw std::runtime_error("System nested snapshot preparation failed collectively");
  }
  p_->parent_step_transactions_.push_back(std::move(p_->external_step_transaction_));
  p_->external_step_transaction_ = std::move(candidate);
  ++solve_outcome_authority_->attempt;
}

template <int Dim>
std::size_t System<Dim>::step_transaction_depth() const noexcept {
  return p_->external_step_transaction_ ? p_->parent_step_transactions_.size() + 1 : 0;
}

template <int Dim>
void System<Dim>::stage_program_exchange(runtime::program::ExchangeRecord record) {
  const auto& lane = prepared_boundary_execution_lane();
  auto& ledger = p_->program_.accepted_exchanges_;
  const auto prior_size = ledger.records().size();
  std::exception_ptr error;
  try {
    ledger.stage(std::move(record));
  } catch (...) {
    error = std::current_exception();
  }
  if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
    ledger.restore_size(prior_size);
    if (lane.size() == 1 && error)
      std::rethrow_exception(error);
    throw std::runtime_error("System exchange staging failed collectively");
  }
}

template <int Dim>
void System<Dim>::stage_program_exchanges(std::span<runtime::program::ExchangeRecord> records) {
  runtime::program::stage_exchange_batch_collectively(p_->program_.accepted_exchanges_, records,
                                                      prepared_boundary_execution_lane());
}

template <int Dim>
std::vector<runtime::program::ExchangeRecord> System<Dim>::program_exchange_records() const {
  return p_->program_.accepted_exchanges_.records();
}

template <int Dim>
std::vector<std::vector<std::uint8_t>> System<Dim>::capture_state_storage_(
    bool provisional_capture) const {
  const auto& lane = prepared_boundary_execution_lane();
  using namespace runtime::checkpoint;
  using Bits = std::conditional_t<sizeof(Real) == 8, std::uint64_t, std::uint32_t>;
  std::string shard;
  std::exception_ptr error;
  std::string invocation;
  std::vector<ExactOrderedBytePair> authority;
  try {
    const auto phase = p_->lifecycle_.state(p_->macro_step_);
    const auto depth = step_transaction_depth();
    const bool provisional = provisional_capture && depth == 1 && p_->external_step_transaction_ &&
                             !p_->external_step_transaction_committed_;
    if ((depth != 0 && !provisional) || p_->external_restart_transaction_ ||
        (phase != "bound" && phase != "running" && phase != "checkpointed"))
      throw std::logic_error(
          "accepted state storage observation requires bound accepted idle state");
    StateCarrierArchive<Dim> image;
    image.real_bits = sizeof(Real) * 8;
    image.ranks = lane.size();
    image.shard = lane.rank();
    image.levels = 1;
    image.blocks = p_->blocks_.names();
    if (image.blocks.empty() && !p_->program_.continuation_.configured())
      throw std::logic_error(
          "accepted state storage observation requires blocks or a complete Continuation");
    if (!std::isfinite(p_->t) || p_->macro_step_ < 0)
      throw std::logic_error("accepted state storage observation has an invalid clock");
    ExactContractBuilder contract;
    contract.text("accepted-state-storage-observation@1")
        .scalar(std::int32_t{Dim})
        .scalar(p_->t)
        .scalar(std::int64_t{p_->macro_step_});
    for (const auto& name : image.blocks)
      contract.text(name);
    invocation = std::move(contract).release();
    authority.emplace_back("accepted-state-storage-observation", invocation);
    Kokkos::fence();
    for (std::size_t block = 0; block < image.blocks.size(); ++block) {
      const auto& field = p_->find(image.blocks[block]).U;
      for (std::size_t local = 0; local < field.local_size(); ++local) {
        const auto& fab = field.fab(local);
        StateCarrierPatch<Dim> row;
        row.block = block;
        row.level = 0;
        row.patch = field.global_index(local);
        row.components = field.ncomp();
        row.owner = field.distribution().replicated() ? -1 : lane.rank();
        for (int axis = 0; axis < Dim; ++axis) {
          row.lo[axis] = fab.box().lo[axis];
          row.hi[axis] = fab.box().hi[axis];
          row.grown_lo[axis] = fab.grown_box().lo[axis];
          row.grown_hi[axis] = fab.grown_box().hi[axis];
        }
        auto host = fab.create_host_mirror();
        fab.copy_to_host(host);
        row.bits.reserve(host.size());
        for (std::size_t value = 0; value < host.size(); ++value)
          row.bits.push_back(std::bit_cast<Bits>(host(value)));
        image.patches.push_back(std::move(row));
      }
    }
    const auto encoded = encode_state_carriers(image);
    shard.assign(reinterpret_cast<const char*>(encoded.data()), encoded.size());
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane, "accepted state storage observation staging");
  if (!all_ranks_agree_exact_ordered_byte_pairs(authority, lane))
    throw std::runtime_error("accepted state storage observation differs across ranks");
  // ExecutionLane owns the authenticated runtime communicator. Its byte broadcast primitive
  // chunks at the MPI count boundary; ObserverMpiLane::allgather_bytes is not a runtime API.
  std::vector<std::string> shards;
  error = {};
  try {
    shards.resize(static_cast<std::size_t>(lane.size()));
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane, "state carrier source-shard allocation");
  for (int source = 0; source < lane.size(); ++source) {
    std::array<char, 8> length_bytes{};
    error = {};
    try {
      if (lane.rank() == source) {
        if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t))
          if (shard.size() > std::numeric_limits<std::uint64_t>::max())
            throw std::length_error("state carrier source shard exceeds uint64 wire capacity");
        const auto length = static_cast<std::uint64_t>(shard.size());
        for (unsigned byte = 0; byte < 8; ++byte)
          length_bytes[byte] = static_cast<char>(length >> (8 * byte));
      }
    } catch (...) {
      error = std::current_exception();
    }
    collectively_rethrow_exception(error, lane, "state carrier source-shard length staging");
    broadcast_bytes_inplace(length_bytes.data(), length_bytes.size(), lane, source);
    std::uint64_t length = 0;
    for (unsigned byte = 0; byte < 8; ++byte)
      length |= std::uint64_t(static_cast<unsigned char>(length_bytes[byte])) << (8 * byte);
    error = {};
    try {
      if (length > std::numeric_limits<std::size_t>::max())
        throw std::length_error("state carrier source shard exceeds destination size_t capacity");
      auto& destination = shards[static_cast<std::size_t>(source)];
      if (lane.rank() == source)
        destination = shard;
      else
        destination.resize(static_cast<std::size_t>(length));
    } catch (...) {
      error = std::current_exception();
    }
    collectively_rethrow_exception(error, lane, "state carrier source-shard payload staging");
    auto& destination = shards[static_cast<std::size_t>(source)];
    broadcast_bytes_inplace(destination.data(), destination.size(), lane, source);
  }
  std::vector<std::uint8_t> result;
  error = {};
  try {
    auto image = merge_state_carrier_shards<Dim>(shards);
    validate_complete_state_carriers(image);
    result = encode_state_carriers(image);
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane, "state carrier capture canonicalization");
  std::vector<std::vector<std::uint8_t>> observation;
  error = {};
  try {
    observation.emplace_back(shard.begin(), shard.end());
    observation.push_back(std::move(result));
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane,
                                 "accepted state storage observation result allocation");
  return observation;
}

template <int Dim>
std::vector<std::vector<std::uint8_t>> System<Dim>::observe_accepted_state_storage() const {
  return capture_state_storage_(false);
}

template <int Dim>
std::vector<std::uint8_t> System<Dim>::checkpoint_state_carriers() const {
  auto images = capture_state_storage_(true);
  return std::move(images[1]);
}

template <int Dim>
std::uint64_t System<Dim>::checkpoint_state_carriers_capacity() const {
  using namespace runtime::checkpoint;
  const auto& lane = prepared_boundary_execution_lane();
  std::uint64_t result = 0;
  std::exception_ptr error;
  try {
    std::vector<StateCarrierStorageCapacity<Dim>> blocks;
    if (p_->blocks_.names().empty() && !p_->program_.continuation_.configured())
      throw std::logic_error("state carrier capacity requires blocks or a complete Continuation");
    std::uint64_t cells = 0;
    for (const auto& name : p_->blocks_.names()) {
      const auto& field = p_->find(name).U;
      StateCarrierStorageCapacity<Dim> block;
      block.name = name;
      block.components = field.ncomp();
      for (int axis = 0; axis < Dim; ++axis)
        block.ghosts[axis] = field.ghosts()[axis];
      blocks.push_back(std::move(block));
      std::uint64_t count = 0;
      const auto& layout = field.layout();
      for (std::size_t patch = 0; patch < layout.size(); ++patch) {
        const auto& box = layout[patch];
        const auto n = static_cast<std::uint64_t>(box.numPts());
        if (n > std::numeric_limits<std::uint64_t>::max() - count)
          throw std::overflow_error("Uniform carrier cells overflow");
        count += n;
      }
      cells = std::max(cells, count);
    }
    const std::array<std::uint64_t, 1> levels{cells};
    result = state_carriers_byte_capacity<Dim>(levels, blocks);
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane, "Uniform carrier configured capacity");
  return result;
}

template <int Dim>
void System<Dim>::validate_checkpoint_state_carriers(std::span<const std::uint8_t> bytes) const {
  const auto image = runtime::checkpoint::decode_state_carriers<Dim>(bytes);
  runtime::checkpoint::validate_complete_state_carriers(image);
  if (image.real_bits != sizeof(Real) * 8 || image.levels != 1 ||
      image.blocks != p_->blocks_.names())
    throw std::invalid_argument("Uniform carrier storage authority differs");
  std::size_t at = 0;
  for (std::size_t block = 0; block < image.blocks.size(); ++block) {
    const auto& field = p_->find(image.blocks[block]).U;
    for (std::size_t patch = 0; patch < field.layout().size(); ++patch) {
      if (at == image.patches.size())
        throw std::invalid_argument("Uniform carrier missing patch");
      const auto& row = image.patches[at++];
      const auto& valid = field.layout()[patch];
      if (row.block != block || row.level != 0 || row.patch != patch ||
          row.components != std::uint64_t(field.ncomp()))
        throw std::invalid_argument("Uniform carrier patch/component authority differs");
      for (int axis = 0; axis < Dim; ++axis)
        if (row.lo[axis] != valid.lo[axis] || row.hi[axis] != valid.hi[axis] ||
            row.grown_lo[axis] != std::int64_t(valid.lo[axis]) - field.ghosts()[axis] ||
            row.grown_hi[axis] != std::int64_t(valid.hi[axis]) + field.ghosts()[axis])
          throw std::invalid_argument("Uniform carrier valid/grown geometry differs");
    }
  }
  if (at != image.patches.size())
    throw std::invalid_argument("Uniform carrier extra patch");
}

template <int Dim>
void System<Dim>::restore_checkpoint_state_carriers(std::span<const std::uint8_t> bytes) {
  using Bits = std::conditional_t<sizeof(Real) == 8, std::uint64_t, std::uint32_t>;
  const auto& lane = prepared_boundary_execution_lane();
  std::vector<std::pair<MultiFab<Dim>*, std::unique_ptr<MultiFab<Dim>>>> candidates;
  std::exception_ptr error;
  try {
    if (!p_->external_restart_transaction_ || p_->external_step_transaction_committed_)
      throw std::logic_error("Uniform carrier restore requires uncommitted restart transaction");
    validate_checkpoint_state_carriers(bytes);
    const auto image = runtime::checkpoint::decode_state_carriers<Dim>(bytes);
    std::size_t begin = 0;
    for (const auto& name : image.blocks) {
      auto& target = p_->find(name).U;
      auto candidate =
          std::make_unique<MultiFab<Dim>>(target.layout(), target.distribution(),
                                          target.local_rank(), target.ncomp(), target.ghosts());
      for (std::size_t local = 0; local < target.local_size(); ++local) {
        const auto& row = image.patches[begin + target.global_index(local)];
        const auto& live = target.fab(local);
        auto prior = live.create_host_mirror();
        live.copy_to_host(prior);
        auto& fab = candidate->fab(local);
        auto host = fab.create_host_mirror();
        if (row.bits.size() != host.size())
          throw std::invalid_argument("Uniform carrier storage size differs");
        const auto stride = static_cast<std::size_t>(fab.grown_box().numPts());
        for (int component = 0; component < target.ncomp(); ++component)
          for (std::size_t cell = 0; cell < static_cast<std::size_t>(fab.box().numPts()); ++cell) {
            auto remaining = cell;
            std::size_t address = std::size_t(component) * stride, axis_stride = 1;
            for (int axis = 0; axis < Dim; ++axis) {
              const auto length = static_cast<std::size_t>(fab.box().length(axis));
              const auto coordinate = std::int64_t(fab.box().lo[axis]) + remaining % length;
              remaining /= length;
              address +=
                  static_cast<std::size_t>(coordinate - fab.grown_box().lo[axis]) * axis_stride;
              axis_stride *= static_cast<std::size_t>(fab.grown_box().length(axis));
            }
            if (row.bits[address] != std::bit_cast<Bits>(prior(address)))
              throw std::invalid_argument(
                  "Uniform carrier valid cells contradict scientific state projection");
          }
        for (std::size_t value = 0; value < host.size(); ++value)
          host(value) = std::bit_cast<Real>(static_cast<Bits>(row.bits[value]));
        fab.copy_from_host(host);
      }
      begin += target.layout().size();
      candidates.emplace_back(&target, std::move(candidate));
    }
    Kokkos::fence();
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane, "Uniform carrier restore preparation");
  const std::string_view authority(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"Uniform-state-carriers", authority}}, lane))
    throw std::invalid_argument("Uniform carrier bytes differ across ranks");
  error = {};
  try {
    for (const auto& [target, candidate] : candidates)
      for (std::size_t local = 0; local < target->local_size(); ++local)
        Kokkos::deep_copy(target->fab(local).storage(), candidate->fab(local).storage());
    Kokkos::fence();
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane, "Uniform carrier restore publication");
}

template <int Dim>
void System<Dim>::declare_program_integral(const std::string& identity, Real initial) {
  runtime::program::declare_integral_collectively(p_->program_.accepted_exchanges_, identity,
                                                  initial, prepared_boundary_execution_lane());
}

template <int Dim>
Real System<Dim>::program_integral(const std::string& identity) const {
  return static_cast<Real>(p_->program_.accepted_exchanges_.integral(identity));
}

template <int Dim>
Real System<Dim>::consume_program_external_trace(
    const std::string& identity,
    const runtime::program::AcceptedExchangeLedger::TraceSelection& selection, Real scale) {
  return static_cast<Real>(runtime::program::consume_external_trace_collectively(
      p_->program_.accepted_exchanges_, identity, selection, scale,
      prepared_boundary_execution_lane()));
}

template <int Dim>
std::vector<std::uint8_t> System<Dim>::checkpoint_program_exchanges(
    bool provisional_capture) const {
  if (!p_->program_.moving_interval_geometry_.empty()) {
    if ((step_transaction_depth() != 0 &&
         !(provisional_capture && step_transaction_depth() == 1)) ||
        p_->external_restart_transaction_)
      throw std::logic_error("moving geometry checkpoint export requires fully accepted state");
    if (step_transaction_depth() == 1)
      for (const auto& [identity, geometry] : p_->program_.moving_interval_geometry_)
        if (!geometry.last_receipt)
          throw std::logic_error(
              "moving candidate checkpoint requires a terminal interval receipt");
    const auto bytes = runtime::program::checkpoint_moving_intervals(
        p_->program_.accepted_exchanges_, p_->program_.moving_interval_geometry_,
        [&](int block) -> const MultiFab<Dim>& {
          return p_->sp.at(static_cast<std::size_t>(block)).U;
        });
    validate_checkpoint_moving_geometry(bytes, p_->t, p_->macro_step_);
    return runtime::program::checkpoint_uniform_history_exchanges(bytes, p_->program_.hist_);
  }
  return runtime::program::checkpoint_uniform_history_exchanges(
      p_->program_.accepted_exchanges_.checkpoint(), p_->program_.hist_);
}

template <int Dim>
void System<Dim>::validate_checkpoint_program_exchanges(std::span<const std::uint8_t> bytes) const {
  const auto history = runtime::program::read_uniform_history_exchanges(bytes, p_->program_.hist_);
  runtime::program::require_history_field_program(history.fields, p_->program_.installed_hash_);
  (void)runtime::program::read_moving_checkpoint(
      history.exchanges, p_->program_.accepted_exchanges_, p_->program_.moving_interval_geometry_,
      [&](int block) -> const MultiFab<Dim>& {
        return p_->sp.at(static_cast<std::size_t>(block)).U;
      });
}

template <int Dim>
std::vector<std::uint8_t> System<Dim>::remap_checkpoint_program_exchanges(
    std::span<const std::span<const std::uint8_t>> source_images) const {
  const auto& lane = prepared_boundary_execution_lane();
  return runtime::program::rematerialize_uniform_history_exchanges<Dim>(
      source_images, p_->program_.hist_,
      [&](int program_block, const runtime::program::FieldExchangeExpressions::Cell& cell) {
        if (program_block < 0 ||
            static_cast<std::size_t>(program_block) >= p_->program_.block_map_.size() ||
            cell.size() != Dim)
          throw std::invalid_argument(
              "retained face expression has no installed field owner/dimension");
        const int block = p_->program_.block_map_.at(static_cast<std::size_t>(program_block));
        if (block < 0 || static_cast<std::size_t>(block) >= p_->sp.size())
          throw std::invalid_argument("retained face expression has a foreign runtime owner");
        Index<Dim> coordinate;
        for (int d = 0; d < Dim; ++d) {
          if (cell[d] < std::numeric_limits<int>::min() ||
              cell[d] > std::numeric_limits<int>::max())
            throw std::invalid_argument("retained face expression exceeds native coordinate range");
          coordinate[d] = static_cast<int>(cell[d]);
        }
        const auto& field = p_->sp.at(static_cast<std::size_t>(block)).U;
        for (std::size_t patch = 0; patch < field.local_size(); ++patch)
          if (field.layout()[field.global_index(patch)].contains(coordinate))
            return true;
        return false;
      },
      lane.rank(), lane.size());
}

template <int Dim>
void System<Dim>::validate_checkpoint_moving_geometry(std::span<const std::uint8_t> bytes,
                                                      double accepted_time, int macro_step) const {
  const auto history = runtime::program::read_uniform_history_exchanges(bytes, p_->program_.hist_);
  runtime::program::require_history_field_program(history.fields, p_->program_.installed_hash_);
  const auto candidate = runtime::program::read_moving_checkpoint(
      history.exchanges, p_->program_.accepted_exchanges_, p_->program_.moving_interval_geometry_,
      [&](int block) -> const MultiFab<Dim>& {
        return p_->sp.at(static_cast<std::size_t>(block)).U;
      });
  runtime::program::require_moving_checkpoint_lifecycle(candidate, accepted_time, macro_step);
}

template <int Dim>
void System<Dim>::restore_checkpoint_program_exchanges(std::span<const std::uint8_t> bytes) {
  const auto& lane = prepared_boundary_execution_lane();
  std::optional<runtime::program::MovingCheckpointCandidate<Dim>> candidate;
  std::optional<runtime::program::UniformHistoryExchangeImage<Dim>> history;
  std::exception_ptr error;
  try {
    if (!p_->external_restart_transaction_)
      throw std::logic_error("accepted exchange restore requires the native restart transaction");
    history.emplace(
        runtime::program::read_uniform_history_exchanges(bytes, p_->program_.hist_, true));
    runtime::program::require_history_field_program(history->fields, p_->program_.installed_hash_);
    candidate.emplace(runtime::program::read_moving_checkpoint(
        history->exchanges, p_->program_.accepted_exchanges_,
        p_->program_.moving_interval_geometry_, [&](int block) -> const MultiFab<Dim>& {
          return p_->sp.at(static_cast<std::size_t>(block)).U;
        }));
    if (!candidate->geometry.empty())
      runtime::program::require_moving_checkpoint_lifecycle(*candidate, p_->t, p_->macro_step_);
    for (const auto& [block, expected] : candidate->states) {
      const auto& restored = p_->sp.at(static_cast<std::size_t>(block)).U;
      for (std::size_t patch = 0; patch < restored.local_size(); ++patch) {
        auto actual = restored.fab(patch).create_host_mirror();
        restored.fab(patch).copy_to_host(actual);
        auto saved = expected.fab(patch).create_host_mirror();
        expected.fab(patch).copy_to_host(saved);
        runtime::system::marshaling::for_each_host_index(
            restored.box(patch), [&](const Index<Dim>& index, std::size_t) {
              for (int component = 0; component < restored.ncomp(); ++component) {
                const auto a = runtime::system::marshaling::storage_ordinal(restored.fab(patch),
                                                                            index, component);
                const auto b = runtime::system::marshaling::storage_ordinal(expected.fab(patch),
                                                                            index, component);
                if (actual(a) != saved(b))
                  throw std::invalid_argument(
                      "moving checkpoint physical state must be restored together with its "
                      "geometry/receipt");
              }
            });
      }
    }
  } catch (...) {
    error = std::current_exception();
  }
  if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && error)
      std::rethrow_exception(error);
    throw std::runtime_error("accepted exchange restore preparation failed collectively");
  }
  if (!history->fields.empty()) {
    std::vector<std::uint8_t> metadata;
    std::exception_ptr metadata_error;
    try {
      metadata = runtime::program::encode_history_field_metadata(
          history->fields, p_->program_.hist_.field_metadata_budget);
    } catch (...) {
      metadata_error = std::current_exception();
    }
    collectively_rethrow_exception(metadata_error, lane, "history Field metadata restart identity");
    const std::array<std::pair<std::string_view, std::string_view>, 1> contract{
        {{"pops.history.field-metadata",
          std::string_view(reinterpret_cast<const char*>(metadata.data()), metadata.size())}}};
    if (!all_ranks_agree_exact_ordered_byte_pairs(contract, lane))
      throw std::invalid_argument("history Field metadata differs between restart ranks");
  }
  runtime::program::require_integral_values_agree_collectively(candidate->exchanges, lane);
  if (!candidate->geometry.empty())
    runtime::program::require_moving_checkpoint_agrees_collectively(*candidate, lane);
  p_->program_.accepted_exchanges_.swap(candidate->exchanges);
  p_->program_.moving_interval_geometry_.swap(candidate->geometry);
  if (!history->histories.empty())
    p_->program_.hist_.slot_exchanges.swap(history->histories);
  if (!history->fields.empty())
    p_->program_.hist_.slot_fields.swap(history->fields);
}

template <int Dim>
void System<Dim>::commit_step_transaction() {
  runtime::program::require_step_transaction_control(
      prepared_boundary_execution_lane(), 2, static_cast<long>(step_transaction_depth()),
      p_->external_step_transaction_ && !p_->external_step_transaction_committed_ &&
          solve_outcome_authority_->pending.load(std::memory_order_acquire) == 0 &&
          !p_->has_pending_field_warm_start(),
      "System::commit_step_transaction");
  Kokkos::fence();
  p_->external_step_transaction_committed_ = true;
  ++solve_outcome_authority_->attempt;
}

template <int Dim>
std::map<std::string, double> System<Dim>::step_change_l2() const {
  if (!p_->external_step_transaction_)
    throw std::runtime_error("System::step_change_l2 requires an active external step transaction");
  const std::vector<MultiFab<Dim>>& previous = p_->external_step_transaction_->states;
  if (previous.size() != p_->sp.size())
    throw std::runtime_error("System::step_change_l2 snapshot composition mismatch");

  double cell_measure = 1.0;
  for (int axis = 0; axis < Dim; ++axis)
    cell_measure *= static_cast<double>(p_->geom.spacing(axis));

  std::map<std::string, double> result;
  for (std::size_t block = 0; block < p_->sp.size(); ++block) {
    const double sum_sq =
        static_cast<double>(difference_sum_sq_all(p_->sp[block].U, previous[block]));
    result.emplace(p_->sp[block].name, std::sqrt(cell_measure * sum_sq));
  }
  return result;
}

template <int Dim>
void System<Dim>::finalize_step_transaction() {
  runtime::program::require_step_transaction_control(
      prepared_boundary_execution_lane(), 3, static_cast<long>(step_transaction_depth()),
      p_->external_step_transaction_ && p_->external_step_transaction_committed_ &&
          solve_outcome_authority_->pending.load(std::memory_order_acquire) == 0,
      "System::finalize_step_transaction");
  Kokkos::fence();
  p_->external_step_transaction_.reset();
  if (!p_->parent_step_transactions_.empty()) {
    p_->external_step_transaction_ = std::move(p_->parent_step_transactions_.back());
    p_->parent_step_transactions_.pop_back();
  }
  p_->external_step_transaction_committed_ = false;
  p_->external_restart_transaction_ = false;
  ++solve_outcome_authority_->attempt;
}

template <int Dim>
void System<Dim>::rollback_step_transaction() {
  runtime::program::require_step_transaction_control(
      prepared_boundary_execution_lane(), 4, static_cast<long>(step_transaction_depth()),
      static_cast<bool>(p_->external_step_transaction_) &&
          solve_outcome_authority_->pending.load(std::memory_order_acquire) == 0,
      "System::rollback_step_transaction");
  Kokkos::fence();
  p_->program_.cancel_cadence_continuation();
  p_->external_step_transaction_->restore(*p_);
  p_->external_step_transaction_.reset();
  if (!p_->parent_step_transactions_.empty()) {
    p_->external_step_transaction_ = std::move(p_->parent_step_transactions_.back());
    p_->parent_step_transactions_.pop_back();
  }
  p_->external_step_transaction_committed_ = false;
  p_->external_restart_transaction_ = false;
  ++solve_outcome_authority_->attempt;
}

template <int Dim>
void System<Dim>::begin_restart_transaction() {
  runtime::program::require_step_transaction_control(
      prepared_boundary_execution_lane(), 5, static_cast<long>(step_transaction_depth()),
      !p_->external_step_transaction_, "System::begin_restart_transaction");
  begin_step_transaction();
  p_->external_restart_transaction_ = true;
}

template <int Dim>
void System<Dim>::commit_restart_transaction() {
  require_continuation_projection();
  runtime::program::require_step_transaction_control(
      prepared_boundary_execution_lane(), 6, static_cast<long>(step_transaction_depth()),
      p_->external_restart_transaction_, "System::commit_restart_transaction");
  commit_step_transaction();
}

template <int Dim>
void System<Dim>::finalize_restart_transaction() noexcept {
  // Commit refused every pending result and new solve creation is barred while committed. This
  // finalizer runs after external effects publish, so it must remain total and non-throwing.
  if (!p_->external_restart_transaction_ || !p_->external_step_transaction_committed_)
    return;
  p_->program_.finalize_restart_resources();
  p_->external_step_transaction_.reset();
  p_->external_step_transaction_committed_ = false;
  p_->external_restart_transaction_ = false;
  ++solve_outcome_authority_->attempt;
}

template <int Dim>
void System<Dim>::rollback_restart_transaction() {
  runtime::program::require_step_transaction_control(
      prepared_boundary_execution_lane(), 7, static_cast<long>(step_transaction_depth()),
      p_->external_restart_transaction_, "System::rollback_restart_transaction");
  rollback_step_transaction();
}

template <int Dim>
double System<Dim>::step_cfl(double fraction, double speed_floor, double max_dt, double min_dt) {
  const double selected = select_step_(fraction, speed_floor, max_dt, min_dt);
  const ExecutionLane& lane = prepared_boundary_execution_lane();
  {
    // The step runs at its fraction (a path rhs checks its budget against it), restored after the
    // step on every exit, before the attempt count moves.
    const StepFractionScope fraction_scope(p_->active_program_step_courant_, fraction);
    p_->execute_step_transaction(
        lane.communicator(),
        [&] {
          p_->program_.dispatch_cadence_step(p_->t, p_->macro_step_, selected, "System",
                                             lane.communicator());
          if (solve_outcome_authority_->pending.load(std::memory_order_acquire) != 0)
            throw std::logic_error("System Program CFL step left a native solve result unconsumed");
        },
        [&] {
          ++solve_outcome_authority_->attempt;
          rollback_field_publication_transaction();
        });
  }
  ++solve_outcome_authority_->attempt;
  return selected;
}

template <int Dim>
double System<Dim>::select_step_(double cfl, double speed_floor, double max_dt, double min_dt) {
  const ExecutionLane& lane = prepared_boundary_execution_lane();
  runtime::program::require_step_transaction_control(
      lane, 10, static_cast<long>(step_transaction_depth()),
      solve_outcome_authority_->pending.load(std::memory_order_acquire) == 0 &&
          !p_->external_step_transaction_committed_,
      "System::step_cfl.solve_results");
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
        .scalar(std::uint32_t{2})
        .scalar(std::int32_t{Dim})
        .scalar(cfl)
        .scalar(speed_floor)
        .scalar(max_dt)
        .scalar(min_dt)
        .scalar(static_cast<std::uint64_t>(p_->sp.size()))
        .presence(p_->explicit_default_poisson_requested_);
    for (const typename Impl::Species& block : p_->sp) {
      contract.text(block.name)
          .scalar(block.evolve)
          .scalar(block.substeps)
          .scalar(block.stride)
          .presence(static_cast<bool>(block.add_poisson_rhs))
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

  const bool has_default_poisson_source =
      std::any_of(p_->sp.begin(), p_->sp.end(),
                  [](const auto& block) { return static_cast<bool>(block.add_poisson_rhs); });
  if (p_->explicit_default_poisson_requested_ || has_default_poisson_source) {
    SolveOutcome field_outcome = solve_fields();
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
    if (!block.stability_consumer_qid.empty()) {
      // CFL consumes the accepted image before the next numerical interval is known.
      // Publish the same exact transport dependency closure without pretending that a
      // residual or future stage has already been evaluated.
      runtime::system::AuxiliaryEvaluationPoint auxiliary;
      std::exception_ptr publication_point_error;
      try {
        runtime::multiblock::BoundaryEvaluationPoint point;
        const auto& metadata = p_->program_.checkpoint_metadata_;
        point.clock = metadata.primary_clock_identity;
        const bool physical_clock = !point.clock.empty();
        if (physical_clock && std::find(metadata.logical_clock_identities.begin(),
                                        metadata.logical_clock_identities.end(),
                                        point.clock) == metadata.logical_clock_identities.end())
          throw std::invalid_argument("System CFL auxiliary Clock is not installed");
        if (!physical_clock)
          point.clock = "pops.system.accepted-stability";
        point.tick = p_->macro_step_;
        point.level = 0;
        point.stage_fraction = {0, 1};
        point.dt = static_cast<double>(p_->program_.last_dt_);
        point.physical_time = p_->t;
        if (point.tick < 0 || !std::isfinite(point.dt) || point.dt < 0 ||
            (physical_clock && point.tick != 0 && point.dt == 0) ||
            !std::isfinite(point.physical_time))
          throw std::invalid_argument("System CFL auxiliary requires its live accepted point");
        auxiliary.clock = point.clock;
        auxiliary.accepted_step = static_cast<std::uint64_t>(point.tick);
        auxiliary.layout_generation = p_->embedded_boundary_generation_;
        auxiliary.event = runtime::system::AuxiliaryEvaluationEvent::before_stability;
        if (physical_clock) {
          if (point.dt > 0)
            auxiliary.qualify_physical_evaluation(point);
          else {
            runtime::system::AuxiliaryPhysicalEvaluation physical(point.stage_fraction, point.dt,
                                                                  point.physical_time);
            physical.qualify_native_accepted_initial(point);
            auxiliary.physical_evaluation = std::move(physical);
          }
        }
        auxiliary.validate();
      } catch (...) {
        publication_point_error = std::current_exception();
      }
      runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
          publication_point_error, &lane, "System CFL auxiliary point failed collectively");
      if (refresh_auxiliary_(auxiliary, {block.stability_consumer_qid}) ==
          runtime::system::AuxiliaryPublicationStatus::nonfinite_candidate)
        throw std::runtime_error("System CFL auxiliary candidate contains non-finite values");
    }
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
  return selected;
}

template <int Dim>
double System<Dim>::step_bound() {
  using runtime::program::StepBoundCode;
  using runtime::program::StepBoundMode;
  const ExecutionLane& lane = prepared_boundary_execution_lane();
  const auto& entry = p_->program_.advance_step_bound_;
  // Every rank holds the same installed Program: the presence of the entry is collective.
  if (all_reduce_max(entry ? 0L : 1L, lane) != 0)
    throw std::logic_error(
        "System::step_bound requires a Program elaborated from dt= as a function of cell");
  const auto reduce = [&](StepBoundMode mode, double target) {
    double local = std::numeric_limits<double>::lowest();
    std::exception_ptr error;
    try {
      if (entry(static_cast<int>(mode), target, &local) != 0)
        throw std::logic_error("System::step_bound: the generated step bound refused its mode");
    } catch (...) {
      error = std::current_exception();
    }
    if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && error)
        std::rethrow_exception(error);
      throw std::runtime_error("System::step_bound evaluation failed collectively");
    }
    return static_cast<double>(all_reduce_max(static_cast<Real>(local), lane.communicator()));
  };
  // The largest code over the cells and the ranks: a failure if any cell fails, otherwise whether
  // a cell bounds the step (PF8).
  const double code = reduce(StepBoundMode::kCode, 0.0);
  if (code >= static_cast<double>(StepBoundCode::kNonpositive)) {
    const bool nonfinite = code >= static_cast<double>(StepBoundCode::kNonfinite);
    const double first = reduce(StepBoundMode::kFirstCell, code);
    // The linear index of the first cell, axis 0 fastest, in the domain of the geometry.
    const Box<Dim> domain = p_->geom.domain();
    std::int64_t linear = static_cast<std::int64_t>(-first);
    std::string cell = "(";
    for (int axis = 0; axis < Dim; ++axis) {
      const std::int64_t length = domain.length(axis);
      cell += (axis ? ", " : "") + std::to_string(domain.lo[axis] + linear % length);
      linear /= length;
    }
    cell += ")";
    throw std::domain_error(
        std::string(nonfinite ? "step_bound_nonfinite" : "step_bound_nonpositive") +
        ": dt= as a function of cell is " +
        (nonfinite ? "not finite (NaN, -infinity or an operation that is not finite)"
                   : "zero or negative") +
        " at the cell " + cell + " of the accepted state at t=" + std::to_string(p_->t));
  }
  if (code < static_cast<double>(StepBoundCode::kFinite))
    return std::numeric_limits<double>::infinity();  // +infinity on every cell: to the frontier
  return -reduce(StepBoundMode::kValue, 0.0);
}

template <int Dim>
void System<Dim>::require_cells_rule_plan_() const {
  // b6's selection does more than read speeds on these plans (design note D2, C3, C5): the
  // elaborator refuses them, and the run checks the installed System again.
  const char* refused = nullptr;
  int evolving = 0;
  if (p_->explicit_default_poisson_requested_)
    refused = "a default Poisson solve";
  for (const typename Impl::Species& block : p_->sp) {
    if (!block.evolve)
      continue;
    ++evolving;
    if (block.add_poisson_rhs)
      refused = "a default Poisson solve";
    else if (!block.stability_consumer_qid.empty())
      refused = "an auxiliary refreshed before the stability selection";
    else if (block.source_frequency)
      refused = "a source frequency";
    else if (block.parabolic_frequency)
      refused = "a parabolic frequency";
    else if (block.stability_dt)
      refused = "a stability_dt";
    else if (block.substeps != 1 || block.stride != 1)
      refused = "substeps or a stride other than 1";
  }
  if (!p_->coupling_.coupled_freqs.empty() || !p_->coupling_.coupled_frequencies.empty())
    refused = "a coupled frequency";
  if (evolving != 1)
    refused = "a number of evolving blocks other than one";
  if (!p_->coupling_.dt_bounds.empty())
    refused = "the global bounds of add_dt_bound, which read outside the accepted state";
  if (p_->program_.dt_bound_)
    refused = "the Program dt bound of set_dt_bound, a closure on cfl that a cells rule has not";
  if (refused != nullptr)
    throw std::logic_error(
        std::string("System::run_advance with dt= as a function of cell refuses ") + refused +
        "; a later lot of N8 « dt= as a function of cell » elaborates it");
}

template <int Dim>
void System<Dim>::step_at_fraction(double dt, double fraction) {
  if (!std::isfinite(fraction) || !(fraction > 0.0) || fraction > 1.0)
    throw std::invalid_argument("System::step_at_fraction requires a Courant budget in (0, 1]");
  // The attempt runs at the budget of its cells rule, as run_advance's do (design note D4).
  const StepFractionScope fraction_scope(p_->active_program_step_courant_, fraction);
  step(dt);
}

template <int Dim>
double System<Dim>::active_program_step_courant() const {
  return p_->active_program_step_courant_;
}

template <int Dim>
int System<Dim>::macro_step() const {
  return p_->macro_step_;
}

template <int Dim>
void System<Dim>::mark_bound() {
  // Local boundary assembly does not establish communicator-wide shared-face ownership.
  // Refuse a divergent mask before sealing auxiliary providers or publishing bound state.
  const ExecutionLane& lane = prepared_boundary_execution_lane();
  std::string boundary_contract;
  std::exception_ptr boundary_error;
  try {
    boundary_contract = p_->boundary_registry_.interface_face_omission_contract();
  } catch (...) {
    boundary_error = std::current_exception();
  }
  if (all_reduce_max(boundary_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && boundary_error)
      std::rethrow_exception(boundary_error);
    throw std::runtime_error("System interface face ownership preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"system-interface-face-omission", boundary_contract}}, lane))
    throw std::runtime_error("System interface face ownership differs across MPI ranks");

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
  prepare_bound_physical_group_();
  p_->lifecycle_.to_bound();
}

template <int Dim>
std::string System<Dim>::lifecycle_state() const {
  return p_->lifecycle_.state(p_->macro_step_);
}

template <int Dim>
runtime::program::CacheManager<Dim>& System<Dim>::program_cache() {
  return p_->program_.cache_;
}

template <int Dim>
Extent<Dim> System<Dim>::spatial_shape() const {
  return p_->cfg.shape;
}

template <int Dim>
double System<Dim>::time() const {
  return p_->t;
}

template <int Dim>
int System<Dim>::n_species() const {
  return p_->blocks_.size();
}

template <int Dim>
std::vector<std::string> System<Dim>::block_names() const {
  return p_->blocks_.names();
}

template <int Dim>
EffectiveOptionsReport System<Dim>::effective_options_report() const {
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

#include "system_continuation.inc"

template std::string System<kNativeDimension>::abi_key();
template System<kNativeDimension>::System(const SystemConfig<kNativeDimension>&);
template System<kNativeDimension>::~System();
template System<kNativeDimension>::System(System&&) noexcept;
template System<kNativeDimension>& System<kNativeDimension>::operator=(System&&) noexcept;
template SolveOutcome System<kNativeDimension>::track_solve_outcome(SolveOutcome) const noexcept;
template void System<kNativeDimension>::require_solve_outcome_creation_(long) const;
template std::uint64_t System<kNativeDimension>::begin_attempt_epoch_();
template void System<kNativeDimension>::count_prepared_solve_(int, double) noexcept;
template void System<kNativeDimension>::step(double);
template void System<kNativeDimension>::advance(double, int);
template runtime::program::AdvanceRunReport System<kNativeDimension>::run_advance(
    double, double, std::int64_t, const std::vector<runtime::program::AdvanceOutputSchedule>&,
    const runtime::program::AdvanceStepRule*);
template int System<kNativeDimension>::advance_max_attempts() const noexcept;
template std::string System<kNativeDimension>::advance_program_region(double);
template void System<kNativeDimension>::begin_step_transaction();
template void System<kNativeDimension>::begin_nested_step_transaction();
template std::size_t System<kNativeDimension>::step_transaction_depth() const noexcept;
template void System<kNativeDimension>::stage_program_exchange(runtime::program::ExchangeRecord);
template void System<kNativeDimension>::stage_program_exchanges(
    std::span<runtime::program::ExchangeRecord>);
template std::vector<runtime::program::ExchangeRecord>
System<kNativeDimension>::program_exchange_records() const;
template void System<kNativeDimension>::declare_program_integral(const std::string&, Real);
template Real System<kNativeDimension>::program_integral(const std::string&) const;
template Real System<kNativeDimension>::consume_program_external_trace(
    const std::string&, const runtime::program::AcceptedExchangeLedger::TraceSelection&, Real);
template std::vector<std::uint8_t> System<kNativeDimension>::checkpoint_program_exchanges(
    bool) const;
template void System<kNativeDimension>::validate_checkpoint_program_exchanges(
    std::span<const std::uint8_t>) const;
template std::vector<std::uint8_t> System<kNativeDimension>::remap_checkpoint_program_exchanges(
    std::span<const std::span<const std::uint8_t>>) const;
template void System<kNativeDimension>::validate_checkpoint_moving_geometry(
    std::span<const std::uint8_t>, double, int) const;
template void System<kNativeDimension>::restore_checkpoint_program_exchanges(
    std::span<const std::uint8_t>);

template void System<kNativeDimension>::commit_step_transaction();
template std::map<std::string, double> System<kNativeDimension>::step_change_l2() const;
template void System<kNativeDimension>::finalize_step_transaction();
template void System<kNativeDimension>::rollback_step_transaction();
template void System<kNativeDimension>::begin_restart_transaction();
template void System<kNativeDimension>::commit_restart_transaction();
template void System<kNativeDimension>::finalize_restart_transaction() noexcept;
template void System<kNativeDimension>::rollback_restart_transaction();
template double System<kNativeDimension>::step_cfl(double, double, double, double);
template double System<kNativeDimension>::select_step_(double, double, double, double);
template double System<kNativeDimension>::step_bound();
template void System<kNativeDimension>::step_at_fraction(double, double);
template void System<kNativeDimension>::require_cells_rule_plan_() const;
template double System<kNativeDimension>::active_program_step_courant() const;
template int System<kNativeDimension>::macro_step() const;
template void System<kNativeDimension>::mark_bound();
template std::string System<kNativeDimension>::lifecycle_state() const;
template runtime::program::CacheManager<kNativeDimension>&
System<kNativeDimension>::program_cache();
template Extent<kNativeDimension> System<kNativeDimension>::spatial_shape() const;
template double System<kNativeDimension>::time() const;
template int System<kNativeDimension>::n_species() const;
template std::vector<std::string> System<kNativeDimension>::block_names() const;
template EffectiveOptionsReport System<kNativeDimension>::effective_options_report() const;

}  // namespace pops

namespace pops {
template std::vector<std::vector<std::uint8_t>>
System<kNativeDimension>::observe_accepted_state_storage() const;
template std::vector<std::uint8_t> System<kNativeDimension>::checkpoint_state_carriers() const;
template std::uint64_t System<kNativeDimension>::checkpoint_state_carriers_capacity() const;
template void System<kNativeDimension>::validate_checkpoint_state_carriers(
    std::span<const std::uint8_t>) const;
template void System<kNativeDimension>::restore_checkpoint_state_carriers(
    std::span<const std::uint8_t>);
}  // namespace pops
