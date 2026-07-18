#pragma once

/// @file
/// @brief Persistent storage for prepared affine Krylov solves.

#include <pops/numerics/elliptic/linear/krylov_method_provider.hpp>
#include <pops/numerics/elliptic/linear/scaled_scalar.hpp>
#include <pops/parallel/solve_report_consensus.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pops {

namespace detail {
struct KrylovWorkspaceAccess;
}

struct KrylovControls {
  PreparedKrylovMethod method{};
  Real rel_tol = Real(1e-8);
  Real abs_tol = Real(0);
  int max_iterations = 1;
};

class KrylovWorkspace {
 public:
  static int max_batched_basis_extent() {
    return max_krylov_batched_basis_extent(detail::PreparedFieldAlgebra::kRobustDotPayloadWidth);
  }

  static int max_batched_basis_extent(std::size_t robust_payload_width) {
    return max_krylov_batched_basis_extent(robust_payload_width);
  }

  KrylovWorkspace(
      const MultiFab& prototype, PreparedKrylovMethod method, KrylovFootprint footprint,
      PreparedVectorDistribution vector_distribution = PreparedVectorDistribution::Distributed,
      PreparedVectorMetric metric = {})
      : KrylovWorkspace(ExecutionCommunicator::world(), "pops.krylov-workspace", prototype,
                        std::move(method), footprint, std::move(vector_distribution),
                        std::move(metric)) {}

  /// Materialize one independent solve workspace on an authenticated communicator. Distinct
  /// workspaces own distinct duplicated lanes; `lane_identity` names the logical workspace in the
  /// parent's canonical materialization order and is never inferred from a process-local address.
  KrylovWorkspace(
      const ExecutionCommunicator& execution_communicator, std::string_view lane_identity,
      const MultiFab& prototype, PreparedKrylovMethod method, KrylovFootprint footprint,
      PreparedVectorDistribution vector_distribution = PreparedVectorDistribution::Distributed,
      PreparedVectorMetric metric = {})
      : method_(std::move(method)),
        footprint_(footprint),
        vector_distribution_(std::move(vector_distribution)),
        metric_(std::move(metric)),
        lane_(ExecutionLane::duplicate_collectively(execution_communicator, lane_identity)) {
    const long footprint_failure_local =
        footprint_.components != prototype.ncomp() || footprint_.input_ghosts != prototype.n_grow()
            ? 1L
            : 0L;
    if (all_reduce_max(footprint_failure_local, lane_) != 0)
      throw std::invalid_argument("KrylovWorkspace footprint disagrees with prototype");

    // Layout authentication is a materialization-time provider callback.  Its exact contract and
    // every dynamic consensus allocation are completed here, never lazily inside solve.
    vector_distribution_.require_collective_layout(prototype, "KrylovWorkspace", lane_);
    long vector_space_failure_local = 0;
    try {
      layout_ = detail::layout_fingerprint(prototype, vector_distribution_);
      vector_distribution_layout_valid_ =
          detail::field_distribution_layout_matches(prototype, vector_distribution_);
      if (!metric_)
        metric_ = PreparedVectorMetric::euclidean(prototype, vector_distribution_);
      if (!metric_.compatible_with(prototype, vector_distribution_))
        vector_space_failure_local = 1;
    } catch (...) {
      vector_space_failure_local = 1;
    }
    if (all_reduce_max(vector_space_failure_local, lane_) != 0)
      throw std::invalid_argument("KrylovWorkspace metric disagrees with the vector space");
    metric_fingerprint_ = detail::fingerprint_seed();
    detail::fingerprint_mix(metric_fingerprint_, metric_.collective_contract());
    distribution_fingerprint_ = detail::fingerprint_seed();
    detail::fingerprint_mix(distribution_fingerprint_, vector_distribution_.collective_contract());
    long provider_failure_local = 0;
    try {
      requirements_ = method_.workspace_requirements(
          KrylovWorkspaceRequest{footprint_, vector_distribution_, metric_.robust_payload_width()});
    } catch (...) {
      provider_failure_local = 1;
    }
    if (all_reduce_max(provider_failure_local, lane_) != 0)
      throw std::invalid_argument(
          "prepared Krylov provider failed to produce workspace requirements on at least one "
          "communicator rank");

    std::string requirements_contract;
    std::exception_ptr requirements_contract_error;
    try {
      ExactContractBuilder builder;
      append_requirements_contract_(builder, requirements_);
      requirements_contract = std::move(builder).release();
    } catch (...) {
      requirements_contract_error = std::current_exception();
    }
    if (all_reduce_max(requirements_contract_error ? 1L : 0L, lane_) != 0)
      throw std::runtime_error(
          "prepared Krylov workspace requirement-contract construction failed on at least one "
          "communicator rank");
    if (!all_ranks_agree_exact_ordered_byte_pairs({{std::string_view("prepared-krylov-workspace"),
                                                    std::string_view(requirements_contract)}},
                                                  lane_))
      throw std::invalid_argument(
          "prepared Krylov workspace requirements differ between communicator ranks");
    long materialization_failure_local = 0;
    try {
      distribution_validation_data_.assign(vector_distribution_.validation_scratch_byte_count(),
                                           char{0});
      fields_.reserve(requirements_.field_count);
      for (std::size_t index = 0; index < requirements_.field_count; ++index)
        fields_.emplace_back(prototype.box_array(), prototype.dmap(), prototype.ncomp(),
                             footprint_.input_ghosts);
      // One private communication cache per workspace. Fields inside this workspace execute
      // sequentially and may share that cache; another workspace never sees it.
      fields_.front().detach_communication_caches();
      fields_.front().halo_cache();
      for (std::size_t index = 1; index < fields_.size(); ++index)
        fields_[index].share_halo_cache_from(fields_.front());
      real_values_.assign(requirements_.real_count, Real(0));
      scaled_values_.assign(requirements_.scaled_scalar_count, detail::ScaledScalar::zero());
      collective_values_.assign(requirements_.collective_value_count, 0.0);
      distribution_reduction_data_.assign(
          vector_distribution_.reduction_scratch_value_count(
              std::max(requirements_.reduction_value_capacity, std::size_t{1})),
          0.0);
      metric_reduction_data_.assign(metric_.reduction_scratch_value_count(), 0.0);
      state_words_.assign(requirements_.state_word_count, std::uint64_t{0});
      if (footprint_.preconditioned) {
        preconditioner_constant_.emplace(prototype.box_array(), prototype.dmap(), prototype.ncomp(),
                                         footprint_.input_ghosts);
        preconditioner_constant_->share_halo_cache_from(fields_.front());
      }
    } catch (...) {
      materialization_failure_local = 1;
    }
    if (all_reduce_max(materialization_failure_local, lane_) != 0) {
      fields_.clear();
      real_values_.clear();
      scaled_values_.clear();
      collective_values_.clear();
      distribution_reduction_data_.clear();
      metric_reduction_data_.clear();
      distribution_validation_data_.clear();
      state_words_.clear();
      preconditioner_constant_.reset();
      throw std::runtime_error(
          "prepared Krylov persistent workspace materialization failed on at least one "
          "communicator rank");
    }
    allocation_count_ = requirements_.field_count + (footprint_.preconditioned ? 1u : 0u);
  }
  static std::size_t required_fields(const PreparedKrylovMethod& method,
                                     const KrylovWorkspaceRequest& request) {
    return method.workspace_requirements(request).field_count;
  }

  void bind(const PreparedAffineLinearProblem& problem) {
    const bool workspace_reserved = try_reserve_mutation_();
    WorkspaceMutationReservation workspace_reservation(workspace_reserved ? this : nullptr);
    const bool problem_reserved = detail::PreparedProblemAccess::try_reserve_use(problem);
    ProblemUseReservation problem_reservation(problem_reserved ? &problem : nullptr);
    const detail::PreparedProblemControlConsensus reservation_consensus =
        detail::coordinate_prepared_problem_control(
            detail::PreparedProblemControlOperation::BindWorkspace, workspace_reserved,
            problem_reserved, detail::PreparedProblemAccess::preparation_lane(problem));
    if (!reservation_consensus.operation_agrees)
      throw std::logic_error(
          "prepared Krylov control operations differ across communicator ranks; prepare, bind, "
          "and solve materialization must use one canonical collective order");
    if (reservation_consensus.workspace_reservation_failed != 0)
      throw std::logic_error(
          "KrylovWorkspace cannot be rebound while another bind or solve invocation is active");
    if (reservation_consensus.problem_reservation_failed != 0)
      throw std::logic_error(
          "KrylovWorkspace cannot be rebound while its prepared problem is mutating or in "
          "exclusive use");
    detail::KrylovCollectivePayload payload;
    long local_failure = detail::PreparedProblemAccess::append_collective_state(problem, payload);
    append_collective_state_(payload);
    if (local_failure == 0) {
      try {
        if (!detail::PreparedProblemAccess::preparation_lane(problem).congruent_with(lane_))
          local_failure = 10;
      } catch (...) {
        // MPI_Comm_compare is fallible. Convert a rank-local MPI error into the same workspace-lane
        // failure gate as every other bind precondition before any provider session is touched.
        local_failure = 11;
      }
    }
    if (local_failure == 0 && !vector_distribution_layout_valid_)
      local_failure = 8;
    if (local_failure == 0 &&
        (problem.layout_fingerprint() != layout_ || problem.footprint() != footprint_ ||
         problem.vector_distribution() != vector_distribution_))
      local_failure = 4;
    if (local_failure == 0 &&
        (detail::PreparedProblemAccess::metric_fingerprint(problem) != metric_fingerprint_ ||
         problem.metric().collective_contract() != metric_.collective_contract()))
      local_failure = 9;
    if (local_failure == 0 && problem.has_preconditioner() != footprint_.preconditioned)
      local_failure = 5;
    const bool agrees = detail::collective_payload_agrees(payload, lane_);
    const long collective_failure = all_reduce_max(local_failure, lane_);
    throw_collective_failure_(collective_failure);
    if (!agrees)
      throw std::logic_error("KrylovWorkspace bind contract differs across communicator ranks");

    const detail::PreparedProviderSourceIdentity operator_source_identity =
        detail::PreparedProblemAccess::operator_source_identity(problem);
    const detail::PreparedProviderSourceIdentity preconditioner_source_identity =
        detail::PreparedProblemAccess::preconditioner_source_identity(problem);
    const bool can_reuse_sessions_local =
        operator_session_ && operator_source_identity_ == operator_source_identity &&
        (!problem.has_preconditioner() ||
         (preconditioner_session_ &&
          preconditioner_source_identity_ == preconditioner_source_identity));
    // Source identities are deliberately process-local. Make the lifecycle decision uniform before
    // any provider callback: if even one rank cannot prove exact source continuity, every rank
    // rematerializes its private sessions from the already-consensual semantic provider contract.
    const bool reuse_sessions = all_reduce_min(can_reuse_sessions_local ? 1L : 0L, lane_) != 0;

    PreparedAffineOperatorSession candidate_operator_session;
    PreparedLinearPreconditionerSession candidate_preconditioner_session;
    long gauge_materialization_failure_local = 0;
    try {
      gauge_coefficients_.assign(detail::PreparedProblemAccess::nullspace_basis_count(problem),
                                 0.0);
    } catch (...) {
      gauge_materialization_failure_local = 1;
    }
    if (all_reduce_max(gauge_materialization_failure_local, lane_) != 0) {
      invalidate_bound_state_();
      throw std::runtime_error(
          "KrylovWorkspace nullspace scratch materialization failed on at least one "
          "communicator rank");
    }

    if (!reuse_sessions) {
      long operator_materialization_failure_local = 0;
      try {
        candidate_operator_session =
            detail::PreparedProblemAccess::make_operator_session(problem, lane_);
      } catch (...) {
        operator_materialization_failure_local = 1;
      }
      if (all_reduce_max(operator_materialization_failure_local, lane_) != 0) {
        invalidate_bound_state_();
        throw std::runtime_error(
            "KrylovWorkspace affine-operator session materialization failed on at least one "
            "communicator rank");
      }

      if (problem.has_preconditioner()) {
        long preconditioner_materialization_failure_local = 0;
        try {
          candidate_preconditioner_session =
              detail::PreparedProblemAccess::make_preconditioner_session(problem, lane_);
        } catch (...) {
          preconditioner_materialization_failure_local = 1;
        }
        if (all_reduce_max(preconditioner_materialization_failure_local, lane_) != 0) {
          invalidate_bound_state_();
          throw std::runtime_error(
              "KrylovWorkspace preconditioner session materialization failed on at least one "
              "communicator rank");
        }
      }
    }

    PreparedAffineOperatorSession& operator_session =
        reuse_sessions ? operator_session_ : candidate_operator_session;
    PreparedLinearPreconditionerSession& preconditioner_session =
        reuse_sessions ? preconditioner_session_ : candidate_preconditioner_session;

    std::size_t operator_session_allocation_count = 0;
    std::size_t preconditioner_session_allocation_count = 0;
    long operator_preparation_failure_local = 0;
    try {
      operator_session.reset_apply_status();
      operator_session.prepare();
      MultiFab& zero = field(0);
      MultiFab& operator_probe = field(1);
      detail::PreparedFieldAlgebra::zero(zero);
      detail::PreparedFieldAlgebra::zero(operator_probe);
    } catch (...) {
      operator_preparation_failure_local = 1;
    }
    if (all_reduce_max(operator_preparation_failure_local, lane_) != 0) {
      invalidate_bound_state_();
      throw std::runtime_error(
          "KrylovWorkspace affine-operator session preparation failed on at least one "
          "communicator rank");
    }

    // Hot apply callbacks are noexcept and return rank-local status.  Every rank must execute the
    // same callback trace, then agree on that status before any exact-value collective is entered.
    // In particular, never turn a local failure into an exception inside a larger try block whose
    // peers may already have advanced to the next provider or collective callback.
    MultiFab& zero = field(0);
    MultiFab& operator_probe = field(1);
    const PreparedApplyStatus operator_probe_status = operator_session.apply(operator_probe, zero);
    if (all_reduce_max(prepared_apply_succeeded(operator_probe_status) ? 0L : 1L, lane_) != 0) {
      invalidate_bound_state_();
      throw std::runtime_error(
          "prepared affine operator failed its workspace bind probe on at least one "
          "communicator rank");
    }
    vector_distribution_.require_exact_values(
        operator_probe, distribution_validation_data_,
        "KrylovWorkspace prepared affine operator zero response", lane_);
    long operator_comparison_failure_local = 0;
    long operator_constant_mismatch_local = 0;
    try {
      operator_constant_mismatch_local =
          detail::PreparedFieldAlgebra::local_exact_values_equal(operator_probe,
                                                                 problem.constant_term())
              ? 0L
              : 1L;
    } catch (...) {
      operator_comparison_failure_local = 1;
    }
    if (all_reduce_max(operator_comparison_failure_local, lane_) != 0) {
      invalidate_bound_state_();
      throw std::runtime_error(
          "KrylovWorkspace affine-operator zero-response comparison failed on at least one "
          "communicator rank");
    }
    if (all_reduce_max(operator_constant_mismatch_local, lane_) != 0) {
      invalidate_bound_state_();
      throw std::runtime_error(
          "KrylovWorkspace affine-operator session disagrees with the prepared problem's exact "
          "zero response");
    }

    long operator_allocation_count_failure_local = 0;
    try {
      operator_session_allocation_count = operator_session.allocation_count();
      if (operator_session_allocation_count >
          static_cast<std::size_t>(std::numeric_limits<long>::max()))
        throw std::overflow_error("prepared session field count exceeds collective capacity");
    } catch (...) {
      operator_allocation_count_failure_local = 1;
    }
    if (all_reduce_max(operator_allocation_count_failure_local, lane_) != 0) {
      invalidate_bound_state_();
      throw std::runtime_error(
          "KrylovWorkspace affine-operator allocation-count query failed on at least one "
          "communicator rank");
    }
    const long operator_count = static_cast<long>(operator_session_allocation_count);
    if (all_reduce_min(operator_count, lane_) != all_reduce_max(operator_count, lane_)) {
      invalidate_bound_state_();
      throw std::runtime_error(
          "prepared affine-operator persistent-field counts differ between communicator ranks");
    }

    if (problem.has_preconditioner()) {
      long preconditioner_preparation_failure_local = 0;
      try {
        preconditioner_session.reset_apply_status();
        preconditioner_session.prepare();
        detail::PreparedFieldAlgebra::zero(zero);
        detail::PreparedFieldAlgebra::zero(*preconditioner_constant_);
      } catch (...) {
        preconditioner_preparation_failure_local = 1;
      }
      if (all_reduce_max(preconditioner_preparation_failure_local, lane_) != 0) {
        invalidate_bound_state_();
        throw std::runtime_error(
            "KrylovWorkspace preconditioner session preparation failed on at least one "
            "communicator rank");
      }

      const PreparedApplyStatus preconditioner_probe_status =
          preconditioner_session.apply(*preconditioner_constant_, zero);
      if (all_reduce_max(prepared_apply_succeeded(preconditioner_probe_status) ? 0L : 1L, lane_) !=
          0) {
        invalidate_bound_state_();
        throw std::runtime_error(
            "prepared preconditioner failed its workspace bind probe on at least one "
            "communicator rank");
      }
      vector_distribution_.require_exact_values(
          *preconditioner_constant_, distribution_validation_data_,
          "KrylovWorkspace prepared preconditioner constant", lane_);

      long preconditioner_allocation_count_failure_local = 0;
      try {
        preconditioner_session_allocation_count = preconditioner_session.allocation_count();
        if (preconditioner_session_allocation_count >
            static_cast<std::size_t>(std::numeric_limits<long>::max()))
          throw std::overflow_error("prepared session field count exceeds collective capacity");
      } catch (...) {
        preconditioner_allocation_count_failure_local = 1;
      }
      if (all_reduce_max(preconditioner_allocation_count_failure_local, lane_) != 0) {
        invalidate_bound_state_();
        throw std::runtime_error(
            "KrylovWorkspace preconditioner allocation-count query failed on at least one "
            "communicator rank");
      }
      const long preconditioner_count = static_cast<long>(preconditioner_session_allocation_count);
      if (all_reduce_min(preconditioner_count, lane_) !=
          all_reduce_max(preconditioner_count, lane_)) {
        invalidate_bound_state_();
        throw std::runtime_error(
            "prepared preconditioner persistent-field counts differ between communicator ranks");
      }
    }
    if (!reuse_sessions) {
      operator_session_ = std::move(candidate_operator_session);
      preconditioner_session_ = std::move(candidate_preconditioner_session);
    }
    operator_source_identity_ = operator_source_identity;
    preconditioner_source_identity_ = problem.has_preconditioner()
                                          ? preconditioner_source_identity
                                          : detail::PreparedProviderSourceIdentity{};
    operator_session_allocation_count_ = operator_session_allocation_count;
    preconditioner_session_allocation_count_ = preconditioner_session_allocation_count;
    snapshot_ = detail::PreparedProblemAccess::stored_snapshot(problem);
  }

  void require_bound(const PreparedAffineLinearProblem& problem,
                     const KrylovControls& controls) const {
    detail::KrylovCollectivePayload payload;
    long local_failure = detail::PreparedProblemAccess::append_collective_state(problem, payload);
    append_collective_state_(payload);
    append_controls_(payload, controls);
    const auto& problem_snapshot = detail::PreparedProblemAccess::stored_snapshot(problem);
    if (local_failure == 0 && (!snapshot_ || !problem_snapshot || *snapshot_ != *problem_snapshot))
      local_failure = 6;
    if (local_failure == 0 && !(controls.method == method_))
      local_failure = 7;
    const bool agrees = detail::collective_payload_agrees(payload, lane_);
    const long collective_failure = all_reduce_max(local_failure, lane_);
    throw_collective_failure_(collective_failure);
    if (!agrees)
      throw std::logic_error("KrylovWorkspace bound contract differs across communicator ranks");
  }

  const PreparedKrylovMethod& method() const { return method_; }
  const PreparedVectorDistribution& vector_distribution() const { return vector_distribution_; }
  const PreparedVectorMetric& metric() const { return metric_; }
  const KrylovFootprint& footprint() const { return footprint_; }
  /// Number of persistent MultiFab work vectors (not heap-allocation events), including storage
  /// owned by the currently bound operator and preconditioner sessions.  Provider sessions report
  /// their real private storage through the prepared-session protocol; zero means no MultiFab
  /// storage, not "unknown".
  std::size_t allocation_count() const noexcept {
    return allocation_count_ + operator_session_allocation_count_ +
           preconditioner_session_allocation_count_;
  }
  std::size_t scalar_value_count() const { return real_values_.size() + scaled_values_.size(); }
  std::size_t collective_value_count() const {
    return collective_values_.size() + distribution_reduction_data_.size();
  }
  std::size_t distribution_validation_byte_count() const {
    return distribution_validation_data_.size();
  }

 private:
  friend struct detail::KrylovWorkspaceAccess;
  friend class PreparedKrylovSolveContext;

  enum class ReservationState : std::uint8_t {
    Idle,
    Mutation,
    Solve,
  };

  class WorkspaceMutationReservation final {
   public:
    explicit WorkspaceMutationReservation(KrylovWorkspace* workspace) noexcept
        : workspace_(workspace) {}
    WorkspaceMutationReservation(const WorkspaceMutationReservation&) = delete;
    WorkspaceMutationReservation& operator=(const WorkspaceMutationReservation&) = delete;
    ~WorkspaceMutationReservation() {
      if (workspace_ != nullptr)
        workspace_->release_mutation_();
    }

   private:
    KrylovWorkspace* workspace_ = nullptr;
  };

  class ProblemUseReservation final {
   public:
    explicit ProblemUseReservation(const PreparedAffineLinearProblem* problem) noexcept
        : problem_(problem) {}
    ProblemUseReservation(const ProblemUseReservation&) = delete;
    ProblemUseReservation& operator=(const ProblemUseReservation&) = delete;
    ~ProblemUseReservation() {
      if (problem_ != nullptr)
        detail::PreparedProblemAccess::release_use(*problem_);
    }

   private:
    const PreparedAffineLinearProblem* problem_ = nullptr;
  };

  static void append_footprint_(detail::KrylovCollectivePayload& payload,
                                const KrylovFootprint& footprint) noexcept {
    payload.append(footprint.components);
    payload.append(footprint.input_ghosts);
    payload.append(static_cast<std::uint8_t>(footprint.preconditioned));
  }

  static void append_controls_(detail::KrylovCollectivePayload& payload,
                               const KrylovControls& controls) noexcept {
    payload.append(controls.method.fingerprint());
    payload.append(std::bit_cast<std::uint64_t>(controls.rel_tol));
    payload.append(std::bit_cast<std::uint64_t>(controls.abs_tol));
    payload.append(controls.max_iterations);
  }

  void append_collective_state_(detail::KrylovCollectivePayload& payload) const noexcept {
    payload.append(method_.fingerprint());
    append_footprint_(payload, footprint_);
    append_requirements_(payload, requirements_);
    payload.append(layout_);
    payload.append(distribution_fingerprint_);
    payload.append(static_cast<std::uint8_t>(vector_distribution_layout_valid_));
    payload.append(static_cast<std::uint8_t>(snapshot_.has_value()));
    payload.append(snapshot_.value_or(OperatorEvaluationSnapshot{}));
    payload.append(metric_fingerprint_);
    payload.append(static_cast<std::uint64_t>(metric_.robust_payload_width()));
  }

  static void append_requirements_(detail::KrylovCollectivePayload& payload,
                                   const KrylovWorkspaceRequirements& requirements) noexcept {
    payload.append(static_cast<std::uint64_t>(requirements.field_count));
    payload.append(static_cast<std::uint64_t>(requirements.real_count));
    payload.append(static_cast<std::uint64_t>(requirements.scaled_scalar_count));
    payload.append(static_cast<std::uint64_t>(requirements.collective_value_count));
    payload.append(static_cast<std::uint64_t>(requirements.reduction_value_capacity));
    payload.append(static_cast<std::uint64_t>(requirements.state_word_count));
    payload.append(static_cast<std::uint64_t>(requirements.initial_residual_field));
  }

  static void append_requirements_contract_(ExactContractBuilder& contract,
                                            const KrylovWorkspaceRequirements& requirements) {
    contract.scalar(static_cast<std::uint64_t>(requirements.field_count))
        .scalar(static_cast<std::uint64_t>(requirements.real_count))
        .scalar(static_cast<std::uint64_t>(requirements.scaled_scalar_count))
        .scalar(static_cast<std::uint64_t>(requirements.collective_value_count))
        .scalar(static_cast<std::uint64_t>(requirements.reduction_value_capacity))
        .scalar(static_cast<std::uint64_t>(requirements.state_word_count))
        .scalar(static_cast<std::uint64_t>(requirements.initial_residual_field));
  }

  static void throw_collective_failure_(long failure) {
    if (failure == 0)
      return;
    if (failure == 1)
      throw std::logic_error(
          "operator snapshot mutated after preparation on at least one communicator rank");
    if (failure == 2)
      throw std::logic_error("operator snapshot probe failed on at least one communicator rank");
    if (failure == 3)
      throw std::logic_error(
          "PreparedAffineLinearProblem is not prepared on every communicator rank");
    if (failure == 4)
      throw std::invalid_argument("KrylovWorkspace is incompatible with prepared problem");
    if (failure == 5)
      throw std::invalid_argument("KrylovWorkspace preconditioner footprint mismatch");
    if (failure == 6)
      throw std::logic_error("KrylovWorkspace snapshot is not bound to prepared problem");
    if (failure == 8)
      throw std::invalid_argument(
          "KrylovWorkspace vector layout was rejected by its distribution provider");
    if (failure == 9)
      throw std::invalid_argument("KrylovWorkspace metric is incompatible with prepared problem");
    if (failure == 10)
      throw std::invalid_argument(
          "KrylovWorkspace and prepared problem use non-congruent execution communicators");
    if (failure == 11)
      throw std::runtime_error(
          "KrylovWorkspace communicator comparison failed on at least one communicator rank");
    throw std::invalid_argument("KrylovWorkspace method/restart mismatch");
  }

  // Work storage is deliberately not a public extension seam. Replacing even one iso-layout field
  // could discard its warmed halo/MPI buffers and reintroduce allocation inside an iteration; all
  // algorithms reach these stable slots through the private detail access object instead.
  MultiFab& field(std::size_t index) {
    if (index >= fields_.size())
      throw std::out_of_range("KrylovWorkspace field index");
    return fields_[index];
  }
  const MultiFab& field(std::size_t index) const {
    if (index >= fields_.size())
      throw std::out_of_range("KrylovWorkspace field index");
    return fields_[index];
  }

  Real& real_value(std::size_t index) { return real_values_.at(index); }
  detail::ScaledScalar& scaled_value(std::size_t index) { return scaled_values_.at(index); }
  double* collective_data() { return collective_values_.data(); }
  std::size_t collective_data_size() const { return collective_values_.size(); }
  std::uint64_t& state_word(std::size_t index) { return state_words_.at(index); }
  double* distribution_reduction_data() { return distribution_reduction_data_.data(); }
  std::size_t distribution_reduction_size() const { return distribution_reduction_data_.size(); }
  char* distribution_validation_data() { return distribution_validation_data_.data(); }
  std::size_t distribution_validation_size() const { return distribution_validation_data_.size(); }
  double* metric_reduction_data() { return metric_reduction_data_.data(); }
  std::size_t metric_reduction_size() const { return metric_reduction_data_.size(); }
  std::span<double> gauge_coefficients() { return gauge_coefficients_; }
  PreparedLinearPreconditionerSession& preconditioner_session() { return preconditioner_session_; }
  PreparedAffineOperatorSession& operator_session() { return operator_session_; }
  const ExecutionLane& execution_lane() const { return lane_; }
  const MultiFab& preconditioner_constant() const {
    if (!preconditioner_constant_)
      throw std::logic_error("KrylovWorkspace has no prepared preconditioner constant");
    return *preconditioner_constant_;
  }
  const KrylovWorkspaceRequirements& requirements() const { return requirements_; }
  std::size_t metric_robust_payload_width() const { return metric_.robust_payload_width(); }

  bool provider_report_agrees_(const SolveReport& report) {
    return provider_report_consensus_.agrees(report, lane_);
  }

  bool try_reserve_solve_() noexcept {
    ReservationState expected = ReservationState::Idle;
    return reservation_state_.compare_exchange_strong(
        expected, ReservationState::Solve, std::memory_order_acq_rel, std::memory_order_acquire);
  }
  void release_solve_() noexcept {
    reservation_state_.store(ReservationState::Idle, std::memory_order_release);
  }
  bool try_reserve_mutation_() noexcept {
    ReservationState expected = ReservationState::Idle;
    return reservation_state_.compare_exchange_strong(
        expected, ReservationState::Mutation, std::memory_order_acq_rel, std::memory_order_acquire);
  }
  void release_mutation_() noexcept {
    reservation_state_.store(ReservationState::Idle, std::memory_order_release);
  }

  void reset_provider_apply_status_() noexcept {
    provider_apply_status_ = PreparedApplyStatus::Success;
    operator_session_.reset_apply_status();
    preconditioner_session_.reset_apply_status();
  }
  void latch_provider_apply_status_(PreparedApplyStatus status) noexcept {
    if (!prepared_apply_succeeded(status))
      provider_apply_status_ = PreparedApplyStatus::Failure;
  }
  [[nodiscard]] bool provider_apply_succeeded_() const noexcept {
    return prepared_apply_succeeded(provider_apply_status_);
  }
  void republish_provider_apply_failure_(MultiFab& out) const noexcept {
    if (!provider_apply_succeeded_())
      out.set_val(std::numeric_limits<Real>::quiet_NaN());
  }

  void invalidate_bound_state_() noexcept {
    snapshot_.reset();
    operator_session_ = PreparedAffineOperatorSession{};
    preconditioner_session_ = PreparedLinearPreconditionerSession{};
    operator_source_identity_ = detail::PreparedProviderSourceIdentity{};
    preconditioner_source_identity_ = detail::PreparedProviderSourceIdentity{};
    operator_session_allocation_count_ = 0;
    preconditioner_session_allocation_count_ = 0;
    provider_apply_status_ = PreparedApplyStatus::Success;
  }

  PreparedKrylovMethod method_;
  KrylovFootprint footprint_;
  KrylovWorkspaceRequirements requirements_{};
  OperatorFingerprint layout_{};
  PreparedVectorDistribution vector_distribution_ = PreparedVectorDistribution::Distributed;
  OperatorFingerprint distribution_fingerprint_{};
  PreparedVectorMetric metric_;
  OperatorFingerprint metric_fingerprint_{};
  bool vector_distribution_layout_valid_ = true;
  std::vector<MultiFab> fields_;
  std::size_t allocation_count_ = 0;
  std::optional<OperatorEvaluationSnapshot> snapshot_{};
  std::vector<Real> real_values_;
  std::vector<detail::ScaledScalar> scaled_values_;
  std::vector<double> collective_values_;
  std::vector<double> distribution_reduction_data_;
  std::vector<double> metric_reduction_data_;
  std::vector<char, comm_allocator<char>> distribution_validation_data_;
  std::vector<std::uint64_t> state_words_;
  std::vector<double> gauge_coefficients_;
  ExecutionLane lane_;
  PreparedAffineOperatorSession operator_session_{};
  PreparedLinearPreconditionerSession preconditioner_session_{};
  detail::PreparedProviderSourceIdentity operator_source_identity_{};
  detail::PreparedProviderSourceIdentity preconditioner_source_identity_{};
  std::size_t operator_session_allocation_count_ = 0;
  std::size_t preconditioner_session_allocation_count_ = 0;
  PreparedApplyStatus provider_apply_status_ = PreparedApplyStatus::Success;
  std::optional<MultiFab> preconditioner_constant_{};
  ExactSolveReportConsensusScratch provider_report_consensus_{};
  std::atomic<ReservationState> reservation_state_{ReservationState::Idle};
};

}  // namespace pops
