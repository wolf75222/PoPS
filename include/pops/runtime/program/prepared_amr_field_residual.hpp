#pragma once

#include <pops/numerics/elliptic/interface/amr_field_newton_krylov.hpp>
#include <pops/numerics/elliptic/interface/full_residual_basis_lu.hpp>
#include <pops/runtime/amr/hierarchy_field_operator.hpp>
#include <pops/runtime/program/accepted_exchange.hpp>
#include <pops/runtime/program/prepared_resource_lifetime.hpp>

namespace pops::runtime::program {

/// Explicit native numerical realizations; identity remains the legacy default.
enum class AmrFieldRightPreconditioner {
  kIdentity = 0,
  kSpatialBasisJacobi = 1,
  kFullResidualBasisLU = 2,
};

enum class AmrFieldCoefficientEvaluation { kFrozen = 0, kPerCandidate = 1 };

/// Authority for one original-field invocation. The Program connector supplies these
/// values from its actual synchronized level envelopes, not from a level-zero proxy.
struct AmrFieldResidualAuthority {
  const void* owner = nullptr;
  PreparedResourceAttempt attempt;
  std::vector<PreparedResourceAttempt> level_attempts;
  std::uint64_t topology_epoch = 0, materialization_generation = 0;
  std::string original_equation_identity;
  std::vector<std::string> capture_identities;
  std::vector<int> capture_components;
  std::vector<multiblock::BoundaryEvaluationPoint> points;
};

/// Attempt-owned original F(q)=A_composite(q)+f(q,captures). This is a numerical
/// carrier, not a temporal equation or runtime publication. Candidate access requires
/// a solved original-residual recheck; the Program still owns SolveOutcome consumption
/// and atomic all-level publication. The provider is retained, even after revocation.
template <int Dim>
class PreparedAmrFieldResidual final {
 public:
  using field_type = MultiFab<Dim>;
  using hierarchy_type = std::vector<field_type>;
  using provider_type = PreparedHierarchyTensorSolver<Dim>;
  static constexpr std::string_view identity = "pops.prepared-amr-original-field-residual@1";
  static constexpr std::string_view preconditioned_identity =
      "pops.prepared-amr-original-field-residual@2";
  static constexpr std::string_view candidate_identity = "pops.prepared-amr-original-field-residual@3";
  static constexpr std::string_view full_lu_identity = "pops.prepared-amr-original-field-residual@4";
  std::string_view invocation_identity() const noexcept {
    if (preconditioner_ == AmrFieldRightPreconditioner::kFullResidualBasisLU) return full_lu_identity;
    if (coefficient_evaluation_ == AmrFieldCoefficientEvaluation::kPerCandidate) return candidate_identity;
    return preconditioner_ == AmrFieldRightPreconditioner::kIdentity
               ? identity : preconditioned_identity;
  }

  static std::shared_ptr<PreparedAmrFieldResidual> prepare(
      std::shared_ptr<provider_type> provider, AmrFieldResidualAuthority authority,
      std::span<const hierarchy_type* const> captures, FieldNewtonOptions options,
      Real difference_step, const ExecutionLane& lane,
      AmrFieldRightPreconditioner preconditioner = AmrFieldRightPreconditioner::kIdentity,
      AmrFieldCoefficientEvaluation coefficient_evaluation = AmrFieldCoefficientEvaluation::kFrozen,
      std::uint64_t max_dense_bytes = 0) {
    std::shared_ptr<PreparedAmrFieldResidual> result;
    const auto* known_operator = dynamic_cast<PreparedHierarchyFieldOperator<Dim>*>(provider.get());
    const auto& preparation_lane = known_operator ? known_operator->original_field_execution_lane() : lane;
    local_phase_(preparation_lane, [&] {
      if (&preparation_lane != &lane)
        throw std::invalid_argument("original field preparation received a foreign lane");
      if (!provider || !std::isfinite(difference_step) || !(difference_step > Real(0)))
        throw std::invalid_argument(
            "original AMR field residual requires a provider and positive FD step");
      validate_field_newton_options(options);
      if ((preconditioner == AmrFieldRightPreconditioner::kFullResidualBasisLU) != (max_dense_bytes > 0))
        throw std::invalid_argument("FullResidualBasisLU@1 requires its explicit positive max_dense_bytes budget only");
      if (coefficient_evaluation != AmrFieldCoefficientEvaluation::kFrozen &&
          coefficient_evaluation != AmrFieldCoefficientEvaluation::kPerCandidate)
        throw std::invalid_argument("unknown coefficient evaluation realization");
      if (coefficient_evaluation == AmrFieldCoefficientEvaluation::kPerCandidate &&
          preconditioner == AmrFieldRightPreconditioner::kSpatialBasisJacobi)
        throw std::invalid_argument("SpatialBasisJacobi@1 requires a frozen linear spatial operator");
      if (preconditioner != AmrFieldRightPreconditioner::kIdentity &&
          preconditioner != AmrFieldRightPreconditioner::kSpatialBasisJacobi &&
          preconditioner != AmrFieldRightPreconditioner::kFullResidualBasisLU)
        throw std::invalid_argument("unknown original AMR field right-preconditioner realization");
    });
    auto& op = require_original_field_operator(*provider, lane);
    local_phase_(lane, [&] {
      result.reset(new PreparedAmrFieldResidual(std::move(provider), op, std::move(authority),
                                                captures, options, difference_step, preconditioner, coefficient_evaluation, max_dense_bytes));
    });
    if (coefficient_evaluation == AmrFieldCoefficientEvaluation::kPerCandidate)
      result->coefficient_generation_ = op.original_field_preparation_generation();
    result->require_authority(result->authority_, lane);
    // Provider preparation has its own collective protocol; do not place it inside
    // a local exception wrapper that could skip an internal collective on a peer.
    if (coefficient_evaluation == AmrFieldCoefficientEvaluation::kFrozen) {
      op.prepare_original_field_operator();
    } else {
      PreparedHierarchyCandidateFieldOperator<Dim>* capability = nullptr;
      local_phase_(lane, [&] {
        capability = dynamic_cast<PreparedHierarchyCandidateFieldOperator<Dim>*>(result->provider_.get());
        if (!capability) throw std::invalid_argument("provider has no per-candidate apply capability");
      });
      result->candidate_evaluation_ = capability->make_candidate_evaluation();
      local_phase_(lane, [&] {
        if (!result->candidate_evaluation_ ||
            &result->candidate_evaluation_->original_field_execution_lane() != &lane ||
            result->candidate_evaluation_->original_field_contract() != op.original_field_contract() ||
            result->candidate_evaluation_->original_field_levels() != op.original_field_levels())
          throw std::invalid_argument("candidate evaluation resource has foreign provider authority");
        result->coefficient_fields_.reserve(result->candidate_.size());
        for (int level = 0; level < op.original_field_levels(); ++level)
          result->coefficient_fields_.push_back(&result->candidate_evaluation_->candidate_coefficient_field(level));
        result->authenticate_coefficients_();
      });
    }
    result->coefficient_generation_ = op.original_field_preparation_generation();
    result->require_authority(result->authority_, lane);
    if (preconditioner == AmrFieldRightPreconditioner::kSpatialBasisJacobi)
      result->prepare_spatial_jacobi_(lane);
    return result;
  }

  void require_authority(const AmrFieldResidualAuthority& current,
                         const ExecutionLane& lane) const {
    const auto& authority_lane = op_->original_field_execution_lane();
    std::string contract;
    local_phase_(authority_lane, [&] {
      if (&lane != &op_->original_field_execution_lane() ||
          !current.owner || current.owner != authority_.owner || !authority_.attempt.visible() ||
          !current.attempt.visible() || !authority_.attempt.same_attempt(current.attempt) ||
          current.topology_epoch != authority_.topology_epoch ||
          current.materialization_generation != authority_.materialization_generation ||
          current.original_equation_identity != authority_.original_equation_identity ||
          current.capture_identities != authority_.capture_identities ||
          current.capture_components != authority_.capture_components ||
          current.points != authority_.points ||
          ((coefficient_generation_ != 0 || coefficient_evaluation_ == AmrFieldCoefficientEvaluation::kPerCandidate) &&
           op_->original_field_preparation_generation() != coefficient_generation_))
        throw std::logic_error(
            "original AMR field residual has stale owner/attempt/point/capture authority");
      if (candidate_evaluation_) {
        if (&candidate_evaluation_->original_field_execution_lane() != &authority_lane ||
            candidate_evaluation_->original_field_contract() != op_->original_field_contract() ||
            candidate_evaluation_->original_field_preparation_generation() != evaluation_generation_)
          throw std::logic_error("candidate coefficient resource generation/lane changed");
        authenticate_coefficients_();
      }
      if (current.level_attempts.size() != authority_.level_attempts.size())
        throw std::logic_error("original field level-attempt count changed");
      for (std::size_t level = 0; level < authority_.level_attempts.size(); ++level)
        if (!authority_.level_attempts[level].visible() ||
            !current.level_attempts[level].visible() ||
            !authority_.level_attempts[level].same_attempt(current.level_attempts[level]))
          throw std::logic_error("original field has a stale synchronized level attempt");
      ExactContractBuilder exact;
      exact.text(invocation_identity())
          .text(op_->original_field_contract())
          .text(current.original_equation_identity)
          .scalar(current.topology_epoch)
          .scalar(current.materialization_generation)
          .scalar(current.attempt.ordinal())
          .scalar(op_->original_field_preparation_generation())
          .scalar(step_)
          .scalar(options_.tolerance)
          .scalar(options_.max_iterations)
          .scalar(options_.linear_tolerance)
          .scalar(options_.linear_max_iterations)
          .scalar(options_.restart)
          .scalar(options_.armijo)
          .scalar(options_.minimum_step)
          .sequence(current.capture_identities,
                    [](ExactContractBuilder& out, const std::string& value) { out.text(value); });
      if (options_.convergence.kind != FieldNewtonConvergenceKind::kLegacy)
        exact.text("pops.newton.original-residual-convergence@1")
            .scalar(static_cast<int>(options_.convergence.kind))
            .scalar(options_.convergence.relative).scalar(options_.convergence.absolute);
      if (candidate_evaluation_)
        exact.text(PreparedHierarchyCandidateFieldOperator<Dim>::identity)
            .text("pops.field.linear.true-correction-residual@1")
            .text("restriction-q/evaluate-D/restriction-D/halo-D/composite-flux@1")
            .scalar(evaluation_generation_);
      if (preconditioner_ == AmrFieldRightPreconditioner::kSpatialBasisJacobi)
        exact.text("pops.amr.original-spatial-jacobi.basis-response@1");
      if (preconditioner_ == AmrFieldRightPreconditioner::kFullResidualBasisLU)
        exact.text(FullResidualBasisLU::identity).scalar(max_dense_bytes_)
            .scalar(static_cast<int>(coefficient_evaluation_));
      exact.sequence(current.capture_components,
                     [](ExactContractBuilder& out, int value) { out.scalar(value); });
      for (const auto& attempt : current.level_attempts)
        exact.scalar(attempt.ordinal());
      for (const auto& p : current.points)
        exact.text(p.clock)
            .scalar(p.tick)
            .scalar(p.level)
            .scalar(p.substep)
            .scalar(p.stage)
            .scalar(p.stage_fraction.numerator)
            .scalar(p.stage_fraction.denominator)
            .scalar(p.dt)
            .scalar(p.physical_time)
            .text(p.graph_identity)
            .text(p.rate_identity)
            .text(p.application_identity);
      contract = std::move(exact).release();
    });
    if (!all_ranks_agree_exact_ordered_byte_pairs({{std::string(invocation_identity()), contract}}, authority_lane))
      throw std::logic_error(
          "original AMR field residual invocation differs across execution ranks");
  }

  /// add_local is the original physical body over the entire hierarchy, with frozen
  /// captures. It adds its local terms to the already evaluated composite diffusion.
  /// It may perform local Kokkos work; it must not enter MPI collectives itself.
  template <class LocalBody>
  SolveReport solve(const AmrFieldResidualAuthority& current, const hierarchy_type* seed,
                    LocalBody&& add_local, const ExecutionLane& lane) {
    if (coefficient_evaluation_ != AmrFieldCoefficientEvaluation::kFrozen) {
      require_authority(current, lane);
      local_phase_(op_->original_field_execution_lane(), [&] {
        throw std::logic_error("PerCandidate@1 requires its actual coefficient body");
      });
    }
    return solve_impl_(current, seed, std::forward<LocalBody>(add_local),
                       [](const auto&, const auto&, auto&, int) {},
                       [](const auto&, int) {}, false, lane);
  }
  template <class LocalBody, class CoefficientBody>
  SolveReport solve_candidate(const AmrFieldResidualAuthority& current, const hierarchy_type* seed,
                              LocalBody&& add_local, CoefficientBody&& coefficient_body,
                              const ExecutionLane& lane) {
    require_authority(current, lane);
    local_phase_(op_->original_field_execution_lane(), [&] {
      if (coefficient_evaluation_ != AmrFieldCoefficientEvaluation::kPerCandidate || !candidate_evaluation_)
        throw std::logic_error("candidate coefficient body requires explicit PerCandidate@1");
    });
    return solve_impl_(current, seed, std::forward<LocalBody>(add_local),
                       std::forward<CoefficientBody>(coefficient_body),
                       [](const auto&, int) {}, false, lane);
  }
  /// Exact lease for the producer phase of F. A same-layout accepted/foreign
  /// tower is insufficient, and the lease is revoked before add_local or on refusal.
  void require_evaluation(const hierarchy_type& q, const AmrFieldResidualAuthority& current,
                          const provider_type* expected_provider, std::uint64_t evaluation,
                          const ExecutionLane& lane) const {
    require_authority(current, lane);
    local_phase_(lane, [&] {
      if (expected_provider != provider_.get() || active_evaluation_ != &q ||
          active_evaluation_ordinal_ != evaluation)
        throw std::logic_error("original interaction requires the exact active F candidate lease");
      authenticate_(q);
    });
  }

  /// Collective producer runs on the synchronized simultaneous unknown tower for
  /// every F, including central JVP perturbations and the original recheck. Its
  /// detached outputs may be read by add_local; it must never publish accepted state.
  template <class LocalBody, class Producer>
  SolveReport solve_interaction(const AmrFieldResidualAuthority& current, const hierarchy_type* seed,
                               LocalBody&& add_local, Producer&& producer, const ExecutionLane& lane) {
    require_authority(current, lane);
    local_phase_(lane, [&] {
      if (coefficient_evaluation_ != AmrFieldCoefficientEvaluation::kFrozen)
        throw std::logic_error("interaction with candidate D requires its coefficient body");
      if (evaluation_q_.empty()) {
        std::vector<const field_type*> layouts;
        for (int level = 0; level < op_->original_field_levels(); ++level)
          layouts.push_back(&op_->original_field_layout(level));
        evaluation_q_ = allocate_(layouts);
      }
    });
    return solve_impl_(current, seed, std::forward<LocalBody>(add_local),
                       [](const auto&, const auto&, auto&, int) {},
                       std::forward<Producer>(producer), true, lane);
  }
  template <class LocalBody, class CoefficientBody, class Producer>
  SolveReport solve_candidate_interaction(const AmrFieldResidualAuthority& current, const hierarchy_type* seed,
      LocalBody&& add_local, CoefficientBody&& coefficient_body, Producer&& producer, const ExecutionLane& lane) {
    require_authority(current, lane);
    local_phase_(lane, [&] {
      if (coefficient_evaluation_ != AmrFieldCoefficientEvaluation::kPerCandidate || !candidate_evaluation_)
        throw std::logic_error("candidate interaction requires explicit PerCandidate@1");
    });
    return solve_impl_(current, seed, std::forward<LocalBody>(add_local),
                       std::forward<CoefficientBody>(coefficient_body),
                       std::forward<Producer>(producer), true, lane);
  }
 private:
  template <class LocalBody, class CoefficientBody, class Producer>
  SolveReport solve_impl_(const AmrFieldResidualAuthority& current, const hierarchy_type* seed,
                         LocalBody&& add_local, CoefficientBody&& coefficient_body,
                         Producer&& producer, bool synchronized_interaction, const ExecutionLane& lane) {
    local_phase_(lane, [&] {
      if (active_evaluation_ != nullptr) throw std::logic_error("original solve reentered an active F evaluation");
    });
    candidate_visible_ = false;
    evaluations_ = derivatives_ = 0;
    lu_ready_ = false; lu_columns_ = lu_factorizations_ = 0;
    require_authority(current, lane);
    local_phase_(lane, [&] {
      if (seed) {
        authenticate_(*seed);
        copy_(*seed, candidate_);
      } else
        for (auto& level : candidate_)
          level.set_val(Real(0));
    });
    auto evaluate = [&](const hierarchy_type& q, hierarchy_type& result, int evaluation) {
      require_authority(current, lane);
      const hierarchy_type* physical_q = &q;
      if (candidate_evaluation_) {
        local_phase_(lane, [&] { authenticate_(q); copy_(q, evaluation_q_); });
        // Order is part of @3: restrict/synchronize q first, then evaluate D on
        // every stored cell, then restrict/synchronize D via the actual FAC provider.
        candidate_evaluation_->synchronize_original_field_candidate(evaluation_q_);
        local_phase_(lane, [&] {
          authenticate_coefficients_();
          coefficient_body(std::as_const(evaluation_q_), std::as_const(captures_), coefficient_fields_, evaluation);
          authenticate_coefficients_();
        });
        candidate_evaluation_->prepare_original_field_operator();
        evaluation_generation_ = candidate_evaluation_->original_field_preparation_generation();
        require_authority(current, lane);
        candidate_evaluation_->apply_original_field_operator(evaluation_q_, result);
        physical_q = &evaluation_q_;
      } else if (synchronized_interaction) {
        local_phase_(lane, [&] { authenticate_(q); copy_(q, evaluation_q_); });
        op_->synchronize_original_field_candidate(evaluation_q_);
        op_->apply_original_field_operator(evaluation_q_, result);
        physical_q = &evaluation_q_;
      } else {
        op_->apply_original_field_operator(q, result);
      }
      if (synchronized_interaction) {
        require_authority(current, lane);
        struct EvaluationLease {
          const hierarchy_type*& slot;
          ~EvaluationLease() noexcept { slot = nullptr; }
        } lease{active_evaluation_};
        local_phase_(lane, [&] {
          if (evaluation_sequence_ == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("original interaction F nonce exhausted");
          active_evaluation_ordinal_ = ++evaluation_sequence_;
        });
        active_evaluation_ = physical_q;
        producer(std::as_const(*physical_q), active_evaluation_ordinal_);
        require_authority(current, lane);
      }
      local_phase_(lane, [&] {
        add_local(*physical_q, captures_, result, evaluation);
        // A physical callback may replace its mutable output without throwing.
        // Reject that locally inside this vote, before Newton enters a reduction.
        authenticate_(result);
        Kokkos::fence();
        if (preconditioner_ == AmrFieldRightPreconditioner::kFullResidualBasisLU &&
            evaluations_ == std::numeric_limits<int>::max())
          throw std::overflow_error("FullResidualBasisLU@1 original evaluation count exceeds SolveReport encoding");
      });
      ++evaluations_;
    };
    auto residual = [&](const hierarchy_type& q, hierarchy_type& result, int evaluation) {
      evaluate(q, result, evaluation);
      // The workspace solves J_F * correction = defect and adds correction to
      // q. Only its right-hand side is -F; finite differences and the accepted
      // original-equation recheck continue to evaluate the authored +F.
      local_phase_(lane, [&] {
        for (auto& level : result)
          scale(level, Real(-1));
      });
    };
    auto derivative = [&](const hierarchy_type& q, const hierarchy_type& direction,
                          hierarchy_type& result, int evaluation) {
      const Real squared = op_->original_field_dot(direction, direction);
      const Real state_squared = op_->original_field_dot(q, q);
      const Real magnitude = std::sqrt(squared), state_norm = std::sqrt(state_squared);
      const Real h = step_ * (Real(1) + state_norm) / (magnitude > Real(0) ? magnitude : Real(1));
      local_phase_(lane, [&] {
        if (!std::isfinite(h) || !(h > Real(0)))
          throw std::invalid_argument("original field finite-difference step is nonfinite");
        for (std::size_t level = 0; level < q.size(); ++level)
          lincomb(perturbed_[level], Real(1), q[level], h, direction[level]);
      });
      evaluate(perturbed_, plus_, evaluation);
      local_phase_(lane, [&] {
        for (std::size_t level = 0; level < q.size(); ++level)
          lincomb(perturbed_[level], Real(1), q[level], -h, direction[level]);
      });
      evaluate(perturbed_, minus_, evaluation);
      local_phase_(lane, [&] {
        for (std::size_t level = 0; level < q.size(); ++level)
          lincomb(result[level], Real(0.5) / h, plus_[level], -Real(0.5) / h, minus_[level]);
      });
      if (preconditioner_ == AmrFieldRightPreconditioner::kFullResidualBasisLU)
        local_phase_(lane, [&] {
          if (derivatives_ == std::numeric_limits<int>::max())
            throw std::overflow_error("FullResidualBasisLU@1 derivative count exceeds report encoding");
        });
      ++derivatives_;
    };
    std::vector<field_type*> destinations;
    local_phase_(lane, [&] {
      destinations.reserve(candidate_.size());
      for (auto& level : candidate_)
        destinations.push_back(&level);
    });
    const auto right = [&](const hierarchy_type& input, hierarchy_type& output) {
      require_authority(current, lane);
      if (preconditioner_ == AmrFieldRightPreconditioner::kFullResidualBasisLU) {
        apply_full_lu_(input, output, lane);
        return;
      }
      local_phase_(lane, [&] {
        authenticate_(input);
        authenticate_(output);
        for (std::size_t level = 0; level < input.size(); ++level)
          for (std::size_t patch = 0; patch < input[level].local_size(); ++patch) {
            const auto in = std::as_const(input[level]).fab(patch).view();
            const auto inverse = std::as_const(inverse_diagonal_[level]).fab(patch).view();
            const auto out = output[level].fab(patch).view();
            const int width = input[level].ncomp();
            for_each_cell(input[level].box(patch), [=] POPS_HD(const Index<Dim>& cell) {
              for (int component = 0; component < width; ++component)
                out(cell, component) = in(cell, component) * inverse(cell, component);
            });
          }
      });
    };
    const auto rebuild_right = [&](const hierarchy_type& q, auto& jvp, int evaluation) {
      prepare_full_lu_(current, q, jvp, evaluation, lane);
    };
    auto report = preconditioner_ == AmrFieldRightPreconditioner::kFullResidualBasisLU
        ? newton_->solve_rebuilt_preconditioned(destinations, residual, derivative, [](auto&) {},
                                                right, rebuild_right, lane, true)
        : preconditioner_ == AmrFieldRightPreconditioner::kIdentity
        ? newton_->solve(destinations, residual, derivative, [](auto&) {}, lane, true)
        : newton_->solve_preconditioned(destinations, residual, derivative, [](auto&) {},
                                        right, lane, true);
    if (report.solved_value_available()) {
      op_->synchronize_original_field_candidate(candidate_);
      evaluate(candidate_, recheck_, 0);
      const Real squared = op_->original_field_dot(recheck_, recheck_);
      const Real norm = std::sqrt(squared);
      const Real original_stop =
          collective_field_newton_stop_tolerance(options_, report.reference_residual_norm, lane);
      if (!std::isfinite(norm) || norm > original_stop)
        report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                           "original_amr_field_residual_recheck_failed");
      report.residual_norm = norm;
      report.rel_residual =
          norm /
          (report.reference_residual_norm > Real(0) ? report.reference_residual_norm : Real(1));
    }
    require_authority(current, lane);
    report.evaluations = evaluations_;
    candidate_visible_ = report.solved_value_available();
    return report;
  }
 public:
  const hierarchy_type& candidate(const AmrFieldResidualAuthority& current,
                                  const ExecutionLane& lane) const {
    require_authority(current, lane);
    local_phase_(lane, [&] {
      if (!candidate_visible_ || !authority_.attempt.visible())
        throw std::logic_error(
            "original AMR field candidate has not passed its original residual recheck");
    });
    return candidate_;
  }
  std::size_t full_lu_columns() const noexcept { return lu_columns_; }
  std::size_t full_lu_factorizations() const noexcept { return lu_factorizations_; }
  std::size_t full_lu_active_dofs() const noexcept { return lu_rows_.size(); }
  int residual_evaluations() const noexcept { return evaluations_; }
  int derivative_evaluations() const noexcept { return derivatives_; }
  std::size_t spatial_jacobi_applications() const noexcept { return jacobi_applications_; }

 private:
  struct ActiveDof {
    std::size_t level, global;
    Index<Dim> cell;
    int component;
  };
  bool contributes_(const field_type& field, std::size_t global, const ExecutionLane& lane) const {
    return field.contains_local(global) && (!field.distribution().replicated() || lane.rank() == 0);
  }
  template <class Visitor>
  void visit_active_cells_(const ExecutionLane& lane, Visitor&& visit) const {
    for (std::size_t level = 0; level < candidate_.size(); ++level) {
      const auto& shape = candidate_[level];
      const auto& mask = op_->original_field_active_cells(static_cast<int>(level));
      local_phase_(lane, [&] {
        if (mask.layout() != shape.layout() || mask.distribution() != shape.distribution() ||
            mask.local_rank() != shape.local_rank() || mask.ncomp() != 1)
          throw std::invalid_argument("FullResidualBasisLU@1 active mask has foreign topology/ownership");
      });
      for (std::size_t global = 0; global < shape.layout().size(); ++global) {
        const auto box = shape.layout()[global];
        const auto points = static_cast<std::size_t>(box.numPts());
        for (std::size_t linear = 0; linear < points; ++linear) {
          auto remainder = linear;
          Index<Dim> index{};
          for (int axis = 0; axis < Dim; ++axis) {
            const auto extent = static_cast<std::size_t>(box.length(axis));
            index[axis] = static_cast<int>(static_cast<std::int64_t>(box.lo[axis]) +
                                         static_cast<std::int64_t>(remainder % extent));
            remainder /= extent;
          }
          Real active = 0, owners = 0;
          local_phase_(lane, [&] {
            if (!contributes_(shape, global, lane)) return;
            owners = 1;
            const auto values = mask.fab_global(global).view();
            active = for_each_cell_reduce_max(Box<Dim>(index, index),
                [=] POPS_HD(const Index<Dim>& cell) {
                  const Real value = values(cell, 0);
                  return std::isfinite(value) ? (value >= Real(.5) ? Real(1) : Real(0)) : Real(2);
                });
            if (active == Real(2)) throw std::invalid_argument("FullResidualBasisLU@1 nonfinite active mask");
          });
          owners = all_reduce_sum(owners, lane);
          active = all_reduce_sum(active, lane);
          local_phase_(lane, [&] {
            if (owners != Real(1) || (active != Real(0) && active != Real(1)))
              throw std::invalid_argument("FullResidualBasisLU@1 active quotient has missing/duplicate physical owner");
            if (active == Real(1)) visit(level, global, index, shape.ncomp());
          });
        }
      }
    }
  }
  template <class Jvp>
  void prepare_full_lu_(const AmrFieldResidualAuthority& current, const hierarchy_type& q,
                        Jvp& jvp, int evaluation, const ExecutionLane& lane) {
    require_authority(current, lane);
    local_phase_(lane, [&] { authenticate_(q); lu_ready_ = false; });
    std::size_t count = 0;
    visit_active_cells_(lane, [&](auto, auto, const auto&, int width) {
      count = FullResidualBasisLU::checked_add(count, static_cast<std::size_t>(width));
    });
    local_phase_(lane, [&] {
      std::size_t external = FullResidualBasisLU::checked_product(count, sizeof(ActiveDof));
      for (const auto& field : candidate_)
        for (std::size_t patch = 0; patch < field.local_size(); ++patch) {
          // Three valid-cell towers: active basis, full JVP image and owning iterate snapshot.
          external = FullResidualBasisLU::checked_add(external,
              FullResidualBasisLU::checked_product(field.fab(patch).size(), 3 * sizeof(Real)));
        }
      const auto required = FullResidualBasisLU::required_bytes(count, external);
      if (required > max_dense_bytes_)
        throw std::length_error("FullResidualBasisLU@1 max_dense_bytes resource budget exceeded:required=" +
            std::to_string(required) + ":limit=" + std::to_string(max_dense_bytes_) + ":active_dofs=" + std::to_string(count));
    });
    local_phase_(lane, [&] {
      std::vector<ActiveDof>().swap(lu_rows_); lu_rows_.reserve(count);
      // Release old storage before replacing, avoiding an unaccounted transient second matrix.
      lu_ = FullResidualBasisLU{};
      lu_rhs_.clear(); lu_rhs_.shrink_to_fit(); lu_solution_.clear(); lu_solution_.shrink_to_fit();
      lu_basis_.clear(); lu_image_.clear(); lu_iterate_.clear();
      lu_.allocate(count); lu_rhs_.resize(count); lu_solution_.resize(count);
      std::vector<const field_type*> layouts;
      for (const auto& field : candidate_) layouts.push_back(&field);
      lu_basis_ = allocate_(layouts); lu_image_ = allocate_(layouts); lu_iterate_ = allocate_(layouts);
      copy_(q, lu_iterate_);
    });
    visit_active_cells_(lane, [&](auto level, auto global, const auto& cell, int width) {
      for (int component = 0; component < width; ++component) {
        if (lu_rows_.size() >= count) throw std::logic_error("FullResidualBasisLU@1 active quotient changed during preparation");
        lu_rows_.push_back({level, global, cell, component});
      }
    });
    local_phase_(lane, [&] {
      if (lu_rows_.size() != count) throw std::logic_error("FullResidualBasisLU@1 active quotient changed during preparation");
    });
    for (std::size_t column = 0; column < count; ++column) {
      require_authority(current, lane);
      local_phase_(lane, [&] {
        for (auto& field : lu_basis_) field.set_val(Real(0));
        const auto& dof = lu_rows_[column];
        auto& field = lu_basis_[dof.level];
        if (field.contains_local(dof.global)) {
          const auto values = field.fab_global(dof.global).view();
          const auto component = dof.component;
          for_each_cell(Box<Dim>(dof.cell, dof.cell), [=] POPS_HD(const Index<Dim>& cell) { values(cell, component) = Real(1); });
        }
      });
      // The actual full central F(q +/- h e_j) provider owns its MPI protocol.
      jvp(lu_iterate_, lu_basis_, lu_image_, evaluation);
      gather_active_(lu_image_, lu_rhs_, lane);
      local_phase_(lane, [&] {
        for (std::size_t row = 0; row < count; ++row) lu_.entry(row, column) = lu_rhs_[row];
      });
      ++lu_columns_;
    }
    local_phase_(lane, [&] { lu_.factor(); lu_ready_ = true; ++lu_factorizations_; });
    require_authority(current, lane);
  }
  void gather_active_(const hierarchy_type& input, std::vector<Real>& result,
                      const ExecutionLane& lane) const {
    local_phase_(lane, [&] {
      authenticate_(input);
      if (result.size() != lu_rows_.size()) throw std::logic_error("FullResidualBasisLU@1 active vector shape drift");
      std::fill(result.begin(), result.end(), Real(0));
      for (std::size_t row = 0; row < lu_rows_.size(); ++row) {
        const auto& dof = lu_rows_[row];
        const auto& field = input[dof.level];
        if (!contributes_(field, dof.global, lane)) continue;
        const auto values = field.fab_global(dof.global).view();
        const auto component = dof.component;
        result[row] = for_each_cell_reduce_sum(Box<Dim>(dof.cell, dof.cell),
            [=] POPS_HD(const Index<Dim>& cell) { return values(cell, component); });
        if (!std::isfinite(result[row])) throw std::invalid_argument("FullResidualBasisLU@1 nonfinite active original vector");
      }
    });
    all_reduce_sum_inplace(result.data(), result.size(), lane);
    local_phase_(lane, [&] {
      for (Real value : result) if (!std::isfinite(value)) throw std::invalid_argument("FullResidualBasisLU@1 nonfinite gathered original vector");
    });
  }
  void apply_full_lu_(const hierarchy_type& input, hierarchy_type& output,
                      const ExecutionLane& lane) {
    local_phase_(lane, [&] {
      if (!lu_ready_) throw std::logic_error("FullResidualBasisLU@1 factor is not available at this Newton iterate");
      authenticate_(output);
    });
    // Re-authenticate the exact active quotient before using a retained factor.
    std::size_t row = 0;
    visit_active_cells_(lane, [&](auto level, auto global, const auto& cell, int width) {
      for (int component = 0; component < width; ++component) {
        if (row >= lu_rows_.size()) throw std::logic_error("FullResidualBasisLU@1 active quotient changed");
        const auto& expected = lu_rows_[row++];
        if (expected.level != level || expected.global != global || expected.cell != cell || expected.component != component)
          throw std::logic_error("FullResidualBasisLU@1 active quotient changed");
      }
    });
    local_phase_(lane, [&] {
      if (row != lu_rows_.size()) throw std::logic_error("FullResidualBasisLU@1 active quotient changed");
    });
    gather_active_(input, lu_rhs_, lane);
    local_phase_(lane, [&] {
      lu_.apply(lu_rhs_, lu_solution_);
      for (auto& field : output) field.set_val(Real(0));
      for (std::size_t row = 0; row < lu_rows_.size(); ++row) {
        const auto& dof = lu_rows_[row];
        auto& field = output[dof.level];
        if (!field.contains_local(dof.global)) continue;
        const auto values = field.fab_global(dof.global).view();
        const auto component = dof.component;
        const Real value = lu_solution_[row];
        for_each_cell(Box<Dim>(dof.cell, dof.cell), [=] POPS_HD(const Index<Dim>& cell) { values(cell, component) = value; });
      }
    });
  }
  // Reference realization: one affine zero response and one actual composite
  // operator application per stored DOF. No stencil diagonal is guessed.
  void prepare_spatial_jacobi_(const ExecutionLane& lane) {
    require_authority(authority_, lane);
    local_phase_(lane, [&] {
      for (auto& field : candidate_) field.set_val(Real(0));
      for (auto& field : inverse_diagonal_) field.set_val(Real(0));
      // Validate the complete host traversal before any rank enters its first apply.
      std::size_t applications = 1;
      for (const auto& field : candidate_)
        for (std::size_t global = 0; global < field.layout().size(); ++global) {
          const auto points = static_cast<std::size_t>(field.layout()[global].numPts());
          if (points > (std::numeric_limits<std::size_t>::max() - applications) /
                           static_cast<std::size_t>(field.ncomp()))
            throw std::length_error("spatial-basis Jacobi application count overflows size_t");
          applications += points * static_cast<std::size_t>(field.ncomp());
        }
    });
    op_->apply_original_field_operator(candidate_, minus_);
    jacobi_applications_ = 1;
    for (std::size_t level = 0; level < candidate_.size(); ++level)
      for (std::size_t global = 0; global < candidate_[level].layout().size(); ++global) {
        const auto box = candidate_[level].layout()[global];
        const auto points = static_cast<std::size_t>(box.numPts());
        for (std::size_t linear = 0; linear < points; ++linear) {
          std::size_t remainder = linear;
          Index<Dim> index{};
          for (int axis = 0; axis < Dim; ++axis) {
            const auto extent = static_cast<std::size_t>(box.length(axis));
            index[axis] = static_cast<int>(static_cast<std::int64_t>(box.lo[axis]) +
                                           static_cast<std::int64_t>(remainder % extent));
            remainder /= extent;
          }
          const Box<Dim> singleton(index, index);
          for (int component = 0; component < candidate_[level].ncomp(); ++component) {
            require_authority(authority_, lane);
            local_phase_(lane, [&] {
              for (auto& field : candidate_) field.set_val(Real(0));
              if (candidate_[level].contains_local(global)) {
                const auto values = candidate_[level].fab_global(global).view();
                for_each_cell(singleton, [=] POPS_HD(const Index<Dim>& cell) {
                  values(cell, component) = Real(1);
                });
              }
            });
            op_->apply_original_field_operator(candidate_, plus_);
            ++jacobi_applications_;
            local_phase_(lane, [&] {
              if (!candidate_[level].contains_local(global)) return;
              const auto value = std::as_const(plus_[level]).fab_global(global).view();
              const auto zero = std::as_const(minus_[level]).fab_global(global).view();
              const auto active = op_->original_field_active_cells(static_cast<int>(level))
                                      .fab_global(global).view();
              const auto inverse = inverse_diagonal_[level].fab_global(global).view();
              const Real invalid = for_each_cell_reduce_max(singleton,
                  [=] POPS_HD(const Index<Dim>& cell) {
                    if (active(cell, 0) < Real(.5)) return Real(0);
                    const Real diagonal = value(cell, component) - zero(cell, component);
                    if (!std::isfinite(diagonal) || diagonal == Real(0)) return Real(1);
                    const Real reciprocal = Real(1) / diagonal;
                    if (!std::isfinite(reciprocal)) return Real(1);
                    inverse(cell, component) = reciprocal;
                    return Real(0);
                  });
              if (invalid != Real(0))
                throw std::invalid_argument(
                    "spatial-basis Jacobi requires a finite nonzero actual active spatial diagonal");
            });
          }
        }
      }
    local_phase_(lane, [&] { for (auto& field : candidate_) field.set_val(Real(0)); });
    require_authority(authority_, lane);
  }
  template <class Operation>
  static void local_phase_(const ExecutionLane& lane, Operation&& operation) {
    std::exception_ptr error;
    try {
      operation();
    } catch (...) {
      error = std::current_exception();
    }
    try { Kokkos::fence(); } catch (...) { if (!error) error = std::current_exception(); }
    collectively_rethrow_exception(error, lane, "original AMR field local preparation/evaluation");
  }
  PreparedAmrFieldResidual(std::shared_ptr<provider_type> provider,
                           PreparedHierarchyFieldOperator<Dim>& op,
                           AmrFieldResidualAuthority authority,
                           std::span<const hierarchy_type* const> captures,
                           FieldNewtonOptions options, Real step,
                           AmrFieldRightPreconditioner preconditioner, AmrFieldCoefficientEvaluation coefficient_evaluation,
                           std::uint64_t max_dense_bytes)
      : provider_(std::move(provider)),
        op_(&op),
        authority_(std::move(authority)),
        options_(options),
        step_(step),
        preconditioner_(preconditioner), coefficient_evaluation_(coefficient_evaluation),
        max_dense_bytes_(max_dense_bytes) {
    const int levels = op.original_field_levels();
    if (levels < 1 || authority_.original_equation_identity.empty() ||
        authority_.capture_identities.size() != captures.size() ||
        authority_.capture_components.size() != captures.size() ||
        authority_.points.size() != std::size_t(levels))
      throw std::invalid_argument(
          "original field invocation requires complete equation/capture/level authority");
    if (authority_.level_attempts.size() != std::size_t(levels))
      throw std::invalid_argument("original field requires an attempt lease for every level");
    std::vector<const field_type*> layouts, masks;
    std::vector<Real> measures;
    for (int level = 0; level < levels; ++level) {
      const auto& p = authority_.points[level];
      if (p.stage_fraction.denominator <= 0 || p.stage_fraction.numerator < 0 ||
          p.stage_fraction.numerator > p.stage_fraction.denominator ||
          ::pops::amr::Rational(p.stage_fraction.numerator, p.stage_fraction.denominator) !=
              p.stage_fraction)
        throw std::invalid_argument(
            "original field v1 requires a canonical stage fraction in [0,1]");
      if (p.level != level || p.clock.empty() || p.tick < 0 || p.substep < 0 || p.stage < 0 ||
          !std::isfinite(p.dt) || p.dt <= 0 || !std::isfinite(p.physical_time) ||
          p.graph_identity.empty() || p.rate_identity.empty() || p.application_identity.empty())
        throw std::invalid_argument(
            "original field requires a complete actual point for every level");
      const auto& first = authority_.points.front();
      if (p.clock != first.clock || p.tick != first.tick || p.stage != first.stage ||
          p.stage_fraction != first.stage_fraction || p.dt != first.dt ||
          p.physical_time != first.physical_time)
        throw std::invalid_argument("original field v1 requires a synchronized hierarchy point");
      layouts.push_back(&op.original_field_layout(level));
      masks.push_back(&op.original_field_active_cells(level));
      measures.push_back(op.original_field_cell_measure(level));
    }
    candidate_ = allocate_(layouts);
    if (coefficient_evaluation_ == AmrFieldCoefficientEvaluation::kPerCandidate)
      evaluation_q_ = allocate_(layouts);
    perturbed_ = allocate_(layouts);
    plus_ = allocate_(layouts);
    minus_ = allocate_(layouts);
    recheck_ = allocate_(layouts);
    if (preconditioner_ == AmrFieldRightPreconditioner::kSpatialBasisJacobi)
      inverse_diagonal_ = allocate_(layouts);
    newton_ =
        std::make_unique<AmrFieldNewtonKrylovWorkspace<Dim>>(layouts, masks, measures, options_);
    for (std::size_t capture = 0; capture < captures.size(); ++capture) {
      if (!captures[capture] || authority_.capture_identities[capture].empty())
        throw std::invalid_argument("original field capture has no exact identity/storage");
      const auto& source = *captures[capture];
      authenticate_(source, false);
      std::vector<const field_type*> capture_layouts;
      for (const auto& level : source) {
        if (level.ncomp() != authority_.capture_components[capture])
          throw std::invalid_argument("original field capture component declaration changed");
        capture_layouts.push_back(&level);
      }
      captures_.push_back(allocate_(capture_layouts));
      copy_(source, captures_.back());
    }
  }
  void authenticate_coefficients_() const {
    if (!candidate_evaluation_ || coefficient_fields_.size() != candidate_.size())
      throw std::invalid_argument("candidate coefficient tower count changed");
    for (std::size_t level = 0; level < candidate_.size(); ++level) {
      const auto* field = coefficient_fields_[level];
      if (!field || field != &candidate_evaluation_->candidate_coefficient_field(static_cast<int>(level)) ||
          field->layout() != candidate_[level].layout() ||
          field->distribution() != candidate_[level].distribution() ||
          field->local_rank() != candidate_[level].local_rank() ||
          static_cast<std::uint64_t>(field->ncomp()) != static_cast<std::uint64_t>(candidate_[level].ncomp()) * static_cast<std::uint64_t>(candidate_[level].ncomp()))
        throw std::invalid_argument("candidate coefficient storage/shape authority changed");
    }
  }
  void authenticate_(const hierarchy_type& fields, bool width = true) const {
    if (fields.size() != candidate_.size())
      throw std::invalid_argument("original field hierarchy level count changed");
    for (std::size_t level = 0; level < fields.size(); ++level)
      if (fields[level].layout() != candidate_[level].layout() ||
          fields[level].distribution() != candidate_[level].distribution() ||
          fields[level].local_rank() != candidate_[level].local_rank() ||
          fields[level].ncomp() < 1 ||
          (width && fields[level].ncomp() != candidate_[level].ncomp()))
        throw std::invalid_argument("original field capture/seed exact layout/width changed");
  }
  static hierarchy_type allocate_(std::span<const field_type* const> layouts) {
    hierarchy_type result;
    for (const auto* field : layouts)
      result.emplace_back(field->layout(), field->distribution(), field->local_rank(),
                          field->ncomp(), Extent<Dim>{});
    return result;
  }
  static void copy_(const hierarchy_type& source, hierarchy_type& target) {
    for (std::size_t level = 0; level < source.size(); ++level)
      lincomb(target[level], Real(1), source[level], Real(0), source[level]);
    Kokkos::fence();
  }
  std::shared_ptr<provider_type> provider_;
  PreparedHierarchyFieldOperator<Dim>* op_;
  AmrFieldResidualAuthority authority_;
  FieldNewtonOptions options_;
  Real step_;
  AmrFieldRightPreconditioner preconditioner_;
  AmrFieldCoefficientEvaluation coefficient_evaluation_;
  std::unique_ptr<PreparedHierarchyCandidateEvaluation<Dim>> candidate_evaluation_;
  std::vector<field_type*> coefficient_fields_;
  hierarchy_type evaluation_q_;
  const hierarchy_type* active_evaluation_ = nullptr;
  std::uint64_t active_evaluation_ordinal_ = 0;
  std::uint64_t evaluation_sequence_ = 0;
  std::uint64_t evaluation_generation_ = 0;
  std::unique_ptr<AmrFieldNewtonKrylovWorkspace<Dim>> newton_;
  hierarchy_type candidate_, perturbed_, plus_, minus_, recheck_, inverse_diagonal_;
  std::vector<hierarchy_type> captures_;
  int evaluations_ = 0, derivatives_ = 0;
  bool candidate_visible_ = false;
  std::uint64_t coefficient_generation_ = 0;
  std::size_t jacobi_applications_ = 0;
  std::uint64_t max_dense_bytes_ = 0;
  FullResidualBasisLU lu_;
  std::vector<ActiveDof> lu_rows_;
  std::vector<Real> lu_rhs_, lu_solution_;
  hierarchy_type lu_basis_, lu_image_, lu_iterate_;
  bool lu_ready_ = false;
  std::size_t lu_columns_ = 0, lu_factorizations_ = 0;
};
}  // namespace pops::runtime::program
