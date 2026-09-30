#pragma once

#include <pops/numerics/elliptic/interface/amr_field_newton_krylov.hpp>
#include <pops/runtime/amr/hierarchy_field_operator.hpp>
#include <pops/runtime/program/accepted_exchange.hpp>
#include <pops/runtime/program/prepared_resource_lifetime.hpp>

namespace pops::runtime::program {

/// Explicit native numerical realizations; identity remains the legacy default.
enum class AmrFieldRightPreconditioner {
  kIdentity = 0,
  kSpatialBasisJacobi = 1,
};

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
  std::string_view invocation_identity() const noexcept {
    return preconditioner_ == AmrFieldRightPreconditioner::kIdentity
               ? identity : preconditioned_identity;
  }

  static std::shared_ptr<PreparedAmrFieldResidual> prepare(
      std::shared_ptr<provider_type> provider, AmrFieldResidualAuthority authority,
      std::span<const hierarchy_type* const> captures, FieldNewtonOptions options,
      Real difference_step, const ExecutionLane& lane,
      AmrFieldRightPreconditioner preconditioner = AmrFieldRightPreconditioner::kIdentity) {
    std::shared_ptr<PreparedAmrFieldResidual> result;
    local_phase_(lane, [&] {
      if (!provider || !std::isfinite(difference_step) || !(difference_step > Real(0)))
        throw std::invalid_argument(
            "original AMR field residual requires a provider and positive FD step");
      validate_field_newton_options(options);
      if (preconditioner != AmrFieldRightPreconditioner::kIdentity &&
          preconditioner != AmrFieldRightPreconditioner::kSpatialBasisJacobi)
        throw std::invalid_argument("unknown original AMR field right-preconditioner realization");
    });
    auto& op = require_original_field_operator(*provider, lane);
    local_phase_(lane, [&] {
      result.reset(new PreparedAmrFieldResidual(std::move(provider), op, std::move(authority),
                                                captures, options, difference_step, preconditioner));
    });
    result->require_authority(result->authority_, lane);
    // Provider preparation has its own collective protocol; do not place it inside
    // a local exception wrapper that could skip an internal collective on a peer.
    op.prepare_original_field_operator();
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
          (coefficient_generation_ != 0 &&
           op_->original_field_preparation_generation() != coefficient_generation_))
        throw std::logic_error(
            "original AMR field residual has stale owner/attempt/point/capture authority");
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
      if (preconditioner_ != AmrFieldRightPreconditioner::kIdentity)
        exact.text("pops.amr.original-spatial-jacobi.basis-response@1");
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
    candidate_visible_ = false;
    evaluations_ = derivatives_ = 0;
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
      op_->apply_original_field_operator(q, result);
      local_phase_(lane, [&] {
        add_local(q, captures_, result, evaluation);
        // A physical callback may replace its mutable output without throwing.
        // Reject that locally inside this vote, before Newton enters a reduction.
        authenticate_(result);
        Kokkos::fence();
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
    auto report = preconditioner_ == AmrFieldRightPreconditioner::kIdentity
        ? newton_->solve(destinations, residual, derivative, [](auto&) {}, lane, true)
        : newton_->solve_preconditioned(destinations, residual, derivative, [](auto&) {},
                                        right, lane, true);
    if (report.solved_value_available()) {
      op_->synchronize_original_field_candidate(candidate_);
      evaluate(candidate_, recheck_, 0);
      const Real squared = op_->original_field_dot(recheck_, recheck_);
      const Real norm = std::sqrt(squared);
      if (!std::isfinite(norm) ||
          norm > options_.tolerance * std::max(Real(1), report.reference_residual_norm))
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
  int residual_evaluations() const noexcept { return evaluations_; }
  int derivative_evaluations() const noexcept { return derivatives_; }
  std::size_t spatial_jacobi_applications() const noexcept { return jacobi_applications_; }

 private:
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
      Kokkos::fence();
    } catch (...) {
      error = std::current_exception();
    }
    collectively_rethrow_exception(error, lane, "original AMR field local preparation/evaluation");
  }
  PreparedAmrFieldResidual(std::shared_ptr<provider_type> provider,
                           PreparedHierarchyFieldOperator<Dim>& op,
                           AmrFieldResidualAuthority authority,
                           std::span<const hierarchy_type* const> captures,
                           FieldNewtonOptions options, Real step,
                           AmrFieldRightPreconditioner preconditioner)
      : provider_(std::move(provider)),
        op_(&op),
        authority_(std::move(authority)),
        options_(options),
        step_(step),
        preconditioner_(preconditioner) {
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
  std::unique_ptr<AmrFieldNewtonKrylovWorkspace<Dim>> newton_;
  hierarchy_type candidate_, perturbed_, plus_, minus_, recheck_, inverse_diagonal_;
  std::vector<hierarchy_type> captures_;
  int evaluations_ = 0, derivatives_ = 0;
  bool candidate_visible_ = false;
  std::uint64_t coefficient_generation_ = 0;
  std::size_t jacobi_applications_ = 0;
};
}  // namespace pops::runtime::program
