#pragma once

/// @file
/// @brief Allocation-free Krylov algorithms over a prepared affine problem.
///
/// This is deliberately the only public generic Krylov entry point.  Callers must prepare and
/// authenticate the operator evaluation, bind persistent workspace, and pass typed controls.  The
/// algorithms therefore never guess that an affine A is linear, lazily build a preconditioner,
/// allocate scratch in an iteration, or publish an Arnoldi/preconditioned residual as scientific
/// convergence.

#include <pops/numerics/elliptic/linear/krylov_workspace.hpp>
#include <pops/numerics/elliptic/linear/scaled_field_algebra.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>

namespace pops {
namespace detail {

/// The algorithms are the sole consumers of persistent workspace storage. Keeping this access
/// object private to detail prevents callers from replacing warmed fields or scalar buffers while
/// still letting every Krylov method share the same allocation-free storage.
struct KrylovWorkspaceAccess {
  static MultiFab& field(KrylovWorkspace& workspace, std::size_t index) {
    return workspace.field(index);
  }
  static Real& h(KrylovWorkspace& workspace, int row, int column, int basis_extent) {
    const std::size_t extent = static_cast<std::size_t>(basis_extent);
    return workspace.real_value(static_cast<std::size_t>(row) * extent +
                                static_cast<std::size_t>(column));
  }
  static Real& cosine(KrylovWorkspace& workspace, int index, int basis_extent) {
    const std::size_t extent = static_cast<std::size_t>(basis_extent);
    return workspace.real_value((extent + 1u) * extent + static_cast<std::size_t>(index));
  }
  static Real& sine(KrylovWorkspace& workspace, int index, int basis_extent) {
    const std::size_t extent = static_cast<std::size_t>(basis_extent);
    return workspace.real_value((extent + 1u) * extent + extent +
                                static_cast<std::size_t>(index));
  }
  static Real& rotated_rhs(KrylovWorkspace& workspace, int index, int basis_extent) {
    const std::size_t extent = static_cast<std::size_t>(basis_extent);
    return workspace.real_value((extent + 1u) * extent + 2u * extent +
                                static_cast<std::size_t>(index));
  }
  static Real& solution_coefficient(KrylovWorkspace& workspace, int index, int basis_extent) {
    const std::size_t extent = static_cast<std::size_t>(basis_extent);
    return workspace.real_value((extent + 1u) * extent + 3u * extent + 1u +
                                static_cast<std::size_t>(index));
  }
  static ScaledScalar& scaled_h(KrylovWorkspace& workspace, int row, int column,
                                int basis_extent) {
    const std::size_t extent = static_cast<std::size_t>(basis_extent);
    return workspace.scaled_value(static_cast<std::size_t>(row) * extent +
                                  static_cast<std::size_t>(column));
  }
  static ScaledScalar& scaled_rotated_rhs(KrylovWorkspace& workspace, int index,
                                          int basis_extent) {
    const std::size_t extent = static_cast<std::size_t>(basis_extent);
    return workspace.scaled_value((extent + 1u) * extent + static_cast<std::size_t>(index));
  }
  static ScaledScalar& scaled_solution_coefficient(KrylovWorkspace& workspace, int index,
                                                   int basis_extent) {
    const std::size_t extent = static_cast<std::size_t>(basis_extent);
    return workspace.scaled_value((extent + 1u) * extent + extent + 1u +
                                  static_cast<std::size_t>(index));
  }
  static double* gmres_reduction_data(KrylovWorkspace& workspace) {
    return workspace.collective_data();
  }
  static double* gmres_robust_reduction_data(KrylovWorkspace& workspace, int basis_extent) {
    return workspace.collective_data() + static_cast<std::size_t>(basis_extent) + 1u;
  }
  static double* distribution_reduction_data(KrylovWorkspace& workspace) {
    return workspace.distribution_reduction_data();
  }
  static std::size_t distribution_reduction_size(const KrylovWorkspace& workspace) {
    return workspace.distribution_reduction_size();
  }
  static char* distribution_validation_data(KrylovWorkspace& workspace) {
    return workspace.distribution_validation_data();
  }
  static std::size_t distribution_validation_size(const KrylovWorkspace& workspace) {
    return workspace.distribution_validation_size();
  }
  static std::size_t gmres_reduction_size(const KrylovWorkspace& workspace) {
    return workspace.collective_data_size();
  }
  static std::size_t metric_robust_payload_width(const KrylovWorkspace& workspace) {
    return workspace.metric_robust_payload_width();
  }
  static std::size_t initial_residual_field(const KrylovWorkspace& workspace) {
    return workspace.requirements_.initial_residual_field;
  }
  static bool provider_report_reason_agrees(KrylovWorkspace& workspace,
                                             std::string_view reason) {
    return workspace.provider_report_reason_agrees_(reason);
  }
  static void append_collective_state(const KrylovWorkspace& workspace,
                                      KrylovCollectivePayload& payload) noexcept {
    workspace.append_collective_state_(payload);
  }
  static long local_binding_failure(const KrylovWorkspace& workspace,
                                    const PreparedAffineLinearProblem& problem,
                                    const KrylovControls& controls) noexcept {
    const auto& problem_snapshot = PreparedProblemAccess::stored_snapshot(problem);
    if (!workspace.snapshot_ || !problem_snapshot || *workspace.snapshot_ != *problem_snapshot)
      return 23;
    if (!workspace.vector_distribution_layout_valid_ ||
        workspace.vector_distribution_ != problem.vector_distribution() ||
        workspace.layout_ != problem.layout_fingerprint() ||
        workspace.footprint_ != problem.footprint() ||
        workspace.footprint_.preconditioned != problem.has_preconditioner())
      return 24;
    if (!(workspace.method_ == controls.method))
      return 25;
    return 0;
  }
};

inline bool finite(Real value) {
  return std::isfinite(static_cast<double>(value));
}

inline void reduce_batched_inner_products(const PreparedAffineLinearProblem& problem,
                                          KrylovWorkspace& workspace, double* values, int count,
                                          const char* quantity) {
  reduce_prepared_vector_values_inplace(problem.vector_distribution(), values, count,
                                        KrylovWorkspaceAccess::distribution_reduction_data(workspace),
                                        KrylovWorkspaceAccess::distribution_reduction_size(workspace),
                                        quantity);
}

template <class RightAt>
inline bool repair_nonfinite_batched_inner_products(const PreparedAffineLinearProblem& problem,
                                                    KrylovWorkspace& workspace,
                                                    const MultiFab& left, double* reduced,
                                                    int count, int basis_extent,
                                                    RightAt&& right_at) {
  bool needs_repair = false;
  for (int index = 0; index < count; ++index)
    needs_repair = needs_repair || !finite(static_cast<Real>(reduced[index]));
  if (!needs_repair)
    return true;

  const std::size_t width = KrylovWorkspaceAccess::metric_robust_payload_width(workspace);
  double* payload =
      KrylovWorkspaceAccess::gmres_robust_reduction_data(workspace, basis_extent);
  const std::size_t payload_count = static_cast<std::size_t>(count) * width;
  std::fill_n(payload, payload_count, 0.0);
  for (int index = 0; index < count; ++index) {
    if (finite(static_cast<Real>(reduced[index])))
      continue;
    PreparedProblemAccess::local_robust_inner_product_payload(
        problem, left, right_at(index),
        std::span<double>(payload + static_cast<std::size_t>(index) * width, width));
  }
  reduce_batched_inner_products(problem, workspace, payload, static_cast<int>(payload_count),
                                "prepared GMRES robust projections");

  bool finite_result = true;
  for (int index = 0; index < count; ++index) {
    if (finite(static_cast<Real>(reduced[index])))
      continue;
    reduced[index] =
        static_cast<double>(PreparedProblemAccess::inner_product_from_global_robust_payload(
            problem,
            std::span<const double>(payload + static_cast<std::size_t>(index) * width, width)));
    finite_result = finite_result && finite(static_cast<Real>(reduced[index]));
  }
  return finite_result;
}

/// Preserve native `Real` rounding whenever the operation is representable, and retain a
/// binary-scaled result only for the exceptional exponent range. This lets the overflow path be
/// added without perturbing mature Krylov trajectories at ordinary scales.
inline ScaledScalar scaled_product(Real left, Real right) {
  const Real product = left * right;
  if (finite(product))
    return ScaledScalar::from(product);
  return ScaledScalar::product(ScaledScalar::from(left), ScaledScalar::from(right));
}

inline ScaledScalar scaled_quotient(Real numerator, Real denominator) {
  const Real quotient = numerator / denominator;
  if (finite(quotient))
    return ScaledScalar::from(quotient);
  return ScaledScalar::quotient(ScaledScalar::from(numerator), ScaledScalar::from(denominator));
}

inline ScaledScalar scaled_product(const ScaledScalar& left, const ScaledScalar& right) {
  Real left_value = Real(0);
  Real right_value = Real(0);
  if (left.try_materialize(left_value) && right.try_materialize(right_value))
    return scaled_product(left_value, right_value);
  return ScaledScalar::product(left, right);
}

inline ScaledScalar scaled_quotient(const ScaledScalar& numerator,
                                    const ScaledScalar& denominator) {
  Real numerator_value = Real(0);
  Real denominator_value = Real(0);
  if (numerator.try_materialize(numerator_value) && denominator.try_materialize(denominator_value))
    return scaled_quotient(numerator_value, denominator_value);
  return ScaledScalar::quotient(numerator, denominator);
}

inline ScaledScalar scaled_sum(const ScaledScalar& left, const ScaledScalar& right) {
  Real left_value = Real(0);
  Real right_value = Real(0);
  if (left.try_materialize(left_value) && right.try_materialize(right_value)) {
    const Real sum = left_value + right_value;
    if (finite(sum))
      return ScaledScalar::from(sum);
  }
  return ScaledScalar::sum(left, right);
}

inline ScaledScalar scaled_difference(const ScaledScalar& left, const ScaledScalar& right) {
  Real left_value = Real(0);
  Real right_value = Real(0);
  if (left.try_materialize(left_value) && right.try_materialize(right_value)) {
    const Real difference = left_value - right_value;
    if (finite(difference))
      return ScaledScalar::from(difference);
  }
  return ScaledScalar::difference(left, right);
}

inline void validate_controls(const KrylovControls& controls) {
  const KrylovMethodValidation validation = controls.method.validate_controls(
      KrylovMethodControls{controls.rel_tol, controls.abs_tol, controls.max_iterations});
  if (!validation.accepted())
    throw std::invalid_argument("prepared Krylov provider '" + std::string(controls.method.identity()) +
                                "' rejected controls: " + std::string(validation.reason));
}

inline long controls_failure(const KrylovControls& controls) noexcept {
  if (!controls.method)
    return 19;
  return controls.method
                 .validate_controls(KrylovMethodControls{
                     controls.rel_tol, controls.abs_tol, controls.max_iterations})
                 .accepted()
             ? 0
             : 28;
}

[[noreturn]] inline void throw_solve_preflight_failure(long failure) {
  switch (failure) {
    case 1:
      throw std::logic_error(
          "operator snapshot mutated after preparation on at least one communicator rank");
    case 2:
      throw std::logic_error("operator snapshot probe failed on at least one communicator rank");
    case 3:
      throw std::logic_error(
          "PreparedAffineLinearProblem is not prepared on every communicator rank");
    case 19:
      throw std::invalid_argument("prepared Krylov method provider is empty");
    case 20:
      throw std::invalid_argument(
          "solve_prepared_affine(iterate): incompatible vector space or ghost footprint");
    case 21:
      throw std::invalid_argument("solve_prepared_affine(rhs): incompatible vector space");
    case 22:
      throw std::invalid_argument(
          "solve_prepared_affine requires iterate and rhs to use distinct storage");
    case 23:
      throw std::logic_error("KrylovWorkspace snapshot is not bound to prepared problem");
    case 24:
      throw std::invalid_argument("KrylovWorkspace is incompatible with prepared problem");
    case 25:
      throw std::invalid_argument("KrylovWorkspace method/restart mismatch");
    case 28:
      throw std::invalid_argument("prepared Krylov method provider rejected the solve request");
    default:
      throw std::logic_error("prepared Krylov collective preflight failed");
  }
}

inline void append_controls(KrylovCollectivePayload& payload,
                            const KrylovControls& controls) noexcept {
  payload.append(controls.method.fingerprint());
  payload.append(std::bit_cast<std::uint64_t>(controls.rel_tol));
  payload.append(std::bit_cast<std::uint64_t>(controls.abs_tol));
  payload.append(controls.max_iterations);
}

inline void append_field_shape(KrylovCollectivePayload& payload,
                               const MultiFab& field) noexcept {
  payload.append(field.ncomp());
  payload.append(field.n_grow());
}

inline void collective_solve_preflight(const PreparedAffineLinearProblem& problem,
                                       KrylovWorkspace& workspace, const MultiFab& iterate,
                                       const MultiFab& rhs, const KrylovControls& controls) {
  KrylovCollectivePayload payload;
  long local_failure = PreparedProblemAccess::append_collective_state(problem, payload);
  KrylovWorkspaceAccess::append_collective_state(workspace, payload);
  append_controls(payload, controls);
  append_field_shape(payload, iterate);
  append_field_shape(payload, rhs);
  payload.append(static_cast<std::uint8_t>(iterate.shares_storage_with(rhs)));

  if (local_failure == 0)
    local_failure = controls_failure(controls);
  if (local_failure == 0 && (!PreparedProblemAccess::matches_vector_space(problem, iterate) ||
                             iterate.n_grow() != problem.footprint().input_ghosts))
    local_failure = 20;
  if (local_failure == 0 && !PreparedProblemAccess::matches_vector_space(problem, rhs))
    local_failure = 21;
  if (local_failure == 0 && iterate.shares_storage_with(rhs))
    local_failure = 22;
  if (local_failure == 0)
    local_failure = KrylovWorkspaceAccess::local_binding_failure(workspace, problem, controls);
  const KrylovMethodProblemFacts method_facts{
      problem.properties(), problem.footprint(), problem.vector_distribution(),
      problem.metric().robust_payload_width(), problem.has_nullspace(),
      problem.has_preconditioner()};
  const KrylovMethodValidation problem_validation = controls.method.validate_problem(method_facts);
  payload.append(problem_validation.code);
  if (local_failure == 0 && !problem_validation.accepted())
    local_failure = 28;

  // This is the only solve-entry collective gate. It is fixed-size, stack-only and precedes every
  // norm, nullspace, halo or Krylov reduction. Exact min/max consensus catches valid-but-different
  // contracts; the error reduction converts a rank-local invalid contract into one exception.
  const bool agrees = collective_payload_agrees(payload);
  const long collective_failure = all_reduce_max(local_failure);
  if (collective_failure == 28) {
    // A provider diagnostic is meaningful only after the complete request is known to be identical
    // on every rank. Otherwise one rank could publish a local reason while another publishes the
    // generic remote-failure text, splitting the public exception contract.
    if (!agrees)
      throw std::logic_error(
          "prepared Krylov collective contract differs across communicator ranks");
    const KrylovMethodValidation control_validation = controls.method.validate_controls(
        KrylovMethodControls{controls.rel_tol, controls.abs_tol, controls.max_iterations});
    const KrylovMethodValidation local_validation =
        control_validation.accepted() ? problem_validation : control_validation;
    if (!local_validation.accepted())
      throw std::invalid_argument("prepared Krylov provider '" +
                                  std::string(controls.method.identity()) +
                                  "' rejected the solve request: " +
                                  std::string(local_validation.reason));
    throw std::invalid_argument(
        "prepared Krylov method provider rejected the solve request on another MPI rank");
  }
  if (collective_failure != 0)
    throw_solve_preflight_failure(collective_failure);
  if (!agrees)
    throw std::logic_error("prepared Krylov collective contract differs across communicator ranks");

  // The problem and workspace authenticated their exact vector-space layout during prepare/bind.
  // At solve entry, allocation-free local vector-space checks above bind both arguments to that
  // cached contract.  Only value consensus remains dynamic for a provider (for example replicas).
  const PreparedVectorDistribution& distribution = problem.vector_distribution();
  char* storage = KrylovWorkspaceAccess::distribution_validation_data(workspace);
  const std::size_t storage_size =
      KrylovWorkspaceAccess::distribution_validation_size(workspace);
  distribution.require_exact_values(iterate, std::span<char>(storage, storage_size),
                                    "solve_prepared_affine(iterate)");
  distribution.require_exact_values(rhs, std::span<char>(storage, storage_size),
                                    "solve_prepared_affine(rhs)");
}

inline Real reference_denominator(Real reference) {
  return reference > Real(0) ? reference : Real(1);
}

inline bool provider_solve_report_agrees(const SolveReport& report,
                                         KrylovWorkspace& workspace) {
  KrylovCollectivePayload payload;
  payload.append(report.iters);
  payload.append(std::bit_cast<std::uint64_t>(report.rel_residual));
  payload.append(std::bit_cast<std::uint64_t>(report.reference_residual_norm));
  payload.append(std::bit_cast<std::uint64_t>(report.residual_norm));
  payload.append(report.status);
  payload.append(report.action);
  const bool fixed_contract_agrees = collective_payload_agrees(payload);
  const bool reason_agrees =
      KrylovWorkspaceAccess::provider_report_reason_agrees(workspace, report.reason);
  return fixed_contract_agrees && reason_agrees;
}

struct SolveNormalization {
  Real reference = Real(0);
  Real scale = Real(1);
  Real normalized_threshold = Real(0);
  Real physical_threshold = Real(0);
};

inline SolveNormalization make_normalization(Real reference, Real scale,
                                             const KrylovControls& controls) {
  if (!finite(reference) || reference < Real(0) || !finite(scale) || !(scale > Real(0)))
    throw std::invalid_argument("invalid prepared Krylov equation normalization");
  const Real relative_physical =
      reference > Real(0) ? rescale_product(controls.rel_tol, reference, Real(1)) : Real(0);
  const Real physical_threshold = std::max(relative_physical, controls.abs_tol);
  return {
      reference,
      scale,
      physical_threshold / scale,
      physical_threshold,
  };
}

inline Real physical_stopping_threshold(Real reference, const KrylovControls& controls) {
  const Real relative =
      reference > Real(0) ? rescale_product(controls.rel_tol, reference, Real(1)) : Real(0);
  return std::max(relative, controls.abs_tol);
}

template <class Report>
inline void set_report_physical_residuals(Report& report, const SolveNormalization& normalization,
                                          Real physical_residual) {
  report.rel_residual = physical_residual / reference_denominator(normalization.reference);
  if constexpr (requires { report.reference_residual_norm; })
    report.reference_residual_norm = normalization.reference;
  if constexpr (requires { report.residual_norm; })
    report.residual_norm = physical_residual;
}

inline SolveReport report_physical(const SolveNormalization& normalization, Real physical_residual,
                                   int iterations, SolveStatus status) {
  SolveReport result;
  result.iters = iterations;
  set_report_physical_residuals(result, normalization, physical_residual);
  if (status == SolveStatus::kSolved)
    result.mark_solved();
  else
    result.mark_failed(status);
  return result;
}

inline Real physical_true_residual_norm(const PreparedAffineLinearProblem& problem,
                                        MultiFab& scratch, const MultiFab& rhs,
                                        const MultiFab& iterate) {
  PreparedProblemAccess::true_residual_physical(problem, scratch, rhs, iterate);
  return PreparedProblemAccess::residual_norm(problem, scratch);
}

struct ResidualMeasurement {
  Real physical = Real(0);
  Real normalized = Real(0);
};

/// Materialize the scientific residual in physical units and measure it scale-safely.  The field is
/// deliberately left in physical units: a caller that is actually going to restart a recurrence
/// can then choose its next cycle scale from this authoritative measurement, without first losing a
/// representable component through division by the old cycle scale.
inline ResidualMeasurement physical_true_residual_measurement(
    const PreparedAffineLinearProblem& problem, MultiFab& scratch, const MultiFab& rhs,
    const MultiFab& iterate) {
  const Real physical = physical_true_residual_norm(problem, scratch, rhs, iterate);
  return {physical, std::numeric_limits<Real>::quiet_NaN()};
}

/// Rebase only the disposable recurrence cycle.  The authored reference and physical stopping
/// threshold remain those in `report_normalization`; reports therefore cannot change meaning when
/// an extreme residual forces a numerical restart.  This operation consumes a residual field that
/// is still in physical units and needs no additional reduction.
inline void rebase_cycle_residual(MultiFab& physical_residual, ResidualMeasurement& measurement,
                                  const SolveNormalization& report_normalization,
                                  SolveNormalization& cycle_normalization) {
  if (!finite(measurement.physical) ||
      !(measurement.physical > report_normalization.physical_threshold))
    throw std::logic_error(
        "prepared Krylov recurrence rebase requires a finite unconverged true residual");
  cycle_normalization = {
      report_normalization.reference,
      measurement.physical,
      report_normalization.physical_threshold / measurement.physical,
      report_normalization.physical_threshold,
  };
  PreparedFieldAlgebra::divide(physical_residual, cycle_normalization.scale);
  measurement.normalized = Real(1);
}

inline bool needs_extreme_recurrence_rebase(Real normalized_norm) {
  return normalized_norm > Real(0) && normalized_norm < std::sqrt(std::numeric_limits<Real>::min());
}

/// Publish a terminal recurrence outcome without repeating the public wrapper's mandatory
/// provider-independent true-residual evaluation. `measurement` is the last authoritative
/// physical residual observed inside the method and keeps the candidate structurally valid; the
/// wrapper overwrites all residual fields and promotes/rejects convergence after its own matvec.
inline SolveReport terminal_candidate_report(const SolveNormalization& normalization,
                                             const ResidualMeasurement& measurement,
                                             int iterations, SolveStatus status) {
  return report_physical(normalization, measurement.physical, iterations, status);
}

/// Remove one arbitrary scalar normalization from a prepared linear preconditioner.  Krylov
/// methods are invariant to M -> cM, but their raw dot products are not representable for finite
/// c=1e+/-200.  The first nonzero preconditioned direction fixes one solve-local positive scale;
/// every later application reuses it, so the mathematical preconditioner changes only by a single
/// constant factor and no allocation or per-iteration norm reduction is introduced.
inline Real apply_scaled_preconditioner(const PreparedAffineLinearProblem& problem, MultiFab& out,
                                        const MultiFab& in, KrylovWorkspace& workspace,
                                        Real& solve_scale) {
  PreparedProblemAccess::apply_preconditioner(problem, out, in);
  if (solve_scale == Real(0))
    solve_scale = PreparedFieldAlgebra::max_abs(
        out, problem.vector_distribution(),
        std::span<double>(KrylovWorkspaceAccess::distribution_reduction_data(workspace),
                          KrylovWorkspaceAccess::distribution_reduction_size(workspace)));
  if (!finite(solve_scale) || !(solve_scale > Real(0)))
    return solve_scale;
  PreparedFieldAlgebra::divide(out, solve_scale);
  return solve_scale;
}

inline MultiFab& initial_residual_field(KrylovWorkspace& workspace,
                                        const KrylovControls&) {
  return KrylovWorkspaceAccess::field(workspace,
                                      KrylovWorkspaceAccess::initial_residual_field(workspace));
}

inline SolveReport solve_richardson(const PreparedAffineLinearProblem& problem,
                                    KrylovWorkspace& workspace, MultiFab& iterate,
                                    const MultiFab& rhs, const KrylovControls& controls,
                                    Real relaxation,
                                    const SolveNormalization& normalization,
                                    ResidualMeasurement measurement) {
  MultiFab& residual = KrylovWorkspaceAccess::field(workspace, 1);
  SolveNormalization cycle_normalization = normalization;
  for (int completed = 0; completed < controls.max_iterations; ++completed) {
    const int iteration = completed + 1;
    // Preserve the public physical-equation method x <- x + omega*(b-A(x)). `residual` is divided
    // by the solve-local equation scale.  Their product may exceed `Real` even when every final
    // cell update is finite, so keep it binary-scaled through the fused field operation.
    const ScaledScalar normalized_relaxation =
        scaled_product(relaxation, cycle_normalization.scale);
    if (!normalized_relaxation.is_finite())
      return report_physical(normalization, measurement.physical, iteration - 1,
                             SolveStatus::kInvalidEvaluation);
    ScaledFieldAlgebra::axpy(iterate, normalized_relaxation, residual);
    if (iteration == controls.max_iterations)
      return terminal_candidate_report(normalization, measurement, iteration,
                                       SolveStatus::kIterationLimit);
    measurement = physical_true_residual_measurement(problem, residual, rhs, iterate);
    if (!finite(measurement.physical))
      return report_physical(normalization, measurement.physical, iteration,
                             SolveStatus::kInvalidEvaluation);
    if (measurement.physical <= normalization.physical_threshold)
      return report_physical(normalization, measurement.physical, iteration, SolveStatus::kSolved);
    if (iteration < controls.max_iterations)
      rebase_cycle_residual(residual, measurement, normalization, cycle_normalization);
  }
  return report_physical(normalization, measurement.physical, controls.max_iterations,
                         SolveStatus::kIterationLimit);
}

inline SolveReport solve_cg(const PreparedAffineLinearProblem& problem, KrylovWorkspace& workspace,
                            MultiFab& iterate, const MultiFab& rhs, const KrylovControls& controls,
                            const SolveNormalization& normalization,
                            ResidualMeasurement measurement) {
  MultiFab& residual = KrylovWorkspaceAccess::field(workspace, 1);
  MultiFab& direction = KrylovWorkspaceAccess::field(workspace, 2);
  MultiFab& applied = KrylovWorkspaceAccess::field(workspace, 3);
  SolveNormalization cycle_normalization = normalization;
  PreparedFieldAlgebra::copy(direction, residual);
  Real squared = PreparedProblemAccess::inner_product(problem, residual, residual);
  for (int completed = 0; completed < controls.max_iterations; ++completed) {
    const int iteration = completed + 1;
    PreparedProblemAccess::apply_linear(problem, applied, direction, cycle_normalization.scale);
    const Real curvature = PreparedProblemAccess::inner_product(problem, direction, applied);
    if (!finite(curvature) || !finite(squared))
      return terminal_candidate_report(normalization, measurement, iteration - 1,
                                       SolveStatus::kInvalidEvaluation);
    // A certified SPD operator has strictly positive curvature.  Refuse the mathematical loss of
    // definiteness, not a dimensioned absolute epsilon that would reject a valid rescaled system.
    if (curvature <= Real(0))
      return terminal_candidate_report(normalization, measurement, iteration - 1,
                                       SolveStatus::kBreakdown);

    const ScaledScalar alpha = scaled_quotient(squared, curvature);
    if (!alpha.is_finite())
      return terminal_candidate_report(normalization, measurement, iteration - 1,
                                       SolveStatus::kInvalidEvaluation);
    ScaledFieldAlgebra::axpy(iterate, alpha, direction);
    ScaledFieldAlgebra::axpy(residual, ScaledScalar::negated(alpha), applied);
    Real next_squared = PreparedProblemAccess::inner_product(problem, residual, residual);
    measurement.normalized =
        next_squared >= Real(0) ? std::sqrt(next_squared) : std::numeric_limits<Real>::quiet_NaN();
    if (!finite(measurement.normalized))
      return terminal_candidate_report(normalization, measurement, iteration,
                                       SolveStatus::kInvalidEvaluation);
    if (iteration == controls.max_iterations)
      return terminal_candidate_report(normalization, measurement, iteration,
                                       SolveStatus::kIterationLimit);
    bool restart_recurrence = false;
    if (measurement.normalized <= cycle_normalization.normalized_threshold ||
        needs_extreme_recurrence_rebase(measurement.normalized)) {
      ResidualMeasurement confirmed =
          physical_true_residual_measurement(problem, applied, rhs, iterate);
      if (!finite(confirmed.physical))
        return report_physical(normalization, confirmed.physical, iteration,
                               SolveStatus::kInvalidEvaluation);
      if (confirmed.physical <= normalization.physical_threshold)
        return report_physical(normalization, confirmed.physical, iteration, SolveStatus::kSolved);
      rebase_cycle_residual(applied, confirmed, normalization, cycle_normalization);
      PreparedFieldAlgebra::copy(residual, applied);
      next_squared = confirmed.normalized * confirmed.normalized;
      if (!finite(next_squared))
        return report_physical(normalization, confirmed.physical, iteration,
                               SolveStatus::kInvalidEvaluation);
      restart_recurrence = true;
    }
    if (restart_recurrence) {
      // The recursive residual reached either the tolerance or the subnormal danger zone. Replace
      // the complete CG recurrence; mixing rebased r with the old p/rTr state is not CG.
      PreparedFieldAlgebra::copy(direction, residual);
    } else {
      if (squared <= Real(0))
        return terminal_candidate_report(normalization, measurement, iteration,
                                         SolveStatus::kBreakdown);
      const ScaledScalar beta = scaled_quotient(next_squared, squared);
      if (!beta.is_finite())
        return terminal_candidate_report(normalization, measurement, iteration,
                                         SolveStatus::kInvalidEvaluation);
      ScaledFieldAlgebra::lincomb(direction, ScaledScalar::from(Real(1)), residual, beta,
                                  direction);
    }
    squared = next_squared;
  }
  return terminal_candidate_report(normalization, measurement, controls.max_iterations,
                                   SolveStatus::kIterationLimit);
}

inline SolveReport solve_bicgstab(const PreparedAffineLinearProblem& problem,
                                  KrylovWorkspace& workspace, MultiFab& iterate,
                                  const MultiFab& rhs, const KrylovControls& controls,
                                  const SolveNormalization& normalization,
                                  ResidualMeasurement measurement) {
  MultiFab& residual = KrylovWorkspaceAccess::field(workspace, 1);
  MultiFab& shadow = KrylovWorkspaceAccess::field(workspace, 2);
  MultiFab& direction = KrylovWorkspaceAccess::field(workspace, 3);
  MultiFab& applied = KrylovWorkspaceAccess::field(workspace, 4);
  MultiFab& intermediate = KrylovWorkspaceAccess::field(workspace, 5);
  MultiFab& second_applied = KrylovWorkspaceAccess::field(workspace, 6);
  MultiFab& prepared_direction =
      problem.has_preconditioner() ? KrylovWorkspaceAccess::field(workspace, 7) : direction;
  MultiFab& prepared_intermediate =
      problem.has_preconditioner() ? KrylovWorkspaceAccess::field(workspace, 8) : intermediate;

  SolveNormalization cycle_normalization = normalization;
  PreparedFieldAlgebra::copy(shadow, residual);
  PreparedFieldAlgebra::zero(direction);
  PreparedFieldAlgebra::zero(applied);
  Real rho_previous = Real(1);
  ScaledScalar alpha = ScaledScalar::from(Real(1));
  ScaledScalar omega = ScaledScalar::from(Real(1));
  Real preconditioner_scale = Real(0);
  bool restart_recurrence = false;

  for (int completed = 0; completed < controls.max_iterations; ++completed) {
    const int iteration = completed + 1;
    const Real rho = PreparedProblemAccess::inner_product(problem, shadow, residual);
    if (!finite(rho))
      return terminal_candidate_report(normalization, measurement, iteration - 1,
                                       SolveStatus::kInvalidEvaluation);
    if (rho == Real(0))
      return terminal_candidate_report(normalization, measurement, iteration - 1,
                                       SolveStatus::kBreakdown);

    if (iteration == 1 || restart_recurrence) {
      PreparedFieldAlgebra::copy(direction, residual);
      restart_recurrence = false;
    } else {
      if (omega.is_zero())
        return terminal_candidate_report(normalization, measurement, iteration - 1,
                                         SolveStatus::kBreakdown);
      const ScaledScalar beta =
          scaled_product(scaled_quotient(rho, rho_previous), scaled_quotient(alpha, omega));
      if (!beta.is_finite())
        return terminal_candidate_report(normalization, measurement, iteration - 1,
                                         SolveStatus::kInvalidEvaluation);
      ScaledFieldAlgebra::axpy(direction, ScaledScalar::negated(omega), applied);
      ScaledFieldAlgebra::lincomb(direction, ScaledScalar::from(Real(1)), residual, beta,
                                  direction);
    }

    if (problem.has_preconditioner()) {
      const Real scale =
          apply_scaled_preconditioner(problem, prepared_direction, direction, workspace,
                                      preconditioner_scale);
      if (!finite(scale) || !(scale > Real(0)))
        return terminal_candidate_report(
            normalization, measurement, iteration - 1,
            finite(scale) ? SolveStatus::kBreakdown : SolveStatus::kInvalidEvaluation);
    }
    PreparedProblemAccess::apply_linear(problem, applied, prepared_direction,
                                        cycle_normalization.scale);
    const Real denominator = PreparedProblemAccess::inner_product(problem, shadow, applied);
    if (!finite(denominator) || denominator == Real(0))
      return terminal_candidate_report(
          normalization, measurement, iteration - 1,
          finite(denominator) ? SolveStatus::kBreakdown : SolveStatus::kInvalidEvaluation);
    alpha = scaled_quotient(rho, denominator);
    if (!alpha.is_finite())
      return terminal_candidate_report(normalization, measurement, iteration - 1,
                                       SolveStatus::kInvalidEvaluation);
    ScaledFieldAlgebra::lincomb(intermediate, ScaledScalar::from(Real(1)), residual,
                                ScaledScalar::negated(alpha), applied);

    const Real intermediate_norm = PreparedProblemAccess::residual_norm(problem, intermediate);
    if (!finite(intermediate_norm))
      return terminal_candidate_report(normalization, measurement, iteration - 1,
                                       SolveStatus::kInvalidEvaluation);
    if (intermediate_norm <= cycle_normalization.normalized_threshold ||
        needs_extreme_recurrence_rebase(intermediate_norm)) {
      ScaledFieldAlgebra::axpy(iterate, alpha, prepared_direction);
      if (iteration == controls.max_iterations)
        return terminal_candidate_report(normalization, measurement, iteration,
                                         SolveStatus::kIterationLimit);
      ResidualMeasurement confirmed =
          physical_true_residual_measurement(problem, second_applied, rhs, iterate);
      if (!finite(confirmed.physical))
        return report_physical(normalization, confirmed.physical, iteration,
                               SolveStatus::kInvalidEvaluation);
      if (confirmed.physical <= normalization.physical_threshold)
        return report_physical(normalization, confirmed.physical, iteration, SolveStatus::kSolved);
      rebase_cycle_residual(second_applied, confirmed, normalization, cycle_normalization);
      PreparedFieldAlgebra::copy(residual, second_applied);
      PreparedFieldAlgebra::copy(shadow, residual);
      measurement = confirmed;
      restart_recurrence = true;
      preconditioner_scale = Real(0);
      rho_previous = rho;
      continue;
    }

    if (problem.has_preconditioner()) {
      const Real scale = apply_scaled_preconditioner(problem, prepared_intermediate, intermediate,
                                                     workspace, preconditioner_scale);
      if (!finite(scale) || !(scale > Real(0)))
        return terminal_candidate_report(
            normalization, measurement, iteration - 1,
            finite(scale) ? SolveStatus::kBreakdown : SolveStatus::kInvalidEvaluation);
    }
    PreparedProblemAccess::apply_linear(problem, second_applied, prepared_intermediate,
                                        cycle_normalization.scale);
    const Real second_norm_squared =
        PreparedProblemAccess::inner_product(problem, second_applied, second_applied);
    const Real projection =
        PreparedProblemAccess::inner_product(problem, second_applied, intermediate);
    if (!finite(second_norm_squared) || !finite(projection))
      return terminal_candidate_report(normalization, measurement, iteration - 1,
                                       SolveStatus::kInvalidEvaluation);
    if (second_norm_squared <= Real(0))
      return terminal_candidate_report(normalization, measurement, iteration - 1,
                                       SolveStatus::kBreakdown);
    omega = scaled_quotient(projection, second_norm_squared);
    if (!omega.is_finite() || omega.is_zero())
      return terminal_candidate_report(
          normalization, measurement, iteration - 1,
          omega.is_finite() ? SolveStatus::kBreakdown : SolveStatus::kInvalidEvaluation);

    ScaledFieldAlgebra::trilincomb(iterate, ScaledScalar::from(Real(1)), iterate, alpha,
                                   prepared_direction, omega, prepared_intermediate);
    // The BiCGStab recurrence already supplies the next residual. Recomputing b-A(x) here would
    // add a third operator application to every full iteration. Its norm may request an
    // authoritative true-residual confirmation, but can never publish success by itself.
    ScaledFieldAlgebra::lincomb(residual, ScaledScalar::from(Real(1)), intermediate,
                                ScaledScalar::negated(omega), second_applied);
    measurement.normalized = PreparedProblemAccess::residual_norm(problem, residual);
    if (!finite(measurement.normalized))
      return terminal_candidate_report(normalization, measurement, iteration,
                                       SolveStatus::kInvalidEvaluation);
    if (iteration == controls.max_iterations)
      return terminal_candidate_report(normalization, measurement, iteration,
                                       SolveStatus::kIterationLimit);
    if (measurement.normalized <= cycle_normalization.normalized_threshold ||
        needs_extreme_recurrence_rebase(measurement.normalized)) {
      ResidualMeasurement confirmed =
          physical_true_residual_measurement(problem, second_applied, rhs, iterate);
      if (!finite(confirmed.physical))
        return report_physical(normalization, confirmed.physical, iteration,
                               SolveStatus::kInvalidEvaluation);
      if (confirmed.physical <= normalization.physical_threshold)
        return report_physical(normalization, confirmed.physical, iteration, SolveStatus::kSolved);
      rebase_cycle_residual(second_applied, confirmed, normalization, cycle_normalization);
      // Recursive drift or a subnormal recurrence requested an authoritative replacement. Rebase
      // from that scientific residual before continuing.
      PreparedFieldAlgebra::copy(residual, second_applied);
      measurement = confirmed;
      PreparedFieldAlgebra::copy(shadow, residual);
      restart_recurrence = true;
      preconditioner_scale = Real(0);
    }
    rho_previous = rho;
  }
  return terminal_candidate_report(normalization, measurement, controls.max_iterations,
                                   SolveStatus::kIterationLimit);
}

inline bool set_scaled_h(KrylovWorkspace& workspace, int row, int column,
                         const ScaledScalar& value, int basis_extent) {
  Real materialized = Real(0);
  if (!value.is_finite() || !value.try_materialize(materialized))
    return false;
  KrylovWorkspaceAccess::h(workspace, row, column, basis_extent) = materialized;
  KrylovWorkspaceAccess::scaled_h(workspace, row, column, basis_extent) = value;
  return true;
}

inline bool set_scaled_h(KrylovWorkspace& workspace, int row, int column, Real value,
                         int basis_extent) {
  return set_scaled_h(workspace, row, column, ScaledScalar::from(value), basis_extent);
}

inline void set_scaled_rotated_rhs(KrylovWorkspace& workspace, int index,
                                   const ScaledScalar& value, int basis_extent) {
  KrylovWorkspaceAccess::scaled_rotated_rhs(workspace, index, basis_extent) = value;
  Real materialized = Real(0);
  KrylovWorkspaceAccess::rotated_rhs(workspace, index, basis_extent) =
      value.try_materialize(materialized) ? materialized : std::numeric_limits<Real>::quiet_NaN();
}

inline void set_scaled_solution_coefficient(KrylovWorkspace& workspace, int index,
                                            const ScaledScalar& value, int basis_extent) {
  KrylovWorkspaceAccess::scaled_solution_coefficient(workspace, index, basis_extent) = value;
  Real materialized = Real(0);
  KrylovWorkspaceAccess::solution_coefficient(workspace, index, basis_extent) =
      value.try_materialize(materialized) ? materialized : std::numeric_limits<Real>::quiet_NaN();
}

inline void reset_gmres_scalars(KrylovWorkspace& workspace, int restart) {
  for (int row = 0; row <= restart; ++row) {
    set_scaled_rotated_rhs(workspace, row, ScaledScalar::zero(), restart);
    if (row < restart) {
      KrylovWorkspaceAccess::cosine(workspace, row, restart) = Real(0);
      KrylovWorkspaceAccess::sine(workspace, row, restart) = Real(0);
      set_scaled_solution_coefficient(workspace, row, ScaledScalar::zero(), restart);
    }
    for (int column = 0; column < restart; ++column)
      (void)set_scaled_h(workspace, row, column, ScaledScalar::zero(), restart);
  }
}

inline bool solve_gmres_upper(KrylovWorkspace& workspace, int dimension, int basis_extent) {
  for (int row = dimension - 1; row >= 0; --row) {
    ScaledScalar value =
        KrylovWorkspaceAccess::scaled_rotated_rhs(workspace, row, basis_extent);
    for (int column = row + 1; column < dimension; ++column)
      value = scaled_difference(
          value,
          scaled_product(KrylovWorkspaceAccess::scaled_h(workspace, row, column, basis_extent),
                         KrylovWorkspaceAccess::scaled_solution_coefficient(
                             workspace, column, basis_extent)));
    const ScaledScalar diagonal =
        KrylovWorkspaceAccess::scaled_h(workspace, row, row, basis_extent);
    if (!value.is_finite() || !diagonal.is_finite() || diagonal.is_zero())
      return false;
    const ScaledScalar coefficient = scaled_quotient(value, diagonal);
    if (!coefficient.is_finite())
      return false;
    set_scaled_solution_coefficient(workspace, row, coefficient, basis_extent);
  }
  return true;
}

inline SolveReport solve_gmres(const PreparedAffineLinearProblem& problem,
                               KrylovWorkspace& workspace, MultiFab& iterate, const MultiFab& rhs,
                               const KrylovControls& controls, int restart,
                               const SolveNormalization& normalization,
                               ResidualMeasurement measurement) {
  const auto basis = [&workspace](int index) -> MultiFab& {
    return KrylovWorkspaceAccess::field(workspace, static_cast<std::size_t>(index) + 1u);
  };
  MultiFab& applied_or_residual =
      KrylovWorkspaceAccess::field(workspace, static_cast<std::size_t>(restart) + 2u);
  MultiFab* prepared_vector =
      problem.has_preconditioner()
          ? &KrylovWorkspaceAccess::field(workspace, static_cast<std::size_t>(restart) + 3u)
          : nullptr;
  if (KrylovWorkspaceAccess::gmres_reduction_size(workspace) <
      static_cast<std::size_t>(restart) + 1u)
    throw std::logic_error("prepared GMRES reduction workspace is undersized");

  int iterations = 0;
  SolveNormalization cycle_normalization = normalization;
  Real preconditioner_scale = Real(0);
  while (iterations < controls.max_iterations) {
    MultiFab* initial_vector = &applied_or_residual;
    if (prepared_vector != nullptr) {
      const Real scale = apply_scaled_preconditioner(problem, *prepared_vector, applied_or_residual,
                                                     workspace, preconditioner_scale);
      if (!finite(scale) || !(scale > Real(0)))
        return report_physical(
            normalization, measurement.physical, iterations,
            finite(scale) ? SolveStatus::kBreakdown : SolveStatus::kInvalidEvaluation);
      initial_vector = prepared_vector;
    }
    const Real beta = PreparedProblemAccess::residual_norm(problem, *initial_vector);
    if (!finite(beta))
      return report_physical(normalization, measurement.physical, iterations,
                             SolveStatus::kInvalidEvaluation);
    if (beta == Real(0))
      return report_physical(normalization, measurement.physical, iterations,
                             SolveStatus::kBreakdown);
    // The Arnoldi estimate lives in the left-preconditioned norm whereas `threshold` and
    // `measurement.normalized` lives in the unpreconditioned normalized equation. Map its remaining
    // reduction into the current preconditioned cycle instead of comparing unrelated norms. This
    // keeps a scalar rescaling of an otherwise identical preconditioner from changing restart
    // behaviour, costs no extra preconditioner application or collective, and remains only a
    // request for the authoritative true-residual confirmation below.
    const ScaledScalar estimate_threshold = scaled_product(
        ScaledScalar::from(beta),
        scaled_quotient(cycle_normalization.normalized_threshold, measurement.normalized));
    if (!estimate_threshold.is_finite())
      return report_physical(normalization, measurement.physical, iterations,
                             SolveStatus::kInvalidEvaluation);
    PreparedFieldAlgebra::copy(basis(0), *initial_vector);
    PreparedFieldAlgebra::divide(basis(0), beta);
    reset_gmres_scalars(workspace, restart);
    set_scaled_rotated_rhs(workspace, 0, ScaledScalar::from(beta), restart);

    int dimension = 0;
    bool estimate_reached = false;
    bool invalid = false;
    for (int column = 0; column < restart && iterations < controls.max_iterations; ++column) {
      PreparedProblemAccess::apply_linear(problem, applied_or_residual, basis(column),
                                          cycle_normalization.scale);
      MultiFab* arnoldi_vector = &applied_or_residual;
      if (prepared_vector != nullptr) {
        const Real scale = apply_scaled_preconditioner(problem, *prepared_vector,
                                                       applied_or_residual, workspace,
                                                       preconditioner_scale);
        if (!finite(scale) || !(scale > Real(0))) {
          invalid = true;
          break;
        }
        arnoldi_vector = prepared_vector;
      }
      // Classical Arnoldi computes all local projections before one vector reduction. The final
      // slot carries the unprojected ||w||^2 used only by the DGKS norm-loss criterion. The
      // projected norm itself is evaluated directly: deriving it as ||w||^2-sum(h^2) assumes an
      // exactly orthonormal basis and can silently mis-normalize a finite-precision CGS basis.
      double* reductions = KrylovWorkspaceAccess::gmres_reduction_data(workspace);
      for (int row = 0; row <= column; ++row)
        reductions[row] = static_cast<double>(
            PreparedProblemAccess::local_inner_product(problem, *arnoldi_vector, basis(row)));
      reductions[column + 1] = static_cast<double>(
          PreparedProblemAccess::local_inner_product(problem, *arnoldi_vector, *arnoldi_vector));
      reduce_batched_inner_products(problem, workspace, reductions, column + 2,
                                    "prepared GMRES Arnoldi projections");

      bool finite_column = repair_nonfinite_batched_inner_products(
          problem, workspace, *arnoldi_vector, reductions, column + 1, restart,
          [&basis](int row) -> const MultiFab& { return basis(row); });

      const Real raw_square = static_cast<Real>(reductions[column + 1]);
      const bool finite_raw_square = finite(raw_square) && raw_square >= Real(0);
      for (int row = 0; row <= column; ++row) {
        const Real projection = static_cast<Real>(reductions[row]);
        finite_column = finite_column && finite(projection);
        if (!set_scaled_h(workspace, row, column, projection, restart)) {
          finite_column = false;
          break;
        }
        PreparedFieldAlgebra::axpy(*arnoldi_vector, -projection, basis(row));
      }
      Real arnoldi_norm = finite_column
                              ? PreparedProblemAccess::residual_norm(problem, *arnoldi_vector)
                              : std::numeric_limits<Real>::quiet_NaN();

      // Selective CGS2 restores MGS-class robustness on the hard columns without returning to one
      // MPI collective per basis vector. A second batched pass is paid only when the first
      // projection loses at least half the vector norm (the standard DGKS trigger).
      constexpr Real kReorthogonalizeRatio = Real(0.5);
      // A finite Arnoldi vector can have an unrepresentable raw square (for example 1e300 squared).
      // That value is used only by the DGKS heuristic, not by the Hessenberg column. Conservatively
      // take the second pass when it overflowed; the scale-safe post-projection norm remains the
      // authority for validity and lucky breakdown.
      if (finite(arnoldi_norm) &&
          (!finite_raw_square || (raw_square > Real(0) &&
                                  arnoldi_norm <= kReorthogonalizeRatio * std::sqrt(raw_square)))) {
        for (int row = 0; row <= column; ++row)
          reductions[row] = static_cast<double>(
              PreparedProblemAccess::local_inner_product(problem, *arnoldi_vector, basis(row)));
        reduce_batched_inner_products(problem, workspace, reductions, column + 1,
                                      "prepared GMRES DGKS projections");
        finite_column =
            finite_column && repair_nonfinite_batched_inner_products(
                                 problem, workspace, *arnoldi_vector, reductions, column + 1,
                                 restart,
                                 [&basis](int row) -> const MultiFab& { return basis(row); });
        for (int row = 0; row <= column; ++row) {
          const Real correction = static_cast<Real>(reductions[row]);
          finite_column = finite_column && finite(correction);
          const ScaledScalar corrected_h =
              scaled_sum(KrylovWorkspaceAccess::scaled_h(workspace, row, column, restart),
                         ScaledScalar::from(correction));
          if (!set_scaled_h(workspace, row, column, corrected_h, restart)) {
            finite_column = false;
            break;
          }
          PreparedFieldAlgebra::axpy(*arnoldi_vector, -correction, basis(row));
        }
        arnoldi_norm = finite_column
                           ? PreparedProblemAccess::residual_norm(problem, *arnoldi_vector)
                           : std::numeric_limits<Real>::quiet_NaN();
      }
      if (!set_scaled_h(workspace, column + 1, column, arnoldi_norm, restart)) {
        invalid = true;
        break;
      }
      if (!finite(arnoldi_norm)) {
        invalid = true;
        break;
      }
      const bool lucky_breakdown = arnoldi_norm == Real(0);
      if (!lucky_breakdown) {
        PreparedFieldAlgebra::copy(basis(column + 1), *arnoldi_vector);
        PreparedFieldAlgebra::divide(basis(column + 1), arnoldi_norm);
      }

      for (int row = 0; row < column; ++row) {
        const Real first = KrylovWorkspaceAccess::h(workspace, row, column, restart);
        const Real second = KrylovWorkspaceAccess::h(workspace, row + 1, column, restart);
        const ScaledScalar rotated_first =
            scaled_sum(scaled_product(KrylovWorkspaceAccess::cosine(workspace, row, restart),
                                      first),
                       scaled_product(KrylovWorkspaceAccess::sine(workspace, row, restart),
                                      second));
        const ScaledScalar rotated_second = scaled_sum(
            scaled_product(ScaledScalar::negated(
                               ScaledScalar::from(
                                   KrylovWorkspaceAccess::sine(workspace, row, restart))),
                           ScaledScalar::from(first)),
            scaled_product(KrylovWorkspaceAccess::cosine(workspace, row, restart), second));
        if (!set_scaled_h(workspace, row, column, rotated_first, restart) ||
            !set_scaled_h(workspace, row + 1, column, rotated_second, restart)) {
          invalid = true;
          break;
        }
      }
      if (invalid)
        break;
      const Real diagonal = KrylovWorkspaceAccess::h(workspace, column, column, restart);
      const Real subdiagonal =
          KrylovWorkspaceAccess::h(workspace, column + 1, column, restart);
      const Real magnitude = std::hypot(diagonal, subdiagonal);
      if (!finite(magnitude) || magnitude == Real(0))
        return terminal_candidate_report(
            normalization, measurement, iterations,
            !finite(magnitude) ? SolveStatus::kInvalidEvaluation : SolveStatus::kBreakdown);
      KrylovWorkspaceAccess::cosine(workspace, column, restart) = diagonal / magnitude;
      KrylovWorkspaceAccess::sine(workspace, column, restart) = subdiagonal / magnitude;
      if (!set_scaled_h(workspace, column, column, magnitude, restart) ||
          !set_scaled_h(workspace, column + 1, column, Real(0), restart)) {
        invalid = true;
        break;
      }
      const ScaledScalar prior_rhs =
          KrylovWorkspaceAccess::scaled_rotated_rhs(workspace, column, restart);
      set_scaled_rotated_rhs(workspace, column + 1,
                             scaled_product(ScaledScalar::negated(ScaledScalar::from(
                                                KrylovWorkspaceAccess::sine(
                                                    workspace, column, restart))),
                                            prior_rhs),
                             restart);
      set_scaled_rotated_rhs(
          workspace, column,
          scaled_product(ScaledScalar::from(
                             KrylovWorkspaceAccess::cosine(workspace, column, restart)),
                         prior_rhs),
          restart);
      dimension = column + 1;
      ++iterations;
      estimate_reached = ScaledScalar::abs_less_equal(
          KrylovWorkspaceAccess::scaled_rotated_rhs(workspace, column + 1, restart),
          estimate_threshold);
      if (estimate_reached || lucky_breakdown)
        break;
    }

    if (invalid)
      return terminal_candidate_report(normalization, measurement, iterations,
                                       SolveStatus::kInvalidEvaluation);
    if (dimension == 0 || !solve_gmres_upper(workspace, dimension, restart))
      return terminal_candidate_report(normalization, measurement, iterations,
                                       SolveStatus::kBreakdown);
    for (int column = 0; column < dimension; ++column)
      ScaledFieldAlgebra::axpy(
          iterate,
          KrylovWorkspaceAccess::scaled_solution_coefficient(workspace, column, restart),
          basis(column));

    if (iterations == controls.max_iterations)
      return terminal_candidate_report(normalization, measurement, iterations,
                                       SolveStatus::kIterationLimit);
    measurement = physical_true_residual_measurement(problem, applied_or_residual, rhs, iterate);
    if (!finite(measurement.physical))
      return report_physical(normalization, measurement.physical, iterations,
                             SolveStatus::kInvalidEvaluation);
    // An Arnoldi or preconditioned estimate may only request this confirmation.  It never publishes
    // success by itself; the raw scientific residual b-A(u) above is authoritative.
    if (measurement.physical <= normalization.physical_threshold)
      return report_physical(normalization, measurement.physical, iterations, SolveStatus::kSolved);
    rebase_cycle_residual(applied_or_residual, measurement, normalization, cycle_normalization);
    // The next restart is a new Krylov recurrence.  It may choose a fresh scalar-equivalent
    // preconditioner normalization suited to its newly rebased residual; within that cycle the
    // scalar remains fixed.
    preconditioner_scale = Real(0);
    (void)estimate_reached;
  }
  return terminal_candidate_report(normalization, measurement, iterations,
                                   SolveStatus::kIterationLimit);
}

}  // namespace detail

/// Allocation-free execution view passed to one prepared method provider.  It exposes only the
/// already-authenticated problem, persistent workspace pools, and primitive field operations; a
/// provider cannot trigger lazy storage construction through this interface.
class PreparedKrylovSolveContext {
 public:
  [[nodiscard]] const KrylovControls& controls() const noexcept { return controls_; }
  [[nodiscard]] const LinearOperatorProperties& operator_properties() const noexcept {
    return problem_.properties();
  }
  [[nodiscard]] bool has_nullspace() const noexcept { return problem_.has_nullspace(); }
  [[nodiscard]] bool has_preconditioner() const noexcept { return problem_.has_preconditioner(); }
  [[nodiscard]] const PreparedVectorDistribution& vector_distribution() const noexcept {
    return problem_.vector_distribution();
  }
  [[nodiscard]] Real equation_scale() const noexcept { return normalization_.scale; }
  [[nodiscard]] Real reference_norm() const noexcept { return normalization_.reference; }
  [[nodiscard]] Real physical_threshold() const noexcept {
    return normalization_.physical_threshold;
  }
  [[nodiscard]] Real initial_physical_residual() const noexcept {
    return initial_measurement_.physical;
  }

  [[nodiscard]] MultiFab& iterate() noexcept { return iterate_; }
  [[nodiscard]] const MultiFab& rhs() const noexcept { return rhs_; }
  [[nodiscard]] MultiFab& field(std::size_t index) {
    return detail::KrylovWorkspaceAccess::field(workspace_, index);
  }
  [[nodiscard]] MultiFab& initial_residual() {
    return field(detail::KrylovWorkspaceAccess::initial_residual_field(workspace_));
  }
  [[nodiscard]] Real& real_value(std::size_t index) {
    return workspace_.real_value(index);
  }
  [[nodiscard]] detail::ScaledScalar& scaled_value(std::size_t index) {
    return workspace_.scaled_value(index);
  }
  [[nodiscard]] std::span<double> collective_values() {
    return {workspace_.collective_data(), workspace_.collective_data_size()};
  }
  [[nodiscard]] std::uint64_t& state_word(std::size_t index) {
    return workspace_.state_word(index);
  }
  [[nodiscard]] std::size_t robust_payload_width() const noexcept {
    return workspace_.metric_robust_payload_width();
  }

  void zero(MultiFab& value) const { detail::PreparedFieldAlgebra::zero(value); }
  void copy(MultiFab& out, const MultiFab& in) const {
    detail::PreparedFieldAlgebra::copy(out, in);
  }
  void divide(MultiFab& value, Real denominator) const {
    detail::PreparedFieldAlgebra::divide(value, denominator);
  }
  void axpy(MultiFab& out, Real coefficient, const MultiFab& in) const {
    detail::PreparedFieldAlgebra::axpy(out, coefficient, in);
  }
  void add_physical_direction(MultiFab& out, Real coefficient,
                              const MultiFab& normalized_direction) const {
    detail::ScaledFieldAlgebra::axpy(
        out, detail::scaled_product(coefficient, normalization_.scale), normalized_direction);
  }
  void apply_linear(MultiFab& out, const MultiFab& direction,
                    Real equation_scale) const {
    detail::PreparedProblemAccess::apply_linear(problem_, out, direction, equation_scale);
  }
  void apply_linear(MultiFab& out, const MultiFab& direction) const {
    apply_linear(out, direction, normalization_.scale);
  }
  void apply_preconditioner(MultiFab& out, const MultiFab& in) const {
    detail::PreparedProblemAccess::apply_preconditioner(problem_, out, in);
  }
  [[nodiscard]] Real inner_product(const MultiFab& left, const MultiFab& right) const {
    return detail::PreparedProblemAccess::inner_product(problem_, left, right);
  }
  [[nodiscard]] Real residual_norm(const MultiFab& value) const {
    return detail::PreparedProblemAccess::residual_norm(problem_, value);
  }
  void local_robust_inner_product_payload(const MultiFab& left, const MultiFab& right,
                                          std::span<double> payload) const {
    detail::PreparedProblemAccess::local_robust_inner_product_payload(problem_, left, right,
                                                                      payload);
  }
  [[nodiscard]] Real inner_product_from_global_robust_payload(
      std::span<const double> payload) const {
    return detail::PreparedProblemAccess::inner_product_from_global_robust_payload(problem_,
                                                                                    payload);
  }
  void reduce_inner_products(double* values, int count, const char* quantity) {
    detail::reduce_batched_inner_products(problem_, workspace_, values, count, quantity);
  }
  [[nodiscard]] Real true_residual_norm(MultiFab& scratch) const {
    return detail::physical_true_residual_norm(problem_, scratch, rhs_, iterate_);
  }
  [[nodiscard]] SolveReport report(Real physical_residual, int iterations,
                                   SolveStatus status) const {
    return detail::report_physical(normalization_, physical_residual, iterations, status);
  }

 private:
  PreparedKrylovSolveContext(const PreparedAffineLinearProblem& problem,
                             KrylovWorkspace& workspace, MultiFab& iterate, const MultiFab& rhs,
                             const KrylovControls& controls,
                             detail::SolveNormalization normalization,
                             detail::ResidualMeasurement initial_measurement)
      : problem_(problem),
        workspace_(workspace),
        iterate_(iterate),
        rhs_(rhs),
        controls_(controls),
        normalization_(normalization),
        initial_measurement_(initial_measurement) {}

  friend class detail::CgKrylovMethodProvider;
  friend class detail::BicgstabKrylovMethodProvider;
  friend class detail::GmresKrylovMethodProvider;
  friend class detail::RichardsonKrylovMethodProvider;
  friend SolveReport solve_prepared_affine(const PreparedAffineLinearProblem&, KrylovWorkspace&,
                                            MultiFab&, const MultiFab&,
                                            const KrylovControls&);

  const PreparedAffineLinearProblem& problem_;
  KrylovWorkspace& workspace_;
  MultiFab& iterate_;
  const MultiFab& rhs_;
  const KrylovControls& controls_;
  detail::SolveNormalization normalization_;
  detail::ResidualMeasurement initial_measurement_;
};

inline SolveReport detail::CgKrylovMethodProvider::solve(
    PreparedKrylovSolveContext& context, const PreparedProviderOptions&) const {
  return detail::solve_cg(context.problem_, context.workspace_, context.iterate_, context.rhs_,
                          context.controls_, context.normalization_,
                          context.initial_measurement_);
}

inline SolveReport detail::BicgstabKrylovMethodProvider::solve(
    PreparedKrylovSolveContext& context, const PreparedProviderOptions&) const {
  return detail::solve_bicgstab(context.problem_, context.workspace_, context.iterate_,
                                context.rhs_, context.controls_, context.normalization_,
                                context.initial_measurement_);
}

inline SolveReport detail::GmresKrylovMethodProvider::solve(
    PreparedKrylovSolveContext& context, const PreparedProviderOptions& options) const {
  const std::int64_t* restart =
      detail::exact_int_option(options, detail::kGmresOptionsSchema, "restart");
  if (restart == nullptr)
    throw std::logic_error("prepared GMRES options were not authenticated");
  return detail::solve_gmres(context.problem_, context.workspace_, context.iterate_, context.rhs_,
                             context.controls_, static_cast<int>(*restart), context.normalization_,
                             context.initial_measurement_);
}

inline SolveReport detail::RichardsonKrylovMethodProvider::solve(
    PreparedKrylovSolveContext& context, const PreparedProviderOptions& options) const {
  const double* relaxation =
      detail::exact_real_option(options, detail::kRichardsonOptionsSchema, "relaxation");
  if (relaxation == nullptr)
    throw std::logic_error("prepared Richardson options were not authenticated");
  return detail::solve_richardson(context.problem_, context.workspace_, context.iterate_,
                                  context.rhs_, context.controls_, static_cast<Real>(*relaxation),
                                  context.normalization_,
                                  context.initial_measurement_);
}

inline std::shared_ptr<PreparedKrylovMethodRegistry>
make_default_krylov_method_provider_registry() {
  auto registry = std::make_shared<PreparedKrylovMethodRegistry>();
  registry->add(std::make_shared<detail::CgKrylovMethodProvider>());
  registry->add(std::make_shared<detail::BicgstabKrylovMethodProvider>());
  registry->add(std::make_shared<detail::GmresKrylovMethodProvider>());
  registry->add(std::make_shared<detail::RichardsonKrylovMethodProvider>());
  return registry;
}

namespace detail {
inline const PreparedKrylovMethodRegistry& default_krylov_method_registry() {
  static const std::shared_ptr<PreparedKrylovMethodRegistry> registry =
      make_default_krylov_method_provider_registry();
  return *registry;
}
}  // namespace detail

inline PreparedKrylovMethod cg_krylov_method() {
  return detail::default_krylov_method_registry().resolve(
      "pops.krylov.cg", PreparedProviderOptions{std::string(detail::kCgOptionsSchema), {}});
}
inline PreparedKrylovMethod bicgstab_krylov_method() {
  return detail::default_krylov_method_registry().resolve(
      "pops.krylov.bicgstab",
      PreparedProviderOptions{std::string(detail::kBicgstabOptionsSchema), {}});
}
inline PreparedKrylovMethod gmres_krylov_method(int restart) {
  if (restart < 1)
    throw std::invalid_argument("prepared GMRES restart must be positive");
  return detail::default_krylov_method_registry().resolve(
      "pops.krylov.gmres",
      PreparedProviderOptions{
          std::string(detail::kGmresOptionsSchema),
          {{"restart", static_cast<std::int64_t>(restart)}}});
}
inline PreparedKrylovMethod richardson_krylov_method(Real relaxation) {
  if (!detail::finite(relaxation) || !(relaxation > Real(0)))
    throw std::invalid_argument("prepared Richardson relaxation must be finite and positive");
  return detail::default_krylov_method_registry().resolve(
      "pops.krylov.richardson",
      PreparedProviderOptions{
          std::string(detail::kRichardsonOptionsSchema),
          {{"relaxation", static_cast<double>(relaxation)}}});
}

/// Solve one explicitly prepared affine problem with persistent workspace.  There are no legacy raw
/// callback overloads: preparation, property checks, exact snapshot binding, and memory footprint are
/// mandatory parts of the API rather than optional caller conventions.
inline SolveReport solve_prepared_affine(const PreparedAffineLinearProblem& problem,
                                         KrylovWorkspace& workspace, MultiFab& iterate,
                                         const MultiFab& rhs, const KrylovControls& controls) {
  detail::collective_solve_preflight(problem, workspace, iterate, rhs, controls);

  MultiFab& compatibility_rhs = detail::KrylovWorkspaceAccess::field(workspace, 0);
  const PreparedEquationReference equation =
      detail::PreparedProblemAccess::prepare_compatibility_rhs(problem, compatibility_rhs, rhs);
  if (!detail::finite(equation.reference_norm)) {
    const detail::SolveNormalization invalid_reference{equation.reference_norm, Real(1), Real(0),
                                                       controls.abs_tol};
    return detail::report_physical(invalid_reference, std::numeric_limits<Real>::quiet_NaN(), 0,
                                   SolveStatus::kInvalidEvaluation);
  }
  const Real physical_threshold =
      detail::physical_stopping_threshold(equation.reference_norm, controls);
  const detail::SolveNormalization report_normalization{equation.reference_norm, Real(1), Real(0),
                                                        physical_threshold};

  // Singular compatibility is checked exactly once, collectively and before either the initial
  // gauge or an iterative operator application. The typed status keeps authored outcome/action
  // handling on the SolveReport path instead of leaking a generic exception past it.
  try {
    detail::PreparedProblemAccess::require_nullspace_compatible(problem, compatibility_rhs);
  } catch (const FieldNullspaceIncompatibleRhs& error) {
    // Compatibility failure leaves the authored iterate untouched.  Its report still carries the
    // exact scientific residual of that iterate, not the generally different ||R(0)|| reference.
    const Real residual =
        detail::physical_true_residual_norm(problem, compatibility_rhs, rhs, iterate);
    SolveReport incompatible = detail::report_physical(
        report_normalization, residual, 0,
        detail::finite(residual) ? SolveStatus::kIncompatibleRhs : SolveStatus::kInvalidEvaluation);
    incompatible.reason = error.what();
    return incompatible;
  } catch (const FieldNullspaceInvalidEvaluation& error) {
    SolveReport invalid =
        detail::report_physical(report_normalization, std::numeric_limits<Real>::quiet_NaN(), 0,
                                SolveStatus::kInvalidEvaluation);
    invalid.reason = error.what();
    return invalid;
  }
  if (problem.has_nullspace())
    detail::PreparedProblemAccess::apply_nullspace_gauge(problem, iterate);

  MultiFab& initial_residual = detail::initial_residual_field(workspace, controls);
  const Real initial_physical =
      detail::physical_true_residual_norm(problem, initial_residual, rhs, iterate);
  if (!detail::finite(initial_physical))
    return detail::report_physical(report_normalization, initial_physical, 0,
                                   SolveStatus::kInvalidEvaluation);
  if (initial_physical <= physical_threshold)
    return detail::report_physical(report_normalization, initial_physical, 0, SolveStatus::kSolved);

  // The authored reference controls tolerance and nullspace compatibility, but it must never scale
  // the recurrence field: an unrelated large component of ||b-A(0)|| can coexist with a finite,
  // tiny warm-start residual and would round that residual to zero.  Scaling by the measured initial
  // residual keeps its normalized norm at one while make_normalization maps the independently
  // authored physical threshold into this recurrence scale.
  const Real solve_scale = initial_physical;
  const detail::SolveNormalization normalization =
      detail::make_normalization(equation.reference_norm, solve_scale, controls);
  detail::PreparedFieldAlgebra::divide(initial_residual, solve_scale);
  const detail::ResidualMeasurement initial_measurement{initial_physical,
                                                        initial_physical / solve_scale};

  PreparedKrylovSolveContext method_context(problem, workspace, iterate, rhs, controls,
                                             normalization, initial_measurement);
  SolveReport result;
  long provider_exception_local = 0;
  try {
    // A provider that enters MPI owns one identical, complete collective trace on every rank.  The
    // wrapper can make an exception uniform only after that trace has completed; it deliberately
    // does not pretend to repair a callback that abandons a collective midway.
    result = controls.method.solve(method_context);
  } catch (...) {
    provider_exception_local = 1;
  }
  if (all_reduce_max(provider_exception_local) != 0) {
    SolveReport invalid = detail::report_physical(
        normalization, std::numeric_limits<Real>::quiet_NaN(), 0,
        SolveStatus::kInvalidEvaluation);
    invalid.reason =
        "prepared Krylov provider failed after its collective solve trace";
    return invalid;
  }
  if (!detail::provider_solve_report_agrees(result, workspace)) {
    SolveReport invalid = detail::report_physical(
        normalization, std::numeric_limits<Real>::quiet_NaN(), 0,
        SolveStatus::kInvalidEvaluation);
    invalid.reason =
        "prepared Krylov provider report differs between communicator ranks";
    return invalid;
  }

  // A method provider controls a recurrence, never scientific convergence authority.  Project a
  // singular candidate to its authored representative first, then independently recompute the
  // physical residual for every provider using an existing workspace field.  This rejects a
  // provider that returns a false Solved/NaN report or leaves the iterate unchanged, without a
  // per-solve allocation or a method-name branch in the core.
  if (problem.has_nullspace()) {
    detail::PreparedProblemAccess::apply_nullspace_gauge(problem, iterate);
  }
  const Real final_residual =
      detail::physical_true_residual_norm(problem, compatibility_rhs, rhs, iterate);
  if (!detail::finite(final_residual)) {
    result.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                       "prepared Krylov provider produced a non-finite true residual");
  } else if (result.iters < 0 || result.iters > controls.max_iterations) {
    result.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                       "prepared Krylov provider returned an invalid iteration count");
  } else if (final_residual <= normalization.physical_threshold) {
    result.mark_solved();
  } else if (result.solved() || !result.valid()) {
    result.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                       problem.has_nullspace()
                           ? "prepared nullspace representative failed the true-residual check"
                           : "prepared Krylov provider claimed an unverified solved value");
  }
  detail::set_report_physical_residuals(result, normalization, final_residual);
  return result;
}

}  // namespace pops
