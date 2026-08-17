#pragma once

/// @file
/// @brief Allocation-free Krylov algorithms over a prepared affine problem.
///
/// This is deliberately the only public generic Krylov entry point.  Callers must prepare and
/// authenticate the operator evaluation, bind persistent workspace, and pass typed controls.  The
/// algorithms therefore never guess that an affine A is linear, lazily build a preconditioner,
/// allocate scratch in an iteration, or publish an Arnoldi/preconditioned residual as scientific
/// convergence.

#include <pops/numerics/elliptic/linear/krylov_method_provider.hpp>
#include <pops/numerics/elliptic/linear/krylov_workspace.hpp>
#include <pops/numerics/elliptic/linear/scaled_field_algebra.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace pops {
template <int Dim>
class PreparedKrylovSolveContext;
template <int Dim>
class PreparedKrylovInvocation;
namespace detail {

template <int Dim>
PreparedKrylovInvocation<Dim> prepare_krylov_solve_in_place(const PreparedAffineLinearProblem<Dim>&,
                                                            KrylovWorkspace<Dim>&, MultiFab<Dim>&,
                                                            const MultiFab<Dim>&,
                                                            const KrylovControls<Dim>&,
                                                            bool staged_publication = false);
template <int Dim>
SolveReport solve_prepared_affine_in_place(const PreparedAffineLinearProblem<Dim>&,
                                           KrylovWorkspace<Dim>&, MultiFab<Dim>&,
                                           const MultiFab<Dim>&, const KrylovControls<Dim>&);

struct PreparedKrylovInvocationAccess {
  template <int Dim>
  static SolveReport execute(const PreparedAffineLinearProblem<Dim>& problem,
                             KrylovWorkspace<Dim>& workspace, MultiFab<Dim>& iterate,
                             const MultiFab<Dim>& rhs, const KrylovControls<Dim>& controls);
};

/// The algorithms are the sole consumers of persistent workspace storage. Keeping this access
/// object private to detail prevents callers from replacing warmed fields or scalar buffers while
/// still letting every Krylov method share the same allocation-free storage.
struct KrylovWorkspaceAccess {
  template <int Dim>
  static MultiFab<Dim>& field(KrylovWorkspace<Dim>& workspace, std::size_t index) {
    return workspace.field(index);
  }
  template <int Dim>
  static Real& h(KrylovWorkspace<Dim>& workspace, int row, int column, int basis_extent) {
    const std::size_t extent = static_cast<std::size_t>(basis_extent);
    return workspace.real_value(static_cast<std::size_t>(row) * extent +
                                static_cast<std::size_t>(column));
  }
  template <int Dim>
  static Real& cosine(KrylovWorkspace<Dim>& workspace, int index, int basis_extent) {
    const std::size_t extent = static_cast<std::size_t>(basis_extent);
    return workspace.real_value((extent + 1u) * extent + static_cast<std::size_t>(index));
  }
  template <int Dim>
  static Real& sine(KrylovWorkspace<Dim>& workspace, int index, int basis_extent) {
    const std::size_t extent = static_cast<std::size_t>(basis_extent);
    return workspace.real_value((extent + 1u) * extent + extent + static_cast<std::size_t>(index));
  }
  template <int Dim>
  static Real& rotated_rhs(KrylovWorkspace<Dim>& workspace, int index, int basis_extent) {
    const std::size_t extent = static_cast<std::size_t>(basis_extent);
    return workspace.real_value((extent + 1u) * extent + 2u * extent +
                                static_cast<std::size_t>(index));
  }
  template <int Dim>
  static Real& solution_coefficient(KrylovWorkspace<Dim>& workspace, int index, int basis_extent) {
    const std::size_t extent = static_cast<std::size_t>(basis_extent);
    return workspace.real_value((extent + 1u) * extent + 3u * extent + 1u +
                                static_cast<std::size_t>(index));
  }
  template <int Dim>
  static ScaledScalar& scaled_h(KrylovWorkspace<Dim>& workspace, int row, int column,
                                int basis_extent) {
    const std::size_t extent = static_cast<std::size_t>(basis_extent);
    return workspace.scaled_value(static_cast<std::size_t>(row) * extent +
                                  static_cast<std::size_t>(column));
  }
  template <int Dim>
  static ScaledScalar& scaled_rotated_rhs(KrylovWorkspace<Dim>& workspace, int index,
                                          int basis_extent) {
    const std::size_t extent = static_cast<std::size_t>(basis_extent);
    return workspace.scaled_value((extent + 1u) * extent + static_cast<std::size_t>(index));
  }
  template <int Dim>
  static ScaledScalar& scaled_solution_coefficient(KrylovWorkspace<Dim>& workspace, int index,
                                                   int basis_extent) {
    const std::size_t extent = static_cast<std::size_t>(basis_extent);
    return workspace.scaled_value((extent + 1u) * extent + extent + 1u +
                                  static_cast<std::size_t>(index));
  }
  template <int Dim>
  static double* gmres_reduction_data(KrylovWorkspace<Dim>& workspace) {
    return workspace.collective_data();
  }
  template <int Dim>
  static double* gmres_robust_reduction_data(KrylovWorkspace<Dim>& workspace, int basis_extent) {
    return workspace.collective_data() + static_cast<std::size_t>(basis_extent) + 1u;
  }
  template <int Dim>
  static double* distribution_reduction_data(KrylovWorkspace<Dim>& workspace) {
    return workspace.distribution_reduction_data();
  }
  template <int Dim>
  static std::size_t distribution_reduction_size(const KrylovWorkspace<Dim>& workspace) {
    return workspace.distribution_reduction_size();
  }
  template <int Dim>
  static char* distribution_validation_data(KrylovWorkspace<Dim>& workspace) {
    return workspace.distribution_validation_data();
  }
  template <int Dim>
  static std::size_t distribution_validation_size(const KrylovWorkspace<Dim>& workspace) {
    return workspace.distribution_validation_size();
  }
  template <int Dim>
  static std::span<double> metric_reduction_scratch(KrylovWorkspace<Dim>& workspace) {
    return {workspace.metric_reduction_data(), workspace.metric_reduction_size()};
  }
  template <int Dim>
  static std::span<char> distribution_validation_scratch(KrylovWorkspace<Dim>& workspace) {
    return {workspace.distribution_validation_data(), workspace.distribution_validation_size()};
  }
  template <int Dim>
  static std::span<double> gauge_coefficients(KrylovWorkspace<Dim>& workspace) {
    return workspace.gauge_coefficients();
  }
  template <int Dim>
  static PreparedLinearPreconditionerSession<Dim>& preconditioner_session(
      KrylovWorkspace<Dim>& workspace) {
    return workspace.preconditioner_session();
  }
  template <int Dim>
  static PreparedAffineOperatorSession<Dim>& operator_session(KrylovWorkspace<Dim>& workspace) {
    return workspace.operator_session();
  }
  template <int Dim>
  static const ExecutionLane& execution_lane(const KrylovWorkspace<Dim>& workspace) {
    return workspace.execution_lane();
  }
  template <int Dim>
  static std::string_view materialization_token(const KrylovWorkspace<Dim>& workspace) noexcept {
    return workspace.materialization_token_;
  }
  template <int Dim>
  static const MultiFab<Dim>& preconditioner_constant(const KrylovWorkspace<Dim>& workspace) {
    return workspace.preconditioner_constant();
  }
  template <int Dim>
  static std::size_t gmres_reduction_size(const KrylovWorkspace<Dim>& workspace) {
    return workspace.collective_data_size();
  }
  template <int Dim>
  static std::size_t metric_robust_payload_width(const KrylovWorkspace<Dim>& workspace) {
    return workspace.metric_robust_payload_width();
  }
  template <int Dim>
  static std::size_t initial_residual_field(const KrylovWorkspace<Dim>& workspace) {
    return workspace.requirements_.initial_residual_field;
  }
  template <int Dim>
  static bool provider_report_agrees(KrylovWorkspace<Dim>& workspace, const SolveReport& report) {
    return workspace.provider_report_agrees_(report);
  }
  template <int Dim>
  static bool try_reserve_solve(KrylovWorkspace<Dim>& workspace) noexcept {
    return workspace.try_reserve_solve_();
  }
  template <int Dim>
  static void release_solve(KrylovWorkspace<Dim>& workspace) noexcept {
    workspace.release_solve_();
  }
  template <int Dim>
  static MultiFab<Dim>& publication_candidate(KrylovWorkspace<Dim>& workspace) {
    return workspace.publication_candidate_field_();
  }
  template <int Dim>
  static void arm_publication(KrylovWorkspace<Dim>& workspace,
                              const PreparedAffineLinearProblem<Dim>& problem,
                              MultiFab<Dim>& destination) noexcept {
    workspace.arm_publication_(problem, destination);
  }
  template <int Dim>
  static void publish_candidate(KrylovWorkspace<Dim>& workspace) {
    workspace.publish_candidate_();
  }
  template <int Dim>
  static void validate_publication(const KrylovWorkspace<Dim>& workspace) {
    workspace.validate_publication_();
  }
  template <int Dim>
  static void release_publication(KrylovWorkspace<Dim>& workspace) noexcept {
    workspace.release_publication_();
  }
  template <int Dim>
  static void reset_provider_apply_status(KrylovWorkspace<Dim>& workspace) noexcept {
    workspace.reset_provider_apply_status_();
  }
  template <int Dim>
  static void latch_provider_apply_status(KrylovWorkspace<Dim>& workspace,
                                          const PreparedApplyResult& result) noexcept {
    workspace.latch_provider_apply_status_(result);
  }
  template <int Dim>
  static bool provider_apply_succeeded(const KrylovWorkspace<Dim>& workspace) noexcept {
    return workspace.provider_apply_succeeded_();
  }
  template <int Dim>
  static PreparedApplyResult collective_provider_apply_result(
      const KrylovWorkspace<Dim>& workspace) {
    return workspace.collective_provider_apply_result_();
  }
  template <int Dim>
  static void republish_provider_apply_failure(KrylovWorkspace<Dim>& workspace,
                                               MultiFab<Dim>& out) noexcept {
    workspace.republish_provider_apply_failure_(out);
  }
  template <int Dim>
  static void append_collective_state(const KrylovWorkspace<Dim>& workspace,
                                      KrylovCollectivePayload& payload) noexcept {
    workspace.append_collective_state_(payload);
  }
  template <int Dim>
  static long local_binding_failure(const KrylovWorkspace<Dim>& workspace,
                                    const PreparedAffineLinearProblem<Dim>& problem,
                                    const KrylovControls<Dim>& controls) noexcept {
    const auto& problem_snapshot = PreparedProblemAccess<Dim>::stored_snapshot(problem);
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

/// Own the gap between local atomic reservation and a fully materialized invocation.  In
/// particular, MPI failures in the control-plane consensus must not leave either the workspace or
/// the prepared problem permanently reserved.
template <int Dim>
class PendingPreparedKrylovReservations final {
 public:
  PendingPreparedKrylovReservations(const PreparedAffineLinearProblem<Dim>& problem,
                                    KrylovWorkspace<Dim>& workspace, bool problem_reserved,
                                    bool workspace_reserved) noexcept
      : problem_(problem_reserved ? &problem : nullptr),
        workspace_(workspace_reserved ? &workspace : nullptr) {}
  PendingPreparedKrylovReservations(const PendingPreparedKrylovReservations&) = delete;
  PendingPreparedKrylovReservations& operator=(const PendingPreparedKrylovReservations&) = delete;
  ~PendingPreparedKrylovReservations() { reset(); }

  void transfer_to_invocation() noexcept {
    problem_ = nullptr;
    workspace_ = nullptr;
  }

 private:
  void reset() noexcept {
    if (workspace_ != nullptr)
      KrylovWorkspaceAccess::release_solve(*workspace_);
    if (problem_ != nullptr)
      PreparedProblemAccess<Dim>::release_use(*problem_);
  }

  const PreparedAffineLinearProblem<Dim>* problem_ = nullptr;
  KrylovWorkspace<Dim>* workspace_ = nullptr;
};

inline bool finite(Real value) {
  return std::isfinite(static_cast<double>(value));
}

template <int Dim>
inline void reduce_batched_inner_products(const PreparedAffineLinearProblem<Dim>& problem,
                                          KrylovWorkspace<Dim>& workspace, double* values,
                                          int count, const char* quantity) {
  reduce_prepared_vector_values_inplace(
      problem.vector_distribution(), values, count,
      KrylovWorkspaceAccess::distribution_reduction_data(workspace),
      KrylovWorkspaceAccess::distribution_reduction_size(workspace), quantity,
      KrylovWorkspaceAccess::execution_lane(workspace));
}

template <int Dim>
inline Real workspace_inner_product(const PreparedAffineLinearProblem<Dim>& problem,
                                    KrylovWorkspace<Dim>& workspace, const MultiFab<Dim>& left,
                                    const MultiFab<Dim>& right) {
  return PreparedProblemAccess<Dim>::inner_product(
      problem, left, right, KrylovWorkspaceAccess::metric_reduction_scratch(workspace),
      KrylovWorkspaceAccess::execution_lane(workspace));
}

template <int Dim>
inline Real workspace_residual_norm(const PreparedAffineLinearProblem<Dim>& problem,
                                    KrylovWorkspace<Dim>& workspace, const MultiFab<Dim>& value) {
  return PreparedProblemAccess<Dim>::residual_norm(
      problem, value, KrylovWorkspaceAccess::metric_reduction_scratch(workspace),
      KrylovWorkspaceAccess::execution_lane(workspace));
}

template <int Dim>
inline void require_exact_scientific_boundary(const PreparedAffineLinearProblem<Dim>& problem,
                                              KrylovWorkspace<Dim>& workspace,
                                              const MultiFab<Dim>& value, const char* where) {
  problem.vector_distribution().require_exact_values(
      value, KrylovWorkspaceAccess::distribution_validation_scratch(workspace), where,
      KrylovWorkspaceAccess::execution_lane(workspace));
}

template <int Dim>
inline void workspace_apply_linear(const PreparedAffineLinearProblem<Dim>& problem,
                                   KrylovWorkspace<Dim>& workspace, MultiFab<Dim>& out,
                                   const MultiFab<Dim>& direction, Real equation_scale) {
  const PreparedApplyResult status = PreparedProblemAccess<Dim>::apply_linear(
      problem, KrylovWorkspaceAccess::operator_session(workspace), out, direction, equation_scale);
  KrylovWorkspaceAccess::latch_provider_apply_status(workspace, status);
  // Every rank still executes every provider callback to preserve its collective trace. Re-publish
  // the sticky local failure only after the callback, so a downstream finite-writing provider can
  // never erase it before the next scientific/control gate.
  KrylovWorkspaceAccess::republish_provider_apply_failure(workspace, out);
}

template <int Dim>
inline void workspace_apply_preconditioner(const PreparedAffineLinearProblem<Dim>& problem,
                                           KrylovWorkspace<Dim>& workspace, MultiFab<Dim>& out,
                                           const MultiFab<Dim>& in) {
  const PreparedApplyResult status = PreparedProblemAccess<Dim>::apply_preconditioner(
      problem, KrylovWorkspaceAccess::preconditioner_session(workspace),
      KrylovWorkspaceAccess::preconditioner_constant(workspace), out, in);
  KrylovWorkspaceAccess::latch_provider_apply_status(workspace, status);
  KrylovWorkspaceAccess::republish_provider_apply_failure(workspace, out);
}

template <int Dim>
inline void workspace_true_residual(const PreparedAffineLinearProblem<Dim>& problem,
                                    KrylovWorkspace<Dim>& workspace, MultiFab<Dim>& out,
                                    const MultiFab<Dim>& rhs, const MultiFab<Dim>& iterate) {
  const PreparedApplyResult status = PreparedProblemAccess<Dim>::true_residual_physical(
      problem, KrylovWorkspaceAccess::operator_session(workspace), out, rhs, iterate);
  KrylovWorkspaceAccess::latch_provider_apply_status(workspace, status);
  KrylovWorkspaceAccess::republish_provider_apply_failure(workspace, out);
}

template <int Dim, class RightAt>
inline bool repair_nonfinite_batched_inner_products(const PreparedAffineLinearProblem<Dim>& problem,
                                                    KrylovWorkspace<Dim>& workspace,
                                                    const MultiFab<Dim>& left, double* reduced,
                                                    int count, int basis_extent,
                                                    RightAt&& right_at) {
  bool needs_repair = false;
  for (int index = 0; index < count; ++index)
    needs_repair = needs_repair || !finite(static_cast<Real>(reduced[index]));
  if (!needs_repair)
    return true;

  const std::size_t width = KrylovWorkspaceAccess::metric_robust_payload_width(workspace);
  double* payload = KrylovWorkspaceAccess::gmres_robust_reduction_data(workspace, basis_extent);
  const std::size_t payload_count = static_cast<std::size_t>(count) * width;
  std::fill_n(payload, payload_count, 0.0);
  for (int index = 0; index < count; ++index) {
    if (finite(static_cast<Real>(reduced[index])))
      continue;
    PreparedProblemAccess<Dim>::local_robust_inner_product_payload(
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
        static_cast<double>(PreparedProblemAccess<Dim>::inner_product_from_global_robust_payload(
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
  if (finite(product) && (product != Real(0) || left == Real(0) || right == Real(0)))
    return ScaledScalar::from(product);
  return ScaledScalar::product(ScaledScalar::from(left), ScaledScalar::from(right));
}

inline ScaledScalar scaled_quotient(Real numerator, Real denominator) {
  const Real quotient = numerator / denominator;
  if (finite(quotient) && (quotient != Real(0) || numerator == Real(0)))
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

template <int Dim>
inline void validate_controls(const KrylovControls<Dim>& controls) {
  const KrylovMethodValidation validation = controls.method.validate_controls(
      KrylovMethodControls{controls.rel_tol, controls.abs_tol, controls.max_iterations});
  if (!validation.accepted())
    throw std::invalid_argument("prepared Krylov provider '" +
                                std::string(controls.method.identity()) +
                                "' rejected controls: " + std::string(validation.reason));
}

template <int Dim>
inline long controls_failure(const KrylovControls<Dim>& controls) noexcept {
  if (!controls.method)
    return 19;
  return controls.method
                 .validate_controls(KrylovMethodControls{controls.rel_tol, controls.abs_tol,
                                                         controls.max_iterations})
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

template <int Dim>
inline void append_controls(KrylovCollectivePayload& payload,
                            const KrylovControls<Dim>& controls) noexcept {
  payload.append(controls.method.fingerprint());
  payload.append(std::bit_cast<std::uint64_t>(controls.rel_tol));
  payload.append(std::bit_cast<std::uint64_t>(controls.abs_tol));
  payload.append(controls.max_iterations);
}

template <int Dim>
inline void append_field_shape(KrylovCollectivePayload& payload,
                               const MultiFab<Dim>& field) noexcept {
  payload.append(field.ncomp());
  payload.append(field.ghosts());
}

template <int Dim>
inline void collective_solve_preflight(const PreparedAffineLinearProblem<Dim>& problem,
                                       KrylovWorkspace<Dim>& workspace,
                                       const MultiFab<Dim>& iterate, const MultiFab<Dim>& rhs,
                                       const KrylovControls<Dim>& controls,
                                       const ExecutionLane& control_lane) {
  KrylovCollectivePayload payload;
  long local_failure = PreparedProblemAccess<Dim>::append_collective_state(problem, payload);
  KrylovWorkspaceAccess::append_collective_state(workspace, payload);
  append_controls(payload, controls);
  append_field_shape(payload, iterate);
  append_field_shape(payload, rhs);
  payload.append(static_cast<std::uint8_t>(iterate.shares_storage_with(rhs)));

  if (local_failure == 0)
    local_failure = controls_failure(controls);
  if (local_failure == 0 && (!PreparedProblemAccess<Dim>::matches_vector_space(problem, iterate) ||
                             iterate.ghosts() != problem.footprint().input_ghosts))
    local_failure = 20;
  if (local_failure == 0 && !PreparedProblemAccess<Dim>::matches_vector_space(problem, rhs))
    local_failure = 21;
  if (local_failure == 0 && iterate.shares_storage_with(rhs))
    local_failure = 22;
  if (local_failure == 0)
    local_failure = KrylovWorkspaceAccess::local_binding_failure(workspace, problem, controls);
  const KrylovMethodProblemFacts<Dim> method_facts{
      problem.properties(),          problem.footprint(),
      problem.vector_distribution(), problem.metric().robust_payload_width(),
      problem.has_nullspace(),       problem.has_preconditioner()};
  const KrylovMethodValidation problem_validation = controls.method.validate_problem(method_facts);
  payload.append(problem_validation.code);
  if (local_failure == 0 && !problem_validation.accepted())
    local_failure = 28;

  // This is the only solve-entry collective gate. It is fixed-size, stack-only and precedes every
  // norm, nullspace, halo or Krylov reduction. Exact min/max consensus catches valid-but-different
  // contracts; the error reduction converts a rank-local invalid contract into one exception.
  const bool agrees = collective_payload_agrees(payload, control_lane);
  const long collective_failure = all_reduce_max(local_failure, control_lane);
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
      throw std::invalid_argument(
          "prepared Krylov provider '" + std::string(controls.method.identity()) +
          "' rejected the solve request: " + std::string(local_validation.reason));
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
  const PreparedVectorDistribution<Dim>& distribution = problem.vector_distribution();
  char* storage = KrylovWorkspaceAccess::distribution_validation_data(workspace);
  const std::size_t storage_size = KrylovWorkspaceAccess::distribution_validation_size(workspace);
  distribution.require_exact_values(iterate, std::span<char>(storage, storage_size),
                                    "solve_prepared_affine(iterate)", control_lane);
  distribution.require_exact_values(rhs, std::span<char>(storage, storage_size),
                                    "solve_prepared_affine(rhs)", control_lane);
}

inline Real reference_denominator(Real reference) {
  return reference > Real(0) ? reference : Real(1);
}

template <int Dim>
inline bool provider_solve_report_agrees(const SolveReport& report,
                                         KrylovWorkspace<Dim>& workspace) {
  return KrylovWorkspaceAccess::provider_report_agrees(workspace, report);
}

struct SolveNormalization {
  Real reference = Real(0);
  Real scale = Real(1);
  Real normalized_threshold = Real(0);
  Real physical_threshold = Real(0);
};

template <int Dim>
inline SolveNormalization make_normalization(Real reference, Real scale,
                                             const KrylovControls<Dim>& controls) {
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

template <int Dim>
inline Real physical_stopping_threshold(Real reference, const KrylovControls<Dim>& controls) {
  const Real relative =
      reference > Real(0) ? rescale_product(controls.rel_tol, reference, Real(1)) : Real(0);
  return std::max(relative, controls.abs_tol);
}

/// Apply the public stopping contract in its authored dimensionless form.  The scaled physical
/// threshold remains the recurrence fast path, but one final division avoids a false failure when
/// the rounded product ``rel_tol * reference`` lands one ULP below the equivalently rounded
/// ``residual / reference`` comparison reported to users.
template <int Dim>
inline bool satisfies_stopping_controls(Real residual, Real reference,
                                        const KrylovControls<Dim>& controls) {
  if (!finite(residual) || residual < Real(0))
    return false;
  if (residual <= controls.abs_tol)
    return true;
  return controls.rel_tol > Real(0) && reference > Real(0) &&
         residual / reference <= controls.rel_tol;
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
  // Preserve the provider's measured values until the common SolveReport authority sees the final
  // status. `mark_failed(kInvalidEvaluation)` alone canonicalizes unavailable evidence; every other
  // status retains its exact measurements so the publication boundary still rejects malformed
  // provider reports.
  set_report_physical_residuals(result, normalization, physical_residual);
  if (status == SolveStatus::kSolved)
    result.mark_solved();
  else
    result.mark_failed(status);
  return result;
}

inline SolveAction solve_action(const PreparedApplyResult& failure) noexcept {
  return failure.action == PreparedApplyFailureAction::kRejectAttempt ? SolveAction::kRejectAttempt
                                                                      : SolveAction::kFailRun;
}

inline std::string prepared_apply_failure_reason(const PreparedApplyResult& failure,
                                                 std::string_view context) {
  std::string reason(context);
  if (failure.kind != PreparedApplyFailureKind::kFluxEvaluation) {
    reason += ": prepared callback raised an unknown exception";
    return reason;
  }

  reason += ": numerical flux evaluation ";
  reason += evaluation_status_name(failure.evaluation_status);
  if (!failure.phase().empty()) {
    reason += " during ";
    reason.append(failure.phase());
    if (failure.phase_truncated)
      reason += "[truncated]";
  }
  reason += ": reason_code=0x";
  constexpr char digits[] = "0123456789abcdef";
  for (int shift = 28; shift >= 0; shift -= 4)
    reason.push_back(digits[(failure.reason_code >> shift) & 0xfu]);
  return reason;
}

inline SolveReport prepared_apply_failure_report(const SolveNormalization& normalization,
                                                 const PreparedApplyResult& failure,
                                                 std::string_view context) {
  SolveReport result = report_physical(normalization, std::numeric_limits<Real>::quiet_NaN(), 0,
                                       SolveStatus::kInvalidEvaluation);
  result.mark_failed(SolveStatus::kInvalidEvaluation, solve_action(failure),
                     prepared_apply_failure_reason(failure, context));
  return result;
}

template <int Dim>
inline Real physical_true_residual_norm(const PreparedAffineLinearProblem<Dim>& problem,
                                        KrylovWorkspace<Dim>& workspace, MultiFab<Dim>& scratch,
                                        const MultiFab<Dim>& rhs, const MultiFab<Dim>& iterate) {
  workspace_true_residual(problem, workspace, scratch, rhs, iterate);
  return workspace_residual_norm(problem, workspace, scratch);
}

struct ResidualMeasurement {
  Real physical = Real(0);
  Real normalized = Real(0);
};

/// Materialize the scientific residual in physical units and measure it scale-safely.  The field is
/// deliberately left in physical units: a caller that is actually going to restart a recurrence
/// can then choose its next cycle scale from this authoritative measurement, without first losing a
/// representable component through division by the old cycle scale.
template <int Dim>
inline ResidualMeasurement physical_true_residual_measurement(
    const PreparedAffineLinearProblem<Dim>& problem, KrylovWorkspace<Dim>& workspace,
    MultiFab<Dim>& scratch, const MultiFab<Dim>& rhs, const MultiFab<Dim>& iterate) {
  const Real physical = physical_true_residual_norm(problem, workspace, scratch, rhs, iterate);
  return {physical, std::numeric_limits<Real>::quiet_NaN()};
}

/// Rebase only the disposable recurrence cycle.  The authored reference and physical stopping
/// threshold remain those in `report_normalization`; reports therefore cannot change meaning when
/// an extreme residual forces a numerical restart.  This operation consumes a residual field that
/// is still in physical units and needs no additional reduction.
template <int Dim>
inline void rebase_cycle_residual(MultiFab<Dim>& physical_residual,
                                  ResidualMeasurement& measurement,
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

/// Recursive BiCGStab residuals lose roughly one unit of relative accuracy per recurrence update.
/// Once a cycle has reduced its residual by sqrt(epsilon), replace it with an authoritative
/// b-A(x) measurement before round-off can dominate the remaining convergence.  The trigger is
/// dimensionless, independent of the equation scaling and authored tolerance, and costs at most
/// one extra matvec per roughly eight decimal digits of reduction in a restarted cycle.
inline bool needs_reliable_residual_replacement(Real normalized_norm, Real cycle_peak) {
  return normalized_norm > Real(0) && cycle_peak > Real(0) &&
         normalized_norm / cycle_peak <= std::sqrt(std::numeric_limits<Real>::epsilon());
}

/// Publish a terminal recurrence outcome without repeating the public wrapper's mandatory
/// provider-independent true-residual evaluation. `measurement` is the last authoritative
/// physical residual observed inside the method and keeps the candidate structurally valid; the
/// wrapper overwrites all residual fields and validates an existing provider success after its own
/// matvec. It never upgrades a provider failure after iteration exhaustion or algebraic breakdown.
inline SolveReport terminal_candidate_report(const SolveNormalization& normalization,
                                             const ResidualMeasurement& measurement, int iterations,
                                             SolveStatus status, std::string_view reason = {}) {
  SolveReport report = report_physical(normalization, measurement.physical, iterations, status);
  if (!reason.empty())
    report.reason.assign(reason);
  return report;
}

/// Remove one arbitrary scalar normalization from a prepared linear preconditioner.  Krylov
/// methods are invariant to M -> cM, but their raw dot products are not representable for finite
/// c=1e+/-200.  The first nonzero preconditioned direction fixes one solve-local positive scale;
/// every later application reuses it, so the mathematical preconditioner changes only by a single
/// constant factor and no allocation or per-iteration norm reduction is introduced.
template <int Dim>
inline Real apply_scaled_preconditioner(const PreparedAffineLinearProblem<Dim>& problem,
                                        MultiFab<Dim>& out, const MultiFab<Dim>& in,
                                        KrylovWorkspace<Dim>& workspace, Real& solve_scale) {
  workspace_apply_preconditioner(problem, workspace, out, in);
  if (solve_scale == Real(0))
    solve_scale = PreparedFieldAlgebra::max_abs(
        out, problem.vector_distribution(),
        std::span<double>(KrylovWorkspaceAccess::distribution_reduction_data(workspace),
                          KrylovWorkspaceAccess::distribution_reduction_size(workspace)),
        KrylovWorkspaceAccess::execution_lane(workspace));
  if (!finite(solve_scale) || !(solve_scale > Real(0)))
    return solve_scale;
  PreparedFieldAlgebra::divide(out, solve_scale);
  return solve_scale;
}

template <int Dim>
inline MultiFab<Dim>& initial_residual_field(KrylovWorkspace<Dim>& workspace,
                                             const KrylovControls<Dim>&) {
  return KrylovWorkspaceAccess::field(workspace,
                                      KrylovWorkspaceAccess::initial_residual_field(workspace));
}

template <int Dim>
inline SolveReport solve_richardson(const PreparedAffineLinearProblem<Dim>& problem,
                                    KrylovWorkspace<Dim>& workspace, MultiFab<Dim>& iterate,
                                    const MultiFab<Dim>& rhs, const KrylovControls<Dim>& controls,
                                    Real relaxation, const SolveNormalization& normalization,
                                    ResidualMeasurement measurement) {
  MultiFab<Dim>& residual = KrylovWorkspaceAccess::field(workspace, 1);
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
    measurement = physical_true_residual_measurement(problem, workspace, residual, rhs, iterate);
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

template <int Dim>
inline SolveReport solve_cg(const PreparedAffineLinearProblem<Dim>& problem,
                            KrylovWorkspace<Dim>& workspace, MultiFab<Dim>& iterate,
                            const MultiFab<Dim>& rhs, const KrylovControls<Dim>& controls,
                            const SolveNormalization& normalization,
                            ResidualMeasurement measurement) {
  MultiFab<Dim>& residual = KrylovWorkspaceAccess::field(workspace, 1);
  MultiFab<Dim>& direction = KrylovWorkspaceAccess::field(workspace, 2);
  MultiFab<Dim>& applied = KrylovWorkspaceAccess::field(workspace, 3);
  SolveNormalization cycle_normalization = normalization;
  PreparedFieldAlgebra::copy(direction, residual);
  Real squared = workspace_inner_product(problem, workspace, residual, residual);
  for (int completed = 0; completed < controls.max_iterations; ++completed) {
    const int iteration = completed + 1;
    workspace_apply_linear(problem, workspace, applied, direction, cycle_normalization.scale);
    const Real curvature = workspace_inner_product(problem, workspace, direction, applied);
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
    Real next_squared = workspace_inner_product(problem, workspace, residual, residual);
    measurement.normalized =
        next_squared >= Real(0) ? std::sqrt(next_squared) : std::numeric_limits<Real>::quiet_NaN();
    if (!finite(measurement.normalized))
      return terminal_candidate_report(normalization, measurement, iteration,
                                       SolveStatus::kInvalidEvaluation);
    const bool estimate_reached =
        measurement.normalized <= cycle_normalization.normalized_threshold;
    if (iteration == controls.max_iterations && estimate_reached)
      // The common wrapper owns the authoritative b-A(x) confirmation. Publishing this
      // recurrence-qualified candidate avoids a redundant matvec while still preventing an
      // exhausted, unconverged provider report from being upgraded after the fact.
      return terminal_candidate_report(normalization, measurement, iteration, SolveStatus::kSolved);
    if (iteration == controls.max_iterations)
      return terminal_candidate_report(normalization, measurement, iteration,
                                       SolveStatus::kIterationLimit);
    bool restart_recurrence = false;
    if (estimate_reached || needs_extreme_recurrence_rebase(measurement.normalized)) {
      ResidualMeasurement confirmed =
          physical_true_residual_measurement(problem, workspace, applied, rhs, iterate);
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

template <int Dim>
inline SolveReport solve_bicgstab(const PreparedAffineLinearProblem<Dim>& problem,
                                  KrylovWorkspace<Dim>& workspace, MultiFab<Dim>& iterate,
                                  const MultiFab<Dim>& rhs, const KrylovControls<Dim>& controls,
                                  const SolveNormalization& normalization,
                                  ResidualMeasurement measurement) {
  MultiFab<Dim>& residual = KrylovWorkspaceAccess::field(workspace, 1);
  MultiFab<Dim>& shadow = KrylovWorkspaceAccess::field(workspace, 2);
  MultiFab<Dim>& direction = KrylovWorkspaceAccess::field(workspace, 3);
  MultiFab<Dim>& applied = KrylovWorkspaceAccess::field(workspace, 4);
  MultiFab<Dim>& intermediate = KrylovWorkspaceAccess::field(workspace, 5);
  MultiFab<Dim>& second_applied = KrylovWorkspaceAccess::field(workspace, 6);
  MultiFab<Dim>& prepared_direction =
      problem.has_preconditioner() ? KrylovWorkspaceAccess::field(workspace, 7) : direction;
  MultiFab<Dim>& prepared_intermediate =
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
  Real recurrence_peak = Real(1);

  for (int completed = 0; completed < controls.max_iterations; ++completed) {
    const int iteration = completed + 1;
    const Real rho = workspace_inner_product(problem, workspace, shadow, residual);
    if (!finite(rho))
      return terminal_candidate_report(normalization, measurement, iteration - 1,
                                       SolveStatus::kInvalidEvaluation);
    if (rho == Real(0))
      return terminal_candidate_report(normalization, measurement, iteration - 1,
                                       SolveStatus::kBreakdown, "BiCGStab rho breakdown");

    if (iteration == 1 || restart_recurrence) {
      PreparedFieldAlgebra::copy(direction, residual);
      restart_recurrence = false;
    } else {
      if (omega.is_zero())
        return terminal_candidate_report(normalization, measurement, iteration - 1,
                                         SolveStatus::kBreakdown,
                                         "BiCGStab recurrence omega breakdown");
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
      const Real scale = apply_scaled_preconditioner(problem, prepared_direction, direction,
                                                     workspace, preconditioner_scale);
      if (!finite(scale) || !(scale > Real(0)))
        return terminal_candidate_report(
            normalization, measurement, iteration - 1,
            finite(scale) ? SolveStatus::kBreakdown : SolveStatus::kInvalidEvaluation,
            finite(scale) ? std::string_view("BiCGStab direction preconditioner breakdown")
                          : std::string_view{});
    }
    workspace_apply_linear(problem, workspace, applied, prepared_direction,
                           cycle_normalization.scale);
    const Real denominator = workspace_inner_product(problem, workspace, shadow, applied);
    if (!finite(denominator))
      return terminal_candidate_report(normalization, measurement, iteration - 1,
                                       SolveStatus::kInvalidEvaluation);
    if (denominator == Real(0))
      return terminal_candidate_report(normalization, measurement, iteration - 1,
                                       SolveStatus::kBreakdown,
                                       "BiCGStab alpha denominator breakdown");
    alpha = scaled_quotient(rho, denominator);
    if (!alpha.is_finite())
      return terminal_candidate_report(normalization, measurement, iteration - 1,
                                       SolveStatus::kInvalidEvaluation);
    ScaledFieldAlgebra::lincomb(intermediate, ScaledScalar::from(Real(1)), residual,
                                ScaledScalar::negated(alpha), applied);

    const Real intermediate_norm = workspace_residual_norm(problem, workspace, intermediate);
    if (!finite(intermediate_norm))
      return terminal_candidate_report(normalization, measurement, iteration - 1,
                                       SolveStatus::kInvalidEvaluation);
    recurrence_peak = std::max(recurrence_peak, intermediate_norm);
    const bool intermediate_estimate_reached =
        intermediate_norm <= cycle_normalization.normalized_threshold;
    if (intermediate_estimate_reached ||
        needs_reliable_residual_replacement(intermediate_norm, recurrence_peak) ||
        needs_extreme_recurrence_rebase(intermediate_norm)) {
      ScaledFieldAlgebra::axpy(iterate, alpha, prepared_direction);
      if (iteration == controls.max_iterations && intermediate_estimate_reached)
        return terminal_candidate_report(normalization, measurement, iteration,
                                         SolveStatus::kSolved);
      if (iteration == controls.max_iterations)
        return terminal_candidate_report(normalization, measurement, iteration,
                                         SolveStatus::kIterationLimit);
      ResidualMeasurement confirmed =
          physical_true_residual_measurement(problem, workspace, second_applied, rhs, iterate);
      if (!finite(confirmed.physical))
        return report_physical(normalization, confirmed.physical, iteration,
                               SolveStatus::kInvalidEvaluation);
      if (confirmed.physical <= normalization.physical_threshold ||
          satisfies_stopping_controls(confirmed.physical, normalization.reference, controls))
        return report_physical(normalization, confirmed.physical, iteration, SolveStatus::kSolved);
      rebase_cycle_residual(second_applied, confirmed, normalization, cycle_normalization);
      PreparedFieldAlgebra::copy(residual, second_applied);
      PreparedFieldAlgebra::copy(shadow, residual);
      measurement = confirmed;
      restart_recurrence = true;
      preconditioner_scale = Real(0);
      recurrence_peak = Real(1);
      rho_previous = rho;
      continue;
    }

    if (problem.has_preconditioner()) {
      const Real scale = apply_scaled_preconditioner(problem, prepared_intermediate, intermediate,
                                                     workspace, preconditioner_scale);
      if (!finite(scale) || !(scale > Real(0)))
        return terminal_candidate_report(
            normalization, measurement, iteration - 1,
            finite(scale) ? SolveStatus::kBreakdown : SolveStatus::kInvalidEvaluation,
            finite(scale) ? std::string_view("BiCGStab intermediate preconditioner breakdown")
                          : std::string_view{});
    }
    workspace_apply_linear(problem, workspace, second_applied, prepared_intermediate,
                           cycle_normalization.scale);
    const Real second_norm_squared =
        workspace_inner_product(problem, workspace, second_applied, second_applied);
    const Real projection =
        workspace_inner_product(problem, workspace, second_applied, intermediate);
    if (!finite(second_norm_squared) || !finite(projection))
      return terminal_candidate_report(normalization, measurement, iteration - 1,
                                       SolveStatus::kInvalidEvaluation);
    if (second_norm_squared <= Real(0)) {
      ScaledFieldAlgebra::axpy(iterate, alpha, prepared_direction);
      return terminal_candidate_report(normalization, measurement, iteration,
                                       SolveStatus::kBreakdown,
                                       "BiCGStab omega denominator breakdown");
    }
    omega = scaled_quotient(projection, second_norm_squared);
    if (!omega.is_finite())
      return terminal_candidate_report(normalization, measurement, iteration - 1,
                                       SolveStatus::kInvalidEvaluation);
    if (omega.is_zero()) {
      ScaledFieldAlgebra::axpy(iterate, alpha, prepared_direction);
      return terminal_candidate_report(normalization, measurement, iteration,
                                       SolveStatus::kBreakdown, "BiCGStab omega breakdown");
    }

    ScaledFieldAlgebra::trilincomb(iterate, ScaledScalar::from(Real(1)), iterate, alpha,
                                   prepared_direction, omega, prepared_intermediate);
    // The BiCGStab recurrence already supplies the next residual. Recomputing b-A(x) here would
    // add a third operator application to every full iteration. Its norm may request an
    // authoritative true-residual confirmation, but can never publish success by itself.
    ScaledFieldAlgebra::lincomb(residual, ScaledScalar::from(Real(1)), intermediate,
                                ScaledScalar::negated(omega), second_applied);
    measurement.normalized = workspace_residual_norm(problem, workspace, residual);
    if (!finite(measurement.normalized))
      return terminal_candidate_report(normalization, measurement, iteration,
                                       SolveStatus::kInvalidEvaluation);
    const bool estimate_reached =
        measurement.normalized <= cycle_normalization.normalized_threshold;
    if (iteration == controls.max_iterations && estimate_reached)
      return terminal_candidate_report(normalization, measurement, iteration, SolveStatus::kSolved);
    if (iteration == controls.max_iterations)
      return terminal_candidate_report(normalization, measurement, iteration,
                                       SolveStatus::kIterationLimit);
    recurrence_peak = std::max(recurrence_peak, measurement.normalized);
    if (estimate_reached ||
        needs_reliable_residual_replacement(measurement.normalized, recurrence_peak) ||
        needs_extreme_recurrence_rebase(measurement.normalized)) {
      ResidualMeasurement confirmed =
          physical_true_residual_measurement(problem, workspace, second_applied, rhs, iterate);
      if (!finite(confirmed.physical))
        return report_physical(normalization, confirmed.physical, iteration,
                               SolveStatus::kInvalidEvaluation);
      if (confirmed.physical <= normalization.physical_threshold ||
          satisfies_stopping_controls(confirmed.physical, normalization.reference, controls))
        return report_physical(normalization, confirmed.physical, iteration, SolveStatus::kSolved);
      rebase_cycle_residual(second_applied, confirmed, normalization, cycle_normalization);
      // Recursive drift or a subnormal recurrence requested an authoritative replacement. Rebase
      // from that scientific residual before continuing.
      PreparedFieldAlgebra::copy(residual, second_applied);
      measurement = confirmed;
      PreparedFieldAlgebra::copy(shadow, residual);
      restart_recurrence = true;
      preconditioner_scale = Real(0);
      recurrence_peak = Real(1);
    }
    rho_previous = rho;
  }
  return terminal_candidate_report(normalization, measurement, controls.max_iterations,
                                   SolveStatus::kIterationLimit);
}

template <int Dim>
inline bool set_scaled_h(KrylovWorkspace<Dim>& workspace, int row, int column,
                         const ScaledScalar& value, int basis_extent) {
  Real materialized = Real(0);
  if (!value.is_finite() || !value.try_materialize(materialized))
    return false;
  KrylovWorkspaceAccess::h(workspace, row, column, basis_extent) = materialized;
  KrylovWorkspaceAccess::scaled_h(workspace, row, column, basis_extent) = value;
  return true;
}

template <int Dim>
inline bool set_scaled_h(KrylovWorkspace<Dim>& workspace, int row, int column, Real value,
                         int basis_extent) {
  return set_scaled_h(workspace, row, column, ScaledScalar::from(value), basis_extent);
}

template <int Dim>
inline void set_scaled_rotated_rhs(KrylovWorkspace<Dim>& workspace, int index,
                                   const ScaledScalar& value, int basis_extent) {
  KrylovWorkspaceAccess::scaled_rotated_rhs(workspace, index, basis_extent) = value;
  Real materialized = Real(0);
  KrylovWorkspaceAccess::rotated_rhs(workspace, index, basis_extent) =
      value.try_materialize(materialized) ? materialized : std::numeric_limits<Real>::quiet_NaN();
}

template <int Dim>
inline void set_scaled_solution_coefficient(KrylovWorkspace<Dim>& workspace, int index,
                                            const ScaledScalar& value, int basis_extent) {
  KrylovWorkspaceAccess::scaled_solution_coefficient(workspace, index, basis_extent) = value;
  Real materialized = Real(0);
  KrylovWorkspaceAccess::solution_coefficient(workspace, index, basis_extent) =
      value.try_materialize(materialized) ? materialized : std::numeric_limits<Real>::quiet_NaN();
}

template <int Dim>
inline void reset_gmres_scalars(KrylovWorkspace<Dim>& workspace, int restart) {
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

template <int Dim>
inline bool solve_gmres_upper(KrylovWorkspace<Dim>& workspace, int dimension, int basis_extent) {
  for (int row = dimension - 1; row >= 0; --row) {
    ScaledScalar value = KrylovWorkspaceAccess::scaled_rotated_rhs(workspace, row, basis_extent);
    for (int column = row + 1; column < dimension; ++column)
      value = scaled_difference(
          value,
          scaled_product(
              KrylovWorkspaceAccess::scaled_h(workspace, row, column, basis_extent),
              KrylovWorkspaceAccess::scaled_solution_coefficient(workspace, column, basis_extent)));
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

template <int Dim>
inline SolveReport solve_gmres(const PreparedAffineLinearProblem<Dim>& problem,
                               KrylovWorkspace<Dim>& workspace, MultiFab<Dim>& iterate,
                               const MultiFab<Dim>& rhs, const KrylovControls<Dim>& controls,
                               int restart, const SolveNormalization& normalization,
                               ResidualMeasurement measurement) {
  const auto basis = [&workspace](int index) -> MultiFab<Dim>& {
    return KrylovWorkspaceAccess::field(workspace, static_cast<std::size_t>(index) + 1u);
  };
  MultiFab<Dim>& applied_or_residual =
      KrylovWorkspaceAccess::field(workspace, static_cast<std::size_t>(restart) + 2u);
  MultiFab<Dim>* prepared_vector =
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
    MultiFab<Dim>* initial_vector = &applied_or_residual;
    if (prepared_vector != nullptr) {
      const Real scale = apply_scaled_preconditioner(problem, *prepared_vector, applied_or_residual,
                                                     workspace, preconditioner_scale);
      if (!finite(scale) || !(scale > Real(0)))
        return report_physical(
            normalization, measurement.physical, iterations,
            finite(scale) ? SolveStatus::kBreakdown : SolveStatus::kInvalidEvaluation);
      initial_vector = prepared_vector;
    }
    const Real beta = workspace_residual_norm(problem, workspace, *initial_vector);
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
      workspace_apply_linear(problem, workspace, applied_or_residual, basis(column),
                             cycle_normalization.scale);
      MultiFab<Dim>* arnoldi_vector = &applied_or_residual;
      if (prepared_vector != nullptr) {
        const Real scale = apply_scaled_preconditioner(
            problem, *prepared_vector, applied_or_residual, workspace, preconditioner_scale);
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
            PreparedProblemAccess<Dim>::local_inner_product(problem, *arnoldi_vector, basis(row)));
      reductions[column + 1] = static_cast<double>(PreparedProblemAccess<Dim>::local_inner_product(
          problem, *arnoldi_vector, *arnoldi_vector));
      reduce_batched_inner_products(problem, workspace, reductions, column + 2,
                                    "prepared GMRES Arnoldi projections");

      bool finite_column = repair_nonfinite_batched_inner_products(
          problem, workspace, *arnoldi_vector, reductions, column + 1, restart,
          [&basis](int row) -> const MultiFab<Dim>& { return basis(row); });

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
                              ? workspace_residual_norm(problem, workspace, *arnoldi_vector)
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
          reductions[row] = static_cast<double>(PreparedProblemAccess<Dim>::local_inner_product(
              problem, *arnoldi_vector, basis(row)));
        reduce_batched_inner_products(problem, workspace, reductions, column + 1,
                                      "prepared GMRES DGKS projections");
        finite_column = finite_column &&
                        repair_nonfinite_batched_inner_products(
                            problem, workspace, *arnoldi_vector, reductions, column + 1, restart,
                            [&basis](int row) -> const MultiFab<Dim>& { return basis(row); });
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
        arnoldi_norm = finite_column ? workspace_residual_norm(problem, workspace, *arnoldi_vector)
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
        const ScaledScalar rotated_first = scaled_sum(
            scaled_product(KrylovWorkspaceAccess::cosine(workspace, row, restart), first),
            scaled_product(KrylovWorkspaceAccess::sine(workspace, row, restart), second));
        const ScaledScalar rotated_second = scaled_sum(
            scaled_product(ScaledScalar::negated(ScaledScalar::from(
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
      const Real subdiagonal = KrylovWorkspaceAccess::h(workspace, column + 1, column, restart);
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
      set_scaled_rotated_rhs(
          workspace, column + 1,
          scaled_product(ScaledScalar::negated(ScaledScalar::from(
                             KrylovWorkspaceAccess::sine(workspace, column, restart))),
                         prior_rhs),
          restart);
      set_scaled_rotated_rhs(workspace, column,
                             scaled_product(ScaledScalar::from(KrylovWorkspaceAccess::cosine(
                                                workspace, column, restart)),
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
          iterate, KrylovWorkspaceAccess::scaled_solution_coefficient(workspace, column, restart),
          basis(column));

    if (iterations == controls.max_iterations && estimate_reached)
      return terminal_candidate_report(normalization, measurement, iterations,
                                       SolveStatus::kSolved);
    if (iterations == controls.max_iterations)
      return terminal_candidate_report(normalization, measurement, iterations,
                                       SolveStatus::kIterationLimit);
    measurement =
        physical_true_residual_measurement(problem, workspace, applied_or_residual, rhs, iterate);
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
template <int Dim>
class PreparedKrylovSolveContext {
 public:
  [[nodiscard]] const KrylovControls<Dim>& controls() const noexcept { return controls_; }
  [[nodiscard]] const LinearOperatorProperties& operator_properties() const noexcept {
    return problem_.properties();
  }
  [[nodiscard]] bool has_nullspace() const noexcept { return problem_.has_nullspace(); }
  [[nodiscard]] bool has_preconditioner() const noexcept { return problem_.has_preconditioner(); }
  [[nodiscard]] const PreparedVectorDistribution<Dim>& vector_distribution() const noexcept {
    return problem_.vector_distribution();
  }
  /// Communicator authority for every provider collective. Providers must never fall back to
  /// MPI_COMM_WORLD: separate prepared invocations may execute concurrently in different lanes.
  [[nodiscard]] const ExecutionLane& execution_lane() const noexcept {
    return detail::KrylovWorkspaceAccess::execution_lane(workspace_);
  }
  [[nodiscard]] Real equation_scale() const noexcept { return normalization_.scale; }
  [[nodiscard]] Real reference_norm() const noexcept { return normalization_.reference; }
  [[nodiscard]] Real physical_threshold() const noexcept {
    return normalization_.physical_threshold;
  }
  [[nodiscard]] Real initial_physical_residual() const noexcept {
    return initial_measurement_.physical;
  }

  [[nodiscard]] MultiFab<Dim>& iterate() noexcept { return iterate_; }
  [[nodiscard]] const MultiFab<Dim>& rhs() const noexcept { return rhs_; }
  [[nodiscard]] MultiFab<Dim>& field(std::size_t index) {
    return detail::KrylovWorkspaceAccess::field(workspace_, index);
  }
  [[nodiscard]] MultiFab<Dim>& initial_residual() {
    return field(detail::KrylovWorkspaceAccess::initial_residual_field(workspace_));
  }
  [[nodiscard]] Real& real_value(std::size_t index) { return workspace_.real_value(index); }
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

  void zero(MultiFab<Dim>& value) const { detail::PreparedFieldAlgebra::zero(value); }
  void copy(MultiFab<Dim>& out, const MultiFab<Dim>& in) const {
    detail::PreparedFieldAlgebra::copy(out, in);
  }
  void divide(MultiFab<Dim>& value, Real denominator) const {
    detail::PreparedFieldAlgebra::divide(value, denominator);
  }
  void axpy(MultiFab<Dim>& out, Real coefficient, const MultiFab<Dim>& in) const {
    detail::PreparedFieldAlgebra::axpy(out, coefficient, in);
  }
  void add_physical_direction(MultiFab<Dim>& out, Real coefficient,
                              const MultiFab<Dim>& normalized_direction) const {
    detail::ScaledFieldAlgebra::axpy(out, detail::scaled_product(coefficient, normalization_.scale),
                                     normalized_direction);
  }
  void apply_linear(MultiFab<Dim>& out, const MultiFab<Dim>& direction, Real equation_scale) const {
    detail::workspace_apply_linear(problem_, workspace_, out, direction, equation_scale);
  }
  void apply_linear(MultiFab<Dim>& out, const MultiFab<Dim>& direction) const {
    apply_linear(out, direction, normalization_.scale);
  }
  void apply_preconditioner(MultiFab<Dim>& out, const MultiFab<Dim>& in) const {
    detail::workspace_apply_preconditioner(problem_, workspace_, out, in);
  }
  [[nodiscard]] Real inner_product(const MultiFab<Dim>& left, const MultiFab<Dim>& right) const {
    return detail::workspace_inner_product(problem_, workspace_, left, right);
  }
  [[nodiscard]] Real residual_norm(const MultiFab<Dim>& value) const {
    return detail::workspace_residual_norm(problem_, workspace_, value);
  }
  void local_robust_inner_product_payload(const MultiFab<Dim>& left, const MultiFab<Dim>& right,
                                          std::span<double> payload) const {
    detail::PreparedProblemAccess<Dim>::local_robust_inner_product_payload(problem_, left, right,
                                                                           payload);
  }
  [[nodiscard]] Real inner_product_from_global_robust_payload(
      std::span<const double> payload) const {
    return detail::PreparedProblemAccess<Dim>::inner_product_from_global_robust_payload(problem_,
                                                                                        payload);
  }
  void reduce_inner_products(double* values, int count, const char* quantity) {
    detail::reduce_batched_inner_products(problem_, workspace_, values, count, quantity);
  }
  [[nodiscard]] Real true_residual_norm(MultiFab<Dim>& scratch) const {
    return detail::physical_true_residual_norm(problem_, workspace_, scratch, rhs_, iterate_);
  }
  [[nodiscard]] SolveReport report(Real physical_residual, int iterations,
                                   SolveStatus status) const {
    return detail::report_physical(normalization_, physical_residual, iterations, status);
  }

 private:
  PreparedKrylovSolveContext(const PreparedAffineLinearProblem<Dim>& problem,
                             KrylovWorkspace<Dim>& workspace, MultiFab<Dim>& iterate,
                             const MultiFab<Dim>& rhs, const KrylovControls<Dim>& controls,
                             detail::SolveNormalization normalization,
                             detail::ResidualMeasurement initial_measurement)
      : problem_(problem),
        workspace_(workspace),
        iterate_(iterate),
        rhs_(rhs),
        controls_(controls),
        normalization_(normalization),
        initial_measurement_(initial_measurement) {}

  friend class detail::CgKrylovMethodProvider<Dim>;
  friend class detail::BicgstabKrylovMethodProvider<Dim>;
  friend class detail::GmresKrylovMethodProvider<Dim>;
  friend class detail::RichardsonKrylovMethodProvider<Dim>;
  friend struct detail::PreparedKrylovInvocationAccess;
  friend SolveReport detail::solve_prepared_affine_in_place<Dim>(
      const PreparedAffineLinearProblem<Dim>&, KrylovWorkspace<Dim>&, MultiFab<Dim>&,
      const MultiFab<Dim>&, const KrylovControls<Dim>&);

  const PreparedAffineLinearProblem<Dim>& problem_;
  KrylovWorkspace<Dim>& workspace_;
  MultiFab<Dim>& iterate_;
  const MultiFab<Dim>& rhs_;
  const KrylovControls<Dim>& controls_;
  detail::SolveNormalization normalization_;
  detail::ResidualMeasurement initial_measurement_;
};

template <int Dim>
inline SolveReport detail::CgKrylovMethodProvider<Dim>::solve(
    PreparedKrylovSolveContext<Dim>& context, const PreparedProviderOptions&) const {
  return detail::solve_cg(context.problem_, context.workspace_, context.iterate_, context.rhs_,
                          context.controls_, context.normalization_, context.initial_measurement_);
}

template <int Dim>
inline SolveReport detail::BicgstabKrylovMethodProvider<Dim>::solve(
    PreparedKrylovSolveContext<Dim>& context, const PreparedProviderOptions&) const {
  return detail::solve_bicgstab(context.problem_, context.workspace_, context.iterate_,
                                context.rhs_, context.controls_, context.normalization_,
                                context.initial_measurement_);
}

template <int Dim>
inline SolveReport detail::GmresKrylovMethodProvider<Dim>::solve(
    PreparedKrylovSolveContext<Dim>& context, const PreparedProviderOptions& options) const {
  const std::int64_t* restart =
      detail::exact_int_option(options, detail::kGmresOptionsSchema, "restart");
  if (restart == nullptr)
    throw std::logic_error("prepared GMRES options were not authenticated");
  return detail::solve_gmres(context.problem_, context.workspace_, context.iterate_, context.rhs_,
                             context.controls_, static_cast<int>(*restart), context.normalization_,
                             context.initial_measurement_);
}

template <int Dim>
inline SolveReport detail::RichardsonKrylovMethodProvider<Dim>::solve(
    PreparedKrylovSolveContext<Dim>& context, const PreparedProviderOptions& options) const {
  const double* relaxation =
      detail::exact_real_option(options, detail::kRichardsonOptionsSchema, "relaxation");
  if (relaxation == nullptr)
    throw std::logic_error("prepared Richardson options were not authenticated");
  return detail::solve_richardson(context.problem_, context.workspace_, context.iterate_,
                                  context.rhs_, context.controls_, static_cast<Real>(*relaxation),
                                  context.normalization_, context.initial_measurement_);
}

template <int Dim>
inline std::shared_ptr<PreparedKrylovMethodRegistry<Dim>>
make_default_krylov_method_provider_registry() {
  auto registry = std::make_shared<PreparedKrylovMethodRegistry<Dim>>();
  registry->add(std::make_shared<detail::CgKrylovMethodProvider<Dim>>());
  registry->add(std::make_shared<detail::BicgstabKrylovMethodProvider<Dim>>());
  registry->add(std::make_shared<detail::GmresKrylovMethodProvider<Dim>>());
  registry->add(std::make_shared<detail::RichardsonKrylovMethodProvider<Dim>>());
  return registry;
}

namespace detail {
template <int Dim>
inline const PreparedKrylovMethodRegistry<Dim>& default_krylov_method_registry() {
  static const std::shared_ptr<PreparedKrylovMethodRegistry<Dim>> registry =
      make_default_krylov_method_provider_registry<Dim>();
  return *registry;
}
}  // namespace detail

template <int Dim>
inline PreparedKrylovMethod<Dim> cg_krylov_method() {
  return detail::default_krylov_method_registry<Dim>().resolve(
      "pops.krylov.cg", PreparedProviderOptions{std::string(detail::kCgOptionsSchema), {}});
}
template <int Dim>
inline PreparedKrylovMethod<Dim> bicgstab_krylov_method() {
  return detail::default_krylov_method_registry<Dim>().resolve(
      "pops.krylov.bicgstab",
      PreparedProviderOptions{std::string(detail::kBicgstabOptionsSchema), {}});
}
template <int Dim>
inline PreparedKrylovMethod<Dim> gmres_krylov_method(int restart) {
  if (restart < 1)
    throw std::invalid_argument("prepared GMRES restart must be positive");
  return detail::default_krylov_method_registry<Dim>().resolve(
      "pops.krylov.gmres",
      PreparedProviderOptions{std::string(detail::kGmresOptionsSchema),
                              {{"restart", static_cast<std::int64_t>(restart)}}});
}
template <int Dim>
inline PreparedKrylovMethod<Dim> richardson_krylov_method(Real relaxation) {
  if (!detail::finite(relaxation) || !(relaxation > Real(0)))
    throw std::invalid_argument("prepared Richardson relaxation must be finite and positive");
  return detail::default_krylov_method_registry<Dim>().resolve(
      "pops.krylov.richardson",
      PreparedProviderOptions{std::string(detail::kRichardsonOptionsSchema),
                              {{"relaxation", static_cast<double>(relaxation)}}});
}

/// Solve one explicitly prepared affine problem with persistent workspace.  There are no legacy raw
/// callback overloads: preparation, property checks, exact snapshot binding, and memory footprint are
/// mandatory parts of the API rather than optional caller conventions.
template <int Dim>
inline SolveReport detail::PreparedKrylovInvocationAccess::execute(
    const PreparedAffineLinearProblem<Dim>& problem, KrylovWorkspace<Dim>& workspace,
    MultiFab<Dim>& iterate, const MultiFab<Dim>& rhs, const KrylovControls<Dim>& controls) {
  MultiFab<Dim>& compatibility_rhs = detail::KrylovWorkspaceAccess::field(workspace, 0);
  const PreparedEquationReference equation =
      detail::PreparedProblemAccess<Dim>::prepare_compatibility_rhs(
          problem, compatibility_rhs, rhs,
          detail::KrylovWorkspaceAccess::metric_reduction_scratch(workspace),
          detail::KrylovWorkspaceAccess::execution_lane(workspace));
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
  const ExecutionLane& lane = detail::KrylovWorkspaceAccess::execution_lane(workspace);

  // Singular compatibility is checked exactly once, collectively and before either the initial
  // gauge or an iterative operator application. The typed status keeps authored outcome/action
  // handling on the SolveReport path instead of leaking a generic exception past it.
  try {
    detail::PreparedProblemAccess<Dim>::require_nullspace_compatible(
        problem, compatibility_rhs,
        detail::KrylovWorkspaceAccess::metric_reduction_scratch(workspace),
        detail::KrylovWorkspaceAccess::execution_lane(workspace));
  } catch (const FieldNullspaceIncompatibleRhs& error) {
    // Compatibility failure leaves the authored iterate untouched.  Its report still carries the
    // exact scientific residual of that iterate, not the generally different ||R(0)|| reference.
    detail::workspace_true_residual(problem, workspace, compatibility_rhs, rhs, iterate);
    const PreparedApplyResult apply_failure =
        detail::KrylovWorkspaceAccess::collective_provider_apply_result(workspace);
    const bool apply_failed = !apply_failure.succeeded();
    Real residual = std::numeric_limits<Real>::quiet_NaN();
    if (!apply_failed) {
      detail::require_exact_scientific_boundary(problem, workspace, compatibility_rhs,
                                                "prepared incompatible-RHS terminal true residual");
      residual = detail::workspace_residual_norm(problem, workspace, compatibility_rhs);
    }
    SolveReport incompatible =
        apply_failed
            ? detail::prepared_apply_failure_report(
                  report_normalization, apply_failure,
                  "prepared operator application failed during incompatible-RHS "
                  "true-residual check")
            : detail::report_physical(report_normalization, residual, 0,
                                      detail::finite(residual) ? SolveStatus::kIncompatibleRhs
                                                               : SolveStatus::kInvalidEvaluation);
    if (!apply_failed)
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
    detail::PreparedProblemAccess<Dim>::apply_nullspace_gauge(
        problem, iterate, detail::KrylovWorkspaceAccess::gauge_coefficients(workspace),
        detail::KrylovWorkspaceAccess::metric_reduction_scratch(workspace),
        detail::KrylovWorkspaceAccess::execution_lane(workspace));

  MultiFab<Dim>& initial_residual = detail::initial_residual_field(workspace, controls);
  detail::workspace_true_residual(problem, workspace, initial_residual, rhs, iterate);
  const PreparedApplyResult initial_apply_failure =
      detail::KrylovWorkspaceAccess::collective_provider_apply_result(workspace);
  if (!initial_apply_failure.succeeded())
    return detail::prepared_apply_failure_report(
        report_normalization, initial_apply_failure,
        "prepared operator application failed before Krylov recurrence");
  // Replica equality is a scientific publication check, not part of the matvec hot path. The
  // initial and final true residuals are the two authoritative boundaries at which a rank-wise
  // isometric permutation must be rejected even though its scalar norm is unchanged.
  detail::require_exact_scientific_boundary(problem, workspace, initial_residual,
                                            "prepared initial true residual");
  const Real initial_physical =
      detail::workspace_residual_norm(problem, workspace, initial_residual);
  if (!detail::finite(initial_physical))
    return detail::report_physical(report_normalization, initial_physical, 0,
                                   SolveStatus::kInvalidEvaluation);
  if (detail::satisfies_stopping_controls(initial_physical, equation.reference_norm, controls))
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

  PreparedKrylovSolveContext<Dim> method_context(problem, workspace, iterate, rhs, controls,
                                                 normalization, initial_measurement);
  std::optional<SolveReport> provider_result;
  long provider_exception_local = 0;
  try {
    // A provider that enters MPI owns one identical, complete collective trace on every rank.  The
    // wrapper can make an exception uniform only after that trace has completed; it deliberately
    // does not pretend to repair a callback that abandons a collective midway.
    provider_result.emplace(controls.method.solve(method_context));
  } catch (...) {
    provider_exception_local = 1;
  }
  // Reduce the two failure channels independently. An infrastructure exception is terminal and
  // must outrank a retryable numerical apply failure reported by another rank.
  const long provider_exception_any = all_reduce_max(provider_exception_local, lane);
  const long provider_apply_failure_any = all_reduce_max(
      detail::KrylovWorkspaceAccess::provider_apply_succeeded(workspace) ? 0L : 1L, lane);
  if (provider_exception_any != 0) {
    SolveReport invalid = detail::report_physical(
        normalization, std::numeric_limits<Real>::quiet_NaN(), 0, SolveStatus::kInvalidEvaluation);
    invalid.reason = "prepared Krylov provider failed after its collective solve trace";
    return invalid;
  }
  if (provider_apply_failure_any != 0) {
    const PreparedApplyResult apply_failure =
        detail::KrylovWorkspaceAccess::collective_provider_apply_result(workspace);
    return detail::prepared_apply_failure_report(
        normalization, apply_failure,
        "prepared operator or preconditioner application failed during Krylov recurrence");
  }
  SolveReport& result = *provider_result;
  const bool malformed = !solve_report_is_publishable(result, controls.max_iterations);
  if (all_reduce_max(malformed ? 1L : 0L, lane) != 0) {
    SolveReport invalid = detail::report_physical(
        normalization, std::numeric_limits<Real>::quiet_NaN(), 0, SolveStatus::kInvalidEvaluation);
    invalid.reason = "prepared Krylov provider published a malformed SolveReport";
    return invalid;
  }
  if (!detail::provider_solve_report_agrees(result, workspace)) {
    SolveReport invalid = detail::report_physical(
        normalization, std::numeric_limits<Real>::quiet_NaN(), 0, SolveStatus::kInvalidEvaluation);
    invalid.reason = "prepared Krylov provider report differs between communicator ranks";
    return invalid;
  }

  // A method provider controls a recurrence, never scientific convergence authority.  Project a
  // singular candidate to its authored representative first, then independently recompute the
  // physical residual for every provider using an existing workspace field.  This rejects a
  // provider that returns a false Solved/NaN report or leaves the iterate unchanged, without a
  // per-solve allocation or a method-name branch in the core.
  if (problem.has_nullspace()) {
    detail::PreparedProblemAccess<Dim>::apply_nullspace_gauge(
        problem, iterate, detail::KrylovWorkspaceAccess::gauge_coefficients(workspace),
        detail::KrylovWorkspaceAccess::metric_reduction_scratch(workspace),
        detail::KrylovWorkspaceAccess::execution_lane(workspace));
  }
  detail::workspace_true_residual(problem, workspace, compatibility_rhs, rhs, iterate);
  const PreparedApplyResult final_apply_failure =
      detail::KrylovWorkspaceAccess::collective_provider_apply_result(workspace);
  const bool final_apply_failed = !final_apply_failure.succeeded();
  Real final_residual = std::numeric_limits<Real>::quiet_NaN();
  if (!final_apply_failed) {
    detail::require_exact_scientific_boundary(problem, workspace, compatibility_rhs,
                                              "prepared final true residual");
    final_residual = detail::workspace_residual_norm(problem, workspace, compatibility_rhs);
  }
  if (final_apply_failed) {
    result.mark_failed(
        SolveStatus::kInvalidEvaluation, detail::solve_action(final_apply_failure),
        detail::prepared_apply_failure_reason(
            final_apply_failure,
            "prepared operator application failed during final true-residual check"));
  } else if (!detail::finite(final_residual)) {
    result.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                       "prepared Krylov provider produced a non-finite true residual");
  } else if (result.iters < 0 || result.iters > controls.max_iterations) {
    result.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                       "prepared Krylov provider returned an invalid iteration count");
  } else if (result.solved() && !detail::satisfies_stopping_controls(
                                    final_residual, equation.reference_norm, controls)) {
    result.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                       problem.has_nullspace()
                           ? "prepared nullspace representative failed the true-residual check"
                           : "prepared Krylov provider claimed an unverified solved value");
  } else if (!result.valid()) {
    result.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                       "prepared Krylov provider returned an invalid status/action pair");
  }
  // When the final evaluation itself fails, retain the last authoritative scientific residual
  // instead of replacing a valid failure report with NaN evidence.
  detail::set_report_physical_residuals(
      result, normalization,
      final_apply_failed || !detail::finite(final_residual) ? initial_physical : final_residual);
  return std::move(*provider_result);
}

/// One collectively materialized solve invocation. Materialization is an ordered control-plane
/// operation on the problem's authenticated preparation lane and authenticates the exact
/// problem/workspace/input contract. Once materialized, distinct invocations may execute
/// concurrently: every numerical collective, provider callback and halo exchange stays on the
/// invocation's workspace-private ExecutionLane.
template <int Dim>
class PreparedKrylovInvocation final {
 public:
  PreparedKrylovInvocation(const PreparedKrylovInvocation&) = delete;
  PreparedKrylovInvocation& operator=(const PreparedKrylovInvocation&) = delete;
  PreparedKrylovInvocation& operator=(PreparedKrylovInvocation&&) = delete;

  PreparedKrylovInvocation(PreparedKrylovInvocation&& other) noexcept
      : problem_(std::exchange(other.problem_, nullptr)),
        workspace_(std::exchange(other.workspace_, nullptr)),
        iterate_(std::exchange(other.iterate_, nullptr)),
        publication_destination_(std::exchange(other.publication_destination_, nullptr)),
        rhs_(std::exchange(other.rhs_, nullptr)),
        controls_(std::move(other.controls_)),
        consumed_(std::exchange(other.consumed_, true)),
        owns_reservation_(std::exchange(other.owns_reservation_, false)) {}

  ~PreparedKrylovInvocation() { release_(); }

  [[nodiscard]] SolveReport execute() {
    if (!owns_reservation_ || problem_ == nullptr || workspace_ == nullptr || iterate_ == nullptr ||
        rhs_ == nullptr)
      throw std::logic_error("prepared Krylov invocation is empty");
    const ExecutionLane& lane = detail::KrylovWorkspaceAccess::execution_lane(*workspace_);
    const long already_consumed = all_reduce_max(consumed_ ? 1L : 0L, lane);
    consumed_ = true;
    if (already_consumed != 0)
      throw std::logic_error("prepared Krylov invocation may execute exactly once");

    // Revalidate on the selected data lane immediately before execution. The earlier control-lane
    // gate made the lane selection itself uniform; this second allocation-free gate catches any
    // input, snapshot or provider mutation between materialization and worker-thread launch.
    detail::collective_solve_preflight(*problem_, *workspace_, *iterate_, *rhs_, controls_, lane);
    detail::KrylovWorkspaceAccess::reset_provider_apply_status(*workspace_);
    std::optional<SolveReport> result;
    long terminal_exception_local = 0;
    try {
      result.emplace(detail::PreparedKrylovInvocationAccess::execute(*problem_, *workspace_,
                                                                     *iterate_, *rhs_, controls_));
    } catch (...) {
      terminal_exception_local = 1;
    }
    // Every allocation and reason publication after a completed scientific/provider trace reaches
    // this final common boundary. A rank-local terminal exception is therefore exposed uniformly
    // instead of letting callers choose different control flow. This cannot and does not claim to
    // repair a provider that abandons its own MPI trace midway.
    if (all_reduce_max(terminal_exception_local, lane) != 0)
      throw std::runtime_error(
          "prepared Krylov solve failed terminally on at least one communicator rank");
    // The entry preflight authenticates the snapshot used by every prepared callback. Recheck it
    // once on the invocation-private lane after the complete provider trace and before publishing
    // the result. This rejects external state mutation without adding a control collective to each
    // matrix-vector product or coupling concurrent invocations through the preparation lane.
    detail::PreparedProblemAccess<Dim>::require_current(*problem_, lane);
    return std::move(*result);
  }

 private:
  struct MaterializedToken {};

  PreparedKrylovInvocation(const PreparedAffineLinearProblem<Dim>& problem,
                           KrylovWorkspace<Dim>& workspace, MultiFab<Dim>& iterate,
                           MultiFab<Dim>* publication_destination, const MultiFab<Dim>& rhs,
                           KrylovControls<Dim> controls, MaterializedToken)
      : problem_(&problem),
        workspace_(&workspace),
        iterate_(&iterate),
        publication_destination_(publication_destination),
        rhs_(&rhs),
        controls_(std::move(controls)),
        owns_reservation_(true) {}

  void transfer_publication_to_outcome_() {
    if (!owns_reservation_ || !consumed_ || problem_ == nullptr || workspace_ == nullptr ||
        publication_destination_ == nullptr)
      throw std::logic_error(
          "prepared Krylov publication requires one completed staged invocation");
    detail::KrylovWorkspaceAccess::arm_publication(*workspace_, *problem_,
                                                   *publication_destination_);
    owns_reservation_ = false;
    publication_destination_ = nullptr;
  }

  void release_() noexcept {
    if (!owns_reservation_)
      return;
    detail::KrylovWorkspaceAccess::release_solve(*workspace_);
    detail::PreparedProblemAccess<Dim>::release_use(*problem_);
    owns_reservation_ = false;
  }

  friend PreparedKrylovInvocation<Dim> detail::prepare_krylov_solve_in_place<Dim>(
      const PreparedAffineLinearProblem<Dim>&, KrylovWorkspace<Dim>&, MultiFab<Dim>&,
      const MultiFab<Dim>&, const KrylovControls<Dim>&, bool);
  template <int OtherDim>
  friend SolveOutcome solve_prepared_affine_outcome(const PreparedAffineLinearProblem<OtherDim>&,
                                                    KrylovWorkspace<OtherDim>&, MultiFab<OtherDim>&,
                                                    const MultiFab<OtherDim>&,
                                                    const KrylovControls<OtherDim>&);

  const PreparedAffineLinearProblem<Dim>* problem_ = nullptr;
  KrylovWorkspace<Dim>* workspace_ = nullptr;
  MultiFab<Dim>* iterate_ = nullptr;
  MultiFab<Dim>* publication_destination_ = nullptr;
  const MultiFab<Dim>* rhs_ = nullptr;
  KrylovControls<Dim> controls_{};
  bool consumed_ = false;
  bool owns_reservation_ = false;
};

/// Materialize invocations collectively in one canonical order before launching worker threads.
/// This is the generic MPI matching boundary: a communicator cannot diagnose ranks selecting
/// different communicators after a collective has already begun, so selection is authenticated on
/// the common control communicator first and numerical execution only then enters private lanes.
template <int Dim>
inline PreparedKrylovInvocation<Dim> detail::prepare_krylov_solve_in_place(
    const PreparedAffineLinearProblem<Dim>& problem, KrylovWorkspace<Dim>& workspace,
    MultiFab<Dim>& iterate, const MultiFab<Dim>& rhs, const KrylovControls<Dim>& controls,
    bool staged_publication) {
  const bool workspace_reserved = detail::KrylovWorkspaceAccess::try_reserve_solve(workspace);
  const bool problem_reserved = detail::PreparedProblemAccess<Dim>::try_reserve_use(problem);
  detail::PendingPreparedKrylovReservations<Dim> pending_reservations(
      problem, workspace, problem_reserved, workspace_reserved);
  const ExecutionLane& control_lane = detail::PreparedProblemAccess<Dim>::preparation_lane(problem);
  const detail::PreparedProblemControlConsensus reservation_consensus =
      detail::coordinate_prepared_problem_control(
          detail::PreparedProblemControlOperation::MaterializeSolve, workspace_reserved,
          problem_reserved, control_lane);
  if (!reservation_consensus.operation_agrees ||
      reservation_consensus.workspace_reservation_failed != 0 ||
      reservation_consensus.problem_reservation_failed != 0) {
    if (!reservation_consensus.operation_agrees)
      throw std::logic_error(
          "prepared Krylov control operations differ across communicator ranks; prepare, bind, "
          "and solve materialization must use one canonical collective order");
    if (reservation_consensus.workspace_reservation_failed != 0)
      throw std::logic_error(
          "KrylovWorkspace is already reserved by another prepared bind or solve invocation");
    throw std::logic_error(
        "prepared affine problem is being mutated or its operator requires exclusive access to "
        "its external execution context");
  }

  MultiFab<Dim>* solve_iterate = &iterate;
  if (staged_publication) {
    long staging_failure_local = 0;
    try {
      MultiFab<Dim>& candidate = detail::KrylovWorkspaceAccess::publication_candidate(workspace);
      detail::PreparedFieldAlgebra::copy(candidate, iterate);
      solve_iterate = &candidate;
    } catch (...) {
      staging_failure_local = 1;
    }
    if (all_reduce_max(staging_failure_local, control_lane) != 0)
      throw std::runtime_error(
          "prepared Krylov publication staging failed on at least one communicator rank");
  }

  detail::collective_solve_preflight(problem, workspace, *solve_iterate, rhs, controls,
                                     control_lane);
  PreparedKrylovInvocation<Dim> invocation(
      problem, workspace, *solve_iterate, staged_publication ? &iterate : nullptr, rhs, controls,
      typename PreparedKrylovInvocation<Dim>::MaterializedToken{});
  pending_reservations.transfer_to_invocation();
  return invocation;
}

/// Internal in-place numerical primitive. Public/runtime callers use
/// solve_prepared_affine_outcome(); only prepared solver implementations and contract tests name
/// this detail route explicitly.
template <int Dim>
inline SolveReport detail::solve_prepared_affine_in_place(
    const PreparedAffineLinearProblem<Dim>& problem, KrylovWorkspace<Dim>& workspace,
    MultiFab<Dim>& iterate, const MultiFab<Dim>& rhs, const KrylovControls<Dim>& controls) {
  PreparedKrylovInvocation<Dim> invocation =
      prepare_krylov_solve_in_place(problem, workspace, iterate, rhs, controls);
  return invocation.execute();
}

/// Generated/runtime publication boundary for a prepared global solve. The direct numerical entry
/// point above remains the provider-level in-place report API. This route solves into a persistent
/// workspace-private candidate, keeps the workspace/problem reserved, and copies the candidate to
/// @p iterate only when the outcome is accepted on the workspace's exact execution lane.
template <int Dim>
inline SolveOutcome solve_prepared_affine_outcome(const PreparedAffineLinearProblem<Dim>& problem,
                                                  KrylovWorkspace<Dim>& workspace,
                                                  MultiFab<Dim>& iterate, const MultiFab<Dim>& rhs,
                                                  const KrylovControls<Dim>& controls) {
  PreparedKrylovInvocation<Dim> invocation =
      detail::prepare_krylov_solve_in_place(problem, workspace, iterate, rhs, controls,
                                            /*staged_publication=*/true);
  SolveReport report = invocation.execute();
  invocation.transfer_publication_to_outcome_();
  return SolveOutcome::collective_lane(
      std::move(report), detail::KrylovWorkspaceAccess::execution_lane(workspace),
      SolveOutcome::PublicationHooks{&workspace,
                                     [](void* context) noexcept {
                                       detail::KrylovWorkspaceAccess::publish_candidate(
                                           *static_cast<KrylovWorkspace<Dim>*>(context));
                                     },
                                     nullptr,
                                     [](void* context) noexcept {
                                       detail::KrylovWorkspaceAccess::release_publication(
                                           *static_cast<KrylovWorkspace<Dim>*>(context));
                                     },
                                     {},
                                     [](void* context) {
                                       detail::KrylovWorkspaceAccess::validate_publication(
                                           *static_cast<KrylovWorkspace<Dim>*>(context));
                                     }});
}

}  // namespace pops
