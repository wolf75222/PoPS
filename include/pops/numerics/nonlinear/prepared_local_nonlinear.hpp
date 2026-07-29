#pragma once

/// @file
/// @brief One prepared, device-callable provider for every cell-local nonlinear solve.
///
/// The residual, Jacobian and admissible-domain providers are concrete template parameters.  A
/// prepared problem is therefore a small immutable value: no std::function, heap allocation, Python
/// callback or runtime algorithm dispatch can enter the per-cell loop.  The solver keeps every
/// iterate and factorization on the stack and returns a candidate plus an explicit status; callers
/// may publish the candidate only when `solved()` is true.

#include <pops/core/foundation/types.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace pops {

enum class LocalNonlinearStatus : int {
  kConverged = 0,
  kIterationLimit = 1,
  kSingularJacobian = 2,
  kInadmissibleCandidate = 3,
  kSafeguardFailure = 4,
  kInvalidEvaluation = 5,
  kUnsupportedCapability = 6,
  kEvaluationRetry = 7,
  kEvaluationReject = 8,
  kEvaluationFailed = 9,
};

/// Collective precedence is intentionally independent of the public status code. A fatal local
/// failure must dominate a recoverable rejection when different cells or MPI ranks fail
/// differently.
POPS_HD inline int local_nonlinear_status_priority(LocalNonlinearStatus status) {
  switch (status) {
    case LocalNonlinearStatus::kConverged:
      return 0;
    case LocalNonlinearStatus::kIterationLimit:
      return 1;
    case LocalNonlinearStatus::kInadmissibleCandidate:
      return 2;
    case LocalNonlinearStatus::kSafeguardFailure:
      return 3;
    case LocalNonlinearStatus::kEvaluationRetry:
      return 4;
    case LocalNonlinearStatus::kEvaluationReject:
      return 5;
    case LocalNonlinearStatus::kSingularJacobian:
      return 6;
    case LocalNonlinearStatus::kInvalidEvaluation:
      return 7;
    case LocalNonlinearStatus::kUnsupportedCapability:
      return 8;
    case LocalNonlinearStatus::kEvaluationFailed:
      return 9;
  }
  return 9;
}

inline LocalNonlinearStatus local_nonlinear_status_from_priority(int priority) {
  switch (priority) {
    case 0:
      return LocalNonlinearStatus::kConverged;
    case 1:
      return LocalNonlinearStatus::kIterationLimit;
    case 2:
      return LocalNonlinearStatus::kInadmissibleCandidate;
    case 3:
      return LocalNonlinearStatus::kSafeguardFailure;
    case 4:
      return LocalNonlinearStatus::kEvaluationRetry;
    case 5:
      return LocalNonlinearStatus::kEvaluationReject;
    case 6:
      return LocalNonlinearStatus::kSingularJacobian;
    case 7:
      return LocalNonlinearStatus::kInvalidEvaluation;
    case 8:
      return LocalNonlinearStatus::kUnsupportedCapability;
    case 9:
      return LocalNonlinearStatus::kEvaluationFailed;
    default:
      throw std::runtime_error("unknown local nonlinear collective priority");
  }
}

enum class LocalNonlinearEvaluationStatus : int {
  kOk = 0,
  kRetry = 1,
  kReject = 2,
  kFailed = 3,
  kInvalid = 4,
};

struct LocalNonlinearEvaluationResult {
  LocalNonlinearEvaluationStatus status = LocalNonlinearEvaluationStatus::kInvalid;
  std::uint32_t reason_code = 0;

  POPS_HD static constexpr LocalNonlinearEvaluationResult ok() {
    return {LocalNonlinearEvaluationStatus::kOk, 0};
  }
  POPS_HD static constexpr LocalNonlinearEvaluationResult retry(std::uint32_t reason) {
    return {LocalNonlinearEvaluationStatus::kRetry, reason};
  }
  POPS_HD static constexpr LocalNonlinearEvaluationResult reject(std::uint32_t reason) {
    return {LocalNonlinearEvaluationStatus::kReject, reason};
  }
  POPS_HD static constexpr LocalNonlinearEvaluationResult failed(std::uint32_t reason) {
    return {LocalNonlinearEvaluationStatus::kFailed, reason};
  }
  POPS_HD static constexpr LocalNonlinearEvaluationResult invalid(std::uint32_t reason) {
    return {LocalNonlinearEvaluationStatus::kInvalid, reason};
  }
  POPS_HD constexpr bool succeeded() const { return status == LocalNonlinearEvaluationStatus::kOk; }
};

enum class LocalJacobianKind : int {
  kFiniteDifference,
  kAnalytic,
  kAutomaticDifferentiation,
  kUnsupported,
};

enum class LocalSafeguardKind : int {
  kExactNewton,
  kFixedDamping,
  kBacktrackingLineSearch,
};

enum class LocalInitialGuessPolicy : int {
  kProvided,
};

struct PreparedLocalNonlinearControls {
  int max_iterations = 20;
  int max_evaluations = 0;  ///< zero means derive the exact finite-difference budget from N
  int max_backtracks = 12;
  Real absolute_tolerance = Real(1e-12);
  Real relative_tolerance = Real(0);
  Real step_tolerance = Real(0);
  Real finite_difference_step = Real(1e-7);
  Real pivot_tolerance = Real(64) * std::numeric_limits<Real>::epsilon();
  Real initial_step = Real(1);
  Real minimum_step = Real(1) / Real(4096);
  Real armijo = Real(1e-4);
  LocalSafeguardKind safeguard = LocalSafeguardKind::kExactNewton;
  LocalInitialGuessPolicy initial_guess = LocalInitialGuessPolicy::kProvided;
};

template <int N>
struct LocalNonlinearCellResult {
  Real value[N] = {};
  LocalNonlinearStatus status = LocalNonlinearStatus::kIterationLimit;
  int iterations = 0;
  int evaluations = 0;
  int safeguard_steps = 0;
  Real reference_residual_norm = Real(0);
  Real residual_norm = Real(0);
  Real step_norm = Real(0);
  Real condition_evidence = Real(0);
  int failing_component = -1;
  std::uint32_t reason_code = 0;

  POPS_HD bool solved() const { return status == LocalNonlinearStatus::kConverged; }
};

template <int N>
struct AcceptAllLocalCandidates {
  POPS_HD bool operator()(const Real (&)[N], int* component = nullptr) const {
    if (component != nullptr)
      *component = -1;
    return true;
  }
};

template <int N>
struct FiniteDifferenceLocalJacobian {
  static constexpr LocalJacobianKind kind = LocalJacobianKind::kFiniteDifference;
};

template <int N, class Functor>
struct AnalyticLocalJacobian {
  static constexpr LocalJacobianKind kind = LocalJacobianKind::kAnalytic;
  Functor functor;

  POPS_HD decltype(auto) operator()(const Real (&x)[N], Real (&jacobian)[N][N]) const {
    return functor(x, jacobian);
  }
};

template <int N, class Functor>
struct AutomaticDifferentiationLocalJacobian {
  static constexpr LocalJacobianKind kind = LocalJacobianKind::kAutomaticDifferentiation;
  Functor functor;

  POPS_HD decltype(auto) operator()(const Real (&x)[N], Real (&jacobian)[N][N]) const {
    return functor(x, jacobian);
  }
};

template <int N>
struct UnsupportedLocalJacobian {
  static constexpr LocalJacobianKind kind = LocalJacobianKind::kUnsupported;
};

template <int N, class Residual, class Jacobian = FiniteDifferenceLocalJacobian<N>,
          class Admissible = AcceptAllLocalCandidates<N>>
struct PreparedLocalNonlinearProblem {
  static_assert(N > 0, "a local nonlinear problem needs at least one unknown");

  Residual residual;
  Jacobian jacobian;
  Admissible admissible;
  PreparedLocalNonlinearControls controls;
  Real variable_scale[N] = {};
  Real residual_scale[N] = {};
};

namespace detail {

inline constexpr long long kLocalNonlinearFailureComponentBase = 1024;
inline constexpr long long kLocalNonlinearFailureCellStride = 1048576;
inline constexpr long long kLocalNonlinearFailureEncodingCeiling = 4503599627370496LL;

/// Reverse-pack one failing cell and component into an exactly representable binary64 value. A max
/// reduction then selects the lexicographically first global cell without atomics, and keeps its
/// component attached to that exact cell.
POPS_HD inline Real encode_local_nonlinear_failure(int i, int j, int component) {
  const Real cell = Real(j) * Real(kLocalNonlinearFailureCellStride) + Real(i);
  return Real(kLocalNonlinearFailureEncodingCeiling) -
         (cell * Real(kLocalNonlinearFailureComponentBase) + Real(component + 1) + Real(1));
}

POPS_HD inline void decode_local_nonlinear_failure(Real encoded, int& i, int& j, int& component) {
  const long long packed =
      kLocalNonlinearFailureEncodingCeiling - static_cast<long long>(encoded) - 1;
  component = static_cast<int>(packed % kLocalNonlinearFailureComponentBase) - 1;
  const long long cell = packed / kLocalNonlinearFailureComponentBase;
  i = static_cast<int>(cell % kLocalNonlinearFailureCellStride);
  j = static_cast<int>(cell / kLocalNonlinearFailureCellStride);
}

/// Pack collective failure precedence together with the exact first cell/component.  Generated
/// Program kernels reduce a single statistics field, so independently reducing precedence and
/// location would be able to pair a fatal status with the location of an unrelated recoverable
/// failure.  Precedence selects a disjoint power-of-two bin while the binary64 significand retains
/// the complete 52-bit location payload, so this adds no model-size restriction.
POPS_HD inline Real encode_ranked_local_nonlinear_failure(int priority, int i, int j,
                                                          int component) {
  const Real cell = Real(j) * Real(kLocalNonlinearFailureCellStride) + Real(i);
  const Real packed = cell * Real(kLocalNonlinearFailureComponentBase) + Real(component + 1);
  const long long location_rank =
      kLocalNonlinearFailureEncodingCeiling - static_cast<long long>(packed) - 1;
  Real priority_scale = Real(1);
  for (int bit = 0; bit < priority; ++bit)
    priority_scale *= Real(2);
  return priority_scale *
         (Real(1) + Real(location_rank) / Real(kLocalNonlinearFailureEncodingCeiling));
}

POPS_HD inline void decode_ranked_local_nonlinear_failure(Real encoded, int& priority, int& i,
                                                          int& j, int& component) {
  priority = 0;
  Real normalized = encoded;
  while (normalized >= Real(2)) {
    normalized *= Real(0.5);
    ++priority;
  }
  const long long location_rank =
      static_cast<long long>((normalized - Real(1)) * Real(kLocalNonlinearFailureEncodingCeiling));
  const long long packed = kLocalNonlinearFailureEncodingCeiling - location_rank - 1;
  component = static_cast<int>(packed % kLocalNonlinearFailureComponentBase) - 1;
  const long long cell = packed / kLocalNonlinearFailureComponentBase;
  i = static_cast<int>(cell % kLocalNonlinearFailureCellStride);
  j = static_cast<int>(cell / kLocalNonlinearFailureCellStride);
}

POPS_HD inline Real local_abs(Real value) {
  return value < Real(0) ? -value : value;
}

POPS_HD inline Real local_max(Real left, Real right) {
  return left > right ? left : right;
}

POPS_HD inline Real local_min(Real left, Real right) {
  return left < right ? left : right;
}

template <class T>
POPS_HD inline void local_swap(T& left, T& right) {
  const T temporary = left;
  left = right;
  right = temporary;
}

POPS_HD inline bool local_finite(Real value) {
  return value == value && value <= std::numeric_limits<Real>::max() &&
         value >= -std::numeric_limits<Real>::max();
}

POPS_HD inline LocalNonlinearStatus local_evaluation_failure(
    LocalNonlinearEvaluationStatus status) {
  switch (status) {
    case LocalNonlinearEvaluationStatus::kOk:
      return LocalNonlinearStatus::kConverged;
    case LocalNonlinearEvaluationStatus::kRetry:
      return LocalNonlinearStatus::kEvaluationRetry;
    case LocalNonlinearEvaluationStatus::kReject:
      return LocalNonlinearStatus::kEvaluationReject;
    case LocalNonlinearEvaluationStatus::kFailed:
      return LocalNonlinearStatus::kEvaluationFailed;
    case LocalNonlinearEvaluationStatus::kInvalid:
      return LocalNonlinearStatus::kInvalidEvaluation;
  }
  return LocalNonlinearStatus::kInvalidEvaluation;
}

template <class Evaluation>
POPS_HD inline LocalNonlinearEvaluationResult normalize_local_evaluation(Evaluation&& evaluation) {
  using Result = std::remove_cvref_t<Evaluation>;
  if constexpr (std::is_same_v<Result, LocalNonlinearEvaluationResult>) {
    switch (evaluation.status) {
      case LocalNonlinearEvaluationStatus::kOk:
      case LocalNonlinearEvaluationStatus::kRetry:
      case LocalNonlinearEvaluationStatus::kReject:
      case LocalNonlinearEvaluationStatus::kFailed:
      case LocalNonlinearEvaluationStatus::kInvalid:
        return evaluation;
    }
    return LocalNonlinearEvaluationResult::invalid(evaluation.reason_code);
  } else if constexpr (std::is_same_v<Result, bool>) {
    return evaluation ? LocalNonlinearEvaluationResult::ok()
                      : LocalNonlinearEvaluationResult::invalid(0);
  } else {
    static_assert(
        std::is_same_v<Result, LocalNonlinearEvaluationResult> || std::is_same_v<Result, bool>,
        "a fallible local evaluation must return bool or "
        "LocalNonlinearEvaluationResult");
  }
}

template <int N, class Residual>
POPS_HD inline LocalNonlinearEvaluationResult invoke_local_residual(
    const Residual& residual_provider, const Real (&x)[N], Real (&residual)[N]) {
  using Result = decltype(residual_provider(x, residual));
  if constexpr (std::is_void_v<Result>) {
    residual_provider(x, residual);
    return LocalNonlinearEvaluationResult::ok();
  } else {
    return normalize_local_evaluation(residual_provider(x, residual));
  }
}

template <int N, class Jacobian>
POPS_HD inline LocalNonlinearEvaluationResult invoke_local_jacobian(
    const Jacobian& jacobian_provider, const Real (&x)[N], Real (&jacobian)[N][N]) {
  using Result = decltype(jacobian_provider(x, jacobian));
  if constexpr (std::is_void_v<Result>) {
    jacobian_provider(x, jacobian);
    return LocalNonlinearEvaluationResult::ok();
  } else {
    return normalize_local_evaluation(jacobian_provider(x, jacobian));
  }
}

template <int N>
POPS_HD inline void copy_local_vector(const Real (&source)[N], Real (&destination)[N]) {
  for (int i = 0; i < N; ++i)
    destination[i] = source[i];
}

template <int N>
POPS_HD inline Real scaled_inf_norm(const Real (&value)[N], const Real (&scale)[N],
                                    int* component = nullptr) {
  Real norm = Real(0);
  int worst = -1;
  for (int i = 0; i < N; ++i) {
    const Real magnitude = local_abs(value[i]) / scale[i];
    if (magnitude > norm) {
      norm = magnitude;
      worst = i;
    }
  }
  if (component != nullptr)
    *component = worst;
  return norm;
}

template <int N, class Problem>
POPS_HD inline LocalNonlinearStatus evaluate_local_residual(const Problem& problem,
                                                            const Real (&x)[N], Real (&residual)[N],
                                                            int& evaluations, int evaluation_budget,
                                                            Real& norm, int& failing_component,
                                                            std::uint32_t& reason_code) {
  if (evaluations >= evaluation_budget)
    return LocalNonlinearStatus::kIterationLimit;
  const LocalNonlinearEvaluationResult evaluation =
      invoke_local_residual<N>(problem.residual, x, residual);
  ++evaluations;
  if (!evaluation.succeeded()) {
    reason_code = evaluation.reason_code;
    return local_evaluation_failure(evaluation.status);
  }
  for (int i = 0; i < N; ++i) {
    if (!local_finite(residual[i])) {
      failing_component = i;
      norm = std::numeric_limits<Real>::max();
      return LocalNonlinearStatus::kInvalidEvaluation;
    }
  }
  norm = scaled_inf_norm<N>(residual, problem.residual_scale, &failing_component);
  if (!local_finite(norm))
    return LocalNonlinearStatus::kInvalidEvaluation;
  return LocalNonlinearStatus::kConverged;
}

template <int N>
struct PivotedSolveEvidence {
  bool solved = false;
  bool invalid_evaluation = false;
  Real condition_evidence = Real(0);
  int failing_pivot = -1;
};

/// Partial-pivot Gaussian factorization and triangular solve.  The matrix and right-hand side are
/// destroyed.  No explicit inverse is ever formed.
template <int N>
POPS_HD inline PivotedSolveEvidence<N> pivoted_dense_solve(Real (&matrix)[N][N], Real (&rhs)[N],
                                                           Real (&solution)[N],
                                                           Real relative_pivot_tolerance) {
  PivotedSolveEvidence<N> evidence;
  Real matrix_scale = Real(0);
  for (int row = 0; row < N; ++row)
    for (int col = 0; col < N; ++col) {
      if (!local_finite(matrix[row][col])) {
        evidence.invalid_evaluation = true;
        evidence.failing_pivot = col;
        return evidence;
      }
      matrix_scale = local_max(matrix_scale, local_abs(matrix[row][col]));
    }
  const Real pivot_floor = relative_pivot_tolerance * matrix_scale;
  Real smallest_pivot = std::numeric_limits<Real>::max();
  Real largest_pivot = Real(0);

  for (int pivot = 0; pivot < N; ++pivot) {
    int selected = pivot;
    Real selected_magnitude = local_abs(matrix[pivot][pivot]);
    for (int row = pivot + 1; row < N; ++row) {
      const Real magnitude = local_abs(matrix[row][pivot]);
      if (magnitude > selected_magnitude) {
        selected = row;
        selected_magnitude = magnitude;
      }
    }
    if (!local_finite(selected_magnitude)) {
      evidence.invalid_evaluation = true;
      evidence.failing_pivot = pivot;
      return evidence;
    }
    if (selected_magnitude <= pivot_floor) {
      evidence.failing_pivot = pivot;
      return evidence;
    }
    if (selected != pivot) {
      for (int col = pivot; col < N; ++col)
        local_swap(matrix[pivot][col], matrix[selected][col]);
      local_swap(rhs[pivot], rhs[selected]);
    }
    const Real diagonal = matrix[pivot][pivot];
    const Real diagonal_magnitude = local_abs(diagonal);
    smallest_pivot = local_min(smallest_pivot, diagonal_magnitude);
    largest_pivot = local_max(largest_pivot, diagonal_magnitude);
    for (int row = pivot + 1; row < N; ++row) {
      const Real factor = matrix[row][pivot] / diagonal;
      if (!local_finite(factor)) {
        evidence.invalid_evaluation = true;
        evidence.failing_pivot = pivot;
        return evidence;
      }
      matrix[row][pivot] = factor;
      for (int col = pivot + 1; col < N; ++col)
        matrix[row][col] -= factor * matrix[pivot][col];
      rhs[row] -= factor * rhs[pivot];
    }
  }
  for (int row = N - 1; row >= 0; --row) {
    Real value = rhs[row];
    for (int col = row + 1; col < N; ++col)
      value -= matrix[row][col] * solution[col];
    const Real diagonal = matrix[row][row];
    if (!local_finite(value) || !local_finite(diagonal) || local_abs(diagonal) <= pivot_floor) {
      evidence.invalid_evaluation = !local_finite(value) || !local_finite(diagonal);
      evidence.failing_pivot = row;
      return evidence;
    }
    solution[row] = value / diagonal;
    if (!local_finite(solution[row])) {
      evidence.invalid_evaluation = true;
      evidence.failing_pivot = row;
      return evidence;
    }
  }
  evidence.solved = true;
  const Real largest_finite = std::numeric_limits<Real>::max();
  evidence.condition_evidence =
      smallest_pivot > Real(0) && smallest_pivot >= largest_pivot / largest_finite
          ? largest_pivot / smallest_pivot
          : largest_finite;
  return evidence;
}

template <int N, class Problem>
POPS_HD inline LocalNonlinearStatus build_local_jacobian(
    const Problem& problem, const Real (&x)[N], const Real (&residual)[N], Real (&jacobian)[N][N],
    int& evaluations, int evaluation_budget, int& failing_component, std::uint32_t& reason_code) {
  using Jacobian = std::remove_cvref_t<decltype(problem.jacobian)>;
  if constexpr (Jacobian::kind == LocalJacobianKind::kUnsupported) {
    return LocalNonlinearStatus::kUnsupportedCapability;
  } else if constexpr (Jacobian::kind == LocalJacobianKind::kAnalytic ||
                       Jacobian::kind == LocalJacobianKind::kAutomaticDifferentiation) {
    if (evaluations >= evaluation_budget)
      return LocalNonlinearStatus::kIterationLimit;
    const LocalNonlinearEvaluationResult evaluation =
        invoke_local_jacobian<N>(problem.jacobian, x, jacobian);
    ++evaluations;
    if (!evaluation.succeeded()) {
      reason_code = evaluation.reason_code;
      return local_evaluation_failure(evaluation.status);
    }
  } else {
    Real perturbed[N];
    Real perturbed_residual[N];
    for (int col = 0; col < N; ++col) {
      copy_local_vector<N>(x, perturbed);
      const Real step = problem.controls.finite_difference_step *
                        local_max(local_abs(x[col]), problem.variable_scale[col]);
      if (!local_finite(step) || step <= Real(0)) {
        failing_component = col;
        return LocalNonlinearStatus::kInvalidEvaluation;
      }
      perturbed[col] += step;
      Real ignored_norm = Real(0);
      int invalid_component = -1;
      const LocalNonlinearStatus evaluated = evaluate_local_residual<N>(
          problem, perturbed, perturbed_residual, evaluations, evaluation_budget, ignored_norm,
          invalid_component, reason_code);
      if (evaluated != LocalNonlinearStatus::kConverged) {
        failing_component = invalid_component >= 0 ? invalid_component : col;
        return evaluated;
      }
      for (int row = 0; row < N; ++row)
        jacobian[row][col] = (perturbed_residual[row] - residual[row]) / step;
    }
  }
  for (int row = 0; row < N; ++row)
    for (int col = 0; col < N; ++col)
      if (!local_finite(jacobian[row][col])) {
        failing_component = col;
        return LocalNonlinearStatus::kInvalidEvaluation;
      }
  return LocalNonlinearStatus::kConverged;
}

template <int N, class Problem>
POPS_HD inline bool valid_prepared_problem(const Problem& problem) {
  const PreparedLocalNonlinearControls& controls = problem.controls;
  const bool known_safeguard = controls.safeguard == LocalSafeguardKind::kExactNewton ||
                               controls.safeguard == LocalSafeguardKind::kFixedDamping ||
                               controls.safeguard == LocalSafeguardKind::kBacktrackingLineSearch;
  if (controls.max_iterations <= 0 || controls.max_evaluations < 0 || controls.max_backtracks < 0 ||
      !known_safeguard || !local_finite(controls.absolute_tolerance) ||
      !local_finite(controls.relative_tolerance) || !local_finite(controls.step_tolerance) ||
      controls.absolute_tolerance < Real(0) || controls.relative_tolerance < Real(0) ||
      controls.step_tolerance < Real(0) ||
      (controls.absolute_tolerance == Real(0) && controls.relative_tolerance == Real(0) &&
       controls.step_tolerance == Real(0)) ||
      !local_finite(controls.finite_difference_step) ||
      controls.finite_difference_step <= Real(0) || !local_finite(controls.pivot_tolerance) ||
      controls.pivot_tolerance <= Real(0) || !local_finite(controls.initial_step) ||
      controls.initial_step <= Real(0) || controls.initial_step > Real(1) ||
      (controls.safeguard == LocalSafeguardKind::kExactNewton &&
       controls.initial_step != Real(1)) ||
      !local_finite(controls.minimum_step) || controls.minimum_step <= Real(0) ||
      controls.minimum_step > controls.initial_step || !local_finite(controls.armijo) ||
      controls.armijo <= Real(0) || controls.armijo >= Real(1) ||
      controls.initial_guess != LocalInitialGuessPolicy::kProvided)
    return false;
  for (int i = 0; i < N; ++i)
    if (!local_finite(problem.variable_scale[i]) || problem.variable_scale[i] <= Real(0) ||
        !local_finite(problem.residual_scale[i]) || problem.residual_scale[i] <= Real(0))
      return false;
  return true;
}

}  // namespace detail

template <int N, class Residual, class Jacobian = FiniteDifferenceLocalJacobian<N>,
          class Admissible = AcceptAllLocalCandidates<N>>
POPS_HD inline PreparedLocalNonlinearProblem<N, Residual, Jacobian, Admissible>
prepare_local_nonlinear_problem(Residual residual, Jacobian jacobian, Admissible admissible,
                                const PreparedLocalNonlinearControls& controls,
                                Real variable_scale = Real(1), Real residual_scale = Real(1)) {
  PreparedLocalNonlinearProblem<N, Residual, Jacobian, Admissible> prepared{
      residual, jacobian, admissible, controls, {}, {}};
  for (int i = 0; i < N; ++i) {
    prepared.variable_scale[i] = variable_scale;
    prepared.residual_scale[i] = residual_scale;
  }
  return prepared;
}

template <int N, class Residual, class Jacobian = FiniteDifferenceLocalJacobian<N>,
          class Admissible = AcceptAllLocalCandidates<N>>
POPS_HD inline PreparedLocalNonlinearProblem<N, Residual, Jacobian, Admissible>
prepare_local_nonlinear_problem(Residual residual, Jacobian jacobian, Admissible admissible,
                                const PreparedLocalNonlinearControls& controls,
                                const Real (&variable_scale)[N], const Real (&residual_scale)[N]) {
  PreparedLocalNonlinearProblem<N, Residual, Jacobian, Admissible> prepared{
      residual, jacobian, admissible, controls, {}, {}};
  for (int i = 0; i < N; ++i) {
    prepared.variable_scale[i] = variable_scale[i];
    prepared.residual_scale[i] = residual_scale[i];
  }
  return prepared;
}

template <int N, class Residual, class Jacobian, class Admissible>
POPS_HD inline LocalNonlinearCellResult<N> solve_prepared_local_nonlinear(
    const PreparedLocalNonlinearProblem<N, Residual, Jacobian, Admissible>& problem,
    const Real (&initial)[N]) {
  LocalNonlinearCellResult<N> result;
  detail::copy_local_vector<N>(initial, result.value);
  if (!detail::valid_prepared_problem<N>(problem)) {
    result.status = LocalNonlinearStatus::kUnsupportedCapability;
    return result;
  }
  int inadmissible_component = -1;
  if (!problem.admissible(result.value, &inadmissible_component)) {
    result.status = LocalNonlinearStatus::kInadmissibleCandidate;
    result.failing_component = inadmissible_component;
    return result;
  }

  const long long derived_budget_wide =
      2LL +
      static_cast<long long>(problem.controls.max_iterations) *
          static_cast<long long>(
              N + 2 + (problem.controls.max_backtracks > 0 ? problem.controls.max_backtracks : 0));
  const int derived_budget =
      derived_budget_wide > static_cast<long long>(std::numeric_limits<int>::max())
          ? std::numeric_limits<int>::max()
          : static_cast<int>(derived_budget_wide);
  const int evaluation_budget =
      problem.controls.max_evaluations > 0 ? problem.controls.max_evaluations : derived_budget;
  Real residual[N];
  int failing_component = -1;
  LocalNonlinearStatus evaluated = detail::evaluate_local_residual<N>(
      problem, result.value, residual, result.evaluations, evaluation_budget, result.residual_norm,
      failing_component, result.reason_code);
  if (evaluated != LocalNonlinearStatus::kConverged) {
    result.status = evaluated;
    result.failing_component = failing_component;
    return result;
  }
  result.reference_residual_norm = result.residual_norm;
  const Real convergence_threshold =
      problem.controls.absolute_tolerance +
      problem.controls.relative_tolerance * result.reference_residual_norm;
  if (result.residual_norm <= convergence_threshold) {
    result.status = LocalNonlinearStatus::kConverged;
    return result;
  }

  for (int iteration = 0; iteration < problem.controls.max_iterations; ++iteration) {
    Real jacobian[N][N];
    evaluated = detail::build_local_jacobian<N>(problem, result.value, residual, jacobian,
                                                result.evaluations, evaluation_budget,
                                                failing_component, result.reason_code);
    if (evaluated != LocalNonlinearStatus::kConverged) {
      result.status = evaluated;
      result.failing_component = failing_component;
      return result;
    }

    Real scaled_matrix[N][N];
    Real scaled_rhs[N];
    Real scaled_step[N] = {};
    for (int row = 0; row < N; ++row) {
      scaled_rhs[row] = residual[row] / problem.residual_scale[row];
      for (int col = 0; col < N; ++col)
        scaled_matrix[row][col] =
            jacobian[row][col] * problem.variable_scale[col] / problem.residual_scale[row];
    }
    const detail::PivotedSolveEvidence<N> factorization = detail::pivoted_dense_solve<N>(
        scaled_matrix, scaled_rhs, scaled_step, problem.controls.pivot_tolerance);
    result.condition_evidence =
        detail::local_max(result.condition_evidence, factorization.condition_evidence);
    if (!factorization.solved) {
      result.status = factorization.invalid_evaluation ? LocalNonlinearStatus::kInvalidEvaluation
                                                       : LocalNonlinearStatus::kSingularJacobian;
      result.failing_component = factorization.failing_pivot;
      return result;
    }

    Real step[N];
    for (int i = 0; i < N; ++i)
      step[i] = problem.variable_scale[i] * scaled_step[i];
    const Real full_step_norm = detail::scaled_inf_norm<N>(step, problem.variable_scale);
    Real alpha = problem.controls.safeguard == LocalSafeguardKind::kExactNewton
                     ? Real(1)
                     : problem.controls.initial_step;
    const int attempts = problem.controls.safeguard == LocalSafeguardKind::kBacktrackingLineSearch
                             ? problem.controls.max_backtracks + 1
                             : 1;
    bool accepted = false;
    bool saw_admissible = false;
    Real trial[N];
    Real trial_residual[N];
    Real trial_norm = std::numeric_limits<Real>::max();
    for (int attempt = 0; attempt < attempts; ++attempt) {
      for (int i = 0; i < N; ++i)
        trial[i] = result.value[i] - alpha * step[i];
      int candidate_component = -1;
      if (problem.admissible(trial, &candidate_component)) {
        saw_admissible = true;
        evaluated = detail::evaluate_local_residual<N>(
            problem, trial, trial_residual, result.evaluations, evaluation_budget, trial_norm,
            failing_component, result.reason_code);
        if (evaluated != LocalNonlinearStatus::kConverged) {
          result.status = evaluated;
          result.failing_component = failing_component;
          return result;
        }
        const bool sufficient =
            problem.controls.safeguard != LocalSafeguardKind::kBacktrackingLineSearch ||
            trial_norm <= (Real(1) - problem.controls.armijo * alpha) * result.residual_norm;
        if (sufficient) {
          accepted = true;
          break;
        }
      } else {
        result.failing_component = candidate_component;
      }
      if (problem.controls.safeguard != LocalSafeguardKind::kBacktrackingLineSearch)
        break;
      ++result.safeguard_steps;
      alpha *= Real(0.5);
      if (alpha < problem.controls.minimum_step)
        break;
    }
    if (!accepted) {
      result.status = saw_admissible ? LocalNonlinearStatus::kSafeguardFailure
                                     : LocalNonlinearStatus::kInadmissibleCandidate;
      return result;
    }
    detail::copy_local_vector<N>(trial, result.value);
    detail::copy_local_vector<N>(trial_residual, residual);
    result.residual_norm = trial_norm;
    result.step_norm = alpha * full_step_norm;
    result.iterations = iteration + 1;
    if (result.residual_norm <= convergence_threshold) {
      result.status = LocalNonlinearStatus::kConverged;
      return result;
    }
    if (problem.controls.step_tolerance > Real(0) &&
        result.step_norm <= problem.controls.step_tolerance) {
      result.status = LocalNonlinearStatus::kConverged;
      return result;
    }
  }
  result.status = LocalNonlinearStatus::kIterationLimit;
  return result;
}

POPS_HD inline int local_nonlinear_status_code(LocalNonlinearStatus status) {
  return static_cast<int>(status);
}

inline const char* local_nonlinear_status_name(LocalNonlinearStatus status) {
  switch (status) {
    case LocalNonlinearStatus::kConverged:
      return "converged";
    case LocalNonlinearStatus::kIterationLimit:
      return "iteration_limit";
    case LocalNonlinearStatus::kSingularJacobian:
      return "singular_jacobian";
    case LocalNonlinearStatus::kInadmissibleCandidate:
      return "inadmissible_candidate";
    case LocalNonlinearStatus::kSafeguardFailure:
      return "safeguard_failure";
    case LocalNonlinearStatus::kInvalidEvaluation:
      return "invalid_evaluation";
    case LocalNonlinearStatus::kUnsupportedCapability:
      return "unsupported_capability";
    case LocalNonlinearStatus::kEvaluationRetry:
      return "evaluation_retry";
    case LocalNonlinearStatus::kEvaluationReject:
      return "evaluation_reject";
    case LocalNonlinearStatus::kEvaluationFailed:
      return "evaluation_failed";
  }
  return "invalid_evaluation";
}

inline SolveStatus solve_status(LocalNonlinearStatus status) {
  switch (status) {
    case LocalNonlinearStatus::kConverged:
      return SolveStatus::kSolved;
    case LocalNonlinearStatus::kIterationLimit:
      return SolveStatus::kIterationLimit;
    case LocalNonlinearStatus::kSingularJacobian:
      return SolveStatus::kSingular;
    case LocalNonlinearStatus::kInadmissibleCandidate:
      return SolveStatus::kInadmissibleCandidate;
    case LocalNonlinearStatus::kSafeguardFailure:
      return SolveStatus::kSafeguardFailure;
    case LocalNonlinearStatus::kInvalidEvaluation:
      return SolveStatus::kInvalidEvaluation;
    case LocalNonlinearStatus::kUnsupportedCapability:
      return SolveStatus::kCapabilityFailure;
    case LocalNonlinearStatus::kEvaluationRetry:
    case LocalNonlinearStatus::kEvaluationReject:
    case LocalNonlinearStatus::kEvaluationFailed:
      return SolveStatus::kInvalidEvaluation;
  }
  return SolveStatus::kInvalidInput;
}

inline SolveReport local_nonlinear_solve_report(
    int status_code, int iterations, int evaluations, Real reference_residual_norm,
    Real residual_norm, Real step_norm, Real condition_evidence, int safeguard_steps,
    int failing_i = -1, int failing_j = -1, int failing_component = -1,
    SolveAction failure_action = SolveAction::kFailRun) {
  if (status_code < local_nonlinear_status_code(LocalNonlinearStatus::kConverged) ||
      status_code > local_nonlinear_status_code(LocalNonlinearStatus::kEvaluationFailed))
    throw std::invalid_argument("local nonlinear provider returned an unknown status code");
  const LocalNonlinearStatus local_status = static_cast<LocalNonlinearStatus>(status_code);
  SolveReport report;
  report.iters = iterations;
  report.evaluations = evaluations;
  report.reference_residual_norm = reference_residual_norm;
  report.residual_norm = residual_norm;
  const Real largest_finite = std::numeric_limits<Real>::max();
  report.rel_residual = reference_residual_norm > Real(0)
                            ? (reference_residual_norm >= residual_norm / largest_finite
                                   ? residual_norm / reference_residual_norm
                                   : largest_finite)
                            : residual_norm;
  report.step_norm = step_norm;
  report.condition_evidence = condition_evidence;
  report.safeguard_steps = safeguard_steps;
  report.failed_i = failing_i;
  report.failed_j = failing_j;
  report.failed_component = failing_component;
  if (local_status == LocalNonlinearStatus::kConverged)
    report.mark_solved("local_nonlinear_converged");
  else
    report.mark_failed(solve_status(local_status), failure_action,
                       std::string("local_nonlinear_") + local_nonlinear_status_name(local_status));
  return report;
}

}  // namespace pops
