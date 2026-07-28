#pragma once

/// @file
/// @brief Fail-closed implicit-source adapter for the prepared local nonlinear provider.
///
/// This header deliberately does not implement Newton.  It prepares a cell-local backward-Euler
/// residual/Jacobian/domain contract and delegates every iteration, safeguard and factorization to
/// `solve_prepared_local_nonlinear`.  Per-cell results are written to a candidate field; the live
/// state is published only after one collective `SolveReport` proves success on every MPI rank.

#include <pops/core/foundation/types.hpp>
#include <pops/core/state/state.hpp>
#include <pops/diagnostics/runtime_diagnostics.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/spatial_operator.hpp>  // load_state, load_aux
#include <pops/runtime/numerical_defaults.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdio>   // stderr diagnostics for an unconsumed outcome contract violation
#include <cstdlib>  // std::abort
#include <memory>
#include <sstream>    // structured host failure message, never in kernel
#include <stdexcept>  // explicit FailRun/contract errors, host after reductions
#include <string>     // std::string (validate_newton_options message prefix)

/// @file
/// @brief Implicit / IMEX block step as a named CONTRACT. Concept ImplicitBlockStepper,
///        ready-to-use default backward_euler_source (local Newton on the model stiff source)
///        and the ImplicitSourceStepper object used by isolated numerical tests.
///        Includes the partial-IMEX mask (ImplicitMask), the analytic Jacobian trait
///        (HasSourceJacobian) and the Newton options (NewtonOptions / NewtonReport).
///
/// Layer: `include/pops/numerics/time`.
/// Role: provide "an IMEX by default without the user writing Newton". backward_euler_source
///        solves IN PLACE W = U + dt S(W, aux) by local Newton (finite-difference Jacobian,
///        or analytic if the model declares source_jacobian); exact in one iteration if S is
///        linear, quadratic convergence otherwise, unconditionally stable for a stiff relaxation
///        (where Picard would diverge as soon as dt*stiffness > 1).
///
/// Invariants:
/// - the source is LOCAL (cell by cell): no flux coupling, no reflux;
/// - PARTIAL IMEX: a model can declare variable by variable which ones are stiff
///   (PartiallyImplicitModel::is_implicit), or an ImplicitMask carried by the BLOCK overrides the
///   model default. Inactive mask (default) -> all implicit -> bit-identical to before;
/// - NewtonOptions always carries a convergence criterion and a finite iteration budget;
/// - device kernels = NAMED functors (BackwardEulerSourceStatKernel, NewtonStat*Kernel);
/// - publication is transactional: every candidate is reduced into the common SolveOutcome and
///   consumed exactly once; invalid/non-converged candidates never overwrite the accepted state.

namespace pops {

template <class Stepper, class Coupler, class Block>
concept ImplicitBlockStepper =
    requires(const Stepper stepper, Coupler& coupler, Block& block, Real dt, int substep,
             int count) { stepper(coupler, block, dt, substep, count); };

template <class Model>
concept PartiallyImplicitModel = requires(int component) {
  { Model::is_implicit(component) } -> std::convertible_to<bool>;
};

template <class Model>
POPS_HD inline bool model_is_implicit(int component) {
  if constexpr (PartiallyImplicitModel<Model>)
    return Model::is_implicit(component);
  return true;
}

template <class Model>
concept HasSourceJacobian =
    requires(const Model model, const typename Model::State state, const Aux aux,
             Real (&jacobian)[Model::n_vars][Model::n_vars]) {
      model.source_jacobian(state, aux, jacobian);
    };

/// Device-safe result of one pointwise implicit source or source-Jacobian evaluation.
///
/// Fallible models expose
///   evaluate_source(state, aux, output)
/// and/or
///   evaluate_source_jacobian(state, aux, output_matrix)
/// returning this POD. Models that only expose the historical source()/source_jacobian() methods
/// remain valid and are adapted to kOk without a runtime branch. A non-success result never makes
/// the model-provided output observable outside the private Newton candidate.
enum class ImplicitEvaluationStatus : int {
  kOk = 0,
  kRetry = 1,
  kReject = 2,
  kFailed = 3,
  kInvalid = 4,
};

struct ImplicitEvaluationResult {
  ImplicitEvaluationStatus status = ImplicitEvaluationStatus::kInvalid;
  std::uint32_t reason_code = 0;

  POPS_HD static constexpr ImplicitEvaluationResult ok() {
    return {ImplicitEvaluationStatus::kOk, 0};
  }
  POPS_HD static constexpr ImplicitEvaluationResult retry(std::uint32_t reason) {
    return {ImplicitEvaluationStatus::kRetry, reason};
  }
  POPS_HD static constexpr ImplicitEvaluationResult reject(std::uint32_t reason) {
    return {ImplicitEvaluationStatus::kReject, reason};
  }
  POPS_HD static constexpr ImplicitEvaluationResult failed(std::uint32_t reason) {
    return {ImplicitEvaluationStatus::kFailed, reason};
  }
  POPS_HD static constexpr ImplicitEvaluationResult invalid(std::uint32_t reason) {
    return {ImplicitEvaluationStatus::kInvalid, reason};
  }
  POPS_HD constexpr bool succeeded() const { return status == ImplicitEvaluationStatus::kOk; }
};

template <class M>
concept HasFallibleSourceEvaluation =
    requires(const M m, const typename M::State u, const Aux a, typename M::State& output) {
      { m.evaluate_source(u, a, output) } -> std::same_as<ImplicitEvaluationResult>;
    };

template <class M>
concept HasFallibleSourceJacobianEvaluation =
    requires(const M m, const typename M::State u, const Aux a, Real (&J)[M::n_vars][M::n_vars]) {
      { m.evaluate_source_jacobian(u, a, J) } -> std::same_as<ImplicitEvaluationResult>;
    };

// Implicit mask CARRIED BY THE BLOCK / time policy (and NOT by the model): device-clean POD carrier
// (fixed array N, passed BY VALUE into the kernel, no host pointer dereference on device).
// When active (active == true), it OVERRIDES the model default (model_is_implicit): only the
// components with flag[c] == true are advanced implicitly, the others explicitly (forward Euler). This is what
// lets you REUSE the SAME model with different implicit treatments depending on the block. Inactive (default:
// active == false) -> falls back to model_is_implicit -> behavior bit-identical to the historical one.
template <int N>
struct ImplicitMask {
  bool active = false;
  bool flag[N] = {};
};

template <class Model, int N>
POPS_HD inline bool is_implicit_component(const ImplicitMask<N>& mask, int component) {
  return mask.active ? mask.flag[component] : model_is_implicit<Model>(component);
}

/// Options of the local Newton of the implicit source (backward-Euler). Every solve has an explicit
/// convergence criterion and a finite iteration budget; exhausting the budget is a typed failure,
/// never a publishable best-effort value. Device-clean POD (passed BY VALUE into the kernel).
///  - max_iters: Newton iteration budget.
///  - rel_tol / abs_tol: stop criterion ||F||_inf <= abs_tol + rel_tol*||F0||_inf, evaluated per
///    CELL at the start of an iteration. At least one tolerance must be positive.
///  - fd_eps: step (relative AND absolute floor) of the finite-difference Jacobian.
///  - damping: damping factor of the update W -= damping * delta. 1 (default) =
///    full Newton, bit-identical (multiplication by 1.0 exact in IEEE). < 1 = damped Newton
///    (very stiff source / poor conditioning: robustness at the cost of speed).
struct NewtonOptions {
  int max_iters = kNewtonDefaultMaxIters;
  Real rel_tol = kNewtonDefaultRelTol;
  Real abs_tol = kNewtonDefaultAbsTol;
  Real fd_eps = kNewtonDefaultFdEps;
  Real damping = kNewtonDefaultDamping;
};

/// Range-validate a NewtonOptions POD; shared by System::add_block and AmrSystem::add_block, which
/// carried this defensive check verbatim. @p where prefixes each message ("System::add_block" /
/// "AmrSystem::add_block"). This does NOT decide whether non-default options are ALLOWED -- the
/// time='imex' gate (and System's extra newton_diagnostics term in the non-default test) differ
/// between the two callers and remain at each call site.
inline void validate_newton_options(const NewtonOptions& newton, const char* where) {
  const std::string ctx = std::string(where) + " : ";
  if (newton.max_iters < 1)
    throw std::runtime_error(ctx + "newton_max_iters >= 1");
  if (newton.rel_tol < 0.0 || newton.abs_tol < 0.0 || newton.fd_eps <= 0.0)
    throw std::runtime_error(ctx + "newton_rel_tol/abs_tol >= 0 and newton_fd_eps > 0");
  if (newton.rel_tol == 0.0 && newton.abs_tol == 0.0)
    throw std::runtime_error(ctx + "at least one of newton_rel_tol/newton_abs_tol must be > 0");
  if (!(newton.damping > 0.0 && newton.damping <= 1.0))
    throw std::runtime_error(ctx + "newton_damping in (0, 1]");
}

enum class NewtonFailureKind : int {
  kNone = 0,
  kIterationLimit = 1,
  kSingular = 2,
  kEvaluationRetry = 3,
  kEvaluationReject = 4,
  kEvaluationFailed = 5,
  kInvalidEvaluation = 6,
};

/// OUTPUT statistic of the Newton of ONE cell (device POD, written into the diagnostics scratch):
/// res = ||F||_inf at exit; reference_res = ||F0||_inf; iters = iterations consumed;
/// failure identifies a non-finite evaluation, singular Jacobian, or exhausted active tolerance;
/// comp = index of the conserved COMPONENT carrying the max residual at exit (-1 if nothing implicit).
struct NewtonCellStat {
  Real res = Real(0);
  Real reference_res = Real(0);
  Real iters = Real(0);
  NewtonFailureKind failure = NewtonFailureKind::kNone;
  Real comp = Real(-1);
  std::uint32_t reason_code = 0;
};

/// AGGREGATED report (whole block, all substeps of one advance) of the implicit-source Newton.
/// Filled by backward_euler_source when a report is requested (OPT-IN diagnostics); max/sum
/// reductions over the cells + MPI all_reduce. reset() at the start of the advance by the caller.
/// OFFENDING CELL: (failed_i, failed_j, failed_comp) designate ONE failed cell -- the one
/// with MAXIMAL encoded index (j then i), enough to go inspect the state; -1 if none. failed_comp
/// is the conserved component carrying the worst residual of THAT cell.
struct NewtonReport {
  bool enabled = false;           ///< a report was computed (at least one instrumented substep)
  bool converged = true;          ///< no failed cell over the advance
  Real max_residual = Real(0);    ///< max over cells/substeps of ||F||_inf at exit
  Real max_iters_used = Real(0);  ///< max over cells/substeps of iterations consumed
  double n_failed =
      0;  ///< number of (cells x substeps) failed (non-finite / pivot / non-convergence)
  double failed_i = -1, failed_j = -1, failed_comp = -1;  ///< one offending cell (-1 if none)
  SolveReport solve{};  ///< authoritative typed outcome of the most recent local solve
  RuntimeDiagnosticsReport diagnostics =
      make_runtime_diagnostics_report("pops.numerics.time.prepared_local_nonlinear");

  void reset() { *this = NewtonReport{}; }
};

/// Finite? (device-safe, without <cmath>: NaN fails x == x; +-inf fails the bounds).
POPS_HD inline bool newton_finite(Real x) {
  return x == x && x < kNewtonFiniteAbsLimit && x > -kNewtonFiniteAbsLimit;
}

namespace detail {
inline constexpr std::uint32_t kImplicitUnknownEvaluationStatus = 0x4e570001u;
inline constexpr std::uint32_t kImplicitNonFiniteSource = 0x4e570002u;
inline constexpr std::uint32_t kImplicitNonFiniteJacobian = 0x4e570003u;

POPS_HD inline bool implicit_evaluation_status_known(ImplicitEvaluationStatus status) {
  switch (status) {
    case ImplicitEvaluationStatus::kOk:
    case ImplicitEvaluationStatus::kRetry:
    case ImplicitEvaluationStatus::kReject:
    case ImplicitEvaluationStatus::kFailed:
    case ImplicitEvaluationStatus::kInvalid:
      return true;
  }
  return false;
}

POPS_HD inline ImplicitEvaluationResult sanitize_implicit_evaluation(
    ImplicitEvaluationResult result) {
  return implicit_evaluation_status_known(result.status)
             ? result
             : ImplicitEvaluationResult::invalid(kImplicitUnknownEvaluationStatus);
}

POPS_HD inline NewtonFailureKind newton_evaluation_failure(ImplicitEvaluationStatus status) {
  switch (status) {
    case ImplicitEvaluationStatus::kOk:
      return NewtonFailureKind::kNone;
    case ImplicitEvaluationStatus::kRetry:
      return NewtonFailureKind::kEvaluationRetry;
    case ImplicitEvaluationStatus::kReject:
      return NewtonFailureKind::kEvaluationReject;
    case ImplicitEvaluationStatus::kFailed:
      return NewtonFailureKind::kEvaluationFailed;
    case ImplicitEvaluationStatus::kInvalid:
      return NewtonFailureKind::kInvalidEvaluation;
  }
  return NewtonFailureKind::kInvalidEvaluation;
}

template <class Model>
POPS_HD inline ImplicitEvaluationResult evaluate_implicit_source(const Model& model,
                                                                 const typename Model::State& state,
                                                                 const Aux& aux,
                                                                 typename Model::State& output) {
  if constexpr (HasFallibleSourceEvaluation<Model>)
    return sanitize_implicit_evaluation(model.evaluate_source(state, aux, output));
  else {
    output = model.source(state, aux);
    return ImplicitEvaluationResult::ok();
  }
}

template <class Model>
POPS_HD inline int first_non_finite_source_component(const typename Model::State& source) {
  for (int component = 0; component < Model::n_vars; ++component)
    if (!newton_finite(source[component]))
      return component;
  return -1;
}

// Dense solve J x = b on the leading n x n block (n <= N), partial pivoting. J and b
// destroyed. N is constexpr (= Model::n_vars) -> fixed array, no allocation,
// device-callable; n (<= N) is the number of implicit variables (partial IMEX).
// @return true if all pivots are finite and non-zero; false otherwise. The candidate transaction
// consumes false as a singular outcome and never publishes the partial update.
template <int N>
POPS_HD inline bool solve_dense(Real J[N][N], Real b[N], Real x[N], int n) {
  for (int p = 0; p < n; ++p) {
    int piv = p;
    Real best = J[p][p] < 0 ? -J[p][p] : J[p][p];
    for (int r = p + 1; r < n; ++r) {
      const Real v = J[r][p] < 0 ? -J[r][p] : J[r][p];
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (piv != p) {
      for (int c = 0; c < n; ++c) {
        const Real t = J[p][c];
        J[p][c] = J[piv][c];
        J[piv][c] = t;
      }
      const Real t = b[p];
      b[p] = b[piv];
      b[piv] = t;
    }
    const Real d = J[p][p];
    if (d == Real(0) || !newton_finite(d))
      return false;
    for (int r = 0; r < n; ++r) {
      if (r == p)
        continue;
      const Real f = J[r][p] / d;
      for (int c = p; c < n; ++c)
        J[r][c] -= f * J[p][c];
      b[r] -= f * b[p];
    }
  }
  for (int p = 0; p < n; ++p)
    x[p] = b[p] / J[p][p];
  return true;
}

// Assemble the Newton Jacobian of the subsystem reduced to the implicit components:
//   J[rr][cc] = (row == col ? 1: 0) - dt * dS_row/dW_col,   row = impl[rr], col = impl[cc].
// HasSourceJacobian trait present => ANALYTIC Jacobian (m.source_jacobian); otherwise finite
// differences (step h = fd_eps*|W| + fd_eps, source perturbed column by column around S0 = S(W)).
// Body EXTRACTED word-for-word from the two paths (2a defaults, 2b instrumented) -> bit-identical;
// POPS_HD because called from newton_source_solve (device-callable). S0 = m.source(W, a) already computed
// by the caller (reused as is by the finite differences).
template <class Model, int N>
POPS_HD inline ImplicitEvaluationResult assemble_newton_jacobian(
    const Model& m, const typename Model::State& W, const Aux& a, Real dt,
    const NewtonOptions& opts, const int impl[N], int m_impl, const typename Model::State& S0,
    Real J[N][N], int& invalid_component) {
  if constexpr (HasFallibleSourceJacobianEvaluation<Model>) {
    Real dS[N][N];
    const ImplicitEvaluationResult evaluation =
        sanitize_implicit_evaluation(m.evaluate_source_jacobian(W, a, dS));
    if (!evaluation.succeeded())
      return evaluation;
    for (int row = 0; row < N; ++row)
      for (int col = 0; col < N; ++col)
        if (!newton_finite(dS[row][col])) {
          invalid_component = row;
          return ImplicitEvaluationResult::invalid(kImplicitNonFiniteJacobian);
        }
    for (int cc = 0; cc < m_impl; ++cc)
      for (int rr = 0; rr < m_impl; ++rr) {
        const int row = impl[rr], col = impl[cc];
        J[rr][cc] = (row == col ? Real(1) : Real(0)) - dt * dS[row][col];
      }
  } else if constexpr (HasSourceJacobian<Model>) {
    // ANALYTIC JACOBIAN (trait, wave 3): J = I - dt * dS/dU restricted to the implicit ones.
    Real dS[N][N];
    m.source_jacobian(W, a, dS);
    for (int row = 0; row < N; ++row)
      for (int col = 0; col < N; ++col)
        if (!newton_finite(dS[row][col])) {
          invalid_component = row;
          return ImplicitEvaluationResult::invalid(kImplicitNonFiniteJacobian);
        }
    for (int cc = 0; cc < m_impl; ++cc)
      for (int rr = 0; rr < m_impl; ++rr) {
        const int row = impl[rr], col = impl[cc];
        J[rr][cc] = (row == col ? Real(1) : Real(0)) - dt * dS[row][col];
      }
  } else {
    for (int cc = 0; cc < m_impl; ++cc) {
      const int col = impl[cc];
      const Real wc = W[col] < 0 ? -W[col] : W[col];
      const Real h = opts.fd_eps * wc + opts.fd_eps;
      typename Model::State Wp = W;
      Wp[col] += h;
      typename Model::State Sp{};
      const ImplicitEvaluationResult evaluation = evaluate_implicit_source(m, Wp, a, Sp);
      if (!evaluation.succeeded())
        return evaluation;
      if (const int component = first_non_finite_source_component<Model>(Sp); component >= 0) {
        invalid_component = component;
        return ImplicitEvaluationResult::invalid(kImplicitNonFiniteSource);
      }
      for (int rr = 0; rr < m_impl; ++rr) {
        const int row = impl[rr];
        const Real dSdW = (Sp[row] - S0[row]) / h;
        J[rr][cc] = (row == col ? Real(1) : Real(0)) - dt * dSdW;
      }
    }
  }
  return ImplicitEvaluationResult::ok();
}

// Solve W such that W = Un + dt*S(W,a) in forward-backward Euler (partial IMEX):
//   - EXPLICIT components: forward Euler at the input state, W_e = Un_e + dt*S_e(Un);
//   - IMPLICIT components: Newton on the reduced subsystem, F_i = W_i - Un_i -
//     dt*S_i(W), Jacobian I - dt*(dS/dW) restricted to the implicit ones (columns by
//     finite differences), the explicit ones frozen at their advanced value (known data).
// WHO is implicit: a mask CARRIED BY THE BLOCK (@p mask) taking priority over the model default
// (is_implicit_component). Inactive mask (default) + model without is_implicit trait: all
// components are implicit -> full backward-Euler, strictly identical to the original behavior.
template <class Model>
POPS_HD inline typename Model::State newton_source_solve(
    const Model& m, const typename Model::State& Un, const Aux& a, Real dt,
    const NewtonOptions& opts, const ImplicitMask<Model::n_vars>& mask, NewtonCellStat& stat) {
  constexpr int N = Model::n_vars;
  typename Model::State explicit_target = initial;
  const typename Model::State initial_source = model.source(initial, aux);
  for (int component = 0; component < N; ++component)
    if (!is_implicit_component<Model>(mask, component))
      explicit_target[component] = initial[component] + dt * initial_source[component];

  typename Model::State W = Un;
  std::uint32_t evaluation_reason = 0;
  // (1) explicit: forward Euler on the non-implicit components (source at the input).
  if (m_impl < N) {
    typename Model::State S_in{};
    const ImplicitEvaluationResult evaluation = evaluate_implicit_source(m, Un, a, S_in);
    if (!evaluation.succeeded()) {
      stat.failure = newton_evaluation_failure(evaluation.status);
      stat.reason_code = evaluation.reason_code;
      return W;
    }
    if (const int component = first_non_finite_source_component<Model>(S_in); component >= 0) {
      stat.failure = NewtonFailureKind::kInvalidEvaluation;
      stat.comp = Real(component);
      stat.reason_code = kImplicitNonFiniteSource;
      return W;
    }
    for (int c = 0; c < N; ++c)
      if (!is_implicit_component<Model>(mask, c))
        W[c] = Un[c] + dt * S_in[c];
  }
  // Instrumented candidate path: Newton plus the stop
  // criterion ||F||_inf <= abs_tol + rel_tol*||F0||_inf at the start of an iteration, the detection of
  // non-finite residual / degenerate pivot, and the exit statistic. One ADDITIONAL source evaluation
  // may happen at exit (honest residual after the last update).
  // Invalid evaluations and singular Jacobians stop before an invalid update is applied. The
  // transaction discards that candidate, so no best-effort iterate can become accepted state.
  Real res = Real(0), res0 = Real(0);
  int used = 0;
  int worst_comp = -1;  // conserved component carrying the max residual at exit (diagnostic)
  NewtonFailureKind failure = NewtonFailureKind::kNone;
  bool converged = (m_impl == 0);  // nothing implicit: trivially converged
  for (int it = 0; it < opts.max_iters; ++it) {
    typename Model::State S0{};
    const ImplicitEvaluationResult source_evaluation = evaluate_implicit_source(m, W, a, S0);
    if (!source_evaluation.succeeded()) {
      failure = newton_evaluation_failure(source_evaluation.status);
      evaluation_reason = source_evaluation.reason_code;
      break;
    }
    if (const int component = first_non_finite_source_component<Model>(S0); component >= 0) {
      failure = NewtonFailureKind::kInvalidEvaluation;
      evaluation_reason = kImplicitNonFiniteSource;
      worst_comp = component;
      break;
    }
    Real F[N];
    res = Real(0);
    for (int r = 0; r < m_impl; ++r) {
      const int c = impl[r];
      F[r] = W[c] - Un[c] - dt * S0[c];
      if (!newton_finite(F[r])) {
        if (failure != NewtonFailureKind::kInvalidEvaluation)
          worst_comp = c;
        failure = NewtonFailureKind::kInvalidEvaluation;
        continue;
      }
      const Real av = F[r] < 0 ? -F[r] : F[r];
      if (failure != NewtonFailureKind::kInvalidEvaluation && av > res) {
        res = av;
        worst_comp = c;
      }
    }
    if (it == 0)
      res0 = res;
    if (!newton_finite(res))
      failure = NewtonFailureKind::kInvalidEvaluation;
    if (failure == NewtonFailureKind::kInvalidEvaluation)
      break;
    if (res <= opts.abs_tol + opts.rel_tol * res0) {
      converged = true;
      break;
    }
    Real J[N][N];
    int invalid_component = -1;
    const ImplicitEvaluationResult jacobian_evaluation = assemble_newton_jacobian<Model, N>(
        m, W, a, dt, opts, impl, m_impl, S0, J, invalid_component);
    if (!jacobian_evaluation.succeeded()) {
      failure = newton_evaluation_failure(jacobian_evaluation.status);
      evaluation_reason = jacobian_evaluation.reason_code;
      worst_comp = invalid_component;
      used = it + 1;
      break;
    }
    Real delta[N];
    const bool ok = solve_dense<N>(J, F, delta, m_impl);
    if (!ok) {
      failure = NewtonFailureKind::kSingular;
      used = it + 1;
      break;
    }
    for (int r = 0; r < m_impl; ++r)
      W[impl[r]] -= opts.damping * delta[r];
    used = it + 1;
  }
  // Exit by budget exhaustion: recompute the residual AFTER the last update (honest
  // report; the loop residual precedes the update). One extra source evaluation,
  // only on this instrumented path.
  if (failure == NewtonFailureKind::kNone && used == opts.max_iters && m_impl > 0) {
    typename Model::State S0{};
    const ImplicitEvaluationResult source_evaluation = evaluate_implicit_source(m, W, a, S0);
    if (!source_evaluation.succeeded()) {
      failure = newton_evaluation_failure(source_evaluation.status);
      evaluation_reason = source_evaluation.reason_code;
    }
    if (failure == NewtonFailureKind::kNone) {
      if (const int component = first_non_finite_source_component<Model>(S0); component >= 0) {
        failure = NewtonFailureKind::kInvalidEvaluation;
        evaluation_reason = kImplicitNonFiniteSource;
        worst_comp = component;
      }
    }
    res = Real(0);
    for (int r = 0; failure == NewtonFailureKind::kNone && r < m_impl; ++r) {
      const int c = impl[r];
      const Real fr = W[c] - Un[c] - dt * S0[c];
      if (!newton_finite(fr)) {
        if (failure != NewtonFailureKind::kInvalidEvaluation)
          worst_comp = c;
        failure = NewtonFailureKind::kInvalidEvaluation;
        continue;
      }
      const Real av = fr < 0 ? -fr : fr;
      if (failure != NewtonFailureKind::kInvalidEvaluation && av > res) {
        res = av;
        worst_comp = c;
      }
    }
    if (!newton_finite(res))
      failure = NewtonFailureKind::kInvalidEvaluation;
    else
      converged = res <= opts.abs_tol + opts.rel_tol * res0;
  }
  if (failure == NewtonFailureKind::kNone && !converged)
    failure = NewtonFailureKind::kIterationLimit;
  stat.res = res;
  stat.reference_res = res0;
  stat.iters = Real(used);
  stat.failure = failure;
  stat.comp = Real(worst_comp);
  stat.reason_code = evaluation_reason;
  return W;
}
}  // namespace detail

namespace detail {
// INSTRUMENTED variant: same Newton, but writes the exit statistic of EACH cell into
// the scratch st (comp 0 = ||F||_inf, 1 = ||F0||_inf, 2 = iterations,
// 3 = NewtonFailureKind, 4 = ENCODED OFFENDING CELL, 5/6 = high/low 16-bit halves of
// the evaluation reason code. Splitting keeps all 32 reason bits exact in float and double builds.
// Encoding comp 4: -1 if the cell did not fail; otherwise (j*2^20 + i)*16 + (offending_comp + 1) --
// exact integer in double up to ~2^44 (i, j < 2^20), so that a MAX reduction yields ONE offending
// cell (the largest in index) decodable host side without a dedicated arg-max reduction. NAMED
// FUNCTOR used for every local solve so invalid candidates never have an unchecked fast path.
template <class Model>
struct BackwardEulerSourceStatKernel {
  Model m;
  ConstArray4 uc, ax;
  Array4 u, st;
  Real dt;
  NewtonOptions opts;
  ImplicitMask<Model::n_vars> mask;
  POPS_HD void operator()(int i, int j) const {
    const typename Model::State Un = load_state<Model>(uc, i, j);
    const Aux a = load_aux<aux_comps<Model>()>(ax, i, j);
    NewtonCellStat s{};
    const typename Model::State W = newton_source_solve<Model>(m, Un, a, dt, opts, mask, s);
    for (int c = 0; c < Model::n_vars; ++c)
      u(i, j, c) = W[c];
    st(i, j, 0) = s.res;
    st(i, j, 1) = s.reference_res;
    st(i, j, 2) = s.iters;
    st(i, j, 3) = Real(static_cast<int>(s.failure));
    st(i, j, 4) = s.failure != NewtonFailureKind::kNone
                      ? (Real(j) * Real(1048576) + Real(i)) * Real(16) + (s.comp + Real(1))
                      : Real(-1);
    st(i, j, 5) = Real(s.reason_code >> 16);
    st(i, j, 6) = Real(s.reason_code & 0xffffu);
  }
};

template <class Model>
struct BackwardEulerSourceActiveStatKernel {
  Model m;
  ConstArray4 uc, ax, active_cells;
  Array4 u, st;
  Real dt;
  NewtonOptions opts;
  ImplicitMask<Model::n_vars> mask;
  POPS_HD void operator()(int i, int j) const {
    if (active_cells(i, j, 0) < Real(0.5)) {
      st(i, j, 0) = Real(0);
      st(i, j, 1) = Real(0);
      st(i, j, 2) = Real(0);
      st(i, j, 3) = Real(0);
      st(i, j, 4) = Real(-1);
      st(i, j, 5) = Real(0);
      st(i, j, 6) = Real(0);
      return;
    }
    const typename Model::State Un = load_state<Model>(uc, i, j);
    const Aux a = load_aux<aux_comps<Model>()>(ax, i, j);
    NewtonCellStat s{};
    const typename Model::State W = newton_source_solve<Model>(m, Un, a, dt, opts, mask, s);
    for (int c = 0; c < Model::n_vars; ++c)
      u(i, j, c) = W[c];
    st(i, j, 0) = s.res;
    st(i, j, 1) = s.reference_res;
    st(i, j, 2) = s.iters;
    st(i, j, 3) = Real(static_cast<int>(s.failure));
    st(i, j, 4) = s.failure != NewtonFailureKind::kNone
                      ? (Real(j) * Real(1048576) + Real(i)) * Real(16) + (s.comp + Real(1))
                      : Real(-1);
    st(i, j, 5) = Real(s.reason_code >> 16);
    st(i, j, 6) = Real(s.reason_code & 0xffffu);
  }
};

struct LocalStatMax {
  ConstArray4 values;
  int component = 0;
  POPS_HD void operator()(int i, int j, Real& result) const {
    const Real value = values(i, j, component);
    if (value > result)
      result = value;
  }
};
struct NewtonStatFailureCountKernel {
  ConstArray4 st;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    if (st(i, j, 3) > Real(0))
      acc += Real(1);
  }
};
struct NewtonStatMaxForFailureKernel {
  ConstArray4 st;
  int failure;
  int comp;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    if (static_cast<int>(st(i, j, 3)) != failure)
      return;
    const Real value = st(i, j, comp);
    if (value > acc)
      acc = value;
  }
};
struct NewtonStatReasonLowKernel {
  ConstArray4 st;
  int failure;
  int reason_high;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    if (static_cast<int>(st(i, j, 3)) != failure || static_cast<int>(st(i, j, 5)) != reason_high)
      return;
    const Real value = st(i, j, 6);
    if (value > acc)
      acc = value;
  }
};

inline Real collective_max_component(const MultiFab& statistics, int component,
                                     Real initial = Real(0)) {
  Real local = initial;
  for (int local_index = 0; local_index < statistics.local_size(); ++local_index) {
    const ConstArray4 values = statistics.fab(local_index).const_array();
    local = std::max(local,
                     reduce_max_cell(statistics.box(local_index), LocalStatMax{values, component}));
  }
  return static_cast<Real>(all_reduce_max(static_cast<double>(local)));
}

inline double collective_sum_component(const MultiFab& statistics, int component) {
  Real local = Real(0);
  for (int local_index = 0; local_index < statistics.local_size(); ++local_index) {
    const ConstArray4 values = statistics.fab(local_index).const_array();
    local += reduce_sum_cell(statistics.box(local_index), LocalStatSum{values, component});
  }
  return all_reduce_sum(static_cast<double>(local));
}

inline void aggregate_legacy_report(NewtonReport* aggregate, const SolveReport& current,
                                    double failed_cells) {
  if (aggregate == nullptr)
    return;
  aggregate->enabled = true;
  aggregate->latest = current;
  aggregate->max_residual = std::max(aggregate->max_residual, current.residual_norm);
  aggregate->max_iters_used = std::max(aggregate->max_iters_used, static_cast<Real>(current.iters));
  aggregate->n_failed += failed_cells;
  if (!current.solved()) {
    aggregate->converged = false;
    aggregate->failed_i = current.failed_i;
    aggregate->failed_j = current.failed_j;
    aggregate->failed_comp = current.failed_component;
  }
}

}  // namespace detail

namespace detail {

/// Publication state owned by the common SolveOutcome returned by a local implicit solve. Keeping
/// the candidate here makes the local route obey the same one-shot contract as Krylov, field and
/// hierarchy solves instead of maintaining a second outcome implementation.
struct ImplicitSourcePublication {
  MultiFab* destination = nullptr;
  std::unique_ptr<MultiFab> candidate;
  NewtonReport* diagnostics = nullptr;
  std::string failure_message;

  bool layout_matches() const noexcept {
    return destination != nullptr && candidate != nullptr &&
           destination->box_array().boxes() == candidate->box_array().boxes() &&
           destination->dmap().ranks() == candidate->dmap().ranks() &&
           destination->ncomp() == candidate->ncomp() &&
           destination->n_grow() == candidate->n_grow() &&
           destination->local_size() == candidate->local_size();
  }

  static void validate_accept(void* context) {
    const auto& publication = *static_cast<const ImplicitSourcePublication*>(context);
    if (!publication.layout_matches())
      throw std::logic_error(
          "cannot accept a local implicit SolveOutcome after its destination layout changed");
  }

  static void accept(void* context) noexcept {
    auto& publication = *static_cast<ImplicitSourcePublication*>(context);
    lincomb(*publication.destination, Real(1), *publication.candidate, Real(0),
            *publication.candidate);
  }

  static void consume_failure(void* context, SolveConsumption action) {
    auto& publication = *static_cast<ImplicitSourcePublication*>(context);
    if (publication.diagnostics == nullptr)
      return;
    publication.diagnostics->solve.action = action == SolveConsumption::kRejectAttempt
                                                ? SolveAction::kRejectAttempt
                                                : SolveAction::kFailRun;
    publication.diagnostics->diagnostics.record(
        action == SolveConsumption::kRejectAttempt ? "newton.outcome.reject_attempt"
                                                   : "newton.outcome.fail_run",
        "ImplicitSourceNewton", action == SolveConsumption::kRejectAttempt ? "warning" : "error",
        publication.failure_message, -1, publication.diagnostics->n_failed);
  }
};

}  // namespace detail

inline SolveStatus newton_failure_status(NewtonFailureKind failure) {
  switch (failure) {
    case NewtonFailureKind::kNone:
      return SolveStatus::kSolved;
    case NewtonFailureKind::kIterationLimit:
      return SolveStatus::kIterationLimit;
    case NewtonFailureKind::kSingular:
      return SolveStatus::kSingular;
    case NewtonFailureKind::kEvaluationRetry:
    case NewtonFailureKind::kEvaluationReject:
    case NewtonFailureKind::kEvaluationFailed:
    case NewtonFailureKind::kInvalidEvaluation:
      return SolveStatus::kInvalidEvaluation;
  }
  return SolveStatus::kInvalidEvaluation;
}

inline SolveAction newton_failure_action(NewtonFailureKind failure) {
  switch (failure) {
    case NewtonFailureKind::kEvaluationRetry:
    case NewtonFailureKind::kEvaluationReject:
    case NewtonFailureKind::kIterationLimit:
      return SolveAction::kRejectAttempt;
    case NewtonFailureKind::kNone:
      return SolveAction::kNone;
    case NewtonFailureKind::kSingular:
    case NewtonFailureKind::kEvaluationFailed:
    case NewtonFailureKind::kInvalidEvaluation:
      return SolveAction::kFailRun;
  }
  return SolveAction::kFailRun;
}

inline const char* newton_failure_reason(NewtonFailureKind failure) {
  switch (failure) {
    case NewtonFailureKind::kNone:
      return "converged";
    case NewtonFailureKind::kIterationLimit:
      return "iteration_limit";
    case NewtonFailureKind::kSingular:
      return "singular";
    case NewtonFailureKind::kEvaluationRetry:
      return "evaluation_retry";
    case NewtonFailureKind::kEvaluationReject:
      return "evaluation_reject";
    case NewtonFailureKind::kEvaluationFailed:
      return "evaluation_failed";
    case NewtonFailureKind::kInvalidEvaluation:
      return "invalid_evaluation";
  }
  return "invalid_evaluation";
}

// Prepare W = U + dt * model.source(W, aux) without publishing W. The candidate is solved by local
// Newton, reduced collectively into one SolveReport, and returned in the common SolveOutcome that
// must be consumed exactly once. Only consume(kAccept) can mutate U.
template <class Model>
[[nodiscard]] SolveOutcome backward_euler_source(const Model& model, const MultiFab& aux,
                                                 MultiFab& U, Real dt, const NewtonOptions& opts,
                                                 const ImplicitMask<Model::n_vars>& mask = {},
                                                 NewtonReport* report = nullptr,
                                                 const MultiFab* active_cells = nullptr) {
  if (active_cells != nullptr &&
      (active_cells->ncomp() != 1 || active_cells->local_size() != state.local_size()))
    throw std::invalid_argument(
        "Implicit source active-cell mask must have one component and match the state layout");

  auto candidate = std::make_unique<MultiFab>(U.box_array(), U.dmap(), U.ncomp(), U.n_grow());
  lincomb(*candidate, Real(1), U, Real(0), U);
  MultiFab stats(U.box_array(), U.dmap(), 7, 0);
  for (int li = 0; li < candidate->local_size(); ++li) {
    Array4 u = candidate->fab(li).array();
    Array4 st = stats.fab(li).array();
    const ConstArray4 uc = candidate->fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    const Box2D b = candidate->box(li);
    if (active_cells != nullptr)
      for_each_cell(b,
                    detail::BackwardEulerSourceActiveStatKernel<Model>{
                        model, uc, ax, active_cells->fab(li).const_array(), u, st, dt, opts, mask});
    else
      for_each_cell(
          b, detail::BackwardEulerSourceStatKernel<Model>{model, uc, ax, u, st, dt, opts, mask});
  }
  Real rmax = Real(0), reference_max = Real(0), imax = Real(0), nfail = Real(0);
  Real failure_max = Real(0);
  for (int li = 0; li < stats.local_size(); ++li) {
    const ConstArray4 st = stats.fab(li).const_array();
    const Box2D b = stats.box(li);
    rmax = std::max(rmax, reduce_max_cell(b, detail::NewtonStatMaxKernel{st, 0}));
    reference_max = std::max(reference_max, reduce_max_cell(b, detail::NewtonStatMaxKernel{st, 1}));
    imax = std::max(imax, reduce_max_cell(b, detail::NewtonStatMaxKernel{st, 2}));
    nfail += reduce_sum_cell(b, detail::NewtonStatFailureCountKernel{st});
    failure_max = std::max(failure_max, reduce_max_cell(b, detail::NewtonStatMaxKernel{st, 3}));
  }
  rmax = static_cast<Real>(all_reduce_max(static_cast<double>(rmax)));
  reference_max = static_cast<Real>(all_reduce_max(static_cast<double>(reference_max)));
  imax = static_cast<Real>(all_reduce_max(static_cast<double>(imax)));
  const double nfail_g = all_reduce_sum(static_cast<double>(nfail));
  const int failure_g = static_cast<int>(all_reduce_max(static_cast<double>(failure_max)));
  Real selected_enc = Real(-1), selected_reason_high = Real(0);
  if (failure_g != static_cast<int>(NewtonFailureKind::kNone)) {
    for (int li = 0; li < stats.local_size(); ++li) {
      const ConstArray4 st = stats.fab(li).const_array();
      const Box2D b = stats.box(li);
      selected_enc =
          std::max(selected_enc,
                   reduce_max_cell(b, detail::NewtonStatMaxForFailureKernel{st, failure_g, 4}));
      selected_reason_high =
          std::max(selected_reason_high,
                   reduce_max_cell(b, detail::NewtonStatMaxForFailureKernel{st, failure_g, 5}));
    }
  }
  const double enc_g = all_reduce_max(static_cast<double>(selected_enc));
  const int reason_high_g =
      static_cast<int>(all_reduce_max(static_cast<double>(selected_reason_high)));
  Real selected_reason_low = Real(0);
  if (failure_g != static_cast<int>(NewtonFailureKind::kNone)) {
    for (int li = 0; li < stats.local_size(); ++li) {
      const ConstArray4 st = stats.fab(li).const_array();
      const Box2D b = stats.box(li);
      selected_reason_low = std::max(
          selected_reason_low,
          reduce_max_cell(b, detail::NewtonStatReasonLowKernel{st, failure_g, reason_high_g}));
    }
  }
  const auto reason_low_g =
      static_cast<std::uint32_t>(all_reduce_max(static_cast<double>(selected_reason_low)));
  const std::uint32_t reason_g = (static_cast<std::uint32_t>(reason_high_g) << 16) | reason_low_g;
  double fi = -1, fj = -1, fc = -1;
  if (nfail_g > 0 && enc_g >= 0) {  // decode the offending cell with maximal index (cf. StatKernel)
    const long long k = static_cast<long long>(enc_g);
    fc = static_cast<double>(k % 16) - 1.0;  // -1 = unknown component (nothing implicit)
    const long long cell = k / 16;
    fi = static_cast<double>(cell % 1048576);
    fj = static_cast<double>(cell / 1048576);
  }

  SolveReport solve;
  solve.iters = static_cast<int>(imax);
  solve.reference_residual_norm = newton_finite(reference_max) ? reference_max : Real(0);
  solve.residual_norm = newton_finite(rmax) ? rmax : Real(0);
  solve.rel_residual = solve.reference_residual_norm > Real(0)
                           ? solve.residual_norm / solve.reference_residual_norm
                           : solve.residual_norm;
  NewtonFailureKind failure = static_cast<NewtonFailureKind>(failure_g);
  if (!newton_finite(solve.rel_residual)) {
    solve.rel_residual = Real(0);
    failure = NewtonFailureKind::kInvalidEvaluation;
  }
  if (failure == NewtonFailureKind::kNone)
    solve.mark_solved("implicit_source_newton_converged");
  else {
    std::string reason = std::string("implicit_source_newton_") + newton_failure_reason(failure);
    if (reason_g != 0)
      reason += "_reason_" + std::to_string(reason_g);
    solve.mark_failed(newton_failure_status(failure), newton_failure_action(failure),
                      std::move(reason));
  }
  if (!solve_report_is_publishable(solve, opts.max_iters))
    throw std::runtime_error("implicit source Newton produced a malformed SolveReport");

  std::ostringstream message;
  if (!solve.solved_value_available())
    message << "Implicit source Newton: " << nfail_g << " cell(s) failed (" << solve.status_name()
            << ", max residual " << static_cast<double>(rmax) << "; cell (" << fi << ", " << fj
            << "), component " << fc << ")";

  if (report) {
    report->enabled = true;
    report->max_residual = std::max(report->max_residual, rmax);
    report->max_iters_used = std::max(report->max_iters_used, imax);
    report->n_failed += nfail_g;
    if (nfail_g > 0) {
      report->failed_i = fi;
      report->failed_j = fj;
      report->failed_comp = fc;
    }
    if (!solve.solved_value_available())
      report->converged = false;
    report->solve = solve;
  }
  auto publication = std::make_shared<detail::ImplicitSourcePublication>(
      detail::ImplicitSourcePublication{&U, std::move(candidate), report, message.str()});
  return SolveOutcome::collective_world(
      std::move(solve),
      SolveOutcome::PublicationHooks{publication.get(), &detail::ImplicitSourcePublication::accept,
                                     nullptr, nullptr, std::static_pointer_cast<void>(publication),
                                     &detail::ImplicitSourcePublication::validate_accept,
                                     &detail::ImplicitSourcePublication::consume_failure});
}

/// Consume one prepared local solve with the fail-fast runtime policy used by native engine routes.
/// The caller still owns the outcome and names the policy at the publication boundary.
inline SolveReport consume_implicit_source_fail_run(SolveOutcome& outcome) {
  if (outcome.report().solved_value_available())
    return outcome.consume(SolveConsumption::kAccept);
  const std::string status = outcome.report().status_name();
  const std::string reason = outcome.report().reason;
  const SolveReport failed = outcome.consume(SolveConsumption::kFailRun);
  throw std::runtime_error("Implicit source Newton failed: status=" + status + " reason=" + reason +
                           " action=" + failed.action_name());
}

// Default implicit stepper: backward-Euler (Newton) on the model source.
// Models ImplicitBlockStepper; passed as is to the test-only reference driver as the implicit
// advance callback. The user writes no solver.
struct ImplicitSourceStepper {
  NewtonOptions options{};

  template <class Coupler, class Block>
  void operator()(Coupler& coupler, Block& block, Real dt, int /*substep*/, int /*nsub*/) const {
    auto outcome = backward_euler_source(block.model, coupler.aux(), block.U(), dt, options);
    (void)consume_implicit_source_fail_run(outcome);
  }
};

}  // namespace pops
