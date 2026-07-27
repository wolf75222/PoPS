#pragma once

#include <pops/core/state/state.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/diagnostics/runtime_diagnostics.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/numerics/spatial_operator.hpp>  // load_state, load_aux
#include <pops/runtime/numerical_defaults.hpp>

#include <algorithm>  // std::max (Newton report aggregation, host)
#include <concepts>
#include <cstdio>     // stderr diagnostics for an unconsumed outcome contract violation
#include <cstdlib>    // std::abort
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
/// - publication is transactional: every candidate is reduced into an ImplicitSolveOutcome and
///   consumed exactly once; invalid/non-converged candidates never overwrite the accepted state.

namespace pops {

// Contract of an implicit/IMEX block stepper. Any object (or lambda) that knows how to
// advance a block over dt by reading the coupler (for up-to-date aux / phi) and the model.
template <class Stepper, class Coupler, class Block>
concept ImplicitBlockStepper =
    requires(const Stepper st, Coupler& c, Block& b, Real dt, int s, int n) { st(c, b, dt, s, n); };

// OPTIONAL trait: a model can declare which conserved variables are treated
// implicitly (the stiff ones). is_implicit(c) -> bool. A model WITHOUT this trait is treated
// fully implicitly (historical default).
template <class M>
concept PartiallyImplicitModel = requires(int c) {
  { M::is_implicit(c) } -> std::convertible_to<bool>;
};

// Is component c of the model implicit? Default (no trait): all of them are.
template <class Model>
POPS_HD inline bool model_is_implicit(int c) {
  if constexpr (PartiallyImplicitModel<Model>)
    return Model::is_implicit(c);
  else
    return true;
}

// OPTIONAL trait: ANALYTIC JACOBIAN of the source (review wave 3, JacobianPolicy). When the
// model (or its source brick, forwarded by CompositeModel) declares
//   source_jacobian(U, aux, J)  with  J[r][c] = dS_r/dU_c  (FULL n_vars x n_vars matrix),
// the implicit-source Newton uses it INSTEAD of finite differences: exactness
// (no more fd_eps noise) and n_impl source evaluations saved per iteration. A model
// WITHOUT the trait keeps the historical finite differences, bit-identical. POPS_HD required.
template <class M>
concept HasSourceJacobian =
    requires(const M m, const typename M::State u, const Aux a, Real (&J)[M::n_vars][M::n_vars]) {
      m.source_jacobian(u, a, J);
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

// Is component c implicit, with the block mask TAKING PRIORITY over the model default? The inactive mask
// (default) delegates to model_is_implicit<Model> -> strictly identical to before this change.
template <class Model, int N>
POPS_HD inline bool is_implicit_component(const ImplicitMask<N>& mask, int c) {
  if (mask.active)
    return mask.flag[c];
  return model_is_implicit<Model>(c);
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
  kInvalidEvaluation = 3,
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
      make_runtime_diagnostics_report("pops.numerics.time.implicit_newton");
  void reset() { *this = NewtonReport{}; }
};

/// Finite? (device-safe, without <cmath>: NaN fails x == x; +-inf fails the bounds).
POPS_HD inline bool newton_finite(Real x) {
  return x == x && x < kNewtonFiniteAbsLimit && x > -kNewtonFiniteAbsLimit;
}

namespace detail {
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
POPS_HD inline void assemble_newton_jacobian(const Model& m, const typename Model::State& W,
                                             const Aux& a, Real dt, const NewtonOptions& opts,
                                             const int impl[N], int m_impl,
                                             const typename Model::State& S0, Real J[N][N]) {
  if constexpr (HasSourceJacobian<Model>) {
    // ANALYTIC JACOBIAN (trait, wave 3): J = I - dt * dS/dU restricted to the implicit ones.
    Real dS[N][N];
    m.source_jacobian(W, a, dS);
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
      const typename Model::State Sp = m.source(Wp, a);
      for (int rr = 0; rr < m_impl; ++rr) {
        const int row = impl[rr];
        const Real dSdW = (Sp[row] - S0[row]) / h;
        J[rr][cc] = (row == col ? Real(1) : Real(0)) - dt * dSdW;
      }
    }
  }
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
  int impl[N];  // indices of the implicit components (the first m_impl useful slots)
  int m_impl = 0;
  for (int c = 0; c < N; ++c)
    if (is_implicit_component<Model>(mask, c))
      impl[m_impl++] = c;

  typename Model::State W = Un;
  // (1) explicit: forward Euler on the non-implicit components (source at the input).
  if (m_impl < N) {
    const typename Model::State S_in = m.source(Un, a);
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
    const typename Model::State S0 = m.source(W, a);
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
    assemble_newton_jacobian<Model, N>(m, W, a, dt, opts, impl, m_impl, S0, J);
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
    const typename Model::State S0 = m.source(W, a);
    res = Real(0);
    for (int r = 0; r < m_impl; ++r) {
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
  return W;
}
}  // namespace detail

namespace detail {
// INSTRUMENTED variant: same Newton, but writes the exit statistic of EACH cell into
// the scratch st (comp 0 = ||F||_inf, 1 = ||F0||_inf, 2 = iterations,
// 3 = NewtonFailureKind, 4 = ENCODED OFFENDING CELL).
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
  }
};

/// REDUCTION kernels of the diagnostics scratch (max / sum of one component). NAMED FUNCTORS
/// passed directly to reduce_max_cell / reduce_sum_cell (device-clean path, cf. for_each.hpp).
struct NewtonStatMaxKernel {
  ConstArray4 st;
  int comp;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    const Real v = st(i, j, comp);
    if (v > acc)
      acc = v;
  }
};
struct NewtonStatFailureCountKernel {
  ConstArray4 st;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    if (st(i, j, 3) > Real(0))
      acc += Real(1);
  }
};
}  // namespace detail

enum class ImplicitSolveConsumption {
  kAccept,
  kRejectAttempt,
  kFailRun,
};

/// Transactional result of one local implicit-source solve.
///
/// The candidate state is private until exactly one explicit consumption action is selected.
/// Accept publishes a solved candidate; RejectAttempt and FailRun discard a failed candidate.
/// Destroying an unconsumed outcome is a contract violation and terminates instead of silently
/// dropping the solver result.
class ImplicitSolveOutcome {
 public:
  ImplicitSolveOutcome(MultiFab& destination, std::unique_ptr<MultiFab> candidate,
                       SolveReport solve, NewtonReport* diagnostics, std::string failure_message)
      : destination_(&destination),
        candidate_(std::move(candidate)),
        solve_(std::move(solve)),
        diagnostics_(diagnostics),
        failure_message_(std::move(failure_message)) {}

  ImplicitSolveOutcome(const ImplicitSolveOutcome&) = delete;
  ImplicitSolveOutcome& operator=(const ImplicitSolveOutcome&) = delete;
  ImplicitSolveOutcome& operator=(ImplicitSolveOutcome&&) = delete;
  ImplicitSolveOutcome(ImplicitSolveOutcome&& other) noexcept
      : destination_(other.destination_),
        candidate_(std::move(other.candidate_)),
        solve_(std::move(other.solve_)),
        diagnostics_(other.diagnostics_),
        failure_message_(std::move(other.failure_message_)),
        consumed_(other.consumed_) {
    other.consumed_ = true;
    other.destination_ = nullptr;
    other.diagnostics_ = nullptr;
  }

  ~ImplicitSolveOutcome() {
    if (!consumed_) {
      std::fputs(
          "PoPS contract violation: ImplicitSolveOutcome destroyed before explicit consumption\n",
          stderr);
      std::fflush(stderr);
      std::abort();
    }
  }

  const SolveReport& report() const noexcept { return solve_; }
  const std::string& failure_message() const noexcept { return failure_message_; }

  SolveReport consume(ImplicitSolveConsumption action) {
    if (consumed_)
      throw std::logic_error("ImplicitSolveOutcome has already been consumed");

    if (action == ImplicitSolveConsumption::kAccept) {
      if (!solve_.solved_value_available())
        throw std::logic_error("cannot accept a failed ImplicitSolveOutcome");
      if (!publication_layout_matches())
        throw std::logic_error(
            "cannot accept an ImplicitSolveOutcome after its destination layout changed");
    } else if (action == ImplicitSolveConsumption::kRejectAttempt ||
               action == ImplicitSolveConsumption::kFailRun) {
      if (solve_.solved_value_available())
        throw std::logic_error("cannot reject a solved ImplicitSolveOutcome");
    } else
      throw std::logic_error("invalid ImplicitSolveConsumption action");

    // From this point the action and its candidate status are compatible. Mark the outcome consumed
    // immediately before the irreversible publication/discard boundary. In particular, a backend
    // exception while launching/copying the accepted candidate propagates to the enclosing
    // StepTransaction, which restores the accepted state; retrying a potentially partial publication
    // through the same outcome is forbidden.
    consumed_ = true;
    if (action == ImplicitSolveConsumption::kAccept) {
      lincomb(*destination_, Real(1), *candidate_, Real(0), *candidate_);
    } else {
      solve_.action = action == ImplicitSolveConsumption::kRejectAttempt
                          ? SolveAction::kRejectAttempt
                          : SolveAction::kFailRun;
      if (diagnostics_) {
        diagnostics_->diagnostics.record(
            action == ImplicitSolveConsumption::kRejectAttempt ? "newton.outcome.reject_attempt"
                                                               : "newton.outcome.fail_run",
            "ImplicitSourceNewton",
            action == ImplicitSolveConsumption::kRejectAttempt ? "warning" : "error",
            failure_message_, -1, diagnostics_->n_failed);
      }
    }
    if (diagnostics_)
      diagnostics_->solve = solve_;
    return solve_;
  }

 private:
  bool publication_layout_matches() const noexcept {
    return destination_ != nullptr && candidate_ != nullptr &&
           destination_->box_array().boxes() == candidate_->box_array().boxes() &&
           destination_->dmap().ranks() == candidate_->dmap().ranks() &&
           destination_->ncomp() == candidate_->ncomp() &&
           destination_->n_grow() == candidate_->n_grow() &&
           destination_->local_size() == candidate_->local_size();
  }

  MultiFab* destination_ = nullptr;
  std::unique_ptr<MultiFab> candidate_;
  SolveReport solve_;
  NewtonReport* diagnostics_ = nullptr;
  std::string failure_message_;
  bool consumed_ = false;
};

inline SolveStatus newton_failure_status(NewtonFailureKind failure) {
  switch (failure) {
    case NewtonFailureKind::kNone:
      return SolveStatus::kSolved;
    case NewtonFailureKind::kIterationLimit:
      return SolveStatus::kIterationLimit;
    case NewtonFailureKind::kSingular:
      return SolveStatus::kSingular;
    case NewtonFailureKind::kInvalidEvaluation:
      return SolveStatus::kInvalidEvaluation;
  }
  return SolveStatus::kInvalidEvaluation;
}

// Prepare W = U + dt * model.source(W, aux) without publishing W. The candidate is solved by local
// Newton, reduced collectively into one SolveReport, and returned in an ImplicitSolveOutcome that
// must be consumed exactly once. Only consume(kAccept) can mutate U.
template <class Model>
[[nodiscard]] ImplicitSolveOutcome backward_euler_source(
    const Model& model, const MultiFab& aux, MultiFab& U, Real dt, const NewtonOptions& opts,
    const ImplicitMask<Model::n_vars>& mask = {}, NewtonReport* report = nullptr,
    const MultiFab* active_cells = nullptr) {
  if (active_cells != nullptr &&
      (active_cells->ncomp() != 1 || active_cells->local_size() != U.local_size()))
    throw std::invalid_argument(
        "Implicit source active-cell mask must have one component and match the state layout");

  auto candidate = std::make_unique<MultiFab>(U.box_array(), U.dmap(), U.ncomp(), U.n_grow());
  lincomb(*candidate, Real(1), U, Real(0), U);
  MultiFab stats(U.box_array(), U.dmap(), 5, 0);
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
  Real failure_max = Real(0), enc = Real(-1);
  for (int li = 0; li < stats.local_size(); ++li) {
    const ConstArray4 st = stats.fab(li).const_array();
    const Box2D b = stats.box(li);
    rmax = std::max(rmax, reduce_max_cell(b, detail::NewtonStatMaxKernel{st, 0}));
    reference_max = std::max(reference_max, reduce_max_cell(b, detail::NewtonStatMaxKernel{st, 1}));
    imax = std::max(imax, reduce_max_cell(b, detail::NewtonStatMaxKernel{st, 2}));
    nfail += reduce_sum_cell(b, detail::NewtonStatFailureCountKernel{st});
    failure_max = std::max(failure_max, reduce_max_cell(b, detail::NewtonStatMaxKernel{st, 3}));
    enc = std::max(enc, reduce_max_cell(b, detail::NewtonStatMaxKernel{st, 4}));
  }
  rmax = static_cast<Real>(all_reduce_max(static_cast<double>(rmax)));
  reference_max = static_cast<Real>(all_reduce_max(static_cast<double>(reference_max)));
  imax = static_cast<Real>(all_reduce_max(static_cast<double>(imax)));
  const double nfail_g = all_reduce_sum(static_cast<double>(nfail));
  const int failure_g = static_cast<int>(all_reduce_max(static_cast<double>(failure_max)));
  const double enc_g = all_reduce_max(static_cast<double>(enc));
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
  else
    solve.mark_failed(
        newton_failure_status(failure), SolveAction::kFailRun,
        std::string("implicit_source_newton_") + solve_status_name(newton_failure_status(failure)));
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
  return ImplicitSolveOutcome(U, std::move(candidate), std::move(solve), report, message.str());
}

/// Consume one prepared local solve with the fail-fast runtime policy used by native engine routes.
/// The caller still owns the outcome and names the policy at the publication boundary.
inline SolveReport consume_implicit_source_fail_run(ImplicitSolveOutcome& outcome) {
  if (outcome.report().solved_value_available())
    return outcome.consume(ImplicitSolveConsumption::kAccept);
  const std::string message = outcome.failure_message();
  const SolveReport failed = outcome.consume(ImplicitSolveConsumption::kFailRun);
  throw std::runtime_error(message + " action=" + failed.action_name());
}

/// Legacy signature with a bare iteration budget (iters = 2 historical). The numerical iteration is
/// unchanged, but its candidate now goes through the same mandatory transactional consumption.
template <class Model>
SolveReport backward_euler_source(const Model& model, const MultiFab& aux, MultiFab& U, Real dt,
                                  int iters = 2, const ImplicitMask<Model::n_vars>& mask = {}) {
  NewtonOptions opts;
  opts.max_iters = iters;
  auto outcome = backward_euler_source(model, aux, U, dt, opts, mask, nullptr);
  return consume_implicit_source_fail_run(outcome);
}

// Default implicit stepper: backward-Euler (Newton) on the model source.
// Models ImplicitBlockStepper; passed as is to the test-only reference driver as the implicit
// advance callback. The user writes no solver.
struct ImplicitSourceStepper {
  NewtonOptions options{};

  template <class Coupler, class Block>
  void operator()(Coupler& coupler, Block& block, Real dt, int /*substep*/, int /*nsub*/) const {
    auto outcome =
        backward_euler_source(block.model, coupler.aux(), block.U(), dt, options);
    (void)consume_implicit_source_fail_run(outcome);
  }
};

}  // namespace pops
