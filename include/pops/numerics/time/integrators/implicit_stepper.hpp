#pragma once

/// @file
/// @brief Fail-closed implicit-source adapter for the prepared local nonlinear provider.
///
/// This header deliberately contains no Newton algorithm. It prepares a cell-local
/// backward-Euler residual/Jacobian contract and delegates every iteration, safeguard and
/// factorization to `solve_prepared_local_nonlinear`. Candidates remain private until one
/// collective `SolveOutcome` is explicitly accepted.

#include <pops/core/foundation/types.hpp>
#include <pops/core/state/state.hpp>
#include <pops/diagnostics/runtime_diagnostics.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/nonlinear/prepared_local_nonlinear.hpp>
#include <pops/numerics/spatial_operator.hpp>
#include <pops/runtime/numerical_defaults.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

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

/// Device-safe result of a fallible source or source-Jacobian evaluation.
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

template <class Model>
concept HasFallibleSourceEvaluation = requires(const Model model, const typename Model::State state,
                                               const Aux aux, typename Model::State& output) {
  { model.evaluate_source(state, aux, output) } -> std::same_as<ImplicitEvaluationResult>;
};

template <class Model>
concept HasFallibleSourceJacobianEvaluation =
    requires(const Model model, const typename Model::State state, const Aux aux,
             Real (&jacobian)[Model::n_vars][Model::n_vars]) {
      {
        model.evaluate_source_jacobian(state, aux, jacobian)
      } -> std::same_as<ImplicitEvaluationResult>;
    };

template <int N>
struct ImplicitMask {
  bool active = false;
  bool flag[N] = {};
};

template <class Model, int N>
POPS_HD inline bool is_implicit_component(const ImplicitMask<N>& mask, int component) {
  return mask.active ? mask.flag[component] : model_is_implicit<Model>(component);
}

/// Public preparation policy. It contains no solver implementation or failure-policy escape hatch.
struct NewtonOptions {
  int max_iters = kNewtonDefaultMaxIters;
  Real rel_tol = kNewtonDefaultRelTol;
  Real abs_tol = kNewtonDefaultAbsTol;
  Real fd_eps = kNewtonDefaultFdEps;
  Real damping = kNewtonDefaultDamping;
};

inline void validate_newton_options(const NewtonOptions& options, const char* where) {
  const std::string prefix = std::string(where) + " : ";
  if (options.max_iters < 1)
    throw std::runtime_error(prefix + "newton_max_iters >= 1");
  if (!std::isfinite(options.rel_tol) || !std::isfinite(options.abs_tol) ||
      !std::isfinite(options.fd_eps) || options.rel_tol < Real(0) || options.abs_tol < Real(0) ||
      (options.rel_tol == Real(0) && options.abs_tol == Real(0)) || options.fd_eps <= Real(0))
    throw std::runtime_error(prefix +
                             "newton_rel_tol/abs_tol >= 0 with at least one positive tolerance, "
                             "and newton_fd_eps > 0");
  if (!std::isfinite(options.damping) || !(options.damping > Real(0) && options.damping <= Real(1)))
    throw std::runtime_error(prefix + "newton_damping in (0, 1]");
}

/// Compatibility inspection aggregate. The common SolveReport remains authoritative.
struct NewtonReport {
  bool enabled = false;
  bool converged = true;
  Real max_residual = Real(0);
  Real max_iters_used = Real(0);
  double n_failed = 0;
  double failed_i = -1;
  double failed_j = -1;
  double failed_comp = -1;
  SolveReport solve{};
  RuntimeDiagnosticsReport diagnostics =
      make_runtime_diagnostics_report("pops.numerics.time.prepared_local_nonlinear");

  void reset() { *this = NewtonReport{}; }
};

namespace detail {

inline constexpr std::uint32_t kImplicitUnknownEvaluationStatus = 0x4e570001u;

POPS_HD inline ImplicitEvaluationResult sanitize_implicit_evaluation(
    ImplicitEvaluationResult result) {
  switch (result.status) {
    case ImplicitEvaluationStatus::kOk:
    case ImplicitEvaluationStatus::kRetry:
    case ImplicitEvaluationStatus::kReject:
    case ImplicitEvaluationStatus::kFailed:
    case ImplicitEvaluationStatus::kInvalid:
      return result;
  }
  return ImplicitEvaluationResult::invalid(kImplicitUnknownEvaluationStatus);
}

POPS_HD inline LocalNonlinearEvaluationResult local_evaluation(ImplicitEvaluationResult result) {
  result = sanitize_implicit_evaluation(result);
  switch (result.status) {
    case ImplicitEvaluationStatus::kOk:
      return LocalNonlinearEvaluationResult::ok();
    case ImplicitEvaluationStatus::kRetry:
      return LocalNonlinearEvaluationResult::retry(result.reason_code);
    case ImplicitEvaluationStatus::kReject:
      return LocalNonlinearEvaluationResult::reject(result.reason_code);
    case ImplicitEvaluationStatus::kFailed:
      return LocalNonlinearEvaluationResult::failed(result.reason_code);
    case ImplicitEvaluationStatus::kInvalid:
      return LocalNonlinearEvaluationResult::invalid(result.reason_code);
  }
  return LocalNonlinearEvaluationResult::invalid(kImplicitUnknownEvaluationStatus);
}

POPS_HD inline LocalNonlinearStatus local_status(ImplicitEvaluationResult result) {
  const LocalNonlinearEvaluationResult evaluation = local_evaluation(result);
  return local_evaluation_failure(evaluation.status);
}

template <class Model>
POPS_HD inline ImplicitEvaluationResult evaluate_implicit_source(const Model& model,
                                                                 const typename Model::State& state,
                                                                 const Aux& aux,
                                                                 typename Model::State& output) {
  if constexpr (HasFallibleSourceEvaluation<Model>)
    return sanitize_implicit_evaluation(model.evaluate_source(state, aux, output));
  output = model.source(state, aux);
  return ImplicitEvaluationResult::ok();
}

template <class Model>
struct ImplicitSourceResidual {
  static constexpr int N = Model::n_vars;
  Model model;
  typename Model::State initial;
  typename Model::State explicit_target;
  Aux aux;
  Real dt = Real(0);
  ImplicitMask<N> mask{};

  POPS_HD LocalNonlinearEvaluationResult operator()(const Real (&candidate)[N],
                                                    Real (&residual)[N]) const {
    typename Model::State state{};
    for (int component = 0; component < N; ++component)
      state[component] = candidate[component];
    typename Model::State source{};
    const ImplicitEvaluationResult evaluation = evaluate_implicit_source(model, state, aux, source);
    if (!evaluation.succeeded())
      return local_evaluation(evaluation);
    for (int component = 0; component < N; ++component) {
      residual[component] = is_implicit_component<Model>(mask, component)
                                ? candidate[component] - initial[component] - dt * source[component]
                                : candidate[component] - explicit_target[component];
    }
    return LocalNonlinearEvaluationResult::ok();
  }
};

template <class Model>
struct ImplicitSourceAnalyticJacobian {
  static constexpr int N = Model::n_vars;
  Model model;
  Aux aux;
  Real dt = Real(0);
  ImplicitMask<N> mask{};

  POPS_HD LocalNonlinearEvaluationResult operator()(const Real (&candidate)[N],
                                                    Real (&jacobian)[N][N]) const {
    typename Model::State state{};
    for (int component = 0; component < N; ++component)
      state[component] = candidate[component];
    Real source_jacobian[N][N];
    if constexpr (HasFallibleSourceJacobianEvaluation<Model>) {
      const ImplicitEvaluationResult evaluation =
          sanitize_implicit_evaluation(model.evaluate_source_jacobian(state, aux, source_jacobian));
      if (!evaluation.succeeded())
        return local_evaluation(evaluation);
    } else {
      model.source_jacobian(state, aux, source_jacobian);
    }
    for (int row = 0; row < N; ++row)
      for (int column = 0; column < N; ++column) {
        const Real identity = row == column ? Real(1) : Real(0);
        jacobian[row][column] = is_implicit_component<Model>(mask, row)
                                    ? identity - dt * source_jacobian[row][column]
                                    : identity;
      }
    return LocalNonlinearEvaluationResult::ok();
  }
};

POPS_HD inline PreparedLocalNonlinearControls prepared_controls(const NewtonOptions& options) {
  PreparedLocalNonlinearControls controls;
  controls.max_iterations = options.max_iters;
  controls.absolute_tolerance = options.abs_tol;
  controls.relative_tolerance = options.rel_tol;
  controls.finite_difference_step = options.fd_eps;
  controls.initial_step = options.damping;
  controls.safeguard = options.damping < Real(1) ? LocalSafeguardKind::kFixedDamping
                                                 : LocalSafeguardKind::kExactNewton;
  return controls;
}

template <class Model>
POPS_HD inline auto prepare_implicit_source_problem(const Model& model,
                                                    const typename Model::State& initial,
                                                    const typename Model::State& initial_source,
                                                    const Aux& aux, Real dt,
                                                    const NewtonOptions& options,
                                                    const ImplicitMask<Model::n_vars>& mask) {
  constexpr int N = Model::n_vars;
  typename Model::State explicit_target = initial;
  for (int component = 0; component < N; ++component)
    if (!is_implicit_component<Model>(mask, component))
      explicit_target[component] = initial[component] + dt * initial_source[component];

  const ImplicitSourceResidual<Model> residual{model, initial, explicit_target, aux, dt, mask};
  const PreparedLocalNonlinearControls controls = prepared_controls(options);
  if constexpr (HasSourceJacobian<Model> || HasFallibleSourceJacobianEvaluation<Model>) {
    const ImplicitSourceAnalyticJacobian<Model> jacobian{model, aux, dt, mask};
    return prepare_local_nonlinear_problem<N>(
        residual, AnalyticLocalJacobian<N, ImplicitSourceAnalyticJacobian<Model>>{jacobian},
        AcceptAllLocalCandidates<N>{}, controls);
  } else {
    return prepare_local_nonlinear_problem<N>(residual, FiniteDifferenceLocalJacobian<N>{},
                                              AcceptAllLocalCandidates<N>{}, controls);
  }
}

template <class Model>
struct PreparedImplicitSourceKernel {
  static constexpr int N = Model::n_vars;
  Model model;
  ConstArray4 state;
  ConstArray4 aux;
  ConstArray4 active_cells;
  bool has_active_cells = false;
  Array4 candidate;
  Array4 statistics;
  Real dt = Real(0);
  NewtonOptions options{};
  ImplicitMask<N> mask{};

  POPS_HD void operator()(int i, int j) const {
    const typename Model::State initial = load_state<Model>(state, i, j);
    if (has_active_cells && active_cells(i, j, 0) < Real(0.5)) {
      for (int component = 0; component < N; ++component)
        candidate(i, j, component) = initial[component];
      for (int component = 0; component < 13; ++component)
        statistics(i, j, component) = Real(0);
      return;
    }

    const Aux cell_aux = load_aux<aux_comps<Model>()>(aux, i, j);
    typename Model::State initial_source{};
    bool requires_initial_source = false;
    for (int component = 0; component < N; ++component)
      requires_initial_source =
          requires_initial_source || !is_implicit_component<Model>(mask, component);
    const ImplicitEvaluationResult initial_evaluation =
        requires_initial_source ? evaluate_implicit_source(model, initial, cell_aux, initial_source)
                                : ImplicitEvaluationResult::ok();
    LocalNonlinearCellResult<N> solved;
    for (int component = 0; component < N; ++component)
      solved.value[component] = initial[component];
    solved.evaluations = requires_initial_source ? 1 : 0;
    if (!initial_evaluation.succeeded()) {
      solved.status = local_status(initial_evaluation);
      solved.reason_code = initial_evaluation.reason_code;
    } else {
      const auto problem = prepare_implicit_source_problem(model, initial, initial_source, cell_aux,
                                                           dt, options, mask);
      Real guess[N];
      for (int component = 0; component < N; ++component)
        guess[component] = initial[component];
      solved = solve_prepared_local_nonlinear(problem, guess);
      if (requires_initial_source)
        ++solved.evaluations;
    }
    for (int component = 0; component < N; ++component)
      candidate(i, j, component) = solved.value[component];

    statistics(i, j, 0) = static_cast<Real>(local_nonlinear_status_code(solved.status));
    statistics(i, j, 1) = static_cast<Real>(solved.iterations);
    statistics(i, j, 2) = static_cast<Real>(solved.evaluations);
    statistics(i, j, 3) = solved.reference_residual_norm;
    statistics(i, j, 4) = solved.residual_norm;
    statistics(i, j, 5) = solved.step_norm;
    statistics(i, j, 6) = solved.condition_evidence;
    statistics(i, j, 7) = static_cast<Real>(solved.safeguard_steps);
    if (!solved.solved()) {
      statistics(i, j, 8) = encode_local_nonlinear_failure(i, j, solved.failing_component);
      statistics(i, j, 9) = Real(1);
      statistics(i, j, 10) = static_cast<Real>((solved.reason_code >> 16) & 0xffffu);
      statistics(i, j, 11) = static_cast<Real>(solved.reason_code & 0xffffu);
    } else {
      for (int component = 8; component < 12; ++component)
        statistics(i, j, component) = Real(0);
    }
    statistics(i, j, 12) = static_cast<Real>(local_nonlinear_status_priority(solved.status));
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

struct LocalStatSum {
  ConstArray4 values;
  int component = 0;
  POPS_HD void operator()(int i, int j, Real& result) const { result += values(i, j, component); }
};

struct LocalStatMaxForStatus {
  ConstArray4 values;
  int status = 0;
  int component = 0;
  POPS_HD void operator()(int i, int j, Real& result) const {
    if (static_cast<int>(values(i, j, 0)) == status)
      if (const Real value = values(i, j, component); value > result)
        result = value;
  }
};

struct LocalStatReasonHighForLocation {
  ConstArray4 values;
  int status = 0;
  Real location = Real(0);
  POPS_HD void operator()(int i, int j, Real& result) const {
    if (static_cast<int>(values(i, j, 0)) == status && values(i, j, 8) == location)
      if (const Real value = values(i, j, 10); value > result)
        result = value;
  }
};

struct LocalStatReasonLowForLocation {
  ConstArray4 values;
  int status = 0;
  Real location = Real(0);
  int reason_high = 0;
  POPS_HD void operator()(int i, int j, Real& result) const {
    if (static_cast<int>(values(i, j, 0)) == status && values(i, j, 8) == location &&
        static_cast<int>(values(i, j, 10)) == reason_high)
      if (const Real value = values(i, j, 11); value > result)
        result = value;
  }
};

inline Real collective_max_component(const MultiFab& statistics, int component) {
  Real local = Real(0);
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

inline Real collective_max_for_status(const MultiFab& statistics, int status, int component) {
  Real local = Real(0);
  for (int local_index = 0; local_index < statistics.local_size(); ++local_index) {
    const ConstArray4 values = statistics.fab(local_index).const_array();
    local = std::max(local, reduce_max_cell(statistics.box(local_index),
                                            LocalStatMaxForStatus{values, status, component}));
  }
  return static_cast<Real>(all_reduce_max(static_cast<double>(local)));
}

inline Real collective_reason_high(const MultiFab& statistics, int status, Real location) {
  Real local = Real(0);
  for (int local_index = 0; local_index < statistics.local_size(); ++local_index) {
    const ConstArray4 values = statistics.fab(local_index).const_array();
    local =
        std::max(local, reduce_max_cell(statistics.box(local_index),
                                        LocalStatReasonHighForLocation{values, status, location}));
  }
  return static_cast<Real>(all_reduce_max(static_cast<double>(local)));
}

inline Real collective_reason_low(const MultiFab& statistics, int status, Real location,
                                  int reason_high) {
  Real local = Real(0);
  for (int local_index = 0; local_index < statistics.local_size(); ++local_index) {
    const ConstArray4 values = statistics.fab(local_index).const_array();
    local = std::max(local, reduce_max_cell(statistics.box(local_index),
                                            LocalStatReasonLowForLocation{values, status, location,
                                                                          reason_high}));
  }
  return static_cast<Real>(all_reduce_max(static_cast<double>(local)));
}

inline SolveAction implicit_failure_action(LocalNonlinearStatus status) {
  switch (status) {
    case LocalNonlinearStatus::kIterationLimit:
    case LocalNonlinearStatus::kInadmissibleCandidate:
    case LocalNonlinearStatus::kSafeguardFailure:
    case LocalNonlinearStatus::kEvaluationRetry:
    case LocalNonlinearStatus::kEvaluationReject:
      return SolveAction::kRejectAttempt;
    case LocalNonlinearStatus::kConverged:
      return SolveAction::kNone;
    case LocalNonlinearStatus::kSingularJacobian:
    case LocalNonlinearStatus::kInvalidEvaluation:
    case LocalNonlinearStatus::kUnsupportedCapability:
    case LocalNonlinearStatus::kEvaluationFailed:
      return SolveAction::kFailRun;
  }
  return SolveAction::kFailRun;
}

inline NewtonReport staged_legacy_report(const NewtonReport* current, const SolveReport& solve,
                                         double failed_cells) {
  NewtonReport staged = current != nullptr ? *current : NewtonReport{};
  staged.enabled = true;
  staged.solve = solve;
  staged.max_residual = std::max(staged.max_residual, solve.residual_norm);
  staged.max_iters_used = std::max(staged.max_iters_used, static_cast<Real>(solve.iters));
  staged.n_failed += failed_cells;
  if (!solve.solved()) {
    staged.converged = false;
    staged.failed_i = solve.failed_i;
    staged.failed_j = solve.failed_j;
    staged.failed_comp = solve.failed_component;
  }
  return staged;
}

struct ImplicitSourcePublication {
  MultiFab* destination = nullptr;
  std::unique_ptr<MultiFab> candidate;
  NewtonReport* diagnostics = nullptr;
  NewtonReport staged_diagnostics{};

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
    if (publication.diagnostics != nullptr)
      *publication.diagnostics = publication.staged_diagnostics;
  }
};

}  // namespace detail

/// Prepare a local backward-Euler source solve without publishing its candidate.
template <class Model>
[[nodiscard]] SolveOutcome backward_euler_source(const Model& model, const MultiFab& aux,
                                                 MultiFab& state, Real dt,
                                                 const NewtonOptions& options,
                                                 const ImplicitMask<Model::n_vars>& mask = {},
                                                 NewtonReport* diagnostics = nullptr,
                                                 const MultiFab* active_cells = nullptr) {
  validate_newton_options(options, "backward_euler_source");
  if (active_cells != nullptr &&
      (active_cells->ncomp() != 1 || active_cells->local_size() != state.local_size()))
    throw std::invalid_argument(
        "Implicit source active-cell mask must have one component and match the state layout");

  auto candidate =
      std::make_unique<MultiFab>(state.box_array(), state.dmap(), state.ncomp(), state.n_grow());
  lincomb(*candidate, Real(1), state, Real(0), state);
  MultiFab statistics(state.box_array(), state.dmap(), 13, 0);
  for (int local_index = 0; local_index < state.local_size(); ++local_index) {
    const ConstArray4 active =
        active_cells != nullptr ? active_cells->fab(local_index).const_array() : ConstArray4{};
    for_each_cell(state.box(local_index), detail::PreparedImplicitSourceKernel<Model>{
                                              model,
                                              state.fab(local_index).const_array(),
                                              aux.fab(local_index).const_array(),
                                              active,
                                              active_cells != nullptr,
                                              candidate->fab(local_index).array(),
                                              statistics.fab(local_index).array(),
                                              dt,
                                              options,
                                              mask,
                                          });
  }

  const int status_priority = static_cast<int>(detail::collective_max_component(statistics, 12));
  const LocalNonlinearStatus status = local_nonlinear_status_from_priority(status_priority);
  const int status_code = local_nonlinear_status_code(status);
  const int iterations = static_cast<int>(detail::collective_max_component(statistics, 1));
  const int evaluations = static_cast<int>(detail::collective_max_component(statistics, 2));
  const Real reference_residual = detail::collective_max_component(statistics, 3);
  const Real residual = detail::collective_max_component(statistics, 4);
  const Real step = detail::collective_max_component(statistics, 5);
  const Real condition = detail::collective_max_component(statistics, 6);
  const int safeguard_steps = static_cast<int>(detail::collective_max_component(statistics, 7));
  const double failed_cells = detail::collective_sum_component(statistics, 9);

  int failed_i = -1;
  int failed_j = -1;
  int failed_component = -1;
  std::uint32_t reason_code = 0;
  if (failed_cells > 0) {
    const Real encoded = detail::collective_max_for_status(statistics, status_code, 8);
    detail::decode_local_nonlinear_failure(encoded, failed_i, failed_j, failed_component);
    const int reason_high =
        static_cast<int>(detail::collective_reason_high(statistics, status_code, encoded));
    const int reason_low = static_cast<int>(
        detail::collective_reason_low(statistics, status_code, encoded, reason_high));
    reason_code =
        (static_cast<std::uint32_t>(reason_high) << 16) | static_cast<std::uint32_t>(reason_low);
  }

  SolveReport solve =
      local_nonlinear_solve_report(status_code, iterations, evaluations, reference_residual,
                                   residual, step, condition, safeguard_steps, failed_i, failed_j,
                                   failed_component, detail::implicit_failure_action(status));
  if (!solve.solved()) {
    solve.reason = std::string("implicit_source_") + local_nonlinear_status_name(status);
    if (reason_code != 0)
      solve.reason += "_reason_" + std::to_string(reason_code);
  } else {
    solve.reason = "implicit_source_converged";
  }
  if (!solve_report_is_publishable(solve, options.max_iters)) {
    std::ostringstream message;
    message << "implicit source produced a malformed SolveReport: status=" << solve.status_name()
            << " action=" << solve.action_name() << " iterations=" << solve.iters
            << " evaluations=" << solve.evaluations
            << " reference_residual=" << solve.reference_residual_norm
            << " residual=" << solve.residual_norm << " relative_residual=" << solve.rel_residual
            << " step=" << solve.step_norm << " condition=" << solve.condition_evidence;
    throw std::runtime_error(message.str());
  }

  const NewtonReport staged = detail::staged_legacy_report(diagnostics, solve, failed_cells);
  auto publication = std::make_shared<detail::ImplicitSourcePublication>(
      detail::ImplicitSourcePublication{&state, std::move(candidate), diagnostics, staged});
  return SolveOutcome::collective_world(std::move(solve),
                                        SolveOutcome::PublicationHooks{
                                            publication.get(),
                                            &detail::ImplicitSourcePublication::accept,
                                            nullptr,
                                            nullptr,
                                            std::static_pointer_cast<void>(publication),
                                            &detail::ImplicitSourcePublication::validate_accept,
                                            nullptr,
                                        });
}

inline SolveReport consume_implicit_source_fail_run(SolveOutcome& outcome) {
  if (outcome.report().solved_value_available())
    return outcome.consume(SolveConsumption::kAccept);
  const std::string status = outcome.report().status_name();
  const std::string reason = outcome.report().reason;
  const SolveReport failed = outcome.consume(SolveConsumption::kFailRun);
  throw std::runtime_error("Implicit source nonlinear solve failed: status=" + status +
                           " reason=" + reason + " action=" + failed.action_name());
}

struct ImplicitSourceStepper {
  NewtonOptions options{};

  template <class Coupler, class Block>
  void operator()(Coupler& coupler, Block& block, Real dt, int /*substep*/,
                  int /*substep_count*/) const {
    auto outcome = backward_euler_source(block.model, coupler.aux(), block.U(), dt, options);
    (void)consume_implicit_source_fail_run(outcome);
  }
};

}  // namespace pops
