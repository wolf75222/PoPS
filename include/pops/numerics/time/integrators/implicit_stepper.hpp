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
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/nonlinear/local_nonlinear_collective.hpp>
#include <pops/numerics/nonlinear/newton_options.hpp>
#include <pops/numerics/nonlinear/prepared_local_nonlinear.hpp>
#include <pops/numerics/spatial/primitives/state_access.hpp>
#include <pops/numerics/time/integrators/prepared_implicit_source_workspace.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace pops {

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

template <class Model, class Auxiliary>
concept HasSourceJacobianFor =
    requires(const Model model, const typename Model::State state, const Auxiliary aux,
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

template <class Model, class Auxiliary>
concept HasFallibleSourceEvaluationFor =
    requires(const Model model, const typename Model::State state, const Auxiliary aux,
             typename Model::State& output) {
      { model.evaluate_source(state, aux, output) } -> std::same_as<ImplicitEvaluationResult>;
    };

template <class Model, class Auxiliary>
concept HasFallibleSourceJacobianEvaluationFor =
    requires(const Model model, const typename Model::State state, const Auxiliary aux,
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

template <class Model, class Auxiliary>
POPS_HD inline ImplicitEvaluationResult evaluate_implicit_source(const Model& model,
                                                                 const typename Model::State& state,
                                                                 const Auxiliary& aux,
                                                                 typename Model::State& output) {
  if constexpr (HasFallibleSourceEvaluationFor<Model, Auxiliary>)
    return sanitize_implicit_evaluation(model.evaluate_source(state, aux, output));
  output = model.source(state, aux);
  return ImplicitEvaluationResult::ok();
}

template <class Model, class Auxiliary>
struct ImplicitSourceResidual {
  static constexpr int N = Model::n_vars;
  Model model;
  typename Model::State initial;
  typename Model::State explicit_target;
  Auxiliary aux;
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

template <class Model, class Auxiliary>
struct ImplicitSourceAnalyticJacobian {
  static constexpr int N = Model::n_vars;
  Model model;
  Auxiliary aux;
  Real dt = Real(0);
  ImplicitMask<N> mask{};

  POPS_HD LocalNonlinearEvaluationResult operator()(const Real (&candidate)[N],
                                                    Real (&jacobian)[N][N]) const {
    typename Model::State state{};
    for (int component = 0; component < N; ++component)
      state[component] = candidate[component];
    Real source_jacobian[N][N];
    if constexpr (HasFallibleSourceJacobianEvaluationFor<Model, Auxiliary>) {
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

template <class Model, class Auxiliary>
POPS_HD inline auto prepare_implicit_source_problem(const Model& model,
                                                    const typename Model::State& initial,
                                                    const typename Model::State& initial_source,
                                                    const Auxiliary& aux, Real dt,
                                                    const NewtonOptions& options,
                                                    const ImplicitMask<Model::n_vars>& mask) {
  constexpr int N = Model::n_vars;
  typename Model::State explicit_target = initial;
  for (int component = 0; component < N; ++component)
    if (!is_implicit_component<Model>(mask, component))
      explicit_target[component] = initial[component] + dt * initial_source[component];

  const ImplicitSourceResidual<Model, Auxiliary> residual{model, initial, explicit_target,
                                                          aux,   dt,      mask};
  const PreparedLocalNonlinearControls controls = prepared_controls(options);
  if constexpr (HasSourceJacobianFor<Model, Auxiliary> ||
                HasFallibleSourceJacobianEvaluationFor<Model, Auxiliary>) {
    const ImplicitSourceAnalyticJacobian<Model, Auxiliary> jacobian{model, aux, dt, mask};
    return prepare_local_nonlinear_problem<N>(
        residual,
        AnalyticLocalJacobian<N, ImplicitSourceAnalyticJacobian<Model, Auxiliary>>{jacobian},
        AcceptAllLocalCandidates<N>{}, controls);
  } else {
    return prepare_local_nonlinear_problem<N>(residual, FiniteDifferenceLocalJacobian<N>{},
                                              AcceptAllLocalCandidates<N>{}, controls);
  }
}

template <int Dim, class Model>
struct PreparedImplicitSourceKernel {
  static constexpr int N = Model::n_vars;
  static constexpr int provider_count = provider_count_for<Model, Dim>();
  Model model;
  FieldView<const Real, Dim> state{};
  ProviderStorageView<Dim, provider_count> providers{};
  FieldView<const Real, Dim> active_cells{};
  bool has_active_cells = false;
  FieldView<Real, Dim> candidate{};
  FieldView<Real, Dim> statistics{};
  Real dt = Real(0);
  NewtonOptions options{};
  ImplicitMask<N> mask{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    const typename Model::State initial = load_state<Model>(state, index);
    if (has_active_cells && active_cells(index) < Real(0.5)) {
      for (int component = 0; component < N; ++component)
        candidate(index, component) = initial[component];
      for (int component = 0; component < 13; ++component)
        statistics(index, component) = Real(0);
      return;
    }

    ProviderValues<provider_count> cell_providers{};
    if constexpr (provider_count > 0)
      cell_providers = load_provider_values<provider_count>(providers, index);
    typename Model::State initial_source{};
    bool requires_initial_source = false;
    for (int component = 0; component < N; ++component)
      requires_initial_source =
          requires_initial_source || !is_implicit_component<Model>(mask, component);
    const ImplicitEvaluationResult initial_evaluation =
        requires_initial_source
            ? evaluate_implicit_source(model, initial, cell_providers, initial_source)
            : ImplicitEvaluationResult::ok();
    LocalNonlinearCellResult<N> solved;
    for (int component = 0; component < N; ++component)
      solved.value[component] = initial[component];
    solved.evaluations = requires_initial_source ? 1 : 0;
    if (!initial_evaluation.succeeded()) {
      solved.status = local_status(initial_evaluation);
      solved.reason_code = initial_evaluation.reason_code;
    } else {
      const auto problem = prepare_implicit_source_problem(model, initial, initial_source,
                                                           cell_providers, dt, options, mask);
      Real guess[N];
      for (int component = 0; component < N; ++component)
        guess[component] = initial[component];
      solved = solve_prepared_local_nonlinear(problem, guess);
      if (requires_initial_source)
        ++solved.evaluations;
    }
    for (int component = 0; component < N; ++component)
      candidate(index, component) = solved.value[component];

    statistics(index, 0) = static_cast<Real>(local_nonlinear_status_code(solved.status));
    statistics(index, 1) = static_cast<Real>(solved.iterations);
    statistics(index, 2) = static_cast<Real>(solved.evaluations);
    statistics(index, 3) = solved.reference_residual_norm;
    statistics(index, 4) = solved.residual_norm;
    statistics(index, 5) = solved.step_norm;
    statistics(index, 6) = solved.condition_evidence;
    statistics(index, 7) = static_cast<Real>(solved.safeguard_steps);
    if (!solved.solved()) {
      statistics(index, 8) = static_cast<Real>(solved.failing_component);
      statistics(index, 9) = Real(1);
      statistics(index, 10) = static_cast<Real>((solved.reason_code >> 16) & 0xffffu);
      statistics(index, 11) = static_cast<Real>(solved.reason_code & 0xffffu);
    } else {
      for (int component = 8; component < 12; ++component)
        statistics(index, component) = Real(0);
    }
    statistics(index, 12) = static_cast<Real>(local_nonlinear_status_priority(solved.status));
  }
};

template <int Dim>
struct LocalStatValue {
  FieldView<const Real, Dim> values{};
  int component = 0;
  POPS_HD Real operator()(const Index<Dim>& index) const { return values(index, component); }
};

template <int Dim, class MemorySpace, class Functor>
inline void prepared_implicit_for_each(const Box<Dim>& box, const Functor& functor) {
  if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>) {
    const Extent<Dim> extent = box.extent();
    for (std::int64_t ordinal = 0; ordinal < box.numPts(); ++ordinal) {
      std::int64_t remainder = ordinal;
      Index<Dim> index{};
      for (int axis = 0; axis < Dim; ++axis) {
        index[axis] = box.lo[axis] + static_cast<int>(remainder % extent[axis]);
        remainder /= extent[axis];
      }
      functor(index);
    }
  } else {
    for_each_cell(box, functor);
  }
}

template <int Dim>
struct LocalStatReasonForLocation {
  FieldView<const Real, Dim> values{};
  int status = 0;
  Index<Dim> selected{};
  int component = 10;
  int required_high = -1;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    for (int axis = 0; axis < Dim; ++axis)
      if (index[axis] != selected[axis])
        return Real(0);
    if (static_cast<int>(values(index, 0)) != status ||
        (required_high >= 0 && static_cast<int>(values(index, 10)) != required_high))
      return Real(0);
    return values(index, component);
  }
};

template <int Dim, class MemorySpace, class ExecutionSpace>
Real collective_max_component(const MultiFab<Dim, MemorySpace>& statistics, int component,
                              const ExecutionLane& lane, const ExecutionSpace& execution,
                              const PreparedCellSumReduction<ExecutionSpace>& reduction) {
  Real local = Real(0);
  for (std::size_t local_index = 0; local_index < statistics.local_size(); ++local_index)
    local =
        std::max(local, for_each_cell_reduce_max(
                            execution, reduction, statistics.box(local_index),
                            LocalStatValue<Dim>{statistics.fab(local_index).view(), component}));
  return static_cast<Real>(all_reduce_max(static_cast<double>(local), lane));
}

template <int Dim, class MemorySpace, class ExecutionSpace>
double collective_sum_component(const MultiFab<Dim, MemorySpace>& statistics, int component,
                                const ExecutionLane& lane, const ExecutionSpace& execution,
                                const PreparedCellSumReduction<ExecutionSpace>& reduction) {
  Real local = Real(0);
  for (std::size_t local_index = 0; local_index < statistics.local_size(); ++local_index)
    local += for_each_cell_reduce_sum(
        execution, reduction, statistics.box(local_index),
        LocalStatValue<Dim>{statistics.fab(local_index).view(), component});
  return all_reduce_sum(static_cast<double>(local), lane);
}

template <int Dim, class MemorySpace, class ExecutionSpace>
Real collective_reason(const MultiFab<Dim, MemorySpace>& statistics, int status,
                       const Index<Dim>& selected, int component, const ExecutionLane& lane,
                       const ExecutionSpace& execution,
                       const PreparedCellSumReduction<ExecutionSpace>& reduction,
                       int required_high = -1) {
  Real local = Real(0);
  for (std::size_t local_index = 0; local_index < statistics.local_size(); ++local_index)
    local = std::max(local,
                     for_each_cell_reduce_max(
                         execution, reduction, statistics.box(local_index),
                         LocalStatReasonForLocation<Dim>{statistics.fab(local_index).view(), status,
                                                         selected, component, required_high}));
  return static_cast<Real>(all_reduce_max(static_cast<double>(local), lane));
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

}  // namespace detail

/// Host-side binder for the exact provider slots of one local patch.
///
/// The prepared consumer plan resolves the producer-qualified storage addresses before a
/// numerical kernel is launched.  This callback transports the resulting device-copyable view
/// for the requested local patch; it deliberately has no representation for a process-global
/// auxiliary component prefix.  Provider-free models never invoke the binder.
template <class ProviderAt, int Dim, int Count>
concept ImplicitProviderPatchBinding =
    Count == 0 || requires(const ProviderAt& provider_at, std::size_t local_patch) {
      { provider_at(local_patch) } -> std::same_as<ProviderStorageView<Dim, Count>>;
    };

namespace detail {

inline bool prepared_implicit_newton_options_valid(const NewtonOptions& options) noexcept {
  return options.max_iters >= 1 && std::isfinite(options.rel_tol) &&
         std::isfinite(options.abs_tol) && std::isfinite(options.fd_eps) &&
         options.rel_tol >= Real(0) && options.abs_tol >= Real(0) &&
         (options.rel_tol > Real(0) || options.abs_tol > Real(0)) && options.fd_eps > Real(0) &&
         std::isfinite(options.damping) && options.damping > Real(0) && options.damping <= Real(1);
}

inline SolveReport prepared_implicit_failure(SolveStatus status, SolveAction action,
                                             const char* reason) {
  SolveReport report;
  report.mark_failed(status, action, reason);
  return report;
}

inline SolveReport prepared_implicit_report(int status_code, int iterations, int evaluations,
                                            Real reference_residual, Real residual, Real step,
                                            Real condition, int safeguard_steps,
                                            SolveFailureLocation failure, SolveAction action) {
  const LocalNonlinearStatus status = static_cast<LocalNonlinearStatus>(status_code);
  SolveReport report;
  report.iters = iterations;
  report.evaluations = evaluations;
  report.reference_residual_norm = reference_residual;
  report.residual_norm = residual;
  const Real largest_finite = std::numeric_limits<Real>::max();
  report.rel_residual =
      reference_residual > Real(0)
          ? (reference_residual >= residual / largest_finite ? residual / reference_residual
                                                             : largest_finite)
          : residual;
  report.step_norm = step;
  report.condition_evidence = condition;
  report.safeguard_steps = safeguard_steps;
  report.failure = failure;
  if (status == LocalNonlinearStatus::kConverged)
    report.mark_solved("ok");
  else
    report.mark_failed(solve_status(status), action, "failed");
  return report;
}

template <int Dim, class Model, class MemorySpace, class ProviderAt, class PrepareState>
SolveOutcome backward_euler_source_prepared(
    const Model& model, const ProviderAt& provider_at, MultiFab<Dim, MemorySpace>& state,
    PreparedImplicitSourceWorkspace<Dim, MemorySpace>& workspace, std::uint64_t state_generation,
    Real dt, const NewtonOptions& options, const ExecutionLane& lane,
    const ImplicitMask<Model::n_vars>& mask, const MultiFab<Dim, MemorySpace>* active_cells,
    std::shared_ptr<void> lifetime, PrepareState&& prepare_state) {
  using Workspace = PreparedImplicitSourceWorkspace<Dim, MemorySpace>;
  using Access = PreparedImplicitSourceWorkspaceAccess<Dim, MemorySpace>;
  constexpr int provider_count = provider_count_for<Model, Dim>();
  const auto fail = [&](SolveStatus status, SolveAction action, const char* reason) {
    return SolveOutcome::collective_lane(prepared_implicit_failure(status, action, reason), lane);
  };
  const auto layout_matches = [&](const MultiFab<Dim, MemorySpace>& other) {
    return state.layout() == other.layout() && state.distribution() == other.distribution() &&
           state.local_rank() == other.local_rank() && state.local_size() == other.local_size();
  };

  if (!lane.active())
    return SolveOutcome::serial(
        prepared_implicit_failure(SolveStatus::kInvalidInput, SolveAction::kFailRun, "lane"));

  int local_preflight_code = 0;
  if (!prepared_implicit_newton_options_valid(options))
    local_preflight_code = 1;
  else if (state.ncomp() != Model::n_vars)
    local_preflight_code = 2;
  else if (active_cells != nullptr &&
           (active_cells->ncomp() != 1 || !layout_matches(*active_cells)))
    local_preflight_code = 3;
  else if (state.rank_space().size() != static_cast<std::size_t>(lane.size()) ||
           state.rank_space().linear_rank(state.local_rank()) !=
               static_cast<std::size_t>(lane.rank()))
    local_preflight_code = 4;
  else if (!Access::matches(workspace, state, state_generation))
    local_preflight_code = 5;
  else if (!Access::diagnostics_capacity_valid(workspace))
    local_preflight_code = 6;
  const long preflight_min = all_reduce_min(static_cast<long>(local_preflight_code), lane);
  const long preflight_max = all_reduce_max(static_cast<long>(local_preflight_code), lane);
  if (preflight_min != 0 || preflight_max != 0) {
    const char* reason = "preflight";
    if (preflight_min == preflight_max) {
      switch (preflight_min) {
        case 1:
          reason = "options";
          break;
        case 2:
          reason = "state";
          break;
        case 3:
          reason = "mask";
          break;
        case 4:
          reason = "rank";
          break;
        case 5:
          reason = "stale";
          break;
        case 6:
          reason = "diagnostic";
          break;
        default:
          break;
      }
    }
    return fail(SolveStatus::kInvalidInput, SolveAction::kFailRun, reason);
  }

  const bool locally_reserved = Access::try_reserve(workspace);
  if (all_reduce_min(locally_reserved ? 1L : 0L, lane) == 0) {
    if (locally_reserved)
      Access::release(workspace);
    return fail(SolveStatus::kCapabilityFailure, SolveAction::kFailRun, "busy");
  }

  struct PendingRelease final {
    Workspace* workspace = nullptr;
    bool armed = true;
    ~PendingRelease() {
      if (armed)
        Access::release(*workspace);
    }
  } pending{&workspace};

  auto& candidate = Access::candidate(workspace);
  auto& statistics = Access::statistics(workspace);
  const auto& execution = Access::execution(workspace);
  const auto& reduction = Access::reduction(workspace);
  long local_candidate_fault = 0;
  try {
    std::forward<PrepareState>(prepare_state)();
    Access::copy_from_state(workspace, state);
    for (std::size_t local_index = 0; local_index < state.local_size(); ++local_index) {
      FieldView<const Real, Dim> active{};
      ProviderStorageView<Dim, provider_count> provider_view{};
      if (active_cells != nullptr)
        active = active_cells->fab(local_index).view();
      if constexpr (provider_count > 0)
        provider_view = provider_at(local_index);
      detail::prepared_implicit_for_each<Dim, MemorySpace>(
          state.box(local_index), detail::PreparedImplicitSourceKernel<Dim, Model>{
                                      model,
                                      std::as_const(state).fab(local_index).view(),
                                      provider_view,
                                      active,
                                      active_cells != nullptr,
                                      candidate.fab(local_index).view(),
                                      statistics.fab(local_index).view(),
                                      dt,
                                      options,
                                      mask,
                                  });
    }
  } catch (...) {
    local_candidate_fault = 1;
  }
  if (all_reduce_max(local_candidate_fault, lane) != 0)
    return fail(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun, "kernel");

  const int status_priority =
      static_cast<int>(collective_max_component(statistics, 12, lane, execution, reduction));
  const LocalNonlinearStatus status = local_nonlinear_status_from_priority(status_priority);
  const int status_code = local_nonlinear_status_code(status);
  const int iterations =
      static_cast<int>(collective_max_component(statistics, 1, lane, execution, reduction));
  const int evaluations =
      static_cast<int>(collective_max_component(statistics, 2, lane, execution, reduction));
  const Real reference_residual =
      collective_max_component(statistics, 3, lane, execution, reduction);
  const Real residual = collective_max_component(statistics, 4, lane, execution, reduction);
  const Real step = collective_max_component(statistics, 5, lane, execution, reduction);
  const Real condition = collective_max_component(statistics, 6, lane, execution, reduction);
  const int safeguard_steps =
      static_cast<int>(collective_max_component(statistics, 7, lane, execution, reduction));
  const double failed_cells = collective_sum_component(statistics, 9, lane, execution, reduction);

  LocalNonlinearFailureLocation<Dim> failure{};
  std::uint32_t reason_code = 0;
  if (failed_cells > 0) {
    failure = collective_first_local_nonlinear_failure(
        statistics, Access::failure_buffer(workspace), status_priority, 12, 8, lane);
    if (!failure.found || failure.priority != status_priority)
      return fail(SolveStatus::kInvalidInput, SolveAction::kFailRun, "rank");
    const int reason_high = static_cast<int>(
        collective_reason(statistics, status_code, failure.index, 10, lane, execution, reduction));
    const int reason_low = static_cast<int>(collective_reason(
        statistics, status_code, failure.index, 11, lane, execution, reduction, reason_high));
    reason_code =
        (static_cast<std::uint32_t>(reason_high) << 16) | static_cast<std::uint32_t>(reason_low);
  }

  const SolveFailureLocation solve_failure =
      failure.found ? SolveFailureLocation::from<Dim>(failure.index, failure.component)
                    : SolveFailureLocation{};
  SolveReport solve = prepared_implicit_report(
      status_code, iterations, evaluations, reference_residual, residual, step, condition,
      safeguard_steps, solve_failure, implicit_failure_action(status));
  // The prepared workspace transports the selected local failure as fixed-width statistics so
  // the kernel itself stays allocation-free.  Reconstruct its public diagnostic only after the
  // collective winner (status, reason and location) has been authenticated.  This is deliberately
  // failure-only: the converged report keeps its short bind-safe reason from
  // prepared_implicit_report().
  if (!solve.solved()) {
    solve.reason = std::string("implicit_source_") + local_nonlinear_status_name(status);
    if (reason_code != 0)
      solve.reason += "_reason_" + std::to_string(reason_code);
    if (failure.found) {
      solve.reason += "_index";
      for (int axis = 0; axis < Dim; ++axis)
        solve.reason += "_" + std::to_string(failure.index[axis]);
    }
  }
  if (all_reduce_max(!solve_report_is_publishable(solve, options.max_iters) ? 1L : 0L, lane) != 0)
    return fail(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun, "malformed");
  const bool staged = Access::stage_report(workspace, solve, failed_cells);
  if (all_reduce_min(staged ? 1L : 0L, lane) == 0)
    return fail(SolveStatus::kInvalidInput, SolveAction::kFailRun, "diagnostic");
  Access::arm_publication(workspace, state, state_generation, reason_code);
  pending.armed = false;
  return SolveOutcome::collective_lane(
      std::move(solve), lane,
      SolveOutcome::PublicationHooks{
          &workspace,
          [](void* context) noexcept { Access::publish(*static_cast<Workspace*>(context)); },
          nullptr,
          [](void* context) noexcept { Access::release(*static_cast<Workspace*>(context)); },
          std::move(lifetime),
          [](void* context) { Access::validate_publication(*static_cast<Workspace*>(context)); },
          nullptr,
      });
}

}  // namespace detail

/// Execute a cold-bound local backward-Euler source solve without allocating after `bind`.
template <int Dim, class Model, class MemorySpace, class ProviderAt>
  requires ImplicitProviderPatchBinding<ProviderAt, Dim, provider_count_for<Model, Dim>()>
[[nodiscard]] SolveOutcome backward_euler_source(
    const Model& model, const ProviderAt& provider_at, MultiFab<Dim, MemorySpace>& state,
    PreparedImplicitSourceWorkspace<Dim, MemorySpace>& workspace, std::uint64_t state_generation,
    Real dt, const NewtonOptions& options, const ExecutionLane& lane,
    const ImplicitMask<Model::n_vars>& mask = {},
    const MultiFab<Dim, MemorySpace>* active_cells = nullptr) {
  return detail::backward_euler_source_prepared(model, provider_at, state, workspace,
                                                state_generation, dt, options, lane, mask,
                                                active_cells, {}, []() noexcept {});
}

/// Same prepared solve with an explicit cold-owned workspace lifetime. Generated Program blocks
/// use this to retain a resident workspace if a hierarchy refresh retires the originating block
/// before its returned outcome is consumed.
template <int Dim, class Model, class MemorySpace, class ProviderAt>
  requires ImplicitProviderPatchBinding<ProviderAt, Dim, provider_count_for<Model, Dim>()>
[[nodiscard]] SolveOutcome backward_euler_source(
    const Model& model, const ProviderAt& provider_at, MultiFab<Dim, MemorySpace>& state,
    PreparedImplicitSourceWorkspace<Dim, MemorySpace>& workspace, std::uint64_t state_generation,
    Real dt, const NewtonOptions& options, const ExecutionLane& lane,
    const ImplicitMask<Model::n_vars>& mask, const MultiFab<Dim, MemorySpace>* active_cells,
    std::shared_ptr<void> lifetime) {
  return detail::backward_euler_source_prepared(
      model, provider_at, state, workspace, state_generation, dt, options, lane, mask, active_cells,
      std::move(lifetime), []() noexcept {});
}

/// Compatibility entry point. New runtime bindings must own and reuse a
/// PreparedImplicitSourceWorkspace; this wrapper retains the historical direct-native API only.
template <int Dim, class Model, class MemorySpace, class ProviderAt>
  requires ImplicitProviderPatchBinding<ProviderAt, Dim, provider_count_for<Model, Dim>()>
[[nodiscard]] SolveOutcome backward_euler_source(
    const Model& model, const ProviderAt& provider_at, MultiFab<Dim, MemorySpace>& state, Real dt,
    const NewtonOptions& options, const ExecutionLane& lane,
    const ImplicitMask<Model::n_vars>& mask = {}, NewtonReport* diagnostics = nullptr,
    const MultiFab<Dim, MemorySpace>* active_cells = nullptr) {
  validate_newton_options(options, "backward_euler_source");
  const auto layout_matches = [&](const MultiFab<Dim, MemorySpace>& other) {
    return state.layout() == other.layout() && state.distribution() == other.distribution() &&
           state.local_rank() == other.local_rank() && state.local_size() == other.local_size();
  };
  if (state.ncomp() != Model::n_vars)
    throw std::invalid_argument("Implicit source state component count differs from its Model");
  if (active_cells != nullptr && (active_cells->ncomp() != 1 || !layout_matches(*active_cells)))
    throw std::invalid_argument(
        "Implicit source active-cell mask must have one component and match the state layout");
  if (!lane.active() || state.rank_space().size() != static_cast<std::size_t>(lane.size()) ||
      state.rank_space().linear_rank(state.local_rank()) != static_cast<std::size_t>(lane.rank()))
    throw std::logic_error(
        "backward_euler_source: ND rank space does not match its prepared execution lane");
  auto workspace = std::make_shared<PreparedImplicitSourceWorkspace<Dim, MemorySpace>>();
  workspace->bind(state, 0, diagnostics);
  return detail::backward_euler_source_prepared(
      model, provider_at, state, *workspace, 0, dt, options, lane, mask, active_cells,
      std::static_pointer_cast<void>(workspace), []() noexcept {});
}

}  // namespace pops
