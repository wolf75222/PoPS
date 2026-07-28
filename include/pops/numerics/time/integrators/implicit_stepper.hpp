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
#include <pops/numerics/nonlinear/prepared_local_nonlinear.hpp>
#include <pops/numerics/spatial_operator.hpp>
#include <pops/runtime/numerical_defaults.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
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

template <int N>
struct ImplicitMask {
  bool active = false;
  bool flag[N] = {};
};

template <class Model, int N>
POPS_HD inline bool is_implicit_component(const ImplicitMask<N>& mask, int component) {
  return mask.active ? mask.flag[component] : model_is_implicit<Model>(component);
}

/// Public preparation policy retained by the System authoring surface.  It is translated once into
/// `PreparedLocalNonlinearControls`; it contains no algorithm and cannot select a fallback engine.
/// There is intentionally no failure-policy knob: every failed solve fails the owning transaction.
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

/// Compatibility diagnostics aggregate for the existing inspection API.  `latest` is the common
/// authoritative report; the remaining fields are derived summaries, never an independent outcome.
struct NewtonReport {
  bool enabled = false;
  bool converged = true;
  Real max_residual = Real(0);
  Real max_iters_used = Real(0);
  double n_failed = 0;
  double failed_i = -1;
  double failed_j = -1;
  double failed_comp = -1;
  SolveReport latest{};
  RuntimeDiagnosticsReport diagnostics =
      make_runtime_diagnostics_report("pops.numerics.time.prepared_local_nonlinear");

  void reset() { *this = NewtonReport{}; }
};

namespace detail {

template <class Model>
struct ImplicitSourceResidual {
  static constexpr int N = Model::n_vars;
  Model model;
  typename Model::State initial;
  typename Model::State explicit_target;
  Aux aux;
  Real dt = Real(0);
  ImplicitMask<N> mask{};

  POPS_HD void operator()(const Real (&candidate)[N], Real (&residual)[N]) const {
    typename Model::State state{};
    for (int component = 0; component < N; ++component)
      state[component] = candidate[component];
    const typename Model::State source = model.source(state, aux);
    for (int component = 0; component < N; ++component) {
      residual[component] = is_implicit_component<Model>(mask, component)
                                ? candidate[component] - initial[component] - dt * source[component]
                                : candidate[component] - explicit_target[component];
    }
  }
};

template <class Model>
struct ImplicitSourceAnalyticJacobian {
  static constexpr int N = Model::n_vars;
  Model model;
  Aux aux;
  Real dt = Real(0);
  ImplicitMask<N> mask{};

  POPS_HD bool operator()(const Real (&candidate)[N], Real (&jacobian)[N][N]) const {
    typename Model::State state{};
    for (int component = 0; component < N; ++component)
      state[component] = candidate[component];
    Real source_jacobian[N][N];
    model.source_jacobian(state, aux, source_jacobian);
    for (int row = 0; row < N; ++row)
      for (int column = 0; column < N; ++column) {
        const Real identity = row == column ? Real(1) : Real(0);
        jacobian[row][column] = is_implicit_component<Model>(mask, row)
                                    ? identity - dt * source_jacobian[row][column]
                                    : identity;
      }
    return true;
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
                                                    const Aux& aux, Real dt,
                                                    const NewtonOptions& options,
                                                    const ImplicitMask<Model::n_vars>& mask) {
  constexpr int N = Model::n_vars;
  typename Model::State explicit_target = initial;
  const typename Model::State initial_source = model.source(initial, aux);
  for (int component = 0; component < N; ++component)
    if (!is_implicit_component<Model>(mask, component))
      explicit_target[component] = initial[component] + dt * initial_source[component];

  const ImplicitSourceResidual<Model> residual{model, initial, explicit_target, aux, dt, mask};
  const PreparedLocalNonlinearControls controls = prepared_controls(options);
  if constexpr (HasSourceJacobian<Model>) {
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
      statistics(i, j, 0) = Real(0);
      for (int component = 1; component < 8; ++component)
        statistics(i, j, component) = Real(0);
      statistics(i, j, 8) = Real(0);
      statistics(i, j, 9) = Real(0);
      return;
    }

    const Aux cell_aux = load_aux<aux_comps<Model>()>(aux, i, j);
    const auto problem =
        prepare_implicit_source_problem(model, initial, cell_aux, dt, options, mask);
    Real guess[N];
    for (int component = 0; component < N; ++component)
      guess[component] = initial[component];
    const LocalNonlinearCellResult<N> solved = solve_prepared_local_nonlinear(problem, guess);
    for (int component = 0; component < N; ++component)
      candidate(i, j, component) = solved.value[component];

    const bool failed = !solved.solved();
    statistics(i, j, 0) = static_cast<Real>(local_nonlinear_status_code(solved.status));
    statistics(i, j, 1) = static_cast<Real>(solved.iterations);
    statistics(i, j, 2) = static_cast<Real>(solved.evaluations);
    statistics(i, j, 3) = solved.reference_residual_norm;
    statistics(i, j, 4) = solved.residual_norm;
    statistics(i, j, 5) = solved.step_norm;
    statistics(i, j, 6) = solved.condition_evidence;
    statistics(i, j, 7) = static_cast<Real>(solved.safeguard_steps);
    if (failed) {
      statistics(i, j, 8) =
          detail::encode_local_nonlinear_failure(i, j, solved.failing_component);
      statistics(i, j, 9) = Real(1);
    } else {
      statistics(i, j, 8) = Real(0);
      statistics(i, j, 9) = Real(0);
    }
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

/// Prepare and execute a local backward-Euler source solve.
///
/// The return value is the common consumed outcome.  On failure every rank observes the same
/// collective status, the live field remains byte-for-byte unchanged, and the function throws
/// `runtime_error` (FailRun).  There is no unchecked or warning-only publication route.
template <class Model>
SolveReport backward_euler_source(const Model& model, const MultiFab& aux, MultiFab& state, Real dt,
                                  const NewtonOptions& options,
                                  const ImplicitMask<Model::n_vars>& mask = {},
                                  NewtonReport* diagnostics = nullptr,
                                  const MultiFab* active_cells = nullptr) {
  validate_newton_options(options, "backward_euler_source");
  if (active_cells != nullptr &&
      (active_cells->ncomp() != 1 || active_cells->local_size() != state.local_size()))
    throw std::invalid_argument(
        "Implicit source active-cell mask must have one component and match the state layout");

  MultiFab candidate(state.box_array(), state.dmap(), state.ncomp(), 0);
  MultiFab statistics(state.box_array(), state.dmap(), 10, 0);
  for (int local_index = 0; local_index < state.local_size(); ++local_index) {
    const ConstArray4 active =
        active_cells != nullptr ? active_cells->fab(local_index).const_array() : ConstArray4{};
    for_each_cell(state.box(local_index), detail::PreparedImplicitSourceKernel<Model>{
                                              model,
                                              state.fab(local_index).const_array(),
                                              aux.fab(local_index).const_array(),
                                              active,
                                              active_cells != nullptr,
                                              candidate.fab(local_index).array(),
                                              statistics.fab(local_index).array(),
                                              dt,
                                              options,
                                              mask,
                                          });
  }

  const int status = static_cast<int>(detail::collective_max_component(statistics, 0));
  const int iterations = static_cast<int>(detail::collective_max_component(statistics, 1));
  const int evaluations = static_cast<int>(detail::collective_max_component(statistics, 2));
  const Real reference_residual = detail::collective_max_component(statistics, 3);
  const Real residual = detail::collective_max_component(statistics, 4);
  const Real step = detail::collective_max_component(statistics, 5);
  const Real condition = detail::collective_max_component(statistics, 6);
  const int safeguard_steps = static_cast<int>(detail::collective_max_component(statistics, 7));
  const double failed_cells = detail::collective_sum_component(statistics, 9);
  const Real encoded = detail::collective_max_component(statistics, 8);
  int failed_i = -1;
  int failed_j = -1;
  int failed_component = -1;
  if (failed_cells > 0)
    detail::decode_local_nonlinear_failure(encoded, failed_i, failed_j, failed_component);

  SolveReport report = local_nonlinear_solve_report(
      status, iterations, evaluations, reference_residual, residual, step, condition,
      safeguard_steps, failed_i, failed_j, failed_component, SolveAction::kFailRun);
  if (!report.solved()) {
    std::ostringstream message;
    message << "Implicit source local nonlinear solve failed: " << report.status_name() << " after "
            << report.iters << " iteration(s), residual "
            << static_cast<double>(report.residual_norm) << ", cell (" << report.failed_i << ", "
            << report.failed_j << "), component " << report.failed_component;
    throw std::runtime_error(message.str());
  }

  detail::aggregate_legacy_report(diagnostics, report, failed_cells);
  lincomb(state, Real(1), candidate, Real(0), candidate);
  return report;
}

/// Default implicit stepper.  It is only an execution adapter; the nonlinear algorithm remains the
/// single prepared provider above.
struct ImplicitSourceStepper {
  NewtonOptions options{};

  template <class Coupler, class Block>
  void operator()(Coupler& coupler, Block& block, Real dt, int /*substep*/,
                  int /*substep_count*/) const {
    (void)backward_euler_source(block.model, coupler.aux(), block.U(), dt, options);
  }
};

}  // namespace pops
