// Core exact-ranked System facade. Provider-specific installation and execution seams live in
// sibling translation units; this file owns only layout-independent lifecycle, clock and cadence.
#include "system_impl.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/program/profiler.hpp>
#include <pops/runtime/program/step_transaction.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pops {

POPS_EXPORT std::string abi_key() {
  return detail::abi_key_string();
}

template <int Dim>
std::string System<Dim>::abi_key() {
  return pops::abi_key();
}

template <int Dim>
System<Dim>::System(const SystemConfig<Dim>& config) {
  validate_system_config(config);
  p_ = std::make_unique<Impl>(config);
}

template <int Dim>
System<Dim>::~System() = default;

template <int Dim>
System<Dim>::System(System&&) noexcept = default;

template <int Dim>
System<Dim>& System<Dim>::operator=(System&&) noexcept = default;

template <int Dim>
void System<Dim>::step(double dt) {
  p_->program_.require_step_installed("System::step");
  runtime::program::ProfileScope scope(p_->program_.profiler_, "step");
  p_->program_.profiler_.count("steps");
  p_->execute_step_transaction(
      [&] { p_->program_.dispatch_cadence_step(p_->t, p_->macro_step_, dt, "System"); });
}

template <int Dim>
void System<Dim>::advance(double dt, int nsteps) {
  p_->program_.require_step_installed("System::advance");
  if (nsteps < 0)
    throw std::invalid_argument("System::advance requires a non-negative step count");
  for (int step_index = 0; step_index < nsteps; ++step_index)
    step(dt);
}

template <int Dim>
void System<Dim>::begin_step_transaction() {
  if (p_->external_step_transaction_)
    throw std::runtime_error("System::begin_step_transaction: transaction already active");
  p_->external_step_transaction_ = std::make_unique<typename Impl::AcceptedSnapshot>(*p_);
  p_->external_step_transaction_committed_ = false;
}

template <int Dim>
void System<Dim>::commit_step_transaction() {
  if (!p_->external_step_transaction_)
    throw std::runtime_error("System::commit_step_transaction: no active transaction");
  if (p_->external_step_transaction_committed_)
    throw std::runtime_error("System::commit_step_transaction: transaction already committed");
  p_->external_step_transaction_committed_ = true;
}

template <int Dim>
std::map<std::string, double> System<Dim>::step_change_l2() const {
  if (!p_->external_step_transaction_)
    throw std::runtime_error("System::step_change_l2 requires an active external step transaction");
  const std::vector<MultiFab<Dim>>& previous = p_->external_step_transaction_->states;
  if (previous.size() != p_->sp.size())
    throw std::runtime_error("System::step_change_l2 snapshot composition mismatch");

  double cell_measure = 1.0;
  for (int axis = 0; axis < Dim; ++axis)
    cell_measure *= static_cast<double>(p_->geom.spacing(axis));

  std::map<std::string, double> result;
  for (std::size_t block = 0; block < p_->sp.size(); ++block) {
    const double sum_sq =
        static_cast<double>(difference_sum_sq_all(p_->sp[block].U, previous[block]));
    result.emplace(p_->sp[block].name, std::sqrt(cell_measure * sum_sq));
  }
  return result;
}

template <int Dim>
void System<Dim>::finalize_step_transaction() {
  if (!p_->external_step_transaction_ || !p_->external_step_transaction_committed_)
    throw std::runtime_error("System::finalize_step_transaction: no committed transaction");
  p_->external_step_transaction_.reset();
  p_->external_step_transaction_committed_ = false;
}

template <int Dim>
void System<Dim>::rollback_step_transaction() {
  if (!p_->external_step_transaction_)
    throw std::runtime_error("System::rollback_step_transaction: no active transaction");
  p_->external_step_transaction_->restore(*p_);
  p_->external_step_transaction_.reset();
  p_->external_step_transaction_committed_ = false;
}

template <int Dim>
double System<Dim>::step_cfl(double cfl, double speed_floor, double max_dt, double min_dt) {
  p_->program_.require_step_installed("System::step_cfl");
  if (!std::isfinite(cfl) || !(cfl > 0.0))
    throw std::invalid_argument("System::step_cfl cfl must be finite and positive");
  if (!std::isfinite(speed_floor) || !(speed_floor > 0.0))
    throw std::invalid_argument("System::step_cfl speed_floor must be finite and positive");
  if (std::isnan(max_dt) || max_dt <= 0.0)
    throw std::invalid_argument("System::step_cfl max_dt must be positive or +infinity");
  if (!std::isfinite(min_dt) || min_dt < 0.0)
    throw std::invalid_argument("System::step_cfl min_dt must be finite and non-negative");

  SolveOutcome field_outcome = solve_fields();
  const SolveConsumption field_consumption =
      field_outcome.report().solved_value_available()
          ? SolveConsumption::kAccept
          : (field_outcome.report().action == SolveAction::kRejectAttempt
                 ? SolveConsumption::kRejectAttempt
                 : SolveConsumption::kFailRun);
  const SolveReport field_report = field_outcome.consume(field_consumption);
  if (!field_report.solved_value_available()) {
    if (field_consumption == SolveConsumption::kRejectAttempt)
      throw runtime::program::StepAttemptRejected(field_report.status, "CFL field evaluation",
                                                  field_report.reason);
    throw std::runtime_error(std::string("System::step_cfl field evaluation failed: status=") +
                             field_report.status_name() + " action=" + field_report.action_name() +
                             " reason=" + field_report.reason);
  }

  Real minimum_spacing = p_->geom.spacing(0);
  for (int axis = 1; axis < Dim; ++axis)
    minimum_spacing = std::min(minimum_spacing, p_->geom.spacing(axis));

  double selected = std::numeric_limits<double>::infinity();
  std::string reason = "degenerate";
  for (typename Impl::Species& block : p_->sp) {
    if (!block.evolve)
      continue;
    if (!block.max_speed)
      throw std::runtime_error("System block '" + block.name +
                               "' lacks a dimension-qualified stability-speed provider");
    const Real speed = std::max(block.max_speed(block.U), static_cast<Real>(speed_floor));
    double block_dt = cfl * static_cast<double>(minimum_spacing) * block.substeps /
                      (static_cast<double>(block.stride) * static_cast<double>(speed));
    const char* block_reason = "transport";
    if (block.source_frequency) {
      const Real frequency = block.source_frequency(block.U);
      if (frequency > Real(0)) {
        const double source_dt =
            cfl * block.substeps /
            (static_cast<double>(block.stride) * static_cast<double>(frequency));
        if (source_dt < block_dt) {
          block_dt = source_dt;
          block_reason = "source_frequency";
        }
      }
    }
    if (block.stability_dt) {
      const Real admissible = block.stability_dt(block.U);
      if (admissible > Real(0)) {
        const double admissible_dt =
            static_cast<double>(admissible) * block.substeps / static_cast<double>(block.stride);
        if (admissible_dt < block_dt) {
          block_dt = admissible_dt;
          block_reason = "stability_dt";
        }
      }
    }
    if (block_dt < selected) {
      selected = block_dt;
      reason = std::string(block_reason) + ":" + block.name;
    }
  }

  for (const runtime::system::CoupledFreq& frequency : p_->coupling_.coupled_freqs) {
    if (!(frequency.mu > 0.0))
      continue;
    const double candidate = cfl / frequency.mu;
    if (candidate < selected) {
      selected = candidate;
      reason = "coupled_source:" + frequency.label;
    }
  }
  for (const runtime::system::PreparedCoupledFrequency& frequency :
       p_->coupling_.coupled_frequencies) {
    if (!frequency.maximum_frequency)
      continue;
    const double maximum_frequency = static_cast<double>(frequency.maximum_frequency());
    if (!std::isfinite(maximum_frequency))
      throw std::runtime_error(
          "System coupled-source frequency provider returned a non-finite "
          "maximum for '" +
          frequency.label + "'");
    if (!(maximum_frequency > 0.0))
      continue;
    const double candidate = cfl / maximum_frequency;
    if (candidate < selected) {
      selected = candidate;
      reason = "coupled_source:" + frequency.label;
    }
  }
  for (const runtime::system::GlobalDtBound& bound : p_->coupling_.dt_bounds) {
    if (!bound.fn)
      continue;
    double candidate = bound.fn();
    if (!(candidate > 0.0) || !std::isfinite(candidate))
      candidate = std::numeric_limits<double>::infinity();
    candidate = all_reduce_min(candidate);
    if (candidate < selected) {
      selected = candidate;
      reason = "global:" + bound.label;
    }
  }

  if (p_->program_.dt_bound_) {
    const double program_dt = static_cast<double>(p_->program_.dt_bound_(static_cast<Real>(cfl)));
    if (std::isfinite(program_dt) && program_dt > 0.0 && program_dt < selected) {
      selected = program_dt;
      reason = "program:dt_bound";
    }
  }
  if (!std::isfinite(selected))
    selected = cfl * static_cast<double>(minimum_spacing) / speed_floor;
  if (max_dt < selected) {
    selected = max_dt;
    reason = "strategy:max_dt";
  }
  if (selected < min_dt)
    throw std::runtime_error("System::step_cfl stability bound is below declared min_dt");

  p_->last_dt_reason_ = std::move(reason);
  p_->execute_step_transaction(
      [&] { p_->program_.dispatch_cadence_step(p_->t, p_->macro_step_, selected, "System"); });
  return selected;
}

template <int Dim>
int System<Dim>::macro_step() const {
  return p_->macro_step_;
}

template <int Dim>
void System<Dim>::mark_bound() {
  if (p_->lifecycle_.frozen())
    p_->lifecycle_.to_bound();

  const auto& state_routes = p_->boundary_registry_.state_routes();
  if (!state_routes.empty() && state_routes.size() != p_->sp.size())
    throw std::runtime_error(
        "System::mark_bound: block state routes do not exactly cover materialized blocks");
  for (typename Impl::Species& block : p_->sp) {
    const auto route = state_routes.find(block.name);
    if (!state_routes.empty() && route == state_routes.end())
      throw std::runtime_error(
          "System::mark_bound: materialized block lacks its exact state route");
    if (route != state_routes.end())
      block.state_identity = route->second;
  }

  for (const auto& [name, installed] : p_->boundary_registry_.boundaries()) {
    typename Impl::Species& block = p_->find(name);
    if (installed.authority->ncomp() != block.ncomp)
      throw std::runtime_error("System::mark_bound: boundary component count differs from block '" +
                               name + "'");
    if (installed.state_identity != block.state_identity)
      throw std::runtime_error("System::mark_bound: boundary state identity differs from block '" +
                               name + "'");
    if (installed.authority->periodic_axes() != p_->periodicity)
      throw std::runtime_error(
          "System::mark_bound: boundary periodicity differs from the domain for block '" + name +
          "'");
    for (int axis = 0; axis < Dim; ++axis)
      if (block.U.ghosts()[axis] < installed.required_depth)
        throw std::runtime_error("System::mark_bound: boundary depth exceeds block storage for '" +
                                 name + "'");
    p_->publish_boundary_to_block(name);
  }
  p_->lifecycle_.to_bound();
}

template <int Dim>
std::string System<Dim>::lifecycle_state() const {
  return p_->lifecycle_.state(p_->macro_step_);
}

template <int Dim>
runtime::program::CacheManager<Dim>& System<Dim>::program_cache() {
  return p_->program_.cache_;
}

template <int Dim>
Extent<Dim> System<Dim>::spatial_shape() const {
  return p_->cfg.shape;
}

template <int Dim>
double System<Dim>::time() const {
  return p_->t;
}

template <int Dim>
int System<Dim>::n_species() const {
  return p_->blocks_.size();
}

template <int Dim>
std::vector<std::string> System<Dim>::block_names() const {
  return p_->blocks_.names();
}

template <int Dim>
EffectiveOptionsReport System<Dim>::effective_options_report() const {
  EffectiveOptionsReport report;
  report.runtime = "system";
  report.topology.dimension = Dim;
  report.topology.periodicity.reserve(Dim);
  for (int axis = 0; axis < Dim; ++axis)
    report.topology.periodicity.push_back(p_->periodicity[axis]);

  report.blocks.reserve(p_->sp.size());
  for (const typename Impl::Species& block : p_->sp) {
    EffectiveBlockOptions row;
    row.name = block.name;
    row.ncomp = block.ncomp;
    row.substeps = block.substeps;
    row.stride = block.stride;
    row.evolve = block.evolve;
    row.gamma = block.gamma;
    row.conservative_vars = block.cons_vars.names;
    row.primitive_vars = block.prim_vars.names;
    const Extent<Dim> ghosts = block.U.ghosts();
    row.n_ghost = ghosts[0];
    for (int axis = 1; axis < Dim; ++axis)
      if (ghosts[axis] != row.n_ghost)
        throw std::runtime_error(
            "System effective-options schema cannot project anisotropic ghost extents");
    report.blocks.push_back(std::move(row));
  }
  return report;
}

template std::string System<kNativeDimension>::abi_key();
template System<kNativeDimension>::System(const SystemConfig<kNativeDimension>&);
template System<kNativeDimension>::~System();
template System<kNativeDimension>::System(System&&) noexcept;
template System<kNativeDimension>& System<kNativeDimension>::operator=(System&&) noexcept;
template void System<kNativeDimension>::step(double);
template void System<kNativeDimension>::advance(double, int);
template void System<kNativeDimension>::begin_step_transaction();
template void System<kNativeDimension>::commit_step_transaction();
template std::map<std::string, double> System<kNativeDimension>::step_change_l2() const;
template void System<kNativeDimension>::finalize_step_transaction();
template void System<kNativeDimension>::rollback_step_transaction();
template double System<kNativeDimension>::step_cfl(double, double, double, double);
template int System<kNativeDimension>::macro_step() const;
template void System<kNativeDimension>::mark_bound();
template std::string System<kNativeDimension>::lifecycle_state() const;
template runtime::program::CacheManager<kNativeDimension>&
System<kNativeDimension>::program_cache();
template Extent<kNativeDimension> System<kNativeDimension>::spatial_shape() const;
template double System<kNativeDimension>::time() const;
template int System<kNativeDimension>::n_species() const;
template std::vector<std::string> System<kNativeDimension>::block_names() const;
template EffectiveOptionsReport System<kNativeDimension>::effective_options_report() const;

}  // namespace pops
