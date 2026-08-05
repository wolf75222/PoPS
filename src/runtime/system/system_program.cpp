/// @file
/// @brief Exact-ranked Program forwarding seam of the uniform System facade.

#include "system_impl.hpp"

#include <pops/core/foundation/native_dimension.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace pops {

template <int Dim>
runtime::program::ProgramRuntimeState<Dim>& System<Dim>::program_runtime_state_() {
  return p_->program_;
}

template <int Dim>
void System<Dim>::install_program_step(std::function<void(double)> step) {
  p_->program_.install_unverified_step(std::move(step));
}

template <int Dim>
void System<Dim>::set_program_cadence(int substeps, int stride) {
  require_assembling(p_->lifecycle_, "set_program_cadence");
  p_->program_.set_cadence(substeps, stride, "System");
}

template <int Dim>
int System<Dim>::program_substeps() const {
  return p_->program_.substeps_;
}

template <int Dim>
int System<Dim>::program_stride() const {
  return p_->program_.stride_;
}

template <int Dim>
double System<Dim>::program_cadence_window_dt() const {
  return p_->program_.cadence_window_dt_;
}

template <int Dim>
int System<Dim>::program_cadence_window_steps() const {
  return p_->program_.cadence_window_steps_;
}

template <int Dim>
double System<Dim>::program_cadence_window_start_time() const {
  return p_->program_.cadence_window_start_time_;
}

template <int Dim>
double System<Dim>::program_last_dt() const {
  return static_cast<double>(p_->program_.last_dt_);
}

template <int Dim>
void System<Dim>::restore_program_cadence_window(double accumulated_dt, int held_steps,
                                                 double window_start_time, double accepted_last_dt,
                                                 double accepted_time, int macro_step) {
  p_->program_.restore_cadence_window(accumulated_dt, held_steps, window_start_time,
                                      accepted_last_dt, accepted_time, macro_step, "System");
}

template <int Dim>
int System<Dim>::n_blocks() const {
  return p_->blocks_.size();
}

template <int Dim>
std::size_t System<Dim>::apply_coupling_operators(
    Real dt, const std::vector<MultiFab<Dim>*>& candidate_states) {
  if (!std::isfinite(static_cast<double>(dt)) || dt < Real(0))
    throw std::invalid_argument(
        "System::apply_coupling_operators requires a finite non-negative dt");
  if (candidate_states.size() != p_->sp.size())
    throw std::invalid_argument(
        "System::apply_coupling_operators requires one candidate state per block");

  for (std::size_t block = 0; block < candidate_states.size(); ++block) {
    const MultiFab<Dim>* candidate = candidate_states[block];
    if (candidate == nullptr)
      throw std::invalid_argument(
          "System::apply_coupling_operators received a null candidate state");
    const MultiFab<Dim>& live = p_->sp[block].U;
    if (candidate->layout() != live.layout() || candidate->distribution() != live.distribution() ||
        candidate->local_rank() != live.local_rank() || candidate->ncomp() != live.ncomp() ||
        candidate->ghosts() != live.ghosts())
      throw std::invalid_argument(
          "System::apply_coupling_operators candidate layout differs from its block");
    for (const typename Impl::Species& accepted : p_->sp)
      if (candidate == &accepted.U)
        throw std::invalid_argument(
            "System::apply_coupling_operators cannot mutate accepted live states");
    for (std::size_t previous = 0; previous < block; ++previous)
      if (candidate_states[previous] == candidate)
        throw std::invalid_argument(
            "System::apply_coupling_operators cannot alias two block candidates");
  }
  return p_->coupling_.apply(dt, candidate_states);
}

template <int Dim>
MultiFab<Dim>& System<Dim>::block_state(int block) {
  if (block < 0 || block >= p_->blocks_.size())
    throw std::out_of_range("System::block_state block index is out of range");
  return p_->sp[static_cast<std::size_t>(block)].U;
}

template <int Dim>
void System<Dim>::block_rhs_into(int block, MultiFab<Dim>& state, MultiFab<Dim>& residual) {
  if (block < 0 || block >= p_->blocks_.size())
    throw std::out_of_range("System::block_rhs_into block index is out of range");
  typename Impl::Species& selected = p_->sp[static_cast<std::size_t>(block)];
  if (selected.boundary)
    throw std::runtime_error(
        "System::block_rhs_into requires an exact evaluation point for a prepared boundary");
  if (!selected.rhs_into)
    throw std::runtime_error("System block '" + selected.name +
                             "' lacks a dimension-qualified residual provider");
  selected.rhs_into(state, residual);
}

template <int Dim>
void System<Dim>::block_rhs_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                    int block, MultiFab<Dim>& state, MultiFab<Dim>& residual) {
  block_rhs_group(point, {block}, {&state}, {&residual}, {0});
}

template <int Dim>
void System<Dim>::block_rhs_group(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                  const std::vector<int>& requested_blocks,
                                  const std::vector<MultiFab<Dim>*>& requested_states,
                                  const std::vector<MultiFab<Dim>*>& requested_residuals,
                                  const std::vector<int>& requested_flux_only) {
  if (requested_blocks.empty() || requested_blocks.size() != requested_states.size() ||
      requested_blocks.size() != requested_residuals.size() ||
      requested_blocks.size() != requested_flux_only.size())
    throw std::invalid_argument("System::block_rhs_group has inconsistent request vectors");

  std::vector<MultiFab<Dim>*> states(p_->sp.size(), nullptr);
  std::vector<MultiFab<Dim>*> residuals(p_->sp.size(), nullptr);
  std::vector<int> flux_only(p_->sp.size(), 0);
  for (std::size_t request = 0; request < requested_blocks.size(); ++request) {
    const int block = requested_blocks[request];
    if (block < 0 || block >= p_->blocks_.size())
      throw std::out_of_range("System::block_rhs_group block index is out of range");
    const std::size_t index = static_cast<std::size_t>(block);
    if (states[index] != nullptr || requested_states[request] == nullptr ||
        requested_residuals[request] == nullptr ||
        (requested_flux_only[request] != 0 && requested_flux_only[request] != 1))
      throw std::invalid_argument(
          "System::block_rhs_group requires unique blocks, non-null storage and boolean modes");
    states[index] = requested_states[request];
    residuals[index] = requested_residuals[request];
    flux_only[index] = requested_flux_only[request];
  }
  p_->blocks_.evaluate_rhs_with_interfaces(point, states, residuals, flux_only);
}

template <int Dim>
void System<Dim>::block_rhs_core_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                         int block, MultiFab<Dim>& state, MultiFab<Dim>& residual,
                                         bool flux_only) {
  if (block < 0 || block >= p_->blocks_.size())
    throw std::out_of_range("System core RHS block index is out of range");
  p_->blocks_.evaluate_rhs_core(point, static_cast<std::size_t>(block), state, residual, flux_only);
}

template <int Dim>
void System<Dim>::block_neg_div_flux_into(int block, MultiFab<Dim>& state,
                                          MultiFab<Dim>& residual) {
  if (block < 0 || block >= p_->blocks_.size())
    throw std::out_of_range("System flux-only block index is out of range");
  typename Impl::Species& selected = p_->sp[static_cast<std::size_t>(block)];
  if (!selected.rhs_flux_only)
    throw std::runtime_error("System block '" + selected.name +
                             "' lacks a dimension-qualified flux-only provider");
  selected.rhs_flux_only(state, residual);
}

template <int Dim>
void System<Dim>::block_neg_div_flux_into_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, int block, MultiFab<Dim>& state,
    MultiFab<Dim>& residual) {
  block_rhs_group(point, {block}, {&state}, {&residual}, {1});
}

template <int Dim>
void System<Dim>::block_source_into(int block, MultiFab<Dim>& state, MultiFab<Dim>& residual) {
  if (block < 0 || block >= p_->blocks_.size())
    throw std::out_of_range("System source-only block index is out of range");
  typename Impl::Species& selected = p_->sp[static_cast<std::size_t>(block)];
  if (!selected.source_only)
    throw std::runtime_error("System block '" + selected.name +
                             "' lacks a dimension-qualified source-only provider");
  selected.source_only(state, residual);
}

template <int Dim>
void System<Dim>::require_cartesian_generated_operator(int block,
                                                       const std::string& operation) const {
  if (block < 0 || block >= p_->blocks_.size())
    throw std::out_of_range("System generated Program operator block index is out of range");
  if (operation.empty())
    throw std::invalid_argument("System generated Program operator identity cannot be empty");
}

template <int Dim>
Real System<Dim>::block_max_speed(int block, const MultiFab<Dim>& state) const {
  if (block < 0 || block >= p_->blocks_.size())
    throw std::out_of_range("System maximum-speed block index is out of range");
  const typename Impl::Species& selected = p_->sp[static_cast<std::size_t>(block)];
  if (!selected.max_speed)
    throw std::runtime_error("System block '" + selected.name +
                             "' lacks a dimension-qualified stability-speed provider");
  return selected.max_speed(state);
}

template <int Dim>
Real System<Dim>::cfl_min_dx() const {
  Real result = p_->geom.spacing(0);
  for (int axis = 1; axis < Dim; ++axis)
    result = std::min(result, p_->geom.spacing(axis));
  return result;
}

template <int Dim>
bool System<Dim>::program_is_polar() const {
  return false;
}

template <int Dim>
std::string System<Dim>::installed_program_hash() const {
  return p_->program_.installed_hash_;
}

template <int Dim>
std::string System<Dim>::poisson_solver() const {
  return p_->poisson_solver_;
}

template <int Dim>
void System<Dim>::set_program_block_map(const std::vector<int>& program_to_system) {
  for (std::size_t program = 0; program < program_to_system.size(); ++program) {
    const int block = program_to_system[program];
    if (block < 0 || block >= p_->blocks_.size())
      throw std::out_of_range("System::set_program_block_map block index is out of range");
    for (std::size_t previous = 0; previous < program; ++previous)
      if (block == program_to_system[previous])
        throw std::invalid_argument("System::set_program_block_map has duplicate block routes");
  }
  p_->program_.block_map_ = program_to_system;
}

template <int Dim>
const std::vector<int>& System<Dim>::program_block_map() const {
  return p_->program_.block_map_;
}

template <int Dim>
bool System<Dim>::program_owns_operator_authority(
    const std::array<std::uint64_t, 4>& authority) const noexcept {
  return std::find(p_->program_.operator_authorities_.begin(),
                   p_->program_.operator_authorities_.end(),
                   authority) != p_->program_.operator_authorities_.end();
}

template <int Dim>
void System<Dim>::block_project(int block, MultiFab<Dim>& state) {
  if (block < 0 || block >= p_->blocks_.size())
    throw std::out_of_range("System projection block index is out of range");
  typename Impl::Species& selected = p_->sp[static_cast<std::size_t>(block)];
  if (!selected.project)
    throw std::runtime_error("System block '" + selected.name +
                             "' has no dimension-qualified projection provider");
  selected.project(state);
}

template <int Dim>
void System<Dim>::record_program_diagnostic(const std::string& name, Real value) {
  p_->program_.record_diagnostic(name, value);
}

template <int Dim>
void System<Dim>::record_program_balance_term(const std::string& route, const std::string& term,
                                              Real value) {
  p_->program_.record_balance_term(route, term, value, "System");
}

template <int Dim>
bool System<Dim>::program_balance_consumer_is_due(const std::string& contract,
                                                  const std::string& route, int every_n) const {
  return p_->program_.balance_consumer_is_due(contract, route, every_n, "System");
}

template <int Dim>
Real System<Dim>::program_diagnostic(const std::string& name) const {
  return p_->program_.diagnostic(name, "System");
}

template <int Dim>
std::map<std::string, Real> System<Dim>::program_diagnostics() const {
  return p_->program_.diagnostics();
}

template <int Dim>
std::map<std::string, Real> System<Dim>::accepted_balance_terms(const std::string& route) const {
  if (!p_->external_step_transaction_ || p_->external_step_transaction_committed_)
    throw std::runtime_error(
        "System::_accepted_balance_terms requires an active uncommitted external step transaction");
  return p_->program_.accepted_balance_terms(route, "System");
}

template <int Dim>
std::map<std::string, Real> System<Dim>::selected_accepted_balance_terms(
    const std::string& route, const std::string& block, int component,
    const std::vector<int>& levels, const std::vector<std::string>& automatic_terms) const {
  if (!p_->external_step_transaction_ || p_->external_step_transaction_committed_)
    throw std::runtime_error(
        "System::_selected_accepted_balance_terms requires an active uncommitted external step "
        "transaction");
  const int runtime_block = p_->index(block);
  if (component < 0 || component >= p_->find(block).ncomp)
    throw std::out_of_range("System balance component is out of range");
  if (levels.size() != 1 || levels.front() != 0)
    throw std::invalid_argument("System balance selection requires exactly uniform level zero");
  return p_->program_.selected_accepted_balance_terms(route, runtime_block, component, levels,
                                                      automatic_terms, "System");
}

template <int Dim>
void System<Dim>::begin_step_projection_report() {
  p_->program_.begin_step_projection_report();
}

template <int Dim>
void System<Dim>::note_step_projection(const std::string& name) {
  p_->program_.note_step_projection(name);
}

template <int Dim>
std::vector<std::string> System<Dim>::consume_step_projections() {
  return p_->program_.consume_step_projections();
}

template <int Dim>
void System<Dim>::seed_program_params(int program_block, const std::vector<double>& defaults) {
  p_->program_.seed_params(program_block, defaults);
}

template <int Dim>
void System<Dim>::set_program_params(int program_block, const std::vector<double>& values) {
  p_->program_.set_params(program_block, values, "System");
}

template <int Dim>
RuntimeParams System<Dim>::program_params(int program_block) const {
  return p_->program_.params(program_block);
}

template runtime::program::ProgramRuntimeState<kNativeDimension>&
System<kNativeDimension>::program_runtime_state_();
template void System<kNativeDimension>::install_program_step(std::function<void(double)>);
template void System<kNativeDimension>::set_program_cadence(int, int);
template int System<kNativeDimension>::program_substeps() const;
template int System<kNativeDimension>::program_stride() const;
template double System<kNativeDimension>::program_cadence_window_dt() const;
template int System<kNativeDimension>::program_cadence_window_steps() const;
template double System<kNativeDimension>::program_cadence_window_start_time() const;
template double System<kNativeDimension>::program_last_dt() const;
template void System<kNativeDimension>::restore_program_cadence_window(double, int, double, double,
                                                                       double, int);
template int System<kNativeDimension>::n_blocks() const;
template std::size_t System<kNativeDimension>::apply_coupling_operators(
    Real, const std::vector<MultiFab<kNativeDimension>*>&);
template MultiFab<kNativeDimension>& System<kNativeDimension>::block_state(int);
template void System<kNativeDimension>::block_rhs_into(int, MultiFab<kNativeDimension>&,
                                                       MultiFab<kNativeDimension>&);
template void System<kNativeDimension>::block_rhs_into_at(
    const runtime::multiblock::BoundaryEvaluationPoint&, int, MultiFab<kNativeDimension>&,
    MultiFab<kNativeDimension>&);
template void System<kNativeDimension>::block_rhs_group(
    const runtime::multiblock::BoundaryEvaluationPoint&, const std::vector<int>&,
    const std::vector<MultiFab<kNativeDimension>*>&,
    const std::vector<MultiFab<kNativeDimension>*>&, const std::vector<int>&);
template void System<kNativeDimension>::block_rhs_core_into_at(
    const runtime::multiblock::BoundaryEvaluationPoint&, int, MultiFab<kNativeDimension>&,
    MultiFab<kNativeDimension>&, bool);
template void System<kNativeDimension>::block_neg_div_flux_into(int, MultiFab<kNativeDimension>&,
                                                                MultiFab<kNativeDimension>&);
template void System<kNativeDimension>::block_neg_div_flux_into_at(
    const runtime::multiblock::BoundaryEvaluationPoint&, int, MultiFab<kNativeDimension>&,
    MultiFab<kNativeDimension>&);
template void System<kNativeDimension>::block_source_into(int, MultiFab<kNativeDimension>&,
                                                          MultiFab<kNativeDimension>&);
template void System<kNativeDimension>::require_cartesian_generated_operator(
    int, const std::string&) const;
template Real System<kNativeDimension>::block_max_speed(int,
                                                        const MultiFab<kNativeDimension>&) const;
template Real System<kNativeDimension>::cfl_min_dx() const;
template bool System<kNativeDimension>::program_is_polar() const;
template std::string System<kNativeDimension>::installed_program_hash() const;
template std::string System<kNativeDimension>::poisson_solver() const;
template void System<kNativeDimension>::set_program_block_map(const std::vector<int>&);
template const std::vector<int>& System<kNativeDimension>::program_block_map() const;
template bool System<kNativeDimension>::program_owns_operator_authority(
    const std::array<std::uint64_t, 4>&) const noexcept;
template void System<kNativeDimension>::block_project(int, MultiFab<kNativeDimension>&);
template void System<kNativeDimension>::record_program_diagnostic(const std::string&, Real);
template void System<kNativeDimension>::record_program_balance_term(const std::string&,
                                                                    const std::string&, Real);
template bool System<kNativeDimension>::program_balance_consumer_is_due(const std::string&,
                                                                        const std::string&,
                                                                        int) const;
template Real System<kNativeDimension>::program_diagnostic(const std::string&) const;
template std::map<std::string, Real> System<kNativeDimension>::program_diagnostics() const;
template std::map<std::string, Real> System<kNativeDimension>::accepted_balance_terms(
    const std::string&) const;
template std::map<std::string, Real> System<kNativeDimension>::selected_accepted_balance_terms(
    const std::string&, const std::string&, int, const std::vector<int>&,
    const std::vector<std::string>&) const;
template void System<kNativeDimension>::begin_step_projection_report();
template void System<kNativeDimension>::note_step_projection(const std::string&);
template std::vector<std::string> System<kNativeDimension>::consume_step_projections();
template void System<kNativeDimension>::seed_program_params(int, const std::vector<double>&);
template void System<kNativeDimension>::set_program_params(int, const std::vector<double>&);
template RuntimeParams System<kNativeDimension>::program_params(int) const;

}  // namespace pops
