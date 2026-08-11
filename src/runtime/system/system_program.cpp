/// @file
/// @brief Exact-ranked Program forwarding seam of the uniform System facade.

#include "system_impl.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/program/prepared_scalar_boundary_session.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {
namespace {

template <int Dim>
typename SystemBlockStore<Dim>::EmbeddedResidualFamily& select_embedded_residual_family(
    typename SystemBlockStore<Dim>::BlockState& block,
    runtime::system::PreparedEmbeddedBoundaryMode mode) {
  switch (mode) {
    case runtime::system::PreparedEmbeddedBoundaryMode::staircase:
      return block.staircase_residuals;
    case runtime::system::PreparedEmbeddedBoundaryMode::cut_cell:
      return block.cutcell_residuals;
    case runtime::system::PreparedEmbeddedBoundaryMode::inactive:
      break;
  }
  throw std::logic_error("inactive embedded-boundary mode has no residual family");
}

template <int Dim>
void require_embedded_residual_route(const SystemBlockStore<Dim>& blocks,
                                     const typename SystemBlockStore<Dim>::BlockState& block,
                                     int block_index, bool available, const char* operation) {
  if (block.boundary)
    throw std::runtime_error(std::string(operation) +
                             " requires an EB-qualified hyperbolic boundary provider");
  if (blocks.has_interfaces(block_index))
    throw std::runtime_error(std::string(operation) +
                             " requires an EB-qualified shared-interface provider");
  if (!available)
    throw std::runtime_error(std::string(operation) +
                             " has no exact provider for the selected EB mode and model");
}

template <int Dim>
void require_same_block_field(const MultiFab<Dim>& candidate, const MultiFab<Dim>& accepted,
                              const char* operation) {
  if (candidate.layout() != accepted.layout() ||
      candidate.distribution() != accepted.distribution() ||
      candidate.local_rank() != accepted.local_rank() || candidate.ncomp() != accepted.ncomp() ||
      candidate.ghosts() != accepted.ghosts())
    throw std::invalid_argument(std::string(operation) +
                                " requires the exact prepared block field contract");
}

template <int Dim>
void copy_field_storage(const MultiFab<Dim>& source, MultiFab<Dim>& destination) {
  require_same_block_field(source, destination, "prepared boundary field copy");
  for (std::size_t local = 0; local < source.local_size(); ++local)
    Kokkos::deep_copy(destination.fab(local).storage(), source.fab(local).storage());
  Kokkos::fence();
}

template <int Dim>
MultiFab<Dim> detached_valid_field(const MultiFab<Dim>& source) {
  MultiFab<Dim> result(source.layout(), source.distribution(), source.local_rank(), source.ncomp(),
                       source.ghosts());
  result.set_val(Real(0));
  lincomb(result, Real(1), source, Real(0), source);
  return result;
}

template <int Dim>
Real prepared_boundary_local_norm_inf(const MultiFab<Dim>& field) {
  Real local_norm = Real(0);
  for (int component = 0; component < field.ncomp(); ++component)
    local_norm = std::max(local_norm, norm_inf(field, component));
  return local_norm;
}

template <int Dim>
void require_boundary_point(const runtime::multiblock::BoundaryEvaluationPoint& point,
                            const char* operation) {
  if (point.clock.empty() || point.tick < 0 || point.level != 0 || point.substep < 0 ||
      point.stage < 0 || point.stage_fraction < amr::Rational(0, 1) ||
      amr::Rational(1, 1) < point.stage_fraction || !std::isfinite(point.dt) || point.dt <= 0.0 ||
      !std::isfinite(point.physical_time))
    throw std::invalid_argument(std::string(operation) +
                                " requires a complete Uniform boundary evaluation point");
}

template <int Dim, class Validate>
void collective_boundary_preflight(
    const runtime::multiblock::BoundaryEvaluationPoint& point, int block,
    const System<Dim>* prepared_system, int prepared_block,
    const runtime::multiblock::BoundaryEvaluationPoint& prepared_point, const ExecutionLane& lane,
    const char* operation, Validate&& validate) {
  std::exception_ptr local_error;
  try {
    require_boundary_point<Dim>(point, operation);
    validate();
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error(std::string(operation) + " preflight failed collectively");
  }

  ExactContractBuilder invocation;
  invocation.text(operation)
      .scalar(std::int32_t{Dim})
      .scalar(std::int32_t{block})
      .text(point.clock)
      .scalar(point.tick)
      .scalar(point.level)
      .scalar(point.substep)
      .scalar(point.stage)
      .scalar(point.stage_fraction.numerator)
      .scalar(point.stage_fraction.denominator)
      .scalar(point.dt)
      .scalar(point.physical_time)
      .scalar(prepared_block)
      .text(prepared_point.clock)
      .scalar(prepared_point.tick)
      .scalar(prepared_point.level)
      .scalar(prepared_point.substep)
      .scalar(prepared_point.stage)
      .scalar(prepared_point.stage_fraction.numerator)
      .scalar(prepared_point.stage_fraction.denominator)
      .scalar(prepared_point.dt)
      .scalar(prepared_point.physical_time)
      .scalar(prepared_system == nullptr ? std::uint8_t{0} : std::uint8_t{1})
      .text(lane.identity());
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("system-prepared-boundary-invocation"), invocation.view()}},
          lane.communicator()))
    throw std::runtime_error(std::string(operation) +
                             " differs across MPI ranks before publication");
}

template <int Dim, class Invoke>
void invoke_prepared_boundary_transaction(MultiFab<Dim>& state, MultiFab<Dim>& result,
                                          const ExecutionLane& lane, const char* operation,
                                          Invoke&& invoke) {
  std::exception_ptr local_error;
  std::unique_ptr<MultiFab<Dim>> state_snapshot;
  std::unique_ptr<MultiFab<Dim>> result_snapshot;
  std::unique_ptr<MultiFab<Dim>> candidate;
  try {
    state_snapshot = std::make_unique<MultiFab<Dim>>(state);
    result_snapshot = std::make_unique<MultiFab<Dim>>(result);
    candidate = std::make_unique<MultiFab<Dim>>(result);
    Kokkos::fence();
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error(std::string(operation) +
                             " rollback allocation failed collectively before publication");
  }

  const auto rollback_and_throw = [&](std::exception_ptr failure, const char* reason) -> void {
    std::exception_ptr restore_error;
    try {
      copy_field_storage(*state_snapshot, state);
      copy_field_storage(*result_snapshot, result);
    } catch (...) {
      restore_error = std::current_exception();
    }
    if (all_reduce_max(restore_error ? 1L : 0L, lane) != 0)
      throw std::runtime_error(std::string(operation) + " rollback failed collectively");
    if (lane.size() == 1 && failure) {
      try {
        std::rethrow_exception(failure);
      } catch (const std::exception& error) {
        throw std::runtime_error(std::string(operation) + " " + reason +
                                 " and rolled back collectively: " + error.what());
      } catch (...) {
        std::rethrow_exception(failure);
      }
    }
    throw std::runtime_error(std::string(operation) + " " + reason +
                             " and rolled back collectively");
  };

  local_error = nullptr;
  try {
    invoke(*candidate);
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0)
    rollback_and_throw(local_error, "evaluation failed");

  Real local_norm = Real(0);
  local_error = nullptr;
  try {
    local_norm = prepared_boundary_local_norm_inf(*candidate);
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0)
    rollback_and_throw(local_error, "finite-result validation failed");
  const Real global_norm = static_cast<Real>(all_reduce_max(static_cast<double>(local_norm), lane));
  if (!std::isfinite(global_norm))
    rollback_and_throw({}, "produced a non-finite all-component result");

  local_error = nullptr;
  try {
    copy_field_storage(*candidate, result);
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0)
    rollback_and_throw(local_error, "publication failed");
}

}  // namespace

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
  std::exception_ptr local_error;
  try {
    if (!std::isfinite(static_cast<double>(dt)) || dt < Real(0))
      throw std::invalid_argument(
          "System::apply_coupling_operators requires a finite non-negative dt");
    if (candidate_states.size() != p_->sp.size())
      throw std::invalid_argument(
          "System::apply_coupling_operators requires one candidate state per block");
    if (p_->coupling_.operator_contracts.size() != p_->coupling_.operators.size() ||
        p_->coupling_.operators.size() != p_->coupling_.coupled_operators.size())
      throw std::logic_error("System coupling registry lost its exact provider contracts");

    for (std::size_t block = 0; block < candidate_states.size(); ++block) {
      const MultiFab<Dim>* candidate = candidate_states[block];
      if (candidate == nullptr)
        throw std::invalid_argument(
            "System::apply_coupling_operators received a null candidate state");
      const MultiFab<Dim>& live = p_->sp[block].U;
      if (candidate->layout() != live.layout() ||
          candidate->distribution() != live.distribution() ||
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
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("System coupling application preflight failed collectively");
  }

  ExactContractBuilder invocation;
  invocation.text("pops.system.coupling-application")
      .scalar(std::uint32_t{1})
      .scalar(std::int32_t{Dim})
      .scalar(dt)
      .scalar(static_cast<std::uint64_t>(candidate_states.size()))
      .sequence(
          p_->coupling_.operator_contracts,
          [](ExactContractBuilder& item, const std::string& provider) { item.bytes(provider); });
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("system-coupling-application"), invocation.view()}}))
    throw std::invalid_argument(
        "System coupling application identity or time step differs between MPI ranks");

  // Allocate and deep-copy every rollback image before the first operator can mutate a candidate.
  // MultiFab/Fab copies stay in the active Kokkos memory space; no host mirror participates.
  std::vector<MultiFab<Dim>> rollback;
  try {
    rollback.reserve(candidate_states.size());
    for (const MultiFab<Dim>* candidate : candidate_states)
      rollback.emplace_back(*candidate);
    Kokkos::fence();
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("System coupling rollback allocation failed collectively");
  }

  local_error = nullptr;
  try {
    p_->coupling_.apply(dt, candidate_states);
    Kokkos::fence();
  } catch (...) {
    local_error = std::current_exception();
  }
  const bool failed = all_reduce_max(local_error ? 1L : 0L) != 0;
  if (!failed)
    return p_->coupling_.operators.size();

  std::exception_ptr restore_error;
  try {
    for (std::size_t block = 0; block < candidate_states.size(); ++block) {
      MultiFab<Dim>& candidate = *candidate_states[block];
      const MultiFab<Dim>& snapshot = rollback[block];
      for (std::size_t local = 0; local < candidate.local_size(); ++local)
        Kokkos::deep_copy(candidate.fab(local).storage(), snapshot.fab(local).storage());
    }
    Kokkos::fence();
  } catch (...) {
    restore_error = std::current_exception();
  }
  if (all_reduce_max(restore_error ? 1L : 0L) != 0)
    throw std::runtime_error("System coupling rollback failed collectively");
  if (n_ranks() == 1 && local_error)
    std::rethrow_exception(local_error);
  throw std::runtime_error("System coupling application failed and rolled back collectively");
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
  if (p_->embedded_boundary_ &&
      p_->embedded_boundary_->mode() != runtime::system::PreparedEmbeddedBoundaryMode::inactive) {
    auto& family = select_embedded_residual_family<Dim>(selected, p_->embedded_boundary_->mode());
    require_embedded_residual_route<Dim>(p_->blocks_, selected, block,
                                         static_cast<bool>(family.full), "System::block_rhs_into");
    family.full(state, residual, *p_->embedded_boundary_);
    return;
  }
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
  if (p_->embedded_boundary_ &&
      p_->embedded_boundary_->mode() != runtime::system::PreparedEmbeddedBoundaryMode::inactive) {
    for (std::size_t request = 0; request < requested_blocks.size(); ++request) {
      const int block = requested_blocks[request];
      typename Impl::Species& selected = p_->sp[static_cast<std::size_t>(block)];
      auto& family = select_embedded_residual_family<Dim>(selected, p_->embedded_boundary_->mode());
      auto& closure = requested_flux_only[request] != 0 ? family.flux_only : family.full;
      require_embedded_residual_route<Dim>(p_->blocks_, selected, block, static_cast<bool>(closure),
                                           "System::block_rhs_group");
      closure(*requested_states[request], *requested_residuals[request], *p_->embedded_boundary_);
    }
    return;
  }
  p_->blocks_.evaluate_rhs_with_interfaces(point, states, residuals, flux_only);
}

template <int Dim>
void System<Dim>::block_rhs_core_into_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, int block, MultiFab<Dim>& state,
    MultiFab<Dim>& residual, bool flux_only, const System* prepared_system, int prepared_block,
    const runtime::multiblock::BoundaryEvaluationPoint& prepared_point, const ExecutionLane& lane,
    const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
  const bool valid_block = block >= 0 && block < p_->blocks_.size();
  collective_boundary_preflight<Dim>(
      point, block, prepared_system, prepared_block, prepared_point, lane,
      "System::block_rhs_core_into_at", [&] {
        if (prepared_system != this)
          throw std::invalid_argument("prepared boundary core session belongs to another System");
        if (!valid_block)
          throw std::out_of_range("System core RHS block index is out of range");
        if (prepared_block != block || prepared_point != point)
          throw std::invalid_argument(
              "prepared boundary core session does not match its block or evaluation point");
        if (&transport.lane() != &lane)
          throw std::invalid_argument(
              "prepared boundary core session carries a different execution lane");
      });
  typename Impl::Species& selected = p_->sp[static_cast<std::size_t>(block)];
  collective_boundary_preflight<Dim>(
      point, block, prepared_system, prepared_block, prepared_point, lane,
      "System::block_rhs_core_into_at", [&] {
        require_same_block_field(state, selected.U, "prepared boundary core state");
        require_same_block_field(residual, selected.U, "prepared boundary core result");
        if (p_->embedded_boundary_ && p_->embedded_boundary_->mode() !=
                                          runtime::system::PreparedEmbeddedBoundaryMode::inactive) {
          auto& family =
              select_embedded_residual_family<Dim>(selected, p_->embedded_boundary_->mode());
          auto& closure = flux_only ? family.flux_only : family.full;
          require_embedded_residual_route<Dim>(p_->blocks_, selected, block,
                                               static_cast<bool>(closure),
                                               "System::block_rhs_core_into_at");
        } else if (p_->blocks_.has_interfaces(block)) {
          throw std::runtime_error(
              "System::block_rhs_core_into_at has no prepared split-boundary interface authority");
        } else if (selected.boundary && (!(flux_only ? selected.boundary_flux_core_at_point_prepared
                                                     : selected.boundary_core_at_point_prepared))) {
          throw std::runtime_error(
              "System::block_rhs_core_into_at requires its exact prepared boundary core "
              "authority");
        }
      });
  if (p_->embedded_boundary_ &&
      p_->embedded_boundary_->mode() != runtime::system::PreparedEmbeddedBoundaryMode::inactive) {
    auto& family = select_embedded_residual_family<Dim>(selected, p_->embedded_boundary_->mode());
    auto& closure = flux_only ? family.flux_only : family.full;
    closure(state, residual, *p_->embedded_boundary_);
    return;
  }
  if (selected.boundary) {
    invoke_prepared_boundary_transaction<Dim>(
        state, residual, lane, "System::block_rhs_core_into_at", [&](MultiFab<Dim>& candidate) {
          MultiFab<Dim> evaluation_state = detached_valid_field(state);
          auto& core = flux_only ? selected.boundary_flux_core_at_point_prepared
                                 : selected.boundary_core_at_point_prepared;
          core(point, evaluation_state, candidate, *selected.boundary, lane, transport);
        });
    return;
  }
  p_->blocks_.evaluate_rhs_core(point, static_cast<std::size_t>(block), state, residual, flux_only);
}

template <int Dim>
void System<Dim>::block_rhs_into_at_prepared(
    const runtime::multiblock::BoundaryEvaluationPoint& point, int block, MultiFab<Dim>& state,
    MultiFab<Dim>& residual, const System* prepared_system, int prepared_block,
    const runtime::multiblock::BoundaryEvaluationPoint& prepared_point, const ExecutionLane& lane,
    const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
  const bool valid_block = block >= 0 && block < p_->blocks_.size();
  collective_boundary_preflight<Dim>(
      point, block, prepared_system, prepared_block, prepared_point, lane,
      "System::block_rhs_into_at_prepared", [&] {
        if (prepared_system != this)
          throw std::invalid_argument("prepared boundary RHS session belongs to another System");
        if (!valid_block)
          throw std::out_of_range("System prepared boundary RHS block index is out of range");
        if (prepared_block != block || prepared_point != point)
          throw std::invalid_argument(
              "prepared boundary RHS session does not match its block or evaluation point");
        if (&transport.lane() != &lane)
          throw std::invalid_argument(
              "prepared boundary RHS session carries a different execution lane");
      });
  typename Impl::Species& selected = p_->sp[static_cast<std::size_t>(block)];
  collective_boundary_preflight<Dim>(
      point, block, prepared_system, prepared_block, prepared_point, lane,
      "System::block_rhs_into_at_prepared", [&] {
        if (&state == &residual)
          throw std::invalid_argument("prepared boundary RHS cannot alias state and result");
        require_same_block_field(state, selected.U, "prepared boundary RHS state");
        require_same_block_field(residual, selected.U, "prepared boundary RHS result");
        if (!selected.boundary || !selected.boundary_full_at_point_prepared)
          throw std::runtime_error(
              "System prepared boundary RHS requires one complete generated authority");
        if (p_->blocks_.has_interfaces(block))
          throw std::runtime_error(
              "System prepared boundary RHS has no split shared-interface authority");
        if (p_->embedded_boundary_ && p_->embedded_boundary_->mode() !=
                                          runtime::system::PreparedEmbeddedBoundaryMode::inactive)
          throw std::runtime_error(
              "System prepared boundary RHS is unavailable with an active embedded boundary");
      });
  invoke_prepared_boundary_transaction<Dim>(
      state, residual, lane, "System::block_rhs_into_at_prepared", [&](MultiFab<Dim>& candidate) {
        MultiFab<Dim> evaluation_state = detached_valid_field(state);
        selected.boundary_full_at_point_prepared(point, evaluation_state, candidate,
                                                 *selected.boundary, lane, transport);
      });
}

template <int Dim>
const ExecutionLane& System<Dim>::prepared_boundary_execution_lane() const {
  if (!prepared_boundary_execution_lane_)
    throw std::logic_error("System has no RuntimeInstance-prepared boundary execution lane");
  return *prepared_boundary_execution_lane_;
}

template <int Dim>
bool System<Dim>::requires_block_boundary_session(int block) const {
  if (block < 0 || block >= p_->blocks_.size())
    return false;
  const typename Impl::Species& selected = p_->sp[static_cast<std::size_t>(block)];
  return selected.boundary != nullptr && !p_->blocks_.has_interfaces(block) &&
         !(p_->embedded_boundary_ && p_->embedded_boundary_->mode() !=
                                         runtime::system::PreparedEmbeddedBoundaryMode::inactive);
}

template <int Dim>
bool System<Dim>::has_block_boundary_linearization(int block) const {
  if (block < 0 || block >= p_->blocks_.size())
    return false;
  const typename Impl::Species& selected = p_->sp[static_cast<std::size_t>(block)];
  return selected.boundary != nullptr &&
         static_cast<bool>(selected.boundary_core_at_point_prepared) &&
         static_cast<bool>(selected.boundary_residual_at_point_prepared) &&
         static_cast<bool>(selected.boundary_full_at_point_prepared) &&
         static_cast<bool>(selected.boundary_jvp_at_point_prepared) &&
         !p_->blocks_.has_interfaces(block) &&
         !(p_->embedded_boundary_ && p_->embedded_boundary_->mode() !=
                                         runtime::system::PreparedEmbeddedBoundaryMode::inactive);
}

template <int Dim>
void System<Dim>::block_boundary_residual_into_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, int block, MultiFab<Dim>& state,
    MultiFab<Dim>& residual, const System* prepared_system, int prepared_block,
    const runtime::multiblock::BoundaryEvaluationPoint& prepared_point, const ExecutionLane& lane,
    const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
  const bool valid_block = block >= 0 && block < p_->blocks_.size();
  collective_boundary_preflight<Dim>(
      point, block, prepared_system, prepared_block, prepared_point, lane,
      "System::block_boundary_residual_into_at", [&] {
        if (prepared_system != this)
          throw std::invalid_argument(
              "prepared boundary residual session belongs to another System");
        if (!valid_block)
          throw std::out_of_range("System boundary residual block index is out of range");
        if (prepared_block != block || prepared_point != point)
          throw std::invalid_argument(
              "prepared boundary residual session does not match its block or evaluation point");
        if (&transport.lane() != &lane)
          throw std::invalid_argument(
              "prepared boundary residual session carries a different execution lane");
      });
  typename Impl::Species& selected = p_->sp[static_cast<std::size_t>(block)];
  collective_boundary_preflight<Dim>(
      point, block, prepared_system, prepared_block, prepared_point, lane,
      "System::block_boundary_residual_into_at", [&] {
        if (&state == &residual)
          throw std::invalid_argument("prepared boundary residual cannot alias state and result");
        require_same_block_field(state, selected.U, "prepared boundary residual state");
        require_same_block_field(residual, selected.U, "prepared boundary residual result");
        if (!selected.boundary || !selected.boundary_residual_at_point_prepared)
          throw std::runtime_error(
              "System boundary residual requires one complete prepared boundary authority");
        if (p_->blocks_.has_interfaces(block))
          throw std::runtime_error(
              "System boundary residual has no prepared shared-interface authority");
        if (p_->embedded_boundary_ && p_->embedded_boundary_->mode() !=
                                          runtime::system::PreparedEmbeddedBoundaryMode::inactive)
          throw std::runtime_error(
              "System boundary residual is unavailable with an active embedded boundary");
      });
  invoke_prepared_boundary_transaction<Dim>(
      state, residual, lane, "System::block_boundary_residual_into_at",
      [&](MultiFab<Dim>& candidate) {
        selected.boundary_residual_at_point_prepared(point, state, candidate, *selected.boundary,
                                                     lane, transport);
      });
}

template <int Dim>
void System<Dim>::block_boundary_jvp_into_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, int block, MultiFab<Dim>& state,
    const MultiFab<Dim>& direction, MultiFab<Dim>& result, const System* prepared_system,
    int prepared_block, const runtime::multiblock::BoundaryEvaluationPoint& prepared_point,
    const ExecutionLane& lane,
    const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
  const bool valid_block = block >= 0 && block < p_->blocks_.size();
  collective_boundary_preflight<Dim>(
      point, block, prepared_system, prepared_block, prepared_point, lane,
      "System::block_boundary_jvp_into_at", [&] {
        if (prepared_system != this)
          throw std::invalid_argument("prepared boundary JVP session belongs to another System");
        if (!valid_block)
          throw std::out_of_range("System boundary JVP block index is out of range");
        if (prepared_block != block || prepared_point != point)
          throw std::invalid_argument(
              "prepared boundary JVP session does not match its block or evaluation point");
        if (&transport.lane() != &lane)
          throw std::invalid_argument(
              "prepared boundary JVP session carries a different execution lane");
      });
  typename Impl::Species& selected = p_->sp[static_cast<std::size_t>(block)];
  collective_boundary_preflight<Dim>(
      point, block, prepared_system, prepared_block, prepared_point, lane,
      "System::block_boundary_jvp_into_at", [&] {
        if (&state == &result || &direction == &result)
          throw std::invalid_argument("prepared boundary JVP result cannot alias an input field");
        require_same_block_field(state, selected.U, "prepared boundary JVP state");
        require_same_block_field(direction, selected.U, "prepared boundary JVP direction");
        require_same_block_field(result, selected.U, "prepared boundary JVP result");
        if (!selected.boundary || !selected.boundary_jvp_at_point_prepared)
          throw std::runtime_error(
              "System boundary JVP requires one complete prepared boundary authority");
        if (p_->blocks_.has_interfaces(block))
          throw std::runtime_error(
              "System boundary JVP has no prepared shared-interface authority");
        if (p_->embedded_boundary_ && p_->embedded_boundary_->mode() !=
                                          runtime::system::PreparedEmbeddedBoundaryMode::inactive)
          throw std::runtime_error(
              "System boundary JVP is unavailable with an active embedded boundary");
      });
  invoke_prepared_boundary_transaction<Dim>(
      state, result, lane, "System::block_boundary_jvp_into_at", [&](MultiFab<Dim>& candidate) {
        selected.boundary_jvp_at_point_prepared(point, state, direction, candidate,
                                                *selected.boundary, lane, transport);
      });
}

template <int Dim>
void System<Dim>::block_prepare_generated_state_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, int block, MultiFab<Dim>& state) {
  if (block < 0 || block >= p_->blocks_.size())
    throw std::out_of_range("System generated-state block index is out of range");
  p_->blocks_.prepare_generated_state(point, static_cast<std::size_t>(block), state);
}

template <int Dim>
void System<Dim>::block_neg_div_flux_into(int block, MultiFab<Dim>& state,
                                          MultiFab<Dim>& residual) {
  if (block < 0 || block >= p_->blocks_.size())
    throw std::out_of_range("System flux-only block index is out of range");
  typename Impl::Species& selected = p_->sp[static_cast<std::size_t>(block)];
  if (p_->embedded_boundary_ &&
      p_->embedded_boundary_->mode() != runtime::system::PreparedEmbeddedBoundaryMode::inactive) {
    auto& family = select_embedded_residual_family<Dim>(selected, p_->embedded_boundary_->mode());
    require_embedded_residual_route<Dim>(p_->blocks_, selected, block,
                                         static_cast<bool>(family.flux_only),
                                         "System::block_neg_div_flux_into");
    family.flux_only(state, residual, *p_->embedded_boundary_);
    return;
  }
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
  if (p_->embedded_boundary_ &&
      p_->embedded_boundary_->mode() != runtime::system::PreparedEmbeddedBoundaryMode::inactive) {
    auto& family = select_embedded_residual_family<Dim>(selected, p_->embedded_boundary_->mode());
    require_embedded_residual_route<Dim>(p_->blocks_, selected, block,
                                         static_cast<bool>(family.source_only),
                                         "System::block_source_into");
    family.source_only(state, residual, *p_->embedded_boundary_);
    return;
  }
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
  if (p_->embedded_boundary_ &&
      p_->embedded_boundary_->mode() != runtime::system::PreparedEmbeddedBoundaryMode::inactive) {
    auto& family = select_embedded_residual_family<Dim>(selected, p_->embedded_boundary_->mode());
    require_embedded_residual_route<Dim>(
        p_->blocks_, selected, block, static_cast<bool>(family.project), "System::block_project");
    family.project(state, *p_->embedded_boundary_);
    return;
  }
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
    MultiFab<kNativeDimension>&, bool, const System<kNativeDimension>*, int,
    const runtime::multiblock::BoundaryEvaluationPoint&, const ExecutionLane&,
    const runtime::program::PreparedScalarBoundarySession<kNativeDimension>&);
template void System<kNativeDimension>::block_rhs_into_at_prepared(
    const runtime::multiblock::BoundaryEvaluationPoint&, int, MultiFab<kNativeDimension>&,
    MultiFab<kNativeDimension>&, const System<kNativeDimension>*, int,
    const runtime::multiblock::BoundaryEvaluationPoint&, const ExecutionLane&,
    const runtime::program::PreparedScalarBoundarySession<kNativeDimension>&);
template const ExecutionLane& System<kNativeDimension>::prepared_boundary_execution_lane() const;
template bool System<kNativeDimension>::requires_block_boundary_session(int) const;
template bool System<kNativeDimension>::has_block_boundary_linearization(int) const;
template void System<kNativeDimension>::block_boundary_residual_into_at(
    const runtime::multiblock::BoundaryEvaluationPoint&, int, MultiFab<kNativeDimension>&,
    MultiFab<kNativeDimension>&, const System<kNativeDimension>*, int,
    const runtime::multiblock::BoundaryEvaluationPoint&, const ExecutionLane&,
    const runtime::program::PreparedScalarBoundarySession<kNativeDimension>&);
template void System<kNativeDimension>::block_boundary_jvp_into_at(
    const runtime::multiblock::BoundaryEvaluationPoint&, int, MultiFab<kNativeDimension>&,
    const MultiFab<kNativeDimension>&, MultiFab<kNativeDimension>&, const System<kNativeDimension>*,
    int, const runtime::multiblock::BoundaryEvaluationPoint&, const ExecutionLane&,
    const runtime::program::PreparedScalarBoundarySession<kNativeDimension>&);
template void System<kNativeDimension>::block_prepare_generated_state_at(
    const runtime::multiblock::BoundaryEvaluationPoint&, int, MultiFab<kNativeDimension>&);
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
