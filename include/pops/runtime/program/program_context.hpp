/// @file
/// @brief Exact compile-time-ranked execution boundary for generated Uniform Programs.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/runtime/config/runtime_params.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <pops/runtime/program/clock_schedule.hpp>
#include <pops/runtime/program/program_runtime_state.hpp>
#include <pops/runtime/system.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace pops::runtime::program {

/// A generated Program and its Uniform runtime have one immutable native rank.
///
/// The context never decodes a dimension tag, infers a missing axis, or substitutes a legacy
/// two-dimensional grid provider.  Every state, scratch, history and cache value is a
/// `MultiFab<Dim>` copied from an already-authenticated runtime layout.  Operations whose providers
/// have not yet crossed the exact-ranked boundary fail before touching storage.
template <int Dim>
class ProgramContext {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "ProgramContext only supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;
  using runtime_type = System<Dim>;
  using field_type = MultiFab<Dim>;
  using runtime_state_type = ProgramRuntimeState<Dim>;

  struct FieldStageOverride {
    int program_block = -1;
    const field_type* state = nullptr;
  };

  struct RhsGroupRequest {
    RhsGroupRequest(int block_value, field_type* state_value, field_type* rhs_value,
                    int rate_id_value, int flux_only_value)
        : block(block_value),
          state(state_value),
          rhs(rhs_value),
          rate_id(rate_id_value),
          flux_only(flux_only_value) {}

    int block = -1;
    field_type* state = nullptr;
    field_type* rhs = nullptr;
    int rate_id = -1;
    int flux_only = 0;
  };

  struct CouplingStateOverride {
    int program_block = -1;
    field_type* state = nullptr;
  };

  explicit ProgramContext(runtime_type* system) : system_(system) {
    if (system_ == nullptr)
      throw std::invalid_argument("ProgramContext requires a non-null ranked System");
  }

  void install(std::function<void(double)> step) const {
    system_->install_program_step(std::move(step));
  }

  void begin_step(double dt) const {
    if (!std::isfinite(dt) || dt <= 0.0)
      throw std::invalid_argument("ProgramContext step requires a finite positive dt");
    current_dt_ = dt;
    stage_time_ = amr::Rational(0, 1);
  }

  void configure_primary_clock(const std::string& clock) const {
    clock_schedule_.configure_primary_clock(clock);
    primary_clock_ = clock;
  }

  void declare_clock_relation(const std::string& parent, const std::string& child,
                              int count) const {
    clock_schedule_.declare_relation(parent, child, count);
  }

  void set_stage_time(std::int64_t numerator, std::int64_t denominator) const {
    if (denominator <= 0 || numerator < 0 || numerator > denominator)
      throw std::invalid_argument("ProgramContext stage time is outside [0, 1]");
    stage_time_ = amr::Rational(numerator, denominator);
  }

  runtime::multiblock::BoundaryEvaluationPoint boundary_evaluation_point(int stage) const {
    require_rate_identity_(stage);
    if (primary_clock_.empty() || !std::isfinite(current_dt_) || current_dt_ <= 0.0)
      throw std::logic_error("ProgramContext boundary evaluation has no prepared clock and dt");
    return {primary_clock_,
            static_cast<std::int64_t>(macro_step()),
            0,
            0,
            stage,
            stage_time_,
            current_dt_,
            physical_time() + stage_time_.value() * current_dt_};
  }

  int n_blocks() const { return system_->n_blocks(); }

  int sys_block(int program_block) const {
    const std::vector<int>& map = system_->program_block_map();
    if (map.empty())
      throw std::runtime_error(
          "ProgramContext has no explicit program-to-runtime block map; positional identity is "
          "not supported");
    if (program_block < 0 || program_block >= static_cast<int>(map.size()))
      throw std::out_of_range("ProgramContext block is outside the authenticated map");
    const int runtime_block = map[static_cast<std::size_t>(program_block)];
    if (runtime_block < 0 || runtime_block >= system_->n_blocks())
      throw std::runtime_error("ProgramContext block map targets an absent runtime block");
    return runtime_block;
  }

  field_type& state(int program_block) const { return system_->block_state(sys_block(program_block)); }
  field_type& aux() const { return system_->prepared_block_auxiliary(); }

  field_type rhs_scratch_like(const field_type& prototype) const {
    return make_scratch_(prototype, prototype.ncomp(), prototype.ghosts());
  }

  field_type scratch_state_like(const field_type& prototype) const {
    return make_scratch_(prototype, prototype.ncomp(), prototype.ghosts());
  }

  field_type& rhs_scratch(std::int64_t value_id, int subslot,
                          const field_type& prototype) const {
    return persistent_scratch_(ScratchKind::Rhs, value_id, subslot, prototype,
                               prototype.ncomp(), prototype.ghosts());
  }

  field_type& scratch_state(std::int64_t value_id, int subslot,
                            const field_type& prototype) const {
    return persistent_scratch_(ScratchKind::State, value_id, subslot, prototype,
                               prototype.ncomp(), prototype.ghosts());
  }

  field_type& scalar_scratch(std::int64_t value_id, int subslot,
                             const field_type& prototype, int ncomp = 1,
                             int ghost_depth = 1) const {
    if (ncomp < 1 || ghost_depth < 0)
      throw std::invalid_argument(
          "ProgramContext scalar scratch requires positive components and non-negative ghosts");
    Extent<Dim> ghosts{};
    for (int axis = 0; axis < Dim; ++axis)
      ghosts[axis] = ghost_depth;
    return persistent_scratch_(ScratchKind::Scalar, value_id, subslot, prototype, ncomp, ghosts);
  }

  field_type alloc_scalar_field(int ncomp = 1, int ghost_depth = 1) const {
    return scalar_field_like_(state(0), ncomp, ghost_depth);
  }

  void rhs_into(int program_block, field_type& state_value, field_type& rhs,
                int rate_id) const {
    require_rate_identity_(rate_id);
    count_kernel_();
    system_->block_rhs_into_at(boundary_evaluation_point(rate_id), sys_block(program_block),
                               state_value, rhs);
  }

  void neg_div_flux_default_into(int program_block, field_type& state_value, field_type& rhs,
                                 int rate_id) const {
    require_rate_identity_(rate_id);
    count_kernel_();
    system_->block_neg_div_flux_into_at(boundary_evaluation_point(rate_id),
                                        sys_block(program_block), state_value, rhs);
  }

  void source_default_into(int program_block, field_type& state_value, field_type& rhs) const {
    count_kernel_();
    system_->block_source_into(sys_block(program_block), state_value, rhs);
  }

  void rhs_group(int group_id, std::initializer_list<RhsGroupRequest> requests) const {
    require_rate_identity_(group_id);
    if (requests.size() == 0)
      throw std::invalid_argument("ProgramContext RHS group cannot be empty");
    std::vector<int> blocks;
    std::vector<field_type*> states;
    std::vector<field_type*> residuals;
    std::vector<int> flux_only;
    blocks.reserve(requests.size());
    states.reserve(requests.size());
    residuals.reserve(requests.size());
    flux_only.reserve(requests.size());
    std::vector<int> rates;
    rates.reserve(requests.size());
    for (const RhsGroupRequest& request : requests) {
      require_rate_identity_(request.rate_id);
      if (request.rate_id == group_id || request.state == nullptr || request.rhs == nullptr ||
          (request.flux_only != 0 && request.flux_only != 1) ||
          std::find(rates.begin(), rates.end(), request.rate_id) != rates.end())
        throw std::invalid_argument("ProgramContext RHS group contains an invalid request");
      rates.push_back(request.rate_id);
      blocks.push_back(sys_block(request.block));
      states.push_back(request.state);
      residuals.push_back(request.rhs);
      flux_only.push_back(request.flux_only);
    }
    count_kernel_(static_cast<std::int64_t>(requests.size()));
    system_->block_rhs_group(boundary_evaluation_point(group_id), blocks, states, residuals,
                             flux_only);
  }

  void require_cartesian_generated_operator(int program_block,
                                             const std::string& operation) const {
    system_->require_cartesian_generated_operator(sys_block(program_block), operation);
  }

  void apply_projection(int program_block, field_type& state_value) const {
    count_kernel_();
    system_->block_project(sys_block(program_block), state_value);
  }

  Real max_wave_speed(int program_block, const field_type& state_value) const {
    return system_->block_max_speed(sys_block(program_block), state_value);
  }

  Real hmin() const { return system_->cfl_min_dx(); }
  RuntimeParams program_params(int program_block) const {
    return system_->program_params(program_block);
  }

  void axpy(field_type& destination, Real factor, const field_type& source) const {
    count_kernel_();
    pops::saxpy(destination, factor, source);
  }

  void axpy(field_type& destination, Real factor, const field_type& source, Real,
            std::initializer_list<ExactCoefficientTerm>) const {
    axpy(destination, factor, source);
  }

  void lincomb(field_type& destination, Real left_factor, const field_type& left,
               Real right_factor, const field_type& right) const {
    count_kernel_();
    pops::lincomb(destination, left_factor, left, right_factor, right);
  }

  void lincomb(field_type& destination, Real left_factor, const field_type& left,
               Real right_factor, const field_type& right, Real,
               std::initializer_list<ExactCoefficientTerm>,
               std::initializer_list<ExactCoefficientTerm>) const {
    lincomb(destination, left_factor, left, right_factor, right);
  }

  void commit_many(
      std::initializer_list<std::pair<field_type*, const field_type*>> commits) const {
    std::vector<field_type*> targets;
    std::vector<std::optional<field_type>> snapshots;
    targets.reserve(commits.size());
    snapshots.reserve(commits.size());
    for (const auto& [target, source] : commits) {
      if (target == nullptr || source == nullptr)
        throw std::invalid_argument("ProgramContext commit contains null storage");
      if (std::find(targets.begin(), targets.end(), target) != targets.end())
        throw std::invalid_argument("ProgramContext commit contains a duplicate target");
      require_same_field_contract_(*target, *source, "ProgramContext commit");
      targets.push_back(target);
      snapshots.emplace_back(target == source ? std::nullopt
                                               : std::optional<field_type>(*source));
    }
    std::size_t candidate = 0;
    for (const auto& [target, source] : commits) {
      const field_type& value = snapshots[candidate] ? *snapshots[candidate] : *source;
      for (int block = 0; block < system_->n_blocks(); ++block)
        if (target == &system_->block_state(block)) {
          system_->validate_program_state_publication_candidate(block, value);
          break;
        }
      ++candidate;
    }
    candidate = 0;
    for (const auto& [target, source] : commits) {
      if (target != source)
        *target = std::move(*snapshots[candidate]);
      ++candidate;
    }
  }

  void apply_coupling_operators(
      Real dt, std::initializer_list<CouplingStateOverride> candidates) const {
    std::vector<field_type*> runtime_states(static_cast<std::size_t>(system_->n_blocks()), nullptr);
    for (const CouplingStateOverride& candidate : candidates) {
      const int block = sys_block(candidate.program_block);
      if (candidate.state == nullptr || runtime_states[static_cast<std::size_t>(block)] != nullptr)
        throw std::invalid_argument("ProgramContext coupling candidates are incomplete or aliased");
      runtime_states[static_cast<std::size_t>(block)] = candidate.state;
    }
    if (std::find(runtime_states.begin(), runtime_states.end(), nullptr) != runtime_states.end())
      throw std::invalid_argument("ProgramContext coupling requires every runtime block candidate");
    count_kernel_(static_cast<std::int64_t>(system_->apply_coupling_operators(dt, runtime_states)));
  }

  Real sum_component(const field_type& field, int component) const {
    return pops::reduce_sum(field, component);
  }
  Real max_component(const field_type& field, int component) const {
    return pops::reduce_max(field, component);
  }
  Real min_component(const field_type& field, int component) const {
    return pops::reduce_min(field, component);
  }
  Real norm2(int, const field_type& field) const {
    return std::sqrt(pops::dot(field, field, 0));
  }
  Real norm_inf(int, const field_type& field) const {
    return pops::reduce_norm_inf(field, 0);
  }
  Real dot(int, const field_type& left, const field_type& right) const {
    return pops::dot(left, right, 0);
  }

  void fill_boundary(field_type&) const {
    field_type::require_communication("ProgramContext::fill_boundary");
  }
  template <class Lane>
  void fill_boundary(field_type&, const Lane&) const {
    field_type::require_communication("ProgramContext::fill_boundary");
  }

  void register_history(const std::string& name, int lag, int ncomp = -1,
                        int program_owner = -1, const std::string& state_identity = {},
                        const std::string& space_identity = {},
                        const std::string& clock_identity = {},
                        const std::string& interpolation_identity = {}) const {
    if (name.empty() || lag < 1)
      throw std::invalid_argument("ProgramContext history requires a name and positive lag");
    const int owner = program_owner < 0 ? 0 : sys_block(program_owner);
    const field_type& prototype = system_->block_state(owner);
    const int components = ncomp < 0 ? prototype.ncomp() : ncomp;
    if (components < 1)
      throw std::invalid_argument("ProgramContext history component count must be positive");
    auto& history = runtime_state().hist_;
    const int ring_depth = lag + 1;
    const auto existing = history.histories.find(name);
    if (existing != history.histories.end()) {
      if (history.depth.at(name) != ring_depth || history.owner.at(name) != owner ||
          existing->second.front().ncomp() != components ||
          history.state_identity.at(name) != state_identity ||
          history.space_identity.at(name) != space_identity ||
          history.clock_identity.at(name) != clock_identity ||
          history.interpolation_identity.at(name) != interpolation_identity)
        throw std::runtime_error("ProgramContext history identity changed after registration");
      return;
    }
    std::vector<field_type> ring;
    ring.reserve(static_cast<std::size_t>(ring_depth));
    for (int slot = 0; slot < ring_depth; ++slot)
      ring.push_back(make_scratch_(prototype, components, prototype.ghosts()));
    history.histories.emplace(name, std::move(ring));
    history.depth[name] = ring_depth;
    history.initialized[name] = false;
    history.fill_count[name] = 0;
    history.store_pending[name] = false;
    history.owner[name] = owner;
    history.state_identity[name] = state_identity;
    history.space_identity[name] = space_identity;
    history.clock_identity[name] = clock_identity;
    history.interpolation_identity[name] = interpolation_identity;
    history.slot_dt[name] = std::vector<Real>(static_cast<std::size_t>(ring_depth), Real(0));
  }

  field_type& history(const std::string& name, int lag, int ncomp = -1) const {
    auto& manager = runtime_state().hist_;
    auto found = manager.histories.find(name);
    if (found == manager.histories.end() || lag < 0 || lag >= manager.depth.at(name))
      throw std::out_of_range("ProgramContext history slot is absent");
    field_type& result = found->second[static_cast<std::size_t>(lag)];
    if (ncomp >= 0 && result.ncomp() != ncomp)
      throw std::invalid_argument("ProgramContext history component contract differs");
    if (!manager.initialized.at(name))
      throw std::runtime_error("ProgramContext history has not been initialized");
    return result;
  }

  field_type& history_zero_start(const std::string& name, int lag, int ncomp = -1) const {
    auto& manager = runtime_state().hist_;
    auto found = manager.histories.find(name);
    if (found == manager.histories.end() || lag < 0 || lag >= manager.depth.at(name))
      throw std::out_of_range("ProgramContext history slot is absent");
    field_type& result = found->second[static_cast<std::size_t>(lag)];
    if (ncomp >= 0 && result.ncomp() != ncomp)
      throw std::invalid_argument("ProgramContext history component contract differs");
    return result;
  }

  void store_history(const std::string& name, const field_type& value) const {
    store_history_(name, value, std::nullopt);
  }
  void store_history(const std::string& name, const field_type& value, double dt) const {
    if (!std::isfinite(dt) || dt <= 0.0)
      throw std::invalid_argument("ProgramContext history dt must be finite and positive");
    store_history_(name, value, static_cast<Real>(dt));
  }
  void rotate_histories() const { runtime_state().hist_.rotate(); }
  void rotate_histories(const std::string& clock) const { runtime_state().hist_.rotate(clock); }

  bool cache_should_update(int node_id, int every_n) const {
    const bool due = runtime_state().cache_.is_due(node_id, macro_step(), every_n);
    runtime_state().profiler_.count(due ? "cache_misses" : "cache_hits");
    return due;
  }
  void cache_store_aux(int node_id) const {
    runtime_state().cache_.store(node_id, aux(), macro_step());
  }
  void cache_restore_aux(int node_id) const { runtime_state().cache_.restore_into(node_id, aux()); }
  void cache_store_scratch(int node_id, const field_type& scratch) const {
    runtime_state().cache_.store(node_id, scratch, macro_step());
  }
  void cache_restore_scratch(int node_id, field_type& scratch) const {
    runtime_state().cache_.restore_into(node_id, scratch);
  }
  void cache_accumulate_dt(int node_id, Real dt) const {
    runtime_state().cache_.accumulate_dt(node_id, dt);
  }
  Real cache_effective_dt(int node_id, Real dt) const {
    return runtime_state().cache_.effective_dt(node_id, dt);
  }

  bool schedule_domain_occurs(ScheduleDomainKind kind, const std::string& clock,
                              const std::string& stage_identity, int level) const {
    return schedule_coordinate_(kind, clock, stage_identity, level).has_value();
  }
  bool schedule_is_due(int node_id, int every_n, ScheduleDomainKind kind,
                       const std::string& clock, const std::string& stage_identity,
                       int level) const {
    if (node_id < 0 || every_n <= 0)
      throw std::invalid_argument("ProgramContext schedule has an invalid node or period");
    const auto coordinate = schedule_coordinate_(kind, clock, stage_identity, level);
    return coordinate && coordinate->value % every_n == 0;
  }
  bool schedule_at_start(ScheduleDomainKind kind, const std::string& clock,
                         const std::string& stage_identity, int level) const {
    const auto coordinate = schedule_coordinate_(kind, clock, stage_identity, level);
    return coordinate && coordinate->value == 0;
  }
  bool schedule_decision(int node_id, bool due, bool cache_backed) const {
    if (node_id < 0)
      throw std::invalid_argument("ProgramContext schedule decision has an invalid node");
    return runtime_state().profiler_.schedule_decision(due, cache_backed);
  }
  ClockScheduleState::SubcycleScope subcycle_scope(const std::string& parent,
                                                   const std::string& child, int count) const {
    return clock_schedule_.subcycle(parent, child, count);
  }
  void synchronize_sample_and_hold(const std::string& source, const std::string& target,
                                   int step, Real offset) const {
    clock_schedule_.synchronize_sample_and_hold(source, target, step,
                                                static_cast<double>(offset));
  }

  int macro_step() const { return system_->macro_step(); }
  Real physical_time() const { return static_cast<Real>(system_->time()); }

  void record_scalar(const std::string& name, Real value) const {
    runtime_state().record_diagnostic(name, value);
  }
  void record_balance_term(const std::string& route, const std::string& term, Real value) const {
    system_->record_program_balance_term(route, term, value);
  }
  bool balance_consumer_is_due(const std::string& contract, const std::string& route,
                               int every_n) const {
    return system_->program_balance_consumer_is_due(contract, route, every_n);
  }
  void note_automatic_balance_capture_due(bool due) const {
    runtime_state().note_automatic_balance_capture_due(due, "ProgramContext");
  }
  void note_step_projection(const std::string& name) const {
    runtime_state().note_step_projection(name);
  }

  void profile_record(const std::string& name,
                      std::chrono::steady_clock::time_point start) const {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    runtime_state().profiler_.record(name, std::chrono::duration<double>(elapsed).count());
  }

  runtime_state_type& runtime_state() const { return system_->program_runtime_state_(); }

  [[noreturn]] SolveOutcome solve_fields() const { unavailable_field_provider_(); }
  [[noreturn]] SolveOutcome solve_fields_from_state(int, field_type&) const {
    unavailable_field_provider_();
  }
  [[noreturn]] SolveOutcome solve_fields_from_state_at(
      const runtime::multiblock::BoundaryEvaluationPoint&, const std::string&, int,
      field_type&) const {
    unavailable_field_provider_();
  }
  [[noreturn]] SolveOutcome solve_fields_from_blocks(
      const std::vector<const field_type*>&) const {
    unavailable_field_provider_();
  }
  [[noreturn]] SolveOutcome solve_fields_from_blocks_at(
      const runtime::multiblock::BoundaryEvaluationPoint&, std::int64_t, std::string_view,
      std::initializer_list<FieldStageOverride>) const {
    unavailable_field_provider_();
  }

  bool is_polar_geometry() const noexcept { return false; }
  [[noreturn]] Real radial_origin() const { unavailable_polar_provider_(); }
  [[noreturn]] Real radial_spacing() const { unavailable_polar_provider_(); }

 private:
  enum class ScratchKind : std::uint8_t { Rhs = 0, State = 1, Scalar = 2 };
  using ScratchKey = std::tuple<ScratchKind, std::int64_t, int>;

  static void require_rate_identity_(int rate_id) {
    if (rate_id < 0)
      throw std::invalid_argument("ProgramContext rate identity must be non-negative");
  }

  static void require_same_field_contract_(const field_type& left, const field_type& right,
                                           const char* operation) {
    if (left.layout() != right.layout() || left.distribution() != right.distribution() ||
        left.local_rank() != right.local_rank() || left.ncomp() != right.ncomp() ||
        left.ghosts() != right.ghosts())
      throw std::invalid_argument(std::string(operation) +
                                  " requires the same exact ranked field contract");
  }

  static field_type make_scratch_(const field_type& prototype, int ncomp,
                                  const Extent<Dim>& ghosts) {
    field_type result(prototype.layout(), prototype.distribution(), prototype.local_rank(), ncomp,
                      ghosts);
    result.set_val(Real(0));
    return result;
  }

  static field_type scalar_field_like_(const field_type& prototype, int ncomp,
                                       int ghost_depth) {
    if (ncomp < 1 || ghost_depth < 0)
      throw std::invalid_argument(
          "ProgramContext scalar field requires positive components and non-negative ghosts");
    Extent<Dim> ghosts{};
    for (int axis = 0; axis < Dim; ++axis)
      ghosts[axis] = ghost_depth;
    return make_scratch_(prototype, ncomp, ghosts);
  }

  field_type& persistent_scratch_(ScratchKind kind, std::int64_t value_id, int subslot,
                                  const field_type& prototype, int ncomp,
                                  const Extent<Dim>& ghosts) const {
    if (value_id < 0 || subslot < 0)
      throw std::invalid_argument("ProgramContext scratch identity must be non-negative");
    const ScratchKey key{kind, value_id, subslot};
    auto [entry, inserted] = scratch_.try_emplace(key);
    field_type& result = entry->second;
    if (inserted || result.layout() != prototype.layout() ||
        result.distribution() != prototype.distribution() ||
        result.local_rank() != prototype.local_rank() || result.ncomp() != ncomp ||
        result.ghosts() != ghosts)
      result = make_scratch_(prototype, ncomp, ghosts);
    else
      result.set_val(Real(0));
    return result;
  }

  void store_history_(const std::string& name, const field_type& value,
                      std::optional<Real> dt) const {
    auto& manager = runtime_state().hist_;
    auto found = manager.histories.find(name);
    if (found == manager.histories.end())
      throw std::out_of_range("ProgramContext history is not registered");
    require_same_field_contract_(found->second.front(), value, "ProgramContext history store");
    found->second.front() = value;
    if (!manager.initialized.at(name))
      for (std::size_t slot = 1; slot < found->second.size(); ++slot)
        found->second[slot] = value;
    manager.initialized[name] = true;
    manager.store_pending[name] = true;
    if (dt)
      manager.slot_dt[name][0] = *dt;
  }

  std::optional<ScheduleCoordinate> schedule_coordinate_(
      ScheduleDomainKind kind, const std::string& clock, const std::string& stage_identity,
      int level) const {
    return clock_schedule_.coordinate(kind, clock, stage_identity, level, 0,
                                      static_cast<std::int64_t>(macro_step()));
  }

  void count_kernel_(std::int64_t count = 1) const {
    runtime_state().profiler_.count("kernels", count);
  }

  [[noreturn]] static void unavailable_field_provider_() {
    throw std::runtime_error(
        "ProgramContext field solve requires a dimension-qualified prepared field provider");
  }
  [[noreturn]] static void unavailable_polar_provider_() {
    throw std::runtime_error(
        "ProgramContext polar operation is outside the exact Cartesian ranked core");
  }

  runtime_type* system_ = nullptr;
  mutable double current_dt_ = 0.0;
  mutable amr::Rational stage_time_{0, 1};
  mutable std::string primary_clock_;
  mutable ClockScheduleState clock_schedule_;
  mutable std::map<ScratchKey, field_type> scratch_;
};

template <int Dim>
std::shared_ptr<ProgramContext<Dim>> make_program_execution_provider(System<Dim>* system) {
  return std::make_shared<ProgramContext<Dim>>(system);
}

template <int Dim>
ProgramContext<Dim> make_program_execution_view(System<Dim>* system) {
  return ProgramContext<Dim>(system);
}

}  // namespace pops::runtime::program
