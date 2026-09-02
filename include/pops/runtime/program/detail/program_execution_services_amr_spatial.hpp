runtime_type& runtime() const noexcept {
  return *runtime_;
}
hierarchy_type& hierarchy() const noexcept {
  return runtime_->hierarchy();
}
/// Borrow the runtime-prepared AMR hierarchy lane; generated execution never materializes a
/// second communicator or falls back to the process world.
[[nodiscard]] const ExecutionLane& prepared_execution_lane() const {
  if (preparation_view_ != nullptr)
    return *preparation_view_->lane;
  require_facade_execution_();
  return facade_->program_prepared_amr_execution_lane_();
}
/// Borrow the exact runtime lane as the parent authority for generated private Krylov lanes.
/// In MPI builds the returned communicator never owns or substitutes WORLD; it only survives the
/// immediate collective duplication performed by PreparedAffineLinearProblem/KrylovWorkspace.
/// (Serial builds have no native communicator to substitute.) Each
/// installed solve pays for exactly those two persistent duplicates: the problem lane owns
/// operator/control traffic (including this tensor session), while the workspace lane owns
/// Krylov reductions. Their generated shared owners retire both lanes with the installed solve.
[[nodiscard]] ExecutionCommunicator prepared_execution_communicator() const {
  const ExecutionLane& lane = prepared_execution_lane();
#ifdef POPS_HAS_MPI
  return ExecutionCommunicator::borrowed(lane.identity(), lane.native_handle());
#else
  return ExecutionCommunicator::world();
#endif
}
const ::pops::amr::hierarchy::LevelLayout<Dim>& layout(std::size_t selected) const {
  return runtime_->hierarchy().layout(selected);
}
field_type& state(std::size_t selected) const {
  if (!active_attempt_states_.empty()) {
    if (selected != static_cast<std::size_t>(active_level_) ||
        active_attempt_states_.front() == nullptr)
      throw std::out_of_range("AMR Program attempt state lies outside the prepared hierarchy");
    return *active_attempt_states_.front();
  }
  return runtime_->hierarchy().state(selected);
}

::pops::runtime::amr::PreparedTaggerCandidates<Dim> execute_prepared_tagging(
    int parent_level) const {
  if (facade_ == nullptr)
    throw std::logic_error(
        "AMR Program tagging execution requires the exact-ranked facade authority");
  return facade_->execute_prepared_tagging(parent_level);
}

bool regrid_from_prepared_tagging(int parent_level) const {
  if (facade_ == nullptr)
    throw std::logic_error(
        "AMR Program regrid publication requires the exact-ranked facade authority");
  require_history_free_for_topology_change_("prepared tagging regrid");
  return facade_->regrid_from_prepared_tagging(parent_level);
}

void begin_restart_regrid_history_sequence() const {
  require_facade_execution_();
  facade_->begin_restart_regrid_history_sequence();
}

void end_restart_regrid_history_sequence() const noexcept {
  facade_->end_restart_regrid_history_sequence();
}

ProgramSpatialSnapshot<Dim> spatial_snapshot() const {
  return {std::string(runtime_->spatial_contract()), runtime_->topology_epoch(),
          runtime_->materialization_generation()};
}

void require_live(const ProgramSpatialSnapshot<Dim>& snapshot) const {
  if (snapshot.spatial_contract != runtime_->spatial_contract() ||
      snapshot.topology_epoch != runtime_->topology_epoch() ||
      snapshot.materialization_generation != runtime_->materialization_generation())
    throw std::invalid_argument("AMR Program spatial snapshot is stale");
}

subcycle_plan_type prepare_subcycling(
    std::span<const int> temporal_substeps,
    ::pops::numerics::time::amr::AmrSubcyclePreparationBudget budget) const {
  return subcycle_plan_type::prepare(*runtime_, temporal_substeps, budget);
}

::pops::amr::regridding::PreparedRegrid<Dim> prepare_regrid(
    std::size_t parent_level, ::pops::amr::RefinementRatio<Dim> ratio,
    ::pops::amr::tagging::ClusterResult<Dim> clustered,
    ::pops::amr::regridding::RegridPreparationBudget preparation_budget) const {
  const ExecutionLane& lane = prepared_execution_lane();
  return runtime_->prepare_regrid(parent_level, ratio, std::move(clustered), preparation_budget,
                                  lane);
}

void publish_regrid(::pops::amr::regridding::PreparedRegrid<Dim> prepared,
                    std::optional<field_type> child_state) const {
  require_facade_execution_();
  require_history_free_for_topology_change_("regrid");
  const int parent_level = prepared.source_level().level;
  if (parent_level < 0)
    throw std::invalid_argument("AMR Program regrid has no source level");
  runtime_->publish_regrid(static_cast<std::size_t>(parent_level), std::move(prepared),
                           std::move(child_state));
}

PreparedRebalanceDecision<Dim> prepare_rebalance(
    std::size_t selected, ResourceEstimates estimates,
    parallel::LoadBalancePreparationBudget preparation_budget,
    const RebalancePolicy& policy) const {
  const ExecutionLane& lane = prepared_execution_lane();
  return runtime_->prepare_rebalance(selected, estimates, preparation_budget, policy, lane);
}

PreparedRebalanceDecision<Dim> prepare_rebalance(
    std::size_t selected, ResourceEstimates estimates,
    parallel::LoadBalancePreparationBudget preparation_budget) const {
  const ExecutionLane& lane = prepared_execution_lane();
  return runtime_->prepare_rebalance(selected, estimates, preparation_budget, lane);
}

void apply_rebalance(std::size_t selected, PreparedRebalanceDecision<Dim> decision,
                     field_type remapped_state) const {
  require_facade_execution_();
  require_history_free_for_topology_change_("rebalance");
  runtime_->apply_rebalance(selected, std::move(decision), std::move(remapped_state));
}

template <class Payload, class Axpy>
::pops::amr::reflux::MetricFaceReflux<Payload> reconcile_reflux(
    const ::pops::amr::reflux::TransactionalFaceFluxLedger<Dim, Payload>& ledger,
    const ::pops::amr::reflux::CoarseFaceRefluxKey<Dim>& key, std::string_view state_identity,
    const ::pops::amr::reflux::FaceRefinementMapping<Dim>& mapping,
    const ::pops::amr::reflux::MetricRefluxBudget& budget, Axpy&& axpy) const {
  return runtime_->reconcile_reflux(ledger, key, state_identity, mapping, budget,
                                    std::forward<Axpy>(axpy));
}

/// Execute the provider-owned accepted hierarchy refresh without installing or replacing any
/// ProgramRuntimeState callback.  A generated artifact may first republish its own accepted image;
/// the provider then requalifies its exact-ranked resources against the current facade hierarchy.
void refresh_accepted_hierarchy(const std::function<void()>& artifact_refresh = {}) const {
  if (artifact_refresh)
    artifact_refresh();
  refresh_accepted_hierarchy_state_();
}

/// Consume one engine-authenticated history remap through this provider only.  The descriptor is
/// passed by the host at the accepted publication boundary and is never retained by the provider.
void accept_history_remap(const AmrProgramHistoryRemapDescriptor& descriptor) const {
  refresh_accepted_hierarchy_state_after_remap_(descriptor);
}

/// Restart lifecycle actions exposed for v5 candidate callbacks.  These methods deliberately do
/// not publish ProgramRuntimeState hooks; they execute the existing provider-owned operation only.
void preflight_restart_regrid() const {
  preflight_restart_regrid_();
}
void restart_regrid() const {
  restart_regrid_();
}
void resync_after_restart() const {
  resync_after_restart_();
}

[[nodiscard]] std::unique_ptr<AcceptedProgramExecutionServicesSnapshot>
create_accepted_context_snapshot() const {
  return capture_accepted_context_snapshot_();
}

std::unique_ptr<AcceptedProgramExecutionServicesSnapshot> accepted_context_snapshot() const {
  return create_accepted_context_snapshot();
}

void begin_step(double dt) const {
  require_facade_execution_();
  if (!std::isfinite(dt) || !(dt > 0.0))
    throw std::invalid_argument("AMR Program step requires a finite positive dt");
  current_dt_ = dt;
  current_interval_start_time_ = facade_->program_time_();
  current_interval_begin_phase_ = {0, 1};
  current_interval_end_phase_ = {1, 1};
  stage_time_ = ::pops::amr::Rational(0, 1);
  logical_substep_ = 0;
}

void configure_primary_clock(const std::string& clock) const {
  clock_schedule_.configure_primary_clock(clock);
  primary_clock_ = clock;
}

/// Move the clock declarations collected on the detached preparation image into the accepted
/// backend before the first callback.  This is an activation-only transfer: generated steps keep
/// using the already-bound schedule and never allocate or mutate the accepted facade to configure
/// it lazily.
void adopt_prepared_clock(ClockScheduleState schedule, std::string primary_clock) const {
  if (primary_clock.empty())
    throw std::invalid_argument("AMR Program prepared clock has no primary identity");
  schedule.seal_for_execution();
  clock_schedule_ = std::move(schedule);
  primary_clock_ = std::move(primary_clock);
  // The generic RHS packs are populated only during cold preparation.  Keep their clock storage
  // resident so a bound Program never constructs/copies a BoundaryEvaluationPoint string.
  hot_path_workspace_.bind_boundary_point_clock(primary_clock_);
}

[[nodiscard]] const ClockScheduleState& accepted_clock_schedule() const noexcept {
  return clock_schedule_;
}

void declare_clock_relation(const std::string& parent, const std::string& child, int count) const {
  clock_schedule_.declare_relation(parent, child, count);
}

void set_stage_time(std::int64_t numerator, std::int64_t denominator) const {
  if (denominator <= 0 || numerator < 0 || numerator > denominator)
    throw std::invalid_argument("AMR Program stage time is outside [0, 1]");
  stage_time_ = ::pops::amr::Rational(numerator, denominator);
}

runtime::multiblock::BoundaryEvaluationPoint boundary_evaluation_point(int stage) const {
  require_rate_identity_(stage);
  require_facade_execution_();
  if (primary_clock_.empty() || !std::isfinite(current_dt_) || !(current_dt_ > 0.0))
    throw std::logic_error("AMR Program evaluation point lacks an active clock interval");
  return {.clock = primary_clock_,
          .tick = static_cast<std::int64_t>(facade_->program_macro_step_()),
          .level = active_level_,
          .substep = logical_substep_,
          .stage = stage,
          .stage_fraction = stage_time_,
          .dt = current_dt_,
          .physical_time = current_interval_start_time_ + stage_time_.value() * current_dt_};
}

/// Cold preparation for a point retained by a generated matrix-free session.  Only the fixed
/// clock string can require heap storage; exact stage coordinates are populated later through the
/// write-into seam below.
void prepare_boundary_evaluation_point(runtime::multiblock::BoundaryEvaluationPoint& point) const {
  if (primary_clock_.empty())
    throw std::logic_error("AMR Program boundary point preparation has no primary clock");
  if (!point.clock.empty() && point.clock != primary_clock_)
    throw std::logic_error("AMR Program boundary point preparation changed its clock");
  if (point.clock.capacity() < primary_clock_.size())
    point.clock.reserve(primary_clock_.size());
  point.clock.assign(primary_clock_);
  point.graph_identity.clear();
  point.rate_identity.clear();
  point.application_identity.clear();
}

void prepare_boundary_evaluation_point(
    runtime::multiblock::BoundaryEvaluationPoint& destination,
    const runtime::multiblock::BoundaryEvaluationPoint& capacity_source) const {
  if (primary_clock_.empty() || capacity_source.clock != primary_clock_)
    throw std::logic_error("AMR Program boundary point clone has a different prepared clock");
  const auto prepare_string = [](std::string& target, const std::string& source) {
    if (target.capacity() < source.capacity())
      target.reserve(source.capacity());
    if (target.capacity() < source.size())
      throw std::logic_error("AMR Program boundary point clone capacity is incomplete");
    target.assign(source);
  };
  prepare_string(destination.clock, capacity_source.clock);
  prepare_string(destination.graph_identity, capacity_source.graph_identity);
  prepare_string(destination.rate_identity, capacity_source.rate_identity);
  prepare_string(destination.application_identity, capacity_source.application_identity);
}

void write_boundary_evaluation_point_into(runtime::multiblock::BoundaryEvaluationPoint& point,
                                          int stage) const {
  require_rate_identity_(stage);
  require_facade_execution_();
  if (primary_clock_.empty() || !std::isfinite(current_dt_) || !(current_dt_ > 0.0) ||
      point.clock != primary_clock_)
    throw std::logic_error("AMR Program resident evaluation point lacks its prepared clock");
  point.tick = static_cast<std::int64_t>(facade_->program_macro_step_());
  point.level = active_level_;
  point.substep = logical_substep_;
  point.stage = stage;
  point.stage_fraction = stage_time_;
  point.dt = current_dt_;
  point.physical_time = current_interval_start_time_ + stage_time_.value() * current_dt_;
  // Ordinary RHS evaluation carries no authored coupling identities.  They were reserved only
  // for the dedicated coupling carrier and must not be assigned on this path.
  point.graph_identity.clear();
  point.rate_identity.clear();
  point.application_identity.clear();
}

[[nodiscard]] const runtime::multiblock::BoundaryEvaluationPoint&
prepared_boundary_evaluation_point(int stage) const {
  auto& point = hot_path_workspace_.direct_point;
  write_boundary_evaluation_point_into(point, stage);
  return point;
}

void copy_boundary_evaluation_point_into(
    runtime::multiblock::BoundaryEvaluationPoint& destination,
    const runtime::multiblock::BoundaryEvaluationPoint& source) const {
  require_facade_execution_();
  if (source.clock.empty() || source.tick < 0 || source.level != active_level_ ||
      source.substep < 0 || source.stage < 0 || source.stage_fraction.denominator <= 0 ||
      source.stage_fraction.numerator < 0 ||
      source.stage_fraction.numerator > source.stage_fraction.denominator ||
      !std::isfinite(source.dt) || !(source.dt > 0.0) || !std::isfinite(source.physical_time))
    throw std::invalid_argument("AMR Program boundary point copy has an invalid source");
  const auto require_capacity = [](const std::string& target, std::string_view value) {
    if (target.capacity() < value.size())
      throw std::logic_error("AMR Program boundary point copy exceeds its prepared capacity");
  };
  require_capacity(destination.clock, source.clock);
  require_capacity(destination.graph_identity, source.graph_identity);
  require_capacity(destination.rate_identity, source.rate_identity);
  require_capacity(destination.application_identity, source.application_identity);
  destination.clock.assign(source.clock);
  destination.tick = source.tick;
  destination.level = source.level;
  destination.substep = source.substep;
  destination.stage = source.stage;
  destination.stage_fraction = source.stage_fraction;
  destination.dt = source.dt;
  destination.physical_time = source.physical_time;
  destination.graph_identity.assign(source.graph_identity);
  destination.rate_identity.assign(source.rate_identity);
  destination.application_identity.assign(source.application_identity);
}

template <class Body>
void advance_hierarchy(double dt, Body&& body) const {
  advance_prepared_hierarchy_(dt, std::forward<Body>(body), "advance_hierarchy");
}

template <class Body>
void advance_synchronized_hierarchy(double dt, Body&& body) const {
  advance_prepared_hierarchy_(dt, std::forward<Body>(body), "advance_synchronized_hierarchy");
}

[[nodiscard]] std::optional<PreparedCellTemporalExecution<Dim>>
prepare_same_level_cell_temporal_execution(
    std::string clock, std::int64_t tick_denominator, int rung,
    std::span<const SameLevelCellTemporalForwardEulerRoute> routes) const {
  return prepare_same_level_cell_temporal_execution_(std::move(clock), tick_denominator, rung,
                                                     routes);
}
void advance_same_level_cell_temporal(double dt) const {
  advance_same_level_cell_temporal_(dt);
}

bool uses_prepared_krylov_fallback() const {
  return configured_hierarchy_tensor_solver_().execution_path() ==
         HierarchyTensorSolverExecutionPath::PreparedKrylovFallback;
}
int nlev() const {
  if (preparation_view_ != nullptr) {
    // A forward bootstrap/regrid image deliberately has no AmrRuntime.  Its copied level
    // geometry is the sole topology authority until the aggregate publication adopts the
    // forward owner.  Validate the complete immutable view before exposing that bounded count;
    // falling through to runtime_ here would dereference the canonical null forward runtime.
    preparation_view_->validate();
    const std::size_t levels = preparation_view_->level_geometries.size();
    if (levels > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("AMR Program resource level count exceeds int");
    return static_cast<int>(levels);
  }
  if (runtime_ == nullptr)
    throw std::logic_error("AMR Program accepted topology has no runtime authority");
  return static_cast<int>(runtime_->hierarchy().num_levels());
}
int level() const noexcept {
  return active_level_;
}

ProgramResourceTopology program_resource_topology() const {
  if (preparation_view_ != nullptr) {
    const int levels = nlev();
    return {levels, preparation_view_->topology_epoch,
            preparation_view_->materialization_generation};
  }
  refresh_resources_();
  if (runtime_ == nullptr)
    throw std::logic_error("AMR Program accepted topology refresh lost its runtime authority");
  return {nlev(), runtime_->topology_epoch(), runtime_->materialization_generation()};
}

template <class Function>
void for_each_program_resource_level(Function&& function) const {
  if (preparation_view_ == nullptr)
    refresh_resources_();
  const int levels = nlev();
  const int prior = active_level_;
  try {
    for (int selected = 0; selected < levels; ++selected) {
      active_level_ = selected;
      function(selected);
    }
    active_level_ = prior;
  } catch (...) {
    active_level_ = prior;
    throw;
  }
}

template <class Function>
decltype(auto) with_program_resource_level(int selected, Function&& function) const {
  if (selected < 0 || selected >= nlev())
    throw std::out_of_range("AMR Program resource level lies outside the live hierarchy");
  const int prior = active_level_;
  active_level_ = selected;
  try {
    if constexpr (std::is_void_v<std::invoke_result_t<Function>>) {
      std::forward<Function>(function)();
      active_level_ = prior;
    } else {
      decltype(auto) result = std::forward<Function>(function)();
      active_level_ = prior;
      return result;
    }
  } catch (...) {
    active_level_ = prior;
    throw;
  }
}

int n_blocks() const {
  if (preparation_view_ != nullptr)
    return static_cast<int>(preparation_view_->block_prototypes.size());
  require_facade_execution_();
  return facade_->program_n_blocks_();
}

int sys_block(int program_block) const {
  if (preparation_view_ != nullptr) {
    const auto& map = preparation_view_->program_block_map;
    if (program_block < 0 || static_cast<std::size_t>(program_block) >= map.size())
      throw std::out_of_range("AMR Program block has no detached preparation mapping");
    return map[static_cast<std::size_t>(program_block)];
  }
  require_facade_execution_();
  const auto& map = facade_->program_block_map_();
  if (program_block < 0 || static_cast<std::size_t>(program_block) >= map.size())
    throw std::out_of_range("AMR Program block " + std::to_string(program_block) +
                            " has no authenticated runtime mapping in a map of size " +
                            std::to_string(map.size()));
  const int selected = map[static_cast<std::size_t>(program_block)];
  if (selected < 0 || selected >= facade_->program_n_blocks_())
    throw std::runtime_error("AMR Program block mapping targets no runtime block");
  return selected;
}

field_type& state(int program_block) const {
  if (preparation_view_ != nullptr) {
    const int runtime_block = sys_block(program_block);
    if (active_level_ < 0 ||
        static_cast<std::size_t>(active_level_) >=
            preparation_view_->block_prototypes.at(static_cast<std::size_t>(runtime_block)).size())
      throw std::out_of_range("AMR Program detached preparation level is outside the topology");
    return const_cast<field_type&>(
        preparation_view_->block_prototypes.at(static_cast<std::size_t>(runtime_block))
            .at(static_cast<std::size_t>(active_level_)));
  }
  refresh_resources_();
  const int runtime_block = sys_block(program_block);
  if (field_type* attempt = live_attempt_state_(runtime_block, active_level_))
    return *attempt;
  return facade_->program_prepared_amr_block_state_(runtime_block, active_level_);
}
