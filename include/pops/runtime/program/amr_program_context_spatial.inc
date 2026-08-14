/// Spatial-only constructor used by preparation tests.  Execution methods require a facade.
explicit AmrProgramContext(runtime_type& runtime) : runtime_(&runtime) {
  synchronize_resource_generation_();
}

runtime_type& runtime() const noexcept {
  return *runtime_;
}
hierarchy_type& hierarchy() const noexcept {
  return runtime_->hierarchy();
}
/// Borrow the runtime-prepared AMR hierarchy lane; generated execution never materializes a
/// second communicator or falls back to the process world.
[[nodiscard]] const ExecutionLane& prepared_execution_lane() const {
  require_facade_execution_();
  return facade_->prepared_amr_multiblock_hierarchy_().lane();
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

void install(std::function<void(double)> step) const {
  require_facade_execution_();
  if (!step)
    throw std::invalid_argument("AMR Program install requires a non-empty step");
  facade_->install_program_step(std::move(step));
}

void install(std::function<void(double)> step, std::shared_ptr<AmrProgramContext> keep_alive,
             std::function<void()> hierarchy_refresh = {}) const {
  require_facade_execution_();
  if (!step || !keep_alive || keep_alive.get() != this)
    throw std::invalid_argument("AMR Program install requires its exact owning context");
  facade_->install_program_step([step = std::move(step), keep_alive](double dt) { step(dt); });
  if (hierarchy_refresh)
    facade_->install_program_hierarchy_refresh(
        [owner = keep_alive, refresh = std::move(hierarchy_refresh)]() mutable {
          refresh();
          owner->refresh_accepted_hierarchy_state_();
        });
  else
    facade_->install_program_hierarchy_refresh(
        [owner = keep_alive]() { owner->refresh_accepted_hierarchy_state_(); });
  facade_->install_program_restart_hooks(
      [owner = keep_alive]() { owner->preflight_restart_regrid_(); },
      [owner = keep_alive]() { owner->restart_regrid_(); },
      [owner = keep_alive]() { owner->resync_after_restart_(); },
      [owner = keep_alive]() { return owner->accepted_context_snapshot(); });
}

std::unique_ptr<AcceptedProgramContextSnapshot> accepted_context_snapshot() const {
  return capture_accepted_context_snapshot_();
}

void begin_step(double dt) const {
  require_facade_execution_();
  if (!std::isfinite(dt) || !(dt > 0.0))
    throw std::invalid_argument("AMR Program step requires a finite positive dt");
  current_dt_ = dt;
  current_interval_start_time_ = facade_->time();
  current_interval_begin_phase_ = {0, 1};
  current_interval_end_phase_ = {1, 1};
  stage_time_ = ::pops::amr::Rational(0, 1);
  logical_substep_ = 0;
}

void configure_primary_clock(const std::string& clock) const {
  clock_schedule_.configure_primary_clock(clock);
  primary_clock_ = clock;
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
          .tick = static_cast<std::int64_t>(facade_->macro_step()),
          .level = active_level_,
          .substep = logical_substep_,
          .stage = stage,
          .stage_fraction = stage_time_,
          .dt = current_dt_,
          .physical_time = current_interval_start_time_ + stage_time_.value() * current_dt_};
}

template <class Body>
void advance_hierarchy(double dt, Body&& body) const {
  advance_prepared_hierarchy_(dt, std::forward<Body>(body), "advance_hierarchy");
}

template <class Body>
void advance_synchronized_hierarchy(double dt, Body&& body) const {
  advance_prepared_hierarchy_(dt, std::forward<Body>(body), "advance_synchronized_hierarchy");
}

void prepare_same_level_cell_temporal_execution(
    std::string clock, std::int64_t tick_denominator, int rung,
    std::span<const SameLevelCellTemporalForwardEulerRoute> routes) const {
  prepare_same_level_cell_temporal_execution_(std::move(clock), tick_denominator, rung, routes);
}
void advance_same_level_cell_temporal(double dt) const {
  advance_same_level_cell_temporal_(dt);
}

bool uses_prepared_krylov_fallback() const {
  return configured_hierarchy_tensor_solver_().execution_path() ==
         HierarchyTensorSolverExecutionPath::PreparedKrylovFallback;
}
int nlev() const {
  return static_cast<int>(runtime_->hierarchy().num_levels());
}
int level() const noexcept {
  return active_level_;
}

ProgramResourceTopology program_resource_topology() const {
  refresh_resources_();
  return {nlev(), runtime_->topology_epoch(), runtime_->materialization_generation()};
}

template <class Function>
void for_each_program_resource_level(Function&& function) const {
  refresh_resources_();
  const int prior = active_level_;
  try {
    for (int selected = 0; selected < nlev(); ++selected) {
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
  require_facade_execution_();
  return facade_->n_blocks();
}

int sys_block(int program_block) const {
  require_facade_execution_();
  const auto& map = facade_->program_block_map();
  if (program_block < 0 || static_cast<std::size_t>(program_block) >= map.size())
    throw std::out_of_range("AMR Program block has no authenticated runtime mapping");
  const int selected = map[static_cast<std::size_t>(program_block)];
  if (selected < 0 || selected >= facade_->n_blocks())
    throw std::runtime_error("AMR Program block mapping targets no runtime block");
  return selected;
}

field_type& state(int program_block) const {
  refresh_resources_();
  const int runtime_block = sys_block(program_block);
  if (!active_attempt_states_.empty())
    return *active_attempt_states_.at(static_cast<std::size_t>(runtime_block));
  return facade_->prepared_amr_block_state(runtime_block, active_level_);
}
