
class AcceptedContextSnapshot final : public AcceptedProgramContextSnapshot {
 public:
  explicit AcceptedContextSnapshot(AmrProgramContext& owner)
      : owner_(&owner),
        clock_schedule_(owner.clock_schedule_),
        resource_epoch_(owner.resource_epoch_),
        resource_generation_(owner.resource_generation_),
        history_epoch_(owner.history_epoch_),
        history_generation_(owner.history_generation_),
        history_levels_(owner.history_levels_),
        history_flux_expressions_(owner.history_flux_expressions_),
        pending_history_remaps_(owner.pending_history_remaps_),
        deferred_history_lag_scratches_(owner.deferred_history_lag_scratches_),
        accepted_temporal_partition_(owner.accepted_temporal_partition_),
        cell_temporal_configuration_(owner.cell_temporal_configuration_),
        accepted_flux_budget_contract_(owner.accepted_flux_budget_contract_),
        accepted_coupling_contract_(owner.accepted_coupling_contract_),
        accepted_face_flux_(owner.accepted_face_flux_),
        interface_flux_ledger_(
            std::make_unique<interface_flux_ledger_type>(*owner.interface_flux_ledger_)),
        accepted_synchronization_events_(owner.accepted_synchronization_events_),
        accepted_state_revision_(owner.accepted_state_revision_) {}

  AcceptedContextSnapshot(const AcceptedContextSnapshot& accepted)
      : owner_(accepted.owner_),
        clock_schedule_(accepted.clock_schedule_),
        resource_epoch_(accepted.resource_epoch_),
        resource_generation_(accepted.resource_generation_),
        history_epoch_(accepted.history_epoch_),
        history_generation_(accepted.history_generation_),
        history_levels_(accepted.history_levels_),
        history_flux_expressions_(accepted.history_flux_expressions_),
        pending_history_remaps_(accepted.pending_history_remaps_),
        deferred_history_lag_scratches_(accepted.deferred_history_lag_scratches_),
        accepted_temporal_partition_(accepted.accepted_temporal_partition_),
        cell_temporal_configuration_(accepted.cell_temporal_configuration_),
        accepted_flux_budget_contract_(accepted.accepted_flux_budget_contract_),
        accepted_coupling_contract_(accepted.accepted_coupling_contract_),
        accepted_face_flux_(accepted.accepted_face_flux_),
        interface_flux_ledger_(
            std::make_unique<interface_flux_ledger_type>(*accepted.interface_flux_ledger_)),
        accepted_synchronization_events_(accepted.accepted_synchronization_events_),
        accepted_state_revision_(accepted.accepted_state_revision_) {}

  std::unique_ptr<AcceptedProgramContextSnapshot> prepare_restore() const override {
    return std::make_unique<AcceptedContextSnapshot>(*this);
  }

  void publish_restore() noexcept override {
    static_assert(std::is_nothrow_swappable_v<ClockScheduleState>);
    static_assert(std::is_nothrow_swappable_v<decltype(history_levels_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(history_flux_expressions_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(pending_history_remaps_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(deferred_history_lag_scratches_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(accepted_temporal_partition_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(cell_temporal_configuration_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(accepted_flux_budget_contract_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(accepted_coupling_contract_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(accepted_face_flux_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(interface_flux_ledger_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(accepted_synchronization_events_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(discarded_scratches_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(discarded_subcycling_contract_)>);
    std::swap(owner_->clock_schedule_, clock_schedule_);
    owner_->resource_epoch_ = resource_epoch_;
    owner_->resource_generation_ = resource_generation_;
    owner_->history_epoch_ = history_epoch_;
    owner_->history_generation_ = history_generation_;
    owner_->history_levels_.swap(history_levels_);
    owner_->history_flux_expressions_.swap(history_flux_expressions_);
    owner_->pending_history_remaps_.swap(pending_history_remaps_);
    owner_->deferred_history_lag_scratches_.swap(deferred_history_lag_scratches_);
    owner_->scratches_.swap(discarded_scratches_);
    owner_->hierarchy_tensor_solver_.reset();
    owner_->hierarchy_tensor_topology_epoch_ = std::numeric_limits<std::uint64_t>::max();
    owner_->hierarchy_tensor_materialization_generation_ =
        std::numeric_limits<std::uint64_t>::max();
    owner_->multiblock_subcycling_.reset();
    owner_->multiblock_subcycling_epoch_ = std::numeric_limits<std::uint64_t>::max();
    owner_->multiblock_subcycling_generation_ = std::numeric_limits<std::uint64_t>::max();
    owner_->multiblock_subcycling_program_budget_contract_.swap(discarded_subcycling_contract_);
    std::swap(owner_->accepted_temporal_partition_, accepted_temporal_partition_);
    std::swap(owner_->cell_temporal_configuration_, cell_temporal_configuration_);
    owner_->accepted_flux_budget_contract_.swap(accepted_flux_budget_contract_);
    owner_->accepted_coupling_contract_.swap(accepted_coupling_contract_);
    std::swap(owner_->accepted_face_flux_, accepted_face_flux_);
    owner_->interface_flux_commit_guard_.reset();
    owner_->interface_flux_ledger_.swap(interface_flux_ledger_);
    owner_->accepted_synchronization_events_.swap(accepted_synchronization_events_);
    owner_->accepted_state_revision_ = accepted_state_revision_;
    for (const auto& diagnostic : owner_->cell_temporal_diagnostics_)
      if (diagnostic)
        diagnostic->invalidate_accepted_publication(
            owner_->accepted_temporal_partition_.synchronization_tick,
            owner_->accepted_temporal_partition_.tick_denominator);
  }

 private:
  AmrProgramContext* owner_ = nullptr;
  ClockScheduleState clock_schedule_;
  std::uint64_t resource_epoch_ = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t resource_generation_ = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t history_epoch_ = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t history_generation_ = std::numeric_limits<std::uint64_t>::max();
  std::map<std::string, int> history_levels_;
  std::map<std::string, std::vector<FluxExpression>> history_flux_expressions_;
  std::map<std::string, AmrProgramPendingHistoryRemap> pending_history_remaps_;
  std::map<std::string, field_type> deferred_history_lag_scratches_;
  CellTemporalPartitionAcceptedState accepted_temporal_partition_;
  std::optional<CellTemporalConfiguration> cell_temporal_configuration_;
  std::string accepted_flux_budget_contract_;
  std::string accepted_coupling_contract_;
  std::array<std::vector<::pops::amr::reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>, Dim>
      accepted_face_flux_;
  std::unique_ptr<interface_flux_ledger_type> interface_flux_ledger_;
  std::vector<AmrProgramSynchronizationEvent> accepted_synchronization_events_;
  std::uint64_t accepted_state_revision_ = std::numeric_limits<std::uint64_t>::max();
  std::map<ScratchKey, field_type> discarded_scratches_;
  std::string discarded_subcycling_contract_;
};

std::unique_ptr<AcceptedProgramContextSnapshot> capture_accepted_context_snapshot_() const {
  if (!active_attempt_states_.empty() || !interface_flux_ledger_ ||
      interface_flux_ledger_->in_transaction())
    throw std::logic_error("AMR Program accepted context snapshot crossed an active attempt");
  return std::make_unique<AcceptedContextSnapshot>(*const_cast<AmrProgramContext*>(this));
}
