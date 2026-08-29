
class AcceptedContextSnapshot final : public AcceptedProgramExecutionServicesSnapshot {
  template <int>
  friend struct AmrProgramHistoryRemapCollectiveTestAccess;

 public:
  /// Value-owned forward/regrid handoff.  It intentionally contains no adapter pointer: the
  /// aggregate builder captures ForwardTopologyView::snapshot() and every staged accepted carrier
  /// here, then binds the resulting snapshot only after HiddenPublish has made its adapter live.
  struct DetachedState {
    ClockScheduleState clock_schedule;
    std::uint64_t resource_epoch = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t resource_generation = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t history_epoch = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t history_generation = std::numeric_limits<std::uint64_t>::max();
    std::map<std::string, int> history_levels;
    std::map<std::string, std::vector<FluxExpression>> history_flux_expressions;
    std::map<std::string, AmrProgramPendingHistoryRemap> pending_history_remaps;
    std::map<std::string, field_type> deferred_history_lag_scratches;
    CellTemporalPartitionAcceptedState accepted_temporal_partition;
    std::optional<CellTemporalConfiguration> cell_temporal_configuration;
    std::string accepted_flux_budget_contract;
    std::string accepted_coupling_contract;
    std::array<std::vector<::pops::amr::reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>, Dim>
        accepted_face_flux;
    std::unique_ptr<interface_flux_ledger_type> interface_flux_ledger;
    std::vector<AmrProgramSynchronizationEvent> accepted_synchronization_events;
    std::uint64_t accepted_state_revision = std::numeric_limits<std::uint64_t>::max();
  };

  static std::unique_ptr<AcceptedContextSnapshot> from_forward(DetachedState staged) {
    if (!staged.interface_flux_ledger || staged.interface_flux_ledger->in_transaction())
      throw std::logic_error("AMR Program detached accepted context requires a sealed ledger");
    // A ledger copy preserves its dense images but not the reserve of its transaction-contract
    // strings.  Forward preparation is cold; re-prime it here, before this detached image can be
    // rebound and used by an accepted attempt.  Refresh, publish and finalization deliberately
    // have no route to this bind-only operation.
    staged.interface_flux_ledger->prime_hot_carriers_at_bind();
    return std::unique_ptr<AcceptedContextSnapshot>(new AcceptedContextSnapshot(std::move(staged)));
  }

  std::unique_ptr<AcceptedProgramExecutionServicesSnapshot> detach_for_forward(
      std::uint64_t topology_epoch, std::uint64_t materialization_generation,
      void*& rebind_token) const override {
    rebind_token = static_cast<void*>(&rebind_owner());
    return from_forward(detach_for_forward(topology_epoch, materialization_generation));
  }

  void rebind_after_forward_publish(void* rebind_token) noexcept override {
    auto* owner = static_cast<AmrStorageTopologyAdapter*>(rebind_token);
    if (owner == nullptr || owner_ != nullptr || !interface_flux_ledger_ ||
        interface_flux_ledger_->in_transaction())
      std::terminate();
    owner_ = owner;
  }

  void prepare_forward_hierarchy_refresh(std::uint64_t topology_epoch,
                                         std::uint64_t materialization_generation) override {
    require_detached_forward_authority_(topology_epoch, materialization_generation,
                                        "hierarchy refresh");
    if (!history_levels_.empty() || !history_flux_expressions_.empty() ||
        !pending_history_remaps_.empty())
      throw std::logic_error(
          "AMR Program forward hierarchy refresh requires an explicit detached history remap");
    invalidate_forward_topology_resources_();
  }

  void prepare_forward_history_remap(const AmrProgramHistoryRemapDescriptor& descriptor) override {
    if (descriptor.published_topology_epoch == std::numeric_limits<std::uint64_t>::max() ||
        descriptor.published_materialization_generation ==
            std::numeric_limits<std::uint64_t>::max())
      throw std::logic_error("AMR Program forward history remap has no published authority");
    // A cumulative regrid retains one detached image of the final forward hierarchy, while each
    // descriptor remains the direct transition which created its child ring.  Requiring every
    // descriptor to equal the final epoch would forge non-direct checkpoint markers.  Instead,
    // each transition is bounded by the detached final authority; the carrier authenticates it
    // against its own staged transaction and the checkpoint validator requires the complete
    // contiguous chain to end at this final image.
    if (owner_ != nullptr || !interface_flux_ledger_ || interface_flux_ledger_->in_transaction() ||
        descriptor.prior_topology_epoch == std::numeric_limits<std::uint64_t>::max() ||
        descriptor.prior_materialization_generation == std::numeric_limits<std::uint64_t>::max() ||
        descriptor.prior_topology_epoch + 1U != descriptor.published_topology_epoch ||
        descriptor.prior_materialization_generation + 1U !=
            descriptor.published_materialization_generation ||
        descriptor.published_topology_epoch > resource_epoch_ ||
        descriptor.published_materialization_generation > resource_generation_)
      throw std::logic_error("AMR Program detached accepted context cannot prepare history remap");
    prepare_detached_history_remap_(descriptor);
  }

  /// Rebuild the topology-bound cell-local provider solely from the forward hierarchy authority.
  /// This runs during Candidate: every allocation is confined to local replacement values and the
  /// only mutation of the detached image is the final no-throw swap/publication of those values.
  void prepare_forward_temporal_partition(
      const PreparedForwardAmrTemporalAuthority& authority) override {
    require_detached_forward_authority_(authority.topology_epoch,
                                        authority.materialization_generation, "temporal partition");
    if (authority.accepted_state_revision == std::numeric_limits<std::uint64_t>::max() ||
        authority.spatial_contract.empty() || authority.lane_identity.empty() ||
        authority.collective_contract.empty() || authority.level_count == 0 ||
        authority.block_count == 0 ||
        authority.block_count > std::numeric_limits<std::size_t>::max() / authority.level_count ||
        authority.block_level_cell_counts.size() != authority.block_count * authority.level_count ||
        authority.periodic_faces.size() != static_cast<std::size_t>(2 * Dim) ||
        authority.temporal_provider_identity.empty() || authority.flux_budget_contract.empty() ||
        authority.coupling_contract.empty() ||
        authority.interface_flux_ledger_budget.exact_contract.empty() ||
        interface_flux_ledger_->topology_epoch() != authority.topology_epoch)
      throw std::logic_error("AMR Program forward temporal authority is incomplete");

    std::optional<CellTemporalConfiguration> next_configuration;
    CellTemporalPartitionAcceptedState next_partition;
    std::string next_flux_budget_contract;
    std::string next_coupling_contract;

    if (accepted_temporal_partition_.kind == TemporalPartitionKind::Global) {
      if (cell_temporal_configuration_ ||
          authority.temporal_provider_identity != kGlobalTemporalPartitionProvider ||
          accepted_temporal_partition_.provider_identity != kGlobalTemporalPartitionProvider ||
          !accepted_temporal_partition_.cells.empty() || authority.coupling_count != 0 ||
          authority.has_interface_flux_provider)
        throw std::logic_error(
            "AMR Program forward global temporal partition changed its provider authority");
      next_partition = accepted_temporal_partition_;
      validate_cell_temporal_partition_state(next_partition);
    } else if (accepted_temporal_partition_.kind == TemporalPartitionKind::CellLocal) {
      if (!cell_temporal_configuration_ ||
          authority.temporal_provider_identity != kSameLevelTransportEulerStageFluxProvider ||
          accepted_temporal_partition_.provider_identity !=
              kSameLevelTransportEulerStageFluxProvider)
        throw std::logic_error(
            "AMR Program forward cell-local temporal provider is not rematerializable");
      next_configuration.emplace(*cell_temporal_configuration_);
      next_partition = prepare_forward_cell_temporal_partition_(
          *next_configuration, authority, accepted_temporal_partition_.synchronization_tick);
    } else {
      throw std::logic_error("AMR Program forward temporal partition has an unsupported kind");
    }
    next_flux_budget_contract = authority.flux_budget_contract;
    next_coupling_contract = authority.coupling_contract;

    // ``prepare_budget`` allocates at Candidate time.  Once it has succeeded the ledger replaces
    // only its detached bound image; the subsequent string/configuration/partition swaps cannot
    // allocate or consult the sealed owner.
    auto prepared_budget =
        interface_flux_ledger_->prepare_budget(authority.interface_flux_ledger_budget);
    interface_flux_ledger_->publish_prepared_budget(prepared_budget);
    std::swap(cell_temporal_configuration_, next_configuration);
    std::swap(accepted_temporal_partition_, next_partition);
    accepted_flux_budget_contract_.swap(next_flux_budget_contract);
    accepted_coupling_contract_.swap(next_coupling_contract);
    accepted_state_revision_ = authority.accepted_state_revision;
  }

  [[nodiscard]] PreparedForwardAmrAcceptedContext prepare_forward_accepted_context(
      std::int64_t accepted_macro_step) const override {
    if (owner_ != nullptr || !interface_flux_ledger_ || interface_flux_ledger_->in_transaction())
      throw std::logic_error(
          "AMR Program forward accepted checkpoint requires one sealed detached context");
    PreparedForwardAmrAcceptedContext result;
    result.topology_epoch = resource_epoch_;
    result.materialization_generation = resource_generation_;
    result.accepted_state_revision = accepted_state_revision_;
    result.logical_clock_ticks = clock_schedule_.accepted_ticks(accepted_macro_step);
    result.pending_history_remaps.reserve(pending_history_remaps_.size());
    for (const auto& [key, marker] : pending_history_remaps_) {
      if (key != marker.key)
        throw std::logic_error(
            "AMR Program detached accepted checkpoint has a foreign deferred history key");
      result.pending_history_remaps.push_back(marker);
    }
    result.history_flux_payload = serialize_history_flux_payload_(history_flux_expressions_);
    result.temporal_partition = accepted_temporal_partition_;
    result.flux_budget_contract = accepted_flux_budget_contract_;
    result.coupling_contract = accepted_coupling_contract_;
    result.topology_scoped_effects_invalidated =
        std::all_of(accepted_face_flux_.begin(), accepted_face_flux_.end(),
                    [](const auto& axis) { return axis.empty(); }) &&
        accepted_synchronization_events_.empty() && deferred_history_lag_scratches_.empty();
    if (!result.topology_scoped_effects_invalidated)
      throw std::logic_error(
          "AMR Program detached accepted checkpoint retained topology-scoped effects");
    return result;
  }

  void prime_at_bind() override {
    // The execution adapter and its accepted snapshot own distinct dense images.  The accepted
    // flux/coupling contracts are not merely metadata: preparing the subcycling carrier seals
    // their exact values.  Materialize that cold authority before cloning the resident snapshot,
    // otherwise the first accepted step would change its shape after bind and the next capture
    // would (correctly) refuse it as an unprimed rollback image.
    if (owner_ == nullptr || !owner_->interface_flux_ledger_ ||
        owner_->interface_flux_ledger_->in_transaction())
      throw std::logic_error("AMR Program accepted context cannot cold-prime its live ledger");
    // `refresh_accepted_hierarchy_state_` serializes the now-frozen authority through the
    // facade.  Seal its maximum checkpoint envelope first, while bind-time allocation remains
    // permitted; candidate restore must only consume this resident capacity.
    (void)owner_->facade_->program_checkpoint_state_capacity_();
    owner_->refresh_accepted_hierarchy_state_();
    owner_->interface_flux_ledger_->prime_hot_carriers_at_bind();
    reprime_from_frozen_owner_at_bind_();
    owner_->interface_flux_ledger_->prime_snapshot_arenas_at_bind();
    owner_->interface_flux_ledger_->prime_snapshot_slots_at_bind();
    prime_interface_flux_snapshot_arenas_at_bind();
    prime_interface_flux_slots_at_bind();
  }

  void prime_copied_image_at_bind() override {
    // Copying strings/vectors preserves logical contents but not their bind-sealed spare
    // capacity. The live owner was already fully refreshed by `prime_at_bind`; only rebuild this
    // copied snapshot's finite carriers, without touching checkpoint metadata a second time.
    prime_copied_capacities_from_owner_at_bind_();
    prime_interface_flux_snapshot_arenas_at_bind();
    prime_interface_flux_slots_at_bind();
  }

  /// Extract a value-owned image for a forward topology while the accepted snapshot is still
  /// detached from the live adapter.  This is the only regrid handoff allowed to adjust epochs;
  /// it never reads an adapter or retains one of its pointers.
  [[nodiscard]] DetachedState detach_for_forward(std::uint64_t topology_epoch,
                                                 std::uint64_t materialization_generation) const {
    if (owner_ == nullptr || !interface_flux_ledger_ || interface_flux_ledger_->in_transaction() ||
        topology_epoch == std::numeric_limits<std::uint64_t>::max() ||
        materialization_generation == std::numeric_limits<std::uint64_t>::max())
      throw std::logic_error("AMR Program forward accepted context is not detachable");
    DetachedState staged{.clock_schedule = clock_schedule_,
                         .resource_epoch = topology_epoch,
                         .resource_generation = materialization_generation,
                         .history_epoch = topology_epoch,
                         .history_generation = materialization_generation,
                         .history_levels = history_levels_,
                         .history_flux_expressions = history_flux_expressions_,
                         .pending_history_remaps = pending_history_remaps_,
                         .deferred_history_lag_scratches = deferred_history_lag_scratches_,
                         .accepted_temporal_partition = accepted_temporal_partition_,
                         .cell_temporal_configuration = cell_temporal_configuration_,
                         .accepted_flux_budget_contract = accepted_flux_budget_contract_,
                         .accepted_coupling_contract = accepted_coupling_contract_,
                         .accepted_face_flux = accepted_face_flux_,
                         .interface_flux_ledger =
                             std::make_unique<interface_flux_ledger_type>(*interface_flux_ledger_),
                         .accepted_synchronization_events = accepted_synchronization_events_,
                         .accepted_state_revision = accepted_state_revision_};
    // std::vector/std::string copies intentionally retain logical contents but are permitted to
    // discard spare capacity.  This handoff runs while regrid preparation is cold, so restore the
    // already-authenticated envelopes before the detached image becomes candidate authority.
    prime_detached_state_capacities_from_cold_source_(staged, *this);
    staged.interface_flux_ledger->prime_hot_carriers_at_bind();
    staged.interface_flux_ledger->advance_topology_epoch(topology_epoch);
    return staged;
  }

  void rebind_after_publish(AmrStorageTopologyAdapter& owner) {
    if (owner_ != nullptr || !interface_flux_ledger_ || interface_flux_ledger_->in_transaction())
      throw std::logic_error("AMR Program detached accepted context cannot be rebound");
    owner_ = &owner;
  }

  [[nodiscard]] AmrStorageTopologyAdapter& rebind_owner() const {
    if (owner_ == nullptr)
      throw std::logic_error("AMR Program accepted context has no live adapter owner");
    return *owner_;
  }

  explicit AcceptedContextSnapshot(AmrStorageTopologyAdapter& owner)
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
        accepted_face_flux_slots_(owner.accepted_face_flux_),
        accepted_synchronization_event_slots_(owner.accepted_synchronization_events_),
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
        accepted_face_flux_slots_(accepted.accepted_face_flux_slots_),
        accepted_synchronization_event_slots_(accepted.accepted_synchronization_event_slots_),
        accepted_state_revision_(accepted.accepted_state_revision_) {}

  std::unique_ptr<AcceptedProgramExecutionServicesSnapshot> prepare_restore() const override {
    // This is a preparation boundary: the clone will later be swapped into the owner by the
    // no-throw restore path.  Rebuild its already-authenticated spare capacity here, never while
    // refreshing, publishing or finalizing an attempt.
    auto prepared = std::make_unique<AcceptedContextSnapshot>(*this);
    prepared->prime_copied_capacities_from_cold_source_(*this);
    return prepared;
  }

  /// Bind-only dense ledger priming.  The table comes from the artifact's finite interface-flux
  /// descriptor: it fixes every slot's string and payload capacity before the first candidate.
  /// No accepted-step/finalizer path may call this method.
  void prime_interface_flux_slots_at_bind() {
    require_owner_cold_prime_();
    interface_flux_ledger_->prime_snapshot_slots_at_bind();
  }

  /// Bind/regrid-preparation only: reserve the authenticated flat identity and payload arenas.
  /// The caller must invoke this before the first accepted candidate; refresh/finalize paths are
  /// forbidden from priming either arena.
  void prime_interface_flux_snapshot_arenas_at_bind() {
    require_owner_cold_prime_();
    interface_flux_ledger_->prime_snapshot_arenas_at_bind();
  }

  void refresh_from_owner_preallocated() override {
    require_owner_cold_prime_();

    // Never fall back to ``prepare_restore`` here: that path clones maps, flux payloads and Fabs
    // and would allocate after a transaction has obtained its writer lease.  A resident image
    // carries the sealed shape/key set from cold bind; all accepted scalar and payload values are
    // refreshed in place below.
    require_refresh_preallocated_();

    // `require_refresh_preallocated_` has already rejected every known structural and capacity
    // failure.  The writes below are scalar copies, swaps, or Kokkos deep copies into existing
    // allocations only; no rollback-visible state is changed on a preflight failure.
    owner_->clock_schedule_.copy_into_preallocated(clock_schedule_);
    copy_history_levels_preallocated_(history_levels_, owner_->history_levels_);
    copy_history_flux_expressions_preallocated_(history_flux_expressions_,
                                                owner_->history_flux_expressions_);
    copy_pending_history_remaps_preallocated_(pending_history_remaps_,
                                              owner_->pending_history_remaps_);
    copy_deferred_history_lag_scratches_preallocated_(deferred_history_lag_scratches_,
                                                      owner_->deferred_history_lag_scratches_);
    copy_temporal_partition_preallocated_(accepted_temporal_partition_,
                                          owner_->accepted_temporal_partition_);
    copy_cell_temporal_configuration_preallocated_(cell_temporal_configuration_,
                                                   owner_->cell_temporal_configuration_);
    if (accepted_flux_budget_contract_ != owner_->accepted_flux_budget_contract_ ||
        accepted_coupling_contract_ != owner_->accepted_coupling_contract_)
      throw std::logic_error("AMR Program accepted flux authority changed after prime");
    copy_events_preallocated_(accepted_synchronization_events_,
                              accepted_synchronization_event_slots_,
                              owner_->accepted_synchronization_events_);
    copy_face_flux_preallocated_(accepted_face_flux_, accepted_face_flux_slots_,
                                 owner_->accepted_face_flux_);
    interface_flux_ledger_->copy_from_preallocated(*owner_->interface_flux_ledger_);
    accepted_state_revision_ = owner_->accepted_state_revision_;
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
    if (owner_ == nullptr || !interface_flux_ledger_ || interface_flux_ledger_->in_transaction())
      std::terminate();
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
  void require_detached_forward_authority_(std::uint64_t topology_epoch,
                                           std::uint64_t materialization_generation,
                                           const char* operation) const {
    if (owner_ != nullptr || !interface_flux_ledger_ || interface_flux_ledger_->in_transaction() ||
        resource_epoch_ != topology_epoch || resource_generation_ != materialization_generation ||
        history_epoch_ != topology_epoch || history_generation_ != materialization_generation)
      throw std::logic_error(std::string("AMR Program detached accepted context cannot prepare ") +
                             operation);
  }

  [[nodiscard]] static std::uint64_t forward_cell_count_(
      const PreparedForwardAmrTemporalAuthority& authority, std::size_t block, std::size_t level) {
    if (block >= authority.block_count || level >= authority.level_count ||
        block > (std::numeric_limits<std::size_t>::max() - level) / authority.level_count)
      throw std::logic_error("AMR Program forward temporal authority has an invalid cell index");
    return authority.block_level_cell_counts[block * authority.level_count + level];
  }

  [[nodiscard]] static std::vector<int> forward_cell_temporal_level_rungs_(
      int finest_rung, const PreparedForwardAmrTemporalAuthority& authority) {
    if (finest_rung < 0 || finest_rung > 30 || authority.level_count == 0 ||
        authority.temporal_relations.size() + 1 != authority.level_count)
      throw std::logic_error(
          "AMR Program forward temporal authority lacks one relation per level transition");
    std::vector<int> rungs(authority.level_count, finest_rung);
    for (std::size_t child = authority.temporal_relations.size(); child != 0; --child) {
      const auto& relation = authority.temporal_relations[child - 1];
      if (relation.parent_level() != static_cast<int>(child - 1) ||
          relation.child_level() != static_cast<int>(child))
        throw std::logic_error(
            "AMR Program forward temporal authority has a non-canonical level relation");
      const auto ratio = relation.temporal_ratio();
      if (ratio.numerator <= 0 || ratio.denominator != 1)
        throw std::invalid_argument(
            "cell-local AMR requires integral power-of-two temporal refinement ratios");
      const auto refinement = static_cast<std::uint64_t>(ratio.numerator);
      if ((refinement & (refinement - 1)) != 0)
        throw std::invalid_argument(
            "cell-local AMR requires power-of-two temporal refinement ratios");
      int exponent = 0;
      for (std::uint64_t value = refinement; value > 1; value >>= 1)
        ++exponent;
      if (rungs[child] > 30 - exponent)
        throw std::invalid_argument("cell-local AMR derived rung exceeds its bounded domain");
      rungs[child - 1] = rungs[child] + exponent;
    }
    return rungs;
  }

  [[nodiscard]] static std::uint64_t forward_block_major_offset_(
      const CellTemporalConfiguration& configuration,
      const PreparedForwardAmrTemporalAuthority& authority, std::size_t route, std::size_t level) {
    std::uint64_t offset = 0;
    for (std::size_t prior = 0; prior <= route; ++prior) {
      const std::size_t stop = prior == route ? level : authority.level_count;
      const int block = configuration.routes[prior].runtime_block;
      if (block < 0)
        throw std::logic_error("AMR Program forward temporal route has a negative block");
      for (std::size_t prior_level = 0; prior_level < stop; ++prior_level) {
        const std::uint64_t count =
            forward_cell_count_(authority, static_cast<std::size_t>(block), prior_level);
        if (count > std::numeric_limits<std::uint64_t>::max() - offset)
          throw std::overflow_error("cell-local AMR block-major identity exceeds uint64_t");
        offset += count;
      }
    }
    return offset;
  }

  [[nodiscard]] static CellTemporalPartitionAcceptedState prepare_forward_cell_temporal_partition_(
      CellTemporalConfiguration& configuration,
      const PreparedForwardAmrTemporalAuthority& authority, std::int64_t synchronization_tick) {
    if constexpr (!cell_temporal_host_execution_supported_)
      throw std::invalid_argument(
          "cell-local AMR execution requires a host default execution and memory space");
    if (configuration.clock.empty() || configuration.tick_denominator <= 0 ||
        configuration.rung < 0 || configuration.rung > 30 || configuration.routes.empty() ||
        configuration.routes.size() != authority.block_count || authority.coupling_count != 0 ||
        authority.has_interface_flux_provider ||
        authority.periodic_faces.size() != static_cast<std::size_t>(2 * Dim) ||
        !std::all_of(authority.periodic_faces.begin(), authority.periodic_faces.end(),
                     [](bool periodic) { return periodic; }))
      throw std::invalid_argument(
          "cell-local AMR forward hierarchy requires periodic uncoupled complete routes");

    std::sort(configuration.routes.begin(), configuration.routes.end(),
              [](const auto& left, const auto& right) {
                return std::tie(left.runtime_block, left.program_block, left.rhs_id) <
                       std::tie(right.runtime_block, right.program_block, right.rhs_id);
              });
    for (std::size_t route = 0; route < configuration.routes.size(); ++route) {
      const auto& entry = configuration.routes[route];
      if (entry.program_block < 0 || entry.runtime_block != static_cast<int>(route) ||
          entry.rhs_id < 0 ||
          (route != 0 && configuration.routes[route - 1].program_block == entry.program_block))
        throw std::logic_error(
            "AMR Program forward cell-local routes are not a complete block bijection");
    }

    configuration.topology_epoch = authority.topology_epoch;
    configuration.materialization_generation = authority.materialization_generation;
    configuration.level_rungs = forward_cell_temporal_level_rungs_(configuration.rung, authority);
    configuration.level_cell_counts.clear();
    configuration.level_cell_counts.reserve(authority.level_count);
    for (std::size_t level = 0; level < authority.level_count; ++level) {
      const std::uint64_t count = forward_cell_count_(authority, 0, level);
      for (std::size_t block = 1; block < authority.block_count; ++block)
        if (forward_cell_count_(authority, block, level) != count)
          throw std::logic_error(
              "AMR Program forward cell-local blocks do not share one staged topology");
      configuration.level_cell_counts.push_back(count);
    }

    ExactContractBuilder contract;
    contract.text("pops.amr-program.cell-local-forward-euler")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(configuration.clock)
        .scalar(configuration.tick_denominator)
        .scalar(std::int32_t{configuration.rung})
        .scalar(configuration.topology_epoch)
        .scalar(configuration.materialization_generation)
        .text(authority.lane_identity)
        .bytes(authority.spatial_contract)
        .text("host-default-execution-and-memory")
        .presence(cell_temporal_host_execution_supported_)
        .scalar(std::uint64_t{2 * Dim})
        .scalar(static_cast<std::uint64_t>(authority.coupling_count))
        .presence(authority.has_interface_flux_provider)
        .scalar(static_cast<std::uint64_t>(configuration.routes.size()))
        .sequence(configuration.level_rungs,
                  [](ExactContractBuilder& item, int rung) { item.scalar(std::int32_t{rung}); })
        .sequence(configuration.level_cell_counts,
                  [](ExactContractBuilder& item, std::uint64_t count) { item.scalar(count); });
    for (bool periodic : authority.periodic_faces)
      contract.presence(periodic);
    for (const auto& route : configuration.routes)
      contract.scalar(std::int32_t{route.program_block})
          .scalar(std::int32_t{route.runtime_block})
          .scalar(std::int32_t{route.rhs_id});
    configuration.exact_contract = std::move(contract).release();

    if (synchronization_tick < 0 ||
        synchronization_tick % (std::int64_t{1} << configuration.level_rungs.front()) != 0)
      throw std::invalid_argument(
          "cell-local AMR forward accepted time is not aligned to its coarsest derived rung");

    CellTemporalPartitionAcceptedState partition;
    partition.kind = TemporalPartitionKind::CellLocal;
    partition.provider_identity = std::string(kSameLevelTransportEulerStageFluxProvider);
    partition.topology_epoch = authority.topology_epoch;
    partition.synchronization_tick = synchronization_tick;
    partition.tick_denominator = configuration.tick_denominator;
    std::uint64_t total_cells = 0;
    for (std::size_t level = 0; level < authority.level_count; ++level)
      for (std::size_t route = 0; route < configuration.routes.size(); ++route) {
        const int block = configuration.routes[route].runtime_block;
        const std::uint64_t count =
            forward_cell_count_(authority, static_cast<std::size_t>(block), level);
        if (count > std::numeric_limits<std::uint64_t>::max() - total_cells)
          throw std::overflow_error("cell-local AMR forward partition exceeds uint64_t");
        total_cells += count;
      }
    if (total_cells == 0 || total_cells > std::numeric_limits<std::size_t>::max())
      throw std::logic_error("cell-local AMR forward partition has an invalid cell capacity");
    partition.cells.reserve(static_cast<std::size_t>(total_cells));
    for (std::size_t level = 0; level < authority.level_count; ++level)
      for (std::size_t route = 0; route < configuration.routes.size(); ++route) {
        std::uint64_t cell = forward_block_major_offset_(configuration, authority, route, level);
        const int block = configuration.routes[route].runtime_block;
        const std::uint64_t count =
            forward_cell_count_(authority, static_cast<std::size_t>(block), level);
        if (count > std::numeric_limits<std::uint64_t>::max() - cell)
          throw std::overflow_error("cell-local AMR forward cell identity exceeds uint64_t");
        for (std::uint64_t ordinal = 0; ordinal < count; ++ordinal)
          partition.cells.push_back({static_cast<int>(level), cell++,
                                     configuration.level_rungs[level], synchronization_tick});
      }
    validate_cell_temporal_partition_state(partition);
    return partition;
  }

  /// Regrid preparation may allocate and build replacement provenance maps, but it must never
  /// query the bound adapter: that adapter still names the last sealed hierarchy.  The descriptor
  /// is the sole authority for affected keys; its deferred markers were prepared from the numeric
  /// HistoryManager before the forward topology was staged.
  void prepare_detached_history_remap_(const AmrProgramHistoryRemapDescriptor& descriptor) {
    if (descriptor.parent_level < 0 || descriptor.child_level < 0 ||
        descriptor.child_level != descriptor.parent_level + 1 ||
        descriptor.prior_topology_epoch == std::numeric_limits<std::uint64_t>::max() ||
        descriptor.prior_materialization_generation == std::numeric_limits<std::uint64_t>::max() ||
        descriptor.accepted_macro_step < 0 || descriptor.temporal_numerator <= 0 ||
        descriptor.temporal_denominator <= 0 || descriptor.operation_identity.empty())
      throw std::logic_error("AMR Program detached history remap descriptor is incomplete");

    std::map<std::string, const AmrProgramHistoryRemapEntry*> plan;
    for (const AmrProgramHistoryRemapEntry& entry : descriptor.history_plan) {
      if (entry.key.empty() || !plan.emplace(entry.key, &entry).second)
        throw std::logic_error("AMR Program detached history remap has a non-canonical plan");
      if (entry.source == AmrProgramHistoryRemapSource::ParentDeferred) {
        if (!descriptor.child_published || entry.parent_key.empty())
          throw std::logic_error(
              "AMR Program detached history remap has an invalid parent-deferred entry");
      } else if (!entry.parent_key.empty()) {
        throw std::logic_error(
            "AMR Program detached history remap has a non-canonical retained/removal parent key");
      }
    }

    // Every prior child provenance carrier must be mentioned by the exact plan.  Leaving one
    // outside the plan would retain a pointer-qualified history/lag image across a hierarchy
    // publication merely because the live callback used to rediscover it later.
    for (const auto& [key, level] : history_levels_)
      if (level == descriptor.child_level && !plan.contains(key))
        throw std::logic_error(
            "AMR Program detached history remap omits a prior child history carrier");
    for (const auto& [key, expressions] : history_flux_expressions_) {
      (void)expressions;
      const auto level = history_levels_.find(key);
      if (level == history_levels_.end())
        throw std::logic_error(
            "AMR Program detached history remap found flux provenance without a history carrier");
      if (level->second == descriptor.child_level && !plan.contains(key))
        throw std::logic_error(
            "AMR Program detached history remap omits prior child flux provenance");
    }
    for (const auto& [key, marker] : pending_history_remaps_)
      if (marker.child_level == descriptor.child_level && !plan.contains(key))
        throw std::logic_error(
            "AMR Program detached history remap omits a prior child deferred marker");

    auto next_levels = history_levels_;
    auto next_flux = history_flux_expressions_;
    auto next_pending = pending_history_remaps_;
    auto next_deferred_scratches = deferred_history_lag_scratches_;

    for (const auto& [key, entry] : plan) {
      const auto level = history_levels_.find(key);
      const auto flux = history_flux_expressions_.find(key);
      switch (entry->source) {
        case AmrProgramHistoryRemapSource::Removed:
          if (descriptor.child_published ||
              (level != history_levels_.end() && level->second != descriptor.child_level) ||
              (flux != history_flux_expressions_.end() && level == history_levels_.end()))
            throw std::logic_error("AMR Program detached history remap removes a foreign ring");
          next_levels.erase(key);
          next_flux.erase(key);
          next_pending.erase(key);
          next_deferred_scratches.erase(key);
          break;
        case AmrProgramHistoryRemapSource::RetainedChild:
          if (!descriptor.child_published || level == history_levels_.end() ||
              level->second != descriptor.child_level || flux == history_flux_expressions_.end())
            throw std::logic_error(
                "AMR Program detached history remap lacks retained child provenance");
          // A lag marker refers to the old child geometry and cannot survive an accepted
          // topology transition without a fresh, descriptor-authenticated replacement.
          next_pending.erase(key);
          next_deferred_scratches.erase(key);
          break;
        case AmrProgramHistoryRemapSource::ParentDeferred: {
          if (!descriptor.child_physical_layout_changed)
            throw std::logic_error(
                "AMR Program detached history remap defers a parent without a physical child "
                "change");
          const auto parent_level = history_levels_.find(entry->parent_key);
          const auto parent_flux = history_flux_expressions_.find(entry->parent_key);
          if (parent_level == history_levels_.end() ||
              parent_level->second != descriptor.parent_level ||
              parent_flux == history_flux_expressions_.end() || parent_flux->second.empty())
            throw std::logic_error(
                "AMR Program detached history remap lacks authenticated parent provenance");
          if (level != history_levels_.end() && level->second != descriptor.child_level)
            throw std::logic_error(
                "AMR Program detached history remap reuses a foreign child history key");
          if (flux != history_flux_expressions_.end() &&
              flux->second.size() != parent_flux->second.size())
            throw std::logic_error(
                "AMR Program detached history remap changes the child provenance ring depth");
          next_levels.insert_or_assign(key, descriptor.child_level);
          next_flux.insert_or_assign(key, parent_flux->second);
          next_pending.erase(key);
          next_deferred_scratches.erase(key);
          break;
        }
      }
    }

    std::set<std::string> marker_keys;
    for (const AmrProgramPendingHistoryRemap& marker : descriptor.prepared_pending_history_remaps) {
      const auto entry = plan.find(marker.key);
      if (marker.key.empty() || !marker_keys.insert(marker.key).second || entry == plan.end() ||
          entry->second->source != AmrProgramHistoryRemapSource::ParentDeferred ||
          marker.parent_level != descriptor.parent_level ||
          marker.child_level != descriptor.child_level ||
          marker.prior_topology_epoch != descriptor.prior_topology_epoch ||
          marker.prior_materialization_generation != descriptor.prior_materialization_generation ||
          marker.published_topology_epoch != descriptor.published_topology_epoch ||
          marker.published_materialization_generation !=
              descriptor.published_materialization_generation ||
          marker.accepted_macro_step != descriptor.accepted_macro_step ||
          marker.temporal_numerator != descriptor.temporal_numerator ||
          marker.temporal_denominator != descriptor.temporal_denominator || marker.consumed ||
          !(marker.source_dt > 0.0) || !(marker.target_dt > 0.0) ||
          marker.target_dt != marker.source_dt / static_cast<double>(marker.temporal_numerator))
        throw std::logic_error(
            "AMR Program detached history remap has a non-canonical deferred marker");
      if (!next_levels.contains(marker.key) || !next_flux.contains(marker.key))
        throw std::logic_error(
            "AMR Program detached history remap marker has no prepared child provenance");
      next_pending.insert_or_assign(marker.key, marker);
    }

    // A marker can only be prepared by the exact direct-child IntegralOnly route.  If that route
    // was not selected, a non-empty marker DTO is a construction error rather than a late live
    // fallback.
    if (!descriptor.prepared_pending_history_remaps.empty() &&
        (!descriptor.child_physical_layout_changed || !descriptor.child_published ||
         !descriptor.integral_only || descriptor.temporal_denominator != 1 ||
         (descriptor.temporal_numerator != 1 && descriptor.temporal_numerator != 2)))
      throw std::logic_error(
          "AMR Program detached history remap has an unsupported deferred temporal relation");

    history_levels_.swap(next_levels);
    history_flux_expressions_.swap(next_flux);
    pending_history_remaps_.swap(next_pending);
    deferred_history_lag_scratches_.swap(next_deferred_scratches);
    invalidate_forward_topology_resources_();
  }

  /// These values carry patch addresses or a completed flux publication.  They are not valid on
  /// a new hierarchy.  Their replacement is prepared by the forward graph before HiddenPublish;
  /// clearing the detached image is therefore a deliberate invalidation, never a live refresh.
  void invalidate_forward_topology_resources_() noexcept {
    deferred_history_lag_scratches_.clear();
    for (auto& axis : accepted_face_flux_)
      axis.clear();
    accepted_synchronization_events_.clear();
  }

  explicit AcceptedContextSnapshot(DetachedState staged)
      : clock_schedule_(std::move(staged.clock_schedule)),
        resource_epoch_(staged.resource_epoch),
        resource_generation_(staged.resource_generation),
        history_epoch_(staged.history_epoch),
        history_generation_(staged.history_generation),
        history_levels_(std::move(staged.history_levels)),
        history_flux_expressions_(std::move(staged.history_flux_expressions)),
        pending_history_remaps_(std::move(staged.pending_history_remaps)),
        deferred_history_lag_scratches_(std::move(staged.deferred_history_lag_scratches)),
        accepted_temporal_partition_(std::move(staged.accepted_temporal_partition)),
        cell_temporal_configuration_(std::move(staged.cell_temporal_configuration)),
        accepted_flux_budget_contract_(std::move(staged.accepted_flux_budget_contract)),
        accepted_coupling_contract_(std::move(staged.accepted_coupling_contract)),
        accepted_face_flux_(std::move(staged.accepted_face_flux)),
        interface_flux_ledger_(std::move(staged.interface_flux_ledger)),
        accepted_synchronization_events_(std::move(staged.accepted_synchronization_events)),
        accepted_face_flux_slots_(accepted_face_flux_),
        accepted_synchronization_event_slots_(accepted_synchronization_events_),
        accepted_state_revision_(staged.accepted_state_revision) {}

  void require_owner_cold_prime_() const {
    if (owner_ == nullptr || !owner_->interface_flux_ledger_ || !interface_flux_ledger_ ||
        owner_->interface_flux_ledger_->in_transaction() ||
        interface_flux_ledger_->in_transaction())
      throw std::logic_error(
          "AMR Program accepted context cannot refresh while an attempt is active");
  }

  /// Rebuild this snapshot only at the bind boundary, after the owner has frozen every
  /// topology/flux authority.  Candidate capture uses `refresh_from_owner_preallocated()` and
  /// therefore never reaches this allocating path.  Keeping the old image in `prepared` until
  /// the final swaps also leaves this snapshot untouched if any copy construction fails.
  void reprime_from_frozen_owner_at_bind_() {
    if (owner_ == nullptr)
      throw std::logic_error("AMR Program accepted context has no owner to cold-reprime");
    AcceptedContextSnapshot prepared(*owner_);
    if (prepared.owner_ != owner_)
      throw std::logic_error("AMR Program accepted context cold-reprime changed its owner");
    prepared.prime_copied_capacities_from_owner_at_bind_();

    using std::swap;
    swap(clock_schedule_, prepared.clock_schedule_);
    swap(resource_epoch_, prepared.resource_epoch_);
    swap(resource_generation_, prepared.resource_generation_);
    swap(history_epoch_, prepared.history_epoch_);
    swap(history_generation_, prepared.history_generation_);
    swap(history_levels_, prepared.history_levels_);
    swap(history_flux_expressions_, prepared.history_flux_expressions_);
    swap(pending_history_remaps_, prepared.pending_history_remaps_);
    swap(deferred_history_lag_scratches_, prepared.deferred_history_lag_scratches_);
    swap(accepted_temporal_partition_, prepared.accepted_temporal_partition_);
    swap(cell_temporal_configuration_, prepared.cell_temporal_configuration_);
    swap(accepted_flux_budget_contract_, prepared.accepted_flux_budget_contract_);
    swap(accepted_coupling_contract_, prepared.accepted_coupling_contract_);
    swap(accepted_face_flux_, prepared.accepted_face_flux_);
    swap(interface_flux_ledger_, prepared.interface_flux_ledger_);
    swap(accepted_synchronization_events_, prepared.accepted_synchronization_events_);
    swap(accepted_face_flux_slots_, prepared.accepted_face_flux_slots_);
    swap(accepted_synchronization_event_slots_, prepared.accepted_synchronization_event_slots_);
    swap(accepted_state_revision_, prepared.accepted_state_revision_);
    swap(discarded_scratches_, prepared.discarded_scratches_);
    swap(discarded_subcycling_contract_, prepared.discarded_subcycling_contract_);
  }

#include <pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_capacity.hpp>

  AmrStorageTopologyAdapter* owner_ = nullptr;
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
  /// Cold-resident string/payload slots for accepted effects.  The public vectors retain their
  /// logical size; these slots retain the identity/capacity envelope needed when that size later
  /// contracts and grows again during a topology-static candidate.
  std::array<std::vector<::pops::amr::reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>, Dim>
      accepted_face_flux_slots_;
  std::vector<AmrProgramSynchronizationEvent> accepted_synchronization_event_slots_;
  std::uint64_t accepted_state_revision_ = std::numeric_limits<std::uint64_t>::max();
  std::map<ScratchKey, field_type> discarded_scratches_;
  std::string discarded_subcycling_contract_;
};

std::unique_ptr<AcceptedProgramExecutionServicesSnapshot> capture_accepted_context_snapshot_()
    const {
  if (!active_attempt_states_.empty() || !interface_flux_ledger_ ||
      interface_flux_ledger_->in_transaction())
    throw std::logic_error("AMR Program accepted context snapshot crossed an active attempt");
  return std::make_unique<AcceptedContextSnapshot>(*const_cast<AmrStorageTopologyAdapter*>(this));
}
