
/// Bind-sealed mutation image for one exact history ring.  Snapshot copies deliberately retain
/// the value-owned rollback fields while their raw live pointers are rebound only after the
/// corresponding ProgramRuntimeState has crossed the aggregate publication boundary.
struct PreparedHistoryMutationSlot {
  std::string name;
  std::string key;
  std::string clock_identity;
  int level = -1;
  std::vector<field_type> rollback_ring;
  std::vector<Real> dts;
  std::vector<FluxExpression> expressions;
  std::string store_contract;
  std::vector<field_type>* live_ring = nullptr;
  std::vector<Real>* live_dts = nullptr;
  std::vector<FluxExpression>* live_expressions = nullptr;
  bool* live_initialized = nullptr;
  bool* live_store_pending = nullptr;
  int* live_fill_count = nullptr;
  const int* live_depth = nullptr;
  const std::string* live_clock_identity = nullptr;
  AmrProgramPendingHistoryRemap* live_pending_remap = nullptr;
};

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
    /// Host-only fallback bundle for artifacts whose accepted snapshot is the common runtime
    /// snapshot rather than a DSO decorator.  It is value-only: no facade/runtime pointer can
    /// survive the forward Candidate boundary.
    std::uint64_t forward_execution_bundle_epoch = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t forward_execution_bundle_generation = std::numeric_limits<std::uint64_t>::max();
    /// Exact rollback-image charge measured while A is still bound.  Candidate preparation uses
    /// it only for the simultaneous A+B gate; it is never a pointer back to the accepted owner.
    std::uint64_t detached_accepted_snapshot_bytes = 0;
    std::optional<std::uint64_t> test_forward_storage_ceiling_override;
    std::map<std::string, int> history_levels;
    std::map<std::string, int> history_runtime_owners;
    std::map<std::string, std::vector<FluxExpression>> history_flux_expressions;
    std::map<std::string, AmrProgramPendingHistoryRemap> pending_history_remaps;
    std::map<std::string, field_type> deferred_history_lag_scratches;
    PreparedScratchStorage prepared_scratch;
    PreparedScratchDescriptors prepared_scratch_descriptors;
    PreparedHotPathWorkspace hot_path_workspace;
    AcceptedStateStaging<Dim> accepted_state_staging;
    std::shared_ptr<const AmrProgramAcceptedStateStagingCapacity<Dim>> forward_storage_capacity;
    PreparedFluxTableCarrier forward_flux_tables;
    std::optional<PreparedSubcyclingBundle> prepared_subcycling_bundle;
    std::optional<HierarchyTensorSelection> hierarchy_tensor_selection;
    std::array<std::vector<AcceptedFaceFluxOrdinal>, Dim> accepted_face_flux_ordinals;
    std::vector<typename AcceptedStateStaging<Dim>::InterfaceFluxSerializationView>
        accepted_interface_flux_staging_sources;
    // The ordinal arena grows with the staged interface budget. Carry its storage with the
    // detached topology image so HiddenPublish can exchange it without allocating.
    std::vector<std::size_t> accepted_interface_flux_wire_ordinals;
    // The accepted-state serializer borrows this envelope while it assembles level clocks.
    // It belongs to the detached image rather than the live adapter so a forward Candidate can
    // grow it for its staged hierarchy and HiddenPublish can install it by a no-throw swap.
    std::vector<::pops::amr::ClockStamp> accepted_checkpoint_level_clock_slots;
    CellTemporalPartitionAcceptedState accepted_temporal_partition;
    std::optional<CellTemporalConfiguration> cell_temporal_configuration;
    std::string accepted_flux_budget_contract;
    std::string accepted_coupling_contract;
    std::array<std::vector<::pops::amr::reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>, Dim>
        accepted_face_flux;
    /// Cold-resident topology templates for the logical accepted face-flux image.  They carry
    /// every static key and payload envelope while ``accepted_face_flux`` itself remains exactly
    /// the currently published cardinality.
    std::array<std::vector<::pops::amr::reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>, Dim>
        accepted_face_flux_slots;
    std::unique_ptr<interface_flux_ledger_type> interface_flux_ledger;
    std::vector<AmrProgramSynchronizationEvent> accepted_synchronization_events;
    /// Full configured event pool.  The logical accepted vector may be empty before bootstrap.
    std::vector<AmrProgramSynchronizationEvent> accepted_synchronization_event_slots;
    std::optional<typename multiblock_subcycling_type::MutableStateImage>
        multiblock_subcycling_state;
    std::uint64_t accepted_state_revision = std::numeric_limits<std::uint64_t>::max();
  };

  // clang-format off
#include <pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_snapshot.hpp>
  // clang-format on

  void prepare_forward_scratch_rematerialization(
      const PreparedForwardAmrScratchTopology& topology) override {
    const auto* forward = dynamic_cast<const ForwardScratchTopology*>(&topology);
    if (forward == nullptr || owner_ != nullptr)
      throw std::logic_error("AMR Program forward scratch authority is not detached");
    const auto& prototypes = forward->values();
    if (prepared_scratch_.size() != prepared_scratch_descriptors_.size())
      throw std::logic_error("AMR Program forward scratch declaration/storage shape differs");

    PreparedScratchStorage candidate;
    candidate.resize(prepared_scratch_descriptors_.size());
    for (std::size_t slot = 0; slot < prepared_scratch_descriptors_.size(); ++slot)
      for (std::size_t kind = 0; kind < candidate[slot].size(); ++kind) {
        const auto& declarations = prepared_scratch_descriptors_[slot][kind];
        const auto& prior = prepared_scratch_[slot][kind];
        if (declarations.size() != prior.size())
          throw std::logic_error("AMR Program forward scratch declaration lost a sealed subslot");
        auto& family = candidate[slot][kind];
        family.resize(declarations.size());
        for (std::size_t subslot = 0; subslot < declarations.size(); ++subslot) {
          if (declarations[subslot].has_value() != prior[subslot].has_value())
            throw std::logic_error("AMR Program forward scratch declaration/storage mismatch");
          if (!declarations[subslot])
            continue;
          const PreparedScratchDescriptor& declaration = *declarations[subslot];
          if (declaration.runtime_block < 0 ||
              static_cast<std::size_t>(declaration.runtime_block) >= prototypes.size() ||
              declaration.declared_level < -1 || declaration.ncomp < 1)
            throw std::logic_error("AMR Program forward scratch has an invalid sealed owner");
          const auto& levels = prototypes[static_cast<std::size_t>(declaration.runtime_block)];
          if (levels.empty() ||
              (declaration.declared_level >= 0 &&
               static_cast<std::size_t>(declaration.declared_level) >= levels.size()))
            throw std::logic_error("AMR Program forward scratch owner has no staged levels");
          std::vector<field_type> image;
          const std::size_t first_level =
              declaration.declared_level < 0 ? 0U
                                             : static_cast<std::size_t>(declaration.declared_level);
          const std::size_t level_count = declaration.declared_level < 0 ? levels.size() : 1U;
          image.reserve(level_count);
          for (std::size_t index = 0; index < level_count; ++index) {
            const field_type* prototype = levels.at(first_level + index);
            if (prototype == nullptr)
              throw std::logic_error("AMR Program forward scratch has a null staged prototype");
            image.emplace_back(prototype->layout(), prototype->distribution(),
                               prototype->local_rank(), declaration.ncomp, declaration.ghosts);
            image.back().set_val(Real(0));
          }
          family[subslot].emplace(std::move(image));
        }
      }
    prepared_scratch_.swap(candidate);
    PreparedHotPathWorkspace workspace;
    workspace.bind(prototypes.size(), prototypes.empty() ? 0 : prototypes.front().size(),
                   [&](std::size_t block, std::size_t level) -> const field_type& {
                     if (block >= prototypes.size() || level >= prototypes[block].size() ||
                         prototypes[block][level] == nullptr)
                       throw std::logic_error(
                           "AMR Program forward hot-path workspace has no staged prototype");
                     return *prototypes[block][level];
                   });
    workspace.bind_sum_reduction(
        prototypes.size(), prototypes.empty() ? 0 : prototypes.front().size(),
        [&](std::size_t block, std::size_t level) -> const field_type& {
          if (block >= prototypes.size() || level >= prototypes[block].size() ||
              prototypes[block][level] == nullptr)
            throw std::logic_error("AMR Program forward SUM workspace has no staged prototype");
          return *prototypes[block][level];
        });
    // Forward scratch owns a new workspace, so carry the bind-sealed boundary clock and coupling
    // envelope explicitly; leaving either at its default would allocate on the first hot route.
    workspace.bind_boundary_point_clock(clock_schedule_.primary_clock());
    workspace.bind_coupling_invocation(std::max(hot_path_workspace_.coupling_identity_capacity,
                                                clock_schedule_.primary_clock().size()));
    using std::swap;
    swap(hot_path_workspace_, workspace);
  }

  template <class HistoryManagerType>
  void prepare_forward_history_mutation_workspace_(
      const HistoryManagerType& manager,
      const std::map<std::string, std::vector<FluxExpression>>& forward_flux,
      std::size_t active_levels) {
    std::vector<PreparedHistoryMutationSlot> slots;
    slots.reserve(manager.histories.size());
    constexpr std::string_view rotation_tag = "pops.amr.history-rotate.v2";
    std::size_t maximum_clock_filter_capacity = 0;
    std::size_t rotation_capacity = rotation_tag.size() + 3 * sizeof(std::uint64_t);
    for (const auto& [key, ring] : manager.histories) {
      const auto decoded = decode_history_key_(key);
      const auto level = history_levels_.find(key);
      const auto clock = manager.clock_identity.find(key);
      const auto dts = manager.slot_dt.find(key);
      const auto flux = forward_flux.find(key);
      if (!decoded || level == history_levels_.end() || clock == manager.clock_identity.end() ||
          dts == manager.slot_dt.end() || flux == forward_flux.end() ||
          decoded->first != level->second || ring.empty() || ring.size() != dts->second.size() ||
          ring.size() != flux->second.size())
        throw std::logic_error(
            "AMR Program forward history mutation registry is incomplete at bind");
      PreparedHistoryMutationSlot slot;
      slot.name = std::move(decoded->second);
      slot.key = key;
      slot.clock_identity = clock->second;
      slot.level = level->second;
      slot.rollback_ring.reserve(ring.size());
      for (const field_type& field : ring) {
        slot.rollback_ring.emplace_back(field.layout(), field.distribution(), field.local_rank(),
                                        field.ncomp(), field.ghosts());
        slot.rollback_ring.back().set_val(Real(0));
      }
      slot.dts = dts->second;
      slot.expressions.reserve(flux->second.capacity());
      for (const FluxExpression& source : flux->second) {
        FluxExpression expression;
        for (const auto& [identity, term] : source) {
          if (!term.basis)
            throw std::logic_error(
                "AMR Program forward history mutation has no bindable flux basis");
          auto basis = std::make_shared<FluxBasis>(*term.basis);
          if (!expression.emplace(identity, FluxExpressionTerm{std::move(basis), term.coefficient})
                   .second)
            throw std::logic_error(
                "AMR Program forward history mutation has a duplicate flux occurrence");
        }
        slot.expressions.push_back(std::move(expression));
      }
      const std::size_t store_capacity =
          sizeof("pops.amr.history-store.v2") + slot.key.size() + 6 * sizeof(std::uint64_t);
      slot.store_contract.reserve(store_capacity);
      maximum_clock_filter_capacity =
          std::max(maximum_clock_filter_capacity, slot.clock_identity.size());
      rotation_capacity += slot.key.size() + slot.clock_identity.size() + 3 * sizeof(std::uint64_t);
      slots.push_back(std::move(slot));
    }
    if (slots.size() != history_levels_.size() || slots.size() != forward_flux.size())
      throw std::logic_error(
          "AMR Program forward history mutation registry changed before publication");

    std::vector<std::size_t> history_ordinals;
    history_ordinals.reserve(accepted_state_staging_.history_slot_bindings.size());
    for (const auto& binding : accepted_state_staging_.history_slot_bindings) {
      if (binding.state_slot >= accepted_state_staging_.history_slot_pool.size())
        throw std::logic_error("AMR Program forward history ordinal has an invalid staging slot");
      const int level = accepted_state_staging_.history_slot_pool[binding.state_slot].level;
      if (level < 0)
        throw std::logic_error("AMR Program forward history ordinal has a negative level");
      if (static_cast<std::size_t>(level) >= active_levels) {
        history_ordinals.push_back(std::numeric_limits<std::size_t>::max());
        continue;
      }
      std::size_t match = slots.size();
      for (std::size_t index = 0; index < slots.size(); ++index) {
        if (slots[index].key != binding.key ||
            binding.source_slot >= slots[index].rollback_ring.size())
          continue;
        if (match != slots.size())
          throw std::logic_error("AMR Program forward history ordinal is ambiguous");
        match = index;
      }
      if (match == slots.size())
        throw std::logic_error("AMR Program forward history ordinal has no source");
      history_ordinals.push_back(match);
    }
    std::vector<const AmrProgramPendingHistoryRemap*> pending_ordinals(
        accepted_state_staging_.pending_history_keys.size(), nullptr);
    std::string rotation_contract;
    rotation_contract.reserve(rotation_capacity + maximum_clock_filter_capacity);
    forward_prepared_history_mutation_slots_.swap(slots);
    forward_prepared_history_rotation_contract_.swap(rotation_contract);
    forward_accepted_history_binding_mutation_slots_.swap(history_ordinals);
    forward_accepted_pending_history_ordinal_sources_.swap(pending_ordinals);
  }

  void prepare_forward_execution_bundle(
      const PreparedForwardAmrExecutionAuthority& authority) override {
    if (owner_ != nullptr || !interface_flux_ledger_ || interface_flux_ledger_->in_transaction() ||
        authority.native_dimension() != static_cast<std::uint32_t>(Dim) ||
        authority.topology_epoch() != resource_epoch_ ||
        authority.materialization_generation() != resource_generation_)
      throw std::logic_error(
          "AMR Program detached accepted context has no matching forward execution authority");
    const auto* typed = dynamic_cast<const PreparedForwardAmrExecutionAuthorityView<Dim>*>(
        std::addressof(authority));
    const auto* forward = typed != nullptr ? typed->forward_subcycling() : nullptr;
    const auto* forward_topology = typed != nullptr ? typed->topology().get() : nullptr;
    const auto* storage_envelope = forward_storage_capacity_.get();
    const bool authority_complete =
        forward != nullptr && forward->block_count() != 0 &&
        forward->temporal_relations().size() + 1U == authority.active_level_count() &&
        forward->flux_expression_budget().program_hash == forward->installed_hash() &&
        forward->program_block_map().canonical_indices.size() == forward->block_count() &&
        !forward->interface_flux_ledger_budget().exact_contract.empty() &&
        storage_envelope != nullptr &&
        storage_envelope->configured_level_cell_bounds.size() == storage_envelope->level_count &&
        storage_envelope->configured_forward_storage_counts.level_cell_bounds ==
            storage_envelope->configured_level_cell_bounds &&
        storage_envelope->configured_forward_storage_counts.multifab_value_counts.size() ==
            runtime::program::PreparedAmrForwardStorageCounts::multifab_family_count &&
        storage_envelope->configured_live_subcycling_bytes != 0 &&
        storage_envelope->configured_forward_snapshot_bytes != 0 &&
        storage_envelope->configured_rank_bound != 0 &&
        !storage_envelope->configured_subcycling_storage_contract.empty();
    if (!authority_complete)
      throw std::logic_error(
          "AMR Program detached accepted context has no complete typed forward "
          "subcycling authority (forward=" +
          std::to_string(forward != nullptr) +
          ", envelope=" + std::to_string(storage_envelope != nullptr) + ")");
    if (forward_topology == nullptr || forward_topology->program_state == nullptr)
      throw std::logic_error(
          "AMR Program detached accepted context has no forward history authority");
    // Deferred lag reads occur on the first hot step after publication.  Materialize their
    // numeric scratch nodes here, from the value-owned forward Program image, so both the next
    // accepted snapshot and the live provider have an identical bind-frozen map shape.
    for (const auto& [key, marker] : pending_history_remaps_) {
      (void)marker;
      const auto history = forward_topology->program_state->hist_.histories.find(key);
      if (history == forward_topology->program_state->hist_.histories.end() ||
          history->second.empty())
        throw std::logic_error(
            "AMR Program forward deferred history scratch has no numeric prototype");
      auto [scratch, inserted] =
          deferred_history_lag_scratches_.try_emplace(key, history->second.front());
      (void)inserted;
      require_same_field_contract_(scratch->second, history->second.front(),
                                   "AMR Program forward deferred history scratch");
    }
    if (storage_envelope->configured_live_subcycling_bytes !=
        storage_envelope->configured_forward_storage_bytes.live_subcycling())
      throw std::logic_error(
          "AMR Program detached accepted context has an inconsistent forward storage receipt");
    if (prepared_subcycling_bundle_)
      throw std::logic_error(
          "AMR Program detached accepted context already owns a forward subcycling bundle");
    if (static_cast<std::size_t>(forward->lane().size()) > storage_envelope->configured_rank_bound)
      throw std::length_error(
          "AMR Program forward subcycling rank count exceeds its sealed storage bound");
    using forward_authority_type = std::remove_reference_t<decltype(*forward)>;
    struct ForwardBundleAuthority final {
      const forward_authority_type& forward;
      std::span<const ::pops::amr::ParentChildClockRelation> relations;
      const flux_expression_budget_type* expression_budget = nullptr;
      const typename prepared_multiblock_type::ProgramBlockMap* program_block_map = nullptr;
      ::pops::amr::InterfaceFluxLedgerBudget interface_budget{};
      std::string_view installed_hash;
      BoundaryTopology<Dim> boundary_topology{};
      const hierarchy_type& topology_hierarchy() const { return forward.hierarchy(); }
      const std::vector<Geometry<Dim>>& prepared_level_geometries() const {
        return forward.level_geometries();
      }
      std::size_t block_count() const { return forward.block_count(); }
      std::string_view block_identity(std::size_t block) const {
        return forward.block_identity(block);
      }
      const ExecutionLane& lane() const { return forward.lane(); }
      std::string_view collective_contract() const { return forward.collective_contract(); }
      std::uint64_t topology_epoch() const { return forward.topology_epoch(); }
      std::uint64_t materialization_generation() const {
        return forward.materialization_generation();
      }
      void validate() const {
        if (expression_budget == nullptr || program_block_map == nullptr ||
            installed_hash.empty() || !lane().active() || collective_contract().empty() ||
            block_count() == 0 || program_block_map->canonical_indices.size() != block_count() ||
            program_block_map->hierarchy_contract != collective_contract() ||
            expression_budget->program_hash != installed_hash ||
            expression_budget->generation != materialization_generation() ||
            expression_budget->blocks.size() != block_count() ||
            relations.size() + 1 != topology_hierarchy().num_levels() ||
            interface_budget.exact_contract.empty())
          throw std::invalid_argument("AMR Program forward subcycling authority is incomplete");
      }
      multiblock_subcycling_type prepare_engine(
          std::span<const ::pops::amr::ParentChildClockRelation> value_relations,
          ::pops::numerics::time::amr::MultiBlockAmrSubcyclingBudget budget) const {
        // This is the only forward engine route.  `eventual_owner` remains an anchor internal to
        // prepare_forward and is not consulted for Candidate topology, state, or collectives.
        return multiblock_subcycling_type::prepare_forward(forward, value_relations, budget);
      }
    } subcycling_authority{.forward = *forward,
                           .relations = forward->temporal_relations(),
                           .expression_budget = std::addressof(forward->flux_expression_budget()),
                           .program_block_map = std::addressof(forward->program_block_map()),
                           .interface_budget = forward->interface_flux_ledger_budget(),
                           .installed_hash = forward->installed_hash(),
                           .boundary_topology = forward->boundary_topology()};
    auto bundle = prepare_multiblock_subcycling_bundle_from_authority_(
        subcycling_authority, forward_flux_tables_, hot_path_workspace_,
        clock_schedule_.primary_clock());
    if (!bundle.engine || !bundle.interface_ledger)
      throw std::logic_error(
          "AMR Program detached accepted context prepared an incomplete forward subcycling bundle");
    // The forward bundle owns a new dense static flux carrier.  Retained history provenance
    // must be rebuilt against that carrier while Candidate is still cold; keeping the old
    // topology's face image would make the first accepted store observe a route-shape drift.
    // The exchange remains detached until every collective storage check below has passed.
    auto forward_history_flux =
        AmrStorageTopologyAdapter::prepare_static_history_flux_provenance_from_sealed_history_(
            bundle.flux_tables, bundle.flux_basis_payloads, forward_history_runtime_owners_,
            history_levels_, history_flux_expressions_, clock_schedule_.primary_clock());
    const bool tensor_receipt_active = static_cast<bool>(forward_hierarchy_tensor_selection_);
    const bool tensor_receipt_canonical =
        tensor_receipt_active
            ? storage_envelope->configured_tensor_provider_bytes != 0 &&
                  !storage_envelope->configured_tensor_provider_request_contract.empty() &&
                  !storage_envelope->configured_tensor_provider_limit_contract.empty()
            : storage_envelope->configured_tensor_provider_bytes == 0 &&
                  storage_envelope->configured_tensor_provider_request_contract.empty() &&
                  storage_envelope->configured_tensor_provider_limit_contract.empty();
    if (!tensor_receipt_canonical)
      throw std::logic_error(
          "AMR Program forward tensor provider has a non-canonical configured storage receipt");
    prepare_forward_hierarchy_tensor_solver_(authority);
    if (static_cast<bool>(forward_hierarchy_tensor_solver_) != tensor_receipt_active)
      throw std::logic_error(
          "AMR Program forward tensor selection and prepared provider disagree at bind");
    if (forward_hierarchy_tensor_solver_) {
      const PreparedResidentStorage tensor_storage =
          forward_hierarchy_tensor_solver_->resident_storage();
      if (!tensor_storage.is_exact() || tensor_storage.bytes == 0 ||
          tensor_storage.bytes > storage_envelope->configured_tensor_provider_bytes)
        throw std::logic_error(
            "AMR Program forward tensor provider exceeds its configured storage receipt");
    }
    auto forward_state = bundle.engine->capture_mutable_state_at_bind();
    prepare_forward_accepted_state_staging_(authority);
    rebuild_forward_accepted_effect_slots_(bundle, authority);
    prepare_forward_history_mutation_workspace_(forward_topology->program_state->hist_,
                                                forward_history_flux,
                                                authority.active_level_count());
    seal_forward_accepted_state_staging_();
    for (int axis = 0; axis < Dim; ++axis) {
      const auto dimension = static_cast<std::size_t>(axis);
      auto& ordinals = accepted_face_flux_ordinals_[dimension];
      const std::size_t face_ordinal_capacity =
          accepted_state_staging_.accepted_face_flux_slots[dimension].size();
      ordinals.clear();
      if (ordinals.capacity() < face_ordinal_capacity)
        ordinals.reserve(face_ordinal_capacity);
      ordinals.resize(face_ordinal_capacity);
      for (auto& ordinal : ordinals)
        ordinal = {};
    }
    const std::size_t interface_ordinal_capacity =
        accepted_interface_flux_staging_sources_.capacity();
    if (accepted_interface_flux_wire_ordinals_.capacity() < interface_ordinal_capacity)
      accepted_interface_flux_wire_ordinals_.reserve(interface_ordinal_capacity);
    accepted_interface_flux_wire_ordinals_.resize(interface_ordinal_capacity);
    for (std::size_t ordinal = 0; ordinal < interface_ordinal_capacity; ++ordinal)
      accepted_interface_flux_wire_ordinals_[ordinal] = ordinal;
    // Measure only after every B-owned arena has been cold-built: bundle, mutable rollback image,
    // accepted face/event slots, checkpoint staging and optional tensor provider.  The bundle is
    // still a local value here, so a rejected receipt cannot publish or overwrite A.
    const ExecutionLane& candidate_lane = forward->lane();
    std::exception_ptr ceiling_error;
    std::uint64_t candidate_bytes = 0;
    std::uint64_t candidate_global_bytes = 0;
    std::uint64_t snapshot_peak_bytes = 0;
    std::uint64_t snapshot_global_bytes = 0;
    try {
      candidate_bytes = forward_candidate_resident_storage_bytes_(bundle, forward_state);
      if (candidate_bytes > static_cast<std::uint64_t>(std::numeric_limits<long>::max()))
        throw std::overflow_error(
            "AMR Program forward subcycling candidate exceeds collective byte range");
      if (detached_accepted_snapshot_bytes_ >
          std::numeric_limits<std::uint64_t>::max() - candidate_bytes)
        throw std::overflow_error("AMR Program forward A+B snapshot storage overflows uint64");
      snapshot_peak_bytes = detached_accepted_snapshot_bytes_ + candidate_bytes;
      if (snapshot_peak_bytes > static_cast<std::uint64_t>(std::numeric_limits<long>::max()))
        throw std::overflow_error("AMR Program forward A+B snapshot exceeds collective byte range");
    } catch (...) {
      ceiling_error = std::current_exception();
    }
    if (all_reduce_max(ceiling_error ? 1L : 0L, candidate_lane) != 0) {
      if (candidate_lane.size() == 1 && ceiling_error)
        std::rethrow_exception(ceiling_error);
      throw std::runtime_error(
          "AMR Program forward subcycling candidate storage could not be measured collectively");
    }
    candidate_global_bytes = static_cast<std::uint64_t>(
        all_reduce_max(static_cast<long>(candidate_bytes), candidate_lane));
    snapshot_global_bytes = static_cast<std::uint64_t>(
        all_reduce_max(static_cast<long>(snapshot_peak_bytes), candidate_lane));
    try {
      auto ceiling = *storage_envelope;
      if (test_forward_storage_ceiling_override_) {
        ceiling.configured_live_subcycling_bytes = *test_forward_storage_ceiling_override_;
        ceiling.configured_forward_snapshot_bytes = *test_forward_storage_ceiling_override_;
      }
      require_forward_storage_ceilings_(candidate_global_bytes, snapshot_global_bytes, ceiling);
    } catch (...) {
      ceiling_error = std::current_exception();
    }
    if (all_reduce_max(ceiling_error ? 1L : 0L, candidate_lane) != 0) {
      if (candidate_lane.size() == 1 && ceiling_error)
        std::rethrow_exception(ceiling_error);
      throw std::runtime_error(
          "AMR Program forward subcycling candidate exceeds its storage ceiling collectively");
    }
    ExactContractBuilder candidate_receipt;
    candidate_receipt.text("pops.amr-program.forward-subcycling-storage-receipt.v2")
        .scalar(std::int32_t{Dim})
        .scalar(authority.topology_epoch())
        .scalar(authority.materialization_generation())
        .scalar(candidate_global_bytes)
        .scalar(snapshot_global_bytes)
        .scalar(storage_envelope->configured_live_subcycling_bytes)
        .scalar(storage_envelope->configured_forward_snapshot_bytes)
        .scalar(storage_envelope->configured_tensor_provider_bytes)
        .bytes(storage_envelope->configured_tensor_provider_request_contract)
        .bytes(storage_envelope->configured_tensor_provider_limit_contract)
        .bytes(storage_envelope->configured_subcycling_storage_contract)
        .bytes(forward->collective_contract());
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"amr-program-forward-subcycling-storage", std::move(candidate_receipt).release()}},
            candidate_lane))
      throw std::runtime_error(
          "AMR Program forward subcycling candidate storage receipt differs between ranks");
    forward_subcycling_state_.emplace(std::move(forward_state));
    prepared_subcycling_bundle_.emplace(std::move(bundle));
    using std::swap;
    swap(history_flux_expressions_, forward_history_flux);
    // This snapshot has no DSO-owned closures.  Its complete topology-bound host resources were
    // rematerialized by prepare_forward_scratch_rematerialization above; retain the exact
    // authority witness only after its forward staging image has been cold-prepared.
    forward_execution_bundle_epoch_ = authority.topology_epoch();
    forward_execution_bundle_generation_ = authority.materialization_generation();
  }

  [[nodiscard]] PreparedForwardAmrAcceptedContext prepare_forward_accepted_context(
      std::int64_t accepted_macro_step) const override {
    if (owner_ != nullptr || !interface_flux_ledger_ || interface_flux_ledger_->in_transaction())
      throw std::logic_error(
          "AMR Program forward accepted checkpoint requires one sealed detached context");
    if (forward_execution_bundle_epoch_ != resource_epoch_ ||
        forward_execution_bundle_generation_ != resource_generation_)
      throw std::logic_error(
          "AMR Program forward accepted checkpoint has no prepared execution bundle");
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
      if (!deferred_history_lag_scratches_.contains(key))
        throw std::logic_error(
            "AMR Program detached accepted checkpoint has no prepared deferred scratch");
      if (marker.consumed)
        continue;
      result.pending_history_remaps.push_back(marker);
    }
    if (deferred_history_lag_scratches_.size() != pending_history_remaps_.size())
      throw std::logic_error(
          "AMR Program detached accepted checkpoint has a foreign deferred scratch");
    result.history_flux_payload = serialize_history_flux_payload_(history_flux_expressions_);
    result.temporal_partition = accepted_temporal_partition_;
    result.flux_budget_contract = accepted_flux_budget_contract_;
    result.coupling_contract = accepted_coupling_contract_;
    result.topology_scoped_effects_invalidated =
        std::all_of(accepted_face_flux_.begin(), accepted_face_flux_.end(),
                    [](const auto& axis) { return axis.empty(); }) &&
        accepted_synchronization_events_.empty();
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
    if (owner_->accepted_flux_budget_contract_.empty() ||
        owner_->accepted_coupling_contract_.empty())
      throw std::logic_error(
          "AMR Program accepted context entered cold-prime without temporal contracts");
    // The complete POPSAND5 bootstrap and its capacity witness were prepared by the host before
    // owner-last publication.  This cold-prime path may rebuild finite snapshot carriers, but it
    // must not start a second checkpoint/capacity cycle through the live facade.
    owner_->clock_schedule_.seal_for_execution();
    owner_->bind_accepted_checkpoint_candidate_buffer_();
    owner_->refresh_accepted_hierarchy_state_();
    if (owner_->accepted_flux_budget_contract_.empty() ||
        owner_->accepted_coupling_contract_.empty())
      throw std::logic_error("AMR Program accepted context refresh erased its temporal contracts");
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
    prime_accepted_state_staging_from_cold_staging_(owner_->accepted_state_staging_);
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
    DetachedState staged{
        .clock_schedule = clock_schedule_,
        .resource_epoch = topology_epoch,
        .resource_generation = materialization_generation,
        .history_epoch = topology_epoch,
        .history_generation = materialization_generation,
        // Bootstrap/restart may detach at the already accepted topology solely
        // to rebuild checkpoint carriers.  Keep its cold-bound bundle witness;
        // a true regrid changes either coordinate and deliberately invalidates it
        // until Candidate prepares the new bundle below.
        .forward_execution_bundle_epoch = topology_epoch == resource_epoch_
                                              ? forward_execution_bundle_epoch_
                                              : std::numeric_limits<std::uint64_t>::max(),
        .forward_execution_bundle_generation = materialization_generation == resource_generation_
                                                   ? forward_execution_bundle_generation_
                                                   : std::numeric_limits<std::uint64_t>::max(),
        .detached_accepted_snapshot_bytes = resident_storage_bytes_for_owner(*owner_),
        .test_forward_storage_ceiling_override = owner_->test_forward_storage_ceiling_override_,
        .history_levels = history_levels_,
        .history_runtime_owners = owner_->runtime_state().hist_.owner,
        .history_flux_expressions = clone_history_flux_expressions_cold_(history_flux_expressions_),
        .pending_history_remaps = pending_history_remaps_,
        .deferred_history_lag_scratches = deferred_history_lag_scratches_,
        .prepared_scratch = prepared_scratch_,
        .prepared_scratch_descriptors = prepared_scratch_descriptors_,
        .hot_path_workspace = hot_path_workspace_,
        // The retained snapshot is the accepted authority for this forward handoff.  The live
        // owner may already contain Candidate-produced face/interface effects by the time regrid
        // preparation runs; copying its staging image here would mix generations before
        // HiddenPublish and make the cold envelope depend on unsealed state.
        .accepted_state_staging = accepted_state_staging_,
        .forward_storage_capacity = owner_->accepted_forward_storage_capacity_,
        .forward_flux_tables = owner_->static_flux_tables_,
        .hierarchy_tensor_selection = owner_->hierarchy_tensor_selection_,
        .accepted_face_flux_ordinals = owner_->accepted_face_flux_ordinals_,
        .accepted_interface_flux_staging_sources = accepted_interface_flux_staging_sources_,
        .accepted_interface_flux_wire_ordinals = accepted_interface_flux_wire_ordinals_,
        .accepted_checkpoint_level_clock_slots = accepted_checkpoint_level_clock_slots_,
        .accepted_temporal_partition = accepted_temporal_partition_,
        .cell_temporal_configuration = cell_temporal_configuration_,
        .accepted_flux_budget_contract = accepted_flux_budget_contract_,
        .accepted_coupling_contract = accepted_coupling_contract_,
        .accepted_face_flux = accepted_face_flux_,
        .accepted_face_flux_slots = accepted_face_flux_slots_,
        .interface_flux_ledger =
            std::make_unique<interface_flux_ledger_type>(*interface_flux_ledger_),
        .accepted_synchronization_events = accepted_synchronization_events_,
        .accepted_synchronization_event_slots = accepted_synchronization_event_slots_,
        .multiblock_subcycling_state = multiblock_subcycling_state_,
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
        forward_execution_bundle_epoch_(owner.resource_epoch_),
        forward_execution_bundle_generation_(owner.resource_generation_),
        history_levels_(owner.history_levels_),
        history_flux_expressions_(owner.history_flux_expressions_),
        pending_history_remaps_(owner.pending_history_remaps_),
        deferred_history_lag_scratches_(owner.deferred_history_lag_scratches_),
        prepared_scratch_(owner.prepared_scratch_),
        prepared_scratch_descriptors_(owner.prepared_scratch_descriptors_),
        hot_path_workspace_(owner.hot_path_workspace_),
        accepted_state_staging_(owner.accepted_state_staging_),
        forward_storage_capacity_(owner.accepted_forward_storage_capacity_),
        forward_flux_tables_(owner.static_flux_tables_),
        accepted_face_flux_ordinals_(owner.accepted_face_flux_ordinals_),
        accepted_interface_flux_staging_sources_(owner.accepted_interface_flux_staging_sources_),
        accepted_interface_flux_wire_ordinals_(owner.accepted_interface_flux_wire_ordinals_),
        accepted_temporal_partition_(owner.accepted_temporal_partition_),
        cell_temporal_configuration_(owner.cell_temporal_configuration_),
        accepted_flux_budget_contract_(owner.accepted_flux_budget_contract_),
        accepted_coupling_contract_(owner.accepted_coupling_contract_),
        accepted_face_flux_(owner.accepted_face_flux_),
        interface_flux_ledger_(
            std::make_unique<interface_flux_ledger_type>(*owner.interface_flux_ledger_)),
        accepted_synchronization_events_(owner.accepted_synchronization_events_),
        // Snapshot pools retain the resident commit envelopes, not only the currently logical
        // accepted effects.  A bootstrap may append a level after this image is captured; copying
        // the logical (often empty) vectors here would erase the route/string capacity needed by
        // the detached Candidate.
        accepted_face_flux_slots_(owner.accepted_face_flux_commit_slots_),
        accepted_synchronization_event_slots_(owner.accepted_synchronization_event_commit_slots_),
        accepted_state_revision_(owner.accepted_state_revision_) {
    history_flux_expressions_ =
        clone_history_flux_expressions_cold_(owner.history_flux_expressions_);
    if (owner.multiblock_subcycling_)
      multiblock_subcycling_state_.emplace(
          owner.multiblock_subcycling_->capture_mutable_state_at_bind());
  }

  AcceptedContextSnapshot(const AcceptedContextSnapshot& accepted)
      : owner_(accepted.owner_),
        clock_schedule_(accepted.clock_schedule_),
        resource_epoch_(accepted.resource_epoch_),
        resource_generation_(accepted.resource_generation_),
        history_epoch_(accepted.history_epoch_),
        history_generation_(accepted.history_generation_),
        forward_execution_bundle_epoch_(accepted.forward_execution_bundle_epoch_),
        forward_execution_bundle_generation_(accepted.forward_execution_bundle_generation_),
        detached_accepted_snapshot_bytes_(accepted.detached_accepted_snapshot_bytes_),
        test_forward_storage_ceiling_override_(accepted.test_forward_storage_ceiling_override_),
        history_levels_(accepted.history_levels_),
        history_flux_expressions_(accepted.history_flux_expressions_),
        pending_history_remaps_(accepted.pending_history_remaps_),
        deferred_history_lag_scratches_(accepted.deferred_history_lag_scratches_),
        prepared_scratch_(accepted.prepared_scratch_),
        prepared_scratch_descriptors_(accepted.prepared_scratch_descriptors_),
        hot_path_workspace_(accepted.hot_path_workspace_),
        accepted_state_staging_(accepted.accepted_state_staging_),
        forward_storage_capacity_(accepted.forward_storage_capacity_),
        forward_flux_tables_(accepted.forward_flux_tables_),
        forward_hierarchy_tensor_selection_(accepted.forward_hierarchy_tensor_selection_),
        accepted_face_flux_ordinals_(accepted.accepted_face_flux_ordinals_),
        accepted_interface_flux_staging_sources_(accepted.accepted_interface_flux_staging_sources_),
        accepted_interface_flux_wire_ordinals_(accepted.accepted_interface_flux_wire_ordinals_),
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
        multiblock_subcycling_state_(accepted.multiblock_subcycling_state_),
        forward_subcycling_state_(accepted.forward_subcycling_state_),
        prepared_subcycling_bundle_published_(accepted.prepared_subcycling_bundle_published_),
        forward_prepared_history_mutation_slots_(accepted.forward_prepared_history_mutation_slots_),
        forward_prepared_history_rotation_contract_(
            accepted.forward_prepared_history_rotation_contract_),
        forward_accepted_history_binding_mutation_slots_(
            accepted.forward_accepted_history_binding_mutation_slots_),
        forward_accepted_pending_history_ordinal_sources_(
            accepted.forward_accepted_pending_history_ordinal_sources_),
        accepted_state_revision_(accepted.accepted_state_revision_) {
    history_flux_expressions_ =
        clone_history_flux_expressions_cold_(accepted.history_flux_expressions_);
  }

  std::unique_ptr<AcceptedProgramExecutionServicesSnapshot> prepare_restore() const override {
    // This is a preparation boundary: the clone will later be swapped into the owner by the
    // no-throw restore path.  Rebuild its already-authenticated spare capacity here, never while
    // refreshing, publishing or finalizing an attempt.
    auto prepared = std::make_unique<AcceptedContextSnapshot>(*this);
    const PreparedHotPathWorkspace* prepared_workspace_source = std::addressof(hot_path_workspace_);
    if (prepared_subcycling_bundle_) {
      if (!prepared_subcycling_bundle_->engine)
        throw std::logic_error(
            "AMR Program detached accepted context lost its prepared forward subcycling engine");
      prepared->multiblock_subcycling_state_.emplace(
          prepared_subcycling_bundle_->engine->capture_mutable_state_at_bind());
      // The forward bundle owns the workspace which will become live at HiddenPublish.  The
      // successor accepted image must be cold-primed from that exact image now: copying this
      // detached snapshot's pre-forward workspace would leave its rollback route vector empty,
      // then the next step would discover the newly published metric reflux routes after the
      // writer lease was acquired.  This is intentionally cold construction; publication stays
      // a no-throw exchange and hot capture only verifies the sealed shape.
      PreparedHotPathWorkspace forward_workspace = prepared_subcycling_bundle_->hot_path_workspace;
      forward_workspace.prime_copied_capacities_from_cold_source(
          prepared_subcycling_bundle_->hot_path_workspace);
      using std::swap;
      swap(prepared->hot_path_workspace_, forward_workspace);
      prepared_workspace_source = std::addressof(prepared_subcycling_bundle_->hot_path_workspace);
    }
    prepared->prime_copied_capacities_from_cold_source_(*this, *prepared_workspace_source);
    prepared->prime_accepted_state_staging_from_cold_source_(*this);
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
    if (multiblock_subcycling_state_) {
      if (!owner_->multiblock_subcycling_)
        throw std::logic_error("AMR Program accepted context lost its prepared subcycling engine");
      owner_->multiblock_subcycling_->copy_mutable_state_into_preallocated(
          *multiblock_subcycling_state_);
    } else if (owner_->multiblock_subcycling_) {
      throw std::logic_error("AMR Program accepted context gained an unprimed subcycling engine");
    }
    accepted_state_revision_ = owner_->accepted_state_revision_;
    // The cell-temporal diagnostic rollback pool is distinct from the accepted and candidate
    // pools.  Snapshot it only after every fallible shape/capacity check and in-place copy has
    // completed, so a rejected Snapshot cannot expose a partially refreshed public image.
    snapshot_transaction_diagnostics_noexcept();
  }

  void snapshot_transaction_diagnostics_noexcept() noexcept override {
    if (owner_ == nullptr)
      std::terminate();
    owner_->snapshot_cell_temporal_diagnostics_noexcept();
  }

  void publish_transaction_diagnostics_noexcept() noexcept override {
    if (owner_ == nullptr)
      std::terminate();
    owner_->publish_cell_temporal_diagnostics_noexcept();
  }

  void restore_transaction_diagnostics_noexcept() noexcept override {
    if (owner_ == nullptr)
      std::terminate();
    owner_->restore_cell_temporal_diagnostics_noexcept();
  }

  void publish_restore() noexcept override {
    static_assert(std::is_nothrow_swappable_v<ClockScheduleState>);
    static_assert(std::is_nothrow_swappable_v<decltype(history_levels_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(history_flux_expressions_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(pending_history_remaps_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(deferred_history_lag_scratches_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(prepared_scratch_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(prepared_scratch_descriptors_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(hot_path_workspace_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(accepted_state_staging_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(forward_prepared_history_mutation_slots_)>);
    static_assert(
        std::is_nothrow_swappable_v<decltype(forward_prepared_history_rotation_contract_)>);
    static_assert(
        std::is_nothrow_swappable_v<decltype(forward_accepted_history_binding_mutation_slots_)>);
    static_assert(
        std::is_nothrow_swappable_v<decltype(forward_accepted_pending_history_ordinal_sources_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(accepted_face_flux_ordinals_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(accepted_interface_flux_staging_sources_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(accepted_interface_flux_wire_ordinals_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(accepted_checkpoint_level_clock_slots_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(accepted_temporal_partition_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(cell_temporal_configuration_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(accepted_flux_budget_contract_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(accepted_coupling_contract_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(accepted_face_flux_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(interface_flux_ledger_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(accepted_synchronization_events_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(discarded_scratches_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(discarded_subcycling_contract_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(resource_epoch_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(resource_generation_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(history_epoch_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(history_generation_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(accepted_state_revision_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(forward_hierarchy_tensor_solver_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(forward_hierarchy_tensor_boundaries_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(forward_hierarchy_tensor_topology_epoch_)>);
    static_assert(std::is_nothrow_swappable_v<
                  decltype(forward_hierarchy_tensor_materialization_generation_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(interface_flux_commit_guard_)>);
    if (owner_ == nullptr || !interface_flux_ledger_ || interface_flux_ledger_->in_transaction())
      std::terminate();
    std::swap(owner_->clock_schedule_, clock_schedule_);
    std::swap(owner_->resource_epoch_, resource_epoch_);
    std::swap(owner_->resource_generation_, resource_generation_);
    std::swap(owner_->history_epoch_, history_epoch_);
    std::swap(owner_->history_generation_, history_generation_);
    owner_->history_levels_.swap(history_levels_);
    owner_->history_flux_expressions_.swap(history_flux_expressions_);
    owner_->pending_history_remaps_.swap(pending_history_remaps_);
    owner_->deferred_history_lag_scratches_.swap(deferred_history_lag_scratches_);
    owner_->prepared_scratch_.swap(prepared_scratch_);
    owner_->prepared_scratch_descriptors_.swap(prepared_scratch_descriptors_);
    if (!prepared_subcycling_bundle_)
      std::swap(owner_->hot_path_workspace_, hot_path_workspace_);
    if (accepted_state_staging_.prepared_envelope)
      std::swap(owner_->accepted_state_staging_, accepted_state_staging_);
    owner_->accepted_face_flux_ordinals_.swap(accepted_face_flux_ordinals_);
    std::swap(owner_->accepted_interface_flux_staging_sources_,
              accepted_interface_flux_staging_sources_);
    owner_->accepted_interface_flux_wire_ordinals_.swap(accepted_interface_flux_wire_ordinals_);
    owner_->accepted_checkpoint_level_clock_slots_.swap(accepted_checkpoint_level_clock_slots_);
    owner_->scratches_.swap(discarded_scratches_);
    if (prepared_subcycling_bundle_) {
      if (!prepared_subcycling_bundle_->engine || !prepared_subcycling_bundle_->interface_ledger)
        std::terminate();
      if (static_cast<bool>(forward_hierarchy_tensor_selection_) !=
              static_cast<bool>(owner_->hierarchy_tensor_selection_) ||
          (forward_hierarchy_tensor_selection_ &&
           forward_hierarchy_tensor_selection_->exact_contract !=
               owner_->hierarchy_tensor_selection_->exact_contract))
        std::terminate();
      owner_->publish_prepared_subcycling_bundle_noexcept(std::move(*prepared_subcycling_bundle_));
      owner_->prepared_history_mutation_slots_.swap(forward_prepared_history_mutation_slots_);
      owner_->prepared_history_rotation_contract_.swap(forward_prepared_history_rotation_contract_);
      owner_->accepted_history_binding_mutation_slots_.swap(
          forward_accepted_history_binding_mutation_slots_);
      owner_->accepted_pending_history_ordinal_sources_.swap(
          forward_accepted_pending_history_ordinal_sources_);
      // The adapter owns the non-owning ledger/wire ordinals.  Rebind them against the newly
      // published bundle before its displaced predecessor can be released; this path is strictly
      // preallocated/noexcept because the aggregate publication is already visible.
      owner_->rebind_accepted_face_flux_ordinals_preallocated_noexcept_();
      owner_->rebind_accepted_interface_flux_ordinals_preallocated_noexcept_();
      const auto& mutable_state = prepared_subcycling_bundle_published_
                                      ? multiblock_subcycling_state_
                                      : forward_subcycling_state_;
      if (!mutable_state || !owner_->multiblock_subcycling_)
        std::terminate();
      owner_->multiblock_subcycling_->restore_mutable_state_from_preallocated(*mutable_state);
      prepared_subcycling_bundle_published_ = !prepared_subcycling_bundle_published_;
    } else if (multiblock_subcycling_state_ && owner_->multiblock_subcycling_) {
      owner_->multiblock_subcycling_->restore_mutable_state_from_preallocated(
          *multiblock_subcycling_state_);
    } else {
      if (owner_->multiblock_subcycling_)
        std::terminate();
    }
    if (prepared_subcycling_bundle_) {
      owner_->hierarchy_tensor_solver_.swap(forward_hierarchy_tensor_solver_);
      owner_->hierarchy_tensor_boundaries_.swap(forward_hierarchy_tensor_boundaries_);
      std::swap(owner_->hierarchy_tensor_topology_epoch_, forward_hierarchy_tensor_topology_epoch_);
      std::swap(owner_->hierarchy_tensor_materialization_generation_,
                forward_hierarchy_tensor_materialization_generation_);
    } else {
      // A cold accepted restore recreates the host PreparedHierarchy after this publication.  The
      // current hierarchy-tensor solver and its boundary sessions pin that graph's ExecutionLane;
      // release those topology-bound borrows before the host destroys the graph.  Keep the sealed
      // selection so configured_hierarchy_tensor_solver_() can prepare the same provider against
      // the restored hierarchy on its next use.
      owner_->hierarchy_tensor_solver_.reset();
      owner_->hierarchy_tensor_boundaries_.clear();
      owner_->hierarchy_tensor_topology_epoch_ = std::numeric_limits<std::uint64_t>::max();
      owner_->hierarchy_tensor_materialization_generation_ =
          std::numeric_limits<std::uint64_t>::max();
    }
    std::swap(owner_->accepted_temporal_partition_, accepted_temporal_partition_);
    std::swap(owner_->cell_temporal_configuration_, cell_temporal_configuration_);
    owner_->accepted_flux_budget_contract_.swap(accepted_flux_budget_contract_);
    owner_->accepted_coupling_contract_.swap(accepted_coupling_contract_);
    std::swap(owner_->accepted_face_flux_, accepted_face_flux_);
    std::swap(owner_->accepted_face_flux_commit_slots_, accepted_face_flux_slots_);
    owner_->interface_flux_commit_guard_.swap(interface_flux_commit_guard_);
    if (!prepared_subcycling_bundle_)
      owner_->interface_flux_ledger_.swap(interface_flux_ledger_);
    owner_->accepted_synchronization_events_.swap(accepted_synchronization_events_);
    owner_->accepted_synchronization_event_commit_slots_.swap(
        accepted_synchronization_event_slots_);
    std::swap(owner_->accepted_state_revision_, accepted_state_revision_);
    owner_->rebind_history_mutation_workspace_preallocated_after_restore_();
    if (!prepared_subcycling_bundle_) {
      // A topology-static restore keeps the live engine but swaps the owner/staging slot images.
      // Rebuild both adapter-owned ordinal views only after those no-throw swaps: face ordinals
      // retain ledger pointers, while interface ordinals retain the dense source permutation.
      owner_->rebind_accepted_face_flux_ordinals_preallocated_noexcept_();
      owner_->rebind_accepted_interface_flux_ordinals_preallocated_noexcept_();
    }
    for (const auto& diagnostic : owner_->cell_temporal_diagnostics_)
      if (diagnostic)
        diagnostic->invalidate_accepted_publication(
            owner_->accepted_temporal_partition_.synchronization_tick,
            owner_->accepted_temporal_partition_.tick_denominator);
  }

 private:
  /// Shared final gate for both production Candidate preparation and the focused no-clobber
  /// regression.  B is compared only with the live-engine ceiling; the independently measured
  /// detached A+B peak is compared only with the snapshot ceiling.
  // clang-format off
#include <pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_forward.hpp>
  // clang-format on

  // clang-format off
#include <pops/runtime/program/detail/program_execution_services_amr_history_checkpoint_history_reseed.hpp>
  // clang-format on

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
    prepared.prime_accepted_state_staging_from_cold_staging_(owner_->accepted_state_staging_);

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
    swap(prepared_scratch_, prepared.prepared_scratch_);
    swap(prepared_scratch_descriptors_, prepared.prepared_scratch_descriptors_);
    swap(hot_path_workspace_, prepared.hot_path_workspace_);
    swap(accepted_state_staging_, prepared.accepted_state_staging_);
    swap(accepted_face_flux_ordinals_, prepared.accepted_face_flux_ordinals_);
    swap(accepted_interface_flux_staging_sources_,
         prepared.accepted_interface_flux_staging_sources_);
    swap(accepted_interface_flux_wire_ordinals_, prepared.accepted_interface_flux_wire_ordinals_);
    swap(accepted_checkpoint_level_clock_slots_, prepared.accepted_checkpoint_level_clock_slots_);
    swap(accepted_temporal_partition_, prepared.accepted_temporal_partition_);
    swap(cell_temporal_configuration_, prepared.cell_temporal_configuration_);
    swap(accepted_flux_budget_contract_, prepared.accepted_flux_budget_contract_);
    swap(accepted_coupling_contract_, prepared.accepted_coupling_contract_);
    swap(accepted_face_flux_, prepared.accepted_face_flux_);
    swap(interface_flux_ledger_, prepared.interface_flux_ledger_);
    swap(accepted_synchronization_events_, prepared.accepted_synchronization_events_);
    swap(accepted_face_flux_slots_, prepared.accepted_face_flux_slots_);
    swap(accepted_synchronization_event_slots_, prepared.accepted_synchronization_event_slots_);
    swap(multiblock_subcycling_state_, prepared.multiblock_subcycling_state_);
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
  std::uint64_t forward_execution_bundle_epoch_ = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t forward_execution_bundle_generation_ = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t detached_accepted_snapshot_bytes_ = 0;
  std::optional<std::uint64_t> test_forward_storage_ceiling_override_;
  std::map<std::string, int> history_levels_;
  std::map<std::string, int> forward_history_runtime_owners_;
  std::map<std::string, std::vector<FluxExpression>> history_flux_expressions_;
  std::map<std::string, AmrProgramPendingHistoryRemap> pending_history_remaps_;
  std::map<std::string, field_type> deferred_history_lag_scratches_;
  PreparedScratchStorage prepared_scratch_;
  PreparedScratchDescriptors prepared_scratch_descriptors_;
  PreparedHotPathWorkspace hot_path_workspace_;
  AcceptedStateStaging<Dim> accepted_state_staging_;
  std::shared_ptr<const AmrProgramAcceptedStateStagingCapacity<Dim>> forward_storage_capacity_;
  PreparedFluxTableCarrier forward_flux_tables_;
  std::optional<PreparedSubcyclingBundle> prepared_subcycling_bundle_;
  std::optional<HierarchyTensorSelection> forward_hierarchy_tensor_selection_;
  /// A forward publication must leave the prior tensor provider owned by this detached image.
  /// The forward slot is intentionally empty unless Candidate cold-prepared a replacement.
  std::unique_ptr<hierarchy_tensor_solver_type> forward_hierarchy_tensor_solver_;
  std::vector<HierarchyTensorLevelBoundary> forward_hierarchy_tensor_boundaries_;
  std::uint64_t forward_hierarchy_tensor_topology_epoch_ =
      std::numeric_limits<std::uint64_t>::max();
  std::uint64_t forward_hierarchy_tensor_materialization_generation_ =
      std::numeric_limits<std::uint64_t>::max();
  std::array<std::vector<AcceptedFaceFluxOrdinal>, Dim> accepted_face_flux_ordinals_;
  std::vector<typename AcceptedStateStaging<Dim>::InterfaceFluxSerializationView>
      accepted_interface_flux_staging_sources_;
  std::vector<std::size_t> accepted_interface_flux_wire_ordinals_;
  std::vector<::pops::amr::ClockStamp> accepted_checkpoint_level_clock_slots_;
  CellTemporalPartitionAcceptedState accepted_temporal_partition_;
  std::optional<CellTemporalConfiguration> cell_temporal_configuration_;
  std::string accepted_flux_budget_contract_;
  std::string accepted_coupling_contract_;
  std::array<std::vector<::pops::amr::reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>, Dim>
      accepted_face_flux_;
  std::unique_ptr<interface_flux_ledger_type> interface_flux_ledger_;
  std::optional<typename interface_flux_ledger_type::PreparedCommit> interface_flux_commit_guard_;
  std::vector<AmrProgramSynchronizationEvent> accepted_synchronization_events_;
  /// Cold-resident string/payload slots for accepted effects.  The public vectors retain their
  /// logical size; these slots retain the identity/capacity envelope needed when that size later
  /// contracts and grows again during a topology-static candidate.
  std::array<std::vector<::pops::amr::reflux::FaceFluxFragment<Dim, AmrProgramFacePayload>>, Dim>
      accepted_face_flux_slots_;
  std::vector<AmrProgramSynchronizationEvent> accepted_synchronization_event_slots_;
  std::optional<typename multiblock_subcycling_type::MutableStateImage>
      multiblock_subcycling_state_;
  std::optional<typename multiblock_subcycling_type::MutableStateImage> forward_subcycling_state_;
  bool prepared_subcycling_bundle_published_ = false;
  std::vector<PreparedHistoryMutationSlot> forward_prepared_history_mutation_slots_;
  std::string forward_prepared_history_rotation_contract_;
  std::vector<std::size_t> forward_accepted_history_binding_mutation_slots_;
  std::vector<const AmrProgramPendingHistoryRemap*>
      forward_accepted_pending_history_ordinal_sources_;
  std::uint64_t accepted_state_revision_ = std::numeric_limits<std::uint64_t>::max();
  std::map<ScratchKey, field_type> discarded_scratches_;
  std::string discarded_subcycling_contract_;
};

[[nodiscard]] std::uint64_t history_effects_resident_storage_bytes_(
    bool include_checkpoint_staging) const {
  const auto checked_add = [](std::uint64_t& total, std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - total)
      throw std::overflow_error("AMR Program history/effects storage overflows uint64");
    total += value;
  };
  const auto logical_map_bytes = [](const auto& values) -> std::uint64_t {
    using value_type = typename std::remove_reference_t<decltype(values)>::value_type;
    if (values.size() > std::numeric_limits<std::uint64_t>::max() / sizeof(value_type))
      throw std::overflow_error("AMR Program history map storage overflows uint64");
    return static_cast<std::uint64_t>(values.size()) * sizeof(value_type);
  };
  const auto vector_bytes = [](const auto& values) -> std::uint64_t {
    using value_type = typename std::remove_reference_t<decltype(values)>::value_type;
    if (values.capacity() > std::numeric_limits<std::uint64_t>::max() / sizeof(value_type))
      throw std::overflow_error("AMR Program history vector storage overflows uint64");
    return static_cast<std::uint64_t>(values.capacity()) * sizeof(value_type);
  };
  const auto string_bytes = [](const std::string& value) -> std::uint64_t {
    const auto begin = reinterpret_cast<std::uintptr_t>(&value);
    const auto end = begin + sizeof(value);
    const auto data = reinterpret_cast<std::uintptr_t>(value.data());
    if (data >= begin && data < end)
      return 0;
    if (value.capacity() == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("AMR Program history string storage overflows uint64");
    return static_cast<std::uint64_t>(value.capacity()) + 1U;
  };

  std::uint64_t total = 0;
  checked_add(total, logical_map_bytes(history_levels_));
  for (const auto& [key, level] : history_levels_) {
    (void)level;
    checked_add(total, string_bytes(key));
  }

  checked_add(total, logical_map_bytes(history_flux_expressions_));
  const auto static_history_basis_bytes = [&](const FluxBasis& basis) {
    std::uint64_t bytes = sizeof(FluxBasis);
    checked_add(bytes, string_bytes(basis.point.clock));
    checked_add(bytes, string_bytes(basis.point.graph_identity));
    checked_add(bytes, string_bytes(basis.point.rate_identity));
    checked_add(bytes, string_bytes(basis.point.application_identity));
    checked_add(bytes, vector_bytes(basis.faces));
    for (const FluxBasisFace& face : basis.faces)
      checked_add(bytes, vector_bytes(face.flux_density));
    return bytes;
  };
  for (const auto& [key, expressions] : history_flux_expressions_) {
    checked_add(total, string_bytes(key));
    checked_add(total, vector_bytes(expressions));
    for (const FluxExpression& expression : expressions) {
      checked_add(total, logical_map_bytes(expression));
      for (const auto& [basis_slot, term] : expression) {
        (void)basis_slot;
        checked_add(total, logical_map_bytes(term.coefficient));
        // Dynamic provenance can share a basis owned by the active expression registry.  The
        // static route instead owns one cold-primed FluxBasis image per history/slot, so charge
        // that retained payload here rather than pretending that the shared handle is its data.
        if (static_flux_tables_.bound && term.basis)
          checked_add(total, static_history_basis_bytes(*term.basis));
      }
    }
  }

  checked_add(total, logical_map_bytes(pending_history_remaps_));
  for (const auto& [key, remap] : pending_history_remaps_) {
    checked_add(total, string_bytes(key));
    checked_add(total, string_bytes(remap.key));
  }

  checked_add(total, logical_map_bytes(deferred_history_lag_scratches_));
  for (const auto& [key, scratch] : deferred_history_lag_scratches_) {
    checked_add(total, string_bytes(key));
    checked_add(total, scratch.resident_storage_bytes());
  }

  checked_add(total, string_bytes(accepted_flux_budget_contract_));
  checked_add(total, string_bytes(accepted_coupling_contract_));
  const auto add_face_flux = [&](const auto& axes) {
    for (const auto& fragments : axes) {
      checked_add(total, vector_bytes(fragments));
      for (const auto& fragment : fragments) {
        checked_add(total, string_bytes(fragment.key.owner));
        checked_add(total, string_bytes(fragment.key.state));
        checked_add(total, string_bytes(fragment.key.stage));
        checked_add(total, vector_bytes(fragment.payload));
      }
    }
  };
  add_face_flux(accepted_face_flux_);
  add_face_flux(accepted_face_flux_commit_slots_);
  const auto add_events = [&](const auto& events) {
    checked_add(total, vector_bytes(events));
    for (const auto& event : events)
      checked_add(total, string_bytes(event.phase));
  };
  add_events(accepted_synchronization_events_);
  add_events(accepted_synchronization_event_commit_slots_);
  checked_add(total, vector_bytes(prepared_history_mutation_slots_));
  for (const auto& slot : prepared_history_mutation_slots_) {
    checked_add(total, string_bytes(slot.name));
    checked_add(total, string_bytes(slot.key));
    checked_add(total, string_bytes(slot.clock_identity));
    checked_add(total, vector_bytes(slot.rollback_ring));
    for (const auto& field : slot.rollback_ring)
      checked_add(total, field.resident_storage_bytes());
    checked_add(total, vector_bytes(slot.dts));
    checked_add(total, vector_bytes(slot.expressions));
    checked_add(total, string_bytes(slot.store_contract));
  }
  checked_add(total, string_bytes(prepared_history_rotation_contract_));
  checked_add(total, vector_bytes(accepted_history_binding_mutation_slots_));
  checked_add(total, vector_bytes(accepted_pending_history_ordinal_sources_));
  for (const auto& ordinals : accepted_face_flux_ordinals_)
    checked_add(total, vector_bytes(ordinals));
  checked_add(total, vector_bytes(accepted_interface_flux_wire_ordinals_));

  if (include_checkpoint_staging) {
    checked_add(total, vector_bytes(accepted_checkpoint_candidate_bytes_));
    checked_add(total, vector_bytes(accepted_checkpoint_level_clock_slots_));
    const auto& staging = accepted_state_staging_;
    const auto& state = staging.state;
    checked_add(total, string_bytes(state.spatial_contract));
    checked_add(total, vector_bytes(state.level_clocks));
    checked_add(total, logical_map_bytes(state.logical_clock_ticks));
    for (const auto& [clock, tick] : state.logical_clock_ticks) {
      (void)tick;
      checked_add(total, string_bytes(clock));
    }
    checked_add(total, vector_bytes(state.histories));
    for (const auto& history : state.histories) {
      checked_add(total, string_bytes(history.name));
      checked_add(total, string_bytes(history.state_identity));
      checked_add(total, string_bytes(history.space_identity));
      checked_add(total, string_bytes(history.clock_identity));
      checked_add(total, string_bytes(history.interpolation_identity));
    }
    checked_add(total, vector_bytes(state.history_slots));
    for (const auto& slot : state.history_slots)
      checked_add(total, string_bytes(slot.name));
    checked_add(total, vector_bytes(state.pending_history_remaps));
    for (const auto& remap : state.pending_history_remaps)
      checked_add(total, string_bytes(remap.key));
    checked_add(total, vector_bytes(state.history_flux_payload));
    checked_add(total, string_bytes(state.temporal_partition.provider_identity));
    checked_add(total, vector_bytes(state.temporal_partition.cells));
    checked_add(total, vector_bytes(state.tagging_hysteresis_state));
    checked_add(total, string_bytes(state.flux_budget_contract));
    checked_add(total, string_bytes(state.coupling_contract));
    add_face_flux(state.accepted_face_flux);
    checked_add(total, vector_bytes(state.accepted_interface_flux));
    for (const auto& fragment : state.accepted_interface_flux) {
      checked_add(total, string_bytes(fragment.key.interface_identity));
      checked_add(total, string_bytes(fragment.key.stage_identity));
      checked_add(total, string_bytes(fragment.key.graph_identity));
      checked_add(total, string_bytes(fragment.key.rate_identity));
      checked_add(total, string_bytes(fragment.key.application_identity));
      checked_add(total, vector_bytes(fragment.payload));
    }
    add_events(state.synchronization_events);
    checked_add(total, vector_bytes(staging.history_slot_bindings));
    for (const auto& binding : staging.history_slot_bindings)
      checked_add(total, string_bytes(binding.key));
    checked_add(total, vector_bytes(staging.history_slot_pool));
    for (const auto& slot : staging.history_slot_pool)
      checked_add(total, string_bytes(slot.name));
    checked_add(total, vector_bytes(staging.history_slot_active_indices));
    checked_add(total, vector_bytes(staging.pending_history_keys));
    for (const auto& key : staging.pending_history_keys)
      checked_add(total, string_bytes(key));
    checked_add(total, vector_bytes(staging.pending_history_remap_slots));
    for (const auto& remap : staging.pending_history_remap_slots)
      checked_add(total, string_bytes(remap.key));
    checked_add(total, vector_bytes(staging.pending_history_active_slots));
    add_face_flux(staging.accepted_face_flux_slots);
    for (const auto& sources : staging.accepted_face_flux_sources)
      checked_add(total, vector_bytes(sources));
    for (const auto& active_slots : staging.accepted_face_flux_active_slots)
      checked_add(total, vector_bytes(active_slots));
    add_events(staging.synchronization_event_slots);
    checked_add(total, vector_bytes(staging.synchronization_event_active_indices));
    checked_add(total, vector_bytes(staging.accepted_interface_flux_slots));
    for (const auto& fragment : staging.accepted_interface_flux_slots) {
      checked_add(total, string_bytes(fragment.key.interface_identity));
      checked_add(total, string_bytes(fragment.key.stage_identity));
      checked_add(total, string_bytes(fragment.key.graph_identity));
      checked_add(total, string_bytes(fragment.key.rate_identity));
      checked_add(total, string_bytes(fragment.key.application_identity));
      checked_add(total, vector_bytes(fragment.payload));
    }
    checked_add(total, vector_bytes(staging.accepted_interface_flux_active_slots));
    checked_add(total, vector_bytes(accepted_interface_flux_staging_sources_));
  }
  return total;
}

[[nodiscard]] std::uint64_t accepted_context_snapshot_resident_storage_bytes_() const {
  return AcceptedContextSnapshot::resident_storage_bytes_for_owner(*this);
}

std::unique_ptr<AcceptedProgramExecutionServicesSnapshot> capture_accepted_context_snapshot_()
    const {
  if (!active_attempt_states_.empty() || !interface_flux_ledger_ ||
      interface_flux_ledger_->in_transaction())
    throw std::logic_error("AMR Program accepted context snapshot crossed an active attempt");
  return std::make_unique<AcceptedContextSnapshot>(*const_cast<AmrStorageTopologyAdapter*>(this));
}
