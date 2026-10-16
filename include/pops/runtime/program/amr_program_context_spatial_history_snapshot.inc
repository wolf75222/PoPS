// Per-macro-attempt source images for composite History@2. Declarations are
// install-time and idempotent; source buffers and lifecycle are never borrowed
// from an external transaction that may encompass several macro steps.
public:
void declare_spatial_interaction_history_source(
    std::int64_t value_id, int program_block, const std::string& name, int lag,
    std::string_view state, std::string_view space, std::string_view clock,
    std::string_view interpolation, std::uint64_t max_bytes, std::string_view identity) const {
  InteractionHistoryDeclaration declaration{sys_block(program_block),
                                            name,
                                            lag,
                                            std::string(state),
                                            std::string(space),
                                            std::string(clock),
                                            std::string(interpolation),
                                            max_bytes,
                                            std::string(identity)};
  if (value_id < 0 || declaration.owner < 0 || name.empty() || lag < 1 || state.empty() ||
      space.empty() || clock.empty() || interpolation.empty() || max_bytes == 0 || identity.empty())
    throw std::invalid_argument("interaction history declaration has incomplete authority");
  const auto found = interaction_history_declarations_.find(value_id);
  if (found != interaction_history_declarations_.end()) {
    if (found->second != declaration)
      throw std::invalid_argument("interaction history SSA declaration changed");
    return;
  }
  if (interaction_history_declarations_sealed_)
    throw std::logic_error("interaction history source registry is sealed");
  for (const auto& [id, previous] : interaction_history_declarations_) {
    (void)id;
    if (previous.owner == declaration.owner && previous.name == declaration.name &&
        (previous.state != declaration.state || previous.space != declaration.space ||
         previous.clock != declaration.clock ||
         previous.interpolation != declaration.interpolation))
      throw std::invalid_argument("interaction history source declarations disagree");
  }
  interaction_history_declarations_.emplace(value_id, std::move(declaration));
}

private:
void revoke_interaction_history_snapshots_() const noexcept {
  interaction_history_snapshots_.clear();
  temporal_field_images_.clear();
}

// Global physical contract, rank-local allocation cost. All arithmetic is
// checked, and every ring slot's grown allocation is included in source cost.
static std::size_t interaction_history_field_bytes_(const field_type& field,
                                                    const ExecutionLane& lane,
                                                    ExactContractBuilder& exact) {
  const auto& ranks = field.distribution().rank_space();
  if (field.ncomp() <= 0 || ranks.size() != static_cast<std::size_t>(lane.size()) ||
      ranks.linear_rank(field.local_rank()) != static_cast<std::size_t>(lane.rank()))
    throw std::invalid_argument("interaction history has foreign ranked storage");
  exact.scalar(field.ncomp())
      .scalar(field.layout().size())
      .scalar(field.distribution().replicated());
  for (int axis = 0; axis < Dim; ++axis)
    exact.scalar(field.ghosts()[axis]).scalar(ranks.origin()[axis]).scalar(ranks.extent()[axis]);
  for (std::size_t global = 0; global < field.layout().size(); ++global) {
    for (int axis = 0; axis < Dim; ++axis) {
      exact.scalar(field.layout()[global].lo[axis]).scalar(field.layout()[global].hi[axis]);
      if (!field.distribution().replicated())
        exact.scalar(field.distribution().owner(global)[axis]);
    }
  }
  std::size_t bytes = sizeof(field_type);
  bytes = interaction_add(
      bytes, interaction_product(field.layout().size(),
                                 2 * sizeof(Box<Dim>) + sizeof(Index<Dim>) + sizeof(std::size_t)));
  bytes = interaction_add(
      bytes, interaction_product(field.local_size(),
                                 sizeof(typename field_type::fab_type) + sizeof(std::size_t)));
  for (std::size_t local = 0; local < field.local_size(); ++local)
    bytes = interaction_add(
        bytes, interaction_product(
                   interaction_product(field.fab(local).grown_box().numPts(), field.ncomp()),
                   sizeof(Real)));
  return bytes;
}

void prepare_interaction_history_snapshots_(double dt) const {
  const auto& lane = prepared_execution_lane();
  // Empty programs pay one presence vote, with no metadata/HistoryManager/field
  // allocation. Rank disagreement cannot strand peers in source collectives.
  if (all_reduce_max(interaction_history_declarations_.empty() ? 0L : 1L, lane) == 0) {
    interaction_history_declarations_sealed_ = true;
    return;
  }
  std::string declaration_authority;
  std::vector<ExactOrderedBytePair> declarations;
  interaction_phase(lane, [&] {
    ExactContractBuilder exact;
    exact.text("pops.interaction-history-declarations@1");
    for (const auto& [id, d] : interaction_history_declarations_)
      exact.scalar(id)
          .scalar(d.owner)
          .text(d.name)
          .scalar(d.lag)
          .text(d.state)
          .text(d.space)
          .text(d.clock)
          .text(d.interpolation)
          .scalar(d.max_bytes)
          .text(d.identity);
    declaration_authority = std::move(exact).release();
    declarations.emplace_back("interaction-history-declarations", declaration_authority);
  });
  if (!all_ranks_agree_exact_ordered_byte_pairs(declarations, lane))
    throw std::invalid_argument("interaction history declarations differ across ranks");
  interaction_history_declarations_sealed_ = true;
  const int count = nlev();
  if (all_reduce_min(static_cast<long>(count), lane) !=
      all_reduce_max(static_cast<long>(count), lane))
    throw std::invalid_argument("interaction history hierarchy level counts differ across ranks");
  if (interaction_history_declarations_.empty() || count == 1)
    return;
  std::map<InteractionHistorySourceKey, std::vector<const InteractionHistoryDeclaration*>> sources;
  interaction_phase(lane, [&] {
    if (count < 1)
      throw std::invalid_argument("interaction history has no hierarchy");
    for (const auto& [id, d] : interaction_history_declarations_) {
      (void)id;
      sources[{d.owner, d.name}].push_back(&d);
    }
  });
  std::map<InteractionHistorySourceKey, std::shared_ptr<const InteractionHistorySnapshot>> staged;
  for (const auto& [source, consumers] : sources) {
    const auto& [owner, name] = source;
    const auto& live = runtime_state().hist_;
    bool cold = false;
    std::size_t bytes = sizeof(InteractionHistorySnapshot);
    std::string authority;
    interaction_phase(lane, [&] {
      ExactContractBuilder exact;
      exact.text("pops.interaction-history-macro-source@1")
          .scalar(owner)
          .text(name)
          .scalar(runtime_->topology_epoch())
          .scalar(runtime_->materialization_generation())
          .scalar(resource_attempt().ordinal())
          .scalar(facade_->macro_step())
          .scalar(facade_->time())
          .scalar(dt);
      if (!resource_attempt().visible())
        throw std::logic_error("interaction history capture requires a live Program attempt");
      for (const auto* d : consumers) {
        const auto reference = history_key_(name, 0);
        for (int level = 0; level < count; ++level) {
          const auto key = history_key_(name, level);
          if (pending_history_remaps_.contains(key))
            throw std::invalid_argument("interaction History@2 has a pending remap");
          const bool level_cold = interaction_history_selected(
              live, key, d->lag, owner, d->state, d->space, d->clock, d->interpolation, exact);
          if (level == 0)
            cold = level_cold;
          if (level_cold != cold)
            throw std::invalid_argument("interaction history maturity differs across levels");
          if (!cold &&
              (live.slot_sample.at(reference).at(d->lag) != live.slot_sample.at(key).at(d->lag) ||
               std::bit_cast<InteractionRealWord>(live.slot_dt.at(reference).at(d->lag)) !=
                   std::bit_cast<InteractionRealWord>(live.slot_dt.at(key).at(d->lag))))
            throw std::invalid_argument("interaction retained sample differs across levels");
        }
      }
      for (int level = 0; level < count; ++level) {
        const auto key = history_key_(name, level);
        const auto& accepted = facade_->prepared_amr_block_state(owner, level);
        const auto& ring = live.histories.at(key);
        // 13 complete map entries, vector controls, and descriptor string storage.
        // Map linkage uses four pointer widths; payload/Fab/index storage is counted separately.
        bytes = interaction_add(
            bytes, interaction_product(
                       13, interaction_add(sizeof(std::string) + 4 * sizeof(void*), key.size())));
        bytes = interaction_add(bytes,
                                4 * sizeof(int) + 2 * sizeof(bool) + 2 * sizeof(std::vector<Real>) +
                                    sizeof(std::vector<field_type>) + 4 * sizeof(std::string));
        for (const auto* identity :
             {&live.state_identity.at(key), &live.space_identity.at(key),
              &live.clock_identity.at(key), &live.interpolation_identity.at(key)})
          bytes = interaction_add(bytes, identity->size());
        bytes = interaction_add(
            bytes, interaction_product(ring.size(), sizeof(Real) + sizeof(HistorySampleIdentity)));
        for (const auto& field : ring) {
          require_same_field_contract_(field, accepted, "interaction history macro snapshot");
          if (field.ghosts() != live.declared_ghosts(key, accepted.ghosts()))
            throw std::invalid_argument("interaction history has foreign declared ghosts");
          bytes = interaction_add(bytes, interaction_history_field_bytes_(field, lane, exact));
        }
        if (cold) {
          if (live.initialized.at(key))
            interaction_history_cold_equal(ring.at(consumers.front()->lag), accepted);
          bytes = interaction_add(bytes, interaction_history_field_bytes_(accepted, lane, exact));
        }
      }
      authority = std::move(exact).release();
      bytes = interaction_add(bytes, authority.size());
      bytes = interaction_add(
          bytes, interaction_add(sizeof(InteractionHistorySourceKey) + 4 * sizeof(void*) +
                                     sizeof(std::shared_ptr<const InteractionHistorySnapshot>),
                                 name.size()));
      if (bytes > static_cast<std::size_t>(std::numeric_limits<long>::max()))
        throw std::length_error("interaction history source budget exceeds collective count range");
    });
    std::vector<ExactOrderedBytePair> source_contract;
    interaction_phase(
        lane, [&] { source_contract.emplace_back("interaction-history-source", authority); });
    if (!all_ranks_agree_exact_ordered_byte_pairs(source_contract, lane))
      throw std::invalid_argument("interaction history source authority differs across ranks");
    // Passing the same remaining budget on all ranks is necessary because the
    // direct operator seals max_bytes in its rank-global authority.
    bytes = static_cast<std::size_t>(all_reduce_max(static_cast<long>(bytes), lane));
    interaction_phase(lane, [&] {
      for (const auto* d : consumers)
        if (bytes >= d->max_bytes)
          throw std::length_error("interaction history macro source exceeds its explicit budget");
    });
    interaction_phase(lane, [&] {
      auto image = std::make_shared<InteractionHistorySnapshot>();
      image->attempt = resource_attempt();
      image->epoch = runtime_->topology_epoch();
      image->generation = runtime_->materialization_generation();
      image->macro_step = facade_->macro_step();
      image->time = facade_->time();
      image->dt = dt;
      image->retained_bytes = bytes;
      image->authority = authority;
      for (int level = 0; level < count; ++level) {
        const auto key = history_key_(name, level);
        auto& frozen = image->history;
        frozen.histories.emplace(key,
                                 live.histories.at(key));  // Fab copy performs Kokkos::deep_copy.
        frozen.depth.emplace(key, live.depth.at(key));
        frozen.initialized.emplace(key, live.initialized.at(key));
        frozen.fill_count.emplace(key, live.fill_count.at(key));
        frozen.store_pending.emplace(key, live.store_pending.at(key));
        frozen.owner.emplace(key, live.owner.at(key));
        frozen.state_identity.emplace(key, live.state_identity.at(key));
        frozen.space_identity.emplace(key, live.space_identity.at(key));
        frozen.clock_identity.emplace(key, live.clock_identity.at(key));
        frozen.interpolation_identity.emplace(key, live.interpolation_identity.at(key));
        frozen.ghost_depth.emplace(key, live.ghost_depth.at(key));
        frozen.slot_dt.emplace(key, live.slot_dt.at(key));
        frozen.slot_sample.emplace(key, live.slot_sample.at(key));
        if (cold)
          image->cold_seed.push_back(facade_->prepared_amr_block_state(owner, level));
      }
      staged.emplace(source, std::move(image));
    });  // Allocation/deep-copy failure converges before the next source or geometry collective.
  }
  interaction_history_snapshots_.swap(staged);
}

std::shared_ptr<const InteractionHistorySnapshot> require_interaction_history_snapshot_(
    std::int64_t value_id, const InteractionHistoryDeclaration& issued) const {
  const auto declaration = interaction_history_declarations_.find(value_id);
  if (declaration == interaction_history_declarations_.end() || declaration->second != issued)
    throw std::invalid_argument("interaction History@2 has no exact static source declaration");
  const auto found = interaction_history_snapshots_.find({issued.owner, issued.name});
  if (found == interaction_history_snapshots_.end())
    throw std::logic_error("interaction history macro snapshot is absent or revoked");
  const auto& image = *found->second;
  if (!image.attempt.visible() || !image.attempt.same_attempt(resource_attempt()) ||
      image.epoch != runtime_->topology_epoch() ||
      image.generation != runtime_->materialization_generation() ||
      image.macro_step != facade_->macro_step() ||
      std::bit_cast<std::uint64_t>(image.time) !=
          std::bit_cast<std::uint64_t>(double(facade_->time())))
    throw std::logic_error("interaction history macro source hierarchy/attempt was revoked");
  // A local prev under subcycling cannot be relabeled as a macro prev. There is
  // no interpolation here: only a complete, identical macro window is admitted.
  if (current_interval_begin_phase_ != ::pops::amr::Rational(0, 1) ||
      current_interval_end_phase_ != ::pops::amr::Rational(1, 1) ||
      std::bit_cast<std::uint64_t>(current_interval_start_time_) !=
          std::bit_cast<std::uint64_t>(image.time) ||
      std::bit_cast<std::uint64_t>(current_dt_) != std::bit_cast<std::uint64_t>(image.dt))
    throw std::invalid_argument(
        "interaction history NoInterpolation source requires the exact macro window");
  return found->second;
}

public:
