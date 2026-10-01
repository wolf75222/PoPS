/// Direct composite interaction uses a complete accepted/history source family.
/// A candidate produced on one level is not a simultaneous composite snapshot.
template <class Kernel>
field_type& spatial_interaction_owned(std::int64_t value_id, int program_block, const field_type& source,
                              std::span<const int> components, std::uint64_t max_bytes,
                              std::string_view identity, std::string_view source_clock, bool accepted_composite, Kernel kernel) const {
  refresh_resources_();
  const auto& lane = prepared_execution_lane();
  const int owner = sys_block(program_block);
  const int count = nlev();
  if (all_reduce_min(static_cast<long>(count), lane) != all_reduce_max(static_cast<long>(count), lane))
    throw std::invalid_argument("direct interaction hierarchy level counts differ across ranks");
  const auto epoch = runtime_->topology_epoch();
  const auto generation = runtime_->materialization_generation();
  PreparedResourceAttempt attempt;
  runtime::multiblock::BoundaryEvaluationPoint point;
  std::optional<Identity> family;
  std::vector<InteractionLevelView<Dim, MemorySpace>> levels;
  interaction_phase(lane, [&] {
    if (value_id < 0) throw std::invalid_argument("interaction result requires an issued SSA id");
    point = boundary_evaluation_point(0);
    if (point.clock != source_clock) throw std::invalid_argument("direct interaction source clock differs from its issued frame");
    attempt = resource_attempt();
    if (!attempt.visible()) throw std::logic_error("direct interaction requires a live Program attempt");
    family = classify_owner_field_(owner, source, "direct spatial interaction");
    if (accepted_composite && family->family != Family::State)
      throw std::invalid_argument("accepted interaction source is not an authenticated State carrier");
    if (!accepted_composite && family->family != Family::History && count != 1)
      throw std::invalid_argument("direct composite interaction requires a simultaneous source snapshot; level-local candidate has no composite barrier");
    if (count < 1 || interaction_product(static_cast<std::size_t>(count), sizeof(InteractionLevelView<Dim, MemorySpace>)) > max_bytes)
      throw std::length_error("direct spatial interaction level metadata exceeds its budget");
    levels.reserve(static_cast<std::size_t>(count));
  });
  // Geometry getters can enter collective preparation. No owner-only callback
  // or fallible rank-local validation precedes a getter inside this loop.
  for (int level = 0; level < count; ++level) {
    const auto geom = facade_->prepared_amr_level_geometry(level);
    interaction_phase(lane, [&] {
      const auto& field = accepted_composite ? facade_->prepared_amr_block_state(owner, level)
          : resolve_owner_field_(owner, source, *family, level, "direct spatial interaction");
      const auto* active = facade_->prepared_amr_block_level_active_mask(owner, level);
      const auto& coverage = facade_->prepared_amr_block_level_coverage_mask(owner, level);
      const auto* kappa = facade_->prepared_amr_block_level_volume_fraction(owner, level);
      levels.push_back({&field, active, &coverage, kappa, geom});
    });
  }
  std::string authority;
  interaction_phase(lane, [&] {
    ExactContractBuilder exact;
    exact.text(identity).scalar(value_id).scalar(epoch).scalar(generation).scalar(attempt.ordinal())
        .scalar(facade_->macro_step()).scalar(facade_->time());
    interaction_append_point(exact, point);
    exact.scalar(static_cast<std::uint8_t>(family->family)).scalar(accepted_composite)
        .text(family->history_name).scalar(family->history_lag);
    if (family->family == Family::History) {
      const auto& history = runtime_state().hist_;
      const auto reference = history_key_(family->history_name, family->direct_level);
      for (int level = 0; level < count; ++level) {
        const auto key = history_key_(family->history_name, level);
        if (pending_history_remaps_.contains(key))
          throw std::invalid_argument("composite interaction history has a pending remap");
        history.matching_authenticated_sample(reference, key, family->history_lag).append_contract(exact);
        history.append_descriptor_contract(key, exact);
      }
    }
    authority = std::move(exact).release();
  });
  auto result = direct_spatial_interaction<Dim, MemorySpace>(
      levels, static_cast<std::size_t>(active_level_), components, max_bytes, authority,
      lane, kernel);
  interaction_phase(lane, [&] {
    if (epoch != runtime_->topology_epoch() || generation != runtime_->materialization_generation() ||
        !attempt.visible() || !attempt.same_attempt(resource_attempt()) || point != boundary_evaluation_point(0))
      throw std::logic_error("direct interaction source hierarchy/attempt was revoked");
  });
  // Allocate a detached registry node before the final vote. No successful
  // rank mutates its issued scratch if another rank cannot allocate its node.
  const ScratchKey key{ScratchKind::Scalar, active_level_, owner, value_id, 0};
  std::map<ScratchKey, field_type> staged;
  interaction_phase(lane, [&] { staged.emplace(key, std::move(result)); });
  static_assert(std::is_nothrow_move_assignable_v<field_type>);
  auto existing = scratches_.find(key);
  if (existing != scratches_.end()) {
    existing->second = std::move(staged.begin()->second);
    return existing->second;
  }
  // Node transfer uses the same allocator and the built-in integral tuple key;
  // it allocates no storage and invokes no user comparator after the vote.
  auto issued = scratches_.insert(staged.extract(staged.begin()));
  return issued.position->second;
}

// IR18 selected-history source. IR17 implementation above remains byte exact.
/// Direct composite interaction uses a complete accepted/history source family.
/// A candidate produced on one level is not a simultaneous composite snapshot.
template <class Kernel>
field_type& spatial_interaction_history_owned(std::int64_t value_id, int program_block, const field_type& source,
                              std::span<const int> components, std::uint64_t max_bytes,
                              std::string_view identity, std::string_view source_clock, bool accepted_composite,
                              const std::string& history_name, int history_lag, std::string_view seed_state,
                              std::string_view seed_space, std::string_view interpolation, Kernel kernel) const {
  refresh_resources_();
  const auto& lane = prepared_execution_lane();
  const int owner = sys_block(program_block);
  const int count = nlev();
  if (all_reduce_min(static_cast<long>(count), lane) != all_reduce_max(static_cast<long>(count), lane))
    throw std::invalid_argument("direct interaction hierarchy level counts differ across ranks");
  const auto epoch = runtime_->topology_epoch();
  const auto generation = runtime_->materialization_generation();
  PreparedResourceAttempt attempt;
  runtime::multiblock::BoundaryEvaluationPoint point;
  std::optional<Identity> family;
  std::vector<InteractionLevelView<Dim, MemorySpace>> levels;
  interaction_phase(lane, [&] {
    if (value_id < 0) throw std::invalid_argument("interaction result requires an issued SSA id");
    point = boundary_evaluation_point(0);
    if (point.clock != source_clock) throw std::invalid_argument("direct interaction source clock differs from its issued frame");
    attempt = resource_attempt();
    if (!attempt.visible()) throw std::logic_error("direct interaction requires a live Program attempt");
    family = classify_owner_field_(owner, source, "direct spatial interaction");
    if (accepted_composite && family->family != Family::State)
      throw std::invalid_argument("accepted interaction source is not an authenticated State carrier");
    if (!accepted_composite && family->family != Family::History && count != 1)
      throw std::invalid_argument("direct composite interaction requires a simultaneous source snapshot; level-local candidate has no composite barrier");
    if (count < 1 || interaction_product(static_cast<std::size_t>(count), sizeof(InteractionLevelView<Dim, MemorySpace>)) > max_bytes)
      throw std::length_error("direct spatial interaction level metadata exceeds its budget");
    levels.reserve(static_cast<std::size_t>(count));
  });
  bool cold = false;
  auto history_authority = [&] {
    ExactContractBuilder exact;
    const auto& history = runtime_state().hist_;
    const auto reference = history_key_(history_name, family->direct_level);
    if (accepted_composite || family->family != Family::History ||
        family->history_name != history_name || family->history_lag != history_lag)
      throw std::invalid_argument("interaction History@2 has no exact issued keeper carrier");
    for (int level = 0; level < count; ++level) {
      const auto key = history_key_(history_name, level);
      if (pending_history_remaps_.contains(key))
        throw std::invalid_argument("interaction History@2 has a pending remap");
      const bool level_cold = interaction_history_selected(history, key, history_lag, owner,
          seed_state, seed_space, source_clock, interpolation, exact);
      if (level == 0) cold = level_cold;
      if (level_cold != cold)
        throw std::invalid_argument("interaction history maturity differs across levels");
      if (!cold && (history.slot_sample.at(reference).at(history_lag) != history.slot_sample.at(key).at(history_lag) ||
          std::bit_cast<InteractionRealWord>(history.slot_dt.at(reference).at(history_lag)) !=
          std::bit_cast<InteractionRealWord>(history.slot_dt.at(key).at(history_lag))))
        throw std::invalid_argument("interaction retained sample differs across levels");
    }
    return std::move(exact).release();
  };
  std::string history_snapshot;
  interaction_phase(lane, [&] { history_snapshot = history_authority(); });
  // Geometry getters can enter collective preparation. No owner-only callback
  // or fallible rank-local validation precedes a getter inside this loop.
  for (int level = 0; level < count; ++level) {
    const auto geom = facade_->prepared_amr_level_geometry(level);
    interaction_phase(lane, [&] {
      const auto& field = cold ?  facade_->prepared_amr_block_state(owner, level)
          : resolve_owner_field_(owner, source, *family, level, "direct spatial interaction");
      const auto history_key = history_key_(history_name, level);
      if (cold && runtime_state().hist_.initialized.at(history_key))
        interaction_history_cold_equal(runtime_state().hist_.histories.at(history_key).at(history_lag), field);
      const auto* active = facade_->prepared_amr_block_level_active_mask(owner, level);
      const auto& coverage = facade_->prepared_amr_block_level_coverage_mask(owner, level);
      const auto* kappa = facade_->prepared_amr_block_level_volume_fraction(owner, level);
      levels.push_back({&field, active, &coverage, kappa, geom});
    });
  }
  std::string authority;
  interaction_phase(lane, [&] {
    ExactContractBuilder exact;
    exact.text(identity).scalar(value_id).scalar(epoch).scalar(generation).scalar(attempt.ordinal())
        .scalar(facade_->macro_step()).scalar(facade_->time());
    interaction_append_point(exact, point);
    exact.scalar(static_cast<std::uint8_t>(family->family)).scalar(accepted_composite)
        .text(family->history_name).scalar(family->history_lag);
    exact.text("pops.spatial-interaction-history-source@1").text(history_snapshot);
    authority = std::move(exact).release();
  });
  auto result = direct_spatial_interaction<Dim, MemorySpace>(
      levels, static_cast<std::size_t>(active_level_), components, max_bytes, authority,
      lane, kernel);
  interaction_phase(lane, [&] {
    if (history_snapshot != history_authority())
      throw std::logic_error("interaction selected history lifecycle changed during snapshot");
    if (epoch != runtime_->topology_epoch() || generation != runtime_->materialization_generation() ||
        !attempt.visible() || !attempt.same_attempt(resource_attempt()) || point != boundary_evaluation_point(0))
      throw std::logic_error("direct interaction source hierarchy/attempt was revoked");
  });
  // Allocate a detached registry node before the final vote. No successful
  // rank mutates its issued scratch if another rank cannot allocate its node.
  const ScratchKey key{ScratchKind::Scalar, active_level_, owner, value_id, 0};
  std::map<ScratchKey, field_type> staged;
  interaction_phase(lane, [&] { staged.emplace(key, std::move(result)); });
  static_assert(std::is_nothrow_move_assignable_v<field_type>);
  auto existing = scratches_.find(key);
  if (existing != scratches_.end()) {
    existing->second = std::move(staged.begin()->second);
    return existing->second;
  }
  // Node transfer uses the same allocator and the built-in integral tuple key;
  // it allocates no storage and invokes no user comparator after the vote.
  auto issued = scratches_.insert(staged.extract(staged.begin()));
  return issued.position->second;
}

template <class Function>
void closed_interaction_phase_(Function&& function) const {
  std::exception_ptr error;
  try { function(); Kokkos::fence(); } catch (...) { error = std::current_exception(); }
  if (error) { try { Kokkos::fence(); } catch (...) {} }
  collectively_rethrow_exception(error, prepared_execution_lane(), "closed original field interaction local phase");
}

// IR19: completed original-field source, separate from accepted State/history.
// The private receipt is issued only after the original SolveOutcome Accept.
void seal_original_field_source(std::int64_t solve_id,
    const std::shared_ptr<hierarchy_tensor_solver_type>& provider,
    const std::shared_ptr<PreparedAmrFieldResidual<Dim>>& core,
    const std::function<AmrFieldResidualAuthority()>& current_authority,
    const std::vector<std::tuple<std::int64_t, std::int64_t, int, int, std::string>>& bindings,
    std::string_view identity, std::uint64_t max_bytes) const {
  const auto& lane = prepared_execution_lane();
  closed_interaction_phase_([&] {
    if (!provider || !core || !current_authority || identity.empty() || solve_id < 0 ||
        hierarchy_field_resources_.at(solve_id).solver != provider)
      throw std::invalid_argument("closed original source has a foreign provider/identity");
  });
  // This callback authenticates all live level envelopes on the prepared lane.
  const auto& tower = core->candidate(current_authority(), lane);
  std::vector<Geometry<Dim>> geometries;
  closed_interaction_phase_([&] { geometries.reserve(tower.size()); });
  for (int level = 0; level < nlev(); ++level) {
    const auto geometry = facade_->prepared_amr_level_geometry(level);
    closed_interaction_phase_([&] { geometries.push_back(geometry); });
  }
  (void)core->candidate(current_authority(), lane);
  // The Accept stamp identifies the original candidate, not mutable public
  // provider storage. Authenticate every published cell bit before allocating
  // a source image. Accept reconstructs its own halos, so only valid cells are
  // part of this comparison; direct quadrature never consumes either halo.
  closed_interaction_phase_([&] {
    (void)provider->accepted_original_candidate_generation(&tower, lane);
    if (tower.empty() || tower.size() != static_cast<std::size_t>(nlev()))
      throw std::invalid_argument("closed original source is not a complete tower");
    for (int level = 0; level < nlev(); ++level) {
      const auto& published = provider->solution(level);
      const auto& accepted = tower.at(level);
      if (published.layout() != accepted.layout() || published.distribution() != accepted.distribution() ||
          published.local_rank() != accepted.local_rank() || published.ncomp() != accepted.ncomp())
        throw std::invalid_argument("accepted original source tuple layout/width changed");
      for (std::size_t patch = 0; patch < accepted.local_size(); ++patch) {
        const auto actual = published.fab(patch).view();
        const auto expected = accepted.fab(patch).view();
        const int width = accepted.ncomp();
        const Real changed = for_each_cell_reduce_max(accepted.box(patch),
            [=] POPS_HD(const Index<Dim>& cell) {
          for (int component = 0; component < width; ++component)
            if (std::bit_cast<InteractionRealWord>(actual(cell, component)) !=
                std::bit_cast<InteractionRealWord>(expected(cell, component))) return Real(1);
          return Real(0);
        });
        if (changed != Real(0))
          throw std::invalid_argument("accepted original source values changed after Accept");
      }
    }
  });
  std::map<std::int64_t, std::shared_ptr<const ClosedOriginalFieldSource>> staged;
  closed_interaction_phase_([&] {
    const auto stamp = provider->accepted_original_candidate_generation(&tower, lane);
    if (tower.empty() || tower.size() != static_cast<std::size_t>(nlev()))
      throw std::invalid_argument("closed original source is not a complete tower");
    auto issued = std::make_shared<ClosedOriginalFieldSource>();
    issued->provider = provider;
    issued->candidate = [this, core, current_authority]() -> const std::vector<field_type>& {
      return core->candidate(current_authority(), prepared_execution_lane());
    };
    issued->accepted_generation = stamp;
    issued->identity = identity;
    std::size_t binding_bytes = interaction_add(sizeof(ClosedOriginalFieldSource), identity.size());
    for (const auto& binding : bindings)
      binding_bytes = interaction_add(binding_bytes, interaction_add(sizeof(binding), std::get<4>(binding).size()));
    if (!max_bytes || binding_bytes >= max_bytes)
      throw std::length_error("closed original source binding metadata exceeds its budget");
    if (bindings.empty()) throw std::invalid_argument("closed original source has no issued source/result bindings");
    for (const auto& [result, source, component, storage, token] : bindings) {
      if (result < 0 || source < 0 || storage < 0 || token.empty() || component < 0 ||
          component >= tower.front().ncomp() || !issued->bindings.emplace(result,
              std::make_tuple(source, component, storage, token)).second)
        throw std::invalid_argument("closed original source has invalid/duplicate tuple bindings");
    }
    std::size_t bytes = interaction_add(binding_bytes, interaction_product(tower.size(), 2 * sizeof(field_type) + sizeof(Geometry<Dim>)));
    for (int level = 0; level < nlev(); ++level) {
      const auto& field = tower.at(level);
      const auto& original = tower.at(level);
      if (field.layout() != original.layout() || field.distribution() != original.distribution() ||
          field.local_rank() != original.local_rank() || field.ncomp() != original.ncomp())
        throw std::invalid_argument("accepted original source tuple layout/width changed");
      bytes = interaction_add(bytes, interaction_product(field.layout().size(),
          2 * sizeof(Box<Dim>) + sizeof(Index<Dim>) + sizeof(std::size_t)));
      bytes = interaction_add(bytes, interaction_product(field.local_size(),
          sizeof(typename field_type::fab_type) + sizeof(std::size_t)));
      for (std::size_t patch = 0; patch < field.local_size(); ++patch)
        bytes = interaction_add(bytes, interaction_product(interaction_product(
            field.fab(patch).grown_box().numPts(), field.ncomp()), sizeof(Real)));
      const auto& mask = provider->active_cell_mask(level);
      if (mask.layout() != field.layout() || mask.distribution() != field.distribution() ||
          mask.local_rank() != field.local_rank() || mask.ncomp() != 1)
        throw std::invalid_argument("closed original source active quotient layout differs");
      bytes = interaction_add(bytes, interaction_product(mask.layout().size(),
          2 * sizeof(Box<Dim>) + sizeof(Index<Dim>) + sizeof(std::size_t)));
      bytes = interaction_add(bytes, interaction_product(mask.local_size(),
          sizeof(typename field_type::fab_type) + sizeof(std::size_t)));
      for (std::size_t patch = 0; patch < mask.local_size(); ++patch)
        bytes = interaction_add(bytes, interaction_product(mask.fab(patch).grown_box().numPts(), sizeof(Real)));
    }
    if (!max_bytes || bytes >= max_bytes)
      throw std::length_error("closed original source snapshot exceeds its explicit budget");
    issued->retained_bytes = bytes;
    issued->image.reserve(tower.size());
    issued->active_image.reserve(tower.size());
    issued->geometries = std::move(geometries);
    for (int level = 0; level < nlev(); ++level) {
      issued->image.emplace_back(tower.at(level));
      issued->active_image.emplace_back(provider->active_cell_mask(level));
    }
    Kokkos::fence();
    staged.emplace(solve_id, std::move(issued));
  });
  auto existing = closed_original_sources_.find(solve_id);
  if (existing == closed_original_sources_.end())
    closed_original_sources_.insert(staged.extract(staged.begin()));
  else existing->second = std::move(staged.begin()->second);
}

// Prepare all output levels once, after Accept, before observing any level.
// storage_block selects only an allocation witness, never the physical source.
template <class Kernel>
void prepare_closed_original_interaction(std::int64_t result_id, std::int64_t solve_id, std::int64_t source_id,
    int storage_block, std::span<const int> components, std::uint64_t max_bytes,
    std::string_view identity, Kernel kernel) const {
  const auto& lane = prepared_execution_lane();
  std::shared_ptr<const ClosedOriginalFieldSource> source;
  closed_interaction_phase_([&] {
    if (result_id < 0 || components.empty() || identity.empty() || !max_bytes)
      throw std::invalid_argument("closed interaction has incomplete authority");
    source = closed_original_sources_.at(solve_id);
    const auto& [issued_source, issued_component, issued_storage, issued_identity] = source->bindings.at(result_id);
    if (source_id != issued_source || components.size() != 1 || components.front() != issued_component ||
        storage_block != issued_storage || identity != issued_identity)
      throw std::invalid_argument("closed interaction changed its immutable source/storage binding");
  });
  const auto& candidate = source->candidate();
  std::shared_ptr<ClosedInteractionTower> result;
  std::vector<InteractionLevelView<Dim, MemorySpace>> levels;
  std::size_t retained = 0;
  closed_interaction_phase_([&] {
    if (source->provider->accepted_original_candidate_generation(&candidate, lane) !=
        source->accepted_generation || candidate.size() != static_cast<std::size_t>(nlev()))
      throw std::logic_error("closed original source Accept receipt was superseded");
    result = std::make_shared<ClosedInteractionTower>();
    result->source = source;
    result->storage_owner = sys_block(storage_block);
    result->identity = identity;
    retained = interaction_add(interaction_add(source->retained_bytes,
        interaction_add(sizeof(ClosedInteractionTower), identity.size())), interaction_product(candidate.size(),
        sizeof(field_type) + sizeof(InteractionLevelView<Dim, MemorySpace>)));
    // Reserve bytes for the deep source and the entire output, including ghosts
    // in the source image. Direct quadrature separately accounts its workspace.
    for (const auto& field : candidate) {
      for (std::size_t patch = 0; patch < field.local_size(); ++patch) {
        retained = interaction_add(retained, interaction_product(
            interaction_product(field.box(patch).numPts(), components.size()), sizeof(Real)));
      }
      retained = interaction_add(retained, interaction_product(field.layout().size(),
          2 * sizeof(Box<Dim>) + sizeof(Index<Dim>) + sizeof(std::size_t)));
      retained = interaction_add(retained, interaction_product(field.local_size(),
          sizeof(typename field_type::fab_type) + sizeof(std::size_t)));
    }
    if (retained >= max_bytes)
      throw std::length_error("closed interaction retained tower exceeds its explicit budget");
    result->output.reserve(candidate.size());
    levels.reserve(candidate.size());
    Kokkos::fence();
  });
  for (int level = 0; level < nlev(); ++level) {
    const auto& storage = facade_->prepared_amr_block_state(result->storage_owner, level);
    closed_interaction_phase_([&] {
      const auto& field = source->image.at(level);
      if (field.layout() != storage.layout() || field.distribution() != storage.distribution() ||
          field.local_rank() != storage.local_rank())
        throw std::invalid_argument("closed interaction storage changes original source geometry/ownership");
      const auto& active = source->active_image.at(level);
      levels.push_back({&field, &active, nullptr, nullptr, source->geometries.at(level)});
    });
  }
  std::uint64_t reserved = retained;
  // Broadcast integer words exactly; no float conversion of a uint64 budget.
  for (int rank = 0; rank < lane.size(); ++rank) {
    std::uint64_t words = 0;
    for (unsigned shift = 0; shift < 64; shift += 16)
      words |= static_cast<std::uint64_t>(all_reduce_sum(lane.rank() == rank ?
          static_cast<long>((static_cast<std::uint64_t>(retained) >> shift) & 0xffffU) : 0L, lane)) << shift;
    reserved = std::max(reserved, words);
  }
  std::string exact;
  closed_interaction_phase_([&] {
    ExactContractBuilder builder;
    builder.text("pops.completed-original-field-interaction@1").text(identity)
        .text(source->identity).scalar(source->accepted_generation).scalar(solve_id)
        .scalar(result_id).scalar(storage_block).scalar(max_bytes);
    exact = std::move(builder).release();
  });
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"closed-original-interaction", exact}}, lane))
    throw std::invalid_argument("closed original interaction authority differs between ranks");
  // retained is rank-local memory, so it is not a physical authority token.
  // Each direct invocation authenticates its own full layout/point contract.
  for (std::size_t level = 0; level < levels.size(); ++level) {
    auto output = direct_spatial_interaction<Dim, MemorySpace>(levels, level, components,
        max_bytes - reserved, identity, lane, kernel);
    closed_interaction_phase_([&] { result->output.push_back(std::move(output)); });
  }
  (void)source->candidate();
  std::map<std::int64_t, std::shared_ptr<const ClosedInteractionTower>> staged;
  closed_interaction_phase_([&] {
    if (source->provider->accepted_original_candidate_generation(&candidate, lane) != source->accepted_generation)
      throw std::logic_error("closed interaction source publication was revoked");
    staged.emplace(result_id, std::move(result));
  });
  auto existing = closed_interactions_.find(result_id);
  if (existing == closed_interactions_.end())
    closed_interactions_.insert(staged.extract(staged.begin()));
  else existing->second = std::move(staged.begin()->second);
}

const ClosedInteractionTower& require_closed_interaction_(std::int64_t result_id) const {
  const auto& lane = prepared_execution_lane();
  std::shared_ptr<const ClosedInteractionTower> result;
  closed_interaction_phase_([&] { result = closed_interactions_.at(result_id); });
  const auto& candidate = result->source->candidate();
  closed_interaction_phase_([&] {
    if (result->source->provider->accepted_original_candidate_generation(&candidate, lane) !=
        result->source->accepted_generation || result->output.size() != candidate.size())
      throw std::logic_error("closed interaction is stale or incomplete");
  });
  return *result;
}

const field_type& closed_original_interaction(std::int64_t result_id) const {
  const auto& result = require_closed_interaction_(result_id);
  closed_interaction_phase_([&] {
    if (active_level_ < 0 || active_level_ >= static_cast<int>(result.output.size()))
      throw std::invalid_argument("closed interaction selected a foreign level");
  });
  return result.output.at(active_level_);
}

Real reduce_closed_original_interaction(std::int64_t result_id, int component,
    std::string_view kind) const {
  const auto& result = require_closed_interaction_(result_id);
  const auto& lane = prepared_execution_lane();
  const bool maximum = kind == "max", minimum = kind == "min";
  Real local = maximum ? -std::numeric_limits<Real>::infinity() :
      minimum ? std::numeric_limits<Real>::infinity() : Real(0);
  closed_interaction_phase_([&] {
    if (kind != "sum" && kind != "abs_sum" && !maximum && !minimum)
      throw std::invalid_argument("unknown closed interaction reduction");
    for (std::size_t level = 0; level < result.output.size(); ++level) {
      const auto& field = result.output[level];
      const auto& active = result.source->active_image.at(level);
      if (component < 0 || component >= field.ncomp())
        throw std::invalid_argument("closed interaction reduction component is invalid");
      if (field.distribution().replicated() && lane.rank() != 0) continue;
      if (kind == "sum") local += pops::reduce_active_sum_local(field, component, &active);
      else if (kind == "abs_sum") local += pops::reduce_active_abs_sum_local(field, component, &active);
      else if (pops::reduce_masked_max_local(field, component, &active).has_active) {
        if (maximum) local = std::max(local, pops::reduce_active_max_local(field, component, &active));
        else local = std::min(local, pops::reduce_active_min_local(field, component, &active));
      }
    }
    Kokkos::fence();
  });
  if (maximum) return static_cast<Real>(all_reduce_max(static_cast<double>(local), lane));
  if (minimum) return static_cast<Real>(all_reduce_min(static_cast<double>(local), lane));
  return static_cast<Real>(all_reduce_sum(static_cast<double>(local), lane));
}
