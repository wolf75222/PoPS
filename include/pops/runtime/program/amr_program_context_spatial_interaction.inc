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
