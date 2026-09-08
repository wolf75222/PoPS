struct SpatialConsumedFlux {
  std::uint64_t attempt;
  std::vector<std::uint8_t> weighted_fragments;
  const field_type *parent_candidate, *child_candidate;
};

// The spatial request validator permits only one direct Q(solved) commit with unit
// wrappers. Therefore its previous-stage affine weights are already final here.
void stage_spatial_hierarchy_previous(int identity, int program_block, const field_type& previous,
                                      const field_type* seed) const {
  prepare_spatial_collectively([&] {
    auto& resource = spatial_hierarchy_resources_.at(identity);
    if (resource.block != program_block)
      throw std::logic_error("composite implicit previous stage changed its block owner");
    resource.workspace->stage(active_level_, previous, seed);
    resource.previous_flux.at(active_level_) = active_flux_expression_(previous);
  });
}

// Reuse the current outer ledger as the sole accepted quadrature authority. A
// committed copy supplies the provisional conservative RHS correction. The original
// stays pending until the outer transaction commits its exact weighted evidence.
void reconcile_spatial_hierarchy_previous(int identity) const {
  auto& resource = spatial_hierarchy_resources_.at(identity);
  auto& workspace = *resource.workspace;
  const auto block = static_cast<std::size_t>(sys_block(resource.block));
  const auto levels = workspace.levels();
  struct StagedLevel {
    multiblock_flux_ledger_type* ledger;
    ::pops::amr::ClockWindow window;
    const field_type* candidate;
  };
  std::vector<StagedLevel> staged;
  prepare_spatial_collectively([&] { staged.resize(levels); });
  for (std::size_t level = 0; level < levels; ++level)
    with_program_attempt_level(static_cast<int>(level), [&] {
      const auto* candidate = active_attempt_states_[block];
      materialize_active_flux_expression_(block, *candidate, &resource.previous_flux[level]);
      staged[level] = {active_outgoing_flux_[block], active_subcycling_window_, candidate};
    });
  for (std::size_t child = levels; child-- > 1;) {
    const auto parent = child - 1;
    std::optional<multiblock_flux_ledger_type> provisional;
    prepare_spatial_collectively([&] {
      if (staged[parent].ledger == nullptr)
        throw std::logic_error("composite implicit previous stage lost its transition ledger");
      provisional.emplace(*staged[parent].ledger);
      provisional->commit();
    });
    std::optional<multiblock_reflux_context_type> context;
    prepare_spatial_collectively([&] {
      const auto ratio = runtime_->hierarchy().layout(child).ratio_from_parent();
      const ::pops::amr::reflux::FaceRefinementMapping<Dim> mapping{
          runtime_->hierarchy().layout(parent).domain().lo,
          runtime_->hierarchy().layout(child).domain().lo};
      context.emplace(multiblock_reflux_context_type{
          block, facade_->prepared_amr_multiblock_hierarchy_().block_identity(block), parent,
          active_subcycling_attempt_, staged[parent].window, workspace.previous(parent),
          workspace.previous(child), *provisional, ratio, mapping});
    });
    prepare_spatial_collectively([&] { apply_multiblock_reflux_(*context); });
    ::pops::numerics::time::amr::execute_average_down_collectively(
        *runtime_, child, workspace.previous(child), workspace.previous(parent),
        prepared_execution_lane());
    prepare_spatial_collectively([&] {
      if (provisional->published_size() != 0)
        spatial_consumed_flux_.insert_or_assign(
            {block, parent},
            SpatialConsumedFlux{active_subcycling_attempt_,
                                spatial_weighted_fragments(*provisional), staged[parent].candidate,
                                staged[child].candidate});
    });
  }
  // These expressions belong to the corrected previous stage. The implicit result
  // has no transport basis to apply again; its consumed entries remain in the outer ledger.
  for (auto& expression : resource.previous_flux)
    expression.clear();
}

// Only the exact solved engine candidate may retain its predictor's pending evidence.
// Authentication here does not consume the later committed-ledger reflux proof.
bool retain_spatial_materialization(std::size_t block, const field_type& candidate,
                                    const auto& expression) const {
  bool retained = false;
  const auto first = spatial_consumed_flux_.lower_bound({block, 0});
  if (first == spatial_consumed_flux_.end() || first->first.first != block)
    return false;
  if (!expression.empty())
    throw std::logic_error("composite implicit result regained an accepted flux expression");
  for (int side = 0; side < 2; ++side) {
    const auto* ledger = side ? active_outgoing_flux_[block] : active_incoming_flux_[block];
    if (!ledger)
      continue;
    const auto parent = static_cast<std::size_t>(active_level_ - (side ? 0 : 1));
    const auto found = spatial_consumed_flux_.find({block, parent});
    if (found == spatial_consumed_flux_.end() &&
        ledger->pending_size() + ledger->published_size() == 0)
      continue;
    const auto& proof = spatial_consumed_flux_.at({block, parent});
    auto committed = *ledger;
    committed.commit();
    if (proof.attempt != active_subcycling_attempt_ ||
        (side ? proof.parent_candidate : proof.child_candidate) != &candidate ||
        proof.weighted_fragments != spatial_weighted_fragments(committed))
      throw std::logic_error(
          "composite implicit pending weighted fragments changed before publication");
    retained = true;
  }
  return retained;
}

bool consume_spatial_reconciliation(const multiblock_reflux_context_type& context) const {
  const auto key = std::make_pair(context.block, context.parent_level);
  const auto found = spatial_consumed_flux_.find(key);
  const long present = found == spatial_consumed_flux_.end() ? 0L : 1L;
  const auto& lane = prepared_execution_lane();
  if (all_reduce_min(present, lane) != all_reduce_max(present, lane))
    throw std::logic_error("composite implicit consumed quadrature differs between ranks");
  if (!present)
    return false;
  prepare_spatial_collectively([&] {
    if (found->second.attempt != context.attempt ||
        found->second.parent_candidate != &context.parent ||
        found->second.child_candidate != &context.child ||
        found->second.weighted_fragments != spatial_weighted_fragments(context.flux))
      throw std::logic_error(
          "composite implicit consumed weighted fragments changed before commit");
  });
  spatial_consumed_flux_.erase(found);
  return true;
}

// Consumed predictor evidence is attempt-local even when the solve rejects before
// the outer engine reaches its final reconciliation callback.
template <class Body>
void with_spatial_reconciliation_attempt(Body&& body) const {
  spatial_consumed_flux_.clear();
  try {
    std::forward<Body>(body)();
  } catch (...) {
    spatial_consumed_flux_.clear();
    throw;
  }
  spatial_consumed_flux_.clear();
}

void reconcile_multiblock_reflux_(multiblock_reflux_context_type& context) const {
  if (!consume_spatial_reconciliation(context))
    apply_multiblock_reflux_(context);
}
