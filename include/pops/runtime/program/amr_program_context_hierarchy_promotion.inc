// Accepted child scopes cross the hierarchy frontier as one authority transaction.
// Preparation and validation are local; generated code votes/fences before the
// noexcept commit. A late invalid child cannot promote any earlier sibling.
public:
template <class Lifetime>
void prepare_hierarchy_scope_promotion(std::int64_t operation,
                                       typename ContinuationAuthority<Dim>::Scope& scope,
                                       typename ContinuationAuthority<Dim>::Accepted& destination,
                                       const std::shared_ptr<Lifetime>& lifetime) const {
  if (!mapping_continuation_ || !mapping_continuation_->executing || !lifetime ||
      active_level_ < 0 ||
      static_cast<std::size_t>(active_level_) >= mapping_continuation_->failures.size())
    throw std::logic_error("child promotion has no exact hierarchy invocation");
  auto& frame = *mapping_continuation_;
  const auto& handlers = frame.failures.at(static_cast<std::size_t>(active_level_));
  if (handlers.empty() || handlers.back().operation != operation)
    throw std::logic_error("child promotion lost its recovery scope");
  // Scope creation already established this issuer. Do not enter a nested
  // collective getter after rank-local validation; the caller owns the vote.
  auto* selected = &continuation_parent_authority_;
  if (!continuation_parent_phase_) {
    const auto found = level_continuation_authorities_.find(active_level_);
    if (found == level_continuation_authorities_.end() || !found->second)
      throw std::logic_error("child promotion has no established level issuer");
    selected = found->second.get();
  }
  auto& issuer = *selected;
  auto plan = issuer.prepare_promotion(scope, destination);
  auto& levels = frame.promotions[operation];
  if (levels.empty())
    levels.resize(frame.failures.size());
  auto& slot = levels.at(static_cast<std::size_t>(active_level_));
  if (slot)
    throw std::logic_error("duplicate child promotion at one hierarchy level");
  slot.emplace(HierarchyScopePromotion{&issuer, &scope, &destination, lifetime, std::move(plan)});
}

void validate_hierarchy_scope_promotions(std::int64_t operation) const {
  require_hierarchy_failure_handlers_(operation);
  const auto found = mapping_continuation_->promotions.find(operation);
  if (found == mapping_continuation_->promotions.end() ||
      found->second.size() != mapping_continuation_->failures.size())
    throw std::logic_error("child promotion does not cover the hierarchy");
  const auto& levels = found->second;
  for (std::size_t level = 0; level < levels.size(); ++level) {
    if (!levels[level])
      throw std::logic_error("child promotion has an absent hierarchy participant");
    const auto& entry = *levels[level];
    if (!entry.issuer || !entry.scope || !entry.destination || !entry.lifetime)
      throw std::logic_error("child promotion lost its retained frame");
    for (std::size_t prior = 0; prior < level; ++prior)
      if (levels[prior]->issuer == entry.issuer || levels[prior]->scope == entry.scope ||
          levels[prior]->destination == entry.destination)
        throw std::logic_error("child promotion aliases another hierarchy authority");
    entry.issuer->validate_promotion(entry.plan, *entry.scope);
  }
}

// The immediately preceding collective preflight authenticates all entries and
// handlers. No callback, allocation, fence or fallible validation intervenes.
void commit_hierarchy_scope_promotions(std::int64_t operation) const noexcept {
  auto& frame = *mapping_continuation_;
  const auto found = frame.promotions.find(operation);
  for (auto& selected : found->second) {
    auto& entry = *selected;
    *entry.destination = entry.issuer->commit_promotion(std::move(entry.plan), *entry.scope);
  }
  for (auto& handlers : frame.failures)
    handlers.pop_back();
  frame.promotions.erase(found);
}

private:
