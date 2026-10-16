// Geometry capabilities are supplied by the exact independent prepared provider.
// Preparation may enter collectives; capture below runs only inside a local phase.
void prepare_continuation_topology_providers_() const {
  std::vector<std::int64_t> solves;
  std::string identity;
  continuation_local_phase(
      [&] {
        ExactContractBuilder contract;
        contract.text("pops.regrid.field-provider-routes@1")
            .scalar(static_cast<std::uint64_t>(continuation_field_providers_.size()));
        for (const auto& [provider_key, provider] : continuation_field_providers_) {
          contract.text(provider_key)
              .scalar(provider.components)
              .scalar(provider.ghosts)
              .scalar(static_cast<std::uint64_t>(provider.routes.size()));
          if (provider.routes.empty())
            throw std::invalid_argument("Field representation has no installed provider route");
          for (const auto& [solve, component] : provider.routes) {
            if (!hierarchy_field_resources_.contains(solve))
              throw std::invalid_argument("Field topology provider route was not installed");
            contract.scalar(solve).scalar(component);
            solves.push_back(solve);
          }
        }
        identity = std::move(contract).release();
      },
      "Field topology provider route preparation");
  const std::array<std::pair<std::string_view, std::string_view>, 1> proof{
      {{"regrid-field-provider-routes", identity}}};
  if (!all_ranks_agree_exact_ordered_byte_pairs(proof, prepared_execution_lane()))
    throw std::invalid_argument("Field topology provider routes differ across ranks");
  for (const auto solve : solves)
    (void)configured_hierarchy_field_solver_(solve);
}

void require_same_topology_mask_(const field_type* left, const field_type* right) const {
  if (bool(left) != bool(right))
    throw std::invalid_argument("Field provider routes disagree on representation masks");
  if (!left)
    return;
  require_same_field_contract_(*left, *right, "Field provider representation mask");
  for (std::size_t patch = 0; patch < left->local_size(); ++patch) {
    auto a = left->fab(patch).create_host_mirror();
    auto b = right->fab(patch).create_host_mirror();
    left->fab(patch).copy_to_host(a);
    right->fab(patch).copy_to_host(b);
    if (a.size() != b.size())
      throw std::invalid_argument("Field provider representation mask size differs");
    for (std::size_t i = 0; i < a.size(); ++i)
      if (std::bit_cast<RealBits>(a(i)) != std::bit_cast<RealBits>(b(i)))
        throw std::invalid_argument("Field provider routes disagree on physical representation");
  }
}

void capture_provider_continuation_topologies_(TopologyViews& views,
                                               typename FieldTopology::Side side,
                                               PreparedResourceAttempt attempt) const {
  for (const auto& [contract, declared] : continuation_field_providers_) {
    const auto& provider = continuation_field_provider_(contract);
    if (provider.routes.empty())
      throw std::invalid_argument("Field representation has no installed provider route");
    for (const auto& [solve, component] : provider.routes) {
      const auto& resource = hierarchy_field_resources_.at(solve);
      if (!resource.solver || resource.topology_epoch != runtime_->topology_epoch() ||
          resource.generation != runtime_->materialization_generation())
        throw std::invalid_argument("Field representation provider was not prepared collectively");
      for (int level = 0; level < nlev(); ++level) {
        const auto representation = resource.solver->representation_topology(level);
        const auto& field = provider.prototypes.at(level);
        const auto& coverage = representation.coverage;
        if (coverage.ncomp() != 1 || coverage.layout() != field.layout() ||
            coverage.distribution() != field.distribution() ||
            coverage.local_rank() != field.local_rank())
          throw std::invalid_argument("Field representation differs from its provider coverage");
        auto data = std::make_shared<typename FieldTopology::Data>(typename FieldTopology::Data{
            attempt, level, -1, runtime_->topology_epoch(), runtime_->materialization_generation(),
            side, representation.geometry, field.layout(), field.distribution(), field.local_rank(),
            field.ghosts(), field.ncomp(),
            representation.active ? std::make_shared<const field_type>(*representation.active)
                                  : nullptr,
            representation.volume_fraction
                ? std::make_shared<const field_type>(*representation.volume_fraction)
                : nullptr,
            std::make_shared<const field_type>(coverage), contract});
        const TopologyKey key{-1, contract, level};
        if (const auto found = views.find(key); found != views.end()) {
          const auto& previous = *found->second.data_;
          if (previous.geometry != data->geometry)
            throw std::invalid_argument("Field provider routes disagree on exact geometry");
          require_same_topology_mask_(previous.active.get(), data->active.get());
          require_same_topology_mask_(previous.fraction.get(), data->fraction.get());
          require_same_topology_mask_(previous.coverage.get(), data->coverage.get());
        } else
          views.emplace(key, FieldTopology(std::move(data)));
      }
    }
  }
}

FieldTopology continuation_provider_topology_(std::string_view provider, std::int64_t level,
                                              typename FieldTopology::Side side) const {
  const auto& frame = require_continuation_regrid_();
  const auto& views = side == FieldTopology::Side::Source ? frame.source_views : frame.target_views;
  if (provider.empty() || level < 0 || level > std::numeric_limits<int>::max())
    throw std::invalid_argument("FieldProblem topology level is outside its representable range");
  const auto found = views.find({-1, std::string(provider), static_cast<int>(level)});
  if (found == views.end())
    throw std::invalid_argument("FieldProblem topology has no exact source/target provider level");
  require_field_topology(found->second);
  return found->second;
}

TopologyViews capture_continuation_topologies_(typename FieldTopology::Side side,
                                               PreparedResourceAttempt attempt) const {
  TopologyViews views;
  for (int block = 0; block < n_blocks(); ++block)
    for (int level = 0; level < nlev(); ++level) {
      const int owner = sys_block(block);
      const auto& field = facade_->prepared_amr_block_state(owner, level);
      const auto* active = facade_->prepared_amr_block_level_active_mask(owner, level);
      const auto* fraction = facade_->prepared_amr_block_level_volume_fraction(owner, level);
      auto data = std::make_shared<typename FieldTopology::Data>(typename FieldTopology::Data{
          attempt, level, block, runtime_->topology_epoch(), runtime_->materialization_generation(),
          side, facade_->prepared_amr_level_geometry(level), field.layout(), field.distribution(),
          field.local_rank(), field.ghosts(), field.ncomp(),
          active ? std::make_shared<const field_type>(*active) : nullptr,
          fraction ? std::make_shared<const field_type>(*fraction) : nullptr,
          std::make_shared<const field_type>(
              facade_->prepared_amr_block_level_coverage_mask(owner, level))});
      views.emplace(TopologyKey{block, {}, level}, FieldTopology(std::move(data)));
    }
  capture_provider_continuation_topologies_(views, side, attempt);
  return views;
}
