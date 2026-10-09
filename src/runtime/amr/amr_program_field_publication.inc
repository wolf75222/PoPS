// Included in amr_system.cpp after Impl and the prepared carrier helpers are complete.
template <int Dim>
void AmrSystem<Dim>::publish_program_field_components(
    const std::string& publication_identity, const std::vector<ProgramFieldLevel>& levels) {
  using namespace runtime::system;
  const ExecutionLane& lane = p_->require_prepared_engine_lane("AMR Program field publication");
  auto& hierarchy = *p_->prepared_hierarchy;
  std::lock_guard<std::recursive_mutex> execution_lock(hierarchy.execution_mutex);
  using transaction_type = typename ExactAuxiliaryRegistry<Dim>::PublicationTransaction;
  const std::size_t depth = p_->engine->hierarchy().num_levels();
  std::vector<std::string> providers;
  typename Impl::AuxiliaryInvalidations source_invalidations, next_invalidations;
  std::vector<AuxiliaryEvaluationPoint> points;
  std::string contract;
  std::exception_ptr error;
  try {
    if (publication_identity.empty() || levels.size() != depth || depth == 0 ||
        !p_->active_field_slot.empty() || hierarchy.provider_storage.size() != depth ||
        hierarchy.provider_candidate_storage.size() != depth ||
        hierarchy.auxiliary_registries.size() != depth ||
        hierarchy.provider_candidate_ghost_fills.size() != depth ||
        hierarchy.provider_candidate_physical_boundaries.size() != depth)
      throw std::invalid_argument("AMR field publication requires the complete idle hierarchy");
    auto root_point = levels.front().point;
    root_point.level = 0;
    if (root_point.clock.empty() || root_point.tick != p_->macro_step || root_point.substep < 0 ||
        root_point.stage < 0 || root_point.stage_fraction < amr::Rational(0, 1) ||
        amr::Rational(1, 1) < root_point.stage_fraction || !std::isfinite(root_point.dt) ||
        root_point.dt <= 0 || !std::isfinite(root_point.physical_time))
      throw std::invalid_argument(
          "AMR field publication has an incomplete current evaluation point");
    ExactContractBuilder exact;
    exact.text("pops.amr.program-field-publication")
        .scalar(std::uint32_t{2})
        .text(publication_identity)
        .scalar(p_->engine->topology_epoch())
        .scalar(p_->engine->materialization_generation())
        .scalar(depth)
        .text(root_point.clock)
        .scalar(root_point.tick)
        .scalar(root_point.substep)
        .scalar(root_point.stage)
        .scalar(root_point.stage_fraction.numerator)
        .scalar(root_point.stage_fraction.denominator)
        .scalar(root_point.dt)
        .scalar(root_point.physical_time)
        .text(root_point.graph_identity)
        .text(root_point.rate_identity)
        .text(root_point.application_identity);
    std::string root_components;
    points.reserve(depth);
    for (std::size_t level = 0; level < depth; ++level) {
      const auto& incoming = levels[level];
      auto comparison = incoming.point;
      comparison.level = 0;
      if (incoming.point.level != static_cast<int>(level) || comparison != root_point ||
          incoming.components.empty() || !hierarchy.provider_storage[level] ||
          !hierarchy.provider_candidate_storage[level] ||
          !hierarchy.provider_candidate_ghost_fills[level] ||
          !hierarchy.provider_candidate_physical_boundaries[level])
        throw std::invalid_argument("AMR field publication level point or carrier is incomplete");
      const auto& registry = hierarchy.auxiliary_registries[level];
      if (!registry.sealed() || registry.has_pending_publication())
        throw std::logic_error("AMR field publication requires idle sealed level registries");
      AuxiliaryCarrierStorage<Dim>{hierarchy.provider_storage[level].get(),
                                   hierarchy.provider_candidate_storage[level].get()}
          .validate();
      std::vector<std::string> keys;
      ExactContractBuilder envelope;
      envelope.scalar(incoming.components.size());
      for (const auto& item : incoming.components) {
        if (!item.values || item.component < 0 || item.component >= item.values->ncomp() ||
            item.key.space_kind != "field" || item.expected_provider_identity.empty())
          throw std::invalid_argument("AMR field publication observation is invalid");
        const auto& provider = registry.provider_for_key(item.key);
        if (provider.kind() != AuxiliaryProviderKind::field_output ||
            provider.identity() != item.expected_provider_identity)
          throw std::invalid_argument("AMR field observation has the wrong sealed producer");
        const auto address = registry.address_of(item.key);
        const auto* destination = hierarchy.provider_storage[level]->find(address.group);
        if (!destination || address.component >= static_cast<std::size_t>(destination->ncomp()) ||
            destination->layout() != item.values->layout() ||
            destination->distribution() != item.values->distribution() ||
            destination->local_rank() != item.values->local_rank())
          throw std::invalid_argument(
              "AMR field observation does not match its exact level layout");
        const auto key = item.key.exact_key();
        if (std::find(keys.begin(), keys.end(), key) != keys.end())
          throw std::invalid_argument("AMR field publication has duplicate destinations");
        keys.push_back(key);
        if (level == 0 &&
            std::find(providers.begin(), providers.end(), provider.identity()) == providers.end())
          providers.push_back(provider.identity());
        item.key.serialize_exact(envelope);
        envelope.text(provider.identity()).scalar(item.component).scalar(item.values->ncomp());
      }
      for (const auto& item : incoming.components)
        for (const auto& output : registry.provider_for_key(item.key).outputs())
          if (std::find(keys.begin(), keys.end(), output.key.exact_key()) == keys.end())
            throw std::invalid_argument("AMR field publication omits a sealed provider output");
      const auto component_contract = std::move(envelope).release();
      if (level == 0)
        root_components = component_contract;
      else if (component_contract != root_components)
        throw std::invalid_argument("AMR field publication components differ between levels");
      AuxiliaryEvaluationPoint point;
      point.clock = incoming.point.clock;
      point.accepted_step = static_cast<std::uint64_t>(incoming.point.tick);
      point.layout_generation = p_->engine->materialization_generation();
      point.level = static_cast<int>(level);
      point.substep = incoming.point.substep;
      point.stage = incoming.point.stage;
      point.event = AuxiliaryEvaluationEvent::before_residual;
      point.validate();
      points.push_back(point);
      exact.scalar(level)
          .scalar(registry.accepted_generation())
          .bytes(registry.collective_contract())
          .bytes(component_contract);
      point.serialize_exact(exact);
    }
    // These are private source/candidate images, not an acknowledgement of the
    // supplied component labels. Actual complete per-level transactions below
    // establish which sealed providers really publish and which revisions they satisfy.
    source_invalidations = p_->auxiliary_invalidations;
    next_invalidations = p_->bound_auxiliary_invalidations();
    source_invalidations.serialize_exact(exact);
    next_invalidations.serialize_exact(exact);
    contract = std::move(exact).release();
  } catch (...) {
    error = std::current_exception();
  }
  auxiliary_ghost_detail::rethrow_collective_failure(
      error, &lane, "AMR Program field publication preflight failed collectively");
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"amr-program-field-publication", contract}},
                                                lane))
    throw std::invalid_argument("AMR field publication identity differs between MPI ranks");

  // Keep the accepted allocations stable: prepared source/flux views borrow their addresses.
  // Full snapshots include ghosts, provider points and dirty dependencies, matching step rollback.
  std::vector<AuxiliaryStorageGroups<Dim>> accepted;
  std::vector<ExactAuxiliaryRegistry<Dim>> accepted_registries;
  typename Impl::AuxiliaryInvalidations accepted_invalidations;
  std::vector<std::optional<transaction_type>> transactions;
  std::vector<std::map<std::string, std::uint64_t>> captured_obligations;
  error = nullptr;
  try {
    if (p_->auxiliary_invalidations != source_invalidations)
      throw std::logic_error("AMR Program field invalidation source changed before staging");
    accepted.reserve(depth);
    for (const auto& groups : hierarchy.provider_storage)
      accepted.push_back(*groups);
    accepted_registries = hierarchy.auxiliary_registries;
    accepted_invalidations = p_->auxiliary_invalidations;
    transactions.resize(depth);
    captured_obligations.resize(depth);
  } catch (...) {
    error = std::current_exception();
  }
  auxiliary_ghost_detail::rethrow_collective_failure(
      error, &lane, "AMR Program field publication snapshot failed collectively");

  const auto rollback = [&](std::exception_ptr failure, const char* phase) {
    for (auto& transaction : transactions)
      if (transaction)
        transaction->reject();
    std::exception_ptr restore_error;
    try {
      for (std::size_t level = 0; level < depth; ++level) {
        copy_auxiliary_groups_in_place(accepted[level], *hierarchy.provider_storage[level]);
        copy_auxiliary_groups_in_place(accepted[level],
                                       *hierarchy.provider_candidate_storage[level]);
      }
      hierarchy.auxiliary_registries.swap(accepted_registries);
      p_->auxiliary_invalidations.swap(accepted_invalidations);
    } catch (...) {
      restore_error = std::current_exception();
    }
    auxiliary_ghost_detail::rethrow_collective_failure(
        restore_error, &lane, "AMR Program field publication rollback failed collectively");
    auxiliary_ghost_detail::rethrow_collective_failure(failure, &lane, phase);
  };

  error = nullptr;
  try {
    // Every fine candidate ghost fill is already bound to the persistent parent candidate.
    for (std::size_t level = 0; level < depth; ++level) {
      auto& registry = hierarchy.auxiliary_registries[level];
      auto& candidate = *hierarchy.provider_candidate_storage[level];
      std::exception_ptr stage_error;
      try {
        copy_auxiliary_groups_in_place(*hierarchy.provider_storage[level], candidate);
        transactions[level].emplace(registry.begin_external_publication(points[level], providers));
        for (const auto index : registry.topological_order()) {
          const auto& identity = registry.provider(index).identity();
          if (transactions[level]->requires_staging(identity))
            captured_obligations[level].emplace(identity,
                next_invalidations.obligation(level, identity));
        }
        for (const auto& item : levels[level].components) {
          const auto address = registry.address_of(item.key);
          auto& destination = *candidate.find(address.group);
          Impl::copy_scalar_component(*item.values, item.component, destination,
                                      static_cast<int>(address.component));
        }
        for (const auto index : registry.topological_order()) {
          const auto& identity = registry.provider(index).identity();
          if (std::find(providers.begin(), providers.end(), identity) != providers.end())
            transactions[level]->stage_external(identity);
        }
        Kokkos::fence();
      } catch (...) {
        stage_error = std::current_exception();
      }
      auxiliary_ghost_detail::rethrow_collective_failure(
          stage_error, &lane, "AMR Program field candidate staging failed collectively");
      const auto refresh_ghosts = [&] {
        std::exception_ptr ghost_error;
        try {
          hierarchy.provider_candidate_ghost_fills[level](candidate, levels[level].point);
        } catch (...) {
          ghost_error = std::current_exception();
        }
        auxiliary_ghost_detail::rethrow_collective_failure(
            ghost_error, &lane, "AMR Program field hierarchy ghost fill failed collectively");
        hierarchy.provider_candidate_physical_boundaries[level]->execute(candidate);
      };
      refresh_ghosts();
      const auto auxiliary_geometry = prepared_amr_level_geometry(static_cast<int>(level));
      transactions[level]->launch_ready_native(
          {hierarchy.provider_storage[level].get(), &candidate, &auxiliary_geometry},
          [&](const auto&, std::exception_ptr provider_error) {
            auxiliary_ghost_detail::rethrow_collective_failure(
                provider_error, &lane, "AMR Program field prerequisite launch failed collectively");
            refresh_ghosts();
          });
      std::exception_ptr fence_error;
      try { Kokkos::fence(); } catch (...) { fence_error = std::current_exception(); }
      auxiliary_ghost_detail::rethrow_collective_failure(
          fence_error, &lane, "AMR Program field prerequisite fence failed collectively");
      require_finite_auxiliary_groups(candidate, &lane,
                                      "AMR Program field observation publication");
      std::exception_ptr completion_error;
      try {
        transactions[level]->validate_complete();
        std::vector<std::string> published;
        for (const auto index : registry.topological_order()) {
          const auto& identity = registry.provider(index).identity();
          if (transactions[level]->requires_staging(identity)) published.push_back(identity);
        }
        // Fine transport obligations opened by this real parent candidate are
        // visible only in the private candidate until every level is accepted.
        p_->record_auxiliary_publication(next_invalidations, level, published,
                                         captured_obligations[level]);
      } catch (...) {
        completion_error = std::current_exception();
      }
      auxiliary_ghost_detail::rethrow_collective_failure(
          completion_error, &lane, "AMR Program field level completion failed collectively");
    }
  } catch (...) {
    error = std::current_exception();
  }
  if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
    rollback(error, "AMR Program field hierarchy preparation failed collectively");
    return;
  }
  std::string accepted_incidence_contract;
  error = nullptr;
  try {
    if (p_->auxiliary_invalidations != source_invalidations)
      throw std::logic_error("AMR Program field invalidation source changed during production");
    ExactContractBuilder exact;
    exact.text("pops.amr.program-field-publication-incidence").scalar(std::uint32_t{1})
        .text(publication_identity);
    next_invalidations.serialize_exact(exact);
    accepted_incidence_contract = std::move(exact).release();
  } catch (...) { error = std::current_exception(); }
  if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
    rollback(error, "AMR Program field incidence preparation failed collectively");
    return;
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"amr-program-field-accepted-incidence", accepted_incidence_contract}}, lane)) {
    try { throw std::invalid_argument("AMR Program field accepted incidence differs across ranks"); }
    catch (...) { error = std::current_exception(); }
    rollback(error, "AMR Program field accepted incidence differs collectively");
    return;
  }
  error = nullptr;
  try {
    for (std::size_t level = 0; level < depth; ++level)
      copy_auxiliary_groups_in_place(*hierarchy.provider_candidate_storage[level],
                                     *hierarchy.provider_storage[level]);
    for (auto& transaction : transactions)
      transaction->accept();
  } catch (...) {
    error = std::current_exception();
  }
  if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
    rollback(error, "AMR Program field hierarchy commit failed collectively");
    return;
  }
  p_->auxiliary_invalidations.swap(next_invalidations);
}
