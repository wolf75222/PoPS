void store_history_(const std::string& name, const field_type& value) const {
  struct StoreCandidate {
    std::string key;
    std::vector<field_type> rollback_ring;
    std::vector<field_type> ring;
    std::vector<Real> dts;
    std::vector<FluxExpression> expressions;
    std::string exact_contract;
  };
  auto& manager = runtime_state().hist_;
  StoreCandidate candidate = prepare_history_mutation_collectively_(
      [&]() {
        StoreCandidate staged;
        staged.key = history_key_(name, active_level_);
        const auto found = manager.histories.find(staged.key);
        if (found == manager.histories.end() || found->second.empty())
          throw std::out_of_range("AMR Program history is not registered on the active level");
        require_same_field_contract_(found->second.front(), value, "AMR Program history store");
        if (!std::isfinite(current_dt_) || !(current_dt_ > 0.0))
          throw std::logic_error("AMR Program history store has no positive exact interval");
        const auto flux_found = history_flux_expressions_.find(staged.key);
        if (flux_found == history_flux_expressions_.end() ||
            flux_found->second.size() != found->second.size())
          throw std::logic_error("AMR Program history flux provenance differs from its ring depth");
        staged.rollback_ring = found->second;
        staged.ring = staged.rollback_ring;
        staged.dts = manager.slot_dt.at(staged.key);
        staged.expressions = flux_found->second;
        if (staged.dts.size() != staged.ring.size())
          throw std::logic_error("AMR Program history dt ledger differs from its ring depth");
        const FluxExpression active_expression = active_flux_expression_(value);
        staged.ring.front() = value;
        staged.expressions.front() = active_expression;
        if (!manager.initialized.at(staged.key)) {
          for (std::size_t slot = 1; slot < staged.ring.size(); ++slot) {
            staged.ring[slot] = value;
            staged.dts[slot] = static_cast<Real>(current_dt_);
            staged.expressions[slot] = active_expression;
          }
        }
        staged.dts.front() = static_cast<Real>(current_dt_);
        ExactContractBuilder contract;
        contract.text("pops.amr.history-store.v1")
            .bytes(staged.key)
            .scalar(active_level_)
            .scalar(std::bit_cast<std::uint64_t>(current_dt_))
            .scalar(static_cast<std::uint64_t>(staged.ring.size()))
            .scalar(manager.initialized.at(staged.key));
        staged.exact_contract = std::move(contract).release();
        return staged;
      },
      [](const StoreCandidate& staged) -> const std::string& { return staged.exact_contract; },
      "AMR Program history store");

  const auto found = manager.histories.find(candidate.key);
  const auto flux_found = history_flux_expressions_.find(candidate.key);
  // Public history() references name individual ring elements.  Keep those elements at stable
  // addresses.  A MultiFab valid-copy has no allocation path after its exact field contract has
  // been prepared above; nevertheless, converge a device/local exception before metadata moves.
  std::size_t published = 0;
  std::exception_ptr copy_error;
  try {
    for (; published < candidate.ring.size(); ++published)
      copy_valid_(candidate.ring[published], found->second[published]);
  } catch (...) {
    copy_error = std::current_exception();
  }
  const ExecutionLane& lane = prepared_execution_lane();
  if (all_reduce_max(copy_error ? 1L : 0L, lane) != 0) {
    std::exception_ptr rollback_error;
    try {
      // A throwing valid-copy may have written part of the current slot before the loop counter
      // advances.  Restore every stable element, not merely completed slots, on every rank.
      for (std::size_t slot = 0; slot < candidate.rollback_ring.size(); ++slot)
        copy_valid_(candidate.rollback_ring[slot], found->second[slot]);
    } catch (...) {
      rollback_error = std::current_exception();
    }
    (void)all_reduce_max(rollback_error ? 1L : 0L, lane);
    if (lane.size() == 1 && copy_error)
      std::rethrow_exception(copy_error);
    throw std::runtime_error("AMR Program history store valid-copy failed collectively");
  }
  for (std::size_t slot = 0; slot < candidate.dts.size(); ++slot) {
    manager.slot_dt.at(candidate.key)[slot] = candidate.dts[slot];
    flux_found->second[slot].swap(candidate.expressions[slot]);
  }
  manager.initialized.at(candidate.key) = true;
  manager.store_pending.at(candidate.key) = true;
}

void rotate_histories_(std::optional<std::string> clock_identity) const {
  auto& manager = runtime_state().hist_;
  struct RotateCandidate {
    std::vector<std::string> selected;
    std::string exact_contract;
  };
  RotateCandidate candidate = prepare_history_mutation_collectively_(
      [&]() {
        RotateCandidate staged;
        ExactContractBuilder contract;
        contract.text("pops.amr.history-rotate.v1")
            .scalar(active_level_)
            .bytes(clock_identity ? *clock_identity : std::string_view{});
        for (const auto& [key, level] : history_levels_) {
          if (level != active_level_)
            continue;
          const auto ring = manager.histories.find(key);
          const auto clock = manager.clock_identity.find(key);
          const auto dts = manager.slot_dt.find(key);
          const auto expressions = history_flux_expressions_.find(key);
          if (ring == manager.histories.end() || clock == manager.clock_identity.end() ||
              dts == manager.slot_dt.end() || expressions == history_flux_expressions_.end() ||
              ring->second.size() != dts->second.size() ||
              ring->second.size() != expressions->second.size() ||
              static_cast<int>(ring->second.size()) != manager.depth.at(key))
            throw std::runtime_error("AMR Program history registry is incomplete");
          if (!clock_identity || clock->second == *clock_identity) {
            (void)manager.fill_count.at(key);
            (void)manager.store_pending.at(key);
            staged.selected.push_back(key);
            contract.bytes(key).bytes(clock->second).scalar(manager.store_pending.at(key));
          }
        }
        staged.exact_contract = std::move(contract).release();
        return staged;
      },
      [](const RotateCandidate& staged) -> const std::string& { return staged.exact_contract; },
      "AMR Program history rotation");
  for (const std::string& key : candidate.selected) {
    auto& ring = manager.histories.at(key);
    for (std::size_t slot = ring.size(); slot-- > 1;)
      std::swap(ring[slot], ring[slot - 1]);
    auto& dts = manager.slot_dt.at(key);
    for (std::size_t slot = dts.size(); slot-- > 1;)
      std::swap(dts[slot], dts[slot - 1]);
    auto& expressions = history_flux_expressions_.at(key);
    for (std::size_t slot = expressions.size(); slot-- > 1;)
      std::swap(expressions[slot], expressions[slot - 1]);
    if (manager.store_pending.at(key)) {
      manager.fill_count.at(key) =
          std::min(static_cast<int>(ring.size()), manager.fill_count.at(key) + 1);
      manager.store_pending.at(key) = false;
      pending_history_remaps_.erase(key);
      deferred_history_lag_scratches_.erase(key);
    }
  }
}

std::optional<ScheduleCoordinate> schedule_coordinate_(ScheduleDomainKind kind,
                                                       const std::string& clock,
                                                       const std::string& stage_identity,
                                                       int level) const {
  return clock_schedule_.coordinate(kind, clock, stage_identity, level, active_level_,
                                    static_cast<std::int64_t>(macro_step()));
}

OperatorFingerprint operator_topology_(const field_type& prototype) const {
  require_same_layout_(prototype, state(0), "AMR Program operator topology");
  OperatorFingerprint fingerprint =
      ::pops::detail::layout_fingerprint(prototype, program_resource_vector_distribution());
  ::pops::detail::fingerprint_geometry(fingerprint, geometry());
  ::pops::detail::fingerprint_mix(fingerprint, runtime_->spatial_contract());
  ::pops::detail::fingerprint_mix(fingerprint, runtime_->topology_epoch());
  ::pops::detail::fingerprint_mix(fingerprint, runtime_->materialization_generation());
  ::pops::detail::fingerprint_mix(fingerprint, static_cast<std::uint64_t>(active_level_));
  return fingerprint;
}

OperatorEvaluationSnapshot current_operator_snapshot_(OperatorFingerprint authority,
                                                      OperatorFingerprint topology,
                                                      OperatorFingerprint resources,
                                                      std::uint64_t revision) const {
  refresh_resources_();
  const std::uint64_t maximum_generation =
      std::max(runtime_->topology_epoch(), runtime_->materialization_generation());
  if (maximum_generation == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("AMR Program topology revision exhausted uint64_t");
  const double evaluation_time =
      static_cast<double>(physical_time()) + stage_time_.value() * current_dt_;
  return {authority,
          revision,
          static_cast<std::int64_t>(macro_step()),
          stage_time_.numerator,
          stage_time_.denominator,
          std::bit_cast<std::uint64_t>(current_dt_),
          std::bit_cast<std::uint64_t>(evaluation_time),
          maximum_generation + 1,
          topology,
          resources};
}

static void require_same_layout_(const field_type& left, const field_type& right,
                                 std::string_view operation) {
  if (left.layout() != right.layout() || left.distribution() != right.distribution() ||
      left.local_rank() != right.local_rank() || left.local_size() != right.local_size())
    throw std::invalid_argument(std::string(operation) + " fields have different exact layouts");
}
static void require_same_field_contract_(const field_type& left, const field_type& right,
                                         std::string_view operation) {
  require_same_layout_(left, right, operation);
  if (left.ncomp() != right.ncomp())
    throw std::invalid_argument(std::string(operation) + " fields have different components");
}
static void require_scalar_stencil_(const field_type& output, const field_type& input,
                                    int output_components, std::string_view operation) {
  require_same_layout_(output, input, operation);
  if (input.ncomp() != 1 || output.ncomp() != output_components)
    throw std::invalid_argument(std::string(operation) + " has an invalid component contract");
  for (int axis = 0; axis < Dim; ++axis)
    if (input.ghosts()[axis] < 1)
      throw std::invalid_argument(std::string(operation) + " requires one ghost per axis");
}
void require_boundary_point_(const runtime::multiblock::BoundaryEvaluationPoint& point,
                             std::string_view operation) const {
  if (point.level != active_level_ || point.clock.empty() || point.stage < 0 ||
      point.stage_fraction.denominator <= 0)
    throw std::invalid_argument(std::string(operation) + " has a foreign evaluation point");
}

void require_current_boundary_point_exact_(
    const runtime::multiblock::BoundaryEvaluationPoint& point, std::string_view operation) const {
  require_rate_identity_(point.stage);
  const double physical_time = current_interval_start_time_ + stage_time_.value() * current_dt_;
  if (primary_clock_.empty() || !std::isfinite(current_dt_) || !(current_dt_ > 0.0) ||
      point.clock != primary_clock_ ||
      point.tick != static_cast<std::int64_t>(facade_->macro_step()) ||
      point.level != active_level_ || point.substep != logical_substep_ ||
      point.stage_fraction != stage_time_ || point.dt != current_dt_ ||
      point.physical_time != physical_time || !point.graph_identity.empty() ||
      !point.rate_identity.empty() || !point.application_identity.empty())
    throw std::invalid_argument(std::string(operation) +
                                " has a stale or foreign exact evaluation point");
}

static std::string history_key_(const std::string& name, int level) {
  if (name.empty() || level < 0)
    throw std::invalid_argument("AMR Program history key is invalid");
  return "pops.amr.level-history.v1/" + std::to_string(level) + "/" + std::to_string(name.size()) +
         ":" + name;
}

void require_history_owner_(int program_owner) const {
  if (program_owner < 0 || sys_block(program_owner) < 0)
    throw std::invalid_argument("AMR Program history has a foreign block owner");
}

field_type& deferred_history_lag_(const std::string& key,
                                  const std::vector<field_type>& ring) const {
  auto& manager = runtime_state().hist_;
  struct DeferredCandidate {
    AmrProgramPendingHistoryRemap marker;
    std::map<std::string, field_type> scratches;
    FluxExpressionRegistry expressions;
    std::uint64_t next_identity = 0;
    std::string exact_contract;
  };
  DeferredCandidate candidate = prepare_history_mutation_collectively_(
      [&]() {
        const auto pending = pending_history_remaps_.find(key);
        if (pending == pending_history_remaps_.end())
          throw std::logic_error("AMR Program deferred history lag has no pending remap");
        const auto& marker = pending->second;
        if (marker.consumed || ring.size() != 2 || marker.child_level != active_level_ ||
            marker.published_topology_epoch != runtime_->topology_epoch() ||
            marker.published_materialization_generation != runtime_->materialization_generation() ||
            !manager.store_pending.at(key) || manager.slot_dt.at(key).size() != 2 ||
            static_cast<double>(manager.slot_dt.at(key)[0]) != current_dt_ ||
            static_cast<double>(manager.slot_dt.at(key)[1]) != marker.source_dt ||
            marker.temporal_denominator != 1 ||
            (marker.temporal_numerator != 1 && marker.temporal_numerator != 2) ||
            marker.target_dt != marker.source_dt / static_cast<double>(marker.temporal_numerator))
          throw std::runtime_error(
              "AMR Program deferred history lag lost its exact IntegralOnly authority");
        const auto provenance = history_flux_expressions_.find(key);
        if (provenance == history_flux_expressions_.end() ||
            provenance->second.size() != ring.size())
          throw std::logic_error(
              "AMR Program deferred history lag lacks its persistent ring provenance");
        DeferredCandidate staged{
            marker, deferred_history_lag_scratches_, {}, next_active_flux_basis_identity_, {}};
        auto [scratch, inserted] = staged.scratches.try_emplace(key, ring.front());
        (void)inserted;
        require_same_field_contract_(scratch->second, ring.front(),
                                     "AMR Program deferred history lag");
        if (marker.temporal_numerator == 1)
          copy_valid_(ring[1], scratch->second);
        else
          pops::lincomb(scratch->second, Real(0.5), ring[0], Real(0.5), ring[1]);

        // A map swap preserves every node address.  Rebind all existing deferred-scratch
        // registry entries to their staged nodes before the old map is replaced, otherwise a
        // repeated deferred read could leave a live expression pointing at a destroyed map.
        staged.expressions = active_flux_expressions_;
        for (const auto& [scratch_key, live_scratch] : deferred_history_lag_scratches_) {
          const auto live_expression = staged.expressions.find(&live_scratch);
          if (live_expression == staged.expressions.end())
            continue;
          const auto staged_scratch = staged.scratches.find(scratch_key);
          if (staged_scratch == staged.scratches.end())
            throw std::logic_error("AMR Program deferred history scratch staging lost a live key");
          FluxExpression expression = std::move(live_expression->second);
          staged.expressions.erase(live_expression);
          staged.expressions.emplace(&staged_scratch->second, std::move(expression));
        }

        FluxExpression current_expression;
        FluxExpression lag_expression;
        FluxExpression detached;
        if (marker.temporal_numerator == 1) {
          auto [rehydrated_lag, next_identity] = rehydrated_history_flux_expression_(
              key, 1, active_flux_expressions_, staged.next_identity);
          staged.next_identity = next_identity;
          lag_expression = std::move(rehydrated_lag);
          if (!lag_expression.empty()) {
            staged.expressions[&ring[1]] = lag_expression;
            detached = lag_expression;
          }
        } else {
          current_expression = provenance->second[0];
          auto [rehydrated_lag, next_identity] = rehydrated_history_flux_expression_(
              key, 1, active_flux_expressions_, staged.next_identity);
          lag_expression = std::move(rehydrated_lag);
          if (current_expression.empty() != lag_expression.empty())
            throw std::logic_error("AMR Program deferred history lag has no lag provenance");
          staged.next_identity = next_identity;
          if (!current_expression.empty()) {
            staged.expressions[&ring[1]] = lag_expression;
            detached = scaled_flux_expression_(current_expression,
                                               ExactPolynomial{{0, ::pops::amr::Rational(1, 2)}});
            add_flux_expression_(
                detached, scaled_flux_expression_(
                              lag_expression, ExactPolynomial{{0, ::pops::amr::Rational(1, 2)}}));
            require_flux_expression_budget_(detached);
          }
        }
        const std::size_t detached_term_count = detached.size();
        if (!detached.empty())
          staged.expressions[&scratch->second] = std::move(detached);
        ExactContractBuilder contract;
        contract.text("pops.amr.deferred-history-lag.v1")
            .bytes(key)
            .scalar(marker.parent_level)
            .scalar(marker.child_level)
            .scalar(marker.prior_topology_epoch)
            .scalar(marker.prior_materialization_generation)
            .scalar(marker.published_topology_epoch)
            .scalar(marker.published_materialization_generation)
            .scalar(marker.accepted_macro_step)
            .scalar(marker.temporal_numerator)
            .scalar(marker.temporal_denominator)
            .scalar(std::bit_cast<std::uint64_t>(marker.source_dt))
            .scalar(std::bit_cast<std::uint64_t>(marker.target_dt))
            .scalar(staged.next_identity)
            .scalar(static_cast<std::uint64_t>(current_expression.size()))
            .scalar(static_cast<std::uint64_t>(lag_expression.size()))
            .scalar(static_cast<std::uint64_t>(detached_term_count))
            .scalar(static_cast<std::uint64_t>(staged.scratches.size()))
            .scalar(static_cast<std::uint64_t>(staged.expressions.size()));
        staged.exact_contract = std::move(contract).release();
        return staged;
      },
      [](const DeferredCandidate& staged) -> const std::string& { return staged.exact_contract; },
      "AMR Program deferred history lag");
  static_assert(std::is_nothrow_swappable_v<decltype(deferred_history_lag_scratches_)>);
  static_assert(std::is_nothrow_swappable_v<FluxExpressionRegistry>);
  deferred_history_lag_scratches_.swap(candidate.scratches);
  active_flux_expressions_.swap(candidate.expressions);
  next_active_flux_basis_identity_ = candidate.next_identity;
  // Profiling is observational: after the no-throw publication boundary it must not turn a
  // committed deferred scratch/provenance pair into a reported failed history read.
  try {
    count_kernel_();
  } catch (...) {
  }
  return deferred_history_lag_scratches_.at(key);
}

field_type& history_slot_(const std::string& name, int lag, bool zero_start, int components) const {
  const std::string key = history_key_(name, active_level_);
  auto& manager = runtime_state().hist_;
  const auto found = manager.histories.find(key);
  if (found == manager.histories.end() || lag < 0 || lag >= manager.depth.at(key))
    throw std::out_of_range("AMR Program history slot is absent");
  field_type& result = found->second[static_cast<std::size_t>(lag)];
  if (components >= 0 && result.ncomp() != components)
    throw std::invalid_argument("AMR Program history component contract differs");
  if (!manager.initialized.at(key)) {
    if (!zero_start)
      throw std::runtime_error("AMR Program history has not been initialized");
    result.set_val(Real(0));
  }
  if (lag == 1 && pending_history_remaps_.contains(key))
    return deferred_history_lag_(key, found->second);
  rehydrate_history_flux_expression_(key, lag, result);
  return result;
}
