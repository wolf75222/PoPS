// Included in AmrProgramContext. The enclosing native restart transaction has
// already staged its candidate carriers before this preflight is entered.
private:
using ContinuationForest = std::map<int, ContinuationStore<Dim>>;
struct ContinuationRestartCandidate {
  ContinuationForest images;
  std::vector<std::uint8_t> source_bytes;
  std::optional<LevelContinuationForestRetention> retention;
};
mutable std::unique_ptr<ContinuationRestartCandidate> continuation_restart_pending_;

template <class Function>
decltype(auto) continuation_checkpoint_at_level_(int level, Function&& function) const {
  using Prototypes = typename ContinuationCodec<Dim>::Prototypes;
  if (level == -1) {
    Prototypes fields = [this](const ContinuationType& type,
                               const ContinuationFieldSupport::Ref& support) -> const field_type& {
      return continuation_field_prototype_(type, -1, support);
    };
    return std::forward<Function>(function)(std::move(fields));
  }
  return with_program_resource_level(level, [&]() -> decltype(auto) {
    Prototypes fields = [this](const ContinuationType& type,
                               const ContinuationFieldSupport::Ref& support) -> const field_type& {
      return continuation_field_prototype_(type, active_level_, support);
    };
    return std::forward<Function>(function)(std::move(fields));
  });
}

void continuation_checkpoint_projection_(
    const ContinuationStore<Dim>& store,
    const typename ContinuationCodec<Dim>::Prototypes& fields) const {
  for (const auto& binding : store.configuration().fields)
    require_continuation_projection<Dim>(continuation_at<Dim>(store.read(), binding.path)->field(),
                                         fields(binding.program_block));
}

// This exporter is local: its callers converge failures before starting a
// collective. It never issues domains, publishes values or rebinds an image.
std::vector<AmrProgramContinuationLevel> continuation_checkpoint_levels_() const {
  const bool initialized =
      std::any_of(amr_continuation_accepted_.begin(), amr_continuation_accepted_.end(),
                  [](const auto& entry) { return entry.second.initialized(); });
  if (!initialized)
    return {};
  continuation_require_forest_(amr_continuation_accepted_);
  std::vector<AmrProgramContinuationLevel> records;
  records.reserve(amr_continuation_accepted_.size());
  for (const auto& [level, store] : amr_continuation_accepted_)
    continuation_checkpoint_at_level_(level, [&](auto prototypes) {
      store.require_frontier(facade_->time(), facade_->macro_step());
      if (level >= 0)
        continuation_checkpoint_projection_(store, prototypes);
      records.push_back(
          {level, runtime_->topology_epoch(), runtime_->materialization_generation(),
           store.domain_high_water(), store.configuration().identity,
           store.codec(prototypes).collective_identity(store.read(), store.publication()),
           store.checkpoint(prototypes)});
    });
  records.front().collective_identity =
      amr_continuation_forest_identity(records, records.front().collective_identity);
  return records;
}

void continuation_prepare_restart_levels_(const std::vector<AmrProgramContinuationLevel>& records,
                                          std::span<const std::uint8_t> bytes, double accepted_time,
                                          std::int64_t accepted_step) const {
  const auto& lane = prepared_execution_lane();
  auto next = continuation_local_phase(
      [&] {
        auto candidate = std::make_unique<ContinuationRestartCandidate>();
        candidate->source_bytes.assign(bytes.begin(), bytes.end());
        if (records.empty()) {
          if (std::any_of(amr_continuation_accepted_.begin(), amr_continuation_accepted_.end(),
                          [](const auto& item) { return item.second.initialized(); }))
            throw std::invalid_argument("AMR restart omits its installed continuation forest");
          candidate->images = amr_continuation_accepted_;
          return candidate;
        }
        if (records.size() != static_cast<std::size_t>(nlev()) + 1 ||
            amr_continuation_accepted_.size() != records.size())
          throw std::invalid_argument("AMR restart does not bind every installed forest schema");
        std::uint64_t workspace = 0;
        for (const auto& [level, store] : amr_continuation_accepted_) {
          (void)level;
          const auto capacity = store.configuration().byte_capacity;
          if (capacity > std::numeric_limits<std::uint64_t>::max() - workspace)
            throw std::length_error("AMR continuation restart identity workspace overflows");
          workspace += capacity;
        }
        typename ContinuationCodec<Dim>::DecodeContext shared_authorities(workspace);
        std::vector<AmrProgramContinuationLevel> authenticated;
        authenticated.reserve(records.size());
        for (std::size_t i = 0; i < records.size(); ++i) {
          const auto& record = records[i];
          if (record.level != static_cast<std::int64_t>(i) - 1 ||
              record.topology_epoch != runtime_->topology_epoch() ||
              record.materialization_generation != runtime_->materialization_generation())
            throw std::invalid_argument(
                "AMR restart continuation lost its exact hierarchy authority");
          const auto& installed = amr_continuation_accepted_.at(record.level);
          if (record.schema_identity != installed.configuration().identity)
            throw std::invalid_argument("AMR restart continuation has a foreign installed schema");
          authenticated.push_back({record.level,
                                   record.topology_epoch,
                                   record.materialization_generation,
                                   record.domain_high_water,
                                   record.schema_identity,
                                   {},
                                   {}});
          continuation_checkpoint_at_level_(record.level, [&](auto prototypes) {
            auto prepared =
                installed.prepare_restart(record.image, prototypes, &lane, &shared_authorities);
            if (record.domain_high_water < prepared.domain_high_water)
              throw std::invalid_argument("AMR restart continuation regresses its domain issuer");
            prepared.domain_high_water = record.domain_high_water;
            auto restored = installed;
            restored.publish(std::move(prepared));
            restored.require_frontier(accepted_time, accepted_step);
            if (record.level >= 0)
              continuation_checkpoint_projection_(restored, prototypes);
            authenticated[i].collective_identity =
                restored.codec(prototypes)
                    .collective_identity(restored.read(), restored.publication());
            if (record.level >= 0 &&
                authenticated[i].collective_identity != record.collective_identity)
              throw std::invalid_argument(
                  "AMR child checkpoint common identity differs from its image");
            if (restored.checkpoint(prototypes) != record.image)
              throw std::invalid_argument(
                  "AMR checkpoint does not contain a canonical complete image");
            candidate->images.emplace(record.level, std::move(restored));
          });
        }
        if (amr_continuation_forest_identity(authenticated,
                                             authenticated.front().collective_identity) !=
            records.front().collective_identity)
          throw std::invalid_argument(
              "AMR root checkpoint does not bind its complete child forest");
        continuation_require_forest_(candidate->images);
        return candidate;
      },
      "AMR continuation forest restart decoding and projection");

  // Rank-local tables cannot prove global reachability. Rebuild the exact shared
  // dependency closure before allowing any image into the accepted map.
  for (const auto& [level, store] : next->images) {
    (void)level;
    if (!store.initialized())
      continue;
    std::span<const std::shared_ptr<distributed::Registry>> registries;
    if (store.closure())
      registries = store.closure()->registries;
    auto closure = continuation_collective_closure<Dim>(store.read(), store.distributed_registry(),
                                                        lane, store.configuration().byte_capacity,
                                                        store.domain_realm(), registries);
    continuation_local_phase(
        [&] {
          if (store.closure()) {
            if (store.closure()->domains != closure->domains ||
                store.closure()->arrays != closure->arrays)
              throw std::invalid_argument("AMR checkpoint tables are not the exact global closure");
          } else if (!closure->domains.empty() || !closure->arrays.empty()) {
            throw std::invalid_argument("AMR checkpoint omits its distributed dependency closure");
          }
        },
        "AMR continuation forest global restart closure");
  }
  continuation_local_phase(
      [&] {
        next->retention.emplace(
            continuation_prepare_forest_retention_(next->images, &amr_continuation_accepted_));
      },
      "AMR continuation restart retained authorities");
  // Publication into the pending slot is atomic; the accepted map is unchanged.
  continuation_restart_pending_.swap(next);
}

std::unique_ptr<ContinuationRestartCandidate> continuation_take_restart_levels_(
    const std::vector<AmrProgramContinuationLevel>& records,
    std::span<const std::uint8_t> bytes) const {
  if (continuation_restart_pending_) {
    const auto& source = continuation_restart_pending_->source_bytes;
    if (source.size() != bytes.size() || !std::equal(source.begin(), source.end(), bytes.begin()))
      throw std::invalid_argument("AMR restart import differs from its authenticated candidate");
    return std::move(continuation_restart_pending_);
  }
  // Ordinary accepted-state refresh imports already published authorities. It
  // must not decode them into a different realm or silently replace their values.
  if (records != continuation_checkpoint_levels_())
    throw std::invalid_argument("AMR continuation import lacks its prepared restart image");
  auto unchanged = std::make_unique<ContinuationRestartCandidate>();
  unchanged->images = amr_continuation_accepted_;
  // No issuer changes on an ordinary accepted-state refresh.
  return unchanged;
}

void continuation_publish_restart_levels_(
    std::unique_ptr<ContinuationRestartCandidate> candidate) const noexcept {
  if (!candidate)
    std::terminate();
  amr_continuation_accepted_.swap(candidate->images);
  if (candidate->retention)
    continuation_install_forest_retention_(std::move(*candidate->retention));
}
void continuation_discard_restart_levels_() const noexcept {
  continuation_restart_pending_.reset();
}

void install_continuation_checkpoint_preflight_(std::shared_ptr<AmrProgramContext> owner) const {
  facade_->install_program_continuation_checkpoint_preflight(

      [owner](std::span<const std::uint8_t> bytes, bool restart) {
        auto state = owner->continuation_local_phase(
            [&] {
              // The restart has already replaced its hierarchy. Rebind only this
              // pointer; resource refresh may be collective and belongs to resync.
              auto* live = require_runtime_(*owner->facade_);
              if (owner->runtime_ != live) {
                if (!owner->active_attempt_states_.empty())
                  throw std::logic_error("AMR checkpoint cannot rebind an active level attempt");
                owner->runtime_ = live;
              }
              auto decoded = deserialize_amr_program_accepted_state<Dim>(
                  bytes,
                  owner->interface_flux_ledger_ ? &owner->interface_flux_ledger_->budget()
                                                : nullptr,
                  owner->max_exact_coefficient_bytes_);
              require_live_amr_program_checkpoint(decoded, *owner->runtime_);
              return decoded;
            },
            "AMR continuation outer checkpoint authentication");
        if (restart) {
          if (state.level_clocks.empty())
            throw std::invalid_argument("AMR restart continuation lacks its accepted frontier");
          const auto& frontier = state.level_clocks.front();
          owner->continuation_prepare_restart_levels_(state.continuation_levels, bytes,
                                                      frontier.physical_time, frontier.macro_step);
        } else {
          owner->continuation_local_phase(
              [&] {
                if (state.continuation_levels != owner->continuation_checkpoint_levels_())
                  throw std::invalid_argument(
                      "AMR checkpoint differs from its accepted native forest");
              },
              "AMR continuation accepted image authentication");
        }
      });

  facade_->install_program_continuation_rematerializer(
      [owner = std::move(owner)](const AmrContinuationMigrationRequest& request) {
        // The outer facade has converged source validation and request identity.
        // Prepare local archive/prototype metadata before entering target collectives.
        const auto& lane = owner->prepared_execution_lane();
        using Prototypes = typename ContinuationCodec<Dim>::Prototypes;
        std::vector<std::vector<AmrProgramContinuationLevel>> ranked;
        std::function<Prototypes(int)> targets;
        distributed::phase(lane, [&] {
          if (!owner->active_attempt_states_.empty())
            throw std::logic_error(
                "AMR continuation migration cannot run inside an active attempt");
          owner->runtime_ = require_runtime_(*owner->facade_);
          ranked.reserve(request.source_states.size());
          for (const auto& bytes : request.source_states) {
            auto state = deserialize_amr_program_accepted_state<Dim>(
                bytes,
                owner->interface_flux_ledger_ ? &owner->interface_flux_ledger_->budget() : nullptr,
                owner->max_exact_coefficient_bytes_);
            ranked.push_back(std::move(state.continuation_levels));
          }
          targets = [owner](int level) -> Prototypes {
            return
                [owner, level](const ContinuationType& type,
                               const ContinuationFieldSupport::Ref& support) -> const field_type& {
                  return owner->continuation_field_prototype_(type, level, support);
                };
          };
        });
        return rematerialize_continuation_forest<Dim>(
            ranked, owner->amr_continuation_accepted_, targets, request.source_level_owners,
            request.source_level_modes, request.target_epoch, request.target_generation,
            request.accepted_time, request.accepted_step, request.workspace_byte_capacity, lane);
      });
}
