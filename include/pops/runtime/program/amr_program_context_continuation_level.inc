// Included in AmrProgramContext. Level trials remain provisional until the
// independently assessed macro parent publishes the complete forest.
public:
using level_continuation_authority_type = ContinuationAuthority<Dim>;
using level_continuation_ref = ContinuationValueRef<Dim>;
using level_continuation_store_type = ContinuationStore<Dim>;
struct LevelContinuationRetention {
  FiniteDomainRealmRef realm;
  std::vector<FiniteDomainRef> replicated, local;
  std::shared_ptr<distributed::Registry> distributed;
  std::vector<std::shared_ptr<distributed::Registry>> protected_distributed;
};
struct LevelContinuationForestRetention {
  struct Entry {
    LevelContinuationRetention keep;
    FiniteDomainRegistry::PreparedAdoption replicated, local;
  };
  std::map<int, std::shared_ptr<AmrFiniteValueLifetime<Dim>>> lifetimes;
  std::map<int, Entry> entries;
};

// Only accepted full images or an authenticated candidate forest may reach here.
// Preparation is side-effect free; every allocation precedes facade publication.
LevelContinuationForestRetention continuation_prepare_forest_retention_(
    const std::map<int, level_continuation_store_type>& stores,
    const std::map<int, level_continuation_store_type>* rollback = nullptr) const {
  LevelContinuationForestRetention prepared;
  prepared.lifetimes = finite_value_lifetimes_;
  LevelContinuationRetention all;
  const auto collect = [&](const auto& forest) {
    for (const auto& [level, store] : forest) {
      (void)level;
      if (store.initialized())
        level_continuation_merge_retention_(
            all, level_continuation_collect_local_retention_(store, nullptr));
    }
  };
  collect(stores);
  if (rollback)
    collect(*rollback);
  for (const auto& [level, store] : stores) {
    (void)store;
    if (!prepared.lifetimes.contains(level))
      prepared.lifetimes.emplace(level, std::make_shared<AmrFiniteValueLifetime<Dim>>());
  }
  if (rollback)
    for (const auto& [level, store] : *rollback) {
      (void)store;
      if (!prepared.lifetimes.contains(level))
        prepared.lifetimes.emplace(level, std::make_shared<AmrFiniteValueLifetime<Dim>>());
    }
  for (const auto& [level, live] : prepared.lifetimes) {
    auto keep = all;
    const auto own = stores.find(level);
    if (own != stores.end() && own->second.initialized()) {
      keep.realm = own->second.domain_realm();
      keep.distributed = own->second.distributed_registry();
    }
    auto replicated = live->finite.prepare_adoption(keep.replicated, keep.realm);
    if (own != stores.end() && own->second.initialized())
      replicated.high_water = std::max(replicated.high_water, own->second.domain_high_water());
    if (rollback) {
      const auto old = rollback->find(level);
      if (old != rollback->end() && old->second.initialized())
        replicated.high_water = std::max(replicated.high_water, old->second.domain_high_water());
    }
    auto local = live->local_finite.prepare_adoption(keep.local);
    prepared.entries.emplace(level, typename LevelContinuationForestRetention::Entry{
                                        std::move(keep), std::move(replicated), std::move(local)});
  }
  return prepared;
}
void continuation_install_forest_retention_(
    LevelContinuationForestRetention&& prepared) const noexcept {
  for (auto& [level, live] : finite_value_lifetimes_) {
    const auto restored = prepared.lifetimes.find(level);
    if (restored == prepared.lifetimes.end() || restored->second != live) {
      live->retained_realm.reset();
      live->retained_domains.clear();
      live->retained_local_domains.clear();
      live->retained_distributed.reset();
      live->protected_distributed.clear();
      live->finish();
    }
  }
  for (auto& [level, entry] : prepared.entries) {
    auto& live = *prepared.lifetimes.find(level)->second;
    // A rollback plan may predate issuance. Revoke those newly issued aliases
    // before replacing the tracking image, while preserving the burned ordinal.
    live.finite.revoke_unretained(entry.keep.replicated);
    live.local_finite.revoke_unretained(entry.keep.local);
    live.finite.install_adoption(std::move(entry.replicated));
    live.local_finite.install_adoption(std::move(entry.local));
    if (live.distributed && live.distributed != entry.keep.distributed &&
        std::find(entry.keep.protected_distributed.begin(), entry.keep.protected_distributed.end(),
                  live.distributed) == entry.keep.protected_distributed.end())
      live.distributed->revoke();
    live.retained_realm.swap(entry.keep.realm);
    live.retained_domains.swap(entry.keep.replicated);
    live.retained_local_domains.swap(entry.keep.local);
    live.retained_distributed.swap(entry.keep.distributed);
    live.protected_distributed.swap(entry.keep.protected_distributed);
  }
  finite_value_lifetimes_.swap(prepared.lifetimes);
}
struct LevelContinuationPending {
  typename level_continuation_store_type::Prepared image;
  typename level_continuation_authority_type::Accepted accepted;
  level_continuation_ref envelope, report;
};

void begin_level_continuation_attempt(double macro_start, double macro_end) const {
  continuation_local_phase(
      [&] {
        if (!std::isfinite(macro_start) || !std::isfinite(macro_end) || !(macro_start < macro_end))
          throw std::invalid_argument("AMR continuation requires a finite forward macro interval");
        std::map<int, level_continuation_store_type> working, bases;
        std::map<int, LevelContinuationRetention> rollback;
        for (const auto& [level, store] : amr_continuation_accepted_) {
          if (level < 0)
            continue;
          if (store.initialized()) {
            level_continuation_require_time_(store, store.read(), macro_start);
            rollback.emplace(level, level_continuation_collect_retention_(store, nullptr));
          }
          working.emplace(level, store);
          bases.emplace(level, store);
        }
        for (auto& [level, authority] : level_continuation_authorities_) {
          (void)level;
          authority->revoke();
        }
        // Retire previous-macro temporary domains only after the current accepted
        // closure is known. New issuers imported by restart retain exact high-water marks.
        for (const auto& [level, keep] : rollback) {
          auto& live = finite_value_lifetimes_[level];
          if (!live)
            live = std::make_shared<AmrFiniteValueLifetime<Dim>>();
          if (live->active)
            throw std::logic_error("AMR macro entry still has a suspended finite invocation");
          auto prepared_keep = keep;
          if (keep.realm)
            live->finite.select_realm(keep.realm);
          live->finite.adopt_domains(keep.replicated);
          live->local_finite.adopt_domains(keep.local);
          live->retained_realm.swap(prepared_keep.realm);
          live->retained_domains.swap(prepared_keep.replicated);
          live->retained_local_domains.swap(prepared_keep.local);
          live->retained_distributed.swap(prepared_keep.distributed);
          live->protected_distributed.swap(prepared_keep.protected_distributed);
          live->finish();
        }
        level_continuation_working_.swap(working);
        level_continuation_macro_bases_.swap(bases);
        level_continuation_rollback_keep_.swap(rollback);
        level_continuation_pending_.clear();
        level_continuation_macro_start_ = macro_start;
        level_continuation_macro_end_ = macro_end;
        level_continuation_sync_ = false;
        level_continuation_macro_attempt_ = level_continuation_lifetime_.begin_attempt();
        level_continuation_attempt_active_ = true;
      },
      "AMR child continuation macro-entry snapshot");
}

void level_continuation_configure(ContinuationConfiguration configuration) const {
  continuation_local_phase(
      [&] {
        if (!level_continuation_attempt_active_ || active_level_ < 0)
          throw std::logic_error("AMR child continuation has no active level attempt");
        configuration.validate();
        if (configuration.fields.size() != static_cast<std::size_t>(n_blocks()))
          throw std::invalid_argument("AMR child continuation must bind every block at this level");
        std::set<int> owners;
        for (const auto& binding : configuration.fields) {
          if (!owners.insert(sys_block(binding.program_block)).second ||
              binding.clock_identity != primary_clock_ ||
              configuration.at(binding.path).field_components !=
                  state(binding.program_block).ncomp())
            throw std::invalid_argument("AMR child continuation has foreign field/clock authority");
        }
        level_continuation_working_[active_level_].configure(configuration);
        level_continuation_macro_bases_[active_level_].configure(std::move(configuration));
      },
      "AMR child continuation schema installation");
}

bool level_continuation_initialized() const {
  const bool initialized = level_continuation_selected_store_().initialized();
  continuation_word_agrees_(initialized ? 1 : 0,
                            "AMR child continuation initialization differs across ranks");
  return initialized;
}
std::uint64_t level_continuation_byte_capacity() const {
  return level_continuation_selected_store_().configuration().byte_capacity;
}
level_continuation_authority_type& level_continuation_authority() const {
  auto* selected = continuation_local_phase(
      [&]() -> level_continuation_authority_type* {
        auto& authority = level_continuation_authorities_[active_level_];
        if (!authority)
          authority = std::make_shared<level_continuation_authority_type>();
        if (authority->logical_state()[1] == 0)
          authority->begin(level_continuation_macro_attempt_);
        return authority.get();
      },
      "AMR child continuation authority");
  return *selected;
}

void level_continuation_initialize(level_continuation_ref initial) const {
  if (level_continuation_sync_)
    throw std::logic_error("AMR synchronization cannot invent a missing macro-entry image");
  auto& store = level_continuation_selected_store_();
  auto prepared = level_continuation_prepare_(store, std::move(initial), true);
  auto keep = continuation_local_phase(
      [&] {
        level_continuation_require_frontier_(store, prepared.value, current_interval_start_time_,
                                             static_cast<std::int64_t>(facade_->macro_step()));
        level_continuation_require_projection_(store, prepared.value);
        return level_continuation_collect_retention_(store, &prepared);
      },
      "AMR child continuation initial frontier");
  auto base_copy = continuation_local_phase(
      [&] {
        auto copy = store;
        auto image_copy = prepared;
        copy.publish(std::move(image_copy));
        return copy;
      },
      "AMR immutable macro-entry initial image");
  level_continuation_image_consensus_(store, prepared);
  level_continuation_adopt_retention_(keep);
  store.publish(std::move(prepared));
  // Copies share only immutable image data. The Base can never follow a later trial publication.
  level_continuation_macro_bases_.at(active_level_).swap(base_copy);
  level_continuation_install_retention_(std::move(keep));
}

typename level_continuation_authority_type::Base level_continuation_read() const {
  auto& store = level_continuation_selected_store_();
  auto& live = finite_values_();
  const auto realm = store.domain_realm();
  continuation_word_agrees_(realm ? 1 : 0, "AMR child continuation realm differs across ranks");
  if (realm)
    continuation_synchronize_realm_(realm);
  auto& authority = level_continuation_authority();
  auto base = continuation_local_phase(
      [&] {
        level_continuation_require_time_(store, store.read(),
                                         level_continuation_sync_ ? level_continuation_macro_start_
                                                                  : current_interval_start_time_);
        if (!level_continuation_sync_)
          level_continuation_require_projection_(store, store.read());
        live.finite.import_high_water(store.domain_high_water());
        if (realm)
          live.finite.select_realm(realm);
        live.distributed = store.distributed_registry();
        const auto bind_registry = [&](const auto& registry) {
          if (!registry)
            return;
          live.transport.bind_distributed_registry(
              *registry, [registry](distributed::Identity version, distributed::DomainRef domain,
                                    const ContinuationType& element) {
                registry->authenticate(domain);
                auto array = std::static_pointer_cast<const ContinuationDistributedArray<Dim>>(
                    registry->resolve_value(version, distributed::array_type_tag<Dim>()));
                if (!array || array->domain != domain || array->element != element ||
                    array->version.issuer != version.issuer ||
                    array->version.ordinal != version.ordinal)
                  throw std::invalid_argument("AMR restored Array lost its exact vault version");
                return array;
              });
        };
        bind_registry(live.distributed);
        if (store.closure())
          for (const auto& registry : store.closure()->registries)
            bind_registry(registry);
        live.transport.capture(store.read());
        if (store.closure())
          for (const auto& array : store.closure()->arrays)
            live.transport.capture(ContinuationValue<Dim>::distributed_array(array));
        return authority.read(store);
      },
      "AMR child continuation accepted read");
  return base;
}

void level_continuation_publish(
    const typename level_continuation_authority_type::Accepted& accepted) const {
  if (level_continuation_sync_)
    throw std::logic_error("AMR synchronized child must await its macro parent publication");
  auto& store = level_continuation_selected_store_();
  auto& authority = level_continuation_authority();
  level_continuation_selected_consensus_(authority, accepted);
  auto value = continuation_local_phase(
      [&] {
        auto out = authority.require_root_publication(accepted, store);
        const auto old_step =
            continuation_at<Dim>(store.read(), store.configuration().step_path)->integer();
        if (old_step == std::numeric_limits<std::int64_t>::max())
          throw std::overflow_error("AMR child continuation step exhausted");
        level_continuation_require_frontier_(store, out, current_interval_start_time_ + current_dt_,
                                             old_step + 1);
        return out;
      },
      "AMR child trial publication frontier");
  auto prepared = level_continuation_prepare_(store, std::move(value), false);
  std::vector<std::pair<field_type*, const field_type*>> projections;
  auto keep = continuation_local_phase(
      [&] {
        for (const auto& binding : store.configuration().fields)
          projections.emplace_back(&state(binding.program_block),
                                   &continuation_at<Dim>(prepared.value, binding.path)->field());
        return level_continuation_collect_retention_(
            level_continuation_macro_bases_.at(active_level_), &prepared);
      },
      "AMR child trial complete publication preparation");
  level_continuation_image_consensus_(store, prepared);
  level_continuation_adopt_retention_(keep);
  commit_many(std::span<const std::pair<field_type*, const field_type*>>(projections));
  store.publish(std::move(prepared));
  level_continuation_install_retention_(std::move(keep));
  authority.revoke();
}

void begin_level_continuation_sync(double macro_end) const {
  continuation_local_phase(
      [&] {
        if (!level_continuation_attempt_active_ ||
            std::bit_cast<std::uint64_t>(macro_end) !=
                std::bit_cast<std::uint64_t>(level_continuation_macro_end_) ||
            !level_continuation_macro_bases_.at(active_level_).initialized() ||
            !level_continuation_working_.at(active_level_).initialized())
          throw std::logic_error(
              "AMR child synchronization lost its complete macro-entry authority");
        level_continuation_require_time_(level_continuation_working_.at(active_level_),
                                         level_continuation_working_.at(active_level_).read(),
                                         macro_end);
        level_continuation_sync_ = true;
        auto& authority = level_continuation_authorities_[active_level_];
        if (!authority)
          authority = std::make_shared<level_continuation_authority_type>();
        authority->begin(level_continuation_macro_attempt_);
      },
      "AMR child synchronization entry");
}

void prepare_level_continuation_sync(
    const typename level_continuation_authority_type::Accepted& accepted,
    level_continuation_ref report = {}) const {
  if (!level_continuation_sync_)
    throw std::logic_error("AMR child synchronization has no selected macro-entry base");
  auto& base = level_continuation_macro_bases_.at(active_level_);
  auto& authority = level_continuation_authority();
  level_continuation_selected_consensus_(authority, accepted);
  auto value = continuation_local_phase(
      [&] {
        auto out = authority.require_root_publication(accepted, base);
        const auto old_step =
            continuation_at<Dim>(base.read(), base.configuration().step_path)->integer();
        if (old_step == std::numeric_limits<std::int64_t>::max())
          throw std::overflow_error("AMR synchronized child step exhausted");
        level_continuation_require_frontier_(base, out, level_continuation_macro_end_,
                                             old_step + 1);
        if (report && report != accepted.value()->members()[2])
          throw std::invalid_argument("AMR child report is not its exact accepted observations");
        return out;
      },
      "AMR synchronized child full candidate");
  auto prepared = level_continuation_prepare_(base, std::move(value), false);
  auto keep = continuation_local_phase(
      [&] { return level_continuation_collect_retention_(base, &prepared); },
      "AMR synchronized child retained domain closure");
  level_continuation_image_consensus_(base, prepared);
  level_continuation_adopt_retention_(keep);
  continuation_local_phase(
      [&] {
        if (level_continuation_pending_.contains(active_level_))
          throw std::logic_error("AMR child synchronization was published twice");
        level_continuation_pending_.emplace(
            active_level_, LevelContinuationPending{std::move(prepared), accepted, accepted.value(),
                                                    accepted.value()->members()[2]});
      },
      "AMR synchronized child staging");
  level_continuation_install_retention_(std::move(keep));
}

void rollback_level_continuation_attempt() const noexcept {
  for (auto& [level, live] : finite_value_lifetimes_) {
    if (level < 0)
      continue;  // Parent owns the global controller rollback image.
    const auto found = level_continuation_rollback_keep_.find(level);
    if (found != level_continuation_rollback_keep_.end()) {
      live->retained_realm.swap(found->second.realm);
      live->retained_domains.swap(found->second.replicated);
      live->retained_local_domains.swap(found->second.local);
      live->retained_distributed.swap(found->second.distributed);
      live->protected_distributed.swap(found->second.protected_distributed);
    } else {
      live->retained_realm.reset();
      live->retained_domains.clear();
      live->retained_local_domains.clear();
      live->retained_distributed.reset();
      live->protected_distributed.clear();
    }
  }
  for (auto& [level, authority] : level_continuation_authorities_) {
    (void)level;
    authority->revoke();
  }
  revoke_finite_values_on_rollback();
  level_continuation_pending_.clear();
  level_continuation_working_.clear();
  level_continuation_macro_bases_.clear();
  level_continuation_lifetime_.reject_attempt();
  level_continuation_attempt_active_ = false;
  level_continuation_sync_ = false;
}

void finish_level_continuation_attempt() const noexcept {
  for (auto& [level, authority] : level_continuation_authorities_) {
    (void)level;
    authority->revoke();
  }
  level_continuation_lifetime_.reject_attempt();
  level_continuation_pending_.clear();
  level_continuation_working_.clear();
  level_continuation_macro_bases_.clear();
  // Keep rollback closures alive until the enclosing macro transaction has
  // returned successfully. The next macro entry retires those old domains.
  level_continuation_attempt_active_ = false;
  level_continuation_sync_ = false;
}

PreparedResourceAttempt level_continuation_macro_attempt() const {
  if (!level_continuation_attempt_active_ || !level_continuation_macro_attempt_.visible())
    throw std::logic_error("AMR continuation has no live macro authority");
  return level_continuation_macro_attempt_;
}

private:
mutable std::map<int, level_continuation_store_type> level_continuation_working_;
mutable std::map<int, level_continuation_store_type> level_continuation_macro_bases_;
mutable std::map<int, LevelContinuationPending> level_continuation_pending_;
mutable std::map<int, std::shared_ptr<level_continuation_authority_type>>
    level_continuation_authorities_;
mutable std::map<int, LevelContinuationRetention> level_continuation_rollback_keep_;
mutable PreparedResourceCache level_continuation_lifetime_;
mutable PreparedResourceAttempt level_continuation_macro_attempt_;
mutable bool level_continuation_sync_ = false, level_continuation_attempt_active_ = false;
mutable double level_continuation_macro_start_ = 0, level_continuation_macro_end_ = 0;

level_continuation_store_type& level_continuation_selected_store_() const {
  if (!level_continuation_attempt_active_)
    throw std::logic_error("AMR child continuation has no active macro attempt");
  return level_continuation_sync_ ? level_continuation_macro_bases_.at(active_level_)
                                  : level_continuation_working_.at(active_level_);
}
typename ContinuationCodec<Dim>::Prototypes level_continuation_prototypes_() const {
  return [this](int block) -> const field_type& { return state(block); };
}
void level_continuation_require_time_(const level_continuation_store_type& store,
                                      const level_continuation_ref& value, double time) const {
  if (std::bit_cast<std::uint64_t>(
          continuation_at<Dim>(value, store.configuration().time_path)->real()) !=
      std::bit_cast<std::uint64_t>(time))
    throw std::logic_error("AMR continuation image has a different physical frontier");
}
void level_continuation_require_frontier_(const level_continuation_store_type& store,
                                          const level_continuation_ref& value, double time,
                                          std::int64_t step) const {
  level_continuation_require_time_(store, value, time);
  if (continuation_at<Dim>(value, store.configuration().step_path)->integer() != step)
    throw std::logic_error("AMR continuation image has a different step frontier");
}
void level_continuation_require_projection_(const level_continuation_store_type& store,
                                            const level_continuation_ref& value) const {
  for (const auto& binding : store.configuration().fields)
    require_continuation_projection<Dim>(continuation_at<Dim>(value, binding.path)->field(),
                                         state(binding.program_block));
}
void level_continuation_selected_consensus_(
    level_continuation_authority_type& authority,
    const typename level_continuation_authority_type::Accepted& accepted) const {
  const auto selected = continuation_local_phase(
      [&] { return authority.publication_state(accepted); }, "AMR selected child capability");
  for (const auto word : selected)
    continuation_word_agrees_(word, "AMR selected child differs across execution ranks");
}
typename level_continuation_store_type::Prepared level_continuation_prepare_(
    const level_continuation_store_type& store, level_continuation_ref value,
    bool initialize) const {
  auto& live = finite_values_();
  auto authorized = continuation_local_phase(
      [&] {
        auto authorized =
            level_continuation_collect_retention_(store, nullptr).protected_distributed;
        for (const auto& registry : live.protected_distributed)
          if (std::find(authorized.begin(), authorized.end(), registry) == authorized.end())
            authorized.push_back(registry);
        if (live.distributed &&
            std::find(authorized.begin(), authorized.end(), live.distributed) == authorized.end())
          authorized.push_back(live.distributed);
        return authorized;
      },
      "AMR continuation closure authority enumeration");
  const auto primary_realm = store.domain_realm() ? store.domain_realm() : live.finite.realm();
  continuation_synchronize_realm_(primary_realm);
  auto closure = continuation_collective_closure<Dim>(
      value, live.distributed, prepared_execution_lane(), store.configuration().byte_capacity,
      primary_realm, authorized);
  return continuation_local_phase(
      [&] {
        return store.prepare(std::move(value), level_continuation_prototypes_(), initialize,
                             live.distributed, std::move(closure));
      },
      "AMR child continuation full image preparation");
}
void level_continuation_image_consensus_(
    const level_continuation_store_type& store,
    const typename level_continuation_store_type::Prepared& prepared) const {
  auto identity = continuation_local_phase(
      [&] {
        return store
            .codec(level_continuation_prototypes_(), nullptr, prepared.distributed_registry.get(),
                   prepared.closure.get())
            .collective_identity(prepared.value, prepared.publication);
      },
      "AMR child continuation image identity");
  const std::array<std::pair<std::string_view, std::string_view>, 1> identities{
      {{"amr-child-continuation", identity}}};
  if (!all_ranks_agree_exact_ordered_byte_pairs(identities, prepared_execution_lane()))
    throw std::logic_error("AMR child continuation image differs across execution ranks");
}
LevelContinuationRetention level_continuation_collect_local_retention_(
    const level_continuation_store_type& base,
    const typename level_continuation_store_type::Prepared* next) const {
  LevelContinuationRetention keep;
  keep.realm = next ? next->realm : base.domain_realm();
  keep.distributed = next ? next->distributed_registry : base.distributed_registry();
  if (base.distributed_registry() && keep.distributed != base.distributed_registry())
    throw std::logic_error("AMR child candidate changed its distributed issuer");
  const auto collect = [&](const level_continuation_ref& root) {
    continuation_detail::walk_value<Dim>(
        root, base.configuration().byte_capacity,
        [&](const ContinuationValue<Dim>& value) {
          const auto& domain = value.domain_ref();
          if (domain) {
            auto& list = domain->realm()->replicated() ? keep.replicated : keep.local;
            if (std::find(list.begin(), list.end(), domain) == list.end())
              list.push_back(domain);
          }
          return true;
        },
        [](const ContinuationValue<Dim>&) {});
  };
  if (base.initialized())
    collect(base.read());
  const auto collect_closure = [&](const auto& closure) {
    if (!closure)
      return;
    for (const auto& registry : closure->registries)
      if (std::find(keep.protected_distributed.begin(), keep.protected_distributed.end(),
                    registry) == keep.protected_distributed.end())
        keep.protected_distributed.push_back(registry);
    for (const auto& array : closure->arrays)
      collect(ContinuationValue<Dim>::distributed_array(array));
  };
  collect_closure(base.closure());
  if (next) {
    collect(next->value);
    collect_closure(next->closure);
  }
  if (keep.distributed)
    keep.protected_distributed.push_back(keep.distributed);
  return keep;
}
static void level_continuation_merge_retention_(LevelContinuationRetention& into,
                                                const LevelContinuationRetention& source) {
  const auto merge = [](auto& out, const auto& input) {
    for (const auto& item : input)
      if (std::find(out.begin(), out.end(), item) == out.end())
        out.push_back(item);
  };
  merge(into.replicated, source.replicated);
  merge(into.local, source.local);
  merge(into.protected_distributed, source.protected_distributed);
}
LevelContinuationRetention level_continuation_collect_retention_(
    const level_continuation_store_type& base,
    const typename level_continuation_store_type::Prepared* next) const {
  auto keep = level_continuation_collect_local_retention_(base, next);
  const auto collect = [&](const auto& forest) {
    for (const auto& [level, store] : forest) {
      (void)level;
      if (store.initialized())
        level_continuation_merge_retention_(
            keep, level_continuation_collect_local_retention_(store, nullptr));
    }
  };
  collect(amr_continuation_accepted_);
  collect(level_continuation_macro_bases_);
  collect(level_continuation_working_);
  for (const auto& [level, pending] : level_continuation_pending_)
    level_continuation_merge_retention_(
        keep, level_continuation_collect_local_retention_(continuation_regrid_pending_store_(level),
                                                          &pending.image));
  return keep;
}
void level_continuation_adopt_retention_(const LevelContinuationRetention& keep) const {
  continuation_local_phase(
      [&] {
        auto& live = finite_values_();
        if (keep.realm)
          live.finite.select_realm(keep.realm);
        live.finite.adopt_domains(keep.replicated);
        live.local_finite.adopt_domains(keep.local);
      },
      "AMR child retained domain adoption");
}
void level_continuation_install_retention_(LevelContinuationRetention&& keep) const noexcept {
  auto& live = *active_finite_invocation_->lifetime_;
  live.retained_realm.swap(keep.realm);
  live.retained_domains.swap(keep.replicated);
  live.retained_local_domains.swap(keep.local);
  live.retained_distributed.swap(keep.distributed);
  live.protected_distributed.swap(keep.protected_distributed);
}
