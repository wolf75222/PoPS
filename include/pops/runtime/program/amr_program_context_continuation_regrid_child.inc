// Included after the regrid frame. A scientific Base is not a target allocator.
private:
level_continuation_store_type& continuation_regrid_target_store_() const {
  auto& frame = require_continuation_regrid_();
  if (!frame.target_ready || frame.published)
    throw std::logic_error("regrid target storage has no unpublished transaction");
  if (continuation_parent_phase_)
    return continuation_parent_working_;
  const auto target = frame.targets.find(active_level_);
  if (active_level_ < 0 || target == frame.targets.end() || !target->second.configured())
    throw std::logic_error("regrid target has no installed complete Continuation schema");
  return target->second;
}

const level_continuation_store_type& continuation_regrid_pending_store_(int level) const {
  if (continuation_regrid_phase())
    return require_continuation_regrid_().targets.at(level);
  return level_continuation_macro_bases_.at(level);
}

void continuation_regrid_capture_source_(const level_continuation_store_type& source) const {
  auto& live = finite_values_();
  const auto bind = [&](const std::shared_ptr<distributed::Registry>& registry) {
    if (!registry)
      return;
    if (std::find(live.protected_distributed.begin(), live.protected_distributed.end(), registry) ==
        live.protected_distributed.end())
      live.protected_distributed.push_back(registry);
    live.transport.bind_distributed_registry(
        *registry, [registry](distributed::Identity version, distributed::DomainRef domain,
                              const ContinuationType& element) {
          registry->authenticate(domain);
          auto array = std::static_pointer_cast<const ContinuationDistributedArray<Dim>>(
              registry->resolve_value(version, distributed::array_type_tag<Dim>()));
          if (!array || array->domain != domain || array->element != element ||
              array->version.issuer != version.issuer || array->version.ordinal != version.ordinal)
            throw std::invalid_argument("regrid source Array lost its exact issuer/version");
          return array;
        });
  };
  bind(source.distributed_registry());
  if (source.closure())
    for (const auto& registry : source.closure()->registries)
      bind(registry);
  live.transport.capture(source.read());
  if (source.closure())
    for (const auto& array : source.closure()->arrays)
      live.transport.capture(ContinuationValue<Dim>::distributed_array(array));
}

public:
// Called immediately after finite_value_scope(), before any authored equation.
// A source read below never changes this target's allocator or its burned ordinals.
void continuation_regrid_bind_target_() const {
  auto* target = continuation_local_phase([&] { return &continuation_regrid_target_store_(); },
                                          "regrid target allocator selection");
  auto& live = finite_values_();
  const auto realm = target->domain_realm();
  continuation_word_agrees_(realm ? 1 : 0, "regrid target allocator differs across ranks");
  if (realm)
    continuation_synchronize_realm_(realm);
  continuation_local_phase(
      [&] {
        live.finite.import_high_water(target->domain_high_water());
        if (realm)
          live.finite.select_realm(realm);
        if (target->distributed_registry()) {
          if (live.distributed && live.distributed != target->distributed_registry())
            throw std::logic_error(
                "regrid target has already used a different distributed allocator");
          live.distributed = target->distributed_registry();
        }
        if (target->initialized())
          continuation_regrid_capture_source_(*target);
      },
      "regrid target allocator binding");
}

level_continuation_ref continuation_regrid_source(std::int64_t level) const {
  continuation_word_agrees_(std::bit_cast<std::uint64_t>(level),
                            "regrid source level differs across ranks");
  return continuation_local_phase(
      [&] {
        const auto& frame = require_continuation_regrid_();
        if (!frame.target_ready || frame.published || level < 0 ||
            level >= static_cast<std::int64_t>(frame.sources.size()) - 1)
          throw std::out_of_range("regrid source level is outside the immutable accepted forest");
        const auto& source = frame.sources.at(static_cast<int>(level));
        level_continuation_require_frontier_(source, source.read(), frame.time, frame.step);
        continuation_regrid_capture_source_(source);
        return source.read();
      },
      "immutable regrid source image read");
}

typename continuation_authority_type::Base continuation_regrid_base(std::int64_t source) const {
  continuation_word_agrees_(std::bit_cast<std::uint64_t>(source),
                            "regrid scientific source Base differs across ranks");
  return continuation_local_phase(
      [&] {
        auto& frame = require_continuation_regrid_();
        if (!frame.target_ready || frame.published || continuation_parent_phase_ || source < 0 ||
            source >= static_cast<std::int64_t>(frame.sources.size()) - 1)
          throw std::invalid_argument("regrid child must choose one exact immutable source Base");
        const auto& selected = frame.sources.at(static_cast<int>(source));
        const auto& target = continuation_regrid_target_store_();
        if (selected.configuration() != target.configuration())
          throw std::invalid_argument("regrid Base and target have different scientific schemas");
        level_continuation_require_frontier_(selected, selected.read(), frame.time, frame.step);
        const auto prior = frame.selected_sources.find(active_level_);
        if (prior != frame.selected_sources.end() && prior->second != source)
          throw std::logic_error("one regrid target cannot substitute its scientific source Base");
        auto& authority = level_continuation_authorities_[active_level_];
        if (prior == frame.selected_sources.end()) {
          if (!authority)
            authority = std::make_shared<level_continuation_authority_type>();
          authority->begin(frame.attempt);
          level_continuation_macro_bases_.insert_or_assign(active_level_, selected);
          frame.selected_sources.emplace(active_level_, static_cast<int>(source));
        }
        if (!authority)
          throw std::logic_error("regrid selected Base lost its native authority");
        continuation_regrid_capture_source_(selected);
        // Intentionally bypass level_continuation_read(): it selects the source
        // allocator, which would alias two children when both select one coarse Base.
        return authority->read(level_continuation_macro_bases_.at(active_level_));
      },
      "explicit regrid scientific source Base selection");
}

void prepare_continuation_regrid_child(
    const typename continuation_authority_type::Accepted& accepted) const {
  const auto selected = continuation_local_phase(
      [&] {
        const auto& frame = require_continuation_regrid_();
        if (!frame.target_ready || frame.published || continuation_parent_phase_ ||
            !frame.selected_sources.contains(active_level_) ||
            level_continuation_pending_.contains(active_level_))
          throw std::logic_error("regrid child requires one unconsumed exact source Base");
        return active_level_;
      },
      "regrid child publication preflight");
  auto& source = level_continuation_macro_bases_.at(selected);
  auto& target = continuation_regrid_target_store_();
  auto& authority = level_continuation_authority();
  level_continuation_selected_consensus_(authority, accepted);
  auto value = continuation_local_phase(
      [&] {
        const auto& frame = require_continuation_regrid_();
        auto out = authority.require_root_publication(accepted, source);
        // Regridding changes representation only, never physical time or the accepted step.
        level_continuation_require_frontier_(target, out, frame.time, frame.step);
        return out;
      },
      "regrid child complete assessed candidate");
  auto prepared = level_continuation_prepare_(target, std::move(value), !target.initialized());
  auto keep = continuation_local_phase(
      [&] { return level_continuation_collect_retention_(target, &prepared); },
      "regrid child complete retained closure");
  level_continuation_image_consensus_(target, prepared);
  level_continuation_adopt_retention_(keep);
  continuation_local_phase(
      [&] {
        if (!level_continuation_pending_
                 .emplace(selected,
                          LevelContinuationPending{std::move(prepared), accepted, accepted.value(),
                                                   accepted.value()->members()[2]})
                 .second)
          throw std::logic_error("regrid child was staged twice");
      },
      "regrid child staging for global parent assessment");
  level_continuation_install_retention_(std::move(keep));
}
