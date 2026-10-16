// The deferred lag owns its numerical cache in the context. Metadata authority
// follows the exact MultiFab and its retained Fab allocations, including snapshot
// copies; neither shape nor a borrowed reference manufactures an owner.
using DeferredHistoryLagAuthorities =
    std::map<std::string, ContinuationFieldStorageAuthority, std::less<>>;

template <class Scratches>
Scratches copy_deferred_history_scratches_(const Scratches& source,
                                           const DeferredHistoryLagAuthorities& authorities) const {
  if (source.size() != authorities.size())
    throw std::logic_error("deferred History cache omits its storage authorities");
  for (const auto& [key, field] : source)
    if (!authorities.at(key).allocations.matches(field))
      throw std::invalid_argument("deferred History cache allocation was reincarnated");
  return source;
}

template <class Scratches>
DeferredHistoryLagAuthorities prepare_deferred_history_authorities_(
    const Scratches& copied, const DeferredHistoryLagAuthorities& original) const {
  if (copied.size() != original.size())
    throw std::logic_error("deferred History snapshot omits its authorities");
  auto result = original;
  for (const auto& [key, field] : copied) {
    auto& authority = result.at(key);
    authority.lifetime = field.storage_identity();
    authority.allocations = continuation_field_allocations_(field);
  }
  return result;
}

std::optional<ContinuationFieldStorageAuthority> deferred_history_field_authority_(
    const field_type& field) const {
  for (const auto& [key, candidate] : deferred_history_lag_scratches_) {
    if (&field != &candidate)
      continue;
    const auto found = deferred_history_lag_authorities_.find(key);
    if (found == deferred_history_lag_authorities_.end() || found->second.lifetime.expired() ||
        !found->second.allocations.matches(field) || found->second.revoked)
      throw std::invalid_argument("deferred History Field has no exact live cache authority");
    const auto& authority = found->second;
    if (authority.support) {
      authority.support->validate();
      if (authority.support->program_identity != facade_->installed_program_hash())
        throw std::invalid_argument("deferred History Field belongs to another Program");
    }
    return authority;
  }
  return std::nullopt;
}

// Append to the same pointer-deduplicating census as a pending History store.
// A live deferred cache remains retained when another ring is replaced.
void append_current_deferred_history_metadata_budget_(
    history_field_detail::Tables& tables, const HistoryFieldMetadataBudget& budget) const {
  for (const auto& [key, authority] : deferred_history_lag_authorities_) {
    const std::array<std::pair<std::string_view, std::array<HistoryFieldMetadata, 1>>, 1> view{
        std::pair{std::string_view(key),
                  std::array{HistoryFieldMetadata{authority.support, authority.representation}}}};
    tables.collect_selected(view, budget, [](const auto&, const auto& slots) {
      return std::span<const HistoryFieldMetadata>(slots);
    });
    history_field_detail::charge(tables.retained,
                                 sizeof(typename DeferredHistoryLagAuthorities::value_type), 1,
                                 budget.retained_bytes);
    history_field_detail::charge(tables.retained, authority.allocations.local.size(),
                                 sizeof(typename field_type::fab_type::storage_type::const_type),
                                 budget.retained_bytes);
  }
}

void require_deferred_history_metadata_budget_(const std::string& replacement_key,
                                               const HistoryFieldMetadata& replacement,
                                               const field_type& prototype) const {
  const auto& manager = runtime_state().hist_;
  const auto& budget = manager.field_metadata_budget;
  budget.validate();
  if (!budget.enabled()) {
    if (!replacement.empty())
      throw std::invalid_argument("deferred History Field metadata has no installed budget");
    for (const auto& [key, authority] : deferred_history_lag_authorities_)
      if (authority.support || authority.representation)
        throw std::invalid_argument("retained deferred History metadata has no installed budget");
    return;
  }
  history_field_detail::Tables tables(budget.workspace_bytes);
  tables.collect(manager.slot_fields, budget);
  std::uint64_t prepared_workspace = 0;
  const auto charge_cache = [&](const std::string& key, const HistoryFieldMetadata& metadata,
                                std::size_t local_allocations) {
    // This is a separate native cache namespace, never serialized as a History
    // ring. Borrow its key/one-slot view and share the exact pointer-dedup tables.
    const std::array<std::pair<std::string_view, std::array<HistoryFieldMetadata, 1>>, 1> view{
        std::pair{std::string_view(key), std::array{metadata}}};
    tables.collect_selected(view, budget, [](const auto&, const auto& slots) {
      return std::span<const HistoryFieldMetadata>(slots);
    });
    constexpr auto descriptor = sizeof(typename DeferredHistoryLagAuthorities::value_type);
    history_field_detail::charge(tables.retained, descriptor, 1, budget.retained_bytes);
    history_field_detail::charge(tables.retained, local_allocations,
                                 sizeof(typename field_type::fab_type::storage_type::const_type),
                                 budget.retained_bytes);
    // The replacement authority map is prepared before the live map is swapped.
    history_field_detail::charge(prepared_workspace, descriptor, 1, budget.workspace_bytes);
    history_field_detail::charge(prepared_workspace, key.size(), 1, budget.workspace_bytes);
    history_field_detail::charge(prepared_workspace, local_allocations,
                                 sizeof(typename field_type::fab_type::storage_type::const_type),
                                 budget.workspace_bytes);
  };
  for (const auto& [key, authority] : deferred_history_lag_authorities_)
    if (key != replacement_key)
      charge_cache(key, {authority.support, authority.representation},
                   authority.allocations.local.size());
  charge_cache(replacement_key, replacement, prototype.local_size());
  history_field_detail::charge(prepared_workspace, tables.workspace.used(), 1,
                               budget.workspace_bytes);
  history_field_detail::charge(prepared_workspace, tables.wire, 1, budget.workspace_bytes);
}
