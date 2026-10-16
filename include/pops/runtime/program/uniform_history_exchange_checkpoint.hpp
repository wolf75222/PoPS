#pragma once
#include <pops/runtime/program/history_sample_identity_codec.hpp>

namespace pops::runtime::program {
template <int Dim>
struct UniformHistoryExchangeImage {
  std::span<const std::uint8_t> exchanges;
  std::map<std::string, std::vector<FieldExchangeExpressions::Ref>> histories;
  std::map<std::string, std::vector<HistorySampleIdentity>> samples;
  HistoryFieldMetadataTable fields;
};

inline std::vector<std::uint8_t> encode_uniform_history_exchanges(
    std::span<const std::uint8_t> exchanges,
    const std::map<std::string, std::vector<FieldExchangeExpressions::Ref>>& histories,
    const std::map<std::string, std::vector<HistorySampleIdentity>>& samples,
    const FieldExchangeExpressions::Limits& limits) {
  std::size_t capacity = exchanges.size();
  field_exchange_detail::charge(capacity, limits.max_bytes,
                                std::numeric_limits<std::size_t>::max());
  std::vector<FieldExchangeExpressions::Ref> roots;
  for (const auto& [name, metadata] : histories) {
    if (metadata.size() != samples.at(name).size())
      throw std::invalid_argument("history metadata differs from sample depth");
    roots.insert(roots.end(), metadata.begin(), metadata.end());
  }
  const auto table = FieldExchangeExpressions::encode_roots(roots, limits);
  field_exchange_detail::Writer out{{}, capacity};
  out.word(UINT64_C(0x3530584553504f50));  // POPSEX05
  out.blob(exchanges);
  out.word(histories.size());
  for (const auto& [name, metadata] : histories) {
    out.text(name);
    out.blob(encode_history_sample_identity(name, -1, samples.at(name)));
    out.word(metadata.size());
  }
  out.blob(table);
  return std::move(out.bytes);
}

inline std::vector<std::uint8_t> encode_uniform_history_fields(
    std::span<const std::uint8_t> exchanges, const HistoryFieldMetadataTable& fields,
    const std::map<std::string, std::vector<HistorySampleIdentity>>& samples,
    const HistoryFieldMetadataBudget& budget) {
  history_field_detail::Tables tables(budget.workspace_bytes);
  tables.collect(fields, budget);
  std::uint64_t total = 32, largest_sample = 0;
  history_field_detail::charge(total, exchanges.size(), 1, budget.workspace_bytes);
  history_field_detail::charge(total, tables.wire, 1, budget.workspace_bytes);
  for (const auto& [name, slots] : fields) {
    if (slots.size() != samples.at(name).size())
      throw std::invalid_argument("history Field metadata differs from exact sample depth");
    std::uint64_t sample_size = 32;
    history_field_detail::charge(sample_size, name.size(), 1, budget.workspace_bytes);
    history_field_detail::charge(sample_size, slots.size(), 64, budget.workspace_bytes);
    largest_sample = std::max(largest_sample, sample_size);
    history_field_detail::charge(total, name.size(), 1, budget.workspace_bytes);
    history_field_detail::charge(total, 16 + sample_size, 1, budget.workspace_bytes);
  }
  // The output, metadata table and one exact sample blob coexist. Retained
  // descriptors/payloads are accounted independently by tables.collect().
  auto peak = total;
  history_field_detail::charge(peak, tables.wire + tables.workspace.used(), 1,
                               budget.workspace_bytes);
  history_field_detail::charge(peak, largest_sample, 1, budget.workspace_bytes);
  const auto metadata = encode_prepared_history_field_metadata(fields, tables);
  field_exchange_detail::Writer out{{}, static_cast<std::size_t>(total)};
  out.bytes.reserve(static_cast<std::size_t>(total));
  out.word(UINT64_C(0x3630584553504f50));  // POPSEX06
  out.blob(exchanges);
  out.word(fields.size());
  for (const auto& [name, slots] : fields) {
    out.text(name);
    out.blob(encode_history_sample_identity(name, -1, samples.at(name)));
  }
  out.blob(metadata);
  if (out.bytes.size() != total)
    throw std::logic_error("history Field checkpoint size preflight differs");
  return std::move(out.bytes);
}

template <int Dim>
std::vector<std::uint8_t> checkpoint_uniform_history_exchanges(
    std::span<const std::uint8_t> exchanges, const HistoryManager<Dim>& manager) {
  std::vector<std::uint8_t> inner;
  if (manager.exchange_limits.max_bases && !manager.histories.empty()) {
    for (const auto& [name, ring] : manager.histories) {
      if (manager.slot_exchanges.at(name).size() != ring.size())
        throw std::invalid_argument("history face provenance differs from ring depth");
      (void)manager.validated_samples(name);
    }
    inner = encode_uniform_history_exchanges(exchanges, manager.slot_exchanges, manager.slot_sample,
                                             manager.exchange_limits);
    exchanges = inner;
  }
  if (manager.field_metadata_budget.enabled() && !manager.histories.empty())
    return encode_uniform_history_fields(exchanges, manager.slot_fields, manager.slot_sample,
                                         manager.field_metadata_budget);
  return {exchanges.begin(), exchanges.end()};
}

template <int Dim>
UniformHistoryExchangeImage<Dim> read_uniform_history_exchanges(
    std::span<const std::uint8_t> bytes, const HistoryManager<Dim>& manager,
    bool require_current_samples = false, std::span<const std::uint8_t> validated_field_wire = {},
    std::uint64_t reserved_workspace = 0) {
  UniformHistoryExchangeImage<Dim> result;
  result.exchanges = bytes;
  const bool fields_required =
      manager.field_metadata_budget.enabled() && !manager.histories.empty();
  field_exchange_detail::Reader envelope{bytes};
  const bool has_fields = bytes.size() >= 8 && envelope.word() == UINT64_C(0x3630584553504f50);
  if (has_fields != fields_required)
    throw std::invalid_argument("history checkpoint retained Field metadata declaration differs");
  if (has_fields) {
    std::uint64_t field_workspace = reserved_workspace;
    history_field_detail::charge(field_workspace, bytes.size(), 1,
                                 manager.field_metadata_budget.workspace_bytes);
    for (const auto& [name, ring] : manager.histories) {
      history_field_detail::charge(field_workspace, ring.size(), sizeof(HistorySampleIdentity),
                                   manager.field_metadata_budget.workspace_bytes);
      history_field_detail::charge(field_workspace, 1,
                                   sizeof(typename decltype(result.samples)::value_type),
                                   manager.field_metadata_budget.workspace_bytes);
      history_field_detail::charge(field_workspace, name.size(), 1,
                                   manager.field_metadata_budget.workspace_bytes);
    }
    result.exchanges = envelope.blob();
    if (envelope.count(manager.histories.size(), 16) != manager.histories.size())
      throw std::invalid_argument("history Field checkpoint omits a ring");
    for (const auto& [name, ring] : manager.histories) {
      if (envelope.text() != name)
        throw std::invalid_argument("history Field checkpoint changes ring identity/order");
      auto samples = decode_history_sample_identity(envelope.blob(), name, -1, ring.size());
      if (require_current_samples && samples != manager.validated_samples(name))
        throw std::invalid_argument("history Field checkpoint changes restored exact samples");
      result.samples.emplace(name, std::move(samples));
    }
    const auto metadata_bytes = envelope.blob();
    auto metadata_budget = manager.field_metadata_budget;
    metadata_budget.workspace_bytes -= field_workspace - metadata_bytes.size();
    if (validated_field_wire.empty())
      result.fields =
          decode_history_field_metadata(metadata_bytes, manager.histories, metadata_budget);
    else if (metadata_bytes.size() != validated_field_wire.size() ||
             !std::equal(metadata_bytes.begin(), metadata_bytes.end(),
                         validated_field_wire.begin()))
      throw std::invalid_argument(
          "history Field migration changes exact support or representation");
    if (envelope.at != bytes.size())
      throw std::invalid_argument("history Field checkpoint has trailing bytes");
    bytes = result.exchanges;
  }
  const bool required = manager.exchange_limits.max_bases && !manager.histories.empty();
  field_exchange_detail::Reader in{bytes};
  const bool extended = bytes.size() >= 8 && in.word() == UINT64_C(0x3530584553504f50);
  if (!extended) {
    if (required)
      throw std::invalid_argument("history checkpoint omits required retained face provenance");
    return result;
  }
  if (!required)
    throw std::invalid_argument("history checkpoint carries undeclared retained face provenance");
  result.exchanges = in.blob();
  const auto count = in.count(manager.histories.size(), 24);
  if (count != manager.histories.size())
    throw std::invalid_argument("history exchange checkpoint omits a ring");
  std::vector<std::pair<std::string, std::size_t>> rows;
  rows.reserve(count);
  for (const auto& [expected, ring] : manager.histories) {
    auto name = in.text();
    if (name != expected)
      throw std::invalid_argument("history exchange checkpoint changes ring identity/order");
    auto samples = decode_history_sample_identity(in.blob(), name, -1, ring.size());
    if (require_current_samples && samples != manager.validated_samples(name))
      throw std::invalid_argument(
          "history exchange checkpoint does not describe the restored exact samples");
    const auto depth = in.word();
    if (depth != ring.size())
      throw std::invalid_argument("history exchange checkpoint changes ring depth");
    if (has_fields && result.samples.at(name) != samples)
      throw std::invalid_argument("history Field and face checkpoint samples differ");
    result.samples.insert_or_assign(name, std::move(samples));
    rows.emplace_back(std::move(name), depth);
  }
  auto roots = FieldExchangeExpressions::decode_roots(in.blob(), manager.exchange_limits);
  std::size_t used = 0;
  for (const auto& [name, depth] : rows) {
    if (depth > roots.size() - used)
      throw std::invalid_argument("history exchange checkpoint omits metadata slots");
    result.histories.emplace(name, std::vector<FieldExchangeExpressions::Ref>(
                                       roots.begin() + used, roots.begin() + used + depth));
    used += depth;
  }
  if (used != roots.size() || in.at != bytes.size())
    throw std::invalid_argument("history exchange checkpoint has unbound metadata");
  return result;
}
/// Pure preparation: no live History or accepted mailbox is modified here. Field-associated
/// sources follow the target cell owners; the mailbox is repartitioned as an incidence set.
template <int Dim, class OwnsCell>
std::vector<std::uint8_t> rematerialize_uniform_history_exchanges(
    std::span<const std::span<const std::uint8_t>> images, const HistoryManager<Dim>& manager,
    OwnsCell owns_cell, int rank, int size) {
  if (images.empty() || size <= 0 || rank < 0 || rank >= size)
    throw std::invalid_argument("history exchange migration has invalid source/target ranks");
  const bool histories = manager.exchange_limits.max_bases && !manager.histories.empty();
  const bool fields = manager.field_metadata_budget.enabled() && !manager.histories.empty();
  std::size_t total = 0;
  // This preflight only views the wire. In particular no rank-local ledger is decoded until
  // all sources have the supported owner-free mailbox shape and fit the installed contract.
  for (const auto bytes : images) {
    field_exchange_detail::charge(total, bytes.size(), std::numeric_limits<std::size_t>::max());
    field_exchange_detail::Reader outer{bytes};
    auto inner = bytes;
    auto magic_outer = bytes.size() >= 8 ? outer.word() : 0;
    if (magic_outer == UINT64_C(0x3630584553504f50)) {
      inner = outer.blob();
      outer = field_exchange_detail::Reader{inner};
      magic_outer = inner.size() >= 8 ? outer.word() : 0;
    }
    if (magic_outer == UINT64_C(0x3530584553504f50))
      inner = outer.blob();
    field_exchange_detail::Reader ledger{inner};
    const auto magic = ledger.word();
    if (magic != UINT64_C(0x3130584553504f50) && magic != UINT64_C(0x3230584553504f50))
      throw std::invalid_argument(
          "exchange migration requires an explicit remap for retained moving geometry");
    (void)ledger.count(std::numeric_limits<std::size_t>::max(), 72);
  }
  if (histories && total > manager.exchange_limits.max_bytes)
    throw std::length_error(
        "history exchange migration source images exceed installed byte capacity");
  if (fields && total > manager.field_metadata_budget.workspace_bytes)
    throw std::length_error(
        "history Field migration source images exceed explicit workspace capacity");
  std::uint64_t field_workspace = total;
  if (fields) {
    history_field_detail::charge(field_workspace, images.size(),
                                 sizeof(UniformHistoryExchangeImage<Dim>),
                                 manager.field_metadata_budget.workspace_bytes);
    for (const auto& [name, ring] : manager.histories) {
      std::uint64_t one_rank = 0;
      history_field_detail::charge(one_rank, ring.size(), sizeof(HistorySampleIdentity),
                                   manager.field_metadata_budget.workspace_bytes);
      history_field_detail::charge(
          one_rank, 1,
          sizeof(typename decltype(UniformHistoryExchangeImage<Dim>::samples)::value_type),
          manager.field_metadata_budget.workspace_bytes);
      history_field_detail::charge(one_rank, name.size(), 1,
                                   manager.field_metadata_budget.workspace_bytes);
      history_field_detail::charge(field_workspace, images.size(), one_rank,
                                   manager.field_metadata_budget.workspace_bytes);
    }
  }
  std::span<const std::uint8_t> field_wire;
  if (fields) {
    field_exchange_detail::Reader first{images.front()};
    if (first.word() != UINT64_C(0x3630584553504f50))
      throw std::invalid_argument("history Field migration omits its declared metadata");
    (void)first.blob();
    if (first.count(manager.histories.size(), 16) != manager.histories.size())
      throw std::invalid_argument("history Field migration changes its ring set");
    for (const auto& [name, ring] : manager.histories) {
      (void)first.text();
      (void)first.blob();
    }
    field_wire = first.blob();
  }
  std::vector<UniformHistoryExchangeImage<Dim>> sources;
  sources.reserve(images.size());
  std::vector<AcceptedExchangeLedger> mailboxes;
  mailboxes.reserve(images.size());
  for (const auto bytes : images) {
    sources.push_back(read_uniform_history_exchanges(
        bytes, manager, false, sources.empty() ? std::span<const std::uint8_t>{} : field_wire,
        fields ? field_workspace - bytes.size() : 0));
    mailboxes.push_back(AcceptedExchangeLedger::from_checkpoint(sources.back().exchanges));
  }
  const auto mailbox =
      AcceptedExchangeLedger::rematerialize(mailboxes, rank, size).checkpoint(true);
  auto output_field_budget = manager.field_metadata_budget;
  if (fields) {
    output_field_budget.workspace_bytes -= field_workspace;
    for (const auto& source : sources)
      if (source.samples != sources.front().samples)
        throw std::invalid_argument("history Field migration changes exact retained samples");
  }
  if (!histories)
    return fields ? encode_uniform_history_fields(mailbox, sources.front().fields,
                                                  sources.front().samples, output_field_budget)
                  : mailbox;
  std::map<std::string, std::vector<FieldExchangeExpressions::Ref>> target;
  FieldExchangeExpressions::RematerializationCache cache;
  std::map<std::vector<const FieldExchangeExpressions::Expression*>, FieldExchangeExpressions::Ref>
      expressions;
  for (const auto& [name, ring] : manager.histories) {
    const auto& samples = sources.front().samples.at(name);
    for (const auto& source : sources)
      if (source.samples.at(name) != samples)
        throw std::invalid_argument("history exchange migration changes exact sample authority");
    auto& rows = target[name];
    rows.resize(ring.size());
    for (std::size_t slot = 0; slot < ring.size(); ++slot) {
      std::vector<FieldExchangeExpressions::Ref> shards;
      shards.reserve(sources.size());
      std::vector<const FieldExchangeExpressions::Expression*> source_aliases;
      source_aliases.reserve(sources.size());
      for (const auto& source : sources) {
        shards.push_back(source.histories.at(name).at(slot));
        source_aliases.push_back(shards.back().get());
      }
      if (const auto previous = expressions.find(source_aliases); previous != expressions.end())
        rows[slot] = previous->second;
      else {
        rows[slot] = FieldExchangeExpressions::rematerialize(shards, manager.exchange_limits,
                                                             owns_cell, &cache);
        expressions.emplace(std::move(source_aliases), rows[slot]);
      }
    }
  }
  auto inner = encode_uniform_history_exchanges(mailbox, target, sources.front().samples,
                                                manager.exchange_limits);
  return fields ? encode_uniform_history_fields(inner, sources.front().fields,
                                                sources.front().samples, output_field_budget)
                : inner;
}

}  // namespace pops::runtime::program
