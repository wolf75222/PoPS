#pragma once
#include <pops/runtime/program/history_sample_identity_codec.hpp>

namespace pops::runtime::program {
template <int Dim>
struct UniformHistoryExchangeImage {
  std::span<const std::uint8_t> exchanges;
  std::map<std::string, std::vector<FieldExchangeExpressions::Ref>> histories;
  std::map<std::string, std::vector<HistorySampleIdentity>> samples;
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

template <int Dim>
std::vector<std::uint8_t> checkpoint_uniform_history_exchanges(
    std::span<const std::uint8_t> exchanges, const HistoryManager<Dim>& manager) {
  if (!manager.exchange_limits.max_bases || manager.histories.empty())
    return {exchanges.begin(), exchanges.end()};
  for (const auto& [name, ring] : manager.histories) {
    if (manager.slot_exchanges.at(name).size() != ring.size())
      throw std::invalid_argument("history face provenance differs from ring depth");
    (void)manager.validated_samples(name);
  }
  return encode_uniform_history_exchanges(exchanges, manager.slot_exchanges, manager.slot_sample,
                                          manager.exchange_limits);
}

template <int Dim>
UniformHistoryExchangeImage<Dim> read_uniform_history_exchanges(
    std::span<const std::uint8_t> bytes, const HistoryManager<Dim>& manager,
    bool require_current_samples = false) {
  UniformHistoryExchangeImage<Dim> result;
  result.exchanges = bytes;
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
    result.samples.emplace(name, std::move(samples));
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
  std::size_t total = 0;
  // This preflight only views the wire. In particular no rank-local ledger is decoded until
  // all sources have the supported owner-free mailbox shape and fit the installed contract.
  for (const auto bytes : images) {
    field_exchange_detail::charge(total, bytes.size(), std::numeric_limits<std::size_t>::max());
    field_exchange_detail::Reader outer{bytes};
    auto inner = bytes;
    if (bytes.size() >= 8 && outer.word() == UINT64_C(0x3530584553504f50))
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
  std::vector<UniformHistoryExchangeImage<Dim>> sources;
  sources.reserve(images.size());
  std::vector<AcceptedExchangeLedger> mailboxes;
  mailboxes.reserve(images.size());
  for (const auto bytes : images) {
    sources.push_back(read_uniform_history_exchanges(bytes, manager));
    mailboxes.push_back(AcceptedExchangeLedger::from_checkpoint(sources.back().exchanges));
  }
  const auto mailbox =
      AcceptedExchangeLedger::rematerialize(mailboxes, rank, size).checkpoint(true);
  if (!histories)
    return mailbox;
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
  return encode_uniform_history_exchanges(mailbox, target, sources.front().samples,
                                          manager.exchange_limits);
}

}  // namespace pops::runtime::program
