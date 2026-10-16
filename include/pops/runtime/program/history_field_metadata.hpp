#pragma once

#include <pops/core/identity/prepared_provider.hpp>
#include <pops/core/identity/sha256.hpp>
#include <pops/runtime/program/continuation_field_support.hpp>
#include <pops/runtime/program/field_representation_obligation.hpp>
#include <pops/runtime/program/structural_traversal.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <limits>
#include <map>
#include <memory_resource>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pops::runtime::program {

// History owns these descriptions together with its numerical slots. Neither
// description grants permission to read another allocation or change support.
struct HistoryFieldMetadata {
  ContinuationFieldSupport::Ref support;
  FieldRepresentationObligation::Ref representation;
  bool empty() const noexcept { return !support && !representation; }
  template <class Builder>
  void append_contract(Builder& contract) const {
    contract.presence(support != nullptr).presence(representation != nullptr);
    if (support) {
      contract.text(support->program_identity)
          .text(support->origin_point)
          .scalar(support->program_owner)
          .scalar(support->producer_level)
          .scalar(support->topology_epoch)
          .scalar(support->materialization_generation)
          .text(support->field_provider);
    }
    if (representation)
      contract.text(representation->contract).text(representation->origin);
  }
};
using HistoryFieldMetadataTable = std::map<std::string, std::vector<HistoryFieldMetadata>>;

struct HistoryFieldMetadataBudget {
  std::uint64_t retained_bytes = 0;
  std::uint64_t workspace_bytes = 0;
  bool operator==(const HistoryFieldMetadataBudget&) const = default;
  bool enabled() const noexcept { return retained_bytes != 0; }
  void validate() const {
    if ((retained_bytes == 0) != (workspace_bytes == 0))
      throw std::invalid_argument("history metadata requires paired storage/workspace capacities");
  }
};

inline void require_history_field_program(const HistoryFieldMetadataTable& fields,
                                          std::string_view installed_program) {
  for (const auto& [name, slots] : fields)
    for (const auto& slot : slots) {
      if (slot.support) {
        slot.support->validate();
        if (installed_program.empty() || slot.support->program_identity != installed_program)
          throw std::invalid_argument("history Field metadata belongs to a foreign Program");
      }
      if (slot.representation) {
        slot.representation->validate();
        if (!slot.support)
          throw std::invalid_argument("history Field representation lacks exact producer support");
      }
    }
}

namespace history_field_detail {
inline void charge(std::uint64_t& used, std::uint64_t count, std::uint64_t width,
                   std::uint64_t capacity) {
  if (used > capacity || (width && count > (capacity - used) / width))
    throw std::length_error("history Field metadata exceeds its explicit byte capacity");
  used += count * width;
}

// A nonidentity numerical combination is only defined for discharged fields.
// Descriptions are derived from the exact inputs; the caller separately owns
// and authenticates the resulting native allocation.
#include <pops/runtime/program/history_field_metadata_temporal.inc>

struct Reader {
  std::span<const std::uint8_t> bytes;
  std::size_t at = 0;
  std::uint64_t word() {
    if (at > bytes.size() || bytes.size() - at < 8)
      throw std::invalid_argument("truncated history Field metadata");
    std::uint64_t value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
      value |= std::uint64_t(bytes[at++]) << shift;
    return value;
  }
  std::string_view text() {
    const auto size = word();
    if (size > bytes.size() - at)
      throw std::invalid_argument("truncated history Field metadata string");
    const std::string_view result(reinterpret_cast<const char*>(bytes.data() + at), size);
    at += size;
    return result;
  }
  std::size_t count(std::size_t width) {
    const auto value = word();
    if (width == 0 || value > (bytes.size() - at) / width)
      throw std::length_error("history Field metadata count exceeds its bounded image");
    return static_cast<std::size_t>(value);
  }
};

struct Writer {
  std::vector<std::uint8_t> bytes;
  explicit Writer(std::size_t size) { bytes.reserve(size); }
  void word(std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8)
      bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  }
  void text(std::string_view value) {
    word(value.size());
    bytes.insert(bytes.end(), value.begin(), value.end());
  }
};

inline int integer(std::uint64_t word) {
  const auto value = std::bit_cast<std::int64_t>(word);
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    throw std::invalid_argument("history Field support exceeds its native integer representation");
  return static_cast<int>(value);
}

// Scratch maps use the exact bounded allocator, including their buckets/nodes.
// The source tables remain owned by the accepted/provisional History image.
struct Tables {
  continuation_detail::StructuralWorkspaceResource workspace;
  std::pmr::unordered_map<const ContinuationFieldSupport*, std::uint64_t> support_ids;
  std::pmr::unordered_map<const FieldRepresentationObligation*, std::uint64_t> obligation_ids;
  std::pmr::vector<const ContinuationFieldSupport*> supports;
  std::pmr::vector<const FieldRepresentationObligation*> obligations;
  std::uint64_t retained = 0, wire = 24;

  explicit Tables(std::uint64_t capacity)
      : workspace(capacity),
        support_ids(&workspace),
        obligation_ids(&workspace),
        supports(&workspace),
        obligations(&workspace) {}

  void collect(const HistoryFieldMetadataTable& fields, const HistoryFieldMetadataBudget& budget,
               std::string_view replacement_name = {},
               std::span<const HistoryFieldMetadata> replacement_slots = {}) {
    collect_selected(fields, budget, [&](const auto& name, const auto& original) {
      return name == replacement_name ? replacement_slots
                                      : std::span<const HistoryFieldMetadata>(original);
    });
  }

  template <class Fields, class Select>
  void collect_selected(const Fields& fields, const HistoryFieldMetadataBudget& budget,
                        Select select) {
    budget.validate();
    for (const auto& [name, original] : fields) {
      const std::span<const HistoryFieldMetadata> slots = select(name, original);
      if (name.empty())
        throw std::invalid_argument("history Field metadata has an empty ring name");
      charge(retained, 1, sizeof(HistoryFieldMetadataTable::value_type), budget.retained_bytes);
      charge(retained, name.size(), 1, budget.retained_bytes);
      charge(retained, slots.size(), sizeof(HistoryFieldMetadata), budget.retained_bytes);
      charge(wire, name.size(), 1, budget.workspace_bytes);
      charge(wire, 2, 8, budget.workspace_bytes);
      charge(wire, slots.size(), 16, budget.workspace_bytes);
      for (const auto& slot : slots) {
        if (slot.representation && !slot.support)
          throw std::invalid_argument(
              "history Field representation lacks its exact producer support");
        if (slot.support) {
          slot.support->validate();
          if (support_ids.try_emplace(slot.support.get(), supports.size() + 1).second) {
            supports.push_back(slot.support.get());
            charge(retained, 1, sizeof(ContinuationFieldSupport), budget.retained_bytes);
            charge(wire, 7, 8, budget.workspace_bytes);
            for (const auto* text : {&slot.support->program_identity, &slot.support->origin_point,
                                     &slot.support->field_provider}) {
              charge(retained, text->size(), 1, budget.retained_bytes);
              charge(wire, text->size(), 1, budget.workspace_bytes);
            }
          }
        }
        if (slot.representation) {
          slot.representation->validate();
          if (obligation_ids.try_emplace(slot.representation.get(), obligations.size() + 1)
                  .second) {
            obligations.push_back(slot.representation.get());
            charge(retained, 1, sizeof(FieldRepresentationObligation), budget.retained_bytes);
            charge(wire, 2, 8, budget.workspace_bytes);
            for (const auto* text :
                 {&slot.representation->contract, &slot.representation->origin}) {
              charge(retained, text->size(), 1, budget.retained_bytes);
              charge(wire, text->size(), 1, budget.workspace_bytes);
            }
          }
        }
      }
    }
    // Output and lookup workspace coexist. No vector allocation precedes this check.
    if (wire > std::numeric_limits<std::size_t>::max())
      throw std::length_error("history Field metadata wire is not addressable");
    auto total = workspace.used();
    charge(total, wire, 1, budget.workspace_bytes);
  }
};
}  // namespace history_field_detail

inline std::uint64_t validate_history_field_metadata_storage(
    const HistoryFieldMetadataTable& fields, const HistoryFieldMetadataBudget& budget) {
  history_field_detail::Tables tables(budget.workspace_bytes);
  tables.collect(fields, budget);
  return tables.retained;
}

inline std::vector<std::uint8_t> encode_prepared_history_field_metadata(
    const HistoryFieldMetadataTable& fields, const history_field_detail::Tables& tables) {
  using namespace history_field_detail;
  Writer out(static_cast<std::size_t>(tables.wire));
  out.word(tables.supports.size());
  for (const auto* support : tables.supports) {
    out.text(support->program_identity);
    out.text(support->origin_point);
    out.word(static_cast<std::uint64_t>(static_cast<std::int64_t>(support->program_owner)));
    out.word(static_cast<std::uint64_t>(static_cast<std::int64_t>(support->producer_level)));
    out.word(support->topology_epoch);
    out.word(support->materialization_generation);
    out.text(support->field_provider);
  }
  out.word(tables.obligations.size());
  for (const auto* obligation : tables.obligations) {
    out.text(obligation->contract);
    out.text(obligation->origin);
  }
  out.word(fields.size());
  for (const auto& [name, slots] : fields) {
    out.text(name);
    out.word(slots.size());
    for (const auto& slot : slots) {
      out.word(slot.support ? tables.support_ids.at(slot.support.get()) : 0);
      out.word(slot.representation ? tables.obligation_ids.at(slot.representation.get()) : 0);
    }
  }
  if (out.bytes.size() != tables.wire)
    throw std::logic_error("history Field metadata wire-size preflight disagrees");
  return std::move(out.bytes);
}

inline std::vector<std::uint8_t> encode_history_field_metadata(
    const HistoryFieldMetadataTable& fields, const HistoryFieldMetadataBudget& budget) {
  history_field_detail::Tables tables(budget.workspace_bytes);
  tables.collect(fields, budget);
  return encode_prepared_history_field_metadata(fields, tables);
}

// RingMap is the already authenticated installed HistoryManager::histories.
// Pass one reads only bounded byte spans and verifies shape/indices/semantic
// scalar fields before any decoded string, support, obligation or ring allocation.
template <class RingMap>
HistoryFieldMetadataTable decode_history_field_metadata(std::span<const std::uint8_t> bytes,
                                                        const RingMap& rings,
                                                        const HistoryFieldMetadataBudget& budget) {
  using namespace history_field_detail;
  budget.validate();
  if (!budget.enabled() || bytes.size() > budget.workspace_bytes)
    throw std::length_error("history Field metadata decoding exceeds its workspace capacity");
  Reader preflight{bytes};
  std::uint64_t retained = 0;
  const auto support_count = preflight.count(56);
  for (std::size_t i = 0; i < support_count; ++i) {
    const auto program = preflight.text(), origin = preflight.text();
    const int owner = integer(preflight.word()), level = integer(preflight.word());
    (void)preflight.word();
    (void)preflight.word();
    const auto provider = preflight.text();
    if (program.empty() || origin.empty() || level < 0 ||
        !((owner >= 0 && provider.empty()) || (owner == -1 && !provider.empty())))
      throw std::invalid_argument("history Field metadata has invalid producer support");
    charge(retained, 1, sizeof(ContinuationFieldSupport), budget.retained_bytes);
    for (const auto text : {program, origin, provider})
      charge(retained, text.size(), 1, budget.retained_bytes);
  }
  const auto obligation_count = preflight.count(16);
  for (std::size_t i = 0; i < obligation_count; ++i) {
    const auto contract = preflight.text(), origin = preflight.text();
    if (contract.empty() || origin.empty())
      throw std::invalid_argument("history Field metadata has an incomplete representation");
    charge(retained, 1, sizeof(FieldRepresentationObligation), budget.retained_bytes);
    charge(retained, contract.size(), 1, budget.retained_bytes);
    charge(retained, origin.size(), 1, budget.retained_bytes);
  }
  if (preflight.count(16) != rings.size())
    throw std::invalid_argument("history Field metadata omits or adds a ring");
  std::uint64_t seen_supports = 0, seen_obligations = 0;
  for (const auto& [name, ring] : rings) {
    if (preflight.text() != name || preflight.word() != ring.size())
      throw std::invalid_argument("history Field metadata changes ring identity/order/depth");
    charge(retained, 1, sizeof(HistoryFieldMetadataTable::value_type), budget.retained_bytes);
    charge(retained, name.size(), 1, budget.retained_bytes);
    charge(retained, ring.size(), sizeof(HistoryFieldMetadata), budget.retained_bytes);
    for (std::size_t slot = 0; slot < ring.size(); ++slot) {
      const auto support = preflight.word(), obligation = preflight.word();
      if (support > support_count || obligation > obligation_count || (obligation && !support) ||
          support > seen_supports + 1 || obligation > seen_obligations + 1)
        throw std::invalid_argument("history Field metadata has an invalid retained object index");
      seen_supports = std::max(seen_supports, support);
      seen_obligations = std::max(seen_obligations, obligation);
    }
  }
  if (seen_supports != support_count || seen_obligations != obligation_count)
    throw std::invalid_argument("history Field metadata contains unreferenced retained objects");
  if (preflight.at != bytes.size())
    throw std::invalid_argument("history Field metadata has trailing bytes");

  continuation_detail::StructuralWorkspaceResource scratch(budget.workspace_bytes - bytes.size());
  std::pmr::vector<ContinuationFieldSupport::Ref> supports(&scratch);
  std::pmr::vector<FieldRepresentationObligation::Ref> obligations(&scratch);
  supports.reserve(support_count);
  obligations.reserve(obligation_count);
  Reader in{bytes};
  (void)in.word();
  for (std::size_t i = 0; i < support_count; ++i) {
    const auto program = in.text(), origin = in.text();
    const int owner = integer(in.word()), level = integer(in.word());
    const auto epoch = in.word(), generation = in.word();
    const auto provider = in.text();
    supports.emplace_back(
        new const ContinuationFieldSupport{std::string(program), std::string(origin), owner, level,
                                           epoch, generation, std::string(provider)});
  }
  (void)in.word();
  for (std::size_t i = 0; i < obligation_count; ++i) {
    const auto contract = in.text(), origin = in.text();
    obligations.emplace_back(
        new const FieldRepresentationObligation{std::string(contract), std::string(origin)});
  }
  (void)in.word();
  HistoryFieldMetadataTable result;
  for (const auto& [expected, ring] : rings) {
    (void)in.text();
    (void)in.word();
    auto& slots = result[expected];
    slots.resize(ring.size());
    for (auto& slot : slots) {
      const auto support = in.word(), obligation = in.word();
      slot.support = support ? supports[support - 1] : ContinuationFieldSupport::Ref{};
      slot.representation =
          obligation ? obligations[obligation - 1] : FieldRepresentationObligation::Ref{};
    }
  }
  return result;
}

}  // namespace pops::runtime::program
