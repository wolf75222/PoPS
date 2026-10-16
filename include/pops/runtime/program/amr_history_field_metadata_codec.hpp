#pragma once

#include <pops/runtime/program/history_field_metadata.hpp>
#include <pops/runtime/program/program_runtime_state.hpp>

#include <array>

namespace pops::runtime::program {

namespace amr_history_field_detail {
inline constexpr std::array<std::uint8_t, 8> magic{'P', 'O', 'P', 'S', 'H', 'F', 'M', '1'};

inline bool enveloped(std::span<const std::uint8_t> bytes) {
  return bytes.size() >= magic.size() && std::equal(magic.begin(), magic.end(), bytes.begin());
}

inline std::span<const std::uint8_t> blob(history_field_detail::Reader& in) {
  const auto size = in.word();
  if (size > in.bytes.size() - in.at)
    throw std::invalid_argument("truncated AMR history Field metadata envelope");
  const auto value = in.bytes.subspan(in.at, static_cast<std::size_t>(size));
  in.at += static_cast<std::size_t>(size);
  return value;
}

inline void sample(history_field_detail::Writer& out, const HistorySampleIdentity& value) {
  value.validate();
  out.word(static_cast<std::uint64_t>(value.kind));
  out.word(value.start_bits);
  out.word(value.interval_bits);
  out.word(value.ordinal);
  for (const auto word : value.reconstruction)
    out.word(word);
}

inline void require_sample(history_field_detail::Reader& in, const HistorySampleIdentity& value) {
  value.validate();
  if (in.word() != static_cast<std::uint64_t>(value.kind) || in.word() != value.start_bits ||
      in.word() != value.interval_bits || in.word() != value.ordinal)
    throw std::invalid_argument("AMR history metadata belongs to a different exact sample");
  for (const auto word : value.reconstruction)
    if (in.word() != word)
      throw std::invalid_argument("AMR history metadata changes a reconstruction identity");
}
// Local-only preflight of the complete manager image. Call before any facade or
// provider observation so the caller can converge a malformed rank collectively.
template <class Manager>
void require_manager_metadata(const Manager& manager, std::string_view program) {
  const auto require_exact_metadata_keys = [&](const auto& metadata) {
    if (metadata.size() != manager.histories.size())
      throw std::runtime_error("AMR Program accepted history remap metadata is incomplete");
    for (const auto& [key, ring] : manager.histories) {
      (void)ring;
      if (!metadata.contains(key))
        throw std::runtime_error("AMR Program accepted history remap metadata is incomplete");
    }
  };
  require_exact_metadata_keys(manager.depth);
  require_exact_metadata_keys(manager.initialized);
  require_exact_metadata_keys(manager.fill_count);
  require_exact_metadata_keys(manager.store_pending);
  require_exact_metadata_keys(manager.owner);
  require_exact_metadata_keys(manager.state_identity);
  require_exact_metadata_keys(manager.space_identity);
  require_exact_metadata_keys(manager.clock_identity);
  require_exact_metadata_keys(manager.interpolation_identity);
  require_exact_metadata_keys(manager.ghost_depth);
  require_exact_metadata_keys(manager.slot_dt);
  require_exact_metadata_keys(manager.slot_sample);
  require_exact_metadata_keys(manager.slot_fields);
  require_history_field_program(manager.slot_fields, program);
}
}  // namespace amr_history_field_detail

// The FLX representation is nested verbatim. Metadata has its own installed
// resource budget and is present even for histories with no face expression.
// Samples bind every metadata slot to the already restored History image.
template <class Manager>
std::vector<std::uint8_t> encode_amr_history_field_metadata(std::vector<std::uint8_t> flux,
                                                            const Manager& manager,
                                                            std::string_view program) {
  using namespace history_field_detail;
  const auto& budget = manager.field_metadata_budget;
  budget.validate();
  if (!budget.enabled()) {
    for (const auto& [name, slots] : manager.slot_fields)
      for (const auto& slot : slots)
        if (!slot.empty())
          throw std::invalid_argument("AMR history metadata has no installed storage budget");
    return flux;
  }
  require_history_field_program(manager.slot_fields, program);
  if (manager.slot_fields.size() != manager.histories.size())
    throw std::invalid_argument("AMR history metadata omits or adds a ring");
  std::uint64_t fixed = 32;
  charge(fixed, flux.size(), 1, budget.workspace_bytes);
  for (const auto& [name, ring] : manager.histories) {
    if (manager.slot_fields.at(name).size() != ring.size() ||
        manager.validated_samples(name).size() != ring.size())
      throw std::invalid_argument("AMR history metadata differs from exact sample depth");
    const auto& samples = manager.validated_samples(name);
    for (std::size_t slot = 0; slot < ring.size(); ++slot)
      if (!samples[slot].authenticated() && !manager.slot_fields.at(name)[slot].empty())
        throw std::invalid_argument("AMR history Field metadata lacks exact sample authority");
    charge(fixed, 16, 1, budget.workspace_bytes);
    charge(fixed, name.size(), 1, budget.workspace_bytes);
    charge(fixed, ring.size(), 64, budget.workspace_bytes);
  }
  // The original FLX vector, PMR lookup, metadata wire and full envelope can
  // coexist. Reserve their complete peak before allocating either output.
  const HistoryFieldMetadataBudget encoding_budget{budget.retained_bytes,
                                                   budget.workspace_bytes - flux.size()};
  Tables tables(encoding_budget.workspace_bytes);
  tables.collect(manager.slot_fields, encoding_budget);
  auto peak = fixed;
  charge(peak, flux.size(), 1, budget.workspace_bytes);
  charge(peak, tables.workspace.used(), 1, budget.workspace_bytes);
  charge(peak, tables.wire, 2, budget.workspace_bytes);
  const auto metadata = encode_prepared_history_field_metadata(manager.slot_fields, tables);
  std::uint64_t size = fixed;
  charge(size, metadata.size(), 1, budget.workspace_bytes);
  if (size > std::numeric_limits<std::size_t>::max())
    throw std::length_error("AMR history metadata envelope is not addressable");
  Writer out(static_cast<std::size_t>(size));
  out.bytes.insert(out.bytes.end(), amr_history_field_detail::magic.begin(),
                   amr_history_field_detail::magic.end());
  out.word(flux.size());
  out.bytes.insert(out.bytes.end(), flux.begin(), flux.end());
  out.word(metadata.size());
  out.bytes.insert(out.bytes.end(), metadata.begin(), metadata.end());
  out.word(manager.histories.size());
  for (const auto& [name, ring] : manager.histories) {
    out.text(name);
    out.word(ring.size());
    for (const auto& value : manager.validated_samples(name))
      amr_history_field_detail::sample(out, value);
  }
  if (out.bytes.size() != size)
    throw std::logic_error("AMR history metadata envelope differs from its preflight");
  return std::move(out.bytes);
}

struct PreparedAmrHistoryFieldMetadata {
  std::span<const std::uint8_t> flux;
  HistoryFieldMetadataTable fields;
};

template <class Manager>
PreparedAmrHistoryFieldMetadata prepare_amr_history_field_metadata(
    std::span<const std::uint8_t> bytes, const Manager& manager, std::string_view program) {
  const auto& budget = manager.field_metadata_budget;
  budget.validate();
  const bool present = amr_history_field_detail::enveloped(bytes);
  if (!present) {
    if (budget.enabled())
      throw std::invalid_argument("AMR checkpoint omits installed history Field metadata");
    return {bytes, {}};
  }
  if (!budget.enabled() || bytes.size() > budget.workspace_bytes)
    throw std::length_error("AMR history metadata envelope exceeds its installed workspace");
  history_field_detail::Reader in{bytes, amr_history_field_detail::magic.size()};
  const auto flux = amr_history_field_detail::blob(in);
  const auto metadata = amr_history_field_detail::blob(in);
  if (in.word() != manager.histories.size())
    throw std::invalid_argument("AMR history metadata envelope omits or adds a ring");
  for (const auto& [name, ring] : manager.histories) {
    if (in.text() != name || in.word() != ring.size())
      throw std::invalid_argument("AMR history metadata changes ring identity/order/depth");
    const auto& samples = manager.validated_samples(name);
    if (samples.size() != ring.size())
      throw std::invalid_argument("AMR history metadata differs from restored sample depth");
    for (const auto& value : samples)
      amr_history_field_detail::require_sample(in, value);
  }
  if (in.at != bytes.size())
    throw std::invalid_argument("AMR history metadata envelope has trailing bytes");
  // The metadata span is already part of the envelope: subtract only the other
  // borrowed bytes here; its decoder charges the metadata bytes plus scratch.
  const HistoryFieldMetadataBudget nested{
      budget.retained_bytes, budget.workspace_bytes - (bytes.size() - metadata.size())};
  auto fields = decode_history_field_metadata(metadata, manager.histories, nested);
  require_history_field_program(fields, program);
  for (const auto& [name, slots] : fields)
    for (std::size_t slot = 0; slot < slots.size(); ++slot)
      if (!manager.validated_samples(name)[slot].authenticated() && !slots[slot].empty())
        throw std::invalid_argument(
            "AMR restored history Field metadata lacks exact sample authority");
  return {flux, std::move(fields)};
}

}  // namespace pops::runtime::program
