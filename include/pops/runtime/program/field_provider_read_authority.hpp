#pragma once
#include <pops/runtime/system/exact_aux_registry.hpp>

#include <bit>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pops::runtime::program {
namespace provider_read_detail {
struct Writer {
  std::string* output = nullptr;
  std::uint64_t size = 0;
  void add(std::uint64_t count) {
    if (count > std::numeric_limits<std::uint64_t>::max() - size)
      throw std::length_error("provider read authority size is not representable");
    size += count;
  }
  void word(std::uint64_t value) {
    add(16);
    if (!output)
      return;
    constexpr char digits[] = "0123456789abcdef";
    for (int shift = 60; shift >= 0; shift -= 4)
      output->push_back(digits[(value >> shift) & 15]);
  }
  void text(std::string_view value) {
    word(value.size());
    add(value.size());
    if (output)
      output->append(value);
  }
};
inline void point(Writer& writer, const runtime::system::AuxiliaryEvaluationPoint& value,
                  runtime::system::AuxiliaryPointReads reads) {
  using R = runtime::system::AuxiliaryPointReads;
  value.validate();
  reads.validate();
  writer.word(reads.fields);
  if (reads.reads(R::clock))
    writer.text(value.clock);
  if (reads.reads(R::accepted_step))
    writer.word(value.accepted_step);
  if (reads.reads(R::layout_generation))
    writer.word(value.layout_generation);
  if (reads.reads(R::level))
    writer.word(std::uint64_t(value.level));
  if (reads.reads(R::substep))
    writer.word(std::uint64_t(value.substep));
  if (reads.reads(R::stage))
    writer.word(std::uint64_t(value.stage));
  if (reads.reads(R::nonlinear_iteration))
    writer.word(std::uint64_t(value.nonlinear_iteration));
  if (reads.reads(R::event))
    writer.word(std::uint64_t(value.event));
  if (reads.reads(R::stage_fraction | R::dt | R::physical_time)) {
    writer.word(bool(value.physical_evaluation));
    if (value.physical_evaluation) {
      const auto& physical = *value.physical_evaluation;
      if (reads.reads(R::stage_fraction)) {
        writer.word(std::bit_cast<std::uint64_t>(physical.stage_fraction.numerator));
        writer.word(std::bit_cast<std::uint64_t>(physical.stage_fraction.denominator));
      }
      if (reads.reads(R::dt))
        writer.word(std::bit_cast<std::uint64_t>(physical.dt));
      if (reads.reads(R::physical_time))
        writer.word(std::bit_cast<std::uint64_t>(physical.physical_time));
    }
  }
}
}  // namespace provider_read_detail

// Close the exact accepted publication backing one compact consumer slot. A
// derived equation contributes its actual point inputs plus every declared
// dependency publication. Consumer slot numbers, allocation addresses and MPI
// shard placement are not scientific input identities. No numerical evaluation
// happens here; the device journal records the values actually read separately.
template <int Dim>
std::string field_provider_read_authority(
    const runtime::system::ExactAuxiliaryRegistry<Dim>& registry, std::string_view consumer,
    std::size_t slot, std::uint64_t capacity) {
  using Provider = runtime::system::PreparedAuxiliaryProvider<Dim>;
  const auto& plan = registry.consumer_plan(consumer);
  const runtime::system::ResolvedAuxiliaryConsumerValue<Dim>* value = nullptr;
  for (const auto& item : plan.values)
    if (item.consumer_slot == slot) {
      if (value)
        throw std::invalid_argument("provider consumer has duplicate compact slot");
      value = &item;
    }
  if (!value)
    throw std::invalid_argument("provider read slot is not in its exact consumer plan");
  const auto charge = [&](std::uint64_t count, std::uint64_t width = 1) {
    if (width && count > capacity / width)
      throw std::length_error("provider read authority exceeds its explicit workspace");
    capacity -= count * width;
  };
  // Registry-owned source data remain borrowed. Compare the exact four key
  // strings without constructing encoded lookup temporaries before the budget.
  const auto producer_for =
      [&](const runtime::system::AuxiliaryComponentKey& key) -> const Provider* {
    for (const auto index : registry.topological_order()) {
      const auto& provider = registry.provider(index);
      for (const auto& output : provider.outputs())
        if (output.key.owner_qid == key.owner_qid && output.key.space_kind == key.space_kind &&
            output.key.space_name == key.space_name && output.key.component == key.component)
          return &provider;
    }
    throw std::invalid_argument("sealed registry does not produce its consumer key");
  };
  const auto count = registry.topological_order().size();
  charge(sizeof(std::vector<const Provider*>));
  charge(count, sizeof(const Provider*));
  std::vector<const Provider*> selected;
  selected.reserve(count);
  selected.push_back(producer_for(value->key));
  for (std::size_t next = 0; next < selected.size(); ++next) {
    for (const auto& dependency : selected[next]->dependencies()) {
      const auto* producer = producer_for(dependency.key);
      if (std::find(selected.begin(), selected.end(), producer) == selected.end()) {
        if (selected.size() == count)
          throw std::invalid_argument("provider dependency closure exceeds registry");
        selected.push_back(producer);
      }
    }
  }
  const auto emit = [&](provider_read_detail::Writer& writer) {
    writer.text("pops.provider-read-authority@1");
    writer.text(value->key.owner_qid);
    writer.text(value->key.space_kind);
    writer.text(value->key.space_name);
    writer.text(value->key.component);
    writer.word(selected.size());
    // Sealed registry order is canonical and independent of traversal order.
    for (const auto index : registry.topological_order()) {
      const auto& provider = registry.provider(index);
      if (std::find(selected.begin(), selected.end(), &provider) == selected.end())
        continue;
      const auto& accepted = registry.last_accepted_point(provider.identity());
      if (!accepted)
        throw std::invalid_argument("provider read has no accepted native publication");
      writer.text(provider.collective_contract());
      provider_read_detail::point(writer, *accepted, provider.point_reads());
    }
  };
  provider_read_detail::Writer measure;
  emit(measure);
  // Source strings are framed in the output; charging another output-size
  // amount conservatively includes all borrowed contracts/points/keys.
  charge(sizeof(std::string));
  charge(measure.size);
  charge(measure.size);
  if (measure.size > std::string{}.max_size())
    throw std::length_error("provider read string is too large");
  std::string result;
  result.reserve(static_cast<std::size_t>(measure.size));
  provider_read_detail::Writer writer{&result};
  emit(writer);
  if (writer.size != measure.size || result.size() != measure.size)
    throw std::logic_error("provider read authority framing changed during local preparation");
  return result;
}
}  // namespace pops::runtime::program
