#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pops::runtime::program {

// An exact description of already authenticated kernel reads. This codec does
// not grant access to a Field and cannot issue/discharge a representation. The
// context must authenticate schema, support, provider instances and each local
// row against the actual field layout before supplying these immutable views.
// Rows are [signed global coordinates..., activity, (read flag, Real bits)...].
// Host shard storage, sorting indices and the result are charged together here;
// a caller retaining a device journal must subtract its charge before this
// call.
template <int Dim>
struct FieldReadClosure {
  static_assert(Dim > 0);

  struct Input {
    std::string_view schema;
    std::string_view support;
    std::span<const std::string_view> binding_authorities;
    std::span<const std::span<const std::uint64_t>> shards;
  };

  static std::string encode(const Input& input, std::uint64_t workspace_byte_capacity) {
    if (input.schema.empty() || input.support.empty())
      throw std::invalid_argument("Field read closure requires exact schema and support");
    std::uint64_t charged = 0;
    const auto charge = [&](std::uint64_t count) {
      if (count > workspace_byte_capacity - charged)
        throw std::length_error("Field read closure exceeds its workspace byte capacity");
      charged += count;
    };
    const auto product = [](std::uint64_t a, std::uint64_t b) {
      if (b && a > std::numeric_limits<std::uint64_t>::max() / b)
        throw std::length_error("Field read closure size is not representable");
      return a * b;
    };
    const auto bindings = input.binding_authorities.size();
    if (bindings > (std::numeric_limits<std::size_t>::max() - Dim - 1) / 2)
      throw std::length_error("Field read closure width is not representable");
    const std::size_t width = Dim + 1 + 2 * bindings;
    std::size_t cells = 0;
    charge(input.schema.size());
    charge(input.support.size());
    charge(product(input.binding_authorities.size(), sizeof(std::string_view)));
    charge(product(input.shards.size(), sizeof(std::span<const std::uint64_t>)));
    for (const auto authority : input.binding_authorities)
      charge(authority.size());
    for (const auto shard : input.shards) {
      if (shard.size() % width)
        throw std::invalid_argument("Field read closure has an incomplete global cell");
      const auto count = shard.size() / width;
      if (count > std::numeric_limits<std::size_t>::max() - cells)
        throw std::length_error("Field read closure cell count is not representable");
      cells += count;
      charge(product(shard.size(), sizeof(std::uint64_t)));
    }
    // Reserve every temporary before allocation. No source payload is copied.
    charge(product(cells, sizeof(const std::uint64_t*)));
    charge(bindings);
    std::vector<unsigned char> used(bindings, 0);
    for (const auto shard : input.shards)
      for (std::size_t start = 0; start < shard.size(); start += width) {
        const auto row = shard.subspan(start, width);
        if (row[Dim] > 1)
          throw std::invalid_argument("Field read closure contains a failed kernel journal");
        for (std::size_t slot = 0; slot < bindings; ++slot) {
          const auto read = row[Dim + 1 + 2 * slot];
          const auto bits = row[Dim + 2 + 2 * slot];
          if (read > 1 || (row[Dim] == 0 && read) || (!read && bits))
            throw std::invalid_argument("Field read closure has a noncanonical read journal");
          if (read)
            used[slot] = 1;
        }
      }

    // Fixed-width hexadecimal integers plus length-framed exact strings avoid
    // ambiguous concatenation, floating-point formatting and commutative
    // hashes.
    constexpr std::string_view magic = "POPSFRC1";
    std::uint64_t output = magic.size();
    const auto add_output = [&](std::uint64_t bytes) {
      if (bytes > std::numeric_limits<std::uint64_t>::max() - output)
        throw std::length_error("Field read closure output is not representable");
      output += bytes;
    };
    add_output(16 * 4);  // dimension, bindings, cells, schema length
    add_output(input.schema.size());
    add_output(16);
    add_output(input.support.size());
    for (std::size_t slot = 0; slot < bindings; ++slot) {
      add_output(16);
      if (used[slot]) {
        if (input.binding_authorities[slot].empty())
          throw std::invalid_argument("Field read closure lacks a consumed input authority");
        add_output(input.binding_authorities[slot].size());
      }
    }
    add_output(product(product(cells, width), 16));
    charge(output);
    if (output > std::string{}.max_size())
      throw std::length_error("Field read closure exceeds native string storage");

    std::vector<const std::uint64_t*> ordered;
    ordered.reserve(cells);
    for (const auto shard : input.shards)
      for (std::size_t start = 0; start < shard.size(); start += width)
        ordered.push_back(shard.data() + start);
    const auto less = [](const auto* a, const auto* b) {
      for (int axis = 0; axis < Dim; ++axis) {
        const auto lhs = std::bit_cast<std::int64_t>(a[axis]);
        const auto rhs = std::bit_cast<std::int64_t>(b[axis]);
        if (lhs != rhs)
          return lhs < rhs;
      }
      return false;
    };
    std::sort(ordered.begin(), ordered.end(), less);
    for (std::size_t index = 1; index < ordered.size(); ++index)
      if (!less(ordered[index - 1], ordered[index]))
        throw std::invalid_argument("Field read closure repeats a global cell");

    std::string result;
    result.reserve(static_cast<std::size_t>(output));
    const auto word = [&](std::uint64_t value) {
      constexpr char digits[] = "0123456789abcdef";
      for (int shift = 60; shift >= 0; shift -= 4)
        result.push_back(digits[(value >> shift) & 15]);
    };
    const auto text = [&](std::string_view value) {
      word(value.size());
      result.append(value);
    };
    result.append(magic);
    word(Dim);
    word(bindings);
    word(cells);
    text(input.schema);
    text(input.support);
    for (std::size_t slot = 0; slot < bindings; ++slot)
      text(used[slot] ? input.binding_authorities[slot] : std::string_view{});
    for (const auto* row : ordered)
      for (std::size_t column = 0; column < width; ++column)
        word(row[column]);
    if (result.size() != output)
      throw std::logic_error("Field read closure framing disagrees with its allocation preflight");
    return result;
  }
};

}  // namespace pops::runtime::program
