#pragma once

#include <pops/core/identity/prepared_provider.hpp>

#include <concepts>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace pops {

using PreparedProviderOptionValue =
    std::variant<std::int64_t, std::uint64_t, double, bool, std::string>;

/// Provider-owned, collision-free option carrier shared by prepared native provider families.
/// Consumers authenticate the schema and exact values but never interpret provider-specific keys.
struct PreparedProviderOptions {
  std::string schema_identity;
  std::map<std::string, PreparedProviderOptionValue> values;

  [[nodiscard]] std::string exact_contract() const {
    if (schema_identity.empty())
      throw std::invalid_argument("prepared provider options require a schema identity");
    ExactContractBuilder contract;
    contract.text("pops.prepared-provider-options")
        .scalar(std::uint32_t{1})
        .text(schema_identity)
        .sequence(values, [](ExactContractBuilder& item, const auto& entry) {
          item.text(entry.first).scalar(static_cast<std::uint32_t>(entry.second.index()));
          std::visit(
              [&](const auto& value) {
                using Value = std::remove_cvref_t<decltype(value)>;
                if constexpr (std::same_as<Value, std::string>)
                  item.text(value);
                else
                  item.scalar(value);
              },
              entry.second);
        });
    return std::move(contract).release();
  }
};

}  // namespace pops
