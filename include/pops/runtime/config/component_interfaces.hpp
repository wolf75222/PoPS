#pragma once

// Narrow native component interfaces and manifest-driven registration (ADC-681).
//
// There is deliberately no universal component base class and no provides(any) escape hatch.
// Each compile-time conformer implements only the concepts it needs; the runtime registry stores
// only the exact interface declarations from ComponentManifest v2.

#include <pops/runtime/config/component_manifest.hpp>
#include <pops/runtime/config/generated_component_catalog.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pops::component {

template <class T>
concept Requirement = requires(const T& value) { value.requirements(); };

template <class T, class Context>
concept Lowering = requires(const T& value, Context& context) { value.lower(context); };

template <class T>
concept Stencil = requires(const T& value) { value.stencil(); };

template <class T>
concept Stability = requires(const T& value) { value.stability(); };

template <class T>
concept Provider = requires(const T& value) { value.providers(); };

template <class T>
concept Effects = requires(const T& value) { value.effects(); };

template <class T>
concept Restart = requires(const T& value) { value.restart(); };

template <class T>
concept Report = requires(const T& value) { value.report(); };

template <class T, class Value>
concept Format = requires(const T& formatter, const Value& value) { formatter.format(value); };

enum class EvaluationStatus : std::uint8_t { kOk, kRetry, kReject, kFailed };

template <class T>
struct EvaluationOutcome {
  EvaluationStatus status = EvaluationStatus::kFailed;
  std::optional<T> value;
  std::string reason;

  [[nodiscard]] bool succeeded() const noexcept { return status == EvaluationStatus::kOk; }

  static EvaluationOutcome ok(T result) {
    return {EvaluationStatus::kOk, std::optional<T>(std::move(result)), {}};
  }
  static EvaluationOutcome retry(std::string why) {
    require_reason(why);
    return {EvaluationStatus::kRetry, std::nullopt, std::move(why)};
  }
  static EvaluationOutcome reject(std::string why) {
    require_reason(why);
    return {EvaluationStatus::kReject, std::nullopt, std::move(why)};
  }
  static EvaluationOutcome failed(std::string why) {
    require_reason(why);
    return {EvaluationStatus::kFailed, std::nullopt, std::move(why)};
  }

 private:
  static void require_reason(const std::string& reason) {
    if (reason.empty())
      throw std::invalid_argument("failed EvaluationOutcome requires a reason");
  }
};

template <>
struct EvaluationOutcome<void> {
  EvaluationStatus status = EvaluationStatus::kFailed;
  std::string reason;

  [[nodiscard]] bool succeeded() const noexcept { return status == EvaluationStatus::kOk; }
  static EvaluationOutcome ok() { return {EvaluationStatus::kOk, {}}; }
  static EvaluationOutcome retry(std::string why) {
    require_reason(why);
    return {EvaluationStatus::kRetry, std::move(why)};
  }
  static EvaluationOutcome reject(std::string why) {
    require_reason(why);
    return {EvaluationStatus::kReject, std::move(why)};
  }
  static EvaluationOutcome failed(std::string why) {
    require_reason(why);
    return {EvaluationStatus::kFailed, std::move(why)};
  }

 private:
  static void require_reason(const std::string& reason) {
    if (reason.empty())
      throw std::invalid_argument("failed EvaluationOutcome requires a reason");
  }
};

template <class T>
struct IsEvaluationOutcome : std::false_type {};

template <class T>
struct IsEvaluationOutcome<EvaluationOutcome<T>> : std::true_type {};

template <class T>
concept EvaluationResult = IsEvaluationOutcome<std::remove_cvref_t<T>>::value;

template <class T, class... Args>
concept FallibleEvaluation = requires(const T& value, Args&&... args) {
  { value.evaluate(std::forward<Args>(args)...) } -> EvaluationResult;
};

enum class InterfaceBindingMode : std::uint8_t { kMethod, kValue, kEntryPoint };

inline InterfaceBindingMode parse_binding_mode(std::string_view mode) {
  if (mode == "method")
    return InterfaceBindingMode::kMethod;
  if (mode == "value")
    return InterfaceBindingMode::kValue;
  if (mode == "entry_point")
    return InterfaceBindingMode::kEntryPoint;
  throw std::invalid_argument("component interface mode must be method, value, or entry_point");
}

struct InterfaceBinding {
  ComponentInterfaceId interface_id{};
  InterfaceBindingMode mode{};
  std::string name;
  std::string binding;
  std::string native_symbol;
};

struct Provenance {
  std::string origin;
  std::string source_uri;
  std::string semantic_identity;
  std::string manifest_identity;
};

struct RegistrationRecord {
  std::string component_id;
  std::string component_type;
  std::vector<InterfaceBinding> interfaces;
  Provenance provenance;

  [[nodiscard]] const InterfaceBinding* find(std::string_view name) const noexcept {
    const auto found = std::find_if(interfaces.begin(), interfaces.end(),
                                    [&](const auto& row) { return row.name == name; });
    return found == interfaces.end() ? nullptr : &*found;
  }
};

inline const identity::CanonicalValue::Map& map_field(const identity::CanonicalValue::Map& source,
                                                      const std::string& name,
                                                      const std::string& path) {
  return component_manifest::require_map(component_manifest::field(source, name, path),
                                         path + "." + name);
}

inline std::string version_string(const identity::CanonicalValue::Map& manifest) {
  const auto& version = map_field(manifest, "version", "ComponentManifest");
  const auto value = [&](const char* name) {
    return component_manifest::require_integer(component_manifest::field(version, name, "version"),
                                               "version." + std::string(name), 0);
  };
  return std::to_string(value("major")) + "." + std::to_string(value("minor")) + "." +
         std::to_string(value("patch"));
}

inline RegistrationRecord registration_from_manifest(const identity::CanonicalValue& input,
                                                     std::string origin,
                                                     std::string source_uri = {}) {
  using component_manifest::field;
  using component_manifest::require_array;
  using component_manifest::require_map;
  using component_manifest::require_text;

  if (origin.empty())
    throw std::invalid_argument("component provenance origin must be non-empty");
  const component_manifest::Normalized normalized = component_manifest::normalize(input);
  const auto& manifest = normalized.full.mapping();
  const std::string& uri = require_text(field(manifest, "uri", "ComponentManifest"), "uri");
  const std::string& component_type =
      require_text(field(manifest, "component_type", "ComponentManifest"), "component_type");
  const auto& digests = map_field(manifest, "digests", "ComponentManifest");
  const auto& entry_points = map_field(manifest, "entry_points", "ComponentManifest");

  RegistrationRecord record;
  record.component_id = uri + "@" + version_string(manifest);
  record.component_type = component_type;
  record.provenance = {
      std::move(origin),
      source_uri.empty() ? uri : std::move(source_uri),
      require_text(field(digests, "semantic", "digests"), "digests.semantic"),
      require_text(field(digests, "manifest", "digests"), "digests.manifest"),
  };

  const auto& rows =
      require_array(field(manifest, "interfaces", "ComponentManifest"), "interfaces");
  record.interfaces.reserve(rows.size());
  for (std::size_t index = 0; index < rows.size(); ++index) {
    const std::string path = "interfaces[" + std::to_string(index) + "]";
    const auto& row = require_map(rows[index], path);
    const std::string& name = require_text(field(row, "name", path), path + ".name");
    const ComponentInterfaceInfo* interface = find_component_interface(name);
    if (interface == nullptr)
      throw std::invalid_argument("unknown component interface '" + name + "'");
    const std::string& mode_text = require_text(field(row, "mode", path), path + ".mode");
    const std::string& binding = require_text(field(row, "binding", path), path + ".binding");
    const InterfaceBindingMode mode = parse_binding_mode(mode_text);
    std::string symbol;
    if (mode == InterfaceBindingMode::kEntryPoint)
      symbol =
          require_text(field(entry_points, binding, "entry_points"), "entry_points." + binding);
    record.interfaces.push_back({interface->id, mode, name, binding, std::move(symbol)});
  }
  return record;
}

class Registry {
 public:
  const RegistrationRecord& register_component(RegistrationRecord record) {
    if (frozen_)
      throw std::logic_error("component Registry is frozen");
    if (record.component_id.empty() || record.provenance.semantic_identity.empty() ||
        record.provenance.manifest_identity.empty())
      throw std::invalid_argument("component registration record is incomplete");
    const auto previous = by_id_.find(record.component_id);
    if (previous != by_id_.end()) {
      const RegistrationRecord& existing = records_[previous->second];
      if (existing.provenance.semantic_identity == record.provenance.semantic_identity)
        return existing;
      throw std::invalid_argument("component identity collision for '" + record.component_id + "'");
    }
    const std::size_t index = records_.size();
    records_.push_back(std::move(record));
    by_id_.emplace(records_.back().component_id, index);
    ++revision_;
    return records_.back();
  }

  const RegistrationRecord& register_manifest(const identity::CanonicalValue& manifest,
                                              std::string origin, std::string source_uri = {}) {
    return register_component(
        registration_from_manifest(manifest, std::move(origin), std::move(source_uri)));
  }

  [[nodiscard]] const RegistrationRecord* find(std::string_view component_id) const noexcept {
    const auto found = by_id_.find(std::string(component_id));
    return found == by_id_.end() ? nullptr : &records_[found->second];
  }

  Registry& freeze() noexcept {
    frozen_ = true;
    return *this;
  }
  [[nodiscard]] bool frozen() const noexcept { return frozen_; }
  [[nodiscard]] std::uint64_t revision() const noexcept { return revision_; }
  [[nodiscard]] const std::vector<RegistrationRecord>& records() const noexcept { return records_; }

 private:
  std::vector<RegistrationRecord> records_;
  std::unordered_map<std::string, std::size_t> by_id_;
  std::uint64_t revision_ = 0;
  bool frozen_ = false;
};

}  // namespace pops::component
