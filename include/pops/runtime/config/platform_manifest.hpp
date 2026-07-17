#pragma once

/// @file
/// Explicit platform/backend/field-view ABI.  This header has no MPI or device-runtime include:
/// communicator, datatype, and device handles enter only through ExecutionContext.

#include <pops/core/identity/canonical_value.hpp>
#include <pops/core/identity/sha256.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::platform {

using identity::CanonicalValue;
inline constexpr int kPlatformContractSchemaVersion = 1;

class ContractError : public std::invalid_argument {
 public:
  ContractError(std::string field, std::string message)
      : std::invalid_argument(std::move(message)), field_(std::move(field)) {}
  [[nodiscard]] const std::string& field() const noexcept { return field_; }

 private:
  std::string field_;
};

/// A value without non-empty evidence is UNKNOWN and cannot satisfy a requirement.
struct CapabilityProof {
  CanonicalValue value{};
  std::string evidence{};

  [[nodiscard]] bool known() const noexcept { return !evidence.empty(); }

  static CapabilityProof unknown() { return {}; }
  static CapabilityProof proven(CanonicalValue value, std::string evidence) {
    if (evidence.empty())
      throw ContractError("proof.evidence", "capability proof evidence must be non-empty");
    return {std::move(value), std::move(evidence)};
  }
};

inline CapabilityProof prove_text(std::string value, std::string evidence) {
  if (value.empty())
    throw ContractError("proof.value", "proved text must be non-empty");
  return CapabilityProof::proven(CanonicalValue::text(std::move(value)), std::move(evidence));
}

inline CapabilityProof prove_bool(bool value, std::string evidence) {
  return CapabilityProof::proven(CanonicalValue(value), std::move(evidence));
}

inline CapabilityProof prove_text_set(std::vector<std::string> values, std::string evidence) {
  CanonicalValue::Array rows;
  rows.reserve(values.size());
  for (auto& value : values) {
    if (value.empty())
      throw ContractError("proof.value", "proved text set contains an empty value");
    rows.push_back(CanonicalValue::text(std::move(value)));
  }
  return CapabilityProof::proven(CanonicalValue::array(std::move(rows)), std::move(evidence));
}

inline CapabilityProof prove_int_set(std::vector<int> values, std::string evidence) {
  CanonicalValue::Array rows;
  rows.reserve(values.size());
  for (int value : values)
    rows.emplace_back(static_cast<std::int64_t>(value));
  return CapabilityProof::proven(CanonicalValue::array(std::move(rows)), std::move(evidence));
}

struct PrecisionPolicy {
  CapabilityProof storage;
  CapabilityProof compute;
  CapabilityProof accumulation;
  CapabilityProof reduction;
};

using ProofMap = std::map<std::string, CapabilityProof>;

struct PlatformManifest {
  CapabilityProof backend;
  CapabilityProof target;
  CapabilityProof abi;
  PrecisionPolicy precision;
  CapabilityProof device;
  CapabilityProof memory_spaces;
  CapabilityProof communicator;
  ProofMap capabilities;
};

struct RuntimeBackendManifest {
  CapabilityProof backend;
  CapabilityProof target;
  CapabilityProof abi;
  PrecisionPolicy precision;
  CapabilityProof device;
  CapabilityProof memory_spaces;
  CapabilityProof communicator;
  ProofMap capabilities;
};

struct ExecutionResource {
  std::string identity;
  std::uintptr_t handle = 0;
  bool has_handle = false;
};

struct ExecutionContext {
  RuntimeBackendManifest backend;
  ExecutionResource communicator;
  ExecutionResource datatype;
  ExecutionResource device;
};

struct FieldViewDescriptor {
  std::string name;
  int dimension = 0;
  std::vector<std::size_t> extents;
  std::vector<std::ptrdiff_t> strides;
  std::string centering;
  std::vector<std::pair<int, int>> ghosts;
  std::string scalar;
  std::string memory_space;
  std::string patch;
  std::string layout;
  std::string ownership;
};

inline const CanonicalValue& require(const CapabilityProof& proof, const std::string& field) {
  if (!proof.known())
    throw ContractError(field, field + " has no capability proof; unknown is absence of proof");
  return proof.value;
}

inline const std::string& require_text(const CapabilityProof& proof, const std::string& field) {
  const auto& value = require(proof, field);
  if (value.kind() != CanonicalValue::Kind::kText || value.text_value().empty())
    throw ContractError(field, field + " proof must contain non-empty text");
  return value.text_value();
}

inline std::vector<std::string> require_text_set(const CapabilityProof& proof,
                                                 const std::string& field) {
  const auto& value = require(proof, field);
  if (value.kind() != CanonicalValue::Kind::kArray)
    throw ContractError(field, field + " proof must contain a text sequence");
  std::vector<std::string> out;
  for (const auto& item : value.items()) {
    if (item.kind() != CanonicalValue::Kind::kText || item.text_value().empty())
      throw ContractError(field, field + " proof contains a non-text value");
    out.push_back(item.text_value());
  }
  return out;
}

inline std::vector<int> require_int_set(const CapabilityProof& proof, const std::string& field) {
  const auto& value = require(proof, field);
  if (value.kind() != CanonicalValue::Kind::kArray)
    throw ContractError(field, field + " proof must contain an integer sequence");
  std::vector<int> out;
  for (const auto& item : value.items()) {
    if (item.kind() != CanonicalValue::Kind::kInt)
      throw ContractError(field, field + " proof contains a non-integer value");
    out.push_back(static_cast<int>(item.integer()));
  }
  return out;
}

inline CanonicalValue proof_data(const CapabilityProof& proof) {
  return CanonicalValue::map(
      {{"value", proof.value}, {"evidence", CanonicalValue::text(proof.evidence)}});
}

inline CanonicalValue precision_data(const PrecisionPolicy& precision) {
  return CanonicalValue::map({
      {"storage", proof_data(precision.storage)},
      {"compute", proof_data(precision.compute)},
      {"accumulation", proof_data(precision.accumulation)},
      {"reduction", proof_data(precision.reduction)},
  });
}

template <class Manifest>
inline CanonicalValue manifest_data(const Manifest& manifest) {
  CanonicalValue::Map capabilities;
  for (const auto& [name, proof] : manifest.capabilities)
    capabilities.emplace_back(name, proof_data(proof));
  return CanonicalValue::map({
      {"schema_version", CanonicalValue(std::int64_t{kPlatformContractSchemaVersion})},
      {"backend", proof_data(manifest.backend)},
      {"target", proof_data(manifest.target)},
      {"abi", proof_data(manifest.abi)},
      {"precision", precision_data(manifest.precision)},
      {"device", proof_data(manifest.device)},
      {"memory_spaces", proof_data(manifest.memory_spaces)},
      {"communicator", proof_data(manifest.communicator)},
      {"capabilities", CanonicalValue::map(std::move(capabilities))},
  });
}

template <class Manifest>
inline std::string identity_token(const char* domain, const Manifest& manifest) {
  const auto envelope = CanonicalValue::map({
      {"protocol", CanonicalValue::text("pops.identity")},
      {"domain", CanonicalValue::text(domain)},
      {"schema_version", CanonicalValue(std::int64_t{kPlatformContractSchemaVersion})},
      {"payload", manifest_data(manifest)},
  });
  return std::string("pops.") + domain +
         ".v1:sha256:" + identity::sha256_hex(identity::canonical_bytes(envelope));
}

inline void require_same(const std::string& field, const CapabilityProof& expected,
                         const CapabilityProof& actual) {
  if (identity::canonical_bytes(require(expected, "artifact." + field)) !=
      identity::canonical_bytes(require(actual, "runtime." + field)))
    throw ContractError(field, field + " mismatch between artifact and runtime backend");
}

inline const CapabilityProof& capability(const RuntimeBackendManifest& backend,
                                         const std::string& name) {
  const auto found = backend.capabilities.find(name);
  if (found == backend.capabilities.end())
    throw ContractError("capabilities." + name,
                        "runtime backend omitted required capability proof " + name);
  return found->second;
}

inline void validate_descriptor(const FieldViewDescriptor& view) {
  if (view.name.empty() || view.dimension < 1)
    throw ContractError("field", "field name must be non-empty and dimension >= 1");
  const auto rank = static_cast<std::size_t>(view.dimension);
  if (view.extents.size() != rank || view.strides.size() != rank || view.ghosts.size() != rank)
    throw ContractError("field.dimension",
                        "extents, strides, and ghosts must preserve the declared dimension");
  if (std::any_of(view.extents.begin(), view.extents.end(), [](std::size_t n) { return n == 0; }) ||
      std::any_of(view.strides.begin(), view.strides.end(),
                  [](std::ptrdiff_t n) { return n <= 0; }))
    throw ContractError("field.extents", "field extents and strides must be positive");
  if (std::any_of(view.ghosts.begin(), view.ghosts.end(),
                  [](const auto& pair) { return pair.first < 0 || pair.second < 0; }))
    throw ContractError("field.ghosts", "field ghost widths must be non-negative");
}

template <class Value>
inline void require_member(const std::string& field, const Value& value,
                           const std::vector<Value>& supported) {
  if (std::find(supported.begin(), supported.end(), value) == supported.end())
    throw ContractError(field, "field requests an unsupported " + field);
}

inline void validate_launch(const PlatformManifest& platform, const ExecutionContext& context,
                            const std::vector<FieldViewDescriptor>& fields,
                            const std::vector<FieldViewDescriptor>& expected = {}) {
  const auto& backend = context.backend;
  require_same("backend", platform.backend, backend.backend);
  require_same("target", platform.target, backend.target);
  require_same("abi", platform.abi, backend.abi);
  require_same("device", platform.device, backend.device);
  require_same("communicator", platform.communicator, backend.communicator);
  require_same("precision.storage", platform.precision.storage, backend.precision.storage);
  require_same("precision.compute", platform.precision.compute, backend.precision.compute);
  require_same("precision.accumulation", platform.precision.accumulation,
               backend.precision.accumulation);
  require_same("precision.reduction", platform.precision.reduction, backend.precision.reduction);
  if (context.communicator.identity != require_text(backend.communicator, "runtime.communicator"))
    throw ContractError("communicator", "ExecutionContext communicator identity mismatch");
  if (context.device.identity != require_text(backend.device, "runtime.device"))
    throw ContractError("device", "ExecutionContext device identity mismatch");
  if (context.communicator.identity != "serial" && !context.communicator.has_handle)
    throw ContractError("communicator", "non-serial execution requires an explicit handle");
  if (context.device.identity != "host" && context.device.identity != "cpu" &&
      !context.device.has_handle)
    throw ContractError("device", "non-host execution requires an explicit handle");

  const auto dimensions =
      require_int_set(capability(backend, "dimensions"), "runtime.capabilities.dimensions");
  const auto centerings =
      require_text_set(capability(backend, "centerings"), "runtime.capabilities.centerings");
  const auto scalars =
      require_text_set(capability(backend, "scalars"), "runtime.capabilities.scalars");
  const auto memories = require_text_set(backend.memory_spaces, "runtime.memory_spaces");
  for (const auto& view : fields) {
    validate_descriptor(view);
    require_member("dimension", view.dimension, dimensions);
    require_member("centering", view.centering, centerings);
    require_member("scalar", view.scalar, scalars);
    require_member("memory_space", view.memory_space, memories);
    if (view.scalar != context.datatype.identity)
      throw ContractError("datatype", "field scalar and ExecutionContext datatype differ");
    const auto wanted = std::find_if(expected.begin(), expected.end(),
                                     [&](const auto& item) { return item.name == view.name; });
    if (wanted != expected.end() &&
        (view.dimension != wanted->dimension || view.extents != wanted->extents ||
         view.centering != wanted->centering || view.scalar != wanted->scalar ||
         view.memory_space != wanted->memory_space))
      throw ContractError("field." + view.name, "field descriptor does not match launch contract");
  }
  for (const auto& wanted : expected)
    if (std::none_of(fields.begin(), fields.end(),
                     [&](const auto& view) { return view.name == wanted.name; }))
      throw ContractError("field." + wanted.name, "required field descriptor is missing");
}

template <class Kernel>
decltype(auto) launch_checked(const PlatformManifest& platform, const ExecutionContext& context,
                              const std::vector<FieldViewDescriptor>& fields, Kernel&& kernel,
                              const std::vector<FieldViewDescriptor>& expected = {}) {
  validate_launch(platform, context, fields, expected);
  return std::forward<Kernel>(kernel)(context, fields);
}

inline PlatformManifest proven_host_platform(const std::string& backend, const std::string& target,
                                             const std::string& abi,
                                             const std::string& communicator,
                                             const std::string& evidence) {
  const auto scalar = prove_text("float64", evidence);
  return {prove_text(backend, evidence),
          prove_text(target, evidence),
          prove_text(abi, evidence),
          {scalar, scalar, scalar, scalar},
          prove_text("host", evidence),
          prove_text_set({"host"}, evidence),
          prove_text(communicator, evidence),
          {{"dimensions", prove_int_set({2}, evidence)},
           {"centerings", prove_text_set({"cell"}, evidence)},
           {"scalars", prove_text_set({"float64"}, evidence)},
           {"layouts", prove_text_set({"right", "left", "strided"}, evidence)},
           {"ownership", prove_text_set({"borrowed", "owned", "shared"}, evidence)},
           {"generic_field_view", prove_bool(true, evidence)}}};
}

inline PlatformManifest proven_serial_platform(const std::string& backend,
                                               const std::string& target, const std::string& abi) {
  return proven_host_platform(backend, target, abi, "serial", "pops.native.2d-float64-host.v1");
}

inline RuntimeBackendManifest proven_host_backend(const std::string& backend,
                                                  const std::string& target, const std::string& abi,
                                                  const std::string& communicator,
                                                  const std::string& evidence) {
  const auto platform = proven_host_platform(backend, target, abi, communicator, evidence);
  return {platform.backend, platform.target,        platform.abi,          platform.precision,
          platform.device,  platform.memory_spaces, platform.communicator, platform.capabilities};
}

inline RuntimeBackendManifest proven_serial_backend(const std::string& backend,
                                                    const std::string& target,
                                                    const std::string& abi) {
  return proven_host_backend(backend, target, abi, "serial", "pops.native.2d-float64-host.v1");
}

}  // namespace pops::platform
