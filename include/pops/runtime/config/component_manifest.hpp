#pragma once

#include <pops/core/identity/canonical_value.hpp>
#include <pops/core/identity/sha256.hpp>
#include <pops/runtime/config/generated_component_catalog.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::component_manifest {

using identity::CanonicalValue;

class Error : public std::invalid_argument {
 public:
  Error(std::string code, std::string path, std::string message)
      : std::invalid_argument(std::move(message)), code_(std::move(code)), path_(std::move(path)) {}

  [[nodiscard]] const std::string& code() const noexcept { return code_; }
  [[nodiscard]] const std::string& path() const noexcept { return path_; }

 private:
  std::string code_;
  std::string path_;
};

[[noreturn]] inline void refuse(const char* code, std::string path, std::string message) {
  throw Error(code, std::move(path), std::move(message));
}

inline const CanonicalValue& field(const CanonicalValue::Map& mapping, const std::string& name,
                                   const std::string& path) {
  const CanonicalValue* found = nullptr;
  for (const auto& [key, value] : mapping) {
    if (key != name)
      continue;
    if (found != nullptr)
      refuse("duplicate_field", path, path + " contains duplicate field '" + name + "'");
    found = &value;
  }
  if (found == nullptr)
    refuse("missing_field", path + "." + name, path + " is missing field '" + name + "'");
  return *found;
}

template <std::size_t N>
inline const CanonicalValue::Map& exact_map(const CanonicalValue& value,
                                            const char* const (&expected)[N],
                                            const std::string& path) {
  if (value.kind() != CanonicalValue::Kind::kMap)
    refuse("expected_mapping", path, path + " must be a mapping");
  const auto& mapping = value.mapping();
  for (const auto& [key, unused] : mapping) {
    (void)unused;
    bool known = false;
    for (const char* candidate : expected)
      known = known || key == candidate;
    if (!known)
      refuse("unknown_semantic_field", path + "." + key,
             path + " contains unknown semantic field '" + key + "'");
  }
  for (const char* candidate : expected)
    (void)field(mapping, candidate, path);
  return mapping;
}

inline const CanonicalValue::Map& require_map(const CanonicalValue& value,
                                              const std::string& path) {
  if (value.kind() != CanonicalValue::Kind::kMap)
    refuse("expected_mapping", path, path + " must be a mapping");
  return value.mapping();
}

inline const CanonicalValue::Array& require_array(const CanonicalValue& value,
                                                  const std::string& path) {
  if (value.kind() != CanonicalValue::Kind::kArray)
    refuse("expected_sequence", path, path + " must be an array");
  return value.items();
}

inline const std::string& require_text(const CanonicalValue& value, const std::string& path,
                                       bool allow_empty = false) {
  if (value.kind() != CanonicalValue::Kind::kText || (!allow_empty && value.text_value().empty()))
    refuse("expected_text", path,
           path + " must be " + std::string(allow_empty ? "text" : "non-empty text"));
  return value.text_value();
}

inline const std::string& require_canonical_text(const CanonicalValue& value,
                                                 const std::string& path,
                                                 bool allow_empty = false) {
  const std::string& text = require_text(value, path, allow_empty);
  if (!text.empty() && (std::isspace(static_cast<unsigned char>(text.front())) != 0 ||
                        std::isspace(static_cast<unsigned char>(text.back())) != 0))
    refuse("invalid_string", path, path + " must not have surrounding whitespace");
  return text;
}

inline bool identifier_spelling(const std::string& text) {
  if (text.empty() || text.front() < 'a' || text.front() > 'z')
    return false;
  return std::all_of(text.begin() + 1, text.end(), [](unsigned char ch) {
    return (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '.' ||
           ch == '-';
  });
}

inline bool member_identifier_spelling(const std::string& text) {
  if (text.empty() || !((text.front() >= 'A' && text.front() <= 'Z') ||
                        (text.front() >= 'a' && text.front() <= 'z') || text.front() == '_'))
    return false;
  return std::all_of(text.begin() + 1, text.end(), [](unsigned char ch) {
    return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') ||
           ch == '_';
  });
}

inline bool absolute_namespaced_uri(const std::string& uri) {
  if (uri.empty() || uri.find_first_of("?#") != std::string::npos)
    return false;
  const std::size_t colon = uri.find(':');
  if (colon == std::string::npos || colon == 0 || uri[0] < 'a' || uri[0] > 'z')
    return false;
  if (std::any_of(uri.begin(), uri.end(), [](unsigned char ch) { return std::isspace(ch) != 0; }))
    return false;
  for (std::size_t index = 1; index < colon; ++index) {
    const unsigned char ch = static_cast<unsigned char>(uri[index]);
    if (!((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')) && ch != '+' && ch != '-' &&
        ch != '.')
      return false;
  }
  const std::string remainder = uri.substr(colon + 1);
  if (remainder.rfind("//", 0) != 0)
    return false;
  const std::size_t authority_end = remainder.find('/', 2);
  return (authority_end == std::string::npos ? remainder.size() : authority_end) > 2;
}

inline std::int64_t require_integer(const CanonicalValue& value, const std::string& path,
                                    std::int64_t minimum) {
  if (value.kind() != CanonicalValue::Kind::kInt || value.integer() < minimum)
    refuse("invalid_integer", path, path + " must be an integer >= " + std::to_string(minimum));
  return value.integer();
}

inline CanonicalValue normalize_set_array(const CanonicalValue& value, const std::string& path,
                                          bool text_only = false) {
  auto rows = require_array(value, path);
  if (text_only)
    for (std::size_t index = 0; index < rows.size(); ++index)
      (void)require_canonical_text(rows[index], path + "[" + std::to_string(index) + "]");
  std::sort(rows.begin(), rows.end(), [](const CanonicalValue& left, const CanonicalValue& right) {
    const auto left_bytes = identity::canonical_bytes(left);
    const auto right_bytes = identity::canonical_bytes(right);
    if (left_bytes.size() != right_bytes.size())
      return left_bytes.size() < right_bytes.size();
    return left_bytes < right_bytes;
  });
  for (std::size_t index = 1; index < rows.size(); ++index)
    if (identity::canonical_bytes(rows[index - 1]) == identity::canonical_bytes(rows[index]))
      refuse("duplicate_value", path, path + " contains duplicate semantic values");
  return CanonicalValue::array(std::move(rows));
}

inline bool set_field(const std::string& name) {
  return name == "facets" || name == "reads" || name == "writes" || name == "parameters" ||
         name == "interfaces" || name == "requirements" || name == "capabilities" ||
         name == "effects" || name == "layouts" || name == "clocks" || name == "conservation";
}

inline CanonicalValue normalize_version(const CanonicalValue& value) {
  static constexpr const char* fields[] = {"major", "minor", "patch"};
  const auto& mapping = exact_map(value, fields, "version");
  for (const char* name : fields)
    (void)require_integer(field(mapping, name, "version"), "version." + std::string(name), 0);
  return value;
}

inline CanonicalValue normalize_target(const CanonicalValue& value) {
  const auto& mapping = exact_map(value, kComponentTargetFields, "target");
  static constexpr const char* variant_fields[] = {"dimension", "scalar", "device", "features"};
  const auto& variants = require_array(field(mapping, "variants", "target"), "target.variants");
  CanonicalValue::Array normalized;
  normalized.reserve(variants.size());
  for (std::size_t index = 0; index < variants.size(); ++index) {
    const std::string path = "target.variants[" + std::to_string(index) + "]";
    const auto& variant = exact_map(variants[index], variant_fields, path);
    (void)require_integer(field(variant, "dimension", path), path + ".dimension", 1);
    (void)require_canonical_text(field(variant, "scalar", path), path + ".scalar");
    (void)require_canonical_text(field(variant, "device", path), path + ".device");
    normalized.push_back(CanonicalValue::map({
        {"dimension", field(variant, "dimension", path)},
        {"scalar", field(variant, "scalar", path)},
        {"device", field(variant, "device", path)},
        {"features",
         normalize_set_array(field(variant, "features", path), path + ".features", true)},
    }));
  }
  return CanonicalValue::map({
      {"variants",
       normalize_set_array(CanonicalValue::array(std::move(normalized)), "target.variants")},
  });
}

inline CanonicalValue normalize_determinism(const CanonicalValue& value) {
  static constexpr const char* fields[] = {"classification", "scope"};
  const auto& mapping = exact_map(value, fields, "determinism");
  const std::string& classification = require_canonical_text(
      field(mapping, "classification", "determinism"), "determinism.classification");
  if (classification != "unspecified" && classification != "bitwise" &&
      classification != "reproducible" && classification != "statistical" &&
      classification != "nondeterministic")
    refuse("invalid_determinism", "determinism.classification",
           "determinism.classification has an unsupported value");
  return CanonicalValue::map({
      {"classification", field(mapping, "classification", "determinism")},
      {"scope",
       normalize_set_array(field(mapping, "scope", "determinism"), "determinism.scope", true)},
  });
}

inline CanonicalValue normalize_restart(const CanonicalValue& value) {
  static constexpr const char* fields[] = {"mode", "schema_uri", "schema_version"};
  const auto& mapping = exact_map(value, fields, "restart");
  const std::string& mode =
      require_canonical_text(field(mapping, "mode", "restart"), "restart.mode");
  if (mode != "stateless" && mode != "stateful" && mode != "unsupported")
    refuse("invalid_restart_mode", "restart.mode", "restart.mode has an unsupported value");
  const std::string& schema_uri =
      require_canonical_text(field(mapping, "schema_uri", "restart"), "restart.schema_uri", true);
  const std::int64_t schema_version =
      require_integer(field(mapping, "schema_version", "restart"), "restart.schema_version", 0);
  if (mode == "stateful") {
    if (!absolute_namespaced_uri(schema_uri) || schema_version < 1)
      refuse("invalid_restart_schema", "restart",
             "stateful restart requires an absolute schema URI and positive version");
  } else if (!schema_uri.empty() || schema_version != 0) {
    refuse("restart_schema_without_state", "restart",
           "stateless/unsupported restart must use an empty schema URI and version 0");
  }
  return value;
}

inline CanonicalValue normalize_precision(const CanonicalValue& value) {
  static constexpr const char* fields[] = {"inputs", "accumulation", "outputs"};
  const auto& mapping = exact_map(value, fields, "precision");
  (void)require_canonical_text(field(mapping, "accumulation", "precision"),
                               "precision.accumulation");
  return CanonicalValue::map({
      {"inputs",
       normalize_set_array(field(mapping, "inputs", "precision"), "precision.inputs", true)},
      {"accumulation", field(mapping, "accumulation", "precision")},
      {"outputs",
       normalize_set_array(field(mapping, "outputs", "precision"), "precision.outputs", true)},
  });
}

inline CanonicalValue normalize_entry_points(const CanonicalValue& value) {
  const auto& mapping = require_map(value, "entry_points");
  for (const auto& [name, entry] : mapping) {
    if (!identifier_spelling(name))
      refuse("invalid_entry_point", "entry_points", "entry point name is not canonical");
    (void)require_canonical_text(entry, "entry_points." + name);
  }
  return value;
}

inline bool has_field(const CanonicalValue::Map& mapping, const std::string& name) {
  return std::any_of(mapping.begin(), mapping.end(),
                     [&](const auto& row) { return row.first == name; });
}

inline CanonicalValue normalize_interfaces(const CanonicalValue& value,
                                           const CanonicalValue& facets,
                                           const CanonicalValue& entry_points) {
  static constexpr const char* fields[] = {"name", "mode", "binding"};
  const auto& rows = require_array(value, "interfaces");
  const auto& entries = require_map(entry_points, "entry_points");
  CanonicalValue::Array normalized;
  std::vector<std::string> names;
  normalized.reserve(rows.size());
  names.reserve(rows.size());
  for (std::size_t index = 0; index < rows.size(); ++index) {
    const std::string path = "interfaces[" + std::to_string(index) + "]";
    const auto& row = exact_map(rows[index], fields, path);
    const std::string& name = require_canonical_text(field(row, "name", path), path + ".name");
    if (find_component_interface(name) == nullptr)
      refuse("unknown_component_interface", path + ".name",
             "unknown component interface '" + name + "'");
    if (std::find(names.begin(), names.end(), name) != names.end())
      refuse("duplicate_component_interface", path + ".name",
             "component interface '" + name + "' is declared more than once");
    names.push_back(name);
    const std::string& mode = require_canonical_text(field(row, "mode", path), path + ".mode");
    if (mode != "method" && mode != "value" && mode != "entry_point")
      refuse("invalid_interface_mode", path + ".mode",
             "interface mode must be method, value, or entry_point");
    const std::string& binding =
        require_canonical_text(field(row, "binding", path), path + ".binding");
    if (!member_identifier_spelling(binding))
      refuse("invalid_string", path + ".binding", path + ".binding has a non-canonical spelling");
    if (mode == "entry_point" && !has_field(entries, binding))
      refuse("missing_interface_entry_point", path + ".binding",
             "interface '" + name + "' binds undeclared entry point '" + binding + "'");
    normalized.push_back(CanonicalValue::map({
        {"name", field(row, "name", path)},
        {"mode", field(row, "mode", path)},
        {"binding", field(row, "binding", path)},
    }));
  }
  std::sort(names.begin(), names.end());
  std::vector<std::string> facet_names;
  for (const auto& facet : require_array(facets, "facets"))
    facet_names.push_back(require_canonical_text(facet, "facets[]"));
  std::sort(facet_names.begin(), facet_names.end());
  if (names != facet_names)
    refuse("interface_facet_mismatch", "interfaces",
           "facets and interface declarations must name the same exact set");
  return normalize_set_array(CanonicalValue::array(std::move(normalized)), "interfaces");
}

inline CanonicalValue normalize_extensions(const CanonicalValue& value) {
  const auto& extensions = require_map(value, "extensions");
  CanonicalValue::Map normalized;
  normalized.reserve(extensions.size());
  for (const auto& [name, extension] : extensions) {
    if (!absolute_namespaced_uri(name))
      refuse("invalid_extension_namespace", "extensions." + name,
             "extension namespace must be an absolute URI");
    const auto& mapping = require_map(extension, "extensions." + name);
    const auto& kind_value = field(mapping, "kind", "extensions." + name);
    const std::string& kind = require_text(kind_value, "extensions." + name + ".kind");
    if (kind == "documentary") {
      static constexpr const char* documentary_fields[] = {"kind", "data"};
      (void)exact_map(extension, documentary_fields, "extensions." + name);
    } else if (kind == "semantic") {
      static constexpr const char* semantic_fields[] = {"kind", "schema_uri", "schema_version",
                                                        "data"};
      const auto& semantic = exact_map(extension, semantic_fields, "extensions." + name);
      const auto& schema_uri = field(semantic, "schema_uri", "extensions." + name);
      const std::string& uri =
          require_canonical_text(schema_uri, "extensions." + name + ".schema_uri");
      if (!absolute_namespaced_uri(uri))
        refuse("invalid_extension_schema_uri", "extensions." + name + ".schema_uri",
               "semantic extension schema_uri must be absolute");
      (void)require_integer(field(semantic, "schema_version", "extensions." + name),
                            "extensions." + name + ".schema_version", 1);
    } else {
      refuse("unknown_extension_kind", "extensions." + name + ".kind",
             "extension kind must be documentary or semantic");
    }
    normalized.emplace_back(name, extension);
  }
  return CanonicalValue::map(std::move(normalized));
}

inline std::string identity_token(const char* domain, const CanonicalValue& payload) {
  CanonicalValue envelope = CanonicalValue::map({
      {"protocol", CanonicalValue::text("pops.identity")},
      {"domain", CanonicalValue::text(domain)},
      {"schema_version",
       CanonicalValue(static_cast<std::int64_t>(kComponentManifestSchemaVersion))},
      {"payload", payload},
  });
  return std::string("pops.") + domain + ".v" + std::to_string(kComponentManifestSchemaVersion) +
         ":sha256:" + identity::sha256_hex(identity::canonical_bytes(envelope));
}

struct Normalized {
  CanonicalValue full;
  CanonicalValue manifest_payload;
  CanonicalValue semantic_payload;
};

inline Normalized normalize(const CanonicalValue& value) {
  const auto& input = exact_map(value, kComponentManifestTopLevelFields, "ComponentManifest");
  if (require_integer(field(input, "schema_version", "ComponentManifest"), "schema_version", 1) !=
      kComponentManifestSchemaVersion)
    refuse("unsupported_schema_version", "schema_version",
           "unsupported ComponentManifest schema_version");
  const std::string& uri = require_canonical_text(field(input, "uri", "ComponentManifest"), "uri");
  if (!absolute_namespaced_uri(uri))
    refuse("invalid_component_uri", "uri", "uri must be an absolute namespaced URI");
  const std::string& component_type =
      require_canonical_text(field(input, "component_type", "ComponentManifest"), "component_type");
  if (!identifier_spelling(component_type))
    refuse("invalid_component_type", "component_type", "component_type is not canonical");

  const CanonicalValue normalized_facets =
      normalize_set_array(field(input, "facets", "ComponentManifest"), "facets", true);
  const CanonicalValue normalized_entry_points =
      normalize_entry_points(field(input, "entry_points", "ComponentManifest"));
  const CanonicalValue normalized_interfaces = normalize_interfaces(
      field(input, "interfaces", "ComponentManifest"), normalized_facets, normalized_entry_points);

  CanonicalValue::Map full;
  CanonicalValue::Map manifest_payload;
  CanonicalValue::Map semantic_payload;
  CanonicalValue normalized_extensions;
  for (const auto& [name, item] : input) {
    CanonicalValue normalized = item;
    if (name == "facets")
      normalized = normalized_facets;
    else if (name == "interfaces")
      normalized = normalized_interfaces;
    else if (name == "entry_points")
      normalized = normalized_entry_points;
    else if (set_field(name))
      normalized = normalize_set_array(item, name, name == "facets");
    else if (name == "version")
      normalized = normalize_version(item);
    else if (name == "signature")
      (void)require_map(item, "signature");
    else if (name == "target")
      normalized = normalize_target(item);
    else if (name == "determinism")
      normalized = normalize_determinism(item);
    else if (name == "restart")
      normalized = normalize_restart(item);
    else if (name == "precision")
      normalized = normalize_precision(item);
    else if (name == "extensions") {
      normalized = normalize_extensions(item);
      normalized_extensions = normalized;
    } else if (name == "digests") {
      const auto& digests = exact_map(item, kComponentDigestFields, "digests");
      (void)require_text(field(digests, "semantic", "digests"), "digests.semantic");
      (void)require_text(field(digests, "manifest", "digests"), "digests.manifest");
    }
    full.emplace_back(name, normalized);
    if (name != "digests")
      manifest_payload.emplace_back(name, normalized);
    if (name != "extensions" && name != "digests")
      semantic_payload.emplace_back(name, normalized);
  }

  const auto& extension_map = require_map(normalized_extensions, "extensions");
  CanonicalValue::Map semantic_extensions;
  for (const auto& [name, extension] : extension_map) {
    const auto& mapping = require_map(extension, "extensions." + name);
    if (require_text(field(mapping, "kind", "extensions." + name),
                     "extensions." + name + ".kind") == "semantic")
      semantic_extensions.emplace_back(name, extension);
  }
  if (!semantic_extensions.empty())
    semantic_payload.emplace_back("semantic_extensions",
                                  CanonicalValue::map(std::move(semantic_extensions)));

  Normalized result{
      CanonicalValue::map(std::move(full)),
      CanonicalValue::map(std::move(manifest_payload)),
      CanonicalValue::map(std::move(semantic_payload)),
  };
  const auto& normalized_full = result.full.mapping();
  const auto& digests = exact_map(field(normalized_full, "digests", "ComponentManifest"),
                                  kComponentDigestFields, "digests");
  const std::string& supplied_semantic =
      require_text(field(digests, "semantic", "digests"), "digests.semantic");
  const std::string& supplied_manifest =
      require_text(field(digests, "manifest", "digests"), "digests.manifest");
  if (supplied_semantic != identity_token("component-semantics", result.semantic_payload))
    refuse("semantic_digest_mismatch", "digests.semantic",
           "ComponentManifest semantic digest does not match canonical semantics");
  if (supplied_manifest != identity_token("component-manifest", result.manifest_payload))
    refuse("manifest_digest_mismatch", "digests.manifest",
           "ComponentManifest digest does not match canonical content");
  return result;
}

inline CanonicalValue::Bytes canonical_bytes(const CanonicalValue& value) {
  return identity::canonical_bytes(normalize(value).full);
}

}  // namespace pops::component_manifest
