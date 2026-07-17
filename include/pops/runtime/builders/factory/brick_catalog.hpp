#pragma once

#include <pops/runtime/config/generated_component_catalog.hpp>

#include <string>

namespace pops {

inline const BrickCatalogEntry* catalog_entry(const std::string& category, const std::string& id) {
  for (const BrickCatalogEntry& entry : kBrickCatalog)
    if (category == entry.category && id == entry.id)
      return &entry;
  return nullptr;
}

inline std::string catalog_csv(const std::string& category) {
  std::string result;
  for (const BrickCatalogEntry& entry : kBrickCatalog) {
    if (category != entry.category)
      continue;
    if (!result.empty())
      result += '|';
    result += entry.id;
  }
  return result;
}

inline void append_catalog_json_string(std::string& out, const char* value) {
  static constexpr char hex[] = "0123456789abcdef";
  out += '"';
  for (const unsigned char ch : std::string(value)) {
    switch (ch) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (ch < 0x20) {
          out += "\\u00";
          out += hex[(ch >> 4) & 0xf];
          out += hex[ch & 0xf];
        } else {
          out += static_cast<char>(ch);
        }
    }
  }
  out += '"';
}

inline std::string brick_catalog_json() {
  auto boolean = [](bool value) { return value ? "true" : "false"; };
  std::string result = "{\"catalog_digest\":";
  append_catalog_json_string(result, kComponentCatalogSha256);
  result += ",\"catalog_semantic_digest\":";
  append_catalog_json_string(result, kComponentCatalogSemanticSha256);
  result += ",\"bricks\":[";
  bool first = true;
  for (const BrickCatalogEntry& entry : kBrickCatalog) {
    if (!first)
      result += ',';
    first = false;
    result += "{\"id\":";
    append_catalog_json_string(result, entry.id);
    result += ",\"category\":";
    append_catalog_json_string(result, entry.category);
    result += ",\"route_index\":";
    result += std::to_string(entry.route_index);
    result += ",\"native_entry\":";
    append_catalog_json_string(result, entry.native_entry);
    result += ",\"parameters\":";
    result += entry.parameters_json;
    result += ",\"n_vars\":";
    result += std::to_string(entry.n_vars);
    result += ",\"polar_ok\":";
    result += boolean(entry.polar_ok);
    result += ",\"requirements\":";
    result += entry.requirements_json;
    result += ",\"limitations\":";
    result += entry.limitations_json;
    result += ",\"summary\":";
    append_catalog_json_string(result, entry.summary);
    result += '}';
  }
  result += "]}";
  return result;
}

}  // namespace pops
