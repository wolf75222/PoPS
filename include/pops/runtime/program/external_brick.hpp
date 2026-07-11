#pragma once

// Process-global registry of EXTERNAL C++ bricks (Spec 3 section 21-22, criterion 20, ADC-463). A
// Spec 3 brick is native / generated / macro / external-C++; this header owns the last category. A
// user ships a brick in a standalone `.so` that, at static-init time, registers a manifest entry via
// `POPS_REGISTER_BRICK(id, category, requirements)`. The host then exports a C `pops_brick_manifest()`
// function (JSON over `BrickRegistry::instance().ids()` + each entry) that `pops.lib.load_cpp_library`
// dlopens and parses, so `pops.lib.riemann.User("id")` surfaces a real, requirement-carrying
// descriptor. This is a HOST registry (no POPS_HD, no device state): it catalogs the brick's identity
// and requirements; the brick's numerical kernel stays a separate concern wired by the codegen.

#include <pops/runtime/dynamic/abi_key.hpp>  // POPS_ABI_KEY_LITERAL: brick .so's own ABI key (ADC-611)
#include <pops/runtime/export.hpp>  // POPS_BRICK_LOCAL: hidden visibility, per-.so registry (ADC-622)

#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace pops::runtime::program {

// STRICT versioned schema of the external-brick manifest (ADC-611 / ADC-544). Emitted at the top level
// of pops_brick_manifest() so the host parser (pops.descriptors.parse_brick_manifest) refuses an
// unversioned/legacy manifest with a clear "regenerate" error. MUST stay in LOCKSTEP with the Python
// BRICK_MANIFEST_SCHEMA_VERSION. v3 makes ABI identity, annotations and every one of the ten row
// fields explicit, and rejects ambiguous duplicate registrations. Runtime readers add no defaults.
inline constexpr int kBrickManifestSchemaVersion = 3;

// Escapes a string so a manifest field built from a user-supplied id/category/CSV is always valid
// JSON: the two structural characters (`"` and `\`) plus every control character (`\n`, `\r`, `\t`,
// and the rest of `\x00`-`\x1f` as `\uXXXX`), which RFC 8259 forbids unescaped inside a string. A
// raw control character would make `json.loads` (lib.py) reject the whole manifest, so the escape
// must be complete, not just the two structural chars. The C++ reader (`field`) reverses it.
inline std::string json_escape(const std::string& s) {
  static const char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
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
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          out += "\\u00";
          out.push_back(kHex[(static_cast<unsigned char>(c) >> 4) & 0xf]);
          out.push_back(kHex[static_cast<unsigned char>(c) & 0xf]);
        } else {
          out.push_back(c);
        }
    }
  }
  return out;
}

// Reverses json_escape so a manifest reader recovers the raw value json_escape encoded: \" \\ \/
// \n \r \t \b \f and \uXXXX (manifest tokens are ASCII, so a \u escape only ever carries a control
// byte) back to the character. The C++ field reader uses this to match what json.loads (lib.py)
// yields from the same manifest. A trailing lone backslash or a truncated \u is copied verbatim.
inline std::string json_unescape(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] != '\\' || i + 1 >= s.size()) {
      out.push_back(s[i]);
      continue;
    }
    const char n = s[++i];
    switch (n) {
      case 'n':
        out.push_back('\n');
        break;
      case 'r':
        out.push_back('\r');
        break;
      case 't':
        out.push_back('\t');
        break;
      case 'b':
        out.push_back('\b');
        break;
      case 'f':
        out.push_back('\f');
        break;
      case 'u':
        if (i + 4 < s.size()) {
          out.push_back(static_cast<char>(std::stoi(s.substr(i + 1, 4), nullptr, 16) & 0xff));
          i += 4;
        }
        break;
      default:
        out.push_back(n);
        break;  // \" \\ \/ and any other escaped char -> itself
    }
  }
  return out;
}

// The manifest of one external C++ brick: its identity plus the requirements/capabilities the
// selector and codegen need. All fields are host strings (no device data); the CSV fields are
// comma-separated lists (the manifest's wire form), empty when none.
//
// Schema v3 always emits native_id / supported_layouts / supported_platforms / params / options /
// exported_symbols. native_id is the exported numerical symbol base (distinct from the user-facing
// selector id) and must be non-empty; supported_layouts /
// supported_platforms feed the compile-time layout/platform gates; exported_symbols drives the
// dlsym gate (the extern "C" symbols the .so MUST export). Empty CSV strings explicitly mean empty.
struct BrickManifestEntry {
  std::string id;                    // the brick id a user selects (e.g. "my_hllc")
  std::string category;              // the catalog slot ("riemann", "preconditioner", ...)
  std::string requirements;          // CSV of required model capabilities ("pressure,wave_speeds")
  std::string capabilities;          // CSV of capabilities the brick PROVIDES, or ""
  std::string native_id;             // required exported numerical symbol base
  std::string supported_layouts;     // CSV of supported layouts ("uniform,amr"), or "" (unconstrained)
  std::string supported_platforms;   // CSV of supported platforms ("cpu,mpi,gpu"), or "" (unknown)
  std::string params;                // CSV of runtime param names the brick reads, or ""
  std::string options;               // CSV of compile-time option keys, or ""
  std::string exported_symbols;      // CSV of extern "C" symbols the .so MUST export, or ""
};

// A PER-IMAGE catalog of registered external bricks, keyed by id. Populated at static-init time by
// `POPS_REGISTER_BRICK` (in the user's `.so`) and read by the host's `pops_brick_manifest()` exporter.
// Construction order across translation units is unspecified, so the registry is a function-local
// static (the Meyers singleton) -- it is constructed on first use, before any `POPS_REGISTER_BRICK`
// static initializer can run against it.
//
// PER-.so ISOLATION (ADC-622). The class AND instance() are POPS_BRICK_LOCAL (hidden visibility): the
// registry symbol stays out of the dynamic symbol table, so it is PRIVATE to each dlopen'd image. Two
// levels of guarantee, enumerated per platform (the contract external_riemann_brick.hpp documents:
// "RTLD_LOCAL keeps the .so's statics private"):
//   * Linux / GCC: WITHOUT hidden visibility the local static is emitted STB_GNU_UNIQUE, which glibc's
//     loader UNIFIES across every dlopen'd .so even under RTLD_LOCAL -- one brick .so's manifest would
//     then list the whole process's bricks. Hidden visibility suppresses GNU_UNIQUE (the symbol is not
//     dynamic), so each .so's registry is private. The brick .so builds also pass -fno-gnu-unique
//     (belt-and-suspenders for odd toolchains); see tests/CMakeLists.txt and the user recipe.
//   * Linux / Clang: no GNU_UNIQUE by default; hidden visibility keeps the isolation regardless.
//   * macOS: the two-level namespace already isolates a bundle's symbols; the test passes trivially.
//   * Windows: no GNU_UNIQUE; per-.dll symbols are isolated (POPS_BRICK_LOCAL is empty there).
// Comdat folding still merges the copies WITHIN one .so (a multi-TU brick shares ONE registry per
// image); the isolation is only ACROSS images -- exactly the intended contract. The exported extern "C"
// pops_brick_manifest() reader stays default-visibility (dlsym-able) and reads its OWN image's registry.
class POPS_BRICK_LOCAL BrickRegistry {
 public:
  // The single PER-IMAGE instance (hidden: not unified across dlopen'd .so, ADC-622).
  POPS_BRICK_LOCAL static BrickRegistry& instance() {
    static BrickRegistry registry;
    return registry;
  }

  // Register `entry` under `entry.id`. An exactly identical registration is idempotent; a different
  // row reusing the same id is a contract violation and is refused rather than hidden by load order.
  void register_brick(const BrickManifestEntry& entry) {
    if (entry.id.empty())
      throw std::runtime_error("BrickRegistry::register_brick: brick id must not be empty");
    if (entry.category.empty() || entry.native_id.empty())
      throw std::runtime_error(
          "BrickRegistry::register_brick: category and native_id must not be empty in schema v3");
    auto it = index_.find(entry.id);
    if (it == index_.end()) {
      index_.emplace(entry.id, entries_.size());
      entries_.push_back(entry);
      ids_.push_back(entry.id);
    } else if (!same_entry(entries_[it->second], entry)) {
      throw std::runtime_error("BrickRegistry::register_brick: conflicting registration for id '" +
                               entry.id + "'");
    }
  }

  // The manifest of brick `id`, or nullptr if no such brick is registered (never throws).
  const BrickManifestEntry* lookup(const std::string& id) const {
    auto it = index_.find(id);
    return it == index_.end() ? nullptr : &entries_[it->second];
  }

  // Every registered brick id, in registration order.
  const std::vector<std::string>& ids() const { return ids_; }

  // Every registered manifest entry, in registration order.
  const std::vector<BrickManifestEntry>& entries() const { return entries_; }

  // The manifest of every registered brick as the STRICT versioned schema (ADC-611 / ADC-544) the host
  // parser `pops.descriptors.parse_brick_manifest` accepts:
  //   {"schema_version": 3, "abi_key": "<this .so's key>", "annotations": {},
  //    "bricks": [{"id", "category", "requirements", "capabilities",
  //                "native_id", "supported_layouts", "supported_platforms",
  //                "params", "options", "exported_symbols"}, ...]}
  // Every top-level and per-entry field is present (the parser refuses missing and unknown fields).
  // Empty CSV strings explicitly mean empty sets; no identity or scientific metadata is defaulted. All CSV
  // fields carry the strings the macro registered. abi_key is the LITERAL of THIS translation unit
  // (POPS_ABI_KEY_LITERAL, a preprocessor string, no symbol -- valid inside a brick .so that never links
  // the module's abi_key()). This is the wire form a brick `.so` exports through `pops_brick_manifest()`
  // (POPS_DEFINE_BRICK_MANIFEST); the host dlopens the `.so` and feeds the returned string to
  // `_register_manifest`. Emitter and parser stay in lockstep on this version and this field set.
  std::string to_json() const {
    std::string out = "{\"schema_version\":";
    out += std::to_string(kBrickManifestSchemaVersion);
    out += ",\"abi_key\":\"";
    out += json_escape(POPS_ABI_KEY_LITERAL);
    out += "\",\"annotations\":{},\"bricks\":[";
    for (std::size_t k = 0; k < entries_.size(); ++k) {
      const BrickManifestEntry& e = entries_[k];
      if (k != 0)
        out += ',';
      out += "{\"id\":\"" + json_escape(e.id) + "\",\"category\":\"" + json_escape(e.category) +
             "\",\"requirements\":\"" + json_escape(e.requirements) + "\",\"capabilities\":\"" +
             json_escape(e.capabilities) + "\",\"native_id\":\"" + json_escape(e.native_id) +
             "\",\"supported_layouts\":\"" + json_escape(e.supported_layouts) +
             "\",\"supported_platforms\":\"" + json_escape(e.supported_platforms) +
             "\",\"params\":\"" + json_escape(e.params) + "\",\"options\":\"" +
             json_escape(e.options) + "\",\"exported_symbols\":\"" +
             json_escape(e.exported_symbols) + "\"}";
    }
    out += "]}";
    return out;
  }

  std::size_t size() const { return entries_.size(); }

  void clear() {
    entries_.clear();
    ids_.clear();
    index_.clear();
  }

 private:
  BrickRegistry() = default;

  static bool same_entry(const BrickManifestEntry& a, const BrickManifestEntry& b) {
    return a.id == b.id && a.category == b.category && a.requirements == b.requirements &&
           a.capabilities == b.capabilities && a.native_id == b.native_id &&
           a.supported_layouts == b.supported_layouts &&
           a.supported_platforms == b.supported_platforms && a.params == b.params &&
           a.options == b.options && a.exported_symbols == b.exported_symbols;
  }

  std::vector<BrickManifestEntry> entries_;  // registration-order manifest list
  std::vector<std::string> ids_;             // registration-order id list (mirrors entries_)
  std::unordered_map<std::string, std::size_t> index_;  // id -> index into entries_
};

// Registers a manifest entry at static-init time. Use at namespace scope in a brick's `.so`:
//   POPS_REGISTER_BRICK("my_hllc", "riemann", "pressure,wave_speeds");
// The capabilities and documentary CSV fields are empty by this 3-argument form; native_id is emitted
// explicitly as brick_id so the host never invents identity. A brick that provides capabilities or
// declares richer fields calls `BrickRegistry::instance().register_brick({...})` directly. The
// trailing static is a unique dummy whose initializer performs the registration (zero-cost at runtime,
// runs once before main).
#define POPS_REGISTER_BRICK(brick_id, brick_category, brick_requirements)            \
  static const bool POPS_REGISTER_BRICK_CAT_(pops_brick_registered_, __LINE__) = [] { \
    ::pops::runtime::program::BrickRegistry::instance().register_brick(              \
        {(brick_id), (brick_category), (brick_requirements), "", (brick_id)});      \
    return true;                                                                    \
  }()

// Token-paste helpers so several POPS_REGISTER_BRICK uses in one TU get distinct static names.
#define POPS_REGISTER_BRICK_CAT_(a, b) POPS_REGISTER_BRICK_CAT2_(a, b)
#define POPS_REGISTER_BRICK_CAT2_(a, b) a##b

// Exports the C reader `pops.lib.load_cpp_library` dlopens after opening a brick `.so`. Use ONCE at
// namespace scope in the brick's `.so` (after its POPS_REGISTER_BRICK calls have populated the
// registry at static-init time):
//   POPS_REGISTER_BRICK("my_hllc", "riemann", "pressure,wave_speeds");
//   POPS_DEFINE_BRICK_MANIFEST();
// It returns the JSON of EVERY brick the `.so` registered (BrickRegistry::to_json). The string is a
// function-local static built once on first call (the registry is fully populated by then, since the
// static initializers ran before any host call), so the `const char*` stays valid for the process.
#define POPS_DEFINE_BRICK_MANIFEST()                                   \
  extern "C" const char* pops_brick_manifest() {                       \
    static const std::string manifest =                               \
        ::pops::runtime::program::BrickRegistry::instance().to_json(); \
    return manifest.c_str();                                          \
  }

}  // namespace pops::runtime::program
