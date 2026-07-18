#ifndef POPS_RUNTIME_PROGRAM_MODULE_METADATA_HPP
#define POPS_RUNTIME_PROGRAM_MODULE_METADATA_HPP

// GeneratedModule metadata (Spec 2 / ADC-442). A combined model+program ``problem.so`` carries,
// alongside ``GeneratedProgram`` (the installed step), a ``GeneratedModule`` descriptor: the typed
// operator registry the Python codegen emits (pops.time.Program._emit_module_metadata) as a set of
// ``extern "C"`` accessors. This header reads that descriptor from an already-dlopen'd handle, for
// INTROSPECTION and install-time requirement validation. It is read ONCE at install; the step body
// never touches it, so operators stay inlined and there is NO string lookup in any hot kernel.
//
#include <pops/runtime/dynamic/dynlib.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {
namespace runtime {
namespace program {

/// Integer id of an operator within a module: its registration index. The generated .so addresses
/// operators by this id; the name/kind/signature strings are metadata only (debug, introspection,
/// validation), never a hot-path lookup.
using OperatorId = std::uint32_t;

/// Integer id of a state or field space within a module.
using SpaceId = std::uint32_t;

/// One operator's metadata, as exported by the .so.
struct OperatorMetadata {
  OperatorId id = 0;
  std::string owner;  ///< canonical model owner
  std::string name;
  std::string kind;          ///< one of the Spec-2 operator kinds (local_rate, field_operator, ...)
  std::string signature;     ///< human-readable typed signature
  std::string requirements;  ///< JSON, e.g. {"kind":"local_source","aux":["grad_x","grad_y"]}
};

/// The mandatory GeneratedModule descriptor read from a problem.so.
struct ModuleMetadata {
  std::vector<OperatorMetadata> operators;
  std::vector<std::string> state_spaces;
  std::vector<std::string> state_space_owners;
  std::vector<std::string> field_spaces;
  std::vector<std::string> field_space_owners;

  /// The exact owner-qualified operator, or nullptr if none.
  const OperatorMetadata* find(const std::string& owner, const std::string& name) const {
    for (const auto& op : operators) {
      if (op.owner == owner && op.name == name) {
        return &op;
      }
    }
    return nullptr;
  }

  /// Unqualified lookup succeeds only when the name is globally unique.
  const OperatorMetadata* find(const std::string& name) const {
    const OperatorMetadata* result = nullptr;
    for (const auto& op : operators) {
      if (op.name == name) {
        if (result != nullptr)
          return nullptr;
        result = &op;
      }
    }
    return result;
  }
};

namespace detail {

template <class Fn>
inline Fn require_module_symbol(pops::dynlib::handle handle, const char* symbol) {
  auto fn = reinterpret_cast<Fn>(pops::dynlib::sym(handle, symbol));
  if (fn == nullptr)
    throw std::runtime_error(std::string("compiled Program metadata symbol '") + symbol +
                             "' is missing; regenerate the artifact");
  return fn;
}

inline std::string require_module_string(const char* (*fn)(int), const char* symbol, int i) {
  const char* s = fn(i);
  if (s == nullptr || s[0] == '\0')
    throw std::runtime_error(std::string("compiled Program metadata symbol '") + symbol +
                             "' returned an empty value at index " + std::to_string(i));
  return std::string(s);
}

inline int require_module_count(pops::dynlib::handle handle, const char* symbol) {
  using CountFn = int (*)();
  const CountFn count = require_module_symbol<CountFn>(handle, symbol);
  const int n = count();
  constexpr int kMaxMetadataRows = 1 << 20;
  if (n < 0 || n > kMaxMetadataRows)
    throw std::runtime_error(std::string("compiled Program metadata symbol '") + symbol +
                             "' returned invalid count " + std::to_string(n));
  return n;
}

/// Read one mandatory owner-qualified state/field-space table.
inline std::pair<std::vector<std::string>, std::vector<std::string>> module_spaces(
    pops::dynlib::handle handle, const char* count_symbol, const char* name_symbol,
    const char* owner_symbol) {
  using StringFn = const char* (*)(int);
  const int n = require_module_count(handle, count_symbol);
  const StringFn name = require_module_symbol<StringFn>(handle, name_symbol);
  const StringFn owner = require_module_symbol<StringFn>(handle, owner_symbol);
  std::vector<std::string> names;
  std::vector<std::string> owners;
  names.reserve(static_cast<std::size_t>(n));
  owners.reserve(static_cast<std::size_t>(n));
  std::set<std::pair<std::string, std::string>> identities;
  for (int i = 0; i < n; ++i) {
    std::string current_name = require_module_string(name, name_symbol, i);
    std::string current_owner = require_module_string(owner, owner_symbol, i);
    if (!identities.emplace(current_owner, current_name).second)
      throw std::runtime_error(std::string("compiled Program metadata contains duplicate space '") +
                               current_owner + "." + current_name + "'");
    names.push_back(std::move(current_name));
    owners.push_back(std::move(current_owner));
  }
  return {std::move(names), std::move(owners)};
}

}  // namespace detail

using ProgramOperatorAuthority = std::array<std::uint64_t, 4>;

/// Read the exact prepared-operator authority table exported by the generated artifact. Both
/// accessors are mandatory even for a Program with zero prepared operators; malformed, zero, or
/// duplicate rows fail before the install entry can issue any trusted hot-apply capability.
inline std::vector<ProgramOperatorAuthority> read_program_operator_authorities(
    pops::dynlib::handle dl_handle) {
  if (!pops::dynlib::valid(dl_handle))
    throw std::runtime_error("compiled Program operator authorities require a valid module handle");
  const int count =
      detail::require_module_count(dl_handle, "pops_program_operator_authority_count");
  using WordFn = std::uint64_t (*)(int, int);
  const WordFn word = detail::require_module_symbol<WordFn>(
      dl_handle, "pops_program_operator_authority_word");
  std::vector<ProgramOperatorAuthority> authorities;
  authorities.reserve(static_cast<std::size_t>(count));
  std::set<ProgramOperatorAuthority> unique;
  for (int index = 0; index < count; ++index) {
    ProgramOperatorAuthority authority{};
    for (int lane = 0; lane < 4; ++lane)
      authority[static_cast<std::size_t>(lane)] = word(index, lane);
    if (std::all_of(authority.begin(), authority.end(),
                    [](std::uint64_t value) { return value == 0; }))
      throw std::runtime_error("compiled Program contains a zero operator authority");
    if (!unique.insert(authority).second)
      throw std::runtime_error("compiled Program contains a duplicate operator authority");
    authorities.push_back(authority);
  }
  return authorities;
}

/// Read and authenticate the complete GeneratedModule metadata from an already-open problem module.
/// Every count/accessor family is mandatory; missing, empty, duplicated, or malformed metadata fails
/// before the program can be installed.
inline ModuleMetadata read_module_metadata(pops::dynlib::handle dl_handle) {
  if (!pops::dynlib::valid(dl_handle))
    throw std::runtime_error("compiled Program metadata requires a valid module handle");
  ModuleMetadata meta;
  using StringFn = const char* (*)(int);
  const int n = detail::require_module_count(dl_handle, "pops_module_operator_count");
  const StringFn owner =
      detail::require_module_symbol<StringFn>(dl_handle, "pops_module_operator_owner");
  const StringFn name =
      detail::require_module_symbol<StringFn>(dl_handle, "pops_module_operator_name");
  const StringFn kind =
      detail::require_module_symbol<StringFn>(dl_handle, "pops_module_operator_kind");
  const StringFn signature =
      detail::require_module_symbol<StringFn>(dl_handle, "pops_module_operator_signature");
  const StringFn requirements =
      detail::require_module_symbol<StringFn>(dl_handle, "pops_module_operator_requirements");
  if (n > 0) {
    meta.operators.reserve(static_cast<std::size_t>(n));
  }
  std::set<std::pair<std::string, std::string>> operator_identities;
  for (int i = 0; i < n; ++i) {
    OperatorMetadata op;
    op.id = static_cast<OperatorId>(i);
    op.owner = detail::require_module_string(owner, "pops_module_operator_owner", i);
    op.name = detail::require_module_string(name, "pops_module_operator_name", i);
    op.kind = detail::require_module_string(kind, "pops_module_operator_kind", i);
    op.signature = detail::require_module_string(signature, "pops_module_operator_signature", i);
    op.requirements =
        detail::require_module_string(requirements, "pops_module_operator_requirements", i);
    if (op.requirements.front() != '{' || op.requirements.back() != '}')
      throw std::runtime_error("compiled Program operator '" + op.owner + "." + op.name +
                               "' has malformed requirements metadata");
    if (!operator_identities.emplace(op.owner, op.name).second)
      throw std::runtime_error("compiled Program metadata contains duplicate operator '" +
                               op.owner + "." + op.name + "'");
    meta.operators.push_back(std::move(op));
  }
  auto states =
      detail::module_spaces(dl_handle, "pops_module_state_space_count",
                            "pops_module_state_space_name", "pops_module_state_space_owner");
  meta.state_spaces = std::move(states.first);
  meta.state_space_owners = std::move(states.second);
  auto fields =
      detail::module_spaces(dl_handle, "pops_module_field_space_count",
                            "pops_module_field_space_name", "pops_module_field_space_owner");
  meta.field_spaces = std::move(fields.first);
  meta.field_space_owners = std::move(fields.second);
  return meta;
}

/// Collect the quoted tokens of a JSON string array keyed by @p key inside the operator's flat
/// ``requirements`` JSON, e.g. key ``"aux"`` over {"kind":"local_source","aux":["grad_x","B_z"]} ->
/// {"grad_x","B_z"}. A dependency-free scan: the core has no JSON library on the install path and the
/// shape is a flat, closed vocabulary (the codegen emits ``"kind"`` plus a handful of requirement
/// arrays/scalars). It locates @p key, the following ``[``, and collects the quoted tokens up to the
/// closing ``]``. Returns empty when the key is absent or is not an array. Shared by required_aux /
/// required_block (Spec criterion 24).
inline std::vector<std::string> required_string_list(const std::string& requirements_json,
                                                     const std::string& key) {
  std::vector<std::string> out;
  const std::size_t k = requirements_json.find(key);
  if (k == std::string::npos) {
    return out;
  }
  const std::size_t lb = requirements_json.find('[', k + key.size());
  if (lb == std::string::npos) {
    return out;
  }
  const std::size_t rb = requirements_json.find(']', lb);
  if (rb == std::string::npos) {
    return out;
  }
  std::size_t p = lb + 1;
  while (p < rb) {
    const std::size_t q1 = requirements_json.find('"', p);
    if (q1 == std::string::npos || q1 >= rb) {
      break;
    }
    const std::size_t q2 = requirements_json.find('"', q1 + 1);
    if (q2 == std::string::npos || q2 > rb) {
      break;
    }
    out.push_back(requirements_json.substr(q1 + 1, q2 - q1 - 1));
    p = q2 + 1;
  }
  return out;
}

/// Read a single quoted JSON string value keyed by @p key inside the operator's flat ``requirements``
/// JSON, e.g. key ``"solver"`` over {"kind":"field_operator","solver":"geometric_mg"} ->
/// "geometric_mg". Returns "" when the key is absent. Dependency-free, same closed-vocabulary scan as
/// required_string_list; used for the scalar requirement kinds (solver, capability, schedule) of
/// Spec criterion 24.
inline std::string requirement_string(const std::string& requirements_json,
                                      const std::string& key) {
  auto is_space = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
  // @p key is the quoted JSON key (e.g. "\"solver\""). Match it as a genuine KEY, not as an array
  // element or a value substring: the first non-space char before it must be '{' or ',', and the
  // first non-space char after it must be ':'. (Without this, an aux field literally named "solver"
  // -- {"aux":["solver"],...} -- or any value equal to the key would yield a bogus requirement and
  // wrongly reject a valid install.) Scan all occurrences until one is a real key.
  std::size_t k = requirements_json.find(key);
  while (k != std::string::npos) {
    std::size_t before = k;
    while (before > 0 && is_space(requirements_json[before - 1])) {
      --before;
    }
    const bool key_start =
        before == 0 || requirements_json[before - 1] == '{' || requirements_json[before - 1] == ',';
    std::size_t after = k + key.size();
    while (after < requirements_json.size() && is_space(requirements_json[after])) {
      ++after;
    }
    if (key_start && after < requirements_json.size() && requirements_json[after] == ':') {
      const std::size_t q1 = requirements_json.find('"', after + 1);
      if (q1 == std::string::npos) {
        return std::string();
      }
      const std::size_t q2 = requirements_json.find('"', q1 + 1);
      if (q2 == std::string::npos) {
        return std::string();
      }
      return requirements_json.substr(q1 + 1, q2 - q1 - 1);
    }
    k = requirements_json.find(key, k + 1);
  }
  return std::string();
}

/// Aux-field names an operator requires (the ``"aux"`` array). Used by install-time requirement
/// validation (Spec criterion 24, ADC-446); kept as a named wrapper for call-site clarity.
inline std::vector<std::string> required_aux(const std::string& requirements_json) {
  return required_string_list(requirements_json, "\"aux\"");
}

/// Block-instance names an operator requires (the ``"block"`` array), e.g. a ``collisions`` operator
/// reading another species: {"kind":"local_source","block":["ions"]} -> {"ions"}. Install-time
/// validation rejects a simulation that did not instantiate one of them (Spec criterion 24).
inline std::vector<std::string> required_blocks(const std::string& requirements_json) {
  return required_string_list(requirements_json, "\"block\"");
}

/// Solver name a field operator requires (the scalar ``"solver"`` value), e.g.
/// {"kind":"field_operator","solver":"geometric_mg"} -> "geometric_mg". Empty when the operator has
/// no solver requirement. Install-time validation rejects a simulation whose configured field solver
/// does not match (Spec criterion 24).
inline std::string required_solver(const std::string& requirements_json) {
  return requirement_string(requirements_json, "\"solver\"");
}

}  // namespace program
}  // namespace runtime
}  // namespace pops

#endif  // POPS_RUNTIME_PROGRAM_MODULE_METADATA_HPP
