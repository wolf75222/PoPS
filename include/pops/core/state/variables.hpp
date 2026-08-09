#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

/// @file
/// @brief Descriptor of a model's variables (Vars). Carried by the HYPERBOLIC brick (along with the
///        flux and the conversions), because variables and flux are physically linked; this is NOT a
///        standalone brick that can be combined freely.
///
/// `Variables` DESCRIBES the variables (conservative or primitive): kind, names, size. It is HOST
/// metadata (it does not drive the computation, which works per component via the cons<->prim
/// conversions), but it is a MANDATORY CONTRACT of the hyperbolic model (HyperbolicModel concept):
/// conservative_vars() and primitive_vars(). Used for introspection, named diagnostics, and labelled
/// output.

namespace pops {

/// Kind of a variable set: conserved (U) or primitive (W).
/// Used as a tag in VariableSet; do not use it to dispatch numerical logic.
enum class VariableKind { Conservative, Primitive };

/// Axis-independent kind of a variable semantic.
///
/// Vector semantics carry their spatial component separately in VariableSemantic::axis.  There is
/// deliberately no X/Y/Z enumerator: an algorithm can ask for `momentum(axis)` for every native
/// axis without carrying a 3D vocabulary through a 1D or 2D artifact.
enum class VariableRoleKind {
  Density,
  Momentum,
  Energy,
  Velocity,
  Pressure,
  Temperature,
  Scalar,
  Custom,
  Axial,
};

/// Physical meaning of a component, optionally qualified by one spatial axis.
///
/// This is host metadata, so `axis == -1` represents a scalar semantic and `axis >= 0` represents
/// a vector semantic.  `validate_for_dimension` is the sole rank gate: producers and consumers
/// must validate their exact native specialization before using an axis.
struct VariableSemantic {
  VariableRoleKind kind = VariableRoleKind::Custom;
  int axis = -1;

  constexpr bool operator==(const VariableSemantic&) const = default;
  constexpr bool has_axis() const noexcept { return axis >= 0; }
  constexpr bool is_vector() const noexcept {
    return kind == VariableRoleKind::Momentum || kind == VariableRoleKind::Velocity ||
           kind == VariableRoleKind::Axial;
  }

  constexpr void validate() const {
    if (is_vector() != has_axis())
      throw std::invalid_argument("vector variable semantics require exactly one non-negative axis");
  }

  template <int Dim>
  constexpr void validate_for_dimension() const {
    validate();
    if (has_axis() && axis >= Dim)
      throw std::invalid_argument("variable semantic axis lies outside the native dimension");
  }

  static constexpr VariableSemantic density() { return {VariableRoleKind::Density, -1}; }
  static constexpr VariableSemantic momentum(int axis) { return {VariableRoleKind::Momentum, axis}; }
  static constexpr VariableSemantic energy() { return {VariableRoleKind::Energy, -1}; }
  static constexpr VariableSemantic velocity(int axis) { return {VariableRoleKind::Velocity, axis}; }
  static constexpr VariableSemantic pressure() { return {VariableRoleKind::Pressure, -1}; }
  static constexpr VariableSemantic temperature() { return {VariableRoleKind::Temperature, -1}; }
  static constexpr VariableSemantic scalar() { return {VariableRoleKind::Scalar, -1}; }
  static constexpr VariableSemantic custom() { return {VariableRoleKind::Custom, -1}; }
  static constexpr VariableSemantic axial(int axis) { return {VariableRoleKind::Axial, axis}; }

  // Scalar aliases keep user models concise.  Vector components always use the factories above.
  static const VariableSemantic Density;
  static const VariableSemantic Energy;
  static const VariableSemantic Pressure;
  static const VariableSemantic Temperature;
  static const VariableSemantic Scalar;
  static const VariableSemantic Custom;
};

inline constexpr VariableSemantic VariableSemantic::Density = VariableSemantic::density();
inline constexpr VariableSemantic VariableSemantic::Energy = VariableSemantic::energy();
inline constexpr VariableSemantic VariableSemantic::Pressure = VariableSemantic::pressure();
inline constexpr VariableSemantic VariableSemantic::Temperature = VariableSemantic::temperature();
inline constexpr VariableSemantic VariableSemantic::Scalar = VariableSemantic::scalar();
inline constexpr VariableSemantic VariableSemantic::Custom = VariableSemantic::custom();

/// Public role spelling used by generic consumers.  It is a structured semantic, not an enum ABI.
using VariableRole = VariableSemantic;

/// Forward declaration: VariableSet::index_of(const std::string&) resolves a canonical role NAME via
/// role_from_name (defined below) before matching a user-defined role label.
inline VariableSemantic role_from_name(const std::string& s);

/// A variable: name, physical role, component index in the state.
struct Variable {
  std::string name;
  VariableRole role;
  int component;
};

/// A model's variable set: kind (cons/prim), names, size, canonical `roles` (optional, parallel to
/// `names`; absent -> Custom), and `user_roles` (optional string labels parallel to `names`, for
/// components whose role is OUTSIDE the canonical enum). Existing calls `{kind, names, size}` and
/// `{kind, names, size, roles}` stay valid (user_roles empty). index_of(role) gives the index of the
/// component carrying that role (-1 if absent).
struct VariableSet {
  VariableKind kind;
  std::vector<std::string> names;
  int size;
  std::vector<VariableSemantic> roles{};  ///< parallel to `names`; empty = roles not provided
  std::vector<std::string> user_roles{};  ///< parallel to `names`; per-component user-defined role
                                          ///< label (Custom role); empty entry = canonical role

  /// Index of the component carrying @p role (first occurrence), -1 if absent.
  int index_of(VariableSemantic role) const {
    for (int i = 0; i < static_cast<int>(roles.size()); ++i)
      if (roles[i] == role)
        return i;
    return -1;
  }
  /// Index of the component carrying @p role addressed BY NAME: a canonical role name
  /// (role_from_name) first, else a user-defined role label (user_roles). -1 if absent. Resolving a
  /// user label by string removes the first-occurrence ambiguity of several `Custom` components. An
  /// EMPTY @p role is never a valid target (it would otherwise match the empty user_roles slot of a
  /// canonical component on a mixed block) and returns -1.
  int index_of(const std::string& role) const {
    if (role.empty())
      return -1;
    const VariableSemantic r = role_from_name(role);
    if (r != VariableSemantic::Custom)
      return index_of(r);
    for (int i = 0; i < static_cast<int>(user_roles.size()); ++i)
      if (user_roles[i] == role)
        return i;
    return -1;
  }
  /// Full descriptor of component @p i (Custom role if not provided).
  Variable at(int i) const {
    return {names[i], i < static_cast<int>(roles.size()) ? roles[i] : VariableSemantic::Custom, i};
  }
};

/// Human-readable name of a role (introspection, Python binding). Stable: used as a key on the
/// application side.
inline std::string role_name(VariableSemantic role) {
  role.validate();
  switch (role.kind) {
    case VariableRoleKind::Density:
      return "density";
    case VariableRoleKind::Momentum:
      return "momentum:" + std::to_string(role.axis);
    case VariableRoleKind::Energy:
      return "energy";
    case VariableRoleKind::Velocity:
      return "velocity:" + std::to_string(role.axis);
    case VariableRoleKind::Pressure:
      return "pressure";
    case VariableRoleKind::Temperature:
      return "temperature";
    case VariableRoleKind::Scalar:
      return "scalar";
    case VariableRoleKind::Custom:
      return "custom";
    case VariableRoleKind::Axial:
      return "axial:" + std::to_string(role.axis);
  }
  return "custom";
}

/// Inverse of role_name: physical role from its stable name (Custom if unknown). Used to
/// reconstruct a VariableSet with roles from TEXT metadata (e.g. the string carried by a compiled /
/// dynamic .so: the extern "C" ABI carries only strings, not the enum).
inline VariableSemantic role_from_name(const std::string& s) {
  if (s == "density")
    return VariableSemantic::Density;
  if (s == "energy")
    return VariableSemantic::Energy;
  if (s == "pressure")
    return VariableSemantic::Pressure;
  if (s == "temperature")
    return VariableSemantic::Temperature;
  if (s == "scalar")
    return VariableSemantic::Scalar;
  const auto parse_axis = [&s](const char* prefix) -> int {
    const std::string_view head(prefix);
    if (!s.starts_with(head))
      return -1;
    const std::string axis_text = s.substr(head.size());
    if (axis_text.empty())
      return -1;
    int axis = 0;
    for (const char character : axis_text) {
      if (character < '0' || character > '9')
        return -1;
      axis = axis * 10 + (character - '0');
    }
    return axis;
  };
  if (const int axis = parse_axis("momentum:"); axis >= 0)
    return VariableSemantic::momentum(axis);
  if (const int axis = parse_axis("velocity:"); axis >= 0)
    return VariableSemantic::velocity(axis);
  if (const int axis = parse_axis("axial:"); axis >= 0)
    return VariableSemantic::axial(axis);
  return VariableSemantic::Custom;
}

/// CSV of a VariableSet's names (separator ','). Building block of the TEXT metadata that a generated
/// .so exposes: the extern "C" ABI does not carry a C++ object, so we serialize to a string.
inline std::string names_csv(const VariableSet& vs) {
  std::string s;
  for (std::size_t i = 0; i < vs.names.size(); ++i) {
    if (i)
      s += ',';
    s += vs.names[i];
  }
  return s;
}

/// CSV of a VariableSet's roles (role_name, separator ','). A component carrying a user-defined role
/// label (user_roles, Custom role) emits its LABEL instead of "custom", so the user role round-trips
/// through the .so ABI (parse_roles_into is the inverse). EMPTY if the model does not provide its
/// roles. Final compiled artifacts must provide one role entry per component; an empty result is
/// therefore valid only for an empty variable set and is rejected at every executable load boundary.
inline std::string roles_csv(const VariableSet& vs) {
  std::string s;
  for (std::size_t i = 0; i < vs.roles.size(); ++i) {
    if (i)
      s += ',';
    if (i < vs.user_roles.size() && !vs.user_roles[i].empty())
      s += vs.user_roles[i];
    else
      s += role_name(vs.roles[i]);
  }
  return s;
}

/// Inverse of roles_csv: fill @p vs.roles (and @p vs.user_roles for any NON-canonical token) from a
/// roles CSV. A canonical token (role_from_name) maps to its enum with an empty user label; a
/// non-canonical token maps to VariableRole::Custom keeping the token as its user-role label, so a
/// user role survives the .so ABI round-trip. user_roles stays EMPTY when every token is canonical
/// for an all-canonical set. Empty @p csv leaves both empty so the loader can diagnose missing
/// metadata before installing the block.
inline void parse_roles_into(VariableSet& vs, const std::string& csv) {
  if (csv.empty())
    return;
  std::vector<std::string> labels;
  bool any_user = false;
  std::size_t start = 0;
  for (;;) {
    const std::size_t comma = csv.find(',', start);
    const std::string tok =
        csv.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    const VariableSemantic r = role_from_name(tok);
    vs.roles.push_back(r);
    const bool is_user = (r == VariableSemantic::Custom && tok != role_name(VariableSemantic::Custom));
    labels.push_back(is_user ? tok : std::string());
    any_user = any_user || is_user;
    if (comma == std::string::npos)
      break;
    start = comma + 1;
  }
  if (any_user)
    vs.user_roles = std::move(labels);
}

/// Resolve one REQUIRED physical role from a complete typed variable descriptor. Missing or partial
/// role metadata is an invalid executable contract: a canonical component-index guess can silently
/// apply physics to another quantity after a state reordering. The caller names the subject in the
/// diagnostic (a block, model, or operator).
inline int require_role_index(const VariableSet& vs, VariableSemantic role, const char* origin,
                              const std::string& subject) {
  if (vs.size < 0 || static_cast<std::size_t>(vs.size) != vs.names.size() ||
      vs.roles.size() != vs.names.size() ||
      (!vs.user_roles.empty() && vs.user_roles.size() != vs.names.size()))
    throw std::runtime_error(std::string(origin) + " : '" + subject +
                             "' must declare exactly one role per state component (names=" +
                             std::to_string(vs.names.size()) +
                             ", roles=" + std::to_string(vs.roles.size()) + ")");
  int resolved = -1;
  for (int component = 0; component < vs.size; ++component) {
    if (vs.roles[static_cast<std::size_t>(component)] != role)
      continue;
    if (resolved >= 0)
      throw std::runtime_error(std::string(origin) + " : '" + subject +
                               "' declares the required role '" + role_name(role) +
                               "' more than once");
    resolved = component;
  }
  if (resolved >= 0)
    return resolved;
  throw std::runtime_error(std::string(origin) + " : '" + subject + "' declares roles (" +
                           roles_csv(vs) + ") but not the required role '" + role_name(role) + "'");
}

/// Validate that every declared role is meaningful for one exact native specialization.
template <int Dim>
inline void validate_variable_semantics(const VariableSet& vs, const char* origin,
                                        const std::string& subject) {
  if (vs.roles.size() != vs.names.size())
    throw std::runtime_error(std::string(origin) + " : '" + subject +
                             "' has incomplete variable semantic metadata");
  for (const VariableSemantic role : vs.roles)
    role.template validate_for_dimension<Dim>();
}

/// A model's "names" metadata: "cons_csv|prim_csv" (separator '|' between the two sets). Read
/// as-is by the consumer (System) via the mandatory current-ABI symbol pops_compiled_var_names.
template <class Model>
std::string var_names_meta() {
  return names_csv(Model::conservative_vars()) + "|" + names_csv(Model::primitive_vars());
}

/// A model's "roles" metadata: "cons_roles_csv|prim_roles_csv". Executable artifacts require both
/// sides to be total and parallel to their corresponding names metadata.
template <class Model>
std::string roles_meta() {
  return roles_csv(Model::conservative_vars()) + "|" + roles_csv(Model::primitive_vars());
}

/// Old name (compat): VariableSet used to be `Variables`. Kept for existing and generated code.
using Variables = VariableSet;

}  // namespace pops

/// Exports the mandatory current-ABI "names + roles" metadata of a .so block via extern "C"
/// symbols read by the System loader. Shared by both generated backends.
/// @p MODEL = type of the model (carries conservative_vars / primitive_vars).
#define POPS_EXPORT_BLOCK_METADATA(MODEL)                       \
  extern "C" const char* pops_compiled_var_names() {            \
    static const std::string s = pops::var_names_meta<MODEL>(); \
    return s.c_str();                                           \
  }                                                             \
  extern "C" const char* pops_compiled_roles() {                \
    static const std::string s = pops::roles_meta<MODEL>();     \
    return s.c_str();                                           \
  }

/// Exports the block's gamma (adiabatic index) via the optional symbol pops_compiled_gamma, read by
/// the System's inter-species couplings (collision, thermal exchange, T_e). EMITTED ONLY if the
/// model declares a gamma: otherwise the symbol stays absent and the System keeps its default 1.4.
#define POPS_EXPORT_BLOCK_GAMMA(GAMMA)      \
  extern "C" double pops_compiled_gamma() { \
    return (GAMMA);                         \
  }
