#pragma once

#include <pops/runtime/config/generated_component_catalog.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>

/// @file
/// @brief SINGLE registry of builtin MODEL BRICK tags (transport / source / elliptic): the shared
///        source of truth for every model dispatch -- detail::dispatch_transport / dispatch_source /
///        dispatch_elliptic (model_factory.hpp), the polar dispatch (block_builder_polar.hpp) and the
///        per-transport seams in python/system.cpp / python/amr_system.cpp.
///
/// Counterpart of dispatch_tags.hpp (limiters + Riemann fluxes) for the MODEL axis. Before this
/// header each model dispatch inlined its OWN tag list inside its rejection message: the transport
/// list `(exb|compressible|isothermal)` was repeated in FIVE sites and the source / elliptic lists
/// once each. Adding a builtin brick then meant editing every message by hand, with silent drift if
/// one was forgotten. Here: ONE kTransports / kSources / kElliptics table plus ONE set of CSV /
/// choices / validation helpers (messages STRICTLY identical to the old inline throws), so a new
/// builtin brick is ONE table row plus its compile-time dispatch case.
///
/// LIGHT by design (no brick TYPE dependency: only names + small integer capability metadata): it
/// stays included early and cost-free, exactly like dispatch_tags.hpp, and can drive
/// validate_model_spec without pulling physics/bricks.hpp. The string->TYPE dispatch stays an
/// `if` / `if constexpr` per call-site (the brick types are COMPILE-TIME, not tabulable); the
/// registry is the source of truth for the VALID TAG LIST + the rejection messages, while a non-drift
/// static_assert on the model_factory.hpp side (which sees BOTH the table and the brick types) locks
/// the n_vars column against the real bricks.
///
/// This is the "registry of builtin bricks" layer of ADC-331, kept distinct from the
/// ModelSpec / capabilities surface (model_spec.hpp) and from the compiled / native backend (the
/// dispatch, which MAY close some combinations -- e.g. `compressible` is not wired in polar geometry,
/// a fluid source needs a transport with >= 3 variables). The capability columns (polar_ok, min_vars)
/// document those supported-vs-not-routed combinations as DATA, read directly by the tests and docs.

namespace pops {

namespace detail {
/// Joins the @c name field of a tag table into "a<sep>b<sep>..." (optionally each name single-quoted).
/// Used to build the rejection-message tag lists from the SINGLE table (no inline duplication).
template <class TagT, std::size_t N>
std::string join_tag_names(const TagT (&tbl)[N], const char* sep, bool quote) {
  std::string out;
  for (std::size_t i = 0; i < N; ++i) {
    if (i)
      out += sep;
    if (quote)
      out += '\'';
    out += tbl[i].name;
    if (quote)
      out += '\'';
  }
  return out;
}
}  // namespace detail

/// Pipe list of transport tags ("exb|compressible|isothermal"), as used in the dispatch rejection
/// messages. @p polar restricts to the polar-wired subset ("exb|isothermal") for the polar message.
inline std::string transport_tags_csv(bool polar = false) {
  std::string out;
  for (const TransportTag& t : kTransports) {
    if (polar && !t.polar_ok)
      continue;
    if (!out.empty())
      out += '|';
    out += t.name;
  }
  return out;
}

/// Pipe list of source / elliptic tags (e.g. "charge|background|gravity").
inline std::string source_tags_csv() {
  return detail::join_tag_names(kSources, "|", false);
}
inline std::string elliptic_tags_csv() {
  return detail::join_tag_names(kElliptics, "|", false);
}

/// Quoted " | "-separated choices (e.g. "'exb' | 'compressible' | 'isothermal'"), as used in the
/// completeness messages of validate_model_spec.
inline std::string transport_choices() {
  return detail::join_tag_names(kTransports, " | ", true);
}
inline std::string source_choices() {
  return detail::join_tag_names(kSources, " | ", true);
}
inline std::string elliptic_choices() {
  return detail::join_tag_names(kElliptics, " | ", true);
}

/// Membership against the builtin tables.
inline bool is_transport(const std::string& tag) {
  for (const TransportTag& t : kTransports)
    if (tag == t.name)
      return true;
  return false;
}
inline bool is_source(const std::string& tag) {
  for (const SourceTag& t : kSources)
    if (tag == t.name)
      return true;
  return false;
}
inline bool is_elliptic(const std::string& tag) {
  for (const EllipticTag& t : kElliptics)
    if (tag == t.name)
      return true;
  return false;
}

/// Conservative-variable count of a transport tag (source of truth for the static_assert below), or
/// -1 if unknown. @c _ct is the COMPILE-TIME variant used by the non-drift static_assert in
/// model_factory.hpp (that TU sees both this table and the brick types). The char compare is inlined
/// to keep this header self-contained (no shared ct_str_eq -> no ODR coupling with dispatch_tags.hpp).
inline int transport_n_vars(const std::string& tag) {
  for (const TransportTag& t : kTransports)
    if (tag == t.name)
      return t.n_vars;
  return -1;
}
constexpr int transport_n_vars_ct(const char* name) {
  for (const TransportTag& t : kTransports) {
    const char* a = name;
    const char* b = t.name;
    while (*a != '\0' && *b != '\0' && *a == *b) {
      ++a;
      ++b;
    }
    if (*a == '\0' && *b == '\0')
      return t.n_vars;
  }
  return -1;
}

/// Rejection message for an unknown transport / elliptic tag, BYTE-IDENTICAL to the historical inline
/// throws (the tag list now comes from the SINGLE table). No context prefix: every model dispatch
/// site shares this exact message (unlike the limiter / flux validators of dispatch_tags.hpp, which
/// carry a per-call-site context because the same tag appears under System / AMR / polar prefixes).
inline std::string unknown_transport_msg(const std::string& tag) {
  return "unknown transport '" + tag + "' (" + transport_tags_csv() + ")";
}
inline std::string unknown_elliptic_msg(const std::string& tag) {
  return "unknown elliptic '" + tag + "' (" + elliptic_tags_csv() + ")";
}

/// Validates a transport / elliptic tag against the builtin registry. Throws the historical message
/// on an unknown tag. The string->TYPE routing stays a per-site `if` chain (compile-time bricks);
/// these validators give the SHARED rejection so the dispatch's final throw becomes a defense-in-depth
/// registry/dispatch-consistency guard (never reached once the tag is accepted here). The validate-first
/// invariant is specific to the model_factory dispatch; the per-transport binding seams
/// (python/system.cpp, python/amr_system.cpp) instead reuse unknown_transport_msg as their if/else tail
/// rejection (same message, single-sourced), since they route to per-TU builds rather than to types.
inline void validate_transport(const std::string& tag) {
  if (!is_transport(tag))
    throw std::runtime_error(unknown_transport_msg(tag));
}
inline void validate_elliptic(const std::string& tag) {
  if (!is_elliptic(tag))
    throw std::runtime_error(unknown_elliptic_msg(tag));
}

}  // namespace pops
