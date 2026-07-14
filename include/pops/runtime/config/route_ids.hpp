#pragma once

#include <pops/runtime/config/generated_component_catalog.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace pops {

namespace detail {

template <std::size_t N>
inline int parse_route_index(const RouteInfo (&table)[N], RouteFamily family,
                             const std::string& token, const char* context) {
  for (const RouteInfo& route : table)
    if (token == route.token)
      return route.index;
  std::string valid;
  for (const RouteInfo& route : table) {
    if (!valid.empty())
      valid += '|';
    valid += route.token;
  }
  throw std::runtime_error(std::string(context) + ": unknown " + route_family_name(family) +
                           " route '" + token + "' (valid: " + valid +
                           "); typed routes never fall back to a default");
}

template <std::size_t N, class Id>
inline const RouteInfo& checked_route_info(const RouteInfo (&table)[N], Id id,
                                           RouteFamily family) {
  const int index = static_cast<int>(id);
  if (index < 0 || index >= static_cast<int>(N) || table[index].index != index)
    throw std::runtime_error(std::string("routes: unknown ") + route_family_name(family) +
                             " wire id " + std::to_string(index));
  return table[index];
}

template <std::size_t N>
constexpr bool route_indices_sequential(const RouteInfo (&table)[N]) {
  for (std::size_t index = 0; index < N; ++index)
    if (table[index].index != static_cast<int>(index))
      return false;
  return true;
}

}  // namespace detail

#define POPS_DEFINE_ROUTE_ACCESSORS(Name, Id, Table, Family)                                  \
  inline Id parse_##Name##_route(const std::string& token, const char* context = "routes") {  \
    return static_cast<Id>(detail::parse_route_index(Table, RouteFamily::Family, token, context)); \
  }                                                                                            \
  inline const RouteInfo& route_info(Id id) {                                                  \
    return detail::checked_route_info(Table, id, RouteFamily::Family);                         \
  }                                                                                            \
  inline const char* route_token(Id id) { return route_info(id).token; }                       \
  static_assert(detail::route_indices_sequential(Table), #Name " route index drift")

#include <pops/runtime/config/generated_route_accessors.inc>

#undef POPS_DEFINE_ROUTE_ACCESSORS

inline constexpr int kReservedRouteWireId255 = 255;

inline std::string route_registry_signature() { return kRouteRegistrySignature; }

inline void verify_route_manifest(const std::string& embedded, const char* context) {
  if (embedded.empty())
    throw std::runtime_error(std::string(context) +
                             ": compiled artifact is missing the required route registry signature");
  const std::string current = route_registry_signature();
  if (embedded == current)
    return;
  throw std::runtime_error(std::string(context) +
                           ": stale compiled artifact -- route registry mismatch (built against '" +
                           embedded + "', current '" + current + "')");
}

}  // namespace pops
