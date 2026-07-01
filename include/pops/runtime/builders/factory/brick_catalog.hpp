#pragma once

#include <pops/runtime/config/route_ids.hpp>  // k*Routes + parse_*_route + RouteInfo (ADC-584)
#include <pops/runtime/dynamic/model_registry.hpp>  // kTransports/kSources/kElliptics (ADC-331)

#include <cstddef>
#include <string>

/// @file
/// @brief The BUILTIN native brick catalog (ADC-586): ONE declarative row per canonical model brick
///        (3 transports + 5 canonical sources + 3 elliptics), the inspectable counterpart of the
///        external-brick catalog (program/external_brick.hpp) for the bricks the core ships.
///
/// Before this header a native brick's identity was scattered: its tag lived in model_registry.hpp
/// (name / n_vars / polar_ok / min_vars / summary), its typed route + native entry + requirements +
/// limitations in route_ids.hpp (the k*Routes tables), and its ModelSpec constructor parameters were
/// implicit in the model_factory.hpp dispatch (`ExBVelocity{Real(m.B0)}` -> "B0", ...). No single
/// place answered "what native bricks exist, with which identity, capabilities and construction
/// contract". This catalog is that place: a light, dependency-free table (no brick TYPE include, like
/// route_ids.hpp and model_registry.hpp) that MIRRORS the two existing single sources row for row and
/// adds the two facts they did not carry -- the CSV of ModelSpec constructor @c params and the route
/// enum @c route_index -- so the codegen / bindings / reports can reference a brick by its canonical id
/// and read its full contract without pulling physics/bricks.hpp.
///
/// It is NOT a new source of truth: static_asserts lock every column against BOTH model_registry.hpp
/// (name / n_vars / polar_ok / min_vars) and route_ids.hpp (token / route index), using the compile-time
/// ct_str_eq pattern (dispatch_tags.hpp). A brick edited in ONE table -- a renamed tag, a moved route,
/// a changed n_vars -- fails THIS build until every mirror agrees, exactly like the route/registry
/// locks of route_ids.hpp. The alias source spellings (lorentz, potential_lorentz) stay PARSE-ONLY in
/// route_ids.hpp (parse_source_route resolves them); the catalog lists only the 5 canonical sources,
/// so each catalog row is one canonical brick.
///
/// The string->TYPE dispatch itself stays a compile-time `switch` per family in model_factory.hpp (the
/// brick types are template parameters, not tabulable); this catalog is the DATA surface -- the valid
/// id set, the capabilities/requirements and the construction params -- read by the tests, the Python
/// mirror (python/pops/runtime/brick_catalog.py) and the codegen manifest.

namespace pops {

/// One builtin native brick catalog row. Every field MIRRORS an existing single source (locked by the
/// static_asserts below); @c params and @c route_index are the two facts the catalog adds.
struct BrickCatalogEntry {
  const char* id;        ///< canonical wire token (== registry name == route token)
  const char* category;  ///< "transport" | "source" | "elliptic"
  int route_index;       ///< enumerator value in the family route enum (TransportRouteId, ...)
  const char*
      native_entry;    ///< native C++ entry point, e.g. "pops::ExBVelocity" (== route native_entry)
  const char* params;  ///< CSV of ModelSpec constructor params ("B0" / "cs2,vacuum_floor" / "")
  int n_vars;          ///< transport: ::n_vars ; source: min_vars ; elliptic: -1 (no var count)
  bool polar_ok;       ///< wired in polar geometry (transport only; false for source/elliptic)
  const char* requirements;  ///< CSV of route requirements (== route requirements; documentary)
  const char* capabilities;  ///< CSV of provided capabilities (== route limitations; documentary)
  const char* summary;       ///< one-line human summary (== registry summary)
};

/// SINGLE catalog of the builtin native bricks (order = the registry / route order within each
/// family: 3 transports, then the 5 canonical sources, then 3 elliptics). Values are copied from
/// kTransports/kSources/kElliptics (name/n_vars/polar_ok/min_vars/summary) and kTransportRoutes/
/// kSourceRoutes/kEllipticRoutes (native_entry/requirements/limitations); the static_asserts below
/// prove the copies stay faithful. @c params names the ModelSpec fields dispatch_transport/
/// dispatch_source/dispatch_elliptic read to construct the brick (the catalog's one original column).
inline constexpr BrickCatalogEntry kBrickCatalog[] = {
    // --- transports (n_vars = TransportTag::n_vars ; polar_ok = TransportTag::polar_ok) ---
    {"exb", "transport", 0, "pops::ExBVelocity", "B0", 1, true, "",
     "scalar (1 var); no fluid source", "scalar ExB drift advection, v = (-d_y phi, d_x phi) / B0"},
    {"compressible", "transport", 1, "pops::CompressibleFlux", "gamma", 4, false, "",
     "polar geometry not wired", "compressible Euler, 4 var (rho, rho u, rho v, E)"},
    {"isothermal", "transport", 2, "pops::IsothermalFlux", "cs2,vacuum_floor", 3, true, "", "",
     "isothermal Euler, 3 var (rho, rho u, rho v)"},
    // --- sources (n_vars = SourceTag::min_vars ; polar_ok = false, N/A for a source) ---
    {"none", "source", 0, "pops::NoSource", "", 1, false, "", "", "neutral: no source term"},
    {"potential", "source", 1, "pops::PotentialForce", "qom", 3, false,
     "fluid transport (>= 3 vars)", "", "(q/m) rho E electrostatic force"},
    {"gravity", "source", 2, "pops::GravityForce", "", 3, false, "fluid transport (>= 3 vars)", "",
     "rho g gravity force"},
    {"magnetic", "source", 3, "pops::MagneticLorentzForce", "qom", 3, false,
     "fluid transport (>= 3 vars),aux B_z channel",
     "explicit regime (stiff regime -> condensed Schur stage)",
     "q v x B_z magnetized Lorentz force (explicit regime)"},
    {"potential_magnetic", "source", 4,
     "pops::CompositeSource<PotentialForce, MagneticLorentzForce>", "qom", 3, false,
     "fluid transport (>= 3 vars),aux B_z channel", "", "electrostatic + Lorentz, summed"},
    // --- elliptics (no var-count axis -> n_vars = -1 ; polar_ok = false) ---
    {"charge", "elliptic", 0, "pops::ChargeDensity", "q", -1, false, "", "",
     "rho - q : charge density (Poisson source)"},
    {"background", "elliptic", 1, "pops::BackgroundDensity", "alpha,n0", -1, false, "", "",
     "alpha (rho - n0) : neutralizing background"},
    {"gravity", "elliptic", 2, "pops::GravityCoupling", "sign,four_pi_G,rho0", -1, false, "", "",
     "sign * 4 pi G (rho - rho0) : gravitational coupling"},
};

// --- Non-drift locks: the catalog MIRRORS model_registry.hpp AND route_ids.hpp row for row --------
namespace detail {

/// Count of catalog rows whose @c category matches @p category (compile-time).
constexpr std::size_t catalog_count(const char* category) {
  std::size_t n = 0;
  for (const BrickCatalogEntry& e : kBrickCatalog)
    if (ct_str_eq(e.category, category))
      ++n;
  return n;
}

/// The @p k-th catalog row (0-based) of @p category (compile-time; k assumed in range).
constexpr const BrickCatalogEntry& catalog_row(const char* category, std::size_t k) {
  std::size_t seen = 0;
  for (const BrickCatalogEntry& e : kBrickCatalog) {
    if (ct_str_eq(e.category, category)) {
      if (seen == k)
        return e;
      ++seen;
    }
  }
  return kBrickCatalog[0];  // unreachable when k < catalog_count(category)
}

/// Every transport row matches kTransports (name/n_vars/polar_ok/summary) AND kTransportRoutes
/// (token/native_entry/requirements/limitations, by index) -- one edit fails the build until all agree.
constexpr bool transports_mirror() {
  if (catalog_count("transport") != sizeof(kTransports) / sizeof(kTransports[0]))
    return false;
  for (std::size_t i = 0; i < sizeof(kTransports) / sizeof(kTransports[0]); ++i) {
    const BrickCatalogEntry& e = catalog_row("transport", i);
    const TransportTag& t = kTransports[i];
    const RouteInfo& r = kTransportRoutes[i];
    if (!ct_str_eq(e.id, t.name) || e.n_vars != t.n_vars || e.polar_ok != t.polar_ok ||
        !ct_str_eq(e.summary, t.summary))
      return false;
    if (e.route_index != r.index || !ct_str_eq(e.id, r.token) ||
        !ct_str_eq(e.native_entry, r.native_entry) || !ct_str_eq(e.requirements, r.requirements) ||
        !ct_str_eq(e.capabilities, r.limitations))
      return false;
  }
  return true;
}

/// Every source row matches kSources (name/min_vars/summary) at the CANONICAL positions (kSources
/// interleaves 2 alias rows: magnetic->lorentz at 4, potential_magnetic->potential_lorentz at 6, both
/// skipped) AND kSourceRoutes (token/native_entry/requirements/limitations, by index).
constexpr bool sources_mirror() {
  const std::size_t n_routes = sizeof(kSourceRoutes) / sizeof(kSourceRoutes[0]);
  if (catalog_count("source") != n_routes)
    return false;
  // Canonical source name at registry position i skips the 2 alias rows (indices 4 and 6).
  const std::size_t canonical_pos[] = {0, 1, 2, 3, 5};
  for (std::size_t i = 0; i < n_routes; ++i) {
    const BrickCatalogEntry& e = catalog_row("source", i);
    const SourceTag& s = kSources[canonical_pos[i]];
    const RouteInfo& r = kSourceRoutes[i];
    if (!ct_str_eq(e.id, s.name) || e.n_vars != s.min_vars || !ct_str_eq(e.summary, s.summary))
      return false;
    if (e.route_index != r.index || !ct_str_eq(e.id, r.token) ||
        !ct_str_eq(e.native_entry, r.native_entry) || !ct_str_eq(e.requirements, r.requirements) ||
        !ct_str_eq(e.capabilities, r.limitations))
      return false;
  }
  return true;
}

/// Every elliptic row matches kElliptics (name/summary) AND kEllipticRoutes (token/native_entry/
/// requirements/limitations, by index). n_vars is -1 for every elliptic (no var-count axis).
constexpr bool elliptics_mirror() {
  if (catalog_count("elliptic") != sizeof(kElliptics) / sizeof(kElliptics[0]))
    return false;
  for (std::size_t i = 0; i < sizeof(kElliptics) / sizeof(kElliptics[0]); ++i) {
    const BrickCatalogEntry& e = catalog_row("elliptic", i);
    const EllipticTag& el = kElliptics[i];
    const RouteInfo& r = kEllipticRoutes[i];
    if (!ct_str_eq(e.id, el.name) || e.n_vars != -1 || e.polar_ok ||
        !ct_str_eq(e.summary, el.summary))
      return false;
    if (e.route_index != r.index || !ct_str_eq(e.id, r.token) ||
        !ct_str_eq(e.native_entry, r.native_entry) || !ct_str_eq(e.requirements, r.requirements) ||
        !ct_str_eq(e.capabilities, r.limitations))
      return false;
  }
  return true;
}

}  // namespace detail

static_assert(sizeof(kBrickCatalog) / sizeof(kBrickCatalog[0]) == 11,
              "brick catalog: 3 transports + 5 canonical sources + 3 elliptics = 11 rows");
static_assert(detail::transports_mirror(),
              "brick catalog drift: transport rows disagree with kTransports / kTransportRoutes");
static_assert(detail::sources_mirror(),
              "brick catalog drift: source rows disagree with kSources / kSourceRoutes");
static_assert(detail::elliptics_mirror(),
              "brick catalog drift: elliptic rows disagree with kElliptics / kEllipticRoutes");

// --- Lookup / inspection helpers --------------------------------------------------------------
/// The catalog row for (@p category, @p id), or nullptr if absent (never throws). @p id is the
/// canonical token; the alias source spellings are parse-only (route_ids.hpp), never catalog rows.
inline const BrickCatalogEntry* catalog_entry(const std::string& category, const std::string& id) {
  for (const BrickCatalogEntry& e : kBrickCatalog)
    if (category == e.category && id == e.id)
      return &e;
  return nullptr;
}

/// Pipe list of the canonical ids of @p category ("exb|compressible|isothermal", "none|potential|
/// gravity|magnetic|potential_magnetic", "charge|background|gravity"). Derived from the table, never
/// hand-written -- the counterpart of the model_registry.hpp *_tags_csv helpers over the CANONICAL set.
inline std::string catalog_csv(const std::string& category) {
  std::string out;
  for (const BrickCatalogEntry& e : kBrickCatalog) {
    if (category != e.category)
      continue;
    if (!out.empty())
      out += '|';
    out += e.id;
  }
  return out;
}

/// The catalog as the same minimal JSON shape external_brick.hpp exports
/// ({"bricks":[{"id","category","requirements","capabilities", ...}, ...]}), extended with the
/// catalog's extra columns (route_index / native_entry / params / n_vars / polar_ok / summary). The
/// ids/categories/tokens are ASCII (no user input), so the string building stays dependency-free (no
/// json_escape needed): a catalog row can never carry a structural or control character.
inline std::string brick_catalog_json() {
  auto b2s = [](bool b) { return b ? "true" : "false"; };
  std::string out = "{\"bricks\":[";
  bool first = true;
  for (const BrickCatalogEntry& e : kBrickCatalog) {
    if (!first)
      out += ',';
    first = false;
    out += "{\"id\":\"";
    out += e.id;
    out += "\",\"category\":\"";
    out += e.category;
    out += "\",\"route_index\":";
    out += std::to_string(e.route_index);
    out += ",\"native_entry\":\"";
    out += e.native_entry;
    out += "\",\"params\":\"";
    out += e.params;
    out += "\",\"n_vars\":";
    out += std::to_string(e.n_vars);
    out += ",\"polar_ok\":";
    out += b2s(e.polar_ok);
    out += ",\"requirements\":\"";
    out += e.requirements;
    out += "\",\"capabilities\":\"";
    out += e.capabilities;
    out += "\",\"summary\":\"";
    out += e.summary;
    out += "\"}";
  }
  out += "]}";
  return out;
}

}  // namespace pops
