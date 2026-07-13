// pops brick_catalog.hpp: the generated builtin component catalog (ADC-679). The schema catalog is
// the only declaration; this test locks the light C++ inspection behavior without Kokkos or MPI.
//   (1) catalog_entry lookup roundtrip for ALL 11 rows (found, category + id + route_index match) ;
//   (2) a spot native_entry ("pops::ExBVelocity") and an unknown id -> nullptr ;
//   (3) catalog_csv(category) matches the registry csv helpers over the CANONICAL set ;
//   (4) brick_catalog_json() lists every id and parses as the same minimal grammar external_brick
//       uses (string checks: "{\"bricks\":[", each "\"id\":\"<id>\"", the extra columns present) ;
//   (5) generated descriptor views agree row for row.

#include <gtest/gtest.h>

#include <pops/runtime/builders/factory/brick_catalog.hpp>
#include <pops/runtime/dynamic/model_registry.hpp>

#include <cstddef>
#include <string>

using namespace pops;

namespace {

bool contains(const std::string& hay, const char* needle) {
  return hay.find(needle) != std::string::npos;
}

}  // namespace

TEST(BrickCatalog, EntryRoundTripsAllElevenRows) {
  bool all_found = true;
  int n = 0;
  for (const BrickCatalogEntry& e : kBrickCatalog) {
    ++n;
    const BrickCatalogEntry* got = catalog_entry(e.category, e.id);
    all_found = all_found && got == &e && got->route_index == e.route_index;
  }
  EXPECT_TRUE(n == 11) << "kBrickCatalog compte 11 lignes (3 transports + 5 sources + 3 elliptics)";
  EXPECT_TRUE(all_found)
      << "catalog_entry(category, id) retrouve chaque ligne (identite + route_index)";
}

TEST(BrickCatalog, NativeEntryKnownAndUnknownIdReturnsNullptr) {
  EXPECT_TRUE(catalog_entry("transport", "exb") != nullptr &&
              std::string(catalog_entry("transport", "exb")->native_entry) == "pops::ExBVelocity")
      << "catalog_entry('transport','exb')->native_entry == 'pops::ExBVelocity'";
  EXPECT_TRUE(std::string(catalog_entry("source", "potential")->native_entry) ==
                  "pops::PotentialForce" &&
              std::string(catalog_entry("source", "potential")->parameters) == "qom")
      << "catalog_entry('source','potential') exposes generated parameter qom";
  EXPECT_TRUE(std::string(catalog_entry("elliptic", "background")->parameters) == "alpha,n0")
      << "catalog_entry('elliptic','background') exposes generated parameters";
  EXPECT_TRUE(catalog_entry("transport", "bogus") == nullptr) << "id inconnu -> nullptr";
  EXPECT_TRUE(catalog_entry("source", "lorentz") == nullptr)
      << "alias source 'lorentz' n'est PAS une ligne de catalog (parse-only) -> nullptr";
  EXPECT_TRUE(catalog_entry("bogus_cat", "exb") == nullptr) << "category inconnue -> nullptr";
}

TEST(BrickCatalog, CatalogCsvMatchesCanonicalIdListPerCategory) {
  EXPECT_TRUE(catalog_csv("transport") == "exb|compressible|isothermal")
      << "catalog_csv('transport') == 'exb|compressible|isothermal' (== transport_tags_csv)";
  EXPECT_TRUE(catalog_csv("transport") == transport_tags_csv())
      << "catalog_csv('transport') == model_registry transport_tags_csv()";
  EXPECT_TRUE(catalog_csv("elliptic") == "charge|background|gravity")
      << "catalog_csv('elliptic') == 'charge|background|gravity' (== elliptic_tags_csv)";
  EXPECT_TRUE(catalog_csv("elliptic") == elliptic_tags_csv())
      << "catalog_csv('elliptic') == model_registry elliptic_tags_csv()";
  // The source csv is the CANONICAL set (no aliases): a strict subset of source_tags_csv().
  EXPECT_TRUE(catalog_csv("source") == "none|potential|gravity|magnetic|potential_magnetic")
      << "catalog_csv('source') == canonique 'none|potential|gravity|magnetic|potential_magnetic'";
}

TEST(BrickCatalog, JsonListsEveryIdWithExternalBrickGrammar) {
  const std::string j = brick_catalog_json();
  bool ok = contains(j, "{\"bricks\":[");
  for (const BrickCatalogEntry& e : kBrickCatalog) {
    const std::string id_field = std::string("\"id\":\"") + e.id + "\"";
    ok = ok && contains(j, id_field.c_str());
  }
  EXPECT_TRUE(ok) << "brick_catalog_json() commence par {\"bricks\":[ et liste chaque \"id\":\"<id>\"";
  EXPECT_TRUE(contains(j, "\"catalog_digest\":\"") &&
              contains(j, "\"catalog_semantic_digest\":\"") &&
              contains(j, "\"category\":\"transport\"") &&
              contains(j, "\"requirements\":[]") &&
              contains(j, "\"limitations\":[") &&
              contains(j, "\"native_entry\":\"pops::ExBVelocity\"") &&
              contains(j, "\"parameters\":[\"cs2\",\"vacuum_floor\"]") &&
              contains(j, "\"route_index\":") &&
              contains(j, "\"n_vars\":") && contains(j, "\"polar_ok\":true"))
      << "brick_catalog_json() exposes structured generated component facts";
}

TEST(BrickCatalog, MirrorsRegistryAndRouteTablesRowForRow) {
  // Transports : catalog == kTransports (name/n_vars/polar_ok/summary) == kTransportRoutes (token/
  // native/req/lim). Re-assert dynamique des static_assert transports_mirror() du header.
  {
    bool ok = true;
    std::size_t i = 0;
    for (const BrickCatalogEntry& e : kBrickCatalog) {
      if (std::string(e.category) != "transport")
        continue;
      const TransportTag& t = kTransports[i];
      const RouteInfo& r = kTransportRoutes[i];
      ok = ok && std::string(e.id) == t.name && e.n_vars == t.n_vars && e.polar_ok == t.polar_ok &&
           std::string(e.summary) == t.summary && e.route_index == r.index &&
           std::string(e.id) == r.token && std::string(e.native_entry) == r.native_entry &&
           std::string(e.requirements) == r.requirements &&
           std::string(e.limitations) == r.limitations;
      ++i;
    }
    EXPECT_TRUE(i == 3 && ok) << "transport : catalog == kTransports == kTransportRoutes (3 lignes)";
  }
  // Sources : catalog == kSources aux positions CANONIQUES (les 2 alias 4 et 6 sont sautes) ==
  // kSourceRoutes. Re-assert dynamique de sources_mirror().
  {
    const std::size_t canonical_pos[] = {0, 1, 2, 3, 5};
    bool ok = true;
    std::size_t i = 0;
    for (const BrickCatalogEntry& e : kBrickCatalog) {
      if (std::string(e.category) != "source")
        continue;
      const SourceTag& s = kSources[canonical_pos[i]];
      const RouteInfo& r = kSourceRoutes[i];
      ok = ok && std::string(e.id) == s.name && e.n_vars == s.min_vars &&
           std::string(e.summary) == s.summary && e.route_index == r.index &&
           std::string(e.id) == r.token && std::string(e.native_entry) == r.native_entry &&
           std::string(e.requirements) == r.requirements &&
           std::string(e.limitations) == r.limitations;
      ++i;
    }
    EXPECT_TRUE(i == 5 && ok) << "source : catalog == kSources canoniques == kSourceRoutes (5 lignes)";
  }
  // Elliptics : catalog == kElliptics (name/summary, n_vars == -1) == kEllipticRoutes.
  {
    bool ok = true;
    std::size_t i = 0;
    for (const BrickCatalogEntry& e : kBrickCatalog) {
      if (std::string(e.category) != "elliptic")
        continue;
      const EllipticTag& el = kElliptics[i];
      const RouteInfo& r = kEllipticRoutes[i];
      ok = ok && std::string(e.id) == el.name && e.n_vars == -1 && !e.polar_ok &&
           std::string(e.summary) == el.summary && e.route_index == r.index &&
           std::string(e.id) == r.token && std::string(e.native_entry) == r.native_entry &&
           std::string(e.requirements) == r.requirements &&
           std::string(e.limitations) == r.limitations;
      ++i;
    }
    EXPECT_TRUE(i == 3 && ok) << "elliptic : catalog == kElliptics == kEllipticRoutes (3 lignes)";
  }
}
