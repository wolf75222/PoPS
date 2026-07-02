// pops brick_catalog.hpp : the BUILTIN native brick catalog (ADC-586). ONE declarative row per
// canonical model brick (3 transports + 5 canonical sources + 3 elliptics), the inspectable
// counterpart of the external-brick catalog. This test locks the PUBLIC contract of the light header
// WITHOUT Kokkos or MPI (strings + enums only, like test_route_ids):
//   (1) catalog_entry lookup roundtrip for ALL 11 rows (found, category + id + route_index match) ;
//   (2) a spot native_entry ("pops::ExBVelocity") and an unknown id -> nullptr ;
//   (3) catalog_csv(category) matches the registry csv helpers over the CANONICAL set ;
//   (4) brick_catalog_json() lists every id and parses as the same minimal grammar external_brick
//       uses (string checks: "{\"bricks\":[", each "\"id\":\"<id>\"", the extra columns present) ;
//   (5) the registry / catalog / route mirrors agree row for row (a DYNAMIC re-assert of the header's
//       compile-time transports_mirror / sources_mirror / elliptics_mirror static_asserts).

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/runtime/builders/factory/brick_catalog.hpp>

#include "test_harness.hpp"  // pops::test::Checker (style verbose), comme test_route_ids

#include <cstdio>
#include <string>

using namespace pops;

// Compteur d'echecs partage, style VERBOSE ([OK ]/[XX ] par ligne), comme test_route_ids.
static pops::test::Checker g_chk{pops::test::Checker::Style::Verbose};

static void chk(bool ok, const char* label) {
  g_chk(ok, label);
}

static bool contains(const std::string& hay, const char* needle) {
  return hay.find(needle) != std::string::npos;
}

static int pops_run_test_brick_catalog() {
  std::printf(
      "== catalog_entry roundtrip : les 11 lignes trouvees, category + id + route_index ==\n");
  {
    bool all_found = true;
    int n = 0;
    for (const BrickCatalogEntry& e : kBrickCatalog) {
      ++n;
      const BrickCatalogEntry* got = catalog_entry(e.category, e.id);
      all_found = all_found && got == &e && got->route_index == e.route_index;
    }
    chk(n == 11, "kBrickCatalog compte 11 lignes (3 transports + 5 sources + 3 elliptics)");
    chk(all_found, "catalog_entry(category, id) retrouve chaque ligne (identite + route_index)");
  }

  std::printf("== native_entry connu + id inconnu -> nullptr ==\n");
  chk(catalog_entry("transport", "exb") != nullptr &&
          std::string(catalog_entry("transport", "exb")->native_entry) == "pops::ExBVelocity",
      "catalog_entry('transport','exb')->native_entry == 'pops::ExBVelocity'");
  chk(std::string(catalog_entry("source", "potential")->native_entry) == "pops::PotentialForce" &&
          std::string(catalog_entry("source", "potential")->params) == "qom",
      "catalog_entry('source','potential') : native 'pops::PotentialForce', params 'qom'");
  chk(std::string(catalog_entry("elliptic", "background")->params) == "alpha,n0",
      "catalog_entry('elliptic','background')->params == 'alpha,n0'");
  chk(catalog_entry("transport", "bogus") == nullptr, "id inconnu -> nullptr");
  chk(catalog_entry("source", "lorentz") == nullptr,
      "alias source 'lorentz' n'est PAS une ligne de catalog (parse-only) -> nullptr");
  chk(catalog_entry("bogus_cat", "exb") == nullptr, "category inconnue -> nullptr");

  std::printf("== catalog_csv(category) == liste canonique des ids (derivee de la table) ==\n");
  chk(catalog_csv("transport") == "exb|compressible|isothermal",
      "catalog_csv('transport') == 'exb|compressible|isothermal' (== transport_tags_csv)");
  chk(catalog_csv("transport") == transport_tags_csv(),
      "catalog_csv('transport') == model_registry transport_tags_csv()");
  chk(catalog_csv("elliptic") == "charge|background|gravity",
      "catalog_csv('elliptic') == 'charge|background|gravity' (== elliptic_tags_csv)");
  chk(catalog_csv("elliptic") == elliptic_tags_csv(),
      "catalog_csv('elliptic') == model_registry elliptic_tags_csv()");
  // The source csv is the CANONICAL set (no aliases): a strict subset of source_tags_csv().
  chk(catalog_csv("source") == "none|potential|gravity|magnetic|potential_magnetic",
      "catalog_csv('source') == canonique 'none|potential|gravity|magnetic|potential_magnetic'");

  std::printf(
      "== brick_catalog_json : chaque id present, grammaire minimale d'external_brick ==\n");
  {
    const std::string j = brick_catalog_json();
    bool ok = contains(j, "{\"bricks\":[");
    for (const BrickCatalogEntry& e : kBrickCatalog) {
      const std::string id_field = std::string("\"id\":\"") + e.id + "\"";
      ok = ok && contains(j, id_field.c_str());
    }
    chk(ok, "brick_catalog_json() commence par {\"bricks\":[ et liste chaque \"id\":\"<id>\"");
    // Les colonnes de la forme minimale d'external_brick (id/category/requirements/capabilities) +
    // les colonnes supplementaires du catalog (route_index/native_entry/params/n_vars/polar_ok/summary).
    chk(contains(j, "\"category\":\"transport\"") && contains(j, "\"requirements\":\"") &&
            contains(j, "\"capabilities\":\"") &&
            contains(j, "\"native_entry\":\"pops::ExBVelocity\"") &&
            contains(j, "\"params\":\"cs2,vacuum_floor\"") && contains(j, "\"route_index\":") &&
            contains(j, "\"n_vars\":") && contains(j, "\"polar_ok\":true"),
        "brick_catalog_json() porte les champs external_brick + les colonnes catalog");
  }

  std::printf(
      "== miroirs registre/catalog/route : accord ligne par ligne (re-assert dynamique) ==\n");
  {
    // Transports : catalog == kTransports (name/n_vars/polar_ok/summary) == kTransportRoutes (token/
    // native/req/lim). Re-assert dynamique des static_assert transports_mirror() du header.
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
           std::string(e.capabilities) == r.limitations;
      ++i;
    }
    chk(i == 3 && ok, "transport : catalog == kTransports == kTransportRoutes (3 lignes)");
  }
  {
    // Sources : catalog == kSources aux positions CANONIQUES (les 2 alias 4 et 6 sont sautes) ==
    // kSourceRoutes. Re-assert dynamique de sources_mirror().
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
           std::string(e.capabilities) == r.limitations;
      ++i;
    }
    chk(i == 5 && ok, "source : catalog == kSources canoniques == kSourceRoutes (5 lignes)");
  }
  {
    // Elliptics : catalog == kElliptics (name/summary, n_vars == -1) == kEllipticRoutes.
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
           std::string(e.capabilities) == r.limitations;
      ++i;
    }
    chk(i == 3 && ok, "elliptic : catalog == kElliptics == kEllipticRoutes (3 lignes)");
  }

  std::printf("FAILS = %d\n", g_chk.fails());
  return g_chk.failed();
}

TEST(test_brick_catalog, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_brick_catalog, "test_brick_catalog"), 0);
}
