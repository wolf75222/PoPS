// Registre UNIQUE des tags de BRIQUES DE MODELE (include/pops/runtime/model_registry.hpp) : source de
// verite partagee par tous les dispatchs modele (detail::dispatch_transport / dispatch_source /
// dispatch_elliptic, le dispatch polaire, et les seams par-transport de python/system.cpp /
// python/amr_system.cpp). Pendant de test_dispatch_tags.cpp (limiteurs + flux) pour l'AXE MODELE.
// Ce test est VOLONTAIREMENT LEGER (il n'inclut QUE model_registry.hpp, aucun System / brique) : il
// verrouille
//   (1) la MATRICE d'appartenance is_transport / is_source / is_elliptic (acceptes + rejets),
//   (2) les helpers de message (transport_tags_csv / *_choices) BYTE-IDENTIQUES aux anciens throws
//       inline ("exb|compressible|isothermal", "'exb' | 'compressible' | 'isothermal'", ...),
//   (3) validate_transport / validate_elliptic : rejet explicite avec le fragment + le tag,
//   (4) les colonnes de capabilite (n_vars, polar_ok, min_vars) et leur variante compile-time
//       (transport_n_vars_ct, utilisee par le static_assert de non-derive de model_factory.hpp),
//   (5) GARDE DE PERIMETRE (ADC-331) : la registry ne contient QUE des briques GENERIQUES, jamais un
//       nom de scenario applicatif (le SET de tags est verrouille -> ajouter "diocotron" echoue ici).
// Le routage effectif (chaque tag builtin atteint bien une branche du dispatch) est verifie cote
// test_config_model_validation.cpp, qui lie la machinerie de dispatch.

#include <gtest/gtest.h>

#include <pops/runtime/dynamic/model_registry.hpp>

#include <stdexcept>
#include <string>

using namespace pops;

namespace {

// Renvoie true si fn() leve, et capture le message dans @p msg (vide sinon).
template <class Fn>
bool throws(Fn&& fn, std::string& msg) {
  try {
    fn();
    msg.clear();
    return false;
  } catch (const std::exception& e) {
    msg = e.what();
    return true;
  }
}

bool contains(const std::string& hay, const char* needle) {
  return hay.find(needle) != std::string::npos;
}

}  // namespace

TEST(test_model_registry, membership_matrix_accepts_and_rejects) {
  EXPECT_TRUE(is_transport("exb") && is_transport("compressible") && is_transport("isothermal"))
      << "is_transport accepte les trois transports builtin";
  EXPECT_TRUE(!is_transport("bogus") && !is_transport("") && !is_transport("EXB"))
      << "is_transport rejette inconnu / vide / casse";
  EXPECT_TRUE(is_source("none") && is_source("potential") && is_source("gravity") &&
              is_source("magnetic") && is_source("potential_magnetic"))
      << "is_source accepts the five canonical builtin source ids";
  EXPECT_TRUE(!is_source("lorentz") && !is_source("potential_lorentz") && !is_source("bogus") &&
              !is_source(""))
      << "parse-only aliases and unknown source ids are not catalog identities";
  EXPECT_TRUE(is_elliptic("charge") && is_elliptic("background") && is_elliptic("gravity"))
      << "is_elliptic accepte les trois elliptiques builtin";
  EXPECT_TRUE(!is_elliptic("bogus") && !is_elliptic("")) << "is_elliptic rejette inconnu / vide";
}

TEST(test_model_registry, message_helpers_are_byte_identical_to_legacy_throws) {
  EXPECT_EQ(transport_tags_csv(), "exb|compressible|isothermal")
      << "transport_tags_csv() == 'exb|compressible|isothermal' (message dispatch)";
  EXPECT_EQ(transport_tags_csv(/*polar=*/true), "exb|isothermal")
      << "transport_tags_csv(polar) == 'exb|isothermal' (sous-ensemble polaire = colonne polar_ok)";
  EXPECT_EQ(elliptic_tags_csv(), "charge|background|gravity")
      << "elliptic_tags_csv() == 'charge|background|gravity'";
  EXPECT_EQ(transport_choices(), "'exb' | 'compressible' | 'isothermal'")
      << "transport_choices() (message validate_model_spec)";
  EXPECT_EQ(elliptic_choices(), "'charge' | 'background' | 'gravity'") << "elliptic_choices()";
  EXPECT_EQ(source_choices(),
            "'none' | 'potential' | 'gravity' | 'magnetic' | 'potential_magnetic'")
      << "source_choices() lists canonical identities only";
  EXPECT_EQ(unknown_transport_msg("foo"), "unknown transport 'foo' (exb|compressible|isothermal)")
      << "unknown_transport_msg byte-identique a l'ancien throw";
  EXPECT_EQ(unknown_elliptic_msg("foo"), "unknown elliptic 'foo' (charge|background|gravity)")
      << "unknown_elliptic_msg byte-identique a l'ancien throw";
}

TEST(test_model_registry, validate_transport_and_elliptic_reject_explicitly) {
  std::string msg;

  EXPECT_FALSE(throws([] { validate_transport("exb"); }, msg)) << "validate_transport(exb) accepte";
  EXPECT_FALSE(throws([] { validate_transport("compressible"); }, msg))
      << "validate_transport(compressible) accepte";
  EXPECT_TRUE(throws([] { validate_transport("bogus"); }, msg))
      << "validate_transport(bogus) rejette";
  EXPECT_TRUE(contains(msg, "unknown transport") && contains(msg, "bogus") &&
              contains(msg, "exb|compressible|isothermal"))
      << "message transport inconnu : fragment + tag + liste valide";
  EXPECT_FALSE(throws([] { validate_elliptic("charge"); }, msg))
      << "validate_elliptic(charge) accepte";
  EXPECT_TRUE(throws([] { validate_elliptic("bogus"); }, msg))
      << "validate_elliptic(bogus) rejette";
  EXPECT_TRUE(contains(msg, "unknown elliptic") && contains(msg, "bogus") &&
              contains(msg, "charge|background|gravity"))
      << "message elliptic inconnu : fragment + tag + liste valide";
}

TEST(test_model_registry, capability_columns_nvars_polar_ok_min_vars) {
  EXPECT_TRUE(transport_n_vars("exb") == 1 && transport_n_vars("isothermal") == 3 &&
              transport_n_vars("compressible") == 4)
      << "transport_n_vars : exb=1, isothermal=3, compressible=4";
  EXPECT_EQ(transport_n_vars("bogus"), -1) << "transport_n_vars(inconnu) == -1";
  // variante compile-time (utilisee par les static_assert de non-derive de model_factory.hpp).
  static_assert(transport_n_vars_ct("exb") == 1, "ct exb");
  static_assert(transport_n_vars_ct("compressible") == 4, "ct compressible");
  static_assert(transport_n_vars_ct("isothermal") == 3, "ct isothermal");
  static_assert(transport_n_vars_ct("bogus") == -1, "ct inconnu == -1");
  // polar_ok : exactement {exb, isothermal} sont cables en polaire (compressible : phase ulterieure).
  int n_polar = 0;
  for (const TransportTag& t : kTransports)
    if (t.polar_ok)
      ++n_polar;
  EXPECT_EQ(n_polar, 2) << "deux transports polar_ok (exb + isothermal)";
  EXPECT_TRUE(is_transport("compressible")) << "compressible est un transport builtin (cartesien)";
  EXPECT_EQ(transport_tags_csv(/*polar=*/true).find("compressible"), std::string::npos)
      << "compressible ABSENT du sous-ensemble polaire (pas polar_ok)";
  // min_vars : 'none' neutre (1), les forces fluides exigent >= 3 variables.
  bool ok = true;
  for (const SourceTag& s : kSources) {
    if (std::string(s.name) == "none")
      ok = ok && (s.min_vars == 1);
    else
      ok = ok && (s.min_vars == 3);
  }
  EXPECT_TRUE(ok) << "source min_vars : none=1, forces fluides=3";
}

TEST(test_model_registry, registry_perimeter_guard_only_generic_bricks) {
  // Le SET de tags est verrouille : ModelSpec ne devient PAS une liste implicite de scenarios
  // (ADC-331). Ajouter un nom de scenario (diocotron, ...) ou retirer une brique fait echouer ici.
  constexpr int kNT = static_cast<int>(sizeof(kTransports) / sizeof(kTransports[0]));
  constexpr int kNS = static_cast<int>(sizeof(kSources) / sizeof(kSources[0]));
  constexpr int kNE = static_cast<int>(sizeof(kElliptics) / sizeof(kElliptics[0]));
  EXPECT_TRUE(kNT == 3 && kNS == 5 && kNE == 3)
      << "canonical table cardinality (3 transports, 5 sources, 3 elliptics)";
  const std::string tr = transport_tags_csv();
  const std::string sr = source_tags_csv();
  const std::string el = elliptic_tags_csv();
  EXPECT_EQ(tr, "exb|compressible|isothermal") << "set transport verrouille (briques generiques)";
  EXPECT_EQ(sr, "none|potential|gravity|magnetic|potential_magnetic")
      << "source registry contains canonical generic bricks only";
  EXPECT_EQ(el, "charge|background|gravity") << "set elliptic verrouille (briques generiques)";
}
