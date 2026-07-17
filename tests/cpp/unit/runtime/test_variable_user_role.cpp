// ADC-292 : roles utilisateurs NOMMES + fin des fallbacks canoniques SILENCIEUX.
//
// Verrouille la couche de role string-keyee ajoutee a VariableSet (parallele a l'enum VariableRole,
// label porte par user_roles) et la resolution stricte partagee (require_role_index) :
//   (1) un ROLE UTILISATEUR nomme se resout par son label (index_of(string)) -- plus d'ambiguite de
//       premiere-occurrence Custom ;
//   (2) un nom de role CANONIQUE se resout toujours (par nom et par enum), meme en layout NON canonique ;
//   (3) un role/label ABSENT renvoie -1 ;
//   (4) roles_csv emet le label utilisateur et parse_roles_into le reconstruit (aller-retour ABI .so),
//       en restant bit-identique pour un jeu purement canonique (user_roles vide) ;
//   (5) require_role_index refuse un descripteur incomplet, resout un role unique et refuse un role
//       absent ou duplique. Aucun index canonique n'est devine.
//
// Test PUR (n'inclut que core/variables.hpp) : aucune dependance runtime, lie pops::pops seul.
#include <gtest/gtest.h>

#include <pops/core/state/variables.hpp>

#include <stdexcept>
#include <string>

using R = pops::VariableRole;

TEST(VariableUserRole, NamedUserRolesAndStrictCouplingFallback) {
  // --- (1)+(2)+(3) index_of(string) : layer role canonique + role utilisateur, layout NON canonique ---
  // Bloc fictif : momentum_x en comp 0, un champ utilisateur "phi" en comp 1, densite en comp 2 (layout
  // NON canonique). roles porte l'enum (Custom pour "phi"), user_roles porte le label parallele.
  pops::VariableSet vs;
  vs.kind = pops::VariableKind::Conservative;
  vs.names = {"mx", "phi", "rho"};
  vs.size = 3;
  vs.roles = {R::MomentumX, R::Custom, R::Density};
  vs.user_roles = {"", "phi", ""};

  EXPECT_EQ(vs.index_of("density"), 2)
      << "index_of_string:canonical_name_non_canonical_layout";                             // (2)
  EXPECT_EQ(vs.index_of(R::Density), 2) << "index_of_enum:canonical_non_canonical_layout";  // (2)
  EXPECT_EQ(vs.index_of("momentum_x"), 0) << "index_of_string:canonical_name_resolves";
  EXPECT_EQ(vs.index_of("phi"), 1) << "index_of_string:user_label_resolves";                  // (1)
  EXPECT_EQ(vs.index_of("energy"), -1) << "index_of_string:absent_canonical_name_is_minus1";  // (3)
  EXPECT_EQ(vs.index_of("psi"), -1) << "index_of_string:absent_user_label_is_minus1";         // (3)
  // An EMPTY role string is never a valid target: on this MIXED block user_roles[0] == "" (the
  // canonical momentum_x slot), so without a guard index_of("") would wrongly resolve to component 0
  // -- exactly the silent fallback ADC-292 kills. It must return -1.
  EXPECT_EQ(vs.index_of(""), -1) << "index_of_string:empty_role_is_minus1";  // (3)

  // --- (4) aller-retour CSV : roles_csv emet le label utilisateur, parse_roles_into le reconstruit ----
  const std::string csv = pops::roles_csv(vs);
  EXPECT_EQ(csv, "momentum_x,phi,density") << "roles_csv:emits_user_label";
  pops::VariableSet rt;
  rt.kind = pops::VariableKind::Conservative;
  rt.names = vs.names;
  rt.size = vs.size;
  pops::parse_roles_into(rt, csv);
  EXPECT_EQ(rt.index_of("phi"), 1) << "parse_roles_into:roundtrips_user_label";
  EXPECT_EQ(rt.index_of(R::Density), 2) << "parse_roles_into:roundtrips_canonical";
  EXPECT_TRUE(rt.roles.size() == 3 && rt.roles[1] == R::Custom)
      << "parse_roles_into:custom_enum_for_user_label";

  // Jeu PUREMENT canonique : user_roles reste vide (bit-identique au comportement historique, aucune
  // regression sur les blocs existants ni sur l'ABI .so des blocs sans role utilisateur).
  pops::VariableSet canon;
  canon.kind = pops::VariableKind::Conservative;
  pops::parse_roles_into(canon, "density,momentum_x,energy");
  EXPECT_TRUE(canon.user_roles.empty()) << "parse_roles_into:canonical_csv_leaves_user_roles_empty";
  EXPECT_EQ(canon.index_of(R::Energy), 2) << "parse_roles_into:canonical_csv_roles_resolve";

  // --- (5) require_role_index : metadonnees totales obligatoires, aucun fallback d'indice -----------
  pops::VariableSet roleless;
  roleless.kind = pops::VariableKind::Conservative;
  roleless.names = {"u0", "u1", "u2"};
  roleless.size = 3;  // roles + user_roles vides
  EXPECT_THROW((void)pops::require_role_index(roleless, R::MomentumX, "test", "blk"),
               std::runtime_error)
      << "require_role_index:roleless_block_is_invalid";

  // Bloc total QUI PORTE le role : on retourne son indice non canonique (density en comp 2).
  EXPECT_EQ(pops::require_role_index(vs, R::Density, "test", "blk"), 2)
      << "require_role_index:present_role_returns_index";

  // Bloc AVEC roles mais SANS le role requis (vs ne porte pas Energy) -> LEVE en nommant bloc + role.
  bool threw = false, names_block = false, names_role = false;
  try {
    (void)pops::require_role_index(vs, R::Energy, "coupling role resolve", "fluid_a");
  } catch (const std::exception& e) {
    threw = true;
    const std::string m = e.what();
    names_block = m.find("fluid_a") != std::string::npos;
    names_role = m.find("energy") != std::string::npos;
  }
  EXPECT_TRUE(threw) << "require_role_index:roles_bearing_missing_role_raises";
  EXPECT_TRUE(names_block) << "require_role_index:error_names_block";
  EXPECT_TRUE(names_role) << "require_role_index:error_names_role";

  pops::VariableSet duplicate = vs;
  duplicate.roles[1] = R::Density;
  EXPECT_THROW((void)pops::require_role_index(duplicate, R::Density, "test", "dup"),
               std::runtime_error)
      << "require_role_index:duplicate_physical_role_is_ambiguous";
}
