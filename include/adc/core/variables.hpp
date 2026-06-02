#pragma once

#include <string>
#include <vector>

/// @file
/// @brief Descripteur des variables d'un modele (Vars). Porte par la brique HYPERBOLIQUE (avec le
///        flux et les conversions), car variables et flux sont physiquement lies ; ce n'est PAS une
///        brique independante combinable librement.
///
/// `Variables` DECRIT les variables (conservatives ou primitives) : nature, noms, taille. C'est une
/// metadonnee HOTE (elle ne pilote pas le calcul, qui travaille par composante via les conversions
/// cons<->prim), mais c'est un CONTRAT OBLIGATOIRE du modele hyperbolique (concept HyperbolicModel) :
/// conservative_vars() et primitive_vars(). Sert a l'introspection, aux diagnostics nommes, a la
/// sortie labellisee.

namespace adc {

enum class VariableKind { Conservative, Primitive };

/// Jeu de variables d'un modele : nature (cons/prim), noms, taille.
struct Variables {
  VariableKind kind;
  std::vector<std::string> names;
  int size;
};

}  // namespace adc
