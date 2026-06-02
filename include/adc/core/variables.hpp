#pragma once

#include <string>
#include <vector>

/// @file
/// @brief Descripteur des variables d'un modele : Vars, brique de premiere classe au meme niveau
///        que Flux et Source.
///
/// `Variables` DECRIT les variables d'un modele (conservatives ou primitives) : leur nature, leurs
/// noms, leur taille. C'est une METADONNEE HOTE : elle ne sert pas au calcul (le coeur travaille par
/// composante, aveugle au sens des composantes), mais a l'introspection, aux diagnostics nommes et a
/// la sortie labellisee. Un modele expose conservative_vars() et primitive_vars().

namespace adc {

enum class VariableKind { Conservative, Primitive };

/// Jeu de variables d'un modele : nature (cons/prim), noms, taille.
struct Variables {
  VariableKind kind;
  std::vector<std::string> names;
  int size;
};

}  // namespace adc
