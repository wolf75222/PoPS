#pragma once

/// @file
/// @brief Helpers partages des tests C++ executes sous GoogleTest.
///
/// Couche : `tests/` (hors `include/pops`, reserve aux executables de test ; non installe).
/// Role : factoriser les briques copiees a l'identique dans des dizaines de tests -- le compteur
///   d'echecs + la fonction d'assertion `chk`, le predicat d'exception `raises`, la comparaison
///   relative `close_rel`, la somme de controle `checksum`, la constante `kPi`.
/// Contrat : header-only, branche sur GoogleTest pour remonter les echecs via `ADD_FAILURE()`,
///   tout en conservant les corps de tests historiques sous forme de fonction qui renvoie 0/1.
///   L'enregistrement, le main, les labels et les rapports XML appartiennent a GoogleTest/CTest.
///
/// Invariants :
/// - aucune variable globale, aucun etat de processus : tout passe par le compteur fourni par
///   l'appelant (`pops::test::Checker` encapsule le compteur ou l'appelant garde son `int fails`) ;
/// - les fonctions n'impriment que sur stdout/stderr et ne touchent ni MPI ni Kokkos (un test MPI
///   garde son propre `long fails` reduit par all_reduce a la main, cf. tests MPI) ;
/// - chaque echec de `Checker` incremente le compteur ET produit un echec GoogleTest visible ;
/// - `close_rel` et `checksum` sont des copies bit-a-bit des versions locales remplacees, pour que
///   le comportement (donc la sortie et le code retour) des tests migres reste STRICTEMENT identique.

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace pops::test {

/// Pi en double precision (copie de la constante `kPi` / `pi` dupliquee dans les tests).
inline constexpr double kPi = 3.14159265358979323846;

/// Compteur d'echecs encapsule + assertion.
///
/// Usage : `Checker chk;` puis `chk(cond, "libelle");` dans le corps du main(), et
///   `return chk.failed();` a la fin. Remplace le couple idiomatique `int fails = 0;` + lambda
///   `auto chk = [&](bool c, const char* w){...}` recopie dans presque chaque test.
/// Contrat : deux styles d'impression, choisis a la construction, pour reproduire EXACTEMENT la
///   sortie des deux familles de tests existantes :
///   - `Style::Terse`   (defaut) : n'imprime QUE les echecs, `FAIL <libelle>\n` (cf. test_box2d) ;
///   - `Style::Verbose` : imprime chaque ligne `  [OK ] <libelle>` / `  [XX ] <libelle>`
///     (cf. test_dense_eig, test_amr_system_contract).
/// Contraintes : non copiable par valeur n'est pas requis ; on le capture par reference dans une
///   lambda `chk` si le main() prefere garder cette forme (cf. exemple ci-dessous).
class Checker {
 public:
  enum class Style { Terse, Verbose };

  explicit Checker(Style style = Style::Terse) : style_(style) {}

  /// Verifie @p cond ; incremente le compteur et imprime un diagnostic si @p cond est faux.
  /// @return @p cond (permet `if (!chk(...)) {...}` si l'appelant veut court-circuiter).
  bool operator()(bool cond, const char* label) {
    if (style_ == Style::Verbose) {
      std::printf("  [%s] %s\n", cond ? "OK " : "XX ", label);
    } else if (!cond) {
      std::printf("FAIL %s\n", label);
    }
    if (!cond) {
      ADD_FAILURE() << label;
      ++fails_;
    }
    return cond;
  }

  /// Nombre d'echecs accumules.
  int fails() const { return fails_; }

  /// Code retour conventionnel : 0 si aucun echec, 1 sinon (a renvoyer depuis main()).
  int failed() const { return fails_ == 0 ? 0 : 1; }

 private:
  Style style_;
  int fails_ = 0;
};

/// Renvoie true si l'appel @p f leve un `std::runtime_error` (le refus attendu d'un contrat).
///
/// Copie de la lambda `raises` dupliquee dans les tests de contrat (test_amr_system_contract, ...).
/// Une exception d'un autre type, ou aucune exception, renvoie false.
template <class F>
bool raises(F&& f) {
  try {
    f();
  } catch (const std::runtime_error&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

/// Comparaison a tolerance relative + absolue : |a - b| <= rtol * max(|a|, |b|) + atol.
///
/// Copie bit-a-bit de la version locale de test_dense_eig (atol par defaut 1e-12). @p T est le type
/// flottant (`pops::Real` ou `double`) ; deduit a l'appel.
template <class T>
bool close_rel(T a, T b, T rtol, T atol = T(1e-12)) {
  const T d = std::fabs(a - b);
  const T s = std::fabs(a) > std::fabs(b) ? std::fabs(a) : std::fabs(b);
  return d <= rtol * s + atol;
}

/// Somme de controle = somme des carres des elements (signature deterministe d'un champ).
///
/// Copie de la lambda `checksum` dupliquee dans les tests de parite MPI/AMR (test_amr_regrid_mpi_parity,
/// test_mpi_amr_*_parity). On somme x*x : invariant au signe, sensible a toute divergence numerique.
inline double checksum(const std::vector<double>& v) {
  double s = 0;
  for (double x : v)
    s += x * x;
  return s;
}

}  // namespace pops::test
