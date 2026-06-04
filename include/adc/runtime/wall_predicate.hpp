#pragma once

#include <adc/core/types.hpp>  // Real

#include <cmath>       // std::hypot
#include <functional>  // std::function
#include <stdexcept>   // std::runtime_error
#include <string>

/// @file
/// @brief Predicat de paroi (conducteur embedded) partage par les runtimes System et AmrSystem.
///        Les deux derivaient le meme predicat depuis les memes parametres (wall, rayon, L) ;
///        seul le prefixe du message d'erreur differait. Centralise ici par extraction PURE :
///        le corps (cercle centre, comparaison std::hypot < R) est repris a l'identique.

namespace adc {
namespace detail {

/// Construit le predicat "interieur du conducteur" (paroi embedded pour le solveur de Poisson)
/// depuis le mode de paroi @p wall, le rayon @p wall_radius et la taille de domaine @p L.
///   - "none"   : pas de paroi -> predicat vide.
///   - "circle" : disque centre en (L/2, L/2) de rayon @p wall_radius.
///   - autre    : erreur, prefixee par @p err_context (p.ex. "System::set_poisson").
/// Corps repris a l'identique des runtimes System / AmrSystem (bit-identique).
inline std::function<bool(Real, Real)> wall_predicate(const std::string& wall,
                                                      double wall_radius, double L,
                                                      const std::string& err_context) {
  if (wall == "none") return {};
  if (wall == "circle") {
    const double cx = 0.5 * L, cy = 0.5 * L, R = wall_radius;
    return [cx, cy, R](Real x, Real y) { return std::hypot(x - cx, y - cy) < R; };
  }
  throw std::runtime_error(err_context + " : wall inconnu '" + wall + "'");
}

}  // namespace detail
}  // namespace adc
