#pragma once

#include <adc/core/types.hpp>

#include <concepts>

// Le contrat de la couche physique.
//
// Un PhysicalModel decrit UNE equation : ses formules ponctuelles. Rien de
// plus. C'est le seul axe "quoi calculer" de l'architecture, separe de l'axe
// "ou / comment iterer" (maillage + dispatch) et de l'axe "dans quel ordre"
// (integrateur + coupleur).
//
// Tout est fonction pure d'etats ponctuels :
//   - flux(U, aux, dir)          : le flux physique dans la direction dir
//   - max_wave_speed(U, aux, dir): la plus grande vitesse d'onde (pour le CFL
//                                  et le solveur de Riemann)
//   - source(U, aux)             : le terme source ponctuel
//   - elliptic_rhs(U)            : le second membre de l'equation elliptique
//                                  (densite de charge / de masse selon le modele)
//
// flux ET source prennent aux : c'est le point qui unifie diocotron (aux dans
// le flux) et Euler-Poisson (aux dans la source) sous un meme operateur spatial.

namespace adc {

template <class M>
concept PhysicalModel =
    requires(const M m, const typename M::State u, const typename M::Aux a,
             int dir) {
      typename M::State;
      typename M::Aux;
      { M::n_vars } -> std::convertible_to<int>;
      { m.flux(u, a, dir) } -> std::same_as<typename M::State>;
      { m.max_wave_speed(u, a, dir) } -> std::convertible_to<Real>;
      { m.source(u, a) } -> std::same_as<typename M::State>;
      { m.elliptic_rhs(u) } -> std::convertible_to<Real>;
    };

}  // namespace adc
