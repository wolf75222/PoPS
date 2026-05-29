#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>

// Flux numerique a une interface, exprime en POLITIQUE (template), au meme titre
// que le limiteur de reconstruction. assemble_rhs<Limiter, NumericalFlux> choisit
// les deux independamment, au lieu d'appeler en dur rusanov_flux.
//
// Contrat d'une politique de flux : un foncteur device-callable (ADC_HD)
//   operator()(model, UL, AL, UR, AR, dir) -> Model::State
// qui rend le flux numerique a l'interface entre l'etat gauche (UL, aux AL) et
// droit (UR, aux AR) dans la direction dir. Etats par valeur, aucun virtuel :
// utilisable tel quel dans un kernel.
//
//   RusanovFlux : Lax-Friedrichs local, alpha = max des |vitesses d'onde| des deux
//                 etats. Robuste, diffusif. Seul flux ne demandant que
//                 max_wave_speed (donc compatible avec le concept PhysicalModel
//                 actuel sans extension).
//
// HLL / HLLC arriveront avec model/euler.hpp : ils exigent les vitesses d'onde
// SIGNEES s_L, s_R (et l'onde de contact s_* pour HLLC), donc une extension du
// concept (p.ex. wave_speeds(U, aux, dir) -> {sL, sR}). Rusanov n'en a pas besoin.

namespace adc {

struct RusanovFlux {
  template <class Model>
  ADC_HD typename Model::State operator()(const Model& m,
                                          const typename Model::State& UL,
                                          const Aux& AL,
                                          const typename Model::State& UR,
                                          const Aux& AR, int dir) const {
    const auto FL = m.flux(UL, AL, dir);
    const auto FR = m.flux(UR, AR, dir);
    const Real sL = m.max_wave_speed(UL, AL, dir);
    const Real sR = m.max_wave_speed(UR, AR, dir);
    const Real alpha = sL > sR ? sL : sR;  // max device-safe (pas de std::max)
    typename Model::State F;
    for (int c = 0; c < Model::n_vars; ++c)
      F[c] = Real(0.5) * (FL[c] + FR[c]) - Real(0.5) * alpha * (UR[c] - UL[c]);
    return F;
  }
};

}  // namespace adc
