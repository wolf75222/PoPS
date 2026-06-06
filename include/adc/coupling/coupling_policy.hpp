/// @file
/// @brief Politiques de couplage temporel hyperbolique-elliptique (tag types compile-time).
///
/// A quelle FREQUENCE on resout l'elliptique dans un pas de temps. Tag types choisis par template au
/// site d'appel (Coupler::advance<Limiter, Policy>), AUCUNE branche a l'execution. PerStageCoupling :
/// phi (donc aux = grad phi) recalcule a CHAQUE etage RK -> couplage le plus precis, une resolution
/// elliptique par etage. OncePerStepCoupling : phi resolu une SEULE fois (debut de pas), aux gele
/// pendant les etages -> moins cher, splitting de fait. (AMR sous-cyclage et redistribution
/// tuiles <-> bandes FFT sont des politiques de la meme famille, portees par AmrCoupler /
/// SpectralCoupler.)

#pragma once

// Politique de couplage temporel hyperbolique-elliptique : a quelle FREQUENCE on
// resout l'elliptique dans un pas de temps. Tag types, choisis par template au
// site d'appel (Coupler::advance<Limiter, Policy>), pas de branche a l'execution.
//
//   PerStageCoupling    : phi (donc aux = grad phi) recalcule a CHAQUE etage RK.
//                         Le potentiel suit l'etat intermediaire -> couplage le
//                         plus precis, mais une resolution elliptique par etage.
//   OncePerStepCoupling : phi resolu une SEULE fois (debut de pas), aux gele
//                         pendant les etages. Une resolution elliptique par pas
//                         (moins cher), au prix d'un splitting de fait sur le
//                         couplage. Utile quand l'elliptique domine le cout.
//
// (AMR sous-cyclage et redistribution tuiles<->bandes FFT sont des politiques de
// la meme famille, portees par AmrCoupler / SpectralCoupler.)

namespace adc {

/// Tag : resout l'elliptique a CHAQUE etage RK (aux suit l'etat intermediaire, plus precis).
struct PerStageCoupling {};
/// Tag : resout l'elliptique UNE fois par pas (aux gele pendant les etages, moins cher).
struct OncePerStepCoupling {};

}  // namespace adc
