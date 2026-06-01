#pragma once

// Tags d'integration en temps, passes au coupleur pour selectionner le schema sans
// le reimplementer dans le cas. Le coupleur appelle le bon enchainement d'etages ;
// le cas ne fait que choisir le tag (comme on choisit un limiteur).
//
// Le tuteur : l'integration en temps vit dans le COEUR (l'equipe a peu d'expertise
// la-dessus), exposee comme une librairie selectionnable. SSPRK2 / SSPRK3 sont les
// schemas Shu-Osher SSP (TVD). En ajouter un = ajouter un tag + son enchainement
// d'etages dans Coupler, sans toucher aux modeles.

namespace adc {

struct SSPRK2 {};  // Shu-Osher SSP-RK2 (2 etages, ordre 2)
struct SSPRK3 {};  // Shu-Osher SSP-RK3 (3 etages, ordre 3)

}  // namespace adc
