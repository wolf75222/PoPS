/// @file
/// @brief Geometry : correspondance espace d'indices (Box2D) <-> espace physique cartesien ;
///        PolarGeometry : SIBLING pour un domaine annulaire global (r, theta).
///
/// Domaine physique FIXE, pas de maille dx/dy decroissant avec le raffinement (refine() raffine
/// l'espace d'indices a EXTENT physique constant). Centres de cellule (et faces, en polaire) definis
/// pour TOUT indice, ghosts compris (indices negatifs). Les accesseurs sont ADC_HD : arithmetique
/// pure, capturables par valeur et appelables depuis un kernel device. POD trivial : annotation
/// gratuite, chemin hote bit-identique.

#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>

// Geometry : la correspondance entre l'espace d'indices (Box2D) et l'espace
// physique d'un niveau. Domaine physique fixe, pas de maille dx/dy decroissant
// avec le raffinement. Centres de cellule definis pour tout indice, y compris
// les ghosts (indices negatifs).

namespace adc {

/// Geometrie cartesienne d'un niveau : domaine d'indices + bornes physiques [xlo, xhi] x [ylo, yhi].
/// POD trivial ; accesseurs ADC_HD. Maille uniforme (dx = (xhi-xlo)/nx, idem dy).
struct Geometry {
  Box2D domain{};
  Real xlo = 0, xhi = 1, ylo = 0, yhi = 1;

  // Accesseurs ADC_HD : arithmetique pure, capturables par valeur et appelables DEPUIS UN KERNEL
  // device. Sans ADC_HD, geom.x_cell(i) dans un kernel Kokkos::Cuda est un appel __host__ depuis
  // __device__ : nvcc rend une valeur GARBAGE (souvent 0) sur device, SANS erreur de compilation ni
  // d'execution. Un noyau d'init qui pose x = geom.x_cell(i) voit alors x = 0 sur GPU (sin(pi*0) = 0)
  // -> champ silencieusement nul (defaut observe sur test_condensed_schur). Geometry est un POD
  // trivial : l'annotation est gratuite et laisse le chemin hote bit-identique.
  /// Pas d'espace en x (= (xhi - xlo) / domain.nx()). ADC_HD.
  ADC_HD Real dx() const { return (xhi - xlo) / domain.nx(); }
  /// Pas d'espace en y (= (yhi - ylo) / domain.ny()). ADC_HD.
  ADC_HD Real dy() const { return (yhi - ylo) / domain.ny(); }
  /// Abscisse au CENTRE de la cellule d'indice i (i = 0 -> xlo + dx/2 ; defini pour i negatif). ADC_HD.
  ADC_HD Real x_cell(int i) const { return xlo + (i + Real(0.5)) * dx(); }
  /// Ordonnee au CENTRE de la cellule d'indice j. ADC_HD.
  ADC_HD Real y_cell(int j) const { return ylo + (j + Real(0.5)) * dy(); }

  // Meme extent physique, domaine d'indices raffine.
  /// Geometrie raffinee de ratio r : MEME extent physique, domaine d'indices raffine (dx -> dx/r).
  Geometry refine(int r) const { return Geometry{domain.refine(r), xlo, xhi, ylo, yhi}; }
};

// PolarGeometry : SIBLING de Geometry pour un domaine ANNULAIRE GLOBAL (r, theta).
// Chantier "grille polaire annulaire", Phase 1 (TRANSPORT seul, opt-in via adc.PolarMesh).
// Le proto Phase-0 (test_polar_ring_advection) a quantifie que la grille cartesienne diffuse
// le gradient RADIAL d'un anneau en rotation azimutale (~18%/5 tours) la ou la polaire le
// preserve (rapport 73) : porter la direction radiale sur un AXE de grille leve ce verrou.
//
// CONVENTION D'AXES (figee) :
//   - direction d'indice 0 = RADIALE   (i parcourt r, de r_min a r_max)
//   - direction d'indice 1 = AZIMUTALE  (j parcourt theta, de 0 a 2pi)
// Le domaine est r in [r_min, r_max] (BC PHYSIQUE en r_min/r_max) x theta in [0, 2pi)
// (PERIODIQUE en theta). C'est un anneau global, PAS un patch local cartesien<->polaire
// (l'interface hybride + interpolation + conservation au bord = Phase 2, hors scope ici).
//
// La maille (dr, dtheta) est uniforme en INDICE ; la maille PHYSIQUE en theta vaut r*dtheta
// et croit donc avec r (d'ou la metrique 1/r de la divergence : cf. assemble_rhs_polar). Les
// centres et faces sont definis pour tout indice (ghosts compris, i negatif ou >= nr).
struct PolarGeometry {
  Box2D domain{};            ///< nx() = nr (cellules radiales), ny() = ntheta (cellules azimutales)
  Real r_min = 0, r_max = 1;  ///< bornes radiales physiques de l'anneau
  // theta couvre [0, 2pi) (periodique) : on ne stocke pas de bornes, dtheta = 2pi/ntheta.

  // Pi local (pas de constante adc::kPi globale : elle entrerait en collision avec les kPi locaux
  // 'using namespace adc;' de plusieurs tests). Constexpr -> aucune surcharge.
  static constexpr Real kTwoPi = Real(2) * Real(3.14159265358979323846);

  // Accesseurs ADC_HD (meme motif que Geometry) : device-callable depuis un kernel sans rendre du
  // garbage sous nvcc. Arithmetique pure, hote bit-identique.
  ADC_HD Real dr() const { return (r_max - r_min) / domain.nx(); }
  ADC_HD Real dtheta() const { return kTwoPi / domain.ny(); }
  /// Rayon au CENTRE de la cellule radiale i (i = 0 -> r_min + dr/2).
  ADC_HD Real r_cell(int i) const { return r_min + (i + Real(0.5)) * dr(); }
  /// Rayon a la FACE radiale i (face entre les cellules i-1 et i ; i = 0 -> r_min, i = nr -> r_max).
  ADC_HD Real r_face(int i) const { return r_min + i * dr(); }
  /// Angle au CENTRE de la cellule azimutale j (j = 0 -> dtheta/2).
  ADC_HD Real theta_cell(int j) const { return (j + Real(0.5)) * dtheta(); }
  /// Angle a la FACE azimutale j (face entre les cellules j-1 et j).
  ADC_HD Real theta_face(int j) const { return j * dtheta(); }

  // Meme extent physique annulaire, domaine d'indices raffine (pendant de Geometry::refine).
  PolarGeometry refine(int r) const { return PolarGeometry{domain.refine(r), r_min, r_max}; }
};

}  // namespace adc
