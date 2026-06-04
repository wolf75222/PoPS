#pragma once

#include <adc/core/state.hpp>      // kAuxBaseComps
#include <adc/mesh/box2d.hpp>      // Box2D
#include <adc/mesh/fab2d.hpp>      // Fab2D
#include <adc/mesh/geometry.hpp>   // Geometry (x_cell / y_cell)
#include <adc/mesh/physical_bc.hpp>  // BCRec / BCType

/// @file
/// @brief Helpers partages par les trois coupleurs (Coupler mono-bloc, SystemAssembler,
///        AmrSystemCoupler) pour le canal aux. Centralise deux corps qui etaient dupliques
///        a l'identique :
///          - derive_aux_bc : CL du canal aux derivees des CL de phi (periodique conserve,
///            tout le reste -> Foextrap).
///          - fill_bz_box   : noyau de pose de B_z(x, y) a la composante kAuxBaseComps sur
///            une boite d'un Fab2D, depuis une geometrie donnee.
///        Extraction PURE : les corps sont repris a l'identique (bit-identique). Ce qui DIFFERE
///        entre coupleurs (garde compile-time vs runtime, boite valide vs grown, geometrie par
///        niveau, appel a fill_ghosts) reste cote appelant ; seul le contenu commun est ici.

namespace adc {
namespace detail {

/// CL du canal aux derivees des CL du potentiel phi : une CL periodique reste periodique,
/// toute autre devient Foextrap (extrapolation d'ordre 0). Corps repris a l'identique des
/// trois coupleurs.
inline BCRec derive_aux_bc(const BCRec& b) {
  auto t = [](BCType x) {
    return x == BCType::Periodic ? BCType::Periodic : BCType::Foextrap;
  };
  BCRec a;
  a.xlo = t(b.xlo);
  a.xhi = t(b.xhi);
  a.ylo = t(b.ylo);
  a.yhi = t(b.yhi);
  return a;
}

/// Pose B_z(x, y) a la composante kAuxBaseComps sur la boite @p box du fab @p f, en
/// echantillonnant @p bz aux centres de cellule de la geometrie @p g. Noyau commun aux trois
/// coupleurs : seule la boite parcourue (valide ou grown) et la geometrie (globale ou par
/// niveau) different cote appelant ; le corps de boucle est bit-identique.
template <class Bz>
inline void fill_bz_box(Fab2D& f, const Box2D& box, const Geometry& g, const Bz& bz) {
  for (int j = box.lo[1]; j <= box.hi[1]; ++j)
    for (int i = box.lo[0]; i <= box.hi[0]; ++i)
      f(i, j, kAuxBaseComps) = bz(g.x_cell(i), g.y_cell(j));
}

}  // namespace detail
}  // namespace adc
