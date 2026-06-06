/// @file
/// @brief AmrLevelStack : stockage de la hierarchie AMR (niveaux + aux) extrait des coupleurs.
///
/// Le coupleur n'ORDONNE plus que les operations ; ce stack DETIENT la pile de niveaux
/// std::vector<Level>, la pile d'aux MultiFab parallele, et porte le cablage L_[k].aux = &aux_[k].
/// Generique sur Level (AmrLevelMF mono-box ou AmrLevelMP multi-box) : seuls les membres U (MultiFab)
/// et aux (const MultiFab*) sont touches ici ; la repartition vit dans les MultiFab (pas suppose
/// mono-rang). INVARIANT D'ADRESSES : aux_ est dimensionne UNE seule fois au ctor puis jamais
/// redimensionne (les pointeurs L_[k].aux pointent dans aux_) ; reattach_aux(k) remplace aux_[k] EN
/// PLACE et recable. Largeur aux propagee en parametre (defaut kAuxBaseComps = 3, bit-identique).

#pragma once

#include <adc/core/state.hpp>  // kAuxBaseComps : largeur aux par defaut (canal de base phi/grad)
#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/multifab.hpp>

#include <utility>
#include <vector>

// Hierarchie AMR extraite des coupleurs (responsabilite a : stockage des niveaux
// + aux). Le coupleur n'ORDONNE plus que les operations ; ce stack DETIENT la pile
// de niveaux std::vector<Level> et la pile d'aux MultiFab parallele, et porte le
// cablage L_[k].aux = &aux_[k].
//
// Generique sur Level (AmrLevelMF mono-box ou AmrLevelMP multi-box) : les deux
// portent un membre U (MultiFab) et un membre aux (const MultiFab*), seuls champs
// touches ici. La repartition (DistributionMapping) est portee par les MultiFab,
// jamais supposee mono-rang : le stack ne fige pas la repartition.
//
// Invariant d'adresses : aux_ est dimensionne UNE seule fois au ctor puis jamais
// redimensionne (les pointeurs L_[k].aux pointent dans aux_). reattach_aux(k)
// remplace l'element aux_[k] en place (pas de resize) et recable L_[k].aux.
//
// Largeur du canal aux : aux_ncomp (defaut kAuxBaseComps = 3, le contrat de base phi/grad).
// Le coupleur, qui connait le Model, passe aux_comps<Model>() pour qu'un modele lisant des
// champs extra (B_z, ... ; n_aux > 3) dispose de la place. Le Model n'etant pas a portee ici
// (le stack est generique sur Level), la largeur est PROPAGEE en parametre. Defaut 3 ->
// allocation MultiFab(..., 3, 1) strictement bit-identique a l'historique.

namespace adc {

/// Detient la pile de niveaux AMR et la pile d'aux parallele. @tparam Level : type de niveau portant
/// U (MultiFab) et aux (const MultiFab*) (AmrLevelMF ou AmrLevelMP). INVARIANT : aux_ a une taille
/// figee au ctor (adresses stables pour L_[k].aux).
template <class Level>
class AmrLevelStack {
 public:
  /// Construit le stack : prend possession des @p levels, alloue un aux (aux_ncomp composantes, 1
  /// ghost) sur le layout de chaque U, et cable L_[k].aux = &aux_[k]. @p dom : domaine du niveau 0.
  AmrLevelStack(const Box2D& dom, std::vector<Level> levels,
                int aux_ncomp = kAuxBaseComps)
      : dom_(dom), L_(std::move(levels)), aux_ncomp_(aux_ncomp) {
    nlev_ = static_cast<int>(L_.size());
    aux_.resize(nlev_);  // addresses stables : aux_ n'est plus redimensionne
    for (int k = 0; k < nlev_; ++k) {
      aux_[k] = MultiFab(L_[k].U.box_array(), L_[k].U.dmap(), aux_ncomp_, 1);
      L_[k].aux = &aux_[k];
    }
  }

  std::vector<Level>& levels() { return L_; }
  const std::vector<Level>& levels() const { return L_; }
  MultiFab& coarse() { return L_[0].U; }
  const MultiFab& coarse() const { return L_[0].U; }
  const Box2D& domain() const { return dom_; }
  int nlev() const { return nlev_; }

  std::vector<Level>& L() { return L_; }
  std::vector<MultiFab>& aux() { return aux_; }
  MultiFab& aux(int k) { return aux_[k]; }
  const MultiFab& aux(int k) const { return aux_[k]; }

  // Largeur du canal aux (composantes), telle que dimensionnee au ctor.
  int aux_ncomp() const { return aux_ncomp_; }

  // realloc en place de aux_[k] sur la box courante de L_[k].U + recablage du
  // pointeur. Conserve la largeur du canal aux (aux_ncomp_) ; defaut 3 -> bit-identique
  // au bloc inline d'origine (meme MultiFab(..., 3, 1)).
  void reattach_aux(int k) {
    aux_[k] = MultiFab(L_[k].U.box_array(), L_[k].U.dmap(), aux_ncomp_, 1);
    L_[k].aux = &aux_[k];
  }

 private:
  Box2D dom_;
  std::vector<Level> L_;
  std::vector<MultiFab> aux_;
  int nlev_ = 0;
  int aux_ncomp_ = kAuxBaseComps;  // largeur du canal aux (defaut : contrat de base)
};

}  // namespace adc
