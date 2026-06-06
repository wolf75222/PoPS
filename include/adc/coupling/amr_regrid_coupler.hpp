#pragma once

#include <adc/amr/cluster.hpp>  // berger_rigoutsos, ClusterParams
#include <adc/amr/regrid.hpp>   // tag_cells, grow_tags
#include <adc/amr/tag_box.hpp>  // TagBox
#include <adc/core/types.hpp>
#include <adc/numerics/time/amr_reflux_mf.hpp>  // AmrLevelMP, mf_find_box
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence (barriere apres parallel_copy async sous Cuda)
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/refinement.hpp>  // coarsen_index
#include <adc/parallel/comm.hpp>    // n_ranks (include explicite, plus de chemin indirect)

#include <algorithm>
#include <utility>
#include <vector>

/// @file
/// @brief amr_regrid_finest : regrid Berger-Rigoutsos du niveau le plus fin (responsabilite b).
///
/// Free function template sur le critere, calquee sur le STYLE de amr/regrid.hpp (regrid_level) mais
/// PAS fusionnee : invariants differents (coords niveau fk = parent x2, clamp de nesting margin,
/// report de l'ancien fin). Corps deplace tel quel depuis AmrCouplerMP::regrid : meme tagging,
/// clustering, clamp, interp parent puis report fin, swap + realloc aux. Ne suppose pas mono-rang
/// (DistributionMapping construite avec n_ranks()). Sous grossier REPARTI, l'OU global des tags
/// (all_reduce_or) garantit des patchs fins IDENTIQUES sur tous les rangs (sinon dmaps incompatibles).

// Regrid Berger-Rigoutsos extrait du coupleur multi-patch (responsabilite b).
// Free function template sur le critere, calquee sur le STYLE de amr/regrid.hpp
// (regrid_level) mais PAS fusionnee : invariants differents (coords niveau fk =
// parent x2, clamp de nesting margin, report de l'ancien fin). Corps deplace TEL
// QUEL depuis AmrCouplerMP::regrid : meme tagging, meme clustering, meme clamp,
// meme interp parent puis report fin, meme swap + realloc aux.
//
// Inclut explicitement adc/parallel/comm.hpp pour n_ranks() (avant : atteint
// transitivement via amr_reflux_mf.hpp -> comm.hpp). Ne suppose pas mono-rang :
// la DistributionMapping reste construite avec n_ranks().

namespace adc {

// Regrid du niveau le plus fin (L.back()) par Berger-Rigoutsos sur le critere
// applique au parent. Reconstruit les patchs (report des donnees fines la ou
// possible, sinon interpolation depuis le parent) + l'aux associe. margin =
// nesting. No-op si moins de 2 niveaux ou si aucun patch ne sort du clustering.
//
// aux_ncomp : largeur du canal aux reconstruit (defaut kAuxBaseComps = 3). Le coupleur,
// qui connait le Model, propage aux_comps<Model>() pour qu'un modele lisant des champs
// extra (B_z, ...) garde la place apres regrid. Le Model n'etant pas a portee ici (free
// function sur le seul critere), la largeur est PROPAGEE en parametre ; defaut 3 ->
// allocation MultiFab(..., 3, 1) strictement bit-identique a l'historique.
/// Regrid le niveau le plus fin (L.back()) par Berger-Rigoutsos sur le critere @p crit applique au
/// parent : reconstruit les patchs (report des donnees fines sinon interp parent) + l'aux. @p grow :
/// dilatation des tags ; @p margin : nesting ; @p aux_ncomp : largeur aux reconstruit ;
/// @p coarse_replicated : politique d'ownership du niveau 0. NO-OP si < 2 niveaux ou aucun patch.
template <class Crit>
void amr_regrid_finest(std::vector<AmrLevelMP>& L, std::vector<MultiFab>& aux,
                       const Box2D& dom, Crit crit, int grow, int margin,
                       int aux_ncomp = kAuxBaseComps, bool coarse_replicated = true) {
  const int nlev = static_cast<int>(L.size());
  if (nlev < 2) return;
  const int fk = nlev - 1, pk = fk - 1;  // fin et son parent
  const int PNX = dom.nx() << pk, PNY = dom.ny() << pk;
  const Box2D pdom = Box2D::from_extents(PNX, PNY);
  TagBox tags = tag_cells(L[pk].U, pdom, crit);
  TagBox grown = grow_tags(tags, grow, pdom);
  // GROSSIER REPARTI (pk == 0 && !coarse_replicated) : tag_cells n'a vu que les boites LOCALES, donc
  // chaque rang n'a tague qu'une partie du domaine. OU global avant le clustering -> tous les rangs
  // partent de la MEME grille de tags et Berger-Rigoutsos produit des patchs fins IDENTIQUES (sinon
  // la BoxArray fine differerait par rang -> dmaps incompatibles, MPI desynchronise). Replique : la
  // grille de tags est deja complete sur chaque rang (no-op, all_reduce_or serait l'identite).
  if (pk == 0 && !coarse_replicated)
    all_reduce_or_inplace(grown.t.data(), static_cast<int>(grown.t.size()));
  std::vector<Box2D> cl = berger_rigoutsos(grown, ClusterParams{});
  std::vector<Box2D> fb;  // patchs fins (coords niveau fk = parent x2)
  for (Box2D b : cl) {
    b.lo[0] = std::max(b.lo[0], margin); b.lo[1] = std::max(b.lo[1], margin);
    b.hi[0] = std::min(b.hi[0], PNX - 1 - margin); b.hi[1] = std::min(b.hi[1], PNY - 1 - margin);
    if (b.hi[0] < b.lo[0] || b.hi[1] < b.lo[1]) continue;
    fb.push_back(Box2D{{2 * b.lo[0], 2 * b.lo[1]}, {2 * b.hi[0] + 1, 2 * b.hi[1] + 1}});
  }
  if (fb.empty()) return;  // rien a raffiner : on garde la grille courante
  // Les nouveaux patchs HERITENT la largeur de ghost du niveau remplace (et non un 1 fige) : un
  // niveau reconstruit en MUSCL ordre 2 (Minmod / VanLeer) porte 2 ghosts, que le regrid doit
  // preserver, sinon la reconstruction lirait hors bornes apres re-raffinement.
  const int ngf = L[fk].U.n_grow();
  MultiFab nU(BoxArray(fb), DistributionMapping((int)fb.size(), n_ranks()), L[fk].U.ncomp(), ngf);
  const MultiFab& par = L[pk].U;
  const MultiFab& old = L[fk].U;
  const int ncf = nU.ncomp();
  // Parent REPARTI (grossier de-replique) : par.fab ne contient que les boites LOCALES, donc
  // mf_find_box renverrait -1 pour une cellule grossiere possedee par un rang DISTANT et le patch
  // resterait non initialise la. On amene les regions parentes necessaires sur une grille
  // enfant-coarsen LOCALE (coarsen du BoxArray fin) par parallel_copy, puis on interpole depuis
  // elle. Parent replique : par est entierement local, lecture directe via mf_find_box (no-op,
  // bit-identique a l'historique mono-box/replique).
  const bool par_replicated = (pk != 0) || coarse_replicated;
  MultiFab parloc;
  if (!par_replicated) {
    parloc = MultiFab(coarsen(nU.box_array(), 2), nU.dmap(), par.ncomp(), 0);
    parallel_copy(parloc, par);
    // parallel_copy lance des kernels de copie ASYNC sous Cuda et, a np=1, retourne SANS fence
    // (la barriere interne n'est sur le chemin que pour np>1). Sans ce fence, la lecture de parloc
    // ci-dessous lirait de la memoire device non encore ecrite -> NaN (vu sur GH200 np=1 reparti).
    device_fence();
  }
  for (int li = 0; li < nU.local_size(); ++li) {
    Array4 a = nU.fab(li).array();
    const Box2D nb = nU.box(li);
    if (par_replicated) {
      for (int j = nb.lo[1]; j <= nb.hi[1]; ++j)  // 1) interp depuis le parent (local)
        for (int i = nb.lo[0]; i <= nb.hi[0]; ++i) {
          const int pb = mf_find_box(par, coarsen_index(i, 2), coarsen_index(j, 2));
          if (pb < 0) continue;
          const ConstArray4 pp = par.fab(pb).const_array();
          for (int k = 0; k < ncf; ++k)
            a(i, j, k) = pp(coarsen_index(i, 2), coarsen_index(j, 2), k);
        }
    } else {
      const ConstArray4 pp = parloc.fab(li).const_array();  // grille enfant-coarsen locale
      for (int j = nb.lo[1]; j <= nb.hi[1]; ++j)
        for (int i = nb.lo[0]; i <= nb.hi[0]; ++i)
          for (int k = 0; k < ncf; ++k)
            a(i, j, k) = pp(coarsen_index(i, 2), coarsen_index(j, 2), k);
    }
    for (int ol = 0; ol < old.local_size(); ++ol) {  // 2) report des donnees fines
      const ConstArray4 o = old.fab(ol).const_array();
      const Box2D inter = nb.intersect(old.box(ol));
      if (inter.empty()) continue;
      for (int j = inter.lo[1]; j <= inter.hi[1]; ++j)
        for (int i = inter.lo[0]; i <= inter.hi[0]; ++i)
          for (int k = 0; k < ncf; ++k) a(i, j, k) = o(i, j, k);
    }
  }
  L[fk].U = std::move(nU);
  aux[fk] = MultiFab(L[fk].U.box_array(), L[fk].U.dmap(), aux_ncomp, 1);  // adresse stable
  L[fk].aux = &aux[fk];
}

}  // namespace adc
