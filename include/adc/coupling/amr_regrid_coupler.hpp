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

// ------------------------------------------------------------------------------------------------
// REFACTOR (docs/AMR_REGRID_UNION_TAGS_DESIGN.md section 6) : amr_regrid_finest scinde en DEUX
// responsabilites pour que le regrid d'UNION multi-blocs (AmrRuntime::regrid) puisse re-griller
// PLUSIEURS champs sur UN SEUL layout impose de l'exterieur (le meme pour tous les blocs) :
//   (1) regrid_compute_fine_layout : tags d'un parent -> grow -> all_reduce_or (si reparti) ->
//       berger_rigoutsos -> clamp de nesting -> (BoxArray fin, DistributionMapping). C'est le
//       CALCUL du layout, fait UNE fois sur l'union des tags par le multi-blocs.
//   (2) regrid_field_on_layout : prend (fb, dmap) IMPOSE et reconstruit UN MultiFab fin (interp
//       parent + report fin), exactement le CORPS de l'ancien amr_regrid_finest, mais sans le
//       calcul du layout. Appele PAR BLOC sur le meme layout d'union.
// amr_regrid_finest reste l'ENCHAINEMENT (1) puis (2) sur UN seul bloc -> le chemin mono-bloc
// (AmrCouplerMP::regrid) reste BIT-IDENTIQUE (memes operations, meme ordre).
// ------------------------------------------------------------------------------------------------

/// Calcule le layout fin (BoxArray + DistributionMapping) d'un regrid Berger-Rigoutsos a partir des
/// tags @p grown DEJA dilates (grow_tags) sur le domaine PARENT @p pdom. @p pk : niveau parent (la
/// reduction MPI cross-rang n'a lieu que pour pk==0 reparti) ; @p margin : nesting ; @p coarse_replicated :
/// politique d'ownership du niveau 0. Coords du niveau fin = parent x2. Renvoie un BoxArray VIDE si rien
/// a raffiner (l'appelant garde alors la grille courante). MPI-safe : sous grossier reparti, l'OU global
/// des tags (all_reduce_or) garantit des patchs IDENTIQUES sur tous les rangs (sinon dmaps incompatibles).
inline std::pair<BoxArray, DistributionMapping> regrid_compute_fine_layout(
    TagBox grown, const Box2D& pdom, int pk, int margin, bool coarse_replicated = true) {
  const int PNX = pdom.nx(), PNY = pdom.ny();
  // GROSSIER REPARTI (pk == 0 && !coarse_replicated) : chaque rang n'a tague que ses boites LOCALES.
  // OU global avant le clustering -> tous les rangs partent de la MEME grille de tags (sinon la
  // BoxArray fine differerait par rang -> dmaps incompatibles, MPI desynchronise). Replique : la
  // grille est deja complete sur chaque rang (no-op, all_reduce_or serait l'identite).
  if (pk == 0 && !coarse_replicated)
    all_reduce_or_inplace(grown.t.data(), static_cast<int>(grown.t.size()));
  std::vector<Box2D> cl = berger_rigoutsos(grown, ClusterParams{});
  std::vector<Box2D> fb;  // patchs fins (coords niveau fin = parent x2)
  for (Box2D b : cl) {
    b.lo[0] = std::max(b.lo[0], margin); b.lo[1] = std::max(b.lo[1], margin);
    b.hi[0] = std::min(b.hi[0], PNX - 1 - margin); b.hi[1] = std::min(b.hi[1], PNY - 1 - margin);
    if (b.hi[0] < b.lo[0] || b.hi[1] < b.lo[1]) continue;
    fb.push_back(Box2D{{2 * b.lo[0], 2 * b.lo[1]}, {2 * b.hi[0] + 1, 2 * b.hi[1] + 1}});
  }
  if (fb.empty()) return {BoxArray{}, DistributionMapping{}};  // rien a raffiner
  BoxArray ba(fb);
  return {ba, DistributionMapping(static_cast<int>(ba.size()), n_ranks())};
}

/// Reconstruit UN MultiFab fin sur le layout @p fb / @p dmap IMPOSE (le meme pour tous les blocs en
/// multi-blocs) : (a) interpolation piecewise-constante depuis le parent @p par la ou le nouveau patch
/// n'est pas couvert par l'ancien fin, (b) report des donnees fines existantes @p old la ou l'ancien
/// patch couvre le nouveau. @p ngf : largeur de ghost du fin (heritee de l'ancien niveau remplace) ;
/// @p coarse_replicated : politique d'ownership du niveau 0 (parent reparti -> parallel_copy + fence).
/// C'est le CORPS de l'ancien amr_regrid_finest, sans le calcul du layout. Le pk du parent est passe
/// pour decider si le parent est replique (pk != 0 -> toujours reparti). Renvoie le nouveau MultiFab.
inline MultiFab regrid_field_on_layout(const BoxArray& fb, const DistributionMapping& dmap,
                                       const MultiFab& par, const MultiFab& old, int pk, int ngf,
                                       bool coarse_replicated = true) {
  MultiFab nU(fb, dmap, old.ncomp(), ngf);
  const int ncf = nU.ncomp();
  // Parent REPARTI (grossier de-replique) : par.fab ne contient que les boites LOCALES, donc
  // mf_find_box renverrait -1 pour une cellule grossiere possedee par un rang DISTANT et le patch
  // resterait non initialise la. On amene les regions parentes necessaires sur une grille
  // enfant-coarsen LOCALE (coarsen du BoxArray fin) par parallel_copy, puis on interpole depuis
  // elle. Parent replique : par est entierement local, lecture directe via mf_find_box.
  const bool par_replicated = (pk != 0) || coarse_replicated;
  MultiFab parloc;
  if (!par_replicated) {
    parloc = MultiFab(coarsen(nU.box_array(), 2), nU.dmap(), par.ncomp(), 0);
    parallel_copy(parloc, par);
    // parallel_copy lance des kernels async sous Cuda et, a np=1, retourne SANS fence : sans ce
    // fence la lecture de parloc ci-dessous lirait de la memoire device non encore ecrite -> NaN.
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
  return nU;
}

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
  // (1) Calcul du layout fin (tags -> grow [deja fait] -> all_reduce_or -> clustering -> clamp).
  auto [fb, dmap] = regrid_compute_fine_layout(std::move(grown), pdom, pk, margin, coarse_replicated);
  if (fb.size() == 0) return;  // rien a raffiner : on garde la grille courante
  // Les nouveaux patchs HERITENT la largeur de ghost du niveau remplace (et non un 1 fige) : un
  // niveau reconstruit en MUSCL ordre 2 (Minmod / VanLeer) porte 2 ghosts, que le regrid doit
  // preserver, sinon la reconstruction lirait hors bornes apres re-raffinement.
  const int ngf = L[fk].U.n_grow();
  // (2) Re-grille du champ U sur ce layout (interp parent + report fin) : MEME corps qu'avant,
  // donc le chemin mono-bloc reste BIT-IDENTIQUE (enchainement (1) puis (2) sur un seul bloc).
  L[fk].U = regrid_field_on_layout(fb, dmap, L[pk].U, L[fk].U, pk, ngf, coarse_replicated);
  aux[fk] = MultiFab(L[fk].U.box_array(), L[fk].U.dmap(), aux_ncomp, 1);  // adresse stable
  L[fk].aux = &aux[fk];
}

}  // namespace adc
