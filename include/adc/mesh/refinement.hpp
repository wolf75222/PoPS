#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/fill_boundary.hpp>  // detail::copy_shifted
#include <adc/mesh/multifab.hpp>

#include <utility>
#include <vector>

// Operateurs de transfert entre niveaux AMR (ratio entier r).
//
//   average_down : fin -> grossier, moyenne conservative sur les blocs r x r
//                  (une cellule grossiere = moyenne des r^2 cellules fines).
//   interpolate  : grossier -> fin, injection constante par morceaux (chaque
//                  cellule fine recoit la valeur de sa cellule grossiere).
//
// Les deux passent par un MultiFab temporaire "fin coarsen" partageant la
// DistributionMapping du fin : le calcul par bloc est alors local (pas de
// croisement de fabs), suivi d'un parallel_copy vers la cible. C'est le schema
// d'AMReX (average_down via une copie parallele), et parallel_copy resservira
// pour le regrid.

namespace adc {

inline int coarsen_index(int a, int r) {
  int q = a / r, rem = a % r;
  return (rem != 0 && ((rem < 0) != (r < 0))) ? q - 1 : q;
}

inline BoxArray coarsen(const BoxArray& ba, int r) {
  std::vector<Box2D> b;
  b.reserve(ba.size());
  for (int i = 0; i < ba.size(); ++i) b.push_back(ba[i].coarsen(r));
  return BoxArray{std::move(b)};
}

// Copie des regions valides qui se recouvrent : src -> dst (memes indices, pas
// de decalage). Brique de base ; la version MPI postera des send/recv pour les
// boxes non locales (aujourd'hui on saute simplement les boxes distantes).
inline void parallel_copy(MultiFab& dst, const MultiFab& src) {
  const int nc = std::min(dst.ncomp(), src.ncomp());
  const BoxArray& sba = src.box_array();
  for (int ld = 0; ld < dst.local_size(); ++ld) {
    Fab2D& D = dst.fab(ld);
    const Box2D vd = D.box();
    for (int gs = 0; gs < sba.size(); ++gs) {
      const int ls = src.local_index_of(gs);
      if (ls < 0) continue;
      const Box2D region = vd.intersect(sba[gs]);
      if (region.empty()) continue;
      detail::copy_shifted(D, src.fab(ls), region, 0, 0, nc);
    }
  }
}

inline void average_down(const MultiFab& fine, MultiFab& coarse, int r) {
  const int nc = std::min(fine.ncomp(), coarse.ncomp());
  MultiFab cfine(coarsen(fine.box_array(), r), fine.dmap(), fine.ncomp(), 0);
  const Real inv = Real(1) / (r * r);
  for (int li = 0; li < fine.local_size(); ++li) {
    const Fab2D& F = fine.fab(li);
    Fab2D& C = cfine.fab(li);
    const Box2D cb = C.box();
    for (int c = 0; c < nc; ++c)
      for (int J = cb.lo[1]; J <= cb.hi[1]; ++J)
        for (int I = cb.lo[0]; I <= cb.hi[0]; ++I) {
          Real s = 0;
          for (int b = 0; b < r; ++b)
            for (int a = 0; a < r; ++a) s += F(r * I + a, r * J + b, c);
          C(I, J, c) = s * inv;
        }
  }
  parallel_copy(coarse, cfine);
}

inline void interpolate(const MultiFab& coarse, MultiFab& fine, int r) {
  const int nc = std::min(fine.ncomp(), coarse.ncomp());
  MultiFab cfine(coarsen(fine.box_array(), r), fine.dmap(), fine.ncomp(), 0);
  parallel_copy(cfine, coarse);  // amene les valeurs grossieres sur la grille fine-coarsen
  for (int li = 0; li < fine.local_size(); ++li) {
    Fab2D& F = fine.fab(li);
    const Fab2D& C = cfine.fab(li);
    const Box2D fb = F.box();
    for (int c = 0; c < nc; ++c)
      for (int j = fb.lo[1]; j <= fb.hi[1]; ++j)
        for (int i = fb.lo[0]; i <= fb.hi[0]; ++i)
          F(i, j, c) = C(coarsen_index(i, r), coarsen_index(j, r), c);
  }
}

}  // namespace adc
