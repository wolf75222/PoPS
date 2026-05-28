#pragma once

#include <adc/mesh/box2d.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>

#include <utility>
#include <vector>

// fill_boundary : echange de halos intra-niveau. Remplit les ghosts de chaque
// Fab depuis les regions valides des boxes voisines de la meme MultiFab, avec
// wrapping periodique optionnel.
//
// Structure prete pour MPI : le seul point a changer plus tard est l'acces a la
// box source (memoire locale aujourd'hui, send/recv MPI ensuite). La logique
// (boucle sur les voisins, intersection, copie decalee) reste identique.
//
// Les ghosts qui tombent hors du domaine sans wrapping periodique ne sont pas
// touches ici : ce sont les conditions aux limites physiques (etape suivante).

namespace adc {

struct Periodicity {
  bool x = false;
  bool y = false;
};

namespace detail {

// dst(i, j, c) = src(i - sx, j - sy, c) pour (i, j) dans region.
inline void copy_shifted(Fab2D& dst, const Fab2D& src, const Box2D& region,
                         int sx, int sy, int ncomp) {
  Array4 d = dst.array();
  ConstArray4 s = src.const_array();
  for (int c = 0; c < ncomp; ++c)
    for_each_cell(region,
                  [=](int i, int j) { d(i, j, c) = s(i - sx, j - sy, c); });
}

}  // namespace detail

inline void fill_boundary(MultiFab& mf, const Box2D& domain,
                          Periodicity per = {}) {
  const int ng = mf.n_grow();
  if (ng == 0) return;
  const int nc = mf.ncomp();
  const int Lx = domain.nx();
  const int Ly = domain.ny();
  const BoxArray& ba = mf.box_array();

  std::vector<int> sxv = {0};
  if (per.x) {
    sxv.push_back(Lx);
    sxv.push_back(-Lx);
  }
  std::vector<int> syv = {0};
  if (per.y) {
    syv.push_back(Ly);
    syv.push_back(-Ly);
  }
  std::vector<std::pair<int, int>> shifts;
  for (int sx : sxv)
    for (int sy : syv) shifts.push_back({sx, sy});

  for (int li = 0; li < mf.local_size(); ++li) {
    Fab2D& F = mf.fab(li);
    const int gF = mf.global_index(li);
    const Box2D gbox = F.box().grow(ng);

    for (int gB = 0; gB < ba.size(); ++gB) {
      const int srcLocal = mf.local_index_of(gB);
      if (srcLocal < 0) continue;  // box non locale (MPI plus tard)
      const Fab2D& S = mf.fab(srcLocal);
      const Box2D vB = ba[gB];

      for (auto [sx, sy] : shifts) {
        if (gB == gF && sx == 0 && sy == 0) continue;  // soi-meme, sans decalage
        const Box2D src = vB.shift(0, sx).shift(1, sy);
        const Box2D region = gbox.intersect(src);
        if (region.empty()) continue;
        detail::copy_shifted(F, S, region, sx, sy, nc);
      }
    }
  }
}

}  // namespace adc
