#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>

#include <algorithm>

// Combinaisons lineaires de MultiFab sur les cellules valides, pour les etages
// des integrateurs. Suppose des layouts identiques (meme BoxArray, meme
// DistributionMapping). Operations point a point, donc l'aliasing (x ou y == z)
// est sans danger.

namespace adc {

// y <- y + a x
inline void saxpy(MultiFab& y, Real a, const MultiFab& x) {
  const int nc = y.ncomp();
  for (int li = 0; li < y.local_size(); ++li) {
    Array4 Y = y.fab(li).array();
    const ConstArray4 X = x.fab(li).const_array();
    const Box2D b = y.fab(li).box();
    for (int c = 0; c < nc; ++c)
      for_each_cell(b, [=] ADC_HD(int i, int j) { Y(i, j, c) += a * X(i, j, c); });
  }
}

// norme infinie sur les cellules valides d'une composante
inline Real norm_inf(const MultiFab& mf, int comp = 0) {
  device_fence();  // GPU : mf a pu etre ecrit par un kernel ; barriere avant
                   // la reduction hote sur la memoire unifiee.
  Real m = 0;
  for (int li = 0; li < mf.local_size(); ++li) {
    const Fab2D& f = mf.fab(li);
    const Box2D b = f.box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        const Real a = f(i, j, comp);
        m = std::max(m, a < 0 ? -a : a);
      }
  }
  return m;  // all-reduce max MPI plus tard
}

// z <- a x + b y
inline void lincomb(MultiFab& z, Real a, const MultiFab& x, Real b,
                    const MultiFab& y) {
  const int nc = z.ncomp();
  for (int li = 0; li < z.local_size(); ++li) {
    Array4 Z = z.fab(li).array();
    const ConstArray4 X = x.fab(li).const_array();
    const ConstArray4 Y = y.fab(li).const_array();
    const Box2D bb = z.fab(li).box();
    for (int c = 0; c < nc; ++c)
      for_each_cell(bb, [=] ADC_HD(int i, int j) {
        Z(i, j, c) = a * X(i, j, c) + b * Y(i, j, c);
      });
  }
}

}  // namespace adc
