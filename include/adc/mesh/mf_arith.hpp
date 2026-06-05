#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/parallel/comm.hpp>  // all_reduce_sum : produit scalaire COLLECTIF (Krylov sous MPI)

#include <algorithm>

// Combinaisons lineaires de MultiFab sur les cellules valides, pour les etages
// des integrateurs. Suppose des layouts identiques (meme BoxArray, meme
// DistributionMapping). Operations point a point, donc l'aliasing (x ou y == z)
// est sans danger.

namespace adc {

namespace detail {
// FONCTEURS NOMMES (et non lambdas ADC_HD) pour les kernels d'arithmetique MultiFab. Meme recette que
// le chemin block (#64) : ces operations sont premiere-instanciees depuis le V-cycle MG, lui-meme tire
// depuis une TU externe (harness/loader natif) ; une lambda etendue a cette place fait buter nvcc sur
// l'emission du kernel device (kernel-stub nul -> segfault Cuda a -O Release sans -g, #93). Corps
// strictement identique aux anciennes lambdas -> bit-identique sur CPU et device.
struct SaxpyKernel {
  Array4 Y;
  ConstArray4 X;
  Real a;
  int c;
  ADC_HD void operator()(int i, int j) const { Y(i, j, c) += a * X(i, j, c); }
};

struct LincombKernel {
  Array4 Z;
  ConstArray4 X, Y;
  Real a, b;
  int c;
  ADC_HD void operator()(int i, int j) const {
    Z(i, j, c) = a * X(i, j, c) + b * Y(i, j, c);
  }
};

// Reducteur |f(i,j,comp)| -> max, passe DIRECTEMENT a reduce_max_cell (aucune lambda etendue
// d'enveloppe, a la difference de for_each_cell_reduce_max). C'est le chemin device-clean documente
// dans for_each.hpp. Signature reducteur (i, j, Real& acc) ; meme Kokkos::Max / meme boucle hote
// sequentielle -> bit-identique a l'ancien norm_inf (max et fabs sans arrondi).
struct NormInfKernel {
  ConstArray4 a;
  int comp;
  ADC_HD void operator()(int i, int j, Real& acc) const {
    const Real v = a(i, j, comp);
    const Real av = v < 0 ? -v : v;
    if (av > acc) acc = av;
  }
};

// Reducteur x(i,j,comp) * y(i,j,comp) -> somme, passe DIRECTEMENT a reduce_sum_cell (aucune lambda
// etendue d'enveloppe). Foncteur NOMME device-clean (meme recette que NormInfKernel) pour le produit
// scalaire du solveur de Krylov, tire d'une TU externe. Signature reducteur (i, j, Real& acc).
struct DotKernel {
  ConstArray4 x, y;
  int comp;
  ADC_HD void operator()(int i, int j, Real& acc) const {
    acc += x(i, j, comp) * y(i, j, comp);
  }
};
}  // namespace detail

// y <- y + a x
inline void saxpy(MultiFab& y, Real a, const MultiFab& x) {
  const int nc = y.ncomp();
  for (int li = 0; li < y.local_size(); ++li) {
    Array4 Y = y.fab(li).array();
    const ConstArray4 X = x.fab(li).const_array();
    const Box2D b = y.fab(li).box();
    for (int c = 0; c < nc; ++c)
      for_each_cell(b, detail::SaxpyKernel{Y, X, a, c});
  }
}

// norme infinie sur les cellules valides d'une composante. Chaque fab local est
// reduit par for_each_cell_reduce_max sur |f(i,j,comp)| (vraie reduction device
// sous Kokkos, boucle hote en serie/OpenMP), agrege par max hote sur les fabs.
//
// Plus de device_fence() en tete : sous Kokkos parallel_reduce est bloquant et
// absorbe la barriere. EXACT partout : max et fabs sont sans arrondi et le max
// est associatif/commutatif en IEEE754, donc bit-identique a l'ancien norm_inf
// quel que soit le backend (l'ordre de reduction ne change aucun bit).
inline Real norm_inf(const MultiFab& mf, int comp = 0) {
  Real m = 0;
  for (int li = 0; li < mf.local_size(); ++li) {
    const ConstArray4 a = mf.fab(li).const_array();
    m = std::max(m, reduce_max_cell(mf.box(li), detail::NormInfKernel{a, comp}));
  }
  return m;  // all-reduce max MPI plus tard (iso-comportement, non ajoute ici)
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
      for_each_cell(bb, detail::LincombKernel{Z, X, Y, a, b, c});
  }
}

// Produit scalaire sum_cells x . y sur les cellules VALIDES de la composante comp, reduit sur tous
// les rangs (all-reduce). Brique des solveurs de Krylov (BiCGStab : rho, alpha, omega, betas). Chaque
// fab local est reduit par reduce_sum_cell (vraie reduction device sous Kokkos, boucle hote en
// serie/OpenMP), les fabs locaux agreges par somme hote, puis all_reduce_sum agrege les rangs.
//
// COLLECTIF, OBLIGATOIRE SOUS MPI : all_reduce_sum est appele sur CHAQUE rang, y compris un rang
// SANS box (local_size()==0, qui contribue alors 0 a la somme locale). Sans cet appel sur tous les
// rangs, MPI_Allreduce interbloque (collective desynchronisee) ; le solveur de Krylov ne doit donc
// JAMAIS court-circuiter dot() sur un rang vide. En serie all_reduce_sum est l'identite.
//
// NOTE FP (comme sum()) : sous Kokkos l'ordre de sommation par tuile differe de la boucle hote, donc
// dot n'est pas bit-identique entre backends ; serie et OpenMP restent exacts. Sous MPI, l'all-reduce
// rend la MEME valeur a tous les rangs (MPI_SUM sur un meme jeu de contributions locales), donc le
// critere d'arret du Krylov se declenche a la MEME iteration partout (pas de desynchronisation).
inline Real dot(const MultiFab& x, const MultiFab& y, int comp = 0) {
  Real s = 0;
  for (int li = 0; li < x.local_size(); ++li) {
    const ConstArray4 X = x.fab(li).const_array();
    const ConstArray4 Y = y.fab(li).const_array();
    s += reduce_sum_cell(x.box(li), detail::DotKernel{X, Y, comp});
  }
  return static_cast<Real>(all_reduce_sum(static_cast<double>(s)));
}

}  // namespace adc
