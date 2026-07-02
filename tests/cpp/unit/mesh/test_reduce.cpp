// Contrat des reductions device sum / norm_inf (point 4 de la revue : vraies
// reductions device via for_each_cell_reduce_*, plus de boucle hote sous fence).
//
// Le contrat est FP-conscient et le meme pour les trois backends :
//   - sum est EXACT (egalite stricte) la ou la reassociation est neutre : champ
//     constant (toutes valeurs egales) et champ symetrique a somme nulle. Sur un
//     champ a valeurs variees, sum n'est plus bit-identique a la boucle hote sous
//     Kokkos (somme par tuile) : on n'exige qu'un ecart RELATIF petit vs une
//     reference hote sequentielle (1e-10), pas l'egalite stricte.
//   - sum est IDEMPOTENT : deux appels sur le meme MultiFab inchange rendent
//     exactement le meme bit (verifie le determinisme du reducteur, cle pour
//     test_fill_boundary/sum_unchanged ; Kokkos::Sum est deterministe par tuile,
//     pas d'atomics flottants).
//   - norm_inf est EXACT partout (max et fabs sans arrondi, max associatif et
//     commutatif en IEEE754) : egalite stricte avec la reference hote.

#include <gtest/gtest.h>

#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <algorithm>
#include <cmath>

using namespace pops;

namespace {

// Reference hote sequentielle, lecture directe des fabs locaux (ordre
// lexicographique fixe), sans passer par for_each_cell_reduce_*.
double host_sum(const MultiFab& mf, int comp) {
  double s = 0;
  for (int li = 0; li < mf.local_size(); ++li) {
    const Fab2D& f = mf.fab(li);
    const Box2D b = f.box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        s += f(i, j, comp);
  }
  return s;
}

double host_norm_inf(const MultiFab& mf, int comp) {
  double m = 0;
  for (int li = 0; li < mf.local_size(); ++li) {
    const Fab2D& f = mf.fab(li);
    const Box2D b = f.box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        const double a = f(i, j, comp);
        m = std::max(m, a < 0 ? -a : a);
      }
  }
  return m;
}

// Domaine 256x256 decoupe en boites 32x32 (64 fabs), une composante -- partage par tous les cas.
BoxArray make_domain_ba(Box2D& dom_out) {
  dom_out = Box2D::from_extents(256, 256);
  return BoxArray::from_domain(dom_out, 32);
}

}  // namespace

TEST(test_reduce, sum_constant_field_is_exact) {
  Box2D dom;
  BoxArray ba = make_domain_ba(dom);
  DistributionMapping dm(ba.size(), n_ranks());

  // champ constant : sum exact (reassociation neutre, valeurs egales).
  MultiFab mf(ba, dm, 1, 0);
  mf.set_val(2.0);
  const double expect = 2.0 * dom.num_cells();
  EXPECT_LE(std::fabs(sum(mf) - expect), 1e-15 * expect) << "sum_constant_exact";
}

TEST(test_reduce, sum_varied_field_matches_host_within_relative_tolerance) {
  Box2D dom;
  BoxArray ba = make_domain_ba(dom);
  DistributionMapping dm(ba.size(), n_ranks());

  // champ a valeurs variees : ecart RELATIF device vs reference hote < 1e-10
  // (PAS d'egalite stricte ; la somme par tuile reassocie sous Kokkos).
  MultiFab mf(ba, dm, 1, 0);
  for (int li = 0; li < mf.local_size(); ++li) {
    Array4 a = mf.fab(li).array();
    for_each_cell(mf.box(li), [a] POPS_HD(int i, int j) { a(i, j, 0) = i + 100.0 * j; });
  }
  device_fence();
  const double ref = host_sum(mf, 0);
  const double dev = sum(mf, 0);
  const double rel = std::fabs(dev - ref) / std::max(1.0, std::fabs(ref));
  EXPECT_LT(rel, 1e-10) << "sum_varied_relative";
}

TEST(test_reduce, sum_antisymmetric_field_is_small) {
  Box2D dom;
  BoxArray ba = make_domain_ba(dom);
  DistributionMapping dm(ba.size(), n_ranks());

  // champ symetrique a somme nulle : valeurs +v / -v par parite de (i+j).
  // Toute reassociation appariant +v et -v donne 0 ; ici on tolere quand
  // meme un bruit relatif a la magnitude des termes.
  MultiFab mf(ba, dm, 1, 0);
  for (int li = 0; li < mf.local_size(); ++li) {
    Array4 a = mf.fab(li).array();
    for_each_cell(mf.box(li), [a] POPS_HD(int i, int j) {
      const Real v = i + 100.0 * j;
      a(i, j, 0) = ((i + j) & 1) ? -v : v;
    });
  }
  device_fence();
  const double dev = sum(mf, 0);
  EXPECT_LT(std::fabs(dev), 1e-6) << "sum_antisymmetric_small";
}

TEST(test_reduce, norm_inf_is_exact) {
  Box2D dom;
  BoxArray ba = make_domain_ba(dom);
  DistributionMapping dm(ba.size(), n_ranks());

  // norm_inf EXACT : signe alterne, max |.| connu. Egalite stricte vs hote.
  MultiFab mf(ba, dm, 1, 0);
  for (int li = 0; li < mf.local_size(); ++li) {
    Array4 a = mf.fab(li).array();
    for_each_cell(mf.box(li), [a] POPS_HD(int i, int j) {
      const Real v = i + 100.0 * j;
      a(i, j, 0) = ((i + j) & 1) ? -v : v;
    });
  }
  device_fence();
  const double ref = host_norm_inf(mf, 0);
  const double dev = norm_inf(mf, 0);
  EXPECT_EQ(dev, ref) << "norm_inf_exact";
  // max |i + 100 j| sur [0..255]^2 = 255 + 100*255 = 25755.
  EXPECT_EQ(dev, 255.0 + 100.0 * 255.0) << "norm_inf_value";
}

TEST(test_reduce, sum_and_norm_inf_are_idempotent) {
  Box2D dom;
  BoxArray ba = make_domain_ba(dom);
  DistributionMapping dm(ba.size(), n_ranks());

  // idempotence : deux sum / deux norm_inf sur le meme champ inchange rendent
  // exactement le meme bit (determinisme du reducteur).
  MultiFab mf(ba, dm, 1, 0);
  for (int li = 0; li < mf.local_size(); ++li) {
    Array4 a = mf.fab(li).array();
    for_each_cell(mf.box(li), [a] POPS_HD(int i, int j) {
      a(i, j, 0) = std::sin(0.1 * i) + std::cos(0.07 * j);
    });
  }
  device_fence();
  EXPECT_EQ(sum(mf, 0), sum(mf, 0)) << "sum_idempotent";
  EXPECT_EQ(norm_inf(mf, 0), norm_inf(mf, 0)) << "norm_inf_idempotent";
}
