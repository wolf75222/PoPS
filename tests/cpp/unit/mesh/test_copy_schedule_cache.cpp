// Cache du schedule de redistribution de parallel_copy (ADC-607), calque sur test_fill_boundary_cache.
//   - reutilisation BIT-IDENTIQUE a une reconstruction a chaque appel (cache ON == cache OFF) ;
//   - le schedule est construit UNE fois puis reutilise sur K copies (engagement, compteur de builds) ;
//   - il est reconstruit (invalide) quand la LAYOUT SRC change (empreinte cle), et abandonne quand le
//     MultiFab DST entier est reassigne (style regrid AMR) ;
//   - hits/misses coherents (K copies stables -> 1 miss + K-1 hits).
// Invariant au nombre de rangs : couvre le chemin local (np=1) comme le chemin MPI (np>1) : la
// redistribution est bit-identique et le cache s'engage sur les deux.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/layout/copy_schedule.hpp>
#include <pops/mesh/layout/refinement.hpp>  // parallel_copy
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/load_balance.hpp>

#include <cmath>
#include <cstdio>

using namespace pops;

static int pops_run_test_copy_schedule_cache(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank(), np = n_ranks();
  long fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL[rank %d] %s\n", me, w);
      ++fails;
    }
  };

  const int L = 64, ncomp = 2;
  const Box2D dom = Box2D::from_extents(L, L);
  auto val = [&](int i, int j, int c) {
    return double(i) + 0.001 * double(j) + 100.0 * c;
  };
  // src / dst are two DIFFERENT decompositions of the SAME domain -> parallel_copy redistributes.
  const BoxArray sba = BoxArray::from_domain(dom, 16);  // 4x4 = 16 boxes
  const BoxArray dba = BoxArray::from_domain(dom, 32);  // 2x2 = 4 boxes
  const DistributionMapping sdm = make_sfc_distribution(sba, np);
  const DistributionMapping ddm = make_sfc_distribution(dba, np);

  // fills the VALID cells of a src fab with val(...).
  auto set_valid = [&](MultiFab& mf) {
    for (int li = 0; li < mf.local_size(); ++li) {
      Fab2D& F = mf.fab(li);
      const Box2D b = F.box();
      for (int c = 0; c < ncomp; ++c)
        for (int j = b.lo[1]; j <= b.hi[1]; ++j)
          for (int i = b.lo[0]; i <= b.hi[0]; ++i)
            F(i, j, c) = val(i, j, c);
    }
  };
  // number of dst VALID cells whose value != val(...), reduced over all ranks. parallel_copy over
  // two full tilings of the same domain covers every dst cell, so a correct copy leaves 0 wrong.
  auto count_wrong = [&](const MultiFab& mf) {
    long w = 0;
    for (int li = 0; li < mf.local_size(); ++li) {
      const Fab2D& F = mf.fab(li);
      const Box2D b = F.box();
      for (int c = 0; c < ncomp; ++c)
        for (int j = b.lo[1]; j <= b.hi[1]; ++j)
          for (int i = b.lo[0]; i <= b.hi[0]; ++i)
            if (std::fabs(F(i, j, c) - val(i, j, c)) > 1e-12)
              ++w;
    }
    return all_reduce_sum(w);
  };

  const int K = 5;

  // (1)+(2) cache ON : K copies sur une paire de layouts stable. Schedule construit UNE fois, dst
  // correct, hits/misses coherents.
  {
    MultiFab src(sba, sdm, ncomp, 0);
    MultiFab dst(dba, ddm, ncomp, 0);
    set_valid(src);
    reset_copy_schedule_build_count();
    for (int k = 0; k < K; ++k) {
      dst.set_val(-1);  // reset dst so each copy is exercised fresh
      parallel_copy(dst, src);
    }
    chk(count_wrong(dst) == 0, "cache_on_correct");
    chk(copy_schedule_build_count() == 1, "cache_built_once");
    chk(dst.copy_cache().size() == 1, "cache_one_entry");
    chk(copy_schedule_miss_count() == 1, "one_miss");
    chk(copy_schedule_hit_count() == K - 1, "k_minus_one_hits");
  }

  // (1') cache ON == cache OFF (clear() force la reconstruction a chaque appel) : BIT-IDENTIQUE.
  {
    MultiFab src(sba, sdm, ncomp, 0);
    MultiFab a(dba, ddm, ncomp, 0), b(dba, ddm, ncomp, 0);
    set_valid(src);
    a.set_val(-1);
    b.set_val(-1);
    for (int k = 0; k < K; ++k)
      parallel_copy(a, src);  // cache reutilise
    reset_copy_schedule_build_count();
    for (int k = 0; k < K; ++k) {  // reconstruit a chaque appel
      b.copy_cache().clear();
      parallel_copy(b, src);
    }
    chk(copy_schedule_build_count() == K, "cache_off_rebuilds_each_call");
    long diff = 0;
    for (int li = 0; li < a.local_size(); ++li) {
      const Fab2D& FA = a.fab(li);
      const Fab2D& FB = b.fab(li);
      const Box2D bx = FA.box();
      for (int c = 0; c < ncomp; ++c)
        for (int j = bx.lo[1]; j <= bx.hi[1]; ++j)
          for (int i = bx.lo[0]; i <= bx.hi[0]; ++i)
            if (FA(i, j, c) != FB(i, j, c))
              ++diff;  // egalite EXACTE (0 ulp)
    }
    chk(all_reduce_sum(diff) == 0, "cache_on_equals_rebuild_bit_identical");
  }

  // (3) invalidation par LAYOUT SRC : deux src de decoupages differents vers le MEME dst construisent
  // deux entrees ; la meme src reutilisee n'en construit pas.
  {
    MultiFab src1(sba, sdm, ncomp, 0);
    const BoxArray sba2 = BoxArray::from_domain(dom, 8);  // 8x8 = 64 boxes (autre decoupage src)
    const DistributionMapping sdm2 = make_sfc_distribution(sba2, np);
    MultiFab src2(sba2, sdm2, ncomp, 0);
    MultiFab dst(dba, ddm, ncomp, 0);
    set_valid(src1);
    set_valid(src2);
    reset_copy_schedule_build_count();
    parallel_copy(dst, src1);  // build #1 (src layout A)
    parallel_copy(dst, src1);  // reutilise A
    chk(copy_schedule_build_count() == 1, "same_src_reused");
    parallel_copy(dst, src2);  // src layout B -> build #2
    chk(copy_schedule_build_count() == 2, "diff_src_rebuilds");
    parallel_copy(dst, src1);  // A a nouveau -> reutilise (deux entrees coexistent)
    chk(copy_schedule_build_count() == 2, "first_src_still_cached");
    chk(dst.copy_cache().size() == 2, "two_distinct_entries");
    chk(count_wrong(dst) == 0, "still_correct_after_switch");  // dernier copy = src1 (val complet)
  }

  // (3') invalidation par reassignation du DST entier (style regrid AMR) : reassigner le MultiFab dst
  // abandonne son cache, donc la prochaine copie reconstruit sur la NOUVELLE layout dst.
  {
    MultiFab src(sba, sdm, ncomp, 0);
    MultiFab dst(dba, ddm, ncomp, 0);
    set_valid(src);
    reset_copy_schedule_build_count();
    parallel_copy(dst, src);  // build pour (dst #1, src)
    chk(copy_schedule_build_count() == 1, "regrid_pre");
    const BoxArray dba2 = BoxArray::from_domain(dom, 16);  // autre decoupage dst
    const DistributionMapping ddm2 = make_sfc_distribution(dba2, np);
    dst = MultiFab(dba2, ddm2, ncomp, 0);  // move-assign d'un dst frais (regrid)
    parallel_copy(dst, src);  // cache abandonne -> reconstruction pour (dst #2, src)
    chk(copy_schedule_build_count() == 2, "regrid_invalidates_cache");
    chk(count_wrong(dst) == 0, "regrid_new_layout_correct");
  }

  // (3'') copy-sharing : a MultiFab DST copy shares the cache via its shared_ptr, so copying from the
  // SAME src into the copy reuses the schedule (no rebuild) and is correct -- the jobs carry global
  // indices resolved against the copy's identical dst layout.
  {
    MultiFab src(sba, sdm, ncomp, 0);
    MultiFab dst(dba, ddm, ncomp, 0);
    set_valid(src);
    parallel_copy(dst, src);  // populate dst's copy cache
    MultiFab cp = dst;        // copy: shares the shared_ptr cache (same dst layout)
    reset_copy_schedule_build_count();
    cp.set_val(-1);
    parallel_copy(cp, src);  // shared cache HIT -> no rebuild
    chk(copy_schedule_build_count() == 0, "copy_shares_cache_no_rebuild");
    chk(count_wrong(cp) == 0, "copy_correct");
  }

  const long gfails = all_reduce_sum(fails);
  if (me == 0) {
    if (gfails == 0)
      std::printf("OK test_copy_schedule_cache (np=%d)\n", np);
    else
      std::printf("FAIL test_copy_schedule_cache : %ld checks (np=%d)\n", gfails, np);
  }
  comm_finalize();
  return gfails == 0 ? 0 : 1;
}

TEST(test_copy_schedule_cache, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_copy_schedule_cache, "test_copy_schedule_cache"), 0);
}
