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
#include <pops/core/foundation/allocator.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/layout/copy_schedule.hpp>
#include <pops/mesh/layout/refinement.hpp>  // parallel_copy
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/parallel/load_balance.hpp>

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

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
  auto val = [&](int i, int j, int c) { return double(i) + 0.001 * double(j) + 100.0 * c; };
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
    dst.set_val(-1);
    parallel_copy(dst, src);  // cold materialization of schedule + pinned communication lease
    const AllocationEventStats before_hot = allocation_event_stats();
    for (int k = 1; k < K; ++k) {
      dst.set_val(-1);  // reset dst so each copy is exercised fresh
      parallel_copy(dst, src);
    }
    const AllocationEventStats after_hot = allocation_event_stats();
    chk(count_wrong(dst) == 0, "cache_on_correct");
    chk(copy_schedule_build_count() == 1, "cache_built_once");
    chk(dst.copy_cache().size() == 1, "cache_one_entry");
    chk(copy_schedule_miss_count() == 1, "one_miss");
    chk(copy_schedule_hit_count() == K - 1, "k_minus_one_hits");
    chk(after_hot.communication_calls == before_hot.communication_calls,
        "hot_path_reuses_pinned_buffers");
    chk(dst.copy_cache().exchange_pool_size() == (np > 1 ? 1u : 0u),
        "one_world_communicator_lease");
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
    parallel_copy(dst, src);               // cache abandonne -> reconstruction pour (dst #2, src)
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

#ifdef POPS_HAS_MPI
  // The same field pair on two duplicated execution lanes receives independent schedule/buffer
  // identities. Repeating either lane is allocation-free and reuses only its own prepared lease.
  if (np > 1) {
    MultiFab src(sba, sdm, ncomp, 0);
    MultiFab dst(dba, ddm, ncomp, 0);
    set_valid(src);
    ExecutionLane lane_a = ExecutionLane::duplicate_world_collectively("copy-cache-lane-a");
    ExecutionLane lane_b = ExecutionLane::duplicate_world_collectively("copy-cache-lane-b");
    reset_copy_schedule_build_count();
    parallel_copy(dst, src);
    parallel_copy(dst, src, lane_a);
    parallel_copy(dst, src, lane_b);
    const AllocationEventStats before_hot = allocation_event_stats();
    parallel_copy(dst, src);
    parallel_copy(dst, src, lane_a);
    parallel_copy(dst, src, lane_b);
    const AllocationEventStats after_hot = allocation_event_stats();
    chk(copy_schedule_build_count() == 3, "one_schedule_per_communicator_context");
    chk(dst.copy_cache().size() == 3, "three_communicator_schedule_entries");
    chk(dst.copy_cache().exchange_pool_size() == 3, "three_communicator_buffer_leases");
    chk(after_hot.communication_calls == before_hot.communication_calls,
        "multi_lane_hot_path_reuses_pinned_buffers");
    chk(count_wrong(dst) == 0, "multi_lane_copy_correct");
  }
#endif

  // A rank-local contract error is converted into one collective exception before any request is
  // posted. All peers remain live and a subsequent valid redistribution succeeds.
  if (np > 1) {
    MultiFab src(sba, sdm, ncomp, 0);
    MultiFab dst(dba, ddm, me == 0 ? ncomp + 1 : ncomp, 0);
    set_valid(src);
    bool rejected = false;
    try {
      parallel_copy(dst, src);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    chk(all_reduce_sum(rejected ? 1L : 0L) == np,
        "rank_local_width_error_rejected_collectively");

    MultiFab valid_dst(dba, ddm, ncomp, 0);
    parallel_copy(valid_dst, src);
    chk(count_wrong(valid_dst) == 0, "communicator_live_after_collective_rejection");

    const BoxArray divergent_boxes =
        me == 0 ? BoxArray::from_domain(dom, 8) : BoxArray::from_domain(dom, 16);
    MultiFab divergent_src(divergent_boxes, make_sfc_distribution(divergent_boxes, np), ncomp, 0);
    MultiFab common_dst(dba, ddm, ncomp, 0);
    set_valid(divergent_src);
    rejected = false;
    try {
      parallel_copy(common_dst, divergent_src);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    chk(all_reduce_sum(rejected ? 1L : 0L) == np,
        "rank_local_layout_error_rejected_collectively");
    parallel_copy(common_dst, src);
    chk(count_wrong(common_dst) == 0, "communicator_live_after_layout_rejection");
  }

  // The portable MPI count guard is independent of storage allocation: an oversized synthetic
  // schedule is rejected before any Fab or communication buffer needs to exist.
  {
    bool rejected = false;
    try {
      (void)detail::checked_parallel_copy_payload_count(
          static_cast<std::int64_t>(std::numeric_limits<int>::max()), 2);
    } catch (const std::overflow_error&) {
      rejected = true;
    }
    chk(rejected, "portable_int_count_guard");
  }

  // A periodic parent carrier owns its image catalogue and warmed redistribution buffers across
  // stage replays. Cover a rectangular, non-zero-origin domain and deep images on both axes; the
  // hot path must allocate neither Fab storage nor communication storage.
  {
    const Box2D periodic_domain{{-11, 7}, {20, 22}};  // 32 x 16
    const BoxArray periodic_source_boxes = BoxArray::from_domain(periodic_domain, 8);
    const DistributionMapping periodic_source_mapping =
        make_sfc_distribution(periodic_source_boxes, np);
    const BoxArray carrier_boxes({Box2D{{-45, -10}, {54, 39}}});
    const DistributionMapping carrier_mapping = make_sfc_distribution(carrier_boxes, np);
    MultiFab periodic_source(periodic_source_boxes, periodic_source_mapping, ncomp, 0);
    MultiFab carrier(carrier_boxes, carrier_mapping, ncomp, 0);
    auto periodic_value = [](int i, int j, int component) {
      return Real(3 * i - 5 * j + 1000 * component);
    };
    for (int local = 0; local < periodic_source.local_size(); ++local) {
      Fab2D& fab = periodic_source.fab(local);
      const Box2D box = fab.box();
      for (int component = 0; component < ncomp; ++component)
        for (int j = box.lo[1]; j <= box.hi[1]; ++j)
          for (int i = box.lo[0]; i <= box.hi[0]; ++i)
            fab(i, j, component) = periodic_value(i, j, component);
    }
    const CommunicatorView communicator = world_communicator_view();
    auto periodic_plan = PreparedPeriodicCopyPlan::prepare(
        carrier, periodic_source, periodic_domain, Periodicity{true, true},
        /*topology_generation=*/17, communicator);
    const AllocationEventStats before_replay = allocation_event_stats();
    for (int replay = 0; replay < 3; ++replay)
      periodic_plan.apply(carrier, periodic_source, /*topology_generation=*/17,
                          communicator);
    const AllocationEventStats after_replay = allocation_event_stats();
    chk(after_replay == before_replay, "prepared_periodic_copy_hot_path_allocation_free");

    auto wrap = [](int index, int lo, int extent) {
      int relative = (index - lo) % extent;
      if (relative < 0)
        relative += extent;
      return lo + relative;
    };
    long periodic_wrong = 0;
    carrier.sync_host();
    for (int local = 0; local < carrier.local_size(); ++local) {
      const Fab2D& fab = carrier.fab(local);
      const Box2D box = fab.box();
      for (int component = 0; component < ncomp; ++component)
        for (int j = box.lo[1]; j <= box.hi[1]; ++j)
          for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
            const int wrapped_i = wrap(i, periodic_domain.lo[0], periodic_domain.nx());
            const int wrapped_j = wrap(j, periodic_domain.lo[1], periodic_domain.ny());
            if (fab(i, j, component) != periodic_value(wrapped_i, wrapped_j, component))
              ++periodic_wrong;
          }
    }
    chk(all_reduce_sum(periodic_wrong) == 0,
        "prepared_periodic_copy_rectangular_nonzero_origin_exact");

    bool generation_rejected = false;
    try {
      periodic_plan.apply(carrier, periodic_source,
                          me == 0 ? 18u : 17u, communicator);
    } catch (const std::invalid_argument&) {
      generation_rejected = true;
    }
    chk(all_reduce_sum(generation_rejected ? 1L : 0L) == np,
        "prepared_periodic_copy_generation_rejected_collectively");
    periodic_plan.apply(carrier, periodic_source, /*topology_generation=*/17,
                        communicator);

    const BoxArray overlapping_source_boxes(
        {periodic_domain, Box2D{{-3, 10}, {4, 14}}});
    MultiFab overlapping_source(
        overlapping_source_boxes,
        make_sfc_distribution(overlapping_source_boxes, np), ncomp, 0);
    MultiFab overlap_destination(carrier_boxes, carrier_mapping, ncomp, 0);
    bool overlap_rejected = false;
    try {
      (void)PreparedPeriodicCopyPlan::prepare(
          overlap_destination, overlapping_source, periodic_domain,
          Periodicity{true, true}, /*topology_generation=*/19, communicator);
    } catch (const std::invalid_argument&) {
      overlap_rejected = true;
    }
    chk(all_reduce_sum(overlap_rejected ? 1L : 0L) == np,
        "prepared_periodic_copy_rejects_ambiguous_source_images");

    const BoxArray incomplete_source_boxes(
        {Box2D{{periodic_domain.lo[0], periodic_domain.lo[1]},
               {periodic_domain.hi[0] - 1, periodic_domain.hi[1]}}});
    MultiFab incomplete_source(
        incomplete_source_boxes,
        make_sfc_distribution(incomplete_source_boxes, np), ncomp, 0);
    MultiFab incomplete_destination(carrier_boxes, carrier_mapping, ncomp, 0);
    bool incomplete_rejected = false;
    try {
      (void)PreparedPeriodicCopyPlan::prepare(
          incomplete_destination, incomplete_source, periodic_domain,
          Periodicity{true, true}, /*topology_generation=*/20, communicator);
    } catch (const std::invalid_argument&) {
      incomplete_rejected = true;
    }
    chk(all_reduce_sum(incomplete_rejected ? 1L : 0L) == np,
        "prepared_periodic_copy_rejects_incomplete_source_coverage");
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
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_copy_schedule_cache, "test_copy_schedule_cache"),
            0);
}
