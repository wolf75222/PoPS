// Native MPI regression for the compiled-Program AMR reflux strip samplers.
//
// The level-0 distributed layout deliberately has four parent boxes while each
// child patch crosses a parent-box seam.  At np=2 the second child is owned by
// rank 1 with local index 0 and global index 1.  At np=4 ranks 2 and 3 own
// required parent source tiles but no child destination: the result therefore
// proves a real cross-rank redistribution rather than a coincidental local
// copy.  The test fails if a sampler reads parent fab(0), confuses a local
// index with the global patch identity, discards its persistent copy schedule,
// or lets every rank write a strip merely because the parent is replicated.
// The complete route_reflux_program transaction deliberately stays in the generated-Program MPI
// acceptance (tests/python/integration/mpi/test_amr_clean_route_program_mpi.py): constructing it here
// would duplicate AmrSystem/compiled-Program installation rather than test the native sampler seam in
// isolation.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr/amr_program_reflux.hpp>

#include <cstdio>
#include <utility>
#include <vector>

using namespace pops;

namespace {

constexpr int kNcomp = 2;

Real x_flux_value(int i, int j, int c) {
  return Real(100000 * c + 1000 * j + i) + Real(0.25);
}

Real y_flux_value(int i, int j, int c) {
  return Real(200000 + 100000 * c + 1000 * j + i) + Real(0.5);
}

BoxArray face_boxes(const BoxArray& cells, bool x_direction) {
  std::vector<Box2D> boxes;
  boxes.reserve(static_cast<std::size_t>(cells.size()));
  for (int g = 0; g < cells.size(); ++g)
    boxes.push_back(x_direction ? xface_box(cells[g]) : yface_box(cells[g]));
  return BoxArray(std::move(boxes));
}

void fill_faces(MultiFab& fx, MultiFab& fy) {
  for (int li = 0; li < fx.local_size(); ++li) {
    Fab2D& fab = fx.fab(li);
    const Box2D box = fab.box();
    for (int c = 0; c < kNcomp; ++c)
      for (int j = box.lo[1]; j <= box.hi[1]; ++j)
        for (int i = box.lo[0]; i <= box.hi[0]; ++i)
          fab(i, j, c) = x_flux_value(i, j, c);
  }
  for (int li = 0; li < fy.local_size(); ++li) {
    Fab2D& fab = fy.fab(li);
    const Box2D box = fab.box();
    for (int c = 0; c < kNcomp; ++c)
      for (int j = box.lo[1]; j <= box.hi[1]; ++j)
        for (int i = box.lo[0]; i <= box.hi[0]; ++i)
          fab(i, j, c) = y_flux_value(i, j, c);
  }
}

bool strip_is_empty(const EdgeStrip& strip) {
  return strip.cL.empty() && strip.cR.empty() && strip.cB.empty() && strip.cT.empty() &&
         strip.fL.empty() && strip.fR.empty() && strip.fB.empty() && strip.fT.empty();
}

long check_coarse_strip(const EdgeStrip& strip, const Box2D& fine_box) {
  const PatchRange range(fine_box);
  long fails = 0;
  fails +=
      strip.I0 != range.I0 || strip.I1 != range.I1 || strip.J0 != range.J0 || strip.J1 != range.J1;
  fails += strip.cL.size() != static_cast<std::size_t>((range.J1 - range.J0 + 1) * kNcomp);
  fails += strip.cB.size() != static_cast<std::size_t>((range.I1 - range.I0 + 1) * kNcomp);
  if (fails != 0)
    return fails;

  for (int j = range.J0; j <= range.J1; ++j)
    for (int c = 0; c < kNcomp; ++c) {
      const std::size_t q = static_cast<std::size_t>((j - range.J0) * kNcomp + c);
      fails += strip.cL[q] != x_flux_value(range.I0, j, c);
      fails += strip.cR[q] != x_flux_value(range.I1 + 1, j, c);
    }
  for (int i = range.I0; i <= range.I1; ++i)
    for (int c = 0; c < kNcomp; ++c) {
      const std::size_t q = static_cast<std::size_t>((i - range.I0) * kNcomp + c);
      fails += strip.cB[q] != y_flux_value(i, range.J0, c);
      fails += strip.cT[q] != y_flux_value(i, range.J1 + 1, c);
    }
  return fails;
}

long check_fine_strip(const EdgeStrip& strip, const Box2D& fine_box) {
  const PatchRange range(fine_box);
  long fails = 0;
  fails +=
      strip.I0 != range.I0 || strip.I1 != range.I1 || strip.J0 != range.J0 || strip.J1 != range.J1;
  fails += strip.fL.size() != static_cast<std::size_t>((range.J1 - range.J0 + 1) * kNcomp);
  fails += strip.fB.size() != static_cast<std::size_t>((range.I1 - range.I0 + 1) * kNcomp);
  if (fails != 0)
    return fails;

  for (int j = range.J0; j <= range.J1; ++j)
    for (int c = 0; c < kNcomp; ++c) {
      const std::size_t q = static_cast<std::size_t>((j - range.J0) * kNcomp + c);
      const Real left = Real(0.5) * (x_flux_value(2 * range.I0, 2 * j, c) +
                                     x_flux_value(2 * range.I0, 2 * j + 1, c));
      const Real right = Real(0.5) * (x_flux_value(2 * range.I1 + 2, 2 * j, c) +
                                      x_flux_value(2 * range.I1 + 2, 2 * j + 1, c));
      fails += strip.fL[q] != left;
      fails += strip.fR[q] != right;
    }
  for (int i = range.I0; i <= range.I1; ++i)
    for (int c = 0; c < kNcomp; ++c) {
      const std::size_t q = static_cast<std::size_t>((i - range.I0) * kNcomp + c);
      const Real bottom = Real(0.5) * (y_flux_value(2 * i, 2 * range.J0, c) +
                                       y_flux_value(2 * i + 1, 2 * range.J0, c));
      const Real top = Real(0.5) * (y_flux_value(2 * i, 2 * range.J1 + 2, c) +
                                    y_flux_value(2 * i + 1, 2 * range.J1 + 2, c));
      fails += strip.fB[q] != bottom;
      fails += strip.fT[q] != top;
    }
  return fails;
}

long check_one_writer_per_patch(const std::vector<EdgeStrip>& strips, bool coarse_role) {
  long fails = 0;
  for (std::size_t g = 0; g < strips.size(); ++g) {
    const long local_writer =
        coarse_role ? (!strips[g].cL.empty() ? 1L : 0L) : (!strips[g].fL.empty() ? 1L : 0L);
    const long writers = all_reduce_sum(local_writer);
    if (my_rank() == 0 && writers != 1)
      ++fails;
  }
  return fails;
}

}  // namespace

static int pops_run_test_mpi_amr_program_reflux(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank();
  const int np = n_ranks();
  long fails = 0;

  if (np != 2 && np != 4) {
    if (me == 0)
      std::printf("FAIL test_mpi_amr_program_reflux requires two or four MPI ranks\n");
    comm_finalize();
    return 1;
  }

  // Negative/non-zero origin: the coarse/fine formulas must remain in global index space and may
  // not silently fall back to zero-based modulo/division.
  const Box2D parent_domain{{-8, -8}, {7, 7}};
  const BoxArray parent_boxes = BoxArray::from_domain(parent_domain, 8);  // four parent tiles
  const DistributionMapping parent_dm(parent_boxes.size(), np);
  MultiFab parent(parent_boxes, parent_dm, kNcomp, 0);
  MultiFab distributed_fx(face_boxes(parent_boxes, true), parent_dm, kNcomp, 0);
  MultiFab distributed_fy(face_boxes(parent_boxes, false), parent_dm, kNcomp, 0);
  fill_faces(distributed_fx, distributed_fy);

  // Both fine patches cross the x=0 parent-tile seam.  Global child 1 belongs to rank 1,
  // where its local index is 0: this makes local/global index confusion observable.
  const BoxArray child_boxes(
      std::vector<Box2D>{Box2D{{-8, -16}, {7, -1}}, Box2D{{-8, 0}, {7, 15}}});
  const DistributionMapping child_dm(child_boxes.size(), np);
  MultiFab child(child_boxes, child_dm, kNcomp, 0);

  // The coarse-role redistribution target is persistent.  Its first use builds one cached schedule
  // per face direction, the next use hits those schedules, and an epoch change rebuilds both targets.
  // The process-local counters are deterministic because every rank participates in parallel_copy.
  detail::CoarseRoleScratch scratch;
  reset_copy_schedule_build_count();
  EdgeFlux distributed;
  scratch.prepare(parent, child, /*replicated=*/false, 7, kNcomp);
  detail::prepare_edge_flux_coarse_role(distributed, child, kNcomp);
  detail::sample_coarse_role_strip(parent, distributed_fx, distributed_fy, child, false, 7, kNcomp,
                                   scratch, distributed);
  fails += copy_schedule_build_count() != 2;
  const std::int64_t hits_after_first = copy_schedule_hit_count();
  const AllocationEventStats allocations_before_replay = allocation_event_stats();
  detail::sample_coarse_role_strip(parent, distributed_fx, distributed_fy, child, false, 7, kNcomp,
                                   scratch, distributed);
  const AllocationEventStats allocations_after_replay = allocation_event_stats();
  fails += !(allocations_before_replay == allocations_after_replay);
  fails += copy_schedule_build_count() != 2;
  fails += copy_schedule_hit_count() != hits_after_first + 2;
  scratch.prepare(parent, child, /*replicated=*/false, 8, kNcomp);
  detail::sample_coarse_role_strip(parent, distributed_fx, distributed_fy, child, false, 8, kNcomp,
                                   scratch, distributed);
  fails += copy_schedule_build_count() != 4;
  fails += scratch.topology_epoch != 8;
  fails += distributed.coarse.size() != static_cast<std::size_t>(child_boxes.size());
  if (distributed.coarse.size() == static_cast<std::size_t>(child_boxes.size())) {
    for (int g = 0; g < child_boxes.size(); ++g) {
      if (child_dm[g] == me)
        fails +=
            check_coarse_strip(distributed.coarse[static_cast<std::size_t>(g)], child_boxes[g]);
      else
        fails += !strip_is_empty(distributed.coarse[static_cast<std::size_t>(g)]);
    }
    fails += check_one_writer_per_patch(distributed.coarse, true);
  }
  if (np == 4) {
    const long source_only = parent.local_size() > 0 && child.local_size() == 0 ? 1L : 0L;
    const long source_only_ranks = all_reduce_sum(source_only);
    if (me == 0)
      fails += source_only_ranks != 2;
  }

  // Replicated parent: every rank owns a full parent copy, but child ownership remains the
  // single-writer authority.  Its local strip must be bit-identical to the distributed route.
  const BoxArray replicated_boxes(std::vector<Box2D>{parent_domain});
  const DistributionMapping replicated_dm(std::vector<int>{me});
  MultiFab replicated_parent(replicated_boxes, replicated_dm, kNcomp, 0);
  MultiFab replicated_fx(face_boxes(replicated_boxes, true), replicated_dm, kNcomp, 0);
  MultiFab replicated_fy(face_boxes(replicated_boxes, false), replicated_dm, kNcomp, 0);
  fill_faces(replicated_fx, replicated_fy);

  detail::CoarseRoleScratch replicated_scratch;
  EdgeFlux replicated;
  replicated_scratch.prepare(replicated_parent, child, /*replicated=*/true, 8, kNcomp);
  detail::prepare_edge_flux_coarse_role(replicated, child, kNcomp);
  detail::sample_coarse_role_strip(replicated_parent, replicated_fx, replicated_fy, child, true, 8,
                                   kNcomp, replicated_scratch, replicated);
  fails += replicated.coarse.size() != static_cast<std::size_t>(child_boxes.size());
  if (replicated.coarse.size() == static_cast<std::size_t>(child_boxes.size())) {
    for (int g = 0; g < child_boxes.size(); ++g) {
      const std::size_t q = static_cast<std::size_t>(g);
      if (child_dm[g] == me) {
        fails += check_coarse_strip(replicated.coarse[q], child_boxes[g]);
        fails += replicated.coarse[q].cL != distributed.coarse[q].cL;
        fails += replicated.coarse[q].cR != distributed.coarse[q].cR;
        fails += replicated.coarse[q].cB != distributed.coarse[q].cB;
        fails += replicated.coarse[q].cT != distributed.coarse[q].cT;
      } else {
        fails += !strip_is_empty(replicated.coarse[q]);
      }
    }
    fails += check_one_writer_per_patch(replicated.coarse, true);
  }

  // Fine-role sampling uses the same distributed child layout.  Rank 1 must write global slot 1,
  // never slot 0, even though this is its first local face Fab.
  MultiFab fine_fx(face_boxes(child_boxes, true), child_dm, kNcomp, 0);
  MultiFab fine_fy(face_boxes(child_boxes, false), child_dm, kNcomp, 0);
  fill_faces(fine_fx, fine_fy);
  EdgeFlux fine;
  detail::prepare_edge_flux_fine_role(fine, child, kNcomp);
  detail::sample_fine_role_strip(child, fine_fx, fine_fy, kNcomp, fine);
  fails += fine.fine.size() != static_cast<std::size_t>(child_boxes.size());
  if (fine.fine.size() == static_cast<std::size_t>(child_boxes.size())) {
    for (int g = 0; g < child_boxes.size(); ++g) {
      if (child_dm[g] == me)
        fails += check_fine_strip(fine.fine[static_cast<std::size_t>(g)], child_boxes[g]);
      else
        fails += !strip_is_empty(fine.fine[static_cast<std::size_t>(g)]);
    }
    fails += check_one_writer_per_patch(fine.fine, false);
  }

  // A face array with the right local size but the wrong centering/order must fail before sampling;
  // no local-index fallback is allowed after regridding or redistribution.
  MultiFab wrong_fy(face_boxes(child_boxes, true), child_dm, kNcomp, 0);
  bool rejected_wrong_layout = false;
  try {
    EdgeFlux invalid;
    detail::sample_fine_role_strip(child, fine_fx, wrong_fy, kNcomp, invalid);
  } catch (const std::runtime_error&) {
    rejected_wrong_layout = true;
  }
  fails += !rejected_wrong_layout;

  const long global_fails = all_reduce_sum(fails);
  if (me == 0) {
    if (global_fails == 0)
      std::printf(
          "OK test_mpi_amr_program_reflux np=%d (persistent remap, distributed multi-box, "
          "global patch identity, replicated-parent single writer)\n",
          np);
    else
      std::printf("FAIL test_mpi_amr_program_reflux: %ld contract violations\n", global_fails);
  }
  comm_finalize();
  return global_fails == 0 ? 0 : 1;
}

TEST(test_mpi_amr_program_reflux, Runs) {
  EXPECT_EQ(
      pops::test::RunTestBody(&pops_run_test_mpi_amr_program_reflux, "test_mpi_amr_program_reflux"),
      0);
}
