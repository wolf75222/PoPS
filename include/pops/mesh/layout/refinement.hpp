/// @file
/// @brief AMR inter-level transfer operators (integer ratio r) + parallel_copy.
///
/// average_down (fine -> coarse): CONSERVATIVE average over r x r blocks (one coarse cell
/// = average of the r^2 fine cells). interpolate (coarse -> fine): piecewise-CONSTANT
/// injection (each fine cell receives the value of its coarse cell). Both go through a
/// temporary "fine coarsen" MultiFab sharing the fine DistributionMapping (LOCAL per-block
/// computation), followed by a parallel_copy to the target (the AMReX scheme). parallel_copy:
/// general redistribution between two MultiFab over the SAME domain with possibly different
/// decompositions (the AMReX ParallelCopy); single-rank = direct copies, multi-rank = jobs
/// enumerated deterministically (tag 1).

#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstdio>

#include <pops/core/foundation/types.hpp>
#include <pops/core/foundation/validation.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/copy_schedule.hpp>  // memoized parallel_copy schedule (ADC-607)
#include <pops/mesh/index/box_hash.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>  // detail::copy_shifted
#include <pops/mesh/storage/multifab.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace pops {

namespace detail {

// Enumerates the parallel_copy schedule for (dst layout, src layout): the BoxHash build over the SRC
// BoxArray + the local (dst AND src local) and, under MPI with n_ranks()>1, the global (cross-rank
// send/recv) job lists. This is the per-call work that ADC-607 hoists out of parallel_copy; it runs
// ONCE per distinct (dst layout, src layout) pair and is then replayed from the cache. Jobs are
// produced in the SAME deterministic order as the legacy inline loops (local: dst-local x sorted src
// candidates; global: gd x sorted src candidates), so the packed buffers stay bit-identical and the
// per-rank send/recv lists stay aligned. Bumps the build counter (cache-engagement test hook).
inline void build_copy_schedule(const MultiFab& dst, const MultiFab& src, CopySchedule& sched) {
  ++copy_schedule_build_counter();
  const BoxArray& sba = src.box_array();
  const BoxArray& dba = dst.box_array();
  const BoxHash shash(sba, suggest_bin(sba));

  // --- local jobs (dst local AND src local), in dst-local order (== the inline loop) ---
  for (int ld = 0; ld < dst.local_size(); ++ld) {
    const int gd = dst.global_index(ld);
    const Box2D vd = dst.box(ld);
    for (int gs : shash.query(vd)) {
      const int ls = src.local_index_of(gs);
      if (ls < 0)
        continue;  // src not local -> MPI below
      const Box2D region = vd.intersect(sba[gs]);
      if (region.empty())
        continue;
      sched.local.push_back({gs, gd, region});
    }
  }

#ifdef POPS_HAS_MPI
  const int np = n_ranks();
  if (np > 1) {
    const int me = my_rank();
    const DistributionMapping& sdm = src.dmap();
    const DistributionMapping& ddm = dst.dmap();
    sched.send.assign(np, {});
    sched.recv.assign(np, {});
    // deterministic global enumeration: (dst gd) x sorted src candidates. Identical on all ranks
    // (replicated metadata) -> aligned send/recv lists.
    for (int gd = 0; gd < dba.size(); ++gd) {
      const int od = ddm[gd];
      const Box2D vd = dba[gd];
      for (int gs : shash.query(vd)) {
        const int os = sdm[gs];
        if (od != me && os != me)
          continue;  // does not concern us
        if (od == me && os == me)
          continue;  // local, already done
        const Box2D region = vd.intersect(sba[gs]);
        if (region.empty())
          continue;
        if (os == me)
          sched.send[od].push_back({gs, gd, region});  // I own the src
        else
          sched.recv[os].push_back({gs, gd, region});  // I own the dst
      }
    }
  }
#endif
}

// Returns the cached schedule for (dst layout, src layout), building and memoizing it on first use.
// The cache lives on the DST MultiFab; each entry is keyed on the SRC-layout fingerprint (src
// BoxArray + DistributionMapping), so a different src layout to the same dst builds a distinct entry.
inline std::shared_ptr<const CopySchedule> get_copy_schedule(const MultiFab& dst,
                                                            const MultiFab& src) {
  CopyScheduleCache& cache = dst.copy_cache();
  if (std::shared_ptr<const CopySchedule> hit = cache.find(src.box_array(), src.dmap())) {
    ++copy_schedule_hit_counter();
    return hit;
  }
  ++copy_schedule_miss_counter();
  std::shared_ptr<CopySchedule> s = cache.add();
  s->key.boxes = src.box_array().boxes();
  s->key.ranks = src.dmap().ranks();
  build_copy_schedule(dst, src, *s);
  return s;
}

}  // namespace detail

/// Index of the coarse cell containing the fine cell a (FLOOR division by r, handles a < 0).
/// POPS_HD: called inside the interpolation / coarse->fine injection kernels. Thin adapter over
/// floor_div (box2d.hpp): same floor division, bit-identical result.
POPS_HD inline int coarsen_index(int a, int r) {
  return floor_div(a, r);
}

/// Coarsens each box of the BoxArray by a ratio r (coarsen box by box, order preserved).
inline BoxArray coarsen(const BoxArray& ba, int r) {
  if (r < 1)
    throw_validation_error("pops/mesh/layout/refinement.hpp: coarsen",
                           "refinement/coarsening ratio r >= 1",
                           "r=" + std::to_string(r));
  std::vector<Box2D> b;
  b.reserve(ba.size());
  for (int i = 0; i < ba.size(); ++i)
    b.push_back(ba[i].coarsen(r));
  return BoxArray{std::move(b)};
}

/// Copies the valid regions that OVERLAP from src to dst (same indices, no shift).
/// General redistribution between two MultiFab over the same domain with different decompositions.
/// Copies min(ncomp) components. A dst cell not covered by src is left intact.
inline void parallel_copy(MultiFab& dst, const MultiFab& src) {
  const int nc = std::min(dst.ncomp(), src.ncomp());
  // memoized schedule (BoxHash + enumeration) for this (dst layout, src layout) pair. Replayed in the
  // SAME order as the legacy inline loops -> bit-identical to the per-call rebuild (ADC-607).
  const std::shared_ptr<const CopySchedule> sched = detail::get_copy_schedule(dst, src);

  // --- local copies (dst local AND src local), replayed from the cached plan ---
  for (const CopyJob& j : sched->local) {
    Fab2D& D = dst.fab(dst.local_index_of(j.gd));
    const Fab2D& S = src.fab(src.local_index_of(j.gs));
    detail::copy_shifted(D, S, j.region, 0, 0, nc);
  }

#ifdef POPS_HAS_MPI
  if (n_ranks() <= 1)
    return;
  const int np = n_ranks();

  auto bufsz = [&](const std::vector<CopyJob>& js) {
    std::int64_t n = 0;
    for (const auto& j : js)
      n += j.region.num_cells() * nc;
    return n;
  };
  device_fence();  // GPU: the local (device) copies precede the HOST pack
  std::vector<std::vector<Real>> sbuf(np), rbuf(np);
  std::vector<MPI_Request> reqs;
  for (int r = 0; r < np; ++r) {
    if (!sched->send[r].empty()) {
      sbuf[r].resize(bufsz(sched->send[r]));
      std::int64_t k = 0;
      for (const auto& j : sched->send[r]) {
        const ConstArray4 s = src.fab(src.local_index_of(j.gs)).const_array();
        for (int c = 0; c < nc; ++c)
          for (int jj = j.region.lo[1]; jj <= j.region.hi[1]; ++jj)
            for (int ii = j.region.lo[0]; ii <= j.region.hi[0]; ++ii)
              sbuf[r][k++] = s(ii, jj, c);
      }
      reqs.emplace_back();
      MPI_Isend(sbuf[r].data(), static_cast<int>(sbuf[r].size()), MPI_DOUBLE, r, 1, MPI_COMM_WORLD,
                &reqs.back());
    }
    if (!sched->recv[r].empty()) {
      rbuf[r].resize(bufsz(sched->recv[r]));
      reqs.emplace_back();
      MPI_Irecv(rbuf[r].data(), static_cast<int>(rbuf[r].size()), MPI_DOUBLE, r, 1, MPI_COMM_WORLD,
                &reqs.back());
    }
  }
  if (!reqs.empty())
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

  device_fence();  // GPU: barrier before the HOST write of the received cells
  for (int r = 0; r < np; ++r) {
    std::int64_t k = 0;
    for (const auto& j : sched->recv[r]) {
      Array4 d = dst.fab(dst.local_index_of(j.gd)).array();
      for (int c = 0; c < nc; ++c)
        for (int jj = j.region.lo[1]; jj <= j.region.hi[1]; ++jj)
          for (int ii = j.region.lo[0]; ii <= j.region.hi[0]; ++ii)
            d(ii, jj, c) = rbuf[r][k++];
    }
  }
#endif
}

namespace detail {

inline std::string transfer_scratch_summary(const MultiFab& fine, const MultiFab& scratch,
                                            int nc, int r) {
  return "r=" + std::to_string(r) + ", fine.boxes=" +
         std::to_string(fine.box_array().size()) + ", scratch.boxes=" +
         std::to_string(scratch.box_array().size()) + ", scratch.ncomp=" +
         std::to_string(scratch.ncomp()) + ", required.ncomp>=" + std::to_string(nc) +
         ", scratch.ngrow=" + std::to_string(scratch.n_grow());
}

inline void validate_transfer_scratch(const char* where, const MultiFab& fine,
                                      const MultiFab& scratch, int nc, int r) {
  if (r < 1)
    throw_validation_error(where, "refinement/coarsening ratio r >= 1",
                           "r=" + std::to_string(r));
  const BoxArray expected = coarsen(fine.box_array(), r);
  if (scratch.box_array().boxes() != expected.boxes() ||
      scratch.dmap().ranks() != fine.dmap().ranks() || scratch.ncomp() < nc ||
      scratch.n_grow() != 0) {
    throw_validation_error(
        where,
        "scratch MultiFab layout == coarsen(fine.box_array(), r), scratch.dmap == fine.dmap, "
        "scratch.ncomp >= min(fine.ncomp, coarse.ncomp), scratch.ngrow == 0",
        transfer_scratch_summary(fine, scratch, nc, r));
  }
}

// NAMED FUNCTORS (rather than POPS_HD lambdas) for the AMR transfer kernels. Same reasons as the
// rest of the mesh/elliptic path (cf. fill_boundary.hpp): refinement is first instantiated from
// the MG V-cycle pulled in by an external TU; an extended lambda there trips up device kernel
// emission under nvcc. Body strictly identical to the old lambdas -> bit-identical CPU and device.

/// CONSERVATIVE average of an r x r block: C(I, J, c) = (sum of the r^2 fine cells) * inv.
struct AverageDownKernel {
  ConstArray4 F;
  Array4 C;
  Real inv;
  int r, c;
  POPS_HD void operator()(int I, int J) const {
    Real s = 0;
    for (int b = 0; b < r; ++b)
      for (int a = 0; a < r; ++a)
        s += F(r * I + a, r * J + b, c);
    C(I, J, c) = s * inv;
  }
};

}  // namespace detail

/// CONSERVATIVE average fine -> coarse (ratio r): coarse(I, J) = average of the r^2 fine cells of
/// the block. Writes the coarse cells covered by fine (via parallel_copy from a local fine-coarsen
/// grid); copies min(ncomp). Preserves the integral (sum * dV) of the fine over the covered area.
// PROVIDED-BUFFER variant: @p cfine is the "fine coarsen" grid (layout coarsen(fine.box_array(), r),
// dmap = fine.dmap(), >= min(ncomp) components, 0 ghost) ALLOCATED by the caller and reused on each
// call (hot path of the MG V-cycle: avoids one MultiFab allocation per restriction). Computation
// STRICTLY identical to the allocating variant below.
inline void average_down(const MultiFab& fine, MultiFab& coarse, int r, MultiFab& cfine) {
  const int nc = std::min(fine.ncomp(), coarse.ncomp());
  detail::validate_transfer_scratch("pops/mesh/layout/refinement.hpp: average_down(scratch)",
                                    fine, cfine, nc, r);
  const Real inv = Real(1) / (r * r);
  for (int li = 0; li < fine.local_size(); ++li) {
    const ConstArray4 F = fine.fab(li).const_array();
    Array4 C = cfine.fab(li).array();
    const Box2D cb = cfine.fab(li).box();
    for (int c = 0; c < nc; ++c)
      for_each_cell(cb, detail::AverageDownKernel{F, C, inv, r, c});
  }
  parallel_copy(coarse, cfine);
}
inline void average_down(const MultiFab& fine, MultiFab& coarse, int r) {
  MultiFab cfine(coarsen(fine.box_array(), r), fine.dmap(), fine.ncomp(), 0);
  average_down(fine, coarse, r, cfine);
}

namespace detail {

/// Piecewise-CONSTANT injection: F(i, j, c) receives the value of its covering coarse cell.
struct InterpolateKernel {
  Array4 F;
  ConstArray4 C;
  int r, c;
  POPS_HD void operator()(int i, int j) const {
    F(i, j, c) = C(coarsen_index(i, r), coarsen_index(j, r), c);
  }
};

}  // namespace detail

/// Interpolation coarse -> fine (ratio r) by piecewise-CONSTANT injection: each fine cell
/// (including the box ghosts) receives the value of its coarse cell (coarsen_index). Copies
/// min(ncomp). First brings the coarse values onto a local fine-coarsen grid (parallel_copy).
// PROVIDED-BUFFER variant: @p cfine is the "fine coarsen" grid (same layout contract as
// average_down above) allocated by the caller and reused (hot path of the MG V-cycle: avoids one
// allocation per prolongation). Computation STRICTLY identical to the allocating variant.
inline void interpolate(const MultiFab& coarse, MultiFab& fine, int r, MultiFab& cfine) {
  const int nc = std::min(fine.ncomp(), coarse.ncomp());
  detail::validate_transfer_scratch("pops/mesh/layout/refinement.hpp: interpolate(scratch)",
                                    fine, cfine, nc, r);
  parallel_copy(cfine, coarse);  // bring the coarse values onto the fine-coarsen grid
  for (int li = 0; li < fine.local_size(); ++li) {
    Array4 F = fine.fab(li).array();
    const ConstArray4 C = cfine.fab(li).const_array();
    const Box2D fb = fine.fab(li).box();
    for (int c = 0; c < nc; ++c)
      for_each_cell(fb, detail::InterpolateKernel{F, C, r, c});
  }
}
inline void interpolate(const MultiFab& coarse, MultiFab& fine, int r) {
  MultiFab cfine(coarsen(fine.box_array(), r), fine.dmap(), fine.ncomp(), 0);
  interpolate(coarse, fine, r, cfine);
}

}  // namespace pops
