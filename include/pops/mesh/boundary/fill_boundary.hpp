/// @file
/// @brief fill_boundary: INTRA-level halo exchange (fills ghosts from neighbors).
///
/// Fills the ghosts of each Fab from the VALID regions of neighboring boxes in the same
/// MultiFab, with optional periodic wrapping. Single-rank: direct memory copies. Multi-rank
/// (POPS_HAS_MPI + communicator size > 1): metadata is REPLICATED -> each rank enumerates DETERMINISTICALLY
/// the same job list, so buffers line up without negotiating sizes (MPI_Isend/Irecv,
/// halo tag). Two-phase API (classic compute/comm overlap): fill_boundary_begin posts the
/// exchanges, fill_boundary_end waits and unpacks; fill_boundary chains both (blocking). Ghosts
/// OUTSIDE the domain without periodicity are NOT touched here (those are the physical BCs,
/// physical_bc.hpp). The pack/unpack kernels are device-clean NAMED FUNCTORS (nvcc limitation).
///
/// The job schedule (BoxHash + local/global enumeration) is MEMOIZED per (layout, Periodicity,
/// domain) on the MultiFab (ADC-260, halo_schedule.hpp): it is a pure function of the invariant
/// layout, so only the copy/pack/MPI/unpack of the live data reruns. The plan is replayed in the
/// SAME deterministic order as the original inline enumeration -> bit-identical buffers.

#pragma once

#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/index/box_hash.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/boundary/halo_schedule.hpp>
#include <pops/mesh/boundary/periodicity.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {

namespace detail {

// Execute one rank-local preparation phase, then establish collective success before any peer may
// post a point-to-point request.  Allocation and other preparation failures are deliberately
// converted to the same exception class on every rank: otherwise one rank could unwind while its
// peers continue into MPI_Isend/MPI_Irecv and wait forever.  The callable must not post MPI work.
template <class Prepare>
inline void collectively_prepare_before_halo_post(const CommunicatorView& communicator,
                                                  Prepare&& prepare) {
  long local_failure = 0;
  try {
    std::forward<Prepare>(prepare)();
  } catch (const std::bad_alloc&) {
    local_failure = 1;
  } catch (...) {
    local_failure = 2;
  }
  const long global_failure = all_reduce_max(local_failure, communicator);
  if (global_failure == 1)
    throw std::bad_alloc();
  if (global_failure != 0)
    throw std::runtime_error(
        "fill_boundary: one peer failed while preparing halo communication buffers");
}

// NAMED FUNCTORS (not POPS_HD lambdas) for the halo-exchange kernels. Same reasons as the rest of the
// elliptic/mesh path (#93, recipe #64): fill_boundary is first instantiated from the MG V-cycle pulled
// from an external TU; an extended lambda there stalls device kernel emission under nvcc (-O Release
// without -g). Strictly identical body -> bit-identical CPU and device.
struct CopyShiftedKernel {
  Array4 d;
  ConstArray4 s;
  int sx, sy, c;
  POPS_HD void operator()(int i, int j) const { d(i, j, c) = s(i - sx, j - sy, c); }
};

// dst(i, j, c) = src(i - sx, j - sy, c) for (i, j) in region.
inline void copy_shifted(Fab2D& dst, const Fab2D& src, const Box2D& region, int sx, int sy,
                         int ncomp) {
  Array4 d = dst.array();
  ConstArray4 s = src.const_array();
  for (int c = 0; c < ncomp; ++c)
    for_each_cell(region, CopyShiftedKernel{d, s, sx, sy, c});
}

struct CopyMappedKernel {
  Array4 d;
  ConstArray4 s;
  int source_axis_0, source_axis_1;
  int source_sign_0, source_sign_1;
  int source_offset_0, source_offset_1;
  int c;
  POPS_HD void operator()(int i, int j) const {
    const int destination[2] = {i, j};
    const int source_i = static_cast<int>(
        static_cast<std::int64_t>(source_sign_0) * destination[source_axis_0] + source_offset_0);
    const int source_j = static_cast<int>(
        static_cast<std::int64_t>(source_sign_1) * destination[source_axis_1] + source_offset_1);
    d(i, j, c) = s(source_i, source_j, c);
  }
};

inline void copy_mapped(Fab2D& dst, const Fab2D& src, const HaloJob& job, int ncomp) {
  Array4 d = dst.array();
  ConstArray4 s = src.const_array();
  for (int c = 0; c < ncomp; ++c)
    for_each_cell(job.region, CopyMappedKernel{d, s, job.source_from_destination_axis[0],
                                               job.source_from_destination_axis[1],
                                               job.source_sign[0], job.source_sign[1],
                                               job.source_offset[0], job.source_offset[1], c});
}

// Pack of a send job: sb[b0 + c*rsz + off] = s(i - sx, jc - sy, c), off = (jc-lo1)*rnx + (i-lo0).
struct PackKernel {
  Real* sb;
  ConstArray4 s;
  std::int64_t b0, rsz;
  int lo0, lo1, rnx, sx, sy, ncl;
  POPS_HD void operator()(int i, int jc) const {
    const std::int64_t off = static_cast<std::int64_t>(jc - lo1) * rnx + (i - lo0);
    for (int c = 0; c < ncl; ++c)
      sb[b0 + static_cast<std::int64_t>(c) * rsz + off] = s(i - sx, jc - sy, c);
  }
};

struct PackMappedKernel {
  Real* sb;
  ConstArray4 s;
  std::int64_t b0, rsz;
  int lo0, lo1, rnx, ncl;
  int source_axis_0, source_axis_1;
  int source_sign_0, source_sign_1;
  int source_offset_0, source_offset_1;
  POPS_HD void operator()(int i, int j) const {
    const std::int64_t off = static_cast<std::int64_t>(j - lo1) * rnx + (i - lo0);
    const int destination[2] = {i, j};
    const int source_i = static_cast<int>(
        static_cast<std::int64_t>(source_sign_0) * destination[source_axis_0] + source_offset_0);
    const int source_j = static_cast<int>(
        static_cast<std::int64_t>(source_sign_1) * destination[source_axis_1] + source_offset_1);
    for (int c = 0; c < ncl; ++c)
      sb[b0 + static_cast<std::int64_t>(c) * rsz + off] = s(source_i, source_j, c);
  }
};

// Unpack of a receive job: d(i, jc, c) = rb[b0 + c*rsz + off], off = (jc-lo1)*rnx + (i-lo0).
struct UnpackKernel {
  const Real* rb;
  Array4 d;
  std::int64_t b0, rsz;
  int lo0, lo1, rnx, ncl;
  POPS_HD void operator()(int i, int jc) const {
    const std::int64_t off = static_cast<std::int64_t>(jc - lo1) * rnx + (i - lo0);
    for (int c = 0; c < ncl; ++c)
      d(i, jc, c) = rb[b0 + static_cast<std::int64_t>(c) * rsz + off];
  }
};

// Enumerates the halo schedule for (mf layout, per, domain): the BoxHash build + the local (dst AND
// src local) and, under MPI with n_ranks()>1, the global (cross-rank send/recv) job lists. This is
// the per-call work that ADC-260 hoists out of fill_boundary_begin; it runs ONCE per distinct
// (layout, Periodicity, domain) and is then replayed from the cache. Jobs are produced in the SAME
// deterministic order as the legacy inline loops (local: li x shifts x sorted gB; global: gF x
// shifts x sorted gB), so the packed buffers stay bit-identical and the per-rank send/recv lists
// stay aligned. Bumps the build counter (cache-engagement test hook).
inline void build_halo_schedule(const MultiFab& mf, const Box2D& domain, Periodicity per,
                                const CommunicatorView& communicator, HaloSchedule& sched) {
  halo_schedule_build_counter().fetch_add(1, std::memory_order_relaxed);
  sched.communicator_size = communicator.size();
  sched.communicator_rank = communicator.rank();
  const int ng = mf.n_grow();
  const int Lx = domain.nx();
  const int Ly = domain.ny();
  const BoxArray& ba = mf.box_array();

  // A halo may be deeper than the complete periodic extent (a one-cell axis with a WENO halo is
  // the smallest useful example).  One +/-L image only fills the first period and leaves the outer
  // layers untouched.  Enumerate every image that can intersect grow(valid, ng), retaining the old
  // 0,+L,-L order as the prefix so ordinary ng<=L schedules stay byte-for-byte unchanged.
  auto periodic_shifts = [ng](bool periodic, int extent, const char* axis) {
    std::vector<int> values{0};
    if (!periodic)
      return values;
    if (extent <= 0)
      throw std::runtime_error(std::string("fill_boundary: periodic ") + axis +
                               " extent must be positive");
    const std::int64_t images64 =
        (static_cast<std::int64_t>(ng) + extent - 1) / static_cast<std::int64_t>(extent);
    if (images64 > (std::numeric_limits<int>::max() - 1) / 2)
      throw std::overflow_error(std::string("fill_boundary: periodic ") + axis +
                                " image count exceeds the native schedule range");
    const int images = static_cast<int>(images64);
    values.reserve(static_cast<std::size_t>(1 + 2 * images));
    for (int image = 1; image <= images; ++image) {
      const std::int64_t shift = static_cast<std::int64_t>(image) * extent;
      if (shift > std::numeric_limits<int>::max())
        throw std::overflow_error(std::string("fill_boundary: periodic ") + axis +
                                  " image shift exceeds the native index range");
      values.push_back(static_cast<int>(shift));
      values.push_back(-static_cast<int>(shift));
    }
    return values;
  };
  const std::vector<int> sxv = periodic_shifts(per.x, Lx, "x");
  const std::vector<int> syv = periodic_shifts(per.y, Ly, "y");
  std::vector<std::pair<int, int>> shifts;
  for (int sx : sxv)
    for (int sy : syv)
      shifts.push_back({sx, sy});

  // spatial hash: restricts the neighbor-box search (see box_hash.hpp).
  const BoxHash hash(ba, suggest_bin(ba));

  // --- local jobs (local dst AND local src) ---
  for (int li = 0; li < mf.local_size(); ++li) {
    const int gF = mf.global_index(li);
    const Box2D gbox = mf.fab(li).box().grow(ng);
    for (auto [sx, sy] : shifts) {
      const Box2D Q = gbox.shift(0, -sx).shift(1, -sy);
      for (int gB : hash.query(Q)) {
        if (gB == gF && sx == 0 && sy == 0)
          continue;  // self, without shift
        const int srcLocal = mf.local_index_of(gB);
        if (srcLocal < 0)
          continue;  // non-local src -> MPI below
        const Box2D region = gbox.intersect(ba[gB].shift(0, sx).shift(1, sy));
        if (region.empty())
          continue;
        sched.local.push_back({gB, gF, sx, sy, region});
      }
    }
  }

#ifdef POPS_HAS_MPI
  const int np = communicator.size();
  if (np > 1) {
    const int me = communicator.rank();
    const DistributionMapping& dm = mf.dmap();
    sched.send.assign(np, {});
    sched.recv.assign(np, {});
    // deterministic global enumeration: (dst gF) x shifts x hash candidates (gB sorted). Identical
    // on all ranks -> aligned send/recv lists.
    for (int gF = 0; gF < ba.size(); ++gF) {
      const int od = dm[gF];
      const Box2D gbox = ba[gF].grow(ng);
      for (auto [sx, sy] : shifts) {
        const Box2D Q = gbox.shift(0, -sx).shift(1, -sy);
        for (int gB : hash.query(Q)) {
          if (gB == gF && sx == 0 && sy == 0)
            continue;
          const int os = dm[gB];
          if (od != me && os != me)
            continue;
          if (od == me && os == me)
            continue;
          const Box2D region = gbox.intersect(ba[gB].shift(0, sx).shift(1, sy));
          if (region.empty())
            continue;
          if (os == me)
            sched.send[od].push_back({gB, gF, sx, sy, region});
          else
            sched.recv[os].push_back({gB, gF, sx, sy, region});
        }
      }
    }
  }
#endif
}

struct MappedPeriodicDirection {
  int destination_face = -1;
  std::array<int, 2> source_from_destination_axis{{0, 1}};
  std::array<int, 2> source_sign{{1, 1}};
  std::array<int, 2> source_offset{{0, 0}};
};

inline int checked_periodic_mapping_index(std::int64_t value, const char* operation) {
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    throw std::overflow_error(operation);
  return static_cast<int>(value);
}

inline std::array<MappedPeriodicDirection, 2> mapped_periodic_directions(
    const PeriodicIdentification2D& identification, const Box2D& domain, int ngrow) {
  identification.validate();
  if (domain.empty())
    throw std::invalid_argument("mapped periodic halo requires a non-empty execution domain");
  const int source_axis = identification.source_face / 2;
  const int target_axis = identification.target_face / 2;
  if (ngrow > domain.length(source_axis) || ngrow > domain.length(target_axis))
    throw std::invalid_argument("mapped periodic halo depth exceeds one endpoint normal extent");

  std::array<int, 2> forward_offset{{0, 0}};
  for (int source_coordinate = 0; source_coordinate < 2; ++source_coordinate) {
    const int target_coordinate =
        identification.permutation[static_cast<std::size_t>(source_coordinate)];
    const int sign = identification.signs[static_cast<std::size_t>(source_coordinate)];
    std::int64_t offset = 0;
    if (source_coordinate == source_axis) {
      const int source_adjacent = identification.source_face % 2 == 0
                                      ? domain.lo[source_coordinate]
                                      : domain.hi[source_coordinate];
      const std::int64_t target_first_ghost =
          identification.target_face % 2 == 0
              ? static_cast<std::int64_t>(domain.lo[target_coordinate]) - 1
              : static_cast<std::int64_t>(domain.hi[target_coordinate]) + 1;
      offset = target_first_ghost - static_cast<std::int64_t>(sign) * source_adjacent;
    } else {
      if (domain.length(source_coordinate) != domain.length(target_coordinate))
        throw std::invalid_argument(
            "axis-permuted periodic identification requires equal mapped tangential extents");
      offset = sign == 1 ? static_cast<std::int64_t>(domain.lo[target_coordinate]) -
                               domain.lo[source_coordinate]
                         : static_cast<std::int64_t>(domain.hi[target_coordinate]) +
                               domain.lo[source_coordinate];
    }
    forward_offset[static_cast<std::size_t>(target_coordinate)] = checked_periodic_mapping_index(
        offset, "mapped periodic coordinate offset exceeds the native index range");
  }

  MappedPeriodicDirection source_side;
  source_side.destination_face = identification.source_face;
  for (int source_coordinate = 0; source_coordinate < 2; ++source_coordinate) {
    const int target_coordinate =
        identification.permutation[static_cast<std::size_t>(source_coordinate)];
    source_side.source_from_destination_axis[static_cast<std::size_t>(target_coordinate)] =
        source_coordinate;
    source_side.source_sign[static_cast<std::size_t>(target_coordinate)] =
        identification.signs[static_cast<std::size_t>(source_coordinate)];
    source_side.source_offset[static_cast<std::size_t>(target_coordinate)] =
        forward_offset[static_cast<std::size_t>(target_coordinate)];
  }

  MappedPeriodicDirection target_side;
  target_side.destination_face = identification.target_face;
  for (int source_coordinate = 0; source_coordinate < 2; ++source_coordinate) {
    const int target_coordinate =
        identification.permutation[static_cast<std::size_t>(source_coordinate)];
    const int sign = identification.signs[static_cast<std::size_t>(source_coordinate)];
    target_side.source_from_destination_axis[static_cast<std::size_t>(source_coordinate)] =
        target_coordinate;
    target_side.source_sign[static_cast<std::size_t>(source_coordinate)] = sign;
    target_side.source_offset[static_cast<std::size_t>(source_coordinate)] =
        checked_periodic_mapping_index(
            -static_cast<std::int64_t>(sign) *
                forward_offset[static_cast<std::size_t>(target_coordinate)],
            "mapped periodic inverse offset exceeds the native index range");
  }
  return {source_side, target_side};
}

inline Box2D mapped_destination_region(const Box2D& grown, const Box2D& domain, int face,
                                       int ngrow) {
  const int axis = face / 2;
  Box2D strip = domain;
  if (face % 2 == 0) {
    strip.lo[axis] = checked_periodic_mapping_index(
        static_cast<std::int64_t>(domain.lo[axis]) - ngrow,
        "mapped periodic lower ghost bound exceeds the native index range");
    strip.hi[axis] = checked_periodic_mapping_index(
        static_cast<std::int64_t>(domain.lo[axis]) - 1,
        "mapped periodic lower ghost bound exceeds the native index range");
  } else {
    strip.lo[axis] = checked_periodic_mapping_index(
        static_cast<std::int64_t>(domain.hi[axis]) + 1,
        "mapped periodic upper ghost bound exceeds the native index range");
    strip.hi[axis] = checked_periodic_mapping_index(
        static_cast<std::int64_t>(domain.hi[axis]) + ngrow,
        "mapped periodic upper ghost bound exceeds the native index range");
  }
  return grown.intersect(strip);
}

inline Box2D map_destination_box_to_source(const Box2D& destination,
                                           const MappedPeriodicDirection& mapping) {
  if (destination.empty())
    return destination;
  Box2D source;
  for (int source_axis = 0; source_axis < 2; ++source_axis) {
    const int destination_axis =
        mapping.source_from_destination_axis[static_cast<std::size_t>(source_axis)];
    const int sign = mapping.source_sign[static_cast<std::size_t>(source_axis)];
    const int offset = mapping.source_offset[static_cast<std::size_t>(source_axis)];
    const std::int64_t first =
        static_cast<std::int64_t>(sign) * destination.lo[destination_axis] + offset;
    const std::int64_t second =
        static_cast<std::int64_t>(sign) * destination.hi[destination_axis] + offset;
    source.lo[source_axis] = checked_periodic_mapping_index(
        std::min(first, second), "mapped periodic source box exceeds native indices");
    source.hi[source_axis] = checked_periodic_mapping_index(
        std::max(first, second), "mapped periodic source box exceeds native indices");
  }
  return source;
}

inline Box2D map_source_box_to_destination(const Box2D& source,
                                           const MappedPeriodicDirection& mapping) {
  if (source.empty())
    return source;
  Box2D destination;
  for (int source_axis = 0; source_axis < 2; ++source_axis) {
    const int destination_axis =
        mapping.source_from_destination_axis[static_cast<std::size_t>(source_axis)];
    const int sign = mapping.source_sign[static_cast<std::size_t>(source_axis)];
    const int offset = mapping.source_offset[static_cast<std::size_t>(source_axis)];
    const std::int64_t first = static_cast<std::int64_t>(sign) *
                               (static_cast<std::int64_t>(source.lo[source_axis]) - offset);
    const std::int64_t second = static_cast<std::int64_t>(sign) *
                                (static_cast<std::int64_t>(source.hi[source_axis]) - offset);
    destination.lo[destination_axis] = checked_periodic_mapping_index(
        std::min(first, second), "mapped periodic destination box exceeds native indices");
    destination.hi[destination_axis] = checked_periodic_mapping_index(
        std::max(first, second), "mapped periodic destination box exceeds native indices");
  }
  return destination;
}

inline HaloJob mapped_halo_job(int source, int destination, const Box2D& region,
                               const MappedPeriodicDirection& mapping) {
  HaloJob result;
  result.src = source;
  result.dst = destination;
  result.region = region;
  result.mapped = true;
  result.source_from_destination_axis = mapping.source_from_destination_axis;
  result.source_sign = mapping.source_sign;
  result.source_offset = mapping.source_offset;
  return result;
}

inline void append_mapped_periodic_jobs(
    const MultiFab& mf, const Box2D& domain,
    const std::vector<PeriodicIdentification2D>& identifications,
    const CommunicatorView& communicator, HaloSchedule& schedule) {
  if (identifications.size() != 1)
    throw std::invalid_argument(
        "mapped periodic halo currently requires exactly one identification; mixed periodic "
        "corner topology needs a composed scheduler");
  const int ngrow = mf.n_grow();
  const BoxArray& boxes = mf.box_array();
  const BoxHash hash(boxes, suggest_bin(boxes));
  const auto directions = mapped_periodic_directions(identifications.front(), domain, ngrow);

  for (int local_destination = 0; local_destination < mf.local_size(); ++local_destination) {
    const int global_destination = mf.global_index(local_destination);
    const Box2D grown = mf.fab(local_destination).box().grow(ngrow);
    for (const auto& mapping : directions) {
      const Box2D candidate =
          mapped_destination_region(grown, domain, mapping.destination_face, ngrow);
      if (candidate.empty())
        continue;
      const Box2D query = map_destination_box_to_source(candidate, mapping);
      for (const int global_source : hash.query(query)) {
        if (mf.local_index_of(global_source) < 0)
          continue;
        const Box2D image = map_source_box_to_destination(boxes[global_source], mapping);
        const Box2D region = candidate.intersect(image);
        if (!region.empty())
          schedule.local.push_back(
              mapped_halo_job(global_source, global_destination, region, mapping));
      }
    }
  }

#ifdef POPS_HAS_MPI
  if (communicator.size() > 1) {
    const int me = communicator.rank();
    const DistributionMapping& distribution = mf.dmap();
    for (int global_destination = 0; global_destination < boxes.size(); ++global_destination) {
      const int destination_owner = distribution[global_destination];
      const Box2D grown = boxes[global_destination].grow(ngrow);
      for (const auto& mapping : directions) {
        const Box2D candidate =
            mapped_destination_region(grown, domain, mapping.destination_face, ngrow);
        if (candidate.empty())
          continue;
        const Box2D query = map_destination_box_to_source(candidate, mapping);
        for (const int global_source : hash.query(query)) {
          const int source_owner = distribution[global_source];
          if ((destination_owner != me && source_owner != me) ||
              (destination_owner == me && source_owner == me))
            continue;
          const Box2D image = map_source_box_to_destination(boxes[global_source], mapping);
          const Box2D region = candidate.intersect(image);
          if (region.empty())
            continue;
          HaloJob job = mapped_halo_job(global_source, global_destination, region, mapping);
          if (source_owner == me)
            schedule.send[destination_owner].push_back(std::move(job));
          else
            schedule.recv[source_owner].push_back(std::move(job));
        }
      }
    }
  }
#else
  (void)communicator;
#endif
}

// Returns the cached schedule for (mf layout, per, domain), building and memoizing it on first use.
inline std::shared_ptr<const HaloSchedule> get_halo_schedule(const MultiFab& mf,
                                                             const Box2D& domain, Periodicity per,
                                                             const CommunicatorView& communicator) {
  HaloScheduleCache& cache = mf.halo_cache();
  if (std::shared_ptr<const HaloSchedule> hit =
          cache.find(per.x, per.y, domain, communicator.size(), communicator.rank())) {
    if (hit->boxes != mf.box_array().boxes() || hit->ranks != mf.dmap().ranks() ||
        hit->ngrow != mf.n_grow())
      throw std::logic_error("fill_boundary: cached schedule crossed an exact layout");
    return hit;
  }
  std::shared_ptr<HaloSchedule> s = std::make_shared<HaloSchedule>();
  s->per_x = per.x;
  s->per_y = per.y;
  s->domain = domain;
  s->boxes = mf.box_array().boxes();
  s->ranks = mf.dmap().ranks();
  s->ngrow = mf.n_grow();
  build_halo_schedule(mf, domain, per, communicator, *s);
  cache.reserve_for_append();
  cache.publish_prepared(s);
  return s;
}

inline std::shared_ptr<const HaloSchedule> get_mapped_halo_schedule(
    const MultiFab& mf, const Box2D& domain,
    const std::vector<PeriodicIdentification2D>& identifications,
    const CommunicatorView& communicator) {
  HaloScheduleCache& cache = mf.halo_cache();
  if (std::shared_ptr<const HaloSchedule> hit = cache.find(
          false, false, domain, communicator.size(), communicator.rank(), identifications)) {
    if (hit->boxes != mf.box_array().boxes() || hit->ranks != mf.dmap().ranks() ||
        hit->ngrow != mf.n_grow())
      throw std::logic_error("fill_boundary: cached mapped schedule crossed an exact layout");
    return hit;
  }
  auto schedule = std::make_shared<HaloSchedule>();
  schedule->per_x = false;
  schedule->per_y = false;
  schedule->mapped_periodic = identifications;
  schedule->domain = domain;
  schedule->boxes = mf.box_array().boxes();
  schedule->ranks = mf.dmap().ranks();
  schedule->ngrow = mf.n_grow();
  build_halo_schedule(mf, domain, Periodicity{}, communicator, *schedule);
  append_mapped_periodic_jobs(mf, domain, identifications, communicator, *schedule);
  cache.reserve_for_append();
  cache.publish_prepared(schedule);
  return schedule;
}

}  // namespace detail

struct HaloExchange;
namespace detail {
HaloExchange fill_boundary_begin_on(MultiFab&, const Box2D&, Periodicity, const CommunicatorView&,
                                    int, const std::vector<PeriodicIdentification2D>*);
}

/// Opaque state of an in-flight halo exchange, returned by fill_boundary_begin and consumed by
/// fill_boundary_end. OWNS the send/receive buffers and the MPI_Request: they stay alive
/// (and at a stable address after move) until fill_boundary_end is called. Empty in single-rank.
/// Holds a shared handle to the cached schedule (ADC-260) so fill_boundary_end unpacks from the SAME
/// recv job list begin posted; the handle keeps the plan alive even if a later fill_boundary on the
/// same MultiFab appends another schedule.
struct HaloExchangeStorage {
  std::shared_ptr<const HaloSchedule> sched;  // replayed plan (null if mf has no ghost)
  int ncomp = 0;
  bool in_use = false;
#ifdef POPS_HAS_MPI
  // Buffers in PINNED HOST memory (comm_allocator = Kokkos::SharedHostPinnedSpace under Kokkos,
  // std::allocator otherwise), NOT managed. The pack/unpack in for_each (device under Kokkos)
  // writes/reads directly into them since pinned host is device-accessible; BUT the pointer passed to
  // MPI is seen as HOST (cuPointerGetAttribute = HOST), so a CUDA-aware MPI (BTL smcuda) does NOT
  // attempt CUDA IPC on it. A managed/UVM pointer, on the other hand, triggered IPC, which DEADLOCKS
  // between two GPUs isolated by cgroup (srun --gpus-per-task=1: each rank sees only its GPU as device
  // 0, cuIpcOpenMemHandle of the peer's buffer impossible). See core/allocator.hpp (comm_allocator).
  std::vector<std::vector<Real, comm_allocator<Real>>> sbuf, rbuf;  // alive until end
  std::vector<MPI_Request> reqs;
  // Persistent count scratch: the schedule and component width are part of this lease identity, so
  // repeated fills recompute exact counts without allocating two O(nranks) vectors on the hot path.
  std::vector<std::int64_t> send_sizes, recv_sizes;
#endif
};

inline std::shared_ptr<HaloExchangeStorage> HaloScheduleCache::acquire_exchange(
    const std::shared_ptr<const HaloSchedule>& schedule, int ncomp) {
  for (const auto& candidate : exchange_pool_) {
    if (!candidate->in_use && candidate->sched.get() == schedule.get() &&
        candidate->ncomp == ncomp) {
      candidate->in_use = true;
      return candidate;
    }
  }
  auto storage = std::make_shared<HaloExchangeStorage>();
  storage->sched = schedule;
  storage->ncomp = ncomp;
  storage->in_use = true;
  exchange_pool_.push_back(storage);
  return storage;
}

struct HaloExchange {
  HaloExchange() = default;
  HaloExchange(const HaloExchange&) = delete;
  HaloExchange& operator=(const HaloExchange&) = delete;
  HaloExchange(HaloExchange&& other) noexcept
      : storage_(std::move(other.storage_)),
        destination_(other.destination_),
        destination_ngrow_(other.destination_ngrow_),
        destination_ncomp_(other.destination_ncomp_),
        device_work_pending_(other.device_work_pending_),
        state_(other.state_) {
    other.destination_ = nullptr;
    other.device_work_pending_ = false;
    other.state_ = State::kInvalid;
  }
  HaloExchange& operator=(HaloExchange&& other) noexcept {
    if (this != &other) {
      abandon();
      storage_ = std::move(other.storage_);
      destination_ = other.destination_;
      destination_ngrow_ = other.destination_ngrow_;
      destination_ncomp_ = other.destination_ncomp_;
      device_work_pending_ = other.device_work_pending_;
      state_ = other.state_;
      other.destination_ = nullptr;
      other.device_work_pending_ = false;
      other.state_ = State::kInvalid;
    }
    return *this;
  }
  ~HaloExchange() { abandon(); }

 private:
  enum class State : std::uint8_t { kInvalid, kActive, kConsumed };

  std::shared_ptr<HaloExchangeStorage> storage_;
  const MultiFab* destination_ = nullptr;
  int destination_ngrow_ = 0;
  int destination_ncomp_ = 0;
  bool device_work_pending_ = false;
  State state_ = State::kInvalid;

  explicit HaloExchange(const MultiFab& destination,
                        std::shared_ptr<HaloExchangeStorage> storage = {})
      : storage_(std::move(storage)),
        destination_(&destination),
        destination_ngrow_(destination.n_grow()),
        destination_ncomp_(destination.ncomp()),
        state_(State::kActive) {}

  // Dropping a begin handle is legal RAII abandonment, but its MPI buffers remain owned by active
  // requests.  Complete those requests before returning the lease to the pool.  If MPI is no longer
  // callable or reports an error, leave in_use=true permanently: leaking one cached lease is safer
  // than recycling memory still owned by MPI.
  void abandon() noexcept {
    if (state_ != State::kActive) {
      storage_.reset();
      return;
    }
    // Local copies (and the serial/one-rank path in particular) are asynchronous Kokkos work even
    // when no MPI_Request exists.  The destination must outlive them just as the communication
    // buffers must outlive MPI.  A device failure at this point is fatal by definition; allowing the
    // destructor to continue and release live storage would be undefined behaviour.
    if (device_work_pending_) {
      device_fence();
      device_work_pending_ = false;
    }
    bool reusable = true;
#ifdef POPS_HAS_MPI
    if (storage_ && !storage_->reqs.empty()) {
      int initialized = 0, finalized = 0;
      if (MPI_Initialized(&initialized) != MPI_SUCCESS || !initialized ||
          MPI_Finalized(&finalized) != MPI_SUCCESS || finalized) {
        reusable = false;
      } else {
        const int error = MPI_Waitall(static_cast<int>(storage_->reqs.size()),
                                      storage_->reqs.data(), MPI_STATUSES_IGNORE);
        reusable = error == MPI_SUCCESS;
        if (reusable)
          storage_->reqs.clear();
      }
    }
#endif
    if (storage_ && reusable) {
      storage_->in_use = false;
    }
    storage_.reset();
    destination_ = nullptr;
    state_ = State::kInvalid;
  }

  friend HaloExchange detail::fill_boundary_begin_on(MultiFab&, const Box2D&, Periodicity,
                                                     const CommunicatorView&, int,
                                                     const std::vector<PeriodicIdentification2D>*);
  friend void fill_boundary_end(MultiFab&, HaloExchange&);
};

/// Phase 1 (non-blocking): does the LOCAL halo copies and posts the Isend/Irecv of the distant halos.
/// Returns the handle to pass to fill_boundary_end. Between begin and end the caller can advance the
/// interior. No-op if mf has no ghost. @p domain is used for periodic wrapping @p per.
namespace detail {

inline HaloExchange fill_boundary_begin_on(
    MultiFab& mf, const Box2D& domain, Periodicity per, const CommunicatorView& communicator,
    int message_tag, const std::vector<PeriodicIdentification2D>* mapped_periodic) {
  const int ng = mf.n_grow();
  if (ng == 0)
    return HaloExchange(mf);
  const int nc = mf.ncomp();
  std::shared_ptr<const HaloSchedule> sched;
  std::shared_ptr<HaloExchangeStorage> lease;
  HaloExchange h;

#ifdef POPS_HAS_MPI
  if (communicator.size() > 1) {
    // Schedule construction allocates rank-local storage once.  It is part of the pre-post
    // transaction too: no rank may continue to a later collective if a peer failed here.
    collectively_prepare_before_halo_post(communicator, [&] {
      sched = mapped_periodic == nullptr
                  ? get_halo_schedule(mf, domain, per, communicator)
                  : get_mapped_halo_schedule(mf, domain, *mapped_periodic, communicator);
    });
    try {
      collectively_prepare_before_halo_post(
          communicator, [&] { lease = mf.halo_cache().acquire_exchange(sched, nc); });
    } catch (...) {
      // A peer may have failed while this rank successfully acquired a reusable lease.  Return it
      // before propagating the uniform collective failure.
      if (lease)
        lease->in_use = false;
      throw;
    }
    h = HaloExchange(mf, std::move(lease));
  } else
#endif
  {
    // Serial and MPI-size-one execution has no peer that could be stranded.
    sched = mapped_periodic == nullptr
                ? get_halo_schedule(mf, domain, per, communicator)
                : get_mapped_halo_schedule(mf, domain, *mapped_periodic, communicator);
    lease = mf.halo_cache().acquire_exchange(sched, nc);
    h = HaloExchange(mf, std::move(lease));
  }
  HaloExchangeStorage& storage = *h.storage_;

  auto launch_local_copies = [&] {
    // Local copies are deliberately published only after every distributed request has been posted.
    // A failed pre-post transaction therefore leaves the destination ghosts untouched.
    for (const HaloJob& job : sched->local) {
      Fab2D& dst = mf.fab(mf.local_index_of(job.dst));
      const Fab2D& src = mf.fab(mf.local_index_of(job.src));
      h.device_work_pending_ = true;
      if (job.mapped)
        detail::copy_mapped(dst, src, job, nc);
      else
        detail::copy_shifted(dst, src, job.region, job.sx, job.sy, nc);
    }
  };

#ifdef POPS_HAS_MPI
  if (communicator.size() <= 1) {
    launch_local_copies();
    return h;
  }
  const int np = communicator.size();
  if (all_reduce_max(np > std::numeric_limits<int>::max() / 2 ? 1L : 0L, communicator) != 0)
    throw std::overflow_error(
        "fill_boundary: execution communicator exceeds the MPI request-count range");
  bool payload_overflow = false;
  auto buf_size = [&](const std::vector<HaloJob>& js) {
    std::int64_t n = 0;
    for (const auto& j : js) {
      const std::int64_t cells = j.region.num_cells();
      if (cells < 0 || nc < 0 || (nc != 0 && cells > (std::numeric_limits<int>::max() - n) / nc)) {
        payload_overflow = true;
        return std::int64_t{0};
      }
      n += cells * nc;
    }
    return n;
  };
  collectively_prepare_before_halo_post(communicator, [&] {
    storage.send_sizes.assign(static_cast<std::size_t>(np), 0);
    storage.recv_sizes.assign(static_cast<std::size_t>(np), 0);
  });
  for (int rank = 0; rank < np; ++rank) {
    storage.send_sizes[static_cast<std::size_t>(rank)] = buf_size(sched->send[rank]);
    storage.recv_sizes[static_cast<std::size_t>(rank)] = buf_size(sched->recv[rank]);
  }
  // Every rank reaches this preflight collective before anybody posts a request.  A malformed or
  // oversized local schedule therefore fails on the complete execution communicator instead of
  // stranding peers in MPI_Waitall after only one rank throws.
  if (all_reduce_max(payload_overflow ? 1L : 0L, communicator) != 0)
    throw std::overflow_error(
        "fill_boundary: one peer halo payload exceeds the portable MPI int-count limit");
  // Complete every host/pinned-buffer allocation, including request-vector capacity, before the
  // collective success witness.  The posting loop below is then allocation-free.
  collectively_prepare_before_halo_post(communicator, [&] {
    storage.sbuf.resize(static_cast<std::size_t>(np));
    storage.rbuf.resize(static_cast<std::size_t>(np));
    storage.reqs.clear();
    storage.reqs.reserve(static_cast<std::size_t>(2) * static_cast<std::size_t>(np));
    for (int rank = 0; rank < np; ++rank) {
      auto& send = storage.sbuf[static_cast<std::size_t>(rank)];
      auto& recv = storage.rbuf[static_cast<std::size_t>(rank)];
      send.clear();
      recv.clear();
      if (!sched->send[rank].empty())
        send.resize(static_cast<std::size_t>(storage.send_sizes[static_cast<std::size_t>(rank)]));
      if (!sched->recv[rank].empty())
        recv.resize(static_cast<std::size_t>(storage.recv_sizes[static_cast<std::size_t>(rank)]));
    }
  });
  // Device PACK and its completion fence are the final fallible pre-post phase.  Only a collective
  // success witness permits any rank to enter the posting loop.
  collectively_prepare_before_halo_post(communicator, [&] {
    // Per-job layout: c-major then (jj, ii), IDENTICAL to the old k++ order. The peer rank enumerates
    // in the same order, so sbuf[A->B] and rbuf[B<-A] align without negotiating sizes.
    for (int r = 0; r < np; ++r) {
      const std::vector<HaloJob>& send_r = sched->send[r];
      if (send_r.empty())
        continue;
      Real* sb = storage.sbuf[static_cast<std::size_t>(r)].data();
      std::int64_t base = 0;
      for (const auto& jb : send_r) {
        const ConstArray4 s = mf.fab(mf.local_index_of(jb.src)).const_array();
        const int lo0 = jb.region.lo[0], lo1 = jb.region.lo[1], rnx = jb.region.nx();
        const std::int64_t rsz = static_cast<std::int64_t>(rnx) * jb.region.ny();
        const int sx = jb.sx, sy = jb.sy, ncl = nc;
        const std::int64_t b0 = base;
        h.device_work_pending_ = true;
        if (jb.mapped)
          for_each_cell(jb.region,
                        detail::PackMappedKernel{
                            sb, s, b0, rsz, lo0, lo1, rnx, ncl, jb.source_from_destination_axis[0],
                            jb.source_from_destination_axis[1], jb.source_sign[0],
                            jb.source_sign[1], jb.source_offset[0], jb.source_offset[1]});
        else
          for_each_cell(jb.region, detail::PackKernel{sb, s, b0, rsz, lo0, lo1, rnx, sx, sy, ncl});
        base += rsz * nc;
      }
    }
    device_fence();
    h.device_work_pending_ = false;
  });
  for (
      int r = 0; r < np;
      ++r) {  // non-blocking posting; MPI receives PINNED HOST pointers (seen HOST, no GPUDirect/CUDA IPC)
    if (!storage.sbuf[static_cast<std::size_t>(r)].empty()) {
      storage.reqs.emplace_back();
      detail::require_mpi_success(
          MPI_Isend(storage.sbuf[static_cast<std::size_t>(r)].data(),
                    static_cast<int>(storage.sbuf[static_cast<std::size_t>(r)].size()), MPI_DOUBLE,
                    r, message_tag, communicator.native_handle(), &storage.reqs.back()),
          "MPI_Isend(fill_boundary)");
    }
    if (!storage.rbuf[static_cast<std::size_t>(r)].empty()) {
      storage.reqs.emplace_back();
      detail::require_mpi_success(
          MPI_Irecv(storage.rbuf[static_cast<std::size_t>(r)].data(),
                    static_cast<int>(storage.rbuf[static_cast<std::size_t>(r)].size()), MPI_DOUBLE,
                    r, message_tag, communicator.native_handle(), &storage.reqs.back()),
          "MPI_Irecv(fill_boundary)");
    }
  }
  launch_local_copies();
#else
  launch_local_copies();
#endif
  return h;
}

}  // namespace detail

/// Process-world compatibility path. Concurrent prepared execution must use the ExecutionLane
/// overload below so independent MPI traces receive distinct communicator context ids.
inline HaloExchange fill_boundary_begin(MultiFab& mf, const Box2D& domain, Periodicity per = {}) {
  return detail::fill_boundary_begin_on(mf, domain, per, world_communicator_view(),
                                        ExecutionLane::halo_message_tag, nullptr);
}

/// Begin one halo exchange on an explicitly prepared execution lane.
inline HaloExchange fill_boundary_begin(MultiFab& mf, const Box2D& domain,
                                        const ExecutionLane& lane, Periodicity per = {}) {
  return detail::fill_boundary_begin_on(mf, domain, per, lane.communicator(),
                                        ExecutionLane::halo_message_tag, nullptr);
}

/// Phase 2 (blocking): MPI_Waitall on the transfers posted by begin, then unpacks the received
/// buffers into the ghosts. @p h MUST come from the matching fill_boundary_begin on the same mf. No-op
/// in serial (no request).
inline void fill_boundary_end(MultiFab& mf, HaloExchange& h) {
  if (h.state_ == HaloExchange::State::kConsumed)
    throw std::logic_error("fill_boundary_end: halo exchange handle was already consumed");
  if (h.state_ != HaloExchange::State::kActive || h.destination_ == nullptr)
    throw std::logic_error("fill_boundary_end: invalid halo exchange handle");
  if (h.destination_ != &mf)
    throw std::invalid_argument(
        "fill_boundary_end: destination differs from the MultiFab passed to fill_boundary_begin");
  const HaloSchedule* const schedule = h.storage_ ? h.storage_->sched.get() : nullptr;
  if ((schedule != nullptr &&
       (schedule->boxes != mf.box_array().boxes() || schedule->ranks != mf.dmap().ranks() ||
        schedule->ngrow != mf.n_grow())) ||
      h.destination_ngrow_ != mf.n_grow() || h.destination_ncomp_ != mf.ncomp())
    throw std::invalid_argument(
        "fill_boundary_end: destination layout changed after fill_boundary_begin");
#ifdef POPS_HAS_MPI
  if (h.storage_ && !h.storage_->reqs.empty()) {
    HaloExchangeStorage& storage = *h.storage_;
    detail::require_mpi_success(MPI_Waitall(static_cast<int>(storage.reqs.size()),
                                            storage.reqs.data(), MPI_STATUSES_IGNORE),
                                "MPI_Waitall(fill_boundary)");
    storage.reqs.clear();
    // device UNPACK (for_each) from the received PINNED HOST buffers. Waitall guarantees the transfer
    // is complete; the kernel launched next reads the pinned host (device-accessible, coherent).
    const HaloSchedule& sched = *storage.sched;
    for (std::size_t r = 0; r < sched.recv.size(); ++r) {
      if (storage.rbuf[r].empty())
        continue;
      const Real* rb = storage.rbuf[r].data();
      std::int64_t base = 0;
      for (const auto& jb : sched.recv[r]) {
        Array4 d = mf.fab(mf.local_index_of(jb.dst)).array();
        const int lo0 = jb.region.lo[0], lo1 = jb.region.lo[1], rnx = jb.region.nx();
        const std::int64_t rsz = static_cast<std::int64_t>(rnx) * jb.region.ny();
        const int ncl = storage.ncomp;
        const std::int64_t b0 = base;
        for_each_cell(jb.region, detail::UnpackKernel{rb, d, b0, rsz, lo0, lo1, rnx, ncl});
        h.device_work_pending_ = true;
        base += rsz * ncl;
      }
    }
    // The unpack kernels above are ASYNC and read persistent pinned buffers. Drain the device before
    // returning the lease to the pool: a later fill may immediately overwrite the same capacity.
    device_fence();
    h.device_work_pending_ = false;
  }
#else
  (void)mf;
#endif
  // Serial and MPI-size-one exchanges have no requests, but their local copy kernels are still
  // asynchronous.  `end` is the publication barrier promised by the API.
  if (h.device_work_pending_) {
    device_fence();
    h.device_work_pending_ = false;
  }
  if (h.storage_) {
    h.storage_->in_use = false;
    h.storage_.reset();
  }
  h.destination_ = nullptr;
  h.state_ = HaloExchange::State::kConsumed;
}

/// BLOCKING halo exchange: begin then end immediately (no overlap). Fills the intra-level +
/// periodic ghosts of @p mf; @p per sets the wrapping, @p domain the periodic fold.
inline void fill_boundary(MultiFab& mf, const Box2D& domain, Periodicity per = {}) {
  HaloExchange h = fill_boundary_begin(mf, domain, per);
  fill_boundary_end(mf, h);
}

/// BLOCKING halo exchange isolated from other concurrent lanes by the duplicated communicator.
inline void fill_boundary(MultiFab& mf, const Box2D& domain, const ExecutionLane& lane,
                          Periodicity per = {}) {
  HaloExchange h = fill_boundary_begin(mf, domain, lane, per);
  fill_boundary_end(mf, h);
}

/// Blocking same-level/MPI exchange plus one explicit signed/permuted periodic identification.
/// The ordinary translation-only API remains the exact historical path.
inline void fill_mapped_boundary(MultiFab& mf, const Box2D& domain,
                                 const std::vector<PeriodicIdentification2D>& identifications) {
  HaloExchange exchange =
      detail::fill_boundary_begin_on(mf, domain, Periodicity{}, world_communicator_view(),
                                     ExecutionLane::halo_message_tag, &identifications);
  fill_boundary_end(mf, exchange);
}

inline void fill_mapped_boundary(MultiFab& mf, const Box2D& domain, const ExecutionLane& lane,
                                 const std::vector<PeriodicIdentification2D>& identifications) {
  HaloExchange exchange =
      detail::fill_boundary_begin_on(mf, domain, Periodicity{}, lane.communicator(),
                                     ExecutionLane::halo_message_tag, &identifications);
  fill_boundary_end(mf, exchange);
}

}  // namespace pops
