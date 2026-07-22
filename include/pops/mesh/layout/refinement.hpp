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
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>

#include <pops/core/foundation/allocator.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/foundation/validation.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/copy_schedule.hpp>  // memoized parallel_copy schedule (ADC-607)
#include <pops/mesh/index/box_hash.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>  // detail::copy_shifted
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace pops {

/// Persistent communication lease for one exact redistribution plan and execution communicator.
/// Pinned buffers are directly accessible to Kokkos pack/unpack kernels but remain host pointers
/// from MPI's point of view. The lease pool lives with the destination MultiFab copy cache.
struct CopyExchangeStorage {
  std::shared_ptr<const CopySchedule> schedule;
  int ncomp = 0;
  std::int64_t communicator_identity = 0;
  bool in_use = false;
#ifdef POPS_HAS_MPI
  std::vector<std::vector<Real, comm_allocator<Real>>> send_buffers;
  std::vector<std::vector<Real, comm_allocator<Real>>> receive_buffers;
  std::vector<MPI_Request> requests;
#endif
};

inline std::shared_ptr<CopyExchangeStorage> CopyScheduleCache::acquire_exchange(
    const std::shared_ptr<const CopySchedule>& schedule, int ncomp,
    std::int64_t communicator_identity) {
  for (const auto& candidate : exchange_pool_) {
    if (!candidate->in_use && candidate->schedule.get() == schedule.get() &&
        candidate->ncomp == ncomp &&
        candidate->communicator_identity == communicator_identity) {
      candidate->in_use = true;
      return candidate;
    }
  }
  auto storage = std::make_shared<CopyExchangeStorage>();
  storage->schedule = schedule;
  storage->ncomp = ncomp;
  storage->communicator_identity = communicator_identity;
  storage->in_use = true;
  exchange_pool_.push_back(storage);
  return storage;
}

namespace detail {

template <class Prepare>
inline void collectively_prepare_before_parallel_copy_post(
    const CommunicatorView& communicator, Prepare&& prepare) {
  if (communicator.size() <= 1) {
    std::forward<Prepare>(prepare)();
    return;
  }
  long local_failure = 0;
  try {
    std::forward<Prepare>(prepare)();
  } catch (const std::bad_alloc&) {
    local_failure = 1;
  } catch (const std::invalid_argument&) {
    local_failure = 2;
  } catch (const std::overflow_error&) {
    local_failure = 3;
  } catch (...) {
    local_failure = 4;
  }
  const long global_failure = all_reduce_max(local_failure, communicator);
  if (global_failure == 1)
    throw std::bad_alloc();
  if (global_failure == 2)
    throw std::invalid_argument(
        "parallel_copy: one rank rejected the distributed copy contract");
  if (global_failure == 3)
    throw std::overflow_error(
        "parallel_copy: one peer payload exceeds the portable MPI int-count limit");
  if (global_failure != 0)
    throw std::runtime_error(
        "parallel_copy: one rank failed while preparing distributed communication");
}

inline std::int64_t copy_job_cell_count(const std::vector<CopyJob>& jobs) {
  std::int64_t result = 0;
  for (const CopyJob& job : jobs) {
    const std::int64_t cells = job.region.num_cells();
    if (cells < 0 || cells > std::numeric_limits<std::int64_t>::max() - result)
      throw std::overflow_error("parallel_copy schedule cell count exceeds int64 capacity");
    result += cells;
  }
  return result;
}

inline int checked_parallel_copy_payload_count(std::int64_t cells, int ncomp) {
  if (cells < 0 || ncomp < 1 ||
      cells > static_cast<std::int64_t>(std::numeric_limits<int>::max()) / ncomp)
    throw std::overflow_error(
        "parallel_copy: one peer payload exceeds the portable MPI int-count limit");
  return static_cast<int>(cells * ncomp);
}

inline void append_copy_contract_u32(std::string& bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<char>((value >> shift) & 0xffu));
}

inline void append_copy_contract_int(std::string& bytes, int value) {
  append_copy_contract_u32(bytes, static_cast<std::uint32_t>(value));
}

inline void append_copy_contract_layout(std::string& bytes, const BoxArray& boxes,
                                        const DistributionMapping& mapping) {
  append_copy_contract_int(bytes, boxes.size());
  for (const Box2D& box : boxes.boxes()) {
    append_copy_contract_int(bytes, box.lo[0]);
    append_copy_contract_int(bytes, box.lo[1]);
    append_copy_contract_int(bytes, box.hi[0]);
    append_copy_contract_int(bytes, box.hi[1]);
  }
  append_copy_contract_int(bytes, mapping.size());
  for (int owner : mapping.ranks())
    append_copy_contract_int(bytes, owner);
}

inline std::string make_parallel_copy_contract_payload(const MultiFab& dst, const MultiFab& src,
                                                       int communicator_size, int message_tag) {
  std::string bytes;
  const std::size_t entries = static_cast<std::size_t>(dst.box_array().size()) +
                              static_cast<std::size_t>(src.box_array().size());
  if (entries > (std::numeric_limits<std::size_t>::max() - 32u) / 20u)
    throw std::length_error("parallel_copy layout contract exceeds size_t capacity");
  bytes.reserve(32u + 20u * entries);
  append_copy_contract_int(bytes, communicator_size);
  append_copy_contract_int(bytes, message_tag);
  append_copy_contract_int(bytes, dst.ncomp());
  append_copy_contract_layout(bytes, dst.box_array(), dst.dmap());
  append_copy_contract_layout(bytes, src.box_array(), src.dmap());
  return bytes;
}

inline std::int64_t parallel_copy_communicator_identity(
    const CommunicatorView& communicator) noexcept {
#ifdef POPS_HAS_MPI
  if (communicator.active())
    return static_cast<std::int64_t>(MPI_Comm_c2f(communicator.native_handle()));
#else
  (void)communicator;
#endif
  return 0;
}

// Enumerates the parallel_copy schedule for (dst layout, src layout): the BoxHash build over the SRC
// BoxArray + the local (dst AND src local) and, under MPI with n_ranks()>1, the global (cross-rank
// send/recv) job lists. This is the per-call work that ADC-607 hoists out of parallel_copy; it runs
// ONCE per distinct (dst layout, src layout) pair and is then replayed from the cache. Jobs are
// produced in the SAME deterministic order as the legacy inline loops (local: dst-local x sorted src
// candidates; global: gd x sorted src candidates), so the packed buffers stay bit-identical and the
// per-rank send/recv lists stay aligned. Bumps the build counter (cache-engagement test hook).
inline void build_copy_schedule(const MultiFab& dst, const MultiFab& src,
                                const CommunicatorView& communicator, CopySchedule& sched) {
  copy_schedule_build_counter().fetch_add(1, std::memory_order_relaxed);
  sched.ncomp = dst.ncomp();
  sched.communicator_size = communicator.size();
  sched.communicator_rank = communicator.rank();
  sched.communicator_identity = parallel_copy_communicator_identity(communicator);
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
  const int np = communicator.size();
  if (np > 1) {
    const int me = communicator.rank();
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
    sched.send_cells.resize(static_cast<std::size_t>(np));
    sched.recv_cells.resize(static_cast<std::size_t>(np));
    for (int peer = 0; peer < np; ++peer) {
      sched.send_cells[static_cast<std::size_t>(peer)] = copy_job_cell_count(sched.send[peer]);
      sched.recv_cells[static_cast<std::size_t>(peer)] = copy_job_cell_count(sched.recv[peer]);
    }
  }
#endif
}

// Returns an authenticated cached schedule. A cold build is kept private until every rank has built
// successfully and reserved cache capacity; publication is then allocation-free and collective.
inline std::shared_ptr<const CopySchedule> get_copy_schedule_collectively(
    const MultiFab& dst, const MultiFab& src, const CommunicatorView& communicator,
    int message_tag) {
  CopyScheduleCache& cache = dst.copy_cache();
  const int communicator_size = communicator.size();
  const int communicator_rank = communicator.rank();
  const std::int64_t communicator_identity =
      parallel_copy_communicator_identity(communicator);
  std::shared_ptr<const CopySchedule> schedule =
      cache.find(src.box_array(), src.dmap(), dst.ncomp(), communicator_size,
                 communicator_rank, communicator_identity, message_tag);
  const long local_state = dst.ncomp() != src.ncomp() ? 2L : (schedule ? 0L : 1L);
  const long global_state = all_reduce_max(local_state, communicator);
  if (global_state >= 2L)
    throw std::invalid_argument(
        "parallel_copy requires identical source and destination provider widths on every rank");
  if (global_state == 0L) {
    copy_schedule_hit_counter().fetch_add(1, std::memory_order_relaxed);
    return schedule;
  }

  std::string contract_payload;
  collectively_prepare_before_parallel_copy_post(communicator, [&] {
    contract_payload =
        make_parallel_copy_contract_payload(dst, src, communicator_size, message_tag);
  });
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("parallel-copy-layout-v2"),
            std::string_view(contract_payload)}},
          communicator))
    throw std::invalid_argument(
        "parallel_copy source/destination layouts differ between MPI ranks");

  std::shared_ptr<CopySchedule> pending;
  collectively_prepare_before_parallel_copy_post(communicator, [&] {
    if (schedule) {
      copy_schedule_hit_counter().fetch_add(1, std::memory_order_relaxed);
      return;
    }
    copy_schedule_miss_counter().fetch_add(1, std::memory_order_relaxed);
    pending = std::make_shared<CopySchedule>();
    pending->key.boxes = src.box_array().boxes();
    pending->key.ranks = src.dmap().ranks();
    pending->message_tag = message_tag;
    build_copy_schedule(dst, src, communicator, *pending);
    cache.reserve_for_append();
  });
  if (pending) {
    cache.publish_prepared(pending);
    schedule = std::move(pending);
  }
  return schedule;
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
                           "refinement/coarsening ratio r >= 1", "r=" + std::to_string(r));
  std::vector<Box2D> b;
  b.reserve(ba.size());
  for (int i = 0; i < ba.size(); ++i)
    b.push_back(ba[i].coarsen(r));
  return BoxArray{std::move(b)};
}

/// Copies the valid regions that OVERLAP from src to dst (same indices, no shift).
/// General redistribution between two MultiFab over the same domain with different decompositions.
/// Provider widths must be identical; a dst cell not covered by src is left intact.
namespace detail {

class CopyExchangeLease {
 public:
  CopyExchangeLease() = default;
  CopyExchangeLease(const CopyExchangeLease&) = delete;
  CopyExchangeLease& operator=(const CopyExchangeLease&) = delete;
  ~CopyExchangeLease() { abandon_(); }

  void attach(std::shared_ptr<CopyExchangeStorage> storage) noexcept {
    storage_ = std::move(storage);
  }
  [[nodiscard]] CopyExchangeStorage& storage() noexcept { return *storage_; }
  void device_work_will_launch() noexcept { device_work_pending_ = true; }
  void device_work_completed() noexcept { device_work_pending_ = false; }

  void release_after_success() noexcept {
    if (storage_)
      storage_->in_use = false;
    storage_.reset();
  }

 private:
  std::shared_ptr<CopyExchangeStorage> storage_;
  bool device_work_pending_ = false;

  void abandon_() noexcept {
    if (!storage_)
      return;
    bool reusable = true;
#ifdef POPS_HAS_MPI
    if (!storage_->requests.empty()) {
      int initialized = 0;
      int finalized = 0;
      if (MPI_Initialized(&initialized) != MPI_SUCCESS || !initialized ||
          MPI_Finalized(&finalized) != MPI_SUCCESS || finalized) {
        reusable = false;
      } else {
        const int error = MPI_Waitall(static_cast<int>(storage_->requests.size()),
                                      storage_->requests.data(), MPI_STATUSES_IGNORE);
        reusable = error == MPI_SUCCESS;
        if (reusable)
          storage_->requests.clear();
      }
    }
#endif
    if (device_work_pending_) {
      device_fence();
      device_work_pending_ = false;
    }
    if (reusable)
      storage_->in_use = false;
    storage_.reset();
  }
};

inline void parallel_copy_on(MultiFab& dst, const MultiFab& src,
                             const CommunicatorView& communicator, int message_tag) {
  // memoized schedule (BoxHash + enumeration) for this (dst layout, src layout) pair. Replayed in the
  // SAME order as the legacy inline loops -> bit-identical to the per-call rebuild (ADC-607).
  const std::shared_ptr<const CopySchedule> sched =
      get_copy_schedule_collectively(dst, src, communicator, message_tag);
  const int nc = dst.ncomp();

  auto launch_local_copies = [&] {
    for (const CopyJob& job : sched->local) {
      Fab2D& destination = dst.fab(dst.local_index_of(job.gd));
      const Fab2D& source = src.fab(src.local_index_of(job.gs));
      detail::copy_shifted(destination, source, job.region, 0, 0, nc);
    }
  };

#ifdef POPS_HAS_MPI
  if (communicator.size() <= 1) {
    launch_local_copies();
    device_fence();
    return;
  }
  const int np = communicator.size();
  CopyExchangeLease lease;
  collectively_prepare_before_parallel_copy_post(communicator, [&] {
    if (np > std::numeric_limits<int>::max() / 2 ||
        sched->send.size() != static_cast<std::size_t>(np) ||
        sched->recv.size() != static_cast<std::size_t>(np) ||
        sched->send_cells.size() != static_cast<std::size_t>(np) ||
        sched->recv_cells.size() != static_cast<std::size_t>(np))
      throw std::overflow_error("parallel_copy communication schedule has invalid peer capacity");

    std::shared_ptr<CopyExchangeStorage> storage = dst.copy_cache().acquire_exchange(
        sched, nc, parallel_copy_communicator_identity(communicator));
    lease.attach(std::move(storage));
    CopyExchangeStorage& exchange = lease.storage();
    exchange.send_buffers.resize(static_cast<std::size_t>(np));
    exchange.receive_buffers.resize(static_cast<std::size_t>(np));
    exchange.requests.clear();
    exchange.requests.reserve(static_cast<std::size_t>(2) * static_cast<std::size_t>(np));

    for (int peer = 0; peer < np; ++peer) {
      const int send_count = checked_parallel_copy_payload_count(
          sched->send_cells[static_cast<std::size_t>(peer)], nc);
      const int receive_count = checked_parallel_copy_payload_count(
          sched->recv_cells[static_cast<std::size_t>(peer)], nc);
      exchange.send_buffers[static_cast<std::size_t>(peer)].resize(
          static_cast<std::size_t>(send_count));
      exchange.receive_buffers[static_cast<std::size_t>(peer)].resize(
          static_cast<std::size_t>(receive_count));
    }

    // Device pack layout is c-major then (j,i), byte-for-byte identical to the historical host
    // loops. All pack work is drained before MPI may observe the pinned host pointers.
    for (int peer = 0; peer < np; ++peer) {
      Real* buffer = exchange.send_buffers[static_cast<std::size_t>(peer)].data();
      std::int64_t base = 0;
      for (const CopyJob& job : sched->send[peer]) {
        const ConstArray4 source = src.fab(src.local_index_of(job.gs)).const_array();
        const int nx = job.region.nx();
        const std::int64_t region_size =
            static_cast<std::int64_t>(nx) * job.region.ny();
        lease.device_work_will_launch();
        for_each_cell(job.region,
                      detail::PackKernel{buffer, source, base, region_size, job.region.lo[0],
                                         job.region.lo[1], nx, 0, 0, nc});
        base += region_size * nc;
      }
    }
    device_fence();
    lease.device_work_completed();
  });

  CopyExchangeStorage& exchange = lease.storage();
  for (int peer = 0; peer < np; ++peer) {
    auto& send = exchange.send_buffers[static_cast<std::size_t>(peer)];
    if (!send.empty()) {
      MPI_Request request = MPI_REQUEST_NULL;
      const int error = MPI_Isend(send.data(), static_cast<int>(send.size()), MPI_DOUBLE, peer,
                                  message_tag, communicator.native_handle(), &request);
      if (request != MPI_REQUEST_NULL)
        exchange.requests.push_back(request);
      detail::require_mpi_success(error, "MPI_Isend(parallel_copy)");
    }
    auto& receive = exchange.receive_buffers[static_cast<std::size_t>(peer)];
    if (!receive.empty()) {
      MPI_Request request = MPI_REQUEST_NULL;
      const int error = MPI_Irecv(receive.data(), static_cast<int>(receive.size()), MPI_DOUBLE,
                                  peer, message_tag, communicator.native_handle(), &request);
      if (request != MPI_REQUEST_NULL)
        exchange.requests.push_back(request);
      detail::require_mpi_success(error, "MPI_Irecv(parallel_copy)");
    }
  }

  // Publish rank-local overlap only after every distributed phase has passed preflight and every
  // peer request is live. Waitall overlaps that Kokkos work with network progress.
  if (!sched->local.empty())
    lease.device_work_will_launch();
  launch_local_copies();
  if (!exchange.requests.empty()) {
    detail::require_mpi_success(
        MPI_Waitall(static_cast<int>(exchange.requests.size()), exchange.requests.data(),
                    MPI_STATUSES_IGNORE),
        "MPI_Waitall(parallel_copy)");
    exchange.requests.clear();
  }

  for (int peer = 0; peer < np; ++peer) {
    const Real* buffer = exchange.receive_buffers[static_cast<std::size_t>(peer)].data();
    std::int64_t base = 0;
    for (const CopyJob& job : sched->recv[peer]) {
      Array4 destination = dst.fab(dst.local_index_of(job.gd)).array();
      const int nx = job.region.nx();
      const std::int64_t region_size = static_cast<std::int64_t>(nx) * job.region.ny();
      lease.device_work_will_launch();
      for_each_cell(job.region,
                    detail::UnpackKernel{buffer, destination, base, region_size,
                                         job.region.lo[0], job.region.lo[1], nx, nc});
      base += region_size * nc;
    }
  }
  device_fence();
  lease.device_work_completed();
  lease.release_after_success();
#else
  launch_local_copies();
  device_fence();
#endif
}

}  // namespace detail

/// Process-world compatibility path. Concurrent prepared execution must pass an ExecutionLane.
inline void parallel_copy(MultiFab& dst, const MultiFab& src) {
  detail::parallel_copy_on(dst, src, world_communicator_view(),
                           ExecutionLane::parallel_copy_message_tag);
}

/// Explicit-communicator control path. The caller owns collective ordering on this communicator;
/// prepared concurrent execution should still retain an ExecutionLane so communicator lifetime and
/// identity remain authenticated by its owner.
inline void parallel_copy(MultiFab& dst, const MultiFab& src,
                          const CommunicatorView& communicator) {
  detail::parallel_copy_on(dst, src, communicator, ExecutionLane::parallel_copy_message_tag);
}

/// Redistribution isolated from other concurrent execution sessions by the lane communicator.
inline void parallel_copy(MultiFab& dst, const MultiFab& src, const ExecutionLane& lane) {
  detail::parallel_copy_on(dst, src, lane.communicator(), ExecutionLane::parallel_copy_message_tag);
}

/// Allocation-free replay object for a periodic redistribution into a persistent carrier.
///
/// Preparation authenticates the exact source/destination layouts and materializes the periodic
/// image catalogue once.  It also performs one copy, warming the ordinary ParallelCopy schedule
/// and its pinned MPI buffers.  Subsequent apply() calls only launch image-copy kernels and replay
/// the prepared redistribution.  A topology generation is part of the contract even when two
/// successive hierarchies happen to have byte-identical boxes: a workspace is never silently
/// carried across a regrid transaction.
class PreparedPeriodicCopyPlan {
 public:
  PreparedPeriodicCopyPlan(const PreparedPeriodicCopyPlan&) = delete;
  PreparedPeriodicCopyPlan& operator=(const PreparedPeriodicCopyPlan&) = delete;
  PreparedPeriodicCopyPlan(PreparedPeriodicCopyPlan&&) noexcept = default;
  PreparedPeriodicCopyPlan& operator=(PreparedPeriodicCopyPlan&&) noexcept = default;

  static PreparedPeriodicCopyPlan prepare(
      MultiFab& destination, const MultiFab& source, const Box2D& domain,
      Periodicity periodicity, std::uint64_t topology_generation,
      const CommunicatorView& communicator,
      int message_tag = ExecutionLane::parallel_copy_message_tag) {
    std::unique_ptr<PreparedPeriodicCopyPlan> pending;
    detail::collectively_prepare_before_parallel_copy_post(communicator, [&] {
      pending.reset(new PreparedPeriodicCopyPlan(destination, source, domain, periodicity,
                                                 topology_generation, communicator, message_tag));
    });
    pending->apply(destination, source, topology_generation, communicator);
    return std::move(*pending);
  }

  void apply(MultiFab& destination, const MultiFab& source,
             std::uint64_t topology_generation,
             const CommunicatorView& communicator) {
    detail::collectively_prepare_before_parallel_copy_post(communicator, [&] {
      validate_replay_(destination, source, topology_generation, communicator);
      for (int local = 0; local < images_.local_size(); ++local) {
        const int global = images_.global_index(local);
        const int source_global = image_sources_[static_cast<std::size_t>(global)];
        const int source_local = source.local_index_of(source_global);
        if (source_local < 0)
          throw std::runtime_error(
              "prepared periodic copy image/source ownership mismatch");
        const auto shift = image_shifts_[static_cast<std::size_t>(global)];
        detail::copy_shifted(images_.fab(local), source.fab(source_local), images_.box(local),
                             shift[0], shift[1], source.ncomp());
      }
      device_fence();
    });
    detail::parallel_copy_on(destination, images_, communicator, message_tag_);
  }

  [[nodiscard]] std::uint64_t topology_generation() const noexcept {
    return topology_generation_;
  }

 private:
  PreparedPeriodicCopyPlan(MultiFab& destination, const MultiFab& source,
                           const Box2D& domain, Periodicity periodicity,
                           std::uint64_t topology_generation,
                           const CommunicatorView& communicator, int message_tag)
      : destination_boxes_(destination.box_array().boxes()),
        destination_ranks_(destination.dmap().ranks()),
        source_boxes_(source.box_array().boxes()),
        source_ranks_(source.dmap().ranks()),
        destination_ngrow_(destination.n_grow()),
        source_ngrow_(source.n_grow()),
        ncomp_(source.ncomp()),
        domain_(domain),
        periodicity_(periodicity),
        topology_generation_(topology_generation),
        communicator_size_(communicator.size()),
        communicator_rank_(communicator.rank()),
        communicator_identity_(detail::parallel_copy_communicator_identity(communicator)),
        message_tag_(message_tag),
        images_(make_images_(destination, source, domain, periodicity,
                             image_sources_, image_shifts_)) {
    validate_replay_(destination, source, topology_generation, communicator);
  }

  static MultiFab make_images_(
      const MultiFab& destination, const MultiFab& source, const Box2D& domain,
      Periodicity periodicity, std::vector<int>& image_sources,
      std::vector<std::array<int, 2>>& image_shifts) {
    if (destination.ncomp() != source.ncomp())
      throw std::invalid_argument("prepared periodic copy component mismatch");
    if (destination.box_array().size() == 0 || source.box_array().size() == 0)
      throw std::invalid_argument("prepared periodic copy requires non-empty layouts");
    if (domain.empty() || domain.nx() <= 0 || domain.ny() <= 0)
      throw std::invalid_argument("prepared periodic copy requires a non-empty domain");
    for (const Box2D& box : source.box_array().boxes())
      if (!(box.intersect(domain) == box))
        throw std::invalid_argument("prepared periodic copy source lies outside its domain");
    for (int source_index = 0; source_index < source.box_array().size(); ++source_index)
      for (int previous = 0; previous < source_index; ++previous)
        if (!source.box_array()[source_index]
                 .intersect(source.box_array()[previous])
                 .empty())
          throw std::invalid_argument(
              "prepared periodic copy requires a non-overlapping source decomposition");

    const Box2D destination_bounds = destination.box_array().bounding_box();
    const auto floor_div64 = [](std::int64_t value, std::int64_t divisor) {
      std::int64_t quotient = value / divisor;
      const std::int64_t remainder = value % divisor;
      if (remainder < 0)
        --quotient;
      return quotient;
    };
    const auto image_range = [&](int axis, bool periodic) {
      if (!periodic)
        return std::pair<std::int64_t, std::int64_t>{0, 0};
      const std::int64_t extent = axis == 0 ? domain.nx() : domain.ny();
      const std::int64_t minimum =
          -floor_div64(-(static_cast<std::int64_t>(destination_bounds.lo[axis]) -
                         domain.hi[axis]),
                       extent);
      const std::int64_t maximum =
          floor_div64(static_cast<std::int64_t>(destination_bounds.hi[axis]) -
                          domain.lo[axis],
                      extent);
      return std::pair<std::int64_t, std::int64_t>{minimum, maximum};
    };
    const auto [qx_lo, qx_hi] = image_range(0, periodicity.x);
    const auto [qy_lo, qy_hi] = image_range(1, periodicity.y);
    const std::int64_t image_count_x = qx_hi - qx_lo + 1;
    const std::int64_t image_count_y = qy_hi - qy_lo + 1;
    const std::int64_t source_count = source.box_array().size();
    if (image_count_x <= 0 || image_count_y <= 0 || source_count <= 0 ||
        image_count_x > std::numeric_limits<int>::max() / image_count_y ||
        image_count_x * image_count_y >
            std::numeric_limits<int>::max() / source_count)
      throw std::overflow_error(
          "prepared periodic copy image catalogue exceeds native range");
    const std::size_t maximum_images = static_cast<std::size_t>(
        image_count_x * image_count_y * source_count);

    std::vector<Box2D> image_boxes;
    std::vector<int> image_owners;
    image_boxes.reserve(maximum_images);
    image_owners.reserve(maximum_images);
    image_sources.reserve(maximum_images);
    image_shifts.reserve(maximum_images);
    const auto shifted_index = [](int index, std::int64_t shift) {
      const std::int64_t result = static_cast<std::int64_t>(index) + shift;
      if (result < std::numeric_limits<int>::min() ||
          result > std::numeric_limits<int>::max())
        throw std::overflow_error(
            "prepared periodic copy shifted box exceeds integer range");
      return static_cast<int>(result);
    };
    for (std::int64_t qy = qy_lo; qy <= qy_hi; ++qy)
      for (std::int64_t qx = qx_lo; qx <= qx_hi; ++qx) {
        const std::int64_t sx64 = qx * static_cast<std::int64_t>(domain.nx());
        const std::int64_t sy64 = qy * static_cast<std::int64_t>(domain.ny());
        if (sx64 < std::numeric_limits<int>::min() ||
            sx64 > std::numeric_limits<int>::max() ||
            sy64 < std::numeric_limits<int>::min() ||
            sy64 > std::numeric_limits<int>::max())
          throw std::overflow_error(
              "prepared periodic copy image shift exceeds integer range");
        const int sx = static_cast<int>(sx64);
        const int sy = static_cast<int>(sy64);
        for (int source_index = 0; source_index < source.box_array().size(); ++source_index) {
          const Box2D original = source.box_array()[source_index];
          const Box2D shifted{{shifted_index(original.lo[0], sx64),
                               shifted_index(original.lo[1], sy64)},
                              {shifted_index(original.hi[0], sx64),
                               shifted_index(original.hi[1], sy64)}};
          bool needed = false;
          for (const Box2D& destination_box : destination.box_array().boxes())
            if (!shifted.intersect(destination_box).empty()) {
              needed = true;
              break;
            }
          if (!needed)
            continue;
          image_boxes.push_back(shifted);
          image_owners.push_back(source.dmap()[source_index]);
          image_sources.push_back(source_index);
          image_shifts.push_back({sx, sy});
        }
      }
    if (image_boxes.empty())
      throw std::invalid_argument(
          "prepared periodic copy source images do not intersect the destination layout");

    // Source boxes are disjoint inside one fundamental domain; translating them by integral
    // domain extents therefore yields a globally disjoint image catalogue.  Prove that every
    // destination cell owned by the parent domain (unbounded along periodic axes, clipped along
    // physical axes) has exactly one source before publishing the plan.  Missing coverage would
    // otherwise leave uninitialised carrier values, while duplicate coverage would make
    // ParallelCopy source selection depend on BoxHash enumeration order.
    for (const Box2D& destination_box : destination.box_array().boxes()) {
      Box2D required = destination_box;
      if (!periodicity.x) {
        required.lo[0] = std::max(required.lo[0], domain.lo[0]);
        required.hi[0] = std::min(required.hi[0], domain.hi[0]);
      }
      if (!periodicity.y) {
        required.lo[1] = std::max(required.lo[1], domain.lo[1]);
        required.hi[1] = std::min(required.hi[1], domain.hi[1]);
      }
      if (required.empty())
        continue;
      std::int64_t covered_cells = 0;
      for (const Box2D& image : image_boxes) {
        const std::int64_t cells = image.intersect(required).num_cells();
        if (cells < 0 || cells > std::numeric_limits<std::int64_t>::max() - covered_cells)
          throw std::overflow_error(
              "prepared periodic copy coverage exceeds int64 capacity");
        covered_cells += cells;
      }
      if (covered_cells != required.num_cells())
        throw std::invalid_argument(
            "prepared periodic copy source images do not cover the required carrier region");
    }
    return MultiFab(BoxArray(std::move(image_boxes)),
                    DistributionMapping(std::move(image_owners)), source.ncomp(), 0);
  }

  void validate_replay_(const MultiFab& destination, const MultiFab& source,
                        std::uint64_t topology_generation,
                        const CommunicatorView& communicator) const {
    if (destination.box_array().boxes() != destination_boxes_ ||
        destination.dmap().ranks() != destination_ranks_ ||
        destination.n_grow() != destination_ngrow_ ||
        source.box_array().boxes() != source_boxes_ ||
        source.dmap().ranks() != source_ranks_ || source.n_grow() != source_ngrow_ ||
        destination.ncomp() != ncomp_ || source.ncomp() != ncomp_)
      throw std::invalid_argument(
          "prepared periodic copy replay does not match its exact layouts");
    if (topology_generation != topology_generation_)
      throw std::invalid_argument(
          "prepared periodic copy replay crossed a topology generation");
    if (communicator.size() != communicator_size_ ||
        communicator.rank() != communicator_rank_ ||
        detail::parallel_copy_communicator_identity(communicator) !=
            communicator_identity_)
      throw std::invalid_argument(
          "prepared periodic copy replay changed execution communicator");
  }

  std::vector<Box2D> destination_boxes_;
  std::vector<int> destination_ranks_;
  std::vector<Box2D> source_boxes_;
  std::vector<int> source_ranks_;
  int destination_ngrow_ = 0;
  int source_ngrow_ = 0;
  int ncomp_ = 0;
  Box2D domain_;
  Periodicity periodicity_{};
  std::uint64_t topology_generation_ = 0;
  int communicator_size_ = 1;
  int communicator_rank_ = 0;
  std::int64_t communicator_identity_ = 0;
  int message_tag_ = 0;
  std::vector<int> image_sources_;
  std::vector<std::array<int, 2>> image_shifts_;
  MultiFab images_;
};

/// Convenience one-shot route. Performance-sensitive AMR paths own a
/// PreparedPeriodicCopyPlan and call apply(); this wrapper intentionally preserves the low-level
/// value API for setup and tests.
inline void parallel_copy_periodic(MultiFab& destination, const MultiFab& source,
                                   const Box2D& domain, Periodicity periodicity) {
  if (destination.box_array().size() == 0 || source.box_array().size() == 0)
    return;
  ExecutionLane lane = ExecutionLane::world();
  (void)PreparedPeriodicCopyPlan::prepare(destination, source, domain, periodicity,
                                          /*topology_generation=*/0,
                                          lane.communicator());
}

namespace detail {

inline std::string transfer_scratch_summary(const MultiFab& fine, const MultiFab& scratch, int nc,
                                            int r) {
  return "r=" + std::to_string(r) + ", fine.boxes=" + std::to_string(fine.box_array().size()) +
         ", scratch.boxes=" + std::to_string(scratch.box_array().size()) +
         ", scratch.ncomp=" + std::to_string(scratch.ncomp()) +
         ", required.ncomp=" + std::to_string(nc) +
         ", scratch.ngrow=" + std::to_string(scratch.n_grow());
}

inline void validate_transfer_scratch(const char* where, const MultiFab& fine,
                                      const MultiFab& scratch, int nc, int r) {
  if (r < 1)
    throw_validation_error(where, "refinement/coarsening ratio r >= 1", "r=" + std::to_string(r));
  const BoxArray expected = coarsen(fine.box_array(), r);
  if (scratch.box_array().boxes() != expected.boxes() ||
      scratch.dmap().ranks() != fine.dmap().ranks() || scratch.ncomp() != nc ||
      scratch.n_grow() != 0) {
    throw_validation_error(
        where,
        "scratch MultiFab layout == coarsen(fine.box_array(), r), scratch.dmap == fine.dmap, "
        "scratch.ncomp equals the provider width, scratch.ngrow == 0",
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
/// grid); provider widths must match. Preserves the fine integral over the covered area.
// PROVIDED-BUFFER variant: @p cfine is the "fine coarsen" grid (layout coarsen(fine.box_array(), r),
// dmap = fine.dmap(), exactly ncomp components, 0 ghost) ALLOCATED by the caller and reused on each
// call (hot path of the MG V-cycle: avoids one MultiFab allocation per restriction). Computation
// STRICTLY identical to the allocating variant below.
namespace detail {

template <class ParallelCopy>
inline void average_down_on(const MultiFab& fine, MultiFab& coarse, int r, MultiFab& cfine,
                            ParallelCopy&& parallel_copy_fn) {
  if (fine.ncomp() != coarse.ncomp())
    throw_validation_error("pops/mesh/layout/refinement.hpp: average_down",
                           "fine and coarse provider widths are identical",
                           "fine.ncomp=" + std::to_string(fine.ncomp()) +
                               ", coarse.ncomp=" + std::to_string(coarse.ncomp()));
  const int nc = fine.ncomp();
  detail::validate_transfer_scratch("pops/mesh/layout/refinement.hpp: average_down(scratch)", fine,
                                    cfine, nc, r);
  const Real inv = Real(1) / (r * r);
  for (int li = 0; li < fine.local_size(); ++li) {
    const ConstArray4 F = fine.fab(li).const_array();
    Array4 C = cfine.fab(li).array();
    const Box2D cb = cfine.fab(li).box();
    for (int c = 0; c < nc; ++c)
      for_each_cell(cb, detail::AverageDownKernel{F, C, inv, r, c});
  }
  std::forward<ParallelCopy>(parallel_copy_fn)(coarse, cfine);
}

}  // namespace detail

inline void average_down(const MultiFab& fine, MultiFab& coarse, int r, MultiFab& cfine) {
  detail::average_down_on(fine, coarse, r, cfine,
                          [](MultiFab& dst, const MultiFab& src) { parallel_copy(dst, src); });
}
inline void average_down(const MultiFab& fine, MultiFab& coarse, int r, MultiFab& cfine,
                         const ExecutionLane& lane) {
  detail::average_down_on(fine, coarse, r, cfine, [&lane](MultiFab& dst, const MultiFab& src) {
    parallel_copy(dst, src, lane);
  });
}
inline void average_down(const MultiFab& fine, MultiFab& coarse, int r) {
  MultiFab cfine(coarsen(fine.box_array(), r), fine.dmap(), fine.ncomp(), 0);
  average_down(fine, coarse, r, cfine);
}
inline void average_down(const MultiFab& fine, MultiFab& coarse, int r, const ExecutionLane& lane) {
  MultiFab cfine(coarsen(fine.box_array(), r), fine.dmap(), fine.ncomp(), 0);
  average_down(fine, coarse, r, cfine, lane);
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
/// identical provider widths. First brings coarse values onto a local fine-coarsen grid.
// PROVIDED-BUFFER variant: @p cfine is the "fine coarsen" grid (same layout contract as
// average_down above) allocated by the caller and reused (hot path of the MG V-cycle: avoids one
// allocation per prolongation). Computation STRICTLY identical to the allocating variant.
namespace detail {

template <class ParallelCopy>
inline void interpolate_on(const MultiFab& coarse, MultiFab& fine, int r, MultiFab& cfine,
                           ParallelCopy&& parallel_copy_fn) {
  if (fine.ncomp() != coarse.ncomp())
    throw_validation_error("pops/mesh/layout/refinement.hpp: interpolate",
                           "fine and coarse provider widths are identical",
                           "fine.ncomp=" + std::to_string(fine.ncomp()) +
                               ", coarse.ncomp=" + std::to_string(coarse.ncomp()));
  const int nc = fine.ncomp();
  detail::validate_transfer_scratch("pops/mesh/layout/refinement.hpp: interpolate(scratch)", fine,
                                    cfine, nc, r);
  std::forward<ParallelCopy>(parallel_copy_fn)(cfine, coarse);
  for (int li = 0; li < fine.local_size(); ++li) {
    Array4 F = fine.fab(li).array();
    const ConstArray4 C = cfine.fab(li).const_array();
    const Box2D fb = fine.fab(li).box();
    for (int c = 0; c < nc; ++c)
      for_each_cell(fb, detail::InterpolateKernel{F, C, r, c});
  }
}

}  // namespace detail

inline void interpolate(const MultiFab& coarse, MultiFab& fine, int r, MultiFab& cfine) {
  detail::interpolate_on(coarse, fine, r, cfine,
                         [](MultiFab& dst, const MultiFab& src) { parallel_copy(dst, src); });
}
inline void interpolate(const MultiFab& coarse, MultiFab& fine, int r, MultiFab& cfine,
                        const ExecutionLane& lane) {
  detail::interpolate_on(coarse, fine, r, cfine, [&lane](MultiFab& dst, const MultiFab& src) {
    parallel_copy(dst, src, lane);
  });
}
inline void interpolate(const MultiFab& coarse, MultiFab& fine, int r, MultiFab& cfine,
                        const CommunicatorView& communicator) {
  detail::interpolate_on(coarse, fine, r, cfine,
                         [&communicator](MultiFab& dst, const MultiFab& src) {
                           parallel_copy(dst, src, communicator);
                         });
}
inline void interpolate(const MultiFab& coarse, MultiFab& fine, int r) {
  MultiFab cfine(coarsen(fine.box_array(), r), fine.dmap(), fine.ncomp(), 0);
  interpolate(coarse, fine, r, cfine);
}
inline void interpolate(const MultiFab& coarse, MultiFab& fine, int r,
                        const CommunicatorView& communicator) {
  MultiFab cfine(coarsen(fine.box_array(), r), fine.dmap(), fine.ncomp(), 0);
  interpolate(coarse, fine, r, cfine, communicator);
}
inline void interpolate(const MultiFab& coarse, MultiFab& fine, int r, const ExecutionLane& lane) {
  MultiFab cfine(coarsen(fine.box_array(), r), fine.dmap(), fine.ncomp(), 0);
  interpolate(coarse, fine, r, cfine, lane);
}

}  // namespace pops
