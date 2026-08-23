/// @file
/// @brief Compile-time-ranked cell, face, product, and reduction iteration through Kokkos.
///        sync_host / sync_device: the residency COHERENCE seam (counterpart for host accesses).
///
/// KOKKOS IS THE ONLY on-node backend: this seam compiles ONLY under POPS_HAS_KOKKOS (cf. CMake, which
/// makes Kokkos mandatory; without it, #error below). The functor is taken BY VALUE and receives
/// a typed Index<Dim>; it captures field views by value (POD), never the Fab nor anything virtual:
/// exactly the constraint of a device kernel. The on-node target (sequential = Kokkos Serial, CPU
/// multi-thread = Kokkos OpenMP, GPU = Kokkos Cuda/HIP) is chosen AT KOKKOS INSTALLATION, not
/// by a PoPS flag: one rank-generic for_each_cell call
/// covers all three. The CPU -> GPU switch therefore does NOT change the call sites.
/// FP CHOICE: the SUM reduction (Kokkos::Sum) reassociates the addition per tile -> DETERMINISTIC per
/// tile (idempotent: same data, same backend -> same bits) but NOT bit-identical to a lexicographic
/// sum; this holds for Serial, OpenMP and Cuda (a single path, Kokkos). The MAX reduction
/// is exact everywhere (max associative/commutative in IEEE754). sync_host() = a targeted device_fence()
/// before a host access; sync_device() = no-op under unified memory (scaffolding for a future
/// non-unified path).

#pragma once

#include <pops/core/foundation/kokkos_env.hpp>  // detail::ensure_kokkos_initialized + device_fence (life cycle)
#include <pops/core/foundation/types.hpp>
#include <pops/diagnostics/fallback_diagnostics.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/index/entity_index.hpp>

#include <cstdint>  // std::int64_t: cell counts (LLP64 portability, no-op on LP64)
#include <cstdlib>  // getenv / strtol: overridable serial fallback threshold (#165)
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>  // std::is_same_v: compile-time guard host vs device exec space (#165)
#include <utility>
#include <vector>

#ifndef POPS_HAS_KOKKOS
// PoPS is KOKKOS-ONLY: there is no longer a standalone OpenMP backend nor a manual host loop
// as a production path. Configure with -DPOPS_USE_KOKKOS=ON (+ -DKokkos_ROOT=...); serial
// goes through a Kokkos install with Kokkos_ENABLE_SERIAL=ON.
#error \
    "PoPS is Kokkos-only: for_each_cell requires POPS_HAS_KOKKOS. Configure with -DPOPS_USE_KOKKOS=ON and a Kokkos Serial/OpenMP/Cuda install."
#endif

#include <Kokkos_Core.hpp>

namespace pops {

// detail::ensure_kokkos_initialized() and device_fence(): defined in pops/core/kokkos_env.hpp
// (Kokkos life cycle shared with the unified allocator, which must also initialize Kokkos BEFORE
// its first kokkos_malloc, otherwise the Kokkos build crashes when constructing a Fab).

// SERIAL FALLBACK THRESHOLD for for_each_cell (#165). Under a HOST Kokkos execution space
// (Serial/OpenMP), launching a Kokkos::parallel_for on a tiny box pays a
// fork/join (and the policy construction) that OVERWHELMS the useful work. The multigrid
// V-cycle descends down to ~2x2/4x4 grids; on those levels the GS smoother, the residual,
// the restriction/prolongation and the copies chain dozens of parallel_for over a few
// cells, and this launch overhead DOMINATES the solve time. Below this threshold we run
// a SEQUENTIAL host loop (internal to the Kokkos path, this is NOT a separate backend),
// above it we keep Kokkos parallel_for for the fine grids.
//
// BIT-IDENTITY. for_each_cell has NO inter-iteration dependency: each f(index)
// writes only its destination cell and reads cells IT DOES NOT WRITE
// in the same call (the GS smoother is RED-BLACK colored -- one color only reads
// the other; residual/restriction/prolongation/copies/saxpy write a destination
// distinct from the source). The result is therefore INDEPENDENT OF the traversal ORDER:
// the sequential loop yields exactly the same bits as the matching flattened Kokkos policy.
// The threshold touches ONLY for_each_cell (not the reductions for_each_cell_reduce_*:
// the Kokkos parallel sum reassociates the addition, so switching them to serial would
// NOT be bit-identical -- we leave them intact; the max is exact but the smoother itself
// does go through for_each_cell, where the overhead of the small grids concentrates).
//
// Overridable at run time via POPS_FOREACH_SERIAL_THRESHOLD (read once) to
// resweep the threshold without recompiling; default 4096 (same fork/join vs computation
// trade-off as the old if() clause of the removed OpenMP path).
namespace detail {

namespace testing {
struct ReductionResultPoolAccess;
}
inline std::int64_t foreach_serial_threshold() {
  static const std::int64_t thr = []() -> std::int64_t {
    if (const char* e = std::getenv("POPS_FOREACH_SERIAL_THRESHOLD")) {
      char* end = nullptr;
      const std::int64_t v = std::strtol(e, &end, 10);
      if (end != e && v >= 0)
        return v;
    }
    return 4096;
  }();
  return thr;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Data residency: sync_host() / sync_device(). The COHERENCE seam, the
// counterpart of for_each_cell for host accesses.
//
// Today the Fab storage lives in UNIFIED memory (Kokkos::SharedSpace, cf.
// allocator.hpp): the same buffer serves the host code (operator(), loops) AND
// the device kernels. Coherence therefore does NOT require a copy, only
// ORDERING: a host access must not read/write a buffer while
// an async kernel still touches it. Until now this ordering was laid down by
// hand by scattered device_fence() calls, without ever saying WHICH residency one
// wants to make valid.
//
// sync_host()/sync_device() ENCODE that intent:
//   - sync_host(): "I am going to read/write this data FROM THE HOST;
//                      make it valid host-side". Under SharedSpace = a
//                      targeted device_fence() (wait for in-flight kernels), so
//                      host accesses are then race-free (data race).
//   - sync_device(): "I am going to read/write this data FROM THE DEVICE
//                      (a kernel); make it valid device-side". Under
//                      SharedSpace the preceding host writes are visible
//                      from the device without a barrier (no async host pipeline to
//                      drain), so it is a REAL no-op today; the
//                      function exists to MARK the intent at the call site.
//
// SEMANTICS UNDER SHAREDSPACE (current state): these calls are at most a fence,
// never a copy. The behavior therefore stays BIT-IDENTICAL to the old code
// (sync_host == the old device_fence() laid before a host access; sync_device
// == nothing). This is deliberately SCAFFOLDING: under unified memory there is
// nothing else to do.
//
// FUTURE NON-UNIFIED PATH (separate host/device buffers + deep_copy): this is WHERE
// the migration would plug in. sync_host() would do a Kokkos::deep_copy
// device->host (and a fence) if the device is the last residency written;
// sync_device() a deep_copy host->device in the other direction. Tracking "who
// owns the up-to-date data" (per-residency dirty flag) would live on the MultiFab,
// not here: this seam stays stateless, the MultiFab overloads carry the state.
// Since all the host-access sites already go through sync_host(), switching to
// that path will NOT touch the operators, exactly as for_each_cell
// isolates the CPU -> GPU switch from the call sites.

/// Makes the HOST residency valid before a host access (read/write from the host). Under unified memory
/// = a targeted device_fence() (waits for in-flight kernels).
inline void sync_host() {
  device_fence();
}

/// Marks a DEVICE residency (upcoming kernel). Under unified memory: NO-OP (host writes
/// are already visible from the device); exists to document the intent and to accommodate a future
/// deep_copy host->device on a non-unified path.
inline void sync_device() {}

namespace detail {

template <int Dim>
inline void require_iterable_box(const Box<Dim>& box) {
  if (box.empty())
    return;
  for (int axis = 0; axis < Dim; ++axis) {
    if (box.length(axis) > std::numeric_limits<int>::max() ||
        box.hi[axis] == std::numeric_limits<int>::max())
      throw std::overflow_error(
          "PoPS Kokkos iteration requires int-addressable extents and an inclusive high index "
          "below "
          "INT_MAX");
  }
}

template <int Dim>
inline bool foreach_small_box(const Box<Dim>& box, std::int64_t threshold) noexcept {
  if (box.empty() || threshold <= 0)
    return false;
  std::int64_t remaining = threshold - 1;
  for (int axis = 0; axis < Dim; ++axis) {
    const std::int64_t extent = box.length(axis);
    if (extent <= 0 || extent > remaining)
      return false;
    remaining /= extent;
  }
  return true;
}

template <int Dim>
POPS_HD CellIndex<Dim> cell_index_from_ordinal(const Index<Dim>& lower, const Extent<Dim>& extent,
                                               std::int64_t ordinal) {
  CellIndex<Dim> index{};
  for (int axis = 0; axis < Dim; ++axis) {
    index[axis] = lower[axis] + static_cast<int>(ordinal % extent[axis]);
    ordinal /= extent[axis];
  }
  return index;
}

/// Named device functor for flattened ranked iteration.
///
/// Keeping this adapter as a normal trivially-copyable Kokkos functor avoids an extended device
/// lambda at every call site.  This is semantically identical to the former lambda: `F` is still
/// captured by value and receives the same lexicographically decoded compile-time-ranked index.
template <int Dim, class F>
struct IndexSpaceKernelAdapter {
  Index<Dim> lower{};
  Extent<Dim> extent{};
  F functor;

  POPS_HD void operator()(std::int64_t ordinal) const {
    functor(cell_index_from_ordinal(lower, extent, ordinal));
  }
};

template <class ExecutionSpace, int Dim, class F>
void launch_index_space(const ExecutionSpace& execution, const Box<Dim>& box, const char* label,
                        F f) {
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value,
                "PoPS iteration requires a Kokkos execution-space instance");
  const Index<Dim> lower = box.lo;
  const Extent<Dim> extent = box.extent();
  Kokkos::parallel_for(label,
                       Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::int64_t>>(
                           execution, 0, box.numPts()),
                       IndexSpaceKernelAdapter<Dim, F>{lower, extent, std::move(f)});
}

template <int Dim, int Axis, class F>
struct FaceKernelAdapter {
  F functor;

  POPS_HD void operator()(const CellIndex<Dim>& index) const {
    functor(FaceIndex<Dim, Axis>{index});
  }
};

/// Reusable rank-0 device results for reductions which must return a host scalar.  A slot remains
/// exclusive until its execution instance has completed the explicit copy to host.  The singleton
/// deliberately outlives ordinary static destruction; its Kokkos finalizer hook clears device
/// allocations while their memory space is still alive.
template <class MemorySpace>
class ReductionResultPool {
 public:
  using view_type = Kokkos::View<Real, MemorySpace>;

  class Lease {
   public:
    Lease() = default;
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;

    Lease(Lease&& other) noexcept
        : pool_(std::exchange(other.pool_, nullptr)),
          slot_(other.slot_),
          result_(std::move(other.result_)) {}

    Lease& operator=(Lease&& other) noexcept {
      if (this != &other) {
        abandon();
        pool_ = std::exchange(other.pool_, nullptr);
        slot_ = other.slot_;
        result_ = std::move(other.result_);
      }
      return *this;
    }

    ~Lease() { abandon(); }

    const view_type& view() const noexcept { return result_; }

    /// Releases only after the caller has fenced the execution instance that used the result.
    void release() noexcept {
      if (pool_ != nullptr) {
        pool_->release(slot_);
        pool_ = nullptr;
      }
    }

    /// A throwing Kokkos operation can leave device work outstanding.  Keep this slot unavailable
    /// until the Kokkos finalizer releases every persistent allocation.
    void abandon() noexcept {
      if (pool_ != nullptr) {
        pool_->abandon(slot_);
        pool_ = nullptr;
      }
    }

   private:
    friend class ReductionResultPool;

    Lease(ReductionResultPool* pool, std::size_t slot, view_type result)
        : pool_(pool), slot_(slot), result_(std::move(result)) {}

    ReductionResultPool* pool_ = nullptr;
    std::size_t slot_ = 0;
    view_type result_{};
  };

  static ReductionResultPool& instance() {
    static ReductionResultPool* pool = new ReductionResultPool();
    return *pool;
  }

  Lease acquire(const char* label) {
    register_finalize_hook();
    std::lock_guard<std::mutex> guard(mutex_);
    for (std::size_t slot = 0; slot < slots_.size(); ++slot) {
      if (!slots_[slot].in_use) {
        slots_[slot].in_use = true;
        return Lease(this, slot, slots_[slot].result);
      }
    }
    slots_.push_back(Slot{view_type(std::string(label)), true, false});
    return Lease(this, slots_.size() - 1, slots_.back().result);
  }

 private:
  friend struct testing::ReductionResultPoolAccess;

  struct Slot {
    view_type result{};
    bool in_use = false;
    bool abandoned = false;
  };

  void register_finalize_hook() {
    std::call_once(finalize_hook_once_,
                   [this] { Kokkos::push_finalize_hook([this] { release_all(); }); });
  }

  void release(std::size_t slot) noexcept {
    std::lock_guard<std::mutex> guard(mutex_);
    if (slot < slots_.size() && !slots_[slot].abandoned)
      slots_[slot].in_use = false;
  }

  void abandon(std::size_t slot) noexcept {
    std::lock_guard<std::mutex> guard(mutex_);
    if (slot < slots_.size()) {
      slots_[slot].in_use = true;
      slots_[slot].abandoned = true;
    }
  }

  void release_all() noexcept {
    release_all_after_global_fence([] { Kokkos::fence(); });
  }

  template <class GlobalFence>
  void release_all_after_global_fence(GlobalFence&& global_fence) noexcept {
    // Finalize hooks run before Kokkos tears down memory spaces.  The global barrier must remain
    // outside the pool lock: Kokkos callbacks may themselves need application threads that hold
    // or later acquire the pool mutex.  If Kokkos refuses the barrier during teardown, retain the
    // views rather than destroying allocations whose work may still be outstanding.
    try {
      std::forward<GlobalFence>(global_fence)();
    } catch (...) {
      return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    slots_.clear();
  }

  std::mutex mutex_;
  std::once_flag finalize_hook_once_;
  std::vector<Slot> slots_;
};

}  // namespace detail

/// Return the face-index box normal to compile-time @p Axis.  A cell box contains one more face
/// than cells along its normal axis and retains the cell extents along every tangent axis.
template <int Axis, int Dim>
Box<Dim> face_box(const Box<Dim>& cells) {
  static_assert(Axis >= 0 && Axis < Dim,
                "pops::face_box axis must lie inside the compile-time rank");
  if (cells.empty())
    return cells;
  detail::require_iterable_box(cells);
  Box<Dim> faces = cells;
  ++faces.hi[Axis];
  return faces;
}

/// Submit a cell kernel to an explicit Kokkos execution-space instance.  Unlike the default host
/// convenience overload, this path never substitutes a synchronous small-box loop, so task-graph
/// and accelerator-stream ordering remain owned by the supplied instance.
template <class ExecutionSpace, int Dim, class F>
void for_each_cell(const ExecutionSpace& execution, const Box<Dim>& b, F f) {
  if (b.empty())
    return;
  detail::require_iterable_box(b);
  detail::ensure_kokkos_initialized();
  detail::launch_index_space(execution, b, "pops_for_each_cell", f);
}

/// Applies @p f to every index of a compile-time-ranked box.  The functor is passed by value and
/// receives CellIndex<Dim>; one flattened policy decodes the box's compile-time rank without a
/// dimension-specific launch branch.
template <int Dim, class F>
void for_each_cell(const Box<Dim>& b, F f) {
  if (b.empty())
    return;
  detail::require_iterable_box(b);
  if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>) {
    if (detail::foreach_small_box(b, detail::foreach_serial_threshold())) {
      record_fallback(FallbackCounter::kForeachSerialSmallBox);
      const Extent<Dim> extent = b.extent();
      const std::int64_t point_count = b.numPts();
      for (std::int64_t ordinal = 0; ordinal < point_count; ++ordinal)
        f(detail::cell_index_from_ordinal(b.lo, extent, ordinal));
      return;
    }
  }
  detail::ensure_kokkos_initialized();
  const Kokkos::DefaultExecutionSpace execution{};
  for_each_cell(execution, b, f);
}

/// Submit the product of a compile-time-ranked integer box.  This is the non-cell semantic facade
/// used by topology, pack/unpack, and task-graph work while sharing the same static Kokkos policies.
template <class ExecutionSpace, int Dim, class F>
void for_each_product(const ExecutionSpace& execution, const Box<Dim>& product, F f) {
  for_each_cell(execution, product, f);
}

template <int Dim, class F>
void for_each_product(const Box<Dim>& product, F f) {
  for_each_cell(product, f);
}

/// Submit faces normal to compile-time @p Axis.  Axis is a type property of every FaceIndex passed
/// to the functor, so flux and metric kernels do not branch on direction in their inner loop.
template <int Axis, class ExecutionSpace, int Dim, class F>
void for_each_face(const ExecutionSpace& execution, const Box<Dim>& cells, F f) {
  const Box<Dim> faces = face_box<Axis>(cells);
  for_each_cell(execution, faces, detail::FaceKernelAdapter<Dim, Axis, F>{f});
}

template <int Axis, int Dim, class F>
void for_each_face(const Box<Dim>& cells, F f) {
  const Box<Dim> faces = face_box<Axis>(cells);
  for_each_cell(faces, detail::FaceKernelAdapter<Dim, Axis, F>{f});
}

/// SUM reduction on an explicit execution-space instance.  The reduction result resides first in
/// that instance's memory space, then is copied explicitly to the host scalar.  The final fence
/// completes work submitted to this execution instance without fencing unrelated streams.
template <class ExecutionSpace, int Dim, class F>
Real for_each_cell_reduce_sum(const ExecutionSpace& execution, const Box<Dim>& b, F f) {
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value,
                "PoPS reduction requires a Kokkos execution-space instance");
  if (b.empty())
    return Real(0);
  detail::require_iterable_box(b);
  detail::ensure_kokkos_initialized();
  using result_memory_space = typename ExecutionSpace::memory_space;
  auto result_lease = detail::ReductionResultPool<result_memory_space>::instance().acquire(
      "pops_reduce_sum_result");
  const auto& device_result = result_lease.view();
  const Index<Dim> lower = b.lo;
  const Extent<Dim> extent = b.extent();
  Kokkos::parallel_reduce(
      "pops_reduce_sum_index",
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::int64_t>>(execution, 0,
                                                                           b.numPts()),
      KOKKOS_LAMBDA(const std::int64_t ordinal, Real& accumulator) {
        accumulator += f(detail::cell_index_from_ordinal(lower, extent, ordinal));
      },
      Kokkos::Sum<Real, result_memory_space>{device_result});
  Real result = 0;
  Kokkos::deep_copy(execution, result, device_result);
  execution.fence("pops_reduce_sum_result_to_host");
  result_lease.release();
  return result;
}

/// SUM reduction over a compile-time-ranked box on the default execution-space instance.
template <int Dim, class F>
Real for_each_cell_reduce_sum(const Box<Dim>& b, F f) {
  if (b.empty())
    return Real(0);
  detail::ensure_kokkos_initialized();
  const Kokkos::DefaultExecutionSpace execution{};
  return for_each_cell_reduce_sum(execution, b, f);
}

/// MAX reduction on an explicit execution-space instance.  As for SUM, keep the reducer output
/// in the execution memory space and make the device-to-host completion explicit.
template <class ExecutionSpace, int Dim, class F>
Real for_each_cell_reduce_max(const ExecutionSpace& execution, const Box<Dim>& b, F f) {
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value,
                "PoPS reduction requires a Kokkos execution-space instance");
  if (b.empty())
    return Real(0);
  detail::require_iterable_box(b);
  detail::ensure_kokkos_initialized();
  using result_memory_space = typename ExecutionSpace::memory_space;
  auto result_lease = detail::ReductionResultPool<result_memory_space>::instance().acquire(
      "pops_reduce_max_result");
  const auto& device_result = result_lease.view();
  const Index<Dim> lower = b.lo;
  const Extent<Dim> extent = b.extent();
  Kokkos::parallel_reduce(
      "pops_reduce_max_index",
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::int64_t>>(execution, 0,
                                                                           b.numPts()),
      KOKKOS_LAMBDA(const std::int64_t ordinal, Real& accumulator) {
        const Real value = f(detail::cell_index_from_ordinal(lower, extent, ordinal));
        if (value > accumulator)
          accumulator = value;
      },
      Kokkos::Max<Real, result_memory_space>{device_result});
  Real result = std::numeric_limits<Real>::lowest();
  Kokkos::deep_copy(execution, result, device_result);
  execution.fence("pops_reduce_max_result_to_host");
  result_lease.release();
  return result;
}

/// MAX reduction over a compile-time-ranked box on the default execution-space instance.
template <int Dim, class F>
Real for_each_cell_reduce_max(const Box<Dim>& b, F f) {
  if (b.empty())
    return Real(0);
  detail::ensure_kokkos_initialized();
  const Kokkos::DefaultExecutionSpace execution{};
  return for_each_cell_reduce_max(execution, b, f);
}

template <class ExecutionSpace, int Dim, class F>
Real for_each_product_reduce_sum(const ExecutionSpace& execution, const Box<Dim>& product, F f) {
  return for_each_cell_reduce_sum(execution, product, f);
}

template <int Dim, class F>
Real for_each_product_reduce_sum(const Box<Dim>& product, F f) {
  return for_each_cell_reduce_sum(product, f);
}

}  // namespace pops
