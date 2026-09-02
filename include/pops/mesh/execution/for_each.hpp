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

#include <algorithm>
#include <cstddef>
#include <cstdint>  // std::int64_t: cell counts (LLP64 portability, no-op on LP64)
#include <cstdlib>  // getenv / strtol: overridable serial fallback threshold (#165)
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>  // std::is_same_v: compile-time guard host vs device exec space (#165)
#include <utility>

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

// Kokkos' public launch overloads take ``const std::string&``.  Passing a string literal creates a
// temporary label on every dispatch; some supported standard-library ABIs allocate even for these
// short labels.  These process-lifetime labels are constructed before the runtime enters a prepared
// Program step and are reused by every rank/dimension/backend invocation.
inline const std::string kForEachKernelLabel{"pops_each"};
inline const std::string kReduceSumKernelLabel{"pops_rsum"};
inline const std::string kReduceSumPartialsLabel{"pops_rsum_partials"};
inline const std::string kReduceMaxKernelLabel{"pops_rmax"};
inline const std::string kReductionFenceLabel{"pops_reduce_fence"};

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

/// Cold-owned workspace for allocation-free prepared reductions.  The fixed, contiguous partition
/// makes the result independent of Kokkos worker scheduling: each kernel iteration owns one partial,
/// and the host combines partials in ascending ordinal order.  The partials live in the execution
/// space's memory and the fold buffer lives in HostSpace, so the exact same prepared route works for
/// Serial/OpenMP as well as Cuda/HIP.
///
/// A prepared workspace has one exact capacity witness.  It must be prepared before a hot path and
/// rejects a non-empty box whose point count exceeds that witness instead of growing storage while a
/// Program transaction is active.  SUM and MAX both use parallel_for (never parallel_reduce) on the
/// prepared route, followed by a resident fence and a preallocated device-to-host copy.
template <class ExecutionSpace>
class PreparedCellSumReduction {
 public:
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value,
                "PoPS reduction requires a Kokkos execution-space instance");

  using execution_space = ExecutionSpace;
  using memory_space = typename ExecutionSpace::memory_space;
  using partial_view_type = Kokkos::View<Real*, memory_space>;
  using host_fold_view_type = Kokkos::View<Real*, Kokkos::HostSpace>;

  PreparedCellSumReduction() = default;
  PreparedCellSumReduction(const PreparedCellSumReduction&) = default;
  PreparedCellSumReduction& operator=(const PreparedCellSumReduction&) = default;

  /// Moving a prepared workspace only transfers already-owned Kokkos view handles and scalar
  /// witnesses.  It never creates storage; the explicit noexcept contract is required by the
  /// aggregate transaction images, which publish by swapping their prepared workspaces.
  PreparedCellSumReduction(PreparedCellSumReduction&& other) noexcept
      : partials_(std::move(other.partials_)),
        host_fold_(std::move(other.host_fold_)),
        maximum_points_(std::exchange(other.maximum_points_, 0)),
        partial_count_(std::exchange(other.partial_count_, 0)),
        prepared_(std::exchange(other.prepared_, false)) {}

  PreparedCellSumReduction& operator=(PreparedCellSumReduction&& other) noexcept {
    if (this != &other) {
      partials_ = std::move(other.partials_);
      host_fold_ = std::move(other.host_fold_);
      maximum_points_ = std::exchange(other.maximum_points_, 0);
      partial_count_ = std::exchange(other.partial_count_, 0);
      prepared_ = std::exchange(other.prepared_, false);
    }
    return *this;
  }

  friend void swap(PreparedCellSumReduction& left, PreparedCellSumReduction& right) noexcept {
    using std::swap;
    swap(left.partials_, right.partials_);
    swap(left.host_fold_, right.host_fold_);
    swap(left.maximum_points_, right.maximum_points_);
    swap(left.partial_count_, right.partial_count_);
    swap(left.prepared_, right.prepared_);
  }

  /// Allocate and bind the exact largest non-empty cell domain this workspace may reduce.
  void prepare(const ExecutionSpace& execution, std::int64_t maximum_points) {
    if (maximum_points <= 0)
      throw std::invalid_argument("prepared cell reduction requires a non-empty capacity");
    if (static_cast<std::uintmax_t>(maximum_points) >
        static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max()))
      throw std::overflow_error("prepared cell reduction capacity exceeds size_t");
    if (prepared_) {
      if (maximum_points_ != maximum_points)
        throw std::logic_error("prepared cell reduction capacity cannot change after preparation");
      return;
    }

    detail::ensure_kokkos_initialized();
    const int reported_concurrency = execution.concurrency();
    const std::int64_t requested_partials =
        reported_concurrency > 0 ? static_cast<std::int64_t>(reported_concurrency) : 1;
    partial_count_ = std::min(maximum_points, requested_partials);
    if (partial_count_ <= 0)
      throw std::logic_error("prepared cell reduction computed an invalid partial count");

    const auto count = static_cast<std::size_t>(partial_count_);
    partials_ = partial_view_type(detail::kReduceSumPartialsLabel, count);
    host_fold_ = host_fold_view_type(detail::kReduceSumPartialsLabel, count);
    maximum_points_ = maximum_points;

    // These cold fills warm the resident allocation/copy paths.  The prepared dispatch below uses
    // the same execution instance and explicit fence label, so no default Kokkos operation can
    // allocate an internal reducer or completion object in a candidate step.
    Kokkos::deep_copy(execution, partials_, Real{0});
    Kokkos::deep_copy(execution, host_fold_, Real{0});
    const auto warm_partials = partials_;
    Kokkos::parallel_for(
        detail::kReduceSumPartialsLabel,
        Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::int64_t>>(execution, 0,
                                                                             partial_count_),
        KOKKOS_LAMBDA(const std::int64_t partial_ordinal) {
          warm_partials(static_cast<std::size_t>(partial_ordinal)) = Real{0};
        });
    execution.fence(detail::kReductionFenceLabel);
    Kokkos::deep_copy(execution, host_fold_, partials_);
    execution.fence(detail::kReductionFenceLabel);
    prepared_ = true;
  }

  [[nodiscard]] bool accepts(std::int64_t point_count) const noexcept {
    return prepared_ && point_count >= 0 && point_count <= maximum_points_ && partial_count_ > 0 &&
           partials_.extent(0) == static_cast<std::size_t>(partial_count_) &&
           host_fold_.extent(0) == static_cast<std::size_t>(partial_count_);
  }

  [[nodiscard]] bool is_prepared() const noexcept { return prepared_; }

  [[nodiscard]] std::int64_t maximum_points() const noexcept { return maximum_points_; }

  [[nodiscard]] std::int64_t partial_count() const noexcept { return partial_count_; }

  /// Exact logical resident arena owned by this prepared reduction: one execution-space partial
  /// vector and one host fold vector.  Kokkos allocation headers are infrastructure overhead;
  /// the Program receipt charges the payload capacity it explicitly requests.
  [[nodiscard]] std::uint64_t resident_storage_bytes() const noexcept {
    if (!prepared_ || partial_count_ <= 0)
      return 0;
    return static_cast<std::uint64_t>(partial_count_) * sizeof(Real) * 2U;
  }

 private:
  template <class ES, int Dim, class F>
  friend Real for_each_cell_reduce_sum(const ES&, const PreparedCellSumReduction<ES>&,
                                       const Box<Dim>&, F);
  template <class ES, int Dim, class F>
  friend Real for_each_cell_reduce_max(const ES&, const PreparedCellSumReduction<ES>&,
                                       const Box<Dim>&, F);

  [[nodiscard]] const partial_view_type& partials() const noexcept { return partials_; }

  [[nodiscard]] const host_fold_view_type& host_fold() const noexcept { return host_fold_; }

  partial_view_type partials_{};
  host_fold_view_type host_fold_{};
  std::int64_t maximum_points_ = 0;
  std::int64_t partial_count_ = 0;
  bool prepared_ = false;
};

/// Generic spelling for code that does not care whether its prepared operation is SUM or MAX.
/// Keep PreparedCellSumReduction as the ABI/source-compatible public name used by installed
/// Program workspaces.
template <class ExecutionSpace>
using PreparedCellReduction = PreparedCellSumReduction<ExecutionSpace>;

template <class ExecutionSpace>
using PreparedCellMaxReduction = PreparedCellSumReduction<ExecutionSpace>;

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

template <class ExecutionSpace, int Dim, class F>
void launch_index_space(const ExecutionSpace& execution, const Box<Dim>& box,
                        const std::string& label, F f) {
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value,
                "PoPS iteration requires a Kokkos execution-space instance");
  const Index<Dim> lower = box.lo;
  const Extent<Dim> extent = box.extent();
  Kokkos::parallel_for(
      label,
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::int64_t>>(execution, 0,
                                                                           box.numPts()),
      KOKKOS_LAMBDA(const std::int64_t ordinal) {
        f(cell_index_from_ordinal(lower, extent, ordinal));
      });
}

template <int Dim, int Axis, class F>
struct FaceKernelAdapter {
  F functor;

  POPS_HD void operator()(const CellIndex<Dim>& index) const {
    functor(FaceIndex<Dim, Axis>{index});
  }
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

/// Submit a cell kernel to an explicit Kokkos execution-space instance.  Threaded and accelerator
/// spaces always retain their supplied dispatch/stream ordering.  The synchronous Serial default
/// is the sole exception: its one-worker ordinal loop is the same execution order and avoids
/// constructing Kokkos policy/profiling state inside an accepted Program step.
template <class ExecutionSpace, int Dim, class F>
void for_each_cell(const ExecutionSpace& execution, const Box<Dim>& b, F f) {
  if (b.empty())
    return;
  detail::require_iterable_box(b);
  detail::ensure_kokkos_initialized();
#if defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL)
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::DefaultExecutionSpace>) {
    const Extent<Dim> extent = b.extent();
    for (std::int64_t ordinal = 0; ordinal < b.numPts(); ++ordinal)
      f(detail::cell_index_from_ordinal(b.lo, extent, ordinal));
    return;
  }
#endif
  detail::launch_index_space(execution, b, detail::kForEachKernelLabel, f);
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
  for_each_cell(detail::default_execution_space(), b, f);
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

/// Cold convenience SUM reduction on an explicit execution-space instance.
///
/// HostSpace callers on a prepared hot route must use the overload taking
/// PreparedCellSumReduction.  This compatibility overload preserves the ordinary Kokkos reduction
/// route for cold and non-prepared call sites, including the unchanged device path.
template <class ExecutionSpace, int Dim, class F>
Real for_each_cell_reduce_sum(const ExecutionSpace& execution, const Box<Dim>& b, F f) {
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value,
                "PoPS reduction requires a Kokkos execution-space instance");
  if (b.empty())
    return Real(0);
  detail::require_iterable_box(b);
  detail::ensure_kokkos_initialized();
  Real result = 0;
  const Index<Dim> lower = b.lo;
  const Extent<Dim> extent = b.extent();
#if defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL)
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::DefaultExecutionSpace>) {
    for (std::int64_t ordinal = 0; ordinal < b.numPts(); ++ordinal)
      result += f(detail::cell_index_from_ordinal(lower, extent, ordinal));
    return result;
  }
#endif
  const auto policy = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::int64_t>>(
      execution, 0, b.numPts());
  const auto kernel = KOKKOS_LAMBDA(const std::int64_t ordinal, Real& accumulator) {
    accumulator += f(detail::cell_index_from_ordinal(lower, extent, ordinal));
  };
  if constexpr (std::is_same_v<typename ExecutionSpace::memory_space, Kokkos::HostSpace>) {
    // Kokkos fences scalar reducer results with an internal long string literal, constructing an
    // allocating ``std::string`` per dispatch.  A non-owning rank-zero result view suppresses that
    // implicit fence; the resident label below provides the exact same completion boundary.
    Kokkos::View<Real, Kokkos::HostSpace> result_view(&result);
    Kokkos::parallel_reduce(detail::kReduceSumKernelLabel, policy, kernel,
                            Kokkos::Sum<Real, Kokkos::HostSpace>{result_view});
    execution.fence(detail::kReductionFenceLabel);
  } else {
    Kokkos::parallel_reduce(detail::kReduceSumKernelLabel, policy, kernel,
                            Kokkos::Sum<Real>{result});
  }
  return result;
}

/// Allocation-free prepared SUM reduction on an explicit execution-space instance.  Every backend
/// uses the same fixed partition and parallel_for route; the only host operation after the fence is
/// the deterministic fold of the resident HostSpace buffer.
template <class ExecutionSpace, int Dim, class F>
Real for_each_cell_reduce_sum(const ExecutionSpace& execution,
                              const PreparedCellSumReduction<ExecutionSpace>& prepared,
                              const Box<Dim>& b, F f) {
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value,
                "PoPS reduction requires a Kokkos execution-space instance");
  if (b.empty())
    return Real(0);
  if (!prepared.is_prepared())
    throw std::logic_error("prepared cell SUM workspace is not prepared");
  detail::require_iterable_box(b);
  detail::ensure_kokkos_initialized();
  const std::int64_t point_count = b.numPts();
  if (!prepared.accepts(point_count))
    throw std::logic_error(
        "prepared cell SUM workspace capacity does not match the requested cell domain");

  const Index<Dim> lower = b.lo;
#if defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL)
  // The Serial backend has exactly one ordered worker.  Reusing the prepared reduction's device
  // arena would still construct a RangePolicy and profiled fence strings for every accepted
  // reduction, although its result is the same scalar fold.  Keep device and threaded backends
  // on the fixed-partition route below; only the synchronous default backend takes this direct,
  // allocation-free fold.
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::DefaultExecutionSpace>) {
    Real result = Real(0);
    const Extent<Dim> serial_extent = b.extent();
    for (std::int64_t ordinal = 0; ordinal < point_count; ++ordinal)
      result += f(detail::cell_index_from_ordinal(lower, serial_extent, ordinal));
    return result;
  }
#endif
  // A one-point host reduction is already exact and has no ordering to preserve.  Dispatching it
  // through Kokkos creates a RangePolicy and two profiled fence labels on Serial/OpenMP even
  // though the prepared workspace is otherwise allocation-free.  Device spaces retain the
  // explicit reduction path because their captured view may not be host-accessible.
  if constexpr (std::is_same_v<typename ExecutionSpace::memory_space, Kokkos::HostSpace>) {
    if (point_count == 1)
      return f(lower);
  }
  const Extent<Dim> extent = b.extent();
  const auto partials = prepared.partials();
  const std::int64_t partial_count = prepared.partial_count();
  const auto policy = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::int64_t>>(
      execution, 0, partial_count);
  Kokkos::parallel_for(
      detail::kReduceSumKernelLabel, policy, KOKKOS_LAMBDA(const std::int64_t partial_ordinal) {
        const std::int64_t begin = point_count * partial_ordinal / partial_count;
        const std::int64_t end = point_count * (partial_ordinal + 1) / partial_count;
        Real partial = Real(0);
        for (std::int64_t ordinal = begin; ordinal < end; ++ordinal)
          partial += f(detail::cell_index_from_ordinal(lower, extent, ordinal));
        partials(static_cast<std::size_t>(partial_ordinal)) = partial;
      });
  execution.fence(detail::kReductionFenceLabel);
  Kokkos::deep_copy(execution, prepared.host_fold(), partials);
  execution.fence(detail::kReductionFenceLabel);

  Real result = Real(0);
  for (std::int64_t partial_ordinal = 0; partial_ordinal < partial_count; ++partial_ordinal)
    result += prepared.host_fold()(static_cast<std::size_t>(partial_ordinal));
  return result;
}

/// SUM reduction over a compile-time-ranked box on the default execution-space instance.
template <int Dim, class F>
Real for_each_cell_reduce_sum(const Box<Dim>& b, F f) {
  if (b.empty())
    return Real(0);
  detail::ensure_kokkos_initialized();
  return for_each_cell_reduce_sum(detail::default_execution_space(), b, f);
}

/// MAX reduction on an explicit execution-space instance.
template <class ExecutionSpace, int Dim, class F>
Real for_each_cell_reduce_max(const ExecutionSpace& execution, const Box<Dim>& b, F f) {
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value,
                "PoPS reduction requires a Kokkos execution-space instance");
  if (b.empty())
    return Real(0);
  detail::require_iterable_box(b);
  detail::ensure_kokkos_initialized();
  Real result = std::numeric_limits<Real>::lowest();
  const Index<Dim> lower = b.lo;
  const Extent<Dim> extent = b.extent();
#if defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL)
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::DefaultExecutionSpace>) {
    for (std::int64_t ordinal = 0; ordinal < b.numPts(); ++ordinal)
      result = std::max(result, f(detail::cell_index_from_ordinal(lower, extent, ordinal)));
    return result;
  }
#endif
  const auto policy = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::int64_t>>(
      execution, 0, b.numPts());
  const auto kernel = KOKKOS_LAMBDA(const std::int64_t ordinal, Real& accumulator) {
    const Real value = f(detail::cell_index_from_ordinal(lower, extent, ordinal));
    if (value > accumulator)
      accumulator = value;
  };
  if constexpr (std::is_same_v<typename ExecutionSpace::memory_space, Kokkos::HostSpace>) {
    // Kokkos::parallel_reduce still builds an internal completion-label string for a host MAX,
    // even when its result is an unmanaged view.  MAX is associative and exact, so use the
    // public atomic path on the preallocated scalar and retain the explicit resident fence.
    Kokkos::View<Real, Kokkos::HostSpace> result_view(&result);
    Kokkos::parallel_for(
        detail::kReduceMaxKernelLabel, policy, KOKKOS_LAMBDA(const std::int64_t ordinal) {
          Kokkos::atomic_max(&result_view(),
                             f(detail::cell_index_from_ordinal(lower, extent, ordinal)));
        });
    execution.fence(detail::kReductionFenceLabel);
  } else {
    Kokkos::parallel_reduce(detail::kReduceMaxKernelLabel, policy, kernel,
                            Kokkos::Max<Real>{result});
  }
  return result;
}

/// Allocation-free prepared MAX reduction on an explicit execution-space instance.  MAX uses the
/// same deterministic partition as SUM and an ordered host fold, so it never needs an atomic or a
/// backend reducer temporary on the prepared route.
template <class ExecutionSpace, int Dim, class F>
Real for_each_cell_reduce_max(const ExecutionSpace& execution,
                              const PreparedCellSumReduction<ExecutionSpace>& prepared,
                              const Box<Dim>& b, F f) {
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value,
                "PoPS reduction requires a Kokkos execution-space instance");
  if (b.empty())
    return Real(0);
  if (!prepared.is_prepared())
    throw std::logic_error("prepared cell MAX workspace is not prepared");
  detail::require_iterable_box(b);
  detail::ensure_kokkos_initialized();
  const std::int64_t point_count = b.numPts();
  if (!prepared.accepts(point_count))
    throw std::logic_error(
        "prepared cell MAX workspace capacity does not match the requested cell domain");

  const Index<Dim> lower = b.lo;
#if defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL)
  // See the prepared SUM route above.  With one Serial worker this is the same ordered MAX
  // reduction, without per-call Kokkos policy/fence profiling allocations.
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::DefaultExecutionSpace>) {
    Real result = std::numeric_limits<Real>::lowest();
    const Extent<Dim> serial_extent = b.extent();
    for (std::int64_t ordinal = 0; ordinal < point_count; ++ordinal)
      result = std::max(result, f(detail::cell_index_from_ordinal(lower, serial_extent, ordinal)));
    return result;
  }
#endif
  const Extent<Dim> extent = b.extent();
  const auto partials = prepared.partials();
  const std::int64_t partial_count = prepared.partial_count();
  const auto policy = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::int64_t>>(
      execution, 0, partial_count);
  Kokkos::parallel_for(
      detail::kReduceMaxKernelLabel, policy, KOKKOS_LAMBDA(const std::int64_t partial_ordinal) {
        const std::int64_t begin = point_count * partial_ordinal / partial_count;
        const std::int64_t end = point_count * (partial_ordinal + 1) / partial_count;
        Real partial = std::numeric_limits<Real>::lowest();
        for (std::int64_t ordinal = begin; ordinal < end; ++ordinal) {
          const Real value = f(detail::cell_index_from_ordinal(lower, extent, ordinal));
          if (value > partial)
            partial = value;
        }
        partials(static_cast<std::size_t>(partial_ordinal)) = partial;
      });
  execution.fence(detail::kReductionFenceLabel);
  Kokkos::deep_copy(execution, prepared.host_fold(), partials);
  execution.fence(detail::kReductionFenceLabel);

  Real result = std::numeric_limits<Real>::lowest();
  for (std::int64_t partial_ordinal = 0; partial_ordinal < partial_count; ++partial_ordinal)
    result = std::max(result, prepared.host_fold()(static_cast<std::size_t>(partial_ordinal)));
  return result;
}

/// MAX reduction over a compile-time-ranked box on the default execution-space instance.
template <int Dim, class F>
Real for_each_cell_reduce_max(const Box<Dim>& b, F f) {
  if (b.empty())
    return Real(0);
  detail::ensure_kokkos_initialized();
  return for_each_cell_reduce_max(detail::default_execution_space(), b, f);
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
