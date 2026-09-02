#include <gtest/gtest.h>

#include <pops/core/foundation/allocator.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/fab.hpp>

#include <Kokkos_Core.hpp>

#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>

#include "nd_multifab_test_utils.hpp"

namespace {

static_assert(
    std::is_nothrow_swappable_v<pops::PreparedCellSumReduction<Kokkos::DefaultExecutionSpace>>);

// This executable owns a narrow global heap witness so the prepared reduction is checked at the
// same boundary as the strict Program transaction test.  Measurement is opt-in and is enabled only
// around the already-prepared dispatch under test.
std::atomic<bool> g_heap_measurement_enabled{false};
std::atomic<std::uint64_t> g_measured_heap_allocations{0};

void note_measured_heap_allocation() noexcept {
  if (g_heap_measurement_enabled.load(std::memory_order_relaxed))
    g_measured_heap_allocations.fetch_add(1, std::memory_order_relaxed);
}

void* measured_allocate(std::size_t size) {
  void* pointer = std::malloc(size == 0 ? 1 : size);
  if (pointer == nullptr)
    throw std::bad_alloc();
  note_measured_heap_allocation();
  return pointer;
}

void* measured_aligned_allocate(std::size_t size, std::size_t alignment) {
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size == 0 ? 1 : size) != 0)
    throw std::bad_alloc();
  note_measured_heap_allocation();
  return pointer;
}

class HeapAllocationWindow {
 public:
  HeapAllocationWindow() : before_(g_measured_heap_allocations.load(std::memory_order_relaxed)) {
    g_heap_measurement_enabled.store(true, std::memory_order_relaxed);
  }

  ~HeapAllocationWindow() { g_heap_measurement_enabled.store(false, std::memory_order_relaxed); }

  [[nodiscard]] std::uint64_t close() noexcept {
    g_heap_measurement_enabled.store(false, std::memory_order_relaxed);
    return g_measured_heap_allocations.load(std::memory_order_relaxed) - before_;
  }

 private:
  std::uint64_t before_ = 0;
};

template <int Dim>
pops::Box<Dim> sample_box() {
  if constexpr (Dim == 1) {
    return pops::Box<1>{pops::Index<1>{-2}, pops::Index<1>{2}};
  } else if constexpr (Dim == 2) {
    return pops::Box<2>{pops::Index<2>{-2, 3}, pops::Index<2>{1, 5}};
  } else {
    return pops::Box<3>{pops::Index<3>{-1, 2, 7}, pops::Index<3>{1, 4, 8}};
  }
}

template <int Dim>
struct SetCellValue {
  pops::FieldView<pops::Real, Dim> values;
  pops::Real value;

  POPS_HD void operator()(const pops::CellIndex<Dim>& index) const { values(index) = value; }
};

template <int Dim>
struct ReadCellValue {
  pops::FieldView<const pops::Real, Dim> values;

  POPS_HD pops::Real operator()(const pops::CellIndex<Dim>& index) const { return values(index); }
};

struct DeterministicSumValue {
  POPS_HD pops::Real operator()(const pops::CellIndex<1>& index) const {
    const pops::Real shifted = static_cast<pops::Real>(index[0] + 37);
    return (index[0] & 1) == 0 ? shifted / static_cast<pops::Real>(7)
                               : -shifted / static_cast<pops::Real>(13);
  }
};

struct DeterministicMaxValue {
  POPS_HD pops::Real operator()(const pops::CellIndex<1>& index) const {
    const pops::Real shifted = static_cast<pops::Real>(index[0] + 37);
    return (index[0] & 1) == 0 ? shifted / static_cast<pops::Real>(7)
                               : -shifted / static_cast<pops::Real>(13);
  }
};

template <int Dim>
int arithmetic_code(const pops::Index<Dim>& index, int component) {
  int code = 17 * component;
  for (int axis = 0; axis < Dim; ++axis)
    code += (axis + 2) * (index[axis] + 7);
  return code;
}

template <int Dim>
pops::Real arithmetic_value(const pops::Index<Dim>& index, int component) {
  return static_cast<pops::Real>((arithmetic_code(index, component) % 19) - 9);
}

template <int Dim>
bool is_active(const pops::Index<Dim>& index) {
  return arithmetic_code(index, 0) % 2 == 0;
}

template <int Dim>
struct PreparedArithmeticResults {
  pops::Real sum = 0;
  pops::Real absolute_sum = 0;
  pops::Real maximum = -std::numeric_limits<pops::Real>::infinity();
  pops::Real minimum = std::numeric_limits<pops::Real>::infinity();
  pops::Real norm_inf = 0;
  pops::Real dot = 0;
  pops::Real dot_all = 0;
  pops::Real norm2 = 0;
  pops::Real active_sum = 0;
  pops::Real active_absolute_sum = 0;
  pops::Real active_maximum = -std::numeric_limits<pops::Real>::infinity();
  pops::Real active_minimum = std::numeric_limits<pops::Real>::infinity();
  pops::Real active_norm_inf = 0;
  pops::Real active_dot = 0;
  pops::Real active_norm2 = 0;
};

template <int Dim>
PreparedArithmeticResults<Dim> prepared_arithmetic_results(
    const pops::test::nd::HostMultiFab<Dim>& field, const pops::test::nd::HostMultiFab<Dim>& active,
    const Kokkos::DefaultHostExecutionSpace& execution,
    const pops::PreparedCellSumReduction<Kokkos::DefaultHostExecutionSpace>& prepared) {
  using pops::reduce_active_abs_sum_local;
  using pops::reduce_active_max_local;
  using pops::reduce_active_min_local;
  using pops::reduce_active_norm_inf_local;
  using pops::reduce_active_sum_local;
  using pops::reduce_abs_sum_local;
  using pops::reduce_max_local;
  using pops::reduce_min_local;
  using pops::reduce_norm_inf;
  using pops::reduce_sum_local;
  using pops::dot_active_local;
  using pops::dot_all_local;
  using pops::dot_local;
  using pops::norm2_local;

  PreparedArithmeticResults<Dim> result;
  result.sum = reduce_sum_local(field, execution, prepared, 0);
  result.absolute_sum = reduce_abs_sum_local(field, execution, prepared, 0);
  result.maximum = reduce_max_local(field, execution, prepared, 0);
  result.minimum = reduce_min_local(field, execution, prepared, 0);
  result.norm_inf = norm_inf(field, execution, prepared, 0);
  result.dot = dot_local(field, field, execution, prepared, 0);
  result.dot_all = dot_all_local(field, field, execution, prepared);
  result.norm2 = norm2_local(field, 0, execution, prepared);
  result.active_sum = reduce_active_sum_local(field, 0, &active, execution, prepared);
  result.active_absolute_sum = reduce_active_abs_sum_local(field, 0, &active, execution, prepared);
  result.active_maximum = reduce_active_max_local(field, 0, &active, execution, prepared);
  result.active_minimum = reduce_active_min_local(field, 0, &active, execution, prepared);
  result.active_norm_inf = reduce_active_norm_inf_local(field, 0, &active, execution, prepared);
  result.active_dot = dot_active_local(field, field, 0, &active, execution, prepared);
  result.active_norm2 = norm2_local(field, 0, &active, execution, prepared);
  return result;
}

template <int Dim>
void expect_same_prepared_arithmetic_bits(const PreparedArithmeticResults<Dim>& first,
                                          const PreparedArithmeticResults<Dim>& repeated) {
  EXPECT_EQ(std::bit_cast<std::uint64_t>(first.sum), std::bit_cast<std::uint64_t>(repeated.sum));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(first.absolute_sum),
            std::bit_cast<std::uint64_t>(repeated.absolute_sum));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(first.maximum),
            std::bit_cast<std::uint64_t>(repeated.maximum));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(first.minimum),
            std::bit_cast<std::uint64_t>(repeated.minimum));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(first.norm_inf),
            std::bit_cast<std::uint64_t>(repeated.norm_inf));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(first.dot), std::bit_cast<std::uint64_t>(repeated.dot));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(first.dot_all),
            std::bit_cast<std::uint64_t>(repeated.dot_all));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(first.norm2),
            std::bit_cast<std::uint64_t>(repeated.norm2));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(first.active_sum),
            std::bit_cast<std::uint64_t>(repeated.active_sum));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(first.active_absolute_sum),
            std::bit_cast<std::uint64_t>(repeated.active_absolute_sum));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(first.active_maximum),
            std::bit_cast<std::uint64_t>(repeated.active_maximum));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(first.active_minimum),
            std::bit_cast<std::uint64_t>(repeated.active_minimum));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(first.active_norm_inf),
            std::bit_cast<std::uint64_t>(repeated.active_norm_inf));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(first.active_dot),
            std::bit_cast<std::uint64_t>(repeated.active_dot));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(first.active_norm2),
            std::bit_cast<std::uint64_t>(repeated.active_norm2));
}

template <int Dim, int Axis>
struct SetFaceValue {
  pops::FieldView<pops::Real, Dim> values;

  POPS_HD void operator()(const pops::FaceIndex<Dim, Axis>& face) const {
    static_assert(pops::FaceIndex<Dim, Axis>::normal_axis == Axis);
    values(face.coordinate) = static_cast<pops::Real>(Axis + 1);
  }
};

template <int Dim>
void expect_cell_and_product_execution() {
  const pops::Box<Dim> cells = sample_box<Dim>();
  pops::Fab<Dim> field(cells, 1);
  Kokkos::DefaultExecutionSpace execution;

  pops::for_each_cell(execution, cells, SetCellValue<Dim>{field.view(), pops::Real(2)});
  EXPECT_EQ(
      pops::for_each_cell_reduce_sum(
          execution, cells, ReadCellValue<Dim>{static_cast<const pops::Fab<Dim>&>(field).view()}),
      static_cast<pops::Real>(2 * cells.numPts()));
  EXPECT_EQ(
      pops::for_each_cell_reduce_max(
          execution, cells, ReadCellValue<Dim>{static_cast<const pops::Fab<Dim>&>(field).view()}),
      pops::Real(2));

  pops::for_each_product(cells, SetCellValue<Dim>{field.view(), pops::Real(3)});
  EXPECT_EQ(pops::for_each_product_reduce_sum(
                cells, ReadCellValue<Dim>{static_cast<const pops::Fab<Dim>&>(field).view()}),
            static_cast<pops::Real>(3 * cells.numPts()));
}

template <int Dim, int Axis>
void expect_face_execution() {
  const pops::Box<Dim> cells = sample_box<Dim>();
  const pops::Box<Dim> faces = pops::face_box<Axis>(cells);
  pops::Fab<Dim> field(faces, 1);
  Kokkos::DefaultExecutionSpace execution;

  pops::for_each_face<Axis>(execution, cells, SetFaceValue<Dim, Axis>{field.view()});
  const pops::Real total = pops::for_each_cell_reduce_sum(
      execution, faces, ReadCellValue<Dim>{static_cast<const pops::Fab<Dim>&>(field).view()});
  EXPECT_EQ(total, static_cast<pops::Real>((Axis + 1) * faces.numPts()));
  EXPECT_EQ(faces.length(Axis), cells.length(Axis) + 1);
  for (int tangent = 0; tangent < Dim; ++tangent)
    if (tangent != Axis)
      EXPECT_EQ(faces.length(tangent), cells.length(tangent));
}

}  // namespace

TEST(test_nd_execution, cell_and_product_facades_share_one_exact_ranked_policy) {
  static_assert(std::is_same_v<pops::CellIndex<2>, pops::Index<2>>);
  static_assert(std::is_trivially_copyable_v<pops::FaceIndex<3, 2>>);

  expect_cell_and_product_execution<1>();
  expect_cell_and_product_execution<2>();
  expect_cell_and_product_execution<3>();
}

TEST(test_nd_execution, face_axis_is_compile_time_and_each_dimension_has_exact_face_extent) {
  expect_face_execution<1, 0>();
  expect_face_execution<2, 0>();
  expect_face_execution<2, 1>();
  expect_face_execution<3, 0>();
  expect_face_execution<3, 1>();
  expect_face_execution<3, 2>();
}

TEST(test_nd_execution, empty_and_non_addressable_face_domains_fail_deterministically) {
  EXPECT_TRUE(pops::face_box<0>(pops::Box<1>{}).empty());
  const pops::Box<1> overflow{pops::Index<1>{0}, pops::Index<1>{std::numeric_limits<int>::max()}};
  EXPECT_THROW((void)pops::face_box<0>(overflow), std::overflow_error);
}

TEST(test_nd_execution, prepared_host_sum_is_deterministic_and_rejects_unprepared_capacity) {
  using HostExecution = Kokkos::DefaultHostExecutionSpace;
  static_assert(std::is_same_v<typename HostExecution::memory_space, Kokkos::HostSpace>);

  pops::detail::ensure_kokkos_initialized();
  const HostExecution execution{};
  const pops::Box<1> cells{pops::Index<1>{-17}, pops::Index<1>{1006}};
  pops::PreparedCellSumReduction<HostExecution> unprepared;
  EXPECT_THROW(
      (void)pops::for_each_cell_reduce_sum(execution, unprepared, cells, DeterministicSumValue{}),
      std::logic_error);
  pops::PreparedCellSumReduction<HostExecution> prepared;
  prepared.prepare(execution, cells.numPts());

  const auto events_before = pops::allocation_event_stats();
  pops::Real first = 0;
  std::uint64_t first_heap_allocations = 0;
  {
    HeapAllocationWindow heap_window;
    first = pops::for_each_cell_reduce_sum(execution, prepared, cells, DeterministicSumValue{});
    first_heap_allocations = heap_window.close();
  }
  EXPECT_EQ(first_heap_allocations, 0u);
  EXPECT_EQ(pops::allocation_event_stats(), events_before);
  const std::uint64_t first_bits = std::bit_cast<std::uint64_t>(first);
  for (int iteration = 0; iteration < 16; ++iteration) {
    const auto repeated_events_before = pops::allocation_event_stats();
    pops::Real repeated = 0;
    std::uint64_t repeated_heap_allocations = 0;
    {
      HeapAllocationWindow heap_window;
      repeated =
          pops::for_each_cell_reduce_sum(execution, prepared, cells, DeterministicSumValue{});
      repeated_heap_allocations = heap_window.close();
    }
    EXPECT_EQ(repeated_heap_allocations, 0u);
    EXPECT_EQ(pops::allocation_event_stats(), repeated_events_before);
    EXPECT_EQ(std::bit_cast<std::uint64_t>(repeated), first_bits);
  }

  pops::PreparedCellSumReduction<HostExecution> undersized;
  undersized.prepare(execution, cells.numPts() - 1);
  EXPECT_THROW(
      (void)pops::for_each_cell_reduce_sum(execution, undersized, cells, DeterministicSumValue{}),
      std::logic_error);
}

TEST(test_nd_execution, prepared_host_max_is_deterministic_and_allocation_free) {
  using HostExecution = Kokkos::DefaultHostExecutionSpace;
  pops::detail::ensure_kokkos_initialized();
  const HostExecution execution{};
  const pops::Box<1> cells{pops::Index<1>{-17}, pops::Index<1>{1006}};
  pops::PreparedCellMaxReduction<HostExecution> prepared;
  prepared.prepare(execution, cells.numPts());

  const auto events_before = pops::allocation_event_stats();
  std::uint64_t first_heap_allocations = 0;
  pops::Real first = 0;
  {
    HeapAllocationWindow heap_window;
    first = pops::for_each_cell_reduce_max(execution, prepared, cells, DeterministicMaxValue{});
    first_heap_allocations = heap_window.close();
  }
  EXPECT_EQ(first_heap_allocations, 0u);
  EXPECT_EQ(pops::allocation_event_stats(), events_before);
  const std::uint64_t first_bits = std::bit_cast<std::uint64_t>(first);
  for (int iteration = 0; iteration < 16; ++iteration) {
    const auto repeated_events_before = pops::allocation_event_stats();
    std::uint64_t repeated_heap_allocations = 0;
    pops::Real repeated = 0;
    {
      HeapAllocationWindow heap_window;
      repeated =
          pops::for_each_cell_reduce_max(execution, prepared, cells, DeterministicMaxValue{});
      repeated_heap_allocations = heap_window.close();
    }
    EXPECT_EQ(repeated_heap_allocations, 0u);
    EXPECT_EQ(pops::allocation_event_stats(), repeated_events_before);
    EXPECT_EQ(std::bit_cast<std::uint64_t>(repeated), first_bits);
  }
}

TEST(test_nd_execution, prepared_multifab_arithmetic_routes_are_exact_and_allocation_free) {
  constexpr int Dim = 2;
  using HostExecution = Kokkos::DefaultHostExecutionSpace;
  using HostMultiFab = pops::test::nd::HostMultiFab<Dim>;
  pops::detail::ensure_kokkos_initialized();
  const HostExecution execution{};
  const pops::Box<Dim> domain = pops::test::nd::cube<Dim>(-2, 3);
  const pops::mesh::BoxArray<Dim> layout =
      pops::mesh::BoxArray<Dim>::from_domain(domain, pops::Extent<Dim>{2, 3});
  const auto distribution =
      pops::mesh::Distribution<Dim>::replicated(layout, pops::test::nd::one_rank_space<Dim>());
  HostMultiFab field(layout, distribution, pops::Index<Dim>{}, 2, pops::Extent<Dim>{});
  HostMultiFab active(layout, distribution, pops::Index<Dim>{}, 1, pops::Extent<Dim>{});
  pops::test::nd::fill_valid(field, pops::Real{0},
                             [](const pops::Index<Dim>& index, int component) {
                               return arithmetic_value(index, component);
                             });
  pops::test::nd::fill_valid(active, pops::Real{0}, [](const pops::Index<Dim>& index, int) {
    return is_active(index) ? pops::Real{1} : pops::Real{0};
  });

  std::int64_t maximum_points = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local)
    maximum_points = std::max(maximum_points, field.box(local).numPts());
  pops::PreparedCellSumReduction<HostExecution> prepared;
  prepared.prepare(execution, maximum_points);

  PreparedArithmeticResults<Dim> expected;
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const pops::Box<Dim>& box = field.box(local);
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(box.numPts()); ++ordinal) {
      const pops::Index<Dim> index = pops::test::nd::index_from_ordinal(box, ordinal);
      const pops::Real value0 = arithmetic_value(index, 0);
      const pops::Real value1 = arithmetic_value(index, 1);
      expected.sum += value0;
      expected.absolute_sum += value0 < 0 ? -value0 : value0;
      expected.maximum = std::max(expected.maximum, value0);
      expected.minimum = std::min(expected.minimum, value0);
      expected.norm_inf = std::max(expected.norm_inf, value0 < 0 ? -value0 : value0);
      expected.dot += value0 * value0;
      expected.dot_all += value0 * value0 + value1 * value1;
      if (is_active(index)) {
        expected.active_sum += value0;
        expected.active_absolute_sum += value0 < 0 ? -value0 : value0;
        expected.active_maximum = std::max(expected.active_maximum, value0);
        expected.active_minimum = std::min(expected.active_minimum, value0);
        expected.active_norm_inf =
            std::max(expected.active_norm_inf, value0 < 0 ? -value0 : value0);
        expected.active_dot += value0 * value0;
      }
    }
  }
  expected.norm2 = std::sqrt(expected.dot);
  expected.active_norm2 = std::sqrt(expected.active_dot);

  const auto events_before = pops::allocation_event_stats();
  PreparedArithmeticResults<Dim> first;
  std::uint64_t first_heap_allocations = 0;
  {
    HeapAllocationWindow heap_window;
    first = prepared_arithmetic_results(field, active, execution, prepared);
    first_heap_allocations = heap_window.close();
  }
  EXPECT_EQ(first_heap_allocations, 0u);
  EXPECT_EQ(pops::allocation_event_stats(), events_before);
  EXPECT_EQ(first.sum, expected.sum);
  EXPECT_EQ(first.absolute_sum, expected.absolute_sum);
  EXPECT_EQ(first.maximum, expected.maximum);
  EXPECT_EQ(first.minimum, expected.minimum);
  EXPECT_EQ(first.norm_inf, expected.norm_inf);
  EXPECT_EQ(first.dot, expected.dot);
  EXPECT_EQ(first.dot_all, expected.dot_all);
  EXPECT_EQ(first.norm2, expected.norm2);
  EXPECT_EQ(first.active_sum, expected.active_sum);
  EXPECT_EQ(first.active_absolute_sum, expected.active_absolute_sum);
  EXPECT_EQ(first.active_maximum, expected.active_maximum);
  EXPECT_EQ(first.active_minimum, expected.active_minimum);
  EXPECT_EQ(first.active_norm_inf, expected.active_norm_inf);
  EXPECT_EQ(first.active_dot, expected.active_dot);
  EXPECT_EQ(first.active_norm2, expected.active_norm2);

  for (int iteration = 0; iteration < 8; ++iteration) {
    const auto repeated_events_before = pops::allocation_event_stats();
    PreparedArithmeticResults<Dim> repeated;
    std::uint64_t repeated_heap_allocations = 0;
    {
      HeapAllocationWindow heap_window;
      repeated = prepared_arithmetic_results(field, active, execution, prepared);
      repeated_heap_allocations = heap_window.close();
    }
    EXPECT_EQ(repeated_heap_allocations, 0u);
    EXPECT_EQ(pops::allocation_event_stats(), repeated_events_before);
    expect_same_prepared_arithmetic_bits(first, repeated);
  }

  pops::PreparedCellSumReduction<HostExecution> undersized;
  undersized.prepare(execution, maximum_points - 1);
  EXPECT_THROW((void)pops::reduce_sum_local(field, execution, undersized), std::logic_error);
}

void* operator new(std::size_t size) {
  return measured_allocate(size);
}

void* operator new[](std::size_t size) {
  return measured_allocate(size);
}

void operator delete(void* pointer) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer) noexcept {
  std::free(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept {
  std::free(pointer);
}

void* operator new(std::size_t size, std::align_val_t alignment) {
  return measured_aligned_allocate(size, static_cast<std::size_t>(alignment));
}

void* operator new[](std::size_t size, std::align_val_t alignment) {
  return measured_aligned_allocate(size, static_cast<std::size_t>(alignment));
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  try {
    return measured_allocate(size);
  } catch (...) {
    return nullptr;
  }
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
  try {
    return measured_allocate(size);
  } catch (...) {
    return nullptr;
  }
}

void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  try {
    return measured_aligned_allocate(size, static_cast<std::size_t>(alignment));
  } catch (...) {
    return nullptr;
  }
}

void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  try {
    return measured_aligned_allocate(size, static_cast<std::size_t>(alignment));
  } catch (...) {
    return nullptr;
  }
}

void operator delete(void* pointer, std::align_val_t) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer, std::align_val_t) noexcept {
  std::free(pointer);
}

void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept {
  std::free(pointer);
}

void operator delete(void* pointer, const std::nothrow_t&) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer, const std::nothrow_t&) noexcept {
  std::free(pointer);
}

void operator delete(void* pointer, std::size_t, const std::nothrow_t&) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer, std::size_t, const std::nothrow_t&) noexcept {
  std::free(pointer);
}

void operator delete(void* pointer, std::align_val_t, const std::nothrow_t&) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer, std::align_val_t, const std::nothrow_t&) noexcept {
  std::free(pointer);
}

void operator delete(void* pointer, std::size_t, std::align_val_t, const std::nothrow_t&) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer, std::size_t, std::align_val_t,
                       const std::nothrow_t&) noexcept {
  std::free(pointer);
}
