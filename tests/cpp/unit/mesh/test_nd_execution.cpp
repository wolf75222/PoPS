#include <gtest/gtest.h>

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/fab.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace pops::detail::testing {

struct ReductionResultPoolAccess {
  template <class MemorySpace, class GlobalFence>
  static void release_all_after_global_fence(ReductionResultPool<MemorySpace>& pool,
                                             GlobalFence&& global_fence) noexcept {
    pool.release_all_after_global_fence(std::forward<GlobalFence>(global_fence));
  }

  template <class MemorySpace>
  static std::size_t slot_count(ReductionResultPool<MemorySpace>& pool) {
    std::lock_guard<std::mutex> guard(pool.mutex_);
    return pool.slots_.size();
  }

  template <class MemorySpace>
  static bool mutex_is_available(ReductionResultPool<MemorySpace>& pool) {
    std::unique_lock<std::mutex> guard(pool.mutex_, std::try_to_lock);
    return guard.owns_lock();
  }

  template <class MemorySpace>
  static bool view_is_active(ReductionResultPool<MemorySpace>& pool,
                             const typename ReductionResultPool<MemorySpace>::view_type& view) {
    std::lock_guard<std::mutex> guard(pool.mutex_);
    for (const auto& slot : pool.slots_)
      if (slot.result.data() == view.data())
        return slot.in_use;
    return false;
  }
};

}  // namespace pops::detail::testing

namespace {

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
  static_assert(
      std::is_trivially_copyable_v<pops::detail::IndexSpaceKernelAdapter<3, SetCellValue<3>>>);

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

TEST(test_nd_execution, empty_reductions_return_identity_and_non_addressable_reductions_fail) {
  const pops::Box<1> empty{};
  const pops::Box<1> overflow{pops::Index<1>{0}, pops::Index<1>{std::numeric_limits<int>::max()}};
  pops::detail::ensure_kokkos_initialized();
  Kokkos::DefaultExecutionSpace execution;
  const auto zero = [](const pops::CellIndex<1>&) -> pops::Real { return pops::Real(0); };

  EXPECT_EQ(pops::for_each_cell_reduce_sum(execution, empty, zero), pops::Real(0));
  EXPECT_EQ(pops::for_each_cell_reduce_max(execution, empty, zero), pops::Real(0));
  EXPECT_THROW((void)pops::for_each_cell_reduce_sum(execution, overflow, zero),
               std::overflow_error);
  EXPECT_THROW((void)pops::for_each_cell_reduce_max(execution, overflow, zero),
               std::overflow_error);
}

TEST(test_reduction_result_pool,
     distinct_execution_instances_keep_overlapping_reductions_in_distinct_slots) {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using memory_space = typename execution_space::memory_space;
  using pool_type = pops::detail::ReductionResultPool<memory_space>;
  using access = pops::detail::testing::ReductionResultPoolAccess;

  pops::detail::ensure_kokkos_initialized();
  execution_space first_execution;
  execution_space second_execution;
  auto& pool = pool_type::instance();
  auto first = pool.acquire("pops_reduction_pool_first_execution_instance");
  auto second = pool.acquire("pops_reduction_pool_second_execution_instance");

  // Both reductions are submitted before either execution instance is fenced.  On CUDA/HIP these
  // are distinct Kokkos execution-space instances (and therefore independently fenceable streams);
  // Serial/OpenMP retain the same slot-exclusivity and result-correctness contract portably.
  ASSERT_NE(first.view().data(), second.view().data());
  EXPECT_TRUE(access::view_is_active(pool, first.view()));
  EXPECT_TRUE(access::view_is_active(pool, second.view()));

  constexpr std::int64_t first_count = 4099;
  constexpr std::int64_t second_count = 6151;
  Kokkos::parallel_reduce(
      "pops_reduction_pool_first_execution_instance",
      Kokkos::RangePolicy<execution_space>(first_execution, 0, first_count),
      KOKKOS_LAMBDA(const int, pops::Real& total) { total += pops::Real(2); },
      Kokkos::Sum<pops::Real, memory_space>{first.view()});
  Kokkos::parallel_reduce(
      "pops_reduction_pool_second_execution_instance",
      Kokkos::RangePolicy<execution_space>(second_execution, 0, second_count),
      KOKKOS_LAMBDA(const int, pops::Real& total) { total += pops::Real(3); },
      Kokkos::Sum<pops::Real, memory_space>{second.view()});

  pops::Real first_result = 0;
  Kokkos::deep_copy(first_execution, first_result, first.view());
  first_execution.fence("pops_reduction_pool_first_execution_instance_result");
  EXPECT_EQ(first_result, pops::Real(2 * first_count));

  // Only the first slot becomes reusable after its own instance-local completion.  The second
  // lease remains held until its own fence/copy completes; no process-wide Kokkos fence is used.
  first.release();
  EXPECT_FALSE(access::view_is_active(pool, first.view()));
  EXPECT_TRUE(access::view_is_active(pool, second.view()));

  pops::Real second_result = 0;
  Kokkos::deep_copy(second_execution, second_result, second.view());
  second_execution.fence("pops_reduction_pool_second_execution_instance_result");
  EXPECT_EQ(second_result, pops::Real(3 * second_count));
  second.release();
  EXPECT_FALSE(access::view_is_active(pool, second.view()));
}

TEST(test_reduction_result_pool, public_lifecycle_registration_remains_observable_after_use) {
  // A Kokkos finalizer hook is intentionally process-global.  This GoogleTest binary keeps Kokkos
  // live across unrelated tests, so calling Kokkos::finalize() here (or in a death-test child after
  // a CUDA runtime has been initialized) is unsafe and would not be a portable lifecycle proof.
  // Instead exercise the public registration state after the pool's real acquire path; the actual
  // hook executes during normal process teardown and is covered by the ROMEO backend campaigns.
  pops::detail::ensure_kokkos_initialized();
  using pool_type = pops::detail::ReductionResultPool<Kokkos::DefaultExecutionSpace::memory_space>;
  auto lease = pool_type::instance().acquire("pops_reduction_pool_public_lifecycle");
  EXPECT_TRUE(Kokkos::is_initialized());
  EXPECT_TRUE(pops::kokkos_atexit_finalize_registered() || !pops::kokkos_initialized_by_pops());
  lease.release();
}

TEST(test_reduction_result_pool, finalize_barrier_failure_preserves_abandoned_host_lease) {
  using pool_type = pops::detail::ReductionResultPool<Kokkos::HostSpace>;
  pops::detail::ensure_kokkos_initialized();
  auto& pool = pool_type::instance();
  using access = pops::detail::testing::ReductionResultPoolAccess;

  access::release_all_after_global_fence(pool, [] {});
  auto abandoned = pool.acquire("pops_reduction_pool_abandoned_host_lease");
  abandoned.abandon();
  const std::size_t slots_before_failure = access::slot_count(pool);
  ASSERT_EQ(slots_before_failure, 1u);

  access::release_all_after_global_fence(pool, [] { throw std::runtime_error("fence failure"); });
  EXPECT_EQ(access::slot_count(pool), slots_before_failure);

  abandoned = {};
  bool barrier_saw_unlocked_pool = false;
  access::release_all_after_global_fence(
      pool, [&] { barrier_saw_unlocked_pool = access::mutex_is_available(pool); });
  EXPECT_TRUE(barrier_saw_unlocked_pool);
  EXPECT_EQ(access::slot_count(pool), 0u);

  auto reusable = pool.acquire("pops_reduction_pool_reusable_host_lease");
  EXPECT_EQ(access::slot_count(pool), 1u);
  reusable.release();
  auto reused = pool.acquire("pops_reduction_pool_reused_host_lease");
  EXPECT_EQ(access::slot_count(pool), 1u);
  reused.release();
}
