#include <gtest/gtest.h>

#include <pops/numerics/nonlinear/prepared_variable_recovery.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <type_traits>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace {

std::atomic<std::uint64_t> g_heap_allocations{0};

void* counted_allocate(std::size_t size) {
  void* pointer = std::malloc(size == 0 ? 1 : size);
  if (pointer == nullptr)
    throw std::bad_alloc();
  g_heap_allocations.fetch_add(1, std::memory_order_relaxed);
  return pointer;
}

void* counted_aligned_allocate(std::size_t size, std::size_t alignment) {
  void* pointer = nullptr;
#if defined(_MSC_VER)
  pointer = _aligned_malloc(size == 0 ? 1 : size, alignment);
#else
  if (posix_memalign(&pointer, alignment, size == 0 ? 1 : size) != 0)
    pointer = nullptr;
#endif
  if (pointer == nullptr)
    throw std::bad_alloc();
  g_heap_allocations.fetch_add(1, std::memory_order_relaxed);
  return pointer;
}

void counted_aligned_free(void* pointer) noexcept {
#if defined(_MSC_VER)
  _aligned_free(pointer);
#else
  std::free(pointer);
#endif
}

using pops::Real;

struct PositiveCandidate {
  POPS_HD bool operator()(const Real (&value)[1], int* component = nullptr) const {
    const bool accepted = value[0] > Real(0);
    if (component != nullptr)
      *component = accepted ? -1 : 0;
    return accepted;
  }
};

struct SquareResidual {
  Real target = 0;

  POPS_HD pops::LocalNonlinearEvaluationResult operator()(const Real (&value)[1],
                                                          Real (&residual)[1]) const {
    residual[0] = value[0] * value[0] - target;
    return pops::LocalNonlinearEvaluationResult::ok();
  }
};

struct SquareProblemFactory {
  POPS_HD auto operator()(const Real (&conserved)[1]) const {
    pops::PreparedLocalNonlinearControls controls;
    controls.max_iterations = 16;
    controls.absolute_tolerance = Real(1e-13);
    return pops::prepare_local_nonlinear_problem<1>(SquareResidual{conserved[0]},
                                                    pops::FiniteDifferenceLocalJacobian<1>{},
                                                    PositiveCandidate{}, controls);
  }
};

struct RejectingResidual {
  POPS_HD pops::LocalNonlinearEvaluationResult operator()(const Real (&)[1], Real (&)[1]) const {
    return pops::LocalNonlinearEvaluationResult::reject(731);
  }
};

struct RejectingProblemFactory {
  POPS_HD auto operator()(const Real (&)[1]) const {
    pops::PreparedLocalNonlinearControls controls;
    return pops::prepare_local_nonlinear_problem<1>(RejectingResidual{},
                                                    pops::FiniteDifferenceLocalJacobian<1>{},
                                                    PositiveCandidate{}, controls);
  }
};

}  // namespace

void* operator new(std::size_t size) {
  return counted_allocate(size);
}

void* operator new[](std::size_t size) {
  return counted_allocate(size);
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

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  try {
    return counted_allocate(size);
  } catch (...) {
    return nullptr;
  }
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
  try {
    return counted_allocate(size);
  } catch (...) {
    return nullptr;
  }
}

void operator delete(void* pointer, const std::nothrow_t&) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer, const std::nothrow_t&) noexcept {
  std::free(pointer);
}

void* operator new(std::size_t size, std::align_val_t alignment) {
  return counted_aligned_allocate(size, static_cast<std::size_t>(alignment));
}

void* operator new[](std::size_t size, std::align_val_t alignment) {
  return counted_aligned_allocate(size, static_cast<std::size_t>(alignment));
}

void operator delete(void* pointer, std::align_val_t) noexcept {
  counted_aligned_free(pointer);
}

void operator delete[](void* pointer, std::align_val_t) noexcept {
  counted_aligned_free(pointer);
}

void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept {
  counted_aligned_free(pointer);
}

void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept {
  counted_aligned_free(pointer);
}

void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  try {
    return counted_aligned_allocate(size, static_cast<std::size_t>(alignment));
  } catch (...) {
    return nullptr;
  }
}

void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  try {
    return counted_aligned_allocate(size, static_cast<std::size_t>(alignment));
  } catch (...) {
    return nullptr;
  }
}

void operator delete(void* pointer, std::align_val_t, const std::nothrow_t&) noexcept {
  counted_aligned_free(pointer);
}

void operator delete[](void* pointer, std::align_val_t, const std::nothrow_t&) noexcept {
  counted_aligned_free(pointer);
}

TEST(PreparedNumericsGate, AllocationProbeDetectsControlHeapTraffic) {
  const std::uint64_t before = g_heap_allocations.load(std::memory_order_relaxed);
  void* ordinary = ::operator new(32);
  void* aligned = ::operator new(64, std::align_val_t{64});
  const std::uint64_t after = g_heap_allocations.load(std::memory_order_relaxed);
  ::operator delete(ordinary);
  ::operator delete(aligned, std::align_val_t{64});

  EXPECT_EQ(after - before, std::uint64_t{2});
}

TEST(PreparedNumericsGate, ConvergedPreparedPathAllocatesNothingAndRollsBack) {
  const Real conserved[1] = {Real(4)};
  const Real initial[1] = {Real(1)};
  const auto problem = SquareProblemFactory{}(conserved);
  const auto methods =
      pops::recovery_methods(pops::prepared_local_nonlinear_recovery<1>(SquareProblemFactory{}));
  const auto plan = pops::prepare_variable_recovery<1>(PositiveCandidate{}, methods);
  static_assert(std::is_trivially_copyable_v<decltype(problem)>);
  static_assert(std::is_trivially_copyable_v<decltype(plan)>);

  Real accepted[1] = {Real(9)};
  pops::RecoveryWarmStartSlot<1> cache;
  const Real cached[1] = {Real(8)};
  cache.store(cached, 3, 7);
  pops::RecoveryPublicationTransaction<1> transaction(accepted, cache);

  const std::uint64_t before = g_heap_allocations.load(std::memory_order_relaxed);
  const auto local = pops::solve_prepared_local_nonlinear(problem, initial);
  const auto recovered = pops::recover_prepared_variable(plan, conserved, initial);
  const bool published = transaction.publish_tentative(recovered, 4, 8);
  const bool rolled_back = transaction.rollback();
  const std::uint64_t after = g_heap_allocations.load(std::memory_order_relaxed);

  ASSERT_TRUE(local.solved());
  ASSERT_TRUE(recovered.recovered());
  EXPECT_TRUE(published);
  EXPECT_TRUE(rolled_back);
  EXPECT_EQ(after, before);
  EXPECT_NEAR(local.value[0], Real(2), Real(1e-10));
  EXPECT_NEAR(recovered.value[0], Real(2), Real(1e-10));
  EXPECT_EQ(accepted[0], Real(9));
  EXPECT_EQ(cache.value[0], Real(8));
  EXPECT_EQ(cache.topology_generation, std::uint64_t{3});
  EXPECT_EQ(cache.state_generation, std::uint64_t{7});
}

TEST(PreparedNumericsGate, FatalEvaluationIsTypedAllocationFreeAndCannotPublish) {
  const Real conserved[1] = {Real(4)};
  const Real initial[1] = {Real(1)};
  const auto problem = RejectingProblemFactory{}(conserved);
  const auto methods =
      pops::recovery_methods(pops::prepared_local_nonlinear_recovery<1>(RejectingProblemFactory{}));
  const auto plan = pops::prepare_variable_recovery<1>(PositiveCandidate{}, methods);

  Real accepted[1] = {Real(9)};
  pops::RecoveryWarmStartSlot<1> cache;
  const Real cached[1] = {Real(8)};
  cache.store(cached, 3, 7);
  pops::RecoveryPublicationTransaction<1> transaction(accepted, cache);

  const std::uint64_t before = g_heap_allocations.load(std::memory_order_relaxed);
  const auto local = pops::solve_prepared_local_nonlinear(problem, initial);
  const auto recovered = pops::recover_prepared_variable(plan, conserved, initial);
  const bool published = transaction.publish_tentative(recovered, 4, 8);
  const bool rolled_back = transaction.rollback();
  const std::uint64_t after = g_heap_allocations.load(std::memory_order_relaxed);

  EXPECT_EQ(local.status, pops::LocalNonlinearStatus::kEvaluationReject);
  EXPECT_EQ(local.reason_code, std::uint32_t{731});
  EXPECT_EQ(local.value[0], initial[0]);
  EXPECT_EQ(recovered.status, pops::RecoveryStatus::kRejected);
  EXPECT_EQ(recovered.cause, pops::RecoveryCause::kEvaluationReject);
  EXPECT_EQ(recovered.reason_code, std::uint32_t{731});
  EXPECT_FALSE(published);
  EXPECT_TRUE(rolled_back);
  EXPECT_EQ(after, before);
  EXPECT_EQ(accepted[0], Real(9));
  EXPECT_EQ(cache.value[0], Real(8));
  EXPECT_EQ(cache.topology_generation, std::uint64_t{3});
  EXPECT_EQ(cache.state_generation, std::uint64_t{7});
}
