#include <pops/runtime/system/prepared_field_solver_component.hpp>
#include <gtest/gtest.h>
#include <limits>

namespace detail = pops::runtime::field::field_solver_component_detail;

namespace {
PopsExecutionContextV1 host_execution() {
  return {sizeof(PopsExecutionContextV1), 1, "test-native-host-authority",
          POPS_MEMORY_SPACE_HOST_V1, "OpenMP", "host", POPS_SCALAR_FLOAT64_V1,
          POPS_PRECISION_FLOAT64_V1, POPS_PRECISION_FLOAT64_V1,
          POPS_PRECISION_FLOAT64_V1, POPS_PRECISION_FLOAT64_V1,
          0, "default", 0, 0, "serial", "none"};
}
PopsFieldViewV1 view(double* values, int dimension) {
  PopsFieldViewV1 v{};
  v.struct_size = sizeof(v); v.data = values; v.dimension = dimension;
  for (int axis = 0; axis < 3; ++axis) { v.extents[axis] = 1; v.axis_strides[axis] = axis < dimension ? 1 : 0; }
  v.extents[0] = 2; v.axis_strides[0] = 1;
  v.component_count = 1; v.component_stride = 64;
  v.centering = POPS_FIELD_CENTERING_CELL_V1; v.scalar_type = POPS_SCALAR_FLOAT64_V1;
  v.memory_space = POPS_MEMORY_SPACE_HOST_V1; v.layout_identity = "layout";
  v.patch_identity = "patch"; v.ownership = POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1;
  return v;
}
struct CountingExecution {
  static inline int fences = 0;
  void fence() const { ++fences; }
};
}
TEST(ExternalFieldBackend, ActualHostResidenceAdmissionRejectsRelabelledDevice) {
  using Memory = Kokkos::HostSpace;
  auto e = host_execution();
  EXPECT_NO_THROW((detail::validate_storage_execution<Memory>(e)));
  e.memory_space = POPS_MEMORY_SPACE_DEVICE_V1;
  EXPECT_THROW((detail::validate_storage_execution<Memory>(e)), std::invalid_argument);
  e = host_execution(); e.device_identity = "foreign-device";
  EXPECT_THROW((detail::validate_storage_execution<Memory>(e)), std::invalid_argument);
}
TEST(ExternalFieldBackend, ActiveFiniteCheckPreservesPaddedStrideAndIgnoresInactiveCells) {
  double values[12] = {1, 2, 0, 0, 3, std::numeric_limits<double>::infinity()};
  auto v = view(values, 2); v.extents[1] = 2; v.axis_strides[1] = 4;
  const std::vector<std::uint8_t> mask{1, 1, 1, 0};
  EXPECT_TRUE((detail::active_patch_is_finite<2, Kokkos::HostSpace>(v, mask)));
  EXPECT_FALSE((detail::active_patch_is_finite<2, Kokkos::HostSpace>(v, {1,1,1,1})));
  EXPECT_FALSE((detail::active_patch_is_finite<2, Kokkos::HostSpace>(v, {1,1,1,2})));
  EXPECT_EQ(values[5], std::numeric_limits<double>::infinity());
}
TEST(ExternalFieldBackend, FiniteCheckRefusesForeignDimensionMemoryMaskAndStride) {
  double values[2] = {1, 2}; auto v = view(values, 1);
  EXPECT_THROW((detail::active_patch_is_finite<3, Kokkos::HostSpace>(v, {1,1})), std::invalid_argument);
  EXPECT_THROW((detail::active_patch_is_finite<1, Kokkos::HostSpace>(v, {1})), std::invalid_argument);
  v.memory_space = POPS_MEMORY_SPACE_DEVICE_V1;
  EXPECT_THROW((detail::active_patch_is_finite<1, Kokkos::HostSpace>(v, {1,1})), std::invalid_argument);
  v.memory_space = POPS_MEMORY_SPACE_HOST_V1; v.scalar_type = POPS_SCALAR_FLOAT32_V1;
  EXPECT_THROW((detail::active_patch_is_finite<1, Kokkos::HostSpace>(v, {1,1})), std::invalid_argument);
  v.scalar_type = POPS_SCALAR_FLOAT64_V1; v.axis_strides[0] = -1;
  EXPECT_THROW((detail::active_patch_is_finite<1, Kokkos::HostSpace>(v, {1,1})), std::invalid_argument);
}
TEST(ExternalFieldBackend, ExactThreeDimensionalScalarStorageIsInspectedWithoutWrites) {
  double values[8] = {0.125,0.25,0.375,0.5,0.625,0.75,0.875,1};
  auto v = view(values, 3); v.extents[1] = v.extents[2] = 2;
  v.axis_strides[1] = 2; v.axis_strides[2] = 4;
  EXPECT_TRUE((detail::active_patch_is_finite<3,Kokkos::HostSpace>(v, {1,1,1,1,1,1,1,1})));
  EXPECT_EQ(values[7], 1);
}
TEST(ExternalFieldBackend, CallbackQueueIsJoinedOnSuccessAndExceptionUnwinding) {
  pops::detail::ensure_kokkos_initialized();
  CountingExecution::fences = 0;
  {
    detail::CallbackCompletion<CountingExecution> completion(host_execution());
    completion.wait();
  }
  EXPECT_EQ(CountingExecution::fences, 1);
  EXPECT_THROW({
    detail::CallbackCompletion<CountingExecution> completion(host_execution());
    throw std::runtime_error("component enqueued work then threw");
  }, std::runtime_error);
  EXPECT_EQ(CountingExecution::fences, 2);
}
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
namespace {
template <class ExecutionSpace>
struct NativeStreamOwner {
  std::uint64_t handle = 0;
  NativeStreamOwner() {
#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>) {
      cudaStream_t stream{};
      if (cudaStreamCreate(&stream) != cudaSuccess) throw std::runtime_error("test CUDA stream");
      handle = reinterpret_cast<std::uintptr_t>(stream);
    }
#endif
#if defined(KOKKOS_ENABLE_HIP)
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>) {
      hipStream_t stream{};
      if (hipStreamCreate(&stream) != hipSuccess) throw std::runtime_error("test HIP stream");
      handle = reinterpret_cast<std::uintptr_t>(stream);
    }
#endif
  }
  ~NativeStreamOwner() {
#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>)
      if (handle) (void)cudaStreamDestroy(reinterpret_cast<cudaStream_t>(handle));
#endif
#if defined(KOKKOS_ENABLE_HIP)
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>)
      if (handle) (void)hipStreamDestroy(reinterpret_cast<hipStream_t>(handle));
#endif
  }
};
template <class Memory>
void inspect_actual_native_queue() {
  using Execution = Kokkos::DefaultExecutionSpace;
  pops::detail::ensure_kokkos_initialized();
  NativeStreamOwner<Execution> owner;
  auto e = host_execution();
  e.backend_identity = Execution::name();
  e.device_identity = detail::execution_device<Execution>().data();
  e.memory_space = detail::memory_kind<Memory>();
  e.stream_handle = owner.handle; e.stream_identity = "test-owned-native-queue";
  EXPECT_NO_THROW((detail::validate_storage_execution<Memory>(e)));
  auto foreign = e; foreign.backend_identity = "foreign-backend";
  EXPECT_THROW((detail::validate_storage_execution<Memory>(foreign)), std::invalid_argument);
  const Kokkos::View<double*, Memory> storage("external_field_queue_test", 2);
  const auto submit = [&](double value, bool throws) {
    detail::CallbackCompletion completion(e);
    const auto instance = detail::execution_instance<Execution>(e);
    Kokkos::parallel_for("external_field_callback", Kokkos::RangePolicy<Execution>(instance,0,2),
                        KOKKOS_LAMBDA(int i) { storage(i) = value; });
    if (throws) throw std::runtime_error("callback threw after enqueue");
    completion.wait();
  };
  auto v = view(storage.data(),1); v.memory_space = e.memory_space;
  submit(0.25,false);
  EXPECT_TRUE((detail::active_patch_is_finite<1,Memory>(v,{1,1})));
  EXPECT_THROW(submit(1.25,true), std::runtime_error);
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},storage);
  EXPECT_EQ(host(0),1.25); EXPECT_EQ(host(1),1.25);
  submit(std::numeric_limits<double>::infinity(),false);
  EXPECT_FALSE((detail::active_patch_is_finite<1,Memory>(v,{1,1})));
  EXPECT_TRUE((detail::active_patch_is_finite<1,Memory>(v,{0,0})));
}
}
TEST(ExternalFieldBackend, ActualDeviceStorageIsMirroredOnlyForInspection) {
  using Memory = typename Kokkos::DefaultExecutionSpace::memory_space;
  if constexpr (!detail::borrows_native_stream<Kokkos::DefaultExecutionSpace>)
    GTEST_SKIP() << "CUDA/HIP macro does not select an actual CUDA/HIP execution type";
  pops::detail::ensure_kokkos_initialized();
  const Kokkos::View<double*, Memory> storage("external_field_device_test", 2);
  Kokkos::parallel_for("field_backend_values",Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,2),
                      KOKKOS_LAMBDA(int i) { storage(i) = i + 0.25; });
  Kokkos::fence();
  auto v = view(storage.data(),1); v.memory_space = detail::memory_kind<Memory>();
  EXPECT_TRUE((detail::active_patch_is_finite<1,Memory>(v,{1,1})));
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, storage);
  EXPECT_EQ(host(0),0.25); EXPECT_EQ(host(1),1.25);
}
TEST(ExternalFieldBackend, ActualDeviceQueueCompletesBeforeInspectionAndUnwinding) {
  if constexpr (!detail::borrows_native_stream<Kokkos::DefaultExecutionSpace>)
    GTEST_SKIP() << "requires actual default CUDA/HIP execution type";
  inspect_actual_native_queue<typename Kokkos::DefaultExecutionSpace::memory_space>();
}
TEST(ExternalFieldBackend, ActualManagedQueueCompletesBeforeInspectionAndUnwinding) {
  if constexpr (!detail::borrows_native_stream<Kokkos::DefaultExecutionSpace> ||
                std::is_same_v<Kokkos::SharedSpace,Kokkos::HostSpace>)
    GTEST_SKIP() << "requires actual CUDA/HIP SharedSpace";
  inspect_actual_native_queue<Kokkos::SharedSpace>();
}
#endif
