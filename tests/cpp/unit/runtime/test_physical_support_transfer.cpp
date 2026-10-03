#include <pops/runtime/dynamic/physical_support_transfer.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using pops::component::PhysicalSupportTransfer;
using pops::component::apply_physical_support_transfer;

constexpr double kCanary = -991.0;

void ensure_kokkos() {
  static Kokkos::ScopeGuard guard;
}

template <class View, class Pointer>
View field_view(Pointer data, int dimension, std::array<std::size_t, 3> extents,
                std::array<std::ptrdiff_t, 3> strides, std::size_t components,
                std::ptrdiff_t component_stride) {
  View view{};
  view.struct_size = sizeof(View);
  view.data = data;
  view.dimension = dimension;
  for (int axis = 0; axis < 3; ++axis) {
    view.extents[axis] = extents[axis];
    view.axis_strides[axis] = strides[axis];
  }
  view.component_count = components;
  view.component_stride = component_stride;
  view.centering = POPS_FIELD_CENTERING_CELL_V1;
  view.scalar_type = POPS_SCALAR_FLOAT64_V1;
  view.memory_space = POPS_MEMORY_SPACE_HOST_V1;
  view.ownership = POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1;
  return view;
}

PopsTransferRequestV1 request_for(const PhysicalSupportTransfer& map, PopsConstFieldViewV1 source,
                                  PopsFieldViewV1 destination) {
  static const std::int32_t identity_ratios[] = {1, 1, 1};
  PopsTransferRequestV1 request{};
  request.struct_size = sizeof(request);
  request.source = source;
  request.destination = destination;
  request.refinement_ratio = identity_ratios;
  request.dimension = map.dimension;
  request.operation = static_cast<PopsTransferOperationV1>(map.operation);
  return request;
}

void expect_canaries(const std::vector<double>& values, const std::vector<bool>& written) {
  ASSERT_EQ(values.size(), written.size());
  for (std::size_t index = 0; index < values.size(); ++index)
    if (!written[index])
      EXPECT_DOUBLE_EQ(values[index], kCanary) << "padding index " << index;
}

TEST(PhysicalSupportTransfer, SignedAxisZeroReductionPreservesComponentsAndPaddedStrides) {
  ensure_kokkos();
  const double weights[] = {-2, 1, 3};
  PhysicalSupportTransfer map{};
  map.dimension = 2;
  map.operation = POPS_TRANSFER_OPERATION_VELOCITY_MOMENT_V1;
  map.source_active[0] = map.source_active[1] = map.target_active[0] = 1;
  map.source_to_target[1] = 0;
  map.reduction_cells[0] = 3;
  map.weights = weights;
  map.weight_count = 3;
  std::vector<double> source(96, kCanary), destination(32, kCanary);
  std::vector<bool> written(destination.size(), false);
  for (std::size_t component = 0; component < 2; ++component)
    for (std::size_t retained = 0; retained < 2; ++retained)
      for (std::size_t reduced = 0; reduced < 3; ++reduced)
        source[61 * component + 17 * retained + 3 * reduced] =
            100 * component + 10 * retained + reduced * reduced;
  auto request = request_for(
      map, field_view<PopsConstFieldViewV1>(source.data(), 2, {3, 2, 1}, {3, 17, 1}, 2, 61),
      field_view<PopsFieldViewV1>(destination.data(), 2, {2, 1, 1}, {4, 19, 1}, 2, 23));
  PopsComponentStatusV1 status{};
  ASSERT_EQ(apply_physical_support_transfer(map, &request, &status), 0);
  EXPECT_EQ(status.action, POPS_COMPONENT_CONTINUE_V1);
  for (std::size_t component = 0; component < 2; ++component)
    for (std::size_t retained = 0; retained < 2; ++retained) {
      const std::size_t offset = 23 * component + 4 * retained;
      // Sum(weights)=2 and Sum(weights*x*x)=13, independent of kernel traversal order.
      EXPECT_DOUBLE_EQ(destination[offset], 200 * component + 20 * retained + 13);
      written[offset] = true;
    }
  expect_canaries(destination, written);
}

TEST(PhysicalSupportTransfer, ThreeDimensionalTensorReductionPermutesTheRetainedAxis) {
  ensure_kokkos();
  const double weights[] = {-1, 2, 1, -2, 4};
  PhysicalSupportTransfer map{};
  map.dimension = 3;
  map.operation = POPS_TRANSFER_OPERATION_VELOCITY_MOMENT_V1;
  map.source_active[0] = map.source_active[1] = map.source_active[2] = 1;
  map.target_active[0] = 1;
  map.source_to_target[1] = 0;
  map.reduction_cells[0] = 2;
  map.reduction_cells[2] = 3;
  map.weight_offsets[2] = 2;
  map.weights = weights;
  map.weight_count = 5;
  std::vector<double> source(280, kCanary), destination(24, kCanary);
  std::vector<bool> written(destination.size(), false);
  for (std::size_t component = 0; component < 2; ++component)
    for (std::size_t retained = 0; retained < 2; ++retained)
      for (std::size_t x = 0; x < 2; ++x)
        for (std::size_t z = 0; z < 3; ++z)
          source[151 * component + 19 * retained + 5 * x + 43 * z] =
              100 * component + 10 * retained + x + z + x * z;
  auto request = request_for(
      map, field_view<PopsConstFieldViewV1>(source.data(), 3, {2, 2, 3}, {5, 19, 43}, 2, 151),
      field_view<PopsFieldViewV1>(destination.data(), 3, {2, 1, 1}, {3, 17, 29}, 2, 13));
  PopsComponentStatusV1 status{};
  ASSERT_EQ(apply_physical_support_transfer(map, &request, &status), 0);
  for (std::size_t component = 0; component < 2; ++component)
    for (std::size_t retained = 0; retained < 2; ++retained) {
      const std::size_t offset = 13 * component + 3 * retained;
      // Tensor moments: sum_x=1, first_x=2, sum_z=3, first_z=6.
      EXPECT_DOUBLE_EQ(destination[offset], 300 * component + 30 * retained + 24);
      written[offset] = true;
    }
  expect_canaries(destination, written);
}

TEST(PhysicalSupportTransfer, ThreeDimensionalPullbackPermutesAndBroadcastsTwoAxes) {
  ensure_kokkos();
  PhysicalSupportTransfer map{};
  map.dimension = 3;
  map.operation = POPS_TRANSFER_OPERATION_PHYSICAL_PULLBACK_V1;
  map.source_active[1] = 1;
  map.target_active[0] = map.target_active[1] = map.target_active[2] = 1;
  map.source_to_target[1] = 2;
  std::vector<double> source(48, kCanary), destination(180, kCanary);
  std::vector<bool> written(destination.size(), false);
  for (std::size_t component = 0; component < 2; ++component) {
    source[31 * component] = -3.0 + 100 * component;
    source[31 * component + 11] = 7.0 + 100 * component;
  }
  auto request = request_for(
      map, field_view<PopsConstFieldViewV1>(source.data(), 3, {1, 2, 1}, {7, 11, 23}, 2, 31),
      field_view<PopsFieldViewV1>(destination.data(), 3, {2, 3, 2}, {3, 11, 41}, 2, 103));
  PopsComponentStatusV1 status{};
  ASSERT_EQ(apply_physical_support_transfer(map, &request, &status), 0);
  for (std::size_t component = 0; component < 2; ++component)
    for (std::size_t x = 0; x < 2; ++x)
      for (std::size_t y = 0; y < 3; ++y)
        for (std::size_t retained = 0; retained < 2; ++retained) {
          const std::size_t offset = 103 * component + 3 * x + 11 * y + 41 * retained;
          EXPECT_DOUBLE_EQ(destination[offset], (retained == 0 ? -3.0 : 7.0) + 100 * component);
          written[offset] = true;
        }
  expect_canaries(destination, written);
}

TEST(PhysicalSupportTransfer, OneDimensionalScalarReductionAndExtension) {
  ensure_kokkos();
  const double weights[] = {-1, 0, 2, 3};
  PhysicalSupportTransfer reduction{};
  reduction.dimension = 1;
  reduction.operation = POPS_TRANSFER_OPERATION_VELOCITY_MOMENT_V1;
  reduction.source_active[0] = 1;
  reduction.reduction_cells[0] = 4;
  reduction.weights = weights;
  reduction.weight_count = 4;
  std::vector<double> source(32, kCanary), scalar(10, kCanary), extension(32, kCanary);
  for (std::size_t component = 0; component < 2; ++component)
    for (std::size_t cell = 0; cell < 4; ++cell)
      source[17 * component + 3 * cell] = 4 - cell + 10 * component;
  auto request = request_for(
      reduction, field_view<PopsConstFieldViewV1>(source.data(), 1, {4, 1, 1}, {3, 1, 1}, 2, 17),
      field_view<PopsFieldViewV1>(scalar.data(), 1, {1, 1, 1}, {5, 1, 1}, 2, 7));
  PopsComponentStatusV1 status{};
  ASSERT_EQ(apply_physical_support_transfer(reduction, &request, &status), 0);
  EXPECT_DOUBLE_EQ(scalar[0], 3);
  EXPECT_DOUBLE_EQ(scalar[7], 43);
  PhysicalSupportTransfer broadcast{};
  broadcast.dimension = 1;
  broadcast.operation = POPS_TRANSFER_OPERATION_PHYSICAL_PULLBACK_V1;
  broadcast.target_active[0] = 1;
  request = request_for(
      broadcast, field_view<PopsConstFieldViewV1>(scalar.data(), 1, {1, 1, 1}, {5, 1, 1}, 2, 7),
      field_view<PopsFieldViewV1>(extension.data(), 1, {4, 1, 1}, {3, 1, 1}, 2, 19));
  ASSERT_EQ(apply_physical_support_transfer(broadcast, &request, &status), 0);
  std::vector<bool> written(extension.size(), false);
  for (std::size_t component = 0; component < 2; ++component)
    for (std::size_t cell = 0; cell < 4; ++cell) {
      const std::size_t offset = 19 * component + 3 * cell;
      EXPECT_DOUBLE_EQ(extension[offset], component == 0 ? 3.0 : 43.0);
      written[offset] = true;
    }
  expect_canaries(extension, written);
}

TEST(PhysicalSupportTransfer, OffsetOverflowFailsPreflightWithoutWritingDestination) {
  ensure_kokkos();
  std::array<double, 4> source{1, 2, 3, 4};
  std::vector<double> destination(8, kCanary);
  PhysicalSupportTransfer broadcast{};
  broadcast.dimension = 1;
  broadcast.operation = POPS_TRANSFER_OPERATION_PHYSICAL_PULLBACK_V1;
  broadcast.target_active[0] = 1;
  auto request = request_for(
      broadcast, field_view<PopsConstFieldViewV1>(source.data(), 1, {1, 1, 1}, {1, 1, 1}, 1, 1),
      field_view<PopsFieldViewV1>(destination.data(), 1, {2, 1, 1},
                                  {std::numeric_limits<std::ptrdiff_t>::max(), 1, 1}, 1, 1));
  PopsComponentStatusV1 status{};
  EXPECT_NE(apply_physical_support_transfer(broadcast, &request, &status), 0);
  EXPECT_EQ(status.struct_size, sizeof(PopsComponentStatusV1));
  EXPECT_EQ(status.code, 3);
  EXPECT_EQ(status.action, POPS_COMPONENT_ABORT_RUN_V1);
  EXPECT_NE(status.reason, nullptr);
  EXPECT_TRUE(std::all_of(destination.begin(), destination.end(),
                          [](double value) { return value == kCanary; }));

  const double weights[] = {1, 1};
  PhysicalSupportTransfer reduction{};
  reduction.dimension = 1;
  reduction.operation = POPS_TRANSFER_OPERATION_VELOCITY_MOMENT_V1;
  reduction.source_active[0] = 1;
  reduction.reduction_cells[0] = 2;
  reduction.weights = weights;
  reduction.weight_count = 2;
  request = request_for(
      reduction,
      field_view<PopsConstFieldViewV1>(source.data(), 1, {2, 1, 1},
                                       {std::numeric_limits<std::ptrdiff_t>::max(), 1, 1}, 1, 1),
      field_view<PopsFieldViewV1>(destination.data(), 1, {1, 1, 1}, {1, 1, 1}, 1, 1));
  EXPECT_NE(apply_physical_support_transfer(reduction, &request, &status), 0);
  EXPECT_TRUE(std::all_of(destination.begin(), destination.end(),
                          [](double value) { return value == kCanary; }));

  request =
      request_for(broadcast,
                  field_view<PopsConstFieldViewV1>(source.data(), 1, {1, 1, 1}, {1, 1, 1}, 2,
                                                   std::numeric_limits<std::ptrdiff_t>::max()),
                  field_view<PopsFieldViewV1>(destination.data(), 1, {2, 1, 1}, {1, 1, 1}, 2, 3));
  EXPECT_NE(apply_physical_support_transfer(broadcast, &request, &status), 0);
  EXPECT_TRUE(std::all_of(destination.begin(), destination.end(),
                          [](double value) { return value == kCanary; }));
}

TEST(PhysicalSupportTransfer, FailureStatusDistinguishesInputAndNumericalFailureThenClears) {
  ensure_kokkos();
  std::array<double, 2> source{1e308, 1e308};
  std::array<double, 1> destination{kCanary};
  const double weights[] = {1, 1};
  PhysicalSupportTransfer reduction{};
  reduction.dimension = 1;
  reduction.operation = POPS_TRANSFER_OPERATION_VELOCITY_MOMENT_V1;
  reduction.source_active[0] = 1;
  reduction.reduction_cells[0] = 2;
  reduction.weights = weights;
  reduction.weight_count = 2;
  auto request = request_for(
      reduction, field_view<PopsConstFieldViewV1>(source.data(), 1, {2, 1, 1}, {1, 1, 1}, 1, 2),
      field_view<PopsFieldViewV1>(destination.data(), 1, {1, 1, 1}, {1, 1, 1}, 1, 1));
  PopsComponentStatusV1 status{};
  request.dimension = 2;
  EXPECT_EQ(apply_physical_support_transfer(reduction, &request, &status), 2);
  EXPECT_EQ(status.code, 2);
  EXPECT_EQ(status.action, POPS_COMPONENT_ABORT_RUN_V1);
  EXPECT_DOUBLE_EQ(destination[0], kCanary);
  request.dimension = 1;
  EXPECT_EQ(apply_physical_support_transfer(reduction, &request, &status), 4);
  EXPECT_EQ(status.code, 4);
  EXPECT_EQ(status.action, POPS_COMPONENT_ABORT_RUN_V1);
  EXPECT_NE(status.reason, nullptr);
  source = {1, 2};
  ASSERT_EQ(apply_physical_support_transfer(reduction, &request, &status), 0);
  EXPECT_EQ(status.code, 0);
  EXPECT_EQ(status.action, POPS_COMPONENT_CONTINUE_V1);
  EXPECT_EQ(status.reason, nullptr);
  EXPECT_DOUBLE_EQ(destination[0], 3);
}

// The finite 4x3 witness is one member of this generic product family. Every component
// occupies its own spatial product; none of the velocity cells is a packed component.
TEST(PhysicalSupportTransfer, ProductReduceThenLiftVariesFibresAndComponentWidths) {
  ensure_kokkos();
  for (const auto shape : {std::array<std::size_t, 3>{4, 3, 3}, {2, 5, 1}, {7, 3, 5}}) {
    const auto nx = shape[0], nv = shape[1], width = shape[2];
    std::vector<double> source(nx * nv * width), moment(nx * width, kCanary);
    std::vector<double> lifted(source.size(), kCanary), weights(nv);
    double zeroth = 0, first = 0;
    for (std::size_t j = 0; j < nv; ++j) {
      weights[j] = (j % 2 ? -1.0 : 1.0) * static_cast<double>(j + 1);
      zeroth += weights[j];
      first += weights[j] * static_cast<double>(j);
    }
    for (std::size_t c = 0; c < width; ++c)
      for (std::size_t x = 0; x < nx; ++x)
        for (std::size_t j = 0; j < nv; ++j)
          source[c * nx * nv + x * nv + j] =
              (c + 1) * (x + 1) + static_cast<double>((c + 2) * j) +
              (c % 2 ? -1.0 : 1.0) * static_cast<double>(x * j);
    PhysicalSupportTransfer reduction{};
    reduction.dimension = 2;
    reduction.operation = POPS_TRANSFER_OPERATION_VELOCITY_MOMENT_V1;
    reduction.source_active[0] = reduction.source_active[1] = reduction.target_active[0] = 1;
    reduction.source_to_target[1] = 0;
    reduction.reduction_cells[0] = nv;
    reduction.weights = weights.data();
    reduction.weight_count = nv;
    auto request = request_for(reduction,
        field_view<PopsConstFieldViewV1>(source.data(), 2, {nv, nx, 1},
                                       {1, static_cast<std::ptrdiff_t>(nv), 1}, width, nx * nv),
        field_view<PopsFieldViewV1>(moment.data(), 2, {nx, 1, 1}, {1, 1, 1}, width, nx));
    PopsComponentStatusV1 status{};
    ASSERT_EQ(apply_physical_support_transfer(reduction, &request, &status), 0);
    PhysicalSupportTransfer extension{};
    extension.dimension = 2;
    extension.operation = POPS_TRANSFER_OPERATION_PHYSICAL_PULLBACK_V1;
    extension.source_active[0] = extension.target_active[0] = extension.target_active[1] = 1;
    extension.source_to_target[0] = 1;
    request = request_for(extension,
        field_view<PopsConstFieldViewV1>(moment.data(), 2, {nx, 1, 1}, {1, 1, 1}, width, nx),
        field_view<PopsFieldViewV1>(lifted.data(), 2, {nv, nx, 1},
                                  {1, static_cast<std::ptrdiff_t>(nv), 1}, width, nx * nv));
    ASSERT_EQ(apply_physical_support_transfer(extension, &request, &status), 0);
    for (std::size_t c = 0; c < width; ++c)
      for (std::size_t x = 0; x < nx; ++x) {
        const double exact = (c + 1) * (x + 1) * zeroth +
            ((c + 2) + (c % 2 ? -1.0 : 1.0) * x) * first;
        EXPECT_DOUBLE_EQ(moment[c * nx + x], exact);
        for (std::size_t j = 0; j < nv; ++j)
          EXPECT_DOUBLE_EQ(lifted[c * nx * nv + x * nv + j], exact);
      }
    // Repeated lifting and integration scales by the authored sum of weights;
    // an inverse closure or hidden normalization would fail this identity.
    request = request_for(reduction,
        field_view<PopsConstFieldViewV1>(lifted.data(), 2, {nv, nx, 1},
                                       {1, static_cast<std::ptrdiff_t>(nv), 1}, width, nx * nv),
        field_view<PopsFieldViewV1>(moment.data(), 2, {nx, 1, 1}, {1, 1, 1}, width, nx));
    ASSERT_EQ(apply_physical_support_transfer(reduction, &request, &status), 0);
    for (std::size_t c = 0; c < width; ++c)
      for (std::size_t x = 0; x < nx; ++x)
        EXPECT_DOUBLE_EQ(moment[c * nx + x], lifted[c * nx * nv + x * nv] * zeroth);
  }
}

TEST(PhysicalSupportTransfer, ActiveNonfiniteCellIsNotHiddenByZeroMomentWeight) {
  ensure_kokkos();
  const double weights[] = {0, 2};
  std::array<double, 2> source{0, 3};
  std::array<double, 1> destination{kCanary};
  PhysicalSupportTransfer reduction{};
  reduction.dimension = 1;
  reduction.operation = POPS_TRANSFER_OPERATION_VELOCITY_MOMENT_V1;
  reduction.source_active[0] = 1;
  reduction.reduction_cells[0] = 2;
  reduction.weights = weights;
  reduction.weight_count = 2;
  auto request = request_for(reduction,
      field_view<PopsConstFieldViewV1>(source.data(), 1, {2, 1, 1}, {1, 1, 1}, 1, 2),
      field_view<PopsFieldViewV1>(destination.data(), 1, {1, 1, 1}, {1, 1, 1}, 1, 1));
  PopsComponentStatusV1 status{};
  for (const double invalid : {std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity()}) {
    source[0] = invalid;
    EXPECT_EQ(apply_physical_support_transfer(reduction, &request, &status), 4);
    EXPECT_EQ(status.action, POPS_COMPONENT_ABORT_RUN_V1);
    // Numerical failure is a scratch-candidate status; publication belongs to the
    // prepared System transaction, not this raw-view kernel seam.
    source[0] = 5;
    ASSERT_EQ(apply_physical_support_transfer(reduction, &request, &status), 0);
    EXPECT_DOUBLE_EQ(destination[0], 6);
    EXPECT_EQ(status.action, POPS_COMPONENT_CONTINUE_V1);
  }
}

}  // namespace

TEST(PhysicalSupportTransfer, OwnedBackendWeightsDoNotAliasAuthoredStorage) {
  ensure_kokkos();
  using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
  double authored[] = {1.0, -.5, .25};
  const ExecutionSpace execution{};
  pops::component::physical_transfer_detail::OwnedWeights<ExecutionSpace> owned(execution, authored, 3);
  ASSERT_NE(owned.data(), authored);
  authored[0] = 991.;
  authored[1] = 992.;
  EXPECT_DOUBLE_EQ(owned.data()[0], 1.);
  EXPECT_DOUBLE_EQ(owned.data()[1], -.5);
  EXPECT_DOUBLE_EQ(owned.data()[2], .25);
}

TEST(PhysicalSupportTransfer, DeviceAdmissionCannotRelabelHostMemoryOrUseMissingAuthority) {
  ensure_kokkos();
  double source[] = {1., 2., 3.}, destination[] = {kCanary};
  const double weights[] = {1., 1., 1.};
  PhysicalSupportTransfer map{};
  map.dimension = 1; map.operation = 2; map.source_active[0] = 1;
  map.reduction_cells[0] = 3; map.weights = weights; map.weight_count = 3;
  auto s = field_view<PopsConstFieldViewV1>(source, 1, {3,1,1}, {1,1,1}, 1, 3);
  auto d = field_view<PopsFieldViewV1>(destination, 1, {1,1,1}, {1,1,1}, 1, 1);
  s.memory_space = POPS_MEMORY_SPACE_DEVICE_V1;
  auto request = request_for(map, s, d);
  PopsComponentStatusV1 status{};
  EXPECT_EQ(apply_physical_support_transfer(map, &request, &status), 2);
  EXPECT_DOUBLE_EQ(destination[0], kCanary);
  d.memory_space = POPS_MEMORY_SPACE_DEVICE_V1;
  request.destination = d;
  // Both views labelled device is insufficient: the exact execution contract is absent.
  EXPECT_EQ(apply_physical_support_transfer(map, &request, &status), 2);
  EXPECT_DOUBLE_EQ(destination[0], kCanary);
}

TEST(PhysicalSupportTransfer, QuadratureRetainsSequentialAssociationForEveryComponent) {
  ensure_kokkos();
  const double weights[] = {1.e16, 1., -1.e16};
  const double source[] = {1., 1., 1., 1., 2., 1., .5, .5, .5};
  double destination[] = {kCanary, kCanary, kCanary};
  PhysicalSupportTransfer map{};
  map.dimension = 1; map.operation = 2; map.source_active[0] = 1;
  map.reduction_cells[0] = 3; map.weights = weights; map.weight_count = 3;
  const auto s = field_view<PopsConstFieldViewV1>(source, 1, {3,1,1}, {1,1,1}, 3, 3);
  const auto d = field_view<PopsFieldViewV1>(destination, 1, {1,1,1}, {1,1,1}, 3, 1);
  auto request = request_for(map, s, d); PopsComponentStatusV1 status{};
  ASSERT_EQ(apply_physical_support_transfer(map, &request, &status), 0);
  EXPECT_DOUBLE_EQ(destination[0], 0.);
  EXPECT_DOUBLE_EQ(destination[1], 2.);
  EXPECT_DOUBLE_EQ(destination[2], 0.);
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_OPENMPTARGET)
TEST(PhysicalSupportTransfer, ActualDeviceReductionUsesOwnedWeightsAndAllComponentStrides) {
  ensure_kokkos();
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;
  Kokkos::View<double*, MemorySpace> source("physical source", 18), destination("physical destination", 6);
  auto host = Kokkos::create_mirror_view(source);
  for (int c = 0; c < 3; ++c)
    for (int x = 0; x < 2; ++x)
      for (int eta = 0; eta < 3; ++eta)
        host(c*6+x*3+eta) = (c+1)*.5 + x + eta*.25;
  Kokkos::deep_copy(source, host);
  const double weights[] = {.25, -.5, 1.};
  PhysicalSupportTransfer map{};
  map.dimension = 2; map.operation = 2; map.source_active[0] = map.source_active[1] = 1;
  map.target_active[1] = 1; map.source_to_target[1] = 1;
  map.reduction_cells[0] = 3; map.weights = weights; map.weight_count = 3;
  auto s = field_view<PopsConstFieldViewV1>(source.data(), 2, {3,2,1}, {1,3,1}, 3, 6);
  auto d = field_view<PopsFieldViewV1>(destination.data(), 2, {1,2,1}, {1,1,1}, 3, 2);
  s.memory_space = d.memory_space = POPS_MEMORY_SPACE_DEVICE_V1;
  auto request = request_for(map, s, d);
  request.execution = {sizeof(PopsExecutionContextV1), 1, "actual-test-device-lane",
      POPS_MEMORY_SPACE_DEVICE_V1, "actual-test-compiled-backend", "actual-test-device",
      POPS_SCALAR_FLOAT64_V1, POPS_PRECISION_FLOAT64_V1, POPS_PRECISION_FLOAT64_V1,
      POPS_PRECISION_FLOAT64_V1, POPS_PRECISION_FLOAT64_V1, 0, "actual-test-default-stream",
      0, 0, "serial", "none"};
  PopsComponentStatusV1 status{};
  ASSERT_EQ(apply_physical_support_transfer(map, &request, &status), 0);
  auto output = Kokkos::create_mirror_view(destination); Kokkos::deep_copy(output, destination);
  for (int c = 0; c < 3; ++c)
    for (int x = 0; x < 2; ++x) {
      double expected = 0.;
      for (int eta = 0; eta < 3; ++eta) expected += host(c*6+x*3+eta)*weights[eta];
      EXPECT_DOUBLE_EQ(output(c*2+x), expected);
    }
}
#endif
