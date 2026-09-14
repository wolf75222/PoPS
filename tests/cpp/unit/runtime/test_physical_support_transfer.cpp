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

}  // namespace
