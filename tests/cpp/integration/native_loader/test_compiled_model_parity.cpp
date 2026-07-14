#include <gtest/gtest.h>

#include <pops/runtime/dynamic/component_consumers.hpp>

#include "component_abi_test_helpers.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace {

namespace abi = pops::component::test_support;

PopsComponentTableHeaderV1 header(std::size_t size) {
  return {static_cast<std::uint32_t>(size), POPS_COMPONENT_PROTOCOL_ABI_V1,
          POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, nullptr, nullptr};
}

double reference_flux(double left, double right) {
  constexpr double speed = 1.5;
  return 0.5 * (left + right) - 0.5 * speed * (right - left);
}

TEST(test_compiled_model_parity, ExactFluxTableMatchesCoreReferenceForWholeBatch) {
  constexpr std::size_t faces = 4;
  constexpr std::size_t components = 2;
  const std::array<double, faces * components> left{
      1.0, 2.0, 1.5, -1.0, 0.25, 3.0, 4.0, 0.5};
  const std::array<double, faces * components> right{
      0.5, 4.0, 2.0, -2.0, 1.25, 2.5, 3.0, 1.5};
  const std::array<double, faces * 2> normals{1.0, 0.0, 0.0, 1.0,
                                               -1.0, 0.0, 0.0, -1.0};
  std::array<double, faces * components> output{};
  std::array<double, faces> stability{};
  std::array<PopsComponentActionV1, faces> actions{};

  const PopsNumericalFluxApiV1 api{
      header(sizeof(PopsNumericalFluxApiV1)),
      +[](void*, const PopsNumericalFluxRequestV1* request,
          PopsNumericalFluxResultV1* result) {
        const auto* left_values = static_cast<const double*>(request->left.data);
        const auto* right_values = static_cast<const double*>(request->right.data);
        auto* output_values = static_cast<double*>(result->normal_flux.data);
        for (std::size_t point = 0; point < faces; ++point) {
          for (std::size_t component = 0; component < request->left.component_count; ++component) {
            const auto index = point * static_cast<std::size_t>(request->left.axis_strides[1]) +
                               component *
                                   static_cast<std::size_t>(request->left.component_stride);
            output_values[index] = reference_flux(left_values[index], right_values[index]);
          }
          result->stability_bounds[point] = 1.5;
          result->actions[point] = POPS_COMPONENT_CONTINUE_V1;
        }
        result->status = {sizeof(PopsComponentStatusV1), 0,
                          POPS_COMPONENT_CONTINUE_V1, nullptr};
        return 0;
      }};

  const PopsNumericalFluxRequestV1 request{
      sizeof(PopsNumericalFluxRequestV1),
      abi::const_field_view(left.data(), 1, faces, components),
      abi::const_field_view(right.data(), 1, faces, components),
      abi::const_field_view(normals.data(), 1, faces, 2),
      nullptr,
      abi::logical_time(), abi::host_execution_context()};
  PopsNumericalFluxResultV1 result{
      sizeof(PopsNumericalFluxResultV1),
      abi::field_view(output.data(), 1, faces, components),
      stability.data(), actions.data(), {}};

  ASSERT_EQ(pops::component::evaluate_faces(api, nullptr, request, result), 0);
  for (std::size_t index = 0; index < output.size(); ++index)
    EXPECT_DOUBLE_EQ(output[index], reference_flux(left[index], right[index]));
  for (std::size_t point = 0; point < faces; ++point) {
    EXPECT_DOUBLE_EQ(stability[point], 1.5);
    EXPECT_EQ(actions[point], POPS_COMPONENT_CONTINUE_V1);
  }
  EXPECT_EQ(result.status.action, POPS_COMPONENT_CONTINUE_V1);
}

TEST(test_compiled_model_parity, MissingHotOperationIsRejectedBeforeExecution) {
  const PopsNumericalFluxApiV1 api{header(sizeof(PopsNumericalFluxApiV1)), nullptr};
  PopsNumericalFluxRequestV1 request{};
  PopsNumericalFluxResultV1 result{};
  EXPECT_THROW(pops::component::evaluate_faces(api, nullptr, request, result),
               std::runtime_error);
}

}  // namespace
