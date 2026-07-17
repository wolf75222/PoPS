#include <gtest/gtest.h>

#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>

#include "component_abi_test_helpers.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

namespace abi = pops::component::test_support;

constexpr double kVelocity = 0.75;

std::size_t periodic(std::ptrdiff_t index, std::size_t size) {
  const auto extent = static_cast<std::ptrdiff_t>(size);
  const auto wrapped = (index % extent + extent) % extent;
  return static_cast<std::size_t>(wrapped);
}

double minmod(double left, double right) {
  if (left * right <= 0.0)
    return 0.0;
  return std::copysign(std::min(std::abs(left), std::abs(right)), left);
}

PopsComponentTableHeaderV1 flux_header() {
  return {sizeof(PopsNumericalFluxApiV1),
          POPS_COMPONENT_PROTOCOL_ABI_V1,
          POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1,
          1,
          nullptr,
          nullptr};
}

PopsNumericalFluxApiV1 flux_api() {
  return {flux_header(),
          +[](void*, const PopsNumericalFluxRequestV1* request, PopsNumericalFluxResultV1* result) {
            const auto* left_values = static_cast<const double*>(request->left.data);
            const auto* right_values = static_cast<const double*>(request->right.data);
            auto* output_values = static_cast<double*>(result->normal_flux.data);
            for (std::size_t point = 0; point < pops::component::field_point_count(request->left);
                 ++point) {
              const double left = left_values[point];
              const double right = right_values[point];
              output_values[point] =
                  0.5 * kVelocity * (left + right) - 0.5 * std::abs(kVelocity) * (right - left);
              result->stability_bounds[point] = std::abs(kVelocity);
              result->actions[point] = POPS_COMPONENT_CONTINUE_V1;
            }
            result->status = {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1,
                              nullptr};
            return 0;
          }};
}

std::vector<double> evaluate_flux(const std::vector<double>& left,
                                  const std::vector<double>& right) {
  const std::size_t faces = left.size();
  std::vector<double> output(faces);
  std::vector<double> stability(faces);
  std::vector<PopsComponentActionV1> actions(faces);
  std::vector<double> normals(faces * 2, 0.0);
  for (std::size_t point = 0; point < faces; ++point)
    normals[2 * point] = 1.0;
  const PopsNumericalFluxRequestV1 request{sizeof(PopsNumericalFluxRequestV1),
                                           abi::const_field_view(left.data(), 1, faces),
                                           abi::const_field_view(right.data(), 1, faces),
                                           abi::const_field_view(normals.data(), 1, faces, 2),
                                           nullptr,
                                           abi::logical_time(),
                                           abi::host_execution_context()};
  PopsNumericalFluxResultV1 result{sizeof(PopsNumericalFluxResultV1),
                                   abi::field_view(output.data(), 1, faces),
                                   stability.data(),
                                   actions.data(),
                                   {}};
  const auto api = flux_api();
  EXPECT_EQ(pops::component::evaluate_faces(api, nullptr, request, result), 0);
  return output;
}

TEST(test_amr_weno5_native, CoreWenoStatesFeedExactExternalFluxTableWithoutMarshallingFallback) {
  constexpr std::size_t cells = 64;
  std::vector<double> state(cells);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const double x = (static_cast<double>(cell) + 0.5) / cells;
    state[cell] = 1.0 + 0.2 * std::sin(2.0 * std::acos(-1.0) * x) +
                  0.05 * std::sin(6.0 * std::acos(-1.0) * x);
  }

  std::vector<double> weno_left(cells), weno_right(cells);
  std::vector<double> minmod_left(cells), minmod_right(cells);
  const auto value = [&](std::ptrdiff_t cell) { return state[periodic(cell, cells)]; };
  for (std::ptrdiff_t cell = 0; cell < static_cast<std::ptrdiff_t>(cells); ++cell) {
    weno_left[static_cast<std::size_t>(cell)] = pops::weno5z(
        value(cell - 2), value(cell - 1), value(cell), value(cell + 1), value(cell + 2));
    weno_right[static_cast<std::size_t>(cell)] = pops::weno5z(
        value(cell + 3), value(cell + 2), value(cell + 1), value(cell), value(cell - 1));
    minmod_left[static_cast<std::size_t>(cell)] =
        value(cell) + 0.5 * minmod(value(cell) - value(cell - 1), value(cell + 1) - value(cell));
    minmod_right[static_cast<std::size_t>(cell)] =
        value(cell + 1) -
        0.5 * minmod(value(cell + 1) - value(cell), value(cell + 2) - value(cell + 1));
  }

  const auto weno_flux = evaluate_flux(weno_left, weno_right);
  const auto minmod_flux = evaluate_flux(minmod_left, minmod_right);
  double difference = 0.0;
  for (std::size_t face = 0; face < cells; ++face) {
    EXPECT_DOUBLE_EQ(weno_flux[face], kVelocity * weno_left[face]);
    difference = std::max(difference, std::abs(weno_flux[face] - minmod_flux[face]));
  }
  EXPECT_GT(difference, 1e-8);
  EXPECT_EQ(pops::Weno5::n_ghost, 3);
}

TEST(test_amr_weno5_native, ConstantStateRemainsExactlyConstantAcrossWenoAndFinalFluxAbi) {
  constexpr std::size_t cells = 16;
  const std::vector<double> state(cells, 2.5);
  std::vector<double> left(cells), right(cells);
  const auto value = [&](std::ptrdiff_t cell) { return state[periodic(cell, cells)]; };
  for (std::ptrdiff_t cell = 0; cell < static_cast<std::ptrdiff_t>(cells); ++cell) {
    left[static_cast<std::size_t>(cell)] = pops::weno5z(
        value(cell - 2), value(cell - 1), value(cell), value(cell + 1), value(cell + 2));
    right[static_cast<std::size_t>(cell)] = pops::weno5z(
        value(cell + 3), value(cell + 2), value(cell + 1), value(cell), value(cell - 1));
  }
  for (const double value_at_face : evaluate_flux(left, right))
    EXPECT_DOUBLE_EQ(value_at_face, kVelocity * 2.5);
}

}  // namespace
