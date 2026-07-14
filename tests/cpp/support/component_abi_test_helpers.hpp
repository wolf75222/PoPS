#pragma once

#include <pops/runtime/config/generated_component_abi.hpp>

#include <cstddef>

namespace pops::component::test_support {

inline PopsExecutionContextV1 host_execution_context() {
  return {
      sizeof(PopsExecutionContextV1), 1, "test::execution-context",
      POPS_MEMORY_SPACE_HOST_V1,
      "test::backend", "test::cpu:0", POPS_SCALAR_FLOAT64_V1,
      POPS_PRECISION_FLOAT64_V1, POPS_PRECISION_FLOAT64_V1,
      POPS_PRECISION_FLOAT64_V1, POPS_PRECISION_FLOAT64_V1,
      0, "test::host-synchronous", 0, 0, "serial", "none"};
}

inline PopsLogicalTimeV1 logical_time() {
  return {sizeof(PopsLogicalTimeV1), "test::clock", 7, 0, 0, 0,
          1, 1, 0.01, 0.25};
}

inline PopsConstFieldViewV1 const_field_view(
    const double* data, std::size_t ny, std::size_t nx,
    std::size_t components = 1, const char* layout_identity = "test::layout",
    const char* patch_identity = "test::patch") {
  return {sizeof(PopsConstFieldViewV1), data, 2, {ny, nx, 1},
          {static_cast<std::ptrdiff_t>(nx * components),
           static_cast<std::ptrdiff_t>(components), 0},
          components, 1, POPS_FIELD_CENTERING_CELL_V1, 0, {0, 0, 0},
          {0, 0, 0}, POPS_SCALAR_FLOAT64_V1, POPS_MEMORY_SPACE_HOST_V1,
          layout_identity, patch_identity,
          POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
}

inline PopsFieldViewV1 field_view(
    double* data, std::size_t ny, std::size_t nx,
    std::size_t components = 1, const char* layout_identity = "test::layout",
    const char* patch_identity = "test::patch") {
  return {sizeof(PopsFieldViewV1), data, 2, {ny, nx, 1},
          {static_cast<std::ptrdiff_t>(nx * components),
           static_cast<std::ptrdiff_t>(components), 0},
          components, 1, POPS_FIELD_CENTERING_CELL_V1, 0, {0, 0, 0},
          {0, 0, 0}, POPS_SCALAR_FLOAT64_V1, POPS_MEMORY_SPACE_HOST_V1,
          layout_identity, patch_identity,
          POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
}

}  // namespace pops::component::test_support
