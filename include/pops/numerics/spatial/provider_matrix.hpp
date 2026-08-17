#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

/// @file
/// @brief Exact compile-time/runtime qualification matrix for native spatial providers.
///
/// A reusable numerical kernel may be dimension-generic while a concrete runtime remains 2D.
/// Likewise, a block may own an embedded-boundary residual without owning a metric-aware
/// characteristic ghost producer or boundary linearization.  This small value type records those
/// facts independently; callers must qualify the complete request and may never infer one
/// capability from another non-empty closure.

namespace pops {

enum class SpatialProviderGeometry : std::uint8_t {
  Cartesian = 0,
  Staircase = 1,
  CutCell = 2,
  Polar = 3,
};

enum class SpatialProviderOperation : std::uint8_t {
  Residual = 0,
  CharacteristicNoInflow = 1,
  BoundaryLinearization = 2,
};

enum class SpatialProviderRefusal : std::uint8_t {
  None = 0,
  UnsupportedDimension = 1,
  UnsupportedGeometry = 2,
  UnsupportedOperation = 3,
};

constexpr std::size_t spatial_geometry_index(SpatialProviderGeometry geometry) {
  return static_cast<std::size_t>(geometry);
}

constexpr std::uint8_t spatial_operation_flag(SpatialProviderOperation operation) {
  return static_cast<std::uint8_t>(1U << static_cast<unsigned>(operation));
}

constexpr bool valid_spatial_dimension(int dimension) {
  return dimension >= 1 && dimension <= 3;
}

constexpr std::size_t spatial_dimension_index(int dimension) {
  return static_cast<std::size_t>(dimension - 1);
}

struct SpatialProviderRequest {
  int dimension = 0;
  SpatialProviderGeometry geometry = SpatialProviderGeometry::Cartesian;
  SpatialProviderOperation operation = SpatialProviderOperation::Residual;
};

struct SpatialProviderCapabilities {
  static constexpr std::size_t dimension_count = 3;
  static constexpr std::size_t geometry_count = 4;

  std::array<std::array<std::uint8_t, geometry_count>, dimension_count> operations{};

  constexpr void enable(int dimension, SpatialProviderGeometry geometry,
                        SpatialProviderOperation operation) {
    if (!valid_spatial_dimension(dimension))
      return;
    auto& cell = operations[spatial_dimension_index(dimension)][spatial_geometry_index(geometry)];
    cell = static_cast<std::uint8_t>(cell | spatial_operation_flag(operation));
  }

  [[nodiscard]] constexpr bool supports_dimension(int dimension) const {
    if (!valid_spatial_dimension(dimension))
      return false;
    for (const std::uint8_t cell : operations[spatial_dimension_index(dimension)])
      if (cell != 0)
        return true;
    return false;
  }

  [[nodiscard]] constexpr bool supports_geometry(int dimension,
                                                 SpatialProviderGeometry geometry) const {
    return valid_spatial_dimension(dimension) &&
           operations[spatial_dimension_index(dimension)][spatial_geometry_index(geometry)] != 0;
  }

  [[nodiscard]] constexpr bool supports(const SpatialProviderRequest& request) const {
    return valid_spatial_dimension(request.dimension) &&
           (operations[spatial_dimension_index(request.dimension)]
                      [spatial_geometry_index(request.geometry)] &
            spatial_operation_flag(request.operation)) != 0;
  }
};

struct SpatialProviderQualification {
  bool executable = false;
  SpatialProviderRefusal refusal = SpatialProviderRefusal::UnsupportedDimension;
};

[[nodiscard]] constexpr SpatialProviderQualification qualify_spatial_provider(
    const SpatialProviderCapabilities& capabilities, const SpatialProviderRequest& request) {
  if (!capabilities.supports_dimension(request.dimension))
    return {false, SpatialProviderRefusal::UnsupportedDimension};
  if (!capabilities.supports_geometry(request.dimension, request.geometry))
    return {false, SpatialProviderRefusal::UnsupportedGeometry};
  if (!capabilities.supports(request))
    return {false, SpatialProviderRefusal::UnsupportedOperation};
  return {true, SpatialProviderRefusal::None};
}

[[nodiscard]] constexpr SpatialProviderCapabilities make_cartesian_spatial_provider(
    int dimension, bool characteristic_no_inflow = false, bool boundary_linearization = false) {
  SpatialProviderCapabilities capabilities;
  capabilities.enable(dimension, SpatialProviderGeometry::Cartesian,
                      SpatialProviderOperation::Residual);
  if (characteristic_no_inflow)
    capabilities.enable(dimension, SpatialProviderGeometry::Cartesian,
                        SpatialProviderOperation::CharacteristicNoInflow);
  if (boundary_linearization)
    capabilities.enable(dimension, SpatialProviderGeometry::Cartesian,
                        SpatialProviderOperation::BoundaryLinearization);
  return capabilities;
}

[[nodiscard]] constexpr SpatialProviderCapabilities with_embedded_boundary_residuals(
    SpatialProviderCapabilities capabilities) {
  for (int dimension = 1; dimension <= 3; ++dimension) {
    if (!capabilities.supports(
            {dimension, SpatialProviderGeometry::Cartesian, SpatialProviderOperation::Residual}))
      continue;
    capabilities.enable(dimension, SpatialProviderGeometry::Staircase,
                        SpatialProviderOperation::Residual);
    capabilities.enable(dimension, SpatialProviderGeometry::CutCell,
                        SpatialProviderOperation::Residual);
    if (capabilities.supports({dimension, SpatialProviderGeometry::Cartesian,
                               SpatialProviderOperation::CharacteristicNoInflow})) {
      capabilities.enable(dimension, SpatialProviderGeometry::Staircase,
                          SpatialProviderOperation::CharacteristicNoInflow);
      capabilities.enable(dimension, SpatialProviderGeometry::CutCell,
                          SpatialProviderOperation::CharacteristicNoInflow);
    }
  }
  return capabilities;
}

[[nodiscard]] constexpr SpatialProviderCapabilities make_polar_spatial_provider(int dimension) {
  SpatialProviderCapabilities capabilities;
  capabilities.enable(dimension, SpatialProviderGeometry::Polar,
                      SpatialProviderOperation::Residual);
  return capabilities;
}

}  // namespace pops
