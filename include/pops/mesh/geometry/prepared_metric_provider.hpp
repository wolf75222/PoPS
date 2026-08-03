/// @file
/// @brief Prepared cell/face metrics over a compile-time coordinate map.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/geometry/coordinate_map.hpp>
#include <pops/mesh/index/box.hpp>

#include <concepts>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace pops {

template <int Dim, int EmbedDim>
struct PreparedMetricIdentity {
  CoordinateMapIdentity<Dim, EmbedDim> coordinate_map{};
  Box<Dim> domain{};

  constexpr bool operator==(const PreparedMetricIdentity&) const = default;
};

struct PreparedMetricCapabilities {
  CoordinateMapCapabilities coordinate_map{};
  bool exact_domain_identity = false;
  bool ghost_coordinates = false;
  bool allocation_free_queries = false;

  constexpr bool operator==(const PreparedMetricCapabilities&) const = default;
};

template <int Dim>
struct ReferenceCell {
  RealVector<Dim> lower{};
  RealVector<Dim> upper{};
  RealVector<Dim> center{};
};

namespace prepared_metric_detail {

template <int Axis, int Dim, class Provider>
concept PreparedMetricAxis = requires(const Provider& provider, const Index<Dim>& index) {
  {
    provider.template face_center<Axis, MetricFaceSide::Lower>(index)
  } -> std::same_as<typename Provider::PhysicalPoint>;
  {
    provider.template face_center<Axis, MetricFaceSide::Upper>(index)
  } -> std::same_as<typename Provider::PhysicalPoint>;
  {
    provider.template oriented_face_area_vector<Axis, MetricFaceSide::Lower>(index)
  } -> std::same_as<typename Provider::PhysicalPoint>;
  {
    provider.template oriented_face_area_vector<Axis, MetricFaceSide::Upper>(index)
  } -> std::same_as<typename Provider::PhysicalPoint>;
};

template <int Dim, class Provider, int... Axis>
consteval bool prepared_metric_axes(std::integer_sequence<int, Axis...>) {
  return (PreparedMetricAxis<Axis, Dim, Provider> && ...);
}

}  // namespace prepared_metric_detail

/// Prepared metric-provider contract.  Axis and face side are compile-time values; providers are
/// trivially copyable values suitable for direct capture in Kokkos kernels.
template <int Dim, class Provider>
concept PreparedMetricProvider =
    Dim >= 1 && Dim <= 3 && std::is_trivially_copyable_v<Provider> &&
    requires(const Provider& provider, const Index<Dim>& index,
             const typename Provider::PhysicalPoint& physical) {
      { Provider::logical_dimension } -> std::convertible_to<int>;
      requires Provider::logical_dimension == Dim;
      { Provider::embedding_dimension } -> std::convertible_to<int>;
      { Provider::capabilities() } -> std::same_as<PreparedMetricCapabilities>;
      {
        provider.identity()
      } -> std::same_as<PreparedMetricIdentity<Dim, Provider::embedding_dimension>>;
      { provider.reference_cell(index) } -> std::same_as<ReferenceCell<Dim>>;
      { provider.cell_center(index) } -> std::same_as<typename Provider::PhysicalPoint>;
      {
        provider.jacobian(index)
      } -> std::same_as<CoordinateJacobian<Dim, Provider::embedding_dimension>>;
      { provider.cell_measure(index) } -> std::same_as<Real>;
      { provider.inverse_map(physical) } -> std::same_as<InverseMapResult<Dim>>;
    } &&
    prepared_metric_detail::prepared_metric_axes<Dim, Provider>(
        std::make_integer_sequence<int, Dim>{});

/// Validated, allocation-free binding of a coordinate map to an inclusive integer domain.  The
/// map type is retained in the provider type, so Cartesian and polar queries never share a run-time
/// dispatch path.  Coordinates outside the domain remain defined for ghost-cell kernels.
template <class Map>
class PreparedMappedMetricProvider {
 public:
  static constexpr int logical_dimension = Map::logical_dimension;
  static constexpr int embedding_dimension = Map::embedding_dimension;
  using PhysicalPoint = RealVector<embedding_dimension>;
  using Identity = PreparedMetricIdentity<logical_dimension, embedding_dimension>;

  static_assert(CoordinateMap<logical_dimension, embedding_dimension, Map>,
                "PreparedMappedMetricProvider requires the complete CoordinateMap contract");

  static PreparedMappedMetricProvider prepare(const Box<logical_dimension>& domain, Map map) {
    if (domain.empty())
      throw std::invalid_argument("prepared metric provider requires a non-empty index domain");
    RealVector<logical_dimension> inverse_extent{};
    for (int axis = 0; axis < logical_dimension; ++axis) {
      const std::int64_t extent = domain.length(axis);
      if (extent <= 0)
        throw std::invalid_argument("prepared metric provider requires positive axis extents");
      inverse_extent[axis] = Real(1) / static_cast<Real>(extent);
    }
    return PreparedMappedMetricProvider(domain, map, inverse_extent);
  }

  static constexpr PreparedMetricCapabilities capabilities() {
    return {Map::capabilities(), true, true, true};
  }

  POPS_HD Identity identity() const { return Identity{map_.identity(), domain_}; }

  POPS_HD ReferenceCell<logical_dimension> reference_cell(
      const Index<logical_dimension>& index) const {
    ReferenceCell<logical_dimension> result{};
    for (int axis = 0; axis < logical_dimension; ++axis) {
      const Real offset = static_cast<Real>(index[axis]) - static_cast<Real>(domain_.lo[axis]);
      result.lower[axis] = offset * inverse_extent_[axis];
      result.upper[axis] = (offset + Real(1)) * inverse_extent_[axis];
      result.center[axis] = (offset + Real(0.5)) * inverse_extent_[axis];
    }
    return result;
  }

  POPS_HD PhysicalPoint cell_center(const Index<logical_dimension>& index) const {
    return map_.map(reference_cell(index).center);
  }

  template <int Axis, MetricFaceSide Side>
  POPS_HD PhysicalPoint face_center(const Index<logical_dimension>& index) const {
    static_assert(Axis >= 0 && Axis < logical_dimension,
                  "prepared metric face axis is outside the provider rank");
    auto reference = reference_cell(index);
    reference.center[Axis] =
        Side == MetricFaceSide::Upper ? reference.upper[Axis] : reference.lower[Axis];
    return map_.map(reference.center);
  }

  POPS_HD CoordinateJacobian<logical_dimension, embedding_dimension> jacobian(
      const Index<logical_dimension>& index) const {
    return map_.jacobian(reference_cell(index).center);
  }

  POPS_HD Real cell_measure(const Index<logical_dimension>& index) const {
    const auto reference = reference_cell(index);
    return map_.cell_measure(reference.lower, reference.upper);
  }

  template <int Axis, MetricFaceSide Side>
  POPS_HD PhysicalPoint oriented_face_area_vector(const Index<logical_dimension>& index) const {
    static_assert(Axis >= 0 && Axis < logical_dimension,
                  "prepared metric face axis is outside the provider rank");
    const auto reference = reference_cell(index);
    return map_.template oriented_face_area_vector<Axis, Side>(reference.lower, reference.upper);
  }

  POPS_HD InverseMapResult<logical_dimension> inverse_map(const PhysicalPoint& physical) const {
    return map_.inverse_map(physical);
  }

  POPS_HD const Box<logical_dimension>& domain() const { return domain_; }
  POPS_HD const Map& coordinate_map() const { return map_; }

 private:
  POPS_HD constexpr PreparedMappedMetricProvider(Box<logical_dimension> domain, Map map,
                                                 RealVector<logical_dimension> inverse_extent)
      : domain_(domain), map_(map), inverse_extent_(inverse_extent) {}

  Box<logical_dimension> domain_{};
  Map map_;
  RealVector<logical_dimension> inverse_extent_{};
};

template <class Map>
[[nodiscard]] auto prepare_metric_provider(const Box<Map::logical_dimension>& domain, Map map) {
  return PreparedMappedMetricProvider<Map>::prepare(domain, map);
}

static_assert(PreparedMetricProvider<1, PreparedMappedMetricProvider<CartesianCoordinateMap<1>>>);
static_assert(PreparedMetricProvider<2, PreparedMappedMetricProvider<CartesianCoordinateMap<2>>>);
static_assert(PreparedMetricProvider<3, PreparedMappedMetricProvider<CartesianCoordinateMap<3>>>);
static_assert(PreparedMetricProvider<2, PreparedMappedMetricProvider<PlanarPolarCoordinateMap>>);

}  // namespace pops
