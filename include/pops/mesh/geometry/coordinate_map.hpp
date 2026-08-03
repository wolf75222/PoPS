/// @file
/// @brief Allocation-free coordinate-map contract for compile-time spatial dimensions.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/index/real_vector.hpp>

#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace pops {

enum class CoordinateMapKind : std::uint8_t {
  Cartesian = 0,
  PlanarPolar = 1,
};

enum class MetricFaceSide : std::int8_t {
  Lower = -1,
  Upper = 1,
};

enum class InverseMapStatus : std::uint8_t {
  Success = 0,
  NonFinitePoint = 1,
  OffEmbeddedManifold = 2,
  SingularPoint = 3,
  OutsidePatch = 4,
};

template <int Dim>
struct InverseMapResult {
  RealVector<Dim> reference{};
  InverseMapStatus status = InverseMapStatus::NonFinitePoint;

  POPS_HD constexpr bool succeeded() const { return status == InverseMapStatus::Success; }
};

template <int Dim, int EmbedDim>
using CoordinateJacobian = std::array<std::array<Real, Dim>, EmbedDim>;

/// Exact structural identity.  Every parameter that changes physical coordinates is represented;
/// no pointer, cache address or rounded hash participates in equality.
template <int Dim, int EmbedDim>
struct CoordinateMapIdentity {
  CoordinateMapKind kind = CoordinateMapKind::Cartesian;
  RealVector<EmbedDim> origin{};
  RealVector<Dim> lower{};
  RealVector<Dim> upper{};
  std::array<int, Dim> embedded_axis{};
  std::array<int, Dim> orientation{};

  constexpr bool operator==(const CoordinateMapIdentity&) const = default;
};

struct CoordinateMapCapabilities {
  int logical_dimension = 0;
  int embedding_dimension = 0;
  CoordinateMapKind kind = CoordinateMapKind::Cartesian;
  bool affine = false;
  bool cell_centers = false;
  bool face_centers = false;
  bool jacobian = false;
  bool exact_cell_measure = false;
  bool exact_oriented_face_area = false;
  bool inverse_map = false;
  bool compile_time_axes = false;
  bool device_callable = false;

  constexpr bool operator==(const CoordinateMapCapabilities&) const = default;
};

namespace coordinate_map_detail {

POPS_HD constexpr bool finite(Real value) {
  return value == value && value != std::numeric_limits<Real>::infinity() &&
         value != -std::numeric_limits<Real>::infinity();
}

POPS_HD constexpr Real abs(Real value) {
  return value < Real(0) ? -value : value;
}

POPS_HD constexpr Real canonical_zero(Real value) {
  return value == Real(0) ? Real(0) : value;
}

POPS_HD constexpr Real inverse_tolerance(Real left, Real right = Real(0)) {
  return Real(64) * std::numeric_limits<Real>::epsilon() * (Real(1) + abs(left) + abs(right));
}

template <int Dim>
constexpr std::array<int, Dim> identity_axes() {
  std::array<int, Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = axis;
  return result;
}

template <int Dim>
constexpr std::array<int, Dim> positive_orientations() {
  std::array<int, Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = 1;
  return result;
}

template <int Axis, int Dim, int EmbedDim, class Map>
concept CoordinateMapAxis =
    requires(const Map& map, const RealVector<Dim>& lower, const RealVector<Dim>& upper) {
      {
        map.template oriented_face_area_vector<Axis, MetricFaceSide::Lower>(lower, upper)
      } -> std::same_as<RealVector<EmbedDim>>;
      {
        map.template oriented_face_area_vector<Axis, MetricFaceSide::Upper>(lower, upper)
      } -> std::same_as<RealVector<EmbedDim>>;
    };

template <int Dim, int EmbedDim, class Map, int... Axis>
consteval bool coordinate_map_axes(std::integer_sequence<int, Axis...>) {
  return (CoordinateMapAxis<Axis, Dim, EmbedDim, Map> && ...);
}

}  // namespace coordinate_map_detail

/// Static coordinate-map contract.  The concrete map type remains visible to the compiler: this
/// concept introduces no virtual dispatch and no run-time geometry switch.
template <int Dim, int EmbedDim, class Map>
concept CoordinateMap =
    Dim >= 1 && Dim <= 3 && EmbedDim >= Dim && EmbedDim <= 3 && std::is_trivially_copyable_v<Map> &&
    requires(const Map& map, const RealVector<Dim>& reference, const RealVector<EmbedDim>& physical,
             const RealVector<Dim>& lower, const RealVector<Dim>& upper) {
      { Map::logical_dimension } -> std::convertible_to<int>;
      { Map::embedding_dimension } -> std::convertible_to<int>;
      requires Map::logical_dimension == Dim;
      requires Map::embedding_dimension == EmbedDim;
      { Map::capabilities() } -> std::same_as<CoordinateMapCapabilities>;
      { map.identity() } -> std::same_as<CoordinateMapIdentity<Dim, EmbedDim>>;
      { map.map(reference) } -> std::same_as<RealVector<EmbedDim>>;
      { map.jacobian(reference) } -> std::same_as<CoordinateJacobian<Dim, EmbedDim>>;
      { map.inverse_map(physical) } -> std::same_as<InverseMapResult<Dim>>;
      { map.cell_measure(lower, upper) } -> std::same_as<Real>;
    } &&
    coordinate_map_detail::coordinate_map_axes<Dim, EmbedDim, Map>(
        std::make_integer_sequence<int, Dim>{});

/// Orthogonal Cartesian map with compile-time rank and an exact signed axis embedding.  Logical
/// coordinates are normalized: reference=(0,...,0) maps to origin and each reference axis spans
/// its positive length in the selected signed physical direction.
template <int Dim, int EmbedDim = Dim>
class CartesianCoordinateMap {
 public:
  static_assert(Dim >= 1 && Dim <= 3,
                "CartesianCoordinateMap supports logical dimensions 1, 2, and 3");
  static_assert(EmbedDim >= Dim && EmbedDim <= 3,
                "CartesianCoordinateMap embedding rank must be between Dim and 3");

  static constexpr int logical_dimension = Dim;
  static constexpr int embedding_dimension = EmbedDim;

  static CartesianCoordinateMap make(
      RealVector<EmbedDim> origin, RealVector<Dim> lengths,
      std::array<int, Dim> embedded_axis = coordinate_map_detail::identity_axes<Dim>(),
      std::array<int, Dim> orientation = coordinate_map_detail::positive_orientations<Dim>()) {
    std::array<bool, EmbedDim> occupied{};
    for (int physical_axis = 0; physical_axis < EmbedDim; ++physical_axis) {
      if (!coordinate_map_detail::finite(origin[physical_axis]))
        throw std::invalid_argument("Cartesian coordinate-map origin must be finite");
      origin[physical_axis] = coordinate_map_detail::canonical_zero(origin[physical_axis]);
    }
    for (int axis = 0; axis < Dim; ++axis) {
      if (!coordinate_map_detail::finite(lengths[axis]) || !(lengths[axis] > Real(0)))
        throw std::invalid_argument("Cartesian coordinate-map lengths must be finite and positive");
      if (embedded_axis[axis] < 0 || embedded_axis[axis] >= EmbedDim ||
          occupied[static_cast<std::size_t>(embedded_axis[axis])])
        throw std::invalid_argument(
            "Cartesian coordinate-map embedded axes must be unique and in range");
      if (orientation[axis] != -1 && orientation[axis] != 1)
        throw std::invalid_argument("Cartesian coordinate-map orientations must be -1 or +1");
      occupied[static_cast<std::size_t>(embedded_axis[axis])] = true;
      lengths[axis] = coordinate_map_detail::canonical_zero(lengths[axis]);
    }
    return CartesianCoordinateMap(origin, lengths, embedded_axis, orientation);
  }

  static constexpr CoordinateMapCapabilities capabilities() {
    return {Dim,  EmbedDim, CoordinateMapKind::Cartesian, true, true, true, true, true, true, true,
            true, true};
  }

  POPS_HD CoordinateMapIdentity<Dim, EmbedDim> identity() const {
    CoordinateMapIdentity<Dim, EmbedDim> result{};
    result.kind = CoordinateMapKind::Cartesian;
    result.origin = origin_;
    result.upper = lengths_;
    result.embedded_axis = embedded_axis_;
    result.orientation = orientation_;
    return result;
  }

  POPS_HD RealVector<EmbedDim> map(const RealVector<Dim>& reference) const {
    RealVector<EmbedDim> result = origin_;
    for (int axis = 0; axis < Dim; ++axis)
      result[embedded_axis_[axis]] += Real(orientation_[axis]) * lengths_[axis] * reference[axis];
    return result;
  }

  POPS_HD CoordinateJacobian<Dim, EmbedDim> jacobian(const RealVector<Dim>&) const {
    CoordinateJacobian<Dim, EmbedDim> result{};
    for (int axis = 0; axis < Dim; ++axis)
      result[embedded_axis_[axis]][axis] = Real(orientation_[axis]) * lengths_[axis];
    return result;
  }

  POPS_HD InverseMapResult<Dim> inverse_map(const RealVector<EmbedDim>& physical) const {
    InverseMapResult<Dim> result{};
    std::array<bool, EmbedDim> occupied{};
    for (int axis = 0; axis < Dim; ++axis)
      occupied[static_cast<std::size_t>(embedded_axis_[axis])] = true;
    for (int physical_axis = 0; physical_axis < EmbedDim; ++physical_axis) {
      if (!coordinate_map_detail::finite(physical[physical_axis])) {
        result.status = InverseMapStatus::NonFinitePoint;
        return result;
      }
      if (!occupied[static_cast<std::size_t>(physical_axis)] &&
          coordinate_map_detail::abs(physical[physical_axis] - origin_[physical_axis]) >
              coordinate_map_detail::inverse_tolerance(physical[physical_axis],
                                                       origin_[physical_axis])) {
        result.status = InverseMapStatus::OffEmbeddedManifold;
        return result;
      }
    }
    for (int axis = 0; axis < Dim; ++axis) {
      const int physical_axis = embedded_axis_[axis];
      result.reference[axis] = Real(orientation_[axis]) *
                               (physical[physical_axis] - origin_[physical_axis]) / lengths_[axis];
    }
    result.status = InverseMapStatus::Success;
    return result;
  }

  POPS_HD Real cell_measure(const RealVector<Dim>& lower, const RealVector<Dim>& upper) const {
    Real measure = Real(1);
    for (int axis = 0; axis < Dim; ++axis)
      measure *= lengths_[axis] * coordinate_map_detail::abs(upper[axis] - lower[axis]);
    return measure;
  }

  template <int Axis, MetricFaceSide Side>
  POPS_HD RealVector<EmbedDim> oriented_face_area_vector(const RealVector<Dim>& lower,
                                                         const RealVector<Dim>& upper) const {
    static_assert(Axis >= 0 && Axis < Dim, "Cartesian metric face axis is outside the map rank");
    Real magnitude = Real(1);
    for (int axis = 0; axis < Dim; ++axis)
      if (axis != Axis)
        magnitude *= lengths_[axis] * coordinate_map_detail::abs(upper[axis] - lower[axis]);
    RealVector<EmbedDim> result{};
    constexpr int side = Side == MetricFaceSide::Upper ? 1 : -1;
    result[embedded_axis_[Axis]] =
        Real(side * orientation_[Axis]) * coordinate_map_detail::abs(magnitude);
    return result;
  }

 private:
  POPS_HD constexpr CartesianCoordinateMap(RealVector<EmbedDim> origin, RealVector<Dim> lengths,
                                           std::array<int, Dim> embedded_axis,
                                           std::array<int, Dim> orientation)
      : origin_(origin),
        lengths_(lengths),
        embedded_axis_(embedded_axis),
        orientation_(orientation) {}

  RealVector<EmbedDim> origin_{};
  RealVector<Dim> lengths_{};
  std::array<int, Dim> embedded_axis_{};
  std::array<int, Dim> orientation_{};
};

/// Exact finite-volume map for an annular sector embedded in the Cartesian plane.  Reference axis
/// 0 is radial and axis 1 is azimuthal.  Cell measures and integrated face vectors use analytic
/// sector integrals, rather than center-point quadrature.
class PlanarPolarCoordinateMap {
 public:
  static constexpr int logical_dimension = 2;
  static constexpr int embedding_dimension = 2;
  static constexpr Real kTwoPi = Real(6.2831853071795864769252867665590057683943387987502);

  static PlanarPolarCoordinateMap make(RealVector<2> center, Real radial_lower, Real radial_upper,
                                       Real angle_lower = Real(0), Real angle_upper = kTwoPi) {
    for (int axis = 0; axis < 2; ++axis) {
      if (!coordinate_map_detail::finite(center[axis]))
        throw std::invalid_argument("planar-polar coordinate-map center must be finite");
      center[axis] = coordinate_map_detail::canonical_zero(center[axis]);
    }
    if (!coordinate_map_detail::finite(radial_lower) ||
        !coordinate_map_detail::finite(radial_upper) || !(radial_lower > Real(0)) ||
        !(radial_upper > radial_lower))
      throw std::invalid_argument(
          "planar-polar coordinate-map radial bounds must be finite, positive and ordered");
    if (!coordinate_map_detail::finite(angle_lower) ||
        !coordinate_map_detail::finite(angle_upper) || !(angle_upper > angle_lower) ||
        angle_upper - angle_lower > kTwoPi)
      throw std::invalid_argument(
          "planar-polar coordinate-map angular span must be finite, positive and at most 2*pi");
    return PlanarPolarCoordinateMap(
        center, coordinate_map_detail::canonical_zero(radial_lower), radial_upper,
        coordinate_map_detail::canonical_zero(angle_lower), angle_upper);
  }

  static constexpr CoordinateMapCapabilities capabilities() {
    return {2,    2,   CoordinateMapKind::PlanarPolar, false, true, true, true, true, true, true,
            true, true};
  }

  POPS_HD CoordinateMapIdentity<2, 2> identity() const {
    CoordinateMapIdentity<2, 2> result{};
    result.kind = CoordinateMapKind::PlanarPolar;
    result.origin = center_;
    result.lower = RealVector<2>{radial_lower_, angle_lower_};
    result.upper = RealVector<2>{radial_upper_, angle_upper_};
    result.embedded_axis = {0, 1};
    result.orientation = {1, 1};
    return result;
  }

  POPS_HD RealVector<2> map(const RealVector<2>& reference) const {
    const Real radius = radius_(reference[0]);
    const Real angle = angle_(reference[1]);
    return RealVector<2>{center_[0] + radius * std::cos(angle),
                         center_[1] + radius * std::sin(angle)};
  }

  POPS_HD CoordinateJacobian<2, 2> jacobian(const RealVector<2>& reference) const {
    const Real radius = radius_(reference[0]);
    const Real angle = angle_(reference[1]);
    const Real radial_span = radial_upper_ - radial_lower_;
    const Real angular_span = angle_upper_ - angle_lower_;
    return {{{radial_span * std::cos(angle), -radius * angular_span * std::sin(angle)},
             {radial_span * std::sin(angle), radius * angular_span * std::cos(angle)}}};
  }

  POPS_HD InverseMapResult<2> inverse_map(const RealVector<2>& physical) const {
    InverseMapResult<2> result{};
    if (!coordinate_map_detail::finite(physical[0]) ||
        !coordinate_map_detail::finite(physical[1])) {
      result.status = InverseMapStatus::NonFinitePoint;
      return result;
    }
    const Real x = physical[0] - center_[0];
    const Real y = physical[1] - center_[1];
    const Real radius = std::sqrt(x * x + y * y);
    if (!(radius > Real(0))) {
      result.status = InverseMapStatus::SingularPoint;
      return result;
    }

    Real angle = std::atan2(y, x);
    angle += std::floor((angle_lower_ - angle) / kTwoPi) * kTwoPi;
    if (angle < angle_lower_)
      angle += kTwoPi;
    if (angle >= angle_lower_ + kTwoPi)
      angle -= kTwoPi;

    const Real radial_span = radial_upper_ - radial_lower_;
    const Real angular_span = angle_upper_ - angle_lower_;
    result.reference = RealVector<2>{(radius - radial_lower_) / radial_span,
                                     (angle - angle_lower_) / angular_span};
    const Real tolerance = Real(64) * std::numeric_limits<Real>::epsilon();
    if (result.reference[0] < -tolerance || result.reference[0] > Real(1) + tolerance ||
        result.reference[1] < -tolerance || result.reference[1] > Real(1) + tolerance) {
      result.status = InverseMapStatus::OutsidePatch;
      return result;
    }
    result.reference[0] = clamp_unit_(result.reference[0]);
    result.reference[1] = clamp_unit_(result.reference[1]);
    result.status = InverseMapStatus::Success;
    return result;
  }

  POPS_HD Real cell_measure(const RealVector<2>& lower, const RealVector<2>& upper) const {
    const Real radial_lower = radius_(lower[0]);
    const Real radial_upper = radius_(upper[0]);
    const Real angle_span = (angle_upper_ - angle_lower_) * (upper[1] - lower[1]);
    return coordinate_map_detail::abs(
        Real(0.5) * (radial_upper * radial_upper - radial_lower * radial_lower) * angle_span);
  }

  template <int Axis, MetricFaceSide Side>
  POPS_HD RealVector<2> oriented_face_area_vector(const RealVector<2>& lower,
                                                  const RealVector<2>& upper) const {
    static_assert(Axis == 0 || Axis == 1, "planar-polar metric face axis must be 0 or 1");
    constexpr Real side = Side == MetricFaceSide::Upper ? Real(1) : Real(-1);
    if constexpr (Axis == 0) {
      const Real reference_radius = Side == MetricFaceSide::Upper ? upper[0] : lower[0];
      const Real radius = radius_(reference_radius);
      const Real angle_lower = angle_(lower[1]);
      const Real angle_upper = angle_(upper[1]);
      return RealVector<2>{side * radius * (std::sin(angle_upper) - std::sin(angle_lower)),
                           side * radius * (-std::cos(angle_upper) + std::cos(angle_lower))};
    } else {
      const Real reference_angle = Side == MetricFaceSide::Upper ? upper[1] : lower[1];
      const Real angle = angle_(reference_angle);
      const Real radial_span = radius_(upper[0]) - radius_(lower[0]);
      return RealVector<2>{side * radial_span * -std::sin(angle),
                           side * radial_span * std::cos(angle)};
    }
  }

 private:
  POPS_HD constexpr PlanarPolarCoordinateMap(RealVector<2> center, Real radial_lower,
                                             Real radial_upper, Real angle_lower, Real angle_upper)
      : center_(center),
        radial_lower_(radial_lower),
        radial_upper_(radial_upper),
        angle_lower_(angle_lower),
        angle_upper_(angle_upper) {}

  POPS_HD Real radius_(Real reference_radius) const {
    return radial_lower_ + reference_radius * (radial_upper_ - radial_lower_);
  }

  POPS_HD Real angle_(Real reference_angle) const {
    return angle_lower_ + reference_angle * (angle_upper_ - angle_lower_);
  }

  POPS_HD static constexpr Real clamp_unit_(Real value) {
    return value < Real(0) ? Real(0) : (value > Real(1) ? Real(1) : value);
  }

  RealVector<2> center_{};
  Real radial_lower_ = Real(1);
  Real radial_upper_ = Real(2);
  Real angle_lower_ = Real(0);
  Real angle_upper_ = kTwoPi;
};

static_assert(CoordinateMap<1, 1, CartesianCoordinateMap<1>>);
static_assert(CoordinateMap<2, 2, CartesianCoordinateMap<2>>);
static_assert(CoordinateMap<3, 3, CartesianCoordinateMap<3>>);
static_assert(CoordinateMap<1, 3, CartesianCoordinateMap<1, 3>>);
static_assert(CoordinateMap<2, 2, PlanarPolarCoordinateMap>);

}  // namespace pops
