/// @file
/// @brief Exact compile-time-ranked annular polar geometry and build request.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace pops {

/// Capability declaration for PoPS' annular coordinate system.  Polar coordinates describe the
/// two physical axes (r, theta); they are not a Cartesian rank-reduction fallback for 1D or 3D.
template <int Dim>
struct PolarGeometryCapabilities {
  static_assert(Dim >= 1 && Dim <= 3,
                "PolarGeometryCapabilities only supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;
  static constexpr bool available = Dim == 2;
  static constexpr std::string_view unavailable_reason =
      available ? std::string_view{}
                : std::string_view{"annular polar geometry has exactly the axes (r, theta)"};
};

static_assert(!PolarGeometryCapabilities<1>::available);
static_assert(PolarGeometryCapabilities<2>::available);
static_assert(!PolarGeometryCapabilities<3>::available);

/// Uniform cell-centred annulus with radial axis 0 and periodic azimuthal axis 1.
template <int Dim>
  requires(PolarGeometryCapabilities<Dim>::available)
class PolarGeometry {
 public:
  static constexpr int dimension = Dim;

  static PolarGeometry annulus(Box<Dim> domain, Real radial_lower, Real radial_upper) {
    if (!std::isfinite(static_cast<double>(radial_lower)) ||
        !std::isfinite(static_cast<double>(radial_upper)) || !(radial_lower > Real(0)) ||
        !(radial_upper > radial_lower))
      throw std::invalid_argument(
          "polar annulus requires finite radii with 0 < radial_lower < radial_upper");
    constexpr Real two_pi =
        Real(6.283185307179586476925286766559005768394338798750211641949889L);
    return PolarGeometry{Geometry<Dim>::from_bounds(
        domain, RealVector<Dim>{radial_lower, Real(0)},
        RealVector<Dim>{radial_upper, two_pi})};
  }

  const Geometry<Dim>& cartesian_mapping() const noexcept { return mapping_; }
  const Box<Dim>& domain() const noexcept { return mapping_.domain(); }
  Real radial_lower() const noexcept { return mapping_.lower()[0]; }
  Real radial_upper() const noexcept { return mapping_.upper()[0]; }
  Real dr() const noexcept { return mapping_.spacing(0); }
  Real dtheta() const noexcept { return mapping_.spacing(1); }

  Real r_cell(int radial_index) const noexcept {
    return mapping_.cell_coordinate(0, radial_index);
  }
  Real r_face(int radial_face_index) const noexcept {
    return mapping_.face_coordinate(0, radial_face_index);
  }
  Real theta_cell(int azimuthal_index) const noexcept {
    return mapping_.cell_coordinate(1, azimuthal_index);
  }
  Real theta_face(int azimuthal_face_index) const noexcept {
    return mapping_.face_coordinate(1, azimuthal_face_index);
  }

  bool operator==(const PolarGeometry&) const = default;

 private:
  explicit PolarGeometry(Geometry<Dim> mapping) : mapping_(mapping) {}

  Geometry<Dim> mapping_;
};

/// Complete exact-rank request shared by polar elliptic providers.
template <int Dim>
  requires(PolarGeometryCapabilities<Dim>::available)
struct PolarEllipticBuildRequest {
  PolarGeometry<Dim> geometry;
  mesh::BoxArray<Dim> boxes;
  mesh::Distribution<Dim> distribution;
  Index<Dim> local_rank{};
  PhysicalBoundaryConditions<Dim> boundary;
  mesh::BoxArrayValidationBudget layout_budget{};
};

}  // namespace pops
