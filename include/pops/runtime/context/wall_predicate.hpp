#pragma once

#include <pops/core/foundation/types.hpp>  // Real
#include <pops/numerics/elliptic/interface/spatial_provider.hpp>
#include <pops/numerics/spatial/embedded_boundary/domain.hpp>  // detail::DiscDomain (the level-set domain it lives in since ADC-327)

#include <cmath>      // std::hypot
#include <stdexcept>  // std::runtime_error
#include <string>

/// @file
/// @brief ELLIPTIC conductor-wall predicate shared by the System and AmrSystem runtimes (the wall acts
///        on the Poisson side). Both derive the same predicate from the wall, radius and exact
///        Cartesian bounds; only the error-message prefix differs. The square overload preserves the
///        historical direct-C++ contract while the axis-resolved overload centres on Lx/Ly.
///
///        The TRANSPORT-side domain descriptor (detail::DiscDomain) and its generic contract moved to
///        numerics/embedded_boundary.hpp (ADC-327); this header re-includes it so existing includers
///        keep seeing detail::DiscDomain unchanged.

namespace pops {
namespace detail {

struct CircleWallActiveRegionSource2D {
  Real radius = Real(0);
  Real center_x = Real(0);
  Real center_y = Real(0);

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.active-region.circle", 1};
  }

  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(radius).scalar(center_x).scalar(center_y);
  }

  [[nodiscard]] bool operator()(Real x, Real y) const {
    return std::hypot(x - center_x, y - center_y) < radius;
  }
};

/// Axis-resolved Cartesian overload.  A circular wall remains circular; only its centre follows the
/// independent physical bounds of the rectangular domain.
inline ActiveRegionProvider2D wall_predicate(const std::string& wall, double wall_radius, double Lx,
                                             double Ly, const std::string& err_context,
                                             double xlo = 0.0, double ylo = 0.0) {
  if (wall == "none")
    return {};
  if (wall == "circle")
    return ActiveRegionProvider2D(CircleWallActiveRegionSource2D{
        Real(wall_radius), Real(xlo + 0.5 * Lx), Real(ylo + 0.5 * Ly)});
  throw std::runtime_error(err_context + ": unknown wall '" + wall + "'");
}

/// Historical square-domain shorthand retained for the uniform runtime and direct C++ callers.
inline ActiveRegionProvider2D wall_predicate(const std::string& wall, double wall_radius, double L,
                                             const std::string& err_context, double xlo = 0.0,
                                             double ylo = 0.0) {
  return wall_predicate(wall, wall_radius, L, L, err_context, xlo, ylo);
}

}  // namespace detail
}  // namespace pops
