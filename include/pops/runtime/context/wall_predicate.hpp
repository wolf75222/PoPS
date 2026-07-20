#pragma once

#include <pops/core/foundation/types.hpp>  // Real
#include <pops/numerics/elliptic/interface/spatial_provider.hpp>
#include <pops/numerics/spatial/embedded_boundary/domain.hpp>  // detail::DiscDomain (the level-set domain it lives in since ADC-327)

#include <cmath>      // std::hypot
#include <stdexcept>  // std::runtime_error
#include <string>

/// @file
/// @brief ELLIPTIC conductor-wall predicate shared by the System and AmrSystem runtimes (the wall acts
///        on the Poisson side). Both derived the same predicate from the same parameters (wall, radius,
///        L); only the error-message prefix differed. Centralized here by PURE extraction: the body
///        (centered circle, std::hypot < R comparison) is reused identically.
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

/// Builds the "inside the conductor" predicate (embedded wall for the Poisson solver)
/// from the wall mode @p wall, the radius @p wall_radius, and the Cartesian domain bounds.
///   - "none": no wall -> empty predicate.
///   - "circle": disc centered in [xlo,xlo+L] x [ylo,ylo+L] with radius @p wall_radius.
///   - other: error, prefixed by @p err_context (e.g. "System::set_poisson").
/// Body reused identically from the System / AmrSystem runtimes (bit-identical).
inline ActiveRegionProvider2D wall_predicate(const std::string& wall, double wall_radius, double L,
                                             const std::string& err_context, double xlo = 0.0,
                                             double ylo = 0.0) {
  if (wall == "none")
    return {};
  if (wall == "circle")
    return ActiveRegionProvider2D(CircleWallActiveRegionSource2D{
        Real(wall_radius), Real(xlo + 0.5 * L), Real(ylo + 0.5 * L)});
  throw std::runtime_error(err_context + ": unknown wall '" + wall + "'");
}

}  // namespace detail
}  // namespace pops
