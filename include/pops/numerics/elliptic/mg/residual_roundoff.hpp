/// @file
/// @brief Explicit binary64 roundoff bounds for a constant Cartesian five-point residual.
#pragma once
#include <pops/mesh/storage/field_view.hpp>
#include <pops/numerics/elliptic/linear/scaled_scalar.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace pops::elliptic::mg::roundoff {

inline Real gamma_for_operations(std::uint64_t evaluation_operations,
                                 std::uint64_t construction_operations) {
  const Real unit = std::numeric_limits<Real>::epsilon() / Real(2);
  const Real evaluation = static_cast<Real>(evaluation_operations) * unit;
  const Real construction = static_cast<Real>(construction_operations) * unit;
  if (!(evaluation < Real(0.5)) || !(construction < Real(0.5)))
    return std::numeric_limits<Real>::infinity();
  // Directed nextafter steps keep this host-side gamma an upper bound even when
  // forming its denominators rounds upward. The operation counts remain exact
  // binary64 integers under the admission limit above.
  const Real infinity = std::numeric_limits<Real>::infinity();
  const Real evaluation_denominator = std::nextafter(Real(1) - evaluation, Real(0));
  const Real construction_denominator = std::nextafter(Real(1) - construction, Real(0));
  const Real evaluation_gamma = std::nextafter(evaluation / evaluation_denominator, infinity);
  const Real construction_gamma = std::nextafter(construction / construction_denominator, infinity);
  const Real sum_denominator = std::nextafter(Real(1) - construction_gamma, Real(0));
  return std::nextafter(evaluation_gamma / sum_denominator, infinity);
}

POPS_HD inline bool normal_or_zero(Real value) {
  return Kokkos::isfinite(value) &&
         (value == Real(0) || Kokkos::abs(value) >= std::numeric_limits<Real>::min());
}

/// Unit roundoff is epsilon/2. The gamma count includes stencil additions/products, inverse
/// spacing formation, and final RHS addition. A second gamma protects construction of the
/// positive absolute-term sum itself. No forcing-relative heuristic enters this bound.
template <int Dim>
inline Real stencil_gamma() {
  return gamma_for_operations(1 + 8 * Dim, 3 + 12 * Dim);
}

POPS_HD inline Real upward_materialize(const ::pops::detail::ScaledScalar& value) {
  Real result = Real(0);
  if (!value.is_finite())
    return std::numeric_limits<Real>::infinity();
  if (value.is_zero())
    return Real(0);
  if (!value.try_materialize(result)) {
    constexpr int smallest_exponent =
        std::numeric_limits<Real>::min_exponent - std::numeric_limits<Real>::digits + 1;
    if (value.exponent() <= smallest_exponent)
      return std::numeric_limits<Real>::denorm_min();
    return std::numeric_limits<Real>::infinity();
  }
#if defined(KOKKOS_ENABLE_SYCL)
  return sycl::nextafter(result, std::numeric_limits<Real>::infinity());
#else
  return std::nextafter(result, std::numeric_limits<Real>::infinity());
#endif
}

template <int Dim>
struct AbsoluteMagnitudeKernel {
  FieldView<const Real, Dim> values, active;
  POPS_HD Real operator()(const Index<Dim>& cell) const {
    if (active.data != nullptr && active(cell, 0) < Real(0.5))
      return Real(0);
    const Real value = values(cell, 0);
    return Kokkos::isfinite(value) ? Kokkos::abs(value) : std::numeric_limits<Real>::infinity();
  }
};

template <int Dim>
struct FivePointBoundKernel {
  FieldView<const Real, Dim> phi, rhs, active;
  Real inverse_spacing_squared[Dim]{};
  Real gamma = Real(0);
  POPS_HD Real operator()(const Index<Dim>& cell) const {
    using Scaled = ::pops::detail::ScaledScalar;
    if (active.data != nullptr && active(cell, 0) < Real(0.5))
      return Real(0);
    if (!normal_or_zero(rhs(cell, 0)) || !normal_or_zero(phi(cell, 0)))
      return std::numeric_limits<Real>::infinity();
    Scaled absolute_terms = Scaled::from(Kokkos::abs(rhs(cell, 0)));
    Real laplacian = Real(0);
    for (int axis = 0; axis < Dim; ++axis) {
      Index<Dim> lower = cell, upper = cell;
      --lower[axis];
      ++upper[axis];
      const Real scale = inverse_spacing_squared[axis];
      const Real twice = Real(2) * phi(cell, 0);
      const Real first = phi(upper, 0) - twice;
      const Real difference = first + phi(lower, 0);
      const Real term = difference * scale;
      if (!normal_or_zero(phi(upper, 0)) || !normal_or_zero(phi(lower, 0)) ||
          !normal_or_zero(twice) || !normal_or_zero(first) || !normal_or_zero(difference) ||
          !normal_or_zero(term) || !(scale >= std::numeric_limits<Real>::min()) ||
          !Kokkos::isfinite(scale) || (difference != Real(0) && term == Real(0)))
        return std::numeric_limits<Real>::infinity();
      laplacian += term;
      if (!normal_or_zero(laplacian))
        return std::numeric_limits<Real>::infinity();
      Scaled values = Scaled::sum(Scaled::from(Kokkos::abs(phi(upper, 0))),
                                  Scaled::from(Kokkos::abs(phi(lower, 0))));
      values = Scaled::sum(
          values, Scaled::product(Scaled::from(Real(2)), Scaled::from(Kokkos::abs(phi(cell, 0)))));
      absolute_terms = Scaled::sum(
          absolute_terms, Scaled::product(values, Scaled::from(inverse_spacing_squared[axis])));
    }
    if (!normal_or_zero(rhs(cell, 0) + laplacian))
      return std::numeric_limits<Real>::infinity();
    return upward_materialize(Scaled::product(Scaled::from(gamma), absolute_terms));
  }
};

inline std::optional<Real> relative_physical_residual_upper(Real physical_residual, Real bound,
                                                            Real rhs_norm) {
  using Scaled = ::pops::detail::ScaledScalar;
  if (!(rhs_norm > Real(0)))
    return {};
  const auto sum = Scaled::sum(Scaled::from(physical_residual), Scaled::from(bound));
  const Real value = upward_materialize(Scaled::quotient(sum, Scaled::from(rhs_norm)));
  return std::isfinite(static_cast<double>(value)) ? std::optional<Real>{value} : std::nullopt;
}

}  // namespace pops::elliptic::mg::roundoff
