#pragma once

#include <pops/numerics/moments/affine_velocity.hpp>
#include <pops/numerics/fv/path_result.hpp>
#include <cmath>
#include <limits>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Raw moment interval certification requires strict floating-point semantics"
#endif

namespace pops::moments {

static_assert(std::numeric_limits<Real>::is_iec559 && std::numeric_limits<Real>::radix == 2,
              "Raw moment covariance certification requires an IEEE binary scalar");

/// Outward intervals certify the EXACT real interpretation of the stored raw
/// values. Every rounded operation is enclosed by adjacent representable
/// values. This requires strict IEEE operations, correctly rounded division,
/// nextafter, and gradual underflow: no reassociation, fast-math, FTZ or DAZ.
/// is_iec559 alone does not prove those execution-mode requirements on a GPU.
struct Interval {
  Real lower, upper;
};

POPS_HD inline Real lower_neighbor(Real value) {
  return std::nextafter(value, -std::numeric_limits<Real>::infinity());
}
POPS_HD inline Real upper_neighbor(Real value) {
  return std::nextafter(value, std::numeric_limits<Real>::infinity());
}
POPS_HD inline bool finite_interval(const Interval& value) {
  return std::isfinite(value.lower) && std::isfinite(value.upper) && value.lower <= value.upper;
}
POPS_HD inline bool zero_interval(const Interval& value) {
  return value.lower == Real(0) && value.upper == Real(0);
}
POPS_HD inline Interval quotient_interval(Real numerator, Real positive_denominator) {
  // These special cases are exact algebraic identities, not tolerance tests.
  if (numerator == Real(0))
    return {Real(0), Real(0)};
  if (positive_denominator == Real(1))
    return {numerator, numerator};
  if (numerator == positive_denominator)
    return {Real(1), Real(1)};
  if (numerator == -positive_denominator)
    return {Real(-1), Real(-1)};
  const Real value = numerator / positive_denominator;
  return {lower_neighbor(value), upper_neighbor(value)};
}
POPS_HD inline Interval rescale_interval(const Interval& value, Real positive_denominator) {
  return {quotient_interval(value.lower, positive_denominator).lower,
          quotient_interval(value.upper, positive_denominator).upper};
}
POPS_HD inline Interval subtract_interval(const Interval& a, const Interval& b) {
  if (zero_interval(b))
    return a;
  return {lower_neighbor(a.lower - b.upper), upper_neighbor(a.upper - b.lower)};
}
POPS_HD inline Interval multiply_interval(const Interval& a, const Interval& b) {
  if (zero_interval(a) || zero_interval(b))
    return {Real(0), Real(0)};
  const Real products[4] = {a.lower * b.lower, a.lower * b.upper, a.upper * b.lower,
                            a.upper * b.upper};
  Real smallest = products[0], largest = products[0];
  for (int k = 1; k < 4; ++k) {
    if (products[k] < smallest)
      smallest = products[k];
    if (products[k] > largest)
      largest = products[k];
  }
  return {lower_neighbor(smallest), upper_neighbor(largest)};
}
POPS_HD inline Interval square_interval(const Interval& value) {
  if (zero_interval(value))
    return {Real(0), Real(0)};
  const Real a = value.lower * value.lower, b = value.upper * value.upper;
  const Real largest = a > b ? a : b;
  if (value.lower <= Real(0) && value.upper >= Real(0))
    return {Real(0), upper_neighbor(largest)};
  const Real smallest = a < b ? a : b;
  return {lower_neighbor(smallest), upper_neighbor(largest)};
}

/// Sylvester certificate via the 2x2 Schur covariance of the raw H1 matrix.
/// Rounded normalization is enclosed, rather than treated as exact input.
/// The result never changes the raw moments or replaces an unproved minor.
template <int Order>
POPS_HD inline PathStatus certify_raw_covariance(
    const Real (&raw)[CartesianMomentBasis<Order>::size]) {
  static_assert(Order >= 2, "raw covariance needs all second moments");
  using Basis = CartesianMomentBasis<Order>;
  const Interval u = quotient_interval(raw[Basis::index(1, 0)], raw[0]),
                 v = quotient_interval(raw[Basis::index(0, 1)], raw[0]);
  const Interval m20 = quotient_interval(raw[Basis::index(2, 0)], raw[0]),
                 m11 = quotient_interval(raw[Basis::index(1, 1)], raw[0]),
                 m02 = quotient_interval(raw[Basis::index(0, 2)], raw[0]);
  if (!finite_interval(u) || !finite_interval(v) || !finite_interval(m20) ||
      !finite_interval(m11) || !finite_interval(m02))
    return PathStatus::IndeterminateCovariance;
  Interval a = subtract_interval(m20, square_interval(u));
  Interval b = subtract_interval(m11, multiply_interval(u, v));
  Interval c = subtract_interval(m02, square_interval(v));
  if (!finite_interval(a) || !finite_interval(b) || !finite_interval(c))
    return PathStatus::IndeterminateCovariance;
  if (a.upper <= Real(0) || c.upper <= Real(0))
    return PathStatus::NonPositiveCovariance;
  if (!(a.lower > Real(0)) || !(c.lower > Real(0)))
    return PathStatus::IndeterminateCovariance;
  // A common positive scale avoids determinant overflow. Its value is
  // a computational rescaling only; it is never written into physical state.
  Real scale = Real(0);
  const Real bounds[6] = {a.lower, a.upper, b.lower, b.upper, c.lower, c.upper};
  for (Real value : bounds) {
    if (std::fabs(value) > scale)
      scale = std::fabs(value);
  }
  a = rescale_interval(a, scale);
  b = rescale_interval(b, scale);
  c = rescale_interval(c, scale);
  if (!finite_interval(a) || !finite_interval(b) || !finite_interval(c))
    return PathStatus::IndeterminateCovariance;
  const Interval determinant = subtract_interval(multiply_interval(a, c), square_interval(b));
  if (!finite_interval(determinant))
    return PathStatus::IndeterminateCovariance;
  if (determinant.lower > Real(0))
    return PathStatus::Success;
  if (determinant.upper <= Real(0))
    return PathStatus::NonPositiveCovariance;
  return PathStatus::IndeterminateCovariance;
}

template <int Order>
struct NormalizedRawMoments {
  static_assert(Order >= 2, "normalized covariance needs all second moments");
  Real density = Real(0);
  Real normalized[CartesianMomentBasis<Order>::size]{};
  Real covariance_scale = Real(0);
  Real xx = Real(0), xy = Real(0), determinant = Real(0);
};

template <int Order>
POPS_HD inline PathStatus recover_raw_moments(const Real (&raw)[CartesianMomentBasis<Order>::size],
                                              NormalizedRawMoments<Order>& state) {
  using Basis = CartesianMomentBasis<Order>;
  static_assert(Order >= 2);
  for (Real value : raw)
    if (!std::isfinite(value))
      return PathStatus::NonFiniteInput;
  if (!(raw[0] > Real(0)))
    return PathStatus::NonPositiveDensity;
  state.density = raw[0];
  for (int k = 0; k < Basis::size; ++k) {
    state.normalized[k] = raw[k] / raw[0];
    if (!std::isfinite(state.normalized[k]))
      return PathStatus::NonFiniteNormalization;
  }
  const Real u = state.normalized[Basis::index(1, 0)], v = state.normalized[Basis::index(0, 1)];
  const Real a = std::fma(-u, u, state.normalized[Basis::index(2, 0)]);
  const Real b = std::fma(-u, v, state.normalized[Basis::index(1, 1)]);
  const Real c = std::fma(-v, v, state.normalized[Basis::index(0, 2)]);
  if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c))
    return PathStatus::NonFiniteNormalization;
  const auto certificate = certify_raw_covariance<Order>(raw);
  if (certificate != PathStatus::Success)
    return certificate;
  Real scale = std::fabs(a);
  if (std::fabs(b) > scale)
    scale = std::fabs(b);
  if (std::fabs(c) > scale)
    scale = std::fabs(c);
  if (!(scale > Real(0)) || !(a > Real(0)) || !(c > Real(0)))
    return PathStatus::NonPositiveCovariance;
  const Real xx = a / scale, xy = b / scale, yy = c / scale;
  const Real determinant = std::fma(xx, yy, -xy * xy);
  if (!(determinant > Real(0)) || !std::isfinite(determinant))
    return PathStatus::NonPositiveCovariance;
  state.covariance_scale = scale;
  state.xx = xx;
  state.xy = xy;
  state.determinant = determinant;
  return PathStatus::Success;
}

/// sqrt(E[(g.v)^2]); Cholesky/hypot evaluate the nonnegative quadratic form.
template <int Order>
POPS_HD inline Real raw_second_moment_norm(const NormalizedRawMoments<Order>& state, Real gx,
                                           Real gy) {
  using Basis = CartesianMomentBasis<Order>;
  const Real l11 = std::sqrt(state.xx);
  const Real p = l11 * gx + (state.xy / l11) * gy;
  const Real q = std::sqrt(state.determinant / state.xx) * gy;
  const Real sigma = std::sqrt(state.covariance_scale) * std::hypot(p, q);
  const Real mean =
      gx * state.normalized[Basis::index(1, 0)] + gy * state.normalized[Basis::index(0, 1)];
  return std::hypot(mean, sigma);
}

}  // namespace pops::moments
