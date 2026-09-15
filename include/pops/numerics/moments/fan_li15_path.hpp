#pragma once

#include <pops/numerics/moments/affine_velocity.hpp>

#include <cmath>
#include <limits>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "FanLi15 interval certification requires strict floating-point semantics"
#endif

namespace pops::moments {

/// The full-temperature Fan--Li D=2/M=4 hyperbolicity domain, not the full moment cone.
enum class FanLi15PathStatus : unsigned char {
  Success,
  NonFiniteInput,
  NonPositiveDensity,
  NonFiniteNormalization,
  NonPositiveCovariance,
  NonFiniteDirection,
  IntegralFailure,
  NonFiniteResult,
  /// Floating intervals cannot prove the exact raw covariance strictly SPD.
  IndeterminateCovariance
};

struct FanLi15PathResult {
  FanLi15PathStatus status = FanLi15PathStatus::IntegralFailure;
  /// Integral of B_g(Phi) dPhi, on the LEFT of U_t+div(F)+B grad(U)=S.
  /// Only q-outer raw slots 4,8,11,13,14 can be nonzero.
  Real integral[15]{};
  /// Both endpoint majorants use the SAME fixed covector g.
  Real speed_bound = Real(0);
  /// 0 or 1 for a refused input state; -1 for a direction/integral failure.
  int input_side = -1;
  POPS_HD bool succeeded() const { return status == FanLi15PathStatus::Success; }
};

namespace fan_li15_detail {

using Basis = CartesianMomentBasis<4>;

static_assert(std::numeric_limits<Real>::is_iec559 && std::numeric_limits<Real>::radix == 2,
              "FanLi15 covariance certification requires an IEEE binary scalar");

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
POPS_HD inline FanLi15PathStatus certify_raw_covariance(const Real (&raw)[15]) {
  const Interval u = quotient_interval(raw[1], raw[0]), v = quotient_interval(raw[5], raw[0]);
  const Interval m20 = quotient_interval(raw[2], raw[0]), m11 = quotient_interval(raw[6], raw[0]),
                 m02 = quotient_interval(raw[9], raw[0]);
  if (!finite_interval(u) || !finite_interval(v) || !finite_interval(m20) ||
      !finite_interval(m11) || !finite_interval(m02))
    return FanLi15PathStatus::IndeterminateCovariance;
  Interval a = subtract_interval(m20, square_interval(u));
  Interval b = subtract_interval(m11, multiply_interval(u, v));
  Interval c = subtract_interval(m02, square_interval(v));
  if (!finite_interval(a) || !finite_interval(b) || !finite_interval(c))
    return FanLi15PathStatus::IndeterminateCovariance;
  if (a.upper <= Real(0) || c.upper <= Real(0))
    return FanLi15PathStatus::NonPositiveCovariance;
  if (!(a.lower > Real(0)) || !(c.lower > Real(0)))
    return FanLi15PathStatus::IndeterminateCovariance;
  // One exact positive binary scale avoids determinant overflow. Its value is
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
    return FanLi15PathStatus::IndeterminateCovariance;
  const Interval determinant = subtract_interval(multiply_interval(a, c), square_interval(b));
  if (!finite_interval(determinant))
    return FanLi15PathStatus::IndeterminateCovariance;
  if (determinant.lower > Real(0))
    return FanLi15PathStatus::Success;
  if (determinant.upper <= Real(0))
    return FanLi15PathStatus::NonPositiveCovariance;
  return FanLi15PathStatus::IndeterminateCovariance;
}

struct State {
  Real density = Real(0);
  Real normalized[15]{};
  Real covariance_scale = Real(0);
  Real xx = Real(0), xy = Real(0), determinant = Real(0);
};

POPS_HD inline FanLi15PathStatus recover(const Real (&raw)[15], State& state) {
  for (Real value : raw)
    if (!std::isfinite(value))
      return FanLi15PathStatus::NonFiniteInput;
  if (!(raw[0] > Real(0)))
    return FanLi15PathStatus::NonPositiveDensity;
  state.density = raw[0];
  for (int k = 0; k < 15; ++k) {
    state.normalized[k] = raw[k] / raw[0];
    if (!std::isfinite(state.normalized[k]))
      return FanLi15PathStatus::NonFiniteNormalization;
  }
  const Real u = state.normalized[1], v = state.normalized[5];
  const Real a = std::fma(-u, u, state.normalized[2]);
  const Real b = std::fma(-u, v, state.normalized[6]);
  const Real c = std::fma(-v, v, state.normalized[9]);
  if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c))
    return FanLi15PathStatus::NonFiniteNormalization;
  const auto certificate = certify_raw_covariance(raw);
  if (certificate != FanLi15PathStatus::Success)
    return certificate;
  Real scale = std::fabs(a);
  if (std::fabs(b) > scale)
    scale = std::fabs(b);
  if (std::fabs(c) > scale)
    scale = std::fabs(c);
  if (!(scale > Real(0)) || !(a > Real(0)) || !(c > Real(0)))
    return FanLi15PathStatus::NonPositiveCovariance;
  const Real xx = a / scale, xy = b / scale, yy = c / scale;
  const Real determinant = std::fma(xx, yy, -xy * xy);
  if (!(determinant > Real(0)) || !std::isfinite(determinant))
    return FanLi15PathStatus::NonPositiveCovariance;
  state.covariance_scale = scale;
  state.xx = xx;
  state.xy = xy;
  state.determinant = determinant;
  return FanLi15PathStatus::Success;
}

/// C sqrt(E[(g.v)^2]); Cholesky/hypot evaluate the nonnegative quadratic form.
POPS_HD inline Real endpoint_bound(const State& state, Real gx, Real gy) {
  const Real l11 = std::sqrt(state.xx);
  const Real p = l11 * gx + (state.xy / l11) * gy;
  const Real q = std::sqrt(state.determinant / state.xx) * gy;
  const Real sigma = std::sqrt(state.covariance_scale) * std::hypot(p, q);
  const Real mean = gx * state.normalized[1] + gy * state.normalized[5];
  return std::sqrt(Real(6) + std::sqrt(Real(10))) * std::hypot(mean, sigma);
}

/// A polynomial in density weight w, truncated only above its proven degree four.
struct Poly {
  Real c[5]{};
  POPS_HD Poly() = default;
  POPS_HD explicit Poly(Real constant) { c[0] = constant; }
};

POPS_HD inline Poly add(const Poly& a, const Poly& b) {
  Poly r;
  for (int k = 0; k < 5; ++k)
    r.c[k] = a.c[k] + b.c[k];
  return r;
}
POPS_HD inline Poly scale(const Poly& a, Real value) {
  Poly r;
  for (int k = 0; k < 5; ++k)
    r.c[k] = a.c[k] * value;
  return r;
}
POPS_HD inline Poly multiply(const Poly& a, const Poly& b) {
  Poly r;
  for (int i = 0; i < 5; ++i)
    for (int j = 0; i + j < 5; ++j)
      r.c[i + j] += a.c[i] * b.c[j];
  return r;
}
POPS_HD inline Poly power(const Poly& a, int degree) {
  Poly r(Real(1));
  for (int k = 0; k < degree; ++k)
    r = multiply(r, a);
  return r;
}
POPS_HD inline Poly derivative(const Poly& a) {
  Poly r;
  for (int k = 1; k < 5; ++k)
    r.c[k - 1] = Real(k) * a.c[k];
  return r;
}
POPS_HD constexpr int factorial(int n) {
  int result = 1;
  for (int k = 2; k <= n; ++k)
    result *= k;
  return result;
}
POPS_HD constexpr int binomial(int n, int k) {
  return factorial(n) / (factorial(k) * factorial(n - k));
}
POPS_HD inline Poly coefficient(const Poly (&h)[15], int p, int q) {
  if (p < 0 || q < 0 || p + q > 4)
    return {};
  return h[Basis::index(p, q)];
}

POPS_HD inline void hermite_polynomials(const State& left, const State& right, Poly (&h)[15],
                                        Poly (&theta)[3], Poly& u, Poly& v) {
  Poly raw[15];
  for (int k = 0; k < 15; ++k) {
    raw[k].c[0] = left.normalized[k];
    raw[k].c[1] = right.normalized[k] - left.normalized[k];
  }
  u = raw[1];
  v = raw[5];
  theta[0] = add(raw[2], scale(multiply(u, u), Real(-1)));
  theta[1] = add(raw[6], scale(multiply(u, v), Real(-1)));
  theta[2] = add(raw[9], scale(multiply(v, v), Real(-1)));
  const Poly minus_u = scale(u, Real(-1)), minus_v = scale(v, Real(-1));
  for (int q = 0; q <= 4; ++q) {
    for (int p = 0; p + q <= 4; ++p) {
      if (p + q < 3)
        continue;
      Poly central;
      for (int j = 0; j <= q; ++j) {
        for (int i = 0; i <= p; ++i) {
          const Poly term = multiply(multiply(power(minus_u, p - i), power(minus_v, q - j)),
                                     raw[Basis::index(i, j)]);
          central = add(central, scale(term, Real(binomial(p, i) * binomial(q, j))));
        }
      }
      if (p + q == 4) {
        Poly gaussian;
        if (p == 4)
          gaussian = scale(multiply(theta[0], theta[0]), Real(3));
        if (p == 3)
          gaussian = scale(multiply(theta[0], theta[1]), Real(3));
        if (p == 2)
          gaussian =
              add(multiply(theta[0], theta[2]), scale(multiply(theta[1], theta[1]), Real(2)));
        if (p == 1)
          gaussian = scale(multiply(theta[1], theta[2]), Real(3));
        if (p == 0)
          gaussian = scale(multiply(theta[2], theta[2]), Real(3));
        central = add(central, scale(gaussian, Real(-1)));
      }
      h[Basis::index(p, q)] = scale(central, Real(1) / Real(factorial(p) * factorial(q)));
    }
  }
}

/// Return I_k/density_scale without reciprocals of possibly tiny densities.
/// Canonical orientation high>=low implies a0>0,b0>=0. The two branches are
/// algebraically identical to integral w^k/(a0+b0*w) dw. No tunable tolerance.
POPS_HD inline bool density_integrals(Real high, Real low, Real (&weight)[5], Real& density_scale) {
  const Real ratio = low / high;
  if (ratio >= Real(2) / Real(3)) {
    density_scale = high;
    const Real delta = (Real(1) - ratio) / ratio;
    for (int k = 0; k < 5; ++k) {
      Real sum = Real(0), compensation = Real(0), power_delta = Real(1);
      bool converged = false;
      for (int j = 0; j < 128; ++j) {
        const Real term = power_delta / Real(k + j + 1);
        const Real corrected = term - compensation;
        const Real next = sum + corrected;
        compensation = (next - sum) - corrected;
        sum = next;
        power_delta *= -delta;
        // Alternating decreasing terms: the next term bounds the entire tail.
        const Real tail = std::fabs(power_delta) / Real(k + j + 2);
        if (tail <= std::numeric_limits<Real>::epsilon() * std::fabs(sum) / Real(4)) {
          converged = true;
          break;
        }
      }
      if (!converged || !std::isfinite(sum) || !(sum > Real(0)))
        return false;
      weight[k] = sum;
    }
  } else {
    density_scale = low;
    const Real denominator = Real(1) - ratio;
    // log1p avoids the log of a ratio near one; separate logs cover a ratio
    // that underflows although both finite densities and the integral exist.
    const Real logarithm =
        ratio >= Real(0.5) ? -std::log1p(ratio - Real(1))
                           : (ratio > Real(0) ? -std::log(ratio) : std::log(high) - std::log(low));
    if (!std::isfinite(logarithm))
      return false;
    weight[0] = logarithm / denominator;
    for (int k = 1; k < 5; ++k)
      weight[k] = (Real(1) / Real(k) - ratio * weight[k - 1]) / denominator;
  }
  for (Real value : weight)
    if (!std::isfinite(value) || !(value > Real(0)))
      return false;
  return true;
}

}  // namespace fan_li15_detail

POPS_HD inline FanLi15PathStatus fan_li15_admissibility(const Real (&raw)[15]) {
  fan_li15_detail::State state;
  return fan_li15_detail::recover(raw, state);
}

/// Straight COMPLETE raw-state path, with fixed Cartesian/mapped covector g.
/// No metric/time factors are applied. The caller supplies -P/2 to BOTH cell
/// residuals, with their own measures; P is not a conservative face transfer.
/// In particular coarse-side AMR P must sum the canonical fine-subface paths.
/// Fan--Li (2014), arXiv:1401.4639, (4.16)/(5.9); first-order path contract.
POPS_HD inline FanLi15PathResult fan_li15_path_integral(const Real (&raw_left)[15],
                                                        const Real (&raw_right)[15], Real gx,
                                                        Real gy) {
  using namespace fan_li15_detail;
  FanLi15PathResult result;
  State left, right;
  auto status = recover(raw_left, left);
  if (status != FanLi15PathStatus::Success) {
    result.status = status;
    result.input_side = 0;
    return result;
  }
  status = recover(raw_right, right);
  if (status != FanLi15PathStatus::Success) {
    result.status = status;
    result.input_side = 1;
    return result;
  }
  if (!std::isfinite(gx) || !std::isfinite(gy)) {
    result.status = FanLi15PathStatus::NonFiniteDirection;
    return result;
  }
  const Real speed_left = endpoint_bound(left, gx, gy), speed_right = endpoint_bound(right, gx, gy);
  const Real speed = speed_left > speed_right ? speed_left : speed_right;
  if (!std::isfinite(speed_left) || !std::isfinite(speed_right)) {
    result.status = FanLi15PathStatus::NonFiniteResult;
    return result;
  }
  bool identical = true, reverse = false;
  // Descending lexicographic ordering starts with density. A tie-break makes
  // equal-density reversal evaluate identical polynomials, with opposite sign.
  for (int k = 0; k < 15; ++k) {
    if (raw_left[k] != raw_right[k]) {
      identical = false;
      reverse = raw_left[k] < raw_right[k];
      break;
    }
  }
  if (identical || (gx == Real(0) && gy == Real(0))) {
    result.status = FanLi15PathStatus::Success;
    result.speed_bound = speed;
    return result;
  }
  const State& first = reverse ? right : left;
  const State& second = reverse ? left : right;
  Real weight[5], density_scale;
  if (!density_integrals(first.density, second.density, weight, density_scale))
    return result;
  Poly h[15], theta[3], u, v;
  hermite_polynomials(first, second, h, theta, u, v);
  const Poly du = derivative(u), dv = derivative(v);
  const Poly da = derivative(theta[0]), db = derivative(theta[1]), dc = derivative(theta[2]);
  Real candidate[15]{};
  for (int q = 0; q <= 4; ++q) {
    const int p = 4 - q;
    Poly rx = add(multiply(coefficient(h, p, q), du), multiply(coefficient(h, p + 1, q - 1), dv));
    rx = add(rx, scale(multiply(coefficient(h, p - 1, q), da), Real(0.5)));
    rx = add(rx, multiply(coefficient(h, p, q - 1), db));
    rx = add(rx, scale(multiply(coefficient(h, p + 1, q - 2), dc), Real(0.5)));
    Poly ry = add(multiply(coefficient(h, p - 1, q + 1), du), multiply(coefficient(h, p, q), dv));
    ry = add(ry, scale(multiply(coefficient(h, p - 2, q + 1), da), Real(0.5)));
    ry = add(ry, multiply(coefficient(h, p - 1, q), db));
    ry = add(ry, scale(multiply(coefficient(h, p, q - 1), dc), Real(0.5)));
    const Poly integrand = add(scale(rx, Real(p + 1) * gx), scale(ry, Real(q + 1) * gy));
    Real sum = Real(0), compensation = Real(0);
    for (int k = 0; k < 5; ++k) {
      const Real term = integrand.c[k] * weight[k];
      const Real corrected = term - compensation, next = sum + corrected;
      compensation = (next - sum) - corrected;
      sum = next;
    }
    candidate[Basis::index(p, q)] =
        (reverse ? Real(1) : Real(-1)) * Real(factorial(p) * factorial(q)) * sum * density_scale;
    if (!std::isfinite(candidate[Basis::index(p, q)])) {
      result.status = FanLi15PathStatus::NonFiniteResult;
      return result;
    }
  }
  for (int k = 0; k < 15; ++k)
    result.integral[k] = candidate[k];
  result.status = FanLi15PathStatus::Success;
  result.speed_bound = speed;
  return result;
}

}  // namespace pops::moments
