#pragma once

#include <pops/core/foundation/types.hpp>
#include <cmath>
#include <limits>

namespace pops::moments {

/// Fixed-capacity coefficient algebra. The generated caller proves its degree bound;
/// coefficients above Degree must be identically zero before this arithmetic is selected.
template <int Degree>
struct Polynomial {
  Real c[Degree + 1]{};
  POPS_HD Polynomial() = default;
  POPS_HD explicit Polynomial(Real constant) { c[0] = constant; }
};

template <int Degree>
POPS_HD inline Polynomial<Degree> add(const Polynomial<Degree>& a, const Polynomial<Degree>& b) {
  Polynomial<Degree> r;
  for (int k = 0; k <= Degree; ++k)
    r.c[k] = a.c[k] + b.c[k];
  return r;
}
template <int Degree>
POPS_HD inline Polynomial<Degree> scale(const Polynomial<Degree>& a, Real value) {
  Polynomial<Degree> r;
  for (int k = 0; k <= Degree; ++k)
    r.c[k] = a.c[k] * value;
  return r;
}
template <int Degree>
POPS_HD inline Polynomial<Degree> multiply(const Polynomial<Degree>& a,
                                           const Polynomial<Degree>& b) {
  Polynomial<Degree> r;
  for (int i = 0; i <= Degree; ++i)
    for (int j = 0; i + j <= Degree; ++j)
      r.c[i + j] += a.c[i] * b.c[j];
  return r;
}
template <int Degree>
POPS_HD inline Polynomial<Degree> power(const Polynomial<Degree>& a, int degree) {
  Polynomial<Degree> r(Real(1));
  for (int k = 0; k < degree; ++k)
    r = multiply(r, a);
  return r;
}
template <int Degree>
POPS_HD inline Polynomial<Degree> derivative(const Polynomial<Degree>& a) {
  Polynomial<Degree> r;
  for (int k = 1; k <= Degree; ++k)
    r.c[k - 1] = Real(k) * a.c[k];
  return r;
}
/// Return I_k/density_scale without reciprocals of possibly tiny densities.
/// Canonical orientation high>=low implies a0>0,b0>=0. The two branches are
/// algebraically identical to integral w^k/(a0+b0*w) dw. No tunable tolerance.
template <int Count>
POPS_HD inline bool density_path_weights(Real high, Real low, Real (&weight)[Count],
                                         Real& density_scale) {
  static_assert(Count > 0);
  const Real ratio = low / high;
  if (ratio >= Real(2) / Real(3)) {
    density_scale = high;
    const Real delta = (Real(1) - ratio) / ratio;
    for (int k = 0; k < Count; ++k) {
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
    for (int k = 1; k < Count; ++k)
      weight[k] = (Real(1) / Real(k) - ratio * weight[k - 1]) / denominator;
  }
  for (Real value : weight)
    if (!std::isfinite(value) || !(value > Real(0)))
      return false;
  return true;
}

}  // namespace pops::moments
