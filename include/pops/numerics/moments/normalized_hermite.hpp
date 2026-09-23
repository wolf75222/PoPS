#pragma once

#include <pops/numerics/moments/raw_moment_recovery.hpp>
#include <pops/numerics/moments/density_path_arithmetic.hpp>

namespace pops::moments {

POPS_HD constexpr int factorial(int n) {
  int result = 1;
  for (int k = 2; k <= n; ++k)
    result *= k;
  return result;
}
POPS_HD constexpr int binomial(int n, int k) {
  return factorial(n) / (factorial(k) * factorial(n - k));
}
template <int Order>
POPS_HD inline Polynomial<Order> hermite_coefficient(
    const Polynomial<Order> (&h)[CartesianMomentBasis<Order>::size], int p, int q) {
  using Basis = CartesianMomentBasis<Order>;
  if (p < 0 || q < 0 || p + q > Order)
    return {};
  return h[Basis::index(p, q)];
}

template <int Order>
POPS_HD inline void normalized_hermite_path(
    const NormalizedRawMoments<Order>& left, const NormalizedRawMoments<Order>& right,
    Polynomial<Order> (&h)[CartesianMomentBasis<Order>::size], Polynomial<Order> (&theta)[3],
    Polynomial<Order>& u, Polynomial<Order>& v) {
  using Basis = CartesianMomentBasis<Order>;
  using Poly = Polynomial<Order>;
  for (int k = 0; k < Basis::size; ++k)
    h[k] = Poly{};
  h[0] = Poly(Real(1));
  Poly raw[Basis::size];
  for (int k = 0; k < Basis::size; ++k) {
    raw[k].c[0] = left.normalized[k];
    raw[k].c[1] = right.normalized[k] - left.normalized[k];
  }
  u = raw[Basis::index(1, 0)];
  v = raw[Basis::index(0, 1)];
  theta[0] = add(raw[Basis::index(2, 0)], scale(multiply(u, u), Real(-1)));
  theta[1] = add(raw[Basis::index(1, 1)], scale(multiply(u, v), Real(-1)));
  theta[2] = add(raw[Basis::index(0, 2)], scale(multiply(v, v), Real(-1)));
  const Poly minus_u = scale(u, Real(-1)), minus_v = scale(v, Real(-1));
  for (int q = 0; q <= Order; ++q) {
    for (int p = 0; p + q <= Order; ++p) {
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

/// Endpoint coefficients f_alpha/rho, not standardized Hermite coefficients.
template <int Order>
POPS_HD inline void normalized_hermite(const NormalizedRawMoments<Order>& state,
                                       Real (&h)[CartesianMomentBasis<Order>::size],
                                       Real (&temperature)[3]) {
  using Basis = CartesianMomentBasis<Order>;
  for (Real& coefficient : h)
    coefficient = Real(0);
  const Real u = state.normalized[Basis::index(1, 0)], v = state.normalized[Basis::index(0, 1)];
  temperature[0] = std::fma(-u, u, state.normalized[Basis::index(2, 0)]);
  temperature[1] = std::fma(-u, v, state.normalized[Basis::index(1, 1)]);
  temperature[2] = std::fma(-v, v, state.normalized[Basis::index(0, 2)]);
  Real minus_u[Order + 1]{Real(1)}, minus_v[Order + 1]{Real(1)};
  for (int k = 1; k <= Order; ++k) {
    minus_u[k] = -u * minus_u[k - 1];
    minus_v[k] = -v * minus_v[k - 1];
  }
  for (int q = 0; q <= Order; ++q) {
    for (int p = 0; p + q <= Order; ++p) {
      if (p + q < 3)
        continue;
      Real central = Real(0), compensation = Real(0);
      for (int j = 0; j <= q; ++j) {
        for (int i = 0; i <= p; ++i) {
          const Real term = Real(binomial(p, i) * binomial(q, j)) * minus_u[p - i] *
                            minus_v[q - j] * state.normalized[Basis::index(i, j)];
          const Real corrected = term - compensation, next = central + corrected;
          compensation = (next - central) - corrected;
          central = next;
        }
      }
      if (p + q == 4) {
        const Real a = temperature[0], b = temperature[1], c = temperature[2];
        Real gaussian = Real(0);
        if (p == 4)
          gaussian = Real(3) * a * a;
        if (p == 3)
          gaussian = Real(3) * a * b;
        if (p == 2)
          gaussian = a * c + Real(2) * b * b;
        if (p == 1)
          gaussian = Real(3) * b * c;
        if (p == 0)
          gaussian = Real(3) * c * c;
        central -= gaussian;
      }
      h[Basis::index(p, q)] = central / Real(factorial(p) * factorial(q));
    }
  }
  h[0] = Real(1);
}

template <int Order>
POPS_HD constexpr int edge_moment_index(int p, int q) {
  return q * (Order + 1) - q * (q - 1) / 2 + p;
}

/// Gaussian integration by parts converts supplied Hermite coefficients to a raw edge:
/// M_alpha/rho = sum_beta (f_beta/rho) alpha!/(alpha-beta)! G_(alpha-beta).
/// Missing coefficients above CoefficientOrder are zero by the caller's closure declaration.
/// This transform does not choose which closure a physical model uses.
template <int CoefficientOrder, int EdgeOrder>
POPS_HD inline void hermite_raw_edge(const NormalizedRawMoments<CoefficientOrder>& state,
                                     const Real (&h)[CartesianMomentBasis<CoefficientOrder>::size],
                                     const Real (&temperature)[3], Real (&edge)[EdgeOrder + 1]) {
  using Basis = CartesianMomentBasis<CoefficientOrder>;
  static_assert(EdgeOrder > CoefficientOrder && EdgeOrder <= 12,
                "raw Hermite edge requires supported factorials through degree twelve");
  const Real u = state.normalized[Basis::index(1, 0)], v = state.normalized[Basis::index(0, 1)];
  const Real a = temperature[0], b = temperature[1], c = temperature[2];
  Real gaussian[(EdgeOrder + 1) * (EdgeOrder + 2) / 2]{};
  gaussian[0] = Real(1);
  for (int q = 0; q <= EdgeOrder; ++q) {
    for (int p = 0; p + q <= EdgeOrder; ++p) {
      if (p + q == 0)
        continue;
      Real moment = Real(0);
      if (p > 0) {
        moment = u * gaussian[edge_moment_index<EdgeOrder>(p - 1, q)];
        if (p > 1)
          moment += Real(p - 1) * a * gaussian[edge_moment_index<EdgeOrder>(p - 2, q)];
        if (q > 0)
          moment += Real(q) * b * gaussian[edge_moment_index<EdgeOrder>(p - 1, q - 1)];
      } else {
        moment = v * gaussian[edge_moment_index<EdgeOrder>(0, q - 1)];
        if (q > 1)
          moment += Real(q - 1) * c * gaussian[edge_moment_index<EdgeOrder>(0, q - 2)];
      }
      gaussian[edge_moment_index<EdgeOrder>(p, q)] = moment;
    }
  }
  for (int q = 0; q <= EdgeOrder; ++q) {
    const int p = EdgeOrder - q;
    Real moment = gaussian[edge_moment_index<EdgeOrder>(p, q)];
    for (int j = 0; j <= q && j <= CoefficientOrder; ++j) {
      for (int i = 0; i <= p && i + j <= CoefficientOrder; ++i) {
        if (i + j < 3)
          continue;
        const int multiplicity = factorial(p) / factorial(p - i) * factorial(q) / factorial(q - j);
        moment += h[Basis::index(i, j)] *
                  (Real(multiplicity) * gaussian[edge_moment_index<EdgeOrder>(p - i, q - j)]);
      }
    }
    edge[q] = moment;
  }
}

}  // namespace pops::moments
