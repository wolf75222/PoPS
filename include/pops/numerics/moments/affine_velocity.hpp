#pragma once

#include <pops/core/foundation/types.hpp>

#include <cmath>

namespace pops::moments {

/// Complete two-velocity raw moments, ordered by y degree, then x degree.
template <int Order>
struct CartesianMomentBasis {
  static_assert(Order >= 1 && Order <= 4, "affine moments support complete orders one to four");
  static constexpr int size = (Order + 1) * (Order + 2) / 2;
  POPS_HD static constexpr int index(int p, int q) {
    return q * (Order + 1) - q * (q - 1) / 2 + p;
  }
};

/// Cayley transform of Omega [[0,1],[-1,0]], avoiding overflow in Omega*theta_dt.
POPS_HD inline bool cayley_rotation(Real omega, Real theta_dt, Real& c, Real& s) {
  if (!std::isfinite(omega) || !std::isfinite(theta_dt) || theta_dt < Real(0))
    return false;
  if (omega == Real(0) || theta_dt == Real(0)) {
    c = Real(1);
    s = Real(0);
    return true;
  }
  if (std::fabs(omega) <= Real(1) / theta_dt) {
    const Real a = omega * theta_dt;
    const Real denominator = Real(1) + a * a;
    c = (Real(1) - a * a) / denominator;
    s = Real(2) * a / denominator;
  } else {
    const Real inverse = std::fabs(omega) >= theta_dt
        ? (Real(1) / omega) / theta_dt : (Real(1) / theta_dt) / omega;
    const Real denominator = Real(1) + inverse * inverse;
    c = (inverse * inverse - Real(1)) / denominator;
    s = Real(2) * inverse / denominator;
  }
  return std::isfinite(c) && std::isfinite(s);
}

/// Push one common affine velocity map through every moment. The supplied mean
/// endpoint is the result of the caller's coupled first-moment/field solve.
/// theta_dt is one half of that source interval (Crank-Nicolson).
///
/// R=Cayley(theta_dt*Omega*J), d=u_endpoint-R*u_old. Thus this is the
/// push-forward of a positive velocity measure, not independent CN solves of
/// moment-degree blocks. Density and both first moments are copied exactly.
/// A common positive metric factor on all moments (for example r*M) is legal.
/// No density/covariance floor or realizability repair is performed.
template <int Order>
POPS_HD inline bool affine_velocity_push_forward(
    const Real (&old)[CartesianMomentBasis<Order>::size],
    const Real (&endpoint)[CartesianMomentBasis<Order>::size],
    Real omega, Real theta_dt,
    Real (&output)[CartesianMomentBasis<Order>::size]) {
  using Basis = CartesianMomentBasis<Order>;
  constexpr int N = Basis::size;
  constexpr int X = Basis::index(1, 0);
  constexpr int Y = Basis::index(0, 1);
  for (int component = 0; component < N; ++component) {
    if (!std::isfinite(old[component]) || !std::isfinite(endpoint[component]))
      return false;
  }
  if (!(old[0] > Real(0)) || endpoint[0] != old[0])
    return false;
  Real c, s;
  if (!cayley_rotation(omega, theta_dt, c, s))
    return false;
  const Real ux = old[X] / old[0];
  const Real uy = old[Y] / old[0];
  const Real bx = endpoint[X] / old[0] - (c * ux + s * uy);
  const Real by = endpoint[Y] / old[0] - (-s * ux + c * uy);
  if (!std::isfinite(bx) || !std::isfinite(by))
    return false;
  Real candidate[N];
  for (int q = 0; q <= Order; ++q) {
    for (int p = 0; p + q <= Order; ++p) {
      // Coefficients of (bx+c*vx+s*vy)^p (by-s*vx+c*vy)^q.
      Real polynomial[Order + 1][Order + 1]{};
      polynomial[0][0] = Real(1);
      for (int factor = 0; factor < p + q; ++factor) {
        const bool first = factor < p;
        const Real constant = first ? bx : by;
        const Real x = first ? c : -s;
        const Real y = first ? s : c;
        Real next[Order + 1][Order + 1]{};
        for (int j = 0; j <= factor; ++j) {
          for (int i = 0; i + j <= factor; ++i) {
            const Real value = polynomial[i][j];
            next[i][j] += constant * value;
            next[i + 1][j] += x * value;
            next[i][j + 1] += y * value;
          }
        }
        for (int j = 0; j <= Order; ++j)
          for (int i = 0; i <= Order; ++i)
            polynomial[i][j] = next[i][j];
      }
      Real total = Real(0), compensation = Real(0);
      for (int j = 0; j <= p + q; ++j) {
        for (int i = 0; i + j <= p + q; ++i) {
          const Real term = polynomial[i][j] * old[Basis::index(i, j)];
          const Real corrected = term - compensation;
          const Real next = total + corrected;
          compensation = (next - total) - corrected;
          total = next;
        }
      }
      candidate[Basis::index(p, q)] = total;
      if (!std::isfinite(total))
        return false;
    }
  }
  candidate[0] = old[0];
  candidate[X] = endpoint[X];
  candidate[Y] = endpoint[Y];
  for (int component = 0; component < N; ++component)
    output[component] = candidate[component];
  return true;
}

}  // namespace pops::moments
