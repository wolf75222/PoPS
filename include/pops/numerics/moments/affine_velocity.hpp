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

enum class AffineVelocityRotation { cayley, exponential };

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

/// Exact centered gyro phase for the represented inputs; theta_dt is half the interval.
/// The full interval and its product with Omega must be finite. No Cayley fallback
/// or approximate 2*pi reduction is used when the exponential range is exceeded.
POPS_HD inline bool exponential_rotation(Real omega, Real theta_dt, Real& c, Real& s) {
  if (!std::isfinite(omega) || !std::isfinite(theta_dt) || theta_dt < Real(0))
    return false;
  if (omega == Real(0) || theta_dt == Real(0)) {
    c = Real(1);
    s = Real(0);
    return true;
  }
  const Real dt = Real(2) * theta_dt;
  const Real high = omega * dt;
  if (!std::isfinite(dt) || !std::isfinite(high))
    return false;
  // Preserve the product's low part before library trig performs argument reduction.
  // Reassociation of the fused residual into omega*dt-high would discard this part.
  const Real low = std::fma(omega, dt, -high);
  const Real ch = std::cos(high), sh = std::sin(high);
  const Real cl = std::cos(low), sl = std::sin(low);
  c = std::fma(ch, cl, -sh * sl);
  s = std::fma(sh, cl, ch * sl);
  return std::isfinite(c) && std::isfinite(s);
}

/// Push one common affine velocity map through every moment. The supplied mean
/// endpoint is the result of the caller's coupled first-moment/field solve.
/// theta_dt is one half of that source interval (Crank-Nicolson).
///
/// By default R=Cayley(theta_dt*Omega*J), d=u_endpoint-R*u_old. Thus this is the
/// push-forward of a positive velocity measure, not independent CN solves of
/// moment-degree blocks. Density and both first moments are copied exactly.
/// The opt-in exponential R=exp(2*theta_dt*Omega*J) gives exact centered moments
/// for a cell-homogeneous source with constant Omega. The supplied mean remains
/// approximate when obtained by CN; this is not a full exact or AP source method.
/// A common positive metric factor on all moments (for example r*M) is legal.
/// No density/covariance floor or realizability repair is performed.
template <int Order, AffineVelocityRotation Rotation = AffineVelocityRotation::cayley>
POPS_HD inline bool affine_velocity_push_forward(
    const Real (&old)[CartesianMomentBasis<Order>::size],
    const Real (&endpoint)[CartesianMomentBasis<Order>::size],
    Real omega, Real theta_dt,
    Real (&output)[CartesianMomentBasis<Order>::size]) {
  static_assert(Rotation == AffineVelocityRotation::cayley ||
                Rotation == AffineVelocityRotation::exponential,
                "unsupported affine velocity rotation");
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
  if constexpr (Rotation == AffineVelocityRotation::cayley) {
    if (!cayley_rotation(omega, theta_dt, c, s))
      return false;
  } else {
    if (!exponential_rotation(omega, theta_dt, c, s))
      return false;
  }
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
