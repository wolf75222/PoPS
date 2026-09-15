#pragma once

#include <pops/numerics/moments/fan_li15_path.hpp>

namespace pops::moments {

/// Shared conservative transfer, to be applied with opposite cell signs.
struct FanLi15ConservativeFlux {
  Real values[15]{};
};

/// A side residual source, not a shared transfer and not a flux-register entry.
struct FanLi15NcpResidual {
  Real values[15]{};
};

struct FanLi15GradFluxResult {
  FanLi15PathStatus status = FanLi15PathStatus::IntegralFailure;
  /// Conservative Grad part only; this alone is not the Fan--Li system.
  FanLi15ConservativeFlux flux;
  POPS_HD bool succeeded() const { return status == FanLi15PathStatus::Success; }
};

struct FanLi15InterfaceResult {
  FanLi15PathStatus status = FanLi15PathStatus::IntegralFailure;
  FanLi15ConservativeFlux conservative_flux;
  FanLi15NcpResidual left_ncp, right_ncp;
  Real speed_bound = Real(0);
  int input_side = -1;
  POPS_HD bool succeeded() const { return status == FanLi15PathStatus::Success; }
};

namespace fan_li15_interface_detail {

POPS_HD constexpr int fifth_index(int p, int q) {
  return q * 6 - q * (q - 1) / 2 + p;
}

/// Endpoint coefficients f_alpha/rho, not standardized Hermite coefficients.
POPS_HD inline void normalized_hermite(const fan_li15_detail::State& state, Real (&h)[15],
                                       Real (&temperature)[3]) {
  using namespace fan_li15_detail;
  const Real u = state.normalized[1], v = state.normalized[5];
  temperature[0] = std::fma(-u, u, state.normalized[2]);
  temperature[1] = std::fma(-u, v, state.normalized[6]);
  temperature[2] = std::fma(-v, v, state.normalized[9]);
  Real minus_u[5]{Real(1)}, minus_v[5]{Real(1)};
  for (int k = 1; k < 5; ++k) {
    minus_u[k] = -u * minus_u[k - 1];
    minus_v[k] = -v * minus_v[k - 1];
  }
  for (int q = 0; q <= 4; ++q) {
    for (int p = 0; p + q <= 4; ++p) {
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

/// The Gaussian integration-by-parts identity closes degree five with f5=0:
/// M_alpha/rho = sum_beta (f_beta/rho) alpha!/(alpha-beta)! G_(alpha-beta).
/// It retains the full covariance and all degree-three/four coefficients.
POPS_HD inline void normalized_fifth_moments(const fan_li15_detail::State& state,
                                             Real (&fifth)[6]) {
  using namespace fan_li15_detail;
  Real h[15]{}, temperature[3];
  normalized_hermite(state, h, temperature);
  const Real u = state.normalized[1], v = state.normalized[5];
  const Real a = temperature[0], b = temperature[1], c = temperature[2];
  Real gaussian[21]{};
  gaussian[0] = Real(1);
  for (int q = 0; q <= 5; ++q) {
    for (int p = 0; p + q <= 5; ++p) {
      if (p + q == 0)
        continue;
      Real moment = Real(0);
      if (p > 0) {
        moment = u * gaussian[fifth_index(p - 1, q)];
        if (p > 1)
          moment += Real(p - 1) * a * gaussian[fifth_index(p - 2, q)];
        if (q > 0)
          moment += Real(q) * b * gaussian[fifth_index(p - 1, q - 1)];
      } else {
        moment = v * gaussian[fifth_index(0, q - 1)];
        if (q > 1)
          moment += Real(q - 1) * c * gaussian[fifth_index(0, q - 2)];
      }
      gaussian[fifth_index(p, q)] = moment;
    }
  }
  for (int q = 0; q <= 5; ++q) {
    const int p = 5 - q;
    Real moment = gaussian[fifth_index(p, q)];
    for (int j = 0; j <= q && j <= 4; ++j) {
      for (int i = 0; i <= p && i + j <= 4; ++i) {
        if (i + j < 3)
          continue;
        const int multiplicity = factorial(p) / factorial(p - i) * factorial(q) / factorial(q - j);
        moment +=
            h[Basis::index(i, j)] * (Real(multiplicity) * gaussian[fifth_index(p - i, q - j)]);
      }
    }
    fifth[q] = moment;
  }
}

}  // namespace fan_li15_interface_detail

/// Physical/mapped Grad endpoint flux for ONE fixed covector g. Consumers
/// must also include the Fan--Li nonconservative product. No cell measure,
/// metric Jacobian or time factor is silently introduced by this function.
POPS_HD inline FanLi15GradFluxResult fan_li15_grad_directional_flux(const Real (&raw)[15], Real gx,
                                                                    Real gy) {
  using namespace fan_li15_detail;
  FanLi15GradFluxResult result;
  State state;
  result.status = recover(raw, state);
  if (result.status != FanLi15PathStatus::Success)
    return result;
  if (!std::isfinite(gx) || !std::isfinite(gy)) {
    result.status = FanLi15PathStatus::NonFiniteDirection;
    return result;
  }
  if (gx == Real(0) && gy == Real(0))
    return result;
  Real fifth[6];
  fan_li15_interface_detail::normalized_fifth_moments(state, fifth);
  Real candidate[15]{};
  for (int q = 0; q <= 4; ++q) {
    for (int p = 0; p + q <= 4; ++p) {
      const Real flux = p + q < 4
                            ? gx * raw[Basis::index(p + 1, q)] + gy * raw[Basis::index(p, q + 1)]
                            : state.density * (gx * fifth[q] + gy * fifth[q + 1]);
      if (!std::isfinite(flux)) {
        result.status = FanLi15PathStatus::NonFiniteResult;
        return result;
      }
      candidate[Basis::index(p, q)] = flux;
    }
  }
  for (int k = 0; k < 15; ++k)
    result.flux.values[k] = candidate[k];
  return result;
}

/// FirstOrder Rusanov/DLM interface oriented left -> right with common g.
/// F*=(F_g(L)+F_g(R)-a(R-L))/2; each separate NCP side residual is -P_g/2.
/// The caller multiplies by face/cell measures: left gets -F*+left_ncp,
/// right gets +F*+right_ncp. Only F* belongs to a conservative flux ledger.
/// Coarse NCP residuals must be replaced by sums of canonical fine-subfaces.
POPS_HD inline FanLi15InterfaceResult fan_li15_rusanov_interface(const Real (&left)[15],
                                                                 const Real (&right)[15], Real gx,
                                                                 Real gy) {
  FanLi15InterfaceResult result;
  const auto path = fan_li15_path_integral(left, right, gx, gy);
  result.status = path.status;
  result.input_side = path.input_side;
  if (!path.succeeded())
    return result;
  const auto left_flux = fan_li15_grad_directional_flux(left, gx, gy);
  if (!left_flux.succeeded()) {
    result.status = left_flux.status;
    result.input_side = 0;
    return result;
  }
  const auto right_flux = fan_li15_grad_directional_flux(right, gx, gy);
  if (!right_flux.succeeded()) {
    result.status = right_flux.status;
    result.input_side = 1;
    return result;
  }
  Real conservative[15]{}, nonconservative[15]{};
  for (int k = 0; k < 15; ++k) {
    const Real dissipative = path.speed_bound == Real(0)
                                 ? Real(0)
                                 : (Real(0.5) * path.speed_bound) * (right[k] - left[k]);
    conservative[k] =
        Real(0.5) * left_flux.flux.values[k] + Real(0.5) * right_flux.flux.values[k] - dissipative;
    nonconservative[k] = Real(-0.5) * path.integral[k];
    if (!std::isfinite(conservative[k]) || !std::isfinite(nonconservative[k])) {
      result.status = FanLi15PathStatus::NonFiniteResult;
      result.input_side = -1;
      return result;
    }
  }
  for (int k = 0; k < 15; ++k) {
    result.conservative_flux.values[k] = conservative[k];
    result.left_ncp.values[k] = nonconservative[k];
    result.right_ncp.values[k] = nonconservative[k];
  }
  result.speed_bound = path.speed_bound;
  return result;
}

}  // namespace pops::moments
