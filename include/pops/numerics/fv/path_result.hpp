#pragma once

#include <pops/core/foundation/types.hpp>

namespace pops {

/// Stable numerical refusal reasons; no repair or floor is implied by a status.
enum class PathStatus : unsigned char {
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

template <int N>
struct PathIntegralResult {
  static_assert(N > 0);
  PathStatus status = PathStatus::IntegralFailure;
  /// Integral of B_g(Phi) dPhi, on the LEFT of U_t+div(F)+B grad(U)=S.
  Real integral[N]{};
  /// Both endpoint majorants use the SAME fixed covector g.
  Real speed_bound = Real(0);
  /// 0 or 1 for a refused input state; -1 for a direction/integral failure.
  int input_side = -1;
  POPS_HD bool succeeded() const { return status == PathStatus::Success; }
};

/// Shared conservative transfer, to be applied with opposite cell signs.
template <int N>
struct PathConservativeFlux {
  static_assert(N > 0);
  Real values[N]{};
};

/// A side residual source, not a shared transfer and not a flux-register entry.
template <int N>
struct PathNcpResidual {
  static_assert(N > 0);
  Real values[N]{};
};

template <int N>
struct PathFluxResult {
  static_assert(N > 0);
  PathStatus status = PathStatus::IntegralFailure;
  /// Conservative part only; nonconservative side terms remain independently required.
  PathConservativeFlux<N> flux;
  POPS_HD bool succeeded() const { return status == PathStatus::Success; }
};

template <int N>
struct PathInterfaceResult {
  static_assert(N > 0);
  PathStatus status = PathStatus::IntegralFailure;
  PathConservativeFlux<N> conservative_flux;
  PathNcpResidual<N> left_ncp, right_ncp;
  Real speed_bound = Real(0);
  int input_side = -1;
  POPS_HD bool succeeded() const { return status == PathStatus::Success; }
};

}  // namespace pops
