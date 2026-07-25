/// @file
/// @brief Diagnostics extracted from the AMR couplers: mass and max drift speed (responsibility c).
///
/// Namespace-scope free functions (same reason as detail:: in coupler.hpp: GPU seam, an
/// extended lambda cannot live in a private method). Both diagnostics go through the Kokkos reducer
/// seam: the mass uses Kokkos::Sum (deterministic/idempotent but reassociated), while the drift speed
/// uses Kokkos::Max (exact). The mono-box variants reduce to the multi-box implementation. NO MPI
/// reduction here: the coupler decides whether to all_reduce according to its ownership policy.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/execution/for_each.hpp>  // device_fence
#include <pops/mesh/storage/multifab.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace pops {

namespace detail {

/// Device-clean pointwise drift-speed diagnostic. A non-finite gradient is not a physical speed:
/// publish +infinity as an explicit marker so Kokkos::Max cannot silently hide NaN. The host seam
/// below turns that marker into a clear failure. The reciprocal is prepared once on the host.
struct AmrDriftSpeedKernel {
  ConstArray4 aux;
  Real inverse_B0;

  POPS_HD Real operator()(int i, int j) const {
    const Real gx = aux(i, j, 1);
    const Real gy = aux(i, j, 2);
    const Real magnitude = Kokkos::hypot(gx, gy);
    if (!(magnitude >= Real(0)) ||
        !(magnitude < std::numeric_limits<Real>::infinity()))
      return std::numeric_limits<Real>::infinity();
    return magnitude * inverse_B0;
  }
};

inline void require_finite_amr_drift_speed(Real value) {
  if (!std::isfinite(static_cast<double>(value)))
    throw std::domain_error("AMR drift speed requires finite gradients and a finite positive B0");
}

inline void require_positive_finite_amr_spacing(Real dx, Real dy) {
  if (!(dx > Real(0)) || !(dy > Real(0)) || !std::isfinite(static_cast<double>(dx)) ||
      !std::isfinite(static_cast<double>(dy)))
    throw std::invalid_argument("AMR field gradients require finite positive grid spacing");
}

}  // namespace detail

// --- MULTI-BOX form (canonical): sum/max over the valid cells of ALL local fabs,
// WITHOUT MPI reduction (the coupler decides whether to all_reduce according to its
// ownership policy). This is the single implementation; the mono-box variants below reduce
// to it (a single fab whose box equals the domain -> bit for bit identical). This removes
// the duplication between AmrCoupler (mono-box) and AmrCouplerMP (multi-box / distributed).

// local sum of u(.,.,0) * dV over the valid cells. dV multiplied INSIDE the kernel.
/// LOCAL mass: sum of u(.,.,0) * dx * dy over the valid cells of ALL local fabs, WITHOUT
/// MPI reduction (the caller decides whether to all_reduce). Canonical multi-box form.
inline Real amr_mass_mb(const MultiFab& coarse, Real dx, Real dy) {
  const Real dV = dx * dy;
  Real M = 0;
  for (int li = 0; li < coarse.local_size(); ++li) {
    const ConstArray4 u = coarse.fab(li).const_array();
    M += for_each_cell_reduce_sum(coarse.box(li),
                                  [u, dV] POPS_HD(int i, int j) { return u(i, j, 0) * dV; });
  }
  return M;
}

// local max of |grad phi| / B0 (aux comp 1,2 = grad phi). WITHOUT floor (applied by the caller).
/// LOCAL max drift speed: max of |grad phi| / B0 (aux comp 1, 2 = grad phi) over the valid cells,
/// WITHOUT floor (applied by the caller) nor MPI reduction. Every valid cell stays on the active
/// Kokkos execution space; only one reduced scalar per local Fab returns to the host.
inline Real amr_max_drift_speed_mb(const MultiFab& aux0, Real B0) {
  if (!(B0 > Real(0)) || !std::isfinite(static_cast<double>(B0)))
    return std::numeric_limits<Real>::infinity();
  if (aux0.ncomp() < 3)
    throw std::invalid_argument("AMR drift speed requires aux components phi, grad_x and grad_y");
  const Real inverse_B0 = Real(1) / B0;
  Real v = 0;
  for (int li = 0; li < aux0.local_size(); ++li) {
    const ConstArray4 a = aux0.fab(li).const_array();
    v = std::max(v, for_each_cell_reduce_max(
                        aux0.box(li), detail::AmrDriftSpeedKernel{a, inverse_B0}));
  }
  return v;
}

// mass of component 0 on the coarse level (single box): degenerate case of
// amr_mass_mb (one fab covering the domain), bit for bit identical. dom kept for the API.
/// Mono-box mass: degenerate case of amr_mass_mb (bit for bit). @p dom is ignored (kept for the API).
inline Real amr_mass(const MultiFab& coarse, const Box2D& dom, Real dx, Real dy) {
  (void)dom;
  return amr_mass_mb(coarse, dx, dy);
}

// max drift speed on the coarse level (single box) + floor kAmrDriftSpeedFloor (CFL guard).
/// Mono-box max drift speed + floor kAmrDriftSpeedFloor (CFL guard). @p dom ignored (kept for the API).
inline Real amr_max_drift_speed(const MultiFab& aux0, const Box2D& dom, Real B0) {
  (void)dom;
  const Real speed = amr_max_drift_speed_mb(aux0, B0);
  detail::require_finite_amr_drift_speed(speed);
  return std::max(speed, kAmrDriftSpeedFloor);
}

}  // namespace pops
