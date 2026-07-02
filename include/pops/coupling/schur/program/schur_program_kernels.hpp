#pragma once

#include <pops/core/foundation/types.hpp>      // Real, POPS_HD
#include <pops/mesh/boundary/physical_bc.hpp>  // BCRec / BCType (coefficient / flux ghost policy)
#include <pops/mesh/storage/fab2d.hpp>         // Array4 / ConstArray4 (per-cell handles)
#include <pops/numerics/linalg/lorentz_eliminator.hpp>  // LorentzEliminator (closed B^{-1}, ADC-421)

/// @file
/// @brief Aux-component-aware Schur/Lorentz per-cell kernels of a compiled time Program (ADC-587).
///
/// These functors were extracted VERBATIM from
/// ``include/pops/runtime/program/program_context.hpp`` (epic ADC-399 / ADC-421) so the generic
/// runtime facade no longer carries any Schur/Lorentz/electrostatic material: a Schur-free Program's
/// generated .so must not pull in ``coupling/schur/**``. The kernel bodies are byte-identical to the
/// pre-split versions (same LorentzEliminator B^{-1}, same coefficients, same centered gradient) -- a
/// pure module move, zero numerical change. They lower the condensed-Schur operator a compiled
/// Program assembles from the live state + the System aux (see condensed_schur_operator.hpp).

namespace pops {
namespace coupling {
namespace schur {
namespace program {
namespace detail {

/// Aux-component-aware variants of the native Schur kernels (coupling/schur/core/schur_condensation.hpp
/// + condensed_schur_source_stepper.hpp). The native kernels read B_z from a DEDICATED B_z MultiFab at
/// component 0; a compiled Program reads B_z straight from the System aux channel at an arbitrary
/// component @c c_bz, so these thin wrappers carry @c c_bz and otherwise REPRODUCE the native formulas
/// verbatim (same LorentzEliminator B^{-1}, same coefficients, same centered gradient) -- the native
/// CondensedSchur path is untouched. Named functors (device-clean, nvcc cross-TU rule, like the native
/// ones). epic ADC-399 / ADC-421.

/// A_op = I + c*rho*B^{-1} per cell (eps_x/eps_y diag, a_xy/a_yx cross). Mirrors
/// detail::SchurOperatorCoeffKernel but reads B_z from the aux at c_bz.
struct SchurOperatorCoeffKernelC {
  ConstArray4 s;    ///< fluid state (rho at c_rho)
  ConstArray4 aux;  ///< System aux (B_z at c_bz)
  Array4 ex, ey;    ///< output: eps_x, eps_y (diagonal of A)
  Array4 axy, ayx;  ///< output: cross terms a_xy, a_yx
  Real c;           ///< c = theta^2 dt^2 alpha
  Real th_dt;       ///< theta*dt (w = th_dt*B_z)
  int c_rho, c_bz;
  POPS_HD void operator()(int i, int j) const {
    const Real rho = s(i, j, c_rho);
    const LorentzEliminator le(th_dt, Real(1), aux(i, j, c_bz));
    const Real cr = c * rho;  // c rho: common factor of the 4 entries of A - I
    ex(i, j, 0) = Real(1) + cr * le.binv_11();
    ey(i, j, 0) = Real(1) + cr * le.binv_22();
    axy(i, j, 0) = cr * le.binv_12();
    ayx(i, j, 0) = cr * le.binv_21();
  }
};

/// out = B^{-1} (mx, my) at the center (Fx in comp 0, Fy in comp 1): the explicit flux F = rho*B^{-1}*v.
struct SchurExplicitFluxKernelC {
  ConstArray4 s;    ///< fluid state (mx, my at c_mx / c_my)
  ConstArray4 aux;  ///< System aux (B_z at c_bz)
  Array4 out;       ///< output: Fx (comp 0), Fy (comp 1)
  Real th_dt;       ///< theta*dt (w = th_dt*B_z)
  int c_mx, c_my, c_bz;
  POPS_HD void operator()(int i, int j) const {
    const LorentzEliminator le(th_dt, Real(1), aux(i, j, c_bz));
    Real Fx, Fy;
    le.apply_Binv(s(i, j, c_mx), s(i, j, c_my), Fx, Fy);  // B^{-1} (mx, my) = rho*B^{-1}*v
    out(i, j, 0) = Fx;
    out(i, j, 1) = Fy;
  }
};

/// rhs = -Lap phi^n - g*div(F), the centered FV divergence of the explicit flux F packed in ONE
/// 2-component buffer (Fx in comp 0, Fy in comp 1 -- the layout schur_explicit_flux writes), fused with
/// the already-negated -Lap phi^n. Mirrors detail::SchurRhsAssembleKernel verbatim except it reads both
/// flux components from the single buffer @c f instead of two separate fx/fy MultiFabs.
struct SchurRhsAssembleKernelC {
  ConstArray4 neg_lap;      ///< -Lap phi^n (already negated)
  ConstArray4 f;            ///< explicit flux F at the center (Fx comp 0, Fy comp 1; ghosts filled)
  Array4 rhs;               ///< output: condensed right-hand side
  Real g;                   ///< theta dt alpha
  Real half_idx, half_idy;  ///< 1/(2 dx), 1/(2 dy)
  POPS_HD void operator()(int i, int j) const {
    const Real divF =
        (f(i + 1, j, 0) - f(i - 1, j, 0)) * half_idx + (f(i, j + 1, 1) - f(i, j - 1, 1)) * half_idy;
    rhs(i, j, 0) = neg_lap(i, j, 0) - g * divF;
  }
};

/// Reconstruct v^{n+theta} = B^{-1}(v^n - theta*dt*grad phi) and write mom = rho^n*v (rho frozen).
/// Mirrors detail::SchurReconstructKernel but reads B_z from the aux at c_bz (no separate vx/vy
/// buffers: v^n = (mx, my)/rho read inline from the state).
struct SchurReconstructKernelC {
  ConstArray4 phi;  ///< phi^{n+theta} (ghosts filled: centered grad reads i+-1, j+-1)
  ConstArray4 aux;  ///< System aux (B_z at c_bz)
  Array4 st;        ///< fluid state (READ rho, mx, my; WRITE mx, my)
  Real th_dt;
  Real half_idx, half_idy;  ///< 1/(2 dx), 1/(2 dy) (centered gradient)
  int c_rho, c_mx, c_my, c_bz;
  POPS_HD void operator()(int i, int j) const {
    const Real rho = st(i, j, c_rho);
    const Real inv_rho = rho != Real(0) ? Real(1) / rho : Real(0);
    const Real vx = st(i, j, c_mx) * inv_rho;  // v^n = (mx, my)/rho
    const Real vy = st(i, j, c_my) * inv_rho;
    const Real gx = (phi(i + 1, j, 0) - phi(i - 1, j, 0)) * half_idx;  // d_x phi^{n+theta}
    const Real gy = (phi(i, j + 1, 0) - phi(i, j - 1, 0)) * half_idy;
    const LorentzEliminator le(th_dt, Real(1), aux(i, j, c_bz));
    Real nx, ny;
    le.apply_Binv(vx - th_dt * gx, vy - th_dt * gy, nx, ny);  // B^{-1}(v^n - theta dt grad phi)
    st(i, j, c_mx) = rho * nx;
    st(i, j, c_my) = rho * ny;
  }
};

/// Condensed-Schur kinetic-energy increment (ADC-427), mirroring the native detail::SchurEnergyKernel:
/// E^{n+1} = E^n + (1/2) rho (|v^{n+1}|^2 - |v^n|^2), v = (mx, my)/rho. v^{n+1} is read from the updated
/// state @p st (after the velocity update + n+1 extrapolation), v^n and the base E^n from @p st_old
/// (U^n). rho is frozen in the source (same value in both states), read from @p st. The energy base E^n
/// already sits in @p st (the reconstruction / extrapolation leave the energy component untouched), so
/// the kernel ADDS the increment in place, exactly as the native stepper does.
struct SchurEnergyKernelC {
  Array4 st;           ///< updated state (READ rho, mx, my = mom^{n+1}; READ+WRITE E)
  ConstArray4 st_old;  ///< U^n (READ mx, my = mom^n)
  int c_rho, c_mx, c_my, c_E;
  POPS_HD void operator()(int i, int j) const {
    const Real rho = st(i, j, c_rho);
    const Real inv_rho = rho != Real(0) ? Real(1) / rho : Real(0);
    const Real vx_new = st(i, j, c_mx) * inv_rho;
    const Real vy_new = st(i, j, c_my) * inv_rho;
    const Real vx_old = st_old(i, j, c_mx) * inv_rho;  // rho frozen: rho^n == rho^{n+1}
    const Real vy_old = st_old(i, j, c_my) * inv_rho;
    const Real ke_new = Real(0.5) * rho * (vx_new * vx_new + vy_new * vy_new);
    const Real ke_old = Real(0.5) * rho * (vx_old * vx_old + vy_old * vy_old);
    st(i, j, c_E) += ke_new - ke_old;
  }
};

}  // namespace detail

/// BC of the coefficient / flux fields (ADC-421): periodic preserved, physical boundary -> zero-
/// gradient (Foextrap). Identical to GeometricMG::eps_bc and the native Schur coeff_bc -- the face
/// value at the domain boundary equals the interior value, consistent with the elliptic operator.
/// Extracted from ProgramContext::coeff_bc (a private static helper) so the condensed-Schur operator
/// can assemble the coefficient/flux halos without any Schur token leaking into program_context.hpp.
inline BCRec schur_coeff_bc(const BCRec& bc) {
  auto fo = [](BCType t) { return t == BCType::Periodic ? t : BCType::Foextrap; };
  BCRec b;
  b.xlo = fo(bc.xlo);
  b.xhi = fo(bc.xhi);
  b.ylo = fo(bc.ylo);
  b.yhi = fo(bc.yhi);
  return b;
}

}  // namespace program
}  // namespace schur
}  // namespace coupling
}  // namespace pops
