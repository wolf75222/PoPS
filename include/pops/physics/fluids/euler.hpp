#pragma once

/// @file
/// @brief 2D compressible Euler model (ideal gas): pure HYPERBOLIC brick satisfying
///        the HyperbolicPhysicalModel concept. Source and elliptic right-hand side are
///        separate bricks (physics/source.hpp, physics/elliptic.hpp); this file contains only
///        Vars + flux + wave speeds + cons<->prim conversions.

#include <pops/core/state/state.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/state/variables.hpp>
#include <pops/runtime/numerical_defaults.hpp>

#include <cmath>

namespace pops {

/**
 * 2D compressible Euler for an ideal gas: HYPERBOLIC brick (HyperbolicModel concept).
 *
 * Conservative variables U = (rho, rho u, rho v, E), with
 * E = p/(gamma-1) + 1/2 rho (u^2 + v^2) and p = (gamma-1)(E - 1/2 rho |v|^2). The
 * directional flux is F_x = (rho u, rho u^2 + p, rho u v, (E+p) u) and symmetrically in y;
 * the maximum wave speed is |v_dir| + c with c = sqrt(gamma p/rho).
 *
 * Pure HYPERBOLIC brick: variables (cons U, prim P) + conversions + flux + wave speeds.
 * NO source or elliptic right-hand side here: those are SEPARATE bricks, assembled by
 * CompositeModel. The aux argument is present for the contract (a drift transport reads grad
 * phi) but does not enter the Euler flux.
 *
 * @note Everything is device-callable (POPS_HD): StateVec over a C array, std::sqrt
 *       (device intrinsic under nvcc), manual abs. Compatible with a GPU kernel like the
 *       scalar transport model.
 */
struct Euler {
  using State = StateVec<4>;        ///< conservative variables (rho, rho u, rho v, E)
  using Prim = StateVec<4>;         ///< primitive variables (rho, u, v, p)
  using Aux = pops::Aux;            ///< auxiliary fields (unused in pure Euler)
  static constexpr int n_vars = 4;  ///< number of conserved variables

  Real gamma = kPhysicalDefaultGamma;  ///< adiabatic index of the ideal gas

  /// Ideal-gas pressure p = (gamma-1)(E - 1/2 rho |v|^2).
  POPS_HD Real pressure(const State& u) const {
    const Real rho = u[0];
    const Real ke = Real(0.5) * (u[1] * u[1] + u[2] * u[2]) / rho;
    return (gamma - Real(1)) * (u[3] - ke);
  }
  /// Sound speed c = sqrt(gamma p / rho).
  POPS_HD Real sound_speed(const State& u) const { return std::sqrt(gamma * pressure(u) / u[0]); }

  /// Conservative -> primitive: (rho, rho u, rho v, E) -> (rho, u, v, p).
  POPS_HD Prim to_primitive(const State& u) const {
    const Real rho = u[0];
    Prim p{};
    p[0] = rho;
    p[1] = u[1] / rho;
    p[2] = u[2] / rho;
    p[3] = pressure(u);
    return p;
  }
  /// Primitive -> conservative: (rho, u, v, p) -> (rho, rho u, rho v, E).
  POPS_HD State to_conservative(const Prim& p) const {
    const Real rho = p[0];
    State u{};
    u[0] = rho;
    u[1] = rho * p[1];
    u[2] = rho * p[2];
    u[3] = p[3] / (gamma - Real(1)) + Real(0.5) * rho * (p[1] * p[1] + p[2] * p[2]);
    return u;
  }

  /**
   * Extreme signed wave speeds in direction dir: v_dir - c and v_dir + c.
   *
   * Required by the HLL/HLLC fluxes, beyond the single max_wave_speed that Rusanov needs.
   *
   * @param      u    conservative state
   * @param      dir  face direction (0 = x, 1 = y)
   * @param[out] smin leftmost wave speed v_dir - c
   * @param[out] smax rightmost wave speed v_dir + c
   */
  POPS_HD void wave_speeds(const State& u, const Aux&, int dir, Real& smin, Real& smax) const {
    const Prim p = to_primitive(u);
    const Real vn = (dir == 0 ? p[1] : p[2]);
    const Real c = std::sqrt(gamma * p[3] / p[0]);
    smin = vn - c;
    smax = vn + c;
  }

  /// Compressible convective flux in direction dir.
  POPS_HD State flux(const State& u, const Aux&, int dir) const {
    const Real rho = u[0];
    const Real vn = (dir == 0 ? u[1] : u[2]) / rho;  // velocity normal to the face
    const Real p = pressure(u);
    State f{};
    f[0] = rho * vn;
    f[1] = u[1] * vn + (dir == 0 ? p : Real(0));
    f[2] = u[2] * vn + (dir == 1 ? p : Real(0));
    f[3] = (u[3] + p) * vn;
    return f;
  }

  // -------------------------------------------------------------------------------------------
  // RIEMANN CAPABILITIES (ADC-590): the native Euler brick now PROVIDES the HLLC / Roe physical
  // structure so HLLCFlux / RoeFlux take their GENERIC path (HasHLLCStructure / HasRoeDissipation)
  // instead of a hidden Euler fallback. The arithmetic is copied VERBATIM from the historical
  // canonical Euler 2D branches of numerical_flux.hpp (Toro 10.37 star speed, the fac/Us star
  // construction, the sqrt-rho Roe average + 3-wave eigenstructure + Harten eps = 0.1 c
  // dissipation) so the trait path is BIT-IDENTICAL to the former fallback -- the AMR riemann /
  // spatial parity oracles prove it. These same formulas are what the DSL emits from the fluid
  // roles for a 4-var Euler (module_emit_riemann.py), so a native Euler and a DSL enable_hllc /
  // enable_roe Euler agree ulp-for-ulp.
  // -------------------------------------------------------------------------------------------

  /// HLLC contact wave speed s* (Toro eq. 10.37). Canonical Euler 2D layout (rho, m_x, m_y, E):
  /// the normal momentum is component 1 (dir == 0) or 2 (dir == 1).
  POPS_HD Real contact_speed(const State& UL, const State& UR, Real pL, Real pR, Real sL, Real sR,
                             int dir) const {
    const int in = (dir == 0) ? 1 : 2;  // normal momentum component
    const Real rL = UL[0], rR = UR[0];
    const Real unL = UL[in] / rL, unR = UR[in] / rR;
    return (pR - pL + rL * unL * (sL - unL) - rR * unR * (sR - unR)) /
           (rL * (sL - unL) - rR * (sR - unR));
  }

  /// HLLC star state U*_k on side k (Toro): fac = r (s - u_n) / (s - s*), then the canonical Euler
  /// 2D construction (density fac, normal momentum fac s*, tangential fac u_t, energy fac(...)).
  POPS_HD State hllc_star_state(const State& U, Real p, Real s, Real sStar, int dir) const {
    const int in = (dir == 0) ? 1 : 2;  // normal momentum
    const int it = (dir == 0) ? 2 : 1;  // tangential momentum
    const Real r = U[0];
    const Real un = U[in] / r;
    const Real fac = r * (s - un) / (s - sStar);
    State Us{};
    Us[0] = fac;
    Us[in] = fac * sStar;
    Us[it] = fac * (U[it] / r);
    Us[3] = fac * (U[3] / r + (sStar - un) * (sStar + p / (r * (s - un))));
    return Us;
  }

  /// Roe dissipation d = |A_roe| (U_R - U_L) for the canonical ideal-gas Euler 2D system: FULL
  /// eigenwave decomposition (F_R - F_L = A_roe (U_R - U_L) exactly), sqrt(rho) Roe average, gamma-1
  /// from the ideal-gas EOS, Harten entropy fix eps = kRoeEntropyFixFraction * c on the acoustic
  /// waves. RoeFlux (HasRoeDissipation) then does F = 1/2 (F_L + F_R) - 1/2 d.
  POPS_HD State roe_dissipation(const State& UL, const Aux&, const State& UR, const Aux&,
                                int dir) const {
    const int in = (dir == 0) ? 1 : 2;  // normal momentum
    const int it = (dir == 0) ? 2 : 1;  // tangential
    const Real rL = UL[0], rR = UR[0];
    const Real unL = UL[in] / rL, unR = UR[in] / rR;
    const Real utL = UL[it] / rL, utR = UR[it] / rR;
    const Real pL = pressure(UL), pR = pressure(UR);
    const Real HL = (UL[3] + pL) / rL, HR = (UR[3] + pR) / rR;

    // Roe average (weighted by sqrt(rho))
    const Real sqL = std::sqrt(rL), sqR = std::sqrt(rR), den = sqL + sqR;
    const Real un = (sqL * unL + sqR * unR) / den;
    const Real ut = (sqL * utL + sqR * utR) / den;
    const Real H = (sqL * HL + sqR * HR) / den;
    const Real rho = sqL * sqR;
    const Real q2 = un * un + ut * ut;
    // gamma-1 derived from the ideal gas: p = (gamma-1) (E - 1/2 rho |v|^2)
    const Real gm1 = pL / (UL[3] - Real(0.5) * rL * (unL * unL + utL * utL));
    const Real c2 = gm1 * (H - Real(0.5) * q2);
    const Real c = std::sqrt(c2);

    // wave jumps and amplitudes
    const Real dr = rR - rL, dp = pR - pL, dun = unR - unL, dut = utR - utL;
    const Real a1 = (dp - rho * c * dun) / (Real(2) * c2);  // un - c wave
    const Real a2 = dr - dp / c2;                           // entropy, un
    const Real a3 = rho * dut;                              // shear, un
    const Real a5 = (dp + rho * c * dun) / (Real(2) * c2);  // un + c wave

    // |eigenvalue| with Harten entropy fix on the acoustic waves (1, 5). eps = 0.1 c matches
    // kRoeEntropyFixFraction (numerical_flux.hpp), the Euler/Roe entropy policy.
    const Real eps = Real(0.1) * c;
    auto absfix = [eps](Real l) {
      const Real al = l < 0 ? -l : l;
      return al < eps ? Real(0.5) * (l * l / eps + eps) : al;
    };
    const Real al1 = absfix(un - c), al2 = (un < 0 ? -un : un), al5 = absfix(un + c);

    // dissipation Sum |lambda_k| a_k r_k, basis (rho, mom_n, mom_t, E)
    State d{};
    d[0] = al1 * a1 + al2 * a2 + al5 * a5;
    d[in] = al1 * a1 * (un - c) + al2 * a2 * un + al5 * a5 * (un + c);
    d[it] = al1 * a1 * ut + al2 * (a2 * ut + a3) + al5 * a5 * ut;
    d[3] =
        al1 * a1 * (H - un * c) + al2 * (a2 * Real(0.5) * q2 + a3 * ut) + al5 * a5 * (H + un * c);
    return d;
  }

  /// Full spectrum in direction dir: (v_dir - c, v_dir, v_dir, v_dir + c). Vector counterpart
  /// of wave_speeds (which only gives the signed extremes); useful for spectrum schemes (Roe).
  POPS_HD State eigenvalues(const State& u, const Aux&, int dir) const {
    const Prim p = to_primitive(u);
    const Real vn = (dir == 0 ? p[1] : p[2]);
    const Real c = std::sqrt(gamma * p[3] / p[0]);
    State e{};
    e[0] = vn - c;
    e[1] = vn;
    e[2] = vn;
    e[3] = vn + c;
    return e;
  }

  /// Maximum wave speed |v_dir| + c (Rusanov estimate), computed in primitive variables.
  POPS_HD Real max_wave_speed(const State& u, const Aux&, int dir) const {
    const Prim p = to_primitive(u);
    const Real vn = (dir == 0 ? p[1] : p[2]);
    const Real a = vn < 0 ? -vn : vn;  // |v_dir| device-safe
    return a + std::sqrt(gamma * p[3] / p[0]);
  }

  /// Variable descriptor (hyperbolic model contract; host introspection metadata).
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative,
            {"rho", "rho_u", "rho_v", "E"},
            4,
            {VariableRole::Density, VariableRole::MomentumX, VariableRole::MomentumY,
             VariableRole::Energy}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive,
            {"rho", "u", "v", "p"},
            4,
            {VariableRole::Density, VariableRole::VelocityX, VariableRole::VelocityY,
             VariableRole::Pressure}};
  }
};

}  // namespace pops
