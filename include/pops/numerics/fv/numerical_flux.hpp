/// @file
/// @brief Single-interface numerical flux policies: Rusanov, HLL, HLLC, Roe.
///
/// Each policy is a stateless POPS_HD functor satisfying the contract:
///   operator()(model, UL, AL, UR, AR, dir) -> Model::State
/// which returns the numerical flux at the interface between the left state (UL, aux AL) and the
/// right state (UR, AR) along direction dir (0 = x, 1 = y). States and auxiliaries are passed by
/// value; no virtuals.
///
/// Accuracy hierarchy (intermediate-wave resolution) AND generality:
///   RusanovFlux: minimal GENERIC; only needs max_wave_speed (any PhysicalModel).
///   HLLFlux: GENERIC with signed waves; requires model.wave_speeds (sL, sR).
///   HLLCFlux: GENERIC-ONLY contact-resolving solver; the model MUST supply HasHLLCStructure
///                  (contact_speed + hllc_star_state). No hidden Euler fallback (ADC-590).
///   RoeFlux: GENERIC-ONLY Roe-like solver; the model MUST supply HasRoeDissipation
///                  (full d = |A_roe| (UR - UL)). No hidden Euler fallback (ADC-590).
///   EulerHLLCFlux2D / EulerRoeFlux2D: the EXPLICIT canonical 2D Euler routes (n_vars == 4,
///                  rho/m/E layout, ideal-gas pressure; Roe: hard-coded eigenstructure, gamma-1 from
///                  the EOS, Harten entropy fix eps = 0.1*c). Chosen by the euler_hllc / euler_roe
///                  routes; NEVER a fallback.
///
/// For a NON-Euler model (moment system, isothermal, scalar...), the generic path is RusanovFlux,
/// or HLLFlux as soon as the model exposes wave_speeds.
///
/// RIEMANN CAPABILITIES (2026-06 audit + ADC-590): HLLC and Roe are GENERIC algorithms whose
/// PHYSICAL STRUCTURE is supplied by the model through the traits below. ADC-590 removed the
/// implicit Euler fallback: HLLCFlux / RoeFlux now static_assert without the capability, and the
/// canonical 2D Euler arithmetic lives in the EXPLICIT EulerHLLCFlux2D / EulerRoeFlux2D structs
/// (the moved historical branches, unchanged).
///   - HasHLLCStructure: the model provides contact_speed (contact wave speed) and
///     hllc_star_state (its star state). HLLCFlux then applies F* = F_k + s_k (U*_k - U_k) without
///     any layout assumption.
///   - HasRoeDissipation: the model provides its full Roe dissipation d = |A_roe| (U_R - U_L)
///     (linearization, eigenstructure and entropy fix INCLUDED, properties of the MODEL, not the
///     core). RoeFlux then does F = 1/2 (F_L + F_R) - 1/2 d.
/// The native Euler brick (physics/fluids/euler.hpp) now PROVIDES both capabilities with the
/// canonical-Euler formulas VERBATIM, so it takes the generic path BIT-IDENTICALLY to the former
/// fallback (and to EulerHLLCFlux2D / EulerRoeFlux2D); the explicit euler_hllc / euler_roe routes
/// and the generic hllc / roe routes are therefore the same arithmetic on a true Euler transport.
///
/// device INVARIANT: no vtable, no std:: in the critical paths (std::sqrt is allowed in RoeFlux
/// for the Roe average, device-clean under Kokkos/nvcc).

#pragma once

#include <pops/core/state/state.hpp>
#include <pops/core/foundation/types.hpp>

#include <cmath>     // std::sqrt (RoeFlux: Roe average); libstdc++ does not pull it transitively
#include <concepts>  // Riemann capabilities (HasHLLCStructure / HasRoeDissipation)

namespace pops {

/// RusanovFlux (local Lax-Friedrichs): robust flux, compatible with any minimal PhysicalModel.
///
/// Fhat = 1/2 (F(UL) + F(UR)) - 1/2 alpha (UR - UL), alpha = max(sL, sR).
/// Only requires model.max_wave_speed -- compatible with ANY PhysicalModel of the base concept
/// (no wave_speeds, no pressure). Diffusive (alpha upper bound) but universal.
/// POPS_HD. INVARIANT: component-by-component treatment (scalar upwind), no coupling.
struct RusanovFlux {
  template <class Model>
  POPS_HD typename Model::State operator()(const Model& m, const typename Model::State& UL,
                                          const Aux& AL, const typename Model::State& UR,
                                          const Aux& AR, int dir) const {
    const auto FL = m.flux(UL, AL, dir);
    const auto FR = m.flux(UR, AR, dir);
    const Real sL = m.max_wave_speed(UL, AL, dir);
    const Real sR = m.max_wave_speed(UR, AR, dir);
    const Real alpha = sL > sR ? sL : sR;  // device-safe max (no std::max)
    typename Model::State F;
    for (int c = 0; c < Model::n_vars; ++c)
      F[c] = Real(0.5) * (FL[c] + FR[c]) - Real(0.5) * alpha * (UR[c] - UL[c]);
    return F;
  }
};

/// hll_speeds: Davis estimates for the signal speeds of the HLL/HLLC solvers.
///
/// sL = min(sL_left, sL_right), sR = max(sR_left, sR_right).
/// Requires model.wave_speeds(U, aux, dir, lo, hi) -> signed speeds (cf. Euler).
/// Shared by HLLFlux and HLLCFlux. POPS_HD.
template <class Model>
POPS_HD inline void hll_speeds(const Model& m, const typename Model::State& UL, const Aux& AL,
                              const typename Model::State& UR, const Aux& AR, int dir, Real& sL,
                              Real& sR) {
  Real lL, hL, lR, hR;
  m.wave_speeds(UL, AL, dir, lL, hL);
  m.wave_speeds(UR, AR, dir, lR, hR);
  sL = lL < lR ? lL : lR;
  sR = hL > hR ? hL : hR;
}

/// hll_flux_with_speeds: HLL flux from ALREADY estimated signal speeds (sL, sR).
///
/// Body of HLLFlux AFTER hll_speeds: same supersonic branches (FL if sL >= 0, FR if sR <= 0), same
/// HLL combination. Extracted as a free function for the OPT-IN path that pre-computes the wave speeds
/// PER CELL (cache) then bounds each face by min/max of the two adjacent cells, instead of recalling
/// model.wave_speeds per face (cf. assemble_rhs_hll_cached). For reconstructed states equal to the
/// cell values (NoSlope) this path is ALGEBRAICALLY identical to HLLFlux. POPS_HD.
template <class Model>
POPS_HD inline typename Model::State hll_flux_with_speeds(const Model& m,
                                                         const typename Model::State& UL,
                                                         const Aux& AL,
                                                         const typename Model::State& UR,
                                                         const Aux& AR, int dir, Real sL, Real sR) {
  const auto FL = m.flux(UL, AL, dir);
  const auto FR = m.flux(UR, AR, dir);
  if (sL >= 0)
    return FL;
  if (sR <= 0)
    return FR;
  typename Model::State F;
  const Real inv = Real(1) / (sR - sL);
  for (int c = 0; c < Model::n_vars; ++c)
    F[c] = (sR * FL[c] - sL * FR[c] + sL * sR * (UR[c] - UL[c])) * inv;
  return F;
}

/// HLLFlux (Harten-Lax-van Leer): 2 signal speeds, less diffusive than Rusanov.
///
/// Requires model.wave_speeds (signed speeds sL, sR). Less diffusive than Rusanov on shocks and
/// rarefactions; still imperfectly captures contact discontinuities (a single star region).
/// Returns FL if sL >= 0, FR if sR <= 0, the HLL flux otherwise. POPS_HD.
struct HLLFlux {
  template <class Model>
  POPS_HD typename Model::State operator()(const Model& m, const typename Model::State& UL,
                                          const Aux& AL, const typename Model::State& UR,
                                          const Aux& AR, int dir) const {
    Real sL, sR;
    hll_speeds(m, UL, AL, UR, AR, dir, sL, sR);
    return hll_flux_with_speeds(m, UL, AL, UR, AR, dir, sL, sR);
  }
};

// ---------------------------------------------------------------------------------------------
// RIEMANN CAPABILITIES: OPTIONAL traits through which a NON-Euler model supplies the physical
// structure required by a contact-resolving (HLLC) or Roe-like solver. cf. the file header.
// ---------------------------------------------------------------------------------------------

/// HLLC capability: the model provides the CONTACT wave speed and the STAR STATE on side k.
///  - contact_speed(UL, UR, pL, pR, sL, sR, dir) -> s* (intermediate wave speed);
///  - hllc_star_state(U, p, s, s_star, dir) -> U*_k (star state on side k, s = sL or sR).
/// With pressure + wave_speeds, HLLCFlux then applies the GENERIC algorithm
/// F* = F_k + s_k (U*_k - U_k) without any layout assumption. Both methods must be POPS_HD (called
/// in the kernels).
template <class M>
concept HasHLLCStructure =
    requires(const M m, const typename M::State u, const typename M::State v, const Aux a, Real p,
             Real q, Real sl, Real sr, Real ss, int dir) {
      { m.pressure(u) } -> std::convertible_to<Real>;
      m.wave_speeds(u, a, dir, sl, sr);  // signed speeds (hll_speeds, outer wave bounds)
      { m.contact_speed(u, v, p, q, sl, sr, dir) } -> std::convertible_to<Real>;
      { m.hllc_star_state(u, p, sl, ss, dir) } -> std::same_as<typename M::State>;
    };

/// Roe capability: the model provides its FULL Roe dissipation
/// d = |A_roe(UL, UR)| (UR - UL) -- Roe average, wave decomposition, entropy fix included
/// (these are properties of the physical system, not the core). RoeFlux then becomes
/// F = 1/2 (F_L + F_R) - 1/2 d, without any Euler assumption. POPS_HD required.
template <class M>
concept HasRoeDissipation = requires(const M m, const typename M::State ul, const Aux al,
                                     const typename M::State ur, const Aux ar, int dir) {
  { m.roe_dissipation(ul, al, ur, ar, dir) } -> std::same_as<typename M::State>;
};

namespace detail {
/// Dependent-false for a static_assert that only fires when the generic-only branch is actually
/// instantiated (a model without the required Riemann capability). cf. always_false_v elsewhere.
template <class>
inline constexpr bool riemann_dependent_false = false;
}  // namespace detail

/// HLLCFlux (HLL + Contact wave, Toro): 3 waves, resolves the contact discontinuity.
///
/// GENERIC-ONLY since ADC-590: the model MUST supply HasHLLCStructure (contact_speed +
/// hllc_star_state). The solver applies F* = F_k + s_k (U*_k - U_k) with no layout or EOS
/// assumption. The canonical 2D Euler layout (n_vars == 4 + pressure) is served by the EXPLICIT
/// route EulerHLLCFlux2D (below); the native Euler brick now PROVIDES the HLLC capability, so it
/// takes THIS generic path with bit-identical arithmetic. POPS_HD.
struct HLLCFlux {
  template <class Model>
  POPS_HD typename Model::State operator()(const Model& m, const typename Model::State& UL,
                                           const Aux& AL, const typename Model::State& UR,
                                           const Aux& AR, int dir) const {
    if constexpr (HasHLLCStructure<Model>) {
      Real sL, sR;
      hll_speeds(m, UL, AL, UR, AR, dir, sL, sR);
      const auto FL = m.flux(UL, AL, dir);
      const auto FR = m.flux(UR, AR, dir);
      if (sL >= 0)
        return FL;
      if (sR <= 0)
        return FR;
      const Real pL = m.pressure(UL), pR = m.pressure(UR);
      const Real sStar = m.contact_speed(UL, UR, pL, pR, sL, sR, dir);
      typename Model::State F;
      if (sStar >= 0) {
        const typename Model::State Us = m.hllc_star_state(UL, pL, sL, sStar, dir);
        for (int c = 0; c < Model::n_vars; ++c)
          F[c] = FL[c] + sL * (Us[c] - UL[c]);
      } else {
        const typename Model::State Us = m.hllc_star_state(UR, pR, sR, sStar, dir);
        for (int c = 0; c < Model::n_vars; ++c)
          F[c] = FR[c] + sR * (Us[c] - UR[c]);
      }
      return F;
    } else {
      static_assert(detail::riemann_dependent_false<Model>,
                    "HLLCFlux is generic-only (ADC-590): the model must provide the HLLC "
                    "capabilities (HasHLLCStructure: pressure + wave_speeds + contact_speed + "
                    "hllc_star_state); the canonical 2D Euler path is EulerHLLCFlux2D, chosen by "
                    "the explicit euler_hllc route");
      return {};  // unreachable (the static_assert fires first)
    }
  }
};

/// EulerHLLCFlux2D: the EXPLICIT canonical 2D Euler HLLC route (ADC-590). Assumes n_vars == 4 with
/// the (rho, m_x, m_y, E) layout and model.pressure; the normal / tangential momentum are selected
/// by dir, the star speed s* comes from Toro's formula (eq. 10.37), FL / FR in the supersonic
/// region. This is the MOVED historical canonical-Euler branch, unchanged: the euler_hllc route
/// pins it directly instead of relying on a fallback inside the generic HLLCFlux. POPS_HD.
struct EulerHLLCFlux2D {
  template <class Model>
  POPS_HD typename Model::State operator()(const Model& m, const typename Model::State& UL,
                                           const Aux& AL, const typename Model::State& UR,
                                           const Aux& AR, int dir) const {
    const int in = (dir == 0) ? 1 : 2;  // normal momentum component
    const int it = (dir == 0) ? 2 : 1;  // tangential
    const Real rL = UL[0], rR = UR[0];
    const Real unL = UL[in] / rL, unR = UR[in] / rR;
    const Real pL = m.pressure(UL), pR = m.pressure(UR);
    Real sL, sR;
    hll_speeds(m, UL, AL, UR, AR, dir, sL, sR);
    const auto FL = m.flux(UL, AL, dir);
    const auto FR = m.flux(UR, AR, dir);
    if (sL >= 0)
      return FL;
    if (sR <= 0)
      return FR;

    // contact wave speed (Toro 10.37)
    const Real sStar = (pR - pL + rL * unL * (sL - unL) - rR * unR * (sR - unR)) /
                       (rL * (sL - unL) - rR * (sR - unR));
    typename Model::State F;
    if (sStar >= 0) {  // left star state
      const Real fac = rL * (sL - unL) / (sL - sStar);
      typename Model::State Us;
      Us[0] = fac;
      Us[in] = fac * sStar;
      Us[it] = fac * (UL[it] / rL);
      Us[3] = fac * (UL[3] / rL + (sStar - unL) * (sStar + pL / (rL * (sL - unL))));
      for (int c = 0; c < 4; ++c)
        F[c] = FL[c] + sL * (Us[c] - UL[c]);
    } else {  // right star state
      const Real fac = rR * (sR - unR) / (sR - sStar);
      typename Model::State Us;
      Us[0] = fac;
      Us[in] = fac * sStar;
      Us[it] = fac * (UR[it] / rR);
      Us[3] = fac * (UR[3] / rR + (sStar - unR) * (sStar + pR / (rR * (sR - unR))));
      for (int c = 0; c < 4; ++c)
        F[c] = FR[c] + sR * (Us[c] - UR[c]);
    }
    return F;
  }
};

/// Width of the RoeFlux Harten entropy-fix smoothing, as a fraction of the Roe sound speed
/// (eps = kRoeEntropyFixFraction * c). DOCUMENTED constant rather than hidden in the kernel;
/// SPECIFIC to Euler/Roe (cf. the comment in RoeFlux::operator()).
inline constexpr Real kRoeEntropyFixFraction = Real(0.1);

/// RoeFlux: Roe linearization + Harten entropy fix (acoustic waves).
///
/// GENERIC-ONLY since ADC-590: the model MUST supply HasRoeDissipation (full d = |A_roe| (U_R - U_L),
/// linearization + eigenstructure + entropy fix). The solver then does F = 1/2 (F_L + F_R) - 1/2 d
/// with no Euler assumption. The canonical ideal-gas 2D Euler layout is served by the EXPLICIT route
/// EulerRoeFlux2D (below); the native Euler brick now PROVIDES the Roe dissipation, so it takes THIS
/// generic path with bit-identical arithmetic. POPS_HD.
struct RoeFlux {
  template <class Model>
  POPS_HD typename Model::State operator()(const Model& m, const typename Model::State& UL,
                                           const Aux& AL, const typename Model::State& UR,
                                           const Aux& AR, int dir) const {
    if constexpr (HasRoeDissipation<Model>) {
      const auto FL = m.flux(UL, AL, dir);
      const auto FR = m.flux(UR, AR, dir);
      const typename Model::State d = m.roe_dissipation(UL, AL, UR, AR, dir);
      typename Model::State F;
      for (int c = 0; c < Model::n_vars; ++c)
        F[c] = Real(0.5) * (FL[c] + FR[c]) - Real(0.5) * d[c];
      return F;
    } else {
      static_assert(
          detail::riemann_dependent_false<Model>,
          "RoeFlux is generic-only (ADC-590): the model must provide the Roe dissipation "
          "(HasRoeDissipation: roe_dissipation d = |A_roe| (U_R - U_L)); the canonical 2D "
          "Euler path is EulerRoeFlux2D, chosen by the explicit euler_roe route");
      return {};  // unreachable (the static_assert fires first)
    }
  }
};

/// EulerRoeFlux2D: the EXPLICIT canonical ideal-gas 2D Euler Roe route (ADC-590). Assumes n_vars == 4
/// with the (rho, m_x, m_y, E) layout and model.pressure; FULL eigenwave decomposition
/// F_R - F_L = A_roe (U_R - U_L) exactly, gamma-1 derived from the current state (ideal-gas EOS),
/// Harten entropy fix eps = kRoeEntropyFixFraction * c on the acoustic waves. This is the MOVED
/// historical canonical-Euler branch, unchanged: the euler_roe route pins it directly instead of a
/// fallback inside the generic RoeFlux. std::sqrt for the Roe average (device-clean). POPS_HD.
struct EulerRoeFlux2D {
  template <class Model>
  POPS_HD typename Model::State operator()(const Model& m, const typename Model::State& UL,
                                           const Aux& AL, const typename Model::State& UR,
                                           const Aux& AR, int dir) const {
    const int in = (dir == 0) ? 1 : 2;  // normal momentum
    const int it = (dir == 0) ? 2 : 1;  // tangential
    const Real rL = UL[0], rR = UR[0];
    const Real unL = UL[in] / rL, unR = UR[in] / rR;
    const Real utL = UL[it] / rL, utR = UR[it] / rR;
    const Real pL = m.pressure(UL), pR = m.pressure(UR);
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

    // |eigenvalue| with Harten entropy fix on the acoustic waves (1, 5).
    // kRoeEntropyFixFraction = 0.1 is an EULER/ROE-SPECIFIC entropy policy (width of the parabolic
    // smoothing as a fraction of the Roe sound speed, the usual value from the literature).
    const Real eps = kRoeEntropyFixFraction * c;
    auto absfix = [eps](Real l) {
      const Real al = l < 0 ? -l : l;
      return al < eps ? Real(0.5) * (l * l / eps + eps) : al;
    };
    const Real al1 = absfix(un - c), al2 = (un < 0 ? -un : un), al5 = absfix(un + c);

    // dissipation Sum |lambda_k| a_k r_k, basis (rho, mom_n, mom_t, E)
    const Real d_rho = al1 * a1 + al2 * a2 + al5 * a5;
    const Real d_mn = al1 * a1 * (un - c) + al2 * a2 * un + al5 * a5 * (un + c);
    const Real d_mt = al1 * a1 * ut + al2 * (a2 * ut + a3) + al5 * a5 * ut;
    const Real d_E =
        al1 * a1 * (H - un * c) + al2 * (a2 * Real(0.5) * q2 + a3 * ut) + al5 * a5 * (H + un * c);

    const auto FL = m.flux(UL, AL, dir);
    const auto FR = m.flux(UR, AR, dir);
    typename Model::State F;
    F[0] = Real(0.5) * (FL[0] + FR[0]) - Real(0.5) * d_rho;
    F[in] = Real(0.5) * (FL[in] + FR[in]) - Real(0.5) * d_mn;
    F[it] = Real(0.5) * (FL[it] + FR[it]) - Real(0.5) * d_mt;
    F[3] = Real(0.5) * (FL[3] + FR[3]) - Real(0.5) * d_E;
    return F;
  }
};

}  // namespace pops
