#pragma once

/// @file
/// @brief Exact-ranked compressible Euler model (ideal gas): pure HYPERBOLIC brick satisfying
///        the HyperbolicPhysicalModel concept. Source and elliptic right-hand side are
///        separate bricks (physics/source.hpp, physics/elliptic.hpp); this file contains only
///        Vars + flux + wave speeds + cons<->prim conversions.

#include <pops/core/state/state.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/core/state/variables.hpp>
#include <pops/numerics/fv/flux_interfaces.hpp>
#include <pops/runtime/numerical_defaults.hpp>

#include <array>
#include <cmath>
#include <utility>

namespace pops {

/**
 * Compressible Euler for an ideal gas in the exact native spatial rank.
 *
 * Conservative variables are U = (rho, rho v_0, ..., rho v_(Dim-1), E), with
 * E = p/(gamma-1) + 1/2 rho |v|^2.  A single axis-indexed implementation provides every
 * directional flux and characteristic closure; there is no dimension-erased or planar fallback.
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
template <int Dim>
struct EulerND {
  static_assert(Dim >= 1 && Dim <= 3, "EulerND supports dimensions 1, 2, and 3");
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Dim + 2;
  static constexpr int density_component = 0;
  static constexpr int energy_component = Dim + 1;
  using State = StateVec<n_vars>;
  using Prim = StateVec<n_vars>;
  using Aux = AuxState<Dim>;

  Real gamma = kPhysicalDefaultGamma;  ///< adiabatic index of the ideal gas

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.physics.hyperbolic.euler-nd", 2};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(std::int32_t{Dim}).scalar(gamma);
  }

  POPS_HD static constexpr int momentum_component(int axis) { return 1 + axis; }

  /// Ideal-gas pressure p = (gamma-1)(E - 1/2 rho |v|^2).
  POPS_HD Real pressure(const State& u) const {
    const Real rho = u[density_component];
    Real momentum_squared = Real(0);
    for (int axis = 0; axis < Dim; ++axis) {
      const Real momentum = u[momentum_component(axis)];
      momentum_squared += momentum * momentum;
    }
    const Real ke = Real(0.5) * momentum_squared / rho;
    return (gamma - Real(1)) * (u[energy_component] - ke);
  }
  /// Sound speed c = sqrt(gamma p / rho).
  POPS_HD Real sound_speed(const State& u) const { return std::sqrt(gamma * pressure(u) / u[0]); }

  /// Conservative -> primitive: (rho, rho v_0, ..., E) -> (rho, v_0, ..., p).
  POPS_HD Prim to_primitive(const State& u) const {
    const Real rho = u[density_component];
    Prim p{};
    p[density_component] = rho;
    for (int axis = 0; axis < Dim; ++axis)
      p[momentum_component(axis)] = u[momentum_component(axis)] / rho;
    p[energy_component] = pressure(u);
    return p;
  }
  /// Primitive -> conservative: (rho, v_0, ..., p) -> (rho, rho v_0, ..., E).
  POPS_HD State to_conservative(const Prim& p) const {
    const Real rho = p[density_component];
    State u{};
    u[density_component] = rho;
    Real velocity_squared = Real(0);
    for (int axis = 0; axis < Dim; ++axis) {
      const Real velocity = p[momentum_component(axis)];
      u[momentum_component(axis)] = rho * velocity;
      velocity_squared += velocity * velocity;
    }
    u[energy_component] =
        p[energy_component] / (gamma - Real(1)) + Real(0.5) * rho * velocity_squared;
    return u;
  }

  /**
   * Extreme signed wave speeds in direction dir: v_dir - c and v_dir + c.
   *
   * Required by the HLL/HLLC fluxes, beyond the single max_wave_speed that Rusanov needs.
   *
   * @param      u    conservative state
   * @param      dir  face axis in [0, Dim)
   * @param[out] smin leftmost wave speed v_dir - c
   * @param[out] smax rightmost wave speed v_dir + c
   */
  POPS_HD void wave_speeds(const State& u, const auto&, int dir, Real& smin, Real& smax) const {
    const Prim p = to_primitive(u);
    const Real vn = p[momentum_component(dir)];
    const Real c = std::sqrt(gamma * p[energy_component] / p[density_component]);
    smin = vn - c;
    smax = vn + c;
  }

  /// Compressible convective flux in direction dir.
  POPS_HD State flux(const State& u, const auto&, int dir) const {
    const Real rho = u[density_component];
    const Real vn = u[momentum_component(dir)] / rho;
    const Real p = pressure(u);
    State f{};
    f[density_component] = rho * vn;
    for (int axis = 0; axis < Dim; ++axis)
      f[momentum_component(axis)] = u[momentum_component(axis)] * vn + (axis == dir ? p : Real(0));
    f[energy_component] = (u[energy_component] + p) * vn;
    return f;
  }

  // -------------------------------------------------------------------------------------------
  // RIEMANN CAPABILITIES: Euler provides its physical HLLC closure and Roe dissipation through
  // the same HasHLLCStructure / HasRoeDissipation contracts as every other model. Numerical-flux
  // policies never inspect this concrete layout.
  // -------------------------------------------------------------------------------------------

  /// HLLC contact wave speed s* (Toro eq. 10.37), using the axis-selected normal momentum.
  POPS_HD Real contact_speed(const State& UL, const State& UR, Real pL, Real pR, Real sL, Real sR,
                             int dir) const {
    const int in = momentum_component(dir);
    const Real rL = UL[density_component], rR = UR[density_component];
    const Real unL = UL[in] / rL, unR = UR[in] / rR;
    return (pR - pL + rL * unL * (sL - unL) - rR * unR * (sR - unR)) /
           (rL * (sL - unL) - rR * (sR - unR));
  }

  /// HLLC star state U*_k on side k (Toro), including every tangential momentum.
  POPS_HD State hllc_star_state(const State& U, Real p, Real s, Real sStar, int dir) const {
    const int in = momentum_component(dir);
    const Real r = U[density_component];
    const Real un = U[in] / r;
    const Real fac = r * (s - un) / (s - sStar);
    State Us{};
    Us[density_component] = fac;
    Us[in] = fac * sStar;
    for (int axis = 0; axis < Dim; ++axis) {
      const int component = momentum_component(axis);
      if (axis != dir)
        Us[component] = fac * (U[component] / r);
    }
    Us[energy_component] =
        fac * (U[energy_component] / r + (sStar - un) * (sStar + p / (r * (s - un))));
    return Us;
  }

  /// Roe dissipation d = |A_roe| (U_R - U_L) for the ideal-gas Euler model: FULL
  /// eigenwave decomposition (F_R - F_L = A_roe (U_R - U_L) exactly), sqrt(rho) Roe average, gamma-1
  /// from the ideal-gas EOS, and a typed Harten entropy policy on the acoustic waves. RoeFlux
  /// (HasRoeDissipation) then does F = 1/2 (F_L + F_R) - 1/2 d.
  POPS_HD State roe_dissipation(const State& UL, const auto&, const State& UR, const auto&,
                                int dir) const {
    const int in = momentum_component(dir);
    const Real rL = UL[density_component], rR = UR[density_component];
    const Real unL = UL[in] / rL, unR = UR[in] / rR;
    const Real pL = pressure(UL), pR = pressure(UR);
    const Real HL = (UL[energy_component] + pL) / rL;
    const Real HR = (UR[energy_component] + pR) / rR;

    // Roe average (weighted by sqrt(rho))
    const Real sqL = std::sqrt(rL), sqR = std::sqrt(rR), den = sqL + sqR;
    std::array<Real, Dim> velocity{};
    std::array<Real, Dim> velocity_left{};
    std::array<Real, Dim> velocity_right{};
    Real q2 = Real(0);
    Real q2_left = Real(0);
    for (int axis = 0; axis < Dim; ++axis) {
      const int component = momentum_component(axis);
      velocity_left[axis] = UL[component] / rL;
      velocity_right[axis] = UR[component] / rR;
      velocity[axis] = (sqL * velocity_left[axis] + sqR * velocity_right[axis]) / den;
      q2 += velocity[axis] * velocity[axis];
      q2_left += velocity_left[axis] * velocity_left[axis];
    }
    const Real un = velocity[dir];
    const Real H = (sqL * HL + sqR * HR) / den;
    const Real rho = sqL * sqR;
    // Recover gamma-1 through the same ideal-gas identity as the accepted planar provider.
    // Besides authenticating the constitutive state, this preserves its bit-level Roe oracle.
    const Real gm1 = pL / (UL[energy_component] - Real(0.5) * rL * q2_left);
    const Real c2 = gm1 * (H - Real(0.5) * q2);
    const Real c = std::sqrt(c2);

    // wave jumps and amplitudes
    const Real dr = rR - rL, dp = pR - pL, dun = unR - unL;
    const Real a1 = (dp - rho * c * dun) / (Real(2) * c2);  // un - c wave
    const Real a2 = dr - dp / c2;                           // entropy, un
    const Real a5 = (dp + rho * c * dun) / (Real(2) * c2);  // un + c wave

    // Entropy correction is an explicit provider policy, not a numerical-flux fallback.
    const HartenEntropyFix entropy_fix{Real(0.1)};
    const Real al1 = entropy_fix(un - c, c);
    const Real al2 = un < Real(0) ? -un : un;
    const Real al5 = entropy_fix(un + c, c);

    // Dissipation Sum |lambda_k| a_k r_k over the acoustic, entropy, and Dim-1 shear waves.
    State d{};
    d[density_component] = al1 * a1 + al2 * a2 + al5 * a5;
    Real shear_energy = Real(0);
    for (int axis = 0; axis < Dim; ++axis) {
      const Real directional = axis == dir ? c : Real(0);
      Real middle = a2 * velocity[axis];
      if (axis != dir) {
        const Real shear = rho * (velocity_right[axis] - velocity_left[axis]);
        middle += shear;
        shear_energy += shear * velocity[axis];
      }
      d[momentum_component(axis)] = al1 * a1 * (velocity[axis] - directional) + al2 * middle +
                                    al5 * a5 * (velocity[axis] + directional);
    }
    d[energy_component] = al1 * a1 * (H - un * c) + al2 * (a2 * Real(0.5) * q2 + shear_energy) +
                          al5 * a5 * (H + un * c);
    return d;
  }

  /// Full spectrum: two acoustic eigenvalues and Dim repeated material/shear eigenvalues.
  POPS_HD State eigenvalues(const State& u, const auto&, int dir) const {
    const Prim p = to_primitive(u);
    const Real vn = p[momentum_component(dir)];
    const Real c = std::sqrt(gamma * p[energy_component] / p[density_component]);
    State e{};
    e[0] = vn - c;
    for (int component = 1; component <= Dim; ++component)
      e[component] = vn;
    e[energy_component] = vn + c;
    return e;
  }

  /// Maximum wave speed |v_dir| + c (Rusanov estimate), computed in primitive variables.
  POPS_HD Real max_wave_speed(const State& u, const auto&, int dir) const {
    const Prim p = to_primitive(u);
    const Real vn = p[momentum_component(dir)];
    const Real a = vn < 0 ? -vn : vn;  // |v_dir| device-safe
    return a + std::sqrt(gamma * p[energy_component] / p[density_component]);
  }

  /// Variable descriptor (hyperbolic model contract; host introspection metadata).
  static VariableSet conservative_vars() {
    constexpr std::array momentum_names{"rho_u", "rho_v", "rho_w"};
    constexpr std::array momentum_roles{VariableRole::MomentumX, VariableRole::MomentumY,
                                        VariableRole::MomentumZ};
    std::vector<std::string> names{"rho"};
    std::vector<VariableRole> roles{VariableRole::Density};
    for (int axis = 0; axis < Dim; ++axis) {
      names.emplace_back(momentum_names[axis]);
      roles.push_back(momentum_roles[axis]);
    }
    names.emplace_back("E");
    roles.push_back(VariableRole::Energy);
    return {VariableKind::Conservative, std::move(names), n_vars, std::move(roles)};
  }
  static VariableSet primitive_vars() {
    constexpr std::array velocity_names{"u", "v", "w"};
    constexpr std::array velocity_roles{VariableRole::VelocityX, VariableRole::VelocityY,
                                        VariableRole::VelocityZ};
    std::vector<std::string> names{"rho"};
    std::vector<VariableRole> roles{VariableRole::Density};
    for (int axis = 0; axis < Dim; ++axis) {
      names.emplace_back(velocity_names[axis]);
      roles.push_back(velocity_roles[axis]);
    }
    names.emplace_back("p");
    roles.push_back(VariableRole::Pressure);
    return {VariableKind::Primitive, std::move(names), n_vars, std::move(roles)};
  }
};

using Euler = EulerND<kNativeDimension>;

}  // namespace pops
