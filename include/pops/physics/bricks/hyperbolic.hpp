#pragma once

#include <pops/core/state/state.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/core/state/variables.hpp>
#include <pops/physics/fluids/euler.hpp>  // Euler: reused as the CompressibleFlux hyperbolic brick

#include <cmath>
#include <limits>
#include <type_traits>

/// @file
/// @brief Generic HYPERBOLIC bricks: Vars (cons U / prim P + conversions + descriptor) +
///        flux + wave speeds. Each one satisfies the HyperbolicPhysicalModel concept: State, Prim,
///        n_vars, flux, max_wave_speed, to_primitive/to_conservative, conservative_vars/primitive_vars
///        (+ pressure/wave_speeds if HLLC flux). Source and elliptic right-hand side are SEPARATE
///        bricks (physics/source.hpp, physics/elliptic.hpp); CompositeModel (physics/composite.hpp)
///        assembles them. ExBVelocity (1 var), CompressibleFlux (= Euler, 4 var), IsothermalFlux (3 var).

namespace pops {

namespace hyperbolic_detail {

template <class Providers>
consteval int provider_dimension() {
  using Provider = std::remove_cvref_t<Providers>;
  if constexpr (requires { Provider::dimension; })
    return static_cast<int>(Provider::dimension);
  return kNativeDimension;
}

template <int Axis, int Dim, class Providers>
POPS_HD Real gradient_provider(const Providers& providers) {
  static_assert(Axis >= 0 && Axis < Dim, "gradient provider axis is outside the spatial rank");
  static_assert(provider_dimension<Providers>() == Dim,
                "physical brick and provider pack carry different spatial ranks");
  constexpr int component = AuxComponentLayout<Dim>::template gradient_component<Axis>();
  return providers.template flux_provider<component>();
}

template <int Axis, int Dim, class Brick, class Providers>
POPS_HD Real runtime_velocity(const Brick& brick, const Providers& providers, int axis) {
  if (axis == Axis)
    return brick.template velocity<Axis>(providers);
  if constexpr (Axis + 1 < Dim)
    return runtime_velocity<Axis + 1, Dim>(brick, providers, axis);
  return std::numeric_limits<Real>::quiet_NaN();
}

}  // namespace hyperbolic_detail

/// Exact-ranked scalar advection by the Cartesian E x B drift for `B = B0 e_z`.
///
/// In 2D this is `(-d_y phi, d_x phi)/B0`. A 3D Cartesian build carries the same planar drift
/// with a zero z component. Under a 1D invariant reduction the projected velocity is zero because
/// the transverse derivative is absent. Every available axis is selected at compile time.
template <int Dim>
struct ExBVelocityND {
  static_assert(Dim >= 1 && Dim <= 3, "ExBVelocityND supports dimensions 1, 2, and 3");
  static constexpr int n_vars = 1;
  static constexpr int dimension = Dim;
  using State = StateVec<1>;
  using Aux = AuxState<Dim>;
  Real B0 = 1;

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.physics.hyperbolic.exb-velocity-nd", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(std::int32_t{Dim}).scalar(B0);
  }

  template <int Axis, class Providers>
  POPS_HD Real velocity(const Providers& providers) const {
    static_assert(Axis >= 0 && Axis < Dim, "Cartesian E x B axis is outside the spatial rank");
    if constexpr (Dim < 2 || Axis >= 2)
      return Real(0);
    else if constexpr (Axis == 0)
      return -hyperbolic_detail::gradient_provider<1, Dim>(providers) / B0;
    else
      return hyperbolic_detail::gradient_provider<0, Dim>(providers) / B0;
  }

  POPS_HD Real velocity(const auto& providers, int dir) const {
    return hyperbolic_detail::runtime_velocity<0, Dim>(*this, providers, dir);
  }

  template <int Axis, class Providers>
  POPS_HD State flux(const State& state, const Providers& providers) const {
    return State{state[0] * velocity<Axis>(providers)};
  }

  POPS_HD StateVec<1> flux(const StateVec<1>& u, const auto& providers, int dir) const {
    return State{u[0] * velocity(providers, dir)};
  }

  template <int Axis, class Providers>
  POPS_HD Real max_wave_speed(const State&, const Providers& providers) const {
    const Real speed = velocity<Axis>(providers);
    return speed < Real(0) ? -speed : speed;
  }

  POPS_HD Real max_wave_speed(const StateVec<1>&, const auto& providers, int dir) const {
    const Real d = velocity(providers, dir);
    return d < 0 ? -d : d;
  }

  template <int Axis, class Providers>
  POPS_HD State eigenvalues(const State&, const Providers& providers) const {
    return State{velocity<Axis>(providers)};
  }

  POPS_HD StateVec<1> eigenvalues(const StateVec<1>&, const auto& providers, int dir) const {
    return State{velocity(providers, dir)};
  }

  using Prim = StateVec<1>;
  POPS_HD Prim to_primitive(const StateVec<1>& u) const { return u; }
  POPS_HD StateVec<1> to_conservative(const Prim& p) const { return p; }
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"n"}, 1, {VariableRole::Density}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"n"}, 1, {VariableRole::Density}};
  }
};

using ExBVelocity = ExBVelocityND<kNativeDimension>;

/// Scalar advection by the E x B drift in POLAR coordinates (r, theta) -- "annular polar grid"
/// capability. This type is intrinsically two-dimensional and remains separate from the ranked
/// Cartesian brick.
///
/// `gradient<0>()` is the radial derivative and `gradient<1>()` is the physical azimuthal
/// derivative `(1/r) d_phi/d_theta`. The metric-aware caller owns that conversion.
///
/// E x B VELOCITY IN POLAR (PHYSICAL components in the local basis):
///   v_r     = -(1/(B r)) d phi/d theta = -grad_theta / B   (dir == 0, radial)
///   v_theta =  (1/B)     d phi/d r     =  grad_r     / B   (dir == 1, azimuthal)
/// The returned flux (dir 0 = F_r = n v_r; dir 1 = F_theta = n v_theta) is PHYSICAL; the 1/r metric
/// and the divergence (1/r) d_r(r F_r) + (1/r) d_theta(F_theta) are carried by assemble_rhs_polar,
/// NOT by this brick. The brick thus stays a pure physics (no box, no r).
struct ExBVelocityPolar {
  static constexpr int n_vars = 1;
  static constexpr int dimension = 2;
  static constexpr bool planar_polar_capability = true;
  using State = StateVec<1>;
  using Aux = AuxState<2>;
  Real B0 = 1;

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.physics.hyperbolic.exb-velocity-polar", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const { contract.scalar(B0); }

  template <int Axis, class Providers>
  POPS_HD Real velocity(const Providers& providers) const {
    static_assert(Axis >= 0 && Axis < dimension, "polar E x B axis must be radial or azimuthal");
    if constexpr (Axis == 0)
      return -hyperbolic_detail::gradient_provider<1, dimension>(providers) / B0;
    else
      return hyperbolic_detail::gradient_provider<0, dimension>(providers) / B0;
  }

  POPS_HD Real velocity(const auto& providers, int dir) const {
    return hyperbolic_detail::runtime_velocity<0, dimension>(*this, providers, dir);
  }

  template <int Axis, class Providers>
  POPS_HD State flux(const State& state, const Providers& providers) const {
    return State{state[0] * velocity<Axis>(providers)};
  }

  POPS_HD StateVec<1> flux(const StateVec<1>& u, const auto& providers, int dir) const {
    return State{u[0] * velocity(providers, dir)};
  }

  template <int Axis, class Providers>
  POPS_HD Real max_wave_speed(const State&, const Providers& providers) const {
    const Real speed = velocity<Axis>(providers);
    return speed < Real(0) ? -speed : speed;
  }

  POPS_HD Real max_wave_speed(const StateVec<1>&, const auto& providers, int dir) const {
    const Real d = velocity(providers, dir);
    return d < 0 ? -d : d;
  }

  template <int Axis, class Providers>
  POPS_HD State eigenvalues(const State&, const Providers& providers) const {
    return State{velocity<Axis>(providers)};
  }

  POPS_HD StateVec<1> eigenvalues(const StateVec<1>&, const auto& providers, int dir) const {
    return State{velocity(providers, dir)};
  }

  using Prim = StateVec<1>;
  POPS_HD Prim to_primitive(const StateVec<1>& u) const { return u; }
  POPS_HD StateVec<1> to_conservative(const Prim& p) const { return p; }
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"n"}, 1, {VariableRole::Density}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"n"}, 1, {VariableRole::Density}};
  }
};

/// Compressible 2D Euler flux (reuses Euler: gamma, pressure, signed wave speeds).
/// Compat alias: CompressibleFlux == Euler; the complete hyperbolic brick.
using CompressibleFlux = Euler;

/// ISOTHERMAL Euler flux (p = cs2 rho), 3 variables (rho, rho u, rho v).
///
/// 3-variable HYPERBOLIC brick (density + momenta). Satisfies
/// HyperbolicPhysicalModel. Isothermal closure law: p = cs2 * rho (no energy
/// equation). CONTRACT: purely pointwise functions, device-callable (POPS_HD).
/// No MultiFab, no allocation, no global access.
/// Invariant: cs2 > 0 so that the wave speed sqrt(cs2) is real.
struct IsothermalFlux {
  static constexpr int n_vars = 3;
  using State = StateVec<3>;  ///< conservative variables (rho, rho u, rho v)
  using Prim = StateVec<3>;   ///< primitive variables (rho, u, v)
  Real cs2 = 1;
  /// Quasi-vacuum density floor (ADC-77). When > 0, the velocity is computed as u = m / max(rho,
  /// vacuum_floor) so it stays bounded where the rollup evacuates the background (rho -> ~0); this
  /// bounds BOTH the CFL wave speed and the advective flux in one place (max_wave_speed and flux both
  /// divide by rho here). Mass and momentum are NOT modified -- only the velocity ESTIMATE is bounded,
  /// so the conservative state is untouched (unlike a cell density clamp). <= 0: inactive, and the raw
  /// 1/rho path is taken verbatim (bit-identical, including for rho <= 0).
  Real vacuum_floor = 0;
  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.physics.hyperbolic.isothermal-flux", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(cs2).scalar(vacuum_floor);
  }
  /// rho clamped from below by vacuum_floor for the velocity division ONLY. Manual max (device-safe,
  /// no std:: in the kernel path). floor <= 0 -> returns rho unchanged (bit-identical).
  POPS_HD Real velocity_rho(Real rho) const {
    return (vacuum_floor > Real(0) && rho < vacuum_floor) ? vacuum_floor : rho;
  }
  POPS_HD StateVec<3> flux(const StateVec<3>& u, const auto&, int dir) const {
    const Real rho = u[0];
    const Real vn = (dir == 0 ? u[1] : u[2]) / velocity_rho(rho);
    const Real p = cs2 * rho;
    StateVec<3> f{};
    f[0] = (dir == 0 ? u[1] : u[2]);
    f[1] = u[1] * vn + (dir == 0 ? p : Real(0));
    f[2] = u[2] * vn + (dir == 1 ? p : Real(0));
    return f;
  }
  /// Conservative -> primitive: (rho, rho u, rho v) -> (rho, u, v). The velocity uses the quasi-vacuum
  /// floored density (velocity_rho); rho itself (p[0]) stays the raw conserved value.
  POPS_HD Prim to_primitive(const StateVec<3>& u) const {
    Prim p{};
    p[0] = u[0];
    const Real rho_v = velocity_rho(u[0]);
    p[1] = u[1] / rho_v;
    p[2] = u[2] / rho_v;
    return p;
  }
  /// Primitive -> conservative: (rho, u, v) -> (rho, rho u, rho v).
  POPS_HD StateVec<3> to_conservative(const Prim& p) const {
    StateVec<3> u{};
    u[0] = p[0];
    u[1] = p[0] * p[1];
    u[2] = p[0] * p[2];
    return u;
  }
  POPS_HD Real max_wave_speed(const StateVec<3>& u, const auto&, int dir) const {
    const Prim p = to_primitive(u);
    const Real vn = (dir == 0 ? p[1] : p[2]);
    const Real a = vn < 0 ? -vn : vn;
    return a + std::sqrt(cs2);
  }
  /// Full spectrum: (v_dir - c, v_dir, v_dir + c), c = sqrt(cs2).
  POPS_HD StateVec<3> eigenvalues(const StateVec<3>& u, const auto&, int dir) const {
    const Prim p = to_primitive(u);
    const Real vn = (dir == 0 ? p[1] : p[2]);
    const Real c = std::sqrt(cs2);
    StateVec<3> e{};
    e[0] = vn - c;
    e[1] = vn;
    e[2] = vn + c;
    return e;
  }
  /// Signed speeds (HLL/HLLC): v_dir -+ c_s.
  POPS_HD void wave_speeds(const StateVec<3>& u, const auto&, int dir, Real& smin,
                           Real& smax) const {
    const Prim p = to_primitive(u);
    const Real vn = (dir == 0 ? p[1] : p[2]);
    const Real c = std::sqrt(cs2);
    smin = vn - c;
    smax = vn + c;
  }

  // -------------------------------------------------------------------------------------------
  // RIEMANN CAPABILITIES: the isothermal closure owns its contact construction and Roe action.
  // HLLCFlux / RoeFlux remain layout-blind and consume these hooks through the same
  // HasHLLCStructure / HasRoeDissipation contracts as every other physical provider.
  // -------------------------------------------------------------------------------------------

  /// Barotropic pressure p = c_s^2 rho used by the HLLC physical provider.
  POPS_HD Real pressure(const State& u) const { return cs2 * u[0]; }

  /// Contact-wave speed for the isothermal Euler closure.
  POPS_HD Real contact_speed(const State& left, const State& right, Real pressure_left,
                             Real pressure_right, Real lower, Real upper, int dir) const {
    const int normal = dir == 0 ? 1 : 2;
    const Real density_left = left[0];
    const Real density_right = right[0];
    const Real velocity_left = left[normal] / velocity_rho(density_left);
    const Real velocity_right = right[normal] / velocity_rho(density_right);
    return (pressure_right - pressure_left +
            density_left * velocity_left * (lower - velocity_left) -
            density_right * velocity_right * (upper - velocity_right)) /
           (density_left * (lower - velocity_left) - density_right * (upper - velocity_right));
  }

  /// HLLC star state for a barotropic state (rho, rho u, rho v).
  POPS_HD State hllc_star_state(const State& value, Real, Real speed, Real contact, int dir) const {
    const int normal = dir == 0 ? 1 : 2;
    const int tangent = dir == 0 ? 2 : 1;
    const Real density = value[0];
    const Real normal_velocity = value[normal] / velocity_rho(density);
    const Real star_density = density * (speed - normal_velocity) / (speed - contact);
    State result{};
    result[0] = star_density;
    result[normal] = star_density * contact;
    result[tangent] = star_density * (value[tangent] / velocity_rho(density));
    return result;
  }

  /// Roe action |A_roe| dU for the isothermal Euler closure.
  POPS_HD State roe_dissipation(const State& left, const auto&, const State& right, const auto&,
                                int dir) const {
    const int normal = dir == 0 ? 1 : 2;
    const int tangent = dir == 0 ? 2 : 1;
    const Real density_left = left[0];
    const Real density_right = right[0];
    const Real velocity_left = left[normal] / velocity_rho(density_left);
    const Real velocity_right = right[normal] / velocity_rho(density_right);
    const Real tangent_left = left[tangent] / velocity_rho(density_left);
    const Real tangent_right = right[tangent] / velocity_rho(density_right);

    const Real root_left = std::sqrt(density_left);
    const Real root_right = std::sqrt(density_right);
    const Real denominator = root_left + root_right;
    const Real normal_velocity =
        (root_left * velocity_left + root_right * velocity_right) / denominator;
    const Real tangent_velocity =
        (root_left * tangent_left + root_right * tangent_right) / denominator;
    const Real roe_density = root_left * root_right;
    const Real sound_speed = std::sqrt(cs2);

    const Real density_jump = density_right - density_left;
    const Real normal_jump = velocity_right - velocity_left;
    const Real tangent_jump = tangent_right - tangent_left;
    const Real acoustic_minus =
        (cs2 * density_jump - roe_density * sound_speed * normal_jump) / (Real(2) * cs2);
    const Real acoustic_plus =
        (cs2 * density_jump + roe_density * sound_speed * normal_jump) / (Real(2) * cs2);
    const Real shear = roe_density * tangent_jump;

    const HartenEntropyFix entropy_fix{Real(0.1)};
    const Real lambda_minus = entropy_fix(normal_velocity - sound_speed, sound_speed);
    const Real lambda_shear = normal_velocity < Real(0) ? -normal_velocity : normal_velocity;
    const Real lambda_plus = entropy_fix(normal_velocity + sound_speed, sound_speed);

    State result{};
    result[0] = lambda_minus * acoustic_minus + lambda_plus * acoustic_plus;
    result[normal] = lambda_minus * acoustic_minus * (normal_velocity - sound_speed) +
                     lambda_plus * acoustic_plus * (normal_velocity + sound_speed);
    result[tangent] = lambda_minus * acoustic_minus * tangent_velocity + lambda_shear * shear +
                      lambda_plus * acoustic_plus * tangent_velocity;
    return result;
  }
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative,
            {"rho", "rho_u", "rho_v"},
            3,
            {VariableRole::Density, VariableRole::MomentumX, VariableRole::MomentumY}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive,
            {"rho", "u", "v"},
            3,
            {VariableRole::Density, VariableRole::VelocityX, VariableRole::VelocityY}};
  }
};

/// ISOTHERMAL Euler flux in POLAR geometry (ring r, theta), 3 variables (rho, rho v_r,
/// rho v_theta) -- "polar fluid grid" effort, Path A step 1. This is a brick SEPARATE
/// from IsothermalFlux (Cartesian): the PHYSICAL flux and the conversions are IDENTICAL (the
/// components 1, 2 are the momentum in the LOCAL ORTHONORMAL BASIS (e_r, e_theta);
/// dir 0 = radial, dir 1 = azimuthal), but this brick adds the GEOMETRIC CURVATURE TERM
/// carried by the polar metric. We inherit IsothermalFlux to NOT duplicate flux /
/// conversions / wave speeds (Cartesian strictly intact, bit-identical) and add ONLY
/// the polar_geom_source method.
///
/// WHY AN EXPLICIT GEOMETRIC TERM (and not a plain conservative divergence):
/// the vector momentum equation d_t(rho v) + div(rho v (x) v) + grad p = 0,
/// projected onto the LOCAL polar basis (e_r, e_theta) which ROTATES with theta, gives for the
/// PHYSICAL components m_r = rho v_r, m_theta = rho v_theta:
///   d_t m_r     + (1/r) d_r(r (rho v_r^2 + p)) + (1/r) d_theta(rho v_r v_theta)
///                 - (rho v_theta^2 + p)/r            = 0      (CENTRIFUGAL + pressure term)
///   d_t m_theta + (1/r) d_r(r rho v_r v_theta)     + (1/r) d_theta(rho v_theta^2 + p)
///                 + (rho v_r v_theta)/r             = 0      (cross CURVATURE term)
/// The assemble_rhs_polar operator computes EXACTLY -(1/r) d_r(r F_r) - (1/r) d_theta(F_theta)
/// with F_r, F_theta = IsothermalFlux::flux: it thus reproduces the divergences, but NOT the
/// algebraic terms -(rho v_theta^2 + p)/r and +(rho v_r v_theta)/r. These terms are NOT
/// captured by the conservative divergence (proof: on the cell (rho, v_r=0, v_theta(r)) in
/// rotational equilibrium d_r p = rho v_theta^2/r, the radial divergence alone would yield
/// d_t m_r = -(d_r p + p/r) != 0, breaking the equilibrium). An explicit GEOMETRIC SOURCE
/// is therefore REQUIRED, provided here and added per cell by assemble_rhs_polar (which alone knows r):
///   S_geom = ( 0 , (rho v_theta^2 + p)/r , -(rho v_r v_theta)/r ).
/// With this source the rotational equilibrium is preserved to the scheme order (cf.
/// test_polar_fluid_equilibrium). r > 0 (ring, r_min > 0): no axis singularity.
///
/// CONTRACT: pointwise PHYSICAL brick, device-callable (POPS_HD), no box, no allocation.
/// polar_geom_source takes ONLY the state and r (no aux): it is pure metric.
struct IsothermalFluxPolar : IsothermalFlux {
  static constexpr int dimension = 2;
  static constexpr bool planar_polar_capability = true;
  using Aux = AuxState<2>;

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.physics.hyperbolic.isothermal-flux-polar", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(cs2).scalar(vacuum_floor);
  }

  /// GEOMETRIC curvature source term in a cell of radius r > 0 (ring). See the @file block
  /// above for the derivation. S_geom = (0, (rho v_theta^2 + p)/r, -(rho v_r v_theta)/r),
  /// p = cs2 rho. Component 0 (mass) is zero: mass is purely conservative in polar.
  POPS_HD StateVec<3> polar_geom_source(const StateVec<3>& u, Real r) const {
    const Real rho = u[0];
    const Real inv_rho =
        Real(1) / velocity_rho(rho);   // quasi-vacuum floored (ADC-77; bit-identical if off)
    const Real mr = u[1], mth = u[2];  // rho v_r, rho v_theta (local basis (e_r, e_theta))
    const Real p = cs2 * rho;
    const Real inv_r = Real(1) / r;
    StateVec<3> s{};
    s[0] = Real(0);
    s[1] = (mth * mth * inv_rho + p) * inv_r;  // (rho v_theta^2 + p)/r: centrifugal + pressure
    s[2] = -(mr * mth * inv_rho) * inv_r;      // -(rho v_r v_theta)/r: cross curvature
    return s;
  }
};

}  // namespace pops
