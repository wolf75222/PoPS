#pragma once

#include <pops/core/state/state.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/core/state/variables.hpp>
#include <pops/physics/fluids/euler.hpp>  // Euler: reused as the CompressibleFlux hyperbolic brick
#include <pops/numerics/fv/flux_interfaces.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

/// @file
/// @brief Generic HYPERBOLIC bricks: Vars (cons U / prim P + conversions + descriptor) +
///        flux + wave speeds. Each one satisfies the HyperbolicPhysicalModel concept: State, Prim,
///        n_vars, flux, max_wave_speed, to_primitive/to_conservative, conservative_vars/primitive_vars
///        (+ pressure/wave_speeds if HLLC flux). Source and elliptic right-hand side are SEPARATE
///        bricks (physics/source.hpp, physics/elliptic.hpp); CompositeModel (physics/composite.hpp)
///        assembles them. CartesianExBDrift has one scalar; CompressibleFlux and IsothermalFlux carry
///        respectively Dim+2 and Dim+1 variables in the selected native rank.

namespace pops {

namespace hyperbolic_detail {

template <int Dim>
struct DefaultGradientProviderSlots;
template <>
struct DefaultGradientProviderSlots<1> {
  using type = ProviderSlots<0>;
};
template <>
struct DefaultGradientProviderSlots<2> {
  using type = ProviderSlots<0, 1>;
};
template <>
struct DefaultGradientProviderSlots<3> {
  using type = ProviderSlots<0, 1, 2>;
};

template <int Axis, int Dim, class GradientSlots, class Providers>
POPS_HD Real gradient_provider(const Providers& providers) {
  static_assert(Axis >= 0 && Axis < Dim, "gradient provider axis is outside the spatial rank");
  static_assert(GradientSlots::count == Dim,
                "gradient consumer requires one explicit provider slot per axis");
  constexpr int component = GradientSlots::template slot<Axis>();
  return provider_value<component>(providers);
}

template <int Axis, class MagneticSlots, class Providers>
POPS_HD Real magnetic_provider(const Providers& providers) {
  static_assert(Axis >= 0 && Axis < 3, "magnetic provider axis is outside Cartesian 3-space");
  static_assert(MagneticSlots::count == 3,
                "magnetic consumer requires all three Cartesian provider components");
  constexpr int component = MagneticSlots::template slot<Axis>();
  return provider_value<component>(providers);
}

template <int Axis, int Dim, class Brick, class Providers>
POPS_HD Real runtime_velocity(const Brick& brick, const Providers& providers, int axis) {
  if (axis == Axis)
    return brick.template velocity<Axis>(providers);
  if constexpr (Axis + 1 < Dim)
    return runtime_velocity<Axis + 1, Dim>(brick, providers, axis);
  return std::numeric_limits<Real>::quiet_NaN();
}

/// Sign of the three-dimensional Levi-Civita symbol.  Every operand is a template argument, so
/// this is resolved while compiling the selected axis kernel rather than by an Axis-specific branch.
template <int I, int J, int K>
consteval int levi_civita_3() {
  static_assert(I >= 0 && I < 3 && J >= 0 && J < 3 && K >= 0 && K < 3,
                "Levi-Civita indices are outside the ambient Cartesian frame");
  constexpr std::array<int, 3> permutation{I, J, K};
  constexpr bool repeated = I == J || I == K || J == K;
  if constexpr (repeated)
    return 0;
  int inversions = 0;
  for (int left = 0; left < 3; ++left)
    for (int right = left + 1; right < 3; ++right)
      inversions += permutation[left] > permutation[right] ? 1 : 0;
  return inversions % 2 == 0 ? 1 : -1;
}

}  // namespace hyperbolic_detail

/// Exact-ranked scalar advection by the Cartesian E x B drift.
///
/// The magnetic field is an explicit three-component provider and the potential gradient is
/// embedded from the exact native rank in the same ambient Cartesian frame.  One Levi-Civita
/// contraction implements every rank: `v_i = epsilon_ijk B_j grad(phi)_k / |B|^2`.  Thus 3-D
/// consumes all magnetic and gradient components; 2-D can explicitly bind a normal B component;
/// and 1-D evaluates the honest longitudinal projection without a special transport branch.
template <int Dim,
          class GradientSlots = typename hyperbolic_detail::DefaultGradientProviderSlots<Dim>::type,
          class MagneticSlots = ProviderSlots<Dim, Dim + 1, Dim + 2>>
struct CartesianExBDriftND {
  static_assert(Dim >= 1 && Dim <= 3, "CartesianExBDriftND supports exact dimensions 1, 2, and 3");
  static constexpr int n_vars = 1;
  static constexpr int dimension = Dim;
  using Schema = nd::ScalarStateSchema<Dim>;
  using State = typename Schema::Conservative;
  using Primitive = typename Schema::Primitive;
  using Prim = Primitive;
  using gradient_slots = GradientSlots;
  using magnetic_slots = MagneticSlots;
  static_assert(gradient_slots::count == Dim,
                "CartesianExBDriftND requires one explicit gradient provider slot per axis");
  static_assert(magnetic_slots::count == 3,
                "CartesianExBDriftND requires all three magnetic provider slots");
  static constexpr int n_providers = gradient_slots::required_count() >
                                             magnetic_slots::required_count()
                                         ? gradient_slots::required_count()
                                         : magnetic_slots::required_count();

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.physics.hyperbolic.cartesian-exb-drift-nd", 2};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(std::int32_t{Dim});
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(std::int32_t{gradient_slots::values[static_cast<std::size_t>(axis)]});
    for (int axis = 0; axis < 3; ++axis)
      contract.scalar(std::int32_t{magnetic_slots::values[static_cast<std::size_t>(axis)]});
  }

  template <int Axis, class Providers>
  POPS_HD Real velocity(const Providers& providers) const {
    static_assert(Axis >= 0 && Axis < Dim, "Cartesian E x B axis is outside the spatial rank");
    const Real norm_squared = magnetic_field_squared(providers, std::make_index_sequence<3>{});
    if (!nd::conservation_law_detail::finite(norm_squared) || !(norm_squared > Real(0)))
      return std::numeric_limits<Real>::quiet_NaN();
    return velocity_impl<Axis>(providers, std::make_index_sequence<3>{}) / norm_squared;
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

  /// The scalar E×B state is already primitive.  These are the same exact ranked law contract
  /// used by generated Cartesian blocks; they contain no separate conversion algorithm.
  POPS_HD nd::StateConversion<Primitive> recover(const State& state) const {
    return {state, nd::conservation_law_detail::finite(state[0])
                      ? nd::StateConversionStatus::Success
                      : nd::StateConversionStatus::NonFiniteState};
  }
  POPS_HD nd::StateConversion<State> make_conservative(const Primitive& primitive) const {
    return {primitive, nd::conservation_law_detail::finite(primitive[0])
                          ? nd::StateConversionStatus::Success
                          : nd::StateConversionStatus::NonFiniteState};
  }
  POPS_HD nd::StateConversionStatus admissibility(const State& state) const {
    return recover(state).status;
  }
  POPS_HD Prim to_primitive(const State& state) const { return state; }
  POPS_HD State to_conservative(const Prim& primitive) const { return primitive; }
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"n"}, 1, {VariableRole::Density}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"n"}, 1, {VariableRole::Density}};
  }

 private:
  template <int GradientAxis, class Providers>
  POPS_HD Real embedded_gradient(const Providers& providers) const {
    if constexpr (GradientAxis < Dim)
      return hyperbolic_detail::gradient_provider<GradientAxis, Dim, gradient_slots>(providers);
    return Real(0);
  }

  template <class Providers, std::size_t... MagneticAxis>
  POPS_HD Real magnetic_field_squared(const Providers& providers,
                                      std::index_sequence<MagneticAxis...>) const {
    return ((hyperbolic_detail::magnetic_provider<static_cast<int>(MagneticAxis), magnetic_slots>(
                 providers) *
             hyperbolic_detail::magnetic_provider<static_cast<int>(MagneticAxis), magnetic_slots>(
                 providers)) +
            ...);
  }

  template <int Axis, int MagneticAxis, class Providers, std::size_t... GradientAxis>
  POPS_HD Real magnetic_contribution(const Providers& providers,
                                     std::index_sequence<GradientAxis...>) const {
    return ((Real(hyperbolic_detail::levi_civita_3<Axis, MagneticAxis,
                                                   static_cast<int>(GradientAxis)>()) *
             embedded_gradient<static_cast<int>(GradientAxis)>(providers)) +
            ...);
  }

  template <int Axis, class Providers, std::size_t... MagneticAxis>
  POPS_HD Real velocity_impl(const Providers& providers,
                             std::index_sequence<MagneticAxis...>) const {
    return ((hyperbolic_detail::magnetic_provider<static_cast<int>(MagneticAxis), magnetic_slots>(
                 providers) *
             magnetic_contribution<Axis, static_cast<int>(MagneticAxis)>(
                 providers, std::make_index_sequence<3>{})) +
            ...);
  }
};

using CartesianExBDrift = CartesianExBDriftND<kNativeDimension>;

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
  using gradient_slots = ProviderSlots<0, 1>;
  static constexpr int n_providers = gradient_slots::required_count();
  Real B0 = 1;

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.physics.hyperbolic.exb-velocity-polar", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(B0);
    for (int axis = 0; axis < dimension; ++axis)
      contract.scalar(std::int32_t{gradient_slots::values[static_cast<std::size_t>(axis)]});
  }

  template <int Axis, class Providers>
  POPS_HD Real velocity(const Providers& providers) const {
    static_assert(Axis >= 0 && Axis < dimension, "polar E x B axis must be radial or azimuthal");
    if constexpr (Axis == 0)
      return -hyperbolic_detail::gradient_provider<1, dimension, gradient_slots>(providers) / B0;
    else
      return hyperbolic_detail::gradient_provider<0, dimension, gradient_slots>(providers) / B0;
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

/// Compressible exact-ranked Euler flux. The build-specialized name is the native `EulerND<Dim>`.
using CompressibleFlux = Euler;

/// Exact-ranked ISOTHERMAL Euler flux (p = cs2 rho), density plus one momentum per axis.
///
/// Exact-ranked HYPERBOLIC brick with Dim+1 variables (density + one momentum per axis). Satisfies
/// HyperbolicPhysicalModel. Isothermal closure law: p = cs2 * rho (no energy
/// equation). CONTRACT: purely pointwise functions, device-callable (POPS_HD).
/// No MultiFab, no allocation, no global access.
/// Invariant: cs2 > 0 so that the wave speed sqrt(cs2) is real.
template <int Dim>
struct IsothermalFluxND {
  static_assert(Dim >= 1 && Dim <= 3, "IsothermalFluxND supports dimensions 1, 2, and 3");
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Dim + 1;
  using State = StateVec<n_vars>;
  using Prim = StateVec<n_vars>;
  Real cs2 = 1;
  /// Quasi-vacuum density floor (ADC-77). When > 0, the velocity is computed as u = m / max(rho,
  /// vacuum_floor) so it stays bounded where the rollup evacuates the background (rho -> ~0); this
  /// bounds BOTH the CFL wave speed and the advective flux in one place (max_wave_speed and flux both
  /// divide by rho here). Mass and momentum are NOT modified -- only the velocity ESTIMATE is bounded,
  /// so the conservative state is untouched (unlike a cell density clamp). <= 0: inactive, and the raw
  /// 1/rho path is taken verbatim (bit-identical, including for rho <= 0).
  Real vacuum_floor = 0;
  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.physics.hyperbolic.isothermal-flux-nd", 2};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(std::int32_t{Dim}).scalar(cs2).scalar(vacuum_floor);
  }
  POPS_HD static constexpr int momentum_component(int axis) { return 1 + axis; }
  /// rho clamped from below by vacuum_floor for the velocity division ONLY. Manual max (device-safe,
  /// no std:: in the kernel path). floor <= 0 -> returns rho unchanged (bit-identical).
  POPS_HD Real velocity_rho(Real rho) const {
    return (vacuum_floor > Real(0) && rho < vacuum_floor) ? vacuum_floor : rho;
  }
  POPS_HD State flux(const State& u, const auto&, int dir) const {
    const Real rho = u[0];
    const int normal = momentum_component(dir);
    const Real vn = u[normal] / velocity_rho(rho);
    const Real p = cs2 * rho;
    State f{};
    f[0] = u[normal];
    for (int axis = 0; axis < Dim; ++axis)
      f[momentum_component(axis)] = u[momentum_component(axis)] * vn + (axis == dir ? p : Real(0));
    return f;
  }
  /// Conservative -> primitive. Each velocity uses the quasi-vacuum
  /// floored density (velocity_rho); rho itself (p[0]) stays the raw conserved value.
  POPS_HD Prim to_primitive(const State& u) const {
    Prim p{};
    p[0] = u[0];
    const Real rho_v = velocity_rho(u[0]);
    for (int axis = 0; axis < Dim; ++axis)
      p[momentum_component(axis)] = u[momentum_component(axis)] / rho_v;
    return p;
  }
  /// Primitive -> conservative for every native momentum component.
  POPS_HD State to_conservative(const Prim& p) const {
    State u{};
    u[0] = p[0];
    for (int axis = 0; axis < Dim; ++axis)
      u[momentum_component(axis)] = p[0] * p[momentum_component(axis)];
    return u;
  }
  POPS_HD Real max_wave_speed(const State& u, const auto&, int dir) const {
    const Prim p = to_primitive(u);
    const Real vn = p[momentum_component(dir)];
    const Real a = vn < 0 ? -vn : vn;
    return a + std::sqrt(cs2);
  }
  /// Full spectrum: two acoustic waves and Dim-1 shear waves, c = sqrt(cs2).
  POPS_HD State eigenvalues(const State& u, const auto&, int dir) const {
    const Prim p = to_primitive(u);
    const Real vn = p[momentum_component(dir)];
    const Real c = std::sqrt(cs2);
    State e{};
    e[0] = vn - c;
    for (int component = 1; component < Dim; ++component)
      e[component] = vn;
    e[Dim] = vn + c;
    return e;
  }
  /// Signed speeds (HLL/HLLC): v_dir -+ c_s.
  POPS_HD void wave_speeds(const State& u, const auto&, int dir, Real& smin, Real& smax) const {
    const Prim p = to_primitive(u);
    const Real vn = p[momentum_component(dir)];
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
    const int normal = momentum_component(dir);
    const Real density_left = left[0];
    const Real density_right = right[0];
    const Real velocity_left = left[normal] / velocity_rho(density_left);
    const Real velocity_right = right[normal] / velocity_rho(density_right);
    return (pressure_right - pressure_left +
            density_left * velocity_left * (lower - velocity_left) -
            density_right * velocity_right * (upper - velocity_right)) /
           (density_left * (lower - velocity_left) - density_right * (upper - velocity_right));
  }

  /// HLLC star state for an exact-ranked barotropic state.
  POPS_HD State hllc_star_state(const State& value, Real, Real speed, Real contact, int dir) const {
    const int normal = momentum_component(dir);
    const Real density = value[0];
    const Real normal_velocity = value[normal] / velocity_rho(density);
    const Real star_density = density * (speed - normal_velocity) / (speed - contact);
    State result{};
    result[0] = star_density;
    result[normal] = star_density * contact;
    for (int axis = 0; axis < Dim; ++axis) {
      const int component = momentum_component(axis);
      if (axis != dir)
        result[component] = star_density * (value[component] / velocity_rho(density));
    }
    return result;
  }

  /// Roe action |A_roe| dU for the isothermal Euler closure.
  POPS_HD State roe_dissipation(const State& left, const auto&, const State& right, const auto&,
                                int dir) const {
    const int normal = momentum_component(dir);
    const Real density_left = left[0];
    const Real density_right = right[0];
    const Real velocity_left = left[normal] / velocity_rho(density_left);
    const Real velocity_right = right[normal] / velocity_rho(density_right);
    const Real root_left = std::sqrt(density_left);
    const Real root_right = std::sqrt(density_right);
    const Real denominator = root_left + root_right;
    const Real normal_velocity =
        (root_left * velocity_left + root_right * velocity_right) / denominator;
    std::array<Real, Dim> velocity{};
    std::array<Real, Dim> velocity_left_all{};
    std::array<Real, Dim> velocity_right_all{};
    for (int axis = 0; axis < Dim; ++axis) {
      const int component = momentum_component(axis);
      velocity_left_all[axis] = left[component] / velocity_rho(density_left);
      velocity_right_all[axis] = right[component] / velocity_rho(density_right);
      velocity[axis] =
          (root_left * velocity_left_all[axis] + root_right * velocity_right_all[axis]) /
          denominator;
    }
    const Real roe_density = root_left * root_right;
    const Real sound_speed = std::sqrt(cs2);

    const Real density_jump = density_right - density_left;
    const Real normal_jump = velocity_right - velocity_left;
    const Real acoustic_minus =
        (cs2 * density_jump - roe_density * sound_speed * normal_jump) / (Real(2) * cs2);
    const Real acoustic_plus =
        (cs2 * density_jump + roe_density * sound_speed * normal_jump) / (Real(2) * cs2);

    const HartenEntropyFix entropy_fix{Real(0.1)};
    const Real lambda_minus = entropy_fix(normal_velocity - sound_speed, sound_speed);
    const Real lambda_shear = normal_velocity < Real(0) ? -normal_velocity : normal_velocity;
    const Real lambda_plus = entropy_fix(normal_velocity + sound_speed, sound_speed);

    State result{};
    result[0] = lambda_minus * acoustic_minus + lambda_plus * acoustic_plus;
    for (int axis = 0; axis < Dim; ++axis) {
      const Real directional = axis == dir ? sound_speed : Real(0);
      const Real shear = axis == dir
                             ? Real(0)
                             : roe_density * (velocity_right_all[axis] - velocity_left_all[axis]);
      result[momentum_component(axis)] =
          lambda_minus * acoustic_minus * (velocity[axis] - directional) + lambda_shear * shear +
          lambda_plus * acoustic_plus * (velocity[axis] + directional);
    }
    return result;
  }
  static VariableSet conservative_vars() {
    std::vector<std::string> names{"rho"};
    std::vector<VariableRole> roles{VariableRole::Density};
    for (int axis = 0; axis < Dim; ++axis) {
      names.emplace_back("momentum_" + std::to_string(axis));
      roles.push_back(VariableRole::momentum(axis));
    }
    return {VariableKind::Conservative, std::move(names), n_vars, std::move(roles)};
  }
  static VariableSet primitive_vars() {
    std::vector<std::string> names{"rho"};
    std::vector<VariableRole> roles{VariableRole::Density};
    for (int axis = 0; axis < Dim; ++axis) {
      names.emplace_back("velocity_" + std::to_string(axis));
      roles.push_back(VariableRole::velocity(axis));
    }
    return {VariableKind::Primitive, std::move(names), n_vars, std::move(roles)};
  }
};

using IsothermalFlux = IsothermalFluxND<kNativeDimension>;

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
struct IsothermalFluxPolar : IsothermalFluxND<2> {
  static constexpr int dimension = 2;
  static constexpr bool planar_polar_capability = true;

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
