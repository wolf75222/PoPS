#include "generated_fan_li15.hpp"
// Standalone Kokkos witness for the prepared face/residual seam. This does not
// load the generated runtime or qualify an AMR/time-integration campaign.
#include <pops/numerics/spatial/operators/cartesian_operator.hpp>
#include <pops/physics/composition/composite.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pops;
using namespace pops::nd;

namespace {
using Raw = StateVec<15>;

// Independent Gaussian raw moments, using the scalar normal recurrence. It
// does not call the production Hermite builder or analytic path recurrence.
Raw gaussian(Real ux, Real uy = 0) {
  Real x[5]{1, ux}, y[5]{1, uy};
  for (int k = 2; k < 5; ++k) {
    x[k] = ux * x[k - 1] + (k - 1) * x[k - 2];
    y[k] = uy * y[k - 1] + (k - 1) * y[k - 2];
  }
  Raw result{};
  int index = 0;
  for (int q = 0; q <= 4; ++q)
    for (int p = 0; p <= 4 - q; ++p)
      result[index++] = x[p] * y[q];
  return result;
}

template <bool Providers = false, bool ZeroLower = false>
struct PathModel : test_fan_li15::Kernel {
  using State = Raw;
  using Primitive = Raw;
  using Prim = Raw;
  struct Schema {
    using Conservative = Raw;
    using Primitive = Raw;
  };
  static constexpr int dimension = 2, n_vars = 15, n_providers = Providers ? 2 : 0;
  static constexpr bool path_conservative = true;
  static constexpr std::string_view path_operator_identity() {
    return "fan-li15-cartesian-witness-v1";
  }
  POPS_HD static constexpr std::array<bool, 4> path_zero_measure_faces() {
    return {ZeroLower, false, false, false};
  }
  template <int Axis, class Pack>
  POPS_HD std::array<Real, 2> path_covector(const Pack& pack) const {
    Real scale = 1;
    if constexpr (Providers)
      scale = pack.template provider<Axis>();
    if constexpr (Axis == 0)
      return {scale, 0};
    return {0, scale};
  }
  POPS_HD bool path_admissible(const Raw& state) const {
    Real raw[15];
    for (int k = 0; k < 15; ++k)
      raw[k] = state[k];
    return test_fan_li15::Kernel::admissibility(raw) == pops::PathStatus::Success;
  }
  POPS_HD StateConversionStatus admissibility(const Raw& state) const {
    return path_admissible(state) ? StateConversionStatus::Success
                                  : StateConversionStatus::NonPositivePressure;
  }
  POPS_HD StateConversion<Raw> recover(const Raw& state) const {
    return {state, admissibility(state)};
  }
  POPS_HD StateConversion<Raw> make_conservative(const Raw& state) const { return recover(state); }
  template <int Axis, class Pack>
  POPS_HD Raw flux(const Raw& state, const Pack& pack) const {
    const auto g = path_covector<Axis>(pack);
    Real raw[15];
    for (int k = 0; k < 15; ++k)
      raw[k] = state[k];
    const auto value = test_fan_li15::flux(raw, g[0], g[1]);
    Raw result{};
    for (int k = 0; k < 15; ++k)
      result[k] = value.flux.values[k];
    return result;
  }
  template <int Axis, class Pack>
  POPS_HD Real max_wave_speed(const Raw& state, const Pack& pack) const {
    const auto g = path_covector<Axis>(pack);
    Real raw[15];
    for (int k = 0; k < 15; ++k)
      raw[k] = state[k];
    const auto result = test_fan_li15::path_integral(raw, raw, g[0], g[1]);
    return result.succeeded() ? result.speed_bound : std::numeric_limits<Real>::quiet_NaN();
  }
};
struct EmptyBrick {
  static constexpr int dimension = 2, n_providers = 0;
};
using Composite = CompositeModel<PathModel<true>, EmptyBrick, EmptyBrick>;
static_assert(path_conservative_model<Composite> && ConservationLaw<2, Composite>);
static_assert(!path_conservative_model<ScalarAdvection<2>>);
using PathPhysical = PhysicalFluxView<PathModel<>>;
template <class Policy, class Physical>
concept DirectPolicyCallable =
    requires(const Policy& policy, const Physical& physical, const typename Physical::Trace& trace,
             const FaceContext& face) { policy(physical, trace, trace, face); };
static_assert(PhysicalFlux<PathPhysical> && !OrdinaryPhysicalFlux<PathPhysical>);
static_assert(!DirectPolicyCallable<RusanovFlux, PathPhysical>);
static_assert(!DirectPolicyCallable<HLLFlux, PathPhysical>);
static_assert(!DirectPolicyCallable<HLLCFlux, PathPhysical>);
static_assert(!DirectPolicyCallable<RoeFlux, PathPhysical>);
static_assert(!DirectPolicyCallable<
              PreparedRiemannRecoveryPolicy<RusanovFlux, RejectRiemannRecovery>, PathPhysical>);
static_assert(DirectPolicyCallable<RusanovFlux, PhysicalFluxView<ScalarAdvection<2>>>);
static_assert(NumericalFlux<RusanovFlux, PhysicalFluxView<ScalarAdvection<2>>>);

std::size_t offset(const Box<2>& box, const Index<2>& index, int c) {
  return static_cast<std::size_t>(c * box.numPts() + index[0] - box.lo[0] +
                                  (index[1] - box.lo[1]) * box.length(0));
}
template <class Function>
void fill(Fab<2>& field, Function value) {
  auto host = field.create_host_mirror();
  const auto box = field.grown_box();
  for (int j = box.lo[1]; j <= box.hi[1]; ++j)
    for (int i = box.lo[0]; i <= box.hi[0]; ++i)
      for (int c = 0; c < field.ncomp(); ++c)
        host(offset(box, Index<2>{i, j}, c)) = value(i, j, c);
  field.copy_from_host(host);
}
Real get(const Fab<2>& field, std::array<int, 2> coordinates, int component = 0) {
  auto host = field.create_host_mirror();
  field.copy_to_host(host);
  return host(offset(field.grown_box(), Index<2>{coordinates[0], coordinates[1]}, component));
}
void near(Real actual, Real expected, const char* what) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > 2e-12 * (1 + std::abs(expected)))
    throw std::runtime_error(std::string(what) + ": actual=" + std::to_string(actual) +
                             " expected=" + std::to_string(expected));
}
void check(bool condition, const char* what) {
  if (!condition)
    throw std::runtime_error(what);
}
template <class Function>
void refuses(Function action, const char* what) {
  bool caught = false;
  try {
    action();
  } catch (const std::exception&) {
    caught = true;
  }
  check(caught, what);
}
struct Tuple {
  FaceField<2> f, l, r, speed;
  PreparedCartesianPathFaceScratch<2> scratch;
  explicit Tuple(const Box<2>& box)
      : f(box, 15), l(box, 15), r(box, 15), speed(box, 1), scratch(box, 15) {}
  void sentinel() {
    f.set_val(123);
    l.set_val(124);
    r.set_val(125);
    speed.set_val(126);
  }
  void unchanged() const {
    const auto all_equal = [](const Fab<2>& field, Real expected, const char* what) {
      auto host = field.create_host_mirror();
      field.copy_to_host(host);
      for (std::size_t i = 0; i < field.size(); ++i)
        near(host(i), expected, what);
    };
    all_equal(f.field<0>(), 123, "F transaction axis0");
    all_equal(f.field<1>(), 123, "F transaction axis1");
    all_equal(l.field<0>(), 124, "L transaction axis0");
    all_equal(l.field<1>(), 124, "L transaction axis1");
    all_equal(r.field<0>(), 125, "R transaction axis0");
    all_equal(r.field<1>(), 125, "R transaction axis1");
    all_equal(speed.field<0>(), 126, "speed transaction axis0");
    all_equal(speed.field<1>(), 126, "speed transaction axis1");
  }
};

std::vector<std::string> passed;
std::atomic<std::uint64_t> evaluation_allocations{0};
void record(const char* name) {
  passed.emplace_back(name);
}

// A different physical system: u_t + u_x = 0, v_t + u v_x = 0.
// Its straight path has P=(0, mean(u_L,u_R)*(v_R-v_L)), with full-path
// speed max(1,|u_L|,|u_R|). No moment closure or density domain is involved.
struct TriangularPathModel {
  using State = StateVec<2>;
  using Primitive = State;
  struct Schema {
    using Conservative = State;
    using Primitive = State;
  };
  static constexpr int dimension = 1, n_vars = 2, n_providers = 0;
  static constexpr bool path_conservative = true;
  static constexpr std::string_view path_operator_identity() { return "triangular-2x1-raw-path@1"; }
  POPS_HD static constexpr std::array<bool, 2> path_zero_measure_faces() { return {}; }
  template <int Axis, class Providers>
  POPS_HD std::array<Real, 1> path_covector(const Providers&) const {
    static_assert(Axis == 0);
    return {Real(1)};
  }
  POPS_HD StateConversionStatus admissibility(const State& state) const {
    return std::isfinite(state[0]) && std::isfinite(state[1])
               ? StateConversionStatus::Success
               : StateConversionStatus::NonFiniteState;
  }
  POPS_HD StateConversion<State> recover(const State& state) const {
    return {state, admissibility(state)};
  }
  POPS_HD StateConversion<State> make_conservative(const State& state) const {
    return recover(state);
  }
  template <int Axis, class Providers>
  POPS_HD State flux(const State& state, const Providers&) const {
    State result{};
    result[0] = state[0];
    return result;
  }
  template <int Axis, class Providers>
  POPS_HD Real max_wave_speed(const State& state, const Providers&) const {
    return std::abs(state[0]) > Real(1) ? std::abs(state[0]) : Real(1);
  }
  POPS_HD PathFluxResult<2> path_directional_flux(const State& state,
                                                  const std::array<Real, 1>& g) const {
    PathFluxResult<2> result;
    result.status = PathStatus::Success;
    result.flux.values[0] = g[0] * state[0];
    return result;
  }
  POPS_HD PathIntegralResult<2> path_integral(const State& left, const State& right,
                                              const std::array<Real, 1>& g) const {
    PathIntegralResult<2> result;
    result.status = PathStatus::Success;
    result.integral[1] = g[0] * (Real(0.5) * (left[0] + right[0])) * (right[1] - left[1]);
    result.speed_bound =
        std::abs(g[0]) * std::max(Real(1), std::max(std::abs(left[0]), std::abs(right[0])));
    return result;
  }
};

void distinct_nonconservative_rank_control() {
  const auto box = Box<1>::from_extents(Extent<1>{2});
  const auto geometry = Geometry<1>::from_bounds(box, RealVector<1>{0}, RealVector<1>{2});
  const auto op = prepare_cartesian_operator<1>(geometry, TriangularPathModel{}, NoSlope{},
                                                PathRusanovFlux<2>{});
  Fab<1> state(box, 2, Extent<1>{1}), residual(box, 2), candidate(box, 2), statuses(box, 1);
  auto input = state.create_host_mirror();
  const auto grown = state.grown_box();
  for (int i = grown.lo[0]; i <= grown.hi[0]; ++i) {
    input(i - grown.lo[0]) = Real(2);
    input(grown.numPts() + i - grown.lo[0]) = Real(i);
  }
  state.copy_from_host(input);
  FaceField<1> flux(box, 2), left(box, 2), right(box, 2), speed(box, 1);
  PreparedCartesianPathFaceScratch<1> scratch(box, 2);
  op.materialize_path_face_contributions(state, flux, left, right, speed, scratch);
  op.assemble_residual_from_path_faces(flux, left, right, residual, candidate, statuses);
  auto output = residual.create_host_mirror();
  residual.copy_to_host(output);
  for (int i = 0; i < 2; ++i) {
    near(output(i), 0, "N2/Dim1 conservative row");
    near(output(2 + i), -2, "N2/Dim1 nonconservative transport");
  }
  auto ncp = left.field<0>().create_host_mirror();
  left.field<0>().copy_to_host(ncp);
  for (int i = 0; i < 3; ++i)
    near(ncp(3 + i), -1, "N2/Dim1 independent nonzero side term");
  record("distinct_N2_Dim1_nonzero_nonconservative_system");
}

struct MixedSignSpeedLaw {
  POPS_HD PathStatus recover(const Real (&raw)[6], moments::NormalizedRawMoments<2>& state) const {
    return moments::recover_raw_moments<2>(raw, state);
  }
  POPS_HD Real speed_bound(const moments::NormalizedRawMoments<2>& state,
                           const std::array<Real, 2>&) const {
    return state.density == Real(1) ? Real(-1) : Real(2);
  }
  POPS_HD void path_polynomials(const moments::NormalizedRawMoments<2>&,
                                const moments::NormalizedRawMoments<2>&, const std::array<Real, 2>&,
                                moments::Polynomial<2> (&)[6], Real (&)[6]) const {}
};

void moment_primitive_order_control() {
  const Real raw[6] = {1, 0, 1, 0, 0, 2}, other[6] = {2, 0, 2, 0, 0, 4};
  moments::NormalizedRawMoments<2> state;
  check(moments::recover_raw_moments<2>(raw, state) == PathStatus::Success,
        "degree-two generic moment recovery");
  Real h[6], temperature[3], third[4];
  moments::normalized_hermite(state, h, temperature);
  near(h[0], 1, "normalized Hermite density");
  for (int k = 1; k < 6; ++k)
    near(h[k], 0, "normalized first and second Hermite coefficients");
  moments::hermite_raw_edge<2, 3>(state, h, temperature, third);
  for (Real value : third)
    near(value, 0, "centered Gaussian degree-three edge");
  const auto refusal = moments::integrate_normalized_moment_path<2>(
      raw, other, std::array<Real, 2>{1, 0}, MixedSignSpeedLaw{});
  check(refusal.status == PathStatus::NonFiniteResult,
        "negative endpoint speed must not be hidden by a positive endpoint");
  record("generic_degree2_moment_primitives_and_endpoint_speed_refusal");
}

template <int Dim>
void ordinary_rank_control() {
  Extent<Dim> extent;
  RealVector<Dim> lower{}, upper{}, velocity{};
  for (int d = 0; d < Dim; ++d) {
    extent[d] = 2;
    upper[d] = 2;
    velocity[d] = 1;
  }
  const auto box = Box<Dim>::from_extents(extent);
  const auto geometry = Geometry<Dim>::from_bounds(box, lower, upper);
  Fab<Dim> state(box, 1, extent), rhs(box, 1);
  state.set_val(1);
  const auto op =
      prepare_cartesian_operator<Dim>(geometry, ScalarAdvection<Dim>::prepare(velocity));
  op.assemble_residual(state, rhs);
  auto host = rhs.create_host_mirror();
  rhs.copy_to_host(host);
  for (std::size_t i = 0; i < rhs.size(); ++i)
    near(host(i), 0, "ordinary rank constant-state preservation");
}

void run() {
  const Real small = std::numeric_limits<Real>::denorm_min(),
             large = std::numeric_limits<Real>::max();
  for (Real a : {small, -small, Real(0), Real(1), large, -large})
    for (Real b : {small, -small, Real(0), Real(1), large, -large}) {
      check(common_covector_component(a, b) == std::midpoint(a, b),
            "common covector midpoint differs from independent standard implementation");
      check(common_covector_component(a, b) == common_covector_component(b, a),
            "common covector is not symmetric");
    }
  record("common_covector_subnormal_overflow_midpoint");
  const auto box = Box<2>::from_extents(Extent<2>{2, 1});
  // dx=2, dy=3: x-face area=3, cell volume=6.
  const auto geometry = Geometry<2>::from_bounds(box, RealVector<2>{0, 0}, RealVector<2>{4, 3});
  const auto op =
      prepare_cartesian_operator<2>(geometry, Composite{}, NoSlope{}, PathRusanovFlux<15>{});
  Fab<2> state(box, 15, Extent<2>{1, 1}), providers(box, 2, Extent<2>{1, 1});
  fill(state, [](int i, int, int c) { return gaussian(i <= 0 ? 0 : 1)[c]; });
  fill(providers, [](int i, int, int c) { return c == 0 ? (i <= 0 ? 1. : 3.) : 1.; });
  Tuple tuple(box);
  op.materialize_path_face_contributions(state, providers, tuple.f, tuple.l, tuple.r, tuple.speed,
                                         tuple.scratch);
  const Real a = 2 * std::sqrt(6 + std::sqrt(10.)) * std::sqrt(2.);
  // Exact straight-Gaussian raw-path P40 at g=(2,0) is -1/3.
  near(get(tuple.l.field<0>(), {1, 0}, 4), .5, "integrated left -P40/2");
  near(get(tuple.r.field<0>(), {1, 0}, 4), .5, "integrated right -P40/2");
  near(get(tuple.f.field<0>(), {1, 0}, 4), 3 * (26 - 3.5 * a), "Grad fifth plus Rusanov");
  near(get(tuple.f.field<0>(), {1, 0}, 0), 3, "common covector density flux");
  near(get(tuple.speed.field<0>(), {1, 0}), a, "unintegrated common-g path bound");
  for (int k : {0, 1, 2, 3, 5, 6, 7, 9, 10, 12}) {
    near(get(tuple.l.field<0>(), {1, 0}, k), 0, "lower degree left side exact zero");
    near(get(tuple.r.field<0>(), {1, 0}, k), 0, "lower degree right side exact zero");
  }
  record("common_covector_fifth_flux_bound_metric_once_conserved_rows");
  near(get(tuple.l.field<0>(), {0, 0}, 4), 0, "equal outflow path zero");
  near(get(tuple.f.field<0>(), {2, 0}, 4), 234, "equal outflow still evaluates Grad fifth");
  record("equal_outflow_is_evaluated");
  Fab<2> rhs(box, 15), candidate(box, 15), statuses(box, 1);
  op.assemble_residual_from_path_faces(tuple.f, tuple.l, tuple.r, rhs, candidate, statuses);
  const Real fs = 3 * (26 - 3.5 * a);
  near(get(rhs, {0, 0}, 4), (-fs + .5) / 6, "left cell top-row residual sign");
  near(get(rhs, {1, 0}, 4), (fs - 234 + .5) / 6, "right cell top-row residual sign");
  near(6 * (get(rhs, {0, 0}, 0) + get(rhs, {1, 0}, 0)), -9, "mass boundary telescoping");
  record("two_cell_residual_sign_volume_and_mass_telescoping");
  const auto scratch_pointer = tuple.scratch.flux().view().axes[0].data;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      [](Kokkos::Tools::SpaceHandle, const char*, const void*, std::uint64_t) {
        ++evaluation_allocations;
      });
  op.materialize_path_face_contributions(state, providers, tuple.f, tuple.l, tuple.r, tuple.speed,
                                         tuple.scratch);
  op.assemble_residual_from_path_faces(tuple.f, tuple.l, tuple.r, rhs, candidate, statuses);
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  check(evaluation_allocations.load() == 0, "prepared evaluation allocated Kokkos data");
  check(scratch_pointer == tuple.scratch.flux().view().axes[0].data,
        "prepared scratch was replaced");
  record("prepared_scratch_reuse");

  ProviderStorageView<2, 2> storage{};
  for (int c = 0; c < 2; ++c) {
    storage.storage[c] = static_cast<const Fab<2>&>(providers).view();
    storage.storage_components[c] = c;
  }
  op.materialize_path_face_contributions(state, storage, tuple.f, tuple.l, tuple.r, tuple.speed,
                                         tuple.scratch);
  near(get(tuple.speed.field<0>(), {1, 0}), a, "plan-mapped provider route");
  record("provider_storage_view_matches_provider_fab");
  // Failure at the last x boundary must preserve the entire published tuple.
  fill(state,
       [](int i, int, int c) { return i == 2 && c == 0 ? -1. : gaussian(i <= 0 ? 0 : 1)[c]; });
  tuple.sentinel();
  refuses(
      [&] {
        op.materialize_path_face_contributions(state, providers, tuple.f, tuple.l, tuple.r,
                                               tuple.speed, tuple.scratch);
      },
      "bad final face accepted");
  tuple.unchanged();
  near(get(tuple.scratch.status().field<0>(), {2, 0}), 1026, "typed density failure retained");
  record("late_face_failure_publishes_no_tuple_field");
  fill(state, [](int i, int, int c) { return gaussian(i <= 0 ? 0 : 1)[c]; });
  auto deep_copy = tuple.f;
  check(deep_copy.view().axes[0].data != tuple.f.view().axes[0].data,
        "Fab copy must own distinct storage");
  auto& aliased_f = tuple.f;
  refuses(
      [&] {
        op.materialize_path_face_contributions(state, providers, tuple.f, aliased_f, tuple.r,
                                               tuple.speed, tuple.scratch);
      },
      "face output alias accepted");
  tuple.unchanged();
  record("face_alias_refused_deep_copy_independent");
  op.materialize_path_face_contributions(state, providers, tuple.f, tuple.l, tuple.r, tuple.speed,
                                         tuple.scratch);
  fill(tuple.r.field<0>(), [](int i, int, int c) {
    return i == 1 && c == 4 ? std::numeric_limits<Real>::quiet_NaN() : 0.;
  });
  rhs.set_val(127);
  refuses(
      [&] {
        op.assemble_residual_from_path_faces(tuple.f, tuple.l, tuple.r, rhs, candidate, statuses);
      },
      "bad side residual accepted");
  near(get(rhs, {0, 0}, 4), 127, "RHS transaction left");
  near(get(rhs, {1, 0}, 4), 127, "RHS transaction right");
  record("side_nonfinite_refuses_whole_residual");
  auto& alias_rhs = rhs;
  refuses(
      [&] {
        op.assemble_residual_from_path_faces(tuple.f, tuple.l, tuple.r, rhs, alias_rhs, statuses);
      },
      "residual output alias accepted");
  record("residual_alias_refused");

  tuple.sentinel();
  refuses([&] { op.materialize_face_fluxes(state, providers, tuple.f); },
          "ordinary faces accepted path model");
  refuses([&] { op.assemble_residual(state, providers, rhs); },
          "ordinary residual accepted path model");
  refuses([&] { op.assemble_residual_from_face_fluxes(tuple.f, rhs); },
          "ordinary face divergence accepted path model");
  const auto rejected = evaluate_numerical_flux_at(
      RusanovFlux{}, Composite{}, gaussian(0), ProviderStorageView<2, 2>{}, Index<2>{0, 0},
      gaussian(1), ProviderStorageView<2, 2>{}, Index<2>{1, 0}, FaceContext::axis_aligned(0));
  check(rejected.reason_code ==
            riemann_reason_code(RiemannFailureCause::kNonconservativePathRequired),
        "ordinary flux lost typed path refusal");
  tuple.unchanged();
  record("ordinary_flux_materialize_residual_refuse_before_provider_binding");

  const auto default_carrier = prepare_cartesian_operator<2>(geometry, Composite{});
  refuses(
      [&] {
        default_carrier.materialize_path_face_contributions(state, providers, tuple.f, tuple.l,
                                                            tuple.r, tuple.speed, tuple.scratch);
      },
      "default ordinary Rusanov carrier accepted dedicated path preparation");
  tuple.unchanged();
  const auto ordinary_carrier =
      prepare_cartesian_operator<2>(geometry, Composite{}, NoSlope{}, RusanovFlux{});
  refuses(
      [&] {
        ordinary_carrier.materialize_path_face_contributions(state, providers, tuple.f, tuple.l,
                                                             tuple.r, tuple.speed, tuple.scratch);
      },
      "explicit ordinary Rusanov carrier accepted dedicated path preparation");
  tuple.unchanged();
  record("ordinary_and_default_carriers_refuse_path_preparation");

  const PathPhysical grad_formula{PathModel<>{}};
  const auto physical_trace = make_face_trace<PathModel<>>(
      gaussian(1), bind_flux_providers<PathModel<>>(FluxProviderValues<PathModel<>>{}));
  const auto physical_flux = grad_formula.evaluate(physical_trace, FaceContext::axis_aligned(0));
  check(physical_flux.status == EvaluationStatus::kOk,
        "pure Grad physical evaluation must remain available");
  near(physical_flux.value[4], 26, "pure physical Grad fifth formula");
  record("direct_ordinary_policies_unavailable_physical_grad_available");
  refuses(
      [&] {
        op.materialize_path_face_contributions(state, providers, tuple.f, tuple.l, tuple.r,
                                               tuple.speed, tuple.scratch,
                                               {true, false, false, false});
      },
      "unauthorized omitted face accepted");
  record("ordinary_noflux_omission_refused");

  const auto zero_op = prepare_cartesian_operator<2>(geometry, PathModel<true, true>{}, NoSlope{},
                                                     PathRusanovFlux<15>{});
  fill(state, [](int i, int, int c) { return i < 0 ? 0. : gaussian(0)[c]; });
  fill(providers,
       [](int i, int, int) { return i < 0 ? std::numeric_limits<Real>::quiet_NaN() : 1.; });
  zero_op.materialize_path_face_contributions(state, providers, tuple.f, tuple.l, tuple.r,
                                              tuple.speed, tuple.scratch);
  for (int c = 0; c < 15; ++c) {
    near(get(tuple.f.field<0>(), {0, 0}, c), 0, "zero-measure F");
    near(get(tuple.l.field<0>(), {0, 0}, c), 0, "zero-measure L");
    near(get(tuple.r.field<0>(), {0, 0}, c), 0, "zero-measure R");
  }
  near(get(tuple.speed.field<0>(), {0, 0}), 0, "zero-measure speed");
  record("authored_zero_face_skips_invalid_state_and_provider_ghosts");
  const auto free_op = prepare_cartesian_operator<2>(geometry, PathModel<false, true>{}, NoSlope{},
                                                     PathRusanovFlux<15>{});
  free_op.materialize_path_face_contributions(state, tuple.f, tuple.l, tuple.r, tuple.speed,
                                              tuple.scratch);
  record("provider_free_path_route");
  refuses(
      [&] {
        const auto wrong = prepare_cartesian_operator<2>(geometry, PathModel<false, true>{},
                                                         Minmod{}, PathRusanovFlux<15>{});
        wrong.materialize_path_face_contributions(state, tuple.f, tuple.l, tuple.r, tuple.speed,
                                                  tuple.scratch);
      },
      "higher reconstruction accepted");
  record("non_firstorder_refused");
  for (Real floor : {Real(-1), Real(1), std::numeric_limits<Real>::quiet_NaN()}) {
    refuses(
        [&] {
          const auto wrong = prepare_cartesian_operator<2>(geometry, PathModel<false, true>{},
                                                           NoSlope{}, PathRusanovFlux<15>{}, floor);
          wrong.materialize_path_face_contributions(state, tuple.f, tuple.l, tuple.r, tuple.speed,
                                                    tuple.scratch);
        },
        "nonzero or nonfinite floor accepted");
  }
  record("no_state_floor_or_heating");
  check(Composite::path_operator_identity() == PathModel<true>::path_operator_identity(),
        "composite identity forwarding");
  check(Composite{}.path_admissible(gaussian(0)), "composite guard forwarding");
  check(!Composite{}.path_admissible(Raw{}), "composite guard refusal forwarding");
  record("composite_sole_hyperbolic_path_forwarding");
  Fab<2> scalar(box, 1, Extent<2>{1, 1}), scalar_rhs(box, 1);
  fill(scalar, [](int i, int, int) { return Real(i); });
  const auto ordinary =
      prepare_cartesian_operator<2>(geometry, ScalarAdvection<2>::prepare(RealVector<2>{1, 0}));
  ordinary.assemble_residual(scalar, scalar_rhs);
  near(get(scalar_rhs, {0, 0}), -.5, "ordinary scalar result");
  near(get(scalar_rhs, {1, 0}), -.5, "ordinary scalar result");
  record("ordinary_scalar_behavior_control");
  ordinary_rank_control<1>();
  ordinary_rank_control<3>();
  record("ordinary_rank1_rank3_compilation_and_constant_state");
  distinct_nonconservative_rank_control();
  moment_primitive_order_control();
  std::cout << std::setprecision(17)
            << "{\"scope\":\"isolated Cartesian Kokkos witness; no generated runtime, AMR, MPI or "
               "time stepping\",\"speed\":"
            << a << ",\"shared_F40\":" << fs << ",\"execution_space\":\""
            << Kokkos::DefaultExecutionSpace::name()
            << "\",\"concurrency\":" << Kokkos::DefaultExecutionSpace().concurrency()
            << ",\"repeat_evaluation_kokkos_allocations\":" << evaluation_allocations.load()
            << ",\"tests\":[";
  for (std::size_t i = 0; i < passed.size(); ++i)
    std::cout << (i ? "," : "") << '"' << passed[i] << '"';
  std::cout << "]}\n";
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int result = 0;
  try {
    run();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    result = 1;
  }
  Kokkos::finalize();
  return result;
}
