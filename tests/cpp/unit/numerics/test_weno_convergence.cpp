// WENO5-Z : la reconstruction de la valeur de face d'une fonction LISSE depuis ses moyennes
// de cellule est d'ordre 5 (les poids non lineaires WENO-Z tendent vers les poids lineaires
// optimaux 1/10, 6/10, 3/10 en zone reguliere). On verifie l'ordre mesure >= 4.5 et la
// preservation des constantes. Brique de la voie haute precision vers le taux analytique 0.911.

#include <gtest/gtest.h>

#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/numerics/spatial/primitives/face_flux.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using namespace pops;

namespace {
constexpr double kPi = 3.14159265358979323846;

struct WideSlopePolicy {
  static constexpr int formal_order = 2;
  static constexpr int n_ghost = 4;
  POPS_HD Real limited_slope(Real backward, Real forward) const {
    return Real(0.5) * (backward + forward);
  }
};

/// Deliberately not WENO: four non-contiguous samples and a different storage requirement prove
/// that the spatial core does not own a five-point stencil.  The coefficients sum to one.
struct ExternalFourSamplePolicy {
  static constexpr int formal_order = 1;
  static constexpr int n_ghost = 4;
  static constexpr int stencil_min_offset = -3;
  static constexpr int stencil_max_offset = 2;

  template <class Sample>
  POPS_HD Real stencil_face_value(const Sample& sample) const {
    return (sample(-3) + Real(2) * sample(-1) + Real(3) * sample(0) + Real(4) * sample(2)) /
           Real(10);
  }
};

struct AmbiguousPolicy {
  static constexpr int formal_order = 1;
  static constexpr int n_ghost = 1;
  static constexpr int stencil_min_offset = 0;
  static constexpr int stencil_max_offset = 0;
  POPS_HD Real cell_face_value(Real value) const { return value; }
  template <class Sample>
  POPS_HD Real stencil_face_value(const Sample& sample) const {
    return sample(0);
  }
};

struct MissingPolicy {};

/// Minimal conservative model used only to exercise the production face-state reconstruction
/// protocol.  The qualification below never calls a limiter formula directly.
struct ScalarReconstructionModel {
  using State = StateVec<1>;
  static constexpr int n_vars = 1;
};

struct PeriodicFaceStates {
  std::vector<Real> left;
  std::vector<Real> right;
  bool publication_permitted = true;
};

double favg(double a, double b);

int periodic_index(int index, int size) {
  const int remainder = index % size;
  return remainder < 0 ? remainder + size : remainder;
}

template <class Limiter>
PeriodicFaceStates reconstruct_periodic_faces(const std::vector<Real>& cell_averages) {
  const int size = static_cast<int>(cell_averages.size());
  Fab2D values(Box2D::from_extents(size, 1), ScalarReconstructionModel::n_vars, Limiter::n_ghost);
  for (int i = values.grown_box().lo[0]; i <= values.grown_box().hi[0]; ++i)
    values(i, 0, 0) = cell_averages[periodic_index(i, size)];

  const ScalarReconstructionModel model{};
  const Limiter limiter{};
  PeriodicFaceStates result{std::vector<Real>(size), std::vector<Real>(size), true};
  for (int i = 0; i < size; ++i) {
    const auto left =
        reconstruct_recovered(model, values.const_array(), i, 0, 0, Real(-1), limiter, false);
    const auto right =
        reconstruct_recovered(model, values.const_array(), i, 0, 0, Real(1), limiter, false);
    result.publication_permitted = result.publication_permitted && left.publication_permitted() &&
                                   right.publication_permitted();
    result.left[i] = left.value[0];
    result.right[i] = right.value[0];
  }
  return result;
}

struct SmoothFaceError {
  double l1 = 0;
  bool publication_permitted = false;
};

template <class Limiter>
SmoothFaceError smooth_periodic_face_error(int size) {
  const double dx = 1.0 / static_cast<double>(size);
  std::vector<Real> cell_averages(size);
  for (int i = 0; i < size; ++i)
    cell_averages[i] = Real(favg(i * dx, (i + 1) * dx));

  const auto reconstructed = reconstruct_periodic_faces<Limiter>(cell_averages);
  double error = 0;
  for (int i = 0; i < size; ++i) {
    const double exact = std::sin(Real(2) * kPi * (i + 1) * dx);
    error += std::fabs(static_cast<double>(reconstructed.right[i]) - exact);
  }
  return {error / static_cast<double>(size), reconstructed.publication_permitted};
}

double interface_jump_budget(const PeriodicFaceStates& states) {
  double result = 0;
  const int size = static_cast<int>(states.left.size());
  for (int i = 0; i < size; ++i)
    result += std::fabs(static_cast<double>(states.left[(i + 1) % size] - states.right[i]));
  return result;
}

/// A nonlinear conservative/primitive conversion makes the two reconstruction paths observably
/// different while remaining exactly invertible for the positive test data.
struct PrimitiveTestModel {
  using State = StateVec<2>;
  using Prim = StateVec<2>;
using Providers = pops::ProviderValues<0>;
  static constexpr int n_vars = 2;
  int* primitive_calls = nullptr;

  POPS_HD State flux(const State& state, const auto&, int) const { return state; }
  POPS_HD Real max_wave_speed(const State&, const auto&, int) const { return Real(1); }
POPS_HD State source(const State&, const Providers&) const { return State{}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }

  POPS_HD Prim to_primitive(const State& state) const {
    if (primitive_calls != nullptr)
      ++*primitive_calls;
    return Prim{state[0] * state[0], state[1]};
  }
  POPS_HD State to_conservative(const Prim& primitive) const {
    using std::sqrt;
    return State{sqrt(primitive[0]), primitive[1]};
  }
};

static_assert(SlopeReconstruction<WideSlopePolicy>);
static_assert(ReconstructionPolicy<MC>);
static_assert(ReconstructionPolicy<Superbee>);
static_assert(MC::formal_order == 2 && MC::n_ghost == 2);
static_assert(Superbee::formal_order == 2 && Superbee::n_ghost == 2);
static_assert(!StencilReconstruction<WideSlopePolicy>);
static_assert(StencilReconstruction<ExternalFourSamplePolicy>);
static_assert(ReconstructionPolicy<ExternalFourSamplePolicy>);
static_assert(reconstruction_protocol_count<AmbiguousPolicy> == 2);
static_assert(!ReconstructionPolicy<AmbiguousPolicy>);
static_assert(reconstruction_protocol_count<MissingPolicy> == 0);
static_assert(!ReconstructionPolicy<MissingPolicy>);

// moyenne de cellule de f(x) = sin(2 pi x) sur [a, b] (primitive exacte).
double favg(double a, double b) {
  return (std::cos(2 * kPi * a) - std::cos(2 * kPi * b)) / (2 * kPi * (b - a));
}
}  // namespace

TEST(test_weno_convergence, preserves_constants) {
  // weno5z(c,c,c,c,c) == c (poids sommes a 1).
  const double c = 3.14;
  EXPECT_LE(std::fabs(weno5z(c, c, c, c, c) - c), 1e-13) << "constante";
}

TEST(test_weno_convergence, reconstruction_protocol_is_independent_of_storage_radius) {
  const auto policy = configured_reconstruction<WideSlopePolicy>();
  EXPECT_EQ(policy.limited_slope(Real(2), Real(4)), Real(3));
}

TEST(test_muscl_limiters, mc_and_superbee_match_reference_formulas) {
  const MC mc{};
  const Superbee superbee{};

  EXPECT_EQ(mc.limited_slope(Real(1), Real(3)), Real(2));
  EXPECT_EQ(mc.limited_slope(Real(2), Real(4)), Real(3));
  EXPECT_EQ(mc.limited_slope(Real(3), Real(1)), Real(2));
  EXPECT_EQ(superbee.limited_slope(Real(1), Real(3)), Real(2));
  EXPECT_EQ(superbee.limited_slope(Real(2), Real(4)), Real(4));
  EXPECT_EQ(superbee.limited_slope(Real(3), Real(1)), Real(2));
}

TEST(test_muscl_limiters, zero_opposite_sign_symmetry_and_homogeneity) {
  const MC mc{};
  const Superbee superbee{};
  for (const auto limiter : {0, 1}) {
    const auto slope = [&](Real a, Real b) {
      return limiter == 0 ? mc.limited_slope(a, b) : superbee.limited_slope(a, b);
    };
    EXPECT_EQ(slope(Real(0), Real(4)), Real(0));
    EXPECT_EQ(slope(Real(4), Real(0)), Real(0));
    EXPECT_EQ(slope(Real(-2), Real(3)), Real(0));
    EXPECT_EQ(slope(Real(2), Real(-3)), Real(0));
    EXPECT_EQ(slope(Real(-2), Real(-4)), -slope(Real(2), Real(4)));
    EXPECT_EQ(slope(Real(6), Real(12)), Real(3) * slope(Real(2), Real(4)));
  }
}

TEST(test_muscl_limiters, sweby_tvd_bounds_and_finite_extremes) {
  const MC mc{};
  const Superbee superbee{};
  for (const Real backward : {Real(0.25), Real(1), Real(2), Real(8)}) {
    for (const Real forward : {Real(0.5), Real(1), Real(4), Real(16)}) {
      const Real tvd_bound = Real(2) * std::min(backward, forward);
      for (const Real slope :
           {mc.limited_slope(backward, forward), superbee.limited_slope(backward, forward)}) {
        EXPECT_GE(slope, Real(0));
        EXPECT_LE(slope, tvd_bound);
      }
    }
  }

  const Real maximum = std::numeric_limits<Real>::max();
  EXPECT_TRUE(std::isfinite(mc.limited_slope(maximum, maximum)));
  EXPECT_TRUE(std::isfinite(superbee.limited_slope(maximum, maximum)));
  EXPECT_EQ(mc.limited_slope(maximum, maximum), maximum);
  EXPECT_EQ(superbee.limited_slope(maximum, maximum), maximum);
  EXPECT_EQ(mc.limited_slope(-maximum, -maximum), -maximum);
  EXPECT_EQ(superbee.limited_slope(-maximum, -maximum), -maximum);
}

TEST(test_muscl_limiter_qualification,
     mc_and_superbee_are_second_order_on_smooth_periodic_cell_averages) {
  const auto mc_128 = smooth_periodic_face_error<MC>(128);
  const auto mc_256 = smooth_periodic_face_error<MC>(256);
  const auto superbee_128 = smooth_periodic_face_error<Superbee>(128);
  const auto superbee_256 = smooth_periodic_face_error<Superbee>(256);
  const auto minmod_256 = smooth_periodic_face_error<Minmod>(256);
  const auto vanleer_256 = smooth_periodic_face_error<VanLeer>(256);

  EXPECT_TRUE(mc_128.publication_permitted && mc_256.publication_permitted);
  EXPECT_TRUE(superbee_128.publication_permitted && superbee_256.publication_permitted);
  EXPECT_TRUE(minmod_256.publication_permitted && vanleer_256.publication_permitted);

  const double mc_order = std::log(mc_128.l1 / mc_256.l1) / std::log(2.0);
  const double superbee_order = std::log(superbee_128.l1 / superbee_256.l1) / std::log(2.0);
  EXPECT_GT(mc_order, 1.85);
  EXPECT_LT(mc_order, 2.20);
  EXPECT_GT(superbee_order, 1.85);
  EXPECT_LT(superbee_order, 2.20);

  // Fixed-resolution characterization, not a universal ranking: MC tracks the smoother Van Leer
  // reconstruction on this wave, while Superbee remains more accurate than Minmod but less
  // accurate than Van Leer around smooth extrema.
  EXPECT_LT(mc_256.l1, minmod_256.l1);
  EXPECT_LT(superbee_256.l1, minmod_256.l1);
  EXPECT_LT(vanleer_256.l1, superbee_256.l1);
}

TEST(test_muscl_limiter_qualification,
     discontinuities_create_no_extremum_and_expose_interface_dissipation_budget) {
  const auto assert_locally_bounded = [](const std::vector<Real>& averages,
                                         const PeriodicFaceStates& states) {
    ASSERT_TRUE(states.publication_permitted);
    const int size = static_cast<int>(averages.size());
    const Real tolerance = Real(32) * std::numeric_limits<Real>::epsilon();
    for (int i = 0; i < size; ++i) {
      const Real left_min = std::min(averages[periodic_index(i - 1, size)], averages[i]);
      const Real left_max = std::max(averages[periodic_index(i - 1, size)], averages[i]);
      const Real right_min = std::min(averages[i], averages[(i + 1) % size]);
      const Real right_max = std::max(averages[i], averages[(i + 1) % size]);
      EXPECT_GE(states.left[i], left_min - tolerance);
      EXPECT_LE(states.left[i], left_max + tolerance);
      EXPECT_GE(states.right[i], right_min - tolerance);
      EXPECT_LE(states.right[i], right_max + tolerance);
    }
  };

  const std::vector<Real> discontinuity = {Real(0), Real(0), Real(0), Real(0),
                                           Real(1), Real(1), Real(1), Real(1)};
  assert_locally_bounded(discontinuity, reconstruct_periodic_faces<MC>(discontinuity));
  assert_locally_bounded(discontinuity, reconstruct_periodic_faces<Superbee>(discontinuity));

  // Binary fractions keep this steep periodic shoulder deterministic in float and double.  For a
  // scalar Rusanov flux at fixed wave speed, sum |U_R-U_L| is proportional to the absolute
  // dissipative interface penalty.  The ordering is deliberately fixture-specific and records the
  // actual trade-off instead of claiming that one limiter is universally least dissipative.
  const std::vector<Real> shoulder = {
      Real(0),       Real(0),       Real(1) / 16, Real(3) / 16, Real(6) / 16,  Real(10) / 16,
      Real(13) / 16, Real(15) / 16, Real(1),      Real(1),      Real(15) / 16, Real(13) / 16,
      Real(10) / 16, Real(6) / 16,  Real(3) / 16, Real(1) / 16,
  };
  const auto minmod = reconstruct_periodic_faces<Minmod>(shoulder);
  const auto vanleer = reconstruct_periodic_faces<VanLeer>(shoulder);
  const auto mc = reconstruct_periodic_faces<MC>(shoulder);
  const auto superbee = reconstruct_periodic_faces<Superbee>(shoulder);
  assert_locally_bounded(shoulder, minmod);
  assert_locally_bounded(shoulder, vanleer);
  assert_locally_bounded(shoulder, mc);
  assert_locally_bounded(shoulder, superbee);

  const double minmod_jump = interface_jump_budget(minmod);
  const double vanleer_jump = interface_jump_budget(vanleer);
  const double mc_jump = interface_jump_budget(mc);
  const double superbee_jump = interface_jump_budget(superbee);
  EXPECT_LT(mc_jump, vanleer_jump);
  EXPECT_LT(vanleer_jump, superbee_jump);
  EXPECT_LT(superbee_jump, minmod_jump);
}

TEST(test_weno_convergence, external_sampled_policy_controls_offsets_and_orientation) {
  const Box2D valid = Box2D::from_extents(11, 1);
  Fab2D values(valid, PrimitiveTestModel::n_vars, ExternalFourSamplePolicy::n_ghost);
  for (int i = values.grown_box().lo[0]; i <= values.grown_box().hi[0]; ++i) {
    values(i, 0, 0) = Real(2) + Real(0.2) * Real(i);
    values(i, 0, 1) = Real(1) + Real(0.1) * Real(i);
  }

  int primitive_calls = 0;
  const PrimitiveTestModel model{&primitive_calls};
  const ExternalFourSamplePolicy policy{};
  const auto sample_x = [](int offset) { return Real(2) + Real(0.2) * Real(offset); };
  const auto sample_y = [](int offset) { return Real(1) + Real(0.1) * Real(offset); };
  const auto combine = [](auto&& sample) {
    return (sample(-3) + Real(2) * sample(-1) + Real(3) * sample(0) + Real(4) * sample(2)) /
           Real(10);
  };

  const auto right = reconstruct(model, values.const_array(), 5, 0, 0, Real(1), policy, false);
  EXPECT_DOUBLE_EQ(right[0], combine([&](int offset) { return sample_x(5 + offset); }));
  EXPECT_DOUBLE_EQ(right[1], combine([&](int offset) { return sample_y(5 + offset); }));

  const auto left = reconstruct(model, values.const_array(), 5, 0, 0, Real(-1), policy, false);
  EXPECT_DOUBLE_EQ(left[0], combine([&](int offset) { return sample_x(5 - offset); }));
  EXPECT_DOUBLE_EQ(left[1], combine([&](int offset) { return sample_y(5 - offset); }));

  const auto primitive = reconstruct(model, values.const_array(), 5, 0, 0, Real(1), policy, true);
  EXPECT_EQ(primitive_calls, ExternalFourSamplePolicy::stencil_max_offset -
                                 ExternalFourSamplePolicy::stencil_min_offset + 1)
      << "primitive states are converted once per declared offset, not once per component";
  const Real primitive_component = combine([&](int offset) {
    const Real conservative = sample_x(5 + offset);
    return conservative * conservative;
  });
  EXPECT_NEAR(primitive[0], std::sqrt(primitive_component), Real(1e-14));
  EXPECT_DOUBLE_EQ(primitive[1], combine([&](int offset) { return sample_y(5 + offset); }));
  EXPECT_NE(primitive[0], right[0]);
}

TEST(test_weno_convergence, sampled_policy_ghost_requirement_is_checked_exactly) {
  const Box2D domain = Box2D::from_extents(8, 8);
  const BoxArray boxes = BoxArray::from_domain(domain, 8);
  const DistributionMapping distribution(boxes.size(), n_ranks());
  MultiFab insufficient(boxes, distribution, 1, ExternalFourSamplePolicy::n_ghost - 1);
  EXPECT_THROW(detail::require_reconstruction_ghosts<ExternalFourSamplePolicy>(insufficient),
               std::runtime_error);
  MultiFab exact(boxes, distribution, 1, ExternalFourSamplePolicy::n_ghost);
  EXPECT_NO_THROW(detail::require_reconstruction_ghosts<ExternalFourSamplePolicy>(exact));
}

// Pipeline stateful : la pente d'ordre est mesuree PROGRESSIVEMENT (log2 du ratio d'erreurs
// successives), donc les resolutions N successives restent dans le meme test.
TEST(test_weno_convergence, fifth_order_on_smooth_function) {
  double prev = 0, last_order = 0;
  for (int N : {32, 64, 128, 256, 512}) {
    const double dx = 1.0 / N;
    double emax = 0;
    for (int i = 3; i < N - 3; ++i) {
      const double xc = (i + 0.5) * dx;
      const double rec =
          weno5z(favg(xc - 2.5 * dx, xc - 1.5 * dx), favg(xc - 1.5 * dx, xc - 0.5 * dx),
                 favg(xc - 0.5 * dx, xc + 0.5 * dx), favg(xc + 0.5 * dx, xc + 1.5 * dx),
                 favg(xc + 1.5 * dx, xc + 2.5 * dx));
      const double exact = std::sin(2 * kPi * (xc + 0.5 * dx));  // valeur a la face x = xc + dx/2
      emax = std::fmax(emax, std::fabs(rec - exact));
    }
    last_order = prev > 0 ? std::log(prev / emax) / std::log(2.0) : 0;
    std::printf("N=%4d  err_inf=%.3e  ordre=%.2f\n", N, emax, last_order);
    prev = emax;
  }
  EXPECT_GE(last_order, 4.5) << "ordre WENO5 mesure < 4.5";
}

// ADC-645: the WENO-Z regulariser eps is a real weno5z parameter now.
//   - the DEFAULT argument is the historical kWenoEpsilon literal, so an argument-less call is
//     bit-identical to the explicit-default call (the byte-identity golden of the knob);
//   - a materially different eps changes the nonlinear weights on a NON-smooth stencil (live proof
//     the parameter reaches the weights, not just the signature);
//   - a default-constructed Weno5 carries eps == kWenoEpsilon (the operator threading contract).
TEST(test_weno_convergence, epsilon_default_bit_identical_and_override_live) {
  // Discontinuous stencil (top-hat edge): the smoothness indicators differ per sub-stencil, so the
  // eps in a_k = d_k (1 + (tau5/(eps+b_k))^2) matters.
  const Real vm2 = Real(1), vm1 = Real(1), v0 = Real(1), vp1 = Real(0), vp2 = Real(0);
  const Real dflt = weno5z(vm2, vm1, v0, vp1, vp2);
  const Real expl = weno5z(vm2, vm1, v0, vp1, vp2, kWenoEpsilon);
  EXPECT_EQ(dflt, expl) << "default eps argument must be bit-identical to the explicit default";
  const Real fat = weno5z(vm2, vm1, v0, vp1, vp2, Real(1e-2));
  EXPECT_NE(dflt, fat) << "a materially different eps must change the WENO-Z weights";
  const Weno5 lim{};
  EXPECT_EQ(lim.eps, kWenoEpsilon) << "default-constructed Weno5 carries the historical eps";
  struct Sample {
    Real values[5];
    POPS_HD Real operator()(int offset) const { return values[offset + 2]; }
  };
  const Sample sample{{vm2, vm1, v0, vp1, vp2}};
  EXPECT_EQ(lim.stencil_face_value(sample), dflt)
      << "the sampled Weno5 protocol remains bit-identical to the direct kernel";
}
