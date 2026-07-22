// WENO5-Z : la reconstruction de la valeur de face d'une fonction LISSE depuis ses moyennes
// de cellule est d'ordre 5 (les poids non lineaires WENO-Z tendent vers les poids lineaires
// optimaux 1/10, 6/10, 3/10 en zone reguliere). On verifie l'ordre mesure >= 4.5 et la
// preservation des constantes. Brique de la voie haute precision vers le taux analytique 0.911.

#include <gtest/gtest.h>

#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/numerics/spatial/primitives/face_flux.hpp>

#include <cmath>
#include <cstdio>

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
    return (sample(-3) + Real(2) * sample(-1) + Real(3) * sample(0) +
            Real(4) * sample(2)) /
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

/// A nonlinear conservative/primitive conversion makes the two reconstruction paths observably
/// different while remaining exactly invertible for the positive test data.
struct PrimitiveTestModel {
  using State = StateVec<2>;
  using Prim = StateVec<2>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 2;
  int* primitive_calls = nullptr;

  POPS_HD State flux(const State& state, const Aux&, int) const { return state; }
  POPS_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(1); }
  POPS_HD State source(const State&, const Aux&) const { return State{}; }
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
    return (sample(-3) + Real(2) * sample(-1) + Real(3) * sample(0) +
            Real(4) * sample(2)) /
           Real(10);
  };

  const auto right = reconstruct(model, values.const_array(), 5, 0, 0, Real(1), policy, false);
  EXPECT_DOUBLE_EQ(right[0], combine([&](int offset) { return sample_x(5 + offset); }));
  EXPECT_DOUBLE_EQ(right[1], combine([&](int offset) { return sample_y(5 + offset); }));

  const auto left = reconstruct(model, values.const_array(), 5, 0, 0, Real(-1), policy, false);
  EXPECT_DOUBLE_EQ(left[0], combine([&](int offset) { return sample_x(5 - offset); }));
  EXPECT_DOUBLE_EQ(left[1], combine([&](int offset) { return sample_y(5 - offset); }));

  const auto primitive =
      reconstruct(model, values.const_array(), 5, 0, 0, Real(1), policy, true);
  EXPECT_EQ(primitive_calls,
            ExternalFourSamplePolicy::stencil_max_offset -
                ExternalFourSamplePolicy::stencil_min_offset + 1)
      << "primitive states are converted once per declared offset, not once per component";
  const Real primitive_component =
      combine([&](int offset) {
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
