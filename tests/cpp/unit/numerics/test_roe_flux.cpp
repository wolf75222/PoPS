// Exact-ranked ideal-gas Roe provider.  The same axis-indexed algorithm must be consistent,
// upwind in supersonic flow, and expose the complete spectrum in 1D, 2D, and 3D.
#include <gtest/gtest.h>

#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/physics/fluids/euler.hpp>

#include <array>
#include <cmath>

namespace {

template <int Dim>
using Model = pops::EulerND<Dim>;

template <int Dim>
using State = typename Model<Dim>::State;

static_assert(pops::nd::ConservationLaw<1, Model<1>>);
static_assert(pops::nd::ConservationLaw<2, Model<2>>);
static_assert(pops::nd::ConservationLaw<3, Model<3>>);

template <int Dim>
State<Dim> conservative(double rho, const std::array<double, Dim>& velocity, double pressure,
                        double gamma) {
  typename Model<Dim>::Prim primitive{};
  primitive[0] = rho;
  for (int axis = 0; axis < Dim; ++axis)
    primitive[Model<Dim>::momentum_component(axis)] = velocity[axis];
  primitive[Model<Dim>::energy_component] = pressure;
  return Model<Dim>{gamma}.to_conservative(primitive);
}

template <int Dim>
double maxdiff(const State<Dim>& left, const State<Dim>& right) {
  double result = 0;
  for (int component = 0; component < Model<Dim>::n_vars; ++component)
    result = std::fmax(result, std::fabs(left[component] - right[component]));
  return result;
}

template <int Dim, class Policy>
State<Dim> face_density(const Policy& policy, const Model<Dim>& model, const State<Dim>& left,
                        const pops::AuxState<Dim>& left_providers, const State<Dim>& right,
                        const pops::AuxState<Dim>& right_providers, int axis) {
  pops::FluxProviderValues<Model<Dim>> left_values{}, right_values{};
  left_values[pops::AuxComponentLayout<Dim>::phi] = left_providers.phi;
  right_values[pops::AuxComponentLayout<Dim>::phi] = right_providers.phi;
  return pops::evaluate_numerical_flux(policy, model, left,
                                       pops::bind_flux_providers<Model<Dim>>(left_values), right,
                                       pops::bind_flux_providers<Model<Dim>>(right_values),
                                       pops::FaceContext::axis_aligned(axis))
      .checked_density()
      .value;
}

template <int Dim>
std::array<double, Dim> velocities(double normal, int normal_axis, double tangent_scale) {
  std::array<double, Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = axis == normal_axis ? normal : tangent_scale * double(axis + 1);
  return result;
}

}  // namespace

TEST(test_roe_flux, consistent_at_constant_state_in_every_exact_rank) {
  const auto check = []<int Dim>() {
    const Model<Dim> model{1.4};
    const pops::AuxState<Dim> providers{};
    const pops::RoeFlux roe;
    for (int axis = 0; axis < Dim; ++axis) {
      const State<Dim> values[] = {
          conservative<Dim>(1.2, velocities<Dim>(0.3, axis, -0.1), 1.5, 1.4),
          conservative<Dim>(0.7, velocities<Dim>(-0.2, axis, 0.2), 0.9, 1.4),
      };
      for (const State<Dim>& value : values) {
        const double error =
            maxdiff<Dim>(face_density(roe, model, value, providers, value, providers, axis),
                         model.flux(value, providers, axis));
        EXPECT_LE(error, 1e-12) << "rank=" << Dim << " axis=" << axis;
      }
    }
  };
  check.template operator()<1>();
  check.template operator()<2>();
  check.template operator()<3>();
}

TEST(test_roe_flux, supersonic_upwind_property_in_every_exact_rank_and_axis) {
  const auto check = []<int Dim>() {
    const Model<Dim> model{1.4};
    const pops::AuxState<Dim> providers{};
    const pops::RoeFlux roe;
    for (int axis = 0; axis < Dim; ++axis) {
      const State<Dim> positive_left =
          conservative<Dim>(1.0, velocities<Dim>(8.0, axis, 0.15), 1.0, 1.4);
      const State<Dim> positive_right =
          conservative<Dim>(1.5, velocities<Dim>(12.0, axis, -0.1), 1.3, 1.4);
      EXPECT_LE(maxdiff<Dim>(face_density(roe, model, positive_left, providers, positive_right,
                                          providers, axis),
                             model.flux(positive_left, providers, axis)),
                1e-9)
          << "positive flow rank=" << Dim << " axis=" << axis;

      const State<Dim> negative_left =
          conservative<Dim>(1.2, velocities<Dim>(-12.0, axis, 0.1), 1.1, 1.4);
      const State<Dim> negative_right =
          conservative<Dim>(0.9, velocities<Dim>(-9.0, axis, -0.2), 0.8, 1.4);
      EXPECT_LE(maxdiff<Dim>(face_density(roe, model, negative_left, providers, negative_right,
                                          providers, axis),
                             model.flux(negative_right, providers, axis)),
                1e-9)
          << "negative flow rank=" << Dim << " axis=" << axis;
    }
  };
  check.template operator()<1>();
  check.template operator()<2>();
  check.template operator()<3>();
}

TEST(test_roe_flux, spectrum_contains_acoustic_material_and_shear_waves_in_every_rank) {
  const auto check = []<int Dim>() {
    const Model<Dim> model{1.4};
    const pops::AuxState<Dim> providers{};
    for (int axis = 0; axis < Dim; ++axis) {
      const auto velocity = velocities<Dim>(0.5, axis, -0.1);
      const State<Dim> value = conservative<Dim>(1.0, velocity, 1.0, 1.4);
      pops::Real lower{}, upper{};
      model.wave_speeds(value, providers, axis, lower, upper);
      const auto primitive = model.recover(value);
      ASSERT_TRUE(primitive.succeeded());
      const double sound_speed = std::sqrt(1.4);
      EXPECT_NEAR(lower, velocity[axis] - sound_speed, 1e-12);
      EXPECT_NEAR(upper, velocity[axis] + sound_speed, 1e-12);
      EXPECT_NEAR(primitive.value[Model<Dim>::momentum_component(axis)], velocity[axis], 1e-12);
    }
  };
  check.template operator()<1>();
  check.template operator()<2>();
  check.template operator()<3>();
}
