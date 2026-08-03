#include <gtest/gtest.h>

#include <pops/numerics/spatial/operators/prepared_cartesian_nd.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

using namespace pops;

namespace {

template <int Dimension>
struct LinearTransport {
  using State = StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;

  std::array<Real, Dimension> velocity{};

  template <class Providers>
  POPS_HD State flux(const State& state, const Providers&, int axis) const {
    return State{velocity[axis] * state[0]};
  }

  template <class Providers>
  POPS_HD Real max_wave_speed(const State&, const Providers&, int axis) const {
    return velocity[axis] < Real(0) ? -velocity[axis] : velocity[axis];
  }
};

template <std::size_t Dimension>
std::size_t linear_index(const std::array<int, Dimension>& index,
                         const std::array<int, Dimension>& extents) {
  std::size_t result = 0;
  std::size_t stride = 1;
  for (std::size_t axis = 0; axis < Dimension; ++axis) {
    result += static_cast<std::size_t>(index[axis]) * stride;
    stride *= static_cast<std::size_t>(extents[axis]);
  }
  return result;
}

template <std::size_t Dimension, class Function>
void for_each_index(const std::array<int, Dimension>& extents, Function&& function) {
  std::size_t cells = 1;
  for (const int extent : extents)
    cells *= static_cast<std::size_t>(extent);
  for (std::size_t linear = 0; linear < cells; ++linear) {
    std::size_t remaining = linear;
    std::array<int, Dimension> index{};
    for (std::size_t axis = 0; axis < Dimension; ++axis) {
      index[axis] = static_cast<int>(remaining % static_cast<std::size_t>(extents[axis]));
      remaining /= static_cast<std::size_t>(extents[axis]);
    }
    function(index);
  }
}

}  // namespace

TEST(test_prepared_cartesian_nd, one_dimensional_kernel_preserves_constant_state_and_conservation) {
  constexpr int dimension = 1;
  const std::array<int, dimension> extents{32};
  const std::array<Real, dimension> lower{Real(-1)};
  const std::array<Real, dimension> upper{Real(2)};
  const PreparedPeriodicCartesianResidual<dimension, LinearTransport<dimension>, VanLeer,
                                          RusanovFlux>
      residual(extents, lower, upper, LinearTransport<dimension>{{Real(0.7)}});

  EXPECT_TRUE(residual.capabilities().supports(
      {1, SpatialProviderGeometry::Cartesian, SpatialProviderOperation::Residual}));
  EXPECT_FALSE(residual.capabilities().supports(
      {2, SpatialProviderGeometry::Cartesian, SpatialProviderOperation::Residual}));

  std::vector<Real> constant(residual.scalar_count(), Real(2.5));
  std::vector<Real> output(residual.scalar_count(), Real(99));
  residual.execute(constant, output);
  EXPECT_TRUE(
      std::all_of(output.begin(), output.end(), [](Real value) { return value == Real(0); }));

  constexpr Real two_pi = Real(6.283185307179586476925286766559);
  std::vector<Real> wave(residual.scalar_count());
  for (int i = 0; i < extents[0]; ++i)
    wave[static_cast<std::size_t>(i)] =
        Real(1) + Real(0.2) * std::sin(two_pi * (Real(i) + Real(0.5)) / Real(extents[0]));
  residual.execute(wave, output);
  EXPECT_TRUE(std::any_of(output.begin(), output.end(),
                          [](Real value) { return std::abs(value) > Real(1e-8); }));
  const Real integral =
      std::accumulate(output.begin(), output.end(), Real(0)) * residual.metric().cell_measure;
  EXPECT_NEAR(integral, Real(0), Real(2e-14));
}

TEST(test_prepared_cartesian_nd,
     three_dimensional_kernel_is_axis_permutation_invariant_and_conservative) {
  constexpr int dimension = 3;
  constexpr std::array<int, dimension> permutation{2, 0, 1};
  const std::array<int, dimension> extents{8, 7, 6};
  const std::array<Real, dimension> lower{Real(-1), Real(2), Real(0.5)};
  const std::array<Real, dimension> upper{Real(3), Real(5), Real(2.5)};
  const std::array<Real, dimension> velocity{Real(0.7), Real(-0.4), Real(0.25)};
  const PreparedPeriodicCartesianResidual<dimension, LinearTransport<dimension>, VanLeer,
                                          RusanovFlux>
      original(extents, lower, upper, LinearTransport<dimension>{velocity});

  std::array<int, dimension> permuted_extents{};
  std::array<Real, dimension> permuted_lower{};
  std::array<Real, dimension> permuted_upper{};
  std::array<Real, dimension> permuted_velocity{};
  for (int axis = 0; axis < dimension; ++axis) {
    permuted_extents[axis] = extents[permutation[axis]];
    permuted_lower[axis] = lower[permutation[axis]];
    permuted_upper[axis] = upper[permutation[axis]];
    permuted_velocity[axis] = velocity[permutation[axis]];
  }
  const PreparedPeriodicCartesianResidual<dimension, LinearTransport<dimension>, VanLeer,
                                          RusanovFlux>
      permuted(permuted_extents, permuted_lower, permuted_upper,
               LinearTransport<dimension>{permuted_velocity});

  std::vector<Real> state(original.scalar_count());
  std::vector<Real> permuted_state(permuted.scalar_count());
  constexpr Real two_pi = Real(6.283185307179586476925286766559);
  for_each_index<dimension>(extents, [&](const auto& index) {
    Real value = Real(0.75);
    for (int axis = 0; axis < dimension; ++axis)
      value += (Real(0.1) + Real(0.05) * Real(axis)) *
               std::sin(two_pi * (Real(index[axis]) + Real(0.5)) / Real(extents[axis]));
    state[linear_index(index, extents)] = value;
    std::array<int, dimension> mapped{};
    for (int axis = 0; axis < dimension; ++axis)
      mapped[axis] = index[permutation[axis]];
    permuted_state[linear_index(mapped, permuted_extents)] = value;
  });

  std::vector<Real> output(original.scalar_count());
  std::vector<Real> permuted_output(permuted.scalar_count());
  original.execute(state, output);
  permuted.execute(permuted_state, permuted_output);
  for_each_index<dimension>(extents, [&](const auto& index) {
    std::array<int, dimension> mapped{};
    for (int axis = 0; axis < dimension; ++axis)
      mapped[axis] = index[permutation[axis]];
    EXPECT_NEAR(output[linear_index(index, extents)],
                permuted_output[linear_index(mapped, permuted_extents)], Real(3e-13));
  });

  const Real integral =
      std::accumulate(output.begin(), output.end(), Real(0)) * original.metric().cell_measure;
  EXPECT_NEAR(integral, Real(0), Real(2e-13));

  std::fill(state.begin(), state.end(), Real(1.25));
  original.execute(state, output);
  EXPECT_TRUE(
      std::all_of(output.begin(), output.end(), [](Real value) { return value == Real(0); }));
}

TEST(test_prepared_cartesian_nd, preparation_refuses_invalid_metric_and_buffer_contracts) {
  using Residual = PreparedPeriodicCartesianResidual<3, LinearTransport<3>, VanLeer, RusanovFlux>;
  EXPECT_THROW((Residual({2, 4, 4}, {Real(0), Real(0), Real(0)}, {Real(1), Real(1), Real(1)},
                         LinearTransport<3>{{Real(1), Real(1), Real(1)}})),
               std::invalid_argument);
  EXPECT_THROW((Residual({4, 4, 4}, {Real(0), Real(0), Real(0)}, {Real(1), Real(0), Real(1)},
                         LinearTransport<3>{{Real(1), Real(1), Real(1)}})),
               std::invalid_argument);

  Residual residual({4, 4, 4}, {Real(0), Real(0), Real(0)}, {Real(1), Real(1), Real(1)},
                    LinearTransport<3>{{Real(1), Real(1), Real(1)}});
  std::vector<Real> state(residual.scalar_count(), Real(1));
  EXPECT_THROW(residual.execute(state, std::span<Real>(state.data(), state.size())),
               std::invalid_argument);
  std::vector<Real> short_output(residual.scalar_count() - 1);
  EXPECT_THROW(residual.execute(state, short_output), std::invalid_argument);
}
