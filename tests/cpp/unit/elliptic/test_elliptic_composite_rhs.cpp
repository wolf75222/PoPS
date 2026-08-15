#include <gtest/gtest.h>

#include <pops/physics/bricks/elliptic.hpp>

#include <array>

namespace {

template <int Dim>
void expect_composed_elliptic_rhs() {
  static_assert(Dim >= 1 && Dim <= 3);
  std::array<pops::Real, Dim> state{};
  state[0] = pops::Real(1.25);
  const pops::BackgroundDensity background{pops::Real(1.3), pops::Real(0.4)};
  const pops::ChargeDensity charge{pops::Real(-0.8)};
  const pops::GravityCoupling gravity{pops::Real(-1), pops::Real(2.5), pops::Real(0.6)};

  EXPECT_EQ(background.rhs(state), pops::Real(1.3) * (state[0] - pops::Real(0.4)));

  pops::Real composed = pops::Real(0);
  composed += charge.rhs(state);
  composed += gravity.rhs(state);
  EXPECT_EQ(composed, pops::Real(-0.8) * state[0] - pops::Real(2.5) * (state[0] - pops::Real(0.6)));
}

}  // namespace

TEST(test_elliptic_composite_rhs, brick_composition_is_rank_agnostic) {
  expect_composed_elliptic_rhs<1>();
  expect_composed_elliptic_rhs<2>();
  expect_composed_elliptic_rhs<3>();
}
