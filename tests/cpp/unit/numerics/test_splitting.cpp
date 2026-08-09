/// @file
/// @brief Temporal-order proof for rank-agnostic Lie and Strang compositions.

#include <gtest/gtest.h>

#include <pops/numerics/time/schemes/splitting.hpp>

#include <algorithm>
#include <cmath>

namespace {

struct CoupledState {
  pops::Real x = 0;
  pops::Real y = 0;
};

void transport(CoupledState& state, pops::Real step) {
  state.x += step * state.y;
}

void source(CoupledState& state, pops::Real step) {
  state.y += step * state.x;
}

pops::Real error(bool use_strang, int steps, pops::Real final_time, CoupledState initial) {
  CoupledState state = initial;
  const pops::Real dt = final_time / static_cast<pops::Real>(steps);
  for (int step = 0; step < steps; ++step) {
    if (use_strang)
      pops::strang_step(state, dt, transport, source);
    else
      pops::lie_step(state, dt, transport, source);
  }
  const pops::Real exact_x = initial.x * std::cosh(final_time) + initial.y * std::sinh(final_time);
  const pops::Real exact_y = initial.x * std::sinh(final_time) + initial.y * std::cosh(final_time);
  return std::max(std::fabs(state.x - exact_x), std::fabs(state.y - exact_y));
}

TEST(OperatorSplitting, StrangIsSecondOrderAndLieIsFirstOrder) {
  constexpr pops::Real final_time = pops::Real(0.8);
  constexpr CoupledState initial{pops::Real(1), pops::Real(0)};
  const pops::Real strang_coarse = error(true, 20, final_time, initial);
  const pops::Real strang_fine = error(true, 40, final_time, initial);
  const pops::Real lie_coarse = error(false, 20, final_time, initial);
  const pops::Real lie_fine = error(false, 40, final_time, initial);
  const pops::Real strang_order = std::log2(strang_coarse / strang_fine);
  const pops::Real lie_order = std::log2(lie_coarse / lie_fine);
  EXPECT_GT(strang_order, pops::Real(1.8));
  EXPECT_LT(strang_order, pops::Real(2.2));
  EXPECT_GT(lie_order, pops::Real(0.8));
  EXPECT_LT(lie_order, pops::Real(1.3));
  EXPECT_LT(strang_coarse, lie_coarse);
}

}  // namespace
