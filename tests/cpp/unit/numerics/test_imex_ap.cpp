/// @file
/// @brief Asymptotic-preserving proof for the rank-agnostic IMEX composition.

#include <gtest/gtest.h>

#include <pops/numerics/time/schemes/imex.hpp>

#include <cmath>

namespace {

struct RelaxationState {
  pops::Real value = 0;
};

pops::Real run_imex(pops::Real epsilon, pops::Real initial, pops::Real equilibrium, pops::Real dt,
                    int steps) {
  RelaxationState state{initial};
  const auto transport = [](RelaxationState&, pops::Real) {};
  const auto implicit_source = [epsilon, equilibrium](RelaxationState& candidate, pops::Real step) {
    const pops::Real stiffness = step / epsilon;
    candidate.value = (candidate.value + stiffness * equilibrium) / (pops::Real(1) + stiffness);
  };
  for (int step = 0; step < steps; ++step)
    pops::imex_euler_step(state, dt, transport, implicit_source);
  return state.value;
}

pops::Real run_explicit(pops::Real epsilon, pops::Real initial, pops::Real equilibrium,
                        pops::Real dt, int steps) {
  pops::Real value = initial;
  for (int step = 0; step < steps; ++step)
    value += dt * (equilibrium - value) / epsilon;
  return value;
}

TEST(ImexEuler, StiffLimitIsAsymptoticPreservingAtFixedStep) {
  constexpr pops::Real initial = pops::Real(2);
  constexpr pops::Real equilibrium = pops::Real(1);
  constexpr pops::Real epsilon = pops::Real(1.0e-6);
  constexpr pops::Real dt = pops::Real(0.1);
  const pops::Real implicit = run_imex(epsilon, initial, equilibrium, dt, 20);
  const pops::Real explicit_value = run_explicit(epsilon, initial, equilibrium, dt, 5);
  EXPECT_TRUE(std::isfinite(implicit));
  EXPECT_NEAR(implicit, equilibrium, pops::Real(1.0e-3));
  EXPECT_GT(std::fabs(explicit_value), pops::Real(1.0e3));
}

TEST(ImexEuler, NonstiffLimitConvergesAtFirstOrder) {
  constexpr pops::Real initial = pops::Real(2);
  constexpr pops::Real equilibrium = pops::Real(1);
  constexpr pops::Real epsilon = pops::Real(1);
  constexpr pops::Real final_time = pops::Real(1);
  const pops::Real exact = equilibrium + (initial - equilibrium) * std::exp(-final_time / epsilon);
  const pops::Real coarse =
      std::fabs(run_imex(epsilon, initial, equilibrium, final_time / 20, 20) - exact);
  const pops::Real fine =
      std::fabs(run_imex(epsilon, initial, equilibrium, final_time / 40, 40) - exact);
  const pops::Real order = std::log2(coarse / fine);
  EXPECT_GT(order, pops::Real(0.8));
  EXPECT_LT(order, pops::Real(1.3));
}

}  // namespace
