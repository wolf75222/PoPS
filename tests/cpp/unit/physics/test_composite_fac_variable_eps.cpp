#include <gtest/gtest.h>

#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>

namespace {

template <int Dim>
void expect_variable_diagonal_capability_remains_fail_closed() {
  static_assert(Dim >= 1 && Dim <= 3);
  constexpr auto capabilities = pops::elliptic::mg::CompositeFacPoisson<Dim>::capabilities();
  EXPECT_TRUE(capabilities.scalar_constant_coefficient);
  EXPECT_FALSE(capabilities.variable_diagonal);
}

}  // namespace

TEST(test_composite_fac_variable_eps,
     variable_diagonal_operator_is_not_claimed_by_the_ranked_scalar_fac) {
  expect_variable_diagonal_capability_remains_fail_closed<1>();
  expect_variable_diagonal_capability_remains_fail_closed<2>();
  expect_variable_diagonal_capability_remains_fail_closed<3>();
}
