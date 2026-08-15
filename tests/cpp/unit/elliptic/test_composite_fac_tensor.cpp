#include <gtest/gtest.h>

#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>

namespace {

template <int Dim>
void expect_tensor_capability_remains_fail_closed() {
  static_assert(Dim >= 1 && Dim <= 3);
  constexpr auto capabilities = pops::elliptic::mg::CompositeFacPoisson<Dim>::capabilities();
  EXPECT_TRUE(capabilities.scalar_constant_coefficient);
  EXPECT_FALSE(capabilities.cross_tensor);
}

}  // namespace

TEST(test_composite_fac_tensor, tensor_operator_is_not_claimed_by_the_ranked_scalar_fac) {
  expect_tensor_capability_remains_fail_closed<1>();
  expect_tensor_capability_remains_fail_closed<2>();
  expect_tensor_capability_remains_fail_closed<3>();
}
