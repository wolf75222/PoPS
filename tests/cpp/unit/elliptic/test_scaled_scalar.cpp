// Focused, standalone coverage for the binary-scaled scalar helper.

#include <gtest/gtest.h>

#include <pops/numerics/elliptic/linear/scaled_scalar.hpp>

#include <cmath>
#include <concepts>
#include <limits>

using pops::Real;
using pops::detail::ScaledScalar;

namespace {

template <class T>
concept SupportsAddition = requires(T left, T right) { left + right; };

static_assert(!SupportsAddition<ScaledScalar>);

TEST(ScaledScalar, AppliesOverflowingCoefficientToFiniteNormalizedResidual) {
  const Real equation_scale = Real(3.2e199);
  const Real omega = Real(1e109);
  const volatile Real raw_coefficient = equation_scale * omega;
  EXPECT_FALSE(std::isfinite(raw_coefficient));

  const ScaledScalar coefficient =
      ScaledScalar::product(ScaledScalar::from(equation_scale), ScaledScalar::from(omega));
  ASSERT_TRUE(coefficient.is_finite());

  Real applied = Real(0);
  ASSERT_TRUE(coefficient.try_apply(Real(0.03125), applied));
  EXPECT_TRUE(std::isfinite(applied));
  EXPECT_NEAR(applied / Real(1e307), Real(1), Real(1e-12));
}

TEST(ScaledScalar, RetainsOverflowingQuotientWithoutInfinity) {
  const Real denominator = Real(1e-309);
  const volatile Real raw_quotient = Real(1) / denominator;
  EXPECT_FALSE(std::isfinite(raw_quotient));

  const ScaledScalar quotient =
      ScaledScalar::quotient(ScaledScalar::from(Real(1)), ScaledScalar::from(denominator));
  ASSERT_TRUE(quotient.is_finite());
  EXPECT_GT(quotient.exponent(), std::numeric_limits<Real>::max_exponent);

  Real materialized = Real(0);
  EXPECT_FALSE(quotient.try_materialize(materialized));
  EXPECT_TRUE(std::isnan(materialized));
}

TEST(ScaledScalar, ComposesBicgstabAndGmresMultiplicativeRecurrences) {
  const ScaledScalar rho = ScaledScalar::from(Real(3e199));
  const ScaledScalar rho_previous = ScaledScalar::from(Real(2.5e-100));
  const ScaledScalar alpha = ScaledScalar::from(Real(2.5e-101));
  const ScaledScalar omega = ScaledScalar::from(Real(5e199));

  // BiCGStab: beta = (rho / rho_previous) * (alpha / omega).
  const ScaledScalar beta = ScaledScalar::product(ScaledScalar::quotient(rho, rho_previous),
                                                  ScaledScalar::quotient(alpha, omega));
  ASSERT_TRUE(beta.is_finite());
  Real beta_value = Real(0);
  ASSERT_TRUE(beta.try_materialize(beta_value));
  EXPECT_NEAR(beta_value, Real(0.06), Real(1e-15));

  // GMRES: beta * (normalized_threshold / current_residual).
  const ScaledScalar estimate_threshold = ScaledScalar::product(
      beta,
      ScaledScalar::quotient(ScaledScalar::from(Real(1e-10)), ScaledScalar::from(Real(0.25))));
  Real estimate_value = Real(0);
  ASSERT_TRUE(estimate_threshold.try_materialize(estimate_value));
  EXPECT_NEAR(estimate_value, Real(2.4e-11), Real(1e-24));
}

TEST(ScaledScalar, FusesSignedOverflowingProductsBeforeFinalMaterialization) {
  const ScaledScalar enormous = ScaledScalar::from(Real(1e308));
  const ScaledScalar negative_enormous = ScaledScalar::negated(enormous);

  // Neither product fits in Real, but their signed cancellation leaves a finite exact cell value.
  Real result = Real(0);
  ASSERT_TRUE(
      ScaledScalar::try_sum_products(enormous, Real(2), negative_enormous, Real(2), result));
  EXPECT_EQ(result, Real(0));

  ASSERT_TRUE(ScaledScalar::try_sum_products(enormous, Real(2), negative_enormous, Real(2),
                                             ScaledScalar::from(Real(1)), Real(7), result));
  EXPECT_EQ(result, Real(7));

  EXPECT_FALSE(
      ScaledScalar::try_sum_products(enormous, Real(2), ScaledScalar::zero(), Real(0), result));
  EXPECT_TRUE(std::isnan(result));
}

TEST(ScaledScalar, PropagatesExceptionalInputsAndRejectsUnrepresentableApplication) {
  const ScaledScalar infinite = ScaledScalar::from(std::numeric_limits<Real>::infinity());
  const ScaledScalar invalid =
      ScaledScalar::quotient(ScaledScalar::from(Real(1)), ScaledScalar::zero());
  EXPECT_TRUE(infinite.is_nonfinite());
  EXPECT_TRUE(invalid.is_nonfinite());

  Real applied = Real(0);
  EXPECT_FALSE(invalid.try_apply(Real(1), applied));
  EXPECT_TRUE(std::isnan(applied));
}

}  // namespace
