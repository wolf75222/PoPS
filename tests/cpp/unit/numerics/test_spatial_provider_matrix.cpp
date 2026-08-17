#include <gtest/gtest.h>

#include <pops/numerics/spatial/provider_matrix.hpp>

using namespace pops;

TEST(test_spatial_provider_matrix, native_cartesian_provider_qualifies_exact_operations) {
  constexpr auto provider = make_cartesian_spatial_provider(2, /*characteristic_no_inflow=*/true,
                                                            /*boundary_linearization=*/true);

  EXPECT_TRUE(provider.supports(
      {2, SpatialProviderGeometry::Cartesian, SpatialProviderOperation::Residual}));
  EXPECT_TRUE(provider.supports(
      {2, SpatialProviderGeometry::Cartesian, SpatialProviderOperation::CharacteristicNoInflow}));
  EXPECT_TRUE(provider.supports(
      {2, SpatialProviderGeometry::Cartesian, SpatialProviderOperation::BoundaryLinearization}));
  EXPECT_FALSE(
      provider.supports({2, SpatialProviderGeometry::CutCell, SpatialProviderOperation::Residual}));
}

TEST(test_spatial_provider_matrix, native_runtime_dimension_refuses_unproved_3d_execution) {
  constexpr auto provider = make_cartesian_spatial_provider(2);
  constexpr auto refusal = qualify_spatial_provider(
      provider, {3, SpatialProviderGeometry::Cartesian, SpatialProviderOperation::Residual});

  static_assert(!refusal.executable);
  static_assert(refusal.refusal == SpatialProviderRefusal::UnsupportedDimension);
  EXPECT_FALSE(refusal.executable);
}

TEST(test_spatial_provider_matrix, independent_axes_do_not_form_false_cross_product_capabilities) {
  SpatialProviderCapabilities provider;
  provider.enable(1, SpatialProviderGeometry::Cartesian, SpatialProviderOperation::Residual);
  provider.enable(3, SpatialProviderGeometry::Polar, SpatialProviderOperation::Residual);

  EXPECT_TRUE(provider.supports(
      {1, SpatialProviderGeometry::Cartesian, SpatialProviderOperation::Residual}));
  EXPECT_TRUE(
      provider.supports({3, SpatialProviderGeometry::Polar, SpatialProviderOperation::Residual}));
  EXPECT_FALSE(provider.supports(
      {3, SpatialProviderGeometry::Cartesian, SpatialProviderOperation::Residual}));
  EXPECT_FALSE(
      provider.supports({1, SpatialProviderGeometry::Polar, SpatialProviderOperation::Residual}));
  EXPECT_EQ(qualify_spatial_provider(provider, {3, SpatialProviderGeometry::Cartesian,
                                                SpatialProviderOperation::Residual})
                .refusal,
            SpatialProviderRefusal::UnsupportedGeometry);
}

TEST(test_spatial_provider_matrix,
     embedded_metric_residuals_claim_characteristic_but_not_linearization) {
  constexpr auto provider = with_embedded_boundary_residuals(
      make_cartesian_spatial_provider(2, /*characteristic_no_inflow=*/true,
                                      /*boundary_linearization=*/true));

  for (const auto geometry :
       {SpatialProviderGeometry::Staircase, SpatialProviderGeometry::CutCell}) {
    EXPECT_TRUE(provider.supports({2, geometry, SpatialProviderOperation::Residual}));
    EXPECT_TRUE(provider.supports({2, geometry, SpatialProviderOperation::CharacteristicNoInflow}));
    const auto characteristic = qualify_spatial_provider(
        provider, {2, geometry, SpatialProviderOperation::CharacteristicNoInflow});
    const auto linearization = qualify_spatial_provider(
        provider, {2, geometry, SpatialProviderOperation::BoundaryLinearization});
    EXPECT_TRUE(characteristic.executable);
    EXPECT_EQ(characteristic.refusal, SpatialProviderRefusal::None);
    EXPECT_EQ(linearization.refusal, SpatialProviderRefusal::UnsupportedOperation);
  }
}

TEST(test_spatial_provider_matrix, polar_metric_provider_is_residual_only) {
  constexpr auto provider = make_polar_spatial_provider(2);

  EXPECT_TRUE(
      provider.supports({2, SpatialProviderGeometry::Polar, SpatialProviderOperation::Residual}));
  EXPECT_EQ(qualify_spatial_provider(provider, {2, SpatialProviderGeometry::Cartesian,
                                                SpatialProviderOperation::Residual})
                .refusal,
            SpatialProviderRefusal::UnsupportedGeometry);
  EXPECT_EQ(qualify_spatial_provider(provider, {2, SpatialProviderGeometry::Polar,
                                                SpatialProviderOperation::CharacteristicNoInflow})
                .refusal,
            SpatialProviderRefusal::UnsupportedOperation);
}
