#include <gtest/gtest.h>

#include <pops/mesh/geometry/prepared_metric_provider.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

using pops::Box;
using pops::CartesianCoordinateMap;
using pops::CoordinateMap;
using pops::CoordinateMapKind;
using pops::Index;
using pops::InverseMapStatus;
using pops::MetricFaceSide;
using pops::PlanarPolarCoordinateMap;
using pops::PreparedMappedMetricProvider;
using pops::PreparedMetricProvider;
using pops::Real;
using pops::RealVector;
using pops::prepare_metric_provider;

namespace {

constexpr Real kPi = Real(3.141592653589793238462643383279502884);

bool close(Real left, Real right, Real tolerance = Real(2e-13)) {
  return std::abs(left - right) <= tolerance * (Real(1) + std::abs(left) + std::abs(right));
}

template <int Dim>
void expect_vector_close(const RealVector<Dim>& actual, const RealVector<Dim>& expected,
                         Real tolerance = Real(2e-13)) {
  for (int axis = 0; axis < Dim; ++axis)
    EXPECT_TRUE(close(actual[axis], expected[axis], tolerance))
        << "axis=" << axis << " actual=" << actual[axis] << " expected=" << expected[axis];
}

template <int Axis, class Provider>
void add_face_pair(const Provider& provider, const Index<Provider::logical_dimension>& index,
                   typename Provider::PhysicalPoint& sum) {
  const auto lower =
      provider.template oriented_face_area_vector<Axis, MetricFaceSide::Lower>(index);
  const auto upper =
      provider.template oriented_face_area_vector<Axis, MetricFaceSide::Upper>(index);
  for (int component = 0; component < Provider::embedding_dimension; ++component)
    sum[component] += lower[component] + upper[component];
}

template <class Provider, int... Axis>
typename Provider::PhysicalPoint face_closure(const Provider& provider,
                                              const Index<Provider::logical_dimension>& index,
                                              std::integer_sequence<int, Axis...>) {
  typename Provider::PhysicalPoint sum{};
  (add_face_pair<Axis>(provider, index, sum), ...);
  return sum;
}

template <class Provider>
typename Provider::PhysicalPoint face_closure(const Provider& provider,
                                              const Index<Provider::logical_dimension>& index) {
  return face_closure(provider, index,
                      std::make_integer_sequence<int, Provider::logical_dimension>{});
}

}  // namespace

static_assert(CoordinateMap<1, 1, CartesianCoordinateMap<1>>);
static_assert(CoordinateMap<2, 3, CartesianCoordinateMap<2, 3>>);
static_assert(CoordinateMap<3, 3, CartesianCoordinateMap<3>>);
static_assert(CoordinateMap<2, 2, PlanarPolarCoordinateMap>);
static_assert(PreparedMetricProvider<1, PreparedMappedMetricProvider<CartesianCoordinateMap<1>>>);
static_assert(PreparedMetricProvider<2, PreparedMappedMetricProvider<PlanarPolarCoordinateMap>>);
static_assert(
    std::is_trivially_copyable_v<PreparedMappedMetricProvider<CartesianCoordinateMap<3>>>);

TEST(test_nd_metric_provider, capabilities_and_identities_are_exact_values) {
  constexpr auto cartesian = CartesianCoordinateMap<3>::capabilities();
  static_assert(cartesian.logical_dimension == 3);
  static_assert(cartesian.embedding_dimension == 3);
  static_assert(cartesian.kind == CoordinateMapKind::Cartesian);
  static_assert(cartesian.affine && cartesian.exact_cell_measure &&
                cartesian.exact_oriented_face_area && cartesian.compile_time_axes &&
                cartesian.device_callable);

  constexpr auto polar = PlanarPolarCoordinateMap::capabilities();
  static_assert(polar.logical_dimension == 2);
  static_assert(polar.embedding_dimension == 2);
  static_assert(polar.kind == CoordinateMapKind::PlanarPolar);
  static_assert(!polar.affine && polar.exact_cell_measure && polar.exact_oriented_face_area);

  const Box<2> domain{Index<2>{-2, 5}, Index<2>{1, 8}};
  const auto map =
      CartesianCoordinateMap<2>::make(RealVector<2>{-1.0, 4.0}, RealVector<2>{2.0, 6.0});
  const auto first = prepare_metric_provider(domain, map);
  const auto same = prepare_metric_provider(domain, map);
  const auto moved = prepare_metric_provider(
      domain, CartesianCoordinateMap<2>::make(RealVector<2>{-1.0, 4.5}, RealVector<2>{2.0, 6.0}));
  const auto resized = prepare_metric_provider(Box<2>{Index<2>{-2, 5}, Index<2>{2, 8}}, map);

  EXPECT_EQ(first.identity(), same.identity());
  EXPECT_NE(first.identity(), moved.identity());
  EXPECT_NE(first.identity(), resized.identity());
  EXPECT_TRUE(decltype(first)::capabilities().exact_domain_identity);
  EXPECT_TRUE(decltype(first)::capabilities().ghost_coordinates);
  EXPECT_TRUE(decltype(first)::capabilities().allocation_free_queries);
}

TEST(test_nd_metric_provider, cartesian_1d_centers_faces_measure_and_ghosts) {
  const auto map = CartesianCoordinateMap<1>::make(RealVector<1>{5.0}, RealVector<1>{8.0},
                                                   std::array<int, 1>{0}, std::array<int, 1>{-1});
  const auto metric = prepare_metric_provider(Box<1>{Index<1>{-2}, Index<1>{1}}, map);

  expect_vector_close(metric.cell_center(Index<1>{-2}), RealVector<1>{4.0});
  expect_vector_close(metric.template face_center<0, MetricFaceSide::Lower>(Index<1>{-2}),
                      RealVector<1>{5.0});
  expect_vector_close(metric.template face_center<0, MetricFaceSide::Upper>(Index<1>{-2}),
                      RealVector<1>{3.0});
  EXPECT_TRUE(close(metric.cell_measure(Index<1>{-2}), Real(2)));
  expect_vector_close(
      metric.template oriented_face_area_vector<0, MetricFaceSide::Lower>(Index<1>{-2}),
      RealVector<1>{1.0});
  expect_vector_close(
      metric.template oriented_face_area_vector<0, MetricFaceSide::Upper>(Index<1>{-2}),
      RealVector<1>{-1.0});

  // Coordinate queries are deliberately defined outside the accepted domain for ghost kernels.
  expect_vector_close(metric.cell_center(Index<1>{-3}), RealVector<1>{6.0});
}

TEST(test_nd_metric_provider, cartesian_axis_permutation_is_reflected_in_every_metric) {
  const auto map =
      CartesianCoordinateMap<3>::make(RealVector<3>{10.0, 20.0, 30.0}, RealVector<3>{2.0, 4.0, 6.0},
                                      std::array<int, 3>{2, 0, 1}, std::array<int, 3>{1, -1, 1});
  const auto metric = prepare_metric_provider(Box<3>{Index<3>{0, 0, 0}, Index<3>{1, 3, 2}}, map);
  const Index<3> index{0, 0, 0};

  expect_vector_close(metric.cell_center(index), RealVector<3>{9.5, 21.0, 30.5});
  expect_vector_close(metric.template face_center<1, MetricFaceSide::Lower>(index),
                      RealVector<3>{10.0, 21.0, 30.5});
  EXPECT_TRUE(close(metric.cell_measure(index), Real(2)));

  const auto jacobian = metric.jacobian(index);
  EXPECT_TRUE(close(jacobian[2][0], Real(2)));
  EXPECT_TRUE(close(jacobian[0][1], Real(-4)));
  EXPECT_TRUE(close(jacobian[1][2], Real(6)));
  EXPECT_TRUE(close(jacobian[0][0], Real(0)));
  EXPECT_TRUE(close(jacobian[1][0], Real(0)));

  expect_vector_close(metric.template oriented_face_area_vector<0, MetricFaceSide::Upper>(index),
                      RealVector<3>{0.0, 0.0, 2.0});
  expect_vector_close(metric.template oriented_face_area_vector<1, MetricFaceSide::Upper>(index),
                      RealVector<3>{-2.0, 0.0, 0.0});
  expect_vector_close(metric.template oriented_face_area_vector<2, MetricFaceSide::Upper>(index),
                      RealVector<3>{0.0, 1.0, 0.0});

  const RealVector<3> reference{0.25, 0.125, Real(1) / Real(6)};
  const auto inverse = metric.inverse_map(map.map(reference));
  ASSERT_TRUE(inverse.succeeded());
  expect_vector_close(inverse.reference, reference);
}

TEST(test_nd_metric_provider, embedded_cartesian_inverse_refuses_off_manifold_points) {
  const auto map = CartesianCoordinateMap<1, 3>::make(RealVector<3>{2.0, 3.0, 4.0},
                                                      RealVector<1>{2.0}, std::array<int, 1>{1});
  const auto metric = prepare_metric_provider(Box<1>{Index<1>{0}, Index<1>{3}}, map);

  const auto accepted = metric.inverse_map(RealVector<3>{2.0, 3.5, 4.0});
  ASSERT_TRUE(accepted.succeeded());
  EXPECT_TRUE(close(accepted.reference[0], Real(0.25)));

  const auto off_manifold = metric.inverse_map(RealVector<3>{2.01, 3.5, 4.0});
  EXPECT_EQ(off_manifold.status, InverseMapStatus::OffEmbeddedManifold);
  const auto non_finite =
      metric.inverse_map(RealVector<3>{2.0, std::numeric_limits<Real>::quiet_NaN(), 4.0});
  EXPECT_EQ(non_finite.status, InverseMapStatus::NonFinitePoint);
}

TEST(test_nd_metric_provider, polar_sector_uses_exact_integrated_measures_and_face_vectors) {
  const auto map =
      PlanarPolarCoordinateMap::make(RealVector<2>{2.0, -1.0}, Real(1), Real(3), Real(0), kPi);
  const auto metric = prepare_metric_provider(Box<2>{Index<2>{0, 0}, Index<2>{1, 3}}, map);
  const Index<2> index{0, 0};
  const Real angle = kPi / Real(8);

  expect_vector_close(
      metric.cell_center(index),
      RealVector<2>{Real(2) + Real(1.5) * std::cos(angle), Real(-1) + Real(1.5) * std::sin(angle)});
  EXPECT_TRUE(close(metric.cell_measure(index), Real(3) * kPi / Real(8)));

  const auto jacobian = metric.jacobian(index);
  EXPECT_TRUE(close(jacobian[0][0], Real(2) * std::cos(angle)));
  EXPECT_TRUE(close(jacobian[1][0], Real(2) * std::sin(angle)));
  EXPECT_TRUE(close(jacobian[0][1], -Real(1.5) * kPi * std::sin(angle)));
  EXPECT_TRUE(close(jacobian[1][1], Real(1.5) * kPi * std::cos(angle)));

  const Real sine = std::sin(kPi / Real(4));
  const Real cosine = std::cos(kPi / Real(4));
  expect_vector_close(metric.template oriented_face_area_vector<0, MetricFaceSide::Upper>(index),
                      RealVector<2>{Real(2) * sine, Real(2) * (Real(1) - cosine)});
  expect_vector_close(metric.template oriented_face_area_vector<1, MetricFaceSide::Lower>(index),
                      RealVector<2>{0.0, -1.0});
  expect_vector_close(metric.template face_center<1, MetricFaceSide::Upper>(index),
                      RealVector<2>{Real(2) + Real(1.5) * cosine, Real(-1) + Real(1.5) * sine});

  const RealVector<2> reference{0.25, 0.125};
  const auto inverse = metric.inverse_map(map.map(reference));
  ASSERT_TRUE(inverse.succeeded());
  expect_vector_close(inverse.reference, reference);
}

TEST(test_nd_metric_provider, integrated_face_vectors_close_for_cartesian_and_polar_cells) {
  const auto cartesian = prepare_metric_provider(
      Box<3>{Index<3>{-2, 3, 7}, Index<3>{1, 5, 8}},
      CartesianCoordinateMap<3>::make(RealVector<3>{1.0, -2.0, 5.0}, RealVector<3>{4.0, 6.0, 8.0},
                                      std::array<int, 3>{1, 2, 0}, std::array<int, 3>{-1, 1, -1}));
  expect_vector_close(face_closure(cartesian, Index<3>{0, 4, 8}), RealVector<3>{});

  const auto polar = prepare_metric_provider(
      Box<2>{Index<2>{-4, 8}, Index<2>{3, 15}},
      PlanarPolarCoordinateMap::make(RealVector<2>{-3.0, 2.0}, Real(0.5), Real(5), -kPi / Real(3),
                                     kPi / Real(2)));
  // This is the geometric-conservation/free-stream identity: a constant physical flux has zero
  // divergence because the exact outward face vectors close on every mapped control volume.
  expect_vector_close(face_closure(polar, Index<2>{-1, 11}), RealVector<2>{}, Real(2e-12));
}

TEST(test_nd_metric_provider, invalid_maps_domains_and_polar_inverse_fail_closed) {
  EXPECT_THROW(
      (void)(CartesianCoordinateMap<2, 3>::make(RealVector<3>{0.0, 0.0, 0.0},
                                                RealVector<2>{1.0, 2.0}, std::array<int, 2>{1, 1})),
      std::invalid_argument);
  EXPECT_THROW((void)CartesianCoordinateMap<1>::make(RealVector<1>{0.0}, RealVector<1>{0.0}),
               std::invalid_argument);
  EXPECT_THROW((void)CartesianCoordinateMap<1>::make(
                   RealVector<1>{std::numeric_limits<Real>::infinity()}, RealVector<1>{1.0}),
               std::invalid_argument);
  EXPECT_THROW((void)PlanarPolarCoordinateMap::make(RealVector<2>{}, Real(0), Real(2)),
               std::invalid_argument);
  EXPECT_THROW((void)PlanarPolarCoordinateMap::make(RealVector<2>{}, Real(1), Real(2), Real(0),
                                                    Real(2) * kPi + Real(0.1)),
               std::invalid_argument);

  const auto cartesian = CartesianCoordinateMap<2>::make(RealVector<2>{}, RealVector<2>{1.0, 1.0});
  EXPECT_THROW((void)prepare_metric_provider(Box<2>{}, cartesian), std::invalid_argument);

  const auto polar = prepare_metric_provider(
      Box<2>{Index<2>{0, 0}, Index<2>{3, 3}},
      PlanarPolarCoordinateMap::make(RealVector<2>{}, Real(1), Real(3), Real(0), kPi));
  EXPECT_EQ(polar.inverse_map(RealVector<2>{}).status, InverseMapStatus::SingularPoint);
  EXPECT_EQ(polar.inverse_map(RealVector<2>{4.0, 0.0}).status, InverseMapStatus::OutsidePatch);
  EXPECT_EQ(polar.inverse_map(RealVector<2>{0.0, -2.0}).status, InverseMapStatus::OutsidePatch);
}
