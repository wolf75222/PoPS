#include <gtest/gtest.h>

#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/geometry/swept_interval.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

using pops::Box;
using pops::Extent;
using pops::Geometry;
using pops::Index;
using pops::Real;
using pops::RealVector;

TEST(test_geometry, swept_interval_uses_actual_endpoints_and_preserves_constant_density) {
  const Real old_faces[] = {0, Real(.2), Real(.5), 1};
  const Real displacements[] = {Real(.01), Real(-.02), Real(.01), Real(.01)};
  const Real density = Real(2.3);
  const Real tol = 32 * std::numeric_limits<Real>::epsilon();
  Real old_total = 0, new_total = 0, amount_total = 0;
  for (int cell = 0; cell < 3; ++cell) {
    const auto geometry = pops::SweptInterval::prepare(
        old_faces[cell], old_faces[cell + 1],
        old_faces[cell] + displacements[cell], old_faces[cell + 1] + displacements[cell + 1],
        displacements[cell], displacements[cell + 1], tol);
    const Real amount = geometry.updated_amount(density * geometry.old_measure(), density, density,
                                                Real(.12), Real(.12), 0);
    EXPECT_NEAR(amount / geometry.new_measure(), density, tol * density);
    EXPECT_NEAR(geometry.gcl_residual(), 0, tol);
    old_total += geometry.old_measure();
    new_total += geometry.new_measure();
    amount_total += amount;
  }
  EXPECT_NEAR(old_total, new_total, tol);
  EXPECT_NEAR(amount_total, density, tol * density);
}

TEST(test_geometry, swept_interval_refuses_stale_duration_even_with_matching_total_volume) {
  // Both cells still sum to length one. The dt=.1 sweep is nevertheless
  // incompatible with the actual endpoints of the dt=.2 attempt.
  const Real tol = 32 * std::numeric_limits<Real>::epsilon();
  EXPECT_THROW((void)pops::SweptInterval::prepare(0, Real(.5), 0, Real(.52),
                                                  0, Real(.01), tol), std::invalid_argument);
  EXPECT_THROW((void)pops::SweptInterval::prepare(0, 1, 0, 0, 0, -1, tol),
               std::invalid_argument);
}

TEST(test_geometry, swept_interval_physical_flux_and_source_are_integrated_exactly_once) {
  const auto geometry = pops::SweptInterval::prepare(0, Real(.5), 0, Real(.6),
                                                     0, Real(.1), Real(1e-6));
  // Q=1, physical net=.3, mesh amount=.3, source=.2 -> Q+=1.2.
  EXPECT_NEAR(geometry.updated_amount(1, 2, 3, Real(.1), Real(.4), Real(.2)),
              Real(1.2), 16 * std::numeric_limits<Real>::epsilon());
}

static_assert(Geometry<1>::rank == 1 && Geometry<2>::rank == 2 && Geometry<3>::rank == 3);
static_assert(std::is_trivially_copyable_v<Geometry<1>> &&
              std::is_trivially_copyable_v<Geometry<2>> &&
              std::is_trivially_copyable_v<Geometry<3>>);

TEST(test_geometry, spacing_cells_and_faces_are_exact_for_1d_2d_and_3d) {
  const Geometry<1> line = Geometry<1>::from_bounds(Box<1>{Index<1>{-2}, Index<1>{1}},
                                                    RealVector<1>{10.0}, RealVector<1>{14.0});
  EXPECT_DOUBLE_EQ(line.spacing(0), 1.0);
  EXPECT_DOUBLE_EQ(line.face_coordinate(0, -2), 10.0);
  EXPECT_DOUBLE_EQ(line.cell_coordinate(0, -2), 10.5);
  EXPECT_DOUBLE_EQ(line.cell_coordinate(0, -3), 9.5);
  EXPECT_EQ(line.cell_center(Index<1>{1}), RealVector<1>{13.5});

  const Geometry<2> plane = Geometry<2>::from_bounds(
      Box<2>{Index<2>{-3, 7}, Index<2>{0, 8}}, RealVector<2>{-2.0, 10.0}, RealVector<2>{2.0, 12.0});
  EXPECT_DOUBLE_EQ(plane.spacing(0), 1.0);
  EXPECT_DOUBLE_EQ(plane.spacing(1), 1.0);
  EXPECT_EQ(plane.cell_center(Index<2>{-3, 7}), (RealVector<2>{-1.5, 10.5}));
  EXPECT_EQ(plane.cell_center(Index<2>{-4, 8}), (RealVector<2>{-2.5, 11.5}));
  EXPECT_EQ(plane.lower_face(Index<2>{0, 8}), (RealVector<2>{1.0, 11.0}));

  const Geometry<3> volume =
      Geometry<3>::from_bounds(Box<3>{Index<3>{1, -2, 4}, Index<3>{2, 1, 8}},
                               RealVector<3>{0.0, 10.0, -1.0}, RealVector<3>{4.0, 14.0, 9.0});
  EXPECT_DOUBLE_EQ(volume.spacing(0), 2.0);
  EXPECT_DOUBLE_EQ(volume.spacing(1), 1.0);
  EXPECT_DOUBLE_EQ(volume.spacing(2), 2.0);
  EXPECT_EQ(volume.cell_center(Index<3>{1, -2, 4}), (RealVector<3>{1.0, 10.5, 0.0}));
  EXPECT_EQ(volume.lower_face(Index<3>{3, -3, 9}), (RealVector<3>{4.0, 9.0, 9.0}));
}

TEST(test_geometry, anisotropic_refinement_preserves_physical_bounds_and_index_origin) {
  const Geometry<2> coarse = Geometry<2>::from_bounds(
      Box<2>{Index<2>{-3, 7}, Index<2>{0, 8}}, RealVector<2>{-2.0, 10.0}, RealVector<2>{2.0, 12.0});
  const Geometry<2> fine = coarse.refine(Extent<2>{2, 3});

  EXPECT_EQ(fine.domain(), (Box<2>{Index<2>{-6, 21}, Index<2>{1, 26}}));
  EXPECT_EQ(fine.lower(), coarse.lower());
  EXPECT_EQ(fine.upper(), coarse.upper());
  EXPECT_DOUBLE_EQ(fine.spacing(0), 0.5);
  EXPECT_DOUBLE_EQ(fine.spacing(1), Real(1) / Real(3));
  EXPECT_DOUBLE_EQ(fine.face_coordinate(0, -6), -2.0);
  EXPECT_DOUBLE_EQ(fine.face_coordinate(1, 21), 10.0);
  EXPECT_DOUBLE_EQ(fine.cell_coordinate(0, 1), 1.75);
}

TEST(test_geometry, invalid_domains_bounds_and_refinement_ratios_fail_closed) {
  const Box<1> line{Index<1>{0}, Index<1>{3}};
  EXPECT_THROW((void)Geometry<1>::from_bounds(Box<1>{}, RealVector<1>{0.0}, RealVector<1>{1.0}),
               std::invalid_argument);
  EXPECT_THROW((void)Geometry<1>::from_bounds(line, RealVector<1>{1.0}, RealVector<1>{1.0}),
               std::invalid_argument);
  EXPECT_THROW((void)Geometry<1>::from_bounds(line, RealVector<1>{2.0}, RealVector<1>{1.0}),
               std::invalid_argument);
  EXPECT_THROW((void)Geometry<1>::from_bounds(
                   line, RealVector<1>{std::numeric_limits<Real>::quiet_NaN()}, RealVector<1>{1.0}),
               std::invalid_argument);
  EXPECT_THROW((void)Geometry<1>::from_bounds(line, RealVector<1>{0.0},
                                              RealVector<1>{std::numeric_limits<Real>::infinity()}),
               std::invalid_argument);
  EXPECT_THROW(
      (void)Geometry<1>::from_bounds(line, RealVector<1>{std::numeric_limits<Real>::lowest()},
                                     RealVector<1>{std::numeric_limits<Real>::max()}),
      std::invalid_argument);
  EXPECT_THROW(
      (void)Geometry<1>::from_bounds(line, RealVector<1>{0.0},
                                     RealVector<1>{std::numeric_limits<Real>::denorm_min()}),
      std::invalid_argument);

  const Geometry<2> plane = Geometry<2>::from_bounds(
      Box<2>{Index<2>{0, 0}, Index<2>{1, 1}}, RealVector<2>{0.0, 0.0}, RealVector<2>{1.0, 1.0});
  EXPECT_THROW((void)plane.refine(Extent<2>{0, 2}), std::invalid_argument);
  EXPECT_THROW((void)plane.refine(Extent<2>{2, -1}), std::invalid_argument);
  EXPECT_THROW((void)plane.refine(
                   Extent<2>{static_cast<std::int64_t>(std::numeric_limits<int>::max()) + 1, 1}),
               std::invalid_argument);

  const Geometry<1> at_max = Geometry<1>::from_bounds(
      Box<1>{Index<1>{std::numeric_limits<int>::max()}, Index<1>{std::numeric_limits<int>::max()}},
      RealVector<1>{0.0}, RealVector<1>{1.0});
  EXPECT_THROW((void)at_max.refine(Extent<1>{2}), std::overflow_error);
}
