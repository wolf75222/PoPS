#include <gtest/gtest.h>

#include <pops/mesh/geometry/geometry.hpp>

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
