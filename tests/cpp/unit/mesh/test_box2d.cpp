// Box2D : arithmetique entiere de l'espace d'indices (grow, refine/coarsen,
// intersect, contains), y compris la division plancher du coarsen sur les
// indices negatifs des ghosts.

#include <gtest/gtest.h>

#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/index/real_vector.hpp>

#include <limits>
#include <type_traits>

using namespace pops;

static_assert(std::is_aggregate_v<Box2D>);
static_assert(std::is_trivially_copyable_v<Box2D>);
static_assert(Index<1>::rank == 1 && Index<2>::rank == 2 && Index<3>::rank == 3);
static_assert(Extent<1>::rank == 1 && Extent<2>::rank == 2 && Extent<3>::rank == 3);
static_assert(RealVector<1>::rank == 1 && RealVector<2>::rank == 2 && RealVector<3>::rank == 3);
static_assert(std::is_trivially_copyable_v<Index<1>> && std::is_trivially_copyable_v<Index<2>> &&
              std::is_trivially_copyable_v<Index<3>>);
static_assert(std::is_standard_layout_v<Index<1>> && std::is_standard_layout_v<Index<2>> &&
              std::is_standard_layout_v<Index<3>>);
static_assert(std::is_trivially_copyable_v<Extent<1>> && std::is_trivially_copyable_v<Extent<2>> &&
              std::is_trivially_copyable_v<Extent<3>>);
static_assert(std::is_standard_layout_v<Extent<1>> && std::is_standard_layout_v<Extent<2>> &&
              std::is_standard_layout_v<Extent<3>>);
static_assert(std::is_trivially_copyable_v<RealVector<1>> &&
              std::is_trivially_copyable_v<RealVector<2>> &&
              std::is_trivially_copyable_v<RealVector<3>>);
static_assert(std::is_standard_layout_v<RealVector<1>> &&
              std::is_standard_layout_v<RealVector<2>> && std::is_standard_layout_v<RealVector<3>>);
static_assert(std::is_constructible_v<Index<2>, int, int>);
static_assert(!std::is_constructible_v<Index<1>, long long>);
static_assert(std::is_constructible_v<Extent<2>, int, unsigned int>);
static_assert(!std::is_constructible_v<Extent<1>, unsigned long long>);
static_assert(std::is_constructible_v<RealVector<2>, float, int>);
static_assert(!std::is_constructible_v<RealVector<1>, long long>);
static_assert(
    std::is_constructible_v<RealVector<1>, long double> ==
    (std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits &&
     std::numeric_limits<long double>::max_exponent <= std::numeric_limits<double>::max_exponent &&
     std::numeric_limits<long double>::min_exponent >= std::numeric_limits<double>::min_exponent));

TEST(test_box2d, extents_and_contains) {
  Box2D b = Box2D::from_extents(4, 3);  // [0..3] x [0..2]
  EXPECT_TRUE(b.nx() == 4 && b.ny() == 3) << "extents";
  EXPECT_EQ(b.num_cells(), 12) << "num_cells";
  EXPECT_TRUE(b.contains(3, 2) && !b.contains(4, 0)) << "contains";
}

TEST(test_box2d, grow) {
  Box2D b = Box2D::from_extents(4, 3);  // [0..3] x [0..2]
  Box2D g = b.grow(1);                  // [-1..4] x [-1..3]
  EXPECT_TRUE(g.lo[0] == -1 && g.hi[0] == 4 && g.lo[1] == -1 && g.hi[1] == 3) << "grow";
  EXPECT_TRUE(g.nx() == 6 && g.ny() == 5) << "grow_extents";
}

TEST(test_box2d, refine_and_coarsen_roundtrip) {
  Box2D b = Box2D::from_extents(4, 3);  // [0..3] x [0..2]
  Box2D r = b.refine(2);                // [0..7] x [0..5]
  EXPECT_TRUE(r.lo[0] == 0 && r.hi[0] == 7 && r.hi[1] == 5) << "refine";
  EXPECT_EQ(r.coarsen(2), b) << "coarsen_roundtrip";
}

TEST(test_box2d, coarsen_floors_negative_indices) {
  // coarsen avec indices negatifs : floor(-1/2) = -1, floor(2/2) = 1
  Box2D neg{{-1, -1}, {2, 2}};
  Box2D c = neg.coarsen(2);
  EXPECT_TRUE(c.lo[0] == -1 && c.hi[0] == 1) << "coarsen_floor";
}

TEST(test_box2d, intersection_clips_to_overlap) {
  Box2D a{{0, 0}, {5, 5}};
  Box2D d{{3, 3}, {9, 9}};
  Box2D in = a.intersect(d);  // [3..5] x [3..5]
  EXPECT_TRUE(in.lo[0] == 3 && in.hi[0] == 5 && in.lo[1] == 3 && in.hi[1] == 5) << "intersect";
  EXPECT_TRUE(a.intersect(Box2D{{10, 10}, {12, 12}}).empty()) << "intersect_empty";
  EXPECT_TRUE(a.contains(in) && !a.contains(d)) << "contains_box";
}

TEST(test_box2d, exact_extents_do_not_overflow_signed_int_arithmetic) {
  constexpr int lo = std::numeric_limits<int>::min();
  constexpr int hi = std::numeric_limits<int>::max();
  const Box2D full_width{{lo, 0}, {hi, 0}};
  const Box2D full_square{{lo, lo}, {hi, hi}};

  EXPECT_EQ(full_width.length64(0), std::int64_t{1} << 32);
  EXPECT_THROW((void)full_width.nx(), std::overflow_error);
  EXPECT_EQ(full_width.num_cells(), std::int64_t{1} << 32);
  EXPECT_THROW((void)full_square.num_cells(), std::overflow_error);
}

TEST(test_box2d, constructors_and_transforms_reject_invalid_or_overflowing_indices) {
  constexpr int lo = std::numeric_limits<int>::min();
  constexpr int hi = std::numeric_limits<int>::max();
  const Box2D at_min{{lo, 0}, {lo, 0}};
  const Box2D at_max{{hi, 0}, {hi, 0}};
  const Box2D refine_overflow{{hi / 2 + 1, 0}, {hi / 2 + 1, 0}};

  EXPECT_THROW((void)Box2D::from_extents(-1, 1), std::invalid_argument);
  EXPECT_TRUE(Box2D::from_extents(0, 3).empty());
  EXPECT_EQ(Box2D{}.grow(4), Box2D{});
  EXPECT_EQ(Box2D{}.shift(0, hi), Box2D{});
  EXPECT_EQ(Box2D{}.refine(hi), Box2D{});
  EXPECT_EQ(Box2D{}.coarsen(hi), Box2D{});

  EXPECT_THROW((void)at_min.grow(1), std::overflow_error);
  EXPECT_THROW((void)at_max.shift(0, 1), std::overflow_error);
  EXPECT_THROW((void)refine_overflow.refine(2), std::overflow_error);
  EXPECT_THROW((void)Box2D::from_extents(1, 1).grow(2, 1), std::invalid_argument);
  EXPECT_THROW((void)Box2D::from_extents(1, 1).shift(-1, 1), std::invalid_argument);
  EXPECT_THROW((void)Box2D::from_extents(1, 1).refine(0), std::invalid_argument);
  EXPECT_THROW((void)Box2D::from_extents(1, 1).coarsen(-1), std::invalid_argument);
}

TEST(test_box2d, floor_div_rejects_undefined_integer_cases) {
  constexpr int lo = std::numeric_limits<int>::min();

  EXPECT_THROW((void)floor_div(1, 0), std::invalid_argument);
  EXPECT_THROW((void)floor_div(lo, -1), std::overflow_error);
  EXPECT_EQ(floor_div(lo, 2), lo / 2);
}

TEST(test_box2d, compile_time_ranked_boxes_cover_1d_2d_and_3d) {
  const Box<1> line = Box<1>::from_extents(Extent<1>{7});
  EXPECT_FALSE(line.empty());
  EXPECT_EQ(line.extent()[0], 7);
  EXPECT_EQ(line.numPts(), 7);
  EXPECT_TRUE(line.contains(Index<1>{6}));

  const Box<2> plane{Index<2>{-3, 4}, Index<2>{2, 6}};
  EXPECT_EQ(plane.extent()[0], 6);
  EXPECT_EQ(plane.extent()[1], 3);
  EXPECT_EQ(plane.numPts(), 18);
  EXPECT_TRUE(plane.contains(Index<2>{-3, 4}));
  EXPECT_FALSE(plane.contains(Index<2>{3, 4}));
  EXPECT_EQ(plane.grow(1).lo[0], -4);
  EXPECT_EQ(plane.refine(2).coarsen(2), plane);

  const Box<3> volume{Index<3>{-2, 5, 9}, Index<3>{1, 6, 11}};
  EXPECT_EQ(volume.extent()[0], 4);
  EXPECT_EQ(volume.extent()[1], 2);
  EXPECT_EQ(volume.extent()[2], 3);
  EXPECT_EQ(volume.numPts(), 24);
  EXPECT_EQ(volume.intersect(Box<3>{Index<3>{0, 4, 10}, Index<3>{4, 8, 10}}).numPts(), 4);

  const RealVector<3> point{1.25, -2.5, 0.75};
  EXPECT_DOUBLE_EQ(point[0], 1.25);
  EXPECT_DOUBLE_EQ(point[1], -2.5);
  EXPECT_DOUBLE_EQ(point[2], 0.75);
}

TEST(test_box2d, compile_time_ranked_box_empty_and_overflow_contracts) {
  const Box<3> empty{};
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.numPts(), 0);
  EXPECT_FALSE(empty.contains(Index<3>{0, 0, 0}));

  const Box<2> full_width{Index<2>{std::numeric_limits<int>::min(), 0},
                          Index<2>{std::numeric_limits<int>::max(), 0}};
  EXPECT_EQ(full_width.extent()[0], std::int64_t{1} << 32);
  EXPECT_EQ(full_width.numPts(), std::int64_t{1} << 32);
  EXPECT_THROW((void)Box<1>::from_extents(Extent<1>{-1}), std::invalid_argument);
  EXPECT_THROW((void)full_width.grow(1), std::overflow_error);
}

TEST(test_box2d, ranked_box_shift_is_checked_and_preserves_empty_boxes) {
  const Box<2> box{Index<2>{-3, 4}, Index<2>{1, 7}};
  EXPECT_EQ(box.shift(Index<2>{5, -2}), (Box<2>{Index<2>{2, 2}, Index<2>{6, 5}}));
  EXPECT_EQ(Box<3>{}.shift(Index<3>{1, 2, 3}), Box<3>{});

  const Box<1> at_max{Index<1>{std::numeric_limits<int>::max()},
                      Index<1>{std::numeric_limits<int>::max()}};
  const Box<1> at_min{Index<1>{std::numeric_limits<int>::min()},
                      Index<1>{std::numeric_limits<int>::min()}};
  EXPECT_THROW((void)at_max.shift(Index<1>{1}), std::overflow_error);
  EXPECT_THROW((void)at_min.shift(Index<1>{-1}), std::overflow_error);
}
