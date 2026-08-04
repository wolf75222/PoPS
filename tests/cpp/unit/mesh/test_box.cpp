#include <gtest/gtest.h>

#include <pops/mesh/index/box.hpp>
#include <pops/mesh/index/real_vector.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

using pops::Box;
using pops::Extent;
using pops::Index;
using pops::RealVector;

static_assert(Index<1>::rank == 1 && Index<2>::rank == 2 && Index<3>::rank == 3);
static_assert(Extent<1>::rank == 1 && Extent<2>::rank == 2 && Extent<3>::rank == 3);
static_assert(RealVector<1>::rank == 1 && RealVector<2>::rank == 2 && RealVector<3>::rank == 3);
static_assert(Box<1>::rank == 1 && Box<2>::rank == 2 && Box<3>::rank == 3);
static_assert(std::is_trivially_copyable_v<Box<1>> && std::is_trivially_copyable_v<Box<2>> &&
              std::is_trivially_copyable_v<Box<3>>);
static_assert(std::is_standard_layout_v<Box<1>> && std::is_standard_layout_v<Box<2>> &&
              std::is_standard_layout_v<Box<3>>);
static_assert(std::is_constructible_v<Index<2>, int, int>);
static_assert(!std::is_constructible_v<Index<1>, long long>);
static_assert(std::is_constructible_v<Extent<2>, int, unsigned int>);
static_assert(!std::is_constructible_v<Extent<1>, unsigned long long>);
static_assert(std::is_constructible_v<RealVector<2>, float, int>);
static_assert(!std::is_constructible_v<RealVector<1>, long long>);

TEST(test_box, compile_time_ranked_boxes_cover_anisotropic_1d_2d_and_3d) {
  const Box<1> line = Box<1>::from_extents(Extent<1>{7});
  EXPECT_EQ(line.extent(), Extent<1>{7});
  EXPECT_EQ(line.numPts(), 7);
  EXPECT_TRUE(line.contains(Index<1>{6}));
  EXPECT_FALSE(line.contains(Index<1>{7}));

  const Box<2> plane{Index<2>{-3, 4}, Index<2>{2, 6}};
  EXPECT_EQ(plane.extent(), (Extent<2>{6, 3}));
  EXPECT_EQ(plane.numPts(), 18);
  EXPECT_TRUE(plane.contains(Index<2>{-3, 4}));
  EXPECT_FALSE(plane.contains(Index<2>{3, 4}));

  const Box<3> volume{Index<3>{-2, 5, 9}, Index<3>{1, 6, 11}};
  EXPECT_EQ(volume.extent(), (Extent<3>{4, 2, 3}));
  EXPECT_EQ(volume.numPts(), 24);
  EXPECT_EQ(volume.intersect(Box<3>{Index<3>{0, 4, 10}, Index<3>{4, 8, 10}}).numPts(), 4);

  const RealVector<3> point{1.25, -2.5, 0.75};
  EXPECT_DOUBLE_EQ(point[0], 1.25);
  EXPECT_DOUBLE_EQ(point[1], -2.5);
  EXPECT_DOUBLE_EQ(point[2], 0.75);
}

TEST(test_box, transforms_are_rank_generic_checked_and_preserve_empty_boxes) {
  const Box<2> plane{Index<2>{-3, 4}, Index<2>{1, 7}};
  EXPECT_EQ(plane.grow(1), (Box<2>{Index<2>{-4, 3}, Index<2>{2, 8}}));
  EXPECT_EQ(plane.grow(0, 2), (Box<2>{Index<2>{-5, 4}, Index<2>{3, 7}}));
  EXPECT_EQ(plane.shift(Index<2>{5, -2}), (Box<2>{Index<2>{2, 2}, Index<2>{6, 5}}));
  EXPECT_EQ(plane.refine(3).coarsen(3), plane);

  const Box<3> negative{Index<3>{-5, -1, 2}, Index<3>{2, 4, 7}};
  EXPECT_EQ(negative.coarsen(2), (Box<3>{Index<3>{-3, -1, 1}, Index<3>{1, 2, 3}}));

  EXPECT_EQ(Box<1>{}.grow(4), Box<1>{});
  EXPECT_EQ(Box<2>{}.shift(Index<2>{1, 2}), Box<2>{});
  EXPECT_EQ(Box<3>{}.refine(4), Box<3>{});
  EXPECT_EQ(Box<3>{}.coarsen(4), Box<3>{});
}

TEST(test_box, intersection_and_containment_are_exact_in_every_rank) {
  const Box<1> line{Index<1>{-4}, Index<1>{3}};
  EXPECT_EQ(line.intersect(Box<1>{Index<1>{1}, Index<1>{8}}), (Box<1>{Index<1>{1}, Index<1>{3}}));

  const Box<2> plane{Index<2>{0, 0}, Index<2>{5, 5}};
  const Box<2> overlap = plane.intersect(Box<2>{Index<2>{3, 3}, Index<2>{9, 9}});
  EXPECT_EQ(overlap, (Box<2>{Index<2>{3, 3}, Index<2>{5, 5}}));
  EXPECT_TRUE(plane.contains(overlap));
  EXPECT_FALSE(plane.contains(Box<2>{Index<2>{3, 3}, Index<2>{9, 9}}));

  const Box<3> volume{Index<3>{0, 0, 0}, Index<3>{2, 2, 2}};
  EXPECT_TRUE(volume.intersect(Box<3>{Index<3>{4, 4, 4}, Index<3>{5, 5, 5}}).empty());
}

TEST(test_box, empty_and_full_signed_ranges_do_not_narrow_exact_extents) {
  const Box<3> empty{};
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.extent(), Extent<3>{});
  EXPECT_EQ(empty.numPts(), 0);
  EXPECT_FALSE(empty.contains(Index<3>{0, 0, 0}));

  constexpr int minimum = std::numeric_limits<int>::min();
  constexpr int maximum = std::numeric_limits<int>::max();
  const Box<2> full_width{Index<2>{minimum, 0}, Index<2>{maximum, 0}};
  EXPECT_EQ(full_width.length(0), std::int64_t{1} << 32);
  EXPECT_EQ(full_width.numPts(), std::int64_t{1} << 32);
  const Box<3> full_volume{Index<3>{minimum, minimum, minimum},
                           Index<3>{maximum, maximum, maximum}};
  EXPECT_THROW((void)full_volume.numPts(), std::overflow_error);
}

TEST(test_box, invalid_parameters_and_signed_index_overflow_fail_closed) {
  constexpr int minimum = std::numeric_limits<int>::min();
  constexpr int maximum = std::numeric_limits<int>::max();
  const Box<1> at_min{Index<1>{minimum}, Index<1>{minimum}};
  const Box<1> at_max{Index<1>{maximum}, Index<1>{maximum}};
  const Box<1> refine_overflow{Index<1>{maximum / 2 + 1}, Index<1>{maximum / 2 + 1}};

  EXPECT_THROW((void)Box<1>::from_extents(Extent<1>{-1}), std::invalid_argument);
  EXPECT_TRUE(Box<2>::from_extents(Extent<2>{0, 3}).empty());
  EXPECT_THROW((void)at_min.grow(1), std::overflow_error);
  EXPECT_THROW((void)at_max.shift(Index<1>{1}), std::overflow_error);
  EXPECT_THROW((void)refine_overflow.refine(2), std::overflow_error);
  EXPECT_THROW((void)Box<2>::from_extents(Extent<2>{1, 1}).grow(2, 1), std::invalid_argument);
  EXPECT_THROW((void)Box<2>::from_extents(Extent<2>{1, 1}).refine(0), std::invalid_argument);
  EXPECT_THROW((void)Box<2>::from_extents(Extent<2>{1, 1}).coarsen(-1), std::invalid_argument);
}
