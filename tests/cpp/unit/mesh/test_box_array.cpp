#include <gtest/gtest.h>

#include <pops/mesh/layout/box_array.hpp>

#include <limits>
#include <stdexcept>
#include <vector>

using pops::Box;
using pops::Extent;
using pops::Index;
using pops::mesh::BoxArray;
using pops::mesh::BoxArrayValidationBudget;
using pops::mesh::ExactCellCount;

namespace {

constexpr BoxArrayValidationBudget kBudget{128, 8192};

}  // namespace

TEST(test_box_array, from_domain_tiles_anisotropic_1d_2d_and_3d_with_axis_zero_contiguous) {
  const Box<1> line_domain{Index<1>{-5}, Index<1>{4}};
  const BoxArray<1> line = BoxArray<1>::from_domain(line_domain, Extent<1>{4});
  ASSERT_EQ(line.size(), 3U);
  EXPECT_EQ(line[0], (Box<1>{Index<1>{-5}, Index<1>{-2}}));
  EXPECT_EQ(line[1], (Box<1>{Index<1>{-1}, Index<1>{1}}));
  EXPECT_EQ(line[2], (Box<1>{Index<1>{2}, Index<1>{4}}));
  EXPECT_TRUE(line.tiles_exactly(line_domain, kBudget));

  const Box<2> plane_domain{Index<2>{-3, 5}, Index<2>{4, 10}};
  const BoxArray<2> plane = BoxArray<2>::from_domain(plane_domain, Extent<2>{3, 4});
  ASSERT_EQ(plane.size(), 6U);
  EXPECT_EQ(plane[0], (Box<2>{Index<2>{-3, 5}, Index<2>{-1, 7}}));
  EXPECT_EQ(plane[1], (Box<2>{Index<2>{0, 5}, Index<2>{2, 7}}));
  EXPECT_EQ(plane[3], (Box<2>{Index<2>{-3, 8}, Index<2>{-1, 10}}));
  EXPECT_EQ(plane.bounding_box(), plane_domain);
  EXPECT_EQ(plane.exact_cell_count(), ExactCellCount::from_uint64(48));
  EXPECT_TRUE(plane.tiles_exactly(plane_domain, kBudget));

  const Box<3> volume_domain{Index<3>{-2, 1, 4}, Index<3>{2, 3, 6}};
  const BoxArray<3> volume = BoxArray<3>::from_domain(volume_domain, Extent<3>{2, 2, 2});
  ASSERT_EQ(volume.size(), 12U);
  EXPECT_EQ(volume[0], (Box<3>{Index<3>{-2, 1, 4}, Index<3>{-1, 2, 5}}));
  EXPECT_EQ(volume[1], (Box<3>{Index<3>{0, 1, 4}, Index<3>{1, 2, 5}}));
  EXPECT_EQ(volume[3], (Box<3>{Index<3>{-2, 3, 4}, Index<3>{-1, 3, 5}}));
  EXPECT_EQ(volume.bounding_box(), volume_domain);
  EXPECT_EQ(volume.exact_cell_count(), ExactCellCount::from_uint64(45));
  EXPECT_TRUE(volume.tiles_exactly(volume_domain, kBudget));
}

TEST(test_box_array, exact_tiling_rejects_holes_overlaps_outside_and_empty_members) {
  const Box<1> line{Index<1>{0}, Index<1>{3}};
  EXPECT_FALSE(BoxArray<1>(std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{1}},
                                               Box<1>{Index<1>{3}, Index<1>{3}}})
                   .tiles_exactly(line, kBudget));
  EXPECT_FALSE(BoxArray<1>(std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{2}},
                                               Box<1>{Index<1>{2}, Index<1>{3}}})
                   .tiles_exactly(line, kBudget));
  EXPECT_FALSE(BoxArray<1>(std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{2}},
                                               Box<1>{Index<1>{3}, Index<1>{4}}})
                   .tiles_exactly(line, kBudget));
  EXPECT_FALSE(BoxArray<1>(std::vector<Box<1>>{Box<1>{}}).tiles_exactly(line, kBudget));

  const Box<3> volume{Index<3>{0, 0, 0}, Index<3>{1, 1, 1}};
  EXPECT_FALSE(BoxArray<3>(std::vector<Box<3>>{Box<3>{Index<3>{0, 0, 0}, Index<3>{1, 1, 0}},
                                               Box<3>{Index<3>{0, 0, 1}, Index<3>{1, 1, 2}}})
                   .tiles_exactly(volume, kBudget));

  EXPECT_TRUE(BoxArray<2>{}.tiles_exactly(Box<2>{}, kBudget));
  EXPECT_FALSE(BoxArray<2>(std::vector<Box<2>>{Box<2>{}}).tiles_exactly(Box<2>{}, kBudget));
}

TEST(test_box_array, exact_counts_cover_the_full_signed_2d_coordinate_space) {
  constexpr int minimum = std::numeric_limits<int>::min();
  constexpr int maximum = std::numeric_limits<int>::max();
  const Box<2> domain{Index<2>{minimum, minimum}, Index<2>{maximum, maximum}};
  const BoxArray<2> halves(
      std::vector<Box<2>>{Box<2>{Index<2>{minimum, minimum}, Index<2>{-1, maximum}},
                          Box<2>{Index<2>{0, minimum}, Index<2>{maximum, maximum}}});

  EXPECT_EQ(halves.exact_cell_count(), ExactCellCount::power_of_two(64));
  EXPECT_TRUE(halves.tiles_exactly(domain, kBudget));

  const BoxArray<2> bounded = BoxArray<2>::from_domain(domain, Extent<2>{maximum, maximum});
  EXPECT_EQ(bounded.size(), 9U);
  EXPECT_TRUE(bounded.tiles_exactly(domain, kBudget));
  EXPECT_THROW((void)BoxArray<2>::from_domain(domain, Extent<2>{1, 1}), std::length_error);
}

TEST(test_box_array, invalid_limits_and_explicit_validation_budgets_fail_before_unbounded_work) {
  const Box<2> domain{Index<2>{0, 0}, Index<2>{3, 3}};
  EXPECT_THROW((void)BoxArray<2>::from_domain(domain, Extent<2>{0, 2}), std::invalid_argument);
  EXPECT_THROW((void)BoxArray<2>::from_domain(domain, Extent<2>{2, -1}), std::invalid_argument);

  const BoxArray<2> boxes = BoxArray<2>::from_domain(domain, Extent<2>{2, 2});
  EXPECT_THROW((void)boxes.tiles_exactly(domain, BoxArrayValidationBudget{3, 6}),
               std::length_error);
  EXPECT_THROW((void)boxes.tiles_exactly(domain, BoxArrayValidationBudget{4, 5}),
               std::length_error);
  EXPECT_TRUE(boxes.tiles_exactly(domain, BoxArrayValidationBudget{4, 6}));
}
