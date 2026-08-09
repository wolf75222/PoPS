#include <gtest/gtest.h>

#include <pops/mesh/index/box_hash.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/rank_space.hpp>
#include <pops/runtime/checkpoint/spatial_contract.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

using pops::Box;
using pops::Extent;
using pops::Index;
using pops::mesh::BinCoordinate;
using pops::mesh::BinCoordinateHash;
using pops::mesh::BoxArray;
using pops::mesh::BoxArrayValidationBudget;
using pops::mesh::BoxHash;
using pops::mesh::BoxHashBudget;
using pops::mesh::ExactCellCount;
using pops::mesh::RankSpace;
using pops::mesh::suggest_bin;
using pops::runtime::checkpoint::EncodedSpatialContract;
using pops::runtime::checkpoint::SpatialContract;
using pops::runtime::checkpoint::decode_spatial_contract;
using pops::runtime::checkpoint::encode_spatial_contract;
using pops::runtime::checkpoint::prepare_spatial_restart;

constexpr BoxHashBudget kHashBudget{128, 128, 256};
constexpr BoxArrayValidationBudget kTilingBudget{128, 4096};

template <int Dim>
SpatialContract<Dim> checkpoint_contract(const std::array<std::int64_t, Dim>& shape,
                                         const std::array<int, Dim>& ratio) {
  SpatialContract<Dim> result;
  for (int axis = 0; axis < Dim; ++axis) {
    const auto index = static_cast<std::size_t>(axis);
    result.shape[axis] = shape[index];
    result.lower[index] = -static_cast<double>(axis + 1);
    result.upper[index] = result.lower[index] + static_cast<double>(shape[index]);
    result.periodicity[index] = axis % 2 == 0;
  }
  result.refinement_ratios.emplace_back(ratio);
  result.native_layout_identity = "native-spatial-layout:1:sha256:native";
  result.spatial_identity = "checkpoint-spatial-layout:1:sha256:checkpoint";
  return result;
}

template <int Dim>
void expect_hash_superset(const BoxArray<Dim>& boxes, const BoxHash<Dim>& hash,
                          const std::vector<Box<Dim>>& queries) {
  for (const Box<Dim>& query : queries) {
    const std::vector<std::size_t> candidates = hash.query(query);
    EXPECT_TRUE(std::is_sorted(candidates.begin(), candidates.end()));
    EXPECT_EQ(std::adjacent_find(candidates.begin(), candidates.end()), candidates.end());
    for (std::size_t index = 0; index < boxes.size(); ++index)
      if (!query.intersect(boxes[index]).empty())
        EXPECT_NE(std::find(candidates.begin(), candidates.end(), index), candidates.end());
  }
}

TEST(test_nd_layout, checkpoint_spatial_schema_round_trips_exact_1d_2d_and_3d_vectors) {
  const auto line = checkpoint_contract<1>({7}, {3});
  const auto plane = checkpoint_contract<2>({3, 4}, {2, 3});
  const auto volume = checkpoint_contract<3>({2, 3, 4}, {2, 1, 4});

  const auto encoded_line = encode_spatial_contract(line);
  const auto encoded_plane = encode_spatial_contract(plane);
  const auto encoded_volume = encode_spatial_contract(volume);
  EXPECT_EQ(encoded_line.shape.size(), 1U);
  EXPECT_EQ(encoded_plane.refinement_ratios.at(0).size(), 2U);
  EXPECT_EQ(encoded_volume.lower.size(), 3U);
  EXPECT_EQ(decode_spatial_contract<1>(encoded_line), line);
  EXPECT_EQ(decode_spatial_contract<2>(encoded_plane), plane);
  EXPECT_EQ(decode_spatial_contract<3>(encoded_volume), volume);
  EXPECT_EQ(line.cell_count(), 7);
  EXPECT_EQ(plane.cell_count(), 12);
  EXPECT_EQ(volume.cell_count(), 24);
  EXPECT_EQ(plane.shape_at_level(1), (Extent<2>{6, 12}));
  EXPECT_EQ(volume.shape_at_level(1), (Extent<3>{4, 3, 16}));
}

TEST(test_nd_layout, checkpoint_restart_refuses_schema_dimension_and_layout_before_state_work) {
  const auto current = checkpoint_contract<2>({3, 4}, {2, 3});
  auto encoded = encode_spatial_contract(current);
  bool state_allocation_started = false;

  encoded.dimension = 3;
  EXPECT_THROW(
      {
        const auto prepared = prepare_spatial_restart<2>(encoded, current);
        state_allocation_started = prepared.cell_count() > 0;
      },
      std::invalid_argument);
  EXPECT_FALSE(state_allocation_started);

  encoded = encode_spatial_contract(current);
  encoded.schema_version += 1;
  EXPECT_THROW((void)prepare_spatial_restart<2>(encoded, current), std::invalid_argument);

  encoded = encode_spatial_contract(current);
  encoded.shape[1] += 1;
  EXPECT_THROW((void)prepare_spatial_restart<2>(encoded, current), std::invalid_argument);
}

TEST(test_nd_layout, rank_spaces_support_anisotropic_1d_2d_and_3d_extents) {
  const RankSpace<1> line{Index<1>{-4}, Extent<1>{7}};
  EXPECT_EQ(line.size(), 7U);
  EXPECT_TRUE(line.contains(Index<1>{-4}));
  EXPECT_TRUE(line.contains(Index<1>{2}));
  EXPECT_FALSE(line.contains(Index<1>{3}));

  const RankSpace<2> plane{Index<2>{-2, 5}, Extent<2>{3, 4}};
  EXPECT_EQ(plane.size(), 12U);
  EXPECT_TRUE(plane.contains(Index<2>{0, 8}));
  EXPECT_FALSE(plane.contains(Index<2>{1, 8}));

  const RankSpace<3> volume{Index<3>{3, -1, 9}, Extent<3>{2, 3, 4}};
  EXPECT_EQ(volume.size(), 24U);
  EXPECT_TRUE(volume.contains(Index<3>{4, 1, 12}));
  EXPECT_FALSE(volume.contains(Index<3>{4, 2, 12}));
}

TEST(test_nd_layout, axis_zero_is_contiguous_and_round_trips_nonzero_origin) {
  const RankSpace<3> space{Index<3>{-3, 10, 7}, Extent<3>{4, 2, 3}};

  EXPECT_EQ(space.linear_rank(Index<3>{-3, 10, 7}), 0U);
  EXPECT_EQ(space.linear_rank(Index<3>{0, 10, 7}), 3U);
  EXPECT_EQ(space.linear_rank(Index<3>{-3, 11, 7}), 4U);
  EXPECT_EQ(space.linear_rank(Index<3>{-3, 10, 8}), 8U);

  for (std::size_t rank = 0; rank < space.size(); ++rank)
    EXPECT_EQ(space.linear_rank(space.coordinate(rank)), rank);
}

TEST(test_nd_layout, empty_rank_spaces_are_valid_but_have_no_coordinates) {
  const RankSpace<2> empty{Index<2>{7, -3}, Extent<2>{0, 5}};
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.size(), 0U);
  EXPECT_FALSE(empty.contains(Index<2>{7, -3}));
  EXPECT_THROW((void)empty.linear_rank(Index<2>{7, -3}), std::out_of_range);
  EXPECT_THROW((void)empty.coordinate(0), std::out_of_range);
}

TEST(test_nd_layout, invalid_extents_coordinates_and_ranks_fail_deterministically) {
  EXPECT_THROW((void)(RankSpace<1>{Index<1>{0}, Extent<1>{-1}}), std::invalid_argument);
  EXPECT_THROW((void)(RankSpace<2>{Index<2>{0, 0}, Extent<2>{0, -1}}), std::invalid_argument);

  const RankSpace<2> space{Index<2>{4, -2}, Extent<2>{2, 3}};
  EXPECT_THROW((void)space.linear_rank(Index<2>{3, -2}), std::out_of_range);
  EXPECT_THROW((void)space.linear_rank(Index<2>{4, 1}), std::out_of_range);
  EXPECT_THROW((void)space.coordinate(space.size()), std::out_of_range);
}

TEST(test_nd_layout, coordinate_and_size_overflows_are_rejected_before_narrowing) {
  constexpr std::int64_t full_axis = std::int64_t{1} << 32;
  constexpr int min = std::numeric_limits<int>::min();

  EXPECT_THROW((void)(RankSpace<1>{Index<1>{0}, Extent<1>{full_axis}}), std::overflow_error);
  EXPECT_THROW((void)(RankSpace<3>{Index<3>{min, min, 0}, Extent<3>{full_axis, full_axis, 1}}),
               std::overflow_error);
}

TEST(test_nd_layout, rank_space_extreme_extent_checks_before_signed_addition) {
  constexpr int minimum = std::numeric_limits<int>::min();
  EXPECT_THROW(
      (void)(RankSpace<1>{Index<1>{minimum}, Extent<1>{std::numeric_limits<std::int64_t>::max()}}),
      std::overflow_error);
  const RankSpace<2> exact_boundary{Index<2>{minimum, 0}, Extent<2>{std::int64_t{1} << 32, 1}};
  EXPECT_EQ(exact_boundary.size(), std::size_t{1} << 32);
}

TEST(test_nd_layout, box_array_balances_negative_anisotropic_tiles_in_axis_zero_order) {
  const Box<1> line_domain{Index<1>{-5}, Index<1>{4}};
  const BoxArray<1> line = BoxArray<1>::from_domain(line_domain, Extent<1>{4});
  ASSERT_EQ(line.size(), 3U);
  const Box<1> first_line{Index<1>{-5}, Index<1>{-2}};
  const Box<1> second_line{Index<1>{-1}, Index<1>{1}};
  const Box<1> third_line{Index<1>{2}, Index<1>{4}};
  EXPECT_EQ(line[0], first_line);
  EXPECT_EQ(line[1], second_line);
  EXPECT_EQ(line[2], third_line);
  EXPECT_TRUE(line.tiles_exactly(line_domain, kTilingBudget));

  const Box<2> plane_domain{Index<2>{-3, 5}, Index<2>{4, 10}};
  const BoxArray<2> plane = BoxArray<2>::from_domain(plane_domain, Extent<2>{3, 4});
  ASSERT_EQ(plane.size(), 6U);
  const Box<2> first_plane{Index<2>{-3, 5}, Index<2>{-1, 7}};
  const Box<2> second_plane{Index<2>{0, 5}, Index<2>{2, 7}};
  const Box<2> fourth_plane{Index<2>{-3, 8}, Index<2>{-1, 10}};
  EXPECT_EQ(plane[0], first_plane);
  EXPECT_EQ(plane[1], second_plane);
  EXPECT_EQ(plane[3], fourth_plane);
  EXPECT_EQ(plane.bounding_box(), plane_domain);
  EXPECT_EQ(plane.exact_cell_count(), ExactCellCount::from_uint64(48));
  EXPECT_TRUE(plane.tiles_exactly(plane_domain, kTilingBudget));

  const Box<3> volume_domain{Index<3>{-2, 1, 4}, Index<3>{2, 3, 6}};
  const BoxArray<3> volume = BoxArray<3>::from_domain(volume_domain, Extent<3>{2, 2, 2});
  ASSERT_EQ(volume.size(), 12U);
  const Box<3> first_volume{Index<3>{-2, 1, 4}, Index<3>{-1, 2, 5}};
  const Box<3> second_volume{Index<3>{0, 1, 4}, Index<3>{1, 2, 5}};
  const Box<3> fourth_volume{Index<3>{-2, 3, 4}, Index<3>{-1, 3, 5}};
  EXPECT_EQ(volume[0], first_volume);
  EXPECT_EQ(volume[1], second_volume);
  EXPECT_EQ(volume[3], fourth_volume);
  EXPECT_EQ(volume.bounding_box(), volume_domain);
  EXPECT_EQ(volume.exact_cell_count(), ExactCellCount::from_uint64(45));
  EXPECT_TRUE(volume.tiles_exactly(volume_domain, kTilingBudget));
}

TEST(test_nd_layout, box_array_rejects_holes_overlaps_outside_and_empty_members) {
  const Box<1> line{Index<1>{0}, Index<1>{3}};
  EXPECT_FALSE(BoxArray<1>(std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{1}},
                                               Box<1>{Index<1>{3}, Index<1>{3}}})
                   .tiles_exactly(line, kTilingBudget));
  EXPECT_FALSE(BoxArray<1>(std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{2}},
                                               Box<1>{Index<1>{2}, Index<1>{3}}})
                   .tiles_exactly(line, kTilingBudget));
  EXPECT_FALSE(BoxArray<1>(std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{2}},
                                               Box<1>{Index<1>{3}, Index<1>{4}}})
                   .tiles_exactly(line, kTilingBudget));
  EXPECT_FALSE(BoxArray<1>(std::vector<Box<1>>{Box<1>{}}).tiles_exactly(line, kTilingBudget));

  const Box<2> plane{Index<2>{0, 0}, Index<2>{1, 1}};
  EXPECT_FALSE(BoxArray<2>(std::vector<Box<2>>{Box<2>{Index<2>{0, 0}, Index<2>{1, 0}},
                                               Box<2>{Index<2>{0, 0}, Index<2>{1, 1}}})
                   .tiles_exactly(plane, kTilingBudget));
  const Box<3> volume{Index<3>{0, 0, 0}, Index<3>{1, 1, 1}};
  EXPECT_FALSE(BoxArray<3>(std::vector<Box<3>>{Box<3>{Index<3>{0, 0, 0}, Index<3>{1, 1, 0}},
                                               Box<3>{Index<3>{0, 0, 1}, Index<3>{1, 1, 2}}})
                   .tiles_exactly(volume, kTilingBudget));

  EXPECT_TRUE(BoxArray<2>{}.tiles_exactly(Box<2>{}, kTilingBudget));
  EXPECT_FALSE(BoxArray<2>(std::vector<Box<2>>{Box<2>{}}).tiles_exactly(Box<2>{}, kTilingBudget));
}

TEST(test_nd_layout, box_array_handles_full_signed_spans_without_narrowing) {
  constexpr int minimum = std::numeric_limits<int>::min();
  constexpr int maximum = std::numeric_limits<int>::max();
  const Box<3> full{Index<3>{minimum, minimum, minimum}, Index<3>{maximum, maximum, maximum}};
  const BoxArray<3> single_full(std::vector<Box<3>>{full});

  EXPECT_TRUE(single_full.tiles_exactly(full, kTilingBudget));
  EXPECT_EQ(single_full.exact_cell_count(), ExactCellCount::power_of_two(96));
  EXPECT_THROW((void)BoxArray<3>::from_domain(full, Extent<3>{1, 1, 1}), std::length_error);
  EXPECT_THROW((void)BoxArray<1>::from_domain(Box<1>{}, Extent<1>{0}), std::invalid_argument);
}

TEST(test_nd_layout, exact_cell_count_carries_across_portable_limbs) {
  ExactCellCount lower = ExactCellCount::power_of_two(31);
  EXPECT_TRUE(lower.add(ExactCellCount::power_of_two(31)));
  EXPECT_EQ(lower, ExactCellCount::power_of_two(32));

  ExactCellCount upper = ExactCellCount::power_of_two(63);
  EXPECT_TRUE(upper.add(ExactCellCount::power_of_two(63)));
  EXPECT_EQ(upper, ExactCellCount::power_of_two(64));
}

TEST(test_nd_layout, box_hash_uses_structural_negative_anisotropic_bins) {
  const BinCoordinate<2> left{{-1, 2}};
  const BinCoordinate<2> same_left{{-1, 2}};
  const BinCoordinate<2> transposed{{2, -1}};
  std::unordered_map<BinCoordinate<2>, int, BinCoordinateHash<2>> structural;
  structural.emplace(left, 3);
  structural.emplace(transposed, 7);
  EXPECT_EQ(structural.size(), 2U);
  EXPECT_EQ(structural.at(same_left), 3);
  EXPECT_EQ(structural.at(transposed), 7);

  const BoxArray<1> line(
      std::vector<Box<1>>{Box<1>{Index<1>{-7}, Index<1>{-3}}, Box<1>{Index<1>{-2}, Index<1>{2}}});
  const BoxHash<1> line_hash(line, Extent<1>{3}, kHashBudget);
  EXPECT_EQ(line_hash.query(Box<1>{Index<1>{-4}, Index<1>{-1}}), (std::vector<std::size_t>{0, 1}));

  const BoxArray<2> plane(std::vector<Box<2>>{Box<2>{Index<2>{-7, -3}, Index<2>{-4, 1}},
                                              Box<2>{Index<2>{-3, -2}, Index<2>{1, 3}},
                                              Box<2>{Index<2>{5, -4}, Index<2>{7, -1}}});
  const BoxHash<2> plane_hash(plane, Extent<2>{3, 2}, kHashBudget);
  EXPECT_EQ(plane_hash.query(Box<2>{Index<2>{-5, -1}, Index<2>{0, 2}}),
            (std::vector<std::size_t>{0, 1}));
  EXPECT_TRUE(plane_hash.query(Box<2>{}).empty());
  EXPECT_EQ(suggest_bin(plane), (Extent<2>{5, 6}));

  const BoxArray<3> volume(std::vector<Box<3>>{Box<3>{Index<3>{-3, -2, -1}, Index<3>{-1, 0, 1}},
                                               Box<3>{Index<3>{0, -1, 0}, Index<3>{2, 1, 2}}});
  const BoxHash<3> volume_hash(volume, Extent<3>{2, 3, 2}, kHashBudget);
  EXPECT_EQ(volume_hash.query(Box<3>{Index<3>{-1, -1, 0}, Index<3>{0, 0, 1}}),
            (std::vector<std::size_t>{0, 1}));
}

TEST(test_nd_layout, box_hash_has_no_omissions_against_bruteforce_intersections) {
  const BoxArray<2> boxes(std::vector<Box<2>>{
      Box<2>{Index<2>{-7, -3}, Index<2>{-4, 1}}, Box<2>{Index<2>{-3, -2}, Index<2>{1, 3}},
      Box<2>{Index<2>{5, -4}, Index<2>{7, -1}}, Box<2>{Index<2>{0, 4}, Index<2>{2, 5}}});
  const BoxHash<2> hash(boxes, Extent<2>{3, 2}, kHashBudget);
  expect_hash_superset(boxes, hash,
                       std::vector<Box<2>>{Box<2>{Index<2>{-8, -4}, Index<2>{-6, -2}},
                                           Box<2>{Index<2>{-5, -1}, Index<2>{0, 2}},
                                           Box<2>{Index<2>{1, 2}, Index<2>{6, 5}},
                                           Box<2>{Index<2>{8, 8}, Index<2>{9, 9}}});
}

TEST(test_nd_layout, box_hash_refuses_invalid_and_unbounded_enumerations) {
  const BoxArray<2> small(std::vector<Box<2>>{Box<2>{Index<2>{0, 0}, Index<2>{1, 1}}});
  EXPECT_THROW((void)(BoxHash<2>{small, Extent<2>{0, 1}, kHashBudget}), std::invalid_argument);

  constexpr int minimum = std::numeric_limits<int>::min();
  constexpr int maximum = std::numeric_limits<int>::max();
  const Box<3> full{Index<3>{minimum, minimum, minimum}, Index<3>{maximum, maximum, maximum}};
  const BoxArray<3> full_layout(std::vector<Box<3>>{full});
  EXPECT_THROW((void)(BoxHash<3>{full_layout, Extent<3>{1, 1, 1}, kHashBudget}), std::length_error);

  const BoxArray<3> one_cell(std::vector<Box<3>>{Box<3>{Index<3>{0, 0, 0}, Index<3>{0, 0, 0}}});
  const BoxHash<3> one_cell_hash(one_cell, Extent<3>{1, 1, 1}, kHashBudget);
  EXPECT_THROW((void)one_cell_hash.query(full), std::length_error);
}

TEST(test_nd_layout, box_array_tiling_requires_explicit_bounded_work) {
  const Box<1> domain{Index<1>{0}, Index<1>{3}};
  const BoxArray<1> boxes = BoxArray<1>::from_domain(domain, Extent<1>{1});
  EXPECT_THROW((void)boxes.tiles_exactly(domain, BoxArrayValidationBudget{3, 6}),
               std::length_error);
  EXPECT_THROW((void)boxes.tiles_exactly(domain, BoxArrayValidationBudget{4, 5}),
               std::length_error);
  EXPECT_TRUE(boxes.tiles_exactly(domain, BoxArrayValidationBudget{4, 6}));
}

TEST(test_nd_layout, box_hash_budgets_are_explicit_and_fail_before_work) {
  const BoxArray<1> one_bin(std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{1}}});
  const BoxHashBudget exact{1, 1, 1};
  const BoxHash<1> exact_hash(one_bin, Extent<1>{2}, exact);
  EXPECT_EQ(exact_hash.query(Box<1>{Index<1>{0}, Index<1>{1}}), (std::vector<std::size_t>{0}));

  const BoxArray<1> two_bins(std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{3}}});
  EXPECT_THROW((void)(BoxHash<1>{two_bins, Extent<1>{2}, BoxHashBudget{1, 2, 2}}),
               std::length_error);
  const BoxHash<1> query_limited(two_bins, Extent<1>{2}, BoxHashBudget{2, 1, 2});
  EXPECT_THROW((void)query_limited.query(Box<1>{Index<1>{0}, Index<1>{3}}), std::length_error);

  const BoxArray<1> same_bin(
      std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{0}}, Box<1>{Index<1>{3}, Index<1>{3}}});
  const BoxHash<1> candidate_limited(same_bin, Extent<1>{4}, BoxHashBudget{2, 1, 1});
  EXPECT_THROW((void)candidate_limited.query(Box<1>{Index<1>{0}, Index<1>{0}}), std::length_error);
}

TEST(test_nd_layout, hash_false_positives_are_filtered_at_the_exact_intersection_boundary) {
  const BoxArray<1> boxes(
      std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{0}}, Box<1>{Index<1>{3}, Index<1>{3}}});
  const BoxHash<1> hash(boxes, Extent<1>{4}, BoxHashBudget{2, 1, 2});
  const Box<1> query{Index<1>{0}, Index<1>{0}};
  const std::vector<std::size_t> candidates = hash.query(query);
  ASSERT_EQ(candidates, (std::vector<std::size_t>{0, 1}));
  std::vector<std::size_t> exact;
  for (const std::size_t index : candidates)
    if (!query.intersect(boxes[index]).empty())
      exact.push_back(index);
  EXPECT_EQ(exact, (std::vector<std::size_t>{0}));
}
