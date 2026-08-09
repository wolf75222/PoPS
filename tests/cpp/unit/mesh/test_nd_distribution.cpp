#include <gtest/gtest.h>

#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using pops::Box;
using pops::Extent;
using pops::Index;
using pops::MultiFab;
using pops::mesh::BoxArray;
using pops::mesh::Distribution;
using pops::mesh::DistributionMode;
using pops::mesh::RankSpace;

TEST(test_nd_distribution, partitioned_ownership_is_ordered_and_rank_coordinates_round_trip) {
  const BoxArray<1> line(std::vector<Box<1>>{Box<1>{Index<1>{-3}, Index<1>{-2}},
                                             Box<1>{Index<1>{-1}, Index<1>{0}},
                                             Box<1>{Index<1>{1}, Index<1>{3}}});
  const RankSpace<1> line_ranks{Index<1>{-4}, Extent<1>{3}};
  const auto line_distribution =
      Distribution<1>::partitioned(line, line_ranks, {Index<1>{-4}, Index<1>{-2}, Index<1>{-4}});
  EXPECT_EQ(line_distribution.owner(0), Index<1>{-4});
  EXPECT_EQ(line_distribution.owner(1), Index<1>{-2});
  EXPECT_EQ(line_distribution.local_box_indices(Index<1>{-4}), (std::vector<std::size_t>{0, 2}));
  EXPECT_TRUE(line_distribution.is_local(2, Index<1>{-4}));
  EXPECT_FALSE(line_distribution.is_local(1, Index<1>{-4}));

  const BoxArray<2> plane(std::vector<Box<2>>{Box<2>{Index<2>{0, 0}, Index<2>{0, 0}},
                                              Box<2>{Index<2>{1, 0}, Index<2>{1, 0}}});
  const RankSpace<2> plane_ranks{Index<2>{-1, 7}, Extent<2>{2, 3}};
  const auto plane_distribution =
      Distribution<2>::partitioned(plane, plane_ranks, {Index<2>{-1, 7}, Index<2>{0, 9}});
  EXPECT_EQ(plane_distribution.owner(1), (Index<2>{0, 9}));

  const BoxArray<3> volume(std::vector<Box<3>>{Box<3>{Index<3>{0, 0, 0}, Index<3>{0, 0, 0}},
                                               Box<3>{Index<3>{1, 0, 0}, Index<3>{1, 0, 0}}});
  const RankSpace<3> volume_ranks{Index<3>{3, -2, 5}, Extent<3>{2, 1, 3}};
  const auto volume_distribution =
      Distribution<3>::partitioned(volume, volume_ranks, {Index<3>{4, -2, 7}, Index<3>{3, -2, 5}});
  EXPECT_EQ(volume_distribution.local_box_indices(Index<3>{3, -2, 5}),
            (std::vector<std::size_t>{1}));

  EXPECT_TRUE(
      line_distribution ==
      Distribution<1>::partitioned(line, line_ranks, {Index<1>{-4}, Index<1>{-2}, Index<1>{-4}}));
  EXPECT_FALSE(
      line_distribution ==
      Distribution<1>::partitioned(line, line_ranks, {Index<1>{-2}, Index<1>{-2}, Index<1>{-4}}));
}

TEST(test_nd_distribution, replicated_layouts_store_no_fake_owner_and_are_local_everywhere) {
  const BoxArray<2> boxes =
      BoxArray<2>::from_domain(Box<2>{Index<2>{-3, 4}, Index<2>{0, 7}}, Extent<2>{2, 2});
  const RankSpace<2> ranks{Index<2>{4, -3}, Extent<2>{2, 3}};
  const auto distribution = Distribution<2>::replicated(boxes, ranks);

  EXPECT_EQ(distribution.mode(), DistributionMode::replicated);
  EXPECT_TRUE(distribution.replicated());
  EXPECT_THROW((void)distribution.owner(0), std::logic_error);
  for (std::size_t global = 0; global < boxes.size(); ++global) {
    EXPECT_TRUE(distribution.is_local(global, Index<2>{4, -3}));
    EXPECT_TRUE(distribution.is_local(global, Index<2>{5, -1}));
  }
  EXPECT_EQ(distribution.local_box_indices(Index<2>{5, -2}),
            (std::vector<std::size_t>{0, 1, 2, 3}));
}

TEST(test_nd_distribution, distribution_rejects_invalid_counts_owners_rank_spaces_and_modes) {
  const BoxArray<1> boxes(
      std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{0}}, Box<1>{Index<1>{1}, Index<1>{1}}});
  const RankSpace<1> ranks{Index<1>{3}, Extent<1>{2}};
  EXPECT_THROW((void)Distribution<1>::partitioned(boxes, ranks, {Index<1>{3}}),
               std::invalid_argument);
  EXPECT_THROW((void)Distribution<1>::partitioned(boxes, ranks, {Index<1>{3}, Index<1>{5}}),
               std::out_of_range);
  EXPECT_THROW((void)Distribution<1>(boxes, ranks, DistributionMode::replicated, {Index<1>{3}}),
               std::invalid_argument);
  EXPECT_THROW((void)Distribution<1>(boxes, ranks, static_cast<DistributionMode>(77),
                                     {Index<1>{3}, Index<1>{4}}),
               std::invalid_argument);
  EXPECT_THROW((void)Distribution<1>::replicated(boxes, RankSpace<1>{Index<1>{0}, Extent<1>{0}}),
               std::invalid_argument);
  const auto distribution = Distribution<1>::partitioned(boxes, ranks, {Index<1>{3}, Index<1>{4}});
  EXPECT_THROW((void)distribution.is_local(2, Index<1>{3}), std::out_of_range);
  EXPECT_THROW((void)distribution.is_local(0, Index<1>{2}), std::out_of_range);
  EXPECT_THROW((void)distribution.local_box_indices(Index<1>{2}), std::out_of_range);
}

TEST(test_nd_distribution,
     multifab_allocates_only_ordered_partitioned_boxes_and_refuses_remote_access) {
  const BoxArray<2> boxes =
      BoxArray<2>::from_domain(Box<2>{Index<2>{-2, 3}, Index<2>{1, 6}}, Extent<2>{2, 2});
  const RankSpace<2> ranks{Index<2>{10, -2}, Extent<2>{2, 2}};
  const Index<2> first_rank{10, -2};
  const auto distribution = Distribution<2>::partitioned(
      boxes, ranks, {first_rank, Index<2>{11, -2}, first_rank, Index<2>{11, -1}});
  MultiFab<2> fields(boxes, distribution, first_rank, /*ncomp=*/2, Extent<2>{1, 2});

  EXPECT_EQ(fields.local_global_indices(), (std::vector<std::size_t>{0, 2}));
  EXPECT_EQ(fields.local_size(), 2U);
  EXPECT_TRUE(fields.contains_local(0));
  EXPECT_FALSE(fields.contains_local(1));
  EXPECT_EQ(fields.global_index(1), 2U);
  EXPECT_EQ(fields.local_index_of(2), 1U);
  EXPECT_EQ(fields.local_index_of(1), MultiFab<2>::not_local);
  EXPECT_EQ(fields.fab_global(0).ghosts(), (Extent<2>{1, 2}));
  EXPECT_EQ(fields.fab_global(0).size(), 48U);
  EXPECT_THROW((void)fields.fab_global(1), std::out_of_range);
  EXPECT_THROW((void)fields.local_index_of(4), std::out_of_range);
  EXPECT_THROW((void)MultiFab<2>(boxes, distribution, Index<2>{12, -2}, 1, Extent<2>{}),
               std::out_of_range);

  fields.fab_global(0).set_val(3.5);
  MultiFab<2> copy = fields;
  EXPECT_NE(copy.fab_global(0).storage().data(), fields.fab_global(0).storage().data());
  copy.fab_global(0).set_val(-2.0);
  auto source = fields.fab_global(0).create_host_mirror();
  auto copied = copy.fab_global(0).create_host_mirror();
  fields.fab_global(0).copy_to_host(source);
  copy.fab_global(0).copy_to_host(copied);
  EXPECT_DOUBLE_EQ(source(0), 3.5);
  EXPECT_DOUBLE_EQ(copied(0), -2.0);

  MultiFab<2> moved(std::move(fields));
  EXPECT_EQ(fields.local_size(), 0U);
  EXPECT_TRUE(fields.layout().empty());
  EXPECT_EQ(moved.local_global_indices(), (std::vector<std::size_t>{0, 2}));
}

TEST(test_nd_distribution,
     multifab_replicates_all_boxes_and_supports_empty_and_memory_space_instantiation) {
  const BoxArray<1> boxes(
      std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{1}}, Box<1>{Index<1>{2}, Index<1>{4}}});
  const RankSpace<1> ranks{Index<1>{-1}, Extent<1>{3}};
  const auto replicated = Distribution<1>::replicated(boxes, ranks);
  MultiFab<1> defaults(boxes, replicated, Index<1>{0}, /*ncomp=*/1, Extent<1>{1});
  MultiFab<1, Kokkos::HostSpace> hosts(boxes, replicated, Index<1>{1}, /*ncomp=*/1, Extent<1>{});
  EXPECT_EQ(defaults.local_global_indices(), (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(hosts.local_global_indices(), (std::vector<std::size_t>{0, 1}));
  static_assert(std::is_same_v<typename MultiFab<1>::fab_type::memory_space,
                               typename Kokkos::DefaultExecutionSpace::memory_space>);

  const BoxArray<3> empty{};
  const RankSpace<3> empty_layout_ranks{Index<3>{1, 2, 3}, Extent<3>{1, 1, 1}};
  const auto empty_distribution = Distribution<3>::partitioned(empty, empty_layout_ranks, {});
  MultiFab<3, Kokkos::HostSpace> empty_fields(empty, empty_distribution, Index<3>{1, 2, 3}, 1,
                                              Extent<3>{});
  EXPECT_EQ(empty_fields.local_size(), 0U);
  EXPECT_THROW((void)empty_fields.fab(0), std::out_of_range);
}

TEST(test_nd_distribution,
     multifab_authenticates_ordered_layout_identity_for_all_distribution_modes) {
  const BoxArray<1> layout(
      std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{1}}, Box<1>{Index<1>{2}, Index<1>{3}}});
  const BoxArray<1> reordered(std::vector<Box<1>>{layout[1], layout[0]});
  const BoxArray<1> different(
      std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{0}}, Box<1>{Index<1>{1}, Index<1>{3}}});
  const RankSpace<1> ranks{Index<1>{0}, Extent<1>{2}};
  const auto partitioned = Distribution<1>::partitioned(layout, ranks, {Index<1>{0}, Index<1>{1}});
  const auto replicated = Distribution<1>::replicated(layout, ranks);
  EXPECT_THROW((void)MultiFab<1>(reordered, partitioned, Index<1>{0}, 1, Extent<1>{1}),
               std::invalid_argument);
  EXPECT_THROW((void)MultiFab<1>(different, replicated, Index<1>{0}, 1, Extent<1>{1}),
               std::invalid_argument);
}

TEST(test_nd_distribution,
     multifab_assignment_and_nonempty_1d_3d_partitioned_layouts_remain_local) {
  const RankSpace<1> ranks1{Index<1>{3}, Extent<1>{2}};
  const BoxArray<1> line = BoxArray<1>::from_domain(Box<1>{Index<1>{-2}, Index<1>{3}}, {2});
  const auto dist1 =
      Distribution<1>::partitioned(line, ranks1, {Index<1>{3}, Index<1>{4}, Index<1>{3}});
  MultiFab<1> first(line, dist1, Index<1>{3}, 2, Extent<1>{2});
  MultiFab<1> assigned;
  assigned = first;
  EXPECT_EQ(assigned.local_global_indices(), (std::vector<std::size_t>{0, 2}));
  EXPECT_NE(assigned.fab(0).storage().data(), first.fab(0).storage().data());
  MultiFab<1> move_assigned;
  move_assigned = std::move(assigned);
  EXPECT_EQ(assigned.local_size(), 0U);
  EXPECT_EQ(move_assigned.fab(0).ghosts(), Extent<1>{2});

  const BoxArray<3> volume =
      BoxArray<3>::from_domain(Box<3>{Index<3>{-1, 2, 4}, Index<3>{2, 3, 5}}, Extent<3>{2, 1, 2});
  const RankSpace<3> ranks3{Index<3>{1, -1, 7}, Extent<3>{2, 1, 1}};
  std::vector<Index<3>> owners(volume.size(), Index<3>{2, -1, 7});
  owners[0] = Index<3>{1, -1, 7};
  const auto dist3 = Distribution<3>::partitioned(volume, ranks3, owners);
  MultiFab<3> three_dimensional(volume, dist3, Index<3>{1, -1, 7}, 1, Extent<3>{1, 2, 1});
  ASSERT_EQ(three_dimensional.local_global_indices(), (std::vector<std::size_t>{0}));
  EXPECT_EQ(three_dimensional.fab(0).ghosts(), (Extent<3>{1, 2, 1}));
  EXPECT_EQ(three_dimensional.fab(0).size(), 80U);
}
