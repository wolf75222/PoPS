#include <gtest/gtest.h>

#include <pops/mesh/boundary/nd_boundary_schedule.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

using namespace pops;

TEST(test_nd_boundary_schedule, faces_and_complete_topology_are_ranked_and_canonical) {
  EXPECT_EQ((Face<1>{0, BoundarySide::lower}.ordinal()), 0);
  EXPECT_EQ((Face<3>{2, BoundarySide::upper}.ordinal()), 5);
  EXPECT_EQ((Face<2>{1, BoundarySide::lower}.outward_sign()), -1);
  EXPECT_EQ((Face<2>{1, BoundarySide::lower}.opposite()), (Face<2>{1, BoundarySide::upper}));
  EXPECT_THROW((Face<2>{2, BoundarySide::lower}), std::invalid_argument);

  const BoundaryTopology<3> physical;
  static_assert(BoundaryTopology<3>::face_count == 6);
  ASSERT_EQ(physical.faces().size(), 6U);
  for (std::size_t ordinal = 0; ordinal < physical.faces().size(); ++ordinal) {
    EXPECT_EQ(physical.faces()[ordinal].face.ordinal(), static_cast<int>(ordinal));
    EXPECT_EQ(physical.faces()[ordinal].kind, BoundaryFaceKind::physical);
  }

  const auto periodic = BoundaryTopology<3>::axis_periodic({true, false, true});
  EXPECT_EQ(periodic.periodic_pair_count(), 2U);
  EXPECT_EQ(periodic.partner(Face<3>{0, BoundarySide::lower}), (Face<3>{0, BoundarySide::upper}));
  EXPECT_TRUE(periodic.is_physical(Face<3>{1, BoundarySide::upper}));
  EXPECT_THROW((void)periodic.partner(Face<3>{1, BoundarySide::lower}), std::invalid_argument);
}

TEST(test_nd_boundary_schedule, topology_refuses_ambiguous_or_non_translation_pairs) {
  EXPECT_THROW(
      (PeriodicFacePair<2>{Face<2>{0, BoundarySide::lower}, Face<2>{1, BoundarySide::upper}}),
      std::invalid_argument);
  EXPECT_THROW(
      (PeriodicFacePair<2>{Face<2>{0, BoundarySide::lower}, Face<2>{0, BoundarySide::lower}}),
      std::invalid_argument);

  const PeriodicFacePair<2> x_pair{Face<2>{0, BoundarySide::upper},
                                   Face<2>{0, BoundarySide::lower}};
  EXPECT_EQ(x_pair.first, (Face<2>{0, BoundarySide::lower}));
  EXPECT_EQ(x_pair.second, (Face<2>{0, BoundarySide::upper}));
  const std::array<PeriodicFacePair<2>, 2> conflicts{x_pair, x_pair};
  EXPECT_THROW((void)BoundaryTopology<2>{conflicts}, std::invalid_argument);
}

TEST(test_nd_boundary_schedule, physical_regions_are_explicit_faces_edges_and_corners) {
  const auto line = prepare_boundary_schedule(Box<1>{Index<1>{0}, Index<1>{3}}, Extent<1>{1},
                                              BoundaryTopology<1>{}, BoundaryScheduleBudget{2});
  ASSERT_EQ(line.size(), 2U);
  EXPECT_EQ(line.entries()[0].region.kind(), BoundaryRegionKind::face);
  EXPECT_EQ(line.entries()[1].region.kind(), BoundaryRegionKind::face);

  const auto plane =
      prepare_boundary_schedule(Box<2>{Index<2>{0, 0}, Index<2>{3, 4}}, Extent<2>{1, 1},
                                BoundaryTopology<2>{}, BoundaryScheduleBudget{8});
  ASSERT_EQ(plane.size(), 8U);
  std::size_t plane_faces = 0;
  std::size_t plane_corners = 0;
  for (const BoundaryRegionPlan<2>& entry : plane.entries()) {
    plane_faces += entry.region.kind() == BoundaryRegionKind::face ? 1U : 0U;
    plane_corners += entry.region.kind() == BoundaryRegionKind::corner ? 1U : 0U;
    EXPECT_TRUE(entry.has_physical());
    EXPECT_FALSE(entry.has_periodic());
  }
  EXPECT_EQ(plane_faces, 4U);
  EXPECT_EQ(plane_corners, 4U);
  EXPECT_EQ(plane.entries()[0].region.ordinal(), 1U);
  EXPECT_EQ(plane.entries()[1].region.ordinal(), 2U);
  EXPECT_EQ(plane.entries()[2].region.ordinal(), 3U);
  EXPECT_EQ(plane.entries()[3].region.ordinal(), 4U);

  const auto volume =
      prepare_boundary_schedule(Box<3>{Index<3>{0, 0, 0}, Index<3>{1, 1, 1}}, Extent<3>{1, 1, 1},
                                BoundaryTopology<3>{}, BoundaryScheduleBudget{26});
  ASSERT_EQ(volume.size(), 26U);
  std::array<std::size_t, 3> kind_counts{};
  for (const BoundaryRegionPlan<3>& entry : volume.entries()) {
    if (entry.region.kind() == BoundaryRegionKind::face)
      ++kind_counts[0];
    else if (entry.region.kind() == BoundaryRegionKind::edge)
      ++kind_counts[1];
    else
      ++kind_counts[2];
  }
  EXPECT_EQ(kind_counts, (std::array<std::size_t, 3>{6, 12, 8}));
}

TEST(test_nd_boundary_schedule, periodic_corner_composition_is_deterministic_and_additive) {
  const Box<2> domain{Index<2>{0, 10}, Index<2>{3, 12}};
  const auto topology = BoundaryTopology<2>::axis_periodic({true, true});
  const auto schedule =
      prepare_boundary_schedule(domain, Extent<2>{1, 1}, topology, BoundaryScheduleBudget{8});
  ASSERT_EQ(schedule.size(), 8U);
  const BoundaryRegionPlan<2>& lower_x_upper_y = schedule.entries()[6];
  EXPECT_EQ(lower_x_upper_y.region.ordinal(), 7U);
  EXPECT_EQ(lower_x_upper_y.region.kind(), BoundaryRegionKind::corner);
  EXPECT_EQ(lower_x_upper_y.destination, (Box<2>{Index<2>{-1, 13}, Index<2>{-1, 13}}));
  EXPECT_EQ(lower_x_upper_y.operation_count, 2);
  EXPECT_EQ(lower_x_upper_y.operations[0].face, (Face<2>{0, BoundarySide::lower}));
  EXPECT_EQ(lower_x_upper_y.operations[1].face, (Face<2>{1, BoundarySide::upper}));
  EXPECT_EQ(lower_x_upper_y.source_from_destination_shift, (Index<2>{4, -3}));
  EXPECT_TRUE(lower_x_upper_y.has_periodic());
  EXPECT_FALSE(lower_x_upper_y.has_physical());

  const auto mixed = prepare_boundary_schedule(domain, Extent<2>{1, 1},
                                               BoundaryTopology<2>::axis_periodic({true, false}),
                                               BoundaryScheduleBudget{8});
  const BoundaryRegionPlan<2>& mixed_corner = mixed.entries()[3];
  EXPECT_TRUE(mixed_corner.has_periodic());
  EXPECT_TRUE(mixed_corner.has_physical());
  EXPECT_EQ(mixed_corner.source_from_destination_shift, (Index<2>{4, 0}));
}

TEST(test_nd_boundary_schedule, deep_periodic_ghosts_are_partitioned_into_exact_wraps) {
  const auto schedule = prepare_boundary_schedule(Box<1>{Index<1>{0}, Index<1>{1}}, Extent<1>{5},
                                                  BoundaryTopology<1>::axis_periodic({true}),
                                                  BoundaryScheduleBudget{6});
  ASSERT_EQ(schedule.size(), 6U);
  EXPECT_EQ(schedule.entries()[0].destination, (Box<1>{Index<1>{-2}, Index<1>{-1}}));
  EXPECT_EQ(schedule.entries()[0].source_from_destination_shift, (Index<1>{2}));
  EXPECT_EQ(schedule.entries()[2].destination, (Box<1>{Index<1>{-4}, Index<1>{-3}}));
  EXPECT_EQ(schedule.entries()[2].source_from_destination_shift, (Index<1>{4}));
  EXPECT_EQ(schedule.entries()[4].destination, (Box<1>{Index<1>{-5}, Index<1>{-5}}));
  EXPECT_EQ(schedule.entries()[4].source_from_destination_shift, (Index<1>{6}));
  EXPECT_EQ(schedule.entries()[5].destination, (Box<1>{Index<1>{6}, Index<1>{6}}));
  EXPECT_EQ(schedule.entries()[5].source_from_destination_shift, (Index<1>{-6}));
}

TEST(test_nd_boundary_schedule, composition_and_planning_fail_closed_on_conflicts_and_limits) {
  std::array<BoundaryOperation<2>, 2> unordered{
      BoundaryOperation<2>{Face<2>{1, BoundarySide::lower}, BoundaryFaceKind::physical, {}},
      BoundaryOperation<2>{Face<2>{0, BoundarySide::upper}, BoundaryFaceKind::periodic,
                           Index<2>{-4, 0}}};
  const auto canonical =
      compose_boundary_region_plan(Box<2>{Index<2>{4, -1}, Index<2>{4, -1}}, unordered, 2);
  EXPECT_EQ(canonical.operations[0].face, (Face<2>{0, BoundarySide::upper}));
  EXPECT_EQ(canonical.operations[1].face, (Face<2>{1, BoundarySide::lower}));

  std::array<BoundaryOperation<2>, 2> duplicate_axis{
      BoundaryOperation<2>{Face<2>{0, BoundarySide::lower}, BoundaryFaceKind::physical, {}},
      BoundaryOperation<2>{Face<2>{0, BoundarySide::upper}, BoundaryFaceKind::physical, {}}};
  EXPECT_THROW(
      (void)compose_boundary_region_plan(Box<2>{Index<2>{0, 0}, Index<2>{0, 0}}, duplicate_axis, 2),
      std::invalid_argument);

  std::array<BoundaryOperation<2>, 2> tangential{
      BoundaryOperation<2>{Face<2>{0, BoundarySide::lower}, BoundaryFaceKind::periodic,
                           Index<2>{4, 1}},
      {}};
  EXPECT_THROW(
      (void)compose_boundary_region_plan(Box<2>{Index<2>{0, 0}, Index<2>{0, 0}}, tangential, 1),
      std::invalid_argument);

  EXPECT_THROW((void)prepare_boundary_schedule(Box<3>{Index<3>{0, 0, 0}, Index<3>{1, 1, 1}},
                                               Extent<3>{1, 1, 1}, BoundaryTopology<3>{},
                                               BoundaryScheduleBudget{25}),
               std::length_error);
  EXPECT_THROW(
      (void)prepare_boundary_schedule(Box<1>{Index<1>{std::numeric_limits<int>::min()},
                                             Index<1>{std::numeric_limits<int>::max()}},
                                      Extent<1>{1}, BoundaryTopology<1>::axis_periodic({true}),
                                      BoundaryScheduleBudget{2}),
      std::overflow_error);

  static_assert(std::is_trivially_copyable_v<BoundaryRegionPlan<1>>);
  static_assert(std::is_trivially_copyable_v<BoundaryRegionPlan<2>>);
  static_assert(std::is_trivially_copyable_v<BoundaryRegionPlan<3>>);
}
