#include <gtest/gtest.h>

#include <pops/mesh/nd_proof/local_neighbors.hpp>
#include <pops/mesh/nd_proof/periodicity.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

using namespace pops;
using namespace pops::mesh::nd_proof;

namespace {

constexpr BoxHashBudget kHashBudget{4096, 4096, 4096};
constexpr LocalNeighborWorkBudget kNeighborBudget{4096, 4096, {4096, 4096}, {4096, 4096}};

template <int Dim>
std::vector<LocalNeighborJob<Dim>> brute_translation_neighbors(
    const BoxArray<Dim>& boxes, const Box<Dim>& domain, const Extent<Dim>& ghosts,
    const PeriodicTopology<Dim>& topology) {
  std::vector<LocalNeighborJob<Dim>> result;
  const auto images =
      enumerate_axis_translation_images(domain, ghosts, topology, AxisTranslationImageBudget{4096});
  for (std::size_t destination = 0; destination < boxes.size(); ++destination) {
    const Box<Dim> grown = periodicity_detail::grow_box(boxes[destination], ghosts);
    for (const AxisTranslationImage<Dim>& image : images) {
      std::array<std::int64_t, Dim> source_from_destination{};
      for (int axis = 0; axis < Dim; ++axis)
        source_from_destination[axis] = -image.translation[axis];
      for (std::size_t source = 0; source < boxes.size(); ++source) {
        if (image.is_zero() && source == destination)
          continue;
        const Box<Dim> region = grown.intersect(image.apply(boxes[source]));
        if (!region.empty())
          result.push_back(
              LocalNeighborJob<Dim>{source, destination, region, source_from_destination});
      }
    }
  }
  return result;
}

template <int Dim>
const LocalNeighborJob<Dim>* find_job(const std::vector<LocalNeighborJob<Dim>>& jobs,
                                      std::size_t source, std::size_t destination,
                                      const std::array<std::int64_t, Dim>& translation) {
  for (const LocalNeighborJob<Dim>& job : jobs)
    if (job.source_box == source && job.destination_box == destination &&
        job.source_from_destination_translation == translation)
      return &job;
  return nullptr;
}

}  // namespace

TEST(test_nd_topology, faces_validate_and_topologies_canonicalize_identity) {
  EXPECT_EQ((Face<1>{0, Side::lower}.ordinal()), 0);
  EXPECT_EQ((Face<3>{2, Side::upper}.ordinal()), 5);
  EXPECT_THROW((Face<2>{2, Side::lower}), std::invalid_argument);

  const PeriodicIdentification<2> forward{Face<2>{0, Side::lower}, Face<2>{0, Side::upper}};
  const PeriodicIdentification<2> reverse{Face<2>{0, Side::upper}, Face<2>{0, Side::lower}};
  EXPECT_EQ((PeriodicTopology<2>{std::vector<PeriodicIdentification<2>>{forward}}),
            (PeriodicTopology<2>{std::vector<PeriodicIdentification<2>>{reverse}}));
  EXPECT_TRUE(
      PeriodicTopology<3>::axis_translations({true, false, true}).is_axis_translation_only());
  EXPECT_THROW(
      (PeriodicTopology<2>{std::vector<PeriodicIdentification<2>>{
          forward, PeriodicIdentification<2>{Face<2>{0, Side::upper}, Face<2>{1, Side::lower},
                                             SignedPermutation<2>{{1, 0}, {1, 1}}}}}),
      std::invalid_argument);
}

TEST(test_nd_topology, signed_permutations_invert_and_compose_in_all_ranks) {
  const SignedPermutation<1> one{{0}, {-1}};
  const SignedPermutation<2> two{{1, 0}, {-1, 1}};
  const SignedPermutation<3> three{{1, 2, 0}, {1, -1, 1}};
  EXPECT_TRUE(one.compose(one.inverse()).is_identity());
  EXPECT_TRUE(two.compose(two.inverse()).is_identity());
  EXPECT_TRUE(three.compose(three.inverse()).is_identity());
  EXPECT_THROW((SignedPermutation<2>{{0, 0}, {1, 1}}), std::invalid_argument);
  EXPECT_THROW((SignedPermutation<3>{{0, 1, 2}, {1, 0, 1}}), std::invalid_argument);
}

TEST(test_nd_topology, affine_identifications_are_exact_for_axis_and_permuted_faces) {
  const Box<2> axis_domain{Index<2>{-4, 10}, Index<2>{1, 13}};
  const PeriodicIdentification<2> axis_aligned{Face<2>{0, Side::lower}, Face<2>{0, Side::upper}};
  const AffineIndexTransform<2> axis_forward =
      axis_aligned.source_interior_to_target_exterior(axis_domain);
  EXPECT_EQ(axis_forward.apply(Index<2>{-4, 11}), (Index<2>{2, 11}));
  EXPECT_EQ(axis_forward.inverse().apply(Index<2>{2, 11}), (Index<2>{-4, 11}));

  const Box<3> compatible{Index<3>{-5, 10, -2}, Index<3>{-2, 13, 4}};
  const SignedPermutation<3> permutation{{1, 0, 2}, {1, -1, 1}};
  const PeriodicIdentification<3> mapped{Face<3>{0, Side::lower}, Face<3>{1, Side::upper},
                                         permutation};
  const AffineIndexTransform<3> mapped_forward =
      mapped.source_interior_to_target_exterior(compatible);
  EXPECT_EQ(mapped_forward.apply(Index<3>{-5, 10, -2}), (Index<3>{-2, 14, -2}));
  EXPECT_EQ(mapped.target_exterior_to_source_interior(compatible).apply(Index<3>{-2, 14, -2}),
            (Index<3>{-5, 10, -2}));
  EXPECT_EQ(mapped_forward.apply(Box<3>{Index<3>{-5, 10, -2}, Index<3>{-4, 11, 0}}),
            (Box<3>{Index<3>{-3, 14, -2}, Index<3>{-2, 15, 0}}));

  const Box<3> incompatible{Index<3>{-5, 10, -2}, Index<3>{-2, 14, 4}};
  EXPECT_THROW((void)mapped.source_interior_to_target_exterior(incompatible),
               std::invalid_argument);

  const PeriodicIdentification<1> upper_to_lower{Face<1>{0, Side::upper}, Face<1>{0, Side::lower}};
  EXPECT_EQ(upper_to_lower.source_interior_to_target_exterior(Box<1>{Index<1>{-2}, Index<1>{1}})
                .apply(Index<1>{1}),
            (Index<1>{-3}));
}

TEST(test_nd_topology, affine_and_translation_narrow_only_after_checked_int64_arithmetic) {
  const AffineIndexTransform<1> overflowing{SignedPermutation<1>{},
                                            {std::numeric_limits<std::int64_t>::max()}};
  EXPECT_THROW((void)overflowing.apply(Index<1>{1}), std::overflow_error);
  const AxisTranslationImage<1> image{{1}, {std::numeric_limits<std::int64_t>::max()}};
  EXPECT_THROW((void)image.apply(Index<1>{1}), std::overflow_error);
  EXPECT_THROW((void)image.apply(Box<1>{Index<1>{0}, Index<1>{1}}), std::overflow_error);

  const AffineIndexTransform<1> reflected_minimum{SignedPermutation<1>{{0}, {-1}},
                                                  {std::numeric_limits<std::int64_t>::min()}};
  EXPECT_EQ(reflected_minimum.inverse().target_offsets()[0],
            std::numeric_limits<std::int64_t>::min());
}

TEST(test_nd_topology, axis_translation_images_cover_deep_halos_with_explicit_order_and_budget) {
  const Box<1> line{Index<1>{0}, Index<1>{3}};
  const auto topology = PeriodicTopology<1>::axis_translations({true});
  const auto images = enumerate_axis_translation_images(line, Extent<1>{5}, topology,
                                                        AxisTranslationImageBudget{5});
  ASSERT_EQ(images.size(), 5U);
  EXPECT_EQ(images[0].translation, (std::array<std::int64_t, 1>{0}));
  EXPECT_EQ(images[1].translation, (std::array<std::int64_t, 1>{-4}));
  EXPECT_EQ(images[2].translation, (std::array<std::int64_t, 1>{4}));
  EXPECT_EQ(images[3].translation, (std::array<std::int64_t, 1>{-8}));
  EXPECT_EQ(images[4].translation, (std::array<std::int64_t, 1>{8}));
  EXPECT_THROW((void)enumerate_axis_translation_images(line, Extent<1>{5}, topology,
                                                       AxisTranslationImageBudget{4}),
               std::length_error);

  const Box<2> plane{Index<2>{0, 5}, Index<2>{1, 7}};
  const auto only_x = PeriodicTopology<2>::axis_translations({true, false});
  const auto anisotropic = enumerate_axis_translation_images(plane, Extent<2>{3, 100}, only_x,
                                                             AxisTranslationImageBudget{5});
  ASSERT_EQ(anisotropic.size(), 5U);
  for (const AxisTranslationImage<2>& candidate : anisotropic)
    EXPECT_EQ(candidate.translation[1], 0);
}

TEST(test_nd_topology, axis_translation_image_corners_are_axis_zero_fastest_and_reject_mapped) {
  const Box<2> plane{Index<2>{0, 0}, Index<2>{1, 2}};
  const auto topology = PeriodicTopology<2>::axis_translations({true, true});
  const auto images = enumerate_axis_translation_images(plane, Extent<2>{1, 1}, topology,
                                                        AxisTranslationImageBudget{9});
  ASSERT_EQ(images.size(), 9U);
  EXPECT_EQ(images[0].multiples, (std::array<std::int64_t, 2>{0, 0}));
  EXPECT_EQ(images[1].multiples, (std::array<std::int64_t, 2>{-1, 0}));
  EXPECT_EQ(images[2].multiples, (std::array<std::int64_t, 2>{1, 0}));
  EXPECT_EQ(images[3].multiples, (std::array<std::int64_t, 2>{0, -1}));
  EXPECT_EQ(images[8].multiples, (std::array<std::int64_t, 2>{1, 1}));

  const Box<3> volume{Index<3>{0, 0, 0}, Index<3>{0, 0, 0}};
  EXPECT_EQ(
      enumerate_axis_translation_images(volume, Extent<3>{1, 1, 1},
                                        PeriodicTopology<3>::axis_translations({true, true, true}),
                                        AxisTranslationImageBudget{27})
          .size(),
      27U);

  const PeriodicTopology<2> mapped{std::vector<PeriodicIdentification<2>>{PeriodicIdentification<2>{
      Face<2>{0, Side::lower}, Face<2>{1, Side::upper}, SignedPermutation<2>{{1, 0}, {1, -1}}}}};
  EXPECT_THROW((void)enumerate_axis_translation_images(plane, Extent<2>{1, 1}, mapped,
                                                       AxisTranslationImageBudget{9}),
               std::invalid_argument);
}

TEST(test_nd_topology, local_neighbors_enumerate_internal_and_periodic_self_seams_in_1d) {
  const Box<1> domain{Index<1>{0}, Index<1>{3}};
  const BoxArray<1> split = BoxArray<1>::from_domain(domain, Extent<1>{2});
  const auto internal =
      enumerate_local_translation_neighbors(split, domain, Extent<1>{1}, PeriodicTopology<1>{},
                                            Extent<1>{2}, kHashBudget, kNeighborBudget);
  EXPECT_EQ(internal,
            brute_translation_neighbors(split, domain, Extent<1>{1}, PeriodicTopology<1>{}));
  ASSERT_EQ(internal.size(), 2U);
  EXPECT_EQ(internal[0].source_box, 1U);
  EXPECT_EQ(internal[0].destination_box, 0U);
  EXPECT_EQ(internal[0].destination_region, (Box<1>{Index<1>{2}, Index<1>{2}}));
  EXPECT_THROW((void)enumerate_local_translation_neighbors(
                   split, domain, Extent<1>{1}, PeriodicTopology<1>{}, Extent<1>{2}, kHashBudget,
                   LocalNeighborWorkBudget{4096, 4096, {4096, 4096}, {1, 4096}}),
               std::length_error);

  const Box<1> small_domain{Index<1>{0}, Index<1>{1}};
  const BoxArray<1> one_box = BoxArray<1>::from_domain(small_domain, Extent<1>{2});
  const auto periodic = enumerate_local_translation_neighbors(
      one_box, small_domain, Extent<1>{3}, PeriodicTopology<1>::axis_translations({true}),
      Extent<1>{2}, kHashBudget, kNeighborBudget);
  EXPECT_EQ(periodic, brute_translation_neighbors(one_box, small_domain, Extent<1>{3},
                                                  PeriodicTopology<1>::axis_translations({true})));
  ASSERT_EQ(periodic.size(), 4U);
  EXPECT_EQ(periodic[0].source_box, 0U);
  EXPECT_EQ(periodic[0].source_from_destination_translation, (std::array<std::int64_t, 1>{2}));
  EXPECT_EQ(periodic[0].destination_region, (Box<1>{Index<1>{-2}, Index<1>{-1}}));
}

TEST(test_nd_topology, local_neighbors_are_exact_unique_and_ordered_for_2d_corners) {
  const Box<2> domain{Index<2>{0, 0}, Index<2>{3, 3}};
  const BoxArray<2> boxes = BoxArray<2>::from_domain(domain, Extent<2>{2, 2});
  const auto topology = PeriodicTopology<2>::axis_translations({true, true});
  const auto jobs = enumerate_local_translation_neighbors(
      boxes, domain, Extent<2>{1, 1}, topology, Extent<2>{2, 2}, kHashBudget, kNeighborBudget);
  const auto brute = brute_translation_neighbors(boxes, domain, Extent<2>{1, 1}, topology);
  EXPECT_EQ(jobs, brute);
  const auto coarse_jobs = enumerate_local_translation_neighbors(
      boxes, domain, Extent<2>{1, 1}, topology, Extent<2>{4, 4}, kHashBudget, kNeighborBudget);
  EXPECT_EQ(coarse_jobs,
            brute);  // Coarse bins produce false positives; exact intersections filter them.
  const LocalNeighborJob<2>* corner = find_job(jobs, 3, 0, {4, 4});
  ASSERT_NE(corner, nullptr);
  EXPECT_EQ(corner->destination_region, (Box<2>{Index<2>{-1, -1}, Index<2>{-1, -1}}));

  EXPECT_THROW((void)enumerate_local_translation_neighbors(
                   boxes, domain, Extent<2>{1, 1}, topology, Extent<2>{2, 2}, kHashBudget,
                   LocalNeighborWorkBudget{9, 1, {4096, 4096}, {4096, 4096}}),
               std::length_error);
}

TEST(test_nd_topology, topology_canonical_reverse_and_affine_round_trips_are_exact) {
  const Box<1> line{Index<1>{-2}, Index<1>{3}};
  const PeriodicIdentification<1> forward1{Face<1>{0, Side::lower}, Face<1>{0, Side::upper}};
  const PeriodicIdentification<1> reverse1{Face<1>{0, Side::upper}, Face<1>{0, Side::lower}};
  EXPECT_EQ(PeriodicTopology<1>{{forward1}}, PeriodicTopology<1>{{reverse1}});
  const auto map1 = forward1.source_interior_to_target_exterior(line);
  const Box<1> box1{Index<1>{-2}, Index<1>{0}};
  EXPECT_EQ(map1.inverse().apply(map1.apply(box1)), box1);
  EXPECT_EQ(map1.inverse().apply(map1.apply(Index<1>{-2})), (Index<1>{-2}));

  const Box<2> plane{Index<2>{0, 0}, Index<2>{3, 3}};
  const SignedPermutation<2> reflected2{{1, 0}, {1, -1}};
  const PeriodicIdentification<2> forward2{Face<2>{0, Side::lower}, Face<2>{1, Side::upper},
                                           reflected2};
  const PeriodicIdentification<2> reverse2{Face<2>{1, Side::upper}, Face<2>{0, Side::lower},
                                           reflected2.inverse()};
  EXPECT_EQ(PeriodicTopology<2>{{forward2}}, PeriodicTopology<2>{{reverse2}});
  const auto map2 = forward2.source_interior_to_target_exterior(plane);
  const Box<2> box2{Index<2>{0, 1}, Index<2>{2, 3}};
  EXPECT_EQ(map2.inverse().apply(map2.apply(box2)), box2);
  EXPECT_EQ(map2.inverse().apply(map2.apply(Index<2>{0, 3})), (Index<2>{0, 3}));

  const Box<3> volume{Index<3>{-1, 2, 4}, Index<3>{2, 5, 7}};
  const SignedPermutation<3> reflected3{{1, 2, 0}, {1, -1, 1}};
  const PeriodicIdentification<3> forward3{Face<3>{0, Side::lower}, Face<3>{1, Side::upper},
                                           reflected3};
  const PeriodicIdentification<3> reverse3{Face<3>{1, Side::upper}, Face<3>{0, Side::lower},
                                           reflected3.inverse()};
  EXPECT_EQ(PeriodicTopology<3>{{forward3}}, PeriodicTopology<3>{{reverse3}});
  const auto map3 = forward3.source_interior_to_target_exterior(volume);
  const Box<3> box3{Index<3>{-1, 3, 4}, Index<3>{1, 5, 6}};
  EXPECT_EQ(map3.inverse().apply(map3.apply(box3)), box3);
  EXPECT_EQ(map3.inverse().apply(map3.apply(Index<3>{-1, 5, 6})), (Index<3>{-1, 5, 6}));
}

TEST(test_nd_topology, local_neighbors_cover_3d_multibox_and_deep_corner_images) {
  const Box<3> domain{Index<3>{0, 0, 0}, Index<3>{3, 1, 1}};
  const BoxArray<3> split = BoxArray<3>::from_domain(domain, Extent<3>{2, 2, 2});
  const auto topology = PeriodicTopology<3>::axis_translations({true, true, true});
  const auto jobs =
      enumerate_local_translation_neighbors(split, domain, Extent<3>{2, 1, 1}, topology,
                                            Extent<3>{2, 2, 2}, kHashBudget, kNeighborBudget);
  EXPECT_EQ(jobs, brute_translation_neighbors(split, domain, Extent<3>{2, 1, 1}, topology));

  const Box<3> one_cell_domain{Index<3>{0, 0, 0}, Index<3>{0, 0, 0}};
  const BoxArray<3> one_cell = BoxArray<3>::from_domain(one_cell_domain, Extent<3>{1, 1, 1});
  const auto deep =
      enumerate_local_translation_neighbors(one_cell, one_cell_domain, Extent<3>{2, 1, 1}, topology,
                                            Extent<3>{1, 1, 1}, kHashBudget, kNeighborBudget);
  EXPECT_EQ(deep,
            brute_translation_neighbors(one_cell, one_cell_domain, Extent<3>{2, 1, 1}, topology));
  EXPECT_NE(find_job(deep, 0, 0, {2, 1, 1}), nullptr);
}

TEST(test_nd_topology, local_neighbors_reject_unmappable_topology_and_checked_ghost_growth) {
  const Box<2> domain{Index<2>{0, 0}, Index<2>{1, 1}};
  const BoxArray<2> boxes = BoxArray<2>::from_domain(domain, Extent<2>{2, 2});
  const PeriodicTopology<2> mapped{std::vector<PeriodicIdentification<2>>{PeriodicIdentification<2>{
      Face<2>{0, Side::lower}, Face<2>{1, Side::upper}, SignedPermutation<2>{{1, 0}, {1, -1}}}}};
  EXPECT_THROW(
      (void)enumerate_local_translation_neighbors(boxes, domain, Extent<2>{1, 1}, mapped,
                                                  Extent<2>{2, 2}, kHashBudget, kNeighborBudget),
      std::invalid_argument);

  const Box<1> edge{Index<1>{std::numeric_limits<int>::min()},
                    Index<1>{std::numeric_limits<int>::min()}};
  const BoxArray<1> edge_boxes(std::vector<Box<1>>{edge});
  EXPECT_THROW((void)enumerate_local_translation_neighbors(edge_boxes, edge, Extent<1>{1},
                                                           PeriodicTopology<1>{}, Extent<1>{1},
                                                           kHashBudget, kNeighborBudget),
               std::overflow_error);
}
