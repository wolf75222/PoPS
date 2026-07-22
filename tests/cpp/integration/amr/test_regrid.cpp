// Regrid : un niveau fin est cree autour de la region taguee, les donnees fines
// sont interpolees depuis le grossier, le buffer dilate la region, un re-regrid
// preserve l'ancien fin, et un tagging vide supprime le niveau fin.

#include <gtest/gtest.h>

#include "load_balance_test_authority.hpp"

#include <pops/amr/hierarchy/amr_hierarchy.hpp>
#include <pops/amr/tagging/cluster.hpp>
#include <pops/amr/tagging/clustering_provider.hpp>
#include <pops/amr/regridding/regrid.hpp>
#include <pops/coupling/amr/amr_regrid_coupler.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace pops;

namespace {

// feature centrale : 1 dans [6..9]^2, 0 ailleurs
double feature(int i, int j) {
  return (i >= 6 && i <= 9 && j >= 6 && j <= 9) ? 1.0 : 0.0;
}

auto threshold_crit() {
  return [] POPS_HD(const ConstArray4& a, int i, int j) { return a(i, j, 0) > 0.5; };
}

bool close(Real x, Real y) {
  return std::fabs(x - y) < 1e-9;
}

void regrid_test_hierarchy(AmrHierarchy& hierarchy, int tag_buffer) {
  const amr::BergerRigoutsosProvider clustering(ClusterParams{});
  const RegridProlongation prolongation = [](const MultiFab& coarse, MultiFab& fine,
                                             int, int refinement_ratio, bool,
                                             const CommunicatorView& communicator) {
    interpolate(coarse, fine, refinement_ratio, communicator);
  };
  const HierarchyRegridOptions options{/*tag_buffer=*/tag_buffer,
                                       /*nesting_margin=*/0,
                                       RegridPeriodicity{false, false}, nullptr};
  (void)regrid_hierarchy_level(hierarchy, 0, threshold_crit(), options, clustering,
                               prolongation, world_communicator_view());
}

class FixedClusteringProvider final : public amr::ClusteringProvider {
 public:
  explicit FixedClusteringProvider(std::vector<Box2D> boxes, bool fail = false)
      : boxes_(std::move(boxes)), fail_(fail) {}

  std::vector<Box2D> cluster(const TagBox&) const override {
    if (fail_)
      throw std::runtime_error("synthetic clustering failure");
    return boxes_;
  }

 private:
  std::vector<Box2D> boxes_;
  bool fail_ = false;
};

}  // namespace

TEST(test_regrid, Runs) {
  Box2D cdom = Box2D::from_extents(16, 16);
  const auto load_balance = test::prepare_test_space_filling_curve_load_balance();

  // --- regrid sans buffer : box fine = refine de la feature ---
  {
    AmrHierarchy h(cdom, 16, 1, 1, load_balance, 2);
    Array4 a = h.data(0).fab(0).array();
    for_each_cell(cdom, [a](int i, int j) { a(i, j, 0) = feature(i, j); });

    regrid_test_hierarchy(h, /*tag_buffer=*/0);

    EXPECT_EQ(h.num_levels(), 2) << "level_created";
    EXPECT_TRUE(h.domain(1) == cdom.refine(2)) << "fine_domain";
    EXPECT_EQ(h.boxes(1).size(), 1) << "one_fine_box";
    EXPECT_TRUE(h.boxes(1)[0] == (Box2D{{12, 12}, {19, 19}})) << "fine_box_extent";
    // interpolation injective : fine(12,12)=coarse(6,6)=1, fine(19,19)=coarse(9,9)=1
    EXPECT_TRUE(close(h.data(1).fab(0)(12, 12, 0), 1.0)) << "interp_lo";
    EXPECT_TRUE(close(h.data(1).fab(0)(19, 19, 0), 1.0)) << "interp_hi";
    // conservation de l'injection : 16 cellules grossieres -> 64 fines a 1
    EXPECT_TRUE(close(sum(h.data(1)), 64.0)) << "interp_sum";
  }

  // --- buffer dilate la region taguee ---
  {
    AmrHierarchy h(cdom, 16, 1, 1, load_balance, 2);
    Array4 a = h.data(0).fab(0).array();
    for_each_cell(cdom, [a](int i, int j) { a(i, j, 0) = feature(i, j); });

    regrid_test_hierarchy(h, /*tag_buffer=*/1);
    // tags [6..9] dilates -> [5..10], refine -> [10..21]
    EXPECT_TRUE(h.boxes(1)[0] == (Box2D{{10, 10}, {21, 21}})) << "buffered_box";
  }

  // --- re-regrid : l'ancien fin est preserve la ou il recouvre ---
  {
    AmrHierarchy h(cdom, 16, 1, 1, load_balance, 2);
    Array4 a = h.data(0).fab(0).array();
    for_each_cell(cdom, [a](int i, int j) { a(i, j, 0) = feature(i, j); });

    regrid_test_hierarchy(h, /*tag_buffer=*/0);
    h.data(1).fab(0)(12, 12, 0) = 999.0;  // marqueur dans le fin

    regrid_test_hierarchy(h, /*tag_buffer=*/0);  // memes boxes
    EXPECT_TRUE(close(h.data(1).fab(0)(12, 12, 0), 999.0)) << "old_fine_preserved";
    EXPECT_TRUE(close(h.data(1).fab(0)(19, 19, 0), 1.0)) << "rest_interpolated";
  }

  // --- tagging vide : le niveau fin disparait ---
  {
    AmrHierarchy h(cdom, 16, 1, 1, load_balance, 2);
    Array4 a = h.data(0).fab(0).array();
    for_each_cell(cdom, [a](int i, int j) { a(i, j, 0) = feature(i, j); });
    regrid_test_hierarchy(h, /*tag_buffer=*/1);
    EXPECT_EQ(h.num_levels(), 2) << "before_clear";

    h.data(0).set_val(0.0);  // plus aucune cellule au-dessus du seuil
    regrid_test_hierarchy(h, /*tag_buffer=*/1);
    EXPECT_EQ(h.num_levels(), 1) << "fine_removed";
  }
}

TEST(test_regrid, HierarchyPublicationIsAtomicAcrossProviderAndTransferFailures) {
  const Box2D domain = Box2D::from_extents(16, 16);
  AmrHierarchy hierarchy(domain, 16, 1, 1,
                         test::prepare_test_space_filling_curve_load_balance(), 2);
  Array4 values = hierarchy.data(0).fab(0).array();
  for_each_cell(domain,
                [values](int i, int j) { values(i, j, 0) = feature(i, j); });
  regrid_test_hierarchy(hierarchy, /*tag_buffer=*/0);
  ASSERT_EQ(hierarchy.num_levels(), 2);
  hierarchy.data(1).fab(0)(12, 12, 0) = 1234.0;
  const std::vector<Box2D> stable_boxes = hierarchy.boxes(1).boxes();

  const HierarchyRegridOptions options{/*tag_buffer=*/0,
                                       /*nesting_margin=*/0,
                                       RegridPeriodicity{false, false}, nullptr};
  const RegridProlongation interpolation = [](const MultiFab& coarse, MultiFab& fine,
                                              int, int refinement_ratio, bool,
                                              const CommunicatorView& communicator) {
    interpolate(coarse, fine, refinement_ratio, communicator);
  };
  const FixedClusteringProvider provider_failure({}, true);
  EXPECT_THROW(regrid_hierarchy_level(hierarchy, 0, threshold_crit(), options,
                                      provider_failure, interpolation,
                                      world_communicator_view()),
               std::runtime_error);
  EXPECT_EQ(hierarchy.num_levels(), 2);
  EXPECT_EQ(hierarchy.boxes(1).boxes(), stable_boxes);
  EXPECT_DOUBLE_EQ(hierarchy.data(1).fab(0)(12, 12, 0), 1234.0);

  const FixedClusteringProvider valid({Box2D{{6, 6}, {9, 9}}});
  const RegridProlongation transfer_failure = [](const MultiFab&, MultiFab&, int, int, bool,
                                                  const CommunicatorView&) {
    throw std::runtime_error("synthetic transfer failure");
  };
  EXPECT_THROW(regrid_hierarchy_level(hierarchy, 0, threshold_crit(), options, valid,
                                      transfer_failure, world_communicator_view()),
               std::runtime_error);
  EXPECT_EQ(hierarchy.num_levels(), 2);
  EXPECT_EQ(hierarchy.boxes(1).boxes(), stable_boxes);
  EXPECT_DOUBLE_EQ(hierarchy.data(1).fab(0)(12, 12, 0), 1234.0);
}

TEST(test_regrid, RejectsNegativeTagGrowthWithoutChangingInput) {
  const Box2D domain = Box2D::from_extents(4, 4);
  TagBox tags(domain);
  tags(1, 1) = 1;
  EXPECT_THROW((void)grow_tags(tags, -1, domain), std::invalid_argument);
  EXPECT_EQ(tags.count(), 1);
  EXPECT_TRUE(tags.tagged(1, 1));
}

TEST(test_regrid, TaggingRejectsLayoutsThatCouldWriteOutsideOrRaceTheMask) {
  const Box2D domain = Box2D::from_extents(4, 4);
  const auto criterion = threshold_crit();
  MultiFab outside(BoxArray({Box2D{{3, 0}, {4, 1}}}),
                   DistributionMapping(std::vector<int>{0}), 1, 0);
  EXPECT_THROW((void)tag_cells(outside, domain, criterion), std::invalid_argument);

  const BoxArray overlapping(
      {Box2D{{0, 0}, {2, 3}}, Box2D{{2, 0}, {3, 3}}});
  MultiFab raced(overlapping, DistributionMapping(std::vector<int>{0, 0}), 1, 0);
  EXPECT_THROW((void)tag_cells(raced, domain, criterion), std::invalid_argument);

  MultiFab empty;
  EXPECT_THROW((void)tag_cells(empty, domain, criterion), std::invalid_argument);
}

TEST(test_regrid, ProperNestingUsesTheParentLevelUnionAcrossPatchSeams) {
  const BoxArray parents(std::vector<Box2D>{Box2D{{4, 4}, {15, 27}}, Box2D{{16, 4}, {27, 27}}});
  // Coarse footprint [14..17] crosses an arbitrary parent-patch join. The adjacent patches jointly
  // supply its one-cell stencil halo, so tiling the same parent level differently cannot change
  // admissibility.
  const BoxArray child(std::vector<Box2D>{Box2D{{28, 16}, {35, 31}}});
  EXPECT_NO_THROW(
      validate_fine_layout_proper_nesting(child, parents, /*refinement_ratio=*/2, /*margin=*/1));
}

TEST(test_regrid, ThirdLevelClusteringUsesTheParentLevelCoverage) {
  const auto load_balance = test::prepare_test_space_filling_curve_load_balance();
  const Box2D parent_domain = Box2D::from_extents(32, 32);
  const BoxArray parents(std::vector<Box2D>{Box2D{{4, 4}, {15, 27}}, Box2D{{16, 4}, {27, 27}}});
  TagBox tags(parent_domain);
  // Tags touch both sides of the join. Clustering may bridge that seam because the parent level is
  // continuous there; it must still remain inside the union coverage plus resolved margin.
  for (int j = 10; j <= 18; ++j)
    for (int i = 13; i <= 18; ++i)
      tags(i, j) = 1;

  auto [children, mapping] = regrid_compute_fine_layout(
      std::move(tags), parent_domain, /*parent_level=*/1, /*margin=*/1,
      /*coarse_replicated=*/true, ClusterParams{}, *load_balance,
      world_communicator_view(), /*refinement_ratio=*/2, &parents);

  ASSERT_GT(children.size(), 0);
  EXPECT_EQ(mapping.size(), children.size());
  EXPECT_NO_THROW(
      validate_fine_layout_proper_nesting(children, parents, /*refinement_ratio=*/2, /*margin=*/1));
}

TEST(test_regrid, ProperNestingWrapsOnlyOnDeclaredPeriodicAxes) {
  const Box2D domain = Box2D::from_extents(8, 8);
  const BoxArray parents(std::vector<Box2D>{domain});
  const BoxArray edge_child(std::vector<Box2D>{Box2D{{0, 0}, {1, 1}}});

  EXPECT_THROW(validate_fine_layout_proper_nesting(
                   edge_child, parents, domain, /*refinement_ratio=*/2, /*margin=*/1,
                   RegridPeriodicity{false, false}),
               std::runtime_error);
  EXPECT_NO_THROW(validate_fine_layout_proper_nesting(
      edge_child, parents, domain, /*refinement_ratio=*/2, /*margin=*/1,
      RegridPeriodicity{true, true}));
}

TEST(test_regrid, TagGrowthWrapsOnlyOnDeclaredPeriodicAxesAtOffsetOrigin) {
  const Box2D domain{{3, 5}, {10, 12}};
  TagBox tags(domain);
  tags(domain.lo[0], domain.lo[1]) = 1;
  const TagBox grown =
      grow_regrid_tags(tags, /*radius=*/1, domain, RegridPeriodicity{true, false});
  EXPECT_TRUE(grown.tagged(domain.hi[0], domain.lo[1]));
  EXPECT_TRUE(grown.tagged(domain.lo[0], domain.lo[1]));
  EXPECT_TRUE(grown.tagged(domain.lo[0] + 1, domain.lo[1]));
  EXPECT_FALSE(grown.tagged(domain.lo[0], domain.hi[1]));
}

TEST(test_regrid, ProperNestingAtPhysicalWallRequiresCertifiedGhostDepth) {
  const Box2D domain{{3, 5}, {10, 12}};
  const BoxArray parents(std::vector<Box2D>{domain});
  const BoxArray wall_child(
      std::vector<Box2D>{Box2D{{2 * domain.lo[0], 2 * domain.lo[1]},
                               {2 * domain.lo[0] + 3, 2 * domain.lo[1] + 3}}});

  EXPECT_THROW(validate_fine_layout_proper_nesting(
                   wall_child, parents, domain, /*refinement_ratio=*/2, /*margin=*/2,
                   RegridPeriodicity{false, false}),
               std::runtime_error);
  const RegridPhysicalGhostSupport insufficient{/*provided_depth=*/1,
                                                /*fills_all_requested_depth=*/false};
  EXPECT_THROW(validate_fine_layout_proper_nesting(
                   wall_child, parents, domain, /*refinement_ratio=*/2, /*margin=*/2,
                   RegridPeriodicity{false, false}, &insufficient),
               std::runtime_error);
  const RegridPhysicalGhostSupport certified{/*provided_depth=*/2,
                                             /*fills_all_requested_depth=*/false};
  EXPECT_NO_THROW(validate_fine_layout_proper_nesting(
      wall_child, parents, domain, /*refinement_ratio=*/2, /*margin=*/2,
      RegridPeriodicity{false, false}, &certified));
}

TEST(test_regrid, ExternalClusteringProviderOutputIsValidatedBeforeLayoutPublication) {
  const Box2D domain{{3, 5}, {10, 12}};
  const auto load_balance = test::prepare_test_space_filling_curve_load_balance();
  auto invoke = [&](const amr::ClusteringProvider& provider) {
    TagBox tags(domain);
    tags(6, 8) = 1;
    return regrid_compute_fine_layout_with_provider(
        std::move(tags), domain, /*parent_level=*/0, /*margin=*/0,
        /*coarse_replicated=*/true, provider, *load_balance, world_communicator_view());
  };

  const FixedClusteringProvider throws({}, true);
  EXPECT_THROW(invoke(throws), std::runtime_error);
  const FixedClusteringProvider empty_box({Box2D{}});
  EXPECT_THROW(invoke(empty_box), std::runtime_error);
  const FixedClusteringProvider outside({Box2D{{2, 5}, {6, 8}}});
  EXPECT_THROW(invoke(outside), std::runtime_error);
  const FixedClusteringProvider overlap(
      {Box2D{{5, 7}, {7, 9}}, Box2D{{6, 8}, {8, 10}}});
  EXPECT_THROW(invoke(overlap), std::runtime_error);
  const FixedClusteringProvider drops_tag({Box2D{{3, 5}, {4, 6}}});
  EXPECT_THROW(invoke(drops_tag), std::runtime_error);

  const FixedClusteringProvider valid({Box2D{{5, 7}, {7, 9}}});
  const auto [boxes, owners] = invoke(valid);
  ASSERT_EQ(boxes.size(), 1);
  EXPECT_EQ(owners.size(), 1);
  EXPECT_TRUE(boxes[0] == (Box2D{{10, 14}, {15, 19}}));
}

TEST(test_regrid, RefinedLayoutOverflowFailsBeforePublication) {
  const int high = std::numeric_limits<int>::max() - 1;
  const Box2D domain{{high, 0}, {high, 0}};
  TagBox tags(domain);
  tags(high, 0) = 1;
  const FixedClusteringProvider provider({domain});
  const auto load_balance = test::prepare_test_space_filling_curve_load_balance();
  EXPECT_THROW(
      (void)regrid_compute_fine_layout_with_provider(
          std::move(tags), domain, /*parent_level=*/0, /*margin=*/0,
          /*coarse_replicated=*/true, provider, *load_balance,
          world_communicator_view(), /*refinement_ratio=*/2),
      std::overflow_error);
}
