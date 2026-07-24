// Regrid : un niveau fin est cree autour de la region taguee, les donnees fines
// sont interpolees depuis le grossier, le buffer dilate la region, un re-regrid
// preserve l'ancien fin, et un tagging vide supprime le niveau fin.

#include <gtest/gtest.h>

#include <pops/amr/hierarchy/amr_hierarchy.hpp>
#include <pops/amr/tagging/cluster.hpp>
#include <pops/amr/regridding/regrid.hpp>
#include <pops/coupling/amr/amr_coupler_mp.hpp>
#include <pops/coupling/amr/amr_regrid_coupler.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cmath>

using namespace pops;

namespace {

// feature centrale : 1 dans [6..9]^2, 0 ailleurs
double feature(int i, int j) {
  return (i >= 6 && i <= 9 && j >= 6 && j <= 9) ? 1.0 : 0.0;
}

auto threshold_crit() {
  return [](const ConstArray4& a, int i, int j) { return a(i, j, 0) > 0.5; };
}

bool close(Real x, Real y) {
  return std::fabs(x - y) < 1e-9;
}

struct DormantFineScalar {
  using State = StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;

  POPS_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  POPS_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  POPS_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

}  // namespace

TEST(test_regrid, Runs) {
  Box2D cdom = Box2D::from_extents(16, 16);

  // --- regrid sans buffer : box fine = refine de la feature ---
  {
    AmrHierarchy h(cdom, 16, 1, 1, 2);
    Array4 a = h.data(0).fab(0).array();
    for_each_cell(cdom, [a](int i, int j) { a(i, j, 0) = feature(i, j); });

    RegridParams rp;
    rp.n_buffer = 0;
    regrid_level(h, 0, threshold_crit(), rp);

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
    AmrHierarchy h(cdom, 16, 1, 1, 2);
    Array4 a = h.data(0).fab(0).array();
    for_each_cell(cdom, [a](int i, int j) { a(i, j, 0) = feature(i, j); });

    RegridParams rp;
    rp.n_buffer = 1;
    regrid_level(h, 0, threshold_crit(), rp);
    // tags [6..9] dilates -> [5..10], refine -> [10..21]
    EXPECT_TRUE(h.boxes(1)[0] == (Box2D{{10, 10}, {21, 21}})) << "buffered_box";
  }

  // --- re-regrid : l'ancien fin est preserve la ou il recouvre ---
  {
    AmrHierarchy h(cdom, 16, 1, 1, 2);
    Array4 a = h.data(0).fab(0).array();
    for_each_cell(cdom, [a](int i, int j) { a(i, j, 0) = feature(i, j); });

    RegridParams rp;
    rp.n_buffer = 0;
    regrid_level(h, 0, threshold_crit(), rp);
    h.data(1).fab(0)(12, 12, 0) = 999.0;  // marqueur dans le fin

    regrid_level(h, 0, threshold_crit(), rp);  // memes boxes
    EXPECT_TRUE(close(h.data(1).fab(0)(12, 12, 0), 999.0)) << "old_fine_preserved";
    EXPECT_TRUE(close(h.data(1).fab(0)(19, 19, 0), 1.0)) << "rest_interpolated";
  }

  // --- tagging vide : le niveau fin disparait ---
  {
    AmrHierarchy h(cdom, 16, 1, 1, 2);
    Array4 a = h.data(0).fab(0).array();
    for_each_cell(cdom, [a](int i, int j) { a(i, j, 0) = feature(i, j); });
    regrid_level(h, 0, threshold_crit(), RegridParams{});
    EXPECT_EQ(h.num_levels(), 2) << "before_clear";

    h.data(0).set_val(0.0);  // plus aucune cellule au-dessus du seuil
    regrid_level(h, 0, threshold_crit(), RegridParams{});
    EXPECT_EQ(h.num_levels(), 1) << "fine_removed";
  }
}

TEST(test_regrid, RejectsAProviderLayoutThatBreaksProperNesting) {
  const Box2D parent_domain = Box2D::from_extents(32, 32);
  const BoxArray parents(std::vector<Box2D>{Box2D{{4, 4}, {15, 27}}, Box2D{{16, 4}, {27, 27}}});
  // Coarse footprint [14..17] crosses the parent-patch join. No single parent can supply the
  // resolved one-cell stencil halo, even though the two parent patches are adjacent.
  const BoxArray invalid(std::vector<Box2D>{Box2D{{28, 16}, {35, 31}}});
  EXPECT_THROW(
      validate_fine_layout_proper_nesting(invalid, parents, parent_domain, /*refinement_ratio=*/2,
                                          /*margin=*/1),
      std::runtime_error);
}

TEST(test_regrid, ThirdLevelClusteringStaysInsideOneParentPatch) {
  const Box2D parent_domain = Box2D::from_extents(32, 32);
  const BoxArray parents(std::vector<Box2D>{Box2D{{4, 4}, {15, 27}}, Box2D{{16, 4}, {27, 27}}});
  TagBox tags(parent_domain);
  // Tags touch both sides of the join. A domain-wide cluster would bridge the patches; the
  // provider must instead return independently nested children.
  for (int j = 10; j <= 18; ++j)
    for (int i = 13; i <= 18; ++i)
      tags(i, j) = 1;

  auto [children, mapping] = regrid_compute_fine_layout(
      std::move(tags), parent_domain, /*parent_level=*/1, /*margin=*/1,
      /*coarse_replicated=*/true, ClusterParams{}, /*refinement_ratio=*/2, &parents);

  ASSERT_GT(children.size(), 0);
  EXPECT_EQ(mapping.size(), children.size());
  EXPECT_NO_THROW(
      validate_fine_layout_proper_nesting(children, parents, parent_domain,
                                          /*refinement_ratio=*/2, /*margin=*/1));
}

TEST(test_regrid, BoundaryTagsAreNotDiscardedByNestingMargin) {
  const Box2D parent_domain = Box2D::from_extents(16, 16);
  TagBox tags(parent_domain);
  tags(0, 0) = 1;
  tags(15, 15) = 1;

  auto [children, mapping] = regrid_compute_fine_layout(
      grow_tags(tags, /*n=*/1, parent_domain), parent_domain, /*parent_level=*/0,
      /*margin=*/1, /*coarse_replicated=*/true, ClusterParams{}, /*refinement_ratio=*/2);

  ASSERT_GT(children.size(), 0);
  EXPECT_EQ(mapping.size(), children.size());
  bool covers_lower_boundary = false;
  bool covers_upper_boundary = false;
  for (int box = 0; box < children.size(); ++box) {
    covers_lower_boundary =
        covers_lower_boundary || children[box].contains(Box2D{{0, 0}, {1, 1}});
    covers_upper_boundary =
        covers_upper_boundary || children[box].contains(Box2D{{30, 30}, {31, 31}});
  }
  EXPECT_TRUE(covers_lower_boundary) << "lower physical boundary tag survives";
  EXPECT_TRUE(covers_upper_boundary) << "upper physical boundary tag survives";
}

TEST(test_regrid, PeriodicTagBufferWrapsAcrossDomainBoundaries) {
  const Box2D parent_domain = Box2D::from_extents(16, 12);
  TagBox tags(parent_domain);
  tags(15, 6) = 1;
  tags(8, 11) = 1;

  const TagBox periodic_x =
      grow_tags(tags, /*n=*/1, parent_domain, /*periodic_x=*/true, /*periodic_y=*/false);
  EXPECT_EQ(periodic_x(0, 6), 1) << "right-edge x tag wraps to the left edge";
  EXPECT_EQ(periodic_x(8, 0), 0) << "non-periodic y growth remains clipped";

  const TagBox periodic_xy =
      grow_tags(tags, /*n=*/1, parent_domain, /*periodic_x=*/true, /*periodic_y=*/true);
  EXPECT_EQ(periodic_xy(0, 6), 1) << "right-edge x tag wraps to the left edge";
  EXPECT_EQ(periodic_xy(8, 0), 1) << "top-edge y tag wraps to the bottom edge";
}

TEST(test_regrid, ParentPatchBoundaryKeepsMarginExceptAtPhysicalBoundary) {
  const Box2D parent_domain = Box2D::from_extents(32, 32);
  const BoxArray parents(std::vector<Box2D>{Box2D{{0, 0}, {15, 31}}, Box2D{{16, 0}, {31, 31}}});
  TagBox tags(parent_domain);
  tags(0, 7) = 1;
  tags(15, 7) = 1;
  tags(31, 7) = 1;

  auto [children, mapping] = regrid_compute_fine_layout(
      grow_tags(tags, /*n=*/0, parent_domain), parent_domain, /*parent_level=*/1,
      /*margin=*/1, /*coarse_replicated=*/true, ClusterParams{}, /*refinement_ratio=*/2,
      &parents);

  ASSERT_GT(children.size(), 0);
  EXPECT_EQ(mapping.size(), children.size());
  bool covers_physical_lower_boundary = false;
  bool covers_internal_parent_boundary = false;
  bool covers_physical_upper_boundary = false;
  for (int box = 0; box < children.size(); ++box) {
    covers_physical_lower_boundary =
        covers_physical_lower_boundary || children[box].contains(Box2D{{0, 14}, {1, 15}});
    covers_internal_parent_boundary =
        covers_internal_parent_boundary || children[box].contains(Box2D{{30, 14}, {31, 15}});
    covers_physical_upper_boundary =
        covers_physical_upper_boundary || children[box].contains(Box2D{{62, 14}, {63, 15}});
  }
  EXPECT_TRUE(covers_physical_lower_boundary) << "physical boundary does not need parent margin";
  EXPECT_FALSE(covers_internal_parent_boundary) << "internal parent edge still keeps margin";
  EXPECT_TRUE(covers_physical_upper_boundary) << "physical boundary does not need parent margin";
  EXPECT_NO_THROW(
      validate_fine_layout_proper_nesting(children, parents, parent_domain,
                                          /*refinement_ratio=*/2, /*margin=*/1));
}

TEST(test_regrid, LegacyFineSlotClearsStalePatchesAndCanRegrow) {
  const Box2D coarse_domain = Box2D::from_extents(16, 16);
  const BoxArray coarse_boxes(std::vector<Box2D>{coarse_domain});
  // The legacy coupler's default level-0 policy is replicated: each rank owns its local full copy.
  const DistributionMapping coarse_mapping(std::vector<int>{my_rank()});
  const BoxArray seed_boxes(std::vector<Box2D>{Box2D{{12, 12}, {19, 19}}});
  const DistributionMapping seed_mapping(/*nboxes=*/1, n_ranks());

  MultiFab coarse(coarse_boxes, coarse_mapping, /*ncomp=*/1, /*ngrow=*/2);
  coarse.set_val(Real(0));
  MultiFab fine(seed_boxes, seed_mapping, /*ncomp=*/1, /*ngrow=*/3);
  fine.set_val(Real(99));

  std::vector<AmrLevelMP> levels;
  levels.push_back({std::move(coarse), nullptr, Real(1) / 16, Real(1) / 16});
  levels.push_back({std::move(fine), nullptr, Real(1) / 32, Real(1) / 32});

  Geometry geometry{coarse_domain, Real(0), Real(1), Real(0), Real(1)};
  BCRec elliptic_bc;
  elliptic_bc.xlo = elliptic_bc.xhi = elliptic_bc.ylo = elliptic_bc.yhi = BCType::Periodic;
  AmrCouplerMP<DormantFineScalar> coupler(DormantFineScalar{}, geometry, coarse_boxes, elliptic_bc,
                                           std::move(levels), {},
                                           /*replicated_coarse=*/true);

  const auto criterion =
      [](const ConstArray4& state, int i, int j) { return state(i, j, 0) > Real(0.5); };
  const auto active_fine_patch_count = [&coupler] {
    int count = 0;
    const auto& active_levels = coupler.levels();
    for (std::size_t level = 1; level < active_levels.size(); ++level)
      count += active_levels[level].U.box_array().size();
    return count;
  };

  // No tag: the configured slot remains address-stable, but its stale seed patch disappears.
  coupler.regrid(criterion, /*grow=*/0, /*margin=*/0);
  ASSERT_EQ(coupler.levels().size(), 2u);
  EXPECT_EQ(coupler.nlev(), 2) << "nlev is configured high-water depth, not active patch count";
  EXPECT_EQ(coupler.levels()[1].U.box_array().size(), 0);
  EXPECT_EQ(coupler.levels()[1].aux->box_array().size(), 0);
  EXPECT_EQ(coupler.levels()[1].U.ncomp(), 1);
  EXPECT_EQ(coupler.levels()[1].U.n_grow(), 3);
  EXPECT_EQ(active_fine_patch_count(), 0);
  for (const PatchBox& piece : coupler.output_geometry_boxes())
    EXPECT_EQ(piece.level, 0) << "legacy output must not expose a dormant fine level";

  // The empty high-water slot participates in a complete legacy update + AMR advance without a
  // bounding-box, coarse/fine-interface or reflux failure. Zero flux/source makes the state stable.
  EXPECT_NO_THROW(coupler.step(Real(1e-3)));
  EXPECT_EQ(active_fine_patch_count(), 0);

  // A later feature repopulates that dormant slot from the current coarse state, not stale fine
  // values. This is the legacy fixed-depth/high-water contract used by AmrCouplerMP.
  for (int local = 0; local < coupler.coarse().local_size(); ++local) {
    const Box2D valid = coupler.coarse().box(local);
    if (valid.contains(Box2D{{7, 8}, {7, 8}}))
      coupler.coarse().fab(local)(7, 8, 0) = Real(4);
  }
  coupler.regrid(criterion, /*grow=*/0, /*margin=*/0);

  ASSERT_GT(coupler.levels()[1].U.box_array().size(), 0);
  EXPECT_GT(active_fine_patch_count(), 0);
  EXPECT_EQ(coupler.levels()[1].U.n_grow(), 3);
  bool found_interpolated_feature = false;
  for (int local = 0; local < coupler.levels()[1].U.local_size(); ++local) {
    const ConstArray4 refined = coupler.levels()[1].U.fab(local).const_array();
    const Box2D valid = coupler.levels()[1].U.box(local);
    if (valid.contains(Box2D{{14, 16}, {15, 17}})) {
      found_interpolated_feature =
          close(refined(14, 16, 0), Real(4)) && close(refined(15, 17, 0), Real(4));
    }
  }
  EXPECT_EQ(all_reduce_max(static_cast<long>(found_interpolated_feature)), 1);
}
