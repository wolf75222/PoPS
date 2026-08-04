#include <gtest/gtest.h>

#include <pops/coupling/amr/amr_coupler_mp.hpp>
#include <pops/coupling/system/amr_system_coupler.hpp>
#include <pops/runtime/program/amr_program_context.hpp>

#include <array>
#include <type_traits>
#include <vector>

namespace {

namespace amr_time = pops::numerics::time::amr;

constexpr pops::mesh::BoxArrayValidationBudget kLayoutBudget{8, 28};

static_assert(pops::coupling::amr::AmrCouplerMP<1>::dimension == 1);
static_assert(pops::coupling::system::AmrSystemCoupler<3>::dimension == 3);
static_assert(pops::runtime::program::AmrProgramContext<3>::dimension == 3);

TEST(test_nd_amr_consumers, parent_footprints_keep_one_and_three_dimensional_ratios) {
  const amr_time::PatchRange<1> line(pops::Box<1>{pops::Index<1>{-8}, pops::Index<1>{-1}},
                                     pops::amr::RefinementRatio<1>{2});
  EXPECT_EQ(line.parent_footprint(), (pops::Box<1>{pops::Index<1>{-4}, pops::Index<1>{-1}}));

  const amr_time::PatchRange<3> volume(
      pops::Box<3>{pops::Index<3>{-4, 6, 7}, pops::Index<3>{-1, 11, 8}},
      pops::amr::RefinementRatio<3>{2, 3, 1});
  EXPECT_EQ(volume.parent_footprint(),
            (pops::Box<3>{pops::Index<3>{-2, 2, 7}, pops::Index<3>{-1, 3, 8}}));
}

TEST(test_nd_amr_consumers, level_domains_refine_every_axis_without_rank_dispatch) {
  const pops::Box<3> coarse{pops::Index<3>{-2, 1, 5}, pops::Index<3>{1, 2, 6}};
  const std::array ratios{pops::amr::RefinementRatio<3>{2, 3, 1}};
  EXPECT_EQ(amr_time::amr_level_index_domain<3>(coarse, ratios, 1),
            pops::amr::hierarchy::refine_box(coarse, ratios.front()));
}

TEST(test_nd_amr_consumers, interface_retains_anisotropic_layout_identity) {
  using Box = pops::Box<3>;
  using Index = pops::Index<3>;
  using Extent = pops::Extent<3>;
  using BoxArray = pops::mesh::BoxArray<3>;
  using Distribution = pops::mesh::Distribution<3>;

  const Box parent_domain{Index{-2, 1, 5}, Index{1, 2, 6}};
  const BoxArray parent_patches(std::vector<Box>{parent_domain});
  const pops::mesh::RankSpace<3> ranks(Index{0, 0, 0}, Extent{1, 1, 1});
  const Distribution parent_distribution = Distribution::replicated(parent_patches, ranks);
  const pops::amr::hierarchy::LevelLayout<3> parent(0, parent_domain, parent_patches,
                                                    parent_distribution,
                                                    pops::amr::RefinementRatio<3>{}, kLayoutBudget);

  const pops::amr::RefinementRatio<3> ratio{2, 3, 1};
  const Box child_domain = pops::amr::hierarchy::refine_box(parent_domain, ratio);
  const BoxArray child_patches(std::vector<Box>{child_domain});
  const Distribution child_distribution = Distribution::replicated(child_patches, ranks);
  const pops::amr::hierarchy::LevelLayout<3> child(1, child_domain, child_patches,
                                                   child_distribution, ratio, kLayoutBudget);

  const amr_time::CoarseFineInterface<3> interface(parent, child, kLayoutBudget);
  EXPECT_EQ(interface.ratio(), ratio);
  ASSERT_EQ(interface.parent_footprints().size(), 1U);
  EXPECT_EQ(interface.parent_footprints().front(), parent_domain);
  EXPECT_EQ(interface.face_mapping().coarse_origin, parent_domain.lo);
  EXPECT_EQ(interface.face_mapping().fine_origin, child_domain.lo);
}

}  // namespace
