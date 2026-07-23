// Native MPI contract for the prepared AMR ownership authority: exact replicated metadata,
// weighted deterministic mapping, and uniform fail-closed behavior on rank-local divergence.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/amr/hierarchy/amr_hierarchy.hpp>
#include <pops/amr/tagging/cluster.hpp>
#include <pops/coupling/amr/amr_regrid_coupler.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/prepared_load_balance.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

class RankFailingClusteringProvider final : public amr::ClusteringProvider {
 public:
  explicit RankFailingClusteringProvider(int failing_rank) : failing_rank_(failing_rank) {}

  std::vector<Box2D> cluster(const TagBox& tags) const override {
    if (my_rank() == failing_rank_)
      throw std::runtime_error("rank-local synthetic clustering failure");
    return berger_rigoutsos(tags, ClusterParams{});
  }

 private:
  int failing_rank_;
};

bool hierarchy_regrid_is_collective_and_owner_authenticated(
    const std::shared_ptr<const PreparedLoadBalanceAuthority>& authority) {
  const int ranks = n_ranks();
  const int rank = my_rank();
  const Box2D domain = Box2D::from_extents(std::max(8, ranks * 4), 4);
  const HierarchyRegridOptions options{/*tag_buffer=*/0, /*nesting_margin=*/0,
                                       RegridPeriodicity{false, false}, nullptr};
  const RegridProlongation prolong = [](const MultiFab& coarse, MultiFab& fine, int,
                                        int ratio, bool,
                                        const CommunicatorView& communicator) {
    interpolate(coarse, fine, ratio, communicator);
  };

  // L0 is unique-owner distributed. Only the final rank authors a tag; every rank must still build
  // and publish the same child layout through the mandatory tag OR.
  AmrHierarchy hierarchy(domain, /*max_grid_size=*/2, /*ncomp=*/1, /*ngrow=*/1, authority);
  hierarchy.data(0).set_val(Real(0));
  if (rank == ranks - 1 && hierarchy.data(0).local_size() > 0) {
    const Box2D local = hierarchy.data(0).box(0);
    hierarchy.data(0).fab(0)(local.lo[0], local.lo[1], 0) = Real(1);
  }
  const amr::BergerRigoutsosProvider clustering(ClusterParams{});
  const auto criterion = [] POPS_HD(const ConstArray4& values, int i, int j) {
    return values(i, j, 0) > Real(0.5);
  };
  if (!regrid_hierarchy_level(hierarchy, 0, criterion, options, clustering, prolong,
                              world_communicator_view()) ||
      hierarchy.num_levels() != 2)
    return false;

  // A rank-local extension failure is converted into one collective failure point and cannot
  // replace the already-published fine level on any peer.
  const std::vector<Box2D> stable_boxes = hierarchy.boxes(1).boxes();
  RankFailingClusteringProvider failing(ranks > 1 ? 1 : 0);
  bool rejected = false;
  try {
    (void)regrid_hierarchy_level(hierarchy, 0, criterion, options, failing, prolong,
                                 world_communicator_view());
  } catch (const std::exception&) {
    rejected = true;
  }
  if (!rejected || hierarchy.num_levels() != 2 ||
      hierarchy.boxes(1).boxes() != stable_boxes)
    return false;

  // install_level authenticates the exact owner vector against the same prepared authority. A
  // geometrically valid field with a rotated owner map is not accepted as equivalent provenance.
  if (ranks > 1) {
    AmrHierarchy install_target(domain, /*max_grid_size=*/2, /*ncomp=*/1, /*ngrow=*/1,
                                authority);
    const BoxArray fine = BoxArray::from_domain(domain.refine(2), 4);
    const DistributionMapping expected =
        authority->distribute(fine, ranks, {}, world_communicator_view());
    std::vector<int> rotated = expected.ranks();
    for (int& owner : rotated)
      owner = (owner + 1) % ranks;
    MultiFab wrong(fine, DistributionMapping(std::move(rotated)), 1, 1);
    rejected = false;
    try {
      install_target.install_level(1, fine, std::move(wrong), world_communicator_view());
    } catch (const std::exception&) {
      rejected = true;
    }
    if (!rejected || install_target.num_levels() != 1)
      return false;
    MultiFab correct(fine, expected, 1, 1);
    install_target.install_level(1, fine, std::move(correct), world_communicator_view());
    if (install_target.num_levels() != 2 ||
        install_target.data(1).dmap().ranks() != expected.ranks())
      return false;
  }
  return true;
}

int run_mpi_load_balance_authority(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const int rank = my_rank();
  const int ranks = n_ranks();
  long failures = 0;

  const int box_count = std::max(4, ranks * 4);
  const BoxArray boxes =
      BoxArray::from_domain(Box2D::from_extents(box_count, 1), 1);
  std::vector<std::int64_t> weights(static_cast<std::size_t>(box_count), 1);
  weights.front() = 17;

  const auto authority = prepare_load_balance_authority(
      "space_filling_curve", "test.mpi.weighted-sfc.semantic-identity",
      PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}});
  const auto hierarchy_authority = std::make_shared<PreparedLoadBalanceAuthority>(
      prepare_load_balance_authority(
          "space_filling_curve", "test.mpi.hierarchy-sfc.semantic-identity",
          PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}}));
  if (!hierarchy_regrid_is_collective_and_owner_authenticated(hierarchy_authority))
    ++failures;
  const DistributionMapping first = authority.distribute(boxes, ranks, weights);
  const DistributionMapping second = authority.distribute(boxes, ranks, weights);
  if (first.ranks() != second.ranks() || first.size() != boxes.size())
    ++failures;
  for (const int owner : first.ranks())
    if (owner < 0 || owner >= ranks)
      ++failures;

  if (ranks > 1) {
    auto divergent_weights = weights;
    if (rank == 1)
      divergent_weights.back() += 1;
    bool rejected = false;
    try {
      (void)authority.distribute(boxes, ranks, divergent_weights);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected)
      ++failures;

    const auto divergent_identity = prepare_load_balance_authority(
        "space_filling_curve",
        rank == 1 ? "test.mpi.other-sfc.semantic-identity"
                  : "test.mpi.weighted-sfc.semantic-identity",
        PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}});
    rejected = false;
    try {
      (void)divergent_identity.distribute(boxes, ranks, weights);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected)
      ++failures;

    auto invalid_weights = weights;
    if (rank == 1)
      invalid_weights.front() = 0;
    rejected = false;
    try {
      (void)authority.distribute(boxes, ranks, invalid_weights);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected)
      ++failures;
  }

  failures = all_reduce_sum(failures);
  if (rank == 0)
    std::printf("%s test_mpi_load_balance_authority (np=%d)\n",
                failures == 0 ? "OK" : "FAIL", ranks);
  comm_finalize();
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_load_balance_authority, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&run_mpi_load_balance_authority,
                                    "test_mpi_load_balance_authority"),
            0);
}
