#include <gtest/gtest.h>
#include <pops/runtime/program/prepared_amr_spatial_residual.hpp>
#include <pops/parallel/comm.hpp>

namespace {
class CommEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override { pops::comm_init(); }
  void TearDown() override { pops::comm_finalize(); }
};
[[maybe_unused]] const auto* environment = ::testing::AddGlobalTestEnvironment(new CommEnvironment);
using namespace pops;

MultiFab<2> field(bool fine, bool replicated) {
  const Box<2> domain =
      fine ? Box<2>{Index<2>{8, 8}, Index<2>{23, 23}} : Box<2>{Index<2>{0, 0}, Index<2>{15, 15}};
  const auto boxes = mesh::BoxArray<2>::from_domain(domain, Extent<2>{8, 8});
  const mesh::RankSpace<2> ranks{Index<2>{}, Extent<2>{n_ranks(), 1}};
  std::vector<Index<2>> owners;
  for (std::size_t patch = 0; patch < boxes.size(); ++patch)
    owners.push_back(Index<2>{static_cast<int>(patch % n_ranks()), 0});
  const auto distribution = replicated ? mesh::Distribution<2>::replicated(boxes, ranks)
                                       : mesh::Distribution<2>::partitioned(boxes, ranks, owners);
  return MultiFab<2>(boxes, distribution, Index<2>{my_rank(), 0}, 1, Extent<2>{});
}
}  // namespace

TEST(AmrSpatialNorm, MixedOwnershipCountsOnlyThePhysicalActiveCover) {
  using namespace pops;
  const auto lane = ExecutionLane::world("test.amr-spatial.active-cover");
  for (const auto ownership :
       {std::array<bool, 2>{false, false}, {true, false}, {false, true}, {true, true}}) {
    auto coarse = field(false, ownership[0]), fine = field(true, ownership[1]);
    auto coarse_mask = field(false, ownership[0]), fine_mask = field(true, ownership[1]);
    coarse.set_val(0);
    fine.set_val(0);
    fine_mask.set_val(1);
    for (std::size_t local = 0; local < coarse_mask.local_size(); ++local) {
      const auto view = coarse_mask.fab(local).view();
      for_each_cell(coarse_mask.box(local), [=] POPS_HD(const Index<2>& cell) {
        view(cell, 0) =
            cell[0] >= 4 && cell[0] <= 11 && cell[1] >= 4 && cell[1] <= 11 ? Real(0) : Real(1);
      });
    }
    const std::array<const MultiFab<2>*, 2> layouts{&coarse, &fine},
        masks{&coarse_mask, &fine_mask};
    const std::array<Real, 2> measures{Real(1) / 256, Real(1) / 1024};
    FieldNewtonOptions options;
    options.tolerance = Real(2);
    AmrFieldNewtonKrylovWorkspace<2> workspace(layouts, masks, measures, options);
    const std::array<MultiFab<2>*, 2> destinations{&coarse, &fine};
    auto residual = [](const auto& q, auto& output, int) {
      for (std::size_t level = 0; level < q.size(); ++level) {
        output[level].set_val(level == 0 ? Real(1) : Real(3));
        saxpy(output[level], Real(-1), q[level]);
      }
    };
    auto derivative = [](const auto&, const auto& direction, auto& output, int) {
      for (std::size_t level = 0; level < output.size(); ++level)
        lincomb(output[level], Real(1), direction[level], Real(0), direction[level]);
    };
    const auto report = workspace.solve(destinations, residual, derivative, [](auto&) {}, lane);
    ASSERT_TRUE(report.solved_value_available()) << report.reason;
    // 3/4 uncovered coarse cells with residual 1, 1/4 fine volume with residual 3.
    EXPECT_NEAR(report.residual_norm, std::sqrt(Real(3)), Real(1e-14));
    EXPECT_EQ(report.iters, 0);
  }
}
