#include <gtest/gtest.h>

#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/linear/pure_field_algebra.hpp>
#include <pops/parallel/comm.hpp>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

using namespace pops;

namespace {

class CommEnvironment : public ::testing::Environment {
 public:
  void SetUp() override { comm_init(); }
  void TearDown() override { comm_finalize(); }
};

::testing::Environment* const kCommEnv = ::testing::AddGlobalTestEnvironment(new CommEnvironment);

#if defined(POPS_HAS_KOKKOS)
class KokkosEnvironment : public ::testing::Environment {
 public:
  void SetUp() override { guard_.emplace(); }
  void TearDown() override { guard_.reset(); }

 private:
  std::optional<Kokkos::ScopeGuard> guard_;
};

::testing::Environment* const kKokkosEnv =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
#endif

BoxArray dot_boxes(int cell_count) {
  std::vector<Box2D> boxes;
  boxes.reserve(static_cast<std::size_t>(cell_count));
  for (int i = 0; i < cell_count; ++i)
    boxes.push_back(Box2D{{i, 0}, {i, 0}});
  return BoxArray(std::move(boxes));
}

DistributionMapping round_robin_mapping(int box_count) {
  std::vector<int> ranks;
  ranks.reserve(static_cast<std::size_t>(box_count));
  const int rank_count = n_ranks();
  for (int i = 0; i < box_count; ++i)
    ranks.push_back(i % rank_count);
  return DistributionMapping(std::move(ranks));
}

struct DotFields {
  MultiFab left;
  MultiFab right;
};

DotFields make_fields(int cell_count, int components = 1) {
  BoxArray boxes = dot_boxes(cell_count);
  DistributionMapping mapping = round_robin_mapping(boxes.size());
  DotFields fields{MultiFab(boxes, mapping, components, 0),
                   MultiFab(boxes, mapping, components, 0)};
  fields.left.set_val(Real(0));
  fields.right.set_val(Real(0));
  return fields;
}

DotFields make_replicated_fields(int cell_count, int components = 1) {
  BoxArray boxes = dot_boxes(cell_count);
  DistributionMapping mapping(std::vector<int>(static_cast<std::size_t>(boxes.size()), my_rank()));
  DotFields fields{MultiFab(boxes, mapping, components, 0),
                   MultiFab(boxes, mapping, components, 0)};
  fields.left.set_val(Real(0));
  fields.right.set_val(Real(0));
  return fields;
}

TEST(test_pure_field_algebra_extreme_dot, MpiRouteInitializesRequestedCommunicator) {
  const char* expected_ranks = std::getenv("POPS_TEST_EXPECT_RANKS");
  if (expected_ranks != nullptr)
    ASSERT_EQ(n_ranks(), std::atoi(expected_ranks))
        << "the MPI CTest route must initialize the requested communicator";
  else if (n_ranks() == 1)
    GTEST_SKIP() << "the serial registration has no remote rank";
}

DotFields make_rank_zero_owned_fields(int cell_count, int components = 1) {
  BoxArray boxes = dot_boxes(cell_count);
  DistributionMapping mapping(std::vector<int>(static_cast<std::size_t>(boxes.size()), 0));
  DotFields fields{MultiFab(boxes, mapping, components, 0),
                   MultiFab(boxes, mapping, components, 0)};
  fields.left.set_val(Real(0));
  fields.right.set_val(Real(0));
  return fields;
}

void set_global_cell(MultiFab& field, int global, Real value, int component = 0) {
  const int local = field.local_index_of(global);
  if (local < 0)
    return;
  const Box2D box = field.box(local);
  field.fab(local)(box.lo[0], box.lo[1], component) = value;
}

void expect_close_to_one(Real value) {
  ASSERT_TRUE(std::isfinite(static_cast<double>(value)));
  EXPECT_NEAR(static_cast<double>(value), 1.0, 8.0 * std::numeric_limits<double>::epsilon());
}

}  // namespace

TEST(test_pure_field_algebra_extreme_dot, PreservesCrossProductHiddenBelowGlobalScale) {
  DotFields fields = make_fields(2);
  set_global_cell(fields.left, 0, Real(1e200));
  set_global_cell(fields.right, 0, Real(0));
  set_global_cell(fields.left, 1, Real(1e-200));
  set_global_cell(fields.right, 1, Real(1e200));

  expect_close_to_one(PureFieldAlgebra::dot(fields.left, fields.right));
  expect_close_to_one(detail::PreparedFieldAlgebra::dot(fields.left, fields.right));
  expect_close_to_one(static_cast<Real>(all_reduce_sum(
      static_cast<double>(detail::PreparedFieldAlgebra::local_dot(fields.left, fields.right)))));
}

TEST(test_pure_field_algebra_extreme_dot, CancelsProductsThatWouldOverflowBeforeSummation) {
  DotFields fields = make_fields(3);
  set_global_cell(fields.left, 0, Real(1e200));
  set_global_cell(fields.right, 0, Real(1e200));
  set_global_cell(fields.left, 1, Real(1e200));
  set_global_cell(fields.right, 1, Real(-1e200));
  set_global_cell(fields.left, 2, Real(1e-200));
  set_global_cell(fields.right, 2, Real(1e200));

  expect_close_to_one(PureFieldAlgebra::dot(fields.left, fields.right));
  expect_close_to_one(detail::PreparedFieldAlgebra::dot(fields.left, fields.right));
}

TEST(test_pure_field_algebra_extreme_dot,
     ReplicatedRobustDotCountsOverflowingCancellationExactlyOnce) {
  DotFields fields = make_replicated_fields(3);
  set_global_cell(fields.left, 0, Real(1e200));
  set_global_cell(fields.right, 0, Real(1e200));
  set_global_cell(fields.left, 1, Real(1e200));
  set_global_cell(fields.right, 1, Real(-1e200));
  set_global_cell(fields.left, 2, Real(1e-200));
  set_global_cell(fields.right, 2, Real(1e200));

  expect_close_to_one(
      PureFieldAlgebra::dot(fields.left, fields.right, PreparedVectorDistribution::Replicated));
  expect_close_to_one(detail::PreparedFieldAlgebra::dot(fields.left, fields.right,
                                                        PreparedVectorDistribution::Replicated));
}

TEST(test_pure_field_algebra_extreme_dot, ProviderHandleRejectsInvalidNativeDescriptor) {
  EXPECT_THROW((void)PreparedVectorDistribution(static_cast<FieldDistribution>(255)),
               std::invalid_argument);
}

TEST(test_pure_field_algebra_extreme_dot, PublicReplicaOverloadsRejectPartialRankLayouts) {
  if (n_ranks() == 1)
    GTEST_SKIP() << "a serial mapping is necessarily a complete local replica";
  DotFields fields = make_fields(2);
  EXPECT_THROW((void)PureFieldAlgebra::dot(fields.left, fields.right,
                                           PreparedVectorDistribution::Replicated),
               std::invalid_argument);
  EXPECT_THROW((void)PureFieldAlgebra::norm(fields.left, PreparedVectorDistribution::Replicated),
               std::invalid_argument);
}

TEST(test_pure_field_algebra_extreme_dot,
     PublicReplicaValidationFailsUniformlyForRankZeroOwnedLayout) {
  if (n_ranks() == 1)
    GTEST_SKIP() << "rank-zero ownership is a complete local replica in serial";
  // Rank zero's local descriptor alone looks like a complete replica while every other rank owns
  // no boxes. All ranks must nevertheless complete the validation collectives and reject it
  // uniformly rather than letting rank zero enter the following physical reduction alone.
  DotFields fields = make_rank_zero_owned_fields(2);
  EXPECT_THROW((void)PureFieldAlgebra::max_abs(fields.left, PreparedVectorDistribution::Replicated),
               std::invalid_argument);
  EXPECT_THROW((void)PureFieldAlgebra::dot(fields.left, fields.right,
                                           PreparedVectorDistribution::Replicated),
               std::invalid_argument);
  EXPECT_THROW((void)PureFieldAlgebra::norm(fields.left, PreparedVectorDistribution::Replicated),
               std::invalid_argument);
}

TEST(test_pure_field_algebra_extreme_dot, PublicOwnershipMustAgreeAcrossRanks) {
  if (n_ranks() == 1)
    GTEST_SKIP() << "ownership descriptors cannot disagree in serial";
  DotFields fields = make_replicated_fields(2);
  const PreparedVectorDistribution ownership = my_rank() == 0
                                                   ? PreparedVectorDistribution::Distributed
                                                   : PreparedVectorDistribution::Replicated;
  EXPECT_THROW((void)PureFieldAlgebra::dot(fields.left, fields.right, ownership),
               std::invalid_argument);
}

TEST(test_pure_field_algebra_extreme_dot, PublicDistributedModeRejectsPhysicalReplicas) {
  if (n_ranks() == 1)
    GTEST_SKIP() << "distribution modes are structurally identical in serial";
  DotFields fields = make_replicated_fields(2);
  EXPECT_THROW((void)PureFieldAlgebra::norm(fields.left), std::invalid_argument);
  EXPECT_THROW((void)PureFieldAlgebra::dot(fields.left, fields.right), std::invalid_argument);
}

TEST(test_pure_field_algebra_extreme_dot, PublicReplicaRejectsIsometricValuePermutations) {
  if (n_ranks() == 1)
    GTEST_SKIP() << "replica values cannot disagree in serial";
  DotFields fields = make_replicated_fields(2);
  set_global_cell(fields.left, my_rank() == 0 ? 0 : 1, Real(1));
  EXPECT_THROW((void)PureFieldAlgebra::norm(fields.left, PreparedVectorDistribution::Replicated),
               std::runtime_error);
}

TEST(test_pure_field_algebra_extreme_dot, RepairsOverflowAfterBatchedGlobalReduction) {
  DotFields fields = make_fields(3);
  set_global_cell(fields.left, 0, Real(1e200));
  set_global_cell(fields.right, 0, Real(1e200));
  set_global_cell(fields.left, 1, Real(1e200));
  set_global_cell(fields.right, 1, Real(-1e200));
  set_global_cell(fields.left, 2, Real(1e-200));
  set_global_cell(fields.right, 2, Real(1e200));

  const Real local_fast = detail::PreparedFieldAlgebra::local_dot(fields.left, fields.right);
  const Real globally_reduced_fast =
      static_cast<Real>(all_reduce_sum(static_cast<double>(local_fast)));
  EXPECT_FALSE(std::isfinite(static_cast<double>(globally_reduced_fast)));
  std::array<double, detail::PreparedFieldAlgebra::kRobustDotPayloadWidth> payload{};
  detail::PreparedFieldAlgebra::local_robust_dot_payload(fields.left, fields.right, payload.data());
  all_reduce_sum_inplace(payload.data(), static_cast<int>(payload.size()));
  expect_close_to_one(detail::PreparedFieldAlgebra::dot_from_global_robust_payload(payload.data()));
}

TEST(test_pure_field_algebra_extreme_dot, CoversEveryComponentInPreparedVectorDot) {
  DotFields fields = make_fields(4, 2);
  set_global_cell(fields.left, 0, Real(1e200), 0);
  set_global_cell(fields.right, 0, Real(1e200), 0);
  set_global_cell(fields.left, 1, Real(1e200), 0);
  set_global_cell(fields.right, 1, Real(-1e200), 0);
  set_global_cell(fields.left, 2, Real(1e-200), 1);
  set_global_cell(fields.right, 2, Real(1e200), 1);
  set_global_cell(fields.left, 3, Real(1e-200), 1);
  set_global_cell(fields.right, 3, Real(1e200), 1);

  const Real value = detail::PreparedFieldAlgebra::dot(fields.left, fields.right);
  ASSERT_TRUE(std::isfinite(static_cast<double>(value)));
  EXPECT_NEAR(static_cast<double>(value), 2.0, 16.0 * std::numeric_limits<double>::epsilon());
}

TEST(test_pure_field_algebra_extreme_dot, NonfiniteInputIsUniformlyInvalid) {
  DotFields fields = make_fields(2);
  set_global_cell(fields.left, 0, std::numeric_limits<Real>::infinity());
  set_global_cell(fields.right, 0, Real(1));
  set_global_cell(fields.left, 1, Real(1));
  set_global_cell(fields.right, 1, Real(1));

  EXPECT_TRUE(std::isnan(static_cast<double>(PureFieldAlgebra::dot(fields.left, fields.right))));
  EXPECT_TRUE(std::isnan(
      static_cast<double>(detail::PreparedFieldAlgebra::dot(fields.left, fields.right))));
}
