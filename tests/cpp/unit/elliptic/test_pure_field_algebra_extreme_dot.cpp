#include <gtest/gtest.h>

#include <pops/mesh/index/box.hpp>
#include <pops/mesh/index/extent.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/layout/rank_space.hpp>
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

inline constexpr int kDim = 2;
using TestBox = Box<kDim>;
using TestLayout = mesh::BoxArray<kDim>;
using TestDistribution = mesh::Distribution<kDim>;
using TestRankSpace = mesh::RankSpace<kDim>;
using TestMultiFab = MultiFab<kDim>;
using TestVectorDistribution = PreparedVectorDistribution<kDim>;

Index<kDim> rank_coordinate(int rank) {
  return Index<kDim>{rank, 0};
}

TestRankSpace world_rank_space() {
  return TestRankSpace{Index<kDim>{}, Extent<kDim>{n_ranks(), 1}};
}

Extent<kDim> no_ghosts() {
  return Extent<kDim>{};
}

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

TestLayout dot_boxes(int cell_count) {
  std::vector<TestBox> boxes;
  boxes.reserve(static_cast<std::size_t>(cell_count));
  for (int i = 0; i < cell_count; ++i)
    boxes.push_back(TestBox{Index<kDim>{i, 0}, Index<kDim>{i, 0}});
  return TestLayout(std::move(boxes));
}

TestDistribution round_robin_mapping(const TestLayout& boxes) {
  std::vector<Index<kDim>> owners;
  owners.reserve(boxes.size());
  const int rank_count = n_ranks();
  for (std::size_t i = 0; i < boxes.size(); ++i)
    owners.push_back(rank_coordinate(static_cast<int>(i) % rank_count));
  return TestDistribution::partitioned(boxes, world_rank_space(), std::move(owners));
}

struct DotFields {
  TestMultiFab left;
  TestMultiFab right;
};

DotFields make_fields(int cell_count, int components = 1) {
  TestLayout boxes = dot_boxes(cell_count);
  TestDistribution mapping = round_robin_mapping(boxes);
  DotFields fields{
      TestMultiFab(boxes, mapping, rank_coordinate(my_rank()), components, no_ghosts()),
      TestMultiFab(boxes, mapping, rank_coordinate(my_rank()), components, no_ghosts())};
  fields.left.set_val(Real(0));
  fields.right.set_val(Real(0));
  return fields;
}

DotFields make_replicated_fields(int cell_count, int components = 1) {
  TestLayout boxes = dot_boxes(cell_count);
  TestDistribution mapping = TestDistribution::replicated(boxes, world_rank_space());
  DotFields fields{
      TestMultiFab(boxes, mapping, rank_coordinate(my_rank()), components, no_ghosts()),
      TestMultiFab(boxes, mapping, rank_coordinate(my_rank()), components, no_ghosts())};
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
  TestLayout boxes = dot_boxes(cell_count);
  TestDistribution mapping = TestDistribution::partitioned(
      boxes, world_rank_space(), std::vector<Index<kDim>>(boxes.size(), rank_coordinate(0)));
  DotFields fields{
      TestMultiFab(boxes, mapping, rank_coordinate(my_rank()), components, no_ghosts()),
      TestMultiFab(boxes, mapping, rank_coordinate(my_rank()), components, no_ghosts())};
  fields.left.set_val(Real(0));
  fields.right.set_val(Real(0));
  return fields;
}

void set_global_cell(TestMultiFab& field, int global, Real value, int component = 0) {
  const std::size_t local = field.local_index_of(static_cast<std::size_t>(global));
  if (local == TestMultiFab::not_local)
    return;
  const TestBox box = field.box(local);
  field.fab(local).view()(box.lo, component) = value;
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
      PureFieldAlgebra::dot(fields.left, fields.right, TestVectorDistribution::Replicated));
  expect_close_to_one(detail::PreparedFieldAlgebra::dot(fields.left, fields.right,
                                                        TestVectorDistribution::Replicated));
}

TEST(test_pure_field_algebra_extreme_dot, ProviderHandleRejectsInvalidNativeDescriptor) {
  EXPECT_THROW((void)TestVectorDistribution(static_cast<FieldDistribution>(255)),
               std::invalid_argument);
}

TEST(test_pure_field_algebra_extreme_dot, PublicReplicaOverloadsRejectPartialRankLayouts) {
  if (n_ranks() == 1)
    GTEST_SKIP() << "a serial mapping is necessarily a complete local replica";
  DotFields fields = make_fields(2);
  EXPECT_THROW(
      (void)PureFieldAlgebra::dot(fields.left, fields.right, TestVectorDistribution::Replicated),
      std::invalid_argument);
  EXPECT_THROW((void)PureFieldAlgebra::norm(fields.left, TestVectorDistribution::Replicated),
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
  EXPECT_THROW((void)PureFieldAlgebra::max_abs(fields.left, TestVectorDistribution::Replicated),
               std::invalid_argument);
  EXPECT_THROW(
      (void)PureFieldAlgebra::dot(fields.left, fields.right, TestVectorDistribution::Replicated),
      std::invalid_argument);
  EXPECT_THROW((void)PureFieldAlgebra::norm(fields.left, TestVectorDistribution::Replicated),
               std::invalid_argument);
}

TEST(test_pure_field_algebra_extreme_dot, PublicOwnershipMustAgreeAcrossRanks) {
  if (n_ranks() == 1)
    GTEST_SKIP() << "ownership descriptors cannot disagree in serial";
  DotFields fields = make_replicated_fields(2);
  const TestVectorDistribution ownership =
      my_rank() == 0 ? TestVectorDistribution::Distributed : TestVectorDistribution::Replicated;
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
  EXPECT_THROW((void)PureFieldAlgebra::norm(fields.left, TestVectorDistribution::Replicated),
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
  all_reduce_sum_inplace(payload.data(), payload.size());
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
