#include <gtest/gtest.h>
#include <pops/numerics/diffusion/prepared_diffusion.hpp>
#include <pops/parallel/comm.hpp>

namespace {
using namespace pops;
using namespace pops::runtime::program;
using Field = MultiFab<1>;
class CommEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override { comm_init(); }
  void TearDown() override { comm_finalize(); }
};
[[maybe_unused]] const auto* environment = ::testing::AddGlobalTestEnvironment(new CommEnvironment);

struct Context {
  Geometry<1> geometry_ = Geometry<1>::from_bounds(Box<1>{Index<1>{0}, Index<1>{11}},
                                                   RealVector<1>{0}, RealVector<1>{1});
  ExecutionLane lane = ExecutionLane::world("test.exchange-batch.boundary-interior");
  AcceptedExchangeLedger ledger;
  const auto& geometry() const { return geometry_; }
  const auto& prepared_execution_lane() const { return lane; }
  auto prepare_mesh_boundary_session(Field& field, const ExecutionLane& execution) {
    return PreparedScalarBoundarySession<1>::prepare(geometry_, BoundaryTopology<1>::physical(),
                                                     field, execution, 1);
  }
  const Field* pointwise_active_mask(int, const Field&) const { return nullptr; }
  template <class Producer>
  void stage_exchange_batch(Producer&& producer) {
    auto records = prepare_exchange_batch(std::forward<Producer>(producer), [](auto&) {}, lane);
    stage_exchange_batch_collectively(ledger, records, lane);
  }
};

Field field() {
  const auto boxes =
      mesh::BoxArray<1>::from_domain(Box<1>{Index<1>{0}, Index<1>{11}}, Extent<1>{4});
  const mesh::RankSpace<1> ranks{Index<1>{0}, Extent<1>{3}};
  const auto distribution = mesh::Distribution<1>::partitioned(
      boxes, ranks, std::vector<Index<1>>{Index<1>{0}, Index<1>{1}, Index<1>{2}});
  return Field(boxes, distribution, Index<1>{my_rank()}, 1, Extent<1>{1});
}
Real integral(const Field& field, const ExecutionLane& lane) {
  Real local = 0;
  for (std::size_t patch = 0; patch < field.local_size(); ++patch) {
    const auto values = field.fab(patch).view();
    local += for_each_cell_reduce_sum(
        field.box(patch), [=] POPS_HD(const Index<1>& cell) { return values(cell, 0) / Real(12); });
  }
  return all_reduce_sum(local, lane);
}
ExchangeRecord record(std::string quadrature) {
  return {"test.batch", "occurrence", "attempt", std::move(quadrature), 1, 1, .02, .05, 1};
}
}  // namespace

TEST(ExchangeBatches, UnequalPhysicalBoundaryOwnershipConservesAndFailuresRollbackCollectively) {
  if (n_ranks() != 3)
    GTEST_SKIP() << "requires two boundary owners and one interior-only rank";
  Context context;
  auto q = field(), rhs = field();
  q.set_val(1);
  const std::array<DiffusiveBoundary<1>, 2> physical{
      DiffusiveBoundary<1>{DiffusiveBoundaryKind::conormal, .01, {}},
      DiffusiveBoundary<1>{DiffusiveBoundaryKind::conormal, .01, {}}};
  PreparedDiffusion<1> prepared(context, q, physical);
  prepared.apply(q, rhs, [&](std::size_t local) {
    const auto values = std::as_const(q).fab(local).view();
    return
        [=] POPS_HD(const Index<1>& cell) { return std::array<Real, 3>{values(cell, 0), .1, 1}; };
  });
  prepared.stage_accepted_exchanges(context, 0, "diffusion", "physical", "accepted", .05, true);
  EXPECT_EQ(context.ledger.records().size(), my_rank() == 1 ? 0u : 1u);
  Real local_amount = 0;
  for (const auto& row : context.ledger.records())
    local_amount += row.integrated_amount();
  const Real amount = all_reduce_sum(local_amount, context.lane);
  EXPECT_NEAR(amount, .001, 1e-14);
  EXPECT_NEAR(.05 * integral(rhs, context.lane), amount, 1e-14);
  const auto before = context.ledger.checkpoint();
  const Real state_before = integral(q, context.lane), rhs_before = integral(rhs, context.lane);

  // The interior rank produces no faces, but still joins preparation failure consensus.
  bool refused = false;
  try {
    context.stage_exchange_batch([&](auto&& stage) {
      if (my_rank() == 0)
        stage(record("prepared-peer"));
      if (my_rank() == 1)
        throw std::invalid_argument("rank-local producer failure");
    });
  } catch (const std::exception&) {
    refused = true;
  }
  EXPECT_EQ(all_reduce_sum(refused ? 1L : 0L, context.lane), 3);
  EXPECT_EQ(context.ledger.checkpoint(), before);

  for (const bool duplicate : {false, true}) {
    refused = false;
    try {
      context.stage_exchange_batch([&](auto&& stage) {
        if (my_rank() == 0) {
          stage(record("retryable"));
          auto bad = record(duplicate ? "retryable" : "invalid");
          if (!duplicate)
            bad.face_measure = -1;
          stage(std::move(bad));
        }
      });
    } catch (const std::exception&) {
      refused = true;
    }
    EXPECT_EQ(all_reduce_sum(refused ? 1L : 0L, context.lane), 3);
    EXPECT_EQ(context.ledger.checkpoint(), before);
    EXPECT_EQ(integral(q, context.lane), state_before);
    EXPECT_EQ(integral(rhs, context.lane), rhs_before);
  }
  // Failure published no keys: the same quadrature can be retried successfully.
  context.stage_exchange_batch([&](auto&& stage) {
    if (my_rank() == 0)
      stage(record("retryable"));
  });
  EXPECT_EQ(context.ledger.records().size(), my_rank() == 0 ? 2u : (my_rank() == 1 ? 0u : 1u));
}
