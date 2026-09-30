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
  ExecutionLane lane =
      ExecutionLane::duplicate_world_collectively("test.exchange-batch.boundary-interior");
  AcceptedExchangeLedger ledger;
  const Field* active_override = nullptr;
  const auto& geometry() const { return geometry_; }
  const auto& prepared_execution_lane() const { return lane; }
  auto prepare_mesh_boundary_session(Field& field, const ExecutionLane& execution) {
    return PreparedScalarBoundarySession<1>::prepare(geometry_, BoundaryTopology<1>::physical(),
                                                     field, execution, 1);
  }
  const Field* pointwise_active_mask(int, const Field& field) const {
    if (active_override != nullptr && active_override->layout() != field.layout())
      throw std::invalid_argument("rank-local active mask layout differs from state");
    return active_override;
  }
  const Field* pointwise_exchange_coverage_mask(int, const Field&) const { return nullptr; }
  bool is_external_trace_face(int axis, int side, const Index<1>& cell) const {
    return cell[axis] == (side == 0 ? geometry_.domain().lo[axis] : geometry_.domain().hi[axis]);
  }
  template <class Producer>
  void stage_exchange_batch(Producer&& producer) {
    auto records = prepare_exchange_batch(
        std::forward<Producer>(producer),
        [](auto& record) { record.source_evaluation_identity = record.evaluation_context; }, lane);
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
  for (const auto& row : context.ledger.records()) {
    EXPECT_TRUE(row.exterior_trace);
    EXPECT_EQ(row.trace_axis, 0);
    EXPECT_EQ(row.trace_side, my_rank() == 0 ? 0 : 1);
    EXPECT_EQ(row.trace_component, 0);
    EXPECT_EQ(row.source_evaluation_identity, "accepted");
  }
  const auto selected =
      context.ledger.prepare_trace({"diffusion", "physical", 0, 1, 0, "accepted"});
  EXPECT_EQ(all_reduce_sum(static_cast<long>(selected.indices.size()), context.lane), 1);
  EXPECT_NEAR(all_reduce_sum(selected.local_amount, context.lane), .0005, 1e-14);
  const auto before = context.ledger.checkpoint();
  const Real state_before = integral(q, context.lane), rhs_before = integral(rhs, context.lane);

  // The interior-only rank rejects a real carrier layout mismatch after the
  // common preparation. Boundary owners must reach the same refusal vote even
  // though their coverage lookup succeeded and they could stage face records.
  const auto wrong_boxes = mesh::BoxArray<1>::from_domain(
      Box<1>{Index<1>{0}, Index<1>{15}}, Extent<1>{4});
  const auto wrong_distribution = mesh::Distribution<1>::partitioned(
      wrong_boxes, mesh::RankSpace<1>{Index<1>{0}, Extent<1>{3}},
      std::vector<Index<1>>{Index<1>{0}, Index<1>{1}, Index<1>{2}, Index<1>{0}});
  Field wrong_mask(wrong_boxes, wrong_distribution, Index<1>{my_rank()}, 1, Extent<1>{1});
  if (my_rank() == 1)
    context.active_override = &wrong_mask;
  bool refused = false;
  std::string refusal;
  try {
    prepared.stage_accepted_exchanges(context, 0, "diffusion", "wrong-layout", "accepted", .05,
                                      true);
  } catch (const std::invalid_argument& error) {
    refused = true;
    refusal = error.what();
  }
  context.active_override = nullptr;
  EXPECT_EQ(all_reduce_sum(refused ? 1L : 0L, context.lane), 3);
  EXPECT_EQ(refusal, "accepted diffusive face mask preparation failed collectively");
  EXPECT_EQ(context.ledger.checkpoint(), before);
  EXPECT_EQ(integral(q, context.lane), state_before);
  EXPECT_EQ(integral(rhs, context.lane), rhs_before);

  // The interior rank produces no faces, but still joins preparation failure consensus.
  refused = false;
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

  // A rejection authored by one producer retains its typed control envelope on every rank.
  bool typed_rejection = false;
  try {
    context.stage_exchange_batch([&](auto&& stage) {
      if (my_rank() == 0)
        stage(record("prepared-before-rejection"));
      if (my_rank() == 1)
        throw StepAttemptRejected(SolveStatus::kInvalidEvaluation, StepAttemptDisposition::kReject,
                                  0x45584241u, "exchange-producer", "retry this attempt");
    });
  } catch (const StepAttemptRejected& rejected) {
    typed_rejection = true;
    EXPECT_EQ(rejected.status(), SolveStatus::kInvalidEvaluation);
    EXPECT_EQ(rejected.disposition(), StepAttemptDisposition::kReject);
    EXPECT_EQ(rejected.reason_code(), 0x45584241u);
    EXPECT_EQ(rejected.phase(), "exchange-producer");
    EXPECT_EQ(rejected.detail(), "retry this attempt");
  }
  EXPECT_EQ(all_reduce_sum(typed_rejection ? 1L : 0L, context.lane), 3);
  EXPECT_EQ(context.ledger.checkpoint(), before);
  EXPECT_EQ(integral(q, context.lane), state_before);
  EXPECT_EQ(integral(rhs, context.lane), rhs_before);

  for (const bool duplicate : {false, true}) {
    refused = false;
    try {
      context.stage_exchange_batch([&](auto&& stage) {
        if (my_rank() == 2)
          stage(record("peer-retryable"));
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
    if (my_rank() == 2)
      stage(record("peer-retryable"));
  });
  EXPECT_EQ(context.ledger.records().size(), my_rank() == 1 ? 0u : 2u);
}
