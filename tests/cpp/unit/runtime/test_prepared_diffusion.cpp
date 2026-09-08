#include <gtest/gtest.h>
#include <pops/numerics/diffusion/prepared_diffusion.hpp>

namespace {
using namespace pops;
using namespace pops::runtime::program;
using Field = MultiFab<2>;
struct DiffusionContext {
  Geometry<2> geometry_ = Geometry<2>::from_bounds(Box<2>{Index<2>{0, 0}, Index<2>{3, 3}},
                                                   RealVector<2>{0, 0}, RealVector<2>{1, 1});
  BoundaryTopology<2> topology = BoundaryTopology<2>::axis_periodic({true, true});
  ExecutionLane lane = ExecutionLane::world("prepared-diffusion-test");
  AcceptedExchangeLedger ledger;
  const auto& geometry() const { return geometry_; }
  const auto& prepared_execution_lane() const { return lane; }
  auto prepare_mesh_boundary_session(Field& field, const ExecutionLane& execution) {
    return PreparedScalarBoundarySession<2>::prepare(geometry_, topology, field, execution, 1);
  }
  const Field* pointwise_active_mask(int, const Field&) const { return nullptr; }
  template <class Producer>
  void stage_exchange_batch(Producer&& producer) {
    auto records = prepare_exchange_batch(std::forward<Producer>(producer), [](auto&) {}, lane);
    stage_exchange_batch_collectively(ledger, records, lane);
  }
};
Field field(Box<2> box = Box<2>{Index<2>{0, 0}, Index<2>{3, 3}},
            Extent<2> ghosts = Extent<2>{1, 1}) {
  auto layout = mesh::BoxArray<2>::from_domain(box, Extent<2>{4, 4});
  auto distribution = mesh::Distribution<2>::replicated(
      layout, mesh::RankSpace<2>{Index<2>{0, 0}, Extent<2>{1, 1}});
  return Field(layout, distribution, Index<2>{0, 0}, 1, ghosts);
}
Field split_field() {
  auto layout =
      mesh::BoxArray<2>::from_domain(Box<2>{Index<2>{0, 0}, Index<2>{3, 3}}, Extent<2>{2, 4});
  auto distribution = mesh::Distribution<2>::replicated(
      layout, mesh::RankSpace<2>{Index<2>{0, 0}, Extent<2>{1, 1}});
  return Field(layout, distribution, Index<2>{0, 0}, 1, Extent<2>{1, 1});
}
auto identity_law(Field& q) {
  return [&q](std::size_t local) {
    const auto values = std::as_const(q).fab(local).view();
    return [=] POPS_HD(const Index<2>& cell) {
      return std::array<Real, 4>{values(cell, 0), .1, .1, 1};
    };
  };
}
}  // namespace

TEST(PreparedDiffusion, RejectsSameSizedForeignLayoutAndGhostsBeforeWriting) {
  DiffusionContext context;
  auto q = field(), output = field(), foreign = field(Box<2>{Index<2>{4, 0}, Index<2>{7, 3}}),
       ghost = field(Box<2>{Index<2>{0, 0}, Index<2>{3, 3}}, Extent<2>{2, 1});
  PreparedDiffusion<2> prepared(context, q, {});
  output.set_val(17);
  const auto layout = q.layout();
  auto other_distribution = mesh::Distribution<2>::partitioned(
      layout, mesh::RankSpace<2>{Index<2>{0, 0}, Extent<2>{1, 1}}, {Index<2>{0, 0}});
  Field partitioned(layout, other_distribution, Index<2>{0, 0}, 1, Extent<2>{1, 1});
  auto two_ranks = mesh::Distribution<2>::replicated(
      layout, mesh::RankSpace<2>{Index<2>{0, 0}, Extent<2>{2, 1}});
  Field other_rank(layout, two_ranks, Index<2>{1, 0}, 1, Extent<2>{1, 1});
  EXPECT_THROW(prepared.apply(foreign, output, identity_law(foreign)), std::invalid_argument);
  EXPECT_THROW(prepared.apply(q, foreign, identity_law(q)), std::invalid_argument);
  EXPECT_THROW(prepared.apply(ghost, output, identity_law(ghost)), std::invalid_argument);
  EXPECT_THROW(prepared.apply(q, ghost, identity_law(q)), std::invalid_argument);
  EXPECT_THROW(prepared.apply(q, partitioned, identity_law(q)), std::invalid_argument);
  EXPECT_THROW(prepared.apply(other_rank, output, identity_law(other_rank)), std::invalid_argument);
  EXPECT_DOUBLE_EQ(reduce_max_local(output), 17);
}

TEST(PreparedDiffusion, RejectsNegativeFaceSecantEvenWhenSampleDerivativesArePositive) {
  DiffusionContext context;
  auto q = field(), output = field();
  const auto values = q.fab(0).view();
  for_each_cell(q.box(0),
                [=] POPS_HD(const Index<2>& cell) { values(cell, 0) = cell[0] == 0 ? -1 : 1; });
  PreparedDiffusion<2> prepared(context, q, {});
  auto law = [&](std::size_t local) {
    const auto values = std::as_const(q).fab(local).view();
    return [=] POPS_HD(const Index<2>& cell) {
      const Real u = values(cell, 0);
      return std::array<Real, 4>{u * u * u - 2 * u, .1, .1, 3 * u * u - 2};
    };
  };
  EXPECT_THROW(prepared.apply(q, output, law), DiffusiveEvaluationError);
  EXPECT_THROW(prepared.explicit_frequency(), std::logic_error);
  EXPECT_TRUE(context.ledger.records().empty());
}

TEST(PreparedDiffusion, RejectsFailedConstitutiveEvaluationInPreparedInternalGhost) {
  DiffusionContext context;
  auto q = split_field(), output = split_field();
  PreparedDiffusion<2> prepared(context, q, {}, true);
  try {
    prepared.apply(q, output, [](std::size_t local) {
      return [=] POPS_HD(const Index<2>& cell) {
        DiffusiveLawResult<2> result;
        result.values = {1, .1, .1, 1};
        if (local == 0 && cell[0] == 2) {
          result.evaluation_status = 2;
          result.reason_code = 777;
        }
        return result;
      };
    });
    EXPECT_TRUE(false);
  } catch (const DiffusiveEvaluationError& error) {
    EXPECT_EQ(error.status(), 2);
    EXPECT_EQ(error.reason(), 777);
  }
  EXPECT_THROW(prepared.explicit_frequency(), std::logic_error);
}

TEST(PreparedDiffusion, VariableDiagonalStaysInsideDivergenceWithPhysicalValueTrace) {
  DiffusionContext context;
  context.topology = BoundaryTopology<2>::physical();
  auto q = field(), output = field();
  const auto values = q.fab(0).view();
  const auto geometry = context.geometry();
  for_each_cell(q.box(0), [=] POPS_HD(const Index<2>& cell) {
    values(cell, 0) =
        1 + geometry.cell_coordinate(0, cell[0]) + geometry.cell_coordinate(1, cell[1]);
  });
  std::array<DiffusiveBoundary<2>, 4> physical{};
  for (auto& row : physical)
    row = {DiffusiveBoundaryKind::value, 1, {1, 1}};
  PreparedDiffusion<2> prepared(context, q, physical);
  prepared.apply(q, output, [&](std::size_t local) {
    const auto values = std::as_const(q).fab(local).view();
    return [=] POPS_HD(const Index<2>& cell) {
      return std::array<Real, 4>{values(cell, 0), 1 + .25 * geometry.cell_coordinate(0, cell[0]),
                                 2 + .5 * geometry.cell_coordinate(1, cell[1]), 1};
    };
  });
  const auto result = std::as_const(output).fab(0).view();
  EXPECT_NEAR(for_each_cell_reduce_max(
                  output.box(0),
                  [=] POPS_HD(const Index<2>& cell) { return Kokkos::abs(result(cell, 0) - .75); }),
              0, 2e-13);
  prepared.stage_accepted_exchanges(context, 0, "operator", "occurrence", "stage0", .05);
  Real exchange = 0;
  for (const auto& record : context.ledger.records())
    exchange += record.integrated_amount();
  EXPECT_EQ(context.ledger.records().size(), 64);
  EXPECT_NEAR(exchange, .05 * .75, 2e-13);
  context.ledger.clear();
  prepared.stage_accepted_exchanges(context, 0, "operator", "physical-boundary", "stage0", .05,
                                    true);
  exchange = 0;
  for (const auto& record : context.ledger.records())
    exchange += record.integrated_amount();
  EXPECT_EQ(context.ledger.records().size(), 16);
  EXPECT_NEAR(exchange, .05 * .75, 2e-13);
}

namespace {
struct FittedContext {
  Geometry<1> geometry_ = Geometry<1>::from_bounds(Box<1>{Index<1>{0}, Index<1>{31}},
                                                   RealVector<1>{0}, RealVector<1>{1});
  BoundaryTopology<1> topology = BoundaryTopology<1>::axis_periodic({true});
  ExecutionLane lane = ExecutionLane::world("fitted-diffusion-test");
  AcceptedExchangeLedger ledger;
  const auto& geometry() const { return geometry_; }
  const auto& prepared_execution_lane() const { return lane; }
  auto prepare_mesh_boundary_session(MultiFab<1>& field, const ExecutionLane& execution) {
    return PreparedScalarBoundarySession<1>::prepare(geometry_, topology, field, execution, 1);
  }
  const MultiFab<1>* pointwise_active_mask(int, const MultiFab<1>&) const { return nullptr; }
  template <class Producer>
  void stage_exchange_batch(Producer&& producer) {
    auto records = prepare_exchange_batch(std::forward<Producer>(producer), [](auto&) {}, lane);
    stage_exchange_batch_collectively(ledger, records, lane);
  }
};
MultiFab<1> fitted_field() {
  auto layout = mesh::BoxArray<1>::from_domain(Box<1>{Index<1>{0}, Index<1>{31}}, Extent<1>{32});
  auto distribution =
      mesh::Distribution<1>::replicated(layout, mesh::RankSpace<1>{Index<1>{0}, Extent<1>{1}});
  return MultiFab<1>(layout, distribution, Index<1>{0}, 1, Extent<1>{1});
}
}  // namespace

TEST(PreparedDiffusion, BernoulliRetainsZeroJumpAndLargeDriftLimits) {
  EXPECT_DOUBLE_EQ(scharfetter_gummel_bernoulli(0), 1);
  EXPECT_NEAR(scharfetter_gummel_bernoulli(1), .58197670686932642439, 2e-15);
  for (Real z : std::array<Real, 5>{1e-12, 1e-6, 1, 100, 1000}) {
    EXPECT_NEAR(scharfetter_gummel_bernoulli(-z) - scharfetter_gummel_bernoulli(z), z, 2e-13);
    EXPECT_TRUE(std::isfinite(scharfetter_gummel_bernoulli(z)));
    EXPECT_TRUE(scharfetter_gummel_bernoulli(z) >= 0);
  }
  EXPECT_NEAR(scharfetter_gummel_bernoulli(100), 3.72007597602083596296e-42, 1e-55);
}

TEST(PreparedDiffusion, FittedZeroPotentialIsTheSameConservativeDiffusion) {
  FittedContext context;
  auto q = fitted_field(), out = fitted_field();
  const auto values = q.fab(0).view();
  for_each_cell(q.box(0), [=] POPS_HD(const Index<1>& cell) {
    values(cell, 0) = 2 + Kokkos::cos(2 * Real(3.14159265358979323846) * (cell[0] + Real(.5)) / 32);
  });
  PreparedDiffusion<1> prepared(context, q, {});
  prepared.apply_fitted(q, out,
                        [](std::size_t) {
                          return
                              [] POPS_HD(const Index<1>&) { return std::array<Real, 3>{0, .1, 0}; };
                        },
                        1, {});
  const auto result = std::as_const(out).fab(0).view();
  EXPECT_NEAR(for_each_cell_reduce_max(
                  out.box(0),
                  [=] POPS_HD(const Index<1>& cell) {
                    const Index<1> left{(cell[0] + 31) % 32}, right{(cell[0] + 1) % 32};
                    const Real expected =
                        .1 * 32 * 32 * (values(right, 0) - 2 * values(cell, 0) + values(left, 0));
                    return Kokkos::abs(result(cell, 0) - expected);
                  }),
              0, 2e-13);
  EXPECT_NEAR(prepared.explicit_frequency(), 2 * .1 * 32 * 32, 2e-13);
}

TEST(PreparedDiffusion, FittedExponentialEquilibriumHasZeroOrientedFaceExchange) {
  FittedContext context;
  auto q = fitted_field(), out = fitted_field();
  const auto values = q.fab(0).view();
  for_each_cell(q.box(0), [=] POPS_HD(const Index<1>& cell) {
    const Real phi = .4 * Kokkos::cos(2 * Real(3.14159265358979323846) * (cell[0] + Real(.5)) / 32);
    values(cell, 0) = Kokkos::exp(-phi);
  });
  PreparedDiffusion<1> prepared(context, q, {});
  prepared.apply_fitted(q, out,
                        [](std::size_t) {
                          return [] POPS_HD(const Index<1>& cell) {
                            const Real phi = .4 * Kokkos::cos(2 * Real(3.14159265358979323846) *
                                                              (cell[0] + Real(.5)) / 32);
                            return std::array<Real, 3>{phi, .1, 0};
                          };
                        },
                        1, {});
  prepared.stage_accepted_exchanges(context, 0, "fitted", "joint-drift-diffusion", "stage0", .01);
  EXPECT_EQ(context.ledger.records().size(), 64);
  for (const auto& record : context.ledger.records())
    EXPECT_NEAR(record.numerical_flux, 0, 2e-12);
  EXPECT_NEAR(reduce_max_local(out), 0, 2e-11);
}

TEST(PreparedDiffusion, FailedConstitutiveOutputsStayUnreadAndPreserveExactWorstReason) {
  FittedContext context;
  auto q = fitted_field(), out = fitted_field();
  out.set_val(17);
  PreparedDiffusion<1> prepared(context, q, {});
  try {
    prepared.apply(q, out, [](std::size_t) {
      return [] POPS_HD(const Index<1>& cell) {
        DiffusiveLawResult<1> result;
        result.values.fill(std::numeric_limits<Real>::quiet_NaN());
        result.evaluation_status = cell[0] % 2 ? 1 : 2;
        result.reason_code = cell[0] % 2 ? 900 : 23;
        return result;
      };
    });
    EXPECT_TRUE(false);
  } catch (const DiffusiveEvaluationError& error) {
    EXPECT_EQ(error.status(), 2);
    EXPECT_EQ(error.reason(), 23);
  }
  EXPECT_DOUBLE_EQ(reduce_max_local(out), 17);
  EXPECT_TRUE(context.ledger.records().empty());
  EXPECT_THROW(prepared.explicit_frequency(), std::logic_error);
}
