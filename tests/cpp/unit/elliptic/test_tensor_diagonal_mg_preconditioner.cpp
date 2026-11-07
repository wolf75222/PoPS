#include <gtest/gtest.h>

#include <pops/numerics/elliptic/mg/tensor_diagonal_preconditioner.hpp>

#include <array>
#include <memory>
#include <vector>

namespace {

using Field = pops::MultiFab<2>;
using Real = pops::Real;

class CommEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override { pops::comm_init(); }
  void TearDown() override { pops::comm_finalize(); }
};
[[maybe_unused]] const auto* environment = ::testing::AddGlobalTestEnvironment(new CommEnvironment);

struct Domain {
  pops::Geometry<2> geometry;
  pops::mesh::BoxArray<2> boxes;
  pops::mesh::Distribution<2> distribution;
  pops::Index<2> rank;
  pops::PhysicalBoundaryConditions<2> boundary;
};

Domain domain() {
  const pops::Box<2> box{pops::Index<2>{0, 0}, pops::Index<2>{7, 7}};
  auto geometry =
      pops::Geometry<2>::from_bounds(box, pops::RealVector<2>{0, 0}, pops::RealVector<2>{1, 1});
  pops::mesh::BoxArray<2> boxes{std::vector<pops::Box<2>>{box}};
  const pops::mesh::RankSpace<2> ranks{
      pops::Index<2>{0, 0},
      pops::Extent<2>{static_cast<std::int64_t>(pops::n_ranks()), std::int64_t{1}}};
  auto distribution = pops::mesh::Distribution<2>::partitioned(
      boxes, ranks, std::vector<pops::Index<2>>{pops::Index<2>{0, 0}});
  std::array<pops::PhysicalBoundaryFace, 4> faces;
  faces.fill({pops::PhysicalBoundaryKind::dirichlet, Real(0)});
  pops::PhysicalBoundaryConditions<2> boundary{
      pops::BoundaryTopology<2>::physical(), faces,
      pops::RealVector<2>{geometry.spacing(0), geometry.spacing(1)}};
  return {geometry, boxes, distribution, pops::Index<2>{pops::my_rank(), 0}, boundary};
}

std::shared_ptr<Field> field(const Domain& dom, int components) {
  return std::make_shared<Field>(dom.boxes, dom.distribution, dom.rank, components,
                                 pops::Extent<2>{1, 1});
}

void tensor(Field& values, Real diagonal, Real cross) {
  for (std::size_t local = 0; local < values.local_size(); ++local) {
    const auto out = values.fab(local).view();
    pops::for_each_cell(values.box(local), [=] POPS_HD(const pops::Index<2>& cell) {
      out(cell, 0) = out(cell, 3) = diagonal;
      out(cell, 1) = cross;
      out(cell, 2) = -cross;
    });
  }
}

Real compare(const Field& first, const Field& second, Real scale, const pops::ExecutionLane& lane) {
  Real error = 0;
  for (std::size_t local = 0; local < first.local_size(); ++local) {
    const auto a = first.fab(local).view();
    const auto b = second.fab(local).view();
    error = std::max(error, pops::for_each_cell_reduce_max(
                                first.box(local), [=] POPS_HD(const pops::Index<2>& cell) {
                                  return Kokkos::abs(a(cell, 0) - scale * b(cell, 0));
                                }));
  }
  return pops::all_reduce_max(error, lane);
}

TEST(TensorDiagonalMGPreconditioner, RefreshesAtSnapshotAndDropsCrossTermsWithStableStorage) {
  const auto dom = domain();
  const auto lane = pops::ExecutionLane::world("tests.tensor-diagonal-mg.snapshot");
  auto prototype = field(dom, 1), coefficients = field(dom, 4), rhs = field(dom, 1);
  auto first = field(dom, 1), held = field(dom, 1), refreshed = field(dom, 1);
  rhs->set_val(Real(1));
  tensor(*coefficients, Real(1), Real(0.75));
  auto factory = pops::elliptic::mg::make_tensor_diagonal_mg_sessions<2>(
      prototype, coefficients, dom.geometry, dom.boundary,
      pops::PreparedVectorDistribution<2>::distributed());
  auto session = factory(lane);
  session.prepare();
  const auto count = session.allocation_count();
  EXPECT_EQ(count, 22u);  // levels 8,4,2; seven objects/level plus extracted diagonal.
  session.apply(*first, *rhs);
  tensor(*coefficients, Real(2), Real(10000));
  session.apply(*held, *rhs);
  EXPECT_EQ(compare(*held, *first, Real(1), lane), Real(0));
  session.prepare();
  session.apply(*refreshed, *rhs);
  EXPECT_EQ(compare(*refreshed, *first, Real(0.5), lane), Real(0));
  EXPECT_EQ(session.allocation_count(), count);
}

TEST(TensorDiagonalMGPreconditioner, RejectsMalformedCoefficientShapeCollectively) {
  const auto dom = domain();
  const auto lane = pops::ExecutionLane::world("tests.tensor-diagonal-mg.shape");
  auto prototype = field(dom, 1), malformed = field(dom, 3);
  auto factory = pops::elliptic::mg::make_tensor_diagonal_mg_sessions<2>(
      prototype, malformed, dom.geometry, dom.boundary,
      pops::PreparedVectorDistribution<2>::distributed());
  auto session = factory(lane);
  EXPECT_THROW(session.prepare(), std::invalid_argument);
}

TEST(TensorDiagonalMGPreconditioner, ZeroInputHasExactZeroOutputAndIndependentSessionStorage) {
  const auto dom = domain();
  const auto lane = pops::ExecutionLane::world("tests.tensor-diagonal-mg.zero");
  auto prototype = field(dom, 1), coefficients = field(dom, 4), rhs = field(dom, 1);
  auto output = field(dom, 1), zero = field(dom, 1);
  tensor(*coefficients, Real(1), Real(0));
  rhs->set_val(Real(0));
  zero->set_val(Real(0));
  auto factory = pops::elliptic::mg::make_tensor_diagonal_mg_sessions<2>(
      prototype, coefficients, dom.geometry, dom.boundary,
      pops::PreparedVectorDistribution<2>::distributed());
  auto session = factory(lane), other = factory(lane);
  session.prepare();
  other.prepare();
  output->set_val(Real(7));
  session.apply(*output, *rhs);
  EXPECT_EQ(compare(*output, *zero, Real(1), lane), Real(0));
  EXPECT_EQ(session.allocation_count(), other.allocation_count());
}

}  // namespace
