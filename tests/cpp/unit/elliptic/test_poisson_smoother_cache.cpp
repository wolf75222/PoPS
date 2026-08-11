#include <gtest/gtest.h>

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/parallel/comm.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kDim = pops::kNativeDimension;
using Solver = pops::elliptic::mg::GeometricMG<kDim>;

template <class Ranked, class Value>
Ranked filled(Value value) {
  Ranked result{};
  for (int axis = 0; axis < kDim; ++axis)
    result[axis] = value;
  return result;
}

pops::EllipticBuildRequest<kDim> request(int cells) {
  const pops::Box<kDim> domain{pops::Index<kDim>{}, filled<pops::Index<kDim>>(cells - 1)};
  const auto geometry = pops::Geometry<kDim>::from_bounds(
      domain, pops::RealVector<kDim>{}, filled<pops::RealVector<kDim>>(pops::Real(1)));
  const pops::mesh::BoxArray<kDim> layout(std::vector<pops::Box<kDim>>{domain});
  pops::Extent<kDim> rank_extent = filled<pops::Extent<kDim>>(std::int64_t{1});
  rank_extent[0] = pops::n_ranks();
  pops::Index<kDim> local_rank{};
  local_rank[0] = pops::my_rank();
  const pops::mesh::RankSpace<kDim> ranks{pops::Index<kDim>{}, rank_extent};
  const auto distribution = pops::mesh::Distribution<kDim>::replicated(layout, ranks);
  std::array<pops::PhysicalBoundaryFace, 2 * kDim> faces{};
  faces.fill(pops::PhysicalBoundaryFace{pops::PhysicalBoundaryKind::dirichlet, pops::Real(0)});
  pops::RealVector<kDim> spacing{};
  for (int axis = 0; axis < kDim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  return {geometry,
          layout,
          distribution,
          local_rank,
          pops::PhysicalBoundaryConditions<kDim>{pops::BoundaryTopology<kDim>::physical(), faces,
                                                 spacing},
          pops::Extent<kDim>{},
          filled<pops::Extent<kDim>>(std::int64_t{1}),
          {layout.size(), 0}};
}

}  // namespace

TEST(test_poisson_smoother_cache,
     repeated_solves_reuse_one_immutable_prepared_multigrid_hierarchy) {
  pops::elliptic::mg::GeometricMultigridOptions options;
  options.relative_tolerance = pops::Real(1e-8);
  options.absolute_tolerance = pops::Real(1e-11);
  options.maximum_cycles = 100;
  options.bottom_sweeps = 50;
  auto build = request(32);
  const auto expected_contract = Solver::expected_operator_contract(build, options);
  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.poisson-smoother-cache");
  Solver solver(std::move(build), lane, options);
  solver.install_nullspace(pops::FieldNullspacePlan<kDim>{},
                           pops::PreparedVectorDistribution<kDim>::replicated());
  ASSERT_GE(solver.num_levels(), 2);
  const int prepared_levels = solver.num_levels();
  const std::string prepared_fingerprint(solver.prepared_operator_contract().exact_fingerprint());
  EXPECT_EQ(prepared_fingerprint, expected_contract.exact_fingerprint());

  solver.rhs().set_val(pops::Real(1));
  solver.phi().set_val(pops::Real(0));
  const pops::SolveReport first = solver.solve();
  ASSERT_TRUE(first.solved()) << first.reason;
  ASSERT_GT(first.iters, 0);
  EXPECT_EQ(solver.prepared_operator_contract().exact_fingerprint(), prepared_fingerprint);

  solver.phi().set_val(pops::Real(0));
  const pops::SolveReport second = solver.solve();
  ASSERT_TRUE(second.solved()) << second.reason;
  EXPECT_EQ(second.iters, first.iters);
  EXPECT_EQ(second.evaluations, first.evaluations);
  EXPECT_EQ(solver.num_levels(), prepared_levels);
  EXPECT_EQ(solver.prepared_operator_contract().exact_fingerprint(), prepared_fingerprint);
}
