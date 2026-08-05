#include <gtest/gtest.h>

#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/parallel/comm.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

using pops::BoundaryTopology;
using pops::Box;
using pops::Extent;
using pops::Geometry;
using pops::Index;
using pops::PhysicalBoundaryConditions;
using pops::PhysicalBoundaryFace;
using pops::PhysicalBoundaryKind;
using pops::Real;
using pops::RealVector;
using pops::amr::RefinementRatio;
using pops::elliptic::mg::CompositeFacBuildRequest;
using pops::elliptic::mg::CompositeFacPoisson;
using pops::elliptic::mg::GeometricMG;
using pops::elliptic::mg::GeometricMultigridOptions;
using pops::mesh::BoxArray;
using pops::mesh::Distribution;
using pops::mesh::RankSpace;

template <int Dim, class Value>
auto filled(Value value) {
  std::array<Value, Dim> result{};
  result.fill(value);
  return result;
}

template <int Dim>
Extent<Dim> extent(std::int64_t value) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
Index<Dim> index(int value) {
  Index<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::EllipticBuildRequest<Dim> request(int cells, BoxArray<Dim> boxes) {
  const Box<Dim> domain{index<Dim>(0), index<Dim>(cells - 1)};
  const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(
      domain, RealVector<Dim>{}, [&] {
        RealVector<Dim> upper{};
        for (int axis = 0; axis < Dim; ++axis)
          upper[axis] = Real(1);
        return upper;
      }());
  Extent<Dim> rank_extent = extent<Dim>(1);
  rank_extent[0] = pops::n_ranks();
  Index<Dim> local_rank{};
  local_rank[0] = pops::my_rank();
  const RankSpace<Dim> ranks{Index<Dim>{}, rank_extent};
  const Distribution<Dim> distribution = Distribution<Dim>::replicated(boxes, ranks);
  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  faces.fill(PhysicalBoundaryFace{PhysicalBoundaryKind::dirichlet, Real(0)});
  RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  const std::size_t pairs = boxes.size() * (boxes.size() - 1) / 2;
  return {geometry,
          std::move(boxes),
          distribution,
          local_rank,
          PhysicalBoundaryConditions<Dim>{BoundaryTopology<Dim>::physical(), faces, spacing},
          Extent<Dim>{},
          extent<Dim>(1),
          {distribution.box_count(), pairs}};
}

template <int Dim>
pops::EllipticBuildRequest<Dim> complete_request(int cells) {
  const Box<Dim> domain{index<Dim>(0), index<Dim>(cells - 1)};
  return request<Dim>(cells, BoxArray<Dim>{std::vector<Box<Dim>>{domain}});
}

template <int Dim>
void expect_geometric_cycle() {
  GeometricMultigridOptions options;
  options.relative_tolerance = Real(1e-7);
  options.absolute_tolerance = Real(1e-10);
  options.maximum_cycles = 80;
  options.bottom_sweeps = 40;
  GeometricMG<Dim> solver(complete_request<Dim>(8), options);
  EXPECT_GE(solver.num_levels(), 2);
  solver.phi().set_val(Real(0));
  solver.rhs().set_val(Real(1));
  const pops::SolveReport report = solver.solve();
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_GT(report.iters, 0);
  EXPECT_LT(report.rel_residual, Real(1e-7));
}

template <int Dim>
void expect_partial_composite_preparation() {
  auto coarse = complete_request<Dim>(8);
  const Box<Dim> fine_patch{index<Dim>(4), index<Dim>(11)};
  auto fine = request<Dim>(16, BoxArray<Dim>{std::vector<Box<Dim>>{fine_patch}});
  CompositeFacBuildRequest<Dim> hierarchy{{std::move(coarse), std::move(fine)},
                                           {RefinementRatio<Dim>{filled<Dim>(2)}}};
  CompositeFacPoisson<Dim> solver(std::move(hierarchy));
  EXPECT_EQ(solver.n_levels(), 2);
  solver.phi_level(0).set_val(Real(0));
  solver.phi_level(1).set_val(Real(0));
  solver.rhs_level(0).set_val(Real(0));
  solver.rhs_level(1).set_val(Real(0));
  const pops::SolveReport report = solver.solve();
  EXPECT_TRUE(report.solved()) << report.reason;
  EXPECT_EQ(report.iters, 0);
}

}  // namespace

TEST(test_geometric_mg_nd, one_algorithm_executes_true_cycles_in_one_two_and_three_dimensions) {
  expect_geometric_cycle<1>();
  expect_geometric_cycle<2>();
  expect_geometric_cycle<3>();
}

TEST(test_geometric_mg_nd, partial_composite_hierarchies_prepare_in_exact_rank) {
  expect_partial_composite_preparation<1>();
  expect_partial_composite_preparation<2>();
  expect_partial_composite_preparation<3>();
}

TEST(test_geometric_mg_nd, unsupported_operator_families_fail_closed_at_capability_selection) {
  constexpr auto mg = GeometricMG<3>::capabilities();
  EXPECT_TRUE(mg.scalar_constant_coefficient);
  EXPECT_TRUE(mg.scalar_reaction);
  EXPECT_FALSE(mg.variable_diagonal);
  EXPECT_FALSE(mg.cross_tensor);
  EXPECT_FALSE(mg.embedded_boundary);

  constexpr auto fac = CompositeFacPoisson<3>::capabilities();
  EXPECT_TRUE(fac.partial_refinement);
  EXPECT_TRUE(fac.arbitrary_level_count);
  EXPECT_FALSE(fac.distributed_mpi);
  EXPECT_FALSE(fac.variable_diagonal);
  EXPECT_FALSE(fac.cross_tensor);
  EXPECT_FALSE(fac.embedded_boundary);
}
