#include <gtest/gtest.h>

#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

template <int Dim>
pops::Index<Dim> index(int value) {
  pops::Index<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::Extent<Dim> extent(std::int64_t value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
std::array<int, Dim> ratio(int value) {
  std::array<int, Dim> result{};
  result.fill(value);
  return result;
}

template <int Dim>
pops::EllipticBuildRequest<Dim> request(const pops::Geometry<Dim>& geometry,
                                        pops::mesh::BoxArray<Dim> boxes) {
  pops::Extent<Dim> rank_extent = extent<Dim>(1);
  rank_extent[0] = pops::n_ranks();
  pops::Index<Dim> local_rank{};
  local_rank[0] = pops::my_rank();
  const pops::mesh::RankSpace<Dim> ranks{pops::Index<Dim>{}, rank_extent};
  const pops::mesh::Distribution<Dim> distribution =
      pops::mesh::Distribution<Dim>::replicated(boxes, ranks);
  std::array<pops::PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  faces.fill({pops::PhysicalBoundaryKind::dirichlet, pops::Real(0)});
  pops::RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  const std::size_t pairs = boxes.size() * (boxes.size() - 1) / 2;
  return {geometry,
          std::move(boxes),
          distribution,
          local_rank,
          {pops::BoundaryTopology<Dim>::physical(), faces, spacing},
          pops::Extent<Dim>{},
          extent<Dim>(1),
          {distribution.box_count(), pairs}};
}

template <int Dim>
pops::EllipticBuildRequest<Dim> complete_level(int cells) {
  const pops::Box<Dim> domain{index<Dim>(0), index<Dim>(cells - 1)};
  pops::RealVector<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = pops::Real(1);
  const pops::Geometry<Dim> geometry =
      pops::Geometry<Dim>::from_bounds(domain, pops::RealVector<Dim>{}, upper);
  return request<Dim>(geometry, pops::mesh::BoxArray<Dim>{std::vector<pops::Box<Dim>>{domain}});
}

template <int Dim>
void expect_zero_rhs_fac_solves_with_an_authenticated_ranked_contract() {
  auto coarse = complete_level<Dim>(8);
  const pops::Geometry<Dim> fine_geometry = coarse.geometry.refine(extent<Dim>(2));
  const pops::Box<Dim> fine_patch{index<Dim>(4), index<Dim>(11)};
  auto fine = request<Dim>(fine_geometry,
                           pops::mesh::BoxArray<Dim>{std::vector<pops::Box<Dim>>{fine_patch}});
  pops::elliptic::mg::CompositeFacBuildRequest<Dim> build{
      {std::move(coarse), std::move(fine)}, {pops::amr::RefinementRatio<Dim>{ratio<Dim>(2)}}};
  const std::string expected =
      pops::elliptic::mg::CompositeFacPoisson<Dim>::expected_prepared_contract(build);
  pops::elliptic::mg::CompositeFacPoisson<Dim> solver(std::move(build));
  EXPECT_EQ(solver.exact_prepared_contract(), expected);
  EXPECT_EQ(solver.n_levels(), 2);
  solver.install_nullspace(pops::FieldNullspacePlan<Dim>{},
                           {pops::PreparedVectorDistribution<Dim>::replicated(),
                            pops::PreparedVectorDistribution<Dim>::replicated()});
  solver.rhs_level(0).set_val(pops::Real(0));
  solver.rhs_level(1).set_val(pops::Real(0));
  const pops::SolveReport report = solver.solve();
  EXPECT_TRUE(report.solved()) << report.reason;
}

}  // namespace

TEST(test_composite_fac_poisson, ranked_hierarchy_prepares_and_solves_in_1d_2d_3d) {
  expect_zero_rhs_fac_solves_with_an_authenticated_ranked_contract<1>();
  expect_zero_rhs_fac_solves_with_an_authenticated_ranked_contract<2>();
  expect_zero_rhs_fac_solves_with_an_authenticated_ranked_contract<3>();
}
