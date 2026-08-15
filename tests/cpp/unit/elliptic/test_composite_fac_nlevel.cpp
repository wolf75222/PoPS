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
  return {geometry,
          std::move(boxes),
          distribution,
          local_rank,
          {pops::BoundaryTopology<Dim>::physical(), faces, spacing},
          pops::Extent<Dim>{},
          extent<Dim>(1),
          {distribution.box_count(), 0}};
}

template <int Dim>
void expect_three_level_hierarchy_prepares_in_one_exact_rank() {
  const pops::Box<Dim> coarse_domain{index<Dim>(0), index<Dim>(7)};
  pops::RealVector<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = pops::Real(1);
  const pops::Geometry<Dim> coarse_geometry =
      pops::Geometry<Dim>::from_bounds(coarse_domain, pops::RealVector<Dim>{}, upper);
  const pops::Geometry<Dim> middle_geometry = coarse_geometry.refine(extent<Dim>(2));
  const pops::Geometry<Dim> fine_geometry = middle_geometry.refine(extent<Dim>(2));
  const pops::Box<Dim> middle_patch{index<Dim>(4), index<Dim>(11)};
  const pops::Box<Dim> fine_patch{index<Dim>(8), index<Dim>(23)};
  pops::elliptic::mg::CompositeFacBuildRequest<Dim> build{
      {request<Dim>(coarse_geometry,
                    pops::mesh::BoxArray<Dim>{std::vector<pops::Box<Dim>>{coarse_domain}}),
       request<Dim>(middle_geometry,
                    pops::mesh::BoxArray<Dim>{std::vector<pops::Box<Dim>>{middle_patch}}),
       request<Dim>(fine_geometry,
                    pops::mesh::BoxArray<Dim>{std::vector<pops::Box<Dim>>{fine_patch}})},
      {pops::amr::RefinementRatio<Dim>{ratio<Dim>(2)},
       pops::amr::RefinementRatio<Dim>{ratio<Dim>(2)}}};
  pops::elliptic::mg::CompositeFacPoisson<Dim> solver(std::move(build));
  EXPECT_EQ(solver.n_levels(), 3);
  solver.install_nullspace(pops::FieldNullspacePlan<Dim>{},
                           {pops::PreparedVectorDistribution<Dim>::replicated(),
                            pops::PreparedVectorDistribution<Dim>::replicated(),
                            pops::PreparedVectorDistribution<Dim>::replicated()});
  for (int level = 0; level < solver.n_levels(); ++level)
    solver.rhs_level(level).set_val(pops::Real(0));
  EXPECT_TRUE(solver.solve().solved());
}

}  // namespace

TEST(test_composite_fac_nlevel, three_levels_prepare_and_solve_in_1d_2d_3d) {
  expect_three_level_hierarchy_prepares_in_one_exact_rank<1>();
  expect_three_level_hierarchy_prepares_in_one_exact_rank<2>();
  expect_three_level_hierarchy_prepares_in_one_exact_rank<3>();
}
