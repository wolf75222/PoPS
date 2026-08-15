#include <gtest/gtest.h>

#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>

#include <array>
#include <cstdint>
#include <exception>
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
          {distribution.box_count(), 1}};
}

template <int Dim>
pops::elliptic::mg::CompositeFacBuildRequest<Dim> adjacent_request(bool overlap) {
  const pops::Box<Dim> domain{index<Dim>(0), index<Dim>(7)};
  pops::RealVector<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = pops::Real(1);
  const pops::Geometry<Dim> coarse_geometry =
      pops::Geometry<Dim>::from_bounds(domain, pops::RealVector<Dim>{}, upper);
  const pops::Geometry<Dim> fine_geometry = coarse_geometry.refine(extent<Dim>(2));
  pops::Index<Dim> first_lower = index<Dim>(4);
  pops::Index<Dim> first_upper = index<Dim>(11);
  pops::Index<Dim> second_lower = first_lower;
  pops::Index<Dim> second_upper = first_upper;
  first_upper[0] = 7;
  second_lower[0] = overlap ? 7 : 8;
  second_upper[0] = 11;
  const pops::mesh::BoxArray<Dim> coarse_boxes{std::vector<pops::Box<Dim>>{domain}};
  const pops::mesh::BoxArray<Dim> fine_boxes{std::vector<pops::Box<Dim>>{
      pops::Box<Dim>{first_lower, first_upper}, pops::Box<Dim>{second_lower, second_upper}}};
  return {{request<Dim>(coarse_geometry, coarse_boxes), request<Dim>(fine_geometry, fine_boxes)},
          {pops::amr::RefinementRatio<Dim>{ratio<Dim>(2)}}};
}

template <int Dim>
void expect_adjacent_patches_are_accepted_and_overlaps_rejected() {
  EXPECT_NO_THROW({
    pops::elliptic::mg::CompositeFacPoisson<Dim> solver(adjacent_request<Dim>(false));
    EXPECT_EQ(solver.n_levels(), 2);
  });
  EXPECT_THROW((void)pops::elliptic::mg::CompositeFacPoisson<Dim>(adjacent_request<Dim>(true)),
               std::exception);
}

}  // namespace

TEST(test_composite_fac_adjacent, exact_ranked_adjacent_patches_accept_without_overlap) {
  expect_adjacent_patches_are_accepted_and_overlaps_rejected<1>();
  expect_adjacent_patches_are_accepted_and_overlaps_rejected<2>();
  expect_adjacent_patches_are_accepted_and_overlaps_rejected<3>();
}
