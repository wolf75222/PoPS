#include <gtest/gtest.h>

#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>

#include "nd_multifab_test_utils.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

using namespace pops;
using namespace pops::mesh;
using namespace pops::test::nd;

namespace {

template <int Dim>
std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> uniform_faces(
    PhysicalBoundaryFace face) {
  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> result{};
  result.fill(face);
  return result;
}

template <int Dim>
RealVector<Dim> unit_spacing() {
  RealVector<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = Real(1);
  return result;
}

template <int Dim>
HaloScheduleBudget halo_budget(std::size_t boxes, std::size_t images = 64) {
  return HaloScheduleBudget{{boxes, boxes * (boxes - 1) / 2},
                            boxes * boxes * images,
                            boxes * boxes * images * static_cast<std::size_t>(2 * Dim),
                            images,
                            boxes,
                            1'000'000,
                            1'000'000,
                            1'000'000};
}

template <int Dim>
void expect_deep_dirichlet() {
  constexpr int depth = 5;
  const Box<Dim> domain = cube<Dim>(-4, -4);
  const BoxArray<Dim> layout(std::vector<Box<Dim>>{domain});
  const auto ranks = one_rank_space<Dim>();
  const auto distribution = Distribution<Dim>::replicated(layout, ranks);
  HostMultiFab<Dim> field(layout, distribution, Index<Dim>{}, 2, uniform_extent<Dim>(depth));
  fill_valid(field, Real{-99},
             [](const Index<Dim>&, int component) { return component == 0 ? Real(6) : Real(11); });

  const auto faces = uniform_faces<Dim>(
      PhysicalBoundaryFace{PhysicalBoundaryKind::dirichlet, Real(2), Real(0), Real(1)});
  const auto prepared =
      prepare_physical_boundary(domain, uniform_extent<Dim>(depth),
                                PhysicalBoundaryConditions<Dim>{BoundaryTopology<Dim>::physical(),
                                                                faces, unit_spacing<Dim>()},
                                BoundaryScheduleBudget{1'000});
  fill_physical_boundary(field, prepared, /*first_component=*/0, /*component_count=*/1);

  const auto& fab = field.fab(0);
  const std::size_t cells = static_cast<std::size_t>(fab.grown_box().numPts());
  for (std::size_t ordinal = 0; ordinal < cells; ++ordinal) {
    const Index<Dim> index = index_from_ordinal(fab.grown_box(), ordinal);
    int reflections = 0;
    for (int axis = 0; axis < Dim; ++axis)
      reflections += std::abs(index[axis] - domain.lo[axis]);
    EXPECT_EQ(value_at(field, 0, index, 0), reflections % 2 == 0 ? Real(6) : Real(-2));
    EXPECT_EQ(value_at(field, 0, index, 1), domain.contains(index) ? Real(11) : Real(-99));
  }
}

}  // namespace

TEST(test_physical_bc, arbitrary_depth_affine_extension_is_exact_in_1d_2d_and_3d) {
  expect_deep_dirichlet<1>();
  expect_deep_dirichlet<2>();
  expect_deep_dirichlet<3>();
}

TEST(test_physical_bc, physical_periodic_corner_composes_from_prepared_nd_schedules) {
  const Box<2> domain{Index<2>{0, 0}, Index<2>{3, 7}};
  const BoxArray<2> layout(std::vector<Box<2>>{
      Box<2>{Index<2>{0, 0}, Index<2>{3, 3}},
      Box<2>{Index<2>{0, 4}, Index<2>{3, 7}},
  });
  const auto ranks = one_rank_space<2>();
  const auto distribution = Distribution<2>::replicated(layout, ranks);
  HostMultiFab<2> field(layout, distribution, Index<2>{}, 1, Extent<2>{1, 1});
  fill_valid(field, Real{-99},
             [](const Index<2>& index, int) { return Real(index[0] + 10 * index[1]); });

  const BoundaryTopology<2> topology = BoundaryTopology<2>::axis_periodic({false, true});
  const auto halo = prepare_halo_schedule(field, domain, topology, halo_budget<2>(layout.size()));
  fill_boundary(field, halo);

  auto faces = uniform_faces<2>(PhysicalBoundaryFace{});
  faces[static_cast<std::size_t>(Face<2>{0, BoundarySide::lower}.ordinal())] =
      PhysicalBoundaryFace{PhysicalBoundaryKind::constant_extrapolation};
  faces[static_cast<std::size_t>(Face<2>{0, BoundarySide::upper}.ordinal())] =
      PhysicalBoundaryFace{PhysicalBoundaryKind::dirichlet, Real(0)};
  const auto prepared = prepare_physical_boundary(
      domain, Extent<2>{1, 1}, PhysicalBoundaryConditions<2>{topology, faces, RealVector<2>{1, 1}},
      BoundaryScheduleBudget{64});
  fill_physical_boundary(field, prepared);

  EXPECT_EQ(value_at(field, 0, Index<2>{-1, 4}), Real(40));
  EXPECT_EQ(value_at(field, 0, Index<2>{4, 4}), Real(-43));
  EXPECT_EQ(value_at(field, 0, Index<2>{-1, -1}), Real(70));
}

TEST(test_physical_bc, Neumann_and_robin_are_ranked_and_fail_closed_before_launch) {
  const Box<3> domain = cube<3>(0, 0);
  const BoxArray<3> layout(std::vector<Box<3>>{domain});
  const auto ranks = one_rank_space<3>();
  const auto distribution = Distribution<3>::replicated(layout, ranks);
  HostMultiFab<3> field(layout, distribution, Index<3>{}, 1, Extent<3>{1, 1, 1});
  fill_valid(field, Real{-7}, [](const Index<3>&, int) { return Real(5); });

  auto neumann = uniform_faces<3>(
      PhysicalBoundaryFace{PhysicalBoundaryKind::neumann, Real(2), Real(0), Real(1)});
  const auto prepared = prepare_physical_boundary(
      domain, Extent<3>{1, 1, 1},
      PhysicalBoundaryConditions<3>{BoundaryTopology<3>::physical(), neumann,
                                    RealVector<3>{0.5, 0.25, 2.0}},
      BoundaryScheduleBudget{64});
  fill_physical_boundary(field, prepared);
  EXPECT_EQ(value_at(field, 0, Index<3>{-1, 0, 0}), Real(6));
  EXPECT_EQ(value_at(field, 0, Index<3>{0, -1, 0}), Real(5.5));
  EXPECT_EQ(value_at(field, 0, Index<3>{0, 0, 1}), Real(9));

  auto singular = uniform_faces<1>(PhysicalBoundaryFace{});
  singular[0] = PhysicalBoundaryFace{PhysicalBoundaryKind::robin, Real(0), Real(2), Real(-1)};
  EXPECT_THROW(
      (void)prepare_physical_boundary(Box<1>{Index<1>{0}, Index<1>{1}}, Extent<1>{1},
                                      PhysicalBoundaryConditions<1>{BoundaryTopology<1>::physical(),
                                                                    singular, RealVector<1>{1}},
                                      BoundaryScheduleBudget{8}),
      std::invalid_argument);

  EXPECT_THROW(fill_physical_boundary(field, prepared, -1, 1), std::out_of_range);
  EXPECT_THROW(fill_physical_boundary(field, prepared, 0, 2), std::out_of_range);
}

TEST(test_physical_bc, periodic_faces_cannot_hide_conflicting_physical_laws) {
  auto faces = uniform_faces<1>(PhysicalBoundaryFace{});
  faces[0] = PhysicalBoundaryFace{PhysicalBoundaryKind::dirichlet, Real(0)};
  EXPECT_THROW(((void)PhysicalBoundaryConditions<1>{BoundaryTopology<1>::axis_periodic({true}),
                                                    faces, RealVector<1>{1}}),
               std::invalid_argument);
}
