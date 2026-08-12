#include <gtest/gtest.h>

#include <pops/mesh/layout/distribution.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

template <class Ranked, int Dim, class Value>
Ranked filled(Value value) {
  Ranked result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::PhysicalBoundaryConditions<Dim> dirichlet(const pops::Geometry<Dim>& geometry) {
  std::array<pops::PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  faces.fill({pops::PhysicalBoundaryKind::dirichlet, pops::Real(0)});
  pops::RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  return {pops::BoundaryTopology<Dim>::physical(), faces, spacing};
}

template <int Dim>
pops::EllipticBuildRequest<Dim> request(const pops::Geometry<Dim>& geometry,
                                        pops::mesh::BoxArray<Dim> layout) {
  const pops::mesh::RankSpace<Dim> ranks{pops::Index<Dim>{},
                                         filled<pops::Extent<Dim>, Dim>(std::int64_t{1})};
  const auto distribution = pops::mesh::Distribution<Dim>::replicated(layout, ranks);
  const std::size_t pairs = layout.size() * (layout.size() - 1) / 2;
  return {geometry,
          std::move(layout),
          distribution,
          pops::Index<Dim>{},
          dirichlet(geometry),
          pops::Extent<Dim>{},
          filled<pops::Extent<Dim>, Dim>(std::int64_t{1}),
          {distribution.box_count(), pairs}};
}

template <int Dim>
void expect_adjacent_layout_prepares() {
  const pops::Box<Dim> coarse_domain{pops::Index<Dim>{}, filled<pops::Index<Dim>, Dim>(15)};
  const auto coarse_geometry = pops::Geometry<Dim>::from_bounds(
      coarse_domain, pops::RealVector<Dim>{}, filled<pops::RealVector<Dim>, Dim>(pops::Real(1)));
  const auto fine_geometry =
      coarse_geometry.refine(filled<pops::Extent<Dim>, Dim>(std::int64_t{2}));
  pops::Box<Dim> left{filled<pops::Index<Dim>, Dim>(8), filled<pops::Index<Dim>, Dim>(23)};
  pops::Box<Dim> right = left;
  left.lo[0] = 4;
  left.hi[0] = 15;
  right.lo[0] = 16;
  right.hi[0] = 27;
  const pops::mesh::BoxArray<Dim> coarse_layout(std::vector<pops::Box<Dim>>{coarse_domain});
  const pops::mesh::BoxArray<Dim> adjacent(std::vector<pops::Box<Dim>>{left, right});
  pops::elliptic::mg::CompositeFacBuildRequest<Dim> hierarchy{
      {request(coarse_geometry, coarse_layout), request(fine_geometry, adjacent)},
      {pops::amr::RefinementRatio<Dim>{filled<std::array<int, Dim>, Dim>(2)}}};
  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.fac.adjacent.prepare");
  pops::elliptic::mg::CompositeFacPoisson<Dim> solver(std::move(hierarchy), lane);
  solver.install_nullspace(pops::FieldNullspacePlan<Dim>{},
                           {pops::PreparedVectorDistribution<Dim>::replicated(),
                            pops::PreparedVectorDistribution<Dim>::replicated()});
  solver.rhs_level(0).set_val(pops::Real(0));
  solver.rhs_level(1).set_val(pops::Real(0));
  EXPECT_TRUE(solver.solve().solved());

  right.lo[0] = 15;
  const pops::mesh::BoxArray<Dim> overlapping(std::vector<pops::Box<Dim>>{left, right});
  pops::elliptic::mg::CompositeFacBuildRequest<Dim> invalid{
      {request(coarse_geometry, coarse_layout), request(fine_geometry, overlapping)},
      {pops::amr::RefinementRatio<Dim>{filled<std::array<int, Dim>, Dim>(2)}}};
  EXPECT_THROW((pops::elliptic::mg::CompositeFacPoisson<Dim>(std::move(invalid), lane)),
               std::invalid_argument);
}

std::size_t ordinal(const pops::Box<2>& box, const pops::Index<2>& cell) {
  return static_cast<std::size_t>(cell[0] - box.lo[0]) +
         static_cast<std::size_t>(cell[1] - box.lo[1]) * static_cast<std::size_t>(box.length(0));
}

void fill_mode(pops::MultiFab<2>& rhs, const pops::Geometry<2>& geometry) {
  const pops::Real pi = std::acos(pops::Real(-1));
  for (std::size_t local = 0; local < rhs.local_size(); ++local) {
    auto& fab = rhs.fab(local);
    auto host = fab.create_host_mirror();
    const auto box = fab.box();
    for (int j = box.lo[1]; j <= box.hi[1]; ++j)
      for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
        const pops::Index<2> cell{i, j};
        const pops::Real x = geometry.cell_coordinate(0, i);
        const pops::Real y = geometry.cell_coordinate(1, j);
        host(ordinal(fab.grown_box(), cell)) =
            pops::Real(2) * pi * pi * std::sin(pi * x) * std::sin(pi * y);
      }
    fab.copy_from_host(host);
  }
}

double mode_error(const pops::MultiFab<2>& field, const pops::Geometry<2>& geometry,
                  const pops::Box<2>& region) {
  const double pi = std::acos(-1.0);
  double error = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const auto overlap = field.box(local).intersect(region);
    if (overlap.empty())
      continue;
    const auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (int j = overlap.lo[1]; j <= overlap.hi[1]; ++j)
      for (int i = overlap.lo[0]; i <= overlap.hi[0]; ++i) {
        const pops::Index<2> cell{i, j};
        const double exact = std::sin(pi * geometry.cell_coordinate(0, i)) *
                             std::sin(pi * geometry.cell_coordinate(1, j));
        error = std::max(
            error, std::abs(host(ordinal(fab.grown_box(), cell)) - static_cast<pops::Real>(exact)));
      }
  }
  return pops::all_reduce_max(error);
}

}  // namespace

TEST(CompositeFacAdjacentTest, exact_ranked_adjacent_patches_prepare_without_overlap_adapter) {
  pops::comm_init();
  expect_adjacent_layout_prepares<1>();
  expect_adjacent_layout_prepares<2>();
  expect_adjacent_layout_prepares<3>();
  pops::comm_finalize();
}

TEST(CompositeFacAdjacentTest, adjacent_patch_join_has_no_numerical_seam) {
  pops::comm_init();
  const pops::Box<2> coarse_domain{pops::Index<2>{0, 0}, pops::Index<2>{31, 31}};
  const auto coarse_geometry = pops::Geometry<2>::from_bounds(
      coarse_domain, pops::RealVector<2>{0, 0}, pops::RealVector<2>{1, 1});
  const auto fine_geometry = coarse_geometry.refine(pops::Extent<2>{2, 2});
  const pops::Box<2> left{pops::Index<2>{16, 16}, pops::Index<2>{31, 47}};
  const pops::Box<2> right{pops::Index<2>{32, 16}, pops::Index<2>{47, 47}};
  const pops::mesh::BoxArray<2> coarse_layout(std::vector<pops::Box<2>>{coarse_domain});
  const pops::mesh::BoxArray<2> fine_layout(std::vector<pops::Box<2>>{left, right});
  pops::elliptic::mg::CompositeFacBuildRequest<2> hierarchy{
      {request(coarse_geometry, coarse_layout), request(fine_geometry, fine_layout)},
      {pops::amr::RefinementRatio<2>{std::array<int, 2>{2, 2}}}};
  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.fac.adjacent.mms");
  pops::elliptic::mg::GeometricMultigridOptions mg_controls;
  mg_controls.relative_tolerance = pops::Real(1e-10);
  mg_controls.maximum_cycles = 100;
  pops::elliptic::mg::GeometricMG<2> coarse_solver(request(coarse_geometry, coarse_layout), lane,
                                                   mg_controls);
  coarse_solver.install_nullspace(pops::FieldNullspacePlan<2>{},
                                  pops::PreparedVectorDistribution<2>::replicated());
  fill_mode(coarse_solver.rhs(), coarse_geometry);
  const pops::SolveReport coarse_report = coarse_solver.solve();
  ASSERT_TRUE(coarse_report.solved()) << coarse_report.reason;

  pops::CompositeFacOptions controls;
  controls.max_iters = 60;
  controls.fine_sweeps = 80;
  controls.rel_tol = pops::Real(1e-9);
  pops::elliptic::mg::CompositeFacPoisson<2> solver(std::move(hierarchy), lane, controls);
  solver.install_nullspace(pops::FieldNullspacePlan<2>{},
                           {pops::PreparedVectorDistribution<2>::replicated(),
                            pops::PreparedVectorDistribution<2>::replicated()});
  fill_mode(solver.rhs_level(0), coarse_geometry);
  fill_mode(solver.rhs_level(1), fine_geometry);
  const pops::SolveReport report = solver.solve();
  ASSERT_TRUE(report.solved()) << report.reason;

  const double coarse_error =
      mode_error(coarse_solver.phi(), coarse_geometry,
                 pops::Box<2>{pops::Index<2>{10, 10}, pops::Index<2>{21, 21}});
  const double fine_error =
      mode_error(solver.phi_level(1), fine_geometry,
                 pops::Box<2>{pops::Index<2>{20, 20}, pops::Index<2>{43, 43}});
  EXPECT_GT(coarse_error, 0.0);
  EXPECT_LT(fine_error, coarse_error)
      << "both adjacent patches must improve the MMS at fixed coarse resolution";

  const auto& left_fab = solver.phi_level(1).fab(0);
  const auto& right_fab = solver.phi_level(1).fab(1);
  auto left_host = left_fab.create_host_mirror();
  auto right_host = right_fab.create_host_mirror();
  left_fab.copy_to_host(left_host);
  right_fab.copy_to_host(right_host);
  double seam_error = 0;
  for (int j = left.lo[1] + 2; j <= left.hi[1] - 2; ++j) {
    const pops::Index<2> left_cell{left.hi[0], j};
    const pops::Index<2> right_cell{right.lo[0], j};
    const double exact_jump =
        std::sin(std::acos(-1.0) * fine_geometry.cell_coordinate(0, left_cell[0])) *
            std::sin(std::acos(-1.0) * fine_geometry.cell_coordinate(1, j)) -
        std::sin(std::acos(-1.0) * fine_geometry.cell_coordinate(0, right_cell[0])) *
            std::sin(std::acos(-1.0) * fine_geometry.cell_coordinate(1, j));
    const double solved_jump = left_host(ordinal(left_fab.grown_box(), left_cell)) -
                               right_host(ordinal(right_fab.grown_box(), right_cell));
    seam_error = std::max(seam_error, std::abs(solved_jump - exact_jump));
  }
  seam_error = pops::all_reduce_max(seam_error);
  EXPECT_LT(seam_error, 2e-2);
  EXPECT_LT(seam_error, coarse_error)
      << "the patch join must be smaller than the unresolved coarse discretization error";
  pops::comm_finalize();
}
