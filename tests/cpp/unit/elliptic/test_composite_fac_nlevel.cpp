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
pops::EllipticBuildRequest<Dim> request(const pops::Geometry<Dim>& geometry,
                                        pops::mesh::BoxArray<Dim> layout) {
  const pops::mesh::RankSpace<Dim> ranks{pops::Index<Dim>{},
                                         filled<pops::Extent<Dim>, Dim>(std::int64_t{1})};
  const auto distribution = pops::mesh::Distribution<Dim>::replicated(layout, ranks);
  std::array<pops::PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  faces.fill({pops::PhysicalBoundaryKind::dirichlet, pops::Real(0)});
  pops::RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  return {geometry,
          std::move(layout),
          distribution,
          pops::Index<Dim>{},
          pops::PhysicalBoundaryConditions<Dim>{pops::BoundaryTopology<Dim>::physical(), faces,
                                                spacing},
          pops::Extent<Dim>{},
          filled<pops::Extent<Dim>, Dim>(std::int64_t{1}),
          {distribution.box_count(), 0}};
}

template <int Dim>
pops::Index<Dim> index_from_ordinal(const pops::Box<Dim>& box, std::size_t ordinal) {
  pops::Index<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    const auto length = static_cast<std::size_t>(box.length(axis));
    result[axis] = box.lo[axis] + static_cast<int>(ordinal % length);
    ordinal /= length;
  }
  return result;
}

template <int Dim>
std::size_t storage_ordinal(const pops::Box<Dim>& box, const pops::Index<Dim>& index) {
  std::size_t result = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    result += static_cast<std::size_t>(index[axis] - box.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(box.length(axis));
  }
  return result;
}

template <int Dim>
void fill_mode(pops::MultiFab<Dim>& rhs, const pops::Geometry<Dim>& geometry) {
  const pops::Real pi = std::acos(pops::Real(-1));
  const pops::Real eigenvalue = static_cast<pops::Real>(Dim) * pi * pi;
  auto& fab = rhs.fab(0);
  auto host = fab.create_host_mirror();
  for (std::size_t n = 0; n < static_cast<std::size_t>(fab.box().numPts()); ++n) {
    const auto index = index_from_ordinal(fab.box(), n);
    pops::Real exact = pops::Real(1);
    for (int axis = 0; axis < Dim; ++axis)
      exact *= std::sin(pi * geometry.cell_coordinate(axis, index[axis]));
    host(storage_ordinal(fab.grown_box(), index)) = eigenvalue * exact;
  }
  fab.copy_from_host(host);
}

template <int Dim>
double mode_error(const pops::MultiFab<Dim>& field, const pops::Geometry<Dim>& geometry,
                  const pops::Box<Dim>& region) {
  const pops::Real pi = std::acos(pops::Real(-1));
  double error = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const auto overlap = field.box(local).intersect(region);
    if (overlap.empty())
      continue;
    const auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t n = 0; n < static_cast<std::size_t>(overlap.numPts()); ++n) {
      const auto index = index_from_ordinal(overlap, n);
      pops::Real exact = pops::Real(1);
      for (int axis = 0; axis < Dim; ++axis)
        exact *= std::sin(pi * geometry.cell_coordinate(axis, index[axis]));
      error = std::max(
          error,
          std::abs(static_cast<double>(host(storage_ordinal(fab.grown_box(), index)) - exact)));
    }
  }
  return pops::all_reduce_max(error);
}

template <int Dim>
double average_down_defect(const pops::MultiFab<Dim>& parent_field,
                           const pops::MultiFab<Dim>& child_field,
                           const pops::Box<Dim>& child_footprint) {
  const auto& parent_fab = parent_field.fab(0);
  auto parent = parent_fab.create_host_mirror();
  parent_fab.copy_to_host(parent);
  const auto& child_fab = child_field.fab(0);
  auto child = child_fab.create_host_mirror();
  child_fab.copy_to_host(child);
  const int child_count = 1 << Dim;
  double defect = 0;
  for (std::size_t n = 0; n < static_cast<std::size_t>(child_footprint.numPts()); ++n) {
    const auto cell = index_from_ordinal(child_footprint, n);
    double average = 0;
    for (int child_ordinal = 0; child_ordinal < child_count; ++child_ordinal) {
      pops::Index<Dim> fine_cell{};
      for (int axis = 0; axis < Dim; ++axis)
        fine_cell[axis] = 2 * cell[axis] + ((child_ordinal >> axis) & 1);
      average += child(storage_ordinal(child_fab.grown_box(), fine_cell));
    }
    average /= child_count;
    defect =
        std::max(defect, std::abs(parent(storage_ordinal(parent_fab.grown_box(), cell)) - average));
  }
  return pops::all_reduce_max(defect);
}

template <int Dim>
void expect_three_level_coupling() {
  constexpr int cells = 24;
  const pops::Box<Dim> coarse_domain{pops::Index<Dim>{}, filled<pops::Index<Dim>, Dim>(cells - 1)};
  const auto coarse = pops::Geometry<Dim>::from_bounds(
      coarse_domain, pops::RealVector<Dim>{}, filled<pops::RealVector<Dim>, Dim>(pops::Real(1)));
  const auto middle = coarse.refine(filled<pops::Extent<Dim>, Dim>(std::int64_t{2}));
  const auto fine = middle.refine(filled<pops::Extent<Dim>, Dim>(std::int64_t{2}));
  pops::Index<Dim> middle_lo{};
  pops::Index<Dim> middle_hi{};
  pops::Index<Dim> fine_lo{};
  pops::Index<Dim> fine_hi{};
  for (int axis = 0; axis < Dim; ++axis) {
    middle_lo[axis] = 12;
    middle_hi[axis] = 35;
    fine_lo[axis] = 36;
    fine_hi[axis] = 59;
  }
  const pops::Box<Dim> middle_patch{middle_lo, middle_hi};
  const pops::Box<Dim> fine_patch{fine_lo, fine_hi};
  const pops::mesh::BoxArray<Dim> coarse_layout(std::vector<pops::Box<Dim>>{coarse_domain});
  const pops::mesh::BoxArray<Dim> middle_layout(std::vector<pops::Box<Dim>>{middle_patch});
  const pops::mesh::BoxArray<Dim> fine_layout(std::vector<pops::Box<Dim>>{fine_patch});
  const auto ratio = pops::amr::RefinementRatio<Dim>{filled<std::array<int, Dim>, Dim>(2)};
  pops::elliptic::mg::CompositeFacBuildRequest<Dim> hierarchy{
      {request(coarse, coarse_layout), request(middle, middle_layout), request(fine, fine_layout)},
      {ratio, ratio}};
  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.fac.nlevel");
  pops::elliptic::mg::GeometricMultigridOptions mg_controls;
  mg_controls.relative_tolerance = pops::Real(1e-10);
  mg_controls.maximum_cycles = 100;
  pops::elliptic::mg::GeometricMG<Dim> coarse_solver(request(coarse, coarse_layout), lane,
                                                     mg_controls);
  coarse_solver.install_nullspace(pops::FieldNullspacePlan<Dim>{},
                                  pops::PreparedVectorDistribution<Dim>::replicated());
  fill_mode(coarse_solver.rhs(), coarse);
  const pops::SolveReport coarse_report = coarse_solver.solve();
  ASSERT_TRUE(coarse_report.solved()) << coarse_report.reason;

  pops::CompositeFacOptions controls;
  controls.max_iters = 80;
  controls.fine_sweeps = 80;
  controls.rel_tol = pops::Real(1e-9);
  pops::elliptic::mg::CompositeFacPoisson<Dim> solver(std::move(hierarchy), lane, controls);
  solver.install_nullspace(pops::FieldNullspacePlan<Dim>{},
                           {pops::PreparedVectorDistribution<Dim>::replicated(),
                            pops::PreparedVectorDistribution<Dim>::replicated(),
                            pops::PreparedVectorDistribution<Dim>::replicated()});
  fill_mode(solver.rhs_level(0), coarse);
  fill_mode(solver.rhs_level(1), middle);
  fill_mode(solver.rhs_level(2), fine);
  const pops::SolveReport report = solver.solve();
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_EQ(solver.n_levels(), 3);
  EXPECT_GT(pops::norm_inf(solver.phi_level(2)), pops::Real(0));

  pops::Index<Dim> coarse_inner_lo{};
  pops::Index<Dim> coarse_inner_hi{};
  pops::Index<Dim> middle_inner_lo{};
  pops::Index<Dim> middle_inner_hi{};
  pops::Index<Dim> fine_inner_lo{};
  pops::Index<Dim> fine_inner_hi{};
  pops::Index<Dim> middle_footprint_lo{};
  pops::Index<Dim> middle_footprint_hi{};
  pops::Index<Dim> fine_footprint_lo{};
  pops::Index<Dim> fine_footprint_hi{};
  for (int axis = 0; axis < Dim; ++axis) {
    coarse_inner_lo[axis] = 10;
    coarse_inner_hi[axis] = 13;
    middle_inner_lo[axis] = 20;
    middle_inner_hi[axis] = 27;
    fine_inner_lo[axis] = 40;
    fine_inner_hi[axis] = 55;
    middle_footprint_lo[axis] = middle_patch.lo[axis] / 2;
    middle_footprint_hi[axis] = middle_patch.hi[axis] / 2;
    fine_footprint_lo[axis] = fine_patch.lo[axis] / 2;
    fine_footprint_hi[axis] = fine_patch.hi[axis] / 2;
  }
  const double coarse_error =
      mode_error(coarse_solver.phi(), coarse, pops::Box<Dim>{coarse_inner_lo, coarse_inner_hi});
  const double middle_error =
      mode_error(solver.phi_level(1), middle, pops::Box<Dim>{middle_inner_lo, middle_inner_hi});
  const double fine_error =
      mode_error(solver.phi_level(2), fine, pops::Box<Dim>{fine_inner_lo, fine_inner_hi});
  EXPECT_GT(coarse_error, 0.0);
  EXPECT_LT(middle_error, coarse_error);
  EXPECT_LT(fine_error, middle_error)
      << "each nested sparse level must add measurable MMS accuracy";

  EXPECT_LT(average_down_defect(solver.phi_level(0), solver.phi_level(1),
                                pops::Box<Dim>{middle_footprint_lo, middle_footprint_hi}),
            1e-12)
      << "level 0 must contain accepted level-1 child averages";
  EXPECT_LT(average_down_defect(solver.phi_level(1), solver.phi_level(2),
                                pops::Box<Dim>{fine_footprint_lo, fine_footprint_hi}),
            1e-12)
      << "level 1 must contain accepted level-2 child averages";
}

}  // namespace

TEST(CompositeFacNlevelTest, arbitrary_depth_preserves_multilevel_coupling_in_exact_rank) {
  pops::comm_init();
  expect_three_level_coupling<1>();
  expect_three_level_coupling<2>();
  expect_three_level_coupling<3>();
  pops::comm_finalize();
}
