#include <gtest/gtest.h>

#include <pops/mesh/layout/distribution.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

class CommEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override { pops::comm_init(); }
  void TearDown() override { pops::comm_finalize(); }
};

[[maybe_unused]] const ::testing::Environment* const kCommEnvironment =
    ::testing::AddGlobalTestEnvironment(new CommEnvironment);

constexpr pops::Real kPi = pops::Real(3.14159265358979323846);

template <class Ranked, int Dim, class Value>
Ranked ranked(Value value) {
  Ranked result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim, class Value>
std::array<Value, Dim> filled(Value value) {
  std::array<Value, Dim> result{};
  result.fill(value);
  return result;
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
  std::size_t ordinal = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    ordinal += static_cast<std::size_t>(index[axis] - box.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(box.length(axis));
  }
  return ordinal;
}

template <int Dim>
pops::EllipticBuildRequest<Dim> request(const pops::Geometry<Dim>& geometry,
                                        pops::mesh::BoxArray<Dim> boxes) {
  pops::Extent<Dim> rank_extent = ranked<pops::Extent<Dim>, Dim>(std::int64_t{1});
  rank_extent[0] = pops::n_ranks();
  pops::Index<Dim> local_rank{};
  local_rank[0] = pops::my_rank();
  const pops::mesh::RankSpace<Dim> ranks{pops::Index<Dim>{}, rank_extent};
  const auto distribution = pops::mesh::Distribution<Dim>::replicated(boxes, ranks);
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
          pops::PhysicalBoundaryConditions<Dim>{pops::BoundaryTopology<Dim>::physical(), faces,
                                                spacing},
          pops::Extent<Dim>{},
          ranked<pops::Extent<Dim>, Dim>(std::int64_t{1}),
          {distribution.box_count(), pairs}};
}

pops::Real exact(pops::Real x, pops::Real y) { return std::sin(kPi * x) * std::sin(kPi * y); }

pops::Real conductivity(pops::Real x, pops::Real y) {
  return pops::Real(1) +
         pops::Real(0.25) * std::sin(pops::Real(2) * kPi * x) * std::sin(pops::Real(2) * kPi * y);
}

pops::Real manufactured_rhs(pops::Real x, pops::Real y) {
  const pops::Real u = exact(x, y);
  const pops::Real ux = kPi * std::cos(kPi * x) * std::sin(kPi * y);
  const pops::Real uy = kPi * std::sin(kPi * x) * std::cos(kPi * y);
  const pops::Real kx =
      pops::Real(0.5) * kPi * std::cos(pops::Real(2) * kPi * x) * std::sin(pops::Real(2) * kPi * y);
  const pops::Real ky =
      pops::Real(0.5) * kPi * std::sin(pops::Real(2) * kPi * x) * std::cos(pops::Real(2) * kPi * y);
  return pops::Real(2) * kPi * kPi * conductivity(x, y) * u - kx * ux - ky * uy;
}

template <int Dim>
void fill_scalar(pops::MultiFab<Dim>& field, const pops::Geometry<Dim>& geometry,
                 pops::Real (*value)(pops::Real, pops::Real)) {
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(fab.box().numPts());
         ++ordinal) {
      const auto index = index_from_ordinal(fab.box(), ordinal);
      const pops::Real x = geometry.cell_coordinate(0, index[0]);
      const pops::Real y = geometry.cell_coordinate(1, index[1]);
      host(storage_ordinal(fab.grown_box(), index)) = value(x, y);
    }
    fab.copy_from_host(host);
  }
}

double interior_error(const pops::MultiFab<2>& field, const pops::Geometry<2>& geometry,
                      const pops::Box<2>& region) {
  double result = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const auto overlap = field.box(local).intersect(region);
    if (overlap.empty())
      continue;
    const auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(overlap.numPts()); ++ordinal) {
      const auto index = index_from_ordinal(overlap, ordinal);
      const pops::Real x = geometry.cell_coordinate(0, index[0]);
      const pops::Real y = geometry.cell_coordinate(1, index[1]);
      result = std::max(result, std::abs(static_cast<double>(
                                    host(storage_ordinal(fab.grown_box(), index)) - exact(x, y))));
    }
  }
  return pops::all_reduce_max(result);
}

double solve_error(int cells) {
  constexpr int Dim = 2;
  const pops::Box<Dim> domain{pops::Index<Dim>{}, ranked<pops::Index<Dim>, Dim>(cells - 1)};
  const auto coarse_geometry = pops::Geometry<Dim>::from_bounds(
      domain, pops::RealVector<Dim>{}, ranked<pops::RealVector<Dim>, Dim>(pops::Real(1)));
  const auto fine_geometry = coarse_geometry.refine(ranked<pops::Extent<Dim>, Dim>(std::int64_t{2}));
  pops::Index<Dim> coarse_lo{};
  pops::Index<Dim> coarse_hi{};
  pops::Index<Dim> fine_lo{};
  pops::Index<Dim> fine_hi{};
  for (int axis = 0; axis < Dim; ++axis) {
    coarse_lo[axis] = cells / 4;
    coarse_hi[axis] = 3 * cells / 4 - 1;
    fine_lo[axis] = 2 * coarse_lo[axis];
    fine_hi[axis] = 2 * coarse_hi[axis] + 1;
  }
  const pops::Box<Dim> fine_patch{fine_lo, fine_hi};
  const pops::mesh::BoxArray<Dim> coarse_layout(std::vector<pops::Box<Dim>>{domain});
  const pops::mesh::BoxArray<Dim> fine_layout(std::vector<pops::Box<Dim>>{fine_patch});
  pops::elliptic::mg::CompositeFacBuildRequest<Dim> hierarchy{
      {request(coarse_geometry, coarse_layout), request(fine_geometry, fine_layout)},
      {pops::amr::RefinementRatio<Dim>{filled<Dim>(2)}}};
  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.composite-fac.variable-k");
  pops::CompositeFacOptions options;
  options.max_iters = 80;
  options.fine_sweeps = 8;
  options.rel_tol = pops::Real(1e-6);
  options.coarse_cycles = 96;
  pops::elliptic::mg::CompositeFacPoisson<Dim> solver(std::move(hierarchy), lane, options);
  pops::Extent<Dim> ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    ghosts[axis] = 1;
  for (int level = 0; level < solver.n_levels(); ++level) {
    const auto& phi = solver.phi_level(level);
    pops::MultiFab<Dim> conductivity_field(phi.layout(), phi.distribution(), phi.local_rank(), 1,
                                           ghosts);
    const auto& geometry = level == 0 ? coarse_geometry : fine_geometry;
    fill_scalar(conductivity_field, geometry, conductivity);
    solver.install_coefficient(level, conductivity_field);
    fill_scalar(solver.rhs_level(level), geometry, manufactured_rhs);
    solver.phi_level(level).set_val(pops::Real(0));
  }
  solver.install_nullspace(
      pops::FieldNullspacePlan<Dim>{},
      std::vector<pops::PreparedVectorDistribution<Dim>>(
          static_cast<std::size_t>(solver.n_levels()),
          pops::PreparedVectorDistribution<Dim>::replicated()));
  const pops::SolveReport report = solver.solve();
  if (!report.solved())
    throw std::runtime_error(report.reason + " residual=" +
                             std::to_string(static_cast<double>(report.residual_norm)) +
                             " rel=" + std::to_string(static_cast<double>(report.rel_residual)) +
                             " iters=" + std::to_string(report.iters));
  const pops::Box<Dim> measured = fine_patch.grow(-std::max(2, cells / 4));
  return interior_error(solver.phi_level(1), fine_geometry, measured);
}

double solve_constant_weighted(int cells) {
  constexpr int Dim = 2;
  const pops::Box<Dim> domain{pops::Index<Dim>{}, ranked<pops::Index<Dim>, Dim>(cells - 1)};
  const auto coarse_geometry = pops::Geometry<Dim>::from_bounds(
      domain, pops::RealVector<Dim>{}, ranked<pops::RealVector<Dim>, Dim>(pops::Real(1)));
  const auto fine_geometry = coarse_geometry.refine(ranked<pops::Extent<Dim>, Dim>(std::int64_t{2}));
  pops::Index<Dim> fine_lo{};
  pops::Index<Dim> fine_hi{};
  for (int axis = 0; axis < Dim; ++axis) {
    fine_lo[axis] = cells / 2;
    fine_hi[axis] = 3 * cells / 2 - 1;
  }
  const pops::Box<Dim> fine_patch{fine_lo, fine_hi};
  pops::elliptic::mg::CompositeFacBuildRequest<Dim> hierarchy{
      {request(coarse_geometry, pops::mesh::BoxArray<Dim>(std::vector<pops::Box<Dim>>{domain})),
       request(fine_geometry, pops::mesh::BoxArray<Dim>(std::vector<pops::Box<Dim>>{fine_patch}))},
      {pops::amr::RefinementRatio<Dim>{filled<Dim>(2)}}};
  const pops::ExecutionLane lane =
      pops::ExecutionLane::world("tests.composite-fac.variable-k.constant");
  pops::CompositeFacOptions options;
  options.max_iters = 40;
  options.fine_sweeps = 4;
  options.rel_tol = pops::Real(1e-4);
  pops::elliptic::mg::CompositeFacPoisson<Dim> solver(std::move(hierarchy), lane, options);
  const pops::Real pi = std::acos(pops::Real(-1));
  for (int level = 0; level < solver.n_levels(); ++level) {
    const auto& phi = solver.phi_level(level);
    pops::Extent<Dim> ghosts{};
    for (int axis = 0; axis < Dim; ++axis)
      ghosts[axis] = 1;
    pops::MultiFab<Dim> conductivity_field(phi.layout(), phi.distribution(), phi.local_rank(), 1,
                                           ghosts);
    conductivity_field.set_val(pops::Real(1));
    solver.install_coefficient(level, conductivity_field);
    const auto& geometry = level == 0 ? coarse_geometry : fine_geometry;
    auto& rhs = solver.rhs_level(level);
    for (std::size_t local = 0; local < rhs.local_size(); ++local) {
      auto& fab = rhs.fab(local);
      auto host = fab.create_host_mirror();
      for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(fab.box().numPts());
           ++ordinal) {
        const auto index = index_from_ordinal(fab.box(), ordinal);
        host(storage_ordinal(fab.grown_box(), index)) =
            pops::Real(2) * pi * pi * exact(geometry.cell_coordinate(0, index[0]),
                                            geometry.cell_coordinate(1, index[1]));
      }
      fab.copy_from_host(host);
    }
    solver.phi_level(level).set_val(pops::Real(0));
  }
  solver.install_nullspace(
      pops::FieldNullspacePlan<Dim>{},
      std::vector<pops::PreparedVectorDistribution<Dim>>(
          2, pops::PreparedVectorDistribution<Dim>::replicated()));
  const pops::SolveReport report = solver.solve();
  if (!report.solved())
    throw std::runtime_error(report.reason + " residual=" +
                             std::to_string(static_cast<double>(report.residual_norm)) +
                             " rel=" + std::to_string(static_cast<double>(report.rel_residual)));
  return interior_error(solver.phi_level(1), fine_geometry, fine_patch.grow(-2));
}

}  // namespace

TEST(test_composite_fac_variable_k, unit_coefficient_matches_constant_operator) {
  if (pops::n_ranks() != 1)
    GTEST_SKIP();
  EXPECT_NO_THROW(solve_constant_weighted(8));
}

TEST(test_composite_fac_variable_k, scalar_variable_coefficient_retains_refinement_accuracy) {
  if (pops::n_ranks() != 1)
    GTEST_SKIP() << "serial variable-k FAC uses a one-rank rank space";
  const double coarse = solve_error(16);
  const double fine = solve_error(32);
  EXPECT_GT(coarse, 0.0);
  EXPECT_LT(fine, 0.55 * coarse)
      << "scalar composite FAC with cell-centered harmonic k must converge under refinement";
}
