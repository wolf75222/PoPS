#include <gtest/gtest.h>

#include <pops/mesh/layout/distribution.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>
#include <pops/parallel/comm.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
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

template <int Dim>
pops::elliptic::mg::CompositeFacBuildRequest<Dim> two_level(int cells) {
  const pops::Box<Dim> domain{pops::Index<Dim>{}, ranked<pops::Index<Dim>, Dim>(cells - 1)};
  const auto coarse = pops::Geometry<Dim>::from_bounds(
      domain, pops::RealVector<Dim>{}, ranked<pops::RealVector<Dim>, Dim>(pops::Real(1)));
  const auto fine = coarse.refine(ranked<pops::Extent<Dim>, Dim>(std::int64_t{2}));
  pops::Index<Dim> fine_lo{};
  pops::Index<Dim> fine_hi{};
  for (int axis = 0; axis < Dim; ++axis) {
    fine_lo[axis] = cells / 2;
    fine_hi[axis] = 3 * cells / 2 - 1;
  }
  return {{request(coarse, pops::mesh::BoxArray<Dim>(std::vector<pops::Box<Dim>>{domain})),
           request(fine, pops::mesh::BoxArray<Dim>(std::vector<pops::Box<Dim>>{{fine_lo, fine_hi}}))},
          {pops::amr::RefinementRatio<Dim>{filled<Dim>(2)}}};
}

template <int Dim>
void unit_metric(const pops::MultiFab<Dim>& layout, pops::MultiFab<Dim>& active,
                 pops::MultiFab<Dim>& inverse_volume, pops::MultiFab<Dim>& aperture_lower,
                 pops::MultiFab<Dim>& aperture_upper) {
  active = pops::MultiFab<Dim>(layout.layout(), layout.distribution(), layout.local_rank(), 1,
                               pops::Extent<Dim>{});
  inverse_volume = pops::MultiFab<Dim>(layout.layout(), layout.distribution(), layout.local_rank(),
                                       1, pops::Extent<Dim>{});
  aperture_lower = pops::MultiFab<Dim>(layout.layout(), layout.distribution(), layout.local_rank(),
                                       Dim, pops::Extent<Dim>{});
  aperture_upper = pops::MultiFab<Dim>(layout.layout(), layout.distribution(), layout.local_rank(),
                                       Dim, pops::Extent<Dim>{});
  active.set_val(pops::Real(1));
  inverse_volume.set_val(pops::Real(1));
  aperture_lower.set_val(pops::Real(1));
  aperture_upper.set_val(pops::Real(1));
}

}  // namespace

TEST(test_composite_fac_eb, incomplete_metric_is_refused) {
  if (pops::n_ranks() != 1)
    GTEST_SKIP();
  constexpr int Dim = 2;
  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.composite-fac.eb.incomplete");
  pops::elliptic::mg::CompositeFacPoisson<Dim> solver(two_level<Dim>(8), lane, {}, pops::Real(1));
  const auto& phi = solver.phi_level(0);
  pops::MultiFab<Dim> inverse_volume(phi.layout(), phi.distribution(), phi.local_rank(), 1,
                                     pops::Extent<Dim>{});
  inverse_volume.set_val(pops::Real(1));
  pops::elliptic::mg::WeightedPoissonFields<Dim, pops::MultiFab<Dim>::memory_space> fields;
  fields.inverse_volume = &inverse_volume;
  EXPECT_THROW(pops::elliptic::mg::validate_weighted_poisson_fields(
                   phi, fields, "composite FAC embedded boundary"),
               std::invalid_argument);
}

TEST(test_composite_fac_eb, aperture_ghosts_are_refused) {
  if (pops::n_ranks() != 1)
    GTEST_SKIP();
  constexpr int Dim = 2;
  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.composite-fac.eb.ghosts");
  pops::elliptic::mg::CompositeFacPoisson<Dim> solver(two_level<Dim>(8), lane, {}, pops::Real(1));
  const auto& phi = solver.phi_level(0);
  pops::Extent<Dim> ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    ghosts[axis] = 1;
  pops::MultiFab<Dim> active(phi.layout(), phi.distribution(), phi.local_rank(), 1,
                             pops::Extent<Dim>{});
  pops::MultiFab<Dim> inverse_volume(phi.layout(), phi.distribution(), phi.local_rank(), 1,
                                     pops::Extent<Dim>{});
  pops::MultiFab<Dim> aperture_lower(phi.layout(), phi.distribution(), phi.local_rank(), Dim,
                                     ghosts);
  pops::MultiFab<Dim> aperture_upper(phi.layout(), phi.distribution(), phi.local_rank(), Dim,
                                     pops::Extent<Dim>{});
  active.set_val(pops::Real(1));
  inverse_volume.set_val(pops::Real(1));
  aperture_lower.set_val(pops::Real(1));
  aperture_upper.set_val(pops::Real(1));
  EXPECT_THROW(solver.install_embedded_boundary(0, active, inverse_volume, aperture_lower,
                                                aperture_upper),
               std::invalid_argument);
}

TEST(test_composite_fac_eb, metric_after_nullspace_is_refused) {
  if (pops::n_ranks() != 1)
    GTEST_SKIP();
  constexpr int Dim = 2;
  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.composite-fac.eb.order");
  pops::elliptic::mg::CompositeFacPoisson<Dim> solver(two_level<Dim>(8), lane, {}, pops::Real(1));
  solver.install_nullspace(
      pops::FieldNullspacePlan<Dim>{},
      std::vector<pops::PreparedVectorDistribution<Dim>>(
          2, pops::PreparedVectorDistribution<Dim>::replicated()));
  pops::MultiFab<Dim> active;
  pops::MultiFab<Dim> inverse_volume;
  pops::MultiFab<Dim> aperture_lower;
  pops::MultiFab<Dim> aperture_upper;
  unit_metric(solver.phi_level(0), active, inverse_volume, aperture_lower, aperture_upper);
  EXPECT_THROW(solver.install_embedded_boundary(0, active, inverse_volume, aperture_lower,
                                                aperture_upper),
               std::logic_error);
}

TEST(test_composite_fac_eb, unit_metric_solves_screened_poisson) {
  if (pops::n_ranks() != 1)
    GTEST_SKIP();
  constexpr int Dim = 2;
  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.composite-fac.eb.unit");
  pops::CompositeFacOptions options;
  options.max_iters = 40;
  options.fine_sweeps = 4;
  options.rel_tol = pops::Real(5e-3);
  pops::elliptic::mg::CompositeFacPoisson<Dim> solver(two_level<Dim>(8), lane, options,
                                                      pops::Real(1));
  for (int level = 0; level < solver.n_levels(); ++level) {
    pops::MultiFab<Dim> active;
    pops::MultiFab<Dim> inverse_volume;
    pops::MultiFab<Dim> aperture_lower;
    pops::MultiFab<Dim> aperture_upper;
    unit_metric(solver.phi_level(level), active, inverse_volume, aperture_lower, aperture_upper);
    solver.install_embedded_boundary(level, active, inverse_volume, aperture_lower,
                                     aperture_upper);
    solver.rhs_level(level).set_val(pops::Real(1));
    solver.phi_level(level).set_val(pops::Real(0));
  }
  solver.install_nullspace(
      pops::FieldNullspacePlan<Dim>{},
      std::vector<pops::PreparedVectorDistribution<Dim>>(
          2, pops::PreparedVectorDistribution<Dim>::replicated()));
  const pops::SolveReport report = solver.solve();
  EXPECT_TRUE(report.solved()) << report.reason << " residual=" << report.residual_norm;
}

TEST(test_composite_fac_eb, interior_cut_stays_on_weighted_operator) {
  if (pops::n_ranks() != 1)
    GTEST_SKIP();
  constexpr int Dim = 2;
  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.composite-fac.eb.cut");
  pops::CompositeFacOptions options;
  options.max_iters = 48;
  options.fine_sweeps = 6;
  options.rel_tol = pops::Real(5e-3);
  pops::elliptic::mg::CompositeFacPoisson<Dim> solver(two_level<Dim>(8), lane, options,
                                                      pops::Real(1));
  for (int level = 0; level < solver.n_levels(); ++level) {
    pops::MultiFab<Dim> active;
    pops::MultiFab<Dim> inverse_volume;
    pops::MultiFab<Dim> aperture_lower;
    pops::MultiFab<Dim> aperture_upper;
    unit_metric(solver.phi_level(level), active, inverse_volume, aperture_lower, aperture_upper);
    if (level == 1) {
      for (std::size_t local = 0; local < active.local_size(); ++local) {
        const pops::Box<Dim> box = active.box(local);
        if (box.length(0) < 3 || box.length(1) < 3)
          continue;
        auto host = active.fab(local).create_host_mirror();
        active.fab(local).copy_to_host(host);
        pops::Index<Dim> mid{};
        mid[0] = (box.lo[0] + box.hi[0]) / 2;
        mid[1] = (box.lo[1] + box.hi[1]) / 2;
        host(static_cast<std::size_t>(mid[0] - box.lo[0]) +
             static_cast<std::size_t>(mid[1] - box.lo[1]) *
                 static_cast<std::size_t>(box.length(0))) = pops::Real(0);
        active.fab(local).copy_from_host(host);
      }
    }
    solver.install_embedded_boundary(level, active, inverse_volume, aperture_lower,
                                     aperture_upper);
    solver.rhs_level(level).set_val(pops::Real(1));
    solver.phi_level(level).set_val(pops::Real(0));
  }
  solver.install_nullspace(
      pops::FieldNullspacePlan<Dim>{},
      std::vector<pops::PreparedVectorDistribution<Dim>>(
          2, pops::PreparedVectorDistribution<Dim>::replicated()));
  const pops::SolveReport report = solver.solve();
  EXPECT_TRUE(report.solved()) << report.reason << " residual=" << report.residual_norm;
}
