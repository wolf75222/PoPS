#include <gtest/gtest.h>

#include <pops/mesh/layout/distribution.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/elliptic/poisson/poisson_fft_multifab.hpp>
#include <pops/parallel/comm.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

constexpr pops::Real kTwoPi = pops::Real(6.283185307179586476925286766559005768L);

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
                                        pops::mesh::BoxArray<Dim> boxes, bool periodic) {
  pops::Extent<Dim> rank_extent = ranked<pops::Extent<Dim>, Dim>(std::int64_t{1});
  rank_extent[0] = pops::n_ranks();
  pops::Index<Dim> local_rank{};
  local_rank[0] = pops::my_rank();
  const pops::mesh::RankSpace<Dim> ranks{pops::Index<Dim>{}, rank_extent};
  const auto distribution = pops::mesh::Distribution<Dim>::replicated(boxes, ranks);
  std::array<pops::PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  if (!periodic)
    faces.fill({pops::PhysicalBoundaryKind::dirichlet, pops::Real(0)});
  pops::RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  std::array<bool, Dim> periodic_axes{};
  periodic_axes.fill(true);
  const std::size_t pairs = boxes.size() * (boxes.size() - 1) / 2;
  return {geometry,
          std::move(boxes),
          distribution,
          local_rank,
          pops::PhysicalBoundaryConditions<Dim>{
              periodic ? pops::BoundaryTopology<Dim>::axis_periodic(periodic_axes)
                       : pops::BoundaryTopology<Dim>::physical(),
              faces, spacing},
          pops::Extent<Dim>{},
          ranked<pops::Extent<Dim>, Dim>(std::int64_t{1}),
          {distribution.box_count(), pairs}};
}

template <int Dim>
pops::elliptic::mg::CompositeFacBuildRequest<Dim> two_level_periodic(int coarse_cells) {
  const pops::Box<Dim> domain{pops::Index<Dim>{},
                              ranked<pops::Index<Dim>, Dim>(coarse_cells - 1)};
  const auto coarse = pops::Geometry<Dim>::from_bounds(
      domain, pops::RealVector<Dim>{}, ranked<pops::RealVector<Dim>, Dim>(pops::Real(1)));
  const auto fine = coarse.refine(ranked<pops::Extent<Dim>, Dim>(std::int64_t{2}));
  pops::Index<Dim> fine_lo{};
  pops::Index<Dim> fine_hi{};
  for (int axis = 0; axis < Dim; ++axis) {
    fine_lo[axis] = coarse_cells / 2;
    fine_hi[axis] = 3 * coarse_cells / 2 - 1;
  }
  return {{request<Dim>(coarse, pops::mesh::BoxArray<Dim>(std::vector<pops::Box<Dim>>{domain}),
                        true),
           request<Dim>(fine, pops::mesh::BoxArray<Dim>(std::vector<pops::Box<Dim>>{{fine_lo, fine_hi}}),
                        true)},
          {pops::amr::RefinementRatio<Dim>{filled<Dim>(2)}}};
}

template <int Dim>
void fill_periodic_mode(pops::MultiFab<Dim>& field, const pops::Geometry<Dim>& geometry) {
  // A = -laplacian; continuous mode sin(2π x_i) has eigenvalue Dim * (2π)^2 on the unit box.
  const pops::Real eigenvalue = static_cast<pops::Real>(Dim) * kTwoPi * kTwoPi;
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    const pops::Box<Dim>& valid = fab.box();
    const pops::Box<Dim>& storage = fab.grown_box();
    for (std::size_t n = 0; n < static_cast<std::size_t>(valid.numPts()); ++n) {
      const auto index = index_from_ordinal(valid, n);
      pops::Real value = pops::Real(1);
      for (int axis = 0; axis < Dim; ++axis)
        value *= std::sin(kTwoPi * geometry.cell_coordinate(axis, index[axis]));
      host(storage_ordinal(storage, index)) = eigenvalue * value;
    }
    fab.copy_from_host(host);
  }
}

}  // namespace

TEST(test_composite_fac_fft_coarse, dirichlet_and_variable_k_stay_ineligible) {
  if (pops::n_ranks() != 1)
    GTEST_SKIP();
  constexpr int Dim = 2;
  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.fft-bottom.ineligible");
  const pops::Box<Dim> domain{pops::Index<Dim>{}, ranked<pops::Index<Dim>, Dim>(15)};
  const auto geometry = pops::Geometry<Dim>::from_bounds(
      domain, pops::RealVector<Dim>{}, ranked<pops::RealVector<Dim>, Dim>(pops::Real(1)));
  const auto boxes = pops::mesh::BoxArray<Dim>(std::vector<pops::Box<Dim>>{domain});
  const auto dirichlet = request<Dim>(geometry, boxes, false);
  EXPECT_FALSE(pops::elliptic::PoissonFftMultiFabAdapter<Dim>::classify(dirichlet, lane, pops::Real(0),
                                                                       false, false)
                   .eligible());
  const auto periodic = request<Dim>(geometry, boxes, true);
  EXPECT_TRUE(pops::elliptic::PoissonFftMultiFabAdapter<Dim>::classify(periodic, lane, pops::Real(0),
                                                                      false, false)
                  .eligible());
  EXPECT_FALSE(pops::elliptic::PoissonFftMultiFabAdapter<Dim>::classify(periodic, lane, pops::Real(1),
                                                                       false, false)
                   .eligible());
  EXPECT_FALSE(pops::elliptic::PoissonFftMultiFabAdapter<Dim>::classify(periodic, lane, pops::Real(0),
                                                                       true, false)
                   .eligible());
  EXPECT_FALSE(pops::elliptic::PoissonFftMultiFabAdapter<Dim>::classify(periodic, lane, pops::Real(0),
                                                                       false, true)
                   .eligible());
}

TEST(test_composite_fac_fft_coarse, geometric_mg_periodic_uses_fft_bottom) {
  if (pops::n_ranks() != 1)
    GTEST_SKIP();
  constexpr int Dim = 2;
  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.fft-bottom.geometric-mg");
  const pops::Box<Dim> domain{pops::Index<Dim>{}, ranked<pops::Index<Dim>, Dim>(15)};
  const auto geometry = pops::Geometry<Dim>::from_bounds(
      domain, pops::RealVector<Dim>{}, ranked<pops::RealVector<Dim>, Dim>(pops::Real(1)));
  pops::elliptic::mg::GeometricMultigridOptions options;
  options.allow_coarsening = false;
  pops::elliptic::mg::GeometricMG<Dim> solver(request<Dim>(geometry, pops::mesh::BoxArray<Dim>(
                                                                        std::vector<pops::Box<Dim>>{domain}),
                                                            true),
                                              lane, options);
  EXPECT_TRUE(solver.fft_coarse_prepared());
  EXPECT_EQ(solver.fft_coarse_kind(), pops::elliptic::PoissonFftBottomKind::native_solver);
  const pops::Real measure = geometry.spacing(0) * geometry.spacing(1);
  solver.install_nullspace(
      pops::constant_mean_zero_nullspace<Dim>("periodic-fft-mg", "unit-test", measure),
      pops::PreparedVectorDistribution<Dim>::replicated());
  fill_periodic_mode(solver.rhs(), geometry);
  solver.phi().set_val(pops::Real(0));
  const pops::SolveReport report = solver.solve();
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_TRUE(solver.used_fft_coarse());
  EXPECT_LT(report.residual_norm, pops::Real(1e-10));
}

TEST(test_composite_fac_fft_coarse, two_level_periodic_fac_uses_fft_on_coarse) {
  if (pops::n_ranks() != 1)
    GTEST_SKIP();
  constexpr int Dim = 2;
  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.fft-bottom.composite-fac");
  auto hierarchy = two_level_periodic<Dim>(16);
  const pops::Geometry<Dim> coarse_geometry = hierarchy.levels.front().geometry;
  const pops::Geometry<Dim> fine_geometry = hierarchy.levels.back().geometry;
  const pops::Real coarse_measure = coarse_geometry.spacing(0) * coarse_geometry.spacing(1);
  const pops::Real fine_measure = fine_geometry.spacing(0) * fine_geometry.spacing(1);
  pops::elliptic::mg::CompositeFacPoisson<Dim> solver(std::move(hierarchy), lane);
  EXPECT_TRUE(solver.fft_coarse_prepared());
  EXPECT_EQ(solver.fft_coarse_kind(), pops::elliptic::PoissonFftBottomKind::native_solver);
  pops::FieldNullspacePlan<Dim> plan = pops::constant_mean_zero_nullspace<Dim>(
      "periodic-fft-fac", "unit-test", coarse_measure);
  plan.bases.front().cell_measure = {coarse_measure, fine_measure};
  solver.install_nullspace(std::move(plan), {pops::PreparedVectorDistribution<Dim>::replicated(),
                                             pops::PreparedVectorDistribution<Dim>::replicated()});
  fill_periodic_mode(solver.rhs_level(0), coarse_geometry);
  fill_periodic_mode(solver.rhs_level(1), fine_geometry);
  solver.phi_level(0).set_val(pops::Real(0));
  solver.phi_level(1).set_val(pops::Real(0));
  const pops::SolveReport report = solver.solve();
  ASSERT_TRUE(report.solved()) << report.reason << " residual=" << report.residual_norm;
  EXPECT_TRUE(solver.used_fft_coarse());
  EXPECT_LT(report.rel_residual, pops::Real(1e-6));
}

TEST(test_composite_fac_fft_coarse, dirichlet_fac_does_not_prepare_fft) {
  if (pops::n_ranks() != 1)
    GTEST_SKIP();
  constexpr int Dim = 2;
  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.fft-bottom.dirichlet-fac");
  const pops::Box<Dim> domain{pops::Index<Dim>{}, ranked<pops::Index<Dim>, Dim>(7)};
  const auto coarse = pops::Geometry<Dim>::from_bounds(
      domain, pops::RealVector<Dim>{}, ranked<pops::RealVector<Dim>, Dim>(pops::Real(1)));
  const auto fine = coarse.refine(ranked<pops::Extent<Dim>, Dim>(std::int64_t{2}));
  pops::Index<Dim> fine_lo{};
  pops::Index<Dim> fine_hi{};
  for (int axis = 0; axis < Dim; ++axis) {
    fine_lo[axis] = 4;
    fine_hi[axis] = 11;
  }
  pops::elliptic::mg::CompositeFacBuildRequest<Dim> hierarchy{
      {request<Dim>(coarse, pops::mesh::BoxArray<Dim>(std::vector<pops::Box<Dim>>{domain}), false),
       request<Dim>(fine, pops::mesh::BoxArray<Dim>(std::vector<pops::Box<Dim>>{{fine_lo, fine_hi}}),
                    false)},
      {pops::amr::RefinementRatio<Dim>{filled<Dim>(2)}}};
  pops::elliptic::mg::CompositeFacPoisson<Dim> solver(std::move(hierarchy), lane);
  EXPECT_FALSE(solver.fft_coarse_prepared());
}
