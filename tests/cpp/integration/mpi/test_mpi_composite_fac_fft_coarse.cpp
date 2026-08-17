#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>
#include <pops/numerics/elliptic/poisson/poisson_fft_multifab.hpp>
#include <pops/parallel/comm.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

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
                                        pops::mesh::BoxArray<Dim> boxes) {
  pops::Extent<Dim> rank_extent = ranked<pops::Extent<Dim>, Dim>(std::int64_t{1});
  rank_extent[0] = pops::n_ranks();
  pops::Index<Dim> local_rank{};
  local_rank[0] = pops::my_rank();
  const pops::mesh::RankSpace<Dim> ranks{pops::Index<Dim>{}, rank_extent};
  const auto distribution = pops::mesh::Distribution<Dim>::replicated(boxes, ranks);
  std::array<pops::PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
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
          pops::PhysicalBoundaryConditions<Dim>{pops::BoundaryTopology<Dim>::axis_periodic(periodic_axes),
                                                faces, spacing},
          pops::Extent<Dim>{},
          ranked<pops::Extent<Dim>, Dim>(std::int64_t{1}),
          {distribution.box_count(), pairs}};
}

template <int Dim>
void fill_periodic_mode(pops::MultiFab<Dim>& field, const pops::Geometry<Dim>& geometry) {
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

template <int Dim>
void expect_replicated_periodic_fft_coarse() {
  const pops::ExecutionLane lane =
      pops::ExecutionLane::world("tests.fft-bottom.mpi-replicated-fac");
  const int coarse_cells = 16;
  const pops::Box<Dim> domain{pops::Index<Dim>{}, ranked<pops::Index<Dim>, Dim>(coarse_cells - 1)};
  const auto coarse = pops::Geometry<Dim>::from_bounds(
      domain, pops::RealVector<Dim>{}, ranked<pops::RealVector<Dim>, Dim>(pops::Real(1)));
  const auto fine = coarse.refine(ranked<pops::Extent<Dim>, Dim>(std::int64_t{2}));
  pops::Index<Dim> fine_lo{};
  pops::Index<Dim> fine_hi{};
  for (int axis = 0; axis < Dim; ++axis) {
    fine_lo[axis] = coarse_cells / 2;
    fine_hi[axis] = 3 * coarse_cells / 2 - 1;
  }
  pops::elliptic::mg::CompositeFacBuildRequest<Dim> hierarchy{
      {request<Dim>(coarse, pops::mesh::BoxArray<Dim>(std::vector<pops::Box<Dim>>{domain})),
       request<Dim>(fine, pops::mesh::BoxArray<Dim>(std::vector<pops::Box<Dim>>{{fine_lo, fine_hi}}))},
      {pops::amr::RefinementRatio<Dim>{filled<Dim>(2)}}};
  pops::elliptic::mg::CompositeFacPoisson<Dim> solver(std::move(hierarchy), lane);
  EXPECT_TRUE(solver.fft_coarse_prepared());
  EXPECT_EQ(solver.fft_coarse_kind(),
            pops::elliptic::PoissonFftBottomKind::replicated_slab_rewrite);
  const pops::Real coarse_measure = coarse.spacing(0) * coarse.spacing(1);
  const pops::Real fine_measure = fine.spacing(0) * fine.spacing(1);
  pops::FieldNullspacePlan<Dim> plan = pops::constant_mean_zero_nullspace<Dim>(
      "periodic-fft-fac-mpi", "unit-test", coarse_measure);
  plan.bases.front().cell_measure = {coarse_measure, fine_measure};
  solver.install_nullspace(std::move(plan), {pops::PreparedVectorDistribution<Dim>::replicated(),
                                             pops::PreparedVectorDistribution<Dim>::replicated()});
  fill_periodic_mode(solver.rhs_level(0), coarse);
  fill_periodic_mode(solver.rhs_level(1), fine);
  solver.phi_level(0).set_val(pops::Real(0));
  solver.phi_level(1).set_val(pops::Real(0));
  const pops::SolveReport report = solver.solve();
  EXPECT_TRUE(report.solved()) << report.reason << " residual=" << report.residual_norm;
  EXPECT_TRUE(solver.used_fft_coarse());
  EXPECT_LT(report.rel_residual, pops::Real(1e-5));
}

int run_fft_coarse(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    EXPECT_EQ(pops::n_ranks(), 2);
    if (pops::n_ranks() == 2)
      expect_replicated_periodic_fft_coarse<2>();
    result = ::testing::Test::HasFailure() ? 1 : 0;
  }
  pops::comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_composite_fac_fft_coarse, ReplicatedPeriodicCoarseUsesFft) {
  EXPECT_EQ(pops::test::RunTestBody(&run_fft_coarse, "test_mpi_composite_fac_fft_coarse"), 0);
}
