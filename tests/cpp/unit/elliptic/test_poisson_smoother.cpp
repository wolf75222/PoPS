#include <gtest/gtest.h>

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

constexpr int kDim = pops::kNativeDimension;
using Solver = pops::elliptic::mg::GeometricMG<kDim>;

template <class Ranked, class Value>
Ranked filled(Value value) {
  Ranked result{};
  for (int axis = 0; axis < kDim; ++axis)
    result[axis] = value;
  return result;
}

pops::Index<kDim> index_from_ordinal(const pops::Box<kDim>& box, std::size_t ordinal) {
  pops::Index<kDim> result{};
  for (int axis = 0; axis < kDim; ++axis) {
    const std::size_t length = static_cast<std::size_t>(box.length(axis));
    result[axis] = box.lo[axis] + static_cast<int>(ordinal % length);
    ordinal /= length;
  }
  return result;
}

std::size_t storage_ordinal(const pops::Box<kDim>& box, const pops::Index<kDim>& index) {
  std::size_t result = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < kDim; ++axis) {
    result += static_cast<std::size_t>(index[axis] - box.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(box.length(axis));
  }
  return result;
}

pops::EllipticBuildRequest<kDim> request(int cells) {
  const pops::Box<kDim> domain{pops::Index<kDim>{}, filled<pops::Index<kDim>>(cells - 1)};
  const auto geometry = pops::Geometry<kDim>::from_bounds(
      domain, pops::RealVector<kDim>{}, filled<pops::RealVector<kDim>>(pops::Real(1)));
  const pops::mesh::BoxArray<kDim> layout(std::vector<pops::Box<kDim>>{domain});
  pops::Extent<kDim> rank_extent = filled<pops::Extent<kDim>>(std::int64_t{1});
  rank_extent[0] = pops::n_ranks();
  pops::Index<kDim> local_rank{};
  local_rank[0] = pops::my_rank();
  const pops::mesh::RankSpace<kDim> ranks{pops::Index<kDim>{}, rank_extent};
  const auto distribution = pops::mesh::Distribution<kDim>::replicated(layout, ranks);
  std::array<pops::PhysicalBoundaryFace, 2 * kDim> faces{};
  faces.fill(pops::PhysicalBoundaryFace{pops::PhysicalBoundaryKind::dirichlet, pops::Real(0)});
  pops::RealVector<kDim> spacing{};
  for (int axis = 0; axis < kDim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  return {geometry,
          layout,
          distribution,
          local_rank,
          pops::PhysicalBoundaryConditions<kDim>{pops::BoundaryTopology<kDim>::physical(), faces,
                                                 spacing},
          pops::Extent<kDim>{},
          filled<pops::Extent<kDim>>(std::int64_t{1}),
          {layout.size(), 0}};
}

pops::Real exact(const pops::Geometry<kDim>& geometry, const pops::Index<kDim>& index) {
  const pops::Real pi = std::acos(pops::Real(-1));
  pops::Real result = pops::Real(1);
  for (int axis = 0; axis < kDim; ++axis)
    result *= std::sin(pi * geometry.cell_coordinate(axis, index[axis]));
  return result;
}

void fill_rhs(Solver& solver) {
  const pops::Real pi = std::acos(pops::Real(-1));
  const pops::Real eigenvalue = static_cast<pops::Real>(kDim) * pi * pi;
  for (std::size_t local = 0; local < solver.rhs().local_size(); ++local) {
    auto& fab = solver.rhs().fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(fab.box().numPts());
         ++ordinal) {
      const auto index = index_from_ordinal(fab.box(), ordinal);
      host(storage_ordinal(fab.grown_box(), index)) = eigenvalue * exact(solver.geom(), index);
    }
    fab.copy_from_host(host);
  }
}

double maximum_error(const Solver& solver) {
  double result = 0;
  for (std::size_t local = 0; local < solver.phi().local_size(); ++local) {
    const auto& fab = solver.phi().fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(fab.box().numPts());
         ++ordinal) {
      const auto index = index_from_ordinal(fab.box(), ordinal);
      result = std::max(result,
                        std::abs(static_cast<double>(host(storage_ordinal(fab.grown_box(), index)) -
                                                     exact(solver.geom(), index))));
    }
  }
  return pops::all_reduce_max(result);
}

}  // namespace

TEST(test_poisson_smoother,
     prepared_multigrid_smoothing_reduces_the_residual_and_recovers_the_exact_mode) {
  pops::elliptic::mg::GeometricMultigridOptions options;
  options.relative_tolerance = pops::Real(1e-10);
  options.absolute_tolerance = pops::Real(1e-12);
  options.maximum_cycles = 100;
  options.bottom_sweeps = 60;
  Solver solver(request(16), options);
  solver.install_nullspace(pops::FieldNullspacePlan<kDim>{},
                           pops::PreparedVectorDistribution<kDim>::replicated());
  fill_rhs(solver);
  solver.phi().set_val(pops::Real(0));

  const pops::SolveReport report = solver.solve();
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_GT(report.reference_residual_norm, pops::Real(0));
  EXPECT_LT(report.residual_norm, pops::Real(1e-8) * report.reference_residual_norm);
  EXPECT_LT(maximum_error(solver), 0.03);
}
