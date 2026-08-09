#include <gtest/gtest.h>

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
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
using Field = pops::MultiFab<kDim>;
using Solver = pops::elliptic::mg::GeometricMG<kDim>;

struct Errors {
  double l2 = 0;
  double linf = 0;
  double rhs_mean = 0;
};

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

std::size_t storage_ordinal(const pops::Box<kDim>& storage, const pops::Index<kDim>& index) {
  std::size_t result = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < kDim; ++axis) {
    result += static_cast<std::size_t>(index[axis] - storage.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(storage.length(axis));
  }
  return result;
}

pops::EllipticBuildRequest<kDim> make_request(int cells, bool periodic) {
  const pops::Box<kDim> domain{pops::Index<kDim>{}, filled<pops::Index<kDim>>(cells - 1)};
  const pops::Geometry<kDim> geometry = pops::Geometry<kDim>::from_bounds(
      domain, pops::RealVector<kDim>{}, filled<pops::RealVector<kDim>>(pops::Real(1)));
  const pops::mesh::BoxArray<kDim> layout(std::vector<pops::Box<kDim>>{domain});
  pops::Extent<kDim> rank_extent = filled<pops::Extent<kDim>>(std::int64_t{1});
  rank_extent[0] = pops::n_ranks();
  pops::Index<kDim> local_rank{};
  local_rank[0] = pops::my_rank();
  const pops::mesh::RankSpace<kDim> ranks{pops::Index<kDim>{}, rank_extent};
  const pops::mesh::Distribution<kDim> distribution =
      pops::mesh::Distribution<kDim>::replicated(layout, ranks);
  std::array<pops::PhysicalBoundaryFace, 2 * kDim> faces{};
  if (!periodic)
    faces.fill({pops::PhysicalBoundaryKind::dirichlet, pops::Real(0)});
  std::array<bool, kDim> periodic_axes{};
  periodic_axes.fill(periodic);
  pops::RealVector<kDim> spacing{};
  for (int axis = 0; axis < kDim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  const auto topology = periodic ? pops::BoundaryTopology<kDim>::axis_periodic(periodic_axes)
                                 : pops::BoundaryTopology<kDim>::physical();
  return {geometry,
          layout,
          distribution,
          local_rank,
          pops::PhysicalBoundaryConditions<kDim>{topology, faces, spacing},
          pops::Extent<kDim>{},
          filled<pops::Extent<kDim>>(std::int64_t{1}),
          {layout.size(), 0}};
}

pops::Real manufactured_value(const pops::Geometry<kDim>& geometry, const pops::Index<kDim>& index,
                              bool periodic) {
  const pops::Real pi = std::acos(pops::Real(-1));
  const pops::Real frequency = periodic ? pops::Real(2) : pops::Real(1);
  pops::Real result = pops::Real(1);
  for (int axis = 0; axis < kDim; ++axis)
    result *= std::sin(frequency * pi * geometry.cell_coordinate(axis, index[axis]));
  return result;
}

void fill_manufactured_rhs(Field& rhs, const pops::Geometry<kDim>& geometry, bool periodic) {
  const pops::Real pi = std::acos(pops::Real(-1));
  const pops::Real frequency = periodic ? pops::Real(2) : pops::Real(1);
  const pops::Real eigenvalue = static_cast<pops::Real>(kDim) * frequency * frequency * pi * pi;
  for (std::size_t local = 0; local < rhs.local_size(); ++local) {
    auto& fab = rhs.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const auto& valid = fab.box();
    const auto& storage = fab.grown_box();
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(valid.numPts()); ++ordinal) {
      const auto index = index_from_ordinal(valid, ordinal);
      host(storage_ordinal(storage, index)) =
          eigenvalue * manufactured_value(geometry, index, periodic);
    }
    fab.copy_from_host(host);
  }

  if (!periodic)
    return;
  const pops::Real mean =
      pops::reduce_sum_local(rhs) / static_cast<pops::Real>(geometry.domain().numPts());
  for (std::size_t local = 0; local < rhs.local_size(); ++local) {
    auto& fab = rhs.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const auto& valid = fab.box();
    const auto& storage = fab.grown_box();
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(valid.numPts()); ++ordinal) {
      const auto index = index_from_ordinal(valid, ordinal);
      host(storage_ordinal(storage, index)) -= mean;
    }
    fab.copy_from_host(host);
  }
}

void install_nullspace(Solver& solver, bool periodic) {
  if (!periodic) {
    solver.install_nullspace(pops::FieldNullspacePlan<kDim>{},
                             pops::PreparedVectorDistribution<kDim>::replicated());
    return;
  }
  pops::Real measure = pops::Real(1);
  for (int axis = 0; axis < kDim; ++axis)
    measure *= solver.geom().spacing(axis);
  solver.install_nullspace(
      pops::constant_mean_zero_nullspace<kDim>("periodic-convergence", "unit-test", measure),
      pops::PreparedVectorDistribution<kDim>::replicated());
}

Errors solve_case(int cells, bool periodic) {
  auto request = make_request(cells, periodic);
  pops::elliptic::mg::GeometricMultigridOptions options;
  options.relative_tolerance = pops::Real(1e-11);
  options.absolute_tolerance = pops::Real(1e-13);
  options.maximum_cycles = 120;
  options.bottom_sweeps = 60;
  Solver solver(std::move(request), options);
  install_nullspace(solver, periodic);
  fill_manufactured_rhs(solver.rhs(), solver.geom(), periodic);
  const double rhs_mean =
      static_cast<double>(pops::reduce_sum_local(solver.rhs()) /
                          static_cast<pops::Real>(solver.geom().domain().numPts()));
  solver.phi().set_val(pops::Real(0));
  const pops::SolveReport report = solver.solve();
  EXPECT_TRUE(report.solved()) << report.reason;

  double sum_squared = 0;
  double maximum = 0;
  std::size_t count = 0;
  for (std::size_t local = 0; local < solver.phi().local_size(); ++local) {
    const auto& fab = solver.phi().fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const auto& valid = fab.box();
    const auto& storage = fab.grown_box();
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(valid.numPts()); ++ordinal) {
      const auto index = index_from_ordinal(valid, ordinal);
      const double error = static_cast<double>(host(storage_ordinal(storage, index)) -
                                               manufactured_value(solver.geom(), index, periodic));
      sum_squared += error * error;
      maximum = std::max(maximum, std::abs(error));
      ++count;
    }
  }
  maximum = pops::all_reduce_max(maximum);
  return {std::sqrt(sum_squared / static_cast<double>(count)), maximum, rhs_mean};
}

double order(double coarse, double fine) {
  return std::log(coarse / fine) / std::log(2.0);
}

void expect_second_order(bool periodic) {
  const Errors coarse = solve_case(8, periodic);
  const Errors medium = solve_case(16, periodic);
  const Errors fine = solve_case(32, periodic);
  const double l2_order = order(medium.l2, fine.l2);
  const double linf_order = order(medium.linf, fine.linf);
  EXPECT_GT(coarse.l2, medium.l2);
  EXPECT_GT(medium.l2, fine.l2);
  EXPECT_GT(l2_order, 1.8);
  EXPECT_LT(l2_order, 2.2);
  EXPECT_GT(linf_order, 1.8);
  EXPECT_LT(linf_order, 2.2);
  if (periodic)
    EXPECT_LT(std::abs(fine.rhs_mean), 1e-12);
}

}  // namespace

TEST(test_poisson_convergence, dirichlet_is_second_order_in_the_native_rank) {
  expect_second_order(false);
}

TEST(test_poisson_convergence, periodic_is_second_order_and_the_explicit_nullspace_is_compatible) {
  expect_second_order(true);
}
