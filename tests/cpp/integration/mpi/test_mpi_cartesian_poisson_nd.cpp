#include <gtest/gtest.h>

#include <pops/numerics/elliptic/nd/cartesian_poisson.hpp>
#include <pops/parallel/comm.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

using namespace pops;
using namespace pops::elliptic::nd;
using pops::mesh::BoxArray;
using pops::mesh::Distribution;
using pops::mesh::RankSpace;

namespace {

template <int Dim>
Extent<Dim> uniform_extent(std::int64_t value) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
Index<Dim> rank_coordinate(int rank) {
  Index<Dim> result{};
  result[0] = rank;
  return result;
}

template <int Dim>
Index<Dim> index_from_ordinal(const Box<Dim>& box, std::size_t ordinal) {
  Index<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    const auto length = static_cast<std::size_t>(box.length(axis));
    result[axis] = box.lo[axis] + static_cast<int>(ordinal % length);
    ordinal /= length;
  }
  return result;
}

template <int Dim>
std::size_t storage_ordinal(const Box<Dim>& box, const Index<Dim>& index) {
  std::size_t result = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    result += static_cast<std::size_t>(index[axis] - box.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(box.length(axis));
  }
  return result;
}

template <int Dim>
Real manufactured_value(const Geometry<Dim>& geometry, const Index<Dim>& index) {
  const Real pi = std::acos(Real{-1});
  Real result = Real{1};
  for (int axis = 0; axis < Dim; ++axis)
    result *= std::sin(pi * geometry.cell_coordinate(axis, index[axis]));
  return result;
}

template <int Dim>
void prove_distributed_manufactured_mode() {
  constexpr int cells = 8;
  Index<Dim> upper{};
  RealVector<Dim> lower_coordinate{};
  RealVector<Dim> upper_coordinate{};
  for (int axis = 0; axis < Dim; ++axis) {
    upper[axis] = cells - 1;
    upper_coordinate[axis] = Real{1};
  }
  const Box<Dim> domain{Index<Dim>{}, upper};
  const Geometry<Dim> geometry =
      Geometry<Dim>::from_bounds(domain, lower_coordinate, upper_coordinate);

  Index<Dim> first_upper = upper;
  Index<Dim> second_lower{};
  first_upper[0] = cells / 2 - 1;
  second_lower[0] = cells / 2;
  const BoxArray<Dim> layout(
      std::vector<Box<Dim>>{Box<Dim>{Index<Dim>{}, first_upper}, Box<Dim>{second_lower, upper}});
  Extent<Dim> rank_extent{};
  rank_extent[0] = 3;
  for (int axis = 1; axis < Dim; ++axis)
    rank_extent[axis] = 1;
  const RankSpace<Dim> rank_space{Index<Dim>{}, rank_extent};
  const Distribution<Dim> distribution = Distribution<Dim>::partitioned(
      layout, rank_space,
      std::vector<Index<Dim>>{rank_coordinate<Dim>(0), rank_coordinate<Dim>(1)});
  const Index<Dim> local_rank = rank_coordinate<Dim>(my_rank());

  std::array<bool, Dim> periodic{};
  const BoundaryTopology<Dim> topology = BoundaryTopology<Dim>::axis_periodic(periodic);
  auto options =
      CartesianPoissonOptions<Dim>::from_topology(topology, CartesianBoundaryKind::dirichlet);
  options.relative_tolerance = Real{1e-12};
  options.absolute_tolerance = Real{1e-14};
  options.maximum_iterations = 32;
  CartesianPoissonSolver<Dim> solver(geometry, layout, distribution, local_rank, topology, options);
  MultiFab<Dim> warm_start(layout, distribution, local_rank, 1, uniform_extent<Dim>(1));
  warm_start.set_val(Real{0});

  const Real pi = std::acos(Real{-1});
  const Real angle = pi / static_cast<Real>(2 * cells);
  const Real inverse_spacing = Real{1} / geometry.spacing(0);
  const Real eigenvalue = static_cast<Real>(Dim) * Real{4} * std::sin(angle) * std::sin(angle) *
                          inverse_spacing * inverse_spacing;
  for (std::size_t local = 0; local < solver.rhs().local_size(); ++local) {
    auto& fab = solver.rhs().fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(fab.box().numPts());
         ++ordinal) {
      const Index<Dim> index = index_from_ordinal(fab.box(), ordinal);
      host(storage_ordinal(fab.grown_box(), index)) =
          eigenvalue * manufactured_value(geometry, index);
    }
    fab.copy_from_host(host);
  }

  const SolveReport report = solver.solve(warm_start);
  ASSERT_TRUE(report.solved_value_available()) << report.reason;
  Real local_error = Real{0};
  for (std::size_t local = 0; local < solver.candidate().local_size(); ++local) {
    const auto& fab = solver.candidate().fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(fab.box().numPts());
         ++ordinal) {
      const Index<Dim> index = index_from_ordinal(fab.box(), ordinal);
      local_error = std::max(local_error, std::abs(host(storage_ordinal(fab.grown_box(), index)) -
                                                   manufactured_value(geometry, index)));
    }
  }
  EXPECT_LT(all_reduce_max(static_cast<double>(local_error)), 5e-10);
  if (my_rank() == 2)
    EXPECT_EQ(solver.candidate().local_size(), 0U);
}

}  // namespace

TEST(test_mpi_cartesian_poisson_nd,
     distributed_exact_ranked_solve_includes_a_rank_without_local_patches) {
  Kokkos::ScopeGuard kokkos;
  ASSERT_EQ(n_ranks(), 3);
  prove_distributed_manufactured_mode<1>();
  prove_distributed_manufactured_mode<2>();
  prove_distributed_manufactured_mode<3>();
}
