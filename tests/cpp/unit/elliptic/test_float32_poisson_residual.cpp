#include <gtest/gtest.h>

#include <pops/numerics/elliptic/nd/cartesian_poisson.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#if !defined(POPS_REAL_TYPE)
#error "test_float32_poisson_residual must compile with -DPOPS_REAL_TYPE=float"
#endif

using namespace pops;
using namespace pops::elliptic::nd;
using pops::mesh::BoxArray;
using pops::mesh::Distribution;
using pops::mesh::RankSpace;

namespace {

constexpr int kDim = 2;
constexpr int kCells = 8;

Extent<kDim> uniform_extent(std::int64_t value) {
  Extent<kDim> result{};
  for (int axis = 0; axis < kDim; ++axis)
    result[axis] = value;
  return result;
}

RealVector<kDim> uniform_coordinate(Real value) {
  RealVector<kDim> result{};
  for (int axis = 0; axis < kDim; ++axis)
    result[axis] = value;
  return result;
}

Index<kDim> index_from_ordinal(const Box<kDim>& box, std::size_t ordinal) {
  Index<kDim> result{};
  for (int axis = 0; axis < kDim; ++axis) {
    const std::size_t length = static_cast<std::size_t>(box.length(axis));
    result[axis] = box.lo[axis] + static_cast<int>(ordinal % length);
    ordinal /= length;
  }
  return result;
}

std::size_t storage_ordinal(const Box<kDim>& box, const Index<kDim>& index) {
  std::size_t result = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < kDim; ++axis) {
    result += static_cast<std::size_t>(index[axis] - box.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(box.length(axis));
  }
  return result;
}

Real mode_value(const Geometry<kDim>& geometry, const Index<kDim>& index) {
  const Real pi = std::acos(Real{-1});
  Real value = Real{1};
  for (int axis = 0; axis < kDim; ++axis)
    value *= std::sin(Real{2} * pi * geometry.cell_coordinate(axis, index[axis]));
  return value;
}

}  // namespace

TEST(test_float32_poisson_residual, real_is_binary32_not_a_cast) {
  static_assert(std::is_same_v<Real, float>, "this suite must instantiate pops::Real as binary32");
  static_assert(sizeof(Real) == 4);
  static_assert(std::numeric_limits<Real>::digits == 24);
  static_assert(std::numeric_limits<Real>::is_iec559);
  EXPECT_EQ(Real(1) + Real(1e-8), Real(1));
  EXPECT_NE(static_cast<double>(1) + 1e-8, static_cast<double>(1));
}

TEST(test_float32_poisson_residual, manufactured_periodic_mode_matches_binary64_reference) {
  const ExecutionLane lane = ExecutionLane::world("tests.float32-poisson.manufactured");
  const Box<kDim> domain{Index<kDim>{}, Index<kDim>{kCells - 1, kCells - 1}};
  const Geometry<kDim> geometry = Geometry<kDim>::from_bounds(
      domain, uniform_coordinate(Real{0}), uniform_coordinate(Real{1}));
  const BoxArray<kDim> layout = BoxArray<kDim>::from_domain(domain, uniform_extent(kCells));
  const RankSpace<kDim> ranks{Index<kDim>{}, uniform_extent(1)};
  const auto distribution = Distribution<kDim>::replicated(layout, ranks);
  std::array<bool, kDim> periodic{};
  periodic.fill(true);
  const BoundaryTopology<kDim> topology = BoundaryTopology<kDim>::axis_periodic(periodic);
  auto options = CartesianPoissonOptions<kDim>::from_topology(topology,
                                                             CartesianBoundaryKind::dirichlet);
  options.relative_tolerance = Real{1e-5};
  options.maximum_iterations = 32;
  CartesianPoissonSolver<kDim> solver(geometry, layout, distribution, Index<kDim>{}, topology,
                                      options, lane);
  Real cell_measure = Real(1);
  for (int axis = 0; axis < kDim; ++axis)
    cell_measure *= geometry.spacing(axis);
  solver.install_nullspace(
      constant_mean_zero_nullspace<kDim>("float32-constant-mode", "unit-test", cell_measure),
      PreparedVectorDistribution<kDim>::replicated());

  MultiFab<kDim> warm_start(layout, distribution, Index<kDim>{}, 1, uniform_extent(1));
  warm_start.set_val(Real{0});

  const Real pi = std::acos(Real{-1});
  const Real angle = pi / static_cast<Real>(kCells);
  const Real inverse_spacing = Real{1} / geometry.spacing(0);
  const Real eigenvalue = static_cast<Real>(kDim) * Real{4} * std::sin(angle) * std::sin(angle) *
                          inverse_spacing * inverse_spacing;
  for (std::size_t local = 0; local < solver.rhs().local_size(); ++local) {
    auto& fab = solver.rhs().fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<kDim>& valid = fab.box();
    const Box<kDim>& storage = fab.grown_box();
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(valid.numPts()); ++ordinal) {
      const Index<kDim> index = index_from_ordinal(valid, ordinal);
      host(storage_ordinal(storage, index)) = eigenvalue * mode_value(geometry, index);
    }
    fab.copy_from_host(host);
  }

  const SolveReport report = solver.solve(warm_start, lane);
  ASSERT_TRUE(report.solved_value_available()) << report.reason;
  EXPECT_LE(report.iters, 8);
  EXPECT_LT(report.residual_norm, Real{5e-4});

  Real maximum_error = 0;
  Real sample_00 = 0;
  Real sample_10 = 0;
  Real sample_11 = 0;
  for (std::size_t local = 0; local < solver.candidate().local_size(); ++local) {
    const auto& fab = solver.candidate().fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<kDim>& valid = fab.box();
    const Box<kDim>& storage = fab.grown_box();
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(valid.numPts()); ++ordinal) {
      const Index<kDim> index = index_from_ordinal(valid, ordinal);
      const Real value = host(storage_ordinal(storage, index));
      maximum_error = std::max(maximum_error, std::abs(value - mode_value(geometry, index)));
      if (index[0] == 0 && index[1] == 0)
        sample_00 = value;
      if (index[0] == 1 && index[1] == 0)
        sample_10 = value;
      if (index[0] == 1 && index[1] == 1)
        sample_11 = value;
    }
  }
  EXPECT_LT(maximum_error, Real{5e-4});

  // Independent binary64 manufactured samples at the same cell centers.
  EXPECT_NEAR(static_cast<double>(sample_00), 0.14644660940672624, 5e-5);
  EXPECT_NEAR(static_cast<double>(sample_10), 0.35355339059327379, 5e-5);
  EXPECT_NEAR(static_cast<double>(sample_11), 0.85355339059327373, 5e-5);
}
