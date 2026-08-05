#include <gtest/gtest.h>

#include <pops/numerics/elliptic/nd/cartesian_poisson.hpp>
#include <pops/runtime/system/exact_named_field.hpp>

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
RealVector<Dim> uniform_coordinate(Real value) {
  RealVector<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
Index<Dim> index_from_ordinal(const Box<Dim>& box, std::size_t ordinal) {
  Index<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t length = static_cast<std::size_t>(box.length(axis));
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
Real mode_value(const Geometry<Dim>& geometry, const Index<Dim>& index,
                CartesianBoundaryKind boundary) {
  const Real pi = std::acos(Real{-1});
  Real value = Real{1};
  for (int axis = 0; axis < Dim; ++axis) {
    const Real coordinate = geometry.cell_coordinate(axis, index[axis]);
    if (boundary == CartesianBoundaryKind::periodic)
      value *= std::sin(Real{2} * pi * coordinate);
    else if (boundary == CartesianBoundaryKind::dirichlet)
      value *= std::sin(pi * coordinate);
    else
      value *= std::cos(pi * coordinate);
  }
  return value;
}

template <int Dim>
void fill_rhs(MultiFab<Dim>& rhs, const Geometry<Dim>& geometry, CartesianBoundaryKind boundary,
              Real eigenvalue) {
  for (std::size_t local = 0; local < rhs.local_size(); ++local) {
    auto& fab = rhs.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& storage = fab.grown_box();
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(valid.numPts()); ++ordinal) {
      const Index<Dim> index = index_from_ordinal(valid, ordinal);
      host(storage_ordinal(storage, index)) = eigenvalue * mode_value(geometry, index, boundary);
    }
    fab.copy_from_host(host);
  }
}

template <int Dim>
Real maximum_error(const MultiFab<Dim>& solution, const Geometry<Dim>& geometry,
                   CartesianBoundaryKind boundary) {
  Real result = 0;
  for (std::size_t local = 0; local < solution.local_size(); ++local) {
    const auto& fab = solution.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& storage = fab.grown_box();
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(valid.numPts()); ++ordinal) {
      const Index<Dim> index = index_from_ordinal(valid, ordinal);
      result = std::max(result, std::abs(host(storage_ordinal(storage, index)) -
                                         mode_value(geometry, index, boundary)));
    }
  }
  return result;
}

template <int Dim>
void expect_manufactured_mode(CartesianBoundaryKind boundary) {
  constexpr int cells = 8;
  const Box<Dim> domain{Index<Dim>{}, [&] {
                          Index<Dim> upper{};
                          for (int axis = 0; axis < Dim; ++axis)
                            upper[axis] = cells - 1;
                          return upper;
                        }()};
  const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(
      domain, uniform_coordinate<Dim>(Real{0}), uniform_coordinate<Dim>(Real{1}));
  const BoxArray<Dim> layout = BoxArray<Dim>::from_domain(domain, uniform_extent<Dim>(cells));
  const RankSpace<Dim> ranks{Index<Dim>{}, uniform_extent<Dim>(1)};
  const auto distribution = Distribution<Dim>::replicated(layout, ranks);
  std::array<bool, Dim> periodic{};
  periodic.fill(boundary == CartesianBoundaryKind::periodic);
  const BoundaryTopology<Dim> topology = BoundaryTopology<Dim>::axis_periodic(periodic);
  auto options = CartesianPoissonOptions<Dim>::from_topology(
      topology,
      boundary == CartesianBoundaryKind::periodic ? CartesianBoundaryKind::dirichlet : boundary);
  options.relative_tolerance = Real{1e-12};
  options.maximum_iterations = 32;
  CartesianPoissonSolver<Dim> solver(geometry, layout, distribution, Index<Dim>{}, topology,
                                     options);
  MultiFab<Dim> warm_start(layout, distribution, Index<Dim>{}, 1, uniform_extent<Dim>(1));
  warm_start.set_val(Real{0});

  const Real pi = std::acos(Real{-1});
  const Real angle = boundary == CartesianBoundaryKind::periodic
                         ? pi / static_cast<Real>(cells)
                         : pi / static_cast<Real>(2 * cells);
  const Real inverse_spacing = Real{1} / geometry.spacing(0);
  const Real eigenvalue = static_cast<Real>(Dim) * Real{4} * std::sin(angle) * std::sin(angle) *
                          inverse_spacing * inverse_spacing;
  fill_rhs(solver.rhs(), geometry, boundary, eigenvalue);

  const SolveReport report = solver.solve(warm_start);
  ASSERT_TRUE(report.solved_value_available()) << report.reason;
  EXPECT_LE(report.iters, 2);
  EXPECT_LT(maximum_error(solver.candidate(), geometry, boundary), Real{5e-11});
}

template <int Dim>
void expect_all_boundaries() {
  expect_manufactured_mode<Dim>(CartesianBoundaryKind::periodic);
  expect_manufactured_mode<Dim>(CartesianBoundaryKind::dirichlet);
  expect_manufactured_mode<Dim>(CartesianBoundaryKind::neumann);
}

}  // namespace

TEST(test_cartesian_poisson_nd, manufactured_modes_solve_in_exact_rank_one_two_and_three) {
  expect_all_boundaries<1>();
  expect_all_boundaries<2>();
  expect_all_boundaries<3>();
}

TEST(test_cartesian_poisson_nd, named_provider_publishes_only_after_candidate_acceptance) {
  constexpr int cells = 8;
  const Box<2> domain{Index<2>{0, 0}, Index<2>{cells - 1, cells - 1}};
  const Geometry<2> geometry =
      Geometry<2>::from_bounds(domain, RealVector<2>{0, 0}, RealVector<2>{1, 1});
  const BoxArray<2> layout = BoxArray<2>::from_domain(domain, Extent<2>{cells, cells});
  const RankSpace<2> ranks{Index<2>{}, Extent<2>{1, 1}};
  const auto distribution = Distribution<2>::replicated(layout, ranks);
  const BoundaryTopology<2> topology = BoundaryTopology<2>::axis_periodic({true, true});
  auto options = CartesianPoissonOptions<2>::from_topology(topology);
  options.relative_tolerance = Real{1e-12};
  runtime::field::NamedFieldOutput<2> output(std::array<int, 3>{0, 1, 2}, 1);
  runtime::system::ExactNamedField<2> provider("electric", "plasma", output, geometry, layout,
                                               distribution, Index<2>{}, topology, options, 1);

  MultiFab<2> state(layout, distribution, Index<2>{}, 1, Extent<2>{});
  const Real pi = std::acos(Real{-1});
  const Real inverse_spacing = Real{1} / geometry.spacing(0);
  const Real eigenvalue = Real{2} * Real{4} * std::sin(pi / Real{cells}) *
                          std::sin(pi / Real{cells}) * inverse_spacing * inverse_spacing;
  fill_rhs(state, geometry, CartesianBoundaryKind::periodic, eigenvalue);
  provider.set_rhs(0, [](const MultiFab<2>& source, MultiFab<2>& rhs) {
    elliptic::nd::detail::copy_component(source, 0, rhs, 0);
  });
  MultiFab<2> live_aux(layout, distribution, Index<2>{}, 3, Extent<2>{});
  live_aux.set_val(Real{-7});

  const std::vector<const MultiFab<2>*> states{&state};
  SolveReport report = provider.solve_candidate(states, live_aux);
  ASSERT_TRUE(report.solved_value_available()) << report.reason;
  EXPECT_DOUBLE_EQ(reduce_max(live_aux, 0), Real{-7});
  provider.validate_candidate();
  provider.reject_candidate();
  EXPECT_DOUBLE_EQ(reduce_max(provider.accepted_potential(), 0), Real{0});

  report = provider.solve_candidate(states, live_aux);
  ASSERT_TRUE(report.solved_value_available()) << report.reason;
  provider.validate_candidate();
  provider.accept_candidate();
  EXPECT_LT(maximum_error(live_aux, geometry, CartesianBoundaryKind::periodic), Real{5e-11});
  EXPECT_GT(reduce_max(live_aux, 1), Real{-7});
  EXPECT_GT(reduce_max(live_aux, 2), Real{-7});
}

TEST(test_cartesian_poisson_nd, remote_halo_requirement_fails_before_candidate_mutation) {
  const Box<1> domain{Index<1>{0}, Index<1>{7}};
  const Geometry<1> geometry = Geometry<1>::from_bounds(domain, RealVector<1>{0}, RealVector<1>{1});
  const BoxArray<1> layout = BoxArray<1>::from_domain(domain, Extent<1>{4});
  const RankSpace<1> ranks{Index<1>{0}, Extent<1>{2}};
  const auto distribution =
      Distribution<1>::partitioned(layout, ranks, std::vector<Index<1>>{Index<1>{0}, Index<1>{1}});
  const BoundaryTopology<1> topology = BoundaryTopology<1>::axis_periodic({false});
  auto options = CartesianPoissonOptions<1>::from_topology(topology);
  CartesianPoissonSolver<1> solver(geometry, layout, distribution, Index<1>{0}, topology, options);
  MultiFab<1> warm_start(layout, distribution, Index<1>{0}, 1, Extent<1>{1});
  warm_start.set_val(Real{5});
  solver.candidate().set_val(Real{13});

  EXPECT_THROW((void)solver.solve(warm_start), std::logic_error);
  const auto& candidate = static_cast<const MultiFab<1>&>(solver.candidate());
  for (std::size_t local = 0; local < candidate.local_size(); ++local) {
    const auto& fab = candidate.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t element = 0; element < host.size(); ++element)
      EXPECT_DOUBLE_EQ(host(element), Real{13});
  }
}
