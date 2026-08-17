#include <gtest/gtest.h>

#include <pops/numerics/elliptic/nd/cartesian_poisson.hpp>
#include <pops/runtime/system/exact_named_field.hpp>
#include <pops/runtime/system/exact_field_marshaling.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
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
void install_mode_nullspace(CartesianPoissonSolver<Dim>& solver, const Geometry<Dim>& geometry,
                            CartesianBoundaryKind boundary) {
  if (boundary == CartesianBoundaryKind::dirichlet) {
    solver.install_nullspace(FieldNullspacePlan<Dim>{},
                             PreparedVectorDistribution<Dim>::replicated());
    return;
  }
  Real cell_measure = Real(1);
  for (int axis = 0; axis < Dim; ++axis)
    cell_measure *= geometry.spacing(axis);
  solver.install_nullspace(
      constant_mean_zero_nullspace<Dim>("manufactured-constant-mode", "unit-test", cell_measure),
      PreparedVectorDistribution<Dim>::replicated());
}

template <int Dim>
void expect_manufactured_mode(CartesianBoundaryKind boundary) {
  const ExecutionLane lane = ExecutionLane::world("tests.cartesian-poisson.manufactured");
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
                                     options, lane);
  install_mode_nullspace(solver, geometry, boundary);
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

  const SolveReport report = solver.solve(warm_start, lane);
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

template <int Dim>
void expect_exact_marshaling_round_trip() {
  Index<Dim> upper{};
  Extent<Dim> patch{};
  for (int axis = 0; axis < Dim; ++axis) {
    upper[axis] = 2 + axis;
    patch[axis] = 2;
  }
  const Box<Dim> domain{Index<Dim>{}, upper};
  const BoxArray<Dim> layout = BoxArray<Dim>::from_domain(domain, patch);
  const RankSpace<Dim> ranks{Index<Dim>{}, uniform_extent<Dim>(1)};
  const auto distribution = Distribution<Dim>::replicated(layout, ranks);
  MultiFab<Dim> field(layout, distribution, Index<Dim>{}, 2, uniform_extent<Dim>(1));
  field.set_val(Real{-9});

  const std::size_t cells = runtime::system::marshaling::checked_cell_count(domain);
  std::vector<double> payload(2 * cells);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    payload[cell] = 0.25 + static_cast<double>(cell);
    payload[cells + cell] = -3.5 - static_cast<double>(cell);
  }
  runtime::system::marshaling::write_global(field, domain, payload, 2);
  EXPECT_EQ(runtime::system::marshaling::gather_global(field, domain, 2), payload);

  const std::vector<double> before = runtime::system::marshaling::gather_global(field, domain, 2);
  EXPECT_THROW(runtime::system::marshaling::write_global(
                   field, domain, std::vector<double>(payload.size() - 1), 2),
               std::invalid_argument);
  EXPECT_EQ(runtime::system::marshaling::gather_global(field, domain, 2), before);
}

template <int Dim>
struct QuadraticResidualKernel {
  FieldView<const Real, Dim> iterate{};
  FieldView<Real, Dim> residual{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    const Real value = iterate(index, 0);
    residual(index, 0) = Real(4) - value * value;
  }
};

template <int Dim>
struct QuadraticJvpKernel {
  FieldView<const Real, Dim> iterate{};
  FieldView<const Real, Dim> direction{};
  FieldView<Real, Dim> output{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    output(index, 0) = Real(2) * iterate(index, 0) * direction(index, 0);
  }
};

template <int Dim>
void expect_exact_ranked_newton() {
  const ExecutionLane lane = ExecutionLane::world("tests.cartesian-poisson.newton");
  Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = 3;
  const Box<Dim> domain{Index<Dim>{}, upper};
  const BoxArray<Dim> layout = BoxArray<Dim>::from_domain(domain, uniform_extent<Dim>(4));
  const RankSpace<Dim> ranks{Index<Dim>{}, uniform_extent<Dim>(1)};
  const auto distribution = Distribution<Dim>::replicated(layout, ranks);
  FieldNewtonOptions options;
  options.tolerance = Real(1e-12);
  options.max_iterations = 12;
  options.linear_tolerance = Real(1e-12);
  options.linear_max_iterations = 8;
  options.restart = 4;
  FieldNewtonKrylovWorkspace<Dim> workspace(layout, distribution, Index<Dim>{}, options);
  MultiFab<Dim> iterate(layout, distribution, Index<Dim>{}, 1, Extent<Dim>{});
  iterate.set_val(Real(1));

  const SolveReport report = workspace.solve(
      iterate,
      [](const MultiFab<Dim>& value, MultiFab<Dim>& residual, int) {
        for (std::size_t local = 0; local < value.local_size(); ++local)
          for_each_cell(value.box(local), QuadraticResidualKernel<Dim>{value.fab(local).view(),
                                                                       residual.fab(local).view()});
      },
      [](const MultiFab<Dim>& value, const MultiFab<Dim>& direction, MultiFab<Dim>& output, int) {
        for (std::size_t local = 0; local < value.local_size(); ++local)
          for_each_cell(value.box(local), QuadraticJvpKernel<Dim>{value.fab(local).view(),
                                                                  direction.fab(local).view(),
                                                                  output.fab(local).view()});
      },
      [](MultiFab<Dim>&) {}, lane);

  ASSERT_TRUE(report.solved_value_available()) << report.reason;
  EXPECT_LT(report.residual_norm, Real(1e-10));
  EXPECT_LT(reduce_norm_inf(iterate) - Real(2), Real(1e-10));
  EXPECT_GT(reduce_min(iterate), Real(2) - Real(1e-10));
}

}  // namespace

TEST(test_cartesian_poisson_nd, manufactured_modes_solve_in_exact_rank_one_two_and_three) {
  static_assert(CartesianPoissonOptions<1>::dimension == 1);
  static_assert(CartesianPoissonOptions<2>::dimension == 2);
  static_assert(CartesianPoissonOptions<3>::dimension == 3);
  static_assert(CartesianPoissonSolver<1>::dimension == 1);
  static_assert(CartesianPoissonSolver<3>::dimension == 3);
  expect_all_boundaries<1>();
  expect_all_boundaries<2>();
  expect_all_boundaries<3>();
}

TEST(test_cartesian_poisson_nd, damped_newton_gmres_executes_one_algorithm_in_all_ranks) {
  expect_exact_ranked_newton<1>();
  expect_exact_ranked_newton<2>();
  expect_exact_ranked_newton<3>();
}

TEST(test_cartesian_poisson_nd, named_provider_publishes_only_after_candidate_acceptance) {
  const ExecutionLane lane = ExecutionLane::world("tests.cartesian-poisson.named-field");
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
  runtime::field::NamedFieldOutput<2> output(3, 1);
  runtime::system::ExactNamedField<2> provider("electric", "plasma", output, geometry, layout,
                                               distribution, Index<2>{}, topology, options, 1,
                                               lane);
  PreparedFieldNullspace<2> prepared_nullspace;
  prepared_nullspace.provider_identity = "tests.constant-nullspace";
  prepared_nullspace.provider_version = 1;
  prepared_nullspace.exact_prepared_contract = "tests.constant-nullspace@1";
  prepared_nullspace.plan = constant_mean_zero_nullspace<2>(
      "electric-constant-mode", "unit-test", geometry.spacing(0) * geometry.spacing(1));
  provider.install_nullspace(std::move(prepared_nullspace),
                             PreparedVectorDistribution<2>::replicated());

  MultiFab<2> state(layout, distribution, Index<2>{}, 1, Extent<2>{});
  const Real pi = std::acos(Real{-1});
  const Real inverse_spacing = Real{1} / geometry.spacing(0);
  const Real eigenvalue = Real{2} * Real{4} * std::sin(pi / Real{cells}) *
                          std::sin(pi / Real{cells}) * inverse_spacing * inverse_spacing;
  fill_rhs(state, geometry, CartesianBoundaryKind::periodic, eigenvalue);
  provider.add_rhs(
      0,
      [](const MultiFab<2>& source, MultiFab<2>& rhs) {
        elliptic::nd::detail::copy_component(source, 0, rhs, 0);
      },
      Real(1));
  const std::vector<const MultiFab<2>*> states{&state};
  SolveReport report = provider.solve_candidate(states, nullptr, lane);
  ASSERT_TRUE(report.solved_value_available()) << report.reason;
  EXPECT_DOUBLE_EQ(reduce_max(provider.accepted_outputs(), 0), Real{0});
  provider.validate_candidate();
  provider.reject_candidate();
  EXPECT_DOUBLE_EQ(reduce_max(provider.accepted_potential(), 0), Real{0});

  report = provider.solve_candidate(states, nullptr, lane);
  ASSERT_TRUE(report.solved_value_available()) << report.reason;
  provider.validate_candidate();
  provider.accept_candidate();
  EXPECT_LT(maximum_error(provider.accepted_outputs(), geometry, CartesianBoundaryKind::periodic),
            Real{5e-11});
  EXPECT_GT(reduce_max(provider.accepted_outputs(), 1), Real{0});
  EXPECT_GT(reduce_max(provider.accepted_outputs(), 2), Real{0});
}

TEST(test_cartesian_poisson_nd, remote_halo_requirement_fails_before_candidate_mutation) {
  const ExecutionLane lane = ExecutionLane::world("tests.cartesian-poisson.remote-halo");
  const Box<1> domain{Index<1>{0}, Index<1>{7}};
  const Geometry<1> geometry = Geometry<1>::from_bounds(domain, RealVector<1>{0}, RealVector<1>{1});
  const BoxArray<1> layout = BoxArray<1>::from_domain(domain, Extent<1>{4});
  const RankSpace<1> ranks{Index<1>{0}, Extent<1>{2}};
  const auto distribution =
      Distribution<1>::partitioned(layout, ranks, std::vector<Index<1>>{Index<1>{0}, Index<1>{1}});
  const BoundaryTopology<1> topology = BoundaryTopology<1>::axis_periodic({false});
  auto options = CartesianPoissonOptions<1>::from_topology(topology);
  MultiFab<1> warm_start(layout, distribution, Index<1>{0}, 1, Extent<1>{1});
  warm_start.set_val(Real{5});
  EXPECT_THROW((void)CartesianPoissonSolver<1>(geometry, layout, distribution, Index<1>{0},
                                               topology, options, lane),
               std::logic_error);
  for (std::size_t local = 0; local < warm_start.local_size(); ++local) {
    const auto& fab = static_cast<const MultiFab<1>&>(warm_start).fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t element = 0; element < host.size(); ++element)
      EXPECT_DOUBLE_EQ(host(element), Real{5});
  }
}

TEST(test_cartesian_poisson_nd, exact_field_marshaling_round_trips_rank_one_two_and_three) {
  expect_exact_marshaling_round_trip<1>();
  expect_exact_marshaling_round_trip<2>();
  expect_exact_marshaling_round_trip<3>();
}

TEST(test_cartesian_poisson_nd, incomplete_distribution_fails_before_resident_mutation) {
  const Box<1> domain{Index<1>{0}, Index<1>{7}};
  const BoxArray<1> layout = BoxArray<1>::from_domain(domain, Extent<1>{4});
  const RankSpace<1> ranks{Index<1>{0}, Extent<1>{2}};
  const auto distribution =
      Distribution<1>::partitioned(layout, ranks, std::vector<Index<1>>{Index<1>{0}, Index<1>{1}});
  MultiFab<1> field(layout, distribution, Index<1>{0}, 1, Extent<1>{1});
  field.set_val(Real{17});

  EXPECT_THROW(
      runtime::system::marshaling::write_global(field, domain, std::vector<double>(8, 2.0), 1),
      std::runtime_error);
  const auto& fab = static_cast<const MultiFab<1>&>(field).fab(0);
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  for (std::size_t element = 0; element < host.size(); ++element)
    EXPECT_DOUBLE_EQ(host(element), Real{17});
}

TEST(test_cartesian_poisson_nd, overlapping_equal_count_layout_fails_before_resident_mutation) {
  const Box<1> domain{Index<1>{0}, Index<1>{7}};
  const BoxArray<1> layout{
      std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{3}}, Box<1>{Index<1>{2}, Index<1>{5}}}};
  const RankSpace<1> ranks{Index<1>{0}, Extent<1>{1}};
  const auto distribution = Distribution<1>::replicated(layout, ranks);
  MultiFab<1> field(layout, distribution, Index<1>{0}, 1, Extent<1>{1});
  field.set_val(Real{29});

  EXPECT_THROW(
      runtime::system::marshaling::write_global(field, domain, std::vector<double>(8, 2.0), 1),
      std::runtime_error);
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const auto& fab = static_cast<const MultiFab<1>&>(field).fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t element = 0; element < host.size(); ++element)
      EXPECT_DOUBLE_EQ(host(element), Real{29});
  }
}
