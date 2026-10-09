#include <gtest/gtest.h>

#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/boundary/halo_exchange.hpp>
#include <pops/numerics/elliptic/polar/prepared_polar_poisson_inverse.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

namespace {
using namespace pops;
using Field = MultiFab<2>;
using Inverse = elliptic::polar::PreparedPolarPoissonInverse<2>;
constexpr Real pi = std::numbers::pi_v<Real>;

PhysicalBoundaryConditions<2> disk_boundaries(const Geometry<2>& geometry) {
  std::array<PhysicalBoundaryFace, 4> faces{};
  faces[0] = {PhysicalBoundaryKind::neumann, Real(0)};
  faces[1] = {PhysicalBoundaryKind::dirichlet, Real(0)};
  return {BoundaryTopology<2>::axis_periodic(std::array<bool, 2>{false, true}), faces,
          RealVector<2>{geometry.spacing(0), geometry.spacing(1)}};
}

// The RHS and residual come from the existing actual CartesianTensorOperator and its halo/
// physical-boundary implementation, not from the inverse's separated radial formula.
struct DiskProblem {
  ExecutionLane lane = ExecutionLane::duplicate_world_collectively("tests.prepared-polar-inverse");
  Geometry<2> geometry;
  PhysicalBoundaryConditions<2> boundary;
  mesh::BoxArray<2> layout;
  mesh::Distribution<2> distribution;
  Field exact, rhs, solution, residual;
  std::array<std::unique_ptr<Field>, 4> coefficients;
  HaloSchedule<2> halo;
  PreparedPhysicalBoundary<2> physical;
  std::optional<HaloExchange<2>> exchange;
  Inverse inverse;

  static mesh::BoxArray<2> boxes(const Geometry<2>& geometry) {
    const auto domain = geometry.domain();
    auto first = domain;
    auto second = domain;
    first.hi[1] = first.lo[1] + static_cast<int>(domain.length(1) / 2) - 1;
    second.lo[1] = first.hi[1] + 1;
    return mesh::BoxArray<2>(std::vector<Box<2>>{first, second});
  }

  static mesh::Distribution<2> ownership(const mesh::BoxArray<2>& layout,
                                         const ExecutionLane& lane, bool replicated,
                                         bool empty_owner) {
    const mesh::RankSpace<2> space{Index<2>{0, 0}, Extent<2>{lane.size(), 1}};
    if (replicated)
      return mesh::Distribution<2>::replicated(layout, space);
    return mesh::Distribution<2>::partitioned(layout, space,
        {Index<2>{0, 0}, Index<2>{empty_owner ? 0 : lane.size() - 1, 0}});
  }

  static HaloScheduleBudget halo_budget(const Geometry<2>& geometry) {
    const auto elements = std::size_t{144} * static_cast<std::size_t>(geometry.domain().numPts());
    return {{2, 1}, 36, 144, 9, 4, elements, elements, elements};
  }

  DiskProblem(int nr = 7, int nt = 16, Real radius = Real(1.7),
              Index<2> origin = Index<2>{0, 0}, bool replicated = false,
              bool empty_owner = false)
      : geometry(Geometry<2>::from_bounds(
            Box<2>{origin, Index<2>{origin[0] + nr - 1, origin[1] + nt - 1}},
            RealVector<2>{0, 0}, RealVector<2>{radius, 2 * pi})),
        boundary(disk_boundaries(geometry)), layout(boxes(geometry)),
        distribution(ownership(layout, lane, replicated, empty_owner)),
        exact(layout, distribution, Index<2>{lane.rank(), 0}, 1, Extent<2>{1, 1}),
        rhs(layout, distribution, Index<2>{lane.rank(), 0}, 1, Extent<2>{}),
        solution(layout, distribution, Index<2>{lane.rank(), 0}, 1, Extent<2>{1, 1}),
        residual(layout, distribution, Index<2>{lane.rank(), 0}, 1, Extent<2>{}),
        halo(prepare_halo_schedule(exact, geometry.domain(), boundary.topology(),
                                  HaloLayoutCoverage::full_domain, halo_budget(geometry))),
        physical(prepare_physical_boundary(geometry.domain(), Extent<2>{1, 1}, boundary, {8})),
        inverse(solution, geometry, physical, {1u, 2u, true}, lane) {
    for (auto& coefficient : coefficients)
      coefficient = std::make_unique<Field>(layout, distribution, Index<2>{lane.rank(), 0},
                                            1, Extent<2>{1, 1});
    for (std::size_t local = 0; local < exact.local_size(); ++local) {
      std::array<FieldView<Real, 2>, 4> values{};
      for (std::size_t slot = 0; slot < 4; ++slot)
        values[slot] = coefficients[slot]->fab(local).view();
      const auto mesh = geometry;
      for_each_cell(exact.fab(local).grown_box(), [=] POPS_HD(const Index<2>& index) {
        const Real radius = mesh.cell_coordinate(0, index[0]);
        values[0](index, 0) = radius;
        values[1](index, 0) = values[2](index, 0) = Real(0);
        values[3](index, 0) = Real(1) / radius;
      });
    }
    Kokkos::fence();
    if (all_reduce_max(halo.has_remote_jobs() ? 1L : 0L, lane) != 0) {
      HaloExchangeContext context{};
      context.context_generation = context.schedule_generation = 1;
      exchange.emplace(halo, lane, context);
    }
  }

  void pattern(Field& target, int kind, int mode = 1) const {
    for (std::size_t local = 0; local < target.local_size(); ++local) {
      const auto values = target.fab(local).view();
      const auto mesh = geometry;
      for_each_cell(target.box(local), [=] POPS_HD(const Index<2>& index) {
        const Real r = mesh.cell_coordinate(0, index[0]) / mesh.upper()[0];
        const int j = index[1] - mesh.domain().lo[1];
        const int nt = static_cast<int>(mesh.domain().length(1));
        const Real theta = Real(2) * pi * static_cast<Real>(j) / static_cast<Real>(nt);
        const Real radial = Real(1) + r + Real(0.1) * r * r;
        Real value = radial;
        if (kind == 1)
          value *= Kokkos::cos(static_cast<Real>(mode) * theta);
        else if (kind == 2)
          value *= Kokkos::sin(static_cast<Real>(mode) * theta);
        else if (kind == 3)
          value *= j % 2 == 0 ? Real(1) : Real(-1);
        else if (kind == 4)
          value += (Real(0.2) + r) * Kokkos::cos(theta) +
                   (Real(0.1) + Real(0.3) * r) * Kokkos::sin(Real(3) * theta) +
                   Real(0.1) * r * Kokkos::cos(static_cast<Real>(nt / 2) * theta);
        values(index, 0) = value;
      });
    }
    Kokkos::fence();
  }

  void apply_stencil(Field& input, Field& output) {
    if (exchange)
      exchange->execute(input, lane);
    else
      fill_boundary(input, halo);
    fill_physical_boundary(input, physical);
    for (std::size_t local = 0; local < output.local_size(); ++local) {
      std::array<FieldView<const Real, 2>, 4> values{};
      for (std::size_t slot = 0; slot < 4; ++slot)
        values[slot] = std::as_const(coefficients[slot]->fab(local)).view();
      const auto stencil = elliptic::nd::make_cartesian_tensor_operator<
          elliptic::nd::CartesianTensorDivergenceSign::negative_divergence>(
          std::as_const(input.fab(local)).view(),
          elliptic::nd::split_cartesian_tensor_coefficients<2>(values), geometry, {1u, 2u, true});
      const auto target = output.fab(local).view();
      for_each_cell(output.box(local), [=] POPS_HD(const Index<2>& index) {
        target(index, 0) = stencil.image(index);
      });
    }
    Kokkos::fence();
  }

  Real difference(const Field& first, const Field& second) const {
    Real maximum = 0;
    for (std::size_t local = 0; local < first.local_size(); ++local) {
      const auto a = first.fab(local).view(), b = second.fab(local).view();
      maximum = std::max(maximum, for_each_cell_reduce_max(first.box(local),
          [=] POPS_HD(const Index<2>& index) { return Kokkos::abs(a(index, 0) - b(index, 0)); }));
    }
    return all_reduce_max(maximum, lane);
  }

  Real maximum(const Field& field) const {
    Real maximum = 0;
    for (std::size_t local = 0; local < field.local_size(); ++local) {
      const auto values = field.fab(local).view();
      maximum = std::max(maximum, for_each_cell_reduce_max(field.box(local),
          [=] POPS_HD(const Index<2>& index) { return Kokkos::abs(values(index, 0)); }));
    }
    return all_reduce_max(maximum, lane);
  }

  void roundtrip(int kind, int mode = 1) {
    pattern(exact, kind, mode);
    apply_stencil(exact, rhs);
    ASSERT_TRUE(inverse.apply(solution, rhs).succeeded());
    EXPECT_LT(difference(solution, exact), Real(3e-11));
    apply_stencil(solution, residual);
    EXPECT_LT(difference(residual, rhs) / std::max(Real(1), maximum(rhs)), Real(3e-12));
  }
};

TEST(PreparedPolarPoissonInverse, ActualStencilConstantCosineSineNyquistAndMixed) {
  DiskProblem problem;
  problem.inverse.prepare();
  problem.roundtrip(0);
  problem.roundtrip(1, 1);
  problem.roundtrip(2, 3);
  problem.roundtrip(3);
  problem.roundtrip(4);
}

TEST(PreparedPolarPoissonInverse, TranslatedOriginsOddGridNonbinarySpacingAndTwoRadialCells) {
  for (const int nr : {2, 9}) {
    DiskProblem problem(nr, 15, Real(2.3), Index<2>{-7, 19});
    problem.inverse.prepare();
    problem.roundtrip(0);
    problem.roundtrip(1, 7);
    problem.roundtrip(2, 7);
    problem.roundtrip(4);
  }
}

TEST(PreparedPolarPoissonInverse, DistributedEmptyOwnerAndExactReplicas) {
  for (const bool replicated : {false, true}) {
    DiskProblem problem(7, 16, Real(1.7), Index<2>{3, -11}, replicated, !replicated);
    if (!replicated && problem.lane.size() > 1 && problem.lane.rank() != 0)
      EXPECT_EQ(problem.rhs.local_size(), 0u);
    problem.inverse.prepare();
    problem.roundtrip(0);
    problem.roundtrip(3);
    problem.roundtrip(4);
  }
}

TEST(PreparedPolarPoissonInverse, FixedLinearityZeroResponseAndTwoPersistentSessions) {
  DiskProblem problem;
  Field a(problem.layout, problem.distribution, problem.rhs.local_rank(), 1, Extent<2>{});
  Field b(a), combined(a), image_a(a), image_b(a), image_combined(a), expected(a), saved_a(a);
  Inverse second(problem.solution, problem.geometry, problem.physical, {1u, 2u, true}, problem.lane);
  problem.inverse.prepare();
  second.prepare();
  EXPECT_EQ(problem.inverse.allocation_count(), 7u);
  EXPECT_EQ(second.allocation_count(), 7u);
  EXPECT_GT(problem.inverse.storage_bytes(), 0u);
  EXPECT_LE(problem.inverse.storage_bytes(), Inverse::maximum_storage_bytes);
  problem.pattern(a, 1, 1);
  problem.pattern(b, 2, 3);
  for (std::size_t local = 0; local < a.local_size(); ++local) {
    const auto va = std::as_const(a.fab(local)).view(), vb = std::as_const(b.fab(local)).view();
    const auto vc = combined.fab(local).view(), saved = saved_a.fab(local).view();
    for_each_cell(a.box(local), [=] POPS_HD(const Index<2>& index) {
      vc(index, 0) = Real(2) * va(index, 0) - Real(0.75) * vb(index, 0);
      saved(index, 0) = va(index, 0);
    });
  }
  Kokkos::fence();
  ASSERT_TRUE(problem.inverse.apply(image_a, a).succeeded());
  ASSERT_TRUE(second.apply(image_b, b).succeeded());
  ASSERT_TRUE(problem.inverse.apply(image_combined, combined).succeeded());
  for (std::size_t local = 0; local < a.local_size(); ++local) {
    const auto va = std::as_const(image_a.fab(local)).view(), vb = std::as_const(image_b.fab(local)).view();
    const auto target = expected.fab(local).view();
    for_each_cell(a.box(local), [=] POPS_HD(const Index<2>& index) {
      target(index, 0) = Real(2) * va(index, 0) - Real(0.75) * vb(index, 0);
    });
  }
  Kokkos::fence();
  EXPECT_LT(problem.difference(image_combined, expected), Real(2e-13));
  EXPECT_EQ(problem.difference(a, saved_a), Real(0));
  const auto allocations = allocation_event_stats();
  for (int repeat = 0; repeat < 4; ++repeat) {
    problem.inverse.prepare();
    second.prepare();
    ASSERT_TRUE(second.apply(image_combined, combined).succeeded());
    ASSERT_TRUE(problem.inverse.apply(image_a, a).succeeded());
  }
  EXPECT_EQ(allocation_event_stats(), allocations);
  EXPECT_EQ(problem.inverse.allocation_count(), 7u);
  EXPECT_LT(problem.difference(image_combined, expected), Real(2e-13));
  combined.set_val(Real(0));
  ASSERT_TRUE(problem.inverse.apply(image_combined, combined).succeeded());
  EXPECT_EQ(problem.maximum(image_combined), Real(0));
}

TEST(PreparedPolarPoissonInverse, UnsupportedBoundaryAndAsymmetricOptionsRefuseCollectively) {
  DiskProblem problem;
  auto bad_options = elliptic::nd::CartesianTensorStencilOptions{1u, 2u, true};
  if (problem.lane.rank() == 0)
    bad_options.arithmetic_diagonal = false;
  Inverse asymmetric(problem.solution, problem.geometry, problem.physical, bad_options, problem.lane);
  EXPECT_THROW(asymmetric.prepare(), std::invalid_argument);
  for (const bool nonzero : {false, true}) {
    std::array<PhysicalBoundaryFace, 4> faces{};
    faces[0] = {PhysicalBoundaryKind::neumann, Real(0)};
    faces[1] = nonzero ? PhysicalBoundaryFace{PhysicalBoundaryKind::dirichlet, Real(1)} :
                        PhysicalBoundaryFace{PhysicalBoundaryKind::neumann, Real(0)};
    const PhysicalBoundaryConditions<2> laws{problem.boundary.topology(), faces,
        RealVector<2>{problem.geometry.spacing(0), problem.geometry.spacing(1)}};
    const auto boundary = prepare_physical_boundary(problem.geometry.domain(), Extent<2>{1, 1}, laws, {8});
    Inverse invalid(problem.solution, problem.geometry, boundary, {1u, 2u, true}, problem.lane);
    EXPECT_THROW(invalid.prepare(), std::invalid_argument);
  }
  problem.inverse.prepare();
  problem.roundtrip(4);
}

TEST(PreparedPolarPoissonInverse, LocallyValidDifferentGeometryFailsExactCollectiveContract) {
  DiskProblem problem;
  if (problem.lane.size() < 2)
    GTEST_SKIP() << "requires two ranks for distinct locally valid geometries";
  const auto geometry = Geometry<2>::from_bounds(problem.geometry.domain(), RealVector<2>{0, 0},
      RealVector<2>{problem.lane.rank() == 0 ? Real(1.7) : Real(2.3), 2 * pi});
  const auto boundary = prepare_physical_boundary(geometry.domain(), Extent<2>{1, 1},
                                                  disk_boundaries(geometry), {8});
  Inverse inconsistent(problem.solution, geometry, boundary, {1u, 2u, true}, problem.lane);
  EXPECT_THROW(inconsistent.prepare(), std::invalid_argument);
  problem.inverse.prepare();
  problem.roundtrip(0);
}

TEST(PreparedPolarPoissonInverse, MissingCoverageOverlapAndWrongRankSpaceRefuse) {
  DiskProblem problem;
  for (const bool overlap : {false, true}) {
    auto first = problem.layout[0];
    auto second = problem.layout[1];
    second.lo[1] += overlap ? -1 : 1;
    const mesh::BoxArray<2> layout(std::vector<Box<2>>{first, second});
    const auto distribution = DiskProblem::ownership(layout, problem.lane, false, false);
    Field prototype(layout, distribution, problem.rhs.local_rank(), 1, Extent<2>{});
    Inverse invalid(prototype, problem.geometry, problem.physical, {1u, 2u, true}, problem.lane);
    EXPECT_THROW(invalid.prepare(), std::invalid_argument);
  }
  const mesh::RankSpace<2> space{Index<2>{0, 0}, Extent<2>{problem.lane.size() + 1, 1}};
  const auto distribution = mesh::Distribution<2>::replicated(problem.layout, space);
  Field prototype(problem.layout, distribution, problem.rhs.local_rank(), 1, Extent<2>{});
  Inverse invalid(prototype, problem.geometry, problem.physical, {1u, 2u, true}, problem.lane);
  EXPECT_THROW(invalid.prepare(), std::invalid_argument);
  problem.inverse.prepare();
  problem.roundtrip(1, 2);
}

TEST(PreparedPolarPoissonInverse, BoundsAndNonDiskGeometryRefuseBeforeWorkspaceAllocation) {
  DiskProblem problem;
  const mesh::BoxArray<2> empty_layout;
  const auto empty_distribution = mesh::Distribution<2>::replicated(empty_layout,
                                                                   problem.rhs.rank_space());
  Field empty(empty_layout, empty_distribution, problem.rhs.local_rank(), 1, Extent<2>{});
  for (const auto extent : {Index<2>{1, 16}, Index<2>{2, 1}, Index<2>{2, 1025},
                            Index<2>{4097, 2}, Index<2>{129, 1024}, Index<2>{263, 1024}}) {
    const auto geometry = Geometry<2>::from_bounds(
        Box<2>{Index<2>{0, 0}, Index<2>{extent[0] - 1, extent[1] - 1}},
        RealVector<2>{0, 0}, RealVector<2>{Real(1.7), 2 * pi});
    const auto boundary = prepare_physical_boundary(geometry.domain(), Extent<2>{},
                                                    disk_boundaries(geometry), {8});
    Inverse invalid(empty, geometry, boundary, {1u, 2u, true}, problem.lane);
    const auto allocations = allocation_event_stats();
    EXPECT_THROW(invalid.prepare(), std::invalid_argument);
    EXPECT_EQ(allocation_event_stats(), allocations);
    EXPECT_EQ(invalid.allocation_count(), 0u);
  }
  for (const auto bounds : {RealVector<2>{Real(0.1), 2 * pi}, RealVector<2>{Real(0), pi}}) {
    const auto geometry = Geometry<2>::from_bounds(problem.geometry.domain(),
        RealVector<2>{bounds[0], 0}, RealVector<2>{Real(1.7), bounds[1]});
    const auto boundary = prepare_physical_boundary(geometry.domain(), Extent<2>{1, 1},
                                                    disk_boundaries(geometry), {8});
    Inverse invalid(problem.solution, geometry, boundary, {1u, 2u, true}, problem.lane);
    EXPECT_THROW(invalid.prepare(), std::invalid_argument);
  }
}

TEST(PreparedPolarPoissonInverse, UnpreparedNonfiniteAndAsymmetricApplyLayoutFailClosed) {
  DiskProblem problem;
  EXPECT_FALSE(problem.inverse.apply(problem.solution, problem.rhs).succeeded());
  problem.inverse.prepare();
  problem.rhs.set_val(problem.lane.rank() == 0 ? std::numeric_limits<Real>::quiet_NaN() : Real(0));
  EXPECT_FALSE(problem.inverse.apply(problem.solution, problem.rhs).succeeded());
  Field wrong(problem.layout, problem.distribution, problem.rhs.local_rank(),
              problem.lane.rank() == 0 ? 2 : 1, Extent<2>{});
  EXPECT_FALSE(problem.inverse.apply(problem.solution, wrong).succeeded());
  problem.roundtrip(4);
}

TEST(PreparedPolarPoissonInverse, OneUlpReplicaDifferenceRefusesWithoutToleranceOrAveraging) {
  DiskProblem problem(7, 16, Real(1.7), Index<2>{0, 0}, true);
  if (problem.lane.size() < 2)
    GTEST_SKIP() << "requires two physically replicated rank copies";
  problem.inverse.prepare();
  problem.rhs.set_val(problem.lane.rank() == 0 ? Real(1) :
                       std::nextafter(Real(1), std::numeric_limits<Real>::infinity()));
  const auto failed = problem.inverse.apply(problem.solution, problem.rhs);
  EXPECT_FALSE(failed.succeeded());
  EXPECT_EQ(failed.action, PreparedApplyFailureAction::kFailRun);
  EXPECT_EQ(failed.phase(), "polar-inverse:replica-or-reduction");
  problem.roundtrip(4);
}

TEST(PreparedPolarPoissonInverse, OtherNativeDimensionsCompileAndRefuseCollectively) {
  const auto lane = ExecutionLane::duplicate_world_collectively("tests.polar-inverse.other-dimensions");
  auto refuse = [&]<int Dim>() {
    using OtherField = MultiFab<Dim>;
    using OtherInverse = elliptic::polar::PreparedPolarPoissonInverse<Dim>;
    Index<Dim> origin{}, high{}, local{};
    Extent<Dim> ranks{};
    RealVector<Dim> lower{}, upper{}, spacing{};
    for (int axis = 0; axis < Dim; ++axis) {
      high[axis] = 1;
      ranks[axis] = 1;
      upper[axis] = Real(1);
      spacing[axis] = Real(0.5);
    }
    ranks[0] = lane.size();
    local[0] = lane.rank();
    const Box<Dim> domain{origin, high};
    const auto geometry = Geometry<Dim>::from_bounds(domain, lower, upper);
    const mesh::BoxArray<Dim> layout(std::vector<Box<Dim>>{domain});
    const auto distribution = mesh::Distribution<Dim>::replicated(layout,
        mesh::RankSpace<Dim>{origin, ranks});
    OtherField prototype(layout, distribution, local, 1, Extent<Dim>{});
    std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
    const PhysicalBoundaryConditions<Dim> laws{
        BoundaryTopology<Dim>::axis_periodic(std::array<bool, Dim>{}), faces, spacing};
    const auto physical = prepare_physical_boundary(domain, Extent<Dim>{}, laws, {26});
    OtherInverse inverse(prototype, geometry, physical, {1u, 2u, true}, lane);
    const auto allocations = allocation_event_stats();
    EXPECT_THROW(inverse.prepare(), std::invalid_argument);
    EXPECT_EQ(inverse.allocation_count(), 0u);
    EXPECT_EQ(allocation_event_stats(), allocations);
  };
  refuse.template operator()<1>();
  refuse.template operator()<3>();
}

static_assert(PreparedLinearPreconditionerSessionSource<2, Inverse>);
}  // namespace
