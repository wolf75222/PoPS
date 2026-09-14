#include <gtest/gtest.h>

#include "mapped_disk_fac_witness.hpp"
#include <pops/core/foundation/allocator.hpp>

namespace {
using namespace pops;
using namespace pops::runtime::program::tensor_fac;
using Field = MultiFab<2>;
constexpr Real pi = Real(3.141592653589793238462643383279502884L);

// An independently filled exact vector is acted on by the existing FV tensor stencil and
// physical/periodic boundary route. Inverting that RHS exercises the new operator session at
// both walls, between patches, and across the angular seam without changing the reference action.
struct RootProblem {
  ExecutionLane lane = ExecutionLane::world("tests.tensor-coarse-gmres");
  Geometry<2> geometry;
  PhysicalBoundaryConditions<2> boundary;
  mesh::BoxArray<2> layout;
  mesh::Distribution<2> distribution;
  std::array<std::unique_ptr<Field>, 4> coefficients;
  Field exact, rhs, correction;
  HaloSchedule<2> halo;
  PreparedPhysicalBoundary<2> physical, coefficient_boundary;
  std::optional<HaloExchange<2>> exchange;
  std::unique_ptr<TensorCoarseGmres<2>> gmres;
  elliptic::nd::CartesianTensorStencilOptions options{1u, 2u, true};

  static PhysicalBoundaryConditions<2> boundaries(const Geometry<2>& geometry) {
    std::array<PhysicalBoundaryFace, 4> faces{};
    faces[0] = {PhysicalBoundaryKind::neumann, Real(0)};
    faces[1] = {PhysicalBoundaryKind::dirichlet, Real(0)};
    return {BoundaryTopology<2>::axis_periodic(std::array<bool, 2>{false, true}), faces,
            RealVector<2>{geometry.spacing(0), geometry.spacing(1)}};
  }

  static mesh::Distribution<2> ownership(const mesh::BoxArray<2>& boxes, int ranks,
                                         bool replicated, bool empty_owner) {
    const mesh::RankSpace<2> space{Index<2>{0, 0}, Extent<2>{ranks, 1}};
    if (replicated)
      return mesh::Distribution<2>::replicated(boxes, space);
    return mesh::Distribution<2>::partitioned(
        boxes, space, {Index<2>{0, 0}, Index<2>{empty_owner ? 0 : ranks - 1, 0}});
  }

  RootProblem(bool replicated = false, bool empty_owner = false)
      : geometry(Geometry<2>::from_bounds({Index<2>{0, 0}, Index<2>{7, 15}},
                   RealVector<2>{0, 0}, RealVector<2>{1, 2 * pi})),
        boundary(boundaries(geometry)),
        layout(std::vector<Box<2>>{{Index<2>{0, 0}, Index<2>{7, 7}},
                                   {Index<2>{0, 8}, Index<2>{7, 15}}}),
        distribution(ownership(layout, lane.size(), replicated, empty_owner)),
        exact(layout, distribution, Index<2>{lane.rank(), 0}, 1, Extent<2>{1, 1}),
        rhs(layout, distribution, Index<2>{lane.rank(), 0}, 1, Extent<2>{}),
        correction(layout, distribution, Index<2>{lane.rank(), 0}, 1, Extent<2>{1, 1}),
        halo(prepare_halo_schedule(exact, geometry.domain(), boundary.topology(),
              HaloLayoutCoverage::full_domain,
              runtime::program::tensor_fac::detail::exact_halo_budget<2>(layout,
                                                                         geometry.domain()))),
        physical(prepare_physical_boundary(geometry.domain(), Extent<2>{1, 1}, boundary,
                    runtime::program::tensor_fac::detail::exact_boundary_budget<2>())),
        coefficient_boundary(prepare_physical_boundary(
            geometry.domain(), Extent<2>{1, 1},
            runtime::program::tensor_fac::detail::boundary_with_values<2>(boundary, geometry,
                                                                          false, true),
            runtime::program::tensor_fac::detail::exact_boundary_budget<2>())) {
    for (auto& field : coefficients)
      field = std::make_unique<Field>(layout, distribution, Index<2>{lane.rank(), 0},
                                      1, Extent<2>{1, 1});
    if (all_reduce_max(halo.has_remote_jobs() ? 1L : 0L, lane) != 0) {
      HaloExchangeContext context{};
      context.context_generation = context.schedule_generation = 1;
      exchange.emplace(halo, lane, context);
    }
    gmres = std::make_unique<TensorCoarseGmres<2>>(
        correction, geometry, halo, physical, options, lane,
        "tests.tensor-coarse-gmres.exact-mixed-boundary-operator", 64);
  }

  void fill(Field& field, const PreparedPhysicalBoundary<2>& physical_boundary) {
    if (exchange)
      exchange->execute(field, lane);
    else
      fill_boundary(field, halo);
    fill_physical_boundary(field, physical_boundary);
  }

  void stage(Real amplitude) {
    for (std::size_t local = 0; local < exact.local_size(); ++local) {
      std::array<FieldView<Real, 2>, 4> values{};
      for (std::size_t slot = 0; slot < 4; ++slot)
        values[slot] = coefficients[slot]->fab(local).view();
      const auto target = exact.fab(local).view();
      const auto mesh = geometry;
      for_each_cell(exact.box(local), [=] POPS_HD(const Index<2>& index) {
        const Real r = mesh.cell_coordinate(0, index[0]);
        const Real theta = mesh.cell_coordinate(1, index[1]);
        const Real density = Real(1) + amplitude * r * r * Kokkos::sin(theta);
        values[0](index, 0) = r * (Real(1) + Real(0.02) * density);
        values[1](index, 0) = Real(0.03) * density;
        values[2](index, 0) = -Real(0.03) * density;
        values[3](index, 0) = (Real(1) + Real(0.02) * density) / r;
        target(index, 0) = r * r * r * (Real(1) - r * r) * Kokkos::cos(Real(3) * theta) +
                           amplitude * r * r * (Real(1) - r) * Kokkos::sin(theta);
      });
    }
    Kokkos::fence();
    for (auto& coefficient : coefficients)
      fill(*coefficient, coefficient_boundary);
    fill(exact, physical);
    for (std::size_t local = 0; local < rhs.local_size(); ++local) {
      std::array<FieldView<const Real, 2>, 4> values{};
      for (std::size_t slot = 0; slot < 4; ++slot)
        values[slot] = std::as_const(coefficients[slot]->fab(local)).view();
      const auto stencil = elliptic::nd::make_cartesian_tensor_operator<
          elliptic::nd::CartesianTensorDivergenceSign::negative_divergence>(
          std::as_const(exact.fab(local)).view(),
          elliptic::nd::split_cartesian_tensor_coefficients<2>(values), geometry, options);
      const auto output = rhs.fab(local).view();
      for_each_cell(rhs.box(local), [=] POPS_HD(const Index<2>& index) {
        output(index, 0) = stencil.image(index);
      });
    }
    Kokkos::fence();
    gmres->prepare_coefficients({coefficients[0].get(), coefficients[1].get(),
                                  coefficients[2].get(), coefficients[3].get()});
  }

  Real error() const {
    Real local_error = 0;
    for (std::size_t local = 0; local < correction.local_size(); ++local) {
      const auto value = correction.fab(local).view(), expected = exact.fab(local).view();
      local_error = std::max(local_error, for_each_cell_reduce_max(
          correction.box(local), [=] POPS_HD(const Index<2>& index) {
            return Kokkos::abs(value(index, 0) - expected(index, 0));
          }));
    }
    return static_cast<Real>(all_reduce_max(static_cast<double>(local_error), lane));
  }
};

TEST(TensorCoarseGMRES, ExactMixedBoundaryOperatorAndPersistentSecondSnapshot) {
  RootProblem problem;
  problem.stage(Real(0.1));
  auto report = problem.gmres->solve(problem.correction, problem.rhs, Real(1e-10), 512);
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_LT(problem.error(), Real(2e-10));
  const auto allocations = allocation_event_stats();
  problem.stage(Real(0.2));
  report = problem.gmres->solve(problem.correction, problem.rhs, Real(1e-10), 512);
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_LT(problem.error(), Real(2e-10));
  EXPECT_EQ(allocation_event_stats(), allocations);
}

TEST(TensorCoarseGMRES, EmptyOwnerAndReplicatedRootRemainExact) {
  for (const bool replicated : {false, true}) {
    RootProblem problem(replicated, !replicated);
    problem.stage(Real(0.1));
    const auto report = problem.gmres->solve(problem.correction, problem.rhs, Real(1e-10), 512);
    ASSERT_TRUE(report.solved()) << report.reason;
    EXPECT_LT(problem.error(), Real(2e-10));
  }
}

TEST(TensorCoarseGMRES, IterationLimitAndInvalidDiagonalFailClosed) {
  RootProblem problem;
  problem.stage(Real(0.1));
  auto report = problem.gmres->solve(problem.correction, problem.rhs, Real(1e-12), 1);
  EXPECT_FALSE(report.solved());
  EXPECT_EQ(report.status, SolveStatus::kIterationLimit);
  EXPECT_EQ(report.action, SolveAction::kFailRun);
  for (auto& coefficient : problem.coefficients)
    coefficient->set_val(Real(0));
  EXPECT_THROW(problem.gmres->prepare_coefficients(
      {problem.coefficients[0].get(), problem.coefficients[1].get(),
       problem.coefficients[2].get(), problem.coefficients[3].get()}), std::invalid_argument);
  EXPECT_THROW(problem.gmres->solve(problem.correction, problem.rhs, Real(1e-10), 512),
               std::invalid_argument);
}

TEST(TensorCoarseGMRES, MethodControlsAreExplicitAndBounded) {
  Controls controls;
  controls.coarse_method = static_cast<CoarseCorrectionMethod>(19);
  EXPECT_THROW(runtime::program::tensor_fac::detail::validate_controls(controls),
               std::invalid_argument);
  controls.coarse_method = CoarseCorrectionMethod::gauss_seidel;
  controls.coarse_restart = 32;
  EXPECT_THROW(runtime::program::tensor_fac::detail::validate_controls(controls),
               std::invalid_argument);
  controls.coarse_method = CoarseCorrectionMethod::gmres;
  controls.coarse_restart = 0;
  EXPECT_THROW(runtime::program::tensor_fac::detail::validate_controls(controls),
               std::invalid_argument);
}

TEST(TensorCoarseGMRES, SingularAndChangedPreparedMethodAreRefused) {
  RootProblem root;
  const auto fine_geometry = root.geometry.refine(Extent<2>{2, 2});
  const mesh::BoxArray<2> fine_layout(std::vector<Box<2>>{fine_geometry.domain()});
  const mesh::RankSpace<2> ranks{Index<2>{0, 0}, Extent<2>{root.lane.size(), 1}};
  const auto fine_distribution = mesh::Distribution<2>::partitioned(
      fine_layout, ranks, std::vector<Index<2>>{Index<2>{0, 0}});
  std::array<std::unique_ptr<Field>, 4> fine_coefficients;
  for (auto& coefficient : fine_coefficients)
    coefficient = std::make_unique<Field>(fine_layout, fine_distribution,
        Index<2>{root.lane.rank(), 0}, 1, Extent<2>{1, 1});
  Field fine_rhs(fine_layout, fine_distribution, Index<2>{root.lane.rank(), 0}, 1, Extent<2>{});
  Field fine_initial(fine_layout, fine_distribution, Index<2>{root.lane.rank(), 0}, 1,
                     Extent<2>{1, 1});
  Field fine_solution(fine_layout, fine_distribution, Index<2>{root.lane.rank(), 0}, 1,
                      Extent<2>{1, 1});
  std::array<PhysicalBoundaryConditions<2>, 2> boundaries{
      root.boundary, RootProblem::boundaries(fine_geometry)};
  const std::array<LevelBinding<2, Field::memory_space>, 2> bindings{{
      {&root.geometry, &boundaries[0],
       {root.coefficients[0].get(), root.coefficients[1].get(),
        root.coefficients[2].get(), root.coefficients[3].get()},
       &root.rhs, &root.exact, &root.correction},
      {&fine_geometry, &boundaries[1],
       {fine_coefficients[0].get(), fine_coefficients[1].get(),
        fine_coefficients[2].get(), fine_coefficients[3].get()},
       &fine_rhs, &fine_initial, &fine_solution}}};
  const std::array<amr::RefinementRatio<2>, 1> ratios{
      amr::RefinementRatio<2>{std::array<int, 2>{2, 2}}};
  {
    FullTensorCompositeFac<2> solver(bindings, ratios, root.lane, root.options,
                                     CoarseCorrectionMethod::gmres);
    Controls wrong_method;
    EXPECT_THROW(solver.solve(wrong_method, root.lane), std::invalid_argument);
    Controls asymmetric_method;
    asymmetric_method.coarse_method = root.lane.rank() == 0
        ? CoarseCorrectionMethod::gauss_seidel : CoarseCorrectionMethod::gmres;
    EXPECT_THROW(solver.solve(asymmetric_method, root.lane), std::invalid_argument);
    Controls asymmetric_restart;
    asymmetric_restart.coarse_method = CoarseCorrectionMethod::gmres;
    asymmetric_restart.coarse_restart = root.lane.rank() == 0 ? 32 : 64;
    EXPECT_THROW(solver.solve(asymmetric_restart, root.lane), std::invalid_argument);
  }
  for (std::size_t level = 0; level < boundaries.size(); ++level) {
    std::array<PhysicalBoundaryFace, 4> faces{};
    faces[0] = faces[1] = {PhysicalBoundaryKind::neumann, Real(0)};
    const auto& geometry = level == 0 ? root.geometry : fine_geometry;
    boundaries[level] = {root.boundary.topology(), faces,
        RealVector<2>{geometry.spacing(0), geometry.spacing(1)}};
  }
  EXPECT_THROW((void)FullTensorCompositeFac<2>(bindings, ratios, root.lane, {3u, 0u, true},
                                               CoarseCorrectionMethod::gmres), std::exception);
}

TEST(TensorCoarseGMRES, CompositeMappedDiskConvergesAndRejectsFailedInnerSolve) {
  const auto lane = ExecutionLane::world("tests.tensor-coarse-gmres.composite");
  const auto solved = pops::test::mapped_disk_fac_witness(
      8, 48, false, lane, {1u, 2u, true}, 4, 64, 512, 200, Real(0.5), Real(2e-9),
      CoarseCorrectionMethod::gmres);
  ASSERT_TRUE(solved.report.solved()) << solved.report.reason;
  const auto failed = pops::test::mapped_disk_fac_witness(
      8, 48, false, lane, {1u, 2u, true}, 4, 64, 1, 200, Real(0.5), Real(2e-9),
      CoarseCorrectionMethod::gmres);
  EXPECT_FALSE(failed.report.solved());
  EXPECT_EQ(failed.report.action, SolveAction::kFailRun);
  EXPECT_NE(failed.report.reason.find("coarse_correction"), std::string::npos);
}

}  // namespace
