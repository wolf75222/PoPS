#include <gtest/gtest.h>

#include <bit>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <unistd.h>

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
  ExecutionLane lane = ExecutionLane::duplicate_world_collectively("tests.tensor-coarse-gmres");
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

  RootProblem(bool replicated = false, bool empty_owner = false,
              std::optional<CoarsePreconditionerKind> preconditioner = std::nullopt)
      : geometry(Geometry<2>::from_bounds({Index<2>{0, 0}, Index<2>{7, 15}}, RealVector<2>{0, 0},
                                          RealVector<2>{1, 2 * pi})),
        boundary(boundaries(geometry)),
        layout(std::vector<Box<2>>{{Index<2>{0, 0}, Index<2>{7, 7}},
                                   {Index<2>{0, 8}, Index<2>{7, 15}}}),
        distribution(ownership(layout, lane.size(), replicated, empty_owner)),
        exact(layout, distribution, Index<2>{lane.rank(), 0}, 1, Extent<2>{1, 1}),
        rhs(layout, distribution, Index<2>{lane.rank(), 0}, 1, Extent<2>{}),
        correction(layout, distribution, Index<2>{lane.rank(), 0}, 1, Extent<2>{1, 1}),
        halo(prepare_halo_schedule(
            exact, geometry.domain(), boundary.topology(), HaloLayoutCoverage::full_domain,
            runtime::program::tensor_fac::detail::exact_halo_budget<2>(layout, geometry.domain()))),
        physical(prepare_physical_boundary(
            geometry.domain(), Extent<2>{1, 1}, boundary,
            runtime::program::tensor_fac::detail::exact_boundary_budget<2>())),
        coefficient_boundary(prepare_physical_boundary(
            geometry.domain(), Extent<2>{1, 1},
            runtime::program::tensor_fac::detail::boundary_with_values<2>(boundary, geometry, false,
                                                                          true),
            runtime::program::tensor_fac::detail::exact_boundary_budget<2>())) {
    for (auto& field : coefficients)
      field = std::make_unique<Field>(layout, distribution, Index<2>{lane.rank(), 0},
                                      1, Extent<2>{1, 1});
    if (all_reduce_max(halo.has_remote_jobs() ? 1L : 0L, lane) != 0) {
      HaloExchangeContext context{};
      context.context_generation = context.schedule_generation = 1;
      exchange.emplace(halo, lane, context);
    }
    if (preconditioner)
      gmres = std::make_unique<TensorCoarseGmres<2>>(
          correction, geometry, halo, physical, options, lane,
          "tests.tensor-coarse-gmres.exact-mixed-boundary-operator", 64, *preconditioner);
    else
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

  // Use the original staged coefficient fields and independent boundary route, never the
  // solver's private coefficient copy or its report, to verify the accepted physical residual.
  Real original_residual() {
    fill(correction, physical);
    Real local_residual = 0;
    for (std::size_t local = 0; local < correction.local_size(); ++local) {
      std::array<FieldView<const Real, 2>, 4> values{};
      for (std::size_t slot = 0; slot < 4; ++slot)
        values[slot] = std::as_const(coefficients[slot]->fab(local)).view();
      const auto stencil = elliptic::nd::make_cartesian_tensor_operator<
          elliptic::nd::CartesianTensorDivergenceSign::negative_divergence>(
          std::as_const(correction.fab(local)).view(),
          elliptic::nd::split_cartesian_tensor_coefficients<2>(values), geometry, options);
      const auto right = std::as_const(rhs.fab(local)).view();
      local_residual = std::max(
          local_residual,
          for_each_cell_reduce_max(correction.box(local), [=] POPS_HD(const Index<2>& index) {
            const Real residual = Kokkos::abs(right(index, 0) - stencil.image(index));
            return residual >= Real(0) && residual < std::numeric_limits<Real>::infinity()
                       ? residual
                       : std::numeric_limits<Real>::infinity();
          }));
    }
    return static_cast<Real>(all_reduce_max(static_cast<double>(local_residual), lane));
  }

  Real difference(const RootProblem& other) const {
    Real local_error = 0;
    for (std::size_t local = 0; local < correction.local_size(); ++local) {
      const auto left = correction.fab(local).view();
      const auto right = other.correction.fab(local).view();
      local_error = std::max(
          local_error,
          for_each_cell_reduce_max(correction.box(local), [=] POPS_HD(const Index<2>& index) {
            return Kokkos::abs(left(index, 0) - right(index, 0));
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

TEST(TensorCoarseGMRES, ExplicitDiagonalPreservesTheExistingDefault) {
  RootProblem implicit;
  RootProblem explicit_diagonal(false, false, CoarsePreconditionerKind::diagonal);
  implicit.stage(Real(0.2));
  explicit_diagonal.stage(Real(0.2));
  const auto original = implicit.gmres->solve(implicit.correction, implicit.rhs, Real(1e-10), 512);
  const auto selected = explicit_diagonal.gmres->solve(explicit_diagonal.correction,
                                                       explicit_diagonal.rhs, Real(1e-10), 512);
  ASSERT_TRUE(original.solved()) << original.reason;
  ASSERT_TRUE(selected.solved()) << selected.reason;
  EXPECT_EQ(original.iters, selected.iters);
  EXPECT_EQ(original.residual_norm, selected.residual_norm);
  EXPECT_EQ(implicit.difference(explicit_diagonal), Real(0));
  EXPECT_LE(implicit.original_residual(), Real(1e-10));
  EXPECT_LE(explicit_diagonal.original_residual(), Real(1e-10));
}

TEST(TensorCoarseGMRES, PolarPreconditionerKeepsTheFullTensorAndRefreshesWithoutAllocation) {
  RootProblem problem(false, false, CoarsePreconditionerKind::polar_poisson);
  problem.stage(Real(0.1));
  auto report = problem.gmres->solve(problem.correction, problem.rhs, Real(1e-10), 512);
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_LE(report.residual_norm, Real(1e-10));
  EXPECT_LE(problem.original_residual(), Real(1e-10));
  EXPECT_LT(problem.error(), Real(2e-10));

  const auto allocations = allocation_event_stats();
  problem.stage(Real(0.2));
  // This API computes a correction from zero; an old or poisoned destination is not a warm start.
  problem.correction.set_val(Real(17));
  report = problem.gmres->solve(problem.correction, problem.rhs, Real(1e-10), 512);
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_LE(report.residual_norm, Real(1e-10));
  EXPECT_LE(problem.original_residual(), Real(1e-10));
  EXPECT_LT(problem.error(), Real(2e-10));
  EXPECT_EQ(allocation_event_stats(), allocations);

  problem.rhs.set_val(Real(0));
  problem.correction.set_val(Real(-9));
  report = problem.gmres->solve(problem.correction, problem.rhs, Real(1e-10), 512);
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_EQ(report.iters, 0);
  EXPECT_EQ(problem.original_residual(), Real(0));
  EXPECT_EQ(allocation_event_stats(), allocations);
}

TEST(TensorCoarseGMRES, PolarPreconditionerRetainsEmptyAndReplicatedOwners) {
  for (const bool replicated : {false, true}) {
    RootProblem problem(replicated, !replicated, CoarsePreconditionerKind::polar_poisson);
    problem.stage(Real(0.2));
    if (!replicated && problem.lane.size() > 1 && problem.lane.rank() != 0)
      EXPECT_EQ(problem.correction.local_size(), 0u);
    const auto report = problem.gmres->solve(problem.correction, problem.rhs, Real(1e-10), 512);
    ASSERT_TRUE(report.solved()) << report.reason;
    EXPECT_LE(problem.original_residual(), Real(1e-10));
    EXPECT_LT(problem.error(), Real(2e-10));
  }
}

TEST(TensorCoarseGMRES, PolarPreconditionerDoesNotAcceptATruncatedOrInvalidFullSolve) {
  RootProblem problem(false, false, CoarsePreconditionerKind::polar_poisson);
  problem.stage(Real(0.2));
  const auto report = problem.gmres->solve(problem.correction, problem.rhs, Real(1e-12), 1);
  EXPECT_FALSE(report.solved());
  EXPECT_EQ(report.status, SolveStatus::kIterationLimit);
  EXPECT_EQ(report.action, SolveAction::kFailRun);
  EXPECT_GT(problem.original_residual(), Real(1e-12));
  for (auto& coefficient : problem.coefficients)
    coefficient->set_val(Real(0));
  EXPECT_THROW(problem.gmres->prepare_coefficients(
                   {problem.coefficients[0].get(), problem.coefficients[1].get(),
                    problem.coefficients[2].get(), problem.coefficients[3].get()}),
               std::invalid_argument);
  EXPECT_THROW(problem.gmres->solve(problem.correction, problem.rhs, Real(1e-10), 512),
               std::invalid_argument);
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
  controls.coarse_restart = 64;
  controls.coarse_preconditioner = static_cast<CoarsePreconditionerKind>(19);
  EXPECT_THROW(runtime::program::tensor_fac::detail::validate_controls(controls),
               std::invalid_argument);
  controls.coarse_method = CoarseCorrectionMethod::gauss_seidel;
  controls.coarse_preconditioner = CoarsePreconditionerKind::polar_poisson;
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
  {
    FullTensorCompositeFac<2> solver(bindings, ratios, root.lane, root.options,
                                     CoarseCorrectionMethod::gmres, 64,
                                     CoarsePreconditionerKind::polar_poisson);
    Controls wrong_preconditioner;
    wrong_preconditioner.coarse_method = CoarseCorrectionMethod::gmres;
    EXPECT_THROW(solver.solve(wrong_preconditioner, root.lane), std::invalid_argument);
    Controls asymmetric_preconditioner = wrong_preconditioner;
    asymmetric_preconditioner.coarse_preconditioner = root.lane.rank() == 0
                                                          ? CoarsePreconditionerKind::diagonal
                                                          : CoarsePreconditionerKind::polar_poisson;
    EXPECT_THROW(solver.solve(asymmetric_preconditioner, root.lane), std::invalid_argument);
  }
  if (root.lane.size() > 1) {
    const auto asymmetric_preconditioner = root.lane.rank() == 0
                                               ? CoarsePreconditionerKind::diagonal
                                               : CoarsePreconditionerKind::polar_poisson;
    EXPECT_THROW((void)FullTensorCompositeFac<2>(bindings, ratios, root.lane, root.options,
                                                 CoarseCorrectionMethod::gmres, 64,
                                                 asymmetric_preconditioner),
                 std::exception);
  }
  const auto invalid_on_one_rank = root.lane.rank() == 0 ? static_cast<CoarsePreconditionerKind>(19)
                                                         : CoarsePreconditionerKind::polar_poisson;
  EXPECT_THROW((void)TensorCoarseGmres<2>(root.correction, root.geometry, root.halo, root.physical,
                                          root.options, root.lane,
                                          "tests.tensor-coarse-gmres.invalid-preconditioner", 64,
                                          invalid_on_one_rank),
               std::exception);
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
  const auto lane = ExecutionLane::duplicate_world_collectively("tests.tensor-coarse-gmres.composite");
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

TEST(TensorCoarseGMRES, PolarCompositeMappedDiskRetainsConvergenceAndInnerFailure) {
  const auto lane = ExecutionLane::duplicate_world_collectively("tests.tensor-polar.composite");
  const auto coarse = pops::test::mapped_disk_fac_witness(
      8, 48, false, lane, {1u, 2u, true}, 4, 64, 512, 200, Real(0.5), Real(2e-9),
      CoarseCorrectionMethod::gmres, 64, CoarsePreconditionerKind::polar_poisson);
  const auto fine = pops::test::mapped_disk_fac_witness(
      16, 96, false, lane, {1u, 2u, true}, 4, 64, 512, 200, Real(0.5), Real(2e-9),
      CoarseCorrectionMethod::gmres, 64, CoarsePreconditionerKind::polar_poisson);
  ASSERT_TRUE(coarse.report.solved()) << coarse.report.reason;
  ASSERT_TRUE(fine.report.solved()) << fine.report.reason;
  EXPECT_LT(fine.maximum_error, coarse.maximum_error / Real(3));
  for (const auto& report : {coarse.report, fine.report})
    EXPECT_LE(report.residual_norm,
              std::max(Real(1e-12), Real(2e-9) * report.reference_residual_norm));
  const auto failed = pops::test::mapped_disk_fac_witness(
      8, 48, false, lane, {1u, 2u, true}, 4, 64, 1, 200, Real(0.5), Real(2e-9),
      CoarseCorrectionMethod::gmres, 64, CoarsePreconditionerKind::polar_poisson);
  EXPECT_FALSE(failed.report.solved());
  EXPECT_EQ(failed.report.action, SolveAction::kFailRun);
  EXPECT_NE(failed.report.reason.find("coarse_correction"), std::string::npos);
  const auto outer_limit = pops::test::mapped_disk_fac_witness(
      8, 48, false, lane, {1u, 2u, true}, 4, 64, 512, 1, Real(0.5), Real(2e-9),
      CoarseCorrectionMethod::gmres, 64, CoarsePreconditionerKind::polar_poisson);
  EXPECT_EQ(outer_limit.report.status, SolveStatus::kIterationLimit);
  EXPECT_EQ(outer_limit.report.action, SolveAction::kFailRun);
  EXPECT_EQ(outer_limit.report.iters, 1);
  EXPECT_GT(outer_limit.report.residual_norm,
            std::max(Real(1e-12), Real(2e-9) * outer_limit.report.reference_residual_norm));
  EXPECT_EQ(outer_limit.report.reason.find("nd_tensor_fac_iteration_limit [iterations=1,"), 0u);
  EXPECT_NE(outer_limit.report.reason.find("residual_linf="), std::string::npos);
  EXPECT_NE(outer_limit.report.reason.find("reference_linf="), std::string::npos);
  EXPECT_NE(outer_limit.report.reason.find("requested_tolerance="), std::string::npos);
}

struct FullMappedHierarchyMeasurement {
  SolveReport report;
  Real maximum_error;
  long finest_owned_cells;
};

// Independent continuous MMS on two or three genuine factor-two levels. Refining the whole
// domain isolates the extra FAC restriction/correction level; the preceding witness covers
// partial refinement, periodic coarse/fine seams and the fine-interface boundary arithmetic.
FullMappedHierarchyMeasurement full_mapped_hierarchy(int level_count, const ExecutionLane& lane) {
  if (level_count != 2 && level_count != 3)
    throw std::invalid_argument("the full mapped hierarchy witness requires two or three levels");
  std::vector<Geometry<2>> geometry;
  std::vector<PhysicalBoundaryConditions<2>> boundary;
  std::vector<mesh::BoxArray<2>> layouts;
  std::vector<mesh::Distribution<2>> distributions;
  geometry.reserve(level_count);
  boundary.reserve(level_count);
  layouts.reserve(level_count);
  distributions.reserve(level_count);
  geometry.push_back(Geometry<2>::from_bounds({Index<2>{0, 0}, Index<2>{7, 31}},
                                              RealVector<2>{0, 0}, RealVector<2>{1, Real(2) * pi}));
  for (int level = 1; level < level_count; ++level)
    geometry.push_back(geometry.back().refine(Extent<2>{2, 2}));
  const Index<2> rank{lane.rank(), 0};
  std::array<std::array<std::unique_ptr<Field>, 4>, 3> coefficients;
  std::array<std::unique_ptr<Field>, 3> rhs, initial, solution;
  std::vector<LevelBinding<2, Field::memory_space>> bindings;
  bindings.reserve(level_count);
  for (int level = 0; level < level_count; ++level) {
    const int nr = static_cast<int>(geometry[level].domain().length(0));
    const int nt = static_cast<int>(geometry[level].domain().length(1));
    boundary.push_back(RootProblem::boundaries(geometry[level]));
    layouts.emplace_back(std::vector<Box<2>>{{Index<2>{0, 0}, Index<2>{nr - 1, nt / 2 - 1}},
                                             {Index<2>{0, nt / 2}, Index<2>{nr - 1, nt - 1}}});
    distributions.push_back(RootProblem::ownership(layouts.back(), lane.size(), false, false));
    rhs[level] =
        std::make_unique<Field>(layouts.back(), distributions.back(), rank, 1, Extent<2>{});
    initial[level] =
        std::make_unique<Field>(layouts.back(), distributions.back(), rank, 1, Extent<2>{1, 1});
    solution[level] =
        std::make_unique<Field>(layouts.back(), distributions.back(), rank, 1, Extent<2>{1, 1});
    initial[level]->set_val(Real(0));
    solution[level]->set_val(Real(0));
    for (auto& coefficient : coefficients[level]) {
      coefficient =
          std::make_unique<Field>(layouts.back(), distributions.back(), rank, 1, Extent<2>{1, 1});
      coefficient->set_val(std::numeric_limits<Real>::quiet_NaN());
    }
    for (std::size_t local = 0; local < rhs[level]->local_size(); ++local) {
      std::array<FieldView<Real, 2>, 4> tensor{};
      for (std::size_t slot = 0; slot < 4; ++slot)
        tensor[slot] = coefficients[level][slot]->fab(local).view();
      const auto forcing = rhs[level]->fab(local).view();
      const auto mesh = geometry[level];
      for_each_cell(rhs[level]->box(local), [=] POPS_HD(const Index<2>& index) {
        const Real r = mesh.cell_coordinate(0, index[0]);
        const Real theta = mesh.cell_coordinate(1, index[1]);
        constexpr Real epsilon = Real(0.1), alpha = Real(39.4784176e12);
        constexpr Real omega = Real(-6.28318531e12), s = Real(1e-4);
        const Real w = s * omega;
        const Real response = s * s * alpha / (Real(1) + w * w);
        const Real r2 = r * r, r3 = r2 * r, r4 = r2 * r2;
        const Real density = Real(1) + epsilon * r2 * Kokkos::sin(theta);
        const Real density_r = Real(2) * epsilon * r * Kokkos::sin(theta);
        const Real density_theta = epsilon * r2 * Kokkos::cos(theta);
        const Real psi_r = (Real(3) * r2 - Real(5) * r4) * Kokkos::cos(Real(3) * theta);
        const Real psi_theta = -Real(3) * r3 * (Real(1) - r2) * Kokkos::sin(Real(3) * theta);
        const Real symmetric = Real(1) + response * density;
        const Real skew = response * w * density;
        tensor[0](index, 0) = r * symmetric;
        tensor[1](index, 0) = skew;
        tensor[2](index, 0) = -skew;
        tensor[3](index, 0) = symmetric / r;
        forcing(index, 0) = symmetric * Real(16) * r4 * Kokkos::cos(Real(3) * theta) -
                            response * (r * density_r * psi_r + density_theta * psi_theta / r) -
                            response * w * (density_r * psi_theta - density_theta * psi_r);
      });
    }
    bindings.push_back({&geometry[level],
                        &boundary[level],
                        {coefficients[level][0].get(), coefficients[level][1].get(),
                         coefficients[level][2].get(), coefficients[level][3].get()},
                        rhs[level].get(),
                        initial[level].get(),
                        solution[level].get()});
  }
  Kokkos::fence();
  const std::vector<amr::RefinementRatio<2>> ratios(
      static_cast<std::size_t>(level_count - 1), amr::RefinementRatio<2>{std::array<int, 2>{2, 2}});
  FullTensorCompositeFac<2> solver(bindings, ratios, lane, {1u, 2u, true},
                                   CoarseCorrectionMethod::gmres, 64,
                                   CoarsePreconditionerKind::polar_poisson);
  Controls controls;
  controls.relative_tolerance = Real(2e-9);
  controls.absolute_tolerance = Real(1e-12);
  controls.maximum_iterations = 200;
  controls.fine_sweeps = 64;
  controls.coarse_cycles = 512;
  controls.coarse_relative_tolerance = Real(1e-11);
  controls.correction_damping = Real(0.5);
  controls.coarse_method = CoarseCorrectionMethod::gmres;
  controls.coarse_preconditioner = CoarsePreconditionerKind::polar_poisson;
  FullMappedHierarchyMeasurement result{solver.solve(controls, lane), Real(0), 0};
  const int finest = level_count - 1;
  for (std::size_t local = 0; local < solution[finest]->local_size(); ++local) {
    const auto value = solution[finest]->fab(local).view();
    const auto mesh = geometry[finest];
    result.maximum_error = std::max(
        result.maximum_error,
        for_each_cell_reduce_max(solution[finest]->box(local), [=] POPS_HD(const Index<2>& index) {
          const Real r = mesh.cell_coordinate(0, index[0]);
          const Real theta = mesh.cell_coordinate(1, index[1]);
          const Real exact = r * r * r * (Real(1) - r * r) * Kokkos::cos(Real(3) * theta);
          const Real error = Kokkos::abs(value(index, 0) - exact);
          return error >= Real(0) && error < std::numeric_limits<Real>::infinity()
                     ? error
                     : std::numeric_limits<Real>::infinity();
        }));
    result.finest_owned_cells += static_cast<long>(solution[finest]->box(local).numPts());
  }
  result.maximum_error =
      static_cast<Real>(all_reduce_max(static_cast<double>(result.maximum_error), lane));
  result.finest_owned_cells = all_reduce_sum(result.finest_owned_cells, lane);
  return result;
}

TEST(TensorCoarseGMRES, PolarPreconditionerTraversesThreeActualMappedLevels) {
  const auto lane =
      ExecutionLane::duplicate_world_collectively("tests.tensor-polar.three-level-mms");
  const auto two = full_mapped_hierarchy(2, lane);
  const auto three = full_mapped_hierarchy(3, lane);
  ASSERT_TRUE(two.report.solved()) << two.report.reason;
  ASSERT_TRUE(three.report.solved()) << three.report.reason;
  EXPECT_EQ(two.finest_owned_cells, 16L * 64L);
  EXPECT_EQ(three.finest_owned_cells, 32L * 128L);
  EXPECT_LT(three.maximum_error, two.maximum_error / Real(3));
  for (const auto& report : {two.report, three.report})
    EXPECT_LE(report.residual_norm,
              std::max(Real(1e-12), Real(2e-9) * report.reference_residual_norm));
}


TEST(TensorCoarseGMRES, OptionalFailureCaptureRetainsCandidateAndExclusiveBytes) {
  if constexpr (!kRealIsBinary64) GTEST_SKIP() << "Capture schema requires binary64";
  comm_init();
  ::pops::detail::ensure_kokkos_initialized();  // Empty owners still prepare halo buffers.
  struct RestoreCaptureEnvironment {
    std::optional<std::string> previous;
    RestoreCaptureEnvironment() {
      if (const char* value = std::getenv("POPS_TENSOR_FAC_CAPTURE_DIR")) previous = value;
    }
    ~RestoreCaptureEnvironment() {
      if (previous) ::setenv("POPS_TENSOR_FAC_CAPTURE_DIR", previous->c_str(), 1);
      else ::unsetenv("POPS_TENSOR_FAC_CAPTURE_DIR");
    }
  } restore;
  std::array<char, 128> directory{};
  if (my_rank() == 0) {
    std::snprintf(directory.data(), directory.size(), "/tmp/pops-fac-capture-XXXXXX");
    if (::mkdtemp(directory.data()) == nullptr) directory[0] = '\0';
  }
  broadcast_bytes_inplace(directory.data(), directory.size());
  ASSERT_NE(directory[0], '\0');
  const std::string path = std::string(directory.data()) + "/coarse-rank-" +
                           std::to_string(my_rank()) + ".txt";
  const auto read_bytes = [&] {
    std::ifstream stream(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
  };
  const auto candidate_bits = [](const Field& field) {
    std::vector<RealBits> result;
    for (std::size_t local = 0; local < field.local_size(); ++local) {
      auto host = field.fab(local).create_host_mirror();
      field.fab(local).copy_to_host(host);
      for (std::size_t index = 0; index < host.size(); ++index)
        result.push_back(std::bit_cast<RealBits>(host(index)));
    }
    return result;
  };
  for (const auto [replicated, empty] :
       std::array<std::pair<bool, bool>, 3>{{{false, false}, {false, true}, {true, false}}}) {
    SCOPED_TRACE(replicated ? "replicated" : empty ? "empty-owner" : "partitioned");
    ::unsetenv("POPS_TENSOR_FAC_CAPTURE_DIR");
    RootProblem original(replicated, empty, CoarsePreconditionerKind::polar_poisson);
    original.stage(Real(0.1));
    // One Arnoldi column cannot solve this manufactured full-tensor fixture.
    // Retain that bounded refusal solely to exercise the observer.
    constexpr Real tolerance = Real(1e-12);
    const auto expected = original.gmres->solve(original.correction, original.rhs, tolerance, 1);
    const Real original_residual = original.original_residual();
    EXPECT_FALSE(expected.solved());
    ::setenv("POPS_TENSOR_FAC_CAPTURE_DIR", directory.data(), 1);
    RootProblem observed(replicated, empty, CoarsePreconditionerKind::polar_poisson);
    observed.stage(Real(0.1));
    const auto report = observed.gmres->solve(observed.correction, observed.rhs, tolerance, 1);
    const Real residual = observed.original_residual();
    EXPECT_EQ(candidate_bits(observed.correction), candidate_bits(original.correction));
    EXPECT_EQ(report.status, expected.status);
    EXPECT_EQ(report.action, expected.action);
    EXPECT_EQ(report.iters, expected.iters);
    EXPECT_EQ(report.reason, expected.reason);
    EXPECT_DOUBLE_EQ(report.residual_norm, expected.residual_norm);
    EXPECT_DOUBLE_EQ(residual, original_residual);
    observed.gmres->capture_failure(observed.correction, observed.rhs, report, residual,
                                    report.reference_residual_norm, tolerance, 1, 1);
    const auto captured = read_bytes();
    EXPECT_EQ(captured.find("POPS_TENSOR_COARSE_CAPTURE 1\n"), 0u);
    EXPECT_NE(captured.find("\nEND\n"), std::string::npos);
    observed.gmres->capture_failure(observed.correction, observed.rhs, report, residual,
                                    report.reference_residual_norm, tolerance, 1, 1);
    EXPECT_EQ(read_bytes(), captured);  // O_EXCL collision must not overwrite existing evidence.
    EXPECT_FALSE(report.solved());
    EXPECT_EQ(::unlink(path.c_str()), 0);
    (void)all_reduce_max(0L);  // Finish per-rank cleanup before the next collective capture.
  }
  if (n_ranks() > 1) {
    if (my_rank() == 0) ::setenv("POPS_TENSOR_FAC_CAPTURE_DIR", directory.data(), 1);
    else ::unsetenv("POPS_TENSOR_FAC_CAPTURE_DIR");
    EXPECT_THROW((RootProblem{false, false, CoarsePreconditionerKind::polar_poisson}),
                 std::invalid_argument);
  }
  (void)all_reduce_max(0L);
  if (my_rank() == 0) EXPECT_EQ(::rmdir(directory.data()), 0);
}

}  // namespace
