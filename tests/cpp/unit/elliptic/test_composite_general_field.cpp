#include <gtest/gtest.h>

#include <pops/numerics/elliptic/nd/prepared_composite_general_field.hpp>

#include <array>
#include <cmath>
#include <limits>

namespace {
using namespace pops;
using namespace pops::runtime::program;
using Solver = elliptic::nd::PreparedCompositeGeneralField<1>;
using Provider = elliptic::nd::CompositeGeneralFieldProvider<1>;

HierarchyTensorSolverBuildRequest<1> three_field_request(const ExecutionLane& lane) {
  HierarchyTensorSolverBuildRequest<1> request;
  request.components = 3;
  request.block = std::numeric_limits<std::size_t>::max();  // field-owned, no species index
  request.plan_identity = "counterexample.three-field.composite@1";
  request.operator_contract_identity = "authored.matrix-diffusion.general-reaction@1";
  request.assembly_field_slots = {"pops.general-field.coefficients", "pops.general-field.rhs"};
  request.solution_field_slot = "pops.general-field.solution";
  request.options.schema_identity = "pops.hierarchy.general-field@1";
  auto& values = request.options.values;
  values["coefficient_components"] = std::uint64_t{9};
  values["restart"] = std::uint64_t{80};
  values["modes.count"] = std::uint64_t{2};
  // R=v v^T, v=(1,2,-1): PSD, non-lambda, with a two-dimensional kernel.
  constexpr std::array<double, 9> reaction{1, 2, -1, 2, 4, -2, -1, -2, 1};
  constexpr std::array<std::array<double, 3>, 2> modes{{{2, -1, 0}, {1, 0, 1}}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      values["reaction." + std::to_string(i) + "." + std::to_string(j)] = reaction[i * 3 + j];
  for (int mode = 0; mode < 2; ++mode)
    for (int i = 0; i < 3; ++i)
      values["modes." + std::to_string(mode) + "." + std::to_string(i)] = modes[mode][i];
  values["gauge.0"] = -1.0;
  values["gauge.1"] = 3.5;
  values["physical.0.0"] = std::string("neumann");
  values["physical.0.1"] = std::string("neumann");
  const mesh::RankSpace<1> ranks{Index<1>{0}, Extent<1>{lane.size()}};
  const Index<1> rank{lane.rank()};
  const auto coarse =
      Geometry<1>::from_bounds(Box<1>{{0}, {15}}, RealVector<1>{0}, RealVector<1>{1});
  const auto fine = coarse.refine(Extent<1>{2});
  // Genuine partial refinement, with coarse/fine interfaces at x=1/4 and x=3/4.
  // Each level is also split into patches, so same-level coefficient halos are exercised.
  const std::array<Geometry<1>, 2> geometries{coarse, fine};
  const std::vector<std::vector<Box<1>>> boxes{{Box<1>{{0}, {7}}, Box<1>{{8}, {15}}},
                                               {Box<1>{{8}, {15}}, Box<1>{{16}, {23}}}};
  for (int level = 0; level < 2; ++level) {
    const mesh::BoxArray<1> layout(boxes[level]);
    const auto distribution = mesh::Distribution<1>::partitioned(
        layout, ranks, std::vector<Index<1>>{{0}, {lane.size() > 1 ? 1 : 0}});
    std::array<PhysicalBoundaryFace, 2> faces{};
    faces.fill({PhysicalBoundaryKind::neumann, Real(0)});
    PhysicalBoundaryConditions<1> boundary{BoundaryTopology<1>::physical(), faces,
                                           RealVector<1>{geometries[level].spacing(0)}};
    request.levels.push_back({geometries[level], boundary, layout, distribution, rank});
  }
  request.ratios.push_back(::pops::amr::RefinementRatio<1>{std::array<int, 1>{2}});
  return request;
}

Solver::hierarchy_type hierarchy(const HierarchyTensorSolverBuildRequest<1>& request) {
  Solver::hierarchy_type result;
  for (const auto& level : request.levels)
    result.emplace_back(level.layout, level.distribution, level.local_rank, 3, Extent<1>{});
  return result;
}

void fill_coefficients(Solver& solver) {
  constexpr std::array<Real, 9> matrix{Real(2),     Real(-0.35), Real(0.2),
                                       Real(-0.35), Real(1.6),   Real(-0.25),
                                       Real(0.2),   Real(-0.25), Real(1.3)};
  for (int level = 0; level < solver.level_count(); ++level) {
    auto& field = solver.assembly_target("pops.general-field.coefficients", level);
    for (std::size_t patch = 0; patch < field.local_size(); ++patch) {
      const auto output = field.fab(patch).view();
      for_each_cell(field.box(patch), [=] POPS_HD(const Index<1>& cell) {
        // Nonconstant positive scalar times a signed SPD component matrix.
        const Real factor = Real(1) + Real(0.01) * Real(cell[0]);
        for (int slot = 0; slot < 9; ++slot)
          output(cell, slot) = factor * matrix[slot];
      });
    }
  }
}

void fill_constant(Solver::hierarchy_type& fields, std::array<Real, 3> components) {
  for (auto& field : fields)
    for (std::size_t patch = 0; patch < field.local_size(); ++patch) {
      const auto output = field.fab(patch).view();
      for_each_cell(field.box(patch), [=] POPS_HD(const Index<1>& cell) {
        for (int component = 0; component < 3; ++component)
          output(cell, component) = components[component];
      });
    }
}

void copy_fields(const Solver::hierarchy_type& input, Solver& solver, std::string_view slot) {
  for (int level = 0; level < solver.level_count(); ++level) {
    auto& destination = solver.assembly_target(slot, level);
    lincomb(destination, Real(1), input[level], Real(0), input[level]);
  }
}

TEST(CompositeGeneralField, ThreeFieldsSignedDiffusionGeneralReactionTwoModesAndReflux) {
  auto lane = ExecutionLane::world();
  const auto request = three_field_request(lane);
  Provider provider;
  ASSERT_TRUE(provider.supports(request).accepted());
  Solver solver(request, provider.expected_prepared_contract(request), lane);
  solver.seal_preparation(lane);
  fill_coefficients(solver);
  solver.prepare_coefficients();
  // The observation coverage excludes 8 covered coarse cells: 8 coarse + 16 fine,
  // whereas summing both storage levels would incorrectly count 32 cells.
  PreparedHierarchyTensorSolver<1>& observations = solver;
  Real active_count = 0;
  for (int level = 0; level < observations.level_count(); ++level)
    active_count += reduce_sum_local(observations.active_cell_mask(level), 0);
  EXPECT_EQ(all_reduce_sum(active_count, lane), Real(24));
  auto exact = hierarchy(request), forcing = hierarchy(request), mode0 = hierarchy(request),
       mode1 = hierarchy(request), difference = hierarchy(request);
  fill_constant(mode0, {Real(2), Real(-1), Real(0)});
  fill_constant(mode1, {Real(1), Real(0), Real(1)});
  for (int level = 0; level < 2; ++level) {
    const auto geometry = request.levels[level].geometry;
    for (std::size_t patch = 0; patch < exact[level].local_size(); ++patch) {
      const auto output = exact[level].fab(patch).view();
      for_each_cell(exact[level].box(patch), [=] POPS_HD(const Index<1>& cell) {
        const Real x = geometry.cell_coordinate(0, cell[0]);
        const Real a = std::cos(Real(3.14159265358979323846) * x);
        const Real b = std::cos(Real(6.28318530717958647692) * x);
        output(cell, 0) = a + Real(0.4) * b;
        output(cell, 1) = Real(2) * a - Real(0.2) * b;
        output(cell, 2) = -a + Real(0.1) * b;
      });
    }
  }
  // Set the two nonorthogonal basis coordinates to the authored gauge. The physical
  // moments are B mean(phi)=(2,5), not independent per-field zero means.
  const Real m0 = solver.composite_dot(exact, mode0), m1 = solver.composite_dot(exact, mode1);
  const Real c0 = (Real(2) * m0 - Real(2) * m1) / Real(6);
  const Real c1 = (-Real(2) * m0 + Real(5) * m1) / Real(6);
  for (int level = 0; level < 2; ++level) {
    saxpy(exact[level], Real(-1) - c0, mode0[level]);
    saxpy(exact[level], Real(3.5) - c1, mode1[level]);
  }
  solver.apply(exact, forcing);
  // Independent conservation witness: a missing coarse/fine flux replacement leaves a
  // nonzero global mode moment even if a self-generated manufactured solve converges.
  EXPECT_NEAR(solver.composite_dot(forcing, mode0), Real(0), Real(2e-10));
  EXPECT_NEAR(solver.composite_dot(forcing, mode1), Real(0), Real(2e-10));
  copy_fields(forcing, solver, "pops.general-field.rhs");
  for (int level = 0; level < 2; ++level)
    solver.solution(level).set_val(Real(7));
  auto outcome = solver.execute_collectively({Real(1e-11), Real(1e-11), 400}, lane);
  const SolveReport report = outcome.report();
  // Provider candidates are not published before the existing transaction is consumed.
  for (int level = 0; level < 2; ++level)
    for (int component = 0; component < 3; ++component)
      EXPECT_NEAR(norm_inf(solver.solution(level), component), Real(7), Real(0));
  const auto consumed =
      outcome.consume(report.solved() ? SolveConsumption::kAccept : SolveConsumption::kFailRun);
  ASSERT_TRUE(consumed.solved()) << consumed.reason;
  EXPECT_LE(consumed.residual_norm, consumed.reference_residual_norm * Real(1.1e-11));
  for (int level = 0; level < 2; ++level)
    lincomb(difference[level], Real(1), solver.solution(level), Real(-1), exact[level]);
  EXPECT_LT(std::sqrt(solver.composite_dot(difference, difference)), Real(2e-8));
  for (int level = 0; level < 2; ++level)
    lincomb(exact[level], Real(1), solver.solution(level), Real(0), solver.solution(level));
  EXPECT_NEAR(solver.composite_dot(exact, mode0), Real(2), Real(2e-10));
  EXPECT_NEAR(solver.composite_dot(exact, mode1), Real(5), Real(2e-10));
}

TEST(CompositeGeneralField, IncompatibleRhsAndIndefiniteDiffusionDoNotPublish) {
  auto lane = ExecutionLane::world();
  const auto request = three_field_request(lane);
  Provider provider;
  Solver solver(request, provider.expected_prepared_contract(request), lane);
  solver.seal_preparation(lane);
  fill_coefficients(solver);
  auto incompatible = hierarchy(request);
  fill_constant(incompatible, {Real(2), Real(-1), Real(0)});
  copy_fields(incompatible, solver, "pops.general-field.rhs");
  for (int level = 0; level < 2; ++level)
    solver.solution(level).set_val(Real(4));
  auto rejected = solver.execute_collectively({Real(1e-10), Real(1e-12), 100}, lane);
  EXPECT_EQ(rejected.consume(SolveConsumption::kFailRun).status, SolveStatus::kIncompatibleRhs);
  for (int level = 0; level < 2; ++level)
    for (int component = 0; component < 3; ++component)
      EXPECT_EQ(norm_inf(solver.solution(level), component), Real(4));
  // A signed cross term is allowed, but a negative principal eigenvalue is not.
  solver.assembly_target("pops.general-field.coefficients", 0).set_val(Real(-1));
  auto invalid = solver.execute_collectively({Real(1e-10), Real(1e-12), 100}, lane);
  EXPECT_EQ(invalid.consume(SolveConsumption::kFailRun).status, SolveStatus::kInvalidEvaluation);
  for (int level = 0; level < 2; ++level)
    for (int component = 0; component < 3; ++component)
      EXPECT_EQ(norm_inf(solver.solution(level), component), Real(4));
}

TEST(CompositeGeneralField, FacBorrowedOwnedLaneDoesNotClaimOwnership) {
  auto lane = ExecutionLane::duplicate_world_collectively("composite-field-borrow-test");
  const auto request = three_field_request(lane);
  const auto scalar = elliptic::nd::general_composite_detail::fac_request(request);
  elliptic::amr::CompositeFacPoisson<1> borrowed(scalar, {}, Real(0), &lane, true);
  EXPECT_FALSE(borrowed.owns_execution_lane());
  elliptic::amr::CompositeFacPoisson<1> owned(scalar, {}, Real(0), nullptr, true);
  EXPECT_TRUE(owned.owns_execution_lane());
}

TEST(CompositeGeneralField, RefusesMissingAndWrongConstantModes) {
  auto lane = ExecutionLane::world();
  auto request = three_field_request(lane);
  Provider provider;
  request.options.values["modes.0.0"] = 1.0;
  EXPECT_FALSE(provider.supports(request).accepted());
  request = three_field_request(lane);
  request.options.values["modes.count"] = std::uint64_t{1};
  EXPECT_FALSE(provider.supports(request).accepted());
}
}  // namespace
