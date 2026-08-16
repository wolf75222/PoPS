#include <gtest/gtest.h>

#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/runtime/amr/amr_tensor_elliptic.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

template <int Dim>
pops::Extent<Dim> extents(std::int64_t value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::RealVector<Dim> coordinates(pops::Real value) {
  pops::RealVector<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::PhysicalBoundaryConditions<Dim> homogeneous_dirichlet(const pops::Geometry<Dim>& geometry) {
  std::array<pops::PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  pops::RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis) {
    spacing[axis] = geometry.spacing(axis);
    for (const pops::BoundarySide side : {pops::BoundarySide::lower, pops::BoundarySide::upper})
      faces[static_cast<std::size_t>(pops::Face<Dim>{axis, side}.ordinal())] =
          pops::PhysicalBoundaryFace{pops::PhysicalBoundaryKind::dirichlet, pops::Real(0)};
  }
  return {pops::BoundaryTopology<Dim>::physical(), faces, spacing};
}

template <int Dim>
pops::runtime::program::HierarchyTensorSolverBuildRequest<Dim> request(
    bool refined, const std::array<int, Dim>& ratio_components) {
  using namespace pops;
  using namespace pops::runtime::program;

  Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = 3;
  const Box<Dim> domain{Index<Dim>{}, upper};
  const Geometry<Dim> geometry =
      Geometry<Dim>::from_bounds(domain, coordinates<Dim>(Real(0)), coordinates<Dim>(Real(1)));
  const mesh::BoxArray<Dim> layout(std::vector<Box<Dim>>{domain});
  const mesh::RankSpace<Dim> rank_space{Index<Dim>{}, extents<Dim>(1)};
  const mesh::Distribution<Dim> distribution =
      mesh::Distribution<Dim>::replicated(layout, rank_space);
  HierarchyTensorLevelBuildRequest<Dim> level{geometry, homogeneous_dirichlet(geometry), layout,
                                              distribution, Index<Dim>{}};

  HierarchyTensorSolverBuildRequest<Dim> result;
  result.block = 4;
  result.components = 1;
  result.levels.push_back(std::move(level));
  if (refined) {
    const amr::RefinementRatio<Dim> ratio{ratio_components};
    Extent<Dim> ratio_extent{};
    for (int axis = 0; axis < Dim; ++axis)
      ratio_extent[axis] = ratio_components[static_cast<std::size_t>(axis)];
    const Geometry<Dim> fine_geometry = geometry.refine(ratio_extent);
    const mesh::BoxArray<Dim> fine_layout(std::vector<Box<Dim>>{fine_geometry.domain()});
    const mesh::Distribution<Dim> fine_distribution = mesh::Distribution<Dim>::partitioned(
        fine_layout, rank_space, std::vector<Index<Dim>>{Index<Dim>{}});
    result.levels.push_back(
        HierarchyTensorLevelBuildRequest<Dim>{fine_geometry, homogeneous_dirichlet(fine_geometry),
                                              fine_layout, fine_distribution, Index<Dim>{}});
    result.ratios.push_back(ratio);
  }
  result.plan_identity = "pops.test.tensor-plan";
  result.operator_contract_identity =
      std::string(tensor_elliptic_detail::kScalarTensorEllipticContract);
  result.assembly_field_slots = tensor_elliptic_detail::assembly_slots<Dim>();
  result.solution_field_slot = "pops.tensor-elliptic.solution";
  result.options = tensor_elliptic_detail::default_options();
  return result;
}

template <int Dim>
pops::runtime::program::HierarchyTensorSolverBuildRequest<Dim> request(bool refined = false) {
  std::array<int, Dim> ratio_components{};
  ratio_components.fill(2);
  return request<Dim>(refined, ratio_components);
}

template <int Dim>
struct ExactIdentityTensorKernel {
  static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.test.exact-identity-tensor", 1};
  }

  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    contract.text("identity-tensor").scalar(std::int32_t{Dim});
  }

  pops::SolveReport operator()(
      const pops::runtime::program::HierarchyTensorSolveInvocation<Dim>& invocation) const {
    pops::Real reference = pops::Real(0);
    for (const auto& level : invocation.levels) {
      pops::runtime::program::tensor_elliptic_detail::copy_valid(*level.solution, *level.rhs);
      reference = std::max(reference, pops::norm_inf(*level.rhs));
    }
    pops::SolveReport report;
    report.reference_residual_norm =
        static_cast<pops::Real>(pops::all_reduce_max(static_cast<double>(reference)));
    report.residual_norm = pops::Real(0);
    report.rel_residual = pops::Real(0);
    report.mark_solved("exact identity tensor kernel");
    return report;
  }
};

struct FillTensorManufacturedRhs {
  pops::Geometry<2> geometry;
  pops::FieldView<pops::Real, 2> rhs;

  POPS_HD void operator()(const pops::Index<2>& cell) const {
    const pops::Real x = geometry.cell_coordinate(0, cell[0]);
    const pops::Real y = geometry.cell_coordinate(1, cell[1]);
    constexpr pops::Real a_xx = pops::Real(2);
    constexpr pops::Real a_xy = pops::Real(0.3);
    constexpr pops::Real a_yx = pops::Real(0.3);
    constexpr pops::Real a_yy = pops::Real(1.5);
    rhs(cell, 0) =
        pops::Real(2) * a_xx * y * (pops::Real(1) - y) +
        pops::Real(2) * a_yy * x * (pops::Real(1) - x) -
        (a_xy + a_yx) * (pops::Real(1) - pops::Real(2) * x) * (pops::Real(1) - pops::Real(2) * y);
  }
};

struct ManufacturedError {
  pops::Geometry<2> geometry;
  pops::FieldView<const pops::Real, 2> solution;
  pops::Box<2> region;

  POPS_HD void operator()(std::int64_t linear, pops::Real& error) const {
    const int x_index = static_cast<int>(linear % region.length(0)) + region.lo[0];
    const int y_index = static_cast<int>(linear / region.length(0)) + region.lo[1];
    const pops::Index<2> cell{x_index, y_index};
    const pops::Real x = geometry.cell_coordinate(0, x_index);
    const pops::Real y = geometry.cell_coordinate(1, y_index);
    const pops::Real exact = x * (pops::Real(1) - x) * y * (pops::Real(1) - y);
    error = std::max(error, Kokkos::abs(solution(cell, 0) - exact));
  }
};

pops::runtime::program::HierarchyTensorSolverBuildRequest<2> manufactured_request(
    int coarse_cells) {
  using namespace pops;
  using namespace pops::runtime::program;
  const Box<2> coarse_domain{Index<2>{0, 0}, Index<2>{coarse_cells - 1, coarse_cells - 1}};
  const Geometry<2> coarse_geometry = Geometry<2>::from_bounds(
      coarse_domain, RealVector<2>{Real(0), Real(0)}, RealVector<2>{Real(1), Real(1)});
  const mesh::BoxArray<2> coarse_layout(std::vector<Box<2>>{coarse_domain});
  const mesh::RankSpace<2> rank_space{Index<2>{0, 0}, Extent<2>{1, 1}};
  const mesh::Distribution<2> coarse_distribution =
      mesh::Distribution<2>::replicated(coarse_layout, rank_space);

  const amr::RefinementRatio<2> ratio{std::array<int, 2>{2, 2}};
  const Geometry<2> fine_geometry = coarse_geometry.refine(Extent<2>{2, 2});
  const int fine_cells = 2 * coarse_cells;
  const Box<2> fine_patch{Index<2>{fine_cells / 4, fine_cells / 4},
                          Index<2>{3 * fine_cells / 4 - 1, 3 * fine_cells / 4 - 1}};
  const mesh::BoxArray<2> fine_layout(std::vector<Box<2>>{fine_patch});
  const mesh::Distribution<2> fine_distribution = mesh::Distribution<2>::partitioned(
      fine_layout, rank_space, std::vector<Index<2>>{Index<2>{0, 0}});

  HierarchyTensorSolverBuildRequest<2> result;
  result.block = 1;
  result.components = 1;
  result.levels.push_back(
      HierarchyTensorLevelBuildRequest<2>{coarse_geometry, homogeneous_dirichlet(coarse_geometry),
                                          coarse_layout, coarse_distribution, Index<2>{0, 0}});
  result.levels.push_back(
      HierarchyTensorLevelBuildRequest<2>{fine_geometry, homogeneous_dirichlet(fine_geometry),
                                          fine_layout, fine_distribution, Index<2>{0, 0}});
  result.ratios.push_back(ratio);
  result.plan_identity = "pops.test.nd-tensor-mms";
  result.operator_contract_identity =
      std::string(tensor_elliptic_detail::kScalarTensorEllipticContract);
  result.assembly_field_slots = tensor_elliptic_detail::assembly_slots<2>();
  result.solution_field_slot = "pops.tensor-elliptic.solution";
  result.options = tensor_elliptic_detail::default_options();
  result.options.values.emplace("fac.fine_sweeps", std::int64_t{32});
  result.options.values.emplace("fac.coarse_cycles", std::int64_t{96});
  result.options.values.emplace("fac.coarse_rel_tol", 1.0e-10);
  return result;
}

pops::Real solve_manufactured_error(int coarse_cells) {
  using namespace pops;
  using namespace pops::runtime::program;
  auto build_request = manufactured_request(coarse_cells);
  const std::array<Geometry<2>, 2> geometries{build_request.levels[0].geometry,
                                              build_request.levels[1].geometry};
  const Box<2> comparison = build_request.levels[1].layout[0].grow(-std::max(2, coarse_cells / 4));
  const ExecutionLane lane = ExecutionLane::world("pops.test.nd-tensor-mms");
  const auto registry = make_default_hierarchy_tensor_solver_provider_registry<2>(lane);
  auto prepared = prepare_hierarchy_tensor_solver_collectively(
      *registry, tensor_elliptic_detail::kCompositeTensorProvider, std::move(build_request), lane);

  for (int level = 0; level < prepared->level_count(); ++level) {
    prepared->assembly_target("pops.tensor-elliptic.coefficient.0.0", level).set_val(Real(2));
    prepared->assembly_target("pops.tensor-elliptic.coefficient.0.1", level).set_val(Real(0.3));
    prepared->assembly_target("pops.tensor-elliptic.coefficient.1.0", level).set_val(Real(0.3));
    prepared->assembly_target("pops.tensor-elliptic.coefficient.1.1", level).set_val(Real(1.5));
    auto& rhs = prepared->assembly_target("pops.tensor-elliptic.rhs", level);
    for (std::size_t local = 0; local < rhs.local_size(); ++local)
      for_each_cell(rhs.box(local),
                    FillTensorManufacturedRhs{geometries[static_cast<std::size_t>(level)],
                                              rhs.fab(local).view()});
    prepared->stage_initial_guess(level, nullptr);
  }
  Kokkos::fence();

  SolveOutcome outcome = solve_prepared_hierarchy_tensor_collectively(
      *prepared, HierarchyTensorSolveControls{Real(8e-7), Real(1e-12), 60}, lane);
  const SolveReport report = outcome.consume(SolveConsumption::kAccept);
  if (!report.solved())
    throw std::runtime_error("dimension-generic tensor MMS did not converge: " + report.reason);

  const auto& fine = prepared->solution(1);
  Real error = Real(0);
  Kokkos::parallel_reduce(
      "pops_nd_tensor_mms_error", Kokkos::RangePolicy<std::int64_t>(0, comparison.numPts()),
      ManufacturedError{geometries[1], std::as_const(fine.fab(0)).view(), comparison},
      Kokkos::Max<Real>(error));
  Kokkos::fence();
  return error;
}

template <int Dim>
void expect_rank_accepted() {
  using namespace pops::runtime::program;
  CompositeTensorHierarchyProvider<Dim> provider;
  const pops::PreparedProviderSupport support = provider.supports(request<Dim>(true));
  EXPECT_TRUE(support.accepted());
}

TEST(HierarchyTensorExactRank, OneTwoAndThreeDimensionalRequestsPrepare) {
  expect_rank_accepted<1>();
  expect_rank_accepted<2>();
  expect_rank_accepted<3>();
}

TEST(HierarchyTensorExactRank, FacAcceptsRankedAndPartiallyRefinedAxes) {
  using namespace pops::runtime::program;
  EXPECT_TRUE(CompositeTensorHierarchyProvider<1>{}
                  .supports(request<1>(true, std::array<int, 1>{3}))
                  .accepted());
  EXPECT_TRUE(CompositeTensorHierarchyProvider<2>{}
                  .supports(request<2>(true, std::array<int, 2>{3, 1}))
                  .accepted());
  EXPECT_TRUE(CompositeTensorHierarchyProvider<3>{}
                  .supports(request<3>(true, std::array<int, 3>{1, 2, 3}))
                  .accepted());
}

template <int Dim>
void expect_transactional_publication() {
  using namespace pops;
  using namespace pops::runtime::program;
  PreparedHierarchyTensorKernel<Dim> kernel{ExactIdentityTensorKernel<Dim>{}};
  const ExecutionLane lane = ExecutionLane::world("pops.test.nd-tensor-transaction");
  const auto registry =
      make_hierarchy_tensor_solver_provider_registry<Dim>(std::move(kernel), lane);
  auto prepared = prepare_hierarchy_tensor_solver_collectively(
      *registry, tensor_elliptic_detail::kCompositeTensorProvider, request<Dim>(true), lane);
  EXPECT_EQ(prepared->assembly_target("pops.tensor-elliptic.flux", 0).ncomp(), Dim);
  EXPECT_EQ(prepared->assembly_target(tensor_elliptic_detail::coefficient_slot(Dim - 1, Dim - 1), 0)
                .ncomp(),
            1);
  prepared->assembly_target("pops.tensor-elliptic.rhs", 0).set_val(Real(2));
  prepared->solution(0).set_val(Real(0));
  SolveOutcome outcome = solve_prepared_hierarchy_tensor_collectively(
      *prepared, HierarchyTensorSolveControls{Real(1e-10), Real(0), 4}, lane);
  EXPECT_EQ(norm_inf(prepared->solution(0)), Real(0));
  EXPECT_TRUE(outcome.consume(SolveConsumption::kAccept).solved());
  EXPECT_EQ(norm_inf(prepared->solution(0)), Real(2));
}

TEST(HierarchyTensorExactRank, OneTwoAndThreeDimensionalPublicationIsTransactional) {
  expect_transactional_publication<1>();
  expect_transactional_publication<2>();
  expect_transactional_publication<3>();
}

template <int Dim>
void expect_direct_fac_zero_residual() {
  using namespace pops;
  using namespace pops::runtime::program;
  const pops::ExecutionLane lane = pops::ExecutionLane::world("pops.test.nd-tensor-zero-residual");
  const auto registry = make_default_hierarchy_tensor_solver_provider_registry<Dim>(lane);
  auto prepared = prepare_hierarchy_tensor_solver_collectively(
      *registry, tensor_elliptic_detail::kCompositeTensorProvider, request<Dim>(true), lane);
  for (int level = 0; level < prepared->level_count(); ++level) {
    for (int row = 0; row < Dim; ++row)
      for (int column = 0; column < Dim; ++column)
        prepared->assembly_target(tensor_elliptic_detail::coefficient_slot(row, column), level)
            .set_val(row == column ? Real(1) : Real(0));
    prepared->assembly_target("pops.tensor-elliptic.rhs", level).set_val(Real(0));
    prepared->stage_initial_guess(level, nullptr);
  }
  EXPECT_TRUE(solve_prepared_hierarchy_tensor_collectively(
                  *prepared, HierarchyTensorSolveControls{Real(1e-10), Real(0), 4}, lane)
                  .consume(SolveConsumption::kAccept)
                  .solved());
}

TEST(HierarchyTensorExactRank, DirectFacUsesTheSameOperatorInOneTwoAndThreeDimensions) {
  expect_direct_fac_zero_residual<1>();
  expect_direct_fac_zero_residual<2>();
  expect_direct_fac_zero_residual<3>();
}

TEST(HierarchyTensorExactRank, ExactRankProvidesTheBuiltinFullTensorFacKernel) {
  const pops::ExecutionLane lane = pops::ExecutionLane::world("pops.test.nd-tensor-provider");
  using namespace pops::runtime::program;
  const auto registry = make_default_hierarchy_tensor_solver_provider_registry<2>(lane);
  const auto provider = registry->resolve(tensor_elliptic_detail::kCompositeTensorProvider);
  const pops::PreparedProviderSupport support = provider->supports(request<2>(true));
  EXPECT_TRUE(support.accepted());
}

TEST(HierarchyTensorExactRank, ExactRankPublishesOnlyAfterSolveOutcomeAccept) {
  using namespace pops;
  using namespace pops::runtime::program;

  PreparedHierarchyTensorKernel<2> kernel{ExactIdentityTensorKernel<2>{}};
  const ExecutionLane lane = ExecutionLane::world("pops.test.nd-tensor-publication");
  const auto registry = make_hierarchy_tensor_solver_provider_registry<2>(std::move(kernel), lane);
  auto prepared = prepare_hierarchy_tensor_solver_collectively(
      *registry, tensor_elliptic_detail::kCompositeTensorProvider, request<2>(true), lane);

  EXPECT_EQ(prepared->level_count(), 2);
  EXPECT_EQ(prepared->assembly_target("pops.tensor-elliptic.flux", 0).ncomp(), 2);
  FieldView<Real, 2> coefficient =
      prepared->assembly_target_view("pops.tensor-elliptic.coefficient.1.0", 0, 0);
  EXPECT_EQ(coefficient.ncomp, 1);

  prepared->assembly_target("pops.tensor-elliptic.rhs", 0).set_val(Real(2));
  prepared->solution(0).set_val(Real(0));
  SolveOutcome outcome = solve_prepared_hierarchy_tensor_collectively(
      *prepared, HierarchyTensorSolveControls{Real(1e-10), Real(0), 4}, lane);

  EXPECT_EQ(norm_inf(prepared->solution(0)), Real(0));
  const SolveReport accepted = outcome.consume(SolveConsumption::kAccept);
  EXPECT_TRUE(accepted.solved());
  EXPECT_EQ(norm_inf(prepared->solution(0)), Real(2));
}

TEST(HierarchyTensorExactRank, BuiltinExactRankFacSolvesARealCrossTensorMms) {
  const pops::Real coarse_error = solve_manufactured_error(8);
  const pops::Real fine_error = solve_manufactured_error(16);
  ASSERT_GT(coarse_error, pops::Real(0));
  ASSERT_GT(fine_error, pops::Real(0));
  EXPECT_LT(fine_error, coarse_error);
  const pops::Real observed_order = std::log(coarse_error / fine_error) / std::log(pops::Real(2));
  EXPECT_GE(observed_order, pops::Real(1.5))
      << "coarse_error=" << coarse_error << " fine_error=" << fine_error;
}

TEST(HierarchyTensorExactRank, ContractsAuthenticateRankGeometryAndOwnership) {
  using namespace pops::runtime::program;
  const auto one = request<1>();
  const auto two = request<2>();
  const auto three = request<3>();
  const std::string one_contract = hierarchy_tensor_detail::request_contract(one);
  const std::string two_contract = hierarchy_tensor_detail::request_contract(two);
  const std::string three_contract = hierarchy_tensor_detail::request_contract(three);
  EXPECT_NE(one_contract, two_contract);
  EXPECT_NE(two_contract, three_contract);
  EXPECT_NE(one_contract, three_contract);
}

}  // namespace
