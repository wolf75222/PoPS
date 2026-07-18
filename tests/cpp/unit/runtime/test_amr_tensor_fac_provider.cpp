#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <pops/runtime/amr/amr_tensor_elliptic.hpp>
#include <pops/runtime/numerical_defaults.hpp>
#include <pops/runtime/program/amr_program_context.hpp>

namespace {

using pops::Real;
using pops::PreparedProviderOptions;
using pops::PreparedProviderSupport;
using pops::runtime::program::AmrProgramContext;
using pops::runtime::program::AmrTensorElliptic;
using pops::runtime::program::HierarchyTensorSolverBuildRequest;
using pops::runtime::program::HierarchyTensorSolverExecutionPath;
using pops::runtime::program::HierarchyTensorSolverProvider;
using pops::runtime::program::PreparedHierarchyTensorSolver;

class DistinctHierarchyPrepared final : public PreparedHierarchyTensorSolver {
 public:
  explicit DistinctHierarchyPrepared(std::string contract) : contract_(std::move(contract)) {}
  std::string_view provider_identity() const noexcept override {
    return "pops.test.hierarchy.distinct";
  }
  std::uint64_t provider_version() const noexcept override { return 1; }
  std::string_view exact_prepared_contract() const noexcept override { return contract_; }
  HierarchyTensorSolverExecutionPath execution_path() const noexcept override {
    return HierarchyTensorSolverExecutionPath::PreparedKrylovFallback;
  }
  pops::MultiFab& assembly_target(std::string_view, int) override {
    throw std::logic_error("test provider has no materialized build request");
  }
  pops::MultiFab& solution(int) override {
    throw std::logic_error("test provider has no materialized build request");
  }
  void stage_initial_guess(int, const pops::MultiFab*) override {
    throw std::logic_error("test provider has no materialized build request");
  }
  pops::SolveReport solve(
      const pops::runtime::program::HierarchyTensorSolveControls&) override {
    return pops::SolveReport::capability_failure();
  }

 private:
  std::string contract_;
};

class DistinctHierarchyProvider final : public HierarchyTensorSolverProvider {
 public:
  explicit DistinctHierarchyProvider(std::string collective_contract =
                                         "pops.test.hierarchy.distinct@1",
                                     std::vector<std::string> capabilities = {
                                         "pops.test.hierarchy.distinct.flat-krylov@1"})
      : collective_contract_(std::move(collective_contract)),
        capabilities_(std::move(capabilities)) {}
  std::string_view identity() const noexcept override { return "pops.test.hierarchy.distinct"; }
  std::uint64_t interface_version() const noexcept override { return 1; }
  std::string_view collective_contract() const noexcept override { return collective_contract_; }
  std::vector<std::string> capability_contracts() const override { return capabilities_; }
  PreparedProviderOptions default_options() const override {
    return {"pops.test.hierarchy.distinct.options@1", {}};
  }
  PreparedProviderSupport accepts_options(
      const PreparedProviderOptions& options) const noexcept override {
    return options.schema_identity == "pops.test.hierarchy.distinct.options@1" &&
                   options.values.empty()
               ? PreparedProviderSupport::accept()
               : PreparedProviderSupport::reject(1, "distinct provider options are invalid");
  }
  PreparedProviderSupport supports(
      const HierarchyTensorSolverBuildRequest& request) const noexcept override {
    return request.levels == 1 && accepts_options(request.options).accepted()
               ? PreparedProviderSupport::accept()
               : PreparedProviderSupport::reject(2, "distinct provider requires one level");
  }
  PreparedProviderSupport accepts_execution(
      const HierarchyTensorSolverBuildRequest& request,
      HierarchyTensorSolverExecutionPath execution) const noexcept override {
    return supports(request).accepted() &&
                   execution == HierarchyTensorSolverExecutionPath::PreparedKrylovFallback
               ? PreparedProviderSupport::accept()
               : PreparedProviderSupport::reject(3, "distinct provider execution is invalid");
  }
  std::string expected_prepared_contract(
      const HierarchyTensorSolverBuildRequest&) const override {
    return "pops.test.hierarchy.distinct.prepared@1";
  }
  std::unique_ptr<PreparedHierarchyTensorSolver> prepare(
      const HierarchyTensorSolverBuildRequest& request) const override {
    if (!supports(request).accepted())
      throw std::invalid_argument("unsupported distinct hierarchy request");
    return std::make_unique<DistinctHierarchyPrepared>(expected_prepared_contract(request));
  }

 private:
  std::string collective_contract_;
  std::vector<std::string> capabilities_;
};

class ReportOnlyHierarchyPrepared final : public PreparedHierarchyTensorSolver {
 public:
  explicit ReportOnlyHierarchyPrepared(pops::SolveReport report) : report_(std::move(report)) {}
  std::string_view provider_identity() const noexcept override {
    return "pops.test.hierarchy.report-only";
  }
  std::uint64_t provider_version() const noexcept override { return 1; }
  std::string_view exact_prepared_contract() const noexcept override {
    return "pops.test.hierarchy.report-only.prepared@1";
  }
  HierarchyTensorSolverExecutionPath execution_path() const noexcept override {
    return HierarchyTensorSolverExecutionPath::DirectProvider;
  }
  pops::MultiFab& assembly_target(std::string_view, int) override {
    throw std::logic_error("report-only provider has no field storage");
  }
  pops::MultiFab& solution(int) override {
    throw std::logic_error("report-only provider has no field storage");
  }
  void stage_initial_guess(int, const pops::MultiFab*) override {}
  pops::SolveReport solve(
      const pops::runtime::program::HierarchyTensorSolveControls&) override {
    return report_;
  }

 private:
  pops::SolveReport report_;
};

class FlatIdentityPrepared final : public PreparedHierarchyTensorSolver {
 public:
  explicit FlatIdentityPrepared(std::string contract)
      : contract_(std::move(contract)),
        boxes_(std::vector<pops::Box2D>{pops::Box2D::from_extents(8, 8)}),
        mapping_(boxes_.size(), pops::n_ranks()),
        rhs_(boxes_, mapping_, 1, 0),
        solution_(boxes_, mapping_, 1, 0) {
    rhs_.set_val(Real(0));
    solution_.set_val(Real(0));
  }
  std::string_view provider_identity() const noexcept override {
    return "pops.test.hierarchy.flat-identity";
  }
  std::uint64_t provider_version() const noexcept override { return 1; }
  std::string_view exact_prepared_contract() const noexcept override { return contract_; }
  HierarchyTensorSolverExecutionPath execution_path() const noexcept override {
    return HierarchyTensorSolverExecutionPath::DirectProvider;
  }
  pops::MultiFab& assembly_target(std::string_view slot, int level) override {
    if (slot != "pops.test.identity.rhs" || level != 0)
      throw std::invalid_argument("flat identity provider received an unknown field slot");
    return rhs_;
  }
  pops::MultiFab& solution(int level) override {
    if (level != 0)
      throw std::out_of_range("flat identity provider has exactly one level");
    return solution_;
  }
  void stage_initial_guess(int level, const pops::MultiFab*) override {
    if (level != 0)
      throw std::out_of_range("flat identity provider has exactly one level");
  }
  pops::SolveReport solve(
      const pops::runtime::program::HierarchyTensorSolveControls&) override {
    pops::parallel_copy(solution_, rhs_);  // Exact inverse of the authenticated identity operator.
    pops::SolveReport report;
    report.reference_residual_norm = pops::norm_inf(rhs_);
    report.residual_norm = Real(0);
    report.rel_residual = Real(0);
    report.mark_solved("exact identity solve");
    return report;
  }

 private:
  std::string contract_;
  pops::BoxArray boxes_;
  pops::DistributionMapping mapping_;
  pops::MultiFab rhs_;
  pops::MultiFab solution_;
};

class FlatIdentityProvider final : public HierarchyTensorSolverProvider {
 public:
  std::string_view identity() const noexcept override {
    return "pops.test.hierarchy.flat-identity";
  }
  std::uint64_t interface_version() const noexcept override { return 1; }
  std::string_view collective_contract() const noexcept override {
    return "pops.test.hierarchy.flat-identity@1";
  }
  std::vector<std::string> capability_contracts() const override {
    return {"pops.test.hierarchy.flat-identity.direct@1",
            "pops.test.hierarchy.flat-identity.distributed@1"};
  }
  PreparedProviderOptions default_options() const override {
    return {"pops.test.hierarchy.flat-identity.options@1", {}};
  }
  PreparedProviderSupport accepts_options(
      const PreparedProviderOptions& options) const noexcept override {
    return options.schema_identity == "pops.test.hierarchy.flat-identity.options@1" &&
                   options.values.empty()
               ? PreparedProviderSupport::accept()
               : PreparedProviderSupport::reject(1, "flat identity options are invalid");
  }
  PreparedProviderSupport supports(
      const HierarchyTensorSolverBuildRequest& request) const noexcept override {
    const bool accepted =
        request.levels == 1 && request.level_populated == std::vector<bool>{true} &&
        request.level_distributions ==
            std::vector<pops::FieldDistribution>{pops::FieldDistribution::Distributed} &&
        request.components == 1 &&
        request.operator_contract_identity == "pops.test.operator.identity@1" &&
        request.assembly_field_slots ==
            std::vector<std::string>{"pops.test.identity.rhs"} &&
        request.solution_field_slot == "pops.test.identity.solution" &&
        accepts_options(request.options).accepted();
    return accepted ? PreparedProviderSupport::accept()
                    : PreparedProviderSupport::reject(2, "flat identity request is invalid");
  }
  PreparedProviderSupport accepts_execution(
      const HierarchyTensorSolverBuildRequest& request,
      HierarchyTensorSolverExecutionPath execution) const noexcept override {
    return supports(request).accepted() &&
                   execution == HierarchyTensorSolverExecutionPath::DirectProvider
               ? PreparedProviderSupport::accept()
               : PreparedProviderSupport::reject(3, "flat identity execution is invalid");
  }
  std::string expected_prepared_contract(
      const HierarchyTensorSolverBuildRequest& request) const override {
    pops::ExactContractBuilder contract;
    contract.text("pops.test.hierarchy.flat-identity.prepared")
        .scalar(std::uint32_t{1})
        .text(request.plan_identity)
        .text(request.operator_contract_identity)
        .sequence(request.assembly_field_slots,
                  [](pops::ExactContractBuilder& item, const std::string& slot) {
                    item.text(slot);
                  })
        .text(request.solution_field_slot)
        .sequence(request.level_populated,
                  [](pops::ExactContractBuilder& item, bool populated) {
                    item.scalar(populated);
                  })
        .sequence(request.level_distributions,
                  [](pops::ExactContractBuilder& item,
                     pops::FieldDistribution distribution) { item.scalar(distribution); })
        .bytes(request.options.exact_contract());
    return std::move(contract).release();
  }
  std::unique_ptr<PreparedHierarchyTensorSolver> prepare(
      const HierarchyTensorSolverBuildRequest& request) const override {
    if (!supports(request).accepted())
      throw std::invalid_argument("flat identity provider rejected the request");
    return std::make_unique<FlatIdentityPrepared>(expected_prepared_contract(request));
  }
};

using ConfigureSolver = void (AmrProgramContext::*)(int, int, const std::string&,
                                                    const std::string&,
                                                    const std::string&,
                                                    const std::vector<std::string>&,
                                                    const std::string&,
                                                    const PreparedProviderOptions&) const;
static_assert(
    std::is_same_v<decltype(&AmrProgramContext::configure_hierarchy_tensor_solver),
                   ConfigureSolver>);

TEST(HierarchyTensorSolverProviderContract, BuiltinAdvertisesOnlyItsStructuralEnvelope) {
  const auto registry =
      pops::runtime::program::make_default_hierarchy_tensor_solver_provider_registry();
  const auto provider = registry->resolve("pops.hierarchy.composite-tensor-fac");
  const std::vector<std::string> capabilities = provider->capability_contracts();
  EXPECT_NE(std::find(capabilities.begin(), capabilities.end(),
                      "pops.hierarchy.composite-tensor-fac.flat-krylov@1"),
            capabilities.end());
  EXPECT_NE(std::find(capabilities.begin(), capabilities.end(),
                      "pops.hierarchy.composite-tensor-fac.refined-direct@1"),
            capabilities.end());
  EXPECT_EQ(provider->default_options().schema_identity,
            "pops.hierarchy.composite-tensor-fac.options@1");
}

TEST(HierarchyTensorSolverProviderContract,
     DistinctCompiledComponentDeclarationIsCollectiveAndAppendOnly) {
  pops::runtime::program::HierarchyTensorSolverProviderRegistry registry;
  registry.add_collectively(std::make_shared<DistinctHierarchyProvider>());
  registry.add_collectively(std::make_shared<DistinctHierarchyProvider>());
  EXPECT_EQ(registry.resolve("pops.test.hierarchy.distinct")->capability_contracts(),
            DistinctHierarchyProvider().capability_contracts());
  EXPECT_THROW(registry.add_collectively(std::make_shared<DistinctHierarchyProvider>(
                   "pops.test.hierarchy.distinct.conflict@1")),
               std::invalid_argument);
}

TEST(HierarchyTensorSolverProviderContract,
     CapabilityDeclarationOrderDoesNotChangeTheExactProviderContract) {
  const DistinctHierarchyProvider first(
      "pops.test.hierarchy.distinct@1",
      {"pops.test.hierarchy.distinct.zeta@1", "pops.test.hierarchy.distinct.alpha@1"});
  const DistinctHierarchyProvider second(
      "pops.test.hierarchy.distinct@1",
      {"pops.test.hierarchy.distinct.alpha@1", "pops.test.hierarchy.distinct.zeta@1"});
  EXPECT_EQ(pops::runtime::program::exact_hierarchy_tensor_solver_provider_declaration(first),
            pops::runtime::program::exact_hierarchy_tensor_solver_provider_declaration(second));
}

TEST(HierarchyTensorSolverProviderContract,
     CapabilityDeclarationsAreOptionalWhenSupportsOwnsCompatibility) {
  pops::runtime::program::HierarchyTensorSolverProviderRegistry registry;
  EXPECT_NO_THROW(registry.add(std::make_shared<DistinctHierarchyProvider>(
      "pops.test.hierarchy.distinct@1", std::vector<std::string>{})));
  EXPECT_TRUE(registry.resolve("pops.test.hierarchy.distinct")
                  ->capability_contracts()
                  .empty());
}

TEST(HierarchyTensorSolverProviderContract,
     InvalidDeclarationIsRejectedBeforeRegistryMutation) {
  pops::runtime::program::HierarchyTensorSolverProviderRegistry registry;
  const std::vector<std::string> duplicate_capabilities = {
      "pops.test.hierarchy.distinct.duplicate@1",
      "pops.test.hierarchy.distinct.duplicate@1"};
  EXPECT_THROW(registry.add(std::make_shared<DistinctHierarchyProvider>(
                   "pops.test.hierarchy.distinct@1", duplicate_capabilities)),
               std::invalid_argument);

  // The rejected declaration must not reserve the provider identity.
  EXPECT_NO_THROW(registry.add(std::make_shared<DistinctHierarchyProvider>()));
  EXPECT_EQ(registry.resolve("pops.test.hierarchy.distinct")->identity(),
            "pops.test.hierarchy.distinct");
}

TEST(HierarchyTensorSolverProviderContract,
     ProviderOwnedSupportCodeAndReasonSurviveTheCollectiveBoundary) {
  pops::runtime::program::HierarchyTensorSolverProviderRegistry registry;
  registry.add(std::make_shared<DistinctHierarchyProvider>());
  HierarchyTensorSolverBuildRequest request;
  request.levels = 2;
  request.options = DistinctHierarchyProvider().default_options();

  try {
    (void)pops::runtime::program::prepare_hierarchy_tensor_solver_collectively(
        registry, "pops.test.hierarchy.distinct", request);
    FAIL() << "the one-level provider accepted a two-level request";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("code 2"), std::string::npos);
    EXPECT_NE(std::string(error.what()).find("distinct provider requires one level"),
              std::string::npos);
  }
}

TEST(HierarchyTensorSolverProviderContract,
     FlatDirectProviderOwnsStorageSolveAndPublicationWithoutKrylov) {
  pops::runtime::program::HierarchyTensorSolverProviderRegistry registry;
  registry.add(std::make_shared<FlatIdentityProvider>());
  HierarchyTensorSolverBuildRequest request;
  request.components = 1;
  request.levels = 1;
  request.level_populated = {true};
  request.level_distributions = {pops::FieldDistribution::Distributed};
  request.plan_identity = "pops.test.flat-direct-plan@1";
  request.operator_contract_identity = "pops.test.operator.identity@1";
  request.assembly_field_slots = {"pops.test.identity.rhs"};
  request.solution_field_slot = "pops.test.identity.solution";
  request.options = FlatIdentityProvider().default_options();

  auto prepared = pops::runtime::program::prepare_hierarchy_tensor_solver_collectively(
      registry, "pops.test.hierarchy.flat-identity", request);
  ASSERT_EQ(prepared->execution_path(), HierarchyTensorSolverExecutionPath::DirectProvider);
  prepared->assembly_target("pops.test.identity.rhs", 0).set_val(Real(3.25));
  const pops::SolveReport report = prepared->solve({Real(1.0e-12), Real(0), 1});
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_EQ(pops::norm_inf(prepared->solution(0)), Real(3.25));
}

TEST(HierarchyTensorSolverProviderContract,
     MalformedThirdPartyReportsAreRejectedAtTheCollectivePublicationBoundary) {
  const pops::runtime::program::HierarchyTensorSolveControls controls{Real(1.0e-8), Real(0), 4};

  pops::SolveReport invalid_status;
  invalid_status.status = static_cast<pops::SolveStatus>(999);
  invalid_status.action = pops::SolveAction::kFailRun;
  invalid_status.reason = "unknown provider status";
  ReportOnlyHierarchyPrepared invalid_status_solver(invalid_status);
  EXPECT_THROW(
      (void)pops::runtime::program::solve_prepared_hierarchy_tensor_collectively(
          invalid_status_solver, controls),
      std::runtime_error);

  pops::SolveReport invalid_action;
  invalid_action.status = pops::SolveStatus::kBreakdown;
  invalid_action.action = static_cast<pops::SolveAction>(999);
  invalid_action.reason = "unknown provider action";
  ReportOnlyHierarchyPrepared invalid_action_solver(invalid_action);
  EXPECT_THROW(
      (void)pops::runtime::program::solve_prepared_hierarchy_tensor_collectively(
          invalid_action_solver, controls),
      std::runtime_error);

  pops::SolveReport nonfinite;
  nonfinite.mark_solved();
  nonfinite.residual_norm = std::numeric_limits<Real>::quiet_NaN();
  ReportOnlyHierarchyPrepared nonfinite_solver(nonfinite);
  EXPECT_THROW(
      (void)pops::runtime::program::solve_prepared_hierarchy_tensor_collectively(
          nonfinite_solver, controls),
      std::runtime_error);

  pops::SolveReport impossible_iterations;
  impossible_iterations.mark_solved();
  impossible_iterations.iters = controls.maximum_iterations + 1;
  ReportOnlyHierarchyPrepared impossible_iterations_solver(impossible_iterations);
  EXPECT_THROW(
      (void)pops::runtime::program::solve_prepared_hierarchy_tensor_collectively(
          impossible_iterations_solver, controls),
      std::runtime_error);
}

TEST(HierarchyTensorSolverProviderContract,
     ProgramContextRegistersADistinctProviderThroughTheRuntimeFacade) {
  pops::AmrSystem system(pops::AmrSystemConfig{16});
  AmrProgramContext context(nullptr, &system);
  context.register_hierarchy_tensor_solver_provider(
      std::make_shared<DistinctHierarchyProvider>());
  EXPECT_EQ(system.hierarchy_tensor_solver_provider_registry()
                ->resolve("pops.test.hierarchy.distinct")
                ->collective_contract(),
            "pops.test.hierarchy.distinct@1");
}

TEST(AmrProgramContextContract, AnonymousRateIdentityIsRejectedBeforeTopologyLookup) {
  AmrProgramContext context(nullptr, nullptr);
  EXPECT_THROW((void)context.boundary_evaluation_point(-1), std::invalid_argument);
}

TEST(AmrTensorFacSolver, OmittedFacControlsResolveFromNativeOptionsOnly) {
  AmrTensorElliptic driver(nullptr, 0, 1);
  EXPECT_THROW(driver.composite_fac_options(Real(1.0e-8), Real(0), 23), std::logic_error);

  driver.configure_composite_tensor_fac(0, Real(0), Real(0), 0, -1);
  const pops::CompositeFacOptions options =
      driver.composite_fac_options(Real(3.0e-8), Real(2.0e-12), 23);

  EXPECT_EQ(options.max_iters, 23);
  EXPECT_EQ(options.rel_tol, Real(3.0e-8));
  EXPECT_EQ(options.abs_tol, Real(2.0e-12));
  EXPECT_EQ(options.fine_sweeps, pops::kFACDefaultFineSweeps);
  EXPECT_EQ(options.coarse_rel_tol, pops::kFACInitialCoarseRelTol);
  EXPECT_EQ(options.coarse_abs_tol, pops::kFACInitialCoarseAbsTol);
  EXPECT_EQ(options.coarse_cycles, pops::kFACInitialCoarseMaxCycles);
  EXPECT_FALSE(options.verbose);
}

TEST(AmrTensorFacSolver, ExplicitFacControlsJoinDirectSolverControls) {
  AmrTensorElliptic driver(nullptr, 0, 1);
  driver.configure_composite_tensor_fac(7, Real(2.0e-7), Real(4.0e-14), 9, 1);
  pops::CompositeFacOptions options = driver.composite_fac_options(Real(4.0e-8), Real(3.0e-13), 17);

  EXPECT_EQ(options.max_iters, 17);
  EXPECT_EQ(options.rel_tol, Real(4.0e-8));
  EXPECT_EQ(options.abs_tol, Real(3.0e-13));
  EXPECT_EQ(options.fine_sweeps, 7);
  EXPECT_EQ(options.coarse_rel_tol, Real(2.0e-7));
  EXPECT_EQ(options.coarse_abs_tol, Real(4.0e-14));
  EXPECT_EQ(options.coarse_cycles, 9);
  EXPECT_TRUE(options.verbose);

  driver.configure_composite_tensor_fac(8, Real(3.0e-7), Real(5.0e-14), 10, 0);
  options = driver.composite_fac_options(Real(5.0e-8), Real(4.0e-13), 19);
  EXPECT_EQ(options.max_iters, 19);
  EXPECT_EQ(options.rel_tol, Real(5.0e-8));
  EXPECT_EQ(options.abs_tol, Real(4.0e-13));
  EXPECT_EQ(options.fine_sweeps, 8);
  EXPECT_EQ(options.coarse_rel_tol, Real(3.0e-7));
  EXPECT_EQ(options.coarse_abs_tol, Real(5.0e-14));
  EXPECT_EQ(options.coarse_cycles, 10);
  EXPECT_FALSE(options.verbose);
}

TEST(AmrTensorFacSolver, WireAndDirectSolverControlsAreStrictlyValidated) {
  AmrTensorElliptic driver(nullptr, 0, 1);
  EXPECT_THROW(driver.configure_composite_tensor_fac(-1, Real(0), Real(0), 0, -1),
               std::invalid_argument);
  EXPECT_THROW(driver.configure_composite_tensor_fac(0, Real(-1.0e-7), Real(0), 0, -1),
               std::invalid_argument);
  EXPECT_THROW(driver.configure_composite_tensor_fac(0, Real(1), Real(0), 0, -1),
               std::invalid_argument);
  EXPECT_THROW(driver.configure_composite_tensor_fac(0, std::numeric_limits<Real>::quiet_NaN(),
                                                     Real(0), 0, -1),
               std::invalid_argument);
  EXPECT_THROW(driver.configure_composite_tensor_fac(0, Real(0), Real(0), -1, -1),
               std::invalid_argument);
  EXPECT_THROW(driver.configure_composite_tensor_fac(0, Real(0), Real(0), 0, -2),
               std::invalid_argument);
  EXPECT_THROW(driver.configure_composite_tensor_fac(0, Real(0), Real(0), 0, 2),
               std::invalid_argument);

  EXPECT_THROW(driver.configure_composite_tensor_fac(0, Real(0), Real(-1), 0, -1),
               std::invalid_argument);
  driver.configure_composite_tensor_fac(0, Real(0), Real(0), 0, -1);
  EXPECT_THROW(driver.composite_fac_options(Real(0), Real(0), 1), std::invalid_argument);
  EXPECT_THROW(driver.composite_fac_options(std::numeric_limits<Real>::quiet_NaN(), Real(0), 1),
               std::invalid_argument);
  EXPECT_THROW(driver.composite_fac_options(Real(1.0e-8), Real(-1), 1), std::invalid_argument);
  EXPECT_THROW(
      driver.composite_fac_options(Real(1.0e-8), std::numeric_limits<Real>::quiet_NaN(), 1),
      std::invalid_argument);
  EXPECT_THROW(driver.composite_fac_options(Real(1.0e-8), Real(0), 0), std::invalid_argument);
}

TEST(AmrTensorFacSolver, NonScalarOperatorIsRejectedAtTheNativeBoundary) {
  EXPECT_THROW((AmrTensorElliptic(nullptr, 0, 2)), std::invalid_argument);
}

}  // namespace
