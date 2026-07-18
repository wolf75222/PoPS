// ADC-578: the typed System runtime registries + lifecycle state machine extracted out of the
// System::Impl god-object. This suite pins two acceptance properties WITHOUT building a full System:
//
//   1. SystemLifecycle -- the typed freeze state machine that replaced `bool bound_`. It PRESERVES the
//      historical observable strings ("assembling"/"bound"/"running") for every pre-existing call
//      sequence, and the NEW checkpointed / finalized states + refusals are reachable only through the
//      new explicit transitions (no current caller). The inverted refusals are argued here.
//   2. The structured reports (options_report / layout_report / newton_report_ptr) the
//      registries expose for a runtime report -- the ADC-578 "define ownerships + structured reports"
//      acceptance.
#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <pops/runtime/system/system_field_solver.hpp>
#include <pops/runtime/system/system_lifecycle.hpp>
#include <pops/runtime/system/system_diagnostics_registry.hpp>
#include <pops/runtime/system/system_coupling_registry.hpp>
#include <pops/runtime/system/system_domain.hpp>

namespace {

using pops::runtime::system::LifecyclePhase;
using pops::runtime::system::SystemLifecycle;

struct EllipticRegistryOwnerProbe;
using EllipticRegistryHarness = pops::field_solver::SystemFieldSolver<EllipticRegistryOwnerProbe>;

class ProbeEllipticBackend final : public EllipticRegistryHarness::NamedFieldBackend {
 public:
  ProbeEllipticBackend(
      const EllipticRegistryHarness::EllipticBackendBuildRequest& request,
      std::string identity, std::string contract, bool wrong_rhs_layout)
      : geometry_(request.elliptic.geometry),
        distribution_(request.elliptic.distribution),
        rhs_(request.elliptic.boxes, request.elliptic.mapping, 1,
             request.elliptic.rhs_ghosts + (wrong_rhs_layout ? 1 : 0)),
        phi_(request.elliptic.boxes, request.elliptic.mapping, 1,
             request.elliptic.phi_ghosts),
        identity_(std::move(identity)),
        contract_(std::move(contract)) {}

  [[nodiscard]] std::string_view provider_identity() const noexcept override {
    return identity_;
  }
  [[nodiscard]] std::uint64_t provider_version() const noexcept override { return 1; }
  [[nodiscard]] std::string_view provider_contract() const noexcept override {
    return contract_;
  }
  pops::MultiFab& rhs() override { return rhs_; }
  pops::MultiFab& phi() override { return phi_; }
  [[nodiscard]] const pops::Geometry& geometry() const noexcept override { return geometry_; }
  [[nodiscard]] pops::FieldDistribution field_distribution() const noexcept override {
    return distribution_;
  }
  [[nodiscard]] pops::MultiFab snapshot() override { return pops::MultiFab(phi_); }
  void restore(const pops::MultiFab& value) override { phi_ = value; }
  void configure_boundary(EllipticRegistryHarness::FieldSolveConfig&) override {}
  void prepare_rhs(EllipticRegistryHarness&, pops::MultiFab&,
                   const EllipticRegistryHarness::FieldSolveConfig&,
                   pops::FieldNullspaceWorkspace&) override {}
  pops::SolveReport solve(EllipticRegistryHarness&) override {
    pops::SolveReport report;
    report.mark_solved();
    return report;
  }
  void finalize(EllipticRegistryHarness&,
                const EllipticRegistryHarness::FieldSolveConfig&,
                pops::FieldNullspaceWorkspace&) override {}
  void reset_diagnostics() override {}
  [[nodiscard]] pops::RuntimeDiagnosticsReport diagnostics_report() const override {
    return pops::make_runtime_diagnostics_report("pops.test.elliptic-registry-backend");
  }
  [[nodiscard]] pops::runtime::system::EllipticBackendMetrics metrics() const noexcept override {
    return {};
  }
  [[nodiscard]] const pops::EllipticOperatorContract* operator_contract()
      const noexcept override {
    return nullptr;
  }
  [[nodiscard]] std::vector<pops::runtime::field::FieldTopologyReportRow> topology_report()
      const override {
    return {};
  }

 private:
  pops::Geometry geometry_;
  pops::FieldDistribution distribution_;
  pops::MultiFab rhs_;
  pops::MultiFab phi_;
  std::string identity_;
  std::string contract_;
};

class ProbeEllipticProvider final : public EllipticRegistryHarness::EllipticBackendProvider {
 public:
  ProbeEllipticProvider(std::vector<std::string> capabilities = {},
                        bool wrong_rhs_layout = false)
      : capabilities_(std::move(capabilities)), wrong_rhs_layout_(wrong_rhs_layout) {}

  [[nodiscard]] std::string_view identity() const noexcept override {
    return "pops.test.system-elliptic-provider";
  }
  [[nodiscard]] std::uint64_t interface_version() const noexcept override { return 1; }
  [[nodiscard]] std::string_view collective_contract() const noexcept override {
    return "pops.test.system-elliptic-provider.contract@1";
  }
  [[nodiscard]] std::vector<std::string> capability_contracts() const override {
    return capabilities_;
  }
  [[nodiscard]] pops::PreparedProviderSupport supports(
      const EllipticRegistryHarness::EllipticBackendBuildRequest& request)
      const noexcept override {
    if (std::holds_alternative<
            EllipticRegistryHarness::ScalarDiffusionCoefficient<pops::MultiFab>>(
            request.diffusion_coefficient))
      return pops::PreparedProviderSupport::reject(
          73, "probe provider rejects scalar coefficient fields");
    return pops::PreparedProviderSupport::accept();
  }
  void write_effective_options(pops::EffectivePoissonOptions&) const override {}
  [[nodiscard]] std::unique_ptr<EllipticRegistryHarness::EllipticBackendProvider> configured(
      std::string, const pops::PreparedProviderOptions&) const override {
    return std::make_unique<ProbeEllipticProvider>(capabilities_, wrong_rhs_layout_);
  }
  [[nodiscard]] std::optional<pops::EllipticOperatorContract> expected_operator_contract(
      const pops::EllipticBuildRequest&) const override {
    return std::nullopt;
  }
  [[nodiscard]] std::unique_ptr<EllipticRegistryHarness::NamedFieldBackend> prepare(
      EllipticRegistryHarness::EllipticBackendBuildRequest request) const override {
    return std::make_unique<ProbeEllipticBackend>(
        request, std::string(identity()), std::string(collective_contract()), wrong_rhs_layout_);
  }

 private:
  std::vector<std::string> capabilities_;
  bool wrong_rhs_layout_ = false;
};

EllipticRegistryHarness::EllipticBackendBuildRequest elliptic_registry_request() {
  const pops::Box2D domain = pops::Box2D::from_extents(8, 8);
  EllipticRegistryHarness::EllipticBackendBuildRequest request;
  request.elliptic = {pops::Geometry{domain, 0.0, 1.0, 0.0, 1.0},
                      pops::BoxArray(std::vector<pops::Box2D>{domain}),
                      pops::DistributionMapping(std::vector<int>{0}),
                      pops::BCRec{},
                      {},
                      pops::FieldDistribution::Distributed,
                      0,
                      1};
  request.require_exact_operator_contract = false;
  request.exact_configuration_contract = "pops.test.system-elliptic-configuration@1";
  return request;
}

// --- 1. Lifecycle: the historical three strings are preserved bit-for-bit ------------------------
TEST(SystemLifecycle, PreservesHistoricalObservableStrings) {
  SystemLifecycle lc;
  EXPECT_FALSE(lc.frozen());
  EXPECT_EQ(lc.state(0), "assembling");
  EXPECT_EQ(lc.state(5), "assembling") << "assembling ignores the macro-step counter";

  lc.to_bound();
  EXPECT_TRUE(lc.frozen());
  EXPECT_EQ(lc.state(0), "bound") << "bound with no macro-step advanced";
  EXPECT_EQ(lc.state(1), "running") << "running is DERIVED from the macro-step > 0";
  EXPECT_EQ(lc.state(42), "running");
}

TEST(SystemLifecycle, DoubleBindThrowsTheSameMessage) {
  SystemLifecycle lc;
  lc.to_bound();
  // Same message text the old `bool bound_` guard raised in System::mark_bound.
  EXPECT_THROW(
      {
        try {
          lc.to_bound();
        } catch (const std::runtime_error& e) {
          EXPECT_NE(std::string(e.what()).find("already bound"), std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

// --- 1b. NEW states: reachable only through the new transitions -----------------------------------
TEST(SystemLifecycle, CheckpointedIsInformationalAndReversible) {
  SystemLifecycle lc;
  lc.to_bound();
  lc.to_checkpointed();
  EXPECT_EQ(lc.state(0), "checkpointed") << "checkpointed surfaces ONLY after the explicit mark";
  EXPECT_TRUE(lc.frozen()) << "checkpointed is still frozen for structural setters";
}

TEST(SystemLifecycle, FinalizedIsTerminalAndSupersetOfBoundForRefusals) {
  SystemLifecycle lc;
  lc.to_bound();
  lc.to_finalized();
  EXPECT_EQ(lc.state(0), "finalized");
  EXPECT_EQ(lc.state(9), "finalized") << "finalized ignores the macro-step counter (terminal)";
  EXPECT_TRUE(lc.frozen()) << "a structural setter after finalize is refused (superset of bound)";
}

TEST(SystemLifecycle, InvertedRefusals) {
  // (a) finalize before bind is refused.
  {
    SystemLifecycle lc;
    EXPECT_THROW(lc.to_finalized(), std::runtime_error);
  }
  // (b) double-finalize is refused.
  {
    SystemLifecycle lc;
    lc.to_bound();
    lc.to_finalized();
    EXPECT_THROW(lc.to_finalized(), std::runtime_error);
  }
  // (c) to_bound after finalize is refused (the terminal state cannot re-bind).
  {
    SystemLifecycle lc;
    lc.to_bound();
    lc.to_finalized();
    EXPECT_THROW(lc.to_bound(), std::runtime_error);
  }
  // (d) checkpoint before bind is refused; checkpoint after finalize is refused.
  {
    SystemLifecycle lc;
    EXPECT_THROW(lc.to_checkpointed(), std::runtime_error);
    lc.to_bound();
    lc.to_finalized();
    EXPECT_THROW(lc.to_checkpointed(), std::runtime_error);
  }
}

TEST(SystemLifecycle, NewStatesNeverSurfaceWithoutTheExplicitTransition) {
  // The historical sequence (bind then step) NEVER yields checkpointed / finalized: those are
  // reachable only via to_checkpointed / to_finalized, which have no current caller -> bit-identity.
  SystemLifecycle lc;
  lc.to_bound();
  for (int step = 0; step <= 3; ++step) {
    const std::string s = lc.state(step);
    EXPECT_TRUE(s == "bound" || s == "running")
        << "unexpected state for step " << step << ": " << s;
  }
}

// --- 2. Registry structured reports --------------------------------------------------------------
TEST(SystemDiagnosticsRegistry, OptionsAndNewtonReportsRoundTrip) {
  pops::runtime::system::SystemDiagnosticsRegistry reg;
  pops::EffectiveBlockOptions opt;
  opt.name = "ions";
  opt.route = "native_model";
  reg.block_options["ions"] = opt;
  EXPECT_EQ(reg.block_options_ptr("ions")->route, "native_model");
  EXPECT_EQ(reg.block_options_ptr("absent"), nullptr);

  const auto rows = reg.options_report();
  ASSERT_EQ(rows.size(), 1u);
  EXPECT_EQ(rows[0].name, "ions");

  EXPECT_EQ(reg.newton_report_ptr("ions"), nullptr)
      << "absent -> nullptr, not a silently empty report";
  auto rep = std::make_shared<pops::NewtonReport>();
  rep->enabled = true;
  reg.newton_reports["ions"] = rep;
  ASSERT_NE(reg.newton_report_ptr("ions"), nullptr);
  EXPECT_TRUE(reg.newton_report_ptr("ions")->enabled);
}

TEST(SystemCouplingRegistry, HoldsOperatorsAndBounds) {
  pops::runtime::system::SystemCouplingRegistry reg;
  reg.dt_bounds.push_back({"schur", [] { return 0.5; }});
  reg.coupled_freqs.push_back({"ionization", 3.0});
  EXPECT_EQ(reg.dt_bounds.size(), 1u);
  EXPECT_EQ(reg.dt_bounds[0].label, "schur");
  EXPECT_DOUBLE_EQ(reg.dt_bounds[0].fn(), 0.5);
  EXPECT_EQ(reg.coupled_freqs[0].mu, 3.0);
  EXPECT_TRUE(reg.operators.empty());
  EXPECT_TRUE(reg.coupled_operators.empty());
}

TEST(SystemDomain, LayoutReportReflectsCartesianConstruction) {
  pops::SystemConfig c;
  c.n = 16;
  c.L = 1.0;
  c.periodic = true;
  c.geometry = "cartesian";
  pops::runtime::system::SystemDomain domain(c);
  const auto rep = domain.layout_report();
  EXPECT_FALSE(rep.polar);
  EXPECT_EQ(rep.nx, 16);
  EXPECT_EQ(rep.ny, 16);
  EXPECT_EQ(rep.n_boxes, 1) << "Cartesian is a single box";
  EXPECT_TRUE(rep.periodic);
  EXPECT_FALSE(rep.eb_active);
  EXPECT_GE(rep.aux_ncomp, 3) << "the shared aux channel is at least 3 wide";
}

TEST(SystemEllipticBackendRegistry, OpaqueCapabilitiesDoNotCloseTheExtensionSet) {
  EllipticRegistryHarness::EllipticBackendRegistry registry;
  registry.add("probe", std::make_unique<ProbeEllipticProvider>(
                            std::vector<std::string>{"pops.test.opaque-zeta@4",
                                                     "pops.test.opaque-alpha@9"}));
  auto backend = registry.prepare("probe", elliptic_registry_request());
  ASSERT_NE(backend, nullptr);
  EXPECT_EQ(backend->provider_identity(), "pops.test.system-elliptic-provider");
  EXPECT_EQ(backend->rhs().n_grow(), 0);
  EXPECT_EQ(backend->phi().n_grow(), 1);
}

TEST(SystemEllipticBackendRegistry, EmptyOpaqueCapabilitiesAreValid) {
  EllipticRegistryHarness::EllipticBackendRegistry registry;
  EXPECT_NO_THROW(registry.add("probe", std::make_unique<ProbeEllipticProvider>()));
}

TEST(SystemEllipticBackendRegistry, RejectsMalformedOpaqueCapabilityDeclarations) {
  EllipticRegistryHarness::EllipticBackendRegistry registry;
  EXPECT_THROW(
      registry.add("probe", std::make_unique<ProbeEllipticProvider>(
                                std::vector<std::string>{"pops.test.duplicate@1",
                                                         "pops.test.duplicate@1"})),
      std::invalid_argument);
}

TEST(SystemEllipticBackendRegistry, ProviderSupportOwnsRequestSemantics) {
  EllipticRegistryHarness::EllipticBackendRegistry registry;
  registry.add("probe", std::make_unique<ProbeEllipticProvider>());
  auto request = elliptic_registry_request();
  request.diffusion_coefficient =
      EllipticRegistryHarness::ScalarDiffusionCoefficient<pops::MultiFab>{
          pops::MultiFab(request.elliptic.boxes, request.elliptic.mapping, 1, 0)};
  EXPECT_THROW(
      {
        try {
          (void)registry.prepare("probe", std::move(request));
        } catch (const std::invalid_argument& error) {
          EXPECT_NE(std::string(error.what()).find("code 73"), std::string::npos);
          EXPECT_NE(std::string(error.what()).find("rejects scalar coefficient"),
                    std::string::npos);
          throw;
        }
      },
      std::invalid_argument);
}

TEST(SystemEllipticBackendRegistry, DiffusionCoefficientHasExactlyOneTypedShape) {
  auto request = elliptic_registry_request();
  using Scalar = EllipticRegistryHarness::ScalarDiffusionCoefficient<pops::MultiFab>;
  using Diagonal = EllipticRegistryHarness::DiagonalDiffusionCoefficient<pops::MultiFab>;
  using FullTensor = EllipticRegistryHarness::FullTensorDiffusionCoefficient<pops::MultiFab>;

  request.diffusion_coefficient =
      Scalar{pops::MultiFab(request.elliptic.boxes, request.elliptic.mapping, 1, 0)};
  EXPECT_TRUE(std::holds_alternative<Scalar>(request.diffusion_coefficient));
  EXPECT_FALSE(std::holds_alternative<Diagonal>(request.diffusion_coefficient));

  request.diffusion_coefficient =
      Diagonal{pops::MultiFab(request.elliptic.boxes, request.elliptic.mapping, 1, 0),
               pops::MultiFab(request.elliptic.boxes, request.elliptic.mapping, 1, 0)};
  EXPECT_FALSE(std::holds_alternative<Scalar>(request.diffusion_coefficient));
  EXPECT_TRUE(std::holds_alternative<Diagonal>(request.diffusion_coefficient));
  EXPECT_FALSE(std::holds_alternative<FullTensor>(request.diffusion_coefficient));

  request.diffusion_coefficient = FullTensor{
      pops::MultiFab(request.elliptic.boxes, request.elliptic.mapping, 1, 0),
      pops::MultiFab(request.elliptic.boxes, request.elliptic.mapping, 1, 0),
      pops::MultiFab(request.elliptic.boxes, request.elliptic.mapping, 1, 0),
      pops::MultiFab(request.elliptic.boxes, request.elliptic.mapping, 1, 0)};
  EXPECT_FALSE(std::holds_alternative<Scalar>(request.diffusion_coefficient));
  EXPECT_FALSE(std::holds_alternative<Diagonal>(request.diffusion_coefficient));
  EXPECT_TRUE(std::holds_alternative<FullTensor>(request.diffusion_coefficient));
}

TEST(SystemEllipticBackendRegistry, ValidatesLayoutsWithoutAnOperatorContract) {
  EllipticRegistryHarness::EllipticBackendRegistry registry;
  registry.add("probe", std::make_unique<ProbeEllipticProvider>(
                            std::vector<std::string>{}, true));
  EXPECT_THROW((void)registry.prepare("probe", elliptic_registry_request()),
               std::runtime_error);
}

}  // namespace
