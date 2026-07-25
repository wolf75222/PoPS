// Exact collective consensus for resolved field-plan registries. Each scenario uses a fresh facade:
// setters are intentionally local/non-collective, then mark_bound compares one canonical std::map-
// ordered sequence of (provider_slot, plan_identity) before field-plan materialization.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/numerics/elliptic/interface/elliptic_solver.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr/hierarchy_tensor_solver_provider.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/system.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

enum class EllipticFactoryFault {
  None,
  ThrowOnRankOne,
  NullOnRankOne,
  WrongComponentsOnRankOne,
  AliasedFieldsOnRankOne,
  WrongGhostsOnRankOne,
  WrongOperatorContractOnRankOne,
  WrongDistributionOnRankOne,
  InspectionThrowsOnRankOne,
};

class ConsensusElliptic {
 public:
  ConsensusElliptic(const Geometry& geometry, const BoxArray& boxes,
                    const DistributionMapping& mapping, const BCRec& boundary,
                    ActiveRegionProvider2D active, FieldDistribution distribution,
                    EllipticFactoryFault fault)
      : geometry_(geometry),
        rhs_(boxes, mapping,
             fault == EllipticFactoryFault::WrongComponentsOnRankOne && my_rank() == 1 ? 2 : 1,
             fault == EllipticFactoryFault::WrongGhostsOnRankOne && my_rank() == 1 ? 1 : 0),
        phi_(boxes, mapping, 1, 1),
        distribution_(fault == EllipticFactoryFault::WrongDistributionOnRankOne && my_rank() == 1
                          ? FieldDistribution::Replicated
                          : distribution),
        alias_fields_(fault == EllipticFactoryFault::AliasedFieldsOnRankOne && my_rank() == 1),
        inspection_throws_(fault == EllipticFactoryFault::InspectionThrowsOnRankOne &&
                           my_rank() == 1),
        operator_contract_(make_materialized_elliptic_operator_contract(
            fault == EllipticFactoryFault::WrongOperatorContractOnRankOne && my_rank() == 1
                ? EllipticOperatorIdentity{"pops.test.consensus-operator.wrong", 1}
                : operator_identity(),
            geometry_, boundary, active, distribution_, rhs_, phi_)) {}

  static constexpr EllipticOperatorIdentity operator_identity() noexcept {
    return {"pops.test.consensus-operator", 1};
  }

  static EllipticOperatorContract expected_operator_contract(const EllipticBuildRequest& request) {
    return make_expected_elliptic_operator_contract(operator_identity(), request);
  }

  MultiFab& rhs() {
    if (inspection_throws_)
      throw std::runtime_error("intentional rank-local elliptic accessor failure");
    return rhs_;
  }
  MultiFab& phi() { return alias_fields_ ? rhs_ : phi_; }
  void solve() {}
  Real residual() const { return Real(0); }
  const Geometry& geom() const { return geometry_; }
  FieldDistribution field_distribution() const noexcept { return distribution_; }
  const EllipticOperatorContract& prepared_operator_contract() const noexcept {
    return operator_contract_;
  }

 private:
  Geometry geometry_;
  MultiFab rhs_;
  MultiFab phi_;
  FieldDistribution distribution_;
  bool alias_fields_;
  bool inspection_throws_;
  EllipticOperatorContract operator_contract_;
};

struct ConsensusEllipticFactory {
  int* constructions;
  std::string contract{"pops.test.consensus-elliptic-factory@1"};
  EllipticFactoryFault fault{EllipticFactoryFault::None};

  [[nodiscard]] std::string_view collective_contract() const noexcept { return contract; }

  [[nodiscard]] EllipticOperatorContract expected_operator_contract(
      const EllipticBuildRequest& request) const {
    return ConsensusElliptic::expected_operator_contract(request);
  }

  [[nodiscard]] FieldDistribution materialized_distribution(
      const EllipticBuildRequest& request) const noexcept {
    return request.distribution;
  }

  [[nodiscard]] bool supports(const EllipticBuildRequest&) const noexcept { return true; }

  EllipticFactoryBuildResult<ConsensusElliptic> build(EllipticBuildRequest request) const noexcept {
    if (fault == EllipticFactoryFault::NullOnRankOne && my_rank() == 1) {
      ++*constructions;
      return {};
    }
    return capture_local_elliptic_factory_build<ConsensusElliptic>([this,
                                                                    request = std::move(request)] {
      ++*constructions;
      if (fault == EllipticFactoryFault::ThrowOnRankOne && my_rank() == 1)
        throw std::runtime_error("intentional rank-local elliptic factory failure");
      return ConsensusElliptic(request.geometry, request.boxes, request.mapping, request.boundary,
                               std::move(request.active), request.distribution, fault);
    });
  }
};

static_assert(EllipticFactory<ConsensusEllipticFactory, ConsensusElliptic>);

enum class SolveReportFault {
  None,
  OutcomeOnRankOne,
  ReasonBytesOnRankOne,
};

SolveReport make_consensus_report(SolveReportFault fault) {
  SolveReport report;
  report.iters = 1;
  report.rel_residual = Real(0.125);
  report.reference_residual_norm = Real(8);
  report.residual_norm = Real(1);
  if (fault == SolveReportFault::OutcomeOnRankOne && my_rank() == 1) {
    report.mark_failed(SolveStatus::kIterationLimit, SolveAction::kRejectAttempt,
                       "rank-one-failed");
  } else if (fault == SolveReportFault::ReasonBytesOnRankOne) {
    std::string reason(1025, 'r');
    reason.back() = my_rank() == 1 ? '1' : '0';  // difference is in the second fixed-size chunk
    report.mark_solved(std::move(reason));
  } else {
    report.mark_solved("collective-solved");
  }
  return report;
}

class ConsensusHierarchyPrepared final
    : public runtime::program::PreparedHierarchyTensorSolver {
 public:
  explicit ConsensusHierarchyPrepared(SolveReportFault fault) : fault_(fault) {}

  std::string_view provider_identity() const noexcept override {
    return "pops.test.mpi-consensus-hierarchy";
  }
  std::uint64_t provider_version() const noexcept override { return 1; }
  std::string_view exact_prepared_contract() const noexcept override {
    return "pops.test.mpi-consensus-hierarchy.prepared@1";
  }
  runtime::program::HierarchyTensorSolverExecutionPath execution_path() const noexcept override {
    return runtime::program::HierarchyTensorSolverExecutionPath::DirectProvider;
  }
  MultiFab& assembly_target(std::string_view, int) override {
    throw std::logic_error("report-only MPI provider has no field storage");
  }
  MultiFab& solution(int) override {
    throw std::logic_error("report-only MPI provider has no field storage");
  }
  void stage_initial_guess(int, const MultiFab*) override {}

  SolveReport solve(const runtime::program::HierarchyTensorSolveControls&) override {
    return make_consensus_report(fault_);
  }

 private:
  SolveReportFault fault_;
};

class ConsensusAmrFieldPrepared final : public AmrPreparedFieldSolver {
 public:
  explicit ConsensusAmrFieldPrepared(SolveReportFault fault) : fault_(fault) {}

  std::string_view provider_identity() const noexcept override {
    return "pops.test.mpi-consensus-amr-field";
  }
  std::string_view exact_prepared_contract() const noexcept override {
    return "pops.test.mpi-consensus-amr-field.prepared@1";
  }
  bool couples_hierarchy_levels() const noexcept override { return false; }
  int level_count() const noexcept override { return 1; }
  FieldDistribution level_distribution(int) const override {
    return FieldDistribution::Distributed;
  }
  MultiFab& rhs_level(int) override {
    throw std::logic_error("report-only AMR field provider has no field storage");
  }
  MultiFab& phi_level(int) override {
    throw std::logic_error("report-only AMR field provider has no field storage");
  }
  void set_boundary_context(const FieldBoundaryExecutionContext&) override {}
  SolveReport solve() override {
    report_ = make_consensus_report(fault_);
    return report_;
  }
  const SolveReport& last_solve_report() const noexcept override { return report_; }

 private:
  SolveReportFault fault_;
  SolveReport report_{};
};

bool elliptic_request_rejected(
    const Geometry& geometry, const BoxArray& boxes, const DistributionMapping& mapping,
    FieldDistribution distribution, int& constructions, ActiveRegionProvider2D active = {},
    std::string factory_contract = "pops.test.consensus-elliptic-factory@1") {
  try {
    (void)make_elliptic_solver<ConsensusElliptic>(
        {geometry, boxes, mapping, BCRec{}, std::move(active), distribution},
        ConsensusEllipticFactory{&constructions, std::move(factory_contract)});
  } catch (const std::invalid_argument&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

bool elliptic_materialization_rejected(EllipticFactoryFault fault, int& constructions) {
  const Box2D domain = Box2D::from_extents(8, 8);
  const Geometry geometry{domain, 0.0, 1.0, 0.0, 1.0};
  const BoxArray boxes = BoxArray::from_domain(domain, 4);
  const DistributionMapping mapping(boxes.size(), n_ranks());
  try {
    (void)make_elliptic_solver<ConsensusElliptic>(
        {geometry, boxes, mapping, BCRec{}, {}, FieldDistribution::Distributed},
        ConsensusEllipticFactory{&constructions, "pops.test.consensus-elliptic-factory@1", fault});
  } catch (const std::exception&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

PreparedProviderOptions system_geometric_options(double rel_tol = 1.0e-8) {
  return {"pops.system.geometric-mg-options@1",
          {{"abs_tol", 0.0},
           {"bottom_sweeps", std::int64_t{50}},
           {"coarse_threshold", std::int64_t{0}},
           {"max_cycles", std::int64_t{50}},
           {"min_coarse", std::int64_t{2}},
           {"post_smooth", std::int64_t{2}},
           {"pre_smooth", std::int64_t{2}},
           {"rel_tol", rel_tol}}};
}

void install(System& system, const std::string& slot, const std::string& plan_identity,
             bool register_backend = true, double provider_coefficient = 1.0) {
  if (register_backend)
    system.register_configured_field_solver_provider("geometric_mg", slot,
                                                     system_geometric_options());
  system.set_field_solver_plan(slot, plan_identity, "provider:" + slot, "output-owner", "plasma",
                               "potential", {"rhs-provider"}, {"plasma"}, {"potential"},
                               {provider_coefficient}, slot);
}

AmrFieldSolverOptions amr_geometric_options() {
  GeometricMgOptions mg;
  mg.abs_tol = Real(0);
  mg.rel_tol = Real(1.0e-8);
  mg.max_cycles = 50;
  mg.min_coarse = 2;
  mg.nu1 = 2;
  mg.nu2 = 2;
  mg.nbottom = 50;
  mg.coarse_threshold = 0;
  return geometric_mg_amr_field_solver_options(mg, CompositeFacOptions{});
}

AmrFieldHierarchyPolicyAuthority composite_hierarchy_policy() {
  return {
      "pops.field-hierarchy.composite",
      1,
      {"pops.field-hierarchy.options.empty@1", {}},
  };
}

void install(AmrSystem& system, const std::string& slot, const std::string& plan_identity,
             double provider_coefficient = 1.0) {
  system.set_field_solver_plan(slot, plan_identity, "provider:" + slot, "output-owner", "plasma",
                               "potential", {"rhs-provider"}, {"plasma"}, {"potential"},
                               {provider_coefficient}, "geometric_mg",
                               composite_hierarchy_policy(),
                               amr_geometric_options());
  system.set_field_nullspace(
      slot, "pops.field-nullspace.operator-topology-derived",
      PreparedProviderOptions{"pops.field-nullspace.operator-topology-derived.options@1",
                              {{"gauge.value", 0.0}}});
}

template <class SystemType>
bool bind_rejected(SystemType& system) {
  try {
    system.mark_bound();
  } catch (const std::runtime_error&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

template <class SystemType>
bool duplicate_rejected(SystemType& system) {
  try {
    install(system, "field-slot", "shared-plan-identity");
    install(system, "field-slot", "shared-plan-identity");
  } catch (const std::runtime_error&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

bool duplicate_rejected(System& system) {
  try {
    install(system, "field-slot", "shared-plan-identity");
    install(system, "field-slot", "shared-plan-identity", false);
  } catch (const std::runtime_error&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

int run_field_plan_consensus(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const int rank = my_rank();
  const int ranks = n_ranks();
  long failures = ranks == 2 ? 0 : 1;
  const auto require = [&failures](bool condition) {
    if (!condition)
      ++failures;
  };

  // A hierarchy provider cannot split publication by returning individually valid but different
  // reports. Both outcome divergence and equal-length reason-byte divergence are rejected with one
  // uniform error on every rank; an identical report remains publishable.
  {
    ConsensusHierarchyPrepared solver(SolveReportFault::None);
    try {
      const SolveReport report =
          runtime::program::solve_prepared_hierarchy_tensor_collectively(
              solver, {Real(1.0e-8), Real(0), 4});
      require(report.solved());
      require(report.reason == "collective-solved");
    } catch (...) {
      require(false);
    }
  }
  for (const SolveReportFault fault : {SolveReportFault::OutcomeOnRankOne,
                                       SolveReportFault::ReasonBytesOnRankOne}) {
    ConsensusHierarchyPrepared solver(fault);
    bool rejected = false;
    bool exact_error = false;
    try {
      (void)runtime::program::solve_prepared_hierarchy_tensor_collectively(
          solver, {Real(1.0e-8), Real(0), 4});
    } catch (const std::runtime_error& error) {
      rejected = true;
      exact_error =
          std::string_view(error.what()) ==
          "hierarchy tensor-solver provider report differs between MPI ranks";
    } catch (...) {
    }
    require(rejected);
    require(exact_error);
  }

  // The same provider-neutral boundary protects both default and named AMR field transactions.
  // A solved/failed split would otherwise commit on one rank and restore the snapshot on the other.
  {
    ConsensusAmrFieldPrepared solver(SolveReportFault::None);
    try {
      const SolveReport report = solve_prepared_amr_field_solver_collectively(solver);
      require(report.solved());
      require(report.reason == "collective-solved");
    } catch (...) {
      require(false);
    }
  }
  for (const SolveReportFault fault : {SolveReportFault::OutcomeOnRankOne,
                                       SolveReportFault::ReasonBytesOnRankOne}) {
    ConsensusAmrFieldPrepared solver(fault);
    bool rejected = false;
    bool exact_error = false;
    try {
      (void)solve_prepared_amr_field_solver_collectively(solver);
    } catch (const std::runtime_error& error) {
      rejected = true;
      exact_error = std::string_view(error.what()) ==
                    "AMR field-solver provider report differs between MPI ranks";
    } catch (...) {
    }
    require(rejected);
    require(exact_error);
  }

  // A malformed or divergent elliptic layout is rejected collectively before an arbitrary backend
  // factory can enter MPI. These are deliberately rank-local descriptor faults.
  {
    const Box2D domain = Box2D::from_extents(8, 8);
    const Geometry geometry{domain, 0.0, 1.0, 0.0, 1.0};
    const BoxArray boxes = BoxArray::from_domain(domain, 4);
    std::vector<int> owners = DistributionMapping(boxes.size(), ranks).ranks();
    if (rank == 1)
      owners.pop_back();
    int constructions = 0;
    require(elliptic_request_rejected(geometry, boxes, DistributionMapping(std::move(owners)),
                                      FieldDistribution::Distributed, constructions));
    require(constructions == 0);
  }

  // Geometry, boundary, prepared-provider and backend-factory identities are exact collective
  // inputs too. None may become a hidden rank-local callback or backend choice.
  {
    const Box2D domain = Box2D::from_extents(8, 8);
    const Geometry geometry{domain, 0.0, rank == 0 ? 1.0 : 2.0, 0.0, 1.0};
    const BoxArray boxes = BoxArray::from_domain(domain, 4);
    const DistributionMapping mapping(boxes.size(), ranks);
    int constructions = 0;
    require(elliptic_request_rejected(geometry, boxes, mapping, FieldDistribution::Distributed,
                                      constructions));
    require(constructions == 0);
  }
  {
    const Box2D domain = Box2D::from_extents(8, 8);
    const Geometry geometry{domain, 0.0, 1.0, 0.0, 1.0};
    const BoxArray boxes = BoxArray::from_domain(domain, 4);
    const DistributionMapping mapping(boxes.size(), ranks);
    const std::string contract =
        rank == 0 ? "pops.test.factory.rank-0@1" : "pops.test.factory.rank-1@1";
    int constructions = 0;
    require(elliptic_request_rejected(geometry, boxes, mapping, FieldDistribution::Distributed,
                                      constructions, {}, contract));
    require(constructions == 0);
  }
  {
    const Box2D domain = Box2D::from_extents(8, 8);
    const Geometry geometry{domain, 0.0, 1.0, 0.0, 1.0};
    const BoxArray boxes = BoxArray::from_domain(domain, 4);
    const DistributionMapping mapping(boxes.size(), ranks);
    BCRec boundary;
    boundary.xlo_val = rank == 0 ? Real(0) : Real(1);
    int constructions = 0;
    try {
      (void)make_elliptic_solver<ConsensusElliptic>(
          {geometry, boxes, mapping, boundary, {}, FieldDistribution::Distributed},
          ConsensusEllipticFactory{&constructions});
      require(false);
    } catch (const std::invalid_argument&) {
      require(true);
    } catch (...) {
      require(false);
    }
    require(constructions == 0);
  }
  {
    const Box2D domain = Box2D::from_extents(8, 8);
    const Geometry geometry{domain, 0.0, 1.0, 0.0, 1.0};
    const BoxArray boxes = BoxArray::from_domain(domain, 4);
    const DistributionMapping mapping(boxes.size(), ranks);
    const Real radius = rank == 0 ? Real(0.25) : Real(0.5);
    ActiveRegionProvider2D active = ActiveRegionProvider2D::trusted_extension(
        {"pops.test.active-region.circle", 1}, exact_provider_parameters(radius),
        [radius](Real x, Real y) { return std::hypot(x - Real(0.5), y - Real(0.5)) < radius; });
    int constructions = 0;
    require(elliptic_request_rejected(geometry, boxes, mapping, FieldDistribution::Distributed,
                                      constructions, std::move(active)));
    require(constructions == 0);
  }
  {
    const Box2D domain = Box2D::from_extents(8, 8);
    const Geometry geometry{domain, 0.0, 1.0, 0.0, 1.0};
    const BoxArray boxes = BoxArray::from_domain(domain, 4);
    std::vector<int> owners = DistributionMapping(boxes.size(), ranks).ranks();
    if (rank == 1)
      std::swap(owners[0], owners[1]);
    int constructions = 0;
    require(elliptic_request_rejected(geometry, boxes, DistributionMapping(std::move(owners)),
                                      FieldDistribution::Distributed, constructions));
    require(constructions == 0);
  }
  {
    const Box2D domain = Box2D::from_extents(8, 8);
    const Geometry geometry{domain, 0.0, 1.0, 0.0, 1.0};
    const BoxArray boxes = BoxArray::from_domain(domain, 4);
    const DistributionMapping mapping(boxes.size(), ranks);
    const FieldDistribution distribution =
        rank == 1 ? static_cast<FieldDistribution>(0xff) : FieldDistribution::Distributed;
    int constructions = 0;
    require(elliptic_request_rejected(geometry, boxes, mapping, distribution, constructions));
    require(constructions == 0);
  }

  // Every post-factory fault is captured locally before the common reduction. One rank may throw,
  // return no object, or materialize a dishonest backend contract; both ranks still reject instead
  // of leaving a peer blocked in MPI.
  for (const EllipticFactoryFault fault : {
           EllipticFactoryFault::ThrowOnRankOne,
           EllipticFactoryFault::NullOnRankOne,
           EllipticFactoryFault::WrongComponentsOnRankOne,
           EllipticFactoryFault::AliasedFieldsOnRankOne,
           EllipticFactoryFault::WrongGhostsOnRankOne,
           EllipticFactoryFault::WrongOperatorContractOnRankOne,
           EllipticFactoryFault::WrongDistributionOnRankOne,
           EllipticFactoryFault::InspectionThrowsOnRankOne,
       }) {
    int constructions = 0;
    require(elliptic_materialization_rejected(fault, constructions));
    require(constructions == 1);
  }

  // Same registry shape and token lengths, but different bytes: both facades reject uniformly.
  {
    const std::string token = rank == 0 ? "plan-rank-0" : "plan-rank-1";
    System system(SystemConfig{16, 1.0, Periodicity{true, true}});
    install(system, "field-slot", token);
    require(bind_rejected(system));
  }
  {
    const std::string token = rank == 0 ? "plan-rank-0" : "plan-rank-1";
    AmrSystem system(AmrSystemConfig{16});
    install(system, "field-slot", token);
    require(bind_rejected(system));
  }

  // A caller token is provenance, not an authority.  Equal slot/token bytes cannot hide a
  // rank-local difference in the resolved provider pack.
  {
    System system(SystemConfig{16, 1.0, Periodicity{true, true}});
    install(system, "field-slot", "shared-plan", true, rank == 0 ? 1.0 : 2.0);
    require(bind_rejected(system));
  }
  {
    AmrSystem system(AmrSystemConfig{16});
    install(system, "field-slot", "shared-plan", rank == 0 ? 1.0 : 2.0);
    require(bind_rejected(system));
  }

  // The slot participates independently in the pair; an equal plan token cannot hide slot drift.
  {
    const std::string slot = rank == 0 ? "field-rank-0" : "field-rank-1";
    System system(SystemConfig{16, 1.0, Periodicity{true, true}});
    install(system, slot, "shared-plan");
    require(bind_rejected(system));
  }
  {
    const std::string slot = rank == 0 ? "field-rank-0" : "field-rank-1";
    AmrSystem system(AmrSystemConfig{16});
    install(system, slot, "shared-plan");
    require(bind_rejected(system));
  }

  // Component length disagreement returns before the byte collective.
  {
    const std::string token = rank == 0 ? "x" : "plan-with-another-length";
    System system(SystemConfig{16, 1.0, Periodicity{true, true}});
    install(system, "field-slot", token);
    require(bind_rejected(system));
  }
  {
    const std::string token = rank == 0 ? "x" : "plan-with-another-length";
    AmrSystem system(AmrSystemConfig{16});
    install(system, "field-slot", token);
    require(bind_rejected(system));
  }

  // A missing/extra plan agrees the pair count first. This is the case that deadlocked when the
  // setter itself was collective: rank 1 executes one more local setter than rank 0.
  {
    System system(SystemConfig{16, 1.0, Periodicity{true, true}});
    install(system, "field-a", "plan-a");
    if (rank == 1)
      install(system, "field-b", "plan-b");
    require(bind_rejected(system));
  }
  {
    AmrSystem system(AmrSystemConfig{16});
    install(system, "field-a", "plan-a");
    if (rank == 1)
      install(system, "field-b", "plan-b");
    require(bind_rejected(system));
  }

  // Setter order is not semantic: std::map canonicalization produces the same two pairs.
  {
    System system(SystemConfig{16, 1.0, Periodicity{true, true}});
    if (rank == 0) {
      install(system, "field-b", "plan-b");
      install(system, "field-a", "plan-a");
    } else {
      install(system, "field-a", "plan-a");
      install(system, "field-b", "plan-b");
    }
    require(!bind_rejected(system));
  }
  {
    AmrSystem system(AmrSystemConfig{16});
    if (rank == 0) {
      install(system, "field-b", "plan-b");
      install(system, "field-a", "plan-a");
    } else {
      install(system, "field-a", "plan-a");
      install(system, "field-b", "plan-b");
    }
    require(!bind_rejected(system));
  }

  // Duplicate slots are a local structural error, including byte-identical repeats; no collective
  // is entered and no partially overwritten plan survives.
  {
    System system(SystemConfig{16, 1.0, Periodicity{true, true}});
    require(duplicate_rejected(system));
  }
  {
    AmrSystem system(AmrSystemConfig{16});
    require(duplicate_rejected(system));
  }

  // Native finite/domain guards remain authoritative even if a caller bypasses Python schemas.
  {
    System system(SystemConfig{16, 1.0, Periodicity{true, true}});
    bool rejected = false;
    try {
      system.register_configured_field_solver_provider(
          "geometric_mg", "field-slot",
          system_geometric_options(std::numeric_limits<double>::infinity()));
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected);
  }
  {
    AmrSystem system(AmrSystemConfig{16});
    CompositeFacOptions invalid;
    invalid.coarse_abs_tol = std::numeric_limits<Real>::quiet_NaN();
    bool rejected = false;
    try {
      system.set_field_solver_plan(
          "field-slot", "plan", "provider", "output-owner", "plasma", "potential", {"rhs-provider"},
          {"plasma"}, {"potential"}, {1.0}, "geometric_mg",
          composite_hierarchy_policy(),
          geometric_mg_amr_field_solver_options(GeometricMgOptions{}, invalid));
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected);
  }

  failures = all_reduce_sum(failures);
  comm_finalize();
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_field_plan_consensus, CanonicalRegistryRefusesDivergenceWithoutDeadlock) {
  EXPECT_EQ(pops::test::RunTestBody(&run_field_plan_consensus, "test_mpi_field_plan_consensus"), 0);
}
