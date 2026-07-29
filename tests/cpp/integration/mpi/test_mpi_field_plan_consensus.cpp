// Exact collective consensus for resolved field-plan registries and level-qualified AMR stage
// packs. Registry scenarios keep setters local/non-collective, then mark_bound compares one
// canonical std::map-ordered sequence of (provider_slot, plan_identity). The stage-pack scenario
// drives distributed L0/L1 storage and proves successful publication, while a second supported
// replicated-L0/distributed-L1 scenario proves the composite field-coupled residual JVP against an
// independent finite difference. It also proves a level-local solved-field physical-boundary JVP
// and pre-solve rejection when provider, evaluation point, or pack presence differs between ranks.

#include <gtest/gtest.h>

#include "amr_transfer_test_authority.hpp"
#include "gtest_compat.hpp"
#include "load_balance_test_authority.hpp"
#include <pops/core/state/state.hpp>
#include <pops/coupling/base/elliptic_rhs.hpp>
#include <pops/numerics/elliptic/interface/elliptic_solver.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/physics/bricks/bricks.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr/hierarchy_tensor_solver_provider.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/system.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
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
  ThrowWithStaleReject,
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

class ConsensusHierarchyPrepared final : public runtime::program::PreparedHierarchyTensorSolver {
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
  int level_count() const noexcept override { return 0; }
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
  explicit ConsensusAmrFieldPrepared(SolveReportFault fault)
      : fault_(fault),
        boxes_(BoxArray::from_domain(Box2D::from_extents(4, 4), 2)),
        mapping_(boxes_.size(), n_ranks()),
        rhs_(boxes_, mapping_, 1, 0),
        phi_(boxes_, mapping_, 1, 0) {
    rhs_.set_val(Real(0));
    phi_.set_val(Real(0));
    if (fault_ == SolveReportFault::ThrowWithStaleReject)
      report_.mark_failed(SolveStatus::kIterationLimit, SolveAction::kRejectAttempt,
                          "stale-prior-reject");
  }

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
  MultiFab& rhs_level(int) override { return rhs_; }
  MultiFab& phi_level(int) override { return phi_; }
  void set_phi_layout_drift(bool drift) { phi_ = MultiFab(boxes_, mapping_, 1, drift ? 1 : 0); }
  void set_boundary_context(const FieldBoundaryExecutionContext&) override {}
  SolveReport solve() override {
    if (fault_ == SolveReportFault::ThrowWithStaleReject)
      throw std::runtime_error("unknown failure after a prior rejected attempt");
    report_ = make_consensus_report(fault_);
    return report_;
  }
  const SolveReport& last_solve_report() const noexcept override { return report_; }

 private:
  SolveReportFault fault_;
  BoxArray boxes_;
  DistributionMapping mapping_;
  MultiFab rhs_;
  MultiFab phi_;
  SolveReport report_{};
};

struct RankLocalPublication {
  bool layout_valid = true;
  bool accepted = false;
  bool rejected = false;

  static void validate(void* context) {
    if (!static_cast<RankLocalPublication*>(context)->layout_valid)
      throw std::logic_error("rank-local destination layout drift");
  }
  static void accept(void* context) noexcept {
    static_cast<RankLocalPublication*>(context)->accepted = true;
  }
  static void reject(void* context) {
    static_cast<RankLocalPublication*>(context)->rejected = true;
  }
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
                               "potential:" + slot, {"rhs-provider"}, {"plasma"}, {"potential"},
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

AmrFieldHierarchyPolicyAuthority level_local_hierarchy_policy() {
  return {
      "pops.field-hierarchy.level-local",
      1,
      {"pops.field-hierarchy.options.empty@1", {}},
  };
}

using StagePackModel = CompositeModel<ExBVelocity, NoSource, ChargeDensity>;

StagePackModel stage_pack_model() {
  return StagePackModel{ExBVelocity{Real(1)}, NoSource{}, ChargeDensity{Real(1)}};
}

std::vector<double> stage_pack_density(int n, double amplitude) {
  std::vector<double> density(static_cast<std::size_t>(n) * n, Real(0));
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (static_cast<double>(i) + 0.5) / static_cast<double>(n) - 0.5;
      const double y = (static_cast<double>(j) + 0.5) / static_cast<double>(n) - 0.5;
      density[static_cast<std::size_t>(j) * n + i] = amplitude * std::exp(-(x * x + y * y) / 0.025);
    }
  return density;
}

Real global_max_allocated_diff(const MultiFab& lhs, const MultiFab& rhs) {
  if (lhs.box_array().boxes() != rhs.box_array().boxes() ||
      lhs.dmap().ranks() != rhs.dmap().ranks() || lhs.ncomp() != rhs.ncomp() ||
      lhs.n_grow() != rhs.n_grow())
    throw std::invalid_argument("MPI stage-pack comparison requires identical layouts");
  device_fence();
  Real local = Real(0);
  for (int li = 0; li < lhs.local_size(); ++li) {
    const ConstArray4 left = lhs.fab(li).const_array();
    const ConstArray4 right = rhs.fab(li).const_array();
    const Box2D grown = lhs.fab(li).grown_box();
    for (int component = 0; component < lhs.ncomp(); ++component)
      for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
        for (int i = grown.lo[0]; i <= grown.hi[0]; ++i)
          local = std::max(local, std::fabs(left(i, j, component) - right(i, j, component)));
  }
  return all_reduce_max(local);
}

Real global_max_valid_scalar_diff(const MultiFab& lhs, const MultiFab& rhs) {
  if (lhs.box_array().boxes() != rhs.box_array().boxes() ||
      lhs.dmap().ranks() != rhs.dmap().ranks())
    throw std::invalid_argument("MPI scalar comparison requires identical layouts");
  device_fence();
  Real local = Real(0);
  for (int li = 0; li < lhs.local_size(); ++li) {
    const ConstArray4 left = lhs.fab(li).const_array();
    const ConstArray4 right = rhs.fab(li).const_array();
    const Box2D valid = lhs.box(li);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        local = std::max(local, std::fabs(left(i, j, 0) - right(i, j, 0)));
  }
  return all_reduce_max(local);
}

std::pair<Real, Real> global_physical_boundary_support(const MultiFab& values,
                                                       const Box2D& domain) {
  device_fence();
  Real local_boundary = Real(0);
  Real local_interior = Real(0);
  for (int li = 0; li < values.local_size(); ++li) {
    const ConstArray4 data = values.fab(li).const_array();
    const Box2D valid = values.box(li);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
        const Real magnitude = std::fabs(data(i, j, 0));
        const bool physical =
            i == domain.lo[0] || i == domain.hi[0] || j == domain.lo[1] || j == domain.hi[1];
        Real& maximum = physical ? local_boundary : local_interior;
        maximum = std::max(maximum, magnitude);
      }
  }
  return {all_reduce_max(local_boundary), all_reduce_max(local_interior)};
}

void add_valid_constant(MultiFab& field, Real value) {
  device_fence();
  for (int li = 0; li < field.local_size(); ++li) {
    Array4 destination = field.fab(li).array();
    const Box2D valid = field.box(li);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        destination(i, j, 0) += value;
  }
}

std::pair<Real, Real> global_stage_pack_superposition_error(const MultiFab& both,
                                                            const MultiFab& only_a,
                                                            const MultiFab& only_b,
                                                            const MultiFab& base) {
  if (both.box_array().boxes() != only_a.box_array().boxes() ||
      both.box_array().boxes() != only_b.box_array().boxes() ||
      both.box_array().boxes() != base.box_array().boxes() ||
      both.dmap().ranks() != only_a.dmap().ranks() ||
      both.dmap().ranks() != only_b.dmap().ranks() || both.dmap().ranks() != base.dmap().ranks())
    throw std::invalid_argument("MPI stage-pack superposition requires identical layouts");
  device_fence();
  Real local_error = Real(0);
  Real local_response = Real(0);
  for (int li = 0; li < both.local_size(); ++li) {
    const ConstArray4 simultaneous = both.fab(li).const_array();
    const ConstArray4 a = only_a.fab(li).const_array();
    const ConstArray4 b = only_b.fab(li).const_array();
    const ConstArray4 origin = base.fab(li).const_array();
    const Box2D valid = both.box(li);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
        const Real simultaneous_response = simultaneous(i, j) - origin(i, j);
        const Real separate_response = (a(i, j) - origin(i, j)) + (b(i, j) - origin(i, j));
        local_error = std::max(local_error, std::fabs(simultaneous_response - separate_response));
        local_response = std::max(local_response, std::fabs(simultaneous_response));
      }
  }
  return {all_reduce_max(local_error), all_reduce_max(local_response)};
}

bool mapping_is_distributed_across_two_ranks(const DistributionMapping& mapping) {
  const auto& owners = mapping.ranks();
  return std::find(owners.begin(), owners.end(), 0) != owners.end() &&
         std::find(owners.begin(), owners.end(), 1) != owners.end();
}

DistributionMapping split_xlow_face_across_ranks(const BoxArray& boxes, const Box2D& domain) {
  if (n_ranks() <= 0)
    throw std::logic_error("physical-boundary distribution requires an active communicator");
  std::vector<int> owners(static_cast<std::size_t>(boxes.size()), 0);
  int next_face_owner = 0;
  int next_other_owner = 0;
  for (int box = 0; box < boxes.size(); ++box) {
    if (boxes[box].lo[0] == domain.lo[0])
      owners[static_cast<std::size_t>(box)] = next_face_owner++ % n_ranks();
    else
      owners[static_cast<std::size_t>(box)] = next_other_owner++ % n_ranks();
  }
  return DistributionMapping(std::move(owners));
}

bool xlow_face_is_distributed_across_two_ranks(const BoxArray& boxes,
                                               const DistributionMapping& mapping,
                                               const Box2D& domain) {
  bool rank_zero = false;
  bool rank_one = false;
  for (int box = 0; box < boxes.size(); ++box) {
    if (boxes[box].lo[0] != domain.lo[0])
      continue;
    rank_zero = rank_zero || mapping[box] == 0;
    rank_one = rank_one || mapping[box] == 1;
  }
  return rank_zero && rank_one;
}

void install(AmrSystem& system, const std::string& slot, const std::string& plan_identity,
             double provider_coefficient = 1.0) {
  system.set_field_solver_plan(slot, plan_identity, "provider:" + slot, "output-owner", "plasma",
                               "potential:" + slot, {"rhs-provider"}, {"plasma"}, {"potential"},
                               {provider_coefficient}, "geometric_mg", composite_hierarchy_policy(),
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

long prove_field_jacvec_route(AmrRuntime& runtime, const std::string& jacvec_field,
                              const std::string& block_name, int block_index,
                              const AmrFieldHierarchyPolicyAuthority& hierarchy_policy,
                              std::string_view topology_digest, std::string_view route_label) {
  constexpr Real c_dt = Real(0.01);
  constexpr Real h = Real(2e-4);
  long failures = 0;
  const auto require = [&failures, route_label](bool condition, std::string_view check) {
    if (!condition) {
      std::fprintf(stderr, "rank %d: %.*s failed: %.*s\n", my_rank(),
                   static_cast<int>(route_label.size()), route_label.data(),
                   static_cast<int>(check.size()), check.data());
      ++failures;
    }
  };
  const auto consume_solved = [route_label](SolveOutcome outcome, std::string_view check) {
    if (!outcome.report().solved()) {
      const SolveConsumption action = outcome.report().action == SolveAction::kRejectAttempt
                                          ? SolveConsumption::kRejectAttempt
                                          : SolveConsumption::kFailRun;
      const SolveReport failed = outcome.consume(action);
      char metrics[192];
      std::snprintf(metrics, sizeof(metrics),
                    " (iters=%d, residual=%.17g, reference=%.17g, relative=%.17g)", failed.iters,
                    static_cast<double>(failed.residual_norm),
                    static_cast<double>(failed.reference_residual_norm),
                    static_cast<double>(failed.rel_residual));
      throw std::runtime_error(std::string(route_label) + " " + std::string(check) +
                               " failed: " + failed.reason + metrics);
    }
    return outcome.consume(SolveConsumption::kAccept);
  };

  AmrFieldSolveConfig jacvec_plan;
  CompositeFacOptions fac_options;
  // Preserve the production tolerance while giving the distributed partial-refinement FAC route
  // enough outer cycles to reach it; the default 30 cycles stops near 2e-7 on this tiny hierarchy.
  fac_options.max_iters = 80;
  jacvec_plan.solver_options =
      geometric_mg_amr_field_solver_options(GeometricMgOptions{}, fac_options);
  jacvec_plan.plan_identity = "tests.mpi." + jacvec_field + ".plan@1";
  jacvec_plan.provider_identity = "tests.mpi." + jacvec_field;
  jacvec_plan.topology_provider_kind = "structured";
  jacvec_plan.topology_provenance = "tests.mpi.periodic-cartesian";
  jacvec_plan.topology_digest = std::string(topology_digest);
  jacvec_plan.output_owner_identity = "tests.mpi.stage-pack." + block_name;
  jacvec_plan.output_block = block_name;
  jacvec_plan.output_key = jacvec_field;
  jacvec_plan.hierarchy_policy = hierarchy_policy;
  jacvec_plan.nullspace = operator_topology_zero_mean_nullspace();
  jacvec_plan.has_reaction = true;
  jacvec_plan.reaction = Real(2);
  jacvec_plan.providers.push_back(FieldProviderBinding{"tests.mpi." + jacvec_field + "/rhs",
                                                       block_name, jacvec_field, Real(1)});
  runtime.install_field_plan(jacvec_field, jacvec_plan);
  runtime.register_named_field(block_name, jacvec_field, 0, 1, 2,
                               /*gradient_sign=*/-1);
  runtime.set_block_named_elliptic_rhs(
      block_index, jacvec_field,
      [](const MultiFab& state, MultiFab& rhs) { add_scaled_component(state, Real(1), 0, rhs); });

  require(consume_solved(runtime.solve_named_fields(&jacvec_field), "baseline").solved(),
          "baseline consumption");
  for (int level = 0; level < runtime.nlev(); ++level) {
    const ::pops::runtime::multiblock::BoundaryEvaluationPoint point{
        "main",
        31 + 10 * block_index + level,
        level,
        level,
        13,
        ::pops::amr::Rational(1, 2),
        0.01 / static_cast<double>(1 << level),
        0.305};
    MultiFab iterate = runtime.level_state(block_index, level);
    MultiFab direction = iterate;
    scale(direction, Real(0.75));

    require(consume_solved(
                runtime.solve_named_fields_from_state_at(point, jacvec_field, block_index, iterate),
                "base")
                .solved(),
            "base consumption");
    std::vector<MultiFab> base_phi;
    base_phi.reserve(static_cast<std::size_t>(runtime.nlev()));
    for (int provider_level = 0; provider_level < runtime.nlev(); ++provider_level)
      base_phi.emplace_back(runtime.provider_potential_level(jacvec_field, provider_level));

    auto residual_at = [&](Real shift, bool coupled) {
      MultiFab state = iterate;
      saxpy(state, shift, direction);
      MultiFab residual(iterate.box_array(), iterate.dmap(), iterate.ncomp(), 0);
      residual.set_val(Real(0));
      if (coupled) {
        require(consume_solved(runtime.solve_named_fields_from_state_at(point, jacvec_field,
                                                                        block_index, state),
                               "perturbed")
                    .solved(),
                "perturbed consumption");
      }
      runtime.level_rhs_core_into_at(block_index, level, point, state, residual,
                                     /*flux_only=*/false);
      if (coupled) {
        require(consume_solved(runtime.solve_named_fields_from_state_at(point, jacvec_field,
                                                                        block_index, iterate),
                               "restore")
                    .solved(),
                "restore consumption");
      }
      return residual;
    };

    const MultiFab r0 = residual_at(Real(0), /*coupled=*/false);
    const MultiFab plus = residual_at(h, /*coupled=*/true);
    const MultiFab minus = residual_at(-h, /*coupled=*/true);
    const MultiFab stale_plus = residual_at(h, /*coupled=*/false);
    const MultiFab restored_r0 = residual_at(Real(0), /*coupled=*/false);

    MultiFab generated = direction;
    saxpy(generated, -c_dt / h, plus);
    saxpy(generated, c_dt / h, r0);
    MultiFab centered = direction;
    saxpy(centered, -c_dt / (Real(2) * h), plus);
    saxpy(centered, c_dt / (Real(2) * h), minus);
    const Real response = global_max_valid_scalar_diff(centered, direction);
    require(response > Real(1e-7), "field-coupled response");
    require(global_max_valid_scalar_diff(generated, centered) < Real(2e-2) * response + Real(2e-7),
            "centered-difference parity");

    MultiFab stale = direction;
    saxpy(stale, -c_dt / h, stale_plus);
    saxpy(stale, c_dt / h, r0);
    require(global_max_valid_scalar_diff(stale, centered) > Real(1e-7),
            level == 0 ? "L0 rejects a frozen provider" : "L1 rejects a frozen provider");
    for (int provider_level = 0; provider_level < runtime.nlev(); ++provider_level)
      require(global_max_valid_scalar_diff(
                  runtime.provider_potential_level(jacvec_field, provider_level),
                  base_phi[static_cast<std::size_t>(provider_level)]) < Real(1e-8),
              "restores its complete provider hierarchy");
    require(global_max_valid_scalar_diff(restored_r0, r0) < Real(1e-8),
            "restores its residual carrier");
  }
  return failures;
}

long prove_exact_distributed_stage_pack() {
  constexpr int n = 8;
  constexpr int phi_component = kAuxNamedBase;
  long failures = 0;
  const auto require = [&failures](bool condition, std::string_view label) {
    if (!condition) {
      std::fprintf(stderr, "rank %d: exact stage-pack check failed: %.*s\n", my_rank(),
                   static_cast<int>(label.size()), label.data());
      ++failures;
    }
  };

  try {
    AmrBuildParams params;
    params.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
    params.mesh.periodicity = Periodicity{true, true};
    params.mesh.n = n;
    params.mesh.L = 1.0;
    params.mesh.regrid_every = 0;
    params.mesh.distribute_coarse = true;
    params.mesh.coarse_max_grid = n / 2;
    params.poisson.bc = BCRec{};
    detail::SharedAmrLayout layout = detail::make_shared_amr_layout(params);

    // Exercise a genuinely distributed stage pack on both materialized levels. The ordinary
    // bootstrap fine seed is one patch; replace it with full-domain tiles so np=2 owns live and
    // staged pieces on L0 and L1 instead of merely carrying empty local views on one rank.
    layout.dm[0] = DistributionMapping(layout.ba[0].size(), n_ranks());
    layout.dm_coarse = layout.dm[0];
    const Box2D fine_domain = layout.geom.domain.refine(kAmrRefRatio);
    layout.ba[1] = BoxArray::from_domain(fine_domain, n);
    layout.dm[1] = DistributionMapping(layout.ba[1].size(), n_ranks());

    std::vector<AmrRuntimeBlock> blocks;
    blocks.push_back(detail::dispatch_amr_block(stage_pack_model(), "minmod", "rusanov", layout,
                                                "a", stage_pack_density(n, 0.35),
                                                /*has_density=*/true, 1.4, 1, false));
    blocks.push_back(detail::dispatch_amr_block(stage_pack_model(), "minmod", "rusanov", layout,
                                                "b", stage_pack_density(n, 0.65),
                                                /*has_density=*/true, 1.4, 1, false));
    for (AmrRuntimeBlock& block : blocks)
      block.aux_ncomp = phi_component + 1;

    int rhs_assembly_calls = 0;
    AmrRuntime runtime(layout.geom, layout.runtime_hierarchy(), layout.poisson_bc,
                       std::move(blocks), layout.base_per, layout.replicated_coarse, layout.wall);
    test::install_second_order_amr_transfer_authorities(runtime, 2);
    runtime.set_parent_child_temporal_relations({::pops::amr::ParentChildClockRelation(
        0, 1, ::pops::amr::Rational(2, 1), ::pops::amr::RemainderPolicy::IntegralOnly)});

    AmrFieldSolveConfig plan;
    plan.solver_options =
        geometric_mg_amr_field_solver_options(GeometricMgOptions{}, CompositeFacOptions{});
    plan.plan_identity = "tests.mpi.stage-pack.coupled-screened.plan@1";
    plan.provider_identity = "tests.mpi.stage-pack.coupled-screened";
    plan.topology_provider_kind = "structured";
    plan.topology_provenance = "tests.mpi.periodic-cartesian";
    plan.topology_digest = "tests.mpi.periodic-cartesian.full-refinement@1";
    plan.output_owner_identity = "tests.mpi.stage-pack.a";
    plan.output_block = "a";
    plan.output_key = "coupled_screened";
    plan.hierarchy_policy = level_local_hierarchy_policy();
    plan.nullspace = operator_topology_zero_mean_nullspace();
    plan.has_reaction = true;
    plan.reaction = Real(2);
    plan.providers.push_back(
        FieldProviderBinding{"tests.mpi.stage-pack.a/rhs", "a", "coupled_screened", Real(1)});
    plan.providers.push_back(
        FieldProviderBinding{"tests.mpi.stage-pack.b/rhs", "b", "coupled_screened", Real(1)});
    runtime.install_field_plan("coupled_screened", plan);
    runtime.register_named_field("a", "coupled_screened", phi_component,
                                 /*gx=*/-1, /*gy=*/-1, /*gradient_sign=*/Real(1));
    runtime.set_block_named_elliptic_rhs(
        0, "coupled_screened", [&rhs_assembly_calls](const MultiFab& state, MultiFab& rhs) {
          ++rhs_assembly_calls;
          add_scaled_component(state, Real(1), 0, rhs);
        });
    runtime.set_block_named_elliptic_rhs(
        1, "coupled_screened", [&rhs_assembly_calls](const MultiFab& state, MultiFab& rhs) {
          ++rhs_assembly_calls;
          add_scaled_component(state, Real(1), 0, rhs);
        });

    const std::string field = "coupled_screened";
    {
      SolveOutcome baseline = runtime.solve_named_fields(&field);
      require(baseline.report().solved(), "baseline report");
      require(baseline.consume(SolveConsumption::kAccept).solved(), "baseline consumption");
    }
    require(runtime.nlev() == 2, "two materialized levels");

    for (int level = 0; level < runtime.nlev(); ++level) {
      const MultiFab& live_a = runtime.level_state(0, level);
      const MultiFab& live_b = runtime.level_state(1, level);
      require(mapping_is_distributed_across_two_ranks(live_a.dmap()), "block a distributed");
      require(mapping_is_distributed_across_two_ranks(live_b.dmap()), "block b distributed");
      require(live_a.local_size() > 0, "block a has a local piece");
      require(live_b.local_size() > 0, "block b has a local piece");

      const MultiFab base_phi = runtime.provider_potential_level(field, level);
      const MultiFab accepted_a = live_a;
      const MultiFab accepted_b = live_b;
      MultiFab stage_a = accepted_a;
      MultiFab stage_b = accepted_b;
      add_valid_constant(stage_a, Real(0.05));
      add_valid_constant(stage_b, Real(0.08));
      const ::pops::runtime::multiblock::BoundaryEvaluationPoint point{
          "main",
          17,
          level,
          level,
          29,
          ::pops::amr::Rational(1, 2),
          0.01 / static_cast<double>(1 << level),
          0.205};

      auto solve_and_accept = [&](const std::vector<const MultiFab*>& stages) {
        const MultiFab visible_before = runtime.provider_potential_level(field, level);
        const int assemblies_before = rhs_assembly_calls;
        SolveOutcome pending = runtime.solve_named_fields_from_states_at(point, field, stages);
        require(rhs_assembly_calls == assemblies_before + 2 * runtime.nlev(),
                "successful request assembled every block and level");
        require(global_max_allocated_diff(runtime.level_state(0, level), accepted_a) == Real(0),
                "block a live state restored before consumption");
        require(global_max_allocated_diff(runtime.level_state(1, level), accepted_b) == Real(0),
                "block b live state restored before consumption");
        require(global_max_allocated_diff(runtime.provider_potential_level(field, level),
                                          visible_before) == Real(0),
                "candidate private before consumption");
        require(pending.report().solved(), "stage-pack report solved");
        require(pending.consume(SolveConsumption::kAccept).solved(), "stage-pack result consumed");
        require(global_max_allocated_diff(runtime.level_state(0, level), accepted_a) == Real(0),
                "block a live state restored after consumption");
        require(global_max_allocated_diff(runtime.level_state(1, level), accepted_b) == Real(0),
                "block b live state restored after consumption");
        return MultiFab(runtime.provider_potential_level(field, level));
      };

      std::vector<const MultiFab*> stages(2, nullptr);
      stages[0] = &stage_a;
      const MultiFab only_a = solve_and_accept(stages);
      stages[0] = nullptr;
      stages[1] = &stage_b;
      const MultiFab only_b = solve_and_accept(stages);
      stages[0] = &stage_a;
      const MultiFab both = solve_and_accept(stages);

      const auto [superposition_error, response] =
          global_stage_pack_superposition_error(both, only_a, only_b, base_phi);
      require(response > Real(1e-7), "both stage states contribute");
      require(superposition_error < Real(5e-4) * response + Real(1e-10),
              "stage-pack superposition");

      stages[0] = &accepted_a;
      stages[1] = &accepted_b;
      const MultiFab restored = solve_and_accept(stages);
      require(global_max_valid_scalar_diff(restored, base_phi) < Real(1e-8),
              "accepted stage pack restores provider result");
    }

    // This runtime deliberately de-replicates both L0 and L1. The builtin composite provider
    // refuses that ownership contract, so this route proves only the level-local policy here.
    failures += prove_field_jacvec_route(
        runtime, "distributed_jacvec", "a", 0, level_local_hierarchy_policy(),
        "tests.mpi.periodic-cartesian.full-refinement@1", "distributed level-local JVP");

    // These request bytes are collective inputs. Keep every local request structurally valid so
    // each mismatch reaches the exact consensus, then prove no solver or publication ran.
    const int level = 0;
    const MultiFab accepted_a = runtime.level_state(0, level);
    const MultiFab accepted_b = runtime.level_state(1, level);
    MultiFab stage_a = accepted_a;
    MultiFab stage_b = accepted_b;
    add_valid_constant(stage_a, Real(0.03));
    add_valid_constant(stage_b, Real(0.04));
    const ::pops::runtime::multiblock::BoundaryEvaluationPoint common_point{
        "main", 41, level, 0, 7, ::pops::amr::Rational(1, 2), 0.01, 0.41};
    const MultiFab visible_before = runtime.provider_potential_level(field, level);
    const int assemblies_before = rhs_assembly_calls;

    auto require_collective_pre_solve_rejection =
        [&](const ::pops::runtime::multiblock::BoundaryEvaluationPoint& point,
            const std::string& provider, const std::vector<const MultiFab*>& stages) {
          bool rejected = false;
          bool exact_error = false;
          try {
            (void)runtime.solve_named_fields_from_states_at(point, provider, stages);
          } catch (const std::invalid_argument& error) {
            rejected = true;
            exact_error =
                std::string_view(error.what()) ==
                "AmrRuntime::solve_named_fields_from_states_at request differs between MPI ranks";
          } catch (...) {
          }
          require(rejected, "divergent request rejected collectively");
          require(exact_error, "divergent request exact diagnostic");
          require(rhs_assembly_calls == assemblies_before,
                  "divergence refused before RHS assembly and solve");
          require(!runtime.field_solve_transaction_active(), "divergence leaves no transaction");
          require(global_max_allocated_diff(runtime.level_state(0, level), accepted_a) == Real(0),
                  "divergence preserves block a live state");
          require(global_max_allocated_diff(runtime.level_state(1, level), accepted_b) == Real(0),
                  "divergence preserves block b live state");
          require(global_max_allocated_diff(runtime.provider_potential_level(field, level),
                                            visible_before) == Real(0),
                  "divergence preserves published provider");
        };

    {
      auto point = common_point;
      if (my_rank() == 1)
        ++point.stage;
      require_collective_pre_solve_rejection(point, field, {&stage_a, &stage_b});
    }
    {
      const std::string provider = my_rank() == 1 ? "rank-one-provider" : field;
      require_collective_pre_solve_rejection(common_point, provider, {&stage_a, &stage_b});
    }
    {
      std::vector<const MultiFab*> stages{&stage_a, &stage_b};
      if (my_rank() == 1)
        stages[1] = nullptr;
      require_collective_pre_solve_rejection(common_point, field, stages);
    }
  } catch (const std::exception& error) {
    if (my_rank() == 0)
      std::fprintf(stderr, "exact distributed stage-pack proof failed: %s\n", error.what());
    ++failures;
  } catch (...) {
    if (my_rank() == 0)
      std::fprintf(stderr, "exact distributed stage-pack proof failed with an unknown error\n");
    ++failures;
  }
  return failures;
}

long prove_replicated_coarse_composite_jvp() {
  constexpr int n = 8;
  long failures = 0;
  const auto require = [&failures](bool condition, std::string_view label) {
    if (!condition) {
      std::fprintf(stderr, "rank %d: replicated-coarse composite JVP failed: %.*s\n", my_rank(),
                   static_cast<int>(label.size()), label.data());
      ++failures;
    }
  };

  try {
    AmrBuildParams params;
    params.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
    params.mesh.periodicity = Periodicity{true, true};
    params.mesh.n = n;
    params.mesh.regrid_every = 0;
    params.mesh.distribute_coarse = false;
    detail::SharedAmrLayout layout = detail::make_shared_amr_layout(params);

    // CompositeFAC's current MPI contract keeps a complete coarse copy on every rank while the
    // refined level is genuinely partitioned. Tile the central fine seed so both ranks own live
    // pieces while uncovered L0 cells continue to exercise the coarse part of the composite solve.
    const Box2D fine_region = layout.ba[1].boxes().front();
    layout.ba[1] = BoxArray::from_domain(fine_region, n / 2);
    layout.dm[1] = DistributionMapping(layout.ba[1].size(), n_ranks());
    require(layout.replicated_coarse, "coarse ownership is explicitly replicated");
    require(mapping_is_distributed_across_two_ranks(layout.dm[1]), "L1 mapping is distributed");

    std::vector<AmrRuntimeBlock> blocks;
    blocks.push_back(detail::dispatch_amr_block(stage_pack_model(), "minmod", "rusanov", layout,
                                                "composite", stage_pack_density(n, 0.5),
                                                /*has_density=*/true, 1.4, 1, false));
    blocks.back().aux_ncomp = kAuxNamedBase + 1;

    AmrRuntime runtime(layout.geom, layout.runtime_hierarchy(), layout.poisson_bc,
                       std::move(blocks), layout.base_per, layout.replicated_coarse, layout.wall);
    test::install_second_order_amr_transfer_authorities(runtime, 1);
    runtime.set_parent_child_temporal_relations({::pops::amr::ParentChildClockRelation(
        0, 1, ::pops::amr::Rational(2, 1), ::pops::amr::RemainderPolicy::IntegralOnly)});

    require(runtime.nlev() == 2, "composite hierarchy has L0/L1");
    require(runtime.level_state(0, 0).local_size() > 0,
            "each rank owns its replicated coarse copy");
    require(runtime.level_state(0, 1).local_size() > 0, "each rank owns a fine piece");
    failures += prove_field_jacvec_route(runtime, "replicated_coarse_composite_jacvec", "composite",
                                         0, composite_hierarchy_policy(),
                                         "tests.mpi.periodic-cartesian.central-refinement@1",
                                         "replicated-coarse distributed-fine composite JVP");
  } catch (const std::exception& error) {
    if (my_rank() == 0)
      std::fprintf(stderr, "replicated-coarse composite JVP proof failed: %s\n", error.what());
    ++failures;
  } catch (...) {
    if (my_rank() == 0)
      std::fprintf(stderr, "replicated-coarse composite JVP proof failed with an unknown error\n");
    ++failures;
  }
  return failures;
}

long prove_distributed_physical_boundary_jvp() {
  constexpr int n = 8;
  constexpr int phi_component = kAuxNamedBase;
  constexpr Real c_dt = Real(0.01);
  constexpr Real h = Real(2e-4);
  const std::string state_identity = "tests://mpi/physical-boundary/state/a";
  const std::string field_identity = "tests://mpi/physical-boundary/field/jacvec";
  const std::string field = "distributed_boundary_jacvec";
  long failures = 0;
  const auto require = [&failures](bool condition, std::string_view label) {
    if (!condition) {
      std::fprintf(stderr, "rank %d: distributed physical-boundary JVP failed: %.*s\n", my_rank(),
                   static_cast<int>(label.size()), label.data());
      ++failures;
    }
  };

  try {
    AmrBuildParams params;
    params.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
    params.mesh.periodicity = Periodicity{false, false};
    params.mesh.n = n;
    params.mesh.L = 1.0;
    params.mesh.regrid_every = 0;
    params.mesh.distribute_coarse = true;
    params.mesh.coarse_max_grid = n / 2;
    BCRec physical_field_bc;
    physical_field_bc.xlo = physical_field_bc.xhi = BCType::Dirichlet;
    physical_field_bc.ylo = physical_field_bc.yhi = BCType::Dirichlet;
    params.poisson.bc = physical_field_bc;
    detail::SharedAmrLayout layout = detail::make_shared_amr_layout(params);

    layout.dm[0] = split_xlow_face_across_ranks(layout.ba[0], layout.geom.domain);
    layout.dm_coarse = layout.dm[0];
    const Box2D fine_domain = layout.geom.domain.refine(kAmrRefRatio);
    layout.ba[1] = BoxArray::from_domain(fine_domain, n);
    layout.dm[1] = split_xlow_face_across_ranks(layout.ba[1], fine_domain);
    require(mapping_is_distributed_across_two_ranks(layout.dm[0]),
            "physical-boundary L0 is distributed");
    require(mapping_is_distributed_across_two_ranks(layout.dm[1]),
            "physical-boundary L1 is distributed");
    require(
        xlow_face_is_distributed_across_two_ranks(layout.ba[0], layout.dm[0], layout.geom.domain),
        "physical x-low face is split across ranks on L0");
    require(xlow_face_is_distributed_across_two_ranks(layout.ba[1], layout.dm[1], fine_domain),
            "physical x-low face is split across ranks on L1");

    BCRec transport_bc;
    transport_bc.xlo = transport_bc.xhi = BCType::Foextrap;
    transport_bc.ylo = transport_bc.yhi = BCType::Foextrap;
    auto boundary_plan = std::make_shared<PreparedBoundaryPlan>(
        "tests://mpi/physical-boundary/plan", 1, std::vector<BCRec>{transport_bc},
        std::vector<int>{}, state_identity, PreparedBoundaryReadDependencies{{}, {field_identity}});
    const PreparedBoundaryFieldRead field_read = boundary_plan->prepare_field_read(field_identity);
    std::map<std::string, std::shared_ptr<PreparedBoundaryPlan>> boundary_plans{
        {"a", boundary_plan}};
    layout.boundary_plans = &boundary_plans;

    std::vector<AmrRuntimeBlock> blocks;
    blocks.push_back(detail::dispatch_amr_block(stage_pack_model(), "minmod", "rusanov", layout,
                                                "a", stage_pack_density(n, 0.5),
                                                /*has_density=*/true, 1.4, 1, false));
    blocks.back().aux_ncomp = phi_component + 1;
    blocks.back().state_identity = state_identity;
    blocks.back().level_boundary_residual_at_point_prepared =
        [field_read](const ::pops::runtime::multiblock::BoundaryEvaluationPoint& point,
                     MultiFab& state, const MultiFab&, const Geometry& geometry, MultiFab& residual,
                     const PreparedGridBoundarySession& boundary) {
          const PreparedBoundaryReadView reads = boundary.bind_reads(point, state);
          const MultiFab& solved_field = reads.field(field_read);
          for (int local = 0; local < residual.local_size(); ++local) {
            const int field_local = solved_field.local_index_of(residual.global_index(local));
            if (field_local < 0)
              throw std::logic_error(
                  "distributed physical boundary lost co-distributed field ownership");
            const Box2D valid = residual.box(local);
            if (valid.lo[0] > geometry.domain.lo[0] || valid.hi[0] < geometry.domain.lo[0])
              continue;
            const ConstArray4 phi = solved_field.fab(field_local).const_array();
            const Array4 output = residual.fab(local).array();
            const int i = geometry.domain.lo[0];
            for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
              output(i, j, 0) += Real(100) * phi(i, j, 0);
          }
        };

    AmrRuntime runtime(layout.geom, layout.runtime_hierarchy(), layout.poisson_bc,
                       std::move(blocks), layout.base_per, layout.replicated_coarse, layout.wall);
    test::install_second_order_amr_transfer_authorities(runtime, 1);
    runtime.set_parent_child_temporal_relations({::pops::amr::ParentChildClockRelation(
        0, 1, ::pops::amr::Rational(2, 1), ::pops::amr::RemainderPolicy::IntegralOnly)});

    AmrFieldSolveConfig plan;
    plan.solver_options =
        geometric_mg_amr_field_solver_options(GeometricMgOptions{}, CompositeFacOptions{});
    plan.plan_identity = "tests.mpi.distributed-boundary-jacvec.plan@1";
    plan.provider_identity = "tests.mpi.distributed-boundary-jacvec";
    plan.topology_provider_kind = "structured";
    plan.topology_provenance = "tests.mpi.physical-cartesian";
    plan.topology_digest = "tests.mpi.physical-cartesian.full-refinement@1";
    plan.output_owner_identity = "tests.mpi.physical-boundary.a";
    plan.output_block = "a";
    plan.output_key = field;
    plan.hierarchy_policy = level_local_hierarchy_policy();
    plan.nullspace = operator_topology_zero_mean_nullspace();
    plan.has_reaction = true;
    plan.reaction = Real(2);
    plan.providers.push_back(
        FieldProviderBinding{"tests.mpi.distributed-boundary-jacvec/rhs", "a", field, Real(1)});
    runtime.install_field_plan(field, plan);
    runtime.register_named_field("a", field, 0, 1, 2, /*gradient_sign=*/-1);
    runtime.set_block_named_elliptic_rhs(0, field, [](const MultiFab& state, MultiFab& rhs) {
      add_scaled_component(state, Real(1), 0, rhs);
    });
    runtime.install_boundary_storage_routes({{field_identity, field}});

    require(runtime.nlev() == 2, "physical-boundary hierarchy has L0/L1");
    for (int level = 0; level < runtime.nlev(); ++level) {
      const ::pops::runtime::multiblock::BoundaryEvaluationPoint point{
          "main",
          71 + level,
          level,
          0,
          17,
          ::pops::amr::Rational(1, 2),
          0.01 / static_cast<double>(1 << level),
          0.405};
      MultiFab iterate = runtime.level_state(0, level);
      MultiFab direction = iterate;
      scale(direction, Real(0.75));

      {
        SolveOutcome base = runtime.solve_named_fields_from_state_at(point, field, 0, iterate);
        require(base.report().solved(), "physical-boundary base report");
        require(base.consume(SolveConsumption::kAccept).solved(),
                "physical-boundary base consumption");
      }
      const MultiFab base_phi = runtime.provider_potential_level(field, level);

      auto residual_at = [&](Real shift, bool coupled, bool include_boundary) {
        MultiFab state = iterate;
        saxpy(state, shift, direction);
        MultiFab residual(iterate.box_array(), iterate.dmap(), iterate.ncomp(), 0);
        residual.set_val(Real(0));
        if (coupled) {
          SolveOutcome perturbed = runtime.solve_named_fields_from_state_at(point, field, 0, state);
          require(perturbed.report().solved(), "physical-boundary perturbed report");
          require(perturbed.consume(SolveConsumption::kAccept).solved(),
                  "physical-boundary perturbed consumption");
        }
        if (include_boundary)
          runtime.level_rhs_into_at(0, level, point, state, residual);
        else
          runtime.level_rhs_core_into_at(0, level, point, state, residual, /*flux_only=*/false);
        if (coupled) {
          SolveOutcome restored =
              runtime.solve_named_fields_from_state_at(point, field, 0, iterate);
          require(restored.report().solved(), "physical-boundary restore report");
          require(restored.consume(SolveConsumption::kAccept).solved(),
                  "physical-boundary restore consumption");
        }
        return residual;
      };

      const MultiFab r0 = residual_at(Real(0), /*coupled=*/false, /*include_boundary=*/true);
      const MultiFab plus = residual_at(h, /*coupled=*/true, /*include_boundary=*/true);
      const MultiFab minus = residual_at(-h, /*coupled=*/true, /*include_boundary=*/true);
      const MultiFab stale_plus = residual_at(h, /*coupled=*/false, /*include_boundary=*/true);
      const MultiFab plus_core = residual_at(h, /*coupled=*/true, /*include_boundary=*/false);
      const MultiFab minus_core = residual_at(-h, /*coupled=*/true, /*include_boundary=*/false);
      const MultiFab stale_plus_core =
          residual_at(h, /*coupled=*/false, /*include_boundary=*/false);
      const MultiFab restored_r0 =
          residual_at(Real(0), /*coupled=*/false, /*include_boundary=*/true);

      MultiFab generated = direction;
      saxpy(generated, -c_dt / h, plus);
      saxpy(generated, c_dt / h, r0);
      MultiFab centered = direction;
      saxpy(centered, -c_dt / (Real(2) * h), plus);
      saxpy(centered, c_dt / (Real(2) * h), minus);
      const Real response = global_max_valid_scalar_diff(centered, direction);
      require(response > Real(1e-7), "physical-boundary field-coupled JVP response");
      require(
          global_max_valid_scalar_diff(generated, centered) < Real(2e-2) * response + Real(2e-7),
          "physical-boundary field-coupled JVP centered-difference parity");

      MultiFab centered_core = direction;
      saxpy(centered_core, -c_dt / (Real(2) * h), plus_core);
      saxpy(centered_core, c_dt / (Real(2) * h), minus_core);
      require(global_max_valid_scalar_diff(centered, centered_core) > Real(1e-8),
              "physical boundary affects distributed JVP");

      MultiFab coupled_boundary = plus;
      saxpy(coupled_boundary, Real(-1), plus_core);
      MultiFab stale_boundary = stale_plus;
      saxpy(stale_boundary, Real(-1), stale_plus_core);
      require(global_max_valid_scalar_diff(coupled_boundary, stale_boundary) > Real(1e-8),
              "frozen provider changes distributed physical boundary");

      const auto [boundary_support, interior_support] =
          global_physical_boundary_support(coupled_boundary, runtime.level_geom(level).domain);
      require(boundary_support > Real(1e-8), "distributed physical-boundary contribution exists");
      require(interior_support < Real(1e-13),
              "distributed physical-boundary contribution remains face-local");
      require(global_max_valid_scalar_diff(runtime.provider_potential_level(field, level),
                                           base_phi) < Real(1e-8),
              "distributed physical-boundary provider restores");
      require(global_max_valid_scalar_diff(restored_r0, r0) < Real(1e-8),
              "distributed physical-boundary residual carrier restores");
    }
  } catch (const std::exception& error) {
    if (my_rank() == 0)
      std::fprintf(stderr, "distributed physical-boundary JVP proof failed: %s\n", error.what());
    ++failures;
  } catch (...) {
    if (my_rank() == 0)
      std::fprintf(stderr,
                   "distributed physical-boundary JVP proof failed with an unknown error\n");
    ++failures;
  }
  return failures;
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

  failures += prove_exact_distributed_stage_pack();
  failures += prove_replicated_coarse_composite_jvp();
  failures += prove_distributed_physical_boundary_jvp();

  // A hierarchy provider cannot split publication by returning individually valid but different
  // reports. Both outcome divergence and equal-length reason-byte divergence are rejected with one
  // uniform error on every rank; an identical report remains publishable.
  {
    ConsensusHierarchyPrepared solver(SolveReportFault::None);
    try {
      SolveOutcome outcome = runtime::program::solve_prepared_hierarchy_tensor_collectively(
          solver, {Real(1.0e-8), Real(0), 4});
      const SolveReport report = outcome.consume(SolveConsumption::kAccept);
      require(report.solved());
      require(report.reason == "collective-solved");
    } catch (...) {
      require(false);
    }
  }
  for (const SolveReportFault fault :
       {SolveReportFault::OutcomeOnRankOne, SolveReportFault::ReasonBytesOnRankOne}) {
    ConsensusHierarchyPrepared solver(fault);
    bool rejected = false;
    bool exact_error = false;
    try {
      (void)runtime::program::solve_prepared_hierarchy_tensor_collectively(
          solver, {Real(1.0e-8), Real(0), 4});
    } catch (const std::runtime_error& error) {
      rejected = true;
      exact_error = std::string_view(error.what()) ==
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
      SolveOutcome outcome = solve_prepared_amr_field_solver_collectively(solver);
      const SolveReport report = outcome.consume(SolveConsumption::kAccept);
      require(report.solved());
      require(report.reason == "collective-solved");
    } catch (...) {
      require(false);
    }
  }
  for (const SolveReportFault fault :
       {SolveReportFault::OutcomeOnRankOne, SolveReportFault::ReasonBytesOnRankOne}) {
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
  {
    ConsensusAmrFieldPrepared solver(SolveReportFault::ThrowWithStaleReject);
    bool rejected = false;
    bool exact_error = false;
    try {
      (void)solve_prepared_amr_field_solver_collectively(solver);
    } catch (const std::runtime_error& error) {
      rejected = true;
      exact_error = std::string_view(error.what()) ==
                    "AMR field-solver provider failed on at least one MPI rank";
    } catch (...) {
    }
    require(rejected);
    require(exact_error);
  }
  {
    SolveReport solved;
    solved.mark_solved("rank-local-publication");
    RankLocalPublication publication{rank == 0, false};
    SolveOutcome outcome = SolveOutcome::collective_world(
        std::move(solved), SolveOutcome::PublicationHooks{&publication,
                                                          &RankLocalPublication::accept,
                                                          nullptr,
                                                          nullptr,
                                                          {},
                                                          &RankLocalPublication::validate});
    bool rejected = false;
    bool exact_error = false;
    try {
      (void)outcome.consume(SolveConsumption::kAccept);
    } catch (const std::logic_error& error) {
      rejected = true;
      exact_error = std::string_view(error.what()) ==
                    "SolveOutcome accept validation failed on at least one MPI rank";
    } catch (...) {
    }
    require(rejected);
    require(exact_error);
    require(!publication.accepted);

    publication.layout_valid = true;
    require(outcome.consume(SolveConsumption::kAccept).solved());
    require(publication.accepted);
  }

  // Consumption is itself a publication collective. A rank-divergent action is rejected before any
  // rank can accept/reject, and the same intact outcome can then be consumed consistently.
  {
    ConsensusHierarchyPrepared solver(SolveReportFault::None);
    SolveOutcome outcome = runtime::program::solve_prepared_hierarchy_tensor_collectively(
        solver, {Real(1.0e-8), Real(0), 4});
    bool rejected = false;
    try {
      (void)outcome.consume(rank == 0 ? SolveConsumption::kAccept : SolveConsumption::kFailRun);
    } catch (const std::logic_error& error) {
      rejected = std::string_view(error.what()) ==
                 "SolveOutcome consumption action differs between MPI ranks";
    } catch (...) {
    }
    require(rejected);
    require(outcome.consume(SolveConsumption::kAccept).solved());
  }

  {
    ConsensusAmrFieldPrepared solver(SolveReportFault::None);
    SolveOutcome outcome = solve_prepared_amr_field_solver_collectively(solver);
    bool rejected = false;
    try {
      (void)outcome.consume(rank == 0 ? SolveConsumption::kAccept
                                      : SolveConsumption::kRejectAttempt);
    } catch (const std::logic_error& error) {
      rejected = std::string_view(error.what()) ==
                 "SolveOutcome consumption action differs between MPI ranks";
    } catch (...) {
    }
    require(rejected);
    require(outcome.consume(SolveConsumption::kAccept).solved());
  }
  {
    ConsensusAmrFieldPrepared solver(SolveReportFault::None);
    SolveOutcome outcome = solve_prepared_amr_field_solver_collectively(solver);
    solver.set_phi_layout_drift(rank == 0);
    bool rejected = false;
    bool exact_error = false;
    try {
      (void)outcome.consume(SolveConsumption::kAccept);
    } catch (const std::logic_error& error) {
      rejected = true;
      exact_error = std::string_view(error.what()) ==
                    "SolveOutcome accept validation failed on at least one MPI rank";
    } catch (...) {
    }
    require(rejected);
    require(exact_error);
    solver.set_phi_layout_drift(false);
    require(outcome.consume(SolveConsumption::kAccept).solved());
  }
  {
    SolveReport divergent = make_consensus_report(SolveReportFault::OutcomeOnRankOne);
    RankLocalPublication publication;
    SolveOutcome outcome = SolveOutcome::collective_world(
        std::move(divergent),
        SolveOutcome::PublicationHooks{&publication, &RankLocalPublication::accept,
                                       &RankLocalPublication::reject, nullptr});
    bool rejected = false;
    bool exact_error = false;
    try {
      (void)outcome.consume(SolveConsumption::kAccept);
    } catch (const std::logic_error& error) {
      rejected = true;
      exact_error = std::string_view(error.what()) ==
                    "SolveOutcome report disposition differs between MPI ranks";
    } catch (...) {
    }
    require(rejected);
    require(exact_error);
    require(!publication.accepted);
    require(publication.rejected);
    bool already_consumed = false;
    try {
      (void)outcome.consume(SolveConsumption::kAccept);
    } catch (const std::logic_error& error) {
      already_consumed = std::string_view(error.what()) == "SolveOutcome has already been consumed";
    } catch (...) {
    }
    require(already_consumed);
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
          {"plasma"}, {"potential"}, {1.0}, "geometric_mg", composite_hierarchy_policy(),
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
