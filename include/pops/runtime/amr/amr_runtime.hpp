#pragma once

#include <pops/amr/regridding/regrid.hpp>  // tag_cells, grow_tags (per-block tags + phi for the union regrid)
#include <pops/amr/tagging/tag_box.hpp>  // TagBox, tag_union (cell-by-cell OR of the tags of all blocks)
#include <pops/amr/tagging/tagging_truth.hpp>
#include <pops/core/state/state.hpp>  // kAuxBaseComps
#include <pops/core/state/variables.hpp>  // VariableSet, VariableRole, role_from_name (role -> component of coupled sources)
#include <pops/coupling/amr/amr_coupler_mp.hpp>  // detail::coupler_inject_aux_mb (aux injection coarse->fine)
#include <pops/coupling/amr/amr_regrid_coupler.hpp>  // regrid_compute_fine_layout + regrid_field_on_layout (split bricks)
#include <pops/coupling/system/amr_system_coupler.hpp>  // detail::same_layout_or_throw (shared-layout guard)
#include <pops/coupling/base/aux_fill.hpp>  // detail::derive_aux_bc (BC of the aux channel)
#include <pops/coupling/source/coupled_source_program.hpp>  // CoupledSourceKernel + CsProgram (flat ABI, P5 bytecode)
#include <pops/coupling/source/coupling_operator.hpp>  // CouplingOperator / CouplingOperatorView (typed contract, ADC-595)
#include <pops/numerics/elliptic/interface/elliptic_problem.hpp>  // field_postprocess, FieldPostProcess
#include <pops/numerics/elliptic/interface/elliptic_solver.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_bc_rec_adapter.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_builtins.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_prepare.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_workspace.hpp>
#include <pops/numerics/elliptic/interface/field_provider.hpp>
#include <pops/numerics/elliptic/linear/pure_field_algebra.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/numerics/time/amr/reflux/amr_reflux_mf.hpp>  // AmrLevelMP, mf_average_down_mb
#include <pops/numerics/time/amr/levels/amr_clock.hpp>
#include <pops/numerics/time/amr/advance/amr_advance.hpp>  // PreparedAmrTemporalPlan
#include <pops/numerics/time/integrators/implicit_stepper.hpp>  // NewtonReport (OPT-IN IMEX diagnostics, aggregated per block)
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/patch_box.hpp>  // PatchBox: index-space signature of a fine patch (patch_boxes())
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/mf_arith.hpp>  // reduce_sum / reduce_abs_sum / reduce_min / reduce_max / dot / norm_inf (composite_reduce)
#include <pops/mesh/layout/copy_schedule.hpp>  // copy_schedule_{hit,miss}_count (ADC-607 counters)
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/runtime/analytic/initial_materialization.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/parallel/comm.hpp>  // n_ranks() / comm_active(): MPI message+reduction counts (Spec 5 criterion 43)
#include <pops/parallel/solve_report_consensus.hpp>
#include <pops/runtime/numerical_defaults.hpp>
#include <pops/runtime/amr/bootstrap_transfer_builtins.hpp>
#include <pops/runtime/amr/bootstrap_transfer_registry.hpp>
#include <pops/runtime/amr/field_solver_options.hpp>
#include <pops/runtime/amr/hierarchy_policy_authority.hpp>
#include <pops/runtime/amr/prepared_component_providers.hpp>
#include <pops/runtime/amr/prepared_tagging_execution.hpp>
#include <pops/runtime/export.hpp>
#include <pops/runtime/output_piece.hpp>
#include <pops/runtime/multiblock/interface_flux_scheduler.hpp>
#include <pops/runtime/context/grid_context.hpp>
#include <pops/runtime/program/profiler.hpp>  // Profiler / ProfileScope: AMR phase timings (Spec 5 criterion 43, ADC-479)

#include <algorithm>  // std::max (substeps/stride-aware CFL step)
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>  // AmrPhaseScope wall-clock timing (Spec 5 criterion 43)
#include <cmath>   // std::isfinite (reject a degenerate dt)
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>  // std::numeric_limits (initial dt = +inf, min over the blocks)
#include <map>     // static_aux_: externally supplied aux fields, re-applied each solve
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// @file
/// @brief AMR multi-block engine at RUNTIME (type-erased registry keyed by name).
///
/// Runtime counterpart of System::Impl (python/system.cpp): where System type-erases the species
/// (struct Species) on a SINGLE-LEVEL grid, AmrRuntime type-erases N blocks on a SHARED AMR
/// hierarchy. It FAITHFULLY reproduces the AmrSystemCoupler::solve_fields / step algorithm
/// (include/pops/coupling/amr_system_coupler.hpp), but over type-erased closures (the runtime facade
/// does not know the blocks' Model/Limiter/Flux types at compile time) rather than over a
/// compile-time CoupledSystem<Blocks...>.
///
/// INVARIANTS (multi-block capstone, docs/AMR_MULTIBLOCK_DESIGN.md):
///  - ONE single shared AMR hierarchy (AmrHierarchyLayout, same_layout_or_throw guard): all
///    blocks live on EXACTLY the same BoxArray + DistributionMapping + dx/dy per level;
///  - ALL blocks live on ALL patches (never a local spatial absence of a block);
///  - SYSTEM Poisson with a SUMMED and CO-LOCATED right-hand side: rhs[coarse] = Sum_b
///    elliptic_rhs_b(U_b) read at the SAME cells of the shared coarse;
///  - aux SHARED per level (phi, grad phi); a single coarse Poisson solve then coarse->fine
///    injection (coupler_inject_aux_mb), exactly like AmrSystemCoupler;
///  - PER-BLOCK conservation (reflux + average_down of the AMR engine, in the advance closure).
///
/// SCOPE (capstone). We carry blocks with potentially DIFFERENT spatial schemes over the FROZEN
/// hierarchy (no regrid: AmrSystemCoupler has none), with per-block MULTIRATE: substeps (explicit
/// substeps) and stride (hold-then-catch-up cadence), honored in step() mirroring
/// AmrSystemCoupler::step (#140). The TEMPORAL TREATMENT is PER BLOCK: explicit (forward-Euler
/// source, carried by the AMR step) OR IMEX (stiff source treated IMPLICITLY by
/// backward_euler_source, transport staying explicit; capstone vii), selected in step().
///
/// IMEX SEMANTICS UNDER substeps (integration decision, follow-up review #184). At substeps=1 &&
/// stride=1 the runtime IMEX branch COINCIDES with AmrSystemCoupler::step (source-free transport + one
/// backward_euler_source over the effective step). FOR substeps>1 the paths DIVERGE DELIBERATELY: the
/// compile-time engine ignores substeps on its IMEX branch (one transport + one implicit_advance over
/// bdt), while the RUNTIME sub-cycles the Lie splitting K=substeps times over bdt/K. This is sound, not
/// a bug: the source-free transport gets CFL-safer, backward-Euler is unconditionally stable at any
/// step, and refining the implicit step brings the stiff relaxation closer to its trajectory. So the
/// runtime is NOT bit-identical to the compile-time engine once substeps>1; it honors substeps like
/// the explicit branch (test_amr_multiblock_imex guards that substeps=4 DIFFERS from substeps=1).

namespace pops {

/// Fully resolved AMR field solve keyed by the digest of its block-qualified provider identity.
/// provider_identity retains that exact readable identity while plan_identity independently commits
/// every resolved field-plan semantic. This POD is installed before block loaders run and contains
/// no authoring object or Python callback.
struct AmrFieldSolveConfig {
  std::string provider_identity;
  std::string plan_identity;
  std::string topology_provider_kind;
  std::string topology_provenance;
  std::string topology_digest;
  std::string output_owner_identity;
  std::string output_block;
  std::string output_key;
  std::vector<FieldProviderBinding> providers;
  std::string solver = "geometric_mg";
  AmrFieldHierarchyPolicyAuthority hierarchy_policy;
  std::string bc = "auto";
  bool has_explicit_bc = false;
  BCRec explicit_bc{};
  bool has_boundary_kernel = false;
  CompiledFieldBoundaryKernel boundary_kernel{};
  std::shared_ptr<std::vector<Real>> boundary_parameters = std::make_shared<std::vector<Real>>();
  std::vector<std::string> boundary_state_blocks;
  std::vector<int> boundary_state_components;
  std::vector<const MultiFab*> boundary_state_buffers;
  std::vector<FieldDistribution> boundary_state_distributions;
  FieldBoundaryExecutionContext boundary_context{};
  bool has_reaction = false;
  Real reaction = Real(0);
  bool has_newton = false;
  FieldNewtonOptions newton{};
  FieldNullspaceProviderSelection nullspace;
  AmrFieldSolverOptions solver_options;
};

/// Canonical native contract of one fully resolved AMR field plan.  The caller-supplied
/// ``plan_identity`` is retained as provenance, never trusted as a substitute for these bytes.
/// Runtime pointers and callable addresses are deliberately excluded; their authenticated semantic
/// identities and every declared dependency/value are included instead.
inline std::string exact_amr_field_solve_config_contract(const AmrFieldSolveConfig& plan) {
  ExactContractBuilder contract;
  contract.text("pops.amr.field-solve-config")
      .scalar(std::uint32_t{1})
      .text(plan.plan_identity)
      .text(plan.provider_identity)
      .text(plan.topology_provider_kind)
      .text(plan.topology_provenance)
      .text(plan.topology_digest)
      .text(plan.output_owner_identity)
      .text(plan.output_block)
      .text(plan.output_key)
      .sequence(plan.providers,
                [](ExactContractBuilder& item, const FieldProviderBinding& value) {
                  item.text(value.identity)
                      .text(value.owner_block)
                      .text(value.native_key)
                      .scalar(value.coefficient);
                })
      .text(plan.solver)
      .bytes(plan.hierarchy_policy.exact_contract())
      .text(plan.bc)
      .scalar(plan.has_explicit_bc);
  if (plan.has_explicit_bc) {
    const BCRec& bc = plan.explicit_bc;
    contract.scalar(bc.xlo)
        .scalar(bc.xhi)
        .scalar(bc.ylo)
        .scalar(bc.yhi)
        .scalar(bc.xlo_val)
        .scalar(bc.xhi_val)
        .scalar(bc.ylo_val)
        .scalar(bc.yhi_val)
        .scalar(bc.xlo_alpha)
        .scalar(bc.xlo_beta)
        .scalar(bc.xhi_alpha)
        .scalar(bc.xhi_beta)
        .scalar(bc.ylo_alpha)
        .scalar(bc.ylo_beta)
        .scalar(bc.yhi_alpha)
        .scalar(bc.yhi_beta)
        .scalar(bc.dx)
        .scalar(bc.dy);
  }
  contract.scalar(plan.has_boundary_kernel);
  if (plan.has_boundary_kernel) {
    contract.text(plan.boundary_kernel.identity)
        .text(plan.boundary_kernel.residual_identity)
        .text(plan.boundary_kernel.jvp_identity)
        .scalar(plan.boundary_kernel.observes_iteration);
  }
  const std::vector<Real> empty_parameters;
  const auto& parameters = plan.boundary_parameters ? *plan.boundary_parameters : empty_parameters;
  contract.sequence(parameters)
      .sequence(plan.boundary_state_blocks,
                [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
      .sequence(plan.boundary_state_components)
      .scalar(plan.has_reaction)
      .scalar(plan.reaction)
      .scalar(plan.has_newton);
  if (plan.has_newton) {
    contract.scalar(plan.newton.tolerance)
        .scalar(static_cast<std::int32_t>(plan.newton.max_iterations))
        .scalar(plan.newton.linear_tolerance)
        .scalar(static_cast<std::int32_t>(plan.newton.linear_max_iterations))
        .scalar(static_cast<std::int32_t>(plan.newton.restart))
        .scalar(plan.newton.armijo)
        .scalar(plan.newton.minimum_step);
  }
  contract.bytes(plan.nullspace.exact_contract()).bytes(plan.solver_options.exact_contract());
  return std::move(contract).release();
}

struct AmrFieldSolverBuildRequest {
  Geometry geometry;
  AmrHierarchyLayout hierarchy;
  BCRec boundary;
  ActiveRegionProvider2D active;
  bool replicated_coarse = false;
  std::string use_contract_identity;
  AmrFieldSolveConfig plan;
};

/// Type-erased, fully materialized hierarchy solver.  Every runtime operation needed after prepare
/// is expressed here, so the AMR core never branches on a concrete solver type.
class AmrPreparedFieldSolver {
 public:
  virtual ~AmrPreparedFieldSolver() = default;
  [[nodiscard]] virtual std::string_view provider_identity() const noexcept = 0;
  [[nodiscard]] virtual std::string_view exact_prepared_contract() const noexcept = 0;
  [[nodiscard]] virtual bool couples_hierarchy_levels() const noexcept = 0;
  [[nodiscard]] virtual int level_count() const noexcept = 0;
  [[nodiscard]] virtual FieldDistribution level_distribution(int level) const = 0;
  virtual MultiFab& rhs_level(int level) = 0;
  virtual MultiFab& phi_level(int level) = 0;
  virtual void set_boundary_context(const FieldBoundaryExecutionContext& context) = 0;
  virtual SolveReport solve() = 0;
  [[nodiscard]] virtual const SolveReport& last_solve_report() const noexcept = 0;
};

/// Execute one provider-owned AMR field solve behind an exact collective publication boundary.
/// Providers may either return a report or throw after recording a typed reject-attempt report;
/// both forms are normalized before the core decides whether to publish or roll back. Unknown
/// exceptions, malformed reports, and valid-but-rank-divergent reports fail uniformly.
inline SolveReport solve_prepared_amr_field_solver_collectively(AmrPreparedFieldSolver& solver) {
  SolveReport report;
  bool provider_failed = false;
  try {
    report = solver.solve();
  } catch (...) {
    const SolveReport& attempted = solver.last_solve_report();
    if (attempted.failed() && attempted.action == SolveAction::kRejectAttempt)
      report = attempted;
    else
      provider_failed = true;
  }
  if (all_reduce_max(provider_failed ? 1L : 0L) != 0)
    throw std::runtime_error("AMR field-solver provider failed on at least one MPI rank");

  const bool malformed = !solve_report_is_publishable(report, std::numeric_limits<int>::max());
  if (all_reduce_max(malformed ? 1L : 0L) != 0)
    throw std::runtime_error("AMR field-solver provider published a malformed SolveReport");

  ExactSolveReportConsensusScratch report_consensus;
  if (!report_consensus.agrees(report))
    throw std::runtime_error("AMR field-solver provider report differs between MPI ranks");
  return report;
}

/// Prepared AMR field solver source.  Providers own construction policy and implementation-specific
/// options; the registry and runtime consume only stable identity and exact contracts. Capability
/// declarations are opaque provider-owned identities; semantic compatibility is decided exclusively
/// by supports(request).
class AmrFieldSolverProvider {
 public:
  virtual ~AmrFieldSolverProvider() = default;
  [[nodiscard]] virtual std::string_view identity() const noexcept = 0;
  [[nodiscard]] virtual std::uint64_t interface_version() const noexcept = 0;
  [[nodiscard]] virtual std::string_view collective_contract() const noexcept = 0;
  [[nodiscard]] virtual std::vector<std::string> capability_contracts() const = 0;
  [[nodiscard]] virtual AmrFieldSolverOptions default_field_options() const = 0;
  /// Optional provider-owned default for a particular use envelope. Providers that only accept
  /// explicit authored policies return std::nullopt; the core never invents or interprets one.
  [[nodiscard]] virtual std::optional<AmrFieldHierarchyPolicyAuthority> default_hierarchy_policy(
      std::string_view use_contract_identity) const = 0;
  [[nodiscard]] virtual PreparedProviderSupport accepts_options(
      const AmrFieldSolverOptions& options) const noexcept = 0;
  [[nodiscard]] virtual PreparedProviderSupport supports(
      const AmrFieldSolverBuildRequest& request) const noexcept = 0;
  [[nodiscard]] virtual std::string expected_prepared_contract(
      const AmrFieldSolverBuildRequest& request) const = 0;
  [[nodiscard]] virtual std::unique_ptr<AmrPreparedFieldSolver> build(
      const AmrFieldSolverBuildRequest& request) const = 0;
};

inline std::string exact_amr_field_solver_provider_declaration(
    const AmrFieldSolverProvider& provider) {
  std::vector<std::string> capabilities = provider.capability_contracts();
  std::sort(capabilities.begin(), capabilities.end());
  if (std::any_of(capabilities.begin(), capabilities.end(),
                  [](const std::string& value) { return value.empty(); }) ||
      std::adjacent_find(capabilities.begin(), capabilities.end()) != capabilities.end())
    throw std::invalid_argument(
        "AMR field solver provider capabilities require unique exact identities");
  ExactContractBuilder contract;
  contract.text("pops.amr.field-solver-provider-declaration")
      .scalar(std::uint32_t{1})
      .text(provider.identity())
      .scalar(provider.interface_version())
      .text(provider.collective_contract())
      .sequence(capabilities, [](ExactContractBuilder& item,
                                 const std::string& capability) { item.text(capability); })
      .bytes(provider.default_field_options().exact_contract());
  return std::move(contract).release();
}

class AmrFieldSolverProviderRegistry {
 public:
  void add(std::shared_ptr<const AmrFieldSolverProvider> provider) {
    if (!provider || provider->identity().empty() || provider->interface_version() == 0 ||
        provider->collective_contract().empty())
      throw std::invalid_argument("AMR field solver provider requires exact identities");
    const std::string identity(provider->identity());
    (void)exact_amr_field_solver_provider_declaration(*provider);
    if (!providers_.emplace(identity, std::move(provider)).second)
      throw std::invalid_argument("duplicate AMR field solver provider identity '" + identity +
                                  "'");
  }

  [[nodiscard]] std::shared_ptr<const AmrFieldSolverProvider> resolve(
      std::string_view identity) const {
    const auto found = providers_.find(std::string(identity));
    if (found == providers_.end())
      throw std::invalid_argument("unknown AMR field solver provider '" + std::string(identity) +
                                  "'");
    return found->second;
  }

 private:
  std::map<std::string, std::shared_ptr<const AmrFieldSolverProvider>> providers_;
};

struct AmrFieldSolverSupportAssessment {
  PreparedProviderSupport options;
  PreparedProviderSupport request;

  [[nodiscard]] bool accepted() const noexcept { return options.accepted() && request.accepted(); }
  [[nodiscard]] std::string exact_contract() const {
    if (!options.well_formed() || !request.well_formed() ||
        (!options.accepted() && request.accepted()))
      throw std::invalid_argument("AMR field solver returned malformed support decisions");
    ExactContractBuilder contract;
    contract.text("pops.amr.field-solver-support")
        .scalar(std::uint32_t{1})
        .bytes(exact_prepared_provider_support(options))
        .bytes(exact_prepared_provider_support(request));
    return std::move(contract).release();
  }
};

inline AmrFieldSolverSupportAssessment inspect_amr_field_solver_support(
    const AmrFieldSolverProvider& provider, const AmrFieldSolverBuildRequest& request) {
  AmrFieldSolverSupportAssessment support{provider.accepts_options(request.plan.solver_options),
                                          provider.supports(request)};
  (void)support.exact_contract();
  return support;
}

inline AmrFieldSolverSupportAssessment inspect_amr_field_solver_support_collectively(
    const AmrFieldSolverProvider& provider, const AmrFieldSolverBuildRequest& request) {
  AmrFieldSolverSupportAssessment support;
  std::string declaration;
  std::string support_contract;
  bool inspection_failed = false;
  try {
    declaration = exact_amr_field_solver_provider_declaration(provider);
    support = inspect_amr_field_solver_support(provider, request);
    support_contract = support.exact_contract();
  } catch (...) {
    inspection_failed = true;
  }
  if (all_reduce_max(inspection_failed ? 1L : 0L) != 0)
    throw std::runtime_error("AMR field solver support inspection failed on at least one MPI rank");
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"amr-field-solver-provider", declaration},
                                                 {"amr-field-solver-support", support_contract}}))
    throw std::runtime_error(
        "AMR field solver declaration or support decision differs across MPI ranks");
  if (!support.options.accepted())
    throw std::invalid_argument("AMR field solver provider rejected options (code " +
                                std::to_string(support.options.code) +
                                "): " + std::string(support.options.reason));
  if (!support.request.accepted())
    throw std::invalid_argument("AMR field solver provider rejected request (code " +
                                std::to_string(support.request.code) +
                                "): " + std::string(support.request.reason));
  return support;
}

inline std::string make_amr_field_solver_contract(std::string_view provider_identity,
                                                  const AmrFieldSolverBuildRequest& request) {
  ExactContractBuilder contract;
  contract.text("pops.amr.prepared-field-solver")
      .scalar(std::uint32_t{1})
      .text(provider_identity)
      .text(request.plan.plan_identity)
      .text(request.use_contract_identity)
      .bytes(request.plan.hierarchy_policy.exact_contract())
      .scalar(request.replicated_coarse)
      .optional_collective_contract(request.active)
      .scalar(request.plan.has_reaction)
      .scalar(request.plan.reaction)
      .scalar(request.plan.has_boundary_kernel)
      .text(request.plan.has_boundary_kernel ? request.plan.boundary_kernel.identity : "")
      .text(request.plan.has_boundary_kernel ? request.plan.boundary_kernel.residual_identity : "")
      .text(request.plan.has_boundary_kernel ? request.plan.boundary_kernel.jvp_identity : "")
      .scalar(request.plan.has_newton)
      .bytes(request.plan.nullspace.exact_contract())
      .bytes(request.plan.solver_options.exact_contract());
  if (request.plan.has_newton) {
    const auto& options = request.plan.newton;
    contract.scalar(options.tolerance)
        .scalar(options.max_iterations)
        .scalar(options.linear_tolerance)
        .scalar(options.linear_max_iterations)
        .scalar(options.restart)
        .scalar(options.armijo)
        .scalar(options.minimum_step);
  }
  int refinement = 1;
  for (std::size_t level = 0; level < request.hierarchy.ba.size(); ++level) {
    const Geometry geometry = request.geometry.refine(refinement);
    const FieldDistribution distribution = level == 0 && request.replicated_coarse
                                               ? FieldDistribution::Replicated
                                               : FieldDistribution::Distributed;
    EllipticBuildRequest level_request{geometry,
                                       request.hierarchy.ba[level],
                                       request.hierarchy.dm[level],
                                       request.boundary,
                                       request.active,
                                       distribution,
                                       0,
                                       1};
    level_request.boundary.dx = geometry.dx();
    level_request.boundary.dy = geometry.dy();
    contract.bytes(detail::elliptic_build_request_contract(level_request));
    if (level < request.hierarchy.refinement_ratios.size())
      refinement *= request.hierarchy.refinement_ratios[level];
  }
  return std::move(contract).release();
}

/// Builtin provider composition lives in its own translation unit. The AMR core depends only on the
/// open registry protocol and never interprets a provider or hierarchy-policy identity.
POPS_EXPORT std::shared_ptr<AmrFieldSolverProviderRegistry>
make_default_amr_field_solver_registry();

namespace detail {

struct
    AmrHistoryOps;  // ADC-631 multistep history-ring operations (friend of AmrRuntime; amr_history.hpp)

struct BootstrapNodeBilinearKernel {
  Array4 fine;
  ConstArray4 coarse;
  int component;
  int coarse_lo_x, coarse_lo_y, fine_lo_x, fine_lo_y;
  POPS_HD void operator()(int i, int j) const {
    const int ci = coarse_lo_x + floor_div(i - fine_lo_x, 2);
    const int cj = coarse_lo_y + floor_div(j - fine_lo_y, 2);
    const bool odd_x = ((i - fine_lo_x) & 1) != 0;
    const bool odd_y = ((j - fine_lo_y) & 1) != 0;
    if (!odd_x && !odd_y) {
      fine(i, j, component) = coarse(ci, cj, component);
    } else if (odd_x && !odd_y) {
      fine(i, j, component) =
          Real(0.5) * (coarse(ci, cj, component) + coarse(ci + 1, cj, component));
    } else if (!odd_x && odd_y) {
      fine(i, j, component) =
          Real(0.5) * (coarse(ci, cj, component) + coarse(ci, cj + 1, component));
    } else {
      fine(i, j, component) =
          Real(0.25) * (coarse(ci, cj, component) + coarse(ci + 1, cj, component) +
                        coarse(ci, cj + 1, component) + coarse(ci + 1, cj + 1, component));
    }
  }
};

struct BootstrapFaceProlongKernel {
  Array4 fine;
  ConstArray4 coarse;
  runtime::amr::TransferCentering centering;
  int component;
  int coarse_lo_x, coarse_lo_y, fine_lo_x, fine_lo_y;
  POPS_HD void operator()(int i, int j) const {
    const int ci = coarse_lo_x + floor_div(i - fine_lo_x, 2);
    const int cj = coarse_lo_y + floor_div(j - fine_lo_y, 2);
    if (centering == runtime::amr::TransferCentering::FaceX) {
      fine(i, j, component) =
          ((i - fine_lo_x) & 1)
              ? Real(0.5) * (coarse(ci, cj, component) + coarse(ci + 1, cj, component))
              : coarse(ci, cj, component);
    } else {
      fine(i, j, component) =
          ((j - fine_lo_y) & 1)
              ? Real(0.5) * (coarse(ci, cj, component) + coarse(ci, cj + 1, component))
              : coarse(ci, cj, component);
    }
  }
};

struct BootstrapConstantKernel {
  Array4 values;
  int component;
  Real value;
  POPS_HD void operator()(int i, int j) const { values(i, j, component) = value; }
};

struct BootstrapFloorKernel {
  Array4 values;
  int component;
  Real floor;
  POPS_HD int operator()(int i, int j) const {
    if (values(i, j, component) >= floor)
      return 0;
    values(i, j, component) = floor;
    return 1;
  }
};

inline void bootstrap_prolong_staggered(const MultiFab& parent, MultiFab& fine,
                                        runtime::amr::TransferCentering centering,
                                        const runtime::amr::SpatialTransferContext& context) {
  if (context.index.coarse_origin.size() != 2 || context.index.fine_origin.size() != 2 ||
      context.index.refinement_ratio != std::vector<int>{2, 2})
    throw std::runtime_error("staggered prolongation received an invalid index transform");
  MultiFab local_parent(coarsen_grown(fine.box_array(), 2, 2), fine.dmap(), parent.ncomp(), 0);
  parallel_copy(local_parent, parent);
  Box2D coarse_domain = context.logical_coarse_domain;
  Box2D fine_domain = context.logical_fine_domain;
  if (centering == runtime::amr::TransferCentering::FaceX ||
      centering == runtime::amr::TransferCentering::Node) {
    ++coarse_domain.hi[0];
    ++fine_domain.hi[0];
  }
  if (centering == runtime::amr::TransferCentering::FaceY ||
      centering == runtime::amr::TransferCentering::Node) {
    ++coarse_domain.hi[1];
    ++fine_domain.hi[1];
  }
  if (context.logical_fine_domain !=
          context.logical_coarse_domain.refine(context.index.refinement_ratio[0]) ||
      coarse_domain.lo[0] != context.index.coarse_origin[0] ||
      coarse_domain.lo[1] != context.index.coarse_origin[1] ||
      fine_domain.lo[0] != context.index.fine_origin[0] ||
      fine_domain.lo[1] != context.index.fine_origin[1])
    throw std::runtime_error("staggered prolongation index origin mismatch");
  for (const Box2D& box : parent.box_array().boxes())
    if (!coarse_domain.contains(box))
      throw std::runtime_error("staggered parent layout exceeds its logical transfer domain");
  for (const Box2D& box : fine.box_array().boxes())
    if (!fine_domain.contains(box))
      throw std::runtime_error("staggered child layout exceeds its logical transfer domain");
  for (int li = 0; li < fine.local_size(); ++li) {
    Array4 destination = fine.fab(li).array();
    const ConstArray4 source = local_parent.fab(li).const_array();
    for (int component = 0; component < fine.ncomp(); ++component) {
      if (centering == runtime::amr::TransferCentering::Node)
        for_each_cell(fine.box(li), BootstrapNodeBilinearKernel{
                                        destination, source, component, coarse_domain.lo[0],
                                        coarse_domain.lo[1], fine_domain.lo[0], fine_domain.lo[1]});
      else
        for_each_cell(fine.box(li),
                      BootstrapFaceProlongKernel{destination, source, centering, component,
                                                 coarse_domain.lo[0], coarse_domain.lo[1],
                                                 fine_domain.lo[0], fine_domain.lo[1]});
    }
  }
}

inline void bootstrap_prolong_face_vector(const MultiFab& coarse_x, const MultiFab& coarse_y,
                                          MultiFab& fine_x, MultiFab& fine_y,
                                          const runtime::amr::SpatialTransferContext& context) {
  if (coarse_x.ncomp() != coarse_y.ncomp() || fine_x.ncomp() != fine_y.ncomp() ||
      coarse_x.ncomp() != fine_x.ncomp())
    throw std::runtime_error("face-vector prolongation component mismatch");
  bootstrap_prolong_staggered(coarse_x, fine_x, runtime::amr::TransferCentering::FaceX, context);
  bootstrap_prolong_staggered(coarse_y, fine_y, runtime::amr::TransferCentering::FaceY, context);
}

struct SetFieldCoverageKernel {
  Array4 mask;
  Real value;
  POPS_HD void operator()(int i, int j) const { mask(i, j, 0) = value; }
};

/// Publishes one solved named potential and its optional signed centered gradient into the
/// hierarchy aux channel.  A named functor keeps generated/external nvcc instantiations portable
/// and routes every valid-cell operation through the common Kokkos execution authority.
struct AmrNamedFieldPostprocessKernel {
  Array4 aux;
  ConstArray4 phi;
  int phi_component;
  int gradient_x_component;
  int gradient_y_component;
  Real gradient_scale;
  Real dx;
  Real dy;
  bool has_gradient;

  POPS_HD void operator()(int i, int j) const {
    aux(i, j, phi_component) = phi(i, j);
    if (has_gradient) {
      aux(i, j, gradient_x_component) =
          gradient_scale * (phi(i + 1, j) - phi(i - 1, j)) / (Real(2) * dx);
      aux(i, j, gradient_y_component) =
          gradient_scale * (phi(i, j + 1) - phi(i, j - 1)) / (Real(2) * dy);
    }
  }
};

struct AmrNamedAuxCopyKernel {
  Array4 aux;
  const Real* field;
  int component;
  int row_width;
  int origin_i;
  int origin_j;

  POPS_HD void operator()(int i, int j) const {
    aux(i, j, component) =
        field[static_cast<std::int64_t>(j - origin_j) * row_width + (i - origin_i)];
  }
};
}  // namespace detail

/// Type-erased closures of ONE AMR block, placed on the shared hierarchy. AMR counterpart of the
/// Species struct of System::Impl: a name + its level stack (on the shared layout) + its closures
/// (advance / elliptic-rhs / max_speed / mass / density). The closures capture the CONCRETE
/// Model/Limiter/Flux of the block (resolved at build): the kernel stays COMPILED, only the block
/// list is type-erased. Produced by detail::build_amr_block (amr_dsl_block.hpp).
struct AmrRuntimeBlock {
  std::string name;
  /// Exact owner-qualified state Handle, installed by the block plan independently of whether this
  /// block owns a physical boundary authority.
  std::string state_identity;
  int ncomp = 1;
  double gamma = static_cast<double>(kPhysicalDefaultGamma);
  /// EXPLICIT substeps of the block within ITS effective macro-step: the effective step (stride * dt)
  /// is split into substeps equal pieces and each piece is advanced by ONE advance_amr (cf.
  /// AmrRuntime::step). substeps=1 => a single advance_amr over the whole effective step (bit-identical).
  int substeps = 1;
  /// HOLD-THEN-CATCH-UP cadence of the block (multirate). stride=1 (default): advances EVERY macro-step
  /// (bit-identical). stride=M>1: HELD at macro-steps 0..M-2 then CATCHES UP at M-1 ((macro_step+1)%M==0)
  /// by an effective step M*dt. Same semantics as block_stride_v / AmrSystemCoupler::step (#140). The
  /// end-of-window catch-up keeps the block temporally CONSISTENT with the fast blocks (never "in the
  /// future"), so the summed-RHS Poisson coupling stays meaningful: a held block contributes with its
  /// FROZEN state (its last advance), not an anticipated state that would falsify q_b n_b in the sum.
  int stride = 1;
  /// Width of the aux channel READ by the block model (aux_comps<Model>(); >= kAuxBaseComps). The aux
  /// channel SHARED per level is sized to the MAX of this width over all blocks, so that a block
  /// reading an extra field (B_z, T_e; n_aux > 3) never reads out of bounds.
  int aux_ncomp = kAuxBaseComps;
  /// Authenticated reconstruction demand used to select a sufficient coarse/fine route and reject
  /// a fine hierarchy when the installed provider cannot meet it.  No order is inferred here.
  int reconstruction_order = 1;
  int reconstruction_ghost_depth = 1;
  /// Exact native face-speed cache policy consumed by this block's prepared advance scratch.
  /// IMEX source-free transport sets this false because its concrete closure does not request HLL
  /// caching; explicit blocks retain the resolved numerical-flux policy.
  bool wave_speed_cache = false;

  /// Descriptor of the model CONSERVATIVE variables (names + physical ROLES, Model::conservative_vars()).
  /// Single source of truth to resolve a role (Density, MomentumX, ...) -> component index in
  /// add_coupled_source, like System::add_coupled_source reads Species::cons_vars. The resolution is
  /// STRICT (#181): if the block does NOT expose the requested canonical role (index_of < 0),
  /// add_coupled_source THROWS instead of falling back to component 0 (a silent fallback would apply
  /// the source to the wrong field).
  VariableSet cons_vars;

  /// Level stack of the block (level 0 = coarse, > 0 = fine patches), ON the shared layout. The aux
  /// pointer of each AmrLevelMP is (re)wired by AmrRuntime to the SHARED aux of the level. shared_ptr:
  /// AmrRuntimeBlock stays MOVABLE (a std::vector<AmrLevelMP> is heavy to move into a std::function,
  /// and the engine ctor needs a stable address for the closures).
  std::shared_ptr<std::vector<AmrLevelMP>> levels;
  /// Prepared boundary authority plus the late-bound exact runtime storage registry shared by all
  /// per-level closures of this block.
  std::shared_ptr<const PreparedBoundaryPlan> boundary_plan;
  std::shared_ptr<GridContext::BoundaryFieldRegistryFactory> boundary_field_registry;
  std::shared_ptr<const AmrBoundaryFillAuthority> transport_boundary_fill;
  /// Stable indirection captured by the native advance closures. AmrRuntime replaces the contained
  /// plan transactionally after every topology generation while the shared holder address remains
  /// unchanged across block moves and regrids.
  std::optional<PreparedAmrFillPatchPlan> fill_patch_plan;
  std::vector<detail::PreparedConservativeCellTransferWorkspace>
      coarse_fine_spatial_workspaces;
  std::optional<PreparedAmrAverageDownPlan> average_down_plan;
  std::optional<PreparedAmrAdvanceScratchPlan> advance_scratch_plan;
  /// One sequential session per level, materialized after all qualified routes are installed.
  /// Prepared Krylov workspaces create separate lane-private sessions.
  std::vector<std::shared_ptr<ExecutionLane>> boundary_lanes;
  std::vector<std::shared_ptr<PreparedGridBoundarySession>> boundary_sessions;

  /// Advances the block by ONE substep of size dt: AMR transport (Berger-Oliger + conservative reflux
  /// + average_down) over the block level stack, with ITS spatial scheme (Limiter, Flux). Captures
  /// advance_amr<Limiter, Flux> on the concrete Model. The substep loop and the stride cadence are
  /// carried by AmrRuntime::step (runtime counterpart of AmrSystemCoupler::step): the closure does ONE
  /// advance_amr, the engine calls it substeps times (dt = effective step/substeps). The signature
  /// passes the base domain + periodicity + coarse ownership policy, rewired by the engine.
  std::function<void(std::vector<AmrLevelMP>&, const Box2D&, Real, Periodicity, bool,
                     PreparedAmrFillPatchPlan*, PreparedAmrAverageDownPlan*,
                     PreparedAmrAdvanceScratchPlan*)>
      advance;

  /// Explicit-clock counterpart of @ref advance.  It is populated by compiled/native block builders
  /// that support the final temporal contract and receives the immutable plan prepared from the
  /// complete contiguous relation chain.
  /// AmrRuntime selects this closure whenever relations were installed; it never installs a relation
  /// and then falls back to the spatial-ratio legacy closure above.
  std::function<void(std::vector<AmrLevelMP>&, const Box2D&, Real, Periodicity, bool,
                     const detail::PreparedAmrTemporalPlan&, PreparedAmrFillPatchPlan*,
                     PreparedAmrAverageDownPlan*, PreparedAmrAdvanceScratchPlan*)>
      advance_with_temporal_plan;

  /// TEMPORAL TREATMENT of the block: false (default) = EXPLICIT (forward-Euler source, in advance);
  /// true = IMEX (stiff source treated IMPLICITLY by backward_euler_source). The facade (AmrSystem)
  /// freezes it from time="imex". Selected EXPLICITLY in AmrRuntime::step (runtime counterpart of the
  /// constexpr block_time_treatment_v dispatch of AmrSystemCoupler::step): an explicit block goes
  /// through advance, an IMEX block through imex_advance. false everywhere -> bit-identical trajectory
  /// to the historical one.
  bool imex = false;

  /// IMEX advance of the block by ONE substep of size dt = ONE Lie step [transport; implicit source]:
  /// (1) EXPLICIT SOURCE-FREE transport (-div F only, SourceFreeModel) via the AMR engine (Berger-
  /// Oliger + reflux + average_down), then (2) IMPLICIT stiff source backward_euler_source AT EACH
  /// LEVEL (local Newton, FD jacobian, block-carried partial-IMEX mask) + a fine -> coarse cascade.
  /// Mirrors the IMEX branch of AmrSystemCoupler::step; at substeps=1 IDENTICAL to it, at substeps>1 the
  /// runtime sub-cycles it (divergence intentional, cf. file header). CONSERVATION: the source is
  /// cell-local (outside the reflux registers) so C/F conservation holds, and the final cascade restores
  /// each covered coarse cell to the 2x2 average of its children. Empty for an explicit block.
  std::function<void(std::vector<AmrLevelMP>&, const Box2D&, Real, Periodicity, bool,
                     PreparedAmrFillPatchPlan*, PreparedAmrAverageDownPlan*,
                     PreparedAmrAdvanceScratchPlan*)>
      imex_advance;

  /// Explicit-clock counterpart of @ref imex_advance.  The source-free transport consumes the same
  /// authored clock chain as an explicit block; the following implicit source/cascade is unchanged.
  std::function<void(std::vector<AmrLevelMP>&, const Box2D&, Real, Periodicity, bool,
                     const detail::PreparedAmrTemporalPlan&, PreparedAmrFillPatchPlan*,
                     PreparedAmrAverageDownPlan*, PreparedAmrAdvanceScratchPlan*)>
      imex_advance_with_temporal_plan;

  /// POINTWISE PROJECTION post-pas (ADC-177) : U <- project(U, aux) appliquee PAR NIVEAU a la FIN
  /// de l'avance complete du bloc (substeps + reflux/cascade faits). Vide -> aucune projection
  /// (modele sans HasPointwiseProjection : trajectoire bit-identique). Locale par niveau (aucun
  /// collectif MPI). Cf. detail::apply_pointwise_project_amr, cable par build_amr_block.
  std::function<void(std::vector<AmrLevelMP>&)> project_per_level;
  /// Same concrete projection applied to a provisional Program scratch on one level.  This is the
  /// typed ProjectAndRecheck seam; it never mutates the live block unless that scratch is committed.
  std::function<void(MultiFab&, const MultiFab&)> project_level_state;

  /// NEWTON DIAGNOSTICS (AMR counterpart of System::newton_report). false (default) -> imex_advance
  /// passes report=nullptr to backward_euler_source: FAST bit-identical path, no extra allocation or
  /// reduction. true -> imex_advance passes @c newton_report.get() (STABLE address since shared_ptr)
  /// to backward_euler_source of EACH level; the report is AGGREGATED (max residual, max iterations,
  /// sum of failed cells, MPI all_reduce, structured fail_policy events) over all levels AND all
  /// substeps of a macro-step. AmrRuntime::step RESETS the report at the head of the block advance
  /// (parity with System::AdvanceImex which resets at the head of operator()). MULTI-BLOCK native only
  /// (the single-block coupler and the .so loaders reject it at build / at the facade). STABLE address
  /// (shared_ptr): captured by the imex_advance closure AND read by AmrRuntime::newton_report.
  bool newton_diagnostics = false;
  std::shared_ptr<NewtonReport> newton_report;

  /// Contribution of the block to the Poisson right-hand side: rhs += elliptic_rhs_b(U_b) on the
  /// coarse. CO-LOCATED: the loop reads U_b and writes rhs AT THE SAME cells (same shared coarse
  /// BoxArray). The SUM of the contributions of all blocks forms the system Poisson RHS.
  std::function<void(const MultiFab&, MultiFab&)> add_elliptic_rhs;

  /// Per-NAMED-field elliptic right-hand-side contributions of the block (ADC-428): field name ->
  /// closure rhs += elliptic_field_rhs_b(U_b) on the coarse, exactly like @c add_elliptic_rhs but for a
  /// SECOND (user-named) elliptic field declared by the block's model (m.elliptic_field). The native
  /// loader attaches one closure here per declared field (set_block_elliptic_field). AmrRuntime sums
  /// them over the blocks into the named field's dedicated solver RHS (solve_named_fields). Empty for a
  /// block that declares no named field -> the named-field solve loop never reads it (bit-identical).
  std::map<std::string, std::function<void(const MultiFab&, MultiFab&)>> named_elliptic_rhs;

  /// SEMI-DISCRETE residual of the block on ONE level (epic ADC-508, compiled-Program AMR driver):
  /// R <- -div F(U) + S(U, aux) over the level's grid, the per-level counterpart of System's
  /// Species::rhs_into. Signature (U, aux, geom, R): @c U the level state, @c aux the SHARED per-level
  /// aux (phi / grad / B_z, filled by solve_fields + coarse->fine injection), @c geom the level metric
  /// (dx/dy >> k, domain << k), @c R the output residual. Captures BlockRhsEval<Limiter, Flux, Model>
  /// (the SAME evaluator System uses, device-clean named functor) on the concrete scheme; the level
  /// geometry / domain are passed in so the SAME closure serves every level. EMPTY for a block built
  /// before the seam (the host .so prototype loader): AmrRuntime::level_rhs_into fails loud then. Used
  /// ONLY by an installed compiled time Program (AmrProgramContext); the native AMR step never calls it.
  std::function<void(MultiFab&, const MultiFab&, const Geometry&, MultiFab&)> level_rhs;
  /// FLUX-ONLY per-level residual R <- -div F(U) (NO default source), the SourceFreeModel<Model> path
  /// (Lie/Strang split, ADC-425). Same signature / device contract as @ref level_rhs.
  std::function<void(MultiFab&, const MultiFab&, const Geometry&, MultiFab&)> level_neg_div_flux;
  /// SOURCE-ONLY per-level residual R <- S(U, aux) (NO flux divergence), the exact MIRROR of @ref
  /// level_neg_div_flux (ADC-430). Same signature / device contract as @ref level_rhs.
  std::function<void(MultiFab&, const MultiFab&, const Geometry&, MultiFab&)> level_source;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                     const MultiFab&, const Geometry&, MultiFab&)>
      level_rhs_at_point;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                     const MultiFab&, const Geometry&, MultiFab&)>
      level_neg_div_flux_at_point;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                     const MultiFab&, const Geometry&, MultiFab&)>
      level_rhs_core_at_point;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                     const MultiFab&, const Geometry&, MultiFab&)>
      level_neg_div_flux_core_at_point;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                     const MultiFab&, const Geometry&, MultiFab&)>
      level_boundary_residual_at_point;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                     const MultiFab&, const MultiFab&, const Geometry&, MultiFab&)>
      level_boundary_jvp_at_point;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                     const MultiFab&, const Geometry&, MultiFab&,
                     const PreparedGridBoundarySession&)>
      level_rhs_core_at_point_prepared;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                     const MultiFab&, const Geometry&, MultiFab&,
                     const PreparedGridBoundarySession&)>
      level_neg_div_flux_core_at_point_prepared;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                     const MultiFab&, const Geometry&, MultiFab&,
                     const PreparedGridBoundarySession&)>
      level_boundary_residual_at_point_prepared;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                     const MultiFab&, const MultiFab&, const Geometry&, MultiFab&,
                     const PreparedGridBoundarySession&)>
      level_boundary_jvp_at_point_prepared;

  /// FLUX-MATERIALISING per-level residual for conservative reflux (ADC-639): computes R <- -div F + S
  /// EXACTLY as @ref level_rhs, but FIRST writes the face fluxes Fx/Fy (compute_face_fluxes) then derives
  /// R from them (mf_eval_rhs), so the divergence uses the SAME flux the reflux register samples --
  /// bit-consistent with the fused @ref level_rhs by construction (face_flux.hpp:236-238). Signature
  /// (U, aux, geom, Fx, Fy, R): Fx/Fy sized by the caller (xface_box/yface_box of the level box_array).
  /// Captures the SAME <Limiter, Flux, Model> as @ref level_rhs. Read ONLY by the reflux capture path
  /// (nlev>1); the coarse-only / flat / native paths never touch it. EMPTY for a pre-seam host .so.
  std::function<void(MultiFab&, const MultiFab&, const Geometry&, MultiFab&, MultiFab&, MultiFab&)>
      level_flux_capture;
  /// FLUX-ONLY (SourceFreeModel) counterpart of @ref level_flux_capture (the neg_div_flux capture path).
  std::function<void(MultiFab&, const MultiFab&, const Geometry&, MultiFab&, MultiFab&, MultiFab&)>
      level_flux_capture_neg_div;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                     const MultiFab&, const Geometry&, MultiFab&, MultiFab&, MultiFab&,
                     const PreparedGridBoundarySession&)>
      level_flux_capture_prepared;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                     const MultiFab&, const Geometry&, MultiFab&, MultiFab&, MultiFab&,
                     const PreparedGridBoundarySession&)>
      level_flux_capture_neg_div_prepared;

  /// Speed driving the block CFL on the coarse. By default max_wave_speed (historical); when the
  /// model declares the HasStabilitySpeed trait, it is lambda* (stability_speed) that the closure
  /// reduces -- SAME policy as System (make_max_speed), cf. build_amr_block.
  std::function<Real(const MultiFab&, const MultiFab&)> max_speed;

  /// OPTIONAL STEP BOUNDS of the block (AMR StabilityPolicy, audit 2026-06): evaluated on the COARSE
  /// (level 0, where the AMR CFL lives -- cf. step_cfl: h is the conservative coarse spacing).
  /// EMPTY (default) -> step_cfl
  /// keeps the transport bound only, bit-identical. Filled by build_amr_block / build_amr_compiled when
  /// the model declares HasSourceFrequency / HasStabilityDt (same semantics as System: mu in 1/s ->
  /// dt <= cfl*substeps/(stride*mu), without h; direct admissible step -> dt <=
  /// dt_adm*substeps/stride, without cfl).
  std::function<Real(const MultiFab&, const MultiFab&)> source_frequency;
  std::function<Real(const MultiFab&, const MultiFab&)> stability_dt;

  /// Mass of component 0 of the block coarse (sum u*dV; cross-rank reduced if distributed).
  std::function<Real()> mass;

  /// Coarse density (component 0) as a global ny*nx row-major field (diagnostic).
  std::function<std::vector<double>()> density;

  /// Coarse potential read from the shared aux (component 0) as a ny*nx row-major field.
  /// Identical for all blocks (shared aux); carried per block for API symmetry.
  std::function<std::vector<double>(const MultiFab&)> potential;

  /// Interface-aware per-level residual.  Like level_rhs, except every face owned by an installed
  /// shared block interface is omitted; InterfaceFluxScheduler inserts that pair flux exactly once.
  /// An interface install rejects an empty closure rather than double-counting level_rhs's BC flux.
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                     const MultiFab&, const Geometry&, MultiFab&)>
      level_rhs_without_prepared_interfaces;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                     const MultiFab&, const Geometry&, MultiFab&)>
      level_neg_div_flux_without_prepared_interfaces;
};

/// AMR multi-block engine at runtime. Owns the SHARED aux per level, the coarse Poisson
/// (GeometricMG), the geometry + BC, and the type-erased block REGISTRY. Reproduces the
/// AmrSystemCoupler algorithm (solve_fields + step) over closures rather than a CoupledSystem.
class AmrRuntime {
  struct BlockTransferAuthority {
    runtime::amr::PreparedTransferKernel prolongation;
    runtime::amr::PreparedTransferKernel restriction;
    runtime::amr::PreparedTransferKernel coarse_fine;
    runtime::amr::PreparedTransferKernel temporal;
    int refinement_ratio = 0;
    bool prepared = false;
    runtime::amr::PreparedTransferCapabilities capabilities;
  };

  struct TemporalParentWorkspace {
    std::uint64_t topology_generation = 0;
    std::size_t block = 0;
    int parent_level = -1;
    MultiFab value;
  };

  struct AuxPublicationWorkspace {
    std::uint64_t topology_generation = 0;
    bool refined_values = false;
    std::vector<int> components;
    std::vector<MultiFab> packed;
    std::vector<detail::PreparedConservativeLinearTransferWorkspace> coarse_transfers;
  };

 public:
  using StaticAuxField = std::vector<Real, fab_allocator<Real>>;

  struct BootstrapStaggeredField {
    runtime::amr::TransferCentering centering{};
    std::vector<MultiFab> levels;
  };

  struct BootstrapCacheState {
    std::uint64_t epoch = 0;
    bool valid = false;
    int materialized_level = -1;
    std::vector<PatchBox> topology;
  };

  /// One exact graph representation is shared by the builtin VM and an external Tagger evaluator.
  /// External implementations receive this program verbatim and return candidates; they never
  /// replace AMRTagging's refine/coarsen/equality/conflict authority.
  using TaggingProgram = runtime::amr::PreparedTaggingProgram;

  /// Move-independent image of every runtime-owned value a macro-step may mutate.  The owning
  /// transaction retains one instance and capture_step_snapshot() refreshes its MultiFab storage in
  /// place when the layout is unchanged; on a real topology change the same image reallocates only
  /// the incompatible levels.  The snapshot therefore carries the hierarchy itself, not merely field
  /// values on the current topology, without imposing per-attempt host allocations.
  struct StepSnapshot {
    struct NamedFieldSnapshot {
      bool allocated = false;
      bool composite = false;
      std::vector<MultiFab> phi;
      std::vector<MultiFab> rhs;
    };

    std::vector<std::vector<AmrLevelMP>> block_levels;
    AmrHierarchyLayout hierarchy;
    std::vector<MultiFab> aux;
    MultiFab phi;
    MultiFab poisson_rhs;
    std::map<std::string, NamedFieldSnapshot> named_fields;
    std::map<std::string, std::vector<std::vector<MultiFab>>> history_rings;
    std::map<std::string, int> history_depth;
    std::map<std::string, std::size_t> history_block_owner;
    std::map<std::string, std::vector<char>> history_initialized;
    std::map<std::string, std::vector<Real>> history_slot_dt;
    std::map<std::string, BootstrapStaggeredField> staggered_fields;
    std::map<std::string, BootstrapCacheState> bootstrap_caches;
    std::vector<NewtonReport> newton_reports;
    std::vector<char> has_newton_report;
    std::string last_dt_reason;
    int nlev = 0;
    int macro_step = 0;
    int solve_count = 0;
    int regrid_count = 0;
    std::uint64_t topology_epoch = 0;
    bool has_profiler = false;
    runtime::program::Profiler profiler;
  };

  /// @param geom        geometry of the coarse level (domain + physical extents).
  /// @param hierarchy   Runtime-owned level topology, distribution and metric authority.
  /// @param bcPhi       BC of the coarse Poisson.
  /// @param blocks      block registry (>= 1), all on the SAME layout (guarded at the ctor).
  /// @param base_per    periodicity of the base domain (transport).
  /// @param replicated_coarse  ownership of level 0 (replicated single-box, or distributed multi-box).
  /// @param active      conductive-wall predicate (passed to MG; empty = none).
  AmrRuntime(const Geometry& geom, AmrHierarchyLayout hierarchy, const BCRec& bcPhi,
             std::vector<AmrRuntimeBlock> blocks, Periodicity base_per,
             bool replicated_coarse, ActiveRegionProvider2D active = {},
             std::shared_ptr<const AmrFieldSolverProviderRegistry> field_solver_registry =
                 make_default_amr_field_solver_registry(),
             std::shared_ptr<const FieldNullspaceProviderRegistry> nullspace_provider_registry =
                 make_default_field_nullspace_provider_registry(),
             FieldNullspaceProviderSelection default_field_nullspace =
                 operator_topology_zero_mean_nullspace(),
             std::string default_field_solver = "geometric_mg",
             AmrFieldSolverOptions default_field_options = {})
      : geom_(geom),
        dom_(geom.domain),
        base_per_(base_per),
        bcPhi_(bcPhi),
        aux_bc_(detail::derive_aux_bc(bcPhi)),
        replicated_coarse_(replicated_coarse),
        hierarchy_(std::move(hierarchy)),
        wall_active_(std::move(active)),
        field_solver_registry_(std::move(field_solver_registry)),
        nullspace_provider_registry_(std::move(nullspace_provider_registry)),
        blocks_(std::move(blocks)) {
    if (!field_solver_registry_)
      throw std::invalid_argument("AmrRuntime requires an AMR field solver provider registry");
    if (!nullspace_provider_registry_)
      throw std::invalid_argument("AmrRuntime requires a field-nullspace provider registry");
    if (!hierarchy_.load_balance)
      throw std::invalid_argument(
          "AmrRuntime requires the hierarchy's prepared load-balance authority");
    if (blocks_.empty())
      throw std::runtime_error("AmrRuntime : at least one block required");
    for (const auto& b : blocks_)
      if (!b.levels || b.levels->empty())
        throw std::runtime_error(
            "AmrRuntime : each block must carry at least one level "
            "(coarse) on the shared layout");
    nlev_ = hierarchy_.nlev();
    if (nlev_ < 1 || hierarchy_.dm.size() != hierarchy_.ba.size() ||
        hierarchy_.dx.size() != hierarchy_.ba.size() ||
        hierarchy_.dy.size() != hierarchy_.ba.size() ||
        hierarchy_.refinement_ratios.size() + 1 != hierarchy_.ba.size())
      throw std::runtime_error("AmrRuntime : invalid runtime-owned hierarchy manifest");
    for (std::size_t level = 0; level < hierarchy_.refinement_ratios.size(); ++level) {
      const int ratio = hierarchy_.refinement_ratios[level];
      if (ratio != kAmrRefRatio ||
          hierarchy_.dx[level] != hierarchy_.dx[level + 1] * Real(ratio) ||
          hierarchy_.dy[level] != hierarchy_.dy[level + 1] * Real(ratio))
        throw std::runtime_error(
            "AmrRuntime : native AMR currently requires spatial refinement ratio 2");
    }
    maximum_refinement_ratios_ = hierarchy_.refinement_ratios;
    // EXACT layout consistency between blocks (the aux is shared per level): same number of levels,
    // and per level same BoxArray (boxes AND order), same DistributionMapping, same dx/dy. SAME guard
    // as AmrSystemCoupler (detail::same_layout_or_throw): all blocks live on ALL patches of the
    // UNIQUE shared hierarchy. A single block matches itself trivially (the loop over the other blocks
    // is empty).
    for (const auto& block : blocks_) {
      if (static_cast<int>(block.levels->size()) != nlev_)
        throw std::runtime_error(
            "AmrRuntime : block level count differs from runtime-owned hierarchy");
      for (int level = 0; level < nlev_; ++level) {
        const auto& value = (*block.levels)[static_cast<std::size_t>(level)];
        if (!detail::same_level_layout(value.U.box_array(), value.U.dmap(), value.dx, value.dy,
                                       hierarchy_.ba[static_cast<std::size_t>(level)],
                                       hierarchy_.dm[static_cast<std::size_t>(level)],
                                       hierarchy_.dx[static_cast<std::size_t>(level)],
                                       hierarchy_.dy[static_cast<std::size_t>(level)]))
          throw std::runtime_error(
              "AmrRuntime : block storage differs from runtime-owned hierarchy");
      }
      if (block.boundary_plan &&
          !same_periodicity(block.boundary_plan->periodicity(), base_per_))
        throw std::runtime_error(
            "AmrRuntime prepared boundary topology differs from the shared hierarchy");
      if (block.transport_boundary_fill)
        validate_amr_boundary_fill_authority(base_per_, block.transport_boundary_fill.get(),
                                             *block.levels);
      else if (!block.boundary_plan && (!base_per_.x || !base_per_.y))
        throw std::runtime_error(
            "AmrRuntime non-periodic hierarchy has no physical boundary authority");
    }

    AmrHierarchyLayout coarse_hierarchy;
    coarse_hierarchy.ba = {hierarchy_.ba.front()};
    coarse_hierarchy.dm = {hierarchy_.dm.front()};
    coarse_hierarchy.dx = {hierarchy_.dx.front()};
    coarse_hierarchy.dy = {hierarchy_.dy.front()};
    coarse_hierarchy.load_balance = hierarchy_.load_balance;
    AmrFieldSolveConfig default_plan;
    default_plan.provider_identity = "pops://amr/default-field";
    default_plan.plan_identity = "pops://amr/default-field/plan@1";
    default_plan.solver = std::move(default_field_solver);
    default_plan.nullspace = default_field_nullspace;
    default_plan.solver_options = std::move(default_field_options);
    std::shared_ptr<const AmrFieldSolverProvider> default_provider;
    std::string default_provider_declaration;
    std::string expected_default_contract;
    bool default_declaration_failed = false;
    try {
      default_provider = field_solver_registry_->resolve(default_plan.solver);
      if (default_plan.solver_options.schema_identity.empty())
        default_plan.solver_options = default_provider->default_field_options();
      const auto hierarchy_policy =
          default_provider->default_hierarchy_policy("pops.amr.field-solver-use.default@1");
      if (!hierarchy_policy)
        throw std::runtime_error(
            "default AMR field provider requires an explicit hierarchy policy");
      default_plan.hierarchy_policy = *hierarchy_policy;
    } catch (...) {
      default_declaration_failed = true;
    }
    if (all_reduce_max(default_declaration_failed ? 1L : 0L) != 0)
      throw std::runtime_error(
          "AmrRuntime: default field solver provider declaration failed on at least one rank");
    const AmrFieldSolverBuildRequest default_request{
        geom_,        coarse_hierarchy,   bcPhi_,
        wall_active_, replicated_coarse_, "pops.amr.field-solver-use.default@1",
        default_plan};
    try {
      default_provider_declaration = exact_amr_field_solver_provider_declaration(*default_provider);
    } catch (...) {
      default_declaration_failed = true;
    }
    if (all_reduce_max(default_declaration_failed ? 1L : 0L) != 0)
      throw std::runtime_error("AmrRuntime: default field solver provider declaration is invalid");
    (void)inspect_amr_field_solver_support_collectively(*default_provider, default_request);
    try {
      expected_default_contract = default_provider->expected_prepared_contract(default_request);
    } catch (...) {
      default_declaration_failed = true;
    }
    if (all_reduce_max(default_declaration_failed || expected_default_contract.empty() ? 1L : 0L) !=
        0)
      throw std::runtime_error(
          "AmrRuntime: default field solver provider failed to publish an exact contract");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"amr-default-field-provider", default_provider_declaration},
             {"amr-default-field-expected-contract", expected_default_contract}}))
      throw std::runtime_error(
          "AmrRuntime: default field solver declaration differs across MPI ranks");
    bool default_build_failed = false;
    try {
      default_field_solver_ = default_provider->build(default_request);
    } catch (...) {
      default_build_failed = true;
    }
    if (all_reduce_max(default_build_failed || !default_field_solver_ ? 1L : 0L) != 0)
      throw std::runtime_error(
          "AmrRuntime: default field solver construction failed on at least one rank");
    bool default_inspection_failed = false;
    bool default_materialization_mismatch = false;
    std::string actual_default_contract;
    try {
      const MultiFab& default_rhs = default_field_solver_->rhs_level(0);
      const MultiFab& default_phi = default_field_solver_->phi_level(0);
      const FieldDistribution default_distribution =
          replicated_coarse_ ? FieldDistribution::Replicated : FieldDistribution::Distributed;
      actual_default_contract = default_field_solver_->exact_prepared_contract();
      default_materialization_mismatch =
          default_field_solver_->provider_identity() != default_provider->identity() ||
          actual_default_contract != expected_default_contract ||
          default_field_solver_->level_count() != 1 ||
          default_field_solver_->couples_hierarchy_levels() || default_rhs.ncomp() != 1 ||
          default_rhs.n_grow() != 0 || default_phi.ncomp() != 1 || default_phi.n_grow() != 1 ||
          default_rhs.box_array().boxes() != coarse_hierarchy.ba.front().boxes() ||
          default_phi.box_array().boxes() != coarse_hierarchy.ba.front().boxes() ||
          default_rhs.dmap().ranks() != coarse_hierarchy.dm.front().ranks() ||
          default_phi.dmap().ranks() != coarse_hierarchy.dm.front().ranks() ||
          !detail::field_distribution_layout_matches(default_rhs, default_distribution) ||
          !detail::field_distribution_layout_matches(default_phi, default_distribution) ||
          default_rhs.shares_storage_with(default_phi);
    } catch (...) {
      default_inspection_failed = true;
    }
    if (all_reduce_max(default_inspection_failed ? 1L : 0L) != 0)
      throw std::runtime_error(
          "AmrRuntime: default field solver inspection failed on at least one rank");
    if (all_reduce_max(default_materialization_mismatch ? 1L : 0L) != 0)
      throw std::runtime_error(
          "AmrRuntime: default field solver provider materialized an invalid coarse contract");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"amr-default-field-actual-contract", actual_default_contract}}))
      throw std::runtime_error(
          "AmrRuntime: default field solver materialization differs across MPI ranks");

    FieldNullspaceProviderRequest default_nullspace_request;
    default_nullspace_request.plan_identity = "pops://amr/default-field/nullspace-plan@1";
    default_nullspace_request.operator_facts = field_nullspace_operator_facts_from_bc_rec(
        bcPhi_, /*has_reaction=*/false, static_cast<bool>(wall_active_));
    default_nullspace_request.topology.identity = "pops://amr/default-field/coarse-layout@1";
    default_nullspace_request.topology.exact_layout_contract = actual_default_contract;
    default_nullspace_request.topology.field_component = 0;
    if (!wall_active_)
      default_nullspace_request.topology.connected_component_contract =
          default_nullspace_request.topology.identity + ":connected-component@1";
    default_nullspace_request.topology.layouts = {&default_field_solver_->phi_level(0)};
    default_nullspace_request.topology.cell_measure = {geom_.dx() * geom_.dy()};
    default_nullspace_request.topology.level_distributions = {
        PreparedVectorDistribution(default_field_solver_->level_distribution(0))};
    default_field_nullspace_ =
        prepare_field_nullspace(default_field_nullspace, std::move(default_nullspace_request)).plan;
    default_field_nullspace_workspace_ = std::make_unique<FieldNullspaceWorkspace>(
        default_field_nullspace_,
        std::vector<const MultiFab*>{&default_field_solver_->phi_level(0)},
        std::vector<PreparedVectorDistribution>{
            PreparedVectorDistribution(default_field_solver_->level_distribution(0))});

    // Width of the SHARED aux channel: max of the blocks' aux_comps (>= kAuxBaseComps). Counterpart of
    // AmrSystemCoupler::system_aux_comps: a block reading an extra field (B_z, T_e) has the room at
    // each level, a base block ignores the extra components. Externally supplied canonical and named
    // fields are published through set_static_aux_component after construction. Without an extra-field
    // block -> kAuxBaseComps (3), identical to the base case.
    aux_ncomp_ = kAuxBaseComps;
    for (const auto& b : blocks_)
      if (b.aux_ncomp > aux_ncomp_)
        aux_ncomp_ = b.aux_ncomp;

    // SHARED aux: one MultiFab (phi, grad phi) per level, on the common grid. Sized once -> stable
    // addresses for the blocks' aux pointers. The layout comes only from the runtime authority.
    aux_.resize(nlev_);
    for (int k = 0; k < nlev_; ++k)
      aux_[k] = MultiFab(hierarchy_.ba[static_cast<std::size_t>(k)],
                         hierarchy_.dm[static_cast<std::size_t>(k)], aux_ncomp_, 1);
    for (auto& b : blocks_)
      for (int k = 0; k < nlev_; ++k)
        (*b.levels)[k].aux = &aux_[k];

    // Transfer executables are installed from the resolved route registry after construction.
    // Keeping these slots empty prevents the runtime from silently substituting its historical
    // order-two coarse/fine provider before the authored capability route arrives.
    block_transfer_authorities_.resize(blocks_.size());
    rematerialize_persistent_topology_resources_(topology_materialization_generation_);
  }

  int nlev() const { return nlev_; }
  Periodicity base_periodicity() const { return base_per_; }
  int max_levels() const { return static_cast<int>(maximum_refinement_ratios_.size()) + 1; }
  int level_refinement(int level) const {
    if (level < 0 || level >= nlev_)
      throw std::runtime_error("AmrRuntime::level_refinement level out of bounds");
    int refinement = 1;
    for (int transition = 0; transition < level; ++transition)
      refinement *= hierarchy_.refinement_ratios[static_cast<std::size_t>(transition)];
    return refinement;
  }
  void set_parent_child_temporal_relations(
      std::vector<::pops::amr::ParentChildClockRelation> relations) {
    if (relations.size() != hierarchy_.refinement_ratios.size())
      throw std::runtime_error(
          "AmrRuntime temporal relation count must match the active hierarchy");
    configured_temporal_relations_ = relations;
    maximum_refinement_ratios_ = hierarchy_.refinement_ratios;
    refresh_active_temporal_plan_();
  }
  void configure_hierarchy_capacity(
      std::vector<int> refinement_ratios,
      std::vector<::pops::amr::ParentChildClockRelation> temporal_relations) {
    if (refinement_ratios.size() + 1 < static_cast<std::size_t>(nlev_) ||
        temporal_relations.size() != refinement_ratios.size())
      throw std::runtime_error("AmrRuntime hierarchy capacity is smaller than its active depth");
    for (std::size_t transition = 0; transition < refinement_ratios.size(); ++transition) {
      if (refinement_ratios[transition] != kAmrRefRatio ||
          (transition < hierarchy_.refinement_ratios.size() &&
           refinement_ratios[transition] != hierarchy_.refinement_ratios[transition]))
        throw std::runtime_error(
            "AmrRuntime hierarchy capacity requires spatial refinement ratio 2");
      if (temporal_relations[transition].parent_level() != static_cast<int>(transition) ||
          temporal_relations[transition].child_level() != static_cast<int>(transition + 1))
        throw std::runtime_error(
            "AmrRuntime hierarchy capacity has a non-contiguous temporal relation");
    }
    maximum_refinement_ratios_ = std::move(refinement_ratios);
    configured_temporal_relations_ = std::move(temporal_relations);
    refresh_active_temporal_plan_();
  }
  const ::pops::amr::ParentChildClockRelation& parent_child_temporal_relation(
      int child_level) const {
    if (child_level < 1 || child_level >= nlev_)
      throw std::runtime_error("AmrRuntime::parent_child_temporal_relation level out of bounds");
    if (temporal_relations_.size() != hierarchy_.refinement_ratios.size())
      throw std::runtime_error(
          "AMR Program execution lacks explicit parent/child temporal relations; spatial "
          "refinement ratios are never reused as time-subcycling ratios");
    return temporal_relations_[static_cast<std::size_t>(child_level - 1)];
  }
  const std::vector<::pops::amr::ParentChildClockRelation>& checkpoint_temporal_relations() const {
    return temporal_relations_;
  }
  runtime::amr::SpatialTransferContext bootstrap_transfer_context(
      const MultiFab& coarse, const MultiFab& fine, int coarse_level, int fine_level,
      int refinement_ratio, bool replicated_parent = false, Periodicity periodicity = {}) const {
    const Box2D coarse_box = amr_level_index_domain(geom_.domain, coarse_level);
    (void)fine;
    const Box2D fine_domain = amr_level_index_domain(geom_.domain, fine_level);
    if (fine_domain != coarse_box.refine(refinement_ratio))
      throw std::runtime_error("AMR transfer levels disagree with the refinement ratio");
    return runtime::amr::SpatialTransferContext{
        coarse_level, fine_level, coarse.ncomp(),
        runtime::amr::IndexTransform{{coarse_box.lo[0], coarse_box.lo[1]},
                                     {fine_domain.lo[0], fine_domain.lo[1]},
                                     {refinement_ratio, refinement_ratio}},
        coarse_box, fine_domain, replicated_parent, periodicity};
  }
  void set_block_transfer_authority(std::size_t block,
                                    runtime::amr::PreparedTransferKernel prolongation,
                                    runtime::amr::PreparedTransferKernel restriction,
                                    runtime::amr::PreparedTransferKernel coarse_fine,
                                    runtime::amr::PreparedTransferKernel temporal,
                                    int refinement_ratio) {
    const auto capabilities = coarse_fine.capabilities;
    if (block >= blocks_.size() || !prolongation.spatial || !restriction.spatial ||
        !coarse_fine.coarse_fine || !coarse_fine.prepared_coarse_fine ||
        !temporal.temporal || refinement_ratio != kAmrRefRatio ||
        bootstrap_pending_ || capabilities.order < 1 ||
        (capabilities.ghost_depth.size() != 1 && capabilities.ghost_depth.size() != 2) ||
        std::any_of(capabilities.ghost_depth.begin(), capabilities.ghost_depth.end(),
                    [](int depth) { return depth <= 0; }))
      throw std::runtime_error("AmrRuntime::set_block_transfer_authority invalid manifest");
    coarse_fine.prepared_coarse_fine->validate();
    BlockTransferAuthority candidate{
        std::move(prolongation), std::move(restriction), std::move(coarse_fine),
        std::move(temporal), refinement_ratio, true, capabilities};
    if (nlev_ > 1)
      require_coarse_fine_reconstruction_contract_(block, candidate);
    block_transfer_authorities_[block] = std::move(candidate);
    // The prepared FillPatch data plane retains the exact native spatial identity selected by
    // resolution.  Route installation is a build-time event; recreating topology workspaces here
    // cannot enter the stepping hot path.
    rematerialize_persistent_topology_resources_(topology_materialization_generation_);
  }

  MultiFab regrid_block_field(std::size_t block, const BoxArray& boxes,
                              const DistributionMapping& distribution, const MultiFab& parent,
                              const MultiFab& old_fine, int parent_level, int ghost_depth,
                              int refinement_ratio) const {
    if (block >= block_transfer_authorities_.size())
      throw std::runtime_error("AmrRuntime::regrid_block_field block out of range");
    const auto& authority = block_transfer_authorities_[block];
    if (!authority.prepared || !authority.prolongation.spatial ||
        authority.refinement_ratio != refinement_ratio)
      throw std::runtime_error(
          "AmrRuntime regrid has no compatible prepared prolongation authority");
    RegridProlongation prolong = [this, &authority](const MultiFab& coarse, MultiFab& fine,
                                                    int coarse_level, int ratio,
                                                    bool replicated_parent,
                                                    const CommunicatorView&) {
      authority.prolongation.spatial(
          coarse, fine,
          bootstrap_transfer_context(coarse, fine, coarse_level, coarse_level + 1, ratio,
                                     replicated_parent, base_per_));
    };
    return regrid_field_on_layout_with_provider(boxes, distribution, parent, old_fine,
                                                parent_level, ghost_depth, prolong,
                                                world_communicator_view(), replicated_coarse_,
                                                refinement_ratio);
  }

  void restrict_block_field(std::size_t block, const MultiFab& fine, MultiFab& parent,
                            int parent_level, int refinement_ratio) const {
    if (block >= block_transfer_authorities_.size())
      throw std::runtime_error("AmrRuntime::restrict_block_field block out of range");
    const auto& authority = block_transfer_authorities_[block];
    if (!authority.prepared || !authority.restriction.spatial ||
        authority.refinement_ratio != refinement_ratio)
      throw std::runtime_error(
          "AmrRuntime coarsening has no compatible prepared restriction authority");
    authority.restriction.spatial(
        fine, parent,
        bootstrap_transfer_context(parent, fine, parent_level, parent_level + 1,
                                   refinement_ratio, parent_level == 0 && replicated_coarse_));
  }
  void set_tagging_program(std::vector<TaggingProgram::Stencil> stencils,
                           std::vector<TaggingProgram::Leaf> leaves,
                           std::vector<std::int32_t> refine_ops,
                           std::vector<std::int32_t> refine_args,
                           std::vector<std::int32_t> coarsen_ops,
                           std::vector<std::int32_t> coarsen_args, int min_cycles,
                           int equality_policy, int conflict_policy, std::string clock_identity,
                           std::string provider_identity) {
    if (bootstrap_pending_ || clock_identity.empty() || provider_identity.empty() ||
        leaves.empty() || refine_ops.empty() || refine_ops.size() != refine_args.size() ||
        coarsen_ops.size() != coarsen_args.size() || min_cycles < 0 ||
        refine_ops.size() + coarsen_ops.size() > POPS_TAGGING_MAXIMUM_INSTRUCTION_COUNT_V1 ||
        equality_policy < 0 || equality_policy > 2 || conflict_policy < 0 || conflict_policy > 3)
      throw std::runtime_error("AmrRuntime::set_tagging_program invalid manifest");
    if (min_cycles != 0)
      throw std::runtime_error(
          "AmrRuntime::set_tagging_program min_cycles requires a persistent tagging-state "
          "provider");
    for (const auto& leaf : leaves) {
      const bool gradient = leaf.opcode == POPS_TAGGING_GRADIENT_ABOVE_V1 ||
                            leaf.opcode == POPS_TAGGING_GRADIENT_BELOW_V1;
      const bool shared_aux = leaf.state_index == blocks_.size();
      const std::size_t component_count =
          shared_aux ? static_cast<std::size_t>(aux_ncomp_)
                     : (leaf.state_index < blocks_.size()
                            ? static_cast<std::size_t>(blocks_[leaf.state_index].ncomp)
                            : std::size_t{0});
      if (leaf.state_index > blocks_.size() || leaf.component >= component_count ||
          !pops_tagging_opcode_is_leaf_v1(leaf.opcode) ||
          !std::isfinite(static_cast<double>(leaf.threshold)) ||
          gradient != (leaf.stencil_index != POPS_TAGGING_NO_STENCIL_V1) ||
          (gradient && leaf.stencil_index >= stencils.size()))
        throw std::runtime_error("AmrRuntime::set_tagging_program invalid leaf");
    }
    const auto validate_program = [&leaves](const std::vector<std::int32_t>& ops,
                                            const std::vector<std::int32_t>& args, bool required) {
      if (ops.empty()) {
        if (required)
          throw std::runtime_error("AMR tagging program has no refine root");
        return;
      }
      if (ops.size() > POPS_TAGGING_MAXIMUM_INSTRUCTION_COUNT_V1)
        throw std::runtime_error("AMR tagging program exceeds the native instruction bound");
      int depth = 0;
      for (std::size_t index = 0; index < ops.size(); ++index) {
        const int op = ops[index], arg = args[index];
        if (pops_tagging_opcode_is_leaf_v1(op)) {
          if (arg < 0 || static_cast<std::size_t>(arg) >= leaves.size() ||
              leaves[static_cast<std::size_t>(arg)].opcode != op)
            throw std::runtime_error("AMR tagging leaf opcode/index mismatch");
          ++depth;
        } else if (op == POPS_TAGGING_ANY_OF_V1 || op == POPS_TAGGING_ALL_OF_V1) {
          if (arg < 2 || depth < arg)
            throw std::runtime_error("AMR tagging n-ary stack underflow");
          depth -= arg - 1;
        } else if (op == POPS_TAGGING_NOT_V1) {
          if (arg != 1 || depth < 1)
            throw std::runtime_error("AMR tagging not stack underflow");
        } else {
          throw std::runtime_error("AMR tagging program has an unknown opcode");
        }
      }
      if (depth != 1)
        throw std::runtime_error("AMR tagging program must leave exactly one result");
    };
    validate_program(refine_ops, refine_args, true);
    validate_program(coarsen_ops, coarsen_args, false);
    for (const auto& leaf : leaves) {
      const bool gradient = leaf.opcode == POPS_TAGGING_GRADIENT_ABOVE_V1 ||
                            leaf.opcode == POPS_TAGGING_GRADIENT_BELOW_V1;
      if (!gradient || (base_per_.x && base_per_.y))
        continue;
      if (leaf.state_index == blocks_.size()) {
        if (aux_.empty() || aux_.front().n_grow() == 0)
          throw std::runtime_error(
              "non-periodic AMR gradient tagging requires shared-aux ghost storage");
        continue;
      }
      const auto& block = blocks_[leaf.state_index];
      if (!block.boundary_plan || block.boundary_plan->has_omitted_faces())
        throw std::runtime_error(
            "non-periodic AMR gradient tagging requires a complete prepared ghost-production "
            "authority on every sampled physical face");
      block.boundary_plan->validate_state_layout(block.levels->front().U);
    }
    TaggingProgram candidate{std::move(stencils),
                             std::move(leaves),
                             std::move(refine_ops),
                             std::move(refine_args),
                             std::move(coarsen_ops),
                             std::move(coarsen_args),
                             equality_policy,
                             conflict_policy,
                             min_cycles,
                             POPS_TAGGING_NON_FINITE_REJECT_V1,
                             std::move(clock_identity),
                             std::move(provider_identity),
                             true};
    runtime::amr::validate_tagging_stencil_program(
        candidate, std::vector<std::string>{POPS_TAGGING_STENCIL_ROUTE_LINEAR_AXIS_STENCIL_L2_V1},
        POPS_TAGGING_MAXIMUM_STENCIL_TERMS_V1, 2, [this](std::size_t state_index) {
          if (state_index == blocks_.size()) {
            if (aux_.empty())
              throw std::invalid_argument("AMR Tagger shared aux is not materialized");
            return static_cast<std::size_t>(aux_.front().n_grow());
          }
          const auto& levels = *blocks_[state_index].levels;
          if (levels.empty())
            throw std::invalid_argument("AMR Tagger state has no materialized level");
          return static_cast<std::size_t>(levels.front().U.n_grow());
        });
    auto execution_candidate =
        make_tagging_execution_plan_(candidate, topology_materialization_generation_);
    tagging_program_ = std::move(candidate);
    tagging_execution_plan_ = std::move(execution_candidate);
  }

  void install_external_tagger(std::shared_ptr<runtime::amr::PreparedTaggerComponent> provider) {
    if (!provider || external_tagger_ || bootstrap_pending_)
      throw std::runtime_error("AmrRuntime external Tagger requires one pre-bootstrap provider");
    external_tagger_ = std::move(provider);
  }

  void install_external_clustering(
      std::shared_ptr<runtime::amr::PreparedClusteringComponent> provider) {
    if (!provider || external_clustering_ || bootstrap_pending_)
      throw std::runtime_error(
          "AmrRuntime external Clustering requires one pre-bootstrap provider");
    clustering_provider_ = provider;
    external_clustering_ = std::move(provider);
  }

  void set_component_logical_time(std::int64_t tick, double physical_time) {
    if (tick < 0 || !std::isfinite(physical_time))
      throw std::runtime_error("AmrRuntime component logical time is invalid");
    component_tick_ = tick;
    component_physical_time_ = physical_time;
  }

  void register_bootstrap_staggered_field(const std::string& subject,
                                          runtime::amr::TransferCentering centering, int ncomp,
                                          const std::vector<double>& values) {
    if (bootstrap_pending_ || subject.empty() || ncomp < 1 ||
        bootstrap_staggered_fields_.count(subject) != 0 ||
        (centering != runtime::amr::TransferCentering::FaceX &&
         centering != runtime::amr::TransferCentering::FaceY &&
         centering != runtime::amr::TransferCentering::Node))
      throw std::runtime_error(
          "AmrRuntime::register_bootstrap_staggered_field received an invalid descriptor");
    const MultiFab& cell = aux_[0];
    std::vector<Box2D> boxes;
    boxes.reserve(static_cast<std::size_t>(cell.box_array().size()));
    for (const Box2D& box : cell.box_array().boxes()) {
      Box2D staggered = box;
      if (centering == runtime::amr::TransferCentering::FaceX ||
          centering == runtime::amr::TransferCentering::Node)
        ++staggered.hi[0];
      if (centering == runtime::amr::TransferCentering::FaceY ||
          centering == runtime::amr::TransferCentering::Node)
        ++staggered.hi[1];
      boxes.push_back(staggered);
    }
    const int nx = dom_.nx() + (centering == runtime::amr::TransferCentering::FaceX ||
                                centering == runtime::amr::TransferCentering::Node);
    const int ny = dom_.ny() + (centering == runtime::amr::TransferCentering::FaceY ||
                                centering == runtime::amr::TransferCentering::Node);
    if (!values.empty() && values.size() != static_cast<std::size_t>(ncomp) * nx * ny)
      throw std::runtime_error(
          "AmrRuntime::register_bootstrap_staggered_field payload shape mismatch");
    MultiFab coarse(BoxArray(std::move(boxes)), cell.dmap(), ncomp, 1);
    coarse.set_val(Real(0));
    if (!values.empty()) {
      coarse.sync_host();
      for (int li = 0; li < coarse.local_size(); ++li) {
        Array4 data = coarse.fab(li).array();
        const Box2D valid = coarse.box(li);
        for (int c = 0; c < ncomp; ++c)
          for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
            for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
              data(i, j, c) = static_cast<Real>(
                  values[(static_cast<std::size_t>(c) * ny +
                          static_cast<std::size_t>(j - dom_.lo[1])) *
                             nx +
                         static_cast<std::size_t>(i - dom_.lo[0])]);
      }
    }
    BootstrapStaggeredField field;
    field.centering = centering;
    field.levels.push_back(std::move(coarse));
    bootstrap_staggered_fields_.emplace(subject, std::move(field));
  }

  std::int64_t prolong_bootstrap_block(std::size_t block, int level,
                                       const runtime::amr::PreparedTransferKernel& kernel,
                                       int refinement_ratio) {
    if (!bootstrap_pending_ || block >= blocks_.size() || level <= 0 || level >= nlev_ ||
        !kernel.spatial)
      throw std::runtime_error("AmrRuntime::prolong_bootstrap_block invalid route or phase");
    auto& levels = *blocks_[block].levels;
    const MultiFab& coarse = levels[static_cast<std::size_t>(level - 1)].U;
    MultiFab& fine = levels[static_cast<std::size_t>(level)].U;
    kernel.spatial(coarse, fine,
                   bootstrap_transfer_context(coarse, fine, level - 1, level, refinement_ratio));
    return fine.box_array().num_cells() * fine.ncomp();
  }

  std::int64_t fill_bootstrap_block_constant(std::size_t block, int level,
                                             const std::vector<double>& components) {
    if (!bootstrap_pending_ || block >= blocks_.size() || level < 0 || level >= nlev_ ||
        components.size() != static_cast<std::size_t>(blocks_[block].ncomp))
      throw std::runtime_error("AmrRuntime::fill_bootstrap_block_constant invalid target/width");
    MultiFab& values = (*blocks_[block].levels)[static_cast<std::size_t>(level)].U;
    for (int li = 0; li < values.local_size(); ++li)
      for (int component = 0; component < values.ncomp(); ++component)
        for_each_cell(values.box(li),
                      detail::BootstrapConstantKernel{values.fab(li).array(), component,
                                                      static_cast<Real>(components[component])});
    return values.box_array().num_cells() * values.ncomp();
  }

  std::int64_t fill_bootstrap_block_gaussian(std::size_t block, int level, Real center_x,
                                             Real center_y, Real background, Real amplitude,
                                             Real inverse_width) {
    if (!bootstrap_pending_ || block >= blocks_.size() || level < 0 || level >= nlev_ ||
        blocks_[block].ncomp != 1 || !(inverse_width > Real(0)))
      throw std::runtime_error("AmrRuntime::fill_bootstrap_block_gaussian invalid target/profile");
    MultiFab& values = (*blocks_[block].levels)[static_cast<std::size_t>(level)].U;
    const Box2D domain = dom_.refine(level_refinement(level));
    const Real dx = static_cast<Real>(geom_.xhi - geom_.xlo) / domain.nx();
    const Real dy = static_cast<Real>(geom_.yhi - geom_.ylo) / domain.ny();
    return analytic::materialize_gaussian_cell_average(
        values, static_cast<Real>(geom_.xlo), static_cast<Real>(geom_.ylo), dx, dy, center_x,
        center_y, background, amplitude, inverse_width);
  }

  std::int64_t fill_bootstrap_block_analytic(
      std::size_t block, int level,
      const std::vector<analytic::AnalyticProgram>& programs) {
    if (!bootstrap_pending_ || block >= blocks_.size() || level < 0 || level >= nlev_ ||
        programs.size() != static_cast<std::size_t>(blocks_[block].ncomp))
      throw std::runtime_error("AmrRuntime::fill_bootstrap_block_analytic invalid target/profile");
    MultiFab& values = (*blocks_[block].levels)[static_cast<std::size_t>(level)].U;
    const Box2D domain = dom_.refine(level_refinement(level));
    const Real dx = static_cast<Real>(geom_.xhi - geom_.xlo) / domain.nx();
    const Real dy = static_cast<Real>(geom_.yhi - geom_.ylo) / domain.ny();
    return analytic::materialize_cell_average(
        values, static_cast<Real>(geom_.xlo), static_cast<Real>(geom_.ylo), dx, dy, programs);
  }

  void synchronize_bootstrap_block(std::size_t block, int fine_level,
                                   const runtime::amr::PreparedTransferKernel& kernel,
                                   int refinement_ratio) {
    if (!bootstrap_pending_ || block >= blocks_.size() || fine_level <= 0 || fine_level >= nlev_ ||
        !kernel.spatial)
      throw std::runtime_error("AmrRuntime::synchronize_bootstrap_block invalid phase/level");
    auto& levels = *blocks_[block].levels;
    MultiFab& coarse = levels[static_cast<std::size_t>(fine_level - 1)].U;
    const MultiFab& fine = levels[static_cast<std::size_t>(fine_level)].U;
    kernel.spatial(
        fine, coarse,
        bootstrap_transfer_context(coarse, fine, fine_level - 1, fine_level, refinement_ratio));
  }

  std::int64_t prolong_bootstrap_staggered_field(const std::string& subject, int level,
                                                 const runtime::amr::PreparedTransferKernel& kernel,
                                                 int refinement_ratio) {
    auto found = bootstrap_staggered_fields_.find(subject);
    if (!bootstrap_pending_ || found == bootstrap_staggered_fields_.end() || level <= 0 ||
        level >= nlev_ || found->second.levels.size() != static_cast<std::size_t>(level) ||
        !kernel.spatial)
      throw std::runtime_error(
          "AmrRuntime::prolong_bootstrap_staggered_field invalid subject/phase");
    BootstrapStaggeredField& field = found->second;
    const MultiFab& cell = aux_[static_cast<std::size_t>(level)];
    std::vector<Box2D> boxes;
    boxes.reserve(static_cast<std::size_t>(cell.box_array().size()));
    for (const Box2D& box : cell.box_array().boxes()) {
      Box2D staggered = box;
      if (field.centering == runtime::amr::TransferCentering::FaceX ||
          field.centering == runtime::amr::TransferCentering::Node)
        ++staggered.hi[0];
      if (field.centering == runtime::amr::TransferCentering::FaceY ||
          field.centering == runtime::amr::TransferCentering::Node)
        ++staggered.hi[1];
      boxes.push_back(staggered);
    }
    MultiFab fine(BoxArray(std::move(boxes)), cell.dmap(), field.levels.back().ncomp(), 1);
    fine.set_val(Real(0));
    const MultiFab& parent = field.levels.back();
    kernel.spatial(parent, fine,
                   bootstrap_transfer_context(parent, fine, level - 1, level, refinement_ratio));
    const std::int64_t materialized = fine.box_array().num_cells() * fine.ncomp();
    field.levels.push_back(std::move(fine));
    return materialized;
  }

  std::int64_t prolong_bootstrap_face_vector(const std::string& subject_x,
                                             const std::string& subject_y, int level,
                                             const runtime::amr::PreparedTransferKernel& kernel,
                                             int refinement_ratio) {
    auto x = bootstrap_staggered_fields_.find(subject_x);
    auto y = bootstrap_staggered_fields_.find(subject_y);
    if (!bootstrap_pending_ || x == bootstrap_staggered_fields_.end() ||
        y == bootstrap_staggered_fields_.end() || level <= 0 || level >= nlev_ ||
        !kernel.face_vector || x->second.centering != runtime::amr::TransferCentering::FaceX ||
        y->second.centering != runtime::amr::TransferCentering::FaceY ||
        x->second.levels[0].ncomp() != y->second.levels[0].ncomp())
      throw std::runtime_error(
          "AmrRuntime::prolong_bootstrap_face_vector invalid paired subject/route");
    if (x->second.levels.size() == static_cast<std::size_t>(level + 1) &&
        y->second.levels.size() == static_cast<std::size_t>(level + 1))
      return x->second.levels.back().box_array().num_cells() * x->second.levels.back().ncomp() +
             y->second.levels.back().box_array().num_cells() * y->second.levels.back().ncomp();
    if (x->second.levels.size() != static_cast<std::size_t>(level) ||
        y->second.levels.size() != static_cast<std::size_t>(level))
      throw std::runtime_error(
          "AmrRuntime::prolong_bootstrap_face_vector has a partially materialized pair");

    const MultiFab& cell = aux_[static_cast<std::size_t>(level)];
    const auto allocate = [&](runtime::amr::TransferCentering centering, int ncomp) {
      std::vector<Box2D> boxes;
      boxes.reserve(static_cast<std::size_t>(cell.box_array().size()));
      for (const Box2D& box : cell.box_array().boxes()) {
        Box2D staggered = box;
        if (centering == runtime::amr::TransferCentering::FaceX)
          ++staggered.hi[0];
        else
          ++staggered.hi[1];
        boxes.push_back(staggered);
      }
      MultiFab result(BoxArray(std::move(boxes)), cell.dmap(), ncomp, 1);
      result.set_val(Real(0));
      return result;
    };
    MultiFab fine_x = allocate(x->second.centering, x->second.levels.back().ncomp());
    MultiFab fine_y = allocate(y->second.centering, y->second.levels.back().ncomp());
    const MultiFab& coarse_x = x->second.levels.back();
    const MultiFab& coarse_y = y->second.levels.back();
    kernel.face_vector(
        coarse_x, coarse_y, fine_x, fine_y,
        bootstrap_transfer_context(coarse_x, fine_x, level - 1, level, refinement_ratio));
    const std::int64_t materialized = fine_x.box_array().num_cells() * fine_x.ncomp() +
                                      fine_y.box_array().num_cells() * fine_y.ncomp();
    x->second.levels.push_back(std::move(fine_x));
    y->second.levels.push_back(std::move(fine_y));
    return materialized;
  }

  std::int64_t fill_bootstrap_staggered_constant(const std::string& subject, int level,
                                                 const std::vector<double>& components) {
    auto found = bootstrap_staggered_fields_.find(subject);
    if (found == bootstrap_staggered_fields_.end() || level < 0 || level >= nlev_ ||
        components.empty())
      throw std::runtime_error("AmrRuntime::fill_bootstrap_staggered_constant invalid subject");
    if (level > 0 && found->second.levels.size() == static_cast<std::size_t>(level)) {
      const MultiFab& topology = aux_[static_cast<std::size_t>(level)];
      std::vector<Box2D> boxes;
      boxes.reserve(static_cast<std::size_t>(topology.box_array().size()));
      for (const Box2D& box : topology.box_array().boxes()) {
        Box2D staggered = box;
        if (found->second.centering == runtime::amr::TransferCentering::FaceX ||
            found->second.centering == runtime::amr::TransferCentering::Node)
          ++staggered.hi[0];
        if (found->second.centering == runtime::amr::TransferCentering::FaceY ||
            found->second.centering == runtime::amr::TransferCentering::Node)
          ++staggered.hi[1];
        boxes.push_back(staggered);
      }
      MultiFab fine(BoxArray(std::move(boxes)), topology.dmap(), found->second.levels[0].ncomp(),
                    1);
      fine.set_val(Real(0));
      found->second.levels.push_back(std::move(fine));
    }
    if (found->second.levels.size() <= static_cast<std::size_t>(level) ||
        found->second.levels[static_cast<std::size_t>(level)].ncomp() !=
            static_cast<int>(components.size()))
      throw std::runtime_error("AmrRuntime::fill_bootstrap_staggered_constant width mismatch");
    MultiFab& values = found->second.levels[static_cast<std::size_t>(level)];
    for (int li = 0; li < values.local_size(); ++li)
      for (int component = 0; component < values.ncomp(); ++component)
        for_each_cell(values.box(li),
                      detail::BootstrapConstantKernel{values.fab(li).array(), component,
                                                      static_cast<Real>(components[component])});
    return values.box_array().num_cells() * values.ncomp();
  }

  std::vector<double> bootstrap_staggered_level(const std::string& subject, int level) const {
    const auto found = bootstrap_staggered_fields_.find(subject);
    if (found == bootstrap_staggered_fields_.end() || level < 0 ||
        level >= static_cast<int>(found->second.levels.size()))
      throw std::runtime_error("AmrRuntime::bootstrap_staggered_level has no such level");
    const MultiFab& values = found->second.levels[static_cast<std::size_t>(level)];
    const Box2D refined_domain = dom_.refine(level_refinement(level));
    const int nx =
        refined_domain.nx() + (found->second.centering == runtime::amr::TransferCentering::FaceX ||
                               found->second.centering == runtime::amr::TransferCentering::Node);
    const int ny =
        refined_domain.ny() + (found->second.centering == runtime::amr::TransferCentering::FaceY ||
                               found->second.centering == runtime::amr::TransferCentering::Node);
    std::vector<double> out(static_cast<std::size_t>(values.ncomp()) * nx * ny, 0.0);
    device_fence();
    for (int li = 0; li < values.local_size(); ++li) {
      const int global = values.global_index(li);
      const ConstArray4 data = values.fab(li).const_array();
      const Box2D valid = values.box(li);
      for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
        for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
          bool canonical = true;
          for (int previous = 0; previous < global; ++previous)
            if (values.box_array()[previous].contains(i, j)) {
              canonical = false;
              break;
            }
          if (!canonical)
            continue;
          for (int component = 0; component < values.ncomp(); ++component)
            out[(static_cast<std::size_t>(component) * ny + (j - refined_domain.lo[1])) * nx +
                (i - refined_domain.lo[0])] = static_cast<double>(data(i, j, component));
        }
    }
    // A replicated coarse carrier is already complete on every rank.  Fine levels and a
    // distributed coarse carrier contain disjoint ownership contributions and require a gather.
    if (n_ranks() > 1 && (level > 0 || !replicated_coarse_))
      all_reduce_sum_inplace(out.data(), out.size());
    return out;
  }

  std::uint64_t invalidate_bootstrap_cache(const std::string& subject) {
    if (!bootstrap_pending_ || subject.empty())
      throw std::runtime_error("AmrRuntime::invalidate_bootstrap_cache invalid phase/key");
    BootstrapCacheState& cache = bootstrap_caches_[subject];
    ++cache.epoch;
    cache.valid = false;
    cache.materialized_level = -1;
    cache.topology.clear();
    return cache.epoch;
  }

  const BootstrapCacheState& rebuild_bootstrap_topology_cache(const std::string& subject,
                                                              int level) {
    auto found = bootstrap_caches_.find(subject);
    if (!bootstrap_pending_ || found == bootstrap_caches_.end() || found->second.valid ||
        level < 0 || level >= nlev_)
      throw std::runtime_error(
          "AmrRuntime::rebuild_bootstrap_topology_cache requires invalidation in transaction");
    BootstrapCacheState& cache = found->second;
    cache.topology.push_back(PatchBox{0, 0, 0, dom_.nx() - 1, dom_.ny() - 1});
    for (const PatchBox& box : patch_boxes())
      if (box.level <= level)
        cache.topology.push_back(box);
    cache.materialized_level = level;
    cache.valid = true;
    return cache;
  }

  const BootstrapCacheState& bootstrap_cache(const std::string& subject) const {
    const auto found = bootstrap_caches_.find(subject);
    if (found == bootstrap_caches_.end())
      throw std::runtime_error("AmrRuntime::bootstrap_cache has no registered cache");
    return found->second;
  }
  std::size_t n_blocks() const { return blocks_.size(); }
  /// Conservative VariableSet (names + physical roles, Model::conservative_vars()) of block @p b. The
  /// SAME cons_vars that add_coupled_source resolves (block, role) against; exposed read-only so the
  /// facade resolves a name/role-selected regrid variable into a component (ADC-296). @throws OOB @p b.
  const VariableSet& block_cons_vars(std::size_t b) const {
    if (b >= blocks_.size())
      throw std::runtime_error("AmrRuntime::block_cons_vars : block index out of bounds");
    return blocks_[b].cons_vars;
  }
  std::size_t n_coupled_sources() const { return coupled_sources_.size(); }
  /// Read-only view of the registered coupling operators (ADC-595, parity with System): label plus the
  /// declared conservation / frequency contracts, in registration order, so a Program or a runtime
  /// report enumerates the AMR couplings as typed operators. A raw add_coupled_source registers an
  /// "unchecked" (empty-contract) entry; add_coupling_operator records the declared contract.
  const std::vector<CouplingOperatorView>& coupled_operators() const { return coupled_operators_; }
  MultiFab& phi() { return default_field_solver_->phi_level(0); }
  // System Poisson right-hand side after the last solve_fields: f = Sum_b elliptic_rhs_b(U_b) on the
  // shared coarse. Exposed to check the CO-LOCATED SUM (PR1 test); same grid as the coarse (the
  // blocks' contributions are accumulated there at the same cells).
  MultiFab& poisson_rhs() { return default_field_solver_->rhs_level(0); }
  const MultiFab& aux(int k) const { return aux_[k]; }
  std::vector<AmrLevelMP>& levels(std::size_t b) { return *blocks_[b].levels; }
  Real mass(std::size_t b) const { return blocks_[b].mass(); }
  std::vector<double> density(std::size_t b) const { return blocks_[b].density(); }
  int solve_count() const { return solve_count_; }
  int regrid_count() const { return regrid_count_; }
  std::uint64_t topology_epoch() const { return topology_epoch_; }
  /// Process-local identity of the currently materialized hierarchy storage. Unlike the
  /// checkpointed epoch, this generation is never restored to an older value: rebuilding a
  /// checkpoint or rolling back a topology-changing attempt must invalidate address/layout-bound
  /// Program resources even when the restored epoch and level count numerically match.
  std::uint64_t topology_materialization_generation() const {
    return topology_materialization_generation_;
  }
  void restore_checkpoint_counters(int regrid_count, std::uint64_t topology_epoch) {
    if (regrid_count < 0)
      throw std::runtime_error("AMR checkpoint regrid count must be non-negative");
    regrid_count_ = regrid_count;
    topology_epoch_ = topology_epoch;
  }

  static bool exact_snapshot_layout_(const MultiFab& destination,
                                     const MultiFab& source) noexcept {
    return destination.box_array().boxes() == source.box_array().boxes() &&
           destination.dmap().ranks() == source.dmap().ranks() &&
           destination.ncomp() == source.ncomp() && destination.n_grow() == source.n_grow();
  }

  static void copy_snapshot_storage_(MultiFab& destination, const MultiFab& source) {
    if (!exact_snapshot_layout_(destination, source))
      destination = MultiFab(source.box_array(), source.dmap(), source.ncomp(), source.n_grow());
    detail::copy_amr_storage(destination, source);
  }

  template <class Map, class Copier>
  static void reconcile_snapshot_map_(Map& destination, const Map& source, Copier&& copy) {
    for (auto entry = destination.begin(); entry != destination.end();) {
      if (source.find(entry->first) == source.end())
        entry = destination.erase(entry);
      else
        ++entry;
    }
    for (const auto& [key, saved] : source) {
      auto [entry, inserted] = destination.try_emplace(key);
      (void)inserted;
      copy(entry->second, saved);
    }
  }

  template <class Map>
  static void copy_snapshot_value_map_(Map& destination, const Map& source) {
    reconcile_snapshot_map_(destination, source,
                            [](auto& value, const auto& saved) { value = saved; });
  }

  template <class Map>
  static void copy_snapshot_vector_map_(Map& destination, const Map& source) {
    reconcile_snapshot_map_(destination, source, [](auto& values, const auto& saved_values) {
      values.resize(saved_values.size());
      std::copy(saved_values.begin(), saved_values.end(), values.begin());
    });
  }

  static void copy_snapshot_hierarchy_(AmrHierarchyLayout& destination,
                                       const AmrHierarchyLayout& source) {
    destination.ba.resize(source.ba.size());
    destination.dm.resize(source.dm.size());
    for (std::size_t level = 0; level < source.ba.size(); ++level) {
      destination.ba[level] = source.ba[level];
      destination.dm[level] = source.dm[level];
    }
    destination.dx.resize(source.dx.size());
    destination.dy.resize(source.dy.size());
    destination.refinement_ratios.resize(source.refinement_ratios.size());
    std::copy(source.dx.begin(), source.dx.end(), destination.dx.begin());
    std::copy(source.dy.begin(), source.dy.end(), destination.dy.begin());
    std::copy(source.refinement_ratios.begin(), source.refinement_ratios.end(),
              destination.refinement_ratios.begin());
    destination.load_balance = source.load_balance;
  }

  static void copy_snapshot_bootstrap_caches_(
      std::map<std::string, BootstrapCacheState>& destination,
      const std::map<std::string, BootstrapCacheState>& source) {
    reconcile_snapshot_map_(destination, source, [](auto& cache, const auto& saved) {
      cache.epoch = saved.epoch;
      cache.valid = saved.valid;
      cache.materialized_level = saved.materialized_level;
      cache.topology.resize(saved.topology.size());
      std::copy(saved.topology.begin(), saved.topology.end(), cache.topology.begin());
    });
  }

  static void copy_snapshot_storage_vector_(std::vector<MultiFab>& destination,
                                            const std::vector<MultiFab>& source) {
    destination.resize(source.size());
    for (std::size_t index = 0; index < source.size(); ++index)
      copy_snapshot_storage_(destination[index], source[index]);
  }

  static void copy_snapshot_history_rings_(
      std::map<std::string, std::vector<std::vector<MultiFab>>>& destination,
      const std::map<std::string, std::vector<std::vector<MultiFab>>>& source) {
    reconcile_snapshot_map_(destination, source, [](auto& ring, const auto& saved_ring) {
      ring.resize(saved_ring.size());
      for (std::size_t slot = 0; slot < saved_ring.size(); ++slot)
        copy_snapshot_storage_vector_(ring[slot], saved_ring[slot]);
    });
  }

  static void copy_snapshot_staggered_fields_(
      std::map<std::string, BootstrapStaggeredField>& destination,
      const std::map<std::string, BootstrapStaggeredField>& source) {
    reconcile_snapshot_map_(destination, source, [](auto& field, const auto& saved_field) {
      field.centering = saved_field.centering;
      copy_snapshot_storage_vector_(field.levels, saved_field.levels);
    });
  }

  static bool exact_snapshot_storage_vector_layout_(const std::vector<MultiFab>& destination,
                                                    const std::vector<MultiFab>& source) noexcept {
    if (destination.size() != source.size())
      return false;
    for (std::size_t index = 0; index < source.size(); ++index)
      if (!exact_snapshot_layout_(destination[index], source[index]))
        return false;
    return true;
  }

  template <class Map>
  static bool same_snapshot_map_keys_(const Map& live, const Map& saved) {
    if (live.size() != saved.size())
      return false;
    auto live_entry = live.begin();
    auto saved_entry = saved.begin();
    for (; live_entry != live.end(); ++live_entry, ++saved_entry)
      if (live_entry->first != saved_entry->first)
        return false;
    return true;
  }

  template <class Map>
  static bool same_snapshot_vector_map_shape_(const Map& live, const Map& saved) {
    if (!same_snapshot_map_keys_(live, saved))
      return false;
    for (const auto& [key, saved_values] : saved)
      if (live.at(key).size() != saved_values.size())
        return false;
    return true;
  }

  bool step_snapshot_layout_matches_(const StepSnapshot& saved) const {
    if (saved.nlev != nlev_ || saved.block_levels.size() != blocks_.size() ||
        saved.has_newton_report.size() != blocks_.size() ||
        saved.newton_reports.size() != blocks_.size() ||
        saved.hierarchy.ba.size() != hierarchy_.ba.size() ||
        saved.hierarchy.dm.size() != hierarchy_.dm.size() || saved.hierarchy.dx != hierarchy_.dx ||
        saved.hierarchy.dy != hierarchy_.dy ||
        saved.hierarchy.refinement_ratios != hierarchy_.refinement_ratios ||
        saved.hierarchy.load_balance != hierarchy_.load_balance || saved.aux.size() != aux_.size() ||
        saved.has_profiler != (profiler_ != nullptr) ||
        !same_snapshot_map_keys_(hist_depth_, saved.history_depth) ||
        !same_snapshot_map_keys_(hist_block_owner_, saved.history_block_owner) ||
        !same_snapshot_vector_map_shape_(hist_init_, saved.history_initialized) ||
        !same_snapshot_vector_map_shape_(hist_slot_dt_, saved.history_slot_dt) ||
        !same_snapshot_map_keys_(bootstrap_caches_, saved.bootstrap_caches))
      return false;
    for (std::size_t level = 0; level < hierarchy_.ba.size(); ++level)
      if (saved.hierarchy.ba[level].boxes() != hierarchy_.ba[level].boxes() ||
          saved.hierarchy.dm[level].ranks() != hierarchy_.dm[level].ranks() ||
          !exact_snapshot_layout_(saved.aux[level], aux_[level]))
        return false;
    for (std::size_t block = 0; block < blocks_.size(); ++block) {
      const auto& live = *blocks_[block].levels;
      const auto& accepted = saved.block_levels[block];
      const bool accepted_report =
          block < saved.has_newton_report.size() && saved.has_newton_report[block];
      if (live.size() != accepted.size() ||
          static_cast<bool>(blocks_[block].newton_report) != accepted_report)
        return false;
      for (std::size_t level = 0; level < live.size(); ++level)
        if (live[level].dx != accepted[level].dx || live[level].dy != accepted[level].dy ||
            !exact_snapshot_layout_(live[level].U, accepted[level].U))
          return false;
    }
    if (!exact_snapshot_layout_(default_field_solver_->phi_level(0), saved.phi) ||
        !exact_snapshot_layout_(default_field_solver_->rhs_level(0), saved.poisson_rhs) ||
        named_fields_.size() != saved.named_fields.size() ||
        hist_rings_.size() != saved.history_rings.size() ||
        bootstrap_staggered_fields_.size() != saved.staggered_fields.size())
      return false;
    for (const auto& [name, accepted] : saved.named_fields) {
      const auto live = named_fields_.find(name);
      if (live == named_fields_.end() || static_cast<bool>(live->second.solver) != accepted.allocated)
        return false;
      if (!accepted.allocated)
        continue;
      if (live->second.solver->couples_hierarchy_levels() != accepted.composite ||
          live->second.solver->level_count() != static_cast<int>(accepted.phi.size()) ||
          accepted.phi.size() != accepted.rhs.size())
        return false;
      for (std::size_t level = 0; level < accepted.phi.size(); ++level)
        if (!exact_snapshot_layout_(live->second.solver->phi_level(static_cast<int>(level)),
                                    accepted.phi[level]) ||
            !exact_snapshot_layout_(live->second.solver->rhs_level(static_cast<int>(level)),
                                    accepted.rhs[level]))
          return false;
    }
    for (const auto& [name, accepted_ring] : saved.history_rings) {
      const auto live = hist_rings_.find(name);
      if (live == hist_rings_.end() || live->second.size() != accepted_ring.size())
        return false;
      for (std::size_t slot = 0; slot < accepted_ring.size(); ++slot)
        if (!exact_snapshot_storage_vector_layout_(live->second[slot], accepted_ring[slot]))
          return false;
    }
    for (const auto& [name, accepted] : saved.staggered_fields) {
      const auto live = bootstrap_staggered_fields_.find(name);
      if (live == bootstrap_staggered_fields_.end() ||
          live->second.centering != accepted.centering ||
          !exact_snapshot_storage_vector_layout_(live->second.levels, accepted.levels))
        return false;
    }
    for (const auto& [name, accepted] : saved.bootstrap_caches) {
      const auto live = bootstrap_caches_.find(name);
      if (live == bootstrap_caches_.end() ||
          live->second.topology.size() != accepted.topology.size())
        return false;
    }
    return true;
  }

  /// Capture/restore the accepted AMR state around one public step attempt.  The snapshot includes
  /// fine layouts, every block/level, shared aux and elliptic warm starts, history rings, diagnostics
  /// and cadence counters.  Restore also rewires every AmrLevelMP::aux pointer after replacing the
  /// topology-carrying MultiFabs.
  void capture_step_snapshot(StepSnapshot& out) {
    out.block_levels.resize(blocks_.size());
    out.has_newton_report.resize(blocks_.size());
    out.newton_reports.resize(blocks_.size());
    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
      const auto& block = blocks_[block_index];
      auto& saved_levels = out.block_levels[block_index];
      saved_levels.resize(block.levels->size());
      for (std::size_t level = 0; level < block.levels->size(); ++level) {
        copy_snapshot_storage_(saved_levels[level].U, (*block.levels)[level].U);
        saved_levels[level].aux = nullptr;
        saved_levels[level].dx = (*block.levels)[level].dx;
        saved_levels[level].dy = (*block.levels)[level].dy;
      }
      out.has_newton_report[block_index] = block.newton_report ? char(1) : char(0);
      out.newton_reports[block_index] =
          block.newton_report ? *block.newton_report : NewtonReport{};
    }
    copy_snapshot_hierarchy_(out.hierarchy, hierarchy_);
    copy_snapshot_storage_vector_(out.aux, aux_);
    copy_snapshot_storage_(out.phi, default_field_solver_->phi_level(0));
    copy_snapshot_storage_(out.poisson_rhs, default_field_solver_->rhs_level(0));
    for (auto& [name, field] : named_fields_) {
      auto& saved = out.named_fields[name];
      saved.allocated = static_cast<bool>(field.solver);
      saved.composite = false;
      if (field.solver) {
        saved.composite = field.solver->couples_hierarchy_levels();
        saved.phi.resize(static_cast<std::size_t>(field.solver->level_count()));
        saved.rhs.resize(static_cast<std::size_t>(field.solver->level_count()));
        for (int k = 0; k < field.solver->level_count(); ++k) {
          copy_snapshot_storage_(saved.phi[static_cast<std::size_t>(k)],
                                 field.solver->phi_level(k));
          copy_snapshot_storage_(saved.rhs[static_cast<std::size_t>(k)],
                                 field.solver->rhs_level(k));
        }
      } else {
        saved.phi.clear();
        saved.rhs.clear();
      }
    }
    for (auto entry = out.named_fields.begin(); entry != out.named_fields.end();) {
      if (named_fields_.find(entry->first) == named_fields_.end())
        entry = out.named_fields.erase(entry);
      else
        ++entry;
    }
    copy_snapshot_history_rings_(out.history_rings, hist_rings_);
    copy_snapshot_value_map_(out.history_depth, hist_depth_);
    copy_snapshot_value_map_(out.history_block_owner, hist_block_owner_);
    copy_snapshot_vector_map_(out.history_initialized, hist_init_);
    copy_snapshot_vector_map_(out.history_slot_dt, hist_slot_dt_);
    copy_snapshot_staggered_fields_(out.staggered_fields, bootstrap_staggered_fields_);
    copy_snapshot_bootstrap_caches_(out.bootstrap_caches, bootstrap_caches_);
    out.last_dt_reason = last_dt_reason_;
    out.nlev = nlev_;
    out.macro_step = macro_step_;
    out.solve_count = solve_count_;
    out.regrid_count = regrid_count_;
    out.topology_epoch = topology_epoch_;
    out.has_profiler = profiler_ != nullptr;
    if (profiler_ != nullptr)
      out.profiler = *profiler_;
  }

  StepSnapshot step_snapshot() {
    StepSnapshot out;
    capture_step_snapshot(out);
    return out;
  }

  void begin_step_rollback_scope() {
    if (step_rollback_scope_depth_ == std::numeric_limits<int>::max())
      throw std::overflow_error("AMR rollback scope depth exceeds int range");
    ++step_rollback_scope_depth_;
  }

  void end_step_rollback_scope() {
    if (step_rollback_scope_depth_ <= 0)
      throw std::logic_error("AMR rollback scope is not active");
    --step_rollback_scope_depth_;
  }

  bool step_rollback_scope_active() const noexcept { return step_rollback_scope_depth_ > 0; }

  void restore_step_snapshot(const StepSnapshot& saved) {
    if (saved.block_levels.size() != blocks_.size())
      throw std::runtime_error(
          "AmrRuntime::restore_step_snapshot: snapshot/runtime composition mismatch");

    // A rejected step can still have Kokkos kernels in flight.  The snapshot
    // restore below replaces MultiFab storage (and can therefore release the
    // buffers those kernels borrowed), so this is the rollback ownership
    // boundary: complete device work before the first copy or destruction.
    device_fence();

    const bool exact_layout = step_snapshot_layout_matches_(saved);
    if (exact_layout) {
      for (std::size_t block = 0; block < blocks_.size(); ++block) {
        auto& live_levels = *blocks_[block].levels;
        const auto& accepted_levels = saved.block_levels[block];
        for (std::size_t level = 0; level < accepted_levels.size(); ++level) {
          detail::copy_amr_storage(live_levels[level].U, accepted_levels[level].U);
          live_levels[level].dx = accepted_levels[level].dx;
          live_levels[level].dy = accepted_levels[level].dy;
        }
        const bool had_report =
            block < saved.has_newton_report.size() && saved.has_newton_report[block];
        if (had_report)
          *blocks_[block].newton_report = saved.newton_reports[block];
      }
      nlev_ = saved.nlev;
      for (std::size_t level = 0; level < aux_.size(); ++level)
        detail::copy_amr_storage(aux_[level], saved.aux[level]);
      for (auto& block : blocks_)
        for (int level = 0; level < nlev_; ++level)
          (*block.levels)[static_cast<std::size_t>(level)].aux =
              &aux_[static_cast<std::size_t>(level)];
      detail::copy_amr_storage(default_field_solver_->phi_level(0), saved.phi);
      detail::copy_amr_storage(default_field_solver_->rhs_level(0), saved.poisson_rhs);
      for (const auto& [name, accepted] : saved.named_fields) {
        auto& solver = named_fields_.at(name).solver;
        if (!accepted.allocated)
          continue;
        for (std::size_t level = 0; level < accepted.phi.size(); ++level) {
          detail::copy_amr_storage(solver->phi_level(static_cast<int>(level)),
                                   accepted.phi[level]);
          detail::copy_amr_storage(solver->rhs_level(static_cast<int>(level)),
                                   accepted.rhs[level]);
        }
      }
      copy_snapshot_history_rings_(hist_rings_, saved.history_rings);
      copy_snapshot_value_map_(hist_depth_, saved.history_depth);
      copy_snapshot_value_map_(hist_block_owner_, saved.history_block_owner);
      copy_snapshot_vector_map_(hist_init_, saved.history_initialized);
      copy_snapshot_vector_map_(hist_slot_dt_, saved.history_slot_dt);
      copy_snapshot_staggered_fields_(bootstrap_staggered_fields_, saved.staggered_fields);
      copy_snapshot_bootstrap_caches_(bootstrap_caches_, saved.bootstrap_caches);
      last_dt_reason_ = saved.last_dt_reason;
      macro_step_ = saved.macro_step;
      solve_count_ = saved.solve_count;
      regrid_count_ = saved.regrid_count;
      topology_epoch_ = saved.topology_epoch;
      if (saved.has_profiler && profiler_ != nullptr)
        *profiler_ = saved.profiler;
      // A completed rollback is immediately observable by host diagnostics and may be followed by a
      // cold topology replacement. Keep snapshot storage alive, but complete every device copy now.
      device_fence();
      return;
    }

    const bool rematerialize_boundary_sessions =
        std::any_of(blocks_.begin(), blocks_.end(),
                    [](const AmrRuntimeBlock& block) {
                      return static_cast<bool>(block.boundary_plan);
                    });
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
      auto& live_levels = *blocks_[b].levels;
      const auto& accepted_levels = saved.block_levels[b];
      live_levels.resize(accepted_levels.size());
      for (std::size_t level = 0; level < accepted_levels.size(); ++level) {
        copy_snapshot_storage_(live_levels[level].U, accepted_levels[level].U);
        live_levels[level].aux = nullptr;
        live_levels[level].dx = accepted_levels[level].dx;
        live_levels[level].dy = accepted_levels[level].dy;
      }
      const bool had_report = b < saved.has_newton_report.size() && saved.has_newton_report[b];
      if (!had_report) {
        blocks_[b].newton_report.reset();
      } else {
        if (!blocks_[b].newton_report)
          blocks_[b].newton_report = std::make_shared<NewtonReport>();
        *blocks_[b].newton_report = saved.newton_reports[b];
      }
    }
    nlev_ = saved.nlev;
    copy_snapshot_hierarchy_(hierarchy_, saved.hierarchy);
    refresh_active_temporal_plan_();
    copy_snapshot_storage_vector_(aux_, saved.aux);
    for (auto& block : blocks_)
      for (int k = 0; k < nlev_; ++k)
        (*block.levels)[static_cast<std::size_t>(k)].aux = &aux_[static_cast<std::size_t>(k)];
    copy_snapshot_storage_(default_field_solver_->phi_level(0), saved.phi);
    copy_snapshot_storage_(default_field_solver_->rhs_level(0), saved.poisson_rhs);
    // Solver objects own topology-specific BoxArrays.  Always rebuild them against the restored
    // accepted hierarchy before replaying their warm starts; a rejected regrid/bootstrap may have
    // replaced a level-local solver with FAC (or changed the number of level-local solvers).
    invalidate_named_field_topology();
    for (auto it = named_fields_.begin(); it != named_fields_.end();) {
      const auto accepted = saved.named_fields.find(it->first);
      if (accepted == saved.named_fields.end()) {
        it = named_fields_.erase(it);  // field registered provisionally during the rejected attempt
        continue;
      }
      auto& field = it->second;
      if (!accepted->second.allocated) {
        field.solver.reset();
        field.nullspace = {};
        field.level_nullspace.clear();
        field.nullspace_ready = false;
      } else {
        ensure_named_elliptic(field);
        if (field.solver->couples_hierarchy_levels() != accepted->second.composite ||
            accepted->second.phi.size() != static_cast<std::size_t>(field.solver->level_count()))
          throw std::runtime_error("AmrRuntime::restore_step_snapshot: field hierarchy mismatch");
        for (int k = 0; k < field.solver->level_count(); ++k) {
          copy_snapshot_storage_(field.solver->phi_level(k),
                                 accepted->second.phi[static_cast<std::size_t>(k)]);
          copy_snapshot_storage_(field.solver->rhs_level(k),
                                 accepted->second.rhs[static_cast<std::size_t>(k)]);
        }
      }
      ++it;
    }
    copy_snapshot_history_rings_(hist_rings_, saved.history_rings);
    copy_snapshot_value_map_(hist_depth_, saved.history_depth);
    copy_snapshot_value_map_(hist_block_owner_, saved.history_block_owner);
    copy_snapshot_vector_map_(hist_init_, saved.history_initialized);
    copy_snapshot_vector_map_(hist_slot_dt_, saved.history_slot_dt);
    copy_snapshot_staggered_fields_(bootstrap_staggered_fields_, saved.staggered_fields);
    copy_snapshot_bootstrap_caches_(bootstrap_caches_, saved.bootstrap_caches);
    last_dt_reason_ = saved.last_dt_reason;
    macro_step_ = saved.macro_step;
    solve_count_ = saved.solve_count;
    regrid_count_ = saved.regrid_count;
    topology_epoch_ = saved.topology_epoch;
    if (saved.has_profiler && profiler_ != nullptr)
      *profiler_ = saved.profiler;

    const auto& reference_levels = *blocks_.front().levels;
    for (std::size_t block = 1; block < blocks_.size(); ++block) {
      const auto& candidate_levels = *blocks_[block].levels;
      if (candidate_levels.size() != reference_levels.size())
        throw std::runtime_error(
            "AmrRuntime::restore_step_snapshot: restored blocks have different level counts");
      for (std::size_t level = 0; level < reference_levels.size(); ++level)
        if (!detail::same_level_layout(
                candidate_levels[level].U.box_array(), candidate_levels[level].U.dmap(),
                candidate_levels[level].dx, candidate_levels[level].dy,
                reference_levels[level].U.box_array(), reference_levels[level].U.dmap(),
                reference_levels[level].dx, reference_levels[level].dy))
          throw std::runtime_error(
              "AmrRuntime::restore_step_snapshot: restored blocks do not share one exact layout");
    }
    rematerialize_persistent_topology_resources_(
        next_topology_materialization_generation_());
    if (rematerialize_boundary_sessions)
      materialize_boundary_sessions_();
    device_fence();
  }

  /// @name Compiled time-Program AMR driver seam (epic ADC-508): per-level primitives exposing the
  /// engine internals an AmrProgramContext composes into a per-level macro-step. APPEND-ONLY: these
  /// surface existing storage / closures; none reimplement numerics. All are NO-OP-safe under MPI
  /// (loops over local_size()). The native AMR step does not call any of them.
  /// @{
  /// The live state MultiFab of block @p b at level @p k (zero-copy; same address an AmrProgramContext
  /// reads each macro-step). @c b is the AMR block index (sys_block-resolved by the caller).
  MultiFab& level_state(std::size_t b, int k) { return (*blocks_[b].levels)[k].U; }
  const MultiFab& level_state(std::size_t b, int k) const { return (*blocks_[b].levels)[k].U; }
  PreparedAmrTransitionAdvanceScratch& prepared_reflux_transition(std::size_t b,
                                                                  int child_level) {
    if (b >= blocks_.size() || !blocks_[b].advance_scratch_plan)
      throw std::logic_error("AMR block has no prepared reflux transition storage");
    return blocks_[b].advance_scratch_plan->transition_for_child(child_level);
  }
  /// Whether level @p k is present in full on every rank.  Replication is an ownership property of
  /// the runtime hierarchy, not something callers may infer from rank-local DistributionMapping
  /// metadata (the replicated level-0 mapping intentionally differs between ranks).
  bool level_is_replicated(int k) const {
    if (k < 0 || k >= nlev_)
      throw std::out_of_range("AmrRuntime::level_is_replicated level is out of range");
    return k == 0 && replicated_coarse_;
  }
  void project_level_state(std::size_t b, int k, MultiFab& state) {
    if (b >= blocks_.size() || k < 0 || k >= nlev_)
      throw std::out_of_range("AmrRuntime::project_level_state owner is out of range");
    auto& projection = blocks_[b].project_level_state;
    if (!projection)
      throw std::runtime_error(
          "AmrRuntime::project_level_state: owning block declares no pointwise projection");
    projection(state, aux_[static_cast<std::size_t>(k)]);
  }
  /// Geometry of level @p k: the coarse metric refined k times (dx/dy >> k, domain << k). The metric
  /// the per-level Laplacian / gradient / RHS read (parity with System's grid_context().geom).
  Geometry level_geom(int k) const { return geom_.refine(level_refinement(k)); }
  /// Transport BCRec derived from the base periodicity (periodic where periodic, else Foextrap) -- the
  /// SAME convention System::make_bc uses, so a Program's per-level ghost fill matches the System path.
  BCRec transport_bc() const {
    BCRec b;  // periodic by default
    if (!base_per_.x)
      b.xlo = b.xhi = BCType::Foextrap;
    if (!base_per_.y)
      b.ylo = b.yhi = BCType::Foextrap;
    return b;
  }
  /// Exact block/level grid authority used to materialize a prepared operator boundary session.
  /// The plan and N-ary registry are block-qualified; geometry, aux and ownership are level-qualified.
  GridContext level_grid_context(std::size_t block, int level) {
    if (block >= blocks_.size() || level < 0 || level >= nlev_)
      throw std::out_of_range("AmrRuntime::level_grid_context owner is out of range");
    const Geometry geometry = level_geom(level);
    GridContext context;
    context.dom = geometry.domain;
    context.bc = transport_bc();
    context.geom = geometry;
    context.aux = &const_cast<MultiFab&>(aux_[static_cast<std::size_t>(level)]);
    context.boundary_plan = blocks_[block].boundary_plan;
    if (blocks_[block].boundary_field_registry)
      context.boundary_field_registry = *blocks_[block].boundary_field_registry;
    context.coarse_fine_fill = [this, block, level](MultiFab& state) {
      fill_level_state_cf_ghosts(block, level, state);
    };
    return context;
  }
  /// BC of the coarse Poisson (for a matrix-free Krylov preconditioner's GeometricMG).
  const BCRec& poisson_bc() const { return bcPhi_; }
  /// A fresh scalar field co-distributed with level @p k's grid (its ba/dm), @p n_comp components,
  /// @p n_ghost ghosts, zero-initialized -- the Krylov scratch (r/p/Ap) of a per-level field solve.
  /// Counterpart of System::alloc_scalar_field, allocated from the runtime-owned layout.
  MultiFab level_scalar_field(int k, int n_comp, int n_ghost) const {
    MultiFab f(hierarchy_.ba.at(static_cast<std::size_t>(k)),
               hierarchy_.dm.at(static_cast<std::size_t>(k)), n_comp, n_ghost);
    f.set_val(Real(0));
    return f;
  }
  /// COARSE-FINE GHOST FILL for a per-level Program residual (ADC-634). A fine level (@p k >= 1) has a
  /// C/F interface whose ghosts sit UNDER the coarse level; the native Berger-Oliger step fills them by
  /// time-interpolation between the coarse old/new states (mf_fill_fine_ghosts_mb). The recursive
  /// Program driver normally calls fill_level_state_cf_ghosts_temporal with an explicit parent clock
  /// window; this spatial-only primitive applies the prepared C/F provider to the accepted parent.
  /// Without
  /// a coarse-fine fill the fine flux reads
  /// UNINITIALIZED C/F ghosts -> a zero/negative density -> a NaN pressure at the very first stage. The
  /// coarse level (@p k == 0) has base-domain physical ghosts only; the block's own level_rhs closure
  /// fills those (fill_boundary), so this is a fine-level-only pre-pass.
  void fill_level_state_cf_ghosts(std::size_t b, int k, MultiFab& U) {
    if (k < 1 || U.n_grow() == 0)
      return;
    const MultiFab& Uc = (*blocks_[b].levels)[k - 1].U;  // the parent (coarse) level state
    const auto& authority = block_transfer_authorities_.at(b);
    require_coarse_fine_reconstruction_contract_(b, authority);
    if (!authority.prepared || !authority.coarse_fine.coarse_fine)
      throw std::runtime_error("AmrRuntime: no prepared coarse/fine transfer authority");
    if (static_cast<std::size_t>(k - 1) >=
        blocks_[b].coarse_fine_spatial_workspaces.size())
      throw std::runtime_error("AmrRuntime: spatial coarse/fine workspace is not materialized");
    const bool replicated_parent = (k == 1) && replicated_coarse_;
    const CommunicatorView communicator =
        replicated_parent ? CommunicatorView{} : world_communicator_view();
    blocks_[b].coarse_fine_spatial_workspaces[static_cast<std::size_t>(k - 1)].apply(
        Uc, U, topology_materialization_generation_, communicator);
  }
  /// Coarse/fine fill at an explicitly qualified child time.  The temporal provider first builds the
  /// parent state at target_time from two distinct parent snapshots; the spatial provider then fills
  /// the child's interface ghosts.  There is no current-parent fallback on this path.
  void fill_level_state_cf_ghosts_temporal(
      std::size_t b, int k, MultiFab& U, const MultiFab& parent_old, const MultiFab& parent_new,
      const runtime::amr::TemporalTransferContext& target_time) {
    if (k < 1 || U.n_grow() == 0)
      return;
    const auto& authority = block_transfer_authorities_.at(b);
    require_coarse_fine_reconstruction_contract_(b, authority);
    if (!authority.prepared || !authority.coarse_fine.coarse_fine || !authority.temporal.temporal)
      throw std::runtime_error("AmrRuntime: no prepared spatial/temporal transfer authority");
    if (b >= temporal_parent_workspaces_.size() ||
        static_cast<std::size_t>(k - 1) >= temporal_parent_workspaces_[b].size())
      throw std::runtime_error("AmrRuntime: temporal parent workspace is not materialized");
    TemporalParentWorkspace& workspace =
        temporal_parent_workspaces_[b][static_cast<std::size_t>(k - 1)];
    if (workspace.topology_generation != topology_materialization_generation_ ||
        workspace.block != b || workspace.parent_level != k - 1 ||
        !same_exact_multifab_layout_(workspace.value, parent_old) ||
        !same_exact_multifab_layout_(workspace.value, parent_new))
      throw std::runtime_error(
          "AmrRuntime: temporal parent workspace crossed an exact topology contract");
    authority.temporal.temporal(parent_old, parent_new, workspace.value, target_time);
    if (static_cast<std::size_t>(k - 1) >=
        blocks_[b].coarse_fine_spatial_workspaces.size())
      throw std::runtime_error("AmrRuntime: spatial coarse/fine workspace is not materialized");
    const bool replicated_parent = (k == 1) && replicated_coarse_;
    const CommunicatorView communicator =
        replicated_parent ? CommunicatorView{} : world_communicator_view();
    blocks_[b].coarse_fine_spatial_workspaces[static_cast<std::size_t>(k - 1)].apply(
        workspace.value, U, topology_materialization_generation_, communicator);
  }
  /// R <- -div F(U) + S(U, aux_[k]) for block @p b on level @p k (the per-level analogue of
  /// System::block_rhs_into). Forwards to the block's level_rhs closure with the level metric + shared
  /// aux; fails loud if the block built no such closure (a host .so prototype). A fine level's C/F
  /// ghosts are refreshed from the coarse state first (ADC-634, the synchronous Program driver has no
  /// native FillPatch), so the fine flux never reads an uninitialized interface ghost.
  void level_rhs_into(std::size_t b, int k, MultiFab& U, MultiFab& R) {
    if (!blocks_[b].level_rhs)
      throw std::runtime_error(
          "AmrRuntime::level_rhs_into: block '" + blocks_[b].name +
          "' has no per-level residual closure (rebuild the AMR block via the production DSL "
          "target='amr_system')");
    fill_level_state_cf_ghosts(b, k, U);
    blocks_[b].level_rhs(U, aux_[k], level_geom(k), R);
  }

  void level_rhs_into_at(std::size_t b, int k,
                         const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
                         MultiFab& R) {
    if (b < blocks_.size() && k >= 0 &&
        static_cast<std::size_t>(k) < blocks_[b].boundary_sessions.size() &&
        blocks_[b].boundary_sessions[static_cast<std::size_t>(k)]) {
      auto& core = blocks_[b].level_rhs_core_at_point_prepared;
      auto& boundary = blocks_[b].level_boundary_residual_at_point_prepared;
      if (!core || !boundary)
        throw std::runtime_error(
            "AmrRuntime block lacks its persistent prepared boundary closures");
      const auto& session = *blocks_[b].boundary_sessions[static_cast<std::size_t>(k)];
      core(point, U, aux_[k], level_geom(k), R, session);
      boundary(point, U, aux_[k], level_geom(k), R, session);
      return;
    }
    if (!blocks_[b].level_rhs_at_point)
      throw std::runtime_error("AmrRuntime block has no point-qualified residual closure");
    fill_level_state_cf_ghosts(b, k, U);
    blocks_[b].level_rhs_at_point(point, U, aux_[k], level_geom(k), R);
  }

  bool has_boundary_linearization(std::size_t b) const {
    if (b >= blocks_.size())
      throw std::out_of_range("AmrRuntime boundary linearization block is out of range");
    return blocks_[b].boundary_plan && blocks_[b].boundary_plan->has_boundary_linearization();
  }

  void level_rhs_core_into_at(std::size_t b, int k,
                              const runtime::multiblock::BoundaryEvaluationPoint& point,
                              MultiFab& U, MultiFab& R, bool flux_only) {
    if (b >= blocks_.size() || k < 0 || k >= nlev_ || point.level != k)
      throw std::out_of_range("AmrRuntime core RHS level/block is out of range");
    if (interface_scheduler_.participates(b, k))
      throw std::runtime_error(
          "AmrRuntime implicit core RHS requires a coupled shared-interface solve");
    auto& closure = flux_only ? blocks_[b].level_neg_div_flux_core_at_point
                              : blocks_[b].level_rhs_core_at_point;
    if (!closure)
      throw std::runtime_error("AmrRuntime block lacks a point-qualified core residual closure");
    struct StageStateReset {
      std::optional<BoundaryStageStateView>* slot = nullptr;
      ~StageStateReset() {
        if (slot != nullptr)
          slot->reset();
      }
    } stage_reset;
    if (!blocks_[b].state_identity.empty()) {
      if (boundary_stage_states_) {
        if (boundary_stage_states_->point != point || boundary_stage_states_->state(b) != &U)
          throw std::runtime_error(
              "AmrRuntime core RHS disagrees with the active grouped stage-state registry");
      } else {
        boundary_stage_states_.emplace(
            BoundaryStageStateView{point, nullptr, static_cast<int>(b), &U});
        stage_reset.slot = &boundary_stage_states_;
      }
    }
    if (static_cast<std::size_t>(k) < blocks_[b].boundary_sessions.size() &&
        blocks_[b].boundary_sessions[static_cast<std::size_t>(k)]) {
      auto& prepared = flux_only ? blocks_[b].level_neg_div_flux_core_at_point_prepared
                                 : blocks_[b].level_rhs_core_at_point_prepared;
      if (!prepared)
        throw std::runtime_error(
            "AmrRuntime block lacks its persistent prepared core residual closure");
      prepared(point, U, aux_[k], level_geom(k), R,
               *blocks_[b].boundary_sessions[static_cast<std::size_t>(k)]);
      return;
    }
    fill_level_state_cf_ghosts(b, k, U);
    closure(point, U, aux_[k], level_geom(k), R);
  }

  void level_rhs_core_into_at(std::size_t b, int k,
                              const runtime::multiblock::BoundaryEvaluationPoint& point,
                              MultiFab& U, MultiFab& R, bool flux_only,
                              const PreparedGridBoundarySession& boundary) {
    if (b >= blocks_.size() || k < 0 || k >= nlev_ || point.level != k)
      throw std::out_of_range("AmrRuntime prepared core RHS level/block is out of range");
    if (boundary.resolved_plan() != blocks_[b].boundary_plan.get())
      throw std::invalid_argument(
          "AmrRuntime prepared core RHS boundary authority differs from block");
    if (interface_scheduler_.participates(b, k))
      throw std::runtime_error(
          "AmrRuntime implicit core RHS requires a coupled shared-interface solve");
    auto& closure = flux_only ? blocks_[b].level_neg_div_flux_core_at_point_prepared
                              : blocks_[b].level_rhs_core_at_point_prepared;
    if (!closure)
      throw std::runtime_error("AmrRuntime block lacks a prepared core residual closure");
    // The independent prepared route is single-block (shared-interface participants were refused
    // above). Its registry binds U directly for this owner and accepted live storage for other
    // dependencies, so no mutable hierarchy-wide staging slot is shared across execution lanes.
    closure(point, U, aux_[k], level_geom(k), R, boundary);
  }

  void level_boundary_residual_into_at(std::size_t b, int k,
                                       const runtime::multiblock::BoundaryEvaluationPoint& point,
                                       MultiFab& U, MultiFab& C) {
    if (b >= blocks_.size() || k < 0 || k >= nlev_ || point.level != k)
      throw std::out_of_range("AmrRuntime boundary residual level/block is out of range");
    if (!has_boundary_linearization(b))
      throw std::runtime_error("AmrRuntime block has no executable boundary residual/JVP pair");
    if (static_cast<std::size_t>(k) < blocks_[b].boundary_sessions.size() &&
        blocks_[b].boundary_sessions[static_cast<std::size_t>(k)]) {
      auto& prepared = blocks_[b].level_boundary_residual_at_point_prepared;
      if (!prepared)
        throw std::runtime_error("AmrRuntime block lacks its prepared boundary residual closure");
      prepared(point, U, aux_[k], level_geom(k), C,
               *blocks_[b].boundary_sessions[static_cast<std::size_t>(k)]);
      return;
    }
    auto& closure = blocks_[b].level_boundary_residual_at_point;
    if (!closure)
      throw std::runtime_error("AmrRuntime block lacks its boundary residual closure");
    closure(point, U, aux_[k], level_geom(k), C);
  }

  void level_boundary_residual_into_at(std::size_t b, int k,
                                       const runtime::multiblock::BoundaryEvaluationPoint& point,
                                       MultiFab& U, MultiFab& C,
                                       const PreparedGridBoundarySession& boundary) {
    if (b >= blocks_.size() || k < 0 || k >= nlev_ || point.level != k)
      throw std::out_of_range("AmrRuntime prepared boundary residual owner is out of range");
    if (!has_boundary_linearization(b))
      throw std::runtime_error("AmrRuntime block has no executable boundary residual/JVP pair");
    if (boundary.resolved_plan() != blocks_[b].boundary_plan.get())
      throw std::invalid_argument(
          "AmrRuntime prepared boundary residual authority differs from block");
    auto& closure = blocks_[b].level_boundary_residual_at_point_prepared;
    if (!closure)
      throw std::runtime_error("AmrRuntime block lacks a prepared boundary residual closure");
    closure(point, U, aux_[k], level_geom(k), C, boundary);
  }

  void level_boundary_jvp_into_at(std::size_t b, int k,
                                  const runtime::multiblock::BoundaryEvaluationPoint& point,
                                  MultiFab& U, const MultiFab& V, MultiFab& J) {
    if (b >= blocks_.size() || k < 0 || k >= nlev_ || point.level != k)
      throw std::out_of_range("AmrRuntime boundary JVP level/block is out of range");
    if (!has_boundary_linearization(b))
      throw std::runtime_error("AmrRuntime block has no executable boundary residual/JVP pair");
    if (static_cast<std::size_t>(k) < blocks_[b].boundary_sessions.size() &&
        blocks_[b].boundary_sessions[static_cast<std::size_t>(k)]) {
      auto& prepared = blocks_[b].level_boundary_jvp_at_point_prepared;
      if (!prepared)
        throw std::runtime_error("AmrRuntime block lacks its prepared boundary JVP closure");
      prepared(point, U, V, aux_[k], level_geom(k), J,
               *blocks_[b].boundary_sessions[static_cast<std::size_t>(k)]);
      return;
    }
    auto& closure = blocks_[b].level_boundary_jvp_at_point;
    if (!closure)
      throw std::runtime_error("AmrRuntime block lacks its boundary JVP closure");
    closure(point, U, V, aux_[k], level_geom(k), J);
  }

  void level_boundary_jvp_into_at(std::size_t b, int k,
                                  const runtime::multiblock::BoundaryEvaluationPoint& point,
                                  MultiFab& U, const MultiFab& V, MultiFab& J,
                                  const PreparedGridBoundarySession& boundary) {
    if (b >= blocks_.size() || k < 0 || k >= nlev_ || point.level != k)
      throw std::out_of_range("AmrRuntime prepared boundary JVP owner is out of range");
    if (!has_boundary_linearization(b))
      throw std::runtime_error("AmrRuntime block has no executable boundary residual/JVP pair");
    if (boundary.resolved_plan() != blocks_[b].boundary_plan.get())
      throw std::invalid_argument("AmrRuntime prepared boundary JVP authority differs from block");
    auto& closure = blocks_[b].level_boundary_jvp_at_point_prepared;
    if (!closure)
      throw std::runtime_error("AmrRuntime block lacks a prepared boundary JVP closure");
    closure(point, U, V, aux_[k], level_geom(k), J, boundary);
  }

  /// Install one prepared interface route on an AMR level.  The current AMR engine owns one shared
  /// layout per level, but the same scheduler contract used by Uniform still proves orientation,
  /// permutation and equal face discretisation before the route becomes executable.
  void install_level_interface_flux(
      int k, runtime::multiblock::AxisAlignedInterface route,
      const PopsExecutionContextV1& execution,
      runtime::multiblock::InterfaceFluxEvaluatorFactory evaluator_factory) {
    if (k < 0 || k >= nlev_ || route.level != k || route.left_block >= blocks_.size() ||
        route.right_block >= blocks_.size())
      throw std::out_of_range("AmrRuntime interface level/block index is out of range");
    if (regrid_every_ != 0)
      throw std::invalid_argument(
          "AmrRuntime interface v1 requires a frozen hierarchy (regrid_every=0)");
    const std::size_t left = route.left_block;
    const std::size_t right = route.right_block;
    if (!blocks_[left].level_rhs_without_prepared_interfaces ||
        !blocks_[right].level_rhs_without_prepared_interfaces ||
        !blocks_[left].level_neg_div_flux_without_prepared_interfaces ||
        !blocks_[right].level_neg_div_flux_without_prepared_interfaces)
      throw std::invalid_argument(
          "AmrRuntime interface blocks lack full/flux-only interface-omitting residuals");
    const Geometry geometry = level_geom(k);
    interface_scheduler_.install(std::move(route),
                                 (*blocks_[left].levels)[static_cast<std::size_t>(k)].U, geometry,
                                 (*blocks_[right].levels)[static_cast<std::size_t>(k)].U, geometry,
                                 execution, std::move(evaluator_factory));
  }

  /// Discard every shared-interface route after a failed pre-bind transaction.
  void discard_interface_fluxes() { interface_scheduler_.clear(); }

  /// Bind the detached qualified Handle routes to the exact per-level native storages.  The tables
  /// are authenticated by Python before hierarchy construction; no block/field name is parsed here.
  void install_boundary_storage_routes(const std::map<std::string, std::string>& field_routes) {
    std::map<std::string, std::size_t> state_routes;
    for (std::size_t block = 0; block < blocks_.size(); ++block) {
      const auto& identity = blocks_[block].state_identity;
      if (identity.empty() || !state_routes.emplace(identity, block).second)
        throw std::invalid_argument(
            "AmrRuntime materialized block state route is empty or duplicated");
    }
    auto states = std::make_shared<const std::map<std::string, std::size_t>>(state_routes);
    auto fields = std::make_shared<const std::map<std::string, std::string>>(field_routes);
    for (std::size_t current = 0; current < blocks_.size(); ++current) {
      auto& block = blocks_[current];
      if (!block.boundary_plan)
        continue;
      const auto primary = state_routes.find(block.boundary_plan->state_identity());
      if (!block.boundary_field_registry || block.boundary_plan->state_identity().empty() ||
          primary == state_routes.end() || primary->second != current)
        throw std::invalid_argument(
            "AmrRuntime block boundary plan differs from its qualified state route");
      const auto plan = block.boundary_plan;
      struct BoundaryRegistryRoutes {
        std::mutex mutex;
        std::atomic<std::uint64_t> materialization_generation{
            std::numeric_limits<std::uint64_t>::max()};
        std::vector<std::size_t> state_owners;
        std::vector<std::vector<MultiFab*>> fields_by_level;
      };
      auto routes = std::make_shared<BoundaryRegistryRoutes>();
      const auto required_states = plan->required_state_identities();
      const auto required_fields = plan->required_field_identities();
      const auto required_directions = plan->required_direction_identities();
      const auto residual_outputs = plan->residual_output_identities();
      const auto jvp_outputs = plan->jvp_output_identities();
      const auto all_outputs = plan->all_output_identities();
      const auto output_slot = [&all_outputs](const std::vector<std::string>& identities) {
        if (identities.empty())
          return std::size_t{0};
        const auto found = std::find(all_outputs.begin(), all_outputs.end(), identities.front());
        if (found == all_outputs.end())
          throw std::logic_error("AmrRuntime boundary output route was not prepared");
        return static_cast<std::size_t>(std::distance(all_outputs.begin(), found));
      };
      const std::size_t residual_output_slot = output_slot(residual_outputs);
      const std::size_t jvp_output_slot = output_slot(jvp_outputs);
      *block.boundary_field_registry =
          [this, current, plan, states, fields, routes, required_states, required_fields,
           required_directions, residual_outputs, jvp_outputs, residual_output_slot,
           jvp_output_slot](const runtime::multiblock::BoundaryEvaluationPoint& point,
                            MultiFab& state, const MultiFab* direction, MultiFab* output,
                            detail::BoundaryFieldRegistry& registry) {
            if (point.level < 0 || point.level >= nlev_)
              throw std::out_of_range("AmrRuntime boundary storage registry level is out of range");
            if (boundary_stage_states_ && boundary_stage_states_->point != point)
              throw std::runtime_error(
                  "AmrRuntime boundary stage-state registry was used at a different point");
            if (routes->materialization_generation.load(std::memory_order_acquire) !=
                topology_materialization_generation_) {
              std::scoped_lock lock(routes->mutex);
              if (routes->materialization_generation.load(std::memory_order_relaxed) !=
                  topology_materialization_generation_) {
                routes->state_owners.clear();
                routes->fields_by_level.clear();
                routes->state_owners.reserve(required_states.size());
                for (const auto& identity : required_states) {
                  const auto owner = states->find(identity);
                  if (owner == states->end())
                    throw std::runtime_error(
                        "AmrRuntime boundary state dependency has no exact provider route");
                  routes->state_owners.push_back(owner->second);
                }
                routes->fields_by_level.resize(static_cast<std::size_t>(nlev_));
                for (int level = 0; level < nlev_; ++level) {
                  auto& level_fields = routes->fields_by_level[static_cast<std::size_t>(level)];
                  level_fields.reserve(required_fields.size());
                  for (const auto& identity : required_fields) {
                    const auto route = fields->find(identity);
                    if (route == fields->end())
                      throw std::runtime_error(
                          "AmrRuntime boundary field dependency has no exact provider route");
                    level_fields.push_back(&provider_potential_level(route->second, level));
                  }
                }
                routes->materialization_generation.store(topology_materialization_generation_,
                                                         std::memory_order_release);
              }
            }
            for (std::size_t slot = 0; slot < routes->state_owners.size(); ++slot) {
              const std::size_t owner = routes->state_owners[slot];
              MultiFab* storage = nullptr;
              if (boundary_stage_states_)
                storage = boundary_stage_states_->state(owner);
              if (storage == nullptr)
                storage = owner == current
                              ? &state
                              : &(*blocks_[owner].levels)[static_cast<std::size_t>(point.level)].U;
              registry.bind_state_slot(slot, *storage);
            }
            const auto& level_fields =
                routes->fields_by_level[static_cast<std::size_t>(point.level)];
            for (std::size_t slot = 0; slot < level_fields.size(); ++slot)
              registry.bind_field_slot(slot, *level_fields[slot]);
            if (direction != nullptr) {
              for (std::size_t slot = 0; slot < required_directions.size(); ++slot) {
                const auto& identity = required_directions[slot];
                if (identity != plan->state_identity())
                  throw std::runtime_error(
                      "AmrRuntime boundary JVP direction has no exact block storage route");
                registry.bind_direction_slot(slot, *direction);
              }
            }
            if (output != nullptr) {
              const auto& identities = direction == nullptr ? residual_outputs : jvp_outputs;
              if (identities.size() > 1)
                throw std::runtime_error(
                    "AmrRuntime boundary operation requires multiple mutable output storages");
              if (!identities.empty())
                registry.bind_output_slot(
                    direction == nullptr ? residual_output_slot : jvp_output_slot, *output);
            }
          };
    }
    materialize_boundary_sessions_();
  }

  void materialize_boundary_sessions_() {
    // Every registry callback is installed before this helper first runs, so cross-block and
    // named-field routes are complete. Regrid/restart call it again after replacing level layouts.
    // Prepare every replacement off to the side and publish only after all blocks/levels succeed;
    // a failed lane or native executor therefore leaves the last accepted sessions intact.
    std::vector<std::vector<std::shared_ptr<ExecutionLane>>> prepared_lanes(blocks_.size());
    std::vector<std::vector<std::shared_ptr<PreparedGridBoundarySession>>> prepared_sessions(
        blocks_.size());
    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
      auto& block = blocks_[block_index];
      if (!block.boundary_plan)
        continue;
      auto& lanes = prepared_lanes[block_index];
      auto& sessions = prepared_sessions[block_index];
      lanes.reserve(static_cast<std::size_t>(nlev_));
      sessions.reserve(static_cast<std::size_t>(nlev_));
      for (int level = 0; level < nlev_; ++level) {
        runtime::multiblock::BoundaryEvaluationPoint preparation_point;
        preparation_point.clock = block.boundary_plan->identity() + "::bound-runtime";
        preparation_point.level = level;
        preparation_point.dt = 0.0;
        preparation_point.physical_time = 0.0;
        std::array<char, std::numeric_limits<int>::digits10 + 3> level_identity{};
        const auto [level_identity_end, level_identity_error] = std::to_chars(
            level_identity.data(), level_identity.data() + level_identity.size(), level);
        if (level_identity_error != std::errc{})
          throw std::logic_error("AMR boundary lane level identity exceeded fixed integer storage");
        const std::array<std::string_view, 3> lane_identity_parts{
            block.boundary_plan->identity(), "::runtime-level-",
            std::string_view(level_identity.data(),
                             static_cast<std::size_t>(level_identity_end - level_identity.data()))};
        auto lane = std::make_shared<ExecutionLane>(
            ExecutionLane::world(std::span<const std::string_view>(lane_identity_parts)));
        MultiFab& prototype = (*block.levels)[static_cast<std::size_t>(level)].U;
        auto session = std::make_shared<PreparedGridBoundarySession>(
            level_grid_context(block_index, level), *lane, prototype, preparation_point);
        lanes.push_back(std::move(lane));
        sessions.push_back(std::move(session));
      }
    }
    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
      blocks_[block_index].boundary_lanes.swap(prepared_lanes[block_index]);
      blocks_[block_index].boundary_sessions.swap(prepared_sessions[block_index]);
    }
  }

  void install_level_interface_flux(int k, runtime::multiblock::AxisAlignedInterface route,
                                    const PopsExecutionContextV1& execution,
                                    runtime::multiblock::InterfaceFluxEvaluator evaluator) {
    install_level_interface_flux(
        k, std::move(route), execution,
        runtime::multiblock::InterfaceFluxEvaluatorFactory(
            [evaluator = std::move(evaluator)]() mutable { return std::move(evaluator); }));
  }

  /// Real AMR multi-block residual executor.  All per-block residuals on the level are complete
  /// before the shared pair flux is evaluated once and scattered, so neither side can consume an
  /// interface-incomplete residual.
  void level_rhs_with_interfaces(int k, const runtime::multiblock::BoundaryEvaluationPoint& point,
                                 const std::vector<MultiFab*>& states,
                                 const std::vector<MultiFab*>& rhs,
                                 const std::vector<int>& flux_only = {}) {
    if (k < 0 || k >= nlev_ || point.level != k || states.size() != blocks_.size() ||
        rhs.size() != blocks_.size() || (!flux_only.empty() && flux_only.size() != blocks_.size()))
      throw std::invalid_argument("AmrRuntime multi-block interface RHS axis mismatch");
    struct StageStateReset {
      std::optional<BoundaryStageStateView>* slot = nullptr;
      ~StageStateReset() {
        if (slot != nullptr)
          slot->reset();
      }
    } stage_reset;
    const bool routed =
        std::any_of(blocks_.begin(), blocks_.end(),
                    [](const AmrRuntimeBlock& block) { return !block.state_identity.empty(); });
    if (routed) {
      if (boundary_stage_states_)
        throw std::runtime_error("AmrRuntime boundary stage-state registry is already active");
      for (std::size_t block = 0; block < blocks_.size(); ++block) {
        const auto& identity = blocks_[block].state_identity;
        if (identity.empty())
          throw std::runtime_error(
              "AmrRuntime materialized block has no exact qualified state identity");
      }
      boundary_stage_states_.emplace(BoundaryStageStateView{point, &states, -1, nullptr});
      stage_reset.slot = &boundary_stage_states_;
    }
    for (std::size_t block = 0; block < blocks_.size(); ++block) {
      if ((states[block] == nullptr) != (rhs[block] == nullptr))
        throw std::invalid_argument("AmrRuntime sparse RHS group has one null storage pointer");
      if (states[block] == nullptr)
        continue;
      const bool flux = !flux_only.empty() && flux_only[block] != 0;
      if (static_cast<std::size_t>(k) < blocks_[block].boundary_sessions.size() &&
          blocks_[block].boundary_sessions[static_cast<std::size_t>(k)]) {
        auto& core = flux ? blocks_[block].level_neg_div_flux_core_at_point_prepared
                          : blocks_[block].level_rhs_core_at_point_prepared;
        auto& boundary = blocks_[block].level_boundary_residual_at_point_prepared;
        if (!core || !boundary)
          throw std::runtime_error(
              "AmrRuntime block lacks its persistent prepared boundary closures");
        const auto& session = *blocks_[block].boundary_sessions[static_cast<std::size_t>(k)];
        core(point, *states[block], aux_[k], level_geom(k), *rhs[block], session);
        boundary(point, *states[block], aux_[k], level_geom(k), *rhs[block], session);
        continue;
      }
      if (interface_scheduler_.participates(block, k)) {
        fill_level_state_cf_ghosts(block, k, *states[block]);
        auto& closure = flux ? blocks_[block].level_neg_div_flux_without_prepared_interfaces
                             : blocks_[block].level_rhs_without_prepared_interfaces;
        if (!closure)
          throw std::runtime_error("AmrRuntime lost an interface-omitting residual closure");
        closure(point, *states[block], aux_[k], level_geom(k), *rhs[block]);
      } else {
        if (flux) {
          if (!blocks_[block].level_neg_div_flux_at_point)
            throw std::runtime_error("AmrRuntime block lacks a point-qualified flux residual");
          fill_level_state_cf_ghosts(block, k, *states[block]);
          blocks_[block].level_neg_div_flux_at_point(point, *states[block], aux_[k], level_geom(k),
                                                     *rhs[block]);
        } else {
          level_rhs_into_at(block, k, point, *states[block], *rhs[block]);
        }
      }
    }
    interface_scheduler_.apply(point, states, rhs);
  }

  /// Publish one atomic sparse view of provisional block states while @p body evaluates a coupled
  /// stage.  Boundary providers may read another block, so a grouped Program evaluation must expose
  /// every member's stage value at one common logical point rather than publishing each block only
  /// while its own residual is captured.  The view is process-local, non-owning and scoped: nested
  /// publication is rejected and RAII removes it on both normal and exceptional exits.
  template <class Body>
  decltype(auto) with_boundary_stage_states(
      const runtime::multiblock::BoundaryEvaluationPoint& point,
      const std::vector<int>& requested_blocks, const std::vector<MultiFab*>& requested_states,
      Body&& body) {
    if (point.level < 0 || point.level >= nlev_ || requested_blocks.empty() ||
        requested_blocks.size() != requested_states.size())
      throw std::invalid_argument(
          "AmrRuntime::with_boundary_stage_states has an invalid level or sparse axis");
    if (boundary_stage_states_)
      throw std::runtime_error("AmrRuntime boundary stage-state registry is already active");

    std::vector<MultiFab*> staged(blocks_.size(), nullptr);
    for (std::size_t slot = 0; slot < requested_blocks.size(); ++slot) {
      const int block = requested_blocks[slot];
      if (block < 0 || static_cast<std::size_t>(block) >= blocks_.size() ||
          requested_states[slot] == nullptr)
        throw std::invalid_argument(
            "AmrRuntime::with_boundary_stage_states has an invalid block or null state");
      if (staged[static_cast<std::size_t>(block)] != nullptr)
        throw std::invalid_argument(
            "AmrRuntime::with_boundary_stage_states contains a duplicate block");
      staged[static_cast<std::size_t>(block)] = requested_states[slot];
    }

    boundary_stage_states_.emplace(BoundaryStageStateView{point, &staged, -1, nullptr});
    struct StageStateReset {
      std::optional<BoundaryStageStateView>* slot;
      ~StageStateReset() { slot->reset(); }
    } reset{&boundary_stage_states_};
    return std::invoke(std::forward<Body>(body));
  }

  void level_rhs_group(int k, const runtime::multiblock::BoundaryEvaluationPoint& point,
                       const std::vector<int>& requested_blocks,
                       const std::vector<MultiFab*>& requested_states,
                       const std::vector<MultiFab*>& requested_rhs,
                       const std::vector<int>& requested_flux_only) {
    if (requested_blocks.empty() || requested_blocks.size() != requested_states.size() ||
        requested_blocks.size() != requested_rhs.size() ||
        requested_blocks.size() != requested_flux_only.size())
      throw std::invalid_argument("AmrRuntime::level_rhs_group has inconsistent vectors");
    std::vector<MultiFab*> states(blocks_.size(), nullptr);
    std::vector<MultiFab*> rhs(blocks_.size(), nullptr);
    std::vector<int> flux_only(blocks_.size(), 0);
    for (std::size_t request = 0; request < requested_blocks.size(); ++request) {
      const int raw = requested_blocks[request];
      if (raw < 0 || raw >= static_cast<int>(blocks_.size()))
        throw std::out_of_range("AmrRuntime::level_rhs_group block index is out of range");
      const std::size_t block = static_cast<std::size_t>(raw);
      if (states[block] != nullptr || requested_states[request] == nullptr ||
          requested_rhs[request] == nullptr ||
          (requested_flux_only[request] != 0 && requested_flux_only[request] != 1))
        throw std::invalid_argument(
            "AmrRuntime::level_rhs_group requires unique blocks and boolean modes");
      states[block] = requested_states[request];
      rhs[block] = requested_rhs[request];
      flux_only[block] = requested_flux_only[request];
    }
    level_rhs_with_interfaces(k, point, states, rhs, flux_only);
  }

  std::size_t interface_evaluation_count(const std::string& identity, int level) const {
    return interface_scheduler_.evaluation_count(identity, level);
  }
  bool has_level_interfaces(int level) const { return interface_scheduler_.has_interfaces(level); }
  /// R <- -div F(U) only (NO default source) for block @p b on level @p k (SourceFreeModel path). Same
  /// fine-level C/F ghost refresh as level_rhs_into (ADC-634).
  void level_neg_div_flux_into(std::size_t b, int k, MultiFab& U, MultiFab& R) {
    if (!blocks_[b].level_neg_div_flux)
      throw std::runtime_error("AmrRuntime::level_neg_div_flux_into: block '" + blocks_[b].name +
                               "' has no flux-only per-level residual closure");
    fill_level_state_cf_ghosts(b, k, U);
    blocks_[b].level_neg_div_flux(U, aux_[k], level_geom(k), R);
  }
  void level_neg_div_flux_into_at(std::size_t b, int k,
                                  const runtime::multiblock::BoundaryEvaluationPoint& point,
                                  MultiFab& U, MultiFab& R) {
    if (!blocks_[b].level_neg_div_flux_at_point)
      throw std::runtime_error("AmrRuntime block has no point-qualified flux residual closure");
    fill_level_state_cf_ghosts(b, k, U);
    blocks_[b].level_neg_div_flux_at_point(point, U, aux_[k], level_geom(k), R);
  }
  /// R <- S(U, aux_[k]) only (NO flux) for block @p b on level @p k (the source half of level_rhs).
  void level_source_into(std::size_t b, int k, MultiFab& U, MultiFab& R) {
    if (!blocks_[b].level_source)
      throw std::runtime_error("AmrRuntime::level_source_into: block '" + blocks_[b].name +
                               "' has no source-only per-level residual closure");
    blocks_[b].level_source(U, aux_[k], level_geom(k), R);
  }
  /// CONSERVATIVE-REFLUX capture (ADC-639): R <- -div F + S for block @p b on level @p k, materialising
  /// the face fluxes into @p Fx / @p Fy (sized here from the level box_array's xface_box/yface_box) so a
  /// reflux register can sample them. R is bit-consistent with the fused level_rhs_into by construction.
  /// The fine-level C/F ghost refresh matches level_rhs_into. Bodies in amr_program_reflux.hpp (the tail
  /// header keeps amr_runtime.hpp at its line budget). Used ONLY by the reflux path (AmrProgramContext,
  /// nlev>1); the native step and the coarse-only / flat Program never call it.
  void level_rhs_capture_into(std::size_t b, int k,
                              const runtime::multiblock::BoundaryEvaluationPoint& point,
                              MultiFab& U, MultiFab& R, MultiFab& Fx, MultiFab& Fy);
  void level_rhs_capture_into_temporal(std::size_t b, int k,
                                       const runtime::multiblock::BoundaryEvaluationPoint& point,
                                       MultiFab& U, MultiFab& R, MultiFab& Fx, MultiFab& Fy,
                                       const MultiFab& parent_old, const MultiFab& parent_new,
                                       const runtime::amr::TemporalTransferContext& target_time);
  /// FLUX-ONLY (SourceFreeModel) counterpart of level_rhs_capture_into (the neg_div_flux capture path).
  void level_neg_div_flux_capture_into(std::size_t b, int k,
                                       const runtime::multiblock::BoundaryEvaluationPoint& point,
                                       MultiFab& U, MultiFab& R, MultiFab& Fx, MultiFab& Fy);
  void level_neg_div_flux_capture_into_temporal(
      std::size_t b, int k, const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& U,
      MultiFab& R, MultiFab& Fx, MultiFab& Fy, const MultiFab& parent_old,
      const MultiFab& parent_new, const runtime::amr::TemporalTransferContext& target_time);
  /// Max |wave speed| of block @p b on @p U (the SAME closure step_cfl reads). Evaluated on the aux of
  /// level @p k. A Program dt bound reads it as cfl*hmin/max_wave_speed.
  Real level_max_speed(std::size_t b, int k, const MultiFab& U) const {
    return blocks_[b].max_speed(U, aux_[k]);
  }
  /// MIN physical cell size of level @p k (min(dx, dy) >> k): the per-level hmin a Program dt bound reads.
  Real level_hmin(int k) const {
    const Real r = static_cast<Real>(level_refinement(k));
    return std::min(geom_.dx(), geom_.dy()) / r;
  }
  /// fine -> coarse restriction of block @p b between levels @p k and @p k-1 (covered coarse cell <-
  /// 2x2 fine average): the SAME mf_average_down_mb solve_fields / the native step run. Exposed for the
  /// synchronous Program driver's inter-level coupling. No-op when k < 1.
  void average_down_level(std::size_t b, int k) {
    if (k < 1)
      return;
    if (b >= blocks_.size() || k >= nlev_ || !blocks_[b].average_down_plan)
      throw std::out_of_range("AmrRuntime::average_down_level owner is out of range");
    auto& L = *blocks_[b].levels;
    auto& plan = *blocks_[b].average_down_plan;
    mf_average_down_mb(L[k].U, L[k - 1].U, plan.transition_for_child(k),
                       plan.topology_generation(), world_communicator_view());
  }

  /// LEVEL-COMPOSITE collective reduction over a NAMED block (ADC-542) -- the AMR counterpart of
  /// System::reduce_component. @p levels is the exact strictly-increasing level selection; empty is
  /// the low-level C++ all-level convention. @p kind is per-component "sum" / "min" / "max" /
  /// "abs_sum" / "sum_sq" (dot) / "abs_max" (norm_inf), or the full-state "*_all" variants.
  /// Every reduction masks a coarser selected level by the next selected finer footprint;
  /// volume-weighted sums are dx*dy-weighted. COLLECTIVE; unknown block / kind throws. Body in
  /// amr_restore.hpp.
  double composite_reduce(const std::string& block, const std::string& kind, int comp,
                          const std::vector<int>& levels = {}) const;

  /// Native level-composite reduction over one qualified elliptic field provider.  The provider is
  /// materialized on the shared hierarchy, then folded directly on its Kokkos MultiFabs with the
  /// same coverage/metric/MPI contract as composite_reduce; no output snapshot arrays participate.
  double composite_reduce_field(const std::string& provider_slot, const std::string& kind, int comp,
                                const std::vector<int>& levels = {});

  /// Impose a mid-run hierarchy from a checkpoint (multi-block, all levels, reusing regrid R6/R7):
  /// build each level's BoxArray + DistributionMapping from the manifest, reallocate every block's
  /// level MultiFab on it (inherited ghost width), rebuild + rewire the shared aux, verify the
  /// shared-layout invariant. The layout AND the data come from the checkpoint (the per-level state
  /// restore overwrites every valid cell), so no tagging / clustering / prolong -- regrid MINUS the
  /// recompute. Body in amr_restore.hpp (ADC-542). @p level_boxes / @p level_owner_ranks are indexed by
  /// level; a level count over the composed max_levels is refused verbatim.
  void rebuild_hierarchy(const std::vector<std::vector<PatchBox>>& level_boxes,
                         const std::vector<std::vector<int>>& level_owner_ranks);
  /// Owner rank per box of level @p k (the shared layout's DistributionMapping), index-aligned with
  /// that level's boxes in patch_boxes(). The v3 checkpoint serializes it so a restart reproduces the
  /// LOCAL-fab iteration order (bit-identity of the host aggregations). Body in amr_restore.hpp.
  std::vector<int> level_owner_ranks(int k) const;
  /// FULL shared aux of level @p k (ALL aux_ncomp_ components, flat component-major over the exact
  /// rectangular level domain; _global = np>1
  /// all-reduce gather, COLLECTIVE) + owner-rank restore -- the v3 aux payload. Bodies in amr_restore.hpp.
  std::vector<double> level_aux_flat(int k) const;
  std::vector<double> level_aux_flat_global(int k) const;
  void set_level_aux_flat(int k, const std::vector<double>& v);
  /// Head-of-step union-tags regrid at the Program driver's cadence (the SAME regrid() the native step
  /// runs at its head). @p macro_step gates it like AmrRuntime::step (skip step 0; honor regrid_every_).
  void regrid_if_due(int macro_step) {
    if (regrid_every_ > 0 && macro_step > 0 && macro_step % regrid_every_ == 0)
      regrid();
  }
  /// @}

  /// Activates the UNION-TAGS REGRID at the cadence @p every (in macro-steps): every @p every
  /// macro-steps, BEFORE the macro-step's step(dt) (D2, consistent with the single-block
  /// amr_dsl_block.hpp:104), the shared hierarchy is re-gridded from the UNION of the tags of all
  /// blocks + phi. @p every == 0 (DEFAULT) -> FROZEN hierarchy, regrid never called -> BIT-IDENTICAL
  /// trajectory to the historical one (the feature is opt-in). @p grow: tag dilation (nesting +
  /// anticipation); @p margin: nesting (clamp the patches to the boundaries). Must be called BEFORE
  /// the first step. Body in amr_restore.hpp.
  void set_regrid(int every, int grow = 2, int margin = 2);

  /// ADC-616: Berger-Rigoutsos clustering params (min_efficiency in (0,1], sizes > 0, min <= max).
  /// Defaults reproduce {0.7, 1, 32}; refuses out-of-domain values STRUCTURALLY. Body in amr_restore.hpp.
  void set_clustering(double min_efficiency, int min_box_size, int max_box_size);

  /// Registers an externally supplied aux component as a coarse global row-major field. This is the
  /// single native storage/publication authority for canonical inputs such as B_z and model-named
  /// fields. It re-applies the field after elliptic postprocessing and hierarchy growth, then publishes
  /// it to every active level. An undeclared component or malformed coarse field is rejected here.
  template <class Allocator>
  void set_static_aux_component(int comp, std::vector<Real, Allocator> field) {
    if (comp < 0 || comp >= aux_ncomp_)
      throw std::runtime_error("AmrRuntime::set_static_aux_component: component " +
                               std::to_string(comp) + " is outside the declared aux width " +
                               std::to_string(aux_ncomp_));
    const std::size_t expected =
        static_cast<std::size_t>(dom_.nx()) * static_cast<std::size_t>(dom_.ny());
    if (field.size() != expected)
      throw std::runtime_error("AmrRuntime::set_static_aux_component: field size " +
                               std::to_string(field.size()) + " differs from coarse cell count " +
                               std::to_string(expected));
    StaticAuxField stored;
    if constexpr (std::is_same_v<Allocator, typename StaticAuxField::allocator_type>)
      stored = std::move(field);
    else
      stored.assign(field.begin(), field.end());
    static_aux_[comp] = std::move(stored);
    if (!aux_.empty()) {
      apply_static_aux();
      // A bound static aux is a hierarchy-wide field, not merely a coarse-level seed. Program
      // projections and rate kernels read aux_[level] directly, including before the first elliptic
      // solve, so publish the newly bound component through the same coarse-to-fine authority used
      // after solve_fields.  Restricting the publication to this component avoids disturbing any
      // independently solved aux fields.
      publish_aux_components({comp});
    }
  }

  /// Registers a model-named aux field. Canonical/reserved components use their typed facade paths
  /// and call set_static_aux_component only after their own contract has been validated.
  template <class Allocator>
  void set_named_aux(int comp, std::vector<Real, Allocator> field) {
    if (comp < kAuxNamedBase)
      throw std::runtime_error("AmrRuntime::set_named_aux: component " + std::to_string(comp) +
                               " is reserved; named aux starts at " +
                               std::to_string(kAuxNamedBase));
    set_static_aux_component(comp, std::move(field));
  }

  /// Registers a per-field aux HALO policy (ADC-369) for the named component @p comp: solve_fields
  /// applies it onto the COARSE aux AFTER the shared fill_ghosts, overriding only that component's
  /// physical-face ghosts (periodic faces stay periodic). Coarse-level scope (fine patches touching the
  /// domain boundary inherit the shared BC). No-op default. AMR counterpart of
  /// System::set_aux_field_halo_component.
  void set_named_aux_bc(int comp, AuxHaloPolicy policy) {
    named_aux_bc_[comp] = policy;
    if (!aux_.empty() && static_aux_.count(comp) != 0) {
      // A halo policy installed after the value is still part of the same bound field contract.
      // Re-publish that component so its hierarchy ghosts cannot retain the previous policy.
      apply_static_aux();
      publish_aux_components({comp});
    }
  }

  /// @name Named multi-elliptic fields (ADC-428)
  /// A SECOND elliptic solve (beyond the default coarse Poisson) for a user-named field
  /// (m.elliptic_field("psi", rhs=..., aux=[...])) on the AMR hierarchy. AMR counterpart of
  /// SystemFieldSolver::register_named_field / solve_named_field_from_state. Each named field owns a
  /// dedicated prepared provider instance built lazily from the exact hierarchy contract, its RHS =
  /// sum over blocks of @c named_elliptic_rhs[field], and its own aux
  /// output components (the model's named aux slots, >= kAuxNamedBase). solve_fields() solves every
  /// registered named field right after the default Poisson and injects its aux to the fine levels, so a
  /// bare run() leaves the field SOLVED (readable via named_field_values). The default Poisson path
  /// is untouched / bit-identical. Empty default -> the named-field loop is a no-op.
  /// @{
  /// Registers named @c field's aux output components: @p phi_comp where the solved potential lands, @p
  /// gx_comp / @p gy_comp where its centered gradient lands. Both equal -1 for phi-only; otherwise
  /// @p gradient_sign (exactly -1 or +1) scales both centered derivatives. Idempotent (re-register overwrites the
  /// components and drops the lazily-built solver so the next solve rebuilds it). The dedicated solver
  /// is built on the first solve, never here.
  void install_field_plan(const std::string& provider_slot, const AmrFieldSolveConfig& plan) {
    if (provider_slot.empty() || plan.plan_identity.empty() || plan.provider_identity.empty() ||
        plan.output_owner_identity.empty() || plan.output_block.empty() ||
        plan.output_key.empty() || plan.providers.empty() || plan.topology_provider_kind.empty() ||
        plan.topology_provenance.empty() || plan.topology_digest.empty())
      throw std::runtime_error("AmrRuntime: incomplete qualified field provider plan");
    for (const auto& provider : plan.providers)
      if (provider.identity.empty() || provider.owner_block.empty() ||
          provider.native_key.empty() || !std::isfinite(static_cast<double>(provider.coefficient)))
        throw std::runtime_error("AmrRuntime: invalid field provider-pack entry");
    try {
      (void)plan.nullspace.exact_contract();
      (void)plan.hierarchy_policy.exact_contract();
    } catch (...) {
      throw std::runtime_error(
          "AmrRuntime: incomplete field-nullspace or hierarchy-policy authority");
    }
    const BCRec field_boundary = plan.has_explicit_bc ? plan.explicit_bc : bcPhi_;
    const AmrFieldSolverBuildRequest request{
        geom_,        hierarchy_,         field_boundary,
        wall_active_, replicated_coarse_, "pops.amr.field-solver-use.named@1",
        plan};
    const auto provider = field_solver_registry_->resolve(plan.solver);
    (void)inspect_amr_field_solver_support_collectively(*provider, request);
    auto existing = named_fields_.find(provider_slot);
    if (existing != named_fields_.end())
      throw std::runtime_error("AmrRuntime: duplicate qualified field provider slot");
    NamedField field;
    field.plan = plan;
    field.has_plan = true;
    named_fields_.emplace(provider_slot, std::move(field));
  }

  void set_field_boundary_kernel(const std::string& provider_slot,
                                 const CompiledFieldBoundaryKernel& kernel) {
    kernel.validate();
    auto found = named_fields_.find(provider_slot);
    if (found == named_fields_.end())
      throw std::runtime_error("AmrRuntime: unknown field provider boundary-kernel slot");
    found->second.plan.boundary_kernel = kernel;
    found->second.plan.has_boundary_kernel = true;
    if (!kernel.observes_iteration)
      found->second.plan.boundary_context.point.iteration = 0;
    invalidate_named_field_solver(found->second);
  }

  void set_field_logical_timepoint(const std::string& provider_slot,
                                   const FieldLogicalTimePoint& point) {
    auto found = named_fields_.find(provider_slot);
    if (found == named_fields_.end())
      throw std::runtime_error("AmrRuntime: unknown field provider logical-time slot");
    auto& field = found->second;
    field.plan.boundary_context.point = point;
    if (!field.plan.has_boundary_kernel || !field.plan.boundary_kernel.observes_iteration)
      field.plan.boundary_context.point.iteration = 0;
    if (field.plan.has_boundary_kernel) {
      if (field.solver)
        field.solver->set_boundary_context(field.plan.boundary_context);
    }
  }

  void set_field_boundary_parameters(const std::string& provider_slot,
                                     const std::vector<Real>& parameters) {
    auto found = named_fields_.find(provider_slot);
    if (found == named_fields_.end())
      throw std::runtime_error("AmrRuntime: unknown field provider boundary-parameter slot");
    auto& plan = found->second.plan;
    if (!plan.boundary_parameters)
      plan.boundary_parameters = std::make_shared<std::vector<Real>>();
    *plan.boundary_parameters = parameters;
    plan.boundary_context.parameters = plan.boundary_parameters.get();
    plan.boundary_context.parameter_count = static_cast<int>(parameters.size());
    if (plan.has_boundary_kernel) {
      if (found->second.solver)
        found->second.solver->set_boundary_context(plan.boundary_context);
    }
  }

  void set_field_boundary_dependencies(const std::string& provider_slot,
                                       const std::vector<std::string>& state_blocks,
                                       const std::vector<int>& state_components) {
    auto found = named_fields_.find(provider_slot);
    if (found == named_fields_.end())
      throw std::runtime_error("AmrRuntime: unknown field boundary-dependency slot");
    found->second.plan.boundary_state_blocks = state_blocks;
    found->second.plan.boundary_state_components = state_components;
    invalidate_named_field_solver(found->second);
  }

  void set_field_newton_plan(const std::string& provider_slot, const FieldNewtonOptions& options) {
    validate_field_newton_options(options);
    auto found = named_fields_.find(provider_slot);
    if (found == named_fields_.end())
      throw std::runtime_error("AmrRuntime: unknown field provider nonlinear-plan slot");
    found->second.plan.newton = options;
    found->second.plan.has_newton = true;
    invalidate_named_field_solver(found->second);
  }

  void register_named_field(const std::string& block, const std::string& provider_key, int phi_comp,
                            int gx_comp, int gy_comp, int gradient_sign) {
    const bool has_gradient = gx_comp >= 0 && gy_comp >= 0;
    const bool has_no_gradient = gx_comp == -1 && gy_comp == -1;
    if (phi_comp < 0 || (!has_gradient && !has_no_gradient) ||
        (has_gradient && (phi_comp == gx_comp || phi_comp == gy_comp || gx_comp == gy_comp)))
      throw std::invalid_argument(
          "AmrRuntime: named elliptic field output components must be one potential or "
          "three distinct potential/gradient components");
    if (gradient_sign != -1 && gradient_sign != 1)
      throw std::invalid_argument(
          "AmrRuntime: named elliptic field gradient sign must be exactly -1 or 1");
    if (!has_gradient && gradient_sign != 1)
      throw std::invalid_argument(
          "AmrRuntime: a named elliptic field without gradient outputs must use sign +1");
    bool matched = false;
    for (auto& [slot, field] : named_fields_) {
      if (!field.has_plan || field.plan.output_block != block ||
          field.plan.output_key != provider_key)
        continue;
      field.phi_comp = phi_comp;
      field.gx_comp = gx_comp;
      field.gy_comp = gy_comp;
      field.gradient_sign = gradient_sign;
      invalidate_named_field_solver(field);
      matched = true;
    }
    if (!matched)
      return;  // loader may declare an unused provider; only resolved Problem plans are installed
  }
  /// Attaches named @p field's RHS contribution closure (rhs += elliptic_field_rhs(U_b)) to block @p b.
  /// Called per declared field once the runtime owns the blocks. @throws if @p b is out of bounds.
  void set_block_named_elliptic_rhs(std::size_t b, const std::string& provider_key,
                                    std::function<void(const MultiFab&, MultiFab&)> rhs) {
    if (b >= blocks_.size())
      throw std::runtime_error(
          "AmrRuntime::set_block_named_elliptic_rhs : block index out of bounds");
    blocks_[b].named_elliptic_rhs[provider_key] = std::move(rhs);
  }
  /// Number of registered named elliptic fields (diagnostic / test).
  std::size_t n_named_fields() const { return named_fields_.size(); }
  /// True if @p field is a registered named elliptic field.
  bool has_named_field(const std::string& field) const {
    return named_fields_.find(field) != named_fields_.end();
  }
  /// Solved potential of named @p field as a coarse ny*nx row-major field. Solves
  /// the fields if needed (counterpart of potential() for the default phi), then reads the field's
  /// phi_comp on the coarse aux. @throws if @p field is unregistered. AMR counterpart of
  /// System::aux_field_component for a named elliptic field.
  std::vector<std::string> provider_slots() const {
    std::vector<std::string> result;
    result.reserve(named_fields_.size());
    for (const auto& item : named_fields_)
      result.push_back(item.first);
    return result;
  }

  /// Report only a topology that a named field solver has actually materialized.  Every supported
  /// AMR field route is full-material on the exact shared hierarchy, exposed patch by patch over all
  /// levels.  Regrid invalidates the solver and therefore makes this report absent again until the
  /// new topology is materialized.
  std::optional<std::vector<PatchBox>> field_topology_patches(
      const std::string& provider_slot) const {
    const auto found = named_fields_.find(provider_slot);
    if (found == named_fields_.end())
      throw std::runtime_error("AmrRuntime: unknown qualified field provider slot '" +
                               provider_slot + "'");
    const NamedField& field = found->second;
    if (!field.solver)
      return std::nullopt;
    std::vector<PatchBox> result;
    for (int level = 0; level < hierarchy_.nlev(); ++level)
      for (const Box2D& box : hierarchy_.ba[static_cast<std::size_t>(level)].boxes())
        result.push_back(PatchBox{level, box.lo[0], box.lo[1], box.hi[0], box.hi[1]});
    return result;
  }

  MultiFab& provider_potential(const std::string& provider_slot) {
    auto it = named_fields_.find(provider_slot);
    if (it == named_fields_.end())
      throw std::runtime_error("AmrRuntime: unknown qualified field provider slot '" +
                               provider_slot + "'");
    ensure_named_elliptic(it->second);
    return it->second.solver->phi_level(0);
  }

  int provider_potential_levels(const std::string& provider_slot) {
    auto it = named_fields_.find(provider_slot);
    if (it == named_fields_.end())
      throw std::runtime_error("AmrRuntime: unknown qualified field provider slot");
    ensure_named_elliptic(it->second);
    return it->second.solver->level_count();
  }

  MultiFab& provider_potential_level(const std::string& provider_slot, int level) {
    auto it = named_fields_.find(provider_slot);
    if (it == named_fields_.end())
      throw std::runtime_error("AmrRuntime: unknown qualified field provider slot");
    ensure_named_elliptic(it->second);
    const int levels = it->second.solver->level_count();
    if (level < 0 || level >= levels)
      throw std::out_of_range("AmrRuntime: qualified field provider level out of range");
    return it->second.solver->phi_level(level);
  }

  std::vector<double> named_field_values(const std::string& provider_slot) {
    auto it = named_fields_.find(provider_slot);
    if (it == named_fields_.end())
      throw std::runtime_error(
          "AmrRuntime::named_field_values : unknown qualified field provider '" + provider_slot +
          "'");
    require_solved_field_report(solve_fields(), "AmrRuntime::named_field_values");
    return coarse_aux_component(it->second.phi_comp);
  }
  /// @}

  /// Registers an inter-species COUPLED SOURCE (DSL CoupledSource, P5 bytecode) on the runtime facade,
  /// counterpart of System::add_coupled_source. The ABI is FLAT (postfix bytecode): we resolve each
  /// (block, role) into (block index, component) then store a closure that, at each macro-step AFTER
  /// the transport, applies the source by additive forward-Euler splitting via coupled_source_step. The
  /// coupling is ENTIRELY baked into a stack machine (device-clean functor CoupledSourceKernel): NO
  /// per-cell Python callback in the hot path.
  ///
  /// CONSERVATION (conservative exchange): with an add_pair construction (one +expr term on one block,
  /// -expr exactly on the other, SAME cell), the two per-cell contributions are opposite up to sign, so
  /// n_a + n_b is conserved PER CELL (and globally) to machine precision, independent of dt and of the
  /// state. The engine does not enforce it (an ionization creating a pair is licit): conservation is a
  /// property of the constructed coupling, checked test-side.
  ///
  /// @param in_blocks/in_roles  READ fields (one register per (block, role)), in register order.
  /// @param consts              constants (parameters), loaded into the registers after the inputs.
  /// @param out_blocks/out_roles target (block, role) of each source term.
  /// @param prog_ops/prog_args  CONCATENATED postfix bytecode of all the terms (split by prog_lens).
  /// @param prog_lens           program length of each term (size == out_blocks).
  /// @throws std::runtime_error on an inconsistent form, an unknown role, an unknown block, an opcode
  ///         or register out of bounds, or a program too long (same guards as System).
  void add_coupled_source(const std::vector<std::string>& in_blocks,
                          const std::vector<std::string>& in_roles,
                          const std::vector<double>& consts,
                          const std::vector<std::string>& out_blocks,
                          const std::vector<std::string>& out_roles,
                          const std::vector<int>& prog_ops, const std::vector<int>& prog_args,
                          const std::vector<int>& prog_lens) {
    const int n_in = static_cast<int>(in_blocks.size());
    const int n_const = static_cast<int>(consts.size());
    const int n_terms = static_cast<int>(out_blocks.size());
    // --- form validation (before any step, EXPLICIT errors); mirror of System::add_coupled_source.
    if (n_terms == 0)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_source : no source term (out_blocks empty)");
    if (static_cast<int>(in_roles.size()) != n_in)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_source : in_blocks / in_roles of different sizes");
    if (static_cast<int>(out_roles.size()) != n_terms ||
        static_cast<int>(prog_lens.size()) != n_terms)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_source : out_blocks / out_roles / prog_lens of "
          "different sizes");
    if (prog_ops.size() != prog_args.size())
      throw std::runtime_error(
          "AmrRuntime::add_coupled_source : prog_ops / prog_args of different sizes");
    if (n_in + n_const > kCsMaxReg)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_source : too many registers (inputs + constants > " +
          std::to_string(kCsMaxReg) + ")");
    if (n_terms > kCsMaxTerms)
      throw std::runtime_error("AmrRuntime::add_coupled_source : too many source terms (> " +
                               std::to_string(kCsMaxTerms) + ")");
    // Resolves (block, role) -> (block index, component) by the block CONSERVATIVE descriptor, like
    // System (#181). An unknown block throws immediately; an unknown (non-canonical) role too.
    auto resolve = [&](const std::string& block, const std::string& role) -> std::pair<int, int> {
      const int b = block_index(block);
      if (b < 0)
        throw std::runtime_error("AmrRuntime::add_coupled_source : no block named '" + block + "'");
      // STRICT (no silent fallback; mirror of System::add_coupled_source #181): a DSL coupled source
      // targets a (block, role) EXPLICITLY requested by the user. The role is addressed BY NAME: a
      // canonical role name OR a user-defined role label (index_of(string), ADC-292). If the block does
      // NOT expose this role, a fallback to component 0 would apply the source to the wrong field
      // SILENTLY (the false-positive identified at the Lot E review). We throw, listing what the block
      // exposes.
      const VariableSet& vs = blocks_[static_cast<std::size_t>(b)].cons_vars;
      const int comp = vs.index_of(role);
      if (comp < 0)
        throw std::runtime_error(
            "AmrRuntime::add_coupled_source : block '" + block + "' does not expose role '" + role +
            "' (roles: " + (vs.roles.empty() ? std::string("<none>") : roles_csv(vs)) +
            ", no silent fallback to component 0)");
      return {b, comp};
    };
    // Inputs: (block, component) read per cell. Captured by INDEX -> we rebuild the Array4 at EACH
    // application (the fabs live in the level stack, repointed per level in the splitting).
    std::vector<CsRef> ins(static_cast<std::size_t>(n_in));
    for (int c = 0; c < n_in; ++c) {
      auto [b, comp] =
          resolve(in_blocks[static_cast<std::size_t>(c)], in_roles[static_cast<std::size_t>(c)]);
      ins[static_cast<std::size_t>(c)] = {b, comp, CsProgram{}};
    }
    std::vector<CsRef> outs(static_cast<std::size_t>(n_terms));
    int off = 0;
    for (int t = 0; t < n_terms; ++t) {
      auto [b, comp] =
          resolve(out_blocks[static_cast<std::size_t>(t)], out_roles[static_cast<std::size_t>(t)]);
      const int len = prog_lens[static_cast<std::size_t>(t)];
      if (len < 0 || len > kCsMaxProg)
        throw std::runtime_error("AmrRuntime::add_coupled_source : program of term " +
                                 std::to_string(t) + " too long (> " + std::to_string(kCsMaxProg) +
                                 ")");
      if (off + len > static_cast<int>(prog_ops.size()))
        throw std::runtime_error(
            "AmrRuntime::add_coupled_source : prog_lens inconsistent with prog_ops");
      CsProgram pg;
      pg.len = len;
      for (int k = 0; k < len; ++k) {
        const int opc = prog_ops[static_cast<std::size_t>(off + k)];
        const int a = prog_args[static_cast<std::size_t>(off + k)];
        if (opc < 0 || opc > static_cast<int>(CsOp::Sqrt))
          throw std::runtime_error("AmrRuntime::add_coupled_source : invalid opcode");
        if (opc == static_cast<int>(CsOp::PushReg) && (a < 0 || a >= n_in + n_const))
          throw std::runtime_error(
              "AmrRuntime::add_coupled_source : register out of bounds in the program");
        pg.op[k] = opc;
        pg.arg[k] = a;
      }
      validate_cs_program_stack(pg, "AmrRuntime::add_coupled_source term " + std::to_string(t));
      outs[static_cast<std::size_t>(t)] = {b, comp, pg};
      off += len;
    }
    std::vector<Real> kconsts(consts.begin(), consts.end());
    coupled_sources_.push_back(CoupledSourceSpec{std::move(ins), std::move(outs),
                                                 std::move(kconsts), n_in, n_const, n_terms});
  }

  /// Registers a TYPED coupling operator on the AMR runtime (ADC-595, parity with
  /// System::add_coupling_operator): validates the DECLARED conservation contract against the terms
  /// (host, fail-loud) BEFORE storing, lowers the program through the SAME add_coupled_source path
  /// (bit-identical), then records the declared contract for coupled_operators(). @p frequency /
  /// @p label name the operator's declared frequency bound in the inspect view (the AMR frequency
  /// bound itself is registered separately via add_coupled_freq / add_coupled_freq_expr).
  void add_coupling_operator(const CouplingOperator& op, double frequency,
                             const std::string& label) {
    validate_coupling_contract(op, "AmrRuntime::add_coupling_operator");
    const CoupledSourceProgram& p = op.program;
    add_coupled_source(p.in_blocks, p.in_roles, p.consts, p.out_blocks, p.out_roles, p.prog_ops,
                       p.prog_args, p.prog_lens);
    CouplingOperatorView view;
    view.label = label;
    view.conservation = op.conservation;
    view.frequency.constant_mu = frequency;
    view.frequency.per_cell = !p.freq_prog_ops.empty() || !p.freq_prog_args.empty();
    coupled_operators_.push_back(std::move(view));
  }

  /// Applies ALL the registered coupled sources of a step dt, by forward-Euler splitting. Runtime
  /// counterpart of AmrSystemCoupler::coupled_source_step: we refresh the fields (aux per level) then,
  /// source by source, we apply the bytecode INDEPENDENTLY AT EACH LEVEL of the shared hierarchy (the
  /// blocks live on ALL levels), followed by a fine -> coarse cascade.
  ///
  /// COVERAGE INVARIANT (#169): the source was applied independently on EACH level, so a coarse cell
  /// COVERED by a fine patch would otherwise carry its own coarse source, unrelated to the source seen
  /// by its fine children. A covered coarse cell MUST be the 2x2 average of its children (it does not
  /// represent matter on its own). We restore this consistency by the SAME fine -> coarse cascade
  /// (mf_average_down_mb) as solve_fields and the compile-time engine: without it, the mass diagnostic
  /// (sum of the coarse only) would count a phantom coarse source under the patch. Single-level
  /// hierarchy: no covered cell, the cascade loops do not run -> bit-identical to the no-patch case.
  ///
  /// PER-CELL CONSERVATION: at a given level, each term writes out(i,j,comp) += dt * S(reg(i,j)) on
  /// the SAME cell (i,j) read by the inputs; an add_pair exchange lays +S on one block and -S on the
  /// other AT THE SAME (i,j), so the sum of the two blocks is unchanged cell by cell. Without a
  /// registered source (coupled_sources_ empty): total no-op -> bit-identical trajectory.
  void coupled_source_step(Real dt) {
    if (coupled_sources_.empty())
      return;  // opt-in: no source -> bit-identical path
    require_solved_field_report(solve_fields(), "AmrRuntime::coupled_source_step");
    for (const auto& cs : coupled_sources_) {
      // PER-LEVEL application: at each level k, the blocks share EXACTLY the same layout
      // (same_layout_or_throw guard), so same local_size() and same local indexing -> we iterate in
      // parallel over the local fabs. local_size()==0 on a rank without a box -> empty loop (MPI-safe).
      for (int k = 0; k < nlev_; ++k) {
        const int sref = cs.n_in > 0 ? cs.ins[0].block : cs.outs[0].block;
        MultiFab& Uref = (*blocks_[static_cast<std::size_t>(sref)].levels)[k].U;
        for (int li = 0; li < Uref.local_size(); ++li) {
          CoupledSourceKernel kern;
          kern.dt = dt;
          kern.n_in = cs.n_in;
          kern.n_const = cs.n_const;
          kern.n_terms = cs.n_terms;
          for (int c = 0; c < cs.n_in; ++c) {
            kern.in[c] =
                (*blocks_[static_cast<std::size_t>(cs.ins[static_cast<std::size_t>(c)].block)]
                      .levels)[k]
                    .U.fab(li)
                    .array();
            kern.in_comp[c] = cs.ins[static_cast<std::size_t>(c)].comp;
          }
          for (int c = 0; c < cs.n_const; ++c)
            kern.consts[c] = cs.kconsts[static_cast<std::size_t>(c)];
          for (int t = 0; t < cs.n_terms; ++t) {
            kern.out[t] =
                (*blocks_[static_cast<std::size_t>(cs.outs[static_cast<std::size_t>(t)].block)]
                      .levels)[k]
                    .U.fab(li)
                    .array();
            kern.out_comp[t] = cs.outs[static_cast<std::size_t>(t)].comp;
            kern.prog[t] = cs.outs[static_cast<std::size_t>(t)].prog;
          }
          for_each_cell(Uref.box(li),
                        kern);  // NAMED functor (device-clean), additive forward-Euler
        }
      }
      // Restore the consistency of the covered coarse cells (cf. COVERAGE INVARIANT above).
      for (auto& b : blocks_) {
        if (!b.average_down_plan)
          throw std::logic_error("AMR block average-down plan was not prepared");
        for (int k = nlev_ - 1; k >= 1; --k)
          mf_average_down_mb((*b.levels)[k].U, (*b.levels)[k - 1].U,
                             b.average_down_plan->transition_for_child(k),
                             b.average_down_plan->topology_generation(),
                             world_communicator_view());
      }
    }
  }

  /// sync_down (per block) + system coarse Poisson (CO-LOCATED SUMMED RHS) + coarse aux + fine
  /// injection. Reproduces AmrSystemCoupler::solve_fields identically, but the system RHS is assembled
  /// by the blocks' add_elliptic_rhs closures (Sum_b elliptic_rhs_b(U_b)) not a compile-time RhsAssembler.
  SolveReport solve_fields() {
    return run_field_solve_transaction(
        FieldSolveScope{true, NamedFieldSnapshotScope::kAll, nullptr}, [&]() {
          SolveReport report = solve_default_field_uncommitted();
          if (!report.solved() || named_fields_.empty())
            return report;
          return solve_named_fields_uncommitted();
        });
  }

  SolveReport solve_default_field() {
    return run_field_solve_transaction(
        FieldSolveScope{true, NamedFieldSnapshotScope::kNone, nullptr},
        [&]() { return solve_default_field_uncommitted(); });
  }

  SolveReport solve_named_fields(const std::string* selected = nullptr) {
    return run_field_solve_transaction(
        FieldSolveScope{false,
                        selected == nullptr ? NamedFieldSnapshotScope::kAll
                                            : NamedFieldSnapshotScope::kSelected,
                        selected},
        [&]() { return solve_named_fields_uncommitted(selected); });
  }

  std::int64_t recompute_bootstrap_field(const std::string& field) {
    if (!bootstrap_pending_)
      throw std::runtime_error("AmrRuntime::recompute_bootstrap_field requires a transaction");
    SolveReport report;
    if (field == "phi") {
      report = solve_default_field();
    } else {
      if (!has_named_field(field))
        throw std::runtime_error("AmrRuntime::recompute_bootstrap_field has no runtime field '" +
                                 field + "'");
      report = solve_named_fields(&field);
    }
    if (!report.solved())
      throw std::runtime_error("AmrRuntime::recompute_bootstrap_field failed to solve '" + field +
                               "': status=" + report.status_name() +
                               " action=" + report.action_name());
    std::int64_t materialized = 0;
    for (const MultiFab& level : aux_)
      materialized += level.box_array().num_cells();
    return materialized;
  }

 private:
  SolveReport solve_default_field_uncommitted() {
    ++solve_count_;
    // 1. average_down per block (fine -> coarse) over the whole hierarchy. AMR PROFILING (Spec 5
    // criterion 43): time the restriction cascade into the "average_down" scope + bump its per-solve
    // count. The scope is per-solve_fields (NOT per-cell), so a profiled run pays one clock pair here;
    // an unprofiled run constructs nothing (profiler_ null or disabled). See profile_amr_scope below.
    {
      auto _ad = profile_amr_scope("average_down");
      if (profiler_ != nullptr)
        profiler_->count("average_down");
      for (auto& b : blocks_) {
        if (!b.average_down_plan)
          throw std::logic_error("AMR block average-down plan was not prepared");
        auto& L = *b.levels;
        for (int k = nlev_ - 1; k >= 1; --k)
          mf_average_down_mb(L[k].U, L[k - 1].U,
                             b.average_down_plan->transition_for_child(k),
                             b.average_down_plan->topology_generation(),
                             world_communicator_view());
      }
    }

    // 2. SUMMED and CO-LOCATED system RHS: f = Sum_b elliptic_rhs_b(U_b) on the coarse. We reset to
    // zero then each block ACCUMULATES (+=) its contribution on the SAME cells of the shared coarse
    // (the prepared provider RHS shares the coarse layout).
    default_field_solver_->rhs_level(0).set_val(Real(0));
    for (auto& b : blocks_)
      b.add_elliptic_rhs((*b.levels)[0].U, default_field_solver_->rhs_level(0));
    default_field_nullspace_workspace_->require_compatible(default_field_solver_->rhs_level(0));
    const SolveReport report = solve_prepared_amr_field_solver_collectively(*default_field_solver_);
    if (!report.solved())
      return report;
    default_field_nullspace_workspace_->apply_gauge(default_field_solver_->phi_level(0));

    // 3. coarse aux = (phi, grad phi) via the SAME clean path as AmrSystemCoupler: fill the ghosts of
    // phi according to bcPhi_, field_postprocess (phi + grad), fill the ghosts of aux according to
    // aux_bc_ (derived from bcPhi_). Handles the non-periodic case (Foextrap).
    fill_ghosts_profiled(default_field_solver_->phi_level(0), dom_, bcPhi_);
    const Real cx = Real(1) / (2 * geom_.dx()), cy = Real(1) / (2 * geom_.dy());
    field_postprocess(default_field_solver_->phi_level(0), aux_[0], cx, cy,
                      FieldPostProcess{FieldPostProcess::GradSign::Plus, true});
    // 3b. external static aux: re-apply canonical and model-named fields onto the coarse valid cells
    // BEFORE fill_ghosts (so their ghosts are filled) and the injection (so they reach every level).
    // No-op when no static field was set; field_postprocess wrote only comps 0..2, so this never clobbers
    // phi/grad. This is what makes supplied aux survive a regrid (regrid re-solves -> re-applies).
    apply_static_aux();
    publish_aux_components(default_aux_components());
    return report;
  }

  /// Solves every registered NAMED elliptic field (ADC-428) on the coarse, writes phi (+ centered grad)
  /// into the field's own aux components, ghost-fills them and injects coarse->fine. Mirror of the
  /// default Poisson block above (steps 2-4), but each named field uses its resolved prepared provider.
  /// The default phi/grad (comps 0..2) are never touched. No-op without a named field (default-only
  /// path stays bit-identical).
  SolveReport solve_named_fields_uncommitted(const std::string* selected = nullptr) {
    SolveReport completed;
    bool has_completed_solve = false;
    if (named_fields_.empty())
      throw std::runtime_error("AmrRuntime::solve_named_fields has no registered field");
    const Real dx = geom_.dx(), dy = geom_.dy();
    for (auto& [field, nf] : named_fields_) {
      if (selected != nullptr && field != *selected)
        continue;
      if (!nf.has_plan)
        throw std::runtime_error("AmrRuntime: field provider slot '" + field +
                                 "' has no resolved install plan");
      if (nf.phi_comp < 0 || nf.phi_comp >= aux_ncomp_)
        throw std::runtime_error("AmrRuntime : named elliptic field '" + field +
                                 "' aux component out of the channel width (add the block that "
                                 "declares its aux fields)");
      if (nf.gradient_sign != -1 && nf.gradient_sign != 1)
        throw std::runtime_error("AmrRuntime: named elliptic field has no valid gradient sign");
      const bool has_gradient = nf.gx_comp >= 0 && nf.gy_comp >= 0;
      if (nf.phi_comp >= aux_ncomp_ ||
          (has_gradient && (nf.gx_comp >= aux_ncomp_ || nf.gy_comp >= aux_ncomp_)))
        throw std::runtime_error(
            "AmrRuntime: named elliptic field output components exceed the aux channel width");
      nf.plan.boundary_state_buffers.clear();
      nf.plan.boundary_state_distributions.clear();
      for (std::size_t index = 0; index < nf.plan.boundary_state_blocks.size(); ++index) {
        const int block = block_index(nf.plan.boundary_state_blocks[index]);
        if (block < 0)
          throw std::runtime_error("AmrRuntime: boundary state dependency names unknown block");
        const MultiFab& state = (*blocks_[static_cast<std::size_t>(block)].levels)[0].U;
        if (nf.plan.boundary_state_components[index] < 0 ||
            nf.plan.boundary_state_components[index] >= state.ncomp())
          throw std::runtime_error("AmrRuntime: boundary state component is out of range");
        nf.plan.boundary_state_buffers.push_back(&state);
        nf.plan.boundary_state_distributions.push_back(
            replicated_coarse_ ? FieldDistribution::Replicated : FieldDistribution::Distributed);
      }
      nf.plan.boundary_context.states =
          nf.plan.boundary_state_buffers.empty() ? nullptr : nf.plan.boundary_state_buffers.data();
      nf.plan.boundary_context.state_distributions =
          nf.plan.boundary_state_distributions.empty()
              ? nullptr
              : nf.plan.boundary_state_distributions.data();
      nf.plan.boundary_context.state_count =
          static_cast<int>(nf.plan.boundary_state_buffers.size());
      ensure_named_elliptic(nf);
      if (nf.plan.has_boundary_kernel)
        nf.solver->set_boundary_context(nf.plan.boundary_context);
      prepare_named_field_providers(nf);
      prepare_named_rhs_scratch_(nf);
      // The provider registry has already resolved the complete block-qualified route.  Assembly
      // reads exactly one block and one closure; local provider names can therefore repeat freely in
      // different blocks and adding a second field never changes an existing RHS.
      auto assemble = [&](int level, MultiFab& rhs) {
        rhs.set_val(Real(0));
        for (const auto& binding : nf.prepared_providers) {
          auto& block = blocks_[binding.block];
          if (binding.coefficient == Real(1)) {
            binding.rhs((*block.levels)[static_cast<std::size_t>(level)].U, rhs);
          } else {
            MultiFab& contribution =
                nf.rhs_contribution_scratch.at(static_cast<std::size_t>(level));
            contribution.set_val(Real(0));
            binding.rhs((*block.levels)[static_cast<std::size_t>(level)].U, contribution);
            saxpy(rhs, binding.coefficient, contribution);
          }
        }
        // Public field equations use ``-div(A grad phi)+kappa*phi=rhs`` while the
        // native MG/FAC residual stores ``div(A grad phi)-kappa*phi=rhs_native``.
        scale(rhs, Real(-1));
      };
      prepare_named_nullspace(nf);
      for (int level = 0; level < nf.solver->level_count(); ++level)
        assemble(level, nf.solver->rhs_level(level));
      if (nf.solver->couples_hierarchy_levels()) {
        nf.nullspace_workspace->require_compatible(nf.nullspace_rhs_levels);
      } else {
        for (int level = 0; level < nf.solver->level_count(); ++level)
          if (!nf.level_nullspace.empty())
            nf.level_nullspace_workspaces[static_cast<std::size_t>(level)]->require_compatible(
                nf.solver->rhs_level(level));
      }
      completed = solve_prepared_amr_field_solver_collectively(*nf.solver);
      has_completed_solve = true;
      if (!completed.solved())
        return completed;
      device_fence();  // CRITICAL: the V-cycle must finish before phi is read (same invariant as mg_)
      if (nf.solver->level_count() > 1)
        continue;  // every multilevel field is published level-by-level below
      // Write phi (+ centered grad) into the field's OWN aux components on the coarse valid cells. The
      // default field_postprocess hardcodes comps 0..2, so we write the named comps with a dedicated
      // loop (mirror of SystemFieldSolver::solve_named_field_from_state). Per-local-fab (MPI-safe).
      MultiFab& phi_mf = nf.solver->phi_level(0);
      if (!nf.level_nullspace.empty())
        nf.level_nullspace_workspaces[0]->apply_gauge(phi_mf);
      device_fence();
      const int cphi = nf.phi_comp, cgx = nf.gx_comp, cgy = nf.gy_comp;
      const Real gradient_scale = static_cast<Real>(nf.gradient_sign);
      const bool grad = has_gradient;
      for (int li = 0; li < aux_[0].local_size(); ++li) {
        const ConstArray4 p = phi_mf.fab(li).const_array();
        Array4 a = aux_[0].fab(li).array();
        const Box2D v = aux_[0].box(li);
        for_each_cell(v, detail::AmrNamedFieldPostprocessKernel{a, p, cphi, cgx, cgy,
                                                                gradient_scale, dx, dy, grad});
      }
    }
    if (!has_completed_solve)
      throw std::runtime_error("AmrRuntime::solve_named_fields selected an unknown field");
    // Composite and level-local fields own a solved potential on every level. Write every valid
    // value before materialising halos so no coarse injection can overwrite refined solutions.
    for (auto& [field, nf] : named_fields_) {
      if (selected != nullptr && field != *selected)
        continue;
      if (nf.solver->level_count() <= 1)
        continue;
      const int cphi = nf.phi_comp, cgx = nf.gx_comp, cgy = nf.gy_comp;
      const Real gradient_scale = static_cast<Real>(nf.gradient_sign);
      const bool grad = nf.gx_comp >= 0 && nf.gy_comp >= 0;
      if (nf.solver->couples_hierarchy_levels())
        nf.nullspace_workspace->apply_gauge(nf.nullspace_phi_levels);
      for (int k = 0; k < nf.solver->level_count(); ++k) {
        MultiFab& phi = nf.solver->phi_level(k);
        if (!nf.solver->couples_hierarchy_levels() && !nf.level_nullspace.empty())
          nf.level_nullspace_workspaces[static_cast<std::size_t>(k)]->apply_gauge(phi);
        const Real refinement = static_cast<Real>(level_refinement(k));
        const Real level_dx = geom_.dx() / refinement;
        const Real level_dy = geom_.dy() / refinement;
        for (int li = 0; li < aux_[static_cast<std::size_t>(k)].local_size(); ++li) {
          const ConstArray4 p = phi.fab(li).const_array();
          Array4 a = aux_[static_cast<std::size_t>(k)].fab(li).array();
          const Box2D valid = aux_[static_cast<std::size_t>(k)].box(li);
          for_each_cell(valid, detail::AmrNamedFieldPostprocessKernel{
                                   a, p, cphi, cgx, cgy, gradient_scale, level_dx, level_dy, grad});
        }
      }
    }
    // The final hierarchy policies are exhaustive: a multilevel field is either composite or
    // level-local, and both own every refined valid cell. A single-level layout has no refined
    // authority to preserve. Publish the selected components once through the matching authority;
    // there is deliberately no hidden coarse-on-refined policy.
    std::set<int> components;
    auto add_component = [&](int component) {
      if (component >= 0 && component < aux_ncomp_)
        components.insert(component);
    };
    for (const auto& [field, nf] : named_fields_) {
      if (selected != nullptr && field != *selected)
        continue;
      add_component(nf.phi_comp);
      add_component(nf.gx_comp);
      add_component(nf.gy_comp);
    }
    const std::vector<int> published{components.begin(), components.end()};
    if (nlev_ > 1)
      publish_refined_aux_components(published);
    else
      publish_aux_components(published);
    return completed;
  }

 public:
  void prepare_tagging_states(int level, const Box2D& domain) {
    if (level < 0 || level >= nlev_ || tagging_program_.clock_identity.empty())
      throw std::runtime_error("AMR Tagger cannot qualify its ghost-production evaluation point");
    std::vector<bool> gradient_state(blocks_.size(), false);
    bool gradient_shared_aux = false;
    for (const auto& leaf : tagging_program_.leaves)
      if (leaf.opcode == POPS_TAGGING_GRADIENT_ABOVE_V1 ||
          leaf.opcode == POPS_TAGGING_GRADIENT_BELOW_V1) {
        if (leaf.state_index == blocks_.size())
          gradient_shared_aux = true;
        else
          gradient_state.at(leaf.state_index) = true;
      }
    if (std::none_of(gradient_state.begin(), gradient_state.end(),
                     [](bool value) { return value; }) &&
        !gradient_shared_aux)
      return;

    runtime::multiblock::BoundaryEvaluationPoint point{tagging_program_.clock_identity,
                                                       component_tick_,
                                                       level,
                                                       0,
                                                       0,
                                                       ::pops::amr::Rational(0, 1),
                                                       0.0,
                                                       component_physical_time_};
    if (boundary_stage_states_)
      throw std::runtime_error("AMR Tagger ghost production overlaps another boundary evaluation");
    std::map<std::string, std::size_t> qualified_routes;
    std::vector<MultiFab*> staged(blocks_.size(), nullptr);
    for (std::size_t block = 0; block < blocks_.size(); ++block) {
      if (blocks_[block].state_identity.empty() ||
          !qualified_routes.emplace(blocks_[block].state_identity, block).second)
        throw std::runtime_error("AMR Tagger requires unique qualified state storage routes");
      staged[block] = &(*blocks_[block].levels)[static_cast<std::size_t>(level)].U;
    }
    boundary_stage_states_.emplace(BoundaryStageStateView{point, &staged, -1, nullptr});
    struct StageStateReset {
      std::optional<BoundaryStageStateView>* slot;
      ~StageStateReset() { slot->reset(); }
    } reset{&boundary_stage_states_};

    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
      if (!gradient_state[block_index])
        continue;
      auto& block = blocks_[block_index];
      MultiFab& state = (*block.levels)[static_cast<std::size_t>(level)].U;
      if (static_cast<std::size_t>(level) < block.boundary_sessions.size() &&
          block.boundary_sessions[static_cast<std::size_t>(level)]) {
        block.boundary_sessions[static_cast<std::size_t>(level)]->fill(state, point);
        continue;
      }
      if (block.boundary_plan)
        throw std::runtime_error("AMR Tagger boundary plan has no persistent prepared session");
      fill_level_state_cf_ghosts(block_index, level, state);
      fill_ghosts(state, domain, transport_bc());
    }
    if (gradient_shared_aux)
      fill_ghosts(aux_.at(static_cast<std::size_t>(level)), domain, aux_bc_);
  }

  TagBox current_fine_coverage(const Box2D& parent_domain, int fine_level,
                               int refinement_ratio) const {
    TagBox current(parent_domain);
    if (fine_level < 0 || fine_level >= nlev_)
      return current;
    for (const Box2D& fine : hierarchy_.ba[static_cast<std::size_t>(fine_level)].boxes()) {
      const Box2D parent = fine.coarsen(refinement_ratio).intersect(parent_domain);
      for (int j = parent.lo[1]; j <= parent.hi[1]; ++j)
        for (int i = parent.lo[0]; i <= parent.hi[0]; ++i)
          current(i, j) = 1;
    }
    return current;
  }

  TagBox apply_tagging_decisions(const TagBox& refine, const TagBox& coarsen,
                                 const TagBox& refine_equalities, const TagBox& coarsen_equalities,
                                 TagBox result) const {
    if (refine.box != result.box || coarsen.box != result.box ||
        refine_equalities.box != result.box || coarsen_equalities.box != result.box)
      throw std::runtime_error("AMR Tagger candidate grids disagree on their parent domain");
    for (int j = result.box.lo[1]; j <= result.box.hi[1]; ++j)
      for (int i = result.box.lo[0]; i <= result.box.hi[0]; ++i) {
        const auto root = [](bool matches, bool equality) {
          return equality ? amr::TagTruth::Unknown
                          : (matches ? amr::TagTruth::True : amr::TagTruth::False);
        };
        const auto decision = amr::resolve_tag_decision(
            root(refine(i, j) != 0, refine_equalities(i, j) != 0),
            root(coarsen(i, j) != 0, coarsen_equalities(i, j) != 0),
            static_cast<amr::TagEqualityPolicy>(tagging_program_.equality_policy),
            static_cast<amr::TagConflictPolicy>(tagging_program_.conflict_policy));
        if (decision.conflict_error)
          throw std::runtime_error(
              "AMR tagging refine/coarsen conflict under ConflictPolicy.ERROR");
        if (decision.refine)
          result(i, j) = 1;
        else if (decision.coarsen)
          result(i, j) = 0;
      }
    return result;
  }

  TagBox execute_runtime_tagging_program(int parent_level, int fine_level,
                                         const Box2D& parent_domain, int refinement_ratio) {
    prepare_tagging_states(parent_level, parent_domain);
    const Geometry geometry = geom_.refine(level_refinement(parent_level));
    const auto& candidates = tagging_execution_plan_.execute(
        parent_level, parent_domain, geometry.dx(), geometry.dy(),
        topology_materialization_generation_);
    return apply_tagging_decisions(
        candidates.refine, candidates.coarsen, candidates.refine_equalities,
        candidates.coarsen_equalities,
        current_fine_coverage(parent_domain, fine_level, refinement_ratio));
  }

  TagBox execute_bootstrap_tagging_program(int level, const Box2D& domain) {
    prepare_tagging_states(level, domain);
    const Geometry geometry = geom_.refine(level_refinement(level));
    const auto& candidates = tagging_execution_plan_.execute(
        level, domain, geometry.dx(), geometry.dy(), topology_materialization_generation_);
    TagBox refine = candidates.refine;
    if (tagging_program_.equality_policy == 1)
      for (std::size_t index = 0; index < refine.t.size(); ++index)
        refine.t[index] = refine.t[index] || candidates.refine_equalities.t[index];
    return refine;
  }

  runtime::amr::PreparedTaggerCandidates execute_external_tagger(int parent_level,
                                                                 const Box2D& domain) {
    if (!external_tagger_ || !tagging_program_.prepared || blocks_.empty())
      throw std::runtime_error("AmrRuntime external Tagger provider is not installed");
    prepare_tagging_states(parent_level, domain);
    const bool uses_shared_aux =
        std::any_of(tagging_program_.leaves.begin(), tagging_program_.leaves.end(),
                    [this](const auto& leaf) { return leaf.state_index == blocks_.size(); });
    std::vector<runtime::amr::PreparedTaggingField> fields;
    fields.reserve(blocks_.size() + (uses_shared_aux ? 1u : 0u));
    for (auto& block : blocks_) {
      MultiFab& state = (*block.levels)[parent_level].U;
      fields.push_back(runtime::amr::PreparedTaggingField{block.state_identity, &state});
    }
    if (uses_shared_aux)
      fields.push_back(runtime::amr::PreparedTaggingField{
          "pops://runtime/amr/shared-aux",
          &aux_.at(static_cast<std::size_t>(parent_level))});
    const Geometry geometry = geom_.refine(level_refinement(parent_level));
    return external_tagger_->tag(fields, tagging_program_, domain, parent_level, component_tick_,
                                 component_physical_time_, static_cast<double>(geometry.dx()),
                                 static_cast<double>(geometry.dy()), base_per_.x, base_per_.y,
                                 parent_level == 0 && replicated_coarse_);
  }

  TagBox execute_external_bootstrap_tagging(int level, const Box2D& domain) {
    auto candidates = execute_external_tagger(level, domain);
    if (tagging_program_.equality_policy == 1)
      for (std::size_t index = 0; index < candidates.refine.t.size(); ++index)
        candidates.refine.t[index] =
            candidates.refine.t[index] || candidates.refine_equalities.t[index];
    return std::move(candidates.refine);
  }

  TagBox execute_external_regrid_tagging(int parent_level, int fine_level, const Box2D& domain,
                                         int refinement_ratio) {
    auto candidates = execute_external_tagger(parent_level, domain);
    return apply_tagging_decisions(candidates.refine, candidates.coarsen,
                                   candidates.refine_equalities, candidates.coarsen_equalities,
                                   current_fine_coverage(domain, fine_level, refinement_ratio));
  }

  /// Append one fine level from tags on the current finest parent. Unlike regrid(), this grows the
  /// hierarchy and is used only by the explicit bootstrap before stepping.
  void begin_bootstrap_plan() {
    if (bootstrap_pending_)
      throw std::runtime_error("AmrRuntime::begin_bootstrap_plan already has a transaction");
    capture_step_snapshot(bootstrap_snapshot_);
    bootstrap_pending_ = true;
  }

  bool bootstrap_next_level(int refinement_ratio) {
    if (!bootstrap_pending_)
      throw std::runtime_error("AmrRuntime::bootstrap_next_level requires begin_bootstrap_plan");
    if (refinement_ratio != kAmrRefRatio)
      throw std::runtime_error(
          "AmrRuntime::bootstrap_next_level requires spatial refinement ratio 2");
    const int pk = nlev_ - 1;
    if (static_cast<std::size_t>(pk) >= maximum_refinement_ratios_.size() ||
        maximum_refinement_ratios_[static_cast<std::size_t>(pk)] != refinement_ratio)
      throw std::runtime_error(
          "AmrRuntime::bootstrap_next_level exceeds or disagrees with hierarchy capacity");
    require_coarse_fine_reconstruction_contract_();
    const Box2D pdom = dom_.refine(level_refinement(pk));
    std::vector<TagBox> parts;
    parts.reserve(blocks_.size() + 1);
    if (external_tagger_) {
      parts.push_back(execute_external_bootstrap_tagging(pk, pdom));
    } else if (tagging_program_.prepared) {
      parts.push_back(execute_bootstrap_tagging_program(pk, pdom));
    } else {
      throw std::runtime_error(
          "AmrRuntime::bootstrap_next_level requires a prepared native tagging program");
    }
    if (parts.empty())
      throw std::runtime_error("AmrRuntime::bootstrap_next_level : no resolved tagging predicate");
    TagBox grown = grow_regrid_tags(tag_union(parts), regrid_grow_, pdom,
                                    RegridPeriodicity{base_per_.x, base_per_.y});
    const BoxArray* parents = pk > 0 ? &hierarchy_.ba[pk] : nullptr;
    const auto physical_support = regrid_physical_ghost_support_();
    auto [fb, dmap] = regrid_compute_fine_layout_with_provider(
        std::move(grown), pdom, pk, regrid_margin_, replicated_coarse_, *clustering_provider_,
        *hierarchy_.load_balance, world_communicator_view(), refinement_ratio, parents,
        RegridPeriodicity{base_per_.x, base_per_.y},
        physical_support ? &*physical_support : nullptr);
    if (fb.size() == 0)
      return false;
    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
      auto& block = blocks_[block_index];
      auto& levels = *block.levels;
      const MultiFab& parent = levels[pk].U;
      const Real fine_dx = levels[pk].dx / Real(refinement_ratio);
      const Real fine_dy = levels[pk].dy / Real(refinement_ratio);
      MultiFab fine(fb, dmap, parent.ncomp(), parent.n_grow());
      fine.set_val(Real(0));
      levels.push_back(AmrLevelMP{std::move(fine), nullptr, fine_dx, fine_dy});
    }
    MultiFab fine_aux(fb, dmap, aux_ncomp_, 1);
    fine_aux.set_val(Real(0));
    hierarchy_.ba.push_back(fb);
    hierarchy_.dm.push_back(dmap);
    hierarchy_.dx.push_back(hierarchy_.dx.back() / Real(refinement_ratio));
    hierarchy_.dy.push_back(hierarchy_.dy.back() / Real(refinement_ratio));
    hierarchy_.refinement_ratios.push_back(refinement_ratio);
    aux_.push_back(std::move(fine_aux));
    ++nlev_;
    refresh_active_temporal_plan_();
    for (auto& block : blocks_)
      for (int level = 0; level < nlev_; ++level)
        (*block.levels)[level].aux = &aux_[level];
    invalidate_named_field_topology();
    record_topology_replacement_();
    if (!static_aux_.empty()) {
      // The hierarchy may grow after bind-time aux publication.  A newly bootstrapped level must
      // therefore receive every static named aux before any following bootstrap materializer or
      // compiled Program can observe it.  Reuse the single coarse-to-fine publication authority and
      // restrict it to externally bound components so analytic/elliptic providers keep ownership of
      // their own aux slots.
      apply_static_aux();
      std::vector<int> components;
      components.reserve(static_aux_.size());
      for (const auto& [component, _] : static_aux_)
        if (component >= 0 && component < aux_ncomp_)
          components.push_back(component);
      publish_aux_components(components);
    }
    // Boundary sessions own level-layout-specific lanes, registries and provider routes.  The
    // bootstrap grows the hierarchy before the next parent can be tagged, so publish sessions for
    // every materialized level now; retaining the pre-bootstrap vector would either omit the new
    // level or expose stale topology-bound routes.
    materialize_boundary_sessions_();
    return true;
  }

  void commit_bootstrap_level() {
    if (!bootstrap_pending_)
      throw std::runtime_error("AmrRuntime::commit_bootstrap_level : no pending transaction");
    if (bootstrap_snapshot_.macro_step != 0 || macro_step_ != 0)
      throw std::runtime_error(
          "AmrRuntime::commit_bootstrap_level requires clocks to remain at t=0/step=0");
    for (const auto& [subject, cache] : bootstrap_caches_)
      if (!cache.valid || cache.materialized_level != nlev_ - 1)
        throw std::runtime_error("AmrRuntime::commit_bootstrap_level has a stale cache '" +
                                 subject + "'");
    for (const auto& [name, ring] : hist_rings_) {
      const auto owner = hist_block_owner_.find(name);
      if (owner == hist_block_owner_.end() || owner->second >= blocks_.size() || ring.size() != 2 ||
          ring[0].size() != static_cast<std::size_t>(nlev_) ||
          ring[1].size() != static_cast<std::size_t>(nlev_) ||
          ring[0][0].ncomp() != blocks_[owner->second].ncomp ||
          hist_init_[name].size() != static_cast<std::size_t>(nlev_) ||
          hist_slot_dt_[name].size() != ring.size())
        throw std::runtime_error("AmrRuntime::commit_bootstrap_level history '" + name +
                                 "' requires an explicit materialization provider");
      if (std::any_of(hist_init_[name].begin(), hist_init_[name].end(),
                      [](char value) { return value == 0; }))
        throw std::runtime_error("AmrRuntime::commit_bootstrap_level history '" + name +
                                 "' contains an uninitialized level");
    }
    bootstrap_pending_ = false;
  }

  void rollback_bootstrap_level() {
    if (!bootstrap_pending_)
      throw std::runtime_error("AmrRuntime::rollback_bootstrap_level : no pending transaction");
    restore_step_snapshot(bootstrap_snapshot_);
    bootstrap_pending_ = false;
  }

  int apply_bootstrap_component_floor(std::size_t block, int level, int component, Real floor) {
    if (block >= blocks_.size() || level < 0 || level >= nlev_ || component < 0 ||
        component >= blocks_[block].ncomp)
      throw std::runtime_error(
          "AmrRuntime::apply_bootstrap_component_floor received an incompatible target");
    MultiFab& state = (*blocks_[block].levels)[level].U;
    Real changed = Real(0);
    for (int li = 0; li < state.local_size(); ++li) {
      Array4 values = state.fab(li).array();
      const Box2D valid = state.box(li);
      changed +=
          for_each_cell_reduce_sum(valid, detail::BootstrapFloorKernel{values, component, floor});
    }
    return static_cast<int>(all_reduce_sum(static_cast<double>(changed)));
  }

  std::vector<TagBox> regrid_tag_parts_(int parent_level, int fine_level,
                                        const Box2D& parent_domain,
                                        int refinement_ratio) {
    std::vector<TagBox> parts;
    parts.reserve(blocks_.size() + 1);
    const bool fine_exists = fine_level < nlev_;
    if (external_tagger_) {
      parts.push_back(fine_exists
                          ? execute_external_regrid_tagging(parent_level, fine_level, parent_domain,
                                                            refinement_ratio)
                          : execute_external_bootstrap_tagging(parent_level, parent_domain));
    } else if (tagging_program_.prepared) {
      parts.push_back(fine_exists
                          ? execute_runtime_tagging_program(parent_level, fine_level, parent_domain,
                                                            refinement_ratio)
                          : execute_bootstrap_tagging_program(parent_level, parent_domain));
    } else {
      throw std::runtime_error(
          "AmrRuntime regrid requires a prepared native tagging program; host predicates are "
          "not a production fallback");
    }
    return parts;
  }

  std::optional<RegridPhysicalGhostSupport> regrid_physical_ghost_support_() const {
    if (base_per_.x && base_per_.y)
      return std::nullopt;
    int shared_depth = std::numeric_limits<int>::max();
    bool all_depths_supported = true;
    for (const AmrRuntimeBlock& block : blocks_) {
      if (block.boundary_plan) {
        if (!same_periodicity(block.boundary_plan->periodicity(), base_per_))
          throw std::runtime_error(
              "AMR regrid prepared boundary topology disagrees with the hierarchy");
        if ((!base_per_.x && (block.boundary_plan->omits_face(0, -1) ||
                              block.boundary_plan->omits_face(0, 1))) ||
            (!base_per_.y && (block.boundary_plan->omits_face(1, -1) ||
                              block.boundary_plan->omits_face(1, 1))))
          throw std::runtime_error(
              "AMR regrid boundary authority omits a physical domain face");
        if (!block.boundary_plan->fills_all_allocated_physical_ghosts()) {
          all_depths_supported = false;
          shared_depth = std::min(shared_depth, block.boundary_plan->required_depth());
        }
        continue;
      }
      if (!block.transport_boundary_fill)
        throw std::runtime_error(
            "non-periodic AMR regrid requires a prepared boundary authority for every block");
      validate_amr_boundary_fill_authority(base_per_, block.transport_boundary_fill.get(),
                                           *block.levels);
      if (!block.transport_boundary_fill->fills_all_allocated_ghosts) {
        all_depths_supported = false;
        shared_depth = std::min(shared_depth, block.transport_boundary_fill->provided_depth);
      }
    }
    if (!all_depths_supported && shared_depth == std::numeric_limits<int>::max())
      throw std::runtime_error("non-periodic AMR regrid has no state boundary authority");
    return RegridPhysicalGhostSupport{all_depths_supported ? 0 : shared_depth,
                                      all_depths_supported};
  }

  void require_coarse_fine_reconstruction_contract_(
      std::size_t block, const BlockTransferAuthority& authority) const {
    const int available_ghost_depth = authority.capabilities.ghost_depth.empty()
                                          ? 0
                                          : *std::min_element(
                                                authority.capabilities.ghost_depth.begin(),
                                                authority.capabilities.ghost_depth.end());
    if (!authority.prepared ||
        authority.capabilities.order < blocks_[block].reconstruction_order ||
        available_ghost_depth < blocks_[block].reconstruction_ghost_depth)
      throw std::runtime_error(
          "AMR block '" + blocks_[block].name + "' requires reconstruction order " +
          std::to_string(blocks_[block].reconstruction_order) + " with ghost depth " +
          std::to_string(blocks_[block].reconstruction_ghost_depth) +
          ", but the installed coarse/fine provider certifies only order " +
          std::to_string(authority.capabilities.order) + " with minimum directional ghost depth " +
          std::to_string(available_ghost_depth) +
          "; multilevel activation is refused instead of lowering the interface scheme");
  }

  void require_coarse_fine_reconstruction_contract_() const {
    if (block_transfer_authorities_.size() != blocks_.size())
      throw std::runtime_error("AMR coarse/fine transfer registry is incomplete");
    for (std::size_t block = 0; block < blocks_.size(); ++block) {
      require_coarse_fine_reconstruction_contract_(block, block_transfer_authorities_[block]);
      if (nlev_ > 1 &&
          (!blocks_[block].fill_patch_plan ||
           blocks_[block].fill_patch_plan->prepared_operator().get() !=
               block_transfer_authorities_[block].coarse_fine.prepared_coarse_fine.get()))
        throw std::runtime_error(
            "AMR FillPatch plan does not retain the resolved coarse/fine provider authority");
      if (nlev_ > 1 &&
          (blocks_[block].coarse_fine_spatial_workspaces.size() !=
               static_cast<std::size_t>(nlev_ - 1) ||
           std::any_of(
               blocks_[block].coarse_fine_spatial_workspaces.begin(),
               blocks_[block].coarse_fine_spatial_workspaces.end(),
               [&](const auto& workspace) {
                 return workspace.prepared_operator().get() !=
                        block_transfer_authorities_[block]
                            .coarse_fine.prepared_coarse_fine.get();
               })))
        throw std::runtime_error(
            "AMR spatial transfer plan does not retain the resolved coarse/fine provider "
            "authority");
    }
  }

  void materialize_regrid_transition_(int parent_level, const BoxArray& boxes,
                                      const DistributionMapping& distribution,
                                      int refinement_ratio) {
    const int fine_level = parent_level + 1;
    const bool existed = fine_level < nlev_;
    if (!existed)
      require_coarse_fine_reconstruction_contract_();
    std::vector<MultiFab> remapped;
    remapped.reserve(blocks_.size());
    for (std::size_t block = 0; block < blocks_.size(); ++block) {
      auto& levels = *blocks_[block].levels;
      const MultiFab& parent = levels[static_cast<std::size_t>(parent_level)].U;
      const int ghost_depth = existed ? levels[static_cast<std::size_t>(fine_level)].U.n_grow()
                                      : parent.n_grow();
      MultiFab empty(BoxArray{}, DistributionMapping{}, parent.ncomp(), ghost_depth);
      const MultiFab& old_fine =
          existed ? levels[static_cast<std::size_t>(fine_level)].U : empty;
      remapped.push_back(regrid_block_field(block, boxes, distribution, parent, old_fine,
                                            parent_level, ghost_depth, refinement_ratio));
    }

    if (!existed) {
      hierarchy_.ba.push_back(boxes);
      hierarchy_.dm.push_back(distribution);
      hierarchy_.dx.push_back(hierarchy_.dx[static_cast<std::size_t>(parent_level)] /
                              Real(refinement_ratio));
      hierarchy_.dy.push_back(hierarchy_.dy[static_cast<std::size_t>(parent_level)] /
                              Real(refinement_ratio));
      hierarchy_.refinement_ratios.push_back(refinement_ratio);
      aux_.emplace_back(boxes, distribution, aux_ncomp_, 1);
      ++nlev_;
      refresh_active_temporal_plan_();
      for (std::size_t block = 0; block < blocks_.size(); ++block) {
        auto& levels = *blocks_[block].levels;
        levels.push_back(AmrLevelMP{std::move(remapped[block]), &aux_.back(),
                                    levels[static_cast<std::size_t>(parent_level)].dx /
                                        Real(refinement_ratio),
                                    levels[static_cast<std::size_t>(parent_level)].dy /
                                        Real(refinement_ratio)});
      }
    } else {
      hierarchy_.ba[static_cast<std::size_t>(fine_level)] = boxes;
      hierarchy_.dm[static_cast<std::size_t>(fine_level)] = distribution;
      aux_[static_cast<std::size_t>(fine_level)] = MultiFab(boxes, distribution, aux_ncomp_, 1);
      for (std::size_t block = 0; block < blocks_.size(); ++block)
        (*blocks_[block].levels)[static_cast<std::size_t>(fine_level)].U =
            std::move(remapped[block]);
    }
    remap_history_rings_(boxes, distribution, fine_level, parent_level, /*prolong=*/true);
    for (auto& block : blocks_)
      for (int level = 0; level < nlev_; ++level)
        (*block.levels)[static_cast<std::size_t>(level)].aux =
            &aux_[static_cast<std::size_t>(level)];
  }

  void remove_levels_above_(int parent_level) {
    if (parent_level < 0 || parent_level >= nlev_)
      throw std::runtime_error("AmrRuntime::remove_levels_above invalid parent level");
    remove_history_levels_above_(parent_level);
    for (int fine_level = nlev_ - 1; fine_level > parent_level; --fine_level) {
      const int coarse_level = fine_level - 1;
      const int ratio = hierarchy_.refinement_ratios[static_cast<std::size_t>(coarse_level)];
      for (std::size_t block = 0; block < blocks_.size(); ++block) {
        auto& levels = *blocks_[block].levels;
        restrict_block_field(block, levels[static_cast<std::size_t>(fine_level)].U,
                             levels[static_cast<std::size_t>(coarse_level)].U, coarse_level,
                             ratio);
        levels.pop_back();
      }
    }
    const std::size_t active = static_cast<std::size_t>(parent_level + 1);
    hierarchy_.ba.resize(active);
    hierarchy_.dm.resize(active);
    hierarchy_.dx.resize(active);
    hierarchy_.dy.resize(active);
    hierarchy_.refinement_ratios.resize(static_cast<std::size_t>(parent_level));
    aux_.resize(active);
    nlev_ = parent_level + 1;
    refresh_active_temporal_plan_();
    for (auto& block : blocks_)
      for (int level = 0; level < nlev_; ++level)
        (*block.levels)[static_cast<std::size_t>(level)].aux =
            &aux_[static_cast<std::size_t>(level)];
  }

  /// UNION-TAGS REGRID (capstone Phase 2, C.6; docs/AMR_REGRID_UNION_TAGS_DESIGN.md, steps R0-R8).
  /// Re-grids the SHARED hierarchy from the UNION (cell-by-cell OR) of the tags of ALL blocks (per-block
  /// predicate, D1) + the phi tags (on |grad phi|, D4), followed by ONE SINGLE Berger-Rigoutsos
  /// clustering -> ONE SINGLE new fine layout applied to ALL blocks (including those held by their
  /// stride, D3) AND to the shared aux. Maintains the shared-layout PRECONDITION (same_layout_or_throw)
  /// after the regrid. Every active transition is rebuilt coarse-to-fine; an empty transition
  /// removes its fine suffix, while a newly tagged transition activates the next level up to the
  /// resolved maximum depth.
  void regrid() {
    if (max_levels() < 2)
      return;

    auto scope = profile_amr_scope("regrid");
    const std::int64_t copy_miss_before = copy_schedule_miss_count();
    const std::int64_t copy_hit_before = copy_schedule_hit_count();
    const bool autonomous_transaction = !step_rollback_scope_active();
    if (autonomous_transaction)
      capture_step_snapshot(regrid_snapshot_);
    bool changed = false;
    try {
      // Field-dependent taggers observe an accepted, hierarchy-consistent state. Each changed
      // transition below republishes fields before the next parent is tagged.
      require_solved_field_report(solve_fields(), "AmrRuntime::regrid precondition");

      for (int parent_level = 0;
           parent_level < max_levels() - 1 && parent_level < nlev_; ++parent_level) {
        const int fine_level = parent_level + 1;
        const int refinement_ratio =
            maximum_refinement_ratios_[static_cast<std::size_t>(parent_level)];
        const Box2D parent_domain = dom_.refine(level_refinement(parent_level));
        std::vector<TagBox> parts =
            regrid_tag_parts_(parent_level, fine_level, parent_domain, refinement_ratio);
        if (parts.empty())
          break;

        TagBox grown = grow_regrid_tags(tag_union(parts), regrid_grow_, parent_domain,
                                        RegridPeriodicity{base_per_.x, base_per_.y});
        if (profiler_ != nullptr) {
          const std::int64_t total = grown.box.num_cells();
          if (total > 0)
            profiler_->count("tag_density", (grown.count() * 1000) / total);
        }

        const BoxArray* parents =
            parent_level > 0 ? &hierarchy_.ba[static_cast<std::size_t>(parent_level)] : nullptr;
        const auto physical_support = regrid_physical_ghost_support_();
        auto [boxes, distribution] = regrid_compute_fine_layout_with_provider(
            std::move(grown), parent_domain, parent_level, regrid_margin_, replicated_coarse_,
            *clustering_provider_, *hierarchy_.load_balance, world_communicator_view(),
            refinement_ratio, parents,
            RegridPeriodicity{base_per_.x, base_per_.y},
            physical_support ? &*physical_support : nullptr);
#ifdef POPS_HAS_MPI
        if (profiler_ != nullptr && n_ranks() > 1 &&
            !(parent_level == 0 && replicated_coarse_))
          profiler_->count("mpi_reductions");
#endif

        if (boxes.size() == 0) {
          if (fine_level < nlev_) {
            // The restriction route owns the last fine -> parent publication before storage is
            // removed. All deeper levels disappear with the first inactive transition.
            remove_levels_above_(parent_level);
            invalidate_named_field_topology();
            record_topology_replacement_();
            require_solved_field_report(solve_fields(), "AmrRuntime::regrid coarsening publication");
            materialize_boundary_sessions_();
            changed = true;
          }
          break;
        }

        // A tagger may reproduce the accepted topology exactly. Preserve every layout-bound cache,
        // history slot, aux buffer and boundary session in that case; deeper transitions must still be
        // inspected because their tags may have changed independently.
        if (fine_level < nlev_ &&
            boxes.boxes() == hierarchy_.ba[static_cast<std::size_t>(fine_level)].boxes() &&
            distribution.ranks() ==
                hierarchy_.dm[static_cast<std::size_t>(fine_level)].ranks())
          continue;

        materialize_regrid_transition_(parent_level, boxes, distribution, refinement_ratio);
        invalidate_named_field_topology();
        record_topology_replacement_();
        require_solved_field_report(solve_fields(), "AmrRuntime::regrid transition publication");
        materialize_boundary_sessions_();
        changed = true;
      }

      if (!changed)
        return;

      const auto& reference_levels = *blocks_.front().levels;
      for (std::size_t block = 1; block < blocks_.size(); ++block) {
        const auto& candidate_levels = *blocks_[block].levels;
        if (candidate_levels.size() != reference_levels.size())
          throw std::runtime_error(
              "AmrRuntime::regrid produced different level counts across blocks");
        for (std::size_t level = 0; level < reference_levels.size(); ++level)
          if (!detail::same_level_layout(
                  candidate_levels[level].U.box_array(), candidate_levels[level].U.dmap(),
                  candidate_levels[level].dx, candidate_levels[level].dy,
                  reference_levels[level].U.box_array(), reference_levels[level].U.dmap(),
                  reference_levels[level].dx, reference_levels[level].dy))
            throw std::runtime_error(
                "AmrRuntime::regrid produced non-shared block layouts");
      }

      ++regrid_count_;
      if (profiler_ != nullptr) {
        profiler_->count("regrid");
        const std::int64_t misses = copy_schedule_miss_count() - copy_miss_before;
        const std::int64_t hits = copy_schedule_hit_count() - copy_hit_before;
        profiler_->count("box_hash_rebuilds", misses);
        profiler_->count("copy_cache_misses", misses);
        profiler_->count("copy_cache_hits", hits);
      }
    } catch (...) {
      if (autonomous_transaction)
        restore_step_snapshot(regrid_snapshot_);
      throw;
    }
  }
  /// Advances the system by one macro-step dt. We first solve the fields (co-located summed Poisson,
  /// ONCE per macro-step: OncePerStep cadence), then each block advances over ITS level stack with ITS
  /// scheme, honoring its stride cadence and its substeps, and ITS temporal treatment. Runtime
  /// counterpart of AmrSystemCoupler::step (OncePerStep): the compile-time version carries
  /// substeps/stride in block_substeps_v / block_stride_v and chooses the treatment by the constexpr
  /// block_time_treatment_v; here the engine carries the substep loop, the stride filter AND the
  /// IMEX-vs-explicit selection.
  ///
  /// TREATMENT SELECTION (capstone vii):
  ///  - EXPLICIT block (b.imex == false): the advance closure does ONE advance_amr (transport +
  ///    forward-Euler source), called substeps times;
  ///  - IMEX block (b.imex == true): the imex_advance closure does ONE SOURCE-FREE advance_amr then the
  ///    IMPLICIT stiff source backward_euler_source per level + cascade (cf.
  ///    AmrRuntimeBlock::imex_advance), called substeps times. Unconditionally stable on a stiff
  ///    relaxation (where the explicit, of factor |1 - dt/eps|, DIVERGES as soon as dt > 2 eps).
  /// The substep loop is COMMON to both treatments (substeps applications of h = bdt/substeps), so the
  /// runtime also SUB-CYCLES the IMEX splitting. At substeps=1 this sub-cycling is a no-op and the IMEX
  /// path coincides with the IMEX branch of the compile-time engine AmrSystemCoupler::step; for
  /// substeps>1 it DIVERGES deliberately from that engine (which itself ignores substeps on its IMEX
  /// branch): see IMEX SEMANTICS UNDER substeps in the header (CFL-safe on the transport,
  /// backward-Euler stable at any step, stiff relaxation more accurate). imex == false everywhere ->
  /// advance path only -> bit-identical trajectory to the historical one (the IMEX is opt-in).
  void step(Real dt) {
    // PREPARE before any observable mutation (including regrid, field warm-start and diagnostics).
    // This rejects an invalid rational IntegralOnly relation or an old block without the explicit
    // execution closure while the accepted numerical state is still untouched.
    preflight_native_temporal_step_();
    solve_count_ = 0;
    // UNION-TAGS REGRID (capstone Phase 2, C.6; D2: BEFORE the macro-step's step, consistent with the
    // single-block amr_dsl_block.hpp:108). regrid_every_ cadence in MACRO-STEPS, OUTSIDE the substep
    // loops and the stride windows (macro-step granularity ONLY, D3). regrid_every_ == 0 -> FROZEN
    // hierarchy, regrid never called -> BIT-IDENTICAL trajectory to the historical one. The guard
    // macro_step_ > 0 (like the single-block) avoids a regrid at the very first step (the initial grid
    // is already the build one). The regrid sits BEFORE solve_fields below: it does its own
    // solve_fields (R0/R8), then the step's solve_fields recomputes phi on the re-gridded grid.
    if (regrid_every_ > 0 && macro_step_ > 0 && macro_step_ % regrid_every_ == 0)
      regrid();
    // System Poisson solved ONCE on the current state (OncePerStep cadence). A HELD block (stride > 1,
    // outside end-of-window) contributed with its FROZEN state since its last advance: loose coupling
    // assumed by the multirate, exactly like System::step / AmrSystemCoupler in OncePerStep. phi stays
    // frozen during the blocks' advance (no per-substep re-solve here). When reached from step_cfl this
    // re-solves an unchanged state (a second solve), kept on purpose; see the ADC-318 note in step_cfl.
    require_solved_field_report(solve_fields(), "AmrRuntime::step");
    for (auto& b : blocks_) {
      // HOLD-THEN-CATCH-UP cadence (cf. AmrRuntimeBlock::stride, #140): the block is HELD as long as
      // (macro_step_+1) % stride != 0, then CATCHES UP at end-of-window by an effective step stride*dt.
      // The end-of-window catch-up keeps the block temporally consistent with the fast ones at the
      // coupling point (never in the future). stride=1: always true -> every step, bit-identical.
      if ((macro_step_ + 1) % b.stride != 0)
        continue;
      // NEWTON DIAGNOSTICS (OPT-IN): RESET of the report at the HEAD of the block advance (parity with
      // System::AdvanceImex::operator() which resets nreport before its substep loop). The report then
      // AGGREGATES over all the levels AND substeps of THIS advance (imex_advance accumulates per level
      // via backward_euler_source; step() calls imex_advance substeps times without re-resetting).
      // Placed AFTER the stride skip: a HELD block keeps the report of its LAST advance ("last advance"
      // semantics of System). No-op for a block without diagnostics (newton_report null).
      if (b.newton_diagnostics && b.newton_report)
        b.newton_report->reset();
      const Real bdt = dt * static_cast<Real>(b.stride);  // catch-up: effective step stride*dt
      // substeps equal substeps of bdt/substeps. The chosen closure does ONE advance per call;
      // substeps=1 -> a single advance of bdt (bit-identical to the single-substep case). Per-block
      // treatment SELECTION: IMEX (source-free transport + implicit stiff source, mirrors the IMEX
      // branch of AmrSystemCoupler::step) if b.imex, otherwise EXPLICIT (transport + forward-Euler
      // source). The test is PER BLOCK and stable: a single IMEX block changes nothing for the
      // neighboring explicit blocks.
      // NOTE substeps>1: the loop below calls step_block substeps times for BOTH treatments, so the
      // IMEX splitting is SUB-CYCLED (K Lie steps over bdt/K). The compile-time, for its part, applies
      // its IMEX only once over bdt (it ignores substeps on its IMEX branch): divergence INTENTIONAL
      // and sound for substeps>1 (cf. IMEX SEMANTICS UNDER substeps in the file header).
      const Real h = bdt / static_cast<Real>(b.substeps);
      for (int s = 0; s < b.substeps; ++s) {
        if (!b.fill_patch_plan || !b.average_down_plan || !b.advance_scratch_plan)
          throw std::logic_error(
              "AMR block lost its prepared topology execution plans");
        if (has_explicit_temporal_relations_()) {
          auto& step_block =
              b.imex ? b.imex_advance_with_temporal_plan : b.advance_with_temporal_plan;
          step_block(*b.levels, dom_, h, base_per_, replicated_coarse_,
                     *temporal_execution_plan_, &*b.fill_patch_plan,
                     &*b.average_down_plan, &*b.advance_scratch_plan);
        } else {
          // Low-level compatibility route: no temporal relation was installed, so the block keeps
          // the historical spatial-ratio cadence.  This branch is unreachable once a relation exists.
          auto& step_block = b.imex ? b.imex_advance : b.advance;
          step_block(*b.levels, dom_, h, base_per_, replicated_coarse_,
                     &*b.fill_patch_plan, &*b.average_down_plan,
                     &*b.advance_scratch_plan);
        }
      }
      // PROJECTION PONCTUELLE post-pas (ADC-177) : par niveau, APRES substeps + reflux/cascade.
      // Cell-local + idempotente -> conservation preservee (flux-registres deja regles). No-op si vide.
      if (b.project_per_level)
        b.project_per_level(*b.levels);
    }
    // Inter-species coupled sources AFTER the transport (same order as AmrSystemCoupler: transport then
    // coupled_source_step), by forward-Euler splitting. No-op if no source registered -> bit-identical
    // trajectory to the historical one (the feature is opt-in).
    coupled_source_step(dt);
    ++macro_step_;
  }

  /// substeps/stride-aware CFL step (runtime counterpart of System::step_cfl, EXACT mirror of its
  /// formula). A block of stride cadence advances by an effective step stride*dt in substeps substeps,
  /// so each substep is worth stride*dt/substeps; the per-substep stability condition
  /// stride*dt/substeps <= cfl*h/w_b gives dt <= cfl*h*substeps_b/(stride_b*w_b). The GLOBAL dt is the
  /// min over the blocks (the most constraining). We first solve the fields (per-block max_speed
  /// requires the aux up to date), compute dt, then advance by one step(dt). @p h = coarse mesh spacing
  /// (dx_coarse). Returns the dt used. Single-block (a single block, stride=1): if w_b is the only
  /// constraining one, dt = cfl*h*substeps/w (identical to System::step_cfl single-block).
  Real step_cfl(Real cfl, Real h, Real speed_floor = kCflSpeedFloor) {
    const Real dt = cfl_dt(cfl, h, speed_floor);
    step(dt);
    return dt;
  }

  /// The CFL dt computation of @ref step_cfl WITHOUT the trailing advance (no step(dt)): solves the
  /// fields (max_speed needs the aux), scans the per-block transport / source / stability bounds + the
  /// coupled-frequency + global bounds, and returns the macro-step dt (records last_dt_reason_). Split
  /// out so an installed compiled Program can take the SAME CFL dt and drive the macro-step itself
  /// (AmrSystem::step_cfl's Program route, parity SystemStepper::step_cfl) instead of the native step.
  /// The native @ref step_cfl path is byte-identical (it is this body + step(dt)).
  Real cfl_dt(Real cfl, Real h, Real speed_floor = kCflSpeedFloor) {
    preflight_native_temporal_step_();
    // This pre-solve provides the field required by max_speed. step(dt) keeps its own transaction-level
    // field solve, but the unchanged warm start now exits at zero V-cycles because GeometricMG measures
    // convergence against ||rhs|| rather than demanding another rel_tol factor from its incoming
    // residual. The two public operations therefore retain independent failure/reporting boundaries
    // without numerical over-solving.
    require_solved_field_report(solve_fields(), "AmrRuntime::cfl_dt");
    Real dt = std::numeric_limits<Real>::infinity();
    last_dt_reason_ = "degenerate";
    for (auto& b : blocks_) {
      Real dt_b = std::numeric_limits<Real>::infinity();
      const char* why = "transport";
      // Every active level contributes a stability bound.  For level l, one block step is divided
      // by the authored temporal product T_l while its cell width is divided by the independent
      // spatial product S_l.  Therefore a macro dt is admissible iff
      //   dt <= cfl*(h/S_l)*substeps*T_l/(stride*w_l).
      // Rational ExplicitFinalSubstep relations use their nominal ratio: the declared remainder is
      // shorter than the nominal child interval and cannot be the restrictive interval.
      for (int level = 0; level < nlev_; ++level) {
        const auto index = static_cast<std::size_t>(level);
        const Real temporal_product = temporal_refinement_product_(level);
        const Real block_scale =
            static_cast<Real>(b.substeps) * temporal_product / static_cast<Real>(b.stride);
        const Real level_h = h / static_cast<Real>(level_refinement(level));
        // ADC-645: caller-facing speed floor (default = historical kCflSpeedFloor).
        const Real w = std::max(b.max_speed((*b.levels)[index].U, aux_[index]), speed_floor);
        const Real dt_transport = cfl * level_h * block_scale / w;
        if (dt_transport < dt_b) {
          dt_b = dt_transport;
          why = "transport";
        }
        // OPTIONAL block bounds use the same per-level temporal product, without a spatial factor.
        if (b.source_frequency) {
          const Real mu = b.source_frequency((*b.levels)[index].U, aux_[index]);
          if (mu > Real(0)) {
            const Real dt_src = cfl * block_scale / mu;
            if (dt_src < dt_b) {
              dt_b = dt_src;
              why = "source_frequency";
            }
          }
        }
        if (b.stability_dt) {
          const Real db = b.stability_dt((*b.levels)[index].U, aux_[index]);
          if (db > Real(0)) {
            const Real dt_adm = db * block_scale;
            if (dt_adm < dt_b) {
              dt_b = dt_adm;
              why = "stability_dt";
            }
          }
        }
      }
      if (dt_b < dt) {
        dt = dt_b;
        last_dt_reason_ = std::string(why) + ":" + b.name;
      }
    }
    // Declared frequencies of the coupled sources (CoupledSource.frequency): bound on the MACRO-step
    // (the couplings apply once per macro-step), dt <= cfl / mu, without substeps/stride.
    for (const auto& cs : coupled_freqs_) {
      const Real dt_cs = cfl / cs.mu;
      if (dt_cs < dt) {
        dt = dt_cs;
        last_dt_reason_ = "coupled_source:" + cs.label;
      }
    }
    // PER-CELL frequencies (CoupledSource.frequency with an Expr): mu(U) reduced by MAX over every
    // active level, then by one GLOBAL all_reduce_max. Covered coarse values may be visited as well,
    // which is harmless for a maximum and conservative for stability; omitting fine-only extrema
    // would under-estimate mu and admit an unsafe macro-step. Array4 views are rebuilt after regrid.
    for (const auto& ce : coupled_freq_exprs_) {
      Real m = 0;
      if (ce.n_in > 0) {
        for (int level = 0; level < nlev_; ++level) {
          auto& Uref =
              (*blocks_[static_cast<std::size_t>(ce.ins[0].block)].levels)[level].U;
          for (int li = 0; li < Uref.local_size(); ++li) {
            CoupledFreqKernel kern;
            kern.n_in = ce.n_in;
            kern.n_const = ce.n_const;
            for (int c = 0; c < ce.n_in; ++c) {
              kern.in[c] =
                  (*blocks_[static_cast<std::size_t>(ce.ins[static_cast<std::size_t>(c)].block)]
                        .levels)[level]
                      .U.fab(li)
                      .array();
              kern.in_comp[c] = ce.ins[static_cast<std::size_t>(c)].comp;
            }
            for (int c = 0; c < ce.n_const; ++c)
              kern.consts[c] = ce.kconsts[static_cast<std::size_t>(c)];
            kern.prog = ce.prog;
            m = std::max(m, reduce_max_cell(Uref.box(li), kern));
          }
        }
      } else {
        // Program WITHOUT an input field (constant in bytecode): evaluated once on the constants.
        Real reg[kCsMaxReg];
        for (int c = 0; c < ce.n_const; ++c)
          reg[c] = ce.kconsts[static_cast<std::size_t>(c)];
        const Real mu0 = ce.prog.eval(reg);
        if (mu0 > Real(0))
          m = mu0;
      }
      const double mu = all_reduce_max(static_cast<double>(m));  // ALL ranks (collective symmetry)
#ifdef POPS_HAS_MPI
      // MPI COLLECTIVE COUNT (Spec 5 criterion 43): one all_reduce_max per per-cell coupled-frequency
      // bound, multi-rank only (serial all_reduce_max is an identity, no collective).
      if (profiler_ != nullptr && n_ranks() > 1)
        profiler_->count("mpi_reductions");
#endif
      if (mu > 0.0) {
        const Real dt_cs = cfl / static_cast<Real>(mu);
        if (dt_cs < dt) {
          dt = dt_cs;
          last_dt_reason_ = "coupled_source:" + ce.label;
        }
      }
    }
    // GLOBAL bounds (AmrRuntime::add_dt_bound, parity with System::add_dt_bound): evaluated PER RANK
    // then reduced all_reduce_min (dt identical on all ranks; <= 0/non-finite = inert).
    for (const auto& g : dt_bounds_) {
      if (!g.fn)
        continue;
      double v = g.fn();
      if (!(v > 0.0) || !std::isfinite(v))
        v = std::numeric_limits<double>::infinity();
      v = all_reduce_min(v);
#ifdef POPS_HAS_MPI
      // MPI COLLECTIVE COUNT (Spec 5 criterion 43): one all_reduce_min per registered global dt bound,
      // multi-rank only (the global min keeps dt identical on all ranks). Serial -> identity, no count.
      if (profiler_ != nullptr && n_ranks() > 1)
        profiler_->count("mpi_reductions");
#endif
      if (static_cast<Real>(v) < dt) {
        dt = static_cast<Real>(v);
        last_dt_reason_ = "global:" + g.label;
      }
    }
    if (!std::isfinite(dt)) {
      dt = cfl * h / kCflSpeedFloor;  // guard (no block: impossible here)
      last_dt_reason_ = "degenerate";
    }
    return dt;
  }

  /// MACRO-STEP counter of the engine (regrid + hold-then-catch-up stride cadence: regrid when
  /// macro_step_ % regrid_every == 0, stride catch-up when (macro_step_+1) % stride == 0).
  int macro_step() const { return macro_step_; }
  /// RESTORES the macro-step counter (accepted-state v3, via AmrSystem::set_clock): without
  /// it the regrid/stride cadence would restart from phase 0 after a resume. No effect on the level
  /// state; only sets the cadence phase.
  void set_macro_step(int s) { macro_step_ = s; }

  /// AMR / MPI PROFILING SEAM (Spec 5 sec.12.5, ADC-479 criterion 43). The AmrSystem owns the
  /// runtime::program::Profiler (parity with System::profiler_) and wires it in here AFTER build, so
  /// the engine times its non-numeric AMR phases -- regrid, fill_boundary (the cross-rank ghost
  /// exchange), average_down (fine -> coarse restriction) -- into the SAME table profile_report()
  /// renders, alongside the coarse step / field_solve phases. The pointer is null by default (the
  /// engine never touches it), and every scope/count is guarded by profiler_->enabled(), so a run
  /// WITHOUT profiling pays ZERO cost (no scope object, no clock read) -- the granularity is
  /// per-regrid / per-solve, NOT per-cell. Passing nullptr detaches the profiler (no-op timing).
  void set_profiler(runtime::program::Profiler* prof) { profiler_ = prof; }

  /// GLOBAL step bound (AMR counterpart of System::add_dt_bound): fn() evaluated once per step_cfl,
  /// all_reduce_min, <= 0/non-finite = inert. For user coupling/scheduler/policies.
  void add_dt_bound(const std::string& label, std::function<double()> fn) {
    dt_bounds_.push_back(GlobalDtBound{label, std::move(fn)});
  }

  /// DECLARED frequency of a coupled source (CoupledSource.frequency, wave-3 audit): step bound
  /// dt <= cfl / mu on the MACRO-step (the couplings apply once per macro-step). mu <= 0 = inert (no
  /// bound).
  void add_coupled_frequency(const std::string& label, Real mu) {
    if (mu > Real(0))
      coupled_freqs_.push_back(CoupledFreqDecl{label, mu});
  }

  /// PER-CELL COUPLED frequency (CoupledSource.frequency with an Expr, refinement of the CONSTANT
  /// frequency above): a bytecode program mu(U) on the SAME register table as the source (inputs
  /// in_blocks/in_roles then constants consts). Evaluated at each step_cfl over every active AMR
  /// level, followed by one global all_reduce_max, so an extremum that exists only on a fine patch
  /// still enforces dt <= cfl / max(mu) on the macro-step. Empty program -> ignored (no bound). Form
  /// validation (opcodes / register bounds) and STRICT role resolution match add_coupled_source.
  void add_coupled_frequency_expr(const std::string& label,
                                  const std::vector<std::string>& in_blocks,
                                  const std::vector<std::string>& in_roles,
                                  const std::vector<double>& consts,
                                  const std::vector<int>& freq_prog_ops,
                                  const std::vector<int>& freq_prog_args) {
    if (freq_prog_ops.empty() && freq_prog_args.empty())
      return;  // no per-cell frequency
    const int n_in = static_cast<int>(in_blocks.size());
    const int n_const = static_cast<int>(consts.size());
    if (static_cast<int>(in_roles.size()) != n_in)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_frequency_expr : in_blocks / in_roles of different sizes");
    if (n_in + n_const > kCsMaxReg)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_frequency_expr : too many registers (inputs + constants > " +
          std::to_string(kCsMaxReg) + ")");
    if (freq_prog_ops.size() != freq_prog_args.size())
      throw std::runtime_error(
          "AmrRuntime::add_coupled_frequency_expr : freq_prog_ops / freq_prog_args of different "
          "sizes");
    if (static_cast<int>(freq_prog_ops.size()) > kCsMaxProg)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_frequency_expr : frequency program too long (> " +
          std::to_string(kCsMaxProg) + ")");
    // Resolves (block, role) -> (block index, component), STRICT (mirror of add_coupled_source).
    std::vector<CsRef> ins(static_cast<std::size_t>(n_in));
    for (int c = 0; c < n_in; ++c) {
      const std::string& block = in_blocks[static_cast<std::size_t>(c)];
      const std::string& role = in_roles[static_cast<std::size_t>(c)];
      const int b = block_index(block);
      if (b < 0)
        throw std::runtime_error("AmrRuntime::add_coupled_frequency_expr : no block named '" +
                                 block + "'");
      // Role addressed BY NAME: a canonical role name OR a user-defined role label (ADC-292), STRICT.
      const VariableSet& vs = blocks_[static_cast<std::size_t>(b)].cons_vars;
      const int comp = vs.index_of(role);
      if (comp < 0)
        throw std::runtime_error("AmrRuntime::add_coupled_frequency_expr : block '" + block +
                                 "' does not expose role '" + role + "' (roles: " +
                                 (vs.roles.empty() ? std::string("<none>") : roles_csv(vs)) +
                                 ", no silent fallback to component 0)");
      ins[static_cast<std::size_t>(c)] = {b, comp, CsProgram{}};
    }
    CsProgram pg;
    pg.len = static_cast<int>(freq_prog_ops.size());
    for (int k = 0; k < pg.len; ++k) {
      const int opc = freq_prog_ops[static_cast<std::size_t>(k)];
      const int a = freq_prog_args[static_cast<std::size_t>(k)];
      if (opc < 0 || opc > static_cast<int>(CsOp::Sqrt))
        throw std::runtime_error(
            "AmrRuntime::add_coupled_frequency_expr : invalid opcode in the frequency");
      if (opc == static_cast<int>(CsOp::PushReg) && (a < 0 || a >= n_in + n_const))
        throw std::runtime_error(
            "AmrRuntime::add_coupled_frequency_expr : register out of bounds in the frequency");
      pg.op[k] = opc;
      pg.arg[k] = a;
    }
    validate_cs_program_stack(pg, "AmrRuntime::add_coupled_frequency_expr");
    std::vector<Real> kconsts(consts.begin(), consts.end());
    coupled_freq_exprs_.push_back(
        CoupledFreqExprDecl{label, std::move(ins), pg, n_in, n_const, std::move(kconsts)});
  }

  /// ACTIVE bound of the last step_cfl ("transport:<block>" / "source_frequency:<block>" /
  /// "stability_dt:<block>" / "global:<label>" / "degenerate" / "" before the first step).
  const std::string& last_dt_bound() const { return last_dt_reason_; }
  void override_last_dt_bound(std::string reason) { last_dt_reason_ = std::move(reason); }

  /// NEWTON REPORT (OPT-IN IMEX diagnostics) of block @p name, AGGREGATED over the levels and substeps
  /// of its LAST advance (cf. AmrRuntimeBlock::newton_report). AMR counterpart of System::newton_report.
  /// @throws std::runtime_error if the block is unknown, or if it was not added with
  ///         newton_diagnostics=true / newton_fail_policy warn|throw (no silently empty report).
  const NewtonReport& newton_report(const std::string& name) const {
    const int b = block_index(name);
    if (b < 0)
      throw std::runtime_error("AmrRuntime::newton_report : no block named '" + name + "'");
    const AmrRuntimeBlock& blk = blocks_[static_cast<std::size_t>(b)];
    if (!blk.newton_diagnostics || !blk.newton_report)
      throw std::runtime_error(
          "AmrRuntime::newton_report : Newton diagnostics not enabled for block '" + name +
          "' ; add the block with newton_diagnostics=True "
          "(pops.IMEX(newton_diagnostics=True)) or newton_fail_policy='warn'/'throw'");
    return *blk.newton_report;
  }

  /// Coarse potential (component 0 of the shared aux) as a ny*nx row-major field. Solves the fields if
  /// needed (counterpart of AmrSystem::potential), then reads aux(0). Identical for all blocks.
  std::vector<double> potential() {
    require_solved_field_report(solve_fields(), "AmrRuntime::potential");
    return coarse_aux_component(0);
  }

  /// Max SYSTEM wave speed (max over the blocks) on the current coarse. Requires the aux up to date.
  Real max_speed() {
    require_solved_field_report(solve_fields(), "AmrRuntime::max_speed");
    Real w = kAmrDriftSpeedFloor;
    for (auto& b : blocks_) {
      const Real wb = b.max_speed((*b.levels)[0].U, aux_[0]);
      if (wb > w)
        w = wb;
    }
    return w;
  }

  int n_patches() const {
    int count = 0;
    for (std::size_t k = 1; k < hierarchy_.ba.size(); ++k)
      count += hierarchy_.ba[k].size();
    return count;
  }

  // Index-space signatures of the fine patches (level + inclusive lo/hi corners), for ALL fine levels.
  // Read-only of the GLOBAL BoxArray (all boxes/all ranks) already stored -> rank-independent, zero
  // communication, NO hot-path cost (query between steps). Mirror of n_patches(): the runtime-owned
  // hierarchy BoxArray that gives the COUNT also gives the BOXES. Loop k = 1..nlev-1.
  std::vector<PatchBox> patch_boxes() const {
    std::vector<PatchBox> out;
    for (int k = 1; k < hierarchy_.nlev(); ++k) {
      const auto& bxs = hierarchy_.ba[static_cast<std::size_t>(k)].boxes();
      for (const Box2D& b : bxs)
        out.push_back(PatchBox{k, b.lo[0], b.lo[1], b.hi[0], b.hi[1]});
    }
    return out;
  }

  /// Every level's GLOBAL BoxArray in exact native order. Scientific-output geometry consumes this
  /// rather than patch_boxes(): level zero may be a distributed multi-box layout, and sorting boxes
  /// would invalidate OutputPiece.global_box_index.
  std::vector<PatchBox> output_geometry_boxes() const {
    std::vector<PatchBox> out;
    for (int k = 0; k < hierarchy_.nlev(); ++k) {
      const auto& boxes = hierarchy_.ba[static_cast<std::size_t>(k)].boxes();
      for (const Box2D& box : boxes)
        out.push_back(PatchBox{k, box.lo[0], box.lo[1], box.hi[0], box.hi[1]});
    }
    return out;
  }

  // COARSE-level (base) box counts (ADC-319, MPI ownership diagnostic). The runtime hierarchy owns
  // the base BoxArray + DistributionMapping common to all blocks. local_size() = base boxes OWNED by
  // this rank; box_array().size() = total base
  // boxes (all ranks). Mirror of n_patches(): a query between steps, no communication, no hot-path cost.
  int coarse_local_boxes() const { return aux_[0].local_size(); }
  int coarse_total_boxes() const { return hierarchy_.ba[0].size(); }

  // ----------------------------------------------------------------------------------------------
  // MULTI-BLOCK AMR CHECKPOINT / RESTART (ADC-509). PER-BLOCK PER-LEVEL state accessors + the
  // level-0 phi (multigrid warm-start), counterpart of AmrCouplerMP::level_state on the SHARED
  // hierarchy. The shared layout is FROZEN at build (make_shared_amr_layout: a deterministic central
  // fine patch, regrid_every==0): replaying the SAME composition reproduces the SAME hierarchy, so a
  // restart only needs to restore each block's valid cells + phi (no set_hierarchy on the runtime).
  // The _global variants gather ownership-distributed levels with all_reduce_sum so every rank owns
  // the complete field, MIRROR of System::state_global / gather_global.  A replicated level 0 is
  // already complete on every rank and must not be reduced (which would multiply it by n_ranks).
  // @p b: block index, @p k: level (0 = coarse, >= 1 = fine).
  // ----------------------------------------------------------------------------------------------

  // Conserved components of block @p b (Model::n_vars, carried by the AmrRuntimeBlock).
  int block_n_vars(std::size_t b) const {
    if (b >= blocks_.size())
      throw std::runtime_error("AmrRuntime::block_n_vars : block index out of bounds");
    return blocks_[b].ncomp;
  }

  // FULL conservative state (all components) of block @p b at level @p k, flat component-major
  // c*cells + (j-jlo)*nx + (i-ilo); zeros outside the patches at the fine level. LOCAL fabs only
  // (no gather): the facade calls this mono-rank. Mirror of AmrCouplerMP::level_state.
  std::vector<double> block_level_state(std::size_t b, int k) const {
    if (b >= blocks_.size())
      throw std::runtime_error("AmrRuntime::block_level_state : block index out of bounds");
    const std::vector<AmrLevelMP>& L = *blocks_[b].levels;
    if (k < 0 || k >= static_cast<int>(L.size()))
      throw std::runtime_error("AmrRuntime::block_level_state : level out of bounds");
    const MultiFab& U = L[k].U;
    const int nc = U.ncomp();
    const Box2D level_domain = amr_level_index_domain(dom_, k);
    const std::size_t cells = static_cast<std::size_t>(level_domain.nx()) * level_domain.ny();
    std::vector<double> out(static_cast<std::size_t>(nc) * cells, 0.0);
    device_fence();
    fill_level_state(U, nc, level_domain, out);
    return out;
  }

  // Same as block_level_state but gather ownership-distributed contributions so every rank holds the
  // complete field.  Replicated level 0 is returned directly.  COLLECTIVE only when a reduction is
  // required; all ranks must nevertheless make the same call for a given hierarchy.
  std::vector<double> block_level_state_global(std::size_t b, int k) const {
    std::vector<double> out = block_level_state(b, k);
    if (k > 0 || !replicated_coarse_)
      all_reduce_sum_inplace(out.data(), out.size());
    return out;
  }

  /// Exact rank-local valid-cell pieces for one block and level.  The explicit replicated bit on
  /// level zero lets the output planner distinguish rank-local from globally unique ownership.
  std::vector<OutputPiece> output_block_state_local_pieces(std::size_t b, int k) const {
    if (b >= blocks_.size())
      throw std::runtime_error(
          "AmrRuntime::output_block_state_local_pieces : block index out of bounds");
    const std::vector<AmrLevelMP>& levels = *blocks_[b].levels;
    if (k < 0 || k >= static_cast<int>(levels.size()))
      throw std::runtime_error("AmrRuntime::output_block_state_local_pieces : level out of bounds");
    return output_local_pieces(levels[static_cast<std::size_t>(k)].U, k,
                               k == 0 && replicated_coarse_);
  }

  /// Exact rank-local valid-cell pieces for a qualified elliptic provider output.
  std::vector<OutputPiece> output_field_local_pieces(const std::string& provider_slot, int k) {
    if (k < 0 || k >= provider_potential_levels(provider_slot))
      throw std::out_of_range("AmrRuntime::output_field_local_pieces level out of range");
    MultiFab& values = provider_potential_level(provider_slot, k);
    return output_local_pieces(values, k, k == 0 && replicated_coarse_);
  }

  // Restores block @p b at level @p k from @p s (same layout as block_level_state). Writes ONLY the
  // VALID cells of the local fabs (the ghosts are redone at the next solve_fields/step, like after a
  // regrid). NO re-prolongation: restored AS-IS. Mirror of AmrCouplerMP::set_level_state.
  void set_block_level_state(std::size_t b, int k, const std::vector<double>& s) {
    if (b >= blocks_.size())
      throw std::runtime_error("AmrRuntime::set_block_level_state : block index out of bounds");
    std::vector<AmrLevelMP>& L = *blocks_[b].levels;
    if (k < 0 || k >= static_cast<int>(L.size()))
      throw std::runtime_error("AmrRuntime::set_block_level_state : level out of bounds");
    MultiFab& U = L[k].U;
    const int nc = U.ncomp();
    const Box2D level_domain = amr_level_index_domain(dom_, k);
    const std::size_t nx = static_cast<std::size_t>(level_domain.nx());
    const std::size_t cells = nx * static_cast<std::size_t>(level_domain.ny());
    if (s.size() != static_cast<std::size_t>(nc) * cells)
      throw std::runtime_error(
          "AmrRuntime::set_block_level_state : state size differs from ncomp*level_cells");
    device_fence();
    for (int li = 0; li < U.local_size(); ++li) {
      Array4 u = U.fab(li).array();
      const Box2D v = U.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          for (int c = 0; c < nc; ++c)
            u(i, j, c) = s[static_cast<std::size_t>(c) * cells +
                           static_cast<std::size_t>(j - level_domain.lo[1]) * nx +
                           static_cast<std::size_t>(i - level_domain.lo[0])];
    }
  }

  // Potential phi of level @p k, flat level-domain row-major, zeros outside patches. Level 0: the multigrid
  // WARM-START mg_.phi() (the state reused by the next solve -> bit-identical restart). Level >= 1:
  // shared aux comp 0 (recomputed at solve_fields). Mirror of AmrCouplerMP::level_potential; the phi
  // is SHARED by all blocks (single aux), so it carries no block index. NON-const like
  // AmrRuntime::potential() (GeometricMG::phi() returns a mutable warm-start reference).
  std::vector<double> level_potential(int k) {
    if (k < 0 || k >= nlev_)
      throw std::runtime_error("AmrRuntime::level_potential : level out of bounds");
    const Box2D level_domain = amr_level_index_domain(dom_, k);
    std::vector<double> out(static_cast<std::size_t>(level_domain.nx()) * level_domain.ny(), 0.0);
    device_fence();
    const MultiFab& P = (k == 0) ? default_field_solver_->phi_level(0) : aux_[k];
    fill_level_phi(P, level_domain, out);
    return out;
  }

  // Same as level_potential, gathering only ownership-distributed levels.  Replicated level 0 is
  // already global and is returned directly.
  std::vector<double> level_potential_global(int k) {
    std::vector<double> out = level_potential(k);
    if (k > 0 || !replicated_coarse_)
      all_reduce_sum_inplace(out.data(), out.size());
    return out;
  }

  // Restores phi of level @p k. Level 0: warm-start mg_.phi() -> bit-identical restart (1st
  // post-restart solve starts from the same guess). Level >= 1: shared aux comp 0 (idempotent,
  // recomputed at solve_fields). Mirror of AmrCouplerMP::set_level_potential.
  void set_level_potential(int k, const std::vector<double>& p) {
    if (k < 0 || k >= nlev_)
      throw std::runtime_error("AmrRuntime::set_level_potential : level out of bounds");
    const Box2D level_domain = amr_level_index_domain(dom_, k);
    const std::size_t nx = static_cast<std::size_t>(level_domain.nx());
    if (p.size() != nx * static_cast<std::size_t>(level_domain.ny()))
      throw std::runtime_error(
          "AmrRuntime::set_level_potential : phi size differs from level cell count");
    device_fence();
    MultiFab& P = (k == 0) ? default_field_solver_->phi_level(0) : aux_[k];
    for (int li = 0; li < P.local_size(); ++li) {
      Array4 q = P.fab(li).array();
      const Box2D v = P.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          q(i, j, 0) = p[static_cast<std::size_t>(j - level_domain.lo[1]) * nx +
                         static_cast<std::size_t>(i - level_domain.lo[0])];
    }
  }

 private:
  void refresh_active_temporal_plan_() {
    const std::size_t active = static_cast<std::size_t>(nlev_ - 1);
    if (configured_temporal_relations_.size() < active)
      throw std::runtime_error(
          "AMR hierarchy activation lacks a resolved parent/child temporal relation");
    temporal_relations_.assign(
        configured_temporal_relations_.begin(),
        configured_temporal_relations_.begin() + static_cast<std::ptrdiff_t>(active));
    temporal_execution_plan_ =
        detail::PreparedAmrTemporalPlan::prepare(temporal_relations_, nlev_);
  }

  bool has_explicit_temporal_relations_() const { return !temporal_relations_.empty(); }

  /// Completes every check that could reject the explicit native route before solve/regrid/state
  /// mutation.  A block built by an older low-level consumer can still use the separate legacy route
  /// when no relations are installed, but it cannot silently ignore an installed relation chain.
  void preflight_native_temporal_step_() const {
    if (nlev_ > 1)
      require_coarse_fine_reconstruction_contract_();
    if (!has_explicit_temporal_relations_())
      return;
    if (!temporal_execution_plan_ || temporal_execution_plan_->nlevels() != nlev_)
      throw std::runtime_error(
          "AMR explicit temporal relations lack their prepared execution plan");
    for (const auto& block : blocks_) {
      const bool prepared = block.imex ? static_cast<bool>(block.imex_advance_with_temporal_plan)
                                       : static_cast<bool>(block.advance_with_temporal_plan);
      if (!prepared)
        throw std::runtime_error("AMR block '" + block.name +
                                 "' cannot execute its installed explicit temporal relations");
    }
  }

  /// Product parent_dt/child_dt from level zero to @p level.  With an explicit clock chain this is
  /// independent from spatial refinement; the spatial product is retained solely for the separate
  /// compatibility route that has no authored temporal relations.
  Real temporal_refinement_product_(int level) const {
    Real product = Real(1);
    for (int transition = 0; transition < level; ++transition) {
      if (has_explicit_temporal_relations_())
        product *= static_cast<Real>(
            temporal_relations_[static_cast<std::size_t>(transition)].temporal_ratio().value());
      else
        product *=
            static_cast<Real>(hierarchy_.refinement_ratios[static_cast<std::size_t>(transition)]);
    }
    return product;
  }

  // ADC-631 multistep history rings: each owner-qualified name authenticates its block carrier,
  // [slot][level] values, depth, per-level stored-once flag and per-slot dt. Data-only (MockImpl-
  // safe); all logic in detail::AmrHistoryOps (amr_history.hpp), a friend taking the engine by ref.
  friend struct detail::AmrHistoryOps;
  std::map<std::string, std::vector<std::vector<MultiFab>>> hist_rings_;
  std::map<std::string, int> hist_depth_;
  std::map<std::string, std::size_t> hist_block_owner_;
  std::map<std::string, std::vector<char>> hist_init_;
  std::map<std::string, std::vector<Real>> hist_slot_dt_;
  // Regrid / rebuild_hierarchy ring remap hook (member so the INLINE regrid() can call it before
  // detail::AmrHistoryOps is complete); body in amr_history.hpp forwards to AmrHistoryOps::remap_rings.
  void remap_history_rings_(const BoxArray& fb, const DistributionMapping& dmap, int fk, int pk,
                            bool prolong);
  void remove_history_levels_above_(int parent_level);
  void resize_history_levels_for_restore_(int target_levels);

  // Fills @p out from LOCAL valid cells at component-major, level-domain-relative indices. Shared
  // by block_level_state and its _global gather
  // variant (the loop is verbatim with AmrCouplerMP::level_state -> bit-identical layout).
  static void fill_level_state(const MultiFab& U, int nc, const Box2D& level_domain,
                               std::vector<double>& out) {
    const std::size_t nx = static_cast<std::size_t>(level_domain.nx());
    const std::size_t cells = nx * static_cast<std::size_t>(level_domain.ny());
    for (int li = 0; li < U.local_size(); ++li) {
      const ConstArray4 u = U.fab(li).const_array();
      const Box2D v = U.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          for (int c = 0; c < nc; ++c)
            out[static_cast<std::size_t>(c) * cells +
                static_cast<std::size_t>(j - level_domain.lo[1]) * nx +
                static_cast<std::size_t>(i - level_domain.lo[0])] = u(i, j, c);
    }
  }

  // Fills @p out from LOCAL valid cells of @p P (comp 0) at level-domain-relative row-major
  // indices. Shared by level_potential and its _global gather variant.
  static void fill_level_phi(const MultiFab& P, const Box2D& level_domain,
                             std::vector<double>& out) {
    const std::size_t nx = static_cast<std::size_t>(level_domain.nx());
    for (int li = 0; li < P.local_size(); ++li) {
      const ConstArray4 p = P.fab(li).const_array();
      const Box2D v = P.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          out[static_cast<std::size_t>(j - level_domain.lo[1]) * nx +
              static_cast<std::size_t>(i - level_domain.lo[0])] = p(i, j, 0);
    }
  }

  // Composite-reduction + hierarchy-rebuild helpers (ADC-542); bodies in amr_restore.hpp (this header
  // is at its line budget). block_index_by_name_ resolves a named block.
  std::size_t block_index_by_name_(const std::string& name) const;

  // Re-applies externally supplied static aux fields onto the COARSE shared aux valid cells. Mirror of
  // SystemFieldSolver::apply_named_aux_one (cartesian System): per LOCAL fab (MPI-safe), valid cells
  // only, global flat index j*nx+i. The coarse layout is frozen across regrid (only fine levels are
  // rebuilt), so the stored coarse field stays valid; solve_fields runs the coarse->fine injection
  // right after, carrying the selected comps to every level. No-op without a supplied field.
  void apply_static_aux() {
    if (static_aux_.empty() || aux_.empty())
      return;
    const int row = dom_.nx();
    // StaticAuxField is backed by fab_allocator (Kokkos SharedSpace on native builds). Host writes
    // completed at the setter; mark device residency before exposing its pointer to kernels.
    sync_device();
    for (const auto& [comp, field] : static_aux_) {
      if (field.empty() || comp >= aux_ncomp_)
        continue;
      for (int li = 0; li < aux_[0].local_size(); ++li) {
        Array4 a = aux_[0].fab(li).array();
        const Box2D v = aux_[0].box(li);
        for_each_cell(v, detail::AmrNamedAuxCopyKernel{a, field.data(), comp, row, dom_.lo[0],
                                                       dom_.lo[1]});
      }
    }
  }

  // NAMED multi-elliptic field (ADC-428): aux outputs plus one type-erased prepared backend.
  struct NamedField {
    struct PreparedProvider {
      std::size_t block = 0;
      Real coefficient = Real(1);
      std::function<void(const MultiFab&, MultiFab&)> rhs;
    };
    int phi_comp = -1;
    int gx_comp = -1;
    int gy_comp = -1;
    int gradient_sign = 0;
    bool has_plan = false;
    AmrFieldSolveConfig plan{};
    std::vector<PreparedProvider> prepared_providers;
    std::unique_ptr<AmrPreparedFieldSolver> solver;
    FieldNullspacePlan nullspace;
    std::vector<FieldNullspacePlan> level_nullspace;
    std::unique_ptr<FieldNullspaceWorkspace> nullspace_workspace;
    std::vector<std::unique_ptr<FieldNullspaceWorkspace>> level_nullspace_workspaces;
    std::vector<const MultiFab*> nullspace_rhs_levels;
    std::vector<MultiFab*> nullspace_phi_levels;
    std::vector<MultiFab> rhs_contribution_scratch;
    std::uint64_t rhs_scratch_generation = 0;
    bool nullspace_ready = false;
  };

  enum class NamedFieldSnapshotScope { kNone, kSelected, kAll };

  struct FieldSolveSnapshot {
    struct NamedFieldState {
      enum class Storage { kUnallocated, kLevelLocal, kComposite };
      Storage storage = Storage::kUnallocated;
      std::vector<MultiFab> phi;
      std::vector<MultiFab> rhs;
      FieldNullspacePlan nullspace;
      std::vector<FieldNullspacePlan> level_nullspace;
      bool nullspace_ready = false;
    };

    bool has_default = false;
    MultiFab default_phi;
    MultiFab default_rhs;
    std::vector<int> aux_components;
    std::vector<MultiFab> packed_aux;
    std::map<std::string, NamedFieldState> named;
    std::uint64_t topology_generation = 0;
    bool scope_default_field = false;
    NamedFieldSnapshotScope scope_named_fields = NamedFieldSnapshotScope::kNone;
    std::string scope_selected_named_field;
  };

  struct FieldSolveScope {
    bool default_field = false;
    NamedFieldSnapshotScope named_fields = NamedFieldSnapshotScope::kNone;
    const std::string* selected_named_field = nullptr;
  };

  std::vector<int> default_aux_components() const;
  std::vector<int> named_aux_components(const std::string* selected) const;
  std::vector<int> field_solve_aux_components(const FieldSolveScope& scope) const;
  std::vector<MultiFab> allocate_aux_component_carriers_(
      const std::vector<int>& components) const;
  void copy_aux_components_to_(std::vector<MultiFab>& packed,
                               const std::vector<int>& components) const;
  void unpack_aux_components(const std::vector<MultiFab>& packed,
                             const std::vector<int>& components) noexcept;
  AuxPublicationWorkspace& acquire_aux_publication_workspace_(
      const std::vector<int>& components, bool refined_values);
  void publish_aux_components(const std::vector<int>& components);
  void publish_refined_aux_components(const std::vector<int>& components);
  FieldSolveSnapshot& capture_field_solve_snapshot(const FieldSolveScope& scope);
  void restore_field_solve_snapshot(FieldSolveSnapshot& snapshot) noexcept;
  void release_field_solve_snapshot_() noexcept { field_solve_transaction_active_ = false; }

  template <class Solve>
  SolveReport run_field_solve_transaction(const FieldSolveScope& scope, Solve&& solve);

  static void require_solved_field_report(const SolveReport& report, const char* where) {
    if (!report.solved())
      throw std::runtime_error(std::string(where) + ": field solve failed: status=" +
                               report.status_name() + " action=" + report.action_name());
  }

  static void invalidate_named_field_solver(NamedField& field) noexcept {
    field.nullspace_workspace.reset();
    field.level_nullspace_workspaces.clear();
    field.nullspace_rhs_levels.clear();
    field.nullspace_phi_levels.clear();
    field.rhs_contribution_scratch.clear();
    field.rhs_scratch_generation = 0;
    field.solver.reset();
    field.nullspace = {};
    field.level_nullspace.clear();
    field.nullspace_ready = false;
  }

  void invalidate_named_field_topology() {
    for (auto& [_, field] : named_fields_)
      invalidate_named_field_solver(field);
  }

  void prepare_named_field_providers(NamedField& field) {
    if (!field.prepared_providers.empty())
      return;
    std::vector<NamedField::PreparedProvider> prepared;
    prepared.reserve(field.plan.providers.size());
    for (const auto& binding : field.plan.providers) {
      const int block = block_index(binding.owner_block);
      if (block < 0)
        throw std::runtime_error("AmrRuntime: field provider names unknown owner block '" +
                                 binding.owner_block + "'");
      auto& owner = blocks_[static_cast<std::size_t>(block)];
      auto found = owner.named_elliptic_rhs.find(binding.native_key);
      if (found == owner.named_elliptic_rhs.end() || !found->second)
        throw std::runtime_error("AmrRuntime: authenticated field provider has no RHS closure");
      prepared.push_back({static_cast<std::size_t>(block), binding.coefficient, found->second});
    }
    field.prepared_providers = std::move(prepared);
  }

  void prepare_named_rhs_scratch_(NamedField& field) {
    if (!field.solver)
      throw std::logic_error("named-field RHS scratch requires a materialized solver");
    const bool required = std::any_of(
        field.prepared_providers.begin(), field.prepared_providers.end(),
        [](const NamedField::PreparedProvider& provider) {
          return provider.coefficient != Real(1);
        });
    if (!required) {
      field.rhs_contribution_scratch.clear();
      field.rhs_scratch_generation = topology_materialization_generation_;
      return;
    }
    bool compatible =
        field.rhs_scratch_generation == topology_materialization_generation_ &&
        field.rhs_contribution_scratch.size() ==
            static_cast<std::size_t>(field.solver->level_count());
    if (compatible) {
      for (int level = 0; level < field.solver->level_count(); ++level) {
        const MultiFab& rhs = field.solver->rhs_level(level);
        const MultiFab& scratch =
            field.rhs_contribution_scratch[static_cast<std::size_t>(level)];
        if (!same_exact_multifab_layout_(scratch, rhs)) {
          compatible = false;
          break;
        }
      }
    }
    if (compatible)
      return;
    std::vector<MultiFab> candidate;
    candidate.reserve(static_cast<std::size_t>(field.solver->level_count()));
    for (int level = 0; level < field.solver->level_count(); ++level) {
      const MultiFab& rhs = field.solver->rhs_level(level);
      candidate.emplace_back(rhs.box_array(), rhs.dmap(), rhs.ncomp(), rhs.n_grow());
    }
    field.rhs_contribution_scratch = std::move(candidate);
    field.rhs_scratch_generation = topology_materialization_generation_;
  }

  // Materializes one resolved named-field provider lazily.  Provider declaration, exact request,
  // construction failure and post-build storage are communicator-wide contracts; no rank may escape
  // around a collective because its local extension failed first.
  void ensure_named_elliptic(NamedField& nf) {
    if (nf.solver)
      return;
    const BCRec boundary = nf.plan.has_explicit_bc ? nf.plan.explicit_bc : bcPhi_;
    const AmrFieldSolverBuildRequest request{
        geom_,        hierarchy_,         boundary,
        wall_active_, replicated_coarse_, "pops.amr.field-solver-use.named@1",
        nf.plan};
    std::shared_ptr<const AmrFieldSolverProvider> provider;
    bool declaration_failed = false;
    std::string provider_declaration;
    std::string expected_contract;
    try {
      provider = field_solver_registry_->resolve(nf.plan.solver);
      provider_declaration = exact_amr_field_solver_provider_declaration(*provider);
    } catch (...) {
      declaration_failed = true;
    }
    if (all_reduce_max(declaration_failed ? 1L : 0L) != 0)
      throw std::runtime_error(
          "AmrRuntime: field solver provider declaration failed on at least one rank");
    (void)inspect_amr_field_solver_support_collectively(*provider, request);
    try {
      expected_contract = provider->expected_prepared_contract(request);
    } catch (...) {
      declaration_failed = true;
    }
    if (all_reduce_max(declaration_failed || expected_contract.empty() ? 1L : 0L) != 0)
      throw std::runtime_error(
          "AmrRuntime: field solver provider failed to publish an exact hierarchy contract");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"amr-field-provider", provider_declaration},
             {"amr-field-expected-contract", expected_contract}}))
      throw std::runtime_error(
          "AmrRuntime: field solver provider declaration differs across MPI ranks");

    std::unique_ptr<AmrPreparedFieldSolver> prepared;
    bool build_failed = false;
    try {
      prepared = provider->build(request);
    } catch (...) {
      build_failed = true;
    }
    if (all_reduce_max(build_failed || !prepared ? 1L : 0L) != 0)
      throw std::runtime_error(
          "AmrRuntime: field solver provider construction failed on at least one rank");

    bool inspection_failed = false;
    bool materialization_mismatch = false;
    std::string actual_contract;
    try {
      actual_contract = prepared->exact_prepared_contract();
      materialization_mismatch = prepared->provider_identity() != provider->identity() ||
                                 actual_contract != expected_contract ||
                                 prepared->level_count() != nlev_;
      for (int level = 0; level < prepared->level_count(); ++level) {
        const auto index = static_cast<std::size_t>(level);
        MultiFab& rhs = prepared->rhs_level(level);
        MultiFab& phi = prepared->phi_level(level);
        const FieldDistribution distribution = level == 0 && replicated_coarse_
                                                   ? FieldDistribution::Replicated
                                                   : FieldDistribution::Distributed;
        materialization_mismatch = materialization_mismatch ||
                                   rhs.box_array().boxes() != hierarchy_.ba[index].boxes() ||
                                   phi.box_array().boxes() != hierarchy_.ba[index].boxes() ||
                                   rhs.dmap().ranks() != hierarchy_.dm[index].ranks() ||
                                   phi.dmap().ranks() != hierarchy_.dm[index].ranks() ||
                                   rhs.ncomp() != 1 || phi.ncomp() != 1 || rhs.n_grow() != 0 ||
                                   phi.n_grow() != 1 || rhs.shares_storage_with(phi) ||
                                   !detail::field_distribution_layout_matches(rhs, distribution) ||
                                   !detail::field_distribution_layout_matches(phi, distribution);
      }
    } catch (...) {
      inspection_failed = true;
    }
    if (all_reduce_max(inspection_failed ? 1L : 0L) != 0)
      throw std::runtime_error(
          "AmrRuntime: field solver provider inspection failed on at least one rank");
    if (all_reduce_max(materialization_mismatch ? 1L : 0L) != 0)
      throw std::runtime_error(
          "AmrRuntime: field solver provider did not materialize the exact hierarchy contract");
    if (!all_ranks_agree_exact_ordered_byte_pairs({{"amr-field-actual-contract", actual_contract}}))
      throw std::runtime_error("AmrRuntime: field solver materialization differs across MPI ranks");
    nf.solver = std::move(prepared);
  }

  std::shared_ptr<const MultiFab> composite_valid_mask(AmrPreparedFieldSolver& solver,
                                                       int level) const {
    const MultiFab& layout = solver.phi_level(level);
    auto mask =
        std::make_shared<MultiFab>(layout.box_array(), layout.dmap(), /*ncomp=*/1, /*ngrow=*/0);
    mask->set_val(Real(1));
    if (level + 1 >= solver.level_count())
      return mask;
    const BoxArray& fine_boxes = solver.phi_level(level + 1).box_array();
    std::vector<Box2D> coarse_coverage;
    coarse_coverage.reserve(static_cast<std::size_t>(fine_boxes.size()));
    for (const Box2D& fine : fine_boxes.boxes())
      coarse_coverage.push_back(fine.coarsen(kAmrRefRatio));
    // Coverage is a box-topology operation, not a cell-search problem.  Launch exactly on each
    // coarse/fine-footprint intersection: O(local coarse boxes * fine boxes) host intersections and
    // O(covered cells) device writes, never O(all cells * fine boxes) host work.
    for (int li = 0; li < mask->local_size(); ++li) {
      Array4 active = mask->fab(li).array();
      const Box2D valid = mask->box(li);
      for (const Box2D& covered : coarse_coverage) {
        const Box2D intersection = valid.intersect(covered);
        if (!intersection.empty())
          for_each_cell(intersection, detail::SetFieldCoverageKernel{active, Real(0)});
      }
    }
    return mask;
  }

  PreparedFieldNullspace prepare_field_nullspace(const FieldNullspaceProviderSelection& selection,
                                                 FieldNullspaceProviderRequest request) const {
    return prepare_field_nullspace_collectively(*nullspace_provider_registry_, selection,
                                                std::move(request));
  }

  void prepare_named_nullspace(NamedField& nf) {
    if (nf.nullspace_ready)
      return;
    nf.nullspace = {};
    nf.level_nullspace.clear();
    nf.nullspace_workspace.reset();
    nf.level_nullspace_workspaces.clear();
    nf.nullspace_rhs_levels.clear();
    nf.nullspace_phi_levels.clear();
    const BCRec boundary = nf.plan.has_explicit_bc ? nf.plan.explicit_bc : bcPhi_;
    const FieldNullspaceOperatorFacts operator_facts = field_nullspace_operator_facts_from_bc_rec(
        boundary, nf.plan.has_reaction, static_cast<bool>(wall_active_));
    if (nf.solver->couples_hierarchy_levels()) {
      FieldNullspaceProviderRequest request;
      request.plan_identity = nf.plan.provider_identity + ":topology-nullspace";
      request.operator_facts = operator_facts;
      request.topology.identity = nf.plan.topology_digest +
                                  ":composite-layout:" + std::to_string(nf.solver->level_count()) +
                                  ":epoch:" + std::to_string(topology_epoch_);
      request.topology.exact_layout_contract = std::string(nf.solver->exact_prepared_contract());
      request.topology.field_component = 0;
      if (!wall_active_)
        request.topology.connected_component_contract =
            request.topology.identity + ":connected-component@1";
      for (int k = 0; k < nf.solver->level_count(); ++k) {
        request.topology.layouts.push_back(&nf.solver->phi_level(k));
        request.topology.coverage.push_back(composite_valid_mask(*nf.solver, k));
        ExactContractBuilder coverage_contract;
        coverage_contract.text("pops.amr.composite-valid-coverage")
            .scalar(std::uint32_t{1})
            .text(nf.plan.topology_digest)
            .bytes(request.topology.exact_layout_contract)
            .scalar(static_cast<std::uint64_t>(topology_epoch_))
            .scalar(static_cast<std::int64_t>(k));
        request.topology.coverage_contracts.push_back(std::move(coverage_contract).release());
        const Geometry g = level_geom(k);
        request.topology.cell_measure.push_back(g.dx() * g.dy());
        request.topology.level_distributions.emplace_back(nf.solver->level_distribution(k));
      }
      nf.nullspace = prepare_field_nullspace(nf.plan.nullspace, std::move(request)).plan;
      std::vector<const MultiFab*> layouts;
      std::vector<PreparedVectorDistribution> distributions;
      layouts.reserve(static_cast<std::size_t>(nf.solver->level_count()));
      distributions.reserve(static_cast<std::size_t>(nf.solver->level_count()));
      nf.nullspace_rhs_levels.reserve(static_cast<std::size_t>(nf.solver->level_count()));
      nf.nullspace_phi_levels.reserve(static_cast<std::size_t>(nf.solver->level_count()));
      for (int k = 0; k < nf.solver->level_count(); ++k) {
        layouts.push_back(&nf.solver->phi_level(k));
        distributions.emplace_back(nf.solver->level_distribution(k));
        nf.nullspace_rhs_levels.push_back(&nf.solver->rhs_level(k));
        nf.nullspace_phi_levels.push_back(&nf.solver->phi_level(k));
      }
      nf.nullspace_workspace = std::make_unique<FieldNullspaceWorkspace>(
          nf.nullspace, std::move(layouts), std::move(distributions));
    } else {
      nf.level_nullspace.reserve(static_cast<std::size_t>(nf.solver->level_count()));
      nf.level_nullspace_workspaces.reserve(static_cast<std::size_t>(nf.solver->level_count()));
      for (int k = 0; k < nf.solver->level_count(); ++k) {
        FieldNullspaceProviderRequest request;
        request.plan_identity =
            nf.plan.provider_identity + ":topology-nullspace:level:" + std::to_string(k);
        request.operator_facts = operator_facts;
        request.topology.identity = nf.plan.topology_digest + ":level:" + std::to_string(k) +
                                    ":epoch:" + std::to_string(topology_epoch_);
        request.topology.exact_layout_contract = std::string(nf.solver->exact_prepared_contract());
        request.topology.first_level = k;
        request.topology.field_component = 0;
        if (!wall_active_)
          request.topology.connected_component_contract =
              request.topology.identity + ":connected-component@1";
        request.topology.layouts = {&nf.solver->phi_level(k)};
        const Geometry g = level_geom(k);
        request.topology.cell_measure = {g.dx() * g.dy()};
        request.topology.level_distributions.emplace_back(nf.solver->level_distribution(k));
        nf.level_nullspace.push_back(
            prepare_field_nullspace(nf.plan.nullspace, std::move(request)).plan);
        nf.level_nullspace_workspaces.push_back(std::make_unique<FieldNullspaceWorkspace>(
            nf.level_nullspace.back(), std::vector<const MultiFab*>{&nf.solver->phi_level(k)},
            std::vector<PreparedVectorDistribution>{
                PreparedVectorDistribution(nf.solver->level_distribution(k))},
            k));
      }
    }
    nf.nullspace_ready = true;
  }

  // Reads aux component @p comp of the COARSE level as a GLOBAL row-major field over the exact
  // rectangular logical domain (diagnostic / read-back). Same marshaling as
  // detail::coupler_read_coarse_phi (the default potential read-back), but for an arbitrary
  // component: local rectangular buffer, all_reduce_sum_inplace when the coarse is DISTRIBUTED
  // (disjoint boxes -> exact recompose; serial / replicated is identity). Used by named_field_values.
  std::vector<double> coarse_aux_component(int comp) const {
    device_fence();
    const int nx = dom_.nx(), ny = dom_.ny();
    std::vector<double> out(static_cast<std::size_t>(nx) * ny, 0.0);
    for (int li = 0; li < aux_[0].local_size(); ++li) {
      const ConstArray4 a = aux_[0].fab(li).const_array();
      const Box2D v = aux_[0].box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          out[static_cast<std::size_t>(j - dom_.lo[1]) * nx +
              static_cast<std::size_t>(i - dom_.lo[0])] = static_cast<double>(a(i, j, comp));
    }
    if (!replicated_coarse_)
      all_reduce_sum_inplace(out.data(), out.size());
    return out;
  }

  // Per-field aux HALO override (ADC-369), AFTER the shared fill_ghosts. Overrides only each
  // declared component's physical-face ghosts (aux_halo_override keeps periodic faces periodic).
  // No-op without a policy. Mirror of SystemFieldSolver::apply_named_aux_bc.
  void apply_named_aux_bc(MultiFab& packed, const std::vector<int>& components,
                          const Box2D& level_domain, const BCRec& level_bc);

  // Index of the block named @p name in the registry (-1 if absent). Counterpart of
  // AmrSystem::Impl::block_index (the facade names the blocks; the coupled sources target them by name,
  // resolved once at registration).
  int block_index(const std::string& name) const {
    for (std::size_t i = 0; i < blocks_.size(); ++i)
      if (blocks_[i].name == name)
        return static_cast<int>(i);
    return -1;
  }

  // Resolved reference of a coupled-source field: (block index, component) + the term bytecode program
  // (empty for an input). Inputs carry only block/comp; outputs carry in addition the postfix program
  // evaluated per cell. We capture the block INDEX (not a fab pointer): the Array4 are rebuilt at each
  // application, per level.
  struct CsRef {
    int block;
    int comp;
    CsProgram prog;  // outputs: term program; inputs: unused (CsProgram{})
  };
  // A registered coupled source: its inputs, its output terms and its constants, ready to be marshaled
  // into a CoupledSourceKernel per level / per fab at application.
  struct CoupledSourceSpec {
    std::vector<CsRef> ins;
    std::vector<CsRef> outs;
    std::vector<Real> kconsts;
    int n_in = 0;
    int n_const = 0;
    int n_terms = 0;
  };

  Geometry geom_;
  Box2D dom_;
  Periodicity base_per_;
  BCRec bcPhi_, aux_bc_;
  bool replicated_coarse_;
  AmrHierarchyLayout hierarchy_;
  // The resolved hierarchy describes a capacity. hierarchy_ contains only the currently active
  // contiguous prefix; these immutable transition contracts allow later activation and exact restart
  // without preallocating fake empty levels.
  std::vector<int> maximum_refinement_ratios_;
  std::vector<::pops::amr::ParentChildClockRelation> configured_temporal_relations_;
  std::vector<::pops::amr::ParentChildClockRelation> temporal_relations_;
  std::optional<detail::PreparedAmrTemporalPlan> temporal_execution_plan_;
  ActiveRegionProvider2D wall_active_;
  std::shared_ptr<const AmrFieldSolverProviderRegistry> field_solver_registry_;
  std::shared_ptr<const FieldNullspaceProviderRegistry> nullspace_provider_registry_;
  std::unique_ptr<AmrPreparedFieldSolver> default_field_solver_;
  FieldNullspacePlan default_field_nullspace_;
  std::unique_ptr<FieldNullspaceWorkspace> default_field_nullspace_workspace_;
  std::vector<AmrRuntimeBlock> blocks_;
  struct BoundaryStageStateView {
    runtime::multiblock::BoundaryEvaluationPoint point;
    const std::vector<MultiFab*>* states = nullptr;
    int single_block = -1;
    MultiFab* single_state = nullptr;

    MultiFab* state(std::size_t block) const {
      if (states != nullptr)
        return block < states->size() ? (*states)[block] : nullptr;
      return single_block >= 0 && block == static_cast<std::size_t>(single_block) ? single_state
                                                                                  : nullptr;
    }
  };
  std::optional<BoundaryStageStateView> boundary_stage_states_;
  runtime::multiblock::InterfaceFluxScheduler interface_scheduler_;
  // GLOBAL step bounds (add_dt_bound, parity with System) + ACTIVE bound of the last step_cfl.
  struct GlobalDtBound {
    std::string label;
    std::function<double()> fn;
  };
  std::vector<GlobalDtBound> dt_bounds_;
  // Declared frequencies of the coupled sources (bound dt <= cfl/mu on the macro-step, wave 3).
  struct CoupledFreqDecl {
    std::string label;
    Real mu;
  };
  std::vector<CoupledFreqDecl> coupled_freqs_;
  // PER-CELL frequencies of the coupled sources (CoupledSource.frequency with an Expr): bytecode
  // program mu(U) evaluated over all active levels at each step_cfl (one global MAX reduction).
  // ins = (block, comp) of the inputs (prog unused); kconsts = constants (same as the source).
  struct CoupledFreqExprDecl {
    std::string label;
    std::vector<CsRef> ins;
    CsProgram prog;
    int n_in = 0;
    int n_const = 0;
    std::vector<Real> kconsts;
  };
  std::vector<CoupledFreqExprDecl> coupled_freq_exprs_;
  std::string last_dt_reason_;
  std::vector<MultiFab> aux_;  // [level], shared by all blocks
  std::map<std::string, BootstrapStaggeredField> bootstrap_staggered_fields_;
  std::map<std::string, BootstrapCacheState> bootstrap_caches_;
  StepSnapshot bootstrap_snapshot_;
  StepSnapshot regrid_snapshot_;
  bool bootstrap_pending_ = false;
  int step_rollback_scope_depth_ = 0;
  // Externally supplied static aux fields: canonical B_z and model-named components -> coarse
  // base-level field (ny*nx row-major). Re-applied by solve_fields each macro-step and after hierarchy
  // growth so they persist across regrid. Empty by default -> bit-identical.
  std::map<int, StaticAuxField> static_aux_;
  // Per-field aux HALO policy (ADC-369): component -> uniform boundary policy, applied to the coarse aux
  // after the shared fill (apply_named_aux_bc). Empty by default -> bit-identical.
  std::map<int, AuxHaloPolicy> named_aux_bc_;
  // NAMED multi-elliptic fields (ADC-428): field name -> aux outputs + prepared provider instance.
  std::map<std::string, NamedField> named_fields_;
  std::vector<CoupledSourceSpec>
      coupled_sources_;  // registered coupled sources (applied after transport)
  // TYPED coupling operator inspect metadata (ADC-595, parity with System::Impl::coupled_operators_):
  // one read-only view (label + declared contracts) per registered coupled source, in registration
  // order. Populated by add_coupled_source (unchecked) / add_coupling_operator (declared). Metadata
  // only; the step never reads it.
  std::vector<CouplingOperatorView> coupled_operators_;
  // UNION-TAGS REGRID (capstone Phase 2, C.6). Runtime criteria exist only as one authenticated
  // PreparedTaggingProgram. Host std::function predicates are intentionally absent from this engine.
  TaggingProgram tagging_program_;
  runtime::amr::PreparedTaggingExecutionPlan tagging_execution_plan_;
  std::vector<BlockTransferAuthority> block_transfer_authorities_;
  std::vector<std::vector<TemporalParentWorkspace>> temporal_parent_workspaces_;
  std::vector<AuxPublicationWorkspace> aux_publication_workspaces_;
  std::vector<FieldSolveSnapshot> field_solve_rollback_workspaces_;
  bool field_solve_transaction_active_ = false;
  int regrid_every_ = 0;
  int regrid_grow_ = 2;
  int regrid_margin_ = 2;
  ClusterParams
      cluster_{};  ///< ADC-616: Berger-Rigoutsos params; default {0.7,1,32} (bit-identical).
  std::shared_ptr<runtime::amr::PreparedTaggerComponent> external_tagger_;
  std::shared_ptr<runtime::amr::PreparedClusteringComponent> external_clustering_;
  std::shared_ptr<const amr::ClusteringProvider> clustering_provider_ =
      std::make_shared<const amr::BergerRigoutsosProvider>(ClusterParams{});
  std::int64_t component_tick_ = 0;
  double component_physical_time_ = 0.0;
  int aux_ncomp_ = kAuxBaseComps;
  int nlev_ = 0;
  int macro_step_ = 0;
  mutable int solve_count_ = 0;
  int regrid_count_ = 0;
  // Monotone identity of the materialized hierarchy.  Every successful topology replacement bumps
  // it; accepted-state rollback restores it.  Nullspace recipes include the epoch so no basis or
  // coverage mask can silently survive a regrid/restart hierarchy rebuild.
  std::uint64_t topology_epoch_ = 0;
  // Runtime-only invalidation key for persistent resources tied to concrete MultiFab layouts and
  // addresses. It intentionally does not belong to StepSnapshot/checkpoint state.
  std::uint64_t topology_materialization_generation_ = 1;

  static bool same_exact_multifab_layout_(const MultiFab& left, const MultiFab& right) {
    return left.box_array().boxes() == right.box_array().boxes() &&
           left.dmap().ranks() == right.dmap().ranks() && left.ncomp() == right.ncomp() &&
           left.n_grow() == right.n_grow();
  }

  [[nodiscard]] Periodicity aux_periodicity_() const noexcept {
    return Periodicity{
        aux_bc_.xlo == BCType::Periodic && aux_bc_.xhi == BCType::Periodic,
        aux_bc_.ylo == BCType::Periodic && aux_bc_.yhi == BCType::Periodic};
  }

  [[nodiscard]] std::uint64_t next_topology_materialization_generation_() const noexcept {
    std::uint64_t next = topology_materialization_generation_ + 1;
    if (next == 0)
      ++next;  // reserve zero; tolerate the theoretical wrap
    return next;
  }

  std::vector<std::vector<TemporalParentWorkspace>> make_temporal_parent_workspaces_(
      std::uint64_t generation) const {
    std::vector<std::vector<TemporalParentWorkspace>> candidate(blocks_.size());
    for (std::size_t block = 0; block < blocks_.size(); ++block) {
      candidate[block].reserve(nlev_ > 0 ? static_cast<std::size_t>(nlev_ - 1) : 0u);
      for (int parent_level = 0; parent_level + 1 < nlev_; ++parent_level) {
        const MultiFab& prototype =
            (*blocks_[block].levels)[static_cast<std::size_t>(parent_level)].U;
        candidate[block].push_back(TemporalParentWorkspace{
            generation, block, parent_level,
            MultiFab(prototype.box_array(), prototype.dmap(), prototype.ncomp(),
                     prototype.n_grow())});
      }
    }
    return candidate;
  }

  AuxPublicationWorkspace make_aux_publication_workspace_(
      const std::vector<int>& components, bool refined_values,
      std::uint64_t generation) const {
    if (components.empty() ||
        !std::is_sorted(components.begin(), components.end()) ||
        std::adjacent_find(components.begin(), components.end()) != components.end() ||
        components.front() < 0 || components.back() >= aux_ncomp_)
      throw std::invalid_argument(
          "AMR aux publication requires a sorted unique in-range component set");
    AuxPublicationWorkspace workspace;
    workspace.topology_generation = generation;
    workspace.refined_values = refined_values;
    workspace.components = components;
    workspace.packed = allocate_aux_component_carriers_(components);
    // Preparation warms transfer schedules before any publication. Newly materialized aux levels
    // may not yet have a solved value, so seed the private carriers instead of reading uninitialized
    // hierarchy storage merely to prepare communication metadata.
    for (MultiFab& carrier : workspace.packed)
      carrier.set_val(Real(0));
    workspace.coarse_transfers.reserve(nlev_ > 0 ? static_cast<std::size_t>(nlev_ - 1) : 0u);
    for (int level = 1; level < nlev_; ++level) {
      const bool replicated_parent = level == 1 && replicated_coarse_;
      const CommunicatorView communicator =
          replicated_parent ? CommunicatorView{} : world_communicator_view();
      workspace.coarse_transfers.push_back(
          detail::PreparedConservativeLinearTransferWorkspace::prepare(
              workspace.packed[static_cast<std::size_t>(level - 1)],
              workspace.packed[static_cast<std::size_t>(level)],
              amr_level_index_domain(dom_, level - 1), amr_level_index_domain(dom_, level),
              replicated_parent,
              refined_values ? detail::ConservativeCellFillRegion::Ghost
                             : detail::ConservativeCellFillRegion::ValidAndGhost,
              aux_periodicity_(), generation, communicator));
    }
    return workspace;
  }

  std::vector<AuxPublicationWorkspace> make_aux_publication_workspaces_(
      std::uint64_t generation) const {
    struct Key {
      bool refined = false;
      std::vector<int> components;
    };
    std::vector<Key> keys;
    const auto add = [&](bool refined, std::vector<int> components) {
      if (!components.empty())
        keys.push_back(Key{refined, std::move(components)});
    };
    for (const AuxPublicationWorkspace& workspace : aux_publication_workspaces_)
      add(workspace.refined_values, workspace.components);
    add(false, default_aux_components());
    if (!named_fields_.empty()) {
      const bool refined = nlev_ > 1;
      add(refined, named_aux_components(nullptr));
      for (const auto& [name, _] : named_fields_)
        add(refined, named_aux_components(&name));
    }
    std::sort(keys.begin(), keys.end(), [](const Key& left, const Key& right) {
      if (left.refined != right.refined)
        return left.refined < right.refined;
      return left.components < right.components;
    });
    keys.erase(std::unique(keys.begin(), keys.end(), [](const Key& left, const Key& right) {
                 return left.refined == right.refined && left.components == right.components;
               }),
               keys.end());
    std::vector<AuxPublicationWorkspace> candidate;
    candidate.reserve(keys.size());
    for (const Key& key : keys)
      candidate.push_back(
          make_aux_publication_workspace_(key.components, key.refined, generation));
    return candidate;
  }

  runtime::amr::PreparedTaggingExecutionPlan make_tagging_execution_plan_(
      const TaggingProgram& program, std::uint64_t generation) {
    if (!program.prepared)
      return {};
    const bool uses_shared_aux =
        std::any_of(program.leaves.begin(), program.leaves.end(), [this](const auto& leaf) {
          return leaf.state_index == blocks_.size();
        });
    std::vector<std::vector<runtime::amr::PreparedTaggingField>> fields_by_level(
        static_cast<std::size_t>(nlev_));
    std::vector<Box2D> domains;
    domains.reserve(static_cast<std::size_t>(nlev_));
    for (int level = 0; level < nlev_; ++level) {
      auto& fields = fields_by_level[static_cast<std::size_t>(level)];
      fields.reserve(blocks_.size() + (uses_shared_aux ? 1u : 0u));
      for (AmrRuntimeBlock& block : blocks_)
        fields.push_back(runtime::amr::PreparedTaggingField{
            block.state_identity, &(*block.levels)[static_cast<std::size_t>(level)].U});
      if (uses_shared_aux)
        fields.push_back(runtime::amr::PreparedTaggingField{
            "pops://runtime/amr/shared-aux", &aux_.at(static_cast<std::size_t>(level))});
      domains.push_back(dom_.refine(level_refinement(level)));
    }
    return runtime::amr::PreparedTaggingExecutionPlan::prepare(program, fields_by_level, domains,
                                                                generation);
  }

  void rematerialize_persistent_topology_resources_(std::uint64_t generation) {
    if (field_solve_transaction_active_)
      throw std::logic_error(
          "AMR topology cannot change during a field-solve transaction");
    auto temporal_candidate = make_temporal_parent_workspaces_(generation);
    auto aux_candidate = make_aux_publication_workspaces_(generation);
    std::vector<std::optional<PreparedAmrFillPatchPlan>> fill_patch_candidate;
    std::vector<std::vector<detail::PreparedConservativeCellTransferWorkspace>>
        coarse_fine_spatial_candidate;
    std::vector<PreparedAmrAverageDownPlan> average_down_candidate;
    std::vector<PreparedAmrAdvanceScratchPlan> advance_scratch_candidate;
    fill_patch_candidate.reserve(blocks_.size());
    coarse_fine_spatial_candidate.reserve(blocks_.size());
    average_down_candidate.reserve(blocks_.size());
    advance_scratch_candidate.reserve(blocks_.size());
    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
      const AmrRuntimeBlock& block = blocks_[block_index];
      const auto& authority = block_transfer_authorities_[block_index];
      if (authority.prepared && authority.coarse_fine.prepared_coarse_fine) {
        fill_patch_candidate.emplace_back(PreparedAmrFillPatchPlan::prepare(
            *block.levels, dom_, base_per_, replicated_coarse_, generation,
            authority.coarse_fine.prepared_coarse_fine));
        std::vector<detail::PreparedConservativeCellTransferWorkspace> spatial;
        spatial.reserve(block.levels->size() - 1);
        for (std::size_t child = 1; child < block.levels->size(); ++child) {
          const bool replicated_parent = child == 1 && replicated_coarse_;
          const CommunicatorView communicator =
              replicated_parent ? CommunicatorView{} : world_communicator_view();
          const MultiFab& parent = (*block.levels)[child - 1].U;
          const MultiFab& fine = (*block.levels)[child].U;
          const Box2D coarse_domain =
              amr_level_index_domain(dom_, static_cast<int>(child - 1));
          spatial.push_back(detail::PreparedConservativeCellTransferWorkspace::prepare(
              parent, fine, coarse_domain, coarse_domain.refine(authority.refinement_ratio),
              replicated_parent, detail::ConservativeCellFillRegion::Ghost, base_per_, generation,
              communicator, authority.coarse_fine.prepared_coarse_fine));
        }
        coarse_fine_spatial_candidate.push_back(std::move(spatial));
      } else {
        fill_patch_candidate.emplace_back(std::nullopt);
        coarse_fine_spatial_candidate.emplace_back();
      }
      average_down_candidate.push_back(
          PreparedAmrAverageDownPlan::prepare(*block.levels, generation));
      advance_scratch_candidate.push_back(PreparedAmrAdvanceScratchPlan::prepare(
          *block.levels, dom_, base_per_, replicated_coarse_, block.wave_speed_cache, generation,
          world_communicator_view()));
    }
    auto tagging_candidate = make_tagging_execution_plan_(tagging_program_, generation);
    temporal_parent_workspaces_.swap(temporal_candidate);
    aux_publication_workspaces_.swap(aux_candidate);
    tagging_execution_plan_ = std::move(tagging_candidate);
    for (std::size_t block = 0; block < blocks_.size(); ++block) {
      blocks_[block].fill_patch_plan = std::move(fill_patch_candidate[block]);
      blocks_[block].coarse_fine_spatial_workspaces =
          std::move(coarse_fine_spatial_candidate[block]);
      blocks_[block].average_down_plan = std::move(average_down_candidate[block]);
      blocks_[block].advance_scratch_plan = std::move(advance_scratch_candidate[block]);
    }
    field_solve_rollback_workspaces_.clear();
    topology_materialization_generation_ = generation;
  }

  void record_topology_replacement_() {
    rematerialize_persistent_topology_resources_(
        next_topology_materialization_generation_());
    ++topology_epoch_;
  }
  // AMR / MPI PROFILING (Spec 5 criterion 43, ADC-479): non-owning pointer to the AmrSystem-owned
  // Profiler (lifetime guaranteed by the facade). Null by default -> the engine never profiles
  // (zero overhead). Set via set_profiler after build (parity with System::profiler_).
  runtime::program::Profiler* profiler_ = nullptr;

  // RAII phase-timing scope for an AMR phase (regrid / average_down / fill_boundary). Mirrors
  // runtime::program::ProfileScope but over a NULLABLE profiler pointer: it reads the clock and
  // records only when profiler_ is non-null AND enabled. A null/disabled run constructs a cheap inert
  // scope (one pointer copy + one clock read) -- the granularity is per-phase, not per-cell, so this
  // is off the hot path. Returned BY VALUE from profile_amr_scope (movable: only POD members).
  class AmrPhaseScope {
   public:
    AmrPhaseScope(runtime::program::Profiler* prof, const char* name)
        : prof_(prof != nullptr && prof->enabled() ? prof : nullptr),
          name_(name),
          t0_(std::chrono::steady_clock::now()) {}
    AmrPhaseScope(AmrPhaseScope&& o) noexcept : prof_(o.prof_), name_(o.name_), t0_(o.t0_) {
      o.prof_ = nullptr;  // the moved-from scope must not record
    }
    AmrPhaseScope(const AmrPhaseScope&) = delete;
    AmrPhaseScope& operator=(const AmrPhaseScope&) = delete;
    AmrPhaseScope& operator=(AmrPhaseScope&&) = delete;
    ~AmrPhaseScope() {
      if (prof_ == nullptr)
        return;
      const auto t1 = std::chrono::steady_clock::now();
      try {
        prof_->record(name_, std::chrono::duration<double>(t1 - t0_).count());
      } catch (...) {  // NOLINT(bugprone-empty-catch) -- a profiler never throws out of a scope
      }
    }

   private:
    runtime::program::Profiler* prof_;
    const char* name_;
    std::chrono::steady_clock::time_point t0_;
  };

  // Build an AMR phase scope (no-op when profiling is off). Used at the head of regrid /
  // average_down / fill_boundary.
  AmrPhaseScope profile_amr_scope(const char* name) { return AmrPhaseScope(profiler_, name); }

  // fill_ghosts wrapped in the "fill_boundary" timing scope + per-call count (Spec 5 criterion 43).
  // Under MPI np>1 the ghost fill is a cross-rank halo exchange -> also count one "mpi_messages" (a
  // point-to-point round, distinct from the all_reduce collectives counted as "mpi_reductions").
  // Serial / single rank: no message count (the fill is a local copy). Off-profiling: inert.
  void fill_ghosts_profiled(MultiFab& mf, const Box2D& dom, const BCRec& bc) {
    auto _fb = profile_amr_scope("fill_boundary");
    if (profiler_ != nullptr) {
      profiler_->count("fill_boundary");
#ifdef POPS_HAS_MPI
      if (n_ranks() > 1)
        profiler_->count("mpi_messages");
#endif
    }
    fill_ghosts(mf, dom, bc);
  }
};

}  // namespace pops

// Out-of-line AmrRuntime member definitions kept OUT of this header (its line budget): the
// transactional field-publication helpers, composite-reduction folds, and checkpoint
// hierarchy-rebuild seam (ADC-542). Included last so the full AmrRuntime class is visible.
// ADC-631 multistep history rings (detail::AmrHistoryOps): store/register/read/rotate, per-level flat
// checkpoint accessors, the regrid remap hook and the native selective-persistence replay. Included
// BEFORE amr_restore.hpp so rebuild_hierarchy can call the ring realloc hook (AmrHistoryOps defined).
#include <pops/runtime/amr/amr_field_solve_transaction.hpp>  // NOLINT(build/include_order)
#include <pops/runtime/amr/amr_history.hpp>                  // NOLINT(build/include_order)
#include <pops/runtime/amr/amr_restore.hpp>                  // NOLINT(build/include_order)
// ADC-639 conservative-reflux capture: the flux-materialising per-level residual seams
// (level_rhs_capture_into / level_neg_div_flux_capture_into) + the strip samplers + route_reflux_program.
// Included last (the full AmrRuntime class + the reflux types from amr_reflux_mf.hpp are visible).
#include <pops/runtime/amr/amr_program_reflux.hpp>  // NOLINT(build/include_order)
