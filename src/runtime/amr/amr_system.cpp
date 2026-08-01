#include <pops/runtime/amr_system.hpp>

#include <pops/runtime/dynamic/abi_key.hpp>  // detail::abi_key_string: ABI key (header-only), compared to the loader's
#include <pops/runtime/config/route_ids.hpp>  // pops::verify_route_manifest (ADC-599: embedded route registry guard)
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>  // compiled/runtime AMR block builders
#include <pops/runtime/amr/amr_runtime.hpp>  // AmrRuntime + AmrRuntimeBlock (multi-block runtime engine)
#include <pops/runtime/amr/amr_tensor_elliptic.hpp>  // builtin hierarchy tensor provider factory
#include <pops/runtime/multiblock/prepared_interface_flux_component.hpp>
#include <pops/runtime/amr/bootstrap_transfer_registry.hpp>
#include <pops/runtime/amr/bootstrap_transfer_builtins.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/analytic/collective_preflight.hpp>
#include <pops/runtime/output_piece_collective.hpp>
#include <pops/runtime/program/profiler.hpp>  // Profiler: AMR / MPI phase timings (Spec 5 criterion 43, ADC-479)
#include <pops/runtime/program/program_runtime_state.hpp>  // ProgramRuntimeState: the shared compiled-Program subsystem (ADC-594)
#include <pops/runtime/program/amr_program_checkpoint.hpp>
#include <pops/runtime/program/external_riemann_brick.hpp>
#include <pops/runtime/program/step_transaction.hpp>  // StepAttemptRejected: atomic public AMR attempts
#include <pops/runtime/program/module_metadata.hpp>  // read_module_metadata / required_blocks / required_solver: install-time validation (ADC-508)
#include <pops/runtime/builders/block/amr_block_seam.hpp>  // ADC-335: per-transport AMR build seam (build_amr_block/_compiled_<transport>)
#include <pops/runtime/builders/factory/model_factory.hpp>  // detail::dispatch_model + compiled bricks
#include <pops/runtime/dynamic/model_registry.hpp>  // validate_transport: single-source transport rejection (ADC-331)
#include <pops/runtime/context/wall_predicate.hpp>  // detail::wall_predicate (wall shared System/AmrSystem)
#include <pops/numerics/fv/flux_failure.hpp>
#include <pops/numerics/time/integrators/implicit_stepper.hpp>  // NewtonOptions + validate_newton_options (shared range check)
#include <pops/core/state/aux_names.hpp>  // canonical B_z component shared with the device Aux layout

#include <algorithm>  // std::find
#include <array>      // std::array<int, 3>: named-elliptic-field aux components (ADC-428)
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>  // std::numeric_limits (global step bounds: neutralization to +inf before the min)
#include <pops/runtime/dynamic/dynlib.hpp>  // portable dlopen<->LoadLibraryW layer (ADC-99); <dlfcn.h> on POSIX
#include <functional>
#include <map>  // ell_field_rhs_ named-elliptic RHS closures (ADC-428) + per-PROGRAM-block RuntimeParams store and install_program defaults grouping (ADC-508)
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace pops {

namespace {
runtime::amr::TransferCentering bootstrap_centering(const std::string& value) {
  using C = runtime::amr::TransferCentering;
  if (value == "cell")
    return C::Cell;
  if (value == "face_x")
    return C::FaceX;
  if (value == "face_y")
    return C::FaceY;
  if (value == "node")
    return C::Node;
  throw std::runtime_error("unknown native AMR transfer centering '" + value + "'");
}

runtime::amr::TransferKernelRegistry bootstrap_transfer_kernels() {
  using namespace runtime::amr;
  TransferKernelRegistry registry;
  const auto exact = [](std::string space, std::string centering, std::string representation,
                        std::string operation, int order, std::vector<int> ghost) {
    return [=](const TransferRouteDescriptor& row) {
      return row.space == space && row.centering == centering &&
             row.representation == representation && row.storage == "dense" &&
             row.operation == operation && row.order == order && row.ghost_depth == ghost &&
             row.dimension == 2 && row.refinement_ratio == 2;
    };
  };
  registry.add({"pops.lib.amr.transfer::conservative_linear",
                exact("cell", "cell", "conservative", "prolongation", 2, {1}),
                [](const TransferRouteDescriptor&) { return prepare_conservative_linear(); }});
  registry.add({"pops.lib.amr.transfer::volume_average",
                exact("cell", "cell", "conservative", "restriction", 1, {0}),
                [](const TransferRouteDescriptor&) { return prepare_volume_average(); }});
  registry.add_exact(
      "pops.lib.amr.transfer::conservative_coarse_fine",
      TransferRouteDescriptor{
          "cell", "cell", "conservative", "dense", "coarse_fine_fill", 2, {2}, 2, 2},
      [](const TransferRouteDescriptor&) { return prepare_conservative_coarse_fine(); });
  registry.add_exact("pops.lib.amr.transfer::conservative_polynomial5_coarse_fine",
                     TransferRouteDescriptor{
                         "cell", "cell", "conservative", "dense", "coarse_fine_fill", 5, {3}, 2, 2},
                     [](const TransferRouteDescriptor&) {
                       return prepare_conservative_polynomial5_coarse_fine();
                     });
  const auto face_accepts = [](const TransferRouteDescriptor& row) {
    return row.space == "face" && (row.centering == "face_x" || row.centering == "face_y") &&
           row.representation == "conservative" && row.storage == "dense" &&
           row.operation == "prolongation" && row.order == 2 &&
           row.ghost_depth == std::vector<int>{1} && row.dimension == 2 &&
           row.refinement_ratio == 2;
  };
  registry.add(
      {"pops.lib.amr.transfer::face_divergence_preserving", face_accepts,
       [](const TransferRouteDescriptor& row) {
         PreparedTransferKernel kernel;
         (void)row;
         kernel.face_vector = [](const MultiFab& coarse_x, const MultiFab& coarse_y,
                                 MultiFab& fine_x, MultiFab& fine_y,
                                 const SpatialTransferContext& context) {
           // One prepared vector operation owns both oriented carriers.  The two normal-face
           // reconstructions are applied in the same transaction and preserve the discrete flux
           // balance as a pair; no scalar face route exists.
           detail::bootstrap_prolong_face_vector(coarse_x, coarse_y, fine_x, fine_y, context);
         };
         return kernel;
       }});
  registry.add({"pops.lib.amr.transfer::node_bilinear",
                exact("node", "node", "primitive", "prolongation", 2, {1}),
                [](const TransferRouteDescriptor&) {
                  PreparedTransferKernel kernel;
                  kernel.spatial = [](const MultiFab& coarse, MultiFab& fine,
                                      const SpatialTransferContext& context) {
                    detail::bootstrap_prolong_staggered(coarse, fine, TransferCentering::Node,
                                                        context);
                  };
                  return kernel;
                }});
  const auto temporal_accepts = [](const TransferRouteDescriptor& row) {
    return row.space != "field" && row.space != "cache" && row.storage == "dense" &&
           row.operation == "temporal_interpolation" && row.order == 2 &&
           row.ghost_depth == std::vector<int>{0} && row.dimension == 2 &&
           row.refinement_ratio == 2;
  };
  registry.add(
      {"pops.lib.amr.transfer::linear_time_interpolation", temporal_accepts,
       [](const TransferRouteDescriptor&) { return prepare_linear_time_interpolation(); }});
  registry.add(
      {"pops.lib.amr.materializer::elliptic_solve",
       exact("field", "cell", "primitive", "coarse_fine_fill", 1, {0}),
       [](const TransferRouteDescriptor&) {
         PreparedTransferKernel kernel;
         kernel.materialize = [](AmrRuntime& runtime, const MaterializationContext& context) {
           if (context.operation != "recompute")
             throw std::runtime_error("elliptic materializer received an incompatible operation");
           return runtime.recompute_bootstrap_field(context.target);
         };
         return kernel;
       }});
  registry.add(
      {"pops.lib.amr.materializer::patch_topology",
       exact("cache", "cell", "primitive", "coarse_fine_fill", 1, {0}),
       [](const TransferRouteDescriptor&) {
         PreparedTransferKernel kernel;
         kernel.materialize = [](AmrRuntime& runtime, const MaterializationContext& context) {
           if (context.operation == "invalidate_cache")
             return static_cast<std::int64_t>(runtime.invalidate_bootstrap_cache(context.subject));
           if (context.operation == "rebuild_cache" ||
               context.operation == "invalidate_then_rebuild") {
             if (context.operation == "invalidate_then_rebuild")
               runtime.invalidate_bootstrap_cache(context.subject);
             return static_cast<std::int64_t>(
                 runtime.rebuild_bootstrap_topology_cache(context.subject, context.level)
                     .topology.size());
           }
           throw std::runtime_error(
               "patch-topology materializer received an incompatible operation");
         };
         return kernel;
       }});
  return registry;
}

std::string amr_time_routes_csv() {
  return std::string(route_token(TimeRouteId::kExplicitSsprk2)) + "|" +
         route_token(TimeRouteId::kForwardEuler) + "|" + route_token(TimeRouteId::kSsprk3) + "|" +
         route_token(TimeRouteId::kImex);
}

std::string amr_program_method_label(const std::string& time) {
  if (time == route_token(TimeRouteId::kExplicitSsprk2))
    return "ssprk2";
  if (time == route_token(TimeRouteId::kForwardEuler) ||
      time == route_token(TimeRouteId::kSsprk3) || time == route_token(TimeRouteId::kImex))
    return time;
  throw std::runtime_error("unknown AMR time route '" + time + "'");
}

bool amr_newton_options_non_default(const NewtonOptions& newton) {
  return newton.max_iters != kNewtonDefaultMaxIters || newton.rel_tol != kNewtonDefaultRelTol ||
         newton.abs_tol != kNewtonDefaultAbsTol || newton.fd_eps != kNewtonDefaultFdEps ||
         newton.damping != kNewtonDefaultDamping;
}

std::string direct_amr_state_identity(const std::string& block_name) {
  // A typed Case installs the exact owner-qualified Handle identity before declaring any block.
  // The private native engine can also be exercised directly; in that mode the engine itself is
  // the storage owner, so qualify its state deterministically from the already-unique block name.
  // The length prefix keeps this mapping injective for arbitrary (including path-like) names.
  return "pops://runtime/amr-direct-state/" + std::to_string(block_name.size()) + ":" + block_name;
}
}  // namespace

struct AmrSystem::Impl {
  AmrSystemConfig cfg;
  std::shared_ptr<const PreparedLoadBalanceAuthority> load_balance_authority_;

  // Specification of ONE block (frozen at add_block, materialized at lazy build). Every registry
  // cardinality uses the same AmrRuntime engine, shared-hierarchy contract and prepared field
  // provider path.
  struct BlockSpec {
    std::string name;
    // Native ModelSpec path (composed bricks) OR compiled path (.so / add_compiled_model).
    bool is_compiled = false;
    ModelSpec spec;  // ModelSpec path (is_compiled == false)
    std::string limiter = "minmod", riemann = "rusanov";
    bool recon_prim = false;  // recon == "primitive"
    int substeps = 1;
    int stride = 1;  // authored Program hold/catch-up cadence
    double gamma = static_cast<double>(kPhysicalDefaultGamma);
    // Compiled production path: type-erasing builder that, on
    // the SHARED layout materialized at lazy build (build_multi), produces the AmrRuntimeBlock of the
    // compiled block -- exactly like dispatch_amr_block for a native block, but with the Model/Limiter/
    // Flux CONCRETE types already captured at add (add_compiled_model) instead of a ModelSpec dispatch.
    AmrCompiledBlockBuilder compiled_block_builder;
    // Initial density of the block (component 0), ny*nx row-major.
    bool has_density = false;
    std::vector<double> density;
    // FULL initial conservative state (all components), ncomp*ny*nx component-major; set by
    // set_conservative_state(name, U). Takes priority over density at seed in the compiled/native
    // runtime block builders.
    bool has_state = false;
    std::vector<double> state;
    // Canonical authoring token retained for Program normalization and introspection. Spatial block
    // construction does not decode it into a second native time method.
    std::string time = std::string(route_token(TimeRouteId::kForwardEuler));
    // Zhang-Shu positivity floor (ADC-259): if > 0, the AMR transport floors the Density-role face
    // states + C/F fine ghost means to >= pos_floor. 0 (default) = inactive, bit-identical. Threaded
    // to dispatch_amr_block through the unique runtime builder.
    // COMPILED blocks carry it too (ADC-322): set_compiled_block stores it here from the regenerated
    // .so loader (pops_install_native_amr -> add_compiled_model), so both routings floor like a native block.
    double pos_floor = 0.0;
    double weno_epsilon = static_cast<double>(kWenoEpsilon);
    bool wave_speed_cache = false;
  };

  // Declared before blocks so deferred builders/closures die before their DSO is unloaded.
  std::vector<std::shared_ptr<void>> external_riemann_libraries_;
  std::vector<BlockSpec> blocks;
  // ADC-672: executable per-block plans installed before lazy hierarchy construction.  Their
  // presence forces the common AmrRuntime route so the same plan is captured by every level RHS.
  std::map<std::string, std::shared_ptr<PreparedBoundaryPlan>> boundary_plans_;
  std::map<std::string, std::string> block_state_identities_;
  std::map<std::string, std::string> field_storage_routes_;
  std::shared_ptr<runtime::amr::PreparedTaggerComponent> amr_tagger_component_;
  std::shared_ptr<runtime::amr::PreparedClusteringComponent> amr_clustering_component_;
  struct BootstrapArray {
    std::string centering;
    int ncomp = 0;
    std::vector<double> initial_values;
  };
  std::unordered_map<std::string, BootstrapArray> bootstrap_arrays;
  std::unordered_map<std::string, std::vector<double>> bootstrap_analytic_constants;
  struct BootstrapGaussian {
    double center_x, center_y, background, amplitude, inverse_width;
  };
  std::unordered_map<std::string, BootstrapGaussian> bootstrap_analytic_gaussians;
  std::unordered_map<std::string, std::vector<analytic::AnalyticProgram>>
      bootstrap_analytic_expressions;
  runtime::amr::TransferRouteRegistry bootstrap_transfer_routes{bootstrap_transfer_kernels()};
  std::map<std::pair<std::string, std::string>, std::string> bootstrap_subject_routes;
  std::unordered_map<std::string, std::string> bootstrap_block_subjects;
  std::unordered_map<std::string, std::array<std::string, 2>> bootstrap_face_vectors;
  std::vector<std::uint8_t> program_accepted_state_;
  std::uint64_t program_accepted_state_revision_ = 0;
  std::vector<::pops::amr::ParentChildClockRelation> temporal_relations_;

  // Coupled inter-species sources (compiled pops.dsl.CoupledSource, flat P5 bytecode ABI) FROZEN at
  // add_coupled_source and injected into the AmrRuntime runtime engine at lazy build (build_multi).
  // The runtime does not yet exist at registration (built at ensure_built): so we store the flat
  // spec here, then replay it on the runtime right after its construction (multi-block only).
  struct CoupledSourceSpec {
    std::vector<std::string> in_blocks, in_roles;
    std::vector<double> consts;
    std::vector<std::string> out_blocks, out_roles;
    std::vector<int> prog_ops, prog_args, prog_lens;
    double frequency = 0.0;  // CONSTANT declared mu (bound dt <= cfl/mu; 0 = no bound)
    std::string label = "coupled_source";
    // Optional PER-CELL frequency mu(U): bytecode program (same inputs/constants/register table
    // as the source). EMPTY = constant frequency only. Replayed on the runtime at build.
    std::vector<int> freq_prog_ops, freq_prog_args;
  };
  std::vector<CoupledSourceSpec> coupled_sources;
  // TYPED coupling operator inspect metadata (ADC-595, parity with System::Impl::coupled_operators_):
  // one read-only view (label + declared contracts) per registered coupled source, in registration
  // order. Populated at add_coupled_source (unchecked) / add_coupling_operator (declared) so the facade
  // exposes the couplings as typed operators BEFORE the lazy multi-block runtime build. Metadata only.
  std::vector<CouplingOperatorView> coupled_operators;

  struct TaggingSpec {
    std::vector<std::string> leaf_subject_kinds;
    std::vector<std::string> leaf_subject_identities;
    std::vector<std::string> leaf_blocks;
    std::vector<std::string> leaf_variables;
    std::vector<int> leaf_field_component_indices;
    std::vector<int> leaf_ops;
    std::vector<double> leaf_thresholds;
    std::vector<int> leaf_stencil_indices;
    std::vector<runtime::amr::PreparedTaggingProgram::Stencil> stencils;
    std::vector<std::int32_t> refine_ops, refine_args;
    std::vector<std::int32_t> coarsen_ops, coarsen_args;
    int min_cycles = 0;
    int equality_policy = 0;
    int conflict_policy = 0;
    std::string clock_identity;
    std::string provider_identity;
  };
  std::unique_ptr<TaggingSpec> tagging_spec;

  std::vector<double> bz_field;  // coarse B_z(x,y), ny*nx row-major (set_magnetic_field)
  // Model-NAMED aux fields (ADC-291): component (>= kAuxNamedBase) -> coarse field (ny*nx row-major).
  // Pending until build, then pushed to the runtime. Empty -> bit-identical.
  // cf. set_aux_field_component.
  std::map<int, AmrRuntime::StaticAuxField> named_aux_;
  // Per-field aux HALO policies (ADC-369): component -> uniform policy. Pending until build, then seeded
  // into the runtime via set_named_aux_bc.
  std::map<int, AuxHaloPolicy> named_aux_bc_;
  // NAMED multi-elliptic fields (ADC-428): the native AMR loader declares them (register_elliptic_field)
  // and attaches each field's per-block RHS closure (set_block_elliptic_field) BEFORE the lazy build,
  // when the AmrRuntime engine does not yet exist. We stash both here and replay them on the runtime at
  // build_multi. ell_field_comps_: field -> {phi_comp, gx_comp, gy_comp, gradient_sign}.
  // ell_field_rhs_: field -> {block
  // name -> RHS closure}. Empty default -> bit-identical (no named field registered).
  std::map<std::pair<std::string, std::string>, std::array<int, 4>> ell_field_comps_;
  std::map<std::string, std::map<std::string, std::function<void(const MultiFab&, MultiFab&)>>>
      ell_field_rhs_;
  // Complete plans are keyed by the digest of the canonical block-qualified provider identity.
  // plan_identity independently commits the complete resolved semantics; provider_identity retains
  // the exact provider identity for collision/audit checks. Duplicate slots are always refused.
  std::map<std::string, AmrFieldSolveConfig> field_plans_;
  std::shared_ptr<AmrFieldSolverProviderRegistry> field_solver_registry_;
  std::shared_ptr<FieldNullspaceProviderRegistry> field_nullspace_provider_registry_;
  std::shared_ptr<runtime::program::HierarchyTensorSolverProviderRegistry>
      hierarchy_tensor_solver_provider_registry_;
  FieldNullspaceProviderSelection default_field_nullspace_ =
      operator_topology_zero_mean_nullspace();
  bool field_plan_consensus_verified_ = false;
  std::string p_rhs = "charge_density", p_solver = "geometric_mg", p_bc = "auto", p_wall = "none";
  AmrFieldSolverOptions p_solver_options;
  double p_wall_radius = 0.0;

  bool built = false;
  // RUNTIME FREEZE LIFECYCLE (ADC-592, parity System::Impl::bound_): false while assembling, true once
  // mark_bound() runs (the Python bind flow calls it LAST). 'bound' is DISTINCT from the lazy 'built'
  // materialization (bind runs BEFORE ensure_built): the structural guards refuse a call when EITHER
  // built (the historical lazy-phase message, unchanged) OR bound_ (the new bind-vocabulary message).
  // false for a direct engine script that never binds -> historical behavior unchanged.
  bool bound_ = false;
  // GLOBAL bounds (AmrSystem::add_dt_bound): registered BEFORE the lazy build, passed to the
  // runtime engine at its construction.
  struct GlobalDtBound {
    std::string label;
    std::function<double()> fn;
  };
  std::vector<GlobalDtBound> dt_bounds;
  std::string last_dt_reason;  // active bound of the last Program-owned CFL step
  // Unique AmrRuntime path (shared hierarchy + summed default field).
  std::shared_ptr<pops::AmrRuntime> runtime;
  double t = 0;
  // The one authoritative accepted macro-step counter (parity System::Impl::macro_step_). The
  // spatial AmrRuntime owns no accepted clock; Program cadence, restart and rollback all read this
  // value through the facade.
  int macro_step_ = 0;

  // COMPILED TIME-PROGRAM RUNTIME STATE (ADC-594): the AMR runtime embeds the SAME ProgramRuntimeState
  // struct System::Impl holds (include/pops/runtime/program/program_runtime_state.hpp) -- the SHARED,
  // non-diverging Program subsystem the issue mandates. AMR uses the COMMON subset (step_ / substeps_ /
  // stride_ / installed_hash_ / block_map_ / block_params_ / diagnostics_ / profiler_ / dt_bound_);
  // cache_ / hist_ stay EMPTY because their hierarchy-aware seams are not wired. AmrRuntime is wired
  // to &program_.profiler_ at build (parity with System::Impl); the Profiler's Impl
  // address stays stable (program_ is a stable Impl member). AmrSystem::step routes unconditionally
  // through run_program_cadence_ (reading program_.step_ / substeps_ / stride_).
  pops::runtime::program::ProgramRuntimeState program_;
  // ADC-635: the in-window regrid schedule the LAST rebuild_history_slots fired (for the
  // accepted-state reader's coherence assertion against the checkpoint fingerprint). Reset each
  // rebuild; empty on a clean window.
  std::vector<int> last_replay_regrid_steps_;

  explicit Impl(const AmrSystemConfig& c)
      : cfg(c),
        load_balance_authority_(
            std::make_shared<const PreparedLoadBalanceAuthority>(prepare_load_balance_authority(
                c.load_balance_route, c.load_balance_identity, c.load_balance_options))),
        field_solver_registry_(make_default_amr_field_solver_registry()),
        field_nullspace_provider_registry_(make_default_field_nullspace_provider_registry()),
        hierarchy_tensor_solver_provider_registry_(
            runtime::program::make_default_hierarchy_tensor_solver_provider_registry()) {
    p_solver_options = field_solver_registry_->resolve(p_solver)->default_field_options();
  }

  // SUBSTEPS/STRIDE cadence around the installed program closure (parity SystemProgramDriver::run_program_
  // cadence): runs the whole program ONCE over the representable accepted facade interval when the
  // stride window closes (the clock still ticks every macro-step), subdivided into equal-coordinate
  // Program calls. The left-folded accepted dt sum remains strict checkpoint provenance; it is not a
  // second endpoint authority. With 1/1 this is a single program_.step_(dt) call (bit-identical to a
  // bare install). The cadence applies to the whole resolved ProgramGraph.
  void run_program_cadence_(double dt) {
    const double accepted_time = t;
    const auto cadence = program_.prepare_cadence_step(accepted_time, macro_step_, dt, "AmrSystem");
    if (macro_step_ == std::numeric_limits<int>::max())
      throw std::overflow_error("AmrSystem Program cadence macro-step counter overflow");
    if (cadence.due) {
      program_.validate_cadence_partition(cadence, program_.substeps_, "AmrSystem");
      const int accepted_macro_step = macro_step_;
      const int held_before_due = cadence.window_steps - 1;
      if (accepted_macro_step < held_before_due)
        throw std::logic_error("AmrSystem Program cadence window starts before macro-step zero");
      const int window_start_macro_step = accepted_macro_step - held_before_due;
      try {
        for (int s = 0; s < program_.substeps_; ++s) {
          const auto partition =
              program_.prepare_cadence_substep(cadence, s, program_.substeps_, "AmrSystem");
          // AmrProgramContext reads the facade clock at Program entry. Move it to the exact accepted
          // start of this substep so stage/tagger coordinates cover the whole catch-up window instead
          // of repeating the outer macro-step time.
          t = partition.start;
          // All internal calls belong to one public stride window. Publish the accepted start tick
          // so schedules, regridding and AmrProgramContext never count Program substeps as facade
          // macro-steps.
          macro_step_ = window_start_macro_step;
          // ADC-626/ADC-631: expose this interval before the Program stores its pre-commit history
          // sample. The ring ledger then records the outgoing dt from that sample toward the next
          // accepted sample (variable-dt replay). Parity with SystemProgramDriver::run_program_cadence.
          program_.last_dt_ = static_cast<Real>(partition.dt);
          program_.step_(partition.dt);
          t = partition.end;
        }
      } catch (...) {
        t = accepted_time;
        macro_step_ = accepted_macro_step;
        throw;
      }
      t = accepted_time;
      macro_step_ = accepted_macro_step;
    }
    program_.commit_cadence_step(cadence, "AmrSystem");
    // One prepared endpoint owns facade, stages and serialized AMR accepted clocks. Do not recompute
    // it as either accepted_time + dt or window_start + effective_dt after Program execution.
    t = cadence.window_end;
  }

  struct AcceptedSnapshot {
    AmrRuntime::StepSnapshot runtime;
    double time = 0.0;
    int macro_step = 0;
    Real last_program_dt = Real(0);
    double cadence_window_dt = 0.0;
    int cadence_window_steps = 0;
    double cadence_window_start_time = 0.0;
    bool cadence_clock_restore_pending = false;
    double cadence_clock_restore_dt = 0.0;
    int cadence_clock_restore_steps = 0;
    double cadence_clock_restore_start_time = 0.0;
    double cadence_clock_restore_last_dt = 0.0;
    double cadence_clock_restore_accepted_time = 0.0;
    int cadence_clock_restore_macro_step = 0;
    std::map<std::string, Real> program_diagnostics;
    pops::runtime::program::CacheManager cache;
    pops::runtime::program::HistoryManager history;
    pops::runtime::program::Profiler profiler;
    std::string last_dt_reason;
    std::vector<int> replay_regrid_steps;
    std::vector<std::uint8_t> program_accepted_state;
    std::uint64_t program_accepted_state_revision = 0;

    template <class Map>
    static void copy_value_map_into(Map& destination, const Map& source) {
      for (auto entry = destination.begin(); entry != destination.end();) {
        if (source.find(entry->first) == source.end())
          entry = destination.erase(entry);
        else
          ++entry;
      }
      for (const auto& [key, value] : source)
        destination.insert_or_assign(key, value);
    }

    template <class T>
    static void copy_vector_into(std::vector<T>& destination, const std::vector<T>& source) {
      destination.resize(source.size());
      std::copy(source.begin(), source.end(), destination.begin());
    }

    void capture(Impl& impl) {
      if (!impl.runtime)
        throw std::runtime_error("AmrSystem snapshot requires a materialized AmrRuntime");
      impl.runtime->capture_step_snapshot(runtime);
      time = impl.t;
      macro_step = impl.macro_step_;
      last_program_dt = impl.program_.last_dt_;
      cadence_window_dt = impl.program_.cadence_window_dt_;
      cadence_window_steps = impl.program_.cadence_window_steps_;
      cadence_window_start_time = impl.program_.cadence_window_start_time_;
      cadence_clock_restore_pending = impl.program_.cadence_clock_restore_pending_;
      cadence_clock_restore_dt = impl.program_.cadence_clock_restore_dt_;
      cadence_clock_restore_steps = impl.program_.cadence_clock_restore_steps_;
      cadence_clock_restore_start_time = impl.program_.cadence_clock_restore_start_time_;
      cadence_clock_restore_last_dt = impl.program_.cadence_clock_restore_last_dt_;
      cadence_clock_restore_accepted_time = impl.program_.cadence_clock_restore_accepted_time_;
      cadence_clock_restore_macro_step = impl.program_.cadence_clock_restore_macro_step_;
      copy_value_map_into(program_diagnostics, impl.program_.diagnostics_);
      // AMR currently owns its native cache/history rings inside AmrRuntime.  These two shared
      // ProgramRuntimeState containers are therefore empty on the AMR path, but retain their value
      // contract so a future target can populate them without weakening rollback semantics.
      cache = impl.program_.cache_;
      history = impl.program_.hist_;
      profiler = impl.program_.profiler_;
      last_dt_reason = impl.last_dt_reason;
      copy_vector_into(replay_regrid_steps, impl.last_replay_regrid_steps_);
      copy_vector_into(program_accepted_state, impl.program_accepted_state_);
      program_accepted_state_revision = impl.program_accepted_state_revision_;
    }

    void restore(Impl& impl) const {
      if (!impl.runtime)
        throw std::runtime_error("AmrSystem snapshot lost its AmrRuntime");
      impl.runtime->restore_step_snapshot(runtime);
      impl.t = time;
      impl.macro_step_ = macro_step;
      impl.program_.last_dt_ = last_program_dt;
      impl.program_.cadence_window_dt_ = cadence_window_dt;
      impl.program_.cadence_window_steps_ = cadence_window_steps;
      impl.program_.cadence_window_start_time_ = cadence_window_start_time;
      impl.program_.cadence_clock_restore_pending_ = cadence_clock_restore_pending;
      impl.program_.cadence_clock_restore_dt_ = cadence_clock_restore_dt;
      impl.program_.cadence_clock_restore_steps_ = cadence_clock_restore_steps;
      impl.program_.cadence_clock_restore_start_time_ = cadence_clock_restore_start_time;
      impl.program_.cadence_clock_restore_last_dt_ = cadence_clock_restore_last_dt;
      impl.program_.cadence_clock_restore_accepted_time_ = cadence_clock_restore_accepted_time;
      impl.program_.cadence_clock_restore_macro_step_ = cadence_clock_restore_macro_step;
      copy_value_map_into(impl.program_.diagnostics_, program_diagnostics);
      impl.program_.cache_ = cache;
      impl.program_.hist_ = history;
      impl.program_.profiler_ = profiler;
      impl.last_dt_reason = last_dt_reason;
      copy_vector_into(impl.last_replay_regrid_steps_, replay_regrid_steps);
      copy_vector_into(impl.program_accepted_state_, program_accepted_state);
      impl.program_accepted_state_revision_ = program_accepted_state_revision;
    }
  };

  AcceptedSnapshot internal_step_snapshot_;
  AcceptedSnapshot external_step_snapshot_;
  AcceptedSnapshot restart_snapshot_;
  bool internal_step_transaction_active_ = false;
  bool internal_step_transaction_restored_ = false;
  bool external_step_transaction_active_ = false;
  bool external_step_transaction_restored_ = false;
  bool external_step_transaction_committed_ = false;
  bool restart_transaction_active_ = false;

  bool has_active_step_transaction_() const noexcept {
    return internal_step_transaction_active_ || external_step_transaction_active_;
  }

  void restore_active_step_transaction_() {
    if (external_step_transaction_active_) {
      if (!external_step_transaction_restored_) {
        external_step_snapshot_.restore(*this);
        external_step_transaction_restored_ = true;
      }
      return;
    }
    if (internal_step_transaction_active_) {
      if (!internal_step_transaction_restored_) {
        internal_step_snapshot_.restore(*this);
        internal_step_transaction_restored_ = true;
      }
      return;
    }
    throw std::logic_error("AmrSystem: no active step transaction to restore");
  }

  struct RuntimeRollbackScope {
    AmrRuntime* runtime = nullptr;

    explicit RuntimeRollbackScope(AmrRuntime& engine) : runtime(&engine) {
      runtime->begin_step_rollback_scope();
    }
    ~RuntimeRollbackScope() {
      if (runtime == nullptr)
        return;
      try {
        runtime->end_step_rollback_scope();
      } catch (...) {
        std::terminate();
      }
    }
    RuntimeRollbackScope(const RuntimeRollbackScope&) = delete;
    RuntimeRollbackScope& operator=(const RuntimeRollbackScope&) = delete;
  };

  /// Execute one public AMR macro-step against an accepted snapshot.  The AmrRuntime snapshot owns
  /// topology + multi-block/per-level data; this facade layer adds its authoritative clock and the
  /// Program-owned publications that live outside the engine.  Any exception rolls both layers back
  /// before propagating (RejectAttempt remains observable to the Python retry policy).
  template <class Body>
  decltype(auto) execute_step_transaction(Body&& body) {
    if (!runtime)
      throw std::logic_error("AmrSystem step transaction requires a materialized runtime");
    if (restart_transaction_active_)
      throw std::logic_error("AmrSystem cannot advance while a restart transaction is active");
    if (external_step_transaction_active_) {
      if (external_step_transaction_committed_)
        throw std::runtime_error(
            "AmrSystem: committed external step transaction must be finalized before another step");
      external_step_transaction_restored_ = false;
      RuntimeRollbackScope rollback_scope(*runtime);
      try {
        return std::forward<Body>(body)();
      } catch (const FluxEvaluationFailure& failure) {
        restore_active_step_transaction_();
        if (failure.action() == TransactionFailureAction::kRetryStep ||
            failure.action() == TransactionFailureAction::kRejectStep) {
          const auto disposition = failure.action() == TransactionFailureAction::kRetryStep
                                       ? runtime::program::StepAttemptDisposition::kRetry
                                       : runtime::program::StepAttemptDisposition::kReject;
          throw runtime::program::StepAttemptRejected(SolveStatus::kInvalidEvaluation, disposition,
                                                      failure.reason_code(), "stage",
                                                      failure.what());
        }
        throw;
      } catch (...) {
        restore_active_step_transaction_();
        throw;
      }
    }
    if (internal_step_transaction_active_)
      throw std::logic_error("AmrSystem step transaction is non-reentrant");
    internal_step_snapshot_.capture(*this);
    internal_step_transaction_active_ = true;
    internal_step_transaction_restored_ = false;
    struct InternalTransactionLease {
      Impl& impl;
      ~InternalTransactionLease() {
        impl.internal_step_transaction_active_ = false;
        impl.internal_step_transaction_restored_ = false;
      }
    } transaction_lease{*this};
    RuntimeRollbackScope rollback_scope(*runtime);
    try {
      return std::forward<Body>(body)();
    } catch (const FluxEvaluationFailure& failure) {
      restore_active_step_transaction_();
      if (failure.action() == TransactionFailureAction::kRetryStep ||
          failure.action() == TransactionFailureAction::kRejectStep) {
        const auto disposition = failure.action() == TransactionFailureAction::kRetryStep
                                     ? runtime::program::StepAttemptDisposition::kRetry
                                     : runtime::program::StepAttemptDisposition::kReject;
        throw runtime::program::StepAttemptRejected(SolveStatus::kInvalidEvaluation, disposition,
                                                    failure.reason_code(), "stage", failure.what());
      }
      throw;
    } catch (...) {
      restore_active_step_transaction_();
      throw;
    }
  }

  // Index of an exactly named block. Empty names never alias another runtime object.
  int block_index(const std::string& name) const {
    if (name.empty())
      return -1;
    for (std::size_t i = 0; i < blocks.size(); ++i)
      if (blocks[i].name == name)
        return static_cast<int>(i);
    return -1;
  }

  // Resolve @p name to its block index, raising if absent (ADC-509 multi-block checkpoint accessors;
  // mirror of the density(name) / mass(name) resolution). Returns a size_t for the runtime accessors.
  std::size_t block_index_or_throw(const std::string& name) const {
    const int idx = block_index(name);
    if (idx < 0)
      throw std::runtime_error("AmrSystem : no block named '" + name + "'");
    return static_cast<std::size_t>(idx);
  }

  std::size_t observable_block_index_or_throw(const std::string& name) const {
    if (name.empty() && blocks.size() == 1)
      return 0;
    return block_index_or_throw(name);
  }

  BCRec poisson_bc() {
    std::string mode = p_bc;
    if (mode == "auto") {
      BCRec automatic;
      if (!cfg.periodicity.x)
        automatic.xlo = automatic.xhi = BCType::Dirichlet;
      if (!cfg.periodicity.y)
        automatic.ylo = automatic.yhi = BCType::Dirichlet;
      return automatic;
    }
    BCRec b;
    if (mode == "periodic")
      return b;
    if (mode == "dirichlet") {
      b.xlo = b.xhi = b.ylo = b.yhi = BCType::Dirichlet;
      return b;
    }
    if (mode == "neumann") {
      b.xlo = b.xhi = b.ylo = b.yhi = BCType::Foextrap;
      return b;
    }
    throw std::runtime_error("AmrSystem::set_poisson : unknown bc '" + mode + "'");
  }
  ActiveRegionProvider2D wall_active() {
    return detail::wall_predicate(p_wall, p_wall_radius, cfg.L, cfg.Ly, "AmrSystem::set_poisson",
                                  cfg.xlo, cfg.ylo);
  }

  // Materializes only hierarchy/system-owned parameters.  Multi-block construction must never select
  // a representative block merely to obtain the common mesh, Poisson, and ownership configuration.
  AmrBuildParams make_layout_params() {
    AmrBuildParams bp;
    // MESH group: coarse geometry + ownership policy.
    bp.mesh.n = cfg.n;
    bp.mesh.ny = cfg.ny;
    bp.mesh.L = cfg.L;
    bp.mesh.Ly = cfg.Ly;
    bp.mesh.xlo = cfg.xlo;
    bp.mesh.ylo = cfg.ylo;
    bp.mesh.regrid_every = cfg.regrid_every;
    bp.mesh.periodicity = cfg.periodicity;
    bp.mesh.distribute_coarse = cfg.distribute_coarse;
    bp.mesh.coarse_max_grid = cfg.coarse_max_grid;
    bp.mesh.load_balance = load_balance_authority_;
    // POISSON group: coarse Poisson BC + conductive wall.
    bp.poisson.bc = poisson_bc();
    bp.poisson.wall = wall_active();
    return bp;
  }

  void require_field_plan_consensus() {
    if (field_plan_consensus_verified_)
      return;
    std::vector<std::pair<std::string_view, std::string_view>> identities;
    identities.reserve(field_plans_.size());
    std::vector<std::string> contracts;
    contracts.reserve(field_plans_.size());
    for (const auto& [slot, plan] : field_plans_)
      contracts.push_back(exact_amr_field_solve_config_contract(plan));
    std::size_t index = 0;
    for (const auto& [slot, plan] : field_plans_) {
      (void)plan;
      identities.emplace_back(slot, contracts[index++]);
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(identities))
      throw std::runtime_error("AmrSystem: ordered resolved field plans differ across MPI ranks");
    field_plan_consensus_verified_ = true;
  }

  void install_active_temporal_relations() {
    if (!runtime)
      throw std::runtime_error(
          "AmrSystem : cannot install temporal relations without an AMR runtime");
    const std::size_t active_transition_count = static_cast<std::size_t>(runtime->nlev() - 1);
    if (temporal_relations_.size() < active_transition_count)
      throw std::runtime_error(
          "AmrSystem : explicit AMR execution lacks a temporal relation for an active "
          "coarse/fine transition");
    // A coarse-only explicit bootstrap has not installed its state-transfer routes yet, so its
    // maximum spatial-transition contract is not available at this point. build_multi configures
    // the complete capacity immediately after those providers are prepared. Subsequent level
    // activation refreshes the active prefix inside AmrRuntime and needs no facade reinvention.
    if (static_cast<std::size_t>(cfg.level_count) > active_transition_count + 1)
      return;
    runtime->set_parent_child_temporal_relations(std::vector<::pops::amr::ParentChildClockRelation>(
        temporal_relations_.begin(),
        temporal_relations_.begin() + static_cast<std::ptrdiff_t>(active_transition_count)));
  }

  // Builds the unique runtime engine (AmrRuntime): one common SharedAmrLayout, then EACH block
  // materializes its type-erased AmrRuntimeBlock on it. The default field RHS is SUMMED and
  // CO-LOCATED across all blocks, including the one-block case.
  void build_multi() {
    // Direct low-level C++ use may intentionally skip mark_bound(). Keep the same canonical
    // collective immediately before lazy runtime construction so no field plan can materialize
    // without the exact ordered-registry witness.
    require_field_plan_consensus();
    // The full conservative state is threaded to both native and compiled
    // deferred builders (dispatch_amr_block -> build_amr_block), seeds every conservative component
    // on the coarse level and injects it to the fine levels. It takes priority over density.
    AmrBuildParams bp =
        make_layout_params();  // hierarchy/system authority; no representative block
    // A single block builds a coarse-only hierarchy when no prepared AMRTagging authority is
    // installed. A genuine multi-block AMR keeps the two-level seed. The runtime never infers an
    // active criterion from a scalar threshold or a representative component.
    const bool refinement_active = tagging_spec != nullptr;
    const bool single_level = blocks.size() == 1 && !refinement_active;
    const int initial_levels = cfg.explicit_bootstrap ? 1 : (single_level ? 1 : 2);
    detail::SharedAmrLayout S = detail::make_shared_amr_layout_levels(bp, initial_levels);
    S.boundary_plans = &boundary_plans_;
    std::vector<pops::AmrRuntimeBlock> rblocks;
    rblocks.reserve(blocks.size());
    for (auto& b : blocks) {
      if (b.is_compiled) {
        // Every compiled block must carry the exact runtime builder exported by its package.
        if (!b.compiled_block_builder)
          throw std::runtime_error(
              "AmrSystem : compiled block '" + b.name +
              "' without an AmrRuntime block provider. "
              "Regenerate the loader via dsl.compile_native(target='amr_system') / "
              "compile(backend='production', target='amr_system').");
        // Compiled path: the CONCRETE Model/Limiter/Flux are already captured
        // in the builder (add_compiled_model), we invoke it on the SHARED layout. It allocates the
        // level stack of the block on S and captures the scheme, EXACTLY like dispatch_amr_block for a
        // native block (the builder CALLS it internally). Partial IMEX masks fail closed at registration
        // until an executable AMR Program primitive consumes them, so only empty selectors reach here.
        // Zhang-Shu positivity floor (ADC-322): the AmrCompiledBlockBuilder now carries a floor slot,
        // so a loader regenerated against this header floors the Density-role face states like a native
        // block (forwarded to dispatch_amr_block -> build_amr_block). b.pos_floor == 0 for an OLDER .so
        // (it never marshals the field) -> inactive, bit-identical. No reject.
        rblocks.push_back(b.compiled_block_builder(
            S, b.name, b.density, b.has_density, b.state, b.has_state, b.gamma, b.substeps,
            b.recon_prim, b.stride, {}, {}, b.pos_floor, b.weno_epsilon, b.wave_speed_cache));
        continue;
      }
      // Native ModelSpec path: model dispatch -> concrete type, then spatial scheme dispatch
      // -> build_amr_block (allocates the block's level stack on the SHARED layout + closures).
      // The block density is carried by the BlockSpec (set_density(name) targets it).
      // Transport-axis seam (ADC-335): each per-transport TU (python/amr_block_<transport>.cpp) runs the
      // SAME dispatch_amr_block as before (build_amr_block_for), but instantiates only its transport's
      // leaves. Empty implicit selectors preserve that seam's ABI until the typed Program primitive
      // exists. The string if/else mirrors detail::dispatch_transport (same unknown-transport message).
      const detail::AmrBlockBuildArgs ba{b.spec,
                                         b.name,
                                         b.limiter,
                                         b.riemann,
                                         b.density,
                                         b.has_density,
                                         b.gamma,
                                         b.substeps,
                                         b.recon_prim,
                                         b.stride,
                                         b.has_state ? &b.state : nullptr,
                                         b.pos_floor,
                                         b.weno_epsilon,
                                         b.wave_speed_cache};
      // Transport dispatch mirrors detail::dispatch_transport (ADC-641): validate_transport preserves
      // the unknown_transport_msg byte-for-byte, then the switch on the typed TransportRouteId routes to
      // the per-transport seam.
      validate_transport(b.spec.transport);
      switch (parse_transport_route(b.spec.transport)) {
        case TransportRouteId::kExb:
          rblocks.push_back(detail::build_amr_block_exb(ba, S));
          break;
        case TransportRouteId::kCompressible:
          rblocks.push_back(detail::build_amr_block_compressible(ba, S));
          break;
        case TransportRouteId::kIsothermal:
          rblocks.push_back(detail::build_amr_block_isothermal(ba, S));
          break;
      }
    }
    if (!block_state_identities_.empty()) {
      if (block_state_identities_.size() != rblocks.size())
        throw std::runtime_error(
            "AmrSystem materialized blocks differ from the exact state-route registry");
      std::map<std::string, std::string> claimed;
      for (auto& block : rblocks) {
        const auto route = block_state_identities_.find(block.name);
        if (route == block_state_identities_.end() ||
            !claimed.emplace(route->second, block.name).second)
          throw std::runtime_error(
              "AmrSystem materialized block has a missing or duplicate state identity");
        block.state_identity = route->second;
      }
    } else {
      // Direct low-level construction has no Case Handle registry, but the prepared Tagger still
      // consumes exact qualified storage identities.  Give every engine-owned block its canonical
      // private identity instead of weakening PreparedTaggingExecutionPlan's authority checks.
      for (auto& block : rblocks)
        block.state_identity = direct_amr_state_identity(block.name);
    }
    std::vector<std::pair<int, int>> reconstruction_requirements;
    reconstruction_requirements.reserve(rblocks.size());
    for (const auto& block : rblocks)
      reconstruction_requirements.emplace_back(block.reconstruction_order,
                                               block.reconstruction_ghost_depth);
    runtime = std::make_shared<pops::AmrRuntime>(
        S.geom, S.runtime_hierarchy(), S.poisson_bc, std::move(rblocks), S.base_per,
        S.replicated_coarse, S.wall, field_solver_registry_, field_nullspace_provider_registry_,
        default_field_nullspace_, p_solver, p_solver_options);
    install_active_temporal_relations();
    if (amr_tagger_component_)
      runtime->install_external_tagger(amr_tagger_component_);
    if (amr_clustering_component_)
      runtime->install_external_clustering(amr_clustering_component_);
    if (!boundary_plans_.empty())
      runtime->install_boundary_storage_routes(field_storage_routes_);
    // Low-level facade compatibility has no authored AMRTransfer object.  Resolve its exact
    // coarse/fine requirement through the same native capability catalogue, then install the
    // selected prepared authority explicitly.  AmrRuntime itself owns no default or scheme switch.
    if (bootstrap_block_subjects.empty()) {
      for (std::size_t block = 0; block < reconstruction_requirements.size(); ++block) {
        const auto [order, ghost_depth] = reconstruction_requirements[block];
        auto coarse_fine = bootstrap_transfer_routes.prepare_minimum(
            pops::runtime::amr::TransferRouteDescriptor{"cell",
                                                        "cell",
                                                        "conservative",
                                                        "dense",
                                                        "coarse_fine_fill",
                                                        order,
                                                        {ghost_depth},
                                                        2,
                                                        kAmrRefRatio});
        runtime->set_block_transfer_authority(
            block, pops::runtime::amr::prepare_conservative_linear(),
            pops::runtime::amr::prepare_volume_average(), std::move(coarse_fine),
            pops::runtime::amr::prepare_linear_time_interpolation(), kAmrRefRatio);
      }
    }
    // The authored AMRTransfer registry is the authority for every public-DSL block's runtime
    // coarse/fine and temporal routes. Resolve the exact owner-qualified subject once, then retain
    // only prepared native callables in AmrRuntime; no string/kernel switch survives into stepping.
    int hierarchy_capacity_ratio = 0;
    for (const auto& [subject, block_name] : bootstrap_block_subjects) {
      const int block = block_index(block_name);
      const auto prolongation_id = bootstrap_subject_routes.find({subject, "prolongation"});
      const auto restriction_id = bootstrap_subject_routes.find({subject, "restriction"});
      const auto coarse_fine_id = bootstrap_subject_routes.find({subject, "coarse_fine_fill"});
      const auto temporal_id = bootstrap_subject_routes.find({subject, "temporal_interpolation"});
      if (block < 0 || prolongation_id == bootstrap_subject_routes.end() ||
          restriction_id == bootstrap_subject_routes.end() ||
          coarse_fine_id == bootstrap_subject_routes.end() ||
          temporal_id == bootstrap_subject_routes.end())
        throw std::runtime_error(
            "AmrSystem : state transfer authority lacks an exact prolongation, restriction, "
            "coarse/fine or temporal route");
      const auto& prolongation = bootstrap_transfer_routes.at(prolongation_id->second);
      const auto& restriction = bootstrap_transfer_routes.at(restriction_id->second);
      const auto& coarse_fine = bootstrap_transfer_routes.at(coarse_fine_id->second);
      const auto& temporal = bootstrap_transfer_routes.at(temporal_id->second);
      const int ratio = prolongation.descriptor.refinement_ratio;
      if (!prolongation.executable.spatial || !restriction.executable.spatial ||
          !coarse_fine.executable.coarse_fine || !temporal.executable.temporal ||
          restriction.descriptor.refinement_ratio != ratio ||
          coarse_fine.descriptor.refinement_ratio != ratio ||
          temporal.descriptor.refinement_ratio != ratio)
        throw std::runtime_error(
            "AmrSystem : state transfer authority did not prepare compatible native callables");
      if (hierarchy_capacity_ratio != 0 && hierarchy_capacity_ratio != ratio)
        throw std::runtime_error(
            "AmrSystem : shared AMR hierarchy state providers disagree on refinement ratio");
      hierarchy_capacity_ratio = ratio;
      runtime->set_block_transfer_authority(static_cast<std::size_t>(block),
                                            prolongation.executable, restriction.executable,
                                            coarse_fine.executable, temporal.executable, ratio);
    }
    if (cfg.explicit_bootstrap) {
      if (hierarchy_capacity_ratio < 2)
        throw std::runtime_error(
            "AmrSystem : explicit hierarchy capacity requires prepared state transfer routes");
      runtime->configure_hierarchy_capacity(
          std::vector<int>(static_cast<std::size_t>(cfg.level_count - 1), hierarchy_capacity_ratio),
          temporal_relations_);
    }
    // AMR / MPI PROFILING (Spec 5 criterion 43, ADC-479): wire the facade-owned Profiler into the
    // engine so it times its AMR phases (regrid / fill_boundary / average_down) into the SAME table
    // profile_report() renders. Non-owning: program_.profiler_ outlives runtime (both live on the
    // Impl, stable address). When profiling is disabled the engine never touches it (enabled()-guarded).
    runtime->set_profiler(&program_.profiler_);
    for (const auto& [subject, array] : bootstrap_arrays)
      runtime->register_bootstrap_staggered_field(subject, bootstrap_centering(array.centering),
                                                  array.ncomp, array.initial_values);
    // Materialize exactly the prepared graph consumed by the builtin Kokkos VM or an installed
    // external Tagger. Missing AMRTagging authority means no tagging program; the runtime never
    // manufactures a component-zero threshold or a shared-potential predicate.
    if ((cfg.explicit_bootstrap || cfg.regrid_every > 0) && tagging_spec) {
      using Program = AmrRuntime::TaggingProgram;
      std::vector<Program::Stencil> stencils = tagging_spec->stencils;
      std::vector<Program::Leaf> leaves;
      leaves.reserve(tagging_spec->leaf_ops.size());
      for (std::size_t index = 0; index < tagging_spec->leaf_ops.size(); ++index) {
        const std::string& kind = tagging_spec->leaf_subject_kinds[index];
        const std::string& identity = tagging_spec->leaf_subject_identities[index];
        std::size_t field_index = blocks.size();
        int component = -1;
        if (kind == "state") {
          const int block = block_index(tagging_spec->leaf_blocks[index]);
          if (block < 0)
            throw std::runtime_error("resolved AMR tagging names an unknown state block");
          const auto installed = block_state_identities_.find(tagging_spec->leaf_blocks[index]);
          const std::string expected =
              installed == block_state_identities_.end()
                  ? direct_amr_state_identity(tagging_spec->leaf_blocks[index])
                  : installed->second;
          if (identity != expected)
            throw std::runtime_error(
                "resolved AMR tagging state identity disagrees with native block storage");
          field_index = static_cast<std::size_t>(block);
          component = detail::resolve_selected_component(
              "AmrSystem::resolved tagging", tagging_spec->leaf_blocks[index],
              runtime->block_cons_vars(field_index), tagging_spec->leaf_variables[index], "");
        } else if (kind == "aux") {
          if (identity != "pops://runtime/amr/shared-aux")
            throw std::runtime_error(
                "resolved AMR tagging aux identity is not the qualified shared-aux carrier");
          component = aux_canonical_index(tagging_spec->leaf_variables[index]);
          if (component < 0)
            throw std::runtime_error(
                "resolved AMR tagging aux subject is not a canonical aux component");
        } else if (kind == "field") {
          const auto route = field_storage_routes_.find(identity);
          if (route == field_storage_routes_.end())
            throw std::runtime_error(
                "resolved AMR tagging field has no authenticated native storage route");
          const auto plan = field_plans_.find(route->second);
          if (plan == field_plans_.end())
            throw std::runtime_error(
                "resolved AMR tagging field route has no installed provider plan");
          const auto outputs =
              ell_field_comps_.find({plan->second.output_block, plan->second.output_key});
          const int output_index = tagging_spec->leaf_field_component_indices[index];
          if (outputs == ell_field_comps_.end() || output_index < 0 || output_index >= 3 ||
              outputs->second[static_cast<std::size_t>(output_index)] < 0)
            throw std::runtime_error(
                "resolved AMR tagging field component is absent from its prepared output route");
          component = outputs->second[static_cast<std::size_t>(output_index)];
        } else {
          throw std::runtime_error("resolved AMR tagging has an unknown subject kind");
        }
        leaves.push_back(Program::Leaf{
            field_index, static_cast<std::size_t>(component), tagging_spec->leaf_ops[index],
            tagging_spec->leaf_thresholds[index],
            tagging_spec->leaf_stencil_indices[index] < 0
                ? POPS_TAGGING_NO_STENCIL_V1
                : static_cast<std::size_t>(tagging_spec->leaf_stencil_indices[index])});
      }
      runtime->set_tagging_program(std::move(stencils), std::move(leaves), tagging_spec->refine_ops,
                                   tagging_spec->refine_args, tagging_spec->coarsen_ops,
                                   tagging_spec->coarsen_args, tagging_spec->min_cycles,
                                   tagging_spec->equality_policy, tagging_spec->conflict_policy,
                                   tagging_spec->clock_identity, tagging_spec->provider_identity);
    }
    // Canonical B_z and model-NAMED aux fields share one native static-field authority. The runtime
    // validates that a block declared each component, publishes coarse->fine immediately, and
    // re-applies the fields after every field solve and hierarchy growth.
    if (!bz_field.empty()) {
      constexpr int bz_component = aux_canonical_index("B_z");
      static_assert(bz_component == 3);
      runtime->set_static_aux_component(bz_component, bz_field);
    }
    for (const auto& kv : named_aux_)
      runtime->set_named_aux(kv.first, kv.second);
    for (const auto& kv : named_aux_bc_)
      runtime->set_named_aux_bc(kv.first, kv.second);  // ADC-369
    // NAMED multi-elliptic fields (ADC-428): replay the loader's declarations + per-block RHS closures on
    // the just-built runtime engine. register_named_field records the field's aux output components; each
    // RHS closure is attached to its block's AmrRuntimeBlock (resolved by name) so solve_named_fields can
    // sum them. A closure naming an unknown block is a loader/codegen bug -> reject loud. Empty maps ->
    // no-op (no named field), bit-identical.
    for (const auto& [slot, plan] : field_plans_)
      runtime->install_field_plan(slot, plan);
    for (const auto& kv : ell_field_comps_)
      runtime->register_named_field(kv.first.first, kv.first.second, kv.second[0], kv.second[1],
                                    kv.second[2], kv.second[3]);
    for (const auto& [field, by_block] : ell_field_rhs_)
      for (const auto& [block_name, rhs] : by_block) {
        const int bi = block_index(block_name);
        if (bi < 0)
          throw std::runtime_error("AmrSystem::set_block_elliptic_field : unknown block '" +
                                   block_name + "' for named elliptic field '" + field + "'");
        runtime->set_block_named_elliptic_rhs(static_cast<std::size_t>(bi), field, rhs);
      }
    // GLOBAL bounds registered BEFORE the lazy build (add_dt_bound): passed to the engine
    // (which aggregates them in its step_cfl, all_reduce_min). Those added AFTER go in directly.
    for (const auto& g : dt_bounds)
      runtime->add_dt_bound(g.label, g.fn);
    // Declared frequencies of the coupled sources (CoupledSource.frequency, wave 3): step bound
    // dt <= cfl/mu on the runtime engine macro-step. CONSTANT frequency then PER-CELL frequency
    // (Expr): the second is evaluated on the coarse grid at each step_cfl (add_coupled_frequency_expr
    // resolves the inputs / validates the bytecode; empty program -> ignored).
    for (const auto& cs : coupled_sources) {
      if (cs.frequency > 0.0)
        runtime->add_coupled_frequency(cs.label, static_cast<Real>(cs.frequency));
      runtime->add_coupled_frequency_expr(cs.label, cs.in_blocks, cs.in_roles, cs.consts,
                                          cs.freq_prog_ops, cs.freq_prog_args);
    }
    // The cadence and clustering provider consume the prepared graph installed above.
    // regrid_every == 0 keeps the hierarchy frozen and therefore needs no tagging execution image.
    runtime->set_regrid(cfg.regrid_every, cfg.regrid_grow, cfg.regrid_margin);
    // ADC-616: Berger-Rigoutsos clustering params. Each <= 0 keeps the ClusterParams default (0.7 /
    // 1 / 32), so an unconfigured AMR run clusters bit-identically. Applied only when set.
    if (cfg.cluster_min_efficiency > 0.0 || cfg.cluster_min_box_size > 0 ||
        cfg.cluster_max_box_size > 0) {
      const double eff = cfg.cluster_min_efficiency > 0.0 ? cfg.cluster_min_efficiency : 0.7;
      const int minb = cfg.cluster_min_box_size > 0 ? cfg.cluster_min_box_size : 1;
      const int maxb = cfg.cluster_max_box_size > 0 ? cfg.cluster_max_box_size : 32;
      runtime->set_clustering(eff, minb, maxb);
    }
    // Replays the coupled sources frozen at add_coupled_source on the just-built runtime engine:
    // each resolves (block, role) -> (index, component) against the blocks' cons_vars and stores its
    // closure (applied after transport at each macro-step). No source -> no-op (the loop is empty),
    // so multi-block without coupling stays bit-identical to before.
    for (const auto& cs : coupled_sources)
      runtime->add_coupled_source(cs.in_blocks, cs.in_roles, cs.consts, cs.out_blocks, cs.out_roles,
                                  cs.prog_ops, cs.prog_args, cs.prog_lens);
    built = true;
  }

  void ensure_built() {
    if (built)
      return;
    if (blocks.empty())
      throw std::runtime_error("AmrSystem : call add_block first");
    build_multi();
  }
};

namespace {
// UPSTREAM configuration guard (ADC-299): validate the AmrSystemConfig invariants BEFORE constructing
// Impl. The AMR Impl ctor is trivial (it only stores cfg, allocating nothing from n), so unlike System
// nothing is built before the check; we still validate ahead of Impl for parity with System and to keep
// every config rejection at a single upstream point. Axis sizes and physical bounds must be valid
// before any dense payload shape is computed; the zero ny/Ly values are the documented direct-C++
// square shorthand and are canonicalized immediately after validation. L,
// regrid_every and coarse_max_grid were unchecked and reach the lazy build (dx, regrid cadence, coarse
// tiling) as is.
void validate_amr_system_config(const AmrSystemConfig& c) {
  if (c.n < 1)
    throw std::runtime_error("AmrSystem : n >= 1 required (coarse x cells) ; got n = " +
                             std::to_string(c.n));
  if (c.ny < 0)
    throw std::runtime_error("AmrSystem : ny must be zero (square shorthand) or >= 1; got ny = " +
                             std::to_string(c.ny));
  if (!(c.L > 0.0))
    throw std::runtime_error("AmrSystem : L > 0 required (Cartesian x extent) ; got L = " +
                             std::to_string(c.L));
  if (c.Ly < 0.0)
    throw std::runtime_error("AmrSystem : Ly must be zero (square shorthand) or > 0; got Ly = " +
                             std::to_string(c.Ly));
  const double resolved_Ly = c.Ly == 0.0 ? c.L : c.Ly;
  if (!std::isfinite(c.L) || !std::isfinite(resolved_Ly) || !std::isfinite(c.xlo) ||
      !std::isfinite(c.ylo) || !std::isfinite(c.xlo + c.L) || !std::isfinite(c.ylo + resolved_Ly))
    throw std::runtime_error(
        "AmrSystem : finite Cartesian origin and upper bounds required; got xlo = " +
        std::to_string(c.xlo) + ", ylo = " + std::to_string(c.ylo));
  if (c.regrid_every < 0)
    throw std::runtime_error(
        "AmrSystem : regrid_every >= 0 required (0 = never regrid after init) ; "
        "got regrid_every = " +
        std::to_string(c.regrid_every));
  if (c.level_count < 1)
    throw std::runtime_error("AmrSystem : level_count >= 1 required");
  if (!c.explicit_bootstrap && c.level_count != 2)
    throw std::runtime_error(
        "AmrSystem : non-default level_count requires explicit_bootstrap=true; deterministic "
        "fine-level preallocation is not a valid substitute");
  if (c.regrid_grow < 0 || c.regrid_margin < 0)
    throw std::runtime_error("AmrSystem : regrid_grow/regrid_margin must be >= 0");
  if (c.coarse_max_grid < 0)
    throw std::runtime_error(
        "AmrSystem : coarse_max_grid >= 0 required (0 = default n/2 tile, "
        "distribute_coarse only) ; got coarse_max_grid = " +
        std::to_string(c.coarse_max_grid));
  if (c.load_balance_route.empty() || c.load_balance_identity.empty())
    throw std::runtime_error(
        "AmrSystem : load-balance route and semantic identity must be non-empty");
  (void)c.load_balance_options.exact_contract();
}

AmrSystemConfig canonical_amr_system_config(const AmrSystemConfig& authored) {
  AmrSystemConfig canonical = authored;
  if (canonical.ny == 0)
    canonical.ny = canonical.n;
  if (canonical.Ly == 0.0)
    canonical.Ly = canonical.L;
  return canonical;
}

// RUNTIME FREEZE LIFECYCLE guard (ADC-592, parity with System::require_assembling): a STRUCTURAL setter
// must not mutate the composition once pops.bind has completed (@p bound == true). Called at the TOP of
// each structural setter, BEFORE its existing 'if (p_->built) throw' lazy-phase guard -- 'bound' is a
// DISTINCT (earlier) phase than the lazy 'built' materialization, so the historical built messages
// (pinned by existing tests) are left verbatim; this adds the bind-vocabulary refusal on top. The
// message NEVER recommends a legacy setter as the remedy (no validation bypass).
void require_assembling_amr(bool bound, const char* what) {
  if (bound)
    throw std::runtime_error(
        std::string("AmrSystem::") + what +
        ": the composition is frozen once pops.bind completes (runtime lifecycle 'bound'); declare "
        "it on the pops.Case (blocks / field problems / AMR layout / source stage / refinement / "
        "solver routes / aux layout / installed Program) and lower it with pops.compile(...) + "
        "pops.bind(...). Only runtime data / params / checkpoint / diagnostics may change on a "
        "bound simulation.");
}
}  // namespace

AmrSystem::AmrSystem(const AmrSystemConfig& c) {
  validate_amr_system_config(c);  // BEFORE Impl (parity with System; single upstream config guard)
  p_ = std::make_unique<Impl>(canonical_amr_system_config(c));
}
AmrSystem::~AmrSystem() = default;
AmrSystem::AmrSystem(AmrSystem&&) noexcept = default;
AmrSystem& AmrSystem::operator=(AmrSystem&&) noexcept = default;

void AmrSystem::add_block(const std::string& name, const ModelSpec& model,
                          const std::string& limiter, const std::string& riemann,
                          const std::string& recon, const std::string& time, int substeps,
                          int stride, const std::vector<std::string>& implicit_vars,
                          const std::vector<std::string>& implicit_roles,
                          const NewtonOptions& newton, bool newton_diagnostics,
                          double positivity_floor, double weno_epsilon, bool wave_speed_cache) {
  require_assembling_amr(p_->bound_, "add_block");  // frozen once pops.bind completes (ADC-592)
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::add_block : the system is already built (call "
        "add_block before any step/mass/density)");
  // Completeness contract of the model (ADC-290, parity with System::add_block): transport / elliptic
  // must be chosen explicitly. Validated before the build_multi transport routing, so a
  // default-constructed ModelSpec fails clearly instead of silently
  // selecting Euler + Poisson-charge.
  detail::validate_model_spec(model);
  if (substeps < 1)
    throw std::runtime_error("AmrSystem::add_block : substeps >= 1");
  if (stride < 1)
    throw std::runtime_error("AmrSystem::add_block : stride >= 1");
  // Zhang-Shu positivity floor (ADC-259): eager validation (parity with System::add_block). 0 =
  // inactive (bit-identical). The Density-role probe + the compiled-.so rejection happen at lazy build.
  if (!(positivity_floor >= 0.0) || !std::isfinite(positivity_floor))
    throw std::runtime_error(
        "AmrSystem::add_block : positivity_floor >= 0 and finite (0 = inactive)");
  if (!std::isfinite(weno_epsilon) || weno_epsilon <= 0.0)
    throw std::runtime_error("AmrSystem::add_block : finite weno_epsilon > 0 required");
  if (weno_epsilon != static_cast<double>(kWenoEpsilon) && limiter != "weno5")
    throw std::runtime_error("AmrSystem::add_block : weno_epsilon applies to limiter='weno5' only");
  if (wave_speed_cache && riemann != "hll")
    throw std::runtime_error("AmrSystem::add_block : wave_speed_cache requires riemann='hll'");
  if (wave_speed_cache && time == "imex")
    throw std::runtime_error(
        "AmrSystem::add_block : wave_speed_cache is supported by explicit AMR transport only");
  // Compatibility POD shared with System::add_block. AMR validates its value domain here, then
  // rejects every non-default request below: source-Newton configuration belongs on an executable
  // typed Program solve and is neither stored nor decoded by this spatial facade.
  // Range check shared with System::add_block (validate_newton_options, in implicit_stepper.hpp).
  validate_newton_options(newton, "AmrSystem::add_block");
  const bool newton_non_default = amr_newton_options_non_default(newton);
  if (time != "imex" && newton_non_default)
    throw std::runtime_error("AmrSystem::add_block : Newton options require time='imex'");
  // A Newton diagnostic request is meaningful only for IMEX, but the executable typed Program solve
  // must own its report. The spatial AMR facade therefore rejects true below instead of exposing a
  // carrier with no writer.
  if (time != "imex" && newton_diagnostics)
    throw std::runtime_error("AmrSystem::add_block : newton_diagnostics requires time='imex'");
  // The route names describe mutually exclusive authored Program methods. The spatial block builder
  // receives no time-method switch.
  if (time != "explicit" && time != "euler" && time != "imex" && time != "ssprk3") {
    if (time == "imexrk_ars222")
      throw std::runtime_error(
          "AmrSystem : time 'imexrk_ars222' (IMEX-RK family, ARS(2,2,2) scheme) not wired on AMR "
          "(scope = Cartesian System). Use one of [" +
          amr_time_routes_csv() +
          "] on AMR, or a "
          "Cartesian System for IMEX-RK.");
    throw std::runtime_error("AmrSystem : time '" + time +
                             "' unknown on AMR (valid: " + amr_time_routes_csv() + ")");
  }
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("AmrSystem : unknown recon '" + recon +
                             "' (valid: " + kReconRouteTokensCsv + ")");
  const bool imex = (time == "imex");
  if (imex && (newton_non_default || newton_diagnostics))
    throw std::runtime_error(
        "AmrSystem::add_block : NewtonOptions / newton_diagnostics are unavailable on AMR because "
        "no executable typed Program implicit-source solve consumes them; the AMR target does not "
        "yet provide the local nonlinear/Newton primitive. Keep the AMR Program explicit, use a "
        "typed LocalLinear solve for a linear implicit operator, or use the uniform System "
        "source-Newton route");
  // The partial IMEX mask belongs only to an authored IMEX Program composition: requesting it for an
  // explicit route is an ERROR (no silent ignore; same guard as System::add_block).
  if (!imex && (!implicit_vars.empty() || !implicit_roles.empty()))
    throw std::runtime_error(
        "AmrSystem::add_block : implicit_vars / implicit_roles require time='imex' "
        "(the implicit mask only applies to an authored IMEX Program ; got time='" +
        time + "')");
  if (imex && (!implicit_vars.empty() || !implicit_roles.empty()))
    throw std::runtime_error(
        "AmrSystem::add_block : implicit_vars / implicit_roles are unavailable on AMR because no "
        "executable AMR Program implicit-source primitive consumes a partial IMEX mask");
  // Every block count uses the AmrRuntime engine (shared hierarchy, co-located summed Poisson). An
  // already COMPILED block (set_compiled_block / add_compiled_model) CAN mix
  // with a native block: its runtime builder (compiled_block_builder) materializes an
  // AmrRuntimeBlock on the SAME shared layout, exactly like a native block (cf. build_multi). A
  // single hard guard: the compiled block must have been registered WITH a runtime builder (an .so
  // loader recompiled against this header provides it; build_multi throws clearly otherwise).
  if (name.empty())
    throw std::runtime_error("AmrSystem::add_block requires an owner-qualified non-empty name");
  if (p_->block_index(name) >= 0)
    throw std::runtime_error("AmrSystem::add_block : block name already used '" + name + "'");
  Impl::BlockSpec b;
  b.name = name;
  b.is_compiled = false;
  b.spec = model;
  b.limiter = limiter;
  b.riemann = riemann;
  b.recon_prim = (recon == "primitive");
  b.time = time;
  b.pos_floor = positivity_floor;  // Zhang-Shu floor (ADC-259); threaded at build (single/multi)
  b.weno_epsilon = weno_epsilon;
  b.wave_speed_cache = wave_speed_cache;
  b.substeps = substeps;
  b.stride = stride;
  b.gamma = model.gamma;  // adiabatic index of the block (Euler), read by coupler_write_coarse
  p_->blocks.push_back(std::move(b));
}

POPS_EXPORT void AmrSystem::install_block_state_route(const std::string& name,
                                                      const std::string& state_identity) {
  Impl* P = p_.get();
  require_assembling_amr(P->bound_, "install_block_state_route");
  if (P->built)
    throw std::runtime_error("AmrSystem::install_block_state_route: system is already built");
  if (!P->blocks.empty())
    throw std::runtime_error(
        "AmrSystem::install_block_state_route must precede block declarations");
  if (name.empty() || state_identity.empty() || P->block_state_identities_.count(name) != 0)
    throw std::runtime_error(
        "AmrSystem block state route requires unique non-empty block/state identities");
  for (const auto& [_, installed_identity] : P->block_state_identities_)
    if (installed_identity == state_identity)
      throw std::runtime_error(
          "AmrSystem block state route has a duplicate qualified state identity");
  P->block_state_identities_.emplace(name, state_identity);
}

POPS_EXPORT void AmrSystem::install_boundary_plan(
    const std::string& name, const std::string& identity, int required_depth,
    const std::vector<std::string>& face_types, const std::vector<double>& face_values,
    const std::vector<std::string>& face_identities,
    const std::vector<std::string>& component_roles,
    const std::vector<int>& omitted_interface_faces, const std::string& state_identity,
    PreparedBoundaryReadDependencies read_dependencies) {
  install_boundary_plan(name, identity, required_depth, face_types, face_values, face_identities,
                        component_roles, omitted_interface_faces, state_identity,
                        std::move(read_dependencies), {});
}

POPS_EXPORT void AmrSystem::install_boundary_plan(
    const std::string& name, const std::string& identity, int required_depth,
    const std::vector<std::string>& face_types, const std::vector<double>& face_values,
    const std::vector<std::string>& face_identities,
    const std::vector<std::string>& component_roles,
    const std::vector<int>& omitted_interface_faces, const std::string& state_identity,
    PreparedBoundaryReadDependencies read_dependencies,
    std::vector<PeriodicIdentification2D> periodic_identifications,
    const std::vector<std::string>& face_representations,
    const std::vector<std::string>& face_converter_identities,
    const std::vector<std::vector<std::string>>& face_analytic_opcodes,
    const std::vector<std::vector<double>>& face_analytic_literals,
    const std::vector<std::string>& face_analytic_clocks) {
  Impl* P = p_.get();
  using BoundaryPlanMap = decltype(P->boundary_plans_);
  using BoundaryPlanNode = typename BoundaryPlanMap::node_type;
  BoundaryPlanNode prepared = analytic::collectively_prepare_exact_analytic_request(
      "AmrSystem::install_boundary_plan",
      [&]() -> BoundaryPlanNode {
        require_assembling_amr(P->bound_, "install_boundary_plan");
        if (P->built)
          throw std::runtime_error("AmrSystem::install_boundary_plan: system is already built");
        if (name.empty() || state_identity.empty() || P->boundary_plans_.count(name) != 0)
          throw std::runtime_error(
              "AmrSystem::install_boundary_plan requires unique block/state-qualified identities");
        const auto state_route = P->block_state_identities_.find(name);
        if (state_route == P->block_state_identities_.end() ||
            state_route->second != state_identity)
          throw std::runtime_error(
              "AmrSystem::install_boundary_plan state differs from the exact block state route");
        for (const auto& [_, installed] : P->boundary_plans_)
          if (installed->state_identity() == state_identity)
            throw std::runtime_error(
                "AmrSystem::install_boundary_plan duplicate qualified state identity");

        auto hyperbolic = prepare_hyperbolic_boundary<2>(
            face_types, face_values, face_identities, component_roles,
            !periodic_identifications.empty(), face_representations, face_converter_identities,
            face_analytic_opcodes, face_analytic_literals, face_analytic_clocks);
        auto plan = std::make_shared<PreparedBoundaryPlan>(
            identity, required_depth, std::move(hyperbolic), omitted_interface_faces,
            state_identity, read_dependencies, periodic_identifications);
        if (plan->has_mapped_periodicity())
          throw std::runtime_error(
              "AmrSystem::install_boundary_plan: mapped periodic topology is not supported by AMR "
              "fill-patch/regrid; use the uniform runtime or an axis-aligned translation");
        BoundaryPlanMap staged;
        staged.emplace(name, std::move(plan));
        return staged.extract(staged.begin());
      },
      [&]() {
        return detail::canonical_prepared_boundary_plan_request(
            name, identity, required_depth, face_types, face_values, face_identities,
            component_roles, omitted_interface_faces, state_identity, read_dependencies,
            periodic_identifications, face_representations, face_converter_identities,
            face_analytic_opcodes, face_analytic_literals, face_analytic_clocks);
      });
  const auto published = P->boundary_plans_.insert(std::move(prepared));
  if (!published.inserted)
    throw std::logic_error("AmrSystem::install_boundary_plan lost its prepared publication slot");
}

POPS_EXPORT void AmrSystem::install_field_storage_route(const std::string& field_identity,
                                                        const std::string& provider_slot) {
  Impl* P = p_.get();
  require_assembling_amr(P->bound_, "install_field_storage_route");
  if (P->built)
    throw std::runtime_error("AmrSystem::install_field_storage_route: system is already built");
  if (field_identity.empty() || provider_slot.empty() ||
      !P->field_storage_routes_.emplace(field_identity, provider_slot).second)
    throw std::runtime_error(
        "AmrSystem field storage route requires unique non-empty qualified identities");
}

POPS_EXPORT void AmrSystem::discard_boundary_plans() {
  Impl* P = p_.get();
  require_assembling_amr(P->bound_, "discard_boundary_plans");
  if (P->built || !P->blocks.empty())
    throw std::runtime_error(
        "AmrSystem::discard_boundary_plans is restricted to a failed pre-block transaction");
  P->boundary_plans_.clear();
  P->block_state_identities_.clear();
  P->field_storage_routes_.clear();
}

POPS_EXPORT void AmrSystem::install_ghost_boundary_component(
    const std::string& name, PreparedBoundaryComponentSpec spec,
    std::shared_ptr<component::LoadedComponent> component) {
  Impl* P = p_.get();
  require_assembling_amr(P->bound_, "install_ghost_boundary_component");
  if (P->built)
    throw std::runtime_error(
        "AmrSystem::install_ghost_boundary_component: system is already built");
  const auto found = P->boundary_plans_.find(name);
  if (found == P->boundary_plans_.end())
    throw std::runtime_error("AmrSystem ghost boundary requires an installed block boundary plan");
  found->second->install_ghost_component(std::move(spec), std::move(component));
}

POPS_EXPORT void AmrSystem::install_field_boundary_residual_component(
    const std::string& name, PreparedBoundaryComponentSpec spec,
    std::shared_ptr<component::LoadedComponent> component) {
  Impl* P = p_.get();
  require_assembling_amr(P->bound_, "install_field_boundary_residual_component");
  if (P->built)
    throw std::runtime_error("AmrSystem field boundary residual: system is already built");
  const auto found = P->boundary_plans_.find(name);
  if (found == P->boundary_plans_.end())
    throw std::runtime_error(
        "AmrSystem field boundary residual requires an installed block boundary plan");
  found->second->install_residual_component(std::move(spec), std::move(component));
}

POPS_EXPORT void AmrSystem::install_field_boundary_jvp_component(
    const std::string& name, PreparedBoundaryComponentSpec spec,
    std::shared_ptr<component::LoadedComponent> component) {
  Impl* P = p_.get();
  require_assembling_amr(P->bound_, "install_field_boundary_jvp_component");
  if (P->built)
    throw std::runtime_error("AmrSystem field boundary JVP: system is already built");
  const auto found = P->boundary_plans_.find(name);
  if (found == P->boundary_plans_.end())
    throw std::runtime_error(
        "AmrSystem field boundary JVP requires an installed block boundary plan");
  found->second->install_jvp_component(std::move(spec), std::move(component));
}

POPS_EXPORT void AmrSystem::install_amr_tagger_component(
    runtime::amr::PreparedTaggerSpec spec, std::shared_ptr<component::LoadedComponent> component) {
  Impl* P = p_.get();
  require_assembling_amr(P->bound_, "install_amr_tagger_component");
  if (P->built || P->amr_tagger_component_)
    throw std::runtime_error(
        "AmrSystem external Tagger requires one installation before runtime build");
  P->amr_tagger_component_ = std::make_shared<runtime::amr::PreparedTaggerComponent>(
      std::move(spec), std::move(component));
}

POPS_EXPORT void AmrSystem::install_amr_clustering_component(
    runtime::amr::PreparedClusteringSpec spec,
    std::shared_ptr<component::LoadedComponent> component) {
  Impl* P = p_.get();
  require_assembling_amr(P->bound_, "install_amr_clustering_component");
  if (P->built || P->amr_clustering_component_)
    throw std::runtime_error(
        "AmrSystem external Clustering requires one installation before runtime build");
  if (P->cfg.cluster_min_efficiency > 0.0 || P->cfg.cluster_min_box_size > 0 ||
      P->cfg.cluster_max_box_size > 0)
    throw std::runtime_error(
        "AmrSystem external Clustering cannot consume Berger-Rigoutsos controls");
  P->amr_clustering_component_ = std::make_shared<runtime::amr::PreparedClusteringComponent>(
      std::move(spec), std::move(component));
}

POPS_EXPORT void AmrSystem::discard_amr_provider_components() {
  Impl* P = p_.get();
  require_assembling_amr(P->bound_, "discard_amr_provider_components");
  if (P->built)
    throw std::runtime_error("AmrSystem cannot discard AMR providers after runtime build");
  P->amr_tagger_component_.reset();
  P->amr_clustering_component_.reset();
}

POPS_EXPORT void AmrSystem::install_interface_flux_component(
    runtime::multiblock::AxisAlignedInterface route,
    runtime::multiblock::PreparedInterfaceFluxSpec spec,
    std::shared_ptr<component::LoadedComponent> component) {
  Impl* P = p_.get();
  require_assembling_amr(P->bound_, "install_interface_flux_component");
  if (route.identity.empty() || spec.interface_identity != route.identity)
    throw std::invalid_argument("AmrSystem shared-interface route/spec identity mismatch");
  if (!spec.execution)
    throw std::invalid_argument(
        "AmrSystem shared-interface component lacks exact ExecutionContext");
  P->ensure_built();
  if (!P->runtime)
    throw std::runtime_error(
        "AmrSystem shared-interface installation requires the common AmrRuntime engine");
  spec.normal_axis = route.left_axis == runtime::multiblock::InterfaceAxis::X ? 0 : 1;
  spec.outward_sign = route.left_side == runtime::multiblock::InterfaceSide::Low ? -1 : 1;
  const int level = route.level;
  const Geometry geometry = P->runtime->level_geom(level);
  spec.face_measure = spec.normal_axis == 0 ? static_cast<double>(geometry.dy())
                                            : static_cast<double>(geometry.dx());
  const PopsExecutionContextV1 execution = spec.execution->view();
  P->runtime->install_level_interface_flux(
      level, std::move(route), execution,
      [spec = std::move(spec), component = std::move(component)]() mutable {
        auto prepared = std::make_shared<runtime::multiblock::PreparedInterfaceFluxComponent>(
            std::move(spec), std::move(component));
        return runtime::multiblock::InterfaceFluxEvaluator(
            [prepared](const runtime::multiblock::BoundaryEvaluationPoint& point,
                       const runtime::multiblock::InterfaceFluxBatch& batch) {
              prepared->evaluate(point, batch);
            });
      });
}

POPS_EXPORT std::size_t AmrSystem::interface_evaluation_count(const std::string& identity,
                                                              int level) const {
  if (!p_->runtime)
    throw std::runtime_error(
        "AmrSystem shared-interface evaluation count requires a built runtime engine");
  return p_->runtime->interface_evaluation_count(identity, level);
}

POPS_EXPORT void AmrSystem::discard_interface_flux_components() {
  Impl* P = p_.get();
  require_assembling_amr(P->bound_, "discard_interface_flux_components");
  if (P->runtime)
    P->runtime->discard_interface_fluxes();
}

POPS_EXPORT std::size_t AmrSystem::interface_flux_installation_checkpoint() const {
  if (!p_->runtime)
    throw std::runtime_error(
        "AmrSystem interface installation checkpoint requires a built runtime engine");
  return p_->runtime->interface_flux_installation_checkpoint();
}

POPS_EXPORT void AmrSystem::rollback_interface_flux_installations(std::size_t accepted_size) {
  Impl* P = p_.get();
  require_assembling_amr(P->bound_, "rollback_interface_flux_installations");
  if (!P->runtime)
    throw std::runtime_error(
        "AmrSystem interface installation rollback requires a built runtime engine");
  P->runtime->rollback_interface_flux_installations(accepted_size);
}

POPS_EXPORT void AmrSystem::set_compiled_block(
    int ncomp, double gamma, int substeps, AmrCompiledBlockBuilder runtime_builder,
    const std::string& name, bool recon_prim, const std::string& time, int stride,
    const std::vector<std::string>& implicit_vars, const std::vector<std::string>& implicit_roles,
    double pos_floor, double weno_epsilon, bool wave_speed_cache) {
  (void)ncomp;  // the number of variables is carried by the concrete Model (Model::n_vars) in the
                // type-erasing builders; the parameter stays for API symmetry with System.
  require_assembling_amr(p_->bound_,
                         "set_compiled_block");  // frozen once pops.bind completes (ADC-592)
  if (p_->built)
    throw std::runtime_error("AmrSystem::set_compiled_block : the system is already built");
  if (substeps < 1)
    throw std::runtime_error("AmrSystem::set_compiled_block : substeps >= 1");
  if (stride < 1)
    throw std::runtime_error("AmrSystem::set_compiled_block : stride >= 1");
  if (!runtime_builder)
    throw std::runtime_error(
        "AmrSystem::set_compiled_block requires an executable AmrRuntime block provider");
  if (!std::isfinite(weno_epsilon) || weno_epsilon <= 0.0)
    throw std::runtime_error("AmrSystem::set_compiled_block requires finite weno_epsilon > 0");
  const bool token_is_imex = time == route_token(TimeRouteId::kImex);
  if (time != route_token(TimeRouteId::kExplicitSsprk2) &&
      time != route_token(TimeRouteId::kForwardEuler) &&
      time != route_token(TimeRouteId::kSsprk3) && !token_is_imex)
    throw std::runtime_error("AmrSystem::set_compiled_block : unknown time route '" + time + "'");
  if (token_is_imex && wave_speed_cache)
    throw std::runtime_error(
        "AmrSystem::set_compiled_block : wave_speed_cache is supported by explicit AMR transport "
        "only");
  // The partial IMEX mask belongs only to an authored IMEX Program composition (same guard as
  // add_block): requesting it for an explicit route is an ERROR (no silent ignore).
  if (!token_is_imex && (!implicit_vars.empty() || !implicit_roles.empty()))
    throw std::runtime_error(
        "AmrSystem::set_compiled_block : implicit_vars / implicit_roles require "
        "time='imex' (the implicit mask only applies to an authored IMEX Program)");
  if (token_is_imex && (!implicit_vars.empty() || !implicit_roles.empty()))
    throw std::runtime_error(
        "AmrSystem::set_compiled_block : implicit_vars / implicit_roles are unavailable on AMR "
        "because no executable AMR Program implicit-source primitive consumes a partial IMEX mask");
  if (name.empty())
    throw std::runtime_error(
        "AmrSystem::set_compiled_block requires an owner-qualified non-empty name");
  if (p_->block_index(name) >= 0)
    throw std::runtime_error("AmrSystem::set_compiled_block : block name already used '" + name +
                             "'");
  Impl::BlockSpec b;
  b.name = name;
  b.is_compiled = true;
  b.gamma = gamma;
  b.substeps = substeps;
  b.stride = stride;
  b.recon_prim = recon_prim;
  b.time = time;
  // Zhang-Shu positivity floor (ADC-322): carried by the regenerated .so loader (pops_install_native_amr
  // -> add_compiled_model). Stored on the block and forwarded through the AmrRuntime builder.
  b.pos_floor = pos_floor;
  b.weno_epsilon = weno_epsilon;
  b.wave_speed_cache = wave_speed_cache;
  b.compiled_block_builder = std::move(runtime_builder);
  p_->blocks.push_back(std::move(b));
}

// NAMED multi-elliptic field declaration (ADC-428): the native AMR loader calls this once per
// m.elliptic_field after the block is installed. We stash the field's aux output components and the
// unique runtime materialization replays it. POPS_EXPORT:
// resolved by the generated AMR .so loader across the dlopen boundary (same as set_compiled_block).
POPS_EXPORT void AmrSystem::register_elliptic_field(const std::string& block_name,
                                                    const std::string& provider_key, int phi_comp,
                                                    int gx_comp, int gy_comp, int gradient_sign) {
  require_assembling_amr(p_->bound_,
                         "register_elliptic_field");  // frozen once pops.bind completes (ADC-592)
  if (p_->built)
    throw std::runtime_error("AmrSystem::register_elliptic_field : the system is already built");
  const bool has_gradient = gx_comp >= 0 && gy_comp >= 0;
  const bool has_no_gradient = gx_comp == -1 && gy_comp == -1;
  if (phi_comp < 0 || (!has_gradient && !has_no_gradient) ||
      (has_gradient && (phi_comp == gx_comp || phi_comp == gy_comp || gx_comp == gy_comp)))
    throw std::invalid_argument(
        "AmrSystem::register_elliptic_field requires one potential or three distinct "
        "potential/gradient components");
  if (gradient_sign != -1 && gradient_sign != 1)
    throw std::invalid_argument(
        "AmrSystem::register_elliptic_field gradient_sign must be exactly -1 or 1");
  if (!has_gradient && gradient_sign != 1)
    throw std::invalid_argument(
        "AmrSystem::register_elliptic_field without gradients requires sign +1");
  p_->ell_field_comps_[{block_name, provider_key}] = {phi_comp, gx_comp, gy_comp, gradient_sign};
}

// Attaches a named field's per-block RHS closure (ADC-428): the native AMR loader builds it
// (make_poisson_rhs of the per-field brick) and calls this for the block that declares the field. We
// stash it per (field, block name); build_multi resolves the block index against the runtime and attaches
// it. POPS_EXPORT: resolved by the generated AMR .so loader across the dlopen boundary.
POPS_EXPORT void AmrSystem::set_block_elliptic_field(
    const std::string& block_name, const std::string& field,
    std::function<void(const MultiFab&, MultiFab&)> rhs) {
  require_assembling_amr(p_->bound_,
                         "set_block_elliptic_field");  // frozen once pops.bind completes (ADC-592)
  if (p_->built)
    throw std::runtime_error("AmrSystem::set_block_elliptic_field : the system is already built");
  p_->ell_field_rhs_[field][block_name] = std::move(rhs);
}

// Solved potential of a named elliptic field on the coarse level, ny*nx row-major.
// Builds the hierarchy on first call (ensure_built) then reads from the AmrRuntime engine, which solves
// the fields (counterpart of potential()).
std::vector<double> AmrSystem::named_field_values(const std::string& field) {
  p_->ensure_built();
  return p_->runtime->named_field_values(field);
}

std::vector<std::string> AmrSystem::field_provider_slots() const {
  const_cast<Impl*>(p_.get())->ensure_built();
  return p_->runtime->provider_slots();
}

void AmrSystem::set_field_potential(const std::string& provider_slot,
                                    const std::vector<double>& phi) {
  set_field_potential_level(provider_slot, 0, phi);
}

int AmrSystem::field_provider_levels(const std::string& provider_slot) {
  p_->ensure_built();
  if (!p_->runtime)
    throw std::runtime_error(
        "AmrSystem::field_provider_levels requires the qualified field runtime");
  return p_->runtime->provider_potential_levels(provider_slot);
}

void AmrSystem::set_field_potential_level(const std::string& provider_slot, int level,
                                          const std::vector<double>& phi) {
  p_->ensure_built();
  if (!p_->runtime)
    throw std::runtime_error(
        "AmrSystem::set_field_potential_level requires the qualified field runtime");
  if (level < 0 || level >= p_->runtime->provider_potential_levels(provider_slot))
    throw std::out_of_range("AmrSystem::set_field_potential_level level out of range");
  const int width = p_->cfg.n << level;
  const int height = p_->cfg.ny << level;
  const std::size_t expected = static_cast<std::size_t>(width) * height;
  if (phi.size() != expected)
    throw std::runtime_error("AmrSystem::set_field_potential_level layout size mismatch");
  MultiFab& target = p_->runtime->provider_potential_level(provider_slot, level);
  // This is a host-side restore into a field that may have been touched by a Kokkos solve.
  // Synchronize the host residence before exposing Array4 write handles; the next native solve
  // will make the device residence current through its normal execution-space transition.
  target.sync_host();
  for (int li = 0; li < target.local_size(); ++li) {
    Array4 out = target.fab(li).array();
    const Box2D valid = target.box(li);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        out(i, j, 0) = static_cast<Real>(
            phi[static_cast<std::size_t>(j) * width + static_cast<std::size_t>(i)]);
  }
}

std::vector<double> AmrSystem::field_potential_global(const std::string& provider_slot) {
  return field_potential_level_global(provider_slot, 0);
}

std::vector<double> AmrSystem::field_potential_level_global(const std::string& provider_slot,
                                                            int level) {
  p_->ensure_built();
  if (!p_->runtime)
    throw std::runtime_error(
        "AmrSystem::field_potential_global requires the qualified field runtime");
  if (level < 0 || level >= p_->runtime->provider_potential_levels(provider_slot))
    throw std::out_of_range("AmrSystem::field_potential_level_global level out of range");
  const int width = p_->cfg.n << level;
  const int height = p_->cfg.ny << level;
  MultiFab& source = p_->runtime->provider_potential_level(provider_slot, level);
  // The provider potential may have been produced by a Kokkos kernel.  This accessor is a
  // host-side global snapshot (it copies into a std::vector before the MPI reduction), so make
  // the host residence authoritative before reading any Array4 values.  The compact output-piece
  // route performs the same synchronization in output_local_pieces().
  source.sync_host();
  std::vector<double> result(static_cast<std::size_t>(width) * height, 0.0);
  for (int li = 0; li < source.local_size(); ++li) {
    const ConstArray4 values = source.fab(li).const_array();
    const Box2D valid = source.box(li);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        result[static_cast<std::size_t>(j) * width + static_cast<std::size_t>(i)] =
            static_cast<double>(values(i, j, 0));
  }
  // Level 0 follows the authored coarse ownership policy; every fine level is distributed over
  // its patch mapping even when the coarse level is replicated.
  if (level > 0 || p_->cfg.distribute_coarse)
    all_reduce_sum_inplace(result.data(), result.size());
  return result;
}

std::vector<OutputPiece> AmrSystem::output_field_local_pieces(const std::string& provider_slot,
                                                              int level) {
  p_->ensure_built();
  if (!p_->runtime)
    throw std::runtime_error(
        "AmrSystem::output_field_local_pieces requires the qualified field runtime");
  return p_->runtime->output_field_local_pieces(provider_slot, level);
}

std::vector<OutputPiece> AmrSystem::output_field_root_pieces(const WorldCommunicator& world,
                                                             const std::string& provider_slot,
                                                             int level) {
  return output_pieces_to_root(
      world, detail::output_collective_identity("AmrSystem", "field", provider_slot, level),
      [&] { return output_field_local_pieces(provider_slot, level); });
}

namespace {
// Module anchor for dladdr: its ADDRESS lives in the image that contains amr_system.cpp (the _pops
// module, or the test binary). add_native_block uses it to locate the module and promote it to
// global scope (RTLD_NOLOAD). A TU-local function suffices (no need to export) and avoids
// depending on a symbol defined elsewhere (pops::abi_key, system.cpp).
void amr_native_anchor() {}

int compiled_param_name_count(const char* raw) {
  if (raw == nullptr || *raw == '\0')
    return 0;
  int count = 1;
  for (const char* cursor = raw; *cursor != '\0'; ++cursor)
    if (*cursor == ',')
      ++count;
  return count;
}

void verify_amr_package(pops::dynlib::handle handle, const std::vector<double>& params) {
  constexpr const char* context = "AmrSystem::_install_native_block";
  auto manifest = reinterpret_cast<const char* (*)()>(
      pops::dynlib::sym(handle, "pops_compiled_route_manifest"));
  auto count = reinterpret_cast<int (*)()>(pops::dynlib::sym(handle, "pops_compiled_nparams"));
  auto names =
      reinterpret_cast<const char* (*)()>(pops::dynlib::sym(handle, "pops_compiled_param_names"));
  if (manifest == nullptr || count == nullptr || names == nullptr) {
    pops::dynlib::close(handle);
    throw std::runtime_error(std::string(context) +
                             ": strict package metadata is missing; rebuild artifact");
  }
  const char* route_manifest = manifest();
  const char* param_names = names();
  if (route_manifest == nullptr || param_names == nullptr) {
    pops::dynlib::close(handle);
    throw std::runtime_error(std::string(context) + ": package metadata returned null");
  }
  try {
    pops::verify_route_manifest(route_manifest, context);
  } catch (...) {
    pops::dynlib::close(handle);
    throw;
  }
  const int expected = count();
  if (expected < 0 || expected > kMaxRuntimeParams ||
      compiled_param_name_count(param_names) != expected ||
      params.size() != static_cast<std::size_t>(expected)) {
    pops::dynlib::close(handle);
    throw std::runtime_error(std::string(context) +
                             ": bound parameter vector disagrees with package metadata");
  }
}
}  // namespace

void AmrSystem::add_native_block(const std::string& name, const std::string& so_path,
                                 const std::string& limiter, const std::string& riemann,
                                 const std::string& recon, const std::string& time, double gamma,
                                 int substeps, const std::vector<double>& params,
                                 double positivity_floor, double weno_epsilon,
                                 bool wave_speed_cache) {
  require_assembling_amr(p_->bound_,
                         "add_native_block");  // frozen once pops.bind completes (ADC-592)
  if (substeps < 1)
    throw std::runtime_error("AmrSystem::add_native_block : substeps >= 1");
  // Zhang-Shu positivity floor (ADC-322): eager validation (parity with add_block). 0 = inactive,
  // bit-identical. Marshaled down to the loader (pops_install_native_amr) -> add_compiled_model; an
  // older .so (no floor slot) ignores it, so a non-zero floor on such a loader is a silent no-op.
  if (!(positivity_floor >= 0.0) || !std::isfinite(positivity_floor))
    throw std::runtime_error(
        "AmrSystem::add_native_block : positivity_floor >= 0 and finite (0 = inactive)");
  if (!std::isfinite(weno_epsilon) || weno_epsilon <= 0.0)
    throw std::runtime_error("AmrSystem::add_native_block : finite weno_epsilon > 0 required");
  if (weno_epsilon != static_cast<double>(kWenoEpsilon) && limiter != "weno5")
    throw std::runtime_error(
        "AmrSystem::add_native_block : weno_epsilon applies to limiter='weno5' only");
  if (wave_speed_cache && riemann != "hll")
    throw std::runtime_error(
        "AmrSystem::add_native_block : wave_speed_cache requires riemann='hll'");
  if (wave_speed_cache && time == "imex")
    throw std::runtime_error(
        "AmrSystem::add_native_block : wave_speed_cache is supported by explicit AMR transport "
        "only");
  // UPSTREAM scheme validation (like add_block): add_compiled_model(AmrSystem&) rejects unknown
  // time routes and recon outside {conservative, primitive}, but we diagnose HERE a
  // typo before the C++ boundary. The installed Program lowers time == "imex" into its explicit and
  // implicit stages. limiter (including weno5, wired #105) and riemann (including hllc/roe, wired at
  // parity #113) are validated by the compiled spatial builder in the loader (clear exception).
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error(
        "AmrSystem::add_native_block : recon 'conservative' | 'primitive' "
        "(got '" +
        recon + "')");
  // The flat loader ABI marshals the canonical time STRING for Program normalization and inspection;
  // the spatial block builder does not decode it into a second time method.
  if (time != "explicit" && time != "euler" && time != "ssprk3" && time != "imex")
    throw std::runtime_error("AmrSystem::add_native_block : time must be one of " +
                             amr_time_routes_csv() + " (got '" + time + "')");
  // DSL "production" path on the AMR side: the generated .so loader (emit_cpp_native_loader with
  // target="amr_system") inlines the header template add_compiled_model(AmrSystem&, ...), which
  // materializes a concrete AmrRuntimeBlock at lazy build and installs its Program primitives via
  // set_compiled_block on the same hierarchy as add_block. The loader thus calls set_compiled_block
  // (out-of-line method of pops::AmrSystem) DEFINED in THIS module;
  // it must be resolved through the dlopen against the already-loaded _pops module.
  // ELF PORTABILITY (Linux): CPython loads _pops with RTLD_LOCAL, so its symbols are NOT in
  // the global scope. We PROMOTE the current module to global scope (RTLD_NOLOAD = without
  // reloading it; RTLD_GLOBAL OR'd into the flags of the already-loaded object), located by dladdr on
  // an ADDRESS of THIS module: amr_native_anchor (TU-local function). We thus avoid depending
  // on pops::abi_key (defined in system.cpp) -- which would link-couple any test compiling
  // amr_system.cpp alone. On macOS, harmless (the loader resolves via dynamic_lookup).
#if defined(_WIN32)
  // Windows (ADC-100): no RTLD_GLOBAL. The generated AMR .dll is linked against _pops.lib (symbol
  // AmrSystem::set_compiled_block POPS_EXPORT) + kokkoscore.lib (shared Kokkos). Undefined symbols resolved
  // by the OS loader against the already-loaded _pops.pyd + kokkos*.dll. We load + resolve pops_install_native_amr.
  pops::dynlib::handle h = pops::dynlib::open(so_path);
  if (!h)
    throw std::runtime_error("AmrSystem::add_native_block : LoadLibrary('" + so_path +
                             "') : " + pops::dynlib::last_error() +
                             " (.dll linked against _pops.lib + kokkoscore.lib ; cf. ADC-100)");
  {
    auto key_fn = reinterpret_cast<const char* (*)()>(pops::dynlib::sym(h, "pops_native_abi_key"));
    if (!key_fn) {
      pops::dynlib::close(h);
      throw std::runtime_error(
          "AmrSystem::add_native_block : pops_native_abi_key missing from the .dll");
    }
    const std::string loader_key = key_fn();
    const std::string module_key = detail::abi_key_string();
    if (loader_key != module_key) {
      pops::dynlib::close(h);
      throw std::runtime_error("AmrSystem::add_native_block : incompatible ABI -- loader '" +
                               loader_key + "' != module '" + module_key + "'");
    }
    verify_amr_package(h, params);
    using install_fn_t =
        void (*)(void*, const char*, const char*, const char*, const char*, const char*, double,
                 int, const double*, int, double, double, bool);
    auto install = reinterpret_cast<install_fn_t>(pops::dynlib::sym(h, "pops_install_native_amr"));
    if (!install) {
      pops::dynlib::close(h);
      throw std::runtime_error(
          "AmrSystem::add_native_block : pops_install_native_amr missing from the .dll");
    }
    install(static_cast<void*>(this), name.c_str(), limiter.c_str(), riemann.c_str(), recon.c_str(),
            time.c_str(), gamma, substeps, params.empty() ? nullptr : params.data(),
            static_cast<int>(params.size()), positivity_floor, weno_epsilon, wave_speed_cache);
  }
#else
  {
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&amr_native_anchor), &info) && info.dli_fname)
      dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
  }
  // Only the host image is promoted globally. The generated package imports set_compiled_block
  // from that host but remains RTLD_LOCAL, preventing identically named generated templates from
  // different semantic artifacts from interposing on one another under ELF.
  pops::dynlib::handle h = pops::dynlib::open(so_path);
  if (!h) {
    throw std::runtime_error(
        "AmrSystem::add_native_block : dlopen('" + so_path + "') : " + pops::dynlib::last_error() +
        " (the symbol pops::AmrSystem::set_compiled_block must be exported AND the "
        "_pops module loaded globally ; cf. POPS_EXPORT)");
  }
  // EXPLICIT ABI GUARD: the key baked into the loader (at ITS compilation) must equal the module's
  // key. A mismatch = divergent headers / compiler / standard -> potentially different memory layout
  // of AmrSystem/AmrBuildParams/AmrCompiledBlockBuilder at the boundary -> UB. We raise
  // a CLEAR error rather than let an incompatible loader through. SAME key symbol as the
  // System path (pops_native_abi_key): only the installer (pops_install_native_amr) differs.
  auto key_fn = reinterpret_cast<const char* (*)()>(dlsym(h, "pops_native_abi_key"));
  if (!key_fn) {
    dlclose(h);
    throw std::runtime_error(
        "AmrSystem::add_native_block : pops_native_abi_key missing from the .so "
        "(regenerate via dsl.compile_native(target='amr_system') / "
        "compile(backend='production', target='amr_system'))");
  }
  const std::string loader_key = key_fn();
  // Module key = SAME computation as pops::abi_key() (header-only detail::abi_key_string()): avoids the
  // dependency on the out-of-line symbol pops::abi_key (system.cpp). The loader bakes its own at ITS compile.
  const std::string module_key = detail::abi_key_string();
  if (loader_key != module_key) {
    dlclose(h);
    throw std::runtime_error("AmrSystem::add_native_block : incompatible ABI -- loader key '" +
                             loader_key + "' != module key '" + module_key +
                             "'. Recompile the loader with the SAME compiler, C++ standard and "
                             "pops headers as the _pops module.");
  }
  verify_amr_package(h, params);
  // AMR native installer of the loader: reinterpret_cast<AmrSystem*>(this) then
  // add_compiled_model<ProdModel>(*amrsys, ...). Scheme marshaled as flat extern "C" arguments. No
  // evolve parameter (single-block AMR, no frozen background block like System). DISTINCT SYMBOL
  // (pops_install_native_amr, vs pops_install_native on the System side): a loader generated for System
  // does NOT export it, so wiring it here fails clearly instead of an inconsistent cast. The trailing
  // double is the Zhang-Shu positivity floor (ADC-322): old 8-argument loaders carry an ABI key from
  // the pre-floor headers and are REJECTED above, so the 9-argument call never reaches a stale .so.
  using install_fn_t = void (*)(void*, const char*, const char*, const char*, const char*,
                                const char*, double, int, const double*, int, double, double, bool);
  auto install = reinterpret_cast<install_fn_t>(dlsym(h, "pops_install_native_amr"));
  if (!install) {
    dlclose(h);
    throw std::runtime_error(
        "AmrSystem::add_native_block : pops_install_native_amr missing from the .so "
        "(loader generated for System, or regenerate via "
        "dsl.compile_native(target='amr_system'))");
  }
  install(static_cast<void*>(this), name.c_str(), limiter.c_str(), riemann.c_str(), recon.c_str(),
          time.c_str(), gamma, substeps, params.empty() ? nullptr : params.data(),
          static_cast<int>(params.size()), positivity_floor, weno_epsilon, wave_speed_cache);
  // The local .so stays loaded for the duration of the process: the type-erasing builder installed
  // by set_compiled_block captures code (header template) that lives there. We do NOT close it.
#endif  // _WIN32 (production AMR POSIX-only; Windows = throw, ADC-100)
  const int installed_idx = p_->block_index(name);
  if (installed_idx >= 0) {
    Impl::BlockSpec& b = p_->blocks[static_cast<std::size_t>(installed_idx)];
    b.limiter = limiter;
    b.riemann = riemann;
    b.recon_prim = (recon == "primitive");
    b.time = time;
    b.gamma = gamma;
    b.substeps = substeps;
    b.pos_floor = positivity_floor;
    b.weno_epsilon = weno_epsilon;
    b.wave_speed_cache = wave_speed_cache;
  }
}

void AmrSystem::add_external_riemann_block(const std::string& name, const std::string& so_path,
                                           const std::string& brick_id, const std::string& sha256,
                                           const std::string& limiter, const std::string& recon,
                                           const std::string& time, double gamma, int substeps,
                                           int stride, int expected_nvars, int expected_naux,
                                           const std::string& expected_model_identity,
                                           double positivity_floor, double weno_epsilon) {
  require_assembling_amr(p_->bound_, "add_external_riemann_block");
  auto library = std::make_shared<runtime::program::ExternalBrickHandle>(
      so_path, brick_id, sha256, expected_nvars, expected_naux, expected_model_identity);
  p_->external_riemann_libraries_.push_back(library);
  library->install_amr(this, name, limiter, recon, time, gamma, substeps, stride, positivity_floor,
                       weno_epsilon);
  const int installed_idx = p_->block_index(name);
  if (installed_idx >= 0) {
    Impl::BlockSpec& block = p_->blocks[static_cast<std::size_t>(installed_idx)];
    block.limiter = limiter;
    block.riemann = brick_id;
  }
}

void AmrSystem::set_bootstrap_tagging(
    const std::vector<std::string>& leaf_subject_kinds,
    const std::vector<std::string>& leaf_subject_identities,
    const std::vector<std::string>& leaf_blocks, const std::vector<std::string>& leaf_variables,
    const std::vector<int>& leaf_field_component_indices, const std::vector<int>& leaf_ops,
    const std::vector<double>& leaf_thresholds, const std::vector<int>& leaf_stencil_indices,
    const std::vector<runtime::amr::PreparedTaggingProgram::Stencil>& stencils,
    const std::vector<std::int32_t>& refine_ops, const std::vector<std::int32_t>& refine_args,
    const std::vector<std::int32_t>& coarsen_ops, const std::vector<std::int32_t>& coarsen_args,
    int min_cycles, const std::string& equality_policy, const std::string& conflict_policy,
    const std::string& clock_identity, const std::string& provider_identity) {
  require_assembling_amr(p_->bound_, "set_bootstrap_tagging");
  const std::size_t leaf_count = leaf_subject_kinds.size();
  if (p_->built || p_->tagging_spec || leaf_count == 0 ||
      leaf_subject_identities.size() != leaf_count || leaf_blocks.size() != leaf_count ||
      leaf_variables.size() != leaf_count || leaf_field_component_indices.size() != leaf_count ||
      leaf_ops.size() != leaf_count || leaf_thresholds.size() != leaf_count ||
      leaf_stencil_indices.size() != leaf_count || refine_ops.empty() ||
      refine_ops.size() != refine_args.size() || coarsen_ops.size() != coarsen_args.size() ||
      min_cycles < 0 || clock_identity.empty() || provider_identity.empty())
    throw std::runtime_error(
        "AmrSystem::set_bootstrap_tagging requires one exact pre-build graph manifest");
  if (std::any_of(leaf_subject_identities.begin(), leaf_subject_identities.end(),
                  [](const std::string& value) { return value.empty(); }) ||
      std::any_of(leaf_variables.begin(), leaf_variables.end(),
                  [](const std::string& value) { return value.empty(); }) ||
      std::any_of(leaf_thresholds.begin(), leaf_thresholds.end(),
                  [](double value) { return !std::isfinite(value); }))
    throw std::runtime_error("AmrSystem::set_bootstrap_tagging has an invalid leaf");
  for (std::size_t index = 0; index < leaf_count; ++index) {
    const std::string& kind = leaf_subject_kinds[index];
    const bool state = kind == "state";
    const bool field = kind == "field";
    const bool aux = kind == "aux";
    if ((!state && !field && !aux) || (state != !leaf_blocks[index].empty()) ||
        (field != (leaf_field_component_indices[index] >= 0)) ||
        (!field && leaf_field_component_indices[index] != -1))
      throw std::runtime_error(
          "AmrSystem::set_bootstrap_tagging has an invalid qualified subject route");
  }
  const auto equality = equality_policy == "hold"      ? 0
                        : equality_policy == "refine"  ? 1
                        : equality_policy == "coarsen" ? 2
                                                       : -1;
  const auto conflict = conflict_policy == "error"          ? 0
                        : conflict_policy == "hold"         ? 1
                        : conflict_policy == "refine_wins"  ? 2
                        : conflict_policy == "coarsen_wins" ? 3
                                                            : -1;
  if (equality < 0 || conflict < 0)
    throw std::runtime_error("AmrSystem::set_bootstrap_tagging has an unknown policy");
  p_->tagging_spec = std::make_unique<Impl::TaggingSpec>(Impl::TaggingSpec{
      leaf_subject_kinds, leaf_subject_identities, leaf_blocks, leaf_variables,
      leaf_field_component_indices, leaf_ops, leaf_thresholds, leaf_stencil_indices, stencils,
      refine_ops, refine_args, coarsen_ops, coarsen_args, min_cycles, equality, conflict,
      clock_identity, provider_identity});
}

void AmrSystem::set_temporal_relations(const std::vector<std::int64_t>& numerators,
                                       const std::vector<std::int64_t>& denominators,
                                       const std::vector<std::string>& remainder_policies) {
  require_assembling_amr(p_->bound_, "set_temporal_relations");
  if (p_->built || numerators.size() != denominators.size() ||
      numerators.size() != remainder_policies.size() ||
      numerators.size() != static_cast<std::size_t>(p_->cfg.level_count - 1))
    throw std::runtime_error(
        "AmrSystem::set_temporal_relations requires one exact pre-build relation per transition");
  std::vector<::pops::amr::ParentChildClockRelation> relations;
  relations.reserve(numerators.size());
  for (std::size_t index = 0; index < numerators.size(); ++index) {
    const auto policy =
        remainder_policies[index] == "integral_only" ? ::pops::amr::RemainderPolicy::IntegralOnly
        : remainder_policies[index] == "explicit_final_substep"
            ? ::pops::amr::RemainderPolicy::ExplicitFinalSubstep
            : throw std::runtime_error(
                  "AmrSystem::set_temporal_relations has an unknown remainder policy");
    relations.emplace_back(static_cast<int>(index), static_cast<int>(index + 1),
                           ::pops::amr::Rational(numerators[index], denominators[index]), policy);
  }
  p_->temporal_relations_ = std::move(relations);
}

void AmrSystem::set_poisson(const std::string& rhs, const std::string& solver,
                            const std::string& bc, const std::string& wall, double wall_radius,
                            const AmrFieldSolverOptions& solver_options) {
  require_assembling_amr(p_->bound_, "set_poisson");  // frozen once pops.bind completes (ADC-592)
  if (p_->built)
    throw std::runtime_error("AmrSystem::set_poisson : the system is already built");
  if (rhs != "charge_density" && rhs != "composite")
    throw std::runtime_error("AmrSystem::set_poisson : unknown rhs '" + rhs +
                             "' (valid: " + kPoissonRhsRouteTokensCsv +
                             "; the right-hand side = sum of the "
                             "block's elliptic bricks)");
  const auto field_provider = p_->field_solver_registry_->resolve(solver);
  AmrFieldSolverOptions provider_options = solver_options.schema_identity.empty()
                                               ? field_provider->default_field_options()
                                               : solver_options;
  const PreparedProviderSupport option_support = field_provider->accepts_options(provider_options);
  if (!option_support.well_formed())
    throw std::runtime_error(
        "AmrSystem::set_poisson : provider returned a malformed option decision");
  if (!option_support.accepted())
    throw std::runtime_error("AmrSystem::set_poisson : provider rejected its field options (code " +
                             std::to_string(option_support.code) +
                             "): " + std::string(option_support.reason));
  (void)provider_options.exact_contract();
  p_->p_rhs = rhs;
  p_->p_solver = solver;
  p_->p_solver_options = std::move(provider_options);
  p_->p_bc = bc;
  p_->p_wall = wall;
  p_->p_wall_radius = wall_radius;
}

void AmrSystem::set_field_solver_plan(
    const std::string& provider_slot, const std::string& plan_identity,
    const std::string& provider_identity, const std::string& output_owner_identity,
    const std::string& output_block, const std::string& output_key,
    const std::vector<std::string>& provider_identities,
    const std::vector<std::string>& provider_blocks, const std::vector<std::string>& provider_keys,
    const std::vector<double>& provider_coefficients, const std::string& solver,
    const AmrFieldHierarchyPolicyAuthority& hierarchy_policy,
    const AmrFieldSolverOptions& solver_options) {
  require_assembling_amr(p_->bound_, "set_field_solver_plan");
  if (p_->built)
    throw std::runtime_error("AmrSystem::set_field_solver_plan: system already built");
  if (provider_slot.empty() || plan_identity.empty() || provider_identity.empty() ||
      output_owner_identity.empty() || output_block.empty() || output_key.empty())
    throw std::runtime_error(
        "AmrSystem::set_field_solver_plan requires qualified plan/provider identities");
  const std::size_t provider_count = provider_identities.size();
  if (provider_count == 0 || provider_blocks.size() != provider_count ||
      provider_keys.size() != provider_count || provider_coefficients.size() != provider_count)
    throw std::runtime_error("AmrSystem::set_field_solver_plan invalid provider-pack shape");
  const auto finite_native_real = [](double value) {
    if (!std::isfinite(value))
      return false;
    return std::isfinite(static_cast<double>(static_cast<Real>(value)));
  };
  for (std::size_t i = 0; i < provider_count; ++i)
    if (provider_identities[i].empty() || provider_blocks[i].empty() || provider_keys[i].empty() ||
        !finite_native_real(provider_coefficients[i]))
      throw std::runtime_error("AmrSystem::set_field_solver_plan invalid provider-pack entry");
  // Resolve through the same per-system registry used at native materialization.  Provider-specific
  // hierarchy policies remain opaque here and are validated by the provider against the exact build
  // request; the facade never grows a switch when an extension adds one.
  const auto resolved_provider = p_->field_solver_registry_->resolve(solver);
  const PreparedProviderSupport option_support = resolved_provider->accepts_options(solver_options);
  if (!option_support.well_formed())
    throw std::runtime_error(
        "AmrSystem::set_field_solver_plan provider returned a malformed option decision");
  if (!option_support.accepted())
    throw std::runtime_error(
        "AmrSystem::set_field_solver_plan provider rejected its options (code " +
        std::to_string(option_support.code) + "): " + std::string(option_support.reason));
  try {
    hierarchy_policy.validate();
  } catch (...) {
    throw std::runtime_error(
        "AmrSystem::set_field_solver_plan received an invalid hierarchy-policy authority");
  }
  (void)solver_options.exact_contract();
  const auto existing = p_->field_plans_.find(provider_slot);
  if (existing != p_->field_plans_.end())
    throw std::runtime_error("AmrSystem::set_field_solver_plan duplicate provider slot");
  AmrFieldSolveConfig plan;
  plan.plan_identity = plan_identity;
  plan.provider_identity = provider_identity;
  plan.output_owner_identity = output_owner_identity;
  plan.output_block = output_block;
  plan.output_key = output_key;
  plan.providers.clear();
  plan.providers.reserve(provider_count);
  for (std::size_t i = 0; i < provider_count; ++i)
    plan.providers.push_back({provider_identities[i], provider_blocks[i], provider_keys[i],
                              static_cast<Real>(provider_coefficients[i])});
  plan.solver = solver;
  plan.hierarchy_policy = hierarchy_policy;
  plan.solver_options = solver_options;
  const bool unique = p_->field_plans_.emplace(provider_slot, std::move(plan)).second;
  if (!unique)
    throw std::runtime_error("AmrSystem::set_field_solver_plan duplicate provider slot");
  p_->field_plan_consensus_verified_ = false;
}

void AmrSystem::register_field_solver_provider(
    std::shared_ptr<const AmrFieldSolverProvider> provider) {
  require_assembling_amr(p_->bound_, "register_field_solver_provider");
  if (p_->built)
    throw std::runtime_error("AmrSystem::register_field_solver_provider: system already built");
  p_->field_solver_registry_->add(std::move(provider));
  p_->field_plan_consensus_verified_ = false;
}

void AmrSystem::register_field_nullspace_provider(
    std::shared_ptr<const FieldNullspaceProvider> provider) {
  require_assembling_amr(p_->bound_, "register_field_nullspace_provider");
  if (p_->built)
    throw std::runtime_error("AmrSystem::register_field_nullspace_provider: system already built");
  p_->field_nullspace_provider_registry_->add(std::move(provider));
  p_->field_plan_consensus_verified_ = false;
}

void AmrSystem::set_default_field_nullspace(const std::string& nullspace_provider_identity,
                                            const PreparedProviderOptions& options) {
  require_assembling_amr(p_->bound_, "set_default_field_nullspace");
  if (p_->built)
    throw std::runtime_error("AmrSystem::set_default_field_nullspace: system already built");
  FieldNullspaceProviderSelection selection{nullspace_provider_identity, options};
  (void)selection.exact_contract();
  p_->default_field_nullspace_ = std::move(selection);
  p_->field_plan_consensus_verified_ = false;
}

void AmrSystem::register_hierarchy_tensor_solver_provider(
    std::shared_ptr<const runtime::program::HierarchyTensorSolverProvider> provider) {
  require_assembling_amr(p_->bound_, "register_hierarchy_tensor_solver_provider");
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::register_hierarchy_tensor_solver_provider: system already built");
  p_->hierarchy_tensor_solver_provider_registry_->add(std::move(provider));
}

void AmrSystem::register_program_hierarchy_tensor_solver_provider(
    std::shared_ptr<const runtime::program::HierarchyTensorSolverProvider> provider) {
  p_->hierarchy_tensor_solver_provider_registry_->add_collectively(std::move(provider));
}

std::shared_ptr<const runtime::program::HierarchyTensorSolverProviderRegistry>
AmrSystem::hierarchy_tensor_solver_provider_registry() const {
  return p_->hierarchy_tensor_solver_provider_registry_;
}

AmrFieldSolverConfiguration AmrSystem::field_solver_configuration(
    const std::string& provider_slot) const {
  const auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("AmrSystem::field_solver_configuration: unknown field provider slot");
  const auto& plan = found->second;
  return {plan.plan_identity, plan.provider_identity, plan.solver, plan.hierarchy_policy,
          plan.solver_options};
}

void AmrSystem::set_field_reaction(const std::string& provider_slot, double reaction) {
  require_assembling_amr(p_->bound_, "set_field_reaction");
  if (p_->built)
    throw std::runtime_error("AmrSystem::set_field_reaction: system already built");
  if (!std::isfinite(reaction) || reaction <= 0.0)
    throw std::runtime_error(
        "AmrSystem::set_field_reaction requires a finite, strictly positive coefficient");
  auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("AmrSystem::set_field_reaction names an unknown provider slot");
  found->second.has_reaction = true;
  found->second.reaction = static_cast<Real>(reaction);
  p_->field_plan_consensus_verified_ = false;
}

void AmrSystem::set_field_topology_authority(const std::string& provider_slot,
                                             const std::string& provider_kind,
                                             const std::string& provenance,
                                             const std::string& topology_digest) {
  require_assembling_amr(p_->bound_, "set_field_topology_authority");
  const auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("AmrSystem::set_field_topology_authority unknown provider slot");
  if (provider_kind.empty() || provenance.empty() || topology_digest.empty())
    throw std::runtime_error(
        "AmrSystem::set_field_topology_authority requires complete topology provenance");
  found->second.topology_provider_kind = provider_kind;
  found->second.topology_provenance = provenance;
  found->second.topology_digest = topology_digest;
  p_->field_plan_consensus_verified_ = false;
}

std::vector<runtime::field::FieldTopologyReportRow> AmrSystem::field_topology_report(
    const std::string& provider_slot) const {
  const auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("AmrSystem::field_topology_report unknown provider slot");
  const auto& plan = found->second;
  if (plan.topology_provider_kind.empty() || plan.topology_provenance.empty() ||
      plan.topology_digest.empty())
    throw std::runtime_error("AmrSystem::field_topology_report topology authority is incomplete");
  // A declared recipe is not a materialized topology.  Field plans force the AmrRuntime route,
  // whose named solver is allocated lazily on the first solve and invalidated on every regrid.
  // Until then the only honest report is an empty one.
  if (!p_->built || !p_->runtime)
    return {};
  const auto patches = p_->runtime->field_topology_patches(provider_slot);
  if (!patches)
    return {};
  std::vector<runtime::field::FieldTopologyReportRow> report;
  report.reserve(patches->size());
  for (const PatchBox& patch : *patches) {
    const auto nx = static_cast<std::size_t>(patch.ihi - patch.ilo + 1);
    const auto ny = static_cast<std::size_t>(patch.jhi - patch.jlo + 1);
    if (nx != 0 && ny > std::numeric_limits<std::size_t>::max() / nx)
      throw std::overflow_error("AMR field topology material-point count overflow");
    report.push_back({"builtin:hierarchy/level=" + std::to_string(patch.level) + "/box=[" +
                          std::to_string(patch.ilo) + "," + std::to_string(patch.jlo) + "]-[" +
                          std::to_string(patch.ihi) + "," + std::to_string(patch.jhi) + "]",
                      plan.topology_digest, plan.topology_provenance, nx * ny, 1});
  }
  return report;
}

void AmrSystem::set_field_boundary_plan(const std::string& provider_slot,
                                        const std::vector<std::string>& kind,
                                        const std::vector<double>& alpha,
                                        const std::vector<double>& beta,
                                        const std::vector<double>& value) {
  require_assembling_amr(p_->bound_, "set_field_boundary_plan");
  auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("AmrSystem::set_field_boundary_plan unknown provider slot");
  if (kind.size() != 4 || alpha.size() != 4 || beta.size() != 4 || value.size() != 4)
    throw std::runtime_error(
        "AmrSystem::set_field_boundary_plan requires four xlo/xhi/ylo/yhi entries");
  BCRec bc;
  bc.dx = static_cast<Real>(p_->cfg.L / p_->cfg.n);
  bc.dy = static_cast<Real>(p_->cfg.Ly / p_->cfg.ny);
  BCType* types[] = {&bc.xlo, &bc.xhi, &bc.ylo, &bc.yhi};
  Real* vals[] = {&bc.xlo_val, &bc.xhi_val, &bc.ylo_val, &bc.yhi_val};
  Real* alphas[] = {&bc.xlo_alpha, &bc.xhi_alpha, &bc.ylo_alpha, &bc.yhi_alpha};
  Real* betas[] = {&bc.xlo_beta, &bc.xhi_beta, &bc.ylo_beta, &bc.yhi_beta};
  for (int face = 0; face < 4; ++face) {
    const Real a = static_cast<Real>(alpha[face]);
    const Real b = static_cast<Real>(beta[face]);
    const Real v = static_cast<Real>(value[face]);
    if (!std::isfinite(alpha[face]) || !std::isfinite(beta[face]) || !std::isfinite(value[face]) ||
        (a == Real(0) && b == Real(0) && kind[face] != "periodic"))
      throw std::runtime_error("AmrSystem::set_field_boundary_plan invalid Robin coefficients");
    if (kind[face] == "periodic") {
      *types[face] = BCType::Periodic;
    } else if (kind[face] == "dirichlet" || (kind[face] == "mixed" && b == Real(0))) {
      if (a == Real(0))
        throw std::runtime_error("AmrSystem::set_field_boundary_plan Dirichlet alpha is zero");
      *types[face] = BCType::Dirichlet;
      *vals[face] = v / a;
    } else if (kind[face] == "neumann" && v == Real(0)) {
      *types[face] = BCType::Foextrap;
    } else if (kind[face] == "neumann" || kind[face] == "mixed") {
      *types[face] = BCType::Robin;
      *vals[face] = v;
      *alphas[face] = a;
      *betas[face] = b;
      const Real h = face < 2 ? bc.dx : bc.dy;
      if (a / Real(2) + b / h == Real(0))
        throw std::runtime_error(
            "AmrSystem::set_field_boundary_plan singular cell-centred Robin denominator");
    } else {
      throw std::runtime_error("AmrSystem::set_field_boundary_plan unknown kind '" + kind[face] +
                               "'");
    }
  }
  found->second.explicit_bc = bc;
  found->second.has_explicit_bc = true;
  p_->field_plan_consensus_verified_ = false;
}

void AmrSystem::set_field_boundary_kernel(const std::string& provider_slot,
                                          const CompiledFieldBoundaryKernel& kernel) {
  kernel.validate();
  auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("AmrSystem::set_field_boundary_kernel unknown provider slot");
  found->second.boundary_kernel = kernel;
  found->second.has_boundary_kernel = true;
  if (!kernel.observes_iteration)
    found->second.boundary_context.point.iteration = 0;
  if (p_->runtime)
    p_->runtime->set_field_boundary_kernel(provider_slot, kernel);
  p_->field_plan_consensus_verified_ = false;
}

void AmrSystem::set_field_boundary_dependencies(const std::string& provider_slot,
                                                const std::vector<std::string>& state_blocks,
                                                const std::vector<int>& state_components,
                                                const std::vector<std::string>& field_blocks,
                                                const std::vector<std::string>& field_keys,
                                                const std::vector<int>& field_components) {
  require_assembling_amr(p_->bound_, "set_field_boundary_dependencies");
  if (state_blocks.size() != state_components.size() || !field_blocks.empty() ||
      !field_keys.empty() || !field_components.empty())
    throw std::runtime_error(
        "AmrSystem::set_field_boundary_dependencies accepts exact state buffers only");
  if (std::any_of(state_blocks.begin(), state_blocks.end(),
                  [](const auto& value) { return value.empty(); }) ||
      std::any_of(state_components.begin(), state_components.end(),
                  [](int value) { return value < 0; }))
    throw std::runtime_error("AmrSystem::set_field_boundary_dependencies contains invalid entries");
  auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("AmrSystem::set_field_boundary_dependencies unknown provider slot");
  found->second.boundary_state_blocks = state_blocks;
  found->second.boundary_state_components = state_components;
  if (p_->runtime)
    p_->runtime->set_field_boundary_dependencies(provider_slot, state_blocks, state_components);
  p_->field_plan_consensus_verified_ = false;
}

void AmrSystem::set_field_logical_timepoint(const std::string& provider_slot,
                                            const FieldLogicalTimePoint& point) {
  auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("AmrSystem::set_field_logical_timepoint unknown provider slot");
  found->second.boundary_context.point = point;
  if (!found->second.has_boundary_kernel || !found->second.boundary_kernel.observes_iteration)
    found->second.boundary_context.point.iteration = 0;
  if (p_->runtime)
    p_->runtime->set_field_logical_timepoint(provider_slot, found->second.boundary_context.point);
}

void AmrSystem::set_field_boundary_parameters(const std::string& provider_slot,
                                              const std::vector<double>& parameters) {
  auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("AmrSystem::set_field_boundary_parameters unknown provider slot");
  auto& plan = found->second;
  if (!plan.boundary_parameters)
    plan.boundary_parameters = std::make_shared<std::vector<Real>>();
  plan.boundary_parameters->assign(parameters.begin(), parameters.end());
  plan.boundary_context.parameters = plan.boundary_parameters.get();
  plan.boundary_context.parameter_count = static_cast<int>(parameters.size());
  if (p_->runtime)
    p_->runtime->set_field_boundary_parameters(
        provider_slot, std::vector<Real>(parameters.begin(), parameters.end()));
  p_->field_plan_consensus_verified_ = false;
}

void AmrSystem::set_field_newton_plan(const std::string& provider_slot, double tolerance,
                                      int max_iterations, double linear_tolerance,
                                      int linear_max_iterations, int restart, double armijo,
                                      double minimum_step) {
  require_assembling_amr(p_->bound_, "set_field_newton_plan");
  auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("AmrSystem::set_field_newton_plan unknown field provider slot");
  FieldNewtonOptions options{
      static_cast<Real>(tolerance),   max_iterations, static_cast<Real>(linear_tolerance),
      linear_max_iterations,          restart,        static_cast<Real>(armijo),
      static_cast<Real>(minimum_step)};
  validate_field_newton_options(options);
  found->second.newton = options;
  found->second.has_newton = true;
  if (p_->runtime)
    p_->runtime->set_field_newton_plan(provider_slot, options);
  p_->field_plan_consensus_verified_ = false;
}

void AmrSystem::set_field_nullspace(const std::string& provider_slot,
                                    const std::string& nullspace_provider_identity,
                                    const PreparedProviderOptions& options) {
  require_assembling_amr(p_->bound_, "set_field_nullspace");
  auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("AmrSystem::set_field_nullspace unknown provider slot");
  const auto provider =
      p_->field_nullspace_provider_registry_->resolve(nullspace_provider_identity);
  if (!provider->accepts_options(options))
    throw std::runtime_error("AmrSystem::set_field_nullspace provider rejected its options");
  FieldNullspaceProviderSelection selection{nullspace_provider_identity, options};
  (void)selection.exact_contract();
  found->second.nullspace = std::move(selection);
  p_->field_plan_consensus_verified_ = false;
}

void AmrSystem::set_density(const std::string& name, const std::vector<double>& rho) {
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::set_density : the system is already built (set the "
        "density before any step/mass/density)");
  if (p_->blocks.empty())
    throw std::runtime_error("AmrSystem::set_density : call add_block first");
  const int resolved = p_->block_index(name);
  if (resolved < 0)
    throw std::runtime_error("AmrSystem::set_density : no block named '" + name + "'");
  const std::size_t cells =
      static_cast<std::size_t>(p_->cfg.n) * static_cast<std::size_t>(p_->cfg.ny);
  if (rho.size() != cells)
    throw std::runtime_error("AmrSystem::set_density : density size (" +
                             std::to_string(rho.size()) + ") differs from ny*nx (" +
                             std::to_string(cells) + ")");
  const std::size_t idx = static_cast<std::size_t>(resolved);
  p_->blocks[idx].density = rho;
  p_->blocks[idx].has_density = true;
}

void AmrSystem::set_conservative_state(const std::string& name, const std::vector<double>& U) {
  require_assembling_amr(p_->bound_,
                         "set_conservative_state");  // frozen once pops.bind completes (ADC-592)
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::set_conservative_state : the system is already built "
        "(set the state before any step/mass/density)");
  if (p_->blocks.empty())
    throw std::runtime_error("AmrSystem::set_conservative_state : call add_block first");
  // UPSTREAM size guard: NON empty state and multiple of ny*nx. The exact size ncomp*ny*nx is checked
  // at build (coupler_write_coarse_state), the only place where ncomp == Model::n_vars is known -- same
  // deferral as the dense-cell guard of set_density. We explicitly reject an EMPTY state (0 % nn == 0 would
  // otherwise set has_state=true with an empty state, which would only throw deep in the 1st step).
  const std::size_t nn = static_cast<std::size_t>(p_->cfg.n) * static_cast<std::size_t>(p_->cfg.ny);
  if (U.empty())
    throw std::runtime_error(
        "AmrSystem::set_conservative_state : empty state (expected ncomp*ny*nx)");
  if (U.size() % nn != 0)
    throw std::runtime_error("AmrSystem::set_conservative_state : state size (" +
                             std::to_string(U.size()) + ") not a multiple of ny*nx (" +
                             std::to_string(nn) + ") ; expected ncomp*ny*nx component-major");
  const int resolved = p_->block_index(name);
  if (resolved < 0)
    throw std::runtime_error("AmrSystem::set_conservative_state : no block named '" + name + "'");
  const std::size_t idx = static_cast<std::size_t>(resolved);
  p_->blocks[idx].state = U;
  p_->blocks[idx].has_state = true;
}

bool AmrSystem::bootstrap_next_level(int refinement_ratio) {
  require_assembling_amr(p_->bound_, "bootstrap_next_level");
  if (!p_->cfg.explicit_bootstrap)
    throw std::runtime_error("AmrSystem::bootstrap_next_level requires explicit_bootstrap=true");
  if (p_->bootstrap_block_subjects.size() != p_->blocks.size())
    throw std::runtime_error(
        "AmrSystem::bootstrap_next_level requires one routed cell state per native block");
  for (const auto& [_, block] : p_->bootstrap_block_subjects)
    if (p_->block_index(block) < 0)
      throw std::runtime_error(
          "AmrSystem::bootstrap_next_level cell route names an unknown native block");
  p_->ensure_built();
  if (!p_->runtime)
    throw std::runtime_error(
        "AmrSystem::bootstrap_next_level requires the shared N-level runtime engine");
  if (p_->runtime->nlev() >= p_->cfg.level_count)
    throw std::runtime_error("AmrSystem::bootstrap_next_level would exceed resolved level_count");
  for (const auto& [_, route] : p_->bootstrap_subject_routes) {
    const auto& descriptor = p_->bootstrap_transfer_routes.at(route).descriptor;
    if ((descriptor.operation == "prolongation" || descriptor.operation == "restriction" ||
         descriptor.operation == "coarse_fine_fill" ||
         descriptor.operation == "temporal_interpolation") &&
        descriptor.refinement_ratio != refinement_ratio)
      throw std::runtime_error(
          "AmrSystem::bootstrap_next_level transition/transfer ratio mismatch");
  }
  const std::size_t next_transition = static_cast<std::size_t>(p_->runtime->nlev() - 1);
  if (next_transition >= p_->temporal_relations_.size())
    throw std::runtime_error(
        "AmrSystem::bootstrap_next_level lacks the explicit temporal relation for the next "
        "coarse/fine transition");
  const bool created = p_->runtime->bootstrap_next_level(refinement_ratio);
  if (created)
    p_->install_active_temporal_relations();
  return created;
}

void AmrSystem::begin_bootstrap_plan() {
  require_assembling_amr(p_->bound_, "begin_bootstrap_plan");
  if (!p_->cfg.explicit_bootstrap)
    throw std::runtime_error("AmrSystem::begin_bootstrap_plan requires explicit_bootstrap=true");
  if (p_->macro_step_ != 0 || p_->t != 0.0)
    throw std::runtime_error(
        "AmrSystem::begin_bootstrap_plan requires the authoritative clock to remain at "
        "t=0/step=0");
  p_->ensure_built();
  if (!p_->runtime)
    throw std::runtime_error(
        "AmrSystem::begin_bootstrap_plan requires the shared N-level runtime engine");
  p_->runtime->begin_bootstrap_plan();
}

void AmrSystem::register_bootstrap_transfer_route(
    const std::string& identity, const std::vector<std::string>& subjects,
    const std::string& provider_identity, const std::string& space, const std::string& centering,
    const std::string& representation, const std::string& storage, const std::string& operation,
    const std::string& kernel, int order, const std::vector<int>& ghost_depth, int dimension,
    int refinement_ratio) {
  require_assembling_amr(p_->bound_, "register_bootstrap_transfer_route");
  if (p_->built || subjects.empty())
    throw std::runtime_error(
        "AmrSystem::register_bootstrap_transfer_route requires pre-build subjects");
  const std::string kernel_namespace = (space == "field" || space == "cache")
                                           ? "pops.lib.amr.materializer::"
                                           : "pops.lib.amr.transfer::";
  runtime::amr::TransferRoute route{
      provider_identity, kernel_namespace + kernel,
      runtime::amr::TransferRouteDescriptor{space, centering, representation, storage, operation,
                                            order, ghost_depth, dimension, refinement_ratio}};
  auto staged_subject_routes = p_->bootstrap_subject_routes;
  for (const std::string& subject : subjects) {
    const auto key = std::make_pair(subject, operation);
    if (subject.empty() || !staged_subject_routes.emplace(key, identity).second)
      throw std::runtime_error(
          "AmrSystem::register_bootstrap_transfer_route requires unique subjects");
  }
  auto staged_transfer_routes = p_->bootstrap_transfer_routes;
  staged_transfer_routes.add(identity, std::move(route));
  // Commit both fully validated copies together.  A rejected provider, identity or subject leaves
  // both live registries byte-for-byte unchanged.
  p_->bootstrap_transfer_routes = std::move(staged_transfer_routes);
  p_->bootstrap_subject_routes = std::move(staged_subject_routes);
}

void AmrSystem::commit_bootstrap_level() {
  if (!p_->runtime)
    throw std::runtime_error("AmrSystem::commit_bootstrap_level has no runtime engine");
  if (p_->macro_step_ != 0 || p_->t != 0.0)
    throw std::runtime_error(
        "AmrSystem::commit_bootstrap_level requires the authoritative clock to remain at "
        "t=0/step=0");
  // The generated Program context was constructed against the coarse-only engine. Requalify its
  // accepted level clocks/history while the hierarchy transition is still rollback-capable, then
  // commit the spatial transaction. If either validation fails, preserve the previous accepted
  // bytes/revision; NativeAMRBootstrapConsumer will roll back the still-pending hierarchy.
  const std::vector<std::uint8_t> previous_program_state = p_->program_accepted_state_;
  const std::uint64_t previous_program_revision = p_->program_accepted_state_revision_;
  try {
    p_->program_.refresh_hierarchy_state("AmrSystem::commit_bootstrap_level");
    p_->runtime->commit_bootstrap_level();
  } catch (...) {
    p_->program_accepted_state_ = previous_program_state;
    p_->program_accepted_state_revision_ = previous_program_revision;
    throw;
  }
}

void AmrSystem::rollback_bootstrap_level() {
  if (!p_->runtime)
    throw std::runtime_error("AmrSystem::rollback_bootstrap_level has no runtime engine");
  p_->runtime->rollback_bootstrap_level();
  p_->install_active_temporal_relations();
}

void AmrSystem::register_bootstrap_array(const std::string& subject, const std::string& centering,
                                         int ncomp, int ny, int nx,
                                         const std::vector<double>& values) {
  require_assembling_amr(p_->bound_, "register_bootstrap_array");
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::register_bootstrap_array requires registration before hierarchy build");
  if (subject.empty() || p_->bootstrap_arrays.count(subject) != 0)
    throw std::runtime_error(
        "AmrSystem::register_bootstrap_array requires a unique non-empty subject");
  const auto route_key = std::make_pair(subject, std::string("prolongation"));
  const auto route_found = p_->bootstrap_subject_routes.find(route_key);
  if (route_found == p_->bootstrap_subject_routes.end())
    throw std::runtime_error(
        "AmrSystem::register_bootstrap_array requires an exact registered prolongation route");
  const auto& route = p_->bootstrap_transfer_routes.at(route_found->second);
  if (route.descriptor.centering != centering ||
      (route.descriptor.space != "face" && route.descriptor.space != "node"))
    throw std::runtime_error(
        "AmrSystem::register_bootstrap_array centering differs from its transfer route");
  const int base_nx = p_->cfg.n, base_ny = p_->cfg.ny;
  const bool shape_ok = (centering == "node" && nx == base_nx + 1 && ny == base_ny + 1) ||
                        (centering == "face_x" && nx == base_nx + 1 && ny == base_ny) ||
                        (centering == "face_y" && nx == base_nx && ny == base_ny + 1);
  if (!shape_ok || ncomp < 1 || values.size() != static_cast<std::size_t>(ncomp) * nx * ny)
    throw std::runtime_error(
        "AmrSystem::register_bootstrap_array received an incompatible centered array shape");
  p_->bootstrap_arrays.emplace(subject, Impl::BootstrapArray{centering, ncomp, values});
}

void AmrSystem::bind_bootstrap_block_subject(const std::string& subject, const std::string& block) {
  require_assembling_amr(p_->bound_, "bind_bootstrap_block_subject");
  const auto route_key = std::make_pair(subject, std::string("prolongation"));
  const auto route_found = p_->bootstrap_subject_routes.find(route_key);
  if (route_found == p_->bootstrap_subject_routes.end() || subject.empty() || block.empty() ||
      p_->bootstrap_block_subjects.count(subject) != 0)
    throw std::runtime_error(
        "AmrSystem::bind_bootstrap_block_subject requires an exact unique cell route");
  const auto& route = p_->bootstrap_transfer_routes.at(route_found->second);
  if (route.descriptor.space != "cell" || route.descriptor.centering != "cell")
    throw std::runtime_error("AmrSystem::bind_bootstrap_block_subject route is not cell-centered");
  p_->bootstrap_block_subjects.emplace(subject, block);
}

void AmrSystem::register_bootstrap_face_vector(const std::vector<std::string>& subjects) {
  require_assembling_amr(p_->bound_, "register_bootstrap_face_vector");
  if (p_->built || subjects.size() != 2 || subjects[0] == subjects[1])
    throw std::runtime_error(
        "AmrSystem::register_bootstrap_face_vector requires one exact x/y subject pair");
  std::array<std::string, 2> ordered{};
  for (const std::string& subject : subjects) {
    const auto route_id = p_->bootstrap_subject_routes.find({subject, "prolongation"});
    if (route_id == p_->bootstrap_subject_routes.end())
      throw std::runtime_error("bootstrap face-vector subject has no prolongation route");
    const auto& route = p_->bootstrap_transfer_routes.at(route_id->second);
    if (route.descriptor.space != "face" || !route.executable.face_vector)
      throw std::runtime_error("bootstrap face-vector route is not a prepared vector kernel");
    const int axis = route.descriptor.centering == "face_x"   ? 0
                     : route.descriptor.centering == "face_y" ? 1
                                                              : -1;
    if (axis < 0 || !ordered[static_cast<std::size_t>(axis)].empty())
      throw std::runtime_error("bootstrap face-vector centering is not an exact x/y pair");
    ordered[static_cast<std::size_t>(axis)] = subject;
  }
  for (const std::string& subject : ordered) {
    if (subject.empty() || p_->bootstrap_face_vectors.count(subject) != 0)
      throw std::runtime_error("bootstrap face subject belongs to multiple vector providers");
    p_->bootstrap_face_vectors.emplace(subject, ordered);
  }
}

void AmrSystem::register_analytic_constant(const std::string& subject, const std::string& block,
                                           const std::string& space, const std::string& centering,
                                           const std::vector<double>& components) {
  require_assembling_amr(p_->bound_, "register_analytic_constant");
  if (p_->built || subject.empty() || components.empty() ||
      p_->bootstrap_analytic_constants.count(subject) != 0 ||
      p_->bootstrap_analytic_gaussians.count(subject) != 0 ||
      p_->bootstrap_analytic_expressions.count(subject) != 0)
    throw std::runtime_error(
        "AmrSystem::register_analytic_constant requires non-empty unique components");
  const auto route_found = p_->bootstrap_subject_routes.find({subject, "prolongation"});
  if (route_found == p_->bootstrap_subject_routes.end())
    throw std::runtime_error(
        "AmrSystem::register_analytic_constant requires an exact registered route");
  const auto& route = p_->bootstrap_transfer_routes.at(route_found->second);
  if (route.descriptor.space != space || route.descriptor.centering != centering)
    throw std::runtime_error(
        "AmrSystem::register_analytic_constant descriptor differs from its transfer route");
  if (space == "cell") {
    bind_bootstrap_block_subject(subject, block);
  } else if (space == "face" || space == "node") {
    if (p_->bootstrap_arrays.count(subject) != 0)
      throw std::runtime_error(
          "AmrSystem::register_analytic_constant requires a unique centered subject");
    // Only the carrier descriptor crosses the authoring/runtime seam.  The native bootstrap
    // transaction allocates the per-patch MultiFab and materializes the constant on device.
    p_->bootstrap_arrays.emplace(
        subject, Impl::BootstrapArray{centering, static_cast<int>(components.size()), {}});
  } else {
    throw std::runtime_error(
        "AmrSystem::register_analytic_constant supports cell/face/node payloads");
  }
  p_->bootstrap_analytic_constants.emplace(subject, components);
}

void AmrSystem::register_analytic_gaussian(const std::string& subject, const std::string& block,
                                           double center_x, double center_y, double background,
                                           double amplitude, double inverse_width) {
  require_assembling_amr(p_->bound_, "register_analytic_gaussian");
  const bool finite = std::isfinite(center_x) && std::isfinite(center_y) &&
                      std::isfinite(background) && std::isfinite(amplitude) &&
                      std::isfinite(inverse_width);
  if (p_->built || subject.empty() || block.empty() || !finite || inverse_width <= 0.0 ||
      p_->bootstrap_analytic_gaussians.count(subject) != 0 ||
      p_->bootstrap_analytic_constants.count(subject) != 0 ||
      p_->bootstrap_analytic_expressions.count(subject) != 0)
    throw std::runtime_error(
        "AmrSystem::register_analytic_gaussian requires one finite unique scalar profile");
  const auto route_found = p_->bootstrap_subject_routes.find({subject, "prolongation"});
  if (route_found == p_->bootstrap_subject_routes.end())
    throw std::runtime_error(
        "AmrSystem::register_analytic_gaussian requires an exact registered route");
  const auto& route = p_->bootstrap_transfer_routes.at(route_found->second);
  if (route.descriptor.space != "cell" || route.descriptor.centering != "cell")
    throw std::runtime_error("gaussian bootstrap source requires a cell-centered route");
  bind_bootstrap_block_subject(subject, block);
  p_->bootstrap_analytic_gaussians.emplace(
      subject, Impl::BootstrapGaussian{center_x, center_y, background, amplitude, inverse_width});
}

void AmrSystem::register_analytic_expression(const std::string& subject, const std::string& block,
                                             const std::string& space, const std::string& centering,
                                             const std::vector<std::vector<std::string>>& opcodes,
                                             const std::vector<std::vector<double>>& literals) {
  using BlockSubjectMap = decltype(p_->bootstrap_block_subjects);
  using ExpressionMap = decltype(p_->bootstrap_analytic_expressions);
  struct PreparedRegistration {
    BlockSubjectMap block_subject;
    ExpressionMap expression;
  };

  PreparedRegistration prepared = analytic::collectively_prepare_analytic_request(
      "AmrSystem::register_analytic_expression",
      {{"block", block}, {"centering", centering}, {"space", space}, {"subject", subject}}, {},
      opcodes, literals, [&]() {
        require_assembling_amr(p_->bound_, "register_analytic_expression");
        if (p_->built || subject.empty() || block.empty() || space != "cell" ||
            centering != "cell" || p_->bootstrap_analytic_constants.count(subject) != 0 ||
            p_->bootstrap_analytic_gaussians.count(subject) != 0 ||
            p_->bootstrap_analytic_expressions.count(subject) != 0 ||
            p_->bootstrap_block_subjects.count(subject) != 0)
          throw std::runtime_error(
              "AmrSystem::register_analytic_expression requires one unique cell-centred state "
              "profile");
        const auto route_found = p_->bootstrap_subject_routes.find({subject, "prolongation"});
        if (route_found == p_->bootstrap_subject_routes.end())
          throw std::runtime_error(
              "AmrSystem::register_analytic_expression requires an exact registered route");
        const auto& route = p_->bootstrap_transfer_routes.at(route_found->second);
        if (route.descriptor.space != space || route.descriptor.centering != centering)
          throw std::runtime_error(
              "analytic expression bootstrap source differs from its transfer route");
        if (p_->block_index(block) < 0)
          throw std::runtime_error("analytic expression targets an unknown block");

        std::vector<analytic::AnalyticProgram> programs =
            analytic::compile_component_programs(opcodes, literals);
        // Allocate both destination bucket arrays and both nodes before consensus. Publication below
        // is then a pair of allocation-free node transfers after every rank accepted the request.
        p_->bootstrap_block_subjects.reserve(p_->bootstrap_block_subjects.size() + 1u);
        p_->bootstrap_analytic_expressions.reserve(p_->bootstrap_analytic_expressions.size() + 1u);
        PreparedRegistration staged;
        staged.block_subject.emplace(subject, block);
        staged.expression.emplace(subject, std::move(programs));
        return staged;
      });

  p_->bootstrap_block_subjects.merge(prepared.block_subject);
  p_->bootstrap_analytic_expressions.merge(prepared.expression);
  if (!prepared.block_subject.empty() || !prepared.expression.empty())
    throw std::logic_error("analytic expression collective publication lost registry uniqueness");
}

std::int64_t AmrSystem::bootstrap_analytic_reproject(const std::string& subject, int level) {
  if (!p_->runtime || level < 0 || level >= p_->runtime->nlev())
    throw std::runtime_error(
        "AmrSystem::bootstrap_analytic_reproject requires a pending hierarchy level");
  const auto constants = p_->bootstrap_analytic_constants.find(subject);
  const auto gaussian = p_->bootstrap_analytic_gaussians.find(subject);
  const auto expression = p_->bootstrap_analytic_expressions.find(subject);
  const auto route_found = p_->bootstrap_subject_routes.find({subject, "prolongation"});
  const int provider_count = (constants != p_->bootstrap_analytic_constants.end() ? 1 : 0) +
                             (gaussian != p_->bootstrap_analytic_gaussians.end() ? 1 : 0) +
                             (expression != p_->bootstrap_analytic_expressions.end() ? 1 : 0);
  if (provider_count != 1 || route_found == p_->bootstrap_subject_routes.end())
    throw std::runtime_error("analytic bootstrap source/route is not registered");
  const auto& route = p_->bootstrap_transfer_routes.at(route_found->second);
  if (route.descriptor.space == "cell") {
    const auto block = p_->bootstrap_block_subjects.find(subject);
    if (block == p_->bootstrap_block_subjects.end())
      throw std::runtime_error("analytic cell source has no bound block");
    const std::size_t index = static_cast<std::size_t>(p_->block_index(block->second));
    if (gaussian != p_->bootstrap_analytic_gaussians.end())
      return p_->runtime->fill_bootstrap_block_gaussian(
          index, level, static_cast<Real>(gaussian->second.center_x),
          static_cast<Real>(gaussian->second.center_y),
          static_cast<Real>(gaussian->second.background),
          static_cast<Real>(gaussian->second.amplitude),
          static_cast<Real>(gaussian->second.inverse_width));
    if (expression != p_->bootstrap_analytic_expressions.end())
      return p_->runtime->fill_bootstrap_block_analytic(index, level, expression->second);
    return p_->runtime->fill_bootstrap_block_constant(index, level, constants->second);
  }
  if (p_->bootstrap_arrays.count(subject) == 0)
    throw std::runtime_error("analytic centered source has no exact parent payload");
  return p_->runtime->fill_bootstrap_staggered_constant(subject, level, constants->second);
}

int AmrSystem::apply_bootstrap_component_floor(const std::string& subject, int level, int component,
                                               double floor) {
  if (!p_->runtime)
    throw std::runtime_error(
        "AmrSystem::apply_bootstrap_component_floor requires the native runtime");
  const auto block = p_->bootstrap_block_subjects.find(subject);
  if (block == p_->bootstrap_block_subjects.end())
    throw std::runtime_error("bootstrap component floor subject is not a cell state");
  const int index = p_->block_index(block->second);
  if (index < 0)
    throw std::runtime_error("bootstrap component floor block is unknown");
  return p_->runtime->apply_bootstrap_component_floor(static_cast<std::size_t>(index), level,
                                                      component, static_cast<Real>(floor));
}

std::int64_t AmrSystem::recompute_bootstrap_field(const std::string& subject,
                                                  const std::string& field_name) {
  if (!p_->runtime)
    throw std::runtime_error("bootstrap field recompute has no runtime engine");
  const auto route_id = p_->bootstrap_subject_routes.find({subject, "coarse_fine_fill"});
  if (route_id == p_->bootstrap_subject_routes.end())
    throw std::runtime_error("bootstrap field has no prepared elliptic materializer");
  const auto& route = p_->bootstrap_transfer_routes.at(route_id->second);
  if (route.descriptor.space != "field" || !route.executable.materialize)
    throw std::runtime_error("bootstrap field route is not an executable materializer");
  return route.executable.materialize(
      *p_->runtime, runtime::amr::MaterializationContext{subject, field_name, "recompute",
                                                         p_->runtime->nlev() - 1});
}

std::int64_t AmrSystem::bootstrap_prolong_array(const std::string& subject, int level) {
  require_assembling_amr(p_->bound_, "bootstrap_prolong_array");
  p_->ensure_built();
  if (!p_->runtime || level <= 0 || level >= p_->runtime->nlev())
    throw std::runtime_error(
        "AmrSystem::bootstrap_prolong_array requires an already-created native fine level");
  const auto route = p_->bootstrap_subject_routes.find({subject, "prolongation"});
  if (route == p_->bootstrap_subject_routes.end())
    throw std::runtime_error("bootstrap transfer subject has no prepared prolongation route");
  const auto& registered = p_->bootstrap_transfer_routes.at(route->second);
  const auto& prepared = registered.executable;
  const auto face = p_->bootstrap_face_vectors.find(subject);
  if (face != p_->bootstrap_face_vectors.end()) {
    const auto& pair = face->second;
    return p_->runtime->prolong_bootstrap_face_vector(pair[0], pair[1], level, prepared,
                                                      registered.descriptor.refinement_ratio);
  }
  const auto block = p_->bootstrap_block_subjects.find(subject);
  if (block != p_->bootstrap_block_subjects.end()) {
    const int index = p_->block_index(block->second);
    if (index < 0)
      throw std::runtime_error("bootstrap cell transfer names an unknown block");
    return p_->runtime->prolong_bootstrap_block(static_cast<std::size_t>(index), level, prepared,
                                                registered.descriptor.refinement_ratio);
  }
  if (p_->bootstrap_arrays.count(subject) == 0)
    throw std::runtime_error("bootstrap staggered transfer has no registered carrier");
  return p_->runtime->prolong_bootstrap_staggered_field(subject, level, prepared,
                                                        registered.descriptor.refinement_ratio);
}

void AmrSystem::synchronize_bootstrap_state(const std::string& subject, int fine_level) {
  if (!p_->runtime)
    throw std::runtime_error("bootstrap synchronization has no runtime engine");
  const auto block = p_->bootstrap_block_subjects.find(subject);
  const auto route = p_->bootstrap_subject_routes.find({subject, "restriction"});
  if (block == p_->bootstrap_block_subjects.end() || route == p_->bootstrap_subject_routes.end() ||
      p_->bootstrap_transfer_routes.at(route->second).descriptor.operation != "restriction")
    throw std::runtime_error(
        "bootstrap synchronization requires an exact volume-average state route");
  const int index = p_->block_index(block->second);
  if (index < 0)
    throw std::runtime_error("bootstrap synchronization names an unknown block");
  p_->runtime->synchronize_bootstrap_block(
      static_cast<std::size_t>(index), fine_level,
      p_->bootstrap_transfer_routes.at(route->second).executable,
      p_->bootstrap_transfer_routes.at(route->second).descriptor.refinement_ratio);
}

std::vector<double> AmrSystem::bootstrap_array_level(const std::string& subject, int level) const {
  if (!p_->runtime)
    throw std::runtime_error("AmrSystem::bootstrap_array_level has no runtime engine");
  return p_->runtime->bootstrap_staggered_level(subject, level);
}

void AmrSystem::invalidate_bootstrap_cache(const std::string& subject, int level) {
  if (!p_->runtime || subject.empty() || level < 0 || level >= p_->runtime->nlev())
    throw std::runtime_error("AmrSystem::invalidate_bootstrap_cache received an invalid key");
  const auto route_id = p_->bootstrap_subject_routes.find({subject, "coarse_fine_fill"});
  if (route_id == p_->bootstrap_subject_routes.end())
    throw std::runtime_error("bootstrap cache has no prepared materializer");
  const auto& route = p_->bootstrap_transfer_routes.at(route_id->second);
  if (route.descriptor.space != "cache" || !route.executable.materialize)
    throw std::runtime_error("bootstrap cache route is not an executable materializer");
  route.executable.materialize(*p_->runtime, runtime::amr::MaterializationContext{
                                                 subject, subject, "invalidate_cache", level});
}

std::vector<PatchBox> AmrSystem::rebuild_bootstrap_topology_cache(const std::string& subject,
                                                                  int level) {
  if (!p_->runtime || level < 0 || level >= p_->runtime->nlev())
    throw std::runtime_error(
        "AmrSystem::rebuild_bootstrap_topology_cache requires a materialized level");
  const auto route_id = p_->bootstrap_subject_routes.find({subject, "coarse_fine_fill"});
  if (route_id == p_->bootstrap_subject_routes.end())
    throw std::runtime_error("bootstrap cache has no prepared materializer");
  const auto& route = p_->bootstrap_transfer_routes.at(route_id->second);
  if (route.descriptor.space != "cache" || !route.executable.materialize)
    throw std::runtime_error("bootstrap cache route is not an executable materializer");
  route.executable.materialize(
      *p_->runtime, runtime::amr::MaterializationContext{subject, subject, "rebuild_cache", level});
  return p_->runtime->bootstrap_cache(subject).topology;
}

std::uint64_t AmrSystem::bootstrap_cache_epoch(const std::string& subject) const {
  if (!p_->runtime)
    throw std::runtime_error("AmrSystem::bootstrap_cache_epoch has no runtime engine");
  return p_->runtime->bootstrap_cache(subject).epoch;
}

void AmrSystem::set_magnetic_field(const std::vector<double>& bz) {
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::set_magnetic_field : the system is already built "
        "(set B_z before any step)");
  const std::size_t nn = static_cast<std::size_t>(p_->cfg.n) * static_cast<std::size_t>(p_->cfg.ny);
  if (bz.size() != nn)
    throw std::runtime_error("AmrSystem::set_magnetic_field : B_z of size " +
                             std::to_string(bz.size()) +
                             " (expected ny*nx = " + std::to_string(nn) + ", coarse row-major)");
  p_->bz_field = bz;
}

void AmrSystem::set_aux_field_component(int comp, const std::vector<double>& field) {
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::set_aux_field : the system is already built (set named aux "
        "fields before any step)");
  // RESERVED components (phi/grad/B_z/T_e): a model-named aux field starts at kAuxNamedBase. B_z keeps
  // its dedicated path (the Python facade intercepts canonical names; this guard covers a direct call).
  if (comp < kAuxNamedBase)
    throw std::runtime_error(
        "AmrSystem::set_aux_field : component " + std::to_string(comp) +
        " reserved (phi/grad_x/grad_y/B_z/T_e) ; a named aux field starts at index " +
        std::to_string(kAuxNamedBase) + " (B_z -> set_magnetic_field)");
  const std::size_t nn = static_cast<std::size_t>(p_->cfg.n) * static_cast<std::size_t>(p_->cfg.ny);
  if (field.size() != nn)
    throw std::runtime_error("AmrSystem::set_aux_field : field of size " +
                             std::to_string(field.size()) +
                             " (expected ny*nx = " + std::to_string(nn) + ", coarse row-major)");
  p_->named_aux_[comp].assign(
      field.begin(),
      field.end());  // pending: seeded into the engine at build (single + multi block)
}

void AmrSystem::set_aux_field_halo_component(int comp, int bc_type, double value) {
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::set_aux_field (halo) : the system is already built (set named "
        "aux halos before any step)");
  if (comp < kAuxNamedBase)
    throw std::runtime_error("AmrSystem::set_aux_field (halo) : component " + std::to_string(comp) +
                             " reserved ; a named aux field starts at index " +
                             std::to_string(kAuxNamedBase));
  if (bc_type != static_cast<int>(BCType::Foextrap) &&
      bc_type != static_cast<int>(BCType::Dirichlet))
    throw std::runtime_error("AmrSystem::set_aux_field (halo) : unsupported halo type " +
                             std::to_string(bc_type) + " ; use foextrap or dirichlet");
  p_->named_aux_bc_[comp] = AuxHaloPolicy{static_cast<BCType>(bc_type), static_cast<Real>(value)};
}

void AmrSystem::add_coupled_source(const CoupledSourceProgram& prog, double frequency,
                                   const std::string& label) {
  require_assembling_amr(p_->bound_,
                         "add_coupled_source");  // frozen once pops.bind completes (ADC-592)
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::add_coupled_source : the system is already built "
        "(register the source before any step/mass/density)");
  // A COUPLED source reads/writes SEVERAL named blocks. Refuse it when fewer than two blocks exist
  // (explicit contract error rather than a silent single-block no-op).
  if (p_->blocks.size() < 2)
    throw std::runtime_error(
        "AmrSystem::add_coupled_source : an inter-species source requires at least two blocks");
  // Bytecode description grouped into a POD (ADC-214). MINIMAL form validation here (list size);
  // the FINE validation (roles, blocks, opcodes, registers) is done by
  // AmrRuntime::add_coupled_source at injection (lazy build), exactly as System delegates to
  // CoupledSourceKernel. We store the flat spec as-is (POD fields copied one by one).
  if (prog.out_blocks.empty())
    throw std::runtime_error("AmrSystem::add_coupled_source : no source term (out_blocks empty)");
  p_->coupled_sources.push_back(Impl::CoupledSourceSpec{
      prog.in_blocks, prog.in_roles, prog.consts, prog.out_blocks, prog.out_roles, prog.prog_ops,
      prog.prog_args, prog.prog_lens, frequency, label, prog.freq_prog_ops, prog.freq_prog_args});
  // Inspect metadata (ADC-595): a raw add_coupled_source declares NO contract -> an "unchecked" view
  // (empty ConservationContract) carrying label + frequency bound. add_coupling_operator overwrites the
  // contract with the declared one.
  CouplingOperatorView view;
  view.label = label;
  view.frequency.constant_mu = frequency;
  view.frequency.per_cell = !prog.freq_prog_ops.empty() || !prog.freq_prog_args.empty();
  p_->coupled_operators.push_back(std::move(view));
}

void AmrSystem::add_coupling_operator(const CouplingOperator& op) {
  // Validate the DECLARED conservation contract against the actual output terms BEFORE anything is
  // stashed (host, fail-loud; anti-phantom-registration). Lower through the SAME add_coupled_source
  // path (bit-identical), then replace the unchecked view's contract with the declared one.
  validate_coupling_contract(op, "AmrSystem::add_coupling_operator");
  add_coupled_source(op.program, op.frequency.constant_mu, op.label);
  p_->coupled_operators.back().conservation = op.conservation;
}

const std::vector<CouplingOperatorView>& AmrSystem::coupled_operators() const {
  return p_->coupled_operators;
}

void AmrSystem::step(double dt) {
  p_->program_.require_step_installed("AmrSystem::step");
  p_->ensure_built();
  p_->program_.begin_step_projection_report();
  p_->execute_step_transaction([&] {
    // The installed Program is the sole temporal authority. It drives the per-level macro-step
    // through AmrProgramContext; AmrRuntime remains available only as the spatial hierarchy engine.
    p_->run_program_cadence_(dt);
    ++p_->macro_step_;  // authoritative counter (parity System: one macro-step = one increment)
  });
}
void AmrSystem::advance(double dt, int nsteps) {
  p_->program_.require_step_installed("AmrSystem::advance");
  for (int s = 0; s < nsteps; ++s)
    step(dt);
}
void AmrSystem::begin_step_transaction() {
  p_->ensure_built();
  if (p_->restart_transaction_active_)
    throw std::runtime_error(
        "AmrSystem::begin_step_transaction: restart transaction already active");
  if (p_->external_step_transaction_active_)
    throw std::runtime_error("AmrSystem::begin_step_transaction: transaction already active");
  if (p_->internal_step_transaction_active_)
    throw std::runtime_error(
        "AmrSystem::begin_step_transaction: cannot begin inside an executing step");
  p_->external_step_snapshot_.capture(*p_);
  p_->external_step_transaction_active_ = true;
  p_->external_step_transaction_restored_ = false;
  p_->external_step_transaction_committed_ = false;
}
void AmrSystem::commit_step_transaction() {
  if (!p_->external_step_transaction_active_)
    throw std::runtime_error("AmrSystem::commit_step_transaction: no active transaction");
  if (p_->external_step_transaction_committed_)
    throw std::runtime_error("AmrSystem::commit_step_transaction: transaction already committed");
  p_->external_step_transaction_committed_ = true;
}
std::map<std::string, double> AmrSystem::step_change_l2() const {
  if (!p_ || !p_->external_step_transaction_active_ || !p_->runtime)
    throw std::runtime_error(
        "AmrSystem::step_change_l2 requires an active external step transaction");
  return p_->runtime->step_change_l2(p_->external_step_snapshot_.runtime);
}
void AmrSystem::finalize_step_transaction() {
  if (!p_->external_step_transaction_active_ || !p_->external_step_transaction_committed_)
    throw std::runtime_error("AmrSystem::finalize_step_transaction: no committed transaction");
  p_->external_step_transaction_active_ = false;
  p_->external_step_transaction_restored_ = false;
  p_->external_step_transaction_committed_ = false;
}
void AmrSystem::rollback_step_transaction() {
  if (!p_->external_step_transaction_active_)
    throw std::runtime_error("AmrSystem::rollback_step_transaction: no active transaction");
  p_->restore_active_step_transaction_();
  p_->external_step_transaction_active_ = false;
  p_->external_step_transaction_restored_ = false;
  p_->external_step_transaction_committed_ = false;
}
bool AmrSystem::has_active_step_transaction() const noexcept {
  return p_ != nullptr && p_->has_active_step_transaction_();
}
void AmrSystem::restore_active_step_transaction_for_program() {
  if (!p_)
    throw std::logic_error("AmrSystem Program rollback requires a live facade");
  p_->restore_active_step_transaction_();
}
double AmrSystem::step_cfl(double cfl, double speed_floor, double max_dt, double min_dt) {
  p_->program_.require_step_installed("AmrSystem::step_cfl");
  p_->ensure_built();
  p_->program_.begin_step_projection_report();
  return p_->execute_step_transaction([&]() -> double {
    if (std::isnan(max_dt) || max_dt <= 0.0)
      throw std::invalid_argument("AmrSystem::step_cfl max_dt must be positive or +infinity");
    if (std::isnan(min_dt) || min_dt < 0.0)
      throw std::invalid_argument("AmrSystem::step_cfl min_dt must be finite and >= 0");
    const double hx = p_->cfg.L / p_->cfg.n;
    const double hy = p_->cfg.Ly / p_->cfg.ny;
    const double h = std::min(hx, hy);  // conservative Cartesian spacing for the scalar CFL API
    double dt = static_cast<double>(p_->runtime->cfl_dt(
        static_cast<Real>(cfl), static_cast<Real>(h), static_cast<Real>(speed_floor)));
    if (p_->program_.dt_bound_) {
      const double pb = static_cast<double>(p_->program_.dt_bound_(static_cast<Real>(cfl)));
      if (std::isfinite(pb) && pb > 0.0 && pb < dt) {
        dt = pb;
        p_->runtime->override_last_dt_bound("program:dt_bound");
      }
    }
    if (std::isfinite(max_dt) && max_dt < dt) {
      dt = std::min(dt, max_dt);
      p_->runtime->override_last_dt_bound("strategy:max_dt");
    }
    if (dt < min_dt)
      throw std::runtime_error("AmrSystem::step_cfl stability bound is below declared min_dt");
    p_->run_program_cadence_(dt);
    ++p_->macro_step_;
    return dt;
  });
}

// GLOBAL step bound (AMR counterpart of System::add_dt_bound): registered BEFORE or AFTER the build
// (passed to the engine at construction or added hot). fn() is evaluated PER RANK then reduced
// all_reduce_min on the consumer side.
void AmrSystem::add_dt_bound(const std::string& label, std::function<double()> fn) {
  require_assembling_amr(p_->bound_, "add_dt_bound");  // frozen once pops.bind completes (ADC-592)
  if (!fn)
    throw std::runtime_error("AmrSystem::add_dt_bound : empty bound function");
  p_->dt_bounds.push_back(Impl::GlobalDtBound{label, fn});
  if (p_->runtime)
    p_->runtime->add_dt_bound(label, std::move(fn));
}

// ACTIVE bound of the last step_cfl ("" before the first CFL step).
std::string AmrSystem::last_dt_bound() const {
  if (p_->runtime)
    return p_->runtime->last_dt_bound();
  return p_->last_dt_reason;
}

int AmrSystem::nx() const {
  return p_->cfg.n;
}
int AmrSystem::ny() const {
  return p_->cfg.ny;
}
double AmrSystem::time() const {
  return p_->t;
}
int AmrSystem::macro_step() const {
  return p_->macro_step_;
}
void AmrSystem::set_clock(double t, int macro_step) {
  try {
    if (macro_step < 0)
      throw std::runtime_error("AmrSystem::set_clock : macro_step >= 0 (restart)");
    if (!std::isfinite(t))
      throw std::runtime_error("AmrSystem::set_clock : time must be finite");
    p_->program_.consume_cadence_clock_restore(t, macro_step, "AmrSystem");
  } catch (...) {
    p_->program_.cancel_cadence_clock_restore();
    throw;
  }
  p_->t = t;
  p_->macro_step_ = macro_step;
  // Program cadence and regrid read this facade authority directly.  AmrRuntime deliberately owns
  // no second accepted clock that could drift after restart or rollback.
}

// --- compiled time-program install seam on the AMR hierarchy (epic ADC-511 / ADC-508, Spec 6) -------
// AMR counterpart of System::install_program_step / set_program_cadence / install_program / the
// per-block RuntimeParams store. The macro-step body the .so installs lives in the shared Program
// subsystem (program_.step_, ADC-594); AmrSystem::step routes through it (run_program_cadence_) when
// set. The cadence + RuntimeParams stores live in program_ (not the .so closure), mirroring System, so
// a value change reaches the captured context.
void AmrSystem::install_program_step(std::function<void(double)> step) {
  p_->program_.install_unverified_step(std::move(step));
}
void AmrSystem::install_program_hierarchy_refresh(std::function<void()> refresh) {
  p_->program_.install_hierarchy_refresh(std::move(refresh), "AmrSystem");
}
void AmrSystem::install_program_restart_hooks(std::function<void()> preflight,
                                              std::function<void()> regrid,
                                              std::function<void()> resync) {
  p_->program_.install_restart_hooks(std::move(preflight), std::move(regrid), std::move(resync),
                                     "AmrSystem");
}
// GLOBAL macro-step cadence around the installed program closure (parity System::set_program_cadence,
// ADC-411). Validates substeps >= 1 && stride >= 1 (fail-loud: a non-positive cadence is meaningless).
void AmrSystem::set_program_cadence(int substeps, int stride) {
  require_assembling_amr(p_->bound_,
                         "set_program_cadence");  // frozen once pops.bind completes (ADC-592)
  p_->program_.set_cadence(substeps, stride, "AmrSystem");
}
// Read the installed GLOBAL cadence (ADC-594, parity System): the tiny const getters the
// ProgramRuntimeReport reads through the bindings (no Python-visible getter existed before).
int AmrSystem::program_substeps() const {
  return p_->program_.substeps_;
}
int AmrSystem::program_stride() const {
  return p_->program_.stride_;
}
double AmrSystem::program_cadence_window_dt() const {
  return p_->program_.cadence_window_dt_;
}
int AmrSystem::program_cadence_window_steps() const {
  return p_->program_.cadence_window_steps_;
}
double AmrSystem::program_cadence_window_start_time() const {
  return p_->program_.cadence_window_start_time_;
}
void AmrSystem::restore_program_cadence_window(double accumulated_dt, int held_steps,
                                               double window_start_time, double accepted_last_dt,
                                               double accepted_time, int macro_step) {
  p_->program_.restore_cadence_window(accumulated_dt, held_steps, window_start_time,
                                      accepted_last_dt, accepted_time, macro_step, "AmrSystem");
}
// NAME-based block binding seam (Spec 3 criterion 23, ADC-457): install_program builds the map after
// matching the .so's block names; the AmrProgramContext reads it to translate a Program block index to
// the name-matched AMR block index.
void AmrSystem::set_program_block_map(const std::vector<int>& prog_to_sys) {
  p_->program_.block_map_ = prog_to_sys;
}
const std::vector<int>& AmrSystem::program_block_map() const {
  return p_->program_.block_map_;
}
bool AmrSystem::program_owns_operator_authority(
    const std::array<std::uint64_t, 4>& authority) const noexcept {
  return std::find(p_->program_.operator_authorities_.begin(),
                   p_->program_.operator_authorities_.end(),
                   authority) != p_->program_.operator_authorities_.end();
}
std::string AmrSystem::installed_program_hash() const {
  return p_->program_.installed_hash_;
}
// ADC-631: the last macro-step dt handed to the installed Program (set by run_program_cadence_
// before each step_). The AmrProgramContext reads it so a pre-commit history sample records its
// outgoing interval (variable-dt replay). Parity with ProgramRuntimeState::last_dt_ on Uniform.
double AmrSystem::program_last_dt() const {
  return static_cast<double>(p_->program_.last_dt_);
}
std::vector<std::uint8_t> AmrSystem::program_accepted_state() const {
  return p_->program_accepted_state_;
}
void AmrSystem::copy_program_accepted_state_into(std::vector<std::uint8_t>& state) const {
  state.resize(p_->program_accepted_state_.size());
  std::copy(p_->program_accepted_state_.begin(), p_->program_accepted_state_.end(), state.begin());
}
void AmrSystem::restore_program_accepted_state(const std::vector<std::uint8_t>& state) {
  p_->program_accepted_state_ = state;
  ++p_->program_accepted_state_revision_;
}
void AmrSystem::restore_checkpoint_accepted_state(const std::vector<std::uint8_t>& state) {
  const bool has_program =
      !p_->program_.installed_hash_.empty() || !p_->program_accepted_state_.empty();
  if (has_program != !state.empty())
    throw std::runtime_error("AMR checkpoint accepted state disagrees with the installed Program");
  if (!p_->runtime)
    throw std::runtime_error(
        "AMR checkpoint accepted state requires a materialized runtime hierarchy");
  runtime::amr::PersistentTaggingState tagging_candidate;
  if (has_program) {
    const auto accepted = runtime::program::deserialize_amr_program_accepted_state(state);
    tagging_candidate =
        p_->runtime->prepare_checkpoint_tagging_state(accepted.tagging_hysteresis_state);
  } else {
    tagging_candidate = p_->runtime->prepare_checkpoint_tagging_state({});
  }
  if (p_->program_accepted_state_revision_ == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("AMR checkpoint accepted-state revision overflow");
  std::vector<std::uint8_t> bytes_candidate = state;
  p_->program_accepted_state_.swap(bytes_candidate);
  p_->runtime->commit_checkpoint_tagging_state(std::move(tagging_candidate));
  ++p_->program_accepted_state_revision_;
}
void AmrSystem::materialize_program_restart_histories(const std::vector<std::uint8_t>& bytes,
                                                      const std::vector<std::string>& names,
                                                      const std::vector<int>& depths,
                                                      const std::vector<int>& ncomps) {
  p_->ensure_built();
  if (names.size() != depths.size() || names.size() != ncomps.size())
    throw std::invalid_argument(
        "AMR Program restart history names/depths/component counts must have equal length");
  const auto state = runtime::program::deserialize_amr_program_accepted_state(bytes);
  if (state.history_owners.size() != names.size())
    throw std::runtime_error(
        "AMR Program restart history registry differs from its accepted-state image");
  const auto exact_registry = [&names](const auto& values) {
    if (values.size() != names.size())
      return false;
    for (const std::string& name : names)
      if (values.find(name) == values.end())
        return false;
    return true;
  };
  if (!exact_registry(state.history_owners) || !exact_registry(state.history_states) ||
      !exact_registry(state.history_spaces) || !exact_registry(state.history_clocks) ||
      !exact_registry(state.history_interpolations) || !exact_registry(state.ring_clocks) ||
      !exact_registry(state.ring_identities) || !exact_registry(state.ring_flux) ||
      !exact_registry(state.ring_flux_contributions) ||
      !exact_registry(state.ring_flux_initialized))
    throw std::runtime_error(
        "AMR Program restart accepted state has an incomplete qualified history registry");
  const std::vector<int>& block_map = p_->program_.block_map_;
  if (block_map.empty())
    throw std::runtime_error(
        "AMR Program restart requires the explicit program-to-runtime block map");
  for (std::size_t index = 0; index < names.size(); ++index) {
    const std::string& name = names[index];
    if (name.empty() || std::find(names.begin(), names.begin() + static_cast<std::ptrdiff_t>(index),
                                  name) != names.begin() + static_cast<std::ptrdiff_t>(index))
      throw std::invalid_argument(
          "AMR Program restart history names must be unique non-empty text");
    const int depth = depths[index];
    const int ncomp = ncomps[index];
    if (depth < 2 || ncomp < 1)
      throw std::invalid_argument(
          "AMR Program restart history depth must be >= 2 and component count >= 1");
    const int program_owner = state.history_owners.at(name);
    if (program_owner < 0 || program_owner >= static_cast<int>(block_map.size()))
      throw std::runtime_error("AMR Program restart history has an invalid program block owner");
    const int runtime_owner = block_map[static_cast<std::size_t>(program_owner)];
    if (runtime_owner < 0 || runtime_owner >= static_cast<int>(p_->runtime->n_blocks()))
      throw std::runtime_error("AMR Program restart history maps to an invalid runtime block");
    if (state.ring_clocks.at(name).size() != static_cast<std::size_t>(depth) ||
        state.ring_identities.at(name).size() != static_cast<std::size_t>(depth) ||
        state.ring_flux.at(name).size() != static_cast<std::size_t>(depth) ||
        state.ring_flux_contributions.at(name).size() != static_cast<std::size_t>(depth))
      throw std::runtime_error(
          "AMR Program restart history depth differs from its accepted-state image");
    detail::AmrHistoryOps::register_history(*p_->runtime, static_cast<std::size_t>(runtime_owner),
                                            name, depth - 1, ncomp);
  }
}
std::uint64_t AmrSystem::program_accepted_state_revision() const {
  return p_->program_accepted_state_revision_;
}
std::vector<std::vector<std::string>> AmrSystem::program_accepted_state_manifest() const {
  std::vector<std::vector<std::string>> rows;
  if (p_->program_accepted_state_.empty())
    return rows;
  const auto state =
      runtime::program::deserialize_amr_program_accepted_state(p_->program_accepted_state_);
  rows.reserve(state.history_owners.size());
  for (const auto& [name, owner] : state.history_owners) {
    const auto ring = state.ring_clocks.find(name);
    const int depth = ring == state.ring_clocks.end()
                          ? (p_->runtime ? detail::AmrHistoryOps::depth(*p_->runtime, name) : 0)
                          : static_cast<int>(ring->second.size());
    rows.push_back({name, "program.block." + std::to_string(owner), state.history_states.at(name),
                    state.history_spaces.at(name), state.history_clocks.at(name),
                    state.history_interpolations.at(name), std::to_string(depth),
                    std::to_string(state.level_clocks.size())});
  }
  return rows;
}
std::vector<std::vector<std::string>> AmrSystem::program_clock_manifest() const {
  std::vector<std::vector<std::string>> rows;
  if (p_->program_accepted_state_.empty())
    return rows;
  const auto state =
      runtime::program::deserialize_amr_program_accepted_state(p_->program_accepted_state_);
  for (const auto& clock : state.level_clocks)
    rows.push_back({"level", std::to_string(clock.level), std::to_string(clock.macro_step),
                    std::to_string(clock.phase.numerator), std::to_string(clock.phase.denominator),
                    std::to_string(clock.physical_time)});
  for (const auto& [identity, tick] : state.logical_clock_ticks)
    rows.push_back({"logical", identity, std::to_string(tick)});
  return rows;
}
std::vector<std::vector<std::string>> AmrSystem::program_flux_ledger_manifest() const {
  std::vector<std::vector<std::string>> rows;
  if (p_->program_accepted_state_.empty())
    return rows;
  const auto state =
      runtime::program::deserialize_amr_program_accepted_state(p_->program_accepted_state_);
  const auto orientation = [](amr::FluxOrientation value) {
    switch (value) {
      case amr::FluxOrientation::XMinus:
        return "x_minus";
      case amr::FluxOrientation::XPlus:
        return "x_plus";
      case amr::FluxOrientation::YMinus:
        return "y_minus";
      case amr::FluxOrientation::YPlus:
        return "y_plus";
    }
    return "invalid";
  };
  for (const auto& entry : state.accepted_flux_ledger)
    rows.push_back({entry.key.owner, entry.key.state, entry.key.rate, entry.key.flux,
                    std::to_string(entry.key.level), std::to_string(entry.key.clock.macro_step),
                    std::to_string(entry.key.clock.phase.numerator),
                    std::to_string(entry.key.clock.phase.denominator),
                    std::to_string(entry.measure.stage_weight.numerator),
                    std::to_string(entry.measure.stage_weight.denominator),
                    orientation(entry.measure.orientation),
                    std::to_string(entry.measure.face_measure),
                    std::to_string(entry.measure.substep_duration)});
  return rows;
}
std::vector<std::vector<std::string>> AmrSystem::program_interface_flux_ledger_manifest() const {
  std::vector<std::vector<std::string>> rows;
  if (p_->program_accepted_state_.empty())
    return rows;
  const auto state =
      runtime::program::deserialize_amr_program_accepted_state(p_->program_accepted_state_);
  const auto orientation = [](amr::InterfaceFluxOrientation value) {
    switch (value) {
      case amr::InterfaceFluxOrientation::CoarseOutward:
        return "coarse_outward";
      case amr::InterfaceFluxOrientation::FineOutward:
        return "fine_outward";
    }
    return "invalid";
  };
  rows.reserve(state.accepted_interface_flux_ledger.size());
  for (const auto& entry : state.accepted_interface_flux_ledger)
    rows.push_back({entry.key.interface_identity,
                    std::to_string(entry.key.topology_epoch),
                    std::to_string(entry.key.coarse_level),
                    std::to_string(entry.key.fine_level),
                    std::to_string(entry.key.clock.level),
                    std::to_string(entry.key.clock.macro_step),
                    std::to_string(entry.key.clock.phase.numerator),
                    std::to_string(entry.key.clock.phase.denominator),
                    std::to_string(entry.key.clock.physical_time),
                    entry.key.stage_identity,
                    std::to_string(entry.key.interval.begin.level),
                    std::to_string(entry.key.interval.begin.macro_step),
                    std::to_string(entry.key.interval.begin.phase.numerator),
                    std::to_string(entry.key.interval.begin.phase.denominator),
                    std::to_string(entry.key.interval.begin.physical_time),
                    std::to_string(entry.key.interval.end.level),
                    std::to_string(entry.key.interval.end.macro_step),
                    std::to_string(entry.key.interval.end.phase.numerator),
                    std::to_string(entry.key.interval.end.phase.denominator),
                    std::to_string(entry.key.interval.end.physical_time),
                    orientation(entry.key.orientation),
                    std::to_string(entry.key.left_block),
                    std::to_string(entry.key.right_block),
                    std::to_string(entry.measure.stage_weight.numerator),
                    std::to_string(entry.measure.stage_weight.denominator),
                    std::to_string(entry.measure.face_measure),
                    std::to_string(entry.measure.substep_duration),
                    entry.measure.stage_weight_resolved ? "resolved" : "unresolved"});
  return rows;
}
std::vector<std::vector<std::string>> AmrSystem::program_sync_manifest() const {
  std::vector<std::vector<std::string>> rows;
  if (p_->program_accepted_state_.empty())
    return rows;
  const auto state =
      runtime::program::deserialize_amr_program_accepted_state(p_->program_accepted_state_);
  for (const auto& event : state.accepted_sync)
    rows.push_back({std::to_string(event.parent_level), std::to_string(event.child_level),
                    std::to_string(event.block), event.phase == 0 ? "reflux" : "average_down",
                    std::to_string(event.clock.macro_step),
                    std::to_string(event.clock.phase.numerator),
                    std::to_string(event.clock.phase.denominator)});
  return rows;
}
// RUNTIME FREEZE LIFECYCLE (ADC-592, parity with System). mark_bound() is the ONE transition into the
// frozen state; the Python bind flow calls it LAST (after every install call), so the install sequence
// itself never trips require_assembling_amr. A second call throws. lifecycle_state() reports
// "assembling" / "bound" / "running" (running derived from the authoritative macro_step_ counter, so it
// needs no extra state).
void AmrSystem::mark_bound() {
  if (p_->bound_)
    throw std::runtime_error(
        "AmrSystem::mark_bound: the composition is already bound (pops.bind binds a compiled Case "
        "exactly once; a fresh run needs a fresh pops.bind)");
  // Run the only field-plan collective before every rank-local validation of the first bind. The
  // ordered registry witness is cached and reused by build_multi; a direct unbound build performs
  // the same check there. The already-bound guard must stay first so a stray second call on one rank
  // cannot enter a collective alone.
  p_->require_field_plan_consensus();
  if (!p_->block_state_identities_.empty() &&
      p_->block_state_identities_.size() != p_->blocks.size())
    throw std::runtime_error(
        "AmrSystem::mark_bound: block state routes do not exactly cover materialized blocks");
  for (const auto& block : p_->blocks)
    if (!p_->block_state_identities_.empty() &&
        p_->block_state_identities_.find(block.name) == p_->block_state_identities_.end())
      throw std::runtime_error(
          "AmrSystem::mark_bound: materialized block lacks its exact state route");
  for (const auto& [name, plan] : p_->boundary_plans_) {
    if (std::none_of(p_->blocks.begin(), p_->blocks.end(),
                     [&name](const Impl::BlockSpec& block) { return block.name == name; }))
      throw std::runtime_error(
          "AmrSystem::mark_bound: prepared boundary plan references unknown block '" + name + "'");
    const Periodicity expected = p_->cfg.periodicity;
    if (!same_periodicity(plan->periodicity(), expected))
      throw std::runtime_error(
          "AmrSystem::mark_bound: prepared boundary plan periodicity disagrees with the domain "
          "topology for block '" +
          name + "'");
    (void)plan->has_boundary_linearization();
  }
  p_->bound_ = true;
}
std::string AmrSystem::lifecycle_state() const {
  if (!p_->bound_)
    return "assembling";
  return p_->macro_step_ > 0 ? "running" : "bound";
}
// COMPILED-PROGRAM RUNTIME PARAMETERS on AMR (ADC-508, parity ADC-510). Seed/overwrite/read the
// per-PROGRAM-block RuntimeParams the installed step closure reads through the AmrProgramContext. The
// store lives on the Impl so a value change reaches the captured context -- the no-recompile contract
// mirrored from System. install_program seeds the defaults; Python's _install_program_params overwrites
// the supplied values (validated against the .so param-name metadata). VERBATIM mirror of System.
void AmrSystem::seed_program_params(int prog_block, const std::vector<double>& defaults) {
  p_->program_.seed_params(prog_block, defaults);  // shared subsystem (ADC-594); resets to baseline
}
void AmrSystem::set_program_params(int prog_block, const std::vector<double>& values) {
  auto it = p_->program_.block_params_.find(prog_block);
  if (it == p_->program_.block_params_.end())
    throw std::out_of_range(
        "AmrSystem::set_program_params : program block " + std::to_string(prog_block) +
        " has no runtime parameter (the installed compiled Program declares none for it; declare "
        "dsl.Param(..., kind='runtime') in the model the Program lowers, or omit params=)");
  RuntimeParams& rp = it->second;
  if (static_cast<int>(values.size()) != rp.count)
    throw std::runtime_error("AmrSystem::set_program_params : program block " +
                             std::to_string(prog_block) + " expects " + std::to_string(rp.count) +
                             " runtime parameters, received " + std::to_string(values.size()));
  for (int k = 0; k < rp.count; ++k)
    rp.values[k] =
        static_cast<Real>(values[static_cast<std::size_t>(k)]);  // current value, next step
}
RuntimeParams AmrSystem::program_params(int prog_block) const {
  // Unseeded block (no runtime param) -> default RuntimeParams (count 0). Shared subsystem (ADC-594).
  return p_->program_.params(prog_block);
}
// The built AMR spatial engine the AmrProgramContext driver wraps (nullptr before the lazy build).
// install_program forces the runtime build so this is live there.
AmrRuntime* AmrSystem::engine() const {
  return p_->runtime.get();
}
// Compatibility predicate retained across the loader ABI. Every built AMR system now owns AmrRuntime;
// n_vars / level_state remain one-block convenience accessors.
bool AmrSystem::uses_runtime_engine() const {
  p_->ensure_built();
  return p_->runtime != nullptr;
}
// The facade-owned Profiler (parity System), forwarded to by the AmrProgramContext's profiling seam.
pops::runtime::program::Profiler& AmrSystem::profiler_handle() {
  return p_->program_.profiler_;
}
runtime::program::ProgramRuntimeState& AmrSystem::program_runtime_state_() {
  return p_->program_;
}
// Record / read a Program runtime diagnostic (parity System::record_program_diagnostic). Pure side
// effect; lives on the Impl (not the .so) so a later checkpoint can reach it.
void AmrSystem::record_program_diagnostic(const std::string& name, double value) {
  p_->program_.record_diagnostic(name, value);  // shared subsystem (ADC-594)
}
double AmrSystem::program_diagnostic(const std::string& name) const {
  // AMR keeps its historical LENIENT read (missing name -> 0.0), distinct from System's fail-loud
  // program_diagnostic; not routed through the struct's throwing diagnostic() helper.
  auto it = p_->program_.diagnostics_.find(name);
  return it == p_->program_.diagnostics_.end() ? 0.0 : it->second;
}
std::map<std::string, double> AmrSystem::program_diagnostics() const {
  return p_->program_.diagnostics_;
}
void AmrSystem::begin_step_projection_report() {
  p_->program_.begin_step_projection_report();
}
void AmrSystem::note_step_projection(const std::string& name) {
  p_->program_.note_step_projection(name);
}
std::vector<std::string> AmrSystem::consume_step_projections() {
  return p_->program_.consume_step_projections();
}

// Exact selected-level collective reduction over a named block. Live MultiFabs are folded through
// Kokkos; checkpoint/global-array gathers are never a diagnostics compute path.
double AmrSystem::composite_reduce(const std::string& block, const std::string& kind, int comp,
                                   const std::vector<int>& levels) const {
  p_->ensure_built();
  return p_->runtime->composite_reduce(block, kind, comp, levels);
}

double AmrSystem::composite_reduce_field(const std::string& provider_slot, const std::string& kind,
                                         int comp, const std::vector<int>& levels) {
  p_->ensure_built();
  return p_->runtime->composite_reduce_field(provider_slot, kind, comp, levels);
}

// Load a generated problem.so and install its compiled time Program on the AMR hierarchy. Mirrors
// System::install_program (the loader logic is VERBATIM, only the AMR ABI conventions differ: the
// global-scope promotion is anchored on amr_native_anchor like add_native_block, not pops::abi_key, and
// the install entry is pops_install_program_amr). The blocks must be ALREADY added (the AMR registry is
// frozen at the first lazy build); install_program runs BEFORE the first step so the .so's shared
// facade factory selects the hierarchy provider over THIS AmrSystem. The .so stays loaded.
POPS_EXPORT void AmrSystem::install_program(const std::string& so_path) {
  require_assembling_amr(p_->bound_,
                         "install_program");  // frozen once pops.bind completes (ADC-592)
  if (p_->built || p_->runtime)
    throw std::logic_error(
        "AmrSystem::install_program must run before the AMR runtime is materialized");
#if defined(_WIN32)
  // Windows: the generated .dll links against _pops.lib at compile time; no global promotion needed.
  pops::dynlib::handle h = pops::dynlib::open(so_path);
  if (!h)
    throw std::runtime_error("AmrSystem::install_program : LoadLibrary('" + so_path +
                             "') : " + pops::dynlib::last_error());
#else
  {
    // Promote the already-loaded module to the global scope so the .so's undefined AmrSystem seam
    // symbols (POPS_EXPORT) resolve against it, anchored on amr_native_anchor (TU-local, like
    // add_native_block) to avoid link-coupling pops::abi_key. macOS: harmless (dynamic_lookup).
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&amr_native_anchor), &info) && info.dli_fname)
      dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
  }
  // Resolve host seams from the promoted module without exporting this generated Program into the
  // process-wide scope, where another semantic variant may reuse the same generated symbol names.
  pops::dynlib::handle h = pops::dynlib::open(so_path);
  if (!h) {
    throw std::runtime_error(
        "AmrSystem::install_program : dlopen('" + so_path + "') : " + pops::dynlib::last_error() +
        " (the pops::AmrSystem seam accessors must be exported and the host module promoted "
        "globally ; cf. POPS_EXPORT)");
  }
#endif
  auto key_fn = reinterpret_cast<const char* (*)()>(pops::dynlib::sym(h, "pops_program_abi_key"));
  if (!key_fn) {
    pops::dynlib::close(h);
    throw std::runtime_error("AmrSystem::install_program : pops_program_abi_key missing from '" +
                             so_path +
                             "' (regenerate the problem module with the current pops headers)");
  }
  const std::string loader_key = key_fn();
  const std::string module_key = detail::abi_key_string();
  if (loader_key != module_key) {
    pops::dynlib::close(h);
    throw std::runtime_error(
        "AmrSystem::install_program : compiled program ABI mismatch : expected '" + module_key +
        "', got '" + loader_key +
        "'. Recompile the problem module with the SAME compiler, C++ standard and "
        "pops headers as the _pops module.");
  }
  // Route registry guard: the manifest is mandatory and must match before any installer is called.
  {
    auto manifest_fn =
        reinterpret_cast<const char* (*)()>(pops::dynlib::sym(h, "pops_program_route_manifest"));
    if (!manifest_fn) {
      pops::dynlib::close(h);
      throw std::runtime_error(
          "AmrSystem::install_program: pops_program_route_manifest missing; regenerate artifact");
    }
    try {
      const char* raw = manifest_fn();
      if (!raw || raw[0] == '\0')
        throw std::runtime_error(
            "AmrSystem::install_program: pops_program_route_manifest returned empty data");
      pops::verify_route_manifest(std::string(raw), "install_program");
    } catch (...) {
      pops::dynlib::close(h);
      throw;
    }
  }
  std::vector<pops::runtime::program::ProgramOperatorAuthority> operator_authorities;
  std::vector<pops::runtime::program::ProgramHistoryReplayAuthority> history_replay_authorities;
  try {
    operator_authorities = pops::runtime::program::read_program_operator_authorities(h);
    history_replay_authorities = pops::runtime::program::read_program_history_replay_authorities(h);
  } catch (...) {
    pops::dynlib::close(h);
    throw;
  }
  auto install =
      reinterpret_cast<void (*)(AmrSystem*)>(pops::dynlib::sym(h, "pops_install_program_amr"));
  if (!install) {
    pops::dynlib::close(h);
    throw std::runtime_error(
        "AmrSystem::install_program : pops_install_program_amr missing from '" + so_path +
        "' (regenerate the time Program with target='amr_system' : a target='system' .so exports "
        "pops_install_program, installable only on System -- use System for that, or AmrSystem "
        "with "
        "an AMR-target Program)");
  }
  // Mandatory install-time requirement validation, including AUX parity with System. B_z is present
  // only after set_magnetic_field; T_e has no AMR provider yet and is therefore rejected explicitly.
  // Derived phi/grad and component-keyed named aux fields remain available by construction.
  try {
    const auto meta = pops::runtime::program::read_module_metadata(h);
    const std::vector<std::string> amr_block_names = block_names();
    auto has_block = [&amr_block_names](const std::string& want) {
      for (const auto& nm : amr_block_names)
        if (nm == want)
          return true;
      return false;
    };
    auto provides_aux = [this](const std::string& name) {
      if (name == "B_z")
        return !p_->bz_field.empty();
      if (name == "T_e")
        return false;
      return true;
    };
    for (const auto& op : meta.operators) {
      for (const auto& aux : pops::runtime::program::required_aux(op.requirements)) {
        if (!provides_aux(aux)) {
          const std::string remedy = aux == "T_e"
                                         ? "the AMR runtime does not implement a typed T_e provider"
                                         : "call set_magnetic_field before installing the Program";
          throw std::runtime_error("AmrSystem::install_program: operator '" + op.name +
                                   "' requires aux field '" + aux + "', but " + remedy);
        }
      }
      // (b) BLOCK-INSTANCE requirements: an operator that reads another species names the block it needs;
      // reject if it was not added. Verbatim spec message (parity System::install_program).
      for (const auto& blk : pops::runtime::program::required_blocks(op.requirements)) {
        if (!has_block(blk)) {
          throw std::runtime_error("operator '" + op.name + "' requires block instance '" + blk +
                                   "'");
        }
      }
      // (c) SOLVER requirement: compare against the resolved default-field provider identity. The
      // requirement is data, not a dispatch branch; adding a provider does not change this core.
      const std::string need_solver = pops::runtime::program::required_solver(op.requirements);
      if (!need_solver.empty() && need_solver != p_->p_solver) {
        throw std::runtime_error("field operator '" + op.name + "' requires solver '" +
                                 need_solver + "'");
      }
    }
  } catch (...) {
    pops::dynlib::close(h);
    throw;
  }
  // Resolve every target-specific scalar entry before the first facade mutation. In particular,
  // never install DSO-backed boundary pointers and then discover that the module must be unloaded.
  using has_dt_t = bool (*)();
  using dt_bound_t = pops::Real (*)(AmrSystem*, pops::Real);
  auto has_dt = reinterpret_cast<has_dt_t>(pops::dynlib::sym(h, "pops_program_has_dt_bound"));
  auto dt_bound = reinterpret_cast<dt_bound_t>(pops::dynlib::sym(h, "pops_program_dt_bound_amr"));
  const bool program_has_dt_bound = has_dt && has_dt();
  if (program_has_dt_bound && !dt_bound) {
    pops::dynlib::close(h);
    throw std::runtime_error(
        "AmrSystem::install_program: Program declares a dt bound but "
        "pops_program_dt_bound_amr is missing; regenerate the AMR artifact");
  }
  auto hash_fn = reinterpret_cast<const char* (*)()>(pops::dynlib::sym(h, "pops_program_hash"));
  const std::string installed_hash = hash_fn ? std::string(hash_fn()) : std::string();
  auto install_boundaries = reinterpret_cast<void (*)(AmrSystem*)>(
      pops::dynlib::sym(h, "pops_install_field_boundaries_amr"));

  // NAME-based block binding (Spec 3 criterion 23, ADC-457). The Program numbers its blocks in P.state
  // declaration order (the .so's pops_program_block_name table); the AMR facade numbers its blocks in
  // add order (block_names). Bind by NAME, store the program-index -> AMR-block-index map (read by the
  // AmrProgramContext). A Program block whose name has no AMR block fails loud. The explicit block
  // identity table is REQUIRED; positional binding is unsupported.
  std::vector<int> program_block_map;
  {
    using count_t = int (*)();
    using name_t = const char* (*)(int);
    auto block_count = reinterpret_cast<count_t>(pops::dynlib::sym(h, "pops_program_block_count"));
    auto block_name = reinterpret_cast<name_t>(pops::dynlib::sym(h, "pops_program_block_name"));
    if (!block_count || !block_name) {
      pops::dynlib::close(h);
      throw std::runtime_error(
          "AmrSystem::install_program: compiled Program '" + so_path +
          "' does not export the required block identity table "
          "(pops_program_block_count + pops_program_block_name). Positional Program-to-AmrSystem "
          "binding has been removed; regenerate the Program library with the current PoPS "
          "codegen and headers.");
    }
    const std::vector<std::string> amr_names = block_names();
    const int n = block_count();
    // The generated Program now emits one point-qualified rhs_group for every simultaneous set of
    // block rates. AmrProgramContext maps every Program block by name and forwards that group to the
    // common AmrRuntime, which evaluates a shared interface exactly once before either residual is
    // consumed. Consequently the former single-block guard is obsolete; retaining it would reject a
    // route whose coupled scheduler semantics are already proved at resolve time.
    program_block_map.assign(static_cast<std::size_t>(n), -1);
    for (int p = 0; p < n; ++p) {
      const std::string want = block_name(p);
      int found = -1;
      for (std::size_t s = 0; s < amr_names.size(); ++s)
        if (amr_names[s] == want) {
          found = static_cast<int>(s);
          break;
        }
      if (found < 0) {
        pops::dynlib::close(h);
        throw std::runtime_error("Program requires block instance '" + want +
                                 "', but simulation did not instantiate it");
      }
      program_block_map[static_cast<std::size_t>(p)] = found;
    }
  }
  // RUNTIME PARAMETERS (ADC-508, parity ADC-510). Seed each PROGRAM block's RuntimeParams to the .so
  // pops_program_param_* declaration defaults so an install WITHOUT a runtime set behaves as with a const
  // param; a later Python params= route overwrites the supplied values via set_program_params. A Program
  // with no runtime param (the count symbol absent or 0) seeds nothing. VERBATIM mirror of System.
  std::map<int, std::vector<double>> program_param_defaults;
  {
    using count_t = int (*)();
    using ival_t = int (*)(int);
    using dval_t = double (*)(int);
    auto pcount = reinterpret_cast<count_t>(pops::dynlib::sym(h, "pops_program_param_count"));
    auto pblock = reinterpret_cast<ival_t>(pops::dynlib::sym(h, "pops_program_param_block"));
    auto pindex = reinterpret_cast<ival_t>(pops::dynlib::sym(h, "pops_program_param_index"));
    auto pdef = reinterpret_cast<dval_t>(pops::dynlib::sym(h, "pops_program_param_default"));
    if (pcount && pblock && pindex && pdef) {
      const int np = pcount();
      for (int i = 0; i < np; ++i) {
        const int blk = pblock(i);
        const int idx = pindex(i);
        std::vector<double>& d = program_param_defaults[blk];
        if (static_cast<int>(d.size()) <= idx)
          d.resize(static_cast<std::size_t>(idx) + 1, 0.0);
        d[static_cast<std::size_t>(idx)] = pdef(i);
      }
    }
  }

  // Install the macro-step body after the unique AmrRuntime materializes its per-level state. Every
  // facade value mutated below has an exact rollback image, and all DSO-backed closures are destroyed
  // before the handle is closed on failure.
  auto previous_install = p_->program_.capture_artifact_step_install();
  auto previous_field_plans = p_->field_plans_;
  const bool previous_field_consensus = p_->field_plan_consensus_verified_;
  auto previous_program_accepted_state = p_->program_accepted_state_;
  const std::uint64_t previous_program_accepted_state_revision =
      p_->program_accepted_state_revision_;
  try {
    p_->program_.reset_artifact_candidate_state();
    p_->program_.block_map_ = program_block_map;
    p_->program_.block_params_.clear();
    for (const auto& [block, defaults] : program_param_defaults)
      seed_program_params(block, defaults);
    p_->program_.operator_authorities_ = operator_authorities;
    p_->ensure_built();
    install(this);
    p_->program_.require_exact_artifact_step_install(previous_install,
                                                     "AmrSystem::install_program:");
    if (!p_->program_.hierarchy_refresh_)
      throw std::runtime_error(
          "AmrSystem::install_program: artifact installer did not install its accepted-state "
          "hierarchy refresh hook");
    if (!p_->program_.restart_regrid_preflight_ || !p_->program_.restart_regrid_ ||
        !p_->program_.restart_resync_)
      throw std::runtime_error(
          "AmrSystem::install_program: artifact installer did not install its restart "
          "preflight/regrid/resync hooks");

    p_->program_.block_map_ = std::move(program_block_map);
    for (const auto& [block, defaults] : program_param_defaults)
      seed_program_params(block, defaults);
    p_->program_.operator_authorities_ = std::move(operator_authorities);
    p_->program_.history_replay_authorities_ = std::move(history_replay_authorities);
    p_->program_.installed_hash_ = installed_hash;
    if (program_has_dt_bound) {
      AmrSystem* self = this;
      p_->program_.dt_bound_ = [self, dt_bound](Real cfl) -> Real { return dt_bound(self, cfl); };
    }
    p_->program_.artifact_backed_ = true;

    // The generated boundary context requires the engine, so install it only after ensure_built().
    // Each setter also updates the live runtime; both the facade plans and runtime are discarded if
    // the generated entry fails part-way through.
    if (install_boundaries)
      install_boundaries(this);
  } catch (...) {
    const std::exception_ptr failure = std::current_exception();
    p_->program_.rollback_artifact_step_install(std::move(previous_install));
    p_->program_accepted_state_ = std::move(previous_program_accepted_state);
    p_->program_accepted_state_revision_ = previous_program_accepted_state_revision;
    p_->field_plans_ = std::move(previous_field_plans);
    p_->field_plan_consensus_verified_ = previous_field_consensus;
    p_->runtime.reset();
    p_->built = false;
    pops::dynlib::close(h);
    std::rethrow_exception(failure);
  }
  // .so left loaded for the duration of the process (the installed closure points to code in it).
}
// --- AMR / MPI profiling (Spec 5 sec.12.5, ADC-479 criterion 43) ---------------------------------
// enable_profiling / profile_report drive the facade-owned Profiler (parity with System). The
// multi-block AmrRuntime engine (wired at build via set_profiler) times its non-numeric AMR phases
// -- regrid / fill_boundary / average_down -- and bumps the per-run + MPI counters into it. The
// Profiler lives on the Impl (NOT on SystemProgramDriver), so the C++ MockImpl never reads it. enable
// BEFORE the run; the engine is enabled()-guarded so toggling between runs is safe.
void AmrSystem::enable_profiling() {
  p_->program_.profiler_.enable();
}
void AmrSystem::disable_profiling() {
  p_->program_.profiler_.disable();
}
bool AmrSystem::is_profiling() const {
  return p_->program_.profiler_.enabled();
}
void AmrSystem::reset_profiling() {
  p_->program_.profiler_.reset();
}
std::string AmrSystem::profile_report() const {
  return p_->program_.profiler_.report();
}

int AmrSystem::n_blocks() const {
  return static_cast<int>(p_->blocks.size());
}
std::vector<std::string> AmrSystem::block_names() const {
  std::vector<std::string> out;
  out.reserve(p_->blocks.size());
  for (const auto& b : p_->blocks)
    out.push_back(b.name);
  return out;
}

EffectiveOptionsReport AmrSystem::effective_options_report() const {
  EffectiveOptionsReport report;
  report.runtime = "amr_system";
  report.topology.periodic_x = p_->cfg.periodicity.x;
  report.topology.periodic_y = p_->cfg.periodicity.y;
  report.has_amr = true;
  report.poisson.rhs = p_->p_rhs;
  report.poisson.solver = p_->p_solver;
  report.poisson.bc = p_->p_bc;
  report.poisson.wall = p_->p_wall;
  report.poisson.wall_radius = p_->p_wall_radius;
  report.poisson.epsilon = 1.0;
  report.poisson.abs_tol = static_cast<double>(kMGDefaultAbsTol);
  const bool has_prepared_graph = p_->tagging_spec != nullptr;
  report.amr_refinement.disabled = !has_prepared_graph;
  report.amr_refinement.scalar_threshold_available = false;
  if (has_prepared_graph) {
    report.amr_refinement.disabled_policy = "prepared_tagging_program";
    report.amr_refinement.tagging_provider_identity = p_->tagging_spec->provider_identity;
    report.amr_refinement.tagging_authority = "prepared_program";
  } else {
    report.amr_refinement.disabled_policy = "no_prepared_tagging_authority";
    report.amr_refinement.tagging_authority = "none";
  }
  report.amr_refinement.tagging_execution_provider_identity =
      p_->amr_tagger_component_ ? p_->amr_tagger_component_->provider_identity()
                                : "pops.amr.tagging.native-kokkos-vm@1";
  report.amr_refinement.phi_grad_threshold =
      static_cast<double>(kAmrPhiRefinementDisabledThreshold);
  report.amr_refinement.phi_refinement_enabled = false;
  if (p_->amr_clustering_component_) {
    report.amr_refinement.clustering_provider_identity =
        p_->amr_clustering_component_->provider_identity();
    report.amr_refinement.clustering_authority = "external_prepared_component";
    report.amr_refinement.clustering_parameters_available = false;
  } else {
    report.amr_refinement.clustering_provider_identity = "pops.lib.amr::berger_rigoutsos";
    report.amr_refinement.clustering_authority = "builtin_berger_rigoutsos";
    report.amr_refinement.cluster_min_efficiency =
        p_->cfg.cluster_min_efficiency > 0.0 ? p_->cfg.cluster_min_efficiency : 0.7;
    report.amr_refinement.cluster_min_box_size =
        p_->cfg.cluster_min_box_size > 0 ? p_->cfg.cluster_min_box_size : 1;
    report.amr_refinement.cluster_max_box_size =
        p_->cfg.cluster_max_box_size > 0 ? p_->cfg.cluster_max_box_size : 32;
  }

  for (const Impl::BlockSpec& b : p_->blocks) {
    EffectiveBlockOptions row;
    row.name = b.name;
    row.route = b.is_compiled ? "native_loader" : "native_model";
    row.compiled = b.is_compiled;
    row.transport = b.is_compiled ? "compiled_artifact" : b.spec.transport.get();
    row.source = b.is_compiled ? "compiled_artifact" : b.spec.source.get();
    row.elliptic = b.is_compiled ? "compiled_artifact" : b.spec.elliptic.get();
    row.limiter = b.limiter;
    row.riemann = b.riemann;
    row.recon = b.recon_prim ? "primitive" : "conservative";
    row.time = b.time;
    row.time_method = amr_program_method_label(b.time);
    row.substeps = b.substeps;
    row.stride = b.stride;
    row.evolve = true;
    row.newton = EffectiveNewtonOptions{};
    row.positivity_floor = b.pos_floor;
    row.wave_speed_cache = b.wave_speed_cache;
    row.weno_epsilon = b.weno_epsilon;
    row.gamma = b.gamma;
    if (!b.is_compiled) {
      row.B0 = b.spec.B0;
      row.cs2 = b.spec.cs2;
      row.vacuum_floor = b.spec.vacuum_floor;
      row.qom = b.spec.qom;
      row.q = b.spec.q;
      row.alpha = b.spec.alpha;
      row.n0 = b.spec.n0;
      row.sign = b.spec.sign;
      row.four_pi_G = b.spec.four_pi_G;
      row.rho0 = b.spec.rho0;
    }
    report.blocks.push_back(std::move(row));
  }
  return report;
}
int AmrSystem::n_patches() {
  p_->ensure_built();
  return p_->runtime->n_patches();
}
std::vector<PatchBox> AmrSystem::patch_boxes() {
  p_->ensure_built();
  return p_->runtime->patch_boxes();
}
int AmrSystem::coarse_local_boxes() {
  p_->ensure_built();
  return p_->runtime->coarse_local_boxes();
}
int AmrSystem::coarse_total_boxes() {
  p_->ensure_built();
  return p_->runtime->coarse_total_boxes();
}
double AmrSystem::mass() {
  return mass(std::string());
}
double AmrSystem::mass(const std::string& name) {
  p_->ensure_built();
  return static_cast<double>(p_->runtime->mass(p_->observable_block_index_or_throw(name)));
}
std::vector<double> AmrSystem::density() {
  return density(std::string());
}
std::vector<double> AmrSystem::density(const std::string& name) {
  p_->ensure_built();
  return p_->runtime->density(p_->observable_block_index_or_throw(name));
}
std::vector<double> AmrSystem::potential() {
  p_->ensure_built();
  return p_->runtime->potential();
}

namespace {
// Unqualified conservative-state checkpoint accessors remain one-block conveniences; the storage is
// nevertheless owned by the same AmrRuntime block stack used by multi-block systems.
const char* const kAmrCkptMonoOnly =
    "AmrSystem : unqualified level_state / set_level_state / n_vars require exactly one block; use "
    "the qualified block_level_state / set_block_level_state / block_n_vars accessors.";
}  // namespace

int AmrSystem::n_levels() {
  p_->ensure_built();
  return p_->runtime->nlev();
}
int AmrSystem::max_levels() {
  p_->ensure_built();
  return p_->runtime->max_levels();
}
int AmrSystem::configured_n_levels() {
  return max_levels();
}
int AmrSystem::n_vars() {
  p_->ensure_built();
  if (p_->blocks.size() != 1)
    throw std::runtime_error(kAmrCkptMonoOnly);
  return p_->runtime->block_n_vars(0);
}
std::vector<double> AmrSystem::level_state(int k) {
  p_->ensure_built();
  if (p_->blocks.size() != 1)
    throw std::runtime_error(kAmrCkptMonoOnly);
  return p_->runtime->block_level_state(0, k);
}
std::vector<double> AmrSystem::level_state_global(int k) {
  p_->ensure_built();
  if (p_->blocks.size() != 1)
    throw std::runtime_error(kAmrCkptMonoOnly);
  return p_->runtime->block_level_state_global(0, k);
}
void AmrSystem::set_level_state(int k, const std::vector<double>& s) {
  p_->ensure_built();
  if (p_->blocks.size() != 1)
    throw std::runtime_error(kAmrCkptMonoOnly);
  p_->runtime->set_block_level_state(0, k, s);
}
std::vector<double> AmrSystem::level_potential(int k) {
  p_->ensure_built();
  return p_->runtime->level_potential(k);
}
std::vector<double> AmrSystem::level_potential_global(int k) {
  p_->ensure_built();
  return p_->runtime->level_potential_global(k);
}
void AmrSystem::set_level_potential(int k, const std::vector<double>& p) {
  p_->ensure_built();
  p_->runtime->set_level_potential(k, p);
}
void AmrSystem::set_hierarchy(const std::vector<PatchBox>& boxes) {
  p_->ensure_built();
  if (n_ranks() != 1)
    throw std::runtime_error(
        "AmrSystem::set_hierarchy under MPI requires rebuild_hierarchy with explicit owner ranks");
  rebuild_hierarchy(boxes, std::vector<int>(boxes.size(), 0));
}

// Impose a mid-run hierarchy from a v3 checkpoint (ADC-542). Regroups the flat level-tagged
// box + owner-rank arrays by level and forwards to AmrRuntime::rebuild_hierarchy (all levels rebuilt,
// reusing regrid R6/R7).
void AmrSystem::rebuild_hierarchy(const std::vector<PatchBox>& boxes,
                                  const std::vector<int>& owner_ranks) {
  p_->ensure_built();
  if (owner_ranks.size() != boxes.size())
    throw std::runtime_error(
        "AmrSystem::rebuild_hierarchy : boxes and owner_ranks length mismatch");
  int nlev = 1;
  for (const PatchBox& box : boxes) {
    if (box.level < 1)
      throw std::runtime_error("AmrSystem::rebuild_hierarchy accepts only fine-level patch boxes");
    nlev = std::max(nlev, box.level + 1);
  }
  if (nlev > p_->runtime->max_levels())
    throw std::runtime_error(
        "AmrSystem::rebuild_hierarchy active depth exceeds the resolved hierarchy capacity");
  std::vector<std::vector<PatchBox>> level_boxes(static_cast<std::size_t>(nlev));
  std::vector<std::vector<int>> level_owners(static_cast<std::size_t>(nlev));
  for (std::size_t idx = 0; idx < boxes.size(); ++idx) {
    const int k = boxes[idx].level;
    if (k < 1 || k >= nlev)
      throw std::runtime_error("AmrSystem::rebuild_hierarchy : box level out of range");
    level_boxes[static_cast<std::size_t>(k)].push_back(boxes[idx]);
    level_owners[static_cast<std::size_t>(k)].push_back(owner_ranks[idx]);
  }
  p_->runtime->rebuild_hierarchy(level_boxes, level_owners);
  p_->install_active_temporal_relations();
}

std::vector<int> AmrSystem::rematerialize_hierarchy_ownership(const std::vector<PatchBox>& boxes) {
  p_->ensure_built();
  int nlev = 1;
  std::string request_contract;
  std::exception_ptr local_failure;
  try {
    if (!p_->load_balance_authority_)
      throw std::runtime_error(
          "AmrSystem::rematerialize_hierarchy_ownership requires a prepared load-balance "
          "authority");
    ExactContractBuilder contract;
    contract.text("pops.amr.restart-ownership-request")
        .scalar(std::uint32_t{1})
        .scalar(static_cast<std::uint64_t>(boxes.size()));
    for (const PatchBox& box : boxes) {
      if (box.level < 1)
        throw std::runtime_error(
            "AmrSystem::rematerialize_hierarchy_ownership accepts only fine-level patch boxes");
      const Box2D value{{box.ilo, box.jlo}, {box.ihi, box.jhi}};
      if (value.empty())
        throw std::runtime_error(
            "AmrSystem::rematerialize_hierarchy_ownership received an empty patch box");
      nlev = std::max(nlev, box.level + 1);
      contract.scalar(static_cast<std::int32_t>(box.level))
          .scalar(static_cast<std::int32_t>(box.ilo))
          .scalar(static_cast<std::int32_t>(box.jlo))
          .scalar(static_cast<std::int32_t>(box.ihi))
          .scalar(static_cast<std::int32_t>(box.jhi));
    }
    if (nlev > p_->runtime->max_levels())
      throw std::runtime_error(
          "AmrSystem::rematerialize_hierarchy_ownership exceeds the resolved hierarchy capacity");
    request_contract = std::move(contract).release();
  } catch (...) {
    local_failure = std::current_exception();
  }
  const long failure_count = all_reduce_sum(local_failure ? 1L : 0L);
  if (failure_count != 0) {
    if (n_ranks() == 1 && local_failure)
      std::rethrow_exception(local_failure);
    throw std::runtime_error(
        "AmrSystem::rematerialize_hierarchy_ownership validation failed collectively on " +
        std::to_string(failure_count) + " rank(s)");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"pops.amr.restart-ownership-request", request_contract}}))
    throw std::runtime_error(
        "AmrSystem::rematerialize_hierarchy_ownership request differs across MPI ranks");
  if (boxes.empty())
    return {};

  std::vector<std::vector<std::size_t>> indices(static_cast<std::size_t>(nlev));
  std::vector<std::vector<Box2D>> level_boxes(static_cast<std::size_t>(nlev));
  for (std::size_t index = 0; index < boxes.size(); ++index) {
    const PatchBox& box = boxes[index];
    const auto level = static_cast<std::size_t>(box.level);
    const Box2D value{{box.ilo, box.jlo}, {box.ihi, box.jhi}};
    indices[level].push_back(index);
    level_boxes[level].push_back(value);
  }

  std::vector<int> owners(boxes.size(), -1);
  for (int level = 1; level < nlev; ++level) {
    const auto index = static_cast<std::size_t>(level);
    if (level_boxes[index].empty())
      throw std::runtime_error(
          "AmrSystem::rematerialize_hierarchy_ownership requires contiguous active fine levels");
    const BoxArray box_array(level_boxes[index]);
    const DistributionMapping mapping =
        p_->load_balance_authority_->distribute(box_array, n_ranks());
    for (std::size_t patch = 0; patch < indices[index].size(); ++patch)
      owners[indices[index][patch]] = mapping[static_cast<int>(patch)];
  }
  return owners;
}

std::vector<std::uint8_t> AmrSystem::rematerialize_program_accepted_state(
    const std::vector<std::vector<std::uint8_t>>& source_states,
    const std::vector<std::vector<int>>& source_level_owners,
    const std::vector<std::vector<int>>& target_level_owners) {
  p_->ensure_built();
  runtime::program::AmrProgramRankOwnership source_ownership;
  source_ownership.rank_count = static_cast<int>(source_states.size());
  source_ownership.level_patch_owners = source_level_owners;
  runtime::program::AmrProgramRankOwnership target_ownership;
  target_ownership.rank_count = n_ranks();
  target_ownership.level_patch_owners = target_level_owners;
  return runtime::program::rematerialize_amr_program_accepted_state_bytes(
      source_states, source_ownership, target_ownership, my_rank());
}

void AmrSystem::begin_restart_transaction() {
  p_->ensure_built();
  if (p_->restart_transaction_active_)
    throw std::runtime_error(
        "AmrSystem::begin_restart_transaction : a restart transaction is already active");
  if (p_->has_active_step_transaction_())
    throw std::runtime_error(
        "AmrSystem::begin_restart_transaction : a step transaction is already active");
  p_->restart_snapshot_.capture(*p_);
  p_->runtime->begin_step_rollback_scope();
  p_->restart_transaction_active_ = true;
}

void AmrSystem::commit_restart_transaction() {
  if (!p_->restart_transaction_active_)
    throw std::runtime_error(
        "AmrSystem::commit_restart_transaction : no restart transaction is active");
  p_->runtime->end_step_rollback_scope();
  p_->restart_transaction_active_ = false;
}

void AmrSystem::rollback_restart_transaction() {
  if (!p_->restart_transaction_active_)
    throw std::runtime_error(
        "AmrSystem::rollback_restart_transaction : no restart transaction is active");
  // Drop the active marker before restoration.  A restoration failure is terminal for this bracket,
  // rather than leaving an unusable nested transaction that masks the original exception.
  p_->restart_transaction_active_ = false;
  p_->runtime->end_step_rollback_scope();
  p_->restart_snapshot_.restore(*p_);
  p_->program_.resync_after_restart_rollback("AmrSystem::rollback_restart_transaction:");
}

void AmrSystem::preflight_regrid_on_restart() {
  p_->ensure_built();
  if (!p_->restart_transaction_active_)
    throw std::runtime_error(
        "AmrSystem::preflight_regrid_on_restart requires an active restart transaction");
  if (p_->has_active_step_transaction_())
    throw std::runtime_error(
        "AmrSystem::preflight_regrid_on_restart cannot overlap a step transaction");
  p_->program_.preflight_regrid_on_restart("AmrSystem::preflight_regrid_on_restart:");
}

void AmrSystem::regrid_on_restart() {
  p_->ensure_built();
  if (!p_->restart_transaction_active_)
    throw std::runtime_error("AmrSystem::regrid_on_restart requires an active restart transaction");
  if (p_->has_active_step_transaction_())
    throw std::runtime_error("AmrSystem::regrid_on_restart cannot overlap a step transaction");
  p_->program_.regrid_on_restart("AmrSystem::regrid_on_restart:");
}

int AmrSystem::checkpoint_regrid_count() const {
  return p_->runtime ? p_->runtime->regrid_count() : 0;
}
std::uint64_t AmrSystem::checkpoint_topology_epoch() const {
  return p_->runtime ? p_->runtime->topology_epoch() : 0;
}
void AmrSystem::restore_checkpoint_counters(int regrid_count, std::uint64_t topology_epoch) {
  if (!p_->runtime) {
    if (regrid_count != 0 || topology_epoch != 0)
      throw std::runtime_error(
          "single-block AMR checkpoint cannot restore runtime regrid counters");
    return;
  }
  p_->runtime->restore_checkpoint_counters(regrid_count, topology_epoch);
}
std::vector<std::vector<std::string>> AmrSystem::checkpoint_temporal_relations() const {
  p_->ensure_built();
  if (!p_->runtime)
    return {};
  std::vector<std::vector<std::string>> rows;
  for (const auto& relation : p_->runtime->checkpoint_temporal_relations())
    rows.push_back({std::to_string(relation.parent_level()), std::to_string(relation.child_level()),
                    std::to_string(relation.temporal_ratio().numerator),
                    std::to_string(relation.temporal_ratio().denominator),
                    relation.remainder_policy() == ::pops::amr::RemainderPolicy::IntegralOnly
                        ? "integral_only"
                        : "explicit_final_substep"});
  return rows;
}
std::vector<std::vector<std::string>> AmrSystem::checkpoint_transfer_routes() const {
  std::vector<std::vector<std::string>> rows;
  rows.reserve(p_->bootstrap_subject_routes.size());
  for (const auto& [key, route_identity] : p_->bootstrap_subject_routes) {
    const auto& route = p_->bootstrap_transfer_routes.at(route_identity);
    std::string ghosts;
    for (std::size_t index = 0; index < route.descriptor.ghost_depth.size(); ++index) {
      if (index)
        ghosts += ",";
      ghosts += std::to_string(route.descriptor.ghost_depth[index]);
    }
    rows.push_back({key.first, key.second, route_identity, route.provider_identity,
                    route.kernel_identity, route.descriptor.space, route.descriptor.centering,
                    route.descriptor.representation, route.descriptor.storage,
                    route.descriptor.operation, std::to_string(route.descriptor.order), ghosts,
                    std::to_string(route.descriptor.dimension),
                    std::to_string(route.descriptor.refinement_ratio)});
  }
  return rows;
}

// --- Per-block per-level checkpoint accessors (ADC-509) ---------------------------------------------
// AmrRuntime carries every block's level stack on the shared layout. The named block is resolved to
// its index; one-block callers may also use the unqualified convenience accessors above.
int AmrSystem::block_n_vars(const std::string& name) {
  p_->ensure_built();
  return p_->runtime->block_n_vars(p_->block_index_or_throw(name));
}
std::vector<double> AmrSystem::block_level_state(const std::string& name, int k) {
  p_->ensure_built();
  return p_->runtime->block_level_state(p_->block_index_or_throw(name), k);
}
std::vector<double> AmrSystem::block_level_state_global(const std::string& name, int k) {
  p_->ensure_built();
  return p_->runtime->block_level_state_global(p_->block_index_or_throw(name), k);
}

std::vector<OutputPiece> AmrSystem::output_state_local_pieces(const std::string& name, int k) {
  p_->ensure_built();
  const std::size_t block = p_->block_index_or_throw(name);
  return p_->runtime->output_block_state_local_pieces(block, k);
}

std::vector<PatchBox> AmrSystem::output_geometry_boxes() {
  p_->ensure_built();
  return p_->runtime->output_geometry_boxes();
}

std::vector<OutputPiece> AmrSystem::output_state_root_pieces(const WorldCommunicator& world,
                                                             const std::string& name, int k) {
  return output_pieces_to_root(world,
                               detail::output_collective_identity("AmrSystem", "state", name, k),
                               [&] { return output_state_local_pieces(name, k); });
}

void AmrSystem::set_block_level_state(const std::string& name, int k,
                                      const std::vector<double>& s) {
  p_->ensure_built();
  p_->runtime->set_block_level_state(p_->block_index_or_throw(name), k, s);
}
std::vector<int> AmrSystem::level_owner_ranks(int k) {
  p_->ensure_built();
  return p_->runtime->level_owner_ranks(k);
}
// Full shared aux of a level (ALL components) -- the v3 checkpoint aux payload.
std::vector<double> AmrSystem::level_aux_flat(int k) {
  p_->ensure_built();
  return p_->runtime->level_aux_flat(k);
}
std::vector<double> AmrSystem::level_aux_flat_global(int k) {
  p_->ensure_built();
  return p_->runtime->level_aux_flat_global(k);
}
void AmrSystem::set_level_aux_flat(int k, const std::vector<double>& v) {
  p_->ensure_built();
  p_->runtime->set_level_aux_flat(k, v);
}

// --- ADC-631 multistep history-ring checkpoint / replay seam (Uniform System seam names) -----------
// Thin facade wrappers over detail::AmrHistoryOps on the built AmrRuntime engine, so the SHARED
// python/pops/runtime/_system_io_history.py serialize/restore is reused verbatim: history_global
// returns the per-level slices concatenated into ONE flat buffer (the level axis hidden inside the
// accessor, parity with level_aux_flat), restore_history scatters it back per level. The engine is the
// shared AmrRuntime (install_program forces its build). rebuild_history_slots replays the
// policy-recomputed slots by re-stepping the installed Program closure (owned by this facade).
namespace {
const char* const kAmrHistNoEngine =
    "AmrSystem : multistep history rings require the resolved AmrRuntime engine.";
}  // namespace

std::vector<std::string> AmrSystem::history_names() const {
  return p_->runtime ? detail::AmrHistoryOps::names(*p_->runtime) : std::vector<std::string>{};
}
int AmrSystem::history_depth(const std::string& name) const {
  if (!p_->runtime)
    throw std::runtime_error(kAmrHistNoEngine);
  return detail::AmrHistoryOps::depth(*p_->runtime, name);
}
int AmrSystem::history_ncomp(const std::string& name) const {
  if (!p_->runtime)
    throw std::runtime_error(kAmrHistNoEngine);
  return detail::AmrHistoryOps::ncomp(*p_->runtime, name);
}
bool AmrSystem::history_initialized(const std::string& name) const {
  if (!p_->runtime)
    throw std::runtime_error(kAmrHistNoEngine);
  return detail::AmrHistoryOps::initialized(*p_->runtime, name);
}
int AmrSystem::history_fill_count(const std::string& name) const {
  if (!p_->runtime)
    throw std::runtime_error(kAmrHistNoEngine);
  return detail::AmrHistoryOps::fill_count(*p_->runtime, name);
}
void AmrSystem::set_history_initialized(const std::string& name, bool initialized) {
  if (!p_->runtime)
    throw std::runtime_error(kAmrHistNoEngine);
  detail::AmrHistoryOps::set_initialized(*p_->runtime, name, initialized);
}
void AmrSystem::restore_history_fill_count(const std::string& name, int fill_count) {
  if (!p_->runtime)
    throw std::runtime_error(kAmrHistNoEngine);
  detail::AmrHistoryOps::restore_fill_count(*p_->runtime, name, fill_count);
}
std::vector<double> AmrSystem::history_global(const std::string& name, int slot) const {
  if (!p_->runtime)
    throw std::runtime_error(kAmrHistNoEngine);
  return detail::AmrHistoryOps::global(*p_->runtime, name, slot, pops::n_ranks() != 1);
}
void AmrSystem::restore_history(const std::string& name, int slot,
                                const std::vector<double>& values) {
  if (!p_->runtime)
    throw std::runtime_error(kAmrHistNoEngine);
  detail::AmrHistoryOps::restore(*p_->runtime, name, slot, values);
}
double AmrSystem::history_slot_dt(const std::string& name, int slot) const {
  if (!p_->runtime)
    throw std::runtime_error(kAmrHistNoEngine);
  return detail::AmrHistoryOps::slot_dt(*p_->runtime, name, slot);
}
void AmrSystem::restore_history_slot_dt(const std::string& name, int slot, double dt) {
  if (!p_->runtime)
    throw std::runtime_error(kAmrHistNoEngine);
  detail::AmrHistoryOps::restore_slot_dt(*p_->runtime, name, slot, dt);
}
int AmrSystem::rebuild_history_slots(const std::string& name,
                                     const std::vector<int>& stored_slots) {
  if (!p_->runtime)
    throw std::runtime_error(kAmrHistNoEngine);
  if (!p_->program_.step_)
    throw std::runtime_error(
        "AmrSystem::rebuild_history_slots : no compiled Program is installed; the ring cannot be "
        "replayed (install_program before restart, or checkpoint the ring with Dense())");
  const int depth = detail::AmrHistoryOps::depth(*p_->runtime, name);
  std::vector<int> anchors = stored_slots;
  std::sort(anchors.begin(), anchors.end());
  anchors.erase(std::unique(anchors.begin(), anchors.end()), anchors.end());
  bool dense = static_cast<int>(anchors.size()) == depth;
  for (int index = 0; dense && index < depth; ++index)
    dense = anchors[static_cast<std::size_t>(index)] == index;
  if (dense)
    return 0;
  if (!p_->program_.authorizes_history_replay(name, depth))
    throw std::runtime_error(
        "AmrSystem::rebuild_history_slots: selective replay for history '" + name + "' at depth " +
        std::to_string(depth) +
        " lacks the installed Program's validated native authority; compile this exact ring with "
        "a selective checkpoint policy or use Dense()");
  detail::AuthenticatedHistoryReplayToken replay_authority(name, depth);
  // ADC-635: the replay re-steps the installed Program with regrid ACTIVE. The head-of-step
  // ctx.regrid_if_due(ctx.macro_step()) reads THIS facade's macro_step_: the closure drives it to the
  // engine-supplied per-re-step cursor (m-1-j) so the original in-window regrid schedule fires. m is
  // the facade cursor (primed by the accepted-state reader); both that cursor and the accepted last
  // Program dt are restored on every exit, including a historical Program exception.
  Impl* imp = p_.get();
  const int m = p_->macro_step_;
  const Real accepted_last_dt = p_->program_.last_dt_;
  detail::AmrHistoryOps::ReplayOutcome outcome;
  try {
    outcome = detail::AmrHistoryOps::rebuild_slots(
        replay_authority, *p_->runtime, name, stored_slots, m, p_->program_.substeps_,
        p_->program_.stride_, [imp](double dt, int cursor) {
          imp->macro_step_ = cursor;  // ctx.macro_step() -> facade cursor -> regrid_if_due schedule
          imp->program_.last_dt_ = static_cast<Real>(dt);
          imp->program_.step_(dt);
        });
  } catch (...) {
    p_->macro_step_ = m;
    p_->program_.last_dt_ = accepted_last_dt;
    throw;
  }
  p_->macro_step_ = m;
  p_->program_.last_dt_ = accepted_last_dt;
  p_->last_replay_regrid_steps_ = outcome.fired_regrid_steps;
  return outcome.recomputed;
}
std::vector<int> AmrSystem::last_replay_regrid_steps() const {
  return p_->last_replay_regrid_steps_;
}

}  // namespace pops
