#include <pops/runtime/amr_system.hpp>

#include <pops/runtime/dynamic/abi_key.hpp>  // detail::abi_key_string: ABI key (header-only), compared to the loader's
#include <pops/runtime/config/route_ids.hpp>  // pops::verify_route_manifest (ADC-599: embedded route registry guard)
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>  // detail::dispatch_amr_compiled + build_amr_compiled (shared path)
#include <pops/runtime/amr/amr_runtime.hpp>  // AmrRuntime + AmrRuntimeBlock (multi-block runtime engine)
#include <pops/runtime/amr/bootstrap_transfer_registry.hpp>
#include <pops/runtime/amr/bootstrap_transfer_builtins.hpp>
#include <pops/runtime/program/profiler.hpp>  // Profiler: AMR / MPI phase timings (Spec 5 criterion 43, ADC-479)
#include <pops/runtime/program/program_runtime_state.hpp>  // ProgramRuntimeState: the shared compiled-Program subsystem (ADC-594)
#include <pops/runtime/program/amr_program_checkpoint.hpp>
#include <pops/runtime/program/step_transaction.hpp>  // StepAttemptRejected: atomic public AMR attempts
#include <pops/runtime/program/module_metadata.hpp>  // read_module_metadata / required_blocks / required_solver: install-time validation (ADC-508)
#include <pops/runtime/builders/block/amr_block_seam.hpp>  // ADC-335: per-transport AMR build seam (build_amr_block/_compiled_<transport>)
#include <pops/runtime/builders/factory/model_factory.hpp>  // detail::dispatch_model + compiled bricks
#include <pops/runtime/dynamic/model_registry.hpp>  // validate_transport: single-source transport rejection (ADC-331)
#include <pops/runtime/context/wall_predicate.hpp>  // detail::wall_predicate (wall shared System/AmrSystem)
#include <pops/numerics/time/integrators/implicit_stepper.hpp>  // NewtonOptions + validate_newton_options (shared range check)

#include <algorithm>  // std::find, std::sort (partial IMEX mask resolution: sorted unique indices)
#include <array>      // std::array<int, 3>: named-elliptic-field aux components (ADC-428)
#include <cmath>
#include <cstddef>
#include <limits>  // std::numeric_limits (global step bounds: neutralization to +inf before the min)
#include <pops/runtime/dynamic/dynlib.hpp>  // portable dlopen<->LoadLibraryW layer (ADC-99); <dlfcn.h> on POSIX
#include <functional>
#include <map>  // ell_field_rhs_ named-elliptic RHS closures (ADC-428) + per-PROGRAM-block RuntimeParams store and install_program defaults grouping (ADC-508)
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "amr_native_param_guard.hpp"  // detail::reject_excessive_amr_runtime_params (ADC-514 loader guard)

namespace pops {

namespace {
runtime::amr::TransferCentering bootstrap_centering(const std::string& value) {
  using C = runtime::amr::TransferCentering;
  if (value == "cell") return C::Cell;
  if (value == "face_x") return C::FaceX;
  if (value == "face_y") return C::FaceY;
  if (value == "node") return C::Node;
  throw std::runtime_error("unknown native AMR transfer centering '" + value + "'");
}

runtime::amr::TransferKernelRegistry bootstrap_transfer_kernels() {
  using namespace runtime::amr;
  TransferKernelRegistry registry;
  const auto exact = [](std::string space, std::string centering,
                        std::string representation, std::string operation,
                        int order, std::vector<int> ghost) {
    return [=](const TransferRouteDescriptor& row) {
      return row.space == space && row.centering == centering &&
             row.representation == representation && row.storage == "dense" &&
             row.operation == operation && row.order == order && row.ghost_depth == ghost &&
             row.dimension == 2 && row.refinement_ratio == 2;
    };
  };
  registry.add({
      "pops.lib.amr.transfer::conservative_linear",
      exact("cell", "cell", "conservative", "prolongation", 2, {1}),
      [](const TransferRouteDescriptor&) {
        PreparedTransferKernel kernel;
        kernel.spatial = [](const MultiFab& coarse, MultiFab& fine,
                            const SpatialTransferContext& context) {
          detail::coupler_conservative_linear_to_fine_mb(
              coarse, fine, context.index.coarse_origin, context.index.fine_origin,
              context.index.refinement_ratio);
        };
        return kernel;
      }});
  registry.add({
      "pops.lib.amr.transfer::volume_average",
      exact("cell", "cell", "conservative", "restriction", 1, {0}),
      [](const TransferRouteDescriptor&) {
        PreparedTransferKernel kernel;
        kernel.spatial = [](const MultiFab& fine, MultiFab& coarse,
                            const SpatialTransferContext& context) {
          if (context.index.refinement_ratio != std::vector<int>{2, 2})
            throw std::runtime_error("volume-average restriction ratio mismatch");
          average_down(fine, coarse, context.index.refinement_ratio[0]);
        };
        return kernel;
      }});
  registry.add({
      "pops.lib.amr.transfer::conservative_coarse_fine",
      exact("cell", "cell", "conservative", "coarse_fine_fill", 1, {1}),
      [](const TransferRouteDescriptor&) { return prepare_conservative_coarse_fine(); }});
  const auto face_accepts = [](const TransferRouteDescriptor& row) {
    return row.space == "face" && (row.centering == "face_x" || row.centering == "face_y") &&
           row.representation == "conservative" && row.storage == "dense" &&
           row.operation == "prolongation" && row.order == 2 &&
           row.ghost_depth == std::vector<int>{1} && row.dimension == 2 &&
           row.refinement_ratio == 2;
  };
  registry.add({
      "pops.lib.amr.transfer::face_divergence_preserving", face_accepts,
      [](const TransferRouteDescriptor& row) {
        PreparedTransferKernel kernel;
        (void)row;
        kernel.face_vector = [](const MultiFab& coarse_x, const MultiFab& coarse_y,
                                MultiFab& fine_x, MultiFab& fine_y,
                                const SpatialTransferContext& context) {
          // One prepared vector operation owns both oriented carriers.  The two normal-face
          // reconstructions are applied in the same transaction and preserve the discrete flux
          // balance as a pair; no scalar face route exists.
          detail::bootstrap_prolong_face_vector(
              coarse_x, coarse_y, fine_x, fine_y, context);
        };
        return kernel;
      }});
  registry.add({
      "pops.lib.amr.transfer::node_bilinear",
      exact("node", "node", "primitive", "prolongation", 2, {1}),
      [](const TransferRouteDescriptor&) {
        PreparedTransferKernel kernel;
        kernel.spatial = [](const MultiFab& coarse, MultiFab& fine,
                            const SpatialTransferContext& context) {
          detail::bootstrap_prolong_staggered(
              coarse, fine, TransferCentering::Node, context);
        };
        return kernel;
      }});
  const auto temporal_accepts = [](const TransferRouteDescriptor& row) {
    return row.space != "field" && row.space != "cache" && row.storage == "dense" &&
           row.operation == "temporal_interpolation" && row.order == 2 &&
           row.ghost_depth == std::vector<int>{0} && row.dimension == 2 &&
           row.refinement_ratio == 2;
  };
  registry.add({
      "pops.lib.amr.transfer::linear_time_interpolation", temporal_accepts,
      [](const TransferRouteDescriptor&) { return prepare_linear_time_interpolation(); }});
  registry.add({
      "pops.lib.amr.materializer::elliptic_solve",
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
  registry.add({
      "pops.lib.amr.materializer::patch_topology",
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
          throw std::runtime_error("patch-topology materializer received an incompatible operation");
        };
        return kernel;
      }});
  return registry;
}

std::string amr_time_routes_csv() {
  return std::string(route_token(TimeRouteId::kExplicitSsprk2)) + "|" +
         route_token(TimeRouteId::kSsprk3) + "|" + route_token(TimeRouteId::kImex);
}

bool amr_newton_options_non_default(const NewtonOptions& newton, bool diagnostics = false) {
  return newton.max_iters != kNewtonDefaultMaxIters || newton.rel_tol != kNewtonDefaultRelTol ||
         newton.abs_tol != kNewtonDefaultAbsTol || newton.fd_eps != kNewtonDefaultFdEps ||
         diagnostics || newton.damping != kNewtonDefaultDamping ||
         newton.fail_policy != kNewtonDefaultFailPolicy;
}

EffectiveNewtonOptions amr_effective_newton_options(const NewtonOptions& newton, bool diagnostics) {
  EffectiveNewtonOptions out;
  out.max_iters = newton.max_iters;
  out.rel_tol = static_cast<double>(newton.rel_tol);
  out.abs_tol = static_cast<double>(newton.abs_tol);
  out.fd_eps = static_cast<double>(newton.fd_eps);
  out.damping = static_cast<double>(newton.damping);
  out.fail_policy = newton_fail_policy_name(newton.fail_policy);
  out.diagnostics = diagnostics;
  out.non_default = amr_newton_options_non_default(newton, diagnostics);
  return out;
}
}  // namespace

// resolve_implicit_components (AMR) moved to amr_block_seam.hpp (pops::detail::
// resolve_implicit_components_amr) so the per-transport seam TUs share one definition; otherwise
// unchanged (AmrSystem-specific error wording preserved verbatim).

struct AmrSystem::Impl {
  AmrSystemConfig cfg;

  // Specification of ONE block (frozen at add_block, materialized at lazy build). The facade holds
  // a REGISTRY BY NAME of blocks (cf. design AMR_MULTIBLOCK_DESIGN.md section 0/3): the single-block
  // path goes through AmrCouplerMP (untouched, bit-identical); two or more blocks go through the
  // multi-block runtime engine AmrRuntime (shared hierarchy, co-located summed Poisson).
  struct BlockSpec {
    std::string name;
    // Native ModelSpec path (composed bricks) OR compiled path (.so / add_compiled_model).
    bool is_compiled = false;
    ModelSpec spec;  // ModelSpec path (is_compiled == false)
    std::string limiter = "minmod", riemann = "rusanov";
    bool recon_prim = false;  // recon == "primitive"
    bool imex = false;        // time == "imex": implicit stiff source
    // Partial IMEX mask CARRIED BY THE BLOCK (cf. System::add_block): conserved components handled
    // implicitly, by NAME (implicit_vars) or by physical ROLE (implicit_roles). We STORE the raw
    // strings here (the concrete Model type -- thus cons_vars -- is only resolved at lazy build, in
    // build_multi via dispatch_model); the names/roles -> indices resolution happens there, against the
    // block's conservative descriptor. Empty (default) -> full backward-Euler (all implicit).
    std::vector<std::string> implicit_vars, implicit_roles;
    int substeps = 1;
    int stride = 1;  // hold-then-catch-up cadence (multi-block; cf. AmrRuntimeBlock)
    double gamma = static_cast<double>(kPhysicalDefaultGamma);
    // Compiled SINGLE-BLOCK path: type-erasing builder (AmrCompiledHooks of a concrete AmrCouplerMP),
    // invoked at lazy build when the compiled block is ALONE (AmrCouplerMP path, bit-identical).
    std::function<AmrCompiledHooks(const AmrBuildParams&)> compiled_hooks_builder;
    // Compiled MULTI-BLOCK path (capstone v, multi-block production DSL): type-erasing builder that, on
    // the SHARED layout materialized at lazy build (build_multi), produces the AmrRuntimeBlock of the
    // compiled block -- exactly like dispatch_amr_block for a native block, but with the Model/Limiter/
    // Flux CONCRETE types already captured at add (add_compiled_model) instead of a ModelSpec dispatch.
    // The partial IMEX mask (implicit_vars/roles above) is resolved into indices IN this builder (the
    // concrete Model type -- thus cons_vars -- is known there), just as the native path resolves it in
    // build_multi. Empty for a native block (is_compiled == false).
    AmrCompiledBlockBuilder compiled_block_builder;
    // Initial density of the block (component 0), n*n row-major; targeted by set_density(name, rho).
    bool has_density = false;
    std::vector<double> density;
    // FULL initial conservative state (all components), ncomp*n*n component-major; set by
    // set_conservative_state(name, U). Takes priority over density at seed (cf. make_build_params /
    // build_amr_compiled). SINGLE-BLOCK only (build_multi throws if has_state).
    bool has_state = false;
    std::vector<double> state;
    NewtonOptions newton{};  // IMEX source Newton options (wave 3; single-block AND multi-block)
    bool newton_non_default = false;  // true -> non-default options (.so loader REJECTED: flat ABI)
    bool newton_diagnostics = false;  // newton_report: native MULTI-BLOCK (single/.so REJECTED)
    // TEMPORAL METHOD of the block (time == "ssprk3" -> kSsprk3). 0 == historical forward Euler (default),
    // 1 == kSsprk3 (order 3 + per-stage reflux). Materialized to AmrTimeMethod at build (single-block via
    // make_build_params -> bp.physics.time_method; multi-block via dispatch_amr_block). Mutually exclusive with imex.
    int time_method = 0;
    // Zhang-Shu positivity floor (ADC-259): if > 0, the AMR transport floors the Density-role face
    // states + C/F fine ghost means to >= pos_floor. 0 (default) = inactive, bit-identical. Threaded
    // to dispatch_amr_block (multi-block) and to AmrBuildParams::pos_floor (single-block, build_amr_compiled).
    // COMPILED blocks carry it too (ADC-322): set_compiled_block stores it here from the regenerated
    // .so loader (pops_install_native_amr -> add_compiled_model), so both routings floor like a native block.
    double pos_floor = 0.0;
    // NATIVE per-block RUNTIME params (ADC-514): SHARED value vector the bricks read; empty -> no
    // injection, byte-identical. See AmrSystem::set_block_params (amr_system_params.hpp).
    std::shared_ptr<std::vector<double>> runtime_params;
  };

  std::vector<BlockSpec> blocks;
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
  runtime::amr::TransferRouteRegistry bootstrap_transfer_routes{
      bootstrap_transfer_kernels()};
  std::map<std::pair<std::string, std::string>, std::string> bootstrap_subject_routes;
  std::unordered_map<std::string, std::string> bootstrap_block_subjects;
  std::unordered_map<std::string, std::array<std::string, 2>> bootstrap_face_vectors;
  std::vector<std::uint8_t> program_accepted_state_;
  std::uint64_t program_accepted_state_revision_ = 0;

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

  double refine_threshold =
      static_cast<double>(kAmrRefinementDisabledThreshold);  // no refinement by default
  // ADC-296: refinement variable selected by NAME (refine_var_name) XOR by physical ROLE
  // (refine_var_role). BOTH empty (default) => component 0 (historical density criterion, bit-identical).
  // Resolved PER BLOCK at build_multi against the block's cons_vars (STRICT, no silent comp-0 fallback).
  std::string refine_var_name;
  std::string refine_var_role;
  struct BootstrapTagSpec {
    std::string block;
    std::string variable;
    double threshold = 0.0;
    std::string provider_identity;
  };
  std::unique_ptr<BootstrapTagSpec> bootstrap_tag_spec;
  struct TaggingSpec {
    std::vector<std::string> leaf_blocks;
    std::vector<std::string> leaf_variables;
    std::vector<int> leaf_ops;
    std::vector<double> leaf_thresholds;
    std::vector<int> refine_ops, refine_args;
    std::vector<int> coarsen_ops, coarsen_args;
    int min_cycles = 0;
    int equality_policy = 0;
    int conflict_policy = 0;
    std::string provider_identity;
  };
  std::unique_ptr<TaggingSpec> tagging_spec;
  // PHI tag threshold on |grad phi| (D4): <= 0 => phi does NOT contribute to the tag union (default,
  // bit-identical). > 0 => in multi-block + regrid_every > 0, build_multi sets the engine's phi predicate
  // (set_phi_tag_predicate): refines where |grad phi| (components 1,2 of the shared aux) exceeds this threshold.
  double phi_grad_threshold = 0.0;

  std::vector<double> bz_field;  // coarse B_z(x,y), n*n row-major (set_magnetic_field)
  // Model-NAMED aux fields (ADC-291): component (>= kAuxNamedBase) -> coarse field (n*n row-major).
  // Pending until build: seeded into the single-block coupler (make_build_params -> bp.named_aux) AND
  // pushed to the multi-block runtime (build_multi). Empty -> bit-identical. cf. set_aux_field_component.
  std::map<int, std::vector<double>> named_aux_;
  // Per-field aux HALO policies (ADC-369): component -> uniform policy. Pending until build, then seeded
  // into the engine (bp.named_aux.halo_policies for the coupler; runtime->set_named_aux_bc for the runtime).
  std::map<int, AuxHaloPolicy> named_aux_bc_;
  // NAMED multi-elliptic fields (ADC-428): the native AMR loader declares them (register_elliptic_field)
  // and attaches each field's per-block RHS closure (set_block_elliptic_field) BEFORE the lazy build,
  // when the AmrRuntime engine does not yet exist. We stash both here and replay them on the runtime at
  // build_multi. ell_field_comps_: field -> {phi_comp, gx_comp, gy_comp}. ell_field_rhs_: field -> {block
  // name -> RHS closure}. Empty default -> bit-identical (no named field registered).
  std::map<std::pair<std::string, std::string>, std::array<int, 3>> ell_field_comps_;
  std::map<std::string, std::map<std::string, std::function<void(const MultiFab&, MultiFab&)>>>
      ell_field_rhs_;
  // Complete plans are keyed by the digest of the canonical block-qualified provider identity.
  // The identity string is retained in AmrFieldSolveConfig and checked on duplicate installation,
  // so hash collisions cannot alias fields.  Installed before loaders register (block,key).
  std::map<std::string, AmrFieldSolveConfig> field_plans_;
  // A named elliptic field is solved only by the AmrRuntime engine (not the single-block AmrCouplerMP
  // coupler), so registering one FORCES the runtime engine even for a single block (AmrRuntime accepts >=
  // 1 block). A compiled time Program (ADC-508) likewise forces it: the AmrProgramContext driver needs
  // the per-level state / RHS / average_down seam that lives in AmrRuntime, so install_program sets this
  // even for a single block. false by default -> the engine choice stays blocks.size() >= 2, bit-identical.
  bool force_runtime_ = false;

  std::string p_rhs = "charge_density", p_solver = "geometric_mg", p_bc = "auto", p_wall = "none";
  double p_wall_radius = 0.0;
  // ADC-645: opt-in composite FAC FIELD solve (set_poisson(composite=true)). Default false =
  // the historical Option A coarse solve + gradient injection, bit-identical. The fac knobs follow
  // the <= 0 = kFAC*-default convention (same as the SchurStage block).
  bool p_composite = false;
  int p_fac_max_iters = 0;
  int p_fac_fine_sweeps = 0;
  double p_fac_tol = 0.0;
  double p_fac_coarse_rel_tol = 0.0;
  int p_fac_coarse_cycles = 0;
  bool p_fac_verbose = false;

  bool built = false;
  // RUNTIME FREEZE LIFECYCLE (ADC-592, parity System::Impl::bound_): false while assembling, true once
  // mark_bound() runs (the Python bind flow calls it LAST). 'bound' is DISTINCT from the lazy 'built'
  // materialization (bind runs BEFORE ensure_built): the structural guards refuse a call when EITHER
  // built (the historical lazy-phase message, unchanged) OR bound_ (the new bind-vocabulary message).
  // false for a direct engine script that never binds -> historical behavior unchanged.
  bool bound_ = false;
  // --- single-block path (AmrCouplerMP, untouched: bit-identical to history) ---
  std::shared_ptr<void> coupler_holder;  // keeps the hooks' AmrCouplerMP<Model> alive
  std::function<void(double)> step_fn;
  std::function<double()> max_speed_fn;
  std::function<double()> mass_fn;
  std::function<int()> n_patches_fn;
  std::function<std::vector<PatchBox>()> patch_boxes_fn;
  std::function<int()>
      coarse_local_boxes_fn;  ///< per-rank owned coarse fab count (ADC-319 diagnostic)
  std::function<int()> coarse_total_boxes_fn;  ///< global coarse box count (ADC-319 diagnostic)
  std::function<std::vector<double>()> density_fn;
  std::function<std::vector<double>()> potential_fn;
  // OPTIONAL step bounds of the single block (AMR StabilityPolicy, audit 2026-06): EMPTY hooks if
  // the model does not declare the traits -> single-block step_cfl keeps the historical formula.
  std::function<double()> source_frequency_fn;
  std::function<double()> stability_dt_fn;
  // GLOBAL bounds (AmrSystem::add_dt_bound): registered BEFORE the lazy build, passed to the
  // multi-block engine at its construction; read directly by the single-block step_cfl.
  struct GlobalDtBound {
    std::string label;
    std::function<double()> fn;
  };
  std::vector<GlobalDtBound> dt_bounds;
  std::string
      last_dt_reason;  // ACTIVE bound of the last single-block step_cfl (multi: via runtime)
  // Restoration of the single-block regrid cadence phase (IO v1, parity System::set_clock): the
  // builder populates this hook (writes the coupler's step_state). EMPTY until the block is installed.
  std::function<void(int)> set_macro_step_fn;
  // AMR single-rank CHECKPOINT / RESTART (ADC-65): per-level state accessors + hierarchy
  // imposition, populated by build_amr_compiled (single-block, AmrCouplerMP coupler). The multi-block
  // (runtime engine) does NOT populate them -> the facade methods reject runtime != nullptr.
  std::function<int()> n_levels_fn;
  std::function<int()> n_vars_fn;
  std::function<std::vector<double>(int)> level_state_fn;
  std::function<void(int, const std::vector<double>&)> set_level_state_fn;
  std::function<std::vector<double>(int)> level_potential_fn;
  std::function<void(int, const std::vector<double>&)> set_level_potential_fn;
  std::function<void(const std::vector<PatchBox>&)> set_hierarchy_fn;
  // GLOBAL (np>1 gather) counterparts of the per-level accessors (ADC-509): the facade routes to
  // these under MPI np>1 so a mono-block checkpoint gathers the distributed fabs onto rank 0.
  std::function<std::vector<double>(int)> level_state_global_fn;
  std::function<std::vector<double>(int)> level_potential_global_fn;
  // --- multi-block path (AmrRuntime, shared hierarchy + summed Poisson) ---
  std::shared_ptr<pops::AmrRuntime> runtime;
  double t = 0;
  // AUTHORITATIVE MACRO-STEP counter (parity System::Impl::macro_step_): incremented by
  // AmrSystem::step / step_cfl, read by macro_step(). The engines (AmrRuntime; single-block step_state)
  // hold their OWN cadence counter, synchronized from this one at build and at set_clock.
  int macro_step_ = 0;
  bool clock_restore_pending_ =
      false;  // a set_clock is waiting to be pushed to the engine (at the next step)

  // COMPILED TIME-PROGRAM RUNTIME STATE (ADC-594): the AMR runtime embeds the SAME ProgramRuntimeState
  // struct System::Impl holds (include/pops/runtime/program/program_runtime_state.hpp) -- the SHARED,
  // non-diverging Program subsystem the issue mandates. AMR uses the COMMON subset (step_ / substeps_ /
  // stride_ / installed_hash_ / block_map_ / block_params_ / diagnostics_ / profiler_); the dt_bound_ /
  // cache_ / hist_ fields stay EMPTY here (AMR has no dt-bound seam and defers the history / cache
  // seams -- the documented Uniform/AMR divergence, not a second subsystem). The multi-block AmrRuntime
  // engine is wired to &program_.profiler_ at build (parity with System::Impl); the Profiler's Impl
  // address stays stable (program_ is a stable Impl member). AmrSystem::step routes through
  // run_program_cadence_ (reading program_.step_ / substeps_ / stride_) when a program is installed.
  pops::runtime::program::ProgramRuntimeState program_;
  // NATIVE per-block RUNTIME params (ADC-514, AMR mirror of System::Impl::block_params_): block name ->
  // SHARED value vector set_block_params retargets WITHOUT recompiling; SEPARATE from program_.block_params_.
  std::map<std::string, std::shared_ptr<std::vector<double>>> block_params_;
  // ADC-635: the in-window regrid schedule the LAST rebuild_history_slots fired (for the v3 reader's
  // coherence assertion against the checkpoint fingerprint). Reset each rebuild; empty on a clean window.
  std::vector<int> last_replay_regrid_steps_;

  explicit Impl(const AmrSystemConfig& c) : cfg(c), force_runtime_(c.explicit_bootstrap) {}

  // SUBSTEPS/STRIDE cadence around the installed program closure (parity SystemStepper::run_program_
  // cadence): runs the whole program ONCE over eff_dt = stride*dt when the stride window closes (the
  // clock still ticks every macro-step), subdivided into substeps equal program calls. With 1/1 this is
  // a single program_.step_(dt) call (bit-identical to a bare install). MULTI-BLOCK stride is GLOBAL
  // (whole-program), equal to the native per-block stride only for a single-block Program.
  void run_program_cadence_(double dt) {
    // stride window: program runs only at the END of each stride window ((macro_step_+1) % stride == 0),
    // mirroring AmrRuntime::step / SystemStepper. stride=1 -> always true (every macro-step).
    if ((macro_step_ + 1) % program_.stride_ != 0)
      return;
    const double eff_dt = dt * static_cast<double>(program_.stride_);  // catch-up effective step
    const double h = eff_dt / static_cast<double>(program_.substeps_);
    for (int s = 0; s < program_.substeps_; ++s) {
      // ADC-626/ADC-631: tag the dt that produced this macro-step so a history ring's store_history
      // records the per-slot dt (variable-dt replay). Parity with SystemStepper::run_program_cadence.
      program_.last_dt_ = h;
      program_.step_(h);
    }
  }

  // Pushes macro_step_ to the engine's cadence counter (regrid/stride): multi-block runtime OR
  // single-block coupler step_state. Called at the 1st step after a set_clock (clock_restore_pending_).
  // Without restoration the cadence starts from 0 (default, bit-identical).
  void push_macro_step_to_engine() {
    if (runtime)
      runtime->set_macro_step(macro_step_);
    else if (set_macro_step_fn)
      set_macro_step_fn(macro_step_);
  }

  struct AcceptedSnapshot {
      std::unique_ptr<AmrRuntime::StepSnapshot> runtime;
      std::vector<PatchBox> coupler_boxes;
      std::vector<std::vector<double>> coupler_states;
      std::vector<std::vector<double>> coupler_potentials;
      double time;
      int macro_step;
      bool clock_restore_pending;
      Real last_program_dt;
      std::map<std::string, Real> program_diagnostics;
      pops::runtime::program::CacheManager cache;
      pops::runtime::program::HistoryManager history;
      pops::runtime::program::Profiler profiler;
      std::string last_dt_reason;
      std::vector<int> replay_regrid_steps;
      std::vector<std::uint8_t> program_accepted_state;
      std::uint64_t program_accepted_state_revision;

      explicit AcceptedSnapshot(Impl& impl)
          : time(impl.t),
            macro_step(impl.macro_step_),
            clock_restore_pending(impl.clock_restore_pending_),
            last_program_dt(impl.program_.last_dt_),
            program_diagnostics(impl.program_.diagnostics_),
            cache(impl.program_.cache_),
            history(impl.program_.hist_),
            profiler(impl.program_.profiler_),
            last_dt_reason(impl.last_dt_reason),
            replay_regrid_steps(impl.last_replay_regrid_steps_),
            program_accepted_state(impl.program_accepted_state_),
            program_accepted_state_revision(impl.program_accepted_state_revision_) {
        if (impl.runtime)
          runtime = std::make_unique<AmrRuntime::StepSnapshot>(impl.runtime->step_snapshot());
        else if (impl.n_levels_fn && impl.level_state_fn && impl.level_potential_fn) {
          const int levels = impl.n_levels_fn();
          if (impl.patch_boxes_fn)
            coupler_boxes = impl.patch_boxes_fn();
          coupler_states.reserve(static_cast<std::size_t>(levels));
          coupler_potentials.reserve(static_cast<std::size_t>(levels));
          for (int level = 0; level < levels; ++level) {
            coupler_states.push_back(impl.level_state_fn(level));
            coupler_potentials.push_back(impl.level_potential_fn(level));
          }
        }
      }

      void restore(Impl& impl) const {
        if (runtime && impl.runtime)
          impl.runtime->restore_step_snapshot(*runtime);
        else if (!coupler_states.empty()) {
          if (!impl.set_hierarchy_fn || !impl.set_level_state_fn || !impl.set_level_potential_fn)
            throw std::runtime_error(
                "AmrSystem restart rollback lost the single-block checkpoint hooks");
          impl.set_hierarchy_fn(coupler_boxes);
          for (std::size_t level = 0; level < coupler_states.size(); ++level) {
            impl.set_level_state_fn(static_cast<int>(level), coupler_states[level]);
            impl.set_level_potential_fn(static_cast<int>(level), coupler_potentials[level]);
          }
        }
        impl.t = time;
        impl.macro_step_ = macro_step;
        impl.clock_restore_pending_ = clock_restore_pending;
        impl.program_.last_dt_ = last_program_dt;
        impl.program_.diagnostics_ = program_diagnostics;
        impl.program_.cache_ = cache;
        impl.program_.hist_ = history;
        impl.program_.profiler_ = profiler;
        impl.last_dt_reason = last_dt_reason;
        impl.last_replay_regrid_steps_ = replay_regrid_steps;
        impl.program_accepted_state_ = program_accepted_state;
        impl.program_accepted_state_revision_ = program_accepted_state_revision;
      }
  };

  std::unique_ptr<AcceptedSnapshot> restart_transaction_;
  std::unique_ptr<AcceptedSnapshot> external_step_transaction_;
  bool external_step_transaction_committed_ = false;

  /// Execute one public AMR macro-step against an accepted snapshot.  The AmrRuntime snapshot owns
  /// topology + multi-block/per-level data; this facade layer adds its authoritative clock and the
  /// Program-owned publications that live outside the engine.  Any exception rolls both layers back
  /// before propagating (RejectAttempt remains observable to the Python retry policy).
  template <class Body>
  decltype(auto) execute_step_transaction(Body&& body) {
    if (external_step_transaction_) {
      if (external_step_transaction_committed_)
        throw std::runtime_error(
            "AmrSystem: committed external step transaction must be finalized before another step");
      try {
        return std::forward<Body>(body)();
      } catch (...) {
        external_step_transaction_->restore(*this);
        throw;
      }
    }
    AcceptedSnapshot accepted(*this);
    try {
      return std::forward<Body>(body)();
    } catch (...) {
      accepted.restore(*this);
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

  BCRec poisson_bc() {
    std::string mode = p_bc;
    if (mode == "auto")
      mode = (p_wall == "circle" || !cfg.periodic) ? "dirichlet" : "periodic";
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
  std::function<bool(Real, Real)> wall_active() {
    return detail::wall_predicate(p_wall, p_wall_radius, cfg.L, "AmrSystem::set_poisson");
  }

  // Materializes only hierarchy/system-owned parameters.  Multi-block construction must never select
  // a representative block merely to obtain the common mesh, Poisson, and ownership configuration.
  AmrBuildParams make_layout_params() {
    AmrBuildParams bp;
    // MESH group: coarse geometry + ownership policy.
    bp.mesh.n = cfg.n;
    bp.mesh.L = cfg.L;
    bp.mesh.regrid_every = cfg.regrid_every;
    bp.mesh.distribute_coarse = cfg.distribute_coarse;
    bp.mesh.coarse_max_grid = cfg.coarse_max_grid;
    // REGRID group: refinement threshold.
    bp.regrid.threshold = refine_threshold;
    // POISSON group: coarse Poisson BC + conductive wall.
    bp.poisson.bc = poisson_bc();
    bp.poisson.wall = wall_active();
    // ADC-645: opt-in composite FAC field solve + its knobs (default false/0 -> Option A, inert).
    bp.poisson.composite = p_composite;
    bp.poisson.fac_max_iters = p_fac_max_iters;
    bp.poisson.fac_fine_sweeps = p_fac_fine_sweeps;
    bp.poisson.fac_tol = p_fac_tol;
    bp.poisson.fac_coarse_rel_tol = p_fac_coarse_rel_tol;
    bp.poisson.fac_coarse_cycles = p_fac_coarse_cycles;
    bp.poisson.fac_verbose = p_fac_verbose;
    return bp;
  }

  // Materializes the frozen parameters of the SINGLE-BLOCK path lazy build (AmrCouplerMP) from
  // the system-owned layout parameters plus the exactly-one installed block. Common to both
  // single-block paths (native ModelSpec and add_compiled_model): both instantiate the SAME shared
  // builder (detail::build_amr_compiled). UNTOUCHED by multi-block -> single-block bit-identical.
  AmrBuildParams make_build_params() {
    if (blocks.size() != 1)
      throw std::runtime_error("AmrSystem::make_build_params requires exactly one block");
    const BlockSpec& b = blocks.front();
    AmrBuildParams bp = make_layout_params();
    // PHYSICS group: block-owned physical and temporal treatment.
    bp.physics.gamma = b.gamma;
    bp.physics.substeps = b.substeps;
    bp.physics.recon_prim = b.recon_prim;
    bp.physics.imex = b.imex;
    bp.physics.time_method = b.time_method;
    bp.physics.newton_options = b.newton;
    bp.physics.pos_floor = b.pos_floor;
    // INITIAL DATA group: density (historical) OR full conservative state (priority at seed).
    bp.initial.has_density = b.has_density;
    bp.initial.density = b.density;
    bp.initial.has_state = b.has_state;
    bp.initial.state = b.state;
    // NAMED AUX group: model-named aux fields (ADC-291) + per-field halo policies (ADC-369).
    bp.named_aux.fields = named_aux_;
    bp.named_aux.halo_policies = named_aux_bc_;
    // RUNTIME group (ADC-514): the single block's SHARED param vector; empty -> count 0 -> no injection
    // (byte-identical). The mono coupler re-injects it into cpl->model_ each macro-step.
    bp.runtime.values = b.runtime_params;
    bp.runtime.count = b.runtime_params ? static_cast<int>(b.runtime_params->size()) : 0;
    return bp;
  }

  // Installs the type-erased closures of the SINGLE-BLOCK path (AmrCouplerMP).
  void install(AmrCompiledHooks&& h) {
    coupler_holder = std::move(h.coupler_holder);  // LIFETIME tier
    // BASE tier: macro-step + primary observables.
    step_fn = std::move(h.base.step);
    max_speed_fn = std::move(h.base.max_speed);
    mass_fn = std::move(h.base.mass);
    n_patches_fn = std::move(h.base.n_patches);
    density_fn = std::move(h.base.density);
    potential_fn = std::move(h.base.potential);
    // STABILITY tier: patch signatures + optional step bounds (empty without trait, bit-identical).
    patch_boxes_fn = std::move(h.stability.patch_boxes);
    source_frequency_fn = std::move(h.stability.source_frequency);
    stability_dt_fn = std::move(h.stability.stability_dt);
    // CHECKPOINT tier: cadence phase (IO v1) + per-level state/phi + hierarchy (ADC-65).
    set_macro_step_fn = std::move(h.checkpoint.set_macro_step);
    n_levels_fn = std::move(h.checkpoint.n_levels);
    n_vars_fn = std::move(h.checkpoint.n_vars);
    level_state_fn = std::move(h.checkpoint.level_state);
    set_level_state_fn = std::move(h.checkpoint.set_level_state);
    level_potential_fn = std::move(h.checkpoint.level_potential);
    set_level_potential_fn = std::move(h.checkpoint.set_level_potential);
    set_hierarchy_fn = std::move(h.checkpoint.set_hierarchy);
    // MPI-GATHER tier: coarse ownership diagnostic (ADC-319) + np>1 gather counterparts (ADC-509).
    coarse_local_boxes_fn = std::move(h.mpi_gather.coarse_local_boxes);
    coarse_total_boxes_fn = std::move(h.mpi_gather.coarse_total_boxes);
    level_state_global_fn = std::move(h.mpi_gather.level_state_global);
    level_potential_global_fn = std::move(h.mpi_gather.level_potential_global);
    built = true;
  }

  bool multi_block() const { return blocks.size() >= 2 || force_runtime_; }

  // Builds the MULTI-BLOCK runtime engine (AmrRuntime): one common SharedAmrLayout (shared
  // hierarchy, frozen), then EACH block materializes its type-erased AmrRuntimeBlock on it (via its
  // block_builder, which captures the concrete Model/Limiter/Flux). The coarse Poisson is SUMMED and
  // CO-LOCATED (Sum_b elliptic_rhs_b(U_b) read at the same cells of the shared coarse grid).
  void build_multi() {
    // MULTI-BLOCK set_conservative_state (wave 3 audit): the full state is now THREADED to the
    // NATIVE builder (dispatch_amr_block -> build_amr_block, seed coupler_write_coarse_state +
    // injection to the fine levels, takes priority over density). The COMPILED (.so) path does not
    // transport it (frozen loader ABI) -> explicit rejection, never a silent density fallback.
    for (const auto& b : blocks)
      if (b.has_state && b.is_compiled)
        throw std::runtime_error(
            "AmrSystem::set_conservative_state : not transported by the compiled .so loader (block "
            "'" +
            b.name + "') in multi-block ; use a native block pops.Model(...), or set_density.");
    // ADC-645: the composite FAC FIELD solve lives on the single-block coupler (AmrCouplerMP); the
    // multi-block AmrRuntime engine has no composite elliptic path. Refuse loud, never a silent
    // fallback to the Option A solve the caller explicitly opted out of.
    if (p_composite)
      throw std::runtime_error(
          "AmrSystem::set_poisson : composite=true is single-block only (the composite FAC field "
          "solve lives on the AmrCouplerMP coupler) ; the multi-block AmrRuntime engine solves the "
          "shared coarse Poisson (Option A). Use one block, or composite=false.");
    AmrBuildParams bp = make_layout_params();  // hierarchy/system authority; no representative block
    // SINGLE-LEVEL Program layout (epic ADC-508): a compiled time Program forced onto the runtime engine
    // for a SINGLE block builds a coarse-only hierarchy WHEN NO REFINEMENT IS CONFIGURED, so a
    // no-refinement single-block AMR Program is BIT-IDENTICAL to the same Program on System (the
    // must-pass parity gate). force_runtime_ is set only by install_program; a genuine multi-block
    // (>= 2) AMR keeps the historical 2-level seed.
    //
    // REFINEMENT WINS OVER THE COARSE-ONLY OPT-IN (ADC-634): a coarse-only layout has a single-level
    // template (make_shared_amr_layout single_level -> S.ba == {coarse}), so the regrid has NO fine
    // BoxArray to grow into and the hierarchy stays nlev == 1 FOREVER, even with a tag predicate armed.
    // When the user configured refinement (set_refinement -> refine_threshold below the disabled
    // sentinel, or set_phi_refinement -> phi_grad_threshold > 0), the single-block Program must get the
    // 2-level template so an active regrid can seed / retain the fine level. Independent of
    // regrid_every so a regrid_every == 0 frozen-seed run and a regrid_every > 0 run at the same
    // threshold keep the SAME level count (the null-regrid parity channel). No set_refinement call ->
    // refine_threshold stays the sentinel -> coarse-only -> the System parity gate is untouched.
    const bool refinement_active =
        refine_threshold < static_cast<double>(kAmrRefinementDisabledThreshold) ||
        phi_grad_threshold > 0.0 || bootstrap_tag_spec != nullptr || tagging_spec != nullptr;
    const bool program_single_level = force_runtime_ && blocks.size() == 1 && !refinement_active;
    const int initial_levels = cfg.explicit_bootstrap ? 1 : (program_single_level ? 1 : 2);
    const detail::SharedAmrLayout S = detail::make_shared_amr_layout_levels(bp, initial_levels);
    std::vector<pops::AmrRuntimeBlock> rblocks;
    rblocks.reserve(blocks.size());
    for (auto& b : blocks) {
      if (b.is_compiled) {
        // A compiled block without a runtime builder (empty multi_builder) CANNOT go multi-block:
        // this is the case of an OLD .so loader (generated/compiled against a header earlier than this
        // capstone, which only called set_compiled_block with the mono_builder). We raise a CLEAR
        // error rather than calling an empty std::function (opaque std::bad_function_call); the
        // remedy is to regenerate the loader (dsl.compile_native(target='amr_system')).
        if (!b.compiled_block_builder)
          throw std::runtime_error(
              "AmrSystem : compiled block '" + b.name +
              "' without multi-block builder (.so loader predating multi-block production DSL "
              "support). "
              "Regenerate the loader via dsl.compile_native(target='amr_system') / "
              "compile(backend='production', target='amr_system').");
        // Compiled MULTI-BLOCK path (capstone v): the CONCRETE Model/Limiter/Flux are already captured
        // in the builder (add_compiled_model), we invoke it on the SHARED layout. It allocates the
        // level stack of the block on S and captures the scheme, EXACTLY like dispatch_amr_block for a
        // native block (the builder CALLS it internally). It resolves the partial IMEX mask ITSELF into
        // component indices against cons_vars of the concrete Model (the raw implicit_vars/roles are
        // passed to it). No throw: the 2nd compiled block (or a mix of compiled + native) is wired.
        // Newton options NOT transported by the .so loader builder (ABI frozen at generation):
        // explicit rejection rather than a silent iters=2 (regenerate the loader = dedicated follow-up).
        if (b.newton_non_default)
          throw std::runtime_error(
              "AmrSystem : Newton options are not transported by the compiled .so loader "
              "(block '" +
              b.name + "') ; use a native block pops.Model(...) in multi-block.");
        // newton_diagnostics report likewise: the .so loader builder allocates no NewtonReport nor
        // threads it (flat ABI). Explicit rejection (defense in depth; the Python facade already
        // filters it upstream) rather than a silently empty report.
        if (b.newton_diagnostics)
          throw std::runtime_error(
              "AmrSystem : newton_diagnostics (newton_report) is not transported by the "
              "compiled .so loader (block '" +
              b.name + "') ; use a native block pops.Model(...).");
        // Zhang-Shu positivity floor (ADC-322): the AmrCompiledBlockBuilder now carries a floor slot,
        // so a loader regenerated against this header floors the Density-role face states like a native
        // block (forwarded to dispatch_amr_block -> build_amr_block). b.pos_floor == 0 for an OLDER .so
        // (it never marshals the field) -> inactive, bit-identical. No reject.
        // Runtime params (ADC-514): thread the SHARED vector so the multi-block builder re-injects it
        // each macro-step (empty for a param-free model -> no injection).
        rblocks.push_back(b.compiled_block_builder(
            S, b.name, b.density, b.has_density, b.gamma, b.substeps, b.recon_prim, b.imex,
            b.stride, b.implicit_vars, b.implicit_roles, b.pos_floor, b.runtime_params));
        continue;
      }
      // Native ModelSpec path: model dispatch -> concrete type, then spatial scheme dispatch
      // -> build_amr_block (allocates the block's level stack on the SHARED layout + closures).
      // The block density is carried by the BlockSpec (set_density(name) targets it). The partial IMEX
      // mask (implicit_vars / implicit_roles) is resolved HERE into component indices, against the
      // conservative descriptor of the concrete Model type (cons_vars), then threaded to build_amr_block.
      // Transport-axis seam (ADC-335): each per-transport TU (python/amr_block_<transport>.cpp) runs the
      // SAME dispatch_amr_block as before (build_amr_block_for), but instantiates only its transport's
      // leaves. The impl-mask resolution + temporal-method mapping move into the seam. The string if/else
      // mirrors detail::dispatch_transport (same unknown-transport message).
      const detail::AmrBlockBuildArgs ba{b.spec,
                                         b.name,
                                         b.limiter,
                                         b.riemann,
                                         b.density,
                                         b.has_density,
                                         b.gamma,
                                         b.substeps,
                                         b.recon_prim,
                                         b.imex,
                                         b.stride,
                                         b.implicit_vars,
                                         b.implicit_roles,
                                         b.newton,
                                         b.has_state ? &b.state : nullptr,
                                         b.newton_diagnostics,
                                         b.time_method,
                                         b.pos_floor};
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
    runtime =
        std::make_shared<pops::AmrRuntime>(S.geom, S.runtime_hierarchy(), S.poisson_bc, std::move(rblocks),
                                           S.base_per, S.replicated_coarse, S.wall);
    // The authored AMRTransfer registry is the authority for every public-DSL block's runtime
    // coarse/fine and temporal routes. Resolve the exact owner-qualified subject once, then retain
    // only prepared native callables in AmrRuntime; no string/kernel switch survives into stepping.
    for (const auto& [subject, block_name] : bootstrap_block_subjects) {
      const int block = block_index(block_name);
      const auto coarse_fine_id = bootstrap_subject_routes.find({subject, "coarse_fine_fill"});
      const auto temporal_id = bootstrap_subject_routes.find({subject, "temporal_interpolation"});
      if (block < 0 || coarse_fine_id == bootstrap_subject_routes.end() ||
          temporal_id == bootstrap_subject_routes.end())
        throw std::runtime_error(
            "AmrSystem : state transfer authority lacks exact coarse/fine or temporal route");
      const auto& coarse_fine = bootstrap_transfer_routes.at(coarse_fine_id->second);
      const auto& temporal = bootstrap_transfer_routes.at(temporal_id->second);
      if (!coarse_fine.executable.coarse_fine || !temporal.executable.temporal ||
          coarse_fine.descriptor.refinement_ratio != temporal.descriptor.refinement_ratio)
        throw std::runtime_error(
            "AmrSystem : state transfer authority did not prepare compatible native callables");
      runtime->set_block_transfer_authority(
          static_cast<std::size_t>(block), coarse_fine.executable, temporal.executable,
          coarse_fine.descriptor.refinement_ratio);
    }
    // AMR / MPI PROFILING (Spec 5 criterion 43, ADC-479): wire the facade-owned Profiler into the
    // engine so it times its AMR phases (regrid / fill_boundary / average_down) into the SAME table
    // profile_report() renders. Non-owning: program_.profiler_ outlives runtime (both live on the
    // Impl, stable address). When profiling is disabled the engine never touches it (enabled()-guarded).
    runtime->set_profiler(&program_.profiler_);
    for (const auto& [subject, array] : bootstrap_arrays)
      runtime->register_bootstrap_staggered_field(
          subject, bootstrap_centering(array.centering), array.ncomp, array.initial_values);
    if (tagging_spec) {
      std::vector<AmrRuntime::TaggingProgram::Leaf> leaves;
      leaves.reserve(tagging_spec->leaf_ops.size());
      for (std::size_t index = 0; index < tagging_spec->leaf_ops.size(); ++index) {
        const int block = block_index(tagging_spec->leaf_blocks[index]);
        if (block < 0 || blocks[static_cast<std::size_t>(block)].is_compiled)
          throw std::runtime_error(
              "resolved AMR tagging names an unknown or compiled block");
        const int component = detail::resolve_selected_component(
            "AmrSystem::resolved tagging", tagging_spec->leaf_blocks[index],
            runtime->block_cons_vars(static_cast<std::size_t>(block)),
            tagging_spec->leaf_variables[index], "");
        leaves.push_back(AmrRuntime::TaggingProgram::Leaf{
            static_cast<std::size_t>(block), component, tagging_spec->leaf_ops[index],
            static_cast<Real>(tagging_spec->leaf_thresholds[index])});
      }
      runtime->set_tagging_program(
          std::move(leaves), tagging_spec->refine_ops, tagging_spec->refine_args,
          tagging_spec->coarsen_ops, tagging_spec->coarsen_args,
          tagging_spec->min_cycles, tagging_spec->equality_policy,
          tagging_spec->conflict_policy, tagging_spec->provider_identity);
    }
    if (cfg.explicit_bootstrap) {
      if (tagging_spec) {
        // The resolved data-only graph was installed above and is shared by bootstrap and regrid.
      } else if (bootstrap_tag_spec) {
        const int b = block_index(bootstrap_tag_spec->block);
        if (b < 0 || blocks[static_cast<std::size_t>(b)].is_compiled)
          throw std::runtime_error(
              "explicit AMR bootstrap tag provider names an unknown or compiled block");
        const int component = detail::resolve_selected_component(
            "AmrSystem::bootstrap tagging", bootstrap_tag_spec->block,
            runtime->block_cons_vars(static_cast<std::size_t>(b)),
            bootstrap_tag_spec->variable, "");
        runtime->set_bootstrap_threshold_tag(
            static_cast<std::size_t>(b), component,
            static_cast<Real>(bootstrap_tag_spec->threshold),
            bootstrap_tag_spec->provider_identity);
      } else {
        const bool selected = !refine_var_name.empty() || !refine_var_role.empty();
        for (std::size_t b = 0; b < blocks.size(); ++b) {
          int component = 0;
          if (selected) {
            if (blocks[b].is_compiled)
              throw std::runtime_error(
                  "explicit AMR bootstrap tag selector requires a native block");
            component = detail::resolve_selected_component(
                "AmrSystem::bootstrap tagging", blocks[b].name,
                runtime->block_cons_vars(b), refine_var_name, refine_var_role);
          }
          runtime->set_bootstrap_threshold_tag(
              b, component, static_cast<Real>(refine_threshold));
        }
      }
    }
    // Model-NAMED aux fields (ADC-291): push the pending coarse fields into the runtime engine, which
    // re-applies them onto the shared aux each solve_fields (so they persist across the union regrid)
    // and injects them to the fine levels. Empty -> no-op (bit-identical).
    for (const auto& kv : named_aux_)
      runtime->set_named_aux(kv.first, std::vector<Real>(kv.second.begin(), kv.second.end()));
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
                                    kv.second[2]);
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
    // TAG-UNION REGRID (capstone Phase 2, C.6): if regrid_every > 0, we ACTIVATE the engine's
    // cadence and set the PER-BLOCK tag predicate (D1). The criterion tags where the SELECTED variable
    // of the block exceeds refine_threshold -> the UNION of the block tags refines where ANY block
    // exceeds it. By DEFAULT the variable is component 0 (historical density criterion, like the
    // single-block path AmrCouplerMP which tags a(i,j,0) > threshold); ADC-296 lets set_refinement pick
    // it PER BLOCK by name/role, resolved against the block's cons_vars (STRICT: absent -> explicit
    // error, never a silent comp-0 fallback). refine_threshold == 1e30 (default, no refinement) -> no
    // tag -> grid unchanged even if regrid_every > 0 (consistent no-op). regrid_every == 0 ->
    // set_regrid(0) -> FROZEN hierarchy, bit-identical to before this PR.
    const Real thr = static_cast<Real>(refine_threshold);
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
    if (cfg.regrid_every > 0) {
      if (tagging_spec) {
        // The engine evaluates the installed refine/coarsen graph directly.
      } else if (bootstrap_tag_spec) {
        const int b = block_index(bootstrap_tag_spec->block);
        if (b < 0 || blocks[static_cast<std::size_t>(b)].is_compiled)
          throw std::runtime_error(
              "runtime AMR tag provider names an unknown or compiled block");
        const int component = detail::resolve_selected_component(
            "AmrSystem::runtime tagging", bootstrap_tag_spec->block,
            runtime->block_cons_vars(static_cast<std::size_t>(b)),
            bootstrap_tag_spec->variable, "");
        runtime->set_block_tag_predicate(
            static_cast<std::size_t>(b),
            [threshold = static_cast<Real>(bootstrap_tag_spec->threshold), component](
                const ConstArray4& a, int i, int j) {
              return a(i, j, component) > threshold;
            });
      } else {
        const bool selected = !refine_var_name.empty() || !refine_var_role.empty();
        for (std::size_t b = 0; b < blocks.size(); ++b) {
          int comp = 0;  // default: component 0 (bit-identical density criterion)
          if (selected) {
            // The compiled .so flat-ABI block carries no role table on its runtime side: a non-default
            // selector there is REFUSED (comp-0 only), not silently ignored (mirror of the other .so rejects).
            if (blocks[b].is_compiled)
              throw std::runtime_error(
                  "AmrSystem::set_refinement : variable/role selector not supported on the compiled "
                  ".so block '" + blocks[b].name +
                  "' (component 0 only) ; use a native block pops.Model(...)");
            comp = detail::resolve_selected_component(
                "AmrSystem::set_refinement", blocks[b].name,
                runtime->block_cons_vars(b), refine_var_name, refine_var_role);
          }
          runtime->set_block_tag_predicate(
              b, [thr, comp](const ConstArray4& a, int i, int j) {
                return a(i, j, comp) > thr;
              });
        }
      }
      // PHI PREDICATE (D4): if the user set a |grad phi| threshold (set_phi_refinement > 0),
      // we wire the engine's phi predicate (read on the shared aux, components 1,2 = grad phi in x,y).
      // It is ADDED to the union of the per-block density predicates: the grid refines where any
      // block exceeds refine_threshold OR |grad phi| exceeds gthr. Physical diocotron criterion (ring
      // edge = potential gradient). <= 0 (default) -> not wired -> phi does not contribute (bit-identical).
      if (phi_grad_threshold > 0.0) {
        const Real gthr = static_cast<Real>(phi_grad_threshold);
        runtime->set_phi_tag_predicate([gthr](const ConstArray4& a, int i, int j) {
          const Real gx = a(i, j, 1), gy = a(i, j, 2);
          return std::sqrt(gx * gx + gy * gy) > gthr;
        });
      }
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
    // UNLOCK (capstone Phase 2, C.6): multi-block + regrid_every > 0 IS NOW SUPPORTED
    // (the AmrRuntime engine carries the tag-union regrid, cf. build_multi -> set_regrid +
    // set_block_tag_predicate). The old REFUSAL (the multi-block hierarchy was FROZEN) is lifted: the
    // grid actually re-grids from the union of the tags of all blocks. regrid_every == 0
    // stays a frozen hierarchy (regrid never called), bit-identical to Phase 1.
    if (multi_block() || cfg.explicit_bootstrap) {
      build_multi();
      return;
    }

    // --- SINGLE-BLOCK path (AmrCouplerMP, untouched: bit-identical to history) ---
    const BlockSpec& b = blocks.front();
    // ADC-296: the name/role refinement selector is resolved per block by the MULTI-BLOCK runtime engine
    // (build_multi -> resolve_selected_component). The single-block AmrCouplerMP path tags on component 0
    // only; a selector requested but staying single-block would be SILENTLY ignored (the comp-0 fallback
    // this milestone forbids), so we REFUSE it explicitly -- same pattern as the implicit-mask reject
    // below. The default (empty selector) stays component 0, bit-identical.
    if (!refine_var_name.empty() || !refine_var_role.empty())
      throw std::runtime_error(
          "AmrSystem::set_refinement : the variable/role selector is only wired in MULTI-BLOCK "
          "(>= 2 add_block, union-of-tags runtime engine). In single-block the AmrCouplerMP path "
          "refines on component 0 only : drop variable=/role= or add a 2nd block.");
    // The single-block IMEX goes through AmrCouplerMP (advance_amr imex flag), which carries NO partial
    // IMEX mask (FULL backward-Euler). A mask requested but staying SINGLE-BLOCK would thus be SILENTLY
    // ignored -> we REFUSE it explicitly (the partial mask requires the multi-block runtime engine).
    if (!b.implicit_vars.empty() || !b.implicit_roles.empty())
      throw std::runtime_error(
          "AmrSystem : implicit_vars / implicit_roles (partial IMEX mask) are only wired in "
          "MULTI-BLOCK (>= 2 add_block, runtime engine). In single-block the IMEX handles ALL "
          "components implicitly (full backward-Euler) : remove the mask or add a 2nd block.");
    // SINGLE-BLOCK NEWTON OPTIONS (wave 3, settled): NOW WIRED on the AmrCouplerMP coupler
    // (make_build_params -> bp.physics.newton_options -> build_amr_compiled -> cpl->step -> advance_amr ->
    // backward_euler_source). No more rejection of b.newton_non_default here: a single IMEX block with
    // newton_max_iters/rel_tol/abs_tol/fd_eps/damping/fail_policy runs correctly. Default = historical
    // iters=2 (bit-identical). Still NOT wired in single-block: the aggregated Newton REPORT
    // (newton_report = multi-block engine only; threading it through the coupler subcycling would be
    // invasive). Therefore diagnostics and warn policy are refused rather than becoming silently empty.
    if (b.newton_diagnostics)
      throw std::runtime_error(
          "AmrSystem : newton_diagnostics (newton_report) is only wired in MULTI-BLOCK "
          "(AmrRuntime runtime engine). In single-block the Newton OPTIONS "
          "(newton_max_iters/rel_tol/"
          "abs_tol/fd_eps/damping/fail_policy) are wired, but not the aggregated report : add a "
          "2nd block for newton_report, use newton_fail_policy='throw', or a single-level System "
          "for the full report.");
    if (b.newton.fail_policy == NewtonOptions::kFailWarn)
      throw std::runtime_error(
          "AmrSystem : newton_fail_policy='warn' requires a structured Newton report, which is "
          "only "
          "wired in MULTI-BLOCK (AmrRuntime runtime engine). Add a 2nd block for newton_report, "
          "use newton_fail_policy='throw', or use a single-level System for warn diagnostics.");
    const AmrBuildParams bp = make_build_params();
    if (b.is_compiled) {  // compiled path: the builder freezes the types (Model, Limiter, Flux)
      // Zhang-Shu positivity floor (ADC-322): bp.physics.pos_floor (= b.pos_floor, set by set_compiled_block
      // from the regenerated loader) flows into the SAME build_amr_compiled leaf as a native block ->
      // cpl->step / advance_transport floor the Density-role face states. An OLDER .so never marshals
      // the field, so b.pos_floor stays 0 -> inactive, bit-identical. No reject.
      install(b.compiled_hooks_builder(bp));
      return;
    }
    // Native ModelSpec path: the model dispatch resolves the concrete type, then the SAME
    // spatial scheme dispatch + coupler build as add_compiled_model (detail, shared).
    // Transport-axis seam (ADC-335): the single-block (AmrCouplerMP) build, one per-transport TU
    // (python/amr_compiled_<transport>.cpp). bp already bundles every single-block parameter. Same
    // unknown-transport message as detail::dispatch_transport.
    // Transport dispatch mirrors detail::dispatch_transport (ADC-641): validate_transport preserves the
    // unknown_transport_msg byte-for-byte, then the switch on the typed TransportRouteId routes to the
    // per-transport single-block seam.
    validate_transport(b.spec.transport);
    switch (parse_transport_route(b.spec.transport)) {
      case TransportRouteId::kExb:
        install(detail::build_amr_compiled_exb(b.spec, b.limiter, b.riemann, bp));
        break;
      case TransportRouteId::kCompressible:
        install(detail::build_amr_compiled_compressible(b.spec, b.limiter, b.riemann, bp));
        break;
      case TransportRouteId::kIsothermal:
        install(detail::build_amr_compiled_isothermal(b.spec, b.limiter, b.riemann, bp));
        break;
    }
  }
};

namespace {
// UPSTREAM configuration guard (ADC-299): validate the AmrSystemConfig invariants BEFORE constructing
// Impl. The AMR Impl ctor is trivial (it only stores cfg, allocating nothing from n), so unlike System
// nothing is built before the check; we still validate ahead of Impl for parity with System and to keep
// every config rejection at a single upstream point. n was already guarded (n == 0 -> nn = n*n = 0 -> a
// division by zero in set_conservative_state, U.size() % nn, and an empty coarse grid downstream); L,
// regrid_every and coarse_max_grid were unchecked and reach the lazy build (dx, regrid cadence, coarse
// tiling) as is.
void validate_amr_system_config(const AmrSystemConfig& c) {
  if (c.n < 1)
    throw std::runtime_error("AmrSystem : n >= 1 required (coarse cells per direction) ; got n = " +
                             std::to_string(c.n));
  if (!(c.L > 0.0))
    throw std::runtime_error("AmrSystem : L > 0 required (square domain [0,L]^2) ; got L = " +
                             std::to_string(c.L));
  if (c.regrid_every < 0)
    throw std::runtime_error(
        "AmrSystem : regrid_every >= 0 required (0 = never regrid after init) ; "
        "got regrid_every = " +
        std::to_string(c.regrid_every));
  if (c.level_count < 1)
    throw std::runtime_error(
        "AmrSystem : level_count >= 1 required");
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
  p_ = std::make_unique<Impl>(c);
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
                          double positivity_floor) {
  require_assembling_amr(p_->bound_, "add_block");  // frozen once pops.bind completes (ADC-592)
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::add_block : the system is already built (call "
        "add_block before any step/mass/density)");
  // Completeness contract of the model (ADC-290, parity with System::add_block): transport / elliptic
  // must be chosen explicitly. Validated before the transport string routing (build_multi /
  // build_amr_compiled), so a default-constructed ModelSpec fails clearly instead of silently
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
  // IMEX source Newton options grouped into a POD (ADC-214; wave 3 audit, parity
  // System::add_block). Defaults {} = historical constants (2 / 0 / 0 / 1e-7 / 1.0 / none),
  // bit-identical. SUPPORT: MULTI-BLOCK engine (AmrRuntime) only -- the single-block (AmrCouplerMP)
  // keeps iters=2 frozen, non-default options are REJECTED there at build (ensure_built), never ignored.
  // Range check shared with System::add_block (validate_newton_options, in implicit_stepper.hpp).
  validate_newton_options(newton, "AmrSystem::add_block");
  const bool newton_non_default = amr_newton_options_non_default(newton);
  if (time != "imex" && newton_non_default)
    throw std::runtime_error("AmrSystem::add_block : Newton options require time='imex'");
  // newton_diagnostics (newton_report) requires time='imex' (the report comes from the IMEX source
  // Newton), parity with System::add_block. SUPPORT: native MULTI-BLOCK only -- the single-block
  // (coupler) and the .so loaders reject it (at build / at the facade), never an empty report.
  if (time != "imex" && newton_diagnostics)
    throw std::runtime_error("AmrSystem::add_block : newton_diagnostics requires time='imex'");
  // time == "ssprk3": SSPRK3 (order 3 + per-stage reflux), explicit transport -> MUTUALLY EXCLUSIVE
  // with imex (single time.kind selector, parity with System). The implicit stiff source (imex) does
  // NOT combine with SSPRK3 (unvalidated combination): the engine also rejects it as defense in depth.
  if (time != "explicit" && time != "imex" && time != "ssprk3") {
    if (time == "imexrk_ars222")
      throw std::runtime_error(
          "AmrSystem : time 'imexrk_ars222' (IMEX-RK family, ARS(2,2,2) scheme) not wired on AMR "
          "(scope = Cartesian System). Use one of [" + amr_time_routes_csv() +
          "] on AMR, or a "
          "Cartesian System for IMEX-RK.");
    throw std::runtime_error("AmrSystem : time '" + time +
                             "' unknown on AMR (valid: " + amr_time_routes_csv() + ")");
  }
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("AmrSystem : unknown recon '" + recon + "' (valid: " +
                             kReconRouteTokensCsv + ")");
  const bool imex = (time == "imex");
  const int time_method = (time == "ssprk3") ? 1 : 0;  // pops::AmrTimeMethod (0 kEuler, 1 kSsprk3)
  // The partial IMEX mask (implicit_vars / implicit_roles) only applies to the IMEX source step:
  // requesting it in explicit is an ERROR (no silent ignore; same guard as System::add_block).
  if (!imex && (!implicit_vars.empty() || !implicit_roles.empty()))
    throw std::runtime_error(
        "AmrSystem::add_block : implicit_vars / implicit_roles require time='imex' "
        "(the implicit mask only applies to the IMEX source step ; got time='" +
        time + "')");
  // MULTI-BLOCK (capstone v): a 2nd block (or more) switches to the AmrRuntime runtime engine
  // (shared hierarchy, co-located summed Poisson). The single block stays on AmrCouplerMP
  // (bit-identical). An already COMPILED block (set_compiled_block / add_compiled_model) CAN now mix
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
  b.imex = imex;
  b.time_method = time_method;  // SSPRK3 (1) or forward Euler (0); threaded at build (single/multi)
  b.implicit_vars =
      implicit_vars;  // partial IMEX mask (resolved into indices at build, build_multi)
  b.implicit_roles = implicit_roles;
  b.newton =
      newton;  // Newton options grouped into a POD (ADC-214; wave 3, single-block AND multi-block)
  b.newton_non_default = newton_non_default;
  b.newton_diagnostics =
      newton_diagnostics;          // newton_report (native multi-block; single/.so rejected)
  b.pos_floor = positivity_floor;  // Zhang-Shu floor (ADC-259); threaded at build (single/multi)
  b.substeps = substeps;
  b.stride = stride;
  b.gamma = model.gamma;  // adiabatic index of the block (Euler), read by coupler_write_coarse
  p_->blocks.push_back(std::move(b));
}

POPS_EXPORT void AmrSystem::set_compiled_block(
    int ncomp, double gamma, int substeps,
    std::function<AmrCompiledHooks(const AmrBuildParams&)> mono_builder,
    AmrCompiledBlockBuilder multi_builder, const std::string& name, bool recon_prim, bool imex,
    int stride, const std::vector<std::string>& implicit_vars,
    const std::vector<std::string>& implicit_roles, double pos_floor,
    std::shared_ptr<std::vector<double>> runtime_params) {
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
  // The partial IMEX mask only applies to the IMEX source step (same guard as add_block):
  // requesting it in explicit is an ERROR (no silent ignore).
  if (!imex && (!implicit_vars.empty() || !implicit_roles.empty()))
    throw std::runtime_error(
        "AmrSystem::set_compiled_block : implicit_vars / implicit_roles require "
        "time='imex' (the implicit mask only applies to the IMEX source step)");
  // MULTI-BLOCK (capstone v): a 2nd block (or more) switches to AmrRuntime; the compiled block is
  // materialized there by multi_builder (its AmrRuntimeBlock on the shared layout). So we STACK the
  if (name.empty())
    throw std::runtime_error(
        "AmrSystem::set_compiled_block requires an owner-qualified non-empty name");
  if (p_->block_index(name) >= 0)
    throw std::runtime_error("AmrSystem::set_compiled_block : block name already used '" + name + "'");
  Impl::BlockSpec b;
  b.name = name;
  b.is_compiled = true;
  b.gamma = gamma;
  b.substeps = substeps;
  b.stride = stride;
  b.recon_prim = recon_prim;
  b.imex = imex;
  b.implicit_vars =
      implicit_vars;  // partial IMEX mask (resolved into indices by multi_builder at build)
  b.implicit_roles = implicit_roles;
  // Zhang-Shu positivity floor (ADC-322): carried by the regenerated .so loader (pops_install_native_amr
  // -> add_compiled_model). Stored on the block so the MONO path reads it via make_build_params ->
  // AmrBuildParams::pos_floor, and the MULTI path forwards it through the AmrCompiledBlockBuilder.
  b.pos_floor = pos_floor;
  // NATIVE per-block runtime params (ADC-514): the SHARED vector add_compiled_model seeded from the
  // model defaults (empty -> no injection); make_build_params (mono) reads it and the builder (multi) threads it.
  b.runtime_params = std::move(runtime_params);
  b.compiled_hooks_builder = std::move(mono_builder);   // single-block path (AmrCouplerMP)
  b.compiled_block_builder = std::move(multi_builder);  // multi-block path (AmrRuntime)
  p_->blocks.push_back(std::move(b));
}

// NAMED multi-elliptic field declaration (ADC-428): the native AMR loader calls this once per
// m.elliptic_field after the block is installed. We stash the field's aux output components and FORCE
// the runtime engine (the named-field solve lives only in AmrRuntime); build_multi replays it. POPS_EXPORT:
// resolved by the generated AMR .so loader across the dlopen boundary (same as set_compiled_block).
POPS_EXPORT void AmrSystem::register_elliptic_field(const std::string& block_name,
                                                    const std::string& provider_key,
                                                    int phi_comp, int gx_comp, int gy_comp) {
  require_assembling_amr(p_->bound_,
                         "register_elliptic_field");  // frozen once pops.bind completes (ADC-592)
  if (p_->built)
    throw std::runtime_error("AmrSystem::register_elliptic_field : the system is already built");
  p_->ell_field_comps_[{block_name, provider_key}] = {phi_comp, gx_comp, gy_comp};
  p_->force_runtime_ =
      true;  // the named-field solve needs the AmrRuntime engine (even single-block)
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

// Solved potential of a named elliptic field on the coarse level, n*n row-major (ADC-428 read-back).
// Builds the hierarchy on first call (ensure_built) then reads from the AmrRuntime engine, which solves
// the fields (counterpart of potential()). Only the runtime path carries named fields, so the single-block
// AmrCouplerMP coupler (no named field registered) rejects with a clear message.
std::vector<double> AmrSystem::named_field_values(const std::string& field) {
  p_->ensure_built();
  if (!p_->runtime)
    throw std::runtime_error(
        "AmrSystem::named_field_values : named elliptic field '" + field +
        "' is only solved by the multi-block runtime engine (AmrRuntime). Declare it via "
        "m.elliptic_field on the block model so the loader registers it.");
  return p_->runtime->named_field_values(field);
}

std::vector<std::string> AmrSystem::field_provider_slots() const {
  const_cast<Impl*>(p_.get())->ensure_built();
  if (!p_->runtime)
    return {};
  return p_->runtime->provider_slots();
}

void AmrSystem::set_field_potential(const std::string& provider_slot,
                                    const std::vector<double>& phi) {
  set_field_potential_level(provider_slot, 0, phi);
}

int AmrSystem::field_provider_levels(const std::string& provider_slot) {
  p_->ensure_built();
  if (!p_->runtime)
    throw std::runtime_error("AmrSystem::field_provider_levels requires the qualified field runtime");
  return p_->runtime->provider_potential_levels(provider_slot);
}

void AmrSystem::set_field_potential_level(const std::string& provider_slot, int level,
                                          const std::vector<double>& phi) {
  p_->ensure_built();
  if (!p_->runtime)
    throw std::runtime_error("AmrSystem::set_field_potential_level requires the qualified field runtime");
  if (level < 0 || level >= p_->runtime->provider_potential_levels(provider_slot))
    throw std::out_of_range("AmrSystem::set_field_potential_level level out of range");
  const int width = p_->cfg.n << level;
  const std::size_t expected = static_cast<std::size_t>(width) * width;
  if (phi.size() != expected)
    throw std::runtime_error("AmrSystem::set_field_potential_level layout size mismatch");
  MultiFab& target = p_->runtime->provider_potential_level(provider_slot, level);
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
  MultiFab& source = p_->runtime->provider_potential_level(provider_slot, level);
  std::vector<double> result(static_cast<std::size_t>(width) * width, 0.0);
  for (int li = 0; li < source.local_size(); ++li) {
    const ConstArray4 values = source.fab(li).const_array();
    const Box2D valid = source.box(li);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        result[static_cast<std::size_t>(j) * width + static_cast<std::size_t>(i)] =
            static_cast<double>(values(i, j, 0));
  }
  if (p_->cfg.distribute_coarse)
    all_reduce_sum_inplace(result.data(), static_cast<int>(result.size()));
  return result;
}

namespace {
// Module anchor for dladdr: its ADDRESS lives in the image that contains amr_system.cpp (the _pops
// module, or the test binary). add_native_block uses it to locate the module and promote it to
// global scope (RTLD_NOLOAD). A TU-local function suffices (no need to export) and avoids
// depending on a symbol defined elsewhere (pops::abi_key, system.cpp).
void amr_native_anchor() {}
}  // namespace

void AmrSystem::add_native_block(const std::string& name, const std::string& so_path,
                                 const std::string& limiter, const std::string& riemann,
                                 const std::string& recon, const std::string& time, double gamma,
                                 int substeps, double positivity_floor) {
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
  // UPSTREAM scheme validation (like add_block): add_compiled_model(AmrSystem&) already rejects
  // time outside {explicit, imex} and recon outside {conservative, primitive}, but we diagnose HERE a
  // typo before the C++ boundary. time == "imex" => stiff source handled IMPLICITLY
  // (backward_euler_source), explicit transport carried by the reflux. limiter (including weno5, wired #105)
  // and riemann (including hllc/roe, wired at parity #113) are validated by dispatch_amr_compiled in the
  // loader (clear exception).
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error(
        "AmrSystem::add_native_block : recon 'conservative' | 'primitive' "
        "(got '" +
        recon + "')");
  // time == "ssprk3": SSPRK3 IS NOT transported by the AMR .so loader flat ABI -- the extern "C"
  // installer (pops_install_native_amr) only marshals the time STRING, and the add_compiled_model
  // template it inlines only maps {explicit, imex} (it does NOT freeze AmrBuildParams::time_method). Rather
  // than SILENTLY FALLING BACK to kEuler (option ignored), we REJECT explicitly here: an SSPRK3 block
  // must go through a NATIVE block pops.Model(...) (AmrSystem.add_block), not a compiled .so loader.
  if (time == "ssprk3")
    throw std::runtime_error(
        "AmrSystem::add_native_block : time='ssprk3' not transported by the compiled .so loader "
        "(flat "
        "ABI : only {explicit, imex} is marshaled) ; use a native block pops.Model(...) with "
        "pops.Explicit(ssprk3=True) via AmrSystem.add_block.");
  if (time != "explicit" && time != "imex")
    throw std::runtime_error(
        "AmrSystem::add_native_block : time 'explicit' | 'imex' on AMR (got '" + time + "')");
  // DSL "production" path on the AMR side: the generated .so loader (emit_cpp_native_loader with
  // target="amr_system") inlines the header template add_compiled_model(AmrSystem&, ...), which
  // materializes a concrete AmrCouplerMP<Model> at lazy build and installs its hooks via
  // set_compiled_block -- NATIVE path, SAME AMR hierarchy as add_block (reflux, regrid). The loader
  // thus calls set_compiled_block (out-of-line method of pops::AmrSystem) DEFINED in THIS module;
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
    using install_fn_t = void (*)(void*, const char*, const char*, const char*, const char*,
                                  const char*, double, int, double);
    auto install = reinterpret_cast<install_fn_t>(pops::dynlib::sym(h, "pops_install_native_amr"));
    if (!install) {
      pops::dynlib::close(h);
      throw std::runtime_error(
          "AmrSystem::add_native_block : pops_install_native_amr missing from the .dll");
    }
    install(static_cast<void*>(this), name.c_str(), limiter.c_str(), riemann.c_str(), recon.c_str(),
            time.c_str(), gamma, substeps, positivity_floor);
  }
#else
  {
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&amr_native_anchor), &info) && info.dli_fname)
      dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
  }
  // RTLD_GLOBAL: places the loader's symbols in the global scope AND lets the loader resolve
  // its undefined symbols (set_compiled_block exported POPS_EXPORT) against the already-loaded images. RTLD_NOW:
  // immediate resolution -> a missing AmrSystem symbol fails HERE, not in flight.
  void* h = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!h) {
    const char* e = dlerror();
    throw std::runtime_error(
        "AmrSystem::add_native_block : dlopen('" + so_path + "') : " + std::string(e ? e : "?") +
        " (the symbol pops::AmrSystem::set_compiled_block must be exported AND the "
        "_pops module loaded globally ; cf. POPS_EXPORT)");
  }
  // EXPLICIT ABI GUARD: the key baked into the loader (at ITS compilation) must equal the module's
  // key. A mismatch = divergent headers / compiler / standard -> potentially different memory layout
  // of AmrSystem/AmrBuildParams/AmrCompiledHooks at the boundary -> UB. We raise
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
  // AMR native installer of the loader: reinterpret_cast<AmrSystem*>(this) then
  // add_compiled_model<ProdModel>(*amrsys, ...). Scheme marshaled as flat extern "C" arguments. No
  // evolve parameter (single-block AMR, no frozen background block like System). DISTINCT SYMBOL
  // (pops_install_native_amr, vs pops_install_native on the System side): a loader generated for System
  // does NOT export it, so wiring it here fails clearly instead of an inconsistent cast. The trailing
  // double is the Zhang-Shu positivity floor (ADC-322): old 8-argument loaders carry an ABI key from
  // the pre-floor headers and are REJECTED above, so the 9-argument call never reaches a stale .so.
  using install_fn_t = void (*)(void*, const char*, const char*, const char*, const char*,
                                const char*, double, int, double);
  auto install = reinterpret_cast<install_fn_t>(dlsym(h, "pops_install_native_amr"));
  if (!install) {
    dlclose(h);
    throw std::runtime_error(
        "AmrSystem::add_native_block : pops_install_native_amr missing from the .so "
        "(loader generated for System, or regenerate via "
        "dsl.compile_native(target='amr_system'))");
  }
  // DEFENSE IN DEPTH (ADC-514): reject a bogus pops_compiled_nparams before add_compiled_model seeds
  // the value block (closes h on overflow; 0 for an older loader without the symbol, bit-identical).
  (void)detail::reject_excessive_amr_runtime_params(h);
  install(static_cast<void*>(this), name.c_str(), limiter.c_str(), riemann.c_str(), recon.c_str(),
          time.c_str(), gamma, substeps, positivity_floor);
  // The .so stays loaded (RTLD_GLOBAL) for the duration of the process: the type-erasing builder
  // installed by set_compiled_block captures code (header template) that lives there. We do NOT close it.
#endif  // _WIN32 (production AMR POSIX-only; Windows = throw, ADC-100)
  const int installed_idx = p_->block_index(name);
  if (installed_idx >= 0) {
    Impl::BlockSpec& b = p_->blocks[static_cast<std::size_t>(installed_idx)];
    b.limiter = limiter;
    b.riemann = riemann;
    b.recon_prim = (recon == "primitive");
    b.imex = (time == "imex");
    b.gamma = gamma;
    b.substeps = substeps;
    b.pos_floor = positivity_floor;
  }
}

void AmrSystem::set_refinement(double threshold, const std::string& variable,
                               const std::string& role) {
  require_assembling_amr(p_->bound_,
                         "set_refinement");  // frozen once pops.bind completes (ADC-592)
  // Reject the ambiguous double selector immediately (fast feedback); cons_vars is only known at the
  // lazy build, so an absent name/role is caught there (build_multi -> resolve_selected_component).
  if (!variable.empty() && !role.empty())
    throw std::runtime_error(
        "AmrSystem::set_refinement : select the refinement variable by NAME (variable=) or by ROLE "
        "(role=), not both");
  if (!std::isfinite(threshold))
    throw std::runtime_error("AmrSystem::set_refinement : threshold must be finite");
  p_->refine_threshold = threshold;
  p_->refine_var_name = variable;
  p_->refine_var_role = role;
}

void AmrSystem::set_bootstrap_refinement(
    const std::string& block, const std::string& variable, double threshold,
    const std::string& provider_identity) {
  require_assembling_amr(p_->bound_, "set_bootstrap_refinement");
  if (p_->built || block.empty() || variable.empty() || provider_identity.empty() ||
      !std::isfinite(threshold) || p_->bootstrap_tag_spec)
    throw std::runtime_error(
        "AmrSystem::set_bootstrap_refinement requires one exact pre-build provider manifest");
  p_->bootstrap_tag_spec = std::make_unique<Impl::BootstrapTagSpec>(
      Impl::BootstrapTagSpec{block, variable, threshold, provider_identity});
  p_->refine_threshold = threshold;
}

void AmrSystem::set_bootstrap_tagging(
    const std::vector<std::string>& leaf_blocks,
    const std::vector<std::string>& leaf_variables,
    const std::vector<int>& leaf_ops,
    const std::vector<double>& leaf_thresholds,
    const std::vector<int>& refine_ops, const std::vector<int>& refine_args,
    const std::vector<int>& coarsen_ops, const std::vector<int>& coarsen_args,
    int min_cycles, const std::string& equality_policy,
    const std::string& conflict_policy, const std::string& provider_identity) {
  require_assembling_amr(p_->bound_, "set_bootstrap_tagging");
  const std::size_t leaf_count = leaf_blocks.size();
  if (p_->built || p_->tagging_spec || p_->bootstrap_tag_spec || leaf_count == 0 ||
      leaf_variables.size() != leaf_count || leaf_ops.size() != leaf_count ||
      leaf_thresholds.size() != leaf_count || refine_ops.empty() ||
      refine_ops.size() != refine_args.size() ||
      coarsen_ops.size() != coarsen_args.size() || min_cycles < 0 ||
      provider_identity.empty())
    throw std::runtime_error(
        "AmrSystem::set_bootstrap_tagging requires one exact pre-build graph manifest");
  if (std::any_of(leaf_blocks.begin(), leaf_blocks.end(),
                  [](const std::string& value) { return value.empty(); }) ||
      std::any_of(leaf_variables.begin(), leaf_variables.end(),
                  [](const std::string& value) { return value.empty(); }) ||
      std::any_of(leaf_thresholds.begin(), leaf_thresholds.end(),
                  [](double value) { return !std::isfinite(value); }))
    throw std::runtime_error("AmrSystem::set_bootstrap_tagging has an invalid leaf");
  const auto equality = equality_policy == "hold"      ? 0
                        : equality_policy == "refine"  ? 1
                        : equality_policy == "coarsen" ? 2
                                                       : -1;
  const auto conflict = conflict_policy == "error"          ? 0
                        : conflict_policy == "hold"          ? 1
                        : conflict_policy == "refine_wins"   ? 2
                        : conflict_policy == "coarsen_wins"  ? 3
                                                             : -1;
  if (equality < 0 || conflict < 0)
    throw std::runtime_error("AmrSystem::set_bootstrap_tagging has an unknown policy");
  p_->tagging_spec = std::make_unique<Impl::TaggingSpec>(Impl::TaggingSpec{
      leaf_blocks, leaf_variables, leaf_ops, leaf_thresholds,
      refine_ops, refine_args, coarsen_ops, coarsen_args,
      min_cycles, equality, conflict, provider_identity});
}

void AmrSystem::set_phi_refinement(double grad_threshold) {
  require_assembling_amr(p_->bound_,
                         "set_phi_refinement");  // frozen once pops.bind completes (ADC-592)
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::set_phi_refinement : the system is already built (set the "
        "refinement criterion before any step/mass/density)");
  // <= 0 (default) -> phi DISABLED (build_multi does not set the phi predicate); bit-identical. > 0 ->
  // phi tag on |grad phi| added to the tag union (D4), set by build_multi. We DO NOT impose a
  // guard on the number of blocks here: the single/multi routing is only decided at ensure_built (>= 2
  // add_block), and the order of configuration calls is free (set_phi_refinement may precede the
  // 2nd add_block). In SINGLE-BLOCK the threshold stays without effect: the AmrCouplerMP path regrids on
  // density alone (no separate phi predicate) and does not call build_multi -> phi_grad_threshold is ignored,
  // without illusion (the phi predicate only makes sense on the multi-block runtime engine).
  if (!std::isfinite(grad_threshold))
    throw std::runtime_error("AmrSystem::set_phi_refinement : grad_threshold must be finite");
  p_->phi_grad_threshold = grad_threshold;
}

void AmrSystem::set_poisson(const std::string& rhs, const std::string& solver,
                            const std::string& bc, const std::string& wall, double wall_radius,
                            bool composite, int fac_max_iters, int fac_fine_sweeps, double fac_tol,
                            double fac_coarse_rel_tol, int fac_coarse_cycles, bool fac_verbose) {
  require_assembling_amr(p_->bound_, "set_poisson");  // frozen once pops.bind completes (ADC-592)
  // single-block/explicit CONTRACT (cf. set_compiled_block): AMR wires a SINGLE elliptic
  // solver (GeometricMG, the AmrCouplerMP template default) and a SINGLE right-hand side
  // (f = model.elliptic_rhs(U), assembled by coupler_eval_rhs). We thus explicitly REFUSE
  // any rhs/solver value outside the actually wired domain, instead of storing it
  // silently (the historical no-op suggested solver='fft' worked on the hierarchy).
  // bc/wall are actually consumed by poisson_bc()/wall_active(): validated there.
  if (rhs != "charge_density" && rhs != "composite")
    throw std::runtime_error("AmrSystem::set_poisson : unknown rhs '" + rhs +
                             "' (valid: " + kPoissonRhsRouteTokensCsv +
                             "; the right-hand side = sum of the "
                             "block's elliptic bricks)");
  if (solver != "geometric_mg")
    throw std::runtime_error("AmrSystem::set_poisson : solver '" + solver +
                             "' unsupported on AMR (only 'geometric_mg' is wired on the "
                             "hierarchy ; 'fft' only exists on a single-level grid, cf. System)");
  // ADC-645: composite-FAC knobs (inert when composite is false): a knob is either the <= 0/0
  // default sentinel or in its domain, never silently clamped.
  if (fac_max_iters < 0 || fac_fine_sweeps < 0 || fac_coarse_cycles < 0)
    throw std::runtime_error(
        "AmrSystem::set_poisson : fac_max_iters/fac_fine_sweeps/fac_coarse_cycles >= 0 required "
        "(0 = default)");
  if (fac_tol < 0.0 || fac_tol >= 1.0 || fac_coarse_rel_tol < 0.0 || fac_coarse_rel_tol >= 1.0)
    throw std::runtime_error(
        "AmrSystem::set_poisson : fac_tol/fac_coarse_rel_tol in [0, 1) required (0 = default)");
  p_->p_rhs = rhs;
  p_->p_solver = solver;
  p_->p_bc = bc;
  p_->p_wall = wall;
  p_->p_wall_radius = wall_radius;
  p_->p_composite = composite;
  p_->p_fac_max_iters = fac_max_iters;
  p_->p_fac_fine_sweeps = fac_fine_sweeps;
  p_->p_fac_tol = fac_tol;
  p_->p_fac_coarse_rel_tol = fac_coarse_rel_tol;
  p_->p_fac_coarse_cycles = fac_coarse_cycles;
  p_->p_fac_verbose = fac_verbose;
}

void AmrSystem::set_field_solver_plan(
    const std::string& provider_slot, const std::string& provider_identity,
    const std::string& output_owner_identity,
    const std::string& output_block, const std::string& output_key,
    const std::vector<std::string>& provider_identities,
    const std::vector<std::string>& provider_blocks,
    const std::vector<std::string>& provider_keys,
    const std::vector<double>& provider_coefficients, const std::string& solver,
    const std::string& hierarchy, double abs_tol, double rel_tol, int max_cycles,
    int min_coarse, int pre_smooth, int post_smooth, int bottom_sweeps,
    int coarse_threshold) {
  require_assembling_amr(p_->bound_, "set_field_solver_plan");
  if (p_->built)
    throw std::runtime_error("AmrSystem::set_field_solver_plan: system already built");
  if (provider_slot.empty() || provider_identity.empty() || output_owner_identity.empty() ||
      output_block.empty() ||
      output_key.empty())
    throw std::runtime_error(
        "AmrSystem::set_field_solver_plan requires a qualified provider identity");
  const std::size_t provider_count = provider_identities.size();
  if (provider_count == 0 || provider_blocks.size() != provider_count ||
      provider_keys.size() != provider_count || provider_coefficients.size() != provider_count)
    throw std::runtime_error("AmrSystem::set_field_solver_plan invalid provider-pack shape");
  for (std::size_t i = 0; i < provider_count; ++i)
    if (provider_identities[i].empty() || provider_blocks[i].empty() || provider_keys[i].empty() ||
        !std::isfinite(provider_coefficients[i]))
      throw std::runtime_error("AmrSystem::set_field_solver_plan invalid provider-pack entry");
  if (solver != "geometric_mg")
    throw std::runtime_error(
        "AmrSystem::set_field_solver_plan requires geometric_mg on AMR");
  if (hierarchy != "composite" && hierarchy != "level_local")
    throw std::runtime_error("AmrSystem::set_field_solver_plan invalid hierarchy policy");
  if (abs_tol < 0.0 || rel_tol <= 0.0 || max_cycles < 1 || min_coarse < 1 ||
      pre_smooth < 0 || post_smooth < 0 || bottom_sweeps < 0 || coarse_threshold < 0)
    throw std::runtime_error("AmrSystem::set_field_solver_plan invalid multigrid options");
  const auto existing = p_->field_plans_.find(provider_slot);
  if (existing != p_->field_plans_.end() &&
      existing->second.provider_identity != provider_identity)
    throw std::runtime_error("AmrSystem::set_field_solver_plan provider digest collision");
  auto& plan = p_->field_plans_[provider_slot];
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
  plan.hierarchy = hierarchy;
  plan.mg_opts.abs_tol = static_cast<Real>(abs_tol);
  plan.mg_opts.rel_tol = static_cast<Real>(rel_tol);
  plan.mg_opts.max_cycles = max_cycles;
  plan.mg_opts.min_coarse = min_coarse;
  plan.mg_opts.nu1 = pre_smooth;
  plan.mg_opts.nu2 = post_smooth;
  plan.mg_opts.nbottom = bottom_sweeps;
  plan.mg_opts.coarse_threshold = coarse_threshold;
  p_->force_runtime_ = true;
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
  bc.dy = bc.dx;
  BCType* types[] = {&bc.xlo, &bc.xhi, &bc.ylo, &bc.yhi};
  Real* vals[] = {&bc.xlo_val, &bc.xhi_val, &bc.ylo_val, &bc.yhi_val};
  Real* alphas[] = {&bc.xlo_alpha, &bc.xhi_alpha, &bc.ylo_alpha, &bc.yhi_alpha};
  Real* betas[] = {&bc.xlo_beta, &bc.xhi_beta, &bc.ylo_beta, &bc.yhi_beta};
  for (int face = 0; face < 4; ++face) {
    const Real a = static_cast<Real>(alpha[face]);
    const Real b = static_cast<Real>(beta[face]);
    const Real v = static_cast<Real>(value[face]);
    if (!std::isfinite(alpha[face]) || !std::isfinite(beta[face]) ||
        !std::isfinite(value[face]) ||
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
}

void AmrSystem::set_field_boundary_dependencies(
    const std::string& provider_slot, const std::vector<std::string>& state_blocks,
    const std::vector<int>& state_components,
    const std::vector<std::string>& field_blocks,
    const std::vector<std::string>& field_keys,
    const std::vector<int>& field_components) {
  require_assembling_amr(p_->bound_, "set_field_boundary_dependencies");
  if (state_blocks.size() != state_components.size() || !field_blocks.empty() ||
      !field_keys.empty() || !field_components.empty())
    throw std::runtime_error(
        "AmrSystem::set_field_boundary_dependencies accepts exact state buffers only");
  if (std::any_of(state_blocks.begin(), state_blocks.end(), [](const auto& value) {
        return value.empty();
      }) ||
      std::any_of(state_components.begin(), state_components.end(), [](int value) {
        return value < 0;
      }))
    throw std::runtime_error(
        "AmrSystem::set_field_boundary_dependencies contains invalid entries");
  auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error(
        "AmrSystem::set_field_boundary_dependencies unknown provider slot");
  found->second.boundary_state_blocks = state_blocks;
  found->second.boundary_state_components = state_components;
  if (p_->runtime)
    p_->runtime->set_field_boundary_dependencies(
        provider_slot, state_blocks, state_components);
}

void AmrSystem::set_field_logical_timepoint(const std::string& provider_slot,
                                            const FieldLogicalTimePoint& point) {
  auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("AmrSystem::set_field_logical_timepoint unknown provider slot");
  found->second.boundary_context.point = point;
  if (!found->second.has_boundary_kernel ||
      !found->second.boundary_kernel.observes_iteration)
    found->second.boundary_context.point.iteration = 0;
  if (p_->runtime)
    p_->runtime->set_field_logical_timepoint(provider_slot,
                                             found->second.boundary_context.point);
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
}

void AmrSystem::set_field_newton_plan(const std::string& provider_slot, double tolerance,
                                      int max_iterations, double linear_tolerance,
                                      int linear_max_iterations, int restart, double armijo,
                                      double minimum_step) {
  require_assembling_amr(p_->bound_, "set_field_newton_plan");
  auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("AmrSystem::set_field_newton_plan unknown field provider slot");
  FieldNewtonOptions options{static_cast<Real>(tolerance), max_iterations,
                             static_cast<Real>(linear_tolerance), linear_max_iterations,
                             restart, static_cast<Real>(armijo),
                             static_cast<Real>(minimum_step)};
  validate_field_newton_options(options);
  found->second.newton = options;
  found->second.has_newton = true;
  if (p_->runtime)
    p_->runtime->set_field_newton_plan(provider_slot, options);
}

void AmrSystem::set_field_nullspace(const std::string& provider_slot, bool constant_kernel,
                                    bool mean_zero_gauge) {
  require_assembling_amr(p_->bound_, "set_field_nullspace");
  auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("AmrSystem::set_field_nullspace unknown provider slot");
  if (constant_kernel != mean_zero_gauge)
    throw std::runtime_error(
        "AmrSystem::set_field_nullspace constant kernel requires explicit mean-zero gauge");
  found->second.nullspace_assertion = constant_kernel ? "constant" : "none";
  found->second.gauge = mean_zero_gauge ? "mean_zero" : "none";
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
  // UPSTREAM size guard: NON empty state and multiple of n*n. The exact size ncomp*n*n is checked
  // at build (coupler_write_coarse_state), the only place where ncomp == Model::n_vars is known -- same
  // deferral as the n*n guard of set_density. We explicitly reject an EMPTY state (0 % nn == 0 would
  // otherwise set has_state=true with an empty state, which would only throw deep in the 1st step).
  const std::size_t nn = static_cast<std::size_t>(p_->cfg.n) * static_cast<std::size_t>(p_->cfg.n);
  if (U.empty())
    throw std::runtime_error(
        "AmrSystem::set_conservative_state : empty state (expected ncomp*n*n)");
  if (U.size() % nn != 0)
    throw std::runtime_error("AmrSystem::set_conservative_state : state size (" +
                             std::to_string(U.size()) + ") not a multiple of n*n (" +
                             std::to_string(nn) + ") ; expected ncomp*n*n component-major");
  const int resolved = p_->block_index(name);
  if (resolved < 0)
    throw std::runtime_error("AmrSystem::set_conservative_state : no block named '" + name + "'");
  const std::size_t idx = static_cast<std::size_t>(resolved);
  p_->blocks[idx].state = U;
  p_->blocks[idx].has_state = true;
}

void AmrSystem::bootstrap_next_level(int refinement_ratio) {
  require_assembling_amr(p_->bound_, "bootstrap_next_level");
  if (!p_->cfg.explicit_bootstrap)
    throw std::runtime_error(
        "AmrSystem::bootstrap_next_level requires explicit_bootstrap=true");
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
    throw std::runtime_error(
        "AmrSystem::bootstrap_next_level would exceed resolved level_count");
  for (const auto& [_, route] : p_->bootstrap_subject_routes) {
    const auto& descriptor = p_->bootstrap_transfer_routes.at(route).descriptor;
    if ((descriptor.operation == "prolongation" || descriptor.operation == "restriction" ||
         descriptor.operation == "coarse_fine_fill" ||
         descriptor.operation == "temporal_interpolation") &&
        descriptor.refinement_ratio != refinement_ratio)
      throw std::runtime_error(
          "AmrSystem::bootstrap_next_level transition/transfer ratio mismatch");
  }
  p_->runtime->bootstrap_next_level(refinement_ratio);
}

void AmrSystem::begin_bootstrap_plan() {
  require_assembling_amr(p_->bound_, "begin_bootstrap_plan");
  if (!p_->cfg.explicit_bootstrap)
    throw std::runtime_error(
        "AmrSystem::begin_bootstrap_plan requires explicit_bootstrap=true");
  p_->ensure_built();
  if (!p_->runtime)
    throw std::runtime_error(
        "AmrSystem::begin_bootstrap_plan requires the shared N-level runtime engine");
  p_->runtime->begin_bootstrap_plan();
}

void AmrSystem::register_bootstrap_transfer_route(
    const std::string& identity, const std::vector<std::string>& subjects,
    const std::string& provider_identity, const std::string& space,
    const std::string& centering, const std::string& representation,
    const std::string& storage, const std::string& operation,
    const std::string& kernel, int order, const std::vector<int>& ghost_depth,
    int dimension, int refinement_ratio) {
  require_assembling_amr(p_->bound_, "register_bootstrap_transfer_route");
  if (p_->built || subjects.empty())
    throw std::runtime_error(
        "AmrSystem::register_bootstrap_transfer_route requires pre-build subjects");
  const std::string kernel_namespace =
      (space == "field" || space == "cache") ? "pops.lib.amr.materializer::"
                                               : "pops.lib.amr.transfer::";
  runtime::amr::TransferRoute route{
      provider_identity,
      kernel_namespace + kernel,
      runtime::amr::TransferRouteDescriptor{
          space, centering, representation, storage, operation,
          order, ghost_depth, dimension, refinement_ratio}};
  p_->bootstrap_transfer_routes.add(identity, std::move(route));
  for (const std::string& subject : subjects) {
    const auto key = std::make_pair(subject, operation);
    if (subject.empty() || p_->bootstrap_subject_routes.count(key) != 0)
      throw std::runtime_error(
          "AmrSystem::register_bootstrap_transfer_route requires unique subjects");
    p_->bootstrap_subject_routes.emplace(key, identity);
  }
}

void AmrSystem::commit_bootstrap_level() {
  if (!p_->runtime)
    throw std::runtime_error("AmrSystem::commit_bootstrap_level has no runtime engine");
  p_->runtime->commit_bootstrap_level();
}

void AmrSystem::rollback_bootstrap_level() {
  if (!p_->runtime)
    throw std::runtime_error("AmrSystem::rollback_bootstrap_level has no runtime engine");
  p_->runtime->rollback_bootstrap_level();
}

void AmrSystem::register_bootstrap_array(const std::string& subject,
                                         const std::string& centering, int ncomp,
                                         int ny, int nx,
                                         const std::vector<double>& values) {
  require_assembling_amr(p_->bound_, "register_bootstrap_array");
  if (p_->built)
    throw std::runtime_error(
        "AmrSystem::register_bootstrap_array requires registration before hierarchy build");
  if (subject.empty() || p_->bootstrap_arrays.count(subject) != 0)
    throw std::runtime_error(
        "AmrSystem::register_bootstrap_array requires a unique non-empty subject");
  const auto route_key = std::make_pair(
      subject, std::string("prolongation"));
  const auto route_found = p_->bootstrap_subject_routes.find(route_key);
  if (route_found == p_->bootstrap_subject_routes.end())
    throw std::runtime_error(
        "AmrSystem::register_bootstrap_array requires an exact registered prolongation route");
  const auto& route = p_->bootstrap_transfer_routes.at(route_found->second);
  if (route.descriptor.centering != centering ||
      (route.descriptor.space != "face" && route.descriptor.space != "node"))
    throw std::runtime_error(
        "AmrSystem::register_bootstrap_array centering differs from its transfer route");
  const int n = p_->cfg.n;
  const bool shape_ok =
      (centering == "node" && nx == n + 1 && ny == n + 1) ||
      (centering == "face_x" && nx == n + 1 && ny == n) ||
      (centering == "face_y" && nx == n && ny == n + 1);
  if (!shape_ok || ncomp < 1 ||
      values.size() != static_cast<std::size_t>(ncomp) * nx * ny)
    throw std::runtime_error(
        "AmrSystem::register_bootstrap_array received an incompatible centered array shape");
  p_->bootstrap_arrays.emplace(
      subject, Impl::BootstrapArray{centering, ncomp, values});
}

void AmrSystem::bind_bootstrap_block_subject(const std::string& subject,
                                             const std::string& block) {
  require_assembling_amr(p_->bound_, "bind_bootstrap_block_subject");
  const auto route_key = std::make_pair(
      subject, std::string("prolongation"));
  const auto route_found = p_->bootstrap_subject_routes.find(route_key);
  if (route_found == p_->bootstrap_subject_routes.end() || subject.empty() || block.empty() ||
      p_->bootstrap_block_subjects.count(subject) != 0)
    throw std::runtime_error(
        "AmrSystem::bind_bootstrap_block_subject requires an exact unique cell route");
  const auto& route = p_->bootstrap_transfer_routes.at(route_found->second);
  if (route.descriptor.space != "cell" || route.descriptor.centering != "cell")
    throw std::runtime_error(
        "AmrSystem::bind_bootstrap_block_subject route is not cell-centered");
  p_->bootstrap_block_subjects.emplace(subject, block);
}

void AmrSystem::register_bootstrap_face_vector(
    const std::vector<std::string>& subjects) {
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
    const int axis = route.descriptor.centering == "face_x" ? 0 :
                     route.descriptor.centering == "face_y" ? 1 : -1;
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

void AmrSystem::register_analytic_constant(const std::string& subject,
                                           const std::string& block,
                                           const std::string& space,
                                           const std::string& centering,
                                           const std::vector<double>& components) {
  require_assembling_amr(p_->bound_, "register_analytic_constant");
  if (p_->built || subject.empty() || components.empty() ||
      p_->bootstrap_analytic_constants.count(subject) != 0)
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

void AmrSystem::register_analytic_gaussian(
    const std::string& subject, const std::string& block, double center_x,
    double center_y, double background, double amplitude, double inverse_width) {
  require_assembling_amr(p_->bound_, "register_analytic_gaussian");
  const bool finite = std::isfinite(center_x) && std::isfinite(center_y) &&
                      std::isfinite(background) && std::isfinite(amplitude) &&
                      std::isfinite(inverse_width);
  if (p_->built || subject.empty() || block.empty() || !finite || inverse_width <= 0.0 ||
      p_->bootstrap_analytic_gaussians.count(subject) != 0 ||
      p_->bootstrap_analytic_constants.count(subject) != 0)
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
      subject, Impl::BootstrapGaussian{
          center_x, center_y, background, amplitude, inverse_width});
}

std::int64_t AmrSystem::bootstrap_analytic_reproject(const std::string& subject, int level) {
  if (!p_->runtime || level < 0 || level >= p_->runtime->nlev())
    throw std::runtime_error(
        "AmrSystem::bootstrap_analytic_reproject requires a pending hierarchy level");
  const auto constants = p_->bootstrap_analytic_constants.find(subject);
  const auto gaussian = p_->bootstrap_analytic_gaussians.find(subject);
  const auto route_found = p_->bootstrap_subject_routes.find(
      {subject, "prolongation"});
  if ((constants == p_->bootstrap_analytic_constants.end()) ==
          (gaussian == p_->bootstrap_analytic_gaussians.end()) ||
      route_found == p_->bootstrap_subject_routes.end())
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
    return p_->runtime->fill_bootstrap_block_constant(index, level, constants->second);
  }
  if (p_->bootstrap_arrays.count(subject) == 0)
    throw std::runtime_error("analytic centered source has no exact parent payload");
  return p_->runtime->fill_bootstrap_staggered_constant(
      subject, level, constants->second);
}

int AmrSystem::apply_bootstrap_component_floor(const std::string& subject, int level,
                                               int component, double floor) {
  if (!p_->runtime)
    throw std::runtime_error(
        "AmrSystem::apply_bootstrap_component_floor requires the native runtime");
  const auto block = p_->bootstrap_block_subjects.find(subject);
  if (block == p_->bootstrap_block_subjects.end())
    throw std::runtime_error("bootstrap component floor subject is not a cell state");
  const int index = p_->block_index(block->second);
  if (index < 0)
    throw std::runtime_error("bootstrap component floor block is unknown");
  return p_->runtime->apply_bootstrap_component_floor(
      static_cast<std::size_t>(index), level, component, static_cast<Real>(floor));
}

std::int64_t AmrSystem::recompute_bootstrap_field(
    const std::string& subject, const std::string& field_name) {
  if (!p_->runtime)
    throw std::runtime_error("bootstrap field recompute has no runtime engine");
  const auto route_id =
      p_->bootstrap_subject_routes.find({subject, "coarse_fine_fill"});
  if (route_id == p_->bootstrap_subject_routes.end())
    throw std::runtime_error("bootstrap field has no prepared elliptic materializer");
  const auto& route = p_->bootstrap_transfer_routes.at(route_id->second);
  if (route.descriptor.space != "field" || !route.executable.materialize)
    throw std::runtime_error("bootstrap field route is not an executable materializer");
  return route.executable.materialize(
      *p_->runtime,
      runtime::amr::MaterializationContext{subject, field_name, "recompute",
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
    return p_->runtime->prolong_bootstrap_face_vector(
        pair[0], pair[1], level, prepared, registered.descriptor.refinement_ratio);
  }
  const auto block = p_->bootstrap_block_subjects.find(subject);
  if (block != p_->bootstrap_block_subjects.end()) {
    const int index = p_->block_index(block->second);
    if (index < 0)
      throw std::runtime_error("bootstrap cell transfer names an unknown block");
    return p_->runtime->prolong_bootstrap_block(
        static_cast<std::size_t>(index), level, prepared,
        registered.descriptor.refinement_ratio);
  }
  if (p_->bootstrap_arrays.count(subject) == 0)
    throw std::runtime_error("bootstrap staggered transfer has no registered carrier");
  return p_->runtime->prolong_bootstrap_staggered_field(
      subject, level, prepared, registered.descriptor.refinement_ratio);
}

void AmrSystem::synchronize_bootstrap_state(const std::string& subject, int fine_level) {
  if (!p_->runtime)
    throw std::runtime_error("bootstrap synchronization has no runtime engine");
  const auto block = p_->bootstrap_block_subjects.find(subject);
  const auto route = p_->bootstrap_subject_routes.find(
      {subject, "restriction"});
  if (block == p_->bootstrap_block_subjects.end() ||
      route == p_->bootstrap_subject_routes.end() ||
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

std::vector<double> AmrSystem::bootstrap_array_level(const std::string& subject,
                                                     int level) const {
  if (!p_->runtime)
    throw std::runtime_error("AmrSystem::bootstrap_array_level has no runtime engine");
  return p_->runtime->bootstrap_staggered_level(subject, level);
}

void AmrSystem::invalidate_bootstrap_cache(const std::string& subject, int level) {
  if (!p_->runtime || subject.empty() || level < 0 || level >= p_->runtime->nlev())
    throw std::runtime_error("AmrSystem::invalidate_bootstrap_cache received an invalid key");
  const auto route_id =
      p_->bootstrap_subject_routes.find({subject, "coarse_fine_fill"});
  if (route_id == p_->bootstrap_subject_routes.end())
    throw std::runtime_error("bootstrap cache has no prepared materializer");
  const auto& route = p_->bootstrap_transfer_routes.at(route_id->second);
  if (route.descriptor.space != "cache" || !route.executable.materialize)
    throw std::runtime_error("bootstrap cache route is not an executable materializer");
  route.executable.materialize(
      *p_->runtime,
      runtime::amr::MaterializationContext{subject, subject, "invalidate_cache", level});
}

std::vector<PatchBox> AmrSystem::rebuild_bootstrap_topology_cache(
    const std::string& subject, int level) {
  if (!p_->runtime || level < 0 || level >= p_->runtime->nlev())
    throw std::runtime_error(
        "AmrSystem::rebuild_bootstrap_topology_cache requires a materialized level");
  const auto route_id =
      p_->bootstrap_subject_routes.find({subject, "coarse_fine_fill"});
  if (route_id == p_->bootstrap_subject_routes.end())
    throw std::runtime_error("bootstrap cache has no prepared materializer");
  const auto& route = p_->bootstrap_transfer_routes.at(route_id->second);
  if (route.descriptor.space != "cache" || !route.executable.materialize)
    throw std::runtime_error("bootstrap cache route is not an executable materializer");
  route.executable.materialize(
      *p_->runtime,
      runtime::amr::MaterializationContext{subject, subject, "rebuild_cache", level});
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
  const std::size_t nn = static_cast<std::size_t>(p_->cfg.n) * static_cast<std::size_t>(p_->cfg.n);
  if (bz.size() != nn)
    throw std::runtime_error("AmrSystem::set_magnetic_field : B_z of size " +
                             std::to_string(bz.size()) + " (expected n*n = " + std::to_string(nn) +
                             ", coarse row-major)");
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
  const std::size_t nn = static_cast<std::size_t>(p_->cfg.n) * static_cast<std::size_t>(p_->cfg.n);
  if (field.size() != nn)
    throw std::runtime_error("AmrSystem::set_aux_field : field of size " +
                             std::to_string(field.size()) +
                             " (expected n*n = " + std::to_string(nn) + ", coarse row-major)");
  p_->named_aux_[comp] = field;  // pending: seeded into the engine at build (single + multi block)
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
  // MULTI-BLOCK only: a COUPLED source reads/writes SEVERAL named blocks; the single-block path
  // (AmrCouplerMP) has no block registry and carries its source via the model. We thus refuse a
  // coupled source as long as there are fewer than two blocks (EXPLICIT error rather than a silent no-op).
  if (p_->blocks.size() < 2)
    throw std::runtime_error(
        "AmrSystem::add_coupled_source : inter-species coupled source supported "
        "only in MULTI-BLOCK (>= 2 add_block) ; the single-block carries its source "
        "via the block model");
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
  p_->ensure_built();
  p_->execute_step_transaction([&] {
    // PENDING cadence phase restoration (set_clock before the 1st step): now that the
    // engine exists (ensure_built), we push macro_step_ to its regrid/stride counter.
    if (p_->clock_restore_pending_) {
      p_->push_macro_step_to_engine();
      p_->clock_restore_pending_ = false;
    }
    // COMPILED time-program path (epic ADC-511 / ADC-508): when a Program is installed, its macro-step
    // closure REPLACES the native AmrRuntime::step body (parity SystemStepper::step routing to program_
    // step_), wrapped by the GLOBAL substeps/stride cadence. The closure drives the per-level Lie/Strang
    // macro-step through the AmrProgramContext. Empty (no program installed) -> the historical path.
    if (p_->program_.step_)
      p_->run_program_cadence_(dt);
    else if (p_->runtime)
      p_->runtime->step(static_cast<Real>(dt));
    else
      p_->step_fn(dt);
    p_->t += dt;
    ++p_->macro_step_;  // authoritative counter (parity System: one macro-step = one increment)
  });
}
void AmrSystem::advance(double dt, int nsteps) {
  for (int s = 0; s < nsteps; ++s)
    step(dt);
}
void AmrSystem::begin_step_transaction() {
  p_->ensure_built();
  if (p_->external_step_transaction_)
    throw std::runtime_error("AmrSystem::begin_step_transaction: transaction already active");
  p_->external_step_transaction_ = std::make_unique<Impl::AcceptedSnapshot>(*p_);
  p_->external_step_transaction_committed_ = false;
}
void AmrSystem::commit_step_transaction() {
  if (!p_->external_step_transaction_)
    throw std::runtime_error("AmrSystem::commit_step_transaction: no active transaction");
  if (p_->external_step_transaction_committed_)
    throw std::runtime_error("AmrSystem::commit_step_transaction: transaction already committed");
  p_->external_step_transaction_committed_ = true;
}
void AmrSystem::finalize_step_transaction() {
  if (!p_->external_step_transaction_ || !p_->external_step_transaction_committed_)
    throw std::runtime_error(
        "AmrSystem::finalize_step_transaction: no committed transaction");
  p_->external_step_transaction_.reset();
  p_->external_step_transaction_committed_ = false;
}
void AmrSystem::rollback_step_transaction() {
  if (!p_->external_step_transaction_)
    throw std::runtime_error("AmrSystem::rollback_step_transaction: no active transaction");
  p_->external_step_transaction_->restore(*p_);
  p_->external_step_transaction_.reset();
  p_->external_step_transaction_committed_ = false;
}
double AmrSystem::step_cfl(double cfl, double speed_floor, double max_dt, double min_dt) {
  p_->ensure_built();
  return p_->execute_step_transaction([&]() -> double {
    if (std::isnan(max_dt) || max_dt <= 0.0)
      throw std::invalid_argument("AmrSystem::step_cfl max_dt must be positive or +infinity");
    if (std::isnan(min_dt) || min_dt < 0.0)
      throw std::invalid_argument("AmrSystem::step_cfl min_dt must be finite and >= 0");
    if (p_->clock_restore_pending_) {  // pending phase restoration (cf. step)
      p_->push_macro_step_to_engine();
      p_->clock_restore_pending_ = false;
    }
    const double hx = p_->cfg.L / p_->cfg.n;  // coarse grid spacing (dx_coarse)
    // A Program is always on the runtime engine: compute its CFL bound there, then run the Program.
    if (p_->program_.step_) {
      double dt = static_cast<double>(p_->runtime->cfl_dt(
          static_cast<Real>(cfl), static_cast<Real>(hx), static_cast<Real>(speed_floor)));
      if (std::isfinite(max_dt) && max_dt < dt) {
        dt = std::min(dt, max_dt);
        p_->runtime->override_last_dt_bound("strategy:max_dt");
      }
      if (dt < min_dt)
        throw std::runtime_error("AmrSystem::step_cfl stability bound is below declared min_dt");
      p_->run_program_cadence_(dt);
      p_->t += dt;
      ++p_->macro_step_;
      return dt;
    }
    if (p_->runtime) {
      double dt = static_cast<double>(p_->runtime->cfl_dt(
          static_cast<Real>(cfl), static_cast<Real>(hx), static_cast<Real>(speed_floor)));
      if (std::isfinite(max_dt) && max_dt < dt) {
        dt = std::min(dt, max_dt);
        p_->runtime->override_last_dt_bound("strategy:max_dt");
      }
      if (dt < min_dt)
        throw std::runtime_error("AmrSystem::step_cfl stability bound is below declared min_dt");
      p_->runtime->step(static_cast<Real>(dt));
      p_->t += dt;
      ++p_->macro_step_;
      return dt;
    }

    // Legacy single-block coupler: facade publications are transactional; native state snapshotting
    // remains a separate seam because this path does not own an AmrRuntime.
    if (speed_floor != static_cast<double>(kCflSpeedFloor))
      throw std::runtime_error(
          "AmrSystem::step_cfl : speed_floor is wired on the multi-block runtime engine only (the "
          "single-block AmrCouplerMP CFL divides by the raw max speed) ; leave it default, or use "
          "the multi-block runtime path");
    double h = cfl * hx / p_->max_speed_fn();
    const auto& block = p_->blocks.front();
    p_->last_dt_reason = "transport:" + block.name;
    const double sub = static_cast<double>(block.substeps);
    if (p_->source_frequency_fn) {
      const double mu = p_->source_frequency_fn();
      if (mu > 0.0) {
        const double dt_src = cfl * sub / mu;
        if (dt_src < h) {
          h = dt_src;
          p_->last_dt_reason = "source_frequency:" + block.name;
        }
      }
    }
    if (p_->stability_dt_fn) {
      const double db = p_->stability_dt_fn();
      if (db > 0.0) {
        const double dt_adm = db * sub;
        if (dt_adm < h) {
          h = dt_adm;
          p_->last_dt_reason = "stability_dt:" + block.name;
        }
      }
    }
    for (const auto& g : p_->dt_bounds) {
      if (!g.fn)
        continue;
      double v = g.fn();
      if (!(v > 0.0) || !std::isfinite(v))
        v = std::numeric_limits<double>::infinity();
      v = all_reduce_min(v);
      if (v < h) {
        h = v;
        p_->last_dt_reason = "global:" + g.label;
      }
    }
    if (std::isfinite(max_dt) && max_dt < h) {
      h = max_dt;
      p_->last_dt_reason = "strategy:max_dt";
    }
    if (h < min_dt)
      throw std::runtime_error("AmrSystem::step_cfl stability bound is below declared min_dt");
    p_->step_fn(h);
    p_->t += h;
    ++p_->macro_step_;
    return h;
  });
}

// GLOBAL step bound (AMR counterpart of System::add_dt_bound): registered BEFORE or AFTER the build
// (single-block: read by step_cfl at each step; multi-block: passed to the engine, or added hot
// if it already exists). fn() is evaluated PER RANK then reduced all_reduce_min on the consumer side.
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

// Newton report (OPT-IN IMEX diagnostics) of the block, AGGREGATED over the levels/sub-steps of its
// LAST advance (reset at the head of advance by AmrRuntime::step). Native MULTI-BLOCK only: the
// single-block (AmrCouplerMP coupler) rejects it at build (ensure_built); a call HERE without a runtime
// engine (thus single-block built) raises a clear error (parity System::newton_report: never an empty report).
AmrSystem::SourceNewtonReport AmrSystem::newton_report(const std::string& name) {
  p_->ensure_built();  // materializes the engine (multi-block) or the coupler (single-block)
  if (!p_->runtime)
    throw std::runtime_error(
        "AmrSystem::newton_report : Newton diagnostics not available in single-block (the "
        "AmrCouplerMP "
        "coupler rejects newton_diagnostics at build) ; add a 2nd block (multi-block engine), "
        "or use a single-level System for the full report.");
  const NewtonReport& r =
      p_->runtime->newton_report(name);  // throws if unknown block / diagnostics off
  return SourceNewtonReport{r.enabled,
                            r.converged,
                            static_cast<double>(r.max_residual),
                            static_cast<double>(r.max_iters_used),
                            r.n_failed,
                            r.failed_i,
                            r.failed_j,
                            r.failed_comp,
                            r.diagnostics.events};
}

int AmrSystem::nx() const {
  return p_->cfg.n;
}
double AmrSystem::time() const {
  return p_->t;
}
int AmrSystem::macro_step() const {
  return p_->macro_step_;
}
void AmrSystem::set_clock(double t, int macro_step) {
  if (macro_step < 0)
    throw std::runtime_error("AmrSystem::set_clock : macro_step >= 0 (restart)");
  p_->t = t;
  p_->macro_step_ = macro_step;
  // Pushes the cadence phase (regrid/stride) to the engine: right away if it is already built, otherwise at
  // the 1st step (clock_restore_pending_). set_clock is typically called BEFORE the 1st step (restart of a
  // replayed composition, lazy build), hence the flag.
  if (p_->built)
    p_->push_macro_step_to_engine();
  else
    p_->clock_restore_pending_ = true;
}

// --- compiled time-program install seam on the AMR hierarchy (epic ADC-511 / ADC-508, Spec 6) -------
// AMR counterpart of System::install_program_step / set_program_cadence / install_program / the
// per-block RuntimeParams store. The macro-step body the .so installs lives in the shared Program
// subsystem (program_.step_, ADC-594); AmrSystem::step routes through it (run_program_cadence_) when
// set. The cadence + RuntimeParams stores live in program_ (not the .so closure), mirroring System, so
// a value change reaches the captured context.
void AmrSystem::install_program_step(std::function<void(double)> step) {
  p_->program_.step_ = std::move(step);
}
// GLOBAL macro-step cadence around the installed program closure (parity System::set_program_cadence,
// ADC-411). Validates substeps >= 1 && stride >= 1 (fail-loud: a non-positive cadence is meaningless).
void AmrSystem::set_program_cadence(int substeps, int stride) {
  require_assembling_amr(p_->bound_,
                         "set_program_cadence");  // frozen once pops.bind completes (ADC-592)
  if (substeps < 1)
    throw std::invalid_argument("AmrSystem::set_program_cadence : substeps >= 1 required (got " +
                                std::to_string(substeps) + ")");
  if (stride < 1)
    throw std::invalid_argument("AmrSystem::set_program_cadence : stride >= 1 required (got " +
                                std::to_string(stride) + ")");
  p_->program_.substeps_ = substeps;
  p_->program_.stride_ = stride;
}
// Read the installed GLOBAL cadence (ADC-594, parity System): the tiny const getters the
// ProgramRuntimeReport reads through the bindings (no Python-visible getter existed before).
int AmrSystem::program_substeps() const {
  return p_->program_.substeps_;
}
int AmrSystem::program_stride() const {
  return p_->program_.stride_;
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
std::string AmrSystem::installed_program_hash() const {
  return p_->program_.installed_hash_;
}
// ADC-631: the last macro-step dt handed to the installed Program (set by run_program_cadence_ before
// each step_). The AmrProgramContext reads it so a history ring's store_history tags the per-slot dt
// (variable-dt replay). Parity with ProgramRuntimeState::last_dt_ on the Uniform side.
double AmrSystem::program_last_dt() const {
  return static_cast<double>(p_->program_.last_dt_);
}
std::vector<std::uint8_t> AmrSystem::program_accepted_state() const {
  return p_->program_accepted_state_;
}
void AmrSystem::restore_program_accepted_state(const std::vector<std::uint8_t>& state) {
  p_->program_accepted_state_ = state;
  ++p_->program_accepted_state_revision_;
}
std::uint64_t AmrSystem::program_accepted_state_revision() const {
  return p_->program_accepted_state_revision_;
}
std::vector<std::vector<std::string>> AmrSystem::program_accepted_state_manifest() const {
  std::vector<std::vector<std::string>> rows;
  if (p_->program_accepted_state_.empty())
    return rows;
  const auto state = runtime::program::deserialize_amr_program_accepted_state(
      p_->program_accepted_state_);
  rows.reserve(state.history_owners.size());
  for (const auto& [name, owner] : state.history_owners) {
    const auto ring = state.ring_clocks.find(name);
    const int depth = ring == state.ring_clocks.end()
                          ? (p_->runtime ? detail::AmrHistoryOps::depth(*p_->runtime, name) : 0)
                          : static_cast<int>(ring->second.size());
    rows.push_back({name, "program.block." + std::to_string(owner),
                    state.history_states.at(name), state.history_spaces.at(name),
                    std::to_string(depth), std::to_string(state.level_clocks.size())});
  }
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
// NATIVE per-block RUNTIME parameters (ADC-514): register_block_params + set_block_params are defined in
// this binding-private header, split out to keep the TU on budget; it sees the complete Impl here.
#include "amr_system_params.hpp"
// The built multi-block AMR engine the AmrProgramContext driver wraps (nullptr before the lazy build /
// on the single-block coupler path). install_program forces the runtime build so this is live there.
AmrRuntime* AmrSystem::engine() const {
  return p_->runtime.get();
}
// True on the multi-block AmrRuntime engine (a compiled Program forces it even for ONE block), false on
// the single-block AmrCouplerMP coupler. The v3 checkpoint routes state I/O on this (n_blocks()==1 does
// NOT imply the coupler): the per-block accessors work for any block count on the runtime engine, while
// n_vars / level_state throw there (kAmrCkptMonoOnly).
bool AmrSystem::uses_runtime_engine() const {
  p_->ensure_built();
  return p_->runtime != nullptr;
}
// The facade-owned Profiler (parity System), forwarded to by the AmrProgramContext's profiling seam.
pops::runtime::program::Profiler& AmrSystem::profiler_handle() {
  return p_->program_.profiler_;
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

// Level-composite collective reduction over a named block (ADC-542). Multi-block / forced-runtime
// routes to AmrRuntime::composite_reduce (native masked level sums + unmasked extrema). Single-block
// (AmrCouplerMP coupler, no runtime) composes the SAME reduction from the coupler's native per-level
// state accessors: it gathers each level's valid cells (the existing AMR checkpoint route) and folds
// them with the covered-cell / extrema discipline on the host -- correct, and single-block AMR carries
// at most a shallow hierarchy. COLLECTIVE: called on every rank.
double AmrSystem::composite_reduce(const std::string& block, const std::string& kind,
                                   int comp) const {
  p_->ensure_built();
  if (p_->runtime)
    return p_->runtime->composite_reduce(block, kind, comp);
  // Single-block coupler path: compose from the per-level state gathers (level_state_global under
  // np>1, else level_state). n_vars / n_levels come from the coupler closures.
  const int nlev = p_->n_levels_fn ? p_->n_levels_fn() : 1;
  const int nc = p_->n_vars_fn ? p_->n_vars_fn() : 1;
  const bool gather = n_ranks() != 1;
  const bool full = kind.size() > 4 && kind.compare(kind.size() - 4, 4, "_all") == 0;
  const std::string base = full ? kind.substr(0, kind.size() - 4) : kind;
  const int c0 = full ? 0 : comp;
  const int c1 = full ? nc : comp + 1;
  const int n = p_->cfg.n;  // coarse cells per direction
  const double L = p_->cfg.L;
  const bool is_sum = (base == "sum" || base == "abs_sum" || base == "sum_sq");
  double acc =
      is_sum ? 0.0
             : (base == "min" ? std::numeric_limits<double>::infinity()
                              : (base == "max" ? -std::numeric_limits<double>::infinity() : 0.0));
  for (int k = 0; k < nlev; ++k) {
    const std::size_t nf = static_cast<std::size_t>(n) << k;
    std::vector<double> st =
        (gather && p_->level_state_global_fn)
            ? p_->level_state_global_fn(k)
            : (p_->level_state_fn ? p_->level_state_fn(k) : std::vector<double>());
    if (st.empty())
      continue;
    // Covered cells of level k: the level-(k+1) patches coarsened by 2 (from patch_boxes()).
    std::vector<PatchBox> covered;
    if (k + 1 < nlev && p_->patch_boxes_fn) {
      for (const PatchBox& fb : p_->patch_boxes_fn())
        if (fb.level == k + 1)
          covered.push_back(PatchBox{k, fb.ilo >> 1, fb.jlo >> 1, fb.ihi >> 1, fb.jhi >> 1});
    }
    const double dx = (L / static_cast<double>(n)) / static_cast<double>(1 << k);
    const double cell = dx * dx;
    for (std::size_t j = 0; j < nf; ++j)
      for (std::size_t i = 0; i < nf; ++i) {
        bool cov = false;
        for (const PatchBox& c : covered)
          if (static_cast<int>(i) >= c.ilo && static_cast<int>(i) <= c.ihi &&
              static_cast<int>(j) >= c.jlo && static_cast<int>(j) <= c.jhi) {
            cov = true;
            break;
          }
        for (int c = c0; c < c1; ++c) {
          const double v = st[static_cast<std::size_t>(c) * nf * nf + j * nf + i];
          if (base == "min")
            acc = std::min(acc, v);
          else if (base == "max")
            acc = std::max(acc, v);
          else if (base == "abs_max")
            acc = std::max(acc, v < 0 ? -v : v);
          else if (!cov) {  // volume-weighted sums exclude covered coarse cells
            if (base == "sum")
              acc += cell * v;
            else if (base == "abs_sum")
              acc += cell * (v < 0 ? -v : v);
            else if (base == "sum_sq")
              acc += cell * v * v;
          }
        }
      }
  }
  if (!is_sum && base != "min" && base != "max" && base != "abs_max")
    throw std::runtime_error("AmrSystem::composite_reduce : unknown reduction kind '" + kind +
                             "' for block '" + block + "'");
  return acc;
}

// Load a generated problem.so and install its compiled time Program on the AMR hierarchy. Mirrors
// System::install_program (the loader logic is VERBATIM, only the AMR ABI conventions differ: the
// global-scope promotion is anchored on amr_native_anchor like add_native_block, not pops::abi_key, and
// the install entry is pops_install_program_amr). The blocks must be ALREADY added (the AMR registry is
// frozen at the first lazy build); install_program runs BEFORE the first step so the .so's
// pops_install_program_amr captures an AmrProgramContext over THIS AmrSystem. The .so stays loaded.
POPS_EXPORT void AmrSystem::install_program(const std::string& so_path) {
  require_assembling_amr(p_->bound_,
                         "install_program");  // frozen once pops.bind completes (ADC-592)
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
  void* h = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!h) {
    const char* e = dlerror();
    throw std::runtime_error(
        "AmrSystem::install_program : dlopen('" + so_path + "') : " + std::string(e ? e : "?") +
        " (the pops::AmrSystem seam accessors must be exported AND the module loaded "
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
  // Route registry guard (ADC-599): refuse a problem.so whose embedded route manifest
  // (pops_program_route_manifest) disagrees with the current registry, right after the ABI-key
  // check. Optional symbol: a pre-ADC-599 .so carries nothing -> verify_route_manifest("") no-op.
  {
    auto manifest_fn =
        reinterpret_cast<const char* (*)()>(pops::dynlib::sym(h, "pops_program_route_manifest"));
    try {
      pops::verify_route_manifest(manifest_fn ? std::string(manifest_fn()) : std::string(),
                                  "install_program");
    } catch (...) {
      pops::dynlib::close(h);
      throw;
    }
  }
  auto install =
      reinterpret_cast<void (*)(void*)>(pops::dynlib::sym(h, "pops_install_program_amr"));
  if (!install) {
    pops::dynlib::close(h);
    throw std::runtime_error(
        "AmrSystem::install_program : pops_install_program_amr missing from '" + so_path +
        "' (regenerate the time Program with target='amr_system' : a target='system' .so exports "
        "pops_install_program, installable only on System -- use System for that, or AmrSystem "
        "with "
        "an AMR-target Program)");
  }
#if !defined(_WIN32)
  // Install-time requirement validation (Spec criterion 24, ADC-446/466), AMR subset. The problem.so
  // carries, per operator, its requirements (pops_module_operator_requirements -> read_module_metadata).
  // We validate the two requirement kinds that map UNAMBIGUOUSLY to the AMR facade: (b) BLOCK-INSTANCE
  // (against block_names()) and (c) SOLVER (AMR is always geometric_mg). The (a) AUX-FIELD requirement
  // (B_z / T_e provided via set_magnetic_field / set_aux_field) has no provides_aux query on the AMR
  // facade yet, so it is NOT validated here (parity follow-up, Spec 6) -- never a wrong rejection. A
  // pre-Spec-2 .so (present == false) carries nothing to check -> skip (backward compatible).
  {
    const auto meta = pops::runtime::program::read_module_metadata(h);
    const std::vector<std::string> amr_block_names = block_names();
    auto has_block = [&amr_block_names](const std::string& want) {
      for (const auto& nm : amr_block_names)
        if (nm == want)
          return true;
      return false;
    };
    for (const auto& op : meta.operators) {
      // (b) BLOCK-INSTANCE requirements: an operator that reads another species names the block it needs;
      // reject if it was not added. Verbatim spec message (parity System::install_program).
      for (const auto& blk : pops::runtime::program::required_blocks(op.requirements)) {
        if (!has_block(blk)) {
          pops::dynlib::close(h);
          throw std::runtime_error("operator '" + op.name + "' requires block instance '" + blk +
                                   "'");
        }
      }
      // (c) SOLVER requirement: the AMR hierarchy always solves the Poisson with geometric_mg, so a
      // field operator that requires a DIFFERENT named solver is rejected (verbatim spec message).
      const std::string need_solver = pops::runtime::program::required_solver(op.requirements);
      if (!need_solver.empty() && need_solver != "geometric_mg") {
        pops::dynlib::close(h);
        throw std::runtime_error("field operator '" + op.name + "' requires solver '" +
                                 need_solver + "'");
      }
    }
  }
#endif
  // NAME-based block binding (Spec 3 criterion 23, ADC-457). The Program numbers its blocks in P.state
  // declaration order (the .so's pops_program_block_name table); the AMR facade numbers its blocks in
  // add order (block_names). Bind by NAME, store the program-index -> AMR-block-index map (read by the
  // AmrProgramContext). A Program block whose name has no AMR block fails loud. The explicit block
  // identity table is REQUIRED; positional binding is unsupported.
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
    // FAIL LOUD (ADC-508 v1): the per-level AmrProgramContext driver supports a SINGLE-block AMR
    // Program only. A Program binding >= 2 blocks would lower to the per-level solve_fields_from_state
    // clobber path with no coupled coarse-fine re-solve -- a silently-wrong multilevel coupling. Reject
    // it here with an actionable remedy rather than installing a half-wired multi-block driver. (The
    // single-block AMR Program is the bit-identical parity gate; a genuine multi-block AMR runs the
    // per-block NATIVE route -- pops.compile(Case(layout=AMR(...)), multi-block), ADC-503 -- or a
    // multi-block Program on System.)
    if (n > 1) {
      pops::dynlib::close(h);
      throw std::runtime_error(
          "AmrSystem::install_program : the installed compiled time Program binds " +
          std::to_string(n) +
          " blocks, but a multi-block AMR Program is not supported in v1 (the per-level "
          "AmrProgramContext driver wires a SINGLE block only). Use the per-block NATIVE AMR "
          "route (pops.compile(Case(layout=AMR(...))) with a multi-block layout, ADC-503), or "
          "install the multi-block Program on System (System.install_program). A single-block "
          "AMR Program is supported.");
    }
    std::vector<int> prog_to_sys(static_cast<std::size_t>(n), -1);
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
      prog_to_sys[static_cast<std::size_t>(p)] = found;
    }
    set_program_block_map(prog_to_sys);
  }
  // RUNTIME PARAMETERS (ADC-508, parity ADC-510). Seed each PROGRAM block's RuntimeParams to the .so
  // pops_program_param_* declaration defaults so an install WITHOUT a runtime set behaves as with a const
  // param; a later Python params= route overwrites the supplied values via set_program_params. A Program
  // with no runtime param (the count symbol absent or 0) seeds nothing. VERBATIM mirror of System.
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
      std::map<int, std::vector<double>>
          defaults_by_block;  // program block -> defaults in index order
      for (int i = 0; i < np; ++i) {
        const int blk = pblock(i);
        const int idx = pindex(i);
        std::vector<double>& d = defaults_by_block[blk];
        if (static_cast<int>(d.size()) <= idx)
          d.resize(static_cast<std::size_t>(idx) + 1, 0.0);
        d[static_cast<std::size_t>(idx)] = pdef(i);
      }
      for (const auto& kv : defaults_by_block)
        seed_program_params(kv.first, kv.second);
    }
  }
  // Target-specific counterpart of the uniform generated boundary-kernel install.  It mutates the
  // facade plans before ensure_built(), so the freshly materialized AmrRuntime receives the exact
  // function pointers and execution context on its first construction.
  if (auto install_boundaries = reinterpret_cast<void (*)(void*)>(
          pops::dynlib::sym(h, "pops_install_field_boundaries_amr")))
    install_boundaries(static_cast<void*>(this));
  // Install the macro-step body: the .so wraps THIS AmrSystem in an AmrProgramContext and calls
  // install_program_step. The AMR blocks must be materialized first (ensure_built): the per-level
  // macro-step reads the hierarchy. The AmrProgramContext driver needs the AmrRuntime engine (per-level
  // state / RHS / average_down seam), so we FORCE the multi-block runtime build even for a single block
  // (the AmrRuntime ctor supports 1 block). ensure_built then routes through build_multi.
  p_->force_runtime_ = true;
  p_->ensure_built();
  install(static_cast<void*>(this));
  // Record the program's IR hash (parity System, checkpoint guard). Missing symbol -> empty hash.
  auto hash_fn = reinterpret_cast<const char* (*)()>(pops::dynlib::sym(h, "pops_program_hash"));
  p_->program_.installed_hash_ = hash_fn ? std::string(hash_fn()) : std::string();
  // .so left loaded for the duration of the process (the installed closure points to code in it).
}
// --- AMR / MPI profiling (Spec 5 sec.12.5, ADC-479 criterion 43) ---------------------------------
// enable_profiling / profile_report drive the facade-owned Profiler (parity with System). The
// multi-block AmrRuntime engine (wired at build via set_profiler) times its non-numeric AMR phases
// -- regrid / fill_boundary / average_down -- and bumps the per-run + MPI counters into it. The
// Profiler lives on the Impl (NOT on SystemStepper), so the C++ MockImpl never reads it. enable
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
  report.has_amr = true;
  report.poisson.rhs = p_->p_rhs;
  report.poisson.solver = p_->p_solver;
  report.poisson.bc = p_->p_bc;
  report.poisson.wall = p_->p_wall;
  report.poisson.wall_radius = p_->p_wall_radius;
  report.poisson.epsilon = 1.0;
  report.poisson.abs_tol = static_cast<double>(kMGDefaultAbsTol);
  report.amr_refinement.threshold = p_->refine_threshold;
  report.amr_refinement.disabled =
      !(p_->refine_threshold < static_cast<double>(kAmrRefinementDisabledThreshold));
  report.amr_refinement.disabled_policy =
      report.amr_refinement.disabled ? "legacy_abi_sentinel_threshold" : "explicit_threshold";
  report.amr_refinement.variable = p_->bootstrap_tag_spec
      ? p_->bootstrap_tag_spec->block + "." + p_->bootstrap_tag_spec->variable
      : p_->refine_var_name;
  report.amr_refinement.role = p_->refine_var_role;
  report.amr_refinement.phi_grad_threshold = p_->phi_grad_threshold;
  report.amr_refinement.phi_refinement_enabled =
      p_->phi_grad_threshold > static_cast<double>(kAmrPhiRefinementDisabledThreshold);
  // ADC-616: effective Berger-Rigoutsos clustering params (default {0.7, 1, 32} unless overridden by
  // the AmrSystemConfig cluster_* fields, which mirror the pops.mesh.amr.PatchClustering descriptor).
  report.amr_refinement.cluster_min_efficiency =
      p_->cfg.cluster_min_efficiency > 0.0 ? p_->cfg.cluster_min_efficiency : 0.7;
  report.amr_refinement.cluster_min_box_size =
      p_->cfg.cluster_min_box_size > 0 ? p_->cfg.cluster_min_box_size : 1;
  report.amr_refinement.cluster_max_box_size =
      p_->cfg.cluster_max_box_size > 0 ? p_->cfg.cluster_max_box_size : 32;

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
    row.time = b.imex ? "imex" : (b.time_method == 1 ? "ssprk3" : "explicit");
    row.time_method = b.imex ? "imex" : (b.time_method == 1 ? "ssprk3" : "euler");
    row.imex = b.imex;
    row.substeps = b.substeps;
    row.stride = b.stride;
    row.evolve = true;
    row.implicit_vars = b.implicit_vars;
    row.implicit_roles = b.implicit_roles;
    row.newton = amr_effective_newton_options(b.newton, b.newton_diagnostics);
    row.positivity_floor = b.pos_floor;
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
  if (p_->runtime)
    return p_->runtime->n_patches();
  return p_->n_patches_fn();
}
std::vector<PatchBox> AmrSystem::patch_boxes() {
  p_->ensure_built();
  if (p_->runtime)
    return p_->runtime->patch_boxes();  // MULTI-BLOCK: AmrRuntime engine
  return p_->patch_boxes_fn();          // SINGLE-BLOCK: AmrCouplerMP hook
}
int AmrSystem::coarse_local_boxes() {
  p_->ensure_built();
  if (p_->runtime)
    return p_->runtime->coarse_local_boxes();  // runtime-owned shared layout
  return p_->coarse_local_boxes_fn();          // SINGLE-BLOCK: AmrCouplerMP hook
}
int AmrSystem::coarse_total_boxes() {
  p_->ensure_built();
  if (p_->runtime)
    return p_->runtime->coarse_total_boxes();
  return p_->coarse_total_boxes_fn();
}
double AmrSystem::mass() {
  return mass(std::string());
}
double AmrSystem::mass(const std::string& name) {
  p_->ensure_built();
  if (!p_->runtime)
    return p_->mass_fn();                 // SINGLE-BLOCK: cosmetic name -> single block
  const int idx = p_->block_index(name);  // MULTI-BLOCK: the name indexes the block
  if (idx < 0)
    throw std::runtime_error("AmrSystem::mass : no block named '" + name + "'");
  return static_cast<double>(p_->runtime->mass(static_cast<std::size_t>(idx)));
}
std::vector<double> AmrSystem::density() {
  return density(std::string());
}
std::vector<double> AmrSystem::density(const std::string& name) {
  p_->ensure_built();
  if (!p_->runtime)
    return p_->density_fn();              // SINGLE-BLOCK: cosmetic name -> single block
  const int idx = p_->block_index(name);  // MULTI-BLOCK: the name indexes the block
  if (idx < 0)
    throw std::runtime_error("AmrSystem::density : no block named '" + name + "'");
  return p_->runtime->density(static_cast<std::size_t>(idx));
}
std::vector<double> AmrSystem::potential() {
  p_->ensure_built();
  if (p_->runtime)
    return p_->runtime->potential();  // shared aux (common to all blocks)
  return p_->potential_fn();
}

namespace {
// Common message of the MONO-BLOCK-only checkpoint accessors (ADC-65): the per-level STATE (all
// components) is carried PER BLOCK in multi-block, so level_state / set_level_state / n_vars and the
// hierarchy imposition are MONO-BLOCK only. In multi-block the facade routes to the per-BLOCK variants
// (block_level_state ...) -> EXPLICIT redirection rather than a silent partial/false state. The SHARED
// observables (n_levels, level_potential) work in BOTH paths (single aux), so they do NOT raise this.
const char* const kAmrCkptMonoOnly =
    "AmrSystem : level_state / set_level_state / set_hierarchy are SINGLE-BLOCK only (AmrCouplerMP "
    "coupler) ; this system is multi-block (AmrRuntime engine : shared layout + aux). Use the "
    "per-block accessors block_level_state / set_block_level_state (the Python checkpoint/restart "
    "facade routes to them automatically), or a single add_block.";
}  // namespace

int AmrSystem::n_levels() {
  p_->ensure_built();
  if (p_->runtime)
    return p_->runtime->nlev();  // SHARED hierarchy (common to all blocks)
  return p_->n_levels_fn();
}
int AmrSystem::n_vars() {
  p_->ensure_built();
  if (p_->runtime)
    throw std::runtime_error(kAmrCkptMonoOnly);
  return p_->n_vars_fn();
}
std::vector<double> AmrSystem::level_state(int k) {
  p_->ensure_built();
  if (p_->runtime)
    throw std::runtime_error(kAmrCkptMonoOnly);
  return p_->level_state_fn(k);
}
std::vector<double> AmrSystem::level_state_global(int k) {
  p_->ensure_built();
  if (p_->runtime)
    throw std::runtime_error(kAmrCkptMonoOnly);
  return p_->level_state_global_fn(k);  // ADC-509 np>1 gather (mono-block)
}
void AmrSystem::set_level_state(int k, const std::vector<double>& s) {
  p_->ensure_built();
  if (p_->runtime)
    throw std::runtime_error(kAmrCkptMonoOnly);
  p_->set_level_state_fn(k, s);
}
std::vector<double> AmrSystem::level_potential(int k) {
  p_->ensure_built();
  if (p_->runtime)
    return p_->runtime->level_potential(k);  // SHARED aux / mg warm-start (common to all blocks)
  return p_->level_potential_fn(k);
}
std::vector<double> AmrSystem::level_potential_global(int k) {
  p_->ensure_built();
  if (p_->runtime)
    return p_->runtime->level_potential_global(k);  // ADC-509 np>1 gather (shared phi)
  return p_->level_potential_global_fn(k);
}
void AmrSystem::set_level_potential(int k, const std::vector<double>& p) {
  p_->ensure_built();
  if (p_->runtime) {
    p_->runtime->set_level_potential(k, p);  // SHARED warm-start (common to all blocks)
    return;
  }
  p_->set_level_potential_fn(k, p);
}
void AmrSystem::set_hierarchy(const std::vector<PatchBox>& boxes) {
  p_->ensure_built();
  if (p_->runtime)
    throw std::runtime_error(kAmrCkptMonoOnly);
  p_->set_hierarchy_fn(boxes);
}

// Impose a mid-run MULTI-BLOCK hierarchy from a v3 checkpoint (ADC-542). Regroups the flat level-tagged
// box + owner-rank arrays by level and forwards to AmrRuntime::rebuild_hierarchy (all levels rebuilt,
// reusing regrid R6/R7). MULTI-BLOCK / runtime engine only.
void AmrSystem::rebuild_hierarchy(const std::vector<PatchBox>& boxes,
                                  const std::vector<int>& owner_ranks) {
  p_->ensure_built();
  if (!p_->runtime)
    throw std::runtime_error(
        "AmrSystem::rebuild_hierarchy : MULTI-BLOCK / runtime engine only (the single-block "
        "coupler "
        "uses set_hierarchy). This system has no AmrRuntime engine.");
  if (owner_ranks.size() != boxes.size())
    throw std::runtime_error(
        "AmrSystem::rebuild_hierarchy : boxes and owner_ranks length mismatch");
  const int nlev = p_->runtime->nlev();
  std::vector<std::vector<PatchBox>> level_boxes(static_cast<std::size_t>(nlev));
  std::vector<std::vector<int>> level_owners(static_cast<std::size_t>(nlev));
  for (std::size_t idx = 0; idx < boxes.size(); ++idx) {
    const int k = boxes[idx].level;
    if (k < 0 || k >= nlev)
      throw std::runtime_error("AmrSystem::rebuild_hierarchy : box level out of range");
    level_boxes[static_cast<std::size_t>(k)].push_back(boxes[idx]);
    level_owners[static_cast<std::size_t>(k)].push_back(owner_ranks[idx]);
  }
  p_->runtime->rebuild_hierarchy(level_boxes, level_owners);
}

void AmrSystem::begin_restart_transaction() {
  p_->ensure_built();
  if (p_->restart_transaction_)
    throw std::runtime_error(
        "AmrSystem::begin_restart_transaction : a restart transaction is already active");
  p_->restart_transaction_ = std::make_unique<Impl::AcceptedSnapshot>(*p_);
}

void AmrSystem::commit_restart_transaction() {
  if (!p_->restart_transaction_)
    throw std::runtime_error(
        "AmrSystem::commit_restart_transaction : no restart transaction is active");
  p_->restart_transaction_.reset();
}

void AmrSystem::rollback_restart_transaction() {
  if (!p_->restart_transaction_)
    throw std::runtime_error(
        "AmrSystem::rollback_restart_transaction : no restart transaction is active");
  // Drop the active marker before restoration.  A restoration failure is terminal for this bracket,
  // rather than leaving an unusable nested transaction that masks the original exception.
  std::unique_ptr<Impl::AcceptedSnapshot> accepted = std::move(p_->restart_transaction_);
  accepted->restore(*p_);
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
      throw std::runtime_error("single-block AMR checkpoint cannot restore runtime regrid counters");
    return;
  }
  p_->runtime->restore_checkpoint_counters(regrid_count, topology_epoch);
}
std::vector<int> AmrSystem::checkpoint_temporal_ratios() const {
  p_->ensure_built();
  if (!p_->runtime)
    return {};
  return p_->runtime->checkpoint_temporal_ratios();
}
std::vector<std::vector<std::string>> AmrSystem::checkpoint_transfer_routes() const {
  std::vector<std::vector<std::string>> rows;
  rows.reserve(p_->bootstrap_subject_routes.size());
  for (const auto& [key, route_identity] : p_->bootstrap_subject_routes) {
    const auto& route = p_->bootstrap_transfer_routes.at(route_identity);
    std::string ghosts;
    for (std::size_t index = 0; index < route.descriptor.ghost_depth.size(); ++index) {
      if (index) ghosts += ",";
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

// --- MULTI-BLOCK per-BLOCK per-level checkpoint accessors (ADC-509) --------------------------------
// All require the multi-block runtime (the AmrRuntime engine carries the per-block level stacks on the
// SHARED layout): mono-block uses the level_state path above (explicit redirection). The named block is
// resolved to its index; the runtime accessors mirror AmrCouplerMP's (verbatim loops -> same layout).
namespace {
const char* const kAmrCkptMultiOnly =
    "AmrSystem : block_level_state / set_block_level_state are MULTI-BLOCK only (>= 2 add_block) ; "
    "this system is mono-block. Use level_state / set_level_state.";
}  // namespace

int AmrSystem::block_n_vars(const std::string& name) {
  p_->ensure_built();
  if (!p_->runtime)
    throw std::runtime_error(kAmrCkptMultiOnly);
  return p_->runtime->block_n_vars(p_->block_index_or_throw(name));
}
std::vector<double> AmrSystem::block_level_state(const std::string& name, int k) {
  p_->ensure_built();
  if (!p_->runtime)
    throw std::runtime_error(kAmrCkptMultiOnly);
  return p_->runtime->block_level_state(p_->block_index_or_throw(name), k);
}
std::vector<double> AmrSystem::block_level_state_global(const std::string& name, int k) {
  p_->ensure_built();
  if (!p_->runtime)
    throw std::runtime_error(kAmrCkptMultiOnly);
  return p_->runtime->block_level_state_global(p_->block_index_or_throw(name), k);
}
void AmrSystem::set_block_level_state(const std::string& name, int k,
                                      const std::vector<double>& s) {
  p_->ensure_built();
  if (!p_->runtime)
    throw std::runtime_error(kAmrCkptMultiOnly);
  p_->runtime->set_block_level_state(p_->block_index_or_throw(name), k, s);
}
std::vector<int> AmrSystem::level_owner_ranks(int k) {
  p_->ensure_built();
  if (!p_->runtime)
    return {};  // single-block coupler: no runtime DistributionMapping to serialize (serial path)
  return p_->runtime->level_owner_ranks(k);
}
// Full shared aux of a level (ALL components) -- the v3 checkpoint aux payload (ADC-542). The coupler
// path returns EMPTY (derived + static-reapplied aux, phi suffices): the writer skips the aux keys.
std::vector<double> AmrSystem::level_aux_flat(int k) {
  p_->ensure_built();
  if (!p_->runtime)
    return {};
  return p_->runtime->level_aux_flat(k);
}
std::vector<double> AmrSystem::level_aux_flat_global(int k) {
  p_->ensure_built();
  if (!p_->runtime)
    return {};
  return p_->runtime->level_aux_flat_global(k);
}
void AmrSystem::set_level_aux_flat(int k, const std::vector<double>& v) {
  p_->ensure_built();
  if (!p_->runtime)
    throw std::runtime_error(
        "AmrSystem::set_level_aux_flat : MULTI-BLOCK / runtime engine only (the coupler aux is "
        "derived and re-applied by solve_fields; restore phi via set_level_potential)");
  p_->runtime->set_level_aux_flat(k, v);
}

// --- ADC-631 multistep history-ring checkpoint / replay seam (Uniform System seam names) -----------
// Thin facade wrappers over detail::AmrHistoryOps on the built AmrRuntime engine, so the SHARED
// python/pops/runtime/_system_io_history.py serialize/restore is reused verbatim: history_global
// returns the per-level slices concatenated into ONE flat buffer (the level axis hidden inside the
// accessor, parity with level_aux_flat), restore_history scatters it back per level. The engine is the
// multi-block AmrRuntime (install_program forces its build); an engine-less coupler has no rings ->
// history_names() is empty and serialize_histories is a no-op. rebuild_history_slots replays the
// policy-recomputed slots by re-stepping the installed Program closure (owned by this facade).
namespace {
const char* const kAmrHistNoEngine =
    "AmrSystem : multistep history rings require the multi-block AmrRuntime engine (a compiled AMR "
    "Program forces its build via install_program); this system has none.";
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
void AmrSystem::set_history_initialized(const std::string& name, bool initialized) {
  if (!p_->runtime)
    throw std::runtime_error(kAmrHistNoEngine);
  detail::AmrHistoryOps::set_initialized(*p_->runtime, name, initialized);
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
  // ADC-635: the replay re-steps the installed Program with regrid ACTIVE. The head-of-step
  // ctx.regrid_if_due(ctx.macro_step()) reads THIS facade's macro_step_: the closure drives it to the
  // engine-supplied per-re-step cursor (m-1-j) so the original in-window regrid schedule fires. m is
  // the facade cursor (primed by the v3 reader); it is restored on every exit, coherence failure too.
  Impl* imp = p_.get();
  const int m = p_->macro_step_;
  detail::AmrHistoryOps::ReplayOutcome outcome;
  try {
    outcome = detail::AmrHistoryOps::rebuild_slots(
        *p_->runtime, name, stored_slots, m, [imp](double dt, int cursor) {
          imp->macro_step_ = cursor;  // ctx.macro_step() -> facade cursor -> regrid_if_due schedule
          imp->program_.last_dt_ = static_cast<Real>(dt);
          imp->program_.step_(dt);
        });
  } catch (...) {
    p_->macro_step_ = m;
    throw;
  }
  p_->macro_step_ = m;
  p_->last_replay_regrid_steps_ = outcome.fired_regrid_steps;
  return outcome.recomputed;
}
std::vector<int> AmrSystem::last_replay_regrid_steps() const {
  return p_->last_replay_regrid_steps_;
}

}  // namespace pops
