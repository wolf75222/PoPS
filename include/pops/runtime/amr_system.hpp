#pragma once

#include <limits>

#include <pops/mesh/boundary/prepared_hyperbolic_boundary.hpp>
#include <pops/numerics/nonlinear/newton_options.hpp>
#include <pops/numerics/elliptic/interface/spatial_provider.hpp>
#include <pops/coupling/source/coupling_operator.hpp>  // CouplingOperator / CouplingOperatorView (typed contract, ADC-595)
#include <pops/runtime/export.hpp>  // POPS_EXPORT: set_compiled_block resolved by the native AMR loader
#include <pops/runtime/facade_options.hpp>  // CoupledSourceProgram (facade POD, ADC-214)
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/config/runtime_params.hpp>  // RuntimeParams (compiled-Program runtime params on AMR, ADC-508)
#include <pops/runtime/config/spatial_domain.hpp>
#include <pops/runtime/amr_patch.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/runtime/numerical_defaults.hpp>
#include <pops/runtime/amr/prepared_component_providers.hpp>
#include <pops/runtime/amr/field_solver_options.hpp>
#include <pops/runtime/amr/hierarchy_policy_authority.hpp>
#include <pops/runtime/amr/hierarchy_tensor_solver_provider.hpp>
#include <pops/parallel/prepared_load_balance.hpp>
#include <pops/runtime/output_piece.hpp>
#include <pops/runtime/system/system_poisson_options.hpp>

#include <array>
#include <functional>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

/// @file
/// @brief Multi-species composition on AMR at runtime: the refined counterpart of System.
///
/// One or SEVERAL blocks (species, described by ModelSpec of generic bricks) carried on an
/// AMR hierarchy. Like System but on an adaptive mesh.
///
/// One and many-block systems use the same AmrRuntime engine. Every block is co-located on ONE
/// SHARED AMR hierarchy (same ranked patches, ownership and spacing per level, guarded by
/// same_layout_or_throw). All blocks live on ALL patches. A single aux per level (phi,
/// grad phi) and a single coarse Poisson whose right-hand side is the CO-LOCATED SUM of the blocks'
/// elliptic bricks (f = Sum_b q_b n_b read at the same cells). Conservation PER BLOCK (reflux +
/// average_down). AmrRuntime runtime engine (type-erased registry by name). Blocks with potentially
/// DIFFERENT spatial schemes. Per-block temporal descriptors are immutable authoring metadata for
/// Program normalization; this spatial runtime does not decode or execute them. Union-of-tags
/// regridding is supported (multi-block + regrid_every > 0 regrids from the union; zero freezes the
/// hierarchy). Multirate cadence and inter-species coupled rates execute only where the installed
/// ProgramGraph places their typed operations. Multiple COMPILED blocks (add_compiled_model) and a
/// MIX of compiled + native blocks share the same hierarchy (capstone v, multi-block production DSL).
///
/// @note Resolved explicit Programs support N ratio-2 levels. An implicit/IMEX AMR composition
/// without a typed Program primitive fails closed; there is no private Newton or time-step fallback.

namespace pops {

class FieldNullspaceProvider;
struct FieldLogicalTimePoint;
template <int Dim>
struct CompiledFieldBoundaryKernel;

class ObserverMpiLane;
namespace runtime::program {
template <int Dim, class MemorySpace>
class AmrProgramContext;
}

namespace runtime::field {
struct PreparedFieldSolverSpec;
struct FieldTopologyReportRow;
}  // namespace runtime::field

/// Exact read-only backend configuration retained for one resolved AMR field solver.
struct AmrFieldSolverConfiguration {
  std::string plan_identity;
  std::string provider_identity;
  std::string solver;
  AmrFieldHierarchyPolicyAuthority hierarchy_policy;
  AmrFieldSolverOptions options;
};

// Forward declarations of the runtime multi-block engine (definitions in amr_runtime.hpp /
// amr_dsl_block.hpp). set_compiled_block stores a DEFERRED runtime-block BUILDER which, given the
// SHARED layout materialized at lazy build, returns the type-erased AmrRuntimeBlock of the compiled
// block: this is what lets SEVERAL compiled blocks (multi-block production DSL) co-exist on the
// SAME AMR hierarchy, exactly like add_block in native multi-block. We forward-declare so as NOT
// to weigh down this public header (read by bindings.cpp and the loaders) with amr_runtime.hpp: only
// the TUs that BUILD/CALL the builder (amr_dsl_block.hpp and python/amr_system.cpp) include the
// complete definitions; a std::function with incomplete-type signature is legal as long as it is
// not instantiated with a concrete callable outside those TUs (PIMPL std::function recipe).
template <int Dim>
struct AmrRuntimeBlock;
// Forward-declared for the named-elliptic-field RHS closure signature (ADC-428,
// set_block_elliptic_field): a std::function with an incomplete-type parameter is legal as long as it is
// only INSTANTIATED with a concrete callable in the TUs that include the full definition (the native AMR
// loader / python/bindings/amr/amr_system.cpp), per the PIMPL std::function recipe noted above.
class AmrFieldSolverProvider;
namespace runtime::amr {
template <int Dim>
class AmrRuntime;
}
namespace detail {
template <int Dim>
struct SharedAmrLayout;
}
namespace runtime {
namespace program {
class
    Profiler;  // forward-declared so engine()/profiler_handle() do not pull profiler.hpp into this header
template <int Dim>
struct ProgramRuntimeState;
}  // namespace program
namespace multiblock {
struct AxisAlignedInterface;
struct PreparedInterfaceFluxSpec;
}  // namespace multiblock
}  // namespace runtime

/// Exact ranked AMR mesh and cadence (per-block physical parameters live in the ModelSpec).
template <int Dim>
struct AmrSystemConfig : RuntimeSpatialDomain<Dim> {
  static_assert(Dim >= 1 && Dim <= 3, "AmrSystemConfig only supports dimensions 1, 2, and 3");

  AmrSystemConfig() { this->shape = runtime_config_detail::filled_extent<Dim>(128); }

  int regrid_every = 20;  ///< re-refinement every N steps (0 = never after init)
  int level_count = 2;    ///< maximum active hierarchy depth (>= 1)
  /// Exact level-to-level hierarchy graph. Each table contains one ranked row per transition;
  /// ratios are >= 2 and buffers/lookaheads are >= 0 component-wise.
  std::vector<Extent<Dim>> transition_ratios{runtime_config_detail::filled_extent<Dim>(2)};
  std::vector<Extent<Dim>> transition_buffers{runtime_config_detail::filled_extent<Dim>(2)};
  std::vector<Extent<Dim>> transition_lookaheads{runtime_config_detail::filled_extent<Dim>(2)};
  bool explicit_bootstrap = false;  ///< coarse-only start; BootstrapPlan creates fine levels
  /// OWNERSHIP POLICY of the coarse level (cf. AmrCouplerMP::replicated_coarse).
  /// false (DEFAULT, historical): coarse mono-box REPLICATED on all ranks. The coarse Poisson
  ///   and the coarse transport are REDUNDANT on each GPU (zero communication,
  ///   better geometric MG) but DO NOT SCALE: only the fine patches are distributed.
  /// true (strong-scaling mode): coarse MULTI-BOX (BoxArray::from_domain, tile size
  ///   coarse_max_grid) distributed round-robin across the ranks. The coarse Poisson and the coarse
  ///   transport distribute (each rank carries only its tiles), which removes the redundancy
  ///   and enables AMR strong-scaling. The geometric MG then operates on a multi-box coarse
  ///   (cf. geometric_mg.hpp): convergence to be measured (may require more cycles).
  bool distribute_coarse = false;
  /// Per-axis coarse tile limits when distribute_coarse=true. Zero selects the hierarchy policy's
  /// deterministic ranked default on that axis. Ignored otherwise.
  Extent<Dim> coarse_max_grid{};
  /// ADC-616: Berger-Rigoutsos clustering params of the regrid layout. <= 0 (default) keeps the
  /// historical ClusterParams {0.7, 1, 32}, bit-identical. min_efficiency in (0,1], sizes >= 1,
  /// min_box_size <= max_box_size (validated at set_clustering / the facade descriptor).
  double cluster_min_efficiency = 0.0;
  int cluster_min_box_size = 0;
  int cluster_max_box_size = 0;
  /// Prepared ownership provider selected by the resolved adaptive-layout authority.  The semantic
  /// identity covers the public route, exact provider options and its weight capability.
  std::string load_balance_route = "space_filling_curve";
  std::string load_balance_identity = "pops.amr.default.space-filling-curve@1";
  PreparedProviderOptions load_balance_options{"pops.amr.load-balance.space-filling-curve@1", {}};
};

/// Frozen parameters passed to the deferred build of the compiled path (add_compiled_model). Materialized
/// by AmrSystem at ensure_built time: the geometry, Poisson and initial-state choices known
/// at that moment. The amr_dsl_block header consumes them to instantiate AmrCouplerMP<Model>.
///
/// STRUCTURE (ADC-610). Settings are grouped into NAMED sub-structs by ownership/role
/// (mesh, poisson, initial data, named aux) instead of one flat
/// append-only bag. A new setting goes INTO its semantic group -- the historical "add at the tail so an
/// older .so loader falls back silently" idiom is RETIRED because it no longer describes how this struct
/// evolves. The ABI story is now the VERSIONED KEY, not tail-only placement: this struct crosses the
/// dlopen .so boundary BY VALUE, and any layout change (a new field, a regroup) shifts POPS_HEADER_SIG
/// (a sha256 over include/, cf. abi_key.hpp / python/CMakeLists.txt), which re-keys pops_native_abi_key.
/// A .so generated before the change then diverges from the module key and add_native_block REJECTS it
/// with a clear "regenerate" error (never silent UB). So an older .so is refused, not silently truncated.
template <int Dim>
struct AmrBuildParams {
  static_assert(Dim >= 1 && Dim <= 3, "AmrBuildParams only supports dimensions 1, 2, and 3");

  /// Coarse mesh geometry + coarse ownership policy (AMR strong-scaling).
  struct Mesh : RuntimeSpatialDomain<Dim> {
    Mesh() { this->shape = runtime_config_detail::filled_extent<Dim>(128); }

    int regrid_every = 20;           ///< re-refinement cadence (0 = never after init)
    bool distribute_coarse = false;  ///< distributed multi-box coarse (AMR strong-scaling)
    Extent<Dim> coarse_max_grid{};   ///< exact per-axis tile cap
    std::shared_ptr<const PreparedLoadBalanceAuthority<Dim>>
        load_balance;  ///< one prepared authority reused by coarse, fine seeds and every regrid
  } mesh;
  /// Exact prepared field provider slot. Legacy 2D Poisson/embedded-boundary objects are installed
  /// through their capability-qualified provider and never become part of this ranked build POD.
  std::string field_provider_slot;
  /// Initial coarse seed: density only (historical) OR the FULL conservative state (priority).
  struct InitialData {
    bool has_density = false;
    std::vector<double> density;  ///< initial coarse density (component 0), ny*nx
    // FULL initial conservative state (all components), takes priority over `density` when has_state.
    bool has_state = false;
    std::vector<double> state;  ///< ncomp*ny*nx, component-major c*cells + j*nx + i
  } initial;
  /// Model-NAMED aux fields (ADC-291) + their per-field HALO policies (ADC-369). Seeded onto the
  /// runtime block at build, like bz_field; re-applied each update (persist across regrid).
  /// Both empty -> bit-identical.
  struct NamedAux {
    std::map<int, std::vector<double>>
        fields;  ///< component (>= kAuxNamedBase) -> coarse field (ny*nx)
    std::map<int, AuxHaloPolicy> halo_policies;  ///< component -> uniform boundary policy
  } named_aux;
};

/// DEFERRED builder of a COMPILED block on the multi-block hierarchy: receives the SHARED layout (created
/// ONCE at lazy build, common to all blocks) plus the block parameters frozen at
/// add time (name, initial density/state, gamma, substeps/stride and reconstruction metadata),
/// and returns the type-erased AmrRuntimeBlock of the block
/// (captures the CONCRETE Model/Limiter/Flux via detail::dispatch_amr_block, the kernel stays
/// COMPILED). Symmetric with the
/// native add_block path: the (sole) difference is only that the types are known at add time (compiled
/// model) rather than resolved from a ModelSpec at build. The SIGNATURE mentions FORWARD-DECLARED types:
/// it is instantiated with a concrete callable only in add_compiled_model(AmrSystem&) (header
/// amr_dsl_block.hpp) where those types are complete, and invoked only in python/amr_system.cpp.
/// The trailing pos_floor is the Zhang-Shu positivity floor of the block (0 = inactive), forwarded to
/// dispatch_amr_block -> build_amr_block exactly like a native multi-block.
template <int Dim>
using AmrCompiledBlockBuilder = std::function<AmrRuntimeBlock<Dim>(
    const detail::SharedAmrLayout<Dim>& layout, const std::string& name,
    const std::vector<double>& density, bool has_density, const std::vector<double>& state,
    bool has_state, double gamma, int substeps, bool recon_prim, int stride,
    // Compatibility slots in the frozen builder ABI. Registration rejects every non-empty selector
    // before this callable is stored; the spatial builder never resolves or executes an IMEX mask.
    const std::vector<std::string>& implicit_vars, const std::vector<std::string>& implicit_roles,
    double pos_floor, double weno_epsilon, bool wave_speed_cache)>;

/// Single block carried on an AMR hierarchy, composed at runtime.
///
/// @code{.cpp}
/// pops::AmrSystemConfig cfg;                // base level: n x ny on independent x/y bounds
/// cfg.n = 64;
/// pops::AmrSystem amr(cfg);
///
/// pops::ModelSpec ne;
/// ne.transport = "exb"; ne.source = "none"; ne.elliptic = "charge";
/// amr.add_block("ne", ne, "minmod", "rusanov", "conservative", "explicit");
/// amr.set_poisson("charge_density", "geometric_mg");
/// amr.set_density("ne", rho0);             // rho0: initial density on the base level
/// amr.step_cfl(0.4);                       // conservative refluxed step + composite FAC Poisson
/// @endcode
template <int Dim>
class AmrSystem {
  static_assert(Dim >= 1 && Dim <= 3, "AmrSystem only supports dimensions 1, 2, and 3");

 public:
  using HyperbolicBoundary = PreparedHyperbolicBoundary<Dim>;
  static constexpr int dimension = Dim;

  explicit AmrSystem(const AmrSystemConfig<Dim>& cfg);
  ~AmrSystem();
  // RULE OF FIVE (C.21): move-only (PIMPL unique_ptr). The copy was already IMPLICITLY deleted
  // (move ctor declared); we make it EXPLICIT for intent. No API change (the copy was
  // already unusable).
  AmrSystem(const AmrSystem&) = delete;
  AmrSystem& operator=(const AmrSystem&) = delete;
  AmrSystem(AmrSystem&&) noexcept;
  AmrSystem& operator=(AmrSystem&&) noexcept;

  /// GLOBAL time-step bound (AMR counterpart of System::add_dt_bound): fn() evaluated ONCE
  /// per step_cfl (host), all_reduce_min (identical dt on all ranks), <= 0 / non-finite =
  /// inert this step. Hook for non-local constraints (coupling, scheduler, user ramp).
  void add_dt_bound(const std::string& label, std::function<double()> fn);

  /// ACTIVE bound of the last step_cfl: "transport:<block>" | "source_frequency:<block>" |
  /// "stability_dt:<block>" | "global:<label>" | "degenerate" | "" (no CFL step yet).
  std::string last_dt_bound() const;

  /// Adds a block carried on the AMR. Same spatial-scheme parameters as System
  /// (limiter x riemann x recon), applied to each level/patch of the hierarchy. The FIRST
  /// add_block defines the block; a 2nd (or more) switches to the multi-block engine (shared
  /// hierarchy, co-located sum Poisson). Blocks can have DIFFERENT SPATIAL SCHEMES.
  /// @param name    block name: INDEXES the block (set_density(name), mass(name), density(name)). In
  ///                multi-block the name must be unique; mono-block an empty name targets the single block.
  /// @param model   composition of bricks (transport/source/elliptic + parameters)
  /// @param limiter "none" | "minmod" | "vanleer" | "weno5" | "mc" | "superbee"
  ///                (weno5 = WENO5-Z, 3 ghosts; native low-level stencil route). The resolved
  ///                Case route derives its
  ///                coarse/fine order and halo requirements from this spatial descriptor and
  ///                selects the minimum sufficient conservative provider.
  /// @param riemann "rusanov" | "hll" (generic signed-wave, requires model.wave_speeds) | "hllc"
  ///                | "roe" (requires the model's exact HasHLLCStructure /
  ///                HasRoeDissipation capability; no layout inference or fallback)
  /// @param time    "explicit" (SSPRK2) | "euler" (forward Euler) | "ssprk3"
  ///                (SSPRK3, order 3, reflux per stage; explicit transport, EXCLUSIVE of imex) |
  ///                "imex". This is immutable authoring metadata consumed by Program normalization;
  ///                the spatial block never executes a time method. A composition without a typed
  ///                Program primitive fails closed instead of reaching a private fallback.
  /// @param substeps declared Program substeps of the block (>= 1): the normalized ProgramGraph
  ///                partitions the effective step into equal pieces.
  /// @param stride  HOLD-THEN-CATCH-UP cadence of the block (>= 1; default 1 = each macro-step). stride=M
  ///                holds the block M-1 macro-steps then catches it up by an effective step M*dt (multirate).
  ///                Program-owned step_cfl honors the cadence: dt =
  ///                cfl*h*min_b(substeps_b/(stride_b*w_b)), mirror of System::step_cfl.
  /// @param implicit_vars / implicit_roles  reserved partial-IMEX selectors. The current AMR target
  ///                has no typed implicit-source Program primitive, so every non-empty selector
  ///                fails closed, including with time="imex"; it is never stored or ignored.
  /// @param newton  compatibility input validated like System::add_block. The spatial AMR block does
  ///                 not retain or execute source-Newton configuration, so every non-default value
  ///                 fails closed until the AMR target provides an executable typed local
  ///                 nonlinear/Newton Program primitive.
  /// @param newton_diagnostics  request for an aggregated Newton report owned by the typed implicit
  ///                 Program solve. The spatial runtime has no such report carrier, so true fails
  ///                 closed instead of exposing a permanently empty diagnostic.
  /// @param positivity_floor  Zhang-Shu positivity floor (ADC-259): if > 0, the AMR transport floors
  ///                 the Density-role face states (reconstruct_pp / zhang_shu_scale) AND the C/F fine
  ///                 ghost means to >= floor. Default 0 = inactive, bit-identical. Guarantee = face /
  ///                 ghost-state Density positivity only (order-1 fallback), NOT updated-mean nor
  ///                 pressure positivity (parity with System::add_block). A model without a Density
  ///                 role rejects floor > 0. The COMPILED .so path carries it too (ADC-322): a loader
  ///                 regenerated against this header marshals the floor (pops_install_native_amr).
  /// @throws std::runtime_error if a block is already defined, if substeps < 1, if stride < 1, if time
  ///         is not in {explicit, euler, ssprk3, imex}, if recon is not in {conservative,
  ///         primitive}, if a non-empty implicit selector is requested, or if source-Newton options
  ///         or diagnostics are requested on the spatial AMR block.
  void add_block(const std::string& name, const ModelSpec& model,
                 const std::string& limiter = "minmod", const std::string& riemann = "rusanov",
                 const std::string& recon = "conservative", const std::string& time = "explicit",
                 int substeps = 1, int stride = 1,
                 const std::vector<std::string>& implicit_vars = {},
                 const std::vector<std::string>& implicit_roles = {},
                 const NewtonOptions& newton = {}, bool newton_diagnostics = false,
                 double positivity_floor = 0.0,
                 double weno_epsilon = static_cast<double>(kWenoEpsilon),
                 bool wave_speed_cache = false);

  /// Registers a COMPILED block (add_compiled_model path, header amr_dsl_block.hpp). The single
  /// type-erased builder materializes an AmrRuntimeBlock on the runtime-owned shared hierarchy.
  /// Single- and multi-block systems deliberately use this same route: no hidden AmrCouplerMP
  /// orchestration or second elliptic-solver authority survives behind this facade.
  /// @p recon_prim / @p imex / @p time / @p stride are authoring metadata frozen at add time. The
  /// canonical @p time token is normalized into the installed ProgramGraph; the runtime block remains
  /// spatial. @p implicit_vars / @p implicit_roles remain in this loader-facing signature for ABI
  /// compatibility only: the current AMR target rejects every non-empty selector before storing the
  /// block because no typed implicit-source Program primitive consumes it.
  /// DO NOT call directly: go through the free function add_compiled_model(AmrSystem&, ...).
  /// @throws std::runtime_error if the system is already built.
  /// DEFAULT VISIBILITY (POPS_EXPORT): the ONLY method called by the header template
  /// add_compiled_model(AmrSystem&) (cf. amr_dsl_block.hpp). A generated .so loader (DSL
  /// "production" path on the AMR side, emit_cpp_native_loader(target="amr_system") / add_native_block) inlines this
  /// template and must resolve this symbol from the already-loaded _pops module; compiled with
  /// -fvisibility=hidden (pybind11), the module would not export it without this annotation and the dlopen
  /// of the loader would fail. Symmetric with the POPS_EXPORT methods of System (grid_context/install_block).
  POPS_EXPORT void set_compiled_block(
      int ncomp, double gamma, int substeps, AmrCompiledBlockBuilder<Dim> runtime_builder,
      const std::string& name = std::string(), bool recon_prim = false,
      const std::string& time = "euler", int stride = 1,
      const std::vector<std::string>& implicit_vars = {},
      const std::vector<std::string>& implicit_roles = {}, double pos_floor = 0.0,
      double weno_epsilon = static_cast<double>(kWenoEpsilon), bool wave_speed_cache = false);

  /// Install the same exact-ranked hyperbolic authority as System. Same-level halo exchange and
  /// coarse/fine transfer remain separate hierarchy operations; physical laws are evaluated only
  /// by this retained model-qualified object.
  POPS_EXPORT void install_hyperbolic_boundary(
      const std::string& name, const std::string& identity, int required_depth,
      const std::vector<std::string>& face_types, const std::vector<double>& face_values,
      const std::vector<std::string>& face_identities,
      const std::vector<std::string>& component_roles, const std::string& state_identity,
      const std::vector<std::string>& face_representations = {},
      const std::vector<std::string>& face_converter_identities = {},
      const std::vector<std::vector<std::string>>& face_analytic_opcodes = {},
      const std::vector<std::vector<double>>& face_analytic_literals = {},
      const std::vector<std::string>& face_analytic_clocks = {});
  POPS_EXPORT void install_prepared_hyperbolic_boundary(
      const std::string& name, const std::string& identity, int required_depth,
      const std::string& state_identity, std::shared_ptr<const HyperbolicBoundary> boundary);
  /// Register the exact state Handle independently from physical-boundary ownership.
  POPS_EXPORT void install_block_state_route(const std::string& name,
                                             const std::string& state_identity);
  /// Bind one exact solved-field Handle identity to its authenticated provider storage slot.
  /// Boundary components and AMR tagging consume this common prepared route.
  POPS_EXPORT void install_field_storage_route(const std::string& field_identity,
                                               const std::string& provider_slot);
  /// Roll back a failed pre-build runtime-authority transaction.  Internal bind seam only.
  POPS_EXPORT void discard_hyperbolic_boundaries();
  POPS_EXPORT void install_amr_tagger_component(
      runtime::amr::PreparedTaggerSpec spec, std::shared_ptr<component::LoadedComponent> component);
  POPS_EXPORT void install_amr_clustering_component(
      runtime::amr::PreparedClusteringSpec spec,
      std::shared_ptr<component::LoadedComponent> component);
  POPS_EXPORT void install_amr_reflux_component(
      runtime::amr::PreparedRefluxSpec spec, std::shared_ptr<component::LoadedComponent> component);
  POPS_EXPORT void discard_amr_provider_components();
  /// Materialize one exact shared NumericalFlux route on a frozen AMR level.  This seam is called
  /// only after the lazy AmrRuntime has been built and before bind freezes composition.
  POPS_EXPORT void install_interface_flux_component(
      runtime::multiblock::AxisAlignedInterface route,
      runtime::multiblock::PreparedInterfaceFluxSpec spec,
      std::shared_ptr<component::LoadedComponent> component);
  /// Roll back a failed all-interface post-block installation transaction.
  POPS_EXPORT void discard_interface_flux_components();
  /// Internal bind transaction checkpoint for incremental per-level interface installation.
  POPS_EXPORT std::size_t interface_flux_installation_checkpoint() const;
  POPS_EXPORT void rollback_interface_flux_installations(std::size_t accepted_size);
  POPS_EXPORT std::size_t interface_evaluation_count(const std::string& identity,
                                                     int level = 0) const;

  /// Internal installation seam for a compiled AMR production package. The .so inlines the header
  /// template add_compiled_model(AmrSystem&, ...), which materializes a concrete AmrRuntimeBlock at
  /// lazy build and installs its spatial Program primitives via set_compiled_block. All block counts
  /// use the same shared hierarchy; no flat-array marshaling or alternate temporal engine is involved.
  ///
  /// The _pops host module is PROMOTED to global scope (RTLD_NOLOAD), then the generated package is
  /// opened RTLD_LOCAL: it can resolve set_compiled_block without exporting its identically named
  /// generated templates to later semantic artifacts. The ABI key baked in the package
  /// (pops_native_abi_key) is compared to the module's (abi_key()) -- mismatch => clear error (no
  /// silent UB at the C++ boundary). Same scheme guard-rails as System (upstream validation).
  ///
  /// MULTI-BLOCK (capstone v): add_native_block CAN now be called several times (or mixed
  /// with native add_block) -> the compiled blocks co-exist on the shared hierarchy via AmrRuntime
  /// (the loader recompiled against this header provides the runtime builder; cf. set_compiled_block). The
  /// name then INDEXES the block (set_density/mass/density), like add_block.
  /// time accepts canonical Program-authoring tokens {explicit, euler, ssprk3, imex}; the spatial
  /// loader never executes one of those methods. In particular, ``imex`` requires a typed implicit
  /// Program primitive and has no backward-Euler/Newton fallback. The multirate stride and partial
  /// IMEX mask do not transit through the flat block-loader ABI: the Python facade rejects them
  /// rather than ignoring them silently. They must be expressed by a compiled typed Program whose
  /// target provides the corresponding primitive. recon "primitive" and flux "roe"/"hllc" are WIRED at parity (#113);
  /// the Python facade applies a pressure guard for hllc/roe.
  /// The low-level loader contains a WENO5-Z stencil route and allocates its three-cell halo. The
  /// resolved Case route authenticates the matching order-five conservative coarse/fine provider;
  /// neither the facade nor AmrRuntime substitutes the order-two route.
  /// @throws std::runtime_error if the ABI diverges, if a symbol is missing, or substeps < 1.
  /// @param name block name: cosmetic in mono-block, INDEXES the block in multi-block (set_density/
  ///             mass/density; must be unique and non-empty from the 2nd block on, like add_block).
  /// @param positivity_floor  Zhang-Shu positivity floor of the block (ADC-322): the .so flat ABI now
  ///             carries it (pops_install_native_amr -> add_compiled_model -> set_compiled_block), so a
  ///             loader regenerated against this header floors the Density-role face states like a
  ///             native add_block. 0 (default) = inactive, bit-identical.
  void add_native_block(const std::string& name, const std::string& so_path,
                        const std::string& limiter = "minmod",
                        const std::string& riemann = "rusanov",
                        const std::string& recon = "conservative",
                        const std::string& time = "explicit",
                        double gamma = static_cast<double>(kPhysicalDefaultGamma), int substeps = 1,
                        const std::vector<double>& params = {}, double positivity_floor = 0.0,
                        double weno_epsilon = static_cast<double>(kWenoEpsilon),
                        bool wave_speed_cache = false);

  /// AMR twin of System::add_external_riemann_block. The external flux is instantiated directly
  /// in the deferred AmrRuntime builder and therefore retains native reflux/halo execution.
  void add_external_riemann_block(const std::string& name, const std::string& so_path,
                                  const std::string& brick_id, const std::string& sha256,
                                  const std::string& limiter, const std::string& recon,
                                  const std::string& time, double gamma, int substeps, int stride,
                                  int expected_nvars, int expected_naux,
                                  const std::string& expected_model_identity,
                                  double positivity_floor = 0.0,
                                  double weno_epsilon = static_cast<double>(kWenoEpsilon));

  /// Install the exact prepared AMRTagging program resolved from the layout authority.
  /// This is the only tagging installation seam: the runtime never synthesizes a scalar
  /// threshold, component-zero default, or shared-potential fallback. The native tagger owns
  /// `min_cycles > 0` as accepted sparse state and persists it through AMR checkpoint/restart.
  void set_bootstrap_tagging(
      const std::vector<std::string>& leaf_subject_kinds,
      const std::vector<std::string>& leaf_subject_identities,
      const std::vector<std::string>& leaf_blocks, const std::vector<std::string>& leaf_variables,
      const std::vector<int>& leaf_field_component_indices, const std::vector<int>& leaf_ops,
      const std::vector<double>& leaf_thresholds, const std::vector<int>& leaf_stencil_indices,
      const std::vector<runtime::amr::PreparedTaggingProgram::Stencil>& stencils,
      const std::vector<std::int32_t>& refine_ops, const std::vector<std::int32_t>& refine_args,
      const std::vector<std::int32_t>& coarsen_ops, const std::vector<std::int32_t>& coarsen_args,
      int min_cycles, const std::string& equality_policy, const std::string& conflict_policy,
      const std::string& clock_identity, const std::string& provider_identity);
  /// Install one exact parent/child temporal relation per AMR transition.  These ratios are an
  /// independent execution authority and are never inferred from spatial refinement.
  void set_temporal_relations(const std::vector<std::int64_t>& numerators,
                              const std::vector<std::int64_t>& denominators,
                              const std::vector<std::string>& remainder_policies);

  /// Configures the default coarse elliptic field. This convenience uses the same provider registry
  /// and exact option contract as resolved named fields; it is not a second solver path. The
  /// right-hand side is always the sum of the blocks' elliptic bricks.
  /// @param rhs    "charge_density" | "composite" (same composed right-hand side as System)
  /// @param solver exact registered provider identity
  /// @param bc     "auto" | "periodic" | "dirichlet" | "neumann"
  /// @param wall   "none" | "circle" (circular conductive wall, requires wall_radius > 0)
  /// @param solver_options provider-owned typed options. An empty carrier requests the provider's
  ///        exact defaults; the AMR facade never interprets option keys.
  /// @throws std::runtime_error if rhs, provider, options, bc or wall violates the provider contract.
  void set_poisson(const std::string& rhs = "charge_density",
                   const std::string& solver = "geometric_mg", const std::string& bc = "auto",
                   const std::string& wall = "none", double wall_radius = 0.0,
                   const AmrFieldSolverOptions& solver_options = {});

  /// Install one fully resolved AMR field route. The registry key is the digest of its
  /// block-qualified provider identity. ``plan_identity`` independently commits the complete
  /// resolved semantics. Before lazy runtime materialization, the canonical ordered
  /// (slot, plan_identity) registry must agree exactly on every MPI rank. Duplicate slots are
  /// refused, including exact repeats.
  void set_field_solver_plan(const std::string& provider_slot, const std::string& plan_identity,
                             const std::string& provider_identity,
                             const std::string& output_owner_identity,
                             const std::string& output_block, const std::string& output_key,
                             const std::vector<std::string>& provider_identities,
                             const std::vector<std::string>& provider_blocks,
                             const std::vector<std::string>& provider_keys,
                             const std::vector<double>& provider_coefficients,
                             const std::string& solver,
                             const AmrFieldHierarchyPolicyAuthority& hierarchy_policy,
                             const AmrFieldSolverOptions& solver_options);
  /// Adds one native AMR field solver provider before binding. Builtins and extensions are resolved
  /// through the same per-system registry and must expose exact collective contracts.
  void register_field_solver_provider(std::shared_ptr<const AmrFieldSolverProvider> provider);
  /// Installs one authenticated external FieldTopology@2 + FieldSolver@2 pair as an AMR provider.
  /// The returned route is exactly ``provider_slot`` and is suitable for set_field_solver_plan.
  POPS_EXPORT std::string register_field_solver_provider(
      const std::string& provider_slot, runtime::field::PreparedFieldSolverSpec spec,
      std::shared_ptr<component::LoadedComponent> topology,
      std::shared_ptr<component::LoadedComponent> solver);
  /// Adds one native field-nullspace provider before binding. The selected route is resolved only
  /// after operator, boundary, topology and distribution facts have materialized.
  void register_field_nullspace_provider(std::shared_ptr<const FieldNullspaceProvider> provider);
  /// Select the provider for the principal field configured by set_poisson. The AMR facade retains
  /// only the opaque provider identity and its exact typed options; it never interprets a nullspace
  /// family or gauge name.
  void set_default_field_nullspace(const std::string& nullspace_provider_identity,
                                   const PreparedProviderOptions& options);
  /// Adds one hierarchy tensor-solver provider before binding. Compiled Programs resolve their
  /// opaque provider identity through this per-system registry; builtins and extensions use the
  /// same preparation protocol.
  void register_hierarchy_tensor_solver_provider(
      std::shared_ptr<const runtime::program::HierarchyTensorSolverProvider> provider);
  /// Collective generated-Program extension seam. Unlike the pre-build authoring registration
  /// above, this may run after materialization: it mutates only the provider registry, authenticates
  /// the exact declaration on every rank, and is idempotent for the same component declaration.
  POPS_EXPORT void register_program_hierarchy_tensor_solver_provider(
      std::shared_ptr<const runtime::program::HierarchyTensorSolverProvider> provider);
  /// Internal generated-Program seam. The returned immutable registry outlives every installed
  /// Program context because it is owned by this facade.
  POPS_EXPORT std::shared_ptr<const runtime::program::HierarchyTensorSolverProviderRegistry>
  hierarchy_tensor_solver_provider_registry() const;
  /// Exact read-only backend configuration retained by one resolved field plan.
  AmrFieldSolverConfiguration field_solver_configuration(const std::string& provider_slot) const;
  /// Install the resolved scalar reaction coefficient of one named screened field.
  void set_field_reaction(const std::string& provider_slot, double reaction);
  void set_field_topology_authority(const std::string& provider_slot,
                                    const std::string& provider_kind, const std::string& provenance,
                                    const std::string& topology_digest);
  std::vector<runtime::field::FieldTopologyReportRow> field_topology_report(
      const std::string& provider_slot) const;
  void set_field_boundary_plan(const std::string& provider_slot,
                               const std::vector<std::string>& kind,
                               const std::vector<double>& alpha, const std::vector<double>& beta,
                               const std::vector<double>& value);
  void set_field_boundary_dependencies(const std::string& provider_slot,
                                       const std::vector<std::string>& state_blocks,
                                       const std::vector<int>& state_components,
                                       const std::vector<std::string>& field_blocks,
                                       const std::vector<std::string>& field_keys,
                                       const std::vector<int>& field_components);
  POPS_EXPORT void set_field_boundary_kernel(const std::string& provider_slot,
                                             const CompiledFieldBoundaryKernel<Dim>& kernel);
  POPS_EXPORT void set_field_logical_timepoint(const std::string& provider_slot,
                                               const FieldLogicalTimePoint& point);
  POPS_EXPORT void set_field_boundary_parameters(const std::string& provider_slot,
                                                 const std::vector<double>& parameters);
  void set_field_newton_plan(const std::string& provider_slot, double tolerance, int max_iterations,
                             double linear_tolerance, int linear_max_iterations, int restart,
                             double armijo, double minimum_step);
  void set_field_nullspace(const std::string& provider_slot,
                           const std::string& nullspace_provider_identity,
                           const PreparedProviderOptions& options);

  /// Sets the initial density on the coarse level (component 0), ny*nx row-major.
  /// @param name cosmetic label (mono-block AMR: the density targets the single block).
  void set_density(const std::string& name, const std::vector<double>& rho);

  /// Sets the FULL INITIAL CONSERVATIVE STATE (all components) on the coarse level, then
  /// prolongs it to the fine levels at build (constant injection, like the density). @p U is flat
  /// component-major (c*ny*nx + j*nx + i) of size ncomp*ny*nx; ncomp == n_vars of the model (checked at
  /// build, where only Model::n_vars is known). Takes priority over set_density: allows starting the AMR
  /// from a full drift state (rho, rho*u, rho*v) instead of m=0. The conversion
  /// primitive -> conservative (rho_u = rho*u) is done on the Python side (the caller already supplies the
  /// conservative). Wired on native and compiled blocks, mono-block as well as multi-block: the full
  /// state is threaded to the deferred concrete builder, seeds the coarse, then is injected to the
  /// fine levels. In multi-block @p name indexes the target block.
  /// @throws std::runtime_error if the system is already built, if U is empty, or if its size
  ///         is not a multiple of ny*nx.
  void set_conservative_state(const std::string& name, const std::vector<double>& U);
  void begin_bootstrap_plan();
  bool bootstrap_next_level(int refinement_ratio);  ///< execute one resolved transition if tagged
  void commit_bootstrap_level();
  void rollback_bootstrap_level();
  void register_bootstrap_transfer_route(
      const std::string& identity, const std::vector<std::string>& subjects,
      const std::string& provider_identity, const std::string& space, const std::string& centering,
      const std::string& representation, const std::string& storage, const std::string& operation,
      const std::string& kernel, int order, const std::vector<int>& ghost_depth, int dimension,
      int refinement_ratio);
  void register_bootstrap_array(const std::string& subject, const std::string& centering, int ncomp,
                                Extent<Dim> shape, const std::vector<double>& values);
  void register_bootstrap_face_vector(const std::vector<std::string>& subjects);
  void bind_bootstrap_block_subject(const std::string& subject, const std::string& block);
  void register_analytic_constant(const std::string& subject, const std::string& block,
                                  const std::string& space, const std::string& centering,
                                  const std::vector<double>& components);
  void register_analytic_gaussian(const std::string& subject, const std::string& block,
                                  const RealVector<Dim>& center, double background,
                                  double amplitude, double inverse_width);
  void register_analytic_expression(const std::string& subject, const std::string& block,
                                    const std::string& space, const std::string& centering,
                                    const std::vector<std::vector<std::string>>& opcodes,
                                    const std::vector<std::vector<double>>& literals);
  std::int64_t bootstrap_analytic_reproject(const std::string& subject, int level);
  int apply_bootstrap_component_floor(const std::string& subject, int level, int component,
                                      double floor);
  std::int64_t recompute_bootstrap_field(const std::string& subject, const std::string& field_name);
  std::int64_t bootstrap_prolong_array(const std::string& subject, int level);
  void synchronize_bootstrap_state(const std::string& subject, int fine_level);
  std::vector<double> bootstrap_array_level(const std::string& subject, int level) const;
  void invalidate_bootstrap_cache(const std::string& subject, int level);
  std::vector<AmrPatch<Dim>> rebuild_bootstrap_topology_cache(const std::string& subject,
                                                              int level);
  std::uint64_t bootstrap_cache_epoch(const std::string& subject) const;

  /// Sets the magnetic field B_z(x, y) of the coarse level (ny*nx row-major), required by the Schur-condensed
  /// source stage (Lorentz term Omega = B_z). AMR counterpart of System::set_magnetic_field.
  /// Available on one- and multi-block AMR. The coarse field is published to every active level and
  /// re-applied after field solves and regrids by the native shared-aux runtime.
  /// @throws std::runtime_error if the system is already built or if bz is not of size ny*nx.
  void set_magnetic_field(const std::vector<double>& bz);

  /// Sets a model-NAMED aux field (ADC-291) at shared-channel component @p comp (>= kAuxNamedBase) from
  /// a coarse base-level field @p field (ny*nx row-major). AMR counterpart of
  /// System::set_aux_field_component: the field is STATIC (re-applied by the engine each update, so it
  /// survives a regrid) and reaches every level via the coarse->fine aux injection. The Python facade
  /// resolves the name to @p comp and reshapes the array. Mono-rank facade (same as set_density). @throws
  /// if the system is
  /// already built, if @p comp is reserved (< kAuxNamedBase), or if @p field is not of size ny*nx.
  void set_aux_field_component(int comp, const std::vector<double>& field);

  /// Declares a per-field aux HALO policy (ADC-369) for the NAMED component @p comp (>= kAuxNamedBase):
  /// @p bc_type is pops::BCType (Foextrap=1 / Dirichlet=2), @p value the Dirichlet boundary value. Seeded
  /// onto the engine at build and applied after the shared coarse aux fill (overriding only that
  /// component's physical-face ghosts; periodic faces keep their wrap). AMR counterpart of
  /// System::set_aux_field_halo_component. @throws on a reserved component or an unsupported type.
  void set_aux_field_halo_component(int comp, int bc_type, double value);

  /// @name Named multi-elliptic fields (ADC-428)
  /// A SECOND elliptic solve (beyond the default coarse Poisson) for a user-named field
  /// (m.elliptic_field("psi", rhs=..., aux=[...])) on the AMR hierarchy. AMR counterpart of
  /// System::register_elliptic_field / set_block_elliptic_field. The named field owns its RHS (a per-block
  /// brick), a DEDICATED coarse GeometricMG, and its OWN aux output components; AmrRuntime solves it each
  /// solve_fields and injects it to the fine levels, so a bare run() leaves it SOLVED. The default Poisson
  /// path is untouched / bit-identical. The field is registered on the same AmrRuntime engine used for
  /// every block count. POPS_EXPORT: resolved by the generated AMR .so / native loader across the dlopen
  /// boundary, like set_compiled_block.
  /// @{
  /// Registers named @p field's aux output components (where its solved phi / centered grad land). Called
  /// by the native AMR loader for each m.elliptic_field. @p gx_comp / @p gy_comp < 0 => only phi is
  /// written (both must equal -1); @p gradient_sign is exactly -1 or +1 and scales both derivatives.
  /// @throws if the system is already built or the output contract is malformed.
  POPS_EXPORT void register_elliptic_field(const std::string& block_name,
                                           const std::string& provider_key, int phi_comp,
                                           int gx_comp, int gy_comp, int gradient_sign);
  /// Attaches named @p field's RHS closure (rhs += elliptic_field_rhs(U)) to block @p block_name. Called
  /// by the native AMR loader (make_poisson_rhs of the per-field brick). @throws if the system is already
  /// built or the block is unknown.
  POPS_EXPORT void set_block_elliptic_field(
      const std::string& block_name, const std::string& field,
      std::function<void(const MultiFab<Dim>&, MultiFab<Dim>&)> rhs);
  /// Solved potential of named @p field on the COARSE level, ny*nx row-major (read-back). Solves the
  /// hierarchy fields if needed (so it is current even before any step), then reads the field's phi
  /// component. AMR counterpart of System::aux_field_component for a named elliptic field. @throws if the
  /// field is unregistered.
  std::vector<double> named_field_values(const std::string& field);
  std::vector<std::string> field_provider_slots() const;
  int field_provider_levels(const std::string& provider_slot);
  void set_field_potential(const std::string& provider_slot, const std::vector<double>& phi);
  void set_field_potential_level(const std::string& provider_slot, int level,
                                 const std::vector<double>& phi);
  std::vector<double> field_potential_global(const std::string& provider_slot);
  std::vector<double> field_potential_level_global(const std::string& provider_slot, int level);
  /// Exact rank-local valid-cell pieces for one qualified field provider.  The returned metadata
  /// explicitly marks replicated level-zero ownership so output modes never infer it from box counts.
  std::vector<OutputPiece<Dim>> output_field_local_pieces(const std::string& provider_slot,
                                                          int level);
  std::vector<OutputPiece<Dim>> output_field_root_pieces(const ObserverMpiLane& lane,
                                                         const std::string& provider_slot,
                                                         int level);
  /// Transaction bracket used by the accepted-state reader after complete payload preflight. Every
  /// hierarchy,
  /// state, aux, field warm-start, history and clock mutation is rolled back if any restore step fails.
  void begin_restart_transaction();
  void commit_restart_transaction();
  void rollback_restart_transaction();
  /// Force exactly one artifact-owned scientific regrid inside an active restart transaction.
  /// The exact recorded accepted state must already have been restored and authenticated.
  void preflight_regrid_on_restart();
  void regrid_on_restart();
  int checkpoint_regrid_count() const;
  std::uint64_t checkpoint_topology_epoch() const;
  void restore_checkpoint_counters(int regrid_count, std::uint64_t topology_epoch);
  std::vector<std::vector<std::string>> checkpoint_temporal_relations() const;
  /// Canonical rows for every required bootstrap transfer route: subject, operation, route identity,
  /// provider, kernel, descriptor fields.  The sealed checkpoint compares these rows byte-for-byte.
  std::vector<std::vector<std::string>> checkpoint_transfer_routes() const;
  /// @}

  /// Registers an inter-species COUPLED SOURCE (compiled pops.dsl.CoupledSource, flat bytecode ABI
  /// P5), refined counterpart of System::add_coupled_source but on the SHARED AMR hierarchy.
  /// Registration stores the typed operator only: the installed Program owns its temporal placement,
  /// applies it to candidate states level by level, and then performs the authored synchronization.
  /// The coupling is baked into a device-clean stack machine (CoupledSourceKernel): NO per-cell Python
  /// callback in the hot path. MULTI-BLOCK only (>= 2 add_block: the coupling reads/writes
  /// SEVERAL named blocks). Must be called BEFORE the first step (the runtime engine is built
  /// at lazy build; the source is injected into it).
  ///
  /// CONSERVATION: an add_pair construction (a term +expr on a block, -expr exactly on the other,
  /// SAME cell) makes the sum of the two blocks conserved PER CELL (and globally) to machine
  /// precision. The engine does NOT IMPOSE it (an ionization creating an e/i pair is legal): it is a
  /// property of the constructed coupling (verify_conservation on the DSL side checks it symbolically).
  ///
  /// @throws std::runtime_error if called in mono-block, if the system is already built, or if the
  ///         shape of the bytecode / a role / a block is invalid (same guards as System).
  /// @param prog      bytecode description of the coupling grouped in a POD (ADC-214; cf.
  ///                  CoupledSourceProgram; parity with System::add_coupled_source): in_blocks /
  ///                  in_roles / consts / out_blocks / out_roles + prog_ops / prog_args / prog_lens
  ///                  (stack machine) + freq_prog_ops / freq_prog_args (PER-CELL frequency mu(U)
  ///                  optional; EMPTY = constant frequency only, bit-identical; non-empty:
  ///                  evaluated on the COARSE LEVEL of the input blocks at each step_cfl, MAX +
  ///                  all_reduce_max, bound dt <= cfl / max(mu) on the coarse, not the patches).
  /// @param frequency CONSTANT declared frequency mu [1/s] of the coupling (wave 3): bound
  ///                  dt <= cfl/mu on the macro-step of step_cfl; <= 0 (default) = no bound.
  /// @param label     name of the coupling (reason "coupled_source:<label>" of last_dt_bound).
  void add_coupled_source(const CoupledSourceProgram& prog, double frequency = 0.0,
                          const std::string& label = "coupled_source");

  /// Registers a TYPED coupling operator (ADC-595, parity with System::add_coupling_operator): the
  /// same coupled-source program PLUS its declared conservation contract and frequency bound. The
  /// declared contract is VALIDATED at registration (host, fail-loud) against the actual output terms,
  /// then the program is lowered through the SAME add_coupled_source path (bit-identical numerics), and
  /// the declared contract is recorded for coupled_operators(). An empty (unchecked) contract is
  /// equivalent to add_coupled_source.
  void add_coupling_operator(const CouplingOperator& op);

  /// Read-only view of the registered coupling operators (ADC-595, parity with System): label plus the
  /// declared conservation / frequency contracts, in registration order, so a Program or a runtime
  /// report enumerates the AMR couplings as typed operators. A raw add_coupled_source registers an
  /// "unchecked" entry (empty contract). Empty until the first coupling is added.
  const std::vector<CouplingOperatorView>& coupled_operators() const;

  void step(double dt);  ///< one AMR macro-step (periodic regrid included)
  void advance(double dt, int nsteps);
  void begin_step_transaction();
  void commit_step_transaction();
  void finalize_step_transaction();
  void rollback_step_transaction();
  /// Internal rollback authority used by the installed Program context.  A Program nested in a
  /// public AmrSystem step borrows the facade's accepted image instead of taking a second full engine
  /// snapshot; a direct C++ Program context remains autonomous.
  POPS_EXPORT bool has_active_step_transaction() const noexcept;
  POPS_EXPORT void restore_active_step_transaction_for_program();
  /// Volume-weighted L2 norm of each block's accepted AMR macro-step change. Collective and valid
  /// while the retained outer transaction snapshot still owns U^n.
  POPS_EXPORT std::map<std::string, double> step_change_l2() const;
  /// Advances at dt = cfl * coarse_dx / max wave speed. @return the dt used.
  double step_cfl(double cfl, double speed_floor = static_cast<double>(kCflSpeedFloor),
                  double max_dt = std::numeric_limits<double>::infinity(), double min_dt = 0.0);

  /// @name Compiled time-program install seam on the AMR hierarchy (epic ADC-511 / ADC-508, Spec 6)
  /// AMR counterpart of System::install_program: load a generated problem.so and install its compiled
  /// time Program over the AMR hierarchy. Mirrors the System seam (install_program_step registers the
  /// macro-step body; the cadence + per-block RuntimeParams stores live HERE on the Impl, NOT in the
  /// .so closure, so a value change reaches the captured context and a later checkpoint can reach
  /// them). A generated AMR Program .so resolves these POPS_EXPORT seams from the globally promoted
  /// host while the generated package remains RTLD_LOCAL, exactly like set_compiled_block on the
  /// native AMR loader.
  /// @{
  /// Install the mandatory macro-step body. AmrSystem::step, advance and step_cfl reject before lazy
  /// hierarchy construction or any other mutation while it is absent. An empty std::function is
  /// rejected: there is no public temporal route that silently clears the whole-system Program.
  /// POPS_EXPORT: the generated AMR Program .so resolves it across the dlopen boundary, like
  /// set_compiled_block. The closure executes the normalized ProgramGraph on the hierarchy through
  /// an AmrProgramContext (the AMR counterpart of ProgramContext).
  POPS_EXPORT void install_program_step(std::function<void(double)> step);
  /// Install the companion callback that republishes Program-owned accepted clocks/history whenever
  /// explicit bootstrap commits a hierarchy level. Generated artifacts own this seam; direct
  /// low-level steps may omit it because they have no authenticated checkpoint context.
  POPS_EXPORT void install_program_hierarchy_refresh(std::function<void()> refresh);
  /// Install the artifact-owned restart preflight, transform and forced rollback-resynchronization
  /// hooks.
  POPS_EXPORT void install_program_restart_hooks(std::function<void()> preflight,
                                                 std::function<void()> regrid,
                                                 std::function<void()> resync);
  /// Set the compiled-Program macro-step cadence (parity with System::set_program_cadence, ADC-411):
  /// GLOBAL @p substeps and @p stride around the installed program closure. @p substeps subdivides each
  /// effective step into @p substeps program closure calls; @p stride runs the program once per @p
  /// stride macro-steps (hold-then-catch-up). Both must be >= 1 (throws std::invalid_argument).
  /// Default 1/1 -> a single program closure call per macro-step. Kept SEPARATE from install_program so
  /// the generated .so ABI is untouched (the cadence is runtime metadata).
  POPS_EXPORT void set_program_cadence(int substeps, int stride);
  /// Installed GLOBAL macro-step cadence (ADC-594, parity System): the current @c substeps / @c stride
  /// the compiled Program runs at (default 1/1 with no cadence set). Const, side-effect-free -- the
  /// structured ProgramRuntimeReport reads them; there was no Python-visible getter before.
  POPS_EXPORT int program_substeps() const;
  POPS_EXPORT int program_stride() const;
  /// Exact duration, accepted public-step count and physical start currently held by the GLOBAL
  /// Program stride window. All are zero at a stride boundary and form mandatory checkpoint state.
  POPS_EXPORT double program_cadence_window_dt() const;
  POPS_EXPORT int program_cadence_window_steps() const;
  POPS_EXPORT double program_cadence_window_start_time() const;
  /// Exact accepted Program interval provenance. Zero means no Program invocation has been accepted.
  POPS_EXPORT double program_last_dt() const;
  /// Stage the exact held-window image before set_clock during strict restart. The image must match
  /// the exact accepted (@p accepted_time, @p macro_step) cursor, @p accepted_last_dt and installed
  /// stride; malformed, missing or mismatched state is rejected without mutating accepted state.
  POPS_EXPORT void restore_program_cadence_window(double accumulated_dt, int held_steps,
                                                  double window_start_time, double accepted_last_dt,
                                                  double accepted_time, int macro_step);
  /// Install the program-index -> AMR-block-index map (entry p = the AMR block index of Program block
  /// p), built by install_program after matching the .so's block names to the instantiated AMR blocks
  /// BY NAME (Spec 3 criterion 23, ADC-457). Empty clears it and is never an implicit positional
  /// identity. Read by the AmrProgramContext to resolve a Program block index to the name-matched block.
  POPS_EXPORT void set_program_block_map(const std::vector<int>& prog_to_sys);
  /// The installed program-index -> AMR-block-index map. Empty means no authenticated mapping and is
  /// rejected by AmrProgramContext.
  POPS_EXPORT const std::vector<int>& program_block_map() const;
  /// Load a generated problem.so and install its compiled time Program on the AMR hierarchy. dlopens
  /// @p so_path, checks its ABI key against this module (fail-loud on mismatch), runs the section-24
  /// install-time requirement validation (aux / solver / block instance, verbatim spec messages), binds
  /// the Program's blocks to the AMR blocks BY NAME, seeds each block's RuntimeParams from the .so
  /// pops_program_param_* metadata, then calls the .so's pops_install_program_amr(this), whose shared
  /// facade factory selects the hierarchy provider and installs the macro-step closure. Mirrors
  /// add_native_block and System::install_program; the .so stays loaded for the process lifetime.
  POPS_EXPORT void install_program(const std::string& so_path);
  /// IR hash of the installed compiled Program (the string returned by the .so's pops_program_hash), or
  /// "" if no program is installed. Parity with System::installed_program_hash (checkpoint guard).
  POPS_EXPORT std::string installed_program_hash() const;
  /// The last macro-step dt handed to the installed Program (ADC-631): the AmrProgramContext reads it
  /// so a pre-commit history sample records its outgoing interval (variable-dt replay). POPS_EXPORT
  /// for the dlopen boundary (the generated AMR Program .so reads it via the AmrProgramContext).
  /// Authenticated accepted-state image owned by the compiled AMR Program context.  This is distinct
  /// from the dense field/history arrays: it preserves exact level clocks, qualified history-slot
  /// identities and lagged effective-flux publications required for conservative multistep restart.
  POPS_EXPORT std::vector<std::uint8_t> program_accepted_state() const;
  /// Copy the same authenticated image into caller-owned storage.  The Program attempt transaction
  /// keeps this storage resident so a stable retry snapshots bytes without allocating a temporary
  /// vector on every macro-step; the returned-by-value accessor remains the public convenience API.
  POPS_EXPORT void copy_program_accepted_state_into(std::vector<std::uint8_t>& state) const;
  /// Replace the accepted image during strict restart.  Each replacement advances a revision observed
  /// by the persistent AmrProgramContext before its next attempt; no stale context state is reused.
  POPS_EXPORT void restore_program_accepted_state(const std::vector<std::uint8_t>& state);
  /// Strict checkpoint counterpart: authenticate the complete accepted image and its runtime-owned
  /// tagging payload before atomically publishing either. A rejected payload changes neither bytes,
  /// revision nor the live AMR hysteresis state.
  POPS_EXPORT void restore_checkpoint_accepted_state(const std::vector<std::uint8_t>& state);
  /// Validate the exact history registry encoded by @p state and materialize its native per-level
  /// rings on the already rebuilt restart hierarchy. This is a transactional restart seam: it never
  /// advances the Program and refuses any name/depth/component/owner mismatch before allocation.
  POPS_EXPORT void materialize_program_restart_histories(const std::vector<std::uint8_t>& state,
                                                         const std::vector<std::string>& names,
                                                         const std::vector<int>& depths,
                                                         const std::vector<int>& ncomps);
  POPS_EXPORT std::uint64_t program_accepted_state_revision() const;
  /// Human/audit-readable qualification rows decoded from the same accepted image persisted as bytes.
  POPS_EXPORT std::vector<std::vector<std::string>> program_accepted_state_manifest() const;
  POPS_EXPORT std::vector<std::vector<std::string>> program_clock_manifest() const;
  /// Accepted temporal-partition provider, synchronization tick and per-rung cell counts. The rows
  /// are decoded from the same opaque image used by strict restart, never a capability ledger.
  POPS_EXPORT std::vector<std::vector<std::string>> program_temporal_partition_manifest() const;
  POPS_EXPORT std::vector<std::vector<std::string>> program_flux_ledger_manifest() const;
  POPS_EXPORT std::vector<std::vector<std::string>> program_interface_flux_ledger_manifest() const;
  POPS_EXPORT std::vector<std::vector<std::string>> program_sync_manifest() const;

  /// @name Runtime freeze lifecycle (ADC-592, parity with System)
  /// Assembly mutable BEFORE bind, composition FROZEN once pops.bind completes. mark_bound() is
  /// called LAST by the Python bind flow (after every install call), so the install sequence itself
  /// never trips the structural-setter guards. NOTE: 'bound' (bind completed) is DISTINCT from the
  /// lazy 'built' materialization (bind runs BEFORE the first step/mass/density triggers ensure_built),
  /// so the existing 'already built' messages of the lazy path are untouched; the structural guards
  /// now also refuse a call once bound_ is set, with the bind-vocabulary message.
  /// @{
  /// Mark the composition as bound (frozen): every structural setter then rejects with a precise
  /// error. Runtime-data setters (set_density / set_conservative_state on the base level BEFORE a step
  /// / set_program_params / set_clock) that DATA-write stay allowed. A second mark_bound() throws.
  void mark_bound();
  /// The runtime lifecycle state: "assembling" (not bound), "bound" (mark_bound() ran, no macro-step),
  /// "running" (bound AND macro_step() > 0). Parity with System::lifecycle_state.
  std::string lifecycle_state() const;
  /// @}
  /// @name Compiled-Program RUNTIME parameters on AMR (epic ADC-511 / ADC-508, parity with ADC-510)
  /// Per-PROGRAM-block RuntimeParams of a compiled time Program whose physics reads a
  /// dsl.Param(..., kind="runtime"), owned HERE so set_program_params changes it at run time WITHOUT
  /// recompiling (the same no-recompile contract as System). install_program seeds each block's defaults
  /// from the .so pops_program_param_* metadata. The lowered kernels read the CURRENT value via the
  /// AmrProgramContext.
  /// @{
  /// Overwrite block @p prog_block's RuntimeParams with @p values (the COMPLETE block, sorted-name order
  /// matching the .so pops_program_param_* metadata). @p prog_block is the PROGRAM block index. @throws
  /// std::out_of_range if the block was not seeded by a runtime-param Program, std::runtime_error on a
  /// size mismatch. Effect on the next step.
  POPS_EXPORT void set_program_params(int prog_block, const std::vector<double>& values);
  /// Block @p prog_block's CURRENT RuntimeParams (a device-clean by-value copy). An UNSEEDED block
  /// returns a default-constructed RuntimeParams (count 0). Read by the AmrProgramContext.
  POPS_EXPORT RuntimeParams program_params(int prog_block) const;
  /// Seed block @p prog_block's RuntimeParams to its DECLARATION defaults (@p count values, the .so
  /// pops_program_param_default metadata). Called by install_program once per runtime-param Program
  /// block; a later set_program_params overwrites only the supplied values. Idempotent.
  POPS_EXPORT void seed_program_params(int prog_block, const std::vector<double>& defaults);
  /// @}
  /// The built AMR spatial engine (the AmrRuntime the AmrProgramContext driver wraps), or nullptr
  /// before the lazy build. install_program forces the build so the .so's pops_install_program_amr
  /// receives a live engine. POPS_EXPORT: the generated AMR Program .so resolves it across the dlopen
  /// boundary.
  POPS_EXPORT runtime::amr::AmrRuntime<Dim>* engine() const;
  /// Compatibility inspection seam. Once built, every resolved AMR system uses AmrRuntime, so this
  /// returns true; before build engine() remains null. POPS_EXPORT for dlopen-boundary parity.
  POPS_EXPORT bool uses_runtime_engine() const;
  /// The facade-owned Profiler (the AmrProgramContext forwards count_kernel / profile_record to it).
  /// POPS_EXPORT for the dlopen boundary. Disabled by default -> zero hot-path cost.
  POPS_EXPORT runtime::program::Profiler& profiler_handle();
  /// Record a runtime Scalar diagnostic under @p name (the AmrProgramContext's record_scalar seam),
  /// retrievable via program_diagnostic / program_diagnostics. A pure side effect (inspection / logging).
  POPS_EXPORT void record_program_diagnostic(const std::string& name, double value);
  /// The recorded diagnostic @p name (0 if absent) / the whole map. Exposed to Python for inspection.
  POPS_EXPORT double program_diagnostic(const std::string& name) const;
  POPS_EXPORT std::map<std::string, double> program_diagnostics() const;
  /// Five current-attempt scalars for one typed balance route. RuntimeInstance calls this only
  /// inside its active outer accepted-step transaction; missing/stale/non-finite evidence fails.
  POPS_EXPORT std::map<std::string, double> accepted_balance_terms(const std::string& route) const;
  /// The same accepted route with selected attempt-local native reflux/projection producers.
  POPS_EXPORT std::map<std::string, double> selected_accepted_balance_terms(
      const std::string& route, const std::string& block, int component,
      const std::vector<int>& levels, const std::vector<std::string>& automatic_terms) const;
  POPS_EXPORT void begin_step_projection_report();
  POPS_EXPORT void note_step_projection(const std::string& name);
  POPS_EXPORT std::vector<std::string> consume_step_projections();
  /// LEVEL-COMPOSITE collective reduction over a named block, the AMR counterpart of
  /// System::reduce_component the diagnostics driver drives (ADC-542). @p kind is per-component
  /// "sum" / "min" / "max" / "abs_sum" / "sum_sq" / "abs_max", or the full-state "*_all" variants.
  /// @p levels is the exact strictly-increasing level selection; empty is the low-level C++
  /// all-level convention. Volume-weighted sums and extrema mask coarser selected cells covered by
  /// the next selected finer level. Multi-block and single-block both remain native/Kokkos.
  POPS_EXPORT double composite_reduce(const std::string& block, const std::string& kind, int comp,
                                      const std::vector<int>& levels = {}) const;
  /// Composite integral/reduction of one qualified native field provider.  This route is available
  /// only after the AMR runtime has been built because that engine owns the typed provider hierarchy.
  POPS_EXPORT double composite_reduce_field(const std::string& provider_slot,
                                            const std::string& kind, int comp,
                                            const std::vector<int>& levels = {});
  /// @}
  /// @}

  Extent<Dim> spatial_shape() const;
  /// Generated Program shared libraries read the accepted clock through the flat loader ABI.
  POPS_EXPORT double time() const;
  /// ACCEPTED macro-step counter (0-indexed; incremented by step / advance / step_cfl), parity with
  /// System::macro_step. Required for checkpoint/restart because Program schedules and regrid cadence
  /// depend on accepted-step phase, not only on physical time. Persisted by accepted-state restart.
  /// POPS_EXPORT: the AmrProgramContext (a generated AMR Program .so) reads it across the dlopen
  /// boundary for the head-of-step regrid cadence, like the other program seam accessors (ADC-508).
  POPS_EXPORT int macro_step() const;
  /// RESTORES the AMR clock (t, macro_step) -- parity with System::set_clock. Sets the time AND the
  /// macro-step counter propagated to Program schedules and regrid cadence. A mid-window cursor also
  /// requires an immediately preceding restore_program_cadence_window; neither the start nor
  /// accumulated variable-dt duration can be inferred from the clock. @throws on invalid state.
  POPS_EXPORT void set_clock(double t, int macro_step);

  /// @name AMR / MPI profiling (Spec 5 sec.12.5, ADC-479 criterion 43)
  /// Per-phase wall-clock timing of the AMR runtime: the engine times its non-numeric phases --
  /// "regrid" (rebuild the patch hierarchy), "fill_boundary" (the cross-rank ghost halo exchange),
  /// "average_down" (restrict fine onto coarse) -- plus integer counters ("regrid" / "fill_boundary"
  /// per-run counts, and under MPI np>1 "mpi_reductions" / "mpi_messages"). Disabled by default (no
  /// hot-path cost when off, parity with System). enable_profiling() then step()/step_cfl() over a
  /// run where a regrid fires (regrid_every set) then profile_report() returns the table; the typed
  /// PerformanceSummary.by_amr_mpi() view surfaces it. Per-rank (no cross-rank reduction of the
  /// report). The same runtime owns these phases for every block count.
  /// @{
  void enable_profiling();
  void disable_profiling();
  bool is_profiling() const;
  void reset_profiling();
  std::string profile_report() const;
  /// @}
  int n_blocks() const;  ///< number of blocks on the shared AmrRuntime engine
  /// Names of the blocks in add order (parity with System::block_names): the IO facade iterates over them
  /// to write EACH block by its name (an empty name -> block 0, historical mono-block compat).
  std::vector<std::string> block_names() const;
  /// Structured report of effective numerical, solver and physical options currently configured.
  EffectiveOptionsReport effective_options_report() const;
  int n_patches();  ///< number of current fine patches (of the shared hierarchy)
  /// Index-space signatures of the current fine patches: one AmrPatch<Dim> (level, ilo, jlo, ihi, jhi) per
  /// fine box, for ALL fine levels (level >= 1). INCLUSIVE corners in the index space of the
  /// level (each base-axis count shifted by ``level``, ratio 2). SAME source as n_patches()
  /// (the GLOBAL fine
  /// BoxArray, all boxes/all ranks -> rank-independent, MPI-safe, zero communication). It is a
  /// QUERY (between steps): read-only of the already-stored boxes, NO hot-path cost. The
  /// conversion to exact physical x/y bounds is done on the Python side. Forces the lazy
  /// build (ensure_built) like n_patches()/mass()/density().
  std::vector<AmrPatch<Dim>> patch_boxes();
  /// COARSE-level (base) box counts, MPI ownership diagnostic (ADC-319). coarse_local_boxes() = number
  /// of base boxes OWNED by this rank (level-0 MultiFab local_size()); coarse_total_boxes() = total base
  /// boxes across all ranks (BoxArray size, identical on every rank). With distribute_coarse=true the
  /// base is split into several boxes spread round-robin, so local < total per rank and the coarse
  /// transport distributes (MPI strong-scaling); a single-box or replicated base gives local == total on
  /// every rank. coarse_local_boxes() is rank-dependent, coarse_total_boxes() is rank-independent.
  /// Forces the lazy build (ensure_built) like n_patches()/mass()/density().
  int coarse_local_boxes();
  int coarse_total_boxes();

  /// AMR CHECKPOINT / RESTART (ADC-65 single-block single-rank; ADC-509 multi-block + np>1):
  /// per-level STATE accessors + hierarchy imposition for a BIT-IDENTICAL resumption (cf.
  /// AmrSystem.checkpoint/restart on the Python side). Unqualified level_state is a one-block
  /// convenience over AmrRuntime; multi-block callers use the qualified block_level_state variants.
  /// Potential, hierarchy and aux remain shared. The _global variants all_reduce_sum
  /// the per-rank fabs so a np>1 checkpoint gathers onto rank 0 (mono-rank: identity, bit-identical).
  /// Force the lazy build (ensure_built) like patch_boxes()/mass(). @p k: level (0 = coarse, >= 1 = fine).
  int n_levels();             ///< number of levels of the hierarchy (>= 1; mono OR multi-block)
  int max_levels();           ///< resolved maximum active hierarchy depth
  int configured_n_levels();  ///< immutable resolved hierarchy capacity
  int n_vars();  ///< number of conserved components (MONO-BLOCK; multi-block: block_n_vars)
  /// FULL conservative state of level @p k, flat component-major c*nf*nf + j*nf + i (nf = n << k;
  /// zeros outside the patches at the fine level -- only the patch interior is defined). MONO-BLOCK.
  std::vector<double> level_state(int k);
  std::vector<double> level_state_global(int k);  ///< one-block, np>1 gather (all ranks call)
  void set_level_state(int k,
                       const std::vector<double>& s);  ///< restores the state of level @p k (as is)
  /// Potential phi of level @p k, flat nf*nf row-major. Level 0 = warm-start of the multigrid
  /// (bit-identical resumption); level >= 1 = aux comp 0 (recomputed at update). Shared across blocks;
  /// the _global variant gathers under np>1.
  std::vector<double> level_potential(int k);
  std::vector<double> level_potential_global(int k);              ///< np>1 gather (all ranks call)
  void set_level_potential(int k, const std::vector<double>& p);  ///< restores phi of level @p k
  /// Imposes the SAVED fine hierarchy (at restart) instead of Berger-Rigoutsos clustering: @p boxes
  /// are the patch_boxes() signatures of the checkpoint. Serial convenience over rebuild_hierarchy.
  void set_hierarchy(const std::vector<AmrPatch<Dim>>& boxes);

  /// Impose a mid-run hierarchy from a v3 checkpoint (ADC-542): @p boxes are ALL the
  /// checkpoint patch boxes (level tagged, level 0 implicit), @p owner_ranks the per-box owner rank
  /// aligned with @p boxes. Routes to AmrRuntime::rebuild_hierarchy (all levels rebuilt, reusing regrid
  /// R6/R7). The v3 restart calls this so restartable=True works under ACTIVE regridding.
  void rebuild_hierarchy(const std::vector<AmrPatch<Dim>>& boxes,
                         const std::vector<int>& owner_ranks);

  /// Re-evaluate ownership for the exact recorded fine-level boxes under the load-balance authority
  /// prepared at bind. This is a collective, non-mutating restart seam: geometry and box ordering
  /// stay unchanged while the returned owner list is aligned with @p boxes for the current
  /// communicator size.
  std::vector<int> rematerialize_hierarchy_ownership(const std::vector<AmrPatch<Dim>>& boxes);

  /// Merge exact source-rank Program images and return this rank's image under the current
  /// communicator ownership. Both ownership tables are indexed [level][global patch].
  std::vector<std::uint8_t> rematerialize_program_accepted_state(
      const std::vector<std::vector<std::uint8_t>>& source_states,
      const std::vector<std::vector<int>>& source_level_owners,
      const std::vector<std::vector<int>>& target_level_owners);

  /// Per-block per-level checkpoint accessors (ADC-509). The AmrRuntime engine shares the
  /// layout AND the aux across blocks, so the per-level STATE is read/restored PER BLOCK (by NAME)
  /// while phi stays shared (level_potential above). @p name indexes the block (block_names()); @p k:
  /// level. The _global variant all_reduce_sum the per-rank fabs (np>1 gather, all ranks call); the
  /// shared hierarchy is the deterministic frozen central patch (regrid_every==0), reproduced at
  /// restart by replaying the same composition. @throws if @p name / @p k is out of bounds.
  int block_n_vars(const std::string& name);  ///< conserved components of the named block
  std::vector<double> block_level_state(const std::string& name, int k);
  std::vector<double> block_level_state_global(const std::string& name,
                                               int k);  ///< np>1 gather (all ranks call)
  void set_block_level_state(const std::string& name, int k, const std::vector<double>& s);
  /// Unified scientific-output state accessor. Unlike the checkpoint names above, this routes an
  /// exactly named block through the shared runtime and returns compact native valid-cell pieces
  /// without allocating a global level buffer.
  std::vector<OutputPiece<Dim>> output_state_local_pieces(const std::string& name, int k);
  std::vector<AmrPatch<Dim>> output_geometry_boxes();
  std::vector<OutputPiece<Dim>> output_state_root_pieces(const ObserverMpiLane& lane,
                                                         const std::string& name, int k);
  /// Owner rank per box of level @p k (the shared ranked ownership plan), aligned with the
  /// level-@p k rows of patch_boxes(). The v3 checkpoint (ADC-542) serializes it so a restart
  /// reproduces the LOCAL-fab iteration order.
  std::vector<int> level_owner_ranks(int k);
  /// FULL shared aux of level @p k (ALL components, flat c*nf*nf+j*nf+i; _global = np>1 gather,
  /// COLLECTIVE) + the owner-rank restore -- the v3 checkpoint aux payload (ADC-542).
  std::vector<double> level_aux_flat(int k);
  std::vector<double> level_aux_flat_global(int k);
  void set_level_aux_flat(int k, const std::vector<double>& v);

  /// @name Multistep history-ring checkpoint / replay (ADC-631, Uniform System seam names)
  /// The compiled-Program AMR route carries per-level `keep_history` / `T.prev` ring slots on the
  /// AmrRuntime engine (remapped through regrid). These wrappers expose the SAME seam names as System
  /// so the shared _system_io_history.py serialize/restore is reused verbatim: history_global returns
  /// the per-level slices concatenated into ONE flat buffer (level axis hidden, parity level_aux_flat),
  /// restore_history scatters it back per level, rebuild_history_slots replays the policy-recomputed
  /// slots by re-stepping the installed Program.
  /// @{
  std::vector<std::string> history_names() const;
  int history_depth(const std::string& name) const;
  int history_ncomp(const std::string& name) const;
  bool history_initialized(const std::string& name) const;
  int history_fill_count(const std::string& name) const;
  void set_history_initialized(const std::string& name, bool initialized);
  void restore_history_fill_count(const std::string& name, int fill_count);
  std::vector<double> history_global(const std::string& name, int slot) const;
  void restore_history(const std::string& name, int slot, const std::vector<double>& values);
  double history_slot_dt(const std::string& name, int slot) const;
  void restore_history_slot_dt(const std::string& name, int slot, double dt);
  int rebuild_history_slots(const std::string& name, const std::vector<int>& stored_slots);
  /// The sorted macro-step cursors at which the LAST rebuild_history_slots fired an in-window regrid
  /// (ADC-635). The accepted-state reader asserts it against the checkpoint's recorded schedule
  /// fingerprint; empty after a Dense / clean-window / no-regrid replay.
  std::vector<int> last_replay_regrid_steps() const;
  /// @}

  double mass();  ///< mass of the 1st block on the coarse (conserved at reflux)
  double mass(
      const std::string& name);   ///< mass of the named block on the coarse (conserved PER BLOCK)
  std::vector<double> density();  ///< coarse density of the 1st block, ny*nx row-major
  std::vector<double> density(const std::string& name);  ///< named-block density, ny*nx row-major
  /// Electrostatic potential phi of the COARSE LEVEL (base), ny*nx row-major. Level 0 covers
  /// the whole domain: enough to sample a median circle (azimuthal FFT), SAME
  /// observable as System::potential() on a single-level mesh. Solves the coarse Poisson if
  /// needed (cf. System::potential / ensure_elliptic), so current value even before any step.
  /// MULTI-BLOCK: phi results from the SYSTEM Poisson (Sum_b q_b n_b co-located); shared by all
  /// the blocks (single aux). The block name therefore does not intervene.
  std::vector<double> potential();

 private:
  template <int ContextDim, class MemorySpace>
  friend class runtime::program::AmrProgramContext;
  /// Dedicated generated-Program sink for one validated, attempt-local balance term. It remains
  /// private to AmrProgramContext and is deliberately absent from Python bindings.
  POPS_EXPORT void record_program_balance_term(const std::string& route, const std::string& term,
                                               double value);
  POPS_EXPORT bool program_balance_consumer_is_due(const std::string& contract,
                                                   const std::string& route, int every_n) const;
  POPS_EXPORT runtime::program::ProgramRuntimeState<Dim>& program_runtime_state_();
  /// Read-only compiled-artifact capability check; artifact authority installation is private to
  /// AmrSystem::install_program and cannot be injected through the public facade.
  POPS_EXPORT bool program_owns_operator_authority(
      const std::array<std::uint64_t, 4>& authority) const noexcept;
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace pops
