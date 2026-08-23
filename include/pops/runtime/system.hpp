#pragma once

#include <limits>

#include <pops/core/state/variables.hpp>  // VariableSet (role-bearing descriptor carried by each block)
#include <pops/core/identity/prepared_provider_options.hpp>
#include <pops/coupling/source/coupling_operator.hpp>  // CouplingOperator / CouplingOperatorView (typed contract, ADC-595)
#include <pops/diagnostics/runtime_diagnostics.hpp>
#include <pops/numerics/nonlinear/newton_options.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/numerics/nonlinear/prepared_variable_recovery.hpp>
#include <pops/mesh/boundary/prepared_hyperbolic_boundary.hpp>
#include <pops/mesh/boundary/prepared_boundary_component.hpp>
#include <pops/runtime/export.hpp>  // POPS_EXPORT (methods resolved by the native loader through dlopen)
#include <pops/runtime/facade_options.hpp>  // CoupledSourceProgram (facade POD, ADC-214)
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/config/runtime_params.hpp>  // RuntimeParams (compiled-Program runtime params, ADC-510)
#include <pops/runtime/config/spatial_domain.hpp>
#include <pops/runtime/numerical_defaults.hpp>
#include <pops/runtime/output_piece.hpp>
#include <pops/runtime/recovery/uniform_recovery_consumer.hpp>
#include <pops/runtime/system/auxiliary_checkpoint.hpp>
#include <pops/runtime/system/derived_aux_provider.hpp>
#include <pops/runtime/system/system_block_closures.hpp>
#include <pops/runtime/system/native_package_capability.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// @file
/// @brief Runtime multi-species composition: one coupled system, block by block.
///
/// Each block is a species (one state U) described by a ModelSpec (composition of generic
/// bricks: transport + source + elliptic right-hand side), with its spatial scheme
/// (limiter + Riemann flux), its time treatment and its substeps. All blocks share a
/// Poisson whose right-hand side is the sum of the per-block elliptic_rhs; the source S
/// acts per block. The core names no scenario; scenarios are compositions defined on the
/// application side (adc_cases).
///
/// Python composes (brick objects); the per-cell computation (assemble_rhs<L,F>, Newton of the
/// implicit source, multigrid/FFT) stays C++-compiled and is frozen when the block is added. No
/// Python callback enters the hot path. ``eval_rhs`` / ``get_state`` / ``set_state`` are bulk
/// inspection, initialization and verification seams; installed time Programs execute natively.

namespace pops {

class ObserverMpiLane;
template <int Dim>
class FieldNullspaceProvider;
struct FieldLogicalTimePoint;
template <int Dim>
struct CompiledFieldBoundaryKernel;
template <int Dim>
class PreparedSystemLayoutTransfer;

namespace component {
class LoadedComponent;
}

namespace runtime::system {

enum class AnalyticMappedInputKind : std::uint8_t { state_component = 0, provider_component = 1 };

/// One exact input to a bind-time mapped analytic state expression.
///
/// State components are local to the target state carrier. Provider components are addressed only
/// by their owner-qualified key and are resolved through a sealed consumer plan. No physical name
/// or global storage component crosses this interface.
struct AnalyticMappedInput {
  AnalyticMappedInputKind kind = AnalyticMappedInputKind::state_component;
  int state_component = -1;
  AuxiliaryComponentKey provider_key;

  static AnalyticMappedInput state(int component) {
    return {AnalyticMappedInputKind::state_component, component, {}};
  }
  static AnalyticMappedInput provider(AuxiliaryComponentKey key) {
    return {AnalyticMappedInputKind::provider_component, -1, std::move(key)};
  }

  void validate() const {
    if (kind == AnalyticMappedInputKind::state_component) {
      if (state_component < 0)
        throw std::invalid_argument("mapped analytic state component must be non-negative");
      return;
    }
    if (kind != AnalyticMappedInputKind::provider_component)
      throw std::invalid_argument("mapped analytic input kind is invalid");
    provider_key.validate();
  }

  void serialize_exact(ExactContractBuilder& contract) const {
    validate();
    contract.scalar(static_cast<std::uint8_t>(kind));
    if (kind == AnalyticMappedInputKind::state_component)
      contract.scalar(state_component);
    else
      provider_key.serialize_exact(contract);
  }
};

}  // namespace runtime::system

/// Immutable bind-time contract for one native transfer between two Uniform System layouts.
/// Ratios follow the native ranked axis order; the runtime validates them against the
/// actual source/target domains before allocating or launching a kernel.
template <int Dim>
struct SystemLayoutTransferSpec {
  static_assert(Dim >= 1 && Dim <= 3,
                "SystemLayoutTransferSpec only supports dimensions 1, 2, and 3");
  std::string mapping_identity;
  std::string provider_identity;
  std::string provider_component_identity;
  std::string provider_manifest_identity;
  std::string source_layout_identity;
  std::string target_layout_identity;
  std::string source_block;
  std::string target_block;
  std::string source_representation;
  std::string target_representation;
  std::string synchronization_identity;
  std::array<std::int32_t, Dim> refinement_ratio = [] {
    std::array<std::int32_t, Dim> value{};
    value.fill(1);
    return value;
  }();
  std::int32_t operation = 0;
};

/// Owned projection of PopsExecutionContextV1. Strings are values, never borrowed Python pointers.
struct SystemLayoutTransferExecution {
  std::uint32_t context_version = 0;
  std::string execution_identity;
  std::int32_t memory_space = 0;
  std::string backend_identity;
  std::string device_identity;
  std::int32_t scalar_type = 0;
  std::int32_t storage_precision = 0;
  std::int32_t compute_precision = 0;
  std::int32_t accumulation_precision = 0;
  std::int32_t reduction_precision = 0;
  std::uint64_t stream_handle = 0;
  std::string stream_identity;
  std::int64_t communicator_f_handle = 0;
  std::int64_t communicator_datatype_f_handle = 0;
  std::string communicator_identity;
  std::string communicator_datatype_identity;
};

/// Authenticated evidence returned after a prepared native mapping has completed collectively.
struct SystemLayoutTransferReceipt {
  bool applied = false;
  std::string mapping_identity;
  std::string provider_identity;
  std::string provider_component_identity;
  std::string provider_manifest_identity;
  std::string source_layout_identity;
  std::string target_layout_identity;
  std::string source_block;
  std::string target_block;
  std::string execution_identity;
  std::int32_t operation = 0;
  std::uint64_t generation = 0;
  std::uint64_t attempt = 0;
  std::uint64_t source_element_count = 0;
  std::uint64_t destination_element_count = 0;
};

namespace runtime::program {
class Profiler;  // per-node wall-clock profiler (ADC-459); full type in program/profiler.hpp
template <int Dim>
class CacheManager;  // scheduler value cache (ADC-458); full type in program/cache_manager.hpp
template <int Dim>
class ProgramContext;
template <int Dim>
class PreparedScalarBoundarySession;
template <int Dim>
struct ProgramRuntimeState;
}  // namespace runtime::program

namespace runtime::field {
struct PreparedFieldSolverSpec;
struct FieldTopologyReportRow;
}  // namespace runtime::field

namespace runtime::multiblock {
struct BoundaryEvaluationPoint;
}  // namespace runtime::multiblock

class ExecutionLane;

/// Exact compile-time-ranked mesh authority shared by every block of one uniform runtime.
/// Shape, physical bounds, topology and decomposition are lowered once from the resolved layout;
/// the native runtime never reconstructs one axis from another or recovers rank from array shape.
template <int Dim>
struct SystemConfig : RuntimeSpatialDomain<Dim> {
  static_assert(Dim >= 1 && Dim <= 3, "SystemConfig only supports dimensions 1, 2, and 3");

  std::string load_balance_route = "round_robin";
  std::string load_balance_identity = "pops.system.default.round-robin@1";
  PreparedProviderOptions load_balance_options{"pops.amr.load-balance.round-robin@1", {}};
};

/// Coupled multi-species system, composed at runtime from generic bricks.
///
/// @code{.cpp}
/// pops::SystemConfig<3> cfg;
/// cfg.shape = pops::Extent<3>{96, 64, 48};
/// pops::System<3> sys(cfg);
///
/// pops::ModelSpec ne;                       // scalar density advected by E x B
/// ne.transport = "exb";
/// ne.source = "none";
/// ne.elliptic = "charge";
/// sys.add_block("ne", ne, "minmod", "rusanov", "conservative", "explicit");
/// sys.set_poisson("charge_density", "cartesian_cg");
///
/// sys.set_density("ne", rho0);             // rho0: initial density, flattened row-major (n*n)
/// const double dt = sys.step_cfl(0.4);     // one CFL-limited step of the coupled system
/// @endcode
template <int Dim>
class System {
  static_assert(Dim >= 1 && Dim <= 3, "System only supports dimensions 1, 2, and 3");

 public:
  static constexpr int dimension = Dim;
  using HyperbolicBoundary = PreparedHyperbolicBoundary<Dim>;

  explicit System(const SystemConfig<Dim>& cfg);
  ~System();
  System(System&&) noexcept;
  System& operator=(System&&) noexcept;

  /// Adds an equation block (one species).
  /// @param model    composition of bricks (transport/source/elliptic + parameters)
  /// @param limiter  reconstruction: "none" | "minmod" | "vanleer" | "weno5" | "mc" |
  ///                 "superbee"
  /// @param riemann  numerical flux: "rusanov" (minimal generic) | "hll" (generic, requires
  ///                 model.wave_speeds) | "hllc" | "roe" (generic when the model supplies the
  ///                 HasHLLCStructure / HasRoeDissipation hooks; no layout inference or fallback)
  /// @param recon    reconstructed variables: "conservative" | "primitive" (Euler: primitive
  ///                 more robust, positivity of rho and p)
  /// @param time     "explicit" (SSPRK2) | "ssprk3" | "imex" (explicit transport, local implicit
  ///                 backward-Euler source, order 1) | "imexrk_ars222" (IMEX-RK family, ARS(2,2,2)
  ///                 scheme, order 2, cartesian only; source FULLY implicit -> incompatible
  ///                 with implicit_vars/implicit_roles)
  /// @param substeps substeps per macro-step: the block advances N times per macro-step, each
  ///                 substep of length dt/N (fast electrons: substeps=10, step dt/10).
  /// @param stride   block cadence, HOLD-THEN-CATCH-UP semantics: 1 = every macro-step (default,
  ///                 bit-identical); M > 1 = block HELD (not advanced) while (macro_step + 1) % M != 0,
  ///                 then advanced by one effective step M*dt at the macro-step where (macro_step + 1) % M == 0
  ///                 (end of an M-step window), thus temporally consistent with the fast blocks (slow block,
  ///                 e.g. neutrals on stride=20). substeps and stride are ORTHOGONAL: stride=M,
  ///                 substeps=N -> N substeps of M*dt/N, once at the end of the window. COUPLING: between two
  ///                 catch-ups, the held block enters the Poisson sum with its STALE state (last frozen
  ///                 advance). step_cfl honors the cadence (dt <= cfl*h*substeps / (stride*w)).
  /// @param evolve   false = FROZEN species (fixed background): not advanced in time, but seen by the
  ///                 system Poisson (and, in the future, by coupled sources)
  /// @param implicit_vars  IMEX only: names of the conservative variables to treat IMPLICITLY in
  ///                 the source step (backward-Euler); the others stay explicit (forward Euler). The
  ///                 mask is CARRIED BY THE BLOCK / time policy (and NOT by the model): the
  ///                 SAME model can thus be reused with different implicit treatments. EMPTY
  ///                 (default) + EMPTY implicit_roles -> model default (Model::is_implicit, or all
  ///                 implicit absent a trait) -> bit-identical. Resolved against the conservative names
  ///                 of the block; an absent name raises an EXPLICIT error.
  /// @param implicit_roles IMEX only: same implicit mask but by physical ROLE ("density",
  ///                 "momentum_x", "energy", ...) instead of the name (cf. variable_roles). Union with
  ///                 implicit_vars. A role absent from the block raises an EXPLICIT error.
  /// @param newton IMEX only: options of the local Newton of the implicit source (backward-Euler),
  ///                 grouped in a POD (ADC-214; cf. NewtonOptions). max_iters is a hard budget;
  ///                 rel_tol / abs_tol define the mandatory per-cell stopping criterion
  ///                 ||F||inf <= abs_tol + rel_tol*||F0||inf; fd_eps controls the finite-difference
  ///                 Jacobian and damping controls W -= damping*delta in (0, 1].
  /// @param newton_diagnostics Reserved compatibility flag. The Program-only System runtime rejects
  ///                 true until a typed implicit Program consumer actually publishes a Newton
  ///                 report; accepting it would otherwise allocate a carrier that no execution
  ///                 route writes.
  /// @param wave_speed_cache riemann='hll' + explicit ONLY: pre-computes model.wave_speeds once for
  ///                 every exact reconstructed face-trace pair, then reuses that interval from both
  ///                 adjacent residual cells. Net gain when wave_speeds is expensive (moment hierarchy).
  ///                 BIT-IDENTICAL to the direct HLL path for first-order, MUSCL and WENO reconstruction.
  ///                 false (default) = direct per-cell face evaluation unchanged. Wired
  ///                 on the full exact-ranked advance only: refused if riemann != 'hll', time IMEX,
  ///                 or a staircase/cutcell disc transport mode is active (explicit error, never a
  ///                 silent ignore).
  void add_block(const std::string& name, const ModelSpec& model,
                 const std::string& limiter = "minmod", const std::string& riemann = "rusanov",
                 const std::string& recon = "conservative", const std::string& time = "explicit",
                 int substeps = 1, bool evolve = true, int stride = 1,
                 const std::vector<std::string>& implicit_vars = {},
                 const std::vector<std::string>& implicit_roles = {},
                 const NewtonOptions& newton = {}, bool newton_diagnostics = false,
                 double positivity_floor = 0.0, bool wave_speed_cache = false,
                 double weno_epsilon = static_cast<double>(kWenoEpsilon));

  /// Internal installation seam for a compiled production package. The loader prepares one
  /// revocable, detached block capability against the exact aggregate provider graph. The complete
  /// canonical BindSchema vector crosses the fixed ABI once and is injected into the generated
  /// model before its closures are constructed. Package and module ABI keys must match; the expected
  /// model identity and ``pops.binary.v1`` token must match the authenticated facade artifact.
  /// @param limiter "none" | "minmod" | "vanleer" | "weno5" | "mc" | "superbee"
  ///                (weno5: add_compiled_model reallocates the block state to block_n_ghost = 3
  ///                ghosts after install_block, like add_block)
  /// @param riemann "rusanov" | "hll" | "hllc" | "roe"
  /// @param recon   "conservative" | "primitive"
  /// @param time    "explicit" (SSPRK2) | "ssprk3" | "euler" | "imex" (the template marshals the explicit
  ///                RK scheme down to the loader's make_block, parity with add_block)
  /// @param gamma   adiabatic index of the block (set_density / inter-species couplings)
  /// @param params complete resolved runtime-parameter vector in declaration order
  /// @param stride block cadence (1 = every step, default; cf. add_block)
  /// Stage one compiled package. Staging authenticates the exact bytes and ABI but neither publishes
  /// routes nor builds blocks. ``finalize_native_packages`` invokes every registrar into one detached
  /// provider graph, prepares the complete block/field image, and publishes only after consensus.
  void register_native_package(
      const std::string& name, const std::string& so_path,
      const std::string& expected_model_identity, const std::string& expected_binary_identity,
      const std::string& limiter = "minmod", const std::string& riemann = "rusanov",
      const std::string& recon = "conservative", const std::string& time = "explicit",
      double gamma = static_cast<double>(kPhysicalDefaultGamma), int substeps = 1,
      bool evolve = true, int stride = 1, const std::vector<double>& params = {},
      double positivity_floor = 0.0, NewtonOptions newton = {}, bool newton_diagnostics = false);

  /// Authenticate and stage one exact-ranked external Riemann package. The external DSO owns the
  /// prepared model and numerical flux type; its handle is retained by the ordinary native-package
  /// transaction so every installed closure remains resident for the lifetime of this System.
  /// Both canonical DSO provider hooks are mandatory. A zero-provider brick exports explicit empty
  /// System/AMR hooks; the System hook is staged before the one global provider graph is sealed.
  void register_external_riemann_package(
      const std::string& name, const std::string& so_path, const std::string& brick_id,
      const std::string& expected_sha256, int expected_nvars, int expected_provider_count,
      const std::string& expected_model_identity, const std::string& provider_consumer_qid,
      const std::string& limiter = "minmod", const std::string& recon = "conservative",
      const std::string& time = "explicit",
      double gamma = static_cast<double>(kPhysicalDefaultGamma), int substeps = 1,
      bool evolve = true, int stride = 1, double positivity_floor = 0.0,
      double weno_epsilon = static_cast<double>(kWenoEpsilon));

  /// Seal the aggregate auxiliary graph, allocate its exact compact carrier, then install every
  /// staged native block in canonical package order. Any registrar, seal, or installer failure
  /// restores the complete pre-finalization System image and re-arms the exact staged journal for
  /// retry without publishing candidate closures.
  void finalize_native_packages();

  /// Native-loader-only hand-off after ABI/manifest validation. The canonical registrar is required
  /// even for an empty provider graph; only private boundary-component staging may omit it. The
  /// package lifetime keeps its local DSO resident until all closures it installed are destroyed.
  /// This is intentionally a typed C++ seam, not a metadata/JSON parser.
  POPS_EXPORT void stage_prepared_native_package(
      std::string identity, std::function<void()> route_registrar, std::function<void()> installer,
      std::shared_ptr<void> package_lifetime,
      std::shared_ptr<runtime::system::NativePackageCapabilityState<Dim>> capability);

  /// ABI key of the module (compiler + C++ standard + signature of the pops headers, frozen at
  /// compilation). Compared to the key baked into a native loader .so by add_native_block; also exposed
  /// on the Python side so that emit_cpp_native_loader (or a diagnostic) can consult it.
  static std::string abi_key();

  /// @name Native compiled-model seam
  /// @{
  /// Install the one model-qualified hyperbolic boundary retained by a block. The parser accepts
  /// exactly 2*Dim oriented faces; mapped periodic identifications and additive boundary
  /// residual/JVP components belong to separately qualified providers and cannot be smuggled into
  /// this Cartesian core.
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
  /// Register the exact state Handle owned by a materialized block.  This registry is independent
  /// of boundary plans: a block with periodic-only or no physical boundary remains a legal N-ary
  /// dependency of another block's boundary component.
  POPS_EXPORT void install_block_state_route(const std::string& name,
                                             const std::string& state_identity);
  /// Bind one exact solved-field Handle identity to its authenticated provider storage slot.
  POPS_EXPORT void install_field_storage_route(const std::string& field_identity,
                                               const std::string& provider_slot);
  /// Retain the exact RuntimeInstance communicator lane used by every prepared Uniform boundary
  /// operation. The lane is materialized from the caller's authenticated ExecutionContext before
  /// any state route or boundary plan is published.
  POPS_EXPORT void install_prepared_boundary_execution_lane(std::shared_ptr<ExecutionLane> lane);
  [[nodiscard]] POPS_EXPORT const ExecutionLane& prepared_boundary_execution_lane() const;
  /// Stage one authenticated host GhostBoundary component before its block package is built. The
  /// late package installer wraps the exact prepared block after materialization, inside the same
  /// collective native-package snapshot/rollback transaction.
  POPS_EXPORT void stage_prepared_ghost_boundary_component(
      const std::string& block, std::shared_ptr<PreparedGhostBoundaryComponent> component);
  POPS_EXPORT void stage_prepared_boundary_flux_component(
      const std::string& block, std::shared_ptr<PreparedBoundaryFluxComponent> component);
  POPS_EXPORT void stage_prepared_field_boundary_component_pair(
      const std::string& block, std::shared_ptr<PreparedFieldBoundaryResidualComponent> residual,
      std::shared_ptr<PreparedFieldBoundaryJvpComponent> jvp);
  /// Roll back a failed all-block pre-build boundary transaction.  Internal bind seam only.
  POPS_EXPORT void discard_hyperbolic_boundaries();
  /// Install one already-authenticated exact-ranked shared-interface provider after every endpoint
  /// block has been materialized. Interface geometry remains private to that provider; the generic
  /// System never rebuilds an axis route from scalar metadata.
  POPS_EXPORT void install_interface_provider(SystemInterfaceProvider<Dim> provider);
  /// Roll back a failed all-interface post-block installation transaction.
  POPS_EXPORT void discard_interface_flux_components();
  POPS_EXPORT std::size_t interface_evaluation_count(const std::string& identity,
                                                     int level = 0) const;
  /// Commit one complete prepared block image. Every callback and exact-ranked storage requirement
  /// is validated before the block registry or shared auxiliary field is mutated.
  POPS_EXPORT void install_prepared_block(PreparedSystemBlock<Dim> block);

  /// Immutable exact geometry consumed by an out-of-line generated block preparer. Returning a
  /// value prevents a native package from retaining a reference into the facade implementation.
  POPS_EXPORT Geometry<Dim> prepared_block_geometry() const;
  /// Exact axis topology captured from the resolved layout. Generated packages use it to prepare
  /// one ranked halo schedule; they never reconstruct periodicity from boundary spellings.
  POPS_EXPORT std::array<bool, Dim> prepared_block_periodicity() const;
  /// Immutable-address compact provider carrier captured by prepared block kernels.  It is null
  /// exactly when the sealed graph has no provider values; callers with ``ProviderValues<0>`` must
  /// not dereference it.  A non-null carrier has exactly ``registry.slot_count()`` components.
  [[nodiscard]] POPS_EXPORT const MultiFab<Dim>* prepared_block_auxiliary_storage() const;
  /// Shared lifetime authority for the same immutable-address carrier. Generated direct-System
  /// adapters retain this owner so prepared callbacks cannot outlive the accepted provider image.
  [[nodiscard]] POPS_EXPORT std::shared_ptr<const runtime::system::AuxiliaryStorageGroups<Dim>>
  prepared_block_provider_storage_owner() const;
  [[nodiscard]] POPS_EXPORT const runtime::system::AuxiliaryStorageGroups<Dim>*
  prepared_block_provider_storage_groups() const;
  /// AMR preparation owns the collective halo-fill phase and therefore receives the accepted group
  /// set through this narrowly scoped mutable seam.  It may fill ghost regions only; publication
  /// values remain owned by the System auxiliary transaction.
  POPS_EXPORT runtime::system::AuxiliaryStorageGroups<Dim>* prepared_amr_provider_storage_groups();

  /// Register one immutable, owner-qualified auxiliary producer.  A producer is either an external
  /// input, a generated native derivation, or a field-output route.  The System does not attach any
  /// physical meaning to an output: every carrier component is identified solely by
  /// ``AuxiliaryComponentKey`` and receives a compact slot when the registry is sealed.
  ///
  /// This is an assembly/program-install operation.  The complete graph is collectively sealed
  /// before its carrier is allocated; no producer can be added afterwards.
  POPS_EXPORT void install_prepared_auxiliary_provider(
      runtime::system::PreparedAuxiliaryProvider<Dim> provider);

  /// Register the immutable value image required by one compiled native consumer.  Its local slots
  /// are resolved to global compact storage at seal and never inferred from physical names.
  POPS_EXPORT void install_auxiliary_consumer_plan(
      runtime::system::AuxiliaryConsumerProviderPlan<Dim> plan);

  /// Commit the complete auxiliary provider graph.  Validates its dependency DAG and exact contract
  /// locally, verifies the exact bytes collectively, then sizes the auxiliary carrier to its compact
  /// slot count.  It is called by ``mark_bound``; generated package installers may call it earlier
  /// when they need to stage initialization inputs.
  POPS_EXPORT void seal_auxiliary_providers();

  /// Stage one owner-qualified external input over the complete exact-ranked domain.  The value is
  /// retained as a candidate only; it becomes visible to native consumers at the next matching
  /// ``refresh_auxiliary`` transaction.  A component with a derived/field-output producer cannot be
  /// uploaded through this path.
  POPS_EXPORT void stage_auxiliary_input(const runtime::system::AuxiliaryComponentKey& key,
                                         const std::vector<double>& values);

  /// Run one exact auxiliary evaluation transaction.  Due external inputs are staged, due native
  /// providers launch in dependency order, every produced component is checked collectively for
  /// finiteness, then the complete candidate carrier and registry generation are published together.
  /// Any failure leaves the accepted carrier and accepted provider points unchanged.
  POPS_EXPORT void refresh_auxiliary(const runtime::system::AuxiliaryEvaluationPoint& point);

  /// Compact slot of a sealed component key and the corresponding accepted scalar field.  The key,
  /// rather than a legacy physical label or a raw component number, is the public authority.
  [[nodiscard]] POPS_EXPORT runtime::system::AuxiliaryStorageAddress<Dim> auxiliary_address(
      const runtime::system::AuxiliaryComponentKey& key) const;
  [[nodiscard]] POPS_EXPORT std::vector<double> auxiliary_component(
      const runtime::system::AuxiliaryComponentKey& key) const;
  [[nodiscard]] POPS_EXPORT std::string auxiliary_registry_contract() const;
  [[nodiscard]] POPS_EXPORT const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>&
  prepared_auxiliary_consumer_plan(const std::string& consumer_qid) const;
  /// Durable accepted auxiliary metadata for the Uniform runtime.  Rank-local group payloads are
  /// staged by the checkpoint backend; this image authenticates exact group identities,
  /// owner-qualified ComponentKeys, shapes, and accepted provider generations before publication.
  [[nodiscard]] POPS_EXPORT runtime::system::AuxiliaryCheckpointAcceptedState<Dim>
  capture_auxiliary_checkpoint_accepted_state() const;
  /// Rank-local checkpoint capacity derived from the sealed auxiliary registry. The pair is
  /// ``(payload-free POPSAUX2 bytes, scalar values per full-domain level)``.
  [[nodiscard]] POPS_EXPORT std::pair<std::size_t, std::size_t> checkpoint_auxiliary_capacity()
      const;
  /// Restore the accepted provider provenance only after the checkpoint backend has staged a
  /// compatible rank-local group payload privately.  The collective preflight and rollback image
  /// ensure a rejected checkpoint cannot expose a partial auxiliary generation.
  POPS_EXPORT void restore_auxiliary_checkpoint_accepted_state(
      const runtime::system::AuxiliaryCheckpointAcceptedState<Dim>& state);
  /// Decode one sealed POPSAUX2 image inside the authenticated System execution lane.  Local
  /// decode/allocation failure is agreed before the typed restore enters its first collective.
  using AuxiliaryCheckpointByteViewProvider = std::function<std::span<const std::uint8_t>()>;
  POPS_EXPORT void restore_auxiliary_checkpoint_accepted_state_bytes(
      const AuxiliaryCheckpointByteViewProvider& payload);
  /// @}

  /// Configures the shared Poisson.
  /// @param rhs    only mode: "charge_density", f = sum_s elliptic_rhs_s(u_s)
  /// @param solver "cartesian_cg", the exact-ranked constant-coefficient uniform solver.
  /// @param bc     "auto" | "periodic" | "dirichlet" | "neumann"
  /// @param abs_tol Absolute residual floor of CartesianCG.
  /// @param rel_tol Relative residual tolerance of CartesianCG.
  /// @param max_iterations CartesianCG iteration cap.
  void set_poisson(const std::string& rhs = "charge_density",
                   const std::string& solver = "cartesian_cg", const std::string& bc = "auto",
                   double abs_tol = static_cast<double>(kCartesianCGDefaultAbsTol),
                   double rel_tol = static_cast<double>(kCartesianCGDefaultRelTol),
                   int max_iterations = kCartesianCGDefaultMaxIterations);
  /// Materialize one immutable provider instance from an already registered family. Provider-owned
  /// code authenticates and decodes @p options; the System core only stores the returned route.
  POPS_EXPORT std::string register_configured_field_solver_provider(
      const std::string& family_route, const std::string& provider_route,
      const PreparedProviderOptions& options);

  /// Install one fully resolved field solver route keyed by the digest of its block-qualified
  /// provider identity. ``plan_identity`` independently commits the complete resolved semantics.
  /// Before any named backend is materialized, the canonical ordered (slot, plan_identity) registry
  /// must agree exactly on every MPI rank. Duplicate slots are refused, including exact repeats.
  void set_field_solver_plan(const std::string& provider_slot, const std::string& plan_identity,
                             const std::string& provider_identity,
                             const std::string& output_owner_identity,
                             const std::string& output_block, const std::string& output_key,
                             const std::vector<std::string>& provider_identities,
                             const std::vector<std::string>& provider_blocks,
                             const std::vector<std::string>& provider_keys,
                             const std::vector<double>& provider_coefficients,
                             const std::string& backend_provider_route);
  /// Install the resolved scalar reaction coefficient of one named screened field.
  void set_field_reaction(const std::string& provider_slot, double reaction);
  /// Register one exact generated FieldTopology+FieldSolver provider under @p provider_slot.
  /// The same route can be selected by the principal Poisson field or any named field; registration
  /// does not depend on a pre-existing field plan. Returns the provider's manifest-qualified exact
  /// identity while the stable slot remains the selection route.
  POPS_EXPORT std::string register_field_solver_provider(
      const std::string& provider_slot, runtime::field::PreparedFieldSolverSpec spec,
      std::shared_ptr<component::LoadedComponent> topology,
      std::shared_ptr<component::LoadedComponent> solver);
  /// Adds a native field-nullspace provider before binding. Builtins and extensions use this same
  /// registry; the System core never interprets a mathematical nullspace family name.
  POPS_EXPORT void register_field_nullspace_provider(
      std::shared_ptr<const FieldNullspaceProvider<Dim>> provider);
  /// Select the provider for the principal field configured by set_poisson.
  void set_default_field_nullspace(const std::string& nullspace_provider_identity,
                                   const PreparedProviderOptions& options);
  POPS_EXPORT void set_field_topology_authority(const std::string& provider_slot,
                                                const std::string& provider_kind,
                                                const std::string& provenance,
                                                const std::string& topology_digest);
  POPS_EXPORT std::vector<runtime::field::FieldTopologyReportRow> field_topology_report(
      const std::string& provider_slot) const;

  /// Install the exact lower/upper boundary residual for every axis of this specialization. ``kind`` is
  /// periodic/dirichlet/neumann/mixed; mixed represents alpha*u + beta*du/dn = value.
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

  /// Install generated boundary residual/JVP launchers owned by the compiled Program artifact.
  /// The shared library remains loaded for the System lifetime, so the direct function pointers are
  /// stable and no registry lookup occurs in a face-cell loop.
  POPS_EXPORT void set_field_boundary_kernel(const std::string& provider_slot,
                                             const CompiledFieldBoundaryKernel<Dim>& kernel);
  POPS_EXPORT void set_field_logical_timepoint(const std::string& provider_slot,
                                               const FieldLogicalTimePoint& point);
  POPS_EXPORT void set_field_boundary_parameters(const std::string& provider_slot,
                                                 const std::vector<double>& parameters);
  void set_field_newton_plan(const std::string& provider_slot, double tolerance, int max_iterations,
                             double linear_tolerance, int linear_max_iterations, int restart,
                             double armijo, double minimum_step);

  /// Select one prepared nullspace provider. The schema and scalar values remain opaque to System;
  /// the selected provider validates them after the concrete operator/layout facts are available.
  void set_field_nullspace(const std::string& provider_slot,
                           const std::string& nullspace_provider_identity,
                           const PreparedProviderOptions& options);

  /// Configured field (Poisson) solver token. A uniform System reports ``cartesian_cg``; the
  /// ``geometric_mg`` token belongs to AmrSystem MG/FAC. Read by install_program for the
  /// Spec criterion-24 solver requirement check (a field operator that requires a named solver is
  /// rejected at install when the configured solver does not match) and exposed for introspection.
  std::string poisson_solver() const;

  /// Runtime-private native seam for a generic Cartesian level set. @p opcodes / @p literals are
  /// one validated postfix scalar program using the analytic VM. The System revalidates it, samples
  /// signed phi once, preflights finiteness collectively over every local patch plus the mask ghost
  /// layer, then publishes phi, mask, and static cut-cell metrics as one transaction. No analytic
  /// interpreter reaches a RHS or time stage. Active means phi < 0.
  /// Python authoring reaches this only through its canonical analytic-expression lowering.
  void set_analytic_level_set(const std::vector<std::string>& opcodes,
                              const std::vector<double>& literals, const std::string& mode = "none",
                              double kappa_min = 0.0, double face_open_eps = 0.0,
                              double cut_theta_min = 0.0);

  /// Sets ONLY the level-set transport mode: "none" | "staircase" | "cutcell". Useful to toggle
  /// the mode after installing a generic analytic level set, or to reset it to "none"
  /// (back to the full cartesian path, bit-identical). Requesting a mode != "none" without a prepared
  /// signed level set raises an explicit error (the mode alone has no geometry to apply).
  void set_geometry_mode(const std::string& mode);

  /// @return the 0/1 cell-centered domain mask over the exact-ranked flattened layout. Without
  /// a level-set installation, returns an ALL-ACTIVE mask (only 1.0): the transport sub-domain is
  /// the entire domain (default path). Diagnostic / contract verification.
  std::vector<double> embedded_boundary_mask() const;

  /// Sets the density of a species (component 0), n*n row-major array. The other
  /// components (momentum, energy) are set to the at-rest equilibrium.
  void set_density(const std::string& name, const std::vector<double>& rho);

  /// Initializes the state of a block from its PRIMITIVE variables (rho, u, v, p ...): @p prim is
  /// a flat ncomp*n*n component-major array in the order of primitive_vars(name). Each cell
  /// is converted to CONSERVATIVE variables by the block's MODEL conversion (M.to_conservative),
  /// then written into the state. Ergonomic counterpart of set_density for a model with several primitives
  /// (compressible 4 var: p; isothermal 3 var; scalar 1 var: identity). cf. get_primitive_state.
  void set_primitive_state(const std::string& name, const std::vector<double>& prim);

  /// Reads the CONSERVATIVE state of the block and converts it to PRIMITIVE variables via the model
  /// conversion (M.to_primitive). @return a flat ncomp*n*n component-major array in the order of
  /// primitive_vars(name) (diagnostics: velocities, pressure). Exact round-trip with set_primitive_state.
  std::vector<double> get_primitive_state(const std::string& name);

  /// Type-erasure of the POINTWISE (one cell) cons <-> prim conversion of a block: in/out are
  /// arrays of ncomp doubles. Installed by install_block / add_compiled_model / push_dynamic from
  /// the block's model and consumed by publication and prepared-boundary validation. Primitive
  /// field materialization exclusively consumes CellBatchRecovery below.
  using CellConvert = std::function<void(const double* in, double* out)>;
  /// Fallible conservative -> primitive conversion. A failed report forbids writing @p out.
  using CellRecovery = std::function<RecoveryReport(const double* in, double* out)>;
  using CellBatchRecovery = UniformCellRecovery;

  /// Adds a GLOBAL time-step bound, evaluated ONCE per step_cfl (host):
  /// dt <= fn() when fn() > 0 and finite (otherwise the bound does not constrain this step).
  /// It is the hook for NON cell-local constraints: multi-block coupling, Schur/Poisson
  /// stage, AMR/scheduler, or a user policy (startup ramp...). @p label
  /// names the bound in last_dt_bound() ("global:<label>"). A Python callback is acceptable HERE
  /// (one evaluation per step, never per cell).
  void add_dt_bound(const std::string& label, std::function<double()> fn);

  /// Name of the ACTIVE bound (the one that set dt) of the last step_cfl: "transport:<block>",
  /// "parabolic_frequency:<block>", "source_frequency:<block>", "stability_dt:<block>",
  /// "global:<label>", "degenerate" (no evolving block), or "" if no step_cfl has run. Diagnostic
  /// of the step policy.
  std::string last_dt_bound() const;

  // The named inter-species couplings (ionization / collision / thermal exchange) are no longer C++
  // methods (ADC-595): they are Python presets (python/pops/physics/coupling_presets.py) that lower to
  // the generic coupled source and register through add_coupling_operator with a declared conservation
  // contract. A new coupling needs no new public C++ method.

  /// Registers a GENERIC inter-species COUPLED SOURCE described by a BYTECODE
  /// (pops.dsl.CoupledSource, P5 phase 1). Unlike the named couplings
  /// (add_ionization / add_collision / add_thermal_exchange) which freeze a formula, this one reads
  /// (block, role) fields as INPUT and writes source terms (block, role) computed by symbolic
  /// EXPRESSIONS compiled to postfix bytecode (stack machine, evaluated in the same
  /// for_each_cell device; no per-cell Python callback). Registration validates the bytecode and
  /// exposes its typed metadata and stability bounds. It does not schedule a hidden post-transport
  /// split: the installed whole-system Program must lower the coupling explicitly.
  ///
  /// FLAT ABI (no C++ object crosses the boundary):
  /// @param prog      bytecode description of the coupling grouped in a POD (ADC-214; cf.
  ///                  CoupledSourceProgram): in_blocks / in_roles (inputs read and their roles),
  ///                  consts (.param() parameters, loaded after the inputs), out_blocks / out_roles
  ///                  (targets of each term), prog_ops / prog_args / prog_lens (concatenated opcodes
  ///                  of ALL terms, stack machine cf. CsOp, parallel arguments, and length
  ///                  per term), and freq_prog_ops / freq_prog_args (OPTIONAL program of a
  ///                  PER-CELL frequency mu(U), same stack machine / register table; EMPTY =
  ///                  constant frequency only, bit-identical). These arrays were a long list
  ///                  of `std::vector` of the same type, interchangeable at the call site.
  /// @param frequency  declared CONSTANT frequency mu [1/s] of the coupling (audit wave 3,
  ///                   CoupledSource.frequency): step bound dt <= cfl / mu aggregated by step_cfl
  ///                   on the Program macro-dt, without a block substeps/stride factor. <= 0 (default) = no
  ///                   bound, bit-identical. Stays flat (a double, outside the homogeneous family).
  /// @param label      name of the coupling (reason "coupled_source:<label>" of last_dt_bound). Stays
  ///                   flat (a string, outside the homogeneous family). When prog.freq_prog_ops/_args are
  ///                   non-empty, step_cfl reduces the MAX of mu over the cells
  ///                   (global all_reduce_max) and bounds dt <= cfl / max(mu) (reason
  ///                   "coupled_source:<label>"). max(mu) <= 0 = no bound this step.
  /// Unknown blocks / roles, an exceeded capacity or a malformed program raise an EXPLICIT
  /// error before any step.
  void add_coupled_source(const CoupledSourceProgram& prog, double frequency = 0.0,
                          const std::string& label = "coupled_source");

  /// Registers a TYPED coupling operator (ADC-595): the same coupled-source program as
  /// add_coupled_source, PLUS its declared conservation contract and frequency bound. The declared
  /// ConservationContract is VALIDATED at registration (host, fail-loud) against the actual output
  /// terms (validate_coupling_contract) BEFORE the program is stored, then the program is lowered
  /// through the SAME add_coupled_source path (bit-identical numerics), and the declared contracts are
  /// recorded for coupled_operators(). An empty (unchecked) contract is equivalent to add_coupled_source.
  void add_coupling_operator(const CouplingOperator& op);

  /// Install one executable coupling that was prepared by an authenticated dimension-qualified
  /// package. `provider_contract` is its immutable identity/version/program digest; the operator
  /// receives the simultaneous candidate-state pack selected by Program.
  POPS_EXPORT void install_prepared_coupling_operator(
      const std::string& label, const std::string& provider_contract, CouplingOperatorView view,
      std::function<void(Real, const std::vector<MultiFab<Dim>*>&)> operation,
      double constant_frequency = 0.0, std::function<Real()> maximum_frequency = {});

  /// Read-only view of the registered coupling operators (ADC-595): label + declared conservation /
  /// frequency contracts, in registration order, so a Program or a runtime report can enumerate the
  /// couplings as typed operators instead of reading raw bytecode. A raw add_coupled_source registers an
  /// "unchecked" entry (empty contract). Empty until the first coupling is added.
  const std::vector<CouplingOperatorView>& coupled_operators() const;

  /// Apply every registered coupling operator to one complete simultaneous candidate-state pack.
  /// The pack is indexed by System block identity and must match every block's exact distributed
  /// layout.  This is the native Program primitive for operator splitting: generated Programs pass
  /// their uncommitted endpoint candidates, then project and atomically commit them.  The accepted
  /// live states are therefore never a hidden coupling workspace.
  POPS_EXPORT std::size_t apply_coupling_operators(
      Real dt, const std::vector<MultiFab<Dim>*>& candidate_states);

  /// Internal Program publication preflight. Validates one terminal candidate through the exact
  /// block model's prepared conservative-to-primitive recovery before commit_many copies any block
  /// into accepted storage. The operation is collective and read-only; refusal leaves every live
  /// state unchanged. The public wrapper discards the authenticated active-cell mask returned by
  /// the private validator.
  POPS_EXPORT void validate_program_state_publication_candidate(
      int block, const MultiFab<Dim>& candidate) const;

  /// Solve Poisson then derive aux = (phi, grad phi). The candidate potential and aux remain
  /// physically private until the returned one-shot outcome is consumed with Accept.
  [[nodiscard]] POPS_EXPORT SolveOutcome solve_fields();
  /// Per-stage field solve (ADC-409): SAME elliptic solve + aux derivation as solve_fields(), but
  /// block @p block_idx assembles its Poisson RHS from @p U_stage instead of its live state (the
  /// other blocks keep theirs). This re-fills the SHARED aux with phi(U_stage) so a field-coupled
  /// multi-stage compiled Program can re-solve the fields from each STAGE state -- the stages run
  /// sequentially, so stage k's RHS (called right after this) reads phi from stage k's own state
  /// before the next stage overwrites the aux. With block_idx 0 and U_stage = U^n (the first stage)
  /// it is identical to solve_fields(). POPS_EXPORT: resolved by a compiled program .so (ProgramContext)
  /// across the dlopen boundary. @throws std::out_of_range if @p block_idx is not a valid block.
  [[nodiscard]] POPS_EXPORT SolveOutcome solve_fields_from_state(int block_idx,
                                                                 const MultiFab<Dim>& U_stage);
  /// Point-qualified stage solve used by generated implicit operators.  System has one mesh level,
  /// but the exact point remains part of the cross-target contract and is never reconstructed.
  [[nodiscard]] POPS_EXPORT SolveOutcome solve_fields_from_state_at(
      const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
      int block_idx, const MultiFab<Dim>& U_stage);
  /// Coupled multi-block field solve (Spec 3 criterion 24, ADC-457): SAME elliptic solve + aux
  /// derivation as solve_fields(), but the system Poisson RHS is assembled from the SIMULTANEOUS stage
  /// states of MULTIPLE blocks at once -- every coupled block reads its OWN stage state, not a single-
  /// target override. @p U_stages is indexed BY BLOCK INDEX (its size must equal n_blocks()); entry b
  /// != nullptr -> block b contributes its stage state, entry b == nullptr -> block b contributes its
  /// live state. With every entry pointing at the corresponding live state it is bit-identical to
  /// solve_fields(). The codegen lowers P.solve_fields_from_blocks([...]) to this -- the seam a multi-
  /// species field-coupled step uses (the IR commit_many guarantee: no operator observes a partially
  /// committed group). POPS_EXPORT: resolved by a compiled program .so (ProgramContext) across the
  /// dlopen boundary. @throws std::invalid_argument if @p U_stages is not sized to n_blocks().
  [[nodiscard]] POPS_EXPORT SolveOutcome
  solve_fields_from_blocks(const std::vector<const MultiFab<Dim>*>& U_stages);
  /// @name Named multi-elliptic fields (ADC-428)
  /// Exact-ranked API for a SECOND elliptic solve (beyond the default Poisson). Installation is
  /// accepted only when the selected native specialization owns a dimension-qualified field-solver
  /// provider; the facade never falls back to the historical 2-D carrier.
  /// @{
  /// Solve named @p field's elliptic problem from block @p block_idx's stage state @p U_stage and write
  /// its solved phi (+ centered gradient) into the field's own aux components. The codegen lowers
  /// P.solve_fields(field=name, state=U) to this. @throws if @p field is unregistered or the block
  /// index is invalid.
  [[nodiscard]] POPS_EXPORT SolveOutcome solve_fields_from_state(const std::string& field,
                                                                 int block_idx,
                                                                 const MultiFab<Dim>& U_stage);
  /// Solve named @p field from the exact simultaneous stage states of all contributing blocks.
  /// @p U_stages is indexed by System block; nullptr keeps that block at its accepted live state.
  /// Unlike the historical ProgramContext route, this contract never selects or mutates a
  /// representative block.
  [[nodiscard]] POPS_EXPORT SolveOutcome solve_fields_from_blocks(
      const std::string& field, const std::vector<const MultiFab<Dim>*>& U_stages);
  /// Register named @p field's exact-ranked provider outputs. ``output_keys`` contains either the
  /// potential alone or the potential followed by one gradient component per native axis. Each key
  /// must already be owned by a sealed ``field_output`` provider; no integer carrier slot crosses
  /// the package boundary.
  /// @throws std::logic_error before mutation when no exact-ranked field-solver provider is installed.
  POPS_EXPORT void register_elliptic_field(
      const std::string& block, const std::string& field,
      const std::vector<runtime::system::AuxiliaryComponentKey>& output_keys, int gradient_sign);
  /// Attach named @p field's RHS closure (+= elliptic_field_rhs(U)) to block @p block_name. Called by
  /// the native loader (make_poisson_rhs of the per-field brick). @throws before mutation if the
  /// block is unknown or no exact-ranked field-solver provider is installed.
  POPS_EXPORT void set_block_elliptic_field(
      const std::string& block_name, const std::string& field,
      std::function<void(const MultiFab<Dim>&, MultiFab<Dim>&)> rhs);
  /// @}
  void step(double dt);  ///< solve_fields, then advances each block according to its scheme
  void advance(double dt, int nsteps);
  /// RuntimeInstance-only outer transaction spanning native advancement and prepared consumers.
  void begin_step_transaction();
  /// Seal the native state while retaining its accepted snapshot until external effects publish.
  void commit_step_transaction();
  /// Release the accepted snapshot after every external effect has published successfully.
  void finalize_step_transaction();
  /// Restore the accepted snapshot, including after commit but before finalize.
  void rollback_step_transaction();
  /// Checkpoint-specific aliases over the same accepted native snapshot.  Commit authenticates the
  /// fallible precondition while retaining rollback authority; finalize only releases ownership.
  void begin_restart_transaction();
  void commit_restart_transaction();
  void finalize_restart_transaction() noexcept;
  void rollback_restart_transaction();
  /// Volume-weighted L2 norm of each block's accepted macro-step change. RuntimeInstance calls
  /// this collective only while an outer transaction still retains U^n.
  POPS_EXPORT std::map<std::string, double> step_change_l2() const;

  /// Advances one step using the smallest prepared bound.  For an explicit Cartesian diffusive
  /// block, dt = cfl * substeps / (stride * (max(speed, speed_floor) / h_min +
  /// 2 nu sum_a h_a^-2)); source, model, coupled, global, Program and strategy bounds may reduce
  /// it further.  The exact request and selected decision are authenticated on the RuntimeInstance
  /// lane before publication. @return the dt used.
  double step_cfl(double cfl, double speed_floor = static_cast<double>(kCflSpeedFloor),
                  double max_dt = std::numeric_limits<double>::infinity(), double min_dt = 0.0);
  /// @name Profiling (Spec 3 section 29-30, ADC-459)
  /// Per-phase / per-brick wall-clock timing of the step. Disabled by default (no hot-path cost
  /// when off). enable_profiling() then step()/step_cfl() then profile_report() returns the table;
  /// reset_profiling() clears it. Per-rank (no MPI reduction); the per-Program-node granularity is
  /// wired through the compiled-program path as a follow-up.
  /// @{
  void enable_profiling();
  void disable_profiling();
  bool is_profiling() const;
  void reset_profiling();
  std::string profile_report() const;
  /// Structured solver/runtime diagnostic events (field solve traces, MG markers when enabled).
  /// Empty unless the relevant diagnostic path was exercised; no stdout/stderr scraping.
  std::vector<RuntimeDiagnosticEvent> solver_diagnostics() const;
  /// The System-owned Profiler (a non-owning reference; lives as long as the System). A compiled time
  /// Program reaches it through ProgramContext::profile_node to time each Program node into the SAME
  /// table sim.profile_report() renders -- so per-node scopes ("node:rhs2", ...) accumulate alongside
  /// the coarse "step" / "field_solve" phases. POPS_EXPORT: a generated problem.so resolves it across
  /// the dlopen boundary like the other ProgramContext seam accessors (block_state, grid_context).
  POPS_EXPORT runtime::program::Profiler& profiler();
  /// @}

  /// @name Primitives for a time integrator written in Python
  /// solve_fields(); R = eval_rhs(name); U = get_state(name); ...; set_state(name, U).
  /// @{
  std::vector<double> eval_rhs(const std::string& name);   ///< -div F + S, size ncomp*n*n
  std::vector<double> get_state(const std::string& name);  ///< U, ncomp*n*n (component-major)
  void set_state(const std::string& name, const std::vector<double>& u);
  std::int64_t set_analytic_expression_state(const std::string& name, const std::string& space,
                                             const std::string& centering,
                                             const std::string& projection,
                                             const std::vector<std::vector<std::string>>& opcodes,
                                             const std::vector<std::vector<double>>& literals);
  std::int64_t set_analytic_mapped_state(
      const std::string& name, const std::vector<std::vector<std::string>>& opcodes,
      const std::vector<std::vector<double>>& literals,
      const std::vector<runtime::system::AnalyticMappedInput>& inputs,
      const std::string& consumer_qid);
  std::int64_t set_analytic_gaussian_state(const std::string& name, const RealVector<Dim>& center,
                                           double background, double amplitude,
                                           double inverse_width);
  int n_vars(const std::string& name) const;
  /// Variable names of a block (introspection): kind = "conservative" | "primitive".
  std::vector<std::string> variable_names(const std::string& name,
                                          const std::string& kind = "conservative") const;
  /// PHYSICAL roles of the variables of a block (parallel to variable_names): "density",
  /// "momentum:0", "energy", ... or "custom" if the block does not provide its roles. This is what
  /// the inter-species couplings resolve (index_of(role)) instead of a literal index.
  std::vector<std::string> variable_roles(const std::string& name,
                                          const std::string& kind = "conservative") const;
  /// Adiabatic index declared by the block and read by model-aware coupling providers.
  double block_gamma(const std::string& name) const;
  /// @}

  /// @name Compiled time-program seam (epic ADC-399 / ADC-401)
  /// Lets a generated problem.so (via pops::runtime::program::ProgramContext) run a time Program during
  /// sim.step(dt): install a macro-step body and reach per-block storage. The .so reimplements nothing
  /// -- it composes these primitives (solve_fields(); ProgramContext::rhs_into(b, U, R, rate_id);
  /// saxpy(U, dt, R)). The authored rate identity is mandatory at the native boundary.
  /// @{
  /// Install the mandatory macro-step body. System::step, advance and step_cfl reject before
  /// mutation while it is absent. An empty std::function is rejected: there is no public temporal
  /// route that silently clears the whole-system Program.
  /// POPS_EXPORT: a generated problem.so resolves these across the dlopen boundary from the globally
  /// promoted host; without default visibility the .so could not find them (_pops is built with
  /// hidden visibility). The generated package itself remains RTLD_LOCAL.
  POPS_EXPORT void install_program_step(std::function<void(double)> step);
  /// Set the compiled-Program macro-step cadence (ADC-411): SYSTEM-level @p substeps and @p stride
  /// around the installed program closure (cf. System::step). @p substeps subdivides each
  /// effective step into @p substeps calls program_.step_(eff_dt/substeps); @p stride runs the whole
  /// program once per @p stride macro-steps with eff_dt = stride*dt (GLOBAL hold-then-catch-up, the
  /// clock still ticks every macro-step). Both must be >= 1 (throws std::invalid_argument otherwise).
  /// Default 1/1 -> byte-identical to a single program_.step_(dt) call. Kept SEPARATE from
  /// install_program so the generated .so ABI is untouched (the cadence is runtime metadata).
  /// NOTE: substeps > 1 is bit-exact vs native substeps ONLY for an UNCOUPLED / transport-only program
  /// (program_.step_ re-runs the whole program, solve_fields included); stride is GLOBAL (whole-system),
  /// equal to native per-block stride only for a single-block system. See System::step.
  POPS_EXPORT void set_program_cadence(int substeps, int stride);
  /// Installed GLOBAL macro-step cadence (ADC-594): the current @c substeps / @c stride the compiled
  /// Program runs at (default 1/1 with no cadence set). Const, side-effect-free -- the structured
  /// ProgramRuntimeReport reads them; there was no Python-visible getter before.
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
  /// Number of blocks (species) installed.
  POPS_EXPORT int n_blocks() const;
  /// The conservative state MultiFab<Dim> of block @p b (zero-copy, non-owning reference).
  POPS_EXPORT MultiFab<Dim>& block_state(int b);
  POPS_EXPORT const MultiFab<Dim>& block_state(int b) const;
  /// @name Compiled-Program NAME-based block binding (Spec 3 criterion 23, ADC-457)
  /// A compiled Program numbers its blocks in P.state declaration order (the .so's
  /// pops_program_block_name table); the System numbers its blocks in add_block / add_equation order
  /// (block_names). They need NOT agree. install_program reads the .so's block names, matches each to
  /// the System block of that name, and stores the resulting program-index -> system-index map here so
  /// ProgramContext::state / rhs_into / commit resolve a Program block index to the name-matched System
  /// block -- NOT the positional index. An EMPTY map means no Program binding is installed;
  /// ProgramContext fails closed instead of inferring positional identity, including for a single
  /// block. Lives in Impl (private to the _pops TU) so it survives the dlopen boundary; the seam is
  /// POPS_EXPORT so the generated .so and ProgramContext resolve it from the globally promoted host.
  /// @{
  /// Install the program-index -> system-index map (entry p = the System block index of Program block
  /// p). Empty clears the binding. Set by install_program after matching the .so's block names.
  POPS_EXPORT void set_program_block_map(const std::vector<int>& prog_to_sys);
  /// The installed program-index -> system-index map (empty = unbound). Read by ProgramContext.
  POPS_EXPORT const std::vector<int>& program_block_map() const;
  /// @}
  /// R <- -div F(U) + S(U, aux) for block @p b (the block's frozen-Poisson residual closure).
  POPS_EXPORT void block_rhs_into(int b, MultiFab<Dim>& U, MultiFab<Dim>& R);
  /// Point-qualified twin used by compiled Programs and native boundary components.
  POPS_EXPORT void block_rhs_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                     int b, MultiFab<Dim>& U, MultiFab<Dim>& R);
  /// R <- -div F(U) for block @p b -- the SAME flux divergence as block_rhs_into but WITHOUT the
  /// model's default/composite source (Poisson frozen, ghosts filled identically). The block's
  /// flux-only closure is the rhs_into path on SourceFreeModel<Model> (the zero-source adapter the
  /// IMEX explicit half-step already uses), so the flux / ghost / geometry handling is bit-identical
  /// -- only the source is dropped (with limiter='none'; the HLL wave-speed cache -- rejected for
  /// compiled Programs -- is the only path where cached cell-center speeds
  /// differ from the per-face reconstruction). A compiled time Program's hyperbolic stage
  /// (ProgramContext::neg_div_flux_default_into) reads it so a Lie/Strang split assembles "flux but no
  /// source" without the default source leaking in (epic ADC-399 / ADC-425, spec criterion 17). FAILS
  /// LOUD (std::runtime_error) on an incomplete internal block provider -- never a silent source leak.
  /// POPS_EXPORT: resolved by the generated problem.so across the
  /// dlopen boundary, like block_rhs_into.
  POPS_EXPORT void block_neg_div_flux_into(int b, MultiFab<Dim>& U, MultiFab<Dim>& R);
  POPS_EXPORT void block_neg_div_flux_into_at(
      const runtime::multiblock::BoundaryEvaluationPoint& point, int b, MultiFab<Dim>& U,
      MultiFab<Dim>& R);
  /// Prepared full flux-only twin of block_neg_div_flux_into_at.  For a physical boundary this
  /// retains its flux boundary law; for a purely periodic distributed block it owns the exact
  /// halo transport session.
  POPS_EXPORT void block_neg_div_flux_into_at_prepared(
      const runtime::multiblock::BoundaryEvaluationPoint& point, int b, MultiFab<Dim>& U,
      MultiFab<Dim>& R, const System* prepared_system, int prepared_block,
      const runtime::multiblock::BoundaryEvaluationPoint& prepared_point, const ExecutionLane& lane,
      const runtime::program::PreparedScalarBoundarySession<Dim>& transport);
  /// Evaluate one simultaneous set of block rates at one exact StagePoint.  Sparse groups are
  /// allowed, but an installed shared interface must have either both sides present or neither.
  POPS_EXPORT void block_rhs_group(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                   const std::vector<int>& blocks,
                                   const std::vector<MultiFab<Dim>*>& states,
                                   const std::vector<MultiFab<Dim>*>& rhs,
                                   const std::vector<int>& flux_only);
  POPS_EXPORT void block_rhs_core_into_at(
      const runtime::multiblock::BoundaryEvaluationPoint& point, int b, MultiFab<Dim>& U,
      MultiFab<Dim>& R, bool flux_only, const System* prepared_system, int prepared_block,
      const runtime::multiblock::BoundaryEvaluationPoint& prepared_point, const ExecutionLane& lane,
      const runtime::program::PreparedScalarBoundarySession<Dim>& transport);
  /// Evaluate one full generated boundary RHS under the exact Program-owned lane and publish it
  /// only after collective finite validation.
  POPS_EXPORT void block_rhs_into_at_prepared(
      const runtime::multiblock::BoundaryEvaluationPoint& point, int b, MultiFab<Dim>& U,
      MultiFab<Dim>& R, const System* prepared_system, int prepared_block,
      const runtime::multiblock::BoundaryEvaluationPoint& prepared_point, const ExecutionLane& lane,
      const runtime::program::PreparedScalarBoundarySession<Dim>& transport);
  /// Whether ordinary Program RHS evaluation must use a prepared session.  A physical boundary
  /// always needs its retained authority; a purely periodic block needs it only when its field is
  /// distributed across more than one rank.
  POPS_EXPORT bool requires_block_boundary_session(int b) const;
  /// Whether one block retains the complete generated boundary residual/JVP pair.  This is a
  /// capability query only; execution still requires one authenticated evaluation point.
  POPS_EXPORT bool has_block_boundary_linearization(int b) const;
  /// Evaluate the generated prepared boundary contribution into a transactionally published
  /// scratch result.  The retained boundary authority is immutable and remains owned by System.
  POPS_EXPORT void block_boundary_residual_into_at(
      const runtime::multiblock::BoundaryEvaluationPoint& point, int b, MultiFab<Dim>& U,
      MultiFab<Dim>& R, const System* prepared_system, int prepared_block,
      const runtime::multiblock::BoundaryEvaluationPoint& prepared_point, const ExecutionLane& lane,
      const runtime::program::PreparedScalarBoundarySession<Dim>& transport);
  /// Apply the generated prepared boundary Jacobian-vector product under the same exact point.
  POPS_EXPORT void block_boundary_jvp_into_at(
      const runtime::multiblock::BoundaryEvaluationPoint& point, int b, MultiFab<Dim>& U,
      const MultiFab<Dim>& direction, MultiFab<Dim>& R, const System* prepared_system,
      int prepared_block, const runtime::multiblock::BoundaryEvaluationPoint& prepared_point,
      const ExecutionLane& lane,
      const runtime::program::PreparedScalarBoundarySession<Dim>& transport);
  /// Fill same-level and physical halos for one generated pointwise stencil through the block's
  /// retained exact-ranked package.  This is a preparation seam, not a second boundary engine.
  POPS_EXPORT void block_prepare_generated_state_at(
      const runtime::multiblock::BoundaryEvaluationPoint& point, int b, MultiFab<Dim>& U);
  POPS_EXPORT void block_prepare_generated_state_at_prepared(
      const runtime::multiblock::BoundaryEvaluationPoint& point, int b, MultiFab<Dim>& U,
      const System* prepared_system, int prepared_block,
      const runtime::multiblock::BoundaryEvaluationPoint& prepared_point, const ExecutionLane& lane,
      const runtime::program::PreparedScalarBoundarySession<Dim>& transport);
  /// R <- S(U, aux) for block @p b -- the model's default/composite SOURCE only, WITHOUT the flux
  /// divergence (the exact MIRROR of block_neg_div_flux_into, which is flux without source). Together
  /// they split block_rhs_into = -div F + S into its two halves (ADC-430, sibling of ADC-425). The
  /// block's source-only closure evaluates m.source per cell into R (the SAME source term assemble_rhs
  /// adds), with NO numerical-flux dispatch -- so it is flux-template agnostic (unlike a zero-flux model
  /// adapter, which HLL/Roe would not zero) and bit-identical to the source term of rhs_into. A compiled
  /// time Program's source stage (ProgramContext::source_default_into) reads it so a Lie/Strang split
  /// assembles "the default source but no flux" -- P.rhs(flux=False, sources with "default") -- without
  /// the -div F base leaking in (epic ADC-399 / ADC-430, spec: rhs flux=False is source-only). FAILS
  /// LOUD (std::runtime_error) on an incomplete internal block provider -- never a silent flux leak.
  /// POPS_EXPORT: resolved by the generated problem.so across the
  /// dlopen boundary, like block_neg_div_flux_into.
  POPS_EXPORT void block_source_into(int b, MultiFab<Dim>& U, MultiFab<Dim>& R);
  /// Prepared local backward-Euler source Newton for compiled block @p b. The install ABI is the
  /// authority for @p options when the Program asks for the block's stored controls.
  POPS_EXPORT SolveOutcome solve_block_source(int b, MultiFab<Dim>& U, Real dt,
                                              const NewtonOptions& options);
  [[nodiscard]] POPS_EXPORT NewtonOptions block_newton_options(int b) const;
  [[nodiscard]] bool block_newton_diagnostics(int b) const;
  POPS_EXPORT void publish_newton_report(int b, const SolveReport& solve);
  [[nodiscard]] const NewtonReport& last_newton_report() const;
  /// Preflight one unqualified Cartesian generated Program operator.  Such kernels cannot evaluate
  /// inactive cells before zeroing their outputs, so an active embedded boundary is rejected before
  /// mutation.  Only unqualified Cartesian kernels use this seam; local_transform and
  /// solve_local_nonlinear are mask-qualified.
  POPS_EXPORT void require_cartesian_generated_operator(int b, const std::string& operation) const;
  /// The maximum |wave speed| of block @p b evaluated on @p U -- the SAME per-block reduction
  /// step_cfl reads (BlockState::max_speed, the HasStabilitySpeed / max_wave_speed closure set at
  /// add_block time): a collective reduction over the block's cells. This Uniform entry point uses
  /// the System's authenticated execution lane; generated Programs pass that lane explicitly through
  /// the private prepared seam below. REUSES the block's wave-speed closure -- it does not recompute
  /// the speed. POPS_EXPORT: resolved across the dlopen boundary like the other seam accessors.
  POPS_EXPORT Real block_max_speed(int b, const MultiFab<Dim>& U) const;
  /// The minimum physical cell spacing across every compiled axis -- the same hmin the native CFL
  /// uses (System::step_cfl). A compiled time Program reads it
  /// (ProgramContext::hmin) to express its own dt bound (epic ADC-399 / ADC-417, spec s18). POPS_EXPORT:
  /// resolved by the generated problem.so across the dlopen boundary.
  POPS_EXPORT Real cfl_min_dx() const;
  /// A collective scalar reduction over a NAMED block's state -- the native seam the Python diagnostics
  /// driver drives to fire a declared typed measure (Norm / Integral / MinMax) each cadence tick
  /// (ADC-542). @p kind selects the reduction over the block's U: per-component
  /// "sum" / "min" / "max" / "abs_sum" (L1) / "sum_sq" (L2 squared, dot(u,u)) / "abs_max" (LInf); the
  /// full-state variants "sum_all" / "abs_sum_all" / "sum_sq_all" / "abs_max_all" fold over ALL
  /// components. @p comp is the component for the per-component kinds (ignored by the _all kinds). An
  /// unknown @p block or @p kind throws (fail loud, never a silent 0). COLLECTIVE, MANDATORY UNDER MPI:
  /// called on every rank (empty ranks included), like dot. POPS_EXPORT: resolved across the dlopen
  /// boundary like the other seam accessors.
  POPS_EXPORT double reduce_component(const std::string& block, const std::string& kind,
                                      int comp) const;
  /// A fresh scalar field co-distributed with the System mesh: block 0's BoxArray and
  /// ranked ownership layout, @p n_comp components, @p n_ghost ghost layers, zero-initialized. Scratch a
  /// compiled time Program allocates for a matrix-free Krylov solve (the residual / search-direction
  /// fields owned by a KrylovWorkspace and fed through ProgramContext::laplacian); shares the block
  /// (ba, dm) so a per-cell kernel pairs it with the state and aux by local fab index.
  POPS_EXPORT MultiFab<Dim> alloc_scalar_field(int n_comp, int n_ghost);
  /// @name Multistep history (epic ADC-399 / ADC-406a)
  /// SYSTEM-OWNED history ring buffers for multistep schemes (Adams-Bashforth and friends): a named
  /// field carried ACROSS macro-steps (e.g. the previous RHS R_{n-1}). The history lives in the System
  /// (a HistoryManager in Impl), NOT in the .so closure, so a later checkpoint slice (ADC-406b) can
  /// serialize it. A generated problem.so reaches it through ProgramContext::history / store_history /
  /// rotate_histories; these are POPS_EXPORT so the .so resolves them across the dlopen boundary.
  /// @{
  /// Register (idempotent) a history named @p name with maximum lag @p lag (>= 1): a ring buffer of
  /// depth @p lag + 1 (slot 0 = the CURRENT value, slot k = the value k macro-steps back after the
  /// rotates), each slot a zero-initialized MultiFab<Dim> on the shared block layout. Qualified calls bind
  /// the exact owner plus logical state/space/clock/interpolation identities; unqualified calls retain
  /// the legacy owner=-1 contract and cannot use selective replay. @p ncomp is the slot component
  /// count: the default -1 resolves to the qualified owner's ncomp (or block 0 for a legacy ring), while an
  /// explicit @p ncomp >= 1 sizes a narrower ring (ADC-427: the 1-component condensed-Schur phi^n
  /// carry). The component count binds at the FIRST register; a later re-register ignores @p ncomp.
  /// Re-registering returns the existing current slot and grows the ring for a larger @p lag. Returns
  /// the current slot [0] -- the read target for lag = 1 after one rotate. @throws if @p lag < 1,
  /// @p ncomp == 0, or no block exists yet.
  POPS_EXPORT MultiFab<Dim>& register_history(const std::string& name, int lag, int ncomp = -1,
                                              int owner = -1,
                                              const std::string& state_identity = "",
                                              const std::string& space_identity = "",
                                              const std::string& clock_identity = "",
                                              const std::string& interpolation_identity = "");
  /// The history slot @p lag macro-steps back (lag 0 = the current slot, lag 1 = the previous step's
  /// stored value, ...). @throws if @p name is unknown, @p lag exceeds the registered depth, or the
  /// history has not been stored yet ("history '<name>' with lag=<lag> was requested but not
  /// initialized") -- a read before the first store is a fail-loud configuration error (spec error 17).
  POPS_EXPORT MultiFab<Dim>& read_history(const std::string& name, int lag);
  /// Copy @p value (valid cells) into the CURRENT slot [0] of history @p name and mark it initialized.
  /// On the FIRST store the value is also broadcast into EVERY deeper slot (the cold-start fill: a
  /// multistep scheme's step 0 then reads the same value at every lag, degenerating to a one-step
  /// method -- deterministic and machine-precision reproducible). @throws if @p name is unknown. The
  /// caller is responsible for layout compatibility: the ring slots share the block's (ba, dm, ncomp),
  /// so a value built from the same block matches (lincomb is a valid-cell copy, no layout check).
  POPS_EXPORT void store_history(const std::string& name, const MultiFab<Dim>& value);
  /// Qualified generated-Program route: identical to the overload above, but records the exact
  /// outgoing interval of the active logical clock. A child-clock subcycle therefore owns child
  /// timestamps rather than inheriting the enclosing macro dt. @p outgoing_dt must be finite and
  /// non-negative; generated Program scopes always provide a strictly positive value.
  POPS_EXPORT void store_history(const std::string& name, const MultiFab<Dim>& value,
                                 double outgoing_dt);
  /// Shift every history ring buffer one step (slot k <- slot k-1, for k = depth-1 .. 1), called ONCE
  /// at the end of each macro-step (the generated step body emits ctx.rotate_histories() last). The
  /// current slot [0] is recycled (it gets the oldest buffer; the next store overwrites it before any
  /// read). O(1) handle swaps -- no deep copy. No-op when no history exists.
  POPS_EXPORT void rotate_histories();
  POPS_EXPORT void rotate_histories(const std::string& clock_identity);
  /// @name Multistep history checkpoint/restart (epic ADC-399 / ADC-406b)
  /// SERIALIZE / RESTORE the System-owned history rings across a checkpoint. The history lives in the
  /// System (HistoryManager in Impl), so the checkpoint facade (sim.checkpoint / sim.restart) gathers
  /// and restores it DIRECTLY -- no .so checkpoint_extra ABI is needed for the buffers (only the program
  /// hash, installed_program_hash() below, is recorded to reject a restart against a different Program).
  /// @{
  /// Names of every registered history (the keys of the HistoryManager), so the facade can iterate the
  /// rings to serialize. Empty when no history exists (the single-step paths). Order is the map order.
  POPS_EXPORT std::vector<std::string> history_names() const;
  /// Ring depth (max lag + 1) of history @p name. @throws if @p name is unknown.
  POPS_EXPORT int history_depth(const std::string& name) const;
  /// Component count of the slots of history @p name (the block's ncomp). @throws if unknown.
  POPS_EXPORT int history_ncomp(const std::string& name) const;
  /// GLOBAL (collective, MPI-safe) gather of slot @p slot (0 = current, k = k macro-steps back) of
  /// history @p name into a component-major buffer of size ncomp times the product of every exact
  /// spatial extent, EXACTLY like state_global
  /// (every rank fills its local boxes then all_reduce_sum). All ranks MUST call it. @throws if @p name
  /// is unknown or @p slot is out of range. Reads the slot even before the first store (the checkpoint
  /// of a never-stored ring is its zero fill); the initialized flag is serialized separately.
  POPS_EXPORT std::vector<double> history_global(const std::string& name, int slot) const;
  /// Whether history @p name has been stored at least once (the cold-start fill already happened). The
  /// facade records it so a restart restores the initialized state without a phantom re-fill. @throws
  /// if @p name is unknown.
  POPS_EXPORT bool history_initialized(const std::string& name) const;
  /// Saturating count of authentic accepted stores represented in the ring (0..history_depth).
  /// Cold-start copies do not advance this count; selective persistence is safe only at full depth.
  POPS_EXPORT int history_fill_count(const std::string& name) const;
  /// RESTORE (restart) slot @p slot of history @p name from a GLOBAL component-major buffer (same layout
  /// as history_global / set_state): the owner rank writes its box, the others are no-ops (MPI-safe, all
  /// ranks call it). Registers the ring (depth = max(slot)+1) if @p name is unknown yet, so the restart
  /// rebuilds the rings the program will re-register on its first step. @throws on a size mismatch.
  POPS_EXPORT void restore_history(const std::string& name, int slot,
                                   const std::vector<double>& values);
  /// Mark history @p name initialized (or not) after a restart: a restored, already-stored ring must
  /// read at lag without a phantom cold-start re-fill on its first post-restart store. @throws if
  /// @p name is unknown (restore its slots first).
  POPS_EXPORT void set_history_initialized(const std::string& name, bool initialized);
  /// Restore the exact authentic fill count persisted by a checkpoint. Also restores the derived
  /// initialized flag (`fill_count > 0`). @throws if outside [0, history_depth].
  POPS_EXPORT void restore_history_fill_count(const std::string& name, int fill_count);
  /// @}
  /// @name Selective history persistence + deterministic ring replay (ADC-626)
  /// A history-persistence policy (pops.time.Dense / Interval / Revolve) stores only a SUBSET of a
  /// ring's slots in a checkpoint; the restart REBUILDS the missing slots by re-stepping the installed
  /// Program. The per-slot outgoing dt is exposed so the checkpoint records the exact interval
  /// between adjacent state samples and replay reproduces a variable-dt history bit-for-bit.
  /// @{
  /// The outgoing dt from slot @p slot toward its newer neighbour (HistoryManager::slot_dt). 0 for a
  /// slot that was never stored (a never-stepped ring). @throws if @p name is unknown or @p slot out
  /// of range.
  POPS_EXPORT double history_slot_dt(const std::string& name, int slot) const;
  /// Restore the outgoing dt recorded with slot @p slot of history @p name (the inverse of
  /// history_slot_dt, used at restart so replay re-steps with the exact interval from an older anchor
  /// to its newer neighbour). Grows the per-slot dt vector to fit the ring. @throws if @p name is
  /// unknown (restore its slots first).
  POPS_EXPORT void restore_history_slot_dt(const std::string& name, int slot, double dt);
  /// REBUILD the MISSING slots of history @p name by deterministic replay (ADC-626). @p stored_slots is
  /// the sorted set of slot indices already restored (via restore_history); every OTHER slot in
  /// [0, depth) is reconstructed by seeding a SCRATCH block state from the nearest OLDER stored slot and
  /// re-stepping the installed Program forward (program_.step_) with the recorded per-slot dt, capturing
  /// each intermediate state into its ring slot. The live block state U and the scheduler cache are
  /// SAVE/RESTORE-bracketed so replay is identity on them (bit-for-bit). Slots are placed BY INDEX (no
  /// rotate), sidestepping the rotation-invalidation edge. Requires an installed Program (program_.step_)
  /// and at least the oldest slot (depth-1) present in @p stored_slots. Returns the number of slots
  /// RECOMPUTED (== depth - stored_slots.size()); the replay-step count equals it (each missing slot is
  /// captured once as a contiguous run passes it). @throws if @p name is unknown, no Program is
  /// installed, or the oldest slot is not stored (the ring would be unreconstructable).
  POPS_EXPORT int rebuild_history_slots(const std::string& name,
                                        const std::vector<int>& stored_slots);
  /// @}
  /// Load a generated problem.so and install its compiled time Program. dlopens @p so_path, checks
  /// its ABI key against this module (fail-loud on mismatch), and calls its pops_install_program(this),
  /// whose shared facade factory selects the Program execution provider and installs the macro-step
  /// closure. The .so resolves the seam accessors above from the globally promoted host, while the
  /// package itself stays local so independent semantic artifacts cannot interpose. Mirrors
  /// add_native_block; the .so stays loaded for the process lifetime.
  POPS_EXPORT void install_program(const std::string& so_path);
  /// IR hash of the installed compiled Program (the string returned by the .so's pops_program_hash),
  /// or "" if no program is installed. Recorded in the checkpoint (sim.checkpoint) so a restart against
  /// a DIFFERENT compiled Program is rejected fail-loud (the buffers / cadence would be meaningless).
  POPS_EXPORT std::string installed_program_hash() const;

  /// @name Runtime freeze lifecycle (ADC-592)
  /// The runtime lifecycle is EXPLICIT: assembly mutable BEFORE bind, composition FROZEN once
  /// pops.bind completes, simulation mutable only through controlled runtime APIs (state data,
  /// runtime params, checkpoint/restart, diagnostics, output). mark_bound() is the ONE transition
  /// into the frozen state; it is called LAST by the Python bind flow (after every install call), so
  /// the install sequence itself never trips the structural-setter guards. A direct engine script
  /// that never binds (the C++ tests, the low-level runtime seam) is UNAFFECTED -- bound_ stays false
  /// until mark_bound() runs, so the historical setters keep working.
  /// @{
  /// Mark the composition as bound (frozen): every structural setter (add_block / set_poisson /
  /// install_program / set_analytic_level_set / ...) then rejects with a precise error.
  /// Runtime-data publication through exact state, field, parameter and ComponentKey provider
  /// authorities stays allowed. A second mark_bound() throws (a composition binds exactly once).
  void mark_bound();
  /// The runtime lifecycle state: "assembling" (not yet bound -- the composition is mutable),
  /// "bound" (mark_bound() ran, no macro-step advanced yet), "running" (bound AND macro_step() > 0).
  std::string lifecycle_state() const;
  /// @}
  /// @name Scheduler value cache (epic ADC-399 / ADC-458, Spec 3 section 17-18 + 30)
  /// The held-node value cache (every(N).hold / accumulate_dt) lives in the SYSTEM (one CacheManager
  /// per installed Program), NOT in the .so step closure -- so the checkpoint can reach it, exactly the
  /// way the history rings do. Every ProgramContext (the step closure's copy and any fresh one) forwards
  /// its cache_* seam ops to this single manager. POPS_EXPORT so the generated problem.so resolves it
  /// across the dlopen boundary like the other ProgramContext seam accessors.
  /// @{
  /// The System-owned scheduler cache (a non-owning reference; lives as long as the System). A compiled
  /// Program's cache_store_aux / cache_restore_aux / cache_should_update reach it through ProgramContext.
  POPS_EXPORT runtime::program::CacheManager<Dim>& program_cache();
  /// @name Scheduler-cache checkpoint/restart (Spec 3 section 30, ADC-458)
  /// SERIALIZE / RESTORE the System-owned cache across a checkpoint, mirroring the history seam: the
  /// facade (sim.checkpoint / sim.restart) gathers each VALID slot (gather_global, MPI-safe) and scatters
  /// it back (write_state) alongside the block state and histories. The program-hash guard
  /// (installed_program_hash) rejects a restart against a different compiled Program; a held scheduled
  /// node the checkpoint never recorded fails loud at restart (the facade compares the restored ids).
  /// @{
  /// Node ids of every VALID cached slot (ascending). Empty when no schedule cached a value.
  POPS_EXPORT std::vector<int> program_cache_nodes() const;
  /// The scheduled node name of slot @p node_id ("fields_from_state"), or "node_<id>" if it was stored
  /// without one (the current nameless codegen). Names a missing node verbatim at restart.
  POPS_EXPORT std::string program_cache_name(int node_id) const;
  /// The macro step at slot @p node_id's last recompute. @throws if absent.
  POPS_EXPORT int program_cache_last_update_step(int node_id) const;
  /// The accumulated skipped dt of slot @p node_id (accumulate_dt policy). 0 if none.
  POPS_EXPORT double program_cache_accumulated_dt(int node_id) const;
  /// The component count of slot @p node_id's cached value. @throws if absent.
  POPS_EXPORT int program_cache_ncomp(int node_id) const;
  /// The uniform ghost-cell width of slot @p node_id's cached value (1 for the aux, the block-state
  /// width for a held scratch) -- serialized so restore rebuilds with the same width on every exact
  /// axis. @throws if absent or if an anisotropic extent cannot be represented by this checkpoint
  /// schema.
  POPS_EXPORT int program_cache_ngrow(int node_id) const;
  /// GLOBAL (collective, MPI-safe) gather of slot @p node_id's cached MultiFab<Dim> into a component-major
  /// buffer of size ncomp times the product of every exact spatial extent, EXACTLY like
  /// state_global / history_global. All ranks MUST call it.
  /// @throws if @p node_id is absent.
  POPS_EXPORT std::vector<double> program_cache_global(int node_id) const;
  /// RESTORE (restart) slot @p node_id from a GLOBAL component-major buffer (same layout as
  /// program_cache_global / set_state): allocate a value MultiFab<Dim> co-distributed with block 0 (@p ncomp
  /// components), scatter the buffer into it (owner rank writes, others no-op -- MPI-safe, all ranks
  /// call it), and re-key the slot with its bookkeeping (@p name may be empty). @throws if no block
  /// exists yet (the cache value is co-distributed with block 0's storage).
  POPS_EXPORT void restore_program_cache(int node_id, int ncomp, int ngrow, int last_update_step,
                                         double accumulated_dt, const std::string& name,
                                         const std::vector<double>& values);
  /// @}
  /// @}
  /// Apply block @p b's post-step positivity projection to @p u in place (ADC-177): U <- project(U,
  /// aux) over the valid cells, the SAME closure the native per-step path runs (s.project). A compiled
  /// time Program reaches it through ProgramContext::apply_projection (spec op 21). REUSES the block's
  /// own projection (set at add_block time); a block without that capability is rejected.
  /// POPS_EXPORT so a generated problem.so resolves it across the dlopen boundary.
  POPS_EXPORT void block_project(int b, MultiFab<Dim>& u);
  /// @name Compiled-Program scalar diagnostics (epic ADC-399 / ADC-414, spec op 23)
  /// A name -> Real map a compiled Program writes via P.record_scalar (ProgramContext::record_scalar),
  /// retrievable AFTER sim.step for inspection / logging. Lives in Impl (private to the _pops TU) so it
  /// survives across the dlopen boundary; the .so writes it through the POPS_EXPORT setter below.
  /// @{
  /// Store @p value under @p name (overwrites a prior value of the same name). Called by the installed
  /// program closure each step. POPS_EXPORT: the generated problem.so resolves it from the globally
  /// promoted host while the generated package remains local.
  POPS_EXPORT void record_program_diagnostic(const std::string& name, Real value);
  /// The recorded value of diagnostic @p name. @throws std::out_of_range if @p name was never
  /// recorded (a typo / a diagnostic the installed program does not write fails loud, not 0).
  POPS_EXPORT Real program_diagnostic(const std::string& name) const;
  /// All recorded diagnostics (name -> last recorded value). Empty when the program records none.
  /// Exposed to Python as sim.program_diagnostics() (a dict); program_diagnostic(name) reads one.
  POPS_EXPORT std::map<std::string, Real> program_diagnostics() const;
  /// Five current-attempt scalars for one typed balance route. RuntimeInstance calls this only
  /// inside its active outer accepted-step transaction; missing/stale/non-finite evidence fails.
  POPS_EXPORT std::map<std::string, Real> accepted_balance_terms(const std::string& route) const;
  /// The same accepted route with selected attempt-local native reflux/projection producers.
  POPS_EXPORT std::map<std::string, Real> selected_accepted_balance_terms(
      const std::string& route, const std::string& block, int component,
      const std::vector<int>& levels, const std::vector<std::string>& automatic_terms) const;
  POPS_EXPORT void begin_step_projection_report();
  POPS_EXPORT void note_step_projection(const std::string& name);
  POPS_EXPORT std::vector<std::string> consume_step_projections();
  /// @}
  /// @name Compiled-Program RUNTIME parameters (epic ADC-479 / ADC-510, Spec 5 C5)
  /// A compiled time Program whose physics reads a dsl.Param(..., kind="runtime") carries the value
  /// in a per-PROGRAM-block RuntimeParams owned HERE (not in the .so closure), so set_program_params
  /// changes it at run time WITHOUT recompiling. Mirrors the program diagnostics / history rings:
  /// System-owned state the step closure
  /// reaches through ProgramContext. install_program seeds each block's defaults from the .so
  /// pops_program_param_* metadata. The lowered source / linear-source kernels read the CURRENT value
  /// via ProgramContext::program_params(block).get(index).
  /// @{
  /// Overwrite block @p prog_block's RuntimeParams with @p values (the COMPLETE block, sorted-name
  /// order matching the .so pops_program_param_* metadata). @p prog_block is the PROGRAM block index
  /// (P.state declaration order). @throws std::out_of_range if the block was not seeded by a runtime-
  /// param Program, std::runtime_error on a size mismatch. POPS_EXPORT: a generated problem.so could
  /// reach it across the dlopen boundary (the runtime set comes from Python today). Effect on the next
  /// step.
  POPS_EXPORT void set_program_params(int prog_block, const std::vector<double>& values);
  /// Block @p prog_block's CURRENT RuntimeParams (a device-clean by-value copy: trivially copyable,
  /// readable in a kernel). An UNSEEDED block (no runtime param declared) returns a default-constructed
  /// RuntimeParams (count 0), so a kernel that reads no param is unaffected. Read by ProgramContext for
  /// the lowered per-cell source / linear-source kernels.
  POPS_EXPORT RuntimeParams program_params(int prog_block) const;
  /// Seed block @p prog_block's RuntimeParams to its DECLARATION defaults (@p count values, the .so
  /// pops_program_param_default metadata), establishing the no-set baseline. Called by install_program
  /// once per runtime-param Program block; a later set_program_params overwrites only the supplied
  /// values. Idempotent (re-seeding resets to defaults).
  POPS_EXPORT void seed_program_params(int prog_block, const std::vector<double>& defaults);
  /// @}
  /// @}

  /// @name Diagnostics
  /// @{
  Extent<Dim> spatial_shape() const;
  /// MACRO-STEP counter (0-indexed; incremented by step / step_cfl). Necessary
  /// for checkpoint/restart: the stride cadence (hold-then-catch-up) depends on macro_step % stride,
  /// not only on the time t (accepted-state restart). POPS_EXPORT: a scheduled (every(N)/hold) program
  /// `.so` calls it for the cadence decision, so it must be in the loader's flat ABI like the other
  /// seam accessors (grid_context / solve_fields_from_state); without it the held-schedule `.so`
  /// fails to dlopen ("symbol not found in flat namespace"), caught by the Spec 3 runtime e2e test.
  POPS_EXPORT int macro_step() const;
  /// RESTORES the clock (t, macro_step) -- reserved for the RESTART (sim.restart). Restoring macro_step
  /// is MANDATORY to resume the stride cadence exactly; a restart that would only restore
  /// t would desynchronize the blocks at stride > 1. A mid-window cursor also requires an immediately
  /// preceding restore_program_cadence_window; its start and accumulated variable-dt duration cannot
  /// be inferred from the clock. @throws if macro_step < 0 or the stride-window state is invalid.
  POPS_EXPORT void set_clock(double t, int macro_step);
  /// Generated Program shared libraries read the accepted clock through the flat loader ABI.
  POPS_EXPORT double time() const;
  int n_species() const;
  /// Block names, in the order of addition. SINGLE SOURCE: the internal block registry, populated by
  /// ALL the addition paths (add_block / install_block). An integrator written in Python iterates
  /// over it, so it must also see blocks installed from a production package.
  std::vector<std::string> block_names() const;
  /// Structured report of effective numerical, solver and physical options currently configured.
  EffectiveOptionsReport effective_options_report() const;
  double mass(const std::string& name) const;
  std::vector<double> density(const std::string& name) const;  ///< native index order
  std::vector<double> potential();                             ///< phi, native index order
  /// RESTORES the potential phi (accepted-state restart): without it the multigrid would restart from
  /// a blank phi and the resume would not be bit-identical (warm start lost). The field uses the
  /// same exact-ranked flattened layout as potential().
  void set_potential(const std::vector<double>& phi);
  std::vector<std::string> field_provider_slots() const;
  /// Read-only restart authority. Named identities match ``field_provider_slots`` exactly and in
  /// order. The default slot is included when the installed prepared RHS/configuration can
  /// materialize that exact field, even if it has not been instantiated yet. This query never
  /// constructs ExactNamedField.
  std::vector<std::string> configured_field_provider_slots() const;
  void set_field_potential(const std::string& provider_slot, const std::vector<double>& phi);

  /// @name GLOBAL accessors (MPI-safe collectives) -- outputs / multi-rank accepted-state checkpoint
  /// A System uses the exact box decomposition carried by SystemConfig; an omitted decomposition
  /// materializes one box covering the whole domain, while RegularBlocks supplies multiple boxes
  /// distributed by the prepared load-balancing authority. The non-global accessors above are
  /// owner-local and therefore invalid on a rank without the requested storage. The _global
  /// variants fill a GLOBAL buffer from all LOCAL fabs (in GLOBAL indices; nothing on an empty rank)
  /// then all_reduce_sum_inplace -> EACH
  /// rank holds the complete field (AMR reflux pattern, comm.hpp). They are COLLECTIVE: all the
  /// ranks MUST call them. On mono-rank they return EXACTLY the same array as the non-global
  /// accessors (all_reduce = identity, box = complete domain) -> bit-identical output.
  /// RuntimeInstance uses them for accepted-state checkpoint capture, then seals and publishes
  /// the single artifact only on rank 0.
  /// @{
  std::vector<double> density_global(
      const std::string& name) const;  ///< comp0, global cell product
  std::vector<double> state_global(
      const std::string& name) const;      ///< U, ncomp*global cell product
  std::vector<double> potential_global();  ///< phi, global cell product
  std::vector<double> field_potential_global(const std::string& provider_slot);
  /// Unified writer getters. Uniform layouts have exactly level zero; other levels fail loudly.
  /// Local pieces preserve native ranked ownership and never gather.
  std::vector<OutputPiece<Dim>> output_state_local_pieces(const std::string& name, int level) const;
  std::vector<OutputPiece<Dim>> output_field_local_pieces(const std::string& provider_slot,
                                                          int level);
  /// Exact embedded-boundary geometry sidecars.  These arrays are never appended to a physical
  /// state or field payload: each reserved name selects one scalar native field on the same
  /// (layout, level) ownership map.  Uniform layouts expose only level zero.
  ///
  /// Supported names are ``pops_active`` (binary cell mask), ``pops_phi`` (signed level set), and
  /// ``pops_kappa`` (cell volume fraction).  A System without a prepared embedded boundary fails
  /// explicitly so a non-owning MPI rank cannot be confused with an absent sidecar.
  std::vector<OutputPiece<Dim>> output_embedded_boundary_local_pieces(const std::string& name,
                                                                      int level) const;
  /// Collective ROOT views.  Local provider errors are agreed before native MPI_Gatherv; only rank
  /// zero receives complete pieces and every non-root rank receives an empty vector.
  std::vector<OutputPiece<Dim>> output_state_root_pieces(const ObserverMpiLane& lane,
                                                         const std::string& name, int level) const;
  std::vector<OutputPiece<Dim>> output_field_root_pieces(const ObserverMpiLane& lane,
                                                         const std::string& provider_slot,
                                                         int level);
  std::vector<OutputPiece<Dim>> output_embedded_boundary_root_pieces(const ObserverMpiLane& lane,
                                                                     const std::string& name,
                                                                     int level) const;
  /// @}

  /// @name LOCAL per-fab accessors -- exact native ownership inspection
  /// Local counterpart of the _global accessors: instead of gathering the whole field by
  /// all_reduce_sum, they expose per rank the list of LOCAL boxes and the state of EACH fab. The
  /// typed scientific-output providers consume the unified OutputPiece API above; these lower-level
  /// views remain useful for native ownership verification. They are NON COLLECTIVE (purely
  /// local: no MPI comm; a rank without a box returns an empty list). A default Cartesian System
  /// has one full-domain box; a RegularBlocks decomposition has every configured box distributed
  /// across the prepared rank space. The API iterates over all local fabs and retains GLOBAL indices
  /// in each box. Layout of local_state is IDENTICAL to state_global but
  /// relative to the local box: component-major with the final native axis contiguous.
  /// @{
  std::vector<Box<Dim>> local_boxes(const std::string& name) const;
  std::vector<double> local_state(const std::string& name,
                                  int li) const;  ///< U of fab li, flat (ncomp*box.numPts())
                                                  /// @}
                                                  /// @}

 private:
  friend class runtime::program::ProgramContext<Dim>;
  friend class PreparedSystemLayoutTransfer<Dim>;
  /// Dedicated generated-Program sink for one validated, attempt-local balance term. It remains
  /// private to ProgramContext and is deliberately absent from Python bindings.
  POPS_EXPORT void record_program_balance_term(const std::string& route, const std::string& term,
                                               Real value);
  POPS_EXPORT bool program_balance_consumer_is_due(const std::string& contract,
                                                   const std::string& route, int every_n) const;
  POPS_EXPORT runtime::program::ProgramRuntimeState<Dim>& program_runtime_state_();
  /// Program-terminal publication validator. Recovers only active valid cells and returns the
  /// authenticated prepared active-cell mask for this block/layout/lane, or null when the
  /// candidate is Cartesian or the embedded boundary is inactive.
  [[nodiscard]] POPS_EXPORT const MultiFab<Dim>* validate_program_state_publication_candidate_(
      int block, const MultiFab<Dim>& candidate, const ExecutionLane& lane) const;
  /// Exact-lane maximum-speed seam for generated ProgramContext. Local block/provider/allocation
  /// failures converge before the closure's scalar reduction; no implicit WORLD lane is permitted.
  POPS_EXPORT Real block_max_speed_prepared_(int block, const MultiFab<Dim>& state,
                                             const ExecutionLane& lane) const;
  /// Generated Uniform pointwise kernels obtain their optional embedded-boundary active mask only
  /// through this authenticated, non-owning seam.  It is private so the stable mask address cannot
  /// become a public publication route.
  [[nodiscard]] POPS_EXPORT const MultiFab<Dim>* prepared_program_block_active_mask_(
      int runtime_block, const MultiFab<Dim>& field, const ExecutionLane& lane) const;
  /// Immediate provider calls are an exported implementation seam for generated ProgramContext
  /// code, never a public publication route. Every public field solve and every Program solve wraps
  /// these methods in the same physical accepted/candidate transaction.
  POPS_EXPORT SolveReport solve_fields_in_place_();
  POPS_EXPORT SolveReport solve_fields_from_state_in_place_(int block_idx,
                                                            const MultiFab<Dim>& U_stage);
  POPS_EXPORT SolveReport solve_fields_from_state_at_in_place_(
      const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
      int block_idx, const MultiFab<Dim>& U_stage);
  POPS_EXPORT SolveReport
  solve_fields_from_blocks_in_place_(const std::vector<const MultiFab<Dim>*>& U_stages);
  POPS_EXPORT SolveReport solve_fields_from_state_in_place_(const std::string& field, int block_idx,
                                                            const MultiFab<Dim>& U_stage);
  POPS_EXPORT SolveReport solve_fields_from_blocks_in_place_(
      const std::string& field, const std::vector<const MultiFab<Dim>*>& U_stages);
  POPS_EXPORT SolveReport solve_fields_from_blocks_at_in_place_(
      const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& field,
      const std::vector<const MultiFab<Dim>*>& U_stages);
  POPS_EXPORT void prepare_default_field_publication_storage_();
  POPS_EXPORT void prepare_named_field_publication_storage_(const std::string& field);
  POPS_EXPORT void begin_field_publication_transaction();
  POPS_EXPORT void stage_field_publication_candidate();
  POPS_EXPORT void validate_field_publication_candidate();
  POPS_EXPORT void accept_field_publication_candidate() noexcept;
  POPS_EXPORT void rollback_field_publication_transaction();
  [[nodiscard]] POPS_EXPORT bool field_publication_transaction_active_() const noexcept;
  POPS_EXPORT void begin_field_publication_outcome_();
  POPS_EXPORT SolveOutcome stage_field_publication_outcome_(SolveReport report);
  SolveOutcome run_field_publication_outcome_(const std::function<SolveReport()>& solve);
  enum class NativePackageKind { generic, prepared_boundary };
  void stage_native_package_(
      std::string identity, std::function<void()> route_registrar, std::function<void()> installer,
      std::shared_ptr<void> package_lifetime,
      std::shared_ptr<runtime::system::NativePackageCapabilityState<Dim>> capability,
      NativePackageKind kind);
  void seal_auxiliary_providers_(const CommunicatorView& communicator);
  /// Read-only compiled-artifact capability check.  Kept private so only ProgramContext can issue
  /// an authenticated apply token; installation writes Impl directly and no public setter exists.
  POPS_EXPORT bool program_owns_operator_authority(
      const std::array<std::uint64_t, 4>& authority) const noexcept;
  void add_coupled_source_prepared_(const CoupledSourceProgram& program, double frequency,
                                    const std::string& label, CouplingOperatorView inspect);
  struct Impl;
  // Declared before Impl so installed Program closures and their immutable lane borrows are
  // destroyed before the owning communicator is released.
  std::shared_ptr<ExecutionLane> prepared_boundary_execution_lane_;
  std::unique_ptr<Impl> p_;
};

/// Persistent System-to-System transfer session with no per-step field allocation or Python staging.
///
/// Preparation authenticates both layouts, blocks, representations, provider and execution
/// context collectively, allocates one source snapshot distributed like the target, and warms its
/// native MPI/Kokkos copy plan. During a transaction every mapping captures before any mapping is
/// applied, preserving cycles such as A->B->C->A without Python array materialization. The
/// enclosing System transactions own target rollback; this object owns the native source snapshot
/// and strict generation/attempt protocol.
template <int Dim>
class POPS_EXPORT PreparedSystemLayoutTransfer final {
  static_assert(Dim >= 1 && Dim <= 3,
                "PreparedSystemLayoutTransfer only supports dimensions 1, 2, and 3");

 public:
  PreparedSystemLayoutTransfer(const PreparedSystemLayoutTransfer&) = delete;
  PreparedSystemLayoutTransfer& operator=(const PreparedSystemLayoutTransfer&) = delete;
  ~PreparedSystemLayoutTransfer();

  static std::shared_ptr<PreparedSystemLayoutTransfer> prepare(
      System<Dim>& source, System<Dim>& target,
      std::shared_ptr<component::LoadedComponent> component, SystemLayoutTransferSpec<Dim> spec,
      SystemLayoutTransferExecution execution);

  const SystemLayoutTransferSpec<Dim>& spec() const noexcept;
  void begin_transaction(std::uint64_t generation);
  void capture(std::uint64_t generation, std::uint64_t attempt);
  SystemLayoutTransferReceipt apply(std::uint64_t generation, std::uint64_t attempt);
  void reject_attempt(std::uint64_t generation, std::uint64_t attempt);
  void finalize_transaction(std::uint64_t generation) noexcept;
  void rollback_transaction(std::uint64_t generation) noexcept;

 private:
  struct Impl;
  explicit PreparedSystemLayoutTransfer(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> p_;
};

}  // namespace pops
