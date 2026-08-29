/// @file
/// @brief Private compile-time-ranked storage shared by the System implementation TUs.

#pragma once

#include <pops/runtime/system.hpp>

#include <pops/runtime/system/system_block_store.hpp>
#include <pops/runtime/system/system_boundary_registry.hpp>
#include <pops/runtime/system/system_coupling_registry.hpp>
#include <pops/runtime/system/system_domain.hpp>
#include <pops/runtime/system/exact_aux_registry.hpp>
#include <pops/runtime/system/auxiliary_ghost_fill.hpp>
#include <pops/runtime/system/exact_named_field.hpp>
#include <pops/runtime/system/prepared_embedded_boundary.hpp>
#include <pops/runtime/system/system_lifecycle.hpp>
#include <pops/runtime/system/native_package_capability.hpp>
#include <pops/runtime/program/program_runtime_state.hpp>
#include <pops/runtime/program/program_transaction.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_builtins.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_prepare.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops {

inline void require_assembling(const runtime::system::SystemLifecycle& lifecycle,
                               const char* operation) {
  if (lifecycle.frozen())
    throw std::runtime_error(
        std::string("System::") + operation +
        ": the composition is frozen once pops.bind completes; create a fresh runtime to change "
        "its structure");
}

template <int Dim>
struct System<Dim>::Impl {
  using domain_type = runtime::system::SystemDomain<Dim>;
  using block_store_type = SystemBlockStore<Dim>;
  using boundary_registry_type = runtime::system::SystemBoundaryRegistry<Dim>;
  using field_type = MultiFab<Dim>;
  using Species = typename block_store_type::BlockState;

  struct PreparedBoundaryHookContract {
    std::string package_identity;
    std::string block;
    std::string hook;
    std::string component_contract;
    std::string component_authority_contract;

    bool operator==(const PreparedBoundaryHookContract&) const = default;
  };

  /// One validated native package.  Package owners are the first Impl members so they are the last
  /// members destroyed: every registry, carrier, block, Program closure, and provider launcher is
  /// gone before a DSO can be unloaded.
  struct PendingNativePackage {
    // Declared first so it is destroyed last within each package too: registrar/installer function
    // managers may themselves be emitted by the DSO.
    std::shared_ptr<void> lifetime;
    std::shared_ptr<runtime::system::NativePackageCapabilityState<Dim>> capability;
    std::string identity;
    std::function<void()> register_routes;
    std::function<void()> install;
    NativePackageKind kind = NativePackageKind::generic;
    std::vector<PreparedBoundaryHookContract> expected_boundary_hooks;

    PendingNativePackage(std::shared_ptr<void> owner,
                         std::shared_ptr<runtime::system::NativePackageCapabilityState<Dim>> state,
                         std::string package_identity, std::function<void()> registrar,
                         std::function<void()> installer, NativePackageKind package_kind)
        : lifetime(std::move(owner)),
          capability(std::move(state)),
          identity(std::move(package_identity)),
          register_routes(std::move(registrar)),
          install(std::move(installer)),
          kind(package_kind) {}
    PendingNativePackage(const PendingNativePackage&) = delete;
    PendingNativePackage& operator=(const PendingNativePackage&) = delete;
    PendingNativePackage(PendingNativePackage&& other) noexcept
        : lifetime(std::move(other.lifetime)),
          capability(std::move(other.capability)),
          identity(std::move(other.identity)),
          register_routes(std::move(other.register_routes)),
          install(std::move(other.install)),
          kind(other.kind),
          expected_boundary_hooks(std::move(other.expected_boundary_hooks)) {}
    PendingNativePackage& operator=(PendingNativePackage&& other) noexcept {
      static_assert(std::is_nothrow_move_assignable_v<std::shared_ptr<void>>);
      static_assert(std::is_nothrow_move_assignable_v<std::string>);
      static_assert(std::is_nothrow_move_assignable_v<std::function<void()>>);
      static_assert(std::is_nothrow_move_assignable_v<std::vector<PreparedBoundaryHookContract>>);
      if (this != &other) {
        // ``previous`` first acquires this package's lifetime, then takes its callable targets. It
        // remains alive until the replacement is complete and destroys those targets before its
        // lifetime. The destination likewise acquires the incoming lifetime before its callables.
        PendingNativePackage previous(std::move(*this));
        lifetime = std::move(other.lifetime);
        capability = std::move(other.capability);
        identity = std::move(other.identity);
        register_routes = std::move(other.register_routes);
        install = std::move(other.install);
        kind = other.kind;
        expected_boundary_hooks = std::move(other.expected_boundary_hooks);
      }
      return *this;
    }
  };
  static_assert(std::is_nothrow_move_constructible_v<PendingNativePackage>);
  static_assert(std::is_nothrow_move_assignable_v<PendingNativePackage>);
  static_assert(std::is_nothrow_destructible_v<PendingNativePackage>);
  std::vector<PendingNativePackage> pending_native_packages_;
  std::vector<PendingNativePackage> installed_native_packages_;

  domain_type domain_;
  SystemConfig<Dim>& cfg = domain_.cfg;
  Box<Dim>& dom = domain_.dom;
  Geometry<Dim>& geom = domain_.geom;
  mesh::BoxArray<Dim>& ba = domain_.ba;
  mesh::Distribution<Dim>& dm = domain_.dm;
  Index<Dim>& local_rank = domain_.local_rank;
  std::array<bool, Dim>& periodicity = domain_.periodicity;

  using auxiliary_registry_type = runtime::system::ExactAuxiliaryRegistry<Dim>;
  using auxiliary_publication_type = typename auxiliary_registry_type::PublicationTransaction;
  using auxiliary_key_type = runtime::system::AuxiliaryComponentKey;
  auxiliary_registry_type auxiliary_registry_;
  // No allocation exists for an empty provider graph. Every non-empty value belongs to one exact
  // storage group resolved from owner-qualified ComponentKeys.
  // Once allocated, the carrier object itself is never replaced. Generated native closures retain
  // its address; exact refresh/publication transactions swap only its value image.
  std::shared_ptr<runtime::system::AuxiliaryStorageGroups<Dim>> provider_carrier_;
  // The raw uploaded values are not carrier storage.  They remain host-side staging evidence until
  // one exact evaluation transaction publishes them into the runtime-owned compact provider carrier.
  std::map<std::string, std::vector<double>> staged_auxiliary_inputs_;
  std::vector<std::string> dirty_auxiliary_providers_;
  bool auxiliary_registry_consensus_verified_ = false;
  // The auxiliary carrier owns its own transport authority.  It cannot borrow an unrelated
  // solver lane because a provider refresh may occur before any block is prepared.
  std::optional<ExecutionLane> auxiliary_ghost_lane_;
  std::optional<runtime::system::PreparedAuxiliaryGhostTransport<Dim>> auxiliary_ghost_transport_;

  block_store_type blocks_;
  std::vector<Species>& sp = blocks_.blocks;
  std::vector<PreparedBoundaryHookContract> prepared_boundary_hook_contracts_;
  boundary_registry_type boundary_registry_;
  runtime::system::SystemCouplingRegistry<Dim> coupling_;
  runtime::system::SystemLifecycle lifecycle_;
  runtime::program::ProgramRuntimeState<Dim> program_;
  using embedded_boundary_type = runtime::system::PreparedEmbeddedBoundaryGeometry<Dim>;
  std::shared_ptr<const embedded_boundary_type> embedded_boundary_;
  std::optional<ExecutionLane> embedded_boundary_lane_;
  std::vector<std::string> embedded_boundary_opcodes_;
  std::vector<double> embedded_boundary_literals_;
  EbThresholds embedded_boundary_thresholds_{};
  std::uint64_t embedded_boundary_generation_ = 0;

  double t = 0.0;
  int macro_step_ = 0;
  std::string last_dt_reason_;
  NewtonReport last_newton_report_{};
  std::string poisson_solver_ = "cartesian_cg";
  std::string poisson_bc_ = "auto";
  double poisson_abs_tol_ = 0.0;
  double poisson_rel_tol_ = static_cast<double>(kCartesianCGDefaultRelTol);
  int poisson_max_iterations_ = kCartesianCGDefaultMaxIterations;

  using exact_field_type = runtime::system::ExactNamedField<Dim>;
  using component_field_solver_type = runtime::field::PreparedFieldSolverComponent<Dim>;

  struct FieldProviderBinding {
    std::string identity;
    std::string block;
    std::string key;
    double coefficient = 1.0;
  };

  struct FieldPlan {
    std::string plan_identity;
    std::string provider_identity;
    std::string output_owner_identity;
    std::string output_block;
    std::string output_key;
    std::vector<FieldProviderBinding> providers;
    std::string backend_provider_route;
    std::vector<std::string> boundary_kind;
    std::vector<double> boundary_alpha;
    std::vector<double> boundary_beta;
    std::vector<double> boundary_value;
    std::vector<std::string> boundary_state_blocks;
    std::vector<int> boundary_state_components;
    std::vector<std::string> boundary_field_blocks;
    std::vector<std::string> boundary_field_keys;
    std::vector<int> boundary_field_components;
    std::vector<Real> boundary_parameters;
    std::string nullspace_provider_identity;
    PreparedProviderOptions nullspace_options;
    std::string topology_provider_kind;
    std::string topology_provenance;
    std::string topology_digest;
    double reaction = 0.0;
    bool has_reaction = false;
    std::optional<CompiledFieldBoundaryKernel<Dim>> boundary_kernel;
    std::optional<FieldLogicalTimePoint> boundary_point;
    std::optional<FieldNewtonOptions> newton;
    std::shared_ptr<void> boundary_context_storage;
    std::shared_ptr<PreparedFieldBoundaryContextSet<Dim>> boundary_contexts;
  };

  struct ConfiguredFieldSolverProvider {
    std::string family_route;
    std::string exact_identity;
    PreparedProviderOptions options;
    double relative_tolerance = 0.0;
    double absolute_tolerance = 0.0;
    int maximum_iterations = 0;
  };

  struct StagedNativeFieldOutput {
    std::string block;
    std::string field;
    std::vector<auxiliary_key_type> output_keys;
    int gradient_sign = 1;
  };

  std::shared_ptr<exact_field_type> default_field_;
  std::map<std::string, std::shared_ptr<exact_field_type>> named_fields_;
  std::shared_ptr<exact_field_type> active_field_;
  std::optional<runtime::system::AuxiliaryStorageGroups<Dim>> active_field_provider_candidate_;
  std::optional<auxiliary_publication_type> active_field_auxiliary_publication_;
  std::vector<std::string> active_field_stale_auxiliary_providers_;
  std::map<std::string, FieldPlan> field_plans_;
  std::map<std::string, ConfiguredFieldSolverProvider> configured_field_solver_providers_;
  std::map<std::string, std::shared_ptr<component_field_solver_type>>
      component_field_solver_providers_;
  // A prepared field provider resolves this exact output route before native package finalization.
  // The finalizer authenticates it on the package lane and materializes it only after every generic
  // block exists in the detached candidate.
  std::map<std::string, StagedNativeFieldOutput> staged_native_field_outputs_;
  std::shared_ptr<FieldNullspaceProviderRegistry<Dim>> field_nullspace_providers_;
  bool field_plan_consensus_verified_ = false;
  std::string default_nullspace_provider_identity_;
  PreparedProviderOptions default_nullspace_options_;

  /// Full detached assembly image for native package finalization. While this pointer is non-null,
  /// the existing installation helpers resolve every mutable registry through the candidate.
  struct NativePackageFinalizeSnapshot;
  NativePackageFinalizeSnapshot* native_package_finalize_candidate_ = nullptr;

  struct NativePackageFinalizeSnapshot {
    struct BoundaryHookImage {
      std::shared_ptr<typename block_store_type::BoundaryFluxTransform> flux_target;
      typename block_store_type::BoundaryFluxTransform flux;
      std::shared_ptr<typename block_store_type::PreparedPointBoundaryResidual> residual_target;
      typename block_store_type::PreparedPointBoundaryResidual residual;
      std::shared_ptr<typename block_store_type::PreparedPointJvp> jvp_target;
      typename block_store_type::PreparedPointJvp jvp;
      std::shared_ptr<typename block_store_type::ExternalGhostBoundary> ghost_target;
      typename block_store_type::ExternalGhostBoundary ghost;
    };

    auxiliary_registry_type auxiliary_registry;
    std::shared_ptr<runtime::system::AuxiliaryStorageGroups<Dim>> provider_carrier_owner;
    std::optional<runtime::system::AuxiliaryStorageGroups<Dim>> provider_carrier;
    std::map<std::string, std::vector<double>> staged_auxiliary_inputs;
    std::vector<std::string> dirty_auxiliary_providers;
    bool auxiliary_registry_consensus_verified = false;
    block_store_type blocks;
    std::vector<BoundaryHookImage> boundary_hooks;
    std::vector<PreparedBoundaryHookContract> prepared_boundary_hook_contracts;
    boundary_registry_type boundary_registry;
    runtime::system::SystemCouplingRegistry<Dim> coupling;
    runtime::program::ProgramRuntimeState<Dim> program;
    std::shared_ptr<const embedded_boundary_type> embedded_boundary;
    std::vector<std::string> embedded_boundary_opcodes;
    std::vector<double> embedded_boundary_literals;
    EbThresholds embedded_boundary_thresholds{};
    std::uint64_t embedded_boundary_generation = 0;
    std::shared_ptr<exact_field_type> default_field;
    std::map<std::string, std::shared_ptr<exact_field_type>> named_fields;
    std::map<std::string, typename exact_field_type::rhs_image_type> named_field_rhs_images;
    std::shared_ptr<exact_field_type> active_field;
    std::map<std::string, FieldPlan> field_plans;
    std::map<std::string, ConfiguredFieldSolverProvider> configured_field_solver_providers;
    std::map<std::string, std::shared_ptr<component_field_solver_type>>
        component_field_solver_providers;
    std::map<std::string, StagedNativeFieldOutput> staged_native_field_outputs;
    std::shared_ptr<FieldNullspaceProviderRegistry<Dim>> field_nullspace_providers;
    bool field_plan_consensus_verified = false;
    std::string default_nullspace_provider_identity;
    PreparedProviderOptions default_nullspace_options;

    explicit NativePackageFinalizeSnapshot(const Impl& owner)
        : auxiliary_registry(owner.auxiliary_registry_),
          provider_carrier_owner(owner.provider_carrier_),
          provider_carrier(owner.provider_carrier_
                               ? std::optional<runtime::system::AuxiliaryStorageGroups<Dim>>(
                                     *owner.provider_carrier_)
                               : std::nullopt),
          staged_auxiliary_inputs(owner.staged_auxiliary_inputs_),
          dirty_auxiliary_providers(owner.dirty_auxiliary_providers_),
          auxiliary_registry_consensus_verified(owner.auxiliary_registry_consensus_verified_),
          blocks(owner.blocks_),
          prepared_boundary_hook_contracts(owner.prepared_boundary_hook_contracts_),
          boundary_registry(owner.boundary_registry_),
          coupling(owner.coupling_),
          program(owner.program_),
          embedded_boundary(owner.embedded_boundary_),
          embedded_boundary_opcodes(owner.embedded_boundary_opcodes_),
          embedded_boundary_literals(owner.embedded_boundary_literals_),
          embedded_boundary_thresholds(owner.embedded_boundary_thresholds_),
          embedded_boundary_generation(owner.embedded_boundary_generation_),
          default_field(owner.default_field_),
          named_fields(owner.named_fields_),
          active_field(owner.active_field_),
          field_plans(owner.field_plans_),
          configured_field_solver_providers(owner.configured_field_solver_providers_),
          component_field_solver_providers(owner.component_field_solver_providers_),
          staged_native_field_outputs(owner.staged_native_field_outputs_),
          field_nullspace_providers(owner.field_nullspace_providers_),
          field_plan_consensus_verified(owner.field_plan_consensus_verified_),
          default_nullspace_provider_identity(owner.default_nullspace_provider_identity_),
          default_nullspace_options(owner.default_nullspace_options_) {
      if (owner.active_field_ || owner.active_field_provider_candidate_ ||
          owner.active_field_auxiliary_publication_ ||
          !owner.active_field_stale_auxiliary_providers_.empty() ||
          owner.native_package_finalize_candidate_ != nullptr)
        throw std::logic_error(
            "System native package finalization cannot snapshot an active field candidate");
      // Prepared solve-context owners contain mutable invocation pointers into the live block and
      // field images. They are session cache, not rollback authority; never retain them in the
      // immutable pre-finalization journal.
      for (auto& [slot, plan] : field_plans) {
        (void)slot;
        plan.boundary_contexts.reset();
        plan.boundary_context_storage.reset();
      }
      boundary_hooks.reserve(owner.blocks_.blocks.size());
      for (const auto& block : owner.blocks_.blocks)
        append_boundary_hook_image(block);
      for (const auto& [slot, field] : owner.named_fields_) {
        if (!field)
          throw std::logic_error("System native package finalization found a null named field");
        named_field_rhs_images.emplace(slot, field->rhs_image());
      }
    }

    void append_boundary_hook_image(const Species& block) {
      BoundaryHookImage image;
      image.flux_target = block.external_boundary_flux;
      if (image.flux_target)
        image.flux = *image.flux_target;
      image.residual_target = block.external_field_boundary_residual;
      if (image.residual_target)
        image.residual = *image.residual_target;
      image.jvp_target = block.external_field_boundary_jvp;
      if (image.jvp_target)
        image.jvp = *image.jvp_target;
      image.ghost_target = block.external_ghost_boundary;
      if (image.ghost_target)
        image.ghost = *image.ghost_target;
      boundary_hooks.push_back(std::move(image));
    }

    BoundaryHookImage& boundary_hook(std::string_view block) {
      const auto found =
          std::find_if(blocks.blocks.begin(), blocks.blocks.end(),
                       [&](const Species& candidate) { return candidate.name == block; });
      if (found == blocks.blocks.end())
        throw std::out_of_range("System detached boundary hook block is not materialized");
      const std::size_t index = static_cast<std::size_t>(found - blocks.blocks.begin());
      if (index >= boundary_hooks.size())
        throw std::logic_error("System detached boundary hook image is incomplete");
      return boundary_hooks[index];
    }

    void publish_boundary_hook_noexcept(std::string_view block, std::string_view hook) noexcept {
      const auto found =
          std::find_if(blocks.blocks.begin(), blocks.blocks.end(),
                       [&](const Species& candidate) { return candidate.name == block; });
      if (found == blocks.blocks.end())
        std::terminate();
      const std::size_t index = static_cast<std::size_t>(found - blocks.blocks.begin());
      if (index >= boundary_hooks.size())
        std::terminate();
      BoundaryHookImage& image = boundary_hooks[index];
      if (hook == "ghost" && image.ghost_target)
        image.ghost_target->swap(image.ghost);
      else if (hook == "flux" && image.flux_target)
        image.flux_target->swap(image.flux);
      else if (hook == "field-residual" && image.residual_target)
        image.residual_target->swap(image.residual);
      else if (hook == "field-jvp" && image.jvp_target)
        image.jvp_target->swap(image.jvp);
      else
        std::terminate();
    }

    void publish_named_field_rhs_noexcept() noexcept {
      for (auto& [slot, image] : named_field_rhs_images) {
        const auto field = named_fields.find(slot);
        if (field == named_fields.end() || !field->second)
          std::terminate();
        field->second->publish_rhs_image(image);
      }
    }

    /// Publish the preallocated rollback image by swaps only.  This path runs after a collective
    /// installer failure, when allocating while packages still hold DSO lifetimes would permit a
    /// rank-asymmetric rollback escape.
    void restore_noexcept(Impl& owner) noexcept {
      using std::swap;
      owner.auxiliary_registry_.swap_complete(auxiliary_registry);
      if (provider_carrier) {
        if (!provider_carrier_owner)
          std::terminate();
        swap(*provider_carrier_owner, *provider_carrier);
        owner.provider_carrier_ = std::move(provider_carrier_owner);
      } else {
        if (owner.provider_carrier_)
          std::terminate();
      }
      // The carrier/registry image above can have a different allocation or resolved component
      // contract after a failed finalizer.  A prepared transport is therefore never rollback
      // state: rebuild it from the restored authoritative carrier on the next refresh.
      owner.auxiliary_ghost_transport_.reset();
      owner.active_field_provider_candidate_.reset();
      owner.active_field_auxiliary_publication_.reset();
      owner.active_field_stale_auxiliary_providers_.clear();
      swap(owner.staged_auxiliary_inputs_, staged_auxiliary_inputs);
      swap(owner.dirty_auxiliary_providers_, dirty_auxiliary_providers);
      swap(owner.auxiliary_registry_consensus_verified_, auxiliary_registry_consensus_verified);
      swap(owner.prepared_boundary_hook_contracts_, prepared_boundary_hook_contracts);
      // BoundaryFlux and FieldBoundary deliberately use shared generated hot-call slots so packages
      // installed after block materialization can extend their chains without installation-order
      // coupling. Restore those slots with noexcept swaps before copying the block-store image. Thus
      // even a later rollback allocation failure cannot leave failed package code reachable through
      // a pre-existing generated block.
      for (std::size_t index = 0; index < boundary_hooks.size(); ++index) {
        BoundaryHookImage& image = boundary_hooks[index];
        if (image.flux_target)
          image.flux_target->swap(image.flux);
        if (image.residual_target)
          image.residual_target->swap(image.residual);
        if (image.jvp_target)
          image.jvp_target->swap(image.jvp);
        if (image.ghost_target)
          image.ghost_target->swap(image.ghost);
      }
      swap(owner.blocks_, blocks);
      swap(owner.boundary_registry_, boundary_registry);
      swap(owner.coupling_, coupling);
      swap(owner.program_, program);
      swap(owner.embedded_boundary_, embedded_boundary);
      swap(owner.embedded_boundary_opcodes_, embedded_boundary_opcodes);
      swap(owner.embedded_boundary_literals_, embedded_boundary_literals);
      swap(owner.embedded_boundary_thresholds_, embedded_boundary_thresholds);
      swap(owner.embedded_boundary_generation_, embedded_boundary_generation);
      swap(owner.default_field_, default_field);
      swap(owner.named_fields_, named_fields);
      swap(owner.active_field_, active_field);
      swap(owner.field_plans_, field_plans);
      swap(owner.configured_field_solver_providers_, configured_field_solver_providers);
      swap(owner.component_field_solver_providers_, component_field_solver_providers);
      swap(owner.staged_native_field_outputs_, staged_native_field_outputs);
      swap(owner.field_nullspace_providers_, field_nullspace_providers);
      swap(owner.field_plan_consensus_verified_, field_plan_consensus_verified);
      swap(owner.default_nullspace_provider_identity_, default_nullspace_provider_identity);
      swap(owner.default_nullspace_options_, default_nullspace_options);
      // Native package installers are assembly-only graph constructors. They cannot enter a
      // field publication transaction, and the restored shared owners therefore preserve the
      // accepted field images without a post-consensus MultiFab assignment. Mutating an accepted
      // field during package installation is outside that installer contract and must fail before
      // finalization reaches this rollback path.
    }
  };

  void reserve_native_package_publication(std::size_t count) {
    if (count > installed_native_packages_.max_size() - installed_native_packages_.size())
      throw std::length_error("System native package publication exceeds vector capacity");
    installed_native_packages_.reserve(installed_native_packages_.size() + count);
  }

  /// The corresponding reserve is collectively completed before live installers run.  Moving a
  /// PendingNativePackage is non-allocating, so publication cannot strand a DSO lifetime after a
  /// successful installation phase.
  void publish_reserved_native_packages_noexcept(
      std::vector<PendingNativePackage>& packages) noexcept {
    static_assert(std::is_nothrow_move_constructible_v<PendingNativePackage>);
    for (auto& package : packages)
      installed_native_packages_.push_back(std::move(package));
    packages.clear();
  }

  static std::string exact_field_plan_contract(const FieldPlan& plan) {
    ExactContractBuilder contract;
    contract.text("pops.system.exact-ranked-field-plan")
        .scalar(std::uint32_t{2})
        .text(plan.plan_identity)
        .text(plan.provider_identity)
        .text(plan.output_owner_identity)
        .text(plan.output_block)
        .text(plan.output_key)
        .sequence(plan.providers,
                  [](ExactContractBuilder& item, const FieldProviderBinding& row) {
                    item.text(row.identity).text(row.block).text(row.key).scalar(row.coefficient);
                  })
        .text(plan.backend_provider_route)
        .sequence(plan.boundary_kind,
                  [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
        .sequence(plan.boundary_alpha)
        .sequence(plan.boundary_beta)
        .sequence(plan.boundary_value)
        .sequence(plan.boundary_state_blocks,
                  [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
        .sequence(plan.boundary_state_components)
        .sequence(plan.boundary_field_blocks,
                  [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
        .sequence(plan.boundary_field_keys,
                  [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
        .sequence(plan.boundary_field_components)
        .sequence(plan.boundary_parameters)
        .text(plan.nullspace_provider_identity)
        .presence(!plan.nullspace_provider_identity.empty());
    if (!plan.nullspace_provider_identity.empty())
      contract.bytes(plan.nullspace_options.exact_contract());
    contract.text(plan.topology_provider_kind)
        .text(plan.topology_provenance)
        .text(plan.topology_digest)
        .presence(plan.has_reaction);
    if (plan.has_reaction)
      contract.scalar(plan.reaction);
    contract.presence(plan.boundary_kernel.has_value());
    if (plan.boundary_kernel)
      contract.text(plan.boundary_kernel->identity)
          .text(plan.boundary_kernel->residual_identity)
          .text(plan.boundary_kernel->jvp_identity)
          .presence(plan.boundary_kernel->observes_iteration);
    contract.presence(plan.newton.has_value());
    if (plan.newton)
      contract.scalar(plan.newton->tolerance)
          .scalar(plan.newton->max_iterations)
          .scalar(plan.newton->linear_tolerance)
          .scalar(plan.newton->linear_max_iterations)
          .scalar(plan.newton->restart)
          .scalar(plan.newton->armijo)
          .scalar(plan.newton->minimum_step);
    return std::move(contract).release();
  }

  [[nodiscard]] std::string field_plan_registry_contract() const {
    ExactContractBuilder registry;
    registry.text("pops.system.exact-ranked-field-plan-registry").scalar(std::uint32_t{3});
    registry.scalar(static_cast<std::uint64_t>(configured_field_solver_providers_.size()));
    for (const auto& [route, provider] : configured_field_solver_providers_)
      registry.text(route)
          .text(provider.family_route)
          .text(provider.exact_identity)
          .bytes(provider.options.exact_contract());
    registry.scalar(static_cast<std::uint64_t>(component_field_solver_providers_.size()));
    for (const auto& [route, provider] : component_field_solver_providers_) {
      if (!provider)
        throw std::runtime_error("System field component provider registry contains null");
      registry.text(route)
          .text(provider->provider_identity())
          .bytes(provider->collective_contract());
    }
    registry.text(default_nullspace_provider_identity_)
        .presence(!default_nullspace_provider_identity_.empty());
    if (!default_nullspace_provider_identity_.empty())
      registry.bytes(default_nullspace_options_.exact_contract());
    registry.scalar(static_cast<std::uint64_t>(field_plans_.size()));
    for (const auto& [slot, plan] : field_plans_) {
      registry.text(slot).bytes(exact_field_plan_contract(plan));
      const auto configured = configured_field_solver_providers_.find(plan.backend_provider_route);
      const auto component = component_field_solver_providers_.find(plan.backend_provider_route);
      if ((configured == configured_field_solver_providers_.end()) ==
          (component == component_field_solver_providers_.end()))
        throw std::runtime_error(
            "System field plan must select exactly one installed exact-ranked backend route");
      if (configured != configured_field_solver_providers_.end())
        registry.text(configured->second.exact_identity)
            .bytes(configured->second.options.exact_contract());
      else
        registry.text(component->second->provider_identity())
            .bytes(component->second->collective_contract());
    }
    return std::move(registry).release();
  }

  [[nodiscard]] std::string staged_native_field_output_contract() const {
    if (staged_native_field_outputs_.size() != field_plans_.size())
      throw std::logic_error(
          "System native finalization requires exactly one staged output per field plan");
    ExactContractBuilder contract;
    contract.text("pops.system.staged-native-field-outputs")
        .scalar(std::uint32_t{1})
        .scalar(static_cast<std::uint64_t>(staged_native_field_outputs_.size()));
    for (const auto& [slot, output] : staged_native_field_outputs_) {
      const auto plan = field_plans_.find(slot);
      if (plan == field_plans_.end() || output.block.empty() || output.field.empty() ||
          output.output_keys.empty() || output.block != plan->second.output_block ||
          output.field != plan->second.output_key)
        throw std::logic_error(
            "System staged native field output differs from its exact resolved plan");
      runtime::field::NamedFieldOutput<Dim> shape(output.output_keys.size(), output.gradient_sign);
      (void)shape;
      contract.text(slot)
          .text(output.block)
          .text(output.field)
          .scalar(std::int32_t{output.gradient_sign})
          .scalar(static_cast<std::uint64_t>(output.output_keys.size()));
      std::set<std::string> exact_keys;
      for (const auto& key : output.output_keys) {
        key.validate();
        const std::string exact = key.exact_key();
        if (!exact_keys.insert(exact).second)
          throw std::logic_error("System staged native field output keys are duplicate");
        contract.text(exact);
      }
    }
    return std::move(contract).release();
  }

  void require_field_plan_consensus() {
    if (field_plan_consensus_verified_)
      return;
    std::string bytes;
    std::exception_ptr local_error;
    try {
      bytes = field_plan_registry_contract();
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L) != 0) {
      if (n_ranks() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("System field-plan registry preparation failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs({{"system-field-plan-registry", bytes}}))
      throw std::runtime_error("System: ordered resolved field plans differ across MPI ranks");
    field_plan_consensus_verified_ = true;
  }

  std::string resolve_named_field_slot(std::string_view field) const {
    if (named_fields_.contains(std::string(field)))
      return std::string(field);
    for (const auto& [slot, plan] : field_plans_)
      if (plan.output_key == field)
        return slot;
    return std::string(field);
  }

  /// Copy one already-materialized field image without constructing a temporary Fab.  A Fab's
  /// ordinary copy assignment is intentionally value-semantic and allocates a fresh Kokkos view;
  /// that is correct for assembly, but it is not a valid operation while the accepted-step
  /// transaction is hot.  The transaction image is dimensioned once at bind and only this
  /// validated deep-copy path is used afterwards.
  static void copy_field_into_preallocated(field_type& destination, const field_type& source) {
    if (destination.layout() != source.layout() ||
        destination.distribution() != source.distribution() ||
        destination.local_rank() != source.local_rank() || destination.ncomp() != source.ncomp() ||
        destination.ghosts() != source.ghosts() || destination.local_size() != source.local_size())
      throw std::logic_error("System accepted transaction field layout changed after bind");
    for (std::size_t local = 0; local < destination.local_size(); ++local) {
      if (destination.global_index(local) != source.global_index(local) ||
          destination.fab(local).size() != source.fab(local).size())
        throw std::logic_error("System accepted transaction field ownership changed after bind");
      Kokkos::deep_copy(destination.fab(local).storage(), source.fab(local).storage());
    }
  }

  static void copy_string_into_preallocated(std::string& destination, std::string_view source) {
    if (source.size() > destination.capacity())
      throw std::logic_error("System accepted transaction string capacity was not preallocated");
    destination.assign(source.data(), source.size());
  }

  template <class T>
  static void copy_vector_into_preallocated(std::vector<T>& destination,
                                            const std::vector<T>& source) {
    if (source.size() > destination.capacity())
      throw std::logic_error("System accepted transaction vector capacity was not preallocated");
    destination.assign(source.begin(), source.end());
  }

  static void copy_staged_inputs_into_preallocated(
      std::map<std::string, std::vector<double>>& destination,
      const std::map<std::string, std::vector<double>>& source) {
    if (destination.size() != source.size())
      throw std::logic_error("System accepted transaction auxiliary composition changed");
    for (const auto& [identity, values] : source) {
      const auto found = destination.find(identity);
      if (found == destination.end() || values.size() > found->second.capacity())
        throw std::logic_error(
            "System accepted transaction auxiliary input capacity was not preallocated");
      found->second.assign(values.begin(), values.end());
    }
  }

  static void copy_provider_groups_into_preallocated(
      runtime::system::AuxiliaryStorageGroups<Dim>& destination,
      const runtime::system::AuxiliaryStorageGroups<Dim>& source) {
    if (destination.groups.size() != source.groups.size())
      throw std::logic_error("System accepted transaction provider composition changed");
    for (const auto& [identity, group] : source.groups) {
      const auto found = destination.groups.find(identity);
      if (found == destination.groups.end())
        throw std::logic_error("System accepted transaction provider group changed after bind");
      copy_field_into_preallocated(found->second, group);
    }
  }

  static void copy_string_vector_into_preallocated(std::vector<std::string>& destination,
                                                   const std::vector<std::string>& source) {
    if (source.size() > destination.capacity())
      throw std::logic_error("System accepted transaction string-vector capacity was not primed");
    if (destination.size() != source.size()) {
      if (source.size() > destination.capacity())
        throw std::logic_error("System accepted transaction string-vector capacity was not primed");
      destination.resize(source.size());
    }
    for (std::size_t index = 0; index < source.size(); ++index)
      copy_string_into_preallocated(destination[index], source[index]);
  }

  struct AcceptedSnapshot {
    std::vector<field_type> states;
    auxiliary_registry_type auxiliary_registry;
    std::optional<runtime::system::AuxiliaryStorageGroups<Dim>> provider_carrier;
    std::map<std::string, std::vector<double>> staged_auxiliary_inputs;
    std::vector<std::string> dirty_auxiliary_providers;
    runtime::program::ProgramRuntimeState<Dim> program;
    std::optional<typename exact_field_type::AcceptedState> default_field_state;
    std::map<std::string, typename exact_field_type::AcceptedState> named_field_states;
    double time = 0.0;
    int macro_step = 0;
    std::string last_dt_reason;
    NewtonReport last_newton_report{};
    bool initialized = false;
    bool program_image_warm = false;
    std::uint64_t program_install_generation = 0;

    AcceptedSnapshot() = default;

    explicit AcceptedSnapshot(const Impl& owner) { capture_from(owner); }

    void capture_from(const Impl& owner) {
      if (owner.active_field_ || owner.active_field_provider_candidate_ ||
          owner.active_field_auxiliary_publication_ ||
          !owner.active_field_stale_auxiliary_providers_.empty())
        throw std::logic_error(
            "System cannot snapshot an unconsumed exact field publication candidate");

      if (!initialized) {
        states.clear();
        auxiliary_registry = {};
        provider_carrier.reset();
        staged_auxiliary_inputs.clear();
        dirty_auxiliary_providers.clear();
        program = {};
        default_field_state.reset();
        named_field_states.clear();
        states.reserve(owner.sp.size());
        for (const Species& block : owner.sp)
          states.push_back(block.U);
        auxiliary_registry = owner.auxiliary_registry_;
        if (owner.provider_carrier_)
          provider_carrier.emplace(*owner.provider_carrier_);
        staged_auxiliary_inputs = owner.staged_auxiliary_inputs_;
        dirty_auxiliary_providers = owner.dirty_auxiliary_providers_;
        program = owner.program_;
        program_image_warm = true;
        program_install_generation = owner.program_.step_install_generation_;
        if (owner.default_field_)
          default_field_state = owner.default_field_->accepted_state();
        for (const auto& [slot, field] : owner.named_fields_) {
          if (!field)
            throw std::logic_error("System materialized named field is null");
          named_field_states.emplace(slot, field->accepted_state());
        }
        // Reasons are assembled from fixed prefixes and bind-frozen user identities. Reserve the
        // exact maximum before entering the first candidate so adaptive-CFL bookkeeping cannot
        // grow its string during a hot rollback image refresh.
        std::size_t max_reason = std::string("degenerate").size();
        for (const Species& block : owner.sp) {
          max_reason = std::max(max_reason, std::string("transport:").size() + block.name.size());
          max_reason =
              std::max(max_reason, std::string("parabolic_frequency:").size() + block.name.size());
          max_reason =
              std::max(max_reason, std::string("source_frequency:").size() + block.name.size());
          max_reason =
              std::max(max_reason, std::string("stability_dt:").size() + block.name.size());
        }
        for (const auto& frequency : owner.coupling_.coupled_freqs)
          max_reason =
              std::max(max_reason, std::string("coupled_source:").size() + frequency.label.size());
        for (const auto& frequency : owner.coupling_.coupled_frequencies)
          max_reason =
              std::max(max_reason, std::string("coupled_source:").size() + frequency.label.size());
        for (const auto& bound : owner.coupling_.dt_bounds)
          max_reason = std::max(max_reason, std::string("global:").size() + bound.label.size());
        max_reason = std::max(max_reason, std::string("program:dt_bound").size());
        max_reason = std::max(max_reason, std::string("strategy:max_dt").size());
        last_dt_reason.reserve(max_reason);
      } else {
        if (states.size() != owner.sp.size() ||
            named_field_states.size() != owner.named_fields_.size())
          throw std::logic_error("System transaction snapshot composition changed");
        for (std::size_t block = 0; block < states.size(); ++block)
          copy_field_into_preallocated(states[block], owner.sp[block].U);
        if (provider_carrier.has_value() != static_cast<bool>(owner.provider_carrier_))
          throw std::logic_error("System accepted transaction provider ownership changed");
        if (provider_carrier)
          copy_provider_groups_into_preallocated(*provider_carrier, *owner.provider_carrier_);
        copy_staged_inputs_into_preallocated(staged_auxiliary_inputs,
                                             owner.staged_auxiliary_inputs_);
        copy_vector_into_preallocated(dirty_auxiliary_providers, owner.dirty_auxiliary_providers_);
        if (program_install_generation != owner.program_.step_install_generation_)
          throw std::logic_error("System accepted transaction Program composition changed");
        // Diagnostics, history/cache slots, profiler scopes and persistent values are all expected
        // to have been materialized by the bind-time Program prelude.  Refresh the resident image
        // strictly in place; a late identity/shape change is a deterministic refusal rather than a
        // hidden allocation in the accepted-step callback.
        if (!program_image_warm)
          throw std::logic_error("System accepted transaction Program image was not primed");
        // ProgramRuntimeState owns the complete bind-sealed transaction composition, including
        // balance mailboxes, due-window flags, and projection activity.  Keep this single
        // exhaustive copy authority instead of silently letting the Uniform carrier drift when a
        // new Program candidate field is introduced.
        program.copy_from_preallocated(owner.program_);
        if (default_field_state.has_value() != static_cast<bool>(owner.default_field_))
          throw std::logic_error("System accepted transaction default-field ownership changed");
        if (default_field_state) {
          copy_field_into_preallocated(default_field_state->potential,
                                       owner.default_field_->accepted_potential());
          copy_field_into_preallocated(default_field_state->outputs,
                                       owner.default_field_->accepted_outputs());
        }
        for (const auto& [slot, field] : owner.named_fields_) {
          if (!field)
            throw std::logic_error("System materialized named field is null");
          const auto image = named_field_states.find(slot);
          if (image == named_field_states.end())
            throw std::logic_error("System accepted transaction named-field composition changed");
          copy_field_into_preallocated(image->second.potential, field->accepted_potential());
          copy_field_into_preallocated(image->second.outputs, field->accepted_outputs());
        }
      }

      time = owner.t;
      macro_step = owner.macro_step_;
      copy_string_into_preallocated(last_dt_reason, owner.last_dt_reason_);
      last_newton_report = owner.last_newton_report_;
      initialized = true;
    }

    void restore(Impl& owner) {
      if (states.size() != owner.sp.size())
        throw std::logic_error("System transaction snapshot composition changed");
      static_assert(std::is_nothrow_swappable_v<field_type>);
      for (std::size_t block = 0; block < states.size(); ++block)
        std::swap(owner.sp[block].U, states[block]);
      // The auxiliary registry is a complete accepted carrier.  Exchange it instead of assigning
      // so rollback remains a closed, allocation-free operation after the transaction has bound.
      owner.auxiliary_registry_.swap_complete(auxiliary_registry);
      if (provider_carrier) {
        if (!owner.provider_carrier_)
          throw std::logic_error("System carrier owner vanished during accepted rollback");
        else
          std::swap(*owner.provider_carrier_, *provider_carrier);
      } else {
        if (owner.provider_carrier_)
          throw std::logic_error(
              "System accepted rollback cannot revoke a published carrier owner");
      }
      owner.active_field_provider_candidate_.reset();
      owner.active_field_auxiliary_publication_.reset();
      owner.active_field_stale_auxiliary_providers_.clear();
      owner.staged_auxiliary_inputs_.swap(staged_auxiliary_inputs);
      owner.dirty_auxiliary_providers_.swap(dirty_auxiliary_providers);
      static_assert(std::is_nothrow_swappable_v<runtime::program::ProgramRuntimeState<Dim>>);
      std::swap(owner.program_, program);
      if (!default_field_state) {
        if (owner.default_field_)
          throw std::logic_error("System transaction default-field ownership changed");
      } else {
        if (!owner.default_field_)
          throw std::logic_error("System transaction snapshot default-field presence vanished");
        copy_field_into_preallocated(owner.default_field_->accepted_potential_for_restore(),
                                     default_field_state->potential);
        copy_field_into_preallocated(owner.default_field_->accepted_outputs_for_restore(),
                                     default_field_state->outputs);
      }
      if (named_field_states.size() != owner.named_fields_.size())
        throw std::logic_error("System transaction snapshot named-field composition changed");
      for (const auto& [slot, values] : named_field_states) {
        const auto field = owner.named_fields_.find(slot);
        if (field == owner.named_fields_.end() || !field->second)
          throw std::logic_error("System transaction snapshot field ownership changed");
        copy_field_into_preallocated(field->second->accepted_potential_for_restore(),
                                     values.potential);
        copy_field_into_preallocated(field->second->accepted_outputs_for_restore(), values.outputs);
      }
      owner.t = time;
      owner.macro_step_ = macro_step;
      owner.last_dt_reason_.swap(last_dt_reason);
      using std::swap;
      swap(owner.last_newton_report_, last_newton_report);
    }
  };

  // External savepoints keep the same ProgramTransaction (and its visibility writer) from begin
  // through finalize/rollback. The registered carrier is the sole restore image; retaining a second
  // AcceptedSnapshot here would create a second transaction authority and duplicate state.
  std::optional<runtime::program::ProgramTransaction> external_program_transaction_;
  bool external_step_transaction_committed_ = false;

  /// The System driver owns one frozen transaction registry for its whole lifetime.  The
  /// participant is an aggregate carrier around the existing accepted image; this keeps the
  /// registry as the sole phase/visibility authority without introducing a second state engine.
  /// Its callback image is deliberately one byte: the typed carrier preallocates and owns the
  /// actual restore image, while the registry still freezes an explicit participant budget/order.
  struct StepTransactionCarrier final {
    Impl* owner = nullptr;
    std::uint64_t step_change_last_dispatches = 0;
    /// One canonical term per valid cell/component.  SharedSpace is the project's portable
    /// unified residency: the hot diagnostic writes it from Kokkos and sums it on the host after
    /// a fence, without a scalar deep-copy, a temporary reduction object, or reduction atomics.
    Kokkos::View<Real*, Kokkos::SharedSpace> step_change_terms;
    Kokkos::View<int, Kokkos::SharedSpace> step_change_invalid;
    /// Exact maximum valid-cell/component term count of one currently bound block.  A block
    /// diagnostic consumes that block's complete canonical prefix, so reusing the one workspace
    /// for another block never needs a hot resize.
    std::size_t step_change_term_capacity = 0;
    std::optional<AcceptedSnapshot> accepted;
    bool snapshot_active = false;

    explicit StepTransactionCarrier(Impl& value) noexcept : owner(&value) {}

    bool capture(void*, std::size_t bytes) noexcept {
      if (owner == nullptr || bytes != 1 || snapshot_active)
        return false;
      try {
        if (!accepted)
          accepted.emplace();
        accepted->capture_from(*owner);
        snapshot_active = true;
        return true;
      } catch (...) {
        snapshot_active = false;
        return false;
      }
    }

    void restore(const void*, std::size_t bytes) noexcept {
      if (owner == nullptr || bytes != 1 || !accepted || !snapshot_active)
        std::terminate();
      try {
        accepted->restore(*owner);
      } catch (...) {
        std::terminate();
      }
      snapshot_active = false;
    }

    /// Materialize the reusable rollback image after composition and Program installation have
    /// settled. This is the bind-time cold path; every subsequent transaction only refreshes the
    /// already-owned buffers.
    void prime() {
      if (owner == nullptr || snapshot_active)
        throw std::logic_error("System transaction carrier cannot be primed while active");
      if (!accepted)
        accepted.emplace();
      std::size_t required_terms = 0;
      for (const Species& block : owner->sp) {
        if (block.U.ncomp() <= 0)
          throw std::logic_error("System transaction carrier block has no state components");
        std::size_t block_terms = 0;
        for (std::size_t local = 0; local < block.U.local_size(); ++local) {
          const std::int64_t signed_cells = block.U.box(local).numPts();
          if (signed_cells <= 0)
            throw std::logic_error("System transaction carrier block has an empty valid patch");
          const std::size_t cells = static_cast<std::size_t>(signed_cells);
          const std::size_t components = static_cast<std::size_t>(block.U.ncomp());
          if (cells > std::numeric_limits<std::size_t>::max() / components ||
              block_terms > std::numeric_limits<std::size_t>::max() - cells * components)
            throw std::overflow_error("System transaction carrier step-change workspace overflow");
          block_terms += cells * components;
        }
        required_terms = std::max(required_terms, block_terms);
      }
      if (required_terms == 0)
        throw std::logic_error("System transaction carrier step-change workspace is empty");
      if (step_change_terms.data() == nullptr || step_change_term_capacity != required_terms) {
        step_change_terms =
            decltype(step_change_terms)("pops_step_change_l2_terms", required_terms);
        step_change_invalid = decltype(step_change_invalid)("pops_step_change_l2_invalid");
        step_change_term_capacity = required_terms;
      }
      step_change_invalid() = 0;
      accepted->capture_from(*owner);
      snapshot_active = false;
    }

    bool publish() noexcept { return owner != nullptr; }

    static bool snapshot_callback(void* object, void* image, std::size_t bytes) noexcept {
      return static_cast<StepTransactionCarrier*>(object)->capture(image, bytes);
    }
    static void restore_callback(void* object, const void* image, std::size_t bytes) noexcept {
      static_cast<StepTransactionCarrier*>(object)->restore(image, bytes);
    }
    static bool publish_callback(void* object) noexcept {
      return static_cast<StepTransactionCarrier*>(object)->publish();
    }
    static void rollback_callback(void* object, const void* image, std::size_t bytes) noexcept {
      static_cast<StepTransactionCarrier*>(object)->restore(image, bytes);
    }

    /// Keep the fully dimensioned image resident for the next step. Resetting the optional here
    /// would destroy every Fab/map node and turn the following capture into a hot-path allocation.
    void discard_snapshot() noexcept { snapshot_active = false; }
  };

  static bool step_transaction_consensus(void* context, std::uint32_t phase,
                                         std::uint32_t status) noexcept {
    try {
      auto* owner = static_cast<Impl*>(context);
      if (owner == nullptr)
        return false;
      // The registry status is a bounded protocol code. Compare the exact code, rather than a
      // success/failure bit, so two distinct local fault classifications cannot be mistaken for
      // a collective agreement.
      const long status_word = static_cast<long>(status);
      const std::uint64_t generation =
          static_cast<std::uint64_t>(owner->step_transaction_registry_.accepted_generation());
      // Keep every collective word representable even on platforms whose `long` is 32-bit.
      const long generation_low = static_cast<long>(generation & UINT64_C(0x1fffff));
      const long generation_middle = static_cast<long>((generation >> 21U) & UINT64_C(0x1fffff));
      const long generation_high = static_cast<long>(generation >> 42U);
      const long phase_word = static_cast<long>(phase);
      // Execute every word unconditionally.  A rank that disagrees on status or phase must still
      // participate in all 21/21/22 generation reductions, otherwise the next collective would
      // become rank-skewed after a local rejection.
      const long status_max = all_reduce_max(status_word);
      const long status_min = all_reduce_min(status_word);
      const long phase_max = all_reduce_max(phase_word);
      const long phase_min = all_reduce_min(phase_word);
      const long generation_low_max = all_reduce_max(generation_low);
      const long generation_low_min = all_reduce_min(generation_low);
      const long generation_middle_max = all_reduce_max(generation_middle);
      const long generation_middle_min = all_reduce_min(generation_middle);
      const long generation_high_max = all_reduce_max(generation_high);
      const long generation_high_min = all_reduce_min(generation_high);
      return status_max == status_min && phase_max == phase_min &&
             generation_low_max == generation_low_min &&
             generation_middle_max == generation_middle_min &&
             generation_high_max == generation_high_min;
    } catch (...) {
      return false;
    }
  }

  runtime::program::ProgramTransactionRegistry step_transaction_registry_;
  StepTransactionCarrier step_transaction_carrier_;
  runtime::program::ParticipantHandle<StepTransactionCarrier> step_transaction_participant_{};

  [[nodiscard]] runtime::program::AcceptedReadLease acquire_accepted_read_lease() const {
    return step_transaction_registry_.acquire_read();
  }

  [[nodiscard]] runtime::program::AcceptedWriteLease acquire_accepted_write_lease() const {
    return step_transaction_registry_.acquire_write();
  }

  void prime_step_transaction_image() { step_transaction_carrier_.prime(); }

  explicit Impl(const SystemConfig<Dim>& config)
      : domain_(config),
        field_nullspace_providers_(make_default_field_nullspace_provider_registry<Dim>()),
        step_transaction_registry_(
            runtime::program::ProgramTransactionBudget{1, 1, 0, 0},
            runtime::program::ProgramTransactionConsensus{&Impl::step_transaction_consensus, this}),
        step_transaction_carrier_(*this) {
    // Candidate execution still uses the resident Uniform carriers. Freeze public readers before
    // entering that phase; detached participants can opt into the historical publish-time lock.
    step_transaction_registry_.set_candidate_visibility_lock(true);
    const FieldNullspaceProviderSelection selection = operator_topology_zero_mean_nullspace();
    default_nullspace_provider_identity_ = selection.provider_identity;
    default_nullspace_options_ = selection.options;

    runtime::program::ProgramParticipantOps ops;
    ops.snapshot = &StepTransactionCarrier::snapshot_callback;
    ops.restore = &StepTransactionCarrier::restore_callback;
    ops.publish = &StepTransactionCarrier::publish_callback;
    ops.rollback = &StepTransactionCarrier::rollback_callback;
    ops.candidate = [](void* object) noexcept -> void* { return object; };
    step_transaction_participant_ = step_transaction_registry_.register_participant(
        step_transaction_carrier_, ops, runtime::program::ProgramParticipantBudget{1, 0});
    step_transaction_registry_.bind();
  }

  ~Impl() noexcept {
    // external_program_transaction_ is declared before the registry (and therefore would
    // otherwise be destroyed after it). Close the active transaction while its registry and
    // carrier are still alive; this preserves the same rollback-before-authority-destruction order
    // as an explicit external rollback.
    external_program_transaction_.reset();
  }

  FieldNullspaceProviderRequest<Dim> prepare_uniform_field_nullspace_request(
      std::string plan_identity, std::string topology_identity,
      const elliptic::nd::CartesianPoissonOptions<Dim>& options, const field_type& layout,
      bool has_reaction) const {
    if (!field_nullspace_providers_)
      throw std::logic_error("System field-nullspace registry is absent");
    std::vector<FieldBoundaryNullspaceFact> boundaries;
    boundaries.reserve(static_cast<std::size_t>(2 * Dim));
    ExactContractBuilder boundary_contract;
    boundary_contract.text("pops.system.uniform-field-boundary-set")
        .scalar(std::uint32_t{1})
        .scalar(static_cast<std::uint32_t>(Dim));
    for (int axis = 0; axis < Dim; ++axis) {
      for (int side = 0; side < 2; ++side) {
        const std::size_t face = static_cast<std::size_t>(2 * axis + side);
        const auto kind = options.boundaries[face];
        const Real alpha = options.boundary_alpha[face];
        FieldBoundaryNullspaceBehavior behavior =
            FieldBoundaryNullspaceBehavior::ConstrainsConstantMode;
        if (kind == elliptic::nd::CartesianBoundaryKind::periodic ||
            kind == elliptic::nd::CartesianBoundaryKind::neumann ||
            (kind == elliptic::nd::CartesianBoundaryKind::mixed && alpha == Real(0)))
          behavior = FieldBoundaryNullspaceBehavior::PreservesConstantMode;
        const std::string face_identity =
            "axis:" + std::to_string(axis) + (side == 0 ? ":lower" : ":upper");
        boundaries.push_back({face_identity, behavior});
        boundary_contract.text(face_identity).scalar(kind).scalar(alpha);
      }
    }

    const PreparedVectorDistribution<Dim> distribution =
        PreparedVectorDistribution<Dim>::distributed();
    FieldNullspaceProviderRequest<Dim> request;
    request.plan_identity = std::move(plan_identity);
    request.operator_facts = make_field_nullspace_operator_facts(
        std::move(boundary_contract).release(), std::move(boundaries), has_reaction);
    request.topology.identity = std::move(topology_identity);
    request.topology.exact_layout_contract = distribution.layout_contract(layout);
    request.topology.field_component = 0;
    ExactContractBuilder connected;
    connected.text("pops.system.uniform-cartesian-connected-component")
        .scalar(std::uint32_t{1})
        .scalar(static_cast<std::uint32_t>(Dim));
    for (int axis = 0; axis < Dim; ++axis)
      connected.scalar(geom.domain().lo[axis]).scalar(geom.domain().hi[axis]);
    request.topology.connected_component_contract = std::move(connected).release();
    request.topology.layouts = {&layout};
    Real cell_measure = Real(1);
    for (int axis = 0; axis < Dim; ++axis)
      cell_measure *= geom.spacing(axis);
    request.topology.cell_measure = {cell_measure};
    request.topology.level_distributions = {distribution};
    return request;
  }

  PreparedFieldNullspace<Dim> finish_uniform_field_nullspace(
      const FieldNullspaceProviderSelection& selection, FieldNullspaceProviderRequest<Dim> request,
      const ExecutionLane& lane) const {
    return prepare_field_nullspace_collectively<Dim>(*field_nullspace_providers_, selection,
                                                     std::move(request), lane);
  }

  PreparedFieldNullspace<Dim> prepare_uniform_field_nullspace(
      std::string plan_identity, std::string topology_identity,
      const FieldNullspaceProviderSelection& selection,
      const elliptic::nd::CartesianPoissonOptions<Dim>& options, const field_type& layout,
      bool has_reaction, const ExecutionLane& lane) const {
    return finish_uniform_field_nullspace(
        selection,
        prepare_uniform_field_nullspace_request(
            std::move(plan_identity), std::move(topology_identity), options, layout, has_reaction),
        lane);
  }

  Species& find(const std::string& name) { return blocks_.find(name); }
  const Species& find(const std::string& name) const { return blocks_.find(name); }
  int index(const std::string& name) const { return blocks_.index(name); }

  const typename boundary_registry_type::InstalledBoundary* boundary_for(
      const std::string& name) const noexcept {
    return boundary_registry_.find_boundary(name);
  }

  void publish_boundary_to_block(const std::string& name) {
    Species& block = find(name);
    const auto* installed = boundary_for(name);
    block.boundary = installed == nullptr ? nullptr : installed->authority;
    if (installed != nullptr)
      block.state_identity = installed->state_identity;
  }

  /// Execute a candidate body after the transaction has entered its candidate phase. The outer
  /// RuntimeInstance envelope owns solve/guard/effect preparation, hidden publication, seal, and
  /// irreversible finalization. A candidate exception intentionally leaves the transaction live
  /// so the outer controller can consume the failure and invoke its single rollback path.
  template <class Function>
  void execute_candidate_body(runtime::program::ProgramTransaction& transaction,
                              Function&& function) {
    std::exception_ptr function_error;
    try {
      std::forward<Function>(function)();
    } catch (...) {
      function_error = std::current_exception();
    }
    bool any_function_error = false;
    try {
      any_function_error = all_reduce_max(function_error ? 1L : 0L) != 0;
    } catch (...) {
      any_function_error = true;
    }
    if (any_function_error) {
      if (function_error)
        std::rethrow_exception(function_error);
      throw std::runtime_error("System transaction candidate execution failed collectively");
    }
  }

  template <class Function>
  void execute_candidate_phase(runtime::program::ProgramTransaction& transaction,
                               Function&& function) {
    if (!transaction.begin_candidate())
      throw std::runtime_error("System transaction candidate phase rejected collectively");
    execute_candidate_body(transaction, std::forward<Function>(function));
  }

  template <class Function>
  decltype(auto) execute_step_transaction(Function&& function) {
    // RuntimeInstance's external envelope already owns the transaction and visibility writer.
    // Borrow it here: re-entering begin()/seal() would publish a generation before Python effects
    // have been prepared and would make a later outer rollback non-atomic.
    if (external_program_transaction_) {
      if (external_step_transaction_committed_ ||
          external_program_transaction_->phase() !=
              runtime::program::ProgramTransactionPhase::kCandidate)
        throw std::logic_error("System transaction candidate is already committed or consumed");
      if constexpr (std::is_void_v<std::invoke_result_t<Function&>>) {
        execute_candidate_body(*external_program_transaction_, std::forward<Function>(function));
        return;
      } else {
        using result_type = std::invoke_result_t<Function&>;
        static_assert(!std::is_reference_v<result_type>,
                      "System transaction functions must return a value");
        std::optional<std::remove_cv_t<result_type>> result;
        execute_candidate_body(*external_program_transaction_,
                               [&] { result.emplace(std::forward<Function>(function)()); });
        return result_type(std::move(*result));
      }
    }

    auto transaction = step_transaction_registry_.begin();
    try {
      if constexpr (std::is_void_v<std::invoke_result_t<Function&>>) {
        execute_candidate_phase(transaction, std::forward<Function>(function));
        if (!transaction.begin_solve_guard_effect_prepare())
          throw std::runtime_error(
              "System transaction solve/guard/effect preparation rejected collectively");
        if (!transaction.hidden_publish())
          throw std::runtime_error("System transaction hidden publication failed collectively");
        if (!transaction.atomic_seal())
          throw std::runtime_error("System transaction atomic seal failed collectively");
        const auto finalized = transaction.irreversible_finalize();
        step_transaction_carrier_.discard_snapshot();
        if (!finalized)
          throw std::runtime_error("System transaction entered fail-stop during finalization");
      } else {
        using result_type = std::invoke_result_t<Function&>;
        static_assert(!std::is_reference_v<result_type>,
                      "System transaction functions must return a value");
        std::optional<std::remove_cv_t<result_type>> result;
        execute_candidate_phase(transaction,
                                [&] { result.emplace(std::forward<Function>(function)()); });
        if (!transaction.begin_solve_guard_effect_prepare())
          throw std::runtime_error(
              "System transaction solve/guard/effect preparation rejected collectively");
        if (!transaction.hidden_publish())
          throw std::runtime_error("System transaction hidden publication failed collectively");
        if (!transaction.atomic_seal())
          throw std::runtime_error("System transaction atomic seal failed collectively");
        const auto finalized = transaction.irreversible_finalize();
        step_transaction_carrier_.discard_snapshot();
        if (!finalized)
          throw std::runtime_error("System transaction entered fail-stop during finalization");
        return result_type(std::move(*result));
      }
    } catch (...) {
      transaction.rollback();
      throw;
    }
  }
};

template <int Dim>
void validate_system_config(const SystemConfig<Dim>& config) {
  config.validate_spatial_domain();
  if (config.coordinate_system != runtime_config_detail::cartesian_coordinate_system<Dim>())
    throw std::invalid_argument(
        "System Cartesian core requires a dimension-qualified Cartesian provider");
}

}  // namespace pops
