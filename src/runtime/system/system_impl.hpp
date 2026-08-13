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
#include <pops/runtime/program/program_runtime_state.hpp>
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

  /// One validated native package.  Package owners are the first Impl members so they are the last
  /// members destroyed: every registry, carrier, block, Program closure, and provider launcher is
  /// gone before a DSO can be unloaded.
  struct PendingNativePackage {
    // Declared first so it is destroyed last within each package too: registrar/installer function
    // managers may themselves be emitted by the DSO.
    std::shared_ptr<void> lifetime;
    std::string identity;
    std::function<void()> register_routes;
    std::function<void()> install;
    NativePackageKind kind = NativePackageKind::generic;

    PendingNativePackage(std::shared_ptr<void> owner, std::string package_identity,
                         std::function<void()> registrar, std::function<void()> installer,
                         NativePackageKind package_kind)
        : lifetime(std::move(owner)),
          identity(std::move(package_identity)),
          register_routes(std::move(registrar)),
          install(std::move(installer)),
          kind(package_kind) {}
    PendingNativePackage(const PendingNativePackage&) = delete;
    PendingNativePackage& operator=(const PendingNativePackage&) = delete;
    PendingNativePackage(PendingNativePackage&& other) noexcept
        : lifetime(std::move(other.lifetime)),
          identity(std::move(other.identity)),
          register_routes(std::move(other.register_routes)),
          install(std::move(other.install)),
          kind(other.kind) {}
    PendingNativePackage& operator=(PendingNativePackage&& other) noexcept {
      static_assert(std::is_nothrow_move_assignable_v<std::shared_ptr<void>>);
      static_assert(std::is_nothrow_move_assignable_v<std::string>);
      static_assert(std::is_nothrow_move_assignable_v<std::function<void()>>);
      if (this != &other) {
        // ``previous`` first acquires this package's lifetime, then takes its callable targets. It
        // remains alive until the replacement is complete and destroys those targets before its
        // lifetime. The destination likewise acquires the incoming lifetime before its callables.
        PendingNativePackage previous(std::move(*this));
        lifetime = std::move(other.lifetime);
        identity = std::move(other.identity);
        register_routes = std::move(other.register_routes);
        install = std::move(other.install);
        kind = other.kind;
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
  std::optional<runtime::system::AuxiliaryStorageGroups<Dim>> provider_carrier_;
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
  std::shared_ptr<FieldNullspaceProviderRegistry<Dim>> field_nullspace_providers_;
  bool field_plan_consensus_verified_ = false;
  std::string default_nullspace_provider_identity_;
  PreparedProviderOptions default_nullspace_options_;

  /// Full assembly rollback image for the native package finalizer. DSO route registration and
  /// block installation both mutate only the candidate image after this snapshot exists, so a
  /// partial registrar or installer can never leak into a retry on any MPI rank.
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
    std::optional<runtime::system::AuxiliaryStorageGroups<Dim>> provider_carrier;
    std::map<std::string, std::vector<double>> staged_auxiliary_inputs;
    std::vector<std::string> dirty_auxiliary_providers;
    bool auxiliary_registry_consensus_verified = false;
    block_store_type blocks;
    std::vector<BoundaryHookImage> boundary_hooks;
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
    std::shared_ptr<exact_field_type> active_field;
    std::map<std::string, FieldPlan> field_plans;
    std::map<std::string, ConfiguredFieldSolverProvider> configured_field_solver_providers;
    std::map<std::string, std::shared_ptr<component_field_solver_type>>
        component_field_solver_providers;
    std::shared_ptr<FieldNullspaceProviderRegistry<Dim>> field_nullspace_providers;
    bool field_plan_consensus_verified = false;
    std::string default_nullspace_provider_identity;
    PreparedProviderOptions default_nullspace_options;

    explicit NativePackageFinalizeSnapshot(const Impl& owner)
        : auxiliary_registry(owner.auxiliary_registry_),
          provider_carrier(owner.provider_carrier_),
          staged_auxiliary_inputs(owner.staged_auxiliary_inputs_),
          dirty_auxiliary_providers(owner.dirty_auxiliary_providers_),
          auxiliary_registry_consensus_verified(owner.auxiliary_registry_consensus_verified_),
          blocks(owner.blocks_),
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
          field_nullspace_providers(owner.field_nullspace_providers_),
          field_plan_consensus_verified(owner.field_plan_consensus_verified_),
          default_nullspace_provider_identity(owner.default_nullspace_provider_identity_),
          default_nullspace_options(owner.default_nullspace_options_) {
      if (owner.active_field_ || owner.active_field_provider_candidate_ ||
          owner.active_field_auxiliary_publication_ ||
          !owner.active_field_stale_auxiliary_providers_.empty())
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
      for (const auto& block : owner.blocks_.blocks) {
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
      for (const auto& [slot, field] : owner.named_fields_) {
        (void)slot;
        if (!field)
          throw std::logic_error("System native package finalization found a null named field");
      }
    }

    /// Publish the preallocated rollback image by swaps only.  This path runs after a collective
    /// installer failure, when allocating while packages still hold DSO lifetimes would permit a
    /// rank-asymmetric rollback escape.
    void restore_noexcept(Impl& owner) noexcept {
      using std::swap;
      owner.auxiliary_registry_.swap_complete(auxiliary_registry);
      swap(owner.provider_carrier_, provider_carrier);
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

  void require_field_plan_consensus() {
    if (field_plan_consensus_verified_)
      return;
    std::string bytes;
    std::exception_ptr local_error;
    try {
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
        const auto configured =
            configured_field_solver_providers_.find(plan.backend_provider_route);
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
      bytes = std::move(registry).release();
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

    explicit AcceptedSnapshot(const Impl& owner)
        : auxiliary_registry(owner.auxiliary_registry_),
          provider_carrier(owner.provider_carrier_),
          staged_auxiliary_inputs(owner.staged_auxiliary_inputs_),
          dirty_auxiliary_providers(owner.dirty_auxiliary_providers_),
          program(owner.program_),
          time(owner.t),
          macro_step(owner.macro_step_) {
      if (owner.active_field_ || owner.active_field_provider_candidate_ ||
          owner.active_field_auxiliary_publication_ ||
          !owner.active_field_stale_auxiliary_providers_.empty())
        throw std::logic_error(
            "System cannot snapshot an unconsumed exact field publication candidate");
      states.reserve(owner.sp.size());
      for (const Species& block : owner.sp)
        states.push_back(block.U);
      if (owner.default_field_)
        default_field_state = owner.default_field_->accepted_state();
      for (const auto& [slot, field] : owner.named_fields_) {
        if (!field)
          throw std::logic_error("System materialized named field is null");
        named_field_states.emplace(slot, field->accepted_state());
      }
    }

    void restore(Impl& owner) const {
      if (states.size() != owner.sp.size())
        throw std::logic_error("System transaction snapshot composition changed");
      for (std::size_t block = 0; block < states.size(); ++block)
        owner.sp[block].U = states[block];
      owner.auxiliary_registry_ = auxiliary_registry;
      owner.provider_carrier_ = provider_carrier;
      owner.active_field_provider_candidate_.reset();
      owner.active_field_auxiliary_publication_.reset();
      owner.active_field_stale_auxiliary_providers_.clear();
      owner.staged_auxiliary_inputs_ = staged_auxiliary_inputs;
      owner.dirty_auxiliary_providers_ = dirty_auxiliary_providers;
      owner.program_ = program;
      if (default_field_state.has_value() != static_cast<bool>(owner.default_field_))
        throw std::logic_error("System transaction snapshot default-field ownership changed");
      if (default_field_state)
        owner.default_field_->restore_accepted_state(*default_field_state);
      if (named_field_states.size() != owner.named_fields_.size())
        throw std::logic_error("System transaction snapshot named-field composition changed");
      for (const auto& [slot, values] : named_field_states) {
        const auto field = owner.named_fields_.find(slot);
        if (field == owner.named_fields_.end() || !field->second)
          throw std::logic_error("System transaction snapshot field ownership changed");
        field->second->restore_accepted_state(values);
      }
      owner.t = time;
      owner.macro_step_ = macro_step;
    }
  };

  std::unique_ptr<AcceptedSnapshot> external_step_transaction_;
  bool external_step_transaction_committed_ = false;

  explicit Impl(const SystemConfig<Dim>& config)
      : domain_(config),
        field_nullspace_providers_(make_default_field_nullspace_provider_registry<Dim>()) {
    const FieldNullspaceProviderSelection selection = operator_topology_zero_mean_nullspace();
    default_nullspace_provider_identity_ = selection.provider_identity;
    default_nullspace_options_ = selection.options;
  }

  PreparedFieldNullspace<Dim> prepare_uniform_field_nullspace(
      std::string plan_identity, std::string topology_identity,
      const FieldNullspaceProviderSelection& selection,
      const elliptic::nd::CartesianPoissonOptions<Dim>& options, const field_type& layout,
      bool has_reaction, const ExecutionLane& lane) const {
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
    return prepare_field_nullspace_collectively<Dim>(*field_nullspace_providers_, selection,
                                                     std::move(request), lane);
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

  template <class Function>
  decltype(auto) execute_step_transaction(Function&& function) {
    AcceptedSnapshot snapshot(*this);
    try {
      return std::forward<Function>(function)();
    } catch (...) {
      snapshot.restore(*this);
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
