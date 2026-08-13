/// @file
/// @brief Revocable, detached native-package capabilities for Uniform System packages.

#pragma once

#include <pops/mesh/geometry/geometry.hpp>
#include <pops/runtime/system/derived_aux_provider.hpp>
#include <pops/runtime/system/exact_aux_registry.hpp>
#include <pops/runtime/system/provider_storage_binding.hpp>
#include <pops/runtime/system/system_block_closures.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::runtime::system {

inline constexpr int kNativeSystemPackageAbiVersion = 2;
inline constexpr const char* kNativeSystemPackageAbiVersionSymbol =
    "pops_native_system_package_abi_version";

template <int Dim>
struct PreparedNativeEllipticAttachment {
  std::string field;
  std::string rhs_identity;
  std::vector<AuxiliaryComponentKey> outputs;
  int gradient_sign = 1;
  std::function<void(const MultiFab<Dim>&, MultiFab<Dim>&)> rhs;
};

template <int Dim>
struct PreparedNativeSystemPackage {
  PreparedSystemBlock<Dim> block;
  std::string consumer_qid;
  std::vector<PreparedNativeEllipticAttachment<Dim>> elliptic_attachments;
};

template <int Dim>
inline std::string exact_native_system_package_contract(
    const PreparedNativeSystemPackage<Dim>& package) {
  const auto encode_variables = [](ExactContractBuilder& contract, const VariableSet& variables) {
    contract.scalar(static_cast<std::int32_t>(variables.kind)).scalar(std::int32_t{variables.size});
    contract
        .sequence(variables.names,
                  [](ExactContractBuilder& item, const std::string& name) { item.text(name); })
        .sequence(
            variables.roles,
            [](ExactContractBuilder& item, const VariableSemantic& role) {
              item.scalar(static_cast<std::int32_t>(role.kind)).scalar(std::int32_t{role.axis});
            })
        .sequence(variables.user_roles,
                  [](ExactContractBuilder& item, const std::string& role) { item.text(role); });
  };
  const PreparedSystemBlock<Dim>& block = package.block;
  ExactContractBuilder contract;
  contract.text("pops.prepared-native-system-package")
      .scalar(std::uint32_t{2})
      .scalar(std::int32_t{Dim})
      .text(package.consumer_qid)
      .text(block.name)
      .text(block.provider_identity)
      .scalar(std::int32_t{block.ncomp})
      .scalar(std::int32_t{block.provider_components})
      .scalar(block.gamma)
      .scalar(std::int32_t{block.substeps})
      .scalar(block.evolve)
      .scalar(std::int32_t{block.stride});
  for (int axis = 0; axis < Dim; ++axis)
    contract.scalar(static_cast<std::int32_t>(block.ghosts[axis]));
  encode_variables(contract, block.conservative_variables);
  encode_variables(contract, block.primitive_variables);
  const auto& closures = block.closures;
  contract.presence(static_cast<bool>(closures.rhs_into))
      .presence(static_cast<bool>(closures.rhs_flux_only))
      .presence(static_cast<bool>(closures.source_only))
      .presence(static_cast<bool>(closures.source_only_masked))
      .presence(static_cast<bool>(closures.rhs_at_point))
      .presence(static_cast<bool>(closures.rhs_flux_only_at_point))
      .presence(static_cast<bool>(closures.rhs_without_prepared_interfaces))
      .presence(static_cast<bool>(closures.rhs_flux_only_without_prepared_interfaces))
      .presence(static_cast<bool>(closures.rhs_core_at_point))
      .presence(static_cast<bool>(closures.rhs_flux_only_core_at_point))
      .presence(static_cast<bool>(closures.rhs_core_at_point_prepared))
      .presence(static_cast<bool>(closures.rhs_flux_only_core_at_point_prepared))
      .presence(static_cast<bool>(closures.boundary_full_at_point_prepared))
      .presence(static_cast<bool>(closures.boundary_core_at_point_prepared))
      .presence(static_cast<bool>(closures.boundary_flux_full_at_point_prepared))
      .presence(static_cast<bool>(closures.boundary_flux_core_at_point_prepared))
      .presence(static_cast<bool>(closures.boundary_residual_at_point_prepared))
      .presence(static_cast<bool>(closures.boundary_jvp_at_point_prepared))
      .presence(static_cast<bool>(closures.external_boundary_flux))
      .presence(closures.external_boundary_flux &&
                static_cast<bool>(*closures.external_boundary_flux))
      .presence(static_cast<bool>(closures.external_field_boundary_residual))
      .presence(closures.external_field_boundary_residual &&
                static_cast<bool>(*closures.external_field_boundary_residual))
      .presence(static_cast<bool>(closures.external_field_boundary_jvp))
      .presence(closures.external_field_boundary_jvp &&
                static_cast<bool>(*closures.external_field_boundary_jvp))
      .presence(static_cast<bool>(closures.prepare_generated_state_at_point))
      .presence(static_cast<bool>(closures.prepare_generated_state_at_point_prepared))
      .presence(static_cast<bool>(closures.prepare_generated_state_with_transport_prepared))
      .presence(static_cast<bool>(closures.external_ghost_boundary))
      .presence(closures.external_ghost_boundary &&
                static_cast<bool>(*closures.external_ghost_boundary))
      .presence(static_cast<bool>(closures.project))
      .presence(static_cast<bool>(closures.project_masked))
      .presence(static_cast<bool>(closures.staircase.full))
      .presence(static_cast<bool>(closures.staircase.flux_only))
      .presence(static_cast<bool>(closures.staircase.source_only))
      .presence(static_cast<bool>(closures.staircase.project))
      .presence(static_cast<bool>(closures.cut_cell.full))
      .presence(static_cast<bool>(closures.cut_cell.flux_only))
      .presence(static_cast<bool>(closures.cut_cell.source_only))
      .presence(static_cast<bool>(closures.cut_cell.project))
      .presence(static_cast<bool>(block.maximum_speed))
      .presence(static_cast<bool>(block.poisson_rhs))
      .presence(static_cast<bool>(block.primitive_to_conservative))
      .presence(static_cast<bool>(block.conservative_to_primitive))
      .presence(static_cast<bool>(block.batch_conservative_to_primitive))
      .presence(static_cast<bool>(block.source_frequency))
      .presence(static_cast<bool>(block.stability_dt));
  contract.sequence(
      package.elliptic_attachments,
      [](ExactContractBuilder& item, const PreparedNativeEllipticAttachment<Dim>& attachment) {
        item.text(attachment.field)
            .text(attachment.rhs_identity)
            .scalar(std::int32_t{attachment.gradient_sign})
            .sequence(attachment.outputs,
                      [](ExactContractBuilder& output, const AuxiliaryComponentKey& key) {
                        output.text(key.owner_qid)
                            .text(key.space_kind)
                            .text(key.space_name)
                            .text(key.component);
                      })
            .presence(static_cast<bool>(attachment.rhs));
      });
  return std::move(contract).release();
}

enum class NativeCapabilityPhase : unsigned char {
  routes_open,
  routes_closed,
  install_open,
  revoked
};

template <int Dim>
struct NativePackageCapabilityState final {
  std::string identity;
  NativeCapabilityPhase phase = NativeCapabilityPhase::revoked;
  std::optional<Geometry<Dim>> geometry;
  std::array<bool, Dim> periodicity{};
  std::shared_ptr<const AuxiliaryStorageGroups<Dim>> provider_storage_owner;
  std::shared_ptr<ExactAuxiliaryRegistry<Dim>> detached_registry;
  // Registry plans live in a value-swapped registry candidate. Keep the one selected plan in
  // package-owned storage so generated closures never retain a vector element from that candidate.
  std::shared_ptr<const ResolvedAuxiliaryConsumerPlan<Dim>> resolved_consumer_plan;
  std::optional<PreparedNativeSystemPackage<Dim>> committed;
  bool commit_called = false;
  bool revoke_called = false;

  void require(NativeCapabilityPhase expected, const char* operation) const {
    if (phase == NativeCapabilityPhase::revoked)
      throw std::logic_error(std::string(operation) + ": native package capability is revoked");
    if (phase != expected)
      throw std::logic_error(std::string(operation) + ": native package capability phase mismatch");
  }
  void revoke() noexcept {
    if (revoke_called)
      return;
    revoke_called = true;
    phase = NativeCapabilityPhase::revoked;
    provider_storage_owner.reset();
    detached_registry.reset();
  }
  void reset_for_retry() noexcept {
    static_assert(std::is_nothrow_destructible_v<PreparedNativeSystemPackage<Dim>>);
    committed.reset();
    resolved_consumer_plan.reset();
    provider_storage_owner.reset();
    detached_registry.reset();
    geometry.reset();
    phase = NativeCapabilityPhase::revoked;
    commit_called = false;
    revoke_called = false;
  }
  void close_routes() {
    require(NativeCapabilityPhase::routes_open, "close native package routes");
    phase = NativeCapabilityPhase::routes_closed;
  }
};

template <int Dim>
class PreparedNativeRouteRegistrar final {
 public:
  PreparedNativeRouteRegistrar(const PreparedNativeRouteRegistrar&) = delete;
  PreparedNativeRouteRegistrar& operator=(const PreparedNativeRouteRegistrar&) = delete;
  PreparedNativeRouteRegistrar(PreparedNativeRouteRegistrar&&) = delete;
  PreparedNativeRouteRegistrar& operator=(PreparedNativeRouteRegistrar&&) = delete;
  void install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<Dim> provider) {
    state_->require(NativeCapabilityPhase::routes_open, "install_prepared_auxiliary_provider");
    if (!state_->detached_registry)
      throw std::logic_error("native provider registrar has no detached registry candidate");
    state_->detached_registry->add(std::move(provider));
  }
  void install_auxiliary_consumer_plan(AuxiliaryConsumerProviderPlan<Dim> plan) {
    state_->require(NativeCapabilityPhase::routes_open, "install_auxiliary_consumer_plan");
    if (!state_->detached_registry)
      throw std::logic_error("native provider registrar has no detached registry candidate");
    state_->detached_registry->add_consumer_plan(std::move(plan));
  }

 private:
  explicit PreparedNativeRouteRegistrar(std::shared_ptr<NativePackageCapabilityState<Dim>> state)
      : state_(std::move(state)) {}
  std::shared_ptr<NativePackageCapabilityState<Dim>> state_;
  template <int>
  friend class NativePackageCapabilityFactory;
};

template <int Dim>
class PreparedNativeBlockInstaller final {
 public:
  PreparedNativeBlockInstaller(const PreparedNativeBlockInstaller&) = delete;
  PreparedNativeBlockInstaller& operator=(const PreparedNativeBlockInstaller&) = delete;
  PreparedNativeBlockInstaller(PreparedNativeBlockInstaller&&) = delete;
  PreparedNativeBlockInstaller& operator=(PreparedNativeBlockInstaller&&) = delete;
  [[nodiscard]] Geometry<Dim> geometry() const {
    state_->require(NativeCapabilityPhase::install_open, "PreparedNativeBlockInstaller::geometry");
    if (!state_->geometry)
      throw std::logic_error("native block installer has no prepared geometry");
    return *state_->geometry;
  }
  [[nodiscard]] std::array<bool, Dim> periodicity() const {
    state_->require(NativeCapabilityPhase::install_open,
                    "PreparedNativeBlockInstaller::periodicity");
    return state_->periodicity;
  }
  [[nodiscard]] std::shared_ptr<const AuxiliaryStorageGroups<Dim>> provider_storage() const {
    state_->require(NativeCapabilityPhase::install_open,
                    "PreparedNativeBlockInstaller::provider_storage");
    return state_->provider_storage_owner;
  }
  [[nodiscard]] std::shared_ptr<const ResolvedAuxiliaryConsumerPlan<Dim>> consumer_plan(
      const std::string& consumer_qid) const {
    state_->require(NativeCapabilityPhase::install_open,
                    "PreparedNativeBlockInstaller::consumer_plan");
    if (!state_->detached_registry || consumer_qid.empty())
      throw std::logic_error("native block installer has no resolved consumer-plan authority");
    const auto& resolved = state_->detached_registry->consumer_plan(consumer_qid);
    if (state_->resolved_consumer_plan &&
        state_->resolved_consumer_plan->consumer_qid != consumer_qid)
      throw std::logic_error("native block installer selected multiple consumer plans");
    if (!state_->resolved_consumer_plan)
      state_->resolved_consumer_plan =
          std::make_shared<ResolvedAuxiliaryConsumerPlan<Dim>>(resolved);
    return state_->resolved_consumer_plan;
  }
  void commit(PreparedNativeSystemPackage<Dim> package) {
    state_->require(NativeCapabilityPhase::install_open, "PreparedNativeBlockInstaller::commit");
    if (state_->commit_called)
      throw std::logic_error("native package installer committed more than once");
    if (package.block.name.empty() || package.consumer_qid.empty() ||
        package.block.name != state_->identity)
      throw std::invalid_argument(
          "native package installer committed a block outside its exact package identity");
    if (package.block.provider_components != 0 &&
        (!state_->resolved_consumer_plan ||
         state_->resolved_consumer_plan->consumer_qid != package.consumer_qid ||
         state_->resolved_consumer_plan->value_count() !=
             static_cast<std::size_t>(package.block.provider_components)))
      throw std::invalid_argument(
          "native package block differs from its exact resolved consumer plan");
    if (package.block.provider_components == 0 && state_->resolved_consumer_plan)
      throw std::invalid_argument(
          "provider-free native package selected an auxiliary consumer plan");
    for (std::size_t index = 0; index < package.elliptic_attachments.size(); ++index) {
      const auto& attachment = package.elliptic_attachments[index];
      if (attachment.field.empty() || attachment.rhs_identity.empty() || !attachment.rhs ||
          (attachment.field != "fields_from_state" && attachment.outputs.empty()) ||
          (attachment.gradient_sign != -1 && attachment.gradient_sign != 1))
        throw std::invalid_argument("native package committed an incomplete elliptic attachment");
      for (std::size_t previous = 0; previous < index; ++previous)
        if (package.elliptic_attachments[previous].field == attachment.field)
          throw std::invalid_argument(
              "native package committed one elliptic attachment more than once");
    }
    state_->committed.emplace(std::move(package));
    state_->commit_called = true;
  }

 private:
  explicit PreparedNativeBlockInstaller(std::shared_ptr<NativePackageCapabilityState<Dim>> state)
      : state_(std::move(state)) {}
  std::shared_ptr<NativePackageCapabilityState<Dim>> state_;
  template <int>
  friend class NativePackageCapabilityFactory;
};

template <int Dim>
class NativePackageCapabilityFactory final {
 public:
  static std::shared_ptr<PreparedNativeRouteRegistrar<Dim>> route_registrar(
      const std::shared_ptr<NativePackageCapabilityState<Dim>>& state) {
    return std::shared_ptr<PreparedNativeRouteRegistrar<Dim>>(
        new PreparedNativeRouteRegistrar<Dim>(state));
  }
  static std::shared_ptr<PreparedNativeBlockInstaller<Dim>> block_installer(
      const std::shared_ptr<NativePackageCapabilityState<Dim>>& state) {
    return std::shared_ptr<PreparedNativeBlockInstaller<Dim>>(
        new PreparedNativeBlockInstaller<Dim>(state));
  }
};

}  // namespace pops::runtime::system
