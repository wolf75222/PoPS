/// @file
/// @brief Exact compact provider-value binding for native runtime consumers.

#pragma once

#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/spatial/primitives/state_access.hpp>
#include <pops/runtime/system/derived_aux_provider.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace pops::runtime::system {

/// Materialize the device-copyable compact map for one local patch from a sealed consumer plan.
///
/// The registry owns all key and contract resolution.  Native kernels receive only their dense
/// consumer slots, whose storage addresses remain late-bound to the prepared runtime.  A
/// provider-free consumer has an exact empty view and must not consult either a plan or storage.
template <int Dim, int Count>
[[nodiscard]] ProviderStorageView<Dim, Count> bind_provider_storage_view(
    const ResolvedAuxiliaryConsumerPlan<Dim>* plan, const AuxiliaryStorageGroups<Dim>* storage,
    std::size_t local) {
  if constexpr (Count == 0) {
    return {};
  } else {
    if (plan == nullptr)
      throw std::invalid_argument("provider consumer requires a sealed plan");
    if (plan->value_count() != static_cast<std::size_t>(Count))
      throw std::invalid_argument("provider plan count differs from native consumer");
    if (storage == nullptr)
      throw std::invalid_argument("provider consumer requires accepted storage groups");

    ProviderStorageView<Dim, Count> result{};
    for (const auto& value : plan->values) {
      const MultiFab<Dim>* const group = storage->find(value.address.group);
      if (value.consumer_slot >= static_cast<std::size_t>(Count) || group == nullptr ||
          value.address.component >= static_cast<std::size_t>(group->ncomp()) ||
          local >= group->local_size())
        throw std::invalid_argument("provider plan has an invalid resolved slot");
      result.storage[value.consumer_slot] = group->fab(local).view();
      result.storage_components[value.consumer_slot] = static_cast<int>(value.address.component);
    }
    return result;
  }
}

/// Prove that every pointwise provider group is directly indexable with a state patch.
///
/// Non-cell-centred, vector, or differently decomposed values must be projected by their provider
/// before this seam.  Sampling them as a scalar cell-centred field would silently change the model.
template <int Dim, int Count>
void require_pointwise_provider_groups(
    const MultiFab<Dim>& state, const AuxiliaryStorageGroups<Dim>* storage,
    const ResolvedAuxiliaryConsumerPlan<Dim>* plan, const char* operation) {
  if constexpr (Count == 0) {
    // An unrelated consumer can still own provider groups in the same System.  This consumer must
    // neither inspect nor require them: its compact ABI is exactly empty.
    (void)state;
    (void)storage;
    (void)plan;
    (void)operation;
  } else {
    if (storage == nullptr || plan == nullptr ||
        plan->value_count() != static_cast<std::size_t>(Count))
      throw std::invalid_argument(std::string(operation) + ": missing compact provider plan");
    for (const auto& value : plan->values) {
      const MultiFab<Dim>* const group = storage->find(value.address.group);
      if (group == nullptr || value.address.component >= static_cast<std::size_t>(group->ncomp()) ||
          group->layout() != state.layout() || group->distribution() != state.distribution() ||
          group->local_rank() != state.local_rank() || group->local_size() != state.local_size())
        throw std::invalid_argument(std::string(operation) +
                                    ": provider group is not pointwise compatible with state");
      if (value.contract.centering != "cell" || value.shape.value_components != 1)
        throw std::invalid_argument(std::string(operation) +
                                    ": non-cell-scalar provider requires an explicit projection");
    }
  }
}

}  // namespace pops::runtime::system
