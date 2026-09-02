#pragma once

#include "component_abi_test_helpers.hpp"

#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/dynamic/prepared_execution_context.hpp>

#include <memory>
#include <string_view>
#include <utility>

namespace pops::test {

/// Install the explicit owned RuntimeInstance authority required before first AMR materialization.
/// This is test-only world-authority construction; production receives its authenticated parent
/// from the Python/native RuntimeInstance binding.
template <int Dim>
inline void install_amr_runtime_authority(AmrSystem<Dim>& system, std::string_view identity) {
  auto lane =
      std::make_shared<ExecutionLane>(ExecutionLane::duplicate_world_collectively(identity));
  const PopsExecutionContextV1 raw = component::test_support::host_execution_context();
  const component::PreparedExecutionContextV1 parent(
      raw.execution_identity, raw.context_version, raw.memory_space, raw.backend_identity,
      raw.device_identity, raw.scalar_type, raw.storage_precision, raw.compute_precision,
      raw.accumulation_precision, raw.reduction_precision, raw.stream_handle, raw.stream_identity,
      raw.communicator_f_handle, raw.communicator_datatype_f_handle, raw.communicator_identity,
      raw.communicator_datatype_identity);
  auto execution =
      std::make_shared<const component::PreparedExecutionContextV1>(parent.for_lane(*lane));
  system.install_prepared_boundary_execution_context(std::move(lane), std::move(execution));
}

}  // namespace pops::test
