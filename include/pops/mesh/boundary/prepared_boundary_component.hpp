#pragma once

#include <pops/runtime/config/generated_component_abi.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>
#include <pops/runtime/dynamic/prepared_execution_context.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {

struct PreparedBoundaryRegion {
  PopsBoundaryRegionKindV1 kind = POPS_BOUNDARY_FACE_V1;
  int dimension = 2;
  int codimension = 1;
  std::vector<std::int32_t> axes;
  std::vector<std::int32_t> sides;
  std::string identity;

  PopsBoundaryRegionV1 view() const {
    return {sizeof(PopsBoundaryRegionV1),
            kind,
            dimension,
            codimension,
            axes.size(),
            axes.data(),
            sides.data(),
            identity.c_str()};
  }
};

struct PreparedBoundaryComponentSpec {
  std::string target_identity;
  std::string component_id;
  std::string manifest_identity;
  std::uint32_t interface_version = 1;
  std::string producer_identity;
  std::string state_identity;
  std::string ghost_identity;
  std::string layout_identity;
  PreparedBoundaryRegion region;
  std::vector<std::string> states;
  std::vector<std::string> directions;
  std::vector<std::string> fields;
  std::vector<std::string> parameter_ids;
  std::vector<double> parameter_values;
  std::vector<std::string> outputs;
  std::string rate;
  std::string nonlinear_iterate;
  std::string parameters_json;
  std::string target_json;
  std::shared_ptr<const component::PreparedExecutionContextV1> execution;
};

enum class PreparedBoundaryOperation { GhostRegion, FieldResidual, FieldJvp };

/// One statically typed prepared component invocation.  The operation is a template argument, never
/// a production string branch: installation chooses one typed entry point and scientific calls retain
/// its direct ABI table, state and exact execution context.
template <PreparedBoundaryOperation Operation>
class PreparedBoundaryComponent final {
 public:
  PreparedBoundaryComponent(PreparedBoundaryComponentSpec spec,
                            std::shared_ptr<component::LoadedComponent> component)
      : spec_(std::move(spec)), component_(std::move(component)) {
    validate();
    const PopsExecutionContextV1 execution = spec_.execution->view();
    state_ = component_->prepared_state(native_interface_id_(), spec_.interface_version, execution,
                                        spec_.parameters_json, spec_.target_json);
  }

  PreparedBoundaryComponent(const PreparedBoundaryComponent&) = delete;
  PreparedBoundaryComponent& operator=(const PreparedBoundaryComponent&) = delete;

  ~PreparedBoundaryComponent() = default;  // LoadedComponent owns prepared-state destruction.

  const PreparedBoundaryComponentSpec& spec() const { return spec_; }
  void* state() const { return state_; }

  const PopsGhostBoundaryApiV1& ghost_api() const {
    static_assert(Operation == PreparedBoundaryOperation::GhostRegion);
    return component_->table<PopsGhostBoundaryApiV1>(POPS_NATIVE_INTERFACE_GHOST_BOUNDARY_V1,
                                                     spec_.interface_version);
  }

  const PopsFieldBoundaryClosureApiV1& field_api() const {
    static_assert(Operation != PreparedBoundaryOperation::GhostRegion);
    return component_->table<PopsFieldBoundaryClosureApiV1>(
        POPS_NATIVE_INTERFACE_FIELD_BOUNDARY_CLOSURE_V1, spec_.interface_version);
  }

  static void require_success(int code, const PopsComponentStatusV1& status,
                              const char* operation) {
    if (code == 0 && status.code == 0 && status.action == POPS_COMPONENT_CONTINUE_V1)
      return;
    throw std::runtime_error(std::string("native boundary component ") + operation + " failed: " +
                             (status.reason == nullptr ? "no reason" : status.reason));
  }

 private:
  static constexpr PopsNativeInterfaceIdV1 native_interface_id_() {
    if constexpr (Operation == PreparedBoundaryOperation::GhostRegion)
      return POPS_NATIVE_INTERFACE_GHOST_BOUNDARY_V1;
    else
      return POPS_NATIVE_INTERFACE_FIELD_BOUNDARY_CLOSURE_V1;
  }

  const PopsComponentTableHeaderV1& table_header() const {
    if constexpr (Operation == PreparedBoundaryOperation::GhostRegion)
      return ghost_api().header;
    else
      return field_api().header;
  }

  void validate() const {
    if (!component_ || spec_.target_identity.empty() || spec_.component_id.empty() ||
        spec_.manifest_identity.empty() || spec_.producer_identity.empty() ||
        spec_.state_identity.empty() || spec_.ghost_identity.empty() ||
        spec_.layout_identity.empty() || spec_.region.identity.empty() ||
        spec_.parameter_ids.size() != spec_.parameter_values.size())
      throw std::invalid_argument("prepared boundary component identity/tables are incomplete");
    if (spec_.interface_version != 1)
      throw std::invalid_argument("prepared boundary component requires interface version 1");
    if (!spec_.execution)
      throw std::invalid_argument("prepared boundary component lacks ExecutionContext authority");
    component::validate_execution_context(spec_.execution->view());
    if constexpr (Operation == PreparedBoundaryOperation::GhostRegion) {
      component::require_operation(ghost_api().apply_region_batch != nullptr, "apply_region_batch");
    } else {
      component::require_operation(
          Operation == PreparedBoundaryOperation::FieldResidual ? field_api().residual != nullptr
                                                                : field_api().jvp != nullptr,
          Operation == PreparedBoundaryOperation::FieldResidual ? "residual" : "jvp");
      if (spec_.states.empty() || spec_.outputs.empty() ||
          (Operation == PreparedBoundaryOperation::FieldResidual && !spec_.directions.empty()) ||
          (Operation == PreparedBoundaryOperation::FieldJvp && spec_.directions.empty()))
        throw std::invalid_argument(
            "FieldBoundaryClosure direction table is inconsistent with its typed operation");
    }
    const auto& api = component_->api();
    if (api.component_id == nullptr || api.manifest_identity == nullptr ||
        spec_.component_id != api.component_id || spec_.manifest_identity != api.manifest_identity)
      throw std::invalid_argument("prepared boundary component changed native identity");
    component::validate_boundary_region(spec_.region.view());
  }

  PreparedBoundaryComponentSpec spec_;
  std::shared_ptr<component::LoadedComponent> component_;
  void* state_ = nullptr;
};

using PreparedGhostBoundaryComponent =
    PreparedBoundaryComponent<PreparedBoundaryOperation::GhostRegion>;
using PreparedFieldBoundaryResidualComponent =
    PreparedBoundaryComponent<PreparedBoundaryOperation::FieldResidual>;
using PreparedFieldBoundaryJvpComponent =
    PreparedBoundaryComponent<PreparedBoundaryOperation::FieldJvp>;

}  // namespace pops
