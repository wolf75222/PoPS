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
  /// One lane-bound invocation session with an independently prepared component state.
  class Session final {
   public:
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&&) noexcept = default;
    Session& operator=(Session&&) noexcept = default;
    ~Session() = default;

    [[nodiscard]] const PreparedBoundaryComponentSpec& spec() const noexcept { return spec_; }
    [[nodiscard]] void* state() const noexcept { return state_.get(); }
    [[nodiscard]] const component::PreparedExecutionContextV1& execution() const noexcept {
      return *execution_;
    }

    [[nodiscard]] const PopsGhostBoundaryApiV1& ghost_api() const {
      static_assert(Operation == PreparedBoundaryOperation::GhostRegion);
      return component_->table<PopsGhostBoundaryApiV1>(POPS_NATIVE_INTERFACE_GHOST_BOUNDARY_V1,
                                                       spec_.interface_version);
    }

    [[nodiscard]] const PopsFieldBoundaryClosureApiV1& field_api() const {
      static_assert(Operation != PreparedBoundaryOperation::GhostRegion);
      return component_->table<PopsFieldBoundaryClosureApiV1>(
          POPS_NATIVE_INTERFACE_FIELD_BOUNDARY_CLOSURE_V1, spec_.interface_version);
    }

   private:
    friend class PreparedBoundaryComponent;

    Session(PreparedBoundaryComponentSpec spec,
            std::shared_ptr<component::LoadedComponent> component,
            std::shared_ptr<const component::PreparedExecutionContextV1> execution,
            component::LoadedComponent::PreparedState state)
        : spec_(std::move(spec)),
          component_(std::move(component)),
          execution_(std::move(execution)),
          state_(std::move(state)) {
      spec_.execution = execution_;
    }

    PreparedBoundaryComponentSpec spec_;
    // Declaration order is intentional: state_ is destroyed before the execution strings and the
    // LoadedComponent that keeps its destroy callback's dynamic library resident.
    std::shared_ptr<component::LoadedComponent> component_;
    std::shared_ptr<const component::PreparedExecutionContextV1> execution_;
    component::LoadedComponent::PreparedState state_;
  };

  PreparedBoundaryComponent(PreparedBoundaryComponentSpec spec,
                            std::shared_ptr<component::LoadedComponent> component)
      : spec_(std::move(spec)), component_(std::move(component)) {
    validate();
  }

  PreparedBoundaryComponent(const PreparedBoundaryComponent&) = delete;
  PreparedBoundaryComponent& operator=(const PreparedBoundaryComponent&) = delete;

  ~PreparedBoundaryComponent() = default;

  const PreparedBoundaryComponentSpec& spec() const { return spec_; }

  /// Materialize one fresh state before entering the numerical hot path. The returned move-only
  /// session retains the component library, the lane-qualified execution POD and its own native
  /// state for the complete invocation lifetime.
  [[nodiscard]] Session make_session(const ExecutionLane& lane) const {
    auto execution = std::make_shared<const component::PreparedExecutionContextV1>(
        spec_.execution->for_lane(lane));
    if (!execution->matches_lane(lane))
      throw std::invalid_argument(
          "prepared boundary component execution authority differs from its lane");
    if (execution->view().memory_space != POPS_MEMORY_SPACE_HOST_V1)
      throw std::invalid_argument(
          "prepared boundary component uses the host-batch ABI but its exact ExecutionContext "
          "requires a non-host memory space; install a device-native boundary provider instead");
    auto state = component_->prepare_fresh_state(native_interface_id_(), spec_.interface_version,
                                                 execution->view(), spec_.parameters_json,
                                                 spec_.target_json);
    return Session(spec_, component_, std::move(execution), std::move(state));
  }

  /// Sequential control-path adapter. Production kernels retain the result of make_session(lane)
  /// instead of preparing during an invocation.
  [[nodiscard]] Session make_world_session() const {
    const auto lane = ExecutionLane::world();
    return make_session(lane);
  }

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
};

using PreparedGhostBoundaryComponent =
    PreparedBoundaryComponent<PreparedBoundaryOperation::GhostRegion>;
using PreparedFieldBoundaryResidualComponent =
    PreparedBoundaryComponent<PreparedBoundaryOperation::FieldResidual>;
using PreparedFieldBoundaryJvpComponent =
    PreparedBoundaryComponent<PreparedBoundaryOperation::FieldJvp>;

}  // namespace pops
