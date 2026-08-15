#pragma once

#include <pops/runtime/config/generated_component_abi.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>
#include <pops/runtime/dynamic/prepared_execution_context.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {

struct PreparedBoundaryRegion {
  PopsBoundaryRegionKindV1 kind = POPS_BOUNDARY_FACE_V1;
  int dimension = 0;
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

enum class PreparedBoundaryOperation { GhostRegion, FluxTransform, FieldResidual, FieldJvp };

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

    [[nodiscard]] const PopsBoundaryFluxApiV1& boundary_flux_api() const {
      static_assert(Operation == PreparedBoundaryOperation::FluxTransform);
      return component_->table<PopsBoundaryFluxApiV1>(POPS_NATIVE_INTERFACE_BOUNDARY_FLUX_V1,
                                                      spec_.interface_version);
    }

    [[nodiscard]] const PopsFieldBoundaryClosureApiV1& field_api() const {
      static_assert(Operation == PreparedBoundaryOperation::FieldResidual ||
                    Operation == PreparedBoundaryOperation::FieldJvp);
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

  const PopsBoundaryFluxApiV1& boundary_flux_api() const {
    static_assert(Operation == PreparedBoundaryOperation::FluxTransform);
    return component_->table<PopsBoundaryFluxApiV1>(POPS_NATIVE_INTERFACE_BOUNDARY_FLUX_V1,
                                                    spec_.interface_version);
  }

  const PopsFieldBoundaryClosureApiV1& field_api() const {
    static_assert(Operation == PreparedBoundaryOperation::FieldResidual ||
                  Operation == PreparedBoundaryOperation::FieldJvp);
    return component_->table<PopsFieldBoundaryClosureApiV1>(
        POPS_NATIVE_INTERFACE_FIELD_BOUNDARY_CLOSURE_V1, spec_.interface_version);
  }

  /// Execute one exact-ranked physical ghost region through the generated ABI. All component
  /// calls write host scratch first; the runtime field is published only after every local patch
  /// succeeds. The enclosing System boundary transaction supplies collective rollback if a later
  /// compiled-boundary operation fails.
  template <int Dim>
  void apply_ghost_region(const runtime::multiblock::BoundaryEvaluationPoint& point,
                          MultiFab<Dim>& state, const Geometry<Dim>& geometry,
                          const ExecutionLane& lane) const
    requires(Operation == PreparedBoundaryOperation::GhostRegion)
  {
    static_assert(Dim >= 1 && Dim <= 3);
    static_assert(sizeof(Real) == sizeof(double),
                  "GhostBoundary ABI v1 requires the binary64 PoPS backend");
    if (spec_.region.dimension != Dim || state.ncomp() < 1 || point.clock.empty() ||
        point.level != 0 || point.stage_fraction.denominator <= 0 ||
        !std::isfinite(point.dt) || !std::isfinite(point.physical_time))
      throw std::invalid_argument(
          "prepared ghost boundary received an incomplete exact-ranked invocation");
    if (!spec_.states.empty() || !spec_.directions.empty() || !spec_.fields.empty() ||
        spec_.outputs.size() != 1 || spec_.outputs.front() != spec_.state_identity)
      throw std::invalid_argument(
          "Uniform GhostBoundary requires one primary-state output and no routed dependencies");

    struct PatchPublication {
      std::size_t local = 0;
      typename Fab<Dim>::host_mirror_type host;
      std::vector<std::size_t> offsets;
      std::vector<double> values;
    };
    std::vector<PatchPublication> publications;
    auto session = make_session(lane);
    const PopsLogicalTimeV1 logical_time{sizeof(PopsLogicalTimeV1),
                                         point.clock.c_str(),
                                         point.tick,
                                         point.level,
                                         point.substep,
                                         point.stage,
                                         point.stage_fraction.numerator,
                                         point.stage_fraction.denominator,
                                         point.dt,
                                         point.physical_time};
    std::vector<PopsQualifiedScalarV1> parameters;
    parameters.reserve(spec_.parameter_ids.size());
    for (std::size_t index = 0; index < spec_.parameter_ids.size(); ++index)
      parameters.push_back({sizeof(PopsQualifiedScalarV1), spec_.parameter_ids[index].c_str(),
                            spec_.parameter_values[index]});

    for (std::size_t local = 0; local < state.local_size(); ++local) {
      Fab<Dim>& fab = state.fab(local);
      const Box<Dim>& valid = fab.box();
      bool touches = true;
      for (std::size_t boundary_axis = 0; boundary_axis < spec_.region.axes.size();
           ++boundary_axis) {
        const int axis = spec_.region.axes[boundary_axis];
        const int side = spec_.region.sides[boundary_axis];
        touches = touches &&
                  (side < 0 ? valid.lo[axis] == geometry.domain().lo[axis]
                            : valid.hi[axis] == geometry.domain().hi[axis]);
      }
      if (!touches)
        continue;

      Box<Dim> region = valid;
      for (std::size_t boundary_axis = 0; boundary_axis < spec_.region.axes.size();
           ++boundary_axis) {
        const int axis = spec_.region.axes[boundary_axis];
        const int side = spec_.region.sides[boundary_axis];
        const int depth = state.ghosts()[axis];
        if (depth < 1)
          throw std::invalid_argument(
              "prepared ghost boundary region requires positive allocated depth");
        if (side < 0) {
          region.lo[axis] = valid.lo[axis] - depth;
          region.hi[axis] = valid.lo[axis] - 1;
        } else {
          region.lo[axis] = valid.hi[axis] + 1;
          region.hi[axis] = valid.hi[axis] + depth;
        }
      }
      if (!fab.grown_box().contains(region))
        throw std::invalid_argument(
            "prepared ghost boundary exact region exceeds its Fab storage");

      std::array<std::size_t, 3> extents{1, 1, 1};
      std::array<std::ptrdiff_t, 3> strides{0, 0, 0};
      std::size_t points = 1;
      for (int axis = 0; axis < Dim; ++axis) {
        extents[axis] = static_cast<std::size_t>(region.length(axis));
        strides[axis] = static_cast<std::ptrdiff_t>(points);
        points *= extents[axis];
      }
      const std::size_t components = static_cast<std::size_t>(state.ncomp());
      std::vector<double> interior(points * components);
      std::vector<double> output(points * components);
      std::vector<double> coordinates(points * static_cast<std::size_t>(Dim));
      std::vector<std::size_t> storage_offsets(points * components);
      auto host = fab.create_host_mirror();
      fab.copy_to_host(host);
      const FieldView<const Real, Dim> storage = std::as_const(fab).view();

      std::size_t point_index = 0;
      Index<Dim> index = region.lo;
      std::function<void(int)> visit = [&](int axis) {
        if (axis == Dim) {
          Index<Dim> source = index;
          for (std::size_t boundary_axis = 0; boundary_axis < spec_.region.axes.size();
               ++boundary_axis) {
            const int selected = spec_.region.axes[boundary_axis];
            source[selected] = spec_.region.sides[boundary_axis] < 0 ? valid.lo[selected]
                                                                      : valid.hi[selected];
          }
          std::size_t ghost_offset = 0;
          std::size_t source_offset = 0;
          for (int selected = 0; selected < Dim; ++selected) {
            ghost_offset += static_cast<std::size_t>(index[selected] - storage.origin[selected]) *
                            static_cast<std::size_t>(storage.strides[selected]);
            source_offset += static_cast<std::size_t>(source[selected] - storage.origin[selected]) *
                             static_cast<std::size_t>(storage.strides[selected]);
            coordinates[point_index + static_cast<std::size_t>(selected) * points] =
                static_cast<double>(geometry.cell_coordinate(selected, index[selected]));
          }
          for (std::size_t component = 0; component < components; ++component) {
            const std::size_t packed = point_index + component * points;
            const std::size_t ghost = ghost_offset + component * storage.component_stride;
            const std::size_t source_value =
                source_offset + component * storage.component_stride;
            interior[packed] = static_cast<double>(host(source_value));
            output[packed] = static_cast<double>(host(ghost));
            storage_offsets[packed] = ghost;
          }
          ++point_index;
          return;
        }
        for (int value = region.lo[axis]; value <= region.hi[axis]; ++value) {
          index[axis] = value;
          visit(axis + 1);
        }
      };
      visit(0);

      const std::string patch_identity =
          spec_.region.identity + "::patch:" + std::to_string(local);
      auto const_view = [&](const void* data, std::size_t component_count) {
        return PopsConstFieldViewV1{sizeof(PopsConstFieldViewV1), data,
                                    static_cast<std::uint32_t>(Dim),
                                    {extents[0], extents[1], extents[2]},
                                    {strides[0], strides[1], strides[2]}, component_count,
                                    static_cast<std::ptrdiff_t>(points),
                                    POPS_FIELD_CENTERING_CELL_V1, 0, {0, 0, 0}, {0, 0, 0},
                                    POPS_SCALAR_FLOAT64_V1, POPS_MEMORY_SPACE_HOST_V1,
                                    spec_.layout_identity.c_str(), patch_identity.c_str(),
                                    POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
      };
      const PopsConstFieldViewV1 interior_view = const_view(interior.data(), components);
      const PopsConstFieldViewV1 coordinate_view =
          const_view(coordinates.data(), static_cast<std::size_t>(Dim));
      PopsFieldViewV1 ghost_view{sizeof(PopsFieldViewV1), output.data(), interior_view.dimension,
                                 {extents[0], extents[1], extents[2]},
                                 {strides[0], strides[1], strides[2]}, components,
                                 static_cast<std::ptrdiff_t>(points),
                                 POPS_FIELD_CENTERING_CELL_V1, 0, {0, 0, 0}, {0, 0, 0},
                                 POPS_SCALAR_FLOAT64_V1, POPS_MEMORY_SPACE_HOST_V1,
                                 spec_.layout_identity.c_str(), patch_identity.c_str(),
                                 POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
      PopsComponentStatusV1 status{sizeof(PopsComponentStatusV1), 0,
                                   POPS_COMPONENT_CONTINUE_V1, nullptr};
      const PopsGhostBoundaryRequestV1 request{
          sizeof(PopsGhostBoundaryRequestV1), spec_.producer_identity.c_str(),
          spec_.state_identity.c_str(), spec_.ghost_identity.c_str(), interior_view, ghost_view,
          coordinate_view, spec_.region.view(), 0, nullptr, parameters.size(), parameters.data(),
          logical_time, session.execution().view()};
      const int code = component::apply_ghost_boundary(
          session.ghost_api(), session.state(), request, status);
      require_success(code, status, "apply_region_batch");
      if (std::any_of(output.begin(), output.end(),
                      [](double value) { return !std::isfinite(value); }))
        throw std::runtime_error("native boundary component published a non-finite ghost value");
      publications.push_back(
          {local, std::move(host), std::move(storage_offsets), std::move(output)});
    }

    for (auto& publication : publications) {
      for (std::size_t value = 0; value < publication.values.size(); ++value)
        publication.host(publication.offsets[value]) =
            static_cast<Real>(publication.values[value]);
      state.fab(publication.local).copy_from_host(publication.host);
    }
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
    else if constexpr (Operation == PreparedBoundaryOperation::FluxTransform)
      return POPS_NATIVE_INTERFACE_BOUNDARY_FLUX_V1;
    else
      return POPS_NATIVE_INTERFACE_FIELD_BOUNDARY_CLOSURE_V1;
  }

  const PopsComponentTableHeaderV1& table_header() const {
    if constexpr (Operation == PreparedBoundaryOperation::GhostRegion)
      return ghost_api().header;
    else if constexpr (Operation == PreparedBoundaryOperation::FluxTransform)
      return boundary_flux_api().header;
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
      if (spec_.outputs.size() != 1 || spec_.outputs.front() != spec_.state_identity ||
          !spec_.states.empty() || !spec_.directions.empty() || !spec_.fields.empty())
        throw std::invalid_argument(
            "Uniform GhostBoundary requires one primary-state output and no routed dependencies");
    } else if constexpr (Operation == PreparedBoundaryOperation::FluxTransform) {
      component::require_operation(boundary_flux_api().transform_faces != nullptr,
                                   "transform_faces");
      if (spec_.region.kind != POPS_BOUNDARY_FACE_V1 || spec_.region.codimension != 1 ||
          spec_.outputs.size() != 1 || spec_.outputs.front() != spec_.state_identity ||
          !spec_.directions.empty())
        throw std::invalid_argument(
            "BoundaryFlux requires one oriented face, one state output and no direction table");
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
using PreparedBoundaryFluxComponent =
    PreparedBoundaryComponent<PreparedBoundaryOperation::FluxTransform>;
using PreparedFieldBoundaryResidualComponent =
    PreparedBoundaryComponent<PreparedBoundaryOperation::FieldResidual>;
using PreparedFieldBoundaryJvpComponent =
    PreparedBoundaryComponent<PreparedBoundaryOperation::FieldJvp>;

}  // namespace pops
