#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>
#include <pops/runtime/dynamic/prepared_execution_context.hpp>
#include <pops/runtime/multiblock/interface_flux_scheduler.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::runtime::multiblock {

struct PreparedInterfaceFluxSpec {
  std::string interface_identity;
  std::string component_id;
  std::string manifest_identity;
  std::uint32_t interface_version = 1;
  std::string canonical_layout_identity;
  int normal_axis = 0;
  int outward_sign = 1;
  double face_measure = 0.0;
  std::string parameters_json;
  std::string target_json;
  std::shared_ptr<const component::PreparedExecutionContextV1> execution;
};

/// Prepared NumericalFlux adapter captured by InterfaceFluxScheduler.  It consumes both traces in
/// one canonical batch and writes the one shared outward-left flux; neither block owns a callback.
class PreparedInterfaceFluxComponent final {
 public:
  PreparedInterfaceFluxComponent(PreparedInterfaceFluxSpec spec,
                                 std::shared_ptr<component::LoadedComponent> component)
      : spec_(std::move(spec)), component_(std::move(component)) {
    validate_();
    const PopsExecutionContextV1 execution = spec_.execution->view();
    state_ =
        component_->prepared_state(POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, spec_.interface_version,
                                   execution, spec_.parameters_json, spec_.target_json);
  }

  void evaluate(const BoundaryEvaluationPoint& point, const InterfaceFluxBatch& batch) const {
    static_assert(sizeof(Real) == sizeof(double),
                  "NumericalFlux ABI v1 requires the binary64 PoPS backend");
    if (batch.left_state == nullptr || batch.right_state == nullptr ||
        batch.shared_flux == nullptr || batch.face_count < 1 || batch.component_count < 1)
      throw std::invalid_argument("prepared NumericalFlux received an incomplete face batch");
    const auto faces = static_cast<std::size_t>(batch.face_count);
    const auto components = static_cast<std::size_t>(batch.component_count);
    prepare_scratch_(faces);
    const std::string& patch = spec_.interface_identity;
    const PopsConstFieldViewV1 left =
        const_view_(batch.left_state, faces, components, spec_.canonical_layout_identity, patch);
    const PopsConstFieldViewV1 right =
        const_view_(batch.right_state, faces, components, spec_.canonical_layout_identity, patch);
    const PopsConstFieldViewV1 normal_view =
        const_view_(normals_.data(), faces, 2u, spec_.canonical_layout_identity, patch);
    const PopsLogicalTimeV1 time{sizeof(PopsLogicalTimeV1),
                                 point.clock.c_str(),
                                 point.tick,
                                 point.level,
                                 point.substep,
                                 point.stage,
                                 point.stage_fraction.numerator,
                                 point.stage_fraction.denominator,
                                 point.dt,
                                 point.physical_time};
    const PopsNumericalFluxRequestV1 request{sizeof(PopsNumericalFluxRequestV1),
                                             left,
                                             right,
                                             normal_view,
                                             measures_.data(),
                                             time,
                                             spec_.execution->view()};
    std::fill(stability_.begin(), stability_.end(),
              std::numeric_limits<double>::quiet_NaN());
    std::fill(actions_.begin(), actions_.end(), POPS_COMPONENT_CONTINUE_V1);
    PopsNumericalFluxResultV1 result{
        sizeof(PopsNumericalFluxResultV1),
        field_view_(batch.shared_flux, faces, components, spec_.canonical_layout_identity, patch),
        stability_.data(),
        actions_.data(),
        {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr}};
    const auto& api = component_->table<PopsNumericalFluxApiV1>(
        POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, spec_.interface_version);
    const int code = component::evaluate_faces(api, state_, request, result);
    if (code != 0 || result.status.code != 0 || result.status.action != POPS_COMPONENT_CONTINUE_V1)
      throw std::runtime_error(result.status.reason == nullptr ? "native NumericalFlux failed"
                                                               : result.status.reason);
    for (std::size_t face = 0; face < faces; ++face) {
      if (actions_[face] != POPS_COMPONENT_CONTINUE_V1)
        throw std::runtime_error("native NumericalFlux returned a non-continue per-face action");
      if (!std::isfinite(stability_[face]) || stability_[face] < 0.0)
        throw std::runtime_error(
            "native NumericalFlux returned an invalid per-face stability bound");
    }
    // This first production route is governed by an explicit FixedDt Program: the
    // validated bound is diagnostic and cannot silently override that time
    // authority.  A future adaptive controller must consume the same typed result
    // explicitly rather than inferring a timestep inside this spatial adapter.
  }

 private:
  void prepare_scratch_(std::size_t faces) const {
    if (scratch_faces_ == faces)
      return;
    if (scratch_faces_ != 0)
      throw std::invalid_argument(
          "prepared NumericalFlux face count changed after its first authenticated batch");
    normals_.assign(faces * 2u, 0.0);
    for (std::size_t face = 0; face < faces; ++face)
      normals_[face * 2u + static_cast<std::size_t>(spec_.normal_axis)] =
          static_cast<double>(spec_.outward_sign);
    measures_.assign(faces, spec_.face_measure);
    stability_.assign(faces, std::numeric_limits<double>::quiet_NaN());
    actions_.assign(faces, POPS_COMPONENT_CONTINUE_V1);
    scratch_faces_ = faces;
  }

  static PopsConstFieldViewV1 const_view_(const void* data, std::size_t faces,
                                          std::size_t components, const std::string& layout,
                                          const std::string& patch) {
    return {sizeof(PopsConstFieldViewV1),
            data,
            2,
            {faces, 1, 1},
            {static_cast<std::ptrdiff_t>(components), static_cast<std::ptrdiff_t>(components), 0},
            components,
            1,
            POPS_FIELD_CENTERING_CELL_V1,
            0,
            {0, 0, 0},
            {0, 0, 0},
            POPS_SCALAR_FLOAT64_V1,
            POPS_MEMORY_SPACE_HOST_V1,
            layout.c_str(),
            patch.c_str(),
            POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
  }

  static PopsFieldViewV1 field_view_(void* data, std::size_t faces, std::size_t components,
                                     const std::string& layout, const std::string& patch) {
    return {sizeof(PopsFieldViewV1),
            data,
            2,
            {faces, 1, 1},
            {static_cast<std::ptrdiff_t>(components), static_cast<std::ptrdiff_t>(components), 0},
            components,
            1,
            POPS_FIELD_CENTERING_CELL_V1,
            0,
            {0, 0, 0},
            {0, 0, 0},
            POPS_SCALAR_FLOAT64_V1,
            POPS_MEMORY_SPACE_HOST_V1,
            layout.c_str(),
            patch.c_str(),
            POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
  }

  void validate_() const {
    if (!component_ || !spec_.execution || spec_.interface_identity.empty() ||
        spec_.component_id.empty() || spec_.manifest_identity.empty() ||
        spec_.canonical_layout_identity.empty() || spec_.interface_version != 1 ||
        spec_.normal_axis < 0 || spec_.normal_axis >= 2 ||
        (spec_.outward_sign != -1 && spec_.outward_sign != 1) || !(spec_.face_measure > 0.0))
      throw std::invalid_argument("prepared NumericalFlux specification is incomplete");
    component::validate_execution_context(spec_.execution->view());
    const auto& api = component_->api();
    if (api.component_id == nullptr || api.manifest_identity == nullptr ||
        spec_.component_id != api.component_id || spec_.manifest_identity != api.manifest_identity)
      throw std::invalid_argument("prepared NumericalFlux changed native component identity");
    const auto& table = component_->table<PopsNumericalFluxApiV1>(
        POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, spec_.interface_version);
    component::require_operation(table.evaluate_faces != nullptr, "evaluate_faces");
  }

  PreparedInterfaceFluxSpec spec_;
  std::shared_ptr<component::LoadedComponent> component_;
  void* state_ = nullptr;
  // One scheduler host thread owns an installed provider.  Reuse its exact ABI buffers after the
  // first authenticated face batch instead of allocating normals/status arrays at every stage.
  mutable std::size_t scratch_faces_ = 0;
  mutable std::vector<double> normals_;
  mutable std::vector<double> measures_;
  mutable std::vector<double> stability_;
  mutable std::vector<PopsComponentActionV1> actions_;
};

}  // namespace pops::runtime::multiblock
