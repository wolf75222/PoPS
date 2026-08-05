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

template <int Dim>
struct PreparedInterfaceFluxSpec {
  static_assert(Dim >= 1 && Dim <= 3,
                "PreparedInterfaceFluxSpec only supports dimensions 1, 2, and 3");
  std::string interface_identity;
  std::string component_id;
  std::string manifest_identity;
  std::uint32_t interface_version = 1;
  std::string canonical_layout_identity;
  std::string parameters_json;
  std::string target_json;
  std::shared_ptr<const component::PreparedExecutionContextV1> execution;
};

/// Prepared NumericalFlux adapter captured by InterfaceFluxScheduler.  It consumes both traces in
/// one canonical batch and writes the one shared outward-left flux; neither block owns a callback.
template <int Dim>
class PreparedInterfaceFluxComponent final {
  static_assert(Dim >= 1 && Dim <= 3,
                "PreparedInterfaceFluxComponent only supports dimensions 1, 2, and 3");

 public:
  PreparedInterfaceFluxComponent(PreparedInterfaceFluxSpec<Dim> spec,
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
        batch.outward_normals == nullptr || batch.shared_flux == nullptr || batch.face_count < 1 ||
        batch.component_count < 1 || !(batch.face_measure > Real(0)) ||
        batch.memory_space != spec_.execution->view().memory_space)
      throw std::invalid_argument("prepared NumericalFlux received an incomplete face batch");
    const auto faces = static_cast<std::size_t>(batch.face_count);
    const auto components = static_cast<std::size_t>(batch.component_count);
    prepare_scratch_(faces, static_cast<double>(batch.face_measure));
    const std::string& patch = spec_.interface_identity;
    const PopsConstFieldViewV1 left =
        const_view_(batch.left_state, faces, components, batch.memory_space,
                    spec_.canonical_layout_identity, patch);
    const PopsConstFieldViewV1 right =
        const_view_(batch.right_state, faces, components, batch.memory_space,
                    spec_.canonical_layout_identity, patch);
    const PopsConstFieldViewV1 normal_view =
        const_view_(batch.outward_normals, faces, static_cast<std::size_t>(Dim), batch.memory_space,
                    spec_.canonical_layout_identity, patch);
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
    std::fill(stability_.begin(), stability_.end(), std::numeric_limits<double>::quiet_NaN());
    std::fill(actions_.begin(), actions_.end(), POPS_COMPONENT_CONTINUE_V1);
    PopsNumericalFluxResultV1 result{
        sizeof(PopsNumericalFluxResultV1),
        field_view_(batch.shared_flux, faces, components, batch.memory_space,
                    spec_.canonical_layout_identity, patch),
        stability_.data(),
        actions_.data(),
        {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr}};
    const auto& api = component_->table<PopsNumericalFluxApiV1>(
        POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, spec_.interface_version);
    const int code = component::evaluate_faces<Dim>(api, state_, request, result);
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
  void prepare_scratch_(std::size_t faces, double face_measure) const {
    if (scratch_faces_ == faces && scratch_face_measure_ == face_measure)
      return;
    if (scratch_faces_ != 0)
      throw std::invalid_argument(
          "prepared NumericalFlux face shape changed after its first authenticated batch");
    measures_.assign(faces, face_measure);
    stability_.assign(faces, std::numeric_limits<double>::quiet_NaN());
    actions_.assign(faces, POPS_COMPONENT_CONTINUE_V1);
    scratch_faces_ = faces;
    scratch_face_measure_ = face_measure;
  }

  static PopsConstFieldViewV1 const_view_(const void* data, std::size_t faces,
                                          std::size_t components, PopsMemorySpaceV1 memory_space,
                                          const std::string& layout, const std::string& patch) {
    return {sizeof(PopsConstFieldViewV1),
            data,
            static_cast<std::uint32_t>(Dim),
            {faces, 1, 1},
            {static_cast<std::ptrdiff_t>(components),
             Dim >= 2 ? static_cast<std::ptrdiff_t>(components) : 0,
             Dim >= 3 ? static_cast<std::ptrdiff_t>(components) : 0},
            components,
            1,
            POPS_FIELD_CENTERING_CELL_V1,
            0,
            {0, 0, 0},
            {0, 0, 0},
            POPS_SCALAR_FLOAT64_V1,
            memory_space,
            layout.c_str(),
            patch.c_str(),
            POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
  }

  static PopsFieldViewV1 field_view_(void* data, std::size_t faces, std::size_t components,
                                     PopsMemorySpaceV1 memory_space, const std::string& layout,
                                     const std::string& patch) {
    return {sizeof(PopsFieldViewV1),
            data,
            static_cast<std::uint32_t>(Dim),
            {faces, 1, 1},
            {static_cast<std::ptrdiff_t>(components),
             Dim >= 2 ? static_cast<std::ptrdiff_t>(components) : 0,
             Dim >= 3 ? static_cast<std::ptrdiff_t>(components) : 0},
            components,
            1,
            POPS_FIELD_CENTERING_CELL_V1,
            0,
            {0, 0, 0},
            {0, 0, 0},
            POPS_SCALAR_FLOAT64_V1,
            memory_space,
            layout.c_str(),
            patch.c_str(),
            POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
  }

  void validate_() const {
    if (!component_ || !spec_.execution || spec_.interface_identity.empty() ||
        spec_.component_id.empty() || spec_.manifest_identity.empty() ||
        spec_.canonical_layout_identity.empty() || spec_.interface_version != 1)
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

  PreparedInterfaceFluxSpec<Dim> spec_;
  std::shared_ptr<component::LoadedComponent> component_;
  void* state_ = nullptr;
  // One scheduler host thread owns an installed provider.  Reuse its exact ABI buffers after the
  // first authenticated face batch instead of allocating normals/status arrays at every stage.
  mutable std::size_t scratch_faces_ = 0;
  mutable double scratch_face_measure_ = 0.0;
  mutable std::vector<double> measures_;
  mutable std::vector<double> stability_;
  mutable std::vector<PopsComponentActionV1> actions_;
};

}  // namespace pops::runtime::multiblock
