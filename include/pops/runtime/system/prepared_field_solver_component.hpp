#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/sha256.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>
#include <pops/runtime/dynamic/prepared_execution_context.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::runtime::field {

struct PreparedFieldSolverSpec {
  std::string provider_slot;
  std::string topology_component_id;
  std::string topology_manifest_identity;
  std::uint32_t topology_interface_version = 2;
  std::string topology_parameters_json;
  std::string solver_component_id;
  std::string solver_manifest_identity;
  std::uint32_t solver_interface_version = 2;
  std::string solver_parameters_json;
  std::string source_layout_identity;
  std::string topology_recipe_identity;
  std::string boundary_contract_json;
  double relative_tolerance = 0.0;
  double absolute_tolerance = 0.0;
  std::int32_t max_iterations = 0;
  std::shared_ptr<const component::PreparedExecutionContextV1> execution;
};

struct FieldTopologyReportRow {
  std::string patch_identity;
  std::string topology_digest;
  std::string provenance;
  std::size_t material_points = 0;
  std::size_t connected_components = 0;
  std::string source_layout_identity;
  std::string materialized_layout_identity;
};

/// Installed adapter for one indivisible generated FieldTopology+FieldSolver provider.
///
/// The two component instances are prepared once at installation.  The global topology is then
/// materialized once from replicated patch metadata and reused for every solve.  A solve sends every
/// local patch view in one request and calls the component exactly once on every participating rank,
/// including ranks with zero local patches.  The currently proven System route is host-resident,
/// serial, Cartesian, cell-centered and full-material; unsupported execution/layout facts are
/// rejected before either component can mutate the solution.
class PreparedFieldSolverComponent final {
 public:
  PreparedFieldSolverComponent(
      PreparedFieldSolverSpec spec,
      std::shared_ptr<component::LoadedComponent> topology,
      std::shared_ptr<component::LoadedComponent> solver)
      : spec_(std::move(spec)), topology_component_(std::move(topology)),
        solver_component_(std::move(solver)) {
    validate_();
    const PopsExecutionContextV1 execution = spec_.execution->view();
    topology_state_ = topology_component_->prepared_state(
        POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, spec_.topology_interface_version,
        execution, spec_.topology_parameters_json);
    solver_state_ = solver_component_->prepared_state(
        POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, spec_.solver_interface_version,
        execution, spec_.solver_parameters_json);
  }

  SolveReport solve(
      MultiFab& rhs, MultiFab& solution, const Geometry& geometry,
      const Periodicity& periodicity) {
    static_assert(sizeof(Real) == sizeof(double),
                  "FieldSolver ABI v2 requires the binary64 PoPS backend");
    validate_solve_layout_(rhs, solution, geometry);
    rhs.sync_host();
    solution.sync_host();
    prepare_topology_once_(rhs, geometry, periodicity);

    prepare_solver_request_once_(rhs, solution);
    PopsSolveReportV2 native{};
    native.struct_size = sizeof(PopsSolveReportV2);
    const auto& api = solver_component_->table<PopsFieldSolverApiV2>(
        POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, spec_.solver_interface_version);
    (void)component::solve_field(api, solver_state_, *solver_request_, native);

    SolveReport report;
    report.iters = native.iterations;
    report.rel_residual = static_cast<Real>(native.relative_residual);
    report.reference_residual_norm =
        static_cast<Real>(native.reference_residual_norm);
    report.residual_norm = static_cast<Real>(native.residual_norm);
    const SolveStatus status = solve_status_(native.status);
    const SolveAction action = solve_action_(native.action);
    if (status == SolveStatus::kSolved) {
      // The component writes directly into the host-resident warm-start buffer.  Do not publish
      // that provisional iterate to the device until every active valid cell has been checked.
      // Inactive material cells and ghosts are outside the provider's solved-value contract.
      if (!active_solution_is_finite_(solution)) {
        report.mark_failed(
            SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
            "native FieldSolver v2 marked a non-finite active solution as solved");
        return report;
      }
      solution.sync_device();
      report.mark_solved(native.reason);
      return report;
    }
    report.mark_failed(status, action, native.reason);
    return report;
  }

  [[nodiscard]] std::vector<FieldTopologyReportRow> topology_report() const {
    if (!topology_) return {};
    std::vector<FieldTopologyReportRow> result;
    result.reserve(topology_->local_patches().size());
    for (const auto& local : topology_->local_patches()) {
      const auto& metadata = topology_->global_patches().at(local.metadata_index);
      std::vector<std::int32_t> components;
      components.reserve(local.component_labels.size());
      for (const auto label : local.component_labels)
        if (label > 0 && std::find(components.begin(), components.end(), label) == components.end())
          components.push_back(label);
      result.push_back({
          metadata.patch_identity, topology_->topology_digest(), topology_->provenance(),
          static_cast<std::size_t>(std::count(
              local.material_mask.begin(), local.material_mask.end(), std::uint8_t{1})),
          components.size(),
          spec_.source_layout_identity,
          materialized_layout_identity_,
      });
    }
    return result;
  }

 private:
  static SolveStatus solve_status_(PopsSolveStatusV2 status) {
    switch (status) {
      case POPS_SOLVE_SOLVED_V2:
        return SolveStatus::kSolved;
      case POPS_SOLVE_SINGULAR_V2:
        return SolveStatus::kSingular;
      case POPS_SOLVE_BREAKDOWN_V2:
        return SolveStatus::kBreakdown;
      case POPS_SOLVE_ITERATION_LIMIT_V2:
        return SolveStatus::kIterationLimit;
      case POPS_SOLVE_INVALID_EVALUATION_V2:
        return SolveStatus::kInvalidEvaluation;
      case POPS_SOLVE_CAPABILITY_FAILURE_V2:
        return SolveStatus::kCapabilityFailure;
      case POPS_SOLVE_INVALID_INPUT_V2:
        return SolveStatus::kInvalidInput;
      case POPS_SOLVE_INCOMPATIBLE_RHS_V2:
        return SolveStatus::kIncompatibleRhs;
    }
    throw std::invalid_argument("FieldSolver v2 returned an unknown solve status");
  }

  static SolveAction solve_action_(PopsSolveActionV2 action) {
    switch (action) {
      case POPS_SOLVE_ACTION_NONE_V2:
        return SolveAction::kNone;
      case POPS_SOLVE_ACTION_FAIL_RUN_V2:
        return SolveAction::kFailRun;
      case POPS_SOLVE_ACTION_REJECT_ATTEMPT_V2:
        return SolveAction::kRejectAttempt;
    }
    throw std::invalid_argument("FieldSolver v2 returned an unknown solve action");
  }

  [[nodiscard]] bool active_solution_is_finite_(const MultiFab& solution) const {
    if (!topology_ ||
        topology_->local_patches().size() !=
            static_cast<std::size_t>(solution.local_size()))
      return false;
    const auto& metadata = topology_->global_patches();
    for (int local = 0; local < solution.local_size(); ++local) {
      const auto& patch = topology_->local_patches()[static_cast<std::size_t>(local)];
      const auto index = static_cast<std::size_t>(solution.global_index(local));
      const Box2D& valid = solution.box(local);
      if (patch.metadata_index != index || index >= metadata.size()) return false;
      const auto& global = metadata[index];
      if (global.dimension != 2 || global.lower[0] != valid.lo[0] ||
          global.lower[1] != valid.lo[1] || global.upper[0] != valid.hi[0] ||
          global.upper[1] != valid.hi[1] ||
          patch.material_mask.size() !=
              static_cast<std::size_t>(valid.num_cells()))
        return false;
      const ConstArray4 values = solution.fab(local).const_array();
      std::size_t point = 0;
      for (int j = valid.lo[1]; j <= valid.hi[1]; ++j) {
        for (int i = valid.lo[0]; i <= valid.hi[0]; ++i, ++point) {
          const std::uint8_t active = patch.material_mask[point];
          if (active > 1 || (active == 1 && !std::isfinite(values(i, j, 0))))
            return false;
        }
      }
    }
    return true;
  }

  static const Real* valid_data_(const Fab2D& fab, const Box2D& valid) {
    const ConstArray4 view = fab.const_array();
    return view.p + static_cast<std::ptrdiff_t>(valid.lo[1] - view.jg0) * view.nx_tot +
           (valid.lo[0] - view.ig0);
  }

  static Real* valid_data_(Fab2D& fab, const Box2D& valid) {
    const Array4 view = fab.array();
    return view.p + static_cast<std::ptrdiff_t>(valid.lo[1] - view.jg0) * view.nx_tot +
           (valid.lo[0] - view.ig0);
  }

  static PopsConstFieldViewV1 const_view_(
      const Fab2D& fab, const Box2D& valid, const char* layout, const char* patch) {
    const ConstArray4 storage = fab.const_array();
    return {
        sizeof(PopsConstFieldViewV1), valid_data_(fab, valid), 2,
        {static_cast<std::size_t>(valid.nx()), static_cast<std::size_t>(valid.ny()), 1},
        {1, storage.nx_tot, 0}, 1, storage.comp_stride,
        POPS_FIELD_CENTERING_CELL_V1, 0, {0, 0, 0}, {0, 0, 0},
        POPS_SCALAR_FLOAT64_V1, POPS_MEMORY_SPACE_HOST_V1, layout, patch,
        POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
  }

  static PopsFieldViewV1 field_view_(
      Fab2D& fab, const Box2D& valid, const char* layout, const char* patch) {
    const Array4 storage = fab.array();
    return {
        sizeof(PopsFieldViewV1), valid_data_(fab, valid), 2,
        {static_cast<std::size_t>(valid.nx()), static_cast<std::size_t>(valid.ny()), 1},
        {1, storage.nx_tot, 0}, 1, storage.comp_stride,
        POPS_FIELD_CENTERING_CELL_V1, 0, {0, 0, 0}, {0, 0, 0},
        POPS_SCALAR_FLOAT64_V1, POPS_MEMORY_SPACE_HOST_V1, layout, patch,
        POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
  }

  template <class View>
  static bool same_field_view_(const View& left, const View& right) {
    if (left.struct_size != right.struct_size || left.data != right.data ||
        left.dimension != right.dimension ||
        left.component_count != right.component_count ||
        left.component_stride != right.component_stride ||
        left.centering != right.centering ||
        left.centering_axes != right.centering_axes ||
        left.scalar_type != right.scalar_type ||
        left.memory_space != right.memory_space ||
        left.layout_identity != right.layout_identity ||
        left.patch_identity != right.patch_identity ||
        left.ownership != right.ownership)
      return false;
    for (std::size_t axis = 0; axis < 3; ++axis)
      if (left.extents[axis] != right.extents[axis] ||
          left.axis_strides[axis] != right.axis_strides[axis] ||
          left.ghost_lower[axis] != right.ghost_lower[axis] ||
          left.ghost_upper[axis] != right.ghost_upper[axis])
        return false;
    return true;
  }

  void prepare_solver_request_once_(MultiFab& rhs, MultiFab& solution) {
    if (!topology_)
      throw std::logic_error("field solver request requires a prepared topology");
    const auto& global = topology_->global_patches();
    if (!solver_request_) {
      std::vector<component::FieldSolverPatchBindingV2> patches;
      patches.reserve(static_cast<std::size_t>(rhs.local_size()));
      for (int local = 0; local < rhs.local_size(); ++local) {
        const auto index = static_cast<std::size_t>(rhs.global_index(local));
        const auto& patch = global.at(index);
        patches.push_back({
            index,
            const_view_(rhs.fab(local), rhs.box(local), patch.layout_identity,
                        patch.patch_identity),
            field_view_(solution.fab(local), solution.box(local),
                        patch.layout_identity, patch.patch_identity),
            {}});
      }
      solver_request_.emplace(component::bind_field_solver_request(
          *topology_, patches, spec_.execution->view(),
          spec_.boundary_contract_json.c_str(), spec_.relative_tolerance,
          spec_.absolute_tolerance, spec_.max_iterations));
      return;
    }

    const auto& cached = solver_request_->request();
    if (cached.local_patch_count != static_cast<std::size_t>(rhs.local_size()) ||
        cached.local_patch_count != static_cast<std::size_t>(solution.local_size()) ||
        (cached.local_patch_count != 0 && cached.local_patches == nullptr))
      throw std::runtime_error(
          "prepared FieldSolver request cannot be reused after local patch storage changed");
    for (int local = 0; local < rhs.local_size(); ++local) {
      const auto index = static_cast<std::size_t>(rhs.global_index(local));
      const auto& metadata = global.at(index);
      const auto expected_rhs = const_view_(
          rhs.fab(local), rhs.box(local), metadata.layout_identity,
          metadata.patch_identity);
      const auto expected_solution = field_view_(
          solution.fab(local), solution.box(local), metadata.layout_identity,
          metadata.patch_identity);
      const auto& patch = cached.local_patches[static_cast<std::size_t>(local)];
      if (patch.struct_size < sizeof(PopsFieldSolverPatchV2) ||
          patch.metadata_index != index ||
          !same_field_view_(patch.rhs, expected_rhs) ||
          !same_field_view_(patch.solution, expected_solution) ||
          !component::empty_field_view(patch.coefficients))
        throw std::runtime_error(
            "prepared FieldSolver request cannot be reused after field storage changed");
    }
  }

  static bool same_box_(const Box2D& left, const Box2D& right) {
    return left.lo[0] == right.lo[0] && left.lo[1] == right.lo[1] &&
           left.hi[0] == right.hi[0] && left.hi[1] == right.hi[1];
  }

  static bool same_layout_(const MultiFab& left, const MultiFab& right) {
    if (left.box_array().size() != right.box_array().size() ||
        left.dmap().ranks() != right.dmap().ranks())
      return false;
    for (int index = 0; index < left.box_array().size(); ++index)
      if (!same_box_(left.box_array()[index], right.box_array()[index])) return false;
    return true;
  }

  static std::string hashed_identity_(const char* domain, const std::string& payload) {
    const std::vector<std::uint8_t> bytes(payload.begin(), payload.end());
    return std::string("pops.") + domain + ".v1:sha256:" +
           identity::sha256_hex(bytes);
  }

  std::string runtime_layout_identity_(
      const MultiFab& field, const Geometry& geometry,
      const Periodicity& periodicity) const {
    std::ostringstream payload;
    payload.imbue(std::locale::classic());
    payload << std::setprecision(std::numeric_limits<double>::max_digits10)
            << spec_.source_layout_identity << '|' << spec_.topology_recipe_identity << '|'
            << (periodicity.x ? 1 : 0) << ':' << (periodicity.y ? 1 : 0) << '|'
            << geometry.xlo << '|'
            << geometry.xhi << '|' << geometry.ylo << '|' << geometry.yhi << '|';
    for (int index = 0; index < field.box_array().size(); ++index) {
      const auto& box = field.box_array()[index];
      payload << index << ':' << field.dmap()[index] << ':' << box.lo[0] << ':'
              << box.lo[1] << ':' << box.hi[0] << ':' << box.hi[1] << ';';
    }
    return hashed_identity_("runtime-field-layout", payload.str());
  }

  static std::string runtime_patch_identity_(
      const std::string& layout, std::size_t index, const Box2D& box) {
    const std::string payload = layout + '|' + std::to_string(index) + '|' +
        std::to_string(box.lo[0]) + '|' + std::to_string(box.lo[1]) + '|' +
        std::to_string(box.hi[0]) + '|' + std::to_string(box.hi[1]);
    return hashed_identity_("runtime-field-patch", payload);
  }

  void validate_topology_reuse_(
      const MultiFab& field, const Geometry& geometry,
      const Periodicity& periodicity) const {
    const auto& global = topology_->global_topology();
    if (global.dimension != 2 ||
        global.topology_recipe_identity == nullptr ||
        global.source_layout_identity == nullptr ||
        global.materialized_layout_identity == nullptr ||
        spec_.topology_recipe_identity != global.topology_recipe_identity ||
        spec_.source_layout_identity != global.source_layout_identity ||
        materialized_layout_identity_ != global.materialized_layout_identity ||
        global.domain_lower[0] != geometry.domain.lo[0] ||
        global.domain_lower[1] != geometry.domain.lo[1] ||
        global.domain_upper[0] != geometry.domain.hi[0] ||
        global.domain_upper[1] != geometry.domain.hi[1] ||
        global.periodic_axes !=
            static_cast<std::uint32_t>((periodicity.x ? 1u : 0u) |
                                       (periodicity.y ? 2u : 0u)) ||
        global.patch_count != static_cast<std::size_t>(field.box_array().size()) ||
        patch_identities_.size() != global.patch_count)
      throw std::runtime_error(
          "prepared external field topology cannot be reused after a layout change");
    for (std::size_t index = 0; index < global.patch_count; ++index) {
      const auto& box = field.box_array()[static_cast<int>(index)];
      const auto& patch = global.patches[index];
      if (patch.global_patch_index != index ||
          patch.owner_rank != field.dmap()[static_cast<int>(index)] ||
          patch.dimension != 2 || patch.lower[0] != box.lo[0] ||
          patch.lower[1] != box.lo[1] || patch.upper[0] != box.hi[0] ||
          patch.upper[1] != box.hi[1] ||
          patch.physical_lower[0] !=
              geometry.xlo + static_cast<double>(box.lo[0]) * geometry.dx() ||
          patch.physical_lower[1] !=
              geometry.ylo + static_cast<double>(box.lo[1]) * geometry.dy() ||
          patch.cell_spacing[0] != geometry.dx() ||
          patch.cell_spacing[1] != geometry.dy() ||
          patch.layout_identity == nullptr || patch.patch_identity == nullptr ||
          materialized_layout_identity_ != patch.layout_identity ||
          patch_identities_[index] != patch.patch_identity)
        throw std::runtime_error(
            "prepared external field topology cannot be reused after a layout change");
    }
  }

  void prepare_topology_once_(
      const MultiFab& field, const Geometry& geometry,
      const Periodicity& periodicity) {
    if (topology_) {
      validate_topology_reuse_(field, geometry, periodicity);
      return;
    }
    const std::string layout = runtime_layout_identity_(field, geometry, periodicity);
    materialized_layout_identity_ = layout;
    patch_identities_.clear();
    patch_identities_.reserve(static_cast<std::size_t>(field.box_array().size()));
    for (int index = 0; index < field.box_array().size(); ++index)
      patch_identities_.push_back(runtime_patch_identity_(
          materialized_layout_identity_, static_cast<std::size_t>(index),
          field.box_array()[index]));

    std::vector<PopsFieldPatchMetadataV1> global;
    global.reserve(static_cast<std::size_t>(field.box_array().size()));
    for (int index = 0; index < field.box_array().size(); ++index) {
      const auto& box = field.box_array()[index];
      PopsFieldPatchMetadataV1 row{
          sizeof(PopsFieldPatchMetadataV1), static_cast<std::size_t>(index),
          field.dmap()[index], 0, 2, {}, {}, {}, {},
          POPS_FIELD_CENTERING_CELL_V1, 0, spec_.source_layout_identity.c_str(),
          patch_identities_[static_cast<std::size_t>(index)].c_str()};
      row.lower[0] = box.lo[0];
      row.lower[1] = box.lo[1];
      row.upper[0] = box.hi[0];
      row.upper[1] = box.hi[1];
      row.physical_lower[0] = geometry.xlo + static_cast<double>(box.lo[0]) * geometry.dx();
      row.physical_lower[1] = geometry.ylo + static_cast<double>(box.lo[1]) * geometry.dy();
      row.cell_spacing[0] = geometry.dx();
      row.cell_spacing[1] = geometry.dy();
      global.push_back(row);
    }
    PopsFieldGlobalTopologyV1 global_topology{
        sizeof(PopsFieldGlobalTopologyV1), spec_.topology_recipe_identity.c_str(),
        spec_.source_layout_identity.c_str(), materialized_layout_identity_.c_str(), 2,
        {}, {}, static_cast<std::uint32_t>((periodicity.x ? 1u : 0u) |
                                           (periodicity.y ? 2u : 0u)),
        global.size(), global.data()};
    global_topology.domain_lower[0] = geometry.domain.lo[0];
    global_topology.domain_lower[1] = geometry.domain.lo[1];
    global_topology.domain_upper[0] = geometry.domain.hi[0];
    global_topology.domain_upper[1] = geometry.domain.hi[1];
    std::vector<component::FieldTopologyPatchInputV2> local;
    local.reserve(static_cast<std::size_t>(field.local_size()));
    for (int index = 0; index < field.local_size(); ++index)
      local.push_back({static_cast<std::size_t>(field.global_index(index)),
                       POPS_FIELD_MATERIAL_FULL_V1, {}, {}, {}});
    const auto& api = topology_component_->table<PopsFieldTopologyApiV2>(
        POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, spec_.topology_interface_version);
    topology_.emplace(component::prepare_field_topology(
        api, topology_state_, global_topology, local, spec_.execution->view()));
  }

  void validate_solve_layout_(
      const MultiFab& rhs, const MultiFab& solution, const Geometry& geometry) const {
    if (rhs.ncomp() != 1 || solution.ncomp() != 1 ||
        rhs.local_size() != solution.local_size() || !same_layout_(rhs, solution))
      throw std::invalid_argument(
          "prepared FieldSolver requires matching scalar RHS/solution global layouts");
    if (!same_box_(geometry.domain, rhs.box_array().bounding_box()))
      throw std::invalid_argument(
          "prepared FieldSolver geometry domain differs from its global patch metadata");
    std::int64_t covered_points = 0;
    for (int left = 0; left < rhs.box_array().size(); ++left) {
      const auto& box = rhs.box_array()[left];
      if (box.lo[0] < geometry.domain.lo[0] || box.lo[1] < geometry.domain.lo[1] ||
          box.hi[0] > geometry.domain.hi[0] || box.hi[1] > geometry.domain.hi[1])
        throw std::invalid_argument(
            "prepared full-material FieldSolver patch lies outside its domain");
      covered_points += static_cast<std::int64_t>(box.num_cells());
      for (int right = left + 1; right < rhs.box_array().size(); ++right) {
        const auto& other = rhs.box_array()[right];
        const bool disjoint = box.hi[0] < other.lo[0] || other.hi[0] < box.lo[0] ||
                              box.hi[1] < other.lo[1] || other.hi[1] < box.lo[1];
        if (!disjoint)
          throw std::invalid_argument(
              "prepared full-material FieldSolver patches overlap");
      }
    }
    if (covered_points != static_cast<std::int64_t>(geometry.domain.num_cells()))
      throw std::invalid_argument(
          "prepared full-material FieldSolver patches do not exactly cover the domain");
    for (int local = 0; local < rhs.local_size(); ++local)
      if (rhs.global_index(local) != solution.global_index(local) ||
          !same_box_(rhs.box(local), solution.box(local)))
        throw std::invalid_argument(
            "prepared FieldSolver local RHS/solution patch identities differ");
  }

  void validate_() const {
    if (!topology_component_ || !solver_component_ || !spec_.execution ||
        spec_.provider_slot.empty() || spec_.topology_component_id.empty() ||
        spec_.topology_manifest_identity.empty() || spec_.solver_component_id.empty() ||
        spec_.topology_parameters_json.empty() ||
        spec_.solver_manifest_identity.empty() || spec_.solver_parameters_json.empty() ||
        spec_.source_layout_identity.empty() || spec_.topology_recipe_identity.empty() ||
        spec_.boundary_contract_json.find("\"identity\"") == std::string::npos ||
        spec_.topology_interface_version != 2 || spec_.solver_interface_version != 2 ||
        !std::isfinite(spec_.relative_tolerance) || spec_.relative_tolerance < 0.0 ||
        !std::isfinite(spec_.absolute_tolerance) || spec_.absolute_tolerance < 0.0 ||
        spec_.max_iterations < 1)
      throw std::invalid_argument("prepared external field solver specification is incomplete");
    const auto execution = spec_.execution->view();
    component::validate_execution_context(execution);
    if (execution.memory_space != POPS_MEMORY_SPACE_HOST_V1 ||
        std::string(execution.communicator_identity) != "serial")
      throw std::invalid_argument(
          "external FieldSolver v2 System adapter currently proves host/serial execution only");
    const auto& topology_api = topology_component_->api();
    const auto& solver_api = solver_component_->api();
    if (topology_api.component_id == nullptr || topology_api.manifest_identity == nullptr ||
        solver_api.component_id == nullptr || solver_api.manifest_identity == nullptr ||
        spec_.topology_component_id != topology_api.component_id ||
        spec_.topology_manifest_identity != topology_api.manifest_identity ||
        spec_.solver_component_id != solver_api.component_id ||
        spec_.solver_manifest_identity != solver_api.manifest_identity)
      throw std::invalid_argument("prepared external field solver changed component identity");
    const auto& topology = topology_component_->table<PopsFieldTopologyApiV2>(
        POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, spec_.topology_interface_version);
    const auto& solver = solver_component_->table<PopsFieldSolverApiV2>(
        POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, spec_.solver_interface_version);
    component::require_operation(topology.prepare_topology != nullptr, "prepare_topology");
    component::require_operation(solver.solve != nullptr, "solve");
  }

  PreparedFieldSolverSpec spec_;
  std::shared_ptr<component::LoadedComponent> topology_component_;
  std::shared_ptr<component::LoadedComponent> solver_component_;
  void* topology_state_ = nullptr;
  void* solver_state_ = nullptr;
  std::optional<component::PreparedFieldTopologyV2> topology_;
  std::optional<component::TopologyBoundFieldSolverRequestV2> solver_request_;
  std::string materialized_layout_identity_;
  std::vector<std::string> patch_identities_;
};

}  // namespace pops::runtime::field
