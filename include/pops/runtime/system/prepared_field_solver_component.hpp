#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/core/identity/sha256.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>
#include <pops/runtime/dynamic/prepared_execution_context.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
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
  bool component_pair_declares_mpi = false;
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
/// including ranks with zero local patches. The specialization carries one exact rank through
/// geometry, patch metadata, borrowed views and topology identities. The proven route is
/// host-resident, Cartesian, cell-centered and full-material; serial and explicitly declared
/// MPI_COMM_WORLD component pairs use the same ranked algorithm. Unsupported execution/layout
/// facts are rejected before either component can mutate the solution.
template <int Dim>
class PreparedFieldSolverComponent final {
 public:
  static_assert(Dim >= 1 && Dim <= 3,
                "PreparedFieldSolverComponent supports dimensions 1, 2, and 3");
  static constexpr int dimension = Dim;
  using field_type = MultiFab<Dim>;
  using fab_type = Fab<Dim>;
  using box_type = Box<Dim>;
  using geometry_type = Geometry<Dim>;
  using periodicity_type = std::array<bool, Dim>;

  PreparedFieldSolverComponent(PreparedFieldSolverSpec spec,
                               std::shared_ptr<component::LoadedComponent> topology,
                               std::shared_ptr<component::LoadedComponent> solver)
      : spec_(std::move(spec)),
        topology_component_(std::move(topology)),
        solver_component_(std::move(solver)) {
    validate_();
    prepare_provider_contract_();
    const PopsExecutionContextV1 execution = spec_.execution->view();
    topology_state_ = topology_component_->prepared_state(
        POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, spec_.topology_interface_version, execution,
        spec_.topology_parameters_json);
    solver_state_ = solver_component_->prepared_state(POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2,
                                                      spec_.solver_interface_version, execution,
                                                      spec_.solver_parameters_json);
  }

  [[nodiscard]] std::string_view provider_identity() const noexcept { return provider_identity_; }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return collective_contract_;
  }
  [[nodiscard]] int maximum_iterations() const noexcept { return spec_.max_iterations; }

  SolveReport solve(field_type& rhs, field_type& solution, const geometry_type& geometry,
                    const periodicity_type& periodicity) {
    static_assert(sizeof(Real) == sizeof(double),
                  "FieldSolver ABI v2 requires the binary64 PoPS backend");
    collective_preflight_([&] { validate_solve_layout_(rhs, solution, geometry); },
                          "external FieldSolver layout validation failed collectively");
    ::pops::device_fence();
    prepare_topology_once_(rhs, geometry, periodicity);

    std::exception_ptr request_error;
    try {
      prepare_solver_request_once_(rhs, solution);
    } catch (...) {
      request_error = std::current_exception();
    }
    if (all_reduce_max(request_error ? 1L : 0L) != 0) {
      solver_request_.reset();
      if (n_ranks() == 1 && request_error)
        std::rethrow_exception(request_error);
      throw std::runtime_error("external FieldSolver request binding failed collectively");
    }
    PopsSolveReportV2 native{};
    std::exception_ptr solve_error;
    try {
      native.struct_size = sizeof(PopsSolveReportV2);
      const auto& api = solver_component_->table<PopsFieldSolverApiV2>(
          POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, spec_.solver_interface_version);
      (void)component::solve_field(api, solver_state_, *solver_request_, native);
    } catch (...) {
      solve_error = std::current_exception();
    }
    if (all_reduce_max(solve_error ? 1L : 0L) != 0) {
      if (n_ranks() == 1 && solve_error)
        std::rethrow_exception(solve_error);
      throw std::runtime_error("external FieldSolver execution failed collectively");
    }

    ExactContractBuilder report_contract;
    report_contract.text("pops.runtime.external-field-solver-report")
        .scalar(std::uint32_t{1})
        .scalar(native.status)
        .scalar(native.action)
        .scalar(native.iterations)
        .scalar(native.relative_residual)
        .scalar(native.reference_residual_norm)
        .scalar(native.residual_norm)
        .text(native.reason);
    const std::string exact_report = std::move(report_contract).release();
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"external-field-solver-report", std::string_view(exact_report)}}))
      throw std::runtime_error("external FieldSolver returned rank-divergent solve reports");

    SolveReport report;
    report.iters = native.iterations;
    report.rel_residual = static_cast<Real>(native.relative_residual);
    report.reference_residual_norm = static_cast<Real>(native.reference_residual_norm);
    report.residual_norm = static_cast<Real>(native.residual_norm);
    const SolveStatus status = solve_status_(native.status);
    const SolveAction action = solve_action_(native.action);
    if (status == SolveStatus::kSolved) {
      // The component writes directly into the host-resident warm-start buffer.  Do not publish
      // that provisional iterate to the device until every active valid cell has been checked.
      // Inactive material cells and ghosts are outside the provider's solved-value contract.
      if (all_reduce_max(active_solution_is_finite_(solution) ? 0L : 1L) != 0) {
        report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                           "native FieldSolver v2 marked a non-finite active solution as solved");
        return report;
      }
      ::pops::device_fence();
      report.mark_solved(native.reason);
      return report;
    }
    report.mark_failed(status, action, native.reason);
    return report;
  }

  [[nodiscard]] std::vector<FieldTopologyReportRow> topology_report() const {
    if (!topology_)
      return {};
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
          metadata.patch_identity,
          topology_->topology_digest(),
          topology_->provenance(),
          static_cast<std::size_t>(
              std::count(local.material_mask.begin(), local.material_mask.end(), std::uint8_t{1})),
          components.size(),
          spec_.source_layout_identity,
          materialized_layout_identity_,
      });
    }
    return result;
  }

 private:
  template <class Function>
  static void collective_preflight_(Function&& function, const char* collective_message) {
    std::exception_ptr local_error;
    try {
      function();
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L) == 0)
      return;
    if (n_ranks() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error(collective_message);
  }

  void prepare_provider_contract_() {
    ExactContractBuilder contract;
    contract.text("pops.runtime.external-field-solver-provider")
        .scalar(std::uint32_t{2})
        .scalar(static_cast<std::uint32_t>(Dim))
        .text(spec_.provider_slot)
        .text(spec_.topology_component_id)
        .text(spec_.topology_manifest_identity)
        .scalar(spec_.topology_interface_version)
        .text(spec_.topology_parameters_json)
        .text(spec_.solver_component_id)
        .text(spec_.solver_manifest_identity)
        .scalar(spec_.solver_interface_version)
        .text(spec_.solver_parameters_json)
        .text(spec_.source_layout_identity)
        .text(spec_.topology_recipe_identity)
        .text(spec_.boundary_contract_json)
        .scalar(spec_.relative_tolerance)
        .scalar(spec_.absolute_tolerance)
        .scalar(spec_.max_iterations)
        .scalar(spec_.component_pair_declares_mpi)
        .text(spec_.execution->identity());
    collective_contract_ = std::move(contract).release();
    provider_identity_ = hashed_identity_("external-field-solver-provider", collective_contract_);
  }

  static SolveStatus solve_status_(std::int32_t status) {
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

  static SolveAction solve_action_(std::int32_t action) {
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

  template <class Function>
  static void for_each_index_(const box_type& box, Function&& function) {
    const std::int64_t signed_points = box.numPts();
    if (signed_points < 0)
      throw std::overflow_error("field topology patch point count is negative");
    const auto points = static_cast<std::size_t>(signed_points);
    for (std::size_t linear = 0; linear < points; ++linear) {
      std::size_t remainder = linear;
      Index<Dim> index{};
      for (int axis = 0; axis < Dim; ++axis) {
        const auto extent = static_cast<std::size_t>(box.length(axis));
        index[axis] = box.lo[axis] + static_cast<int>(remainder % extent);
        remainder /= extent;
      }
      function(index, linear);
    }
  }

  [[nodiscard]] bool active_solution_is_finite_(const field_type& solution) const {
    if (!topology_ || topology_->local_patches().size() != solution.local_size())
      return false;
    const auto& metadata = topology_->global_patches();
    for (std::size_t local = 0; local < solution.local_size(); ++local) {
      const auto& patch = topology_->local_patches()[local];
      const auto index = solution.global_index(local);
      const box_type& valid = solution.box(local);
      if (patch.metadata_index != index || index >= metadata.size())
        return false;
      const auto& global = metadata[index];
      bool metadata_matches = global.dimension == static_cast<std::uint32_t>(Dim);
      for (int axis = 0; axis < Dim; ++axis)
        metadata_matches = metadata_matches && global.lower[axis] == valid.lo[axis] &&
                           global.upper[axis] == valid.hi[axis];
      for (int axis = Dim; axis < 3; ++axis)
        metadata_matches = metadata_matches && global.lower[axis] == 0 && global.upper[axis] == 0;
      if (!metadata_matches ||
          patch.material_mask.size() != static_cast<std::size_t>(valid.numPts()))
        return false;
      const FieldView<const Real, Dim> values = solution.fab(local).view();
      bool finite = true;
      for_each_index_(valid, [&](const Index<Dim>& cell, std::size_t point) {
        const std::uint8_t active = patch.material_mask[point];
        finite = finite && active <= 1 && (active == 0 || std::isfinite(values(cell, 0)));
      });
      if (!finite)
        return false;
    }
    return true;
  }

  static const Real* valid_data_(const fab_type& fab, const box_type& valid) {
    const FieldView<const Real, Dim> view = fab.view();
    std::int64_t offset = 0;
    for (int axis = 0; axis < Dim; ++axis)
      offset += static_cast<std::int64_t>(valid.lo[axis] - view.origin[axis]) * view.strides[axis];
    return view.data + offset;
  }

  static Real* valid_data_(fab_type& fab, const box_type& valid) {
    const FieldView<Real, Dim> view = fab.view();
    std::int64_t offset = 0;
    for (int axis = 0; axis < Dim; ++axis)
      offset += static_cast<std::int64_t>(valid.lo[axis] - view.origin[axis]) * view.strides[axis];
    return view.data + offset;
  }

  static PopsConstFieldViewV1 const_view_(const fab_type& fab, const box_type& valid,
                                          const char* layout, const char* patch) {
    const FieldView<const Real, Dim> storage = fab.view();
    PopsConstFieldViewV1 result{};
    result.struct_size = sizeof(PopsConstFieldViewV1);
    result.data = valid_data_(fab, valid);
    result.dimension = Dim;
    for (int axis = 0; axis < 3; ++axis) {
      result.extents[axis] = 1;
      result.axis_strides[axis] = 0;
    }
    for (int axis = 0; axis < Dim; ++axis) {
      result.extents[axis] = static_cast<std::size_t>(valid.length(axis));
      result.axis_strides[axis] = storage.strides[axis];
    }
    result.component_count = 1;
    result.component_stride = storage.component_stride;
    result.centering = POPS_FIELD_CENTERING_CELL_V1;
    result.scalar_type = POPS_SCALAR_FLOAT64_V1;
    result.memory_space = POPS_MEMORY_SPACE_HOST_V1;
    result.layout_identity = layout;
    result.patch_identity = patch;
    result.ownership = POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1;
    return result;
  }

  static PopsFieldViewV1 field_view_(fab_type& fab, const box_type& valid, const char* layout,
                                     const char* patch) {
    const FieldView<Real, Dim> storage = fab.view();
    PopsFieldViewV1 result{};
    result.struct_size = sizeof(PopsFieldViewV1);
    result.data = valid_data_(fab, valid);
    result.dimension = Dim;
    for (int axis = 0; axis < 3; ++axis) {
      result.extents[axis] = 1;
      result.axis_strides[axis] = 0;
    }
    for (int axis = 0; axis < Dim; ++axis) {
      result.extents[axis] = static_cast<std::size_t>(valid.length(axis));
      result.axis_strides[axis] = storage.strides[axis];
    }
    result.component_count = 1;
    result.component_stride = storage.component_stride;
    result.centering = POPS_FIELD_CENTERING_CELL_V1;
    result.scalar_type = POPS_SCALAR_FLOAT64_V1;
    result.memory_space = POPS_MEMORY_SPACE_HOST_V1;
    result.layout_identity = layout;
    result.patch_identity = patch;
    result.ownership = POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1;
    return result;
  }

  template <class View>
  static bool same_field_view_(const View& left, const View& right) {
    if (left.struct_size != right.struct_size || left.data != right.data ||
        left.dimension != right.dimension || left.component_count != right.component_count ||
        left.component_stride != right.component_stride || left.centering != right.centering ||
        left.centering_axes != right.centering_axes || left.scalar_type != right.scalar_type ||
        left.memory_space != right.memory_space || left.layout_identity != right.layout_identity ||
        left.patch_identity != right.patch_identity || left.ownership != right.ownership)
      return false;
    for (std::size_t axis = 0; axis < 3; ++axis)
      if (left.extents[axis] != right.extents[axis] ||
          left.axis_strides[axis] != right.axis_strides[axis] ||
          left.ghost_lower[axis] != right.ghost_lower[axis] ||
          left.ghost_upper[axis] != right.ghost_upper[axis])
        return false;
    return true;
  }

  void prepare_solver_request_once_(field_type& rhs, field_type& solution) {
    if (!topology_)
      throw std::logic_error("field solver request requires a prepared topology");
    const auto& global = topology_->global_patches();
    if (!solver_request_) {
      std::vector<component::FieldSolverPatchBindingV2> patches;
      patches.reserve(static_cast<std::size_t>(rhs.local_size()));
      for (std::size_t local = 0; local < rhs.local_size(); ++local) {
        const auto index = rhs.global_index(local);
        const auto& patch = global.at(index);
        patches.push_back({index,
                           const_view_(rhs.fab(local), rhs.box(local), patch.layout_identity,
                                       patch.patch_identity),
                           field_view_(solution.fab(local), solution.box(local),
                                       patch.layout_identity, patch.patch_identity),
                           {}});
      }
      solver_request_.emplace(component::bind_field_solver_request(
          *topology_, patches, spec_.execution->view(), spec_.boundary_contract_json.c_str(),
          spec_.relative_tolerance, spec_.absolute_tolerance, spec_.max_iterations));
      return;
    }

    const auto& cached = solver_request_->request();
    if (cached.local_patch_count != static_cast<std::size_t>(rhs.local_size()) ||
        cached.local_patch_count != static_cast<std::size_t>(solution.local_size()) ||
        (cached.local_patch_count != 0 && cached.local_patches == nullptr))
      throw std::runtime_error(
          "prepared FieldSolver request cannot be reused after local patch storage changed");
    for (std::size_t local = 0; local < rhs.local_size(); ++local) {
      const auto index = rhs.global_index(local);
      const auto& metadata = global.at(index);
      const auto expected_rhs = const_view_(rhs.fab(local), rhs.box(local),
                                            metadata.layout_identity, metadata.patch_identity);
      const auto expected_solution = field_view_(solution.fab(local), solution.box(local),
                                                 metadata.layout_identity, metadata.patch_identity);
      const auto& patch = cached.local_patches[static_cast<std::size_t>(local)];
      if (patch.struct_size < sizeof(PopsFieldSolverPatchV2) || patch.metadata_index != index ||
          !same_field_view_(patch.rhs, expected_rhs) ||
          !same_field_view_(patch.solution, expected_solution) ||
          !component::empty_field_view(patch.coefficients))
        throw std::runtime_error(
            "prepared FieldSolver request cannot be reused after field storage changed");
    }
  }

  static bool same_box_(const box_type& left, const box_type& right) { return left == right; }

  static bool same_layout_(const field_type& left, const field_type& right) {
    return left.layout() == right.layout() && left.distribution() == right.distribution() &&
           left.local_rank() == right.local_rank();
  }

  static mesh::BoxArrayValidationBudget layout_budget_(std::size_t boxes) {
    std::size_t pairs = 0;
    if (boxes > 1) {
      if (boxes - 1 > std::numeric_limits<std::size_t>::max() / boxes)
        throw std::length_error("field topology overlap proof exceeds size_t");
      pairs = boxes * (boxes - 1) / 2;
    }
    return {boxes, pairs};
  }

  static std::uint32_t periodic_axes_(const periodicity_type& periodicity) noexcept {
    std::uint32_t result = 0;
    for (int axis = 0; axis < Dim; ++axis)
      if (periodicity[static_cast<std::size_t>(axis)])
        result |= std::uint32_t{1} << static_cast<unsigned>(axis);
    return result;
  }

  static std::int32_t owner_rank_(const field_type& field, std::size_t patch) {
    std::size_t rank = 0;
    if (field.distribution().replicated()) {
      if (field.rank_space().size() != 1)
        throw std::invalid_argument(
            "external FieldSolver cannot encode a multi-rank replicated patch owner");
    } else {
      rank = field.rank_space().linear_rank(field.distribution().owner(patch));
    }
    if (rank > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
      throw std::overflow_error("external FieldSolver owner rank exceeds int32_t");
    return static_cast<std::int32_t>(rank);
  }

  static std::string hashed_identity_(const char* domain, const std::string& payload) {
    const std::vector<std::uint8_t> bytes(payload.begin(), payload.end());
    return std::string("pops.") + domain + ".v1:sha256:" + identity::sha256_hex(bytes);
  }

  std::string runtime_layout_identity_(const field_type& field, const geometry_type& geometry,
                                       const periodicity_type& periodicity) const {
    std::ostringstream payload;
    payload.imbue(std::locale::classic());
    payload << std::setprecision(std::numeric_limits<double>::max_digits10)
            << spec_.source_layout_identity << '|' << spec_.topology_recipe_identity << '|' << Dim
            << '|';
    for (int axis = 0; axis < Dim; ++axis)
      payload << (periodicity[static_cast<std::size_t>(axis)] ? 1 : 0) << ':'
              << geometry.domain().lo[axis] << ':' << geometry.domain().hi[axis] << ':'
              << geometry.lower()[axis] << ':' << geometry.upper()[axis] << ';';
    payload << '|';
    for (std::size_t index = 0; index < field.layout().size(); ++index) {
      const auto& box = field.layout()[index];
      payload << index << ':' << owner_rank_(field, index) << ':';
      for (int axis = 0; axis < Dim; ++axis)
        payload << box.lo[axis] << ':' << box.hi[axis] << ':';
      payload << ';';
    }
    return hashed_identity_("runtime-field-layout", payload.str());
  }

  static std::string runtime_patch_identity_(const std::string& layout, std::size_t index,
                                             const box_type& box) {
    std::string payload = layout + '|' + std::to_string(index);
    for (int axis = 0; axis < Dim; ++axis)
      payload += '|' + std::to_string(box.lo[axis]) + '|' + std::to_string(box.hi[axis]);
    return hashed_identity_("runtime-field-patch", payload);
  }

  void validate_topology_reuse_(const field_type& field, const geometry_type& geometry,
                                const periodicity_type& periodicity) const {
    const auto& global = topology_->global_topology();
    bool exact = global.dimension == static_cast<std::uint32_t>(Dim) &&
                 global.topology_recipe_identity != nullptr &&
                 global.source_layout_identity != nullptr &&
                 global.materialized_layout_identity != nullptr &&
                 spec_.topology_recipe_identity == global.topology_recipe_identity &&
                 spec_.source_layout_identity == global.source_layout_identity &&
                 materialized_layout_identity_ == global.materialized_layout_identity &&
                 global.periodic_axes == periodic_axes_(periodicity) &&
                 global.patch_count == field.layout().size() &&
                 patch_identities_.size() == global.patch_count;
    for (int axis = 0; axis < Dim; ++axis)
      exact = exact && global.domain_lower[axis] == geometry.domain().lo[axis] &&
              global.domain_upper[axis] == geometry.domain().hi[axis];
    for (int axis = Dim; axis < 3; ++axis)
      exact = exact && global.domain_lower[axis] == 0 && global.domain_upper[axis] == 0;
    if (!exact)
      throw std::runtime_error(
          "prepared external field topology cannot be reused after a layout change");
    for (std::size_t index = 0; index < global.patch_count; ++index) {
      const auto& box = field.layout()[index];
      const auto& patch = global.patches[index];
      bool row_exact = patch.global_patch_index == index &&
                       patch.owner_rank == owner_rank_(field, index) &&
                       patch.dimension == static_cast<std::uint32_t>(Dim) &&
                       patch.layout_identity != nullptr && patch.patch_identity != nullptr &&
                       spec_.source_layout_identity == patch.layout_identity &&
                       patch_identities_[index] == patch.patch_identity;
      for (int axis = 0; axis < Dim; ++axis) {
        const double physical_lower =
            geometry.lower()[axis] +
            static_cast<double>(box.lo[axis] - geometry.domain().lo[axis]) * geometry.spacing(axis);
        row_exact = row_exact && patch.lower[axis] == box.lo[axis] &&
                    patch.upper[axis] == box.hi[axis] &&
                    patch.physical_lower[axis] == physical_lower &&
                    patch.cell_spacing[axis] == geometry.spacing(axis);
      }
      for (int axis = Dim; axis < 3; ++axis)
        row_exact = row_exact && patch.lower[axis] == 0 && patch.upper[axis] == 0 &&
                    patch.physical_lower[axis] == 0.0 && patch.cell_spacing[axis] == 0.0;
      if (!row_exact)
        throw std::runtime_error(
            "prepared external field topology cannot be reused after a layout change");
    }
  }

  void prepare_topology_once_(const field_type& field, const geometry_type& geometry,
                              const periodicity_type& periodicity) {
    if (topology_) {
      collective_preflight_([&] { validate_topology_reuse_(field, geometry, periodicity); },
                            "external FieldTopology reuse validation failed collectively");
      return;
    }
    std::string layout;
    collective_preflight_([&] { layout = runtime_layout_identity_(field, geometry, periodicity); },
                          "external FieldTopology layout preparation failed collectively");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"external-system-field-layout", std::string_view(layout)}}))
      throw std::invalid_argument("external FieldTopology global layout differs between MPI ranks");
    materialized_layout_identity_ = layout;
    patch_identities_.clear();
    patch_identities_.reserve(field.layout().size());
    for (std::size_t index = 0; index < field.layout().size(); ++index)
      patch_identities_.push_back(
          runtime_patch_identity_(materialized_layout_identity_, index, field.layout()[index]));

    std::vector<PopsFieldPatchMetadataV1> global;
    global.reserve(field.layout().size());
    for (std::size_t index = 0; index < field.layout().size(); ++index) {
      const auto& box = field.layout()[index];
      PopsFieldPatchMetadataV1 row{};
      row.struct_size = sizeof(PopsFieldPatchMetadataV1);
      row.global_patch_index = index;
      row.owner_rank = owner_rank_(field, index);
      row.dimension = Dim;
      row.centering = POPS_FIELD_CENTERING_CELL_V1;
      row.layout_identity = spec_.source_layout_identity.c_str();
      row.patch_identity = patch_identities_[index].c_str();
      for (int axis = 0; axis < Dim; ++axis) {
        row.lower[axis] = box.lo[axis];
        row.upper[axis] = box.hi[axis];
        row.physical_lower[axis] =
            geometry.lower()[axis] +
            static_cast<double>(box.lo[axis] - geometry.domain().lo[axis]) * geometry.spacing(axis);
        row.cell_spacing[axis] = geometry.spacing(axis);
      }
      global.push_back(row);
    }
    PopsFieldGlobalTopologyV1 global_topology{};
    global_topology.struct_size = sizeof(PopsFieldGlobalTopologyV1);
    global_topology.topology_recipe_identity = spec_.topology_recipe_identity.c_str();
    global_topology.source_layout_identity = spec_.source_layout_identity.c_str();
    global_topology.materialized_layout_identity = materialized_layout_identity_.c_str();
    global_topology.dimension = Dim;
    global_topology.periodic_axes = periodic_axes_(periodicity);
    global_topology.patch_count = global.size();
    global_topology.patches = global.data();
    for (int axis = 0; axis < Dim; ++axis) {
      global_topology.domain_lower[axis] = geometry.domain().lo[axis];
      global_topology.domain_upper[axis] = geometry.domain().hi[axis];
    }
    std::vector<component::FieldTopologyPatchInputV2> local;
    local.reserve(field.local_size());
    for (std::size_t index = 0; index < field.local_size(); ++index)
      local.push_back({field.global_index(index), POPS_FIELD_MATERIAL_FULL_V1, {}, {}, {}});
    const auto& api = topology_component_->table<PopsFieldTopologyApiV2>(
        POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, spec_.topology_interface_version);
    std::optional<component::PreparedFieldTopologyV2> prepared;
    std::exception_ptr topology_error;
    try {
      prepared.emplace(component::prepare_field_topology(api, topology_state_, global_topology,
                                                         local, spec_.execution->view()));
    } catch (...) {
      topology_error = std::current_exception();
    }
    if (all_reduce_max(topology_error ? 1L : 0L) != 0) {
      if (n_ranks() == 1 && topology_error)
        std::rethrow_exception(topology_error);
      throw std::runtime_error("external FieldTopology preparation failed collectively");
    }
    topology_ = std::move(prepared);
  }

  void validate_solve_layout_(const field_type& rhs, const field_type& solution,
                              const geometry_type& geometry) const {
    if (rhs.ncomp() != 1 || solution.ncomp() != 1 || rhs.local_size() != solution.local_size() ||
        !same_layout_(rhs, solution))
      throw std::invalid_argument(
          "prepared FieldSolver requires matching scalar RHS/solution global layouts");
    if (!rhs.layout().tiles_exactly(geometry.domain(), layout_budget_(rhs.layout().size())))
      throw std::invalid_argument(
          "prepared full-material FieldSolver patches do not exactly cover the domain");
    for (std::size_t local = 0; local < rhs.local_size(); ++local)
      if (rhs.global_index(local) != solution.global_index(local) ||
          !same_box_(rhs.box(local), solution.box(local)))
        throw std::invalid_argument(
            "prepared FieldSolver local RHS/solution patch identities differ");
  }

  void validate_() const {
    if (!topology_component_ || !solver_component_ || !spec_.execution ||
        spec_.provider_slot.empty() || spec_.topology_component_id.empty() ||
        spec_.topology_manifest_identity.empty() || spec_.solver_component_id.empty() ||
        spec_.topology_parameters_json.empty() || spec_.solver_manifest_identity.empty() ||
        spec_.solver_parameters_json.empty() || spec_.source_layout_identity.empty() ||
        spec_.topology_recipe_identity.empty() ||
        spec_.boundary_contract_json.find("\"identity\"") == std::string::npos ||
        spec_.topology_interface_version != 2 || spec_.solver_interface_version != 2 ||
        !std::isfinite(spec_.relative_tolerance) || spec_.relative_tolerance < 0.0 ||
        !std::isfinite(spec_.absolute_tolerance) || spec_.absolute_tolerance < 0.0 ||
        spec_.max_iterations < 1)
      throw std::invalid_argument("prepared external field solver specification is incomplete");
    const auto execution = spec_.execution->view();
    component::validate_execution_context(execution);
    if constexpr (!Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                              typename field_type::memory_space>::accessible)
      throw std::invalid_argument(
          "external FieldSolver host execution cannot borrow device-only System storage");
    if (execution.memory_space != POPS_MEMORY_SPACE_HOST_V1 ||
        (std::string_view(execution.device_identity) != "host" &&
         std::string_view(execution.device_identity) != "cpu"))
      throw std::invalid_argument("external FieldSolver requires host-resident execution");
    const std::string communicator_identity(execution.communicator_identity);
    if (communicator_identity == "serial") {
      if (n_ranks() != 1)
        throw std::invalid_argument(
            "external FieldSolver cannot use a serial component context on multiple ranks");
    } else {
      if (communicator_identity != "MPI_COMM_WORLD" || !spec_.component_pair_declares_mpi)
        throw std::invalid_argument(
            "external FieldSolver MPI requires a declared MPI_COMM_WORLD component pair");
#ifdef POPS_HAS_MPI
      int initialized = 0;
      pops::detail::require_mpi_success(MPI_Initialized(&initialized),
                                        "MPI_Initialized(external System field)");
      if (initialized == 0)
        throw std::invalid_argument(
            "external FieldSolver received MPI authority before MPI initialization");
      int relation = MPI_UNEQUAL;
      pops::detail::require_mpi_success(
          MPI_Comm_compare(MPI_Comm_f2c(static_cast<MPI_Fint>(execution.communicator_f_handle)),
                           MPI_COMM_WORLD, &relation),
          "MPI_Comm_compare(external System field)");
      if (relation != MPI_IDENT || MPI_Type_f2c(static_cast<MPI_Fint>(
                                       execution.communicator_datatype_f_handle)) != MPI_DOUBLE)
        throw std::invalid_argument(
            "external FieldSolver execution handles are not exact MPI_COMM_WORLD/MPI_DOUBLE");
#else
      throw std::invalid_argument(
          "external FieldSolver MPI execution requires an MPI-enabled PoPS build");
#endif
    }
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
  std::string provider_identity_;
  std::string collective_contract_;
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
