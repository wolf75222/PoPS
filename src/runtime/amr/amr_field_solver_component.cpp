#include <pops/runtime/amr/amr_runtime.hpp>

#include <pops/core/identity/sha256.hpp>
#include <pops/runtime/system/prepared_field_solver_component.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace pops {
namespace {

using runtime::field::PreparedFieldSolverSpec;

constexpr std::string_view kExternalOptionsSchema = "pops.external.field-solver-request@2";
constexpr std::string_view kCompositePolicy = "pops.field-hierarchy.composite";

std::string hashed_identity(std::string_view domain, std::string_view payload) {
  const std::vector<std::uint8_t> bytes(payload.begin(), payload.end());
  return "pops." + std::string(domain) + ".v1:sha256:" + identity::sha256_hex(bytes);
}

std::string exact_external_provider_contract(const PreparedFieldSolverSpec& spec) {
  ExactContractBuilder contract;
  contract.text("pops.runtime.external-amr-field-solver-provider")
      .scalar(std::uint32_t{1})
      .text(spec.provider_slot)
      .text(spec.topology_component_id)
      .text(spec.topology_manifest_identity)
      .scalar(spec.topology_interface_version)
      .text(spec.topology_parameters_json)
      .text(spec.solver_component_id)
      .text(spec.solver_manifest_identity)
      .scalar(spec.solver_interface_version)
      .text(spec.solver_parameters_json)
      .text(spec.source_layout_identity)
      .text(spec.topology_recipe_identity)
      .text(spec.boundary_contract_json)
      .scalar(spec.relative_tolerance)
      .scalar(spec.absolute_tolerance)
      .scalar(spec.max_iterations)
      .scalar(spec.component_pair_declares_mpi)
      .text(spec.execution == nullptr ? "" : spec.execution->identity());
  return std::move(contract).release();
}

AmrFieldSolverOptions external_options(const PreparedFieldSolverSpec& spec) {
  return {std::string(kExternalOptionsSchema),
          {{"absolute_tolerance", spec.absolute_tolerance},
           {"max_iterations", static_cast<std::int64_t>(spec.max_iterations)},
           {"relative_tolerance", spec.relative_tolerance}}};
}

bool exact_external_options(const AmrFieldSolverOptions& options,
                            const PreparedFieldSolverSpec& spec) noexcept {
  try {
    if (options.schema_identity != kExternalOptionsSchema || options.values.size() != 3)
      return false;
    return std::get<double>(options.values.at("relative_tolerance")) == spec.relative_tolerance &&
           std::get<double>(options.values.at("absolute_tolerance")) == spec.absolute_tolerance &&
           std::get<std::int64_t>(options.values.at("max_iterations")) == spec.max_iterations;
  } catch (...) {
    return false;
  }
}

bool exact_composite_policy(const AmrFieldHierarchyPolicyAuthority& authority) noexcept {
  return authority.policy_id == kCompositePolicy && authority.interface_version == 1 &&
         authority.options.schema_identity == "pops.field-hierarchy.options.empty@1" &&
         authority.options.values.empty();
}

bool paired_periodic_boundary(const BCRec& boundary) noexcept {
  return (boundary.xlo == BCType::Periodic) == (boundary.xhi == BCType::Periodic) &&
         (boundary.ylo == BCType::Periodic) == (boundary.yhi == BCType::Periodic);
}

std::uint32_t periodic_axes(const BCRec& boundary) {
  if (!paired_periodic_boundary(boundary))
    throw std::invalid_argument(
        "external AMR field solver requires paired periodic boundary faces");
  return (boundary.xlo == BCType::Periodic ? 1u : 0u) |
         (boundary.ylo == BCType::Periodic ? 2u : 0u);
}

void validate_external_execution(const PreparedFieldSolverSpec& spec) {
  if (spec.execution == nullptr)
    throw std::invalid_argument("external AMR field solver requires an execution authority");
  const PopsExecutionContextV1 execution = spec.execution->view();
  component::validate_execution_context(execution);
  if (execution.memory_space != POPS_MEMORY_SPACE_HOST_V1 ||
      (std::string_view(execution.device_identity) != "host" &&
       std::string_view(execution.device_identity) != "cpu"))
    throw std::invalid_argument("external AMR field solver supports host-resident execution only");
  const std::string_view communicator(execution.communicator_identity);
  if (communicator == "serial") {
    if (n_ranks() != 1)
      throw std::invalid_argument(
          "external AMR field solver cannot use a serial component context on multiple ranks");
    return;
  }
  if (communicator != "MPI_COMM_WORLD" || !spec.component_pair_declares_mpi)
    throw std::invalid_argument(
        "external AMR field solver MPI requires an exact declared MPI_COMM_WORLD component pair");
#ifdef POPS_HAS_MPI
  int initialized = 0;
  detail::require_mpi_success(MPI_Initialized(&initialized), "MPI_Initialized(external field)");
  if (initialized == 0)
    throw std::invalid_argument(
        "external AMR field solver received MPI authority before MPI initialization");
  int relation = MPI_UNEQUAL;
  detail::require_mpi_success(
      MPI_Comm_compare(MPI_Comm_f2c(static_cast<MPI_Fint>(execution.communicator_f_handle)),
                       MPI_COMM_WORLD, &relation),
      "MPI_Comm_compare(external field)");
  if (relation != MPI_IDENT ||
      MPI_Type_f2c(static_cast<MPI_Fint>(execution.communicator_datatype_f_handle)) != MPI_DOUBLE)
    throw std::invalid_argument(
        "external AMR field solver execution handles are not exact MPI_COMM_WORLD/MPI_DOUBLE");
#else
  throw std::invalid_argument(
      "external AMR field solver cannot install MPI execution in a serial PoPS build");
#endif
}

SolveStatus solve_status(std::int32_t status) {
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
  throw std::invalid_argument("FieldSolver@2 returned an unknown solve status");
}

SolveAction solve_action(std::int32_t action) {
  switch (action) {
    case POPS_SOLVE_ACTION_NONE_V2:
      return SolveAction::kNone;
    case POPS_SOLVE_ACTION_FAIL_RUN_V2:
      return SolveAction::kFailRun;
    case POPS_SOLVE_ACTION_REJECT_ATTEMPT_V2:
      return SolveAction::kRejectAttempt;
  }
  throw std::invalid_argument("FieldSolver@2 returned an unknown solve action");
}

const Real* valid_data(const Fab2D& fab, const Box2D& valid) {
  const ConstArray4 view = fab.const_array();
  return view.p + static_cast<std::ptrdiff_t>(valid.lo[1] - view.jg0) * view.nx_tot +
         (valid.lo[0] - view.ig0);
}

Real* valid_data(Fab2D& fab, const Box2D& valid) {
  const Array4 view = fab.array();
  return view.p + static_cast<std::ptrdiff_t>(valid.lo[1] - view.jg0) * view.nx_tot +
         (valid.lo[0] - view.ig0);
}

PopsConstFieldViewV1 const_view(const Fab2D& fab, const Box2D& valid, const char* layout,
                                const char* patch) {
  const ConstArray4 storage = fab.const_array();
  return {sizeof(PopsConstFieldViewV1),
          valid_data(fab, valid),
          2,
          {static_cast<std::size_t>(valid.nx()), static_cast<std::size_t>(valid.ny()), 1},
          {1, storage.nx_tot, 0},
          1,
          storage.comp_stride,
          POPS_FIELD_CENTERING_CELL_V1,
          0,
          {0, 0, 0},
          {0, 0, 0},
          POPS_SCALAR_FLOAT64_V1,
          POPS_MEMORY_SPACE_HOST_V1,
          layout,
          patch,
          POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
}

PopsFieldViewV1 field_view(Fab2D& fab, const Box2D& valid, const char* layout, const char* patch) {
  const Array4 storage = fab.array();
  return {sizeof(PopsFieldViewV1),
          valid_data(fab, valid),
          2,
          {static_cast<std::size_t>(valid.nx()), static_cast<std::size_t>(valid.ny()), 1},
          {1, storage.nx_tot, 0},
          1,
          storage.comp_stride,
          POPS_FIELD_CENTERING_CELL_V1,
          0,
          {0, 0, 0},
          {0, 0, 0},
          POPS_SCALAR_FLOAT64_V1,
          POPS_MEMORY_SPACE_HOST_V1,
          layout,
          patch,
          POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
}

class PreparedExternalAmrFieldSolver final : public AmrPreparedFieldSolver {
 public:
  PreparedExternalAmrFieldSolver(const AmrFieldSolverBuildRequest& request,
                                 PreparedFieldSolverSpec spec,
                                 std::shared_ptr<component::LoadedComponent> topology_component,
                                 std::shared_ptr<component::LoadedComponent> solver_component,
                                 std::string exact_contract)
      : spec_(std::move(spec)),
        exact_contract_(std::move(exact_contract)),
        topology_component_(std::move(topology_component)),
        solver_component_(std::move(solver_component)) {
    static_assert(sizeof(Real) == sizeof(double),
                  "FieldSolver ABI v2 requires the binary64 PoPS backend");
    if (!topology_component_ || !solver_component_)
      throw std::invalid_argument("external AMR field solver lost its component handles");
    topology_state_ = topology_component_->prepare_fresh_state(
        POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, spec_.topology_interface_version,
        spec_.execution->view(), spec_.topology_parameters_json);
    solver_state_ = solver_component_->prepare_fresh_state(
        POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, spec_.solver_interface_version,
        spec_.execution->view(), spec_.solver_parameters_json);
    materialize_(request);
  }

  [[nodiscard]] std::string_view provider_identity() const noexcept override {
    return spec_.provider_slot;
  }
  [[nodiscard]] std::string_view exact_prepared_contract() const noexcept override {
    return exact_contract_;
  }
  [[nodiscard]] std::string_view exact_materialization_evidence() const noexcept override {
    return materialization_evidence_;
  }
  [[nodiscard]] bool requires_runtime_solution_halos() const noexcept override { return true; }
  [[nodiscard]] bool couples_hierarchy_levels() const noexcept override { return true; }
  [[nodiscard]] int level_count() const noexcept override { return static_cast<int>(rhs_.size()); }
  [[nodiscard]] FieldDistribution level_distribution(int level) const override {
    return distributions_.at(static_cast<std::size_t>(level));
  }
  MultiFab& rhs_level(int level) override { return rhs_.at(static_cast<std::size_t>(level)); }
  MultiFab& phi_level(int level) override { return phi_.at(static_cast<std::size_t>(level)); }
  void set_boundary_context(const FieldBoundaryExecutionContext&) override {
    throw std::runtime_error("external FieldSolver@2 carries only its immutable boundary contract");
  }
  [[nodiscard]] const SolveReport& last_solve_report() const noexcept override { return report_; }

 private:
  SolveReport solve() override {
    for (auto& level : rhs_)
      level.sync_host();
    for (auto& level : phi_)
      level.sync_host();
    PopsSolveReportV2 native{};
    native.struct_size = sizeof(PopsSolveReportV2);
    const auto& api = solver_component_->table<PopsFieldSolverApiV2>(
        POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, spec_.solver_interface_version);
    (void)component::solve_field(api, solver_state_.get(), *solver_request_, native);

    report_ = {};
    report_.iters = native.iterations;
    report_.rel_residual = static_cast<Real>(native.relative_residual);
    report_.reference_residual_norm = static_cast<Real>(native.reference_residual_norm);
    report_.residual_norm = static_cast<Real>(native.residual_norm);
    const SolveStatus status = solve_status(native.status);
    const SolveAction action = solve_action(native.action);
    if (status == SolveStatus::kSolved) {
      if (!active_solution_is_finite_()) {
        report_.mark_failed(
            SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
            "native FieldSolver@2 marked a non-finite active hierarchy solution as solved");
        return report_;
      }
      for (auto& level : phi_)
        level.sync_device();
      report_.mark_solved(native.reason);
      return report_;
    }
    report_.mark_failed(status, action, native.reason);
    return report_;
  }

  bool active_solution_is_finite_() const {
    if (!topology_ || topology_->local_patches().size() != local_locations_.size())
      return false;
    for (std::size_t index = 0; index < local_locations_.size(); ++index) {
      const auto [level, local] = local_locations_[index];
      const MultiFab& field = phi_.at(static_cast<std::size_t>(level));
      const Box2D valid = field.box(local);
      const auto& patch = topology_->local_patches()[index];
      if (patch.material_mask.size() != static_cast<std::size_t>(valid.num_cells()))
        return false;
      const ConstArray4 values = field.fab(local).const_array();
      std::size_t point = 0;
      for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
        for (int i = valid.lo[0]; i <= valid.hi[0]; ++i, ++point)
          if (patch.material_mask[point] > 1 ||
              (patch.material_mask[point] == 1 && !std::isfinite(values(i, j, 0))))
            return false;
    }
    return true;
  }

  static std::vector<std::uint8_t> binary_coverage_(const Box2D& valid, const BoxArray* fine_boxes,
                                                    int ratio) {
    std::vector<Box2D> footprints;
    if (fine_boxes != nullptr) {
      footprints.reserve(static_cast<std::size_t>(fine_boxes->size()));
      for (const Box2D& fine : fine_boxes->boxes()) {
        const Box2D footprint = fine.coarsen(ratio);
        if (footprint.refine(ratio) != fine)
          throw std::invalid_argument(
              "external AMR field solver requires refinement-aligned fine patches");
        footprints.push_back(footprint);
      }
    }
    std::vector<std::uint8_t> result(static_cast<std::size_t>(valid.num_cells()), 1);
    std::size_t point = 0;
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i, ++point)
        if (std::any_of(footprints.begin(), footprints.end(),
                        [i, j](const Box2D& footprint) { return footprint.contains(i, j); }))
          result[point] = 0;
    return result;
  }

  void materialize_(const AmrFieldSolverBuildRequest& request) {
    const int levels = request.hierarchy.nlev();
    if (levels < 1 || request.hierarchy.ba.size() != request.hierarchy.dm.size() ||
        request.hierarchy.ba.size() != request.hierarchy.dx.size() ||
        request.hierarchy.ba.size() != request.hierarchy.dy.size() ||
        request.hierarchy.refinement_ratios.size() + 1 != request.hierarchy.ba.size())
      throw std::invalid_argument("external AMR field solver hierarchy is incomplete");
    geometries_.reserve(static_cast<std::size_t>(levels));
    rhs_.reserve(static_cast<std::size_t>(levels));
    phi_.reserve(static_cast<std::size_t>(levels));
    distributions_.reserve(static_cast<std::size_t>(levels));
    level_offsets_.reserve(static_cast<std::size_t>(levels));
    std::size_t global_patch_count = 0;
    int refinement = 1;
    for (int level = 0; level < levels; ++level) {
      const auto index = static_cast<std::size_t>(level);
      level_offsets_.push_back(global_patch_count);
      global_patch_count += static_cast<std::size_t>(request.hierarchy.ba[index].size());
      geometries_.push_back(request.geometry.refine(refinement));
      if (geometries_.back().dx() != request.hierarchy.dx[index] ||
          geometries_.back().dy() != request.hierarchy.dy[index])
        throw std::invalid_argument(
            "external AMR field solver geometry differs from the hierarchy spacing");
      rhs_.emplace_back(request.hierarchy.ba[index], request.hierarchy.dm[index], 1, 0);
      phi_.emplace_back(request.hierarchy.ba[index], request.hierarchy.dm[index], 1, 1);
      rhs_.back().set_val(Real(0));
      phi_.back().set_val(Real(0));
      distributions_.push_back(level == 0 && request.replicated_coarse
                                   ? FieldDistribution::Replicated
                                   : FieldDistribution::Distributed);
      if (index < request.hierarchy.refinement_ratios.size()) {
        const int ratio = request.hierarchy.refinement_ratios[index];
        if (ratio != kAmrRefRatio || refinement > std::numeric_limits<int>::max() / ratio)
          throw std::invalid_argument(
              "external AMR field solver requires representable ratio-2 transitions");
        refinement *= ratio;
      }
    }
    if (global_patch_count == 0)
      throw std::invalid_argument("external AMR field solver hierarchy has no global patches");

    ExactContractBuilder layout_contract;
    layout_contract.text("pops.runtime.external-amr-field-layout")
        .scalar(std::uint32_t{1})
        .text(spec_.source_layout_identity)
        .text(spec_.topology_recipe_identity)
        .scalar(periodic_axes(request.boundary))
        .scalar(levels)
        .scalar(request.geometry.xlo)
        .scalar(request.geometry.xhi)
        .scalar(request.geometry.ylo)
        .scalar(request.geometry.yhi);
    for (int level = 0; level < levels; ++level) {
      const auto li = static_cast<std::size_t>(level);
      layout_contract.scalar(level)
          .scalar(request.hierarchy.dx[li])
          .scalar(request.hierarchy.dy[li])
          .scalar(request.hierarchy.ba[li].size());
      for (int patch = 0; patch < request.hierarchy.ba[li].size(); ++patch) {
        const Box2D& box = request.hierarchy.ba[li][patch];
        layout_contract.scalar(patch)
            .scalar(request.hierarchy.dm[li][patch])
            .scalar(box.lo[0])
            .scalar(box.lo[1])
            .scalar(box.hi[0])
            .scalar(box.hi[1]);
      }
      layout_contract.scalar(li < request.hierarchy.refinement_ratios.size()
                                 ? request.hierarchy.refinement_ratios[li]
                                 : 1);
    }
    materialized_layout_identity_ =
        hashed_identity("runtime-amr-field-layout", std::move(layout_contract).release());
    patch_identities_.reserve(global_patch_count);
    std::vector<PopsFieldPatchMetadataV1> global;
    global.reserve(global_patch_count);
    for (int level = 0; level < levels; ++level) {
      const auto li = static_cast<std::size_t>(level);
      const Geometry& geometry = geometries_[li];
      const BoxArray& boxes = request.hierarchy.ba[li];
      for (int patch = 0; patch < boxes.size(); ++patch) {
        const std::size_t global_index = level_offsets_[li] + static_cast<std::size_t>(patch);
        const Box2D& box = boxes[patch];
        ExactContractBuilder identity_contract;
        identity_contract.text(materialized_layout_identity_)
            .scalar(level)
            .scalar(patch)
            .scalar(box.lo[0])
            .scalar(box.lo[1])
            .scalar(box.hi[0])
            .scalar(box.hi[1]);
        patch_identities_.push_back(
            hashed_identity("runtime-amr-field-patch", std::move(identity_contract).release()));
        const int owner = request.hierarchy.dm[li][patch];
        PopsFieldPatchMetadataV1 row{sizeof(PopsFieldPatchMetadataV1),
                                     global_index,
                                     owner,
                                     level,
                                     2,
                                     {},
                                     {},
                                     {},
                                     {},
                                     POPS_FIELD_CENTERING_CELL_V1,
                                     0,
                                     spec_.source_layout_identity.c_str(),
                                     patch_identities_.back().c_str()};
        row.lower[0] = box.lo[0];
        row.lower[1] = box.lo[1];
        row.upper[0] = box.hi[0];
        row.upper[1] = box.hi[1];
        row.physical_lower[0] =
            geometry.xlo + static_cast<double>(box.lo[0] - geometry.domain.lo[0]) * geometry.dx();
        row.physical_lower[1] =
            geometry.ylo + static_cast<double>(box.lo[1] - geometry.domain.lo[1]) * geometry.dy();
        row.cell_spacing[0] = geometry.dx();
        row.cell_spacing[1] = geometry.dy();
        global.push_back(row);
      }
    }

    PopsFieldGlobalTopologyV1 global_topology{sizeof(PopsFieldGlobalTopologyV1),
                                              spec_.topology_recipe_identity.c_str(),
                                              spec_.source_layout_identity.c_str(),
                                              materialized_layout_identity_.c_str(),
                                              2,
                                              {},
                                              {},
                                              periodic_axes(request.boundary),
                                              global.size(),
                                              global.data()};
    for (int axis = 0; axis < 2; ++axis) {
      global_topology.domain_lower[axis] = std::numeric_limits<std::int64_t>::max();
      global_topology.domain_upper[axis] = std::numeric_limits<std::int64_t>::min();
      for (const auto& patch : global) {
        global_topology.domain_lower[axis] =
            std::min(global_topology.domain_lower[axis], patch.lower[axis]);
        global_topology.domain_upper[axis] =
            std::max(global_topology.domain_upper[axis], patch.upper[axis]);
      }
    }

    std::size_t local_patch_count = 0;
    for (const MultiFab& level : rhs_)
      local_patch_count += static_cast<std::size_t>(level.local_size());
    local_locations_.reserve(local_patch_count);
    coverage_.reserve(local_patch_count);
    std::vector<component::FieldTopologyPatchInputV2> local_topology;
    local_topology.reserve(local_patch_count);
    for (int level = 0; level < levels; ++level) {
      const auto li = static_cast<std::size_t>(level);
      const BoxArray* fine = level + 1 < levels ? &request.hierarchy.ba[li + 1] : nullptr;
      const int ratio = fine == nullptr ? 1 : request.hierarchy.refinement_ratios.at(li);
      for (int local = 0; local < rhs_[li].local_size(); ++local) {
        const int patch = rhs_[li].global_index(local);
        const std::size_t metadata_index = level_offsets_[li] + static_cast<std::size_t>(patch);
        local_locations_.emplace_back(level, local);
        coverage_.push_back(binary_coverage_(rhs_[li].box(local), fine, ratio));
        const auto& mask = coverage_.back();
        local_topology.push_back({metadata_index,
                                  POPS_FIELD_MATERIAL_BINARY_COVERAGE_V1,
                                  {sizeof(PopsConstByteViewV1), mask.data(), mask.size()},
                                  {},
                                  {}});
      }
    }

    const auto& topology_api = topology_component_->table<PopsFieldTopologyApiV2>(
        POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, spec_.topology_interface_version);
    topology_.emplace(component::prepare_field_topology(topology_api, topology_state_.get(),
                                                        global_topology, local_topology,
                                                        spec_.execution->view()));
    ExactContractBuilder label_contract;
    label_contract.text("pops.external-amr-field-topology-labels")
        .scalar(std::uint32_t{1})
        .sequence(topology_->labels(),
                  [](ExactContractBuilder& row, const component::PreparedTopologyLabelV2& label) {
                    row.scalar(label.id).text(label.label).text(label.provenance);
                  });
    ExactContractBuilder evidence;
    evidence.text("pops.external-amr-field-materialization")
        .scalar(std::uint32_t{1})
        .text(topology_->topology_digest())
        .text(topology_->provenance())
        .bytes(label_contract.view())
        .text(materialized_layout_identity_);
    materialization_evidence_ = std::move(evidence).release();

    std::vector<component::FieldSolverPatchBindingV2> bindings;
    bindings.reserve(local_locations_.size());
    for (std::size_t index = 0; index < local_locations_.size(); ++index) {
      const auto [level, local] = local_locations_[index];
      const auto li = static_cast<std::size_t>(level);
      const int patch = rhs_[li].global_index(local);
      const std::size_t metadata_index = level_offsets_[li] + static_cast<std::size_t>(patch);
      const auto& metadata = topology_->global_patches().at(metadata_index);
      bindings.push_back({metadata_index,
                          const_view(rhs_[li].fab(local), rhs_[li].box(local),
                                     metadata.layout_identity, metadata.patch_identity),
                          field_view(phi_[li].fab(local), phi_[li].box(local),
                                     metadata.layout_identity, metadata.patch_identity),
                          {}});
    }
    solver_request_.emplace(component::bind_field_solver_request(
        *topology_, bindings, spec_.execution->view(), spec_.boundary_contract_json.c_str(),
        spec_.relative_tolerance, spec_.absolute_tolerance, spec_.max_iterations));
  }

  PreparedFieldSolverSpec spec_;
  std::string exact_contract_;
  std::shared_ptr<component::LoadedComponent> topology_component_;
  std::shared_ptr<component::LoadedComponent> solver_component_;
  component::LoadedComponent::PreparedState topology_state_;
  component::LoadedComponent::PreparedState solver_state_;
  std::vector<Geometry> geometries_;
  std::vector<MultiFab> rhs_;
  std::vector<MultiFab> phi_;
  std::vector<FieldDistribution> distributions_;
  std::vector<std::size_t> level_offsets_;
  std::vector<std::string> patch_identities_;
  std::string materialized_layout_identity_;
  std::vector<std::pair<int, int>> local_locations_;
  std::vector<std::vector<std::uint8_t>> coverage_;
  std::optional<component::PreparedFieldTopologyV2> topology_;
  std::optional<component::TopologyBoundFieldSolverRequestV2> solver_request_;
  std::string materialization_evidence_;
  SolveReport report_{};
};

class ExternalAmrFieldSolverProvider final : public AmrFieldSolverProvider {
 public:
  ExternalAmrFieldSolverProvider(PreparedFieldSolverSpec spec,
                                 std::shared_ptr<component::LoadedComponent> topology,
                                 std::shared_ptr<component::LoadedComponent> solver)
      : spec_(std::move(spec)),
        topology_(std::move(topology)),
        solver_(std::move(solver)),
        collective_contract_(exact_external_provider_contract(spec_)) {
    if (spec_.provider_slot.empty() || spec_.topology_component_id.empty() ||
        spec_.topology_manifest_identity.empty() || spec_.solver_component_id.empty() ||
        spec_.topology_parameters_json.empty() || spec_.solver_manifest_identity.empty() ||
        spec_.solver_parameters_json.empty() || spec_.source_layout_identity.empty() ||
        spec_.topology_recipe_identity.empty() || spec_.boundary_contract_json.empty() ||
        spec_.boundary_contract_json.find("\"identity\"") == std::string::npos ||
        spec_.topology_interface_version != 2 || spec_.solver_interface_version != 2 ||
        !std::isfinite(spec_.relative_tolerance) || spec_.relative_tolerance < 0.0 ||
        !std::isfinite(spec_.absolute_tolerance) || spec_.absolute_tolerance < 0.0 ||
        spec_.max_iterations < 1 || !topology_ || !solver_)
      throw std::invalid_argument("external AMR field solver specification is incomplete");
    validate_external_execution(spec_);
    multi_rank_execution_ = n_ranks() > 1;
    const auto& topology_api = topology_->api();
    const auto& solver_api = solver_->api();
    if (topology_api.component_id == nullptr || topology_api.manifest_identity == nullptr ||
        solver_api.component_id == nullptr || solver_api.manifest_identity == nullptr ||
        spec_.topology_component_id != topology_api.component_id ||
        spec_.topology_manifest_identity != topology_api.manifest_identity ||
        spec_.solver_component_id != solver_api.component_id ||
        spec_.solver_manifest_identity != solver_api.manifest_identity)
      throw std::invalid_argument("external AMR field solver changed component identity");
    const auto& topology_table = topology_->table<PopsFieldTopologyApiV2>(
        POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, spec_.topology_interface_version);
    const auto& solver_table = solver_->table<PopsFieldSolverApiV2>(
        POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, spec_.solver_interface_version);
    component::require_operation(topology_table.prepare_topology != nullptr, "prepare_topology");
    component::require_operation(solver_table.solve != nullptr, "solve");
  }

  [[nodiscard]] std::string_view identity() const noexcept override { return spec_.provider_slot; }
  [[nodiscard]] std::uint64_t interface_version() const noexcept override { return 1; }
  [[nodiscard]] std::string_view collective_contract() const noexcept override {
    return collective_contract_;
  }
  [[nodiscard]] std::vector<std::string> capability_contracts() const override {
    std::vector<std::string> result{
        "pops.amr.external-field-solver.binary-coarse-fine-coverage@1",
        "pops.amr.external-field-solver.exact-component-pair@1",
        "pops.amr.external-field-solver.full-hierarchy-batch@1",
        "pops.amr.external-field-solver.host-serial@1",
        "pops.amr.external-field-solver.regrid-rematerialization@1",
        "pops.amr.external-field-solver.single-collective-solve@1",
    };
    if (spec_.component_pair_declares_mpi) {
      result.push_back("pops.amr.external-field-solver.declared-mpi-world@1");
      result.push_back("pops.amr.external-field-solver.mpi-distributed-coarse@1");
    }
    return result;
  }
  [[nodiscard]] AmrFieldSolverOptions default_field_options() const override {
    return external_options(spec_);
  }
  [[nodiscard]] std::optional<AmrFieldHierarchyPolicyAuthority> default_hierarchy_policy(
      std::string_view) const override {
    return std::nullopt;
  }
  [[nodiscard]] PreparedProviderSupport accepts_options(
      const AmrFieldSolverOptions& options) const noexcept override {
    return exact_external_options(options, spec_)
               ? PreparedProviderSupport::accept()
               : PreparedProviderSupport::reject(1,
                                                 "external field solver options differ from the "
                                                 "authenticated component request");
  }
  [[nodiscard]] PreparedProviderSupport supports(
      const AmrFieldSolverBuildRequest& request) const noexcept override {
    if (!exact_external_options(request.plan.solver_options, spec_))
      return PreparedProviderSupport::reject(10, "external field solver options are incompatible");
    if (request.use_contract_identity != "pops.amr.field-solver-use.named@1")
      return PreparedProviderSupport::reject(11,
                                             "external field solver supports named fields only");
    if (!exact_composite_policy(request.plan.hierarchy_policy))
      return PreparedProviderSupport::reject(
          12, "external field solver requires the composite hierarchy policy");
    if (request.hierarchy.nlev() < 1 ||
        request.hierarchy.ba.size() != request.hierarchy.dm.size() ||
        request.hierarchy.ba.size() != request.hierarchy.dx.size() ||
        request.hierarchy.ba.size() != request.hierarchy.dy.size() ||
        request.hierarchy.refinement_ratios.size() + 1 != request.hierarchy.ba.size())
      return PreparedProviderSupport::reject(13, "external field solver hierarchy is incomplete");
    if (std::any_of(request.hierarchy.refinement_ratios.begin(),
                    request.hierarchy.refinement_ratios.end(),
                    [](int ratio) { return ratio != kAmrRefRatio; }))
      return PreparedProviderSupport::reject(
          14, "external field solver currently requires ratio-2 AMR transitions");
    if (static_cast<bool>(request.active))
      return PreparedProviderSupport::reject(
          15, "external FieldTopology@2 bridge does not carry an active-region predicate");
    if (request.plan.has_reaction)
      return PreparedProviderSupport::reject(
          16, "external FieldSolver@2 has no reaction-coefficient carrier");
    if (request.plan.has_boundary_kernel)
      return PreparedProviderSupport::reject(
          17, "external FieldSolver@2 carries only an immutable boundary contract");
    if (request.plan.has_newton)
      return PreparedProviderSupport::reject(
          18, "external FieldSolver@2 has no shared nonlinear iterate/JVP protocol");
    if (!paired_periodic_boundary(request.boundary))
      return PreparedProviderSupport::reject(19, "periodic boundary faces are not paired");
    if (request.replicated_coarse && multi_rank_execution_)
      return PreparedProviderSupport::reject(
          20, "FieldSolver@2 has no MPI replicated-coarse ownership representation");
    return PreparedProviderSupport::accept();
  }
  [[nodiscard]] std::string expected_prepared_contract(
      const AmrFieldSolverBuildRequest& request) const override {
    ExactContractBuilder contract;
    contract.bytes(make_amr_field_solver_contract(identity(), request)).bytes(collective_contract_);
    return std::move(contract).release();
  }
  [[nodiscard]] std::unique_ptr<AmrPreparedFieldSolver> build(
      const AmrFieldSolverBuildRequest& request) const override {
    return std::make_unique<PreparedExternalAmrFieldSolver>(request, spec_, topology_, solver_,
                                                            expected_prepared_contract(request));
  }

 private:
  PreparedFieldSolverSpec spec_;
  std::shared_ptr<component::LoadedComponent> topology_;
  std::shared_ptr<component::LoadedComponent> solver_;
  std::string collective_contract_;
  bool multi_rank_execution_ = false;
};

}  // namespace

POPS_EXPORT std::shared_ptr<const AmrFieldSolverProvider> make_external_amr_field_solver_provider(
    runtime::field::PreparedFieldSolverSpec spec,
    std::shared_ptr<component::LoadedComponent> topology,
    std::shared_ptr<component::LoadedComponent> solver) {
  return std::make_shared<ExternalAmrFieldSolverProvider>(std::move(spec), std::move(topology),
                                                          std::move(solver));
}

}  // namespace pops
