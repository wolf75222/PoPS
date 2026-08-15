/// @file
/// @brief Compile-time-ranked adapters for prepared AMR Tagger, Reflux, and Clustering providers.

#pragma once

#include <pops/amr/tagging/clustering_provider.hpp>
#include <pops/core/foundation/allocator.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr/prepared_tagging_execution.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>
#include <pops/runtime/dynamic/prepared_execution_context.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace pops::runtime::amr {

/// Immutable RuntimeInstance authority for one native Tagger session.
struct PreparedTaggerComponentSpec {
  std::string component_id{};
  std::string manifest_identity{};
  std::string provider_identity{};
  std::string tagging_graph_identity{};
  std::string layout_identity{};
  std::string clock_identity{};
  std::uint32_t interface_version = 0;
  PopsTaggerExecutionModeV2 execution_mode = POPS_TAGGER_EXECUTION_NATIVE_BACKEND_V2;
  std::shared_ptr<const component::PreparedExecutionContextV1> execution{};
  std::string parameters_json{};
  std::string target_json{};
};

/// Authenticated, non-owning input presented to a local Tagger kernel.
template <int Dim, class MemorySpace>
struct PreparedTaggingRequest {
  const MultiFab<Dim, MemorySpace>* state = nullptr;
  ::pops::amr::hierarchy::LevelStateSpatialContract<Dim> source_level{};
  ::pops::amr::tagging::TagMaskBudget budget{};
  std::string_view runtime_spatial_contract{};
};

template <int Dim, class MemorySpace>
using PreparedTaggerKernel = PreparedProvider<::pops::amr::tagging::TagMask<Dim>(
    const PreparedTaggingRequest<Dim, MemorySpace>&)>;

/// Prepared native Tagger adapter. It only produces owner-local candidate masks; the canonical
/// AMR engine retains equality policy, persistent hysteresis, clustering and regrid publication.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class PreparedTaggerComponent {
 public:
  using Program = PreparedTaggingProgram<Dim>;
  using Field = PreparedTaggingField<Dim, MemorySpace>;
  using Candidates = PreparedTaggerCandidates<Dim>;
  using mask_type = ::pops::amr::tagging::TagMask<Dim>;

  PreparedTaggerComponent() = default;
  PreparedTaggerComponent(const PreparedTaggerComponent&) = delete;
  PreparedTaggerComponent& operator=(const PreparedTaggerComponent&) = delete;
  PreparedTaggerComponent(PreparedTaggerComponent&&) noexcept = default;
  PreparedTaggerComponent& operator=(PreparedTaggerComponent&&) noexcept = default;

  static PreparedTaggerComponent prepare(
      std::shared_ptr<component::LoadedComponent> component, PreparedTaggerComponentSpec spec,
      const Program& program, const std::vector<std::vector<Field>>& fields_by_level,
      const std::vector<::pops::amr::hierarchy::LevelLayout<Dim>>& layouts,
      const std::vector<PreparedTaggingExecutionBudget>& budgets, std::uint64_t topology_generation,
      std::uint32_t periodic_axes, const CommunicatorView& communicator) {
    std::optional<PreparedTaggerComponent> candidate;
    std::exception_ptr local_error;
    try {
      if (comm_active() && !communicator.active())
        throw std::invalid_argument(
            "native AMR Tagger requires an explicit active execution communicator");
      candidate.emplace(prepare_local_(std::move(component), std::move(spec), program,
                                       fields_by_level, layouts, budgets, topology_generation,
                                       periodic_axes, communicator));
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, communicator) != 0) {
      if (local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("native AMR Tagger preparation failed on another rank");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"native-amr-tagger", candidate->storage_->collective_contract}}, communicator))
      throw std::invalid_argument("native AMR Tagger prepared contracts differ between ranks");
    candidate->storage_->prepared = true;
    return std::move(*candidate);
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return storage_ != nullptr && storage_->prepared;
  }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return storage_ == nullptr ? std::string_view{} : storage_->collective_contract;
  }
  [[nodiscard]] std::uint64_t topology_generation() const noexcept {
    return storage_ == nullptr ? 0 : storage_->topology_generation;
  }

  const Candidates& execute(std::size_t level_index,
                            const ::pops::amr::hierarchy::LevelLayout<Dim>& layout,
                            const std::array<Real, Dim>& spacing, std::uint64_t topology_generation,
                            std::int64_t tick, double physical_time) {
    long preflight_failure = storage_ == nullptr || !storage_->prepared ||
                                     level_index >= storage_->levels.size() ||
                                     topology_generation != storage_->topology_generation ||
                                     tick < 0 || !std::isfinite(physical_time)
                                 ? 1L
                                 : 0L;
    for (int axis = 0; axis < Dim; ++axis)
      if (!(spacing[axis] > Real(0)) || !std::isfinite(static_cast<double>(spacing[axis])))
        preflight_failure = 1;
    if (preflight_failure == 0 && storage_->levels[level_index].identity != layout.exact_identity())
      preflight_failure = 1;
    if (all_reduce_max(preflight_failure, storage_->communicator) != 0)
      throw std::runtime_error("native AMR Tagger collective execution preflight failed");

    Level& level = storage_->levels[level_index];
    std::exception_ptr local_error;
    try {
      for (Patch& patch : level.patches)
        execute_patch_(*storage_, patch, layout.domain(), spacing, level_index, tick,
                       physical_time);
      device_fence();
      for (const Patch& patch : level.patches)
        for (const auto& output : patch.outputs)
          if (std::any_of(output.begin(), output.end(),
                          [](std::uint8_t value) { return value > 1u; }))
            throw std::runtime_error("native AMR Tagger returned a non-binary candidate mask");
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, storage_->communicator) != 0) {
      if (local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("native AMR Tagger failed on another rank");
    }

    if (level.replicated) {
      std::size_t offset = 0;
      for (const Patch& patch : level.patches) {
        for (std::size_t point = 0; point < patch.outputs[0].size(); ++point) {
          std::uint8_t packed = 0;
          for (std::size_t output = 0; output < patch.outputs.size(); ++output) {
            packed = static_cast<std::uint8_t>(
                packed | static_cast<std::uint8_t>(patch.outputs[output][point] << output));
          }
          level.replica_min[offset] = static_cast<char>(packed);
          level.replica_max[offset] = level.replica_min[offset];
          ++offset;
        }
      }
      all_reduce_min_inplace(level.replica_min.data(), level.replica_min.size(),
                             storage_->communicator);
      all_reduce_max_inplace(level.replica_max.data(), level.replica_max.size(),
                             storage_->communicator);
      if (!std::equal(level.replica_min.begin(), level.replica_min.end(),
                      level.replica_max.begin()))
        throw std::runtime_error(
            "native AMR Tagger replicated fields produced different masks between ranks");
    }

    for (const Patch& patch : level.patches)
      for_each_host_index_(patch.box, [&](const Index<Dim>& index, std::size_t ordinal) {
        level.candidates.refine.set(patch.global_patch, index, patch.outputs[0][ordinal] != 0);
        level.candidates.coarsen.set(patch.global_patch, index, patch.outputs[1][ordinal] != 0);
        level.candidates.refine_equalities.set(patch.global_patch, index,
                                               patch.outputs[2][ordinal] != 0);
        level.candidates.coarsen_equalities.set(patch.global_patch, index,
                                                patch.outputs[3][ordinal] != 0);
      });
    return level.candidates;
  }

 private:
  template <class T>
  using DeviceVector = std::vector<T, fab_allocator<T>>;
  template <class T>
  using CommunicationVector = std::vector<T, comm_allocator<T>>;

  struct Patch {
    Box<Dim> box{};
    std::size_t global_patch = 0;
    std::string patch_identity{};
    std::vector<std::string> state_identities{};
    std::vector<PopsQualifiedConstFieldV1> states{};
    std::array<DeviceVector<std::uint8_t>, 4> outputs{};
  };

  struct Level {
    ::pops::amr::hierarchy::LevelLayoutIdentity<Dim> identity{};
    std::vector<Patch> patches{};
    bool replicated = false;
    CommunicationVector<char> replica_min{};
    CommunicationVector<char> replica_max{};
    Candidates candidates;

    Level(const ::pops::amr::hierarchy::LevelLayout<Dim>& layout, const Index<Dim>& local_rank,
          const PreparedTaggingExecutionBudget& budget, std::size_t local_cells)
        : identity(layout.exact_identity()),
          replicated(layout.distribution().replicated()),
          replica_min(replicated ? local_cells : 0, char{0}),
          replica_max(replicated ? local_cells : 0, char{0}),
          candidates{mask_type(layout, local_rank, budget.candidate_mask),
                     mask_type(layout, local_rank, budget.candidate_mask),
                     mask_type(layout, local_rank, budget.candidate_mask),
                     mask_type(layout, local_rank, budget.candidate_mask)} {}
  };

  struct Storage {
    // LoadedComponent must outlive the fresh PreparedState that calls into its destroy function.
    std::shared_ptr<component::LoadedComponent> component{};
    component::LoadedComponent::PreparedState state{};
    PreparedTaggerComponentSpec spec{};
    Program program{};
    std::vector<PopsTaggingLeafV1> abi_leaves{};
    std::vector<std::array<PopsTaggingAxisStencilV1, Dim>> abi_axes{};
    std::vector<PopsTaggingStencilV1> abi_stencils{};
    std::vector<Level> levels{};
    std::uint64_t topology_generation = 0;
    std::uint32_t periodic_axes = 0;
    CommunicatorView communicator{};
    std::string collective_contract{};
    bool prepared = false;
  };

  static PreparedTaggerComponent prepare_local_(
      std::shared_ptr<component::LoadedComponent> component, PreparedTaggerComponentSpec spec,
      const Program& program, const std::vector<std::vector<Field>>& fields_by_level,
      const std::vector<::pops::amr::hierarchy::LevelLayout<Dim>>& layouts,
      const std::vector<PreparedTaggingExecutionBudget>& budgets, std::uint64_t topology_generation,
      std::uint32_t periodic_axes, const CommunicatorView& communicator) {
    if (!component || !spec.execution || spec.component_id.empty() ||
        spec.manifest_identity.empty() || spec.provider_identity.empty() ||
        spec.tagging_graph_identity.empty() || spec.layout_identity.empty() ||
        spec.clock_identity.empty() || spec.interface_version != 2 || !program.prepared ||
        program.clock_identity != spec.clock_identity || program.leaves.empty() ||
        fields_by_level.empty() || fields_by_level.size() != layouts.size() ||
        layouts.size() != budgets.size())
      throw std::invalid_argument("native AMR Tagger preparation authority is incomplete");
    const PopsComponentApiV1& component_api = component->api();
    if (component_api.component_id == nullptr || component_api.manifest_identity == nullptr ||
        spec.component_id != component_api.component_id ||
        spec.manifest_identity != component_api.manifest_identity)
      throw std::invalid_argument("native AMR Tagger component identity changed after load");
    (void)component->table<PopsTaggerApiV2>(POPS_NATIVE_INTERFACE_TAGGER_V2,
                                            spec.interface_version);
    const auto execution = spec.execution->without_collective_authority();
    if (!execution_memory_matches_(execution.view().memory_space) ||
        (spec.execution_mode == POPS_TAGGER_EXECUTION_HOST_V2 &&
         execution.view().memory_space != POPS_MEMORY_SPACE_HOST_V1) ||
        (spec.execution_mode != POPS_TAGGER_EXECUTION_HOST_V2 &&
         spec.execution_mode != POPS_TAGGER_EXECUTION_NATIVE_BACKEND_V2))
      throw std::invalid_argument("native AMR Tagger execution or memory authority is inexact");

    auto storage = std::make_unique<Storage>();
    storage->component = std::move(component);
    storage->spec = std::move(spec);
    storage->spec.execution =
        std::make_shared<const component::PreparedExecutionContextV1>(execution);
    storage->program = program;
    storage->topology_generation = topology_generation;
    storage->periodic_axes = periodic_axes;
    storage->communicator = communicator;
    storage->state = storage->component->prepare_fresh_state(
        POPS_NATIVE_INTERFACE_TAGGER_V2, storage->spec.interface_version,
        storage->spec.execution->view(), storage->spec.parameters_json, storage->spec.target_json);
    prepare_program_views_(*storage);

    ExactContractBuilder contract;
    contract.text("pops.amr.native-tagger-session")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(storage->spec.component_id)
        .text(storage->spec.manifest_identity)
        .text(storage->spec.provider_identity)
        .text(storage->spec.tagging_graph_identity)
        .text(storage->spec.layout_identity)
        .text(storage->spec.clock_identity)
        .scalar(storage->spec.interface_version)
        .scalar(storage->spec.execution_mode)
        .text(storage->spec.execution->identity())
        .text(storage->spec.parameters_json)
        .text(storage->spec.target_json)
        .scalar(topology_generation)
        .scalar(periodic_axes)
        .scalar(static_cast<std::uint64_t>(layouts.size()));

    std::vector<std::vector<tagging_detail::PreparedTaggingFieldContract<Dim>>> field_contracts;
    field_contracts.reserve(layouts.size());
    storage->levels.reserve(layouts.size());
    for (std::size_t level_index = 0; level_index < layouts.size(); ++level_index) {
      const auto& layout = layouts[level_index];
      const auto& fields = fields_by_level[level_index];
      if (fields.empty() || fields.front().values == nullptr)
        throw std::invalid_argument("native AMR Tagger level has no qualified field authority");
      const auto& reference = *fields.front().values;
      if (reference.layout() != layout.patches() ||
          reference.distribution() != layout.distribution())
        throw std::invalid_argument("native AMR Tagger field layout is unauthenticated");
      const std::size_t rank_count = layout.distribution().rank_space().size();
      if (communicator.size() < 1 || communicator.rank() < 0 ||
          static_cast<std::size_t>(communicator.size()) != rank_count ||
          layout.distribution().rank_space().linear_rank(reference.local_rank()) !=
              static_cast<std::size_t>(communicator.rank()))
        throw std::invalid_argument(
            "native AMR Tagger rank coordinate differs from its communicator rank space");
      std::vector<tagging_detail::PreparedTaggingFieldContract<Dim>> level_field_contracts;
      level_field_contracts.reserve(fields.size());
      for (const Field& field : fields) {
        if (field.values == nullptr || field.qualified_identity.empty() ||
            field.values->layout() != reference.layout() ||
            field.values->distribution() != reference.distribution() ||
            field.values->local_rank() != reference.local_rank() ||
            field.values->local_global_indices() != reference.local_global_indices())
          throw std::invalid_argument(
              "native AMR Tagger fields do not share one exact qualified layout");
        if (std::any_of(level_field_contracts.begin(), level_field_contracts.end(),
                        [&](const auto& existing) {
                          return existing.qualified_identity == field.qualified_identity;
                        }))
          throw std::invalid_argument("native AMR Tagger field identity is not unique");
        level_field_contracts.push_back(
            {field.qualified_identity, field.values->ncomp(), field.values->ghosts()});
      }
      field_contracts.push_back(std::move(level_field_contracts));
      std::size_t local_cells = 0;
      for (std::size_t local = 0; local < reference.local_size(); ++local)
        local_cells = checked_sum_(local_cells, checked_cell_count_(reference.box(local)));
      if (local_cells > budgets[level_index].scratch_bytes)
        throw std::length_error("native AMR Tagger exceeds its explicit scratch budget");
      const std::size_t consensus_bytes =
          layout.distribution().replicated() ? checked_product_(local_cells, 2u) : 0u;
      if (consensus_bytes > budgets[level_index].replicated_consensus_bytes)
        throw std::length_error(
            "native AMR Tagger exceeds its explicit replicated-consensus budget");
      storage->levels.emplace_back(layout, reference.local_rank(), budgets[level_index],
                                   local_cells);
      Level& level = storage->levels.back();
      level.patches.reserve(reference.local_size());
      for (std::size_t local = 0; local < reference.local_size(); ++local) {
        const std::size_t global = reference.global_index(local);
        level.patches.emplace_back();
        Patch& patch = level.patches.back();
        patch.box = reference.box(local);
        patch.global_patch = global;
        if (!layout.domain().contains(patch.box) || patch.box != layout.patches()[global])
          throw std::invalid_argument(
              "native AMR Tagger local patch differs from its authenticated level");
        patch.patch_identity = storage->spec.layout_identity + "/level/" +
                               std::to_string(level_index) + "/patch/" + std::to_string(global);
        patch.state_identities.reserve(fields.size());
        for (const Field& field : fields)
          patch.state_identities.push_back(field.qualified_identity);
        patch.states.reserve(fields.size());
        for (std::size_t field_index = 0; field_index < fields.size(); ++field_index)
          patch.states.push_back(qualified_field_view_(
              *fields[field_index].values, local, patch.state_identities[field_index],
              storage->spec.layout_identity, patch.patch_identity,
              storage->spec.execution->view().memory_space));
        const std::size_t cells = checked_cell_count_(patch.box);
        for (auto& output : patch.outputs)
          output.assign(cells, std::uint8_t{0});
      }
    }
    contract.bytes(tagging_detail::exact_program_contract(program, field_contracts, layouts,
                                                          topology_generation));
    storage->collective_contract = std::move(contract).release();
    PreparedTaggerComponent result;
    result.storage_ = std::move(storage);
    return result;
  }

  static void prepare_program_views_(Storage& storage) {
    storage.abi_leaves.reserve(storage.program.leaves.size());
    for (const auto& leaf : storage.program.leaves)
      storage.abi_leaves.push_back({sizeof(PopsTaggingLeafV1), leaf.state_index, leaf.component,
                                    leaf.opcode, leaf.threshold, leaf.stencil_index});
    storage.abi_axes.resize(storage.program.stencils.size());
    storage.abi_stencils.reserve(storage.program.stencils.size());
    for (std::size_t stencil_index = 0; stencil_index < storage.program.stencils.size();
         ++stencil_index) {
      const auto& stencil = storage.program.stencils[stencil_index];
      auto& axes = storage.abi_axes[stencil_index];
      for (int axis = 0; axis < Dim; ++axis) {
        const auto& source = stencil.axes[static_cast<std::size_t>(axis)];
        axes[static_cast<std::size_t>(axis)] = {sizeof(PopsTaggingAxisStencilV1),
                                                source.axis,
                                                source.derivative_order,
                                                source.formal_order,
                                                source.ghost_lower,
                                                source.ghost_upper,
                                                source.offsets.size(),
                                                source.offsets.data(),
                                                source.coefficients.data()};
      }
      storage.abi_stencils.push_back({sizeof(PopsTaggingStencilV1), stencil.identity.c_str(),
                                      stencil.route.c_str(), stencil.norm.c_str(),
                                      stencil.scale.c_str(), stencil.boundary_mode.c_str(), Dim,
                                      static_cast<std::size_t>(Dim), axes.data()});
    }
  }

  static PopsTaggingProgramV1 program_view_(const Storage& storage) {
    return {sizeof(PopsTaggingProgramV1),       storage.program.provider_identity.c_str(),
            storage.abi_stencils.size(),        storage.abi_stencils.data(),
            storage.abi_leaves.size(),          storage.abi_leaves.data(),
            storage.program.refine_ops.size(),  storage.program.refine_ops.data(),
            storage.program.refine_args.data(), storage.program.coarsen_ops.size(),
            storage.program.coarsen_ops.data(), storage.program.coarsen_args.data(),
            storage.program.minimum_cycles,     storage.program.equality_policy,
            storage.program.conflict_policy,    storage.program.non_finite_policy};
  }

  static PopsQualifiedConstFieldV1 qualified_field_view_(const MultiFab<Dim, MemorySpace>& field,
                                                         std::size_t local,
                                                         const std::string& qualified_identity,
                                                         const std::string& layout_identity,
                                                         const std::string& patch_identity,
                                                         PopsMemorySpaceV1 memory_space) {
    const auto view = field.fab(local).view();
    PopsConstFieldViewV1 values{};
    values.struct_size = sizeof(PopsConstFieldViewV1);
    values.data = view.data;
    values.dimension = Dim;
    for (int axis = 0; axis < 3; ++axis) {
      values.extents[axis] = 1;
      values.axis_strides[axis] = 0;
      values.ghost_lower[axis] = 0;
      values.ghost_upper[axis] = 0;
    }
    for (int axis = 0; axis < Dim; ++axis) {
      values.extents[axis] = static_cast<std::size_t>(view.extents[axis]);
      values.axis_strides[axis] = view.strides[axis];
      values.ghost_lower[axis] = static_cast<std::size_t>(field.ghosts()[axis]);
      values.ghost_upper[axis] = static_cast<std::size_t>(field.ghosts()[axis]);
    }
    values.component_count = static_cast<std::size_t>(view.ncomp);
    values.component_stride = view.component_stride;
    values.centering = POPS_FIELD_CENTERING_CELL_V1;
    values.scalar_type = POPS_SCALAR_FLOAT64_V1;
    values.memory_space = memory_space;
    values.layout_identity = layout_identity.c_str();
    values.patch_identity = patch_identity.c_str();
    values.ownership = POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1;
    return {sizeof(PopsQualifiedConstFieldV1), 1, qualified_identity.c_str(), values};
  }

  static PopsTaggerMaskViewV2 mask_view_(DeviceVector<std::uint8_t>& values,
                                         PopsMemorySpaceV1 memory_space) {
    return {sizeof(PopsTaggerMaskViewV2), values.data(), values.size(), memory_space,
            POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
  }

  static void execute_patch_(Storage& storage, Patch& patch, const Box<Dim>& domain,
                             const std::array<Real, Dim>& spacing, std::size_t level,
                             std::int64_t tick, double physical_time) {
    for (auto& output : patch.outputs)
      std::fill(output.begin(), output.end(), std::uint8_t{0});
    PopsTaggerRequestV2 request{};
    request.struct_size = sizeof(PopsTaggerRequestV2);
    request.execution_mode = storage.spec.execution_mode;
    request.collective_scope = POPS_TAGGER_COLLECTIVE_NONE_V2;
    request.state_count = patch.states.size();
    request.states = patch.states.data();
    request.program = program_view_(storage);
    for (int axis = 0; axis < 3; ++axis) {
      request.patch_lower[axis] = 0;
      request.domain_lower[axis] = 0;
      request.domain_upper[axis] = 0;
      request.cell_size[axis] = 0.0;
    }
    for (int axis = 0; axis < Dim; ++axis) {
      request.patch_lower[axis] = patch.box.lo[axis];
      request.domain_lower[axis] = domain.lo[axis];
      request.domain_upper[axis] = domain.hi[axis];
      request.cell_size[axis] = static_cast<double>(spacing[axis]);
    }
    request.periodic_axes = storage.periodic_axes;
    const PopsMemorySpaceV1 memory_space = storage.spec.execution->view().memory_space;
    request.refine_candidates = mask_view_(patch.outputs[0], memory_space);
    request.coarsen_candidates = mask_view_(patch.outputs[1], memory_space);
    request.refine_equalities = mask_view_(patch.outputs[2], memory_space);
    request.coarsen_equalities = mask_view_(patch.outputs[3], memory_space);
    request.logical_time = {sizeof(PopsLogicalTimeV1),
                            storage.spec.clock_identity.c_str(),
                            tick,
                            static_cast<std::int32_t>(level),
                            0,
                            0,
                            0,
                            1,
                            0.0,
                            physical_time};
    request.execution = storage.spec.execution->view();
    PopsComponentStatusV1 status = component::unwritten_component_status();
    const auto& api = storage.component->template table<PopsTaggerApiV2>(
        POPS_NATIVE_INTERFACE_TAGGER_V2, storage.spec.interface_version);
    const int code = component::tag_batch(api, storage.state.get(), request, status);
    if (!component::component_status_is_well_formed(status) || code != 0 || status.code != 0 ||
        status.action != POPS_COMPONENT_CONTINUE_V1)
      throw std::runtime_error(status.reason == nullptr ? "native AMR Tagger failed"
                                                        : status.reason);
  }

  static std::size_t checked_cell_count_(const Box<Dim>& box) {
    const std::int64_t cells = box.numPts();
    if (cells < 0 || static_cast<std::uint64_t>(cells) >
                         static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
      throw std::length_error("native AMR Tagger patch cell count exceeds size_t");
    return static_cast<std::size_t>(cells);
  }

  static std::size_t checked_sum_(std::size_t left, std::size_t right) {
    if (right > std::numeric_limits<std::size_t>::max() - left)
      throw std::length_error("native AMR Tagger scratch count exceeds size_t");
    return left + right;
  }

  static std::size_t checked_product_(std::size_t left, std::size_t right) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
      throw std::length_error("native AMR Tagger consensus count exceeds size_t");
    return left * right;
  }

  template <class Function>
  static void for_each_host_index_(const Box<Dim>& box, Function&& function) {
    const std::size_t cells = checked_cell_count_(box);
    for (std::size_t ordinal = 0; ordinal < cells; ++ordinal) {
      Index<Dim> index{};
      std::size_t quotient = ordinal;
      for (int axis = 0; axis < Dim; ++axis) {
        const std::size_t length = static_cast<std::size_t>(box.length(axis));
        index[axis] = static_cast<int>(static_cast<std::int64_t>(box.lo[axis]) +
                                       static_cast<std::int64_t>(quotient % length));
        quotient /= length;
      }
      function(index, ordinal);
    }
  }

  static constexpr bool execution_memory_matches_(PopsMemorySpaceV1 claimed) {
    if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>)
      return claimed == POPS_MEMORY_SPACE_HOST_V1;
    if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible)
      return claimed == POPS_MEMORY_SPACE_MANAGED_V1;
    return claimed == POPS_MEMORY_SPACE_DEVICE_V1;
  }

  std::unique_ptr<Storage> storage_{};
};

/// Ranked wrapper around the canonical clustering authority.
template <int Dim>
class PreparedClusteringComponent {
 public:
  using provider_type = ::pops::amr::tagging::ClusterProvider<Dim>;
  using mask_type = ::pops::amr::tagging::TagMask<Dim>;
  using options_type = ::pops::amr::tagging::ClusterOptions<Dim>;
  using result_type = ::pops::amr::tagging::ClusterResult<Dim>;

  explicit PreparedClusteringComponent(std::shared_ptr<const provider_type> provider)
      : provider_(std::move(provider)) {
    if (!provider_ || provider_->provider_identity().empty())
      throw std::invalid_argument("prepared ND Clustering component requires an identity");
    ExactContractBuilder contract;
    contract.text("pops.amr.prepared-clustering")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(provider_->provider_identity());
    collective_contract_ = std::move(contract).release();
  }

  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(provider_); }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return collective_contract_;
  }

  result_type cluster(std::span<const mask_type> shards, const options_type& options) const {
    result_type result = provider_->cluster(shards, options);
    if (std::string_view(result.identity.provider) != provider_->provider_identity() ||
        result.identity.boxes != result.boxes.boxes())
      throw std::runtime_error(
          "prepared ND Clustering provider returned an unauthenticated result");
    return result;
  }

 private:
  std::shared_ptr<const provider_type> provider_{};
  std::string collective_contract_{};
};

/// Input to a local Reflux correction kernel after the AMR runtime authenticated and reconciled
/// the complete metric-time face product.
template <int Dim, class Payload>
struct PreparedRefluxRequest {
  const ::pops::amr::reflux::CoarseFaceRefluxKey<Dim>* key = nullptr;
  const ::pops::amr::reflux::MetricFaceReflux<Payload>* reconciliation = nullptr;
  double coarse_cell_measure = 0.0;
  ::pops::amr::reflux::CoarseCellFaceSide side = ::pops::amr::reflux::CoarseCellFaceSide::Lower;
};

template <int Dim, class Payload>
using PreparedRefluxKernel = PreparedProvider<Payload(const PreparedRefluxRequest<Dim, Payload>&)>;

/// Reflux adapter that delegates every topology, clock, and metric decision to AmrRuntime first.
template <int Dim, class Payload>
class PreparedRefluxComponent {
 public:
  explicit PreparedRefluxComponent(PreparedRefluxKernel<Dim, Payload> kernel)
      : kernel_(std::move(kernel)) {
    if (!kernel_)
      throw std::invalid_argument("prepared ND Reflux component requires a kernel");
    ExactContractBuilder contract;
    contract.text("pops.amr.prepared-reflux")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .bytes(kernel_.collective_contract());
    collective_contract_ = std::move(contract).release();
  }

  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(kernel_); }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return collective_contract_;
  }

  template <class MemorySpace, class Axpy>
  Payload correction(const AmrRuntime<Dim, MemorySpace>& runtime,
                     const ::pops::amr::reflux::TransactionalFaceFluxLedger<Dim, Payload>& ledger,
                     const ::pops::amr::reflux::CoarseFaceRefluxKey<Dim>& key,
                     std::string_view state_identity,
                     const ::pops::amr::reflux::FaceRefinementMapping<Dim>& mapping,
                     const ::pops::amr::reflux::MetricRefluxBudget& budget,
                     double coarse_cell_measure, ::pops::amr::reflux::CoarseCellFaceSide side,
                     Axpy&& axpy) const {
    if (!(coarse_cell_measure > 0.0) || !std::isfinite(coarse_cell_measure))
      throw std::invalid_argument(
          "prepared ND Reflux correction requires a finite positive cell measure");
    const auto reconciliation = runtime.reconcile_reflux(ledger, key, state_identity, mapping,
                                                         budget, std::forward<Axpy>(axpy));
    return kernel_(
        PreparedRefluxRequest<Dim, Payload>{&key, &reconciliation, coarse_cell_measure, side});
  }

 private:
  PreparedRefluxKernel<Dim, Payload> kernel_{};
  std::string collective_contract_{};
};

}  // namespace pops::runtime::amr
