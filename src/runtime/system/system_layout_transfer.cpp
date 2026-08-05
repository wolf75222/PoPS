#include "system_impl.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {
namespace {

constexpr std::string_view kCellAverageRepresentation = "pops://representations/cell-average@1";
constexpr std::string_view kBeforeStepSynchronization = "pops://synchronization/before-step@1";

void require_text(const std::string& value, const char* where) {
  if (value.empty())
    throw std::invalid_argument(std::string("prepared System layout transfer requires ") + where);
}

void append_u64(std::string& bytes, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8)
    bytes.push_back(static_cast<char>((value >> shift) & 0xffu));
}

void append_i32(std::string& bytes, std::int32_t value) {
  append_u64(bytes, static_cast<std::uint32_t>(value));
}

void append_text(std::string& bytes, std::string_view value) {
  append_u64(bytes, static_cast<std::uint64_t>(value.size()));
  bytes.append(value.data(), value.size());
}

void append_double(std::string& bytes, double value) {
  append_u64(bytes, std::bit_cast<std::uint64_t>(value));
}

template <int Dim>
void append_layout(std::string& bytes, const mesh::BoxArray<Dim>& boxes,
                   const mesh::Distribution<Dim>& owners) {
  append_u64(bytes, static_cast<std::uint64_t>(boxes.size()));
  for (const Box<Dim>& box : boxes.boxes())
    for (int axis = 0; axis < Dim; ++axis) {
      append_i32(bytes, box.lo[axis]);
      append_i32(bytes, box.hi[axis]);
    }
  append_u64(bytes, static_cast<std::uint64_t>(owners.owners().size()));
  for (const Index<Dim>& owner : owners.owners())
    for (int axis = 0; axis < Dim; ++axis)
      append_i32(bytes, owner[axis]);
}

PopsExecutionContextV1 execution_view(const SystemLayoutTransferExecution& execution) noexcept {
  return {sizeof(PopsExecutionContextV1),
          execution.context_version,
          execution.execution_identity.c_str(),
          static_cast<PopsMemorySpaceV1>(execution.memory_space),
          execution.backend_identity.c_str(),
          execution.device_identity.c_str(),
          static_cast<PopsScalarTypeV1>(execution.scalar_type),
          static_cast<PopsPrecisionV1>(execution.storage_precision),
          static_cast<PopsPrecisionV1>(execution.compute_precision),
          static_cast<PopsPrecisionV1>(execution.accumulation_precision),
          static_cast<PopsPrecisionV1>(execution.reduction_precision),
          execution.stream_handle,
          execution.stream_identity.c_str(),
          execution.communicator_f_handle,
          execution.communicator_datatype_f_handle,
          execution.communicator_identity.c_str(),
          execution.communicator_datatype_identity.c_str()};
}

CommunicatorView resolve_execution_communicator(const SystemLayoutTransferExecution& execution,
                                                const CommunicatorView& field_rank_space) {
  const PopsExecutionContextV1 view = execution_view(execution);
  component::validate_execution_context(view);
  if (execution.communicator_identity == "serial") {
    if (field_rank_space.active())
      throw std::invalid_argument(
          "serial layout-transfer execution requires native MPI to be inactive");
    return CommunicatorView{};
  }
  if (execution.communicator_identity == POPS_EXECUTION_NONCOLLECTIVE_IDENTITY_V1)
    throw std::invalid_argument(
        "prepared System layout transfer requires collective execution authority");
#ifdef POPS_HAS_MPI
  if (!field_rank_space.active())
    throw std::invalid_argument(
        "collective layout-transfer execution requires initialized native MPI");
  const MPI_Comm communicator =
      MPI_Comm_f2c(static_cast<MPI_Fint>(execution.communicator_f_handle));
  if (communicator == MPI_COMM_NULL ||
      MPI_Type_f2c(static_cast<MPI_Fint>(execution.communicator_datatype_f_handle)) != MPI_DOUBLE ||
      execution.communicator_datatype_identity != "MPI_DOUBLE")
    throw std::invalid_argument(
        "layout-transfer execution handles do not identify a live communicator/MPI_DOUBLE "
        "authority");
  int relation = MPI_UNEQUAL;
  ::pops::detail::require_mpi_success(
      MPI_Comm_compare(communicator, field_rank_space.native_handle(), &relation),
      "MPI_Comm_compare(layout-transfer field rank space)");
  if (relation != MPI_IDENT && relation != MPI_CONGRUENT)
    throw std::invalid_argument(
        "layout-transfer execution communicator must preserve the field rank space");
  return CommunicatorView{communicator};
#else
  (void)field_rank_space;
  throw std::invalid_argument(
      "collective layout-transfer execution requires an MPI-enabled PoPS build");
#endif
}

template <class Function>
void collectively_validate(const CommunicatorView& communicator, const char* where,
                           Function&& function) {
  std::exception_ptr failure;
  try {
    std::forward<Function>(function)();
  } catch (...) {
    failure = std::current_exception();
  }
  const long failures = all_reduce_sum(failure ? 1L : 0L, communicator);
  if (failures == 0)
    return;
  if (communicator.size() == 1 && failure)
    std::rethrow_exception(failure);
  throw std::runtime_error(std::string(where) + " failed on at least one MPI rank");
}

int checked_index(std::int64_t value, const char* where) {
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    throw std::overflow_error(std::string(where) + " exceeds the native index range");
  return static_cast<int>(value);
}

template <int Dim>
mesh::BoxArray<Dim> source_carrier_boxes(const mesh::BoxArray<Dim>& target_boxes,
                                         const Box<Dim>& source_domain,
                                         const Box<Dim>& target_domain,
                                         const std::array<std::int32_t, Dim>& ratio) {
  std::vector<Box<Dim>> boxes;
  boxes.reserve(target_boxes.size());
  for (const Box<Dim>& target : target_boxes.boxes()) {
    Box<Dim> source;
    for (int axis = 0; axis < Dim; ++axis) {
      if (ratio[static_cast<std::size_t>(axis)] <= 0)
        throw std::invalid_argument("prepared System transfer ratios must be positive");
      const std::int64_t axis_ratio = ratio[static_cast<std::size_t>(axis)];
      source.lo[axis] = checked_index(
          source_domain.lo[axis] +
              (static_cast<std::int64_t>(target.lo[axis]) - target_domain.lo[axis]) * axis_ratio,
          "source carrier lower bound");
      source.hi[axis] = checked_index(
          source_domain.lo[axis] +
              (static_cast<std::int64_t>(target.hi[axis]) - target_domain.lo[axis] + 1) *
                  axis_ratio -
              1,
          "source carrier upper bound");
    }
    boxes.push_back(source);
  }
  return mesh::BoxArray<Dim>(std::move(boxes));
}

template <int Dim>
mesh::Distribution<Dim> rebind_distribution(const mesh::BoxArray<Dim>& layout,
                                            const mesh::Distribution<Dim>& model) {
  if (layout.size() != model.box_count())
    throw std::invalid_argument(
        "layout-transfer carrier and target ownership cardinalities differ");
  if (model.replicated())
    return mesh::Distribution<Dim>::replicated(layout, model.rank_space());
  return mesh::Distribution<Dim>::partitioned(layout, model.rank_space(), model.owners());
}

inline std::size_t checked_product(std::size_t left, std::size_t right, const char* where) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(where);
  return left * right;
}

template <int Dim, class DestinationMemorySpace, class SourceMemorySpace>
CopySchedule<Dim> prepare_exact_copy_schedule(
    const MultiFab<Dim, DestinationMemorySpace>& destination,
    const MultiFab<Dim, SourceMemorySpace>& source) {
  const std::size_t pairs = checked_product(destination.layout().size(), source.layout().size(),
                                            "layout-transfer copy schedule exceeds size_t");
  const auto overlap_pairs = [](std::size_t count) {
    return count < 2 ? std::size_t{0}
                     : checked_product(count, count - 1,
                                       "layout-transfer overlap budget exceeds size_t") /
                           2;
  };
  return prepare_copy_schedule(
      destination, source,
      CopyScheduleBudget{pairs, pairs, overlap_pairs(destination.layout().size()),
                         overlap_pairs(source.layout().size())});
}

template <int Dim>
std::uint64_t checked_elements(const Box<Dim>& box, int components) {
  const std::int64_t cells = box.numPts();
  if (cells <= 0 || components <= 0 ||
      static_cast<std::uint64_t>(cells) >
          std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(components))
    throw std::overflow_error("layout-transfer field element count exceeds uint64 capacity");
  return static_cast<std::uint64_t>(cells) * static_cast<std::uint64_t>(components);
}

std::uint64_t collective_elements(std::uint64_t local, const CommunicatorView& communicator) {
  const auto ranks = static_cast<std::uint64_t>(communicator.size());
  const std::uint64_t per_rank_limit =
      static_cast<std::uint64_t>(std::numeric_limits<long>::max()) / ranks;
  const long invalid = all_reduce_max(local > per_rank_limit ? 1L : 0L, communicator);
  if (invalid != 0)
    throw std::overflow_error("layout-transfer global element count exceeds MPI long capacity");
  const long global = all_reduce_sum(static_cast<long>(local), communicator);
  return static_cast<std::uint64_t>(global);
}

template <class Value, int Dim>
Value* valid_data(FieldView<Value, Dim> view, const Box<Dim>& valid) {
  std::int64_t offset = 0;
  for (int axis = 0; axis < Dim; ++axis)
    offset += (static_cast<std::int64_t>(valid.lo[axis]) - view.origin[axis]) * view.strides[axis];
  return view.data + offset;
}

template <class AbiView, class Value, int Dim>
AbiView component_field_view(FieldView<Value, Dim> view, const Box<Dim>& valid, int components,
                             PopsMemorySpaceV1 memory_space, const char* layout_identity,
                             const char* patch_identity) {
  AbiView result{};
  result.struct_size = sizeof(AbiView);
  result.data = valid_data(view, valid);
  result.dimension = Dim;
  for (int axis = 0; axis < 3; ++axis) {
    result.extents[axis] = 1;
    result.axis_strides[axis] = 0;
    result.ghost_lower[axis] = 0;
    result.ghost_upper[axis] = 0;
  }
  for (int axis = 0; axis < Dim; ++axis) {
    result.extents[axis] = static_cast<std::size_t>(valid.length(axis));
    result.axis_strides[axis] = static_cast<std::ptrdiff_t>(view.strides[axis]);
  }
  result.component_count = static_cast<std::size_t>(components);
  result.component_stride = static_cast<std::ptrdiff_t>(view.component_stride);
  result.centering = POPS_FIELD_CENTERING_CELL_V1;
  result.centering_axes = 0;
  result.scalar_type = POPS_SCALAR_FLOAT64_V1;
  result.memory_space = memory_space;
  result.layout_identity = layout_identity;
  result.patch_identity = patch_identity;
  result.ownership = POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1;
  return result;
}

}  // namespace

template <int Dim>
struct PreparedSystemLayoutTransfer<Dim>::Impl {
  using system_type = System<Dim>;
  using system_impl_type = typename system_type::Impl;
  using field_type = MultiFab<Dim>;

  system_type* source_owner = nullptr;
  system_type* target_owner = nullptr;
  system_impl_type* source = nullptr;
  system_impl_type* target = nullptr;
  std::shared_ptr<component::LoadedComponent> component_handle;
  component::LoadedComponent::PreparedState component_state;
  const PopsTransferApiV1* transfer_api = nullptr;
  SystemLayoutTransferSpec<Dim> spec;
  SystemLayoutTransferExecution execution;
  PopsExecutionContextV1 execution_abi{};
  CommunicatorView communicator;
  int source_block_index = -1;
  int target_block_index = -1;
  int components = 0;
  field_type source_snapshot;
  std::optional<CopySchedule<Dim>> source_copy_schedule;
  std::vector<std::string> source_patch_identities;
  std::vector<std::string> target_patch_identities;
  std::uint64_t active_generation = 0;
  std::uint64_t last_generation = 0;
  std::uint64_t captured_attempt = 0;
  bool active = false;
  bool applied = false;

  Impl(system_type& source_system, system_type& target_system,
       std::shared_ptr<component::LoadedComponent> loaded,
       SystemLayoutTransferSpec<Dim> transfer_spec,
       SystemLayoutTransferExecution transfer_execution,
       const CommunicatorView& transfer_communicator)
      : source_owner(&source_system),
        target_owner(&target_system),
        source(source_system.p_.get()),
        target(target_system.p_.get()),
        component_handle(std::move(loaded)),
        spec(std::move(transfer_spec)),
        execution(std::move(transfer_execution)),
        execution_abi(execution_view(execution)),
        communicator(transfer_communicator) {
    validate_static_contract();
    source_block_index = source->blocks_.index(spec.source_block);
    target_block_index = target->blocks_.index(spec.target_block);
    components = source->sp[static_cast<std::size_t>(source_block_index)].ncomp;
    const mesh::BoxArray<Dim> carrier =
        source_carrier_boxes<Dim>(target->ba, source->dom, target->dom, spec.refinement_ratio);
    source_snapshot = field_type(carrier, rebind_distribution(carrier, target->dm),
                                 target->local_rank, components, Extent<Dim>{});
    source_copy_schedule.emplace(prepare_exact_copy_schedule(source_snapshot, source_state()));
    source_copy_schedule->require_local_execution();
    source_patch_identities.reserve(carrier.size());
    target_patch_identities.reserve(carrier.size());
    for (std::size_t global = 0; global < carrier.size(); ++global) {
      source_patch_identities.push_back(spec.source_block +
                                        "::source-patch::" + std::to_string(global));
      target_patch_identities.push_back(spec.target_block +
                                        "::target-patch::" + std::to_string(global));
    }
  }

  field_type& source_state() { return source->sp[static_cast<std::size_t>(source_block_index)].U; }
  field_type& target_state() { return target->sp[static_cast<std::size_t>(target_block_index)].U; }

  void validate_static_contract() const {
    if (source_owner == target_owner || source == nullptr || target == nullptr)
      throw std::invalid_argument(
          "prepared System layout transfer requires two distinct live Systems");
    if (!component_handle)
      throw std::invalid_argument("prepared System layout transfer requires a loaded component");
    for (const auto* field :
         {&spec.mapping_identity, &spec.provider_identity, &spec.provider_component_identity,
          &spec.provider_manifest_identity, &spec.source_layout_identity,
          &spec.target_layout_identity, &spec.source_block, &spec.target_block})
      require_text(*field, "non-empty authenticated identities");
    if (spec.source_layout_identity == spec.target_layout_identity)
      throw std::invalid_argument("prepared layout transfer must cross distinct layouts");
    if (spec.source_representation != kCellAverageRepresentation ||
        spec.target_representation != kCellAverageRepresentation)
      throw std::invalid_argument(
          "prepared conservative transfer requires exact cell-average representations");
    if (spec.synchronization_identity != kBeforeStepSynchronization)
      throw std::invalid_argument(
          "prepared System transfer requires exact before-step synchronization");
    if (spec.operation != POPS_TRANSFER_OPERATION_CONSERVATIVE_CELL_AVERAGE_V1)
      throw std::invalid_argument("prepared System transfer operation is unsupported");
    for (int axis = 0; axis < Dim; ++axis) {
      const auto position = static_cast<std::size_t>(axis);
      if (spec.refinement_ratio[position] <= 0)
        throw std::invalid_argument("prepared System transfer ratios must be positive");
      if (source->cfg.lower[axis] != target->cfg.lower[axis] ||
          source->cfg.upper[axis] != target->cfg.upper[axis] ||
          source->periodicity[position] != target->periodicity[position])
        throw std::invalid_argument(
            "prepared System conservative transfer requires one exact physical domain/topology");
      const std::int64_t expected = target->dom.length(axis) * spec.refinement_ratio[position];
      if (source->dom.length(axis) != expected)
        throw std::invalid_argument(
            "prepared System transfer ratio does not authenticate the source/target extents");
    }
    if (!source->dm.matches_layout(source->ba) || !target->dm.matches_layout(target->ba))
      throw std::invalid_argument("prepared System transfer received an invalid native layout");
    const auto& source_block = source->blocks_.find(spec.source_block);
    const auto& target_block = target->blocks_.find(spec.target_block);
    if (source_block.ncomp != target_block.ncomp || source_block.ncomp <= 0)
      throw std::invalid_argument("prepared System transfer source/target provider widths differ");
    if (source_block.U.layout() != source->ba || source_block.U.distribution() != source->dm ||
        target_block.U.layout() != target->ba || target_block.U.distribution() != target->dm)
      throw std::invalid_argument(
          "prepared System transfer block storage differs from its owning layout");
    if (source_owner->lifecycle_state() == "assembling" ||
        target_owner->lifecycle_state() == "assembling")
      throw std::invalid_argument("prepared System transfer requires bound native Systems");
    const PopsComponentApiV1& api = component_handle->api();
    if (api.component_id == nullptr || api.manifest_identity == nullptr ||
        api.semantic_identity == nullptr || api.catalog_sha256 == nullptr ||
        api.abi_key == nullptr || api.semantic_identity[0] == '\0' ||
        api.catalog_sha256[0] == '\0' || api.abi_key[0] == '\0' ||
        spec.provider_component_identity != api.component_id ||
        spec.provider_manifest_identity != api.manifest_identity)
      throw std::invalid_argument(
          "prepared System transfer provider identity differs from its loaded component");
    (void)component_handle->table<PopsTransferApiV1>(POPS_NATIVE_INTERFACE_TRANSFER_V1, 1u);
  }

  std::string consensus_payload() const {
    std::string bytes;
    for (const auto* field :
         {&spec.mapping_identity, &spec.provider_identity, &spec.provider_component_identity,
          &spec.provider_manifest_identity, &spec.source_layout_identity,
          &spec.target_layout_identity, &spec.source_block, &spec.target_block,
          &spec.source_representation, &spec.target_representation, &spec.synchronization_identity})
      append_text(bytes, *field);
    for (const std::int32_t ratio : spec.refinement_ratio)
      append_i32(bytes, ratio);
    append_i32(bytes, spec.operation);
    append_text(bytes, execution.execution_identity);
    append_i32(bytes, static_cast<std::int32_t>(execution.context_version));
    append_i32(bytes, execution.memory_space);
    append_text(bytes, execution.backend_identity);
    append_text(bytes, execution.device_identity);
    append_i32(bytes, execution.scalar_type);
    append_i32(bytes, execution.storage_precision);
    append_i32(bytes, execution.compute_precision);
    append_i32(bytes, execution.accumulation_precision);
    append_i32(bytes, execution.reduction_precision);
    append_text(bytes, execution.stream_identity);
    append_text(bytes, execution.communicator_identity);
    append_text(bytes, execution.communicator_datatype_identity);
    const PopsComponentApiV1& api = component_handle->api();
    append_text(bytes, api.semantic_identity == nullptr ? "" : api.semantic_identity);
    append_text(bytes, api.catalog_sha256 == nullptr ? "" : api.catalog_sha256);
    append_text(bytes, api.abi_key == nullptr ? "" : api.abi_key);
    for (int axis = 0; axis < Dim; ++axis) {
      append_double(bytes, source->cfg.lower[axis]);
      append_double(bytes, source->cfg.upper[axis]);
      append_double(bytes, target->cfg.lower[axis]);
      append_double(bytes, target->cfg.upper[axis]);
    }
    append_i32(bytes, components);
    append_layout(bytes, source->ba, source->dm);
    append_layout(bytes, target->ba, target->dm);
    return bytes;
  }

  void prepare_provider() {
    transfer_api =
        &component_handle->table<PopsTransferApiV1>(POPS_NATIVE_INTERFACE_TRANSFER_V1, 1u);
    component_state =
        component_handle->prepare_fresh_state(POPS_NATIVE_INTERFACE_TRANSFER_V1, 1u, execution_abi);
  }

  void validate_active(std::uint64_t generation, std::uint64_t attempt, const char* where) const {
    if (!active || generation == 0 || generation != active_generation)
      throw std::logic_error(std::string(where) + " crossed its active transfer generation");
    if (attempt == 0)
      throw std::invalid_argument(std::string(where) + " requires a positive attempt");
    if (!source->external_step_transaction_ || !target->external_step_transaction_ ||
        source->external_step_transaction_committed_ ||
        target->external_step_transaction_committed_)
      throw std::logic_error(std::string(where) +
                             " requires active uncommitted native System transactions");
  }
};

template <int Dim>
PreparedSystemLayoutTransfer<Dim>::PreparedSystemLayoutTransfer(std::unique_ptr<Impl> impl) noexcept
    : p_(std::move(impl)) {}

template <int Dim>
PreparedSystemLayoutTransfer<Dim>::~PreparedSystemLayoutTransfer() = default;

template <int Dim>
std::shared_ptr<PreparedSystemLayoutTransfer<Dim>> PreparedSystemLayoutTransfer<Dim>::prepare(
    System<Dim>& source, System<Dim>& target, std::shared_ptr<component::LoadedComponent> component,
    SystemLayoutTransferSpec<Dim> spec, SystemLayoutTransferExecution execution) {
  const CommunicatorView field_rank_space = world_communicator_view();
  CommunicatorView communicator;
  collectively_validate(field_rank_space, "layout-transfer execution communicator", [&] {
    communicator = resolve_execution_communicator(execution, field_rank_space);
  });
  std::unique_ptr<Impl> pending;
  collectively_validate(communicator, "prepared System layout-transfer allocation", [&] {
    pending = std::make_unique<Impl>(source, target, std::move(component), std::move(spec),
                                     std::move(execution), communicator);
  });
  const std::string payload = pending->consensus_payload();
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"prepared-system-layout-transfer-v2", payload}},
                                                communicator))
    throw std::invalid_argument(
        "prepared System layout-transfer contract differs between MPI ranks");
  collectively_validate(communicator, "native Transfer provider preparation",
                        [&] { pending->prepare_provider(); });
  collectively_validate(communicator, "prepared System layout-transfer warmup", [&] {
    parallel_copy(pending->source_snapshot, pending->source_state(),
                  *pending->source_copy_schedule);
  });
  return std::shared_ptr<PreparedSystemLayoutTransfer>(
      new PreparedSystemLayoutTransfer(std::move(pending)));
}

template <int Dim>
const SystemLayoutTransferSpec<Dim>& PreparedSystemLayoutTransfer<Dim>::spec() const noexcept {
  return p_->spec;
}

template <int Dim>
void PreparedSystemLayoutTransfer<Dim>::begin_transaction(std::uint64_t generation) {
  collectively_validate(p_->communicator, "layout-transfer begin", [&] {
    if (p_->active)
      throw std::logic_error("layout-transfer transaction is already active");
    if (generation == 0 || generation <= p_->last_generation)
      throw std::invalid_argument("layout-transfer generation must be positive and monotonic");
    if (!p_->source->external_step_transaction_ || !p_->target->external_step_transaction_ ||
        p_->source->external_step_transaction_committed_ ||
        p_->target->external_step_transaction_committed_)
      throw std::logic_error(
          "layout-transfer begin requires active uncommitted native System transactions");
  });
  p_->active = true;
  p_->active_generation = generation;
  p_->captured_attempt = 0;
  p_->applied = false;
}

template <int Dim>
void PreparedSystemLayoutTransfer<Dim>::capture(std::uint64_t generation, std::uint64_t attempt) {
  collectively_validate(p_->communicator, "layout-transfer capture", [&] {
    p_->validate_active(generation, attempt, "layout-transfer capture");
    if (p_->applied)
      throw std::logic_error(
          "layout-transfer retry requires rollback of the enclosing transaction");
    if (p_->captured_attempt != 0 && p_->captured_attempt != attempt)
      throw std::logic_error("layout-transfer source was already captured for another attempt");
  });
  collectively_validate(p_->communicator, "layout-transfer source capture", [&] {
    parallel_copy(p_->source_snapshot, p_->source_state(), *p_->source_copy_schedule);
  });
  p_->captured_attempt = attempt;
}

template <int Dim>
SystemLayoutTransferReceipt PreparedSystemLayoutTransfer<Dim>::apply(std::uint64_t generation,
                                                                     std::uint64_t attempt) {
  collectively_validate(p_->communicator, "layout-transfer apply preflight", [&] {
    p_->validate_active(generation, attempt, "layout-transfer apply");
    if (p_->captured_attempt != attempt)
      throw std::logic_error("layout-transfer apply requires the exact captured attempt");
    if (p_->applied)
      throw std::logic_error("layout-transfer attempt was already applied");
  });

  std::uint64_t local_source_elements = 0;
  std::uint64_t local_target_elements = 0;
  collectively_validate(p_->communicator, "native Transfer apply", [&] {
    MultiFab<Dim>& destination = p_->target_state();
    try {
      for (std::size_t local = 0; local < p_->source_snapshot.local_size(); ++local) {
        const std::size_t global = p_->source_snapshot.global_index(local);
        const std::size_t destination_local = destination.local_index_of(global);
        if (destination_local == MultiFab<Dim>::not_local)
          throw std::logic_error(
              "prepared layout-transfer source/target ownership diverged after bind");
        const Fab<Dim>& source_fab = p_->source_snapshot.fab(local);
        Fab<Dim>& destination_fab = destination.fab(destination_local);
        const Box<Dim>& source_box = source_fab.box();
        const Box<Dim>& destination_box = destination_fab.box();
        const PopsConstFieldViewV1 source_view = component_field_view<PopsConstFieldViewV1>(
            source_fab.view(), source_box, p_->components,
            static_cast<PopsMemorySpaceV1>(p_->execution.memory_space),
            p_->spec.source_layout_identity.c_str(), p_->source_patch_identities[global].c_str());
        const PopsFieldViewV1 destination_view = component_field_view<PopsFieldViewV1>(
            destination_fab.view(), destination_box, p_->components,
            static_cast<PopsMemorySpaceV1>(p_->execution.memory_space),
            p_->spec.target_layout_identity.c_str(), p_->target_patch_identities[global].c_str());
        PopsTransferRequestV1 request{sizeof(PopsTransferRequestV1),
                                      source_view,
                                      destination_view,
                                      p_->spec.refinement_ratio.data(),
                                      Dim,
                                      static_cast<PopsTransferOperationV1>(p_->spec.operation),
                                      p_->execution_abi};
        PopsComponentStatusV1 status{sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1,
                                     nullptr};
        const int code = component::apply_transfer(*p_->transfer_api, p_->component_state.get(),
                                                   request, status);
        if (!component::component_status_is_well_formed(status) || code != 0 || status.code != 0 ||
            status.action != POPS_COMPONENT_CONTINUE_V1)
          throw std::runtime_error(status.reason == nullptr ? "native Transfer provider failed"
                                                            : status.reason);
        const std::uint64_t source_count = checked_elements(source_box, p_->components);
        const std::uint64_t target_count = checked_elements(destination_box, p_->components);
        if (source_count > std::numeric_limits<std::uint64_t>::max() - local_source_elements ||
            target_count > std::numeric_limits<std::uint64_t>::max() - local_target_elements)
          throw std::overflow_error("layout-transfer receipt element count overflow");
        local_source_elements += source_count;
        local_target_elements += target_count;
      }
      device_fence();
    } catch (...) {
      device_fence();
      throw;
    }
  });
  p_->applied = true;

  SystemLayoutTransferReceipt receipt;
  receipt.applied = true;
  receipt.mapping_identity = p_->spec.mapping_identity;
  receipt.provider_identity = p_->spec.provider_identity;
  receipt.provider_component_identity = p_->spec.provider_component_identity;
  receipt.provider_manifest_identity = p_->spec.provider_manifest_identity;
  receipt.source_layout_identity = p_->spec.source_layout_identity;
  receipt.target_layout_identity = p_->spec.target_layout_identity;
  receipt.source_block = p_->spec.source_block;
  receipt.target_block = p_->spec.target_block;
  receipt.execution_identity = p_->execution.execution_identity;
  receipt.operation = p_->spec.operation;
  receipt.generation = generation;
  receipt.attempt = attempt;
  receipt.source_element_count = collective_elements(local_source_elements, p_->communicator);
  receipt.destination_element_count = collective_elements(local_target_elements, p_->communicator);
  return receipt;
}

template <int Dim>
void PreparedSystemLayoutTransfer<Dim>::reject_attempt(std::uint64_t generation,
                                                       std::uint64_t attempt) {
  collectively_validate(p_->communicator, "layout-transfer rejected-attempt reset", [&] {
    p_->validate_active(generation, attempt, "layout-transfer rejected-attempt reset");
    if (p_->captured_attempt != attempt)
      throw std::logic_error(
          "layout-transfer rejected-attempt reset does not match the captured attempt");
  });
  p_->captured_attempt = 0;
  p_->applied = false;
}

template <int Dim>
void PreparedSystemLayoutTransfer<Dim>::finalize_transaction(std::uint64_t generation) noexcept {
  if (!p_->active || generation != p_->active_generation)
    return;
  p_->last_generation = generation;
  p_->active_generation = 0;
  p_->captured_attempt = 0;
  p_->active = false;
  p_->applied = false;
}

template <int Dim>
void PreparedSystemLayoutTransfer<Dim>::rollback_transaction(std::uint64_t generation) noexcept {
  if (!p_->active || generation != p_->active_generation)
    return;
  p_->last_generation = generation;
  p_->active_generation = 0;
  p_->captured_attempt = 0;
  p_->active = false;
  p_->applied = false;
}

template class PreparedSystemLayoutTransfer<kNativeDimension>;

}  // namespace pops
