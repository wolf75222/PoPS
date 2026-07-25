#include "system_impl.hpp"

#include <pops/mesh/layout/refinement.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>

#include <array>
#include <bit>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
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

void append_layout(std::string& bytes, const BoxArray& boxes, const DistributionMapping& owners) {
  append_i32(bytes, boxes.size());
  for (const Box2D& box : boxes.boxes()) {
    append_i32(bytes, box.lo[0]);
    append_i32(bytes, box.lo[1]);
    append_i32(bytes, box.hi[0]);
    append_i32(bytes, box.hi[1]);
  }
  append_i32(bytes, owners.size());
  for (const int owner : owners.ranks())
    append_i32(bytes, owner);
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

void validate_world_execution(const SystemLayoutTransferExecution& execution,
                              const CommunicatorView& world) {
  const PopsExecutionContextV1 view = execution_view(execution);
  component::validate_execution_context(view);
  if (execution.memory_space != POPS_MEMORY_SPACE_HOST_V1 &&
      execution.memory_space != POPS_MEMORY_SPACE_MANAGED_V1)
    throw std::invalid_argument(
        "prepared System layout transfer requires host-addressable native field storage");
  if (execution.communicator_identity == "serial") {
    if (world.active())
      throw std::invalid_argument(
          "serial layout-transfer execution requires native MPI to be inactive");
    return;
  }
  if (execution.communicator_identity != "MPI_COMM_WORLD")
    throw std::invalid_argument(
        "prepared System layout transfer supports serial or exact MPI_COMM_WORLD execution");
#ifdef POPS_HAS_MPI
  if (!world.active())
    throw std::invalid_argument(
        "MPI_COMM_WORLD layout-transfer execution requires initialized native MPI");
  if (execution.communicator_f_handle != static_cast<std::int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD)) ||
      execution.communicator_datatype_f_handle !=
          static_cast<std::int64_t>(MPI_Type_c2f(MPI_DOUBLE)) ||
      execution.communicator_datatype_identity != "MPI_DOUBLE")
    throw std::invalid_argument(
        "layout-transfer execution handles are not exact MPI_COMM_WORLD/MPI_DOUBLE authorities");
#else
  (void)world;
  throw std::invalid_argument(
      "MPI_COMM_WORLD layout-transfer execution requires an MPI-enabled PoPS build");
#endif
}

template <class Function>
void collectively_validate(const CommunicatorView& world, const char* where, Function&& function) {
  std::exception_ptr failure;
  try {
    std::forward<Function>(function)();
  } catch (...) {
    failure = std::current_exception();
  }
  const long failures = all_reduce_sum(failure ? 1L : 0L, world);
  if (failures == 0)
    return;
  if (world.size() == 1 && failure)
    std::rethrow_exception(failure);
  throw std::runtime_error(std::string(where) + " failed on at least one MPI rank");
}

int checked_index(std::int64_t value, const char* where) {
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    throw std::overflow_error(std::string(where) + " exceeds the native Box2D index range");
  return static_cast<int>(value);
}

BoxArray source_carrier_boxes(const BoxArray& target_boxes, const Box2D& source_domain,
                              const Box2D& target_domain,
                              const std::array<std::int32_t, 2>& ratio) {
  const std::int64_t ratio_x = ratio[1];
  const std::int64_t ratio_y = ratio[0];
  std::vector<Box2D> boxes;
  boxes.reserve(static_cast<std::size_t>(target_boxes.size()));
  for (const Box2D& target : target_boxes.boxes()) {
    const std::int64_t xlo =
        source_domain.lo[0] +
        (static_cast<std::int64_t>(target.lo[0]) - target_domain.lo[0]) * ratio_x;
    const std::int64_t ylo =
        source_domain.lo[1] +
        (static_cast<std::int64_t>(target.lo[1]) - target_domain.lo[1]) * ratio_y;
    const std::int64_t xhi =
        source_domain.lo[0] +
        (static_cast<std::int64_t>(target.hi[0]) - target_domain.lo[0] + 1) * ratio_x - 1;
    const std::int64_t yhi =
        source_domain.lo[1] +
        (static_cast<std::int64_t>(target.hi[1]) - target_domain.lo[1] + 1) * ratio_y - 1;
    boxes.push_back({{checked_index(xlo, "source carrier lower x"),
                      checked_index(ylo, "source carrier lower y")},
                     {checked_index(xhi, "source carrier upper x"),
                      checked_index(yhi, "source carrier upper y")}});
  }
  return BoxArray(std::move(boxes));
}

std::uint64_t checked_elements(const Box2D& box, int components) {
  const std::int64_t cells = box.num_cells();
  if (cells <= 0 || components <= 0 ||
      static_cast<std::uint64_t>(cells) >
          std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(components))
    throw std::overflow_error("layout-transfer field element count exceeds uint64 capacity");
  return static_cast<std::uint64_t>(cells) * static_cast<std::uint64_t>(components);
}

std::uint64_t collective_elements(std::uint64_t local, const CommunicatorView& world) {
  const auto ranks = static_cast<std::uint64_t>(world.size());
  const std::uint64_t per_rank_limit =
      static_cast<std::uint64_t>(std::numeric_limits<long>::max()) / ranks;
  const long invalid = all_reduce_max(local > per_rank_limit ? 1L : 0L, world);
  if (invalid != 0)
    throw std::overflow_error("layout-transfer global element count exceeds MPI long capacity");
  const long global = all_reduce_sum(static_cast<long>(local), world);
  return static_cast<std::uint64_t>(global);
}

}  // namespace

struct PreparedSystemLayoutTransfer::Impl {
  System* source_owner = nullptr;
  System* target_owner = nullptr;
  System::Impl* source = nullptr;
  System::Impl* target = nullptr;
  std::shared_ptr<component::LoadedComponent> component_handle;
  component::LoadedComponent::PreparedState component_state;
  const PopsTransferApiV1* transfer_api = nullptr;
  SystemLayoutTransferSpec spec;
  SystemLayoutTransferExecution execution;
  PopsExecutionContextV1 execution_abi{};
  CommunicatorView world;
  int source_block_index = -1;
  int target_block_index = -1;
  int components = 0;
  MultiFab source_snapshot;
  std::vector<std::string> source_patch_identities;
  std::vector<std::string> target_patch_identities;
  std::uint64_t active_generation = 0;
  std::uint64_t last_generation = 0;
  std::uint64_t captured_attempt = 0;
  bool active = false;
  bool applied = false;

  Impl(System& source_system, System& target_system,
       std::shared_ptr<component::LoadedComponent> loaded, SystemLayoutTransferSpec transfer_spec,
       SystemLayoutTransferExecution transfer_execution)
      : source_owner(&source_system),
        target_owner(&target_system),
        source(source_system.p_.get()),
        target(target_system.p_.get()),
        component_handle(std::move(loaded)),
        spec(std::move(transfer_spec)),
        execution(std::move(transfer_execution)),
        execution_abi(execution_view(execution)),
        world(world_communicator_view()) {
    validate_static_contract();
    source_block_index = source->blocks_.index(spec.source_block);
    target_block_index = target->blocks_.index(spec.target_block);
    components = source->sp[static_cast<std::size_t>(source_block_index)].ncomp;
    const BoxArray carrier =
        source_carrier_boxes(target->ba, source->dom, target->dom, spec.refinement_ratio);
    source_snapshot = MultiFab(carrier, target->dm, components, 0);
    source_patch_identities.reserve(static_cast<std::size_t>(carrier.size()));
    target_patch_identities.reserve(static_cast<std::size_t>(carrier.size()));
    for (int global = 0; global < carrier.size(); ++global) {
      source_patch_identities.push_back(spec.source_block +
                                        "::source-patch::" + std::to_string(global));
      target_patch_identities.push_back(spec.target_block +
                                        "::target-patch::" + std::to_string(global));
    }
  }

  MultiFab& source_state() { return source->sp[static_cast<std::size_t>(source_block_index)].U; }
  MultiFab& target_state() { return target->sp[static_cast<std::size_t>(target_block_index)].U; }

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
    if (spec.refinement_ratio[0] <= 0 || spec.refinement_ratio[1] <= 0)
      throw std::invalid_argument("prepared System transfer ratios must be positive");
    if (source->polar_ || target->polar_ || source->cfg.geometry != "cartesian" ||
        target->cfg.geometry != "cartesian")
      throw std::invalid_argument(
          "prepared System conservative transfer currently requires Cartesian layouts");
    if (source->cfg.L != target->cfg.L || source->cfg.xlo != target->cfg.xlo ||
        source->cfg.ylo != target->cfg.ylo)
      throw std::invalid_argument(
          "prepared System conservative transfer requires one exact physical domain");
    if (source->per_.x != target->per_.x || source->per_.y != target->per_.y)
      throw std::invalid_argument(
          "prepared System conservative transfer requires one exact boundary topology");
    const std::int64_t expected_x =
        static_cast<std::int64_t>(target->dom.nx()) * spec.refinement_ratio[1];
    const std::int64_t expected_y =
        static_cast<std::int64_t>(target->dom.ny()) * spec.refinement_ratio[0];
    if (source->dom.nx() != expected_x || source->dom.ny() != expected_y)
      throw std::invalid_argument(
          "prepared System transfer ratio does not authenticate the exact source/target extents");
    if (!source->ba.tiles_exactly(source->dom) || !target->ba.tiles_exactly(target->dom) ||
        source->ba.size() != source->dm.size() || target->ba.size() != target->dm.size())
      throw std::invalid_argument("prepared System transfer received an invalid native layout");
    const auto& source_block = source->blocks_.find(spec.source_block);
    const auto& target_block = target->blocks_.find(spec.target_block);
    if (source_block.ncomp != target_block.ncomp || source_block.ncomp <= 0)
      throw std::invalid_argument("prepared System transfer source/target provider widths differ");
    if (source_block.U.box_array().boxes() != source->ba.boxes() ||
        source_block.U.dmap().ranks() != source->dm.ranks() ||
        target_block.U.box_array().boxes() != target->ba.boxes() ||
        target_block.U.dmap().ranks() != target->dm.ranks())
      throw std::invalid_argument(
          "prepared System transfer block storage differs from its owning layout");
    if (source_owner->lifecycle_state() == "assembling" ||
        target_owner->lifecycle_state() == "assembling")
      throw std::invalid_argument("prepared System transfer requires bound native Systems");
    validate_world_execution(execution, world);
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
    bytes.reserve(512u + 24u * static_cast<std::size_t>(source->ba.size() + target->ba.size()));
    for (const auto* field :
         {&spec.mapping_identity, &spec.provider_identity, &spec.provider_component_identity,
          &spec.provider_manifest_identity, &spec.source_layout_identity,
          &spec.target_layout_identity, &spec.source_block, &spec.target_block,
          &spec.source_representation, &spec.target_representation, &spec.synchronization_identity})
      append_text(bytes, *field);
    append_i32(bytes, spec.refinement_ratio[0]);
    append_i32(bytes, spec.refinement_ratio[1]);
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
    append_double(bytes, source->cfg.L);
    append_double(bytes, source->cfg.xlo);
    append_double(bytes, source->cfg.ylo);
    append_double(bytes, target->cfg.L);
    append_double(bytes, target->cfg.xlo);
    append_double(bytes, target->cfg.ylo);
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

PreparedSystemLayoutTransfer::PreparedSystemLayoutTransfer(std::unique_ptr<Impl> impl) noexcept
    : p_(std::move(impl)) {}

PreparedSystemLayoutTransfer::~PreparedSystemLayoutTransfer() = default;

std::shared_ptr<PreparedSystemLayoutTransfer> PreparedSystemLayoutTransfer::prepare(
    System& source, System& target, std::shared_ptr<component::LoadedComponent> component,
    SystemLayoutTransferSpec spec, SystemLayoutTransferExecution execution) {
  const CommunicatorView world = world_communicator_view();
  std::unique_ptr<Impl> pending;
  collectively_validate(world, "prepared System layout-transfer allocation", [&] {
    pending = std::make_unique<Impl>(source, target, std::move(component), std::move(spec),
                                     std::move(execution));
  });
  const std::string payload = pending->consensus_payload();
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"prepared-system-layout-transfer-v1", payload}},
                                                world))
    throw std::invalid_argument(
        "prepared System layout-transfer contract differs between MPI ranks");
  collectively_validate(world, "native Transfer provider preparation",
                        [&] { pending->prepare_provider(); });
  // Warm the persistent copy schedule and MPI buffers before the first run step.  This copy is
  // observationally inert: the carrier is private until capture() authenticates an attempt.
  collectively_validate(world, "prepared System layout-transfer warmup",
                        [&] { parallel_copy(pending->source_snapshot, pending->source_state()); });
  return std::shared_ptr<PreparedSystemLayoutTransfer>(
      new PreparedSystemLayoutTransfer(std::move(pending)));
}

const SystemLayoutTransferSpec& PreparedSystemLayoutTransfer::spec() const noexcept {
  return p_->spec;
}

void PreparedSystemLayoutTransfer::begin_transaction(std::uint64_t generation) {
  collectively_validate(p_->world, "layout-transfer begin", [&] {
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

void PreparedSystemLayoutTransfer::capture(std::uint64_t generation, std::uint64_t attempt) {
  collectively_validate(p_->world, "layout-transfer capture", [&] {
    p_->validate_active(generation, attempt, "layout-transfer capture");
    if (p_->applied)
      throw std::logic_error(
          "layout-transfer retry requires rollback of the enclosing transaction");
    if (p_->captured_attempt != 0 && p_->captured_attempt != attempt)
      throw std::logic_error("layout-transfer source was already captured for another attempt");
  });
  collectively_validate(p_->world, "layout-transfer source capture",
                        [&] { parallel_copy(p_->source_snapshot, p_->source_state()); });
  p_->captured_attempt = attempt;
}

SystemLayoutTransferReceipt PreparedSystemLayoutTransfer::apply(std::uint64_t generation,
                                                                std::uint64_t attempt) {
  collectively_validate(p_->world, "layout-transfer apply preflight", [&] {
    p_->validate_active(generation, attempt, "layout-transfer apply");
    if (p_->captured_attempt != attempt)
      throw std::logic_error("layout-transfer apply requires the exact captured attempt");
    if (p_->applied)
      throw std::logic_error("layout-transfer attempt was already applied");
  });

  std::uint64_t local_source_elements = 0;
  std::uint64_t local_target_elements = 0;
  collectively_validate(p_->world, "native Transfer apply", [&] {
    MultiFab& destination = p_->target_state();
    try {
      for (int local = 0; local < p_->source_snapshot.local_size(); ++local) {
        const int global = p_->source_snapshot.global_index(local);
        const int destination_local = destination.local_index_of(global);
        if (destination_local < 0)
          throw std::logic_error(
              "prepared layout-transfer source/target ownership diverged after bind");
        const Fab2D& source_fab = p_->source_snapshot.fab(local);
        Fab2D& destination_fab = destination.fab(destination_local);
        const Box2D& source_box = source_fab.box();
        const Box2D& destination_box = destination_fab.box();
        const ConstArray4 source_values = source_fab.const_array();
        const Array4 destination_values = destination_fab.array();
        const PopsConstFieldViewV1 source_view{
            sizeof(PopsConstFieldViewV1),
            source_values.p,
            2,
            {static_cast<std::size_t>(source_box.ny()), static_cast<std::size_t>(source_box.nx()),
             1},
            {source_values.nx_tot, 1, 0},
            static_cast<std::size_t>(p_->components),
            source_values.comp_stride,
            POPS_FIELD_CENTERING_CELL_V1,
            0,
            {0, 0, 0},
            {0, 0, 0},
            POPS_SCALAR_FLOAT64_V1,
            static_cast<PopsMemorySpaceV1>(p_->execution.memory_space),
            p_->spec.source_layout_identity.c_str(),
            p_->source_patch_identities[static_cast<std::size_t>(global)].c_str(),
            POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
        const PopsFieldViewV1 destination_view{
            sizeof(PopsFieldViewV1),
            &destination_values(destination_box.lo[0], destination_box.lo[1], 0),
            2,
            {static_cast<std::size_t>(destination_box.ny()),
             static_cast<std::size_t>(destination_box.nx()), 1},
            {destination_values.nx_tot, 1, 0},
            static_cast<std::size_t>(p_->components),
            destination_values.comp_stride,
            POPS_FIELD_CENTERING_CELL_V1,
            0,
            {0, 0, 0},
            {0, 0, 0},
            POPS_SCALAR_FLOAT64_V1,
            static_cast<PopsMemorySpaceV1>(p_->execution.memory_space),
            p_->spec.target_layout_identity.c_str(),
            p_->target_patch_identities[static_cast<std::size_t>(global)].c_str(),
            POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
        PopsTransferRequestV1 request{sizeof(PopsTransferRequestV1),
                                      source_view,
                                      destination_view,
                                      p_->spec.refinement_ratio.data(),
                                      2,
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
      // A provider may have launched asynchronous work before reporting an error.
      // Fence before collective error propagation so no borrowed field view outlives
      // this call or races the enclosing transaction rollback.
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
  receipt.source_element_count = collective_elements(local_source_elements, p_->world);
  receipt.destination_element_count = collective_elements(local_target_elements, p_->world);
  return receipt;
}

void PreparedSystemLayoutTransfer::reject_attempt(std::uint64_t generation, std::uint64_t attempt) {
  collectively_validate(p_->world, "layout-transfer rejected-attempt reset", [&] {
    p_->validate_active(generation, attempt, "layout-transfer rejected-attempt reset");
    if (p_->captured_attempt != attempt)
      throw std::logic_error(
          "layout-transfer rejected-attempt reset does not match the captured attempt");
  });
  p_->captured_attempt = 0;
  p_->applied = false;
}

void PreparedSystemLayoutTransfer::finalize_transaction(std::uint64_t generation) noexcept {
  if (!p_->active || generation != p_->active_generation)
    return;
  p_->last_generation = generation;
  p_->active_generation = 0;
  p_->captured_attempt = 0;
  p_->active = false;
  p_->applied = false;
}

void PreparedSystemLayoutTransfer::rollback_transaction(std::uint64_t generation) noexcept {
  if (!p_->active || generation != p_->active_generation)
    return;
  p_->last_generation = generation;
  p_->active_generation = 0;
  p_->captured_attempt = 0;
  p_->active = false;
  p_->applied = false;
}

}  // namespace pops
