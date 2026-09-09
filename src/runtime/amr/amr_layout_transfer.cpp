#include <pops/runtime/amr/amr_layout_transfer.hpp>

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/parallel/region_transfer.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <exception>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace pops {
namespace {

constexpr std::string_view kCartesian = "pops://measure/cartesian-cells@1";
constexpr std::string_view kRelative = "pops://measure/cellwise-constant-relative-volume@1";

void add(std::size_t& value, std::size_t increment, std::size_t limit, const char* message) {
  if (value > limit || increment > limit - value)
    throw std::length_error(message);
  value += increment;
}

std::size_t multiply(std::size_t left, std::size_t right) {
  if (left && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error("AMR layout transfer size exceeds size_t");
  return left * right;
}

void integer(std::string& bytes, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8)
    bytes.push_back(static_cast<char>((value >> shift) & 255));
}
void text(std::string& bytes, std::string_view value) {
  integer(bytes, value.size());
  bytes.append(value);
}
void real(std::string& bytes, double value) {
  integer(bytes, std::bit_cast<std::uint64_t>(value));
}
void require_text(std::string_view value) {
  if (value.empty())
    throw std::invalid_argument("AMR layout transfer requires explicit nonempty identities");
}

template <class Function>
void collective(const ExecutionLane& lane, const char* operation, Function&& function) {
  std::exception_ptr error;
  try {
    function();
  } catch (...) {
    error = std::current_exception();
  }
  if (all_reduce_max(error ? 1L : 0L, lane)) {
    if (lane.size() == 1 && error)
      std::rethrow_exception(error);
    throw std::runtime_error(std::string(operation) + " failed on a lane rank");
  }
}

void agree(const ExecutionLane& lane, std::string_view key, const std::string& value) {
  if (!all_ranks_agree_exact_ordered_byte_pairs({{key, value}}, lane))
    throw std::invalid_argument("AMR layout transfer exact contract differs between lane ranks");
}

std::uint64_t global_count(std::size_t local, const ExecutionLane& lane) {
  if (all_reduce_max(local > static_cast<std::size_t>(std::numeric_limits<long>::max()) /
                                 static_cast<std::size_t>(lane.size())
                         ? 1L
                         : 0L,
                     lane))
    throw std::overflow_error("AMR transfer collective count exceeds long");
  return static_cast<std::uint64_t>(all_reduce_sum(static_cast<long>(local), lane));
}

template <int Dim>
std::size_t cells(const Box<Dim>& box) {
  if (box.empty())
    return 0;
  std::size_t count = 1;
  for (int axis = 0; axis < Dim; ++axis)
    count = multiply(count, static_cast<std::size_t>(box.length(axis)));
  return count;
}

template <int Dim>
Index<Dim> cell_at(const Box<Dim>& box, std::size_t ordinal) {
  Index<Dim> index{};
  for (int axis = 0; axis < Dim; ++axis) {
    const auto extent = static_cast<std::size_t>(box.length(axis));
    index[axis] = static_cast<int>(static_cast<std::int64_t>(box.lo[axis]) + ordinal % extent);
    ordinal /= extent;
  }
  return index;
}

template <int Dim>
std::size_t offset(const Box<Dim>& box, const Index<Dim>& index) {
  std::size_t result = 0, stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    result +=
        static_cast<std::size_t>(static_cast<std::int64_t>(index[axis]) - box.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(box.length(axis));
  }
  return result;
}

std::uint64_t common_lattice(std::uint64_t left, std::uint64_t right) {
  if (!left || !right)
    throw std::invalid_argument("AMR transfer lattice extents must be positive");
  const auto quotient = left / std::gcd(left, right);
  if (quotient > std::numeric_limits<std::uint64_t>::max() / right)
    throw std::overflow_error("AMR transfer exact common lattice exceeds uint64");
  return quotient * right;
}

// Exact integer topology until the final rational measure is converted to float64.
double overlap_fraction(std::uint64_t source_cell, std::uint64_t source_cells,
                        std::uint64_t target_cell, std::uint64_t target_cells) {
  const auto lattice = common_lattice(source_cells, target_cells);
  const auto source_width = lattice / source_cells;
  const auto target_width = lattice / target_cells;
  const auto lower = std::max(source_cell * source_width, target_cell * target_width);
  const auto upper = std::min((source_cell + 1) * source_width, (target_cell + 1) * target_width);
  return upper <= lower ? 0.0
                        : static_cast<double>(upper - lower) / static_cast<double>(target_width);
}

PopsExecutionContextV1 execution_view(const SystemLayoutTransferExecution& e) {
  return {sizeof(PopsExecutionContextV1),
          e.context_version,
          e.execution_identity.c_str(),
          static_cast<PopsMemorySpaceV1>(e.memory_space),
          e.backend_identity.c_str(),
          e.device_identity.c_str(),
          static_cast<PopsScalarTypeV1>(e.scalar_type),
          static_cast<PopsPrecisionV1>(e.storage_precision),
          static_cast<PopsPrecisionV1>(e.compute_precision),
          static_cast<PopsPrecisionV1>(e.accumulation_precision),
          static_cast<PopsPrecisionV1>(e.reduction_precision),
          e.stream_handle,
          e.stream_identity.c_str(),
          e.communicator_f_handle,
          e.communicator_datatype_f_handle,
          e.communicator_identity.c_str(),
          e.communicator_datatype_identity.c_str()};
}

void validate_execution(const SystemLayoutTransferExecution& execution,
                        const ExecutionLane& authority) {
  component::validate_execution_context(execution_view(execution));
  if (execution.memory_space != POPS_MEMORY_SPACE_HOST_V1)
    throw std::invalid_argument("AMR physical transfer requires declared host execution");
  if constexpr (!std::is_same_v<Real, double>)
    throw std::invalid_argument("AMR physical float64 provider requires native float64 storage");
#ifdef POPS_HAS_MPI
  if (authority.active()) {
    const auto communicator = MPI_Comm_f2c(static_cast<MPI_Fint>(execution.communicator_f_handle));
    if (execution.communicator_identity == "serial" || communicator == MPI_COMM_NULL ||
        MPI_Type_f2c(static_cast<MPI_Fint>(execution.communicator_datatype_f_handle)) != MPI_DOUBLE)
      throw std::invalid_argument("AMR physical execution has no exact MPI communicator/datatype");
    int relation = MPI_UNEQUAL;
    detail::require_mpi_success(
        MPI_Comm_compare(communicator, authority.native_handle(), &relation),
        "MPI_Comm_compare(AMR layout transfer)");
    if (relation != MPI_IDENT && relation != MPI_CONGRUENT)
      throw std::invalid_argument("AMR physical execution changes the prepared rank space");
  } else
#endif
      if (execution.communicator_identity != "serial")
    throw std::invalid_argument(
        "AMR physical serial execution has a collective communicator identity");
}

template <int Dim>
struct LevelContract {
  int level;
  Geometry<Dim> geometry;
  mesh::BoxArray<Dim> layout;
  mesh::Distribution<Dim> distribution;
  Index<Dim> rank;
  Extent<Dim> ghosts, coverage_ghosts, activity_ghosts, measure_ghosts;
  int components;
  bool activity;
  bool relative_measure;
};

template <int Dim>
std::vector<LevelContract<Dim>> endpoint_contract(const AmrTransferEndpoint<Dim>& endpoint,
                                                  const ExecutionLane& lane) {
  require_text(endpoint.layout_identity);
  require_text(endpoint.state_identity);
  require_text(endpoint.hierarchy_identity);
  require_text(endpoint.stage_identity);
  if (endpoint.levels.empty() ||
      (endpoint.measure_identity != kCartesian && endpoint.measure_identity != kRelative))
    throw std::invalid_argument("AMR transfer endpoint requires levels and an exact measure model");
  std::vector<LevelContract<Dim>> result;
  for (std::size_t level = 0; level < endpoint.levels.size(); ++level) {
    const auto& view = endpoint.levels[level];
    if (view.level != static_cast<int>(level) || !view.state || !view.coverage)
      throw std::invalid_argument("AMR transfer endpoint levels/state/coverage are incomplete");
    const auto& state = *view.state;
    if (state.ncomp() <= 0 || state.rank_space().size() != static_cast<std::size_t>(lane.size()) ||
        state.rank_space().linear_rank(state.local_rank()) != static_cast<std::size_t>(lane.rank()))
      throw std::invalid_argument("AMR transfer endpoint differs from its execution rank space");
    if (state.distribution().replicated() && lane.size() != 1)
      throw std::invalid_argument("AMR transfer requires unique distributed patch ownership");
    for (const auto* mask : {view.coverage, view.activity, view.relative_measure})
      if (mask && (mask->ncomp() != 1 || mask->layout() != state.layout() ||
                   mask->distribution() != state.distribution() ||
                   mask->local_rank() != state.local_rank()))
        throw std::invalid_argument(
            "AMR transfer measure/coverage differs from exact state carrier");
    if ((endpoint.measure_identity == kCartesian && (view.activity || view.relative_measure)) ||
        (endpoint.measure_identity == kRelative && (!view.activity || !view.relative_measure)))
      throw std::invalid_argument(
          "AMR transfer masks do not authenticate the declared measure model");
    const auto count = state.layout().size();
    const auto pairs = count < 2 ? 0 : multiply(count, count - 1) / 2;
    if (!state.layout().is_disjoint_within(view.geometry.domain(), {count, pairs}) ||
        (level == 0 && !state.layout().tiles_exactly(view.geometry.domain(), {count, pairs})))
      throw std::invalid_argument("AMR transfer requires disjoint patches and complete level zero");
    if (level) {
      const auto& parent = result.back();
      if (state.rank_space() != parent.distribution.rank_space() ||
          state.ncomp() != parent.components)
        throw std::invalid_argument("AMR transfer levels disagree on components or rank space");
      Extent<Dim> ratio{};
      for (int axis = 0; axis < Dim; ++axis) {
        const auto fine_cells = view.geometry.domain().length(axis);
        const auto parent_cells = parent.geometry.domain().length(axis);
        if (view.geometry.lower()[axis] != parent.geometry.lower()[axis] ||
            view.geometry.upper()[axis] != parent.geometry.upper()[axis] ||
            fine_cells % parent_cells != 0 || fine_cells < parent_cells)
          throw std::invalid_argument("AMR transfer levels require exact nested Cartesian domains");
        ratio[axis] = fine_cells / parent_cells;
      }
      for (const auto& fine : state.layout().boxes()) {
        Box<Dim> footprint{};
        for (int axis = 0; axis < Dim; ++axis) {
          const auto lower =
              static_cast<std::int64_t>(fine.lo[axis]) - view.geometry.domain().lo[axis];
          const auto upper =
              static_cast<std::int64_t>(fine.hi[axis]) - view.geometry.domain().lo[axis] + 1;
          if (lower % ratio[axis] || upper % ratio[axis])
            throw std::invalid_argument("AMR transfer child patches must align with parent cells");
          footprint.lo[axis] =
              static_cast<int>(parent.geometry.domain().lo[axis] + lower / ratio[axis]);
          footprint.hi[axis] =
              static_cast<int>(parent.geometry.domain().lo[axis] + upper / ratio[axis] - 1);
        }
        std::size_t covered = 0;
        for (const auto& coarse : parent.layout.boxes())
          add(covered, cells(footprint.intersect(coarse)), std::numeric_limits<std::size_t>::max(),
              "AMR transfer nested coverage overflow");
        if (covered != cells(footprint))
          throw std::invalid_argument(
              "AMR transfer child patch lies outside parent patch coverage");
      }
    }
    result.push_back({view.level, view.geometry, state.layout(), state.distribution(),
                      state.local_rank(), state.ghosts(), view.coverage->ghosts(),
                      view.activity ? view.activity->ghosts() : Extent<Dim>{},
                      view.relative_measure ? view.relative_measure->ghosts() : Extent<Dim>{},
                      state.ncomp(), view.activity != nullptr, view.relative_measure != nullptr});
  }
  return result;
}

template <int Dim>
std::string endpoint_bytes(const AmrTransferEndpoint<Dim>& endpoint,
                           const std::vector<LevelContract<Dim>>& levels) {
  std::string bytes;
  for (const auto* value : {&endpoint.layout_identity, &endpoint.state_identity,
                            &endpoint.hierarchy_identity, &endpoint.measure_identity})
    text(bytes, *value);
  integer(bytes, endpoint.hierarchy_generation);
  for (bool periodic : endpoint.periodicity)
    integer(bytes, periodic);
  integer(bytes, levels.size());
  for (const auto& level : levels) {
    integer(bytes, level.level);
    integer(bytes, level.components);
    integer(bytes, level.activity);
    integer(bytes, level.relative_measure);
    integer(bytes, level.distribution.replicated());
    for (int axis = 0; axis < Dim; ++axis) {
      integer(bytes, level.geometry.domain().lo[axis]);
      integer(bytes, level.geometry.domain().hi[axis]);
      real(bytes, level.geometry.lower()[axis]);
      real(bytes, level.geometry.upper()[axis]);
      integer(bytes, level.distribution.rank_space().origin()[axis]);
      integer(bytes, level.distribution.rank_space().extent()[axis]);
      integer(bytes, level.ghosts[axis]);
      integer(bytes, level.coverage_ghosts[axis]);
      integer(bytes, level.activity_ghosts[axis]);
      integer(bytes, level.measure_ghosts[axis]);
    }
    integer(bytes, level.layout.size());
    for (std::size_t patch = 0; patch < level.layout.size(); ++patch)
      for (int axis = 0; axis < Dim; ++axis) {
        integer(bytes, level.layout[patch].lo[axis]);
        integer(bytes, level.layout[patch].hi[axis]);
        integer(bytes, level.distribution.replicated() ? 0 : level.distribution.owner(patch)[axis]);
      }
  }
  return bytes;
}

template <int Dim>
bool finest_owner(const std::vector<LevelContract<Dim>>& levels, std::size_t level,
                  const Index<Dim>& cell) {
  if (level + 1 == levels.size())
    return true;
  const auto& coarse = levels[level].geometry.domain();
  const auto& fine = levels[level + 1].geometry.domain();
  Index<Dim> child{};
  for (int axis = 0; axis < Dim; ++axis)
    child[axis] =
        static_cast<int>(fine.lo[axis] + (static_cast<std::int64_t>(cell[axis]) - coarse.lo[axis]) *
                                             (fine.length(axis) / coarse.length(axis)));
  for (const auto& patch : levels[level + 1].layout.boxes())
    if (patch.contains(child))
      return false;
  return true;
}

template <int Dim>
void read_fab(const Fab<Dim>& fab, std::vector<double>& values) {
  if (values.size() != fab.size())
    throw std::invalid_argument("AMR transfer host staging extent changed");
  using Device = Kokkos::View<const Real*, typename Fab<Dim>::memory_space,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using Host = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  Kokkos::deep_copy(Host(values.data(), values.size()), Device(fab.view().data, fab.size()));
}

template <int Dim>
void write_fab(Fab<Dim>& fab, const std::vector<double>& values) {
  if (values.size() != fab.size())
    throw std::invalid_argument("AMR transfer host publication extent changed");
  using Device =
      Kokkos::View<Real*, typename Fab<Dim>::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using Host =
      Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  Kokkos::deep_copy(Device(fab.view().data, fab.size()), Host(values.data(), values.size()));
}

template <int Dim>
PopsConstFieldViewV1 source_abi(const std::vector<double>& values, const Box<Dim>& box,
                                int components, const char* layout_identity,
                                const char* patch_identity) {
  PopsConstFieldViewV1 view{};
  view.struct_size = sizeof(view);
  view.data = values.data();
  view.dimension = Dim;
  view.component_count = static_cast<std::size_t>(components);
  view.component_stride = static_cast<std::ptrdiff_t>(cells(box));
  std::size_t stride = 1;
  for (int axis = 0; axis < 3; ++axis) {
    view.extents[axis] = axis < Dim ? static_cast<std::size_t>(box.length(axis)) : 1;
    view.axis_strides[axis] = axis < Dim ? static_cast<std::ptrdiff_t>(stride) : 0;
    stride *= view.extents[axis];
  }
  view.scalar_type = POPS_SCALAR_FLOAT64_V1;
  view.memory_space = POPS_MEMORY_SPACE_HOST_V1;
  view.centering = POPS_FIELD_CENTERING_CELL_V1;
  view.ownership = POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1;
  view.layout_identity = layout_identity;
  view.patch_identity = patch_identity;
  return view;
}

PopsFieldViewV1 scalar_abi(std::vector<double>& values, int dimension, const char* layout_identity,
                           const char* patch_identity) {
  PopsFieldViewV1 view{};
  view.struct_size = sizeof(view);
  view.data = values.data();
  view.dimension = dimension;
  view.component_count = values.size();
  view.component_stride = 1;
  for (int axis = 0; axis < 3; ++axis) {
    view.extents[axis] = 1;
    view.axis_strides[axis] = axis < dimension ? 1 : 0;
  }
  view.scalar_type = POPS_SCALAR_FLOAT64_V1;
  view.memory_space = POPS_MEMORY_SPACE_HOST_V1;
  view.centering = POPS_FIELD_CENTERING_CELL_V1;
  view.ownership = POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1;
  view.layout_identity = layout_identity;
  view.patch_identity = patch_identity;
  return view;
}

}  // namespace

template <int Dim>
struct PreparedAmrLayoutTransfer<Dim>::Impl {
  struct Contribution {
    std::size_t source_level, source_patch, target_level, target_patch;
    Index<Dim> target_cell;
    Box<Dim> source_region;
    Index<Dim> owner;
    std::string source_patch_identity, target_patch_identity;
    std::array<std::size_t, Dim> weight_offsets{};
    std::vector<double> weights;
    std::vector<double> captured;
    std::vector<double> result;
    std::size_t target_storage = std::numeric_limits<std::size_t>::max();
  };
  struct PatchStorage {
    std::size_t level, patch, flat;
    std::vector<double> values, coverage, activity, measure, publication, accumulation;
  };
  AmrPhysicalTransferSpec<Dim> spec;
  std::shared_ptr<component::LoadedComponent> provider_component;
  component::LoadedComponent::PreparedState provider_state;
  const PopsTransferApiV2* provider_api = nullptr;
  SystemLayoutTransferExecution execution;
  std::vector<LevelContract<Dim>> source_levels, target_levels;
  std::string source_contract, target_contract;
  std::string source_hierarchy, target_hierarchy;
  std::uint64_t source_epoch = 0, target_epoch = 0;
  std::string source_stage;
  std::uint64_t source_stage_generation = 0;
  std::optional<ExecutionLane> lane;
  MultiFab<Dim> packed_source, captured_source;
  std::vector<PatchStorage> source_storage, target_storage;
  std::vector<Contribution> contributions;
  std::optional<mesh::parallel::RegionTransport<Dim>> transport;
  int components = 0;
  std::size_t transported = 0, bytes = 0;
  std::uint64_t generation = 0, last_generation = 0, captured_attempt = 0;
  std::uint64_t source_active = 0;
  bool applied = false;

  void budget_bytes(std::size_t count, std::size_t width = sizeof(double)) {
    add(bytes, multiply(count, width), spec.budget.prepared_bytes,
        "AMR transfer prepared byte budget exceeded");
  }

  Impl(const AmrTransferEndpoint<Dim>& source, const AmrTransferEndpoint<Dim>& target,
       AmrPhysicalTransferSpec<Dim> contract, std::shared_ptr<component::LoadedComponent> provider,
       SystemLayoutTransferExecution execution_contract, const ExecutionLane& authority)
      : spec(std::move(contract)),
        provider_component(std::move(provider)),
        execution(std::move(execution_contract)),
        source_levels(endpoint_contract(source, authority)),
        target_levels(endpoint_contract(target, authority)),
        source_contract(endpoint_bytes(source, source_levels)),
        target_contract(endpoint_bytes(target, target_levels)),
        source_hierarchy(source.hierarchy_identity),
        target_hierarchy(target.hierarchy_identity),
        source_epoch(source.hierarchy_generation),
        target_epoch(target.hierarchy_generation) {
    components = source_levels.front().components;
    validate_execution(execution, authority);
    if (!provider_component)
      throw std::invalid_argument("AMR physical transfer requires a loaded provider");
    const auto& api = provider_component->api();
    if (!api.component_id || !api.manifest_identity ||
        spec.authentication.provider_component_identity != api.component_id ||
        spec.authentication.provider_manifest_identity != api.manifest_identity)
      throw std::invalid_argument(
          "AMR physical provider identity differs from its authenticated component");
    provider_api = &provider_component->template table<PopsTransferApiV2>(
        POPS_NATIVE_INTERFACE_TRANSFER_V2, 2u);
    if (!provider_api->apply || !provider_api->apply_integral)
      throw std::invalid_argument(
          "AMR physical provider lacks the indivisible Transfer V2 operations");
    require_text(spec.physical_contract_identity);
    validate_spec(source, target);
    prepare_storage(source, target);
    prepare_contributions();
  }

  void validate_spec(const AmrTransferEndpoint<Dim>& source,
                     const AmrTransferEndpoint<Dim>& target) {
    const auto& map = spec.authentication;
    for (const auto* id :
         {&map.mapping_identity, &map.provider_identity, &map.provider_component_identity,
          &map.provider_manifest_identity, &map.source_block, &map.target_block})
      require_text(*id);
    if (!map.physical_contract || map.source_layout_identity != source.layout_identity ||
        map.target_layout_identity != target.layout_identity ||
        source.layout_identity == target.layout_identity ||
        components != target_levels.front().components ||
        map.source_representation != "pops://representations/cell-average@1" ||
        map.target_representation != "pops://representations/cell-average@1" ||
        spec.quadrature_identity != "pops://measure/piecewise-constant-base-bins@1")
      throw std::invalid_argument(
          "AMR physical transfer authentication/representation is incomplete");
    if (map.synchronization_identity != "pops://synchronization/before-step@1" &&
        map.synchronization_identity != "pops://synchronization/after-source-step@1")
      throw std::invalid_argument("AMR physical synchronization identity is unknown");
    const bool reduce = map.operation == POPS_TRANSFER_OPERATION_VELOCITY_MOMENT_V1;
    if (!reduce && map.operation != POPS_TRANSFER_OPERATION_PHYSICAL_PULLBACK_V1)
      throw std::invalid_argument("AMR physical transfer operation is unknown");
    if (source_levels.front().distribution.rank_space() !=
        target_levels.front().distribution.rank_space())
      throw std::invalid_argument("AMR transfer source and target rank spaces differ");
    int source_active_axes = 0, target_active_axes = 0, shared = 0;
    std::array<bool, Dim> used{};
    for (int axis = 0; axis < Dim; ++axis) {
      const int source_flag = map.physical_source_active[axis],
                target_flag = map.physical_target_active[axis];
      if ((source_flag != 0 && source_flag != 1) || (target_flag != 0 && target_flag != 1) ||
          map.refinement_ratio[axis] != 1)
        throw std::invalid_argument("AMR physical axis flags or identity ratio are invalid");
      source_active_axes += source_flag;
      target_active_axes += target_flag;
      const int target_axis = map.physical_source_to_target[axis];
      if (target_axis < -1 || target_axis >= Dim || (!source_flag && target_axis != -1))
        throw std::invalid_argument("AMR physical source axis map is invalid");
      if (target_axis >= 0) {
        if (used[target_axis] || map.physical_target_active[target_axis] != 1)
          throw std::invalid_argument("AMR physical retained axes must be injective and active");
        used[target_axis] = true;
        ++shared;
        const auto& source_geometry = source_levels.front().geometry;
        const auto& target_geometry = target_levels.front().geometry;
        if (source_geometry.lower()[axis] != target_geometry.lower()[target_axis] ||
            source_geometry.upper()[axis] != target_geometry.upper()[target_axis] ||
            source.periodicity[axis] != target.periodicity[target_axis])
          throw std::invalid_argument("AMR physical retained axis domains differ");
      }
      const bool eliminated = source_flag && target_axis < 0;
      if (eliminated) {
        if (!reduce || spec.base_bin_weights[axis].empty() ||
            spec.base_bin_lower[axis] != source_levels.front().geometry.lower()[axis] ||
            spec.base_bin_upper[axis] != source_levels.front().geometry.upper()[axis])
          throw std::invalid_argument("AMR physical eliminated axis has no exact base-bin measure");
        for (double weight : spec.base_bin_weights[axis])
          if (!std::isfinite(weight))
            throw std::invalid_argument("AMR base-bin measure is non-finite");
      } else if (!spec.base_bin_weights[axis].empty()) {
        throw std::invalid_argument(
            "AMR physical quadrature must belong to an eliminated source axis");
      }
      for (const auto* levels : {&source_levels, &target_levels}) {
        const int active = levels == &source_levels ? source_flag : target_flag;
        if (!active)
          for (const auto& level : *levels)
            if (level.geometry.domain().length(axis) != 1 || level.geometry.lower()[axis] != 0.0 ||
                level.geometry.upper()[axis] != 1.0 ||
                !(levels == &source_levels ? source.periodicity[axis] : target.periodicity[axis]))
              throw std::invalid_argument(
                  "AMR hidden embedding axes require neutral unit singletons");
      }
    }
    if (reduce ? (source_active_axes <= target_active_axes || shared != target_active_axes)
               : (target_active_axes <= source_active_axes || shared != source_active_axes))
      throw std::invalid_argument("AMR physical transfer supports must be strictly nested");
  }

  void account_patch(const LevelContract<Dim>& contract, const Box<Dim>& box, bool source) {
    const auto grown_cells = [&](const Extent<Dim>& ghosts) {
      Box<Dim> grown = box;
      for (int axis = 0; axis < Dim; ++axis)
        grown = grown.grow(axis, ghosts[axis]);
      return cells(grown);
    };
    // Upper bound over all ranks, including host staging, packed fields and RegionTransport's
    // device/pinned pack storage. Local allocations never exceed this deterministic contract.
    budget_bytes(multiply(grown_cells(contract.ghosts), static_cast<std::size_t>(components)));
    budget_bytes(grown_cells(contract.coverage_ghosts));
    if (contract.activity)
      budget_bytes(grown_cells(contract.activity_ghosts));
    if (contract.relative_measure)
      budget_bytes(grown_cells(contract.measure_ghosts));
    budget_bytes(multiply(cells(box), static_cast<std::size_t>(components)),
                 (source ? 2 : 1) * sizeof(double));
    budget_bytes(1, sizeof(PatchStorage) + sizeof(Box<Dim>) + sizeof(Index<Dim>));
  }

  void prepare_storage(const AmrTransferEndpoint<Dim>& source,
                       const AmrTransferEndpoint<Dim>& target) {
    std::vector<Box<Dim>> boxes;
    std::vector<Index<Dim>> owners;
    for (std::size_t level = 0; level < source_levels.size(); ++level) {
      const auto& contract = source_levels[level];
      for (std::size_t patch = 0; patch < contract.layout.size(); ++patch) {
        const auto flat = boxes.size();
        boxes.push_back(contract.layout[patch]);
        owners.push_back(contract.distribution.replicated() ? contract.rank
                                                            : contract.distribution.owner(patch));
        account_patch(contract, contract.layout[patch], true);
        if (!source.levels[level].state->contains_local(patch))
          continue;
        const auto& view = source.levels[level];
        PatchStorage stored{level, patch, flat};
        stored.values.resize(view.state->fab_global(patch).size());
        stored.coverage.resize(view.coverage->fab_global(patch).size());
        if (view.activity)
          stored.activity.resize(view.activity->fab_global(patch).size());
        if (view.relative_measure)
          stored.measure.resize(view.relative_measure->fab_global(patch).size());
        stored.publication.resize(
            multiply(cells(contract.layout[patch]), static_cast<std::size_t>(components)));
        source_storage.push_back(std::move(stored));
      }
    }
    const auto& rank_space = source_levels.front().distribution.rank_space();
    mesh::BoxArray<Dim> layout(std::move(boxes));
    packed_source = MultiFab<Dim>(
        layout, mesh::Distribution<Dim>::partitioned(layout, rank_space, std::move(owners)),
        source_levels.front().rank, components, Extent<Dim>{});
    for (std::size_t level = 0; level < target_levels.size(); ++level) {
      const auto& contract = target_levels[level];
      for (std::size_t patch = 0; patch < contract.layout.size(); ++patch) {
        const auto& view = target.levels[level];
        account_patch(contract, contract.layout[patch], false);
        if (!view.state->contains_local(patch))
          continue;
        PatchStorage stored{level, patch, 0};
        stored.coverage.resize(view.coverage->fab_global(patch).size());
        if (view.activity)
          stored.activity.resize(view.activity->fab_global(patch).size());
        if (view.relative_measure)
          stored.measure.resize(view.relative_measure->fab_global(patch).size());
        stored.publication.resize(view.state->fab_global(patch).size());
        stored.accumulation.resize(
            multiply(cells(contract.layout[patch]), static_cast<std::size_t>(components)));
        target_storage.push_back(std::move(stored));
      }
    }
  }

  double bin_weight(std::size_t axis, std::uint64_t cell, std::uint64_t source_cells) const {
    const auto& weights = spec.base_bin_weights[axis];
    const auto bins = static_cast<std::uint64_t>(weights.size());
    const auto lattice = common_lattice(source_cells, bins);
    const auto width = lattice / source_cells, bin_width = lattice / bins;
    const auto lower = cell * width, upper = (cell + 1) * width;
    const auto first = lower / bin_width, last = (upper - 1) / bin_width;
    long double sum = 0;
    for (auto bin = first; bin <= last; ++bin) {
      const auto overlap =
          std::min(upper, (bin + 1) * bin_width) - std::max(lower, bin * bin_width);
      sum += static_cast<long double>(weights[bin]) * overlap / bin_width;
    }
    const double result = static_cast<double>(sum);
    if (!std::isfinite(result) || (sum != 0 && result == 0))
      throw std::overflow_error("AMR integrated base-bin weight is not representable");
    return result;
  }

  void prepare_contributions() {
    std::vector<Box<Dim>> carriers;
    std::vector<Index<Dim>> owners;
    std::vector<mesh::parallel::RegionTransferJob<Dim>> jobs;
    std::size_t destinations = 0, probes = 0;
    for (std::size_t tl = 0; tl < target_levels.size(); ++tl) {
      const auto& target = target_levels[tl];
      for (std::size_t tp = 0; tp < target.layout.size(); ++tp) {
        const auto target_owner =
            target.distribution.replicated() ? target.rank : target.distribution.owner(tp);
        for (std::size_t ordinal = 0; ordinal < cells(target.layout[tp]); ++ordinal) {
          const auto target_cell = cell_at(target.layout[tp], ordinal);
          if (!finest_owner(target_levels, tl, target_cell))
            continue;
          add(destinations, 1, spec.budget.destination_cells,
              "AMR transfer destination-cell budget exceeded");
          std::size_t source_flat = 0;
          for (std::size_t sl = 0; sl < source_levels.size(); ++sl) {
            const auto& source = source_levels[sl];
            for (std::size_t sp = 0; sp < source.layout.size(); ++sp, ++source_flat) {
              add(probes, 1, spec.budget.intersection_probes,
                  "AMR transfer intersection-probe budget exceeded");
              Box<Dim> region = source.layout[sp];
              for (int axis = 0; axis < Dim; ++axis) {
                const int target_axis = spec.authentication.physical_source_to_target[axis];
                if (target_axis < 0)
                  continue;
                const auto source_count =
                    static_cast<std::uint64_t>(source.geometry.domain().length(axis));
                const auto target_count =
                    static_cast<std::uint64_t>(target.geometry.domain().length(target_axis));
                const auto coordinate =
                    static_cast<std::uint64_t>(static_cast<std::int64_t>(target_cell[target_axis]) -
                                               target.geometry.domain().lo[target_axis]);
                const auto lattice = common_lattice(source_count, target_count);
                const auto lower = coordinate * (lattice / target_count);
                const auto upper = (coordinate + 1) * (lattice / target_count);
                const auto lo = source.geometry.domain().lo[axis] +
                                static_cast<std::int64_t>(lower / (lattice / source_count));
                const auto hi = source.geometry.domain().lo[axis] +
                                static_cast<std::int64_t>((upper - 1) / (lattice / source_count));
                region.lo[axis] = std::max(region.lo[axis], static_cast<int>(lo));
                region.hi[axis] = std::min(region.hi[axis], static_cast<int>(hi));
              }
              if (region.empty())
                continue;
              if (jobs.size() >= spec.budget.canonical_jobs)
                throw std::length_error("AMR transfer canonical-job budget exceeded");
              std::size_t weight_count = 0;
              for (int axis = 0; axis < Dim; ++axis)
                add(weight_count, static_cast<std::size_t>(region.length(axis)),
                    std::numeric_limits<std::size_t>::max(), "AMR tensor weight count overflow");
              budget_bytes(weight_count);
              const auto elements = multiply(cells(region), static_cast<std::size_t>(components));
              add(transported, elements, spec.budget.transported_elements,
                  "AMR transfer transport-element budget exceeded");
              budget_bytes(elements, 6 * sizeof(double));
              budget_bytes(static_cast<std::size_t>(components));
              budget_bytes(1, sizeof(Contribution) +
                                  16 * sizeof(mesh::parallel::RegionTransferJob<Dim>) +
                                  2 * sizeof(Box<Dim>) + 2 * sizeof(Index<Dim>));
              Contribution contribution{sl, sp, tl, tp, target_cell, region, target_owner};
              contribution.source_patch_identity = spec.authentication.source_block +
                                                   "::level::" + std::to_string(sl) +
                                                   "::patch::" + std::to_string(sp);
              contribution.target_patch_identity =
                  spec.authentication.target_block + "::level::" + std::to_string(tl) +
                  "::patch::" + std::to_string(tp) + "::cell::" + std::to_string(ordinal);
              budget_bytes(contribution.source_patch_identity.size() + 1, sizeof(char));
              budget_bytes(contribution.target_patch_identity.size() + 1, sizeof(char));
              contribution.weights.reserve(weight_count);
              for (int axis = 0; axis < Dim; ++axis) {
                contribution.weight_offsets[axis] = contribution.weights.size();
                const int target_axis = spec.authentication.physical_source_to_target[axis];
                const auto source_count =
                    static_cast<std::uint64_t>(source.geometry.domain().length(axis));
                for (std::int64_t index = region.lo[axis]; index <= region.hi[axis]; ++index) {
                  const auto source_cell =
                      static_cast<std::uint64_t>(index - source.geometry.domain().lo[axis]);
                  double weight = 1;
                  if (target_axis >= 0)
                    weight = overlap_fraction(
                        source_cell, source_count,
                        static_cast<std::uint64_t>(
                            static_cast<std::int64_t>(target_cell[target_axis]) -
                            target.geometry.domain().lo[target_axis]),
                        static_cast<std::uint64_t>(target.geometry.domain().length(target_axis)));
                  else if (spec.authentication.physical_source_active[axis])
                    weight = bin_weight(static_cast<std::size_t>(axis), source_cell, source_count);
                  contribution.weights.push_back(weight);
                }
              }
              if (target_owner == target.rank) {
                contribution.captured.resize(elements);
                contribution.result.resize(static_cast<std::size_t>(components));
                const auto found = std::find_if(
                    target_storage.begin(), target_storage.end(),
                    [&](const auto& patch) { return patch.level == tl && patch.patch == tp; });
                if (found == target_storage.end())
                  throw std::logic_error("AMR target storage was not prepared");
                contribution.target_storage =
                    static_cast<std::size_t>(found - target_storage.begin());
              }
              const auto carrier = carriers.size();
              carriers.push_back(region);
              owners.push_back(target_owner);
              jobs.push_back(
                  {source_flat, carrier,
                   source.distribution.replicated() ? source.rank : source.distribution.owner(sp),
                   target_owner, region, region});
              contributions.push_back(std::move(contribution));
            }
          }
        }
      }
    }
    mesh::BoxArray<Dim> layout(std::move(carriers));
    const auto& space = source_levels.front().distribution.rank_space();
    captured_source = MultiFab<Dim>(
        layout, mesh::Distribution<Dim>::partitioned(layout, space, std::move(owners)),
        source_levels.front().rank, components, Extent<Dim>{});
    transport.emplace(mesh::parallel::RegionTransferPlan<Dim>{
        space,
        source_levels.front().rank,
        components,
        std::move(jobs),
        {contributions.size(), space.size(), transported, transported, transported}});
  }

  std::string exact_contract() const {
    std::string result = source_contract;
    text(result, target_contract);
    text(result, spec.physical_contract_identity);
    for (const auto* id :
         {&execution.execution_identity, &execution.backend_identity, &execution.device_identity,
          &execution.stream_identity, &execution.communicator_identity,
          &execution.communicator_datatype_identity})
      text(result, *id);
    integer(result, execution.context_version);
    integer(result, execution.memory_space);
    integer(result, execution.scalar_type);
    integer(result, execution.storage_precision);
    integer(result, execution.compute_precision);
    integer(result, execution.accumulation_precision);
    integer(result, execution.reduction_precision);
    const auto& api = provider_component->api();
    text(result, api.semantic_identity ? api.semantic_identity : "");
    text(result, api.catalog_sha256 ? api.catalog_sha256 : "");
    text(result, api.abi_key ? api.abi_key : "");
    const auto& a = spec.authentication;
    for (const auto* id :
         {&a.mapping_identity, &a.provider_identity, &a.provider_component_identity,
          &a.provider_manifest_identity, &a.source_layout_identity, &a.target_layout_identity,
          &a.source_block, &a.target_block, &a.source_representation, &a.target_representation,
          &a.synchronization_identity, &spec.quadrature_identity})
      text(result, *id);
    text(result, a.program_invocation);
    integer(result, a.operation);
    integer(result, a.physical_contract);
    for (int axis = 0; axis < Dim; ++axis) {
      integer(result, a.refinement_ratio[axis]);
      integer(result, a.physical_source_to_target[axis]);
      integer(result, a.physical_source_active[axis]);
      integer(result, a.physical_target_active[axis]);
      real(result, spec.base_bin_lower[axis]);
      real(result, spec.base_bin_upper[axis]);
      integer(result, spec.base_bin_weights[axis].size());
      for (double weight : spec.base_bin_weights[axis])
        real(result, weight);
    }
    for (auto value :
         {spec.budget.destination_cells, spec.budget.intersection_probes,
          spec.budget.canonical_jobs, spec.budget.transported_elements, spec.budget.prepared_bytes})
      integer(result, value);
    return result;
  }

  void require_attempt(std::uint64_t supplied_generation, std::uint64_t attempt) const {
    if (!generation || supplied_generation != generation || !attempt)
      throw std::logic_error("AMR transfer operation crossed its active generation/attempt");
  }

  void validate_endpoint(const AmrTransferEndpoint<Dim>& endpoint, bool source) const {
    const auto& levels = source ? source_levels : target_levels;
    require_text(endpoint.stage_identity);
    if (endpoint.levels.size() != levels.size() ||
        endpoint_bytes(endpoint, levels) != (source ? source_contract : target_contract))
      throw std::invalid_argument(
          "AMR transfer hierarchy changed; reprepare after topology/restart publication");
    // Compare current stage storage to the already-proved graph without rebuilding intersections,
    // copying layouts, or repeating the quadratic nesting/disjointness proof on every step.
    for (std::size_t level = 0; level < levels.size(); ++level) {
      const auto& current = endpoint.levels[level];
      const auto& expected = levels[level];
      if (current.level != expected.level || current.geometry != expected.geometry ||
          !current.state || !current.coverage ||
          (current.activity != nullptr) != expected.activity ||
          (current.relative_measure != nullptr) != expected.relative_measure)
        throw std::invalid_argument("AMR transfer stage no longer matches its prepared level");
      const auto check = [&](const MultiFab<Dim>* field, int width, const Extent<Dim>& ghosts) {
        if (!field || field->layout() != expected.layout ||
            field->distribution() != expected.distribution ||
            field->local_rank() != expected.rank || field->ncomp() != width ||
            field->ghosts() != ghosts)
          throw std::invalid_argument("AMR transfer stage field storage changed after preparation");
      };
      check(current.state, expected.components, expected.ghosts);
      check(current.coverage, 1, expected.coverage_ghosts);
      if (current.activity)
        check(current.activity, 1, expected.activity_ghosts);
      if (current.relative_measure)
        check(current.relative_measure, 1, expected.measure_ghosts);
    }
  }

  void agree_stage(const AmrTransferEndpoint<Dim>& endpoint, std::uint64_t attempt) const {
    std::string stage;
    collective(*lane, "AMR transfer stage serialization", [&] {
      text(stage, endpoint.stage_identity);
      integer(stage, endpoint.stage_generation);
      integer(stage, generation);
      integer(stage, attempt);
    });
    agree(*lane, "amr-layout-transfer-stage", stage);
  }

  double cell_measure(const AmrTransferLevelView<Dim>& view, const PatchStorage& stored,
                      const Index<Dim>& cell, bool source) const {
    const auto& levels = source ? source_levels : target_levels;
    const auto coverage_offset = offset(view.coverage->fab_global(stored.patch).grown_box(), cell);
    const double coverage = stored.coverage[coverage_offset];
    if (coverage != (finest_owner(levels, stored.level, cell) ? 1.0 : 0.0))
      throw std::invalid_argument(
          "AMR transfer coverage mask disagrees with finest-owner topology");
    double active = 1, measure = 1;
    if (view.activity)
      active = stored.activity[offset(view.activity->fab_global(stored.patch).grown_box(), cell)];
    if (view.relative_measure)
      measure =
          stored.measure[offset(view.relative_measure->fab_global(stored.patch).grown_box(), cell)];
    if ((active != 0 && active != 1) || !std::isfinite(measure) || measure < 0 || measure > 1 ||
        (active == 1 && measure == 0))
      throw std::invalid_argument("AMR transfer cell activity/relative measure is invalid");
    return coverage == 0 ? 0 : active * measure;
  }

  std::size_t active_element_census(const AmrTransferEndpoint<Dim>& endpoint, bool source) const {
    std::size_t result = 0;
    for (std::size_t level = 0; level < endpoint.levels.size(); ++level) {
      const auto& view = endpoint.levels[level];
      for (std::size_t local = 0; local < view.state->local_size(); ++local) {
        const auto patch = view.state->global_index(local);
        PatchStorage masks{level, patch, 0};
        masks.coverage.resize(view.coverage->fab_global(patch).size());
        if (view.activity)
          masks.activity.resize(view.activity->fab_global(patch).size());
        if (view.relative_measure)
          masks.measure.resize(view.relative_measure->fab_global(patch).size());
        read_masks(view, masks);
        const auto& box = view.state->fab(local).box();
        for (std::size_t ordinal = 0; ordinal < cells(box); ++ordinal)
          if (cell_measure(view, masks, cell_at(box, ordinal), source) > 0)
            add(result, static_cast<std::size_t>(components),
                std::numeric_limits<std::size_t>::max(),
                "AMR expected active element count overflow");
      }
    }
    return result;
  }

  AmrLayoutTransferReceipt receipt_contract(std::uint64_t source_count,
                                            std::uint64_t destination_count,
                                            const std::string& source_stage_identity,
                                            std::uint64_t source_stage_epoch,
                                            const std::string& target_stage_identity,
                                            std::uint64_t target_stage_epoch) const {
    AmrLayoutTransferReceipt receipt;
    const auto& auth = spec.authentication;
    auto& transfer = receipt.transfer;
    transfer.program_invocation = auth.program_invocation;
    transfer.mapping_identity = auth.mapping_identity;
    transfer.provider_identity = auth.provider_identity;
    transfer.provider_component_identity = auth.provider_component_identity;
    transfer.provider_manifest_identity = auth.provider_manifest_identity;
    transfer.source_layout_identity = auth.source_layout_identity;
    transfer.target_layout_identity = auth.target_layout_identity;
    transfer.source_block = auth.source_block;
    transfer.target_block = auth.target_block;
    transfer.execution_identity = execution.execution_identity;
    transfer.operation = auth.operation;
    transfer.generation = generation;
    receipt.source_active_elements = source_count;
    receipt.destination_active_elements = destination_count;
    transfer.source_element_count = source_count;
    transfer.destination_element_count = destination_count;
    receipt.physical_contract_identity = spec.physical_contract_identity;
    receipt.source_hierarchy_identity = source_hierarchy;
    receipt.target_hierarchy_identity = target_hierarchy;
    receipt.source_hierarchy_generation = source_epoch;
    receipt.target_hierarchy_generation = target_epoch;
    receipt.source_stage_identity = source_stage_identity;
    receipt.source_stage_generation = source_stage_epoch;
    receipt.target_stage_identity = target_stage_identity;
    receipt.target_stage_generation = target_stage_epoch;
    receipt.canonical_jobs = contributions.size();
    receipt.transported_elements = transported;
    receipt.prepared_bytes = bytes;
    return receipt;
  }

  void read_masks(const AmrTransferLevelView<Dim>& view, PatchStorage& storage) const {
    read_fab(view.coverage->fab_global(storage.patch), storage.coverage);
    if (view.activity)
      read_fab(view.activity->fab_global(storage.patch), storage.activity);
    if (view.relative_measure)
      read_fab(view.relative_measure->fab_global(storage.patch), storage.measure);
  }
};

template <int Dim>
PreparedAmrLayoutTransfer<Dim>::PreparedAmrLayoutTransfer(std::unique_ptr<Impl> impl) noexcept
    : p_(std::move(impl)) {}
template <int Dim>
PreparedAmrLayoutTransfer<Dim>::~PreparedAmrLayoutTransfer() = default;

template <int Dim>
std::shared_ptr<PreparedAmrLayoutTransfer<Dim>> PreparedAmrLayoutTransfer<Dim>::prepare(
    const AmrTransferEndpoint<Dim>& source, const AmrTransferEndpoint<Dim>& target,
    AmrPhysicalTransferSpec<Dim> spec,
    std::shared_ptr<component::LoadedComponent> provider_component,
    SystemLayoutTransferExecution execution, const ExecutionLane& authority) {
  std::shared_ptr<PreparedAmrLayoutTransfer> result;
  collective(authority, "AMR transfer preparation", [&] {
    auto pending =
        std::make_unique<Impl>(source, target, std::move(spec), std::move(provider_component),
                               std::move(execution), authority);
    // Allocate both the facade and its reference control block before any private MPI lane
    // exists. Allocation failure can then reach the authority consensus without MPI teardown.
    result = std::shared_ptr<PreparedAmrLayoutTransfer>(
        new PreparedAmrLayoutTransfer(std::move(pending)));
  });
  auto& prepared = *result->p_;
  std::string contract;
  collective(authority, "AMR transfer contract serialization",
             [&] { contract = prepared.exact_contract(); });
  agree(authority, "amr-physical-layout-transfer-v1", contract);
  std::optional<component::LoadedComponent::PreparedStateRequest> provider_request;
  collective(authority, "AMR Transfer V2 provider request", [&] {
    provider_request.emplace(prepared.provider_component->prepare_state_request(
        POPS_NATIVE_INTERFACE_TRANSFER_V2, 2u, execution_view(prepared.execution)));
  });
  collective(authority, "AMR Transfer V2 provider preparation", [&] {
    prepared.provider_state =
        prepared.provider_component->execute_prepared_state(std::move(*provider_request));
  });
  prepared.lane.emplace(ExecutionLane::duplicate_collectively(
      authority, prepared.spec.authentication.mapping_identity));
  prepared.transport->prepare_collectively(*prepared.lane);
  return result;
}

template <int Dim>
AmrLayoutTransferReceipt PreparedAmrLayoutTransfer<Dim>::expected_receipt_contract(
    const AmrTransferEndpoint<Dim>& source, const AmrTransferEndpoint<Dim>& target) const {
  std::size_t local_source = 0, local_target = 0;
  collective(*p_->lane, "AMR expected receipt endpoint census", [&] {
    p_->validate_endpoint(source, true);
    p_->validate_endpoint(target, false);
    local_source = p_->active_element_census(source, true);
    local_target = p_->active_element_census(target, false);
  });
  p_->agree_stage(source, 0);
  p_->agree_stage(target, 0);
  const auto source_count = global_count(local_source, *p_->lane);
  const auto target_count = global_count(local_target, *p_->lane);
  AmrLayoutTransferReceipt result;
  collective(*p_->lane, "AMR expected receipt provenance", [&] {
    result = p_->receipt_contract(source_count, target_count, source.stage_identity,
                                  source.stage_generation, target.stage_identity,
                                  target.stage_generation);
  });
  return result;
}

template <int Dim>
void PreparedAmrLayoutTransfer<Dim>::begin_transaction(std::uint64_t generation) {
  collective(*p_->lane, "AMR transfer begin", [&] {
    if (p_->generation || !generation || generation <= p_->last_generation)
      throw std::logic_error("AMR transfer generation must be positive, monotonic and inactive");
  });
  std::string value;
  collective(*p_->lane, "AMR transfer generation serialization",
             [&] { integer(value, generation); });
  agree(*p_->lane, "amr-transfer-generation", value);
  p_->generation = generation;
  p_->captured_attempt = 0;
  p_->applied = false;
}

template <int Dim>
void PreparedAmrLayoutTransfer<Dim>::capture(const AmrTransferEndpoint<Dim>& source,
                                             std::uint64_t generation, std::uint64_t attempt) {
  collective(*p_->lane, "AMR transfer capture preflight", [&] {
    p_->require_attempt(generation, attempt);
    if (p_->applied || (p_->captured_attempt && p_->captured_attempt != attempt))
      throw std::logic_error("AMR transfer capture requires an unapplied exact attempt");
    p_->validate_endpoint(source, true);
  });
  p_->agree_stage(source, attempt);
  std::size_t active_elements = 0;
  collective(*p_->lane, "AMR transfer source packing", [&] {
    for (auto& stored : p_->source_storage) {
      const auto& view = source.levels[stored.level];
      const auto& fab = view.state->fab_global(stored.patch);
      read_fab(fab, stored.values);
      p_->read_masks(view, stored);
      const auto count = cells(fab.box()), source_stride = cells(fab.grown_box());
      for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
        const auto cell = cell_at(fab.box(), ordinal);
        const double measure = p_->cell_measure(view, stored, cell, true);
        if (measure > 0)
          add(active_elements, static_cast<std::size_t>(p_->components),
              std::numeric_limits<std::size_t>::max(), "AMR active source count overflow");
        const auto origin = offset(fab.grown_box(), cell);
        for (int component = 0; component < p_->components; ++component) {
          const double value =
              measure == 0 ? 0 : stored.values[component * source_stride + origin] * measure;
          if (!std::isfinite(value))
            throw std::invalid_argument("AMR active physical source is non-finite");
          stored.publication[component * count + ordinal] = value;
        }
      }
      write_fab(p_->packed_source.fab_global(stored.flat), stored.publication);
    }
    Kokkos::fence();
  });
  p_->transport->execute(
      [&](const auto& job) {
        return std::as_const(p_->packed_source).fab_global(job.source_patch).view();
      },
      [&](const auto& job) {
        return p_->captured_source.fab_global(job.destination_patch).view();
      });
  p_->source_active = global_count(active_elements, *p_->lane);
  collective(*p_->lane, "AMR transfer captured provenance", [&] {
    p_->source_stage = source.stage_identity;
    p_->source_stage_generation = source.stage_generation;
  });
  p_->captured_attempt = attempt;
}

template <int Dim>
AmrLayoutTransferReceipt PreparedAmrLayoutTransfer<Dim>::apply(
    const AmrTransferEndpoint<Dim>& target, const std::vector<MultiFab<Dim>*>& candidates,
    std::uint64_t generation, std::uint64_t attempt) {
  collective(*p_->lane, "AMR transfer apply preflight", [&] {
    p_->require_attempt(generation, attempt);
    if (p_->captured_attempt != attempt || p_->applied)
      throw std::logic_error("AMR transfer apply requires the exact unapplied capture");
    p_->validate_endpoint(target, false);
    if (candidates.size() != target.levels.size())
      throw std::invalid_argument("AMR transfer candidates do not span the hierarchy");
    for (std::size_t level = 0; level < candidates.size(); ++level) {
      const auto* candidate = candidates[level];
      const auto* state = target.levels[level].state;
      if (!candidate || candidate == state || candidate->layout() != state->layout() ||
          candidate->distribution() != state->distribution() ||
          candidate->local_rank() != state->local_rank() || candidate->ncomp() != state->ncomp() ||
          candidate->ghosts() != state->ghosts())
        throw std::invalid_argument(
            "AMR transfer candidate must be detached with exact level storage");
      for (std::size_t local = 0; local < candidate->local_size(); ++local)
        if (candidate->fab(local).view().data == state->fab(local).view().data)
          throw std::invalid_argument("AMR transfer candidate aliases accepted/stage target state");
    }
  });
  p_->agree_stage(target, attempt);
  std::size_t active_elements = 0;
  collective(*p_->lane, "AMR transfer intersection integrals", [&] {
    for (auto& stored : p_->target_storage) {
      p_->read_masks(target.levels[stored.level], stored);
      read_fab(candidates[stored.level]->fab_global(stored.patch), stored.publication);
      std::fill(stored.accumulation.begin(), stored.accumulation.end(), 0.0);
    }
    for (std::size_t global = 0; global < p_->contributions.size(); ++global) {
      auto& contribution = p_->contributions[global];
      if (contribution.owner != p_->target_levels.front().rank)
        continue;
      read_fab(p_->captured_source.fab_global(global), contribution.captured);
      PopsTransferIntegralRequestV2 request{};
      request.struct_size = sizeof(request);
      request.source = source_abi(contribution.captured, contribution.source_region, p_->components,
                                  p_->spec.authentication.source_layout_identity.c_str(),
                                  contribution.source_patch_identity.c_str());
      request.destination = scalar_abi(contribution.result, Dim,
                                       p_->spec.authentication.target_layout_identity.c_str(),
                                       contribution.target_patch_identity.c_str());
      request.dimension = Dim;
      request.operation = static_cast<PopsTransferOperationV1>(p_->spec.authentication.operation);
      request.physical_contract_identity = p_->spec.physical_contract_identity.c_str();
      request.axis_weights = contribution.weights.data();
      request.weight_count = contribution.weights.size();
      for (int axis = 0; axis < Dim; ++axis)
        request.weight_offsets[axis] = contribution.weight_offsets[axis];
      request.execution = execution_view(p_->execution);
      PopsComponentStatusV1 status = component::unwritten_component_status();
      int code = 0;
      try {
        code = component::apply_transfer_integral(*p_->provider_api, p_->provider_state.get(),
                                                  request, status);
        Kokkos::fence();
      } catch (...) {
        Kokkos::fence();
        throw;
      }
      if (!component::component_status_is_well_formed(status) || code != 0 || status.code != 0 ||
          status.action != POPS_COMPONENT_CONTINUE_V1)
        throw std::runtime_error(status.reason ? status.reason
                                               : "AMR Transfer V2 integral provider failed");
      auto& stored = p_->target_storage.at(contribution.target_storage);
      const auto& box = p_->target_levels[stored.level].layout[stored.patch];
      const auto count = cells(box), location = offset(box, contribution.target_cell);
      for (int component = 0; component < p_->components; ++component)
        stored.accumulation[component * count + location] += contribution.result[component];
    }
    for (auto& stored : p_->target_storage) {
      const auto& view = target.levels[stored.level];
      const auto& fab = candidates[stored.level]->fab_global(stored.patch);
      const auto count = cells(fab.box()), stride = cells(fab.grown_box());
      for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
        const auto cell = cell_at(fab.box(), ordinal);
        const double measure = p_->cell_measure(view, stored, cell, false);
        if (measure == 0)
          continue;
        add(active_elements, static_cast<std::size_t>(p_->components),
            std::numeric_limits<std::size_t>::max(), "AMR active target count overflow");
        for (int component = 0; component < p_->components; ++component) {
          const double value = stored.accumulation[component * count + ordinal] / measure;
          if (!std::isfinite(value))
            throw std::runtime_error("AMR physical candidate is non-finite");
          stored.publication[component * stride + offset(fab.grown_box(), cell)] = value;
        }
      }
    }
  });
  const auto destination_active_elements = global_count(active_elements, *p_->lane);
  AmrLayoutTransferReceipt receipt;
  collective(*p_->lane, "AMR transfer receipt materialization", [&] {
    receipt = p_->receipt_contract(p_->source_active, destination_active_elements, p_->source_stage,
                                   p_->source_stage_generation, target.stage_identity,
                                   target.stage_generation);
    receipt.transfer.applied = true;
    receipt.transfer.attempt = attempt;
  });
  // Every integral and mask preflight succeeded on every rank before any candidate is changed.
  collective(*p_->lane, "AMR transfer candidate publication", [&] {
    for (auto& stored : p_->target_storage)
      write_fab(candidates[stored.level]->fab_global(stored.patch), stored.publication);
    Kokkos::fence();
  });
  p_->applied = true;
  return receipt;
}

template <int Dim>
void PreparedAmrLayoutTransfer<Dim>::reject_attempt(std::uint64_t generation,
                                                    std::uint64_t attempt) {
  collective(*p_->lane, "AMR transfer reject", [&] {
    p_->require_attempt(generation, attempt);
    if (p_->captured_attempt != attempt)
      throw std::logic_error("AMR transfer rejection does not match capture");
  });
  p_->captured_attempt = 0;
  p_->applied = false;
}

template <int Dim>
void PreparedAmrLayoutTransfer<Dim>::finalize_transaction(std::uint64_t generation) noexcept {
  if (p_->generation != generation)
    return;
  p_->last_generation = generation;
  p_->generation = 0;
  p_->captured_attempt = 0;
  p_->applied = false;
}
template <int Dim>
void PreparedAmrLayoutTransfer<Dim>::rollback_transaction(std::uint64_t generation) noexcept {
  finalize_transaction(generation);
}
template <int Dim>
std::size_t PreparedAmrLayoutTransfer<Dim>::canonical_jobs() const noexcept {
  return p_->contributions.size();
}
template <int Dim>
std::size_t PreparedAmrLayoutTransfer<Dim>::transported_elements() const noexcept {
  return p_->transported;
}
template <int Dim>
std::size_t PreparedAmrLayoutTransfer<Dim>::prepared_bytes() const noexcept {
  return p_->bytes;
}

template class PreparedAmrLayoutTransfer<kNativeDimension>;

}  // namespace pops
