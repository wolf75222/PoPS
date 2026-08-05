/// @file
/// @brief Exact compile-time-ranked AMR facade over runtime::amr::AmrRuntime<Dim>.

#include <pops/runtime/amr_system.hpp>

#include <pops/amr/hierarchy/amr_hierarchy.hpp>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/builders/compiled/generated_amr_system_block.hpp>
#include <pops/runtime/named_field_output.hpp>
#include <pops/runtime/output_piece_collective.hpp>
#include <pops/runtime/program/program_runtime_state.hpp>
#include <pops/runtime/program/profiler.hpp>
#include <pops/runtime/system/system_boundary_registry.hpp>
#include <pops/runtime/system/system_lifecycle.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {
namespace {

inline void require_amr_assembling(const runtime::system::SystemLifecycle& lifecycle,
                                   const char* operation) {
  if (lifecycle.frozen())
    throw std::runtime_error(std::string("AmrSystem::") + operation +
                             ": composition is frozen after bind");
}

template <int Dim>
void validate_amr_config(const AmrSystemConfig<Dim>& config) {
  config.validate_spatial_domain();
  if (config.coordinate_system != runtime_config_detail::cartesian_coordinate_system<Dim>())
    throw std::invalid_argument(
        "AmrSystem Cartesian core requires a dimension-qualified Cartesian provider");
  if (config.level_count < 1)
    throw std::invalid_argument("AmrSystem level_count must be positive");
  if (config.regrid_every < 0)
    throw std::invalid_argument("AmrSystem regrid_every must be non-negative");

  const std::size_t transitions = static_cast<std::size_t>(config.level_count - 1);
  if (config.transition_ratios.size() != transitions ||
      config.transition_buffers.size() != transitions ||
      config.transition_lookaheads.size() != transitions)
    throw std::invalid_argument(
        "AmrSystem transition tables must contain exactly level_count - 1 ranked rows");
  for (std::size_t transition = 0; transition < transitions; ++transition)
    for (int axis = 0; axis < Dim; ++axis) {
      if (config.transition_ratios[transition][axis] < 2 ||
          config.transition_ratios[transition][axis] > std::numeric_limits<int>::max())
        throw std::invalid_argument(
            "AmrSystem transition refinement ratios must be at least two on every axis");
      if (config.transition_buffers[transition][axis] < 0 ||
          config.transition_lookaheads[transition][axis] < 0)
        throw std::invalid_argument(
            "AmrSystem transition buffers and lookaheads must be non-negative");
    }
  for (int axis = 0; axis < Dim; ++axis)
    if (config.coarse_max_grid[axis] < 0)
      throw std::invalid_argument("AmrSystem coarse_max_grid must be non-negative");
}

template <int Dim>
mesh::RankSpace<Dim> process_rank_space() {
  Extent<Dim> shape = runtime_config_detail::filled_extent<Dim>(1);
  shape[0] = n_ranks();
  return mesh::RankSpace<Dim>(Index<Dim>{}, shape);
}

template <int Dim>
std::int64_t checked_layout_cells(const mesh::BoxArray<Dim>& layout) {
  std::int64_t total = 0;
  for (const Box<Dim>& box : layout.boxes()) {
    const std::int64_t cells = box.numPts();
    if (cells < 1 || cells > std::numeric_limits<std::int64_t>::max() - total)
      throw std::overflow_error("AmrSystem layout cell budget exceeds int64_t");
    total += cells;
  }
  return total;
}

inline std::size_t checked_pair_count(std::size_t count) {
  if (count > 1 && count - 1 > std::numeric_limits<std::size_t>::max() / count)
    throw std::length_error("AmrSystem layout pair budget exceeds size_t");
  return count < 2 ? 0 : count * (count - 1) / 2;
}

inline std::size_t checked_square_count(std::size_t count) {
  if (count != 0 && count > std::numeric_limits<std::size_t>::max() / count)
    throw std::length_error("AmrSystem hierarchy pair budget exceeds size_t");
  return count * count;
}

template <int Dim>
std::size_t checked_cells(const Box<Dim>& box) {
  const std::int64_t cells = box.numPts();
  if (cells < 0 || static_cast<std::uint64_t>(cells) >
                       static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error("AmrSystem ranked field exceeds size_t");
  return static_cast<std::size_t>(cells);
}

template <int Dim>
Index<Dim> unflatten(const Box<Dim>& box, std::size_t linear) {
  Index<Dim> index{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t extent = static_cast<std::size_t>(box.length(axis));
    index[axis] = box.lo[axis] + static_cast<int>(linear % extent);
    linear /= extent;
  }
  return index;
}

template <int Dim>
std::size_t offset(const Index<Dim>& index, const Box<Dim>& box) {
  std::size_t result = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    result += static_cast<std::size_t>(index[axis] - box.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(box.length(axis));
  }
  return result;
}

template <int Dim>
std::vector<double> gather_field(const MultiFab<Dim>& field, const Box<Dim>& domain,
                                 int components) {
  if (components < 1 || components > field.ncomp())
    throw std::invalid_argument("AmrSystem gather component count is invalid");
  const std::size_t domain_cells = checked_cells(domain);
  if (static_cast<std::size_t>(components) > std::numeric_limits<std::size_t>::max() / domain_cells)
    throw std::overflow_error("AmrSystem gather buffer exceeds size_t");
  std::vector<double> result(static_cast<std::size_t>(components) * domain_cells, 0.0);
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const Fab<Dim>& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t local_cells = checked_cells(valid);
    const std::size_t component_stride = checked_cells(grown);
    for (int component = 0; component < components; ++component)
      for (std::size_t linear = 0; linear < local_cells; ++linear) {
        const Index<Dim> index = unflatten(valid, linear);
        result[static_cast<std::size_t>(component) * domain_cells + offset(index, domain)] =
            static_cast<double>(host(static_cast<std::size_t>(component) * component_stride +
                                     offset(index, grown)));
      }
  }
  all_reduce_sum_inplace(result.data(), result.size());
  return result;
}

template <int Dim>
void write_field(MultiFab<Dim>& field, const Box<Dim>& domain, const std::vector<double>& values,
                 int components) {
  const std::size_t domain_cells = checked_cells(domain);
  if (components < 1 || components > field.ncomp() ||
      values.size() != static_cast<std::size_t>(components) * domain_cells)
    throw std::invalid_argument("AmrSystem field input differs from its exact ranked shape");
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    Fab<Dim>& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t local_cells = checked_cells(valid);
    const std::size_t component_stride = checked_cells(grown);
    for (int component = 0; component < components; ++component)
      for (std::size_t linear = 0; linear < local_cells; ++linear) {
        const Index<Dim> index = unflatten(valid, linear);
        host(static_cast<std::size_t>(component) * component_stride + offset(index, grown)) =
            static_cast<Real>(
                values[static_cast<std::size_t>(component) * domain_cells + offset(index, domain)]);
      }
    fab.copy_from_host(host);
  }
}

template <int Dim>
void write_component(MultiFab<Dim>& field, const Box<Dim>& domain,
                     const std::vector<double>& values, int component) {
  const std::size_t domain_cells = checked_cells(domain);
  if (component < 0 || component >= field.ncomp() || values.size() != domain_cells)
    throw std::invalid_argument("AmrSystem component input differs from its exact ranked shape");
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    Fab<Dim>& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t local_cells = checked_cells(valid);
    const std::size_t component_stride = checked_cells(grown);
    for (std::size_t linear = 0; linear < local_cells; ++linear) {
      const Index<Dim> index = unflatten(valid, linear);
      host(static_cast<std::size_t>(component) * component_stride + offset(index, grown)) =
          static_cast<Real>(values[offset(index, domain)]);
    }
    fab.copy_from_host(host);
  }
}

template <int Dim>
double cell_measure(const AmrSystemConfig<Dim>& config, const Box<Dim>& domain) {
  double measure = 1.0;
  for (int axis = 0; axis < Dim; ++axis)
    measure *= static_cast<double>(config.upper[axis] - config.lower[axis]) /
               static_cast<double>(domain.length(axis));
  return measure;
}

inline std::size_t checked_size_product(std::size_t left, std::size_t right,
                                        const char* operation) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(operation);
  return left * right;
}

inline std::size_t checked_size_sum(std::size_t left, std::size_t right, const char* operation) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::length_error(operation);
  return left + right;
}

template <int Dim>
std::size_t periodic_image_bound(const Box<Dim>& domain, const Extent<Dim>& ghosts,
                                 const BoundaryTopology<Dim>& topology) {
  std::size_t images = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    std::size_t axis_images = 1;
    if (topology.is_periodic(Face<Dim>{axis, BoundarySide::lower}) && ghosts[axis] > 0) {
      const std::int64_t length = domain.length(axis);
      if (length <= 0)
        throw std::invalid_argument("AMR halo budget requires a non-empty periodic domain");
      const std::int64_t wraps = 1 + (ghosts[axis] - 1) / length;
      if (wraps > static_cast<std::int64_t>((std::numeric_limits<std::size_t>::max() - 1) / 2))
        throw std::length_error("AMR halo periodic-image budget exceeds size_t");
      axis_images = 1 + 2 * static_cast<std::size_t>(wraps);
    }
    images =
        checked_size_product(images, axis_images, "AMR halo periodic-image product exceeds size_t");
  }
  return images;
}

template <int Dim>
std::size_t grown_layout_elements(const mesh::BoxArray<Dim>& layout, const Extent<Dim>& ghosts,
                                  int ncomp) {
  if (ncomp < 1)
    throw std::invalid_argument("AMR halo element budget requires positive components");
  std::size_t cells = 0;
  for (const Box<Dim>& valid : layout.boxes()) {
    Box<Dim> grown = valid;
    for (int axis = 0; axis < Dim; ++axis) {
      if (ghosts[axis] < 0 || ghosts[axis] > std::numeric_limits<int>::max())
        throw std::invalid_argument("AMR halo ghost budget exceeds native coordinates");
      grown = grown.grow(axis, static_cast<int>(ghosts[axis]));
    }
    cells =
        checked_size_sum(cells, checked_cells(grown), "AMR halo grown-cell budget exceeds size_t");
  }
  return checked_size_product(cells, static_cast<std::size_t>(ncomp),
                              "AMR halo element budget exceeds size_t");
}

template <int Dim>
HaloScheduleBudget exact_halo_budget(const MultiFab<Dim>& field, const Box<Dim>& domain,
                                     const BoundaryTopology<Dim>& topology) {
  const std::size_t patches = field.layout().size();
  const std::size_t images = periodic_image_bound(domain, field.ghosts(), topology);
  const std::size_t patch_pairs =
      checked_size_product(patches, patches, "AMR halo patch-pair budget exceeds size_t");
  const std::size_t work =
      checked_size_product(patch_pairs, images, "AMR halo box-image budget exceeds size_t");
  const std::size_t jobs = checked_size_product(work, static_cast<std::size_t>(2 * Dim),
                                                "AMR halo job budget exceeds size_t");
  const std::size_t elements = checked_size_product(
      checked_size_product(grown_layout_elements(field.layout(), field.ghosts(), field.ncomp()),
                           std::max<std::size_t>(patches, 1),
                           "AMR halo patch-element budget exceeds size_t"),
      images, "AMR halo image-element budget exceeds size_t");
  return HaloScheduleBudget{
      mesh::BoxArrayValidationBudget{patches, checked_pair_count(patches)},
      work,
      jobs,
      images,
      field.rank_space().size(),
      elements,
      elements,
      elements,
  };
}

template <int Dim>
runtime::amr::AmrGhostFillBudget exact_amr_ghost_budget(const MultiFab<Dim>& coarse,
                                                        const MultiFab<Dim>& fine,
                                                        const Box<Dim>& coarse_domain,
                                                        const Box<Dim>& fine_domain,
                                                        const BoundaryTopology<Dim>& topology) {
  const std::size_t coarse_patches = coarse.layout().size();
  const std::size_t fine_patches = fine.layout().size();
  const std::size_t ranks = fine.rank_space().size();
  const std::size_t images = periodic_image_bound(fine_domain, fine.ghosts(), topology);
  const std::size_t regions =
      checked_size_product(checked_size_product(fine_patches, static_cast<std::size_t>(2 * Dim),
                                                "AMR coarse/fine region budget exceeds size_t"),
                           images, "AMR coarse/fine periodic-region budget exceeds size_t");
  const std::size_t cross_pairs = checked_size_product(
      coarse_patches, fine_patches, "AMR coarse/fine patch-pair budget exceeds size_t");
  const std::size_t pair_budget =
      std::max({cross_pairs, checked_pair_count(coarse_patches), checked_pair_count(fine_patches)});
  const std::size_t jobs = checked_size_product(
      checked_size_product(std::max<std::size_t>(regions, fine_patches),
                           std::max<std::size_t>(coarse_patches, 1),
                           "AMR coarse/fine job budget exceeds size_t"),
      std::max<std::size_t>(ranks, 1), "AMR coarse/fine rank-job budget exceeds size_t");
  const std::size_t elements = checked_size_product(
      checked_size_product(
          checked_size_product(checked_cells(coarse_domain), static_cast<std::size_t>(fine.ncomp()),
                               "AMR coarse/fine component budget exceeds size_t"),
          std::max<std::size_t>(fine_patches, 1),
          "AMR coarse/fine patch-element budget exceeds size_t"),
      checked_size_product(std::max<std::size_t>(images, 1), std::max<std::size_t>(ranks, 1),
                           "AMR coarse/fine image-rank budget exceeds size_t"),
      "AMR coarse/fine element budget exceeds size_t");

  runtime::amr::AmrGhostFillBudget budget;
  budget.coarse_fine = runtime::amr::CoarseFineGhostScheduleBudget{
      fine_patches, regions, pair_budget, jobs, ranks, elements, elements, elements};
  budget.same_level = exact_halo_budget(fine, fine_domain, topology);
  return budget;
}

template <int Dim>
std::vector<const Real*> field_storage_identity(const MultiFab<Dim>& field) {
  std::vector<const Real*> identity;
  identity.reserve(field.local_size());
  for (std::size_t local = 0; local < field.local_size(); ++local)
    identity.push_back(field.fab(local).view().data);
  return identity;
}

template <int Dim>
bool field_storage_matches(const MultiFab<Dim>& field,
                           const std::vector<const Real*>& identity) noexcept {
  try {
    if (field.local_size() != identity.size())
      return false;
    for (std::size_t local = 0; local < field.local_size(); ++local)
      if (field.fab(local).view().data != identity[local])
        return false;
    return true;
  } catch (...) {
    return false;
  }
}

template <int Dim>
bool same_field_shape(const MultiFab<Dim>& left, const MultiFab<Dim>& right) noexcept {
  return left.layout() == right.layout() && left.distribution() == right.distribution() &&
         left.local_rank() == right.local_rank() && left.local_size() == right.local_size() &&
         left.ncomp() == right.ncomp();
}

template <int Dim>
void copy_valid_field(const MultiFab<Dim>& source, MultiFab<Dim>& destination) {
  if (!same_field_shape(source, destination))
    throw std::invalid_argument("AMR auxiliary copy requires one exact ranked field shape");
  for (std::size_t local = 0; local < source.local_size(); ++local) {
    const Fab<Dim>& source_fab = source.fab(local);
    Fab<Dim>& destination_fab = destination.fab(local);
    auto source_host = source_fab.create_host_mirror();
    auto destination_host = destination_fab.create_host_mirror();
    source_fab.copy_to_host(source_host);
    destination_fab.copy_to_host(destination_host);
    const Box<Dim>& valid = source_fab.box();
    const Box<Dim>& source_storage = source_fab.grown_box();
    const Box<Dim>& destination_storage = destination_fab.grown_box();
    const std::size_t source_stride = checked_cells(source_storage);
    const std::size_t destination_stride = checked_cells(destination_storage);
    const std::size_t cells = checked_cells(valid);
    for (int component = 0; component < source.ncomp(); ++component)
      for (std::size_t linear = 0; linear < cells; ++linear) {
        const Index<Dim> index = unflatten(valid, linear);
        destination_host(static_cast<std::size_t>(component) * destination_stride +
                         offset(index, destination_storage)) =
            source_host(static_cast<std::size_t>(component) * source_stride +
                        offset(index, source_storage));
      }
    destination_fab.copy_from_host(destination_host);
  }
}

template <int Dim>
std::string exact_root_ghost_contract(const HaloSchedule<Dim>& schedule,
                                      std::string_view field_identity,
                                      std::uint64_t topology_generation,
                                      std::uint64_t materialization_generation,
                                      std::string_view lane_identity) {
  ExactContractBuilder contract;
  contract.text("pops.generated-amr-root-ghost-fill")
      .scalar(std::uint32_t{1})
      .scalar(std::int32_t{Dim})
      .text(field_identity)
      .text(lane_identity)
      .scalar(topology_generation)
      .scalar(materialization_generation)
      .scalar(schedule.coverage())
      .scalar(std::int32_t{schedule.ncomp()});
  for (int axis = 0; axis < Dim; ++axis)
    contract.scalar(std::int64_t{schedule.domain().lo[axis]})
        .scalar(std::int64_t{schedule.domain().hi[axis]})
        .scalar(std::int64_t{schedule.ghosts()[axis]})
        .scalar(schedule.topology().kind(Face<Dim>{axis, BoundarySide::lower}))
        .scalar(schedule.topology().kind(Face<Dim>{axis, BoundarySide::upper}));
  contract.sequence(schedule.layout().boxes(), [](ExactContractBuilder& item, const Box<Dim>& box) {
    for (int axis = 0; axis < Dim; ++axis)
      item.scalar(std::int64_t{box.lo[axis]}).scalar(std::int64_t{box.hi[axis]});
  });
  contract.sequence(schedule.distribution().owners(),
                    [](ExactContractBuilder& item, const Index<Dim>& owner) {
                      for (int axis = 0; axis < Dim; ++axis)
                        item.scalar(std::int64_t{owner[axis]});
                    });
  contract.sequence(schedule.canonical_jobs(), [](ExactContractBuilder& item, const auto& job) {
    item.scalar(static_cast<std::uint64_t>(job.source_box))
        .scalar(static_cast<std::uint64_t>(job.destination_box))
        .scalar(static_cast<std::uint64_t>(job.elements));
    for (int axis = 0; axis < Dim; ++axis)
      item.scalar(std::int64_t{job.destination_region.lo[axis]})
          .scalar(std::int64_t{job.destination_region.hi[axis]})
          .scalar(std::int64_t{job.source_from_destination[axis]});
  });
  return std::move(contract).release();
}

template <int Dim>
struct GeneratedRootGhostSource {
  using field_type = MultiFab<Dim>;

  struct State {
    std::optional<HaloSchedule<Dim>> schedule;
    const ExecutionLane* lane = nullptr;
    std::optional<ExecutionLane::ImmutableBorrow> lane_borrow;
    std::optional<HaloExchange<Dim>> exchange;
    std::vector<const Real*> storage;
    std::string contract;

    void execute(field_type& field, const runtime::multiblock::BoundaryEvaluationPoint& point) {
      const long binding_invalid = !schedule || lane == nullptr ? 1L : 0L;
      if (all_reduce_max(binding_invalid) != 0)
        throw std::logic_error("prepared AMR root ghost provider lost its immutable binding");
      const long invalid = point.level != 0 || !field_storage_matches(field, storage) ? 1L : 0L;
      if (all_reduce_max(invalid, lane->communicator()) != 0)
        throw std::invalid_argument(
            "prepared AMR root ghost provider received stale storage or a non-root level");
      if (exchange)
        fill_boundary(field, *exchange, *lane);
      else
        fill_boundary(field, *schedule);
    }
  };

  std::shared_ptr<State> state;

  [[nodiscard]] static PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.generated.amr.root-ghost-fill", 1};
  }

  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    if (!state)
      throw std::logic_error("generated AMR root ghost source is empty");
    contract.bytes(state->contract);
  }

  void operator()(field_type& field,
                  const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    if (!state)
      throw std::logic_error("generated AMR root ghost source is empty");
    state->execute(field, point);
  }
};

inline std::uint64_t exchange_generation(std::uint64_t generation, const char* label) {
  if (generation == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error(std::string(label) + " cannot form a halo exchange generation");
  return generation + 1;
}

template <int Dim>
PreparedRootAmrGhostFill<Dim> prepare_root_ghost_fill(MultiFab<Dim>& field, const Box<Dim>& domain,
                                                      const BoundaryTopology<Dim>& topology,
                                                      std::string field_identity,
                                                      std::uint64_t topology_generation,
                                                      std::uint64_t materialization_generation,
                                                      const ExecutionLane& lane) {
  using source_type = GeneratedRootGhostSource<Dim>;
  std::shared_ptr<typename source_type::State> state;
  std::exception_ptr metadata_error;
  long metadata_failure = 0;
  try {
    state = std::make_shared<typename source_type::State>();
    state->schedule.emplace(prepare_halo_schedule(field, domain, topology,
                                                  HaloLayoutCoverage::full_domain,
                                                  exact_halo_budget(field, domain, topology)));
    state->lane = &lane;
    state->storage = field_storage_identity(field);
    state->contract =
        exact_root_ghost_contract(*state->schedule, field_identity, topology_generation,
                                  materialization_generation, lane.identity());
  } catch (...) {
    metadata_failure = 1;
    metadata_error = std::current_exception();
  }
  if (all_reduce_max(metadata_failure, lane.communicator()) != 0) {
    if (metadata_error)
      std::rethrow_exception(metadata_error);
    throw std::runtime_error("generated AMR root ghost metadata failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("generated-amr-root-ghost"), std::string_view(state->contract)}},
          lane.communicator()))
    throw std::invalid_argument("generated AMR root ghost contracts differ between ranks");

  const long remote_any =
      all_reduce_max(state->schedule->has_remote_jobs() ? 1L : 0L, lane.communicator());
  std::exception_ptr transport_error;
  long transport_failure = 0;
  try {
    if (remote_any != 0) {
      state->exchange.emplace(
          *state->schedule, lane,
          HaloExchangeContext{
              .context_generation = exchange_generation(topology_generation, "topology"),
              .schedule_generation =
                  exchange_generation(materialization_generation, "materialization"),
          });
    } else {
      state->lane_borrow.emplace(lane.borrow_immutably());
    }
  } catch (...) {
    transport_failure = 1;
    transport_error = std::current_exception();
  }
  if (all_reduce_max(transport_failure, lane.communicator()) != 0) {
    if (transport_error)
      std::rethrow_exception(transport_error);
    throw std::runtime_error("generated AMR root ghost transport failed collectively");
  }
  std::optional<PreparedRootAmrGhostFill<Dim>> prepared;
  std::exception_ptr provider_error;
  long provider_failure = 0;
  try {
    prepared.emplace(source_type{std::move(state)});
  } catch (...) {
    provider_failure = 1;
    provider_error = std::current_exception();
  }
  if (all_reduce_max(provider_failure, lane.communicator()) != 0) {
    if (provider_error)
      std::rethrow_exception(provider_error);
    throw std::runtime_error("generated AMR root ghost provider failed collectively");
  }
  return std::move(*prepared);
}

void validate_variable_set(const VariableSet& variables, VariableKind expected, int ncomp,
                           const char* label) {
  if (variables.kind != expected || variables.size != ncomp ||
      variables.names.size() != static_cast<std::size_t>(ncomp) ||
      variables.roles.size() != static_cast<std::size_t>(ncomp) ||
      (!variables.user_roles.empty() &&
       variables.user_roles.size() != static_cast<std::size_t>(ncomp)))
    throw std::invalid_argument(std::string("prepared AMR block ") + label +
                                " variable metadata differs from its component count");
  std::set<std::string> names;
  for (const std::string& name : variables.names)
    if (name.empty() || !names.insert(name).second)
      throw std::invalid_argument(std::string("prepared AMR block ") + label +
                                  " variable names must be unique and non-empty");
}

template <int Dim>
void validate_prepared_amr_block(const PreparedAmrSystemBlock<Dim>& block) {
  if (block.name.empty() || block.provider_identity.empty() || block.collective_contract.empty())
    throw std::invalid_argument(
        "prepared AMR block requires non-empty block, provider, and collective identities");
  if (block.ncomp < 1 || block.aux_components < 1)
    throw std::invalid_argument(
        "prepared AMR block requires positive state and auxiliary component counts");
  if (!std::isfinite(block.gamma) || !(block.gamma > 0.0) || block.substeps < 1 ||
      block.stride < 1 || block.time_route.empty())
    throw std::invalid_argument("prepared AMR block gamma, cadence, or time route is invalid");
  for (int axis = 0; axis < Dim; ++axis)
    if (block.ghosts[axis] < 1)
      throw std::invalid_argument(
          "prepared AMR block must declare a positive ghost extent on every native axis");
  validate_variable_set(block.conservative_variables, VariableKind::Conservative, block.ncomp,
                        "conservative");
  validate_variable_set(block.primitive_variables, VariableKind::Primitive, block.ncomp,
                        "primitive");
  if (!block.materialize_level || !block.primitive_to_conservative ||
      !block.conservative_to_primitive || !block.batch_conservative_to_primitive)
    throw std::invalid_argument(
        "prepared AMR block does not implement its complete exact-ranked execution contract");
}

template <int Dim>
std::string exact_hyperbolic_boundary_contract(const PreparedHyperbolicBoundary<Dim>& boundary) {
  ExactContractBuilder contract;
  contract.text("pops.prepared-hyperbolic-boundary")
      .scalar(std::uint32_t{1})
      .scalar(std::int32_t{Dim})
      .scalar(std::int32_t{boundary.ncomp()})
      .scalar(boundary.corner_policy());
  for (int component = 0; component < boundary.ncomp(); ++component) {
    const auto& transform = boundary.component_transform(component);
    contract.scalar(transform.parity).scalar(std::int32_t{transform.axis});
  }
  for (int axis = 0; axis < Dim; ++axis) {
    for (const int side : {-1, 1}) {
      const PreparedHyperbolicFace& face = boundary.face(axis, side);
      contract.scalar(std::int32_t{axis})
          .scalar(std::int32_t{side})
          .scalar(face.law)
          .text(face.identity)
          .scalar(face.identity_token)
          .scalar(face.authored_representation)
          .text(face.converter_identity)
          .scalar(face.fixed_state_converted)
          .text(face.analytic_clock)
          .scalar(static_cast<std::uint64_t>(face.fixed_state.size()));
      for (const Real value : face.fixed_state)
        contract.scalar(value);
      contract.scalar(static_cast<std::uint64_t>(face.analytic_state.size()));
      for (const analytic::AnalyticProgram& program : face.analytic_state) {
        const analytic::AnalyticProgramView view = program.view();
        contract.scalar(static_cast<std::uint64_t>(program.instruction_count()))
            .scalar(static_cast<std::uint64_t>(program.literal_count()))
            .scalar(static_cast<std::uint64_t>(program.required_stack()))
            .scalar(std::uint8_t{static_cast<std::uint8_t>(program.required_dimension())})
            .scalar(program.result_type());
        for (std::size_t index = 0; index < program.instruction_count(); ++index)
          contract.scalar(view.instructions[index].op).scalar(view.instructions[index].operand);
        for (std::size_t index = 0; index < program.literal_count(); ++index)
          contract.scalar(view.literals[index]);
      }
    }
  }
  return std::move(contract).release();
}

}  // namespace

template <int Dim>
struct AmrSystem<Dim>::Impl {
  using engine_type = runtime::amr::AmrRuntime<Dim>;
  using field_type = MultiFab<Dim>;
  using boundary_registry_type = runtime::system::SystemBoundaryRegistry<Dim>;
  using prepared_block_type = PreparedAmrSystemBlock<Dim>;
  using level_block_type = PreparedGeneratedAmrLevelBlock<Dim>;
  using evaluation_type = PreparedAmrLevelEvaluation<Dim>;

  struct BlockSpec {
    std::string name;
    int ncomp = 0;
    double gamma = static_cast<double>(kPhysicalDefaultGamma);
    int substeps = 1;
    int stride = 1;
    int required_ghost_depth = 1;
    Extent<Dim> ghosts{};
    std::string time = "euler";
    bool has_density = false;
    std::vector<double> density;
    bool has_state = false;
    std::vector<double> state;
  };

  struct GlobalDtBound {
    std::string label;
    std::function<double()> evaluate;
  };

  struct PreparedHierarchy {
    // The lane is declared first so every provider that pins it is destroyed before MPI_Comm_free.
    std::optional<ExecutionLane> lane;
    std::vector<std::unique_ptr<field_type>> auxiliary;
    std::vector<level_block_type> levels;
    std::vector<std::optional<evaluation_type>> evaluations;
    std::vector<std::vector<const Real*>> state_storage;
    std::vector<std::vector<const Real*>> auxiliary_storage;
    std::vector<std::string> state_field_identities;
    std::vector<std::string> auxiliary_field_identities;
    std::string spatial_contract;
    std::string package_contract;
    std::string collective_contract;
    std::uint64_t topology_epoch = 0;
    std::uint64_t materialization_generation = 0;

    bool matches(const engine_type& live, std::string_view expected_package) const noexcept {
      try {
        if (!lane || spatial_contract != live.spatial_contract() ||
            package_contract != expected_package || topology_epoch != live.topology_epoch() ||
            materialization_generation != live.materialization_generation() ||
            levels.size() != live.hierarchy().num_levels() ||
            state_storage.size() != live.hierarchy().num_levels() ||
            auxiliary_storage.size() != auxiliary.size())
          return false;
        for (std::size_t level = 0; level < state_storage.size(); ++level)
          if (!field_storage_matches(live.hierarchy().state(level), state_storage[level]) ||
              !auxiliary[level] ||
              !field_storage_matches(*auxiliary[level], auxiliary_storage[level]))
            return false;
        return true;
      } catch (...) {
        return false;
      }
    }
  };

  using auxiliary_snapshot_type = std::vector<field_type>;

  AmrSystemConfig<Dim> cfg;
  std::shared_ptr<const PreparedLoadBalanceAuthority<Dim>> load_balance;
  std::vector<BlockSpec> blocks;
  boundary_registry_type boundary_registry;
  runtime::program::ProgramRuntimeState<Dim> program;
  runtime::system::SystemLifecycle lifecycle;
  mutable std::unique_ptr<engine_type> engine;
  std::optional<prepared_block_type> prepared_block;
  mutable std::unique_ptr<PreparedHierarchy> prepared_hierarchy;
  mutable std::shared_ptr<const auxiliary_snapshot_type> pending_auxiliary_restore;
  std::vector<GlobalDtBound> dt_bounds;
  std::vector<CouplingOperatorView> coupling_views;
  double accepted_time = 0.0;
  int macro_step = 0;
  std::string last_dt_reason;
  std::vector<std::uint8_t> program_accepted_bytes;
  std::uint64_t program_accepted_revision = 0;

  struct AcceptedSnapshot {
    std::optional<typename engine_type::Snapshot> engine;
    std::shared_ptr<const auxiliary_snapshot_type> auxiliary;
    runtime::program::ProgramRuntimeState<Dim> program;
    double accepted_time = 0.0;
    int macro_step = 0;
    std::vector<std::uint8_t> program_accepted_bytes;
    std::uint64_t program_accepted_revision = 0;

    explicit AcceptedSnapshot(const Impl& owner)
        : engine(owner.engine
                     ? std::optional<typename engine_type::Snapshot>(owner.engine->snapshot())
                     : std::nullopt),
          auxiliary(owner.snapshot_auxiliary()),
          program(owner.program),
          accepted_time(owner.accepted_time),
          macro_step(owner.macro_step),
          program_accepted_bytes(owner.program_accepted_bytes),
          program_accepted_revision(owner.program_accepted_revision) {}

    void restore(Impl& owner) {
      if (engine.has_value() != static_cast<bool>(owner.engine))
        throw std::logic_error("AmrSystem transaction changed engine materialization");
      owner.prepared_hierarchy.reset();
      if (engine) {
        owner.engine->restore(*engine);
        owner.pending_auxiliary_restore = auxiliary;
      }
      owner.program = program;
      owner.accepted_time = accepted_time;
      owner.macro_step = macro_step;
      owner.program_accepted_bytes = program_accepted_bytes;
      owner.program_accepted_revision = program_accepted_revision;
    }
  };

  std::unique_ptr<AcceptedSnapshot> external_step_transaction;
  bool external_step_committed = false;

  explicit Impl(const AmrSystemConfig<Dim>& config)
      : cfg(config),
        load_balance(std::make_shared<const PreparedLoadBalanceAuthority<Dim>>(
            prepare_load_balance_authority<Dim>(cfg.load_balance_route, cfg.load_balance_identity,
                                                cfg.load_balance_options))) {}

  BlockSpec& block(const std::string& name) {
    for (BlockSpec& candidate : blocks)
      if (candidate.name == name || (blocks.size() == 1 && name.empty()))
        return candidate;
    throw std::runtime_error("AmrSystem has no block named '" + name + "'");
  }

  const BlockSpec& block(const std::string& name) const {
    return const_cast<Impl*>(this)->block(name);
  }

  std::shared_ptr<const auxiliary_snapshot_type> snapshot_auxiliary() const {
    if (!prepared_hierarchy)
      return {};
    auto snapshot = std::make_shared<auxiliary_snapshot_type>();
    snapshot->reserve(prepared_hierarchy->auxiliary.size());
    for (const std::unique_ptr<field_type>& level : prepared_hierarchy->auxiliary) {
      if (!level)
        throw std::logic_error("prepared AMR hierarchy contains an empty auxiliary owner");
      snapshot->push_back(*level);
    }
    return snapshot;
  }

  BoundaryTopology<Dim> topology() const {
    return BoundaryTopology<Dim>::axis_periodic(cfg.periodicity);
  }

  std::unique_ptr<PreparedHierarchy> prepare_hierarchy_graph(
      engine_type& candidate_engine, const PreparedHierarchy* previous) const {
    if (!prepared_block)
      throw std::logic_error("AmrSystem has no retained generated package");

    const std::size_t level_count = candidate_engine.hierarchy().num_levels();
    std::unique_ptr<PreparedHierarchy> candidate;
    std::string lane_identity;
    std::shared_ptr<const PreparedHyperbolicBoundary<Dim>> boundary;
    std::string boundary_identity;
    std::exception_ptr allocation_error;
    long allocation_failure = 0;
    try {
      candidate = std::make_unique<PreparedHierarchy>();
      candidate->spatial_contract.assign(candidate_engine.spatial_contract());
      candidate->package_contract = prepared_block->collective_contract;
      candidate->topology_epoch = candidate_engine.topology_epoch();
      candidate->materialization_generation = candidate_engine.materialization_generation();
      candidate->auxiliary.reserve(level_count);
      candidate->levels.reserve(level_count);
      candidate->evaluations.resize(level_count);
      candidate->state_storage.reserve(level_count);
      candidate->auxiliary_storage.reserve(level_count);
      candidate->state_field_identities.reserve(level_count);
      candidate->auxiliary_field_identities.reserve(level_count);

      const std::string& state_route = boundary_registry.state_route(prepared_block->name);
      const auto* installed_boundary = boundary_registry.find_boundary(prepared_block->name);
      if (installed_boundary != nullptr) {
        boundary = installed_boundary->authority;
        boundary_identity = installed_boundary->identity;
      }
      lane_identity = "pops.generated-amr-levels/" + std::to_string(candidate->topology_epoch) +
                      "/" + std::to_string(candidate->materialization_generation);
      if (pending_auxiliary_restore && pending_auxiliary_restore->size() != level_count)
        throw std::invalid_argument(
            "AMR rollback auxiliary image differs from the restored hierarchy depth");
      for (std::size_t level = 0; level < level_count; ++level) {
        field_type& state = candidate_engine.hierarchy().state(level);
        auto auxiliary =
            std::make_unique<field_type>(state.layout(), state.distribution(), state.local_rank(),
                                         prepared_block->aux_components, prepared_block->ghosts);
        auxiliary->set_val(Real(0));
        if (pending_auxiliary_restore) {
          const field_type& restored = (*pending_auxiliary_restore)[level];
          if (!same_field_shape(restored, *auxiliary))
            throw std::invalid_argument(
                "AMR rollback auxiliary image differs from the restored level layout");
          copy_valid_field(restored, *auxiliary);
        } else if (previous != nullptr && level < previous->auxiliary.size() &&
                   previous->auxiliary[level] &&
                   same_field_shape(*previous->auxiliary[level], *auxiliary)) {
          copy_valid_field(*previous->auxiliary[level], *auxiliary);
        }
        candidate->state_storage.push_back(field_storage_identity(state));
        candidate->auxiliary_storage.push_back(field_storage_identity(*auxiliary));
        candidate->state_field_identities.push_back(state_route + "/level/" +
                                                    std::to_string(level));
        candidate->auxiliary_field_identities.push_back(
            prepared_block->provider_identity + "/auxiliary/level/" + std::to_string(level));
        candidate->auxiliary.push_back(std::move(auxiliary));
      }
    } catch (...) {
      allocation_failure = 1;
      allocation_error = std::current_exception();
    }
    if (all_reduce_max(allocation_failure) != 0) {
      if (allocation_error)
        std::rethrow_exception(allocation_error);
      throw std::runtime_error(
          "generated AMR hierarchy allocation failed collectively before lane publication");
    }

    candidate->lane.emplace(ExecutionLane::duplicate_world_collectively(lane_identity));
    const BoundaryTopology<Dim> exact_topology = topology();

    for (std::size_t level = 0; level < level_count; ++level) {
      std::optional<level_block_type> prepared_level;
      std::exception_ptr level_error;
      long level_failure = 0;
      try {
        field_type& state = candidate_engine.hierarchy().state(level);
        field_type& auxiliary = *candidate->auxiliary[level];
        const Box<Dim>& level_domain = candidate_engine.hierarchy().layout(level).domain();
        runtime::amr::PreparedAmrGhostFill<Dim> state_ghost_fill;
        runtime::amr::PreparedAmrGhostFill<Dim> auxiliary_ghost_fill;
        PreparedRootAmrGhostFill<Dim> root_state_ghost_fill;
        PreparedRootAmrGhostFill<Dim> root_auxiliary_ghost_fill;
        if (level == 0) {
          root_state_ghost_fill = prepare_root_ghost_fill(
              state, level_domain, exact_topology, candidate->state_field_identities[level],
              candidate->topology_epoch, candidate->materialization_generation, *candidate->lane);
          root_auxiliary_ghost_fill = prepare_root_ghost_fill(
              auxiliary, level_domain, exact_topology, candidate->auxiliary_field_identities[level],
              candidate->topology_epoch, candidate->materialization_generation, *candidate->lane);
        } else {
          const Box<Dim>& coarse_domain = candidate_engine.hierarchy().layout(level - 1).domain();
          const auto& ratio = candidate_engine.hierarchy().layout(level).ratio_from_parent();
          state_ghost_fill = runtime::amr::prepare_amr_ghost_fill(
              candidate_engine.hierarchy().state(level - 1), state,
              runtime::amr::AmrGhostFillPreparation<Dim>{
                  .fine_level = static_cast<int>(level),
                  .coarse_domain = coarse_domain,
                  .fine_domain = level_domain,
                  .ratio = ratio,
                  .topology = exact_topology,
                  .topology_generation = candidate->topology_epoch,
                  .materialization_generation = candidate->materialization_generation,
                  .field_identity = candidate->state_field_identities[level],
                  .budget =
                      exact_amr_ghost_budget(candidate_engine.hierarchy().state(level - 1), state,
                                             coarse_domain, level_domain, exact_topology),
              },
              *candidate->lane);
          auxiliary_ghost_fill = runtime::amr::prepare_amr_ghost_fill(
              *candidate->auxiliary[level - 1], auxiliary,
              runtime::amr::AmrGhostFillPreparation<Dim>{
                  .fine_level = static_cast<int>(level),
                  .coarse_domain = coarse_domain,
                  .fine_domain = level_domain,
                  .ratio = ratio,
                  .topology = exact_topology,
                  .topology_generation = candidate->topology_epoch,
                  .materialization_generation = candidate->materialization_generation,
                  .field_identity = candidate->auxiliary_field_identities[level],
                  .budget = exact_amr_ghost_budget(*candidate->auxiliary[level - 1], auxiliary,
                                                   coarse_domain, level_domain, exact_topology),
              },
              *candidate->lane);
        }
        GeneratedAmrLevelContext<Dim> context{
            .level = level,
            .geometry = Geometry<Dim>::from_bounds(level_domain, cfg.lower, cfg.upper),
            .topology = exact_topology,
            .auxiliary = &auxiliary,
            .state_ghost_fill = std::move(state_ghost_fill),
            .auxiliary_ghost_fill = std::move(auxiliary_ghost_fill),
            .root_state_ghost_fill = std::move(root_state_ghost_fill),
            .root_auxiliary_ghost_fill = std::move(root_auxiliary_ghost_fill),
            .physical_boundary = boundary,
            .state_identity = candidate->state_field_identities[level],
            .auxiliary_identity = candidate->auxiliary_field_identities[level],
            .boundary_identity = boundary_identity,
        };
        prepared_level.emplace(prepared_block->prepare_level(candidate_engine, std::move(context)));
      } catch (...) {
        level_failure = 1;
        level_error = std::current_exception();
      }
      if (all_reduce_max(level_failure, candidate->lane->communicator()) != 0) {
        if (level_error)
          std::rethrow_exception(level_error);
        throw std::runtime_error("generated AMR level preparation failed collectively");
      }
      candidate->levels.push_back(std::move(*prepared_level));
    }

    std::exception_ptr contract_error;
    long contract_failure = 0;
    try {
      ExactContractBuilder contract;
      contract.text("pops.generated-amr-hierarchy-graph")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .bytes(candidate->spatial_contract)
          .bytes(candidate->package_contract)
          .text(candidate->lane->identity())
          .scalar(candidate->topology_epoch)
          .scalar(candidate->materialization_generation)
          .scalar(static_cast<std::uint64_t>(candidate->levels.size()));
      for (const level_block_type& level : candidate->levels)
        contract.bytes(level.collective_contract());
      candidate->collective_contract = std::move(contract).release();
    } catch (...) {
      contract_failure = 1;
      contract_error = std::current_exception();
    }
    if (all_reduce_max(contract_failure, candidate->lane->communicator()) != 0) {
      if (contract_error)
        std::rethrow_exception(contract_error);
      throw std::runtime_error("generated AMR hierarchy contract failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("generated-amr-hierarchy"),
              std::string_view(candidate->collective_contract)}},
            candidate->lane->communicator()))
      throw std::invalid_argument("generated AMR hierarchy contracts differ between MPI ranks");
    return candidate;
  }

  void refresh_prepared_hierarchy() const {
    if (!engine || !prepared_block)
      throw std::logic_error("AmrSystem cannot prepare levels before package materialization");
    const long has_graph = prepared_hierarchy ? 1L : 0L;
    if (all_reduce_min(has_graph) != all_reduce_max(has_graph))
      throw std::runtime_error("prepared AMR graph publication differs between MPI ranks");
    const long stale = prepared_hierarchy && prepared_hierarchy->matches(
                                                 *engine, prepared_block->collective_contract)
                           ? 0L
                           : 1L;
    if (all_reduce_max(stale) == 0)
      return;
    std::unique_ptr<PreparedHierarchy> candidate =
        prepare_hierarchy_graph(*engine, prepared_hierarchy.get());
    prepared_hierarchy.swap(candidate);
    pending_auxiliary_restore.reset();
  }

  void discard_level_evaluations() const noexcept {
    if (!prepared_hierarchy)
      return;
    for (std::optional<evaluation_type>& evaluation : prepared_hierarchy->evaluations)
      evaluation.reset();
  }

  void ensure_engine() const {
    const long materialized = engine ? 1L : 0L;
    if (all_reduce_min(materialized) != all_reduce_max(materialized))
      throw std::runtime_error("AmrSystem hierarchy materialization differs between MPI ranks");
    if (materialized != 0) {
      refresh_prepared_hierarchy();
      return;
    }
    if (blocks.empty())
      throw std::logic_error(
          "AmrSystem requires a dimension-qualified block before materialization");
    if (blocks.size() != 1)
      throw std::logic_error(
          "AmrSystem exact-ranked core requires a prepared multi-block hierarchy provider");

    std::unique_ptr<engine_type> engine_candidate;
    std::exception_ptr engine_error;
    long engine_failure = 0;
    try {
      const BlockSpec& block = blocks.front();
      const Box<Dim> domain = cfg.index_domain();
      const mesh::BoxArray<Dim> patches(cfg.materialized_boxes());
      const mesh::RankSpace<Dim> ranks = process_rank_space<Dim>();
      const Index<Dim> local_rank = ranks.coordinate(static_cast<std::size_t>(my_rank()));
      mesh::Distribution<Dim> distribution;
      if (cfg.distribute_coarse) {
        const parallel::LoadBalancePreparationBudget budget{patches.size(), ranks.size(),
                                                            checked_layout_cells(patches)};
        distribution = load_balance->prepare(patches, ranks, budget).plan().distribution();
      } else {
        distribution = mesh::Distribution<Dim>::replicated(patches, ranks);
      }

      const mesh::BoxArrayValidationBudget layout_budget{patches.size(),
                                                         checked_pair_count(patches.size())};
      const amr::hierarchy::LevelLayout<Dim> coarse(0, domain, patches, distribution,
                                                    amr::RefinementRatio<Dim>{}, layout_budget);
      field_type state(patches, distribution, local_rank, block.ncomp, block.ghosts);
      if (block.has_state)
        write_field(state, domain, block.state, block.ncomp);
      else if (block.has_density)
        write_component(state, domain, block.density, 0);

      const std::size_t hierarchy_pairs =
          cfg.level_count == 1 ? 0 : std::max(checked_square_count(patches.size()), patches.size());
      auto hierarchy = amr::hierarchy::AmrHierarchy<Dim>::from_coarse(
          coarse, std::move(state),
          amr::hierarchy::HierarchyValidationBudget{static_cast<std::size_t>(cfg.level_count),
                                                    hierarchy_pairs});
      engine_candidate = std::make_unique<engine_type>(std::move(hierarchy), load_balance,
                                                       "pops.amr-system.exact-ranked");
    } catch (...) {
      engine_failure = 1;
      engine_error = std::current_exception();
    }
    if (all_reduce_max(engine_failure) != 0) {
      if (engine_error)
        std::rethrow_exception(engine_error);
      throw std::runtime_error("AmrSystem hierarchy preparation failed collectively");
    }
    std::unique_ptr<PreparedHierarchy> hierarchy_candidate =
        prepare_hierarchy_graph(*engine_candidate, nullptr);
    engine = std::move(engine_candidate);
    prepared_hierarchy = std::move(hierarchy_candidate);
    pending_auxiliary_restore.reset();
  }

  template <class Function>
  decltype(auto) execute_transaction(Function&& function) {
    ensure_engine();
    AcceptedSnapshot snapshot(*this);
    try {
      return std::forward<Function>(function)();
    } catch (...) {
      snapshot.restore(*this);
      throw;
    }
  }
};

template <int Dim>
AmrSystem<Dim>::AmrSystem(const AmrSystemConfig<Dim>& config) {
  validate_amr_config(config);
  p_ = std::make_unique<Impl>(config);
}

template <int Dim>
AmrSystem<Dim>::~AmrSystem() = default;

template <int Dim>
AmrSystem<Dim>::AmrSystem(AmrSystem&&) noexcept = default;

template <int Dim>
AmrSystem<Dim>& AmrSystem<Dim>::operator=(AmrSystem&&) noexcept = default;

template <int Dim>
void AmrSystem<Dim>::add_block(const std::string&, const ModelSpec&, const std::string&,
                               const std::string&, const std::string&, const std::string&, int, int,
                               const std::vector<std::string>&, const std::vector<std::string>&,
                               const NewtonOptions&, bool, double, double, bool) {
  throw std::runtime_error(
      "AmrSystem ModelSpec installation requires a dimension-qualified compiled block provider");
}

template <int Dim>
void AmrSystem<Dim>::set_compiled_block(int ncomp, double gamma, int substeps,
                                        AmrCompiledBlockBuilder<Dim> runtime_builder,
                                        const std::string& name, bool, const std::string& time,
                                        int stride, const std::vector<std::string>& implicit_vars,
                                        const std::vector<std::string>& implicit_roles, double,
                                        double, bool) {
  (void)ncomp;
  (void)gamma;
  (void)substeps;
  (void)runtime_builder;
  (void)name;
  (void)time;
  (void)stride;
  (void)implicit_vars;
  (void)implicit_roles;
  throw std::logic_error(
      "AmrSystem::set_compiled_block is retired: install one complete "
      "PreparedAmrSystemBlock<Dim> through the exact generated-package seam");
}

template <int Dim>
void AmrSystem<Dim>::install_prepared_amr_block(PreparedBlock prepared) {
  std::exception_ptr preparation_error;
  long preparation_failure = 0;
  std::vector<typename Impl::BlockSpec> block_candidate;
  std::optional<PreparedBlock> prepared_candidate;
  std::shared_ptr<const HyperbolicBoundary> converted_boundary;
  std::string install_contract;
  bool has_boundary = false;
  try {
    require_amr_assembling(p_->lifecycle, "install_prepared_amr_block");
    validate_prepared_amr_block(prepared);
    if (p_->engine || p_->prepared_hierarchy || p_->prepared_block || !p_->blocks.empty())
      throw std::logic_error("AmrSystem accepts exactly one prepared generated block");

    const auto route = p_->boundary_registry.state_routes().find(prepared.name);
    if (route == p_->boundary_registry.state_routes().end())
      throw std::runtime_error("prepared AMR block lacks its exact pre-installed state identity");
    const auto* installed_boundary = p_->boundary_registry.find_boundary(prepared.name);
    const BoundaryTopology<Dim> exact_topology =
        BoundaryTopology<Dim>::axis_periodic(p_->cfg.periodicity);
    if (generated_amr_detail::has_physical_faces(exact_topology) && installed_boundary == nullptr)
      throw std::runtime_error(
          "prepared AMR block with physical faces requires a model-qualified boundary");
    if (installed_boundary != nullptr) {
      if (installed_boundary->state_identity != route->second ||
          installed_boundary->authority->ncomp() != prepared.ncomp ||
          installed_boundary->authority->periodic_axes() != p_->cfg.periodicity)
        throw std::invalid_argument(
            "prepared AMR block boundary differs from its exact state/domain contract");
      for (int axis = 0; axis < Dim; ++axis)
        if (prepared.ghosts[axis] < installed_boundary->required_depth)
          throw std::invalid_argument(
              "prepared AMR block ghosts are narrower than its boundary requirement");
      converted_boundary = std::make_shared<HyperbolicBoundary>(
          installed_boundary->authority->with_converted_fixed_states(
              prepared.primitive_to_conservative));
      has_boundary = true;
    }

    typename Impl::BlockSpec block;
    block.name = prepared.name;
    block.ncomp = prepared.ncomp;
    block.gamma = prepared.gamma;
    block.substeps = prepared.substeps;
    block.stride = prepared.stride;
    block.ghosts = prepared.ghosts;
    block.time = prepared.time_route;
    block.required_ghost_depth = 0;
    for (int axis = 0; axis < Dim; ++axis)
      block.required_ghost_depth =
          std::max(block.required_ghost_depth, static_cast<int>(prepared.ghosts[axis]));
    block_candidate.reserve(1);
    block_candidate.push_back(std::move(block));

    ExactContractBuilder contract;
    contract.text("pops.amr-system.prepared-install")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .bytes(prepared.collective_contract)
        .text(route->second)
        .scalar(has_boundary);
    if (installed_boundary != nullptr)
      contract.text(installed_boundary->identity)
          .scalar(std::int32_t{installed_boundary->required_depth})
          .scalar(std::int32_t{installed_boundary->authority->ncomp()})
          .bytes(exact_hyperbolic_boundary_contract(*converted_boundary));
    install_contract = std::move(contract).release();
    prepared_candidate.emplace(std::move(prepared));
  } catch (...) {
    preparation_failure = 1;
    preparation_error = std::current_exception();
  }
  if (all_reduce_max(preparation_failure) != 0) {
    if (preparation_error)
      std::rethrow_exception(preparation_error);
    throw std::runtime_error("prepared AMR block validation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("prepared-amr-install"), std::string_view(install_contract)}}))
    throw std::invalid_argument("prepared AMR block install contracts differ between MPI ranks");

  p_->blocks.swap(block_candidate);
  p_->prepared_block.swap(prepared_candidate);
  if (has_boundary)
    p_->boundary_registry.boundary(p_->blocks.front().name).authority =
        std::move(converted_boundary);
}

template <int Dim>
void AmrSystem<Dim>::refresh_prepared_amr_levels() {
  p_->ensure_engine();
}

template <int Dim>
const typename AmrSystem<Dim>::PreparedLevelEvaluation& AmrSystem<Dim>::evaluate_prepared_amr_level(
    const runtime::multiblock::BoundaryEvaluationPoint& point) {
  p_->ensure_engine();
  std::string point_contract;
  std::exception_ptr point_error;
  long point_failure = 0;
  try {
    ExactContractBuilder contract;
    contract.text("pops.generated-amr-evaluation-point")
        .scalar(std::uint32_t{1})
        .text(point.clock)
        .scalar(point.tick)
        .scalar(std::int32_t{point.level})
        .scalar(std::int32_t{point.substep})
        .scalar(std::int32_t{point.stage})
        .scalar(point.stage_fraction.numerator)
        .scalar(point.stage_fraction.denominator)
        .scalar(point.dt)
        .scalar(point.physical_time);
    point_contract = std::move(contract).release();
  } catch (...) {
    point_failure = 1;
    point_error = std::current_exception();
  }
  if (all_reduce_max(point_failure, p_->prepared_hierarchy->lane->communicator()) != 0) {
    if (point_error)
      std::rethrow_exception(point_error);
    throw std::runtime_error("prepared AMR evaluation point failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("generated-amr-evaluation-point"), std::string_view(point_contract)}},
          p_->prepared_hierarchy->lane->communicator()))
    throw std::invalid_argument("prepared AMR evaluation points differ between MPI ranks");
  if (point.level < 0 ||
      static_cast<std::size_t>(point.level) >= p_->prepared_hierarchy->levels.size())
    throw std::out_of_range("prepared AMR evaluation level lies outside the live hierarchy");
  std::optional<PreparedLevelEvaluation> candidate;
  std::exception_ptr evaluation_error;
  long evaluation_failure = 0;
  try {
    candidate.emplace(
        p_->prepared_hierarchy->levels[static_cast<std::size_t>(point.level)].evaluate(point));
  } catch (...) {
    evaluation_failure = 1;
    evaluation_error = std::current_exception();
  }
  if (all_reduce_max(evaluation_failure, p_->prepared_hierarchy->lane->communicator()) != 0) {
    if (evaluation_error)
      std::rethrow_exception(evaluation_error);
    throw std::runtime_error("prepared AMR level evaluation failed collectively");
  }
  std::optional<PreparedLevelEvaluation>& published =
      p_->prepared_hierarchy->evaluations[static_cast<std::size_t>(point.level)];
  published.swap(candidate);
  return *published;
}

template <int Dim>
const typename AmrSystem<Dim>::PreparedLevelEvaluation&
AmrSystem<Dim>::prepared_amr_level_evaluation(int level) const {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->prepared_hierarchy->evaluations.size())
    throw std::out_of_range("prepared AMR ledger level lies outside the live hierarchy");
  const std::optional<PreparedLevelEvaluation>& evaluation =
      p_->prepared_hierarchy->evaluations[static_cast<std::size_t>(level)];
  if (!evaluation)
    throw std::logic_error("prepared AMR level has no published residual/flux evaluation");
  return *evaluation;
}

template <int Dim>
MultiFab<Dim>& AmrSystem<Dim>::prepared_amr_level_auxiliary(int level) {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->prepared_hierarchy->auxiliary.size())
    throw std::out_of_range("prepared AMR auxiliary level lies outside the live hierarchy");
  p_->discard_level_evaluations();
  return *p_->prepared_hierarchy->auxiliary[static_cast<std::size_t>(level)];
}

template <int Dim>
const MultiFab<Dim>& AmrSystem<Dim>::prepared_amr_level_auxiliary(int level) const {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->prepared_hierarchy->auxiliary.size())
    throw std::out_of_range("prepared AMR auxiliary level lies outside the live hierarchy");
  return *p_->prepared_hierarchy->auxiliary[static_cast<std::size_t>(level)];
}

template <int Dim>
void AmrSystem<Dim>::add_prepared_amr_poisson_rhs(int level, MultiFab<Dim>& rhs) {
  p_->ensure_engine();
  if (all_reduce_min(static_cast<long>(level)) != all_reduce_max(static_cast<long>(level)))
    throw std::invalid_argument("prepared AMR Poisson RHS levels differ between MPI ranks");
  if (level < 0 || static_cast<std::size_t>(level) >= p_->prepared_hierarchy->levels.size())
    throw std::out_of_range("prepared AMR Poisson RHS level lies outside the live hierarchy");
  p_->prepared_hierarchy->levels[static_cast<std::size_t>(level)].add_poisson_rhs(rhs);
}

template <int Dim>
void AmrSystem<Dim>::install_block_state_route(const std::string& name,
                                               const std::string& state_identity) {
  require_amr_assembling(p_->lifecycle, "install_block_state_route");
  if (!p_->blocks.empty())
    throw std::logic_error("AmrSystem state routes must be installed before their block");
  p_->boundary_registry.install_state_route(name, state_identity);
}

template <int Dim>
void AmrSystem<Dim>::install_field_storage_route(const std::string& field_identity,
                                                 const std::string& provider_slot) {
  require_amr_assembling(p_->lifecycle, "install_field_storage_route");
  p_->boundary_registry.install_field_storage_route(field_identity, provider_slot);
}

template <int Dim>
void AmrSystem<Dim>::register_elliptic_field(const std::string& block_name,
                                             const std::string& provider_key,
                                             const std::vector<int>& output_components,
                                             int gradient_sign) {
  require_amr_assembling(p_->lifecycle, "register_elliptic_field");
  if (p_->engine)
    throw std::runtime_error(
        "AmrSystem cannot register a named elliptic field after hierarchy materialization");
  if (provider_key.empty())
    throw std::invalid_argument("AmrSystem named elliptic field identity must be non-empty");
  (void)p_->block(block_name);
  const runtime::field::NamedFieldOutput<Dim> output(output_components, gradient_sign);
  (void)output;
  throw std::logic_error(
      "AmrSystem exact-ranked named elliptic fields require an installed "
      "dimension-qualified hierarchy field-solver provider");
}

template <int Dim>
void AmrSystem<Dim>::set_block_elliptic_field(
    const std::string& block_name, const std::string& field,
    std::function<void(const MultiFab<Dim>&, MultiFab<Dim>&)> rhs) {
  require_amr_assembling(p_->lifecycle, "set_block_elliptic_field");
  if (p_->engine)
    throw std::runtime_error(
        "AmrSystem cannot install a named elliptic RHS after hierarchy materialization");
  if (field.empty() || !rhs)
    throw std::invalid_argument(
        "AmrSystem named elliptic RHS requires a field identity and prepared closure");
  (void)p_->block(block_name);
  throw std::logic_error(
      "AmrSystem exact-ranked named elliptic fields require an installed "
      "dimension-qualified hierarchy field-solver provider");
}

template <int Dim>
void AmrSystem<Dim>::install_hyperbolic_boundary(
    const std::string& name, const std::string& identity, int required_depth,
    const std::vector<std::string>& face_types, const std::vector<double>& face_values,
    const std::vector<std::string>& face_identities,
    const std::vector<std::string>& component_roles, const std::string& state_identity,
    const std::vector<std::string>& face_representations,
    const std::vector<std::string>& face_converter_identities,
    const std::vector<std::vector<std::string>>& face_analytic_opcodes,
    const std::vector<std::vector<double>>& face_analytic_literals,
    const std::vector<std::string>& face_analytic_clocks) {
  require_amr_assembling(p_->lifecycle, "install_hyperbolic_boundary");
  if (!p_->blocks.empty() || p_->engine)
    throw std::logic_error("AmrSystem boundaries must be prepared before their block");
  p_->boundary_registry.install_boundary(
      name, identity, required_depth, face_types, face_values, face_identities, component_roles,
      state_identity, face_representations, face_converter_identities, face_analytic_opcodes,
      face_analytic_literals, face_analytic_clocks);
}

template <int Dim>
void AmrSystem<Dim>::install_prepared_hyperbolic_boundary(
    const std::string& name, const std::string& identity, int required_depth,
    const std::string& state_identity, std::shared_ptr<const HyperbolicBoundary> boundary) {
  require_amr_assembling(p_->lifecycle, "install_prepared_hyperbolic_boundary");
  if (!p_->blocks.empty() || p_->engine)
    throw std::logic_error("AmrSystem boundaries must be prepared before their block");
  p_->boundary_registry.install_boundary(name, identity, required_depth, state_identity,
                                         std::move(boundary));
}

template <int Dim>
void AmrSystem<Dim>::discard_hyperbolic_boundaries() {
  require_amr_assembling(p_->lifecycle, "discard_hyperbolic_boundaries");
  if (!p_->blocks.empty())
    throw std::logic_error("AmrSystem cannot discard state routes after block publication");
  p_->boundary_registry.discard_transaction();
}

template <int Dim>
void AmrSystem<Dim>::set_density(const std::string& name, const std::vector<double>& density) {
  typename Impl::BlockSpec& block = p_->block(name);
  if (density.size() != checked_cells(p_->cfg.index_domain()))
    throw std::invalid_argument("AmrSystem density differs from the exact coarse shape");
  if (p_->engine)
    write_component(p_->engine->hierarchy().state(0), p_->cfg.index_domain(), density, 0);
  p_->discard_level_evaluations();
  block.density = density;
  block.has_density = true;
}

template <int Dim>
void AmrSystem<Dim>::set_conservative_state(const std::string& name,
                                            const std::vector<double>& state) {
  typename Impl::BlockSpec& block = p_->block(name);
  const std::size_t cells = checked_cells(p_->cfg.index_domain());
  if (state.size() != static_cast<std::size_t>(block.ncomp) * cells)
    throw std::invalid_argument("AmrSystem state differs from the exact coarse shape");
  if (p_->engine)
    write_field(p_->engine->hierarchy().state(0), p_->cfg.index_domain(), state, block.ncomp);
  p_->discard_level_evaluations();
  block.state = state;
  block.has_state = true;
}

template <int Dim>
void AmrSystem<Dim>::add_dt_bound(const std::string& label, std::function<double()> evaluate) {
  require_amr_assembling(p_->lifecycle, "add_dt_bound");
  if (label.empty() || !evaluate)
    throw std::invalid_argument("AmrSystem dt bound requires a label and provider");
  p_->dt_bounds.push_back(typename Impl::GlobalDtBound{label, std::move(evaluate)});
}

template <int Dim>
std::string AmrSystem<Dim>::last_dt_bound() const {
  return p_->last_dt_reason;
}

template <int Dim>
void AmrSystem<Dim>::step(double dt) {
  p_->program.require_step_installed("AmrSystem::step");
  runtime::program::ProfileScope scope(p_->program.profiler_, "step");
  p_->program.profiler_.count("steps");
  p_->execute_transaction([&] {
    p_->program.dispatch_cadence_step(p_->accepted_time, p_->macro_step, dt, "AmrSystem");
  });
  p_->discard_level_evaluations();
}

template <int Dim>
void AmrSystem<Dim>::advance(double dt, int nsteps) {
  p_->program.require_step_installed("AmrSystem::advance");
  if (nsteps < 0)
    throw std::invalid_argument("AmrSystem::advance requires a non-negative step count");
  for (int step_index = 0; step_index < nsteps; ++step_index)
    step(dt);
}

template <int Dim>
double AmrSystem<Dim>::step_cfl(double cfl, double speed_floor, double max_dt, double min_dt) {
  std::string request_contract;
  std::exception_ptr request_error;
  long request_failure = 0;
  try {
    p_->program.require_step_installed("AmrSystem::step_cfl");
    if (!std::isfinite(cfl) || cfl <= 0.0 || !std::isfinite(speed_floor) || speed_floor <= 0.0)
      throw std::invalid_argument("AmrSystem::step_cfl requires positive finite CFL inputs");
    if (std::isnan(max_dt) || max_dt <= 0.0 || !std::isfinite(min_dt) || min_dt < 0.0)
      throw std::invalid_argument("AmrSystem::step_cfl received invalid strategy bounds");
    if (p_->blocks.size() != 1 || !p_->prepared_block)
      throw std::logic_error("AmrSystem::step_cfl requires one retained generated block");

    ExactContractBuilder contract;
    contract.text("pops.amr-system.step-cfl-request")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .scalar(cfl)
        .scalar(speed_floor)
        .scalar(max_dt)
        .scalar(min_dt)
        .bytes(p_->prepared_block->collective_contract)
        .scalar(static_cast<std::uint64_t>(p_->dt_bounds.size()));
    for (const typename Impl::GlobalDtBound& bound : p_->dt_bounds)
      contract.text(bound.label);
    contract.scalar(static_cast<bool>(p_->program.dt_bound_));
    request_contract = std::move(contract).release();
  } catch (...) {
    request_failure = 1;
    request_error = std::current_exception();
  }
  if (all_reduce_max(request_failure) != 0) {
    if (request_error)
      std::rethrow_exception(request_error);
    throw std::runtime_error("AmrSystem::step_cfl request validation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-step-cfl-request"), std::string_view(request_contract)}}))
    throw std::invalid_argument(
        "AmrSystem::step_cfl inputs or prepared bound authorities differ between MPI ranks");

  p_->ensure_engine();
  const long invalid_hierarchy =
      !p_->prepared_hierarchy || p_->prepared_hierarchy->levels.empty() ? 1L : 0L;
  if (all_reduce_max(invalid_hierarchy) != 0)
    throw std::logic_error("AmrSystem::step_cfl requires one live prepared hierarchy graph");

  enum class BoundKind : std::int32_t {
    degenerate,
    transport,
    source_frequency,
    stability_dt,
    global,
    program,
    maximum_dt,
  };
  const typename Impl::BlockSpec& block = p_->blocks.front();
  double selected = std::numeric_limits<double>::infinity();
  BoundKind reason_kind = BoundKind::degenerate;
  std::size_t global_reason_index = std::numeric_limits<std::size_t>::max();
  for (std::size_t level = 0; level < p_->prepared_hierarchy->levels.size(); ++level) {
    const Box<Dim>& domain = p_->engine->hierarchy().layout(level).domain();
    const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(domain, p_->cfg.lower, p_->cfg.upper);
    Real spacing = geometry.spacing(0);
    for (int axis = 1; axis < Dim; ++axis)
      spacing = std::min(spacing, geometry.spacing(axis));

    const typename Impl::level_block_type& prepared_level = p_->prepared_hierarchy->levels[level];
    const Real speed = std::max(prepared_level.maximum_speed(), static_cast<Real>(speed_floor));
    double level_dt = cfl * static_cast<double>(spacing) * block.substeps /
                      (static_cast<double>(block.stride) * static_cast<double>(speed));
    const char* level_reason = "transport";
    if (const std::optional<Real> frequency = prepared_level.source_frequency();
        frequency && *frequency > Real(0)) {
      const double source_dt =
          cfl * block.substeps /
          (static_cast<double>(block.stride) * static_cast<double>(*frequency));
      if (source_dt < level_dt) {
        level_dt = source_dt;
        level_reason = "source_frequency";
      }
    }
    if (const std::optional<Real> admissible = prepared_level.stability_dt();
        admissible && *admissible > Real(0)) {
      const double stability_dt =
          static_cast<double>(*admissible) * block.substeps / static_cast<double>(block.stride);
      if (stability_dt < level_dt) {
        level_dt = stability_dt;
        level_reason = "stability_dt";
      }
    }
    if (level_dt < selected) {
      selected = level_dt;
      if (std::string_view(level_reason) == "source_frequency")
        reason_kind = BoundKind::source_frequency;
      else if (std::string_view(level_reason) == "stability_dt")
        reason_kind = BoundKind::stability_dt;
      else
        reason_kind = BoundKind::transport;
    }
  }
  if (!std::isfinite(selected))
    throw std::runtime_error("AmrSystem::step_cfl found no finite generated stability bound");

  for (std::size_t bound_index = 0; bound_index < p_->dt_bounds.size(); ++bound_index) {
    const typename Impl::GlobalDtBound& bound = p_->dt_bounds[bound_index];
    double candidate = std::numeric_limits<double>::infinity();
    std::exception_ptr bound_error;
    long bound_failure = 0;
    try {
      candidate = bound.evaluate();
    } catch (...) {
      bound_failure = 1;
      bound_error = std::current_exception();
    }
    if (all_reduce_max(bound_failure) != 0) {
      if (bound_error)
        std::rethrow_exception(bound_error);
      throw std::runtime_error("AmrSystem global dt-bound evaluation failed collectively");
    }
    const long active = std::isfinite(candidate) && candidate > 0.0 ? 1L : 0L;
    if (all_reduce_min(active) != all_reduce_max(active))
      throw std::invalid_argument("AmrSystem global dt-bound activity differs between MPI ranks");
    if (active == 0)
      candidate = std::numeric_limits<double>::infinity();
    candidate = all_reduce_min(candidate);
    if (candidate < selected) {
      selected = candidate;
      reason_kind = BoundKind::global;
      global_reason_index = bound_index;
    }
  }
  const bool has_program_bound = static_cast<bool>(p_->program.dt_bound_);
  if (has_program_bound) {
    double candidate = std::numeric_limits<double>::infinity();
    std::exception_ptr bound_error;
    long bound_failure = 0;
    try {
      candidate = static_cast<double>(p_->program.dt_bound_(static_cast<Real>(cfl)));
    } catch (...) {
      bound_failure = 1;
      bound_error = std::current_exception();
    }
    if (all_reduce_max(bound_failure) != 0) {
      if (bound_error)
        std::rethrow_exception(bound_error);
      throw std::runtime_error("AmrSystem Program dt-bound evaluation failed collectively");
    }
    const long active = std::isfinite(candidate) && candidate > 0.0 ? 1L : 0L;
    if (all_reduce_min(active) != all_reduce_max(active))
      throw std::invalid_argument("AmrSystem Program dt-bound activity differs between MPI ranks");
    if (active == 0)
      candidate = std::numeric_limits<double>::infinity();
    candidate = all_reduce_min(candidate);
    if (std::isfinite(candidate) && candidate > 0.0 && candidate < selected) {
      selected = candidate;
      reason_kind = BoundKind::program;
    }
  }
  if (max_dt < selected) {
    selected = max_dt;
    reason_kind = BoundKind::maximum_dt;
  }
  if (selected < min_dt)
    throw std::runtime_error("AmrSystem::step_cfl stability bound is below declared min_dt");

  std::string reason;
  std::string decision_contract;
  std::exception_ptr decision_error;
  long decision_failure = 0;
  try {
    switch (reason_kind) {
      case BoundKind::transport:
        reason = "transport:" + block.name;
        break;
      case BoundKind::source_frequency:
        reason = "source_frequency:" + block.name;
        break;
      case BoundKind::stability_dt:
        reason = "stability_dt:" + block.name;
        break;
      case BoundKind::global:
        if (global_reason_index >= p_->dt_bounds.size())
          throw std::logic_error("AmrSystem::step_cfl lost its selected global bound identity");
        reason = "global:" + p_->dt_bounds[global_reason_index].label;
        break;
      case BoundKind::program:
        reason = "program:dt_bound";
        break;
      case BoundKind::maximum_dt:
        reason = "strategy:max_dt";
        break;
      case BoundKind::degenerate:
        reason = "degenerate";
        break;
    }
    ExactContractBuilder contract;
    contract.text("pops.amr-system.step-cfl-decision")
        .scalar(std::uint32_t{1})
        .scalar(selected)
        .scalar(reason_kind)
        .scalar(static_cast<std::uint64_t>(global_reason_index))
        .text(reason);
    decision_contract = std::move(contract).release();
  } catch (...) {
    decision_failure = 1;
    decision_error = std::current_exception();
  }
  if (all_reduce_max(decision_failure) != 0) {
    if (decision_error)
      std::rethrow_exception(decision_error);
    throw std::runtime_error("AmrSystem::step_cfl decision preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-step-cfl-decision"), std::string_view(decision_contract)}}))
    throw std::runtime_error("AmrSystem::step_cfl selected different bounds across MPI ranks");
  p_->last_dt_reason = std::move(reason);
  step(selected);
  return selected;
}

template <int Dim>
void AmrSystem<Dim>::begin_step_transaction() {
  p_->ensure_engine();
  if (p_->external_step_transaction)
    throw std::runtime_error("AmrSystem step transaction is already active");
  p_->external_step_transaction = std::make_unique<typename Impl::AcceptedSnapshot>(*p_);
  p_->external_step_committed = false;
}

template <int Dim>
void AmrSystem<Dim>::commit_step_transaction() {
  if (!p_->external_step_transaction || p_->external_step_committed)
    throw std::runtime_error("AmrSystem has no active uncommitted step transaction");
  p_->external_step_committed = true;
}

template <int Dim>
void AmrSystem<Dim>::finalize_step_transaction() {
  if (!p_->external_step_transaction || !p_->external_step_committed)
    throw std::runtime_error("AmrSystem has no committed step transaction");
  p_->external_step_transaction.reset();
  p_->external_step_committed = false;
}

template <int Dim>
void AmrSystem<Dim>::rollback_step_transaction() {
  if (!p_->external_step_transaction)
    throw std::runtime_error("AmrSystem has no active step transaction");
  p_->external_step_transaction->restore(*p_);
  p_->external_step_transaction.reset();
  p_->external_step_committed = false;
}

template <int Dim>
bool AmrSystem<Dim>::has_active_step_transaction() const noexcept {
  return static_cast<bool>(p_->external_step_transaction);
}

template <int Dim>
void AmrSystem<Dim>::restore_active_step_transaction_for_program() {
  if (!p_->external_step_transaction)
    throw std::runtime_error("AmrSystem has no outer Program rollback image");
  p_->external_step_transaction->restore(*p_);
}

template <int Dim>
std::map<std::string, double> AmrSystem<Dim>::step_change_l2() const {
  if (!p_->external_step_transaction || !p_->engine || !p_->external_step_transaction->engine)
    throw std::runtime_error("AmrSystem::step_change_l2 requires an active transaction");
  if (p_->engine->hierarchy().num_levels() != 1)
    throw std::runtime_error(
        "AmrSystem::step_change_l2 requires a prepared composite coverage provider");
  const MultiFab<Dim>& current = p_->engine->hierarchy().state(0);
  const MultiFab<Dim>& previous = p_->external_step_transaction->engine->hierarchy.state(0);
  const double value = std::sqrt(cell_measure(p_->cfg, p_->cfg.index_domain()) *
                                 static_cast<double>(difference_sum_sq_all(current, previous)));
  return {{p_->blocks.front().name, value}};
}

template <int Dim>
void AmrSystem<Dim>::install_program_step(std::function<void(double)> step) {
  p_->program.install_unverified_step(std::move(step));
}

template <int Dim>
void AmrSystem<Dim>::install_program_hierarchy_refresh(std::function<void()> refresh) {
  p_->program.install_hierarchy_refresh(std::move(refresh), "AmrSystem");
}

template <int Dim>
void AmrSystem<Dim>::install_program_restart_hooks(std::function<void()> preflight,
                                                   std::function<void()> regrid,
                                                   std::function<void()> resync) {
  p_->program.install_restart_hooks(std::move(preflight), std::move(regrid), std::move(resync),
                                    "AmrSystem");
}

template <int Dim>
void AmrSystem<Dim>::set_program_cadence(int substeps, int stride) {
  require_amr_assembling(p_->lifecycle, "set_program_cadence");
  p_->program.set_cadence(substeps, stride, "AmrSystem");
}

template <int Dim>
int AmrSystem<Dim>::program_substeps() const {
  return p_->program.substeps_;
}

template <int Dim>
int AmrSystem<Dim>::program_stride() const {
  return p_->program.stride_;
}

template <int Dim>
double AmrSystem<Dim>::program_cadence_window_dt() const {
  return p_->program.cadence_window_dt_;
}

template <int Dim>
int AmrSystem<Dim>::program_cadence_window_steps() const {
  return p_->program.cadence_window_steps_;
}

template <int Dim>
double AmrSystem<Dim>::program_cadence_window_start_time() const {
  return p_->program.cadence_window_start_time_;
}

template <int Dim>
double AmrSystem<Dim>::program_last_dt() const {
  return static_cast<double>(p_->program.last_dt_);
}

template <int Dim>
void AmrSystem<Dim>::restore_program_cadence_window(double accumulated_dt, int held_steps,
                                                    double window_start_time,
                                                    double accepted_last_dt, double accepted_time,
                                                    int macro_step) {
  p_->program.restore_cadence_window(accumulated_dt, held_steps, window_start_time,
                                     accepted_last_dt, accepted_time, macro_step, "AmrSystem");
}

template <int Dim>
void AmrSystem<Dim>::set_program_block_map(const std::vector<int>& program_to_runtime) {
  for (std::size_t program = 0; program < program_to_runtime.size(); ++program) {
    const int block = program_to_runtime[program];
    if (block < 0 || block >= static_cast<int>(p_->blocks.size()))
      throw std::out_of_range("AmrSystem Program block map is out of range");
    for (std::size_t previous = 0; previous < program; ++previous)
      if (program_to_runtime[previous] == block)
        throw std::invalid_argument("AmrSystem Program block map contains duplicate routes");
  }
  p_->program.block_map_ = program_to_runtime;
}

template <int Dim>
const std::vector<int>& AmrSystem<Dim>::program_block_map() const {
  return p_->program.block_map_;
}

template <int Dim>
std::string AmrSystem<Dim>::installed_program_hash() const {
  return p_->program.installed_hash_;
}

template <int Dim>
void AmrSystem<Dim>::seed_program_params(int block, const std::vector<double>& defaults) {
  p_->program.seed_params(block, defaults);
}

template <int Dim>
void AmrSystem<Dim>::set_program_params(int block, const std::vector<double>& values) {
  p_->program.set_params(block, values, "AmrSystem");
}

template <int Dim>
RuntimeParams AmrSystem<Dim>::program_params(int block) const {
  return p_->program.params(block);
}

template <int Dim>
runtime::amr::AmrRuntime<Dim>* AmrSystem<Dim>::engine() const {
  p_->ensure_engine();
  return p_->engine.get();
}

template <int Dim>
bool AmrSystem<Dim>::uses_runtime_engine() const {
  return static_cast<bool>(p_->engine);
}

template <int Dim>
runtime::program::Profiler& AmrSystem<Dim>::profiler_handle() {
  return p_->program.profiler_;
}

template <int Dim>
runtime::program::ProgramRuntimeState<Dim>& AmrSystem<Dim>::program_runtime_state_() {
  return p_->program;
}

template <int Dim>
void AmrSystem<Dim>::record_program_diagnostic(const std::string& name, double value) {
  p_->program.record_diagnostic(name, static_cast<Real>(value));
}

template <int Dim>
void AmrSystem<Dim>::record_program_balance_term(const std::string& route, const std::string& term,
                                                 double value) {
  p_->program.record_balance_term(route, term, static_cast<Real>(value), "AmrSystem");
}

template <int Dim>
bool AmrSystem<Dim>::program_balance_consumer_is_due(const std::string& contract,
                                                     const std::string& route, int every_n) const {
  return p_->program.balance_consumer_is_due(contract, route, every_n, "AmrSystem");
}

template <int Dim>
double AmrSystem<Dim>::program_diagnostic(const std::string& name) const {
  return static_cast<double>(p_->program.diagnostic(name, "AmrSystem"));
}

template <int Dim>
std::map<std::string, double> AmrSystem<Dim>::program_diagnostics() const {
  std::map<std::string, double> result;
  for (const auto& [name, value] : p_->program.diagnostics())
    result.emplace(name, static_cast<double>(value));
  return result;
}

template <int Dim>
void AmrSystem<Dim>::begin_step_projection_report() {
  p_->program.begin_step_projection_report();
}

template <int Dim>
void AmrSystem<Dim>::note_step_projection(const std::string& name) {
  p_->program.note_step_projection(name);
}

template <int Dim>
std::vector<std::string> AmrSystem<Dim>::consume_step_projections() {
  return p_->program.consume_step_projections();
}

template <int Dim>
void AmrSystem<Dim>::mark_bound() {
  if (p_->lifecycle.frozen())
    p_->lifecycle.to_bound();
  p_->ensure_engine();
  const auto& routes = p_->boundary_registry.state_routes();
  if (!routes.empty() && routes.size() != p_->blocks.size())
    throw std::runtime_error("AmrSystem state routes do not exactly cover its blocks");
  for (const typename Impl::BlockSpec& block : p_->blocks) {
    if (!routes.empty() && !routes.contains(block.name))
      throw std::runtime_error("AmrSystem block lacks its exact state route");
    const auto* boundary = p_->boundary_registry.find_boundary(block.name);
    if (boundary == nullptr)
      continue;
    if (boundary->authority->ncomp() != block.ncomp ||
        boundary->authority->periodic_axes() != p_->cfg.periodicity)
      throw std::runtime_error("AmrSystem boundary differs from its block/domain contract");
    for (std::size_t level = 0; level < p_->engine->hierarchy().num_levels(); ++level)
      for (int axis = 0; axis < Dim; ++axis)
        if (p_->engine->hierarchy().state(level).ghosts()[axis] < boundary->required_depth)
          throw std::runtime_error("AmrSystem boundary depth exceeds level storage");
  }
  p_->lifecycle.to_bound();
}

template <int Dim>
std::string AmrSystem<Dim>::lifecycle_state() const {
  return p_->lifecycle.state(p_->macro_step);
}

template <int Dim>
Extent<Dim> AmrSystem<Dim>::spatial_shape() const {
  return p_->cfg.shape;
}

template <int Dim>
double AmrSystem<Dim>::time() const {
  return p_->accepted_time;
}

template <int Dim>
int AmrSystem<Dim>::macro_step() const {
  return p_->macro_step;
}

template <int Dim>
void AmrSystem<Dim>::set_clock(double accepted_time, int macro_step) {
  if (!std::isfinite(accepted_time) || macro_step < 0)
    throw std::invalid_argument("AmrSystem clock requires finite time and non-negative step");
  p_->program.consume_cadence_clock_restore(accepted_time, macro_step, "AmrSystem");
  p_->accepted_time = accepted_time;
  p_->macro_step = macro_step;
}

template <int Dim>
void AmrSystem<Dim>::enable_profiling() {
  p_->program.profiler_.enable();
}

template <int Dim>
void AmrSystem<Dim>::disable_profiling() {
  p_->program.profiler_.disable();
}

template <int Dim>
bool AmrSystem<Dim>::is_profiling() const {
  return p_->program.profiler_.enabled();
}

template <int Dim>
void AmrSystem<Dim>::reset_profiling() {
  p_->program.profiler_.reset();
}

template <int Dim>
std::string AmrSystem<Dim>::profile_report() const {
  return p_->program.profiler_.report();
}

template <int Dim>
int AmrSystem<Dim>::n_blocks() const {
  return static_cast<int>(p_->blocks.size());
}

template <int Dim>
std::vector<std::string> AmrSystem<Dim>::block_names() const {
  std::vector<std::string> names;
  names.reserve(p_->blocks.size());
  for (const typename Impl::BlockSpec& block : p_->blocks)
    names.push_back(block.name);
  return names;
}

template <int Dim>
EffectiveOptionsReport AmrSystem<Dim>::effective_options_report() const {
  EffectiveOptionsReport report;
  report.runtime = "amr_system";
  report.has_amr = true;
  report.topology.dimension = Dim;
  report.topology.periodicity.reserve(Dim);
  for (int axis = 0; axis < Dim; ++axis)
    report.topology.periodicity.push_back(p_->cfg.periodicity[axis]);
  report.blocks.reserve(p_->blocks.size());
  for (const typename Impl::BlockSpec& block : p_->blocks) {
    EffectiveBlockOptions row;
    row.name = block.name;
    row.ncomp = block.ncomp;
    row.n_ghost = block.required_ghost_depth;
    row.gamma = block.gamma;
    row.substeps = block.substeps;
    row.stride = block.stride;
    row.time = block.time;
    report.blocks.push_back(std::move(row));
  }
  return report;
}

template <int Dim>
int AmrSystem<Dim>::n_levels() {
  p_->ensure_engine();
  return static_cast<int>(p_->engine->hierarchy().num_levels());
}

template <int Dim>
int AmrSystem<Dim>::max_levels() {
  return p_->cfg.level_count;
}

template <int Dim>
int AmrSystem<Dim>::configured_n_levels() {
  return p_->cfg.level_count;
}

template <int Dim>
int AmrSystem<Dim>::n_vars() {
  if (p_->blocks.size() != 1)
    throw std::runtime_error("AmrSystem::n_vars requires exactly one block");
  return p_->blocks.front().ncomp;
}

template <int Dim>
int AmrSystem<Dim>::block_n_vars(const std::string& name) {
  return p_->block(name).ncomp;
}

template <int Dim>
int AmrSystem<Dim>::n_patches() {
  p_->ensure_engine();
  const auto& hierarchy = p_->engine->hierarchy();
  return static_cast<int>(hierarchy.layout(hierarchy.num_levels() - 1).patches().size());
}

template <int Dim>
std::vector<AmrPatch<Dim>> AmrSystem<Dim>::patch_boxes() {
  p_->ensure_engine();
  std::vector<AmrPatch<Dim>> result;
  for (std::size_t level = 1; level < p_->engine->hierarchy().num_levels(); ++level)
    for (const Box<Dim>& box : p_->engine->hierarchy().layout(level).patches().boxes())
      result.push_back({static_cast<int>(level), box});
  return result;
}

template <int Dim>
std::vector<AmrPatch<Dim>> AmrSystem<Dim>::output_geometry_boxes() {
  p_->ensure_engine();
  std::vector<AmrPatch<Dim>> result;
  for (std::size_t level = 0; level < p_->engine->hierarchy().num_levels(); ++level)
    for (const Box<Dim>& box : p_->engine->hierarchy().layout(level).patches().boxes())
      result.push_back({static_cast<int>(level), box});
  return result;
}

template <int Dim>
int AmrSystem<Dim>::coarse_local_boxes() {
  p_->ensure_engine();
  return static_cast<int>(p_->engine->hierarchy().state(0).local_size());
}

template <int Dim>
int AmrSystem<Dim>::coarse_total_boxes() {
  p_->ensure_engine();
  return static_cast<int>(p_->engine->hierarchy().layout(0).patches().size());
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::level_state(int level) {
  return level_state_global(level);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::level_state_global(int level) {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("AmrSystem level is out of range");
  const auto& hierarchy = p_->engine->hierarchy();
  return gather_field(hierarchy.state(static_cast<std::size_t>(level)),
                      hierarchy.layout(static_cast<std::size_t>(level)).domain(),
                      hierarchy.state(static_cast<std::size_t>(level)).ncomp());
}

template <int Dim>
void AmrSystem<Dim>::set_level_state(int level, const std::vector<double>& state) {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("AmrSystem level is out of range");
  auto& hierarchy = p_->engine->hierarchy();
  write_field(hierarchy.state(static_cast<std::size_t>(level)),
              hierarchy.layout(static_cast<std::size_t>(level)).domain(), state,
              hierarchy.state(static_cast<std::size_t>(level)).ncomp());
  p_->discard_level_evaluations();
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::block_level_state(const std::string& name, int level) {
  (void)p_->block(name);
  return level_state(level);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::block_level_state_global(const std::string& name, int level) {
  (void)p_->block(name);
  return level_state_global(level);
}

template <int Dim>
void AmrSystem<Dim>::set_block_level_state(const std::string& name, int level,
                                           const std::vector<double>& state) {
  (void)p_->block(name);
  set_level_state(level, state);
}

template <int Dim>
std::vector<OutputPiece<Dim>> AmrSystem<Dim>::output_state_local_pieces(const std::string& name,
                                                                        int level) {
  (void)p_->block(name);
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("AmrSystem output level is out of range");
  const MultiFab<Dim>& state = p_->engine->hierarchy().state(static_cast<std::size_t>(level));
  return output_local_pieces(state, level, state.distribution().replicated());
}

template <int Dim>
std::vector<OutputPiece<Dim>> AmrSystem<Dim>::output_state_root_pieces(const ObserverMpiLane& lane,
                                                                       const std::string& name,
                                                                       int level) {
  return output_pieces_to_root(
      lane, detail::output_collective_identity("amr_system", "state", name, level),
      [&] { return output_state_local_pieces(name, level); });
}

template <int Dim>
std::vector<int> AmrSystem<Dim>::level_owner_ranks(int level) {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("AmrSystem owner level is out of range");
  const auto& distribution =
      p_->engine->hierarchy().layout(static_cast<std::size_t>(level)).distribution();
  if (distribution.replicated())
    throw std::runtime_error("replicated AMR levels have no unique owner ranks");
  std::vector<int> result;
  result.reserve(distribution.owners().size());
  for (const Index<Dim>& owner : distribution.owners())
    result.push_back(static_cast<int>(distribution.rank_space().linear_rank(owner)));
  return result;
}

template <int Dim>
double AmrSystem<Dim>::mass() {
  if (p_->blocks.empty())
    throw std::logic_error("AmrSystem has no block");
  return mass(p_->blocks.front().name);
}

template <int Dim>
double AmrSystem<Dim>::mass(const std::string& name) {
  (void)p_->block(name);
  p_->ensure_engine();
  if (p_->engine->hierarchy().num_levels() != 1)
    throw std::runtime_error("AmrSystem mass requires a prepared composite coverage provider");
  return static_cast<double>(reduce_sum(p_->engine->hierarchy().state(0), 0)) *
         cell_measure(p_->cfg, p_->cfg.index_domain());
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::density() {
  if (p_->blocks.empty())
    throw std::logic_error("AmrSystem has no block");
  return density(p_->blocks.front().name);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::density(const std::string& name) {
  (void)p_->block(name);
  p_->ensure_engine();
  return gather_field(p_->engine->hierarchy().state(0), p_->cfg.index_domain(), 1);
}

template <int Dim>
std::vector<std::uint8_t> AmrSystem<Dim>::program_accepted_state() const {
  return p_->program_accepted_bytes;
}

template <int Dim>
void AmrSystem<Dim>::copy_program_accepted_state_into(std::vector<std::uint8_t>& state) const {
  state = p_->program_accepted_bytes;
}

template <int Dim>
void AmrSystem<Dim>::restore_program_accepted_state(const std::vector<std::uint8_t>& state) {
  if (p_->program_accepted_revision == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("AmrSystem Program accepted-state revision overflow");
  p_->program_accepted_bytes = state;
  ++p_->program_accepted_revision;
}

template <int Dim>
void AmrSystem<Dim>::restore_checkpoint_accepted_state(const std::vector<std::uint8_t>& state) {
  restore_program_accepted_state(state);
}

template <int Dim>
std::uint64_t AmrSystem<Dim>::program_accepted_state_revision() const {
  return p_->program_accepted_revision;
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_accepted_state_manifest() const {
  return {};
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_clock_manifest() const {
  return {};
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_temporal_partition_manifest() const {
  return {};
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_flux_ledger_manifest() const {
  return {};
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_interface_flux_ledger_manifest()
    const {
  return {};
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_sync_manifest() const {
  return {};
}

template <int Dim>
const std::vector<CouplingOperatorView>& AmrSystem<Dim>::coupled_operators() const {
  return p_->coupling_views;
}

template AmrSystem<kNativeDimension>::AmrSystem(const AmrSystemConfig<kNativeDimension>&);
template AmrSystem<kNativeDimension>::~AmrSystem();
template AmrSystem<kNativeDimension>::AmrSystem(AmrSystem&&) noexcept;
template AmrSystem<kNativeDimension>& AmrSystem<kNativeDimension>::operator=(AmrSystem&&) noexcept;
template void AmrSystem<kNativeDimension>::add_block(
    const std::string&, const ModelSpec&, const std::string&, const std::string&,
    const std::string&, const std::string&, int, int, const std::vector<std::string>&,
    const std::vector<std::string>&, const NewtonOptions&, bool, double, double, bool);
template void AmrSystem<kNativeDimension>::set_compiled_block(
    int, double, int, AmrCompiledBlockBuilder<kNativeDimension>, const std::string&, bool,
    const std::string&, int, const std::vector<std::string>&, const std::vector<std::string>&,
    double, double, bool);
template void AmrSystem<kNativeDimension>::install_prepared_amr_block(
    PreparedAmrSystemBlock<kNativeDimension>);
template void AmrSystem<kNativeDimension>::refresh_prepared_amr_levels();
template const PreparedAmrLevelEvaluation<kNativeDimension>&
AmrSystem<kNativeDimension>::evaluate_prepared_amr_level(
    const runtime::multiblock::BoundaryEvaluationPoint&);
template const PreparedAmrLevelEvaluation<kNativeDimension>&
AmrSystem<kNativeDimension>::prepared_amr_level_evaluation(int) const;
template MultiFab<kNativeDimension>& AmrSystem<kNativeDimension>::prepared_amr_level_auxiliary(int);
template const MultiFab<kNativeDimension>&
AmrSystem<kNativeDimension>::prepared_amr_level_auxiliary(int) const;
template void AmrSystem<kNativeDimension>::add_prepared_amr_poisson_rhs(
    int, MultiFab<kNativeDimension>&);
template void AmrSystem<kNativeDimension>::install_block_state_route(const std::string&,
                                                                     const std::string&);
template void AmrSystem<kNativeDimension>::install_field_storage_route(const std::string&,
                                                                       const std::string&);
template void AmrSystem<kNativeDimension>::register_elliptic_field(const std::string&,
                                                                   const std::string&,
                                                                   const std::vector<int>&, int);
template void AmrSystem<kNativeDimension>::set_block_elliptic_field(
    const std::string&, const std::string&,
    std::function<void(const MultiFab<kNativeDimension>&, MultiFab<kNativeDimension>&)>);
template void AmrSystem<kNativeDimension>::install_hyperbolic_boundary(
    const std::string&, const std::string&, int, const std::vector<std::string>&,
    const std::vector<double>&, const std::vector<std::string>&, const std::vector<std::string>&,
    const std::string&, const std::vector<std::string>&, const std::vector<std::string>&,
    const std::vector<std::vector<std::string>>&, const std::vector<std::vector<double>>&,
    const std::vector<std::string>&);
template void AmrSystem<kNativeDimension>::install_prepared_hyperbolic_boundary(
    const std::string&, const std::string&, int, const std::string&,
    std::shared_ptr<const HyperbolicBoundary>);
template void AmrSystem<kNativeDimension>::discard_hyperbolic_boundaries();
template void AmrSystem<kNativeDimension>::set_density(const std::string&,
                                                       const std::vector<double>&);
template void AmrSystem<kNativeDimension>::set_conservative_state(const std::string&,
                                                                  const std::vector<double>&);
template void AmrSystem<kNativeDimension>::add_dt_bound(const std::string&,
                                                        std::function<double()>);
template std::string AmrSystem<kNativeDimension>::last_dt_bound() const;
template void AmrSystem<kNativeDimension>::step(double);
template void AmrSystem<kNativeDimension>::advance(double, int);
template double AmrSystem<kNativeDimension>::step_cfl(double, double, double, double);
template void AmrSystem<kNativeDimension>::begin_step_transaction();
template void AmrSystem<kNativeDimension>::commit_step_transaction();
template void AmrSystem<kNativeDimension>::finalize_step_transaction();
template void AmrSystem<kNativeDimension>::rollback_step_transaction();
template bool AmrSystem<kNativeDimension>::has_active_step_transaction() const noexcept;
template void AmrSystem<kNativeDimension>::restore_active_step_transaction_for_program();
template std::map<std::string, double> AmrSystem<kNativeDimension>::step_change_l2() const;
template void AmrSystem<kNativeDimension>::install_program_step(std::function<void(double)>);
template void AmrSystem<kNativeDimension>::install_program_hierarchy_refresh(std::function<void()>);
template void AmrSystem<kNativeDimension>::install_program_restart_hooks(std::function<void()>,
                                                                         std::function<void()>,
                                                                         std::function<void()>);
template void AmrSystem<kNativeDimension>::set_program_cadence(int, int);
template int AmrSystem<kNativeDimension>::program_substeps() const;
template int AmrSystem<kNativeDimension>::program_stride() const;
template double AmrSystem<kNativeDimension>::program_cadence_window_dt() const;
template int AmrSystem<kNativeDimension>::program_cadence_window_steps() const;
template double AmrSystem<kNativeDimension>::program_cadence_window_start_time() const;
template double AmrSystem<kNativeDimension>::program_last_dt() const;
template void AmrSystem<kNativeDimension>::restore_program_cadence_window(double, int, double,
                                                                          double, double, int);
template void AmrSystem<kNativeDimension>::set_program_block_map(const std::vector<int>&);
template const std::vector<int>& AmrSystem<kNativeDimension>::program_block_map() const;
template std::string AmrSystem<kNativeDimension>::installed_program_hash() const;
template void AmrSystem<kNativeDimension>::seed_program_params(int, const std::vector<double>&);
template void AmrSystem<kNativeDimension>::set_program_params(int, const std::vector<double>&);
template RuntimeParams AmrSystem<kNativeDimension>::program_params(int) const;
template runtime::amr::AmrRuntime<kNativeDimension>* AmrSystem<kNativeDimension>::engine() const;
template bool AmrSystem<kNativeDimension>::uses_runtime_engine() const;
template runtime::program::Profiler& AmrSystem<kNativeDimension>::profiler_handle();
template runtime::program::ProgramRuntimeState<kNativeDimension>&
AmrSystem<kNativeDimension>::program_runtime_state_();
template void AmrSystem<kNativeDimension>::record_program_diagnostic(const std::string&, double);
template void AmrSystem<kNativeDimension>::record_program_balance_term(const std::string&,
                                                                       const std::string&, double);
template bool AmrSystem<kNativeDimension>::program_balance_consumer_is_due(const std::string&,
                                                                           const std::string&,
                                                                           int) const;
template double AmrSystem<kNativeDimension>::program_diagnostic(const std::string&) const;
template std::map<std::string, double> AmrSystem<kNativeDimension>::program_diagnostics() const;
template void AmrSystem<kNativeDimension>::begin_step_projection_report();
template void AmrSystem<kNativeDimension>::note_step_projection(const std::string&);
template std::vector<std::string> AmrSystem<kNativeDimension>::consume_step_projections();
template void AmrSystem<kNativeDimension>::mark_bound();
template std::string AmrSystem<kNativeDimension>::lifecycle_state() const;
template Extent<kNativeDimension> AmrSystem<kNativeDimension>::spatial_shape() const;
template double AmrSystem<kNativeDimension>::time() const;
template int AmrSystem<kNativeDimension>::macro_step() const;
template void AmrSystem<kNativeDimension>::set_clock(double, int);
template void AmrSystem<kNativeDimension>::enable_profiling();
template void AmrSystem<kNativeDimension>::disable_profiling();
template bool AmrSystem<kNativeDimension>::is_profiling() const;
template void AmrSystem<kNativeDimension>::reset_profiling();
template std::string AmrSystem<kNativeDimension>::profile_report() const;
template int AmrSystem<kNativeDimension>::n_blocks() const;
template std::vector<std::string> AmrSystem<kNativeDimension>::block_names() const;
template EffectiveOptionsReport AmrSystem<kNativeDimension>::effective_options_report() const;
template int AmrSystem<kNativeDimension>::n_levels();
template int AmrSystem<kNativeDimension>::max_levels();
template int AmrSystem<kNativeDimension>::configured_n_levels();
template int AmrSystem<kNativeDimension>::n_vars();
template int AmrSystem<kNativeDimension>::block_n_vars(const std::string&);
template int AmrSystem<kNativeDimension>::n_patches();
template std::vector<AmrPatch<kNativeDimension>> AmrSystem<kNativeDimension>::patch_boxes();
template std::vector<AmrPatch<kNativeDimension>>
AmrSystem<kNativeDimension>::output_geometry_boxes();
template int AmrSystem<kNativeDimension>::coarse_local_boxes();
template int AmrSystem<kNativeDimension>::coarse_total_boxes();
template std::vector<double> AmrSystem<kNativeDimension>::level_state(int);
template std::vector<double> AmrSystem<kNativeDimension>::level_state_global(int);
template void AmrSystem<kNativeDimension>::set_level_state(int, const std::vector<double>&);
template std::vector<double> AmrSystem<kNativeDimension>::block_level_state(const std::string&,
                                                                            int);
template std::vector<double> AmrSystem<kNativeDimension>::block_level_state_global(
    const std::string&, int);
template void AmrSystem<kNativeDimension>::set_block_level_state(const std::string&, int,
                                                                 const std::vector<double>&);
template std::vector<OutputPiece<kNativeDimension>>
AmrSystem<kNativeDimension>::output_state_local_pieces(const std::string&, int);
template std::vector<OutputPiece<kNativeDimension>>
AmrSystem<kNativeDimension>::output_state_root_pieces(const ObserverMpiLane&, const std::string&,
                                                      int);
template std::vector<int> AmrSystem<kNativeDimension>::level_owner_ranks(int);
template double AmrSystem<kNativeDimension>::mass();
template double AmrSystem<kNativeDimension>::mass(const std::string&);
template std::vector<double> AmrSystem<kNativeDimension>::density();
template std::vector<double> AmrSystem<kNativeDimension>::density(const std::string&);
template std::vector<std::uint8_t> AmrSystem<kNativeDimension>::program_accepted_state() const;
template void AmrSystem<kNativeDimension>::copy_program_accepted_state_into(
    std::vector<std::uint8_t>&) const;
template void AmrSystem<kNativeDimension>::restore_program_accepted_state(
    const std::vector<std::uint8_t>&);
template void AmrSystem<kNativeDimension>::restore_checkpoint_accepted_state(
    const std::vector<std::uint8_t>&);
template std::uint64_t AmrSystem<kNativeDimension>::program_accepted_state_revision() const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::program_accepted_state_manifest() const;
template std::vector<std::vector<std::string>> AmrSystem<kNativeDimension>::program_clock_manifest()
    const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::program_temporal_partition_manifest() const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::program_flux_ledger_manifest() const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::program_interface_flux_ledger_manifest() const;
template std::vector<std::vector<std::string>> AmrSystem<kNativeDimension>::program_sync_manifest()
    const;
template const std::vector<CouplingOperatorView>& AmrSystem<kNativeDimension>::coupled_operators()
    const;

}  // namespace pops
