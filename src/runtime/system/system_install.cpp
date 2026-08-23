/// @file
/// @brief Atomic installation of exact-ranked Uniform System providers.

#include "system_impl.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/coupling/source/coupled_source_program.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/runtime/builders/compiled/native_loader.hpp>
#include <pops/runtime/named_field_output.hpp>
#include <pops/runtime/program/external_riemann_brick.hpp>
#include <pops/runtime/program/prepared_scalar_boundary_session.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace pops {
namespace {

std::string prepared_system_boundary_package_identity(std::string_view operation,
                                                      std::string_view block,
                                                      const PreparedBoundaryComponentSpec& spec) {
  ExactContractBuilder contract;
  contract.text("pops.system.prepared-boundary-component")
      .scalar(std::uint32_t{1})
      .text(operation)
      .text(block);
  append_prepared_boundary_component_contract(contract, spec);
  return std::move(contract).release();
}

std::string prepared_system_boundary_component_contract(const PreparedBoundaryComponentSpec& spec) {
  ExactContractBuilder contract;
  append_prepared_boundary_component_contract(contract, spec);
  return std::move(contract).release();
}

std::string prepared_system_boundary_component_authority_contract(
    const component::LoadedComponent* owner, PopsNativeInterfaceIdV1 selected_interface,
    std::uint32_t selected_version) {
  if (owner == nullptr)
    throw std::invalid_argument("prepared System boundary component has no loaded authority");
  const PopsComponentApiV1& api = owner->api();
  const auto require_text = [](const char* value, const char* field) -> std::string_view {
    if (value == nullptr || *value == '\0')
      throw std::invalid_argument(std::string("prepared System boundary component has empty ") +
                                  field);
    return value;
  };
  if (api.struct_size < sizeof(PopsComponentApiV1) ||
      api.protocol_abi != POPS_COMPONENT_PROTOCOL_ABI_V1 || api.interface_count == 0 ||
      api.interfaces == nullptr)
    throw std::invalid_argument("prepared System boundary component API authority is malformed");

  struct InterfaceAuthority {
    std::uint32_t id = 0;
    std::uint32_t version = 0;
    std::uint32_t table_size = 0;
    std::uint32_t header_size = 0;
    std::uint32_t header_abi = 0;
    bool has_prepare = false;
    bool has_destroy = false;
  };
  std::vector<InterfaceAuthority> interfaces;
  interfaces.reserve(api.interface_count);
  bool selected = false;
  for (std::size_t index = 0; index < api.interface_count; ++index) {
    const PopsComponentInterfaceEntryV1& row = api.interfaces[index];
    if (row.table == nullptr || row.table_size < sizeof(PopsComponentTableHeaderV1))
      throw std::invalid_argument(
          "prepared System boundary component interface authority is truncated");
    const auto& header = *static_cast<const PopsComponentTableHeaderV1*>(row.table);
    if (header.struct_size < sizeof(PopsComponentTableHeaderV1) ||
        header.struct_size > row.table_size || header.abi_version != api.protocol_abi ||
        header.interface_id != row.interface_id ||
        header.interface_version != row.interface_version ||
        ((header.prepare == nullptr) != (header.destroy == nullptr)))
      throw std::invalid_argument(
          "prepared System boundary component interface authority is malformed");
    interfaces.push_back({static_cast<std::uint32_t>(row.interface_id), row.interface_version,
                          row.table_size, header.struct_size, header.abi_version,
                          header.prepare != nullptr, header.destroy != nullptr});
    selected = selected || (row.interface_id == selected_interface &&
                            row.interface_version == selected_version);
  }
  std::sort(interfaces.begin(), interfaces.end(), [](const auto& left, const auto& right) {
    return std::tie(left.id, left.version) < std::tie(right.id, right.version);
  });
  for (std::size_t index = 1; index < interfaces.size(); ++index)
    if (interfaces[index - 1].id == interfaces[index].id &&
        interfaces[index - 1].version == interfaces[index].version)
      throw std::invalid_argument(
          "prepared System boundary component interface authority is duplicated");
  if (!selected)
    throw std::invalid_argument(
        "prepared System boundary component lacks its selected interface authority");

  ExactContractBuilder contract;
  contract.text("pops.system.prepared-boundary-component-authority")
      .scalar(std::uint32_t{2})
      .text(owner->binary_identity())
      .scalar(api.struct_size)
      .scalar(api.protocol_abi)
      .text(require_text(api.abi_key, "abi_key"))
      .text(require_text(api.catalog_sha256, "catalog_sha256"))
      .text(require_text(api.component_id, "component_id"))
      .text(require_text(api.semantic_identity, "semantic_identity"))
      .text(require_text(api.manifest_identity, "manifest_identity"))
      .scalar(static_cast<std::uint32_t>(selected_interface))
      .scalar(selected_version)
      .sequence(interfaces, [](ExactContractBuilder& item, const InterfaceAuthority& interface) {
        item.scalar(interface.id)
            .scalar(interface.version)
            .scalar(interface.table_size)
            .scalar(interface.header_size)
            .scalar(interface.header_abi)
            .presence(interface.has_prepare)
            .presence(interface.has_destroy);
      });
  return std::move(contract).release();
}

template <int Dim>
std::size_t checked_cells(const Box<Dim>& box, const char* operation) {
  const std::int64_t cells = box.numPts();
  if (cells < 0 || static_cast<std::uint64_t>(cells) >
                       static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error(std::string(operation) + ": ranked box exceeds size_t");
  return static_cast<std::size_t>(cells);
}

template <int Dim>
Index<Dim> unflatten(const Box<Dim>& box, std::size_t linear) {
  Index<Dim> index{};
  const Extent<Dim> extent = box.extent();
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t axis_extent = static_cast<std::size_t>(extent[axis]);
    index[axis] = box.lo[axis] + static_cast<int>(linear % axis_extent);
    linear /= axis_extent;
  }
  return index;
}

template <int Dim>
std::size_t storage_offset(const Index<Dim>& index, const Box<Dim>& storage) {
  const Extent<Dim> extent = storage.extent();
  std::size_t offset = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    offset += static_cast<std::size_t>(index[axis] - storage.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(extent[axis]);
  }
  return offset;
}

struct CoupledSourceInputReference {
  int block = -1;
  int component = -1;
};

// Kept outside System because NVCC requires the lexical parent of the POPS_HD reduction lambda to
// be public.  The provider retains the same borrowed Impl pointer in a fixed host-only
// std::function, so no local lambda type becomes a template argument of the device parent.
template <int Dim>
struct CoupledSourceMaximumFrequency {
 public:
  std::vector<CoupledSourceInputReference> inputs;
  std::vector<Real> constants;
  CsProgram frequency_program{};
  int input_count = 0;
  int constant_count = 0;
  const ExecutionLane* lane = nullptr;
  std::function<const MultiFab<Dim>&(int)> state_for_block;

  Real operator()() const {
    if (input_count == 0)
      throw std::logic_error("state-dependent coupling frequency requires at least one input");
    const MultiFab<Dim>& reference = state_for_block(inputs.front().block);
    Real local_maximum = Real(0);
    for (std::size_t local = 0; local < reference.local_size(); ++local) {
      CoupledFreqKernel<Dim> kernel{};
      kernel.n_in = input_count;
      kernel.n_const = constant_count;
      kernel.prog = frequency_program;
      for (int input = 0; input < input_count; ++input) {
        const auto& ref = inputs[static_cast<std::size_t>(input)];
        const Fab<Dim>& fab = state_for_block(ref.block).fab(local);
        kernel.in[input] = fab.view();
        kernel.in_comp[input] = ref.component;
      }
      for (int constant = 0; constant < constant_count; ++constant)
        kernel.consts[constant] = constants[static_cast<std::size_t>(constant)];
      local_maximum = std::max(
          local_maximum,
          for_each_cell_reduce_max(reference.box(local), [=] POPS_HD(const Index<Dim>& index) {
            Real value = std::numeric_limits<Real>::lowest();
            kernel(index, value);
            return value;
          }));
    }
    return all_reduce_max(local_maximum, *lane);
  }
};

template <int Dim>
std::function<Real()> make_coupled_source_maximum_frequency(
    std::vector<CoupledSourceInputReference> inputs, std::vector<Real> constants,
    CsProgram frequency_program, int input_count, int constant_count, const ExecutionLane& lane,
    std::function<const MultiFab<Dim>&(int)> state_for_block) {
  return CoupledSourceMaximumFrequency<Dim>{
      std::move(inputs), std::move(constants), frequency_program, input_count, constant_count,
      &lane, std::move(state_for_block)};
}

template <int Dim>
mesh::BoxArrayValidationBudget exact_layout_budget(const mesh::BoxArray<Dim>& layout) {
  const std::size_t boxes = layout.size();
  if (boxes > 1 && boxes - 1 > std::numeric_limits<std::size_t>::max() / boxes)
    throw std::length_error("FFT field layout validation budget exceeds size_t");
  return {boxes, boxes < 2 ? 0 : boxes * (boxes - 1) / 2};
}

template <int Dim>
PhysicalBoundaryConditions<Dim> fft_periodic_boundary(const Geometry<Dim>& geometry) {
  std::array<bool, Dim> periodic{};
  std::array<PhysicalBoundaryFace, 2 * Dim> faces{};
  RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis) {
    periodic[axis] = true;
    spacing[axis] = geometry.spacing(axis);
  }
  return {BoundaryTopology<Dim>::axis_periodic(periodic), faces, spacing};
}

template <int Dim>
EllipticBuildRequest<Dim> fft_build_request(const Geometry<Dim>& geometry,
                                            const mesh::BoxArray<Dim>& layout,
                                            const mesh::Distribution<Dim>& distribution,
                                            Index<Dim> local_rank) {
  Extent<Dim> rhs_ghosts{};
  Extent<Dim> phi_ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    phi_ghosts[axis] = 1;
  return {geometry,
          layout,
          distribution,
          local_rank,
          fft_periodic_boundary(geometry),
          rhs_ghosts,
          phi_ghosts,
          exact_layout_budget(layout)};
}

/// Copy only valid cells. Ghost values are owned by prepared communication/boundary producers and
/// must be regenerated after storage width changes.
template <int Dim>
struct CopyValidComponentsKernel {
  FieldView<const Real, Dim> source{};
  FieldView<Real, Dim> destination{};
  int components = 0;

  POPS_HD void operator()(const Index<Dim>& index) const {
    for (int component = 0; component < components; ++component)
      destination(index, component) = source(index, component);
  }
};

template <int Dim>
void copy_valid_components(const MultiFab<Dim>& source, MultiFab<Dim>& destination, int components,
                           const char* operation) {
  if (source.layout() != destination.layout() ||
      source.distribution() != destination.distribution() ||
      source.local_rank() != destination.local_rank() ||
      source.local_global_indices() != destination.local_global_indices() ||
      source.local_size() != destination.local_size() || components < 0 ||
      components > source.ncomp() || components > destination.ncomp())
    throw std::invalid_argument(std::string(operation) +
                                ": fields do not share one exact ranked layout");

  for (std::size_t local = 0; local < source.local_size(); ++local) {
    const std::size_t global = destination.global_index(local);
    if (!source.contains_local(global))
      throw std::invalid_argument(std::string(operation) + ": source lacks destination patch");
    const Fab<Dim>& source_fab = source.fab(source.local_index_of(global));
    Fab<Dim>& destination_fab = destination.fab(local);
    if (source_fab.box() != destination_fab.box())
      throw std::invalid_argument(std::string(operation) + ": local valid boxes differ");
    const Box<Dim>& valid = source_fab.box();
    for_each_cell(valid, CopyValidComponentsKernel<Dim>{source_fab.view(), destination_fab.view(),
                                                        components});
  }
  Kokkos::fence();
}

enum class ExternalBoundaryDependencyOperation { ghost_region, flux_transform, field_closure };

template <int Dim>
HaloScheduleBudget external_boundary_halo_budget(const MultiFab<Dim>& field, const Box<Dim>& domain,
                                                 const BoundaryTopology<Dim>& topology) {
  const std::size_t patches = field.layout().size();
  if (patches != 0 && patches > std::numeric_limits<std::size_t>::max() / patches)
    throw std::overflow_error("prepared boundary halo patch budget overflows size_t");
  const std::size_t pairs = patches * patches;
  std::size_t images = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    std::size_t axis_images = 1;
    if (topology.is_periodic(Face<Dim>{axis, BoundarySide::lower}) && field.ghosts()[axis] > 0) {
      const std::int64_t extent = domain.length(axis);
      if (extent <= 0)
        throw std::invalid_argument("prepared boundary halo periodic domain is empty");
      const std::int64_t wraps = 1 + (field.ghosts()[axis] - 1) / extent;
      if (wraps < 0 ||
          static_cast<std::uint64_t>(wraps) > (std::numeric_limits<std::size_t>::max() - 1u) / 2u)
        throw std::overflow_error("prepared boundary halo periodic-image budget overflows size_t");
      axis_images += 2u * static_cast<std::size_t>(wraps);
    }
    if (axis_images != 0 && images > std::numeric_limits<std::size_t>::max() / axis_images)
      throw std::overflow_error("prepared boundary halo image budget overflows size_t");
    images *= axis_images;
  }
  if (images != 0 && pairs > std::numeric_limits<std::size_t>::max() / images)
    throw std::overflow_error("prepared boundary halo work budget overflows size_t");
  const std::size_t work = pairs * images;
  if (work > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(2 * Dim))
    throw std::overflow_error("prepared boundary halo job budget overflows size_t");
  const std::size_t jobs = work * static_cast<std::size_t>(2 * Dim);
  const std::int64_t signed_cells = domain.numPts();
  if (signed_cells <= 0 || field.ncomp() < 1)
    throw std::invalid_argument("prepared boundary halo field contract is empty");
  const std::size_t cells = static_cast<std::size_t>(signed_cells);
  if (jobs != 0 && cells > std::numeric_limits<std::size_t>::max() / jobs)
    throw std::overflow_error("prepared boundary halo element budget overflows size_t");
  const std::size_t cells_per_job = jobs * cells;
  if (static_cast<std::size_t>(field.ncomp()) > 0 &&
      cells_per_job >
          std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(field.ncomp()))
    throw std::overflow_error("prepared boundary halo component budget overflows size_t");
  const std::size_t elements = cells_per_job * static_cast<std::size_t>(field.ncomp());
  return {mesh::BoxArrayValidationBudget{patches, pairs},
          work,
          jobs,
          images,
          patches * 2,
          elements,
          elements,
          elements};
}

void validate_variable_set(const VariableSet& variables, VariableKind expected, int ncomp,
                           const char* label) {
  if (variables.kind != expected || variables.size != ncomp ||
      variables.names.size() != static_cast<std::size_t>(ncomp) ||
      variables.roles.size() != static_cast<std::size_t>(ncomp) ||
      (!variables.user_roles.empty() &&
       variables.user_roles.size() != static_cast<std::size_t>(ncomp)))
    throw std::invalid_argument(std::string("prepared System block ") + label +
                                " variable metadata differs from its component count");
  std::set<std::string> names;
  for (const std::string& name : variables.names)
    if (name.empty() || !names.insert(name).second)
      throw std::invalid_argument(std::string("prepared System block ") + label +
                                  " variable names must be unique and non-empty");
}

template <int Dim>
struct ExternalBoundaryDependencyStorage {
  struct GhostTransport {
    HaloSchedule<Dim> schedule;
    std::optional<ExecutionLane::ImmutableBorrow> lane_borrow;
    std::optional<HaloExchange<Dim>> exchange;
    std::shared_ptr<const PreparedHyperbolicBoundary<Dim>> physical_boundary;

    GhostTransport(const MultiFab<Dim>& image, const Geometry<Dim>& geometry,
                   const BoundaryTopology<Dim>& topology,
                   std::shared_ptr<const PreparedHyperbolicBoundary<Dim>> physical)
        : schedule(prepare_halo_schedule(
              image, geometry.domain(), topology, HaloLayoutCoverage::full_domain,
              external_boundary_halo_budget(image, geometry.domain(), topology))),
          physical_boundary(std::move(physical)) {}

    void prepare_collectively(const ExecutionLane& lane) {
      runtime::program::collective_boundary_provider_phase(
          lane, "prepared System boundary halo preflight failed collectively", [&] {
            if (exchange || lane_borrow)
              throw std::logic_error("prepared System boundary halo transport was prepared twice");
          });
      const bool distributed = all_reduce_max(schedule.has_remote_jobs() ? 1L : 0L, lane) != 0;
      if (distributed) {
        HaloExchangeContext context{};
        context.context_generation = 1;
        context.schedule_generation = 1;
        // Storage is embedded in the local candidate so every rank reaches HaloExchange's
        // collective constructor before any wrapper allocation can fail.
        exchange.emplace(schedule, lane, context);
      } else {
        runtime::program::collective_boundary_provider_phase(
            lane, "prepared System local halo lane borrow failed collectively",
            [&] { lane_borrow.emplace(lane.borrow_immutably()); });
      }
    }

    void materialize(MultiFab<Dim>& image, const ExecutionLane& lane) {
      if (exchange)
        fill_boundary(image, *exchange, lane);
      else
        fill_boundary(image, schedule);
      if (physical_boundary)
        physical_boundary->fill_physical(image, schedule.domain());
    }
  };

  struct DetachedDependency {
    const MultiFab<Dim>* source = nullptr;
    std::shared_ptr<runtime::system::ExactNamedField<Dim>> field_owner;
    MultiFab<Dim> image;
    std::shared_ptr<GhostTransport> ghost_transport;
    typename SystemBlockClosures<Dim>::PreparedPointStateTransport source_prepare;
    std::shared_ptr<runtime::program::PreparedScalarBoundarySession<Dim>> source_transport;
    bool preserve_full_storage = false;
  };

  std::vector<const MultiFab<Dim>*> states;
  std::vector<FieldDistribution> state_distributions;
  std::vector<std::string> state_identities;
  std::vector<const MultiFab<Dim>*> fields;
  std::vector<FieldDistribution> field_distributions;
  std::vector<std::string> field_identities;
  std::vector<DetachedDependency> detached;

  void prepare_collectively(const Geometry<Dim>& geometry, const BoundaryTopology<Dim>& topology,
                            const ExecutionLane& lane) {
    for (auto& dependency : detached) {
      if (!dependency.ghost_transport)
        continue;
      dependency.ghost_transport->prepare_collectively(lane);
      if (dependency.source_prepare)
        dependency.source_transport =
            runtime::program::PreparedScalarBoundarySession<Dim>::prepare_block(
                geometry, topology, dependency.image, lane, 1);
    }
  }

  void refresh(const runtime::multiblock::BoundaryEvaluationPoint& point,
               const ExecutionLane& lane) {
    const auto source_for = [](DetachedDependency& dependency) -> const MultiFab<Dim>* {
      return dependency.field_owner ? &dependency.field_owner->dependency_potential()
                                    : dependency.source;
    };
    std::exception_ptr preflight_error;
    try {
      for (auto& dependency : detached) {
        const MultiFab<Dim>* source = source_for(dependency);
        if (dependency.source_prepare &&
            (!dependency.source_transport || !dependency.ghost_transport ||
             !dependency.ghost_transport->physical_boundary))
          throw std::logic_error("prepared System ghost dependency lost its source transport");
        if (source == nullptr || source->layout() != dependency.image.layout() ||
            source->distribution() != dependency.image.distribution() ||
            source->local_rank() != dependency.image.local_rank() ||
            source->ncomp() != dependency.image.ncomp() ||
            source->ghosts() != dependency.image.ghosts() ||
            source->local_size() != dependency.image.local_size())
          throw std::logic_error("prepared System boundary dependency source is absent");
        for (std::size_t local = 0; local < dependency.image.local_size(); ++local) {
          const std::size_t global = dependency.image.global_index(local);
          if (!source->contains_local(global) ||
              source->fab(source->local_index_of(global)).box() !=
                  dependency.image.fab(local).box() ||
              source->fab(source->local_index_of(global)).grown_box() !=
                  dependency.image.fab(local).grown_box())
            throw std::logic_error("prepared System boundary dependency patch route changed");
        }
      }
    } catch (...) {
      preflight_error = std::current_exception();
    }
    if (all_reduce_max(preflight_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && preflight_error)
        std::rethrow_exception(preflight_error);
      throw std::runtime_error("prepared System boundary dependency preflight failed collectively");
    }
    std::exception_ptr copy_error;
    try {
      for (auto& dependency : detached) {
        const MultiFab<Dim>* source = source_for(dependency);
        if (dependency.preserve_full_storage) {
          for (std::size_t local = 0; local < dependency.image.local_size(); ++local) {
            const std::size_t global = dependency.image.global_index(local);
            Kokkos::deep_copy(dependency.image.fab(local).storage(),
                              source->fab(source->local_index_of(global)).storage());
          }
        } else {
          dependency.image.set_val(Real(0));
          copy_valid_components(*source, dependency.image, dependency.image.ncomp(),
                                "prepared System boundary dependency refresh");
        }
      }
      Kokkos::fence();
    } catch (...) {
      copy_error = std::current_exception();
    }
    if (all_reduce_max(copy_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && copy_error)
        std::rethrow_exception(copy_error);
      throw std::runtime_error("prepared System boundary dependency copy failed collectively");
    }
    for (auto& dependency : detached)
      runtime::program::collective_boundary_provider_phase(
          lane, "prepared System boundary dependency transport failed collectively", [&] {
            if (dependency.source_prepare) {
              // The source generated closure performs same-level transport, model-qualified
              // physical fill, and the source's generic external hook on this detached image.
              dependency.source_prepare(point, dependency.image,
                                        *dependency.ghost_transport->physical_boundary, lane,
                                        *dependency.source_transport);
            } else if (dependency.ghost_transport) {
              dependency.ghost_transport->materialize(dependency.image, lane);
            }
          });
  }

  FieldBoundaryExecutionContext<Dim> view(
      const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    FieldBoundaryExecutionContext<Dim> context;
    context.point.time = static_cast<Real>(point.physical_time);
    context.point.dt = static_cast<Real>(point.dt);
    context.point.stage_slot = point.stage;
    context.point.level = point.level;
    context.point.step = point.tick;
    context.point.substep = point.substep;
    context.point.stage_fraction_numerator = point.stage_fraction.numerator;
    context.point.stage_fraction_denominator = point.stage_fraction.denominator;
    context.states = states.empty() ? nullptr : states.data();
    context.state_distributions =
        state_distributions.empty() ? nullptr : state_distributions.data();
    context.state_identities = state_identities.empty() ? nullptr : state_identities.data();
    context.state_count = static_cast<int>(states.size());
    context.fields = fields.empty() ? nullptr : fields.data();
    context.field_distributions =
        field_distributions.empty() ? nullptr : field_distributions.data();
    context.field_identities = field_identities.empty() ? nullptr : field_identities.data();
    context.field_count = static_cast<int>(fields.size());
    context.clock_identity = &point.clock;
    return context;
  }
};

template <int Dim, class BlockStore, class BoundaryRegistry, class NamedFields>
ExternalBoundaryDependencyStorage<Dim> prepare_external_boundary_dependencies(
    const std::array<bool, Dim>& periodicity, const Geometry<Dim>& geometry,
    const BlockStore& blocks, const BoundaryRegistry& boundary_registry,
    const NamedFields& named_fields, const PreparedBoundaryComponentSpec& spec,
    const MultiFab<Dim>& owning, const ExecutionLane& lane,
    ExternalBoundaryDependencyOperation operation) {
  ExternalBoundaryDependencyStorage<Dim> storage;
  storage.detached.reserve(spec.states.size() + spec.fields.size());
  auto authenticate = [&](const MultiFab<Dim>& dependency, std::string_view identity) {
    if (dependency.layout() != owning.layout() || dependency.local_rank() != owning.local_rank() ||
        dependency.rank_space() != owning.rank_space() ||
        lane.size() != static_cast<int>(dependency.rank_space().size()) ||
        lane.rank() !=
            static_cast<int>(dependency.rank_space().linear_rank(dependency.local_rank())))
      throw std::invalid_argument("prepared boundary dependency " + std::string(identity) +
                                  " differs from its exact block layout/lane");
    for (std::size_t local = 0; local < owning.local_size(); ++local) {
      const std::size_t global = owning.global_index(local);
      Box<Dim> required = owning.fab(local).box();
      if (operation == ExternalBoundaryDependencyOperation::ghost_region)
        for (std::size_t ordinal = 0; ordinal < spec.region.axes.size(); ++ordinal) {
          const int axis = spec.region.axes[ordinal];
          if (spec.region.sides[ordinal] < 0)
            required.lo[axis] -= owning.ghosts()[axis];
          else
            required.hi[axis] += owning.ghosts()[axis];
        }
      if (!dependency.contains_local(global) ||
          !dependency.fab(dependency.local_index_of(global)).grown_box().contains(required))
        throw std::invalid_argument("prepared boundary dependency " + std::string(identity) +
                                    " lacks exact patch colocation/halo");
    }
  };
  const BoundaryTopology<Dim> topology = BoundaryTopology<Dim>::axis_periodic(periodicity);
  const bool needs_physical =
      operation == ExternalBoundaryDependencyOperation::ghost_region &&
      std::any_of(spec.region.axes.begin(), spec.region.axes.end(), [&](int axis) {
        return !topology.is_periodic(Face<Dim>{axis, BoundarySide::lower});
      });
  auto make_image =
      [&](const MultiFab<Dim>& dependency,
          std::shared_ptr<const PreparedHyperbolicBoundary<Dim>> physical,
          typename SystemBlockClosures<Dim>::PreparedPointStateTransport source_prepare,
          bool preserve_full_storage = false) {
        typename ExternalBoundaryDependencyStorage<Dim>::DetachedDependency detached{
            nullptr,
            {},
            MultiFab<Dim>(dependency.layout(), dependency.distribution(), dependency.local_rank(),
                          dependency.ncomp(), dependency.ghosts()),
            {},
            {},
            {},
            preserve_full_storage};
        if (operation == ExternalBoundaryDependencyOperation::ghost_region) {
          if (needs_physical && !physical && !source_prepare && !preserve_full_storage)
            throw std::runtime_error(
                "prepared System cross-route GhostBoundary lacks sealed source physical authority");
          if (physical && !source_prepare)
            for (int axis = 0; axis < Dim; ++axis)
              for (int side : {-1, 1}) {
                const HyperbolicBoundaryLaw law = physical->face(axis, side).law;
                if (law == HyperbolicBoundaryLaw::External ||
                    law == HyperbolicBoundaryLaw::CharacteristicNoInflow ||
                    physical->has_analytic_state())
                  throw std::runtime_error(
                      "prepared System cross-route GhostBoundary requires a source model transport "
                      "hook for external, characteristic, or analytic physical faces");
              }
          detached.ghost_transport =
              std::make_shared<typename ExternalBoundaryDependencyStorage<Dim>::GhostTransport>(
                  detached.image, geometry, topology, std::move(physical));
          if (source_prepare)
            detached.source_prepare = std::move(source_prepare);
        }
        return detached;
      };
  storage.states.reserve(spec.states.size());
  storage.state_distributions.reserve(spec.states.size());
  storage.state_identities.reserve(spec.states.size());
  for (const std::string& identity : spec.states) {
    const MultiFab<Dim>* dependency = nullptr;
    std::shared_ptr<const PreparedHyperbolicBoundary<Dim>> physical;
    typename SystemBlockClosures<Dim>::PreparedPointStateTransport source_prepare;
    if (identity == spec.state_identity)
      dependency = &owning;
    else
      for (const auto& block : blocks.blocks)
        if (block.state_identity == identity) {
          dependency = &block.U;
          physical = block.boundary;
          source_prepare = block.prepare_generated_state_with_transport_prepared;
          break;
        }
    if (dependency == nullptr)
      throw std::runtime_error("prepared boundary state dependency has no sealed block route");
    if (!physical)
      source_prepare = {};
    authenticate(*dependency, identity);
    if (identity == spec.state_identity)
      storage.states.push_back(nullptr);
    else {
      storage.detached.push_back(
          make_image(*dependency, std::move(physical), std::move(source_prepare)));
      storage.detached.back().source = dependency;
      storage.states.push_back(&storage.detached.back().image);
    }
    storage.state_distributions.push_back(dependency->distribution().replicated()
                                              ? FieldDistribution::Replicated
                                              : FieldDistribution::Distributed);
    storage.state_identities.push_back(identity);
  }
  storage.fields.reserve(spec.fields.size());
  storage.field_distributions.reserve(spec.fields.size());
  storage.field_identities.reserve(spec.fields.size());
  for (const std::string& identity : spec.fields) {
    const std::string& slot = boundary_registry.field_storage_route(identity);
    const auto found = named_fields.find(slot);
    if (found == named_fields.end() || !found->second)
      throw std::runtime_error("prepared boundary field dependency is not materialized");
    const MultiFab<Dim>& dependency = found->second->accepted_potential();
    authenticate(dependency, identity);
    storage.detached.push_back(make_image(
        dependency, {}, {}, operation == ExternalBoundaryDependencyOperation::ghost_region));
    storage.detached.back().field_owner = found->second;
    storage.fields.push_back(&storage.detached.back().image);
    storage.field_distributions.push_back(dependency.distribution().replicated()
                                              ? FieldDistribution::Replicated
                                              : FieldDistribution::Distributed);
    storage.field_identities.push_back(identity);
  }
  return storage;
}

template <int Dim, class Component>
struct PreparedSystemBoundaryComponentCandidate {
  using local_session_type = typename Component::template LocalSessionCandidate<Dim>;
  using session_holder_type = std::optional<typename Component::Session>;

  std::shared_ptr<ExternalBoundaryDependencyStorage<Dim>> dependencies;
  std::optional<local_session_type> local_session;
  /// The shared control block and Session storage are allocated in the local prepass.  Provider
  /// callbacks only emplace into this stable holder, so a rank cannot lose a post-callback
  /// allocation race before the exact-lane result is converged.
  std::shared_ptr<session_holder_type> session;
};

/// Prepare one external boundary provider in three non-mixing phases: purely local dependency and
/// scratch allocation, dependency transport collective construction, then the component-owned
/// prepare callback.  Every phase converges before the canonical package installer may advance.
template <int Dim, class BlockStore, class BoundaryRegistry, class NamedFields, class Component>
PreparedSystemBoundaryComponentCandidate<Dim, Component>
prepare_system_boundary_component_candidate(
    const std::array<bool, Dim>& periodicity, const BlockStore& blocks,
    const BoundaryRegistry& boundary_registry, const NamedFields& named_fields,
    const std::shared_ptr<Component>& component, const MultiFab<Dim>& owning,
    const Geometry<Dim>& geometry, const BoundaryTopology<Dim>& topology, const ExecutionLane& lane,
    ExternalBoundaryDependencyOperation operation) {
  PreparedSystemBoundaryComponentCandidate<Dim, Component> candidate;
  runtime::program::collective_boundary_provider_phase(
      lane, "prepared System boundary local candidate preparation failed collectively", [&] {
        candidate.dependencies = std::make_shared<ExternalBoundaryDependencyStorage<Dim>>(
            prepare_external_boundary_dependencies<Dim>(
                periodicity, geometry, blocks, boundary_registry, named_fields, component->spec(),
                owning, lane, operation));
        candidate.local_session.emplace(component->template make_local_session_candidate<Dim>(
            lane, owning, geometry, candidate.dependencies->view({})));
        candidate.session = std::make_shared<typename PreparedSystemBoundaryComponentCandidate<
            Dim, Component>::session_holder_type>();
      });
  candidate.dependencies->prepare_collectively(geometry, topology, lane);
  runtime::program::collective_boundary_provider_phase(
      lane, "prepared System boundary provider preparation failed collectively", [&] {
        if (!candidate.session || candidate.session->has_value() || !candidate.local_session)
          throw std::logic_error("prepared System boundary session holder is invalid");
        candidate.session->emplace(
            component->template finish_session<Dim>(std::move(*candidate.local_session)));
        candidate.local_session.reset();
      });
  return candidate;
}

template <int Dim>
void validate_prepared_block(const PreparedSystemBlock<Dim>& block) {
  if (block.name.empty() || block.provider_identity.empty())
    throw std::invalid_argument(
        "prepared System block requires non-empty block and provider identities");
  if (block.ncomp < 1 || block.provider_components < 0)
    throw std::invalid_argument(
        "prepared System block requires a positive state and non-negative provider value count");
  if (!std::isfinite(block.gamma) || !(block.gamma > 0.0) || block.substeps < 1 || block.stride < 1)
    throw std::invalid_argument("prepared System block gamma, substeps, and stride are invalid");
  for (int axis = 0; axis < Dim; ++axis)
    if (block.ghosts[axis] < 1)
      throw std::invalid_argument(
          "prepared System block must declare a positive ghost extent on every native axis");
  validate_variable_set(block.conservative_variables, VariableKind::Conservative, block.ncomp,
                        "conservative");
  validate_variable_set(block.primitive_variables, VariableKind::Primitive, block.ncomp,
                        "primitive");

  const auto& closures = block.closures;
  if (!closures.rhs_into || !closures.rhs_flux_only || !closures.source_only ||
      !closures.rhs_at_point || !closures.rhs_flux_only_at_point || !closures.rhs_core_at_point ||
      !closures.rhs_flux_only_core_at_point || !closures.rhs_core_at_point_prepared ||
      !closures.rhs_flux_only_core_at_point_prepared ||
      !closures.transport_rhs_at_point_prepared ||
      !closures.transport_flux_at_point_prepared ||
      !closures.prepare_generated_state_at_point ||
      !closures.prepare_generated_state_at_point_prepared ||
      !closures.prepare_generated_state_with_transport_prepared ||
      !closures.transport_prepare_generated_state_at_point_prepared ||
      !closures.external_ghost_boundary || !block.maximum_speed || !block.poisson_rhs ||
      !block.primitive_to_conservative || !block.conservative_to_primitive ||
      !block.batch_conservative_to_primitive)
    throw std::invalid_argument(
        "prepared System block does not implement the complete exact-ranked execution contract");
}

template <int Dim, class Implementation>
struct PreparedBlockInstallation {
  typename Implementation::Species block;
  std::string name;
  std::shared_ptr<const PreparedHyperbolicBoundary<Dim>> converted_boundary;
};

template <int Dim, class Implementation>
PreparedBlockInstallation<Dim, Implementation> prepare_block_installation(
    const Implementation& implementation,
    const runtime::system::ExactAuxiliaryRegistry<Dim>& auxiliary_registry,
    const typename Implementation::block_store_type& blocks,
    const typename Implementation::boundary_registry_type& boundary_registry,
    std::string_view provider_consumer_qid, PreparedSystemBlock<Dim> prepared) {
  validate_prepared_block(prepared);
  if (std::any_of(blocks.blocks.begin(), blocks.blocks.end(),
                  [&](const typename Implementation::Species& block) {
                    return block.name == prepared.name;
                  }))
    throw std::invalid_argument("System prepared block name is already installed");

  const auto route = boundary_registry.state_routes().find(prepared.name);
  if (route == boundary_registry.state_routes().end())
    throw std::runtime_error("System prepared block lacks its exact pre-installed state identity");
  const std::string state_identity = route->second;

  const auto* installed_boundary = boundary_registry.find_boundary(prepared.name);
  std::shared_ptr<const PreparedHyperbolicBoundary<Dim>> boundary;
  if (installed_boundary != nullptr) {
    if (installed_boundary->state_identity != state_identity ||
        installed_boundary->authority->ncomp() != prepared.ncomp ||
        installed_boundary->authority->periodic_axes() != implementation.periodicity)
      throw std::invalid_argument(
          "System prepared block boundary differs from its exact state/domain contract");
    for (int axis = 0; axis < Dim; ++axis)
      if (prepared.ghosts[axis] < installed_boundary->required_depth)
        throw std::invalid_argument(
            "System prepared block ghosts are narrower than its boundary requirement");
    boundary = std::make_shared<PreparedHyperbolicBoundary<Dim>>(
        installed_boundary->authority->with_converted_fixed_states(
            prepared.primitive_to_conservative));
  } else if (std::any_of(implementation.periodicity.begin(), implementation.periodicity.end(),
                         [](bool periodic) { return !periodic; })) {
    throw std::invalid_argument(
        "prepared System block with a physical domain face requires one "
        "PreparedHyperbolicBoundary; a boundary-less block is periodic on every native axis");
  }

  if (prepared.provider_components != 0 && auxiliary_registry.sealed()) {
    const auto& plan = auxiliary_registry.consumer_plan(provider_consumer_qid);
    if (plan.value_count() != static_cast<std::size_t>(prepared.provider_components))
      throw std::invalid_argument(
          "prepared System block provider count differs from its resolved consumer plan");
  } else if (prepared.provider_components != 0) {
    throw std::logic_error(
        "prepared System block with provider values requires the sealed global provider registry");
  }

  typename Implementation::Species candidate;
  candidate.name = prepared.name;
  candidate.U = MultiFab<Dim>(implementation.ba, implementation.dm, implementation.local_rank,
                              prepared.ncomp, prepared.ghosts);
  candidate.ncomp = prepared.ncomp;
  candidate.substeps = prepared.substeps;
  candidate.evolve = prepared.evolve;
  candidate.stride = prepared.stride;
  candidate.newton = prepared.newton;
  candidate.newton_diagnostics = prepared.newton_diagnostics;
  candidate.gamma = prepared.gamma;
  candidate.rhs_into = std::move(prepared.closures.rhs_into);
  candidate.max_speed = std::move(prepared.maximum_speed);
  candidate.add_poisson_rhs = std::move(prepared.poisson_rhs);
  candidate.cons_vars = std::move(prepared.conservative_variables);
  candidate.prim_vars = std::move(prepared.primitive_variables);
  candidate.prim_to_cons = std::move(prepared.primitive_to_conservative);
  candidate.cons_to_prim = std::move(prepared.conservative_to_primitive);
  candidate.batch_cons_to_prim = std::move(prepared.batch_conservative_to_primitive);
  candidate.source_frequency = std::move(prepared.source_frequency);
  candidate.parabolic_frequency = prepared.parabolic_frequency;
  candidate.stability_dt = std::move(prepared.stability_dt);
  candidate.project = std::move(prepared.closures.project);
  candidate.project_masked = std::move(prepared.closures.project_masked);
  candidate.rhs_flux_only = std::move(prepared.closures.rhs_flux_only);
  candidate.source_only = std::move(prepared.closures.source_only);
  candidate.source_only_masked = std::move(prepared.closures.source_only_masked);
  candidate.solve_implicit_source = std::move(prepared.closures.solve_implicit_source);
  candidate.staircase_residuals = std::move(prepared.closures.staircase);
  candidate.cutcell_residuals = std::move(prepared.closures.cut_cell);
  candidate.rhs_at_point = std::move(prepared.closures.rhs_at_point);
  candidate.rhs_flux_only_at_point = std::move(prepared.closures.rhs_flux_only_at_point);
  candidate.rhs_without_prepared_interfaces =
      std::move(prepared.closures.rhs_without_prepared_interfaces);
  candidate.rhs_flux_only_without_prepared_interfaces =
      std::move(prepared.closures.rhs_flux_only_without_prepared_interfaces);
  candidate.rhs_core_at_point = std::move(prepared.closures.rhs_core_at_point);
  candidate.rhs_flux_only_core_at_point = std::move(prepared.closures.rhs_flux_only_core_at_point);
  candidate.rhs_core_at_point_prepared = std::move(prepared.closures.rhs_core_at_point_prepared);
  candidate.rhs_flux_only_core_at_point_prepared =
      std::move(prepared.closures.rhs_flux_only_core_at_point_prepared);
  candidate.boundary_full_at_point_prepared =
      std::move(prepared.closures.boundary_full_at_point_prepared);
  candidate.boundary_core_at_point_prepared =
      std::move(prepared.closures.boundary_core_at_point_prepared);
  candidate.transport_rhs_at_point_prepared =
      std::move(prepared.closures.transport_rhs_at_point_prepared);
  candidate.boundary_flux_full_at_point_prepared =
      std::move(prepared.closures.boundary_flux_full_at_point_prepared);
  candidate.boundary_flux_core_at_point_prepared =
      std::move(prepared.closures.boundary_flux_core_at_point_prepared);
  candidate.transport_flux_at_point_prepared =
      std::move(prepared.closures.transport_flux_at_point_prepared);
  candidate.boundary_residual_at_point_prepared =
      std::move(prepared.closures.boundary_residual_at_point_prepared);
  candidate.boundary_jvp_at_point_prepared =
      std::move(prepared.closures.boundary_jvp_at_point_prepared);
  candidate.external_boundary_flux = std::move(prepared.closures.external_boundary_flux);
  candidate.external_field_boundary_residual =
      std::move(prepared.closures.external_field_boundary_residual);
  candidate.external_field_boundary_jvp = std::move(prepared.closures.external_field_boundary_jvp);
  candidate.prepare_generated_state_at_point =
      std::move(prepared.closures.prepare_generated_state_at_point);
  candidate.prepare_generated_state_at_point_prepared =
      std::move(prepared.closures.prepare_generated_state_at_point_prepared);
  candidate.prepare_generated_state_with_transport_prepared =
      std::move(prepared.closures.prepare_generated_state_with_transport_prepared);
  candidate.transport_prepare_generated_state_at_point_prepared =
      std::move(prepared.closures.transport_prepare_generated_state_at_point_prepared);
  candidate.external_ghost_boundary = std::move(prepared.closures.external_ghost_boundary);
  candidate.boundary = boundary;
  candidate.state_identity = state_identity;
  if (candidate.boundary &&
      (!candidate.boundary_full_at_point_prepared || !candidate.boundary_core_at_point_prepared ||
       !candidate.boundary_flux_full_at_point_prepared ||
       !candidate.boundary_flux_core_at_point_prepared ||
       !candidate.boundary_residual_at_point_prepared ||
       !candidate.boundary_jvp_at_point_prepared || !candidate.external_boundary_flux ||
       !candidate.external_field_boundary_residual || !candidate.external_field_boundary_jvp ||
       !candidate.prepare_generated_state_with_transport_prepared ||
       !candidate.external_ghost_boundary))
    throw std::invalid_argument(
        "prepared System boundary lacks its complete full/core/residual/JVP authority");

  return {std::move(candidate), prepared.name, std::move(boundary)};
}

}  // namespace

template <int Dim>
void System<Dim>::add_block(const std::string&, const ModelSpec&, const std::string&,
                            const std::string&, const std::string&, const std::string&, int, bool,
                            int, const std::vector<std::string>&, const std::vector<std::string>&,
                            const NewtonOptions&, bool, double, bool, double) {
  throw std::logic_error(
      "System::add_block(ModelSpec) was removed from the native core: resolve and compile one "
      "dimension-qualified provider, then install its PreparedSystemBlock<Dim>");
}

template <int Dim>
void System<Dim>::install_block_state_route(const std::string& name,
                                            const std::string& state_identity) {
  require_assembling(p_->lifecycle_, "install_block_state_route");
  if (std::any_of(p_->sp.begin(), p_->sp.end(),
                  [&](const typename Impl::Species& block) { return block.name == name; }))
    throw std::logic_error("System block state route must be installed before its prepared block");
  p_->boundary_registry_.install_state_route(name, state_identity);
}

template <int Dim>
void System<Dim>::install_field_storage_route(const std::string& field_identity,
                                              const std::string& provider_slot) {
  require_assembling(p_->lifecycle_, "install_field_storage_route");
  p_->boundary_registry_.install_field_storage_route(field_identity, provider_slot);
}

template <int Dim>
void System<Dim>::install_prepared_boundary_execution_lane(std::shared_ptr<ExecutionLane> lane) {
  if (!lane)
    throw std::invalid_argument("prepared System boundary lane is null");
  std::exception_ptr local_error;
  try {
    require_assembling(p_->lifecycle_, "install_prepared_boundary_execution_lane");
    if (lane->identity().empty() || lane->size() != static_cast<int>(p_->dm.rank_space().size()) ||
        lane->rank() != static_cast<int>(p_->dm.rank_space().linear_rank(p_->local_rank)))
      throw std::invalid_argument(
          "prepared System boundary lane differs from its exact runtime rank space");
    if (prepared_boundary_execution_lane_)
      throw std::logic_error("prepared System boundary lane is already installed");
  } catch (...) {
    local_error = std::current_exception();
  }
  // ``lane`` owns a duplicated communicator.  Converge every local installation check before
  // moving it into System so all temporary holders either survive or release collectively.
  if (all_reduce_max(local_error ? 1L : 0L, *lane) != 0) {
    if (lane->size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("prepared System boundary lane installation failed collectively");
  }
  prepared_boundary_execution_lane_ = std::move(lane);
}

template <int Dim>
void System<Dim>::stage_prepared_ghost_boundary_component(
    const std::string& block, std::shared_ptr<PreparedGhostBoundaryComponent> component) {
  require_assembling(p_->lifecycle_, "stage_prepared_ghost_boundary_component");
  if (block.empty() || !component)
    throw std::invalid_argument(
        "prepared System GhostBoundary staging requires a block and typed component");
  const auto& spec = component->spec();
  if (spec.region.dimension != Dim ||
      spec.state_identity != p_->boundary_registry_.state_route(block))
    throw std::invalid_argument(
        "prepared System GhostBoundary differs from its exact block/rank route");
  const std::string package_identity =
      prepared_system_boundary_package_identity("ghost", block, spec);
  const std::string component_contract = prepared_system_boundary_component_contract(spec);
  const std::string component_authority_contract =
      prepared_system_boundary_component_authority_contract(component->package_owner_identity(),
                                                            POPS_NATIVE_INTERFACE_GHOST_BOUNDARY_V1,
                                                            spec.interface_version);
  std::vector<typename Impl::PreparedBoundaryHookContract> expected_hooks;
  expected_hooks.push_back(
      {package_identity, block, "ghost", component_contract, component_authority_contract});
  stage_native_package_(
      package_identity, {},
      [this, block, component, package_identity, component_contract, component_authority_contract] {
        const ExecutionLane& lane = prepared_boundary_execution_lane();
        if (p_->native_package_finalize_candidate_ == nullptr)
          throw std::logic_error(
              "prepared System GhostBoundary installer requires a detached candidate");
        auto& snapshot = *p_->native_package_finalize_candidate_;
        typename Impl::Species* selected = nullptr;
        typename Impl::NativePackageFinalizeSnapshot::BoundaryHookImage* hook = nullptr;
        std::optional<Geometry<Dim>> geometry;
        std::optional<BoundaryTopology<Dim>> topology;
        runtime::program::collective_boundary_provider_phase(
            lane, "prepared System GhostBoundary local installer preflight failed collectively",
            [&] {
              selected = &snapshot.blocks.find(block);
              hook = &snapshot.boundary_hook(block);
              if (!selected->boundary ||
                  selected->state_identity != component->spec().state_identity)
                throw std::invalid_argument(
                    "prepared System GhostBoundary block was not materialized with its exact "
                    "boundary");
              if (!selected->boundary_full_at_point_prepared ||
                  !selected->boundary_core_at_point_prepared ||
                  !selected->boundary_flux_full_at_point_prepared ||
                  !selected->boundary_flux_core_at_point_prepared)
                throw std::invalid_argument(
                    "prepared System GhostBoundary requires complete compiled full and core "
                    "closures");
              if (!hook->ghost_target)
                throw std::invalid_argument(
                    "prepared System GhostBoundary requires its generated hook target");
              geometry.emplace(p_->geom);
              topology.emplace(BoundaryTopology<Dim>::axis_periodic(p_->periodicity));
            });
        // The RuntimeInstance lane was materialized from its authenticated communicator before
        // plan publication. Each invocation borrows that exact lane through ProgramContext.
        auto prepared = prepare_system_boundary_component_candidate<Dim>(
            p_->periodicity, snapshot.blocks, snapshot.boundary_registry, snapshot.named_fields,
            component, selected->U, *geometry, *topology, lane,
            ExternalBoundaryDependencyOperation::ghost_region);
        auto dependencies = std::move(prepared.dependencies);
        auto session = std::move(prepared.session);
        const auto previous = *hook->ghost_target;
        const Geometry<Dim> prepared_geometry = *geometry;
        typename SystemBlockClosures<Dim>::ExternalGhostBoundary candidate =
            [previous, component, session, dependencies, geometry = prepared_geometry](
                const auto& point, MultiFab<Dim>& state, const auto& source_geometry,
                const ExecutionLane& lane) {
              if (previous)
                runtime::program::collective_boundary_provider_phase(
                    lane, "prepared System GhostBoundary predecessor failed collectively",
                    [&] { previous(point, state, source_geometry, lane); });
              std::unique_lock invocation_lock(session->value().invocation_mutex(),
                                               std::defer_lock);
              runtime::program::collective_boundary_provider_phase(
                  lane, "prepared System GhostBoundary session admission failed collectively", [&] {
                    if (!invocation_lock.try_lock())
                      throw std::logic_error(
                          "prepared System GhostBoundary dependency cycle/reentrancy detected");
                  });
              runtime::program::collective_boundary_provider_phase(
                  lane, "prepared System GhostBoundary dependency refresh failed collectively",
                  [&] { dependencies->refresh(point, lane); });
              FieldBoundaryExecutionContext<Dim> context;
              runtime::program::collective_boundary_provider_phase(
                  lane, "prepared System GhostBoundary context validation failed collectively",
                  [&] { context = dependencies->view(point); });
              runtime::program::collective_boundary_provider_phase(
                  lane, "prepared System GhostBoundary callback failed collectively", [&] {
                    component->template apply_ghost_region<Dim>(session->value(), point, state,
                                                                geometry, lane, context);
                  });
            };
        hook->ghost.swap(candidate);
        snapshot.prepared_boundary_hook_contracts.push_back(
            {package_identity, block, "ghost", component_contract, component_authority_contract});
      },
      component, nullptr, NativePackageKind::prepared_boundary);
  p_->pending_native_packages_.back().expected_boundary_hooks = std::move(expected_hooks);
}

template <int Dim>
void System<Dim>::stage_prepared_boundary_flux_component(
    const std::string& block, std::shared_ptr<PreparedBoundaryFluxComponent> component) {
  require_assembling(p_->lifecycle_, "stage_prepared_boundary_flux_component");
  if (block.empty() || !component || component->spec().region.dimension != Dim ||
      component->spec().state_identity != p_->boundary_registry_.state_route(block))
    throw std::invalid_argument("prepared System BoundaryFlux differs from its exact block route");
  const std::string package_identity =
      prepared_system_boundary_package_identity("flux", block, component->spec());
  const std::string component_contract =
      prepared_system_boundary_component_contract(component->spec());
  const std::string component_authority_contract =
      prepared_system_boundary_component_authority_contract(component->package_owner_identity(),
                                                            POPS_NATIVE_INTERFACE_BOUNDARY_FLUX_V1,
                                                            component->spec().interface_version);
  std::vector<typename Impl::PreparedBoundaryHookContract> expected_hooks;
  expected_hooks.push_back(
      {package_identity, block, "flux", component_contract, component_authority_contract});
  stage_native_package_(
      package_identity, {},
      [this, block, component, package_identity, component_contract, component_authority_contract] {
        const ExecutionLane& lane = prepared_boundary_execution_lane();
        if (p_->native_package_finalize_candidate_ == nullptr)
          throw std::logic_error(
              "prepared System BoundaryFlux installer requires a detached candidate");
        auto& snapshot = *p_->native_package_finalize_candidate_;
        typename Impl::Species* selected = nullptr;
        typename Impl::NativePackageFinalizeSnapshot::BoundaryHookImage* hook = nullptr;
        std::optional<Geometry<Dim>> geometry;
        std::optional<BoundaryTopology<Dim>> topology;
        runtime::program::collective_boundary_provider_phase(
            lane, "prepared System BoundaryFlux local installer preflight failed collectively",
            [&] {
              selected = &snapshot.blocks.find(block);
              hook = &snapshot.boundary_hook(block);
              if (!selected->boundary || !hook->flux_target)
                throw std::invalid_argument(
                    "prepared System BoundaryFlux requires its generated post-Riemann hook");
              geometry.emplace(p_->geom);
              topology.emplace(BoundaryTopology<Dim>::axis_periodic(p_->periodicity));
            });
        auto prepared = prepare_system_boundary_component_candidate<Dim>(
            p_->periodicity, snapshot.blocks, snapshot.boundary_registry, snapshot.named_fields,
            component, selected->U, *geometry, *topology, lane,
            ExternalBoundaryDependencyOperation::flux_transform);
        auto dependencies = std::move(prepared.dependencies);
        auto session = std::move(prepared.session);
        const auto previous = *hook->flux_target;
        typename SystemBlockClosures<Dim>::BoundaryFluxTransform candidate =
            [previous, component, session, dependencies](const auto& point, const auto& state,
                                                         auto& faces, const auto& geometry,
                                                         const auto& lane) {
              if (previous)
                runtime::program::collective_boundary_provider_phase(
                    lane, "prepared System BoundaryFlux predecessor failed collectively",
                    [&] { previous(point, state, faces, geometry, lane); });
              std::unique_lock invocation_lock(session->value().invocation_mutex(),
                                               std::defer_lock);
              runtime::program::collective_boundary_provider_phase(
                  lane, "prepared System BoundaryFlux session admission failed collectively", [&] {
                    if (!invocation_lock.try_lock())
                      throw std::logic_error(
                          "prepared BoundaryFlux dependency cycle/reentrancy detected");
                  });
              runtime::program::collective_boundary_provider_phase(
                  lane, "prepared System BoundaryFlux dependency refresh failed collectively",
                  [&] { dependencies->refresh(point, lane); });
              FieldBoundaryExecutionContext<Dim> context;
              runtime::program::collective_boundary_provider_phase(
                  lane, "prepared System BoundaryFlux context validation failed collectively",
                  [&] { context = dependencies->view(point); });
              runtime::program::collective_boundary_provider_phase(
                  lane, "prepared System BoundaryFlux callback failed collectively", [&] {
                    component->template transform_boundary_flux<Dim>(
                        session->value(), point, state, faces, geometry, lane, context);
                  });
            };
        hook->flux.swap(candidate);
        snapshot.prepared_boundary_hook_contracts.push_back(
            {package_identity, block, "flux", component_contract, component_authority_contract});
      },
      component, nullptr, NativePackageKind::prepared_boundary);
  p_->pending_native_packages_.back().expected_boundary_hooks = std::move(expected_hooks);
}

template <int Dim>
void System<Dim>::stage_prepared_field_boundary_component_pair(
    const std::string& block, std::shared_ptr<PreparedFieldBoundaryResidualComponent> residual,
    std::shared_ptr<PreparedFieldBoundaryJvpComponent> jvp) {
  require_assembling(p_->lifecycle_, "stage_prepared_field_boundary_component_pair");
  if (block.empty() || !residual || !jvp || residual->spec().region.dimension != Dim ||
      jvp->spec().region.dimension != Dim ||
      residual->spec().state_identity != p_->boundary_registry_.state_route(block))
    throw std::invalid_argument(
        "prepared System FieldBoundary pair differs from its exact block route");
  const auto& left = residual->spec();
  const auto& right = jvp->spec();
  if (left.target_identity != right.target_identity || left.target_json != right.target_json ||
      left.component_id != right.component_id ||
      left.manifest_identity != right.manifest_identity ||
      left.producer_identity != right.producer_identity ||
      left.state_identity != right.state_identity || left.ghost_identity != right.ghost_identity ||
      left.layout_identity != right.layout_identity ||
      left.region.identity != right.region.identity || left.region.kind != right.region.kind ||
      left.region.codimension != right.region.codimension ||
      left.region.axes != right.region.axes || left.region.sides != right.region.sides ||
      left.interface_version != right.interface_version || left.states != right.states ||
      left.fields != right.fields || left.parameter_ids != right.parameter_ids ||
      left.parameter_values != right.parameter_values ||
      left.parameters_json != right.parameters_json || left.outputs.size() != 1 ||
      right.outputs.size() != 1 || left.outputs.front() == right.outputs.front() ||
      left.rate != right.rate || left.nonlinear_iterate != right.nonlinear_iterate ||
      !left.directions.empty() || right.directions.size() != 1 ||
      right.directions.front() != right.state_identity)
    throw std::invalid_argument("prepared System FieldBoundary residual/JVP contract differs");
  if (residual->package_owner_identity() != jvp->package_owner_identity() ||
      !left.execution->equivalent_to(*right.execution))
    throw std::invalid_argument(
        "prepared System FieldBoundary residual/JVP package/execution authority differs");
  ExactContractBuilder package_contract;
  package_contract.text("pops.system.prepared-boundary-component-pair")
      .scalar(std::uint32_t{1})
      .text(block);
  append_prepared_boundary_component_contract(package_contract, left);
  append_prepared_boundary_component_contract(package_contract, right);
  const std::string package_identity = std::move(package_contract).release();
  const std::string residual_contract = prepared_system_boundary_component_contract(left);
  const std::string jvp_contract = prepared_system_boundary_component_contract(right);
  const std::string residual_authority_contract =
      prepared_system_boundary_component_authority_contract(
          residual->package_owner_identity(), POPS_NATIVE_INTERFACE_FIELD_BOUNDARY_CLOSURE_V1,
          left.interface_version);
  const std::string jvp_authority_contract = prepared_system_boundary_component_authority_contract(
      jvp->package_owner_identity(), POPS_NATIVE_INTERFACE_FIELD_BOUNDARY_CLOSURE_V1,
      right.interface_version);
  std::vector<typename Impl::PreparedBoundaryHookContract> expected_hooks;
  expected_hooks.push_back(
      {package_identity, block, "field-residual", residual_contract, residual_authority_contract});
  expected_hooks.push_back(
      {package_identity, block, "field-jvp", jvp_contract, jvp_authority_contract});
  stage_native_package_(
      package_identity, {},
      [this, block, residual, jvp, package_identity, residual_contract, jvp_contract,
       residual_authority_contract, jvp_authority_contract] {
        const ExecutionLane& lane = prepared_boundary_execution_lane();
        if (p_->native_package_finalize_candidate_ == nullptr)
          throw std::logic_error(
              "prepared System FieldBoundary installer requires a detached candidate");
        auto& snapshot = *p_->native_package_finalize_candidate_;
        typename Impl::Species* selected = nullptr;
        typename Impl::NativePackageFinalizeSnapshot::BoundaryHookImage* hook = nullptr;
        std::optional<Geometry<Dim>> geometry;
        std::optional<BoundaryTopology<Dim>> topology;
        runtime::program::collective_boundary_provider_phase(
            lane, "prepared System FieldBoundary local installer preflight failed collectively",
            [&] {
              selected = &snapshot.blocks.find(block);
              hook = &snapshot.boundary_hook(block);
              if (!selected->boundary || !hook->residual_target || !hook->jvp_target)
                throw std::invalid_argument(
                    "prepared System FieldBoundary pair requires generated boundary closures");
              geometry.emplace(p_->geom);
              topology.emplace(BoundaryTopology<Dim>::axis_periodic(p_->periodicity));
            });
        const auto& spec = residual->spec();
        const int face = spec.region.axes.front() * 2 + (spec.region.sides.front() > 0 ? 1 : 0);
        auto prepared_residual = prepare_system_boundary_component_candidate<Dim>(
            p_->periodicity, snapshot.blocks, snapshot.boundary_registry, snapshot.named_fields,
            residual, selected->U, *geometry, *topology, lane,
            ExternalBoundaryDependencyOperation::field_closure);
        auto residual_dependencies = std::move(prepared_residual.dependencies);
        auto residual_session = std::move(prepared_residual.session);
        auto prepared_jvp = prepare_system_boundary_component_candidate<Dim>(
            p_->periodicity, snapshot.blocks, snapshot.boundary_registry, snapshot.named_fields,
            jvp, selected->U, *geometry, *topology, lane,
            ExternalBoundaryDependencyOperation::field_closure);
        auto jvp_dependencies = std::move(prepared_jvp.dependencies);
        auto jvp_session = std::move(prepared_jvp.session);
        const auto previous_residual = *hook->residual_target;
        const auto previous_jvp = *hook->jvp_target;
        const Geometry<Dim> prepared_geometry = *geometry;
        typename SystemBlockClosures<Dim>::PreparedPointBoundaryResidual residual_candidate =
            [previous_residual, residual, residual_session, residual_dependencies,
             geometry = prepared_geometry, face](const auto& point, auto& state, auto& result,
                                                 const auto& boundary, const auto& lane,
                                                 const auto& transport) {
              if (previous_residual)
                runtime::program::collective_boundary_provider_phase(
                    lane, "prepared System FieldBoundary residual predecessor failed",
                    [&] { previous_residual(point, state, result, boundary, lane, transport); });
              std::unique_lock invocation_lock(residual_session->value().invocation_mutex(),
                                               std::defer_lock);
              runtime::program::collective_boundary_provider_phase(
                  lane, "prepared System FieldBoundary residual session admission failed", [&] {
                    if (!invocation_lock.try_lock())
                      throw std::logic_error(
                          "prepared FieldBoundary residual dependency cycle/reentrancy detected");
                  });
              runtime::program::collective_boundary_provider_phase(
                  lane, "prepared System FieldBoundary residual dependency refresh failed",
                  [&] { residual_dependencies->refresh(point, lane); });
              FieldBoundaryExecutionContext<Dim> context;
              runtime::program::collective_boundary_provider_phase(
                  lane, "prepared System FieldBoundary residual context validation failed",
                  [&] { context = residual_dependencies->view(point); });
              runtime::program::collective_boundary_provider_phase(
                  lane, "prepared System FieldBoundary residual callback failed", [&] {
                    residual->template evaluate_field_boundary_face<Dim>(
                        residual_session->value(), face, state, nullptr, result, geometry, context);
                  });
            };
        typename SystemBlockClosures<Dim>::PreparedPointJvp jvp_candidate =
            [previous_jvp, jvp, jvp_session, jvp_dependencies, geometry = prepared_geometry, face](
                const auto& point, auto& state, const auto& direction, auto& result,
                const auto& boundary, const auto& lane, const auto& transport) {
              if (previous_jvp)
                runtime::program::collective_boundary_provider_phase(
                    lane, "prepared System FieldBoundary JVP predecessor failed", [&] {
                      previous_jvp(point, state, direction, result, boundary, lane, transport);
                    });
              std::unique_lock invocation_lock(jvp_session->value().invocation_mutex(),
                                               std::defer_lock);
              runtime::program::collective_boundary_provider_phase(
                  lane, "prepared System FieldBoundary JVP session admission failed", [&] {
                    if (!invocation_lock.try_lock())
                      throw std::logic_error(
                          "prepared FieldBoundary JVP dependency cycle/reentrancy detected");
                  });
              runtime::program::collective_boundary_provider_phase(
                  lane, "prepared System FieldBoundary JVP dependency refresh failed",
                  [&] { jvp_dependencies->refresh(point, lane); });
              FieldBoundaryExecutionContext<Dim> context;
              runtime::program::collective_boundary_provider_phase(
                  lane, "prepared System FieldBoundary JVP context validation failed",
                  [&] { context = jvp_dependencies->view(point); });
              runtime::program::collective_boundary_provider_phase(
                  lane, "prepared System FieldBoundary JVP callback failed", [&] {
                    jvp->template evaluate_field_boundary_face<Dim>(
                        jvp_session->value(), face, state, &direction, result, geometry, context);
                  });
            };
        hook->residual.swap(residual_candidate);
        hook->jvp.swap(jvp_candidate);
        snapshot.prepared_boundary_hook_contracts.push_back({package_identity, block,
                                                             "field-residual", residual_contract,
                                                             residual_authority_contract});
        snapshot.prepared_boundary_hook_contracts.push_back(
            {package_identity, block, "field-jvp", jvp_contract, jvp_authority_contract});
      },
      std::make_shared<std::pair<std::shared_ptr<PreparedFieldBoundaryResidualComponent>,
                                 std::shared_ptr<PreparedFieldBoundaryJvpComponent>>>(residual,
                                                                                      jvp),
      nullptr, NativePackageKind::prepared_boundary);
  p_->pending_native_packages_.back().expected_boundary_hooks = std::move(expected_hooks);
}

template <int Dim>
void System<Dim>::install_hyperbolic_boundary(
    const std::string& name, const std::string& identity, int required_depth,
    const std::vector<std::string>& face_types, const std::vector<double>& face_values,
    const std::vector<std::string>& face_identities,
    const std::vector<std::string>& component_roles, const std::string& state_identity,
    const std::vector<std::string>& face_representations,
    const std::vector<std::string>& face_converter_identities,
    const std::vector<std::vector<std::string>>& face_analytic_opcodes,
    const std::vector<std::vector<double>>& face_analytic_literals,
    const std::vector<std::string>& face_analytic_clocks) {
  require_assembling(p_->lifecycle_, "install_hyperbolic_boundary");
  if (std::any_of(p_->sp.begin(), p_->sp.end(),
                  [&](const typename Impl::Species& block) { return block.name == name; }))
    throw std::logic_error(
        "System hyperbolic boundary must be prepared before its block is committed");
  p_->boundary_registry_.install_boundary(
      name, identity, required_depth, face_types, face_values, face_identities, component_roles,
      state_identity, face_representations, face_converter_identities, face_analytic_opcodes,
      face_analytic_literals, face_analytic_clocks);
}

template <int Dim>
void System<Dim>::install_prepared_hyperbolic_boundary(
    const std::string& name, const std::string& identity, int required_depth,
    const std::string& state_identity, std::shared_ptr<const HyperbolicBoundary> boundary) {
  require_assembling(p_->lifecycle_, "install_prepared_hyperbolic_boundary");
  if (std::any_of(p_->sp.begin(), p_->sp.end(),
                  [&](const typename Impl::Species& block) { return block.name == name; }))
    throw std::logic_error(
        "System hyperbolic boundary must be prepared before its block is committed");
  p_->boundary_registry_.install_boundary(name, identity, required_depth, state_identity,
                                          std::move(boundary));
}

template <int Dim>
void System<Dim>::discard_hyperbolic_boundaries() {
  require_assembling(p_->lifecycle_, "discard_hyperbolic_boundaries");
  for (typename Impl::Species& block : p_->sp) {
    block.boundary.reset();
    block.state_identity.clear();
  }
  p_->boundary_registry_.discard_transaction();
  std::erase_if(p_->pending_native_packages_, [](const auto& package) {
    return package.kind == NativePackageKind::prepared_boundary;
  });
  prepared_boundary_execution_lane_.reset();
}

template <int Dim>
void System<Dim>::install_interface_provider(SystemInterfaceProvider<Dim> provider) {
  require_assembling(p_->lifecycle_, "install_interface_provider");
  p_->blocks_.install_interface_provider(std::move(provider));
}

template <int Dim>
void System<Dim>::discard_interface_flux_components() {
  require_assembling(p_->lifecycle_, "discard_interface_flux_components");
  p_->blocks_.discard_interface_fluxes();
}

template <int Dim>
std::size_t System<Dim>::interface_evaluation_count(const std::string& identity, int level) const {
  if (identity.empty() || level < 0)
    throw std::invalid_argument(
        "System interface evaluation query requires an identity and non-negative level");
  return p_->blocks_.interface_evaluation_count(identity, level);
}

template <int Dim>
Geometry<Dim> System<Dim>::prepared_block_geometry() const {
  return p_->geom;
}

template <int Dim>
std::array<bool, Dim> System<Dim>::prepared_block_periodicity() const {
  return p_->periodicity;
}

template <int Dim>
void System<Dim>::install_prepared_block(PreparedSystemBlock<Dim> prepared) {
  require_assembling(p_->lifecycle_, "install_prepared_block");
  const std::string provider_consumer_qid = prepared.name;
  auto* finalize = p_->native_package_finalize_candidate_;
  auto& auxiliary_registry =
      finalize == nullptr ? p_->auxiliary_registry_ : finalize->auxiliary_registry;
  auto& blocks = finalize == nullptr ? p_->blocks_ : finalize->blocks;
  auto& boundary_registry =
      finalize == nullptr ? p_->boundary_registry_ : finalize->boundary_registry;
  auto candidate =
      prepare_block_installation<Dim>(*p_, auxiliary_registry, blocks, boundary_registry,
                                      provider_consumer_qid, std::move(prepared));
  blocks.blocks.push_back(std::move(candidate.block));
  if (candidate.converted_boundary)
    boundary_registry.boundary(candidate.name).authority = std::move(candidate.converted_boundary);
}

template <int Dim>
void System<Dim>::register_elliptic_field(
    const std::string& block, const std::string& field,
    const std::vector<runtime::system::AuxiliaryComponentKey>& output_keys, int gradient_sign) {
  require_assembling(p_->lifecycle_, "register_elliptic_field");
  auto* finalize = p_->native_package_finalize_candidate_;
  if (finalize == nullptr && !p_->pending_native_packages_.empty()) {
    if (block.empty() || field.empty() || output_keys.empty())
      throw std::invalid_argument(
          "System staged native field output requires non-empty block, field, and keys");
    runtime::field::NamedFieldOutput<Dim> shape(output_keys.size(), gradient_sign);
    (void)shape;
    auto selected = p_->field_plans_.end();
    for (auto plan = p_->field_plans_.begin(); plan != p_->field_plans_.end(); ++plan) {
      if (plan->second.output_block != block || plan->second.output_key != field)
        continue;
      if (selected != p_->field_plans_.end())
        throw std::runtime_error(
            "System staged native field output resolves to multiple qualified provider slots");
      selected = plan;
    }
    if (selected == p_->field_plans_.end())
      throw std::invalid_argument(
          "System staged native field output has no resolved exact-ranked field plan");
    std::set<std::string> exact_keys;
    for (const auto& key : output_keys) {
      key.validate();
      if (!exact_keys.insert(key.exact_key()).second)
        throw std::invalid_argument("System staged native field output keys must be unique");
    }
    if (p_->named_fields_.contains(selected->first) ||
        !p_->staged_native_field_outputs_
             .emplace(
                 selected->first,
                 typename Impl::StagedNativeFieldOutput{block, field, output_keys, gradient_sign})
             .second)
      throw std::invalid_argument("System native field output is already registered: " +
                                  selected->first);
    return;
  }
  const ExecutionLane& lane = prepared_boundary_execution_lane();
  auto& auxiliary_registry =
      finalize == nullptr ? p_->auxiliary_registry_ : finalize->auxiliary_registry;
  auto& blocks = finalize == nullptr ? p_->blocks_ : finalize->blocks;
  auto& field_plans = finalize == nullptr ? p_->field_plans_ : finalize->field_plans;
  auto& configured_field_solver_providers = finalize == nullptr
                                                ? p_->configured_field_solver_providers_
                                                : finalize->configured_field_solver_providers;
  auto& component_field_solver_providers = finalize == nullptr
                                               ? p_->component_field_solver_providers_
                                               : finalize->component_field_solver_providers;
  auto& named_fields = finalize == nullptr ? p_->named_fields_ : finalize->named_fields;
  const auto& default_nullspace_provider_identity =
      finalize == nullptr ? p_->default_nullspace_provider_identity_
                          : finalize->default_nullspace_provider_identity;
  const auto& default_nullspace_options =
      finalize == nullptr ? p_->default_nullspace_options_ : finalize->default_nullspace_options;
  std::optional<runtime::field::NamedFieldOutput<Dim>> output;
  auto selected = field_plans.end();
  std::optional<BoundaryTopology<Dim>> topology;
  std::optional<elliptic::nd::CartesianPoissonOptions<Dim>> operator_options;
  runtime::program::collective_boundary_provider_phase(
      lane, "System named elliptic local preflight failed collectively", [&] {
        if (field.empty())
          throw std::invalid_argument("System named elliptic field identity must be non-empty");
        (void)blocks.find(block);
        output.emplace(output_keys.size(), gradient_sign);
        if (!auxiliary_registry.sealed())
          throw std::logic_error(
              "System named elliptic outputs require a sealed auxiliary provider registry");
        std::vector<std::string> exact_output_keys;
        exact_output_keys.reserve(output_keys.size());
        for (const auto& key : output_keys) {
          key.validate();
          const std::string exact_key = key.exact_key();
          if (std::find(exact_output_keys.begin(), exact_output_keys.end(), exact_key) !=
              exact_output_keys.end())
            throw std::invalid_argument("System named elliptic output keys must be unique");
          exact_output_keys.push_back(exact_key);
          if (auxiliary_registry.provider_for_key(key).kind() !=
              runtime::system::AuxiliaryProviderKind::field_output)
            throw std::invalid_argument(
                "System named elliptic output key is not owned by a field-output provider");
        }
        for (auto plan = field_plans.begin(); plan != field_plans.end(); ++plan) {
          if (plan->second.output_block != block || plan->second.output_key != field)
            continue;
          if (selected != field_plans.end())
            throw std::runtime_error(
                "System named elliptic output resolves to multiple qualified provider slots");
          selected = plan;
        }
        if (selected == field_plans.end())
          throw std::invalid_argument(
              "System named elliptic output has no resolved exact-ranked field plan");
        const auto& candidate_plan = selected->second;
        if (named_fields.contains(selected->first))
          throw std::invalid_argument("System named elliptic field is already registered: " +
                                      selected->first);
        if ((!candidate_plan.boundary_state_blocks.empty() ||
             !candidate_plan.boundary_field_blocks.empty()) &&
            !candidate_plan.boundary_kernel)
          throw std::logic_error(
              "System field boundary dependencies require one compiled exact-ranked kernel");
        topology.emplace(BoundaryTopology<Dim>::axis_periodic(p_->periodicity));
        elliptic::nd::CartesianBoundaryKind physical =
            elliptic::nd::CartesianBoundaryKind::dirichlet;
        if (p_->poisson_bc_ == "neumann")
          physical = elliptic::nd::CartesianBoundaryKind::neumann;
        else if (p_->poisson_bc_ != "auto" && p_->poisson_bc_ != "dirichlet" &&
                 p_->poisson_bc_ != "periodic")
          throw std::invalid_argument("System Poisson boundary mode is unknown");
        operator_options.emplace(
            elliptic::nd::CartesianPoissonOptions<Dim>::from_topology(*topology, physical));
        if (!candidate_plan.boundary_kind.empty()) {
          for (int axis = 0; axis < Dim; ++axis) {
            for (int side = 0; side < 2; ++side) {
              const std::size_t face = static_cast<std::size_t>(2 * axis + side);
              const std::string& kind = candidate_plan.boundary_kind[face];
              const bool periodic = kind == "periodic";
              if (periodic != p_->periodicity[static_cast<std::size_t>(axis)])
                throw std::invalid_argument(
                    "System field boundary periodicity differs from its exact domain topology");
              if (kind == "periodic")
                operator_options->boundaries[face] = elliptic::nd::CartesianBoundaryKind::periodic;
              else if (kind == "dirichlet")
                operator_options->boundaries[face] = elliptic::nd::CartesianBoundaryKind::dirichlet;
              else if (kind == "neumann")
                operator_options->boundaries[face] = elliptic::nd::CartesianBoundaryKind::neumann;
              else if (kind == "mixed")
                operator_options->boundaries[face] = elliptic::nd::CartesianBoundaryKind::mixed;
              else
                throw std::logic_error("System exact-ranked field boundary kind is unsupported");
              operator_options->boundary_alpha[face] =
                  static_cast<Real>(candidate_plan.boundary_alpha[face]);
              operator_options->boundary_beta[face] =
                  static_cast<Real>(candidate_plan.boundary_beta[face]);
              operator_options->boundary_values[face] =
                  static_cast<Real>(candidate_plan.boundary_value[face]);
            }
          }
        }
        const auto component =
            component_field_solver_providers.find(candidate_plan.backend_provider_route);
        const auto configured =
            configured_field_solver_providers.find(candidate_plan.backend_provider_route);
        if ((component == component_field_solver_providers.end()) ==
            (configured == configured_field_solver_providers.end()))
          throw std::runtime_error(
              "System field plan must select exactly one installed exact-ranked backend route");
        if (component != component_field_solver_providers.end()) {
          if (candidate_plan.boundary_kernel)
            throw std::logic_error(
                "external exact field components must own dynamic boundaries in their component "
                "ABI");
          if (candidate_plan.has_reaction)
            throw std::logic_error(
                "external field reaction must be carried by the component's exact operator "
                "contract");
          if (!candidate_plan.boundary_kind.empty() &&
              std::any_of(candidate_plan.boundary_kind.begin(), candidate_plan.boundary_kind.end(),
                          [](const std::string& kind) { return kind != "periodic"; }))
            throw std::logic_error(
                "external field components currently require a fully periodic exact topology");
        } else if (configured->second.family_route == "fft") {
          if (candidate_plan.has_reaction || candidate_plan.boundary_kernel ||
              candidate_plan.newton)
            throw std::logic_error(
                "FFT field solver rejects reaction, dynamic boundary, and Newton contracts");
          for (int axis = 0; axis < Dim; ++axis) {
            if (!p_->periodicity[static_cast<std::size_t>(axis)])
              throw std::logic_error(
                  "FFT field solver requires periodic System topology on every axis");
            for (int side = 0; side < 2; ++side)
              if (operator_options->boundaries[static_cast<std::size_t>(2 * axis + side)] !=
                  elliptic::nd::CartesianBoundaryKind::periodic)
                throw std::logic_error(
                    "FFT field solver requires periodic exact field boundaries on every face");
          }
        } else if (configured->second.family_route == "cartesian_cg") {
          if (candidate_plan.has_reaction)
            throw std::logic_error(
                "configured exact Cartesian field solver does not implement a reaction operator");
          operator_options->absolute_tolerance =
              static_cast<Real>(configured->second.absolute_tolerance);
          operator_options->relative_tolerance =
              static_cast<Real>(configured->second.relative_tolerance);
          operator_options->maximum_iterations = configured->second.maximum_iterations;
        } else {
          throw std::logic_error("configured exact field solver route was not decoded exactly");
        }
      });
  const std::string& provider_slot = selected->first;
  const typename Impl::FieldPlan& plan = selected->second;
  auto& prepared_topology = *topology;
  auto& prepared_options = *operator_options;
  std::unique_ptr<runtime::system::ExactFieldSolverBackend<Dim>> backend;
  const auto component = component_field_solver_providers.find(plan.backend_provider_route);
  const auto configured = configured_field_solver_providers.find(plan.backend_provider_route);
  if ((component == component_field_solver_providers.end()) ==
      (configured == configured_field_solver_providers.end()))
    throw std::runtime_error(
        "System field plan must select exactly one installed exact-ranked backend route");

  if (component != component_field_solver_providers.end()) {
    backend = runtime::system::ComponentFieldSolverBackend<Dim>::prepare_collectively(
        component->second->provider_identity(), p_->geom, p_->ba, p_->dm, p_->local_rank,
        prepared_topology, p_->periodicity, component->second, lane);
  } else {
    if (configured->second.family_route == "fft") {
      std::optional<EllipticBuildRequest<Dim>> request;
      std::exception_ptr request_error;
      try {
        request.emplace(fft_build_request(p_->geom, p_->ba, p_->dm, p_->local_rank));
      } catch (...) {
        request_error = std::current_exception();
      }
      if (all_reduce_max(request_error ? 1L : 0L, lane) != 0) {
        if (lane.size() == 1 && request_error)
          std::rethrow_exception(request_error);
        throw std::runtime_error("FFT field solver request allocation failed collectively");
      }
      backend = runtime::system::PoissonFftFieldSolverBackend<Dim>::prepare_collectively(
          std::move(*request), configured->second.exact_identity, lane);
    } else if (configured->second.family_route == "cartesian_cg") {
      backend = runtime::system::CartesianCgFieldSolverBackend<Dim>::prepare_collectively(
          p_->geom, p_->ba, p_->dm, p_->local_rank, prepared_topology, prepared_options, lane,
          configured->second.exact_identity);
    }
  }

  std::shared_ptr<typename System<Dim>::Impl::exact_field_type> prepared;
  std::optional<FieldNullspaceProviderSelection> nullspace_selection;
  std::string topology_identity;
  runtime::program::collective_boundary_provider_phase(
      lane, "System named elliptic image preparation failed collectively", [&] {
        prepared = std::make_shared<typename System<Dim>::Impl::exact_field_type>(
            provider_slot, block, *output, p_->geom, p_->ba, p_->dm, p_->local_rank,
            std::move(backend), blocks.blocks.size(), lane, output_keys);
        if (plan.boundary_kernel)
          prepared->install_boundary_kernel(*plan.boundary_kernel);
        if (plan.newton)
          prepared->install_newton(*plan.newton);
        nullspace_selection.emplace(FieldNullspaceProviderSelection{
            plan.nullspace_provider_identity.empty() ? default_nullspace_provider_identity
                                                     : plan.nullspace_provider_identity,
            plan.nullspace_provider_identity.empty() ? default_nullspace_options
                                                     : plan.nullspace_options});
        topology_identity = plan.topology_digest.empty() ? plan.plan_identity + ":uniform-topology"
                                                         : plan.topology_digest;
      });
  std::optional<FieldNullspaceProviderRequest<Dim>> nullspace_request;
  runtime::program::collective_boundary_provider_phase(
      lane, "System named elliptic nullspace request preparation failed collectively", [&] {
        nullspace_request.emplace(p_->prepare_uniform_field_nullspace_request(
            plan.plan_identity, topology_identity, prepared_options, prepared->accepted_potential(),
            plan.has_reaction));
      });
  PreparedFieldNullspace<Dim> prepared_nullspace =
      p_->finish_uniform_field_nullspace(*nullspace_selection, std::move(*nullspace_request), lane);
  runtime::program::collective_boundary_provider_phase(
      lane, "System named elliptic nullspace installation failed collectively", [&] {
        prepared->install_nullspace(std::move(prepared_nullspace),
                                    PreparedVectorDistribution<Dim>::distributed());
      });
  runtime::program::collective_boundary_provider_phase(
      lane, "System named elliptic publication preparation failed collectively",
      [&] { named_fields.emplace(provider_slot, std::move(prepared)); });
}

template <int Dim>
void System<Dim>::set_block_elliptic_field(
    const std::string& block_name, const std::string& field,
    std::function<void(const MultiFab<Dim>&, MultiFab<Dim>&)> rhs) {
  require_assembling(p_->lifecycle_, "set_block_elliptic_field");
  if (field.empty() || !rhs)
    throw std::invalid_argument(
        "System named elliptic RHS requires a field identity and prepared closure");
  auto* finalize = p_->native_package_finalize_candidate_;
  auto& blocks = finalize == nullptr ? p_->blocks_ : finalize->blocks;
  auto& field_plans = finalize == nullptr ? p_->field_plans_ : finalize->field_plans;
  auto& named_fields = finalize == nullptr ? p_->named_fields_ : finalize->named_fields;
  const int block = blocks.index(block_name);
  auto selected = field_plans.end();
  Real coefficient = Real(0);
  for (auto plan = field_plans.begin(); plan != field_plans.end(); ++plan) {
    if (plan->second.output_key != field)
      continue;
    Real candidate = Real(0);
    bool contributes = false;
    for (const typename Impl::FieldProviderBinding& binding : plan->second.providers) {
      // ``field`` identifies the qualified output route selected above.  A source provider key is
      // deliberately independent (for example electron_charge -> electrostatic); generated block
      // installation supplies the one RHS closure owned by this block, so its exact coefficient is
      // resolved by block ownership rather than by equating input and output names.
      if (binding.block == block_name) {
        candidate += static_cast<Real>(binding.coefficient);
        contributes = true;
      }
    }
    if (!contributes)
      continue;
    if (selected != field_plans.end())
      throw std::runtime_error("System elliptic RHS resolves to multiple qualified provider slots");
    selected = plan;
    coefficient = candidate;
  }
  if (selected == field_plans.end())
    throw std::invalid_argument("System elliptic RHS has no resolved provider binding");
  const auto provider = named_fields.find(selected->first);
  if (provider == named_fields.end())
    throw std::invalid_argument("System named elliptic field is not registered: " +
                                selected->first);
  if (finalize == nullptr) {
    provider->second->add_rhs(static_cast<std::size_t>(block), std::move(rhs), coefficient);
  } else {
    auto image = finalize->named_field_rhs_images.find(selected->first);
    if (image == finalize->named_field_rhs_images.end()) {
      provider->second->add_rhs(static_cast<std::size_t>(block), std::move(rhs), coefficient);
    } else {
      const std::size_t block_index = static_cast<std::size_t>(block);
      if (block_index >= image->second.size())
        throw std::out_of_range(
            "System detached named-field RHS block is outside its prepared image");
      provider->second->append_rhs(image->second, block_index, std::move(rhs), coefficient);
    }
  }
}

template <int Dim>
void System<Dim>::add_dt_bound(const std::string& label, std::function<double()> function) {
  require_assembling(p_->lifecycle_, "add_dt_bound");
  if (label.empty() || !function)
    throw std::invalid_argument("System global dt bound requires a non-empty label and provider");
  p_->coupling_.dt_bounds.push_back({label, std::move(function)});
}

template <int Dim>
std::string System<Dim>::last_dt_bound() const {
  return p_->last_dt_reason_;
}

template <int Dim>
void System<Dim>::register_native_package(const std::string& name, const std::string& so_path,
                                          const std::string& expected_model_identity,
                                          const std::string& expected_binary_identity,
                                          const std::string& limiter, const std::string& riemann,
                                          const std::string& recon, const std::string& time,
                                          double gamma, int substeps, bool evolve, int stride,
                                          const std::vector<double>& params,
                                          double positivity_floor, NewtonOptions newton,
                                          bool newton_diagnostics) {
  require_assembling(p_->lifecycle_, "register_native_package");
  native_loader::register_native_package<Dim>(
      this, name, so_path, expected_model_identity, expected_binary_identity, limiter, riemann,
      recon, time, gamma, substeps, evolve, stride, params, positivity_floor, newton,
      newton_diagnostics);
}

template <int Dim>
void System<Dim>::register_external_riemann_package(
    const std::string& name, const std::string& so_path, const std::string& brick_id,
    const std::string& expected_sha256, int expected_nvars, int expected_provider_count,
    const std::string& expected_model_identity, const std::string& provider_consumer_qid,
    const std::string& limiter, const std::string& recon, const std::string& time, double gamma,
    int substeps, bool evolve, int stride, double positivity_floor, double weno_epsilon) {
  require_assembling(p_->lifecycle_, "register_external_riemann_package");
  runtime::program::detail::validate_external_install(name, limiter, recon, time,
                                                      provider_consumer_qid, gamma, substeps,
                                                      stride, positivity_floor, weno_epsilon);
  if (expected_sha256.size() != 64 ||
      !std::all_of(expected_sha256.begin(), expected_sha256.end(), [](char value) {
        return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
      }))
    throw std::invalid_argument(
        "System external Riemann package requires one lowercase SHA-256 digest");

  auto authority = std::make_shared<runtime::program::ExternalBrickHandle>(
      so_path, brick_id, expected_nvars, expected_provider_count, expected_model_identity,
      expected_sha256, true);
  // ExternalBrickHandle is also the AMR v5 authority. System opts into the distinct v7 prepared
  // receiver contract explicitly and rejects an older System installer before manifest or
  // staging callbacks.

  ExactContractBuilder exact;
  exact.text("pops.external-riemann.system-package")
      .scalar(std::uint32_t{7})
      .scalar(std::int32_t{Dim})
      .text(name)
      .text(brick_id)
      .text(expected_sha256)
      .scalar(std::int32_t{expected_nvars})
      .scalar(std::int32_t{expected_provider_count})
      .text(expected_model_identity)
      .text(provider_consumer_qid)
      .text(limiter)
      .text(recon)
      .text(time)
      .scalar(gamma)
      .scalar(std::int32_t{substeps})
      .scalar(evolve)
      .scalar(std::int32_t{stride})
      .scalar(positivity_floor)
      .scalar(weno_epsilon);
  std::string identity = std::move(exact).release();

  for (const auto& package : p_->pending_native_packages_)
    if (package.identity == identity)
      throw std::invalid_argument(
          "System external Riemann package identity is registered more than once");
  for (const auto& package : p_->installed_native_packages_)
    if (package.identity == identity)
      throw std::invalid_argument("System external Riemann package identity was already finalized");

  auto capability = std::make_shared<runtime::system::NativePackageCapabilityState<Dim>>();
  capability->identity = name;
  auto registrar_capability =
      runtime::system::NativePackageCapabilityFactory<Dim>::route_registrar(capability);
  auto installer_capability =
      runtime::system::NativePackageCapabilityFactory<Dim>::block_installer(capability);
  std::function<void()> installer = [authority, installer_capability, name, provider_consumer_qid,
                                     limiter, recon, time, gamma, substeps, evolve, stride,
                                     positivity_floor, weno_epsilon] {
    authority->install_system(installer_capability.get(), name, provider_consumer_qid, limiter,
                              recon, time, gamma, substeps, evolve, stride, positivity_floor,
                              weno_epsilon);
  };
  std::function<void()> route_registrar = [authority, capability, registrar_capability] {
    capability->phase = runtime::system::NativeCapabilityPhase::routes_open;
    authority->register_system_routes(*registrar_capability);
    capability->close_routes();
  };
  stage_prepared_native_package(std::move(identity), std::move(route_registrar),
                                std::move(installer), authority, std::move(capability));
}

template <int Dim>
void System<Dim>::stage_prepared_native_package(
    std::string identity, std::function<void()> route_registrar, std::function<void()> installer,
    std::shared_ptr<void> package_lifetime,
    std::shared_ptr<runtime::system::NativePackageCapabilityState<Dim>> capability) {
  stage_native_package_(std::move(identity), std::move(route_registrar), std::move(installer),
                        std::move(package_lifetime), std::move(capability),
                        NativePackageKind::generic);
}

template <int Dim>
void System<Dim>::stage_native_package_(
    std::string identity, std::function<void()> route_registrar, std::function<void()> installer,
    std::shared_ptr<void> package_lifetime,
    std::shared_ptr<runtime::system::NativePackageCapabilityState<Dim>> capability,
    NativePackageKind kind) {
  require_assembling(p_->lifecycle_, "stage_prepared_native_package");
  if (identity.empty() || !installer || !package_lifetime ||
      (kind == NativePackageKind::generic && !route_registrar))
    throw std::invalid_argument(
        "System native package staging requires an identity, canonical provider registrar, "
        "installer, and DSO lifetime");
  for (const auto& package : p_->pending_native_packages_)
    if (package.identity == identity)
      throw std::invalid_argument("System native package identity is registered more than once");
  for (const auto& package : p_->installed_native_packages_)
    if (package.identity == identity)
      throw std::invalid_argument("System native package identity was already finalized");
  p_->pending_native_packages_.push_back({std::move(package_lifetime), std::move(capability),
                                          std::move(identity), std::move(route_registrar),
                                          std::move(installer), kind});
}

template <int Dim>
void System<Dim>::finalize_native_packages() {
  require_assembling(p_->lifecycle_, "finalize_native_packages");
  if (p_->pending_native_packages_.empty())
    return;
  const ExecutionLane& lane = prepared_boundary_execution_lane();
  // Keep the package journal in its live owner until the final no-throw publication. A collective
  // rollback can then re-arm the revocable capabilities and retry the exact same authenticated
  // registrar/installer thunks without reallocating or losing their DSO lifetimes.
  auto& packages = p_->pending_native_packages_;
  std::vector<std::size_t> package_order;
  std::vector<std::string> exact_package_storage;
  std::vector<ExactOrderedBytePair> exact_packages;
  std::exception_ptr package_preparation_error;
  try {
    if (packages.empty())
      throw std::logic_error(
          "System native package finalization requires at least one staged package");
    package_order.resize(packages.size());
    std::iota(package_order.begin(), package_order.end(), std::size_t{0});
    std::sort(package_order.begin(), package_order.end(), [&](std::size_t left, std::size_t right) {
      return packages[left].identity < packages[right].identity;
    });
    exact_package_storage.reserve(packages.size());
    for (const std::size_t index : package_order) {
      const auto& package = packages[index];
      ExactContractBuilder exact_package;
      exact_package.text("pops.system-native-staged-package")
          .scalar(std::uint32_t{2})
          .text(package.identity)
          .scalar(static_cast<std::uint8_t>(package.kind))
          .scalar(static_cast<std::uint64_t>(package.expected_boundary_hooks.size()));
      for (const auto& expected : package.expected_boundary_hooks) {
        if (expected.package_identity != package.identity || expected.block.empty() ||
            expected.hook.empty() || expected.component_contract.empty() ||
            expected.component_authority_contract.empty())
          throw std::logic_error(
              "prepared boundary package has an incomplete exact staged hook contract");
        exact_package.text(expected.package_identity)
            .text(expected.block)
            .text(expected.hook)
            .bytes(expected.component_contract)
            .bytes(expected.component_authority_contract);
      }
      if (package.kind == NativePackageKind::generic) {
        if (!package.capability || !package.expected_boundary_hooks.empty())
          throw std::logic_error("generic native package has an invalid staged capability image");
      } else if (package.kind == NativePackageKind::prepared_boundary) {
        if (package.capability || package.expected_boundary_hooks.empty())
          throw std::logic_error("prepared boundary package has an invalid staged hook image");
      } else {
        throw std::logic_error("System native package kind is invalid");
      }
      exact_package_storage.push_back(std::move(exact_package).release());
    }
    exact_packages.reserve(exact_package_storage.size());
    for (const auto& bytes : exact_package_storage)
      exact_packages.emplace_back("system-native-package", bytes);
  } catch (...) {
    package_preparation_error = std::current_exception();
  }
  if (all_reduce_max(package_preparation_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && package_preparation_error)
      std::rethrow_exception(package_preparation_error);
    throw std::runtime_error("System native package identity preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(exact_packages, lane))
    throw std::runtime_error("System staged native packages differ across MPI ranks");

  std::vector<std::string> field_input_storage;
  std::vector<ExactOrderedBytePair> exact_field_inputs;
  std::exception_ptr field_input_error;
  try {
    field_input_storage.reserve(2);
    field_input_storage.push_back(p_->field_plan_registry_contract());
    field_input_storage.push_back(p_->staged_native_field_output_contract());
    exact_field_inputs.emplace_back("system-field-plan-registry", field_input_storage[0]);
    exact_field_inputs.emplace_back("system-staged-native-field-outputs", field_input_storage[1]);
  } catch (...) {
    field_input_error = std::current_exception();
  }
  if (all_reduce_max(field_input_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && field_input_error)
      std::rethrow_exception(field_input_error);
    throw std::runtime_error("System native field input witness preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(exact_field_inputs, lane))
    throw std::runtime_error("System native field inputs differ across MPI ranks");

  // Reserve publication storage before an installer can publish a live hook. The later move-only
  // publication is therefore allocation-free and retains every DSO owner until rollback has
  // completed on the exact lane.
  std::exception_ptr publication_prepare_error;
  try {
    p_->reserve_native_package_publication(packages.size());
  } catch (...) {
    publication_prepare_error = std::current_exception();
  }
  if (all_reduce_max(publication_prepare_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && publication_prepare_error)
      std::rethrow_exception(publication_prepare_error);
    throw std::runtime_error("System native package publication storage failed collectively");
  }

  std::optional<typename Impl::NativePackageFinalizeSnapshot> rollback_snapshot;
  std::optional<typename Impl::NativePackageFinalizeSnapshot> snapshot;
  std::exception_ptr snapshot_error;
  try {
    rollback_snapshot.emplace(*p_);
  } catch (...) {
    snapshot_error = std::current_exception();
  }
  if (all_reduce_max(snapshot_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && snapshot_error)
      std::rethrow_exception(snapshot_error);
    throw std::runtime_error("System native package rollback snapshot failed collectively");
  }

  std::exception_ptr failure;
  try {
    // Native callbacks write one detached aggregate; the live registry remains untouched until all
    // package-local work and the exact witness have converged.
    auto registry_candidate =
        std::make_shared<runtime::system::ExactAuxiliaryRegistry<Dim>>(p_->auxiliary_registry_);
    for (const std::size_t index : package_order) {
      const auto& package = packages[index];
      std::exception_ptr local_error;
      try {
        if (package.kind == NativePackageKind::generic) {
          if (!package.capability)
            throw std::logic_error("generic native package has no capability state");
          package.capability->detached_registry = registry_candidate;
        }
        if (package.register_routes)
          pops::dynlib::invoke_with_host_exception(package.register_routes,
                                                   "pops_register_provider_routes");
      } catch (...) {
        local_error = std::current_exception();
      }
      if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
        if (lane.size() == 1 && local_error)
          std::rethrow_exception(local_error);
        throw std::runtime_error("System native package route registrar failed collectively: " +
                                 package.identity);
      }
    }
    // The carrier object has stable address/lifetime for every generated closure. Its complete
    // candidate value is prepared before the transaction exposes that value to package callbacks.
    auto published_carrier_owner = p_->provider_carrier_;
    std::optional<runtime::system::AuxiliaryStorageGroups<Dim>> carrier_candidate;
    std::exception_ptr registry_prepare_error;
    try {
      registry_candidate->seal();
      if (registry_candidate->slot_count() != 0) {
        if (!published_carrier_owner)
          published_carrier_owner =
              std::make_shared<runtime::system::AuxiliaryStorageGroups<Dim>>();
        carrier_candidate.emplace();
        for (const auto& group : registry_candidate->storage_groups()) {
          Extent<Dim> ghosts{};
          for (int axis = 0; axis < Dim; ++axis)
            ghosts[axis] = group.shape.halo[axis];
          carrier_candidate->groups.emplace(
              group.identity, MultiFab<Dim>(p_->ba, p_->dm, p_->local_rank,
                                            static_cast<int>(group.component_count), ghosts));
        }
      }
    } catch (...) {
      registry_prepare_error = std::current_exception();
    }
    if (all_reduce_max(registry_prepare_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && registry_prepare_error)
        std::rethrow_exception(registry_prepare_error);
      throw std::runtime_error(
          "System detached auxiliary registry preparation failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"system-auxiliary-registry", registry_candidate->collective_contract()}}, lane))
      throw std::runtime_error("System detached auxiliary registry differs across MPI ranks");

    // This assembling-only transaction makes the exact candidate executable for generated package
    // and boundary dependency preflight. No user/runtime callback can enter concurrently. Failure
    // restores the previous value image from rollback_snapshot before any DSO owner is released;
    // success retains this exact image without a second carrier publication.
    if (carrier_candidate) {
      static_assert(std::is_nothrow_swappable_v<runtime::system::AuxiliaryStorageGroups<Dim>>);
      std::swap(*published_carrier_owner, *carrier_candidate);
    }

    // Freeze the complete live image before any installer runs. Generic packages first commit
    // their typed candidates; host materialization then extends this detached snapshot, and only
    // afterwards may prepared-boundary installers read or mutate its hook values.
    std::exception_ptr candidate_snapshot_error;
    try {
      snapshot.emplace(*p_);
    } catch (...) {
      candidate_snapshot_error = std::current_exception();
    }
    if (all_reduce_max(candidate_snapshot_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && candidate_snapshot_error)
        std::rethrow_exception(candidate_snapshot_error);
      throw std::runtime_error(
          "System native package detached image preparation failed collectively");
    }

    for (const std::size_t index : package_order) {
      const auto& package = packages[index];
      if (package.kind != NativePackageKind::generic)
        continue;
      std::exception_ptr local_error;
      try {
        auto& state = *package.capability;
        state.require(runtime::system::NativeCapabilityPhase::routes_closed,
                      "begin native package install");
        state.geometry.emplace(p_->geom);
        state.periodicity = p_->periodicity;
        state.provider_storage_owner = published_carrier_owner;
        state.phase = runtime::system::NativeCapabilityPhase::install_open;
        pops::dynlib::invoke_with_host_exception(package.install, "pops_install_native");
        if (!package.capability->commit_called || !package.capability->committed)
          throw std::logic_error(
              "System native package installer did not commit one complete candidate");
      } catch (...) {
        local_error = std::current_exception();
      }
      package.capability->revoke();
      if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
        if (lane.size() == 1 && local_error)
          std::rethrow_exception(local_error);
        throw std::runtime_error("System native package installer failed collectively: " +
                                 package.identity);
      }
    }
    std::vector<std::string> committed_contracts;
    std::vector<ExactOrderedBytePair> committed_witnesses;
    std::exception_ptr committed_witness_error;
    try {
      committed_contracts.reserve(package_order.size());
      committed_witnesses.reserve(package_order.size());
      for (const std::size_t index : package_order) {
        const auto& package = packages[index];
        if (package.kind != NativePackageKind::generic)
          continue;
        committed_contracts.push_back(
            runtime::system::exact_native_system_package_contract(*package.capability->committed));
        committed_witnesses.emplace_back("system-native-committed-package",
                                         committed_contracts.back());
      }
    } catch (...) {
      committed_witness_error = std::current_exception();
    }
    if (all_reduce_max(committed_witness_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && committed_witness_error)
        std::rethrow_exception(committed_witness_error);
      throw std::runtime_error(
          "System committed native package witness preparation failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(committed_witnesses, lane))
      throw std::runtime_error("System committed native package differs across MPI ranks");
    // Materialize every host-owned block and elliptic/backend image into the detached candidate.
    // No live registry, carrier, block, boundary, or field image is reachable through the
    // preparation helpers below.
    snapshot->auxiliary_registry.swap_complete(*registry_candidate);
    snapshot->provider_carrier_owner = published_carrier_owner;
    snapshot->auxiliary_registry_consensus_verified = true;
    p_->native_package_finalize_candidate_ = &*snapshot;
    for (const std::size_t index : package_order) {
      const auto& package = packages[index];
      if (package.kind != NativePackageKind::generic)
        continue;
      std::exception_ptr block_error;
      try {
        auto candidate = prepare_block_installation<Dim>(
            *p_, snapshot->auxiliary_registry, snapshot->blocks, snapshot->boundary_registry,
            package.capability->committed->consumer_qid,
            std::move(package.capability->committed->block));
        snapshot->blocks.blocks.push_back(std::move(candidate.block));
        snapshot->append_boundary_hook_image(snapshot->blocks.blocks.back());
        if (candidate.converted_boundary)
          snapshot->boundary_registry.boundary(candidate.name).authority =
              std::move(candidate.converted_boundary);
      } catch (...) {
        block_error = std::current_exception();
      }
      if (all_reduce_max(block_error ? 1L : 0L, lane) != 0) {
        if (lane.size() == 1 && block_error)
          std::rethrow_exception(block_error);
        throw std::runtime_error("System native block preparation failed collectively: " +
                                 package.identity);
      }
    }

    // Field plans and provider-bound output payloads were prepared before finalization and already
    // agree exactly on this lane. Materialize each backend once, against the complete detached block
    // and auxiliary image, before any package-owned RHS closure is attached.
    for (const auto& [slot, output] : snapshot->staged_native_field_outputs) {
      std::exception_ptr field_output_error;
      try {
        const auto plan = snapshot->field_plans.find(slot);
        if (plan == snapshot->field_plans.end() || output.block != plan->second.output_block ||
            output.field != plan->second.output_key)
          throw std::logic_error(
              "System detached native field output differs from its exact field plan");
        register_elliptic_field(output.block, output.field, output.output_keys,
                                output.gradient_sign);
      } catch (...) {
        field_output_error = std::current_exception();
      }
      if (all_reduce_max(field_output_error ? 1L : 0L, lane) != 0) {
        if (lane.size() == 1 && field_output_error)
          std::rethrow_exception(field_output_error);
        throw std::runtime_error("System native field output failed collectively: " + slot);
      }
    }

    std::set<std::pair<std::string, std::string>> expected_field_attachments;
    std::exception_ptr expected_field_error;
    try {
      for (const auto& [slot, plan] : snapshot->field_plans)
        for (const typename Impl::FieldProviderBinding& binding : plan.providers)
          expected_field_attachments.emplace(slot, binding.block);
    } catch (...) {
      expected_field_error = std::current_exception();
    }
    if (all_reduce_max(expected_field_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && expected_field_error)
        std::rethrow_exception(expected_field_error);
      throw std::runtime_error("System exact field attachment preparation failed collectively");
    }
    for (const std::size_t index : package_order) {
      const auto& package = packages[index];
      if (package.kind != NativePackageKind::generic)
        continue;
      auto& candidate = *package.capability->committed;
      for (auto& attachment : candidate.elliptic_attachments) {
        std::exception_ptr field_error;
        try {
          auto selected = snapshot->field_plans.end();
          for (auto plan = snapshot->field_plans.begin(); plan != snapshot->field_plans.end();
               ++plan) {
            if (plan->second.output_key != attachment.field)
              continue;
            const bool contributes =
                std::any_of(plan->second.providers.begin(), plan->second.providers.end(),
                            [&](const typename Impl::FieldProviderBinding& binding) {
                              return binding.block == package.capability->identity;
                            });
            if (!contributes)
              continue;
            if (selected != snapshot->field_plans.end())
              throw std::logic_error(
                  "System native elliptic attachment resolves to multiple field plans");
            selected = plan;
          }
          if (selected == snapshot->field_plans.end()) {
            // Convenience ChargeDensity packages emit an RHS-only fields_from_state
            // attachment. The default Poisson already lives on the prepared block
            // (poisson_rhs -> add_poisson_rhs). AMR stages ensure_default_field_plan();
            // uniform System keeps that default field off the named-plan registry.
            if (attachment.field == "fields_from_state" && attachment.outputs.empty())
              continue;
            throw std::logic_error("System native elliptic attachment has no resolved field plan");
          }
          const auto staged = snapshot->staged_native_field_outputs.find(selected->first);
          if (staged == snapshot->staged_native_field_outputs.end() ||
              staged->second.gradient_sign != attachment.gradient_sign ||
              (!attachment.outputs.empty() && staged->second.output_keys != attachment.outputs))
            throw std::logic_error(
                "System native elliptic attachment differs from its staged output contract");
          if (!expected_field_attachments.erase({selected->first, package.capability->identity}))
            throw std::logic_error(
                "System native elliptic attachment is duplicate or was not required");
          set_block_elliptic_field(package.capability->identity, attachment.field,
                                   std::move(attachment.rhs));
        } catch (...) {
          field_error = std::current_exception();
        }
        if (all_reduce_max(field_error ? 1L : 0L, lane) != 0) {
          if (lane.size() == 1 && field_error)
            std::rethrow_exception(field_error);
          throw std::runtime_error("System native elliptic attachment failed collectively: " +
                                   attachment.rhs_identity);
        }
      }
    }
    const long missing_field_attachments = expected_field_attachments.empty() ? 0L : 1L;
    if (all_reduce_max(missing_field_attachments, lane) != 0)
      throw std::runtime_error(
          "System native packages collectively omitted a required exact field RHS attachment");

    // Prepared-boundary packages are deliberately last: every generic package has already become
    // one complete block/field candidate. Their installers can only extend the detached hook
    // values above and cannot observe or mutate the live System block store.
    const std::size_t boundary_contract_begin = snapshot->prepared_boundary_hook_contracts.size();
    for (const std::size_t index : package_order) {
      const auto& package = packages[index];
      if (package.kind != NativePackageKind::prepared_boundary)
        continue;
      std::exception_ptr boundary_error;
      try {
        pops::dynlib::invoke_with_host_exception(package.install,
                                                 "prepared System boundary installer");
      } catch (...) {
        boundary_error = std::current_exception();
      }
      if (all_reduce_max(boundary_error ? 1L : 0L, lane) != 0) {
        if (lane.size() == 1 && boundary_error)
          std::rethrow_exception(boundary_error);
        throw std::runtime_error("System prepared boundary installer failed collectively: " +
                                 package.identity);
      }
      for (const auto& expected : package.expected_boundary_hooks)
        snapshot->publish_boundary_hook_noexcept(expected.block, expected.hook);
    }

    std::exception_ptr boundary_witness_error;
    try {
      std::set<std::string> expected_packages;
      std::vector<typename Impl::PreparedBoundaryHookContract> expected_hooks;
      for (const std::size_t index : package_order) {
        const auto& package = packages[index];
        if (package.kind != NativePackageKind::prepared_boundary)
          continue;
        if (package.expected_boundary_hooks.empty())
          throw std::logic_error(
              "System prepared boundary package has no exact staged hook contract");
        expected_packages.insert(package.identity);
        for (const auto& expected : package.expected_boundary_hooks) {
          if (expected.package_identity != package.identity)
            throw std::logic_error(
                "System prepared boundary staged hook differs from its package identity");
          expected_hooks.push_back(expected);
        }
      }
      const std::size_t boundary_contract_count =
          snapshot->prepared_boundary_hook_contracts.size() - boundary_contract_begin;
      if (boundary_contract_count != expected_hooks.size() ||
          !std::equal(expected_hooks.begin(), expected_hooks.end(),
                      snapshot->prepared_boundary_hook_contracts.begin() +
                          static_cast<std::ptrdiff_t>(boundary_contract_begin)))
        throw std::logic_error(
            "System prepared boundary installers differ from their exact staged hook contracts");
      std::set<std::string> materialized_packages;
      std::set<std::tuple<std::string, std::string, std::string, std::string, std::string>>
          exact_hooks;
      for (std::size_t index = boundary_contract_begin;
           index < snapshot->prepared_boundary_hook_contracts.size(); ++index) {
        const auto& record = snapshot->prepared_boundary_hook_contracts[index];
        if (!expected_packages.contains(record.package_identity) || record.block.empty() ||
            record.component_contract.empty() || record.component_authority_contract.empty())
          throw std::logic_error(
              "System prepared boundary installer published an unexpected hook contract");
        auto& hook = snapshot->boundary_hook(record.block);
        const bool callable =
            (record.hook == "ghost" && hook.ghost_target && *hook.ghost_target) ||
            (record.hook == "flux" && hook.flux_target && *hook.flux_target) ||
            (record.hook == "field-residual" && hook.residual_target && *hook.residual_target) ||
            (record.hook == "field-jvp" && hook.jvp_target && *hook.jvp_target);
        if (!callable ||
            !exact_hooks
                 .emplace(record.package_identity, record.block, record.hook,
                          record.component_contract, record.component_authority_contract)
                 .second)
          throw std::logic_error(
              "System prepared boundary hook contract is duplicate or not callable");
        materialized_packages.insert(record.package_identity);
      }
      if (materialized_packages != expected_packages)
        throw std::logic_error(
            "System prepared boundary packages did not materialize their exact hook contracts");
    } catch (...) {
      boundary_witness_error = std::current_exception();
    }
    if (all_reduce_max(boundary_witness_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && boundary_witness_error)
        std::rethrow_exception(boundary_witness_error);
      throw std::runtime_error("System prepared boundary hook witness failed collectively");
    }

    std::string materialized_contract;
    std::exception_ptr materialized_error;
    try {
      ExactContractBuilder materialized;
      materialized.text("pops.system-native-materialized-candidate")
          .scalar(std::uint32_t{2})
          .bytes(snapshot->auxiliary_registry.collective_contract())
          .scalar(static_cast<std::uint64_t>(snapshot->blocks.blocks.size()));
      for (std::size_t index = 0; index < snapshot->blocks.blocks.size(); ++index) {
        const auto& block = snapshot->blocks.blocks[index];
        const auto& hook = snapshot->boundary_hooks[index];
        materialized.text(block.name)
            .text(block.state_identity)
            .scalar(std::int32_t{block.ncomp})
            .scalar(std::int32_t{block.substeps})
            .scalar(block.evolve)
            .scalar(std::int32_t{block.stride})
            .scalar(std::int32_t{block.newton.max_iters})
            .scalar(static_cast<double>(block.newton.rel_tol))
            .scalar(static_cast<double>(block.newton.abs_tol))
            .scalar(static_cast<double>(block.newton.fd_eps))
            .scalar(static_cast<double>(block.newton.damping))
            .scalar(block.newton_diagnostics)
            .presence(static_cast<bool>(block.boundary))
            .presence(hook.ghost_target && static_cast<bool>(*hook.ghost_target))
            .presence(hook.flux_target && static_cast<bool>(*hook.flux_target))
            .presence(hook.residual_target && static_cast<bool>(*hook.residual_target))
            .presence(hook.jvp_target && static_cast<bool>(*hook.jvp_target));
      }
      materialized.scalar(
          static_cast<std::uint64_t>(snapshot->prepared_boundary_hook_contracts.size()));
      for (const auto& record : snapshot->prepared_boundary_hook_contracts)
        materialized.text(record.package_identity)
            .text(record.block)
            .text(record.hook)
            .bytes(record.component_contract)
            .bytes(record.component_authority_contract);
      materialized.scalar(static_cast<std::uint64_t>(snapshot->named_fields.size()));
      for (const auto& [slot, field] : snapshot->named_fields) {
        if (!field)
          throw std::logic_error("System detached native field candidate is null");
        materialized.text(slot)
            .text(field->identity())
            .text(field->output_block())
            .text(field->solver_provider_identity())
            .scalar(static_cast<std::uint64_t>(field->output_keys().size()));
        for (const auto& key : field->output_keys())
          materialized.text(key.exact_key());
      }
      materialized_contract = std::move(materialized).release();
    } catch (...) {
      materialized_error = std::current_exception();
    }
    if (all_reduce_max(materialized_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && materialized_error)
        std::rethrow_exception(materialized_error);
      throw std::runtime_error(
          "System materialized native candidate witness preparation failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"system-native-materialized-candidate", materialized_contract}}, lane))
      throw std::runtime_error("System materialized native candidate differs across MPI ranks");
    snapshot->staged_native_field_outputs.clear();
    p_->native_package_finalize_candidate_ = nullptr;
  } catch (...) {
    p_->native_package_finalize_candidate_ = nullptr;
    for (auto& package : packages)
      if (package.kind == NativePackageKind::generic && package.capability &&
          !package.capability->revoke_called)
        package.capability->revoke();
    // Sealing can copy DSO-owned launcher managers. Normalize even indirect package exceptions
    // while every package lifetime is still retained by ``packages``.
    try {
      const auto diagnostic = pops::dynlib::capture_foreign_exception("native finalization");
      pops::dynlib::throw_host_exception(diagnostic);
    } catch (...) {
      failure = std::current_exception();
    }
  }

  const long failed = failure ? 1L : 0L;
  if (all_reduce_max(failed, lane) != 0) {
    rollback_snapshot->restore_noexcept(*p_);
    for (auto& package : packages)
      if (package.kind == NativePackageKind::generic && package.capability)
        package.capability->reset_for_retry();
    if (lane.size() == 1 && failure)
      std::rethrow_exception(failure);
    throw std::runtime_error("System native package finalization rolled back collectively");
  }

  // All fallible work and exact-lane witnesses are complete. Publication is swaps/moves of already
  // allocated owners only, and cannot expose a partially committed native package.
  p_->auxiliary_registry_.swap_complete(snapshot->auxiliary_registry);
  if (p_->auxiliary_registry_.slot_count() != 0) {
    if (!snapshot->provider_carrier_owner)
      std::terminate();
    p_->provider_carrier_ = std::move(snapshot->provider_carrier_owner);
  } else {
    if (p_->provider_carrier_)
      std::terminate();
  }
  p_->auxiliary_registry_consensus_verified_ = true;
  p_->blocks_.blocks.swap(snapshot->blocks.blocks);
  p_->prepared_boundary_hook_contracts_.swap(snapshot->prepared_boundary_hook_contracts);
  static_assert(std::is_nothrow_swappable_v<typename Impl::boundary_registry_type>);
  std::swap(p_->boundary_registry_, snapshot->boundary_registry);
  snapshot->publish_named_field_rhs_noexcept();
  p_->named_fields_.swap(snapshot->named_fields);
  p_->staged_native_field_outputs_.swap(snapshot->staged_native_field_outputs);
  p_->field_plan_consensus_verified_ = true;

  p_->publish_reserved_native_packages_noexcept(packages);
}

template <int Dim>
void System<Dim>::add_coupled_source_prepared_(const CoupledSourceProgram& description,
                                               double frequency, const std::string& label,
                                               CouplingOperatorView inspect) {
  const ExecutionLane& lane = prepared_boundary_execution_lane();
  struct OutputRef {
    int block = -1;
    int component = -1;
    CsProgram program{};
  };

  std::vector<CoupledSourceInputReference> inputs;
  std::vector<OutputRef> outputs;
  std::vector<Real> constants;
  CsProgram frequency_program{};
  bool has_frequency_program = false;
  std::exception_ptr local_error;
  try {
    require_assembling(p_->lifecycle_, "add_coupled_source");
    if (label.empty() || !std::isfinite(frequency) || frequency < 0.0)
      throw std::invalid_argument(
          "System coupled source requires a non-empty label and finite non-negative frequency");
    if (description.in_blocks.size() != description.in_roles.size() ||
        description.out_blocks.size() != description.out_roles.size() ||
        description.out_blocks.size() != description.prog_lens.size() ||
        description.prog_ops.size() != description.prog_args.size() ||
        description.freq_prog_ops.size() != description.freq_prog_args.size())
      throw std::invalid_argument("System coupled-source arrays have inconsistent lengths");
    if (description.out_blocks.empty())
      throw std::invalid_argument("System coupled source requires at least one output term");
    if (description.in_blocks.size() + description.consts.size() >
            static_cast<std::size_t>(kCsMaxReg) ||
        description.out_blocks.size() > static_cast<std::size_t>(kCsMaxTerms))
      throw std::length_error("System coupled source exceeds its device register/term capacity");

    const auto resolve = [&](const std::string& block_name,
                             const std::string& token) -> CoupledSourceInputReference {
      const int block = p_->blocks_.index(block_name);
      const VariableSet& variables = p_->sp[static_cast<std::size_t>(block)].cons_vars;
      validate_variable_semantics<Dim>(variables, "System::add_coupled_source", block_name);
      if (!variables.user_roles.empty() && variables.user_roles.size() != variables.roles.size())
        throw std::invalid_argument("System block '" + block_name +
                                    "' has incomplete custom-role metadata");
      int component = -1;
      for (int candidate = 0; candidate < variables.size; ++candidate) {
        const VariableSemantic semantic = variables.roles[static_cast<std::size_t>(candidate)];
        bool matches = false;
        if (semantic == VariableSemantic::Custom) {
          const std::string_view label =
              variables.user_roles.empty()
                  ? std::string_view{}
                  : std::string_view(variables.user_roles[static_cast<std::size_t>(candidate)]);
          matches = label.empty() ? token == "custom" : label == token;
        } else {
          matches = role_name(semantic) == token;
        }
        if (!matches)
          continue;
        if (component >= 0)
          throw std::invalid_argument("System block '" + block_name + "' declares role token '" +
                                      token + "' more than once");
        component = candidate;
      }
      if (component < 0)
        throw std::invalid_argument("System block '" + block_name +
                                    "' does not declare role token '" + token +
                                    "' (declared: " + roles_csv(variables) + ")");
      return {block, component};
    };

    inputs.reserve(description.in_blocks.size());
    for (std::size_t index = 0; index < description.in_blocks.size(); ++index)
      inputs.push_back(resolve(description.in_blocks[index], description.in_roles[index]));

    outputs.reserve(description.out_blocks.size());
    std::size_t offset = 0;
    const int register_count =
        static_cast<int>(description.in_blocks.size() + description.consts.size());
    for (std::size_t term = 0; term < description.out_blocks.size(); ++term) {
      const CoupledSourceInputReference target =
          resolve(description.out_blocks[term], description.out_roles[term]);
      const int length = description.prog_lens[term];
      if (length < 0 || length > kCsMaxProg || offset > description.prog_ops.size() ||
          static_cast<std::size_t>(length) > description.prog_ops.size() - offset)
        throw std::invalid_argument("System coupled-source term program has an invalid length");
      CsProgram program{};
      program.len = length;
      for (int instruction = 0; instruction < length; ++instruction) {
        const int opcode = description.prog_ops[offset + static_cast<std::size_t>(instruction)];
        const int argument = description.prog_args[offset + static_cast<std::size_t>(instruction)];
        if (opcode < 0 || opcode > static_cast<int>(CsOp::Sqrt) ||
            (opcode == static_cast<int>(CsOp::PushReg) &&
             (argument < 0 || argument >= register_count)))
          throw std::invalid_argument("System coupled-source term contains an invalid instruction");
        program.op[instruction] = opcode;
        program.arg[instruction] = argument;
      }
      validate_cs_program_stack(program, "System::add_coupled_source term " + std::to_string(term));
      outputs.push_back({target.block, target.component, program});
      offset += static_cast<std::size_t>(length);
    }
    if (offset != description.prog_ops.size())
      throw std::invalid_argument("System coupled-source term lengths do not consume the program");

    has_frequency_program = !description.freq_prog_ops.empty();
    if (has_frequency_program) {
      if (description.freq_prog_ops.size() > static_cast<std::size_t>(kCsMaxProg))
        throw std::length_error("System coupled-source frequency program is too long");
      frequency_program.len = static_cast<int>(description.freq_prog_ops.size());
      for (int instruction = 0; instruction < frequency_program.len; ++instruction) {
        const int opcode = description.freq_prog_ops[static_cast<std::size_t>(instruction)];
        const int argument = description.freq_prog_args[static_cast<std::size_t>(instruction)];
        if (opcode < 0 || opcode > static_cast<int>(CsOp::Sqrt) ||
            (opcode == static_cast<int>(CsOp::PushReg) &&
             (argument < 0 || argument >= register_count)))
          throw std::invalid_argument(
              "System coupled-source frequency contains an invalid instruction");
        frequency_program.op[instruction] = opcode;
        frequency_program.arg[instruction] = argument;
      }
      validate_cs_program_stack(frequency_program, "System::add_coupled_source frequency");
    }
    constants.assign(description.consts.begin(), description.consts.end());
    if (std::any_of(constants.begin(), constants.end(),
                    [](Real value) { return !std::isfinite(value); }))
      throw std::invalid_argument("System coupled-source constants must be finite native reals");
    if (!inspect.label.empty()) {
      if (inspect.label != label || inspect.frequency.constant_mu != frequency ||
          inspect.frequency.per_cell != has_frequency_program)
        throw std::invalid_argument(
            "typed System coupling inspect metadata differs from its executable program");
    } else {
      inspect.label = label;
      inspect.frequency.constant_mu = frequency;
      inspect.frequency.per_cell = has_frequency_program;
    }
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("System coupled-source preparation failed collectively");
  }

  ExactContractBuilder contract;
  contract.text("pops.system.coupled-source")
      .scalar(std::uint32_t{1})
      .scalar(std::int32_t{Dim})
      .text(label)
      .scalar(frequency)
      .sequence(description.in_blocks,
                [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
      .sequence(description.in_roles,
                [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
      .sequence(description.consts)
      .sequence(description.out_blocks,
                [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
      .sequence(description.out_roles,
                [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
      .sequence(description.prog_ops)
      .sequence(description.prog_args)
      .sequence(description.prog_lens)
      .sequence(description.freq_prog_ops)
      .sequence(description.freq_prog_args)
      .sequence(inspect.conservation.conserved_roles,
                [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
      .sequence(inspect.conservation.created_roles,
                [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
      .scalar(static_cast<std::uint8_t>(inspect.frequency.per_cell ? 1 : 0));
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("system-coupled-source"), contract.view()}}))
    throw std::invalid_argument("System coupled-source descriptions differ between MPI ranks");

  const int input_count = static_cast<int>(inputs.size());
  const int constant_count = static_cast<int>(constants.size());
  const int output_count = static_cast<int>(outputs.size());
  auto operation = [inputs, outputs, constants, input_count, constant_count, output_count](
                       Real dt, const std::vector<MultiFab<Dim>*>& states) {
    const int reference_block = input_count != 0 ? inputs.front().block : outputs.front().block;
    MultiFab<Dim>& reference = *states[static_cast<std::size_t>(reference_block)];
    for (const CoupledSourceInputReference& input : inputs)
      if (states[static_cast<std::size_t>(input.block)]->local_global_indices() !=
          reference.local_global_indices())
        throw std::invalid_argument("System coupled-source input layouts are not co-located");
    for (const OutputRef& output : outputs)
      if (states[static_cast<std::size_t>(output.block)]->local_global_indices() !=
          reference.local_global_indices())
        throw std::invalid_argument("System coupled-source output layouts are not co-located");
    for (std::size_t local = 0; local < reference.local_size(); ++local) {
      CoupledSourceKernel<Dim> kernel{};
      kernel.dt = dt;
      kernel.n_in = input_count;
      kernel.n_const = constant_count;
      kernel.n_terms = output_count;
      for (int input = 0; input < input_count; ++input) {
        const CoupledSourceInputReference& ref = inputs[static_cast<std::size_t>(input)];
        const Fab<Dim>& fab = states[static_cast<std::size_t>(ref.block)]->fab(local);
        kernel.in[input] = fab.view();
        kernel.in_comp[input] = ref.component;
      }
      for (int constant = 0; constant < constant_count; ++constant)
        kernel.consts[constant] = constants[static_cast<std::size_t>(constant)];
      for (int output = 0; output < output_count; ++output) {
        const OutputRef& ref = outputs[static_cast<std::size_t>(output)];
        kernel.out[output] = states[static_cast<std::size_t>(ref.block)]->fab(local).view();
        kernel.out_comp[output] = ref.component;
        kernel.prog[output] = ref.program;
      }
      for_each_cell(reference.box(local), kernel);
    }
  };

  std::function<Real()> maximum_frequency;
  if (has_frequency_program) {
    Impl* implementation = p_.get();
    maximum_frequency = make_coupled_source_maximum_frequency<Dim>(
        inputs, constants, frequency_program, input_count, constant_count, lane,
        [implementation](int block) -> const MultiFab<Dim>& {
          return implementation->sp[static_cast<std::size_t>(block)].U;
        });
  }
  install_prepared_coupling_operator(label, std::string(contract.view()), std::move(inspect),
                                     std::move(operation), frequency, std::move(maximum_frequency));
}

template <int Dim>
void System<Dim>::add_coupled_source(const CoupledSourceProgram& description, double frequency,
                                     const std::string& label) {
  add_coupled_source_prepared_(description, frequency, label, {});
}

template <int Dim>
void System<Dim>::add_coupling_operator(const CouplingOperator& op) {
  validate_coupling_contract(op, "System::add_coupling_operator");
  add_coupled_source_prepared_(op.program, op.frequency.constant_mu, op.label,
                               CouplingOperatorView{op.label, op.conservation, op.frequency});
}

template <int Dim>
void System<Dim>::install_prepared_coupling_operator(
    const std::string& label, const std::string& provider_contract, CouplingOperatorView view,
    std::function<void(Real, const std::vector<MultiFab<Dim>*>&)> operation,
    double constant_frequency, std::function<Real()> maximum_frequency) {
  const ExecutionLane& lane = prepared_boundary_execution_lane();
  std::exception_ptr local_error;
  runtime::system::SystemCouplingRegistry<Dim> candidate;
  std::string exact;
  try {
    require_assembling(p_->lifecycle_, "install_prepared_coupling_operator");
    if (label.empty() || provider_contract.empty() || !operation ||
        !std::isfinite(constant_frequency) || constant_frequency < 0.0)
      throw std::invalid_argument(
          "prepared System coupling requires exact provider identity, executable, and finite "
          "frequency");
    if (!view.label.empty() && view.label != label)
      throw std::invalid_argument(
          "prepared System coupling inspect identity differs from its executable provider");
    if (p_->coupling_.operator_contracts.size() != p_->coupling_.operators.size() ||
        p_->coupling_.operators.size() != p_->coupling_.coupled_operators.size())
      throw std::logic_error("prepared System coupling registry is inconsistent");
    view.label = label;
    view.frequency.constant_mu = constant_frequency;
    view.frequency.per_cell = static_cast<bool>(maximum_frequency);

    ExactContractBuilder contract;
    contract.text("pops.system.prepared-coupling")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(label)
        .bytes(provider_contract)
        .scalar(constant_frequency)
        .scalar(static_cast<std::uint8_t>(maximum_frequency ? 1 : 0))
        .sequence(view.conservation.conserved_roles,
                  [](ExactContractBuilder& item, const std::string& role) { item.text(role); })
        .sequence(view.conservation.created_roles,
                  [](ExactContractBuilder& item, const std::string& role) { item.text(role); })
        .scalar(static_cast<std::uint64_t>(p_->sp.size()));
    for (const typename Impl::Species& block : p_->sp) {
      if (block.name.empty() || block.state_identity.empty())
        throw std::invalid_argument(
            "prepared System coupling block map lacks an exact state identity");
      contract.text(block.name)
          .text(block.state_identity)
          .scalar(block.ncomp)
          .scalar(static_cast<std::uint64_t>(block.cons_vars.roles.size()));
      for (const VariableSemantic role : block.cons_vars.roles)
        contract.scalar(static_cast<std::uint8_t>(role.kind)).scalar(role.axis);
      contract.sequence(
          block.cons_vars.user_roles,
          [](ExactContractBuilder& item, const std::string& role) { item.text(role); });
    }
    exact = std::move(contract).release();

    // Copy and allocate the complete candidate registry before consensus.  A rank-local allocation
    // failure therefore cannot publish a shorter provider list on another rank.
    candidate = p_->coupling_;
    candidate.operators.push_back(std::move(operation));
    candidate.operator_contracts.push_back(exact);
    candidate.coupled_operators.push_back(std::move(view));
    if (constant_frequency > 0.0)
      candidate.coupled_freqs.push_back({label, constant_frequency});
    if (maximum_frequency)
      candidate.coupled_frequencies.push_back({label, std::move(maximum_frequency)});
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("prepared System coupling installation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("system-prepared-coupling"), exact}}, lane))
    throw std::invalid_argument("prepared System coupling contract differs between MPI ranks");
  p_->coupling_ = std::move(candidate);
}

template <int Dim>
const std::vector<CouplingOperatorView>& System<Dim>::coupled_operators() const {
  return p_->coupling_.coupled_operators;
}

template void System<kNativeDimension>::add_block(const std::string&, const ModelSpec&,
                                                  const std::string&, const std::string&,
                                                  const std::string&, const std::string&, int, bool,
                                                  int, const std::vector<std::string>&,
                                                  const std::vector<std::string>&,
                                                  const NewtonOptions&, bool, double, bool, double);
template void System<kNativeDimension>::install_block_state_route(const std::string&,
                                                                  const std::string&);
template void System<kNativeDimension>::install_field_storage_route(const std::string&,
                                                                    const std::string&);
template void System<kNativeDimension>::install_prepared_boundary_execution_lane(
    std::shared_ptr<ExecutionLane>);
template void System<kNativeDimension>::stage_prepared_ghost_boundary_component(
    const std::string&, std::shared_ptr<PreparedGhostBoundaryComponent>);
template void System<kNativeDimension>::stage_prepared_boundary_flux_component(
    const std::string&, std::shared_ptr<PreparedBoundaryFluxComponent>);
template void System<kNativeDimension>::stage_prepared_field_boundary_component_pair(
    const std::string&, std::shared_ptr<PreparedFieldBoundaryResidualComponent>,
    std::shared_ptr<PreparedFieldBoundaryJvpComponent>);
template void System<kNativeDimension>::install_hyperbolic_boundary(
    const std::string&, const std::string&, int, const std::vector<std::string>&,
    const std::vector<double>&, const std::vector<std::string>&, const std::vector<std::string>&,
    const std::string&, const std::vector<std::string>&, const std::vector<std::string>&,
    const std::vector<std::vector<std::string>>&, const std::vector<std::vector<double>>&,
    const std::vector<std::string>&);
template void System<kNativeDimension>::install_prepared_hyperbolic_boundary(
    const std::string&, const std::string&, int, const std::string&,
    std::shared_ptr<const System<kNativeDimension>::HyperbolicBoundary>);
template void System<kNativeDimension>::discard_hyperbolic_boundaries();
template void System<kNativeDimension>::install_interface_provider(
    SystemInterfaceProvider<kNativeDimension>);
template void System<kNativeDimension>::discard_interface_flux_components();
template std::size_t System<kNativeDimension>::interface_evaluation_count(const std::string&,
                                                                          int) const;
template Geometry<kNativeDimension> System<kNativeDimension>::prepared_block_geometry() const;
template std::array<bool, kNativeDimension> System<kNativeDimension>::prepared_block_periodicity()
    const;
template void System<kNativeDimension>::install_prepared_block(
    PreparedSystemBlock<kNativeDimension>);
template void System<kNativeDimension>::register_elliptic_field(
    const std::string&, const std::string&,
    const std::vector<runtime::system::AuxiliaryComponentKey>&, int);
template void System<kNativeDimension>::set_block_elliptic_field(
    const std::string&, const std::string&,
    std::function<void(const MultiFab<kNativeDimension>&, MultiFab<kNativeDimension>&)>);
template void System<kNativeDimension>::add_dt_bound(const std::string&, std::function<double()>);
template std::string System<kNativeDimension>::last_dt_bound() const;
template void System<kNativeDimension>::register_native_package(
    const std::string&, const std::string&, const std::string&, const std::string&,
    const std::string&, const std::string&, const std::string&, const std::string&, double, int,
    bool, int, const std::vector<double>&, double, NewtonOptions, bool);
template void System<kNativeDimension>::register_external_riemann_package(
    const std::string&, const std::string&, const std::string&, const std::string&, int, int,
    const std::string&, const std::string&, const std::string&, const std::string&,
    const std::string&, double, int, bool, int, double, double);
template void System<kNativeDimension>::stage_prepared_native_package(
    std::string, std::function<void()>, std::function<void()>, std::shared_ptr<void>,
    std::shared_ptr<runtime::system::NativePackageCapabilityState<kNativeDimension>>);
template void System<kNativeDimension>::finalize_native_packages();
template void System<kNativeDimension>::add_coupled_source(const CoupledSourceProgram&, double,
                                                           const std::string&);
template void System<kNativeDimension>::add_coupling_operator(const CouplingOperator&);
template void System<kNativeDimension>::install_prepared_coupling_operator(
    const std::string&, const std::string&, CouplingOperatorView,
    std::function<void(Real, const std::vector<MultiFab<kNativeDimension>*>&)>, double,
    std::function<Real()>);
template const std::vector<CouplingOperatorView>& System<kNativeDimension>::coupled_operators()
    const;

}  // namespace pops
