#pragma once

#include <pops/mesh/boundary/prepared_boundary_component.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/runtime/multiblock/interface_flux_scheduler.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::detail {

struct BoundaryPointLocation {
  int local_fab = 0;
  int i = 0;
  int j = 0;
};

/// Reusable, exact storage table for a prepared boundary session.
///
/// Identities are installed while the session is materialized.  A hot binding starts a new epoch
/// and only replaces pointers in those stable slots; it does not allocate nodes, copy strings or
/// resolve a Handle through a process-global table.  Executor workspaces cache the resulting slot
/// ordinals, so native invocations do not perform string lookup either.
class BoundaryFieldRegistry {
 public:
  void configure_states(const std::vector<std::string>& identities) {
    configure_(states_, identities, "state");
  }
  void configure_directions(const std::vector<std::string>& identities) {
    configure_(directions_, identities, "direction");
  }
  void configure_fields(const std::vector<std::string>& identities) {
    configure_(fields_, identities, "field");
  }
  void configure_outputs(const std::vector<std::string>& identities) {
    configure_(outputs_, identities, "output");
  }

  void begin_binding() noexcept {
    if (epoch_ == std::numeric_limits<std::uint64_t>::max()) {
      clear_epochs_(states_);
      clear_epochs_(directions_);
      clear_epochs_(fields_);
      clear_epochs_(outputs_);
      epoch_ = 1;
    } else {
      ++epoch_;
    }
  }

  void bind_state(std::string_view identity, const MultiFab& field) {
    bind_const_(states_, identity, field, "state");
  }
  void bind_direction(std::string_view identity, const MultiFab& field) {
    bind_const_(directions_, identity, field, "direction");
  }
  void bind_field(std::string_view identity, const MultiFab& field) {
    bind_const_(fields_, identity, field, "field");
  }
  void bind_output(std::string_view identity, MultiFab& field) {
    if (identity.empty())
      throw std::invalid_argument("boundary output identity is empty");
    std::size_t slot = find_(outputs_, identity);
    if (slot == outputs_.size()) {
      // Control/preparation adapters may discover an output while materializing a session.  The
      // production hot path always uses bind_output_slot() against preconfigured identities.
      outputs_.push_back({std::string(identity), nullptr, 0});
      slot = outputs_.size() - 1;
    }
    bind_output_(slot, field);
  }

  void bind_state_slot(std::size_t slot, const MultiFab& field) {
    bind_const_slot_(states_, slot, field, "state");
  }
  void bind_direction_slot(std::size_t slot, const MultiFab& field) {
    bind_const_slot_(directions_, slot, field, "direction");
  }
  void bind_field_slot(std::size_t slot, const MultiFab& field) {
    bind_const_slot_(fields_, slot, field, "field");
  }
  void bind_output_slot(std::size_t slot, MultiFab& field) { bind_output_(slot, field); }

  [[nodiscard]] std::size_t state_index(std::string_view identity) const {
    return index_(states_, identity, "state");
  }
  [[nodiscard]] std::size_t direction_index(std::string_view identity) const {
    return index_(directions_, identity, "direction");
  }
  [[nodiscard]] std::size_t field_index(std::string_view identity) const {
    return index_(fields_, identity, "field");
  }
  [[nodiscard]] std::size_t output_index(std::string_view identity) const {
    return index_(outputs_, identity, "output");
  }

  [[nodiscard]] const MultiFab& state_at(std::size_t slot) const {
    return require_const_(states_, slot, "state");
  }
  [[nodiscard]] const MultiFab& direction_at(std::size_t slot) const {
    return require_const_(directions_, slot, "direction");
  }
  [[nodiscard]] const MultiFab& field_at(std::size_t slot) const {
    return require_const_(fields_, slot, "field");
  }
  [[nodiscard]] MultiFab& output_at(std::size_t slot) const {
    if (slot >= outputs_.size() || outputs_[slot].epoch != epoch_ ||
        outputs_[slot].value == nullptr)
      throw std::invalid_argument("boundary component has no exact mutable output slot binding");
    return *outputs_[slot].value;
  }

  [[nodiscard]] const MultiFab& state(std::string_view identity) const {
    return state_at(state_index(identity));
  }
  [[nodiscard]] const MultiFab& direction(std::string_view identity) const {
    return direction_at(direction_index(identity));
  }
  [[nodiscard]] const MultiFab& field(std::string_view identity) const {
    return field_at(field_index(identity));
  }
  [[nodiscard]] MultiFab& output(std::string_view identity) const {
    return output_at(output_index(identity));
  }

 private:
  template <class Pointer>
  struct Entry {
    std::string identity;
    Pointer value = nullptr;
    std::uint64_t epoch = 0;
  };
  using ConstEntries = std::vector<Entry<const MultiFab*>>;
  using MutableEntries = std::vector<Entry<MultiFab*>>;

  ConstEntries states_;
  ConstEntries directions_;
  ConstEntries fields_;
  MutableEntries outputs_;
  std::uint64_t epoch_ = 1;

  template <class Entries>
  static void clear_epochs_(Entries& entries) noexcept {
    for (auto& entry : entries)
      entry.epoch = 0;
  }

  template <class Entries>
  static std::size_t find_(const Entries& entries, std::string_view identity) noexcept {
    for (std::size_t slot = 0; slot < entries.size(); ++slot)
      if (entries[slot].identity == identity)
        return slot;
    return entries.size();
  }

  template <class Entries>
  static std::size_t index_(const Entries& entries, std::string_view identity, const char* role) {
    const std::size_t slot = find_(entries, identity);
    if (identity.empty() || slot == entries.size())
      throw std::invalid_argument(std::string("boundary component has no prepared ") + role +
                                  " slot for '" + std::string(identity) + "'");
    return slot;
  }

  template <class Entries>
  static void configure_(Entries& entries, const std::vector<std::string>& identities,
                         const char* role) {
    entries.reserve(entries.size() + identities.size());
    for (const std::string& identity : identities) {
      if (identity.empty())
        throw std::invalid_argument(std::string("boundary ") + role + " identity is empty");
      if (find_(entries, identity) == entries.size())
        entries.push_back({identity, nullptr, 0});
    }
  }

  static void bind_const_slot_(ConstEntries& entries, std::size_t slot, const MultiFab& field,
                               const char* role, std::uint64_t epoch) {
    if (slot >= entries.size())
      throw std::out_of_range(std::string("boundary ") + role + " slot is out of range");
    if (entries[slot].epoch == epoch)
      throw std::invalid_argument(std::string("boundary ") + role + " slot is multiply bound");
    entries[slot].value = &field;
    entries[slot].epoch = epoch;
  }

  void bind_const_slot_(ConstEntries& entries, std::size_t slot, const MultiFab& field,
                        const char* role) {
    bind_const_slot_(entries, slot, field, role, epoch_);
  }

  void bind_const_(ConstEntries& entries, std::string_view identity, const MultiFab& field,
                   const char* role) {
    std::size_t slot = find_(entries, identity);
    if (identity.empty())
      throw std::invalid_argument(std::string("boundary ") + role + " identity is empty");
    if (slot == entries.size()) {
      entries.push_back({std::string(identity), nullptr, 0});
      slot = entries.size() - 1;
    }
    bind_const_slot_(entries, slot, field, role);
  }

  void bind_output_(std::size_t slot, MultiFab& field) {
    if (slot >= outputs_.size())
      throw std::out_of_range("boundary output slot is out of range");
    if (outputs_[slot].epoch == epoch_)
      throw std::invalid_argument("boundary output slot is multiply bound");
    for (std::size_t other = 0; other < outputs_.size(); ++other)
      if (other != slot && outputs_[other].epoch == epoch_ && outputs_[other].value == &field)
        throw std::invalid_argument(
            "distinct boundary output identities may not alias one mutable field");
    outputs_[slot].value = &field;
    outputs_[slot].epoch = epoch_;
  }

  const MultiFab& require_const_(const ConstEntries& entries, std::size_t slot,
                                 const char* role) const {
    if (slot >= entries.size() || entries[slot].epoch != epoch_ || entries[slot].value == nullptr)
      throw std::invalid_argument(std::string("boundary component has no exact ") + role +
                                  " slot binding");
    return *entries[slot].value;
  }
};

inline bool region_coordinate_matches(int coordinate, int axis, int side, const Box2D& domain,
                                      bool ghosts, int depth) {
  const int lower = domain.lo[axis];
  const int upper = domain.hi[axis];
  if (!ghosts)
    return coordinate == (side < 0 ? lower : upper);
  if (side < 0)
    return coordinate < lower && coordinate >= lower - depth;
  return coordinate > upper && coordinate <= upper + depth;
}

inline std::vector<BoundaryPointLocation> boundary_locations(const MultiFab& field,
                                                             const Box2D& domain,
                                                             const PreparedBoundaryRegion& region,
                                                             int depth, bool ghosts) {
  if (region.dimension != 2 || region.codimension < 1 || region.codimension > 2 ||
      region.axes.size() != static_cast<std::size_t>(region.codimension) ||
      region.sides.size() != region.axes.size())
    throw std::invalid_argument("boundary component executor requires an exact 2D region");
  int side_for_axis[2] = {0, 0};
  for (std::size_t index = 0; index < region.axes.size(); ++index)
    side_for_axis[static_cast<std::size_t>(region.axes[index])] = region.sides[index];

  std::vector<BoundaryPointLocation> locations;
  for (int li = 0; li < field.local_size(); ++li) {
    const Box2D valid = field.box(li);
    const int grow = ghosts ? std::min(depth, field.n_grow()) : 0;
    const int ilo = valid.lo[0] - (side_for_axis[0] == 0 ? 0 : grow);
    const int ihi = valid.hi[0] + (side_for_axis[0] == 0 ? 0 : grow);
    const int jlo = valid.lo[1] - (side_for_axis[1] == 0 ? 0 : grow);
    const int jhi = valid.hi[1] + (side_for_axis[1] == 0 ? 0 : grow);
    for (int j = jlo; j <= jhi; ++j) {
      for (int i = ilo; i <= ihi; ++i) {
        const int coordinates[2] = {i, j};
        bool selected = true;
        for (int axis = 0; axis < 2; ++axis) {
          const int side = side_for_axis[axis];
          selected =
              selected && (side == 0 ? coordinates[axis] >= domain.lo[axis] &&
                                           coordinates[axis] <= domain.hi[axis]
                                     : region_coordinate_matches(coordinates[axis], axis, side,
                                                                 domain, ghosts, depth));
        }
        if (selected)
          locations.push_back({li, i, j});
      }
    }
  }
  return locations;
}

inline void validate_layout(const MultiFab& field, const MultiFab& layout_reference) {
  if (field.box_array().boxes() != layout_reference.box_array().boxes() ||
      field.dmap().ranks() != layout_reference.dmap().ranks() ||
      field.local_size() != layout_reference.local_size())
    throw std::runtime_error(
        "boundary component qualified field differs from the prepared "
        "BoxArray/DistributionMapping");
}

inline void pack_field_into(const MultiFab& field,
                            const std::vector<BoundaryPointLocation>& locations,
                            const Box2D& domain, bool clamp_to_domain,
                            const MultiFab& layout_reference, std::vector<double>& packed) {
  validate_layout(field, layout_reference);
  const std::size_t required = locations.size() * static_cast<std::size_t>(field.ncomp());
  if (packed.size() != required)
    throw std::runtime_error("boundary component field shape changed after session preparation");
  // PreparedBoundaryComponent::make_session has already proved a HOST execution context.  Kokkos
  // Serial/OpenMP launches are complete when control returns, so a per-component residency fence
  // here would only serialize every residual/JVP application.  Device/managed contexts are refused
  // at preparation until a device-native provider ABI exists.
  // Packing is an irregular O(boundary-measure) gather, normally below the repository's small-box
  // launch threshold; keep it serial instead of paying one host Kokkos launch per qualified field.
  for (std::size_t point = 0; point < locations.size(); ++point) {
    const auto& location = locations[point];
    if (location.local_fab < 0 || location.local_fab >= field.local_size() ||
        field.box(location.local_fab) != layout_reference.box(location.local_fab))
      throw std::runtime_error(
          "boundary component local Fab ordering differs from the prepared state layout");
    const int i = clamp_to_domain ? std::clamp(location.i, domain.lo[0], domain.hi[0]) : location.i;
    const int j = clamp_to_domain ? std::clamp(location.j, domain.lo[1], domain.hi[1]) : location.j;
    const ConstArray4 values = field.fab(location.local_fab).const_array();
    for (int component = 0; component < field.ncomp(); ++component)
      packed[point * static_cast<std::size_t>(field.ncomp()) +
             static_cast<std::size_t>(component)] = values(i, j, component);
  }
}

inline void scatter_field(MultiFab& field, const std::vector<BoundaryPointLocation>& locations,
                          const std::vector<double>& packed) {
  if (packed.size() != locations.size() * static_cast<std::size_t>(field.ncomp()))
    throw std::runtime_error("boundary component output shape changed across native ABI call");
  for (std::size_t point = 0; point < locations.size(); ++point) {
    const auto& location = locations[point];
    Array4 values = field.fab(location.local_fab).array();
    for (int component = 0; component < field.ncomp(); ++component)
      values(location.i, location.j, component) =
          static_cast<Real>(packed[point * static_cast<std::size_t>(field.ncomp()) +
                                   static_cast<std::size_t>(component)]);
  }
}

inline PopsConstFieldViewV1 const_view(const std::vector<double>& values, std::size_t points,
                                       std::size_t components, const std::string& layout_identity,
                                       const std::string& patch_identity) {
  return {sizeof(PopsConstFieldViewV1),
          values.data(),
          2,
          {points, 1, 1},
          {static_cast<std::ptrdiff_t>(components), static_cast<std::ptrdiff_t>(components), 0},
          components,
          1,
          POPS_FIELD_CENTERING_CELL_V1,
          0,
          {0, 0, 0},
          {0, 0, 0},
          POPS_SCALAR_FLOAT64_V1,
          POPS_MEMORY_SPACE_HOST_V1,
          layout_identity.c_str(),
          patch_identity.c_str(),
          POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
}

inline PopsFieldViewV1 field_view(std::vector<double>& values, std::size_t points,
                                  std::size_t components, const std::string& layout_identity,
                                  const std::string& patch_identity) {
  return {sizeof(PopsFieldViewV1),
          values.data(),
          2,
          {points, 1, 1},
          {static_cast<std::ptrdiff_t>(components), static_cast<std::ptrdiff_t>(components), 0},
          components,
          1,
          POPS_FIELD_CENTERING_CELL_V1,
          0,
          {0, 0, 0},
          {0, 0, 0},
          POPS_SCALAR_FLOAT64_V1,
          POPS_MEMORY_SPACE_HOST_V1,
          layout_identity.c_str(),
          patch_identity.c_str(),
          POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
}

inline PopsLogicalTimeV1 logical_time(const runtime::multiblock::BoundaryEvaluationPoint& point) {
  return {sizeof(PopsLogicalTimeV1),
          point.clock.c_str(),
          point.tick,
          point.level,
          point.substep,
          point.stage,
          point.stage_fraction.numerator,
          point.stage_fraction.denominator,
          point.dt,
          point.physical_time};
}

inline std::vector<double> coordinates(const std::vector<BoundaryPointLocation>& locations,
                                       const Geometry& geometry) {
  std::vector<double> result(locations.size() * 2u);
  for (std::size_t point = 0; point < locations.size(); ++point) {
    result[2u * point] = geometry.x_cell(locations[point].i);
    result[2u * point + 1u] = geometry.y_cell(locations[point].j);
  }
  return result;
}

inline std::vector<PopsQualifiedScalarV1> scalar_table(const PreparedBoundaryComponentSpec& spec) {
  std::vector<PopsQualifiedScalarV1> result;
  result.reserve(spec.parameter_ids.size());
  for (std::size_t index = 0; index < spec.parameter_ids.size(); ++index)
    result.push_back({sizeof(PopsQualifiedScalarV1), spec.parameter_ids[index].c_str(),
                      spec.parameter_values[index]});
  return result;
}

enum class BoundaryConstRole { State, Direction, Field };

struct PackedConstBoundaryField {
  BoundaryConstRole role = BoundaryConstRole::State;
  std::size_t slot = 0;
  std::size_t components = 0;
  std::vector<double> values;
  PopsQualifiedConstFieldV1 view{};
};

struct PackedMutableBoundaryField {
  std::size_t slot = 0;
  std::size_t components = 0;
  std::vector<double> values;
  PopsQualifiedFieldV1 view{};
};

inline const MultiFab& bound_const(const BoundaryFieldRegistry& registry, BoundaryConstRole role,
                                   std::size_t slot) {
  switch (role) {
    case BoundaryConstRole::State:
      return registry.state_at(slot);
    case BoundaryConstRole::Direction:
      return registry.direction_at(slot);
    case BoundaryConstRole::Field:
      return registry.field_at(slot);
  }
  throw std::logic_error("invalid prepared boundary field role");
}

inline PackedConstBoundaryField prepare_const_field(BoundaryConstRole role, std::size_t slot,
                                                    const std::string& identity,
                                                    const BoundaryFieldRegistry& registry,
                                                    std::size_t count, const std::string& layout,
                                                    const std::string& patch) {
  const MultiFab& field = bound_const(registry, role, slot);
  PackedConstBoundaryField row;
  row.role = role;
  row.slot = slot;
  row.components = static_cast<std::size_t>(field.ncomp());
  row.values.resize(count * row.components);
  row.view = {sizeof(PopsQualifiedConstFieldV1), 1u, identity.c_str(),
              const_view(row.values, count, row.components, layout, patch)};
  return row;
}

struct PreparedGhostBoundaryWorkspace {
  std::vector<BoundaryPointLocation> locations;
  std::vector<double> interior;
  std::vector<double> ghosts;
  std::vector<double> xy;
  std::vector<PackedConstBoundaryField> packed_dependencies;
  std::vector<PopsQualifiedConstFieldV1> dependencies;
  std::vector<PopsQualifiedScalarV1> parameters;
  int state_components = 0;
};

inline PreparedGhostBoundaryWorkspace prepare_ghost_workspace(
    const PreparedGhostBoundaryComponent::Session& component, const MultiFab& prototype,
    const BoundaryFieldRegistry& registry, const Geometry& geometry, int depth) {
  const auto& spec = component.spec();
  PreparedGhostBoundaryWorkspace workspace;
  workspace.locations = boundary_locations(prototype, geometry.domain, spec.region, depth, true);
  workspace.state_components = prototype.ncomp();
  const std::size_t count = workspace.locations.size();
  workspace.interior.resize(count * static_cast<std::size_t>(prototype.ncomp()));
  workspace.ghosts.resize(count * static_cast<std::size_t>(prototype.ncomp()));
  workspace.xy = coordinates(workspace.locations, geometry);
  workspace.packed_dependencies.reserve(spec.states.size() + spec.fields.size());
  for (const std::string& identity : spec.states)
    workspace.packed_dependencies.push_back(
        prepare_const_field(BoundaryConstRole::State, registry.state_index(identity), identity,
                            registry, count, spec.layout_identity, spec.region.identity));
  for (const std::string& identity : spec.fields)
    workspace.packed_dependencies.push_back(
        prepare_const_field(BoundaryConstRole::Field, registry.field_index(identity), identity,
                            registry, count, spec.layout_identity, spec.region.identity));
  workspace.dependencies.reserve(workspace.packed_dependencies.size());
  for (const auto& row : workspace.packed_dependencies)
    workspace.dependencies.push_back(row.view);
  workspace.parameters = scalar_table(spec);
  return workspace;
}

inline void apply_ghost_component(const PreparedGhostBoundaryComponent::Session& component,
                                  PreparedGhostBoundaryWorkspace& workspace, MultiFab& state,
                                  const BoundaryFieldRegistry& registry, const Geometry& geometry,
                                  const runtime::multiblock::BoundaryEvaluationPoint& point) {
  const auto& spec = component.spec();
  if (state.ncomp() != workspace.state_components)
    throw std::runtime_error("boundary ghost state component count changed after preparation");
  if (workspace.locations.empty())
    return;
  const std::size_t count = workspace.locations.size();
  pack_field_into(state, workspace.locations, geometry.domain, true, state, workspace.interior);
  pack_field_into(state, workspace.locations, geometry.domain, false, state, workspace.ghosts);
  for (auto& row : workspace.packed_dependencies)
    pack_field_into(bound_const(registry, row.role, row.slot), workspace.locations, geometry.domain,
                    true, state, row.values);
  const PopsConstFieldViewV1 interior_view =
      const_view(workspace.interior, count, static_cast<std::size_t>(state.ncomp()),
                 spec.layout_identity, spec.region.identity);
  const PopsConstFieldViewV1 coordinate_view =
      const_view(workspace.xy, count, 2, spec.layout_identity, spec.region.identity);
  const PopsFieldViewV1 ghost_view =
      field_view(workspace.ghosts, count, static_cast<std::size_t>(state.ncomp()),
                 spec.layout_identity, spec.region.identity);
  PopsGhostBoundaryRequestV1 request{sizeof(PopsGhostBoundaryRequestV1),
                                     spec.producer_identity.c_str(),
                                     spec.state_identity.c_str(),
                                     spec.ghost_identity.c_str(),
                                     interior_view,
                                     ghost_view,
                                     coordinate_view,
                                     spec.region.view(),
                                     workspace.dependencies.size(),
                                     workspace.dependencies.data(),
                                     workspace.parameters.size(),
                                     workspace.parameters.data(),
                                     logical_time(point),
                                     component.execution().view()};
  PopsComponentStatusV1 status{sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1,
                               nullptr};
  const int code =
      component::apply_ghost_boundary(component.ghost_api(), component.state(), request, status);
  PreparedGhostBoundaryComponent::require_success(code, status, "apply_region_batch");
  scatter_field(state, workspace.locations, workspace.ghosts);
}

struct PreparedFieldBoundaryWorkspace {
  std::vector<BoundaryPointLocation> locations;
  std::vector<double> xy;
  std::size_t layout_state_slot = 0;
  std::vector<PackedConstBoundaryField> packed_states;
  std::vector<PackedConstBoundaryField> packed_directions;
  std::vector<PackedConstBoundaryField> packed_fields;
  std::vector<PopsQualifiedConstFieldV1> states;
  std::vector<PopsQualifiedConstFieldV1> directions;
  std::vector<PopsQualifiedConstFieldV1> fields;
  std::vector<PackedMutableBoundaryField> packed_outputs;
  std::vector<PopsQualifiedFieldV1> outputs;
  std::vector<PopsQualifiedScalarV1> parameters;
  PopsQualifiedConstFieldV1 rate{};
  PopsQualifiedConstFieldV1 nonlinear{};
};

template <PreparedBoundaryOperation Operation>
inline PreparedFieldBoundaryWorkspace prepare_field_workspace(
    const typename PreparedBoundaryComponent<Operation>::Session& component,
    const BoundaryFieldRegistry& registry, const Geometry& geometry) {
  static_assert(Operation == PreparedBoundaryOperation::FieldResidual ||
                Operation == PreparedBoundaryOperation::FieldJvp);
  const auto& spec = component.spec();
  PreparedFieldBoundaryWorkspace workspace;
  workspace.layout_state_slot = registry.state_index(spec.states.front());
  const MultiFab& layout_reference = registry.state_at(workspace.layout_state_slot);
  workspace.locations =
      boundary_locations(layout_reference, geometry.domain, spec.region, 1, false);
  const std::size_t count = workspace.locations.size();
  workspace.xy = coordinates(workspace.locations, geometry);

  auto prepare_rows = [&](const std::vector<std::string>& identities, BoundaryConstRole role,
                          auto index) {
    std::vector<PackedConstBoundaryField> rows;
    rows.reserve(identities.size());
    for (const std::string& identity : identities)
      rows.push_back(prepare_const_field(role, index(identity), identity, registry, count,
                                         spec.layout_identity, spec.region.identity));
    return rows;
  };
  workspace.packed_states = prepare_rows(
      spec.states, BoundaryConstRole::State,
      [&registry](const std::string& identity) { return registry.state_index(identity); });
  workspace.packed_directions = prepare_rows(
      spec.directions, BoundaryConstRole::Direction,
      [&registry](const std::string& identity) { return registry.direction_index(identity); });
  workspace.packed_fields = prepare_rows(
      spec.fields, BoundaryConstRole::Field,
      [&registry](const std::string& identity) { return registry.field_index(identity); });
  const auto copy_views = [](const auto& rows) {
    std::vector<PopsQualifiedConstFieldV1> views;
    views.reserve(rows.size());
    for (const auto& row : rows)
      views.push_back(row.view);
    return views;
  };
  workspace.states = copy_views(workspace.packed_states);
  workspace.directions = copy_views(workspace.packed_directions);
  workspace.fields = copy_views(workspace.packed_fields);

  workspace.packed_outputs.reserve(spec.outputs.size());
  for (const std::string& identity : spec.outputs) {
    const std::size_t slot = registry.output_index(identity);
    MultiFab& destination = registry.output_at(slot);
    PackedMutableBoundaryField row;
    row.slot = slot;
    row.components = static_cast<std::size_t>(destination.ncomp());
    row.values.resize(count * row.components);
    row.view = {
        sizeof(PopsQualifiedFieldV1), identity.c_str(),
        field_view(row.values, count, row.components, spec.layout_identity, spec.region.identity)};
    workspace.packed_outputs.push_back(std::move(row));
  }
  workspace.outputs.reserve(workspace.packed_outputs.size());
  for (const auto& row : workspace.packed_outputs)
    workspace.outputs.push_back(row.view);
  workspace.parameters = scalar_table(spec);

  const auto optional = [&](const std::string& identity) {
    if (identity.empty())
      return PopsQualifiedConstFieldV1{sizeof(PopsQualifiedConstFieldV1), 0u, nullptr, {}};
    PopsConstFieldViewV1 view{};
    bool found = false;
    for (const auto* table : {&workspace.states, &workspace.directions, &workspace.fields}) {
      for (const auto& row : *table) {
        if (row.present != 0u && std::string_view(row.qualified_id) == identity) {
          view = row.values;
          found = true;
          break;
        }
      }
      if (found)
        break;
    }
    if (!found) {
      for (std::size_t index = 0; index < workspace.outputs.size(); ++index) {
        if (std::string_view(workspace.outputs[index].qualified_id) == identity) {
          const auto& row = workspace.packed_outputs[index];
          view = const_view(row.values, count, row.components, spec.layout_identity,
                            spec.region.identity);
          found = true;
          break;
        }
      }
    }
    if (!found)
      throw std::invalid_argument(
          "boundary optional field has no exact qualified storage binding for '" + identity + "'");
    return PopsQualifiedConstFieldV1{sizeof(PopsQualifiedConstFieldV1), 1u, identity.c_str(), view};
  };
  workspace.rate = optional(spec.rate);
  workspace.nonlinear = optional(spec.nonlinear_iterate);
  return workspace;
}

template <PreparedBoundaryOperation Operation>
inline void apply_field_component(
    const typename PreparedBoundaryComponent<Operation>::Session& component,
    PreparedFieldBoundaryWorkspace& workspace, const BoundaryFieldRegistry& registry,
    const Geometry& geometry, const runtime::multiblock::BoundaryEvaluationPoint& point) {
  static_assert(Operation == PreparedBoundaryOperation::FieldResidual ||
                Operation == PreparedBoundaryOperation::FieldJvp);
  constexpr bool jvp = Operation == PreparedBoundaryOperation::FieldJvp;
  const auto& spec = component.spec();
  const MultiFab& layout_reference = registry.state_at(workspace.layout_state_slot);
  if (workspace.locations.empty())
    return;
  const auto pack_rows = [&](auto& rows) {
    for (auto& row : rows)
      pack_field_into(bound_const(registry, row.role, row.slot), workspace.locations,
                      geometry.domain, false, layout_reference, row.values);
  };
  pack_rows(workspace.packed_states);
  pack_rows(workspace.packed_directions);
  pack_rows(workspace.packed_fields);
  for (auto& row : workspace.packed_outputs)
    pack_field_into(registry.output_at(row.slot), workspace.locations, geometry.domain, false,
                    layout_reference, row.values);

  const std::size_t count = workspace.locations.size();
  const PopsConstFieldViewV1 coordinate_view =
      const_view(workspace.xy, count, 2, spec.layout_identity, spec.region.identity);
  PopsFieldBoundaryRequestV1 request{sizeof(PopsFieldBoundaryRequestV1),
                                     spec.target_identity.c_str(),
                                     spec.region.view(),
                                     coordinate_view,
                                     workspace.states.size(),
                                     workspace.states.data(),
                                     workspace.directions.size(),
                                     workspace.directions.data(),
                                     workspace.fields.size(),
                                     workspace.fields.data(),
                                     workspace.parameters.size(),
                                     workspace.parameters.data(),
                                     workspace.outputs.size(),
                                     workspace.outputs.data(),
                                     workspace.rate,
                                     workspace.nonlinear,
                                     point.level,
                                     logical_time(point),
                                     component.execution().view()};
  PopsComponentStatusV1 status{sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1,
                               nullptr};
  const int code = component::evaluate_field_boundary(component.field_api(), component.state(),
                                                      request, status, jvp);
  PreparedBoundaryComponent<Operation>::require_success(
      code, status, jvp ? "field boundary jvp" : "field boundary residual");
  for (auto& row : workspace.packed_outputs)
    scatter_field(registry.output_at(row.slot), workspace.locations, row.values);
}

}  // namespace pops::detail
