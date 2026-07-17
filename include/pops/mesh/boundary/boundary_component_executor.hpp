#pragma once

#include <pops/mesh/boundary/prepared_boundary_component.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/runtime/multiblock/interface_flux_scheduler.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pops::detail {

struct BoundaryPointLocation {
  int local_fab = 0;
  int i = 0;
  int j = 0;
};

/// Exact qualified-storage registry for the N-ary component ABI.  A Handle identity is bound once;
/// no executor duplicates one view merely because a table contains several identities.  The
/// convenience PreparedBoundaryPlan path binds its one block state/output, while richer field
/// operators may call the registry overload with arbitrarily many independently owned MultiFabs.
class BoundaryFieldRegistry {
 public:
  void bind_state(std::string identity, const MultiFab& field) {
    bind_const_(states_, std::move(identity), field, "state");
  }
  void bind_direction(std::string identity, const MultiFab& field) {
    bind_const_(directions_, std::move(identity), field, "direction");
  }
  void bind_field(std::string identity, const MultiFab& field) {
    bind_const_(fields_, std::move(identity), field, "field");
  }
  void bind_output(std::string identity, MultiFab& field) {
    if (identity.empty() || outputs_.count(identity) != 0)
      throw std::invalid_argument("boundary output identity is empty or multiply bound");
    for (const auto& [_, existing] : outputs_)
      if (existing == &field)
        throw std::invalid_argument(
            "distinct boundary output identities may not alias one mutable field");
    outputs_.emplace(std::move(identity), &field);
  }

  const MultiFab& state(const std::string& identity) const {
    return require_const_(states_, identity, "state");
  }
  const MultiFab& direction(const std::string& identity) const {
    return require_const_(directions_, identity, "direction");
  }
  const MultiFab& field(const std::string& identity) const {
    return require_const_(fields_, identity, "field");
  }
  MultiFab& output(const std::string& identity) const {
    const auto found = outputs_.find(identity);
    if (found == outputs_.end())
      throw std::invalid_argument("boundary component has no exact mutable output binding for '" +
                                  identity + "'");
    return *found->second;
  }

 private:
  using ConstTable = std::unordered_map<std::string, const MultiFab*>;
  ConstTable states_, directions_, fields_;
  std::unordered_map<std::string, MultiFab*> outputs_;

  static void bind_const_(ConstTable& table, std::string identity, const MultiFab& field,
                          const char* role) {
    if (identity.empty() || table.count(identity) != 0)
      throw std::invalid_argument(std::string("boundary ") + role +
                                  " identity is empty or multiply bound");
    table.emplace(std::move(identity), &field);
  }
  static const MultiFab& require_const_(const ConstTable& table, const std::string& identity,
                                        const char* role) {
    const auto found = table.find(identity);
    if (found == table.end())
      throw std::invalid_argument(std::string("boundary component has no exact ") + role +
                                  " binding for '" + identity + "'");
    return *found->second;
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
  std::vector<int> side_for_axis(2, 0);
  for (std::size_t index = 0; index < region.axes.size(); ++index)
    side_for_axis[static_cast<std::size_t>(region.axes[index])] = region.sides[index];

  std::vector<BoundaryPointLocation> locations;
  for (int li = 0; li < field.local_size(); ++li) {
    const Box2D valid = field.box(li);
    const int grow = ghosts ? std::min(depth, field.n_grow()) : 0;
    // Grow only the axes fixed by the boundary region.  Growing the tangential axis would pack the
    // same physical point once from each neighbouring Fab's tangential ghost strip, so a component
    // result would be scattered ambiguously at internal box seams.
    const int ilo = valid.lo[0] - (side_for_axis[0] == 0 ? 0 : grow);
    const int ihi = valid.hi[0] + (side_for_axis[0] == 0 ? 0 : grow);
    const int jlo = valid.lo[1] - (side_for_axis[1] == 0 ? 0 : grow);
    const int jhi = valid.hi[1] + (side_for_axis[1] == 0 ? 0 : grow);
    for (int j = jlo; j <= jhi; ++j) {
      for (int i = ilo; i <= ihi; ++i) {
        const int coordinates[2] = {i, j};
        bool selected = true;
        for (int axis = 0; axis < 2; ++axis) {
          const int side = side_for_axis[static_cast<std::size_t>(axis)];
          if (side != 0) {
            selected = selected && region_coordinate_matches(coordinates[axis], axis, side, domain,
                                                             ghosts, depth);
          } else {
            selected = selected && coordinates[axis] >= domain.lo[axis] &&
                       coordinates[axis] <= domain.hi[axis];
          }
        }
        if (selected)
          locations.push_back({li, i, j});
      }
    }
  }
  return locations;
}

inline std::vector<double> pack_field(const MultiFab& field,
                                      const std::vector<BoundaryPointLocation>& locations,
                                      const Box2D& domain, bool clamp_to_domain,
                                      const MultiFab& layout_reference) {
  if (field.box_array().boxes() != layout_reference.box_array().boxes() ||
      field.dmap().ranks() != layout_reference.dmap().ranks() ||
      field.local_size() != layout_reference.local_size())
    throw std::runtime_error(
        "boundary component qualified field differs from the prepared "
        "BoxArray/DistributionMapping");
  const_cast<MultiFab&>(field).sync_host();
  std::vector<double> packed(locations.size() * static_cast<std::size_t>(field.ncomp()));
  for (std::size_t point = 0; point < locations.size(); ++point) {
    const auto& location = locations[point];
    if (location.local_fab < 0 || location.local_fab >= field.local_size())
      throw std::runtime_error("boundary component field layout differs from state layout");
    if (field.box(location.local_fab) != layout_reference.box(location.local_fab))
      throw std::runtime_error(
          "boundary component local Fab ordering differs from the prepared state layout");
    const int i = clamp_to_domain ? std::clamp(location.i, domain.lo[0], domain.hi[0]) : location.i;
    const int j = clamp_to_domain ? std::clamp(location.j, domain.lo[1], domain.hi[1]) : location.j;
    const ConstArray4 values = field.fab(location.local_fab).const_array();
    for (int component = 0; component < field.ncomp(); ++component)
      packed[point * static_cast<std::size_t>(field.ncomp()) +
             static_cast<std::size_t>(component)] = values(i, j, component);
  }
  return packed;
}

inline void scatter_field(MultiFab& field, const std::vector<BoundaryPointLocation>& locations,
                          const std::vector<double>& packed) {
  if (packed.size() != locations.size() * static_cast<std::size_t>(field.ncomp()))
    throw std::runtime_error("boundary component output shape changed across native ABI call");
  field.sync_host();
  for (std::size_t point = 0; point < locations.size(); ++point) {
    const auto& location = locations[point];
    Array4 values = field.fab(location.local_fab).array();
    for (int component = 0; component < field.ncomp(); ++component)
      values(location.i, location.j, component) =
          static_cast<Real>(packed[point * static_cast<std::size_t>(field.ncomp()) +
                                   static_cast<std::size_t>(component)]);
  }
  field.sync_device();
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

struct PackedConstBoundaryField {
  std::string identity;
  std::vector<double> values;
};

struct PackedMutableBoundaryField {
  std::string identity;
  MultiFab* destination = nullptr;
  std::vector<double> values;
};

template <class Lookup>
inline std::vector<PackedConstBoundaryField> pack_qualified_const_fields(
    const std::vector<std::string>& identities, Lookup&& lookup,
    const std::vector<BoundaryPointLocation>& locations, const Box2D& domain,
    const MultiFab& layout_reference, bool clamp_to_domain = false) {
  std::vector<PackedConstBoundaryField> packed;
  packed.reserve(identities.size());
  for (const std::string& identity : identities) {
    const MultiFab& field = lookup(identity);
    packed.push_back(
        {identity, pack_field(field, locations, domain, clamp_to_domain, layout_reference)});
  }
  return packed;
}

inline std::vector<PopsQualifiedConstFieldV1> qualified_const_views(
    const std::vector<PackedConstBoundaryField>& packed, std::size_t count,
    const std::string& layout, const std::string& patch) {
  std::vector<PopsQualifiedConstFieldV1> result;
  result.reserve(packed.size());
  for (const auto& row : packed)
    result.push_back({sizeof(PopsQualifiedConstFieldV1), 1u, row.identity.c_str(),
                      const_view(row.values, count, row.values.size() / count, layout, patch)});
  return result;
}

inline PopsQualifiedConstFieldV1 optional_field(const std::string& identity,
                                                const PopsConstFieldViewV1& values) {
  if (identity.empty())
    return {sizeof(PopsQualifiedConstFieldV1), 0u, nullptr, {}};
  return {sizeof(PopsQualifiedConstFieldV1), 1u, identity.c_str(), values};
}

inline void apply_ghost_component(const PreparedGhostBoundaryComponent& component, MultiFab& state,
                                  const BoundaryFieldRegistry& registry, const Geometry& geometry,
                                  int depth,
                                  const runtime::multiblock::BoundaryEvaluationPoint& point) {
  const auto& spec = component.spec();
  const auto locations = boundary_locations(state, geometry.domain, spec.region, depth, true);
  // A distributed rank is allowed to own no point of a globally installed boundary region.  The
  // component ABI is a local host batch, so only owning ranks invoke it; treating an empty local
  // batch as a configuration failure would make any decomposed boundary unusable.
  if (locations.empty())
    return;
  const std::size_t count = locations.size();
  std::vector<double> interior = pack_field(state, locations, geometry.domain, true, state);
  std::vector<double> ghosts = pack_field(state, locations, geometry.domain, false, state);
  std::vector<double> xy = coordinates(locations, geometry);
  const PopsConstFieldViewV1 interior_view =
      const_view(interior, count, static_cast<std::size_t>(state.ncomp()), spec.layout_identity,
                 spec.region.identity);
  const PopsConstFieldViewV1 coordinate_view =
      const_view(xy, count, 2, spec.layout_identity, spec.region.identity);
  const PopsFieldViewV1 ghost_view =
      field_view(ghosts, count, static_cast<std::size_t>(state.ncomp()), spec.layout_identity,
                 spec.region.identity);

  auto packed_dependencies = pack_qualified_const_fields(
      spec.states,
      [&registry](const std::string& identity) -> const MultiFab& {
        return registry.state(identity);
      },
      locations, geometry.domain, state, true);
  auto packed_fields = pack_qualified_const_fields(
      spec.fields,
      [&registry](const std::string& identity) -> const MultiFab& {
        return registry.field(identity);
      },
      locations, geometry.domain, state, true);
  packed_dependencies.insert(packed_dependencies.end(),
                             std::make_move_iterator(packed_fields.begin()),
                             std::make_move_iterator(packed_fields.end()));
  const auto dependencies =
      qualified_const_views(packed_dependencies, count, spec.layout_identity, spec.region.identity);
  const auto parameters = scalar_table(spec);
  PopsGhostBoundaryRequestV1 request{sizeof(PopsGhostBoundaryRequestV1),
                                     spec.producer_identity.c_str(),
                                     spec.state_identity.c_str(),
                                     spec.ghost_identity.c_str(),
                                     interior_view,
                                     ghost_view,
                                     coordinate_view,
                                     spec.region.view(),
                                     dependencies.size(),
                                     dependencies.data(),
                                     parameters.size(),
                                     parameters.data(),
                                     logical_time(point),
                                     spec.execution->view()};
  PopsComponentStatusV1 status{sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1,
                               nullptr};
  const int code =
      component::apply_ghost_boundary(component.ghost_api(), component.state(), request, status);
  PreparedGhostBoundaryComponent::require_success(code, status, "apply_region_batch");
  scatter_field(state, locations, ghosts);
}

inline void apply_ghost_component(const PreparedGhostBoundaryComponent& component, MultiFab& state,
                                  const MultiFab* auxiliary, const Geometry& geometry, int depth,
                                  const runtime::multiblock::BoundaryEvaluationPoint& point) {
  const auto& spec = component.spec();
  BoundaryFieldRegistry registry;
  registry.bind_state(spec.state_identity, state);
  if (!spec.fields.empty()) {
    if (auxiliary == nullptr)
      throw std::runtime_error("GhostBoundary field dependencies have no prepared auxiliary field");
    if (spec.fields.size() != 1)
      throw std::runtime_error(
          "GhostBoundary has multiple qualified field dependencies; use the N-ary registry seam");
    registry.bind_field(spec.fields.front(), *auxiliary);
  }
  apply_ghost_component(component, state, registry, geometry, depth, point);
}

template <PreparedBoundaryOperation Operation>
inline void apply_field_component(const PreparedBoundaryComponent<Operation>& component,
                                  const BoundaryFieldRegistry& registry, const Geometry& geometry,
                                  const runtime::multiblock::BoundaryEvaluationPoint& point) {
  const auto& spec = component.spec();
  static_assert(Operation == PreparedBoundaryOperation::FieldResidual ||
                Operation == PreparedBoundaryOperation::FieldJvp);
  constexpr bool jvp = Operation == PreparedBoundaryOperation::FieldJvp;
  const MultiFab& layout_reference = registry.state(spec.states.front());
  const auto locations =
      boundary_locations(layout_reference, geometry.domain, spec.region, 1, false);
  if (locations.empty())
    return;
  const std::size_t count = locations.size();
  std::vector<double> xy = coordinates(locations, geometry);
  const PopsConstFieldViewV1 coordinate_view =
      const_view(xy, count, 2, spec.layout_identity, spec.region.identity);
  auto packed_states = pack_qualified_const_fields(
      spec.states,
      [&registry](const std::string& identity) -> const MultiFab& {
        return registry.state(identity);
      },
      locations, geometry.domain, layout_reference);
  auto packed_directions = pack_qualified_const_fields(
      spec.directions,
      [&registry](const std::string& identity) -> const MultiFab& {
        return registry.direction(identity);
      },
      locations, geometry.domain, layout_reference);
  auto packed_fields = pack_qualified_const_fields(
      spec.fields,
      [&registry](const std::string& identity) -> const MultiFab& {
        return registry.field(identity);
      },
      locations, geometry.domain, layout_reference);
  const auto states =
      qualified_const_views(packed_states, count, spec.layout_identity, spec.region.identity);
  const auto directions =
      qualified_const_views(packed_directions, count, spec.layout_identity, spec.region.identity);
  const auto fields =
      qualified_const_views(packed_fields, count, spec.layout_identity, spec.region.identity);
  std::vector<PackedMutableBoundaryField> packed_outputs;
  packed_outputs.reserve(spec.outputs.size());
  for (const std::string& identity : spec.outputs) {
    MultiFab& destination = registry.output(identity);
    packed_outputs.push_back(
        {identity, &destination,
         pack_field(destination, locations, geometry.domain, false, layout_reference)});
  }
  std::vector<PopsQualifiedFieldV1> outputs;
  outputs.reserve(packed_outputs.size());
  for (auto& row : packed_outputs)
    outputs.push_back({sizeof(PopsQualifiedFieldV1), row.identity.c_str(),
                       field_view(row.values, count, row.values.size() / count,
                                  spec.layout_identity, spec.region.identity)});

  const auto find_const_view = [&](const std::string& identity) -> PopsConstFieldViewV1 {
    for (const auto* table : {&packed_states, &packed_directions, &packed_fields})
      for (const auto& row : *table)
        if (row.identity == identity)
          return const_view(row.values, count, row.values.size() / count, spec.layout_identity,
                            spec.region.identity);
    for (const auto& row : packed_outputs)
      if (row.identity == identity)
        return const_view(row.values, count, row.values.size() / count, spec.layout_identity,
                          spec.region.identity);
    throw std::invalid_argument(
        "boundary optional field has no exact qualified storage binding for '" + identity + "'");
  };
  const auto parameters = scalar_table(spec);
  const PopsQualifiedConstFieldV1 rate =
      spec.rate.empty()
          ? PopsQualifiedConstFieldV1{sizeof(PopsQualifiedConstFieldV1), 0u, nullptr, {}}
          : PopsQualifiedConstFieldV1{sizeof(PopsQualifiedConstFieldV1), 1u, spec.rate.c_str(),
                                      find_const_view(spec.rate)};
  const PopsQualifiedConstFieldV1 nonlinear =
      spec.nonlinear_iterate.empty()
          ? PopsQualifiedConstFieldV1{sizeof(PopsQualifiedConstFieldV1), 0u, nullptr, {}}
          : PopsQualifiedConstFieldV1{sizeof(PopsQualifiedConstFieldV1), 1u,
                                      spec.nonlinear_iterate.c_str(),
                                      find_const_view(spec.nonlinear_iterate)};
  PopsFieldBoundaryRequestV1 request{sizeof(PopsFieldBoundaryRequestV1),
                                     spec.target_identity.c_str(),
                                     spec.region.view(),
                                     coordinate_view,
                                     states.size(),
                                     states.data(),
                                     directions.size(),
                                     directions.data(),
                                     fields.size(),
                                     fields.data(),
                                     parameters.size(),
                                     parameters.data(),
                                     outputs.size(),
                                     outputs.data(),
                                     rate,
                                     nonlinear,
                                     point.level,
                                     logical_time(point),
                                     spec.execution->view()};
  PopsComponentStatusV1 status{sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1,
                               nullptr};
  const int code = component::evaluate_field_boundary(component.field_api(), component.state(),
                                                      request, status, jvp);
  PreparedBoundaryComponent<Operation>::require_success(
      code, status, jvp ? "field boundary jvp" : "field boundary residual");
  for (auto& row : packed_outputs)
    scatter_field(*row.destination, locations, row.values);
}

template <PreparedBoundaryOperation Operation>
inline void apply_field_component(const PreparedBoundaryComponent<Operation>& component,
                                  const MultiFab& state, const MultiFab* direction,
                                  const MultiFab* auxiliary, const Geometry& geometry,
                                  MultiFab& output,
                                  const runtime::multiblock::BoundaryEvaluationPoint& point) {
  const auto& spec = component.spec();
  BoundaryFieldRegistry registry;
  registry.bind_state(spec.state_identity, state);
  if (!spec.directions.empty()) {
    if (direction == nullptr || spec.directions.size() != 1)
      throw std::runtime_error(
          "FieldBoundaryClosure directions require the N-ary qualified registry seam");
    registry.bind_direction(spec.directions.front(), *direction);
  } else if (direction != nullptr) {
    throw std::invalid_argument(
        "FieldBoundaryClosure received a direction for a residual operation");
  }
  if (!spec.fields.empty()) {
    if (auxiliary == nullptr || spec.fields.size() != 1)
      throw std::runtime_error(
          "FieldBoundaryClosure fields require the N-ary qualified registry seam");
    registry.bind_field(spec.fields.front(), *auxiliary);
  }
  if (spec.outputs.size() != 1)
    throw std::runtime_error(
        "FieldBoundaryClosure outputs require the N-ary qualified registry seam");
  registry.bind_output(spec.outputs.front(), output);
  apply_field_component(component, registry, geometry, point);
}

}  // namespace pops::detail
