#pragma once

// Representation copies between native Cartesian storage and the ordinary
// finite algebra. No prolongation, restriction, boundary law or physical
// equation lives here.
#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <pops/runtime/program/distributed_array.hpp>
#include <pops/runtime/program/field_topology_view.hpp>
#include <string>
#include <vector>

namespace pops::runtime::program::field_bundle {
inline std::size_t member(const ContinuationType& type, const std::string& name) {
  if (type.kind != ContinuationKind::Product)
    throw std::invalid_argument("Field Bundle port requires an exact Product representation");
  const auto found = std::find(type.names.begin(), type.names.end(), name);
  if (found == type.names.end())
    throw std::invalid_argument("Field Bundle representation lacks member " + name);
  return std::size_t(found - type.names.begin());
}
inline const ContinuationType& child(const ContinuationType& type, const std::string& name) {
  return type.children.at(member(type, name));
}
template <int Dim>
using Ref = ContinuationValueRef<Dim>;
template <int Dim>
using Value = ContinuationValue<Dim>;
template <int Dim>
Ref<Dim> get(const ContinuationType& type, const Ref<Dim>& value, const std::string& name) {
  if (!value || value->kind() != ContinuationKind::Product ||
      value->members().size() != type.children.size())
    throw std::invalid_argument("Field Bundle value differs from its representation");
  return value->members().at(member(type, name));
}
template <int Dim>
Ref<Dim> product(const ContinuationType& type, const std::map<std::string, Ref<Dim>>& values) {
  if (type.kind != ContinuationKind::Product || type.names.size() != values.size())
    throw std::invalid_argument("Field Bundle port cannot erase representation members");
  std::vector<Ref<Dim>> result;
  for (const auto& name : type.names)
    result.push_back(values.at(name));
  return Value<Dim>::product(std::move(result));
}
inline std::int64_t checked_add(std::int64_t a, std::int64_t b) {
  if (a < 0 || b < 0 || b > std::numeric_limits<std::int64_t>::max() - a)
    throw std::overflow_error("Field Bundle storage cardinality exceeds Int64");
  return a + b;
}
template <int Dim>
Box<Dim> grown_box(const Box<Dim>& valid, const Extent<Dim>& ghosts) {
  Box<Dim> result = valid;
  for (int axis = 0; axis < Dim; ++axis) {
    if (ghosts[axis] < 0)
      throw std::invalid_argument("Field Bundle ghosts must be nonnegative");
    result.lo[axis] = pops::detail::checked_box_index(std::int64_t(valid.lo[axis]) - ghosts[axis],
                                                      "Field Bundle ghost lower bound");
    result.hi[axis] = pops::detail::checked_box_index(std::int64_t(valid.hi[axis]) + ghosts[axis],
                                                      "Field Bundle ghost upper bound");
  }
  return result;
}
template <int Dim>
Index<Dim> cell_index(const Box<Dim>& box, std::int64_t offset) {
  Index<Dim> index{};
  for (int axis = 0; axis < Dim; ++axis) {
    index[axis] = pops::detail::checked_box_index(
        std::int64_t(box.lo[axis]) + offset % box.length(axis), "Field Bundle cell index");
    offset /= box.length(axis);
  }
  return index;
}
template <int Dim>
std::size_t cell_offset(const Box<Dim>& box, const Index<Dim>& index) {
  std::int64_t offset = 0, stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    if (index[axis] < box.lo[axis] || index[axis] > box.hi[axis])
      throw std::invalid_argument("Field Bundle measure has no cell in this representation");
    offset += (std::int64_t(index[axis]) - box.lo[axis]) * stride;
    stride *= box.length(axis);
  }
  return static_cast<std::size_t>(offset);
}
template <int Dim>
struct Patch {
  std::int64_t owner, global, begin, count;
  bool replica;
  Box<Dim> valid, grown;
};
template <int Dim>
struct Layout {
  std::vector<Patch<Dim>> patches;
  std::vector<std::int64_t> offsets;
};
template <int Dim>
Layout<Dim> layout(const FieldTopologyView<Dim>& topology, const ExecutionLane& lane,
                   std::uint64_t capacity) {
  topology.require();
  Layout<Dim> out;
  const auto& boxes = topology.layout();
  const auto& distribution = topology.distribution();
  const auto replicated = distribution.replicated();
  if (!replicated &&
      (distribution.rank_space().size() != std::size_t(lane.size()) ||
       distribution.rank_space().linear_rank(topology.local_rank()) != std::size_t(lane.rank())))
    throw std::invalid_argument("Field topology and finite algebra use different execution lanes");
  continuation_detail::StructuralBudget budget(capacity);
  budget.frames(std::size_t(lane.size()) + 1, sizeof(std::int64_t));
  out.offsets.push_back(0);
  std::int64_t next = 0;
  for (int rank = 0; rank < lane.size(); ++rank) {
    for (std::size_t p = 0; p < boxes.size(); ++p) {
      const auto owner =
          replicated
              ? rank
              : static_cast<int>(distribution.rank_space().linear_rank(distribution.owner(p)));
      if (owner != rank)
        continue;
      const auto grown = grown_box(boxes[p], topology.ghosts());
      const auto count = grown.numPts();
      budget.frames(out.patches.size() + 1, sizeof(Patch<Dim>));
      out.patches.push_back({rank, static_cast<std::int64_t>(p), next, count,
                             replicated && rank != 0, boxes[p], grown});
      next = checked_add(next, count);
    }
    out.offsets.push_back(next);
  }
  return out;
}
template <int Dim>
Ref<Dim> patch_value(const ContinuationType& type, const Patch<Dim>& patch,
                     const Extent<Dim>& ghosts) {
  std::map<std::string, Ref<Dim>> values{{"owner_rank", Value<Dim>::integer(patch.owner)},
                                         {"global_patch", Value<Dim>::integer(patch.global)},
                                         {"global_offset", Value<Dim>::integer(patch.begin)},
                                         {"count", Value<Dim>::integer(patch.count)},
                                         {"is_replica", Value<Dim>::boolean(patch.replica)}};
  for (int axis = 0; axis < Dim; ++axis) {
    const auto a = std::to_string(axis);
    values.emplace("lo" + a, Value<Dim>::integer(patch.valid.lo[axis]));
    values.emplace("hi" + a, Value<Dim>::integer(patch.valid.hi[axis]));
    values.emplace("ghost" + a, Value<Dim>::integer(ghosts[axis]));
  }
  return product<Dim>(type, values);
}
// Metadata is copied bit for bit and authenticated on import. Comparing it is
// representation validation, never an approximation or a scientific tolerance.
template <int Dim>
bool same_metadata(const Ref<Dim>& a, const Ref<Dim>& b) {
  if (!a || !b || a->kind() != b->kind())
    return false;
  switch (a->kind()) {
    case ContinuationKind::Boolean:
      return a->boolean() == b->boolean();
    case ContinuationKind::Int64:
      return a->integer() == b->integer();
    case ContinuationKind::Float64:
      return std::bit_cast<std::uint64_t>(a->real()) == std::bit_cast<std::uint64_t>(b->real());
    case ContinuationKind::Product:
      if (a->members().size() != b->members().size())
        return false;
      for (std::size_t n = 0; n < a->members().size(); ++n)
        if (!same_metadata<Dim>(a->members()[n], b->members()[n]))
          return false;
      return true;
    default:
      return false;
  }
}
template <int Dim>
class MeasureMirror {
 public:
  MeasureMirror(const MultiFab<Dim>* field, std::size_t global) {
    if (field) {
      const auto local = field->local_index_of(global);
      const auto& fab = field->fab(local);
      if (field->ncomp() != 1)
        throw std::invalid_argument("Field Bundle measure must have one component");
      box_ = fab.grown_box();
      mirror_.emplace(fab.create_host_mirror());
      fab.copy_to_host(*mirror_);
    }
  }
  double at(const Index<Dim>& index, double fallback) const {
    return mirror_ ? static_cast<double>((*mirror_)(cell_offset(box_, index))) : fallback;
  }

 private:
  Box<Dim> box_;
  std::optional<typename Fab<Dim>::host_mirror_type> mirror_;
};
template <int Dim, class F>
void each_local_cell(const FieldTopologyView<Dim>& topology, const Layout<Dim>& storage, int rank,
                     const ContinuationType& geometry, F&& consume) {
  double volume = 1.;
  for (int axis = 0; axis < Dim; ++axis)
    volume *= topology.geometry().spacing(axis);
  if (!(volume > 0.) || !std::isfinite(volume))
    throw std::invalid_argument("Field Bundle cell measure must be finite and positive");
  for (const auto& patch : storage.patches) {
    if (patch.owner != rank)
      continue;
    const auto global = static_cast<std::size_t>(patch.global);
    MeasureMirror<Dim> active(topology.active_mask(), global);
    MeasureMirror<Dim> fraction(topology.volume_fraction(), global);
    MeasureMirror<Dim> coverage(&topology.coverage_mask(), global);
    for (std::int64_t cell = 0; cell < patch.count; ++cell) {
      const auto index = cell_index(patch.grown, cell);
      bool valid = true;
      for (int axis = 0; axis < Dim; ++axis)
        valid = valid && index[axis] >= patch.valid.lo[axis] && index[axis] <= patch.valid.hi[axis];
      const auto a = valid ? active.at(index, 1.) : 0.;
      const auto k = valid ? fraction.at(index, 1.) : 0.;
      const auto c = valid ? coverage.at(index, 1.) : 0.;
      if (!std::isfinite(a) || !std::isfinite(k) || !std::isfinite(c) || k < 0. || k > 1.)
        throw std::invalid_argument("Field Bundle geometry contains an invalid physical measure");
      std::map<std::string, Ref<Dim>> coordinates, indices;
      for (int axis = 0; axis < Dim; ++axis) {
        coordinates.emplace(
            "x" + std::to_string(axis),
            Value<Dim>::real(topology.geometry().cell_coordinate(axis, index[axis])));
        indices.emplace("i" + std::to_string(axis), Value<Dim>::integer(index[axis]));
      }
      std::map<std::string, Ref<Dim>> values{
          {"coordinates", product<Dim>(child(geometry, "coordinates").children.at(0), coordinates)},
          {"indices", product<Dim>(child(geometry, "indices").children.at(0), indices)},
          {"measure",
           Value<Dim>::real(valid && a >= .5 && c >= .5 && !patch.replica ? volume * k : 0.)},
          {"volume_fraction", Value<Dim>::real(k)},
          {"is_valid", Value<Dim>::boolean(valid)},
          {"is_active", Value<Dim>::boolean(valid && a >= .5)},
          {"is_covered", Value<Dim>::boolean(valid && c < .5)}};
      consume(patch, cell, values);
    }
  }
}
template <int Dim, class Context>
Ref<Dim> export_field(Context& ctx, const MultiFab<Dim>& field,
                      const FieldTopologyView<Dim>& topology, const ContinuationType& schema,
                      std::uint64_t capacity) {
  auto& registry = ctx.distributed_registry();
  auto& transport = ctx.continuation_transport_session();
  const auto& lane = registry.lane();
  Layout<Dim> storage;
  ctx.continuation_local_phase(
      [&] {
        ctx.require_field_topology(topology);
        topology.require_field(field);
        schema.validate(capacity);
        if (!schema.dependent_bundle)
          throw std::invalid_argument("Field export requires a dependent Bundle schema");
        storage = layout(topology, lane, capacity);
        transport.bind_distributed_registry(
            registry, [&registry](auto version, const auto& domain, const auto& element) {
              return distributed::resolve_array<Dim>(registry, version, domain, element);
            });
      },
      "Field Bundle representation preflight");
  const auto& domain_type = child(schema, "instance");
  const auto& maps_type = child(schema, "maps");
  const auto& patches_type = child(maps_type, "patches");
  const auto& patch_domain_type = child(maps_type, "patch_instance");
  const auto& geometry_type = child(schema, "geometry");
  const auto& fields_type = child(schema, "fields");
  const auto& values_type = child(fields_type, "values");
  const auto domain = registry.issue(domain_type.family, domain_type.capacity, storage.offsets);
  const auto patch_domain =
      ctx.continuation_new_domain(static_cast<std::int64_t>(storage.patches.size()),
                                  patch_domain_type.capacity, patch_domain_type.family);
  std::map<std::string, std::shared_ptr<distributed::Array<Dim>>> pending;
  std::vector<Ref<Dim>> patches;
  ctx.continuation_local_phase(
      [&] {
        continuation_detail::StructuralBudget budget(capacity);
        budget.frames(
            static_cast<std::size_t>(domain->local_size()),
            (static_cast<std::size_t>(field.ncomp()) + 2 * Dim + 16) * sizeof(Value<Dim>));
        auto allocate = [&](const std::string& name, const ContinuationType& type) {
          if (type.kind != ContinuationKind::DistributedArray ||
              type.family != domain_type.family || type.capacity != domain_type.capacity)
            throw std::invalid_argument(
                "Field Bundle cell arrays must share "
                "the exact distributed domain");
          auto array = std::make_shared<distributed::Array<Dim>>();
          array->domain = domain;
          array->element = type.children.at(0);
          array->values.resize(static_cast<std::size_t>(domain->local_size()));
          pending.emplace(name, std::move(array));
        };
        allocate("values", values_type);
        for (const auto& name : geometry_type.names)
          allocate(name, child(geometry_type, name));
        for (const auto& patch : storage.patches)
          patches.push_back(
              patch_value<Dim>(patches_type.children.at(0), patch, topology.ghosts()));
        each_local_cell(topology, storage, lane.rank(), geometry_type,
                        [&](const Patch<Dim>& patch, std::int64_t cell, const auto& metadata) {
                          const auto offset =
                              static_cast<std::size_t>(patch.begin + cell - domain->begin());
                          for (const auto& [name, value] : metadata)
                            pending.at(name)->values[offset] = value;
                        });
        for (const auto& patch : storage.patches) {
          if (patch.owner != lane.rank())
            continue;
          const auto local = field.local_index_of(static_cast<std::size_t>(patch.global));
          const auto& fab = field.fab(local);
          auto mirror = fab.create_host_mirror();
          fab.copy_to_host(mirror);
          for (std::int64_t cell = 0; cell < patch.count; ++cell) {
            std::map<std::string, Ref<Dim>> components;
            for (int c = 0; c < field.ncomp(); ++c)
              components.emplace(
                  "c" + std::to_string(c),
                  Value<Dim>::real(mirror(static_cast<std::size_t>(c * patch.count + cell))));
            pending.at("values")->values.at(
                static_cast<std::size_t>(patch.begin + cell - domain->begin())) =
                product<Dim>(values_type.children.at(0), components);
          }
        }
      },
      "Field Bundle storage extraction");
  std::map<std::string, Ref<Dim>> published;
  for (const auto& [name, array] : pending) {
    auto value = distributed::publish_array<Dim>(registry, array, capacity, &transport);
    ctx.continuation_local_phase(
        [&] { published.emplace(name, Value<Dim>::distributed_array(value)); },
        "Field Bundle array reference publication");
  }
  return ctx.continuation_local_phase(
      [&] {
        std::map<std::string, Ref<Dim>> geometry;
        for (const auto& name : geometry_type.names)
          geometry.emplace(name, published.at(name));
        auto result = product<Dim>(
            schema,
            {{"instance", Value<Dim>::distributed_domain(domain)},
             {"fields", product<Dim>(fields_type, {{"values", published.at("values")}})},
             {"geometry", product<Dim>(geometry_type, geometry)},
             {"maps",
              product<Dim>(maps_type,
                           {{"patch_instance", Value<Dim>::domain(patch_domain)},
                            {"patches", Value<Dim>::array(patch_domain, std::move(patches))}})},
             {"histories", Value<Dim>::product({})}});
        (void)encode_continuation_transport<Dim>(schema, result, capacity, transport);
        transport.capture(result);
        return result;
      },
      "Field Bundle dependent publication");
}
template <int Dim, class Context>
std::shared_ptr<MultiFab<Dim>> import_field(Context& ctx, const Ref<Dim>& bundle,
                                            const MultiFab<Dim>& prototype,
                                            const FieldTopologyView<Dim>& topology,
                                            const ContinuationType& schema,
                                            std::uint64_t capacity) {
  auto& registry = ctx.distributed_registry();
  auto& transport = ctx.continuation_transport_session();
  const auto& lane = registry.lane();
  return ctx.continuation_local_phase(
      [&] {
        ctx.require_field_topology(topology);
        topology.require();
        // The template authenticates the statically checked scientific Field
        // type. Its old layout may differ: storage is allocated from the target
        // capability.
        if (prototype.ncomp() != topology.components())
          throw std::invalid_argument("Field Bundle import changes the Field component contract");
        schema.validate(capacity);
        if (!schema.dependent_bundle)
          throw std::invalid_argument("Field import requires a dependent Bundle schema");
        (void)encode_continuation_transport<Dim>(schema, bundle, capacity, transport);
        const auto storage = layout(topology, lane, capacity);
        const auto domain = get<Dim>(schema, bundle, "instance")->distributed_domain_ref();
        transport.authenticate_distributed_domain(domain);
        if (domain->partition_offsets() != storage.offsets)
          throw std::invalid_argument("Field Bundle domain differs from the exact target storage");
        const auto& maps_type = child(schema, "maps");
        const auto maps = get<Dim>(schema, bundle, "maps");
        const auto patches = get<Dim>(maps_type, maps, "patches");
        const auto& patch_type = child(maps_type, "patches").children.at(0);
        if (patches->members().size() != storage.patches.size())
          throw std::invalid_argument("Field Bundle patch table changes the target storage");
        for (std::size_t p = 0; p < storage.patches.size(); ++p)
          if (!same_metadata<Dim>(
                  patches->members()[p],
                  patch_value<Dim>(patch_type, storage.patches[p], topology.ghosts())))
            throw std::invalid_argument("Field Bundle patch metadata was changed");
        const auto& geometry_type = child(schema, "geometry");
        const auto geometry = get<Dim>(schema, bundle, "geometry");
        const auto& fields_type = child(schema, "fields");
        const auto fields = get<Dim>(schema, bundle, "fields");
        const auto values = get<Dim>(fields_type, fields, "values")->distributed_array_ref();
        if (!values || values->domain != domain)
          throw std::invalid_argument("Field Bundle values lost their instance domain");
        std::map<std::string, distributed::ArrayRef<Dim>> metadata;
        for (const auto& name : geometry_type.names) {
          auto array = get<Dim>(geometry_type, geometry, name)->distributed_array_ref();
          if (!array || array->domain != domain)
            throw std::invalid_argument("Field Bundle geometry lost its instance domain");
          metadata.emplace(name, std::move(array));
        }
        each_local_cell(topology, storage, lane.rank(), geometry_type,
                        [&](const Patch<Dim>& patch, std::int64_t cell, const auto& expected) {
                          const auto offset =
                              static_cast<std::size_t>(patch.begin + cell - domain->begin());
                          for (const auto& [name, value] : expected)
                            if (!same_metadata<Dim>(metadata.at(name)->values.at(offset), value))
                              throw std::invalid_argument(
                                  "Field Bundle geometry differs "
                                  "from the target capability");
                        });
        auto result = std::make_shared<MultiFab<Dim>>(topology.layout(), topology.distribution(),
                                                      topology.local_rank(), topology.components(),
                                                      topology.ghosts());
        const auto& element = child(fields_type, "values").children.at(0);
        for (const auto& patch : storage.patches) {
          if (patch.owner != lane.rank())
            continue;
          auto& fab = result->fab(result->local_index_of(static_cast<std::size_t>(patch.global)));
          auto mirror = fab.create_host_mirror();
          for (std::int64_t cell = 0; cell < patch.count; ++cell) {
            const auto& value =
                values->values.at(static_cast<std::size_t>(patch.begin + cell - domain->begin()));
            for (int c = 0; c < result->ncomp(); ++c)
              mirror(static_cast<std::size_t>(c * patch.count + cell)) =
                  static_cast<Real>(get<Dim>(element, value, "c" + std::to_string(c))->real());
          }
          fab.copy_from_host(mirror);
        }
        topology.require();
        return result;
      },
      "Field Bundle complete owned storage reconstruction");
}
}  // namespace pops::runtime::program::field_bundle
