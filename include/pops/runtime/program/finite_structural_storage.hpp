#pragma once

#include <any>
#include <array>
#include <functional>
#include <type_traits>
#include <utility>
#include <pops/runtime/program/distributed_array.hpp>

namespace pops::runtime::program::finite_structural {

// Storage adapters for already evaluated Program values. These types contain no
// numerical operations; conversion uses the installed structural type language.
using DomainRef = FiniteDomainRef;
struct Index {
  DomainRef domain;
  std::int64_t offset = 0;
};
struct Record {
  std::vector<std::any> values;
};
struct Array {
  DomainRef domain;
  std::shared_ptr<const std::vector<std::any>> values;
};

namespace detail {
template <class Result>
ContinuationFieldMetadata field_metadata(Result&& result) {
  if constexpr (std::is_convertible_v<Result, ContinuationFieldMetadata>)
    return std::forward<Result>(result);
  else {
    static_assert(std::is_convertible_v<Result, FieldExchangeExpressions::Ref>);
    return {std::forward<Result>(result), {}};
  }
}
template <class Observer, class Field, class Lifetime>
ContinuationFieldMetadata copied_field(const Observer& observer, const Field& source,
                                       const Field& destination, const Lifetime& lifetime,
                                       const ContinuationType& type) {
  if constexpr (std::is_invocable_v<Observer, const Field&, const Field&, const Lifetime&,
                                    const ContinuationType&>) {
    if constexpr (std::is_void_v<std::invoke_result_t<Observer, const Field&, const Field&,
                                                      const Lifetime&, const ContinuationType&>>)
      observer(source, destination, lifetime, type);
    else
      return field_metadata(observer(source, destination, lifetime, type));
  } else if constexpr (std::is_invocable_v<Observer, const Field&, const Field&, const Lifetime&>) {
    if constexpr (std::is_void_v<
                      std::invoke_result_t<Observer, const Field&, const Field&, const Lifetime&>>)
      observer(source, destination, lifetime);
    else
      return field_metadata(observer(source, destination, lifetime));
  } else
    observer(source, destination);
  return {};
}
template <class Observer, class Field, class Lifetime>
void copied_field(const Observer& observer, const Field& source, const Field& destination,
                  const Lifetime& lifetime, const ContinuationFieldMetadata& metadata) {
  if constexpr (std::is_invocable_v<Observer, const Field&, const Field&, const Lifetime&,
                                    const ContinuationFieldMetadata&>)
    observer(source, destination, lifetime, metadata);
  else {
    if (metadata.support || metadata.representation)
      throw std::invalid_argument("Field copy observer cannot preserve retained producer support");
    if constexpr (std::is_invocable_v<Observer, const Field&, const Field&, const Lifetime&,
                                      const FieldExchangeExpressions::Ref&>)
      observer(source, destination, lifetime, metadata.expression);
    else {
      if (metadata.expression)
        throw std::invalid_argument(
            "Field copy observer cannot preserve retained exchange provenance");
      if constexpr (std::is_invocable_v<Observer, const Field&, const Field&, const Lifetime&>)
        observer(source, destination, lifetime);
      else
        observer(source, destination);
    }
  }
}

struct Buffer {
  std::vector<std::any> values;
  Buffer* next = nullptr;
  static void retire(Buffer* value) noexcept {
    static thread_local Buffer* pending = nullptr;
    static thread_local bool draining = false;
    value->next = pending;
    pending = value;
    if (draining)
      return;
    draining = true;
    while (pending) {
      auto* current = pending;
      pending = current->next;
      delete current;  // Nested Array owners enqueue instead of recursively deleting buffers.
    }
    draining = false;
  }
};
}  // namespace detail

inline std::shared_ptr<std::vector<std::any>> make_values(std::vector<std::any> values) {
  auto owner = std::shared_ptr<detail::Buffer>(new detail::Buffer{std::move(values)},
                                               &detail::Buffer::retire);
  auto* buffer = &owner->values;
  return {std::move(owner), buffer};
}
inline std::shared_ptr<std::vector<std::any>> make_values(std::size_t count) {
  return make_values(std::vector<std::any>(count));
}

// Preparation cells borrow existing values but own expression temporaries. In particular,
// borrowing Requests never duplicates its index vector before the declared source budget.
template <class T>
std::any source_cell(T&& value) {
  if constexpr (std::is_lvalue_reference_v<T>)
    return std::any(std::cref(value));
  else
    return std::any(std::forward<T>(value));
}

template <class T>
const T& held(const std::any& cell) {
  if (const auto* borrowed = std::any_cast<std::reference_wrapper<const T>>(&cell))
    return borrowed->get();
  return std::any_cast<const T&>(cell);
}

// Separate bounds avoid hiding temporary/source memory in the published image
// capacity. Counts cover requested payload storage, not allocator bookkeeping.
struct Limits {
  std::uint64_t source_bytes;
  std::uint64_t destination_bytes;
  std::uint64_t workspace_bytes;
};

template <int Dim>
const ContinuationValueRef<Dim>& project(const ContinuationValueRef<Dim>& root,
                                         std::span<const std::size_t> path,
                                         std::uint64_t capacity) {
  if (path.size() > capacity / sizeof(std::size_t))
    throw std::length_error("finite structural projection exceeds declared workspace");
  const auto* current = &root;
  for (const auto member : path) {
    if (!*current)
      throw std::invalid_argument("finite structural projection has a missing value");
    current = &(*current)->members().at(member);
  }
  if (!*current)
    throw std::invalid_argument("finite structural projection has a missing value");
  return *current;
}

namespace detail {
class Budget {
 public:
  explicit Budget(std::uint64_t bytes) : remaining_(bytes) {}
  void charge(std::size_t count, std::size_t width) {
    if (width && count > remaining_ / width)
      throw std::length_error("finite structural conversion exceeds declared storage budget");
    remaining_ -= count * width;
  }

 private:
  std::uint64_t remaining_;
};

struct Layout {
  const ContinuationType* type;
  std::vector<std::size_t> children;
  std::vector<std::size_t> offsets;
  std::size_t slots = 1;
};

inline std::vector<Layout> layout(const ContinuationType& type, Budget& budget) {
  budget.charge(1, sizeof(Layout));
  std::vector<Layout> nodes{{&type, {}, {}, 1}};
  // Appending children produces a topological order, independent of depth.
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const auto* current = nodes[i].type;
    budget.charge(current->children.size(), sizeof(Layout) + 2 * sizeof(std::size_t));
    nodes[i].children.reserve(current->children.size());
    nodes[i].offsets.resize(current->children.size());
    for (const auto& child : current->children) {
      const auto index = nodes.size();
      nodes[i].children.push_back(index);
      nodes.push_back({&child, {}, {}, 1});
    }
  }
  for (std::size_t i = nodes.size(); i != 0; --i) {
    auto& node = nodes[i - 1];
    if (node.type->kind != ContinuationKind::Product &&
        node.type->kind != ContinuationKind::Variant)
      continue;
    node.slots = node.type->kind == ContinuationKind::Variant ? 1 : 0;
    for (std::size_t c = 0; c < node.children.size(); ++c) {
      node.offsets[c] = node.slots;
      const auto count = nodes[node.children[c]].slots;
      if (count > std::numeric_limits<std::size_t>::max() - node.slots)
        throw std::length_error("finite structural layout slot count overflows");
      node.slots += count;
    }
  }
  return nodes;
}

inline bool record_kind(ContinuationKind kind) noexcept {
  return kind == ContinuationKind::Product || kind == ContinuationKind::Variant;
}
template <int Dim>
void field_storage(const MultiFab<Dim>& field, Budget& source, Budget& destination) {
  for (std::size_t p = 0; p < field.local_size(); ++p) {
    source.charge(field.fab(p).size(), sizeof(Real));
    destination.charge(field.fab(p).size(), sizeof(Real));
  }
}
inline void domain(const DomainRef& value, const ContinuationType& type) {
  continuation_detail::domain(value, type);
}
inline void domain(const distributed::DomainRef& value, const ContinuationType& type) {
  if (!value || value->family() != type.family || value->capacity() != type.capacity)
    throw std::invalid_argument("finite structural distributed domain differs from its type");
  value->require_live();
}

template <int Dim, class FieldCopy, class FieldPreflight>
ContinuationValueRef<Dim> pack_leaf(const ContinuationType& type, const std::any& source,
                                    Budget& input, Budget& output, const FieldCopy& on_field_copy,
                                    const FieldPreflight& preflight) {
  using Value = ContinuationValue<Dim>;
  switch (type.kind) {
    case ContinuationKind::Boolean:
      return Value::boolean(std::any_cast<bool>(source));
    case ContinuationKind::Int64:
      return Value::integer(std::any_cast<std::int64_t>(source));
    case ContinuationKind::Float64:
      static_assert(sizeof(Real) == sizeof(double));
      return Value::real(std::any_cast<Real>(source));
    case ContinuationKind::Domain: {
      const auto& value = held<DomainRef>(source);
      domain(value, type);
      return Value::domain(value);
    }
    case ContinuationKind::Index: {
      const auto& value = held<Index>(source);
      domain(value.domain, type);
      if (value.offset < 0 || value.offset >= value.domain->size)
        throw std::invalid_argument("finite structural Index is outside its domain");
      return Value::index(value.domain, value.offset);
    }
    case ContinuationKind::DistributedDomain: {
      const auto& value = held<distributed::DomainRef>(source);
      domain(value, type);
      return Value::distributed_domain(value);
    }
    case ContinuationKind::DistributedIndex: {
      const auto& value = held<distributed::Index>(source);
      domain(value.domain, type);
      return Value::distributed_index(value.domain, value.gid);
    }
    case ContinuationKind::DistributedArray: {
      distributed::ArrayRef<Dim> value;
      if (type.children.at(0).kind == ContinuationKind::DistributedIndex) {
        const auto& requests = held<distributed::Requests>(source);
        input.charge(requests.indices.size(), sizeof(distributed::Index));
        value = distributed::requests_array<Dim>(requests);
      } else
        value = held<distributed::ArrayRef<Dim>>(source);
      if (!value || value->element != type.children.at(0))
        throw std::invalid_argument("finite structural distributed Array element type differs");
      domain(value->domain, type);
      // Preserve the published ArrayRef/version and target witness, never republish its shard.
      return Value::distributed_array(std::move(value));
    }
    case ContinuationKind::Field: {
      const auto* borrowed = std::any_cast<std::reference_wrapper<const MultiFab<Dim>>>(&source);
      const auto* value = borrowed
                              ? &borrowed->get()
                              : std::any_cast<const std::shared_ptr<MultiFab<Dim>>&>(source).get();
      if (!value || value->ncomp() != type.field_components)
        throw std::invalid_argument("finite structural Field component count differs");
      preflight(*value, type);
      field_storage(*value, input, output);
      auto copied = Value::field(*value, type.field_owner, type.scientific_contract, {}, {},
                                 type.field_provider);
      auto metadata =
          copied_field(on_field_copy, *value, copied->field(), copied->field_storage_owner(), type);
      return Value::with_field_metadata(copied, std::move(metadata));
    }
    default:
      throw std::invalid_argument("finite structural leaf has a container type");
  }
}
}  // namespace detail

struct IgnoreFieldCopy {
  template <class Field>
  void operator()(const Field&, const Field&) const noexcept {}
};

struct LegacyFieldPreflight {
  template <class Field, class... Metadata>
  void operator()(const Field&, const ContinuationType& type, const Metadata&...) const {
    if (!type.field_provider.empty())
      throw std::invalid_argument("independent Field requires a live source preflight");
  }
};

template <int Dim, class FieldCopy = IgnoreFieldCopy, class FieldPreflight = LegacyFieldPreflight>
ContinuationValueRef<Dim> pack(const ContinuationType& type, const std::any& source, Limits limits,
                               const FieldCopy& on_field_copy = {},
                               const FieldPreflight& preflight = {}) {
  using Ref = ContinuationValueRef<Dim>;
  using Value = ContinuationValue<Dim>;
  type.validate(limits.workspace_bytes);
  detail::Budget input(limits.source_bytes), output(limits.destination_bytes),
      work(limits.workspace_bytes);
  const auto nodes = detail::layout(type, work);
  struct Frame {
    std::size_t node, offset = 0, next = 0, arm = 0, count = 0;
    const std::any* source = nullptr;
    const Record* record = nullptr;
    const Array* array = nullptr;
    std::vector<Ref> children;
    Ref leaf;
  };
  std::vector<Frame> stack;
  auto push = [&](std::size_t index, const std::any* cell, const Record* record,
                  std::size_t offset) {
    // Cumulative accounting also bounds total work, including repeated aliases.
    work.charge(1, sizeof(Frame));
    input.charge(1, sizeof(std::any));
    output.charge(1, sizeof(Value));
    const auto& node = nodes[index];
    const auto& schema = *node.type;
    Frame frame{index, offset, 0, 0, 0, cell, record, nullptr, {}, {}};
    if (detail::record_kind(schema.kind)) {
      if (!record) {
        frame.record = &held<Record>(*cell);
        input.charge(frame.record->values.size(), sizeof(std::any));
        if (frame.record->values.size() != node.slots)
          throw std::invalid_argument("finite structural Record has the wrong slot count");
      }
      if (schema.kind == ContinuationKind::Variant) {
        const auto arm = std::any_cast<std::int64_t>(frame.record->values.at(offset));
        if (arm < 0 || static_cast<std::size_t>(arm) >= node.children.size())
          throw std::invalid_argument("finite structural Variant has an invalid discriminant");
        frame.arm = static_cast<std::size_t>(arm);
        frame.count = 1;
      } else
        frame.count = node.children.size();
    } else {
      if (record)
        frame.source = &record->values.at(offset);
      if (schema.kind == ContinuationKind::Array) {
        frame.array = &held<Array>(*frame.source);
        detail::domain(frame.array->domain, schema);
        if (!frame.array->values ||
            frame.array->values->size() != std::size_t(frame.array->domain->size))
          throw std::invalid_argument(
              "finite structural Array differs from its domain cardinality");
        frame.count = frame.array->values->size();
        input.charge(frame.count, sizeof(std::any));
      } else
        frame.leaf =
            detail::pack_leaf<Dim>(schema, *frame.source, input, output, on_field_copy, preflight);
    }
    output.charge(frame.count, sizeof(Ref));
    frame.children.reserve(frame.count);
    stack.push_back(std::move(frame));
  };
  push(0, &source, nullptr, 0);
  for (;;) {
    auto& frame = stack.back();
    const auto& node = nodes[frame.node];
    if (frame.next < frame.count) {
      const auto n = frame.next++;
      if (frame.array)
        push(node.children[0], &frame.array->values->at(n), nullptr, 0);
      else {
        const auto selected = node.type->kind == ContinuationKind::Variant ? frame.arm : n;
        push(node.children[selected], nullptr, frame.record, frame.offset + node.offsets[selected]);
      }
      continue;
    }
    Ref result = std::move(frame.leaf);
    if (!result) {
      if (frame.array)
        result = Value::array(frame.array->domain, std::move(frame.children));
      else if (node.type->kind == ContinuationKind::Variant)
        result =
            Value::variant(static_cast<std::uint32_t>(frame.arm), std::move(frame.children[0]));
      else
        result = Value::product(std::move(frame.children));
    }
    stack.pop_back();
    if (stack.empty())
      return result;
    stack.back().children.push_back(std::move(result));
  }
}

template <int Dim, class FieldCopy = IgnoreFieldCopy, class FieldPreflight = LegacyFieldPreflight>
std::any unpack(const ContinuationType& type, const ContinuationValueRef<Dim>& source,
                Limits limits, const FieldCopy& on_field_copy = {},
                const FieldPreflight& preflight = {}) {
  using Ref = ContinuationValueRef<Dim>;
  using Value = ContinuationValue<Dim>;
  type.validate(limits.workspace_bytes);
  detail::Budget input(limits.source_bytes), output(limits.destination_bytes),
      work(limits.workspace_bytes);
  const auto nodes = detail::layout(type, work);
  struct Task {
    std::size_t node, offset;
    const Ref* source;
    std::any* destination;
    Record* record;
  };
  std::any result;
  std::vector<Task> tasks;
  auto push = [&](Task task) {
    work.charge(1, sizeof(Task));
    tasks.push_back(task);
  };
  push({0, 0, &source, &result, nullptr});
  while (!tasks.empty()) {
    auto task = tasks.back();
    tasks.pop_back();
    input.charge(1, sizeof(Value));
    const auto& node = nodes[task.node];
    const auto& schema = *node.type;
    const auto& value = *task.source;
    if (!value || value->kind() != schema.kind)
      throw std::invalid_argument("finite structural value differs from its type");
    if (schema.kind == ContinuationKind::Product || schema.kind == ContinuationKind::Variant ||
        schema.kind == ContinuationKind::Array)
      input.charge(value->members().size(), sizeof(Ref));
    if (detail::record_kind(schema.kind)) {
      if (!task.record) {
        output.charge(node.slots, sizeof(std::any));
        *task.destination = Record{std::vector<std::any>(node.slots)};
        task.record = &std::any_cast<Record&>(*task.destination);
      }
      if (schema.kind == ContinuationKind::Variant) {
        const auto arm = value->arm();
        if (arm >= node.children.size() || value->members().size() != 1)
          throw std::invalid_argument("finite structural Variant has an invalid active arm");
        task.record->values.at(task.offset) = std::int64_t(arm);
        push({node.children[arm], task.offset + node.offsets[arm], &value->members()[0], nullptr,
              task.record});
      } else {
        if (value->members().size() != node.children.size())
          throw std::invalid_argument("finite structural Product has an invalid member count");
        for (std::size_t n = node.children.size(); n != 0; --n)
          push({node.children[n - 1], task.offset + node.offsets[n - 1], &value->members()[n - 1],
                nullptr, task.record});
      }
      continue;
    }
    auto& destination = task.record ? task.record->values.at(task.offset) : *task.destination;
    output.charge(1, sizeof(std::any));
    switch (schema.kind) {
      case ContinuationKind::Boolean:
        destination = value->boolean();
        break;
      case ContinuationKind::Int64:
        destination = value->integer();
        break;
      case ContinuationKind::Float64:
        destination = Real(value->real());
        break;
      case ContinuationKind::Domain:
        detail::domain(value->domain_ref(), schema);
        destination = value->domain_ref();
        break;
      case ContinuationKind::Index:
        detail::domain(value->domain_ref(), schema);
        if (value->index_offset() < 0 || value->index_offset() >= value->domain_ref()->size)
          throw std::invalid_argument("finite structural Index is outside its domain");
        destination = Index{value->domain_ref(), value->index_offset()};
        break;
      case ContinuationKind::DistributedDomain:
        detail::domain(value->distributed_domain_ref(), schema);
        destination = value->distributed_domain_ref();
        break;
      case ContinuationKind::DistributedIndex:
        detail::domain(value->distributed_domain_ref(), schema);
        destination =
            distributed::index(value->distributed_domain_ref(), value->distributed_index_offset());
        break;
      case ContinuationKind::DistributedArray: {
        const auto& array = value->distributed_array_ref();
        detail::domain(value->distributed_domain_ref(), schema);
        if (!array || array->element != schema.children.at(0))
          throw std::invalid_argument("finite structural distributed Array element type differs");
        if (schema.children[0].kind == ContinuationKind::DistributedIndex) {
          output.charge(array->values.size(), sizeof(distributed::Index));
          destination = distributed::requests_from_array<Dim>(array);
        } else
          destination = array;
        break;
      }
      case ContinuationKind::Field:
        // Authoring projections carry the original capability's exact role.
        // A different schema cannot turn copying storage into relabelling authority.
        if (value->field_owner() != schema.field_owner ||
            value->field_provider() != schema.field_provider ||
            value->scientific_contract() != schema.scientific_contract ||
            value->field().ncomp() != schema.field_components)
          throw std::invalid_argument("finite structural Field contract differs");
        preflight(value->field(), schema, value->field_metadata());
        detail::field_storage(value->field(), input, output);
        destination = std::make_shared<MultiFab<Dim>>(value->field());
        detail::copied_field(on_field_copy, value->field(),
                             *std::any_cast<const std::shared_ptr<MultiFab<Dim>>&>(destination),
                             std::any_cast<const std::shared_ptr<MultiFab<Dim>>&>(destination),
                             value->field_metadata());
        break;
      case ContinuationKind::Array: {
        detail::domain(value->domain_ref(), schema);
        const auto count = value->members().size();
        if (count != std::size_t(value->domain_ref()->size))
          throw std::invalid_argument(
              "finite structural Array differs from its domain cardinality");
        output.charge(count, sizeof(std::any));
        auto buffer = make_values(count);
        destination = Array{value->domain_ref(), buffer};
        for (std::size_t n = count; n != 0; --n)
          push({node.children[0], 0, &value->members()[n - 1], &(*buffer)[n - 1], nullptr});
        break;
      }
      default:
        throw std::invalid_argument("finite structural projection has an invalid type");
    }
  }
  return result;
}

}  // namespace pops::runtime::program::finite_structural
