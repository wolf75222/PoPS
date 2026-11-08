#pragma once

#include <Kokkos_BitManipulation.hpp>
#include <Kokkos_Core.hpp>
#include <pops/runtime/program/continuation_image.hpp>
#include <pops/runtime/program/finite_kokkos.hpp>

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace pops::runtime::program::finite_device {

// Storage of a replicated finite Array whose element is a Product tree of Bool,
// Int64 and Float64 leaves: one immutable column per leaf, in preorder, in the
// memory space of the default execution space (structure of arrays, DM7). Bool
// leaves are stored as unsigned char. No element is boxed and no host container
// holds the values. Every host access below is explicit and is counted into the
// Residency of the Program that performs it (DM8, PF2).

struct Residency {
  std::atomic<std::uint64_t> device_to_host_bytes{0};
  std::atomic<std::uint64_t> host_to_device_bytes{0};
  std::atomic<std::uint64_t> transfers{0};
};

inline void record(Residency* residency, std::uint64_t bytes, bool to_host) noexcept {
  if (!residency)
    return;
  (to_host ? residency->device_to_host_bytes : residency->host_to_device_bytes)
      .fetch_add(bytes, std::memory_order_relaxed);
  residency->transfers.fetch_add(1, std::memory_order_relaxed);
}

template <class T>
using HostView = Kokkos::View<T*, Kokkos::HostSpace>;
using Column =
    std::variant<finite_kokkos::View<const unsigned char>, finite_kokkos::View<const std::int64_t>,
                 finite_kokkos::View<const Real>>;
using HostColumn = std::variant<HostView<unsigned char>, HostView<std::int64_t>, HostView<Real>>;

struct Array {
  FiniteDomainRef domain;
  std::shared_ptr<const std::vector<Column>> columns;
};

// Labels let Kokkos Tools tell array data from O(1) control scalars.
inline constexpr std::string_view kArrayLabel = "pops_finite_array:";
inline constexpr std::string_view kHostLabel = "pops_finite_host:";
inline std::string relabel(std::string_view label, std::string_view prefix) {
  for (const auto old : {kArrayLabel, kHostLabel})
    if (label.substr(0, old.size()) == old)
      label.remove_prefix(old.size());
  return std::string(prefix) + std::string(label);
}

inline void require_columns(const Array& array, std::size_t count) {
  if (!array.domain || !array.columns || array.columns->size() != count)
    throw std::logic_error("finite device Array differs from its element layout");
}

template <class T>
finite_kokkos::View<const T> column(const Array& array, std::size_t index) {
  if (!array.domain || !array.columns || index >= array.columns->size())
    throw std::logic_error("finite device Array has no column " + std::to_string(index));
  const auto* view = std::get_if<finite_kokkos::View<const T>>(&(*array.columns)[index]);
  if (!view || view->extent(0) != static_cast<std::size_t>(array.domain->size))
    throw std::logic_error("finite device Array column differs from its type or domain");
  return *view;
}

inline Array publish(FiniteDomainRef domain, std::vector<Column> columns) {
  return Array{std::move(domain), std::make_shared<const std::vector<Column>>(std::move(columns))};
}

template <class T>
T read(const Array& array, std::size_t index, std::int64_t offset, Residency* residency) {
  const auto values = column<T>(array, index);
  if (offset < 0 || static_cast<std::size_t>(offset) >= values.extent(0))
    throw std::out_of_range("finite device Array read is outside its domain");
  T value{};
  Kokkos::deep_copy(value, Kokkos::subview(values, static_cast<std::size_t>(offset)));
  record(residency, sizeof(T), true);
  return value;
}

// Allocates a device column per host column and copies it once (the publication
// of a host phase). The host columns are released by the caller.
inline Array from_host(FiniteDomainRef domain, const std::vector<HostColumn>& host,
                       Residency* residency) {
  std::vector<Column> columns;
  columns.reserve(host.size());
  for (const auto& source : host)
    std::visit(
        [&](const auto& values) {
          using Value = typename std::decay_t<decltype(values)>::non_const_value_type;
          finite_kokkos::View<Value> device(
              Kokkos::view_alloc(Kokkos::WithoutInitializing, relabel(values.label(), kArrayLabel)),
              values.extent(0));
          Kokkos::deep_copy(device, values);
          record(residency, values.extent(0) * sizeof(Value), false);
          columns.emplace_back(finite_kokkos::View<const Value>(device));
        },
        source);
  return publish(std::move(domain), std::move(columns));
}

// One counted copy of a device column into a new host view.
template <class T>
HostView<T> host_copy(const finite_kokkos::View<const T>& values, Residency* residency) {
  HostView<T> copy(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, relabel(values.label(), kHostLabel)),
      values.extent(0));
  Kokkos::deep_copy(copy, values);
  record(residency, values.extent(0) * sizeof(T), true);
  return copy;
}

inline std::vector<HostColumn> to_host(const Array& array, Residency* residency) {
  if (!array.domain || !array.columns)
    throw std::logic_error("finite device Array has no storage");
  std::vector<HostColumn> host;
  host.reserve(array.columns->size());
  for (const auto& source : *array.columns)
    std::visit(
        [&](const auto& values) {
          if (values.extent(0) != static_cast<std::size_t>(array.domain->size))
            throw std::logic_error("finite device Array column differs from its domain");
          host.emplace_back(host_copy(values, residency));
        },
        source);
  return host;
}

// First failing lane of a map, reduced on the device: one control scalar.
struct FirstFailure {
  finite_kokkos::View<const unsigned char> flags;
  KOKKOS_FUNCTION void operator()(const std::int64_t index, std::int64_t& first) const {
    if (flags(index) && index < first)
      first = index;
  }
};

inline std::int64_t first_failure(const finite_kokkos::View<unsigned char>& flags,
                                  std::int64_t count) {
  if (count <= 0)
    return count;
  std::int64_t first = count;
  Kokkos::parallel_reduce(
      "pops_finite_first_failure",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<std::int64_t>>(0, count),
      FirstFailure{flags}, Kokkos::Min<std::int64_t>(first));
  return first < count ? first : count;
}

// Replication fingerprint, reduced on the device. Each term is a bijection of
// the element bits for its (column, position) key, so arrays that differ in one
// element always differ in both sums; several differences cancel in both sums
// with probability about 2^-128. Float64 and Int64/Bool columns are kept apart.
KOKKOS_INLINE_FUNCTION std::uint64_t mix_first(std::uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
KOKKOS_INLINE_FUNCTION std::uint64_t mix_second(std::uint64_t z) {
  z = (z ^ (z >> 33)) * 0xff51afd7ed558ccdULL;
  z = (z ^ (z >> 33)) * 0xc4ceb9fe1a85ec53ULL;
  return z ^ (z >> 33);
}

struct FingerprintTerms {
  std::uint64_t first = 0, second = 0;
  std::int64_t nonfinite = 0;
};

template <class T>
struct ColumnFingerprint {
  using value_type = FingerprintTerms;
  finite_kokkos::View<const T> values;
  std::uint64_t salt;
  KOKKOS_FUNCTION void init(value_type& terms) const {
    terms.first = 0;
    terms.second = 0;
    terms.nonfinite = 0;
  }
  KOKKOS_FUNCTION void join(value_type& target, const value_type& source) const {
    target.first += source.first;
    target.second += source.second;
    target.nonfinite += source.nonfinite;
  }
  KOKKOS_FUNCTION void operator()(const std::int64_t index, value_type& terms) const {
    std::uint64_t bits = 0;
    if constexpr (std::is_same_v<T, Real>) {
      bits = Kokkos::bit_cast<std::uint64_t>(values(index));
      terms.nonfinite += Kokkos::isfinite(values(index)) ? 0 : 1;
    } else
      bits = static_cast<std::uint64_t>(values(index));
    const auto key = mix_first(salt ^ (static_cast<std::uint64_t>(index) * 0x9e3779b97f4a7c15ULL));
    terms.first += mix_first(bits ^ key);
    terms.second += mix_second(bits + mix_second(key));
  }
};

struct Fingerprint {
  std::uint64_t real_first = 0, real_second = 0;
  std::uint64_t integer_first = 0, integer_second = 0;
  std::int64_t nonfinite = 0;
};

inline Fingerprint fingerprint(const Array& array) {
  if (!array.domain || !array.columns)
    throw std::logic_error("finite device Array has no storage");
  Fingerprint out;
  const auto count = array.domain->size;
  for (std::size_t j = 0; j < array.columns->size(); ++j)
    std::visit(
        [&](const auto& values) {
          using Value = typename std::decay_t<decltype(values)>::non_const_value_type;
          if (values.extent(0) != static_cast<std::size_t>(count))
            throw std::logic_error("finite device Array column differs from its domain");
          FingerprintTerms terms;
          if (count > 0)
            Kokkos::parallel_reduce(
                "pops_finite_fingerprint",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<std::int64_t>>(
                    0, count),
                ColumnFingerprint<Value>{values, mix_first(std::uint64_t(j) + 1)}, terms);
          if constexpr (std::is_same_v<Value, Real>) {
            out.real_first += terms.first;
            out.real_second += terms.second;
          } else {
            out.integer_first += terms.first;
            out.integer_second += terms.second;
          }
          out.nonfinite += terms.nonfinite;
        },
        (*array.columns)[j]);
  return out;
}

// The element layouts that use this storage: a Product tree of scalar leaves,
// or one scalar leaf. Leaves are listed in preorder, as the generator lists them.
inline bool flat_element(const ContinuationType& type) {
  std::vector<const ContinuationType*> stack{&type};
  while (!stack.empty()) {
    const auto* current = stack.back();
    stack.pop_back();
    switch (current->kind) {
      case ContinuationKind::Product:
        for (const auto& child : current->children)
          stack.push_back(&child);
        break;
      case ContinuationKind::Boolean:
      case ContinuationKind::Int64:
      case ContinuationKind::Float64:
        break;
      default:
        return false;
    }
  }
  return true;
}

inline std::vector<ContinuationKind> leaf_kinds(const ContinuationType& type) {
  std::vector<ContinuationKind> leaves;
  std::vector<const ContinuationType*> stack{&type};
  while (!stack.empty()) {
    const auto* current = stack.back();
    stack.pop_back();
    if (current->kind == ContinuationKind::Product) {
      for (auto child = current->children.rbegin(); child != current->children.rend(); ++child)
        stack.push_back(&*child);
    } else
      leaves.push_back(current->kind);
  }
  return leaves;
}

}  // namespace pops::runtime::program::finite_device
