#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/storage/field_view.hpp>

#include <Kokkos_Array.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pops::runtime::program {

inline int field_read_capture_components(std::size_t bindings) {
  constexpr auto max = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (bindings > (max - 1) / 2)
    throw std::length_error("Field read journal component count is not representable");
  return static_cast<int>(1 + 2 * bindings);
}

inline std::uint64_t field_read_capture_bytes(std::uint64_t cells, std::size_t bindings,
                                              std::uint64_t byte_capacity) {
  const auto components = static_cast<std::uint64_t>(field_read_capture_components(bindings));
  if (cells > byte_capacity / sizeof(Real) / components)
    throw std::length_error("Field read journal exceeds its workspace byte capacity");
  return cells * components * sizeof(Real);
}

// This is a device journal, not an authority or a numerical evaluator. The
// caller owns its storage and authenticates the schema, support and allocation
// budget. One kernel invocation owns one cell; no atomic operation or extra
// source read is needed. Component 0 is activity/status (0=inactive, 1=active,
// -1=invalid), followed by (was_read, exact_value) for each declared free
// input.
template <int Dim>
struct FieldReadCapture {
  FieldView<Real, Dim> storage;
  Index<Dim> cell;
  int count = 0;

  POPS_HD bool consistent() const {
    return storage.ncomp > 0 && count >= 0 && (storage.ncomp - 1) % 2 == 0 &&
           count == (storage.ncomp - 1) / 2;
  }

  POPS_HD void initialize() const {
    for (int component = 0; component < storage.ncomp; ++component)
      storage(cell, component) = Real(0);
    if (!consistent())
      invalidate();
  }

  POPS_HD void activate() const {
    if (consistent() && storage(cell, 0) == Real(0))
      storage(cell, 0) = Real(1);
    else
      invalidate();
  }

  POPS_HD void invalidate() const {
    if (storage.ncomp > 0)
      storage(cell, 0) = Real(-1);
  }

  POPS_HD bool valid() const { return consistent() && storage(cell, 0) >= Real(0); }

  POPS_HD bool active() const { return consistent() && storage(cell, 0) == Real(1); }

  POPS_HD void record(int slot, Real value) const {
    if (!active() || slot < 0 || slot >= count) {
      invalidate();
      return;
    }
    const int flag = 1 + 2 * slot;
    if (storage(cell, flag) != Real(0)) {
      // Immutable inputs must give the same bits on every actual read. Signed
      // zero is significant. Byte-wise comparison is device-safe and avoids an
      // aliasing cast or a floating-point equivalence test.
      const Real previous = storage(cell, flag + 1);
      const auto* a = reinterpret_cast<const unsigned char*>(&previous);
      const auto* b = reinterpret_cast<const unsigned char*>(&value);
      for (std::size_t byte = 0; byte < sizeof(Real); ++byte)
        if (a[byte] != b[byte]) {
          invalidate();
          return;
        }
    } else {
      storage(cell, flag) = Real(1);
      storage(cell, flag + 1) = value;
    }
  }
};

template <class Parameters, std::size_t N, int Dim>
struct CapturedParameterReads {
  Parameters source;
  Kokkos::Array<int, N> source_indices;
  FieldReadCapture<Dim> journal;

  POPS_HD Real get(int index) const {
    if (!journal.active()) {
      journal.invalidate();
      return Real(0);
    }
    std::size_t first = 0, last = N;
    while (first < last) {
      const std::size_t middle = first + (last - first) / 2;
      if (source_indices[middle] < index)
        first = middle + 1;
      else
        last = middle;
    }
    if (first == N || first >= static_cast<std::size_t>(journal.count) ||
        source_indices[first] != index) {
      journal.invalidate();
      return Real(0);  // Unknown generated read never reaches the source view.
    }
    const Real value = source.get(index);
    journal.record(static_cast<int>(first), value);
    return value;
  }
};

template <class Providers, int Dim>
struct CapturedProviderReads {
  Providers source;
  FieldReadCapture<Dim> journal;
  int first_slot = 0;
  int count = 0;

  POPS_HD Real operator()(const Index<Dim>& cell, int slot) const {
    if (!journal.active() || first_slot < 0 || first_slot > journal.count || count < 0 ||
        count > journal.count - first_slot || slot < 0 || slot >= count) {
      journal.invalidate();
      return Real(0);
    }
    for (int axis = 0; axis < Dim; ++axis)
      if (cell[axis] != journal.cell[axis]) {
        journal.invalidate();
        return Real(0);
      }
    const Real value = source(cell, slot);
    journal.record(first_slot + slot, value);
    return value;
  }
};

}  // namespace pops::runtime::program
