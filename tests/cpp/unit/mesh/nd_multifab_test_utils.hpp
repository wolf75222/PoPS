#pragma once

#include <pops/mesh/layout/nd/distribution.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace pops::test::nd {

template <int Dim>
using HostMultiFab = MultiFab<Dim, Kokkos::HostSpace>;

template <int Dim>
Index<Dim> index_from_ordinal(const Box<Dim>& box, std::size_t ordinal) {
  Index<Dim> index{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t extent = static_cast<std::size_t>(box.length(axis));
    index[axis] = box.lo[axis] + static_cast<int>(ordinal % extent);
    ordinal /= extent;
  }
  return index;
}

template <int Dim>
std::size_t offset(const Box<Dim>& grown, const Index<Dim>& index, int component) {
  std::size_t cell = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    cell += static_cast<std::size_t>(index[axis] - grown.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(grown.length(axis));
  }
  return static_cast<std::size_t>(component) * stride + cell;
}

template <int Dim>
Real encoded_value(const Index<Dim>& index, int component = 0) {
  Real result = static_cast<Real>(10000 * component);
  Real scale = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    result += scale * static_cast<Real>(index[axis]);
    scale *= 97;
  }
  return result;
}

template <int Dim>
void fill_valid_encoded(HostMultiFab<Dim>& fields, Real ghost_value) {
  for (std::size_t local = 0; local < fields.local_size(); ++local) {
    auto& fab = fields.fab(local);
    auto host = fab.create_host_mirror();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t cells = static_cast<std::size_t>(grown.numPts());
    for (int component = 0; component < fab.ncomp(); ++component)
      for (std::size_t cell = 0; cell < cells; ++cell) {
        const Index<Dim> index = index_from_ordinal(grown, cell);
        host(offset(grown, index, component)) =
            fab.box().contains(index) ? encoded_value(index, component) : ghost_value;
      }
    fab.copy_from_host(host);
  }
}

template <int Dim, class Value>
void fill_valid(HostMultiFab<Dim>& fields, Real ghost_value, Value value) {
  for (std::size_t local = 0; local < fields.local_size(); ++local) {
    auto& fab = fields.fab(local);
    auto host = fab.create_host_mirror();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t cells = static_cast<std::size_t>(grown.numPts());
    for (int component = 0; component < fab.ncomp(); ++component)
      for (std::size_t cell = 0; cell < cells; ++cell) {
        const Index<Dim> index = index_from_ordinal(grown, cell);
        host(offset(grown, index, component)) =
            fab.box().contains(index) ? value(index, component) : ghost_value;
      }
    fab.copy_from_host(host);
  }
}

template <int Dim>
Real value_at(const HostMultiFab<Dim>& fields, std::size_t global_box, const Index<Dim>& index,
              int component = 0) {
  const auto& fab = fields.fab_global(global_box);
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  return host(offset(fab.grown_box(), index, component));
}

template <int Dim>
std::vector<Real> snapshot(const HostMultiFab<Dim>& fields) {
  std::vector<Real> result;
  for (std::size_t local = 0; local < fields.local_size(); ++local) {
    const auto& fab = fields.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t element = 0; element < host.size(); ++element)
      result.push_back(host(element));
  }
  return result;
}

template <int Dim>
mesh::RankSpace<Dim> one_rank_space() {
  Extent<Dim> extent{};
  for (int axis = 0; axis < Dim; ++axis)
    extent[axis] = 1;
  return mesh::RankSpace<Dim>{Index<Dim>{}, extent};
}

template <int Dim>
std::array<int, Dim> axis_sizes(int axis_zero, int other) {
  std::array<int, Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = axis == 0 ? axis_zero : other;
  return result;
}

template <int Dim>
Box<Dim> cube(int lower, int upper) {
  Index<Dim> lo{};
  Index<Dim> hi{};
  for (int axis = 0; axis < Dim; ++axis) {
    lo[axis] = lower;
    hi[axis] = upper;
  }
  return Box<Dim>{lo, hi};
}

template <int Dim>
Extent<Dim> uniform_extent(std::int64_t value) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

}  // namespace pops::test::nd
