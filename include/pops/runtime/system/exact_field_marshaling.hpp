/// @file
/// @brief Collective, exact-ranked marshaling for System-owned distributed fields.

#pragma once

#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::runtime::system::marshaling {

template <int Dim, class MemorySpace>
bool contributes_collective_payload(const MultiFab<Dim, MemorySpace>& field) {
  return !field.distribution().replicated() || field.local_rank() == field.rank_space().origin();
}

template <class Integer>
void append_integral_bytes(std::string& output, Integer value) {
  static_assert(std::is_integral_v<Integer> || std::is_enum_v<Integer>);
  output.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <int Dim, class MemorySpace>
std::string exact_field_contract_bytes(const MultiFab<Dim, MemorySpace>& field,
                                       const Box<Dim>& domain) {
  std::string result;
  append_integral_bytes(result, Dim);
  append_integral_bytes(result, field.ncomp());
  for (int axis = 0; axis < Dim; ++axis) {
    append_integral_bytes(result, domain.lo[axis]);
    append_integral_bytes(result, domain.hi[axis]);
    append_integral_bytes(result, field.ghosts()[axis]);
    append_integral_bytes(result, field.rank_space().origin()[axis]);
    append_integral_bytes(result, field.rank_space().extent()[axis]);
  }
  append_integral_bytes(result, static_cast<std::uint64_t>(field.layout().size()));
  for (std::size_t patch = 0; patch < field.layout().size(); ++patch)
    for (int axis = 0; axis < Dim; ++axis) {
      append_integral_bytes(result, field.layout()[patch].lo[axis]);
      append_integral_bytes(result, field.layout()[patch].hi[axis]);
    }
  append_integral_bytes(result, field.distribution().mode());
  append_integral_bytes(result, static_cast<std::uint64_t>(field.distribution().owners().size()));
  for (const Index<Dim>& owner : field.distribution().owners())
    for (int axis = 0; axis < Dim; ++axis)
      append_integral_bytes(result, owner[axis]);
  return result;
}

template <int Dim, class MemorySpace>
void require_collective_field_contract(const MultiFab<Dim, MemorySpace>& field,
                                       const Box<Dim>& domain) {
  std::string contract;
  long local_failure = 0;
  try {
    contract = exact_field_contract_bytes(field, domain);
  } catch (...) {
    local_failure = 1;
  }
  if (all_reduce_max(local_failure) != 0)
    throw std::runtime_error(
        "exact field marshaling contract could not be encoded on at least one MPI rank");
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("pops.exact-field-marshaling.contract.v1"),
            std::string_view(contract)}}))
    throw std::invalid_argument("exact field marshaling contract differs between MPI ranks");
}

template <int Dim>
std::size_t checked_cell_count(const Box<Dim>& domain) {
  const std::int64_t cells = domain.numPts();
  if (cells < 0 || static_cast<std::uint64_t>(cells) >
                       static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error("exact field cell count exceeds size_t");
  return static_cast<std::size_t>(cells);
}

template <int Dim>
std::size_t domain_ordinal(const Box<Dim>& domain, const Index<Dim>& index) {
  std::size_t ordinal = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    if (index[axis] < domain.lo[axis] || index[axis] > domain.hi[axis])
      throw std::out_of_range("exact field index lies outside its declared domain");
    ordinal += static_cast<std::size_t>(index[axis] - domain.lo[axis]) * stride;
    const std::size_t extent = static_cast<std::size_t>(domain.length(axis));
    if (axis + 1 < Dim && extent > std::numeric_limits<std::size_t>::max() / stride)
      throw std::overflow_error("exact field linear stride exceeds size_t");
    stride *= extent;
  }
  return ordinal;
}

template <int Dim, class Function>
void for_each_host_index(const Box<Dim>& box, Function&& function) {
  const std::size_t cells = checked_cell_count(box);
  for (std::size_t linear = 0; linear < cells; ++linear) {
    std::size_t remainder = linear;
    Index<Dim> index{};
    for (int axis = 0; axis < Dim; ++axis) {
      const std::size_t extent = static_cast<std::size_t>(box.length(axis));
      index[axis] = box.lo[axis] + static_cast<int>(remainder % extent);
      remainder /= extent;
    }
    function(index, linear);
  }
}

template <int Dim, class MemorySpace>
std::size_t storage_ordinal(const Fab<Dim, MemorySpace>& fab, const Index<Dim>& index,
                            int component) {
  if (component < 0 || component >= fab.ncomp())
    throw std::out_of_range("exact field component lies outside its storage");
  const Box<Dim>& grown = fab.grown_box();
  std::size_t cell = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    if (index[axis] < grown.lo[axis] || index[axis] > grown.hi[axis])
      throw std::out_of_range("exact field index lies outside its storage");
    cell += static_cast<std::size_t>(index[axis] - grown.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(grown.length(axis));
  }
  return static_cast<std::size_t>(component) * checked_cell_count(grown) + cell;
}

template <int Dim, class MemorySpace>
void require_exact_domain_decomposition(const MultiFab<Dim, MemorySpace>& field,
                                        const Box<Dim>& domain) {
  require_collective_field_contract(field, domain);
  long local_invalid = 0;
  try {
    const std::size_t patches = field.layout().size();
    std::size_t overlap_pairs = 0;
    if (patches > 1) {
      if (patches > std::numeric_limits<std::size_t>::max() / (patches - 1))
        throw std::length_error("exact field decomposition validation budget exceeds size_t");
      overlap_pairs = patches * (patches - 1) / 2;
    }
    const mesh::BoxArrayValidationBudget budget{patches, overlap_pairs};
    if (field.rank_space().size() != static_cast<std::size_t>(n_ranks()) ||
        field.rank_space().linear_rank(field.local_rank()) != static_cast<std::size_t>(my_rank()) ||
        !field.layout().tiles_exactly(domain, budget))
      local_invalid = 1;
  } catch (...) {
    local_invalid = 1;
  }
  if (all_reduce_max(local_invalid) != 0)
    throw std::runtime_error(
        "exact field decomposition must tile the domain once and map one rank coordinate per MPI "
        "process");
}

template <int Dim, class MemorySpace>
std::vector<double> gather_global(const MultiFab<Dim, MemorySpace>& field, const Box<Dim>& domain,
                                  int components) {
  // Authenticate the decomposition before allocating or reading a payload whose collective size
  // would otherwise be rank-dependent.
  require_exact_domain_decomposition(field, domain);
  const long requested_components = static_cast<long>(components);
  if (all_reduce_min(requested_components) != all_reduce_max(requested_components))
    throw std::invalid_argument("exact global gather component count differs between MPI ranks");
  const long local_shape_invalid = components < 1 || components > field.ncomp() ? 1L : 0L;
  if (all_reduce_max(local_shape_invalid) != 0)
    throw std::invalid_argument("exact global gather component count is invalid");
  const std::size_t cells = checked_cell_count(domain);
  if (cells != 0 &&
      static_cast<std::size_t>(components) > std::numeric_limits<std::size_t>::max() / cells)
    throw std::overflow_error("exact global gather size exceeds size_t");

  std::vector<double> result;
  long local_failure = 0;
  try {
    result.assign(static_cast<std::size_t>(components) * cells, 0.0);
    if (contributes_collective_payload(field))
      for (std::size_t local = 0; local < field.local_size(); ++local) {
        const Fab<Dim, MemorySpace>& fab = field.fab(local);
        auto host = fab.create_host_mirror();
        fab.copy_to_host(host);
        for_each_host_index(fab.box(), [&](const Index<Dim>& index, std::size_t) {
          const std::size_t global = domain_ordinal(domain, index);
          for (int component = 0; component < components; ++component)
            result[static_cast<std::size_t>(component) * cells + global] =
                static_cast<double>(host(storage_ordinal(fab, index, component)));
        });
      }
  } catch (...) {
    local_failure = 1;
  }
  if (all_reduce_max(local_failure) != 0)
    throw std::runtime_error("exact global field gather failed on at least one MPI rank");
  all_reduce_sum_inplace(result.data(), result.size());
  return result;
}

template <int Dim, class MemorySpace>
std::vector<double> gather_local_compact(const MultiFab<Dim, MemorySpace>& field, int components) {
  if (components < 1 || components > field.ncomp())
    throw std::invalid_argument("exact local gather component count is invalid");
  std::size_t cells = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const std::size_t local_cells = checked_cell_count(field.box(local));
    if (local_cells > std::numeric_limits<std::size_t>::max() - cells)
      throw std::overflow_error("exact local gather cell count exceeds size_t");
    cells += local_cells;
  }
  if (cells != 0 &&
      static_cast<std::size_t>(components) > std::numeric_limits<std::size_t>::max() / cells)
    throw std::overflow_error("exact local gather size exceeds size_t");
  std::vector<double> result(static_cast<std::size_t>(components) * cells, 0.0);
  std::size_t base = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const Fab<Dim, MemorySpace>& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const std::size_t local_cells = checked_cell_count(fab.box());
    for_each_host_index(fab.box(), [&](const Index<Dim>& index, std::size_t linear) {
      for (int component = 0; component < components; ++component)
        result[static_cast<std::size_t>(component) * cells + base + linear] =
            static_cast<double>(host(storage_ordinal(fab, index, component)));
    });
    base += local_cells;
  }
  return result;
}

template <int Dim, class MemorySpace>
void write_global(MultiFab<Dim, MemorySpace>& field, const Box<Dim>& domain,
                  const std::vector<double>& values, int components) {
  // Authenticate the exact decomposition before inspecting a payload whose expected size depends
  // on it. This also makes domain/layout disagreement a uniform collective failure.
  require_exact_domain_decomposition(field, domain);
  const std::size_t cells = checked_cell_count(domain);
  const bool multiplication_overflows =
      components > 0 && cells != 0 &&
      static_cast<std::size_t>(components) > std::numeric_limits<std::size_t>::max() / cells;
  const long local_shape_invalid =
      components < 1 || components > field.ncomp() || multiplication_overflows ||
              (!multiplication_overflows &&
               values.size() != static_cast<std::size_t>(components) * cells)
          ? 1L
          : 0L;
  if (all_reduce_max(local_shape_invalid) != 0)
    throw std::invalid_argument("exact global field payload has the wrong shape");

  if (values.size() > std::numeric_limits<std::size_t>::max() / sizeof(double))
    throw std::length_error("exact global field payload byte count exceeds size_t");
  const std::string shape = std::to_string(components) + ":" + std::to_string(values.size());
  const std::string_view payload(reinterpret_cast<const char*>(values.data()),
                                 values.size() * sizeof(double));
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("pops.exact-field-marshaling.write-shape.v1"), shape},
           {std::string_view("pops.exact-field-marshaling.write-payload.v1"), payload}}))
    throw std::invalid_argument("exact global field restore payload differs between MPI ranks");

  // Collective structural and payload preflight completed before the first resident byte mutates.
  using host_mirror_type = typename Fab<Dim, MemorySpace>::host_mirror_type;
  std::vector<host_mirror_type> staged;
  long local_stage_failure = 0;
  try {
    staged.reserve(field.local_size());
    for (std::size_t local = 0; local < field.local_size(); ++local) {
      Fab<Dim, MemorySpace>& fab = field.fab(local);
      auto host = fab.create_host_mirror();
      fab.copy_to_host(host);
      for_each_host_index(fab.box(), [&](const Index<Dim>& index, std::size_t) {
        const std::size_t global = domain_ordinal(domain, index);
        for (int component = 0; component < components; ++component)
          host(storage_ordinal(fab, index, component)) =
              static_cast<Real>(values[static_cast<std::size_t>(component) * cells + global]);
      });
      staged.push_back(std::move(host));
    }
  } catch (...) {
    local_stage_failure = 1;
  }
  if (all_reduce_max(local_stage_failure) != 0)
    throw std::runtime_error("exact global field restore staging failed on at least one MPI rank");
  for (std::size_t local = 0; local < field.local_size(); ++local)
    field.fab(local).copy_from_host(staged[local]);
  Kokkos::fence();
}

}  // namespace pops::runtime::system::marshaling
