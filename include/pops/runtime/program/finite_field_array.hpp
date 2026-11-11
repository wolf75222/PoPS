#pragma once

#include <Kokkos_Core.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/runtime/program/continuation_image.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::runtime::program::finite_device {

// Storage of a replicated finite Array whose element is a Field (`state`, `rhs` or
// `scalar_field`): for each local box, one column in the memory space of the field holds every
// element, element-major (element i of a box whose element Fab has S values occupies
// [i S, (i + 1) S) of that box's column). It is the field array as an array over the index set
// times the cells (spec DM7, MG3). Each element is a MultiFab over its own disjoint slices: it
// owns no storage and exists so that the continuation and exchange provenance of the element,
// which is keyed by field identity, binds to it as it bound to a boxed element copy. No element
// is boxed, no value is copied to the host, and the copies in and out stay in the memory space
// of the field (DM8).
inline constexpr std::string_view kFieldColumnLabel = "pops_mesh_array:finite_field_column";

template <int Dim>
struct FieldArrayStorage {
  using field_type = MultiFab<Dim>;
  using column_type = typename field_type::fab_type::storage_type;

  std::vector<column_type> columns;
  std::vector<field_type> elements;
};

template <int Dim>
struct FieldArray {
  using field_type = MultiFab<Dim>;

  FiniteDomainRef domain;
  std::shared_ptr<const FieldArrayStorage<Dim>> storage;

  // The element at a checked offset; reading it is a copy out of its slices
  // (`own_structural_field`), never a host access.
  const field_type& element(std::int64_t offset) const {
    if (!domain || !storage || storage->elements.size() != static_cast<std::size_t>(domain->size))
      throw std::logic_error("finite field Array differs from its domain cardinality");
    if (offset < 0 || offset >= domain->size)
      throw std::out_of_range("finite field Array read is outside its domain");
    return storage->elements[static_cast<std::size_t>(offset)];
  }
};

// The private buffer of one map or scan over a field-valued body. The columns are allocated at
// the first element, shaped like it; every later element must have the same layout,
// distribution, rank, components and ghosts. Each element is written exactly once before the
// only publication.
template <int Dim>
class FieldArrayBuilder {
 public:
  using field_type = MultiFab<Dim>;
  using fab_type = typename field_type::fab_type;
  using column_type = typename FieldArrayStorage<Dim>::column_type;

  explicit FieldArrayBuilder(FiniteDomainRef domain)
      : domain_(std::move(domain)), storage_(std::make_shared<FieldArrayStorage<Dim>>()) {
    if (!domain_ || domain_->size < 0)
      throw std::logic_error("finite field Array requires a live domain");
    written_.assign(static_cast<std::size_t>(domain_->size), 0);
  }

  // Copies `source` into the slices of element `offset` and returns that element, to which the
  // caller binds the provenance of `source` with `lifetime()`.
  const field_type& store(std::int64_t offset, const field_type& source) {
    if (offset < 0 || offset >= domain_->size)
      throw std::out_of_range("finite field Array element is outside its domain");
    const auto position = static_cast<std::size_t>(offset);
    if (written_[position])
      throw std::logic_error("finite field Array element is published twice");
    if (storage_->elements.empty())
      allocate_(source);
    field_type& slot = storage_->elements[position];
    require_same_shape_(slot, source);
    for (std::size_t local = 0; local < source.local_size(); ++local)
      if (source.fab(local).size() != 0)
        Kokkos::deep_copy(slot.fab(local).storage(), source.fab(local).storage());
    written_[position] = 1;
    return slot;
  }

  std::shared_ptr<const void> lifetime() const noexcept { return storage_; }

  FieldArray<Dim> publish() && {
    for (const auto written : written_)
      if (!written)
        throw std::logic_error("finite field Array publishes an unwritten element");
    return FieldArray<Dim>{std::move(domain_), std::move(storage_)};
  }

 private:
  void allocate_(const field_type& prototype) {
    const auto count = static_cast<std::size_t>(domain_->size);
    std::vector<std::size_t> sizes(prototype.local_size());
    storage_->columns.reserve(sizes.size());
    for (std::size_t local = 0; local < sizes.size(); ++local) {
      sizes[local] = prototype.fab(local).size();
      if (sizes[local] != 0 && count > static_cast<std::size_t>(-1) / sizes[local])
        throw std::length_error("finite field Array column exceeds the native size domain");
      storage_->columns.emplace_back(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(kFieldColumnLabel)),
          count * sizes[local]);
    }
    storage_->elements.reserve(count);
    for (std::size_t element = 0; element < count; ++element) {
      std::vector<fab_type> fabs;
      fabs.reserve(sizes.size());
      for (std::size_t local = 0; local < sizes.size(); ++local) {
        const auto first = element * sizes[local];
        column_type slice =
            Kokkos::subview(storage_->columns[local], std::make_pair(first, first + sizes[local]));
        fabs.emplace_back(prototype.box(local), prototype.ncomp(), prototype.ghosts(),
                          std::move(slice));
      }
      storage_->elements.emplace_back(prototype.layout(), prototype.distribution(),
                                      prototype.local_rank(), prototype.ncomp(), prototype.ghosts(),
                                      std::move(fabs));
    }
  }

  static void require_same_shape_(const field_type& slot, const field_type& source) {
    if (!(slot.layout() == source.layout()) || !(slot.distribution() == source.distribution()) ||
        !(slot.local_rank() == source.local_rank()) || slot.ncomp() != source.ncomp() ||
        !(slot.ghosts() == source.ghosts()))
      throw std::invalid_argument(
          "finite field Array elements differ in layout, distribution, components or ghosts");
  }

  FiniteDomainRef domain_;
  std::shared_ptr<FieldArrayStorage<Dim>> storage_;
  std::vector<unsigned char> written_;
};

}  // namespace pops::runtime::program::finite_device
