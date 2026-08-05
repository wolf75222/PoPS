/// @file
/// @brief Exact-ranked materialization of prepared auxiliary scalar providers.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/index/real_vector.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace pops::coupling {

enum class AuxiliaryFillRegion : unsigned char { valid, allocated };

namespace auxiliary_fill_detail {

template <int Dim, class Provider>
struct MaterializeAuxiliaryScalar {
  FieldView<Real, Dim> output{};
  Geometry<Dim> geometry;
  Provider provider;
  int component = 0;

  POPS_HD void operator()(const Index<Dim>& index) const {
    output(index, component) = provider(geometry.cell_center(index));
  }
};

}  // namespace auxiliary_fill_detail

/// Materialize one device-callable scalar provider over every local exact-ranked patch.
///
/// Provider must be a copyable value with `Real operator()(RealVector<Dim>) const` callable on the
/// active execution backend.  Python authoring is lowered to such a provider before this seam; no
/// Python callback, std::function, communicator, or runtime rank branch is retained here.
template <int Dim, class MemorySpace, class Provider>
void fill_auxiliary_component(MultiFab<Dim, MemorySpace>& auxiliary, const Geometry<Dim>& geometry,
                              int component, Provider provider,
                              AuxiliaryFillRegion region = AuxiliaryFillRegion::valid) {
  static_assert(std::is_copy_constructible_v<Provider>,
                "auxiliary providers must be copy-constructible execution values");
  if (component < 0 || component >= auxiliary.ncomp())
    throw std::out_of_range("auxiliary fill component lies outside the field channel");
  for (const Box<Dim>& patch : auxiliary.layout().boxes())
    if (!geometry.domain().contains(patch))
      throw std::invalid_argument(
          "auxiliary fill geometry does not contain the exact field layout");

  for (std::size_t local = 0; local < auxiliary.local_size(); ++local) {
    auto& fab = auxiliary.fab(local);
    const Box<Dim>& cells = region == AuxiliaryFillRegion::valid ? fab.box() : fab.grown_box();
    for_each_cell(cells, auxiliary_fill_detail::MaterializeAuxiliaryScalar<Dim, Provider>{
                             fab.view(), geometry, provider, component});
  }
  Kokkos::fence();
}

}  // namespace pops::coupling
