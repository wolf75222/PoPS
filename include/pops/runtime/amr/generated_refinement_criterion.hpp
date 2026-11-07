/// @file
/// @brief The refinement criterion as a region generated into the compiled Program (DM13b).
///
/// A refinement criterion is written in Python as a region over the cells of a level and is
/// generated like any other region. The compiled Program exports one entry that evaluates it over
/// one patch: the core only loops over the patches of a level, hands each one to the entry and
/// reads back one decision byte per valid cell. The core never knows which quantities, offsets or
/// arithmetic the criterion uses.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/storage/field_view.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace pops::runtime::amr {

/// Decision bits written by a generated criterion, one byte per cell. A cell that is neither
/// refined nor coarsened is kept. `kGeneratedCriterionNonFinite` marks a cell whose evaluation met
/// a non-finite value; the level is then rejected collectively.
enum GeneratedCriterionMark : std::uint8_t {
  kGeneratedCriterionRefine = 1u << 0,
  kGeneratedCriterionCoarsen = 1u << 2,
  kGeneratedCriterionNonFinite = 1u << 4,
};

/// Box-linear view of the decision bytes of one patch: axis 0 is contiguous.
template <int Dim>
struct GeneratedCriterionMarks {
  std::uint8_t* values = nullptr;
  Box<Dim> box{};

  POPS_HD std::uint8_t& operator()(const Index<Dim>& index) const {
    std::int64_t linear = 0;
    std::int64_t stride = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      linear += (static_cast<std::int64_t>(index[axis]) - box.lo[axis]) * stride;
      stride *= box.length(axis);
    }
    return values[linear];
  }
};

/// One patch handed to the generated criterion. `fields` holds one view per field that the
/// criterion reads, in the order of its field table; each view covers the valid cells and the
/// halo that the criterion reads. `parameters` holds the bound values of its runtime parameters.
template <int Dim>
struct GeneratedCriterionPatch {
  Box<Dim> box{};
  const FieldView<const Real, Dim>* fields = nullptr;
  std::size_t field_count = 0;
  std::array<Real, Dim> spacing{};
  const Real* parameters = nullptr;
  std::size_t parameter_count = 0;
  std::uint8_t* marks = nullptr;
};

/// The exported entry: evaluates the criterion on every valid cell of the patch and returns 0, or
/// a nonzero value when the patch does not match the generated field and parameter tables.
template <int Dim>
using GeneratedCriterionEntry = int (*)(const GeneratedCriterionPatch<Dim>*);

/// Exported symbols of a compiled Program that carries a refinement criterion.
inline constexpr const char* kGeneratedCriterionEntrySymbol =
    "pops_program_refinement_criterion_amr";
inline constexpr const char* kGeneratedCriterionIdentitySymbol =
    "pops_program_refinement_criterion_identity";

}  // namespace pops::runtime::amr
