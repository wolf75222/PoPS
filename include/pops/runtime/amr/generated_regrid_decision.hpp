/// @file
/// @brief The regrid decision as a closed body generated into the compiled Program (DM13b, T33).
///
/// `Program(..., regrid=Regrid(criterion, when))` writes the decision to regrid in Python: `when`
/// is a closed body over the accepted continuation that returns a Bool, such as
/// `pops.lib.time.every(4)` or a predicate on an indicator of the state. The compiled Program
/// exports two entries. The first evaluates one indicator, the largest value of a function of
/// `cell` over the valid cells of one patch; the core only loops over the patches of every level,
/// and reduces the patch values over the patches and the ranks. The second decides from the
/// accepted step count, the accepted time and the reduced indicators. The core never knows which
/// quantities, offsets or arithmetic the decision uses.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/storage/field_view.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace pops::runtime::amr {

/// One patch handed to the generated indicator `indicator`. `fields` holds one view per field of
/// the decision's field table, each covering the valid cells of the patch. The entry writes the
/// largest value of the indicator over the valid cells to `largest` (the smallest is generated as
/// the largest of the opposite value) and 1 to `non_finite` when a value it computed is not finite.
template <int Dim>
struct GeneratedIndicatorPatch {
  Box<Dim> box{};
  const FieldView<const Real, Dim>* fields = nullptr;
  std::size_t field_count = 0;
  std::array<Real, Dim> spacing{};
  const Real* parameters = nullptr;
  std::size_t parameter_count = 0;
  std::size_t indicator = 0;
  Real* largest = nullptr;
  std::uint8_t* non_finite = nullptr;
};

/// The reduced inputs of the decision: the accepted step count of the run and the time after the
/// step, and one value per indicator, already reduced over every level and rank. `steps` counts
/// the accepted steps from the start of the run (AmrSystem::run_start_step; spec 5.2), not from the
/// start of the simulation. The entry writes 1 to `regrid`
/// when the body is true and 0 otherwise.
struct GeneratedRegridDecisionRequest {
  std::int64_t steps = 0;
  Real time = 0;
  const Real* indicators = nullptr;
  std::size_t indicator_count = 0;
  const Real* parameters = nullptr;
  std::size_t parameter_count = 0;
  std::uint8_t* regrid = nullptr;
};

/// Both entries return 0, or a nonzero value when the request does not match the generated tables.
template <int Dim>
using GeneratedIndicatorEntry = int (*)(const GeneratedIndicatorPatch<Dim>*);
using GeneratedRegridDecisionEntry = int (*)(const GeneratedRegridDecisionRequest*);

/// Exported symbols of a compiled Program that carries a regrid decision.
inline constexpr const char* kGeneratedIndicatorEntrySymbol = "pops_program_regrid_indicator_amr";
inline constexpr const char* kGeneratedRegridDecisionEntrySymbol =
    "pops_program_regrid_decision_amr";
inline constexpr const char* kGeneratedRegridDecisionIdentitySymbol =
    "pops_program_regrid_decision_identity";

}  // namespace pops::runtime::amr
