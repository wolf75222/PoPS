#pragma once

/// @file
/// @brief The step bound of a cells rule, generated into the compiled Program (N8 « dt= as a
/// function of cell », design note D2).
///
/// `pops.Advance(method, dt=f)`, with `f` a function of `cell`, bounds each step by the smallest
/// value of `f` over the valid cells of the evolving block, at the accepted state (spec 0.3 2.6.2,
/// PF8). The compiled Program exports one entry that reduces `f` over the cells that this rank
/// owns, in one of three modes, and the identity of the bound. The core calls the entry, reduces
/// each mode over the ranks with one maximum, and never knows which quantities, widths or
/// arithmetic the bound reads (CF2).
///
/// Each cell has a code. The bound at a cell is +infinity (a step to the frontier, 2.6.9), finite
/// and positive, not positive, or not finite: NaN, -infinity, or an operation of the bound whose
/// value is not finite, the branch of a `where` that the cell does not take excepted.

#include <cstdint>

namespace pops::runtime::program {

/// The code of a cell, ordered so that the largest code over the cells decides the step: a
/// failure if any cell fails, otherwise whether a cell has a finite bound.
enum class StepBoundCode : std::uint8_t {
  kInfinite = 0,     ///< +infinity at the cell
  kFinite = 1,       ///< a finite positive bound at the cell
  kNonpositive = 2,  ///< zero or a negative value: step_bound_nonpositive
  kNonfinite = 3,    ///< NaN, -infinity or a non-finite operation: step_bound_nonfinite
};

/// The reduction that the entry computes over the cells of this rank, each the largest value of:
enum class StepBoundMode : std::int32_t {
  kCode = 0,       ///< the code of the cell, as a Real
  kValue = 1,      ///< minus the bound at a cell whose code is kFinite, the lowest Real elsewhere
  kFirstCell = 2,  ///< minus the linear index of a cell whose code is the target, the lowest
                   ///< Real elsewhere: the first such cell in the order of the domain
};

/// int entry(System<Dim>* system, int mode, double target, double* result): writes the largest
/// value of @p mode over the valid cells of this rank to @p result (the lowest Real when it owns
/// none) and returns 0, or a nonzero value for a mode it does not know.
inline constexpr const char* kGeneratedStepBoundEntrySymbol = "pops_program_advance_step_bound";
/// const char* identity(): the identity of the bound, which the cells rule of the run names.
inline constexpr const char* kGeneratedStepBoundIdentitySymbol =
    "pops_program_advance_step_bound_identity";

}  // namespace pops::runtime::program
