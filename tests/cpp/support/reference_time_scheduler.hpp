#pragma once

#include <pops/core/model/coupled_system.hpp>
#include <pops/core/foundation/types.hpp>

#include <type_traits>
#include <utility>

/// @file
/// @brief Test-only reference scheduler for historical per-block cadence formulas.
///
/// Production `System` and `AmrSystem` execute only an installed `ProgramGraph`. This utility keeps
/// the former block-policy formulas available to numerical regression tests without exposing a
/// second temporal authority in installed PoPS headers.

namespace pops::test_support {

template <class Block>
constexpr int reference_block_substeps_v =
    TimePolicyTraits<typename std::decay_t<Block>::Time>::substeps;

template <class Block>
constexpr TimeTreatment reference_block_time_treatment_v =
    TimePolicyTraits<typename std::decay_t<Block>::Time>::treatment;

template <class Block>
constexpr int reference_block_stride_v =
    TimePolicyTraits<typename std::decay_t<Block>::Time>::stride;

// A block with cadence `stride` advances only on matching macro-steps and then catches up with an
// effective step `stride * dt`. Substeps split that effective interval without changing its total.
template <CoupledSystemLike System, class AdvanceBlock>
void advance_subcycled(System& system, Real dt, int macro_step, AdvanceBlock&& advance_block) {
  system.for_each_block([&](auto& block) {
    using Block = std::decay_t<decltype(block)>;
    if constexpr (reference_block_time_treatment_v<Block> != TimeTreatment::Prescribed) {
      constexpr int stride = reference_block_stride_v<Block>;
      if (macro_step % stride != 0)
        return;
      constexpr int count = reference_block_substeps_v<Block>;
      const Real substep_dt = (dt * static_cast<Real>(stride)) / static_cast<Real>(count);
      for (int substep = 0; substep < count; ++substep)
        advance_block(block, substep_dt, substep, count);
    }
  });
}

// Historical overload: macro_step=0 makes every non-prescribed block due.
template <CoupledSystemLike System, class AdvanceBlock>
void advance_subcycled(System& system, Real dt, AdvanceBlock&& advance_block) {
  advance_subcycled(system, dt, 0, std::forward<AdvanceBlock>(advance_block));
}

}  // namespace pops::test_support
