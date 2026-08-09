#pragma once

#include <pops/core/foundation/types.hpp>

/// @file
/// @brief Operator splitting: decomposes dU/dt = T(U) + S(U) into separate substeps.
///        lie_step (Godunov, 1st order) = T(dt) then S(dt); strang_step (2nd order) =
///        S(dt/2), T(dt), S(dt/2).
///
/// Layer: `include/pops/numerics/time`.
/// Role: handle a stiff source (relaxation, collisions, ionization) with an integrator
///       DIFFERENT from the transport one, without mixing the two stiffnesses.
/// Contract: T and S advance the same exact state type in place; the composition is agnostic to
///           spatial rank and storage.
///
/// Invariants:
/// - Strang is 2nd order as soon as each sub-integrator is (commutation error [T,S] is
///   O(dt^3) per step, O(dt^2) globally).

namespace pops {

template <class State, class TransportStep, class SourceStep>
void lie_step(State& state, Real dt, TransportStep transport, SourceStep source) {
  transport(state, dt);
  source(state, dt);
}

template <class State, class TransportStep, class SourceStep>
void strang_step(State& state, Real dt, TransportStep transport, SourceStep source) {
  source(state, Real(0.5) * dt);
  transport(state, dt);
  source(state, Real(0.5) * dt);
}

}  // namespace pops
