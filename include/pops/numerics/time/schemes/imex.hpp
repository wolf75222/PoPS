#pragma once

#include <pops/core/foundation/types.hpp>

/// @file
/// @brief Asymptotic-preserving IMEX (implicit-explicit) integrator: imex_euler_step, order-1
///        forward-backward Euler step, U^{n+1} = U^n + dt T(U^n) + dt S(U^{n+1}).
///
/// Layer: `include/pops/numerics/time`.
/// Role: take the STIFF terms (Lorentz, Debye limit, quasi-neutrality) IMPLICITLY and the
///       transport EXPLICITLY. AP property: when the small parameter (lambda_D^2, 1/omega_c)
///       -> 0, the scheme stays stable at FIXED dt and captures the limit dynamics.
/// Contract: Texpl(U, dt) advances the transport IN PLACE (after the call U holds the known term
///           U^n + dt T(U^n)); Simpl(U, dt) solves IN PLACE U <- W with W = U + dt S(W), U being
///           the known term (linear relaxation: analytic; full Lorentz: local Newton).
///
/// Invariants:
/// - integrator agnostic of model, rank, and storage: Texpl/Simpl consume the same exact state;
/// - the order is enforced -- explicit THEN implicit; no state held by the integrator.

namespace pops {

template <class State, class TransportStep, class ImplicitSourceSolve>
void imex_euler_step(State& state, Real dt, TransportStep transport,
                     ImplicitSourceSolve implicit_source) {
  transport(state, dt);        // explicit: state becomes U^n + dt T(U^n)
  implicit_source(state, dt);  // implicit: solve U = known + dt S(U) in place
}

}  // namespace pops
