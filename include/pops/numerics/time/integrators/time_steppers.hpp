#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/storage/mf_arith.hpp>  // saxpy, lincomb
#include <pops/mesh/storage/multifab.hpp>

#include <utility>

/// @file
/// @brief Exact-ranked time integrators as first-class objects.
///
/// Layer: `include/pops/numerics/time`.
/// Role: keep the mathematical scheme alive in the core, with the coupler CALLING it instead
///        of inlining SSPRK everywhere. The "give a TimeIntegrator to the coupler like you give
///        a PhysicalModel" contract: the user can provide their own (same signature
///        take_step(rhs_eval, U, dt)).
/// Contract: an integrator is agnostic of the model and the discretization -- it only sees
///           rhs_eval(U_stage, R) (the method-of-lines arrow R = -div F + S) and the MultiFab
///           operations (saxpy / lincomb).
///
/// Invariants:
/// - no carried state: the scratch (R, stages U1/U2/U3) is sized from the layout of U. The
///   one-arg take_step allocates it per call; the scratch-taking overload reuses a caller-owned
///   buffer (see run_explicit_substeps) to hoist the allocation out of a substep loop -- both
///   paths are bit-identical;
/// - SSPRK2Step / SSPRK3Step preserve the retired static-driver algebra exactly.

namespace pops {

namespace time_stepper_detail {

template <int Dim, class MemorySpace>
struct ResidualProbe {
  void operator()(MultiFab<Dim, MemorySpace>&, MultiFab<Dim, MemorySpace>&) const {}
};

template <int Dim, class MemorySpace>
MultiFab<Dim, MemorySpace> make_scratch(const MultiFab<Dim, MemorySpace>& state,
                                        Extent<Dim> ghosts) {
  return MultiFab<Dim, MemorySpace>(state.layout(), state.distribution(), state.local_rank(),
                                    state.ncomp(), ghosts);
}

}  // namespace time_stepper_detail

/// Contract: an integrator knows how to advance one exact-ranked field via a residual evaluator.
template <class I, int Dim,
          class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
concept TimeStepperFor = requires(const I integrator, MultiFab<Dim, MemorySpace>& state, Real dt) {
  integrator.take_step(time_stepper_detail::ResidualProbe<Dim, MemorySpace>{}, state, dt);
};

// Forward Euler (order 1): U <- U + dt R(U).
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct ForwardEuler {
  using field_type = MultiFab<Dim, MemorySpace>;
  static constexpr int dimension = Dim;

  // Reusable residual buffer, sized from a U layout. Hoist it out of a substep loop (see
  // run_explicit_substeps) to avoid re-allocating per substep; rhs() overwrites R every call,
  // so reuse is bit-identical.
  struct Scratch {
    field_type R;
    explicit Scratch(const field_type& state)
        : R(time_stepper_detail::make_scratch(state, Extent<Dim>{})) {}
  };
  template <class RhsEval>
  void take_step(RhsEval&& rhs, field_type& state, Real dt, Scratch& scratch) const {
    rhs(state, scratch.R);
    saxpy(state, dt, scratch.R);
  }
  template <class RhsEval>
  void take_step_active(RhsEval&& rhs, field_type& state, Real dt, Scratch& scratch,
                        const field_type& active_cells) const {
    rhs(state, scratch.R);
    saxpy_active(state, dt, scratch.R, active_cells);
  }
  template <class RhsEval>
  void take_step(RhsEval&& rhs, field_type& state, Real dt) const {
    Scratch scratch(state);
    take_step(std::forward<RhsEval>(rhs), state, dt, scratch);
  }
};

// SSP-RK2 (Shu-Osher, 2 stages, order 2).
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct SSPRK2Step {
  using field_type = MultiFab<Dim, MemorySpace>;
  static constexpr int dimension = Dim;

  // Reusable stage buffers (residual R with 0 ghosts; stage U1 with U's ghosts, since rhs reads
  // its ghosts). Hoist out of a substep loop via run_explicit_substeps. Reuse is bit-identical:
  // R is overwritten by rhs and U1's valid cells are overwritten by the lincomb copy each substep,
  // while U1's ghosts are re-derived by rhs's internal fill_ghosts before any ghost read.
  struct Scratch {
    field_type R, U1;
    explicit Scratch(const field_type& state)
        : R(time_stepper_detail::make_scratch(state, Extent<Dim>{})),
          U1(time_stepper_detail::make_scratch(state, state.ghosts())) {}
  };
  template <class RhsEval>
  void take_step(RhsEval&& rhs, field_type& state, Real dt, Scratch& scratch) const {
    rhs(state, scratch.R);
    lincomb(scratch.U1, Real(1), state, Real(0), state);
    saxpy(scratch.U1, dt, scratch.R);
    rhs(scratch.U1, scratch.R);
    saxpy(scratch.U1, dt, scratch.R);
    lincomb(state, Real(0.5), state, Real(0.5), scratch.U1);
  }
  template <class RhsEval>
  void take_step_active(RhsEval&& rhs, field_type& state, Real dt, Scratch& scratch,
                        const field_type& active_cells) const {
    rhs(state, scratch.R);
    lincomb_active(scratch.U1, Real(1), state, Real(0), state, active_cells);
    saxpy_active(scratch.U1, dt, scratch.R, active_cells);
    rhs(scratch.U1, scratch.R);
    saxpy_active(scratch.U1, dt, scratch.R, active_cells);
    lincomb_active(state, Real(0.5), state, Real(0.5), scratch.U1, active_cells);
  }
  template <class RhsEval>
  void take_step(RhsEval&& rhs, field_type& state, Real dt) const {
    Scratch scratch(state);
    take_step(std::forward<RhsEval>(rhs), state, dt, scratch);
  }
};

// SSP-RK3 (Shu-Osher, 3 stages, order 3).
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct SSPRK3Step {
  using field_type = MultiFab<Dim, MemorySpace>;
  static constexpr int dimension = Dim;

  // Reusable stage buffers (residual R with 0 ghosts; stages U1/U2/U3 with U's ghosts, since the
  // first two are passed back to rhs which reads their ghosts). Hoist out of a substep loop via
  // run_explicit_substeps. Reuse is bit-identical: every buffer is fully overwritten each substep
  // (R by rhs, U1/U2/U3 valid cells by the lincomb copy), and stage ghosts are re-derived by rhs.
  struct Scratch {
    field_type R, U1, U2, U3;
    explicit Scratch(const field_type& state)
        : R(time_stepper_detail::make_scratch(state, Extent<Dim>{})),
          U1(time_stepper_detail::make_scratch(state, state.ghosts())),
          U2(time_stepper_detail::make_scratch(state, state.ghosts())),
          U3(time_stepper_detail::make_scratch(state, state.ghosts())) {}
  };
  template <class RhsEval>
  void take_step(RhsEval&& rhs, field_type& state, Real dt, Scratch& scratch) const {
    rhs(state, scratch.R);
    lincomb(scratch.U1, Real(1), state, Real(0), state);
    saxpy(scratch.U1, dt, scratch.R);

    rhs(scratch.U1, scratch.R);
    lincomb(scratch.U2, Real(1), scratch.U1, Real(0), scratch.U1);
    saxpy(scratch.U2, dt, scratch.R);
    lincomb(scratch.U2, Real(3) / 4, state, Real(1) / 4, scratch.U2);

    rhs(scratch.U2, scratch.R);
    lincomb(scratch.U3, Real(1), scratch.U2, Real(0), scratch.U2);
    saxpy(scratch.U3, dt, scratch.R);
    lincomb(state, Real(1) / 3, state, Real(2) / 3, scratch.U3);
  }
  template <class RhsEval>
  void take_step_active(RhsEval&& rhs, field_type& state, Real dt, Scratch& scratch,
                        const field_type& active_cells) const {
    rhs(state, scratch.R);
    lincomb_active(scratch.U1, Real(1), state, Real(0), state, active_cells);
    saxpy_active(scratch.U1, dt, scratch.R, active_cells);

    rhs(scratch.U1, scratch.R);
    lincomb_active(scratch.U2, Real(1), scratch.U1, Real(0), scratch.U1, active_cells);
    saxpy_active(scratch.U2, dt, scratch.R, active_cells);
    lincomb_active(scratch.U2, Real(3) / 4, state, Real(1) / 4, scratch.U2, active_cells);

    rhs(scratch.U2, scratch.R);
    lincomb_active(scratch.U3, Real(1), scratch.U2, Real(0), scratch.U2, active_cells);
    saxpy_active(scratch.U3, dt, scratch.R, active_cells);
    lincomb_active(state, Real(1) / 3, state, Real(2) / 3, scratch.U3, active_cells);
  }
  template <class RhsEval>
  void take_step(RhsEval&& rhs, field_type& state, Real dt) const {
    Scratch scratch(state);
    take_step(std::forward<RhsEval>(rhs), state, dt, scratch);
  }
};

// Runs @p n explicit RK substeps of @c Stepper on @p U with step @p h. When the stepper exposes a
// reusable Scratch (the built-in ForwardEuler / SSPRK2Step / SSPRK3Step), the scratch is allocated
// ONCE here and reused across substeps via the scratch-taking take_step overload -- removing the
// per-substep alloc/zero/free churn (ADC-261). A custom TimeStepper without a Scratch falls back to
// the one-shot take_step. The result is bit-identical either way (same saxpy/lincomb sequence on
// freshly-overwritten buffers); the substep loop never changes U's layout, so one Scratch suffices.
template <class Stepper, class RhsEval, int Dim, class MemorySpace>
  requires TimeStepperFor<Stepper, Dim, MemorySpace>
inline void run_explicit_substeps(RhsEval&& rhs, MultiFab<Dim, MemorySpace>& state, Real h, int n) {
  // Probe the actual scratch-taking overload (not just a nested type named Scratch): a custom
  // TimeStepper that happens to expose an unrelated Scratch type but no four-arg take_step still
  // takes the one-shot fallback instead of hitting a hard error.
  if constexpr (requires(Stepper stepper, RhsEval evaluator, MultiFab<Dim, MemorySpace>& field,
                         Real dt, typename Stepper::Scratch& scratch) {
                  stepper.take_step(evaluator, field, dt, scratch);
                }) {
    typename Stepper::Scratch scratch(state);
    for (int s = 0; s < n; ++s)
      Stepper{}.take_step(rhs, state, h, scratch);
  } else {
    for (int s = 0; s < n; ++s)
      Stepper{}.take_step(rhs, state, h);
  }
}

/// Active-domain counterpart of run_explicit_substeps. Built-in steppers must expose an explicit
/// masked algebra implementation; absence is a compile-time error instead of a silent full-grid
/// fallback that could alter inactive storage.
template <class Stepper, class RhsEval, int Dim, class MemorySpace>
  requires TimeStepperFor<Stepper, Dim, MemorySpace>
inline void run_explicit_substeps_active(RhsEval&& rhs, MultiFab<Dim, MemorySpace>& state, Real h,
                                         int n, const MultiFab<Dim, MemorySpace>& active_cells) {
  typename Stepper::Scratch scratch(state);
  for (int s = 0; s < n; ++s)
    Stepper{}.take_step_active(rhs, state, h, scratch, active_cells);
}

}  // namespace pops
