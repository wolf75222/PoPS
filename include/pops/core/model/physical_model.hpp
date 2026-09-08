/// @file
/// @brief C++20 concepts defining the contract of the physics layer.
///
/// Concept hierarchy:
///   PhysicalStateFor: exact-ranked state metadata, with no mandatory operations.
///   PhysicalTransportFor / PhysicalSourceFor / PhysicalEllipticRhsFor: pointwise operations.
///   PhysicalModel: compatibility aggregate (flux, source, wave speed, elliptic RHS).
///   HasPrimitiveVars: optional extension (primitive variables + cons<->prim conversions).
///   HyperbolicPhysicalModel: complete hyperbolic brick (flux + conversions + Variables).
///   HyperbolicModel: compat alias for HyperbolicPhysicalModel.
///
/// PROVIDER INVARIANT: every pointwise consumer receives an exact compact
/// `ProviderValues<N>` pack.  Slots are resolved by the host-side qualified provider plan; this
/// layer never reserves physical names, axes, or a process-global auxiliary layout.
///
/// device INVARIANT: the concept methods (flux, source, ...) must be POPS_HD if
/// they are called in kernels. The concept does not check it -- that is the
/// responsibility of the model author.

#pragma once

#include <pops/core/state/state.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/state/variables.hpp>  // Variables: mandatory contract of the hyperbolic model

#include <concepts>

// The contract of the physics layer.
//
// A PhysicalModel describes ONE equation: its pointwise formulas. Nothing
// more. It is the only "what to compute" axis of the architecture, separate from
// the "where / how to iterate" axis (mesh + dispatch) and the "in what order" axis
// (integrator + coupler).
//
// Everything is a pure function of pointwise states:
//   - flux(U, aux, dir): the physical flux in direction dir
//   - max_wave_speed(U, aux, dir): the largest wave speed (for the CFL
//                                  and the Riemann solver)
//   - source(U, aux): the pointwise source term
//   - elliptic_rhs(U): the right-hand side of the elliptic equation
//                                  (charge / mass density depending on the model)
//
// flux AND source take aux: this is the point that unifies drift transport
// (aux in the flux) and the self-gravitating compressible fluid (aux in the
// source) under one same spatial operator.
//
// Provider contract: the model's `n_providers` is the exact number of values consumed by the
// assembled model.  A provider-free law declares none and receives ProviderValues<0>.

namespace pops {

/// Number of values a model consumes from its qualified provider plan.
///
/// This is intentionally independent from spatial rank.  A field-free Euler law is a real
/// `ProviderValues<0>` consumer; an electrostatic source can consume exactly `Dim` gradient
/// values only when that source is selected.  The number is evaluated in device-instantiated
/// templates, hence POPS_HD.
template <class M, int Dim>
POPS_HD constexpr int provider_count_for() {
  static_assert(Dim >= 1 && Dim <= 3, "physical model rank must be 1, 2, or 3");
  if constexpr (requires { M::n_providers; })
    return M::n_providers;
  else
    return 0;
}

/// Exact provider value count of a model in this compiled native artifact.
template <class M>
POPS_HD constexpr int provider_count() {
  return provider_count_for<M, kNativeDimension>();
}

/// Metadata shared by exact-ranked operation consumers. A storage association does not imply
/// transport, source, elliptic, or recovery capabilities; each selected consumer checks its own.
template <class M, int Dim>
concept PhysicalStateFor = (Dim >= 1 && Dim <= 3) && requires {
  typename M::State;
  requires(M::dimension == Dim);
  { M::n_vars } -> std::convertible_to<int>;
};

/// Pointwise transport spelling retained by the aggregate model adapter. Numerical-flux
/// consumers additionally bind the existing PhysicalFluxView/PhysicalFlux contract, which supports
/// statically selected axes and the qualified opaque provider carrier.
template <class M, int Dim>
concept PhysicalTransportFor =
    PhysicalStateFor<M, Dim> && requires { requires(provider_count_for<M, Dim>() >= 0); } &&
    requires(const M m, const typename M::State u,
             const ProviderValues<provider_count_for<M, Dim>()> providers, int dir) {
      { m.flux(u, providers, dir) } -> std::same_as<typename M::State>;
      { m.max_wave_speed(u, providers, dir) } -> std::convertible_to<Real>;
    };

/// Local source evaluation consumes exactly its declared provider pack. No flux, wave-speed,
/// primitive-variable, or elliptic method is needed by the source materialization kernel.
template <class M, int Dim>
concept PhysicalSourceFor =
    PhysicalStateFor<M, Dim> && requires { requires(provider_count_for<M, Dim>() >= 0); } &&
    requires(const M m, const typename M::State u,
             const ProviderValues<provider_count_for<M, Dim>()> providers) {
      { m.source(u, providers) } -> std::same_as<typename M::State>;
    };

/// Existing scalar problem-load observation. This is a pointwise load contract, not a claim
/// that an arbitrary field problem has a supported solver or discretization.
template <class M, int Dim>
concept PhysicalEllipticRhsFor =
    PhysicalStateFor<M, Dim> && requires(const M m, const typename M::State u) {
      { m.elliptic_rhs(u) } -> std::convertible_to<Real>;
    };

/// Compatibility aggregate used by existing CompositeModel consumers. Its methods are the
/// conjunction of meaningful operation contracts; new consumers select only the needed contract.
/// Device-callable methods must still be POPS_HD (not checked by C++ concepts).
template <class M, int Dim>
concept PhysicalModelFor =
    PhysicalTransportFor<M, Dim> && PhysicalSourceFor<M, Dim> && PhysicalEllipticRhsFor<M, Dim>;

template <class M>
concept PhysicalModel = PhysicalModelFor<M, kNativeDimension>;

// ---------------------------------------------------------------------------------------------
// OPTIONAL TIME STEP BOUNDS of the model contract (audit 2026-06, "step_cfl" workstream).
//
// Historically, step_cfl knew ONLY the hyperbolic transport bound
// dt <= cfl * h / max_wave_speed. But a model may impose other bounds: source frequency
// (collision/reaction, mu = eig(dS/dU), unit 1/time, WITHOUT h), or directly an admissible
// step (coupled transport-source formula not reducible). These three OPTIONAL traits let the
// model declare them; a model that declares none keeps STRICTLY the historical
// behavior (max_wave_speed fallback, bit-identical).
//
// SEMANTICS (all bounds apply to the EFFECTIVE SUBSTEP stride*dt/substeps of the block,
// see System<Dim>::step_cfl):
//  - stability_speed(U, aux, dir): stability speed lambda* [length/time] which REPLACES
//    max_wave_speed in the block CFL reduction (dt <= cfl * h / max_cells(lambda*)). For
//    when the speed relevant for STABILITY is not the physical wave speed (declared
//    conservative bound, speed modified by a coupling...). The Riemann solvers, themselves,
//    keep reading max_wave_speed (accuracy != stability).
//  - source_frequency(U, aux): local frequency mu [1/time] of the local source/coupling;
//    imposes dt <= cfl / max_cells(mu) -- NO h (the source bound has no space
//    dimension). Shortcut for explicit relaxation/collision/reaction.
//  - stability_dt(U, aux): direct ADMISSIBLE step [time] per cell; imposes
//    dt <= min_cells(stability_dt). The cfl is NOT applied (the model already declares an
//    admissible step; applying cfl on top would mix two margins). This is the most general form.
//
// STABILITY vs ACCURACY: these traits declare STABILITY bounds. A source treated
// implicitly (SourceImplicit/IMEX) may no longer impose a stability bound while keeping an
// ACCURACY constraint: it is up to the model to choose what stability_dt/source_frequency
// return in that case (or to not declare them). NON-local bounds (multi-block
// coupling, Schur/Poisson, AMR/scheduler) do NOT go through these cell-by-cell traits:
// they go through System::add_dt_bound (global host bound, one evaluation per step).
//
// GPU/MPI PRODUCTION: like flux/source, these methods must be POPS_HD (evaluated in
// reduction kernels) -- a per-cell Python callback is not a production path;
// the DSL compiles them (m.stability_speed(...) / m.stability_dt(...)).
// ---------------------------------------------------------------------------------------------

/// OPTIONAL trait: stability speed lambda* replacing max_wave_speed in the block CFL.
template <class M>
concept HasStabilitySpeed = requires(const M m, const typename M::State u,
                                     const ProviderValues<provider_count<M>()> providers, int dir) {
  { m.stability_speed(u, providers, dir) } -> std::convertible_to<Real>;
};

/// OPTIONAL trait: local source frequency mu [1/s] (bound dt <= cfl / max mu, without h).
template <class M>
concept HasSourceFrequency = requires(const M m, const typename M::State u,
                                      const ProviderValues<provider_count<M>()> providers) {
  { m.source_frequency(u, providers) } -> std::convertible_to<Real>;
};

/// OPTIONAL trait: direct admissible step per cell (bound dt <= min stability_dt, without cfl).
template <class M>
concept HasStabilityDt = requires(const M m, const typename M::State u,
                                  const ProviderValues<provider_count<M>()> providers) {
  { m.stability_dt(u, providers) } -> std::convertible_to<Real>;
};

/// Trait OPTIONNEL : PROJECTION PONCTUELLE post-pas U -> project(U, aux) (ADC-177). Le stepper
/// l'applique sur les cellules VALIDES de chaque bloc a la FIN de chaque macro-pas ENTIER (apres
/// transport + etage source + couplages ; jamais par etage RK). CONTRAT : project doit etre une
/// PROJECTION (idempotente : project(project(U), a) == project(U, a)) et PONCTUELLE (aucune lecture
/// de voisin) ; les formules elles-memes (realisabilite, clamps -- ecrits en max/min via abs/sign)
/// restent cote cas, seul le hook est coeur. POPS_HD obligatoire (evaluee dans un kernel).
template <class M>
concept HasPointwiseProjection = requires(const M m, const typename M::State u,
                                          const ProviderValues<provider_count<M>()> providers) {
  { m.project(u, providers) } -> std::same_as<typename M::State>;
};

/// OPTIONAL state conversion contract: primitive variables + cons<->prim conversions.
/// Source-only and transport-only laws can use the existing prepared recovery consumer without
/// acquiring unrelated flux, source, or elliptic methods.
///
/// Lets the spatial operator reconstruct in primitive variables (rho, u, p) rather than
/// conservative (more robust for Euler: positivity of rho and p), and centralizes the
/// cons <-> prim conversion (the wave speed, the collision terms u_a - u_b are expressed
/// naturally in primitive form). A model that does not expose it reconstructs in conservative form.
/// INVARIANT: to_primitive/to_conservative must be inverses of each other.
/// The spatial operator then reconstructs in primitive form (more robust: rho/p positivity).
template <class M>
concept HasPrimitiveVars =
    PhysicalStateFor<M, kNativeDimension> &&
    requires(const M m, const typename M::State u, const typename M::Prim p) {
      typename M::Prim;
      { m.to_primitive(u) } -> std::same_as<typename M::Prim>;
      { m.to_conservative(p) } -> std::same_as<typename M::State>;
    };

/// OPTIONAL physical admissibility contract for conservative-to-primitive recovery.
///
/// The conversion formula and the admissibility policy are deliberately separate: a finite
/// primitive candidate may still be physically invalid (for example non-positive density or
/// pressure).  When present, the prepared recovery service invokes this device-callable predicate
/// before publication.  `failing_component` identifies the primitive component whose declared
/// constraint failed; implementations set it to -1 on success.
template <class M>
concept HasRecoveryAdmissibility =
    HasPrimitiveVars<M> && requires(const M m, const typename M::Prim p, int* failing_component) {
      { m.recovery_admissible(p, failing_component) } -> std::same_as<bool>;
    };

/// Hyperbolic brick of a model: flux + wave speed + variables + cons<->prim conversions.
///
/// Variables, conversions and flux are physically LINKED (a flux is written for a given layout
/// of variables): they form a single brick distinct from the source and the elliptic.
/// conservative_vars() and primitive_vars() are MANDATORY (not an optional extra).
/// No source nor elliptic RHS here: those are other bricks of CompositeModel.
template <class M>
concept HyperbolicPhysicalModel =
    requires(const M m, const typename M::State u, const typename M::Prim p,
             const ProviderValues<provider_count<M>()> providers, int dir) {
      typename M::State;
      typename M::Prim;
      { M::n_vars } -> std::convertible_to<int>;
      { m.flux(u, providers, dir) } -> std::same_as<typename M::State>;
      { m.max_wave_speed(u, providers, dir) } -> std::convertible_to<Real>;
      { m.to_primitive(u) } -> std::same_as<typename M::Prim>;
      { m.to_conservative(p) } -> std::same_as<typename M::State>;
      { M::conservative_vars() } -> std::same_as<VariableSet>;
      { M::primitive_vars() } -> std::same_as<VariableSet>;
    };

/// Old name (compat): HyperbolicPhysicalModel used to be `HyperbolicModel`.
template <class M>
concept HyperbolicModel = HyperbolicPhysicalModel<M>;

}  // namespace pops
