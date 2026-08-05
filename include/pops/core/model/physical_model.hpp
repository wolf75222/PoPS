/// @file
/// @brief C++20 concepts defining the contract of the physics layer.
///
/// Concept hierarchy:
///   PhysicalModel: minimal contract (flux, source, wave speed, elliptic RHS).
///   HasPrimitiveVars: optional extension (primitive variables + cons<->prim conversions).
///   HyperbolicPhysicalModel: complete hyperbolic brick (flux + conversions + Variables).
///   HyperbolicModel: compat alias for HyperbolicPhysicalModel.
///
/// Aux INVARIANT: `PhysicalModelFor<M, Dim>` receives exactly `AuxState<Dim>`; the public
/// `PhysicalModel` concept binds that contract to the immutable native build rank.
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
// Aux contract: the model and its field loader carry one immutable compile-time rank. The build
// facade names AuxState<kNativeDimension> as pops::Aux, while generic core proofs use AuxState<Dim>
// directly. No arbitrary Model::Aux can pass the contract without matching that exact rank.

namespace pops {

/// Width of the aux channel a model CONSUMES.
///
/// Returns `M::n_aux` if the model declares it, otherwise the exact ranked base width
/// (`phi + Dim gradients`).
/// Lives in this header (contract) and not in the spatial operator, so that
/// CompositeModel can propagate n_aux without pulling in all the numerics.
// POPS_HD : aux_comps() est evaluee a la compilation (argument de template non-type
// load_aux<aux_comps<Model>()>) DANS les kernels device (cf. spatial_operator_eb.hpp). Sous nvcc,
// appeler une constexpr __host__ depuis une fonction __host__ __device__ est refuse (#20013-D) ;
// la marquer POPS_HD la rend callable des deux cotes. Hors nvcc, POPS_HD est vide -> constexpr pur.
template <class M, int Dim>
POPS_HD constexpr int aux_comps_for() {
  if constexpr (requires { M::n_aux; })
    return M::n_aux;
  else
    return kAuxBaseCompsFor<Dim>;
}

/// Auxiliary width of a model in this compiled native artifact.
template <class M>
POPS_HD constexpr int aux_comps() {
  return aux_comps_for<M, kNativeDimension>();
}

/// Minimal contract of a physical model.
///
/// Requires: State, Aux == AuxState<Dim>, a valid ranked aux width, n_vars, flux(u,a,dir),
/// source(u,a), elliptic_rhs(u). All these methods must be POPS_HD if called
/// in kernels (not checked by the concept; responsibility of the author).
/// Finite-volume execution additionally instantiates the hyperbolic methods with the exact
/// BoundFluxProviders<Model> protocol; Aux remains the pointwise source/implicit carrier.
/// Do not confuse with HyperbolicPhysicalModel which adds the variables and conversions.
template <class M, int Dim>
concept PhysicalModelFor =
    requires(const M m, const typename M::State u, const typename M::Aux a, int dir) {
      typename M::State;
      typename M::Aux;
      requires std::same_as<typename M::Aux, AuxState<Dim>>;
      requires(aux_comps_for<M, Dim>() >= kAuxBaseCompsFor<Dim>);
      requires(aux_comps_for<M, Dim>() <= kAuxMaxCompsFor<Dim>);
      { M::n_vars } -> std::convertible_to<int>;
      { m.flux(u, a, dir) } -> std::same_as<typename M::State>;
      { m.max_wave_speed(u, a, dir) } -> std::convertible_to<Real>;
      { m.source(u, a) } -> std::same_as<typename M::State>;
      { m.elliptic_rhs(u) } -> std::convertible_to<Real>;
    };

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
concept HasStabilitySpeed = requires(const M m, const typename M::State u, const Aux a, int dir) {
  { m.stability_speed(u, a, dir) } -> std::convertible_to<Real>;
};

/// OPTIONAL trait: local source frequency mu [1/s] (bound dt <= cfl / max mu, without h).
template <class M>
concept HasSourceFrequency = requires(const M m, const typename M::State u, const Aux a) {
  { m.source_frequency(u, a) } -> std::convertible_to<Real>;
};

/// OPTIONAL trait: direct admissible step per cell (bound dt <= min stability_dt, without cfl).
template <class M>
concept HasStabilityDt = requires(const M m, const typename M::State u, const Aux a) {
  { m.stability_dt(u, a) } -> std::convertible_to<Real>;
};

/// Trait OPTIONNEL : PROJECTION PONCTUELLE post-pas U -> project(U, aux) (ADC-177). Le stepper
/// l'applique sur les cellules VALIDES de chaque bloc a la FIN de chaque macro-pas ENTIER (apres
/// transport + etage source + couplages ; jamais par etage RK). CONTRAT : project doit etre une
/// PROJECTION (idempotente : project(project(U), a) == project(U, a)) et PONCTUELLE (aucune lecture
/// de voisin) ; les formules elles-memes (realisabilite, clamps -- ecrits en max/min via abs/sign)
/// restent cote cas, seul le hook est coeur. POPS_HD obligatoire (evaluee dans un kernel).
template <class M>
concept HasPointwiseProjection = requires(const M m, const typename M::State u, const Aux a) {
  { m.project(u, a) } -> std::same_as<typename M::State>;
};

/// OPTIONAL extension of a PhysicalModel: primitive variables + cons<->prim conversions.
///
/// Lets the spatial operator reconstruct in primitive variables (rho, u, p) rather than
/// conservative (more robust for Euler: positivity of rho and p), and centralizes the
/// cons <-> prim conversion (the wave speed, the collision terms u_a - u_b are expressed
/// naturally in primitive form). A model that does not expose it reconstructs in conservative form.
/// INVARIANT: to_primitive/to_conservative must be inverses of each other.
/// The spatial operator then reconstructs in primitive form (more robust: rho/p positivity).
template <class M>
concept HasPrimitiveVars =
    PhysicalModel<M> && requires(const M m, const typename M::State u, const typename M::Prim p) {
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
    requires(const M m, const typename M::State u, const typename M::Prim p, const Aux a, int dir) {
      typename M::State;
      typename M::Prim;
      { M::n_vars } -> std::convertible_to<int>;
      { m.flux(u, a, dir) } -> std::same_as<typename M::State>;
      { m.max_wave_speed(u, a, dir) } -> std::convertible_to<Real>;
      { m.to_primitive(u) } -> std::same_as<typename M::Prim>;
      { m.to_conservative(p) } -> std::same_as<typename M::State>;
      { M::conservative_vars() } -> std::same_as<VariableSet>;
      { M::primitive_vars() } -> std::same_as<VariableSet>;
    };

/// Old name (compat): HyperbolicPhysicalModel used to be `HyperbolicModel`.
template <class M>
concept HyperbolicModel = HyperbolicPhysicalModel<M>;

}  // namespace pops
