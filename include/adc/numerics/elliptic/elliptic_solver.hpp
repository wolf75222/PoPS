#pragma once

/// @file
/// @brief Concept EllipticSolver : contrat commun des solveurs elliptiques au niveau MultiFab
///        (resoudre D phi = f), pour que les coupleurs dependent du concept et non d'une classe concrete.
///
/// Couche : `include/adc/numerics/elliptic`.
/// Role : exprimer la dependance "solveur elliptique" par un concept C++20 plutot que par GeometricMG en
/// dur, ce qui prepare l'echange MG <-> autre backend (FFT enveloppe, PETSc, Hypre) sans toucher la
/// logique de couplage.
/// Contrat : un EllipticSolver expose rhs() -> MultiFab& (second membre f, ecrit avant solve),
/// phi() -> MultiFab& (solution lue apres solve, conservee entre appels pour le warm start),
/// solve() (resout phi a partir de rhs en place), residual() -> Real (norme du residu ||D phi - f||),
/// geom() -> const Geometry& (geometrie du niveau resolu).
///
/// Invariants :
/// - le contrat est au niveau MultiFab : poisson_fft.hpp (slabs + vecteurs bruts) ne le modele PAS
///   directement ; c'est PoissonFFTSolver/DistributedFFTSolver qui l'enveloppent ;
/// - phi() est conservee entre appels (warm start) : ne PAS supposer une remise a zero implicite.

#include <adc/core/types.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>

#include <concepts>

// Contrat commun des solveurs elliptiques (resoudre D phi = f sur un MultiFab).
//
// But : que les coupleurs dependent du CONCEPT, pas d'une implementation concrete.
// Aujourd'hui Coupler nomme GeometricMG en dur ; en exprimant la dependance par
// EllipticSolver on prepare l'echange MG <-> autre backend (FFT enveloppe, PETSc,
// Hypre) sans toucher la logique de couplage.
//
// Interface au niveau MultiFab (celle de GeometricMG) :
//   rhs()      -> MultiFab&        : second membre f (on y ecrit avant solve)
//   phi()      -> MultiFab&        : solution (on y lit apres solve ; conservee
//                                    entre appels -> warm start)
//   solve()                        : resout phi a partir de rhs, en place
//   residual() -> Real             : norme du residu courant ||D phi - f||
//   geom()     -> const Geometry&  : geometrie du niveau resolu
//
// Note : poisson_fft.hpp est une brique de plus bas niveau (slabs + vecteurs
// bruts, pas MultiFab) ; un PoissonFFTSolver qui l'enveloppe modelerait ce concept.

namespace adc {

template <class S>
concept EllipticSolver = requires(S s) {
  { s.rhs() } -> std::same_as<MultiFab&>;
  { s.phi() } -> std::same_as<MultiFab&>;
  s.solve();
  { s.residual() } -> std::convertible_to<Real>;
  { s.geom() } -> std::convertible_to<const Geometry&>;
};

}  // namespace adc
