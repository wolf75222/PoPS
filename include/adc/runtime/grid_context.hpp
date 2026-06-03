#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>

#include <functional>

/// @file
/// @brief Contexte de grille + fermetures d'un bloc, partages entre System (qui les installe) et
///        block_builder.hpp (qui les fabrique a partir d'un modele compile). En-tete LEGER (mailles +
///        std::function, sans la numerique) pour pouvoir etre inclus dans l'API publique du System
///        sans y tirer assemble_rhs / flux / steppers.

namespace adc {

/// Maillage + CL transport + aux partages par les fermetures d'un bloc. @c aux n'est PAS possede :
/// il pointe l'aux du System (duree de vie superieure au bloc, adresse stable).
struct GridContext {
  Box2D dom;                ///< domaine (sans ghost)
  BCRec bc;                 ///< CL de transport
  Geometry geom;            ///< geometrie (dx, dy, bornes)
  MultiFab* aux = nullptr;  ///< aux du System (phi, grad phi) ; NON possede
};

/// Fermetures compilees d'un bloc, figees a l'ajout.
struct BlockClosures {
  std::function<void(MultiFab&, Real, int)> advance;  ///< (U, dt, n) : n sous-pas de dt/n
  std::function<void(MultiFab&, MultiFab&)> rhs_into;  ///< R <- -div F + S (Poisson fige)
};

}  // namespace adc
