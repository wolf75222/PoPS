#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>
#include <adc/numerics/spatial_operator.hpp>
#include <adc/numerics/time/implicit_stepper.hpp>
#include <adc/numerics/time/time_steppers.hpp>
#include <adc/runtime/grid_context.hpp>  // GridContext + BlockClosures (en-tete leger partage)

#include <functional>
#include <stdexcept>
#include <string>

/// @file
/// @brief Construit les fermetures d'un bloc (avance en temps + residu + contribution Poisson) a
///        partir d'un modele COMPILE (CompositeModel) et d'un contexte de grille.
///
/// Ce code etait dans System::Impl ; il est extrait en en-tete pour que le MEME chemin template
/// (assemble_rhs<Limiter, Flux>, inlinable et device-ready) soit instanciable depuis une UNITE DE
/// TRADUCTION EXTERNE. C'est la brique qui permettra a un modele genere par le DSL d'etre compile
/// AOT (ahead-of-time) puis branche dans le System par le chemin de PRODUCTION (flux HLLC/Roe,
/// ordre 2, GPU), et non plus seulement par le chemin hote virtuel du bloc dynamique.
///
/// Le System reste l'unique proprietaire du maillage et de l'aux ; GridContext n'en porte que des
/// copies immuables (domaine, CL, geometrie) et un POINTEUR non possedant vers l'aux (adresse stable,
/// duree de vie superieure au bloc).

namespace adc {

// GridContext et BlockClosures : definis dans adc/runtime/grid_context.hpp (en-tete leger, inclus
// aussi par system.hpp pour exposer grid_context() / install_block() sans tirer la numerique).

namespace detail {
/// Foncteur residu -div F + S (fill_ghosts puis assemble_rhs) capture par les TimeStepper.
template <class Limiter, class Flux, class Model>
auto block_rhs_eval(const Model& model, const GridContext& ctx, bool recon_prim) {
  return [model, ctx, recon_prim](MultiFab& U, MultiFab& R) {
    fill_ghosts(U, ctx.dom, ctx.bc);
    assemble_rhs<Limiter, Flux>(model, U, *ctx.aux, ctx.geom, R, recon_prim);
  };
}
}  // namespace detail

/// Fermetures (avance + residu) pour un schema spatial (Limiter x Flux) fige. La math RK vient des
/// TimeStepper du coeur : SSPRK2 en explicite ; ForwardEuler + backward_euler_source en IMEX.
template <class Limiter, class Flux, class Model>
BlockClosures build_block(const Model& m, const GridContext& ctx, bool imex, bool recon_prim) {
  BlockClosures bc;
  if (imex)
    bc.advance = [m, ctx, recon_prim](MultiFab& U, Real dt, int n) {
      const Real h = dt / static_cast<Real>(n);
      for (int s = 0; s < n; ++s) {
        const SourceFreeModel<Model> sf{m};  // demi-pas explicite : transport sans source
        ForwardEuler{}.take_step(detail::block_rhs_eval<Limiter, Flux>(sf, ctx, recon_prim), U, h);
        backward_euler_source(m, *ctx.aux, U, h);  // source implicite (rappel raide)
      }
    };
  else
    bc.advance = [m, ctx, recon_prim](MultiFab& U, Real dt, int n) {
      const Real h = dt / static_cast<Real>(n);
      for (int s = 0; s < n; ++s)
        SSPRK2Step{}.take_step(detail::block_rhs_eval<Limiter, Flux>(m, ctx, recon_prim), U, h);
    };
  bc.rhs_into = [m, ctx, recon_prim](MultiFab& U, MultiFab& R) {
    fill_ghosts(U, ctx.dom, ctx.bc);
    assemble_rhs<Limiter, Flux>(m, U, *ctx.aux, ctx.geom, R, recon_prim);
  };
  return bc;
}

/// Dispatch du schema spatial (limiteur x flux Riemann) -> fermetures compilees. HLLC / Roe gardes
/// par requires : exigent un transport a 4 variables exposant pressure (sinon erreur explicite).
template <class Model>
BlockClosures make_block(const Model& m, const std::string& lim, const std::string& riem,
                         const GridContext& ctx, bool imex, bool recon_prim) {
  if (riem == "rusanov") {
    if (lim == "none") return build_block<NoSlope, RusanovFlux>(m, ctx, imex, recon_prim);
    if (lim == "minmod") return build_block<Minmod, RusanovFlux>(m, ctx, imex, recon_prim);
    if (lim == "vanleer") return build_block<VanLeer, RusanovFlux>(m, ctx, imex, recon_prim);
    throw std::runtime_error("System : limiter inconnu '" + lim + "'");
  }
  if (riem == "hllc") {
    if constexpr (Model::n_vars == 4 &&
                  requires(const Model mm, typename Model::State s) { mm.pressure(s); }) {
      if (lim == "none") return build_block<NoSlope, HLLCFlux>(m, ctx, imex, recon_prim);
      if (lim == "minmod") return build_block<Minmod, HLLCFlux>(m, ctx, imex, recon_prim);
      if (lim == "vanleer") return build_block<VanLeer, HLLCFlux>(m, ctx, imex, recon_prim);
      throw std::runtime_error("System : limiter inconnu '" + lim + "'");
    } else {
      throw std::runtime_error("System : flux 'hllc' exige un transport compressible "
                               "(4 variables + pression) ; ce transport -> 'rusanov'");
    }
  }
  if (riem == "roe") {
    if constexpr (Model::n_vars == 4 &&
                  requires(const Model mm, typename Model::State s) { mm.pressure(s); }) {
      if (lim == "none") return build_block<NoSlope, RoeFlux>(m, ctx, imex, recon_prim);
      if (lim == "minmod") return build_block<Minmod, RoeFlux>(m, ctx, imex, recon_prim);
      if (lim == "vanleer") return build_block<VanLeer, RoeFlux>(m, ctx, imex, recon_prim);
      throw std::runtime_error("System : limiter inconnu '" + lim + "'");
    } else {
      throw std::runtime_error("System : flux 'roe' exige un transport compressible "
                               "(4 variables + pression) ; ce transport -> 'rusanov'");
    }
  }
  throw std::runtime_error("System : flux Riemann inconnu '" + riem + "' (rusanov|hllc|roe)");
}

/// Fermeture de vitesse d'onde max du bloc (pour le pas CFL).
template <class Model>
std::function<Real(const MultiFab&)> make_max_speed(const Model& m, const GridContext& ctx) {
  return [m, ctx](const MultiFab& U) { return max_wave_speed_mf(m, U, *ctx.aux); };
}

/// Contribution du bloc au second membre de Poisson : rhs += elliptic_rhs(U) (boucle hote).
template <class Model>
std::function<void(const MultiFab&, MultiFab&)> make_poisson_rhs(const Model& m) {
  return [m](const MultiFab& U, MultiFab& rhs) {
    for (int li = 0; li < rhs.local_size(); ++li) {
      Array4 r = rhs.fab(li).array();
      const ConstArray4 u = U.fab(li).const_array();
      const Box2D b = rhs.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          r(i, j) += m.elliptic_rhs(load_state<Model>(u, i, j));
    }
  };
}

}  // namespace adc
