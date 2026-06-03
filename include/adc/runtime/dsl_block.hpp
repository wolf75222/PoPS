#pragma once

#include <adc/runtime/block_builder.hpp>
#include <adc/runtime/system.hpp>

#include <functional>
#include <string>
#include <utility>

/// @file
/// @brief add_compiled_model : branche un modele COMPILE (un CompositeModel, typiquement genere par
///        le DSL puis inclus au moment de la COMPILATION) comme bloc NATIF du System.
///
/// Difference avec System::add_compiled_block (.so + ABI extern "C" + marshaling de tableaux plats,
/// pour le prototypage RUNTIME cote Python) : ici le modele est connu a la compilation, donc
/// block_builder fabrique les fermetures sur le CONTEXTE DE GRILLE REEL du System (grid_context) et le
/// bloc tourne EXACTEMENT le chemin de production -- le residu fait fill_boundary (halos MPI) +
/// assemble_rhs (device Kokkos) sur les vrais MultiFab du System, SANS recopie. Parite complete
/// Kokkos + MPI avec un bloc add_block (le meme make_block est utilise). C'est le backend "compile"
/// de l'ideal m.compile_or_jit() pour un binaire de production.

namespace adc {

/// Ajoute @p model (CompositeModel) comme bloc natif de @p sys avec le schema (limiter x riemann,
/// reconstruction, traitement temporel) demande. @p gamma sert a set_density (energie au repos, 4 var).
template <class Model>
void add_compiled_model(System& sys, const std::string& name, Model model,
                        const std::string& limiter = "minmod",
                        const std::string& riemann = "rusanov",
                        const std::string& recon = "conservative",
                        const std::string& time = "explicit", double gamma = 1.4,
                        int substeps = 1, bool evolve = true) {
  const bool imex = (time == "imex");
  const bool recon_prim = (recon == "primitive");
  const GridContext ctx = sys.grid_context();
  BlockClosures clo = make_block(model, limiter, riemann, ctx, imex, recon_prim);
  std::function<Real(const MultiFab&)> ms = make_max_speed(model, ctx);
  std::function<void(const MultiFab&, MultiFab&)> pr = make_poisson_rhs(model);
  sys.install_block(name, Model::n_vars, Model::conservative_vars().names,
                    Model::primitive_vars().names, gamma, std::move(clo), std::move(ms),
                    std::move(pr), substeps, evolve);
}

}  // namespace adc
