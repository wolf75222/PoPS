#pragma once

#include <adc/runtime/export.hpp>  // ADC_EXPORT (visibilite defaut a travers le module _adc hidden)

#include <string>

/// @file
/// @brief Cle d'ABI du coeur adc : chaine stable identifiant la combinaison (compilateur,
///        standard C++, signature de l'arbre d'en-tetes) avec laquelle une unite a ete compilee.
///
/// MOTIVATION (chemin DSL "production"). Un loader .so genere par le DSL (cf.
/// dsl.emit_cpp_native_loader) inline le gabarit en-tete adc::add_compiled_model et appelle des
/// methodes hors-ligne de adc::System DEFINIES dans le module _adc deja charge. Le loader et le
/// module DOIVENT partager la meme ABI (memes en-tetes, meme compilateur, meme standard), sinon
/// l'agencement memoire des objets traversant la frontiere (System, GridContext, BlockClosures...)
/// diverge -> comportement indefini SILENCIEUX. On rend l'incompatibilite EXPLICITE : le loader
/// expose adc_native_abi_key() (cle figee a SA compilation) et le System compare a SA propre
/// abi_key() au chargement (add_native_block) ; un ecart leve une erreur claire au lieu d'un UB.
///
/// Construction de la cle (parallele a adc_cases/common/native.py::_abi_key) :
///   - __VERSION__   : identite + version du compilateur (g++/clang++/Apple clang...) ;
///   - __cplusplus   : standard C++ effectif (donc -std= et le mode du compilateur) ;
///   - ADC_HEADER_SIG: signature de l'arbre d'en-tetes du coeur, INJECTEE par le build (CMake cote
///                     module, flag -D cote loader) ; sa valeur change si un en-tete change, donc
///                     la cle change et l'incompatibilite est detectee. Absente (vieux build / build
///                     manuel) -> jeton litteral "unknown" : la cle reste stable et comparable, elle
///                     ne capture alors que compilateur + standard (degradation gracieuse, jamais UB
///                     silencieux car les deux cotes voient le meme "unknown" s'ils sont batis pareil).

#ifndef ADC_HEADER_SIG
#define ADC_HEADER_SIG "unknown"
#endif

// Indirection pour stringifier la valeur d'une macro (et non son nom).
#define ADC_ABI_STR_(x) #x
#define ADC_ABI_STR(x) ADC_ABI_STR_(x)

namespace adc {
namespace detail {

/// Cle d'ABI de l'UNITE DE TRADUCTION courante, calculee a SA compilation par le preprocesseur :
/// "compiler=<__VERSION__>;std=<__cplusplus>;headers=<ADC_HEADER_SIG>". inline => chaque TU qui
/// inclut cet en-tete (le module _adc ET un loader .so genere) calcule sa PROPRE cle ; deux TU
/// baties avec la meme toolchain et les memes en-tetes obtiennent la MEME chaine. C'est sur cette
/// identite que repose la comparaison add_native_block (cle du loader vs abi_key() du module).
inline std::string abi_key_string() {
  return std::string("compiler=") + __VERSION__ + ";std=" + ADC_ABI_STR(__cplusplus) +
         ";headers=" + ADC_HEADER_SIG;
}

}  // namespace detail

/// Cle d'ABI du module (TU system.cpp). ADC_EXPORT : exportee pour que add_native_block puisse lire
/// la cle du module deja charge et la comparer a celle baked dans le loader .so. Definie hors-ligne
/// (system.cpp) pour figer la cle a la compilation DU MODULE.
ADC_EXPORT std::string abi_key();

}  // namespace adc
