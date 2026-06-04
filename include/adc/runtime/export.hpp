#pragma once

/// @file
/// @brief ADC_EXPORT : force la VISIBILITE DEFAUT d'un symbole hors-ligne, meme quand l'unite est
///        compilee en -fvisibility=hidden (cas du module pybind11 _adc).
///
/// Sert au chemin "production" du DSL : un loader .so genere, dlopen-e a l'execution
/// (System::add_native_block), inclut le gabarit en-tete add_compiled_model qui appelle des methodes
/// HORS-LIGNE de adc::System (install_block / grid_context / ensure_aux_width) DEFINIES dans le
/// module _adc deja charge. Sans visibilite defaut, ces symboles ne figurent pas dans la table
/// dynamique du module et le loader ne peut PAS les resoudre (echec d'edition de liens au dlopen).
/// On exporte donc EXACTEMENT ces methodes + adc::abi_key (surface minimale). MSVC / Windows : sans
/// effet ici (chemin POSIX dlopen ; un futur portage Windows utiliserait __declspec(dllexport)).

#if defined(_WIN32)
#define ADC_EXPORT
#else
#define ADC_EXPORT __attribute__((visibility("default")))
#endif
