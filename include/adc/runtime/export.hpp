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

// Windows : le module _adc (qui DEFINIT ces symboles) doit definir ADC_EXPORT_BUILDING_MODULE a sa
// compilation -> dllexport ; le loader .dll genere qui les IMPORTE retombe sur dllimport. Unix :
// visibilite par defaut (le module est compile -fvisibility=hidden). cf. ADC-99 (couche portable).
#if defined(_WIN32)
#if defined(ADC_EXPORT_BUILDING_MODULE)
#define ADC_EXPORT __declspec(dllexport)
#else
#define ADC_EXPORT __declspec(dllimport)
#endif
#else
#define ADC_EXPORT __attribute__((visibility("default")))
#endif
