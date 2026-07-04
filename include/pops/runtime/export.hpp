#pragma once

/// @file
/// @brief POPS_EXPORT: force DEFAULT VISIBILITY on an out-of-line symbol, even when the unit is
///        compiled with -fvisibility=hidden (case of the pybind11 _pops module).
///
/// Used by the DSL "production" path: a generated .so loader, dlopen-ed at run time
/// (System::add_native_block), includes the add_compiled_model header template which calls
/// OUT-OF-LINE methods of pops::System (install_block / grid_context / ensure_aux_width) DEFINED in the
/// already-loaded _pops module. Without default visibility, these symbols do not appear in the
/// dynamic table of the module and the loader CANNOT resolve them (link failure at dlopen).
/// We therefore export EXACTLY these methods + pops::abi_key (minimal surface). MSVC / Windows: no
/// effect here (POSIX dlopen path; a future Windows port would use __declspec(dllexport)).

// Windows: the _pops module (which DEFINES these symbols) must define POPS_EXPORT_BUILDING_MODULE at its
// compilation -> dllexport; the generated .dll loader that IMPORTS them falls back on dllimport. Unix:
// default visibility (the module is compiled with -fvisibility=hidden). See ADC-99 (portable layer).
#if defined(_WIN32)
#if defined(POPS_EXPORT_BUILDING_MODULE)
#define POPS_EXPORT __declspec(dllexport)
#else
#define POPS_EXPORT __declspec(dllimport)
#endif
#else
#define POPS_EXPORT __attribute__((visibility("default")))
#endif

/// @brief POPS_BRICK_LOCAL: force HIDDEN VISIBILITY on a symbol so it is NOT unified across
///        independently dlopen'd shared objects, even under RTLD_LOCAL (ADC-622).
///
/// The DUAL of POPS_EXPORT. It exists for the external-brick registry (external_brick.hpp): each
/// user brick .so registers its manifest into a header-only Meyers singleton (BrickRegistry). On
/// Linux/GCC that function-local static gets DEFAULT visibility + vague linkage, which the compiler
/// emits as STB_GNU_UNIQUE; glibc's loader then UNIFIES it across every dlopen'd image (even under
/// RTLD_LOCAL), so one brick .so's pops_brick_manifest() would report the whole process's bricks.
/// Marking the class/method hidden keeps the symbol out of the dynamic symbol table -> GCC does NOT
/// emit STB_GNU_UNIQUE and the loader cannot unify or RTLD_GLOBAL-interpose it, so each .so keeps a
/// PRIVATE registry (comdat still folds copies WITHIN one .so; each .so is isolated ACROSS images).
/// GCC / Clang: __attribute__((visibility("hidden"))). MSVC / Windows: empty (the two-level
/// namespace / .dll model already isolates per-module symbols, and there is no GNU_UNIQUE).
#if defined(_WIN32)
#define POPS_BRICK_LOCAL
#else
#define POPS_BRICK_LOCAL __attribute__((visibility("hidden")))
#endif
