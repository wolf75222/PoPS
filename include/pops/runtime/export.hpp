#pragma once

/// @file
/// @brief POPS_EXPORT: publish a runtime seam from the module which owns its implementation.
///
/// Used by the DSL "production" path: a generated .so loader, dlopen-ed at run time
/// (System::add_native_block), includes the add_compiled_model header template which calls
/// OUT-OF-LINE methods of pops::System (install_block / grid_context / ensure_aux_width) DEFINED in the
/// already-loaded _pops module. Without default visibility, these symbols do not appear in the
/// dynamic table of the module and the loader CANNOT resolve them (link failure at dlopen).
/// The same macro may annotate an exception class when POPS_RUNTIME_SHARED_EXCEPTION_ABI explicitly
/// selects a DSO contract. Such a class owns an exported out-of-line key function in the module so
/// generated loaders reference one canonical vtable/typeinfo instead of emitting DSO-local copies.
/// Without that switch the exception remains an ordinary inline, header-only pops::pops type.

// Windows shared-ABI producers (the central runtime targets and _pops adapter) define both
// POPS_RUNTIME_SHARED_EXCEPTION_ABI and POPS_EXPORT_BUILDING_MODULE -> dllexport; generated .dll
// consumers replay only the former -> dllimport. Unix producers and consumers use default
// visibility even though _pops itself is built with hidden visibility. See ADC-99 (portable layer).
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
