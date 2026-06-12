#pragma once

/// @file
/// @brief Chargement dynamique PORTABLE : `dlopen`/`dlsym`/`dlclose` (POSIX) <-> `LoadLibraryW`/
///        `GetProcAddress`/`FreeLibrary` (Windows). Surface minimale pour le runtime adc (loader
///        natif / DSL `.dll`), chantier ADC-99.
///
/// POSIX : strictement equivalent a l'usage historique (`RTLD_NOW | RTLD_LOCAL`). Le chemin
/// "production" du DSL a un besoin SPECIAL (promotion `RTLD_GLOBAL` pour resoudre les symboles de
/// `_adc` exportes `ADC_EXPORT` a travers le dlopen) qui reste gere a son site d'appel ; cote
/// Windows l'equivalent passe par `__declspec(dllexport)` (cf. export.hpp) + import library, ADC-100.

#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace adc {
namespace dynlib {

#if defined(_WIN32)
using handle = HMODULE;
#else
using handle = void*;
#endif

/// Suffixe de bibliotheque dynamique de la plateforme.
inline const char* suffix() {
#if defined(_WIN32)
  return ".dll";
#elif defined(__APPLE__)
  return ".dylib";
#else
  return ".so";
#endif
}

/// Ouvre une bibliotheque dynamique (@p path en UTF-8). Renvoie un handle nul si echec.
inline handle open(const std::string& path) {
#if defined(_WIN32)
  // UTF-8 -> UTF-16 pour LoadLibraryW (chemins Unicode et avec espaces).
  const int n = ::MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
  std::wstring w(n > 0 ? n - 1 : 0, L'\0');
  if (n > 0) ::MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), n);
  return ::LoadLibraryW(w.c_str());
#else
  return ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

/// Resout @p name dans @p h. Renvoie nullptr si absent.
inline void* sym(handle h, const char* name) {
#if defined(_WIN32)
  return reinterpret_cast<void*>(::GetProcAddress(h, name));
#else
  return ::dlsym(h, name);
#endif
}

/// Ferme @p h (no-op sur handle nul).
inline void close(handle h) {
#if defined(_WIN32)
  if (h) ::FreeLibrary(h);
#else
  if (h) ::dlclose(h);
#endif
}

/// true si @p h est un handle valide.
inline bool valid(handle h) { return h != handle{}; }

/// Message de la derniere erreur (best-effort, pour les diagnostics).
inline std::string last_error() {
#if defined(_WIN32)
  const DWORD e = ::GetLastError();
  if (!e) return {};
  char* buf = nullptr;
  ::FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                       FORMAT_MESSAGE_IGNORE_INSERTS,
                   nullptr, e, 0, reinterpret_cast<char*>(&buf), 0, nullptr);
  std::string m = buf ? buf : "";
  if (buf) ::LocalFree(buf);
  return m;
#else
  const char* e = ::dlerror();
  return e ? std::string(e) : std::string();
#endif
}

}  // namespace dynlib
}  // namespace adc
