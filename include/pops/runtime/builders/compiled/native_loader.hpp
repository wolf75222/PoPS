#pragma once

#include <pops/runtime/config/route_ids.hpp>
#include <pops/runtime/config/runtime_params.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/dynamic/dynlib.hpp>
#include <pops/runtime/system.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace pops::native_loader {

inline void verify_block_route_manifest(pops::dynlib::handle handle, const char* context) {
  auto manifest = reinterpret_cast<const char* (*)()>(
      pops::dynlib::sym(handle, "pops_compiled_route_manifest"));
  if (manifest == nullptr) {
    pops::dynlib::close(handle);
    throw std::runtime_error(std::string(context) +
                             ": pops_compiled_route_manifest is missing; rebuild artifact");
  }
  const char* raw = manifest();
  if (raw == nullptr) {
    pops::dynlib::close(handle);
    throw std::runtime_error(std::string(context) + ": pops_compiled_route_manifest returned null");
  }
  try {
    pops::verify_route_manifest(raw, context);
  } catch (...) {
    pops::dynlib::close(handle);
    throw;
  }
}

inline int csv_field_count(const char* raw) {
  if (raw == nullptr || *raw == '\0')
    return 0;
  int count = 1;
  for (const char* cursor = raw; *cursor != '\0'; ++cursor)
    if (*cursor == ',')
      ++count;
  return count;
}

inline void verify_runtime_params(pops::dynlib::handle handle, const std::vector<double>& values,
                                  const char* context) {
  auto count = reinterpret_cast<int (*)()>(pops::dynlib::sym(handle, "pops_compiled_nparams"));
  auto names =
      reinterpret_cast<const char* (*)()>(pops::dynlib::sym(handle, "pops_compiled_param_names"));
  if (count == nullptr || names == nullptr) {
    pops::dynlib::close(handle);
    throw std::runtime_error(std::string(context) +
                             ": compiled parameter metadata is missing; rebuild artifact");
  }
  const int expected = count();
  const char* raw_names = names();
  if (expected < 0 || expected > kMaxRuntimeParams) {
    pops::dynlib::close(handle);
    throw std::runtime_error(
        std::string(context) + ": artifact declares " + std::to_string(expected) +
        " runtime parameters; supported range is 0.." + std::to_string(kMaxRuntimeParams));
  }
  if (raw_names == nullptr || csv_field_count(raw_names) != expected) {
    pops::dynlib::close(handle);
    throw std::runtime_error(std::string(context) +
                             ": compiled parameter names disagree with nparams");
  }
  if (values.size() != static_cast<std::size_t>(expected)) {
    pops::dynlib::close(handle);
    throw std::runtime_error(std::string(context) + ": received " + std::to_string(values.size()) +
                             " bound parameters but artifact requires " + std::to_string(expected));
  }
}

template <typename ImplT>
void add_native_block(System* system, ImplT*, const std::string& name, const std::string& so_path,
                      const std::string& limiter, const std::string& riemann,
                      const std::string& recon, const std::string& time, double gamma, int substeps,
                      bool evolve, int stride, const std::vector<double>& params,
                      double positivity_floor) {
  constexpr const char* context = "System::_install_native_block";
  if (substeps < 1)
    throw std::runtime_error(std::string(context) + ": substeps >= 1");
  if (stride < 1)
    throw std::runtime_error(std::string(context) + ": stride >= 1");
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error(std::string(context) +
                             ": recon 'conservative' | 'primitive' required");
  if (time != "explicit" && time != "ssprk3" && time != "euler" && time != "imex")
    throw std::runtime_error(std::string(context) +
                             ": time 'explicit' | 'ssprk3' | 'euler' | 'imex' required");

#if defined(_WIN32)
  pops::dynlib::handle handle = pops::dynlib::open(so_path);
  if (handle == nullptr)
    throw std::runtime_error(std::string(context) + ": LoadLibrary('" + so_path +
                             "'): " + pops::dynlib::last_error());
#else
  {
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&pops::abi_key), &info) && info.dli_fname)
      dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
  }
  // Only the host module is global. Keep each content-addressed package local so two semantic
  // variants that emit the same C++ template names cannot interpose on one another under ELF.
  pops::dynlib::handle handle = pops::dynlib::open(so_path);
  if (handle == nullptr) {
    throw std::runtime_error(std::string(context) + ": dlopen('" + so_path +
                             "'): " + pops::dynlib::last_error());
  }
#endif

  auto key = reinterpret_cast<const char* (*)()>(pops::dynlib::sym(handle, "pops_native_abi_key"));
  if (key == nullptr) {
    pops::dynlib::close(handle);
    throw std::runtime_error(std::string(context) +
                             ": pops_native_abi_key is missing; rebuild artifact");
  }
  const std::string artifact_key = key();
  const std::string module_key = abi_key();
  if (artifact_key != module_key) {
    pops::dynlib::close(handle);
    throw std::runtime_error(std::string(context) + ": incompatible native ABI: artifact '" +
                             artifact_key + "' != module '" + module_key + "'");
  }
  verify_block_route_manifest(handle, context);
  verify_runtime_params(handle, params, context);

  using install_fn = void (*)(void*, const char*, const char*, const char*, const char*,
                              const char*, double, int, int, int, const double*, int, double);
  auto install = reinterpret_cast<install_fn>(pops::dynlib::sym(handle, "pops_install_native"));
  if (install == nullptr) {
    pops::dynlib::close(handle);
    throw std::runtime_error(std::string(context) +
                             ": pops_install_native is missing; rebuild artifact");
  }
  const double* data = params.empty() ? nullptr : params.data();
  install(static_cast<void*>(system), name.c_str(), limiter.c_str(), riemann.c_str(), recon.c_str(),
          time.c_str(), gamma, substeps, evolve ? 1 : 0, stride, data,
          static_cast<int>(params.size()), positivity_floor);
  // The installed closures execute code owned by the package, so the handle stays loaded.
}

}  // namespace pops::native_loader
