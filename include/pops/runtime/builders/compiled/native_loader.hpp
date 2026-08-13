#pragma once

#include <pops/runtime/config/route_ids.hpp>
#include <pops/runtime/config/runtime_params.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/dynamic/dynlib.hpp>
#include <pops/runtime/system.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pops::native_loader {

inline std::string verify_block_route_manifest(pops::dynlib::handle handle, const char* context) {
  auto manifest = reinterpret_cast<const char* (*)()>(
      pops::dynlib::sym(handle, "pops_compiled_route_manifest"));
  if (manifest == nullptr)
    throw std::runtime_error(std::string(context) +
                             ": pops_compiled_route_manifest is missing; rebuild artifact");
  const char* raw = pops::dynlib::invoke_with_host_exception([manifest] { return manifest(); },
                                                             "pops_compiled_route_manifest");
  if (raw == nullptr)
    throw std::runtime_error(std::string(context) + ": pops_compiled_route_manifest returned null");
  std::string owned(raw);
  pops::verify_route_manifest(owned, context);
  return owned;
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
  if (count == nullptr || names == nullptr)
    throw std::runtime_error(std::string(context) +
                             ": compiled parameter metadata is missing; rebuild artifact");
  const int expected = pops::dynlib::invoke_with_host_exception([count] { return count(); },
                                                                "pops_compiled_nparams");
  const char* raw_names = pops::dynlib::invoke_with_host_exception([names] { return names(); },
                                                                   "pops_compiled_param_names");
  if (expected < 0 || expected > kMaxRuntimeParams)
    throw std::runtime_error(
        std::string(context) + ": artifact declares " + std::to_string(expected) +
        " runtime parameters; supported range is 0.." + std::to_string(kMaxRuntimeParams));
  if (raw_names == nullptr || csv_field_count(raw_names) != expected)
    throw std::runtime_error(std::string(context) +
                             ": compiled parameter names disagree with nparams");
  if (values.size() != static_cast<std::size_t>(expected))
    throw std::runtime_error(std::string(context) + ": received " + std::to_string(values.size()) +
                             " bound parameters but artifact requires " + std::to_string(expected));
}

template <int Dim>
void register_native_package(System<Dim>* system, const std::string& name,
                             const std::string& so_path, const std::string& limiter,
                             const std::string& riemann, const std::string& recon,
                             const std::string& time, double gamma, int substeps, bool evolve,
                             int stride, const std::vector<double>& params,
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

  // Own the successful dlopen before invoking any artifact callback or allocating host metadata.
  pops::dynlib::UniqueHandle opened(handle);

  auto key = reinterpret_cast<const char* (*)()>(pops::dynlib::sym(handle, "pops_native_abi_key"));
  if (key == nullptr)
    throw std::runtime_error(std::string(context) +
                             ": pops_native_abi_key is missing; rebuild artifact");
  const char* raw_artifact_key =
      pops::dynlib::invoke_with_host_exception([key] { return key(); }, "pops_native_abi_key");
  if (raw_artifact_key == nullptr)
    throw std::runtime_error(std::string(context) + ": pops_native_abi_key returned null");
  const std::string artifact_key = raw_artifact_key;
  const std::string module_key = abi_key();
  if (artifact_key != module_key)
    throw std::runtime_error(std::string(context) + ": incompatible native ABI: artifact '" +
                             artifact_key + "' != module '" + module_key + "'");
  const std::string manifest = verify_block_route_manifest(handle, context);
  verify_runtime_params(handle, params, context);

  // make_shared allocates its control block before moving ``opened`` into the managed object. Thus
  // allocation failure leaves the stack owner intact, while success transfers the handle exactly
  // once; no raw-pointer deleter can race the stack guard on a control-block failure.
  auto package_lifetime = std::make_shared<pops::dynlib::UniqueHandle>(std::move(opened));

  // Route registration is staged with the package rather than mutating the live registry here.
  // Finalization retains every DSO owner while it constructs, validates, and either publishes or
  // rolls back the complete provider graph.  ``pops_register_provider_routes`` is the sole System
  // symbol; there is no legacy spelling or physical-name fallback.
  using register_provider_routes_fn = void (*)(System<Dim>*);
  auto register_provider_routes = reinterpret_cast<register_provider_routes_fn>(
      pops::dynlib::sym(handle, "pops_register_provider_routes"));
  if (register_provider_routes == nullptr)
    throw std::runtime_error(std::string(context) +
                             ": pops_register_provider_routes is missing; rebuild artifact");
  std::function<void()> route_registrar = [system, register_provider_routes] {
    register_provider_routes(system);
  };

  using install_fn = void (*)(void*, const char*, const char*, const char*, const char*,
                              const char*, double, int, int, int, const double*, int, double);
  auto install = reinterpret_cast<install_fn>(pops::dynlib::sym(handle, "pops_install_native"));
  if (install == nullptr) {
    throw std::runtime_error(std::string(context) +
                             ": pops_install_native is missing; rebuild artifact");
  }
  // Do not call the block installer here.  It captures the final provider carrier/consumer plan,
  // which does not exist until every package has registered its routes and the global graph seals.
  // Capturing values (not caller pointers) also makes this thunk independent from Python storage.
  std::function<void()> thunk = [system, install, name, limiter, riemann, recon, time, gamma,
                                 substeps, evolve, stride, params, positivity_floor] {
    const double* data = params.empty() ? nullptr : params.data();
    install(static_cast<void*>(system), name.c_str(), limiter.c_str(), riemann.c_str(),
            recon.c_str(), time.c_str(), gamma, substeps, evolve ? 1 : 0, stride, data,
            static_cast<int>(params.size()), positivity_floor);
  };
  system->stage_prepared_native_package(name + "\n" + manifest, std::move(route_registrar),
                                        std::move(thunk), std::move(package_lifetime));
}

}  // namespace pops::native_loader
