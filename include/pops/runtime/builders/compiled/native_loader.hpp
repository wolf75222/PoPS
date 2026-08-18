#pragma once

#include <pops/numerics/nonlinear/newton_options.hpp>
#include <pops/runtime/config/route_ids.hpp>
#include <pops/runtime/config/runtime_params.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/dynamic/dynlib.hpp>
#include <pops/runtime/system.hpp>
#include <pops/runtime/system/native_package_capability.hpp>
#include <pops/runtime/dynamic/authenticated_native_file.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace pops::native_loader {

struct AuthenticatedPackageLifetime final {
  // Members are destroyed in reverse declaration order: unload the DSO before releasing and
  // deleting the exact private backing image.
  std::shared_ptr<dynlib::AuthenticatedNativeFile> authenticated_file;
  dynlib::UniqueHandle handle;

  AuthenticatedPackageLifetime(std::shared_ptr<dynlib::AuthenticatedNativeFile> authenticated,
                               dynlib::UniqueHandle opened) noexcept
      : authenticated_file(std::move(authenticated)), handle(std::move(opened)) {}
};

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
                             const std::string& so_path, const std::string& expected_model_identity,
                             const std::string& expected_binary_identity,
                             const std::string& limiter, const std::string& riemann,
                             const std::string& recon, const std::string& time, double gamma,
                             int substeps, bool evolve, int stride,
                             const std::vector<double>& params, double positivity_floor,
                             NewtonOptions newton = {}, bool newton_diagnostics = false) {
  constexpr const char* context = "System::_install_native_block";
  if (substeps < 1)
    throw std::runtime_error(std::string(context) + ": substeps >= 1");
  if (stride < 1)
    throw std::runtime_error(std::string(context) + ": stride >= 1");
  validate_newton_options(newton, context);
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error(std::string(context) +
                             ": recon 'conservative' | 'primitive' required");
  if (time != "explicit" && time != "ssprk3" && time != "euler" && time != "imex")
    throw std::runtime_error(std::string(context) +
                             ": time 'explicit' | 'ssprk3' | 'euler' | 'imex' required");
  if (expected_model_identity.size() != 64 ||
      expected_model_identity.find_first_not_of("0123456789abcdef") != std::string::npos)
    throw std::invalid_argument(std::string(context) +
                                ": expected compiled model identity must be lowercase SHA-256");
  if (expected_binary_identity.rfind("pops.binary.v1:sha256:", 0) != 0 ||
      expected_binary_identity.size() != std::string_view("pops.binary.v1:sha256:").size() + 64 ||
      expected_binary_identity.substr(std::string_view("pops.binary.v1:sha256:").size())
              .find_first_not_of("0123456789abcdef") != std::string::npos)
    throw std::invalid_argument(std::string(context) +
                                ": expected compiled binary identity token is invalid");
  // Keep the authenticated platform image alive until the loader has opened those exact bytes.
  auto authenticated_file = std::make_shared<dynlib::AuthenticatedNativeFile>(so_path);
  const std::string observed_binary_identity = authenticated_file->binary_identity();
  if (observed_binary_identity != expected_binary_identity)
    throw std::runtime_error(std::string(context) +
                             ": compiled binary differs from the authenticated facade artifact"
                             " (path='" +
                             so_path + "', observed=" + observed_binary_identity +
                             ", expected=" + expected_binary_identity + ")");

#if defined(_WIN32)
  pops::dynlib::handle handle = pops::dynlib::open_private_image(authenticated_file->load_path());
  if (handle == nullptr)
    throw std::runtime_error(std::string(context) + ": LoadLibrary('" + so_path +
                             "'): " + pops::dynlib::last_error());
#else
  std::optional<pops::dynlib::UniqueHandle> host_promotion;
  {
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&pops::abi_key), &info) && info.dli_fname)
      host_promotion.emplace(dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD));
  }
  // Only the host module is global. Keep each content-addressed package local so two semantic
  // variants that emit the same C++ template names cannot interpose on one another under ELF.
  pops::dynlib::handle handle = pops::dynlib::open_private_image(authenticated_file->load_path());
  if (handle == nullptr) {
    throw std::runtime_error(std::string(context) + ": dlopen('" + so_path +
                             "'): " + pops::dynlib::last_error());
  }
  host_promotion.reset();
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
  auto protocol_version = reinterpret_cast<int (*)()>(
      pops::dynlib::sym(handle, runtime::system::kNativeSystemPackageAbiVersionSymbol));
  if (protocol_version == nullptr)
    throw std::runtime_error(std::string(context) + ": " +
                             runtime::system::kNativeSystemPackageAbiVersionSymbol +
                             " is missing; rebuild artifact");
  if (pops::dynlib::invoke_with_host_exception(
          [protocol_version] { return protocol_version(); },
          runtime::system::kNativeSystemPackageAbiVersionSymbol) !=
      runtime::system::kNativeSystemPackageAbiVersion)
    throw std::runtime_error(std::string(context) +
                             ": incompatible System native package protocol");
  const std::string manifest = verify_block_route_manifest(handle, context);
  verify_runtime_params(handle, params, context);
  auto model_identity = reinterpret_cast<const char* (*)()>(
      pops::dynlib::sym(handle, "pops_compiled_model_identity"));
  if (model_identity == nullptr)
    throw std::runtime_error(std::string(context) +
                             ": pops_compiled_model_identity is missing; rebuild artifact");
  const char* raw_model_identity = pops::dynlib::invoke_with_host_exception(
      [model_identity] { return model_identity(); }, "pops_compiled_model_identity");
  if (raw_model_identity == nullptr || std::string_view(raw_model_identity).size() != 64 ||
      std::string_view(raw_model_identity).find_first_not_of("0123456789abcdef") !=
          std::string_view::npos ||
      expected_model_identity != raw_model_identity)
    throw std::runtime_error(std::string(context) +
                             ": compiled model identity differs from authenticated facade model");

  // make_shared allocates its control block before moving ``opened`` into the managed object. Thus
  // allocation failure leaves the stack owner intact, while success transfers the handle exactly
  // once; no raw-pointer deleter can race the stack guard on a control-block failure.
  auto package_lifetime = std::make_shared<AuthenticatedPackageLifetime>(
      std::move(authenticated_file), std::move(opened));

  // Route registration is staged with the package rather than mutating the live registry here.
  // Finalization retains every DSO owner while it constructs, validates, and either publishes or
  // rolls back the complete provider graph.  ``pops_register_provider_routes`` is the sole System
  // symbol; there is no legacy spelling or physical-name fallback.
  using capability_state = runtime::system::NativePackageCapabilityState<Dim>;
  using route_capability = runtime::system::PreparedNativeRouteRegistrar<Dim>;
  using installer_capability = runtime::system::PreparedNativeBlockInstaller<Dim>;
  auto capability = std::make_shared<capability_state>();
  capability->identity = name;
  auto routes_capability =
      runtime::system::NativePackageCapabilityFactory<Dim>::route_registrar(capability);
  auto block_capability =
      runtime::system::NativePackageCapabilityFactory<Dim>::block_installer(capability);

  using register_provider_routes_fn = void (*)(void*);
  auto register_provider_routes = reinterpret_cast<register_provider_routes_fn>(
      pops::dynlib::sym(handle, "pops_register_provider_routes"));
  if (register_provider_routes == nullptr)
    throw std::runtime_error(std::string(context) +
                             ": pops_register_provider_routes is missing; rebuild artifact");
  std::function<void()> route_registrar = [capability, routes_capability,
                                           register_provider_routes] {
    capability->phase = runtime::system::NativeCapabilityPhase::routes_open;
    register_provider_routes(static_cast<void*>(routes_capability.get()));
    capability->close_routes();
  };

  using install_fn = void (*)(void*, const char*, const char*, const char*, const char*,
                              const char*, double, int, int, int, const double*, int, double, int,
                              double, double, double, double, int);
  auto install = reinterpret_cast<install_fn>(pops::dynlib::sym(handle, "pops_install_native"));
  if (install == nullptr) {
    throw std::runtime_error(std::string(context) +
                             ": pops_install_native is missing; rebuild artifact");
  }
  // Do not call the block installer here.  It captures the final provider carrier/consumer plan,
  // which does not exist until every package has registered its routes and the global graph seals.
  // Capturing values (not caller pointers) also makes this thunk independent from Python storage.
  std::function<void()> thunk = [capability, block_capability, install, name, limiter, riemann,
                                 recon, time, gamma, substeps, evolve, stride, params,
                                 positivity_floor, newton, newton_diagnostics] {
    const double* data = params.empty() ? nullptr : params.data();
    install(static_cast<void*>(block_capability.get()), name.c_str(), limiter.c_str(),
            riemann.c_str(), recon.c_str(), time.c_str(), gamma, substeps, evolve ? 1 : 0, stride,
            data, static_cast<int>(params.size()), positivity_floor, newton.max_iters,
            static_cast<double>(newton.rel_tol), static_cast<double>(newton.abs_tol),
            static_cast<double>(newton.fd_eps), static_cast<double>(newton.damping),
            newton_diagnostics ? 1 : 0);
    if (!capability->commit_called || !capability->committed)
      throw std::logic_error("native package installer did not commit one complete package");
  };
  ExactContractBuilder contract;
  contract.text("pops.system-native-package")
      .scalar(std::uint32_t{3})
      .scalar(std::int32_t{Dim})
      .text(name)
      .text(expected_model_identity)
      .text(expected_binary_identity)
      .text(module_key)
      .text(manifest)
      .text(limiter)
      .text(riemann)
      .text(recon)
      .text(time)
      .scalar(gamma)
      .scalar(std::int32_t{substeps})
      .scalar(evolve)
      .scalar(std::int32_t{stride})
      .scalar(positivity_floor)
      .scalar(std::int32_t{newton.max_iters})
      .scalar(static_cast<double>(newton.rel_tol))
      .scalar(static_cast<double>(newton.abs_tol))
      .scalar(static_cast<double>(newton.fd_eps))
      .scalar(static_cast<double>(newton.damping))
      .scalar(newton_diagnostics)
      .sequence(params);
  system->stage_prepared_native_package(std::move(contract).release(), std::move(route_registrar),
                                        std::move(thunk), std::move(package_lifetime), capability);
}

}  // namespace pops::native_loader
