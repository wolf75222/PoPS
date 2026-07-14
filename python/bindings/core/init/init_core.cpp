#include "../bindings_detail.hpp"

#include <pops/core/state/aux_names.hpp>  // ADC-291: canonical aux name<->component table + bounds
#include <pops/runtime/config/runtime_params.hpp>  // ADC-610: kMaxRuntimeParams (mirrored to Python)
#include <pops/runtime/config/platform_manifest.hpp>  // ADC-683: explicit launch contracts
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/module_capabilities.hpp>  // ADC-479 (#36/#37): authoritative static capability facts
#include <pops/runtime/runtime_environment.hpp>  // ADC-609: runtime environment/precision/communicator report

#include <utility>

namespace pops::detail {

/// Binding-only friend used by the capability-gated Python freeze transaction.
///
/// The public C++ ModelSpec API deliberately has no restore operation: freeze() is irreversible
/// for native callers.  This access type is defined only in the binding translation unit, so an
/// ordinary consumer of the public header cannot name or invoke a rollback operation.
struct ModelSpecFreezeTransactionAccess {
  [[nodiscard]] static bool snapshot(const ModelSpec& spec) noexcept { return spec.frozen_; }
  static void restore(ModelSpec& spec, bool state) noexcept { spec.frozen_ = state; }
};

}  // namespace pops::detail

namespace {

void require_freeze_transaction_capability(const py::handle& capability) {
  const py::object expected =
      py::module_::import("pops.problem._freeze_transaction").attr("_FREEZE_CAPABILITY");
  if (!capability.is(expected))
    throw std::runtime_error(
        "ModelSpec freeze rollback requires the private PoPS transaction capability");
}

template <typename Field>
void bind_model_spec_property(py::class_<pops::ModelSpec>& binding, const char* name,
                              Field pops::ModelSpec::* member) {
  using T = typename Field::value_type;
  binding.def_property(
      name, [member](const pops::ModelSpec& spec) { return T((spec.*member).get()); },
      [member](pops::ModelSpec& spec, T value) { spec.*member = std::move(value); });
}

pops::CapabilityTarget parse_capability_target(const std::string& target, const char* where) {
  if (target == "production")
    return pops::CapabilityTarget::kProduction;
  if (target == "module" || target.empty())
    return pops::CapabilityTarget::kModule;
  throw std::invalid_argument(std::string(where) +
                              ": target must be 'module' or 'production' (got '" + target + "')");
}

py::dict runtime_environment_to_dict(const pops::RuntimeEnvironmentReport& r) {
  py::dict d;
  d["dimension"] = r.dimension;
  d["amr_refinement_ratio"] = r.amr_refinement_ratio;
  d["precision"] = r.precision;
  d["real_bytes"] = r.real_bytes;
  d["supports_single_precision"] = r.supports_single_precision;
  d["supports_mixed_precision"] = r.supports_mixed_precision;
  d["has_kokkos"] = r.has_kokkos;
  d["kokkos_initialized"] = r.kokkos_initialized;
  d["kokkos_finalized"] = r.kokkos_finalized;
  d["kokkos_initialized_by_pops"] = r.kokkos_initialized_by_pops;
  d["kokkos_atexit_finalize_registered"] = r.kokkos_atexit_finalize_registered;
  d["kokkos_backend"] = r.kokkos_backend;
  d["kokkos_ownership"] = r.kokkos_ownership;
  d["kokkos_lifecycle"] = r.kokkos_lifecycle;
  d["mpi_compiled"] = r.mpi_compiled;
  d["mpi_active"] = r.mpi_active;
  d["mpi_rank"] = r.mpi_rank;
  d["mpi_ranks"] = r.mpi_ranks;
  d["communicator"] = r.communicator;
  d["supports_custom_communicator"] = r.supports_custom_communicator;
  d["allocator_mode"] = r.allocator_mode;
  d["comm_allocator_mode"] = r.comm_allocator_mode;
  d["allocator_lifetime"] = r.allocator_lifetime;
  return d;
}

py::dict serial_runtime_backend_manifest_to_dict(const std::string& backend,
                                                 const std::string& target) {
  const auto manifest = pops::platform::proven_serial_backend(
      backend, target, pops::abi_key());
  py::dict precision;
  precision["storage"] = pops::platform::require_text(
      manifest.precision.storage, "precision.storage");
  precision["compute"] = pops::platform::require_text(
      manifest.precision.compute, "precision.compute");
  precision["accumulation"] = pops::platform::require_text(
      manifest.precision.accumulation, "precision.accumulation");
  precision["reduction"] = pops::platform::require_text(
      manifest.precision.reduction, "precision.reduction");
  py::dict capabilities;
  capabilities["dimensions"] = pops::platform::require_int_set(
      pops::platform::capability(manifest, "dimensions"), "capabilities.dimensions");
  capabilities["centerings"] = pops::platform::require_text_set(
      pops::platform::capability(manifest, "centerings"), "capabilities.centerings");
  capabilities["scalars"] = pops::platform::require_text_set(
      pops::platform::capability(manifest, "scalars"), "capabilities.scalars");
  capabilities["layouts"] = pops::platform::require_text_set(
      pops::platform::capability(manifest, "layouts"), "capabilities.layouts");
  capabilities["ownership"] = pops::platform::require_text_set(
      pops::platform::capability(manifest, "ownership"), "capabilities.ownership");
  capabilities["generic_field_view"] = true;
  py::dict result;
  result["schema_version"] = pops::platform::kPlatformContractSchemaVersion;
  result["backend"] = pops::platform::require_text(manifest.backend, "backend");
  result["target"] = pops::platform::require_text(manifest.target, "target");
  result["abi"] = pops::platform::require_text(manifest.abi, "abi");
  result["precision"] = std::move(precision);
  result["device"] = pops::platform::require_text(manifest.device, "device");
  result["memory_spaces"] = pops::platform::require_text_set(
      manifest.memory_spaces, "memory_spaces");
  result["communicator"] = pops::platform::require_text(
      manifest.communicator, "communicator");
  result["capabilities"] = std::move(capabilities);
  result["evidence"] = "pops.native.2d-float64-host.v1";
  result["identity"] = pops::platform::identity_token(
      "runtime-backend-manifest", manifest);
  return result;
}

py::dict module_capabilities_to_dict(const pops::ModuleCapabilities& c,
                                     const pops::RuntimeEnvironmentReport& env) {
  py::dict d;
  d["abi_version"] = c.abi_version;
  d["supports_uniform"] = c.supports_uniform;
  d["supports_amr"] = c.supports_amr;
  d["supports_mpi"] = c.supports_mpi;
  d["supports_gpu"] = c.supports_gpu;
  d["supports_stride"] = c.supports_stride;
  d["supports_named_fields"] = c.supports_named_fields;
  d["supports_partial_imex_mask"] = c.supports_partial_imex_mask;
  d["dimension"] = env.dimension;
  d["amr_refinement_ratio"] = env.amr_refinement_ratio;
  d["precision"] = env.precision;
  d["real_bytes"] = env.real_bytes;
  d["communicator"] = env.communicator;
  d["supports_custom_communicator"] = env.supports_custom_communicator;
  return d;
}

py::dict capability_route_to_dict(const pops::CapabilityRouteReport& row) {
  py::dict d;
  d["route_id"] = row.route_id;
  d["feature"] = row.feature;
  d["layout"] = row.layout;
  d["backend"] = row.backend;
  d["platform"] = row.platform;
  d["mpi"] = row.mpi;
  d["gpu"] = row.gpu;
  d["status"] = row.status;
  d["reason"] = row.reason;
  d["limitation"] = row.reason;
  d["requested"] = row.requested;
  d["available_route"] = row.available_route;
  d["alternative"] = row.alternative;
  d["source"] = row.source;
  return d;
}

py::dict native_capability_report_to_dict(const pops::NativeCapabilityReport& report) {
  py::list routes;
  for (const auto& row : report.routes)
    routes.append(capability_route_to_dict(row));
  py::dict d;
  d["schema_version"] = report.schema_version;
  d["abi_version"] = report.abi_version;
  d["target"] = report.target;
  d["abi_key"] = report.abi_key;
  d["platform"] = report.platform;
  d["capabilities"] = module_capabilities_to_dict(report.capabilities, report.runtime);
  d["runtime"] = runtime_environment_to_dict(report.runtime);
  d["routes"] = routes;
  return d;
}

}  // namespace

// ADC-365: module attributes/globals + SystemConfig + ModelSpec (registered first so System/
// AmrSystem signatures resolve them).
void init_core(py::module_& m) {
  m.doc() =
      "PoPS (lib): runtime multi-species composition. System composes a "
      "system block by block; the compute stays compiled C++.";

  // Module ABI key (compiler + C++ standard + signature of the pops headers). The DSL reads it
  // (diagnostic); add_native_block compares it to the key baked into a native loader.
  m.def("abi_key", &pops::abi_key,
        "Module ABI key (compiler, C++ standard, signature of the pops headers).");

  // MPI rank / rank count of the communicator (0 / 1 in serial or when MPI is not initialized, cf.
  // pops/parallel/comm.hpp). Exposed so the IO facade (sim.write / sim.checkpoint) writes the file
  // only on rank 0 after a collective gather (state_global / potential_global).
  m.def("my_rank", &pops::my_rank, "MPI rank of the process (0 in serial).");
  m.def("n_ranks", &pops::n_ranks, "Number of MPI ranks (1 in serial).");

  // C++ standard of the LOADER (POPS_CXX_STD injected by the build: 20 under Kokkos, 23 otherwise). The
  // DSL backend="production" MUST compile the native model with this SAME standard, otherwise __cplusplus
  // diverges -> different ABI key -> add_native_block raises "incompatible ABI". We expose it as an
  // integer (20/23); dsl.compile derives the -std=c++NN flag from it instead of hardcoding c++23.
#ifdef POPS_CXX_STD
  m.attr("__cxx_std__") = static_cast<int>(POPS_CXX_STD);
#else
  // Manual build without -DPOPS_CXX_STD: we fall back on __cplusplus to stay consistent with the ABI
  // key (which itself always encodes __cplusplus). 202002L -> 20, beyond -> 23.
  m.attr("__cxx_std__") = static_cast<int>(__cplusplus > 202002L ? 23 : 20);
#endif

  // Compute backend COMPILED into the module: True if _pops was built with Kokkos
  // (-DPOPS_USE_KOKKOS=ON -> POPS_HAS_KOKKOS), hence capable of multi-thread (OpenMP device) / GPU.
  // Runtime diagnostics use it to report whether threaded/device execution is available. A serial
  // build exposes False; no false negative.
#ifdef POPS_HAS_KOKKOS
  m.attr("__has_kokkos__") = true;
#else
  m.attr("__has_kokkos__") = false;
#endif

  // MPI seam COMPILED into the module (POPS_HAS_MPI via the pops INTERFACE under -DPOPS_USE_MPI=ON) plus
  // the MPI include dir(s) used by the build (POPS_MPI_INCLUDE, baked by CMake; '|'-joined). The DSL
  // Production packages are compiled OUTSIDE CMake and inherit none of this: codegen reads these
  // attributes (_native_mpi_flags) to re-bake -DPOPS_HAS_MPI + -I<inc> so the loader uses comm.hpp's
  // REAL MPI rather than its serial stubs (n_ranks()=1). Without it a distributed layout built inside
  // the loader replicates on every rank (ADC-319). A serial module exposes False / empty.
#if defined(POPS_HAS_MPI)
  m.attr("__has_mpi__") = true;
#if defined(POPS_MPI_INCLUDE)
  m.attr("__mpi_include__") = POPS_MPI_INCLUDE;
#else
  m.attr("__mpi_include__") = "";
#endif
#else
  m.attr("__has_mpi__") = false;
  m.attr("__mpi_include__") = "";
#endif

  // Path of the COMPILER that built this module (POPS_CXX_COMPILER, injected by CMake). Since the ABI
  // key encodes __VERSION__, the "production" DSL MUST recompile its loaders with THIS compiler:
  // dsl.py prefers it to the PATH's `which c++` (which, in a conda env, often designates another
  // compiler -> "-std=c++23 invalid" or ABI rejection). Manual build without -D: empty string, dsl.py
  // then falls back on its historical detection.
#ifdef POPS_CXX_COMPILER
  m.attr("__cxx_compiler__") = POPS_CXX_COMPILER;
#else
  m.attr("__cxx_compiler__") = "";
#endif

  // Project version (POPS_VERSION = CMake PROJECT_VERSION, single source). Re-exposed as
  // pops.__version__ by the package; "unknown" on a manual build without -D.
#ifdef POPS_VERSION
  m.attr("__version__") = POPS_VERSION;
#else
  m.attr("__version__") = "unknown";
#endif
  m.attr("__release_contract_sha256__") = pops::release_contract::kContractSha256;
  m.attr("__public_api_version__") = pops::release_contract::kPublicApiVersion;
  m.attr("__semantic_ir_version__") = pops::release_contract::kSemanticIrVersion;
  m.attr("__normalization_version__") = pops::release_contract::kNormalizationVersion;
  m.attr("__component_registry_version__") = pops::release_contract::kComponentRegistryVersion;
  m.attr("__checkpoint_schema_version__") =
      pops::release_contract::kCheckpointEnvelopeSchemaVersion;

  // AUTHORITATIVE STATIC capability facts (Spec 5 sec.13.12 / sec.13.12.1, criteria #36/#37). The
  // (backend, layout, platform) transport capabilities the built module ACTUALLY provides, sourced from
  // pops::module_capabilities() -- the SAME compile-time tokens as the attrs above (POPS_HAS_KOKKOS /
  // POPS_HAS_MPI), never a Python computation. pops._capabilities.inspect_capabilities cross-checks its
  // descriptor walk against this so the two cannot SILENTLY disagree; problem.explain_routes sources the
  // route matrix from it. We expose kAbiVersion separately (it versions the capability vocabulary, not
  // the toolchain ABI key) and module_capabilities(target) returns a plain dict (route-dependent: the
  // production package carries a stride while the route-agnostic module report does not).
  m.attr("__abi_version__") = static_cast<int>(pops::kAbiVersion);
  m.def(
      "module_capabilities",
      [](const std::string& target) {
        const pops::NativeCapabilityReport report =
            pops::native_capability_report(parse_capability_target(target, "module_capabilities"));
        return module_capabilities_to_dict(report.capabilities, report.runtime);
      },
      py::arg("target") = "module",
      "Authoritative static capability facts of the built module (Spec 5 sec.13.12, #36): "
      "{abi_version, supports_uniform/amr/mpi/gpu/stride/named_fields/partial_imex_mask}, sourced "
      "from the C++ compile-time tokens. target in {'module','production'} selects whether "
      "route-specific production facts are included.");

  m.def(
      "capability_report",
      [](const std::string& target) {
        return native_capability_report_to_dict(
            pops::native_capability_report(parse_capability_target(target, "capability_report")));
      },
      py::arg("target") = "module",
      "Structured native capability report: schema_version, ABI, runtime facts, capability flags "
      "and "
      "route rows. Pretty strings are views of this object; callers should not parse text "
      "reports.");

  m.def(
      "runtime_environment_report",
      []() { return runtime_environment_to_dict(pops::runtime_environment_report()); },
      "Runtime environment facts: Kokkos lifecycle/ownership, MPI communicator, precision and "
      "allocator lifetime. Reading it does not initialize Kokkos or MPI.");

  m.def(
      "runtime_backend_manifest", &serial_runtime_backend_manifest_to_dict,
      py::arg("backend"), py::arg("target"),
      "Explicit 2D/float64/host RuntimeBackendManifest. It captures no global MPI/device state; "
      "non-serial/non-host routes must supply their own ExecutionContext.");

  m.def(
      "numerical_defaults_report", []() { return numerical_defaults_report_to_dict(); },
      "Structured native numerical/solver/physical defaults. Reading it is metadata-only.");

  m.def(
      "fallback_diagnostics_report",
      []() { return fallback_diagnostics_report_to_dict(pops::fallback_diagnostics_report()); },
      "Structured fallback/degraded-route diagnostics and policies. Reading it is metadata-only.");
  m.def("reset_fallback_diagnostics", &pops::reset_fallback_diagnostics_counters,
        "Reset process-local fallback/degraded-route diagnostic counters.");

  // AUX channel limits + canonical name table (ADC-291), exposed from the SINGLE C++ source
  // (pops/core/state.hpp + aux_names.hpp). The DSL/capabilities() read these so the Python mirrors
  // (AUX_NAMED_MAX / AUX_NAMED_BASE / AUX_CANONICAL in dsl.py) cannot SILENTLY drift from C++:
  // test_capabilities.py asserts they match. kAuxMaxExtra is the only remaining compile-time aux
  // limit and is now declarative + introspectable here.
  m.attr("__aux_base_comps__") = static_cast<int>(pops::kAuxBaseComps);
  m.attr("__aux_named_base__") = static_cast<int>(pops::kAuxNamedBase);
  m.attr("__aux_max_extra__") = static_cast<int>(pops::kAuxMaxExtra);
  m.attr("__aux_max_comps__") = static_cast<int>(pops::kAuxMaxComps);
  // Runtime-param capacity (ADC-610): the SINGLE C++ source of kMaxRuntimeParams
  // (pops/runtime/config/runtime_params.hpp). The codegen guard reads this so the Python literal
  // fallback (physics/aux.py) cannot SILENTLY drift from the fixed-size device array bound.
  m.attr("__max_runtime_params__") = static_cast<int>(pops::kMaxRuntimeParams);
  {
    py::dict canon;
    for (const auto& [name, comp] : pops::kAuxCanonicalNames)
      canon[py::str(std::string(name))] = static_cast<int>(comp);
    m.attr("__aux_canonical__") = canon;
  }

  // REAL state of the Kokkos init (lazy: first Fab allocation, through ANY path --
  // System, AmrSystem, DSL .so...). Internal environment diagnostics rely on this rather than on a
  // Python flag that only saw System/AmrSystem, so a "too late" report remains reliable.
  // Serial build: always False (nothing to initialize, the thread setting is moot).
  m.def(
      "kokkos_is_initialized",
      []() {
#ifdef POPS_HAS_KOKKOS
        return Kokkos::is_initialized();
#else
        return false;
#endif
      },
      "True if the module's Kokkos runtime is already initialized.");

  py::class_<SystemConfig>(m, "SystemConfig")
      .def(py::init<>())
      .def_readwrite("n", &SystemConfig::n)
      .def_readwrite("L", &SystemConfig::L)
      .def_readwrite("periodic", &SystemConfig::periodic)
      // Opt-in geometry ("polar grid" work, Phase 1). "cartesian" (default) = bit-identical;
      // "polar" = global ring carried by pops.PolarMesh. Polar fields ignored if geometry=="cartesian".
      .def_readwrite("geometry", &SystemConfig::geometry)
      .def_readwrite("nr", &SystemConfig::nr)
      .def_readwrite("ntheta", &SystemConfig::ntheta)
      .def_readwrite("r_min", &SystemConfig::r_min)
      .def_readwrite("r_max", &SystemConfig::r_max)
      .def_readwrite("theta_boxes", &SystemConfig::theta_boxes);

  // ModelSpec: composition of generic bricks (transport/source/elliptic + parameters).
  // No named scenario; the pops.Model(...) sugar on the Python side fills these fields.
  auto model_spec = py::class_<ModelSpec>(m, "ModelSpec");
  model_spec.def(py::init<>())
      .def("freeze", &ModelSpec::freeze,
           "Seal ModelSpec authoring; subsequent Python property writes fail.")
      .def_property_readonly("frozen", &ModelSpec::frozen, "Whether ModelSpec authoring is sealed.")
      .def(
          "_semantic_data",
          [](const ModelSpec& spec) {
            py::dict data;
            data["kind"] = "native-model-spec";
            data["transport"] = spec.transport.get();
            data["source"] = spec.source.get();
            data["elliptic"] = spec.elliptic.get();
            data["B0"] = spec.B0.get();
            data["gamma"] = spec.gamma.get();
            data["cs2"] = spec.cs2.get();
            data["vacuum_floor"] = spec.vacuum_floor.get();
            data["qom"] = spec.qom.get();
            data["q"] = spec.q.get();
            data["alpha"] = spec.alpha.get();
            data["n0"] = spec.n0.get();
            data["sign"] = spec.sign.get();
            data["four_pi_G"] = spec.four_pi_G.get();
            data["rho0"] = spec.rho0.get();
            return data;
          },
          "Return the closed scientific projection consumed by PoPS semantic identity.")
      .def(
          "_pops_freeze_snapshot",
          [](const ModelSpec& spec, const py::handle& capability) {
            require_freeze_transaction_capability(capability);
            return pops::detail::ModelSpecFreezeTransactionAccess::snapshot(spec);
          },
          py::arg("capability"))
      .def(
          "_pops_freeze_restore",
          [](ModelSpec& spec, const py::handle& capability, bool state) {
            require_freeze_transaction_capability(capability);
            pops::detail::ModelSpecFreezeTransactionAccess::restore(spec, state);
          },
          py::arg("capability"), py::arg("state"));
  bind_model_spec_property(model_spec, "transport", &ModelSpec::transport);
  bind_model_spec_property(model_spec, "source", &ModelSpec::source);
  bind_model_spec_property(model_spec, "elliptic", &ModelSpec::elliptic);
  bind_model_spec_property(model_spec, "B0", &ModelSpec::B0);
  bind_model_spec_property(model_spec, "gamma", &ModelSpec::gamma);
  bind_model_spec_property(model_spec, "cs2", &ModelSpec::cs2);
  bind_model_spec_property(model_spec, "vacuum_floor", &ModelSpec::vacuum_floor);
  bind_model_spec_property(model_spec, "qom", &ModelSpec::qom);
  bind_model_spec_property(model_spec, "q", &ModelSpec::q);
  bind_model_spec_property(model_spec, "alpha", &ModelSpec::alpha);
  bind_model_spec_property(model_spec, "n0", &ModelSpec::n0);
  bind_model_spec_property(model_spec, "sign", &ModelSpec::sign);
  bind_model_spec_property(model_spec, "four_pi_G", &ModelSpec::four_pi_G);
  bind_model_spec_property(model_spec, "rho0", &ModelSpec::rho0);
}
