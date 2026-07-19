#include "../bindings_detail.hpp"

#if !defined(POPS_RUNTIME_SHARED_EXCEPTION_ABI) || !defined(POPS_EXPORT_BUILDING_MODULE)
#error "the _pops host must build the shared runtime exception ABI as its exporting producer"
#endif

#include <pops/core/state/aux_names.hpp>  // ADC-291: canonical aux name<->component table + bounds
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/parallel/world_communicator.hpp>
#include <pops/runtime/config/runtime_params.hpp>  // ADC-610: kMaxRuntimeParams (mirrored to Python)
#include <pops/runtime/config/platform_manifest.hpp>  // ADC-683: explicit launch contracts
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/module_capabilities.hpp>  // ADC-479 (#36/#37): authoritative static capability facts
#include <pops/runtime/runtime_environment.hpp>  // ADC-609: runtime environment/precision/communicator report

#include <utility>
#include <vector>

#ifndef POPS_MPI_INCLUDE
#define POPS_MPI_INCLUDE ""
#endif
#ifndef POPS_MPI_COMPILER
#define POPS_MPI_COMPILER ""
#endif
#ifndef POPS_MPI_STANDARD
#define POPS_MPI_STANDARD ""
#endif
#ifndef POPS_MPI_COMPILE_OPTIONS
#define POPS_MPI_COMPILE_OPTIONS ""
#endif
#ifndef POPS_MPI_COMPILE_DEFINITIONS
#define POPS_MPI_COMPILE_DEFINITIONS ""
#endif
#ifndef POPS_MPI_LINK_OPTIONS
#define POPS_MPI_LINK_OPTIONS ""
#endif
#ifndef POPS_MPI_LINK_LIBRARIES
#define POPS_MPI_LINK_LIBRARIES ""
#endif
#ifndef POPS_MPI_HEADER_PATHS
#define POPS_MPI_HEADER_PATHS ""
#endif
#ifndef POPS_MPI_HEADER_HASHES
#define POPS_MPI_HEADER_HASHES ""
#endif
#ifndef POPS_MPI_LIBRARY_PATHS
#define POPS_MPI_LIBRARY_PATHS ""
#endif
#ifndef POPS_MPI_LIBRARY_HASHES
#define POPS_MPI_LIBRARY_HASHES ""
#endif
#ifndef POPS_KOKKOS_ABI
#define POPS_KOKKOS_ABI ""
#endif
#ifndef POPS_KOKKOS_INCLUDE
#define POPS_KOKKOS_INCLUDE ""
#endif
#ifndef POPS_KOKKOS_HEADER_PATHS
#define POPS_KOKKOS_HEADER_PATHS ""
#endif
#ifndef POPS_KOKKOS_HEADER_HASHES
#define POPS_KOKKOS_HEADER_HASHES ""
#endif

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

py::tuple pipe_tuple(const std::string& serialized) {
  if (serialized.empty())
    return py::tuple(0);
  std::vector<std::string> values;
  std::size_t begin = 0;
  while (begin <= serialized.size()) {
    const auto end = serialized.find('|', begin);
    values.emplace_back(serialized.substr(begin, end - begin));
    if (end == std::string::npos)
      break;
    begin = end + 1;
  }
  py::tuple result(values.size());
  for (std::size_t index = 0; index < values.size(); ++index)
    result[index] = values[index];
  return result;
}

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
  d["mpi_initialized_by_pops"] = r.mpi_initialized_by_pops;
  d["mpi_atexit_finalize_registered"] = r.mpi_atexit_finalize_registered;
  d["mpi_thread_level"] = r.mpi_thread_level;
  d["mpi_ownership"] = r.mpi_ownership;
  d["allocator_mode"] = r.allocator_mode;
  d["comm_allocator_mode"] = r.comm_allocator_mode;
  d["allocator_lifetime"] = r.allocator_lifetime;
  return d;
}

py::dict runtime_backend_manifest_to_dict(const std::string& backend, const std::string& target,
                                          const std::string& communicator) {
  const auto runtime = pops::runtime_environment_report();
  std::string evidence;
  if (communicator == "serial") {
    if (runtime.mpi_active) {
      throw std::runtime_error(
          "serial RuntimeBackendManifest requested while native MPI_COMM_WORLD is active");
    }
    evidence = "pops.native.2d-float64-host.v1";
  } else if (communicator == "MPI_COMM_WORLD") {
    if (!runtime.mpi_compiled || !runtime.mpi_active || runtime.communicator != "MPI_COMM_WORLD") {
      throw std::runtime_error(
          "MPI_COMM_WORLD RuntimeBackendManifest requires an MPI-enabled module in an active "
          "MPI world launch");
    }
    evidence = "pops.native.2d-float64-host-mpi-world.v1";
  } else {
    throw std::invalid_argument("runtime_backend_manifest supports only serial or MPI_COMM_WORLD");
  }
  const auto manifest =
      pops::platform::proven_host_backend(backend, target, pops::abi_key(), communicator, evidence);
  py::dict precision;
  precision["storage"] =
      pops::platform::require_text(manifest.precision.storage, "precision.storage");
  precision["compute"] =
      pops::platform::require_text(manifest.precision.compute, "precision.compute");
  precision["accumulation"] =
      pops::platform::require_text(manifest.precision.accumulation, "precision.accumulation");
  precision["reduction"] =
      pops::platform::require_text(manifest.precision.reduction, "precision.reduction");
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
  result["memory_spaces"] =
      pops::platform::require_text_set(manifest.memory_spaces, "memory_spaces");
  result["communicator"] = pops::platform::require_text(manifest.communicator, "communicator");
  result["capabilities"] = std::move(capabilities);
  result["evidence"] = evidence;
  result["identity"] = pops::platform::identity_token("runtime-backend-manifest", manifest);
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
#ifdef POPS_HAS_MPI
  // An MPI-enabled module is an explicitly distributed runtime.  Initialize/attach from C++ while
  // the module is imported so compile-time platform discovery sees the real world topology before
  // ExecutionContext is materialized.  WorldCommunicator owns finalization only when it performed
  // MPI_Init_thread itself; an externally initialized MPI remains externally owned.
  (void)pops::WorldCommunicator::world();
#endif

  m.doc() =
      "PoPS (lib): runtime multi-species composition. System composes a "
      "system block by block; the compute stays compiled C++.";

  // Exact native distributed resources.  Neither class has a Python constructor: all consumers
  // receive the same process-world singleton and the singleton-owned MPI_DOUBLE identity.  The
  // byte-only collective methods release the GIL while C++ executes MPI.
  py::class_<pops::NativeMpiDatatype>(m, "_NativeMpiDatatype")
      .def_property_readonly(
          "identity",
          [](const pops::NativeMpiDatatype& datatype) { return std::string(datatype.identity()); })
      .def_property_readonly("fortran_handle", &pops::NativeMpiDatatype::fortran_handle);

  py::class_<pops::WorldCommunicator>(m, "_NativeWorldCommunicator")
      .def_property_readonly("rank", &pops::WorldCommunicator::rank)
      .def_property_readonly("size", &pops::WorldCommunicator::size)
      .def_property_readonly("active", &pops::WorldCommunicator::active)
      .def_property_readonly(
          "identity",
          [](const pops::WorldCommunicator& world) { return std::string(world.identity()); })
      .def_property_readonly("initialized_by_pops", &pops::WorldCommunicator::initialized_by_pops)
      .def_property_readonly("atexit_finalize_registered",
                             &pops::WorldCommunicator::atexit_finalize_registered)
      .def_property_readonly("thread_level", &pops::WorldCommunicator::thread_level)
      .def_property_readonly("fortran_handle", &pops::WorldCommunicator::fortran_handle)
      .def_property_readonly("datatype_float64", &pops::WorldCommunicator::datatype_float64,
                             py::return_value_policy::reference_internal)
      .def(
          "is_float64_datatype",
          [](const pops::WorldCommunicator& world, const py::handle& candidate) {
            try {
              return world.owns_float64_datatype(candidate.cast<const pops::NativeMpiDatatype&>());
            } catch (const py::cast_error&) {
              return false;
            }
          },
          py::arg("candidate"))
      .def("barrier",
           [](const pops::WorldCommunicator& world) {
             py::gil_scoped_release release;
             world.barrier();
           })
      .def(
          "broadcast_bytes",
          [](const pops::WorldCommunicator& world, const py::bytes& payload, int root) {
            std::string native = payload.cast<std::string>();
            {
              py::gil_scoped_release release;
              native = world.broadcast_bytes(std::move(native), root);
            }
            return py::bytes(native);
          },
          py::arg("payload"), py::arg("root") = 0)
      .def(
          "allgather_bytes",
          [](const pops::WorldCommunicator& world, const py::bytes& payload) {
            const std::string native = payload.cast<std::string>();
            std::vector<std::string> gathered;
            {
              py::gil_scoped_release release;
              gathered = world.allgather_bytes(native);
            }
            py::tuple result(gathered.size());
            for (std::size_t index = 0; index < gathered.size(); ++index)
              result[index] = py::bytes(gathered[index]);
            return result;
          },
          py::arg("payload"))
      .def(
          "gather_bytes",
          [](const pops::WorldCommunicator& world, const py::bytes& payload,
             int root) -> py::object {
            const std::string native = payload.cast<std::string>();
            std::optional<std::vector<std::string>> gathered;
            {
              py::gil_scoped_release release;
              gathered = world.gather_bytes(native, root);
            }
            if (!gathered)
              return py::none();
            py::tuple result(gathered->size());
            for (std::size_t index = 0; index < gathered->size(); ++index)
              result[index] = py::bytes((*gathered)[index]);
            return result;
          },
          py::arg("payload"), py::arg("root") = 0);

  m.def(
      "mpi_world", []() -> pops::WorldCommunicator& { return pops::WorldCommunicator::world(); },
      py::return_value_policy::reference,
      "Return the exact native process-world authority (serial singleton in a non-MPI build).");

  // Native iterative solves have one authoritative result contract.  Register it before System so
  // every method returning SolveReport (notably System::solve_fields) has a concrete Python value
  // type.  Status/action are exposed as their stable semantic names instead of leaking C++ enum
  // ordinals into the private bootstrap ABI.
  py::class_<pops::SolveReport>(m, "_SolveReport")
      .def_readonly("iters", &pops::SolveReport::iters)
      .def_readonly("rel_residual", &pops::SolveReport::rel_residual)
      .def_readonly("reference_residual_norm", &pops::SolveReport::reference_residual_norm)
      .def_readonly("residual_norm", &pops::SolveReport::residual_norm)
      .def_property_readonly("status",
                             [](const pops::SolveReport& report) { return report.status_name(); })
      .def_property_readonly("action",
                             [](const pops::SolveReport& report) { return report.action_name(); })
      .def_readonly("reason", &pops::SolveReport::reason)
      .def("valid", &pops::SolveReport::valid)
      .def("solved", &pops::SolveReport::solved)
      .def("solved_value_available", &pops::SolveReport::solved_value_available)
      .def("failed", &pops::SolveReport::failed);

  // Module ABI key (compiler + C++ standard + signature of the pops headers). The DSL reads it
  // (diagnostic); add_native_block compares it to the key baked into a native loader.
  m.def("abi_key", &pops::abi_key,
        "Module ABI key (compiler, C++ standard, signature of the pops headers).");

  // MPI rank / rank count of the communicator (0 / 1 in serial or when MPI is not initialized, cf.
  // pops/parallel/comm.hpp). The private runtime uses these values to authenticate the exact
  // ExecutionContext topology; publication and checkpoint ownership live in RuntimeInstance.
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
  py::dict kokkos_contract;
  kokkos_contract["schema_version"] = 1;
  kokkos_contract["abi_sha256"] = POPS_KOKKOS_ABI;
  kokkos_contract["include_dirs"] = pipe_tuple(POPS_KOKKOS_INCLUDE);
  kokkos_contract["header_paths"] = pipe_tuple(POPS_KOKKOS_HEADER_PATHS);
  kokkos_contract["header_sha256"] = pipe_tuple(POPS_KOKKOS_HEADER_HASHES);
  m.attr("__kokkos_contract__") = std::move(kokkos_contract);
#else
  m.attr("__has_kokkos__") = false;
  m.attr("__kokkos_contract__") = py::none();
#endif

  // Central, closed compile-definition manifest replayed by every native plugin compiler. The host
  // defines both SHARED_EXCEPTION_ABI and BUILDING_MODULE; generated loaders replay only the former,
  // so they consume the host's one exported StepAttemptRejected key function/typeinfo.
  py::dict native_loader_contract;
  native_loader_contract["schema_version"] = 1;
  native_loader_contract["compile_definitions"] =
      py::make_tuple("POPS_RUNTIME_SHARED_EXCEPTION_ABI");
  m.attr("__native_loader_contract__") = std::move(native_loader_contract);

  // Exact, replayable MPI::MPI_CXX build manifest.  Codegen re-hashes every path immediately before
  // compilation, so an in-place MPI upgrade cannot reuse the cached ABI digest with different bytes.
#if defined(POPS_HAS_MPI)
  m.attr("__has_mpi__") = true;
  py::dict mpi_contract;
  mpi_contract["schema_version"] = 1;
  mpi_contract["abi_sha256"] = POPS_MPI_ABI;
  mpi_contract["compiler"] = POPS_MPI_COMPILER;
  mpi_contract["standard"] = POPS_MPI_STANDARD;
  mpi_contract["include_dirs"] = pipe_tuple(POPS_MPI_INCLUDE);
  mpi_contract["compile_options"] = pipe_tuple(POPS_MPI_COMPILE_OPTIONS);
  mpi_contract["compile_definitions"] = pipe_tuple(POPS_MPI_COMPILE_DEFINITIONS);
  mpi_contract["link_options"] = pipe_tuple(POPS_MPI_LINK_OPTIONS);
  mpi_contract["link_libraries"] = pipe_tuple(POPS_MPI_LINK_LIBRARIES);
  mpi_contract["header_paths"] = pipe_tuple(POPS_MPI_HEADER_PATHS);
  mpi_contract["header_sha256"] = pipe_tuple(POPS_MPI_HEADER_HASHES);
  mpi_contract["library_paths"] = pipe_tuple(POPS_MPI_LIBRARY_PATHS);
  mpi_contract["library_sha256"] = pipe_tuple(POPS_MPI_LIBRARY_HASHES);
  m.attr("__mpi_contract__") = std::move(mpi_contract);
#else
  m.attr("__has_mpi__") = false;
  m.attr("__mpi_contract__") = py::none();
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

  m.def("runtime_backend_manifest", &runtime_backend_manifest_to_dict, py::arg("backend"),
        py::arg("target"), py::arg("communicator"),
        "Explicit 2D/float64/host RuntimeBackendManifest for serial or the active exact "
        "MPI_COMM_WORLD route. Custom communicators are rejected.");

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
      // "polar" = global ring carried by pops.mesh.PolarMesh. Polar fields ignored for cartesian.
      .def_readwrite("geometry", &SystemConfig::geometry)
      .def_readwrite("nr", &SystemConfig::nr)
      .def_readwrite("ntheta", &SystemConfig::ntheta)
      .def_readwrite("r_min", &SystemConfig::r_min)
      .def_readwrite("r_max", &SystemConfig::r_max)
      .def_readwrite("theta_boxes", &SystemConfig::theta_boxes);

  // ModelSpec: composition of generic bricks (transport/source/elliptic + parameters).
  // No named scenario; the private Python ModelSpec composer fills these engine fields.
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
