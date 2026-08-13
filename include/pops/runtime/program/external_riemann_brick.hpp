#pragma once

// Static dispatch of an EXTERNAL C++ Riemann brick (Spec 3 section 21-22, criterion 20, ADC-463).
//
// `external_brick.hpp` owns the HOST IDENTITY catalog: `POPS_REGISTER_BRICK` records a brick's id +
// requirements, `pops_brick_manifest()` exports them, and `pops.lib.load_cpp_library` surfaces a
// requirement-carrying `riemann.User(id)` descriptor. This header owns the NUMERICAL half: how the
// brick's flux is actually DISPATCHED into the finite-volume machinery without a per-cell string
// lookup.
//
// The flux of an external brick is a `NumericalFlux` policy (numerics/fv/numerical_flux.hpp) living
// in a SEPARATE `.so`. The `.so` performs the exact-rank static instantiation itself: the
// `POPS_DEFINE_EXTERNAL_RIEMANN_BRICK` macro emits installers specialized for `kNativeDimension`
// and the user's flux type. The host resolves those entry points once at install time; the per-cell
// kernel then runs the statically-instantiated functor with no string comparison on the hot path.
// The limiter route is selected once while preparing the installed operator.
//
// ABI v4 retains a flat residual seam for diagnostics and adds native System/AMR installers.
// Those installers build directly on the runtime-owned MultiFab/hierarchy, so production execution
// is zero-copy and keeps the ordinary Kokkos, MPI-halo and AMR-reflux paths.  The exact ABI identity,
// native rank, exported symbol set and library digest are authenticated before any installer is
// called. Shape, spacing and periodicity are exact-ranked arrays; no 2D mesh object crosses the ABI.

#include <pops/runtime/program/external_brick.hpp>

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/core/identity/sha256.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/boundary/prepared_hyperbolic_boundary.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/builders/compiled/generated_amr_system_block.hpp>
#include <pops/runtime/builders/scheme_dispatch.hpp>
#include <pops/runtime/config/dispatch_tags.hpp>
#include <pops/numerics/fv/reconstruction.hpp>

#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/dynamic/dynlib.hpp>  // portable dlopen<->LoadLibraryW (ADC-99)

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace pops::runtime::program {

#define POPS_EXTERNAL_RIEMANN_STRINGIFY_IMPL_(value) #value
#define POPS_EXTERNAL_RIEMANN_STRINGIFY_(value) POPS_EXTERNAL_RIEMANN_STRINGIFY_IMPL_(value)

inline constexpr int kExternalRiemannBrickAbiVersion = 4;
inline constexpr const char* kExternalRiemannBrickAbiKey =
    "pops.external-riemann/"
    "v4;scalar=f64;index=i32;periodicity=nd;providers=qualified;"
    "dim=" POPS_EXTERNAL_RIEMANN_STRINGIFY_(POPS_NATIVE_DIM);

inline constexpr const char* kExternalRiemannBrickAbiVersionSymbol =
    "pops_external_riemann_abi_version";
inline constexpr const char* kExternalRiemannBrickAbiKeySymbol = "pops_external_riemann_abi_key";
inline constexpr const char* kExternalRiemannBrickDimensionSymbol =
    "pops_external_riemann_dimension";
inline constexpr const char* kExternalRiemannBrickResidualSymbol = "pops_brick_residual_v4";
inline constexpr const char* kExternalRiemannBrickInstallSystemSymbol =
    "pops_brick_install_system_v4";
inline constexpr const char* kExternalRiemannBrickInstallAmrSymbol = "pops_brick_install_amr_v4";
inline constexpr const char* kExternalRiemannBrickModelIdentitySymbol = "pops_brick_model_identity";
inline constexpr const char* kExternalRiemannBrickProviderCountSymbol = "pops_brick_nproviders";
inline constexpr const char* kExternalRiemannBrickKokkosBackendSymbol = "pops_brick_kokkos_backend";
inline constexpr const char* kExternalRiemannBrickKokkosVersionSymbol = "pops_brick_kokkos_version";
inline constexpr const char* kExternalRiemannBrickSystemRoutesSymbol =
    "pops_register_provider_routes";
inline constexpr const char* kExternalRiemannBrickAmrRoutesSymbol =
    "pops_register_provider_routes_amr";

namespace detail {

inline const char* external_kokkos_backend_identity() noexcept {
#ifdef POPS_HAS_KOKKOS
  return Kokkos::DefaultExecutionSpace::name();
#else
  return "none";
#endif
}

inline constexpr int external_kokkos_version_identity() noexcept {
#ifdef POPS_HAS_KOKKOS
  return KOKKOS_VERSION;
#else
  return 0;
#endif
}

template <int Dim>
std::size_t flat_cell_count(const Extent<Dim>& shape) {
  std::size_t count = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    if (shape[axis] <= 0)
      throw std::invalid_argument("external riemann brick: every shape extent must be positive");
    const auto extent = static_cast<std::size_t>(shape[axis]);
    if (count > std::numeric_limits<std::size_t>::max() / extent)
      throw std::length_error("external riemann brick: flat cell count overflows size_t");
    count *= extent;
  }
  return count;
}

inline std::size_t flat_value_count(std::size_t cells, int components) {
  if (components < 0 || (components > 0 && cells > std::numeric_limits<std::size_t>::max() /
                                                       static_cast<std::size_t>(components)))
    throw std::length_error("external riemann brick: flat value count overflows size_t");
  return cells * static_cast<std::size_t>(components);
}

inline void require_finite_values(const double* values, std::size_t count, const char* label) {
  for (std::size_t index = 0; index < count; ++index)
    if (!std::isfinite(values[index]))
      throw std::invalid_argument(std::string("external riemann brick: non-finite ") + label);
}

template <int Dim>
std::size_t field_offset(const Box<Dim>& storage, const Index<Dim>& index, int component) {
  std::size_t linear = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    linear += static_cast<std::size_t>(index[axis] - storage.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(storage.length(axis));
  }
  return static_cast<std::size_t>(component) * static_cast<std::size_t>(storage.numPts()) + linear;
}

template <int Dim, class Function>
void for_each_flat_index(const Box<Dim>& box, Function&& function) {
  for (std::int64_t linear = 0; linear < box.numPts(); ++linear) {
    std::int64_t remaining = linear;
    Index<Dim> index{};
    for (int axis = 0; axis < Dim; ++axis) {
      index[axis] = box.lo[axis] + static_cast<int>(remaining % box.length(axis));
      remaining /= box.length(axis);
    }
    function(index, static_cast<std::size_t>(linear));
  }
}

template <int Dim, class MemorySpace>
void import_component_major(Fab<Dim, MemorySpace>& destination, const double* source, int ncomp) {
  if (source == nullptr)
    return;
  auto host = destination.create_host_mirror();
  destination.copy_to_host(host);
  const std::size_t cells = static_cast<std::size_t>(destination.box().numPts());
  for_each_flat_index(destination.box(), [&](const Index<Dim>& index, std::size_t linear) {
    for (int component = 0; component < ncomp; ++component)
      host(field_offset(destination.grown_box(), index, component)) =
          static_cast<Real>(source[static_cast<std::size_t>(component) * cells + linear]);
  });
  destination.copy_from_host(host);
}

template <int Dim, class MemorySpace>
void export_component_major(const Fab<Dim, MemorySpace>& source, double* destination, int ncomp) {
  auto host = source.create_host_mirror();
  source.copy_to_host(host);
  const std::size_t cells = static_cast<std::size_t>(source.box().numPts());
  for_each_flat_index(source.box(), [&](const Index<Dim>& index, std::size_t linear) {
    for (int component = 0; component < ncomp; ++component)
      destination[static_cast<std::size_t>(component) * cells + linear] =
          static_cast<double>(host(field_offset(source.grown_box(), index, component)));
  });
}

template <int Dim>
HaloScheduleBudget external_halo_budget(const Box<Dim>& domain, const Extent<Dim>& ghosts,
                                        int ncomp) {
  std::size_t periodic_images = 1;
  for (int axis = 0; axis < Dim; ++axis)
    periodic_images *= 3;
  const std::size_t jobs = periodic_images * static_cast<std::size_t>(2 * Dim + 1);
  Box<Dim> grown = domain;
  for (int axis = 0; axis < Dim; ++axis)
    grown = grown.grow(axis, ghosts[axis]);
  const std::size_t elements =
      static_cast<std::size_t>(grown.numPts()) * static_cast<std::size_t>(ncomp) * jobs;
  return {{1, 0}, periodic_images, jobs, periodic_images, 1, elements, elements, elements};
}

template <int Dim>
PreparedHyperbolicBoundary<Dim> external_flat_boundary(const std::array<bool, Dim>& periodic,
                                                       int ncomp,
                                                       std::string_view identity_prefix) {
  std::vector<std::string> laws;
  std::vector<std::string> identities;
  laws.reserve(static_cast<std::size_t>(2 * Dim));
  identities.reserve(static_cast<std::size_t>(2 * Dim));
  for (int axis = 0; axis < Dim; ++axis) {
    for (int side = 0; side < 2; ++side) {
      laws.push_back(periodic[static_cast<std::size_t>(axis)] ? "periodic" : "foextrap");
      identities.push_back(std::string(identity_prefix) + "/axis=" + std::to_string(axis) +
                           "/side=" + std::to_string(side));
    }
  }
  return prepare_hyperbolic_boundary<Dim>(
      laws, std::vector<double>(static_cast<std::size_t>(2 * Dim * ncomp), 0.0), identities,
      std::vector<std::string>(static_cast<std::size_t>(ncomp), "Scalar"));
}

template <class Reconstruction>
Reconstruction external_reconstruction(Real weno_epsilon) {
  if constexpr (std::is_same_v<Reconstruction, Weno5>)
    return configured_reconstruction<Weno5>(weno_epsilon);
  else
    return Reconstruction{};
}

template <class Model>
Model external_model(Real gamma) {
  if constexpr (requires {
                  { Model::prepare(gamma) } -> std::same_as<Model>;
                })
    return Model::prepare(gamma);
  else
    return Model{};
}

template <nd::ReconstructionVariables Variables, int Dim, class Model, class Flux,
          class Reconstruction>
void external_residual_prepared(const double* state_values, double* residual_values,
                                const double* provider_values, const Extent<Dim>& shape,
                                const RealVector<Dim>& spacing,
                                const std::array<bool, Dim>& periodic, Real positivity_floor,
                                Real weno_epsilon) {
  static_assert(Model::dimension == Dim);
  const Box<Dim> domain = Box<Dim>::from_extents(shape);
  RealVector<Dim> lower{};
  RealVector<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = spacing[axis] * static_cast<Real>(shape[axis]);
  const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(domain, lower, upper);
  const mesh::BoxArray<Dim> layout(std::vector<Box<Dim>>{domain});
  Extent<Dim> rank_extent{};
  Extent<Dim> ghosts{};
  for (int axis = 0; axis < Dim; ++axis) {
    rank_extent[axis] = 1;
    ghosts[axis] = Reconstruction::n_ghost;
  }
  const mesh::RankSpace<Dim> ranks{Index<Dim>{}, rank_extent};
  const auto distribution = mesh::Distribution<Dim>::replicated(layout, ranks);
  MultiFab<Dim> state(layout, distribution, Index<Dim>{}, Model::n_vars, ghosts);
  MultiFab<Dim> residual(layout, distribution, Index<Dim>{}, Model::n_vars, ghosts);
  state.set_val(Real(0));
  residual.set_val(Real(0));
  import_component_major(state.fab(0), state_values, Model::n_vars);
  constexpr int provider_count = flux_provider_count<Model>;
  std::optional<MultiFab<Dim>> provider_storage;
  if constexpr (provider_count > 0) {
    provider_storage.emplace(layout, distribution, Index<Dim>{}, provider_count, ghosts);
    provider_storage->set_val(Real(0));
    import_component_major(provider_storage->fab(0), provider_values, provider_count);
  }

  const BoundaryTopology<Dim> topology(periodic);
  fill_boundary(state, domain, topology, external_halo_budget(domain, ghosts, state.ncomp()));
  if constexpr (provider_count > 0)
    fill_boundary(*provider_storage, domain, topology,
                  external_halo_budget(domain, ghosts, provider_count));
  external_flat_boundary<Dim>(periodic, state.ncomp(), "external-riemann/state")
      .fill_physical(state, geometry);
  if constexpr (provider_count > 0)
    external_flat_boundary<Dim>(periodic, provider_count, "external-riemann/providers")
        .fill_physical(*provider_storage, geometry);

  Model model{};
  const auto spatial = nd::prepare_cartesian_operator<Dim, Model, Reconstruction, Flux, Variables>(
      geometry, model, external_reconstruction<Reconstruction>(weno_epsilon), Flux{},
      positivity_floor);
  if constexpr (flux_provider_count<Model> == 0)
    spatial.assemble_residual(state, residual);
  else
    spatial.assemble_residual(state, *provider_storage, residual);
  std::vector<double> candidate(flat_value_count(flat_cell_count(shape), Model::n_vars));
  export_component_major(residual.fab(0), candidate.data(), Model::n_vars);
  require_finite_values(candidate.data(), candidate.size(), "residual candidate");
  std::copy(candidate.begin(), candidate.end(), residual_values);
}

template <int Dim, class Model, class Flux>
void external_residual(const double* state_values, double* residual_values,
                       const double* provider_values, const int* shape_values,
                       const double* spacing_values, const int* periodic_values,
                       const std::string& limiter, bool reconstruct_primitive,
                       double positivity_floor,
                       double weno_epsilon = static_cast<double>(kWenoEpsilon)) {
  static_assert(Model::dimension == Dim);
  if (state_values == nullptr || residual_values == nullptr || shape_values == nullptr ||
      spacing_values == nullptr || periodic_values == nullptr)
    throw std::invalid_argument("external riemann brick: ranked residual arguments are null");
  if constexpr (flux_provider_count<Model> > 0)
    if (provider_values == nullptr)
      throw std::invalid_argument("external riemann brick: model requires compact provider values");
  if (!std::isfinite(positivity_floor) || positivity_floor < 0.0 || !std::isfinite(weno_epsilon) ||
      weno_epsilon <= 0.0)
    throw std::invalid_argument("external riemann brick: numerical parameters are invalid");
  Extent<Dim> shape{};
  RealVector<Dim> spacing{};
  std::array<bool, Dim> periodic{};
  for (int axis = 0; axis < Dim; ++axis) {
    shape[axis] = shape_values[axis];
    if (shape[axis] <= 0 || !std::isfinite(spacing_values[axis]) || spacing_values[axis] <= 0.0 ||
        (periodic_values[axis] != 0 && periodic_values[axis] != 1))
      throw std::invalid_argument("external riemann brick: invalid ranked mesh metadata");
    spacing[axis] = static_cast<Real>(spacing_values[axis]);
    periodic[static_cast<std::size_t>(axis)] = periodic_values[axis] != 0;
  }
  const std::size_t cells = flat_cell_count(shape);
  require_finite_values(state_values, flat_value_count(cells, Model::n_vars), "state input");
  if constexpr (flux_provider_count<Model> > 0)
    require_finite_values(provider_values, flat_value_count(cells, flux_provider_count<Model>),
                          "provider input");
  validate_limiter(limiter, "external riemann brick");
  dispatch_limiter(parse_limiter_route(limiter, "external riemann brick"), "external riemann brick",
                   [&](auto tag) {
                     using Reconstruction = typename decltype(tag)::type;
                     if (reconstruct_primitive)
                       external_residual_prepared<nd::ReconstructionVariables::Primitive, Dim,
                                                  Model, Flux, Reconstruction>(
                           state_values, residual_values, provider_values, shape, spacing, periodic,
                           static_cast<Real>(positivity_floor), static_cast<Real>(weno_epsilon));
                     else
                       external_residual_prepared<nd::ReconstructionVariables::Conservative, Dim,
                                                  Model, Flux, Reconstruction>(
                           state_values, residual_values, provider_values, shape, spacing, periodic,
                           static_cast<Real>(positivity_floor), static_cast<Real>(weno_epsilon));
                   });
}

inline void validate_external_install(const std::string& name, const std::string& limiter,
                                      const std::string& reconstruction, const std::string& time,
                                      const std::string& provider_consumer_qid, double gamma,
                                      int substeps, int stride, double positivity_floor,
                                      double weno_epsilon) {
  if (name.empty() || provider_consumer_qid.empty() || substeps < 1 || stride < 1 ||
      !std::isfinite(gamma) || !(gamma > 0.0) || !std::isfinite(positivity_floor) ||
      positivity_floor < 0.0 || !std::isfinite(weno_epsilon) || weno_epsilon <= 0.0)
    throw std::invalid_argument("external riemann brick: invalid exact-ranked install request");
  validate_limiter(limiter, "external riemann brick");
  (void)parse_recon_route(reconstruction, "external riemann brick");
  (void)parse_time_route(time, "external riemann brick");
  if (limiter != "weno5" && weno_epsilon != static_cast<double>(kWenoEpsilon))
    throw std::invalid_argument(
        "external riemann brick: WENO epsilon is only meaningful for limiter='weno5'");
}

template <int Dim, class Model, class Flux, class Request>
PreparedSystemBlock<Dim> prepare_external_system_block(Request request, Real weno_epsilon) {
  return dispatch_limiter(
      parse_limiter_route(request.routes.limiter, "external riemann brick"),
      "external riemann brick", [&](auto tag) {
        using Reconstruction = typename decltype(tag)::type;
        const auto prepared = external_reconstruction<Reconstruction>(weno_epsilon);
        if (request.routes.reconstruction == "primitive")
          return generated_system_detail::materialize_block<Dim, Model, Reconstruction, Flux,
                                                            nd::ReconstructionVariables::Primitive>(
              std::move(request), prepared, Flux{});
        return generated_system_detail::materialize_block<
            Dim, Model, Reconstruction, Flux, nd::ReconstructionVariables::Conservative>(
            std::move(request), prepared, Flux{});
      });
}

template <int Dim, class Model, class Flux, class Request>
PreparedAmrSystemBlock<Dim> prepare_external_amr_block(Request request, Real weno_epsilon) {
  return dispatch_limiter(
      parse_limiter_route(request.routes.limiter, "external riemann brick"),
      "external riemann brick", [&](auto tag) {
        using Reconstruction = typename decltype(tag)::type;
        const auto prepared = external_reconstruction<Reconstruction>(weno_epsilon);
        if (request.routes.reconstruction == "primitive")
          return generated_amr_detail::materialize_system<Dim, Model, Reconstruction, Flux,
                                                          nd::ReconstructionVariables::Primitive>(
              std::move(request), prepared, Flux{});
        return generated_amr_detail::materialize_system<Dim, Model, Reconstruction, Flux,
                                                        nd::ReconstructionVariables::Conservative>(
            std::move(request), prepared, Flux{});
      });
}

template <int Dim, class Model, class Flux>
void external_install_system(System<Dim>& system, std::string_view flux_identity,
                             const std::string& name, const std::string& provider_consumer_qid,
                             const std::string& limiter, const std::string& reconstruction,
                             const std::string& time, double gamma, int substeps, bool evolve,
                             int stride, double positivity_floor, double weno_epsilon) {
  static_assert(Model::dimension == Dim);
  validate_external_install(name, limiter, reconstruction, time, provider_consumer_qid, gamma,
                            substeps, stride, positivity_floor, weno_epsilon);
  CompiledSystemBlockRoutes routes{limiter, "external:" + std::string(flux_identity),
                                   reconstruction, time, static_cast<Real>(positivity_floor)};
  const runtime::system::AuxiliaryStorageGroups<Dim>* provider_storage = nullptr;
  const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* provider_plan = nullptr;
  if constexpr (provider_count_for<Model, Dim>() > 0) {
    provider_storage = system.prepared_block_provider_storage_groups();
    provider_plan = &system.prepared_auxiliary_consumer_plan(provider_consumer_qid);
  }
  auto prepared = prepare_external_system_block<Dim, Model, Flux>(
      CompiledSystemBlockPreparation<Dim, Model>{
          name, external_model<Model>(static_cast<Real>(gamma)), std::move(routes),
          system.prepared_block_geometry(),
          BoundaryTopology<Dim>::axis_periodic(system.prepared_block_periodicity()),
          provider_storage, provider_plan},
      static_cast<Real>(weno_epsilon));
  prepared.name = name;
  prepared.ncomp = Model::n_vars;
  prepared.conservative_variables = Model::conservative_vars();
  prepared.primitive_variables = Model::primitive_vars();
  prepared.gamma = gamma;
  prepared.substeps = substeps;
  prepared.evolve = evolve;
  prepared.stride = stride;
  system.install_prepared_block(std::move(prepared));
}

template <int Dim, class Model, class Flux>
void external_install_amr(AmrSystem<Dim>& system, std::string_view flux_identity,
                          const std::string& name, const std::string& provider_consumer_qid,
                          const std::string& limiter, const std::string& reconstruction,
                          const std::string& time, double gamma, int substeps, int stride,
                          double positivity_floor, double weno_epsilon) {
  static_assert(Model::dimension == Dim);
  validate_external_install(name, limiter, reconstruction, time, provider_consumer_qid, gamma,
                            substeps, stride, positivity_floor, weno_epsilon);
  CompiledAmrSystemBlockRoutes routes{
      limiter, "external:" + std::string(flux_identity), reconstruction,
      time,    static_cast<Real>(positivity_floor),      static_cast<Real>(weno_epsilon),
      false};
  auto prepared = prepare_external_amr_block<Dim, Model, Flux>(
      CompiledAmrSystemBlockPreparation<Dim, Model>{name, provider_consumer_qid,
                                                    external_model<Model>(static_cast<Real>(gamma)),
                                                    std::move(routes), gamma, substeps, stride},
      static_cast<Real>(weno_epsilon));
  system.install_prepared_amr_block(std::move(prepared));
}

}  // namespace detail

// The host-side handle to a loaded external Riemann brick `.so`: dlopen the library, read its
// manifest, and resolve the typed entry-point function pointers ONCE. After construction the brick
// is dispatched by calling the resolved residual() pointer -- a direct C call into the `.so`'s
// statically-instantiated flux, never a per-cell string lookup. The manifest is also registered in
// the process catalog (BrickRegistry) so the brick's id + requirements are visible to a later host
// query (mirroring what pops.lib.load_cpp_library does on the Python side).
//
// This is the C++ counterpart of pops.lib.load_cpp_library: the Python path surfaces the descriptor
// (requirements/capabilities) for the board/install layer; this path resolves the numerical entry
// point for a host that drives the brick from C++. A brick `.so` not exporting the expected symbols
// is rejected with a clear error (it is not an pops external Riemann brick `.so`).
class ExternalBrickHandle {
 public:
  // Function-pointer type of the brick's residual entry point (POPS_DEFINE_EXTERNAL_RIEMANN_BRICK).
  using ResidualFn = void (*)(const double*, double*, const double*, const int*, const double*,
                              const int*, const char*, int, double, double);
  using InstallSystemFn = void (*)(void*, const char*, const char*, const char*, const char*,
                                   const char*, double, int, int, int, double, double);
  using InstallAmrFn = void (*)(void*, const char*, const char*, const char*, const char*,
                                const char*, double, int, int, double, double);

  // dlopen @p so_path, read + register its manifest, and resolve the entry points for brick @p id.
  // Throws std::runtime_error if the library cannot be opened, does not export pops_brick_manifest /
  // the versioned residual ABI (not a PoPS external Riemann brick), or does not register @p id as a
  // riemann brick (a clear, actionable message names the id).
  ExternalBrickHandle(const std::string& so_path, const std::string& id, int expected_nvars,
                      int expected_provider_count, const std::string& expected_model_identity,
                      const std::string& expected_sha256 = {})
      : id_(id) {
    if (id_.empty() || expected_nvars < 1 || expected_provider_count < 0 ||
        expected_model_identity.empty())
      throw std::invalid_argument(
          "external riemann brick: exact id/model/provider expectations are required");
    if (!expected_sha256.empty() && file_sha256(so_path) != expected_sha256)
      throw std::runtime_error("external riemann brick '" + id_ +
                               "' library digest changed after descriptor resolution");
#if !defined(_WIN32)
    Dl_info host_info;
    if (dladdr(reinterpret_cast<void*>(&pops::abi_key), &host_info) && host_info.dli_fname)
      dlopen(host_info.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
#endif
    handle_ = dynlib::open(so_path);
    if (!dynlib::valid(handle_))
      throw std::runtime_error("external riemann brick: cannot dlopen '" + so_path +
                               "': " + dynlib::last_error());
    std::exception_ptr preflight_error;
    try {
      auto manifest_fn =
          reinterpret_cast<const char* (*)()>(dynlib::sym(handle_, "pops_brick_manifest"));
      if (manifest_fn == nullptr)
        throw std::runtime_error(
            "external riemann brick '" + so_path +
            "' does not export pops_brick_manifest(); it is not a PoPS brick .so");
      const char* raw_manifest = dynlib::invoke_with_host_exception(
          [manifest_fn] { return manifest_fn(); }, "pops_brick_manifest");
      const std::vector<BrickManifestEntry> entries = parse_manifest_json(raw_manifest);
      const auto selected = std::find_if(entries.begin(), entries.end(),
                                         [&](const auto& entry) { return entry.id == id_; });
      if (selected == entries.end())
        throw std::runtime_error("external riemann brick '" + id_ +
                                 "' not found in the manifest of '" + so_path + "'");
      if (selected->category != "riemann")
        throw std::runtime_error("external brick '" + id_ + "' is registered as category '" +
                                 selected->category + "', not 'riemann'");

      require_abi_symbol(*selected, kExternalRiemannBrickAbiVersionSymbol);
      require_abi_symbol(*selected, kExternalRiemannBrickAbiKeySymbol);
      require_abi_symbol(*selected, kExternalRiemannBrickDimensionSymbol);
      require_abi_symbol(*selected, kExternalRiemannBrickResidualSymbol);
      require_abi_symbol(*selected, kExternalRiemannBrickInstallSystemSymbol);
      require_abi_symbol(*selected, kExternalRiemannBrickInstallAmrSymbol);
      require_abi_symbol(*selected, kExternalRiemannBrickModelIdentitySymbol);
      require_abi_symbol(*selected, kExternalRiemannBrickProviderCountSymbol);
      require_abi_symbol(*selected, kExternalRiemannBrickKokkosBackendSymbol);
      require_abi_symbol(*selected, kExternalRiemannBrickKokkosVersionSymbol);
      require_abi_symbol(*selected, kExternalRiemannBrickSystemRoutesSymbol);
      require_abi_symbol(*selected, kExternalRiemannBrickAmrRoutesSymbol);
      auto version_fn =
          reinterpret_cast<int (*)()>(dynlib::sym(handle_, kExternalRiemannBrickAbiVersionSymbol));
      auto abi_key_fn = reinterpret_cast<const char* (*)()>(
          dynlib::sym(handle_, kExternalRiemannBrickAbiKeySymbol));
      auto dimension_fn =
          reinterpret_cast<int (*)()>(dynlib::sym(handle_, kExternalRiemannBrickDimensionSymbol));
      if (version_fn == nullptr || abi_key_fn == nullptr || dimension_fn == nullptr)
        throw std::runtime_error(
            "external riemann brick '" + id_ +
            "' uses the legacy unversioned residual ABI; rebuild it with the current "
            "POPS_DEFINE_EXTERNAL_RIEMANN_BRICK macro");
      const int version = dynlib::invoke_with_host_exception([version_fn] { return version_fn(); },
                                                             kExternalRiemannBrickAbiVersionSymbol);
      const char* abi_key = dynlib::invoke_with_host_exception(
          [abi_key_fn] { return abi_key_fn(); }, kExternalRiemannBrickAbiKeySymbol);
      if (version != kExternalRiemannBrickAbiVersion || abi_key == nullptr ||
          std::string(abi_key) != kExternalRiemannBrickAbiKey)
        throw std::runtime_error("external riemann brick '" + id_ +
                                 "' has incompatible residual ABI version/key; "
                                 "rebuild it with the current PoPS headers");
      const int candidate_dimension = dynlib::invoke_with_host_exception(
          [dimension_fn] { return dimension_fn(); }, kExternalRiemannBrickDimensionSymbol);
      if (candidate_dimension != kNativeDimension)
        throw std::runtime_error("external riemann brick '" + id_ +
                                 "' targets a different compile-time spatial dimension");
      const auto candidate_residual =
          reinterpret_cast<ResidualFn>(dynlib::sym(handle_, kExternalRiemannBrickResidualSymbol));
      if (candidate_residual == nullptr)
        throw std::runtime_error("external riemann brick '" + id_ +
                                 "' declares but does not export " +
                                 std::string(kExternalRiemannBrickResidualSymbol));
      const auto candidate_install_system = reinterpret_cast<InstallSystemFn>(
          dynlib::sym(handle_, kExternalRiemannBrickInstallSystemSymbol));
      const auto candidate_install_amr = reinterpret_cast<InstallAmrFn>(
          dynlib::sym(handle_, kExternalRiemannBrickInstallAmrSymbol));
      auto nvars_fn = reinterpret_cast<int (*)()>(dynlib::sym(handle_, "pops_brick_nvars"));
      auto provider_count_fn = reinterpret_cast<int (*)()>(
          dynlib::sym(handle_, kExternalRiemannBrickProviderCountSymbol));
      auto model_identity_fn = reinterpret_cast<const char* (*)()>(
          dynlib::sym(handle_, kExternalRiemannBrickModelIdentitySymbol));
      auto kokkos_backend_fn = reinterpret_cast<const char* (*)()>(
          dynlib::sym(handle_, kExternalRiemannBrickKokkosBackendSymbol));
      auto kokkos_version_fn = reinterpret_cast<int (*)()>(
          dynlib::sym(handle_, kExternalRiemannBrickKokkosVersionSymbol));
      if (candidate_install_system == nullptr || candidate_install_amr == nullptr ||
          nvars_fn == nullptr || provider_count_fn == nullptr || model_identity_fn == nullptr ||
          kokkos_backend_fn == nullptr || kokkos_version_fn == nullptr)
        throw std::runtime_error("external riemann brick '" + id_ +
                                 "' is missing a declared native installer/count symbol");
      const int candidate_nvars =
          dynlib::invoke_with_host_exception([nvars_fn] { return nvars_fn(); }, "pops_brick_nvars");
      const int candidate_provider_count =
          dynlib::invoke_with_host_exception([provider_count_fn] { return provider_count_fn(); },
                                             kExternalRiemannBrickProviderCountSymbol);
      void* const candidate_register_system_routes =
          dynlib::sym(handle_, kExternalRiemannBrickSystemRoutesSymbol);
      void* const candidate_register_amr_routes =
          dynlib::sym(handle_, kExternalRiemannBrickAmrRoutesSymbol);
      if (candidate_register_system_routes == nullptr || candidate_register_amr_routes == nullptr)
        throw std::runtime_error("external riemann brick '" + id_ +
                                 "' does not export canonical pops_register_provider_routes and "
                                 "pops_register_provider_routes_amr");
      const char* model_identity =
          dynlib::invoke_with_host_exception([model_identity_fn] { return model_identity_fn(); },
                                             kExternalRiemannBrickModelIdentitySymbol);
      if (model_identity == nullptr || *model_identity == '\0')
        throw std::runtime_error("external riemann brick '" + id_ +
                                 "' exports an empty model identity");
      if (candidate_nvars != expected_nvars || candidate_provider_count != expected_provider_count)
        throw std::runtime_error("external riemann brick '" + id_ +
                                 "' model shape disagrees with the compiled model descriptor");
      if (model_identity != expected_model_identity)
        throw std::runtime_error("external riemann brick '" + id_ +
                                 "' targets a different compiled model identity");
      const char* brick_backend =
          dynlib::invoke_with_host_exception([kokkos_backend_fn] { return kokkos_backend_fn(); },
                                             kExternalRiemannBrickKokkosBackendSymbol);
      const char* host_backend = detail::external_kokkos_backend_identity();
      const int host_kokkos_version = detail::external_kokkos_version_identity();
      const int brick_kokkos_version =
          dynlib::invoke_with_host_exception([kokkos_version_fn] { return kokkos_version_fn(); },
                                             kExternalRiemannBrickKokkosVersionSymbol);
      if (brick_backend == nullptr || std::string(brick_backend) != host_backend ||
          brick_kokkos_version != host_kokkos_version)
        throw std::runtime_error("external riemann brick '" + id_ +
                                 "' was built for a different Kokkos backend/version");

      // Prepare all handle state, including the only allocating member, before publishing the
      // descriptor. BrickRegistry itself publishes one complete candidate by noexcept swaps.
      requirements_ = selected->requirements;
      residual_ = candidate_residual;
      install_system_ = candidate_install_system;
      install_amr_ = candidate_install_amr;
      dimension_ = candidate_dimension;
      nvars_ = candidate_nvars;
      provider_count_ = candidate_provider_count;
      register_system_routes_symbol_ = candidate_register_system_routes;
      register_amr_routes_symbol_ = candidate_register_amr_routes;
      BrickRegistry::instance().register_brick(*selected);
    } catch (...) {
      preflight_error = std::current_exception();
    }
    if (preflight_error) {
      // Every DSO callback above is translated before this point. Close only after its original
      // exception object has been destroyed at handler exit, then propagate the host-owned copy.
      dynlib::close(handle_);
      handle_ = nullptr;
      std::rethrow_exception(preflight_error);
    }
  }

  ExternalBrickHandle(const ExternalBrickHandle&) = delete;
  ExternalBrickHandle& operator=(const ExternalBrickHandle&) = delete;
  ~ExternalBrickHandle() {
    if (dynlib::valid(handle_))
      dynlib::close(handle_);
  }

  // The resolved residual entry point: a direct call into the `.so`'s statically-instantiated flux.
  ResidualFn residual() const { return residual_; }

  void install_system(void* system, const std::string& name,
                      const std::string& provider_consumer_qid, const std::string& limiter,
                      const std::string& recon, const std::string& time, double gamma, int substeps,
                      bool evolve, int stride, double positivity_floor, double weno_epsilon) const {
    dynlib::invoke_with_host_exception(
        [&] {
          install_system_(system, name.c_str(), provider_consumer_qid.c_str(), limiter.c_str(),
                          recon.c_str(), time.c_str(), gamma, substeps, evolve ? 1 : 0, stride,
                          positivity_floor, weno_epsilon);
        },
        kExternalRiemannBrickInstallSystemSymbol);
  }

  void install_amr(void* system, const std::string& name, const std::string& provider_consumer_qid,
                   const std::string& limiter, const std::string& recon, const std::string& time,
                   double gamma, int substeps, int stride, double positivity_floor,
                   double weno_epsilon) const {
    dynlib::invoke_with_host_exception(
        [&] {
          install_amr_(system, name.c_str(), provider_consumer_qid.c_str(), limiter.c_str(),
                       recon.c_str(), time.c_str(), gamma, substeps, stride, positivity_floor,
                       weno_epsilon);
        },
        kExternalRiemannBrickInstallAmrSymbol);
  }

  // The CSV of model capabilities the brick requires (from its manifest); "" when none.
  const std::string& requirements() const { return requirements_; }

  const std::string& id() const { return id_; }
  int dimension() const noexcept { return dimension_; }
  int nvars() const noexcept { return nvars_; }
  int provider_count() const noexcept { return provider_count_; }

  template <int Dim>
  void register_system_routes(System<Dim>& system) const {
    static_assert(Dim == kNativeDimension);
    using RegisterSystemRoutesFn = void (*)(System<Dim>*);
    const auto registrar = reinterpret_cast<RegisterSystemRoutesFn>(register_system_routes_symbol_);
    dynlib::invoke_with_host_exception([&] { registrar(&system); },
                                       kExternalRiemannBrickSystemRoutesSymbol);
  }

  template <int Dim>
  void register_amr_routes(AmrSystem<Dim>& system) const {
    static_assert(Dim == kNativeDimension);
    using RegisterAmrRoutesFn = void (*)(AmrSystem<Dim>*);
    const auto registrar = reinterpret_cast<RegisterAmrRoutesFn>(register_amr_routes_symbol_);
    dynlib::invoke_with_host_exception([&] { registrar(&system); },
                                       kExternalRiemannBrickAmrRoutesSymbol);
  }

 private:
  static std::string file_sha256(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
      throw std::runtime_error("external riemann brick: cannot read '" + path + "'");
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                          std::istreambuf_iterator<char>());
    return identity::sha256_hex(bytes);
  }

  static bool csv_has(const std::string& csv, const std::string& token) {
    std::size_t begin = 0;
    while (begin <= csv.size()) {
      const std::size_t end = csv.find(',', begin);
      if (csv.substr(begin, end == std::string::npos ? end : end - begin) == token)
        return true;
      if (end == std::string::npos)
        break;
      begin = end + 1;
    }
    return false;
  }

  static void require_abi_symbol(const BrickManifestEntry& entry, const char* symbol) {
    if (!csv_has(entry.exported_symbols, symbol))
      throw std::runtime_error("external riemann brick '" + entry.id +
                               "' manifest does not declare required ABI symbol '" + symbol +
                               "'; legacy manifests must be rebuilt, not adapted");
  }

  // Minimal strict reader for the flat manifest emitted by BrickRegistry::to_json().  It returns the
  // rows from THIS loaded image, so an id already registered by another DSO cannot authenticate the
  // wrong library.  Registration is deliberately deferred until after the residual ABI check.
  static std::vector<BrickManifestEntry> parse_manifest_json(const char* json) {
    if (json == nullptr)
      throw std::runtime_error("external riemann brick: pops_brick_manifest() returned NULL");
    const std::string s = json;
    if (integer_field(s, "schema_version") != kBrickManifestSchemaVersion)
      throw std::runtime_error(
          "external riemann brick manifest schema is incompatible; regenerate the library");
    const std::string manifest_abi_key = field(s, "abi_key");
    if (manifest_abi_key.empty())
      throw std::runtime_error("external riemann brick manifest has no ABI identity");
    // Compare against this HOST translation unit's frozen literal. Calling pops::abi_key() here
    // would introduce a hidden link dependency on runtime/system.cpp into AMR-only consumers of
    // this header. The literal carries the same compiler/std/header/Kokkos/MPI/stdlib identity and,
    // unlike an inline function, cannot be interposed by the just-loaded DSO.
    if (manifest_abi_key != POPS_ABI_KEY_LITERAL)
      throw std::runtime_error(
          "external riemann brick manifest ABI differs from the loaded PoPS runtime");
    const std::size_t bricks = s.find("\"bricks\":[");
    const std::size_t arr = bricks == std::string::npos ? std::string::npos : s.find('[', bricks);
    if (arr == std::string::npos)
      throw std::runtime_error("external riemann brick manifest has no bricks array");
    std::vector<BrickManifestEntry> entries;
    std::size_t pos = arr + 1;
    while (true) {
      const std::size_t obj = s.find('{', pos);
      const std::size_t array_end = s.find(']', pos);
      if (array_end == std::string::npos)
        throw std::runtime_error(
            "external riemann brick manifest has an unterminated bricks array");
      if (obj == std::string::npos || obj > array_end)
        break;
      const std::size_t end = s.find('}', obj);
      if (end == std::string::npos || end > array_end)
        throw std::runtime_error("external riemann brick manifest has a malformed brick row");
      const std::string rec = s.substr(obj, end - obj + 1);
      BrickManifestEntry e;
      e.id = field(rec, "id");
      e.category = field(rec, "category");
      e.requirements = field(rec, "requirements");
      e.capabilities = field(rec, "capabilities");
      e.native_id = field(rec, "native_id");
      e.supported_layouts = field(rec, "supported_layouts");
      e.supported_platforms = field(rec, "supported_platforms");
      e.params = field(rec, "params");
      e.options = field(rec, "options");
      e.exported_symbols = field(rec, "exported_symbols");
      if (e.id.empty())
        throw std::runtime_error("external riemann brick manifest contains an empty brick id");
      if (std::any_of(entries.begin(), entries.end(),
                      [&](const auto& row) { return row.id == e.id; }))
        throw std::runtime_error("external riemann brick manifest contains duplicate id '" + e.id +
                                 "'");
      entries.push_back(std::move(e));
      pos = end + 1;
    }
    return entries;
  }

  static int integer_field(const std::string& document, const std::string& key) {
    const std::string pattern = "\"" + key + "\":";
    const std::size_t found = document.find(pattern);
    if (found == std::string::npos)
      return -1;
    const char* first = document.c_str() + found + pattern.size();
    char* last = nullptr;
    const long value = std::strtol(first, &last, 10);
    return last == first ? -1 : static_cast<int>(value);
  }

  // Extracts the value of "key":"value" from one manifest record (the fields to_json() emits are flat
  // quoted strings; this is a targeted reader, not a general JSON parser). It skips backslash-escaped
  // characters when scanning for the closing quote (so an escaped `\"` inside the value does not end
  // it early) and json_unescape's the result. "" when the key is absent.
  static std::string field(const std::string& rec, const std::string& key) {
    const std::string pat = "\"" + key + "\":\"";
    const std::size_t k = rec.find(pat);
    if (k == std::string::npos)
      return "";
    const std::size_t start = k + pat.size();
    std::size_t end = start;
    while (end < rec.size() && rec[end] != '"') {
      end += (rec[end] == '\\' && end + 1 < rec.size()) ? 2 : 1;  // skip an escaped pair atomically
    }
    if (end >= rec.size())
      return "";
    return json_unescape(rec.substr(start, end - start));
  }

  dynlib::handle handle_ = nullptr;
  ResidualFn residual_ = nullptr;
  InstallSystemFn install_system_ = nullptr;
  InstallAmrFn install_amr_ = nullptr;
  int nvars_ = -1;
  int provider_count_ = -1;
  int dimension_ = 0;
  void* register_system_routes_symbol_ = nullptr;
  void* register_amr_routes_symbol_ = nullptr;
  std::string id_;
  std::string requirements_;
};

}  // namespace pops::runtime::program

// Defines the static-dispatch ABI of an external Riemann brick `.so`: registers its identity in the
// host catalog AND emits the entry point the host calls. Use ONCE at namespace scope:
//   struct MyRiemann {
//     template <pops::PhysicalFlux F>
//     POPS_HD pops::FluxEvaluation<typename F::State>
//     operator()(const F&, const typename F::Trace&, const typename F::Trace&,
//                const pops::FaceContext&) const;
//   };
//   using Model = pops::nd::IdealGasEuler<pops::kNativeDimension>;
//   POPS_DEFINE_EMPTY_EXTERNAL_RIEMANN_PROVIDER_ROUTES(Model);  // only when provider_count == 0
//   POPS_DEFINE_EXTERNAL_RIEMANN_BRICK("my_riemann", MyRiemann, Model,
//                                     "<compiled-model-hash>", "pressure,wave_speeds");
//   POPS_DEFINE_BRICK_MANIFEST();  // exports the manifest reader (once per .so)
//
// @p id          the brick id a user selects via pops.lib.riemann.User(id);
// @p Flux        the narrow two-trace NumericalFlux policy (numerics/fv/numerical_flux.hpp);
// @p Model       a TOP-LEVEL ALIAS of the exact-ranked ConservationLaw the .so instantiates. Its
//                dimension must equal kNativeDimension and its variable metadata is authenticated;
// @p model_identity the exact CompiledModel.model_hash this DSO targets; same-size models are not
//                interchangeable and are rejected before install;
// @p reqs_csv    the CSV of model capabilities the brick requires (surfaced in the manifest).
//
// The emitted pops_brick_residual_v4 instantiates the exact-ranked Cartesian operator at the .so's
// compile time: the flux and native rank are STATIC template arguments, never per-cell or runtime
// dimension lookups. pops_brick_nvars / pops_brick_nproviders let the host validate its exact
// state and qualified-provider carrier before an installer can publish a block.
// Both canonical System/AMR provider registrar symbols are mandatory. Zero-provider models emit
// exact empty hooks with POPS_DEFINE_EMPTY_EXTERNAL_RIEMANN_PROVIDER_ROUTES; provider-consuming
// artifacts emit their real typed routes and must not use the empty-hook macro.
//
// ABI WARNING: the brick `.so` MUST be compiled against the SAME Kokkos backend and version (and the
// same pops headers) as the host binary that dlopens it -- the residual runs the host's Kokkos
// runtime. Installation must therefore pass through the authenticated component loader, which
// validates the exact component manifest and platform/ABI evidence before publishing the handle.
#define POPS_DEFINE_EMPTY_EXTERNAL_RIEMANN_PROVIDER_ROUTES(Model)                            \
  static_assert(::pops::provider_count_for<Model, ::pops::kNativeDimension>() == 0,          \
                "empty external Riemann provider hooks require a zero-provider model");      \
  extern "C" void pops_register_provider_routes(                                             \
      ::pops::System<::pops::kNativeDimension>* system) {                                    \
    if (system == nullptr)                                                                   \
      throw std::invalid_argument("external riemann brick: null System provider registrar"); \
  }                                                                                          \
  extern "C" void pops_register_provider_routes_amr(                                         \
      ::pops::AmrSystem<::pops::kNativeDimension>* system) {                                 \
    if (system == nullptr)                                                                   \
      throw std::invalid_argument("external riemann brick: null AMR provider registrar");    \
  }

#define POPS_DEFINE_EXTERNAL_RIEMANN_BRICK(id, Flux, Model, model_identity, reqs_csv)              \
  static_assert(Model::dimension == ::pops::kNativeDimension,                                      \
                "external Riemann model must match the artifact native dimension");                \
  static_assert(::pops::nd::ConservationLaw<::pops::kNativeDimension, Model>,                      \
                "external Riemann model must satisfy the final exact-ranked law contract");        \
  static const bool POPS_REGISTER_BRICK_CAT_(pops_external_riemann_registered_, __LINE__) = [] {   \
    ::pops::runtime::program::BrickRegistry::instance().register_brick(                            \
        {(id), ("riemann"), (reqs_csv), "", (id), "uniform,amr", "", "", "",                       \
         "pops_brick_nvars,pops_brick_nproviders,pops_external_riemann_abi_version,"               \
         "pops_external_riemann_abi_key,pops_external_riemann_dimension,"                          \
         "pops_brick_residual_v4,pops_brick_install_system_v4,pops_brick_install_amr_v4,"          \
         "pops_brick_model_identity,pops_brick_kokkos_backend,"                                    \
         "pops_brick_kokkos_version,pops_register_provider_routes,"                                \
         "pops_register_provider_routes_amr"});                                                    \
    return true;                                                                                   \
  }();                                                                                             \
  extern "C" int pops_brick_nvars() {                                                              \
    return Model::n_vars;                                                                          \
  }                                                                                                \
  extern "C" int pops_brick_nproviders() {                                                         \
    return pops::provider_count_for<Model, ::pops::kNativeDimension>();                            \
  }                                                                                                \
  extern "C" const char* pops_brick_model_identity() {                                             \
    return (model_identity);                                                                       \
  }                                                                                                \
  extern "C" const char* pops_brick_kokkos_backend() {                                             \
    return ::pops::runtime::program::detail::external_kokkos_backend_identity();                   \
  }                                                                                                \
  extern "C" int pops_brick_kokkos_version() {                                                     \
    return ::pops::runtime::program::detail::external_kokkos_version_identity();                   \
  }                                                                                                \
  extern "C" int pops_external_riemann_abi_version() {                                             \
    return ::pops::runtime::program::kExternalRiemannBrickAbiVersion;                              \
  }                                                                                                \
  extern "C" const char* pops_external_riemann_abi_key() {                                         \
    return ::pops::runtime::program::kExternalRiemannBrickAbiKey;                                  \
  }                                                                                                \
  extern "C" int pops_external_riemann_dimension() {                                               \
    return ::pops::kNativeDimension;                                                               \
  }                                                                                                \
  extern "C" void pops_brick_install_system_v4(                                                    \
      void* system, const char* name, const char* provider_consumer_qid, const char* limiter,      \
      const char* recon, const char* time, double gamma, int substeps, int evolve, int stride,     \
      double positivity_floor, double weno_epsilon) {                                              \
    if (system == nullptr || name == nullptr || provider_consumer_qid == nullptr ||                \
        limiter == nullptr || recon == nullptr || time == nullptr)                                 \
      throw std::invalid_argument("external riemann brick: null System installer argument");       \
    ::pops::runtime::program::detail::external_install_system<::pops::kNativeDimension, Model,     \
                                                              Flux>(                               \
        *static_cast<::pops::System<::pops::kNativeDimension>*>(system), (id), name,               \
        provider_consumer_qid, limiter, recon, time, gamma, substeps, evolve != 0, stride,         \
        positivity_floor, weno_epsilon);                                                           \
  }                                                                                                \
  extern "C" void pops_brick_install_amr_v4(                                                       \
      void* system, const char* name, const char* provider_consumer_qid, const char* limiter,      \
      const char* recon, const char* time, double gamma, int substeps, int stride,                 \
      double positivity_floor, double weno_epsilon) {                                              \
    if (system == nullptr || name == nullptr || provider_consumer_qid == nullptr ||                \
        limiter == nullptr || recon == nullptr || time == nullptr)                                 \
      throw std::invalid_argument("external riemann brick: null AMR installer argument");          \
    ::pops::runtime::program::detail::external_install_amr<::pops::kNativeDimension, Model, Flux>( \
        *static_cast<::pops::AmrSystem<::pops::kNativeDimension>*>(system), (id), name,            \
        provider_consumer_qid, limiter, recon, time, gamma, substeps, stride, positivity_floor,    \
        weno_epsilon);                                                                             \
  }                                                                                                \
  extern "C" void pops_brick_residual_v4(                                                          \
      const double* U, double* R, const double* provider_values, const int* shape,                 \
      const double* spacing, const int* periodic, const char* lim, int recon_prim,                 \
      double pos_floor, double weno_epsilon) {                                                     \
    if (lim == nullptr)                                                                            \
      throw std::invalid_argument("external riemann brick: limiter id must be non-null");          \
    ::pops::runtime::program::detail::external_residual<::pops::kNativeDimension, Model, Flux>(    \
        U, R, provider_values, shape, spacing, periodic, lim, recon_prim != 0, pos_floor,          \
        weno_epsilon);                                                                             \
  }
