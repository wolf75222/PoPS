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
// in a SEPARATE `.so`, so it can never be a compile-time template parameter of the host's pre-built
// `make_block` (whose `if (riem == "hllc") build_block<..., HLLCFlux>` ladder is closed over the
// native fluxes). Instead the `.so` ITSELF performs the static instantiation: the
// `POPS_DEFINE_EXTERNAL_RIEMANN_BRICK` macro emits an `extern "C"` entry point that calls
// `build_block<Limiter, UserFlux>(...)` -- the user flux is a compile-time template parameter inside
// the `.so`, fully inlined, exactly like a native flux's `build_block` leaf. The host dlopens the
// `.so`, resolves that entry-point function pointer ONCE at install time, and calls it; the per-cell
// kernel then runs the statically-instantiated `UserFlux` functor with NO string comparison on the
// hot path. The only string is the limiter (a 4-way `if` resolved once per install, mirroring the
// built-in static-dispatch path).
//
// ABI v2 retains the flat residual adapter for diagnostics and adds native System/AMR installers.
// Those installers build directly on the runtime-owned MultiFab/hierarchy, so production execution
// is zero-copy and keeps the ordinary Kokkos, MPI-halo and AMR-reflux paths.  The exact ABI identity,
// exported symbol set and library digest are authenticated before any installer is called.

#include <pops/runtime/program/external_brick.hpp>

#include <pops/core/identity/sha256.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/builders/compiled/flat_grid.hpp>
#include <pops/runtime/builders/block/block_builder.hpp>  // build_block<Limiter, Flux>, block_n_ghost
#include <pops/runtime/builders/scheme_dispatch.hpp>  // dispatch_limiter: ONE limiter-route dispatch generator (ADC-640)
#include <pops/runtime/config/dispatch_tags.hpp>  // validate_limiter
#include <pops/numerics/fv/reconstruction.hpp>    // NoSlope / Minmod / VanLeer / Weno5

#include <pops/runtime/dynamic/dynlib.hpp>  // portable dlopen<->LoadLibraryW (ADC-99)

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pops::runtime::program {

inline constexpr int kExternalRiemannBrickAbiVersion = 2;
inline constexpr const char* kExternalRiemannBrickAbiKey =
    "pops.external-riemann/v2;scalar=f64;index=i32;periodicity=xy";

inline constexpr const char* kExternalRiemannBrickAbiVersionSymbol =
    "pops_external_riemann_abi_version";
inline constexpr const char* kExternalRiemannBrickAbiKeySymbol =
    "pops_external_riemann_abi_key";
inline constexpr const char* kExternalRiemannBrickResidualSymbol = "pops_brick_residual_v2";
inline constexpr const char* kExternalRiemannBrickInstallSystemSymbol =
    "pops_brick_install_system_v2";
inline constexpr const char* kExternalRiemannBrickInstallAmrSymbol =
    "pops_brick_install_amr_v2";
inline constexpr const char* kExternalRiemannBrickModelIdentitySymbol =
    "pops_brick_model_identity";
inline constexpr const char* kExternalRiemannBrickKokkosBackendSymbol =
    "pops_brick_kokkos_backend";
inline constexpr const char* kExternalRiemannBrickKokkosVersionSymbol =
    "pops_brick_kokkos_version";

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

// Builds the block closures for the external flux @p Flux at limiter @p lim. The flux is a
// COMPILE-TIME template parameter of build_block (the same leaf the native string ladder routes to),
// so it is fully inlined; the only runtime branch is the limiter (resolved ONCE here, not per cell).
// Mirrors make_block's limiter ladder but with the flux fixed -- no riemann string comparison.
template <class Model, class Flux>
BlockClosures external_make_block(const Model& m, const std::string& lim, const GridContext& ctx,
                                  bool recon_prim, Real pos_floor) {
  validate_limiter(lim, "external riemann brick");
  const char* kCtx = "external riemann brick";
  return dispatch_limiter(parse_limiter_route(lim, kCtx), kCtx, [&](auto tag) {
    using L = typename decltype(tag)::type;
    return build_block<L, Flux>(m, ctx, /*imex=*/false, recon_prim, "explicit", {}, {}, nullptr,
                                pos_floor);
  });
}

// One explicit residual R = -div F(U) + S evaluated with the external flux @p Flux on @p Model. Same
// marshaling as compiled_block::residual (flat arrays, local single-grid mesh, aux from the host) --
// only the flux is the user's, instantiated statically by build_block. Used by the macro's
// extern "C" pops_brick_residual_v2 entry point.
template <class Model, class Flux>
void external_residual(const double* U, double* R, const double* aux_in, int n, double dx,
                       double dy, Periodicity periodicity, const std::string& lim, bool recon_prim,
                       double pos_floor) {
  if (U == nullptr || R == nullptr)
    throw std::invalid_argument("external riemann brick: U and R must be non-null");
  if (n <= 0 || !std::isfinite(dx) || dx <= 0 || !std::isfinite(dy) || dy <= 0)
    throw std::invalid_argument(
        "external riemann brick: n and Cartesian spacings must be strictly positive");
  flat_grid::LocalGrid lg =
      flat_grid::make_grid(n, dx, dy, periodicity, aux_in, aux_comps<Model>());
  MultiFab Umf(lg.ba, lg.dm, Model::n_vars, block_n_ghost(lim)),
      Rmf(lg.ba, lg.dm, Model::n_vars, 0);
  flat_grid::fill_interior(Umf, U, n, Model::n_vars);
  const GridContext ctx{lg.dom, lg.bc, lg.geom, &lg.aux};
  Model model{};
  BlockClosures clo =
      external_make_block<Model, Flux>(model, lim, ctx, recon_prim, static_cast<Real>(pos_floor));
  clo.rhs_into(Umf, Rmf);
  flat_grid::extract(Rmf, R, n, Model::n_vars);
}

template <class Model, class Flux>
void external_install_system(System& sys, const std::string& name, const std::string& limiter,
                             const std::string& recon, const std::string& time, double gamma,
                             int substeps, bool evolve, int stride, double positivity_floor,
                             double weno_epsilon) {
  if (name.empty())
    throw std::invalid_argument("external riemann brick: block name must be non-empty");
  if (substeps < 1 || stride < 1)
    throw std::invalid_argument("external riemann brick: substeps and stride must be >= 1");
  if (!(positivity_floor >= 0.0) || !std::isfinite(positivity_floor))
    throw std::invalid_argument(
        "external riemann brick: positivity_floor must be finite and >= 0");
  if (!std::isfinite(weno_epsilon) || weno_epsilon <= 0.0)
    throw std::invalid_argument("external riemann brick: weno_epsilon must be finite and > 0");
  if (weno_epsilon != static_cast<double>(kWenoEpsilon) && limiter != "weno5")
    throw std::invalid_argument(
        "external riemann brick: weno_epsilon applies to limiter='weno5' only");

  const TimeRouteId time_route = parse_time_route(time, "external riemann brick");
  if (time_route == TimeRouteId::kImexRkArs222)
    throw std::runtime_error(
        "external riemann brick: time route 'imexrk_ars222' is not wired on the compiled path");
  const bool imex = time_route == TimeRouteId::kImex;
  const bool recon_prim =
      parse_recon_route(recon, "external riemann brick") == ReconRouteId::kPrimitive;
  validate_limiter(limiter, "external riemann brick");

  Model model{};
  sys.ensure_aux_width(aux_comps<Model>());
  const GridContext ctx = sys.grid_context(name);
  BlockClosures closures = dispatch_limiter(
      parse_limiter_route(limiter, "external riemann brick"), "external riemann brick",
      [&](auto tag) {
        using Limiter = typename decltype(tag)::type;
        return build_block<Limiter, Flux>(model, ctx, imex, recon_prim, route_token(time_route),
                                          {}, {}, nullptr,
                                          static_cast<Real>(positivity_floor), false,
                                          static_cast<Real>(weno_epsilon));
      });
  auto max_speed = make_max_speed(model, ctx);
  auto poisson_rhs = make_poisson_rhs(model);
  sys.install_block(name, Model::n_vars, Model::conservative_vars(), Model::primitive_vars(), gamma,
                    std::move(closures), std::move(max_speed), std::move(poisson_rhs), substeps,
                    evolve, stride);
  auto conversion = make_cell_convert(model);
  sys.set_block_conversion(name, std::move(conversion.first), std::move(conversion.second));
  sys.set_block_dt_bounds(name, make_source_frequency(model, ctx), make_stability_dt(model, ctx));
  sys.set_block_ghosts(name, block_n_ghost(limiter));
}

template <class Model, class Flux>
void external_install_amr(AmrSystem& sys, const std::string& name,
                          const std::string& limiter, const std::string& recon,
                          const std::string& time, double gamma, int substeps, int stride,
                          double positivity_floor, double weno_epsilon) {
  if (name.empty())
    throw std::invalid_argument("external riemann brick: block name must be non-empty");
  if (substeps < 1 || stride < 1)
    throw std::invalid_argument("external riemann brick: substeps and stride must be >= 1");
  if (!(positivity_floor >= 0.0) || !std::isfinite(positivity_floor))
    throw std::invalid_argument(
        "external riemann brick: positivity_floor must be finite and >= 0");
  if (!std::isfinite(weno_epsilon) || weno_epsilon <= 0.0)
    throw std::invalid_argument("external riemann brick: weno_epsilon must be finite and > 0");
  if (weno_epsilon != static_cast<double>(kWenoEpsilon) && limiter != "weno5")
    throw std::invalid_argument(
        "external riemann brick: weno_epsilon applies to limiter='weno5' only");

  validate_limiter(limiter, "external riemann brick");
  const bool recon_prim =
      parse_recon_route(recon, "external riemann brick") == ReconRouteId::kPrimitive;
  const TimeRouteId time_route = parse_time_route(time, "external riemann brick");
  if (time_route == TimeRouteId::kImexRkArs222)
    throw std::runtime_error(
        "external riemann brick: time route 'imexrk_ars222' is not wired on the AMR compiled path");
  const bool imex = time_route == TimeRouteId::kImex;
  AmrTimeMethod time_method = AmrTimeMethod::kEuler;
  if (time_route == TimeRouteId::kExplicitSsprk2)
    time_method = AmrTimeMethod::kSsprk2;
  else if (time_route == TimeRouteId::kSsprk3)
    time_method = AmrTimeMethod::kSsprk3;
  if (imex && time_method != AmrTimeMethod::kEuler)
    throw std::runtime_error(
        "external riemann brick: SSPRK2/SSPRK3 cannot be combined with AMR IMEX");

  Model model{};
  AmrCompiledBlockBuilder builder =
      [model, limiter, time_method](
          const ::pops::detail::SharedAmrLayout& layout, const std::string& block_name,
          const std::vector<double>& density, bool has_density,
          const std::vector<double>& state, bool has_state, double block_gamma,
          int block_substeps, bool block_recon_prim, bool block_imex, int block_stride,
          const std::vector<std::string>& implicit_vars,
          const std::vector<std::string>& implicit_roles, double block_positivity_floor,
          double block_weno_epsilon, bool wave_speed_cache) {
        if (!implicit_vars.empty() || !implicit_roles.empty())
          throw std::runtime_error(
              "external riemann brick: partial IMEX masks are not part of ABI v2");
        if (wave_speed_cache)
          throw std::runtime_error(
              "external riemann brick: wave_speed_cache is not part of ABI v2");
        return dispatch_limiter(
            parse_limiter_route(limiter, "external riemann brick"),
            "external riemann brick", [&](auto tag) {
              using Limiter = typename decltype(tag)::type;
              return ::pops::detail::build_amr_block<Model, Limiter, Flux>(
                  model, layout, block_name, density, has_density, block_gamma,
                  block_substeps, block_recon_prim, block_imex, block_stride, {}, NewtonOptions{},
                  has_state ? &state : nullptr, false, time_method, block_positivity_floor,
                  block_weno_epsilon, false);
            });
      };
  sys.set_compiled_block(Model::n_vars, gamma, substeps, std::move(builder), name, recon_prim, imex,
                         static_cast<int>(time_method), stride, {}, {}, positivity_floor,
                         weno_epsilon, false);
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
  using ResidualFn = void (*)(const double*, double*, const double*, int, double, double, int, int,
                              const char*, int, double);
  using InstallSystemFn = void (*)(void*, const char*, const char*, const char*, const char*, double,
                                   int, int, int, double, double);
  using InstallAmrFn = void (*)(void*, const char*, const char*, const char*, const char*, double,
                                int, int, double, double);

  // dlopen @p so_path, read + register its manifest, and resolve the entry points for brick @p id.
  // Throws std::runtime_error if the library cannot be opened, does not export pops_brick_manifest /
  // the versioned residual ABI (not a PoPS external Riemann brick), or does not register @p id as a
  // riemann brick (a clear, actionable message names the id).
  ExternalBrickHandle(const std::string& so_path, const std::string& id,
                      const std::string& expected_sha256 = {}, int expected_nvars = -1,
                      int expected_naux = -1, const std::string& expected_model_identity = {})
      : id_(id) {
    if (!expected_sha256.empty() && file_sha256(so_path) != expected_sha256)
      throw std::runtime_error("external riemann brick '" + id_ +
                               "' library digest changed after descriptor resolution");
    handle_ = dynlib::open(so_path);
    if (!dynlib::valid(handle_))
      throw std::runtime_error("external riemann brick: cannot dlopen '" + so_path +
                               "': " + dynlib::last_error());
    try {
      auto manifest_fn =
          reinterpret_cast<const char* (*)()>(dynlib::sym(handle_, "pops_brick_manifest"));
      if (manifest_fn == nullptr)
        throw std::runtime_error(
            "external riemann brick '" + so_path +
            "' does not export pops_brick_manifest(); it is not a PoPS brick .so");
      const std::vector<BrickManifestEntry> entries = parse_manifest_json(manifest_fn());
      const auto selected = std::find_if(entries.begin(), entries.end(), [&](const auto& entry) {
        return entry.id == id_;
      });
      if (selected == entries.end())
        throw std::runtime_error("external riemann brick '" + id_ +
                                 "' not found in the manifest of '" + so_path + "'");
      if (selected->category != "riemann")
        throw std::runtime_error("external brick '" + id_ + "' is registered as category '" +
                                 selected->category + "', not 'riemann'");

      require_abi_symbol(*selected, kExternalRiemannBrickAbiVersionSymbol);
      require_abi_symbol(*selected, kExternalRiemannBrickAbiKeySymbol);
      require_abi_symbol(*selected, kExternalRiemannBrickResidualSymbol);
      require_abi_symbol(*selected, kExternalRiemannBrickInstallSystemSymbol);
      require_abi_symbol(*selected, kExternalRiemannBrickInstallAmrSymbol);
      require_abi_symbol(*selected, kExternalRiemannBrickModelIdentitySymbol);
      require_abi_symbol(*selected, kExternalRiemannBrickKokkosBackendSymbol);
      require_abi_symbol(*selected, kExternalRiemannBrickKokkosVersionSymbol);
      auto version_fn = reinterpret_cast<int (*)()>(
          dynlib::sym(handle_, kExternalRiemannBrickAbiVersionSymbol));
      auto abi_key_fn = reinterpret_cast<const char* (*)()>(
          dynlib::sym(handle_, kExternalRiemannBrickAbiKeySymbol));
      if (version_fn == nullptr || abi_key_fn == nullptr)
        throw std::runtime_error(
            "external riemann brick '" + id_ +
            "' uses the legacy unversioned residual ABI; rebuild it with the current "
            "POPS_DEFINE_EXTERNAL_RIEMANN_BRICK macro");
      const int version = version_fn();
      const char* abi_key = abi_key_fn();
      if (version != kExternalRiemannBrickAbiVersion || abi_key == nullptr ||
          std::string(abi_key) != kExternalRiemannBrickAbiKey)
        throw std::runtime_error(
            "external riemann brick '" + id_ + "' has incompatible residual ABI version/key; "
            "rebuild it with the current PoPS headers");
      residual_ = reinterpret_cast<ResidualFn>(
          dynlib::sym(handle_, kExternalRiemannBrickResidualSymbol));
      if (residual_ == nullptr)
        throw std::runtime_error("external riemann brick '" + id_ + "' declares but does not export " +
                                 std::string(kExternalRiemannBrickResidualSymbol));
      install_system_ = reinterpret_cast<InstallSystemFn>(
          dynlib::sym(handle_, kExternalRiemannBrickInstallSystemSymbol));
      install_amr_ = reinterpret_cast<InstallAmrFn>(
          dynlib::sym(handle_, kExternalRiemannBrickInstallAmrSymbol));
      auto nvars_fn = reinterpret_cast<int (*)()>(dynlib::sym(handle_, "pops_brick_nvars"));
      auto naux_fn = reinterpret_cast<int (*)()>(dynlib::sym(handle_, "pops_brick_naux"));
      auto model_identity_fn = reinterpret_cast<const char* (*)()>(
          dynlib::sym(handle_, kExternalRiemannBrickModelIdentitySymbol));
      auto kokkos_backend_fn = reinterpret_cast<const char* (*)()>(
          dynlib::sym(handle_, kExternalRiemannBrickKokkosBackendSymbol));
      auto kokkos_version_fn = reinterpret_cast<int (*)()>(
          dynlib::sym(handle_, kExternalRiemannBrickKokkosVersionSymbol));
      if (install_system_ == nullptr || install_amr_ == nullptr || nvars_fn == nullptr ||
          naux_fn == nullptr || model_identity_fn == nullptr || kokkos_backend_fn == nullptr ||
          kokkos_version_fn == nullptr)
        throw std::runtime_error("external riemann brick '" + id_ +
                                 "' is missing a declared native installer/count symbol");
      nvars_ = nvars_fn();
      naux_ = naux_fn();
      const char* model_identity = model_identity_fn();
      if (model_identity == nullptr || *model_identity == '\0')
        throw std::runtime_error("external riemann brick '" + id_ +
                                 "' exports an empty model identity");
      if ((expected_nvars >= 0 && nvars_ != expected_nvars) ||
          (expected_naux >= 0 && naux_ != expected_naux))
        throw std::runtime_error("external riemann brick '" + id_ +
                                 "' model shape disagrees with the compiled model descriptor");
      if (!expected_model_identity.empty() && model_identity != expected_model_identity)
        throw std::runtime_error("external riemann brick '" + id_ +
                                 "' targets a different compiled model identity");
      const char* brick_backend = kokkos_backend_fn();
      const char* host_backend = detail::external_kokkos_backend_identity();
      const int host_kokkos_version = detail::external_kokkos_version_identity();
      if (brick_backend == nullptr || std::string(brick_backend) != host_backend ||
          kokkos_version_fn() != host_kokkos_version)
        throw std::runtime_error("external riemann brick '" + id_ +
                                 "' was built for a different Kokkos backend/version");

      // Registration happens only after the category-specific ABI is authenticated.  A rejected
      // library can therefore never publish an unusable external_cpp descriptor in this image.
      for (const auto& entry : entries)
        BrickRegistry::instance().register_brick(entry);
      requirements_ = selected->requirements;
    } catch (...) {
      dynlib::close(handle_);
      handle_ = nullptr;
      throw;
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

  void install_system(void* system, const std::string& name, const std::string& limiter,
                      const std::string& recon, const std::string& time, double gamma,
                      int substeps, bool evolve, int stride, double positivity_floor,
                      double weno_epsilon) const {
    install_system_(system, name.c_str(), limiter.c_str(), recon.c_str(), time.c_str(), gamma,
                    substeps, evolve ? 1 : 0, stride, positivity_floor, weno_epsilon);
  }

  void install_amr(void* system, const std::string& name, const std::string& limiter,
                   const std::string& recon, const std::string& time, double gamma, int substeps,
                   int stride, double positivity_floor, double weno_epsilon) const {
    install_amr_(system, name.c_str(), limiter.c_str(), recon.c_str(), time.c_str(), gamma,
                 substeps, stride, positivity_floor, weno_epsilon);
  }

  // The CSV of model capabilities the brick requires (from its manifest); "" when none.
  const std::string& requirements() const { return requirements_; }

  const std::string& id() const { return id_; }

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
      throw std::runtime_error(
          "external riemann brick '" + entry.id + "' manifest does not declare required ABI symbol '" +
          symbol + "'; legacy manifests must be rebuilt, not adapted");
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
        throw std::runtime_error("external riemann brick manifest has an unterminated bricks array");
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
      if (std::any_of(entries.begin(), entries.end(), [&](const auto& row) { return row.id == e.id; }))
        throw std::runtime_error("external riemann brick manifest contains duplicate id '" + e.id + "'");
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
  int naux_ = -1;
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
//   POPS_DEFINE_EXTERNAL_RIEMANN_BRICK("my_riemann", MyRiemann,
//                                     pops::CompositeModel<pops::Euler, ...>,
//                                     "<compiled-model-hash>", "pressure,wave_speeds");
//   POPS_DEFINE_BRICK_MANIFEST();  // exports the manifest reader (once per .so)
//
// @p id          the brick id a user selects via pops.lib.riemann.User(id);
// @p Flux        the narrow two-trace NumericalFlux policy (numerics/fv/numerical_flux.hpp);
// @p Model       a TOP-LEVEL ALIAS of the CompositeModel the .so instantiates the flux against (write
//                `using Model = pops::CompositeModel<...>;` first and pass the alias -- a bare
//                CompositeModel<A, B, C> has commas the preprocessor would split);
// @p model_identity the exact CompiledModel.model_hash this DSO targets; same-size models are not
//                interchangeable and are rejected before install;
// @p reqs_csv    the CSV of model capabilities the brick requires (surfaced in the manifest).
//
// The emitted pops_brick_residual_v2 instantiates build_block<Limiter, Flux> at the .so's compile
// time: the flux is a STATIC template argument, never a per-cell string lookup. pops_brick_nvars /
// pops_brick_naux let the host size its marshaling arrays (same role as pops_compiled_nvars/_naux).
//
// ABI WARNING: the brick `.so` MUST be compiled against the SAME Kokkos backend and version (and the
// same pops headers) as the host binary that dlopens it -- the residual runs the host's Kokkos
// runtime. Installation must therefore pass through the authenticated component loader, which
// validates the exact component manifest and platform/ABI evidence before publishing the handle.
#define POPS_DEFINE_EXTERNAL_RIEMANN_BRICK(id, Flux, Model, model_identity, reqs_csv)         \
  static const bool POPS_REGISTER_BRICK_CAT_(pops_external_riemann_registered_, __LINE__) =   \
      [] {                                                                                    \
        ::pops::runtime::program::BrickRegistry::instance().register_brick(                   \
            {(id), ("riemann"), (reqs_csv), "", (id), "uniform,amr", "", "", "",          \
             "pops_brick_nvars,pops_brick_naux,pops_external_riemann_abi_version,"           \
             "pops_external_riemann_abi_key,pops_brick_residual_v2,"                         \
             "pops_brick_install_system_v2,pops_brick_install_amr_v2,"                       \
             "pops_brick_model_identity,pops_brick_kokkos_backend,"                          \
             "pops_brick_kokkos_version"});                                                   \
        return true;                                                                          \
      }();                                                                                    \
  extern "C" int pops_brick_nvars() { return Model::n_vars; }                                \
  extern "C" int pops_brick_naux() { return pops::aux_comps<Model>(); }                      \
  extern "C" const char* pops_brick_model_identity() { return (model_identity); }            \
  extern "C" const char* pops_brick_kokkos_backend() {                                     \
    return ::pops::runtime::program::detail::external_kokkos_backend_identity();                 \
  }                                                                                             \
  extern "C" int pops_brick_kokkos_version() {                                             \
    return ::pops::runtime::program::detail::external_kokkos_version_identity();                 \
  }                                                                                             \
  extern "C" int pops_external_riemann_abi_version() {                                      \
    return ::pops::runtime::program::kExternalRiemannBrickAbiVersion;                         \
  }                                                                                           \
  extern "C" const char* pops_external_riemann_abi_key() {                                  \
    return ::pops::runtime::program::kExternalRiemannBrickAbiKey;                             \
  }                                                                                           \
  extern "C" void pops_brick_install_system_v2(                                             \
      void* system, const char* name, const char* limiter, const char* recon, const char* time, \
      double gamma, int substeps, int evolve, int stride, double positivity_floor,             \
      double weno_epsilon) {                                                                   \
    if (system == nullptr || name == nullptr || limiter == nullptr || recon == nullptr ||       \
        time == nullptr)                                                                       \
      throw std::invalid_argument("external riemann brick: null System installer argument");   \
    ::pops::runtime::program::detail::external_install_system<Model, Flux>(                    \
        *static_cast<::pops::System*>(system), name, limiter, recon, time, gamma, substeps,     \
        evolve != 0, stride, positivity_floor, weno_epsilon);                                  \
  }                                                                                            \
  extern "C" void pops_brick_install_amr_v2(                                                \
      void* system, const char* name, const char* limiter, const char* recon, const char* time, \
      double gamma, int substeps, int stride, double positivity_floor, double weno_epsilon) {  \
    if (system == nullptr || name == nullptr || limiter == nullptr || recon == nullptr ||       \
        time == nullptr)                                                                       \
      throw std::invalid_argument("external riemann brick: null AMR installer argument");      \
    ::pops::runtime::program::detail::external_install_amr<Model, Flux>(                       \
        *static_cast<::pops::AmrSystem*>(system), name, limiter, recon, time, gamma, substeps,  \
        stride, positivity_floor, weno_epsilon);                                               \
  }                                                                                            \
  extern "C" void pops_brick_residual_v2(                                                   \
      const double* U, double* R, const double* aux, int n, double dx, double dy,             \
      int periodic_x, int periodic_y, const char* lim, int recon_prim, double pos_floor) {    \
    if ((periodic_x != 0 && periodic_x != 1) || (periodic_y != 0 && periodic_y != 1))         \
      throw std::invalid_argument(                                                            \
          "external riemann brick: periodic_x and periodic_y must be exact 0/1 values");     \
    if (lim == nullptr)                                                                       \
      throw std::invalid_argument("external riemann brick: limiter id must be non-null");     \
    ::pops::runtime::program::detail::external_residual<Model, Flux>(                         \
        U, R, aux, n, dx, dy, ::pops::Periodicity{periodic_x != 0, periodic_y != 0}, lim,      \
        recon_prim != 0, pos_floor);                                                          \
  }
