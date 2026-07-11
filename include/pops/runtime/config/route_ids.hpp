#pragma once

#include <pops/runtime/config/dispatch_tags.hpp>    // kLimiters / kRiemanns (spatial scheme tags)
#include <pops/runtime/dynamic/model_registry.hpp>  // kTransports / kSources / kElliptics

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

/// @file
/// @brief TYPED native route IDs (ADC-584): one enum per behavior family, anchored to the
///        existing single-source tag tables.
///
/// Before this header the algorithmic choices (Riemann flux, limiter, reconstruction variables,
/// time treatment, field solver, Poisson BC, layout, model bricks) crossed the runtime as free
/// strings; the registries (dispatch_tags.hpp, model_registry.hpp) validated them but nothing
/// TYPED existed, so every consumer re-compared string literals and two silent defaults survived
/// (add_compiled_model time -> ssprk2, limiter_n_ghost -> 2). Here every family gets:
///
///   - an `enum class *RouteId` (the typed identity, the ONLY normal selection mechanism);
///   - a `k*Routes[]` row per route: wire token (the legacy string, kept for ABI/debug/messages),
///     native entry point, requirements and limitations (CSV, documentary -- the hard guards stay
///     at the call sites, exactly like the capability flags of RiemannTag);
///   - `parse_*_route(token)` (throws on an unknown token, citing the family, the requested token
///     and the valid set -- NEVER a default) and `route_token(id)` (the BOUNDED adapter back to
///     the strings the current C++ calls consume).
///
/// The tables MIRROR the historical registries row for row; static_asserts below lock the absence
/// of drift (same pattern as the limiter_n_ghost_ct / transport_n_vars_ct locks). This header is
/// deliberately LIGHT (strings + enums only, no numerics dependency) so it can be included by the
/// bindings, the loaders and the tests at zero cost. The Python mirror is
/// python/pops/runtime/routes.py; tests/python/architecture/test_route_registry_parity.py locks the two
/// registries against each other at the source level (no build required).

namespace pops {

/// Behavior family of a route (one enum per family below; this tag names the family in
/// diagnostics and in the aggregated manifest).
enum class RouteFamily {
  kLimiter,
  kRiemann,
  kRecon,
  kTime,
  kSplitting,
  kFieldSolver,
  kPoissonBc,
  kLayout,
  kTransport,
  kSource,
  kElliptic,
  kSourceStage,
  kPoissonRhs,
  kWall,
};

/// Family name for messages / structured reports ("riemann", "limiter", ...).
constexpr const char* route_family_name(RouteFamily f) {
  switch (f) {
    case RouteFamily::kLimiter:
      return "limiter";
    case RouteFamily::kRiemann:
      return "riemann";
    case RouteFamily::kRecon:
      return "recon";
    case RouteFamily::kTime:
      return "time";
    case RouteFamily::kSplitting:
      return "splitting";
    case RouteFamily::kFieldSolver:
      return "field_solver";
    case RouteFamily::kPoissonBc:
      return "poisson_bc";
    case RouteFamily::kLayout:
      return "layout";
    case RouteFamily::kTransport:
      return "transport";
    case RouteFamily::kSource:
      return "source";
    case RouteFamily::kElliptic:
      return "elliptic";
    case RouteFamily::kSourceStage:
      return "source_stage";
    case RouteFamily::kPoissonRhs:
      return "poisson_rhs";
    case RouteFamily::kWall:
      return "wall";
  }
  return "?";  // unreachable (all enumerators handled); keeps -Wreturn-type quiet
}

/// One typed route: wire token (legacy string, ABI/debug only), native entry point and the
/// documentary requirements / limitations (CSV; "" = none). `index` mirrors the enumerator value
/// of the family enum -- locked by the static_asserts at the end of this file.
struct RouteInfo {
  int index;                 ///< enumerator value in the family enum (locked, no drift)
  const char* token;         ///< wire token, e.g. "hll" (the ONLY string the adapter emits)
  const char* native_entry;  ///< native entry point, e.g. "pops::HLLFlux"
  const char* requirements;  ///< CSV of model/route requirements (documentary, cf. RiemannTag)
  const char* limitations;   ///< CSV of route limitations (documentary; drives reports/tests)
};

// --- Riemann flux routes (mirror of kRiemanns, dispatch_tags.hpp) ---------------------------
enum class RiemannRouteId : int {
  kRusanov = 0, kHll = 1, kHllc = 2, kRoe = 3, kEulerHllc = 4, kEulerRoe = 5
};
inline constexpr RouteInfo kRiemannRoutes[] = {
    {0, "rusanov", "pops::RusanovFlux", "max_wave_speed", ""},
    {1, "hll", "pops::HLLFlux", "physical_flux,wave_speeds", ""},
    {2, "hllc", "pops::HLLCFlux",
     "physical_flux,pressure,wave_speeds,contact_speed,hllc_star_state",
     "polar geometry not wired; generic-only (ADC-590), requires HasHLLCStructure"},
    {3, "roe", "pops::RoeFlux", "physical_flux,roe_average",
     "polar geometry not wired; generic-only (ADC-590), requires HasRoeDissipation"},
    {4, "euler_hllc", "pops::EulerHLLCFlux2D", "physical_flux,pressure,euler_2d_layout",
     "4-variable canonical Euler (rho,mx,my,E) only; explicit route, never a fallback; polar not "
     "wired"},
    {5, "euler_roe", "pops::EulerRoeFlux2D", "physical_flux,pressure,euler_2d_layout",
     "4-variable canonical Euler (rho,mx,my,E) only; explicit route, never a fallback; polar not "
     "wired"},
};

// --- Reconstruction limiter routes (mirror of kLimiters, dispatch_tags.hpp) ------------------
enum class LimiterRouteId : int {
  kNone = 0, kMinmod = 1, kVanLeer = 2, kWeno5 = 3
};
inline constexpr RouteInfo kLimiterRoutes[] = {
    {0, "none", "pops::NoSlope", "", ""},
    {1, "minmod", "pops::Minmod", "", ""},
    {2, "vanleer", "pops::VanLeer", "", ""},
    {3, "weno5", "pops::Weno5Z", "3-cell halo",
     "prototype backend not wired (host order-1 residual)"},
};

// --- Reconstructed-variable routes (recon_prim flag today) ------------------------------------
enum class ReconRouteId : int { kConservative = 0, kPrimitive = 1 };
inline constexpr RouteInfo kReconRoutes[] = {
    {0, "conservative", "pops::make_block(recon_prim=false)", "", ""},
    {1, "primitive", "pops::make_block(recon_prim=true)", "primitive_vars",
     "requires a model exposing primitive variables"},
};

// --- Time-treatment routes (NEW single source: no historical registry existed; the accepted
// value sets of the entry points diverged -- pybind add_block took the full set while the AOT
// .so ABI rejects ssprk3/euler. The limitations column carries that split as DATA.) -----------
enum class TimeRouteId : int {
  kExplicitSsprk2 = 0, kSsprk3 = 1, kForwardEuler = 2, kImex = 3, kImexRkArs222 = 4
};
inline constexpr RouteInfo kTimeRoutes[] = {
    {0, "explicit", "pops::SSPRK2", "", ""},
    {1, "ssprk3", "pops::SSPRK3", "",
     "aot .so ABI not wired (SSPRK2-only extern C entry); native add_block/add_native_block only"},
    {2, "euler", "pops::ForwardEuler", "",
     "aot .so ABI not wired; native add_block/add_native_block only; validation use, never "
     "default"},
    {3, "imex", "pops::AdvanceImex", "implicit source term", ""},
    {4, "imexrk_ars222", "pops::ImexRkArs222", "implicit source term",
     "composed native add_block only (.so ABIs do not carry the RK tableau)"},
};

// --- Splitting routes (system time scheme: Lie / Strang) --------------------------------------
enum class SplittingRouteId : int { kLie = 0, kStrang = 1 };
inline constexpr RouteInfo kSplittingRoutes[] = {
    {0, "lie", "pops::SystemStepper(lie)", "", ""},
    {1, "strang", "pops::SystemStepper(strang)", "",
     "H(dt/2) S(dt) H(dt/2); requires a condensed source stage"},
};

// --- Field-solver (elliptic solve) routes ------------------------------------------------------
enum class FieldSolverRouteId : int {
  kGeometricMg = 0, kFft = 1, kFftSpectral = 2, kPolar = 3
};
inline constexpr RouteInfo kFieldSolverRoutes[] = {
    {0, "geometric_mg", "pops::GeometricMG", "", ""},
    {1, "fft", "pops::PoissonFFTSolver", "periodic bc,constant coefficient",
     "walls / variable epsilon not wired; non power-of-two grid falls back to O(n^2) DFT"},
    {2, "fft_spectral", "pops::PoissonFFTSolver(spectral)", "periodic bc,constant coefficient",
     "walls / variable epsilon not wired; continuous symbol -(kx^2+ky^2)"},
    {3, "polar", "pops::PolarPoissonSolver", "polar geometry", "annular polar only (r_min > 0)"},
};

// --- Poisson boundary-condition routes ---------------------------------------------------------
enum class PoissonBcRouteId : int {
  kAuto = 0, kPeriodic = 1, kDirichlet = 2, kNeumann = 3
};
inline constexpr RouteInfo kPoissonBcRoutes[] = {
    {0, "auto", "resolved from the wall/periodic system config", "", ""},
    {1, "periodic", "pops::fill_boundary(periodic)", "", ""},
    {2, "dirichlet", "pops::PhysicalBc(dirichlet)", "", ""},
    {3, "neumann", "pops::PhysicalBc(neumann)", "", ""},
};

// --- Layout routes (single-level System vs AMR hierarchy) --------------------------------------
enum class LayoutRouteId : int { kUniform = 0, kAmr = 1 };
inline constexpr RouteInfo kLayoutRoutes[] = {
    {0, "uniform", "pops::System", "", ""},
    {1, "amr", "pops::AmrSystem", "",
     "refinement ratio 2 (kAmrRefRatio); fft field solver not wired"},
};

// --- Model brick routes (mirror of kTransports / kSources / kElliptics, model_registry.hpp).
// The alias spellings of kSources (magnetic==lorentz, potential_magnetic==potential_lorentz) are
// PARSE-ONLY compatibility: they resolve to the canonical route; route_token emits the canonical
// spelling. ---------------------------------------------------------------------------------
enum class TransportRouteId : int {
  kExb = 0, kCompressible = 1, kIsothermal = 2
};
inline constexpr RouteInfo kTransportRoutes[] = {
    {0, "exb", "pops::ExBVelocity", "", "scalar (1 var); no fluid source"},
    {1, "compressible", "pops::CompressibleFlux", "", "polar geometry not wired"},
    {2, "isothermal", "pops::IsothermalFlux", "", ""},
};

enum class SourceRouteId {
  kNone = 0,
  kPotential = 1,
  kGravity = 2,
  kMagneticLorentz = 3,
  kPotentialMagneticLorentz = 4,
};
inline constexpr RouteInfo kSourceRoutes[] = {
    {0, "none", "pops::NoSource", "", ""},
    {1, "potential", "pops::PotentialForce", "fluid transport (>= 3 vars)", ""},
    {2, "gravity", "pops::GravityForce", "fluid transport (>= 3 vars)", ""},
    {3, "magnetic", "pops::MagneticLorentzForce", "fluid transport (>= 3 vars),aux B_z channel",
     "explicit regime (stiff regime -> condensed Schur stage)"},
    {4, "potential_magnetic", "pops::CompositeSource<PotentialForce, MagneticLorentzForce>",
     "fluid transport (>= 3 vars),aux B_z channel", ""},
};

enum class EllipticRouteId : int {
  kCharge = 0, kBackground = 1, kGravity = 2
};
inline constexpr RouteInfo kEllipticRoutes[] = {
    {0, "charge", "pops::ChargeDensity", "", ""},
    {1, "background", "pops::BackgroundDensity", "", ""},
    {2, "gravity", "pops::GravityCoupling", "", ""},
};

// --- Condensed source-stage routes (set_source_stage kind) -------------------------------------
enum class SourceStageRouteId : int { kElectrostaticLorentz = 0 };
inline constexpr RouteInfo kSourceStageRoutes[] = {
    {0, "electrostatic_lorentz", "pops::ElectrostaticLorentzCondensedSchur",
     "magnetic field B_z,system potential phi", "theta in (0, 1]"},
};

// --- Poisson right-hand-side routes (set_poisson rhs) ------------------------------------------
enum class PoissonRhsRouteId : int { kChargeDensity = 0, kComposite = 1 };
inline constexpr RouteInfo kPoissonRhsRoutes[] = {
    {0, "charge_density", "per-block ChargeDensity bricks summed", "",
     "alias of composite when every block carries a charge density (bit-identical)"},
    {1, "composite", "per-block elliptic bricks summed", "", ""},
};

// --- Wall predicate routes (set_poisson wall) ---------------------------------------------------
enum class WallRouteId : int { kNone = 0, kCircle = 1 };
inline constexpr RouteInfo kWallRoutes[] = {
    {0, "none", "no wall (fully periodic/physical domain)", "", ""},
    {1, "circle", "pops::make_wall_predicate(circle)", "wall_radius > 0", ""},
};

namespace detail {

/// Shared parse loop: token -> enumerator index against a family table. Throws the ADC-584
/// refusal (family + requested token + valid set), NEVER returns a default.
template <std::size_t N>
inline int parse_route_index(const RouteInfo (&tbl)[N], RouteFamily family,
                             const std::string& token, const char* ctx) {
  for (const RouteInfo& r : tbl)
    if (token == r.token)
      return r.index;
  std::string valid;
  for (const RouteInfo& r : tbl) {
    if (!valid.empty())
      valid += '|';
    valid += r.token;
  }
  throw std::runtime_error(std::string(ctx) + ": unknown " + route_family_name(family) +
                           " route '" + token + "' (valid: " + valid +
                           "); typed routes never fall back to a default");
}

template <std::size_t N, class Id>
inline const RouteInfo& checked_route_info(const RouteInfo (&tbl)[N], Id id, RouteFamily family) {
  const int index = static_cast<int>(id);
  if (index < 0 || index >= static_cast<int>(N) || tbl[index].index != index)
    throw std::runtime_error(std::string("routes: unknown ") + route_family_name(family) +
                             " wire id " + std::to_string(index));
  return tbl[index];
}

}  // namespace detail

/// parse_*_route: wire token -> typed route ID (throws on unknown, never defaults).
/// route_token / route_info: typed route ID -> wire token / full row (the BOUNDED adapter --
/// every typed-ID-to-string conversion in the runtime goes through here, nowhere else).
inline RiemannRouteId parse_riemann_route(const std::string& token, const char* ctx = "routes") {
  return static_cast<RiemannRouteId>(
      detail::parse_route_index(kRiemannRoutes, RouteFamily::kRiemann, token, ctx));
}
inline const RouteInfo& route_info(RiemannRouteId id) {
  return detail::checked_route_info(kRiemannRoutes, id, RouteFamily::kRiemann);
}
inline const char* route_token(RiemannRouteId id) {
  return route_info(id).token;
}

inline LimiterRouteId parse_limiter_route(const std::string& token, const char* ctx = "routes") {
  return static_cast<LimiterRouteId>(
      detail::parse_route_index(kLimiterRoutes, RouteFamily::kLimiter, token, ctx));
}
inline const RouteInfo& route_info(LimiterRouteId id) {
  return detail::checked_route_info(kLimiterRoutes, id, RouteFamily::kLimiter);
}
inline const char* route_token(LimiterRouteId id) {
  return route_info(id).token;
}

inline ReconRouteId parse_recon_route(const std::string& token, const char* ctx = "routes") {
  return static_cast<ReconRouteId>(
      detail::parse_route_index(kReconRoutes, RouteFamily::kRecon, token, ctx));
}
inline const RouteInfo& route_info(ReconRouteId id) {
  return detail::checked_route_info(kReconRoutes, id, RouteFamily::kRecon);
}
inline const char* route_token(ReconRouteId id) {
  return route_info(id).token;
}

/// Time routes accept the historical alias spelling "ssprk2" (the pybind add_block seam exposed
/// it next to "explicit"; both mean the SSPRK2 explicit advance) and resolve it to the canonical
/// route -- parse-only compatibility, route_token always emits "explicit".
inline TimeRouteId parse_time_route(const std::string& token, const char* ctx = "routes") {
  if (token == "ssprk2")
    return TimeRouteId::kExplicitSsprk2;
  return static_cast<TimeRouteId>(
      detail::parse_route_index(kTimeRoutes, RouteFamily::kTime, token, ctx));
}
inline const RouteInfo& route_info(TimeRouteId id) {
  return detail::checked_route_info(kTimeRoutes, id, RouteFamily::kTime);
}
inline const char* route_token(TimeRouteId id) {
  return route_info(id).token;
}

inline SplittingRouteId parse_splitting_route(const std::string& token,
                                              const char* ctx = "routes") {
  return static_cast<SplittingRouteId>(
      detail::parse_route_index(kSplittingRoutes, RouteFamily::kSplitting, token, ctx));
}
inline const RouteInfo& route_info(SplittingRouteId id) {
  return detail::checked_route_info(kSplittingRoutes, id, RouteFamily::kSplitting);
}
inline const char* route_token(SplittingRouteId id) {
  return route_info(id).token;
}

inline FieldSolverRouteId parse_field_solver_route(const std::string& token,
                                                   const char* ctx = "routes") {
  return static_cast<FieldSolverRouteId>(
      detail::parse_route_index(kFieldSolverRoutes, RouteFamily::kFieldSolver, token, ctx));
}
inline const RouteInfo& route_info(FieldSolverRouteId id) {
  return detail::checked_route_info(kFieldSolverRoutes, id, RouteFamily::kFieldSolver);
}
inline const char* route_token(FieldSolverRouteId id) {
  return route_info(id).token;
}

inline PoissonBcRouteId parse_poisson_bc_route(const std::string& token,
                                               const char* ctx = "routes") {
  return static_cast<PoissonBcRouteId>(
      detail::parse_route_index(kPoissonBcRoutes, RouteFamily::kPoissonBc, token, ctx));
}
inline const RouteInfo& route_info(PoissonBcRouteId id) {
  return detail::checked_route_info(kPoissonBcRoutes, id, RouteFamily::kPoissonBc);
}
inline const char* route_token(PoissonBcRouteId id) {
  return route_info(id).token;
}

inline LayoutRouteId parse_layout_route(const std::string& token, const char* ctx = "routes") {
  return static_cast<LayoutRouteId>(
      detail::parse_route_index(kLayoutRoutes, RouteFamily::kLayout, token, ctx));
}
inline const RouteInfo& route_info(LayoutRouteId id) {
  return detail::checked_route_info(kLayoutRoutes, id, RouteFamily::kLayout);
}
inline const char* route_token(LayoutRouteId id) {
  return route_info(id).token;
}

inline TransportRouteId parse_transport_route(const std::string& token,
                                              const char* ctx = "routes") {
  return static_cast<TransportRouteId>(
      detail::parse_route_index(kTransportRoutes, RouteFamily::kTransport, token, ctx));
}
inline const RouteInfo& route_info(TransportRouteId id) {
  return detail::checked_route_info(kTransportRoutes, id, RouteFamily::kTransport);
}
inline const char* route_token(TransportRouteId id) {
  return route_info(id).token;
}

/// Source routes accept the historical alias spellings (kSources rows "lorentz" and
/// "potential_lorentz") and resolve them to the canonical route -- parse-only compatibility,
/// route_token always emits the canonical spelling.
inline SourceRouteId parse_source_route(const std::string& token, const char* ctx = "routes") {
  if (token == "lorentz")
    return SourceRouteId::kMagneticLorentz;
  if (token == "potential_lorentz")
    return SourceRouteId::kPotentialMagneticLorentz;
  return static_cast<SourceRouteId>(
      detail::parse_route_index(kSourceRoutes, RouteFamily::kSource, token, ctx));
}
inline const RouteInfo& route_info(SourceRouteId id) {
  return detail::checked_route_info(kSourceRoutes, id, RouteFamily::kSource);
}
inline const char* route_token(SourceRouteId id) {
  return route_info(id).token;
}

inline EllipticRouteId parse_elliptic_route(const std::string& token, const char* ctx = "routes") {
  return static_cast<EllipticRouteId>(
      detail::parse_route_index(kEllipticRoutes, RouteFamily::kElliptic, token, ctx));
}
inline const RouteInfo& route_info(EllipticRouteId id) {
  return detail::checked_route_info(kEllipticRoutes, id, RouteFamily::kElliptic);
}
inline const char* route_token(EllipticRouteId id) {
  return route_info(id).token;
}

inline SourceStageRouteId parse_source_stage_route(const std::string& token,
                                                   const char* ctx = "routes") {
  return static_cast<SourceStageRouteId>(
      detail::parse_route_index(kSourceStageRoutes, RouteFamily::kSourceStage, token, ctx));
}
inline const RouteInfo& route_info(SourceStageRouteId id) {
  return detail::checked_route_info(kSourceStageRoutes, id, RouteFamily::kSourceStage);
}
inline const char* route_token(SourceStageRouteId id) {
  return route_info(id).token;
}

inline PoissonRhsRouteId parse_poisson_rhs_route(const std::string& token,
                                                 const char* ctx = "routes") {
  return static_cast<PoissonRhsRouteId>(
      detail::parse_route_index(kPoissonRhsRoutes, RouteFamily::kPoissonRhs, token, ctx));
}
inline const RouteInfo& route_info(PoissonRhsRouteId id) {
  return detail::checked_route_info(kPoissonRhsRoutes, id, RouteFamily::kPoissonRhs);
}
inline const char* route_token(PoissonRhsRouteId id) {
  return route_info(id).token;
}

inline WallRouteId parse_wall_route(const std::string& token, const char* ctx = "routes") {
  return static_cast<WallRouteId>(
      detail::parse_route_index(kWallRoutes, RouteFamily::kWall, token, ctx));
}
inline const RouteInfo& route_info(WallRouteId id) {
  return detail::checked_route_info(kWallRoutes, id, RouteFamily::kWall);
}
inline const char* route_token(WallRouteId id) {
  return route_info(id).token;
}

/// Native route catalog version (ADC-599): bumped on any INCOMPATIBLE registry change (a removed
/// or re-tokenized route). MIRROR of ROUTE_REGISTRY_VERSION in python/pops/runtime/routes.py.
inline constexpr int kRouteRegistryVersion = 1;
/// Reserved tombstone shared by every route-id family. It is intentionally outside every enum so
/// exhaustive numerical switches stay warning-clean; deserializers must reject it and never reuse it.
inline constexpr int kReservedRouteWireId255 = 255;

/// Content-authenticated signature embedded in generated artifacts. FNV-1a-64 is applied to the
/// canonical row stream ``family US id US token US native_entry US requirements US limitations LF``
/// (US = byte 0x1f), in family/table order. Unlike the former family-count signature, this detects a
/// retokenized, reordered or semantically edited row even when every family keeps the same size.
/// MIRROR of routes.py::route_registry_signature().
inline std::string route_registry_signature() {
  std::uint64_t hash = UINT64_C(14695981039346656037);
  auto feed = [&hash](const std::string& value) {
    for (const unsigned char c : value) {
      hash ^= static_cast<std::uint64_t>(c);
      hash *= UINT64_C(1099511628211);
    }
  };
  auto add_family = [&feed](const char* family, const auto& table) {
    for (const RouteInfo& row : table) {
      feed(family);
      feed("\x1f");
      feed(std::to_string(row.index));
      feed("\x1f");
      feed(row.token);
      feed("\x1f");
      feed(row.native_entry);
      feed("\x1f");
      feed(row.requirements);
      feed("\x1f");
      feed(row.limitations);
      feed("\n");
    }
  };
  add_family("riemann", kRiemannRoutes);
  add_family("limiter", kLimiterRoutes);
  add_family("recon", kReconRoutes);
  add_family("time", kTimeRoutes);
  add_family("splitting", kSplittingRoutes);
  add_family("field_solver", kFieldSolverRoutes);
  add_family("poisson_bc", kPoissonBcRoutes);
  add_family("layout", kLayoutRoutes);
  add_family("transport", kTransportRoutes);
  add_family("source", kSourceRoutes);
  add_family("elliptic", kEllipticRoutes);
  add_family("source_stage", kSourceStageRoutes);
  add_family("poisson_rhs", kPoissonRhsRoutes);
  add_family("wall", kWallRoutes);
  std::ostringstream out;
  out << "v" << kRouteRegistryVersion << ':' << std::hex << std::setfill('0') << std::setw(16)
      << hash;
  return out.str();
}

/// Refuses a compiled artifact whose EMBEDDED route manifest differs from the current registry
/// (ADC-599: no silent reuse of a stale artifact). @p embedded is the artifact's
/// route_registry_signature() at build time. Strict artifacts must carry it: absence and every
/// content mismatch are rejected before any install mutation.
inline void verify_route_manifest(const std::string& embedded, const char* ctx) {
  if (embedded.empty())
    throw std::runtime_error(std::string(ctx) +
                             ": compiled artifact is missing the required route registry signature");
  const std::string current = route_registry_signature();
  if (embedded == current)
    return;
  throw std::runtime_error(std::string(ctx) +
                           ": stale compiled artifact -- route registry mismatch (built against '" +
                           embedded + "', current '" + current + "')");
}

// --- Non-drift locks --------------------------------------------------------------------------
// (1) Every table row's `index` equals its position (the enumerator value): a reordered or
// inserted row fails the build instead of silently remapping the enum.
namespace detail {
template <std::size_t N>
constexpr bool route_indices_sequential(const RouteInfo (&tbl)[N]) {
  for (std::size_t i = 0; i < N; ++i)
    if (tbl[i].index != static_cast<int>(i))
      return false;
  return true;
}
}  // namespace detail
static_assert(detail::route_indices_sequential(kRiemannRoutes), "riemann route index drift");
static_assert(detail::route_indices_sequential(kLimiterRoutes), "limiter route index drift");
static_assert(detail::route_indices_sequential(kReconRoutes), "recon route index drift");
static_assert(detail::route_indices_sequential(kTimeRoutes), "time route index drift");
static_assert(detail::route_indices_sequential(kSplittingRoutes), "splitting route index drift");
static_assert(detail::route_indices_sequential(kFieldSolverRoutes),
              "field_solver route index drift");
static_assert(detail::route_indices_sequential(kPoissonBcRoutes), "poisson_bc route index drift");
static_assert(detail::route_indices_sequential(kLayoutRoutes), "layout route index drift");
static_assert(detail::route_indices_sequential(kTransportRoutes), "transport route index drift");
static_assert(detail::route_indices_sequential(kSourceRoutes), "source route index drift");
static_assert(detail::route_indices_sequential(kEllipticRoutes), "elliptic route index drift");
static_assert(detail::route_indices_sequential(kSourceStageRoutes),
              "source_stage route index drift");
static_assert(detail::route_indices_sequential(kPoissonRhsRoutes), "poisson_rhs route index drift");
static_assert(detail::route_indices_sequential(kWallRoutes), "wall route index drift");

// (2) The route tables MIRROR the historical registries row for row (same size, same token at the
// same position). This TU sees BOTH tables, so the lock is compile-time -- the historical tables
// stay the single source of the tag lists; the route tables add the typed identity on top.
static_assert(sizeof(kRiemannRoutes) / sizeof(kRiemannRoutes[0]) ==
                  sizeof(kRiemanns) / sizeof(kRiemanns[0]),
              "route/registry drift: riemann row count");
static_assert(sizeof(kLimiterRoutes) / sizeof(kLimiterRoutes[0]) ==
                  sizeof(kLimiters) / sizeof(kLimiters[0]),
              "route/registry drift: limiter row count");
static_assert(sizeof(kTransportRoutes) / sizeof(kTransportRoutes[0]) ==
                  sizeof(kTransports) / sizeof(kTransports[0]),
              "route/registry drift: transport row count");
static_assert(sizeof(kEllipticRoutes) / sizeof(kEllipticRoutes[0]) ==
                  sizeof(kElliptics) / sizeof(kElliptics[0]),
              "route/registry drift: elliptic row count");
// kSources has 7 rows (2 aliases); the canonical route table has 5. The aliases are parse-only
// (see parse_source_route). 7 == 5 + 2 locked here so an added source updates BOTH tables.
static_assert(sizeof(kSourceRoutes) / sizeof(kSourceRoutes[0]) + 2 ==
                  sizeof(kSources) / sizeof(kSources[0]),
              "route/registry drift: source row count (canonical + 2 aliases)");

namespace detail {
constexpr bool route_tokens_match(const RouteInfo* routes, std::size_t n_routes,
                                  const RiemannTag* tags) {
  for (std::size_t i = 0; i < n_routes; ++i)
    if (!ct_str_eq(routes[i].token, tags[i].name))
      return false;
  return true;
}
constexpr bool route_tokens_match(const RouteInfo* routes, std::size_t n_routes,
                                  const LimiterTag* tags) {
  for (std::size_t i = 0; i < n_routes; ++i)
    if (!ct_str_eq(routes[i].token, tags[i].name))
      return false;
  return true;
}
constexpr bool route_tokens_match(const RouteInfo* routes, std::size_t n_routes,
                                  const TransportTag* tags) {
  for (std::size_t i = 0; i < n_routes; ++i)
    if (!ct_str_eq(routes[i].token, tags[i].name))
      return false;
  return true;
}
constexpr bool route_tokens_match(const RouteInfo* routes, std::size_t n_routes,
                                  const EllipticTag* tags) {
  for (std::size_t i = 0; i < n_routes; ++i)
    if (!ct_str_eq(routes[i].token, tags[i].name))
      return false;
  return true;
}
}  // namespace detail
static_assert(detail::route_tokens_match(kRiemannRoutes,
                                         sizeof(kRiemannRoutes) / sizeof(kRiemannRoutes[0]),
                                         kRiemanns),
              "route/registry drift: riemann tokens");
static_assert(detail::route_tokens_match(kLimiterRoutes, 4, kLimiters),
              "route/registry drift: limiter tokens");
static_assert(detail::route_tokens_match(kTransportRoutes, 3, kTransports),
              "route/registry drift: transport tokens");
static_assert(detail::route_tokens_match(kEllipticRoutes, 3, kElliptics),
              "route/registry drift: elliptic tokens");

}  // namespace pops
