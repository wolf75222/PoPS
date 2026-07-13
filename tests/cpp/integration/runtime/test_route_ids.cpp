// pops route_ids.hpp : TYPED native route IDs (ADC-584). One enum per behavior family, anchored to
// the single-source tag tables (dispatch_tags.hpp, model_registry.hpp). This test locks the PUBLIC
// contract of the light header WITHOUT Kokkos or MPI (strings + enums only):
//   (1) parse/token round-trip over EVERY row of EVERY k*Routes table (iterate by index) ;
//   (2) an unknown token is REFUSED with a message naming the family, the requested token, the valid
//       token set AND the phrase "never fall back to a default" (never a silent default) ;
//   (3) the historical alias spellings resolve to the canonical route while route_token emits the
//       canonical spelling ;
//   (4) route_info carries the native entry point / requirements / limitations columns ;
//   (5) the route tables MIRROR the historical registries (dispatch_tags / model_registry) row for
//       row ; the header static_asserts this too -- we assert it dynamically so the test documents
//       the contract ;
//   (6) every RouteFamily enumerator names itself (route_family_name never returns "?").

#include <gtest/gtest.h>

#include <pops/runtime/config/route_ids.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>

using namespace pops;

namespace {

bool contains(const std::string& hay, const char* needle) {
  return hay.find(needle) != std::string::npos;
}

/// Message d'un std::runtime_error leve par @p f (chaine VIDE si aucun runtime_error n'est leve : le
/// test de refus echoue alors, car "" ne contient aucun des fragments attendus).
template <class F>
static std::string throw_message(F&& f) {
  try {
    f();
  } catch (const std::runtime_error& e) {
    return e.what();
  } catch (...) {
    return "";
  }
  return "";
}

/// Round-trip sur une table ENTIERE (parcours par index) : parse(route_token(id)) == id pour
/// l'enumerateur de chaque ligne. @p parse enveloppe le parse_*_route libre de la famille.
template <class Id, std::size_t N, class ParseFn>
static bool table_round_trip(const RouteInfo (&tbl)[N], ParseFn parse) {
  for (const RouteInfo& r : tbl) {
    const Id id = static_cast<Id>(r.index);
    if (parse(route_token(id)) != id)
      return false;
  }
  return true;
}

/// Miroir de tokens : la table de routes et la table historique ont la meme taille et le meme nom a
/// chaque position (kRiemannRoutes[i].token == kRiemanns[i].name, ...). @p Tag = RiemannTag / ...
template <std::size_t NR, std::size_t NT, class Tag>
static bool tokens_mirror(const RouteInfo (&routes)[NR], const Tag (&tags)[NT]) {
  if (NR != NT)
    return false;
  for (std::size_t i = 0; i < NR; ++i)
    if (std::string(routes[i].token) != tags[i].name)
      return false;
  return true;
}

}  // namespace

TEST(RouteIds, RoundTripEveryRowOfEveryTable) {
  EXPECT_TRUE(table_round_trip<RiemannRouteId>(
      kRiemannRoutes, [](const std::string& t) { return parse_riemann_route(t); }))
      << "riemann round-trip";
  EXPECT_TRUE(table_round_trip<LimiterRouteId>(
      kLimiterRoutes, [](const std::string& t) { return parse_limiter_route(t); }))
      << "limiter round-trip";
  EXPECT_TRUE(table_round_trip<ReconRouteId>(
      kReconRoutes, [](const std::string& t) { return parse_recon_route(t); }))
      << "recon round-trip";
  EXPECT_TRUE(table_round_trip<TimeRouteId>(
      kTimeRoutes, [](const std::string& t) { return parse_time_route(t); }))
      << "time round-trip";
  EXPECT_TRUE(table_round_trip<FieldSolverRouteId>(
      kFieldSolverRoutes, [](const std::string& t) { return parse_field_solver_route(t); }))
      << "field_solver round-trip";
  EXPECT_TRUE(table_round_trip<PoissonBcRouteId>(
      kPoissonBcRoutes, [](const std::string& t) { return parse_poisson_bc_route(t); }))
      << "poisson_bc round-trip";
  EXPECT_TRUE(table_round_trip<LayoutRouteId>(
      kLayoutRoutes, [](const std::string& t) { return parse_layout_route(t); }))
      << "layout round-trip";
  EXPECT_TRUE(table_round_trip<TransportRouteId>(
      kTransportRoutes, [](const std::string& t) { return parse_transport_route(t); }))
      << "transport round-trip";
  EXPECT_TRUE(table_round_trip<SourceRouteId>(
      kSourceRoutes, [](const std::string& t) { return parse_source_route(t); }))
      << "source round-trip";
  EXPECT_TRUE(table_round_trip<EllipticRouteId>(
      kEllipticRoutes, [](const std::string& t) { return parse_elliptic_route(t); }))
      << "elliptic round-trip";
  EXPECT_TRUE(table_round_trip<PoissonRhsRouteId>(
      kPoissonRhsRoutes, [](const std::string& t) { return parse_poisson_rhs_route(t); }))
      << "poisson_rhs round-trip";
  EXPECT_TRUE(table_round_trip<WallRouteId>(
      kWallRoutes, [](const std::string& t) { return parse_wall_route(t); }))
      << "wall round-trip";
}

TEST(RouteIds, UnknownTokenRefusedWithFamilyTokenValidSetAndNoDefaultPhrase) {
  {
    const std::string m = throw_message([] { parse_riemann_route("upwind"); });
    EXPECT_TRUE(contains(m, "riemann") && contains(m, "upwind") &&
                contains(m, "rusanov|hll|hllc|roe|euler_hllc|euler_roe") &&
                contains(m, "never fall back to a default"))
        << "riemann 'upwind' refuse (famille, token, set valide, no-default)";
  }
  {
    const std::string m = throw_message([] { parse_time_route("rk4"); });
    EXPECT_TRUE(contains(m, "time") && contains(m, "rk4") &&
                contains(m, "explicit|ssprk3|euler|imex|imexrk_ars222") &&
                contains(m, "never fall back to a default"))
        << "time 'rk4' refuse (famille, token, set valide, no-default)";
  }
  {
    const std::string m = throw_message([] { parse_field_solver_route("amg"); });
    EXPECT_TRUE(contains(m, "field_solver") && contains(m, "amg") &&
                contains(m, "geometric_mg|fft|fft_spectral|polar") &&
                contains(m, "never fall back to a default"))
        << "field_solver 'amg' refuse (famille, token, set valide, no-default)";
  }
  {
    const std::string m = throw_message([] { parse_limiter_route("superbee"); });
    EXPECT_TRUE(contains(m, "limiter") && contains(m, "superbee") &&
                contains(m, "none|minmod|vanleer|weno5") &&
                contains(m, "never fall back to a default"))
        << "limiter 'superbee' refuse (famille, token, set valide, no-default)";
  }
  {
    const std::string m = throw_message([] { parse_transport_route("upwind"); });
    EXPECT_TRUE(contains(m, "transport") && contains(m, "upwind") &&
                contains(m, "exb|compressible|isothermal") &&
                contains(m, "never fall back to a default"))
        << "transport 'upwind' refuse (famille, token, set valide, no-default)";
  }
}

TEST(RouteIds, HistoricalAliasesResolveToCanonicalRoute) {
  EXPECT_TRUE(parse_source_route("lorentz") == SourceRouteId::kMagneticLorentz)
      << "alias 'lorentz' -> kMagneticLorentz";
  EXPECT_TRUE(parse_source_route("potential_lorentz") == SourceRouteId::kPotentialMagneticLorentz)
      << "alias 'potential_lorentz' -> kPotentialMagneticLorentz";
  EXPECT_TRUE(parse_time_route("ssprk2") == TimeRouteId::kExplicitSsprk2)
      << "alias 'ssprk2' -> kExplicitSsprk2";
  EXPECT_TRUE(std::string(route_token(SourceRouteId::kMagneticLorentz)) == "magnetic")
      << "route_token(kMagneticLorentz) == 'magnetic' (canonique)";
  EXPECT_TRUE(std::string(route_token(SourceRouteId::kPotentialMagneticLorentz)) ==
              "potential_magnetic")
      << "route_token(kPotentialMagneticLorentz) == 'potential_magnetic' (canonique)";
  EXPECT_TRUE(std::string(route_token(TimeRouteId::kExplicitSsprk2)) == "explicit")
      << "route_token(kExplicitSsprk2) == 'explicit' (canonique)";
}

TEST(RouteIds, UnknownAndReservedNumericIdsAreRefused) {
  const std::string unknown =
      throw_message([] { (void)route_info(static_cast<TimeRouteId>(99)); });
  EXPECT_TRUE(contains(unknown, "time") && contains(unknown, "99"));
  const std::string reserved =
      throw_message([] { (void)route_info(static_cast<TimeRouteId>(kReservedRouteWireId255)); });
  EXPECT_TRUE(contains(reserved, "time") && contains(reserved, "255"));
}

TEST(RouteIds, RegistrySignatureAuthenticatesFullContent) {
  const std::string signature = route_registry_signature();
  EXPECT_TRUE(signature.rfind("v2:", 0) == 0) << signature;
  EXPECT_TRUE(signature.size() == 67) << "v2: plus complete sha256 catalog digest";
  EXPECT_TRUE(throw_message([&] { verify_route_manifest("", "test"); }).find("missing") !=
              std::string::npos);
  EXPECT_TRUE(throw_message([&] { verify_route_manifest("v2:0000000000000000000000000000000000000000000000000000000000000000", "test"); })
                  .find("mismatch") != std::string::npos);
  EXPECT_NO_THROW(verify_route_manifest(signature, "test"));
}

TEST(RouteIds, RouteInfoCarriesNativeEntryRequirementsAndLimitations) {
  EXPECT_TRUE(std::string(route_info(RiemannRouteId::kHllc).native_entry) == "pops::HLLCFlux" &&
              contains(route_info(RiemannRouteId::kHllc).requirements, "pressure"))
      << "route_info(kHllc) : native 'pops::HLLCFlux', requirements contient 'pressure'";
  // ADC-590 explicit canonical Euler routes: round-trip + native entry + euler_2d_layout marker.
  EXPECT_TRUE(parse_riemann_route("euler_hllc") == RiemannRouteId::kEulerHllc &&
              std::string(route_token(RiemannRouteId::kEulerHllc)) == "euler_hllc" &&
              std::string(route_info(RiemannRouteId::kEulerHllc).native_entry) ==
                  "pops::EulerHLLCFlux2D" &&
              contains(route_info(RiemannRouteId::kEulerHllc).requirements, "euler_2d_layout"))
      << "route euler_hllc : round-trip + native 'pops::EulerHLLCFlux2D' + euler_2d_layout";
  EXPECT_TRUE(parse_riemann_route("euler_roe") == RiemannRouteId::kEulerRoe &&
              std::string(route_info(RiemannRouteId::kEulerRoe).native_entry) ==
                  "pops::EulerRoeFlux2D")
      << "route euler_roe : round-trip + native 'pops::EulerRoeFlux2D'";
  EXPECT_TRUE(std::string(route_info(TimeRouteId::kSsprk3).limitations).size() > 0 &&
              contains(route_info(TimeRouteId::kSsprk3).limitations, "aot"))
      << "route_info(kSsprk3) : limitations non vide (mentionne 'aot')";
}

TEST(RouteIds, RouteTokensMirrorRegistryTagNamesInOrder) {
  EXPECT_TRUE(tokens_mirror(kRiemannRoutes, kRiemanns))
      << "kRiemannRoutes tokens == kRiemanns names (ordre)";
  EXPECT_TRUE(tokens_mirror(kLimiterRoutes, kLimiters))
      << "kLimiterRoutes tokens == kLimiters names (ordre)";
  EXPECT_TRUE(tokens_mirror(kTransportRoutes, kTransports))
      << "kTransportRoutes tokens == kTransports names (ordre)";
  EXPECT_TRUE(tokens_mirror(kEllipticRoutes, kElliptics))
      << "kEllipticRoutes tokens == kElliptics names (ordre)";
}

TEST(RouteIds, EveryRouteFamilyEnumeratorNamesItself) {
  const RouteFamily fams[] = {
      RouteFamily::kLimiter,    RouteFamily::kRiemann,   RouteFamily::kRecon,
      RouteFamily::kTime,       RouteFamily::kFieldSolver,
      RouteFamily::kPoissonBc,  RouteFamily::kLayout,    RouteFamily::kTransport,
      RouteFamily::kSource,     RouteFamily::kElliptic,
      RouteFamily::kPoissonRhs, RouteFamily::kWall,
  };
  bool ok = true;
  for (RouteFamily f : fams)
    ok = ok && (std::string(route_family_name(f)) != "?");
  EXPECT_TRUE(ok) << "chaque RouteFamily -> nom non-'?'";
}
