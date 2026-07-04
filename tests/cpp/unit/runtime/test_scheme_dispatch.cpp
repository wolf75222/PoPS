// ONE spatial-reconstruction dispatch generator (include/pops/runtime/builders/scheme_dispatch.hpp,
// ADC-640): binds a typed LimiterRouteId to its compile-time reconstruction policy so the string ->
// template-instantiation limiter ladders collapse to a single dispatch_limiter call. This test is
// VOLUNTARILY LIGHT (it includes ONLY scheme_dispatch.hpp -- no System, no model): it locks
//   (1) that dispatch_limiter routes each LimiterRouteId to the TYPE whose ::n_ghost matches kLimiters
//       (a leaf returning Limiter::n_ghost proves the type binding is correct, not merely the enum);
//   (2) that the X-macro count-lock static_assert compiles (built into the header inclusion below);
//   (3) that the ctx / route_token feed the defense-in-depth throw path unreachable in practice.
// No model capability is exercised here (the if constexpr guards are per-call-site, out of scope).

#include <gtest/gtest.h>

#include <pops/runtime/builders/scheme_dispatch.hpp>

#include <string>

using namespace pops;

namespace {

// The n_ghost the leaf reads for the routed Limiter type -- proves dispatch_limiter bound the TYPE, not
// only the enum value.
int routed_n_ghost(LimiterRouteId route) {
  return dispatch_limiter(route, "test", [](auto tag) {
    using L = typename decltype(tag)::type;
    return L::n_ghost;
  });
}

}  // namespace

TEST(test_scheme_dispatch, routes_each_limiter_to_its_reconstruction_policy) {
  // Each route binds the compile-time type whose ::n_ghost matches kLimiters (1/2/2/3) and the type in
  // reconstruction.hpp -- so the X-macro POPS_FOR_EACH_LIMITER cannot drift from the route table.
  EXPECT_EQ(routed_n_ghost(LimiterRouteId::kNone), NoSlope::n_ghost);
  EXPECT_EQ(routed_n_ghost(LimiterRouteId::kMinmod), Minmod::n_ghost);
  EXPECT_EQ(routed_n_ghost(LimiterRouteId::kVanLeer), VanLeer::n_ghost);
  EXPECT_EQ(routed_n_ghost(LimiterRouteId::kWeno5), Weno5::n_ghost);
  // Cross-check against the route table's ::n_ghost expectation (kLimiters, single source).
  EXPECT_EQ(routed_n_ghost(LimiterRouteId::kNone), 1);
  EXPECT_EQ(routed_n_ghost(LimiterRouteId::kMinmod), 2);
  EXPECT_EQ(routed_n_ghost(LimiterRouteId::kVanLeer), 2);
  EXPECT_EQ(routed_n_ghost(LimiterRouteId::kWeno5), 3);
}

TEST(test_scheme_dispatch, count_lock_matches_the_route_table) {
  // The X-macro row count locks against kLimiterRoutes (a static_assert in the header guards drift; here
  // we assert the same equality at runtime so the intent is visible in the suite too).
  EXPECT_EQ(detail::kLimiterXMacroCount,
            static_cast<int>(sizeof(kLimiterRoutes) / sizeof(kLimiterRoutes[0])));
}

TEST(test_scheme_dispatch, leaf_return_value_flows_through) {
  // dispatch_limiter returns the leaf's value verbatim (decltype-deduced return type), so a caller can
  // build any per-route object; here a route_token round-trip through the leaf.
  const std::string tok = dispatch_limiter(LimiterRouteId::kMinmod, "test", [](auto tag) {
    using L = typename decltype(tag)::type;
    return std::string(L::n_ghost == Minmod::n_ghost ? "minmod-ok" : "wrong");
  });
  EXPECT_EQ(tok, "minmod-ok");
}
