#pragma once

#include <pops/core/foundation/cold.hpp>          // POPS_COLD_FN
#include <pops/numerics/fv/reconstruction.hpp>    // NoSlope / Minmod / VanLeer / Weno5
#include <pops/runtime/config/dispatch_tags.hpp>  // throw_registry_dispatch_mismatch
#include <pops/runtime/config/route_ids.hpp>      // LimiterRouteId, route_token, kLimiterRoutes

#include <string>
#include <type_traits>  // std::type_identity

/// @file
/// @brief ONE spatial-reconstruction dispatch generator (ADC-640): binds a typed LimiterRouteId to its
///        compile-time Limiter policy so every string->template-instantiation limiter ladder collapses to
///        a single dispatch_limiter call + a leaf.
///
/// This header is the ONLY place allowed to bind a LimiterRouteId to its compile-time reconstruction
/// TYPE -- the one fact route_ids.hpp / dispatch_tags.hpp deliberately cannot hold (they are string/enum
/// only, no numerics dependency; the limiter policies live in the heavy numerics/fv/reconstruction.hpp).
/// It sits at builders/ level because it is consumed by builders/block/, builders/compiled/ and program/.
///
/// Why an X-macro (POPS_FOR_EACH_LIMITER) plus a count-lock static_assert, not a switch or a table:
///   - a table of function pointers cannot carry a TYPE (build_block<Limiter, Flux> needs Limiter as a
///     template argument, resolved per TU per flux) without type-erasing the closure (defeats inlining,
///     adds a heap allocation on a device-adjacent path) or pre-enumerating the whole Limiter x Flux x
///     Model product in one TU (the ~1700-leaf blow-up seam_combinations.cmake exists to avoid);
///   - a hand-written switch on LimiterRouteId gives exhaustiveness only via -Wswitch -- but this repo
///     compiles WITHOUT -Werror (cmake/PopsDevTooling.cmake: "Informatif d'abord : PAS de -Werror"), so a
///     missing arm is a silent warning, not a build failure;
///   - the X-macro + count-lock static_assert IS a hard build error: adding a LimiterRouteId enumerator +
///     a kLimiterRoutes row without adding the X-macro row fails the static_assert below at compile time.
/// No RTTI, no virtual, no std::function; all host cold-path -- Kokkos-clean.

namespace pops {

/// SINGLE binding of each typed limiter route to its compile-time reconstruction policy -- the one fact
/// route_ids.hpp (string/enum only) cannot hold. Adding a limiter = one row here + one kLimiterRoutes row
/// + the type in reconstruction.hpp. The static_assert locks this list against the route table, so a
/// forgotten row is a BUILD ERROR (this repo compiles without -Werror, so -Wswitch alone would only warn).
#define POPS_FOR_EACH_LIMITER(X) \
  X(kNone, NoSlope)              \
  X(kMinmod, Minmod)             \
  X(kVanLeer, VanLeer)           \
  X(kWeno5, Weno5)

namespace detail {
constexpr int kLimiterXMacroCount = 0
#define POPS_LIM_COUNT(Id, T) +1
    POPS_FOR_EACH_LIMITER(POPS_LIM_COUNT)
#undef POPS_LIM_COUNT
    ;
}  // namespace detail
static_assert(detail::kLimiterXMacroCount ==
                  static_cast<int>(sizeof(kLimiterRoutes) / sizeof(kLimiterRoutes[0])),
              "POPS_FOR_EACH_LIMITER drifted from kLimiterRoutes (add the X-macro row)");

/// ONE spatial-reconstruction dispatch generator (ADC-640). Routes a typed LimiterRouteId to its
/// compile-time Limiter policy and invokes @p leaf with std::type_identity<Limiter>{}. The leaf (a generic
/// lambda) closes over the Flux type and every build argument, so the SAME generator serves System
/// (build_block<L, Flux>), polar (build_block_polar<L, Flux>), AMR multi-block (build_amr_block<Model, L,
/// Flux>) and AMR compiled (build_amr_compiled<Model, L, Flux>). Every arm instantiates the leaf, so the
/// reachable build_block<Limiter, Flux, ...> set is IDENTICAL to the hand-written ladders. @p route is
/// produced by parse_limiter_route at the boundary, AFTER validate_limiter, so the trailing throw is the
/// historical defense-in-depth guard (unreachable in practice). Host cold-path only: the device kernels
/// remain the named functors inside the build_* factory the leaf calls.
template <class Leaf>
POPS_COLD_FN auto dispatch_limiter(LimiterRouteId route, const char* ctx, Leaf&& leaf)
    -> decltype(leaf(std::type_identity<NoSlope>{})) {
  switch (route) {
#define POPS_LIM_CASE(Id, T) \
  case LimiterRouteId::Id:   \
    return leaf(std::type_identity<T>{});
    POPS_FOR_EACH_LIMITER(POPS_LIM_CASE)
#undef POPS_LIM_CASE
  }
  throw_registry_dispatch_mismatch(ctx, "limiter", std::string(route_token(route)));
}

}  // namespace pops
