// ADC-335 (P0-B): multi-block AMR seam for the compressible (Euler 4-var) transport -- the heaviest AMR
// leaf (all fluxes + the SourceFreeModel IMEX doubling). ADC-359 flux subdivision: this TU is now the thin
// riemann dispatcher, routing to the per-flux build_amr_block_compressible_<flux> seam TUs (each compiles
// ONE flux's build_amr_block leaves in parallel). See amr_block_seam.hpp.
#include <pops/runtime/builders/block/amr_block_seam.hpp>

namespace pops::detail {

AmrRuntimeBlock build_amr_block_compressible(const AmrBlockBuildArgs& a, const SharedAmrLayout& S) {
  // Every flux is valid for Euler (no capability rejection here); an unknown flux is caught by the shared
  // validate_riemann + the registry throw, same wording as dispatch_amr_block.
  // Parse the validated tag ONCE into the typed RiemannRouteId (ADC-641). One public provider maps
  // to one seam leaf; the default is the defense-in-depth registry/dispatch guard.
  validate_riemann(a.riemann, /*polar=*/false, "add_block(AmrSystem, multi-block)");
  validate_limiter(a.limiter, "add_block(AmrSystem, multi-block)");
  switch (parse_riemann_route(a.riemann, "add_block(AmrSystem, multi-block)")) {
    case RiemannRouteId::kRusanov:
      return build_amr_block_compressible_rusanov(a, S);
    case RiemannRouteId::kHll:
      return build_amr_block_compressible_hll(a, S);
    case RiemannRouteId::kHllc:
      return build_amr_block_compressible_hllc(a, S);
    case RiemannRouteId::kRoe:
      return build_amr_block_compressible_roe(a, S);
  }
  throw_registry_dispatch_mismatch("add_block(AmrSystem, multi-block)", "flux", a.riemann);
}

}  // namespace pops::detail
