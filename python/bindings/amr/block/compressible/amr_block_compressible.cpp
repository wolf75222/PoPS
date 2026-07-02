// ADC-335 (P0-B): multi-block AMR seam for the compressible (Euler 4-var) transport -- the heaviest AMR
// leaf (all fluxes + the SourceFreeModel IMEX doubling). ADC-359 flux subdivision: this TU is now the thin
// riemann dispatcher, routing to the per-flux build_amr_block_compressible_<flux> seam TUs (each compiles
// ONE flux's build_amr_block leaves in parallel). See amr_block_seam.hpp.
#include <pops/runtime/builders/block/amr_block_seam.hpp>

namespace pops::detail {

AmrRuntimeBlock build_amr_block_compressible(const AmrBlockBuildArgs& a, const SharedAmrLayout& S) {
  // Every flux is valid for Euler (no capability rejection here); an unknown flux is caught by the shared
  // validate_riemann + the registry throw, same wording as dispatch_amr_block.
  validate_riemann(a.riemann, /*polar=*/false, "add_block(AmrSystem, multi-block)");
  validate_limiter(a.limiter, "add_block(AmrSystem, multi-block)");
  if (a.riemann == "rusanov")
    return build_amr_block_compressible_rusanov(a, S);
  if (a.riemann == "hll")
    return build_amr_block_compressible_hll(a, S);
  // hllc / euler_hllc share the leaf: on the true Euler brick the generic HLLCFlux and the explicit
  // EulerHLLCFlux2D are bit-identical (ADC-590). Same for roe / euler_roe.
  if (a.riemann == "hllc" || a.riemann == "euler_hllc")
    return build_amr_block_compressible_hllc(a, S);
  if (a.riemann == "roe" || a.riemann == "euler_roe")
    return build_amr_block_compressible_roe(a, S);
  throw_registry_dispatch_mismatch("add_block(AmrSystem, multi-block)", "flux", a.riemann);
}

}  // namespace pops::detail
