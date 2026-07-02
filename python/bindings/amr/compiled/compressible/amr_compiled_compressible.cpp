// ADC-335 (P0-B): single-block AMR seam (AmrCouplerMP) for the compressible (Euler) transport. ADC-359
// flux subdivision: this TU is now the thin riemann dispatcher, routing to the per-flux
// build_amr_compiled_compressible_<flux> seam TUs (each compiles ONE flux's build_amr_compiled leaves in
// parallel). See amr_block_seam.hpp.
#include <pops/runtime/builders/block/amr_block_seam.hpp>

namespace pops::detail {

AmrCompiledHooks build_amr_compiled_compressible(const ModelSpec& spec, const std::string& limiter,
                                                 const std::string& riemann,
                                                 const AmrBuildParams& bp) {
  // Every flux is valid for Euler (no capability rejection here); an unknown flux is caught by the shared
  // validate_riemann + the registry throw, same wording as dispatch_amr_compiled.
  validate_riemann(riemann, /*polar=*/false, "add_compiled_model(AmrSystem)");
  validate_limiter(limiter, "add_compiled_model(AmrSystem)");
  if (riemann == "rusanov")
    return build_amr_compiled_compressible_rusanov(spec, limiter, bp);
  if (riemann == "hll")
    return build_amr_compiled_compressible_hll(spec, limiter, bp);
  // hllc / euler_hllc share the leaf: on the true Euler brick the generic HLLCFlux and the explicit
  // EulerHLLCFlux2D are bit-identical (ADC-590). Same for roe / euler_roe.
  if (riemann == "hllc" || riemann == "euler_hllc")
    return build_amr_compiled_compressible_hllc(spec, limiter, bp);
  if (riemann == "roe" || riemann == "euler_roe")
    return build_amr_compiled_compressible_roe(spec, limiter, bp);
  throw_registry_dispatch_mismatch("add_compiled_model(AmrSystem)", "flux", riemann);
}

}  // namespace pops::detail
