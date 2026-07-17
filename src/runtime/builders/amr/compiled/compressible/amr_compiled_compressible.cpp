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
  // Parse the validated tag ONCE into the typed RiemannRouteId (ADC-641): the switch decodes it, the
  // euler_* fall-through mirrors dispatch_amr_compiled's fusion, and the default is the defense-in-depth
  // registry/dispatch guard (unreachable past validate_riemann).
  validate_riemann(riemann, /*polar=*/false, "add_compiled_model(AmrSystem)");
  validate_limiter(limiter, "add_compiled_model(AmrSystem)");
  switch (parse_riemann_route(riemann, "add_compiled_model(AmrSystem)")) {
    case RiemannRouteId::kRusanov:
      return build_amr_compiled_compressible_rusanov(spec, limiter, bp);
    case RiemannRouteId::kHll:
      return build_amr_compiled_compressible_hll(spec, limiter, bp);
    // hllc / euler_hllc share the leaf: on the true Euler brick the generic HLLCFlux and the explicit
    // EulerHLLCFlux2D are bit-identical (ADC-590). Same for roe / euler_roe.
    case RiemannRouteId::kHllc:
    case RiemannRouteId::kEulerHllc:
      return build_amr_compiled_compressible_hllc(spec, limiter, bp);
    case RiemannRouteId::kRoe:
    case RiemannRouteId::kEulerRoe:
      return build_amr_compiled_compressible_roe(spec, limiter, bp);
  }
  throw_registry_dispatch_mismatch("add_compiled_model(AmrSystem)", "flux", riemann);
}

}  // namespace pops::detail
