#pragma once

#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>  // dispatch_amr_block + AmrBuildParams
#include <pops/runtime/amr/amr_runtime.hpp>                  // AmrRuntimeBlock spatial registry
#include <pops/runtime/builders/factory/model_factory.hpp>  // dispatch_model_for + compiled bricks + ModelSpec

#include <string>
#include <utility>
#include <vector>

/// @file
/// @brief Per-transport block-build seam for the AMR runtime (ADC-335 / P0-B).
///
/// The multi-block build resolves each ModelSpec behind a fixed-signature, hidden-visibility,
/// non-template free function per transport, so the full AMR dispatch product (all transports x flux x
/// limiter) is split across translation units and compiles in parallel. The type-erased
/// AmrRuntimeBlock boundary carries no template surface.

namespace pops::detail {

/// Non-model inputs of a MULTI-block AMR build (the fields the build_multi visitor read off the BlockSpec).
/// Held by value (cheap setup-time copies); @p state points at the BlockSpec's owned vector (non-owning).
struct AmrBlockBuildArgs {
  ModelSpec spec;
  std::string name;
  std::string limiter;
  std::string riemann;
  std::vector<double> density;
  bool has_density;
  double gamma;
  int substeps;
  bool recon_prim;
  int stride;
  const std::vector<double>* state;  // &BlockSpec::state when has_state, else nullptr (non-owning)
  double pos_floor;
  double weno_epsilon;
  bool wave_speed_cache;
};

/// Build-multi visitor body with the transport pinned: produce the type-erased spatial
/// AmrRuntimeBlock. Time integration and implicit solves are exclusively Program-owned.
template <class TR>
AmrRuntimeBlock build_amr_block_for(TR tr, const AmrBlockBuildArgs& a, const SharedAmrLayout& S) {
  AmrRuntimeBlock out;
  dispatch_model_for(a.spec, std::move(tr), [&](auto m) {
    out = dispatch_amr_block(m, a.limiter, a.riemann, S, a.name, a.density, a.has_density, a.gamma,
                             a.substeps, a.recon_prim, a.stride, a.state, a.pos_floor,
                             a.weno_epsilon, a.wave_speed_cache);
  });
  return out;
}

/// ADC-359 flux subdivision (compressible only): like build_amr_block_for, but the riemann dispatch is
/// supplied by @p dispatch (a flux-pinned detail::dispatch_amr_block_<flux>), so each per-flux compressible
/// seam TU instantiates ONE flux's build_amr_block leaves and they compile in parallel.
/// validate_riemann/limiter run once in the thin dispatcher (amr_block_compressible.cpp), so the
/// reachable leaf set stays the same.
template <class TR, class DispatchFn>
AmrRuntimeBlock build_amr_block_for_flux(TR tr, const AmrBlockBuildArgs& a,
                                         const SharedAmrLayout& S, DispatchFn dispatch) {
  AmrRuntimeBlock out;
  dispatch_model_for(a.spec, std::move(tr), [&](auto m) { out = dispatch(m, a, S); });
  return out;
}

// Per-transport seam functions (defined in amr/block/**).
// TR construction matches dispatch_transport VERBATIM
// (ExBVelocity{B0}/CompressibleFlux{gamma}/IsothermalFlux{cs2, vacuum_floor}).
AmrRuntimeBlock build_amr_block_exb(const AmrBlockBuildArgs& a, const SharedAmrLayout& S);
AmrRuntimeBlock build_amr_block_isothermal(const AmrBlockBuildArgs& a, const SharedAmrLayout& S);
AmrRuntimeBlock build_amr_block_compressible(const AmrBlockBuildArgs& a, const SharedAmrLayout& S);

// ADC-359 per-flux compressible seam leaves: each is defined in its own .cpp with
// build_amr_block_for_flux pinned to one flux, so they compile in parallel.
AmrRuntimeBlock build_amr_block_compressible_rusanov(const AmrBlockBuildArgs& a,
                                                     const SharedAmrLayout& S);
AmrRuntimeBlock build_amr_block_compressible_hll(const AmrBlockBuildArgs& a,
                                                 const SharedAmrLayout& S);
AmrRuntimeBlock build_amr_block_compressible_hllc(const AmrBlockBuildArgs& a,
                                                  const SharedAmrLayout& S);
AmrRuntimeBlock build_amr_block_compressible_roe(const AmrBlockBuildArgs& a,
                                                 const SharedAmrLayout& S);
AmrRuntimeBlock build_amr_block_compressible_roe_hll_rusanov_recovery(const AmrBlockBuildArgs& a,
                                                                      const SharedAmrLayout& S);
}  // namespace pops::detail
