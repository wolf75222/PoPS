#pragma once

#include <pops/core/state/variables.hpp>  // VariableSet/VariableRole/role_from_name (resolve mask)
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>  // dispatch_amr_block + AmrBuildParams
#include <pops/runtime/amr/amr_runtime.hpp>                  // AmrRuntimeBlock spatial registry
#include <pops/runtime/builders/factory/model_factory.hpp>  // dispatch_model_for + compiled bricks + ModelSpec

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

/// @file
/// @brief Per-transport block-build seam for AmrSystem (ADC-335 / P0-B), mirror of block_seam.hpp.
///
/// The multi-block build resolves each ModelSpec behind a fixed-signature, hidden-visibility,
/// non-template free function per transport, so the full AMR dispatch product (all transports x flux x
/// limiter) is split across translation units and compiles in parallel. The type-erased
/// AmrRuntimeBlock boundary carries no template surface.

namespace pops::detail {

/// AMR partial-IMEX-mask resolution, moved out of amr_system.cpp's anonymous namespace (ADC-335) so the
/// per-transport seam TUs share one definition. Distinct from the System resolve_implicit_components
/// (model_factory.hpp): the AmrSystem error wording differs, kept VERBATIM here.
inline std::vector<int> resolve_implicit_components_amr(const std::string& block,
                                                        const VariableSet& cons,
                                                        const std::vector<std::string>& names,
                                                        const std::vector<std::string>& roles) {
  std::vector<int> out;
  auto push_unique = [&out](int c) {
    if (std::find(out.begin(), out.end(), c) == out.end())
      out.push_back(c);
  };
  for (const std::string& nm : names) {
    int idx = -1;
    for (int i = 0; i < static_cast<int>(cons.names.size()); ++i)
      if (cons.names[i] == nm) {
        idx = i;
        break;
      }
    if (idx < 0) {
      std::string have;
      for (std::size_t i = 0; i < cons.names.size(); ++i) {
        if (i)
          have += ", ";
        have += cons.names[i];
      }
      throw std::runtime_error("AmrSystem::add_block : implicit_vars : variable '" + nm +
                               "' missing from block '" + block +
                               "' (conserved variables : " + have + ")");
    }
    push_unique(idx);
  }
  for (const std::string& rn : roles) {
    const VariableRole role = role_from_name(rn);
    const int idx = cons.index_of(role);
    if (role == VariableRole::Custom || idx < 0)
      throw std::runtime_error("AmrSystem::add_block : implicit_roles : role '" + rn +
                               "' missing from block '" + block +
                               "' (the block does not provide this role)");
    push_unique(idx);
  }
  std::sort(out.begin(), out.end());
  return out;
}

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
  bool imex;
  int stride;
  std::vector<std::string> implicit_vars;
  std::vector<std::string> implicit_roles;
  NewtonOptions newton;
  const std::vector<double>* state;  // &BlockSpec::state when has_state, else nullptr (non-owning)
  bool newton_diagnostics;
  double pos_floor;
  double weno_epsilon;
  bool wave_speed_cache;
};

/// Build-multi visitor body with the transport pinned: resolve the partial IMEX mask against the
/// concrete model and produce the type-erased spatial AmrRuntimeBlock.
template <class TR>
AmrRuntimeBlock build_amr_block_for(TR tr, const AmrBlockBuildArgs& a, const SharedAmrLayout& S) {
  AmrRuntimeBlock out;
  dispatch_model_for(a.spec, std::move(tr), [&](auto m) {
    using M = decltype(m);
    const std::vector<int> impl_components =
        a.imex ? resolve_implicit_components_amr(a.name, M::conservative_vars(), a.implicit_vars,
                                                 a.implicit_roles)
               : std::vector<int>{};
    out = dispatch_amr_block(m, a.limiter, a.riemann, S, a.name, a.density, a.has_density, a.gamma,
                             a.substeps, a.recon_prim, a.imex, a.stride, impl_components, a.newton,
                             a.state, a.newton_diagnostics, a.pos_floor, a.weno_epsilon,
                             a.wave_speed_cache);
  });
  return out;
}

/// ADC-359 flux subdivision (compressible only): like build_amr_block_for, but the riemann dispatch is
/// supplied by @p dispatch (a flux-pinned detail::dispatch_amr_block_<flux>), so each per-flux compressible
/// seam TU instantiates ONE flux's build_amr_block leaves and they compile in parallel. The implicit
/// component resolution is IDENTICAL to build_amr_block_for; validate_riemann/limiter run once in the thin
/// dispatcher (python/amr_block_compressible.cpp), so the reachable leaf set stays the same.
template <class TR, class DispatchFn>
AmrRuntimeBlock build_amr_block_for_flux(TR tr, const AmrBlockBuildArgs& a,
                                         const SharedAmrLayout& S, DispatchFn dispatch) {
  AmrRuntimeBlock out;
  dispatch_model_for(a.spec, std::move(tr), [&](auto m) {
    using M = decltype(m);
    const std::vector<int> impl_components =
        a.imex ? resolve_implicit_components_amr(a.name, M::conservative_vars(), a.implicit_vars,
                                                 a.implicit_roles)
               : std::vector<int>{};
    out = dispatch(m, a, S, impl_components);
  });
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
}  // namespace pops::detail
