/// @file
/// @brief Model/state/aux access layer of the Cartesian spatial operator.
///
/// CONTRACT: how the spatial operator reads a model and its data.
///   - DiffusiveModel: optional concept (model.diffusivity() -> nu); the Fickian flux
///     F = -nu grad U is added to the hyperbolic flux when present (face_flux.hpp,
///     cartesian_operator.hpp).
///   - SourceFreeModel<M>: adapter that zeroes the source (explicit IMEX half-step).
///   - load_state<Model>: reads the conservative state from a ranked FieldView (POPS_HD).
///   - load_aux<NComp>: reads `AuxState<Dim>` (phi, Dim gradients, extras) from that authority.
///
/// This module carries no grid loop: every entry is POINTWISE (POPS_HD) or a compile-time
/// model adapter. It is the bottom of the spatial/ dependency DAG and depends only on the core
/// contracts plus the non-owning ranked field descriptor.

#pragma once

#include <pops/core/model/physical_model.hpp>  // aux_comps, HasPrimitiveVars: optional primitive reconstruction
#include <pops/core/state/state.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/state/variables.hpp>  // VariableSet: SourceFreeModel::conservative_vars forwarding
#include <pops/mesh/storage/field_view.hpp>

#include <concepts>

namespace pops {

// aux_comps<Model>() (aux channel width of a model) now lives in the contract header
// pops/core/physical_model.hpp (included above) so that CompositeModel can propagate it.

/// DiffusiveModel: optional concept for models with isotropic scalar diffusion.
///
/// A model satisfies DiffusiveModel if and only if m.diffusivity() -> Real (nu >= 0).
/// The Fickian flux F = -nu grad U is ADDED to the hyperbolic flux in assemble_rhs /
/// compute_face_fluxes. The divergence yields exactly +nu Lap(U).
/// INVARIANT: a model without diffusivity() does not change by a single bit (the hyperbolic
/// path is strictly unchanged -- the if constexpr is false, zero extra codegen).
template <class M>
concept DiffusiveModel = requires(const M m) {
  { m.diffusivity() } -> std::convertible_to<Real>;
};

/// SourceFreeModel<M>: adapter that cancels the source of M (explicit IMEX half-step).
///
/// Same flux and max_wave_speed as M, but source() always returns the zero state.
/// Used for the EXPLICIT half-step of an IMEX scheme (transport only, -div F); the stiff source
/// is treated implicitly by backward_euler_source. Does not expose diffusivity() so as not to
/// break non-diffusive models. Transparent to the HLL/HLLC contract: forwards pressure and
/// wave_speeds only if M exposes them (requires clause).
template <class M>
struct SourceFreeModel {
  using State = typename M::State;
  using Aux = typename M::Aux;
  static constexpr int n_vars = M::n_vars;
  static constexpr int n_aux =
      aux_comps_for<M, Aux::dimension>();  // transparent to the wrapped model's ranked width
  M m;
  template <class Providers>
  POPS_HD State flux(const State& u, const Providers& providers, int dir) const {
    return m.flux(u, providers, dir);
  }
  template <class Providers>
  POPS_HD Real max_wave_speed(const State& u, const Providers& providers, int dir) const {
    return m.max_wave_speed(u, providers, dir);
  }
  POPS_HD State source(const State&, const Aux&) const { return State{}; }
  POPS_HD Real elliptic_rhs(const State& u) const { return m.elliptic_rhs(u); }
  // SourceFreeModel does not expose the primitive variables: the explicit IMEX half-step that
  // uses it therefore reconstructs in conservative variables (the direct explicit path itself
  // has the composite model's conversions and can reconstruct in primitive variables).
  // Transparent to the HLL/HLLC contract: forwards pressure and signed wave speeds ONLY if M
  // exposes them (requires clause), so an IMEX half-step can stay on the HLLC flux.
  POPS_HD Real pressure(const State& u) const
    requires requires(const M& mm, const State& s) { mm.pressure(s); }
  {
    return m.pressure(u);
  }
  template <class Providers>
  POPS_HD void wave_speeds(const State& u, const Providers& providers, int dir, Real& smin,
                           Real& smax) const
    requires requires(const M& mm, const State& s, const Providers& p, int d, Real& lo, Real& hi) {
      mm.wave_speeds(s, p, d, lo, hi);
    }
  {
    m.wave_speeds(u, providers, dir, smin, smax);
  }
  // Roe / HLLC CAPABILITIES (HasRoeDissipation / HasHLLCStructure): forwarded ONLY if M exposes
  // them (requires clause), exactly like pressure / wave_speeds above and like composite.hpp.
  // WITHOUT these, route resolution rejects Roe/HLLC before an IMEX half-step can be built.
  POPS_HD Real contact_speed(const State& ul, const State& ur, Real pl, Real pr, Real sl, Real sr,
                             int dir) const
    requires requires(const M& mm, const State a_, const State b_, Real p, Real q, Real x, Real y,
                      int d) { mm.contact_speed(a_, b_, p, q, x, y, d); }
  {
    return m.contact_speed(ul, ur, pl, pr, sl, sr, dir);
  }
  POPS_HD State hllc_star_state(const State& u, Real p, Real s, Real sStar, int dir) const
    requires requires(const M& mm, const State a_, Real p_, Real s_, Real ss_, int d) {
      mm.hllc_star_state(a_, p_, s_, ss_, d);
    }
  {
    return m.hllc_star_state(u, p, s, sStar, dir);
  }
  template <class LeftProviders, class RightProviders>
  POPS_HD State roe_dissipation(const State& ul, const LeftProviders& left_providers,
                                const State& ur, const RightProviders& right_providers,
                                int dir) const
    requires requires(const M& mm, const State a_, const LeftProviders& x_, const State b_,
                      const RightProviders& y_, int d) { mm.roe_dissipation(a_, x_, b_, y_, d); }
  {
    return m.roe_dissipation(ul, left_providers, ur, right_providers, dir);
  }
  // Forward the VariableSet introspection (HOST): lets positivity_comp resolve the Density role
  // through the explicit IMEX half-step. Conditional (requires), like pressure / wave_speeds.
  static VariableSet conservative_vars()
    requires requires { M::conservative_vars(); }
  {
    return M::conservative_vars();
  }
};

/// Read Model::n_vars conservative components at one compile-time-ranked cell.
///
/// Returns a StateVec<n_vars> initialized from components 0..n_vars-1 of the channel.
/// POPS_HD, zero allocation. Does NOT read components beyond n_vars.
template <class Model, int Dim>
POPS_HD inline typename Model::State load_state(const FieldView<const Real, Dim>& field,
                                                const Index<Dim>& index) {
  typename Model::State u;
  for (int component = 0; component < Model::n_vars; ++component)
    u[component] = field(index, component);
  return u;
}

/// Read exactly `NComp` ranked auxiliary components at one cell.
///
/// The required prefix is `phi` plus one gradient per axis. Canonical and named extras are loaded
/// only when present in `NComp`; every offset comes from `AuxComponentLayout<Dim>`. The returned POD
/// therefore has the same spatial rank as the field view, including when several ranks are
/// instantiated in one compile-only proof.
template <int NComp, int Dim>
POPS_HD inline AuxState<Dim> load_aux(const FieldView<const Real, Dim>& field,
                                      const Index<Dim>& index) {
  using layout = AuxComponentLayout<Dim>;
  static_assert(NComp >= layout::base_components,
                "provider pack is missing required ranked base aux fields");
  static_assert(NComp <= layout::max_components,
                "provider pack exceeds ranked capacity; never clamp it");

  AuxState<Dim> result{};
  result.phi = field(index, layout::phi);
  for (int axis = 0; axis < Dim; ++axis)
    result.gradients[axis] = field(index, layout::gradient_begin + axis);
  if constexpr (NComp > layout::b_z)
    result.B_z = field(index, layout::b_z);
  if constexpr (NComp > layout::t_e)
    result.T_e = field(index, layout::t_e);
  if constexpr (NComp > layout::named_begin) {
    constexpr int named_count = NComp - layout::named_begin;
    for (int slot = 0; slot < named_count; ++slot)
      result.extra[slot] = field(index, layout::named_begin + slot);
  }
  return result;
}

}  // namespace pops
