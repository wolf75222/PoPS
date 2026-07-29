#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/coupling/base/aux_fill.hpp>  // detail::derive_aux_bc + detail::fill_bz_box (shared)
#include <pops/coupling/base/elliptic_rhs.hpp>
#include <pops/numerics/elliptic/interface/elliptic_problem.hpp>
#include <pops/numerics/elliptic/interface/elliptic_solver.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/numerics/spatial_operator.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/prepared_provider_consensus.hpp>

#include <utility>

/// @file
/// @brief Spatial single-block hyperbolic-elliptic coupler (Poisson -> aux -> residual).
///
/// The caller places these spatial operations in its Program: (1) RHS f = elliptic_rhs(model, U);
/// (2) solve lap(phi) = f with the elliptic backend (warm start); (3) aux = (phi, grad phi) by
/// centered differences; (4) assemble the hyperbolic residual with this aux. For drift transport
/// aux enters through the FLUX (E x B); for a self-gravitating fluid through the SOURCE. The
/// residual keeps Limiter and NumericalFlux as independent spatial template axes. Compatible with
/// a SINGLE model; multi-species goes through the whole-system Program/runtime path. The detail::
/// helpers are at namespace scope (a POPS_HD extended lambda cannot live in a private method, an
/// nvcc restriction).

namespace pops {

namespace detail {
// Namespace-scope helpers: an extended __host__ __device__ lambda CANNOT be
// defined in a private/protected method (nvcc restriction), hence the
// extraction out of the Coupler class.

// Single-model compatibility: f = model.elliptic_rhs(U) on valid cells,
// delegated to a named assembler so this responsibility is not buried in Coupler.
/// Assemble the single-model elliptic RHS: rhs = model.elliptic_rhs(U) on valid cells
/// (delegated to SingleModelEllipticRhs). Shared by Coupler and AmrCouplerMP.
template <class Model>
inline void coupler_eval_rhs(const MultiFab& state, MultiFab& rhs, const Model& model) {
  SingleModelEllipticRhs<Model>{model}(state, rhs);
}

// aux = (phi, d phi/dx, d phi/dy) by centered differences. Delegates to the
// named FieldPostProcess convention with GradSign::Plus and store_phi=true: the
// coupler stores +grad phi (the physical sign E = -grad phi is carried by the
// transport drift velocity). Multiplicative form *cx / *cy kept identical
// -> bit-identical.
/// Set aux = (phi, d phi/dx, d phi/dy) by centered differences (factors cx, cy = 1/(2 dx),
/// 1/(2 dy)). Stores +grad phi (the physical sign E = -grad phi is carried by the drift velocity).
inline void coupler_grad_phi(const MultiFab& phi, MultiFab& aux, Real cx, Real cy) {
  field_postprocess(phi, aux, cx, cy, FieldPostProcess{FieldPostProcess::GradSign::Plus, true});
}
}  // namespace detail

/// Single-block hyperbolic-elliptic coupler. @tparam Model: PhysicalModel (flux, source,
/// elliptic_rhs, max_wave_speed, aux channel). @tparam Elliptic: elliptic backend (concept
/// EllipticSolver, default GeometricMG). Owns the aux and the solver, but never chooses a timestep,
/// a stage tableau, or a coupling cadence. The Program must place solve_fields() and
/// assemble_residual() explicitly. PRECONDITION: state carries at least Limiter::n_ghost ghosts.
template <class Model, class Elliptic = GeometricMG>
class Coupler {
  static_assert(EllipticSolver<Elliptic>, "the Coupler elliptic backend must model EllipticSolver");

 public:
  // active: optional "inside the conductor" predicate (embedded wall for
  // the Poisson solver). Empty => no internal wall.
  // bz: out-of-plane magnetic field B_z(x, y) PROVIDED by the user (constant or
  // field). Only has effect if the model declares the B_z aux component (aux_comps>3);
  // then fills aux component 3 once and for all (B_z static, external to the
  // elliptic solve: derive_aux does not touch it). Empty => no B_z. The aux channel is
  // allocated to the MODEL width: a base model (3) stays bit-identical.
  template <class FactoryT = DefaultEllipticFactory<Elliptic>>
    requires pops::EllipticFactory<FactoryT, Elliptic>
  Coupler(const Model& model, const Geometry& geom, const BoxArray& ba, const BCRec& bcU,
          const BCRec& bcPhi, ActiveRegionProvider2D active = {}, ScalarFieldProvider2D bz = {},
          FactoryT elliptic_factory = {})
      : Coupler(model, geom, ba, DistributionMapping(ba.size(), n_ranks()), bcU, bcPhi,
                std::move(active), std::move(bz), std::move(elliptic_factory)) {}

  /// Explicit-layout overload for externally load-balanced fields. The mapping is copied into the
  /// coupler and remains the single layout authority for aux storage and the elliptic backend.
  template <class FactoryT = DefaultEllipticFactory<Elliptic>>
    requires pops::EllipticFactory<FactoryT, Elliptic>
  Coupler(const Model& model, const Geometry& geom, const BoxArray& ba,
          const DistributionMapping& mapping, const BCRec& bcU, const BCRec& bcPhi,
          ActiveRegionProvider2D active = {}, ScalarFieldProvider2D bz = {},
          FactoryT elliptic_factory = {})
      : model_(model),
        geom_(geom),
        ba_(ba),
        dm_(mapping),
        bcU_(bcU),
        bcPhi_(bcPhi),
        aux_bc_(detail::derive_aux_bc(bcPhi)),
        mg_(make_elliptic_solver<Elliptic>(
            {geom_, ba_, dm_, bcPhi_, std::move(active), FieldDistribution::Distributed},
            std::move(elliptic_factory))),
        aux_(ba, dm_, aux_comps<Model>(), 1),
        bz_(std::move(bz)) {
    require_prepared_provider_collective_consensus(bz_);
    fill_bz();  // fills the B_z component (no-op if base model or empty bz)
  }

  /// Solve phi and derive aux = (phi, grad phi) for @p U without advancing in time. aux() is up to
  /// date on return. The Program decides at which logical point this operation runs.
  void solve_fields(const MultiFab& U) { update_aux(U); }

  /// Assemble the spatial finite-volume residual from @p state and the fields prepared by the most
  /// recent solve_fields() call. This method neither solves a field nor advances @p state: the
  /// Program explicitly owns their ordering and every temporal coefficient.
  template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux>
  void assemble_residual(MultiFab& state, MultiFab& residual) {
    fill_ghosts(state, geom_.domain, bcU_);
    assemble_rhs<Limiter, NumericalFlux>(model_, state, aux_, geom_, residual);
  }

  MultiFab& phi() { return mg_.phi(); }
  const MultiFab& aux() const { return aux_; }

 private:
  void update_aux(const MultiFab& state) {
    detail::coupler_eval_rhs(state, mg_.rhs(), model_);
    mg_.solve();  // EllipticSolver concept interface (backend-agnostic)
    derive_aux();
  }

  void derive_aux() {
    fill_ghosts(mg_.phi(), geom_.domain, bcPhi_);
    const Real cx = Real(1) / (2 * geom_.dx());
    const Real cy = Real(1) / (2 * geom_.dy());
    detail::coupler_grad_phi(mg_.phi(), aux_, cx, cy);
    fill_ghosts(aux_, geom_.domain, aux_bc_);
  }

  // Fills the B_z aux component (index kAuxBaseComps) on valid cells from
  // bz_(x, y), once only (B_z static). Compile-time guard: without a B_z field in the
  // model (aux_comps == 3) the component does not exist -> no code, no out-of-bound access.
  // The B_z halos are then maintained by derive_aux (Foextrap/periodic of aux_bc_,
  // cf. grad); field_postprocess only writes phi/grad (components 0..2), B_z is preserved.
  void fill_bz() {
    if constexpr (aux_comps<Model>() > kAuxBaseComps) {
      if (!bz_)
        return;
      for (int li = 0; li < aux_.local_size(); ++li)
        detail::fill_bz_box(aux_.fab(li), aux_.box(li), geom_, bz_);  // valid box
      fill_ghosts(aux_, geom_.domain, aux_bc_);  // B_z halos before the 1st solve
    }
  }

  Model model_;
  Geometry geom_;
  BoxArray ba_;
  DistributionMapping dm_;
  BCRec bcU_, bcPhi_, aux_bc_;
  Elliptic mg_;
  MultiFab aux_;
  ScalarFieldProvider2D bz_;  // prepared external B_z(x, y) (empty if not provided)
};

// The coupler elliptic backend honors the common contract: swapping
// GeometricMG for another conforming solver (FFT wrapper, PETSc) will only
// require changing the member type, not the coupling logic.
static_assert(EllipticSolver<GeometricMG>, "GeometricMG must model the EllipticSolver concept");

}  // namespace pops
