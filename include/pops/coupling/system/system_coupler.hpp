#pragma once

#include <pops/core/model/coupled_system.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/coupling/base/aux_fill.hpp>  // detail::derive_aux_bc + detail::fill_bz_box (shared)
#include <pops/coupling/base/elliptic_rhs.hpp>
#include <pops/numerics/elliptic/interface/elliptic_problem.hpp>
#include <pops/numerics/elliptic/interface/elliptic_solver.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/spatial_operator.hpp>
#include <pops/parallel/prepared_provider_consensus.hpp>

#include <stdexcept>
#include <type_traits>
#include <utility>

/// @file
/// @brief Single-level multi-species field and residual assembler.
///
/// SystemAssembler assembles the system RHS (f = Sum_s q_s n_s), Poisson field, shared aux channel
/// (phi, grad phi), and block residual R = -div F + S. It owns no scheme, cadence, clock, or time
/// step. Production temporal composition belongs exclusively to the installed ProgramGraph.

namespace pops {

namespace detail {
template <class Block>
struct ScopedBlockState {
  Block& block;
  MultiFab* old_state;

  ScopedBlockState(Block& b, MultiFab& stage_state) : block(b), old_state(b.state) {
    block.state = &stage_state;
  }

  // RULE OF FIVE (C.21): scope-guard with a side effect in the dtor (restores block.state). Copy/move
  // BY DEFAULT -> double restoration or restoration from a dead copy. Never copied nor moved
  // (always a block-scoped local variable): delete the four operations.
  ScopedBlockState(const ScopedBlockState&) = delete;
  ScopedBlockState& operator=(const ScopedBlockState&) = delete;
  ScopedBlockState(ScopedBlockState&&) = delete;
  ScopedBlockState& operator=(ScopedBlockState&&) = delete;

  ~ScopedBlockState() { block.state = old_state; }
};

template <CoupledSystemLike System>
DistributionMapping system_layout_mapping_or_throw(System& system, const BoxArray& boxes) {
  DistributionMapping mapping;
  bool initialized = false;
  system.for_each_block([&](const auto& block) {
    const MultiFab& state = block.U();
    if (state.box_array().boxes() != boxes.boxes())
      throw std::invalid_argument(
          "SystemAssembler state BoxArray disagrees with its authored solver layout");
    if (!initialized) {
      mapping = state.dmap();
      initialized = true;
    } else if (state.dmap().ranks() != mapping.ranks()) {
      throw std::invalid_argument(
          "SystemAssembler blocks must share one exact DistributionMapping");
    }
  });
  if (!initialized)
    throw std::invalid_argument("SystemAssembler requires at least one state-bearing block");
  return mapping;
}
}  // namespace detail

// === ASSEMBLER: fields (system Poisson + aux) + block residual. No stepping. ======
/// ASSEMBLES the fields (system Poisson + shared aux) and a block residual evaluator. No time
/// stepping. @tparam System: CoupledSystem. @tparam RhsAssembler: Poisson RHS assembler.
/// @tparam Elliptic: elliptic backend (EllipticSolver concept, default GeometricMG).
template <CoupledSystemLike System, class RhsAssembler, class Elliptic = GeometricMG>
class SystemAssembler {
  static_assert(EllipticSolver<Elliptic>, "the elliptic backend must model EllipticSolver");

 public:
  // bz: out-of-plane magnetic field B_z(x, y) supplied by the user (constant or field),
  // shared by ALL blocks. The SHARED aux channel is allocated at the MAXIMUM width requested
  // by the blocks (aux_comps): a block reading B_z (n_aux=4) sees it, a base block (3)
  // ignores the component. Without an extra-field block the width stays 3 -> allocation and numerics
  // strictly bit-identical to history.
  template <class FactoryT = DefaultEllipticFactory<Elliptic>>
    requires pops::EllipticFactory<FactoryT, Elliptic>
  SystemAssembler(System system, const Geometry& geom, const BoxArray& ba, const BCRec& bcPhi,
                  RhsAssembler rhs_assembler, ActiveRegionProvider2D active = {},
                  ScalarFieldProvider2D bz = {}, FactoryT elliptic_factory = {})
      : system_(std::move(system)),
        rhs_assembler_(std::move(rhs_assembler)),
        geom_(geom),
        ba_(ba),
        dm_(detail::system_layout_mapping_or_throw(system_, ba_)),
        bcPhi_(bcPhi),
        aux_bc_(detail::derive_aux_bc(bcPhi)),
        mg_(make_elliptic_solver<Elliptic>(
            {geom_, ba_, dm_, bcPhi_, std::move(active), FieldDistribution::Distributed},
            std::move(elliptic_factory))),
        aux_ncomp_(system_aux_comps(system_)),
        aux_(ba, dm_, aux_ncomp_, 1),
        bz_(std::move(bz)) {
    require_prepared_provider_collective_consensus(bz_);
    fill_bz();  // populates B_z (no-op if no block requests it or if bz is empty)
  }

  System& system() { return system_; }
  const System& system() const { return system_; }
  MultiFab& phi() { return mg_.phi(); }
  MultiFab& aux() { return aux_; }
  const MultiFab& aux() const { return aux_; }
  const Geometry& geom() const { return geom_; }
  const BoxArray& ba() const { return ba_; }
  const DistributionMapping& dm() const { return dm_; }

  /// Solves the system RHS (Sum_s q_s n_s), the Poisson, then derives aux = (phi, grad phi). aux()
  /// is up to date on return.
  void solve_fields() {
    rhs_assembler_(system_, mg_.rhs());
    mg_.solve();
    derive_aux();
  }

  /// Residual R = -div F + S of a block at a stage (with field re-solve if @p recompute_aux).
  /// This is the spatial method-of-lines operation consumed by a Program stage. Fills the ghosts of
  /// @p state per block.bc before assembly.
  template <class Limiter, class NumericalFlux, class Block>
  void block_residual(Block& block, MultiFab& state, MultiFab& R, bool recompute_aux) {
    if (recompute_aux) {
      detail::ScopedBlockState<Block> swap(block, state);
      solve_fields();
    }
    fill_ghosts(state, geom_.domain, block.bc);
    assemble_rhs<Limiter, NumericalFlux>(block.model, state, aux_, geom_, R);
  }

 private:
  void derive_aux() {
    fill_ghosts(mg_.phi(), geom_.domain, bcPhi_);
    const Real cx = Real(1) / (2 * geom_.dx());
    const Real cy = Real(1) / (2 * geom_.dy());
    field_postprocess(mg_.phi(), aux_, cx, cy,
                      FieldPostProcess{FieldPostProcess::GradSign::Plus, true});
    fill_ghosts(aux_, geom_.domain, aux_bc_);
  }

  // Width of the SHARED aux channel: maximum of aux_comps<Model> over all blocks (at least
  // kAuxBaseComps). The shared channel must be at least as wide as the most demanding block
  // so that load_aux<aux_comps<Model>> never reads out of bounds; a less demanding block
  // simply ignores the extra components.
  static int system_aux_comps(const System& sys) {
    int w = kAuxBaseComps;
    sys.for_each_block([&](const auto& b) {
      using Model = std::decay_t<decltype(b.model)>;
      const int c = aux_comps<Model>();
      if (c > w)
        w = c;
    });
    return w;
  }

  // Populates the aux B_z component (index kAuxBaseComps) of the shared channel from bz_(x, y), a
  // single time (static B_z). No-op if no block declares B_z (width 3) or if bz_ is empty:
  // RUNTIME guard on aux_ncomp_ (the width is only known at construction). Halos are then
  // maintained by derive_aux (aux_bc_); field_postprocess only writes phi/grad (comp 0..2).
  void fill_bz() {
    if (!bz_ || aux_ncomp_ <= kAuxBaseComps)
      return;
    for (int li = 0; li < aux_.local_size(); ++li)
      detail::fill_bz_box(aux_.fab(li), aux_.box(li), geom_, bz_);  // valid box
    fill_ghosts(aux_, geom_.domain, aux_bc_);  // B_z halos before the 1st solve
  }

  System system_;
  RhsAssembler rhs_assembler_;
  Geometry geom_;
  BoxArray ba_;
  DistributionMapping dm_;
  BCRec bcPhi_, aux_bc_;
  Elliptic mg_;
  int aux_ncomp_;  // width of the shared aux channel (max over blocks); init before aux_
  MultiFab aux_;
  ScalarFieldProvider2D bz_;  // prepared external B_z(x, y) (empty if not supplied)
};

}  // namespace pops
