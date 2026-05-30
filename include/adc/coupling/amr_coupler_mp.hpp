#pragma once

#include <adc/amr/cluster.hpp>    // berger_rigoutsos, ClusterParams
#include <adc/amr/regrid.hpp>     // tag_cells, grow_tags
#include <adc/amr/tag_box.hpp>    // TagBox
#include <adc/core/types.hpp>
#include <adc/coupling/coupler.hpp>  // detail::coupler_eval_rhs (f = model.elliptic_rhs(U))
#include <adc/elliptic/elliptic_solver.hpp>
#include <adc/elliptic/geometric_mg.hpp>
#include <adc/integrator/amr_reflux_mf.hpp>  // AmrLevelMP, amr_step_multilevel_multipatch, mf_*_mb
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/mesh/refinement.hpp>  // coarsen_index

#include <algorithm>
#include <cmath>
#include <vector>

// Coupleur AMR E x B MULTI-PATCH : meme role qu'AmrCoupler (Poisson grossier -> aux =
// grad phi -> injection -> pas AMR conservatif) mais la hierarchie est multi-box a chaque
// niveau (std::vector<AmrLevelMP>) et l'integration passe par amr_step_multilevel_multipatch
// (reflux coverage-aware route vers la box parente). De plus regrid() reconstruit le niveau
// fin a la volee par Berger-Rigoutsos. Niveau 0 = box unique (le domaine, pour le Poisson).
//
// Se reduit BIT A BIT a AmrCoupler quand chaque niveau n'a qu'une box (garde de validation).

namespace adc {

template <class Model, class Elliptic = GeometricMG>
class AmrCouplerMP {
  static_assert(EllipticSolver<Elliptic>, "Elliptic doit modeler EllipticSolver");

 public:
  AmrCouplerMP(const Model& model, const Geometry& geom, const BoxArray& ba_coarse,
               const BCRec& bc, std::vector<AmrLevelMP> levels)
      : model_(model), geom_(geom), dom_(geom.domain), mg_(geom, ba_coarse, bc),
        L_(std::move(levels)) {
    nlev_ = static_cast<int>(L_.size());
    aux_.resize(nlev_);  // jamais redimensionne -> &aux_[k] stable
    for (int k = 0; k < nlev_; ++k) {
      aux_[k] = MultiFab(L_[k].U.box_array(), L_[k].U.dmap(), 3, 1);
      L_[k].aux = &aux_[k];
    }
  }

  std::vector<AmrLevelMP>& levels() { return L_; }
  MultiFab& coarse() { return L_[0].U; }
  const MultiFab& coarse() const { return L_[0].U; }
  const Box2D& domain() const { return dom_; }
  int nlev() const { return nlev_; }

  void sync_down() {  // moyenne fin -> grossier sur toute la hierarchie (multi-box)
    for (int k = nlev_ - 1; k >= 1; --k) mf_average_down_mb(L_[k].U, L_[k - 1].U);
  }

  void compute_aux() {  // Poisson grossier + grad phi + injection vers les fins
    const int nx = dom_.nx(), ny = dom_.ny();
    const Real dx = geom_.dx(), dy = geom_.dy();
    // second membre via le modele (pas de formule recopiee) : f = elliptic_rhs(U)
    detail::coupler_eval_rhs(L_[0].U, mg_.rhs(), model_);
    mg_.solve();
    const ConstArray4 p = mg_.phi().fab(0).const_array();
    Array4 a0 = aux_[0].fab(0).array();
    device_fence();
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        a0(i, j, 0) = p(i, j);
        a0(i, j, 1) = (p(i + 1, j) - p(i - 1, j)) / (2 * dx);
        a0(i, j, 2) = (p(i, j + 1) - p(i, j - 1)) / (2 * dy);
      }
    fill_boundary(aux_[0], dom_, Periodicity{true, true});
    for (int k = 1; k < nlev_; ++k) inject_aux_mb(aux_[k - 1], aux_[k]);
  }

  void update() { sync_down(); compute_aux(); }

  void step(Real dt) {
    update();
    amr_step_multilevel_multipatch<NoSlope, RusanovFlux>(model_, L_, dom_, dt);
  }

  // Regrid du niveau FIN (le plus fin) par Berger-Rigoutsos sur le critere applique au
  // niveau parent. Reconstruit les patchs (report des donnees fines existantes la ou
  // possible, sinon interpolation depuis le parent) + l'aux associe. margin = nesting.
  template <class Crit>
  void regrid(Crit crit, int grow = 2, int margin = 2) {
    if (nlev_ < 2) return;
    const int fk = nlev_ - 1, pk = fk - 1;  // fin et son parent
    const int PNX = dom_.nx() << pk, PNY = dom_.ny() << pk;
    const Box2D pdom = Box2D::from_extents(PNX, PNY);
    TagBox tags = tag_cells(L_[pk].U, pdom, crit);
    TagBox grown = grow_tags(tags, grow, pdom);
    std::vector<Box2D> cl = berger_rigoutsos(grown, ClusterParams{});
    std::vector<Box2D> fb;  // patchs fins (coords niveau fk = parent x2)
    for (Box2D b : cl) {
      b.lo[0] = std::max(b.lo[0], margin); b.lo[1] = std::max(b.lo[1], margin);
      b.hi[0] = std::min(b.hi[0], PNX - 1 - margin); b.hi[1] = std::min(b.hi[1], PNY - 1 - margin);
      if (b.hi[0] < b.lo[0] || b.hi[1] < b.lo[1]) continue;
      fb.push_back(Box2D{{2 * b.lo[0], 2 * b.lo[1]}, {2 * b.hi[0] + 1, 2 * b.hi[1] + 1}});
    }
    if (fb.empty()) return;  // rien a raffiner : on garde la grille courante
    MultiFab nU(BoxArray(fb), DistributionMapping((int)fb.size(), n_ranks()), L_[fk].U.ncomp(), 1);
    const MultiFab& par = L_[pk].U;
    const MultiFab& old = L_[fk].U;
    const int ncf = nU.ncomp();
    for (int li = 0; li < nU.local_size(); ++li) {
      Array4 a = nU.fab(li).array();
      const Box2D nb = nU.box(li);
      for (int j = nb.lo[1]; j <= nb.hi[1]; ++j)  // 1) interp depuis le parent
        for (int i = nb.lo[0]; i <= nb.hi[0]; ++i) {
          const int pb = mf_find_box(par, coarsen_index(i, 2), coarsen_index(j, 2));
          if (pb < 0) continue;
          const ConstArray4 pp = par.fab(pb).const_array();
          for (int k = 0; k < ncf; ++k) a(i, j, k) = pp(coarsen_index(i, 2), coarsen_index(j, 2), k);
        }
      for (int ol = 0; ol < old.local_size(); ++ol) {  // 2) report des donnees fines
        const ConstArray4 o = old.fab(ol).const_array();
        const Box2D inter = nb.intersect(old.box(ol));
        if (inter.empty()) continue;
        for (int j = inter.lo[1]; j <= inter.hi[1]; ++j)
          for (int i = inter.lo[0]; i <= inter.hi[0]; ++i)
            for (int k = 0; k < ncf; ++k) a(i, j, k) = o(i, j, k);
      }
    }
    L_[fk].U = std::move(nU);
    aux_[fk] = MultiFab(L_[fk].U.box_array(), L_[fk].U.dmap(), 3, 1);  // adresse stable
    L_[fk].aux = &aux_[fk];
  }

  Real mass() const {
    device_fence();
    const ConstArray4 u = L_[0].U.fab(0).const_array();
    const int nx = dom_.nx(), ny = dom_.ny();
    const Real dV = geom_.dx() * geom_.dy();
    Real M = 0;
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) M += u(i, j, 0) * dV;
    return M;
  }

  Real max_drift_speed() const {
    device_fence();
    const ConstArray4 a = aux_[0].fab(0).const_array();
    const int nx = dom_.nx(), ny = dom_.ny();
    Real v = 0;
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) v = std::max(v, std::hypot(a(i, j, 1), a(i, j, 2)) / model_.B0);
    return std::max(v, Real(1e-12));
  }

 private:
  // injection piecewise-constante de aux (3 comp) parent multi-box -> enfant multi-box.
  static void inject_aux_mb(const MultiFab& parent, MultiFab& child) {
    device_fence();
    for (int lc = 0; lc < child.local_size(); ++lc) {
      Array4 c = child.fab(lc).array();
      const Box2D g = child.fab(lc).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
          const int ci = coarsen_index(i, 2), cj = coarsen_index(j, 2);
          const int pb = mf_find_box(parent, ci, cj);
          if (pb < 0) continue;
          const ConstArray4 pp = parent.fab(pb).const_array();
          for (int k = 0; k < 3; ++k) c(i, j, k) = pp(ci, cj, k);
        }
    }
  }

  Model model_;
  Geometry geom_;
  Box2D dom_;
  Elliptic mg_;
  std::vector<AmrLevelMP> L_;
  std::vector<MultiFab> aux_;
  int nlev_ = 0;
};

}  // namespace adc
