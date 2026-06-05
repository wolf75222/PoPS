#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/numerics/elliptic/poisson_operator.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>

// Solveur de Krylov MATRICE-LIBRE (BiCGStab) pour l'operateur elliptique a TENSEUR PLEIN
// L(phi) = -div(A grad phi) + kappa phi, A = [[Axx, Axy], [Ayx, Ayy]] eventuellement NON
// SYMETRIQUE (#120). Le verrou trouve par #120 : sur un A non symetrique (terme croise
// Axy != Ayx, p.ex. la rotation B^{-1} de la condensation de Schur, arXiv:2510.11808), le
// V-cycle GeometricMG SEUL (lisseur Gauss-Seidel 5 points, bloc diagonal, termes croises
// EXPLICITES) STAGNE (c = 0.1-0.4) ou DIVERGE (c = 0.7) : son lisseur suppose un operateur
// auto-adjoint. Un solveur de Krylov non symetrique est alors requis (SCHUR_CONDENSATION_DESIGN
// section "niveau 2", point a trancher de la PR3).
//
// CHOIX : BiCGStab (et non GMRES). Raisons :
//   - gere le NON SYMETRIQUE (comme GMRES) ;
//   - PAS de parametre de redemarrage ni de base de Krylov croissante a stocker : empreinte
//     memoire FIXE (une poignee de MultiFab : r, rhat, p, v, s, t, + le preconditionne phat/shat),
//     plus simple qu'un GMRES(m) (orthogonalisation d'Arnoldi, rotations de Givens, choix de m) ;
//   - matvec = l'APPLICATION de l'operateur plein (apply_laplacian, #120) sur MultiFab, JAMAIS une
//     matrice globale assemblee : strictement matrice-libre, GPU/MPI-pret.
//
// PRECONDITIONNEUR : N V-cycles (N petit, 1-2) du GeometricMG EXISTANT applique a la PARTIE
// SYMETRIQUE de l'operateur (le bloc DIAGONAL : termes croises Axy/Ayx LARGUES). C'est exactement ce
// que GeometricMG sait deja faire et fait bien (Poisson scalaire ou anisotrope eps_x/eps_y, +kappa) :
// on REUTILISE GeometricMG tel quel. La partie antisymetrique est en O(theta^2 dt^2 alpha), petite a
// CFL source raisonnable, donc le preconditionneur symetrique capture l'essentiel du spectre.
//
// CONVENTION (alignee sur poisson_operator.hpp / GeometricMG) : on resout L_int(phi) = rhs avec
// L_int = div(A grad phi) - kappa phi (ce que current_residual() de GeometricMG traite comme
// "L phi", res = rhs - L_int). La matvec est donc apply_laplacian (qui calcule EXACTEMENT L_int) et
// le residu rhs - L_int(phi), bit-coherent avec poisson_residual. Le cas A = I, kappa = 0 redonne le
// Laplacien canonique : BiCGStab y converge vers la MEME solution que GeometricMG (a la tolerance).
//
// DEVICE / MPI : foncteurs nommes uniquement (mf_arith : saxpy/lincomb/norm_inf/dot ; apply_laplacian
// ; V-cycle MG). Les produits scalaires (dot, norm via dot) sont COLLECTIFS (all_reduce_sum) et
// appeles sur TOUS les rangs, y compris un rang SANS box (local_size()==0) : aucun court-circuit, donc
// pas d'interblocage MPI ni de desynchronisation du critere d'arret.
//
// ADDITIF : aucun chemin existant (GeometricMG / Poisson) ne passe par ce header. Opt-in.

namespace adc {

// Resultat d'un solve BiCGStab : iterations effectuees, residu relatif final, drapeau de convergence.
struct KrylovResult {
  int iters = 0;          ///< nombre d'iterations BiCGStab effectuees
  Real rel_residual = 0;  ///< ||r_final|| / ||r_0|| (norme L2 globale)
  bool converged = false; ///< true si rel_residual <= rel_tol atteint
};

// BiCGStab matrice-libre, preconditionne par N V-cycles du GeometricMG sur la partie SYMETRIQUE.
//
// @p op       : GeometricMG portant l'operateur PLEIN (configure via set_cross_terms / set_epsilon*
//               / set_reaction selon le cas). Sert (a) de stockage des champs phi/rhs du niveau fin
//               (phi()/rhs()/geom()), et (b) de source des coefficients de l'operateur pour la matvec
//               (op_eps(), op_a_xy(), ...). C'est l'objet que modele le concept EllipticSolver.
// @p precond  : GeometricMG portant la partie SYMETRIQUE (memes eps/eps_y/kappa, MAIS set_cross_terms
//               NON appele -> bloc diagonal). Son phi()/rhs() servent de tampon de preconditionnement.
//               PEUT etre le MEME objet que @p op (alors le preconditionneur tourne avec les termes
//               croises EXPLICITES dans le residu MG -- toujours valable, le lisseur reste diagonal),
//               mais un objet SEPARE sans termes croises est le preconditionneur symetrique propre.
class TensorKrylovSolver {
 public:
  // @p n_precond_vcycles : nombre N de V-cycles MG par application du preconditionneur (1 ou 2).
  TensorKrylovSolver(GeometricMG& op, GeometricMG& precond, int n_precond_vcycles = 1)
      : op_(op), precond_(precond), n_precond_(n_precond_vcycles),
        ba_(op.box_array()), dm_(op.dmap()),
        r_(ba_, dm_, 1, 0), rhat_(ba_, dm_, 1, 0), p_(ba_, dm_, 1, 0),
        v_(ba_, dm_, 1, 0), s_(ba_, dm_, 1, 0), t_(ba_, dm_, 1, 0),
        phat_(ba_, dm_, 1, 1), shat_(ba_, dm_, 1, 1) {}

  // --- concept EllipticSolver ---
  MultiFab& phi() { return op_.phi(); }
  MultiFab& rhs() { return op_.rhs(); }
  const Geometry& geom() const { return op_.geom(); }
  // residu L2 GLOBAL courant ||rhs - L_int(phi)|| (collectif). Pendant L2 du norm_inf de GeometricMG.
  Real residual() {
    apply_operator(phi(), r_);          // r_ = L_int(phi)
    lincomb(r_, Real(1), rhs(), Real(-1), r_);  // r_ = rhs - L_int(phi)
    return l2_norm(r_);
  }
  void solve() { solve(Real(1e-10), 200); }

  // BiCGStab preconditionne. phi() est l'inconnue (warm start : valeur entrante = point de depart) ;
  // rhs() le second membre. Renvoie iterations + residu relatif + convergence.
  KrylovResult solve(Real rel_tol, int max_iters) {
    // r0 = rhs - L_int(phi)  (residu vrai, warm start respecte)
    apply_operator(phi(), v_);                       // v_ = L_int(phi)
    lincomb(r_, Real(1), rhs(), Real(-1), v_);       // r_ = rhs - L_int(phi)
    const Real bnorm = l2_norm(rhs());
    const Real norm0 = bnorm > Real(0) ? bnorm : Real(1);  // base relative (rhs nul -> absolue)
    Real rnorm = l2_norm(r_);
    KrylovResult res;
    res.rel_residual = rnorm / norm0;
    if (rnorm <= rel_tol * norm0) { res.converged = true; return res; }  // deja convergent

    // rhat = r0 fige (vecteur fantome de BiCGStab) ; p, v <- 0.
    copy_into(rhat_, r_);
    p_.set_val(Real(0));
    v_.set_val(Real(0));
    Real rho_prev = Real(1), alpha = Real(1), omega = Real(1);

    for (int k = 1; k <= max_iters; ++k) {
      const Real rho = dot(rhat_, r_);  // COLLECTIF (tous rangs)
      // garde-fou : rupture de BiCGStab (rho ou omega ~ 0). On rend le meilleur effort courant.
      if (std::fabs(rho) < kTiny || std::fabs(omega) < kTiny) {
        res.iters = k - 1;
        res.rel_residual = rnorm / norm0;
        return res;
      }
      const Real beta = (rho / rho_prev) * (alpha / omega);
      // p <- r + beta (p - omega v)
      lincomb(p_, Real(1), p_, -omega, v_);     // p <- p - omega v
      lincomb(p_, beta, p_, Real(1), r_);       // p <- r + beta p
      // phat = M^{-1} p  (N V-cycles MG sur la partie symetrique)
      apply_precond(p_, phat_);
      apply_operator(phat_, v_);                // v = L_int(phat)
      const Real rhat_dot_v = dot(rhat_, v_);   // COLLECTIF
      if (std::fabs(rhat_dot_v) < kTiny) { res.iters = k - 1; res.rel_residual = rnorm / norm0; return res; }
      alpha = rho / rhat_dot_v;
      // s <- r - alpha v
      lincomb(s_, Real(1), r_, -alpha, v_);
      // phi <- phi + alpha phat   (correction partielle ; tampon avant le test sur ||s||)
      saxpy(phi(), alpha, phat_);
      const Real snorm = l2_norm(s_);
      if (snorm <= rel_tol * norm0) {  // convergence a mi-iteration
        rnorm = snorm;
        res.iters = k; res.rel_residual = rnorm / norm0; res.converged = true; return res;
      }
      // shat = M^{-1} s ; t = L_int(shat)
      apply_precond(s_, shat_);
      apply_operator(shat_, t_);
      const Real tt = dot(t_, t_);              // COLLECTIF
      omega = tt > kTiny ? dot(t_, s_) / tt : Real(0);
      // phi <- phi + omega shat ; r <- s - omega t
      saxpy(phi(), omega, shat_);
      lincomb(r_, Real(1), s_, -omega, t_);
      rnorm = l2_norm(r_);
      res.iters = k;
      res.rel_residual = rnorm / norm0;
      if (rnorm <= rel_tol * norm0) { res.converged = true; return res; }
      rho_prev = rho;
    }
    return res;  // max_iters atteint sans convergence : meilleur effort (converged=false)
  }

 private:
  static constexpr Real kTiny = Real(1e-300);  // garde-fou rupture / division par 0

  // matvec MATRICE-LIBRE : out = L_int(in) = div(A grad in) - kappa in, ghosts de in remplis avant.
  // Reutilise les coefficients de l'operateur PLEIN de op_ (memes pointeurs que current_residual).
  void apply_operator(MultiFab& in, MultiFab& out) {
    device_fence();  // un kernel a pu ecrire in ; on attend avant la lecture hote de fill_ghosts
    fill_ghosts(in, op_.geom().domain, op_.bc());
    apply_laplacian(in, op_.geom(), out, op_.op_coef(), op_.op_eps(), op_.op_kappa(),
                    op_.op_eps_y(), op_.op_a_xy(), op_.op_a_yx());
    // cellules conductrices (mask==0) : L_int y est 0 (Dirichlet phi=0), comme poisson_residual.
    if (const MultiFab* mk = op_.op_mask()) mask_zero(out, *mk);
  }

  // preconditionneur M^{-1} : out = (N V-cycles MG sur la partie symetrique) appliques a in.
  // On resout M out = in a CL HOMOGENES, depart out = 0 (pas de warm start : M^{-1} est un operateur
  // lineaire fige par iteration BiCGStab). precond_ ne porte PAS les termes croises -> bloc diagonal.
  void apply_precond(MultiFab& in, MultiFab& out) {
    copy_into(precond_.rhs(), in);
    precond_.phi().set_val(Real(0));
    for (int i = 0; i < n_precond_; ++i) precond_.vcycle();
    copy_into(out, precond_.phi());
  }

  // norme L2 GLOBALE sqrt(sum x^2), collective (dot). sqrt hote, identique sur tous les rangs.
  Real l2_norm(const MultiFab& x) { return std::sqrt(dot(x, x)); }

  // copie composante 0 des cellules valides (src -> dst), foncteur nomme (device-clean).
  void copy_into(MultiFab& dst, const MultiFab& src) {
    for (int li = 0; li < dst.local_size(); ++li) {
      Array4 d = dst.fab(li).array();
      const ConstArray4 s = src.fab(li).const_array();
      for_each_cell(dst.box(li), detail::CopyComp0Kernel{d, s});
    }
  }

  // fige out=0 sur les cellules conductrices (mask==0), foncteur nomme (reutilise ZeroConductorKernel).
  void mask_zero(MultiFab& out, const MultiFab& mask) {
    for (int li = 0; li < out.local_size(); ++li) {
      Array4 o = out.fab(li).array();
      const ConstArray4 m = mask.fab(li).const_array();
      for_each_cell(out.box(li), detail::ZeroConductorKernel{o, m});
    }
  }

  GeometricMG& op_;
  GeometricMG& precond_;
  int n_precond_;
  BoxArray ba_;
  DistributionMapping dm_;
  MultiFab r_, rhat_, p_, v_, s_, t_;  // 0 ghost : ops point a point
  MultiFab phat_, shat_;               // 1 ghost : entrees de apply_operator (fill_ghosts)
};

}  // namespace adc
