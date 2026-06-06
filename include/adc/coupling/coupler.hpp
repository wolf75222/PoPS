#pragma once

#include <adc/core/types.hpp>
#include <adc/coupling/aux_fill.hpp>  // detail::derive_aux_bc + detail::fill_bz_box (partages)
#include <adc/coupling/coupling_policy.hpp>
#include <adc/coupling/elliptic_rhs.hpp>
#include <adc/numerics/elliptic/elliptic_problem.hpp>
#include <adc/numerics/elliptic/elliptic_solver.hpp>
#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/numerics/time/time_integrator.hpp>
#include <adc/numerics/time/time_steppers.hpp>  // SSPRK2Step / SSPRK3Step (schema partage)
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/reconstruction.hpp>
#include <adc/numerics/spatial_discretisation.hpp>
#include <adc/numerics/spatial_operator.hpp>
#include <adc/parallel/comm.hpp>

#include <functional>
#include <type_traits>
#include <utility>

/// @file
/// @brief Coupler : coupleur hyperbolique-elliptique MONO-bloc (boucle Poisson -> aux -> advance).
///
/// A chaque etage de l'integrateur (couplage stade par stade) : (1) RHS f = elliptic_rhs(model, U) ;
/// (2) resolution lap(phi) = f par le backend elliptique (warm start) ; (3) aux = (phi, grad phi) par
/// differences centrees ; (4) assemblage du residu hyperbolique avec cet aux. Pour un transport a
/// derive l'aux entre par le FLUX (E x B) ; pour un fluide auto-gravitant par la SOURCE. Trois axes
/// orthogonaux, tous parametres de template : Limiter (reconstruction), Policy (PerStage vs
/// OncePerStep), NumericalFlux (Rusanov par defaut). Compatible MONO-modele ; le multi-especes passe
/// par SystemCoupler. Les detail:: sont a portee de namespace (un lambda etendu ADC_HD ne peut pas
/// vivre dans une methode privee, restriction nvcc).

// Coupleur hyperbolique-elliptique mono-bloc : ferme la boucle Poisson -> aux -> advance.
//
// A chaque etage de l'integrateur (couplage stade par stade) :
//   1. second membre f via SingleModelEllipticRhs(model, U)
//   2. resolution lap(phi) = f par la multigrille geometrique (warm start)
//   3. aux = (phi, grad phi) par differences centrees
//   4. assemblage du residu hyperbolique avec ce aux
//
// Pour un transport a derive aux entre par le flux (derive E x B) ; pour un fluide
// compressible auto-gravitant il entrerait par la source. Le coupleur reste compatible mono-modele ; le niveau
// multi-especes doit assembler le rhs elliptique depuis un CoupledSystem.

namespace adc {

namespace detail {
// Helpers a portee de namespace : un lambda etendu __host__ __device__ ne peut
// PAS etre defini dans une methode privee/protegee (restriction nvcc), d'ou
// l'extraction hors de la classe Coupler.

// Compatibilite mono-modele : f = model.elliptic_rhs(U) sur les cellules valides,
// deleguee a un assembleur nomme pour ne pas enfermer cette responsabilite dans Coupler.
/// Assemble le RHS elliptique mono-modele : rhs = model.elliptic_rhs(U) sur les cellules valides
/// (delegue a SingleModelEllipticRhs). Partage par Coupler et AmrCouplerMP.
template <class Model>
inline void coupler_eval_rhs(const MultiFab& state, MultiFab& rhs,
                             const Model& model) {
  SingleModelEllipticRhs<Model>{model}(state, rhs);
}

// aux = (phi, d phi/dx, d phi/dy) par differences centrees. Delegue a la
// convention nommee FieldPostProcess avec GradSign::Plus et store_phi=true : le
// coupler stocke +grad phi (le signe physique E = -grad phi est porte par
// la vitesse de derive du transport). Forme multiplicative *cx / *cy conservee a
// l'identique -> bit-identique.
/// Pose aux = (phi, d phi/dx, d phi/dy) par differences centrees (facteurs cx, cy = 1/(2 dx),
/// 1/(2 dy)). Stocke +grad phi (le signe physique E = -grad phi est porte par la vitesse de derive).
inline void coupler_grad_phi(const MultiFab& phi, MultiFab& aux, Real cx,
                             Real cy) {
  field_postprocess(phi, aux, cx, cy,
                    FieldPostProcess{FieldPostProcess::GradSign::Plus, true});
}
}  // namespace detail

/// Coupleur hyperbolique-elliptique mono-bloc. @tparam Model : PhysicalModel (flux, source,
/// elliptic_rhs, max_wave_speed, canal aux). @tparam Elliptic : backend elliptique (concept
/// EllipticSolver, defaut GeometricMG). Possede l'aux et le solveur ; advance/step ferment la boucle
/// Poisson -> aux -> residu a chaque pas. PRECONDITION : U porte au moins Limiter::n_ghost ghosts.
template <class Model, class Elliptic = GeometricMG>
class Coupler {
  static_assert(EllipticSolver<Elliptic>,
                "le backend elliptique du Coupler doit modeler EllipticSolver");

 public:
  // active : predicat optionnel "interieur du conducteur" (paroi embedded pour
  // le solveur de Poisson). Vide => pas de paroi interne.
  // bz : champ magnetique hors-plan B_z(x, y) FOURNI par l'utilisateur (constante ou
  // champ). N'a d'effet que si le modele declare la composante aux B_z (aux_comps>3) ;
  // peuple alors la composante aux 3 une fois pour toutes (B_z statique, externe a
  // l'elliptique : derive_aux ne la touche pas). Vide => pas de B_z. Le canal aux est
  // alloue a la largeur DU MODELE : un modele de base (3) reste bit-identique.
  Coupler(const Model& model, const Geometry& geom, const BoxArray& ba,
          const BCRec& bcU, const BCRec& bcPhi,
          std::function<bool(Real, Real)> active = {},
          std::function<Real(Real, Real)> bz = {})
      : model_(model),
        geom_(geom),
        ba_(ba),
        dm_(ba.size(), n_ranks()),
        bcU_(bcU),
        bcPhi_(bcPhi),
        aux_bc_(detail::derive_aux_bc(bcPhi)),
        mg_(geom, ba, bcPhi, std::move(active)),
        aux_(ba, dm_, aux_comps<Model>(), 1),
        bz_(std::move(bz)) {
    fill_bz();  // peuple la composante B_z (no-op si modele de base ou bz vide)
  }

  // SSPRK2 couple. Trois axes orthogonaux, tous parametres de template :
  //   - Limiter      : reconstruction (NoSlope / Minmod / VanLeer ...)
  //   - Policy       : couplage temporel (PerStage = phi a chaque etage ; OncePerStep
  //                    = un seul solve par pas, aux gele)
  //   - NumericalFlux : flux de Riemann (Rusanov par defaut, HLL, HLLC ...)
  // U doit avoir au moins Limiter::n_ghost ghosts. La signature historique
  // advance<Limiter, Policy> reste valide (NumericalFlux defaut = Rusanov).
  template <class Limiter = NoSlope, class Policy = PerStageCoupling,
            class NumericalFlux = RusanovFlux>
  void advance(MultiFab& U, Real dt) {
    static_assert(std::is_same_v<Policy, PerStageCoupling> ||
                      std::is_same_v<Policy, OncePerStepCoupling>,
                  "Policy doit etre PerStageCoupling ou OncePerStepCoupling");
    constexpr bool per = std::is_same_v<Policy, PerStageCoupling>;
    // DELEGUE le schema a l'objet SSPRK2Step du coeur (dedup, §8.2 A4). L'evaluateur de
    // residu compte les etages : recompute_aux=true a l'etage 0, =per ensuite (PerStage :
    // phi recalcule pour l'etat intermediaire ; OncePerStep : aux gele). Bit-identique.
    int stage = 0;
    SSPRK2Step{}.take_step(
        [&](MultiFab& s, MultiFab& R) {
          stage_rhs<Limiter, NumericalFlux>(s, R, (stage++ == 0) ? true : per);
        },
        U, dt);
  }

  // SSPRK3 couple (Shu-Osher, 3 etages). Memes axes que advance.
  template <class Limiter = NoSlope, class Policy = PerStageCoupling,
            class NumericalFlux = RusanovFlux>
  void advance_ssprk3(MultiFab& U, Real dt) {
    static_assert(std::is_same_v<Policy, PerStageCoupling> ||
                      std::is_same_v<Policy, OncePerStepCoupling>,
                  "Policy doit etre PerStageCoupling ou OncePerStepCoupling");
    constexpr bool per = std::is_same_v<Policy, PerStageCoupling>;
    // Idem advance : delegue a SSPRK3Step, recompute_aux=true a l'etage 0, =per ensuite.
    int stage = 0;
    SSPRK3Step{}.take_step(
        [&](MultiFab& s, MultiFab& R) {
          stage_rhs<Limiter, NumericalFlux>(s, R, (stage++ == 0) ? true : per);
        },
        U, dt);
  }

  // Point d'entree unifie : on donne au coupleur une DISCRETISATION SPATIALE
  // (limiteur + flux) et une politique de temps explicite. Les anciens tags
  // SSPRK2/SSPRK3 restent valides ; la forme nouvelle permet aussi le sous-cyclage :
  //   sim.step<MusclVanLeerHLLC, ExplicitTime<SSPRK3, 4>>(U, dt);
  template <class Disc = FirstOrder, class TimeInteg = SSPRK2,
            class Policy = PerStageCoupling>
  void step(MultiFab& U, Real dt) {
    using L = typename Disc::Limiter;
    using F = typename Disc::NumericalFlux;
    using T = typename TimePolicyTraits<TimeInteg>::Method;
    static_assert(TimePolicyTraits<TimeInteg>::treatment == TimeTreatment::Explicit,
                  "Coupler::step ne sait executer que des politiques explicites ; "
                  "utiliser un scheduler/systeme pour IMEX ou implicite");
    static_assert(std::is_same_v<T, SSPRK2> || std::is_same_v<T, SSPRK3>,
                  "Coupler::step supporte SSPRK2 ou SSPRK3");
    constexpr int n = TimePolicyTraits<TimeInteg>::substeps;
    const Real h = dt / static_cast<Real>(n);
    for (int s = 0; s < n; ++s) {
      if constexpr (std::is_same_v<T, SSPRK3>)
        advance_ssprk3<L, Policy, F>(U, h);
      else
        advance<L, Policy, F>(U, h);
    }
  }

  // Resout phi et derive aux pour un etat donne, sans avancer en temps
  // (utile pour estimer la vitesse E x B avant de fixer le pas de temps).
  /// Resout phi et derive aux = (phi, grad phi) pour @p U SANS avancer en temps (utile pour estimer
  /// la vitesse E x B avant de fixer dt). aux() est a jour au retour.
  void solve_fields(const MultiFab& U) { update_aux(U); }

  MultiFab& phi() { return mg_.phi(); }
  const MultiFab& aux() const { return aux_; }

 private:
  void update_aux(const MultiFab& state) {
    detail::coupler_eval_rhs(state, mg_.rhs(), model_);
    mg_.solve();  // interface du concept EllipticSolver (backend-agnostique)
    derive_aux();
  }

  // Un etage : (option) resolution elliptique, halos, residu hyperbolique dans R.
  // Mutualise par advance (SSPRK2) et advance_ssprk3 ; ordre des operations conserve
  // pour rester bit-identique a l'ancien advance.
  template <class Limiter, class NumericalFlux>
  void stage_rhs(MultiFab& s, MultiFab& R, bool recompute_aux) {
    if (recompute_aux) update_aux(s);
    fill_ghosts(s, geom_.domain, bcU_);
    assemble_rhs<Limiter, NumericalFlux>(model_, s, aux_, geom_, R);
  }

  void derive_aux() {
    fill_ghosts(mg_.phi(), geom_.domain, bcPhi_);
    const Real cx = Real(1) / (2 * geom_.dx());
    const Real cy = Real(1) / (2 * geom_.dy());
    detail::coupler_grad_phi(mg_.phi(), aux_, cx, cy);
    fill_ghosts(aux_, geom_.domain, aux_bc_);
  }

  // Peuple la composante aux B_z (indice kAuxBaseComps) sur les cellules valides depuis
  // bz_(x, y), une seule fois (B_z statique). Garde compile-time : sans champ B_z dans le
  // modele (aux_comps == 3) la composante n'existe pas -> aucun code, aucun acces hors borne.
  // Les halos de B_z sont ensuite maintenus par derive_aux (Foextrap/periodique d'aux_bc_,
  // cf. grad) ; field_postprocess n'ecrit que phi/grad (composantes 0..2), B_z est preserve.
  void fill_bz() {
    if constexpr (aux_comps<Model>() > kAuxBaseComps) {
      if (!bz_) return;
      for (int li = 0; li < aux_.local_size(); ++li)
        detail::fill_bz_box(aux_.fab(li), aux_.box(li), geom_, bz_);  // boite valide
      fill_ghosts(aux_, geom_.domain, aux_bc_);  // halos de B_z avant le 1er solve
    }
  }

  Model model_;
  Geometry geom_;
  BoxArray ba_;
  DistributionMapping dm_;
  BCRec bcU_, bcPhi_, aux_bc_;
  Elliptic mg_;
  MultiFab aux_;
  std::function<Real(Real, Real)> bz_;  // B_z(x, y) externe (vide si non fourni)
};

// Le backend elliptique du coupleur respecte le contrat commun : echanger
// GeometricMG contre un autre solveur conforme (FFT enveloppe, PETSc) ne demandera
// que de changer le type du membre, pas la logique de couplage.
static_assert(EllipticSolver<GeometricMG>,
              "GeometricMG doit modeler le concept EllipticSolver");

}  // namespace adc
