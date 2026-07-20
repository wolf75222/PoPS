// Chantier T2 : MASQUE DE DOMAINE DISQUE conservatif (CONTRAT, inerte par defaut).
//
// Le coeur expose deux briques de transport FV :
//   - assemble_rhs<L, F>           : residu -div Fhat + S sur TOUT le domaine (chemin historique) ;
//   - assemble_rhs_masked<L, F>    : MEME residu RESTREINT a un sous-domaine actif (masque 0/1
//                                    cellule-centre), avec flux normal NUL aux faces active/inactive
//                                    (paroi FV) -> conservation de masse sur le sous-domaine actif.
//
// On valide les DEUX proprietes du contrat (vraies assertions, pas de no-op) :
//   (a) BIT-IDENTITE : un masque TOUT ACTIF rend assemble_rhs_masked STRICTEMENT egal a assemble_rhs
//       (egalite bit a bit, diff exactement 0). C'est l'invariant "inerte par defaut" : tant que le
//       sous-domaine est le domaine entier, le residu est celui du chemin historique.
//   (b) CONSERVATION : avec un masque DISQUE (DiscDomain), une advection a vitesse CONSTANTE
//       (champ de transport a divergence nulle) avancee par Euler avant conserve la masse sur les
//       cellules ACTIVES a la PRECISION MACHINE, et le residu est EXACTEMENT 0 sur les cellules
//       inactives (aucun flux ne traverse la frontiere du disque).
//
// Modele jouet INLINE (le coeur ne connait aucune physique) : scalaire advecte a vitesse (vx, vy)
// constante, flux F = (vx u, vy u), vitesse d'onde max |v|. Aucune source, aucun elliptique.

#include <gtest/gtest.h>

#include <pops/core/model/physical_model.hpp>
#include <pops/core/state/state.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/numerics/time/amr/reflux/amr_flux_helpers.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/numerics/spatial_operator.hpp>
#include <pops/runtime/context/wall_predicate.hpp>  // detail::DiscDomain (descripteur source-unique)

#include <cmath>
#include <cstdio>
#include <limits>

using namespace pops;

// Advection scalaire a vitesse constante (vx, vy). Flux F = (vx u, vy u).
struct Advect {
  using State = StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;
  Real vx = 0.0, vy = 0.0;
  POPS_HD State flux(const State& u, const Aux&, int dir) const {
    return State{(dir == 0 ? vx : vy) * u[0]};
  }
  POPS_HD Real max_wave_speed(const State&, const Aux&, int dir) const {
    return std::fabs(dir == 0 ? vx : vy);
  }
  POPS_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

// Device-side Roe providers cannot throw.  This model is the exact failure carrier emitted by a
// dense-Jacobian Roe provider when its eigensolve reports a complex or unresolved spectrum.
struct FailedRoeAdvect : Advect {
  POPS_HD State roe_dissipation(const State&, const Aux&, const State&, const Aux&, int) const {
    return State{std::numeric_limits<Real>::quiet_NaN()};
  }
};

static_assert(PhysicalModel<Advect>, "Advect est un PhysicalModel");
static_assert(!DiffusiveModel<Advect>,
              "Advect n'est pas diffusif (le masque cible le flux hyperbolique)");

// Fixture partagee : boite, geometrie, modele et etat initial (bosse lisse) communs aux deux
// proprietes du contrat (bit-identite masque-tout-actif et conservation sur un masque disque).
class DiscDomainMask : public ::testing::Test {
 protected:
  static constexpr int n = 48;
  static constexpr double L = 1.0;

  DiscDomainMask()
      : dom(Box2D::from_extents(n, n)),
        geom{dom, 0.0, L, 0.0, L},
        ba(std::vector<Box2D>{dom}),
        dm(1, n_ranks()),
        model{0.7, -0.4},  // vitesse constante quelconque, div v = 0
        // U porte Minmod::n_ghost (= 2) couches de ghost : les deux chemins exerces ici (assemble_rhs
        // et assemble_rhs_masked, instancies avec Minmod) reconstruisent les cellules voisines i+-1
        // -> lecture i+-2 au bord de la boite valide. Avec 1 seul ghost cette lecture sortait du
        // buffer (ADC-163, heap-buffer-overflow ASan). aux / masque ne sont lus qu'a i+-1 -> 1 ghost
        // suffit.
        U(ba, dm, 1, Minmod::n_ghost),
        aux(ba, dm, kAuxBaseComps, 1) {
    aux.set_val(0.0);
    // Etat initial : bosse lisse (recouvrement avec le disque). Aux nul (flux ignore aux).
    Array4 a = U.fab(0).array();
    const Box2D g = U.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        const double x = geom.x_cell(i), y = geom.y_cell(j);
        a(i, j, 0) =
            1.0 + 0.5 * std::exp(-(((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) / 0.02));
      }
  }

  const Box2D dom;
  const Geometry geom;
  const BoxArray ba;
  const DistributionMapping dm;
  BCRec bc;  // periodique par defaut (suffit : le masque ferme la frontiere physique du disque)
  const Advect model;
  MultiFab U, aux;
};

TEST_F(DiscDomainMask, AllActiveMaskIsBitIdenticalToUnmaskedResidual) {
  MultiFab mask(ba, dm, 1, 1);
  mask.set_val(Real(1));            // tout actif : le sous-domaine est le domaine entier
  fill_ghosts(U, geom.domain, bc);  // memes ghosts pour les deux chemins
  MultiFab R_ref(ba, dm, 1, 0), R_msk(ba, dm, 1, 0);
  assemble_rhs<Minmod, RusanovFlux>(model, U, aux, geom, R_ref);
  assemble_rhs_masked<Minmod, RusanovFlux>(model, U, aux, mask, geom, R_msk);

  double max_abs_diff = 0.0;
  const ConstArray4 rr = R_ref.fab(0).const_array();
  const ConstArray4 rm = R_msk.fab(0).const_array();
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      max_abs_diff = std::max(max_abs_diff, std::fabs(double(rr(i, j, 0)) - double(rm(i, j, 0))));
  // Egalite BIT A BIT : le chemin masque tout-actif emprunte le MEME flux/reconstruction, on exige
  // une difference EXACTEMENT nulle (pas une tolerance) -- c'est l'invariant "inerte par defaut".
  EXPECT_TRUE(max_abs_diff == 0.0)
      << "masque tout actif : residu masque BIT-IDENTIQUE au residu historique (diff = 0), got "
      << max_abs_diff;
}

TEST_F(DiscDomainMask, DiscMaskConservesActiveMassAndZeroesInactiveResidual) {
  // Disque centre dans la boite, rayon < L/2 pour qu'il y ait de vraies cellules inactives.
  const detail::DiscDomain disc = detail::DiscDomain::centered_in_box(L, 0.35);
  MultiFab mask(ba, dm, 1, 1);
  {
    Array4 m = mask.fab(0).array();
    const Box2D g = mask.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        m(i, j, 0) = disc.cell_active(geom.x_cell(i), geom.y_cell(j)) ? Real(1) : Real(0);
  }

  // Compte les cellules actives ET inactives valides : le test n'a de sens que si les DEUX existent.
  int n_active = 0, n_inactive = 0;
  {
    const ConstArray4 m = mask.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        (m(i, j, 0) >= Real(0.5) ? n_active : n_inactive)++;
  }
  ASSERT_TRUE(n_active > 0 && n_inactive > 0)
      << "le disque partitionne la grille en cellules actives ET inactives (test non vide)";

  // Masse initiale sur les cellules ACTIVES (somme ponderee par le masque). dx2 = aire de cellule.
  const double dx2 = geom.dx() * geom.dy();
  auto active_mass = [&](const MultiFab& F) {
    device_fence();
    const ConstArray4 f = F.fab(0).const_array();
    const ConstArray4 m = mask.fab(0).const_array();
    double s = 0.0;
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        if (m(i, j, 0) >= Real(0.5))
          s += double(f(i, j, 0));
    return s * dx2;
  };

  const double m0 = active_mass(U);
  ASSERT_TRUE(m0 > 0.0) << "masse active initiale strictement positive (bosse couvre le disque)";

  // Avance EXPLICITE Euler avant sur le residu MASQUE : U^{n+1} = U^n + dt R_masked(U^n).
  // Sur une cellule inactive R = 0 -> U y reste fige ; sur une cellule active, le flux normal des
  // faces touchant une inactive est nul -> aucune masse ne franchit la frontiere du disque.
  const double v = std::hypot(model.vx, model.vy);
  const double dt = 0.2 * geom.dx() / v;  // CFL transport
  double max_inactive_residual = 0.0;
  for (int s = 0; s < 60; ++s) {
    fill_ghosts(U, geom.domain, bc);
    MultiFab R(ba, dm, 1, 0);
    assemble_rhs_masked<Minmod, RusanovFlux>(model, U, aux, mask, geom, R);
    // Le residu DOIT etre exactement nul sur les cellules inactives (elles ne sont pas avancees).
    {
      const ConstArray4 r = R.fab(0).const_array();
      const ConstArray4 m = mask.fab(0).const_array();
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
          if (m(i, j, 0) < Real(0.5))
            max_inactive_residual = std::max(max_inactive_residual, std::fabs(double(r(i, j, 0))));
    }
    saxpy(U, Real(dt), R);  // U += dt R (cellules valides)
  }

  const double m1 = active_mass(U);
  const double rel_drift = std::fabs(m1 - m0) / std::fabs(m0);

  // Le residu sur les cellules inactives est EXACTEMENT 0 (le kernel les met a zero) : egalite bit
  // a bit, pas une tolerance.
  EXPECT_TRUE(max_inactive_residual == 0.0)
      << "residu EXACTEMENT nul sur les cellules inactives (aucune avance hors du disque), got "
      << max_inactive_residual;
  // La masse active derive seulement du non-bit-identisme de l'arithmetique flottante (somme de
  // flux internes telescopiques) : borne JUSTE au-dessus du bruit machine (~1e-15 attendu).
  EXPECT_TRUE(rel_drift < 1e-12)
      << "masse sur les cellules actives conservee a la machine (flux normal nul a la frontiere "
         "du disque ; drift < 1e-12), got drift="
      << rel_drift;
  // Temoin que la dynamique a bien TOURNE (sinon la conservation serait triviale) : l'etat a bouge.
  {
    device_fence();
    const ConstArray4 u = U.fab(0).const_array();
    double max_dev = 0.0;
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        max_dev = std::max(max_dev, std::fabs(double(u(i, j, 0)) - 1.0));
    EXPECT_TRUE(max_dev > 1e-3)
        << "le transport a effectivement avance l'etat (la conservation n'est pas triviale)";
  }
}

TEST_F(DiscDomainMask, NonFiniteRoeCannotReachMaskedStateOrAmrFaceLedger) {
  MultiFab mask(ba, dm, 1, 1);
  mask.set_val(Real(1));
  fill_ghosts(U, geom.domain, bc);
  FailedRoeAdvect failed_roe;
  failed_roe.vx = 0.7;
  failed_roe.vy = -0.4;

  MultiFab residual(ba, dm, 1, 0);
  EXPECT_THROW(
      (assemble_rhs_masked<NoSlope, RoeFlux>(failed_roe, U, aux, mask, geom, residual)),
      std::runtime_error);

  MultiFab flux_x(BoxArray(std::vector<Box2D>{xface_box(dom)}), dm, 1, 0);
  MultiFab flux_y(BoxArray(std::vector<Box2D>{yface_box(dom)}), dm, 1, 0);
  compute_face_fluxes<NoSlope, RoeFlux>(failed_roe, U, aux, flux_x, flux_y, geom.dx(), geom.dy());
  EXPECT_THROW(mf_eval_rhs(failed_roe, U, aux, flux_x, flux_y, geom.dx(), geom.dy(), residual),
               std::runtime_error);
}
