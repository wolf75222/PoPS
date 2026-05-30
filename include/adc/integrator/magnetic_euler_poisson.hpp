#pragma once

#include <adc/core/types.hpp>
#include <adc/coupling/coupler.hpp>
#include <adc/coupling/coupling_policy.hpp>
#include <adc/elliptic/geometric_mg.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/euler_poisson.hpp>
#include <adc/operator/reconstruction.hpp>

#include <cmath>
#include <functional>
#include <utility>

// Euler-Poisson MAGNETIQUE (Hoffart, arXiv:2510.11808, eq 2.4). Euler compressible
// couple a Poisson AVEC la force de Lorentz magnetique m x Omega, Omega = champ
// magnetique hors-plan uniforme. C'est le systeme COMPLET : la dynamique cyclotron est
// resolue, pas supposee instantanee comme dans la limite de derive (le modele Diocotron).
//
//   d_t rho + div m                      = 0
//   d_t m   + div(rho^-1 m m^T + I p)     = -rho grad phi + m x Omega
//   d_t E   + div(rho^-1 m (E+p))         = -m . grad phi
//   -lap phi = alpha (rho - rho0)
//
// (la forme stationnaire de Poisson est equivalente a d_t(-lap phi) = -alpha div m :
// deriver -lap phi = alpha(rho-rho0) en temps et injecter la continuite.)
//
// La force m x Omega (Omega = Omega z, hors-plan) ne fait que FAIRE TOURNER (m_x, m_y)
// a la frequence cyclotron omega_c = |Omega| : elle ne fait pas de travail (absente de
// l'energie). La rotation exacte est inconditionnellement stable (aucune CFL en
// omega_c), donc le schema est ASYMPTOTIC-PRESERVING : quand Omega -> inf, le point fixe
// du moment -rho grad phi + m x Omega = 0 donne m = rho v avec la derive E x B
// v = (-d_y phi, d_x phi)/Omega, EXACTEMENT la vitesse du modele Diocotron. Le systeme
// complet se reduit donc a la limite de derive (M1/M2) sans pas de temps qui s'effondre.
//
// Integration : splitting de Strang autour du Coupler<EulerPoisson> deja valide.
//   1/2 rotation cyclotron -> 1 pas transport + electrostatique (SSPRK2, Poisson par
//   etage) -> 1/2 rotation. A Omega = 0, la rotation est l'identite BIT A BIT, donc le
//   pas magnetique est bit-identique au Coupler nu (filet de validation).

namespace adc {

// Rotation cyclotron exacte de la quantite de mouvement (composantes 1, 2) d'angle
// theta, convention [[c, s], [-s, c]] (identique a tfap_rotate_mom du chemin deux-fluides
// AP). rho (comp 0) et E (comp 3) sont inchanges : la rotation conserve |m|, donc
// l'energie cinetique |m|^2/2rho ET, E etant fixe, l'energie interne. Boucle sur les fabs
// locales (correct en multi-box / MPI). A theta = 0 : c = 1, s = 0 -> identite bit a bit.
inline void magnetic_rotate(MultiFab& U, Real theta) {
  const Real c = std::cos(theta), s = std::sin(theta);
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 m = U.fab(li).array();
    const Box2D v = U.box(li);
    for_each_cell(v, [=] ADC_HD(int i, int j) {
      const Real mx = m(i, j, 1), my = m(i, j, 2);
      m(i, j, 1) = c * mx + s * my;
      m(i, j, 2) = -s * mx + c * my;
    });
  }
}

// Pas magnetique Euler-Poisson par splitting de Strang autour de Coupler<EulerPoisson>.
template <class Elliptic = GeometricMG>
class MagneticEulerPoissonCoupler {
 public:
  // Omega : composante hors-plan du champ magnetique rescale (frequence cyclotron signee).
  // active : predicat de paroi conductrice optionnel, transmis tel quel au Coupler.
  MagneticEulerPoissonCoupler(const EulerPoisson& model, const Geometry& geom,
                              const BoxArray& ba, const BCRec& bcU, const BCRec& bcPhi,
                              Real Omega, std::function<bool(Real, Real)> active = {})
      : cpl_(model, geom, ba, bcU, bcPhi, std::move(active)), omega_(Omega) {}

  // Strang : 1/2 rotation (angle Omega*dt/2), transport + electrostatique (dt), 1/2
  // rotation. Symetrique -> ordre 2 en temps. Limiter et Policy passent au Coupler.
  template <class Limiter = NoSlope, class Policy = PerStageCoupling>
  void step(MultiFab& U, Real dt) {
    const Real half = Real(0.5) * omega_ * dt;
    magnetic_rotate(U, half);
    cpl_.template advance<Limiter, Policy>(U, dt);
    magnetic_rotate(U, half);
  }

  void solve_fields(const MultiFab& U) { cpl_.solve_fields(U); }
  MultiFab& phi() { return cpl_.phi(); }
  const MultiFab& aux() const { return cpl_.aux(); }
  Real omega() const { return omega_; }
  Coupler<EulerPoisson, Elliptic>& coupler() { return cpl_; }

 private:
  Coupler<EulerPoisson, Elliptic> cpl_;
  Real omega_;
};

}  // namespace adc
