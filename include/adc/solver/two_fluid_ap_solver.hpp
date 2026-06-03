#pragma once

#include <memory>
#include <vector>

/// @file
/// @brief Facade du solveur deux-fluides isotherme 2D asymptotic-preserving (AP).
///
/// Instancie `TwoFluidAP2D<GeometricMG>` (transport Rusanov des deux especes + Lorentz
/// implicite + Poisson reformule par multigrille). Le terme raide (frequence plasma) est
/// integre implicitement : le schema reste stable et consistant meme quand dt*omega_pe >> 1,
/// la ou un schema explicite exploserait. L'elliptique etant on-device, la facade se compile
/// telle quelle pour le GPU sous -DADC_USE_KOKKOS=ON (backend herite de la cible `adc`).
///
/// @note Integrateur SUR MESURE, non composable bloc a bloc comme `System` (la stabilisation AP
///       couple la raideur au pas de temps DANS l'elliptique). Ce n'est donc PAS un scenario de
///       l'API Python publique : il n'est pas reexporte par le paquet `adc`. Sa methode reste
///       compilee dans le module prive `_adc` sous le nom `_adc._TwoFluidAP` (echappatoire interne,
///       hors contrat d'API stable).

namespace adc {

/// Parametres du solveur deux-fluides AP.
struct TwoFluidAPConfig {
  int n = 64;                      ///< cellules par direction
  double L = 6.283185307179586;    ///< taille du domaine (2*pi)
  double cse2 = 1.0;               ///< vitesse du son electronique au carre
  double csi2 = 0.04;              ///< vitesse du son ionique au carre
  double omega_pe = 5.0;           ///< frequence plasma electronique (echelle raide)
  double omega_pi = 1.0;           ///< frequence plasma ionique
  bool stabilize = true;           ///< schema AP (Poisson reformule) ; false = non stabilise
  double eps = 1e-3;               ///< amplitude de la perturbation cosinus initiale
  bool upwind_continuity = false;  ///< flux de masse Rusanov (anti-Gibbs) au lieu de centre
  double omega_ce = 0.0;           ///< frequence cyclotron electronique (0 = pas de B)
  double omega_ci = 0.0;           ///< frequence cyclotron ionique
};

/// Solveur deux-fluides isotherme asymptotic-preserving (electrons + ions + Poisson).
class TwoFluidAPSolver {
 public:
  explicit TwoFluidAPSolver(const TwoFluidAPConfig& cfg = {});
  ~TwoFluidAPSolver();
  TwoFluidAPSolver(TwoFluidAPSolver&&) noexcept;
  TwoFluidAPSolver& operator=(TwoFluidAPSolver&&) noexcept;

  void step(double dt);               ///< un pas AP (Lorentz implicite + Poisson reformule)
  void advance(double dt, int nsteps);

  int nx() const;
  double mass_e() const;  ///< masse electronique totale (conservee)
  double mass_i() const;  ///< masse ionique totale
  double max_charge() const;  ///< max|n_i - n_e| (ecart a la quasi-neutralite)
  double max_dev() const;     ///< max|n_e - n0| (borne en regime AP raide)
  std::vector<double> density_e() const;  ///< n_e (n x n), row-major, copie
  std::vector<double> density_i() const;  ///< n_i (n x n), row-major, copie

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace adc
