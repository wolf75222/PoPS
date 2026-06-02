#pragma once

#include <memory>
#include <string>
#include <vector>

/// @file
/// @brief Composition mono-espece sur AMR a l'execution : le pendant raffine de System.
///
/// Un bloc (une espece) porte sur une hierarchie AMR (grossier + un niveau fin suivi par
/// regrid Berger-Rigoutsos, reflux conservatif). On choisit le modele, le schema spatial
/// (limiteur + flux), le critere de raffinement et le Poisson, comme pour System mais sur
/// grille adaptative. Le calcul reste en C++ compile (AmrCouplerMP) ; Python compose.
///
/// Remplace l'ancienne facade specialisee DiocotronAmrSolver : le diocotron sur AMR se
/// compose ici generiquement, sans solveur dedie.
///
/// @note Un seul bloc (AmrCouplerMP est mono-modele) ; deux niveaux (un raffinement, ratio
///       2). Traitement temporel explicite (la source du modele est appliquee par le pas
///       AMR) ; l'IMEX sur AMR n'est pas cable ici.

namespace adc {

/// Parametres d'un AmrSystem. Champs par modele lus selon le tag (cf. SystemConfig).
struct AmrSystemConfig {
  int n = 128;            ///< cellules du niveau grossier par direction (domaine n x n)
  double L = 1.0;         ///< taille du domaine carre [0,L]^2
  double B0 = 1.0;        ///< champ magnetique ("diocotron")
  double n_i0 = 0.0;      ///< fond ionique neutralisant ("diocotron")
  double alpha = 1.0;     ///< constante de couplage Poisson ("diocotron")
  double gamma = 1.4;     ///< indice adiabatique ("electron_euler", "euler_poisson")
  double cs2 = 0.5;       ///< vitesse du son au carre, isotherme ("ion_isothermal")
  double four_pi_G = 1.0; ///< intensite de couplage ("euler_poisson")
  double rho0 = 1.0;      ///< fond neutralisant ("euler_poisson")
  int regrid_every = 20;  ///< re-raffinement tous les N pas (0 = jamais apres l'init)
  bool periodic = true;   ///< domaine periodique
};

/// Bloc unique porte sur une hierarchie AMR, compose a l'execution.
class AmrSystem {
 public:
  explicit AmrSystem(const AmrSystemConfig& cfg);
  ~AmrSystem();
  AmrSystem(AmrSystem&&) noexcept;
  AmrSystem& operator=(AmrSystem&&) noexcept;

  /// Definit l'unique bloc (une espece) porte sur l'AMR.
  /// @param charge   signe dans elliptic_rhs des modeles de fluide charge
  /// @param limiter  "none" | "minmod" | "vanleer"
  /// @param flux     "rusanov" | "hllc" (hllc exige un modele Euler a 4 variables)
  /// @param time     "explicit" uniquement (l'IMEX sur AMR n'est pas cable ici)
  /// @param substeps sous-pas du bloc par macro-pas
  /// @throws std::runtime_error si un bloc est deja defini, si le modele/flux est invalide,
  ///         ou si time != "explicit".
  void add_block(const std::string& name, const std::string& model, double charge,
                 const std::string& limiter = "minmod",
                 const std::string& flux = "rusanov",
                 const std::string& time = "explicit", int substeps = 1);

  /// Raffine les cellules ou la densite (composante 0) depasse @p threshold.
  void set_refinement(double threshold);

  /// Configure le Poisson grossier (cf. System::set_poisson).
  void set_poisson(const std::string& rhs = "charge_density",
                   const std::string& solver = "geometric_mg",
                   const std::string& bc = "auto", const std::string& wall = "none",
                   double wall_radius = 0.0);

  /// Fixe la densite initiale sur le niveau grossier (composante 0), n*n row-major ; les
  /// autres composantes posees a l'equilibre au repos. Le niveau fin est reconstruit par
  /// le premier regrid.
  void set_density(const std::string& name, const std::vector<double>& rho);

  void step(double dt);  ///< un macro-pas AMR (regrid periodique inclus)
  void advance(double dt, int nsteps);
  /// Avance a dt = cfl * dx_grossier / vitesse d'onde max. @returns le dt utilise.
  double step_cfl(double cfl);

  int nx() const;                 ///< n (niveau grossier)
  double time() const;
  int n_patches();                ///< nombre de patchs fins courants
  double mass();                  ///< masse sur le grossier (conservee au reflux)
  std::vector<double> density();  ///< densite grossiere (composante 0), n*n row-major

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace adc
