#pragma once

#include <adc/mesh/physical_bc.hpp>  // BCRec
#include <adc/runtime/model_spec.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

/// @file
/// @brief Composition mono-espece sur AMR a l'execution : le pendant raffine de System.
///
/// Un bloc (une espece, decrite par une ModelSpec de briques generiques) porte sur une
/// hierarchie AMR (grossier + un niveau fin suivi par regrid, reflux conservatif). Comme
/// System mais sur grille adaptative. Le coeur ne nomme aucun scenario.
///
/// @note Un seul bloc (AmrCouplerMP est mono-modele) ; deux niveaux (ratio 2) ; traitement
///       temporel explicite (la source du modele est appliquee par le pas AMR).

namespace adc {

/// Maillage et cadence AMR (parametres physiques par bloc, dans la ModelSpec).
struct AmrSystemConfig {
  int n = 128;            ///< cellules du niveau grossier par direction
  double L = 1.0;         ///< taille du domaine carre [0,L]^2
  int regrid_every = 20;  ///< re-raffinement tous les N pas (0 = jamais apres l'init)
  bool periodic = true;   ///< domaine periodique
  /// POLITIQUE D'OWNERSHIP du niveau grossier (cf. AmrCouplerMP::replicated_coarse).
  /// false (DEFAUT, historique) : grossier mono-box REPLIQUE sur tous les rangs. Le Poisson
  ///   grossier et le transport grossier sont REDONDANTS sur chaque GPU (zero communication,
  ///   meilleur MG geometrique) mais NE SCALENT PAS : seuls les patchs fins se repartissent.
  /// true (mode strong-scaling) : grossier MULTI-BOX (BoxArray::from_domain, taille de tuile
  ///   coarse_max_grid) REPARTI round-robin sur les rangs. Le Poisson grossier et le transport
  ///   grossier se distribuent (chaque rang ne porte que ses tuiles), ce qui leve la redondance
  ///   et permet le strong-scaling AMR. Le MG geometrique opere alors sur un grossier multi-box
  ///   (cf. geometric_mg.hpp) : convergence a mesurer (peut demander plus de cycles).
  bool distribute_coarse = false;
  /// Taille de tuile du grossier quand distribute_coarse=true (BoxArray::from_domain). 0 => n/2
  /// (decoupage minimal 2x2, le moins agressif pour le MG). Ignore si distribute_coarse=false.
  int coarse_max_grid = 0;
};

/// Parametres figes passes au build differe du chemin compile (add_compiled_model). Materialises
/// par AmrSystem au moment de ensure_built : la geometrie + les choix refine/poisson/density connus
/// a ce moment-la. Le header amr_dsl_block les consomme pour instancier AmrCouplerMP<Model>.
struct AmrBuildParams {
  int n = 128;
  double L = 1.0;
  int regrid_every = 20;
  double gamma = 1.4;
  int substeps = 1;
  bool recon_prim = false;            ///< recon == "primitive" (fige par add_compiled_model)
  double refine_threshold = 1e30;     ///< 1e30 => aucun raffinement
  BCRec poisson_bc;                   ///< CL Poisson grossier (resolue par set_poisson)
  std::function<bool(Real, Real)> wall;  ///< predicat paroi conductrice (vide = aucune)
  bool has_density = false;
  std::vector<double> density;        ///< densite initiale grossiere (composante 0), n*n
  bool distribute_coarse = false;     ///< grossier multi-box reparti (strong-scaling AMR)
  int coarse_max_grid = 0;            ///< taille de tuile du grossier reparti (0 => n/2)
};

/// Fermetures type-erased d'un bloc AMR compile, produites par amr_dsl_block::build_amr_compiled et
/// installees par AmrSystem::install_compiled. Symetrique des hooks std::function de AmrSystem::Impl.
struct AmrCompiledHooks {
  std::shared_ptr<void> coupler_holder;   ///< maintient en vie le AmrCouplerMP<Model>
  std::function<void(double)> step;       ///< un macro-pas (regrid periodique inclus)
  std::function<double()> max_speed;      ///< vitesse d'onde max (pas CFL)
  std::function<double()> mass;           ///< masse grossiere
  std::function<int()> n_patches;         ///< nombre de patchs fins
  std::function<std::vector<double>()> density;  ///< densite grossiere, n*n row-major
};

/// Bloc unique porte sur une hierarchie AMR, compose a l'execution.
class AmrSystem {
 public:
  explicit AmrSystem(const AmrSystemConfig& cfg);
  ~AmrSystem();
  AmrSystem(AmrSystem&&) noexcept;
  AmrSystem& operator=(AmrSystem&&) noexcept;

  /// Definit l'unique bloc porte sur l'AMR. Memes parametres de schema spatial que System
  /// (limiter x riemann x recon), appliques a chaque niveau/patch de la hierarchie.
  /// @param model   composition de briques (transport/source/elliptic + parametres)
  /// @param limiter "none" | "minmod" | "vanleer"
  /// @param riemann "rusanov" | "hllc" | "roe" (hllc/roe exigent un transport compressible)
  /// @param recon   "conservative" | "primitive" (variables reconstruites ; primitif plus
  ///                robuste pour Euler : positivite de rho et p)
  /// @param time    "explicit" uniquement (l'IMEX sur AMR n'est pas cable ici)
  /// @throws std::runtime_error si un bloc est deja defini ou si time != "explicit".
  void add_block(const std::string& name, const ModelSpec& model,
                 const std::string& limiter = "minmod",
                 const std::string& riemann = "rusanov",
                 const std::string& recon = "conservative",
                 const std::string& time = "explicit", int substeps = 1);

  /// Enregistre un bloc COMPILE (chemin add_compiled_model, header amr_dsl_block.hpp) : @p builder
  /// est une fermeture type-erased qui, recevant les AmrBuildParams figes au build paresseux, rend
  /// les AmrCompiledHooks d'un AmrCouplerMP<Model> concret. NE PAS appeler directement : passer par
  /// la fonction libre add_compiled_model(AmrSystem&, ...). @throws si un bloc est deja defini.
  void set_compiled_block(int ncomp, double gamma, int substeps,
                          std::function<AmrCompiledHooks(const AmrBuildParams&)> builder);

  /// Raffine les cellules ou la densite (composante 0) depasse @p threshold.
  void set_refinement(double threshold);

  /// Configure le Poisson grossier (cf. System::set_poisson).
  void set_poisson(const std::string& rhs = "charge_density",
                   const std::string& solver = "geometric_mg",
                   const std::string& bc = "auto", const std::string& wall = "none",
                   double wall_radius = 0.0);

  /// Fixe la densite initiale sur le niveau grossier (composante 0), n*n row-major.
  void set_density(const std::string& name, const std::vector<double>& rho);

  void step(double dt);  ///< un macro-pas AMR (regrid periodique inclus)
  void advance(double dt, int nsteps);
  /// Avance a dt = cfl * dx_grossier / vitesse d'onde max. @returns le dt utilise.
  double step_cfl(double cfl);

  int nx() const;
  double time() const;
  int n_patches();                ///< nombre de patchs fins courants
  double mass();                  ///< masse sur le grossier (conservee au reflux)
  std::vector<double> density();  ///< densite grossiere (composante 0), n*n row-major

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace adc
