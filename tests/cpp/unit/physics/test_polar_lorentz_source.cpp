// FORCE DE LORENTZ MAGNETIQUE v x B_z NATIVE pour le fluide isotherme POLAIRE (3 var). Verifie la
// brique de SOURCE MagneticLorentzForce (physics/source.hpp) et sa composition avec l'electrostatique
// (CompositeSource<PotentialForce, MagneticLorentzForce>), cablees dans le transport polaire via
// dispatch_source (model_factory.hpp) -- la force que le demonstrateur diocotron polaire DEVAIT
// CONTOURNER (la geometrie polaire ne cablait que none/potential/gravity ; le faux ressort centrifuge
// remplacait la VRAIE force magnetique).
//
// PHYSIQUE (revue #212) : en base locale orthonormee (e_r, e_theta), B = B_z z_hat, la force de Lorentz
// sur la quantite de mouvement vaut rho (v x B) = (+B_z m_theta, -B_z m_r), m_r=u[1], m_theta=u[2].
// C'est ALGEBRIQUE (ponctuel, pas de derivee) et INVARIANT par orientation du repere : identique a la
// forme cartesienne (+B_z m_y, -B_z m_x). Regime EXPLICITE (le raide passe par le Schur, #212).
//
//   (A) ALGEBRE PONCTUELLE de la brique : MagneticLorentzForce.apply rend EXACTEMENT
//       (0, +qom B_z m_theta, -qom B_z m_r), composante energie nulle (v x B perpendiculaire a v ->
//       travail nul) ; CompositeSource SOMME bien electrostatique + Lorentz ; n_aux = 4 (lit B_z) et
//       CompositeModel le remonte. PREUVE que la brique encode la bonne formule.
//
//   (B) GIRATION CYCLOTRON SANS TRAVAIL : sous la SEULE force de Lorentz (omega_c = qom B_z), une
//       cellule de quantite de mouvement tourne a |m| CONSTANT. On integre dm/dt = (omega_c m_theta,
//       -omega_c m_r) (RK4 sur la brique) : la norme |m| est conservee a ~machine et l'angle tourne de
//       -omega_c t. PREUVE que la force est une rotation pure (cyclotron) -- pas un faux ressort.
//
//   (C) DIOCOTRON POLAIRE NATIF : fluide isotherme polaire (assemble_rhs_polar) avec la source COMPOSEE
//       potential_magnetic (electrostatique -rho grad phi + Lorentz q v x B_z), B_z constant fourni par
//       l'aux, potentiel d'equilibre + perturbation de mode m. On avance en temps (SSPRK3) et on verifie :
//       (C1) AUCUN NaN ; (C2) masse conservee a ~machine (paroi radiale -- la force de Lorentz n'agit
//       QUE sur la qdm, composante 0 nulle, elle ne cree ni ne detruit de masse) ; (C3) l'amplitude du
//       mode azimutal m de la perturbation CROIT (instabilite portee par la VRAIE force magnetique).
//   (C') CONTROLE : le MEME run avec B_z = 0 (Lorentz inactif) ne developpe PAS la meme croissance ->
//       isole le role de la force magnetique native.
//
// Host / Serial-safe (UNE box, n_ranks()==1 : non enregistre MPI, comme les autres tests polaires).

#include <gtest/gtest.h>

#include <pops/core/state/state.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/numerics/spatial/operators/polar_operator.hpp>
#include <pops/numerics/time/integrators/time_steppers.hpp>
#include <pops/physics/bricks/bricks.hpp>  // CompositeModel + briques source/hyperbolique/elliptique

#include <cmath>
#include <vector>

using namespace pops;

static constexpr double kPiL = 3.14159265358979323846;
static constexpr double kRmin = 0.30;
static constexpr double kRmax = 1.00;
static constexpr double kCs2 = 0.7;

// =====================================================================================
// (A) ALGEBRE PONCTUELLE de la brique + composition + propagation du canal aux.
// =====================================================================================
using CSrc = CompositeSource<PotentialForce, MagneticLorentzForce>;
using CModel = CompositeModel<IsothermalFluxPolar, CSrc, ChargeDensity>;

// n_aux de la brique magnetisee : lit B_z (canal extra, indice 3) -> 4. CompositeSource propage
// le canal aux : max(3 electrostatique, 4 Lorentz) = 4. CompositeModel remonte n_aux au systeme
// (canal B_z dimensionne).
static_assert(MagneticLorentzForce::n_aux == 4,
              "MagneticLorentzForce doit declarer n_aux = 4 (lit B_z)");
static_assert(CSrc::n_aux == 4, "CompositeSource doit propager n_aux = 4 (sous-brique magnetisee)");
static_assert(CModel::n_aux == 4, "CompositeModel doit remonter n_aux = 4 (source magnetisee)");

TEST(test_polar_lorentz_source, MagneticLorentzForceFormulaIsExact) {
  // Formule exacte : (0, +qom B_z m_theta, -qom B_z m_r), energie nulle.
  const Real qom = Real(-1.5), Bz = Real(2.0);
  StateVec<3> u{};
  u[0] = Real(1.3);   // rho
  u[1] = Real(0.7);   // m_r
  u[2] = Real(-0.4);  // m_theta
  Aux a{};
  a.B_z = Bz;
  const MagneticLorentzForce lor{qom};
  const StateVec<3> s = lor.apply(u, a);
  const Real ex1 = qom * Bz * u[2];   // +qom B_z m_theta
  const Real ex2 = -qom * Bz * u[1];  // -qom B_z m_r
  const double e0 = std::fabs(s[0]);
  const double e1 = std::fabs(s[1] - ex1);
  const double e2 = std::fabs(s[2] - ex2);
  EXPECT_TRUE(e0 <= 1e-14 && e1 <= 1e-14 && e2 <= 1e-14)
      << "formule Lorentz incorrecte : s=(" << s[0] << ", " << s[1] << ", " << s[2]
      << ") attendu (0, " << ex1 << ", " << ex2 << ")";
}

TEST(test_polar_lorentz_source, MagneticLorentzForceDoesNoWork) {
  // Travail nul : F . v = s[1] v_r + s[2] v_theta = 0 (perpendiculaire).
  const Real qom = Real(-1.5), Bz = Real(2.0);
  StateVec<3> u{};
  u[0] = Real(1.3);
  u[1] = Real(0.7);
  u[2] = Real(-0.4);
  Aux a{};
  a.B_z = Bz;
  const MagneticLorentzForce lor{qom};
  const StateVec<3> s = lor.apply(u, a);
  const Real vr = u[1] / u[0], vth = u[2] / u[0];
  const double work = std::fabs(s[1] * vr + s[2] * vth);
  EXPECT_TRUE(work <= 1e-14) << "la force de Lorentz fait un travail non nul : " << work;
}

TEST(test_polar_lorentz_source, CompositeSourceSumsElectrostaticAndLorentz) {
  const Real qom = Real(-1.5), Bz = Real(2.0);
  StateVec<3> u{};
  u[0] = Real(1.3);
  u[1] = Real(0.7);
  u[2] = Real(-0.4);
  Aux a{};
  a.B_z = Bz;
  a.grad_x = Real(0.9);
  a.grad_y = Real(-0.2);
  const MagneticLorentzForce lor{qom};
  const PotentialForce es{qom};
  const StateVec<3> s_es = es.apply(u, a);
  const StateVec<3> s_lor = lor.apply(u, a);
  const CSrc comp{es, lor};
  const StateVec<3> s_comp = comp.apply(u, a);
  double ecomp = 0.0;
  for (int c = 0; c < 3; ++c)
    ecomp = std::max(ecomp, std::fabs(s_comp[c] - (s_es[c] + s_lor[c])));
  EXPECT_TRUE(ecomp <= 1e-14) << "CompositeSource ne somme pas les deux forces : ecart max="
                              << ecomp;
}

// =====================================================================================
// (B) GIRATION CYCLOTRON SANS TRAVAIL : dm/dt = (omega_c m_theta, -omega_c m_r), |m| constant.
// =====================================================================================
// Pipeline avec etat (integration RK4 pas a pas) : un seul TEST natif, assertions inline aux 2
// points de verification (norme conservee, sens de rotation).
TEST(test_polar_lorentz_source, GyrationConservesMomentumNorm) {
  const Real qom = Real(1.0), Bz = Real(3.0);
  const Real omega_c = qom * Bz;
  const MagneticLorentzForce lor{qom};
  Aux a{};
  a.B_z = Bz;

  StateVec<3> u{};
  u[0] = Real(1.0);  // rho (fixe : la masse n'est pas touchee par la qdm-source)
  u[1] = Real(1.0);  // m_r
  u[2] = Real(0.0);  // m_theta
  const double m0 = std::sqrt(double(u[1]) * u[1] + double(u[2]) * u[2]);

  // RK4 sur dm/dt = source (la composante 0 reste a rho : source[0]=0). t parcourt un demi-tour.
  const double T = kPiL / std::fabs(double(omega_c));  // demi-periode cyclotron
  const int n = 4000;
  const Real h = Real(T / n);
  auto rhs = [&](const StateVec<3>& s) { return lor.apply(s, a); };
  for (int k = 0; k < n; ++k) {
    const StateVec<3> k1 = rhs(u);
    const StateVec<3> k2 = rhs(u + (Real(0.5) * h) * k1);
    const StateVec<3> k3 = rhs(u + (Real(0.5) * h) * k2);
    const StateVec<3> k4 = rhs(u + h * k3);
    u = u + (h / Real(6)) * (k1 + Real(2) * k2 + Real(2) * k3 + k4);
  }
  const double m1 = std::sqrt(double(u[1]) * u[1] + double(u[2]) * u[2]);
  const double dmag = std::fabs(m1 - m0) / m0;
  // Apres un demi-tour : m doit avoir tourne de -omega_c T = -pi (rotation horaire si omega_c>0) :
  // m_r -> -1, m_theta -> 0. On verifie |m| conserve + le sens de rotation.
  EXPECT_TRUE(dmag <= 1e-8) << "|m| non conserve (la force devrait etre une rotation pure) : m0="
                            << m0 << " m1=" << m1 << " ecart_relatif=" << dmag;
  EXPECT_TRUE(double(u[1]) < -0.99)
      << "m_r n'a pas tourne de ~pi : m_final=(" << double(u[1]) << ", " << double(u[2]) << ")";
}

// =====================================================================================
// (C) DIOCOTRON POLAIRE NATIF : fluide isotherme + source COMPOSEE potential_magnetic.
// =====================================================================================
// Source du modele : on construit la brique COMPOSEE EXACTEMENT comme dispatch_source le ferait pour
// m.source == "potential_magnetic" (electrostatique q/m + Lorentz q/m, meme espece). On la passe a un
// CompositeModel<IsothermalFluxPolar, CompositeSource, ChargeDensity>. C'est le chemin de PRODUCTION.
using DiocotronSrc = CompositeSource<PotentialForce, MagneticLorentzForce>;
using DiocotronModel = CompositeModel<IsothermalFluxPolar, DiocotronSrc, ChargeDensity>;

static DiocotronModel make_diocotron_model(double qom) {
  DiocotronModel model{};
  model.hyp.cs2 = kCs2;
  model.src = DiocotronSrc{PotentialForce{Real(qom)}, MagneticLorentzForce{Real(qom)}};
  model.ell = ChargeDensity{Real(1.0)};
  return model;
}

// Profil de densite d'equilibre : anneau lisse strictement positif (couche de charge).
static double base_rho(double r) {
  const double x = (r - kRmin) / (kRmax - kRmin);
  return 1.0 + 0.6 * std::sin(kPiL * x);  // s'annule en pente aux parois, max au milieu
}

// Potentiel d'aux FROZEN : phi d'equilibre radial + perturbation de mode m (declenche le diocotron).
// grad_r = d phi/dr (aux[1]), grad_theta = (1/r) d phi/d theta (aux[2]), B_z = aux[3].
static constexpr int kModeC = 4;
static void fill_aux(MultiFab& aux, const PolarGeometry& g, double Bz, double pert) {
  Array4 a = aux.fab(0).array();
  const Box2D gb = aux.fab(0).grown_box();
  for (int j = gb.lo[1]; j <= gb.hi[1]; ++j) {
    const double th = g.theta_cell(j);
    for (int i = gb.lo[0]; i <= gb.hi[0]; ++i) {
      const double r = g.r_cell(i);
      const double x = (r - kRmin) / (kRmax - kRmin);
      // phi(r, theta) = phi_r(r) + pert * env(r) cos(m theta), phi_r d'equilibre confinant.
      const double env = std::sin(kPiL * x);  // enveloppe radiale (nulle aux parois)
      const double denv = (kPiL / (kRmax - kRmin)) * std::cos(kPiL * x);
      const double phir = 0.5 * x * x;  // potentiel radial monotone
      const double dphir = (1.0 / (kRmax - kRmin)) * x;
      a(i, j, 0) = phir + pert * env * std::cos(kModeC * th);    // phi
      a(i, j, 1) = dphir + pert * denv * std::cos(kModeC * th);  // grad_r
      a(i, j, 2) =
          pert * env * (-kModeC * std::sin(kModeC * th)) / r;  // grad_theta = (1/r) d phi/d theta
      a(i, j, 3) = Bz;                                         // B_z constant
    }
  }
}

static void fill_state(MultiFab& U, const PolarGeometry& g) {
  Array4 u = U.fab(0).array();
  const Box2D gb = U.fab(0).grown_box();
  for (int j = gb.lo[1]; j <= gb.hi[1]; ++j)
    for (int i = gb.lo[0]; i <= gb.hi[0]; ++i) {
      const double r = g.r_cell(i);
      const double rho = base_rho(r);
      u(i, j, 0) = rho;
      u(i, j, 1) = 0.0;  // rho v_r = 0
      u(i, j, 2) = 0.0;  // rho v_theta = 0 (mis en mouvement par la force de Lorentz / E)
    }
}

static double total_mass(const MultiFab& U, const PolarGeometry& g, const Box2D& dom) {
  sync_host();
  const ConstArray4 u = U.fab(0).const_array();
  const double dr = g.dr(), dth = g.dtheta();
  double m = 0.0;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      m += u(i, j, 0) * g.r_cell(i) * dr * dth;
  return m;
}

// Amplitude du mode azimutal m du champ radial de quantite de mouvement m_r(theta), moyennee en r :
// A_m = sqrt( (sum_th m_r cos m th)^2 + (sum_th m_r sin m th)^2 ) normalisee. Croit avec l'instabilite.
static double mode_amplitude(const MultiFab& U, const Box2D& dom) {
  sync_host();
  const ConstArray4 u = U.fab(0).const_array();
  const int nth = dom.hi[1] - dom.lo[1] + 1;
  double cc = 0.0, ss = 0.0, cnt = 0.0;
  for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j) {
      const double th = 2.0 * kPiL * (j - dom.lo[1] + 0.5) / nth;
      const double mr = u(i, j, 1);  // quantite de mouvement radiale (reponse a la force)
      cc += mr * std::cos(kModeC * th);
      ss += mr * std::sin(kModeC * th);
      cnt += 1.0;
    }
  return std::sqrt(cc * cc + ss * ss) / cnt;
}

static bool has_nan(const MultiFab& U, const Box2D& dom) {
  sync_host();
  const ConstArray4 u = U.fab(0).const_array();
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      for (int c = 0; c < 3; ++c)
        if (!std::isfinite(u(i, j, c)))
          return true;
  return false;
}

struct DiocoResult {
  bool nan;
  double mass_rel;
  double amp0, amp1;
};

static DiocoResult run_diocotron(double Bz) {
  const int nr = 48, nth = 96;
  Box2D dom = Box2D::from_extents(nr, nth);
  PolarGeometry g{dom, kRmin, kRmax};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());

  BCRec bc;
  bc.xlo = bc.xhi = BCType::Foextrap;  // r physique
  bc.ylo = bc.yhi = BCType::Periodic;  // theta periodique

  const int ng = Weno5::n_ghost;
  MultiFab U(ba, dm, 3, ng);
  MultiFab aux(ba, dm, 4, ng);  // phi, grad_r, grad_theta, B_z (la source magnetisee lit B_z)
  U.set_val(0.0);
  aux.set_val(0.0);

  const double qom = -1.0;  // espece chargee (signe inclus)
  DiocotronModel model = make_diocotron_model(qom);

  const double pert = 0.05;
  fill_aux(aux, g, Bz, pert);
  fill_ghosts(aux, dom, bc);
  fill_state(U, g);

  const double m0 = total_mass(U, g, dom);
  const double dr = g.dr();
  const double ds_min = std::min(dr, kRmin * g.dtheta());
  const double vmax = 1.5 + std::sqrt(kCs2);  // borne large (la qdm grandit)
  const double dt = 0.15 * ds_min / vmax;
  const int nsteps = 60;

  DiocoResult res{};
  // Amplitude apres un court transitoire (laisse la force etablir une reponse), puis a la fin.
  int probe0 = 8;
  for (int s = 0; s < nsteps; ++s) {
    SSPRK3Step{}.take_step(
        [&](MultiFab& stage, MultiFab& R) {
          fill_ghosts(stage, dom, bc);
          // wall_radial=true : paroi solide -> masse conservee a la machine (la force de Lorentz
          // n'agit que sur la qdm, composante 0 nulle).
          assemble_rhs_polar<Weno5, RusanovFlux>(model, stage, aux, g, R, /*recon_prim=*/true,
                                                 /*wall_radial=*/true);
        },
        U, dt);
    if (s + 1 == probe0)
      res.amp0 = mode_amplitude(U, dom);
  }
  res.amp1 = mode_amplitude(U, dom);
  res.nan = has_nan(U, dom);
  const double m1 = total_mass(U, g, dom);
  res.mass_rel = std::fabs(m1 - m0) / std::fabs(m0);
  return res;
}

// =====================================================================================
// (C) DIOCOTRON POLAIRE NATIF : les checks C1/C2/C3 portent sur le MEME run "on" (B_z=2.5), le
// controle C' compare aux run "off" (B_z=0). Les deux runs sont couteux (60 pas SSPRK3 chacun) :
// une fixture avec SetUpTestSuite les calcule UNE fois pour toute la suite de tests (C).
// =====================================================================================
class PolarDiocotronNative : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    on_ = new DiocoResult(run_diocotron(/*Bz=*/2.5));
    off_ = new DiocoResult(run_diocotron(/*Bz=*/0.0));
  }
  static void TearDownTestSuite() {
    delete on_;
    delete off_;
    on_ = nullptr;
    off_ = nullptr;
  }

  static DiocoResult* on_;
  static DiocoResult* off_;
};
DiocoResult* PolarDiocotronNative::on_ = nullptr;
DiocoResult* PolarDiocotronNative::off_ = nullptr;

// (C1) pas de NaN dans le run avec Lorentz natif.
TEST_F(PolarDiocotronNative, NativeRunHasNoNan) {
  EXPECT_TRUE(!on_->nan) << "NaN dans le run avec Lorentz natif";
}

// (C2) masse conservee a ~machine (paroi radiale) dans le run avec Lorentz natif.
TEST_F(PolarDiocotronNative, NativeRunConservesMass) {
  EXPECT_TRUE(on_->mass_rel <= 1e-12) << "masse non conservee : " << on_->mass_rel << " > 1e-12";
}

// (C3) le mode croit (la force magnetique native met le fluide en mouvement et l'amplifie).
TEST_F(PolarDiocotronNative, NativeRunGrowsAzimuthalMode) {
  EXPECT_TRUE(on_->amp1 > on_->amp0 && on_->amp1 > 1e-6)
      << "le mode azimutal ne croit pas avec la force de Lorentz native : amp0=" << on_->amp0
      << " amp1=" << on_->amp1;
}

// (C') controle : B_z = 0 -> la force de Lorentz est INACTIVE. La reponse magnetique disparait :
// l'amplitude finale du mode m_r doit etre NETTEMENT plus faible (la giration cyclotron, seule
// capable de convertir grad phi azimutal en mouvement RADIAL coherent du mode, est absente).
TEST_F(PolarDiocotronNative, ControlRunWithoutLorentzHasNoNanAndWeakerGrowth) {
  EXPECT_TRUE(!off_->nan) << "NaN dans le run controle";
  EXPECT_TRUE(on_->amp1 > 1.5 * off_->amp1)
      << "la croissance du mode n'est pas portee par la force magnetique : amp(B_z=2.5)="
      << on_->amp1 << " amp(B_z=0)=" << off_->amp1;
}
