// Newton de la source implicite GENERALISE (audit 2026-06, vague 2) : options (tolerances,
// damping, fail_policy) et diagnostics (cellule fautive / composante) -- preuves :
//  (1) NON-EULER MULTI-VARIABLES : un systeme de relaxation NON LINEAIRE 3 variables (aucun layout
//      rho/m/E, aucune pression) converge sous tolerance -- le solveur n'est pas hardcode Euler.
//      La solution verifie l'equation BE W = Un + dt*S(W) au residu pres.
//  (2) DAMPING : newton amorti (damping < 1) converge vers la MEME racine (plus d'iterations).
//  (3) PATHOLOGIE PROPRE : une source qui produit NaN sur UNE cellule -> fail_policy=throw leve
//      une erreur claire, le rapport identifie LA cellule fautive (i, j) et la composante.
//  (4) OBSERVATEUR PUR : avec defauts + diagnostics, W est BIT-IDENTIQUE au chemin historique.
#include <gtest/gtest.h>

#include <pops/core/state/state.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/time/integrators/implicit_stepper.hpp>
#include <pops/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <stdexcept>

using pops::Aux;
using pops::Real;

// Relaxation NON LINEAIRE 3 variables, sans aucun layout fluide (ni densite, ni pression) :
//   S0 = -k (u0 - u1 u2) ; S1 = -k (u1 - u0/2) ; S2 = -k u2^3.
struct StiffModel {
  using State = pops::StateVec<3>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 3;
  Real k = 200.0;
  POPS_HD State flux(const State&, const Aux&, int) const { return State{}; }
  POPS_HD Real max_wave_speed(const State&, const Aux&, int) const { return 0; }
  POPS_HD State source(const State& u, const Aux&) const {
    State s{};
    s[0] = -k * (u[0] - u[1] * u[2]);
    s[1] = -k * (u[1] - Real(0.5) * u[0]);
    s[2] = -k * u[2] * u[2] * u[2];
    return s;
  }
  POPS_HD Real elliptic_rhs(const State&) const { return 0; }
};

// StiffModel + JACOBIEN ANALYTIQUE exact (trait HasSourceJacobian, vague 3) : le Newton doit
// converger vers la MEME racine que les differences finies (l'equation BE est identique).
struct JacStiffModel : StiffModel {
  POPS_HD void source_jacobian(const State& u, const Aux&, Real (&J)[3][3]) const {
    J[0][0] = -k;
    J[0][1] = k * u[2];
    J[0][2] = k * u[1];
    J[1][0] = k * Real(0.5);
    J[1][1] = -k;
    J[1][2] = 0;
    J[2][0] = 0;
    J[2][1] = 0;
    J[2][2] = -Real(3) * k * u[2] * u[2];
  }
};

// Source PATHOLOGIQUE : sqrt(u0 - 10) -> NaN des que u0 < 10 (toutes nos cellules), sur la
// composante 1 SEULEMENT quand u0 < seuil bas (pour viser UNE cellule fautive).
struct NanModel {
  using State = pops::StateVec<3>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 3;
  POPS_HD State flux(const State&, const Aux&, int) const { return State{}; }
  POPS_HD Real max_wave_speed(const State&, const Aux&, int) const { return 0; }
  POPS_HD State source(const State& u, const Aux&) const {
    State s{};
    s[0] = -u[0];
    s[1] = u[0] < Real(0) ? std::sqrt(u[0]) : -u[1];  // u0 < 0 -> NaN sur la composante 1
    s[2] = -u[2];
    return s;
  }
  POPS_HD Real elliptic_rhs(const State&) const { return 0; }
};

static pops::MultiFab make_mf(const pops::BoxArray& ba, const pops::DistributionMapping& dm, int nc) {
  pops::MultiFab m(ba, dm, nc, 0);
  m.set_val(Real(0));
  return m;
}

// Copie de src vers dst sur les 3 composantes (idiome recopie dans chaque section).
static void copy3(const pops::MultiFab& src, pops::MultiFab& dst) {
  for (int li = 0; li < dst.local_size(); ++li) {
    pops::Array4 d = dst.fab(li).array();
    const pops::ConstArray4 s = src.fab(li).const_array();
    const pops::Box2D b = dst.box(li);
    for (int c = 0; c < 3; ++c)
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          d(i, j, c) = s(i, j, c);
  }
}

// pas de temps commun : k*dt = 10 (raide, un point-fixe explicite divergerait).
static constexpr Real kDt = 0.05;

// Fixture partageant la grille 4x4 mono-boite et l'etat initial U0 (meme grille/etat pour toutes
// les preuves de robustesse du Newton generalise). SetUpTestSuite : construit une fois par suite.
class NewtonRobustnessTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    dom_ = new pops::Box2D(pops::Box2D::from_extents(4, 4));
    ba_ = new pops::BoxArray(std::vector<pops::Box2D>{*dom_});
    dm_ = new pops::DistributionMapping(1, pops::n_ranks());
    aux_ = new pops::MultiFab(make_mf(*ba_, *dm_, pops::kAuxBaseComps));
    U0_ = new pops::MultiFab(make_mf(*ba_, *dm_, 3));
    for (int li = 0; li < U0_->local_size(); ++li) {
      pops::Array4 u = U0_->fab(li).array();
      const pops::Box2D b = U0_->box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
          u(i, j, 0) = 1.0 + 0.1 * i;
          u(i, j, 1) = -0.5 + 0.05 * j;
          u(i, j, 2) = 0.3;
        }
    }
  }
  static void TearDownTestSuite() {
    delete dom_;
    delete ba_;
    delete dm_;
    delete aux_;
    delete U0_;
    dom_ = nullptr;
    ba_ = nullptr;
    dm_ = nullptr;
    aux_ = nullptr;
    U0_ = nullptr;
  }

  static pops::Box2D* dom_;
  static pops::BoxArray* ba_;
  static pops::DistributionMapping* dm_;
  static pops::MultiFab* aux_;
  static pops::MultiFab* U0_;  // etat initial commun (verification BE, damping, jacobien, observateur)
};
pops::Box2D* NewtonRobustnessTest::dom_ = nullptr;
pops::BoxArray* NewtonRobustnessTest::ba_ = nullptr;
pops::DistributionMapping* NewtonRobustnessTest::dm_ = nullptr;
pops::MultiFab* NewtonRobustnessTest::aux_ = nullptr;
pops::MultiFab* NewtonRobustnessTest::U0_ = nullptr;

// (1) NON-EULER MULTI-VARIABLES : converge sous tolerance ; W verifie l'equation BE au residu pres.
TEST_F(NewtonRobustnessTest, stiff_multivariable_relaxation_converges_to_backward_euler_root) {
  StiffModel m;
  pops::MultiFab U = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, U);

  pops::NewtonOptions opts;
  opts.max_iters = 25;
  opts.rel_tol = 1e-12;
  opts.abs_tol = 1e-13;
  pops::NewtonReport rep;
  pops::backward_euler_source(m, *aux_, U, kDt, opts, {}, &rep);
  ASSERT_TRUE(rep.converged && rep.n_failed == 0)
      << "non converge (n_failed=" << rep.n_failed
      << ", res=" << static_cast<double>(rep.max_residual) << ")";

  // verification BE : W - Un - dt S(W) ~ 0 sur chaque cellule.
  double worst = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const pops::ConstArray4 w = U.fab(li).const_array();
    const pops::ConstArray4 un = U0_->fab(li).const_array();
    const pops::Box2D b = U.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        StiffModel::State W{};
        for (int c = 0; c < 3; ++c)
          W[c] = w(i, j, c);
        const StiffModel::State S = m.source(W, Aux{});
        for (int c = 0; c < 3; ++c)
          worst = std::fmax(worst, std::fabs(w(i, j, c) - un(i, j, c) - kDt * S[c]));
      }
  }
  EXPECT_TRUE(worst <= 1e-10) << "residu BE " << worst << " > 1e-10";
  std::printf(
      "OK  (1) relaxation non lineaire 3-var NON Euler : converge (res BE %.1e, iters max "
      "%.0f/25)\n",
      worst, static_cast<double>(rep.max_iters_used));
}

// (2) DAMPING : Newton amorti (damping < 1) converge vers la MEME racine (plus d'iterations).
TEST_F(NewtonRobustnessTest, damped_newton_converges_to_same_root_as_undamped) {
  StiffModel m;
  pops::MultiFab U = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, U);
  pops::NewtonOptions opts;
  opts.max_iters = 25;
  opts.rel_tol = 1e-12;
  opts.abs_tol = 1e-13;
  pops::NewtonReport rep;
  pops::backward_euler_source(m, *aux_, U, kDt, opts, {}, &rep);
  ASSERT_TRUE(rep.converged && rep.n_failed == 0) << "racine de reference non convergee";

  pops::MultiFab Ud = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, Ud);
  pops::NewtonOptions od = opts;
  od.damping = 0.5;
  od.max_iters = 80;
  pops::NewtonReport repd;
  pops::backward_euler_source(m, *aux_, Ud, kDt, od, {}, &repd);

  double dmax = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const pops::ConstArray4 a4 = U.fab(li).const_array();
    const pops::ConstArray4 b4 = Ud.fab(li).const_array();
    const pops::Box2D b = U.box(li);
    for (int c = 0; c < 3; ++c)
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          dmax = std::fmax(dmax, std::fabs(a4(i, j, c) - b4(i, j, c)));
  }
  EXPECT_TRUE(repd.converged) << "damping : non converge";
  EXPECT_TRUE(dmax <= 1e-8) << "damping : ecart racine " << dmax << " > 1e-8";
  std::printf("OK  (2) Newton amorti (damping=0.5) : meme racine (ecart %.1e), iters %.0f\n", dmax,
              static_cast<double>(repd.max_iters_used));
}

// (3) PATHOLOGIE PROPRE : source qui produit NaN sur UNE cellule -> fail_policy=throw leve une
// erreur claire, le rapport identifie LA cellule fautive (i, j) et la composante.
TEST_F(NewtonRobustnessTest, fail_policy_throw_reports_offending_cell_on_nan) {
  NanModel nm;
  pops::MultiFab Un2 = make_mf(*ba_, *dm_, 3);
  for (int li = 0; li < Un2.local_size(); ++li) {
    pops::Array4 u = Un2.fab(li).array();
    const pops::Box2D b = Un2.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        u(i, j, 0) = 1.0;  // sain partout...
        u(i, j, 1) = 0.2;
        u(i, j, 2) = 0.1;
      }
  }
  Un2.fab(0).array()(2, 3, 0) = -4.0;  // ...sauf la cellule (2, 3) : sqrt(-4) -> NaN composante 1
  pops::NewtonOptions opf;
  opf.fail_policy = pops::NewtonOptions::kFailThrow;
  pops::NewtonReport repf;
  bool threw = false;
  try {
    pops::backward_euler_source(nm, *aux_, Un2, 0.1, opf, {}, &repf);
  } catch (const std::runtime_error& e) {
    threw = true;
    std::printf("OK  (3) fail_policy=throw : %s\n", e.what());
  }
  ASSERT_TRUE(threw) << "pas de throw (n_failed=" << repf.n_failed << ")";
  EXPECT_TRUE(repf.n_failed >= 1) << "pas d'echec rapporte (n_failed=" << repf.n_failed << ")";
  EXPECT_EQ(repf.diagnostics.count("newton.fail_policy.throw"), 1u)
      << "fail_policy=throw non reporte comme evenement structure";
  EXPECT_TRUE(repf.failed_i == 2 && repf.failed_j == 3)
      << "cellule fautive (" << repf.failed_i << ", " << repf.failed_j << ") != (2, 3)";
  std::printf("OK  (3) cellule fautive identifiee (%g, %g), composante %g\n", repf.failed_i,
              repf.failed_j, repf.failed_comp);
}

// (3b) WARN : pas de stderr requis, l'evenement est dans NewtonReport.
TEST_F(NewtonRobustnessTest, fail_policy_warn_exposes_event_without_throwing) {
  NanModel nm;
  pops::MultiFab Unw = make_mf(*ba_, *dm_, 3);
  for (int li = 0; li < Unw.local_size(); ++li) {
    pops::Array4 u = Unw.fab(li).array();
    const pops::Box2D b = Unw.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        u(i, j, 0) = 1.0;
        u(i, j, 1) = 0.2;
        u(i, j, 2) = 0.1;
      }
  }
  Unw.fab(0).array()(2, 3, 0) = -4.0;
  pops::NewtonOptions opw;
  opw.fail_policy = pops::NewtonOptions::kFailWarn;
  pops::NewtonReport repw;
  pops::backward_euler_source(nm, *aux_, Unw, 0.1, opw, {}, &repw);
  EXPECT_EQ(repw.diagnostics.count("newton.fail_policy.warn"), 1u)
      << "fail_policy=warn non expose dans NewtonReport";
  EXPECT_TRUE(repw.n_failed >= 1) << "fail_policy=warn : n_failed=" << repw.n_failed;
  std::printf("OK  (3b) fail_policy=warn : evenement structure sans stderr\n");
}

// (4) OBSERVATEUR PUR : avec defauts + diagnostics, W est BIT-IDENTIQUE au chemin historique.
TEST_F(NewtonRobustnessTest, diagnostics_path_is_a_pure_observer) {
  StiffModel m;
  pops::MultiFab Ua = make_mf(*ba_, *dm_, 3), Ub = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, Ua);
  copy3(*U0_, Ub);

  pops::backward_euler_source(m, *aux_, Ua, kDt, 2);  // chemin historique (surcharge iters)
  pops::NewtonOptions odef;                           // defauts stricts
  pops::NewtonReport repo;
  pops::backward_euler_source(m, *aux_, Ub, kDt, odef, {}, &repo);  // instrumente, defauts

  for (int li = 0; li < Ua.local_size(); ++li) {
    const pops::ConstArray4 a4 = Ua.fab(li).const_array();
    const pops::ConstArray4 b4 = Ub.fab(li).const_array();
    const pops::Box2D b = Ua.box(li);
    for (int c = 0; c < 3; ++c)
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          EXPECT_EQ(a4(i, j, c), b4(i, j, c))
              << "diagnostics non observateur pur en (" << i << "," << j << ",c" << c << ")";
  }
  std::printf("OK  (4) diagnostics = observateur pur (W bit-identique au chemin historique)\n");
}

// (5) JACOBIEN ANALYTIQUE (vague 3) : meme racine que les differences finies.
TEST_F(NewtonRobustnessTest, analytic_jacobian_matches_finite_difference_root) {
  static_assert(!pops::HasSourceJacobian<StiffModel>, "StiffModel sans jacobien : FD historiques");
  static_assert(pops::HasSourceJacobian<JacStiffModel>, "JacStiffModel doit declarer le trait");

  StiffModel m;
  pops::MultiFab U = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, U);
  pops::NewtonOptions opts;
  opts.max_iters = 25;
  opts.rel_tol = 1e-12;
  opts.abs_tol = 1e-13;
  pops::NewtonReport rep;
  pops::backward_euler_source(m, *aux_, U, kDt, opts, {}, &rep);
  ASSERT_TRUE(rep.converged && rep.n_failed == 0) << "racine FD de reference non convergee";

  JacStiffModel jm;
  pops::MultiFab Uj = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, Uj);
  pops::NewtonReport repj;
  pops::backward_euler_source(jm, *aux_, Uj, kDt, opts, {}, &repj);

  double jdiff = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const pops::ConstArray4 a4 = U.fab(li).const_array();
    const pops::ConstArray4 b4 = Uj.fab(li).const_array();
    const pops::Box2D b = U.box(li);
    for (int c = 0; c < 3; ++c)
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          jdiff = std::fmax(jdiff, std::fabs(a4(i, j, c) - b4(i, j, c)));
  }
  EXPECT_TRUE(repj.converged) << "jacobien analytique : non converge";
  EXPECT_TRUE(jdiff <= 1e-9) << "jacobien analytique : ecart racine " << jdiff << " > 1e-9";
  std::printf("OK  (5) jacobien analytique : meme racine que les FD (ecart %.1e), iters %.0f\n",
              jdiff, static_cast<double>(repj.max_iters_used));
}
