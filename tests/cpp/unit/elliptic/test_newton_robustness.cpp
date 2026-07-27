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
#include <limits>
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

// Jacobien local exactement singulier pour dt=0.125 : dS0/du0 = 8, donc
// J00 = 1 - dt*dS0/du0 = 0 alors que le residu reste fini et non nul.
struct SingularModel {
  using State = pops::StateVec<3>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 3;
  POPS_HD State flux(const State&, const Aux&, int) const { return State{}; }
  POPS_HD Real max_wave_speed(const State&, const Aux&, int) const { return 0; }
  POPS_HD State source(const State& u, const Aux&) const {
    State s{};
    s[0] = Real(8) * u[0] + Real(1);
    s[1] = -u[1];
    s[2] = -u[2];
    return s;
  }
  POPS_HD void source_jacobian(const State&, const Aux&, Real (&J)[3][3]) const {
    J[0][0] = Real(8);
    J[0][1] = J[0][2] = Real(0);
    J[1][0] = J[1][2] = Real(0);
    J[1][1] = Real(-1);
    J[2][0] = J[2][1] = Real(0);
    J[2][2] = Real(-1);
  }
  POPS_HD Real elliptic_rhs(const State&) const { return 0; }
};

static pops::MultiFab make_mf(const pops::BoxArray& ba, const pops::DistributionMapping& dm,
                              int nc) {
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

static double max_difference3(const pops::MultiFab& left, const pops::MultiFab& right) {
  double difference = 0;
  for (int li = 0; li < left.local_size(); ++li) {
    const pops::ConstArray4 a = left.fab(li).const_array();
    const pops::ConstArray4 b = right.fab(li).const_array();
    const pops::Box2D box = left.box(li);
    for (int c = 0; c < 3; ++c)
      for (int j = box.lo[1]; j <= box.hi[1]; ++j)
        for (int i = box.lo[0]; i <= box.hi[0]; ++i)
          if (const double value = std::fabs(static_cast<double>(a(i, j, c) - b(i, j, c)));
              std::isfinite(value))
            difference = std::fmax(difference, value);
          else
            return std::numeric_limits<double>::infinity();
  }
  return difference;
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
  static pops::MultiFab*
      U0_;  // etat initial commun (verification BE, damping, jacobien, observateur)
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
  pops::MultiFab accepted = make_mf(*ba_, *dm_, 3);
  copy3(Un2, accepted);
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
  EXPECT_EQ(max_difference3(Un2, accepted), 0.0)
      << "un candidat invalide a ete publie dans l'etat accepte";
  EXPECT_TRUE(repf.n_failed >= 1) << "pas d'echec rapporte (n_failed=" << repf.n_failed << ")";
  EXPECT_EQ(repf.solve.status, pops::SolveStatus::kInvalidEvaluation);
  EXPECT_EQ(repf.solve.action, pops::SolveAction::kFailRun);
  EXPECT_EQ(repf.diagnostics.count("newton.outcome.fail_run"), 1u)
      << "FailRun non reporte comme consommation explicite";
  EXPECT_TRUE(repf.failed_i == 2 && repf.failed_j == 3)
      << "cellule fautive (" << repf.failed_i << ", " << repf.failed_j << ") != (2, 3)";
  EXPECT_EQ(repf.failed_comp, 1) << "la composante NaN initiale doit rester l'origine de l'echec";
  std::printf("OK  (3) cellule fautive identifiee (%g, %g), composante %g\n", repf.failed_i,
              repf.failed_j, repf.failed_comp);
}

// (3b) Les anciens modes none/warn ne peuvent plus publier silencieusement un candidat invalide.
TEST_F(NewtonRobustnessTest, legacy_warn_policy_fails_closed_without_publishing) {
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
  pops::MultiFab accepted = make_mf(*ba_, *dm_, 3);
  copy3(Unw, accepted);
  pops::NewtonOptions opw;
  opw.fail_policy = pops::NewtonOptions::kFailWarn;
  pops::NewtonReport repw;
  EXPECT_THROW(pops::backward_euler_source(nm, *aux_, Unw, 0.1, opw, {}, &repw),
               std::runtime_error);
  EXPECT_EQ(max_difference3(Unw, accepted), 0.0)
      << "fail_policy=warn a publie un candidat invalide";
  EXPECT_EQ(repw.solve.status, pops::SolveStatus::kInvalidEvaluation);
  EXPECT_EQ(repw.solve.action, pops::SolveAction::kFailRun);
  EXPECT_EQ(repw.diagnostics.count("newton.outcome.fail_run"), 1u)
      << "l'ancien warn doit etre consomme comme FailRun";
  EXPECT_TRUE(repw.n_failed >= 1) << "fail_policy=warn : n_failed=" << repw.n_failed;
  std::printf("OK  (3b) fail_policy=warn : FailRun sans publication\n");
}

// (3c) Une tolerance non satisfaite a l'epuisement du budget ne publie jamais le dernier itere.
TEST_F(NewtonRobustnessTest, iteration_limit_fails_closed_without_publishing) {
  StiffModel model;
  pops::MultiFab accepted = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, accepted);
  pops::MultiFab state = make_mf(*ba_, *dm_, 3);
  copy3(accepted, state);
  pops::NewtonOptions options;
  options.max_iters = 1;
  options.rel_tol = 1e-15;
  options.abs_tol = 1e-15;
  pops::NewtonReport report;

  EXPECT_THROW(pops::backward_euler_source(model, *aux_, state, kDt, options, {}, &report),
               std::runtime_error);
  EXPECT_EQ(report.solve.status, pops::SolveStatus::kIterationLimit);
  EXPECT_EQ(report.solve.action, pops::SolveAction::kFailRun);
  EXPECT_EQ(max_difference3(state, accepted), 0.0) << "le dernier itere non converge a ete publie";
}

// (3d) Un Jacobien singulier a son propre statut et ne publie pas l'itere partiel.
TEST_F(NewtonRobustnessTest, singular_jacobian_fails_closed_without_publishing) {
  SingularModel model;
  pops::MultiFab accepted = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, accepted);
  pops::MultiFab state = make_mf(*ba_, *dm_, 3);
  copy3(accepted, state);
  pops::NewtonOptions options;
  pops::NewtonReport report;

  EXPECT_THROW(pops::backward_euler_source(model, *aux_, state, 0.125, options, {}, &report),
               std::runtime_error);
  EXPECT_EQ(report.solve.status, pops::SolveStatus::kSingular);
  EXPECT_EQ(report.solve.action, pops::SolveAction::kFailRun);
  EXPECT_EQ(max_difference3(state, accepted), 0.0);
}

// (3e) La voie preparee permet au controleur temporel de consommer explicitement RejectAttempt.
TEST_F(NewtonRobustnessTest, prepared_failure_is_consumed_once_as_reject_attempt) {
  NanModel model;
  pops::MultiFab accepted = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, accepted);
  accepted.fab(0).array()(2, 3, 0) = -4.0;
  pops::MultiFab state = make_mf(*ba_, *dm_, 3);
  copy3(accepted, state);
  pops::NewtonOptions options;
  pops::NewtonReport diagnostics;

  auto outcome =
      pops::prepare_backward_euler_source(model, *aux_, state, 0.1, options, {}, &diagnostics);
  EXPECT_EQ(outcome.report().status, pops::SolveStatus::kInvalidEvaluation);
  const pops::SolveReport rejected =
      outcome.consume(pops::ImplicitSolveConsumption::kRejectAttempt);
  EXPECT_EQ(rejected.action, pops::SolveAction::kRejectAttempt);
  EXPECT_EQ(max_difference3(state, accepted), 0.0);
  EXPECT_EQ(diagnostics.diagnostics.count("newton.outcome.reject_attempt"), 1u);
  EXPECT_THROW(outcome.consume(pops::ImplicitSolveConsumption::kRejectAttempt), std::logic_error);
}

// (3f) Meme un succes reste prive jusqu'a sa consommation explicite, puis ne peut etre consomme deux
// fois.
TEST_F(NewtonRobustnessTest, prepared_success_publishes_only_on_single_accept) {
  StiffModel model;
  pops::MultiFab accepted = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, accepted);
  pops::MultiFab state = make_mf(*ba_, *dm_, 3);
  copy3(accepted, state);
  pops::NewtonOptions options;
  options.max_iters = 25;
  options.rel_tol = 1e-12;
  options.abs_tol = 1e-13;

  auto outcome = pops::prepare_backward_euler_source(model, *aux_, state, kDt, options);
  ASSERT_TRUE(outcome.report().solved_value_available());
  EXPECT_EQ(max_difference3(state, accepted), 0.0) << "le candidat est visible avant consommation";
  const pops::SolveReport solved = outcome.consume(pops::ImplicitSolveConsumption::kAccept);
  EXPECT_TRUE(solved.solved_value_available());
  EXPECT_GT(max_difference3(state, accepted), 0.0);
  EXPECT_THROW(outcome.consume(pops::ImplicitSolveConsumption::kAccept), std::logic_error);
}

// (4) Demander le rapport optionnel ne change pas un candidat sain ni sa publication.
TEST_F(NewtonRobustnessTest, optional_diagnostics_preserve_healthy_accepted_state) {
  StiffModel m;
  pops::MultiFab Ua = make_mf(*ba_, *dm_, 3), Ub = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, Ua);
  copy3(*U0_, Ub);

  pops::backward_euler_source(m, *aux_, Ua, kDt, 2);
  pops::NewtonOptions odef;
  pops::NewtonReport repo;
  pops::backward_euler_source(m, *aux_, Ub, kDt, odef, {}, &repo);

  for (int li = 0; li < Ua.local_size(); ++li) {
    const pops::ConstArray4 a4 = Ua.fab(li).const_array();
    const pops::ConstArray4 b4 = Ub.fab(li).const_array();
    const pops::Box2D b = Ua.box(li);
    for (int c = 0; c < 3; ++c)
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          EXPECT_EQ(a4(i, j, c), b4(i, j, c)) << "diagnostics ont modifie le candidat sain en ("
                                              << i << "," << j << ",c" << c << ")";
  }
  std::printf("OK  (4) diagnostics optionnels : meme etat accepte\n");
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
