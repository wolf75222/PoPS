// Fournisseur local non lineaire prepare : options, statuts distincts, publication atomique et
// diagnostics de cellule/composante -- preuves :
//  (1) NON-EULER MULTI-VARIABLES : un systeme de relaxation NON LINEAIRE 3 variables (aucun layout
//      rho/m/E, aucune pression) converge sous tolerance -- le solveur n'est pas hardcode Euler.
//      La solution verifie l'equation BE W = Un + dt*S(W) au residu pres.
//  (2) DAMPING : newton amorti (damping < 1) converge vers la MEME racine (plus d'iterations).
//  (3) PATHOLOGIE PROPRE : une source qui produit NaN sur UNE cellule echoue collectivement,
//      identifie la cellule et ne publie aucune mutation.
//  (4) OBSERVATEUR PUR : activer les diagnostics ne change aucun bit de la solution.
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

// (3) PATHOLOGIE PROPRE : source qui produit NaN sur UNE cellule -> FailRun, sans publication.
TEST_F(NewtonRobustnessTest, invalid_evaluation_reports_cell_and_preserves_live_state) {
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
  pops::MultiFab before = make_mf(*ba_, *dm_, 3);
  copy3(Un2, before);
  pops::NewtonOptions opf;
  pops::NewtonReport repf;
  repf.enabled = true;
  repf.converged = true;
  repf.max_residual = Real(7);
  repf.max_iters_used = Real(3);
  repf.n_failed = 2;
  repf.latest.mark_solved("accepted_sentinel");
  repf.diagnostics.record("accepted.sentinel", "test", "info", "must survive failure");
  const pops::NewtonReport accepted_report = repf;
  bool threw = false;
  std::string failure_message;
  try {
    pops::backward_euler_source(nm, *aux_, Un2, 0.1, opf, {}, &repf);
  } catch (const std::runtime_error& e) {
    threw = true;
    failure_message = e.what();
    std::printf("OK  (3) fail-closed : %s\n", e.what());
  }
  ASSERT_TRUE(threw) << "pas de throw (n_failed=" << repf.n_failed << ")";
  EXPECT_NE(failure_message.find("cell (2, 3), component 1"), std::string::npos);
  EXPECT_EQ(repf.enabled, accepted_report.enabled);
  EXPECT_EQ(repf.converged, accepted_report.converged);
  EXPECT_EQ(repf.max_residual, accepted_report.max_residual);
  EXPECT_EQ(repf.max_iters_used, accepted_report.max_iters_used);
  EXPECT_EQ(repf.n_failed, accepted_report.n_failed);
  EXPECT_EQ(repf.latest.status, accepted_report.latest.status);
  EXPECT_EQ(repf.latest.action, accepted_report.latest.action);
  EXPECT_EQ(repf.latest.reason, accepted_report.latest.reason);
  EXPECT_EQ(repf.diagnostics.events.size(), accepted_report.diagnostics.events.size());
  EXPECT_EQ(repf.diagnostics.count("accepted.sentinel"), 1u);
  EXPECT_EQ(repf.diagnostics.count("local_nonlinear.fail_run"), 0u);
  for (int li = 0; li < Un2.local_size(); ++li) {
    const pops::ConstArray4 actual = Un2.fab(li).const_array();
    const pops::ConstArray4 expected = before.fab(li).const_array();
    const pops::Box2D box = Un2.box(li);
    for (int c = 0; c < 3; ++c)
      for (int j = box.lo[1]; j <= box.hi[1]; ++j)
        for (int i = box.lo[0]; i <= box.hi[0]; ++i)
          EXPECT_EQ(actual(i, j, c), expected(i, j, c));
  }
  std::printf("OK  (3) cellule fautive dans l'outcome, diagnostics acceptes inchanges\n");
}

// (3b) Aucune politique "none"/"warn" ne peut publier un dernier itere.
TEST_F(NewtonRobustnessTest, failure_without_diagnostics_is_still_fail_closed) {
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
  const Real sentinel = Unw.fab(0).const_array()(2, 3, 0);
  EXPECT_THROW((void)pops::backward_euler_source(nm, *aux_, Unw, Real(0.1), pops::NewtonOptions{}),
               std::runtime_error);
  EXPECT_EQ(Unw.fab(0).const_array()(2, 3, 0), sentinel);
}

// (4) OBSERVATEUR PUR : avec ou sans diagnostics, W est BIT-IDENTIQUE.
TEST_F(NewtonRobustnessTest, diagnostics_path_is_a_pure_observer) {
  StiffModel m;
  pops::MultiFab Ua = make_mf(*ba_, *dm_, 3), Ub = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, Ua);
  copy3(*U0_, Ub);

  pops::NewtonOptions odef;
  (void)pops::backward_euler_source(m, *aux_, Ua, kDt, odef);
  pops::NewtonReport repo;
  (void)pops::backward_euler_source(m, *aux_, Ub, kDt, odef, {}, &repo);

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
  std::printf("OK  (4) diagnostics = observateur pur\n");
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

namespace {

struct SquareRootResidual {
  POPS_HD void operator()(const Real (&x)[1], Real (&residual)[1]) const {
    residual[0] = x[0] * x[0] - Real(2);
  }
};

struct SquareRootJacobian {
  POPS_HD bool operator()(const Real (&x)[1], Real (&jacobian)[1][1]) const {
    jacobian[0][0] = Real(2) * x[0];
    return true;
  }
};

struct ConstantResidual {
  POPS_HD void operator()(const Real (&)[1], Real (&residual)[1]) const { residual[0] = Real(1); }
};

struct ZeroJacobian {
  POPS_HD bool operator()(const Real (&)[1], Real (&jacobian)[1][1]) const {
    jacobian[0][0] = Real(0);
    return true;
  }
};

struct InvalidResidual {
  POPS_HD void operator()(const Real (&)[1], Real (&residual)[1]) const {
    residual[0] = std::numeric_limits<Real>::quiet_NaN();
  }
};

struct PositiveDomain {
  POPS_HD bool operator()(const Real (&x)[1], int* component) const {
    if (component != nullptr)
      *component = x[0] >= Real(0) ? -1 : 0;
    return x[0] >= Real(0);
  }
};

struct LinearResidual {
  POPS_HD void operator()(const Real (&x)[1], Real (&residual)[1]) const {
    residual[0] = x[0] - Real(1);
  }
};

struct WrongDirectionJacobian {
  POPS_HD bool operator()(const Real (&)[1], Real (&jacobian)[1][1]) const {
    jacobian[0][0] = Real(-1);
    return true;
  }
};

struct TinyScaledResidual {
  POPS_HD void operator()(const Real (&x)[1], Real (&residual)[1]) const {
    residual[0] = Real(1e-20) * (x[0] - Real(1));
  }
};

struct TinyScaledJacobian {
  POPS_HD bool operator()(const Real (&)[1], Real (&jacobian)[1][1]) const {
    jacobian[0][0] = Real(1e-20);
    return true;
  }
};

struct HugeJacobian {
  POPS_HD bool operator()(const Real (&)[1], Real (&jacobian)[1][1]) const {
    jacobian[0][0] = std::numeric_limits<Real>::max();
    return true;
  }
};

pops::PreparedLocalNonlinearControls scalar_controls() {
  pops::PreparedLocalNonlinearControls controls;
  controls.max_iterations = 20;
  controls.absolute_tolerance = Real(1e-13);
  return controls;
}

}  // namespace

TEST(PreparedLocalNonlinear, FiniteDifferenceAnalyticAndAdUseOneOutcomeContract) {
  const Real initial[1] = {Real(2)};
  const auto controls = scalar_controls();
  const auto finite_difference = pops::prepare_local_nonlinear_problem<1>(
      SquareRootResidual{}, pops::FiniteDifferenceLocalJacobian<1>{},
      pops::AcceptAllLocalCandidates<1>{}, controls);
  const auto analytic = pops::prepare_local_nonlinear_problem<1>(
      SquareRootResidual{},
      pops::AnalyticLocalJacobian<1, SquareRootJacobian>{SquareRootJacobian{}},
      pops::AcceptAllLocalCandidates<1>{}, controls);
  const auto automatic_differentiation = pops::prepare_local_nonlinear_problem<1>(
      SquareRootResidual{},
      pops::AutomaticDifferentiationLocalJacobian<1, SquareRootJacobian>{SquareRootJacobian{}},
      pops::AcceptAllLocalCandidates<1>{}, controls);

  const auto fd = pops::solve_prepared_local_nonlinear(finite_difference, initial);
  const auto exact = pops::solve_prepared_local_nonlinear(analytic, initial);
  const auto ad = pops::solve_prepared_local_nonlinear(automatic_differentiation, initial);
  ASSERT_TRUE(fd.solved());
  ASSERT_TRUE(exact.solved());
  ASSERT_TRUE(ad.solved());
  EXPECT_NEAR(fd.value[0], std::sqrt(Real(2)), 1e-11);
  EXPECT_NEAR(exact.value[0], fd.value[0], 1e-11);
  EXPECT_NEAR(ad.value[0], exact.value[0], 1e-13);
  EXPECT_EQ(initial[0], Real(2));
}

TEST(PreparedLocalNonlinear, PivotThresholdIsRelativeToTheScaledEquation) {
  auto controls = scalar_controls();
  controls.absolute_tolerance = Real(1e-30);
  const Real initial[1] = {Real(2)};
  const auto problem = pops::prepare_local_nonlinear_problem<1>(
      TinyScaledResidual{},
      pops::AnalyticLocalJacobian<1, TinyScaledJacobian>{TinyScaledJacobian{}},
      pops::AcceptAllLocalCandidates<1>{}, controls);
  const auto result = pops::solve_prepared_local_nonlinear(problem, initial);
  ASSERT_TRUE(result.solved());
  EXPECT_NEAR(result.value[0], Real(1), 1e-13);
}

TEST(PreparedLocalNonlinear, EveryFailureClassIsExplicitAndLeavesTheGuessUntouched) {
  Real initial[1] = {Real(10)};
  auto budget_controls = scalar_controls();
  budget_controls.max_iterations = 1;
  const auto budget_problem = pops::prepare_local_nonlinear_problem<1>(
      SquareRootResidual{}, pops::FiniteDifferenceLocalJacobian<1>{},
      pops::AcceptAllLocalCandidates<1>{}, budget_controls);
  EXPECT_EQ(pops::solve_prepared_local_nonlinear(budget_problem, initial).status,
            pops::LocalNonlinearStatus::kIterationLimit);

  const auto singular_problem = pops::prepare_local_nonlinear_problem<1>(
      ConstantResidual{}, pops::AnalyticLocalJacobian<1, ZeroJacobian>{ZeroJacobian{}},
      pops::AcceptAllLocalCandidates<1>{}, scalar_controls());
  EXPECT_EQ(pops::solve_prepared_local_nonlinear(singular_problem, initial).status,
            pops::LocalNonlinearStatus::kSingularJacobian);

  const auto invalid_problem = pops::prepare_local_nonlinear_problem<1>(
      InvalidResidual{}, pops::FiniteDifferenceLocalJacobian<1>{},
      pops::AcceptAllLocalCandidates<1>{}, scalar_controls());
  EXPECT_EQ(pops::solve_prepared_local_nonlinear(invalid_problem, initial).status,
            pops::LocalNonlinearStatus::kInvalidEvaluation);

  const auto overflow_problem = pops::prepare_local_nonlinear_problem<1>(
      LinearResidual{}, pops::AnalyticLocalJacobian<1, HugeJacobian>{HugeJacobian{}},
      pops::AcceptAllLocalCandidates<1>{}, scalar_controls(), Real(2), Real(1));
  EXPECT_EQ(pops::solve_prepared_local_nonlinear(overflow_problem, initial).status,
            pops::LocalNonlinearStatus::kInvalidEvaluation);

  Real inadmissible_initial[1] = {Real(-1)};
  const auto inadmissible_problem = pops::prepare_local_nonlinear_problem<1>(
      LinearResidual{}, pops::FiniteDifferenceLocalJacobian<1>{}, PositiveDomain{},
      scalar_controls());
  EXPECT_EQ(pops::solve_prepared_local_nonlinear(inadmissible_problem, inadmissible_initial).status,
            pops::LocalNonlinearStatus::kInadmissibleCandidate);

  auto safeguard_controls = scalar_controls();
  safeguard_controls.safeguard = pops::LocalSafeguardKind::kBacktrackingLineSearch;
  safeguard_controls.max_backtracks = 3;
  safeguard_controls.minimum_step = Real(0.01);
  Real safeguard_initial[1] = {Real(0)};
  const auto safeguard_problem = pops::prepare_local_nonlinear_problem<1>(
      LinearResidual{},
      pops::AnalyticLocalJacobian<1, WrongDirectionJacobian>{WrongDirectionJacobian{}},
      pops::AcceptAllLocalCandidates<1>{}, safeguard_controls);
  EXPECT_EQ(pops::solve_prepared_local_nonlinear(safeguard_problem, safeguard_initial).status,
            pops::LocalNonlinearStatus::kSafeguardFailure);

  const auto unsupported_problem = pops::prepare_local_nonlinear_problem<1>(
      LinearResidual{}, pops::UnsupportedLocalJacobian<1>{}, pops::AcceptAllLocalCandidates<1>{},
      scalar_controls());
  EXPECT_EQ(pops::solve_prepared_local_nonlinear(unsupported_problem, safeguard_initial).status,
            pops::LocalNonlinearStatus::kUnsupportedCapability);

  auto invalid_controls = scalar_controls();
  invalid_controls.absolute_tolerance = Real(0);
  invalid_controls.relative_tolerance = Real(0);
  invalid_controls.step_tolerance = Real(0);
  const auto invalid_controls_problem = pops::prepare_local_nonlinear_problem<1>(
      LinearResidual{}, pops::FiniteDifferenceLocalJacobian<1>{},
      pops::AcceptAllLocalCandidates<1>{}, invalid_controls);
  EXPECT_EQ(
      pops::solve_prepared_local_nonlinear(invalid_controls_problem, safeguard_initial).status,
      pops::LocalNonlinearStatus::kUnsupportedCapability);

  auto ignored_damping_controls = scalar_controls();
  ignored_damping_controls.initial_step = Real(0.5);
  const auto ignored_damping_problem = pops::prepare_local_nonlinear_problem<1>(
      LinearResidual{}, pops::FiniteDifferenceLocalJacobian<1>{},
      pops::AcceptAllLocalCandidates<1>{}, ignored_damping_controls);
  EXPECT_EQ(pops::solve_prepared_local_nonlinear(ignored_damping_problem, safeguard_initial).status,
            pops::LocalNonlinearStatus::kUnsupportedCapability);

  auto unknown_safeguard_controls = scalar_controls();
  unknown_safeguard_controls.safeguard = static_cast<pops::LocalSafeguardKind>(99);
  const auto unknown_safeguard_problem = pops::prepare_local_nonlinear_problem<1>(
      LinearResidual{}, pops::FiniteDifferenceLocalJacobian<1>{},
      pops::AcceptAllLocalCandidates<1>{}, unknown_safeguard_controls);
  EXPECT_EQ(pops::solve_prepared_local_nonlinear(unknown_safeguard_problem, safeguard_initial).status,
            pops::LocalNonlinearStatus::kUnsupportedCapability);

  int decoded_i = -1;
  int decoded_j = -1;
  int decoded_component = -1;
  pops::detail::decode_local_nonlinear_failure(
      pops::detail::encode_local_nonlinear_failure(17, 23, 4), decoded_i, decoded_j,
      decoded_component);
  EXPECT_EQ(decoded_i, 17);
  EXPECT_EQ(decoded_j, 23);
  EXPECT_EQ(decoded_component, 4);

  EXPECT_EQ(initial[0], Real(10));
  EXPECT_EQ(inadmissible_initial[0], Real(-1));
  EXPECT_EQ(safeguard_initial[0], Real(0));
}
