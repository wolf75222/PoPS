// EXEMPLE C++ MINIMAL deux especes, SANS Python (jalon 2.4).
//
// Le test "est-ce qu'un utilisateur peut construire son cas ?" : electrons
// IMPLICITES (source de relaxation raide) + ions EXPLICITES (SSPRK2) + Poisson
// rhs = n_i - n_e, assemble par ChargeDensityRhs a N especes. L'utilisateur ne
// compose que des briques (modele local, schema spatial, politique temps, charge)
// et appelle SystemCoupler ; aucun solveur implicite n'est ecrit a la main (le
// defaut ImplicitSourceStepper s'en charge).
//
// Couvre aussi : RHS Poisson non nul a N blocs (jalon 2.1.1 / 2.5.1) et le defaut
// implicite inconditionnellement stable sur source raide (jalon 2.2.1 / 2.5.3).

#include <gtest/gtest.h>

#include <pops/core/model/coupled_system.hpp>
#include <pops/core/state/state.hpp>
#include <pops/coupling/system/system_coupler.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cmath>
#include <type_traits>

using namespace pops;

// Electrons : densite scalaire qui RELAXE vers neq a un taux RAIDE k. Pas de flux
// (drift gele dans ce squelette). C'est exactement le terme qu'on veut implicite :
// en explicite il imposerait dt < 1/k.
struct ElectronRelax {
  using State = StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;

  Real k = Real(1000);  // raideur
  Real neq = Real(1);   // densite d'equilibre

  POPS_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  POPS_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  POPS_HD State source(const State& u, const Aux&) const { return State{-k * (u[0] - neq)}; }
  POPS_HD Real elliptic_rhs(const State& u) const { return -u[0]; }
};

// Ions : production constante, explicite. Pas de flux.
struct IonProduction {
  using State = StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;

  Real rate = Real(3);

  POPS_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  POPS_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  POPS_HD State source(const State&, const Aux&) const { return State{rate}; }
  POPS_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

using ElectronBlock = EquationBlock<ElectronRelax, FirstOrder, ImplicitTime<UserTimeIntegrator, 1>>;
using IonBlock = EquationBlock<IonProduction, FirstOrder, ExplicitTime<SSPRK2, 1>>;

static_assert(EquationBlockLike<ElectronBlock>);
static_assert(EquationBlockLike<IonBlock>);
static_assert(ElectronBlock::Time::treatment == TimeTreatment::Implicit);
static_assert(IonBlock::Time::treatment == TimeTreatment::Explicit);

// Fixture partageant la composition du cas (electrons implicites + ions explicites + Poisson a
// 2 especes) et l'unique macro-pas : les 3 groupes de verification (ions, electrons, RHS Poisson)
// portent sur l'etat resultant de ce meme pas, mais sont independants entre eux.
class TwoSpeciesMinimal : public ::testing::Test {
 protected:
  void SetUp() override {
    ba_ = BoxArray::from_domain(dom_, 4);
    dm_ = DistributionMapping(ba_.size(), n_ranks());
    Ue_ = MultiFab(ba_, dm_, 1, 2);
    Ui_ = MultiFab(ba_, dm_, 1, 2);
    Ue_.set_val(Real(5));  // loin de neq=1 : la relaxation doit etre forte
    Ui_.set_val(Real(0));
  }

  static constexpr int kNcell = 16;
  static constexpr Real kDt = Real(0.1);

  Box2D dom_ = Box2D::from_extents(4, 4);
  Geometry geom_{dom_, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba_;
  DistributionMapping dm_{1, 1};
  BCRec bc_;  // periodique partout (le defaut)
  MultiFab Ue_, Ui_;
};

TEST_F(TwoSpeciesMinimal, IonExplicitBlockAdvancesExactly) {
  ElectronBlock electrons{"electrons", ElectronRelax{}, Ue_, bc_};
  IonBlock ions{"ions", IonProduction{}, Ui_, bc_};
  CoupledSystem system{electrons, ions};
  ChargeDensityRhs charge{{{Real(-1), 0}, {Real(1), 0}}};  // [electrons, ions]
  auto sim = make_system_coupler(system, geom_, ba_, bc_, charge);

  sim.step(kDt, ImplicitSourceStepper{});

  // Ions explicites : production constante exacte, n_i = dt * rate = 0.3.
  EXPECT_TRUE(std::fabs(sum(Ui_) - Real(0.3) * kNcell) < Real(1e-12)) << "ion_explicit";
}

TEST_F(TwoSpeciesMinimal, ElectronImplicitBlockIsBackwardEulerExactAndBounded) {
  ElectronBlock electrons{"electrons", ElectronRelax{}, Ue_, bc_};
  IonBlock ions{"ions", IonProduction{}, Ui_, bc_};
  CoupledSystem system{electrons, ions};
  ChargeDensityRhs charge{{{Real(-1), 0}, {Real(1), 0}}};  // [electrons, ions]
  auto sim = make_system_coupler(system, geom_, ba_, bc_, charge);

  sim.step(kDt, ImplicitSourceStepper{});

  // Electrons implicites : backward-Euler exact pour la relaxation lineaire,
  // n_e = (n0 + dt k neq) / (1 + dt k). dt*k = 100 : un schema explicite
  // EXPLOSERAIT (n_e ~ 5 - 400). Ici la valeur reste bornee et proche de neq.
  const Real ne_be = (Real(5) + kDt * Real(1000) * Real(1)) / (Real(1) + kDt * Real(1000));
  EXPECT_TRUE(std::fabs(sum(Ue_) - ne_be * kNcell) < Real(1e-9)) << "electron_implicit_exact";
  EXPECT_TRUE(sum(Ue_) > Real(0) && sum(Ue_) < Real(5) * kNcell) << "electron_implicit_bounded";
}

TEST_F(TwoSpeciesMinimal, PoissonRhsSumsAcrossSpeciesAndIsNonZero) {
  ElectronBlock electrons{"electrons", ElectronRelax{}, Ue_, bc_};
  IonBlock ions{"ions", IonProduction{}, Ui_, bc_};
  CoupledSystem system{electrons, ions};
  // Poisson rhs = Sum_s q_s n_s = (+1) n_i + (-1) n_e = n_i - n_e.
  ChargeDensityRhs charge{{{Real(-1), 0}, {Real(1), 0}}};  // [electrons, ions]
  auto sim = make_system_coupler(system, geom_, ba_, bc_, charge);

  sim.step(kDt, ImplicitSourceStepper{});

  // RHS Poisson a N especes, non nul (jalon 2.1.1 / 2.5.1) : f = n_i - n_e, et l'assembleur
  // somme bien sur tous les blocs.
  MultiFab rhs(ba_, dm_, 1, 0);
  charge(system, rhs);
  EXPECT_TRUE(std::fabs(sum(rhs) - (sum(Ui_) - sum(Ue_))) < Real(1e-12)) << "charge_density_rhs";
  EXPECT_TRUE(std::fabs(sum(rhs)) > Real(1)) << "poisson_rhs_nonzero";
}
