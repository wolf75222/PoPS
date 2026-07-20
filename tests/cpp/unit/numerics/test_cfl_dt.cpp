// Pas macro choisi par CFL multi-especes (sec.8.2 C) : SystemDriver::step_cfl. Le pas est
// dt = cfl * min(dx,dy) / w_max, ou w_max est la plus grande vitesse d'onde sur TOUTES les
// especes -> l'espece la plus rapide contraint le pas. Combine au Stride d'une espece lente,
// cela donne le multirate pratique.

#include <gtest/gtest.h>

#include <pops/core/model/coupled_system.hpp>
#include <pops/core/state/state.hpp>
#include <pops/coupling/system/system_coupler.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/spatial/primitives/wave_speed.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace pops;

namespace {

// Advection a vitesse constante a : vitesse d'onde max = |a|.
struct AdvectX {
  using State = StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;
  Real a = Real(1);
  POPS_HD State flux(const State& u, const Aux&, int dir) const {
    return State{dir == 0 ? a * u[0] : Real(0)};
  }
  POPS_HD Real max_wave_speed(const State&, const Aux&, int) const { return a < 0 ? -a : a; }
  POPS_HD State source(const State&, const Aux&) const { return State{}; }
  POPS_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

// Modele dont la vitesse d'onde est NaN (flux physique fini, 0). Le calcul CFL doit refuser cette
// violation du contrat au point de reduction collectif, avant tout pas de temps.
struct NanSpeed {
  using State = StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;
  POPS_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  POPS_HD Real max_wave_speed(const State&, const Aux&, int) const {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  POPS_HD State source(const State&, const Aux&) const { return State{}; }
  POPS_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

struct BoundProbe {
  using State = StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;
  Real wave_x = Real(0);
  Real wave_y = Real(0);
  Real stability_x = Real(0);
  Real stability_y = Real(0);
  Real frequency = Real(0);
  Real direct_dt = std::numeric_limits<Real>::infinity();

  POPS_HD Real max_wave_speed(const State&, const Aux&, int direction) const {
    return direction == 0 ? wave_x : wave_y;
  }
  POPS_HD Real stability_speed(const State&, const Aux&, int direction) const {
    return direction == 0 ? stability_x : stability_y;
  }
  POPS_HD Real source_frequency(const State&, const Aux&) const { return frequency; }
  POPS_HD Real stability_dt(const State&, const Aux&) const { return direct_dt; }
};

struct ZeroSystemRhs {
  template <class System>
  void operator()(const System&, MultiFab& rhs) const {
    rhs.set_val(Real(0));
  }
};

using Blk = EquationBlock<AdvectX, FirstOrder, ExplicitTime<SSPRK2, 1>>;

}  // namespace

TEST(test_cfl_dt, fastest_species_sets_the_step) {
  const int n = 16;
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  BCRec bc;

  MultiFab Uf(ba, dm, 1, 2), Us(ba, dm, 1, 2);
  Uf.set_val(Real(1));
  Us.set_val(Real(1));
  // espece rapide (a=2) + espece lente (a=0.5) : le pas CFL est fixe par la rapide.
  Blk fast{"fast", AdvectX{Real(2)}, Uf, bc};
  Blk slow{"slow", AdvectX{Real(0.5)}, Us, bc};
  CoupledSystem system{fast, slow};
  auto sim = make_system_coupler(system, geom, ba, bc, ZeroSystemRhs{});

  const Real cfl = Real(0.4);
  const Real h = std::min(geom.dx(), geom.dy());
  const Real expected = cfl * h / Real(2);  // w_max = max(2, 0.5) = 2

  const Real dt = sim.step_cfl(cfl);
  EXPECT_LT(std::fabs(dt - expected), Real(1e-14)) << "cfl_dt_from_fastest_species";
  EXPECT_GT(dt, Real(0)) << "cfl_dt_positive";
  // le systeme a avance (etat fini, masse conservee pour l'advection periodique).
  EXPECT_LT(std::fabs(sum(Uf, 0) - Real(1) * n * n), Real(1e-10)) << "fast_mass_conserved";
  EXPECT_LT(std::fabs(sum(Us, 0) - Real(1) * n * n), Real(1e-10)) << "slow_mass_conserved";
}

// ADC-267 : systeme au repos (w_max = 0) -> le garde CFL clampe le denominateur a 1e-30.
// dt = cfl*h / max(w_max, 1e-30) ; sans le plancher, dt = +inf sur un etat au repos.
TEST(test_cfl_dt, quiescent_system_dt_clamped_to_floor) {
  const int n = 16;
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  BCRec bc;
  const Real cfl = Real(0.4);
  const Real h = std::min(geom.dx(), geom.dy());

  MultiFab Uq(ba, dm, 1, 2);
  Uq.set_val(Real(1));
  Blk quiet{"quiet", AdvectX{Real(0)}, Uq, bc};  // a = 0 -> w_max = 0
  CoupledSystem qsys{quiet};
  SystemCoupler qsim(qsys, geom, ba, bc, ZeroSystemRhs{});
  const Real dtq = qsim.step_cfl(cfl);
  const Real floor_dt = cfl * h / Real(1e-30);  // denominateur clampe au plancher
  EXPECT_TRUE(std::isfinite(dtq)) << "quiescent_dt_finite";
  EXPECT_LE(std::fabs(dtq - floor_dt), floor_dt * Real(1e-12)) << "quiescent_dt_clamped_to_floor";
  EXPECT_TRUE(std::isfinite(sum(Uq, 0))) << "quiescent_state_finite";  // a = 0 -> aucune advection
}

TEST(test_cfl_dt, nan_wave_speed_is_rejected_before_the_step) {
  const int n = 16;
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  BCRec bc;
  const Real cfl = Real(0.4);

  MultiFab Un(ba, dm, 1, 2);
  Un.set_val(Real(1));
  using NanBlk = EquationBlock<NanSpeed, FirstOrder, ExplicitTime<SSPRK2, 1>>;
  NanBlk nblk{"nan", NanSpeed{}, Un, bc};
  CoupledSystem nsys{nblk};
  SystemCoupler nsim(nsys, geom, ba, bc, ZeroSystemRhs{});
  EXPECT_THROW((void)nsim.cfl_dt(cfl), std::domain_error);
}

TEST(test_cfl_dt, native_bound_reductions_reject_every_invalid_scalar_category) {
  const Box2D dom = Box2D::from_extents(2, 2);
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  MultiFab U(ba, dm, 1, 0), aux(ba, dm, kAuxBaseComps, 0);
  U.set_val(Real(1));
  aux.set_val(Real(0));

  const Real nan = std::numeric_limits<Real>::quiet_NaN();
  const Real inf = std::numeric_limits<Real>::infinity();
  for (const Real invalid : {Real(-1), nan, inf}) {
    EXPECT_THROW((void)max_wave_speed_mf(
                     BoundProbe{invalid, Real(1), Real(0), Real(0), Real(0)}, U, aux),
                 std::domain_error);
    EXPECT_THROW((void)max_stability_speed_mf(
                     BoundProbe{Real(0), Real(0), invalid, Real(1), Real(0)}, U, aux),
                 std::domain_error);
    EXPECT_THROW((void)max_source_frequency_mf(
                     BoundProbe{Real(0), Real(0), Real(0), Real(0), invalid}, U, aux),
                 std::domain_error);
  }

  for (const Real invalid_dt : {Real(-1), Real(0), nan,
                                -std::numeric_limits<Real>::infinity()}) {
    EXPECT_THROW((void)min_stability_dt_mf(
                     BoundProbe{Real(0), Real(0), Real(0), Real(0), Real(0), invalid_dt}, U, aux),
                 std::domain_error);
  }

  const BoundProbe valid{Real(2), Real(1), Real(3), Real(2), Real(4)};
  EXPECT_EQ(max_wave_speed_mf(valid, U, aux), Real(2));
  EXPECT_EQ(max_stability_speed_mf(valid, U, aux), Real(3));
  EXPECT_EQ(max_source_frequency_mf(valid, U, aux), Real(4));
  EXPECT_EQ(min_stability_dt_mf(valid, U, aux), Real(0));  // documented +inf means no direct bound
  const Real tiny = std::numeric_limits<Real>::denorm_min();
  EXPECT_EQ(min_stability_dt_mf(
                BoundProbe{Real(0), Real(0), Real(0), Real(0), Real(0), tiny}, U, aux),
            tiny);
}
