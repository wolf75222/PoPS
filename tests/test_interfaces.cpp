// Verifie que les interfaces extraites sont VIVES (pas seulement qu'elles
// compilent) :
//   1. flux numerique en politique : assemble_rhs<Minmod> (defaut RusanovFlux)
//      == assemble_rhs<Minmod, RusanovFlux> (explicite), bit a bit.
//   2. politique de couplage : Coupler::advance<Minmod, OncePerStepCoupling>
//      tourne et conserve la masse (un seul solve elliptique par pas).
//   3. concept EllipticSolver : GeometricMG le modele (static_assert).

#include <adc/coupling/coupler.hpp>
#include <adc/coupling/coupling_policy.hpp>
#include <adc/elliptic/elliptic_solver.hpp>
#include <adc/elliptic/geometric_mg.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/operator/numerical_flux.hpp>
#include <adc/operator/reconstruction.hpp>
#include <adc/operator/spatial_operator.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

static_assert(EllipticSolver<GeometricMG>, "GeometricMG modele EllipticSolver");

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const int N = 64, ng = 2;
  Box2D dom = Box2D::from_extents(N, N);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);
  Diocotron model;

  // --- 1. flux : defaut == RusanovFlux explicite ---
  {
    MultiFab U(ba, dm, 1, ng), aux(ba, dm, 3, 1);
    MultiFab R1(ba, dm, 1, 0), R2(ba, dm, 1, 0);
    auto wrap = [&](int x) { return (x % N + N) % N; };
    {
      Fab2D& f = U.fab(0);
      const Box2D g = f.grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
          const int ii = wrap(i), jj = wrap(j);
          f(i, j) = 1.0 + 0.3 * std::sin(2 * kPi * (ii + 0.5) / N) *
                              std::cos(2 * kPi * (jj + 0.5) / N);
        }
      Fab2D& a = aux.fab(0);
      const Box2D ga = a.grown_box();
      for (int j = ga.lo[1]; j <= ga.hi[1]; ++j)
        for (int i = ga.lo[0]; i <= ga.hi[0]; ++i) {
          a(i, j, 0) = 0.0;
          a(i, j, 1) = 0.2 * std::cos(2 * kPi * (wrap(i) + 0.5) / N);
          a(i, j, 2) = -0.5;
        }
    }
    assemble_rhs<Minmod>(model, U, aux, geom, R1);              // defaut Rusanov
    assemble_rhs<Minmod, RusanovFlux>(model, U, aux, geom, R2);  // explicite
    double maxdiff = 0;
    const ConstArray4 r1 = R1.fab(0).const_array();
    const ConstArray4 r2 = R2.fab(0).const_array();
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < N; ++i)
        maxdiff = std::fmax(maxdiff, std::fabs(r1(i, j) - r2(i, j)));
    std::printf("flux policy : maxdiff(defaut, RusanovFlux explicite)=%.3e\n",
                maxdiff);
    chk(maxdiff == 0.0, "flux_policy_defaut_egal_explicite");
  }

  // --- 2. couplage : OncePerStepCoupling tourne et conserve la masse ---
  {
    BCRec bcU, bcPhi;  // periodique
    Coupler<Diocotron> cpl(model, geom, ba, bcU, bcPhi);
    MultiFab U(ba, dm, 1, ng);
    auto wrap = [&](int x) { return (x % N + N) % N; };
    Fab2D& f = U.fab(0);
    const Box2D g = f.grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        f(i, j) = 1.0 + 0.3 * std::sin(2 * kPi * (wrap(i) + 0.5) / N) *
                            std::sin(2 * kPi * (wrap(j) + 0.5) / N);
    const double m0 = sum(U);
    const double dt = 0.02;
    for (int s = 0; s < 10; ++s)
      cpl.advance<Minmod, OncePerStepCoupling>(U, dt);
    const double m = sum(U);
    std::printf("OncePerStep : masse0=%.10e masse=%.10e dmasse=%.3e\n", m0, m,
                m - m0);
    chk(std::isfinite(m) && std::fabs(m - m0) < 1e-9, "oncePerStep_masse_conservee");
  }

  if (fails == 0) std::printf("OK test_interfaces\n");
  return fails == 0 ? 0 : 1;
}
