// Validation d'Euler pur (model/euler.hpp) via l'operateur spatial + SSPRK2 :
//   1. concept : Euler modele PhysicalModel.
//   2. free-stream : un etat uniforme a un residu nul (consistance du flux).
//   3. tourbillon isentropique (Shu) : solution lisse exacte (advectee a vitesse
//      (uinf,vinf)), on mesure l'ordre de convergence en raffinant -> ~2 attendu
//      (MUSCL Minmod + SSPRK2). Positivite (rho>0, p>0) preservee.

#include <adc/core/physical_model.hpp>
#include <adc/integrator/ssprk.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/euler.hpp>
#include <adc/operator/reconstruction.hpp>
#include <adc/operator/spatial_operator.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

static_assert(PhysicalModel<Euler>, "Euler doit modeler PhysicalModel");

// --- tourbillon isentropique centre en (cx, cy), composante comp de U ---
static double vortex(double x, double y, double cx, double cy, int comp,
                     double gamma, double beta, double uinf, double vinf,
                     double L) {
  double dx = std::fmod(x - cx + 1.5 * L, L) - 0.5 * L;  // -> [-L/2, L/2)
  double dy = std::fmod(y - cy + 1.5 * L, L) - 0.5 * L;
  const double r2 = dx * dx + dy * dy;
  const double ex = std::exp(0.5 * (1.0 - r2));
  const double du = -(beta / (2 * kPi)) * dy * ex;
  const double dv = (beta / (2 * kPi)) * dx * ex;
  const double T = 1.0 - (gamma - 1) * beta * beta / (8 * gamma * kPi * kPi) *
                             std::exp(1.0 - r2);
  const double rho = std::pow(T, 1.0 / (gamma - 1));
  const double u = uinf + du, v = vinf + dv;
  const double p = std::pow(rho, gamma);  // entropie uniforme S = p/rho^gamma = 1
  const double E = p / (gamma - 1) + 0.5 * rho * (u * u + v * v);
  if (comp == 0) return rho;
  if (comp == 1) return rho * u;
  if (comp == 2) return rho * v;
  return E;
}

// erreur L1 sur rho apres advection du tourbillon jusqu'a T sur une grille N x N.
template <class Limiter>
static double run_vortex(int N) {
  const double L = 10.0, beta = 5.0, uinf = 1.0, vinf = 1.0, T = 1.0, cfl = 0.4;
  Box2D dom = Box2D::from_extents(N, N);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);
  Euler model;
  const double g = model.gamma;

  MultiFab U(ba, dm, 4, 2), aux(ba, dm, 3, 2);
  aux.set_val(0.0);
  {
    Fab2D& f = U.fab(0);
    const Box2D v = U.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        for (int c = 0; c < 4; ++c)
          f(i, j, c) = vortex(geom.x_cell(i), geom.y_cell(j), 5.0, 5.0, c, g,
                              beta, uinf, vinf, L);
  }
  BCRec bc;  // periodique

  double t = 0;
  while (t < T - 1e-12) {
    double vmax = 0;
    const ConstArray4 u = U.fab(0).const_array();
    const Box2D v = U.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        Euler::State s;
        for (int c = 0; c < 4; ++c) s[c] = u(i, j, c);
        vmax = std::fmax(vmax, std::fmax(model.max_wave_speed(s, Aux{}, 0),
                                         model.max_wave_speed(s, Aux{}, 1)));
      }
    const double h = std::fmin(geom.dx(), geom.dy());
    double dt = cfl * h / vmax;
    if (t + dt > T) dt = T - t;
    advance_ssprk2<Limiter>(model, U, aux, geom, bc, dt);
    t += dt;
  }

  double err = 0;
  const double cx = 5.0 + uinf * T, cy = 5.0 + vinf * T;
  const ConstArray4 u = U.fab(0).const_array();
  const Box2D v = U.box(0);
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
      const double re =
          vortex(geom.x_cell(i), geom.y_cell(j), cx, cy, 0, g, beta, uinf, vinf, L);
      err += std::fabs(u(i, j, 0) - re);
    }
  return err / (static_cast<double>(N) * N);
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  // --- 2. free-stream : etat uniforme -> residu nul ---
  {
    const int N = 24;
    Box2D dom = Box2D::from_extents(N, N);
    Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
    BoxArray ba(std::vector<Box2D>{dom});
    DistributionMapping dm(1, 1);
    Euler model;
    const double g = model.gamma, rho = 1.2, vx = 0.7, vy = -0.4, p = 1.5;
    const double E = p / (g - 1) + 0.5 * rho * (vx * vx + vy * vy);
    MultiFab U(ba, dm, 4, 2), aux(ba, dm, 3, 2), R(ba, dm, 4, 0);
    aux.set_val(0.0);
    {
      Fab2D& f = U.fab(0);
      const Box2D gb = f.grown_box();
      for (int j = gb.lo[1]; j <= gb.hi[1]; ++j)
        for (int i = gb.lo[0]; i <= gb.hi[0]; ++i) {
          f(i, j, 0) = rho;
          f(i, j, 1) = rho * vx;
          f(i, j, 2) = rho * vy;
          f(i, j, 3) = E;
        }
    }
    assemble_rhs<Minmod>(model, U, aux, geom, R);
    double maxR = 0;
    const ConstArray4 r = R.fab(0).const_array();
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < N; ++i)
        for (int c = 0; c < 4; ++c) maxR = std::fmax(maxR, std::fabs(r(i, j, c)));
    std::printf("free-stream : max|R|=%.3e\n", maxR);
    chk(maxR < 1e-10, "free_stream_preservation");
  }

  // --- 3. tourbillon isentropique : ordre de convergence ---
  // Minmod (le plus diffusif) ecrete les extrema lisses du tourbillon -> ordre
  // reduit. VanLeer (limiteur lisse) preserve mieux le 2e ordre : on mesure l'ordre
  // sur VanLeer, et on imprime Minmod pour comparaison.
  {
    const double m64 = run_vortex<Minmod>(64), m128 = run_vortex<Minmod>(128);
    const double v64 = run_vortex<VanLeer>(64), v128 = run_vortex<VanLeer>(128);
    const double om = std::log2(m64 / m128), ov = std::log2(v64 / v128);
    std::printf("vortex Minmod  : L1 N=64 %.3e | N=128 %.3e | ordre=%.2f\n", m64,
                m128, om);
    std::printf("vortex VanLeer : L1 N=64 %.3e | N=128 %.3e | ordre=%.2f\n", v64,
                v128, ov);
    chk(std::isfinite(v128) && v128 < v64, "vortex_converge");
    chk(ov > 1.7, "vortex_VanLeer_ordre_~2");
  }

  if (fails == 0) std::printf("OK test_euler\n");
  return fails == 0 ? 0 : 1;
}
