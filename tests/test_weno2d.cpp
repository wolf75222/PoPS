// WENO5-Z dans le FLUX 2D REEL (assemble_rhs), pas juste le noyau scalaire weno5z.
// On isole l'ordre SPATIAL : sur un champ lisse advecte a vitesse constante (div v = 0),
// assemble_rhs<Weno5, RusanovFlux> doit approcher -div F = -v . grad(rho) a l'ordre eleve
// (le terme diffusif de Rusanov est O(dx^5) car les etats reconstruits gauche/droite
// coincident a l'ordre 5 sur un champ lisse). On mesure l'ordre par raffinement et on
// verifie qu'il DEPASSE l'ordre 2 du MUSCL ; plus la preservation EXACTE d'un etat constant.
//
// rho0(x) = 1 + 0.5 cos(2 pi x), vitesse (vx, vy) = (1, 0) => -div F = -drho0/dx = pi sin(2 pi x).

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/operator/reconstruction.hpp>
#include <adc/operator/spatial_operator.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;
static double rho0(double x) { return 1.0 + 0.5 * std::cos(2 * kPi * x); }

// erreur L1 de assemble_rhs<Lim> vs l'exact pi sin(2 pi x), sur grille n x n periodique.
template <class Lim>
static double rhs_l1(int n) {
  const int ng = Lim::n_ghost;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());
  Diocotron model; model.B0 = 1.0;

  MultiFab U(ba, dm, 1, ng), aux(ba, dm, 3, ng), R(ba, dm, 1, 0);
  // aux uniforme : vx = -grad_y/B0 = 1, vy = grad_x/B0 = 0  =>  (phi, grad_x, grad_y) = (0, 0, -1).
  for (int li = 0; li < aux.local_size(); ++li) {
    Array4 a = aux.fab(li).array();
    const Box2D gb = aux.fab(li).grown_box();
    for (int j = gb.lo[1]; j <= gb.hi[1]; ++j)
      for (int i = gb.lo[0]; i <= gb.hi[0]; ++i) { a(i, j, 0) = 0; a(i, j, 1) = 0; a(i, j, 2) = -1; }
  }
  Array4 u = U.fab(0).array();
  for_each_cell(dom, [u, geom](int i, int j) { u(i, j, 0) = rho0(geom.x_cell(i)); });

  BCRec bc;  // tout periodique
  fill_ghosts(U, dom, bc);
  assemble_rhs<Lim, RusanovFlux>(model, U, aux, geom, R);

  const ConstArray4 r = R.fab(0).const_array();
  double s = 0; long cnt = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double exact = kPi * std::sin(2 * kPi * geom.x_cell(i));  // -drho0/dx
      s += std::fabs(r(i, j, 0) - exact); ++cnt;
    }
  return s / cnt;
}

template <class Lim>
static double order(int n1, int n2) {
  return std::log(rhs_l1<Lim>(n1) / rhs_l1<Lim>(n2)) / std::log(double(n2) / n1);
}

// preservation d'un etat CONSTANT : assemble_rhs<Weno5> doit rendre R = 0 a l'arrondi.
static double const_state_residual(int n) {
  const int ng = Weno5::n_ghost;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());
  Diocotron model; model.B0 = 1.0;
  MultiFab U(ba, dm, 1, ng), aux(ba, dm, 3, ng), R(ba, dm, 1, 0);
  U.set_val(2.5);
  for (int li = 0; li < aux.local_size(); ++li) {
    Array4 a = aux.fab(li).array(); const Box2D gb = aux.fab(li).grown_box();
    for (int j = gb.lo[1]; j <= gb.hi[1]; ++j)
      for (int i = gb.lo[0]; i <= gb.hi[0]; ++i) { a(i, j, 0) = 0; a(i, j, 1) = 0; a(i, j, 2) = -1; }
  }
  BCRec bc; fill_ghosts(U, dom, bc);
  assemble_rhs<Weno5, RusanovFlux>(model, U, aux, geom, R);
  return norm_inf(R);
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) { if (!c) { std::printf("FAIL %s\n", w); ++fails; } };

  const double o_ns = order<NoSlope>(32, 64);
  const double o_vl = order<VanLeer>(32, 64);
  const double o_w5 = order<Weno5>(64, 128);  // ordre eleve mesure plus haut en resolution
  std::printf("ordre spatial assemble_rhs : NoSlope=%.2f  VanLeer=%.2f  WENO5=%.2f\n", o_ns, o_vl, o_w5);
  const double cs = const_state_residual(64);
  std::printf("WENO5 etat constant : |R|_inf = %.3e\n", cs);

  chk(o_w5 > 3.0, "weno5_ordre_eleve_dans_le_flux_2d");   // nettement au-dessus du MUSCL (~2)
  chk(o_w5 > o_vl, "weno5_plus_precis_que_muscl");
  chk(cs < 1e-12, "weno5_preserve_etat_constant");

  if (fails == 0) std::printf("OK test_weno2d\n");
  return fails == 0 ? 0 : 1;
}
