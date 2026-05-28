// Recouvrement calcul/comm (sect. 4.3), lance via mpirun -np N. On compare un
// pas d'advection en DEUX schemas qui doivent donner le meme resultat bit a bit :
//
//   bloquant : fill_boundary (halos complets) puis advance.
//   recouvert : fill_boundary_begin (poste les Isend/Irecv) -> advance de
//               l'INTERIEUR (independant des ghosts distants) pendant le transit
//               -> fill_boundary_end (attend + deballe) -> advance du BORD.
//
// L'interieur ne lit aucun ghost distant, donc son avance peut se faire pendant
// que les halos transitent ; le bord est avance apres reception. Resultat
// identique a l'arrondi (en fait bit a bit) -> le recouvrement est correct.

#include <adc/integrator/amr_reflux.hpp>  // compute_fluxes_1c, *face_box
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank(), np = n_ranks();
  long fails = 0;

  const int Nx = 64, Ny = 64;
  const double dx = 1.0 / Nx, dy = 1.0 / Ny, dt = 0.3 * dx;
  Box2D dom = Box2D::from_extents(Nx, Ny);
  if (Ny % np != 0) {
    if (me == 0) std::printf("Ny doit etre divisible par np\n");
    comm_finalize();
    return 1;
  }

  Diocotron m;
  m.B0 = 1.0;
  auto u0 = [&](int i, int j) {
    return 1.0 + 0.5 * std::sin(2 * kPi * (i + 0.5) / Nx) *
                     std::sin(2 * kPi * (j + 0.5) / Ny);
  };
  // bandes : 1 box/rang
  const int nyl = Ny / np, y0 = me * nyl;
  std::vector<Box2D> slabs;
  for (int r = 0; r < np; ++r)
    slabs.push_back(Box2D{{0, r * nyl}, {Nx - 1, (r + 1) * nyl - 1}});
  BoxArray ba(std::move(slabs));
  DistributionMapping dm(np, np);

  MultiFab Aux(ba, dm, 3, 1);
  {
    Fab2D& a = Aux.fab(0);
    const Box2D g = a.grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        a(i, j, 0) = 0.0;
        a(i, j, 1) = 0.3;   // gx -> vy
        a(i, j, 2) = -1.0;  // gy -> vx
      }
  }
  auto init = [&](MultiFab& U) {
    Fab2D& F = U.fab(0);
    for (int j = y0; j < y0 + nyl; ++j)
      for (int i = 0; i < Nx; ++i) F(i, j) = u0(i, j);
  };

  // --- reference bloquante ---
  MultiFab Uref(ba, dm, 1, 1);
  init(Uref);
  fill_boundary(Uref, dom, Periodicity{true, true});
  {
    Fab2D fx(xface_box(Uref.box(0)), 1, 0), fy(yface_box(Uref.box(0)), 1, 0);
    advance_fab_1c(m, Uref.fab(0), Aux.fab(0), dx, dy, dt, fx, fy);
  }

  // --- schema recouvert ---
  MultiFab Uold(ba, dm, 1, 1), Unew(ba, dm, 1, 1);
  init(Uold);
  const Box2D b = Uold.box(0);
  Fab2D fx(xface_box(b), 1, 0), fy(yface_box(b), 1, 0);

  auto apply = [&](const ConstArray4& FX, const ConstArray4& FY,
                   const ConstArray4& uo, Array4 un, int i, int j) {
    un(i, j) = uo(i, j) - dt * ((FX(i + 1, j) - FX(i, j)) / dx +
                                (FY(i, j + 1) - FY(i, j)) / dy);
  };

  HaloExchange h = fill_boundary_begin(Uold, dom, Periodicity{true, true});
  // avance de l'INTERIEUR pendant le transit des halos (flux interieurs exacts
  // a partir des cellules valides ; aucune dependance aux ghosts distants).
  compute_fluxes_1c(m, Uold.fab(0), Aux.fab(0), fx, fy);
  {
    const ConstArray4 uo = Uold.fab(0).const_array();
    Array4 un = Unew.fab(0).array();
    const ConstArray4 FX = fx.const_array(), FY = fy.const_array();
    for (int j = b.lo[1] + 1; j <= b.hi[1] - 1; ++j)
      for (int i = b.lo[0] + 1; i <= b.hi[0] - 1; ++i)
        apply(FX, FY, uo, un, i, j);
  }
  fill_boundary_end(Uold, h);  // halos distants recus
  // avance du BORD (ses flux dependent des ghosts maintenant remplis).
  compute_fluxes_1c(m, Uold.fab(0), Aux.fab(0), fx, fy);
  {
    const ConstArray4 uo = Uold.fab(0).const_array();
    Array4 un = Unew.fab(0).array();
    const ConstArray4 FX = fx.const_array(), FY = fy.const_array();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        if (i == b.lo[0] || i == b.hi[0] || j == b.lo[1] || j == b.hi[1])
          apply(FX, FY, uo, un, i, j);
  }

  // --- comparaison bit a bit ---
  double maxdiff = 0;
  {
    const ConstArray4 r = Uref.fab(0).const_array();
    const ConstArray4 n = Unew.fab(0).const_array();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        maxdiff = std::max(maxdiff, std::fabs(n(i, j) - r(i, j)));
  }
  const double gmax = all_reduce_max(maxdiff);
  if (gmax > 1e-13) ++fails;
  const long gfails = all_reduce_sum(fails);
  if (me == 0) {
    std::printf("np=%d  maxdiff(recouvert vs bloquant)=%.3e\n", np, gmax);
    if (gfails == 0) std::printf("OK test_mpi_overlap (np=%d)\n", np);
  }
  comm_finalize();
  return gfails == 0 ? 0 : 1;
}
