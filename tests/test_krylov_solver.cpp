// Solveur de Krylov MATRICE-LIBRE (BiCGStab) preconditionne par le V-cycle GeometricMG sur la
// partie SYMETRIQUE, pour l'operateur a TENSEUR PLEIN L(phi) = -div(A grad phi) + kappa phi
// (#120, krylov_solver.hpp). C'est la piece DECISIVE de la feuille de route Schur (PR3) : elle
// resout les cas ou le V-cycle MG SEUL echoue (#120 a constate STAGNATION pour c = 0.1-0.4 et
// DIVERGENCE pour c = 0.7 sur un A non symetrique).
//
// On valide :
//   (A) A = I (Axy=Ayx=0, kappa=0) : BiCGStab converge vers la MEME solution que GeometricMG
//       (Poisson canonique), a la tolerance MG (consistance du chemin nouveau == chemin de
//       reference). On compare phi_krylov a phi_mg cellule par cellule.
//   (B) MMS NON DIAGONALE, SOLVE (la ou MG seul echoue) : A SYMETRIQUE (Axy=Ayx=c) ET A NON
//       SYMETRIQUE (Axy=c, Ayx=-c), pour c in {0.1, 0.4, 0.7}. f = div(A grad phi_exact) analytique
//       (A constant) ; on resout et on exige residu RELATIF < 1e-10. On rapporte le nombre
//       d'iterations BiCGStab et, en CONTRASTE, l'etat du V-cycle MG SEUL (stagne / diverge).
//
// phi_exact(x,y) = sin(pi x) sin(pi y), nulle au bord du carre unite (Dirichlet exact). Pour A
// constant : div(A grad phi) = -pi^2 (Axx+Ayy) phi + (Axy+Ayx) pi^2 cos(pi x) cos(pi y).
//
// MPI : binaire rejoue a np=1/2/4 (CMake). Les produits scalaires (dot) sont COLLECTIFS, donc le
// critere d'arret BiCGStab se declenche a la MEME iteration sur tous les rangs : le nombre
// d'iterations et la convergence sont invariants au nombre de rangs (verifie par all_reduce).

#include <adc/numerics/elliptic/geometric_mg.hpp>
#include <adc/numerics/elliptic/krylov_solver.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

static double phi_exact(double x, double y) {
  return std::sin(kPi * x) * std::sin(kPi * y);
}

// Remplit rhs() = div(A grad phi_exact) (A constant : axx, ayy diagonaux ; cxy, cyx croises). Le
// systeme resolu est L_int(phi) = rhs avec L_int = div(A grad phi) (convention poisson_operator).
static void fill_mms_rhs(GeometricMG& mg, const Geometry& geom, const Box2D& dom,
                         double axx, double ayy, double cxy, double cyx) {
  const double csum = cxy + cyx;
  for (int li = 0; li < mg.rhs().local_size(); ++li) {
    Array4 af = mg.rhs().fab(li).array();
    for_each_cell(mg.rhs().box(li), [af, geom, axx, ayy, csum](int i, int j) {
      const double x = geom.x_cell(i), y = geom.y_cell(j);
      const double s = std::sin(kPi * x) * std::sin(kPi * y);
      const double cc = std::cos(kPi * x) * std::cos(kPi * y);
      af(i, j) = -kPi * kPi * (axx + ayy) * s + csum * kPi * kPi * cc;
    });
  }
}

// (B) un cas MMS : construit l'operateur PLEIN (op) + le preconditionneur SYMETRIQUE (precond, sans
// termes croises), resout par BiCGStab, renvoie iterations + convergence ; rapporte le V-cycle MG
// SEUL en contraste (meme operateur op, vcycle direct). n = resolution, c = amplitude croisee,
// non_sym = true -> Ayx = -c (A non symetrique), sinon Ayx = c (A symetrique).
struct SolveReport {
  int kry_iters; bool kry_conv; double kry_rel;
  double mg_r0, mg_rN; int mg_cycles; const char* mg_state;
};

static SolveReport solve_case(int n, double c, bool non_sym) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  BCRec bc; bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
  const double cyx = non_sym ? -c : c;

  // operateur PLEIN : A = [[1, c], [cyx, 1]].
  GeometricMG op(geom, ba, bc);
  op.set_epsilon_anisotropic([](Real, Real) { return Real(1); }, [](Real, Real) { return Real(1); });
  op.set_cross_terms([c](Real, Real) { return Real(c); }, [cyx](Real, Real) { return Real(cyx); });
  fill_mms_rhs(op, geom, dom, 1.0, 1.0, c, cyx);
  op.phi().set_val(0.0);

  // preconditionneur SYMETRIQUE : meme bloc diagonal, SANS set_cross_terms (-> partie symetrique).
  GeometricMG precond(geom, ba, bc);
  precond.set_epsilon_anisotropic([](Real, Real) { return Real(1); }, [](Real, Real) { return Real(1); });

  TensorKrylovSolver kry(op, precond, /*n_precond_vcycles=*/1);
  const KrylovResult kr = kry.solve(Real(1e-10), 300);

  // CONTRASTE : V-cycle MG SEUL sur le MEME operateur plein (lisseur 5 points, croises explicites).
  GeometricMG mg(geom, ba, bc);
  mg.set_epsilon_anisotropic([](Real, Real) { return Real(1); }, [](Real, Real) { return Real(1); });
  mg.set_cross_terms([c](Real, Real) { return Real(c); }, [cyx](Real, Real) { return Real(cyx); });
  fill_mms_rhs(mg, geom, dom, 1.0, 1.0, c, cyx);
  mg.phi().set_val(0.0);
  const double r0 = static_cast<double>(mg.current_residual());
  double rn = r0; int cyc = 0;
  for (int k = 0; k < 60 && rn > 1e-10 * r0; ++k) { mg.vcycle(); rn = static_cast<double>(mg.current_residual()); ++cyc; }
  const char* st = (rn < 1e-6 * r0) ? "CONVERGE" : (rn < r0 ? "stagne (incomplet)" : "DIVERGE/STAGNE");

  return SolveReport{kr.iters, kr.converged, static_cast<double>(kr.rel_residual), r0, rn, cyc, st};
}

// (A) A = I : ecart MAX phi_krylov vs phi_mg (Poisson canonique), reduit sur tous les rangs.
static double consistency_identity(int n) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  BCRec bc; bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

  // RHS de Poisson : f = div(grad phi_exact) = -2 pi^2 phi_exact (A = I, kappa = 0).
  auto fill = [&](GeometricMG& mg) {
    for (int li = 0; li < mg.rhs().local_size(); ++li) {
      Array4 af = mg.rhs().fab(li).array();
      for_each_cell(mg.rhs().box(li), [af, geom](int i, int j) {
        const double x = geom.x_cell(i), y = geom.y_cell(j);
        af(i, j) = -2.0 * kPi * kPi * phi_exact(x, y);
      });
    }
    mg.phi().set_val(0.0);
  };

  // reference : GeometricMG Poisson canonique (aucun coefficient).
  GeometricMG mg_ref(geom, ba, bc);
  fill(mg_ref);
  mg_ref.solve(Real(1e-12), 100);

  // Krylov : operateur PLEIN avec Axy=Ayx=0 (donc A = I) ; precond = Poisson canonique.
  GeometricMG op(geom, ba, bc);
  op.set_cross_terms([](Real, Real) { return Real(0); }, [](Real, Real) { return Real(0); });
  fill(op);
  GeometricMG precond(geom, ba, bc);
  TensorKrylovSolver kry(op, precond, 1);
  kry.solve(Real(1e-12), 300);

  // ecart MAX |phi_krylov - phi_mg| sur les cellules valides, all_reduce_max.
  double d = 0;
  for (int li = 0; li < op.phi().local_size(); ++li) {
    const ConstArray4 a = op.phi().fab(li).const_array();
    const ConstArray4 b = mg_ref.phi().fab(li).const_array();
    const Box2D bx = op.phi().box(li);
    for (int j = bx.lo[1]; j <= bx.hi[1]; ++j)
      for (int i = bx.lo[0]; i <= bx.hi[0]; ++i)
        d = std::fmax(d, std::fabs(a(i, j) - b(i, j)));
  }
  return all_reduce_max(d);
}

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank(), np = n_ranks();
  long fails = 0;
  auto chk = [&](bool cond, const char* w) {
    if (!cond) { if (me == 0) std::printf("FAIL %s\n", w); ++fails; }
  };

  // (A) consistance A = I : Krylov colle a GeometricMG Poisson.
  const double gA = consistency_identity(64);
  if (me == 0) std::printf("(A) A=I : max|phi_krylov - phi_mg| = %.3e\n", gA);
  chk(gA < 1e-8, "A_eq_I_consistance_MG");

  // (B) MMS non diagonale, SOLVE : BiCGStab converge la ou MG seul echoue. c = 0.1, 0.4, 0.7.
  const int n = 64;
  const double cs[3] = {0.1, 0.4, 0.7};
  for (int t = 0; t < 3; ++t) {
    const double c = cs[t];
    // A SYMETRIQUE (Axy = Ayx = c).
    const SolveReport rs = solve_case(n, c, /*non_sym=*/false);
    if (me == 0)
      std::printf("(B) SYM c=%.1f : BiCGStab %s en %d iters (rel=%.2e) | MG seul: r0=%.2e rN=%.2e (%d cyc) -> %s\n",
                  c, rs.kry_conv ? "CONVERGE" : "ECHOUE", rs.kry_iters, rs.kry_rel,
                  rs.mg_r0, rs.mg_rN, rs.mg_cycles, rs.mg_state);
    chk(rs.kry_conv, "B_sym_bicgstab_converge");
    chk(rs.kry_rel < 1e-10, "B_sym_residu_sous_1e-10");

    // A NON SYMETRIQUE (Axy = c, Ayx = -c) : le cas verrou de #120.
    const SolveReport ru = solve_case(n, c, /*non_sym=*/true);
    if (me == 0)
      std::printf("(B) NONSYM c=%.1f : BiCGStab %s en %d iters (rel=%.2e) | MG seul: r0=%.2e rN=%.2e (%d cyc) -> %s\n",
                  c, ru.kry_conv ? "CONVERGE" : "ECHOUE", ru.kry_iters, ru.kry_rel,
                  ru.mg_r0, ru.mg_rN, ru.mg_cycles, ru.mg_state);
    chk(ru.kry_conv, "B_nonsym_bicgstab_converge");
    chk(ru.kry_rel < 1e-10, "B_nonsym_residu_sous_1e-10");
  }

  // MPI : convergence et iterations invariantes au nombre de rangs (dot collectif). On reverifie le
  // cas verrou non symetrique fort et on all_reduce le nombre d'iterations : spread nul attendu.
  {
    const SolveReport r = solve_case(n, 0.7, /*non_sym=*/true);
    const long it = r.kry_iters;
    const long it_min = -static_cast<long>(all_reduce_max(static_cast<double>(-it)));
    const long it_max = static_cast<long>(all_reduce_max(static_cast<double>(it)));
    if (me == 0) std::printf("[mpi] np=%d : iters BiCGStab (nonsym c=0.7) min=%ld max=%ld (spread attendu 0)\n",
                             np, it_min, it_max);
    chk(it_min == it_max, "mpi_iters_invariant_rangs");
  }

  fails = static_cast<long>(all_reduce_max(static_cast<double>(fails)));  // un FAIL sur un rang -> tous
  if (me == 0 && fails == 0) std::printf("OK test_krylov_solver (np=%d)\n", np);
  comm_finalize();
  return fails == 0 ? 0 : 1;
}
