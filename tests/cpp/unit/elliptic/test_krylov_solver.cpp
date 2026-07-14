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

#include <gtest/gtest.h>

#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/elliptic/linear/krylov_solver.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/parallel/comm.hpp>

#include <cmath>
#include <cstdio>

using namespace pops;
static constexpr double kPi = 3.14159265358979323846;

// POPS_HD : phi_exact est appelee depuis PoissonRhsKernel::operator() (POPS_HD), donc DEPUIS UN KERNEL
// DEVICE. Sans POPS_HD c'est un __host__ appele depuis un __host__ __device__ : sous Kokkos Cuda, nvcc
// emet le kernel SANS cet appel (warning #20011-D "calling a __host__ function ... is not allowed") et
// le rhs reste a 0 sur device. Le MG resout alors Lap(phi)=0 a Dirichlet V => phi=V plat, d'ou le cas
// (C) DIRICHLET err vs exact ~= max(sin sin) ~= 0.999 sur GH200 (mais 2e-4 sur l'oracle Serial, ou
// l'appel hote est licite). MmsRhsKernel (cas B) inline deja sin/cos SANS passer par phi_exact, d'ou
// son succes device. POPS_HD rend phi_exact device-callable ; corps inchange (sin sin) -> hote
// bit-identique, device desormais correct (rhs rempli, MG -> V + sin sin).
POPS_HD static double phi_exact(double x, double y) {
  return std::sin(kPi * x) * std::sin(kPi * y);
}

// FONCTEURS NOMMES (et non lambdas POPS_HD) pour les noyaux de remplissage : ce test premiere-instancie
// le V-cycle MG / la matvec depuis une UNITE DE TRADUCTION externe, ou nvcc n'emet pas fiablement une
// lambda etendue (meme recette #64/#97 que physical_bc.hpp / les foncteurs InitKernel du repo). Le corps
// est IDENTIQUE aux anciennes lambdas (memes types double, meme arithmetique) -> numerique bit-identique.

// rhs(i,j) = div(A grad phi_exact) pour A constant : -pi^2 (axx+ayy) sin sin + (cxy+cyx) pi^2 cos cos.
struct MmsRhsKernel {
  Array4 af;
  Geometry geom;
  double axx, ayy, csum;
  POPS_HD void operator()(int i, int j) const {
    const double x = geom.x_cell(i), y = geom.y_cell(j);
    const double s = std::sin(kPi * x) * std::sin(kPi * y);
    const double cc = std::cos(kPi * x) * std::cos(kPi * y);
    af(i, j) = -kPi * kPi * (axx + ayy) * s + csum * kPi * kPi * cc;
  }
};

// rhs(i,j) = div(grad phi_exact) = -2 pi^2 phi_exact (Poisson canonique A = I, kappa = 0).
struct PoissonRhsKernel {
  Array4 af;
  Geometry geom;
  POPS_HD void operator()(int i, int j) const {
    const double x = geom.x_cell(i), y = geom.y_cell(j);
    af(i, j) = -2.0 * kPi * kPi * phi_exact(x, y);
  }
};

// Remplit rhs() = div(A grad phi_exact) (A constant : axx, ayy diagonaux ; cxy, cyx croises). Le
// systeme resolu est L_int(phi) = rhs avec L_int = div(A grad phi) (convention poisson_operator).
static void fill_mms_rhs(GeometricMG& mg, const Geometry& geom, const Box2D& dom, double axx,
                         double ayy, double cxy, double cyx) {
  const double csum = cxy + cyx;
  for (int li = 0; li < mg.rhs().local_size(); ++li) {
    Array4 af = mg.rhs().fab(li).array();
    for_each_cell(mg.rhs().box(li), MmsRhsKernel{af, geom, axx, ayy, csum});
  }
}

// Materialize c_bc = L_int(0) independently of TensorKrylovSolver. This lets the regression build a
// deliberately cancelling stored RHS, rhs = c_bc + b_eff, and verify that the solve is normalized by
// the small effective forcing b_eff rather than by the large affine boundary offset.
static void fill_affine_operator_offset(GeometricMG& op, MultiFab& offset) {
  MultiFab zero(op.box_array(), op.dmap(), 1, 1);
  zero.set_val(Real(0));
  device_fence();
  fill_ghosts(zero, op.geom().domain, op.bc());
  apply_laplacian(zero, op.geom(), offset, op.op_coef(), op.op_eps(), op.op_kappa(),
                  op.op_eps_y(), op.op_a_xy(), op.op_a_yx());
}

struct AffineForcingReport {
  SolveReport probe;
  SolveReport converged;
  SolveReport warm;
  double true_relative_residual;
};

static AffineForcingReport affine_forcing_case(BCType type) {
  constexpr int n = 24;
  constexpr Real boundary_value = Real(1e3);
  constexpr Real tight_tol = Real(1e-9);
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba = BoxArray::from_domain(dom, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = type;
  bc.xlo_val = bc.xhi_val = bc.ylo_val = bc.yhi_val = boundary_value;
  bc.dx = geom.dx();
  bc.dy = geom.dy();
  if (type == BCType::Robin) {
    bc.xlo_alpha = bc.xhi_alpha = bc.ylo_alpha = bc.yhi_alpha = Real(1);
    bc.xlo_beta = bc.xhi_beta = bc.ylo_beta = bc.yhi_beta = Real(0.25);
  }

  GeometricMG op(geom, ba, bc);
  op.set_cross_terms([](Real, Real) { return Real(0); }, [](Real, Real) { return Real(0); });
  GeometricMG precond(geom, ba, bc);
  MultiFab offset(ba, op.dmap(), 1, 0);
  MultiFab effective(ba, op.dmap(), 1, 0);
  fill_affine_operator_offset(op, offset);
  for (int li = 0; li < effective.local_size(); ++li) {
    Array4 forcing = effective.fab(li).array();
    for_each_cell(effective.box(li), PoissonRhsKernel{forcing, geom});
  }
  lincomb(op.rhs(), Real(1), offset, Real(1), effective);

  TensorKrylovSolver kry(op, precond, 1);
  op.phi().set_val(Real(0));
  // With the correct scale, the initial relative residual is one and cannot pass a 0.5 tolerance.
  // Normalizing by the huge stored rhs instead would incorrectly publish a zero-iteration solve.
  const SolveReport probe = kry.solve(Real(0.5), 1);

  op.phi().set_val(Real(0));
  const SolveReport converged = kry.solve(tight_tol, 300);
  const double effective_norm = std::sqrt(static_cast<double>(dot(effective, effective)));
  const double true_relative_residual = static_cast<double>(kry.residual()) / effective_norm;
  const SolveReport warm = kry.solve(tight_tol, 300);
  return AffineForcingReport{probe, converged, warm, true_relative_residual};
}

// (B) un cas MMS : construit l'operateur PLEIN (op) + le preconditionneur SYMETRIQUE (precond, sans
// termes croises), resout par BiCGStab, renvoie iterations + convergence ; rapporte le V-cycle MG
// SEUL en contraste (meme operateur op, vcycle direct). n = resolution, c = amplitude croisee,
// non_sym = true -> Ayx = -c (A non symetrique), sinon Ayx = c (A symetrique).
struct TensorSolveCaseReport {
  int kry_iters;
  bool kry_conv;
  double kry_rel;
  double mg_r0, mg_rN;
  int mg_cycles;
  const char* mg_state;
};

static TensorSolveCaseReport solve_case(int n, double c, bool non_sym) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
  const double cyx = non_sym ? -c : c;

  // operateur PLEIN : A = [[1, c], [cyx, 1]].
  GeometricMG op(geom, ba, bc);
  op.set_epsilon_anisotropic([](Real, Real) { return Real(1); },
                             [](Real, Real) { return Real(1); });
  op.set_cross_terms([c](Real, Real) { return Real(c); }, [cyx](Real, Real) { return Real(cyx); });
  fill_mms_rhs(op, geom, dom, 1.0, 1.0, c, cyx);
  op.phi().set_val(0.0);

  // preconditionneur SYMETRIQUE : meme bloc diagonal, SANS set_cross_terms (-> partie symetrique).
  GeometricMG precond(geom, ba, bc);
  precond.set_epsilon_anisotropic([](Real, Real) { return Real(1); },
                                  [](Real, Real) { return Real(1); });

  TensorKrylovSolver kry(op, precond, /*n_precond_vcycles=*/1);
  const SolveReport kr = kry.solve(Real(1e-10), 300);

  // CONTRASTE : V-cycle MG SEUL sur le MEME operateur plein (lisseur 5 points, croises explicites).
  GeometricMG mg(geom, ba, bc);
  mg.set_epsilon_anisotropic([](Real, Real) { return Real(1); },
                             [](Real, Real) { return Real(1); });
  mg.set_cross_terms([c](Real, Real) { return Real(c); }, [cyx](Real, Real) { return Real(cyx); });
  fill_mms_rhs(mg, geom, dom, 1.0, 1.0, c, cyx);
  mg.phi().set_val(0.0);
  const double r0 = static_cast<double>(mg.current_residual());
  double rn = r0;
  int cyc = 0;
  for (int k = 0; k < 60 && rn > 1e-10 * r0; ++k) {
    mg.vcycle();
    rn = static_cast<double>(mg.current_residual());
    ++cyc;
  }
  const char* st =
      (rn < 1e-6 * r0) ? "CONVERGE" : (rn < r0 ? "stagne (incomplet)" : "DIVERGE/STAGNE");

  return TensorSolveCaseReport{kr.iters, kr.solved(), static_cast<double>(kr.rel_residual), r0, rn,
                               cyc, st};
}

// (A) A = I : ecart MAX phi_krylov vs phi_mg (Poisson canonique), reduit sur tous les rangs.
static double consistency_identity(int n) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

  // RHS de Poisson : f = div(grad phi_exact) = -2 pi^2 phi_exact (A = I, kappa = 0).
  auto fill = [&](GeometricMG& mg) {
    for (int li = 0; li < mg.rhs().local_size(); ++li) {
      Array4 af = mg.rhs().fab(li).array();
      for_each_cell(mg.rhs().box(li), PoissonRhsKernel{af, geom});
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

// (C) CL DIRICHLET NON NULLE (regression du preconditionneur). Solution manufacturee
//   phi_exact(x,y) = V + sin(pi x) sin(pi y),
// nulle au bord pour la partie sin -> phi_exact == V sur LES QUATRE faces (Dirichlet CONSTANTE V par
// face), donc bcPhi.*_val = V. La partie constante V est harmonique (Lap V = 0), exactement representee
// par le stencil 5 points ET par la reflexion Dirichlet (ghost = 2 V - V = V), donc le rhs est le MEME
// que le Poisson canonique : rhs = Lap(phi_exact) = -2 pi^2 sin(pi x) sin(pi y). La solution discrete
// est alors la solution homogene + V.
//
// AVANT LE FIX : apply_precond lance le V-cycle de precond_ avec la CL bc_ ENTIERE (V != 0). Depart
// phi=0, fill_physical_bc injecte ghost = 2 V - 0 : le V-cycle brut devient AFFINE (precond_raw(in) =
// M^{-1} in + d_bc) et phat/shat charrient l'offset constant d_bc. phi += alpha phat + omega shat accumule
// alors alpha d_bc + omega d_bc a chaque iteration : BiCGStab ne descend plus sous la tolerance (residu
// stagne) ET le potentiel derive loin de la reference. APRES LE FIX : apply_precond retranche d_bc (V-cycle
// HOMOGENE) ; le preconditionneur redevient lineaire et BiCGStab converge vers phi_exact.
//
// On rapporte : convergence + residu relatif Krylov, et l'ecart MAX a phi_exact (cellules valides) compare
// a une reference GeometricMG du MEME probleme Dirichlet (consistance, comme (A)) ET a l'analytique
// (avec la tolerance O(h^2) du schema 2 points). Renvoie le tout par TensorSolveCaseReport-like via parametres.
struct DirichletReport {
  bool kry_conv;
  double kry_rel;
  double err_vs_mg;     // max|phi_krylov - phi_mg_ref| (consistance solveur, tres serre)
  double err_vs_exact;  // max|phi_krylov - phi_exact| (truncature O(h^2) du schema)
  double ref_mag;       // |V| + 1 : echelle de reference pour la tolerance relative
};

static DirichletReport dirichlet_mms(int n, double V) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  // CL DIRICHLET NON NULLE : phi = V sur les quatre faces (phi_exact = V + sin sin, sin nulle au bord).
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
  bc.xlo_val = bc.xhi_val = bc.ylo_val = bc.yhi_val = static_cast<Real>(V);

  // RHS Poisson : f = Lap(phi_exact) = -2 pi^2 sin sin (la constante V est harmonique).
  auto fill = [&](GeometricMG& mg) {
    for (int li = 0; li < mg.rhs().local_size(); ++li) {
      Array4 af = mg.rhs().fab(li).array();
      for_each_cell(mg.rhs().box(li), PoissonRhsKernel{af, geom});
    }
    mg.phi().set_val(0.0);
  };

  // reference : GeometricMG du MEME probleme Dirichlet (operateur canonique, meme bcPhi non nulle).
  GeometricMG mg_ref(geom, ba, bc);
  fill(mg_ref);
  mg_ref.solve(Real(1e-12), 200);

  // Krylov : operateur PLEIN A = I (Axy=Ayx=0) ; preconditionneur Poisson canonique, MEME bcPhi non nulle.
  GeometricMG op(geom, ba, bc);
  op.set_cross_terms([](Real, Real) { return Real(0); }, [](Real, Real) { return Real(0); });
  fill(op);
  GeometricMG precond(geom, ba, bc);
  TensorKrylovSolver kry(op, precond, 1);
  const SolveReport kr = kry.solve(Real(1e-10), 300);

  // ecart MAX a la reference MG (consistance) et a l'analytique phi_exact = V + sin sin (cellules valides).
  double dmg = 0, dex = 0;
  for (int li = 0; li < op.phi().local_size(); ++li) {
    const ConstArray4 a = op.phi().fab(li).const_array();
    const ConstArray4 b = mg_ref.phi().fab(li).const_array();
    const Box2D bx = op.phi().box(li);
    for (int j = bx.lo[1]; j <= bx.hi[1]; ++j)
      for (int i = bx.lo[0]; i <= bx.hi[0]; ++i) {
        dmg = std::fmax(dmg, std::fabs(a(i, j) - b(i, j)));
        const double ex = V + phi_exact(geom.x_cell(i), geom.y_cell(j));
        dex = std::fmax(dex, std::fabs(a(i, j) - ex));
      }
  }
  return DirichletReport{kr.solved(), static_cast<double>(kr.rel_residual), all_reduce_max(dmg),
                         all_reduce_max(dex), std::fabs(V) + 1.0};
}

TEST(test_krylov_solver, affine_boundary_detection_is_typed_per_face) {
  BCRec bc;
  bc.xlo_val = bc.xhi_val = bc.ylo_val = bc.yhi_val = Real(7);
  EXPECT_FALSE(detail::tensor_krylov_has_affine_boundary_offset(bc));

  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Foextrap;
  EXPECT_FALSE(detail::tensor_krylov_has_affine_boundary_offset(bc));

  bc.xlo = BCType::Dirichlet;
  EXPECT_TRUE(detail::tensor_krylov_has_affine_boundary_offset(bc));
  bc.xlo = BCType::Foextrap;
  bc.yhi = BCType::Robin;
  EXPECT_TRUE(detail::tensor_krylov_has_affine_boundary_offset(bc));
  bc.yhi_val = Real(0);
  EXPECT_FALSE(detail::tensor_krylov_has_affine_boundary_offset(bc));
}

TEST(test_krylov_solver, affine_boundary_forcing_uses_effective_rhs_scale) {
  comm_init();
  for (const BCType type : {BCType::Dirichlet, BCType::Robin}) {
    SCOPED_TRACE(type == BCType::Dirichlet ? "Dirichlet" : "Robin");
    const AffineForcingReport report = affine_forcing_case(type);
    EXPECT_EQ(report.probe.iters, 1);
    EXPECT_TRUE(report.converged.solved());
    EXPECT_LT(report.converged.rel_residual, Real(1e-9));
    EXPECT_LT(report.true_relative_residual, 1e-9);
    EXPECT_TRUE(report.warm.solved());
    EXPECT_EQ(report.warm.iters, 0);
    EXPECT_LT(report.warm.rel_residual, Real(1e-9));
  }
}

TEST(test_krylov_solver, zero_forcing_requires_exact_zero_without_absolute_tolerance) {
  comm_init();
  constexpr int n = 8;
  const Box2D domain = Box2D::from_extents(n, n);
  const Geometry geometry{domain, 0.0, 1.0, 0.0, 1.0};
  const BoxArray boxes = BoxArray::from_domain(domain, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

  GeometricMG op(geometry, boxes, bc);
  op.set_cross_terms([](Real, Real) { return Real(0); },
                     [](Real, Real) { return Real(0); });
  GeometricMG precond(geometry, boxes, bc);
  TensorKrylovSolver solver(op, precond);
  solver.rhs().set_val(Real(0));  // exact affine forcing R(0) = 0
  solver.phi().set_val(Real(1e-6));
  const Real initial_residual = solver.residual();
  ASSERT_GT(initial_residual, Real(0));
  ASSERT_LT(initial_residual, Real(1));

  // The report denominator is one, but the stop remains max(rel_tol * 0, 0) = 0.
  const SolveReport exact_only =
      solver.solve(Real(1), /*max_iters=*/1, /*abs_tol=*/Real(0));
  EXPECT_FALSE(exact_only.solved() && exact_only.iters == 0);

  solver.phi().set_val(Real(1e-6));
  const Real reset_residual = solver.residual();
  const SolveReport absolute = solver.solve(
      Real(1e-8), /*max_iters=*/1, /*abs_tol=*/Real(2) * reset_residual);
  EXPECT_TRUE(absolute.solved());
  EXPECT_EQ(absolute.iters, 0);
  EXPECT_NEAR(absolute.rel_residual, reset_residual, Real(1e-14));
}

TEST(test_krylov_solver, tensor_krylov_solver_converges_where_mg_alone_stalls) {
  comm_init();
  const int me = my_rank(), np = n_ranks();
  long fails = 0;
  auto chk = [&](bool cond, const char* w) {
    if (!cond) {
      if (me == 0)
        std::printf("FAIL %s\n", w);
      ++fails;
      ADD_FAILURE() << w;
    }
  };

  // (A) consistance A = I : Krylov colle a GeometricMG Poisson.
  const double gA = consistency_identity(64);
  if (me == 0)
    std::printf("(A) A=I : max|phi_krylov - phi_mg| = %.3e\n", gA);
  chk(gA < 1e-8, "A_eq_I_consistance_MG");

  // (B) MMS non diagonale, SOLVE : BiCGStab converge la ou MG seul echoue. c = 0.1, 0.4, 0.7.
  const int n = 64;
  const double cs[3] = {0.1, 0.4, 0.7};
  for (int t = 0; t < 3; ++t) {
    const double c = cs[t];
    // A SYMETRIQUE (Axy = Ayx = c).
    const TensorSolveCaseReport rs = solve_case(n, c, /*non_sym=*/false);
    if (me == 0)
      std::printf(
          "(B) SYM c=%.1f : BiCGStab %s en %d iters (rel=%.2e) | MG seul: r0=%.2e rN=%.2e (%d cyc) "
          "-> %s\n",
          c, rs.kry_conv ? "CONVERGE" : "ECHOUE", rs.kry_iters, rs.kry_rel, rs.mg_r0, rs.mg_rN,
          rs.mg_cycles, rs.mg_state);
    chk(rs.kry_conv, "B_sym_bicgstab_converge");
    chk(rs.kry_rel < 1e-10, "B_sym_residu_sous_1e-10");

    // A NON SYMETRIQUE (Axy = c, Ayx = -c) : le cas verrou de #120.
    const TensorSolveCaseReport ru = solve_case(n, c, /*non_sym=*/true);
    if (me == 0)
      std::printf(
          "(B) NONSYM c=%.1f : BiCGStab %s en %d iters (rel=%.2e) | MG seul: r0=%.2e rN=%.2e (%d "
          "cyc) -> %s\n",
          c, ru.kry_conv ? "CONVERGE" : "ECHOUE", ru.kry_iters, ru.kry_rel, ru.mg_r0, ru.mg_rN,
          ru.mg_cycles, ru.mg_state);
    chk(ru.kry_conv, "B_nonsym_bicgstab_converge");
    chk(ru.kry_rel < 1e-10, "B_nonsym_residu_sous_1e-10");
  }

  // (C) CL DIRICHLET NON NULLE : regression du preconditionneur. AVANT le fix, le V-cycle de precond_
  // tourne avec la CL inhomogene (V != 0) et devient AFFINE : BiCGStab ne converge plus / derive de la
  // reference. APRES le fix (V-cycle HOMOGENE dans apply_precond), il converge vers phi_exact = V + sin sin.
  {
    const double V = 1.0;  // bcPhi = V sur les quatre faces (xlo_val = ... = 1.0)
    const DirichletReport rc = dirichlet_mms(n, V);
    // tolerance combinee err <= atol + rtol * max(1, |reference|). atol couvre l'arrondi du solve,
    // rtol*ref couvre la truncature O(h^2) du schema 2 points (n=64). Le bug PRE-FIX donne une erreur
    // O(1) (offset accumule), tres au-dessus de ce seuil ET un residu qui ne descend pas sous 1e-10.
    const double atol = 1e-8, rtol = 1e-2;
    const double tol_exact = atol + rtol * std::fmax(1.0, rc.ref_mag);
    const double tol_mg = atol + 1e-7 * std::fmax(1.0, rc.ref_mag);  // consistance solveur (serre)
    if (me == 0)
      std::printf(
          "(C) DIRICHLET V=%.1f : BiCGStab %s (rel=%.2e) | err vs MG=%.3e (tol %.1e) | err vs "
          "exact=%.3e (tol %.1e)\n",
          V, rc.kry_conv ? "CONVERGE" : "ECHOUE", rc.kry_rel, rc.err_vs_mg, tol_mg, rc.err_vs_exact,
          tol_exact);
    chk(rc.kry_conv, "C_dirichlet_bicgstab_converge");
    chk(rc.kry_rel < 1e-10, "C_dirichlet_residu_sous_1e-10");
    if (rc.err_vs_mg > tol_mg && me == 0)
      std::printf("  -> ecart Krylov vs MG = %.3e DEPASSE %.3e (gap %.3e)\n", rc.err_vs_mg, tol_mg,
                  rc.err_vs_mg - tol_mg);
    chk(rc.err_vs_mg <= tol_mg, "C_dirichlet_consistance_MG");
    if (rc.err_vs_exact > tol_exact && me == 0)
      std::printf("  -> ecart Krylov vs exact = %.3e DEPASSE %.3e (gap %.3e)\n", rc.err_vs_exact,
                  tol_exact, rc.err_vs_exact - tol_exact);
    chk(rc.err_vs_exact <= tol_exact, "C_dirichlet_vs_analytique");
  }

  // MPI : convergence et iterations invariantes au nombre de rangs (dot collectif). On reverifie le
  // cas verrou non symetrique fort et on all_reduce le nombre d'iterations : spread nul attendu.
  {
    const TensorSolveCaseReport r = solve_case(n, 0.7, /*non_sym=*/true);
    const long it = r.kry_iters;
    const long it_min = -static_cast<long>(all_reduce_max(static_cast<double>(-it)));
    const long it_max = static_cast<long>(all_reduce_max(static_cast<double>(it)));
    if (me == 0)
      std::printf(
          "[mpi] np=%d : iters BiCGStab (nonsym c=0.7) min=%ld max=%ld (spread attendu 0)\n", np,
          it_min, it_max);
    chk(it_min == it_max, "mpi_iters_invariant_rangs");
  }

  fails =
      static_cast<long>(all_reduce_max(static_cast<double>(fails)));  // un FAIL sur un rang -> tous
  if (me == 0 && fails == 0)
    std::printf("OK test_krylov_solver (np=%d)\n", np);
  comm_finalize();
}
