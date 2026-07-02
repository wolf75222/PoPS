// ETAGE SOURCE condense par Schur en geometrie POLAIRE (PolarCondensedSchurSourceStepper, Voie A etape
// 2b) : test SOURCE-ONLY (transport gele) d'un fluide magnetise RAIDE sur un anneau (r, theta), sous la
// source implicite couplee potentiel / vitesse / Lorentz (Hoffart et al., arXiv:2510.11808). Pendant
// POLAIRE de test_condensed_schur_source_stepper (cartesien). Cf.
// include/pops/coupling/polar_condensed_schur_source_stepper.hpp.
//
// SYSTEME SOURCE (transport gele), rho = rho0 (gele), B_z = B0, alpha. Vitesse (v_r, v_theta) dans la
// base locale orthonormee (e_r, e_theta) ; la force de Lorentz y a la MEME forme qu'en cartesien
// ((v x B)_r = +B0 v_theta, (v x B)_theta = -B0 v_r) :
//   d_t v_r     = -(grad_polar phi)_r     + B0 v_theta
//   d_t v_theta = -(grad_polar phi)_theta - B0 v_r
//   -Lap_polar phi = alpha rho0 div_polar v
// grad_polar phi = (d_r phi, (1/r) d_theta phi), div_polar v = (1/r)[d_r(r v_r) + d_theta v_theta]
// (differences CENTREES, memes stencils que le stepper). C'est un systeme d'EDO lineaire RAIDE : la
// frequence cyclotron B0 (eleve, 100) fixe un pas explicite maximal. Au-dela, l'Euler explicite EXPLOSE ;
// l'etage de Schur (theta = 1 = Euler retrograde) reste STABLE sans condition.
//
// On valide TROIS choses (memes que le cartesien) :
//   (A) RELATION IMPLICITE reconstruite : apres step(), v^{n+1} satisfait B v^{n+1} = v^n - dt
//       grad_polar phi^{n+1} (B = I - dt [Omega], grad_polar CENTRE), a la tolerance du SOLVE. Verif
//       EXACTE que la reconstruction v = B^{-1}(v^n - dt grad_polar phi) est coherente avec le phi resolu.
//   (B) STABILITE vs EXPLOSION explicite, a GRAND dt (50 x le pas explicite stable, K pas) : l'Euler
//       explicite EXPLOSE (||v|| croit de plusieurs ordres) ; le Schur garde ||v|| BORNEE. On compare.
//   (C) ACCORD AVEC UNE REFERENCE fine-dt (RK4) a dt MODERE : le Schur (theta=1, ordre 1) en est proche
//       et l'ecart DECROIT a l'ordre 1 (dt, dt/2 -> ratio > 1.5). Garde-fou : l'etage resout le BON
//       systeme (pas seulement "stable").
//
// Host / Serial-safe : UNE box, n_ranks()==1 (PolarTensorKrylovSolver / PolarPoissonSolver mono-rang).

#include <gtest/gtest.h>

#include <pops/coupling/schur/source/polar_condensed_schur_source_stepper.hpp>

#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/elliptic/polar/polar_poisson_solver.hpp>
#include <pops/numerics/linalg/lorentz_eliminator.hpp>
#include <pops/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace pops;
static constexpr double kPi = 3.14159265358979323846;
static constexpr double kRmin = 0.30;
static constexpr double kRmax = 1.00;

// VariableSet d'un fluide polaire minimal : rho, mom_r (MomentumX), mom_theta (MomentumY) + energie opt.
static VariableSet fluid_vars(bool with_E) {
  VariableSet vs;
  vs.kind = VariableKind::Conservative;
  if (with_E) {
    vs.names = {"rho", "mr", "mth", "E"};
    vs.roles = {VariableRole::Density, VariableRole::MomentumX, VariableRole::MomentumY,
                VariableRole::Energy};
    vs.size = 4;
  } else {
    vs.names = {"rho", "mr", "mth"};
    vs.roles = {VariableRole::Density, VariableRole::MomentumX, VariableRole::MomentumY};
    vs.size = 3;
  }
  return vs;
}

struct Setup {
  Box2D dom;
  PolarGeometry geom;
  BoxArray ba;
  DistributionMapping dm;
  BCRec bc;
  Setup(int nr, int nth)
      : dom(Box2D::from_extents(nr, nth)),
        geom{dom, kRmin, kRmax},
        ba(std::vector<Box2D>{dom}),
        dm(ba.size(), n_ranks()) {
    // phi : Dirichlet radial (= 0 a la paroi -> operateur inversible, pas de jauge) ; theta periodique.
    bc.xlo = bc.xhi = BCType::Dirichlet;
    bc.ylo = bc.yhi = BCType::Periodic;
    bc.xlo_val = bc.xhi_val = 0.0;
  }
};

// Profils analytiques lisses, periodiques en theta, nuls aux bords radiaux (compatibles Dirichlet phi=0).
// v_r0, v_theta0 : champs de vitesse initiaux ; phi0 : potentiel initial. h(r)=sin(...) s'annule en
// r_min/r_max (compatible Dirichlet homogene phi=0 aux deux bords radiaux).
struct InitKernel {
  PolarGeometry geom;
  Array4 st, phi;
  Real rho0;
  int c_rho, c_mx, c_my, c_E;
  POPS_HD void operator()(int i, int j) const {
    const Real r = geom.r_cell(i);
    const Real th = geom.theta_cell(j);
    const Real h = std::sin(Real(kPi) * (r - Real(kRmin)) / Real(kRmax - kRmin));  // 0 aux bords
    const Real vr = Real(0.6) * h * std::cos(Real(2) * th);                        // v_r0
    const Real vth = Real(-0.4) * h * std::sin(th);                                // v_theta0
    const Real ph = Real(0.3) * h * std::cos(th);  // phi0 (0 aux bords radiaux)
    st(i, j, c_rho) = rho0;
    st(i, j, c_mx) = rho0 * vr;
    st(i, j, c_my) = rho0 * vth;
    if (c_E >= 0)
      st(i, j, c_E) = Real(1.0) + Real(0.5) * rho0 * (vr * vr + vth * vth);
    phi(i, j, 0) = ph;
  }
};

struct ConstBzKernel {
  Array4 bz;
  Real B0;
  POPS_HD void operator()(int i, int j) const { bz(i, j, 0) = B0; }
};

// norme L2 GLOBALE de la VITESSE (mr, mtheta)/rho de l'etat (diagnostic de stabilite).
static double vel_l2(const MultiFab& st, int c_rho, int c_mx, int c_my) {
  sync_host();
  double s = 0;
  for (int li = 0; li < st.local_size(); ++li) {
    const ConstArray4 u = st.fab(li).const_array();
    const Box2D b = st.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        const double rho = u(i, j, c_rho);
        const double vr = u(i, j, c_mx) / rho, vt = u(i, j, c_my) / rho;
        s += vr * vr + vt * vt;
      }
  }
  return std::sqrt(all_reduce_sum(s));
}

// ----------------------------------------------------------------------------------------------
// (A) RELATION IMPLICITE : ecart MAX |B v^{n+1} - (v^n - dt grad_polar phi^{n+1})|, grad_polar CENTRE.
// ----------------------------------------------------------------------------------------------
static double implicit_residual(const MultiFab& st_new, const MultiFab& vrn, const MultiFab& vtn,
                                MultiFab& phi_new, const PolarGeometry& g, const BCRec& bc, Real B0,
                                Real dt, int c_rho, int c_mx, int c_my) {
  device_fence();
  fill_ghosts(phi_new, g.domain, bc);
  device_fence();
  const Real half_idr = Real(1) / (Real(2) * g.dr());
  const Real half_idth = Real(1) / (Real(2) * g.dtheta());
  double d = 0;
  for (int li = 0; li < st_new.local_size(); ++li) {
    const ConstArray4 u = st_new.fab(li).const_array();
    const ConstArray4 vr = vrn.fab(li).const_array();
    const ConstArray4 vt = vtn.fab(li).const_array();
    const ConstArray4 p = phi_new.fab(li).const_array();
    const Box2D b = st_new.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        const double rho = u(i, j, c_rho);
        const double vnr = u(i, j, c_mx) / rho, vnt = u(i, j, c_my) / rho;  // v^{n+1}
        // B v^{n+1} : (B v)_r = vr - dt B0 vtheta, (B v)_theta = vtheta + dt B0 vr.
        const double Bvr = vnr - dt * B0 * vnt;
        const double Bvt = vnt + dt * B0 * vnr;
        const double ri = g.r_cell(i);
        const double gr = (p(i + 1, j) - p(i - 1, j)) * half_idr;          // d_r phi
        const double gt = (p(i, j + 1) - p(i, j - 1)) * (half_idth / ri);  // (1/r) d_theta phi
        const double rhsr = vr(i, j, 0) - dt * gr;
        const double rhst = vt(i, j, 0) - dt * gt;
        d = std::fmax(d, std::fmax(std::fabs(Bvr - rhsr), std::fabs(Bvt - rhst)));
      }
  }
  return all_reduce_max(d);
}

// ----------------------------------------------------------------------------------------------
// REFERENCE (B explicite + C fine-dt) : integration du MEME systeme discret polaire. Etat de reference
// porte v (vr, vt) et phi. La derivee de phi exige (-Lap_polar)^{-1} : Lap_polar(phi_dot) = alpha rho0
// div_polar v, resolu par PolarPoissonSolver (direct, Dirichlet phi_dot=0 au bord). grad_polar / div_polar
// CENTRES (memes stencils que le stepper).
// ----------------------------------------------------------------------------------------------

// rhs(i,j) = alpha rho0 div_polar v (pour Lap_polar(phi_dot) = rhs). div_polar v centree :
//   div_polar v = (1/r_i)[(r_{i+1} v_r(i+1) - r_{i-1} v_r(i-1))/(2dr) + (v_th(j+1) - v_th(j-1))/(2dtheta)].
struct DivVPolarKernel {
  ConstArray4 vr, vt;
  Array4 rhs;
  Real coef;  // alpha rho0
  Real half_idr, half_idth, r_min, dr;
  POPS_HD void operator()(int i, int j) const {
    const Real ri = r_min + (i + Real(0.5)) * dr;
    const Real rip = r_min + (i + Real(1.5)) * dr;
    const Real rim = r_min + (i - Real(0.5)) * dr;
    const Real dr_term = (rip * vr(i + 1, j, 0) - rim * vr(i - 1, j, 0)) * half_idr;
    const Real dt_term = (vt(i, j + 1, 0) - vt(i, j - 1, 0)) * half_idth;
    rhs(i, j, 0) = coef * (dr_term + dt_term) / ri;
  }
};

// derivee de v : dvr = -(grad_polar phi)_r + B0 v_theta ; dvtheta = -(grad_polar phi)_theta - B0 v_r.
struct DvPolarKernel {
  ConstArray4 vr, vt, phi;
  Array4 dvr, dvt;
  Real B0, half_idr, half_idth, r_min, dr;
  POPS_HD void operator()(int i, int j) const {
    const Real ri = r_min + (i + Real(0.5)) * dr;
    const Real gr = (phi(i + 1, j, 0) - phi(i - 1, j, 0)) * half_idr;
    const Real gt = (phi(i, j + 1, 0) - phi(i, j - 1, 0)) * (half_idth / ri);
    dvr(i, j, 0) = -gr + B0 * vt(i, j, 0);
    dvt(i, j, 0) = -gt - B0 * vr(i, j, 0);
  }
};

struct AxpyKernel {
  Array4 y;
  ConstArray4 x;
  Real a;
  POPS_HD void operator()(int i, int j) const { y(i, j, 0) += a * x(i, j, 0); }
};
struct ScaleAddKernel {
  Array4 z;
  ConstArray4 base, x;
  Real a;
  POPS_HD void operator()(int i, int j) const { z(i, j, 0) = base(i, j, 0) + a * x(i, j, 0); }
};
struct CopyKernel {
  Array4 d;
  ConstArray4 s;
  POPS_HD void operator()(int i, int j) const { d(i, j, 0) = s(i, j, 0); }
};

struct RefIntegrator {
  Setup& S;
  Real B0, alpha, rho0;
  Real half_idr, half_idth, r_min, dr;
  PolarPoissonSolver poisson;  // Lap_polar(phi_dot) = alpha rho0 div_polar v (Dirichlet)
  MultiFab vr, vt, phi;
  MultiFab dvr, dvt, dphi;
  MultiFab kvr[4], kvt[4], kphi[4], tvr, tvt, tphi;

  RefIntegrator(Setup& s, Real B0_, Real alpha_, Real rho0_)
      : S(s),
        B0(B0_),
        alpha(alpha_),
        rho0(rho0_),
        half_idr(Real(1) / (Real(2) * s.geom.dr())),
        half_idth(Real(1) / (Real(2) * s.geom.dtheta())),
        r_min(s.geom.r_min),
        dr(s.geom.dr()),
        poisson(s.geom, s.ba, s.bc),
        vr(s.ba, s.dm, 1, 1),
        vt(s.ba, s.dm, 1, 1),
        phi(s.ba, s.dm, 1, 1),
        dvr(s.ba, s.dm, 1, 0),
        dvt(s.ba, s.dm, 1, 0),
        dphi(s.ba, s.dm, 1, 1),
        tvr(s.ba, s.dm, 1, 1),
        tvt(s.ba, s.dm, 1, 1),
        tphi(s.ba, s.dm, 1, 1) {
    for (int k = 0; k < 4; ++k) {
      kvr[k] = MultiFab(s.ba, s.dm, 1, 0);
      kvt[k] = MultiFab(s.ba, s.dm, 1, 0);
      kphi[k] = MultiFab(s.ba, s.dm, 1, 0);
    }
  }

  void deriv(MultiFab& avr, MultiFab& avt, MultiFab& aphi, MultiFab& odvr, MultiFab& odvt,
             MultiFab& odphi) {
    device_fence();
    fill_ghosts(avr, S.geom.domain, S.bc);
    fill_ghosts(avt, S.geom.domain, S.bc);
    fill_ghosts(aphi, S.geom.domain, S.bc);
    for (int li = 0; li < avr.local_size(); ++li)
      for_each_cell(odvr.box(li),
                    DvPolarKernel{avr.fab(li).const_array(), avt.fab(li).const_array(),
                                  aphi.fab(li).const_array(), odvr.fab(li).array(),
                                  odvt.fab(li).array(), B0, half_idr, half_idth, r_min, dr});
    // dphi : rhs = alpha rho0 div_polar v, solve Lap_polar(dphi) = rhs (Dirichlet, dphi=0 au bord).
    for (int li = 0; li < poisson.rhs().local_size(); ++li)
      for_each_cell(poisson.rhs().box(li),
                    DivVPolarKernel{avr.fab(li).const_array(), avt.fab(li).const_array(),
                                    poisson.rhs().fab(li).array(), alpha * rho0, half_idr,
                                    half_idth, r_min, dr});
    poisson.phi().set_val(Real(0));
    poisson.solve();
    for (int li = 0; li < odphi.local_size(); ++li)
      for_each_cell(odphi.box(li),
                    CopyKernel{odphi.fab(li).array(), poisson.phi().fab(li).const_array()});
  }

  void rk4_step(Real h) {
    deriv(vr, vt, phi, kvr[0], kvt[0], kphi[0]);
    stage(h * Real(0.5), kvr[0], kvt[0], kphi[0]);
    deriv(tvr, tvt, tphi, kvr[1], kvt[1], kphi[1]);
    stage(h * Real(0.5), kvr[1], kvt[1], kphi[1]);
    deriv(tvr, tvt, tphi, kvr[2], kvt[2], kphi[2]);
    stage(h, kvr[2], kvt[2], kphi[2]);
    deriv(tvr, tvt, tphi, kvr[3], kvt[3], kphi[3]);
    combine(vr, kvr, h);
    combine(vt, kvt, h);
    combine(phi, kphi, h);
  }

  void euler_step(Real h) {
    deriv(vr, vt, phi, kvr[0], kvt[0], kphi[0]);
    axpy(vr, h, kvr[0]);
    axpy(vt, h, kvt[0]);
    axpy(phi, h, kphi[0]);
  }

 private:
  void stage(Real a, MultiFab& kr, MultiFab& kt, MultiFab& kp) {
    for (int li = 0; li < tvr.local_size(); ++li) {
      for_each_cell(tvr.box(li), ScaleAddKernel{tvr.fab(li).array(), vr.fab(li).const_array(),
                                                kr.fab(li).const_array(), a});
      for_each_cell(tvt.box(li), ScaleAddKernel{tvt.fab(li).array(), vt.fab(li).const_array(),
                                                kt.fab(li).const_array(), a});
      for_each_cell(tphi.box(li), ScaleAddKernel{tphi.fab(li).array(), phi.fab(li).const_array(),
                                                 kp.fab(li).const_array(), a});
    }
  }
  void axpy(MultiFab& y, Real a, MultiFab& x) {
    for (int li = 0; li < y.local_size(); ++li)
      for_each_cell(y.box(li), AxpyKernel{y.fab(li).array(), x.fab(li).const_array(), a});
  }
  void combine(MultiFab& y, MultiFab (&k)[4], Real h) {
    const Real s = h / Real(6);
    sync_host();
    for (int li = 0; li < y.local_size(); ++li) {
      Array4 Y = y.fab(li).array();
      const ConstArray4 k0 = k[0].fab(li).const_array(), k1 = k[1].fab(li).const_array(),
                        k2 = k[2].fab(li).const_array(), k3 = k[3].fab(li).const_array();
      const Box2D b = y.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          Y(i, j, 0) +=
              s * (k0(i, j, 0) + Real(2) * k1(i, j, 0) + Real(2) * k2(i, j, 0) + k3(i, j, 0));
    }
  }
};

static void load_ref_from_state(RefIntegrator& R, const MultiFab& st, const MultiFab& phi0,
                                int c_rho, int c_mx, int c_my) {
  sync_host();
  for (int li = 0; li < st.local_size(); ++li) {
    const ConstArray4 u = st.fab(li).const_array();
    Array4 vr = R.vr.fab(li).array(), vt = R.vt.fab(li).array(), p = R.phi.fab(li).array();
    const ConstArray4 p0 = phi0.fab(li).const_array();
    const Box2D b = st.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        const Real rho = u(i, j, c_rho);
        vr(i, j, 0) = u(i, j, c_mx) / rho;
        vt(i, j, 0) = u(i, j, c_my) / rho;
        p(i, j, 0) = p0(i, j, 0);
      }
  }
}

static double vel_rel_state_vs_ref(const MultiFab& st, const RefIntegrator& R, int c_rho, int c_mx,
                                   int c_my) {
  sync_host();
  double num = 0, den = 0;
  for (int li = 0; li < st.local_size(); ++li) {
    const ConstArray4 u = st.fab(li).const_array();
    const ConstArray4 rx = R.vr.fab(li).const_array(), ry = R.vt.fab(li).const_array();
    const Box2D b = st.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        const double rho = u(i, j, c_rho);
        const double ar = u(i, j, c_mx) / rho, at = u(i, j, c_my) / rho;
        const double br = rx(i, j, 0), bt = ry(i, j, 0);
        num += (ar - br) * (ar - br) + (at - bt) * (at - bt);
        den += br * br + bt * bt;
      }
  }
  num = all_reduce_sum(num);
  den = all_reduce_sum(den);
  return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

// Fixture partageant l'anneau (r, theta), le champ Bz constant et le pas explicite stable estime
// -- construction couteuse (grille 48x64 + PolarTensorKrylovSolver) faite UNE fois par suite via
// SetUpTestSuite. Chaque TEST reconstruit son propre etat (st, phi) par make_state() : les sections
// (A)-(D) sont independantes, seules les donnees geometriques/physiques immuables sont partagees.
class PolarCondensedSchurSourceStepperTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    comm_init();
    me_ = my_rank();
    if (me_ == 0) {
      std::printf("=== ETAGE SOURCE Schur POLAIRE (Voie A etape 2b) ===\n");
      std::printf(
          "Anneau r in [%.2f, %.2f], theta in [0, 2pi). PolarTensorKrylovSolver (RadialLine).\n",
          kRmin, kRmax);
    }

    nr_ = 48;
    nth_ = 64;
    rho0_ = 1.5;
    B0_ = 100.0;
    alpha_ = 3.0;  // B0 = 100 : RAIDE (cyclotron eleve)
    S_ = new struct Setup(nr_, nth_);
    const double dr = static_cast<double>(S_->geom.dr());
    const double ds_min = std::min(dr, kRmin * static_cast<double>(S_->geom.dtheta()));
    const double w_plasma = std::sqrt(static_cast<double>(alpha_) * static_cast<double>(rho0_)) / ds_min;
    const double w_max = std::max(static_cast<double>(B0_), w_plasma);
    dt_stable_ = 0.5 / w_max;  // estimation prudente du pas explicite stable
    if (me_ == 0)
      std::printf("[setup] nr=%d nth=%d dr=%.4f B0=%.1f alpha=%.1f rho0=%.1f -> dt_stable~%.4e\n",
                  nr_, nth_, dr, static_cast<double>(B0_), static_cast<double>(alpha_),
                  static_cast<double>(rho0_), dt_stable_);

    vars_ = new VariableSet(fluid_vars(/*with_E=*/true));
    c_rho_ = vars_->index_of(VariableRole::Density);
    c_mx_ = vars_->index_of(VariableRole::MomentumX);
    c_my_ = vars_->index_of(VariableRole::MomentumY);

    bz_ = new MultiFab(S_->ba, S_->dm, 1, 1);
    for (int li = 0; li < bz_->local_size(); ++li)
      for_each_cell(bz_->box(li), ConstBzKernel{bz_->fab(li).array(), B0_});
  }

  static void TearDownTestSuite() {
    delete S_;
    delete vars_;
    delete bz_;
    S_ = nullptr;
    vars_ = nullptr;
    bz_ = nullptr;
    comm_finalize();
  }

  // Construit un etat initial (st, phi) frais sur la grille partagee (profils analytiques InitKernel).
  void make_state(MultiFab& st, MultiFab& phi) const {
    st.set_val(0.0);
    for (int li = 0; li < st.local_size(); ++li)
      for_each_cell(st.box(li),
                    InitKernel{S_->geom, st.fab(li).array(), phi.fab(li).array(), rho0_, c_rho_,
                              c_mx_, c_my_, vars_->index_of(VariableRole::Energy)});
  }

  static int me_;
  static int nr_, nth_;
  static Real rho0_, B0_, alpha_;
  static struct Setup* S_;
  static double dt_stable_;
  static VariableSet* vars_;
  static int c_rho_, c_mx_, c_my_;
  static MultiFab* bz_;
};
int PolarCondensedSchurSourceStepperTest::me_ = 0;
int PolarCondensedSchurSourceStepperTest::nr_ = 0;
int PolarCondensedSchurSourceStepperTest::nth_ = 0;
Real PolarCondensedSchurSourceStepperTest::rho0_ = 0;
Real PolarCondensedSchurSourceStepperTest::B0_ = 0;
Real PolarCondensedSchurSourceStepperTest::alpha_ = 0;
struct Setup* PolarCondensedSchurSourceStepperTest::S_ = nullptr;
double PolarCondensedSchurSourceStepperTest::dt_stable_ = 0;
VariableSet* PolarCondensedSchurSourceStepperTest::vars_ = nullptr;
int PolarCondensedSchurSourceStepperTest::c_rho_ = 0;
int PolarCondensedSchurSourceStepperTest::c_mx_ = 0;
int PolarCondensedSchurSourceStepperTest::c_my_ = 0;
MultiFab* PolarCondensedSchurSourceStepperTest::bz_ = nullptr;

// (A) RELATION IMPLICITE reconstruite a GRAND dt : apres step(), v^{n+1} satisfait
// B v^{n+1} = v^n - dt grad_polar phi^{n+1} a la tolerance du solve.
TEST_F(PolarCondensedSchurSourceStepperTest, implicit_relation_holds_at_large_dt) {
  const Real dt = Real(50.0 * dt_stable_);
  MultiFab st(S_->ba, S_->dm, vars_->size, 1), phi(S_->ba, S_->dm, 1, 1);
  make_state(st, phi);
  MultiFab vrn(S_->ba, S_->dm, 1, 0), vtn(S_->ba, S_->dm, 1, 0);
  for (int li = 0; li < st.local_size(); ++li)
    for_each_cell(st.box(li),
                  detail::ExtractVelocityKernel{st.fab(li).const_array(), vrn.fab(li).array(),
                                                vtn.fab(li).array(), c_rho_, c_mx_, c_my_});

  PolarCondensedSchurSourceStepper stepper(*vars_, S_->geom, S_->ba, S_->bc, alpha_);
  stepper.step(st, phi, *bz_, /*c_bz=*/0, /*theta=*/Real(1.0), dt);
  const PolarKrylovResult kr = stepper.last_solve();

  const double rimp = implicit_residual(st, vrn, vtn, phi, S_->geom, S_->bc, B0_, dt, c_rho_, c_mx_,
                                        c_my_);
  if (me_ == 0)
    std::printf(
        "(A) implicite : BiCGStab %s en %d iters (rel=%.2e) | max|B v - (v^n - dt grad_polar "
        "phi)| = %.3e\n",
        kr.converged ? "CONVERGE" : "ECHOUE", kr.iters, static_cast<double>(kr.rel_residual), rimp);
  EXPECT_TRUE(kr.converged) << "A_solve_converge";
  EXPECT_TRUE(rimp < 1e-6) << "A_relation_implicite: rimp=" << rimp;
}

// (B) STABILITE vs EXPLOSION explicite, a GRAND dt (50x le pas explicite stable, K pas) : l'Euler
// explicite EXPLOSE (||v|| croit de plusieurs ordres) ; le Schur garde ||v|| BORNEE.
TEST_F(PolarCondensedSchurSourceStepperTest, schur_stays_bounded_while_explicit_euler_explodes) {
  const Real dt = Real(8.0 * dt_stable_);
  const int K = 12;
  MultiFab st(S_->ba, S_->dm, vars_->size, 1), phi(S_->ba, S_->dm, 1, 1);
  make_state(st, phi);
  const double v0 = vel_l2(st, c_rho_, c_mx_, c_my_);

  PolarCondensedSchurSourceStepper stepper(*vars_, S_->geom, S_->ba, S_->bc, alpha_);
  for (int k = 0; k < K; ++k)
    stepper.step(st, phi, *bz_, 0, Real(1.0), dt);
  const double v_schur = vel_l2(st, c_rho_, c_mx_, c_my_);

  MultiFab st0(S_->ba, S_->dm, vars_->size, 1), phi0(S_->ba, S_->dm, 1, 1);
  make_state(st0, phi0);
  RefIntegrator expl(*S_, B0_, alpha_, rho0_);
  load_ref_from_state(expl, st0, phi0, c_rho_, c_mx_, c_my_);
  for (int k = 0; k < K; ++k)
    expl.euler_step(dt);
  double v_expl = 0;
  {
    sync_host();
    double s = 0;
    for (int li = 0; li < expl.vr.local_size(); ++li) {
      const ConstArray4 vr = expl.vr.fab(li).const_array(), vt = expl.vt.fab(li).const_array();
      const Box2D b = expl.vr.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          s += static_cast<double>(vr(i, j, 0)) * vr(i, j, 0) +
               static_cast<double>(vt(i, j, 0)) * vt(i, j, 0);
    }
    v_expl = std::sqrt(all_reduce_sum(s));
  }
  if (me_ == 0)
    std::printf(
        "(B) dt=%.3e (=8x dt_stable), K=%d pas : ||v0||=%.3e | Schur ||v||=%.3e (x%.2f) | "
        "Euler ||v||=%.3e (x%.2e)\n",
        static_cast<double>(dt), K, v0, v_schur, v_schur / v0, v_expl, v_expl / v0);
  EXPECT_TRUE(v_schur < 5.0 * v0) << "B_schur_stable_borne: v_schur=" << v_schur << " v0=" << v0;
  EXPECT_TRUE(std::isfinite(v_schur)) << "B_schur_fini: v_schur=" << v_schur;
  EXPECT_TRUE(v_expl > 100.0 * v0) << "B_explicite_explose: v_expl=" << v_expl << " v0=" << v0;
  EXPECT_TRUE(v_expl > 50.0 * v_schur)
      << "B_explicite_pire_que_schur: v_expl=" << v_expl << " v_schur=" << v_schur;
}

// (C) ACCORD AVEC UNE REFERENCE fine-dt (RK4) a dt MODERE : le Schur (theta=1, ordre 1) en est
// proche et l'ecart DECROIT a l'ordre 1 (dt, dt/2 -> ratio > 1.5).
TEST_F(PolarCondensedSchurSourceStepperTest, matches_fine_dt_rk4_reference_at_first_order) {
  auto schur_vs_ref = [&](Real dt) -> double {
    const int Nsub = 256;
    MultiFab st0(S_->ba, S_->dm, vars_->size, 1), phi0(S_->ba, S_->dm, 1, 1);
    make_state(st0, phi0);
    RefIntegrator ref(*S_, B0_, alpha_, rho0_);
    load_ref_from_state(ref, st0, phi0, c_rho_, c_mx_, c_my_);
    const Real h = dt / Real(Nsub);
    for (int s = 0; s < Nsub; ++s)
      ref.rk4_step(h);

    MultiFab st(S_->ba, S_->dm, vars_->size, 1), phi(S_->ba, S_->dm, 1, 1);
    make_state(st, phi);
    PolarCondensedSchurSourceStepper stepper(*vars_, S_->geom, S_->ba, S_->bc, alpha_);
    stepper.step(st, phi, *bz_, 0, Real(1.0), dt);
    return vel_rel_state_vs_ref(st, ref, c_rho_, c_mx_, c_my_);
  };

  const Real dtC = Real(0.5 * dt_stable_);
  const double e1 = schur_vs_ref(dtC);
  const double e2 = schur_vs_ref(Real(0.5) * dtC);
  const double ratio = e2 > 0 ? e1 / e2 : 0;
  if (me_ == 0)
    std::printf(
        "(C) reference RK4 fine : err(dt)=%.3e err(dt/2)=%.3e ratio=%.2f (ordre 1 attendu ~2)\n",
        e1, e2, ratio);
  EXPECT_TRUE(e1 < 5e-2) << "C_accord_reference_modere: e1=" << e1;
  EXPECT_TRUE(ratio > 1.5) << "C_ordre_un_decroissance: ratio=" << ratio;
}

// (D) REGRESSION seam theta (adc_cases ADC-62) : un BCRec "a la System::poisson_bc" (Dirichlet sur
// les QUATRE faces, y compris theta) doit produire un pas BIT-IDENTIQUE au BCRec canonique (theta
// periodique) : l'anneau n'a pas de bord physique azimutal, le stepper et le solveur NORMALISENT
// (phi_bc / force_theta_periodic). Avant le fix, les ghosts azimutaux de phi etaient remplis par
// reflexion impaire (ghost = -phi) au seam theta=0/2pi -> dipole parasite O(phi/(r dtheta)) dans
// mom_r aux deux colonnes du seam (||R_eq||~83 du cas Hoffart polaire, divergence a t~0.01).
// theta=0.5 pour exercer aussi l'extrapolation pas-plein.
TEST_F(PolarCondensedSchurSourceStepperTest, dirichlet_theta_bc_matches_periodic_seam_bit_identical) {
  const Real dt = Real(8.0 * dt_stable_);
  auto one_step = [&](const BCRec& bc, MultiFab& st, MultiFab& phi) {
    make_state(st, phi);
    PolarCondensedSchurSourceStepper stepper(*vars_, S_->geom, S_->ba, bc, alpha_);
    stepper.step(st, phi, *bz_, 0, Real(0.5), dt);
  };
  MultiFab stP(S_->ba, S_->dm, vars_->size, 1), phiP(S_->ba, S_->dm, 1, 1);
  one_step(S_->bc, stP, phiP);  // canonique : Dirichlet radial, theta periodique
  BCRec bcSys = S_->bc;         // "System::poisson_bc" : Dirichlet sur les 4 faces
  bcSys.ylo = bcSys.yhi = BCType::Dirichlet;
  bcSys.ylo_val = bcSys.yhi_val = 0.0;
  MultiFab stD(S_->ba, S_->dm, vars_->size, 1), phiD(S_->ba, S_->dm, 1, 1);
  one_step(bcSys, stD, phiD);

  sync_host();
  double dmax_st = 0, dmax_phi = 0;
  for (int li = 0; li < stP.local_size(); ++li) {
    const ConstArray4 a = stP.fab(li).const_array(), b = stD.fab(li).const_array();
    const ConstArray4 pa = phiP.fab(li).const_array(), pb = phiD.fab(li).const_array();
    const Box2D bx = stP.box(li);
    for (int j = bx.lo[1]; j <= bx.hi[1]; ++j)
      for (int i = bx.lo[0]; i <= bx.hi[0]; ++i) {
        for (int c = 0; c < vars_->size; ++c)
          dmax_st = std::max(dmax_st, std::abs(static_cast<double>(a(i, j, c)) - b(i, j, c)));
        dmax_phi = std::max(dmax_phi, std::abs(static_cast<double>(pa(i, j, 0)) - pb(i, j, 0)));
      }
  }
  dmax_st = all_reduce_max(dmax_st);
  dmax_phi = all_reduce_max(dmax_phi);
  if (me_ == 0)
    std::printf(
        "(D) seam theta : max|U_dir4 - U_per| = %.3e | max|phi_dir4 - phi_per| = %.3e "
        "(0 attendu, BC theta normalisee)\n",
        dmax_st, dmax_phi);
  EXPECT_TRUE(dmax_st == 0.0) << "D_seam_etat_bit_identique: dmax_st=" << dmax_st;
  EXPECT_TRUE(dmax_phi == 0.0) << "D_seam_phi_bit_identique: dmax_phi=" << dmax_phi;
}
