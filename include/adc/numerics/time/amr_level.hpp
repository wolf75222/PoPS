#pragma once
// Oracle AMR mono-box MultiFab (detail::AmrLevelMF + subcycle_level_mf) : validation MF->MP.

#include <adc/numerics/time/amr_flux_helpers.hpp>

/// @file
/// @brief Pile AMR MultiFab mono-box : struct detail::AmrLevelMF et le moteur recursif
///        (amr_step_2level_mf, subcycle_level_mf, amr_step_multilevel_mf). ORACLE DE TEST,
///        hors production.
///
/// Couche : `include/adc/numerics/time`.
/// Role : moteur AMR MONO-BOX d'origine, conserve en detail:: comme oracle de validation.
///        Maillon intermediaire de la chaine de parite Fab2D (amr_reflux.hpp) -> MF (ici) ->
///        MP (amr_subcycling.hpp), qui prouve le moteur multi-patch bit-identique au mono-box
///        jusqu'a N niveaux (ce que le test Fab2D 2-niveaux/1-comp/Rusanov seul ne couvre pas).
/// Contrat : un pas conservatif a sous-cyclage Berger-Oliger (ratio 2, dt/2 par sous-pas) avec
///           reflux a chaque interface coarse-fine ; CL periodiques au niveau 0.
///
/// Invariants :
/// - plus appele en production (AmrCoupler passe par advance_amr) ;
/// - gardes de parite : test_amr_reflux_mf (vs Fab2D), test_amr_multilevel_mf,
///   test_amr_multilevel_multipatch garde 1 (MP vs MF, max|dUc| == 0) ;
/// - aux detenu ailleurs (pointeur) ; rC* = region (coords de CE niveau) raffinee par l'enfant,
///   valable seulement si has_fine.

namespace adc {

// === PILE MF : ORACLE DE TEST, HORS PRODUCTION ===================================
// amr_step_2level_mf / amr_step_multilevel_mf + AmrLevelMF : le moteur AMR MONO-BOX
// d'origine. Plus appele en production (AmrCoupler passe par advance_amr, le moteur
// unifie multi-patch). Conserve en detail:: comme ORACLE de validation : la chaine
// Fab2D (amr_reflux.hpp::amr_step_2level) -> MF (ici) -> MP (detail::subcycle_level_mp)
// prouve le moteur MP bit-identique au mono-box, jusqu'a N niveaux (ce que Fab2D seul,
// 2-niveaux/1-comp/Rusanov, ne couvre pas). Gardes : test_amr_reflux_mf (vs Fab2D),
// test_amr_multilevel_mf, test_amr_multilevel_multipatch garde 1 (MP vs MF, max|dUc|=0).
// Les helpers partages restent a portee adc:: (mf_advance_faces, mf_average_down,
// mf_fill_fine_ghosts_t), utilises aussi par les tests d'init.
namespace detail {

// Un niveau de la hierarchie MultiFab. aux detenu ailleurs (pointeur). rC* = region
// (coords de CE niveau) raffinee par l'enfant ; valable si has_fine.
struct AmrLevelMF {
  MultiFab U;
  const MultiFab* aux;
  Real dx, dy;
  int rCI0, rCI1, rCJ0, rCJ1;
  bool has_fine;
};

// Un pas 2-niveaux conservatif. Uc : grossier (domaine periodique, ghosts pour le
// Limiter). Uf : fin (box raffinee). auxc/auxf : (phi, grad phi) prescrits, ghosts
// remplis. dt = pas grossier ; le fin fait r=2 sous-pas de dt/2 puis reflux.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void amr_step_2level_mf(const Model& m, MultiFab& Uc, const Box2D& dom, Real dxc,
                        Real dyc, MultiFab& Uf, int CI0, int CI1, int CJ0, int CJ1,
                        const MultiFab& auxc, const MultiFab& auxf, Real dt) {
  const int r = 2, nc = Uc.ncomp();
  const Real dxf = dxc / 2, dyf = dyc / 2, dtf = dt / r;
  const int nJ = CJ1 - CJ0 + 1, nI = CI1 - CI0 + 1;
  MultiFab Uc_old = Uc;  // etat grossier au temps t (interp temporelle des ghosts fins)

  // --- flux grossiers aux 4 faces de la region fine (avant maj) ---
  fill_boundary(Uc, dom, Periodicity{true, true});
  MultiFab fxc(BoxArray(std::vector<Box2D>{xface_box(Uc.box(0))}), Uc.dmap(), nc, 0);
  MultiFab fyc(BoxArray(std::vector<Box2D>{yface_box(Uc.box(0))}), Uc.dmap(), nc, 0);
  compute_face_fluxes<Limiter, NumericalFlux>(m, Uc, auxc, fxc, fyc, dxc, dyc);
  std::vector<Real> cL(nJ * nc), cR(nJ * nc), cB(nI * nc), cT(nI * nc);
  {
    device_fence();
    const ConstArray4 FX = fxc.fab(0).const_array(), FY = fyc.fab(0).const_array();
    for (int J = CJ0; J <= CJ1; ++J)
      for (int k = 0; k < nc; ++k) {
        cL[(J - CJ0) * nc + k] = FX(CI0, J, k);
        cR[(J - CJ0) * nc + k] = FX(CI1 + 1, J, k);
      }
    for (int I = CI0; I <= CI1; ++I)
      for (int k = 0; k < nc; ++k) {
        cB[(I - CI0) * nc + k] = FY(I, CJ0, k);
        cT[(I - CI0) * nc + k] = FY(I, CJ1 + 1, k);
      }
  }
  mf_advance_faces(Uc, fxc, fyc, dxc, dyc, dt);  // Uc -> etat t+dt
  mf_apply_source(m, Uc, auxc, dt);  // source S(U,aux) au sous-pas

  // --- sous-cyclage fin : r sous-pas, accumulation des flux fins (x dtf) ---
  std::vector<Real> fL(nJ * nc, 0), fR(nJ * nc, 0), fB(nI * nc, 0), fT(nI * nc, 0);
  MultiFab fxf(BoxArray(std::vector<Box2D>{xface_box(Uf.box(0))}), Uf.dmap(), nc, 0);
  MultiFab fyf(BoxArray(std::vector<Box2D>{yface_box(Uf.box(0))}), Uf.dmap(), nc, 0);
  for (int s = 0; s < r; ++s) {
    mf_fill_fine_ghosts_t(Uf, Uc_old, Uc, Real(s) / r);
    compute_face_fluxes<Limiter, NumericalFlux>(m, Uf, auxf, fxf, fyf, dxf, dyf);
    device_fence();
    const ConstArray4 FX = fxf.fab(0).const_array(), FY = fyf.fab(0).const_array();
    for (int J = CJ0; J <= CJ1; ++J)
      for (int k = 0; k < nc; ++k) {
        fL[(J - CJ0) * nc + k] +=
            Real(0.5) * (FX(2 * CI0, 2 * J, k) + FX(2 * CI0, 2 * J + 1, k)) * dtf;
        fR[(J - CJ0) * nc + k] += Real(0.5) *
            (FX(2 * CI1 + 2, 2 * J, k) + FX(2 * CI1 + 2, 2 * J + 1, k)) * dtf;
      }
    for (int I = CI0; I <= CI1; ++I)
      for (int k = 0; k < nc; ++k) {
        fB[(I - CI0) * nc + k] +=
            Real(0.5) * (FY(2 * I, 2 * CJ0, k) + FY(2 * I + 1, 2 * CJ0, k)) * dtf;
        fT[(I - CI0) * nc + k] += Real(0.5) *
            (FY(2 * I, 2 * CJ1 + 2, k) + FY(2 * I + 1, 2 * CJ1 + 2, k)) * dtf;
      }
    mf_advance_faces(Uf, fxf, fyf, dxf, dyf, dtf);
    mf_apply_source(m, Uf, auxf, dtf);  // source S(U,aux) au sous-pas
  }

  mf_average_down(Uf, Uc, CI0, CI1, CJ0, CJ1);  // sync des cellules couvertes

  // --- reflux : flux grossier (x dt) remplace par somme des flux fins (x dtf) ---
  device_fence();
  Array4 c = Uc.fab(0).array();
  for (int J = CJ0; J <= CJ1; ++J)
    for (int k = 0; k < nc; ++k) {
      c(CI0 - 1, J, k) -= (fL[(J - CJ0) * nc + k] - cL[(J - CJ0) * nc + k] * dt) / dxc;
      c(CI1 + 1, J, k) += (fR[(J - CJ0) * nc + k] - cR[(J - CJ0) * nc + k] * dt) / dxc;
    }
  for (int I = CI0; I <= CI1; ++I)
    for (int k = 0; k < nc; ++k) {
      c(I, CJ0 - 1, k) -= (fB[(I - CI0) * nc + k] - cB[(I - CI0) * nc + k] * dt) / dyc;
      c(I, CJ1 + 1, k) += (fT[(I - CI0) * nc + k] - cT[(I - CI0) * nc + k] * dt) / dyc;
    }
}

// --- recursion N-niveaux (pendant MultiFab de amr_multilevel.hpp) ---

// Avance recursivement le niveau lev de dt (sous-cyclage Berger-Oliger r=2 + reflux).
// pOld/pNew = etats parent bornant le pas ; preg* = registre du parent (flux fins de
// CE niveau), nul si lev==0. Generique (Limiter, NumericalFlux, N comp), seam GPU.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void subcycle_level_mf(const Model& m, std::vector<AmrLevelMF>& L, int lev, Real dt,
                       const Box2D& dom, const MultiFab* pOld, const MultiFab* pNew,
                       Real frac, std::vector<Real>* pregL, std::vector<Real>* pregR,
                       std::vector<Real>* pregB, std::vector<Real>* pregT) {
  const int r = 2;
  AmrLevelMF& lv = L[lev];
  const int nc = lv.U.ncomp();

  if (lev == 0)
    fill_boundary(lv.U, dom, Periodicity{true, true});
  else
    mf_fill_fine_ghosts_t(lv.U, *pOld, *pNew, frac);

  MultiFab fx(BoxArray(std::vector<Box2D>{xface_box(lv.U.box(0))}), lv.U.dmap(), nc, 0);
  MultiFab fy(BoxArray(std::vector<Box2D>{yface_box(lv.U.box(0))}), lv.U.dmap(), nc, 0);
  compute_face_fluxes<Limiter, NumericalFlux>(m, lv.U, *lv.aux, fx, fy, lv.dx, lv.dy);

  if (lev > 0) {  // contribution au registre du parent (flux fins x dt)
    device_fence();
    const ConstArray4 FX = fx.fab(0).const_array(), FY = fy.fab(0).const_array();
    const AmrLevelMF& par = L[lev - 1];
    const int pI0 = par.rCI0, pI1 = par.rCI1, pJ0 = par.rCJ0, pJ1 = par.rCJ1;
    for (int J = pJ0; J <= pJ1; ++J)
      for (int k = 0; k < nc; ++k) {
        (*pregL)[(J - pJ0) * nc + k] +=
            Real(0.5) * (FX(2 * pI0, 2 * J, k) + FX(2 * pI0, 2 * J + 1, k)) * dt;
        (*pregR)[(J - pJ0) * nc + k] += Real(0.5) *
            (FX(2 * pI1 + 2, 2 * J, k) + FX(2 * pI1 + 2, 2 * J + 1, k)) * dt;
      }
    for (int I = pI0; I <= pI1; ++I)
      for (int k = 0; k < nc; ++k) {
        (*pregB)[(I - pI0) * nc + k] +=
            Real(0.5) * (FY(2 * I, 2 * pJ0, k) + FY(2 * I + 1, 2 * pJ0, k)) * dt;
        (*pregT)[(I - pI0) * nc + k] += Real(0.5) *
            (FY(2 * I, 2 * pJ1 + 2, k) + FY(2 * I + 1, 2 * pJ1 + 2, k)) * dt;
      }
  }

  if (!lv.has_fine) {
    mf_advance_faces(lv.U, fx, fy, lv.dx, lv.dy, dt);  // feuille
    mf_apply_source(m, lv.U, *lv.aux, dt);  // source S(U,aux) au sous-pas
    return;
  }

  // niveau avec enfant : registre local + flux grossier sauve (sans dt)
  const int cI0 = lv.rCI0, cI1 = lv.rCI1, cJ0 = lv.rCJ0, cJ1 = lv.rCJ1;
  const int nI = cI1 - cI0 + 1, nJ = cJ1 - cJ0 + 1;
  std::vector<Real> cL(nJ * nc), cR(nJ * nc), cB(nI * nc), cT(nI * nc);
  {
    device_fence();
    const ConstArray4 FX = fx.fab(0).const_array(), FY = fy.fab(0).const_array();
    for (int J = cJ0; J <= cJ1; ++J)
      for (int k = 0; k < nc; ++k) {
        cL[(J - cJ0) * nc + k] = FX(cI0, J, k);
        cR[(J - cJ0) * nc + k] = FX(cI1 + 1, J, k);
      }
    for (int I = cI0; I <= cI1; ++I)
      for (int k = 0; k < nc; ++k) {
        cB[(I - cI0) * nc + k] = FY(I, cJ0, k);
        cT[(I - cI0) * nc + k] = FY(I, cJ1 + 1, k);
      }
  }
  std::vector<Real> fL(nJ * nc, 0), fR(nJ * nc, 0), fB(nI * nc, 0), fT(nI * nc, 0);

  MultiFab U_old = lv.U;  // etat t (interp temporelle de l'enfant)
  mf_advance_faces(lv.U, fx, fy, lv.dx, lv.dy, dt);  // lv.U -> t+dt
  mf_apply_source(m, lv.U, *lv.aux, dt);  // source S(U,aux) au sous-pas

  for (int s = 0; s < r; ++s)
    subcycle_level_mf<Limiter, NumericalFlux>(m, L, lev + 1, dt / r, dom, &U_old,
                                              &lv.U, Real(s) / r, &fL, &fR, &fB, &fT);

  mf_average_down(L[lev + 1].U, lv.U, cI0, cI1, cJ0, cJ1);

  device_fence();
  Array4 c = lv.U.fab(0).array();
  for (int J = cJ0; J <= cJ1; ++J)
    for (int k = 0; k < nc; ++k) {
      c(cI0 - 1, J, k) -= (fL[(J - cJ0) * nc + k] - cL[(J - cJ0) * nc + k] * dt) / lv.dx;
      c(cI1 + 1, J, k) += (fR[(J - cJ0) * nc + k] - cR[(J - cJ0) * nc + k] * dt) / lv.dx;
    }
  for (int I = cI0; I <= cI1; ++I)
    for (int k = 0; k < nc; ++k) {
      c(I, cJ0 - 1, k) -= (fB[(I - cI0) * nc + k] - cB[(I - cI0) * nc + k] * dt) / lv.dy;
      c(I, cJ1 + 1, k) += (fT[(I - cI0) * nc + k] - cT[(I - cI0) * nc + k] * dt) / lv.dy;
    }
}

// Driver : un pas dt de la hierarchie complete (niveau 0 = grossier).
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void amr_step_multilevel_mf(const Model& m, std::vector<AmrLevelMF>& L,
                            const Box2D& dom, Real dt) {
  subcycle_level_mf<Limiter, NumericalFlux>(m, L, 0, dt, dom, nullptr, nullptr,
                                            Real(0), nullptr, nullptr, nullptr, nullptr);
}

}  // namespace detail (oracle MF mono-box : Fab2D -> MF -> MP)

}  // namespace adc
