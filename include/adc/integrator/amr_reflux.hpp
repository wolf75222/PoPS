#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/operator/spatial_operator.hpp>

#include <vector>

// AMR dans le temps, brique conservation-critique : advance 2-niveaux avec
// reflux. Le flux numerique a l'interface coarse-fine est incoherent entre la
// grille grossiere et la grille fine ; le reflux corrige les cellules grossieres
// adjacentes pour retablir la conservation exacte (FluxRegister, facon AMReX).
//
// Version minimale et testable : 1 composante (advection), vitesse constante
// (aux uniforme), Euler explicite, ratio 2, une box fine rectangulaire stricte-
// ment interieure au domaine grossier periodique. Le sous-cyclage en temps et le
// couplage Poisson composite (FAC) viendront ensuite ; le test de conservation
// valide ici l'arithmetique du reflux.

namespace adc {

inline Box2D xface_box(const Box2D& v) {
  return Box2D{{v.lo[0], v.lo[1]}, {v.hi[0] + 1, v.hi[1]}};
}
inline Box2D yface_box(const Box2D& v) {
  return Box2D{{v.lo[0], v.lo[1]}, {v.hi[0], v.hi[1] + 1}};
}

// flux de Rusanov premier ordre, 1 composante, aux uniforme, sur un Fab2D.
// fx(i,j) = flux a la face gauche de la cellule i ; fy(i,j) = face basse de j.
template <class Model>
void compute_fluxes_1c(const Model& m, const Fab2D& U, const Aux& a, Fab2D& fx,
                       Fab2D& fy) {
  const ConstArray4 u = U.const_array();
  {
    Array4 F = fx.array();
    const Box2D b = fx.box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        typename Model::State UL{}, UR{};
        UL[0] = u(i - 1, j);
        UR[0] = u(i, j);
        F(i, j) = rusanov_flux(m, UL, a, UR, a, 0)[0];
      }
  }
  {
    Array4 F = fy.array();
    const Box2D b = fy.box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        typename Model::State UL{}, UR{};
        UL[0] = u(i, j - 1);
        UR[0] = u(i, j);
        F(i, j) = rusanov_flux(m, UL, a, UR, a, 1)[0];
      }
  }
}

// Euler explicite : U -= dt div(F). Les ghosts de U doivent etre remplis.
template <class Model>
void advance_fab_1c(const Model& m, Fab2D& U, const Aux& a, double dx, double dy,
                    double dt, Fab2D& fx, Fab2D& fy) {
  compute_fluxes_1c(m, U, a, fx, fy);
  Array4 uu = U.array();
  const ConstArray4 FX = fx.const_array();
  const ConstArray4 FY = fy.const_array();
  const Box2D v = U.box();
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i)
      uu(i, j) -= dt * ((FX(i + 1, j) - FX(i, j)) / dx +
                        (FY(i, j + 1) - FY(i, j)) / dy);
}

// ghosts periodiques pour un Fab2D unique couvrant le domaine.
inline void fill_periodic_fab(Fab2D& U, const Box2D& dom) {
  const int ng = U.n_ghost();
  const int nx = dom.nx(), ny = dom.ny();
  Array4 a = U.array();
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int g = 1; g <= ng; ++g) {
      a(dom.lo[0] - g, j) = a(dom.hi[0] - g + 1, j);
      a(dom.hi[0] + g, j) = a(dom.lo[0] + g - 1, j);
    }
  for (int i = dom.lo[0] - ng; i <= dom.hi[0] + ng; ++i)
    for (int g = 1; g <= ng; ++g) {
      a(i, dom.lo[1] - g) = a(i, dom.hi[1] - g + 1);
      a(i, dom.hi[1] + g) = a(i, dom.lo[1] + g - 1);
    }
}

// ghosts du fin par injection depuis le grossier (ratio 2).
inline void fill_fine_ghosts(Fab2D& Uf, const Fab2D& Uc) {
  const ConstArray4 c = Uc.const_array();
  Array4 f = Uf.array();
  const Box2D g = Uf.grown_box();
  const Box2D v = Uf.box();
  auto coarsen = [](int x) { return (x >= 0) ? x / 2 : -((-x + 1) / 2); };
  for (int j = g.lo[1]; j <= g.hi[1]; ++j)
    for (int i = g.lo[0]; i <= g.hi[0]; ++i)
      if (!v.contains(i, j)) f(i, j) = c(coarsen(i), coarsen(j));
}

// moyenne fin -> grossier sur la region couverte (ratio 2).
inline void average_down_fab(const Fab2D& Uf, Fab2D& Uc, int CI0, int CI1,
                             int CJ0, int CJ1) {
  const ConstArray4 f = Uf.const_array();
  Array4 c = Uc.array();
  for (int J = CJ0; J <= CJ1; ++J)
    for (int I = CI0; I <= CI1; ++I)
      c(I, J) = 0.25 * (f(2 * I, 2 * J) + f(2 * I + 1, 2 * J) +
                        f(2 * I, 2 * J + 1) + f(2 * I + 1, 2 * J + 1));
}

// Un pas 2-niveaux conservatif (non sous-cycle : dt commun) avec reflux.
// region fine = cellules grossieres [CI0..CI1] x [CJ0..CJ1] (strictement
// interieures), raffinees en [2CI0..2CI1+1] x [...]. dom = domaine grossier.
template <class Model>
void amr_step_2level(const Model& m, Fab2D& Uc, const Box2D& dom, double dxc,
                     double dyc, Fab2D& Uf, int CI0, int CI1, int CJ0, int CJ1,
                     const Aux& a, double dt) {
  const double dxf = dxc / 2, dyf = dyc / 2;

  // --- flux grossiers (avant mise a jour) ---
  fill_periodic_fab(Uc, dom);
  Fab2D fxc(xface_box(Uc.box()), 1, 0), fyc(yface_box(Uc.box()), 1, 0);
  compute_fluxes_1c(m, Uc, a, fxc, fyc);

  // sauve les flux grossiers aux 4 faces de la region fine
  const ConstArray4 FXc = fxc.const_array();
  const ConstArray4 FYc = fyc.const_array();
  std::vector<double> cL(CJ1 - CJ0 + 1), cR(CJ1 - CJ0 + 1);
  std::vector<double> cB(CI1 - CI0 + 1), cT(CI1 - CI0 + 1);
  for (int J = CJ0; J <= CJ1; ++J) {
    cL[J - CJ0] = FXc(CI0, J);      // face gauche de CI0
    cR[J - CJ0] = FXc(CI1 + 1, J);  // face droite de CI1
  }
  for (int I = CI0; I <= CI1; ++I) {
    cB[I - CI0] = FYc(I, CJ0);      // face basse de CJ0
    cT[I - CI0] = FYc(I, CJ1 + 1);  // face haute de CJ1
  }

  // --- mise a jour grossiere (tout le domaine) ---
  advance_fab_1c(m, Uc, a, dxc, dyc, dt, fxc, fyc);

  // --- mise a jour fine + accumulation des flux fins aux 4 faces ---
  std::vector<double> fL(CJ1 - CJ0 + 1, 0), fR(CJ1 - CJ0 + 1, 0);
  std::vector<double> fB(CI1 - CI0 + 1, 0), fT(CI1 - CI0 + 1, 0);
  fill_fine_ghosts(Uf, Uc);  // Uc est deja avance ; injection (1er ordre en temps)
  Fab2D fxf(xface_box(Uf.box()), 1, 0), fyf(yface_box(Uf.box()), 1, 0);
  compute_fluxes_1c(m, Uf, a, fxf, fyf);
  const ConstArray4 FXf = fxf.const_array();
  const ConstArray4 FYf = fyf.const_array();
  // moyenne spatiale des flux fins sur chaque face grossiere
  for (int J = CJ0; J <= CJ1; ++J) {
    fL[J - CJ0] = 0.5 * (FXf(2 * CI0, 2 * J) + FXf(2 * CI0, 2 * J + 1));
    fR[J - CJ0] =
        0.5 * (FXf(2 * CI1 + 2, 2 * J) + FXf(2 * CI1 + 2, 2 * J + 1));
  }
  for (int I = CI0; I <= CI1; ++I) {
    fB[I - CI0] = 0.5 * (FYf(2 * I, 2 * CJ0) + FYf(2 * I + 1, 2 * CJ0));
    fT[I - CI0] =
        0.5 * (FYf(2 * I, 2 * CJ1 + 2) + FYf(2 * I + 1, 2 * CJ1 + 2));
  }
  advance_fab_1c(m, Uf, a, dxf, dyf, dt, fxf, fyf);

  // --- average down (sync des cellules grossieres couvertes) ---
  average_down_fab(Uf, Uc, CI0, CI1, CJ0, CJ1);

  // --- reflux des cellules grossieres adjacentes (cote grossier) ---
  Array4 c = Uc.array();
  for (int J = CJ0; J <= CJ1; ++J) {
    // cellule grossiere a gauche de la region : CI0-1, face droite = face
    // gauche de CI0. flux grossier cL, flux fin fL. correction +(fin-grossier).
    c(CI0 - 1, J) -= dt / dxc * (fL[J - CJ0] - cL[J - CJ0]);
    // cellule grossiere a droite : CI1+1, face gauche = face droite de CI1.
    c(CI1 + 1, J) += dt / dxc * (fR[J - CJ0] - cR[J - CJ0]);
  }
  for (int I = CI0; I <= CI1; ++I) {
    c(I, CJ0 - 1) -= dt / dyc * (fB[I - CI0] - cB[I - CI0]);
    c(I, CJ1 + 1) += dt / dyc * (fT[I - CI0] - cT[I - CI0]);
  }
}

}  // namespace adc
