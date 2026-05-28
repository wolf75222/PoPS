#pragma once

#include <adc/core/types.hpp>

#include <array>

// State et Aux : les deux types ponctuels manipules par la couche physique.
//
// Regle d'architecture : ce sont des agregats trivialement copiables (POD).
// Un PhysicalModel ne voit jamais autre chose que ca. Aucune box, aucun rang
// MPI, aucune vue Kokkos n'entre ici. C'est ce qui rend la physique portable :
// le meme code compile pour CPU et device parce qu'il ne touche a aucun
// parallelisme.

namespace adc {

// Vecteur d'etat conserve de taille fixe (connue a la compilation).
// Pour un scalaire (advection / diocotron) : StateVec<1>.
// Pour Euler 2D : StateVec<4>. La taille pilote n_vars du modele.
template <int N>
struct StateVec {
  std::array<Real, N> v{};

  Real& operator[](int i) { return v[i]; }
  Real operator[](int i) const { return v[i]; }

  static constexpr int size() { return N; }
};

template <int N>
StateVec<N> operator+(StateVec<N> a, const StateVec<N>& b) {
  for (int i = 0; i < N; ++i) a[i] += b[i];
  return a;
}

template <int N>
StateVec<N> operator-(StateVec<N> a, const StateVec<N>& b) {
  for (int i = 0; i < N; ++i) a[i] -= b[i];
  return a;
}

template <int N>
StateVec<N> operator*(Real s, StateVec<N> a) {
  for (int i = 0; i < N; ++i) a[i] *= s;
  return a;
}

// Champs auxiliaires derives de la resolution elliptique : le potentiel et
// son gradient au point. C'est le canal unique par lequel le couplage entre
// dans la physique. Il alimente A LA FOIS le flux (cas diocotron : la vitesse
// E x B vient de grad phi) et la source (cas Euler-Poisson : S = -rho grad phi).
// Cette dualite est ce qui permet a un seul operateur spatial de servir les
// deux problemes cibles.
struct Aux {
  Real phi{};     // potentiel
  Real grad_x{};  // d phi / d x
  Real grad_y{};  // d phi / d y
};

}  // namespace adc
