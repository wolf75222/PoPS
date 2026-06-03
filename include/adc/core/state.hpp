#pragma once

#include <adc/core/types.hpp>

// State et Aux : les deux types ponctuels manipules par la couche physique.
//
// Regle d'architecture : ce sont des agregats trivialement copiables (POD).
// Un PhysicalModel ne voit jamais autre chose que ca. Aucune box, aucun rang
// MPI, aucune vue Kokkos n'entre ici. C'est ce qui rend la physique portable :
// le meme code compile pour CPU et device parce qu'il ne touche a aucun
// parallelisme.

namespace adc {

// Vecteur d'etat conserve de taille fixe (connue a la compilation).
// Pour un scalaire (advection / transport a derive) : StateVec<1>.
// Pour Euler 2D : StateVec<4>. La taille pilote n_vars du modele.
template <int N>
struct StateVec {
  Real v[N]{};  // tableau C : trivialement utilisable sur device (pas std::array)

  ADC_HD Real& operator[](int i) { return v[i]; }
  ADC_HD Real operator[](int i) const { return v[i]; }

  ADC_HD static constexpr int size() { return N; }
};

template <int N>
ADC_HD StateVec<N> operator+(StateVec<N> a, const StateVec<N>& b) {
  for (int i = 0; i < N; ++i) a[i] += b[i];
  return a;
}

template <int N>
ADC_HD StateVec<N> operator-(StateVec<N> a, const StateVec<N>& b) {
  for (int i = 0; i < N; ++i) a[i] -= b[i];
  return a;
}

template <int N>
ADC_HD StateVec<N> operator*(Real s, StateVec<N> a) {
  for (int i = 0; i < N; ++i) a[i] *= s;
  return a;
}

// Champs auxiliaires derives de la resolution elliptique : le potentiel et
// son gradient au point. C'est le canal unique par lequel le couplage entre
// dans la physique. Il alimente A LA FOIS le flux (transport a derive : la
// vitesse E x B vient de grad phi) et la source (fluide compressible
// auto-gravitant : S = -rho grad phi). Cette dualite est ce qui permet a un seul
// operateur spatial de servir les deux problemes cibles.
//
// Canal aux EXTENSIBLE (cf. aux_comps()/load_aux dans spatial_operator.hpp). Les trois
// premieres composantes sont le contrat de BASE, identique a l'historique :
//   [0] = phi, [1] = grad_x, [2] = grad_y.
// Les suivantes sont des champs auxiliaires SUPPLEMENTAIRES, optionnels, dans un ordre
// canonique fixe ([3] = B_z, ...). Un modele declare combien de composantes il lit via
// un membre statique n_aux (defaut kAuxBaseComps = 3) ; un modele sans n_aux ne lit
// jamais les champs extra et reste strictement bit-identique. Les champs extra valent 0
// par defaut : load_aux ne les ecrase que si le modele les demande.
struct Aux {
  Real phi{};     // potentiel       (composante aux 0)
  Real grad_x{};  // d phi / d x     (composante aux 1)
  Real grad_y{};  // d phi / d y     (composante aux 2)
  Real B_z{};     // champ B hors-plan, fourni par le systeme (composante aux 3, optionnel)
};

// Largeur du canal aux du contrat de base (phi, grad phi). Un modele lisant des champs
// supplementaires declare un n_aux plus grand ; cf. aux_comps()/load_aux().
inline constexpr int kAuxBaseComps = 3;

}  // namespace adc
