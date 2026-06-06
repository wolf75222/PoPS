/// @file
/// @brief DistributionMapping : associe a chaque box (par indice global) le rang MPI proprietaire.
///
/// SEPARE de BoxArray (convention AMReX) : on peut redistribuer sans rebatir le decoupage. Le ctor
/// (nboxes, nranks) donne un round-robin ; les strategies equilibrees (Z-order SFC, knapsack) vivent
/// dans parallel/load_balance.hpp et construisent un DistributionMapping depuis un vecteur de rangs
/// explicite. Metadonnee REPLIQUEE : chaque rang connait l'attribution complete (cle des chemins
/// distribues fill_boundary / parallel_copy qui enumerent les memes jobs sur tous les rangs).

#pragma once

#include <utility>
#include <vector>

// DistributionMapping : associe a chaque box (par indice global) le rang MPI
// qui la possede. Le ctor (nboxes, nranks) donne un round-robin ; les strategies
// equilibrees (Z-order SFC, knapsack) sont dans parallel/load_balance.hpp et
// construisent un DistributionMapping a partir d'un vecteur de rangs explicite.
// Separe de BoxArray (convention AMReX) : on peut redistribuer sans rebatir le
// decoupage.

namespace adc {

/// Rang MPI proprietaire de chaque box, indexe par l'indice GLOBAL de box (parallele a un
/// BoxArray). Metadonnee repliquee sur tous les rangs.
class DistributionMapping {
 public:
  DistributionMapping() = default;

  /// Round-robin : box i -> rang i % nranks (rang 0 si nranks <= 0). Repartition par defaut.
  DistributionMapping(int nboxes, int nranks) {
    rank_.resize(nboxes);
    for (int i = 0; i < nboxes; ++i) rank_[i] = (nranks > 0) ? i % nranks : 0;
  }

  /// Attribution EXPLICITE : rank[i] = rang proprietaire de la box i (move). Pour les strategies
  /// equilibrees externes (load_balance.hpp).
  explicit DistributionMapping(std::vector<int> rank) : rank_(std::move(rank)) {}

  /// Rang proprietaire de la box d'indice global i.
  int operator[](int i) const { return rank_[i]; }
  /// Nombre de boxes couvertes (= taille du BoxArray associe).
  int size() const { return static_cast<int>(rank_.size()); }
  /// Vue sur le vecteur de rangs (egalite element par element = meme attribution).
  const std::vector<int>& ranks() const { return rank_; }

 private:
  std::vector<int> rank_{};
};

}  // namespace adc
