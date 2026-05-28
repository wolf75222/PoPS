#pragma once

#include <vector>

// DistributionMapping : associe a chaque box (par indice global) le rang MPI
// qui la possede. Round-robin pour l'instant ; l'equilibrage par knapsack sur
// la taille des boxes viendra avec MPI. Separe de BoxArray (convention AMReX) :
// on peut redistribuer sans rebatir le decoupage.

namespace adc {

class DistributionMapping {
 public:
  DistributionMapping() = default;

  DistributionMapping(int nboxes, int nranks) {
    rank_.resize(nboxes);
    for (int i = 0; i < nboxes; ++i) rank_[i] = (nranks > 0) ? i % nranks : 0;
  }

  int operator[](int i) const { return rank_[i]; }
  int size() const { return static_cast<int>(rank_.size()); }

 private:
  std::vector<int> rank_{};
};

}  // namespace adc
