/// @file
/// @brief Private explicit ND layout-to-rank ownership proof.
///
/// Non-installed proof scaffolding. It represents ownership only; it has no communication or
/// process-global semantics and is promoted or deleted in the one-shot ND cutover.

#pragma once

#include <pops/mesh/nd_proof/box_array.hpp>
#include <pops/mesh/nd_proof/rank_space.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pops::mesh::nd_proof {

enum class DistributionMode { partitioned, replicated };

/// Ordered ownership of a BoxArray over an explicit rank-coordinate space.
///
/// A partitioned layout stores exactly one authenticated coordinate per global box. A replicated
/// layout deliberately stores no owners: every valid rank has every global box.
template <int Dim>
class Distribution {
  static_assert(Dim >= 1 && Dim <= 3,
                "nd_proof::Distribution only supports dimensions 1, 2, and 3");

 public:
  using rank_type = Index<Dim>;

  Distribution() = default;

  Distribution(const BoxArray<Dim>& boxes, RankSpace<Dim> rank_space, DistributionMode mode,
               std::vector<rank_type> owners = {})
      : layout_(boxes),
        rank_space_(std::move(rank_space)),
        mode_(mode),
        owners_(std::move(owners)) {
    validate();
  }

  static Distribution partitioned(const BoxArray<Dim>& boxes, RankSpace<Dim> rank_space,
                                  std::vector<rank_type> owners) {
    return Distribution(boxes, std::move(rank_space), DistributionMode::partitioned,
                        std::move(owners));
  }

  static Distribution replicated(const BoxArray<Dim>& boxes, RankSpace<Dim> rank_space) {
    return Distribution(boxes, std::move(rank_space), DistributionMode::replicated);
  }

  std::size_t box_count() const noexcept { return layout_.size(); }
  bool matches_layout(const BoxArray<Dim>& layout) const noexcept { return layout_ == layout; }
  const RankSpace<Dim>& rank_space() const noexcept { return rank_space_; }
  DistributionMode mode() const noexcept { return mode_; }
  bool replicated() const noexcept { return mode_ == DistributionMode::replicated; }

  const rank_type& owner(std::size_t global_box) const {
    require_global_box(global_box);
    if (mode_ != DistributionMode::partitioned)
      throw std::logic_error("nd_proof::Distribution replicated layouts have no unique owner");
    return owners_[global_box];
  }

  bool is_local(std::size_t global_box, const rank_type& rank) const {
    require_global_box(global_box);
    if (!rank_space_.contains(rank))
      throw std::out_of_range("nd_proof::Distribution rank coordinate is outside the rank space");
    return mode_ == DistributionMode::replicated || owners_[global_box] == rank;
  }

  std::vector<std::size_t> local_box_indices(const rank_type& rank) const {
    if (!rank_space_.contains(rank))
      throw std::out_of_range("nd_proof::Distribution rank coordinate is outside the rank space");
    std::vector<std::size_t> result;
    result.reserve(mode_ == DistributionMode::replicated ? layout_.size() : owners_.size());
    for (std::size_t global_box = 0; global_box < layout_.size(); ++global_box)
      if (mode_ == DistributionMode::replicated || owners_[global_box] == rank)
        result.push_back(global_box);
    return result;
  }

  bool operator==(const Distribution& other) const noexcept {
    return layout_ == other.layout_ && mode_ == other.mode_ && owners_ == other.owners_ &&
           rank_space_.origin() == other.rank_space_.origin() &&
           rank_space_.extent() == other.rank_space_.extent();
  }

 private:
  void validate() const {
    if (mode_ != DistributionMode::partitioned && mode_ != DistributionMode::replicated)
      throw std::invalid_argument("nd_proof::Distribution mode is invalid");
    if (!layout_.empty() && rank_space_.empty())
      throw std::invalid_argument(
          "nd_proof::Distribution non-empty layout requires a non-empty rank space");
    if (mode_ == DistributionMode::replicated) {
      if (!owners_.empty())
        throw std::invalid_argument(
            "nd_proof::Distribution replicated layouts must not store owners");
      return;
    }
    if (owners_.size() != layout_.size())
      throw std::invalid_argument(
          "nd_proof::Distribution partitioned owner count must equal box count");
    for (const rank_type& owner_coordinate : owners_)
      if (!rank_space_.contains(owner_coordinate))
        throw std::out_of_range("nd_proof::Distribution owner is outside the rank space");
  }

  void require_global_box(std::size_t global_box) const {
    if (global_box >= layout_.size())
      throw std::out_of_range("nd_proof::Distribution global box index is outside the layout");
  }

  BoxArray<Dim> layout_{};
  RankSpace<Dim> rank_space_{Index<Dim>{}, Extent<Dim>{}};
  DistributionMode mode_ = DistributionMode::replicated;
  std::vector<rank_type> owners_{};
};

}  // namespace pops::mesh::nd_proof
