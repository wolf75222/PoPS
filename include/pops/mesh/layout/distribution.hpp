/// @file
/// @brief Exact compile-time-ranked patch ownership over an explicit process space.

#pragma once

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/rank_space.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pops::mesh {

enum class DistributionMode { partitioned, replicated };

/// Ordered ownership of a BoxArray. Replicated layouts intentionally have no unique owner vector.
template <int Dim>
class Distribution {
  static_assert(Dim >= 1 && Dim <= 3, "Distribution only supports dimensions 1, 2, and 3");

 public:
  using rank_type = Index<Dim>;

  Distribution() = default;

  Distribution(const BoxArray<Dim>& boxes, RankSpace<Dim> rank_space, DistributionMode mode,
               std::vector<rank_type> owners = {})
      : layout_(boxes),
        rank_space_(std::move(rank_space)),
        mode_(mode),
        owners_(std::move(owners)) {
    validate_();
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
  const BoxArray<Dim>& layout() const noexcept { return layout_; }
  bool matches_layout(const BoxArray<Dim>& layout) const noexcept { return layout_ == layout; }
  const RankSpace<Dim>& rank_space() const noexcept { return rank_space_; }
  DistributionMode mode() const noexcept { return mode_; }
  bool replicated() const noexcept { return mode_ == DistributionMode::replicated; }
  const std::vector<rank_type>& owners() const noexcept { return owners_; }

  const rank_type& owner(std::size_t global_box) const {
    require_global_box_(global_box);
    if (replicated())
      throw std::logic_error("replicated Distribution layouts have no unique owner");
    return owners_[global_box];
  }

  bool is_local(std::size_t global_box, const rank_type& rank) const {
    require_global_box_(global_box);
    if (!rank_space_.contains(rank))
      throw std::out_of_range("Distribution rank coordinate is outside the process space");
    return replicated() || owners_[global_box] == rank;
  }

  std::vector<std::size_t> local_box_indices(const rank_type& rank) const {
    if (!rank_space_.contains(rank))
      throw std::out_of_range("Distribution rank coordinate is outside the process space");
    std::vector<std::size_t> result;
    result.reserve(replicated() ? layout_.size() : owners_.size());
    for (std::size_t global_box = 0; global_box < layout_.size(); ++global_box)
      if (replicated() || owners_[global_box] == rank)
        result.push_back(global_box);
    return result;
  }

  bool operator==(const Distribution&) const = default;

 private:
  void validate_() const {
    if (mode_ != DistributionMode::partitioned && mode_ != DistributionMode::replicated)
      throw std::invalid_argument("Distribution mode is invalid");
    if (!layout_.empty() && rank_space_.empty())
      throw std::invalid_argument("a non-empty Distribution requires a non-empty rank space");
    if (replicated()) {
      if (!owners_.empty())
        throw std::invalid_argument("a replicated Distribution must not store unique owners");
      return;
    }
    if (owners_.size() != layout_.size())
      throw std::invalid_argument("Distribution owner count must equal its patch count");
    for (const rank_type& owner_coordinate : owners_)
      if (!rank_space_.contains(owner_coordinate))
        throw std::out_of_range("Distribution owner is outside the process space");
  }

  void require_global_box_(std::size_t global_box) const {
    if (global_box >= layout_.size())
      throw std::out_of_range("Distribution global patch index is outside the layout");
  }

  BoxArray<Dim> layout_{};
  RankSpace<Dim> rank_space_{};
  DistributionMode mode_ = DistributionMode::replicated;
  std::vector<rank_type> owners_{};
};

}  // namespace pops::mesh
