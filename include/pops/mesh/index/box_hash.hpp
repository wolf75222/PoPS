/// @file
/// @brief Bounded structural spatial hash over compile-time-ranked patch layouts.

#pragma once

#include <pops/mesh/layout/box_array.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace pops::mesh {

template <int Dim>
struct BinCoordinate {
  static_assert(Dim >= 1 && Dim <= 3, "BinCoordinate only supports dimensions 1, 2, and 3");

  std::array<std::int64_t, Dim> axes{};

  constexpr bool operator==(const BinCoordinate&) const = default;
};

template <int Dim>
struct BinCoordinateHash {
  std::size_t operator()(const BinCoordinate<Dim>& coordinate) const noexcept {
    std::size_t hash = 1469598103934665603ULL;
    for (int axis = 0; axis < Dim; ++axis) {
      hash ^= std::hash<std::int64_t>{}(coordinate.axes[axis]);
      hash *= 1099511628211ULL;
    }
    return hash;
  }
};

/// Explicit work limits. No hash construction or query limit is silently inferred.
struct BoxHashBudget {
  std::size_t build_bin_visits;
  std::size_t query_bin_visits;
  std::size_t candidate_references;
};

/// Remaining cumulative query work for a sequence of hash queries.
struct BoxHashQueryBudget {
  std::size_t bin_visits;
  std::size_t candidate_references;
};

template <int Dim>
class BoxHash {
  static_assert(Dim >= 1 && Dim <= 3, "BoxHash only supports dimensions 1, 2, and 3");

 public:
  using box_type = Box<Dim>;

  BoxHash(const BoxArray<Dim>& boxes, const Extent<Dim>& bin_extent, BoxHashBudget budget)
      : bin_extent_(bin_extent), budget_(budget) {
    for (int axis = 0; axis < Dim; ++axis)
      if (bin_extent_[axis] <= 0)
        throw std::invalid_argument("BoxHash bin extents must be strictly positive");

    std::size_t total_visits = 0;
    for (std::size_t index = 0; index < boxes.size(); ++index) {
      if (boxes[index].empty())
        continue;
      const std::size_t visits = checked_bin_visits_(boxes[index]);
      if (total_visits > budget_.build_bin_visits ||
          visits > budget_.build_bin_visits - total_visits || total_visits > bins_.max_size() ||
          visits > bins_.max_size() - total_visits)
        throw std::length_error("BoxHash bin enumeration exceeds its explicit capacity");
      total_visits += visits;
    }

    for (std::size_t index = 0; index < boxes.size(); ++index)
      if (!boxes[index].empty())
        for_each_bin_(boxes[index], [this, index](const BinCoordinate<Dim>& key) {
          bins_[key].push_back(index);
        });
  }

  std::vector<std::size_t> query(const box_type& query_box,
                                 BoxHashQueryBudget* cumulative_budget = nullptr) const {
    std::vector<std::size_t> candidates;
    if (query_box.empty())
      return candidates;
    const std::size_t query_visits = checked_bin_visits_(query_box);
    if (query_visits > budget_.query_bin_visits)
      throw std::length_error("BoxHash query enumeration exceeds its explicit budget");
    if (cumulative_budget != nullptr) {
      if (query_visits > cumulative_budget->bin_visits)
        throw std::length_error("BoxHash cumulative query bins exceed their explicit budget");
      cumulative_budget->bin_visits -= query_visits;
    }
    std::size_t references = 0;
    for_each_bin_(query_box, [this, &candidates, &references,
                              cumulative_budget](const BinCoordinate<Dim>& key) {
      const auto found = bins_.find(key);
      if (found == bins_.end())
        return;
      if (references > budget_.candidate_references ||
          found->second.size() > budget_.candidate_references - references ||
          candidates.size() > candidates.max_size() - found->second.size())
        throw std::length_error("BoxHash candidate references exceed their explicit budget");
      references += found->second.size();
      if (cumulative_budget != nullptr) {
        if (found->second.size() > cumulative_budget->candidate_references)
          throw std::length_error(
              "BoxHash cumulative candidate references exceed their explicit budget");
        cumulative_budget->candidate_references -= found->second.size();
      }
      candidates.insert(candidates.end(), found->second.begin(), found->second.end());
    });
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    return candidates;
  }

 private:
  static std::int64_t floor_div_(int numerator, std::int64_t denominator) {
    const std::int64_t quotient = static_cast<std::int64_t>(numerator) / denominator;
    const std::int64_t remainder = static_cast<std::int64_t>(numerator) % denominator;
    return remainder < 0 ? quotient - 1 : quotient;
  }

  std::size_t checked_bin_visits_(const box_type& box) const {
    std::size_t visits = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      const std::int64_t lower = floor_div_(box.lo[axis], bin_extent_[axis]);
      const std::int64_t upper = floor_div_(box.hi[axis], bin_extent_[axis]);
      const std::uint64_t axis_visits = static_cast<std::uint64_t>(upper - lower) + 1;
      if (axis_visits > std::numeric_limits<std::size_t>::max() / visits)
        throw std::length_error("BoxHash bin enumeration exceeds its explicit capacity");
      visits *= static_cast<std::size_t>(axis_visits);
    }
    return visits;
  }

  template <class Callback>
  void for_each_bin_(const box_type& box, Callback&& callback) const {
    BinCoordinate<Dim> lower{};
    BinCoordinate<Dim> upper{};
    for (int axis = 0; axis < Dim; ++axis) {
      lower.axes[axis] = floor_div_(box.lo[axis], bin_extent_[axis]);
      upper.axes[axis] = floor_div_(box.hi[axis], bin_extent_[axis]);
    }

    BinCoordinate<Dim> current = lower;
    for (;;) {
      callback(current);
      int axis = 0;
      for (; axis < Dim; ++axis) {
        if (current.axes[axis] != upper.axes[axis]) {
          ++current.axes[axis];
          break;
        }
        current.axes[axis] = lower.axes[axis];
      }
      if (axis == Dim)
        return;
    }
  }

  Extent<Dim> bin_extent_{};
  BoxHashBudget budget_{};
  std::unordered_map<BinCoordinate<Dim>, std::vector<std::size_t>, BinCoordinateHash<Dim>> bins_{};
};

template <int Dim>
Extent<Dim> suggest_bin(const BoxArray<Dim>& boxes) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = 1;
  for (const Box<Dim>& box : boxes.boxes()) {
    if (box.empty())
      continue;
    for (int axis = 0; axis < Dim; ++axis)
      result[axis] = result[axis] < box.length(axis) ? box.length(axis) : result[axis];
  }
  return result;
}

}  // namespace pops::mesh
