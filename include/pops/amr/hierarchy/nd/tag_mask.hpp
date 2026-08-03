/// @file
/// @brief Patch-tiled ND AMR tags with explicit local-storage budgets.

#pragma once

#include <pops/amr/hierarchy/nd/level_layout.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pops::amr::hierarchy::nd {

struct TagMaskBudget {
  std::size_t owned_patches = 0;
  std::size_t cells_per_patch = 0;
  std::size_t owned_cells = 0;
  std::size_t bytes = 0;

  bool operator==(const TagMaskBudget&) const = default;
};

template <int Dim>
struct PatchTagIdentity {
  std::size_t global_patch = 0;
  Box<Dim> box{};
  std::vector<std::uint8_t> tags{};

  bool operator==(const PatchTagIdentity&) const = default;
};

template <int Dim>
struct TagMaskIdentity {
  LevelLayoutIdentity<Dim> level{};
  Index<Dim> local_rank{};
  std::vector<PatchTagIdentity<Dim>> patches{};

  bool operator==(const TagMaskIdentity&) const = default;
};

/// Stores one byte per cell only for patches visible to the selected rank coordinate.
template <int Dim>
class TagMask {
  static_assert(Dim >= 1 && Dim <= 3, "TagMask only supports dimensions 1, 2, and 3");

 public:
  struct PatchTags {
    std::size_t global_patch = 0;
    Box<Dim> box{};
    std::vector<std::uint8_t> tags{};

    bool operator==(const PatchTags&) const = default;
  };

  TagMask(const LevelLayout<Dim>& level, Index<Dim> local_rank, TagMaskBudget budget)
      : level_identity_(level.exact_identity()), local_rank_(local_rank) {
    const mesh::Distribution<Dim>& distribution = level.distribution();
    if (!distribution.rank_space().contains(local_rank_))
      throw std::out_of_range("TagMask rank coordinate is outside the level process space");
    const std::vector<std::size_t> local = distribution.local_box_indices(local_rank_);
    if (local.size() > budget.owned_patches)
      throw std::length_error("TagMask exceeds its explicit owned-patch budget");

    std::size_t cells = 0;
    for (const std::size_t global_patch : local) {
      const std::int64_t exact_cells = level.patches()[global_patch].numPts();
      if (exact_cells < 0 ||
          static_cast<std::uint64_t>(exact_cells) >
              static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::length_error("TagMask patch cell count exceeds size_t");
      const std::size_t patch_cells = static_cast<std::size_t>(exact_cells);
      if (patch_cells > budget.cells_per_patch)
        throw std::length_error("TagMask exceeds its explicit per-patch cell budget");
      if (cells > budget.owned_cells || patch_cells > budget.owned_cells - cells)
        throw std::length_error("TagMask exceeds its explicit owned-cell budget");
      cells += patch_cells;
    }
    if (cells > budget.bytes)
      throw std::length_error("TagMask exceeds its explicit byte budget");

    patches_.reserve(local.size());
    for (const std::size_t global_patch : local) {
      const Box<Dim>& box = level.patches()[global_patch];
      patches_.push_back(PatchTags{
          global_patch, box, std::vector<std::uint8_t>(static_cast<std::size_t>(box.numPts()))});
    }
  }

  const LevelLayoutIdentity<Dim>& level_identity() const noexcept { return level_identity_; }
  const Index<Dim>& local_rank() const noexcept { return local_rank_; }
  const std::vector<PatchTags>& patches() const noexcept { return patches_; }
  std::size_t local_patch_count() const noexcept { return patches_.size(); }

  std::size_t local_cell_count() const noexcept {
    std::size_t total = 0;
    for (const PatchTags& patch : patches_)
      total += patch.tags.size();
    return total;
  }

  std::size_t count() const noexcept {
    std::size_t total = 0;
    for (const PatchTags& patch : patches_)
      for (const std::uint8_t value : patch.tags)
        total += value != 0 ? 1u : 0u;
    return total;
  }

  void set(std::size_t global_patch, const Index<Dim>& index, bool tagged = true) {
    PatchTags& patch = require_patch_(global_patch);
    patch.tags.at(linear_index_(patch.box, index)) = tagged ? std::uint8_t{1} : std::uint8_t{0};
  }

  void set(const Index<Dim>& index, bool tagged = true) {
    for (PatchTags& patch : patches_)
      if (patch.box.contains(index)) {
        patch.tags.at(linear_index_(patch.box, index)) = tagged ? std::uint8_t{1} : std::uint8_t{0};
        return;
      }
    throw std::out_of_range("TagMask cell is not in a patch visible to this rank");
  }

  bool tagged(std::size_t global_patch, const Index<Dim>& index) const {
    const PatchTags& patch = require_patch_(global_patch);
    return patch.tags.at(linear_index_(patch.box, index)) != 0;
  }

  template <class Function>
  void for_each_tagged_in(const Box<Dim>& region, Function&& function) const {
    for_each_cell_in(region, [&](const Index<Dim>& index, bool is_tagged) {
      if (is_tagged)
        function(index);
    });
  }

  template <class Function>
  void for_each_cell_in(const Box<Dim>& region, Function&& function) const {
    if (region.empty())
      return;
    for (const PatchTags& patch : patches_) {
      const Box<Dim> overlap = patch.box.intersect(region);
      if (overlap.empty())
        continue;
      for_each_index_(overlap, [&](const Index<Dim>& index) {
        function(index, patch.tags[linear_index_(patch.box, index)] != 0);
      });
    }
  }

  TagMaskIdentity<Dim> exact_identity() const {
    TagMaskIdentity<Dim> identity{level_identity_, local_rank_, {}};
    identity.patches.reserve(patches_.size());
    for (const PatchTags& patch : patches_)
      identity.patches.push_back(PatchTagIdentity<Dim>{patch.global_patch, patch.box, patch.tags});
    return identity;
  }

  bool operator==(const TagMask& other) const { return exact_identity() == other.exact_identity(); }

 private:
  static std::size_t linear_index_(const Box<Dim>& box, const Index<Dim>& index) {
    if (!box.contains(index))
      throw std::out_of_range("TagMask cell is outside the selected patch");
    std::size_t linear = 0;
    std::size_t stride = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      const std::size_t offset =
          static_cast<std::size_t>(static_cast<std::int64_t>(index[axis]) - box.lo[axis]);
      linear += offset * stride;
      stride *= static_cast<std::size_t>(box.length(axis));
    }
    return linear;
  }

  template <class Function>
  static void for_each_index_(const Box<Dim>& box, Function&& function) {
    const std::size_t count = static_cast<std::size_t>(box.numPts());
    for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
      Index<Dim> index{};
      std::size_t quotient = ordinal;
      for (int axis = 0; axis < Dim; ++axis) {
        const std::size_t length = static_cast<std::size_t>(box.length(axis));
        index[axis] = static_cast<int>(static_cast<std::int64_t>(box.lo[axis]) +
                                       static_cast<std::int64_t>(quotient % length));
        quotient /= length;
      }
      function(index);
    }
  }

  PatchTags& require_patch_(std::size_t global_patch) {
    for (PatchTags& patch : patches_)
      if (patch.global_patch == global_patch)
        return patch;
    throw std::out_of_range("TagMask patch is not visible to this rank");
  }

  const PatchTags& require_patch_(std::size_t global_patch) const {
    for (const PatchTags& patch : patches_)
      if (patch.global_patch == global_patch)
        return patch;
    throw std::out_of_range("TagMask patch is not visible to this rank");
  }

  LevelLayoutIdentity<Dim> level_identity_{};
  Index<Dim> local_rank_{};
  std::vector<PatchTags> patches_{};
};

}  // namespace pops::amr::hierarchy::nd
