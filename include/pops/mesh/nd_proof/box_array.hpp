/// @file
/// @brief Private ordered ND box-layout proof with portable exact cell counts.
///
/// Non-installed proof scaffolding.  It is promoted or deleted in the one-shot ND cutover.

#pragma once

#include <pops/mesh/index/box.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pops::mesh::nd_proof {

/// Explicit finite proof budget for exact layout validation.
struct BoxArrayValidationBudget {
  std::size_t boxes;
  std::size_t overlap_pairs;
};

/// Unsigned four-limb count.  It represents the exact 2^96 cell count of a full 3D signed-index
/// box without compiler-specific wide integers.
class ExactCellCount {
 public:
  constexpr ExactCellCount() = default;

  constexpr bool operator==(const ExactCellCount&) const = default;

  static ExactCellCount from_uint64(std::uint64_t value) {
    ExactCellCount result;
    result.limbs_[0] = static_cast<std::uint32_t>(value);
    result.limbs_[1] = static_cast<std::uint32_t>(value >> 32);
    return result;
  }

  static ExactCellCount power_of_two(unsigned int bit) {
    if (bit >= 128)
      throw std::overflow_error("nd_proof::ExactCellCount bit is outside four limbs");
    ExactCellCount result;
    result.limbs_[bit / 32] = std::uint32_t{1} << (bit % 32);
    return result;
  }

  bool add(const ExactCellCount& other) {
    std::uint64_t carry = 0;
    for (std::size_t limb = 0; limb < limbs_.size(); ++limb) {
      const std::uint64_t sum =
          static_cast<std::uint64_t>(limbs_[limb]) + other.limbs_[limb] + carry;
      limbs_[limb] = static_cast<std::uint32_t>(sum);
      carry = sum >> 32;
    }
    return carry == 0;
  }

  template <int Dim>
  static ExactCellCount from_box(const Box<Dim>& box) {
    static_assert(Dim >= 1 && Dim <= 3, "nd_proof only supports dimensions 1, 2, and 3");
    ExactCellCount result = from_uint64(1);
    if (box.empty())
      return ExactCellCount{};
    for (int axis = 0; axis < Dim; ++axis)
      result.multiply(static_cast<std::uint64_t>(box.length(axis)));
    return result;
  }

 private:
  void multiply(std::uint64_t factor) {
    ExactCellCount result;
    const std::uint32_t low = static_cast<std::uint32_t>(factor);
    const std::uint32_t high = static_cast<std::uint32_t>(factor >> 32);
    for (std::size_t limb = 0; limb < limbs_.size(); ++limb) {
      if (low != 0)
        result.add_product(limb, limbs_[limb], low);
      if (high != 0)
        result.add_product(limb + 1, limbs_[limb], high);
    }
    *this = result;
  }

  void add_product(std::size_t offset, std::uint32_t left, std::uint32_t right) {
    const std::uint64_t product = static_cast<std::uint64_t>(left) * right;
    add_word(offset, static_cast<std::uint32_t>(product));
    add_word(offset + 1, static_cast<std::uint32_t>(product >> 32));
  }

  void add_word(std::size_t offset, std::uint32_t word) {
    while (word != 0) {
      if (offset >= limbs_.size())
        throw std::overflow_error("nd_proof::ExactCellCount exceeds four limbs");
      const std::uint64_t sum = static_cast<std::uint64_t>(limbs_[offset]) + word;
      limbs_[offset] = static_cast<std::uint32_t>(sum);
      word = static_cast<std::uint32_t>(sum >> 32);
      ++offset;
    }
  }

  std::array<std::uint32_t, 4> limbs_{};
};

template <int Dim>
class BoxArray {
  static_assert(Dim >= 1 && Dim <= 3, "nd_proof::BoxArray only supports dimensions 1, 2, and 3");

 public:
  using box_type = Box<Dim>;

  BoxArray() = default;
  explicit BoxArray(std::vector<box_type> boxes) : boxes_(std::move(boxes)) {}

  static BoxArray from_domain(const box_type& domain, const std::array<int, Dim>& max_grid_size) {
    for (int axis = 0; axis < Dim; ++axis)
      if (max_grid_size[axis] <= 0)
        throw std::invalid_argument("nd_proof::BoxArray max grid sizes must be positive");
    if (domain.empty())
      return BoxArray{};

    std::array<std::uint64_t, Dim> segments{};
    std::size_t tile_count = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      const std::uint64_t length = static_cast<std::uint64_t>(domain.length(axis));
      const std::uint64_t limit = static_cast<std::uint64_t>(max_grid_size[axis]);
      segments[axis] = 1 + (length - 1) / limit;
      if (segments[axis] > std::numeric_limits<std::size_t>::max() / tile_count)
        throw std::length_error("nd_proof::BoxArray tile count exceeds size_t");
      tile_count *= static_cast<std::size_t>(segments[axis]);
    }
    if (tile_count > std::vector<box_type>{}.max_size())
      throw std::length_error("nd_proof::BoxArray tile count exceeds vector capacity");

    std::vector<box_type> boxes;
    boxes.reserve(tile_count);
    for (std::size_t ordinal = 0; ordinal < tile_count; ++ordinal) {
      box_type tile{};
      std::size_t quotient = ordinal;
      for (int axis = 0; axis < Dim; ++axis) {
        const std::uint64_t segment = quotient % segments[axis];
        quotient /= segments[axis];  // Axis 0 is the contiguous ordering axis.
        const std::uint64_t length = static_cast<std::uint64_t>(domain.length(axis));
        const std::uint64_t base = length / segments[axis];
        const std::uint64_t remainder = length % segments[axis];
        const std::uint64_t offset = segment * base + (segment < remainder ? segment : remainder);
        const std::uint64_t width = base + (segment < remainder ? 1 : 0);
        const std::int64_t lower =
            static_cast<std::int64_t>(domain.lo[axis]) + static_cast<std::int64_t>(offset);
        tile.lo[axis] = static_cast<int>(lower);
        tile.hi[axis] = static_cast<int>(lower + static_cast<std::int64_t>(width) - 1);
      }
      boxes.push_back(tile);
    }
    return BoxArray{std::move(boxes)};
  }

  std::size_t size() const noexcept { return boxes_.size(); }
  bool empty() const noexcept { return boxes_.empty(); }
  const box_type& operator[](std::size_t index) const { return boxes_.at(index); }
  const std::vector<box_type>& boxes() const noexcept { return boxes_; }

  bool operator==(const BoxArray&) const = default;

  box_type bounding_box() const {
    box_type result{};
    bool found = false;
    for (const box_type& box : boxes_) {
      if (box.empty())
        continue;
      if (!found) {
        result = box;
        found = true;
        continue;
      }
      for (int axis = 0; axis < Dim; ++axis) {
        result.lo[axis] = result.lo[axis] < box.lo[axis] ? result.lo[axis] : box.lo[axis];
        result.hi[axis] = result.hi[axis] < box.hi[axis] ? box.hi[axis] : result.hi[axis];
      }
    }
    return result;
  }

  ExactCellCount exact_cell_count() const {
    ExactCellCount total;
    for (const box_type& box : boxes_)
      if (!total.add(ExactCellCount::from_box(box)))
        throw std::overflow_error("nd_proof::BoxArray cell count exceeds four limbs");
    return total;
  }

  bool tiles_exactly(const box_type& domain, BoxArrayValidationBudget budget) const {
    if (boxes_.size() > budget.boxes)
      throw std::length_error("nd_proof::BoxArray tiling box checks exceed explicit budget");
    if (domain.empty())
      return boxes_.empty();
    std::size_t overlap_pairs = 0;
    if (boxes_.size() > 1) {
      if (boxes_.size() - 1 > std::numeric_limits<std::size_t>::max() / boxes_.size())
        throw std::length_error("nd_proof::BoxArray tiling overlap count overflows size_t");
      overlap_pairs = boxes_.size() * (boxes_.size() - 1) / 2;
    }
    if (overlap_pairs > budget.overlap_pairs)
      throw std::length_error("nd_proof::BoxArray tiling overlap checks exceed explicit budget");

    ExactCellCount total;
    for (std::size_t left = 0; left < boxes_.size(); ++left) {
      const box_type& box = boxes_[left];
      if (box.empty() || !domain.contains(box) || !total.add(ExactCellCount::from_box(box)))
        return false;
      for (std::size_t right = 0; right < left; ++right)
        if (!box.intersect(boxes_[right]).empty())
          return false;
    }
    return total == ExactCellCount::from_box(domain);
  }

 private:
  std::vector<box_type> boxes_;
};

}  // namespace pops::mesh::nd_proof
