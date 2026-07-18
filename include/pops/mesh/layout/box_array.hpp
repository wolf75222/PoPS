/// @file
/// @brief BoxArray: the set of boxes tiling a level (disjoint, covering).
///
/// Equivalent of AMReX's BoxArray. from_domain splits a domain into tiles of at most
/// max_grid_size per direction, distributed as EVENLY as possible (better balancing than
/// greedy chunks). Carries NO field data and no MPI distribution (cf. MultiFab /
/// DistributionMapping): it is only the geometric decomposition of the level.

#pragma once

#include <pops/mesh/index/box2d.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <set>
#include <utility>
#include <vector>

namespace pops {

/// Ordered list of boxes tiling a level. Construction preserves arbitrary box lists; use
/// tiles_exactly() where the disjoint-and-covering invariant is required. The ORDER is significant
/// (global box index = position in the vector; shared by MultiFab / DistributionMapping). Copyable
/// (vector of Box2D).
class BoxArray {
 public:
  BoxArray() = default;
  /// Build from an already-computed list of boxes (move). The order is kept as is.
  explicit BoxArray(std::vector<Box2D> boxes) : boxes_(std::move(boxes)) {}

  /// Tile the domain into tiles of at most max_grid_size per direction, distributed evenly.
  /// Traversal order is y outer, x inner (deterministic, identical on all ranks).
  static BoxArray from_domain(const Box2D& domain, int max_grid_size) {
    auto sx = split_range(domain.lo[0], domain.hi[0], max_grid_size);
    auto sy = split_range(domain.lo[1], domain.hi[1], max_grid_size);
    std::vector<Box2D> boxes;
    boxes.reserve(sx.size() * sy.size());
    for (auto [ylo, yhi] : sy)
      for (auto [xlo, xhi] : sx)
        boxes.push_back(Box2D{{xlo, ylo}, {xhi, yhi}});
    return BoxArray{std::move(boxes)};
  }

  /// Number of boxes in the tiling.
  int size() const { return static_cast<int>(boxes_.size()); }
  /// Box at global index i (0 <= i < size()); the index is the box identity throughout the code.
  const Box2D& operator[](int i) const { return boxes_[i]; }
  /// View on the underlying vector (element-by-element equality = same boxes AND same order).
  const std::vector<Box2D>& boxes() const { return boxes_; }

  /// Return whether the boxes form an exact tiling of `domain`.
  ///
  /// Every box must be non-empty and contained in `domain`; boxes must be pairwise disjoint and
  /// their exact integer area must equal the domain area.  Overlaps are detected in O(N log N)
  /// with a sweep over half-open rectangles.  At a shared x edge, removals are processed before
  /// insertions so adjacent boxes are accepted.  Coordinate and area arithmetic cannot overflow.
  /// An empty BoxArray exactly tiles an empty domain.
  bool tiles_exactly(const Box2D& domain) const noexcept {
    try {
      return tiles_exactly_impl(domain);
    } catch (...) {
      // Validation is a total predicate.  Allocation failure (or a standard-container size
      // failure) cannot turn an invalid/unverified layout into a valid one.
      return false;
    }
  }

  /// Total number of valid cells (sum of num_cells over all boxes).
  std::int64_t num_cells() const {
    std::int64_t n = 0;
    for (const auto& b : boxes_)
      n += b.num_cells();
    return n;
  }

  /// Smallest box enclosing all boxes (empty box if the tiling is empty).
  Box2D bounding_box() const {
    if (boxes_.empty())
      return Box2D{};
    Box2D b = boxes_[0];
    for (const auto& o : boxes_) {
      b.lo[0] = std::min(b.lo[0], o.lo[0]);
      b.lo[1] = std::min(b.lo[1], o.lo[1]);
      b.hi[0] = std::max(b.hi[0], o.hi[0]);
      b.hi[1] = std::max(b.hi[1], o.hi[1]);
    }
    return b;
  }

 private:
  struct ExactArea {
    std::uint64_t high = 0;  // coefficient of 2^64 (at most one for a Box2D)
    std::uint64_t low = 0;
  };

  static ExactArea exact_area(const Box2D& box) noexcept {
    static_assert(std::numeric_limits<int>::digits <= 31,
                  "BoxArray exact-area arithmetic assumes the 32-bit Box2D index contract");
    const auto width = static_cast<std::uint64_t>(static_cast<std::int64_t>(box.hi[0]) -
                                                  static_cast<std::int64_t>(box.lo[0]) + 1);
    const auto height = static_cast<std::uint64_t>(static_cast<std::int64_t>(box.hi[1]) -
                                                   static_cast<std::int64_t>(box.lo[1]) + 1);
    if (width != 0 && height > std::numeric_limits<std::uint64_t>::max() / width)
      return {1, 0};  // the only possible overflow is exactly 2^32 * 2^32 = 2^64
    return {0, width * height};
  }

  static bool area_less(const ExactArea& lhs, const ExactArea& rhs) noexcept {
    return lhs.high < rhs.high || (lhs.high == rhs.high && lhs.low < rhs.low);
  }

  static void subtract_area(ExactArea& lhs, const ExactArea& rhs) noexcept {
    const bool borrow = lhs.low < rhs.low;
    lhs.low -= rhs.low;
    lhs.high -= rhs.high;
    if (borrow)
      --lhs.high;
  }

  bool tiles_exactly_impl(const Box2D& domain) const {
    if (domain.empty())
      return boxes_.empty();
    if (boxes_.empty())
      return false;

    struct Event {
      std::int64_t x;
      bool insertion;  // false sorts first: remove [x0, x) before inserting [x, x1)
      std::size_t box;
    };
    struct Interval {
      std::int64_t y0;
      std::int64_t y1;
      std::size_t box;
    };
    struct IntervalLess {
      bool operator()(const Interval& lhs, const Interval& rhs) const noexcept {
        if (lhs.y0 != rhs.y0)
          return lhs.y0 < rhs.y0;
        if (lhs.y1 != rhs.y1)
          return lhs.y1 < rhs.y1;
        return lhs.box < rhs.box;
      }
    };

    std::vector<Event> events;
    if (boxes_.size() > events.max_size() / 2)
      return false;
    events.reserve(boxes_.size() * 2);

    ExactArea remaining = exact_area(domain);
    for (std::size_t index = 0; index < boxes_.size(); ++index) {
      const Box2D& box = boxes_[index];
      if (box.empty() || !domain.contains(box))
        return false;

      const ExactArea area = exact_area(box);
      if (area_less(remaining, area))
        return false;
      subtract_area(remaining, area);

      // Inclusive Box2D -> half-open sweep interval.  Widen before adding one so INT_MAX is safe.
      const auto x0 = static_cast<std::int64_t>(box.lo[0]);
      const auto x1 = static_cast<std::int64_t>(box.hi[0]) + 1;
      events.push_back({x0, true, index});
      events.push_back({x1, false, index});
    }
    if (remaining.high != 0 || remaining.low != 0)
      return false;

    std::sort(events.begin(), events.end(), [](const Event& lhs, const Event& rhs) {
      if (lhs.x != rhs.x)
        return lhs.x < rhs.x;
      if (lhs.insertion != rhs.insertion)
        return lhs.insertion < rhs.insertion;
      return lhs.box < rhs.box;
    });

    std::set<Interval, IntervalLess> active;
    for (const Event& event : events) {
      const Box2D& box = boxes_[event.box];
      const Interval interval{static_cast<std::int64_t>(box.lo[1]),
                              static_cast<std::int64_t>(box.hi[1]) + 1, event.box};
      if (!event.insertion) {
        if (active.erase(interval) != 1)
          return false;
        continue;
      }

      const auto next = active.lower_bound(interval);
      if (next != active.end() && next->y0 < interval.y1)
        return false;
      if (next != active.begin()) {
        const auto previous = std::prev(next);
        if (previous->y1 > interval.y0)
          return false;
      }
      active.insert(next, interval);
    }
    return active.empty();
  }

  // Split [lo, hi] into segments of length <= m, distributed evenly:
  // n = ceil(len/m) segments, the first `rem` of them one notch longer.
  static std::vector<std::pair<int, int>> split_range(int lo, int hi, int m) {
    std::vector<std::pair<int, int>> segs;
    int len = hi - lo + 1;
    if (len <= 0 || m <= 0)
      return segs;
    int n = (len + m - 1) / m;
    int base = len / n, rem = len % n;
    int cur = lo;
    for (int k = 0; k < n; ++k) {
      int l = base + (k < rem ? 1 : 0);
      segs.push_back({cur, cur + l - 1});
      cur += l;
    }
    return segs;
  }

  std::vector<Box2D> boxes_{};
};

}  // namespace pops
