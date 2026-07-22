/// @file
/// @brief Portable tagging primitives shared by the canonical AMR regrid transaction.
///
/// Layer: `include/pops/amr` (AMR geometric primitives).
/// Role: tags a level and provides the low-level bounded tag dilation primitive. Layout consensus,
/// clustering, proper nesting, load balancing and field publication are a single transaction in
/// coupling/amr/amr_regrid_coupler.hpp; there is intentionally no second regrid pipeline here.
/// Contract: the low-level tagging criterion is a trivially-copyable, device-callable predicate on
/// (ConstArray4, i, j); we stay agnostic of the physics. For a gradient criterion, the caller fills
/// the ghosts beforehand. Runtime-authored tagging uses PreparedTaggingExecutionPlan instead.
///
/// Invariants:
/// - conservative regrid = common hierarchy, co-located cells, regrid by union of tags;
/// - the fine level lives in the refined index space of the coarse level (refine(ref_ratio));
/// - without any tag, the fine level (and the finer ones) is removed.

#pragma once

#include <pops/amr/tagging/tag_box.hpp>
#include <pops/core/foundation/allocator.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace pops {

namespace detail {

template <class Crit>
struct PortableTagCellsKernel {
  ConstArray4 values;
  Crit criterion;
  char* mask;
  int nx;
  int lo_x;
  int lo_y;

  POPS_HD void operator()(int i, int j) const {
    if (criterion(values, i, j)) {
      const std::int64_t row = static_cast<std::int64_t>(j) - lo_y;
      const std::int64_t column = static_cast<std::int64_t>(i) - lo_x;
      mask[row * nx + column] = char{1};
    }
  }
};

}  // namespace detail

/// Marks the valid cells where the predicate is true, on a TagBox covering the domain.
/// @tparam Crit trivially-copyable device predicate (ConstArray4, i, j) -> bool, evaluated on the
/// valid cells of each fab. Host-only type erasure such as std::function is deliberately rejected.
/// @param mf source field (local: only iterates over the rank's local fabs).
/// @param domain domain covered by the returned TagBox (level index space).
/// @return TagBox over domain, marked where crit is true.
template <class Crit>
TagBox tag_cells(const MultiFab& mf, const Box2D& domain, Crit crit) {
  static_assert(std::is_trivially_copyable_v<Crit>,
                "tag_cells requires a trivially-copyable device predicate; use the prepared "
                "tagging program for runtime-authored criteria");
  if (domain.empty() || mf.box_array().size() == 0)
    throw std::invalid_argument("tag_cells requires a non-empty domain and field layout");
  for (int current = 0; current < mf.box_array().size(); ++current) {
    const Box2D& box = mf.box_array()[current];
    if (box.empty() || !domain.contains(box))
      throw std::invalid_argument("tag_cells field box lies outside the tag domain");
    for (int previous = 0; previous < current; ++previous)
      if (!box.intersect(mf.box_array()[previous]).empty())
        throw std::invalid_argument("tag_cells requires a non-overlapping field layout");
  }
  TagBox tb(domain);
  std::vector<char, fab_allocator<char>> device_mask(tb.t.size(), char{0});
  for (int li = 0; li < mf.local_size(); ++li) {
    const Fab2D& f = mf.fab(li);
    const Box2D v = f.box();
    for_each_cell(v, detail::PortableTagCellsKernel<Crit>{
                         f.const_array(), crit, device_mask.data(), domain.nx(), domain.lo[0],
                         domain.lo[1]});
  }
  device_fence();
  std::copy(device_mask.begin(), device_mask.end(), tb.t.begin());
  return tb;
}

/// Grows the tags by n cells (square neighborhood), staying within the domain.
/// @param n dilation radius (buffer); used for nesting and to anticipate the motion of structures.
/// @param domain bounds the neighborhood: no tag is placed outside the domain.
/// @return new TagBox over in.box, marked over the union of the square neighborhoods of the tagged cells.
inline TagBox grow_tags(const TagBox& in, int n, const Box2D& domain) {
  if (n < 0)
    throw std::invalid_argument("grow_tags radius must be non-negative");
  if (in.box != domain)
    throw std::invalid_argument("grow_tags requires the tag box to match the bounded domain");
  TagBox out(in.box);
  const Box2D& b = in.box;
  for (int j = b.lo[1]; j <= b.hi[1]; ++j)
    for (int i = b.lo[0]; i <= b.hi[0]; ++i)
      if (in(i, j))
        for (int dj = -n; dj <= n; ++dj)
          for (int di = -n; di <= n; ++di) {
            const int ii = i + di, jj = j + dj;
            if (b.contains(ii, jj) && domain.contains(ii, jj))
              out(ii, jj) = 1;
          }
  return out;
}

}  // namespace pops
