/// @file
/// @brief Ranked parent-footprint and coarse/fine interface contracts.

#pragma once

#include <pops/amr/hierarchy/level_layout.hpp>
#include <pops/amr/reflux/metric_reflux.hpp>
#include <pops/mesh/layout/box_array.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pops::numerics::time::amr {

/// One ratio-aligned fine patch and its exact parent-cell footprint.
template <int Dim>
class PatchRange {
 public:
  PatchRange(Box<Dim> fine, ::pops::amr::RefinementRatio<Dim> ratio)
      : fine_(fine), ratio_(ratio), parent_(::pops::amr::hierarchy::coarsen_box(fine, ratio)) {
    if (fine_.empty() || !ratio_.refines_any_axis() ||
        ::pops::amr::hierarchy::refine_box(parent_, ratio_) != fine_)
      throw std::invalid_argument(
          "AMR patch range requires one non-empty ratio-aligned fine patch");
  }

  const Box<Dim>& fine_box() const noexcept { return fine_; }
  const Box<Dim>& parent_footprint() const noexcept { return parent_; }
  const ::pops::amr::RefinementRatio<Dim>& ratio() const noexcept { return ratio_; }

 private:
  Box<Dim> fine_{};
  ::pops::amr::RefinementRatio<Dim> ratio_{};
  Box<Dim> parent_{};
};

/// Validate a fine layout and return its ordered parent-cell footprints.
template <int Dim>
std::vector<Box<Dim>> validate_ratio_aligned_disjoint_fine_layout(
    const mesh::BoxArray<Dim>& fine_boxes, const Box<Dim>& parent_domain,
    ::pops::amr::RefinementRatio<Dim> ratio, mesh::BoxArrayValidationBudget budget) {
  if (parent_domain.empty() || !ratio.refines_any_axis())
    throw std::invalid_argument("AMR fine-layout validation requires a parent and true refinement");
  const Box<Dim> fine_domain = ::pops::amr::hierarchy::refine_box(parent_domain, ratio);
  if (fine_boxes.empty() || !fine_boxes.is_disjoint_within(fine_domain, budget))
    throw std::invalid_argument(
        "AMR fine layout must contain bounded disjoint patches inside the refined domain");

  std::vector<Box<Dim>> footprints;
  footprints.reserve(fine_boxes.size());
  for (const Box<Dim>& fine : fine_boxes.boxes()) {
    PatchRange<Dim> range(fine, ratio);
    if (!parent_domain.contains(range.parent_footprint()))
      throw std::invalid_argument("AMR fine patch footprint lies outside its parent domain");
    footprints.push_back(range.parent_footprint());
  }
  return footprints;
}

template <int Dim>
struct CoarseFineInterfaceIdentity {
  ::pops::amr::hierarchy::LevelLayoutIdentity<Dim> parent{};
  ::pops::amr::hierarchy::LevelLayoutIdentity<Dim> child{};
  std::vector<Box<Dim>> parent_footprints{};
  mesh::BoxArrayValidationBudget validation_budget{};

  bool operator==(const CoarseFineInterfaceIdentity&) const = default;
};

/// Authenticated adjacent-level interface. Reflux storage and temporal coverage remain owned by
/// the canonical metric ledger, never reconstructed here.
template <int Dim>
class CoarseFineInterface {
 public:
  CoarseFineInterface(const ::pops::amr::hierarchy::LevelLayout<Dim>& parent,
                      const ::pops::amr::hierarchy::LevelLayout<Dim>& child,
                      mesh::BoxArrayValidationBudget budget)
      : identity_{parent.exact_identity(), child.exact_identity(),
                  validate_ratio_aligned_disjoint_fine_layout(child.patches(), parent.domain(),
                                                              child.ratio_from_parent(), budget),
                  budget} {
    if (parent.level() == std::numeric_limits<int>::max() || child.level() != parent.level() + 1 ||
        child.domain() !=
            ::pops::amr::hierarchy::refine_box(parent.domain(), child.ratio_from_parent()) ||
        child.distribution().rank_space() != parent.distribution().rank_space())
      throw std::invalid_argument(
          "AMR coarse/fine interface requires adjacent layouts on one process space");

    for (const Box<Dim>& footprint : identity_.parent_footprints) {
      mesh::ExactCellCount covered;
      for (const Box<Dim>& parent_patch : parent.patches().boxes())
        if (!covered.add(mesh::ExactCellCount::from_box(footprint.intersect(parent_patch))))
          throw std::overflow_error("AMR coarse/fine interface coverage exceeds exact capacity");
      if (covered != mesh::ExactCellCount::from_box(footprint))
        throw std::invalid_argument(
            "AMR coarse/fine interface footprint is not covered by the parent layout");
    }
  }

  const CoarseFineInterfaceIdentity<Dim>& exact_identity() const noexcept { return identity_; }
  const ::pops::amr::RefinementRatio<Dim>& ratio() const noexcept {
    return identity_.child.ratio_from_parent;
  }
  const std::vector<Box<Dim>>& parent_footprints() const noexcept {
    return identity_.parent_footprints;
  }

  ::pops::amr::reflux::FaceRefinementMapping<Dim> face_mapping() const noexcept {
    return {identity_.parent.domain.lo, identity_.child.domain.lo};
  }

 private:
  CoarseFineInterfaceIdentity<Dim> identity_{};
};

}  // namespace pops::numerics::time::amr
