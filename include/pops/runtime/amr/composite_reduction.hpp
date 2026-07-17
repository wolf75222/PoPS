/// @file
/// @brief Exact selected-level AMR reductions on native Kokkos storage.

#pragma once

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/time/amr/levels/amr_subcycling.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::runtime::amr {

namespace composite_detail {

struct CompositeLevelView {
  const MultiFab* values = nullptr;
  Real dx = Real(0);
  Real dy = Real(0);
};

struct SetCompositeMask {
  Array4 mask;
  Real value;
  POPS_HD void operator()(int i, int j) const { mask(i, j, 0) = value; }
};

enum class CompositeSumKind : int { Sum, AbsSum, SumSq };

struct CompositeSum {
  ConstArray4 values;
  ConstArray4 mask;
  int component;
  CompositeSumKind kind;
  POPS_HD void operator()(int i, int j, Real& result) const {
    if (mask(i, j, 0) == Real(0))
      return;
    const Real value = values(i, j, component);
    if (kind == CompositeSumKind::Sum)
      result += value;
    else if (kind == CompositeSumKind::AbsSum)
      result += value < Real(0) ? -value : value;
    else
      result += value * value;
  }
};

struct CompositeMin {
  ConstArray4 values;
  ConstArray4 mask;
  int component;
  POPS_HD void operator()(int i, int j, Real& result) const {
    if (mask(i, j, 0) == Real(0))
      return;
    const Real value = values(i, j, component);
    if (!(value <= std::numeric_limits<Real>::max() &&
          value >= std::numeric_limits<Real>::lowest())) {
      result = -std::numeric_limits<Real>::infinity();
      return;
    }
    if (value < result)
      result = value;
  }
};

struct CompositeMax {
  ConstArray4 values;
  ConstArray4 mask;
  int component;
  bool absolute;
  POPS_HD void operator()(int i, int j, Real& result) const {
    if (mask(i, j, 0) == Real(0))
      return;
    const Real value = values(i, j, component);
    if (!(value <= std::numeric_limits<Real>::max() &&
          value >= std::numeric_limits<Real>::lowest())) {
      result = std::numeric_limits<Real>::infinity();
      return;
    }
    const Real selected = absolute && value < Real(0) ? -value : value;
    if (selected > result)
      result = selected;
  }
};

inline std::vector<int> selected_levels(int count, const std::vector<int>& requested) {
  if (count < 1)
    throw std::runtime_error("composite_reduce: AMR hierarchy has no active level");
  if (requested.empty()) {
    std::vector<int> all(static_cast<std::size_t>(count));
    for (int level = 0; level < count; ++level)
      all[static_cast<std::size_t>(level)] = level;
    return all;
  }
  int previous = -1;
  for (const int level : requested) {
    if (level < 0 || level >= count)
      throw std::out_of_range("composite_reduce: selected AMR level is out of bounds");
    if (level <= previous)
      throw std::invalid_argument(
          "composite_reduce: selected AMR levels must be strictly increasing and unique");
    previous = level;
  }
  return requested;
}

inline int adjacent_ratio(const CompositeLevelView& coarse, const CompositeLevelView& fine) {
  const double rx = static_cast<double>(coarse.dx) / static_cast<double>(fine.dx);
  const double ry = static_cast<double>(coarse.dy) / static_cast<double>(fine.dy);
  const int ratio = static_cast<int>(std::llround(rx));
  const double tolerance =
      64.0 * std::numeric_limits<double>::epsilon() * std::max({1.0, std::abs(rx), std::abs(ry)});
  if (!std::isfinite(rx) || !std::isfinite(ry) || ratio < 2 || std::abs(rx - ratio) > tolerance ||
      std::abs(ry - ratio) > tolerance)
    throw std::runtime_error(
        "composite_reduce: adjacent AMR levels lack one isotropic integer refinement ratio");
  return ratio;
}

inline int ratio_between(const std::vector<CompositeLevelView>& hierarchy, int coarse, int fine) {
  int ratio = 1;
  for (int level = coarse; level < fine; ++level) {
    const int next = adjacent_ratio(hierarchy[static_cast<std::size_t>(level)],
                                    hierarchy[static_cast<std::size_t>(level + 1)]);
    if (ratio > std::numeric_limits<int>::max() / next)
      throw std::overflow_error("composite_reduce: cumulative AMR refinement ratio overflows int");
    ratio *= next;
  }
  return ratio;
}

inline MultiFab active_mask(const std::vector<CompositeLevelView>& hierarchy, int level,
                            int next_selected) {
  const MultiFab& values = *hierarchy[static_cast<std::size_t>(level)].values;
  MultiFab mask(values.box_array(), values.dmap(), 1, 0);
  for (int local = 0; local < mask.local_size(); ++local)
    for_each_cell(mask.box(local), SetCompositeMask{mask.fab(local).array(), Real(1)});
  if (next_selected < 0)
    return mask;

  const int ratio = ratio_between(hierarchy, level, next_selected);
  const BoxArray& finer =
      hierarchy[static_cast<std::size_t>(next_selected)].values->box_array();
  for (int local = 0; local < mask.local_size(); ++local) {
    const Box2D valid = mask.box(local);
    const Array4 active = mask.fab(local).array();
    for (const Box2D& fine_box : finer.boxes()) {
      const Box2D intersection = valid.intersect(fine_box.coarsen(ratio));
      if (!intersection.empty())
        for_each_cell(intersection, SetCompositeMask{active, Real(0)});
    }
  }
  return mask;
}

inline Real local_sum(const MultiFab& values, const MultiFab& mask, int component,
                      CompositeSumKind kind) {
  Real result = 0;
  for (int local = 0; local < values.local_size(); ++local)
    result += reduce_sum_cell(values.box(local),
                              CompositeSum{values.fab(local).const_array(),
                                           mask.fab(local).const_array(), component, kind});
  return result;
}

inline Real local_min(const MultiFab& values, const MultiFab& mask, int component) {
  Real result = std::numeric_limits<Real>::infinity();
  for (int local = 0; local < values.local_size(); ++local)
    result = std::min(
        result,
        reduce_min_cell(values.box(local), CompositeMin{values.fab(local).const_array(),
                                                        mask.fab(local).const_array(), component}));
  return result;
}

inline Real local_max(const MultiFab& values, const MultiFab& mask, int component, bool absolute) {
  Real result = absolute ? Real(0) : -std::numeric_limits<Real>::infinity();
  for (int local = 0; local < values.local_size(); ++local)
    result = std::max(
        result, reduce_max_cell(values.box(local),
                                CompositeMax{values.fab(local).const_array(),
                                             mask.fab(local).const_array(), component, absolute}));
  return result;
}

}  // namespace composite_detail

inline double composite_reduce_views(
    const std::vector<composite_detail::CompositeLevelView>& hierarchy,
    bool replicated_coarse, const std::string& kind, int component,
    const std::vector<int>& requested_levels = {}) {
  if (hierarchy.empty())
    throw std::runtime_error("composite_reduce: AMR hierarchy has no active level");
  for (const auto& level : hierarchy)
    if (level.values == nullptr || !std::isfinite(static_cast<double>(level.dx)) ||
        !std::isfinite(static_cast<double>(level.dy)) || level.dx <= Real(0) ||
        level.dy <= Real(0))
      throw std::invalid_argument(
          "composite_reduce: every level requires native storage and positive finite metrics");

  const std::vector<int> levels = composite_detail::selected_levels(
      static_cast<int>(hierarchy.size()), requested_levels);
  const bool full = kind.size() > 4 && kind.compare(kind.size() - 4, 4, "_all") == 0;
  const std::string base = full ? kind.substr(0, kind.size() - 4) : kind;
  const bool additive = base == "sum" || base == "abs_sum" || base == "sum_sq";
  if (!additive && base != "min" && base != "max" && base != "abs_max")
    throw std::invalid_argument(
        "composite_reduce: unknown kind '" + kind +
        "' (expected sum/min/max/abs_sum/sum_sq/abs_max and optional _all)");
  const int components = hierarchy.front().values->ncomp();
  if (!full && (component < 0 || component >= components))
    throw std::out_of_range("composite_reduce: selected component is out of bounds");
  for (const auto& level : hierarchy)
    if (level.values->ncomp() != components)
      throw std::runtime_error("composite_reduce: AMR levels disagree on component count");

  double result = additive ? 0.0
                           : (base == "min" ? std::numeric_limits<double>::infinity()
                                            : -std::numeric_limits<double>::infinity());
  if (base == "abs_max")
    result = 0.0;
  for (std::size_t selected = 0; selected < levels.size(); ++selected) {
    const int level = levels[selected];
    const int next = selected + 1 < levels.size() ? levels[selected + 1] : -1;
    const auto& entry = hierarchy[static_cast<std::size_t>(level)];
    const MultiFab& values = *entry.values;
    MultiFab mask = composite_detail::active_mask(hierarchy, level, next);
    const int first_component = full ? 0 : component;
    const int end_component = full ? components : component + 1;

    if (additive) {
      Real local = 0;
      const composite_detail::CompositeSumKind sum_kind =
          base == "sum" ? composite_detail::CompositeSumKind::Sum
                        : (base == "abs_sum" ? composite_detail::CompositeSumKind::AbsSum
                                             : composite_detail::CompositeSumKind::SumSq);
      for (int current = first_component; current < end_component; ++current)
        local += composite_detail::local_sum(values, mask, current, sum_kind);
      const double global = level == 0 && replicated_coarse
                                ? static_cast<double>(local)
                                : all_reduce_sum(static_cast<double>(local));
      result += static_cast<double>(entry.dx) * static_cast<double>(entry.dy) * global;
      continue;
    }

    if (base == "min") {
      Real local = std::numeric_limits<Real>::infinity();
      for (int current = first_component; current < end_component; ++current)
        local = std::min(local, composite_detail::local_min(values, mask, current));
      result = std::min(result, all_reduce_min(static_cast<double>(local)));
    } else {
      const bool absolute = base == "abs_max";
      Real local = absolute ? Real(0) : -std::numeric_limits<Real>::infinity();
      for (int current = first_component; current < end_component; ++current)
        local = std::max(local,
                         composite_detail::local_max(values, mask, current, absolute));
      result = std::max(result, all_reduce_max(static_cast<double>(local)));
    }
  }
  if (!std::isfinite(result))
    throw std::runtime_error("composite_reduce: selected AMR levels contain no active cell");
  return result;
}

/// Reduce exactly @p requested_levels. Empty keeps the low-level C++ all-level convention; the
/// resolved Python DSL always supplies its explicit level tuple. Coarser selected levels are masked
/// only by the next SELECTED finer level, so CoarseOnly, SelectedLevels, and AllLevels have distinct
/// and predictable meanings. Every per-cell fold remains on Kokkos storage; only one scalar per level
/// crosses the host/MPI boundary.
inline double composite_reduce_levels(const std::vector<AmrLevelMP>& hierarchy,
                                      bool replicated_coarse, const std::string& kind,
                                      int component,
                                      const std::vector<int>& requested_levels = {}) {
  std::vector<composite_detail::CompositeLevelView> views;
  views.reserve(hierarchy.size());
  for (const AmrLevelMP& level : hierarchy)
    views.push_back({&level.U, level.dx, level.dy});
  return composite_reduce_views(views, replicated_coarse, kind, component, requested_levels);
}

/// The same native composite fold for a hierarchy whose scalar values do not live in AmrLevelMP::U
/// (for example a qualified elliptic output field).  Layout and metric metadata stay explicit and
/// index-aligned; no field is gathered or copied to Python/host storage.
inline double composite_reduce_fields(
    const std::vector<const MultiFab*>& hierarchy,
    const std::vector<std::pair<Real, Real>>& metrics, bool replicated_coarse,
    const std::string& kind, int component,
    const std::vector<int>& requested_levels = {}) {
  if (hierarchy.size() != metrics.size())
    throw std::invalid_argument(
        "composite_reduce: field hierarchy and metric hierarchy sizes differ");
  std::vector<composite_detail::CompositeLevelView> views;
  views.reserve(hierarchy.size());
  for (std::size_t level = 0; level < hierarchy.size(); ++level)
    views.push_back({hierarchy[level], metrics[level].first, metrics[level].second});
  return composite_reduce_views(views, replicated_coarse, kind, component, requested_levels);
}

}  // namespace pops::runtime::amr
