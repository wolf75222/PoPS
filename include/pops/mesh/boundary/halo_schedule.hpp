/// @file
/// @brief Authenticated compile-time-ranked halo schedules.

#pragma once

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/topology/boundary_topology.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {

/// Finite preparation budget.  Halo construction is quadratic in the patch count and also visits
/// every periodic image that can intersect the requested ghost depth.
struct HaloScheduleBudget {
  mesh::BoxArrayValidationBudget layout{};
  std::size_t box_image_pairs = 0;
  std::size_t jobs = 0;
  std::size_t periodic_images = 0;
};

template <int Dim>
struct HaloJob {
  std::size_t source_box = 0;
  std::size_t destination_box = 0;
  Box<Dim> destination_region{};
  Index<Dim> source_from_destination{};

  bool operator==(const HaloJob&) const = default;
};

template <int Dim>
struct HaloPeerPlan {
  Index<Dim> peer{};
  std::vector<HaloJob<Dim>> jobs{};

  bool operator==(const HaloPeerPlan&) const = default;
};

namespace halo_schedule_detail {

inline std::size_t checked_product(std::size_t left, std::size_t right, const char* operation) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(operation);
  return left * right;
}

inline int checked_index(std::int64_t value, const char* operation) {
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    throw std::overflow_error(operation);
  return static_cast<int>(value);
}

inline std::int64_t checked_multiply(std::int64_t left, std::int64_t right, const char* operation) {
  if (left < 0 || right < 0)
    throw std::invalid_argument(operation);
  if (left != 0 && right > std::numeric_limits<std::int64_t>::max() / left)
    throw std::overflow_error(operation);
  return left * right;
}

template <int Dim>
Box<Dim> grow(const Box<Dim>& box, const Extent<Dim>& ghosts) {
  Box<Dim> result = box;
  for (int axis = 0; axis < Dim; ++axis) {
    if (ghosts[axis] < 0)
      throw std::invalid_argument("pops::HaloSchedule ghost extents must be non-negative");
    result = result.grow(
        axis, checked_index(ghosts[axis], "pops::HaloSchedule ghost extent exceeds int"));
  }
  return result;
}

/// Return a disjoint decomposition of subject minus cut.  At most 2*Dim boxes are produced.
template <int Dim>
std::vector<Box<Dim>> subtract(const Box<Dim>& subject, const Box<Dim>& cut) {
  const Box<Dim> overlap = subject.intersect(cut);
  if (overlap.empty())
    return {subject};
  if (overlap == subject)
    return {};

  std::vector<Box<Dim>> result;
  result.reserve(static_cast<std::size_t>(2 * Dim));
  Box<Dim> remainder = subject;
  for (int axis = 0; axis < Dim; ++axis) {
    if (remainder.lo[axis] < overlap.lo[axis]) {
      Box<Dim> lower = remainder;
      lower.hi[axis] = overlap.lo[axis] - 1;
      result.push_back(lower);
      remainder.lo[axis] = overlap.lo[axis];
    }
    if (overlap.hi[axis] < remainder.hi[axis]) {
      Box<Dim> upper = remainder;
      upper.lo[axis] = overlap.hi[axis] + 1;
      result.push_back(upper);
      remainder.hi[axis] = overlap.hi[axis];
    }
  }
  return result;
}

template <int Dim>
std::array<std::size_t, Dim> image_counts(const Box<Dim>& domain, const Extent<Dim>& ghosts,
                                          const BoundaryTopology<Dim>& topology,
                                          std::size_t& total) {
  std::array<std::size_t, Dim> counts{};
  total = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    if (ghosts[axis] < 0)
      throw std::invalid_argument("pops::HaloSchedule ghost extents must be non-negative");
    const Face<Dim> lower{axis, BoundarySide::lower};
    counts[axis] = 1;
    if (topology.is_periodic(lower) && ghosts[axis] != 0) {
      const std::int64_t extent = domain.length(axis);
      if (extent <= 0)
        throw std::invalid_argument("pops::HaloSchedule periodic domain must be non-empty");
      const std::int64_t wraps = 1 + (ghosts[axis] - 1) / extent;
      if (wraps > static_cast<std::int64_t>((std::numeric_limits<std::size_t>::max() - 1) / 2))
        throw std::length_error("pops::HaloSchedule periodic image count exceeds size_t");
      counts[axis] = 1 + 2 * static_cast<std::size_t>(wraps);
    }
    total = checked_product(total, counts[axis],
                            "pops::HaloSchedule periodic image product exceeds size_t");
  }
  return counts;
}

template <int Dim>
Index<Dim> image_shift(std::size_t ordinal, const std::array<std::size_t, Dim>& counts,
                       const Box<Dim>& domain) {
  Index<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t option = ordinal % counts[axis];
    ordinal /= counts[axis];
    if (option == 0)
      continue;
    const std::size_t wrap = 1 + (option - 1) / 2;
    if (wrap > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
      throw std::overflow_error("pops::HaloSchedule periodic wrap exceeds int64_t");
    const std::int64_t magnitude =
        checked_multiply(static_cast<std::int64_t>(wrap), domain.length(axis),
                         "pops::HaloSchedule periodic shift overflows int64_t");
    result[axis] = checked_index(option % 2 == 1 ? magnitude : -magnitude,
                                 "pops::HaloSchedule periodic shift exceeds Index range");
  }
  return result;
}

template <int Dim>
Index<Dim> negate(const Index<Dim>& value) {
  Index<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = checked_index(-static_cast<std::int64_t>(value[axis]),
                                 "pops::HaloSchedule periodic shift negation overflows");
  return result;
}

template <int Dim>
HaloPeerPlan<Dim>& peer_plan(std::vector<HaloPeerPlan<Dim>>& plans, const Index<Dim>& peer) {
  for (HaloPeerPlan<Dim>& plan : plans)
    if (plan.peer == peer)
      return plan;
  plans.push_back(HaloPeerPlan<Dim>{peer, {}});
  return plans.back();
}

}  // namespace halo_schedule_detail

/// A per-rank immutable halo plan over exact ND layout and ownership identities.
template <int Dim>
class HaloSchedule {
 public:
  using job_type = HaloJob<Dim>;
  using peer_plan_type = HaloPeerPlan<Dim>;

  HaloSchedule(const mesh::BoxArray<Dim>& layout, const mesh::Distribution<Dim>& distribution,
               Index<Dim> local_rank, Box<Dim> domain, Extent<Dim> ghosts,
               BoundaryTopology<Dim> topology, HaloScheduleBudget budget)
      : layout_(layout),
        distribution_(distribution),
        local_rank_(local_rank),
        domain_(domain),
        ghosts_(ghosts),
        topology_(topology) {
    if (!distribution_.matches_layout(layout_))
      throw std::invalid_argument("pops::HaloSchedule distribution does not match its layout");
    if (!distribution_.rank_space().contains(local_rank_))
      throw std::out_of_range("pops::HaloSchedule local rank is outside the rank space");
    if (!layout_.tiles_exactly(domain_, budget.layout))
      throw std::invalid_argument("pops::HaloSchedule layout must tile the domain exactly");

    std::size_t image_count = 0;
    const std::array<std::size_t, Dim> image_count_by_axis =
        halo_schedule_detail::image_counts(domain_, ghosts_, topology_, image_count);
    if (image_count > budget.periodic_images)
      throw std::length_error("pops::HaloSchedule periodic image budget exceeded");
    const std::size_t pair_count = halo_schedule_detail::checked_product(
        layout_.size(), layout_.size(), "pops::HaloSchedule patch pair count exceeds size_t");
    const std::size_t work = halo_schedule_detail::checked_product(
        pair_count, image_count, "pops::HaloSchedule box-image work exceeds size_t");
    if (work > budget.box_image_pairs)
      throw std::length_error("pops::HaloSchedule box-image pair budget exceeded");

    std::vector<job_type> jobs;
    for (std::size_t destination = 0; destination < layout_.size(); ++destination) {
      const Box<Dim> grown = halo_schedule_detail::grow(layout_[destination], ghosts_);
      for (std::size_t image = 0; image < image_count; ++image) {
        const Index<Dim> source_from_destination =
            halo_schedule_detail::image_shift<Dim>(image, image_count_by_axis, domain_);
        const Index<Dim> image_from_source = halo_schedule_detail::negate(source_from_destination);
        for (std::size_t source = 0; source < layout_.size(); ++source) {
          const Box<Dim> candidate = grown.intersect(layout_[source].shift(image_from_source));
          if (candidate.empty())
            continue;
          for (const Box<Dim>& region :
               halo_schedule_detail::subtract(candidate, layout_[destination])) {
            if (jobs.size() >= budget.jobs)
              throw std::length_error("pops::HaloSchedule job budget exceeded");
            const Box<Dim> source_region = region.shift(source_from_destination);
            if (!layout_[source].contains(source_region))
              throw std::logic_error("pops::HaloSchedule generated an invalid source region");
            jobs.push_back(job_type{source, destination, region, source_from_destination});
          }
        }
      }
    }

    for (const job_type& job : jobs)
      classify_(job);
  }

  const mesh::BoxArray<Dim>& layout() const noexcept { return layout_; }
  const mesh::Distribution<Dim>& distribution() const noexcept { return distribution_; }
  const Index<Dim>& local_rank() const noexcept { return local_rank_; }
  const Box<Dim>& domain() const noexcept { return domain_; }
  const Extent<Dim>& ghosts() const noexcept { return ghosts_; }
  const BoundaryTopology<Dim>& topology() const noexcept { return topology_; }
  const std::vector<job_type>& local_jobs() const noexcept { return local_; }
  const std::vector<peer_plan_type>& send_plans() const noexcept { return send_; }
  const std::vector<peer_plan_type>& receive_plans() const noexcept { return receive_; }

  bool has_remote_jobs() const noexcept { return !send_.empty() || !receive_.empty(); }

  void require_local_execution() const {
    if (has_remote_jobs())
      throw std::logic_error(
          "pops::HaloSchedule contains remote ND jobs; this build has no production ND transport");
  }

  template <class MemorySpace>
  void authenticate(const MultiFab<Dim, MemorySpace>& fields) const {
    if (fields.layout() != layout_ || fields.distribution() != distribution_ ||
        fields.local_rank() != local_rank_ || fields.ghosts() != ghosts_)
      throw std::invalid_argument("pops::HaloSchedule does not match the destination MultiFab");
  }

 private:
  void classify_(const job_type& job) {
    const bool source_local = distribution_.is_local(job.source_box, local_rank_);
    const bool destination_local = distribution_.is_local(job.destination_box, local_rank_);
    if (source_local && destination_local) {
      local_.push_back(job);
      return;
    }
    if (distribution_.replicated())
      throw std::logic_error("pops::HaloSchedule replicated ownership classification is invalid");
    if (source_local) {
      halo_schedule_detail::peer_plan(send_, distribution_.owner(job.destination_box))
          .jobs.push_back(job);
    } else if (destination_local) {
      halo_schedule_detail::peer_plan(receive_, distribution_.owner(job.source_box))
          .jobs.push_back(job);
    }
  }

  mesh::BoxArray<Dim> layout_{};
  mesh::Distribution<Dim> distribution_{};
  Index<Dim> local_rank_{};
  Box<Dim> domain_{};
  Extent<Dim> ghosts_{};
  BoundaryTopology<Dim> topology_{};
  std::vector<job_type> local_{};
  std::vector<peer_plan_type> send_{};
  std::vector<peer_plan_type> receive_{};
};

template <int Dim, class MemorySpace>
HaloSchedule<Dim> prepare_halo_schedule(const MultiFab<Dim, MemorySpace>& fields,
                                        const Box<Dim>& domain,
                                        const BoundaryTopology<Dim>& topology,
                                        HaloScheduleBudget budget) {
  return HaloSchedule<Dim>{fields.layout(), fields.distribution(), fields.local_rank(),
                           domain,          fields.ghosts(),       topology,
                           budget};
}

}  // namespace pops
