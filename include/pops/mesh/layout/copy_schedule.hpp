/// @file
/// @brief Authenticated compile-time-ranked inter-layout copy schedules.

#pragma once

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {

struct CopyScheduleBudget {
  std::size_t box_pairs = 0;
  std::size_t jobs = 0;
  std::size_t destination_overlap_pairs = 0;
  std::size_t source_overlap_pairs = 0;
};

template <int Dim>
struct CopyJob {
  std::size_t source_box = 0;
  std::size_t destination_box = 0;
  Box<Dim> region{};

  bool operator==(const CopyJob&) const = default;
};

template <int Dim>
struct CopyPeerPlan {
  Index<Dim> peer{};
  std::vector<CopyJob<Dim>> jobs{};

  bool operator==(const CopyPeerPlan&) const = default;
};

namespace copy_schedule_detail {

inline std::size_t checked_pair_count(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error("pops::CopySchedule patch pair count exceeds size_t");
  return left * right;
}

template <int Dim>
void validate_disjoint(const mesh::BoxArray<Dim>& layout, std::size_t pair_budget,
                       const char* name) {
  const std::size_t pair_count =
      layout.size() < 2 ? 0 : checked_pair_count(layout.size(), layout.size() - 1) / 2;
  if (pair_count > pair_budget)
    throw std::length_error(std::string("pops::CopySchedule ") + name +
                            " overlap-pair budget exceeded");
  for (std::size_t left = 0; left < layout.size(); ++left) {
    if (layout[left].empty())
      throw std::invalid_argument(std::string("pops::CopySchedule ") + name +
                                  " layout contains an empty patch");
    for (std::size_t right = 0; right < left; ++right)
      if (!layout[left].intersect(layout[right]).empty())
        throw std::invalid_argument(std::string("pops::CopySchedule ") + name +
                                    " layout contains overlapping patches");
  }
}

template <int Dim>
CopyPeerPlan<Dim>& peer_plan(std::vector<CopyPeerPlan<Dim>>& plans, const Index<Dim>& peer) {
  for (CopyPeerPlan<Dim>& plan : plans)
    if (plan.peer == peer)
      return plan;
  plans.push_back(CopyPeerPlan<Dim>{peer, {}});
  return plans.back();
}

}  // namespace copy_schedule_detail

/// Immutable per-rank overlap plan.  Source and destination layouts may be partitioned differently,
/// but they must cover the same cell set exactly; partial copies are never published as complete.
template <int Dim>
class CopySchedule {
 public:
  using job_type = CopyJob<Dim>;
  using peer_plan_type = CopyPeerPlan<Dim>;

  CopySchedule(const mesh::BoxArray<Dim>& destination_layout,
               const mesh::Distribution<Dim>& destination_distribution,
               const mesh::BoxArray<Dim>& source_layout,
               const mesh::Distribution<Dim>& source_distribution, Index<Dim> local_rank,
               CopyScheduleBudget budget)
      : destination_layout_(destination_layout),
        destination_distribution_(destination_distribution),
        source_layout_(source_layout),
        source_distribution_(source_distribution),
        local_rank_(local_rank) {
    validate_metadata_(budget);

    const std::size_t pair_count =
        copy_schedule_detail::checked_pair_count(destination_layout_.size(), source_layout_.size());
    if (pair_count > budget.box_pairs)
      throw std::length_error("pops::CopySchedule patch-pair budget exceeded");

    mesh::ExactCellCount covered;
    for (std::size_t destination = 0; destination < destination_layout_.size(); ++destination) {
      for (std::size_t source = 0; source < source_layout_.size(); ++source) {
        const Box<Dim> region = destination_layout_[destination].intersect(source_layout_[source]);
        if (region.empty())
          continue;
        if (canonical_.size() >= budget.jobs)
          throw std::length_error("pops::CopySchedule job budget exceeded");
        if (!covered.add(mesh::ExactCellCount::from_box(region)))
          throw std::overflow_error("pops::CopySchedule covered-cell count overflow");
        canonical_.push_back(job_type{source, destination, region});
      }
    }
    if (covered != destination_layout_.exact_cell_count() ||
        covered != source_layout_.exact_cell_count())
      throw std::invalid_argument(
          "pops::CopySchedule source and destination layouts must cover the same cells exactly");

    for (const job_type& job : canonical_)
      classify_(job);
  }

  const mesh::BoxArray<Dim>& destination_layout() const noexcept { return destination_layout_; }
  const mesh::Distribution<Dim>& destination_distribution() const noexcept {
    return destination_distribution_;
  }
  const mesh::BoxArray<Dim>& source_layout() const noexcept { return source_layout_; }
  const mesh::Distribution<Dim>& source_distribution() const noexcept {
    return source_distribution_;
  }
  const Index<Dim>& local_rank() const noexcept { return local_rank_; }
  const std::vector<job_type>& canonical_jobs() const noexcept { return canonical_; }
  const std::vector<job_type>& local_jobs() const noexcept { return local_; }
  const std::vector<peer_plan_type>& send_plans() const noexcept { return send_; }
  const std::vector<peer_plan_type>& receive_plans() const noexcept { return receive_; }

  bool has_remote_jobs() const noexcept { return !send_.empty() || !receive_.empty(); }

  void require_local_execution() const {
    if (has_remote_jobs())
      throw std::logic_error(
          "pops::CopySchedule contains remote jobs; use a prepared mesh::parallel copy transport");
  }

  template <class DestinationMemorySpace, class SourceMemorySpace>
  void authenticate(const MultiFab<Dim, DestinationMemorySpace>& destination,
                    const MultiFab<Dim, SourceMemorySpace>& source) const {
    if (destination.layout() != destination_layout_ ||
        destination.distribution() != destination_distribution_ ||
        source.layout() != source_layout_ || source.distribution() != source_distribution_ ||
        destination.local_rank() != local_rank_ || source.local_rank() != local_rank_)
      throw std::invalid_argument(
          "pops::CopySchedule does not match the source/destination fields");
  }

 private:
  void validate_metadata_(const CopyScheduleBudget& budget) const {
    if (!destination_distribution_.matches_layout(destination_layout_) ||
        !source_distribution_.matches_layout(source_layout_))
      throw std::invalid_argument("pops::CopySchedule distribution/layout identity mismatch");
    if (destination_distribution_.rank_space() != source_distribution_.rank_space())
      throw std::invalid_argument("pops::CopySchedule fields use different rank spaces");
    if (!destination_distribution_.rank_space().contains(local_rank_))
      throw std::out_of_range("pops::CopySchedule local rank is outside the rank space");
    if (destination_distribution_.replicated() != source_distribution_.replicated())
      throw std::invalid_argument(
          "pops::CopySchedule cannot infer ownership for mixed replicated/partitioned fields");
    copy_schedule_detail::validate_disjoint(destination_layout_, budget.destination_overlap_pairs,
                                            "destination");
    copy_schedule_detail::validate_disjoint(source_layout_, budget.source_overlap_pairs, "source");
  }

  void classify_(const job_type& job) {
    const bool source_local = source_distribution_.is_local(job.source_box, local_rank_);
    const bool destination_local =
        destination_distribution_.is_local(job.destination_box, local_rank_);
    if (source_local && destination_local) {
      local_.push_back(job);
      return;
    }
    if (source_distribution_.replicated())
      throw std::logic_error("pops::CopySchedule replicated ownership classification is invalid");
    if (source_local) {
      copy_schedule_detail::peer_plan(send_, destination_distribution_.owner(job.destination_box))
          .jobs.push_back(job);
    } else if (destination_local) {
      copy_schedule_detail::peer_plan(receive_, source_distribution_.owner(job.source_box))
          .jobs.push_back(job);
    }
  }

  mesh::BoxArray<Dim> destination_layout_{};
  mesh::Distribution<Dim> destination_distribution_{};
  mesh::BoxArray<Dim> source_layout_{};
  mesh::Distribution<Dim> source_distribution_{};
  Index<Dim> local_rank_{};
  std::vector<job_type> canonical_{};
  std::vector<job_type> local_{};
  std::vector<peer_plan_type> send_{};
  std::vector<peer_plan_type> receive_{};
};

template <int Dim, class DestinationMemorySpace, class SourceMemorySpace>
CopySchedule<Dim> prepare_copy_schedule(const MultiFab<Dim, DestinationMemorySpace>& destination,
                                        const MultiFab<Dim, SourceMemorySpace>& source,
                                        CopyScheduleBudget budget) {
  if (destination.local_rank() != source.local_rank())
    throw std::invalid_argument("pops::prepare_copy_schedule fields use different local ranks");
  return CopySchedule<Dim>{destination.layout(),  destination.distribution(), source.layout(),
                           source.distribution(), destination.local_rank(),   budget};
}

}  // namespace pops
