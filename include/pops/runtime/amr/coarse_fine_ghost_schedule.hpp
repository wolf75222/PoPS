/// @file
/// @brief Prepared exact-ranked sparse AMR coarse/fine ghost schedule.

#pragma once

#include <pops/amr/hierarchy/level_layout.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/topology/boundary_topology.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::runtime::amr {

struct CoarseFineGhostScheduleBudget {
  std::size_t fine_patches = 0;
  std::size_t destination_regions = 0;
  std::size_t parent_child_patch_pairs = 0;
  std::size_t canonical_jobs = 0;
  std::size_t peer_plans = 0;
  std::size_t local_elements = 0;
  std::size_t send_elements = 0;
  std::size_t receive_elements = 0;
};

template <int Dim>
struct CoarseFineGhostRegion {
  Box<Dim> destination{};
  Index<Dim> periodic_source_from_destination{};

  bool operator==(const CoarseFineGhostRegion&) const = default;
};

template <int Dim>
struct CoarseFineGhostPatchPlan {
  std::size_t fine_patch = 0;
  Box<Dim> coarse_staging_region{};
  std::vector<CoarseFineGhostRegion<Dim>> fine_destination_regions{};

  bool operator==(const CoarseFineGhostPatchPlan&) const = default;
};

template <int Dim>
struct CoarseFineGhostJob {
  std::size_t coarse_patch = 0;
  std::size_t fine_patch = 0;
  Index<Dim> destination_rank{};
  Box<Dim> coarse_region{};
  Index<Dim> source_from_destination{};
  std::size_t offset = 0;
  std::size_t elements = 0;

  bool operator==(const CoarseFineGhostJob&) const = default;
};

template <int Dim>
struct CoarseFineGhostPeerPlan {
  Index<Dim> peer{};
  std::vector<CoarseFineGhostJob<Dim>> jobs{};
  std::size_t elements = 0;

  bool operator==(const CoarseFineGhostPeerPlan&) const = default;
};

namespace coarse_fine_ghost_detail {

inline void checked_add(std::size_t& destination, std::size_t value, std::size_t budget,
                        const char* operation) {
  if (destination > budget || value > budget - destination)
    throw std::length_error(operation);
  destination += value;
}

inline std::size_t checked_product(std::size_t left, std::size_t right, const char* operation) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(operation);
  return left * right;
}

inline int checked_negate_native(int value, const char* operation) {
  if (value == std::numeric_limits<int>::min())
    throw std::overflow_error(operation);
  return -value;
}

template <int Dim>
Box<Dim> grow(const Box<Dim>& box, const Extent<Dim>& ghosts) {
  Box<Dim> result = box;
  for (int axis = 0; axis < Dim; ++axis) {
    if (ghosts[axis] < 0 || ghosts[axis] > std::numeric_limits<int>::max())
      throw std::invalid_argument(
          "coarse/fine ghost schedule requires non-negative native ghost extents");
    result = result.grow(axis, static_cast<int>(ghosts[axis]));
  }
  return result;
}

/// Disjoint rectangular decomposition of subject minus cut.
template <int Dim>
std::vector<Box<Dim>> subtract(const Box<Dim>& subject, const Box<Dim>& cut) {
  const Box<Dim> overlap = subject.intersect(cut);
  if (overlap.empty())
    return subject.empty() ? std::vector<Box<Dim>>{} : std::vector<Box<Dim>>{subject};
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
Box<Dim> required_parent_stencil(const Box<Dim>& fine_region,
                                 const ::pops::amr::RefinementRatio<Dim>& ratio,
                                 const Box<Dim>& coarse_domain, const Box<Dim>& fine_domain,
                                 const Index<Dim>& periodic_source_from_destination) {
  Box<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    const auto parent = [&](int fine_coordinate) {
      const std::int64_t relative = static_cast<std::int64_t>(fine_coordinate) +
                                    periodic_source_from_destination[axis] - fine_domain.lo[axis];
      const std::int64_t quotient = relative / ratio[axis];
      const std::int64_t remainder = relative % ratio[axis];
      return static_cast<std::int64_t>(coarse_domain.lo[axis]) +
             (remainder < 0 ? quotient - 1 : quotient);
    };
    std::int64_t lower = parent(fine_region.lo[axis]);
    std::int64_t upper = parent(fine_region.hi[axis]);
    if (ratio[axis] > 1) {
      --lower;
      ++upper;
    }
    if (lower < std::numeric_limits<int>::min() || upper > std::numeric_limits<int>::max())
      throw std::overflow_error("coarse/fine interpolation stencil exceeds native coordinates");
    result.lo[axis] = static_cast<int>(lower);
    result.hi[axis] = static_cast<int>(upper);
  }
  return result;
}

inline std::int64_t floor_div(std::int64_t numerator, std::int64_t denominator) {
  const std::int64_t quotient = numerator / denominator;
  const std::int64_t remainder = numerator % denominator;
  return remainder < 0 ? quotient - 1 : quotient;
}

template <int Dim>
std::vector<CoarseFineGhostRegion<Dim>> split_periodic(const Box<Dim>& region,
                                                       const Box<Dim>& domain,
                                                       const BoundaryTopology<Dim>& topology,
                                                       std::size_t budget) {
  std::vector<CoarseFineGhostRegion<Dim>> pieces{{region, Index<Dim>{}}};
  for (int axis = 0; axis < Dim; ++axis) {
    if (!topology.is_periodic(Face<Dim>{axis, BoundarySide::lower}))
      continue;
    const std::int64_t length = domain.length(axis);
    std::vector<CoarseFineGhostRegion<Dim>> next;
    for (const auto& piece : pieces) {
      const std::int64_t first = floor_div(
          static_cast<std::int64_t>(piece.destination.lo[axis]) - domain.lo[axis], length);
      const std::int64_t last = floor_div(
          static_cast<std::int64_t>(piece.destination.hi[axis]) - domain.lo[axis], length);
      for (std::int64_t image = first; image <= last; ++image) {
        if (image > 0 && length > std::numeric_limits<std::int64_t>::max() / image)
          throw std::overflow_error("coarse/fine periodic image shift exceeds int64_t");
        if (image < 0 && length > std::numeric_limits<std::int64_t>::max() / -image)
          throw std::overflow_error("coarse/fine periodic image shift exceeds int64_t");
        const std::int64_t shift = -image * length;
        const std::int64_t image_lower = static_cast<std::int64_t>(domain.lo[axis]) - shift;
        const std::int64_t image_upper = static_cast<std::int64_t>(domain.hi[axis]) - shift;
        Box<Dim> slice = piece.destination;
        const std::int64_t lower = std::max<std::int64_t>(slice.lo[axis], image_lower);
        const std::int64_t upper = std::min<std::int64_t>(slice.hi[axis], image_upper);
        if (upper < lower)
          continue;
        if (lower < std::numeric_limits<int>::min() || upper > std::numeric_limits<int>::max() ||
            shift < std::numeric_limits<int>::min() || shift > std::numeric_limits<int>::max())
          throw std::overflow_error("coarse/fine periodic image exceeds native coordinates");
        slice.lo[axis] = static_cast<int>(lower);
        slice.hi[axis] = static_cast<int>(upper);
        auto mapped = piece;
        mapped.destination = slice;
        mapped.periodic_source_from_destination[axis] = static_cast<int>(shift);
        if (next.size() >= budget || next.size() >= next.max_size())
          throw std::length_error("coarse/fine periodic region budget exceeded");
        next.push_back(mapped);
      }
    }
    pieces = std::move(next);
  }
  return pieces;
}

template <int Dim>
Box<Dim> bounding_union(const Box<Dim>& left, const Box<Dim>& right) {
  if (left.empty())
    return right;
  if (right.empty())
    return left;
  Box<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    result.lo[axis] = std::min(left.lo[axis], right.lo[axis]);
    result.hi[axis] = std::max(left.hi[axis], right.hi[axis]);
  }
  return result;
}

template <int Dim>
CoarseFineGhostPeerPlan<Dim>& peer_plan(std::vector<CoarseFineGhostPeerPlan<Dim>>& plans,
                                        const Index<Dim>& peer, std::size_t budget) {
  for (auto& plan : plans)
    if (plan.peer == peer)
      return plan;
  if (plans.size() >= budget || plans.size() >= plans.max_size())
    throw std::length_error("coarse/fine ghost schedule peer-plan budget exceeded");
  plans.push_back(CoarseFineGhostPeerPlan<Dim>{peer, {}, 0});
  return plans.back();
}

}  // namespace coarse_fine_ghost_detail

/// Immutable inter-level gather plan for one sparse fine level.
///
/// Parent data is staged into one private coarse-coordinate Fab per local fine patch.  The staging
/// ownership follows the fine patch, not the parent patch, so partitioned parent and child layouts
/// may differ arbitrarily while retaining one exact process-coordinate space.
template <int Dim>
class CoarseFineGhostSchedule {
 public:
  using field_type = MultiFab<Dim>;
  using patch_plan_type = CoarseFineGhostPatchPlan<Dim>;
  using job_type = CoarseFineGhostJob<Dim>;
  using peer_plan_type = CoarseFineGhostPeerPlan<Dim>;

  template <class CoarseMemorySpace, class FineMemorySpace>
  CoarseFineGhostSchedule(const MultiFab<Dim, CoarseMemorySpace>& coarse,
                          const MultiFab<Dim, FineMemorySpace>& fine, const Box<Dim>& coarse_domain,
                          const Box<Dim>& fine_domain, ::pops::amr::RefinementRatio<Dim> ratio,
                          BoundaryTopology<Dim> topology, CoarseFineGhostScheduleBudget budget)
      : coarse_layout_(coarse.layout()),
        coarse_distribution_(coarse.distribution()),
        fine_layout_(fine.layout()),
        fine_distribution_(fine.distribution()),
        local_rank_(fine.local_rank()),
        coarse_domain_(coarse_domain),
        fine_domain_(fine_domain),
        ratio_(ratio),
        topology_(topology),
        ghosts_(fine.ghosts()),
        ncomp_(fine.ncomp()) {
    validate_metadata_(coarse, fine, budget);
    prepare_patch_plans_(budget);
    prepare_jobs_(budget);
  }

  const mesh::BoxArray<Dim>& coarse_layout() const noexcept { return coarse_layout_; }
  const mesh::Distribution<Dim>& coarse_distribution() const noexcept {
    return coarse_distribution_;
  }
  const mesh::BoxArray<Dim>& fine_layout() const noexcept { return fine_layout_; }
  const mesh::Distribution<Dim>& fine_distribution() const noexcept { return fine_distribution_; }
  const Index<Dim>& local_rank() const noexcept { return local_rank_; }
  const Box<Dim>& coarse_domain() const noexcept { return coarse_domain_; }
  const Box<Dim>& fine_domain() const noexcept { return fine_domain_; }
  const ::pops::amr::RefinementRatio<Dim>& ratio() const noexcept { return ratio_; }
  const BoundaryTopology<Dim>& topology() const noexcept { return topology_; }
  const Extent<Dim>& ghosts() const noexcept { return ghosts_; }
  int ncomp() const noexcept { return ncomp_; }
  const std::vector<patch_plan_type>& patch_plans() const noexcept { return patch_plans_; }
  const std::vector<job_type>& canonical_jobs() const noexcept { return canonical_jobs_; }
  const std::vector<job_type>& local_jobs() const noexcept { return local_jobs_; }
  const std::vector<peer_plan_type>& send_plans() const noexcept { return send_plans_; }
  const std::vector<peer_plan_type>& receive_plans() const noexcept { return receive_plans_; }
  std::size_t local_elements() const noexcept { return local_elements_; }
  std::size_t send_elements() const noexcept { return send_elements_; }
  std::size_t receive_elements() const noexcept { return receive_elements_; }
  bool has_remote_jobs() const noexcept { return !send_plans_.empty() || !receive_plans_.empty(); }

  template <class CoarseMemorySpace, class FineMemorySpace>
  void authenticate(const MultiFab<Dim, CoarseMemorySpace>& coarse,
                    const MultiFab<Dim, FineMemorySpace>& fine) const {
    if (coarse.layout() != coarse_layout_ || coarse.distribution() != coarse_distribution_ ||
        fine.layout() != fine_layout_ || fine.distribution() != fine_distribution_ ||
        coarse.local_rank() != local_rank_ || fine.local_rank() != local_rank_ ||
        coarse.ncomp() != ncomp_ || fine.ncomp() != ncomp_ || fine.ghosts() != ghosts_)
      throw std::invalid_argument(
          "coarse/fine ghost schedule does not match its exact level fields");
  }

 private:
  template <class CoarseMemorySpace, class FineMemorySpace>
  void validate_metadata_(const MultiFab<Dim, CoarseMemorySpace>& coarse,
                          const MultiFab<Dim, FineMemorySpace>& fine,
                          const CoarseFineGhostScheduleBudget& budget) const {
    if (coarse_domain_.empty() || fine_domain_.empty() || !ratio_.refines_any_axis())
      throw std::invalid_argument(
          "coarse/fine ghost schedule requires non-empty adjacent refined domains");
    if (fine_domain_ != ::pops::amr::hierarchy::refine_box(coarse_domain_, ratio_))
      throw std::invalid_argument(
          "coarse/fine ghost schedule child domain is not the refined parent domain");
    if (coarse.ncomp() != fine.ncomp() || coarse.local_rank() != fine.local_rank() ||
        coarse.rank_space() != fine.rank_space())
      throw std::invalid_argument(
          "coarse/fine ghost schedule fields have incompatible components or rank spaces");
    if (!coarse.layout().tiles_exactly(
            coarse_domain_, mesh::BoxArrayValidationBudget{coarse.layout().size(),
                                                           budget.parent_child_patch_pairs}) ||
        !fine.layout().is_disjoint_within(
            fine_domain_,
            mesh::BoxArrayValidationBudget{fine.layout().size(), budget.parent_child_patch_pairs}))
      throw std::invalid_argument(
          "coarse/fine ghost schedule requires a complete parent and sparse valid child layout");
    if (fine.layout().size() > budget.fine_patches)
      throw std::length_error("coarse/fine ghost schedule fine-patch budget exceeded");
  }

  void prepare_patch_plans_(const CoarseFineGhostScheduleBudget& budget) {
    patch_plans_.reserve(fine_layout_.size());
    std::size_t region_count = 0;
    for (std::size_t fine_patch = 0; fine_patch < fine_layout_.size(); ++fine_patch) {
      const Box<Dim>& valid = fine_layout_[fine_patch];
      Box<Dim> admissible_growth = coarse_fine_ghost_detail::grow(valid, ghosts_);
      for (int axis = 0; axis < Dim; ++axis) {
        if (topology_.is_physical(Face<Dim>{axis, BoundarySide::lower}))
          admissible_growth.lo[axis] = std::max(admissible_growth.lo[axis], fine_domain_.lo[axis]);
        if (topology_.is_physical(Face<Dim>{axis, BoundarySide::upper}))
          admissible_growth.hi[axis] = std::min(admissible_growth.hi[axis], fine_domain_.hi[axis]);
      }
      std::vector<CoarseFineGhostRegion<Dim>> destinations;
      for (const Box<Dim>& raw : coarse_fine_ghost_detail::subtract(admissible_growth, valid)) {
        auto split = coarse_fine_ghost_detail::split_periodic(raw, fine_domain_, topology_,
                                                              budget.destination_regions);
        if (destinations.size() > budget.destination_regions ||
            split.size() > budget.destination_regions - destinations.size())
          throw std::length_error("coarse/fine ghost destination-region budget exceeded");
        destinations.insert(destinations.end(), split.begin(), split.end());
      }
      coarse_fine_ghost_detail::checked_add(
          region_count, destinations.size(), budget.destination_regions,
          "coarse/fine ghost schedule destination-region budget exceeded");

      Box<Dim> staging{};
      for (const auto& destination : destinations)
        staging = coarse_fine_ghost_detail::bounding_union(
            staging, coarse_fine_ghost_detail::required_parent_stencil(
                         destination.destination, ratio_, coarse_domain_, fine_domain_,
                         destination.periodic_source_from_destination));
      if (!staging.empty())
        for (int axis = 0; axis < Dim; ++axis)
          if (topology_.is_physical(Face<Dim>{axis, BoundarySide::lower}) &&
              (staging.lo[axis] < coarse_domain_.lo[axis] ||
               staging.hi[axis] > coarse_domain_.hi[axis]))
            throw std::invalid_argument(
                "coarse/fine interpolation stencil crosses a physical parent face; "
                "use a boundary-aware transfer provider");
      patch_plans_.push_back(patch_plan_type{fine_patch, staging, std::move(destinations)});
    }
  }

  std::vector<Index<Dim>> destination_ranks_(std::size_t fine_patch) const {
    if (!fine_distribution_.replicated())
      return {fine_distribution_.owner(fine_patch)};
    std::vector<Index<Dim>> result;
    result.reserve(fine_distribution_.rank_space().size());
    for (std::size_t rank = 0; rank < fine_distribution_.rank_space().size(); ++rank)
      result.push_back(fine_distribution_.rank_space().coordinate(rank));
    return result;
  }

  Index<Dim> source_rank_(std::size_t coarse_patch, const Index<Dim>& destination_rank) const {
    return coarse_distribution_.replicated() ? destination_rank
                                             : coarse_distribution_.owner(coarse_patch);
  }

  std::size_t elements_(const Box<Dim>& region) const {
    const std::int64_t cells = region.numPts();
    if (cells <= 0 || static_cast<std::uint64_t>(cells) > std::numeric_limits<std::size_t>::max() /
                                                              static_cast<std::size_t>(ncomp_))
      throw std::overflow_error("coarse/fine ghost payload exceeds size_t");
    return static_cast<std::size_t>(cells) * static_cast<std::size_t>(ncomp_);
  }

  void append_job_(job_type job, const Index<Dim>& source_rank,
                   const CoarseFineGhostScheduleBudget& budget) {
    if (canonical_jobs_.size() >= budget.canonical_jobs ||
        canonical_jobs_.size() >= canonical_jobs_.max_size())
      throw std::length_error("coarse/fine ghost canonical-job budget exceeded");
    canonical_jobs_.push_back(job);
    const bool source_local = source_rank == local_rank_;
    const bool destination_local = job.destination_rank == local_rank_;
    if (source_local && destination_local) {
      job.offset = local_elements_;
      coarse_fine_ghost_detail::checked_add(local_elements_, job.elements, budget.local_elements,
                                            "coarse/fine ghost local-element budget exceeded");
      local_jobs_.push_back(std::move(job));
    } else if (source_local) {
      auto& plan =
          coarse_fine_ghost_detail::peer_plan(send_plans_, job.destination_rank, budget.peer_plans);
      job.offset = plan.elements;
      coarse_fine_ghost_detail::checked_add(plan.elements, job.elements, budget.send_elements,
                                            "coarse/fine ghost peer send budget exceeded");
      coarse_fine_ghost_detail::checked_add(send_elements_, job.elements, budget.send_elements,
                                            "coarse/fine ghost send-element budget exceeded");
      plan.jobs.push_back(std::move(job));
    } else if (destination_local) {
      auto& plan =
          coarse_fine_ghost_detail::peer_plan(receive_plans_, source_rank, budget.peer_plans);
      job.offset = plan.elements;
      coarse_fine_ghost_detail::checked_add(plan.elements, job.elements, budget.receive_elements,
                                            "coarse/fine ghost peer receive budget exceeded");
      coarse_fine_ghost_detail::checked_add(receive_elements_, job.elements,
                                            budget.receive_elements,
                                            "coarse/fine ghost receive-element budget exceeded");
      plan.jobs.push_back(std::move(job));
    }
  }

  void prepare_jobs_(const CoarseFineGhostScheduleBudget& budget) {
    const std::size_t pair_count = coarse_fine_ghost_detail::checked_product(
        patch_plans_.size(), coarse_layout_.size(),
        "coarse/fine ghost parent-child pair count overflow");
    if (pair_count > budget.parent_child_patch_pairs)
      throw std::length_error("coarse/fine ghost parent-child pair budget exceeded");

    for (const patch_plan_type& patch : patch_plans_) {
      if (patch.coarse_staging_region.empty())
        continue;
      mesh::ExactCellCount covered;
      const auto staging_pieces = coarse_fine_ghost_detail::split_periodic(
          patch.coarse_staging_region, coarse_domain_, topology_, budget.destination_regions);
      for (const auto& staging_piece : staging_pieces) {
        const Box<Dim> source_region =
            staging_piece.destination.shift(staging_piece.periodic_source_from_destination);
        for (std::size_t coarse_patch = 0; coarse_patch < coarse_layout_.size(); ++coarse_patch) {
          const Box<Dim> source_overlap = source_region.intersect(coarse_layout_[coarse_patch]);
          if (source_overlap.empty())
            continue;
          Index<Dim> destination_from_source{};
          for (int axis = 0; axis < Dim; ++axis)
            destination_from_source[axis] = coarse_fine_ghost_detail::checked_negate_native(
                staging_piece.periodic_source_from_destination[axis],
                "coarse/fine periodic inverse shift exceeds native coordinates");
          const Box<Dim> destination_region = source_overlap.shift(destination_from_source);
          if (!covered.add(mesh::ExactCellCount::from_box(destination_region)))
            throw std::overflow_error("coarse/fine ghost source coverage exceeds exact count");
          for (const Index<Dim>& destination_rank : destination_ranks_(patch.fine_patch)) {
            job_type job{coarse_patch,
                         patch.fine_patch,
                         destination_rank,
                         destination_region,
                         staging_piece.periodic_source_from_destination,
                         0,
                         elements_(destination_region)};
            append_job_(std::move(job), source_rank_(coarse_patch, destination_rank), budget);
          }
        }
      }
      if (covered != mesh::ExactCellCount::from_box(patch.coarse_staging_region))
        throw std::invalid_argument(
            "coarse/fine ghost parent layout does not cover a required interpolation stencil");
    }
    const auto order = [this](const peer_plan_type& left, const peer_plan_type& right) {
      return fine_distribution_.rank_space().linear_rank(left.peer) <
             fine_distribution_.rank_space().linear_rank(right.peer);
    };
    std::sort(send_plans_.begin(), send_plans_.end(), order);
    std::sort(receive_plans_.begin(), receive_plans_.end(), order);
  }

  mesh::BoxArray<Dim> coarse_layout_{};
  mesh::Distribution<Dim> coarse_distribution_{};
  mesh::BoxArray<Dim> fine_layout_{};
  mesh::Distribution<Dim> fine_distribution_{};
  Index<Dim> local_rank_{};
  Box<Dim> coarse_domain_{};
  Box<Dim> fine_domain_{};
  ::pops::amr::RefinementRatio<Dim> ratio_{};
  BoundaryTopology<Dim> topology_{};
  Extent<Dim> ghosts_{};
  int ncomp_ = 0;
  std::vector<patch_plan_type> patch_plans_{};
  std::vector<job_type> canonical_jobs_{};
  std::vector<job_type> local_jobs_{};
  std::vector<peer_plan_type> send_plans_{};
  std::vector<peer_plan_type> receive_plans_{};
  std::size_t local_elements_ = 0;
  std::size_t send_elements_ = 0;
  std::size_t receive_elements_ = 0;
};

}  // namespace pops::runtime::amr
