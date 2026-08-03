/// @file
/// @brief Deterministic axis-indexed Berger-Rigoutsos clustering for tiled ND tags.

#pragma once

#include <pops/amr/hierarchy/nd/cluster_provider.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::amr::hierarchy::nd {

template <int Dim>
class BergerRigoutsosProvider final : public ClusterProvider<Dim> {
  static_assert(Dim >= 1 && Dim <= 3,
                "BergerRigoutsosProvider only supports dimensions 1, 2, and 3");

 public:
  static constexpr std::string_view kIdentity = "pops.amr.cluster.berger-rigoutsos.nd.v1";

  std::string_view provider_identity() const noexcept override { return kIdentity; }

  ClusterResult<Dim> cluster(std::span<const TagMask<Dim>> shards,
                             const ClusterOptions<Dim>& options) const override {
    validate_options_(options);
    const std::vector<const TagMask<Dim>*> canonical = authenticate_shards_(shards, options);
    Work work{options.budget};
    std::vector<Box<Dim>> raw;
    const LevelLayoutIdentity<Dim>& source = canonical.front()->level_identity();

    for (std::size_t global_patch = 0; global_patch < source.patches.size(); ++global_patch) {
      for (int axis = 0; axis < Dim; ++axis)
        if (source.patches[global_patch].length(axis) > std::numeric_limits<int>::max())
          throw std::length_error(
              "Berger-Rigoutsos patch axis exceeds deterministic signature indexing");
      const TagMask<Dim>& owner = owner_for_patch_(canonical, source, global_patch);
      cluster_rec_(owner, source.patches[global_patch], options, work, raw);
    }

    std::vector<Box<Dim>> boxes;
    for (const Box<Dim>& box : raw) {
      const std::size_t chopped = chopped_count_(box, options.max_box_size);
      work.require_output(chopped);
      const mesh::BoxArray<Dim> pieces =
          mesh::BoxArray<Dim>::from_domain(box, options.max_box_size);
      boxes.insert(boxes.end(), pieces.boxes().begin(), pieces.boxes().end());
    }
    std::sort(boxes.begin(), boxes.end(), lexicographic_less_);

    ClusterResultIdentity<Dim> identity;
    identity.provider = std::string(kIdentity);
    identity.source_level = source;
    identity.options = options;
    identity.canonical_shards.reserve(canonical.size());
    for (const TagMask<Dim>* shard : canonical)
      identity.canonical_shards.push_back(shard->exact_identity());
    identity.boxes = boxes;
    return ClusterResult<Dim>{mesh::BoxArray<Dim>{std::move(boxes)}, std::move(identity)};
  }

 private:
  struct Work {
    explicit Work(ClusterWorkBudget allowed) : allowed(allowed) {}

    void visit_node() {
      if (nodes == allowed.recursion_nodes)
        throw std::length_error("Berger-Rigoutsos exceeds its recursion-node budget");
      ++nodes;
    }

    void visit_cells(std::size_t count) {
      if (visited_cells > allowed.cell_visits || count > allowed.cell_visits - visited_cells)
        throw std::length_error("Berger-Rigoutsos exceeds its cell-visit budget");
      visited_cells += count;
    }

    void require_output(std::size_t count) {
      if (output_boxes > allowed.output_boxes || count > allowed.output_boxes - output_boxes)
        throw std::length_error("Berger-Rigoutsos exceeds its output-box budget");
      output_boxes += count;
    }

    ClusterWorkBudget allowed{};
    std::size_t nodes = 0;
    std::size_t visited_cells = 0;
    std::size_t output_boxes = 0;
  };

  struct Scan {
    Box<Dim> bounds{};
    std::size_t tagged = 0;
  };

  static bool lexicographic_less_(const Box<Dim>& left, const Box<Dim>& right) {
    for (int axis = 0; axis < Dim; ++axis) {
      if (left.lo[axis] != right.lo[axis])
        return left.lo[axis] < right.lo[axis];
      if (left.hi[axis] != right.hi[axis])
        return left.hi[axis] < right.hi[axis];
    }
    return false;
  }

  static void validate_options_(const ClusterOptions<Dim>& options) {
    if (!std::isfinite(options.min_efficiency) || options.min_efficiency <= 0.0 ||
        options.min_efficiency > 1.0)
      throw std::invalid_argument("Berger-Rigoutsos efficiency must lie in (0, 1]");
    for (int axis = 0; axis < Dim; ++axis) {
      if (options.min_box_size[axis] <= 0 || options.max_box_size[axis] <= 0)
        throw std::invalid_argument("Berger-Rigoutsos box sizes must be strictly positive");
      if (options.min_box_size[axis] > options.max_box_size[axis])
        throw std::invalid_argument("Berger-Rigoutsos minimum box size cannot exceed its maximum");
    }
    if (options.budget.shards == 0 || options.budget.recursion_nodes == 0 ||
        options.budget.cell_visits == 0 || options.budget.output_boxes == 0)
      throw std::invalid_argument("Berger-Rigoutsos work budgets must be strictly positive");
    if (options.budget.cell_visits >
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
      throw std::invalid_argument("Berger-Rigoutsos cell budget exceeds exact signed counters");
  }

  static std::vector<const TagMask<Dim>*> authenticate_shards_(std::span<const TagMask<Dim>> shards,
                                                               const ClusterOptions<Dim>& options) {
    if (shards.empty())
      throw std::invalid_argument("Berger-Rigoutsos requires at least one tag shard");
    if (shards.size() > options.budget.shards)
      throw std::length_error("Berger-Rigoutsos exceeds its tag-shard budget");

    const LevelLayoutIdentity<Dim>& source = shards.front().level_identity();
    if (source.patches.empty() || source.rank_space.empty())
      throw std::invalid_argument("Berger-Rigoutsos source identity is incomplete");
    std::vector<const TagMask<Dim>*> canonical;
    canonical.reserve(shards.size());
    for (const TagMask<Dim>& shard : shards) {
      if (shard.level_identity() != source)
        throw std::invalid_argument("Berger-Rigoutsos tag shards disagree on exact level identity");
      canonical.push_back(&shard);
    }
    std::sort(canonical.begin(), canonical.end(), [&](const auto* left, const auto* right) {
      return source.rank_space.linear_rank(left->local_rank()) <
             source.rank_space.linear_rank(right->local_rank());
    });
    for (std::size_t index = 1; index < canonical.size(); ++index)
      if (canonical[index - 1]->local_rank() == canonical[index]->local_rank())
        throw std::invalid_argument("Berger-Rigoutsos received duplicate rank tag shards");

    if (source.distribution_mode == mesh::DistributionMode::replicated) {
      if (canonical.size() != 1)
        throw std::invalid_argument(
            "Berger-Rigoutsos requires exactly one shard for a replicated tag layout");
    } else {
      if (canonical.size() != source.rank_space.size())
        throw std::invalid_argument(
            "Berger-Rigoutsos partitioned tags require one shard for every process coordinate");
      for (std::size_t rank = 0; rank < canonical.size(); ++rank)
        if (canonical[rank]->local_rank() != source.rank_space.coordinate(rank))
          throw std::invalid_argument(
              "Berger-Rigoutsos partitioned tag shards do not cover the process space");
    }

    std::vector<unsigned char> seen(source.patches.size(), 0);
    for (const TagMask<Dim>* shard : canonical) {
      for (const auto& patch : shard->patches()) {
        if (patch.global_patch >= source.patches.size() ||
            patch.box != source.patches[patch.global_patch])
          throw std::invalid_argument("Berger-Rigoutsos tag shard patch identity is invalid");
        const bool expected = source.distribution_mode == mesh::DistributionMode::replicated ||
                              source.owners[patch.global_patch] == shard->local_rank();
        if (!expected || seen[patch.global_patch] != 0)
          throw std::invalid_argument("Berger-Rigoutsos tag shard ownership is invalid");
        seen[patch.global_patch] = 1;
      }
    }
    if (std::find(seen.begin(), seen.end(), 0) != seen.end())
      throw std::invalid_argument("Berger-Rigoutsos tag shards omit an owned patch");
    return canonical;
  }

  static const TagMask<Dim>& owner_for_patch_(const std::vector<const TagMask<Dim>*>& shards,
                                              const LevelLayoutIdentity<Dim>& source,
                                              std::size_t global_patch) {
    if (source.distribution_mode == mesh::DistributionMode::replicated)
      return *shards.front();
    const std::size_t rank = source.rank_space.linear_rank(source.owners.at(global_patch));
    return *shards.at(rank);
  }

  static Scan scan_(const TagMask<Dim>& mask, const Box<Dim>& region, Work& work) {
    work.visit_cells(static_cast<std::size_t>(region.numPts()));
    Scan scan;
    bool found = false;
    mask.for_each_cell_in(region, [&](const Index<Dim>& index, bool tagged) {
      if (!tagged)
        return;
      ++scan.tagged;
      if (!found) {
        scan.bounds = Box<Dim>{index, index};
        found = true;
        return;
      }
      for (int axis = 0; axis < Dim; ++axis) {
        scan.bounds.lo[axis] = std::min(scan.bounds.lo[axis], index[axis]);
        scan.bounds.hi[axis] = std::max(scan.bounds.hi[axis], index[axis]);
      }
    });
    return scan;
  }

  static std::array<std::vector<std::int64_t>, Dim> signatures_(const TagMask<Dim>& mask,
                                                                const Box<Dim>& region,
                                                                Work& work) {
    work.visit_cells(static_cast<std::size_t>(region.numPts()));
    std::array<std::vector<std::int64_t>, Dim> signatures;
    for (int axis = 0; axis < Dim; ++axis)
      signatures[axis].assign(static_cast<std::size_t>(region.length(axis)), 0);
    mask.for_each_cell_in(region, [&](const Index<Dim>& index, bool tagged) {
      if (!tagged)
        return;
      for (int axis = 0; axis < Dim; ++axis)
        ++signatures[axis][static_cast<std::size_t>(index[axis] - region.lo[axis])];
    });
    return signatures;
  }

  static int best_hole_(const std::vector<std::int64_t>& signature, int minimum) {
    const int length = static_cast<int>(signature.size());
    int best = -1;
    int best_distance = std::numeric_limits<int>::max();
    const int center = length / 2;
    for (int cut = minimum; cut <= length - minimum; ++cut) {
      if (signature[static_cast<std::size_t>(cut)] != 0)
        continue;
      const int distance = std::abs(cut - center);
      if (distance < best_distance || (distance == best_distance && cut < best)) {
        best = cut;
        best_distance = distance;
      }
    }
    return best;
  }

  static std::pair<int, long double> best_inflection_(const std::vector<std::int64_t>& signature,
                                                      int minimum) {
    const int length = static_cast<int>(signature.size());
    if (length < 3)
      return {-1, 0.0L};
    std::vector<long double> laplacian(static_cast<std::size_t>(length), 0.0L);
    for (int index = 1; index < length - 1; ++index)
      laplacian[static_cast<std::size_t>(index)] =
          static_cast<long double>(signature[static_cast<std::size_t>(index + 1)]) -
          2.0L * signature[static_cast<std::size_t>(index)] +
          signature[static_cast<std::size_t>(index - 1)];
    int best = -1;
    long double score = 0.0L;
    const int lower = std::max(minimum, 2);
    const int upper = std::min(length - minimum, length - 2);
    for (int cut = lower; cut <= upper; ++cut) {
      const long double candidate = std::abs(laplacian[static_cast<std::size_t>(cut)] -
                                             laplacian[static_cast<std::size_t>(cut - 1)]);
      if (candidate > score) {
        best = cut;
        score = candidate;
      }
    }
    return {best, score};
  }

  static void cluster_rec_(const TagMask<Dim>& mask, const Box<Dim>& candidate,
                           const ClusterOptions<Dim>& options, Work& work,
                           std::vector<Box<Dim>>& output) {
    work.visit_node();
    const Scan scan = scan_(mask, candidate, work);
    if (scan.tagged == 0)
      return;
    const Box<Dim>& region = scan.bounds;
    const long double efficiency =
        static_cast<long double>(scan.tagged) / static_cast<long double>(region.numPts());

    std::array<bool, Dim> splittable{};
    bool any_split = false;
    for (int axis = 0; axis < Dim; ++axis) {
      splittable[axis] = region.length(axis) >= 2LL * options.min_box_size[axis];
      any_split = any_split || splittable[axis];
    }
    if (efficiency >= options.min_efficiency || !any_split) {
      if (output.size() == options.budget.output_boxes)
        throw std::length_error("Berger-Rigoutsos exceeds its raw output-box budget");
      output.push_back(region);
      return;
    }

    const auto signatures = signatures_(mask, region, work);
    int axis = -1;
    int cut = -1;
    for (int candidate_axis = 0; candidate_axis < Dim; ++candidate_axis) {
      if (!splittable[candidate_axis])
        continue;
      const int candidate_cut =
          best_hole_(signatures[candidate_axis], options.min_box_size[candidate_axis]);
      if (candidate_cut < 0)
        continue;
      if (axis < 0 || region.length(candidate_axis) > region.length(axis) ||
          (region.length(candidate_axis) == region.length(axis) && candidate_axis < axis)) {
        axis = candidate_axis;
        cut = candidate_cut;
      }
    }

    if (axis < 0) {
      long double best_score = 0.0L;
      for (int candidate_axis = 0; candidate_axis < Dim; ++candidate_axis) {
        if (!splittable[candidate_axis])
          continue;
        const auto [candidate_cut, score] =
            best_inflection_(signatures[candidate_axis], options.min_box_size[candidate_axis]);
        if (candidate_cut < 0)
          continue;
        if (axis < 0 || score > best_score ||
            (score == best_score && region.length(candidate_axis) > region.length(axis)) ||
            (score == best_score && region.length(candidate_axis) == region.length(axis) &&
             candidate_axis < axis)) {
          axis = candidate_axis;
          cut = candidate_cut;
          best_score = score;
        }
      }
    }

    if (axis < 0) {
      for (int candidate_axis = 0; candidate_axis < Dim; ++candidate_axis)
        if (splittable[candidate_axis] &&
            (axis < 0 || region.length(candidate_axis) > region.length(axis)))
          axis = candidate_axis;
      cut = static_cast<int>(region.length(axis) / 2);
    }
    if (axis < 0 || cut <= 0 || cut >= region.length(axis))
      throw std::logic_error("Berger-Rigoutsos failed to produce a strict deterministic split");

    Box<Dim> left = region;
    Box<Dim> right = region;
    left.hi[axis] = region.lo[axis] + cut - 1;
    right.lo[axis] = region.lo[axis] + cut;
    cluster_rec_(mask, left, options, work, output);
    cluster_rec_(mask, right, options, work, output);
  }

  static std::size_t chopped_count_(const Box<Dim>& box, const std::array<int, Dim>& max_box_size) {
    std::size_t result = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      const std::uint64_t length = static_cast<std::uint64_t>(box.length(axis));
      const std::uint64_t limit = static_cast<std::uint64_t>(max_box_size[axis]);
      const std::uint64_t segments = 1 + (length - 1) / limit;
      if (segments > std::numeric_limits<std::size_t>::max() / result)
        throw std::length_error("Berger-Rigoutsos chopped box count exceeds size_t");
      result *= static_cast<std::size_t>(segments);
    }
    return result;
  }
};

}  // namespace pops::amr::hierarchy::nd
