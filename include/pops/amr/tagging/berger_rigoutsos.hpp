/// @file
/// @brief Deterministic axis-indexed Berger-Rigoutsos clustering for tiled ND tags.

#pragma once

#include <pops/amr/tagging/clustering_provider.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::amr::tagging {

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
    const hierarchy::LevelLayoutIdentity<Dim>& source = canonical.front()->level_identity();

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

    work.require_identity(std::string_view{kIdentity}.size());
    work.require_identity(checked_product_(source.patches.size(), sizeof(Box<Dim>)));
    work.require_identity(checked_product_(source.owners.size(), sizeof(Index<Dim>)));
    work.require_identity(checked_product_(boxes.size(), sizeof(Box<Dim>)));
    work.require_identity(checked_product_(canonical.size(), sizeof(TagShardIdentity<Dim>)));
    for (std::size_t shard_index = 0; shard_index < canonical.size(); ++shard_index) {
      const TagMask<Dim>* shard = canonical[shard_index];
      if (source.distribution_mode == mesh::DistributionMode::replicated && shard_index != 0)
        continue;
      work.require_identity(
          checked_product_(shard->patches().size(), sizeof(PatchTagIdentity<Dim>)));
      for (const auto& patch : shard->patches())
        work.require_identity(patch.tags.size());
    }

    ClusterResultIdentity<Dim> identity;
    identity.provider = std::string(kIdentity);
    identity.source_level = source;
    identity.options = options;
    identity.canonical_shards.reserve(canonical.size());
    for (std::size_t shard_index = 0; shard_index < canonical.size(); ++shard_index) {
      if (source.distribution_mode == mesh::DistributionMode::replicated && shard_index != 0) {
        identity.canonical_shards.push_back(
            TagShardIdentity<Dim>{canonical[shard_index]->local_rank(), {}, true});
      } else {
        identity.canonical_shards.push_back(canonical[shard_index]->shard_identity());
      }
    }
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

    void require_identity(std::size_t count) {
      if (identity_bytes > allowed.identity_bytes ||
          count > allowed.identity_bytes - identity_bytes)
        throw std::length_error("Berger-Rigoutsos exceeds its identity-copy byte budget");
      identity_bytes += count;
    }

    ClusterWorkBudget allowed{};
    std::size_t nodes = 0;
    std::size_t visited_cells = 0;
    std::size_t output_boxes = 0;
    std::size_t identity_bytes = 0;
  };

  struct Scan {
    Box<Dim> bounds{};
    std::size_t tagged = 0;
  };

  struct AxisCut {
    int axis = -1;
    int offset = -1;
    std::int64_t length = 0;
    long double score = 0.0L;
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
        options.budget.cell_visits == 0 || options.budget.output_boxes == 0 ||
        options.budget.identity_bytes == 0)
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

    const hierarchy::LevelLayoutIdentity<Dim>& source = shards.front().level_identity();
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

    if (canonical.size() != source.rank_space.size())
      throw std::invalid_argument(
          "Berger-Rigoutsos requires one tag shard for every process coordinate");
    for (std::size_t rank = 0; rank < canonical.size(); ++rank)
      if (canonical[rank]->local_rank() != source.rank_space.coordinate(rank))
        throw std::invalid_argument("Berger-Rigoutsos tag shards do not cover the process space");

    if (source.distribution_mode == mesh::DistributionMode::replicated) {
      const auto& reference = canonical.front()->patches();
      for (const TagMask<Dim>* shard : canonical) {
        if (shard->patches() != reference)
          throw std::invalid_argument(
              "Berger-Rigoutsos replicated tag shards do not have identical tag bits");
        if (shard->patches().size() != source.patches.size())
          throw std::invalid_argument("Berger-Rigoutsos replicated tag shard omits a patch");
        for (std::size_t patch = 0; patch < source.patches.size(); ++patch)
          if (shard->patches()[patch].global_patch != patch ||
              shard->patches()[patch].box != source.patches[patch])
            throw std::invalid_argument(
                "Berger-Rigoutsos replicated tag shard patch identity is invalid");
      }
      return canonical;
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

  static std::size_t checked_product_(std::size_t left, std::size_t right) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
      throw std::length_error("Berger-Rigoutsos identity byte count exceeds size_t");
    return left * right;
  }

  static const TagMask<Dim>& owner_for_patch_(const std::vector<const TagMask<Dim>*>& shards,
                                              const hierarchy::LevelLayoutIdentity<Dim>& source,
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
    std::vector<AxisCut> cuts;
    for (int candidate_axis = 0; candidate_axis < Dim; ++candidate_axis) {
      if (!splittable[candidate_axis])
        continue;
      const int candidate_cut =
          best_hole_(signatures[candidate_axis], options.min_box_size[candidate_axis]);
      if (candidate_cut < 0)
        continue;
      cuts.push_back(AxisCut{candidate_axis, candidate_cut, region.length(candidate_axis), 0.0L});
    }
    if (!cuts.empty()) {
      const auto longest = std::max_element(
          cuts.begin(), cuts.end(),
          [](const AxisCut& left, const AxisCut& right) { return left.length < right.length; });
      const std::int64_t selected_length = longest->length;
      std::erase_if(cuts, [=](const AxisCut& candidate_cut) {
        return candidate_cut.length != selected_length;
      });
    }

    if (cuts.empty()) {
      for (int candidate_axis = 0; candidate_axis < Dim; ++candidate_axis) {
        if (!splittable[candidate_axis])
          continue;
        const auto [candidate_cut, score] =
            best_inflection_(signatures[candidate_axis], options.min_box_size[candidate_axis]);
        if (candidate_cut < 0)
          continue;
        cuts.push_back(
            AxisCut{candidate_axis, candidate_cut, region.length(candidate_axis), score});
      }
      if (!cuts.empty()) {
        const auto strongest = std::max_element(cuts.begin(), cuts.end(),
                                                [](const AxisCut& left, const AxisCut& right) {
                                                  if (left.score != right.score)
                                                    return left.score < right.score;
                                                  return left.length < right.length;
                                                });
        const long double selected_score = strongest->score;
        const std::int64_t selected_length = strongest->length;
        std::erase_if(cuts, [=](const AxisCut& candidate_cut) {
          return candidate_cut.score != selected_score || candidate_cut.length != selected_length;
        });
      }
    }

    if (cuts.empty()) {
      std::int64_t longest = 0;
      for (int candidate_axis = 0; candidate_axis < Dim; ++candidate_axis)
        if (splittable[candidate_axis])
          longest = std::max(longest, region.length(candidate_axis));
      for (int candidate_axis = 0; candidate_axis < Dim; ++candidate_axis)
        if (splittable[candidate_axis] && region.length(candidate_axis) == longest)
          cuts.push_back(AxisCut{candidate_axis,
                                 static_cast<int>(region.length(candidate_axis) / 2),
                                 region.length(candidate_axis), 0.0L});
    }
    if (cuts.empty())
      throw std::logic_error("Berger-Rigoutsos failed to select a deterministic split");

    std::vector<Box<Dim>> children{region};
    for (const AxisCut& selected : cuts) {
      if (selected.axis < 0 || selected.offset <= 0 ||
          selected.offset >= region.length(selected.axis))
        throw std::logic_error("Berger-Rigoutsos failed to produce a strict split");
      const std::size_t previous_size = children.size();
      for (std::size_t child = 0; child < previous_size; ++child) {
        Box<Dim> right = children[child];
        children[child].hi[selected.axis] = region.lo[selected.axis] + selected.offset - 1;
        right.lo[selected.axis] = region.lo[selected.axis] + selected.offset;
        children.push_back(right);
      }
    }
    for (const Box<Dim>& child : children)
      cluster_rec_(mask, child, options, work, output);
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

}  // namespace pops::amr::tagging
