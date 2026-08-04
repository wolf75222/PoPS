/// @file
/// @brief Prepared compile-time-ranked AMR regrid transaction.

#pragma once

#include <pops/amr/hierarchy/level_layout.hpp>
#include <pops/amr/tagging/clustering_provider.hpp>
#include <pops/parallel/prepared_load_balance.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::amr::regridding {

struct RegridPreparationBudget {
  mesh::BoxArrayValidationBudget clustered_parent_layout{};
  mesh::BoxArrayValidationBudget fine_layout{};
  parallel::LoadBalancePreparationBudget load_balance{};

  bool operator==(const RegridPreparationBudget&) const = default;
};

template <int Dim>
class PreparedRegrid {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "PreparedRegrid only supports dimensions 1, 2, and 3");

  PreparedRegrid(const PreparedRegrid&) = default;
  PreparedRegrid(PreparedRegrid&&) noexcept = default;
  PreparedRegrid& operator=(const PreparedRegrid&) = delete;
  PreparedRegrid& operator=(PreparedRegrid&&) = delete;

  const hierarchy::LevelLayoutIdentity<Dim>& source_level() const noexcept { return source_level_; }
  const RefinementRatio<Dim>& ratio() const noexcept { return ratio_; }
  const tagging::ClusterResultIdentity<Dim>& clustering() const noexcept { return clustering_; }
  bool removes_fine_level() const noexcept { return !fine_layout_.has_value(); }
  const std::optional<hierarchy::LevelLayout<Dim>>& fine_layout() const noexcept {
    return fine_layout_;
  }
  const std::optional<PreparedLoadBalanceResult<Dim>>& ownership() const noexcept {
    return ownership_;
  }
  std::string_view exact_contract() const noexcept { return exact_contract_; }

 private:
  template <int Rank>
  friend PreparedRegrid<Rank> prepare_regrid(const hierarchy::LevelLayout<Rank>&,
                                             RefinementRatio<Rank>, tagging::ClusterResult<Rank>,
                                             const PreparedLoadBalanceAuthority<Rank>&,
                                             RegridPreparationBudget, const ExecutionLane&);

  PreparedRegrid(hierarchy::LevelLayoutIdentity<Dim> source_level, RefinementRatio<Dim> ratio,
                 tagging::ClusterResultIdentity<Dim> clustering,
                 std::optional<PreparedLoadBalanceResult<Dim>> ownership,
                 std::optional<hierarchy::LevelLayout<Dim>> fine_layout, std::string exact_contract)
      : source_level_(std::move(source_level)),
        ratio_(ratio),
        clustering_(std::move(clustering)),
        ownership_(std::move(ownership)),
        fine_layout_(std::move(fine_layout)),
        exact_contract_(std::move(exact_contract)) {}

  hierarchy::LevelLayoutIdentity<Dim> source_level_;
  RefinementRatio<Dim> ratio_;
  tagging::ClusterResultIdentity<Dim> clustering_;
  std::optional<PreparedLoadBalanceResult<Dim>> ownership_;
  std::optional<hierarchy::LevelLayout<Dim>> fine_layout_;
  std::string exact_contract_;
};

namespace detail {

template <int Dim>
std::string exact_regrid_contract(const hierarchy::LevelLayoutIdentity<Dim>& source,
                                  const RefinementRatio<Dim>& ratio,
                                  const tagging::ClusterResultIdentity<Dim>& clustering,
                                  const std::optional<PreparedLoadBalanceResult<Dim>>& ownership,
                                  const std::optional<hierarchy::LevelLayout<Dim>>& fine_layout,
                                  const RegridPreparationBudget& budget) {
  ExactContractBuilder contract;
  contract.text("pops.prepared-regrid")
      .scalar(std::uint32_t{1})
      .scalar(static_cast<std::uint32_t>(Dim))
      .scalar(source.level);
  for (int axis = 0; axis < Dim; ++axis)
    contract.scalar(source.domain.lo[axis])
        .scalar(source.domain.hi[axis])
        .scalar(source.ratio_from_parent[axis])
        .scalar(ratio[axis])
        .scalar(source.rank_space.origin()[axis])
        .scalar(source.rank_space.extent()[axis]);
  contract.scalar(static_cast<std::uint8_t>(source.distribution_mode))
      .scalar(static_cast<std::uint64_t>(source.validation_budget.boxes))
      .scalar(static_cast<std::uint64_t>(source.validation_budget.overlap_pairs))
      .sequence(source.patches,
                [](ExactContractBuilder& item, const Box<Dim>& patch) {
                  for (int axis = 0; axis < Dim; ++axis)
                    item.scalar(patch.lo[axis]).scalar(patch.hi[axis]);
                })
      .sequence(source.owners,
                [](ExactContractBuilder& item, const Index<Dim>& owner) {
                  for (int axis = 0; axis < Dim; ++axis)
                    item.scalar(owner[axis]);
                })
      .text(clustering.provider)
      .scalar(clustering.options.min_efficiency);
  for (int axis = 0; axis < Dim; ++axis)
    contract.scalar(clustering.options.min_box_size[static_cast<std::size_t>(axis)])
        .scalar(clustering.options.max_box_size[static_cast<std::size_t>(axis)]);
  contract.scalar(static_cast<std::uint64_t>(clustering.options.budget.shards))
      .scalar(static_cast<std::uint64_t>(clustering.options.budget.recursion_nodes))
      .scalar(static_cast<std::uint64_t>(clustering.options.budget.cell_visits))
      .scalar(static_cast<std::uint64_t>(clustering.options.budget.output_boxes))
      .scalar(static_cast<std::uint64_t>(clustering.options.budget.identity_bytes))
      .scalar(static_cast<std::uint64_t>(clustering.canonical_shards.size()));
  for (const tagging::TagShardIdentity<Dim>& shard : clustering.canonical_shards) {
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(shard.local_rank[axis]);
    contract.scalar(static_cast<std::uint8_t>(shard.replicated_alias ? 1 : 0))
        .scalar(static_cast<std::uint64_t>(shard.patches.size()));
    for (const tagging::PatchTagIdentity<Dim>& patch : shard.patches) {
      contract.scalar(static_cast<std::uint64_t>(patch.global_patch));
      for (int axis = 0; axis < Dim; ++axis)
        contract.scalar(patch.box.lo[axis]).scalar(patch.box.hi[axis]);
      contract.sequence(patch.tags);
    }
  }
  contract
      .sequence(clustering.boxes,
                [](ExactContractBuilder& item, const Box<Dim>& patch) {
                  for (int axis = 0; axis < Dim; ++axis)
                    item.scalar(patch.lo[axis]).scalar(patch.hi[axis]);
                })
      .scalar(static_cast<std::uint64_t>(budget.clustered_parent_layout.boxes))
      .scalar(static_cast<std::uint64_t>(budget.clustered_parent_layout.overlap_pairs))
      .scalar(static_cast<std::uint64_t>(budget.fine_layout.boxes))
      .scalar(static_cast<std::uint64_t>(budget.fine_layout.overlap_pairs))
      .scalar(static_cast<std::uint64_t>(budget.load_balance.patches))
      .scalar(static_cast<std::uint64_t>(budget.load_balance.ranks))
      .scalar(budget.load_balance.total_weight)
      .scalar(static_cast<std::uint8_t>(fine_layout.has_value() ? 1 : 0));
  if (fine_layout) {
    contract.sequence(fine_layout->patches().boxes(),
                      [](ExactContractBuilder& item, const Box<Dim>& patch) {
                        for (int axis = 0; axis < Dim; ++axis)
                          item.scalar(patch.lo[axis]).scalar(patch.hi[axis]);
                      });
  }
  contract.bytes(ownership ? ownership->exact_contract() : std::string_view{});
  return std::move(contract).release();
}

}  // namespace detail

/// Prepare clustering-to-ownership cutover for the child of `parent`.
///
/// Cluster boxes are expressed in the parent index space.  Each is refined along every axis before
/// ownership preparation, so child patches contain complete parent cells by construction.  An empty
/// cluster result is an authenticated request to remove the child and all finer levels.
template <int Dim>
PreparedRegrid<Dim> prepare_regrid(const hierarchy::LevelLayout<Dim>& parent,
                                   RefinementRatio<Dim> ratio,
                                   tagging::ClusterResult<Dim> clustered,
                                   const PreparedLoadBalanceAuthority<Dim>& load_balance,
                                   RegridPreparationBudget budget,
                                   const ExecutionLane& lane = ExecutionLane::world()) {
  if (!ratio.refines_any_axis())
    throw std::invalid_argument("prepared regrid requires a non-identity inter-level ratio");
  if (clustered.identity.provider.empty() ||
      clustered.identity.source_level != parent.exact_identity() ||
      clustered.identity.boxes != clustered.boxes.boxes())
    throw std::invalid_argument(
        "prepared regrid clustering result does not authenticate its parent level and boxes");
  if (!clustered.boxes.is_disjoint_within(parent.domain(), budget.clustered_parent_layout))
    throw std::invalid_argument(
        "prepared regrid cluster boxes must be disjoint and inside the parent");

  std::optional<PreparedLoadBalanceResult<Dim>> ownership;
  std::optional<hierarchy::LevelLayout<Dim>> fine_layout;
  if (!clustered.boxes.empty()) {
    std::vector<Box<Dim>> refined;
    refined.reserve(clustered.boxes.size());
    for (const Box<Dim>& parent_patch : clustered.boxes.boxes())
      refined.push_back(hierarchy::refine_box(parent_patch, ratio));
    mesh::BoxArray<Dim> fine_patches(std::move(refined));
    const Box<Dim> fine_domain = hierarchy::refine_box(parent.domain(), ratio);
    if (!fine_patches.is_disjoint_within(fine_domain, budget.fine_layout))
      throw std::invalid_argument("prepared regrid refined patches are not a valid child layout");

    ownership.emplace(load_balance.prepare(fine_patches, parent.distribution().rank_space(),
                                           budget.load_balance, {}, lane));
    fine_layout.emplace(parent.level() + 1, fine_domain, std::move(fine_patches),
                        ownership->plan().distribution(), ratio, budget.fine_layout);
  }

  hierarchy::LevelLayoutIdentity<Dim> source = parent.exact_identity();
  tagging::ClusterResultIdentity<Dim> clustering = std::move(clustered.identity);
  const std::string exact_contract =
      detail::exact_regrid_contract(source, ratio, clustering, ownership, fine_layout, budget);
  return PreparedRegrid<Dim>(std::move(source), ratio, std::move(clustering), std::move(ownership),
                             std::move(fine_layout), exact_contract);
}

}  // namespace pops::amr::regridding
