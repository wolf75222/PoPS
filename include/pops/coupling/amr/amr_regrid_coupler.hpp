#pragma once

#include <pops/amr/hierarchy/amr_hierarchy.hpp>
#include <pops/amr/tagging/cluster.hpp>  // berger_rigoutsos, ClusterParams
#include <pops/amr/tagging/clustering_provider.hpp>
#include <pops/amr/hierarchy/refinement_ratio.hpp>
#include <pops/amr/regridding/regrid.hpp>  // tag_cells, grow_tags
#include <pops/amr/tagging/tag_box.hpp>    // TagBox
#include <pops/core/foundation/types.hpp>
#include <pops/numerics/time/amr/reflux/amr_reflux_mf.hpp>  // AmrLevelMP, mf_find_box
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>  // device_fence (barrier after async parallel_copy under Cuda)
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/layout/refinement.hpp>  // coarsen_index
#include <pops/parallel/comm.hpp>  // n_ranks (explicit include, no longer an indirect path)
#include <pops/parallel/prepared_load_balance.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

/// @file
/// @brief amr_regrid_finest: Berger-Rigoutsos regrid of the finest level (responsibility b).
///
/// Free function template on the criterion, modeled on the STYLE of amr/regrid.hpp (regrid_level) but
/// NOT merged: different invariants (level fk coords = parent x2, nesting margin clamp,
/// carry-over of the old fine). Body moved as-is from AmrCouplerMP::regrid: same tagging,
/// clustering, clamp, parent interp then fine carry-over, swap + aux realloc. Does not assume single-rank
/// (DistributionMapping selected by the prepared ownership authority). Under a DISTRIBUTED coarse
/// level, the global OR of the tags
/// (all_reduce_or) guarantees IDENTICAL fine patches on all ranks (otherwise incompatible dmaps).

namespace pops {

namespace regrid_detail {

/// Run a rank-local preparation stage, then reach one failure consensus before any later
/// topology-dependent collective. In serial the original exception is preserved for diagnostics;
/// in MPI every rank throws from the same post-consensus point.
template <class Operation>
inline void collective_stage(const char* context, const CommunicatorView& communicator,
                             Operation&& operation) {
  std::exception_ptr local_failure;
  try {
    std::forward<Operation>(operation)();
  } catch (...) {
    local_failure = std::current_exception();
  }
  if (all_reduce_max(local_failure ? 1L : 0L, communicator) == 0)
    return;
  if (communicator.size() == 1 && local_failure)
    std::rethrow_exception(local_failure);
  throw std::runtime_error(std::string(context) + " failed on at least one MPI rank");
}

}  // namespace regrid_detail

// ------------------------------------------------------------------------------------------------
// REFACTOR (docs/AMR_REGRID_UNION_TAGS_DESIGN.md section 6): amr_regrid_finest split into TWO
// responsibilities so the multi-block UNION regrid (AmrRuntime::regrid) can re-grid
// SEVERAL fields on a SINGLE layout imposed from outside (the same one for all blocks):
//   (1) regrid_compute_fine_layout: tags of a parent -> grow -> all_reduce_or (if distributed) ->
//       berger_rigoutsos -> nesting clamp -> (fine BoxArray, DistributionMapping). This is the
//       COMPUTATION of the layout, done ONCE on the union of the tags by the multi-block path.
//   (2) regrid_field_on_layout: takes an IMPOSED (fb, dmap) and rebuilds ONE fine MultiFab (parent
//       interp + fine carry-over), exactly the BODY of the old amr_regrid_finest, but without the
//       layout computation. Called PER BLOCK on the same union layout.
// amr_regrid_finest stays the CHAINING of (1) then (2) on a SINGLE block -> the single-block path
// (AmrCouplerMP::regrid) stays BIT-IDENTICAL (same operations, same order).
// ------------------------------------------------------------------------------------------------

/// Compute the fine layout (BoxArray + DistributionMapping) of a Berger-Rigoutsos regrid from the
/// @p grown tags ALREADY dilated (grow_tags) on the PARENT domain @p pdom. @p pk: parent level;
/// @p margin: nesting; @p coarse_replicated: ownership policy of level 0. Fine level coords =
/// parent x2. Returns an EMPTY BoxArray if there is nothing to refine. MPI-safe: every distributed
/// parent (all levels above zero, plus a de-replicated level zero) globally ORs its tags before
/// clustering so every rank constructs the identical layout. When @p proper_nesting_parents is
/// supplied, each child cluster is built inside one parent patch shrunk by @p margin: a provider
/// cannot return a geometrically non-nested layout and leave the failure to a downstream stencil.
struct RegridPeriodicity {
  bool x = false;
  bool y = false;
};

struct RegridPhysicalGhostSupport {
  int provided_depth = 0;
  bool fills_all_requested_depth = false;
};

inline int regrid_periodic_index(std::int64_t value, int lo, int hi) {
  const std::int64_t width =
      static_cast<std::int64_t>(hi) - static_cast<std::int64_t>(lo) + 1;
  if (width <= 0)
    throw std::overflow_error("regrid periodic extent is not representable");
  std::int64_t shifted = (value - static_cast<std::int64_t>(lo)) % width;
  if (shifted < 0)
    shifted += width;
  const std::int64_t wrapped = static_cast<std::int64_t>(lo) + shifted;
  if (wrapped < std::numeric_limits<int>::min() ||
      wrapped > std::numeric_limits<int>::max())
    throw std::overflow_error("regrid periodic index is not representable");
  return static_cast<int>(wrapped);
}

inline TagBox grow_regrid_tags(const TagBox& input, int radius, const Box2D& domain,
                               RegridPeriodicity periodicity) {
  if (radius < 0)
    throw std::runtime_error("grow_regrid_tags radius must be non-negative");
  TagBox output(input.box);
  // TagBox is the canonical host-side clustering decision buffer, not field storage: native
  // tagging has already reduced any Array4 data into this byte mask before entering the regrid
  // transaction, and ClusteringProvider::cluster consumes the same host representation.  These
  // loops therefore never dereference device field memory and run only at authored regrid cadence.
  for (std::int64_t j64 = input.box.lo[1]; j64 <= input.box.hi[1]; ++j64)
    for (std::int64_t i64 = input.box.lo[0]; i64 <= input.box.hi[0]; ++i64) {
      const int i = static_cast<int>(i64);
      const int j = static_cast<int>(j64);
      if (!input.tagged(i, j))
        continue;
      for (std::int64_t dj = -static_cast<std::int64_t>(radius); dj <= radius; ++dj)
        for (std::int64_t di = -static_cast<std::int64_t>(radius); di <= radius; ++di) {
          std::int64_t ii64 = i64 + di;
          std::int64_t jj64 = j64 + dj;
          if (ii64 < domain.lo[0] || ii64 > domain.hi[0]) {
            if (!periodicity.x)
              continue;
            ii64 = regrid_periodic_index(ii64, domain.lo[0], domain.hi[0]);
          }
          if (jj64 < domain.lo[1] || jj64 > domain.hi[1]) {
            if (!periodicity.y)
              continue;
            jj64 = regrid_periodic_index(jj64, domain.lo[1], domain.hi[1]);
          }
          if (ii64 < std::numeric_limits<int>::min() ||
              ii64 > std::numeric_limits<int>::max() ||
              jj64 < std::numeric_limits<int>::min() ||
              jj64 > std::numeric_limits<int>::max())
            continue;
          const int ii = static_cast<int>(ii64);
          const int jj = static_cast<int>(jj64);
          if (input.box.contains(ii, jj))
            output(ii, jj) = 1;
        }
    }
  return output;
}

inline bool regrid_parent_cell_is_covered(std::int64_t i, std::int64_t j, const Box2D& domain,
                                          const BoxArray* parents, RegridPeriodicity periodicity,
                                          const RegridPhysicalGhostSupport* physical_support = nullptr) {
  if (i < domain.lo[0] || i > domain.hi[0]) {
    if (periodicity.x) {
      i = regrid_periodic_index(i, domain.lo[0], domain.hi[0]);
    } else {
      const std::int64_t distance =
          i < domain.lo[0] ? static_cast<std::int64_t>(domain.lo[0]) - i
                           : i - static_cast<std::int64_t>(domain.hi[0]);
      if (physical_support == nullptr ||
          (!physical_support->fills_all_requested_depth &&
           physical_support->provided_depth < distance))
        return false;
      i = std::clamp(i, static_cast<std::int64_t>(domain.lo[0]),
                     static_cast<std::int64_t>(domain.hi[0]));
    }
  }
  if (j < domain.lo[1] || j > domain.hi[1]) {
    if (periodicity.y) {
      j = regrid_periodic_index(j, domain.lo[1], domain.hi[1]);
    } else {
      const std::int64_t distance =
          j < domain.lo[1] ? static_cast<std::int64_t>(domain.lo[1]) - j
                           : j - static_cast<std::int64_t>(domain.hi[1]);
      if (physical_support == nullptr ||
          (!physical_support->fills_all_requested_depth &&
           physical_support->provided_depth < distance))
        return false;
      j = std::clamp(j, static_cast<std::int64_t>(domain.lo[1]),
                     static_cast<std::int64_t>(domain.hi[1]));
    }
  }
  if (parents == nullptr)
    return true;
  for (const Box2D& parent : parents->boxes())
    if (parent.contains(static_cast<int>(i), static_cast<int>(j)))
      return true;
  return false;
}

inline bool regrid_parent_cell_has_nesting_support(int i, int j, int margin,
                                                   const Box2D& domain,
                                                   const BoxArray* parents,
                                                   RegridPeriodicity periodicity,
                                                   const RegridPhysicalGhostSupport* physical_support = nullptr) {
  for (std::int64_t dj = -static_cast<std::int64_t>(margin); dj <= margin; ++dj)
    for (std::int64_t di = -static_cast<std::int64_t>(margin); di <= margin; ++di)
      if (!regrid_parent_cell_is_covered(static_cast<std::int64_t>(i) + di,
                                         static_cast<std::int64_t>(j) + dj, domain, parents, periodicity,
                                         physical_support))
        return false;
  return true;
}

inline void validate_fine_layout_proper_nesting(const BoxArray& fine, const BoxArray& parents,
                                                const Box2D& parent_domain,
                                                int refinement_ratio, int margin,
                                                RegridPeriodicity periodicity = {},
                                                const RegridPhysicalGhostSupport* physical_support = nullptr) {
  if (refinement_ratio < 2)
    throw std::runtime_error("validate_fine_layout_proper_nesting: refinement_ratio must be >= 2");
  if (margin < 0)
    throw std::runtime_error("validate_fine_layout_proper_nesting: margin must be >= 0");
  for (int child = 0; child < fine.size(); ++child) {
    const Box2D footprint = fine[child].coarsen(refinement_ratio);
    for (int j = footprint.lo[1]; j <= footprint.hi[1]; ++j)
      for (int i = footprint.lo[0]; i <= footprint.hi[0]; ++i)
        if (!regrid_parent_cell_has_nesting_support(i, j, margin, parent_domain, &parents,
                                                    periodicity, physical_support))
          throw std::runtime_error(
              "AMR patch provider returned a non-nested child layout: the child footprint plus "
              "the resolved nesting buffer is not covered by the parent level");
  }
}

inline void validate_fine_layout_proper_nesting(const BoxArray& fine, const BoxArray& parents,
                                                int refinement_ratio, int margin) {
  validate_fine_layout_proper_nesting(fine, parents, parents.bounding_box(), refinement_ratio,
                                      margin);
}

inline std::pair<BoxArray, DistributionMapping> regrid_compute_fine_layout_with_provider(
    TagBox grown, const Box2D& pdom, int pk, int margin, bool coarse_replicated,
    const amr::ClusteringProvider& clustering,
    const PreparedLoadBalanceAuthority& load_balance, const CommunicatorView& communicator,
    int refinement_ratio = kAmrRefRatio,
    const BoxArray* proper_nesting_parents = nullptr,
    RegridPeriodicity periodicity = {},
    const RegridPhysicalGhostSupport* physical_support = nullptr) {
  regrid_detail::collective_stage("regrid layout contract", communicator, [&] {
    if (pk < 0)
      throw std::runtime_error("parent level must be non-negative");
    if (refinement_ratio < 2)
      throw std::runtime_error("refinement_ratio must be >= 2");
    if (margin < 0)
      throw std::runtime_error("margin must be >= 0");
    if (pdom.empty() || grown.box != pdom ||
        grown.t.size() != static_cast<std::size_t>(pdom.num_cells()))
      throw std::runtime_error("tag storage must exactly match the parent domain");
    if (communicator.size() != n_ranks() || communicator.rank() != my_rank())
      throw std::runtime_error(
          "execution communicator must preserve the field-storage rank space");
    if (physical_support != nullptr && physical_support->provided_depth < 0)
      throw std::runtime_error("physical ghost support depth must be non-negative");
  });

  // Every option that could change collective ordering or the published topology is authenticated
  // before the replicated-parent branch. This prevents one rank from skipping the tag reduction
  // while its peers enter it, and makes per-axis periodicity part of the exact transaction.
  const std::array<long, 17> contract{
      static_cast<long>(pk),
      static_cast<long>(margin),
      coarse_replicated ? 1L : 0L,
      static_cast<long>(refinement_ratio),
      static_cast<long>(pdom.lo[0]),
      static_cast<long>(pdom.lo[1]),
      static_cast<long>(pdom.hi[0]),
      static_cast<long>(pdom.hi[1]),
      periodicity.x ? 1L : 0L,
      periodicity.y ? 1L : 0L,
      physical_support != nullptr ? 1L : 0L,
      physical_support != nullptr ? static_cast<long>(physical_support->provided_depth) : 0L,
      physical_support != nullptr && physical_support->fills_all_requested_depth ? 1L : 0L,
      proper_nesting_parents != nullptr ? 1L : 0L,
      proper_nesting_parents != nullptr ? static_cast<long>(proper_nesting_parents->size()) : 0L,
      static_cast<long>(load_balance.semantic_identity().size()),
      static_cast<long>(load_balance.collective_contract().size())};
  std::array<long, 17> contract_min = contract;
  std::array<long, 17> contract_max = contract;
  all_reduce_min_inplace(contract_min.data(), contract_min.size(), communicator);
  all_reduce_max_inplace(contract_max.data(), contract_max.size(), communicator);
  if (contract_min != contract_max)
    throw std::runtime_error(
        "regrid_compute_fine_layout contract differs between MPI ranks");
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{load_balance.semantic_identity(), load_balance.collective_contract()}}, communicator))
    throw std::runtime_error(
        "regrid_compute_fine_layout load-balance authority differs between MPI ranks");
  if (proper_nesting_parents != nullptr) {
    std::vector<long> parent_layout;
    regrid_detail::collective_stage("regrid parent-layout preparation", communicator, [&] {
      parent_layout.reserve(static_cast<std::size_t>(proper_nesting_parents->size()) * 4u);
      for (const Box2D& box : proper_nesting_parents->boxes()) {
        parent_layout.push_back(box.lo[0]);
        parent_layout.push_back(box.lo[1]);
        parent_layout.push_back(box.hi[0]);
        parent_layout.push_back(box.hi[1]);
      }
    });
    std::vector<long> parent_min;
    std::vector<long> parent_max;
    regrid_detail::collective_stage("regrid parent-layout consensus buffers", communicator, [&] {
      parent_min = parent_layout;
      parent_max = parent_layout;
    });
    all_reduce_min_inplace(parent_min.data(), parent_min.size(), communicator);
    all_reduce_max_inplace(parent_max.data(), parent_max.size(), communicator);
    if (parent_min != parent_max)
      throw std::runtime_error(
          "regrid_compute_fine_layout parent layout differs between MPI ranks");
  }
  // Only a replicated level zero is complete locally. Every intermediate parent is distributed.
  const bool parent_replicated = (pk == 0) && coarse_replicated;
  if (!parent_replicated)
    all_reduce_or_inplace(grown.t.data(), grown.t.size(), communicator);
  // Proper nesting is a level-coverage invariant, never a patch-ownership invariant. Adjacent parent
  // boxes collectively provide a stencil neighborhood, and periodic boundaries wrap that
  // neighborhood onto the opposite side of the parent domain. Restrict only cells whose complete
  // margin is covered by the parent LEVEL, then cluster once so the result is independent of the
  // parent's arbitrary tiling/DistributionMapping.
  TagBox restricted;
  TagBox admissible;
  // This is topology preparation over host TagBox/BoxArray metadata.  It intentionally stays next
  // to the host clustering provider; moving it to Kokkos would require a device round-trip before
  // cluster() without moving any field computation off the device.
  regrid_detail::collective_stage("regrid nesting-mask preparation", communicator, [&] {
    restricted = TagBox(pdom);
    admissible = TagBox(pdom);
    for (std::int64_t j64 = pdom.lo[1]; j64 <= pdom.hi[1]; ++j64)
      for (std::int64_t i64 = pdom.lo[0]; i64 <= pdom.hi[0]; ++i64) {
        const int i = static_cast<int>(i64);
        const int j = static_cast<int>(j64);
        const bool supported = regrid_parent_cell_has_nesting_support(
            i, j, margin, pdom, proper_nesting_parents, periodicity, physical_support);
        if (grown.tagged(i, j) && !supported)
          throw std::runtime_error(
              "AMR tagged cell lacks certified parent or physical-ghost nesting support");
        admissible(i, j) = supported ? 1 : 0;
        restricted(i, j) = supported && grown.tagged(i, j) ? 1 : 0;
      }
  });

  // Clustering is an extension seam, so its output is untrusted even when the provider is
  // built in.  Catch locally first and make the failure collective: throwing on one rank while
  // peers proceed to layout-dependent communication would otherwise deadlock later.
  std::vector<Box2D> pending;
  regrid_detail::collective_stage("AMR clustering provider", communicator, [&] {
    pending = clustering.cluster(restricted);
    for (std::size_t index = 0; index < pending.size(); ++index) {
      const Box2D& box = pending[index];
      if (box.empty())
        throw std::runtime_error("AMR clustering provider returned an empty box");
      if (!pdom.contains(box))
        throw std::runtime_error("AMR clustering provider returned a box outside the tag domain");
      for (std::size_t previous = 0; previous < index; ++previous)
        if (!box.intersect(pending[previous]).empty())
          throw std::runtime_error("AMR clustering provider returned overlapping boxes");
    }
    for (std::int64_t j64 = pdom.lo[1]; j64 <= pdom.hi[1]; ++j64)
      for (std::int64_t i64 = pdom.lo[0]; i64 <= pdom.hi[0]; ++i64) {
        const int i = static_cast<int>(i64);
        const int j = static_cast<int>(j64);
        if (!restricted.tagged(i, j))
          continue;
        const bool covered = std::any_of(pending.begin(), pending.end(),
                                         [=](const Box2D& box) { return box.contains(i, j); });
        if (!covered)
          throw std::runtime_error("AMR clustering provider dropped a tagged cell");
      }
  });

  // Canonicalize provider ordering, then authenticate exact cross-rank layout consensus.  Tags
  // are collective above, but an external provider may still be nondeterministic or rank-aware.
  regrid_detail::collective_stage("regrid provider-order canonicalization", communicator, [&] {
    std::sort(pending.begin(), pending.end(), [](const Box2D& left, const Box2D& right) {
      return std::tie(left.lo[0], left.lo[1], left.hi[0], left.hi[1]) <
             std::tie(right.lo[0], right.lo[1], right.hi[0], right.hi[1]);
    });
  });
  const long local_box_count = static_cast<long>(pending.size());
  if (all_reduce_min(local_box_count, communicator) !=
      all_reduce_max(local_box_count, communicator))
    throw std::runtime_error("AMR clustering provider returned inconsistent MPI layouts");
  std::vector<long> canonical_layout;
  std::vector<long> minimum_layout;
  std::vector<long> maximum_layout;
  regrid_detail::collective_stage("regrid layout-consensus buffers", communicator, [&] {
    canonical_layout.reserve(pending.size() * 4u);
    for (const Box2D& box : pending) {
      canonical_layout.push_back(box.lo[0]);
      canonical_layout.push_back(box.lo[1]);
      canonical_layout.push_back(box.hi[0]);
      canonical_layout.push_back(box.hi[1]);
    }
    minimum_layout = canonical_layout;
    maximum_layout = canonical_layout;
  });
  all_reduce_min_inplace(minimum_layout.data(), minimum_layout.size(), communicator);
  all_reduce_max_inplace(maximum_layout.data(), maximum_layout.size(), communicator);
  if (minimum_layout != maximum_layout)
    throw std::runtime_error("AMR clustering provider returned inconsistent MPI layouts");

  std::optional<BoxArray> candidate_layout;
  // A provider may legally return a bounding box spanning a hole in the admissible mask. Split such
  // boxes deterministically until every published box is fully supported; tagged cells are never
  // dropped merely because the parent level is tiled or has a genuine coverage hole.
  regrid_detail::collective_stage("regrid candidate-layout preparation", communicator, [&] {
    std::vector<Box2D> clusters;
    while (!pending.empty()) {
      const Box2D box = pending.back();
      pending.pop_back();
      bool has_tag = false, fully_admissible = true;
      for (std::int64_t j64 = box.lo[1]; j64 <= box.hi[1]; ++j64)
        for (std::int64_t i64 = box.lo[0]; i64 <= box.hi[0]; ++i64) {
          const int i = static_cast<int>(i64);
          const int j = static_cast<int>(j64);
          has_tag = has_tag || restricted.tagged(i, j);
          fully_admissible = fully_admissible && admissible.tagged(i, j);
        }
      if (!has_tag)
        continue;
      if (fully_admissible) {
        clusters.push_back(box);
        continue;
      }
      if (box.num_cells() == 1)
        continue;
      if (box.nx() >= box.ny()) {
        const int split = static_cast<int>(static_cast<std::int64_t>(box.lo[0]) +
                                           box.length64(0) / 2 - 1);
        pending.push_back(Box2D{{box.lo[0], box.lo[1]}, {split, box.hi[1]}});
        pending.push_back(Box2D{{split + 1, box.lo[1]}, {box.hi[0], box.hi[1]}});
      } else {
        const int split = static_cast<int>(static_cast<std::int64_t>(box.lo[1]) +
                                           box.length64(1) / 2 - 1);
        pending.push_back(Box2D{{box.lo[0], box.lo[1]}, {box.hi[0], split}});
        pending.push_back(Box2D{{box.lo[0], split + 1}, {box.hi[0], box.hi[1]}});
      }
    }
    std::sort(clusters.begin(), clusters.end(), [](const Box2D& left, const Box2D& right) {
      return std::tie(left.lo[0], left.lo[1], left.hi[0], left.hi[1]) <
             std::tie(right.lo[0], right.lo[1], right.hi[0], right.hi[1]);
    });
    std::vector<Box2D> fine_boxes;
    fine_boxes.reserve(clusters.size());
    for (const Box2D& cluster : clusters)
      fine_boxes.push_back(cluster.refine(refinement_ratio));
    candidate_layout.emplace(std::move(fine_boxes));
    if (proper_nesting_parents != nullptr && candidate_layout->size() > 0)
      validate_fine_layout_proper_nesting(*candidate_layout, *proper_nesting_parents, pdom,
                                          refinement_ratio, margin, periodicity,
                                          physical_support);
  });
  if (candidate_layout->size() == 0)
    return {BoxArray{}, DistributionMapping{}};
  DistributionMapping mapping =
      load_balance.distribute(*candidate_layout, communicator.size(), {}, communicator);
  return {std::move(*candidate_layout), std::move(mapping)};
}

inline std::pair<BoxArray, DistributionMapping> regrid_compute_fine_layout(
    TagBox grown, const Box2D& pdom, int pk, int margin, bool coarse_replicated,
    const ClusterParams& cluster, const PreparedLoadBalanceAuthority& load_balance,
    const CommunicatorView& communicator,
    int refinement_ratio = kAmrRefRatio,
    const BoxArray* proper_nesting_parents = nullptr,
    RegridPeriodicity periodicity = {},
    const RegridPhysicalGhostSupport* physical_support = nullptr) {
  const amr::BergerRigoutsosProvider provider(cluster);
  return regrid_compute_fine_layout_with_provider(std::move(grown), pdom, pk, margin,
                                                  coarse_replicated, provider, load_balance,
                                                  communicator,
                                                  refinement_ratio,
                                                  proper_nesting_parents, periodicity,
                                                  physical_support);
}

using RegridProlongation = std::function<void(const MultiFab&, MultiFab&, int, int, bool,
                                              const CommunicatorView&)>;

inline MultiFab regrid_field_on_layout_with_provider(
    const BoxArray& fb, const DistributionMapping& dmap, const MultiFab& par, const MultiFab& old,
    int pk, int ngf, const RegridProlongation& prolong, const CommunicatorView& communicator,
    bool coarse_replicated,
    int refinement_ratio = kAmrRefRatio) {
  int ncomp = 0;
  regrid_detail::collective_stage("regrid field contract", communicator, [&] {
    if (!prolong)
      throw std::runtime_error("regrid_field_on_layout requires a prepared prolongation provider");
    if (communicator.size() != n_ranks() || communicator.rank() != my_rank())
      throw std::runtime_error(
          "regrid field communicator must preserve the field-storage rank space");
    if (pk < 0 || refinement_ratio < 2 || ngf < 0)
      throw std::runtime_error("regrid field indices, ratio, or ghost width are invalid");
    ncomp = old.box_array().size() > 0 ? old.ncomp() : par.ncomp();
    if (ncomp != par.ncomp())
      throw std::runtime_error("regrid_field_on_layout parent/fine component mismatch");
  });

  std::optional<MultiFab> candidate;
  regrid_detail::collective_stage("regrid field allocation", communicator, [&] {
    candidate.emplace(fb, dmap, ncomp, ngf);
    candidate->set_val(Real(0));
  });
  regrid_detail::collective_stage("regrid field prolongation", communicator, [&] {
    prolong(par, *candidate, pk, refinement_ratio, (pk == 0) && coarse_replicated, communicator);
    device_fence();
  });
  if (old.box_array().size() > 0) {
    regrid_detail::collective_stage("regrid old-fine carry-over", communicator, [&] {
      parallel_copy(*candidate, old, communicator);
      device_fence();
    });
  }
  return std::move(*candidate);
}

/// Complete hierarchy regrid policy. Every value which changes topology is explicit; there is no
/// hidden all-periodic assumption and no inferred physical-boundary support.
struct HierarchyRegridOptions {
  int tag_buffer = 0;
  int nesting_margin = 0;
  RegridPeriodicity periodicity{};
  const RegridPhysicalGhostSupport* physical_support = nullptr;
};

/// Canonical dynamic regrid transaction for AmrHierarchy.
///
/// The candidate layout is formed through the same collective/provider-authenticated path used by
/// AmrRuntime. The candidate field is completely prolonged and receives old-fine carry-over before
/// AmrHierarchy publishes it; any validation/provider/allocation failure leaves the hierarchy
/// unchanged. An empty global tag set removes every level above @p coarse_level on every rank.
template <class Crit>
bool regrid_hierarchy_level(AmrHierarchy& hierarchy, int coarse_level, Crit criterion,
                            const HierarchyRegridOptions& options,
                            const amr::ClusteringProvider& clustering,
                            const RegridProlongation& prolongation,
                            const CommunicatorView& communicator) {
  Box2D parent_domain;
  regrid_detail::collective_stage("hierarchy regrid contract", communicator, [&] {
    if (options.tag_buffer < 0 || options.nesting_margin < 0)
      throw std::invalid_argument("hierarchy regrid buffers must be non-negative");
    if (communicator.size() != n_ranks() || communicator.rank() != my_rank())
      throw std::invalid_argument(
          "hierarchy regrid communicator must preserve the field-storage rank space");
    parent_domain = hierarchy.domain(coarse_level);
    (void)hierarchy.data(coarse_level);
  });
  const bool parent_replicated =
      hierarchy.level_is_replicated(coarse_level, communicator);
  const std::array<long, 14> hierarchy_contract{
      static_cast<long>(coarse_level),       static_cast<long>(hierarchy.num_levels()),
      static_cast<long>(hierarchy.ref_ratio()),
      static_cast<long>(hierarchy.ncomp()),  static_cast<long>(hierarchy.n_grow()),
      static_cast<long>(options.tag_buffer), static_cast<long>(options.nesting_margin),
      options.periodicity.x ? 1L : 0L,       options.periodicity.y ? 1L : 0L,
      parent_replicated ? 1L : 0L,
      static_cast<long>(parent_domain.lo[0]), static_cast<long>(parent_domain.lo[1]),
      static_cast<long>(parent_domain.hi[0]), static_cast<long>(parent_domain.hi[1])};
  std::array<long, 14> hierarchy_min = hierarchy_contract;
  std::array<long, 14> hierarchy_max = hierarchy_contract;
  all_reduce_min_inplace(hierarchy_min.data(), hierarchy_min.size(), communicator);
  all_reduce_max_inplace(hierarchy_max.data(), hierarchy_max.size(), communicator);
  if (hierarchy_min != hierarchy_max)
    throw std::runtime_error("hierarchy regrid contract differs between MPI ranks");

  TagBox grown;
  regrid_detail::collective_stage("hierarchy regrid tagging", communicator, [&] {
    TagBox local_tags = tag_cells(hierarchy.data(coarse_level), parent_domain, criterion);
    grown = grow_regrid_tags(local_tags, options.tag_buffer, parent_domain,
                             options.periodicity);
  });
  const BoxArray& parents = hierarchy.boxes(coarse_level);
  auto [fine_boxes, distribution] = regrid_compute_fine_layout_with_provider(
      std::move(grown), parent_domain, coarse_level, options.nesting_margin,
      parent_replicated, clustering, hierarchy.load_balance_authority(),
      communicator, hierarchy.ref_ratio(), &parents, options.periodicity,
      options.physical_support);
  if (fine_boxes.size() == 0) {
    regrid_detail::collective_stage("hierarchy empty-regrid publication", communicator, [&] {
      if (coarse_level < 0 || coarse_level >= hierarchy.num_levels())
        throw std::out_of_range("hierarchy changed during empty regrid publication");
    });
    hierarchy.clear_above(coarse_level);
    return false;
  }

  MultiFab empty_old(BoxArray{}, DistributionMapping{}, hierarchy.ncomp(), hierarchy.n_grow());
  const MultiFab& old_fine =
      hierarchy.num_levels() > coarse_level + 1 ? hierarchy.data(coarse_level + 1) : empty_old;
  MultiFab candidate = regrid_field_on_layout_with_provider(
      fine_boxes, distribution, hierarchy.data(coarse_level), old_fine, coarse_level,
      hierarchy.n_grow(), prolongation, communicator, parent_replicated,
      hierarchy.ref_ratio());
  hierarchy.install_level(coarse_level + 1, fine_boxes, std::move(candidate), communicator);
  return true;
}

/// Regrid the finest level (L.back()) by Berger-Rigoutsos on the criterion @p crit applied to the
/// parent: rebuilds the patches (fine data carry-over otherwise parent interp) + the aux. @p grow:
/// tag dilation; @p margin: nesting; @p aux_ncomp: rebuilt aux width;
/// @p coarse_replicated: ownership policy of level 0. NO-OP if < 2 levels or no patch.
///
/// The caller must provide the aux width, coarse ownership, periodicity and execution communicator
/// explicitly: this free function cannot infer runtime authorities from a criterion or silently
/// assume a periodic/process-world topology.
template <class Crit>
void amr_regrid_finest(std::vector<AmrLevelMP>& L, std::vector<MultiFab>& aux, const Box2D& dom,
                       Crit crit, int grow, int margin, const RegridProlongation& prolong,
                       int aux_ncomp, bool coarse_replicated,
                       const PreparedLoadBalanceAuthority& load_balance,
                       RegridPeriodicity periodicity,
                       const CommunicatorView& communicator,
                       const RegridPhysicalGhostSupport* physical_support = nullptr) {
  int nlev = 0;
  regrid_detail::collective_stage("finest-level regrid contract", communicator, [&] {
    if (communicator.size() != n_ranks() || communicator.rank() != my_rank())
      throw std::invalid_argument(
          "finest-level regrid communicator must preserve the field-storage rank space");
    if (L.size() != aux.size())
      throw std::invalid_argument("finest-level regrid state and aux level counts differ");
    if (L.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("finest-level regrid level count is not representable");
    if (grow < 0 || margin < 0 || aux_ncomp < 1)
      throw std::invalid_argument("finest-level regrid buffers and aux width are invalid");
    nlev = static_cast<int>(L.size());
  });
  const std::array<long, 11> regrid_contract{
      static_cast<long>(nlev),         static_cast<long>(grow),
      static_cast<long>(margin),       static_cast<long>(aux_ncomp),
      coarse_replicated ? 1L : 0L,     periodicity.x ? 1L : 0L,
      periodicity.y ? 1L : 0L,         static_cast<long>(dom.lo[0]),
      static_cast<long>(dom.lo[1]),    static_cast<long>(dom.hi[0]),
      static_cast<long>(dom.hi[1])};
  std::array<long, 11> regrid_min = regrid_contract;
  std::array<long, 11> regrid_max = regrid_contract;
  all_reduce_min_inplace(regrid_min.data(), regrid_min.size(), communicator);
  all_reduce_max_inplace(regrid_max.data(), regrid_max.size(), communicator);
  if (regrid_min != regrid_max)
    throw std::runtime_error("finest-level regrid contract differs between MPI ranks");
  if (nlev < 2)
    return;
  const int fk = nlev - 1, pk = fk - 1;  // fine and its parent
  Box2D pdom;
  TagBox grown;
  regrid_detail::collective_stage("finest-level regrid tagging", communicator, [&] {
    pdom = amr_level_index_domain(dom, pk);
    TagBox tags = tag_cells(L[pk].U, pdom, crit);
    grown = grow_regrid_tags(tags, grow, pdom, periodicity);
  });
  // (1) Compute the fine layout (tags -> grow [already done] -> all_reduce_or -> clustering -> clamp).
  const BoxArray* parents = pk > 0 ? &L[pk].U.box_array() : nullptr;
  auto [fb, dmap] =
      regrid_compute_fine_layout(std::move(grown), pdom, pk, margin, coarse_replicated,
                                 ClusterParams{}, load_balance, communicator, kAmrRefRatio, parents,
                                 periodicity, physical_support);
  if (fb.size() == 0)
    return;  // nothing to refine: keep the current grid
  // The new patches INHERIT the ghost width of the level being replaced (not a frozen 1): a
  // level rebuilt in 2nd-order MUSCL (Minmod / VanLeer) carries 2 ghosts, which the regrid must
  // preserve, otherwise the reconstruction would read out of bounds after re-refinement.
  const int ngf = L[fk].U.n_grow();
  // (2) Re-grid the field with the caller's authenticated prolongation authority, then carry over
  // every cell still covered by the previous fine layout.
  MultiFab candidate_state = regrid_field_on_layout_with_provider(
      fb, dmap, L[pk].U, L[fk].U, pk, ngf, prolong, communicator, coarse_replicated);
  std::optional<MultiFab> candidate_aux;
  regrid_detail::collective_stage("finest-level regrid auxiliary allocation", communicator, [&] {
    candidate_aux.emplace(fb, dmap, aux_ncomp, 1);
  });

  // Both candidates exist and every rank has acknowledged success before either public object is
  // touched. MultiFab's implicit move assignment is no-throw, so this commit cannot strand U and
  // aux on different topology generations.
  static_assert(std::is_nothrow_move_assignable_v<MultiFab>);
  L[fk].U = std::move(candidate_state);
  aux[fk] = std::move(*candidate_aux);
  L[fk].aux = &aux[fk];
}

}  // namespace pops
