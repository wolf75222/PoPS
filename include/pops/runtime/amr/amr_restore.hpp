/// @file
/// @brief Out-of-line AmrRuntime member definitions kept out of amr_runtime.hpp (its line budget):
/// the ADC-542 level-composite reductions (composite_reduce + its folds), the checkpoint
/// hierarchy-rebuild seam (rebuild_hierarchy), and the regrid / clustering config setters
/// (set_regrid, set_clustering). Included at the END of amr_runtime.hpp, so the full AmrRuntime class
/// is visible; NOT a standalone header (it defines AmrRuntime members).
///
/// rebuild_hierarchy imposes a mid-run hierarchy from a v5 checkpoint by REUSING the regrid R6/R7
/// machinery (amr_runtime.hpp regrid()) MINUS tagging / clustering / prolong: the checkpoint supplies
/// the layout (BoxArrays + DistributionMappings) AND the data (the per-level state restore overwrites
/// every valid cell), so the divergence argument of the frozen-hierarchy limitation evaporates -- a
/// restart that imposes the exact mid-run hierarchy makes every post-restart regrid reproduce the
/// uninterrupted layout sequence (the determinism theorem, ADC-542 addendum B.2).

#pragma once

#include <pops/runtime/amr/amr_runtime.hpp>  // the class this file defines members of
#include <pops/runtime/amr/composite_reduction.hpp>

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

namespace pops {

// --- ADC-542 composite reductions (declared in amr_runtime.hpp) -----------------------------------

inline std::size_t AmrRuntime::block_index_by_name_(const std::string& name) const {
  for (std::size_t b = 0; b < blocks_.size(); ++b)
    if (blocks_[b].name == name)
      return b;
  throw std::runtime_error("AmrRuntime::composite_reduce : no block named '" + name + "'");
}

inline double AmrRuntime::composite_reduce(const std::string& block, const std::string& kind,
                                           int comp, const std::vector<int>& levels) const {
  const std::size_t b = block_index_by_name_(block);
  return runtime::amr::composite_reduce_levels(*blocks_[b].levels, replicated_coarse_, kind, comp,
                                               levels);
}

inline std::map<std::string, double> AmrRuntime::step_change_l2(
    const StepSnapshot& previous) const {
  if (previous.block_levels.size() != blocks_.size())
    throw std::runtime_error("AmrRuntime::step_change_l2 snapshot composition mismatch");
  std::map<std::string, double> result;
  for (std::size_t block = 0; block < blocks_.size(); ++block)
    result.emplace(blocks_[block].name,
                   runtime::amr::composite_difference_l2_levels(
                       *blocks_[block].levels, previous.block_levels[block], replicated_coarse_));
  return result;
}

inline double AmrRuntime::composite_reduce_field(const std::string& provider_slot,
                                                 const std::string& kind, int comp,
                                                 const std::vector<int>& levels) {
  const int count = provider_potential_levels(provider_slot);
  if (blocks_.empty() || blocks_.front().levels == nullptr ||
      static_cast<int>(blocks_.front().levels->size()) != count)
    throw std::runtime_error(
        "AmrRuntime::composite_reduce_field: field and shared state hierarchies disagree");
  std::vector<const MultiFab*> values;
  std::vector<std::pair<Real, Real>> metrics;
  values.reserve(static_cast<std::size_t>(count));
  metrics.reserve(static_cast<std::size_t>(count));
  for (int level = 0; level < count; ++level) {
    values.push_back(&provider_potential_level(provider_slot, level));
    const AmrLevelMP& shared = blocks_.front().levels->at(static_cast<std::size_t>(level));
    metrics.emplace_back(shared.dx, shared.dy);
  }
  return runtime::amr::composite_reduce_fields(values, metrics, replicated_coarse_, kind, comp,
                                               levels);
}

inline std::vector<int> AmrRuntime::level_owner_ranks(int k) const {
  if (k < 0 || k >= nlev_)
    throw std::runtime_error("AmrRuntime::level_owner_ranks : level out of bounds");
  return hierarchy_.dm[static_cast<std::size_t>(k)].ranks();
}

// FULL shared aux of level k: ALL aux_ncomp_ components of aux_[k], LOCAL valid cells at
// level-domain-relative component-major flat indices (zeros outside the patches at a fine level) -- the
// exact layout of block_level_state, so the v5 checkpoint reader/writer share one convention. phi
// (comp 0) is included; the level-0 multigrid WARM START stays a separate phi_<k> payload
// (level_potential), which reads mg_.phi(), not aux_[0].
inline std::vector<double> AmrRuntime::level_aux_flat(int k) const {
  if (k < 0 || k >= nlev_)
    throw std::runtime_error("AmrRuntime::level_aux_flat : level out of bounds");
  const MultiFab& A = aux_[k];
  const int nc = A.ncomp();
  const Box2D level_domain = amr_level_index_domain(dom_, k);
  const std::size_t nx = static_cast<std::size_t>(level_domain.nx());
  const std::size_t cells = nx * static_cast<std::size_t>(level_domain.ny());
  std::vector<double> out(static_cast<std::size_t>(nc) * cells, 0.0);
  device_fence();
  for (int li = 0; li < A.local_size(); ++li) {
    const ConstArray4 a = A.fab(li).const_array();
    const Box2D v = A.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        for (int c = 0; c < nc; ++c)
          out[static_cast<std::size_t>(c) * cells +
              static_cast<std::size_t>(j - level_domain.lo[1]) * nx +
              static_cast<std::size_t>(i - level_domain.lo[0])] = a(i, j, c);
  }
  return out;
}

// Global form of level_aux_flat. Ownership-distributed levels are gathered from their disjoint
// per-rank contributions; replicated level 0 is already complete on every rank and must not be
// reduced (which would multiply the checkpoint payload by n_ranks).
inline std::vector<double> AmrRuntime::level_aux_flat_global(int k) const {
  std::vector<double> out = level_aux_flat(k);
  if (k > 0 || !replicated_coarse_)
    all_reduce_sum_inplace(out.data(), out.size());
  return out;
}

// Restores the FULL shared aux of level k from the flat layout above. Writes ONLY the VALID cells of
// the LOCAL fabs (owner-rank writes; a rank without a box is a no-op); the ghosts are redone by the
// next solve_fields, exactly like after a regrid.
inline void AmrRuntime::set_level_aux_flat(int k, const std::vector<double>& v) {
  if (k < 0 || k >= nlev_)
    throw std::runtime_error("AmrRuntime::set_level_aux_flat : level out of bounds");
  MultiFab& A = aux_[k];
  const int nc = A.ncomp();
  const Box2D level_domain = amr_level_index_domain(dom_, k);
  const std::size_t nx = static_cast<std::size_t>(level_domain.nx());
  const std::size_t cells = nx * static_cast<std::size_t>(level_domain.ny());
  if (v.size() != static_cast<std::size_t>(nc) * cells)
    throw std::runtime_error(
        "AmrRuntime::set_level_aux_flat : aux size differs from ncomp*level_cells");
  device_fence();
  for (int li = 0; li < A.local_size(); ++li) {
    Array4 a = A.fab(li).array();
    const Box2D b = A.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        for (int c = 0; c < nc; ++c)
          a(i, j, c) = v[static_cast<std::size_t>(c) * cells +
                         static_cast<std::size_t>(j - level_domain.lo[1]) * nx +
                         static_cast<std::size_t>(i - level_domain.lo[0])];
  }
}

// --- ADC-542 hierarchy rebuild (v5 checkpoint restore) --------------------------------------------

inline void AmrRuntime::rebuild_hierarchy(const std::vector<std::vector<PatchBox>>& level_boxes,
                                          const std::vector<std::vector<int>>& level_owner_ranks) {
  const int n_levels = static_cast<int>(level_boxes.size());
  if (n_levels < 1)
    throw std::runtime_error("AmrRuntime::rebuild_hierarchy : need at least the coarse level (0)");
  if (level_owner_ranks.size() != level_boxes.size())
    throw std::runtime_error(
        "AmrRuntime::rebuild_hierarchy : level_boxes and level_owner_ranks length mismatch");
  if (n_levels > max_levels())
    throw std::runtime_error("AmrRuntime::rebuild_hierarchy : checkpoint has " +
                             std::to_string(n_levels) +
                             " active levels but the replayed composition resolves a maximum of " +
                             std::to_string(max_levels()));

  // Validate and materialize the complete target topology before replacing any accepted storage.
  // Level zero is the composition-owned base layout and is intentionally absent from patch_boxes();
  // every fine level must be a non-empty contiguous prefix with an explicit owner per patch.
  std::vector<BoxArray> target_boxes(static_cast<std::size_t>(n_levels));
  std::vector<DistributionMapping> target_mappings(static_cast<std::size_t>(n_levels));
  std::vector<Real> target_dx(static_cast<std::size_t>(n_levels));
  std::vector<Real> target_dy(static_cast<std::size_t>(n_levels));
  target_boxes[0] = hierarchy_.ba[0];
  target_mappings[0] = hierarchy_.dm[0];
  target_dx[0] = hierarchy_.dx[0];
  target_dy[0] = hierarchy_.dy[0];
  if (!level_boxes[0].empty() || !level_owner_ranks[0].empty())
    throw std::runtime_error(
        "AmrRuntime::rebuild_hierarchy : level zero is owned by the resolved base layout");
  auto checked_refine_domain = [](const Box2D& domain, int ratio) {
    if (ratio != kAmrRefRatio)
      throw std::runtime_error(
          "AmrRuntime::rebuild_hierarchy : native AMR requires spatial refinement ratio 2");
    Box2D refined;
    for (int direction = 0; direction < 2; ++direction) {
      const std::int64_t lo = static_cast<std::int64_t>(domain.lo[direction]) * ratio;
      const std::int64_t hi = static_cast<std::int64_t>(domain.hi[direction]) * ratio + ratio - 1;
      if (lo < std::numeric_limits<int>::min() || lo > std::numeric_limits<int>::max() ||
          hi < std::numeric_limits<int>::min() || hi > std::numeric_limits<int>::max())
        throw std::runtime_error(
            "AmrRuntime::rebuild_hierarchy : refined index domain overflows native integers");
      refined.lo[direction] = static_cast<int>(lo);
      refined.hi[direction] = static_cast<int>(hi);
    }
    return refined;
  };

  Box2D parent_domain = dom_;
  for (int level = 1; level < n_levels; ++level) {
    const auto index = static_cast<std::size_t>(level);
    if (level_boxes[index].empty() || level_boxes[index].size() != level_owner_ranks[index].size())
      throw std::runtime_error(
          "AmrRuntime::rebuild_hierarchy : every active fine level requires boxes and owners");
    const int ratio = maximum_refinement_ratios_[index - 1];
    const Box2D level_domain = checked_refine_domain(parent_domain, ratio);
    std::vector<Box2D> boxes;
    boxes.reserve(level_boxes[index].size());
    for (std::size_t patch = 0; patch < level_boxes[index].size(); ++patch) {
      const PatchBox& value = level_boxes[index][patch];
      const Box2D box{{value.ilo, value.jlo}, {value.ihi, value.jhi}};
      if (value.level != level || !level_domain.contains(box))
        throw std::runtime_error(
            "AmrRuntime::rebuild_hierarchy : checkpoint patch is outside its declared level");
      for (int direction = 0; direction < 2; ++direction) {
        const std::int64_t aligned_lo =
            static_cast<std::int64_t>(box.lo[direction]) - level_domain.lo[direction];
        const std::int64_t aligned_end =
            static_cast<std::int64_t>(box.hi[direction]) + 1 - level_domain.lo[direction];
        if (aligned_lo % ratio != 0 || aligned_end % ratio != 0)
          throw std::runtime_error(
              "AmrRuntime::rebuild_hierarchy : checkpoint patch is not aligned to parent cells");
      }
      for (const Box2D& prior : boxes)
        if (!prior.intersect(box).empty())
          throw std::runtime_error(
              "AmrRuntime::rebuild_hierarchy : checkpoint fine patches overlap");
      const int owner = level_owner_ranks[index][patch];
      if (owner < 0 || owner >= n_ranks())
        throw std::runtime_error(
            "AmrRuntime::rebuild_hierarchy : checkpoint owner rank is out of range");
      boxes.push_back(box);
    }
    target_boxes[index] = BoxArray(std::move(boxes));
    target_mappings[index] = DistributionMapping(level_owner_ranks[index]);
    const auto physical_support = regrid_physical_ghost_support_(level - 1);
    validate_fine_layout_proper_nesting(target_boxes[index], target_boxes[index - 1], parent_domain,
                                        ratio, regrid_margin_,
                                        RegridPeriodicity{base_per_.x, base_per_.y},
                                        physical_support ? &*physical_support : nullptr);
    target_dx[index] = target_dx[index - 1] / Real(ratio);
    target_dy[index] = target_dy[index - 1] / Real(ratio);
    parent_domain = level_domain;
  }

  const StepSnapshot accepted = step_snapshot();
  try {
    device_fence();
    const int previous_levels = nlev_;
    if (n_levels < previous_levels)
      resize_history_levels_for_restore_(n_levels);

    hierarchy_.ba = std::move(target_boxes);
    hierarchy_.dm = std::move(target_mappings);
    hierarchy_.dx = std::move(target_dx);
    hierarchy_.dy = std::move(target_dy);
    hierarchy_.refinement_ratios.assign(
        maximum_refinement_ratios_.begin(),
        maximum_refinement_ratios_.begin() + static_cast<std::ptrdiff_t>(n_levels - 1));
    nlev_ = n_levels;
    refresh_active_temporal_relations_();

    // Checkpoint payloads overwrite every valid value. Reallocate the exact active prefix without
    // prolongation/restriction, retaining only level-zero accepted storage until its payload lands.
    for (auto& block : blocks_) {
      auto& levels = *block.levels;
      const int ncomp = levels.front().U.ncomp();
      const int ngrow = levels.front().U.n_grow();
      levels.resize(1);
      levels.reserve(static_cast<std::size_t>(n_levels));
      for (int level = 1; level < n_levels; ++level) {
        const auto index = static_cast<std::size_t>(level);
        MultiFab state(hierarchy_.ba[index], hierarchy_.dm[index], ncomp, ngrow);
        levels.push_back(
            AmrLevelMP{std::move(state), nullptr, hierarchy_.dx[index], hierarchy_.dy[index]});
      }
    }
    aux_.resize(1);
    aux_.reserve(static_cast<std::size_t>(n_levels));
    for (int level = 1; level < n_levels; ++level) {
      const auto index = static_cast<std::size_t>(level);
      aux_.emplace_back(hierarchy_.ba[index], hierarchy_.dm[index], aux_ncomp_, 1);
    }
    for (auto& block : blocks_)
      for (int level = 0; level < n_levels; ++level)
        (*block.levels)[static_cast<std::size_t>(level)].aux =
            &aux_[static_cast<std::size_t>(level)];

    // Existing rings may outlive a low-level rebuild. Reallocate every fine slot on the imposed
    // topology and append missing levels without interpolation; restore writes the authenticated
    // buffers immediately afterwards.
    for (int level = 1; level < n_levels; ++level)
      remap_history_rings_(hierarchy_.ba[static_cast<std::size_t>(level)],
                           hierarchy_.dm[static_cast<std::size_t>(level)], level, level - 1,
                           /*prolong=*/false);

    std::vector<std::vector<AmrLevelMP>> shared;
    shared.reserve(blocks_.size());
    for (const auto& block : blocks_)
      shared.push_back(*block.levels);
    detail::same_layout_or_throw(shared);

    invalidate_named_field_topology();
    record_topology_replacement_();
    materialize_boundary_sessions_();
  } catch (...) {
    restore_step_snapshot(accepted);
    throw;
  }
}

// --- Accepted-boundary ownership migration --------------------------------------------------------

inline RebalanceDecision AmrRuntime::decide_rebalance(int level, ResourceEstimates estimates,
                                                      const RebalancePolicy& policy) const {
  const CommunicatorView communicator = world_communicator_view();
  detail::collective_load_balance_preflight("AMR rebalance decision preflight", communicator, [&] {
    if (communicator.size() != n_ranks() || communicator.rank() != my_rank())
      throw std::invalid_argument(
          "AMR rebalance communicator does not preserve the hierarchy rank space");
    if (level <= 0 || level >= nlev_)
      throw std::out_of_range("AMR rebalance currently accepts only an active fine level");
    if (!hierarchy_.load_balance)
      throw std::logic_error("AMR hierarchy has no prepared load-balance authority");
  });
  const std::size_t index = static_cast<std::size_t>(level);
  return hierarchy_.load_balance->decide_rebalance(
      level, hierarchy_.ba[index], hierarchy_.dm[index], n_ranks(), topology_epoch_,
      topology_materialization_generation_, estimates, policy, communicator);
}

inline bool AmrRuntime::apply_rebalance_decision(int level, const RebalanceDecision& decision) {
  const CommunicatorView communicator = world_communicator_view();
  std::string live_contract;
  std::int64_t moved_patches = 0;
  detail::collective_load_balance_preflight("AMR rebalance migration preflight", communicator, [&] {
    if (communicator.size() != n_ranks() || communicator.rank() != my_rank())
      throw std::invalid_argument(
          "AMR rebalance communicator does not preserve the hierarchy rank space");
    if (level <= 0 || level >= nlev_)
      throw std::out_of_range("AMR rebalance currently accepts only an active fine level");
    if (step_rollback_scope_active() || field_solve_transaction_active() || boundary_stage_states_)
      throw std::logic_error("AMR rebalance requires a clean accepted runtime boundary");
    if (decision.topology_epoch != topology_epoch_ ||
        decision.materialization_generation != topology_materialization_generation_)
      throw std::invalid_argument("AMR rebalance decision targets stale topology or storage");
    if (decision.source_contract.empty() || decision.exact_contract.empty() ||
        decision.exact_contract != detail::exact_rebalance_decision(decision))
      throw std::invalid_argument("AMR rebalance decision exact contract is invalid");

    const std::size_t index = static_cast<std::size_t>(level);
    const BoxArray& boxes = hierarchy_.ba[index];
    const DistributionMapping& current = hierarchy_.dm[index];
    for (const auto& [name, field] : bootstrap_staggered_fields_) {
      (void)name;
      if (field.levels.size() > index)
        throw std::logic_error(
            "AMR rebalance does not yet support materialized staggered bootstrap fields");
    }
    if (!hierarchy_.load_balance)
      throw std::logic_error("AMR hierarchy has no prepared load-balance authority");
    live_contract = detail::exact_rebalance_source(
        hierarchy_.load_balance->semantic_identity(),
        hierarchy_.load_balance->collective_contract(), level, n_ranks(), topology_epoch_,
        topology_materialization_generation_, boxes, current);
    if (decision.source_contract != live_contract)
      throw std::invalid_argument(
          "AMR rebalance decision does not target the live prepared level authority");
    if (boxes.size() <= 0 || current.size() != boxes.size() ||
        decision.proposed_mapping.size() != boxes.size())
      throw std::invalid_argument("AMR rebalance decision does not match the active fine BoxArray");
    for (int patch = 0; patch < boxes.size(); ++patch) {
      const int owner = decision.proposed_mapping[patch];
      if (owner < 0 || owner >= n_ranks())
        throw std::invalid_argument("AMR rebalance decision contains an invalid owner rank");
      if (owner != current[patch])
        ++moved_patches;
    }
    if (decision.moved_patches != moved_patches || decision.migration_bytes < 0 ||
        decision.migration_nanoseconds < 0 || decision.current_max_nanoseconds_per_step <= 0 ||
        decision.proposed_max_nanoseconds_per_step <= 0 ||
        !std::isfinite(decision.current_imbalance) || !std::isfinite(decision.proposed_imbalance) ||
        !std::isfinite(decision.predicted_net_speedup) || decision.current_imbalance < 1.0 ||
        decision.proposed_imbalance < 1.0 || decision.predicted_net_speedup <= 0.0)
      throw std::invalid_argument("AMR rebalance decision metrics are incomplete or inconsistent");

    switch (decision.reason) {
      case RebalanceReason::MappingUnchanged:
        if (decision.accepted || moved_patches != 0)
          throw std::invalid_argument(
              "AMR rebalance unchanged decision disagrees with the live mapping");
        break;
      case RebalanceReason::NetBenefit:
        if (!decision.accepted || moved_patches == 0)
          throw std::invalid_argument(
              "AMR rebalance accepted decision has no beneficial migration");
        break;
      case RebalanceReason::InsufficientNetBenefit:
        if (decision.accepted || moved_patches == 0)
          throw std::invalid_argument(
              "AMR rebalance refusal disagrees with the proposed migration");
        break;
      case RebalanceReason::EmptyHierarchy:
        throw std::invalid_argument(
            "AMR rebalance cannot apply an empty-hierarchy decision to an active fine level");
      default:
        throw std::invalid_argument("AMR rebalance decision reason is unsupported");
    }
  });

  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"pops.amr.rebalance-source", live_contract},
           {"pops.amr.rebalance-decision", decision.exact_contract}},
          communicator))
    throw std::invalid_argument(
        "AMR rebalance live hierarchy or decision differs across MPI ranks");
  if (!decision.accepted)
    return false;

  StepSnapshot accepted;
  detail::collective_load_balance_preflight("AMR rebalance snapshot capture", communicator,
                                            [&] { capture_step_snapshot(accepted); });

  const std::size_t index = static_cast<std::size_t>(level);
  const int parent_level = level - 1;
  int refinement_ratio = 0;
  std::optional<BoxArray> boxes;
  std::optional<MultiFab> migrated_aux;
  detail::collective_load_balance_preflight("AMR rebalance carrier allocation", communicator, [&] {
    // Aux fields are not part of a block's conservative prolongation route. Prepare an exact
    // owner-only copy before mutating the hierarchy; field publication may refresh derived ghosts
    // and provider-owned components only after these accepted valid cells are restored.
    boxes.emplace(hierarchy_.ba[index]);
    refinement_ratio = hierarchy_.refinement_ratios[static_cast<std::size_t>(parent_level)];
    migrated_aux.emplace(*boxes, decision.proposed_mapping, aux_[index].ncomp(),
                         aux_[index].n_grow());
  });

  std::exception_ptr migration_failure;
  try {
    regrid_detail::collective_stage("AMR rebalance aux redistribution", communicator, [&] {
      parallel_copy(*migrated_aux, aux_[index], communicator);
    });

    materialize_regrid_transition_(parent_level, *boxes, decision.proposed_mapping,
                                   refinement_ratio);
    detail::collective_load_balance_preflight(
        "AMR rebalance carrier publication", communicator, [&] {
          aux_[index] = std::move(*migrated_aux);
          for (auto& block : blocks_)
            for (int active_level = 0; active_level < nlev_; ++active_level)
              (*block.levels)[static_cast<std::size_t>(active_level)].aux =
                  &aux_[static_cast<std::size_t>(active_level)];
        });

    regrid_detail::collective_stage("AMR rebalance topology publication", communicator, [&] {
      invalidate_named_field_topology();
      record_topology_replacement_();
    });
    regrid_detail::collective_stage("AMR rebalance field publication", communicator, [&] {
      require_solved_field_outcome(solve_fields(),
                                   "AmrRuntime::apply_rebalance_decision publication");
    });
    regrid_detail::collective_stage("AMR rebalance boundary publication", communicator,
                                    [&] { materialize_boundary_sessions_(); });

    detail::collective_load_balance_preflight(
        "AMR rebalance publication validation", communicator, [&] {
          const auto& reference = *blocks_.front().levels;
          for (std::size_t block = 0; block < blocks_.size(); ++block) {
            const auto& levels = *blocks_[block].levels;
            if (levels.size() != reference.size())
              throw std::runtime_error(
                  "AMR rebalance produced different level "
                  "counts across blocks");
            if (levels[index].U.box_array().boxes() != boxes->boxes() ||
                levels[index].U.dmap().ranks() != decision.proposed_mapping.ranks())
              throw std::runtime_error(
                  "AMR rebalance did not publish its exact "
                  "owner mapping on every block");
          }
        });
    require_complete_history_materialization_collective_("AmrRuntime::apply_rebalance_decision");
    regrid_detail::collective_stage("AMR rebalance final device fence", communicator,
                                    [] { device_fence(); });
  } catch (...) {
    migration_failure = std::current_exception();
  }

  const long migration_failures = all_reduce_max(migration_failure ? 1L : 0L, communicator);
  if (migration_failures != 0) {
    std::exception_ptr rollback_failure;
    try {
      restore_step_snapshot(accepted);
    } catch (...) {
      rollback_failure = std::current_exception();
    }
    if (all_reduce_max(rollback_failure ? 1L : 0L, communicator) != 0) {
      if (rollback_failure)
        std::rethrow_exception(rollback_failure);
      throw std::runtime_error("AMR rebalance rollback failed on another MPI rank");
    }
    if (migration_failure)
      std::rethrow_exception(migration_failure);
    throw std::runtime_error("AMR rebalance migration failed on another MPI rank");
  }

  // Profiling is observational. It must never turn an already collectively committed hierarchy
  // into a rank-local rollback attempt.
  if (profiler_ != nullptr)
    try {
      profiler_->count("rebalance");
      profiler_->count("rebalance_moved_patches", moved_patches);
      profiler_->count("rebalance_migration_bytes", decision.migration_bytes);
    } catch (...) {  // NOLINT(bugprone-empty-catch) -- profiling cannot invalidate publication
    }
  return true;
}

// --- regrid / clustering config setters (declared in amr_runtime.hpp) -----------------------------

inline void AmrRuntime::set_regrid(int every, int grow, int margin) {
  if (every < 0)
    throw std::runtime_error("AmrRuntime::set_regrid : regrid_every >= 0");
  regrid_every_ = every;
  regrid_grow_ = grow;
  regrid_margin_ = margin;
}

inline void AmrRuntime::set_clustering(double min_efficiency, int min_box_size, int max_box_size) {
  if (!(min_efficiency > 0.0 && min_efficiency <= 1.0))
    throw std::runtime_error("AmrRuntime::set_clustering : min_efficiency must be in (0, 1]");
  if (min_box_size < 1 || max_box_size < 1)
    throw std::runtime_error("AmrRuntime::set_clustering : box sizes must be >= 1");
  if (min_box_size > max_box_size)
    throw std::runtime_error("AmrRuntime::set_clustering : min_box_size <= max_box_size required");
  cluster_.min_efficiency = min_efficiency;
  cluster_.min_box_size = min_box_size;
  cluster_.max_box_size = max_box_size;
  if (!external_clustering_)
    clustering_provider_ = std::make_shared<const amr::BergerRigoutsosProvider>(cluster_);
}

}  // namespace pops
