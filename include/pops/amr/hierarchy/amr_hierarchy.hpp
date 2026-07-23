/// @file
/// @brief AmrHierarchy: the stack of refinement levels (container of the AMR hierarchy).
///
/// Layer: `include/pops/amr` (AMR geometric primitives).
/// Role: carries, per level, the domain in index space and the MultiFab field (whose immutable
/// layout is the level BoxArray).
/// Level 0 = the coarsest; fixed integer refinement ratio (2 by default).
/// Contract: hierarchy mutations validate a complete candidate before publishing it. Dynamic
/// tagging/clustering is owned by the canonical regrid transaction in amr_regrid_coupler.hpp;
/// this type only installs an already-authenticated level.
///
/// Invariants:
/// - domain(lev) == domain(lev-1).refine(ref_ratio): domains nested by the fixed ratio;
/// - replacing or clearing a level invalidates and removes all finer levels.

#pragma once

#include <pops/amr/hierarchy/refinement_ratio.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/prepared_load_balance.hpp>

#include <array>
#include <exception>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops {

/// Stack of refined levels (domain + BoxArray + MultiFab per level), level 0 the coarsest.
///
/// Usage: built with only level 0 (coarse), then expanded by add_level / install_level.
/// Contract: fixed integer ref_ratio; domain(lev) follows from domain(lev-1) by refine(ref_ratio).
/// Invariants: every published level is non-empty, ratio-aligned, confined to its logical domain,
/// disjoint, covered by the parent level, and has the hierarchy's exact component/ghost width.
/// install/clear truncate finer levels to keep the stack consistent.
class AmrHierarchy {
 public:
  /// Builds the hierarchy with only its level 0 (coarse).
  /// @param coarse_domain domain of level 0 in index space (INCLUSIVE lo/hi corners).
  /// @param max_grid_size max box size for the split into BoxArray.
  /// @param ncomp number of field components per cell.
  /// @param ngrow number of ghost layers of the MultiFab.
  /// @param ref_ratio integer refinement ratio; only kAmrRefRatio (2) is supported today
  ///        and any other value is rejected at construction (see refinement_ratio.hpp).
  AmrHierarchy(const Box2D& coarse_domain, int max_grid_size, int ncomp, int ngrow,
               std::shared_ptr<const PreparedLoadBalanceAuthority> load_balance,
               int ref_ratio = kAmrRefRatio)
      : ref_ratio_(ref_ratio),
        ncomp_(ncomp),
        ngrow_(ngrow),
        load_balance_(std::move(load_balance)) {
    require_supported_ref_ratio(ref_ratio);
    validate_constructor_inputs_(coarse_domain, max_grid_size);
    BoxArray boxes = BoxArray::from_domain(coarse_domain, max_grid_size);
    if (!boxes.tiles_exactly(coarse_domain))
      throw std::invalid_argument("AmrHierarchy coarse boxes must tile the coarse domain exactly");
    MultiFab data(boxes, load_balance_->distribute(boxes, n_ranks()), ncomp_, ngrow_);
    levels_.push_back(
        std::make_unique<LevelStorage>(LevelStorage{coarse_domain, std::move(data)}));
  }

  /// Adds a fine level defined by its BoxArray (in fine index space).
  /// @param fine_ba boxes of the new level, expressed in the refined index space.
  /// The level domain is deduced by refine(ref_ratio) from the previous level domain.
  void add_level(const BoxArray& fine_ba) {
    const Box2D domain = levels_.back()->domain.refine(ref_ratio_);
    validate_fine_layout_(fine_ba, domain, levels_.back()->data.box_array());
    MultiFab data(fine_ba, load_balance_->distribute(fine_ba, n_ranks()), ncomp_, ngrow_);
    auto candidate = std::make_unique<LevelStorage>(LevelStorage{domain, std::move(data)});
    // unique_ptr has a no-throw move. If vector growth fails, the hierarchy remains unchanged.
    levels_.push_back(std::move(candidate));
  }

  /// Installs (adds or replaces) a fine level at index lev. Used by the regrid.
  /// @param lev index of the level to install; must satisfy 1 <= lev <= num_levels().
  /// @param fine_ba boxes of the level in the refined index space.
  /// @param data MultiFab already built for this level (transferred by move).
  /// Replacing an existing level INVALIDATES and removes all finer levels.
  void install_level(int lev, const BoxArray& fine_ba, MultiFab data,
                     const CommunicatorView& communicator) {
    collective_stage_("AmrHierarchy::install_level contract", communicator, [&] {
      if (communicator.size() != n_ranks() || communicator.rank() != my_rank())
        throw std::invalid_argument(
            "execution communicator must preserve the field-storage rank space");
      if (lev < 1 || lev > num_levels())
        throw std::out_of_range(
            "AmrHierarchy::install_level: lev must satisfy 1 <= lev <= num_levels()");
    });

    const Box2D dom = levels_[static_cast<std::size_t>(lev - 1)]->domain.refine(ref_ratio_);
    const std::array<long, 10> metadata{
        static_cast<long>(lev),         static_cast<long>(num_levels()),
        static_cast<long>(ref_ratio_),  static_cast<long>(ncomp_),
        static_cast<long>(ngrow_),      static_cast<long>(dom.lo[0]),
        static_cast<long>(dom.lo[1]),   static_cast<long>(dom.hi[0]),
        static_cast<long>(dom.hi[1]),   static_cast<long>(fine_ba.size())};
    std::array<long, 10> minimum = metadata;
    std::array<long, 10> maximum = metadata;
    all_reduce_min_inplace(minimum.data(), minimum.size(), communicator);
    all_reduce_max_inplace(maximum.data(), maximum.size(), communicator);
    if (minimum != maximum)
      throw std::runtime_error(
          "AmrHierarchy::install_level hierarchy metadata differs between MPI ranks");

    std::vector<long> parent_layout;
    collective_stage_("AmrHierarchy::install_level parent layout", communicator, [&] {
      const BoxArray& parents =
          levels_[static_cast<std::size_t>(lev - 1)]->data.box_array();
      parent_layout.reserve(static_cast<std::size_t>(parents.size()) * 4u);
      for (const Box2D& box : parents.boxes()) {
        parent_layout.push_back(box.lo[0]);
        parent_layout.push_back(box.lo[1]);
        parent_layout.push_back(box.hi[0]);
        parent_layout.push_back(box.hi[1]);
      }
      validate_fine_layout_(fine_ba, dom, parents);
    });
    const long parent_words = static_cast<long>(parent_layout.size());
    if (all_reduce_min(parent_words, communicator) !=
        all_reduce_max(parent_words, communicator))
      throw std::runtime_error(
          "AmrHierarchy::install_level parent layout differs between MPI ranks");
    std::vector<long> parent_min;
    std::vector<long> parent_max;
    collective_stage_("AmrHierarchy::install_level parent consensus buffers", communicator, [&] {
      parent_min = parent_layout;
      parent_max = parent_layout;
    });
    all_reduce_min_inplace(parent_min.data(), parent_min.size(), communicator);
    all_reduce_max_inplace(parent_max.data(), parent_max.size(), communicator);
    if (parent_min != parent_max)
      throw std::runtime_error(
          "AmrHierarchy::install_level parent layout differs between MPI ranks");

    // Re-run the prepared ownership authority on the exact candidate BoxArray. Equality with this
    // collectively authenticated result is the install provenance: a caller cannot publish an
    // arbitrary owner vector merely because its boxes/components happen to match.
    const DistributionMapping expected =
        load_balance_->distribute(fine_ba, communicator.size(), {}, communicator);
    std::unique_ptr<LevelStorage> candidate;
    collective_stage_("AmrHierarchy::install_level candidate", communicator, [&] {
      validate_level_data_(fine_ba, data);
      if (data.dmap().ranks() != expected.ranks())
        throw std::invalid_argument(
            "AmrHierarchy installed field distribution does not match the prepared "
            "load-balance authority");
      candidate =
          std::make_unique<LevelStorage>(LevelStorage{dom, std::move(data)});
    });

    // Make append capacity part of preparation. After this consensus, publishing is a sequence of
    // no-throw unique_ptr moves/erasures and therefore cannot expose only a prefix of the topology.
    collective_stage_("AmrHierarchy::install_level publication capacity", communicator, [&] {
      if (lev == num_levels())
        levels_.reserve(levels_.size() + 1u);
    });
    static_assert(std::is_nothrow_move_constructible_v<MultiFab>);
    if (lev == num_levels()) {
      levels_.push_back(std::move(candidate));
    } else {
      levels_[static_cast<std::size_t>(lev)] = std::move(candidate);
      levels_.erase(levels_.begin() + lev + 1, levels_.end());
    }
  }

  /// Removes all levels strictly finer than lev (no-op if lev is already the finest).
  void clear_above(int lev) {
    validate_level_index_(lev, "clear_above");
    if (lev + 1 < num_levels()) {
      levels_.resize(static_cast<std::size_t>(lev + 1));
    }
  }

  /// Number of levels present (>= 1: level 0 always exists).
  int num_levels() const { return static_cast<int>(levels_.size()); }
  /// Integer refinement ratio between consecutive levels.
  int ref_ratio() const { return ref_ratio_; }
  /// Number of field components per cell.
  int ncomp() const { return ncomp_; }
  /// Number of ghost layers of the MultiFab.
  int n_grow() const { return ngrow_; }

  /// Domain of level lev in index space (INCLUSIVE lo/hi corners).
  const Box2D& domain(int lev) const {
    validate_level_index_(lev, "domain");
    return levels_[static_cast<std::size_t>(lev)]->domain;
  }
  /// BoxArray (split into boxes) of level lev.
  const BoxArray& boxes(int lev) const {
    validate_level_index_(lev, "boxes");
    return levels_[static_cast<std::size_t>(lev)]->data.box_array();
  }
  /// Field of level lev (mutable access).
  MultiFab& data(int lev) {
    validate_level_index_(lev, "data");
    return levels_[static_cast<std::size_t>(lev)]->data;
  }
  /// Field of level lev (const access).
  const MultiFab& data(int lev) const {
    validate_level_index_(lev, "data const");
    return levels_[static_cast<std::size_t>(lev)]->data;
  }
  /// Prepared ownership authority shared by initial levels and every subsequent regrid.
  const PreparedLoadBalanceAuthority& load_balance_authority() const noexcept {
    return *load_balance_;
  }

  /// Authenticate a published level against the prepared ownership authority and report whether
  /// every communicator rank owns a complete local copy. AmrHierarchy levels are normally
  /// unique-owner distributed; the one-rank case is trivially complete.
  bool level_is_replicated(int lev, const CommunicatorView& communicator) const {
    collective_stage_("AmrHierarchy::level_is_replicated contract", communicator, [&] {
      if (communicator.size() != n_ranks() || communicator.rank() != my_rank())
        throw std::invalid_argument(
            "execution communicator must preserve the field-storage rank space");
      validate_level_index_(lev, "level_is_replicated");
    });
    const MultiFab& field = levels_[static_cast<std::size_t>(lev)]->data;
    const DistributionMapping expected =
        load_balance_->distribute(field.box_array(), communicator.size(), {}, communicator);
    collective_stage_("AmrHierarchy::level_is_replicated ownership", communicator, [&] {
      if (field.dmap().ranks() != expected.ranks())
        throw std::invalid_argument(
            "published level distribution does not match the prepared load-balance authority");
    });
    const long owns_complete_level = field.local_size() == field.box_array().size() ? 1L : 0L;
    return all_reduce_min(owns_complete_level, communicator) == 1L;
  }

 private:
  struct LevelStorage {
    Box2D domain;
    MultiFab data;
  };

  int ref_ratio_;
  int ncomp_;
  int ngrow_;
  std::shared_ptr<const PreparedLoadBalanceAuthority> load_balance_;
  std::vector<std::unique_ptr<LevelStorage>> levels_{};

  template <class Operation>
  static void collective_stage_(const char* context, const CommunicatorView& communicator,
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

  void validate_constructor_inputs_(const Box2D& coarse_domain, int max_grid_size) const {
    if (!load_balance_)
      throw std::invalid_argument("AmrHierarchy requires a prepared load-balance authority");
    if (coarse_domain.empty())
      throw std::invalid_argument("AmrHierarchy requires a non-empty coarse domain");
    if (max_grid_size <= 0)
      throw std::invalid_argument("AmrHierarchy max_grid_size must be strictly positive");
    if (ncomp_ < 1)
      throw std::invalid_argument("AmrHierarchy ncomp must be at least one");
    if (ngrow_ < 0)
      throw std::invalid_argument("AmrHierarchy ngrow must be non-negative");
  }

  void validate_level_index_(int level, const char* operation) const {
    if (level < 0 || level >= num_levels())
      throw std::out_of_range(std::string("AmrHierarchy::") + operation +
                              ": level is outside [0, num_levels)");
  }

  static std::int64_t covered_cell_count_(const Box2D& footprint,
                                          const BoxArray& parents) {
    std::int64_t covered = 0;
    for (const Box2D& parent : parents.boxes()) {
      const std::int64_t cells = footprint.intersect(parent).num_cells();
      if (cells > footprint.num_cells() - covered)
        throw std::invalid_argument("AmrHierarchy parent layout overlaps a child footprint");
      covered += cells;
    }
    return covered;
  }

  void validate_fine_layout_(const BoxArray& fine, const Box2D& fine_domain,
                             const BoxArray& parents) const {
    if (fine.size() == 0)
      throw std::invalid_argument("AmrHierarchy fine level must contain at least one box");
    for (int current = 0; current < fine.size(); ++current) {
      const Box2D& box = fine[current];
      if (box.empty())
        throw std::invalid_argument("AmrHierarchy fine level contains an empty box");
      if (!fine_domain.contains(box))
        throw std::invalid_argument("AmrHierarchy fine box lies outside its logical domain");
      const Box2D footprint = box.coarsen(ref_ratio_);
      if (footprint.refine(ref_ratio_) != box)
        throw std::invalid_argument(
            "AmrHierarchy fine box is not aligned to complete parent cells");
      if (covered_cell_count_(footprint, parents) != footprint.num_cells())
        throw std::invalid_argument(
            "AmrHierarchy fine box is not confined to the parent-level coverage");
      for (int previous = 0; previous < current; ++previous)
        if (!box.intersect(fine[previous]).empty())
          throw std::invalid_argument("AmrHierarchy fine boxes must not overlap");
    }
  }

  void validate_level_data_(const BoxArray& boxes, const MultiFab& data) const {
    if (data.box_array().boxes() != boxes.boxes())
      throw std::invalid_argument(
          "AmrHierarchy installed field layout must exactly match the level boxes");
    if (data.ncomp() != ncomp_)
      throw std::invalid_argument(
          "AmrHierarchy installed field component count differs from the hierarchy");
    if (data.n_grow() != ngrow_)
      throw std::invalid_argument(
          "AmrHierarchy installed field ghost width differs from the hierarchy");
  }
};

}  // namespace pops
