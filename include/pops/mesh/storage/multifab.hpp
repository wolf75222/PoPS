/// @file
/// @brief Owning compile-time-ranked field collection over an exact distributed patch layout.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/layout/rank_space.hpp>
#include <pops/mesh/storage/fab.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {

/// Capabilities of the canonical ND MultiFab storage foundation.
///
/// Communication remains deliberately unavailable until dimension-specialized halo, copy, and MPI
/// schedules are promoted.  Callers can inspect this value during preparation and must not infer a
/// legacy two-dimensional fallback from the presence of distributed metadata.
struct MultiFabCapabilities {
  bool local_storage = false;
  bool halo_exchange = false;
  bool parallel_copy = false;
  bool mpi_exchange = false;

  constexpr bool operator==(const MultiFabCapabilities&) const = default;
};

/// Deep-owning local storage selected from an exact ND layout and coordinate distribution.
///
/// Global patch order is retained in `layout()`. `local_global_indices()` records the ordered
/// subset materialized at `local_rank()`, while `global_index()` and `local_index_of()` make the two
/// index spaces explicit.  No process-global rank lookup or communication policy is hidden in this
/// type.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class MultiFab {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "pops::MultiFab only supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;
  static constexpr std::size_t not_local = std::numeric_limits<std::size_t>::max();

  using memory_space = MemorySpace;
  using fab_type = Fab<Dim, MemorySpace>;
  using box_type = Box<Dim>;
  using layout_type = mesh::BoxArray<Dim>;
  using distribution_type = mesh::Distribution<Dim>;
  using rank_space_type = mesh::RankSpace<Dim>;
  using rank_type = Index<Dim>;
  using ghost_type = Extent<Dim>;

  MultiFab() = default;

  MultiFab(layout_type layout, distribution_type distribution, rank_type local_rank, int ncomp,
           ghost_type ghosts)
      : layout_(std::move(layout)),
        distribution_(std::move(distribution)),
        local_rank_(local_rank),
        ncomp_(ncomp),
        ghosts_(ghosts) {
    validate_metadata_();

    local_global_indices_ = distribution_.local_box_indices(local_rank_);
    global_to_local_.assign(layout_.size(), not_local);

    std::vector<fab_type> allocated;
    allocated.reserve(local_global_indices_.size());
    for (const std::size_t global : local_global_indices_) {
      const std::size_t local = allocated.size();
      global_to_local_[global] = local;
      allocated.emplace_back(layout_[global], ncomp_, ghosts_);
    }
    fabs_ = std::move(allocated);
  }

  MultiFab(const MultiFab&) = default;
  MultiFab& operator=(const MultiFab&) = default;

  MultiFab(MultiFab&& other) noexcept { move_from_(std::move(other)); }

  MultiFab& operator=(MultiFab&& other) noexcept {
    if (this != &other) {
      reset_moved_from_();
      move_from_(std::move(other));
    }
    return *this;
  }

  static constexpr MultiFabCapabilities capabilities() noexcept {
    return MultiFabCapabilities{/*local_storage=*/true, /*halo_exchange=*/false,
                                /*parallel_copy=*/false, /*mpi_exchange=*/false};
  }

  /// Fail closed when a caller reaches a communication path before an ND schedule is installed.
  [[noreturn]] static void require_communication(std::string_view operation) {
    const std::string requested =
        operation.empty() ? "unspecified operation" : std::string(operation);
    throw std::logic_error(
        "pops::MultiFab<Dim>: " + requested +
        " requires a prepared ND communication schedule; no halo, parallel-copy, "
        "or MPI schedule is available in the storage foundation");
  }

  const layout_type& layout() const noexcept { return layout_; }
  const layout_type& box_array() const noexcept { return layout_; }
  const distribution_type& distribution() const noexcept { return distribution_; }
  const rank_space_type& rank_space() const noexcept { return distribution_.rank_space(); }
  const rank_type& local_rank() const noexcept { return local_rank_; }
  int ncomp() const noexcept { return ncomp_; }
  const ghost_type& ghosts() const noexcept { return ghosts_; }

  std::size_t local_size() const noexcept { return fabs_.size(); }
  const std::vector<std::size_t>& local_global_indices() const noexcept {
    return local_global_indices_;
  }

  bool contains_local(std::size_t global) const noexcept {
    return global < global_to_local_.size() && global_to_local_[global] != not_local;
  }

  std::size_t global_index(std::size_t local) const {
    require_local_index_(local);
    return local_global_indices_[local];
  }

  /// Return the local offset for a valid global patch, or `not_local` for a remote patch.
  std::size_t local_index_of(std::size_t global) const {
    require_global_index_(global);
    return global_to_local_[global];
  }

  fab_type& fab(std::size_t local) {
    require_local_index_(local);
    return fabs_[local];
  }

  const fab_type& fab(std::size_t local) const {
    require_local_index_(local);
    return fabs_[local];
  }

  fab_type& fab_global(std::size_t global) { return fabs_[require_local_global_index_(global)]; }

  const fab_type& fab_global(std::size_t global) const {
    return fabs_[require_local_global_index_(global)];
  }

  const box_type& box(std::size_t local) const { return fab(local).box(); }

  /// Storage is always deep-owned; only the same object can expose the same owned allocation set.
  bool shares_storage_with(const MultiFab& other) const noexcept { return this == &other; }

  /// Fill valid and ghost cells in the selected memory space through Kokkos deep copies.
  void set_val(Real value) {
    for (fab_type& local_fab : fabs_)
      local_fab.set_val(value);
  }

 private:
  void validate_metadata_() const {
    if (!distribution_.matches_layout(layout_))
      throw std::invalid_argument(
          "pops::MultiFab: distribution layout does not exactly match the field layout");
    if (!distribution_.rank_space().contains(local_rank_))
      throw std::out_of_range("pops::MultiFab: local rank is outside the distribution rank space");
    if (ncomp_ < 1)
      throw std::invalid_argument("pops::MultiFab: ncomp must be positive");
    for (int axis = 0; axis < Dim; ++axis)
      if (ghosts_[axis] < 0)
        throw std::invalid_argument("pops::MultiFab: ghost extents must be non-negative");
  }

  void require_local_index_(std::size_t local) const {
    if (local >= fabs_.size())
      throw std::out_of_range("pops::MultiFab: local patch index is outside local storage");
  }

  void require_global_index_(std::size_t global) const {
    if (global >= layout_.size())
      throw std::out_of_range("pops::MultiFab: global patch index is outside the layout");
  }

  std::size_t require_local_global_index_(std::size_t global) const {
    require_global_index_(global);
    const std::size_t local = global_to_local_[global];
    if (local == not_local)
      throw std::out_of_range("pops::MultiFab: global patch is not materialized on this rank");
    return local;
  }

  void reset_moved_from_() noexcept {
    layout_ = layout_type{};
    distribution_ = distribution_type{};
    local_rank_ = rank_type{};
    ncomp_ = 0;
    ghosts_ = ghost_type{};
    local_global_indices_.clear();
    global_to_local_.clear();
    fabs_.clear();
  }

  void move_from_(MultiFab&& other) noexcept {
    layout_ = std::move(other.layout_);
    distribution_ = std::move(other.distribution_);
    local_rank_ = other.local_rank_;
    ncomp_ = other.ncomp_;
    ghosts_ = other.ghosts_;
    local_global_indices_ = std::move(other.local_global_indices_);
    global_to_local_ = std::move(other.global_to_local_);
    fabs_ = std::move(other.fabs_);
    other.reset_moved_from_();
  }

  layout_type layout_{};
  distribution_type distribution_{};
  rank_type local_rank_{};
  int ncomp_ = 0;
  ghost_type ghosts_{};
  std::vector<std::size_t> local_global_indices_{};
  std::vector<std::size_t> global_to_local_{};
  std::vector<fab_type> fabs_{};
};

}  // namespace pops
