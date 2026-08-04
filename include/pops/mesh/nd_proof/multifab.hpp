/// @file
/// @brief Private local-storage proof over explicit ND distribution metadata.
///
/// This has no halo, copy schedule, staging, or communication semantics.

#pragma once

#include <pops/mesh/nd_proof/distribution.hpp>
#include <pops/mesh/storage/fab.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pops::mesh::nd_proof {

/// Local Fab collection selected by explicit coordinate ownership.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class MultiFab {
  static_assert(Dim >= 1 && Dim <= 3, "nd_proof::MultiFab only supports dimensions 1, 2, and 3");

 public:
  using fab_type = Fab<Dim, MemorySpace>;
  using rank_type = Index<Dim>;

  MultiFab() = default;

  MultiFab(const BoxArray<Dim>& layout, const Distribution<Dim>& distribution,
           const rank_type& local_rank, int ncomp, Extent<Dim> ghosts)
      : layout_(layout),
        distribution_(distribution),
        local_rank_(local_rank),
        ncomp_(ncomp),
        ghosts_(ghosts) {
    validate_metadata();
    local_global_indices_ = distribution_.local_box_indices(local_rank_);

    std::vector<fab_type> allocated;
    allocated.reserve(local_global_indices_.size());
    for (const std::size_t global_box : local_global_indices_)
      allocated.emplace_back(layout_[global_box], ncomp_, ghosts_);
    fabs_ = std::move(allocated);
  }

  MultiFab(const MultiFab&) = default;
  MultiFab& operator=(const MultiFab&) = default;

  MultiFab(MultiFab&& other) noexcept { move_from(std::move(other)); }
  MultiFab& operator=(MultiFab&& other) noexcept {
    if (this != &other) {
      reset_moved_from();
      move_from(std::move(other));
    }
    return *this;
  }

  const BoxArray<Dim>& layout() const noexcept { return layout_; }
  const Distribution<Dim>& distribution() const noexcept { return distribution_; }
  const rank_type& local_rank() const noexcept { return local_rank_; }
  int ncomp() const noexcept { return ncomp_; }
  const Extent<Dim>& ghosts() const noexcept { return ghosts_; }
  const std::vector<std::size_t>& local_global_indices() const noexcept {
    return local_global_indices_;
  }
  std::size_t local_size() const noexcept { return fabs_.size(); }

  bool contains_local(std::size_t global_box) const noexcept {
    for (const std::size_t local_global : local_global_indices_)
      if (local_global == global_box)
        return true;
    return false;
  }

  fab_type& fab(std::size_t global_box) { return fabs_.at(local_offset(global_box)); }
  const fab_type& fab(std::size_t global_box) const { return fabs_.at(local_offset(global_box)); }

 private:
  void validate_metadata() const {
    if (distribution_.box_count() != layout_.size() || !distribution_.matches_layout(layout_))
      throw std::invalid_argument(
          "nd_proof::MultiFab distribution layout does not structurally match layout");
    if (!distribution_.rank_space().contains(local_rank_))
      throw std::out_of_range("nd_proof::MultiFab local rank is outside the rank space");
    if (ncomp_ < 1)
      throw std::invalid_argument("nd_proof::MultiFab ncomp must be positive");
    for (int axis = 0; axis < Dim; ++axis)
      if (ghosts_[axis] < 0)
        throw std::invalid_argument("nd_proof::MultiFab ghost extents must be non-negative");
  }

  std::size_t local_offset(std::size_t global_box) const {
    for (std::size_t local = 0; local < local_global_indices_.size(); ++local)
      if (local_global_indices_[local] == global_box)
        return local;
    throw std::out_of_range("nd_proof::MultiFab global box is not local to this rank");
  }

  void reset_moved_from() noexcept {
    layout_ = BoxArray<Dim>{};
    distribution_ = Distribution<Dim>{};
    local_rank_ = rank_type{};
    ncomp_ = 0;
    ghosts_ = Extent<Dim>{};
    local_global_indices_.clear();
    fabs_.clear();
  }

  void move_from(MultiFab&& other) noexcept {
    layout_ = std::move(other.layout_);
    distribution_ = std::move(other.distribution_);
    local_rank_ = other.local_rank_;
    ncomp_ = other.ncomp_;
    ghosts_ = other.ghosts_;
    local_global_indices_ = std::move(other.local_global_indices_);
    fabs_ = std::move(other.fabs_);
    other.reset_moved_from();
  }

  BoxArray<Dim> layout_{};
  Distribution<Dim> distribution_{};
  rank_type local_rank_{};
  int ncomp_ = 0;
  Extent<Dim> ghosts_{};
  std::vector<std::size_t> local_global_indices_{};
  std::vector<fab_type> fabs_{};
};

}  // namespace pops::mesh::nd_proof
