/// @file
/// @brief Ranked spatial-layout coupling across AMR block runtimes.

#pragma once

#include <pops/runtime/amr/amr_runtime.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pops::coupling::system {

template <int Dim>
struct AmrHierarchyLayout {
  std::vector<::pops::amr::hierarchy::LevelLayoutIdentity<Dim>> levels{};
  Index<Dim> local_rank{};

  bool operator==(const AmrHierarchyLayout&) const = default;
};

template <int Dim, class MemorySpace>
AmrHierarchyLayout<Dim> extract_hierarchy_layout(
    const ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime) {
  AmrHierarchyLayout<Dim> result;
  result.levels.reserve(runtime.hierarchy().num_levels());
  for (std::size_t level = 0; level < runtime.hierarchy().num_levels(); ++level)
    result.levels.push_back(runtime.hierarchy().layout(level).exact_identity());
  result.local_rank = runtime.hierarchy().state(0).local_rank();
  return result;
}

template <int Dim>
bool same_level_layout(const ::pops::amr::hierarchy::LevelLayout<Dim>& left,
                       const ::pops::amr::hierarchy::LevelLayout<Dim>& right) {
  return left.exact_identity() == right.exact_identity();
}

/// A system coupler authenticates shared AMR topology only. Physics assembly, time integration,
/// configuration, and Python binding remain outside this spatial boundary.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class AmrSystemCoupler {
 public:
  using runtime_type = ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>;

  explicit AmrSystemCoupler(std::vector<runtime_type*> blocks) : blocks_(std::move(blocks)) {
    if (blocks_.empty())
      throw std::invalid_argument("AMR system coupler requires at least one block runtime");
    for (runtime_type* block : blocks_)
      if (block == nullptr)
        throw std::invalid_argument("AMR system coupler received a null block runtime");
    refresh_layout_or_throw();
  }

  static constexpr int dimension = Dim;

  std::size_t block_count() const noexcept { return blocks_.size(); }

  runtime_type& block(std::size_t index) const {
    if (index >= blocks_.size())
      throw std::out_of_range("AMR system block lies outside the coupled set");
    return *blocks_[index];
  }

  const AmrHierarchyLayout<Dim>& layout_identity() const {
    require_layout_current();
    return layout_;
  }

  void refresh_layout_or_throw() {
    const AmrHierarchyLayout<Dim> candidate = extract_hierarchy_layout(*blocks_.front());
    for (std::size_t block_index = 1; block_index < blocks_.size(); ++block_index)
      if (extract_hierarchy_layout(*blocks_[block_index]) != candidate)
        throw std::invalid_argument(
            "AMR system blocks must share ordered patches, spatial ownership, and local rank");
    layout_ = candidate;
  }

  void require_layout_current() const {
    for (runtime_type* current : blocks_)
      if (extract_hierarchy_layout(*current) != layout_)
        throw std::invalid_argument("AMR system layout changed without an authenticated refresh");
  }

 private:
  std::vector<runtime_type*> blocks_;
  AmrHierarchyLayout<Dim> layout_{};
};

}  // namespace pops::coupling::system
