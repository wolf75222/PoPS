#pragma once

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/rank_space.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/load_balance.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pops::bench {

template <class Ranked, class Value>
Ranked filled_ranked(Value value) {
  Ranked result{};
  for (int axis = 0; axis < Ranked::rank; ++axis)
    result[axis] = value;
  return result;
}

inline std::size_t checked_product(std::size_t left, std::size_t right, const char* message) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(message);
  return left * right;
}

template <int Dim>
mesh::RankSpace<Dim> benchmark_rank_space() {
  Extent<Dim> extents = filled_ranked<Extent<Dim>>(1);
  extents[0] = n_ranks();
  return mesh::RankSpace<Dim>(Index<Dim>{}, extents);
}

template <int Dim>
Index<Dim> benchmark_local_rank(const mesh::RankSpace<Dim>& ranks) {
  return ranks.coordinate(static_cast<std::size_t>(my_rank()));
}

template <int Dim>
mesh::BoxArrayValidationBudget layout_validation_budget(const mesh::BoxArray<Dim>& layout) {
  const std::size_t pairs = layout.size() < 2
                                ? 0
                                : checked_product(layout.size(), layout.size() - 1,
                                                  "benchmark layout-pair budget overflow") /
                                      2;
  return {layout.size(), pairs};
}

template <int Dim>
parallel::LoadBalancePreparationBudget load_balance_budget(const mesh::BoxArray<Dim>& layout,
                                                           const Box<Dim>& domain) {
  return {layout.size(), static_cast<std::size_t>(n_ranks()), domain.numPts()};
}

template <int Dim>
Index<Dim> index_from_ordinal(const Box<Dim>& box, std::int64_t ordinal) {
  Index<Dim> index{};
  for (int axis = 0; axis < Dim; ++axis) {
    index[axis] = box.lo[axis] + static_cast<int>(ordinal % box.length(axis));
    ordinal /= box.length(axis);
  }
  return index;
}

template <int Dim>
std::size_t host_offset(const Box<Dim>& storage, const Index<Dim>& index, int component = 0) {
  std::int64_t cell = 0;
  std::int64_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    cell += static_cast<std::int64_t>(index[axis] - storage.lo[axis]) * stride;
    stride *= storage.length(axis);
  }
  return static_cast<std::size_t>(component * storage.numPts() + cell);
}

}  // namespace pops::bench
