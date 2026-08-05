#pragma once

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pops {

/// One rank-local, valid-cell native array piece for scientific output.
///
/// Values are compact and component-major inside the inclusive index-space box, with native axis
/// zero contiguous. Unlike checkpoint accessors this contract never allocates a global-sized buffer
/// or represents cells outside a patch. ``replicated`` is explicit because AMR level zero may
/// intentionally exist in full on every rank; collective writers use that fact to select one
/// canonical contributor while per-rank writers retain the exact local view.
template <int Dim>
struct OutputPiece {
  static_assert(Dim >= 1 && Dim <= 3, "scientific output only supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;
  int level = 0;
  Box<Dim> box{};
  int global_box_index = -1;
  int owner_rank = -1;
  bool replicated = false;
  int ncomp = 0;
  std::vector<double> values;
};

/// Copy the valid cells of every locally allocated ranked fab into exact output pieces.
template <int Dim, class MemorySpace>
std::vector<OutputPiece<Dim>> output_local_pieces(const MultiFab<Dim, MemorySpace>& source,
                                                  int level, bool replicated) {
  if (level < 0)
    throw std::out_of_range("output_local_pieces level must be nonnegative");
  if (source.ncomp() < 1)
    throw std::runtime_error("output_local_pieces requires at least one component");

  std::vector<OutputPiece<Dim>> result;
  result.reserve(source.local_size());
  for (std::size_t local = 0; local < source.local_size(); ++local) {
    const std::size_t global = source.global_index(local);
    const Box<Dim>& valid = source.box(local);
    const std::size_t cells = static_cast<std::size_t>(valid.numPts());
    const int ncomp = source.ncomp();
    OutputPiece<Dim> piece;
    piece.level = level;
    piece.box = valid;
    piece.global_box_index = static_cast<int>(global);
    piece.owner_rank = replicated ? my_rank()
                                  : static_cast<int>(source.rank_space().linear_rank(
                                        source.distribution().owner(global)));
    piece.replicated = replicated;
    piece.ncomp = ncomp;
    piece.values.resize(static_cast<std::size_t>(ncomp) * cells);
    const Fab<Dim, MemorySpace>& fab = source.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const FieldView<const Real, Dim> view = fab.view();
    for (int component = 0; component < ncomp; ++component)
      for (std::size_t linear = 0; linear < cells; ++linear) {
        std::size_t remainder = linear;
        std::int64_t storage = static_cast<std::int64_t>(component) * view.component_stride;
        for (int axis = 0; axis < Dim; ++axis) {
          const std::size_t extent = static_cast<std::size_t>(valid.length(axis));
          const int coordinate = valid.lo[axis] + static_cast<int>(remainder % extent);
          remainder /= extent;
          storage += static_cast<std::int64_t>(coordinate - view.origin[axis]) * view.strides[axis];
        }
        piece.values[static_cast<std::size_t>(component) * cells + linear] =
            static_cast<double>(host(static_cast<std::size_t>(storage)));
      }
    result.push_back(std::move(piece));
  }
  return result;
}

}  // namespace pops
