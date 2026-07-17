#pragma once

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/patch_box.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pops {

/// One rank-local, valid-cell native array piece for scientific output.
///
/// Values are compact and component-major inside the inclusive index-space box:
/// ``c * ny * nx + (j - jlo) * nx + (i - ilo)``.  Unlike checkpoint accessors this contract never
/// allocates a global-sized buffer or represents cells outside a patch.  ``replicated`` is explicit
/// because AMR level zero may intentionally exist in full on every rank; collective writers use that
/// fact to select one canonical contributor while per-rank writers retain the exact local view.
struct OutputPiece {
  PatchBox box{};
  int global_box_index = -1;
  int owner_rank = -1;
  bool replicated = false;
  int ncomp = 0;
  std::vector<double> values;
};

/// Copy the valid cells of every locally allocated fab into exact compact output pieces.
inline std::vector<OutputPiece> output_local_pieces(const MultiFab& source, int level,
                                                    bool replicated) {
  if (level < 0)
    throw std::out_of_range("output_local_pieces level must be nonnegative");
  if (source.ncomp() < 1)
    throw std::runtime_error("output_local_pieces requires at least one component");

  source.sync_host();
  std::vector<OutputPiece> result;
  result.reserve(static_cast<std::size_t>(source.local_size()));
  for (int local = 0; local < source.local_size(); ++local) {
    const int global = source.global_index(local);
    const Box2D& valid = source.box(local);
    const int nx = valid.nx();
    const int ny = valid.ny();
    const int ncomp = source.ncomp();
    OutputPiece piece;
    piece.box = PatchBox{level, valid.lo[0], valid.lo[1], valid.hi[0], valid.hi[1]};
    piece.global_box_index = global;
    piece.owner_rank = replicated ? my_rank() : source.dmap()[global];
    piece.replicated = replicated;
    piece.ncomp = ncomp;
    piece.values.resize(static_cast<std::size_t>(ncomp) * static_cast<std::size_t>(ny) *
                        static_cast<std::size_t>(nx));
    const ConstArray4 values = source.fab(local).const_array();
    for (int c = 0; c < ncomp; ++c)
      for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
        for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
          piece.values[static_cast<std::size_t>(c) * static_cast<std::size_t>(ny) *
                           static_cast<std::size_t>(nx) +
                       static_cast<std::size_t>(j - valid.lo[1]) * static_cast<std::size_t>(nx) +
                       static_cast<std::size_t>(i - valid.lo[0])] =
              static_cast<double>(values(i, j, c));
    result.push_back(std::move(piece));
  }
  return result;
}

}  // namespace pops
