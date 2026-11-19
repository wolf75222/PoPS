/// @file
/// @brief Cold, box-wise resolution of oriented Cartesian halo images.
#pragma once

#include <pops/mesh/boundary/affine_index_transform.hpp>
#include <pops/mesh/topology/boundary_topology.hpp>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace pops::identified_halo_detail {

template <int Dim>
using Map = mesh::AffineIndexTransform<Dim>;

template <int Dim>
struct Image {
  Box<Dim> destination;
  Map<Dim> read_from_destination;
};

template <int Dim>
std::array<Map<Dim>, 2 * Dim> prepare_crossings(const BoundaryTopology<Dim>& topology,
                                                const Box<Dim>& domain) {
  std::array<Map<Dim>, 2 * Dim> result{};
  for (const auto& face : topology.faces()) {
    const int axis = face.face.axis;
    if (face.kind == BoundaryFaceKind::periodic) {
      std::array<std::int64_t, Dim> offsets{};
      offsets[axis] = domain.length(axis) * (face.face.side == BoundarySide::lower ? 1 : -1);
      result[face.face.ordinal()] = Map<Dim>{{}, offsets};
      continue;
    }
    if (!face.identification)
      throw std::invalid_argument("identified halo requires an assigned crossing at each face");
    const auto& row = *face.identification;
    const int from_axis = row.source_face / 2;
    const int to_axis = row.target_face / 2;
    for (int j = 0; j < Dim; ++j) {
      const int k = row.permutation[j];
      if (domain.length(j) != domain.length(k))
        throw std::invalid_argument("identified halo has mismatched face extents");
      const auto sign = row.signs[j];
      const std::int64_t offset = row.target_offsets[k];
      if (j == from_axis) {
        const std::int64_t interior = row.source_face % 2 ? domain.hi[j] : domain.lo[j];
        const std::int64_t exterior = row.target_face % 2
                                          ? static_cast<std::int64_t>(domain.hi[k]) + 1
                                          : static_cast<std::int64_t>(domain.lo[k]) - 1;
        if (k != to_axis || sign * interior + offset != exterior)
          throw std::invalid_argument("identified halo normal image differs from its exact face");
      } else {
        const std::int64_t first = sign > 0 ? domain.lo[j] : domain.hi[j];
        const std::int64_t shift = sign * first + offset - domain.lo[k];
        const bool wraps = topology.is_periodic({k, BoundarySide::lower});
        if (shift < 0 || shift >= domain.length(k) || (shift && !wraps))
          throw std::invalid_argument(
              "identified halo tangential image lacks an exact periodic wrap");
      }
    }
    const Map<Dim> forward{mesh::SignedPermutation<Dim>{row.permutation, row.signs}, [&] {
                             std::array<std::int64_t, Dim> offsets{};
                             for (int k = 0; k < Dim; ++k)
                               offsets[k] = row.target_offsets[k];
                             return offsets;
                           }()};
    result[face.face.ordinal()] =
        face.face.ordinal() == row.source_face ? forward : forward.inverse();
  }
  return result;
}

template <int Dim>
struct ExteriorPiece {
  Box<Dim> box;
  std::array<int, Dim> faces{};
  int count = 0;
};

template <int Dim>
std::vector<ExteriorPiece<Dim>> split_exterior(const Box<Dim>& box, const Box<Dim>& domain) {
  std::vector<ExteriorPiece<Dim>> pieces{{box}};
  for (int axis = 0; axis < Dim; ++axis) {
    std::vector<ExteriorPiece<Dim>> cut;
    for (auto piece : pieces) {
      if (piece.box.lo[axis] < domain.lo[axis]) {
        auto lower = piece;
        lower.box.hi[axis] = std::min(piece.box.hi[axis], domain.lo[axis] - 1);
        lower.faces[lower.count++] = 2 * axis;
        cut.push_back(lower);
        piece.box.lo[axis] = domain.lo[axis];
      }
      if (piece.box.empty())
        continue;
      if (piece.box.hi[axis] > domain.hi[axis]) {
        auto upper = piece;
        upper.box.lo[axis] = std::max(piece.box.lo[axis], domain.hi[axis] + 1);
        upper.faces[upper.count++] = 2 * axis + 1;
        cut.push_back(upper);
        piece.box.hi[axis] = domain.hi[axis];
      }
      if (!piece.box.empty())
        cut.push_back(piece);
    }
    pieces = std::move(cut);
  }
  return pieces;
}

/// Each intermediate image is cut at all physical index bounds before another crossing.
/// Every legal order is checked on its common destination sub-box against the first order.
template <int Dim>
std::vector<Image<Dim>> resolve(const Box<Dim>& destination, const Box<Dim>& domain,
                                const std::array<Map<Dim>, 2 * Dim>& crossings,
                                const Map<Dim>& read_from_destination, int crossings_used,
                                std::size_t& work) {
  if (work == 0)
    throw std::length_error("identified halo affine image budget exceeded");
  --work;
  std::vector<Image<Dim>> result;
  const auto inverse = read_from_destination.inverse();
  for (const auto& piece : split_exterior(read_from_destination.apply(destination), domain)) {
    const auto region = inverse.apply(piece.box);
    if (piece.count == 0) {
      result.push_back({region, read_from_destination});
      continue;
    }
    // At depth no greater than the corresponding domain extent, a valid chain can use at
    // most one crossing and one tangential wrap per axis. Permit one extra wrap per axis
    // during checking, then refuse a cycle or an inaccessible image before any publication.
    if (crossings_used == 3 * Dim)
      throw std::invalid_argument("identified halo corner chain cannot reach an owned cell");
    auto selected = resolve<Dim>(region, domain, crossings,
                                 read_from_destination.compose(crossings[piece.faces[0]]),
                                 crossings_used + 1, work);
    for (int alternative = 1; alternative < piece.count; ++alternative) {
      auto other = resolve<Dim>(region, domain, crossings,
                                read_from_destination.compose(crossings[piece.faces[alternative]]),
                                crossings_used + 1, work);
      for (const auto& a : selected)
        for (const auto& b : other)
          if (!a.destination.intersect(b.destination).empty() &&
              a.read_from_destination != b.read_from_destination)
            throw std::invalid_argument(
                "identified halo corner paths differ in source or orientation at faces " +
                std::to_string(piece.faces[0]) + " and " +
                std::to_string(piece.faces[alternative]));
    }
    result.insert(result.end(), selected.begin(), selected.end());
  }
  return result;
}

}  // namespace pops::identified_halo_detail
