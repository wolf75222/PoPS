/// @file
/// @brief Private exact local neighbor enumeration over ND box layouts.
///
/// Non-installed proof scaffolding.  It handles only ordinary axis translations; mapped periodic
/// identifications remain an affine-topology concern until a dedicated mapped job representation
/// exists.

#pragma once

#include <pops/mesh/nd_proof/box_hash.hpp>
#include <pops/mesh/nd_proof/periodicity.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pops::mesh::nd_proof {

/// One local copy candidate.  ``destination_region`` is in destination coordinates and source
/// coordinates are ``destination + source_from_destination_translation``.
template <int Dim>
struct LocalNeighborJob {
  std::size_t source_box = 0;
  std::size_t destination_box = 0;
  Box<Dim> destination_region{};
  std::array<std::int64_t, Dim> source_from_destination_translation{};

  bool operator==(const LocalNeighborJob&) const = default;
};

/// Explicit caps for the image catalogue and the result job vector.  Hash work is controlled by
/// the separate caller-supplied BoxHashBudget.
struct LocalNeighborWorkBudget {
  std::size_t images;
  std::size_t jobs;
  BoxArrayValidationBudget tiling;
  BoxHashQueryBudget queries;
};

namespace local_neighbors_detail {

template <int Dim>
std::array<std::int64_t, Dim> inverse_translation(const AxisTranslationImage<Dim>& image) {
  std::array<std::int64_t, Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = periodicity_detail::checked_negate(
        image.translation[axis], "nd_proof local neighbor inverse translation overflows int64_t");
  return result;
}

}  // namespace local_neighbors_detail

/// Enumerates zero-shift seams and ordinary axis-translation images.  The output order is
/// destination-box order, then enumerate_axis_translation_images order, then sorted source index.
/// The zero-shift self job is omitted; nonzero periodic self images are retained.
template <int Dim>
std::vector<LocalNeighborJob<Dim>> enumerate_local_translation_neighbors(
    const BoxArray<Dim>& boxes, const Box<Dim>& domain, const Extent<Dim>& destination_ghosts,
    const PeriodicTopology<Dim>& topology,
    const std::array<int, static_cast<std::size_t>(Dim)>& hash_bin_extent,
    BoxHashBudget hash_budget, LocalNeighborWorkBudget work_budget) {
  if (domain.empty())
    throw std::invalid_argument("nd_proof local neighbors require a non-empty domain");
  if (!boxes.tiles_exactly(domain, work_budget.tiling))
    throw std::invalid_argument("nd_proof local neighbors require an exact domain tiling");
  topology.validate(domain);
  if (!topology.is_axis_translation_only())
    throw std::invalid_argument(
        "nd_proof local translation neighbors do not support mapped periodic identifications");

  const std::vector<AxisTranslationImage<Dim>> images = enumerate_axis_translation_images(
      domain, destination_ghosts, topology, AxisTranslationImageBudget{work_budget.images});
  const BoxHash<Dim> hash(boxes, hash_bin_extent, hash_budget);
  BoxHashQueryBudget remaining_queries = work_budget.queries;

  std::vector<LocalNeighborJob<Dim>> jobs;
  for (std::size_t destination = 0; destination < boxes.size(); ++destination) {
    const Box<Dim> destination_grown =
        periodicity_detail::grow_box<Dim>(boxes[destination], destination_ghosts);
    for (const AxisTranslationImage<Dim>& image : images) {
      const std::array<std::int64_t, Dim> source_from_destination =
          local_neighbors_detail::inverse_translation(image);
      const Box<Dim> source_query = periodicity_detail::translate_box<Dim>(
          destination_grown, source_from_destination,
          "nd_proof local neighbor query translation overflow");
      const std::vector<std::size_t> candidates = hash.query(source_query, &remaining_queries);
      for (const std::size_t source : candidates) {
        if (image.is_zero() && source == destination)
          continue;
        const Box<Dim> source_image = image.apply(boxes[source]);
        const Box<Dim> destination_region = destination_grown.intersect(source_image);
        if (destination_region.empty())
          continue;
        if (jobs.size() >= work_budget.jobs || jobs.size() >= jobs.max_size())
          throw std::length_error("nd_proof local neighbor jobs exceed their explicit budget");
        jobs.push_back(LocalNeighborJob<Dim>{source, destination, destination_region,
                                             source_from_destination});
      }
    }
  }
  return jobs;
}

}  // namespace pops::mesh::nd_proof
