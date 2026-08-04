/// @file
/// @brief Versioned rank-generic spatial authority for checkpoint/restart preparation.

#pragma once

#include <pops/amr/refinement_ratio.hpp>
#include <pops/mesh/index/extent.hpp>
#include <pops/runtime/config/generated_release_contract.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::runtime::checkpoint {

inline constexpr int kSpatialContractSchemaVersion =
    ::pops::release_contract::kCheckpointSpatialSchemaVersion;

/// Data-only representation authenticated by the outer checkpoint envelope.
struct EncodedSpatialContract {
  int schema_version = kSpatialContractSchemaVersion;
  int dimension = 0;
  std::vector<std::int64_t> shape;
  std::vector<double> lower;
  std::vector<double> upper;
  std::vector<unsigned char> periodicity;
  std::vector<std::vector<int>> refinement_ratios;
  std::string native_layout_identity;
  std::string spatial_identity;
};

template <int Dim>
struct SpatialContract {
  static_assert(Dim >= 1 && Dim <= 3,
                "checkpoint spatial contracts support dimensions 1, 2, and 3");
  static constexpr int dimension = Dim;

  Extent<Dim> shape{};
  std::array<double, Dim> lower{};
  std::array<double, Dim> upper{};
  std::array<bool, Dim> periodicity{};
  std::vector<amr::RefinementRatio<Dim>> refinement_ratios;
  std::string native_layout_identity;
  std::string spatial_identity;

  void validate() const {
    (void)cell_count();
    for (int axis = 0; axis < Dim; ++axis) {
      const auto index = static_cast<std::size_t>(axis);
      if (!std::isfinite(lower[index]) || !std::isfinite(upper[index]) ||
          !(lower[index] < upper[index]))
        throw std::invalid_argument(
            "checkpoint spatial upper bounds must exceed lower bounds on every axis");
    }
    for (const auto& ratio : refinement_ratios) {
      bool refines = false;
      for (int axis = 0; axis < Dim; ++axis)
        if (ratio[axis] < 1)
          throw std::invalid_argument(
              "checkpoint refinement-ratio components must be strictly positive");
        else if (ratio[axis] > 1)
          refines = true;
      if (!refines)
        throw std::invalid_argument("checkpoint refinement ratios must refine at least one axis");
    }
    if (native_layout_identity.empty() || spatial_identity.empty())
      throw std::invalid_argument("checkpoint spatial identities must be non-empty");
  }

  std::int64_t cell_count() const {
    std::int64_t count = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      const std::int64_t extent = shape[axis];
      if (extent < 1)
        throw std::invalid_argument(
            "checkpoint spatial shape must be strictly positive on every axis");
      if (count > std::numeric_limits<std::int64_t>::max() / extent)
        throw std::overflow_error("checkpoint spatial cell count exceeds int64_t");
      count *= extent;
    }
    return count;
  }

  Extent<Dim> shape_at_level(std::size_t level) const {
    validate();
    if (level > refinement_ratios.size())
      throw std::out_of_range("checkpoint level lies outside its refinement-ratio envelope");
    Extent<Dim> result = shape;
    for (std::size_t transition = 0; transition < level; ++transition) {
      const auto& ratio = refinement_ratios[transition];
      for (int axis = 0; axis < Dim; ++axis) {
        if (result[axis] > std::numeric_limits<std::int64_t>::max() / ratio[axis])
          throw std::overflow_error("checkpoint refined shape exceeds int64_t");
        result[axis] *= ratio[axis];
      }
    }
    return result;
  }

  bool operator==(const SpatialContract& other) const {
    return shape == other.shape && lower == other.lower && upper == other.upper &&
           periodicity == other.periodicity && refinement_ratios == other.refinement_ratios &&
           native_layout_identity == other.native_layout_identity &&
           spatial_identity == other.spatial_identity;
  }
};

template <int Dim>
EncodedSpatialContract encode_spatial_contract(const SpatialContract<Dim>& contract) {
  contract.validate();
  EncodedSpatialContract encoded;
  encoded.dimension = Dim;
  encoded.shape.reserve(Dim);
  encoded.lower.reserve(Dim);
  encoded.upper.reserve(Dim);
  encoded.periodicity.reserve(Dim);
  for (int axis = 0; axis < Dim; ++axis) {
    const auto index = static_cast<std::size_t>(axis);
    encoded.shape.push_back(contract.shape[axis]);
    encoded.lower.push_back(contract.lower[index]);
    encoded.upper.push_back(contract.upper[index]);
    encoded.periodicity.push_back(contract.periodicity[index] ? 1U : 0U);
  }
  encoded.refinement_ratios.reserve(contract.refinement_ratios.size());
  for (const auto& ratio : contract.refinement_ratios) {
    std::vector<int> row;
    row.reserve(Dim);
    for (int axis = 0; axis < Dim; ++axis)
      row.push_back(ratio[axis]);
    encoded.refinement_ratios.push_back(std::move(row));
  }
  encoded.native_layout_identity = contract.native_layout_identity;
  encoded.spatial_identity = contract.spatial_identity;
  return encoded;
}

template <int Dim>
SpatialContract<Dim> decode_spatial_contract(const EncodedSpatialContract& encoded) {
  // The schema and dimension are checked before constructing any rank-specialized state or
  // refinement-ratio storage. Restart callers use this preparation result before allocation.
  if (encoded.schema_version != kSpatialContractSchemaVersion)
    throw std::invalid_argument("checkpoint spatial schema version is unsupported");
  if (encoded.dimension != Dim)
    throw std::invalid_argument("checkpoint dimension does not match the native specialization");
  if (encoded.shape.size() != Dim || encoded.lower.size() != Dim || encoded.upper.size() != Dim ||
      encoded.periodicity.size() != Dim)
    throw std::invalid_argument("checkpoint spatial vectors must have exactly Dim components");

  SpatialContract<Dim> result;
  for (int axis = 0; axis < Dim; ++axis) {
    const auto index = static_cast<std::size_t>(axis);
    result.shape[axis] = encoded.shape[index];
    result.lower[index] = encoded.lower[index];
    result.upper[index] = encoded.upper[index];
    if (encoded.periodicity[index] > 1U)
      throw std::invalid_argument("checkpoint periodicity must contain exact boolean values");
    result.periodicity[index] = encoded.periodicity[index] != 0U;
  }
  result.refinement_ratios.reserve(encoded.refinement_ratios.size());
  for (const auto& row : encoded.refinement_ratios) {
    if (row.size() != Dim)
      throw std::invalid_argument(
          "checkpoint refinement-ratio vectors must have exactly Dim components");
    std::array<int, Dim> ratio{};
    for (int axis = 0; axis < Dim; ++axis)
      ratio[static_cast<std::size_t>(axis)] = row[static_cast<std::size_t>(axis)];
    result.refinement_ratios.emplace_back(ratio);
  }
  result.native_layout_identity = encoded.native_layout_identity;
  result.spatial_identity = encoded.spatial_identity;
  result.validate();
  return result;
}

template <int Dim>
SpatialContract<Dim> prepare_spatial_restart(const EncodedSpatialContract& encoded,
                                             const SpatialContract<Dim>& current) {
  current.validate();
  auto recorded = decode_spatial_contract<Dim>(encoded);
  if (!(recorded == current))
    throw std::invalid_argument(
        "checkpoint spatial authority does not match the installed native specialization");
  return recorded;
}

}  // namespace pops::runtime::checkpoint
