#pragma once

/// @file
/// @brief Ownership-aware native reductions for distributed and physically replicated fields.

#include <pops/mesh/layout/field_distribution.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace pops::detail {

enum class FieldDistributionReductionStatus {
  Success,
  NonfiniteReplica,
  InconsistentReplica,
};

inline std::size_t field_distribution_consensus_storage_size(std::size_t value_count) {
  if (value_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) / 2u)
    throw std::length_error(
        "field distribution consensus exceeds the native MPI collective count capacity");
  return value_count * 2u;
}

/// Establish one deterministic contribution for a physically replicated vector. Every rank owns
/// a complete copy, so SUM would count it once per rank and SUM/n_ranks would perturb rounding.
/// One MAX over [value,-value] obtains both extrema, proves consensus, and selects the exact maximum
/// as the representative. The caller owns scratch so prepared hot paths allocate nothing.
inline FieldDistributionReductionStatus reduce_replicated_field_values_inplace(
    double* values, std::size_t value_count, double* consensus, std::size_t consensus_count) {
  if (value_count == 0)
    return FieldDistributionReductionStatus::Success;
  const std::size_t required = field_distribution_consensus_storage_size(value_count);
  if (values == nullptr || consensus == nullptr || consensus_count < required)
    throw std::logic_error("replicated field reduction storage is incoherent");

  for (std::size_t index = 0; index < value_count; ++index) {
    const double value = values[index];
    if (!std::isfinite(value)) {
      consensus[index] = std::numeric_limits<double>::infinity();
      consensus[value_count + index] = std::numeric_limits<double>::infinity();
    } else {
      consensus[index] = value;
      consensus[value_count + index] = -value;
    }
  }
  all_reduce_max_inplace(consensus, static_cast<int>(required));

  bool inconsistent = false;
  for (std::size_t index = 0; index < value_count; ++index) {
    const double maximum = consensus[index];
    const double minimum = -consensus[value_count + index];
    if (!std::isfinite(maximum) || !std::isfinite(minimum))
      return FieldDistributionReductionStatus::NonfiniteReplica;
    const double scale = std::max({1.0, std::abs(minimum), std::abs(maximum)});
    const double tolerance = 128.0 * std::numeric_limits<double>::epsilon() * scale;
    inconsistent = inconsistent || maximum - minimum > tolerance;
    values[index] = maximum;
  }
  return inconsistent ? FieldDistributionReductionStatus::InconsistentReplica
                      : FieldDistributionReductionStatus::Success;
}

inline FieldDistributionReductionStatus reduce_field_values_inplace(FieldDistribution distribution,
                                                                    double* values,
                                                                    std::size_t value_count,
                                                                    double* consensus,
                                                                    std::size_t consensus_count) {
  if (!field_distribution_is_valid(distribution))
    throw std::invalid_argument("field reduction received invalid distribution");
  if (value_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("field reduction exceeds the native MPI collective count capacity");
  if (value_count != 0 && values == nullptr)
    throw std::logic_error("field reduction values are missing");
  if (distribution == FieldDistribution::Distributed) {
    all_reduce_sum_inplace(values, static_cast<int>(value_count));
    return FieldDistributionReductionStatus::Success;
  }
  return reduce_replicated_field_values_inplace(values, value_count, consensus, consensus_count);
}

/// Combine a hierarchy containing both disjoint distributed contributions and complete replicated
/// contributions. After replica consensus, rank zero injects that physical contribution exactly
/// once into the distributed SUM. This preserves serial arithmetic and is independent of AMR or
/// any concrete hierarchy provider.
inline FieldDistributionReductionStatus reduce_mixed_field_values_inplace(
    double* distributed, double* replicated, std::size_t value_count, bool has_distributed,
    bool has_replicated, double* consensus, std::size_t consensus_count) {
  if (value_count != 0 && (distributed == nullptr || replicated == nullptr))
    throw std::logic_error("mixed field reduction storage is incoherent");
  if (has_replicated) {
    const FieldDistributionReductionStatus status =
        reduce_replicated_field_values_inplace(replicated, value_count, consensus, consensus_count);
    if (status != FieldDistributionReductionStatus::Success)
      return status;
  }
  if (has_distributed) {
    if (has_replicated && my_rank() == 0)
      for (std::size_t index = 0; index < value_count; ++index)
        distributed[index] += replicated[index];
    return reduce_field_values_inplace(FieldDistribution::Distributed, distributed, value_count,
                                       nullptr, 0);
  }
  if (has_replicated && value_count != 0)
    std::copy_n(replicated, value_count, distributed);
  return FieldDistributionReductionStatus::Success;
}

inline void require_consistent_field_distribution_reduction(FieldDistributionReductionStatus status,
                                                            const char* quantity) {
  if (status == FieldDistributionReductionStatus::NonfiniteReplica)
    throw std::runtime_error(std::string(quantity) +
                             " contains a non-finite replicated contribution");
  if (status == FieldDistributionReductionStatus::InconsistentReplica)
    throw std::runtime_error(std::string(quantity) +
                             " disagrees between physically replicated rank copies");
}

}  // namespace pops::detail
