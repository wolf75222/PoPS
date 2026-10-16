#pragma once

#include <cmath>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <pops/runtime/program/program_runtime_state.hpp>

namespace pops::runtime::program {

/// A retained interval sample and a native point value have distinct authorities.
/// A point has no invented outgoing interval. Its event contract is produced by
/// the owning native transaction, never by the authored interpolation callback.
struct TemporalSampleSource {
  enum class Kind { Retained, NativePoint };
  Kind kind = Kind::Retained;
  HistorySampleIdentity retained;
  double time = 0.;
  double interval = 0.;
  bool available = false;
  std::string point_authority;

  static TemporalSampleSource history(const HistorySampleIdentity& sample, bool initialized,
                                      Real outgoing_dt) {
    validate_history_sample_provenance(sample, initialized, outgoing_dt);
    TemporalSampleSource result;
    result.retained = sample;
    result.available = initialized && sample.authenticated() &&
                       sample.kind != HistorySampleKind::RegisteredZeroStart;
    if (result.available) {
      result.time = std::bit_cast<double>(sample.start_bits);
      result.interval = static_cast<double>(outgoing_dt);
    }
    result.validate();
    return result;
  }

  static TemporalSampleSource point(double time, std::string authority) {
    TemporalSampleSource result;
    result.kind = Kind::NativePoint;
    result.time = time;
    result.available = true;
    result.point_authority = std::move(authority);
    result.validate();
    return result;
  }

  void validate() const {
    if (!std::isfinite(time) || !std::isfinite(interval))
      throw std::invalid_argument("temporal source has a nonfinite coordinate");
    if (kind == Kind::NativePoint) {
      if (!available || std::bit_cast<std::uint64_t>(interval) != 0 || point_authority.empty())
        throw std::invalid_argument("native temporal point requires its exact event authority");
    } else if (kind == Kind::Retained) {
      retained.validate();
      if (!point_authority.empty() ||
          (available &&
           (!retained.authenticated() || retained.kind == HistorySampleKind::RegisteredZeroStart ||
            !(interval > 0.) || time != std::bit_cast<double>(retained.start_bits) ||
            interval != std::bit_cast<double>(retained.interval_bits))) ||
          (!available && (std::bit_cast<std::uint64_t>(time) != 0 ||
                          std::bit_cast<std::uint64_t>(interval) != 0)))
        throw std::invalid_argument("retained temporal source has inconsistent availability");
    } else {
      throw std::invalid_argument("temporal source has an invalid authority kind");
    }
  }

  void append_contract(ExactContractBuilder& contract) const {
    validate();
    contract.scalar(kind).scalar(time).scalar(interval).scalar(available);
    if (kind == Kind::Retained)
      retained.append_contract(contract);
    else
      contract.bytes(point_authority);
  }
};

/// Exact source selection shared by interval-history reconstruction and point
/// evaluation. The caller adds its own typed target coordinates to `operation`.
inline std::string temporal_reconstruction_contract(std::span<const TemporalSampleSource> sources,
                                                    std::span<const Real> weights,
                                                    std::string_view operation) {
  if (sources.size() != weights.size() || operation.empty())
    throw std::invalid_argument("temporal reconstruction has incomplete source authority");
  ExactContractBuilder contract;
  contract.text("pops.temporal-source-reconstruction.v1").bytes(operation).scalar(sources.size());
  for (std::size_t i = 0; i < sources.size(); ++i) {
    sources[i].append_contract(contract);
    if (!std::isfinite(weights[i]) || (weights[i] != Real(0) && !sources[i].available))
      throw std::invalid_argument("temporal reconstruction uses a nonfinite or unavailable source");
    contract.scalar(static_cast<double>(weights[i]));
  }
  return std::move(contract).release();
}

inline HistorySampleIdentity reconstructed_temporal_sample(
    std::span<const TemporalSampleSource> sources, const HistoryTransferRow& row,
    std::string_view operation, std::string_view ring, std::uint64_t topology,
    std::uint64_t generation) {
  if (ring.empty() || !std::isfinite(row.time) || !std::isfinite(row.interval) ||
      !(row.interval > 0.))
    throw std::invalid_argument(
        "temporal history target requires a finite positive outgoing interval");
  validate_history_transfer_weights(row);
  if (sources.size() != row.weights.size())
    throw std::invalid_argument("temporal history target source width differs from matrix");
  for (std::size_t i = 0; i < sources.size(); ++i)
    if (!row.exact_weights.empty() && !row.exact_weights[i].zero() && !sources[i].available)
      throw std::invalid_argument("temporal reconstruction uses an unavailable exact source");
  ExactContractBuilder target;
  target.text("pops.temporal-history-reconstruction.v1")
      .bytes(operation)
      .text(ring)
      .scalar(topology)
      .scalar(generation)
      .scalar(row.time)
      .scalar(row.interval);
  append_history_transfer_weight_contract(target, row);
  const auto contract =
      temporal_reconstruction_contract(sources, row.weights, std::move(target).release());
  const auto digest =
      identity::sha256_hex(std::vector<std::uint8_t>(contract.begin(), contract.end()));
  HistorySampleIdentity result{std::bit_cast<std::uint64_t>(static_cast<double>(row.time)),
                               std::bit_cast<std::uint64_t>(static_cast<double>(row.interval)), 1,
                               HistorySampleKind::Reconstruction};
  for (std::size_t i = 0; i < result.reconstruction.size(); ++i)
    result.reconstruction[i] = std::stoull(digest.substr(16 * i, 16), nullptr, 16);
  result.validate();
  return result;
}

}  // namespace pops::runtime::program
