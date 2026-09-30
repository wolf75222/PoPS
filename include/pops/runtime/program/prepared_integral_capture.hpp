#pragma once

#include <pops/core/identity/sha256.hpp>
#include <pops/runtime/program/accepted_exchange.hpp>
#include <pops/runtime/program/prepared_resource_lifetime.hpp>

namespace pops::runtime::program {
template <int Dim> class ProgramContext;
template <int Dim, class MemorySpace> class AmrProgramContext;

/// A global candidate read, never a durable accepted-state receipt or a cell field.
/// Only the owning Program context can capture or authorize its POD kernel value.
class PreparedIntegralCapture {
 public:
  PreparedIntegralCapture(const PreparedIntegralCapture&) = default;
  PreparedIntegralCapture(PreparedIntegralCapture&&) noexcept = default;

 private:
  template <int> friend class ProgramContext;
  template <int, class> friend class AmrProgramContext;
  PreparedIntegralCapture() = default;
  struct ReadImage {
    double value;
    std::vector<std::uint8_t> ledger;
  };

  static void require_units_(const std::string& identity, const std::string& units) {
    const auto digest = ::pops::identity::sha256_hex(
        std::vector<std::uint8_t>(units.begin(), units.end()));
    if (!identity.starts_with("pops.integral.v2/") || units.empty() ||
        identity.find("/" + digest + "/") == std::string::npos)
      throw std::invalid_argument("integral candidate capture requires its exact typed v2 units");
  }
  static std::string contract_(const std::string& identity, const std::string& units,
                              const multiblock::BoundaryEvaluationPoint& point, double value) {
    if (point.clock.empty() || point.tick < 0 || point.level < 0 || point.substep < 0 ||
        point.stage < 0 || !std::isfinite(point.dt) || point.dt <= 0 ||
        !std::isfinite(point.physical_time) || !std::isfinite(value))
      throw std::invalid_argument("integral candidate capture requires a complete finite point/value");
    ExactContractBuilder contract;
    contract.text("pops.integral-candidate-capture.v1").text(identity).text(units)
        .text(point.clock).scalar(point.tick).scalar(point.level).scalar(point.substep)
        .scalar(point.stage).scalar(point.stage_fraction.numerator)
        .scalar(point.stage_fraction.denominator).scalar(point.dt)
        .scalar(point.physical_time).text(point.graph_identity).text(point.rate_identity)
        .text(point.application_identity).scalar(value);
    return std::move(contract).release();
  }
  template <class Attempt, class Point, class Read>
  static PreparedIntegralCapture prepare_(const void* owner, Attempt&& attempt_get,
      Point&& point_get, const std::string& identity,
      const std::string& units, Read&& read, const ExecutionLane& lane) {
    PreparedIntegralCapture result;
    std::string payload;
    std::exception_ptr error;
    try {
      require_units_(identity, units);
      auto attempt = attempt_get();
      const auto point = point_get();
      if (!owner || !attempt.visible())
        throw std::logic_error("integral candidate capture requires a live Program attempt");
      result.owner_ = owner; result.attempt_ = std::move(attempt);
      result.point_ = point; result.identity_ = identity; result.units_ = units;
      auto image = read();
      result.value_ = image.value;
      result.ledger_ = std::move(image.ledger);
      payload = contract_(identity, units, point, result.value_);
    } catch (...) { error = std::current_exception(); }
    collectively_rethrow_exception(error, lane, "integral candidate capture preparation");
    if (!all_ranks_agree_exact_ordered_byte_pairs({{"integral-candidate",payload}}, lane))
      throw std::runtime_error("integral candidate capture differs across MPI ranks");
    return result;
  }
  template <class Attempt, class Point, class Read>
  double consume_(const void* owner, Attempt&& attempt_get,
      Point&& point_get, const std::string& identity,
      const std::string& units, Read&& read, const ExecutionLane& lane) const {
    std::string payload;
    std::exception_ptr error;
    try {
      require_units_(identity, units);
      const auto attempt = attempt_get();
      const auto point = point_get();
      if (owner != owner_ || !attempt_.visible() || !attempt_.same_attempt(attempt) ||
          point != point_ || identity != identity_ || units != units_)
        throw std::logic_error("integral candidate capture has stale owner/attempt/point/provenance");
      const auto image = read();
      const double current = image.value;
      if (std::bit_cast<std::uint64_t>(current) != std::bit_cast<std::uint64_t>(value_))
        throw std::logic_error("integral candidate capture value changed before consumption");
      if (image.ledger != ledger_)
        throw std::logic_error("integral candidate capture ledger provenance changed before consumption");
      payload = contract_(identity, units, point, value_);
    } catch (...) { error = std::current_exception(); }
    collectively_rethrow_exception(error, lane, "integral candidate capture consumption");
    if (!all_ranks_agree_exact_ordered_byte_pairs({{"integral-candidate",payload}}, lane))
      throw std::runtime_error("integral candidate capture consumption differs across MPI ranks");
    return value_;
  }
  const void* owner_ = nullptr;
  PreparedResourceAttempt attempt_;
  multiblock::BoundaryEvaluationPoint point_;
  std::string identity_, units_;
  std::vector<std::uint8_t> ledger_;
  double value_ = 0;
};
}  // namespace pops::runtime::program
