#pragma once
#include <pops/runtime/dynamic/component_consumers.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace pops::runtime {
/// Non-integrating accepted initial Field authority@1. The owner constructs it only
/// under its genuine bootstrap transaction, from authenticated primary-clock metadata.
/// It does not grant a positive interval or a general permission to solve at dt=0.
class PreparedAcceptedInitialFieldPointV1 {
 public:
  static constexpr std::uint32_t contract_version = 1;
  PreparedAcceptedInitialFieldPointV1(const multiblock::BoundaryEvaluationPoint& point,
      std::string_view owner_clock, std::int64_t owner_tick, double owner_time,
      int active_level, bool bootstrap_active) : point_(point) {
    const PopsLogicalTimeV1 logical{sizeof(PopsLogicalTimeV1), point.clock.c_str(),
        point.tick, point.level, point.substep, point.stage,
        point.stage_fraction.numerator, point.stage_fraction.denominator,
        point.dt, point.physical_time};
    component::validate_accepted_initial_evaluation_point(logical, contract_version);
    if (!bootstrap_active || owner_clock.empty() || point.clock != owner_clock ||
        owner_tick != 0 || point.tick != owner_tick || point.physical_time != owner_time ||
        point.level != active_level)
      throw std::invalid_argument("accepted initial Field point@1 differs from bootstrap owner authority");
  }
  bool authenticates(const multiblock::BoundaryEvaluationPoint& point) const noexcept {
    return !std::signbit(point.dt) && point == point_;
  }
 private:
  multiblock::BoundaryEvaluationPoint point_;
};
} // namespace pops::runtime
