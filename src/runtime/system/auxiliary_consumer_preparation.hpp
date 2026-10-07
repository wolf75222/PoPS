#pragma once

#include <pops/runtime/system/auxiliary_ghost_fill.hpp>
#include <pops/runtime/system/derived_aux_provider.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace pops { template <int> class AmrSystem; }

namespace pops::runtime::system::detail {

// Only the owning AMR facade can issue this transient accepted/topology authority.
// The validator captures live session, hierarchy, original carrier and Clock membership.
class NativeAcceptedAuxiliaryPoint {
 public:
  void validate() const { validate_(); }
  multiblock::BoundaryEvaluationPoint point_for_level(int level) const {
    validate();
    if (level < 0 || static_cast<std::size_t>(level) >= levels_)
      throw std::out_of_range("accepted auxiliary point requires an actual owned level");
    auto result = point_;
    result.level = level;
    return result;
  }
 private:
  template <int> friend class ::pops::AmrSystem;
  NativeAcceptedAuxiliaryPoint(multiblock::BoundaryEvaluationPoint point, std::size_t levels,
                               std::function<void()> validate)
      : point_(std::move(point)), levels_(levels), validate_(std::move(validate)) {}
  multiblock::BoundaryEvaluationPoint point_;
  std::size_t levels_;
  std::function<void()> validate_;
};

// Private Core seam: the consumer chooses its actual execution event. Source/layout
// authentication stays in the runtime caller and is voted before publication starts.
template <class Authenticate>
AuxiliaryEvaluationPoint prepare_auxiliary_consumer_point(
    const multiblock::BoundaryEvaluationPoint& point, std::uint64_t topology,
    std::uint64_t generation, int sequence, AuxiliaryEvaluationEvent event,
    const ExecutionLane& lane, std::string_view consumer, int block,
    std::string_view operation, Authenticate&& authenticate) {
  AuxiliaryEvaluationPoint auxiliary;
  std::string contract;
  std::exception_ptr error;
  try {
    authenticate();
    if (point.tick < 0 || sequence < 0)
      throw std::invalid_argument("auxiliary consumer requires a non-negative point/sequence");
    auxiliary.clock = point.clock;
    auxiliary.accepted_step = static_cast<std::uint64_t>(point.tick);
    auxiliary.layout_generation = generation;
    auxiliary.level = point.level;
    auxiliary.substep = point.substep;
    auxiliary.stage = point.stage;
    auxiliary.nonlinear_iteration = sequence;
    auxiliary.event = event;
    auxiliary.qualify_physical_evaluation(point);
    ExactContractBuilder exact;
    exact.text("pops.runtime.auxiliary-consumer-preparation")
        .scalar(std::uint32_t{1}).text(operation).text(consumer).scalar(block).scalar(topology);
    auxiliary.serialize_exact(exact);
    contract = std::move(exact).release();
  } catch (...) {
    error = std::current_exception();
  }
  auxiliary_ghost_detail::rethrow_collective_failure(
      error, &lane, "auxiliary consumer preparation failed collectively");
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"auxiliary-consumer-preparation", contract}}, lane))
    throw std::invalid_argument("auxiliary consumer event/point/source differs across MPI ranks");
  return auxiliary;
}

}  // namespace pops::runtime::system::detail
