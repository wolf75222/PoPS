#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace pops::runtime::program {

// Persisted scientific origin of a Field, not a live permission to read it.
// The context must authenticate its current storage instance separately. A
// restart may re-materialize that instance without changing this old origin.
struct ContinuationFieldSupport {
  using Ref = std::shared_ptr<const ContinuationFieldSupport>;

  std::string program_identity;
  std::string origin_point;
  int program_owner = -1;
  int producer_level = -1;
  std::uint64_t topology_epoch = 0;
  std::uint64_t materialization_generation = 0;

  void validate() const {
    if (program_identity.empty() || origin_point.empty() || program_owner < 0 || producer_level < 0)
      throw std::invalid_argument("Continuation Field has an incomplete producer support");
  }

  friend bool operator==(const ContinuationFieldSupport&,
                         const ContinuationFieldSupport&) = default;
};

}  // namespace pops::runtime::program
