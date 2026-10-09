#pragma once
#include <cstdint>
#include <cstddef>
#include <stdexcept>

namespace pops {
// Private test protocol; independent from scientific accepted/checkpoint contracts.
enum class AcceptedHaloTestFailurePhase : std::uint32_t { after_block_level_preparation = 1 };
struct AcceptedHaloTestFailureRequest {
  std::uint32_t version = 1;
  AcceptedHaloTestFailurePhase phase = AcceptedHaloTestFailurePhase::after_block_level_preparation;
  int block = -1;
  int level = -1;
  int rank = -1;
  friend bool operator==(const AcceptedHaloTestFailureRequest&, const AcceptedHaloTestFailureRequest&) = default;
};
struct AcceptedHaloTestFailureReceipt {
  AcceptedHaloTestFailureRequest request;
  bool requested = false;
  bool reached = false;
  bool consumed = false;
  bool before_publication = false;
  bool local_error = false;
  std::int64_t tick = -1;
  std::int64_t armed_tick = -1;
  std::uint64_t topology_epoch = 0;
  double physical_time = 0;
  double dt = 0;
};
inline void validate_accepted_halo_test_failure_request(const AcceptedHaloTestFailureRequest& r,
                                                       std::size_t blocks, std::size_t levels, int ranks) {
  if (r.version != 1 || r.phase != AcceptedHaloTestFailurePhase::after_block_level_preparation ||
      r.block < 0 || static_cast<std::size_t>(r.block) >= blocks || r.level < 0 ||
      static_cast<std::size_t>(r.level) >= levels || r.rank < 0 || r.rank >= ranks)
    throw std::invalid_argument("accepted halo test failure request has unsupported version/phase/target");
}
} // namespace pops
