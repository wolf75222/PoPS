#pragma once

#include <pops/parallel/comm.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <typeindex>
#include <utility>

namespace pops::runtime::program {

/// Context-owned numerical storage, separate from accepted state and evaluated data.
/// A Program node owns its resources until the context's layout generation changes;
/// different nodes/blocks/levels never share retained stage outputs. A caller must
/// reevaluate its law after acquire. Constructors containing collectives must make
/// their internal allocations/failures collective too, as prepared providers do.
class PreparedResourceCache {
 public:
  PreparedResourceCache() = default;
  PreparedResourceCache(const PreparedResourceCache&) = delete;
  PreparedResourceCache& operator=(const PreparedResourceCache&) = delete;
  PreparedResourceCache(PreparedResourceCache&&) = default;
  PreparedResourceCache& operator=(PreparedResourceCache&&) = default;

  template <class Resource, class Matches, class... Args>
  Resource& acquire(std::int64_t node, int block, int level, const ExecutionLane& lane,
                    Matches&& matches, Args&&... args) {
    using Slot = std::optional<Resource>;
    std::shared_ptr<Slot> slot;
    std::exception_ptr error;
    long disposition = 0;
    try {
      if (node < 0 || block < 0 || level < 0)
        throw std::invalid_argument("prepared resource requires exact nonnegative scope ids");
      const Key key{std::type_index(typeid(Resource)), node, block, level};
      auto entry = entries_.try_emplace(key).first;
      // Allocate the empty holder before peers can enter a provider constructor.
      // Allocation failure leaves a retryable empty entry, never a readable resource.
      if (!entry->second)
        entry->second = std::make_shared<Slot>();
      slot = std::static_pointer_cast<Slot>(entry->second);
      disposition = !slot->has_value() || !matches(slot->value()) ? 1L : 0L;
    } catch (...) {
      error = std::current_exception();
      disposition = 2;
    }
    const long collective = all_reduce_max(disposition, lane);
    if (collective == 2) {
      if (error)
        std::rethrow_exception(error);
      throw std::runtime_error("prepared resource preflight failed on another rank");
    }
    if (collective == 1)
      slot->emplace(std::forward<Args>(args)...);
    return slot->value();
  }

  void clear() noexcept { entries_.clear(); }

 private:
  using Key = std::tuple<std::type_index, std::int64_t, int, int>;
  std::map<Key, std::shared_ptr<void>> entries_;
};

}  // namespace pops::runtime::program
