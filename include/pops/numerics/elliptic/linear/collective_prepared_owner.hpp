#pragma once

/// @file
/// @brief Rank-local prepared solve construction before entry into collective constructors.

#include <pops/parallel/collective_exception.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

namespace pops::detail {

/// The callback must contain only local work. All collective context lookups must already have
/// completed in the parent's canonical order. The returned value cannot throw while crossing the
/// fence: a failing move there would let peers enter the following constructor collective alone.
template <class Prepare>
auto prepare_linear_local(const ExecutionCommunicator& parent, Prepare&& prepare) {
  using Value = std::invoke_result_t<Prepare>;
  static_assert(!std::is_reference_v<Value> && std::is_nothrow_move_constructible_v<Value>);
  std::optional<Value> value;
  std::exception_ptr error;
  try {
    value.emplace(std::invoke(std::forward<Prepare>(prepare)));
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, parent.communicator(),
                                 "prepared linear construction failed before collective entry");
  return std::move(*value);
}

template <class T, class Inputs>
struct PreparedSharedCandidate {
  std::shared_ptr<std::optional<T>> owner;
  Inputs inputs;
};

/// Reserve the shared owner and all exact constructor inputs before any rank enters T. The empty
/// optional allocates no T or ExecutionLane. An aliasing shared_ptr subsequently exposes T without
/// a second control-block allocation and keeps its storage alive until the last borrower retires.
template <class T, class Inputs, class Prepare, class Allocator>
auto prepare_shared_candidate(const ExecutionCommunicator& parent, Prepare&& prepare,
                              const Allocator& allocator) {
  static_assert(std::is_same_v<Inputs, std::invoke_result_t<Prepare>>);
  return prepare_linear_local(parent, [&]() {
    auto owner = std::allocate_shared<std::optional<T>>(allocator);
    return PreparedSharedCandidate<T, Inputs>{std::move(owner),
                                              std::invoke(std::forward<Prepare>(prepare))};
  });
}

}  // namespace pops::detail
