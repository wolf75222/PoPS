#pragma once

// Binding-only preparation protocol. These callbacks are strictly local: a callback must not
// enter MPI, return a lazy computation, or publish accepted state. Every rank calls the same method.
#include <pybind11/pybind11.h>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::python_collective {
namespace py = pybind11;

// ObserverMpiLane::communicator() alone returns an inactive view after close. Its rank() accessor
// authenticates openness; a closed owner must never silently turn this protocol into local work.
// As for the existing byte APIs, close is allowed only after every observer worker has joined.
template<class Communicator>
CommunicatorView open_communicator(const Communicator& owner) {
  (void)owner.rank();
  return owner.communicator();
}

// Only nothrow moves leave this phase after its collective. Python objects are destroyed with
// the GIL held, including the pending local result when another rank could not prepare its value.
template<class Operation>
auto prepare(const CommunicatorView& communicator, Operation&& operation) {
  using Value = std::invoke_result_t<Operation>;
  static_assert(!std::is_void_v<Value> && std::is_nothrow_move_constructible_v<Value>);
  std::optional<Value> pending;
  long failed = 0;
  try {
    pending.emplace(std::forward<Operation>(operation)());
  } catch (py::error_already_set& error) {
    // Fetch/clear without calling what(), formatting a traceback or allocating a diagnostic.
    error.restore();
    PyErr_Clear();
    failed = 1;
  } catch (...) {
    failed = 1;
  }
  long any_failed = 0;
  {
    py::gil_scoped_release release;
    any_failed = all_reduce_max(failed, communicator);
  }
  if (any_failed)
    throw std::runtime_error("collective Python local preparation failed");
  return std::move(*pending);
}

inline std::string_view exact_bytes(py::handle value) {
  if (!PyBytes_CheckExact(value.ptr()))
    throw std::invalid_argument("collective control transport requires exact bytes");
  char* data = nullptr;
  Py_ssize_t length = 0;
  if (PyBytes_AsStringAndSize(value.ptr(), &data, &length) != 0)
    throw py::error_already_set();
  if (length < 0 || static_cast<std::uintmax_t>(length) >
                        static_cast<std::uintmax_t>(std::numeric_limits<long>::max()))
    throw std::length_error("collective control bytes exceed native length capacity");
  return {data, static_cast<std::size_t>(length)};
}

template<class Communicator>
py::object collective_prepare(const Communicator& owner, py::handle callback) {
  const auto communicator = open_communicator(owner);
  return prepare(communicator, [&]() -> py::object {
    if (!PyCallable_Check(callback.ptr()))
      throw std::invalid_argument("collective preparation requires a callable");
    return py::reinterpret_borrow<py::function>(callback)();
  });
}

// Immutable Python bytes remain alive as binding arguments while the GIL is released. We borrow
// their storage locally; no pointer is transmitted. Fixed stack chunks bound extra comparison
// memory independently of payload length and rank count, and preserve exact embedded NUL bytes.
template<class Communicator>
bool agree_control_bytes(const Communicator& owner, py::handle schema, py::handle payload) {
  const auto communicator = open_communicator(owner);
  const auto values = prepare(communicator, [&] {
    return std::array<std::string_view, 2>{exact_bytes(schema), exact_bytes(payload)};
  });
  py::gil_scoped_release release;
  std::array<long, 2> minimum_lengths{static_cast<long>(values[0].size()),
                                      static_cast<long>(values[1].size())};
  auto maximum_lengths = minimum_lengths;
  all_reduce_min_inplace(minimum_lengths.data(), minimum_lengths.size(), communicator);
  all_reduce_max_inplace(maximum_lengths.data(), maximum_lengths.size(), communicator);
  if (minimum_lengths != maximum_lengths)
    return false;
  constexpr std::size_t chunk_size = 16384;
  std::array<char, chunk_size> minimum{}, maximum{};
  bool same = true;
  for (const auto value : values) {
    for (std::size_t offset = 0; offset < value.size();) {
      const auto count = std::min(chunk_size, value.size() - offset);
      std::copy_n(value.data() + offset, count, minimum.data());
      std::copy_n(minimum.data(), count, maximum.data());
      all_reduce_min_inplace(minimum.data(), count, communicator);
      all_reduce_max_inplace(maximum.data(), count, communicator);
      same = same && std::equal(minimum.begin(), minimum.begin() + count, maximum.begin());
      offset += count;
    }
  }
  return same;
}

template<class Communicator>
py::tuple allgather_bytes(const Communicator& owner, py::handle payload) {
  const auto communicator = open_communicator(owner);
  const auto native = prepare(communicator, [&] {
    return std::string(exact_bytes(payload));
  });
  std::vector<std::string> gathered;
  {
    py::gil_scoped_release release;
    gathered = owner.allgather_bytes(native);
  }
  return prepare(communicator, [&] {
    py::tuple result(gathered.size());
    for (std::size_t index = 0; index < gathered.size(); ++index)
      result[index] = py::bytes(gathered[index]);
    return result;
  });
}
} // namespace pops::python_collective
