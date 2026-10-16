#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace pops::runtime::program::finite_kokkos {

// Device storage is a staging representation of existing immutable Program values.
// Ownership, collectives, failure publication and continuations stay on the host.
template <class T>
using View = Kokkos::View<T*>;
template <class T>
View<T> allocate(std::int64_t count, const char* label) {
  if (count < 0 ||
      static_cast<std::uint64_t>(count) > std::numeric_limits<std::size_t>::max() / sizeof(T))
    throw std::length_error("finite Kokkos staging allocation exceeds addressable memory");
  return View<T>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)),
                 std::size_t(count));
}

// No undefined signed arithmetic, exceptions or host intrinsics inside kernels.
// A failed operation does not write a value consumed by another operation.
KOKKOS_INLINE_FUNCTION bool integer(unsigned op, std::int64_t a, std::int64_t b,
                                    std::int64_t& out) {
  constexpr auto lo = std::numeric_limits<std::int64_t>::min();
  constexpr auto hi = std::numeric_limits<std::int64_t>::max();
  if (op == 0) {
    if (b > 0 ? a > hi - b : a < lo - b)
      return false;
    out = a + b;
  } else if (op == 1) {
    if (b > 0 ? a < lo + b : a > hi + b)
      return false;
    out = a - b;
  } else if (op == 2) {
    if (a != 0 && b != 0 &&
        !(a > 0 ? (b > 0 ? a <= hi / b : b >= lo / a) : (b > 0 ? a >= lo / b : a >= hi / b)))
      return false;
    out = a * b;
  } else {
    if (b == 0 || (a == lo && b == -1))
      return false;
    const auto rem = a % b;
    const bool adjust = rem != 0 && ((rem < 0) != (b < 0));
    out = op == 3 ? a / b - (adjust ? 1 : 0) : rem + (adjust ? b : 0);
  }
  return true;
}

}  // namespace pops::runtime::program::finite_kokkos
