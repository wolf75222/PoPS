/// @file
/// @brief Device-safe pointwise state and exact provider-value packs.

#pragma once

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/core/foundation/types.hpp>

#include <cassert>
#include <array>
#include <type_traits>

namespace pops {

/// Conserved state vector of fixed size, known at compile time.
template <int N>
struct StateVec {
  Real v[N]{};

  POPS_HD Real& operator[](int i) { return v[i]; }
  POPS_HD Real operator[](int i) const { return v[i]; }

  POPS_HD static constexpr int size() { return N; }
};

template <int N>
POPS_HD StateVec<N> operator+(StateVec<N> a, const StateVec<N>& b) {
  for (int i = 0; i < N; ++i)
    a[i] += b[i];
  return a;
}

template <int N>
POPS_HD StateVec<N> operator-(StateVec<N> a, const StateVec<N>& b) {
  for (int i = 0; i < N; ++i)
    a[i] -= b[i];
  return a;
}

template <int N>
POPS_HD StateVec<N> operator*(Real s, StateVec<N> a) {
  for (int i = 0; i < N; ++i)
    a[i] *= s;
  return a;
}

/// A compact pointwise value pack for exactly one consumer's resolved providers.
///
/// `Count` comes from the consumer/provider requirement graph.  Its slots are deliberately
/// *only* the dense, local ABI positions ``0..Count-1``: a value has no implicit physical name,
/// spatial axis, model, or globally reserved channel.  A consumer owns the interpretation through
/// its explicitly bound indices.  The zero-provider pack is valid and device-copyable.
template <int Count>
struct ProviderValues {
  static_assert(Count >= 0, "provider value count cannot be negative");

  static constexpr int size = Count;
  std::array<Real, static_cast<std::size_t>(Count)> values{};

  POPS_HD Real& operator[](int slot) {
    assert(slot >= 0 && slot < Count);
    return values[static_cast<std::size_t>(slot)];
  }
  POPS_HD Real operator[](int slot) const {
    assert(slot >= 0 && slot < Count);
    return values[static_cast<std::size_t>(slot)];
  }

  template <int Slot>
  POPS_HD Real value() const {
    static_assert(Slot >= 0 && Slot < Count,
                  "provider slot is outside this consumer's exact compact pack");
    return values[static_cast<std::size_t>(Slot)];
  }
};

/// Compile-time binding of one consumer's logical inputs to compact provider slots.
///
/// This type carries no semantic label: a brick gives each position its physical role.  Keeping
/// the binding in the consumer type makes a permuted assignment a different native specialization,
/// which is important for the exact artifact identity and prevents accidental name-based aliasing.
template <int... Slots>
struct ProviderSlots {
  static_assert(((Slots >= 0) && ...), "provider slots cannot be negative");
  static constexpr int count = sizeof...(Slots);
  inline static constexpr std::array<int, count> values{Slots...};

  template <int Position>
  static consteval int slot() {
    static_assert(Position >= 0 && Position < count,
                  "provider binding position is outside this consumer map");
    return values[static_cast<std::size_t>(Position)];
  }

  static consteval int required_count() {
    int count_required = 0;
    for (const int value : values)
      if (value + 1 > count_required)
        count_required = value + 1;
    return count_required;
  }
};

/// Read a provider through either the common compact pack or a qualified bound carrier.
template <int Slot, class Providers>
POPS_HD Real provider_value(const Providers& providers) {
  static_assert(Slot >= 0, "provider slot cannot be negative");
  if constexpr (requires { providers.template value<Slot>(); })
    return providers.template value<Slot>();
  else
    return providers.template provider<Slot>();
}

static_assert(std::is_trivially_copyable_v<ProviderValues<0>> &&
                  std::is_trivially_copyable_v<ProviderValues<1>> &&
                  std::is_trivially_copyable_v<ProviderValues<7>>,
              "ProviderValues must remain trivially copyable for device kernels");
static_assert(std::is_standard_layout_v<ProviderValues<0>> &&
                  std::is_standard_layout_v<ProviderValues<7>>,
              "ProviderValues must remain standard-layout");
static_assert(std::is_aggregate_v<ProviderValues<0>> && std::is_aggregate_v<ProviderValues<7>>,
              "ProviderValues must remain an aggregate POD");

}  // namespace pops
