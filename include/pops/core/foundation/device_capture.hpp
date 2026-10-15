#pragma once

#include <Kokkos_Core.hpp>
#include <type_traits>

namespace pops {

/// This is a representation/lifetime contract, not a claim that an arbitrary
/// pointer is device-accessible. Owned Kokkos views retain their allocations
/// across host-side kernel capture. The called device methods still have to
/// compile for the selected backend and may only dereference the correct space.
/// Generated algebra aggregates list all nonstatic data members. Handwritten
/// aggregates must supply the same exhaustive contract; unknown host containers
/// never become admissible just because they have a copy constructor.
template <class T, class = void>
struct DeviceCapture : std::bool_constant<std::is_trivially_copyable_v<T>> {};

// Only managed views supply the ownership needed by the additional capture
// route. An unmanaged device pointer needs an independent lifetime contract.
template <class T>
struct DeviceCapture<T, std::enable_if_t<Kokkos::is_view<T>::value>>
    : std::bool_constant<!T::memory_traits::is_unmanaged> {};

template <class... Members>
struct DeviceCaptureMembers : std::bool_constant<(DeviceCapture<Members>::value && ...)> {};

template <class T>
struct DeviceCapture<T, std::void_t<typename T::pops_device_capture_members>>
    : std::bool_constant<T::pops_device_capture_members::value && std::is_copy_constructible_v<T> &&
                         std::is_destructible_v<T>> {};

template <class T>
inline constexpr bool device_capture_v = DeviceCapture<T>::value;

}  // namespace pops
