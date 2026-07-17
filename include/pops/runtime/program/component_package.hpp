#pragma once

// Generic AOT component protocol types.  This header performs no registration and owns no global
// state: a generated wrapper instantiates the component against these resolved types, then exports
// an audited fixed ABI for the selected target.

#include <array>
#include <cstddef>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define POPS_COMPONENT_HD __host__ __device__
#else
#define POPS_COMPONENT_HD
#endif

namespace pops::component_package {

template <class Scalar, std::size_t Components>
struct Trace {
  std::array<Scalar, Components> values{};

  POPS_COMPONENT_HD const Scalar& operator[](std::size_t index) const { return values[index]; }
  POPS_COMPONENT_HD Scalar& operator[](std::size_t index) { return values[index]; }
};

template <class Scalar, std::size_t Dimension>
struct Face {
  std::array<Scalar, Dimension> normal{};
  Scalar measure{Scalar{1}};
};

template <class Scalar, std::size_t Components>
struct NumericalFluxProviders {
  using normal_flux_type = std::array<Scalar, Components>;
  using normal_speed_type = Scalar;
};

template <class Component>
struct RegisteredComponent final {
  using type = Component;
};

}  // namespace pops::component_package

// Compile-time marker only.  Package discovery and registration are manifest-driven and happen
// before compilation; this macro deliberately creates no static initializer or process-global map.
#define POPS_COMPONENT_PACKAGE_CONCAT_IMPL(left, right) left##right
#define POPS_COMPONENT_PACKAGE_CONCAT(left, right) POPS_COMPONENT_PACKAGE_CONCAT_IMPL(left, right)
#define POPS_REGISTER_COMPONENT(Component)                                       \
  using POPS_COMPONENT_PACKAGE_CONCAT(pops_registered_component_, __COUNTER__) = \
      ::pops::component_package::RegisteredComponent<Component>
