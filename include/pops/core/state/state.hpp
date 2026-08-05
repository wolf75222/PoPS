/// @file
/// @brief Device-safe pointwise state and exact-ranked auxiliary-provider values.

#pragma once

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/core/foundation/types.hpp>

#include <cassert>
#include <limits>
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

/// Maximum number of model-named provider values retained in the pointwise POD.
inline constexpr int kAuxMaxExtra = 4;

/// Compile-time component map for one spatial rank.
///
/// The channel is exactly `phi`, one gradient component per spatial axis, `B_z`, `T_e`, then the
/// bounded model-named values. Consequently every index after `phi` moves with `Dim`; a 1D channel
/// has no `grad_y` slot and a 3D channel has a real third gradient slot.
template <int Dim>
struct AuxComponentLayout {
  static_assert(Dim >= 1 && Dim <= 3, "pops::AuxComponentLayout supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;
  static constexpr int phi = 0;
  static constexpr int gradient_begin = phi + 1;

  template <int Axis>
  POPS_HD static constexpr int gradient_component() {
    static_assert(Axis >= 0 && Axis < Dim, "aux gradient axis is outside the spatial rank");
    return gradient_begin + Axis;
  }

  static constexpr int base_components = gradient_begin + Dim;
  static constexpr int b_z = base_components;
  static constexpr int t_e = b_z + 1;
  static constexpr int named_begin = t_e + 1;
  static constexpr int max_components = named_begin + kAuxMaxExtra;
};

template <int Dim>
inline constexpr int kAuxBaseCompsFor = AuxComponentLayout<Dim>::base_components;

template <int Dim, int Axis>
inline constexpr int kAuxGradientComponentFor =
    AuxComponentLayout<Dim>::template gradient_component<Axis>();

template <int Dim>
inline constexpr int kAuxBzComponentFor = AuxComponentLayout<Dim>::b_z;

template <int Dim>
inline constexpr int kAuxTeComponentFor = AuxComponentLayout<Dim>::t_e;

template <int Dim>
inline constexpr int kAuxNamedBaseFor = AuxComponentLayout<Dim>::named_begin;

template <int Dim>
inline constexpr int kAuxMaxCompsFor = AuxComponentLayout<Dim>::max_components;

/// Build-specialized compatibility constants. They describe only the immutable native rank.
inline constexpr int kAuxBaseComps = kAuxBaseCompsFor<kNativeDimension>;
template <int Axis>
inline constexpr int kAuxGradientComponent = kAuxGradientComponentFor<kNativeDimension, Axis>;
inline constexpr int kAuxBzComponent = kAuxBzComponentFor<kNativeDimension>;
inline constexpr int kAuxTeComponent = kAuxTeComponentFor<kNativeDimension>;
inline constexpr int kAuxNamedBase = kAuxNamedBaseFor<kNativeDimension>;
inline constexpr int kAuxMaxComps = kAuxMaxCompsFor<kNativeDimension>;

/// Build-specialized canonical-extra table retained for native host/code-generation adapters.
#define POPS_AUX_FIELDS(X)        \
  X(B_z, ::pops::kAuxBzComponent) \
  X(T_e, ::pops::kAuxTeComponent)

/// Pointwise auxiliary-provider value with storage exact to the compile-time spatial rank.
template <int Dim>
struct AuxState {
  static_assert(Dim >= 1 && Dim <= 3, "pops::AuxState supports dimensions 1, 2, and 3");

  using component_layout = AuxComponentLayout<Dim>;
  static constexpr int dimension = Dim;

  Real phi{};
  Real gradients[Dim]{};
  Real B_z{};
  Real T_e{};
  Real extra[kAuxMaxExtra]{};

  /// Access the physical gradient component carried by compile-time axis `Axis`.
  template <int Axis>
  POPS_HD Real& gradient() {
    static_assert(Axis >= 0 && Axis < Dim, "aux gradient axis is outside the spatial rank");
    return gradients[Axis];
  }

  template <int Axis>
  POPS_HD Real gradient() const {
    static_assert(Axis >= 0 && Axis < Dim, "aux gradient axis is outside the spatial rank");
    return gradients[Axis];
  }

  /// Read one declared model-named provider slot.
  POPS_HD Real extra_field(int slot) const {
    assert(slot >= 0 && slot < kAuxMaxExtra);
    return (slot >= 0 && slot < kAuxMaxExtra) ? extra[slot]
                                              : std::numeric_limits<Real>::quiet_NaN();
  }

  /// Compile-time provider read shared by the finite-volume and pointwise source laws.
  template <int Component>
  POPS_HD Real flux_provider() const {
    static_assert(Component >= 0 && Component < component_layout::max_components,
                  "physical flux provider component is outside the ranked native capability");
    if constexpr (Component == component_layout::phi)
      return phi;
    else if constexpr (Component >= component_layout::gradient_begin &&
                       Component < component_layout::base_components)
      return gradients[Component - component_layout::gradient_begin];
    else if constexpr (Component == component_layout::b_z)
      return B_z;
    else if constexpr (Component == component_layout::t_e)
      return T_e;
    else
      return extra[Component - component_layout::named_begin];
  }
};

/// Pointwise auxiliary type of this compiled native artifact.
using Aux = AuxState<kNativeDimension>;

static_assert(kAuxMaxExtra >= 1, "kAuxMaxExtra must allow at least one model-named aux field");
static_assert(std::is_trivially_copyable_v<AuxState<1>> &&
                  std::is_trivially_copyable_v<AuxState<2>> &&
                  std::is_trivially_copyable_v<AuxState<3>>,
              "AuxState must remain trivially copyable for device kernels");
static_assert(std::is_standard_layout_v<AuxState<1>> && std::is_standard_layout_v<AuxState<2>> &&
                  std::is_standard_layout_v<AuxState<3>>,
              "AuxState must remain standard-layout");
static_assert(std::is_aggregate_v<AuxState<1>> && std::is_aggregate_v<AuxState<2>> &&
                  std::is_aggregate_v<AuxState<3>>,
              "AuxState must remain an aggregate POD");
static_assert(sizeof(AuxState<1>) == sizeof(Real) * kAuxMaxCompsFor<1> &&
                  sizeof(AuxState<2>) == sizeof(Real) * kAuxMaxCompsFor<2> &&
                  sizeof(AuxState<3>) == sizeof(Real) * kAuxMaxCompsFor<3>,
              "AuxState storage must contain exactly its ranked channel components");
static_assert(static_cast<int>(sizeof(Aux::extra) / sizeof(Real)) == kAuxMaxExtra,
              "AuxState::extra[] must match the declared named-provider bound");

}  // namespace pops
