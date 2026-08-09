/// @file
/// @brief Compile-time state schemas shared by the 1D, 2D and 3D finite-volume laws.

#pragma once

#include <pops/core/state/state.hpp>

#include <array>
#include <cstddef>

namespace pops::nd {

template <int Dim>
struct ScalarStateSchema {
  static_assert(Dim >= 1 && Dim <= 3, "scalar finite-volume states support dimensions 1..3");

  static constexpr int dimension = Dim;
  static constexpr int nvars = 1;
  static constexpr int scalar = 0;
  using Conservative = StateVec<nvars>;
  using Primitive = StateVec<nvars>;
};

/// Axis-indexed Euler layout used by the ND laws.
///
/// Conservative components are ``[rho, rho*u_0, ..., rho*u_(Dim-1), E]`` and primitive
/// components are ``[rho, u_0, ..., u_(Dim-1), p]``.  Normal and tangent identities are compile
/// time values: a face kernel never performs a run-time permutation of its state schema.
template <int Dim>
struct EulerStateSchema {
  static_assert(Dim >= 1 && Dim <= 3, "Euler finite-volume states support dimensions 1..3");

  static constexpr int dimension = Dim;
  static constexpr int nvars = Dim + 2;
  static constexpr int density = 0;
  static constexpr int energy = Dim + 1;
  static constexpr int pressure = Dim + 1;

  using Conservative = StateVec<nvars>;
  using Primitive = StateVec<nvars>;

  template <int Axis>
  static constexpr int momentum = [] {
    static_assert(Axis >= 0 && Axis < Dim, "Euler momentum axis is outside the state dimension");
    return Axis + 1;
  }();

  template <int Axis>
  static constexpr int velocity = momentum<Axis>;

  template <int NormalAxis, int TangentOrdinal>
  static constexpr int tangent_axis = [] {
    static_assert(NormalAxis >= 0 && NormalAxis < Dim,
                  "Euler normal axis is outside the state dimension");
    static_assert(TangentOrdinal >= 0 && TangentOrdinal < Dim - 1,
                  "Euler tangent ordinal is outside the tangent subspace");
    return TangentOrdinal < NormalAxis ? TangentOrdinal : TangentOrdinal + 1;
  }();

  template <int NormalAxis, int TangentOrdinal>
  static constexpr int tangent_momentum = momentum<tangent_axis<NormalAxis, TangentOrdinal>>;

  template <int NormalAxis>
  static consteval std::array<int, Dim - 1> tangent_axes() {
    static_assert(NormalAxis >= 0 && NormalAxis < Dim,
                  "Euler normal axis is outside the state dimension");
    std::array<int, Dim - 1> result{};
    int ordinal = 0;
    for (int axis = 0; axis < Dim; ++axis)
      if (axis != NormalAxis)
        result[static_cast<std::size_t>(ordinal++)] = axis;
    return result;
  }
};

enum class StateConversionStatus : unsigned char {
  Success = 0,
  NonFiniteState = 1,
  NonPositiveDensity = 2,
  NonPositivePressure = 3,
  InvalidEquationOfState = 4,
};

template <class State>
struct StateConversion {
  State value{};
  StateConversionStatus status = StateConversionStatus::NonFiniteState;

  POPS_HD constexpr bool succeeded() const { return status == StateConversionStatus::Success; }
};

static_assert(ScalarStateSchema<1>::nvars == ScalarStateSchema<3>::nvars);
static_assert(EulerStateSchema<1>::nvars == 3);
static_assert(EulerStateSchema<2>::nvars == 4);
static_assert(EulerStateSchema<3>::nvars == 5);
static_assert(EulerStateSchema<3>::template momentum<2> == 3);
static_assert(EulerStateSchema<3>::template tangent_axis<1, 0> == 0);
static_assert(EulerStateSchema<3>::template tangent_axis<1, 1> == 2);

}  // namespace pops::nd
