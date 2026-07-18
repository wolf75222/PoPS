#pragma once

/// @file
/// @brief Prepared spatial-provider aliases used by elliptic construction and embedded boundaries.

#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/prepared_provider.hpp>

#include <utility>

namespace pops {

template <class Value, class... Coordinates>
using SpatialProvider = PreparedProvider<Value(Coordinates...)>;

template <class Value>
using SpatialProvider2D = SpatialProvider<Value, Real, Real>;

using ActiveRegionProvider2D = SpatialProvider2D<bool>;
using LevelSetProvider2D = SpatialProvider2D<Real>;
using ScalarFieldProvider2D = SpatialProvider2D<Real>;

namespace detail {

struct ConstantScalarFieldSource2D {
  Real value = Real(0);

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.scalar-field.constant", 1};
  }

  void serialize_exact_parameters(ExactContractBuilder& contract) const { contract.scalar(value); }

  [[nodiscard]] Real operator()(Real, Real) const noexcept { return value; }
};

struct NegativeLevelSetActiveRegionSource2D {
  LevelSetProvider2D level_set;

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.active-region.level-set-negative", 1};
  }

  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.optional_collective_contract(level_set);
  }

  [[nodiscard]] bool operator()(Real x, Real y) const { return level_set(x, y) < Real(0); }
};

}  // namespace detail

inline ScalarFieldProvider2D constant_scalar_field_provider(Real value) {
  return ScalarFieldProvider2D(detail::ConstantScalarFieldSource2D{value});
}

/// Derive the conventional negative-level-set active region without hiding its provenance. The
/// complete prepared level-set contract is nested as one exact byte field in the derived contract.
inline ActiveRegionProvider2D active_region_from_level_set(LevelSetProvider2D level_set) {
  if (!level_set)
    return {};
  return ActiveRegionProvider2D(detail::NegativeLevelSetActiveRegionSource2D{std::move(level_set)});
}

}  // namespace pops
