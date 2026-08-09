/// @file
/// @brief Exact-ranked association of a model, state field, spatial scheme, and time policy.
///
/// A PhysicalModel describes a local pointwise law. An EquationBlock says how this law
/// is carried by one immutable native rank: which MultiFab U, spatial discretisation, and time
/// policy. Boundary execution belongs to prepared runtime plans, not to this model aggregate.
///
/// INVARIANT: `state` is never null after construction (points to the MultiFab passed to the
/// constructor). The EquationBlock does not own the MultiFab; the MultiFab lifetime must
/// exceed that of the block.
///
/// `EquationBlockLike`: minimal concept letting the CoupledSystem and the scheduler
/// manipulate a block without knowing its concrete types (ModelT, SpatialT, TimeT).

#pragma once

#include <pops/core/model/physical_model.hpp>
#include <pops/numerics/time/integrators/time_integrator.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/fv/spatial_discretisation.hpp>

#include <concepts>
#include <string_view>

namespace pops {

/// Association of a PhysicalModel with one exact-ranked field, spatial scheme, and time policy.
///
/// Template parameters:
///   Dim: immutable spatial rank.
///   ModelT: must satisfy PhysicalModelFor<ModelT, Dim>.
///   SpatialT: must satisfy SpatialDiscretisationLike (default FirstOrder).
///   TimeT: time policy (default ExplicitTime<SSPRK2>).
///
/// INVARIANT: `state != nullptr` after construction. The EquationBlock does NOT own
/// the MultiFab; the MultiFab lifetime must exceed that of the block.
/// Do not store in a container by value if the MultiFab is dynamically allocated
/// and could be moved (the pointer would become invalid).
template <int Dim, class ModelT, class SpatialT = FirstOrder, class TimeT = ExplicitTime<SSPRK2>,
          class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct EquationBlock {
  static_assert(Dim >= 1 && Dim <= 3, "EquationBlock only supports dimensions 1, 2, and 3");
  static_assert(PhysicalModelFor<ModelT, Dim>,
                "EquationBlock expects a model matching its exact spatial rank");
  static_assert(SpatialDiscretisationLike<SpatialT>,
                "EquationBlock expects a named spatial discretisation");

  using Model = ModelT;
  using Spatial = SpatialT;
  using Time = TimeT;
  using field_type = MultiFab<Dim, MemorySpace>;
  using memory_space = MemorySpace;
  static constexpr int dimension = Dim;

  std::string_view name{};
  Model model;
  field_type* state = nullptr;

  EquationBlock(std::string_view block_name, const Model& block_model, field_type& block_state)
      : name(block_name), model(block_model), state(&block_state) {}

  field_type& U() { return *state; }
  const field_type& U() const { return *state; }
};

/// Minimal concept for equation blocks: Model, Spatial, Time, name, state, U().
/// Lets the CoupledSystem and the scheduler manipulate a block without knowing its
/// concrete types. The field carries the same immutable rank as the block.
template <class B>
concept EquationBlockLike = requires(B b) {
  typename B::Model;
  typename B::Spatial;
  typename B::Time;
  typename B::field_type;
  typename B::memory_space;
  requires(B::dimension >= 1 && B::dimension <= 3);
  requires(B::field_type::dimension == B::dimension);
  { b.name } -> std::convertible_to<std::string_view>;
  { b.state } -> std::same_as<typename B::field_type*&>;
  { b.U() } -> std::same_as<typename B::field_type&>;
};

}  // namespace pops
