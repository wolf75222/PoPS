/// @file
/// @brief Axis-static conservative or primitive reconstruction over canonical ND field views.

#pragma once

#include <pops/mesh/index/entity_index.hpp>
#include <pops/mesh/storage/fab.hpp>
#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/spatial/primitives/state_access.hpp>

#include <concepts>
#include <stdexcept>
#include <type_traits>

namespace pops::nd {

/// Reconstruction variables are a preparation-time type choice.  Face kernels never branch on
/// this value and never infer the algorithm from a ghost width.
enum class ReconstructionVariables { Conservative, Primitive };

template <class State>
struct ReconstructedFacePair {
  State left{};
  State right{};
  StateConversionStatus left_status = StateConversionStatus::NonFiniteState;
  StateConversionStatus right_status = StateConversionStatus::NonFiniteState;

  POPS_HD bool succeeded() const {
    return left_status == StateConversionStatus::Success &&
           right_status == StateConversionStatus::Success;
  }
};

namespace reconstruction_detail {

template <int Axis, int Orientation, int Dim>
POPS_HD Index<Dim> displaced(Index<Dim> index, int offset) {
  static_assert(Axis >= 0 && Axis < Dim,
                "reconstruction axis is outside the compile-time dimension");
  static_assert(Orientation == -1 || Orientation == 1,
                "reconstruction orientation must be a compile-time sign");
  index[Axis] += Orientation * offset;
  return index;
}

template <int Axis, int Orientation, int Dim>
struct ConservativeComponentSampler {
  FieldView<const Real, Dim> state{};
  Index<Dim> source{};
  int component = 0;

  POPS_HD Real operator()(int offset) const {
    return state(displaced<Axis, Orientation>(source, offset), component);
  }
};

template <class Primitive, int MinimumOffset, int MaximumOffset>
struct PrimitiveComponentSampler {
  static_assert(MinimumOffset <= MaximumOffset);
  static constexpr int count = MaximumOffset - MinimumOffset + 1;

  const Primitive* values = nullptr;
  int component = 0;

  POPS_HD Real operator()(int offset) const { return values[offset - MinimumOffset][component]; }
};

template <class Model>
POPS_HD StateConversion<typename Model::State> checked_conservative(
    const Model& model, const typename Model::State& state) {
  return {state, model.admissibility(state)};
}

template <int Axis, int Orientation, int Dim, class Model, class Reconstruction>
POPS_HD StateConversion<typename Model::State> reconstruct_conservative(
    const Model& model, const FieldView<const Real, Dim>& state, const Index<Dim>& source,
    const Reconstruction& reconstruction) {
  typename Model::State face = pops::load_state<Model>(state, source);
  for (int component = 0; component < Model::n_vars; ++component) {
    if constexpr (CellValueReconstruction<Reconstruction>) {
      const ConservativeComponentSampler<Axis, Orientation, Dim> sample{state, source, component};
      const Real center = sample(0);
      face[component] = reconstruction.cell_face_value(center);
    } else if constexpr (SlopeReconstruction<Reconstruction>) {
      // Limiter differences are always formed in canonical axis order. Orientation selects the
      // side of the resulting centered slope exactly once below.
      const ConservativeComponentSampler<Axis, 1, Dim> sample{state, source, component};
      const Real center = sample(0);
      face[component] =
          center + Real(0.5) * Real(Orientation) *
                       reconstruction.limited_slope(center - sample(-1), sample(1) - center);
    } else {
      const ConservativeComponentSampler<Axis, Orientation, Dim> sample{state, source, component};
      face[component] = reconstruction.stencil_face_value(sample);
    }
  }
  return checked_conservative(model, face);
}

template <int Axis, int Orientation, int Dim, class Model, class Reconstruction>
POPS_HD StateConversion<typename Model::State> reconstruct_primitive(
    const Model& model, const FieldView<const Real, Dim>& state, const Index<Dim>& source,
    const Reconstruction& reconstruction) {
  using Primitive = typename Model::Primitive;

  if constexpr (CellValueReconstruction<Reconstruction>) {
    const auto primitive = model.recover(pops::load_state<Model>(state, source));
    if (!primitive.succeeded())
      return {{}, primitive.status};
    return model.make_conservative(primitive.value);
  } else if constexpr (SlopeReconstruction<Reconstruction>) {
    const auto center = model.recover(pops::load_state<Model>(state, source));
    if (!center.succeeded())
      return {{}, center.status};
    const auto lower =
        model.recover(pops::load_state<Model>(state, displaced<Axis, 1>(source, -1)));
    if (!lower.succeeded())
      return {{}, lower.status};
    const auto upper = model.recover(pops::load_state<Model>(state, displaced<Axis, 1>(source, 1)));
    if (!upper.succeeded())
      return {{}, upper.status};

    Primitive face{};
    for (int component = 0; component < Model::n_vars; ++component)
      face[component] =
          center.value[component] +
          Real(0.5) * Real(Orientation) *
              reconstruction.limited_slope(center.value[component] - lower.value[component],
                                           upper.value[component] - center.value[component]);
    return model.make_conservative(face);
  } else {
    using Envelope = ReconstructionStencilEnvelope<Reconstruction>;
    constexpr int minimum = Envelope::min_offset;
    constexpr int maximum = Envelope::max_offset;
    constexpr int count = maximum - minimum + 1;
    Primitive values[count]{};
    for (int offset = minimum; offset <= maximum; ++offset) {
      const auto recovered = model.recover(
          pops::load_state<Model>(state, displaced<Axis, Orientation>(source, offset)));
      if (!recovered.succeeded())
        return {{}, recovered.status};
      values[offset - minimum] = recovered.value;
    }

    Primitive face{};
    for (int component = 0; component < Model::n_vars; ++component) {
      const PrimitiveComponentSampler<Primitive, minimum, maximum> sample{values, component};
      face[component] = reconstruction.stencil_face_value(sample);
    }
    return model.make_conservative(face);
  }
}

template <int Dim>
Box<Dim> required_reconstruction_box(const Box<Dim>& cells, int ghost_depth) {
  if (cells.empty())
    throw std::invalid_argument("ND reconstruction requires a non-empty cell box");
  return cells.grow(ghost_depth);
}

}  // namespace reconstruction_detail

/// Reconstruct one source-cell trace toward the compile-time-oriented face.
template <int Axis, int Orientation,
          ReconstructionVariables Variables = ReconstructionVariables::Conservative, int Dim,
          class Model, class Reconstruction>
  requires(ConservationLaw<Dim, Model> && ReconstructionPolicy<Reconstruction> &&
           requires(const Model& model, const typename Model::Primitive& primitive) {
             {
               model.make_conservative(primitive)
             } -> std::same_as<StateConversion<typename Model::State>>;
           })
POPS_HD StateConversion<typename Model::State> reconstruct_face_state(
    const Model& model, const FieldView<const Real, Dim>& state, const Index<Dim>& source,
    const Reconstruction& reconstruction) {
  static_assert(Axis >= 0 && Axis < Dim,
                "reconstruction axis is outside the compile-time dimension");
  static_assert(Orientation == -1 || Orientation == 1,
                "reconstruction orientation must be a compile-time sign");
  static_assert(stencil_envelope_fits_storage<Reconstruction>,
                "sampled reconstruction offsets exceed the declared ghost contract");
  if constexpr (Variables == ReconstructionVariables::Primitive)
    return reconstruction_detail::reconstruct_primitive<Axis, Orientation>(model, state, source,
                                                                           reconstruction);
  else
    return reconstruction_detail::reconstruct_conservative<Axis, Orientation>(model, state, source,
                                                                              reconstruction);
}

/// Reconstruct the two traces adjacent to one canonical positive-axis face.
template <int Axis, ReconstructionVariables Variables = ReconstructionVariables::Conservative,
          int Dim, class Model, class Reconstruction>
POPS_HD ReconstructedFacePair<typename Model::State> reconstruct_face_pair(
    const Model& model, const FieldView<const Real, Dim>& state, const FaceIndex<Dim, Axis>& face,
    const Reconstruction& reconstruction) {
  Index<Dim> left_source = face.coordinate;
  --left_source[Axis];
  const Index<Dim> right_source = face.coordinate;
  const auto left =
      reconstruct_face_state<Axis, 1, Variables>(model, state, left_source, reconstruction);
  const auto right =
      reconstruct_face_state<Axis, -1, Variables>(model, state, right_source, reconstruction);
  return {left.value, right.value, left.status, right.status};
}

/// Host preflight for the exact storage envelope used by every axis-static face kernel.
template <class Reconstruction, int Dim, class MemorySpace>
void require_reconstruction_storage(const Fab<Dim, MemorySpace>& state, const Box<Dim>& cells,
                                    int nvars) {
  static_assert(ReconstructionPolicy<Reconstruction>);
  static_assert(stencil_envelope_fits_storage<Reconstruction>);
  if (nvars < 1 || state.ncomp() != nvars)
    throw std::invalid_argument("ND reconstruction state component count does not match the law");
  if (!(state.box() == cells))
    throw std::invalid_argument("ND reconstruction state valid box does not match the operator");
  const Box<Dim> required =
      reconstruction_detail::required_reconstruction_box(cells, Reconstruction::n_ghost);
  if (!state.grown_box().contains(required))
    throw std::invalid_argument(
        "ND reconstruction state does not carry the policy's exact ghost envelope");
}

}  // namespace pops::nd
