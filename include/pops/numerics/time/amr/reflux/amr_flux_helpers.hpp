/// @file
/// @brief Ranked AMR transfer preparation and execution helpers.

#pragma once

#include <pops/amr/transfer/temporal_interpolation_provider.hpp>
#include <pops/amr/transfer/transfer_provider.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>

#include <array>
#include <cstddef>
#include <exception>
#include <optional>
#include <stdexcept>
#include <type_traits>
namespace pops::numerics::time::amr {

namespace detail {

template <int Dim>
struct PreparedTransferKernel {
  ::pops::amr::transfer::PreparedTransfer<Dim> transfer;

  POPS_HD void operator()(const Index<Dim>& index) const { transfer(index); }
};

template <int Dim>
struct PreparedFaceTransferKernel {
  ::pops::amr::transfer::PreparedDivergencePreservingFaceTransfer<Dim> transfer;
  int normal_axis = 0;

  POPS_HD void operator()(const Index<Dim>& index) const { transfer(normal_axis, index); }
};

template <int Dim>
struct PreparedTemporalTransferKernel {
  ::pops::amr::transfer::PreparedLinearTemporalInterpolation<Dim> transfer;

  POPS_HD void operator()(const Index<Dim>& index) const { transfer(index); }
};

template <int Dim, class MemorySpace>
void append_spatial_transfer_contract(
    ExactContractBuilder& contract,
    const ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime, std::size_t parent_level,
    const ::pops::amr::transfer::IndexMapping<Dim>& mapping) {
  contract.scalar(std::int32_t{Dim})
      .scalar(static_cast<std::uint64_t>(parent_level))
      .text(runtime.spatial_identity())
      .scalar(runtime.topology_epoch())
      .scalar(runtime.materialization_generation());
  const auto ratio = runtime.hierarchy().layout(parent_level + 1).ratio_from_parent();
  for (int axis = 0; axis < Dim; ++axis)
    contract.scalar(std::int32_t{ratio[axis]})
        .scalar(std::int32_t{mapping.coarse_origin[axis]})
        .scalar(std::int32_t{mapping.fine_origin[axis]});
}

inline void append_temporal_state_contract(
    ExactContractBuilder& contract, const ::pops::amr::transfer::QualifiedTemporalState& state) {
  contract.text(state.state_identity)
      .scalar(state.topology_generation)
      .scalar(state.materialization_generation)
      .scalar(std::int32_t{state.clock.level})
      .scalar(state.clock.macro_step)
      .scalar(state.clock.phase.numerator)
      .scalar(state.clock.phase.denominator)
      .scalar(state.clock.physical_time);
}

template <class Prepare, class Serialize>
auto prepare_collectively(const ExecutionLane& lane, Prepare&& prepare, Serialize&& serialize) {
  using Prepared = std::invoke_result_t<Prepare>;
  std::optional<Prepared> prepared;
  std::string exact_contract;
  long local_failure = 0;
  try {
    ExactContractBuilder contract;
    serialize(contract);
    exact_contract = std::move(contract).release();
    prepared.emplace(prepare());
  } catch (...) {
    local_failure = 1;
  }
  if (all_reduce_max(local_failure, lane) != 0)
    throw std::invalid_argument("prepared AMR transfer failed on at least one execution-lane rank");
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"pops.amr.prepared-transfer", exact_contract}},
                                                lane))
    throw std::invalid_argument(
        "prepared AMR transfer contract differs between execution-lane ranks");
  return std::move(*prepared);
}

}  // namespace detail

/// Execute an already authenticated transfer on an explicit execution-space instance.
template <class ExecutionSpace, int Dim>
void execute_prepared_transfer(const ExecutionSpace& execution,
                               const ::pops::amr::transfer::PreparedTransfer<Dim>& prepared) {
  for_each_cell(execution, prepared.destination_region(),
                detail::PreparedTransferKernel<Dim>{prepared});
}

/// Execute an already authenticated transfer on the default execution-space instance.
template <int Dim>
void execute_prepared_transfer(const ::pops::amr::transfer::PreparedTransfer<Dim>& prepared) {
  for_each_cell(prepared.destination_region(), detail::PreparedTransferKernel<Dim>{prepared});
}

template <class ExecutionSpace, int Dim>
void execute_prepared_transfer(
    const ExecutionSpace& execution,
    const ::pops::amr::transfer::PreparedDivergencePreservingFaceTransfer<Dim>& prepared) {
  for (int normal_axis = 0; normal_axis < Dim; ++normal_axis)
    for_each_cell(execution, prepared.destination_face_region(normal_axis),
                  detail::PreparedFaceTransferKernel<Dim>{prepared, normal_axis});
}

template <int Dim>
void execute_prepared_transfer(
    const ::pops::amr::transfer::PreparedDivergencePreservingFaceTransfer<Dim>& prepared) {
  for (int normal_axis = 0; normal_axis < Dim; ++normal_axis)
    for_each_cell(prepared.destination_face_region(normal_axis),
                  detail::PreparedFaceTransferKernel<Dim>{prepared, normal_axis});
}

template <class ExecutionSpace, int Dim>
void execute_prepared_transfer(
    const ExecutionSpace& execution,
    const ::pops::amr::transfer::PreparedLinearTemporalInterpolation<Dim>& prepared) {
  for_each_cell(execution, prepared.destination_region(),
                detail::PreparedTemporalTransferKernel<Dim>{prepared});
}

template <int Dim>
void execute_prepared_transfer(
    const ::pops::amr::transfer::PreparedLinearTemporalInterpolation<Dim>& prepared) {
  for_each_cell(prepared.destination_region(),
                detail::PreparedTemporalTransferKernel<Dim>{prepared});
}

/// Prepare conservative fine-to-parent restriction through the live AMR runtime authority.
template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedTransfer<Dim> prepare_average_down(
    const ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime, std::size_t fine_level,
    FieldView<const Real, Dim> fine, FieldView<Real, Dim> parent, const Box<Dim>& parent_region,
    ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    ::pops::amr::transfer::ComponentRange components = {}) {
  if (fine_level == 0)
    throw std::invalid_argument("AMR average-down requires a fine level");
  const std::size_t parent_level = fine_level - 1;
  return runtime.template prepare_transfer<::pops::amr::transfer::Centering::Cell>(
      fine_level, parent_level, runtime.hierarchy().level(fine_level).spatial_contract(),
      runtime.hierarchy().level(parent_level).spatial_contract(),
      ::pops::amr::transfer::TransferKind::ConservativeRestriction, fine, parent, parent_region,
      mapping, components);
}

/// Prepare parent-to-child linear prolongation through the live AMR runtime authority.
template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedTransfer<Dim> prepare_linear_prolongation(
    const ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime, std::size_t parent_level,
    FieldView<const Real, Dim> parent, FieldView<Real, Dim> child, const Box<Dim>& child_region,
    ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    ::pops::amr::transfer::ComponentRange components = {}) {
  if (parent_level >= runtime.hierarchy().num_levels() ||
      runtime.hierarchy().num_levels() - parent_level < 2)
    throw std::invalid_argument("AMR prolongation requires adjacent live levels");
  const std::size_t child_level = parent_level + 1;
  return runtime.template prepare_transfer<::pops::amr::transfer::Centering::Cell>(
      parent_level, child_level, runtime.hierarchy().level(parent_level).spatial_contract(),
      runtime.hierarchy().level(child_level).spatial_contract(),
      ::pops::amr::transfer::TransferKind::LinearProlongation, parent, child, child_region, mapping,
      components);
}

/// Prepare explicitly selected first-order parent injection through the live AMR authority.
/// This route is intentionally separate from linear prolongation: a missing linear stencil never
/// causes an implicit downgrade.
template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedTransfer<Dim> prepare_constant_injection(
    const ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime, std::size_t parent_level,
    FieldView<const Real, Dim> parent, FieldView<Real, Dim> child, const Box<Dim>& child_region,
    ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    ::pops::amr::transfer::ComponentRange components = {}) {
  if (parent_level >= runtime.hierarchy().num_levels() ||
      runtime.hierarchy().num_levels() - parent_level < 2)
    throw std::invalid_argument("AMR injection requires adjacent live levels");
  const std::size_t child_level = parent_level + 1;
  return runtime.template prepare_transfer<::pops::amr::transfer::Centering::Cell>(
      parent_level, child_level, runtime.hierarchy().level(parent_level).spatial_contract(),
      runtime.hierarchy().level(child_level).spatial_contract(),
      ::pops::amr::transfer::TransferKind::ConstantInjection, parent, child, child_region, mapping,
      components);
}

/// Prepare exact-ranked multilinear interpolation between adjacent node grids.
template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedTransfer<Dim> prepare_node_multilinear(
    const ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime, std::size_t parent_level,
    FieldView<const Real, Dim> parent_nodes, FieldView<Real, Dim> child_nodes,
    const Box<Dim>& child_node_region, ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    ::pops::amr::transfer::ComponentRange components = {}) {
  if (parent_level >= runtime.hierarchy().num_levels() ||
      runtime.hierarchy().num_levels() - parent_level < 2)
    throw std::invalid_argument("AMR node interpolation requires adjacent live levels");
  const std::size_t child_level = parent_level + 1;
  return runtime.template prepare_transfer<::pops::amr::transfer::Centering::Node>(
      parent_level, child_level, runtime.hierarchy().level(parent_level).spatial_contract(),
      runtime.hierarchy().level(child_level).spatial_contract(),
      ::pops::amr::transfer::TransferKind::NodeMultilinearProlongation, parent_nodes, child_nodes,
      child_node_region, mapping, components);
}

/// Prepare one coupled Cartesian face vector.  All oriented components are validated before the
/// returned candidate-writing kernels can run.
template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedDivergencePreservingFaceTransfer<Dim>
prepare_divergence_preserving_faces(
    const ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime, std::size_t parent_level,
    std::type_identity_t<std::array<FieldView<const Real, Dim>, Dim>> parent_faces,
    std::type_identity_t<std::array<FieldView<Real, Dim>, Dim>> child_faces,
    const Box<Dim>& child_cell_region, ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    ::pops::amr::transfer::ComponentRange components = {}) {
  if (parent_level >= runtime.hierarchy().num_levels() ||
      runtime.hierarchy().num_levels() - parent_level < 2)
    throw std::invalid_argument("AMR face interpolation requires adjacent live levels");
  const auto ratio = runtime.hierarchy().layout(parent_level + 1).ratio_from_parent();
  return ::pops::amr::transfer::DivergencePreservingFaceTransferProvider<Dim>{}.prepare(
      parent_faces, child_faces, child_cell_region, ratio, mapping, components);
}

/// Prepare a pointwise temporal candidate between two immutable parent states.  The exact AMR
/// transition, topology/materialization generations, spatial contract and rational clock window
/// must all agree before any destination cell can be written.
template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedLinearTemporalInterpolation<Dim> prepare_linear_time_interpolation(
    const ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime, std::size_t parent_level,
    FieldView<const Real, Dim> older, FieldView<const Real, Dim> newer,
    FieldView<Real, Dim> candidate, const Box<Dim>& destination_region,
    const ::pops::amr::transfer::QualifiedTemporalState& older_state,
    const ::pops::amr::transfer::QualifiedTemporalState& newer_state,
    const ::pops::amr::transfer::QualifiedTemporalState& target_state,
    ::pops::amr::transfer::TemporalComponentRange components = {}) {
  if (parent_level >= runtime.hierarchy().num_levels() ||
      runtime.hierarchy().num_levels() - parent_level < 2)
    throw std::invalid_argument("AMR temporal interpolation requires an adjacent child level");
  const auto require_live_authority =
      [&](const ::pops::amr::transfer::QualifiedTemporalState& state) {
        if (state.clock.level != static_cast<int>(parent_level) ||
            state.spatial_contract != runtime.spatial_contract() ||
            state.topology_generation != runtime.topology_epoch() ||
            state.materialization_generation != runtime.materialization_generation())
          throw std::invalid_argument(
              "AMR temporal interpolation state does not authenticate the live parent level");
      };
  require_live_authority(older_state);
  require_live_authority(newer_state);
  require_live_authority(target_state);
  return ::pops::amr::transfer::LinearTemporalInterpolationProvider<Dim>{}.prepare(
      older, newer, candidate, destination_region,
      runtime.hierarchy().layout(parent_level + 1).ratio_from_parent(), older_state, newer_state,
      target_state, components);
}

template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedTransfer<Dim> prepare_node_multilinear_collectively(
    const ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime, std::size_t parent_level,
    FieldView<const Real, Dim> parent_nodes, FieldView<Real, Dim> child_nodes,
    const Box<Dim>& child_node_region, const ExecutionLane& lane,
    ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    ::pops::amr::transfer::ComponentRange components = {}) {
  return detail::prepare_collectively(
      lane,
      [&] {
        return prepare_node_multilinear(runtime, parent_level, parent_nodes, child_nodes,
                                        child_node_region, mapping, components);
      },
      [&](ExactContractBuilder& contract) {
        contract.text("pops.amr.node-multilinear-transfer").scalar(std::uint32_t{1});
        detail::append_spatial_transfer_contract(contract, runtime, parent_level, mapping);
        contract.scalar(std::int32_t{components.source_begin})
            .scalar(std::int32_t{components.destination_begin})
            .scalar(std::int32_t{components.count});
      });
}

template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedDivergencePreservingFaceTransfer<Dim>
prepare_divergence_preserving_faces_collectively(
    const ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime, std::size_t parent_level,
    std::type_identity_t<std::array<FieldView<const Real, Dim>, Dim>> parent_faces,
    std::type_identity_t<std::array<FieldView<Real, Dim>, Dim>> child_faces,
    const Box<Dim>& child_cell_region, const ExecutionLane& lane,
    ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    ::pops::amr::transfer::ComponentRange components = {}) {
  return detail::prepare_collectively(
      lane,
      [&] {
        return prepare_divergence_preserving_faces(runtime, parent_level, parent_faces, child_faces,
                                                   child_cell_region, mapping, components);
      },
      [&](ExactContractBuilder& contract) {
        contract.text("pops.amr.divergence-preserving-face-transfer").scalar(std::uint32_t{1});
        detail::append_spatial_transfer_contract(contract, runtime, parent_level, mapping);
        contract.scalar(std::int32_t{components.source_begin})
            .scalar(std::int32_t{components.destination_begin})
            .scalar(std::int32_t{components.count});
      });
}

template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedLinearTemporalInterpolation<Dim>
prepare_linear_time_interpolation_collectively(
    const ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime, std::size_t parent_level,
    FieldView<const Real, Dim> older, FieldView<const Real, Dim> newer,
    FieldView<Real, Dim> candidate, const Box<Dim>& destination_region,
    const ::pops::amr::transfer::QualifiedTemporalState& older_state,
    const ::pops::amr::transfer::QualifiedTemporalState& newer_state,
    const ::pops::amr::transfer::QualifiedTemporalState& target_state, const ExecutionLane& lane,
    ::pops::amr::transfer::TemporalComponentRange components = {}) {
  return detail::prepare_collectively(
      lane,
      [&] {
        return prepare_linear_time_interpolation(runtime, parent_level, older, newer, candidate,
                                                 destination_region, older_state, newer_state,
                                                 target_state, components);
      },
      [&](ExactContractBuilder& contract) {
        contract.text("pops.amr.linear-time-interpolation").scalar(std::uint32_t{1});
        detail::append_spatial_transfer_contract(contract, runtime, parent_level,
                                                 ::pops::amr::transfer::IndexMapping<Dim>{});
        detail::append_temporal_state_contract(contract, older_state);
        detail::append_temporal_state_contract(contract, newer_state);
        detail::append_temporal_state_contract(contract, target_state);
        contract.scalar(std::int32_t{components.older_begin})
            .scalar(std::int32_t{components.newer_begin})
            .scalar(std::int32_t{components.destination_begin})
            .scalar(std::int32_t{components.count});
      });
}

/// Prepare parent-to-child coarse/fine ghost interpolation through the live runtime authority.
template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedTransfer<Dim> prepare_fill_patch(
    const ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime, std::size_t parent_level,
    FieldView<const Real, Dim> parent, FieldView<Real, Dim> child, const Box<Dim>& ghost_region,
    ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    ::pops::amr::transfer::ComponentRange components = {}) {
  if (parent_level >= runtime.hierarchy().num_levels() ||
      runtime.hierarchy().num_levels() - parent_level < 2)
    throw std::invalid_argument("AMR fill-patch preparation requires adjacent live levels");
  const std::size_t child_level = parent_level + 1;
  return runtime.template prepare_transfer<::pops::amr::transfer::Centering::Cell>(
      parent_level, child_level, runtime.hierarchy().level(parent_level).spatial_contract(),
      runtime.hierarchy().level(child_level).spatial_contract(),
      ::pops::amr::transfer::TransferKind::CoarseFineGhostInterpolation, parent, child,
      ghost_region, mapping, components);
}

/// Prepare the distinct fifth-order parent-to-child cell-average interpolation used by
/// high-order reconstructions.  Missing radius-two parent data is rejected during preparation;
/// this route never falls back to the second-order fill-patch provider.
template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedTransfer<Dim> prepare_fifth_order_fill_patch(
    const ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime, std::size_t parent_level,
    FieldView<const Real, Dim> parent, FieldView<Real, Dim> child, const Box<Dim>& ghost_region,
    ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    ::pops::amr::transfer::ComponentRange components = {}) {
  if (parent_level >= runtime.hierarchy().num_levels() ||
      runtime.hierarchy().num_levels() - parent_level < 2)
    throw std::invalid_argument(
        "AMR fifth-order fill-patch preparation requires adjacent live levels");
  const std::size_t child_level = parent_level + 1;
  return runtime.template prepare_transfer<::pops::amr::transfer::Centering::Cell>(
      parent_level, child_level, runtime.hierarchy().level(parent_level).spatial_contract(),
      runtime.hierarchy().level(child_level).spatial_contract(),
      ::pops::amr::transfer::TransferKind::FifthOrderCoarseFineGhostInterpolation, parent, child,
      ghost_region, mapping, components);
}

}  // namespace pops::numerics::time::amr
