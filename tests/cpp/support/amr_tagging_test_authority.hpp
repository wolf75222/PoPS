#pragma once

#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr_system.hpp>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::test {

enum class PreparedThresholdRelation : std::uint8_t { Above, Below };

struct PreparedNamedThresholdTag {
  std::string block;
  std::string variable;
  double threshold = 0.0;
  PreparedThresholdRelation relation = PreparedThresholdRelation::Above;
  std::string state_identity;
};

inline std::string direct_amr_state_identity(const std::string& block) {
  return "pops://runtime/amr-direct-state/" + std::to_string(block.size()) + ":" + block;
}

template <int Dim>
inline typename runtime::amr::PreparedTaggingProgram<Dim>::Stencil centered_gradient_stencil() {
  typename runtime::amr::PreparedTaggingProgram<Dim>::Stencil stencil;
  stencil.identity = "test::shared-aux-centered-gradient";
  stencil.route = POPS_TAGGING_STENCIL_ROUTE_LINEAR_AXIS_STENCIL_L2_V1;
  stencil.norm = "l2";
  stencil.scale = "inverse_cell_size";
  stencil.boundary_mode = "ghost_extension";
  for (int axis = 0; axis < Dim; ++axis)
    stencil.axes[axis] = {axis, 1, 2, 1, 1, {-1, 1}, {-0.5, 0.5}};
  return stencil;
}

template <int Dim>
inline void install_prepared_threshold_union(
    AmrSystem<Dim>& system, std::initializer_list<PreparedNamedThresholdTag> criteria,
    std::string provider_identity = "test::prepared-named-threshold-union@1",
    std::string clock_identity = "test::prepared-tagging-clock", int min_cycles = 0) {
  if (criteria.size() == 0)
    throw std::invalid_argument("test named threshold union requires a refine root");
  std::vector<std::string> subject_kinds, subject_identities, blocks, variables;
  std::vector<int> field_component_indices;
  std::vector<int> leaf_ops, stencil_indices;
  std::vector<double> thresholds;
  std::vector<std::int32_t> refine_ops, refine_args;
  subject_kinds.reserve(criteria.size());
  subject_identities.reserve(criteria.size());
  blocks.reserve(criteria.size());
  variables.reserve(criteria.size());
  field_component_indices.reserve(criteria.size());
  leaf_ops.reserve(criteria.size());
  thresholds.reserve(criteria.size());
  stencil_indices.reserve(criteria.size());
  refine_ops.reserve(criteria.size() + (criteria.size() > 1 ? 1u : 0u));
  refine_args.reserve(refine_ops.capacity());
  for (const PreparedNamedThresholdTag& criterion : criteria) {
    if (criterion.block.empty() || criterion.variable.empty())
      throw std::invalid_argument("test named threshold decision requires block and variable");
    const int opcode = criterion.relation == PreparedThresholdRelation::Above
                           ? POPS_TAGGING_ABOVE_V1
                           : POPS_TAGGING_BELOW_V1;
    const auto leaf_index = static_cast<std::int32_t>(blocks.size());
    subject_kinds.emplace_back("state");
    subject_identities.push_back(criterion.state_identity.empty()
                                     ? direct_amr_state_identity(criterion.block)
                                     : criterion.state_identity);
    blocks.push_back(criterion.block);
    variables.push_back(criterion.variable);
    field_component_indices.push_back(-1);
    leaf_ops.push_back(opcode);
    thresholds.push_back(criterion.threshold);
    stencil_indices.push_back(-1);
    refine_ops.push_back(opcode);
    refine_args.push_back(leaf_index);
  }
  if (criteria.size() > 1) {
    refine_ops.push_back(POPS_TAGGING_ANY_OF_V1);
    refine_args.push_back(static_cast<std::int32_t>(criteria.size()));
  }
  system.set_bootstrap_tagging(subject_kinds, subject_identities, blocks, variables,
                               field_component_indices, leaf_ops, thresholds, stencil_indices, {},
                               refine_ops, refine_args, {}, {}, min_cycles, "hold", "error",
                               std::move(clock_identity), std::move(provider_identity));
}

template <int Dim>
inline void install_prepared_thresholds_and_shared_aux_gradient(
    AmrSystem<Dim>& system, std::initializer_list<PreparedNamedThresholdTag> criteria,
    Real gradient_threshold,
    std::string provider_identity = "test::prepared-thresholds-and-gradient@1") {
  if (criteria.size() == 0)
    throw std::invalid_argument("test combined tagging program requires a state threshold");
  using Program = runtime::amr::PreparedTaggingProgram<Dim>;
  std::vector<typename Program::Stencil> stencils{centered_gradient_stencil<Dim>()};
  std::vector<std::string> subject_kinds, subject_identities, blocks, variables;
  std::vector<int> field_component_indices, leaf_ops, stencil_indices;
  std::vector<double> thresholds;
  std::vector<std::int32_t> refine_ops, refine_args;
  const std::size_t leaf_count = criteria.size() + 1;
  subject_kinds.reserve(leaf_count);
  subject_identities.reserve(leaf_count);
  blocks.reserve(leaf_count);
  variables.reserve(leaf_count);
  field_component_indices.reserve(leaf_count);
  leaf_ops.reserve(leaf_count);
  thresholds.reserve(leaf_count);
  stencil_indices.reserve(leaf_count);
  for (const PreparedNamedThresholdTag& criterion : criteria) {
    const int opcode = criterion.relation == PreparedThresholdRelation::Above
                           ? POPS_TAGGING_ABOVE_V1
                           : POPS_TAGGING_BELOW_V1;
    const auto leaf_index = static_cast<std::int32_t>(leaf_ops.size());
    subject_kinds.emplace_back("state");
    subject_identities.push_back(criterion.state_identity.empty()
                                     ? direct_amr_state_identity(criterion.block)
                                     : criterion.state_identity);
    blocks.push_back(criterion.block);
    variables.push_back(criterion.variable);
    field_component_indices.push_back(-1);
    leaf_ops.push_back(opcode);
    thresholds.push_back(criterion.threshold);
    stencil_indices.push_back(-1);
    refine_ops.push_back(opcode);
    refine_args.push_back(leaf_index);
  }
  if (criteria.size() > 1) {
    refine_ops.push_back(POPS_TAGGING_ANY_OF_V1);
    refine_args.push_back(static_cast<std::int32_t>(criteria.size()));
  }
  const auto gradient_index = static_cast<std::int32_t>(leaf_ops.size());
  subject_kinds.emplace_back("aux");
  subject_identities.emplace_back("pops://runtime/amr/shared-aux");
  blocks.emplace_back();
  variables.emplace_back("phi");
  field_component_indices.push_back(-1);
  leaf_ops.push_back(POPS_TAGGING_GRADIENT_ABOVE_V1);
  thresholds.push_back(static_cast<double>(gradient_threshold));
  stencil_indices.push_back(0);
  refine_ops.push_back(POPS_TAGGING_GRADIENT_ABOVE_V1);
  refine_args.push_back(gradient_index);
  refine_ops.push_back(POPS_TAGGING_ANY_OF_V1);
  refine_args.push_back(2);
  system.set_bootstrap_tagging(subject_kinds, subject_identities, blocks, variables,
                               field_component_indices, leaf_ops, thresholds, stencil_indices,
                               stencils, refine_ops, refine_args, {}, {}, 0, "hold", "error",
                               "test::prepared-tagging-clock", std::move(provider_identity));
}

}  // namespace pops::test
