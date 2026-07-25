#pragma once

#include <pops/runtime/amr/amr_runtime.hpp>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::test {

enum class PreparedThresholdRelation : std::uint8_t { Above, Below };

struct PreparedThresholdTag {
  std::size_t field_index = 0;
  int component = 0;
  Real threshold = Real(0);
  PreparedThresholdRelation relation = PreparedThresholdRelation::Above;
};

inline void install_prepared_threshold_decisions(
    AmrRuntime& runtime, std::initializer_list<PreparedThresholdTag> refine_criteria,
    std::initializer_list<PreparedThresholdTag> coarsen_criteria,
    std::string provider_identity = "test::prepared-threshold-decisions@1") {
  using Program = AmrRuntime::TaggingProgram;
  if (refine_criteria.size() == 0)
    throw std::invalid_argument("test threshold decisions require a refine root");
  std::vector<Program::Leaf> leaves;
  std::vector<std::int32_t> refine_ops, refine_args, coarsen_ops, coarsen_args;
  leaves.reserve(refine_criteria.size() + coarsen_criteria.size());
  const auto append_union = [&](std::initializer_list<PreparedThresholdTag> criteria,
                                std::vector<std::int32_t>& ops,
                                std::vector<std::int32_t>& args) {
    ops.reserve(criteria.size() + (criteria.size() > 1 ? 1u : 0u));
    args.reserve(ops.capacity());
    for (const PreparedThresholdTag& criterion : criteria) {
      if (criterion.component < 0)
        throw std::invalid_argument("test threshold decision has a negative component");
      const std::int32_t opcode =
          criterion.relation == PreparedThresholdRelation::Above ? POPS_TAGGING_ABOVE_V1
                                                                 : POPS_TAGGING_BELOW_V1;
      const auto leaf_index = static_cast<std::int32_t>(leaves.size());
      leaves.push_back(Program::Leaf{criterion.field_index,
                                     static_cast<std::size_t>(criterion.component),
                                     opcode, criterion.threshold, POPS_TAGGING_NO_STENCIL_V1});
      ops.push_back(opcode);
      args.push_back(leaf_index);
    }
    if (criteria.size() > 1) {
      ops.push_back(POPS_TAGGING_ANY_OF_V1);
      args.push_back(static_cast<std::int32_t>(criteria.size()));
    }
  };
  append_union(refine_criteria, refine_ops, refine_args);
  append_union(coarsen_criteria, coarsen_ops, coarsen_args);
  runtime.set_tagging_program({}, std::move(leaves), std::move(refine_ops),
                              std::move(refine_args), std::move(coarsen_ops),
                              std::move(coarsen_args), 0, 0, 0,
                              "test::prepared-tagging-clock", std::move(provider_identity));
}

inline void install_prepared_threshold_union(
    AmrRuntime& runtime, std::initializer_list<PreparedThresholdTag> criteria,
    std::string provider_identity = "test::prepared-threshold-union@1") {
  install_prepared_threshold_decisions(runtime, criteria, {}, std::move(provider_identity));
}

inline void install_prepared_shared_aux_gradient(
    AmrRuntime& runtime, std::size_t block_count, Real threshold,
    std::string provider_identity = "test::prepared-shared-aux-gradient@1") {
  using Program = AmrRuntime::TaggingProgram;
  std::vector<Program::Stencil> stencils{
      Program::Stencil{"test::shared-aux-centered-gradient",
                       POPS_TAGGING_STENCIL_ROUTE_LINEAR_AXIS_STENCIL_L2_V1,
                       "l2",
                       "inverse_cell_size",
                       "ghost_extension",
                       2,
                       {Program::AxisStencil{0, 1, 2, 1, 1, {-1, 1}, {-0.5, 0.5}},
                        Program::AxisStencil{1, 1, 2, 1, 1, {-1, 1}, {-0.5, 0.5}}}}};
  runtime.set_tagging_program(
      std::move(stencils),
      {Program::Leaf{block_count, 0, POPS_TAGGING_GRADIENT_ABOVE_V1, threshold, 0}},
      {POPS_TAGGING_GRADIENT_ABOVE_V1}, {0}, {}, {}, 0, 0, 0,
      "test::prepared-tagging-clock", std::move(provider_identity));
}

}  // namespace pops::test
