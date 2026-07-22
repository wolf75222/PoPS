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

struct PreparedThresholdTag {
  std::size_t field_index = 0;
  int component = 0;
  Real threshold = Real(0);
};

inline void install_prepared_threshold_union(
    AmrRuntime& runtime, std::initializer_list<PreparedThresholdTag> criteria,
    std::string provider_identity = "test::prepared-threshold-union@1") {
  using Program = AmrRuntime::TaggingProgram;
  if (criteria.size() == 0)
    throw std::invalid_argument("test threshold union requires at least one criterion");
  std::vector<Program::Leaf> leaves;
  std::vector<std::int32_t> refine_ops, refine_args;
  leaves.reserve(criteria.size());
  refine_ops.reserve(criteria.size() + (criteria.size() > 1 ? 1u : 0u));
  refine_args.reserve(refine_ops.capacity());
  for (const PreparedThresholdTag& criterion : criteria) {
    if (criterion.component < 0)
      throw std::invalid_argument("test threshold union has a negative component");
    const auto leaf_index = static_cast<std::int32_t>(leaves.size());
    leaves.push_back(Program::Leaf{criterion.field_index,
                                   static_cast<std::size_t>(criterion.component),
                                   POPS_TAGGING_ABOVE_V1, criterion.threshold,
                                   POPS_TAGGING_NO_STENCIL_V1});
    refine_ops.push_back(POPS_TAGGING_ABOVE_V1);
    refine_args.push_back(leaf_index);
  }
  if (leaves.size() > 1) {
    refine_ops.push_back(POPS_TAGGING_ANY_OF_V1);
    refine_args.push_back(static_cast<std::int32_t>(leaves.size()));
  }
  runtime.set_tagging_program({}, std::move(leaves), std::move(refine_ops),
                              std::move(refine_args), {}, {}, 0, 0, 0,
                              "test::prepared-tagging-clock", std::move(provider_identity));
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
