#pragma once

#include "component_abi_test_helpers.hpp"

#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/dynamic/prepared_execution_context.hpp>
#include <pops/runtime/program/amr_program_context.hpp>

#include <memory>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::test {

/// Install the explicit owned RuntimeInstance authority required before first AMR materialization.
/// This is test-only world-authority construction; production receives its authenticated parent
/// from the Python/native RuntimeInstance binding.
template <int Dim>
inline void install_amr_runtime_authority(AmrSystem<Dim>& system, std::string_view identity) {
  auto lane =
      std::make_shared<ExecutionLane>(ExecutionLane::duplicate_world_collectively(identity));
  const PopsExecutionContextV1 raw = component::test_support::host_execution_context();
  const component::PreparedExecutionContextV1 parent(
      raw.execution_identity, raw.context_version, raw.memory_space, raw.backend_identity,
      raw.device_identity, raw.scalar_type, raw.storage_precision, raw.compute_precision,
      raw.accumulation_precision, raw.reduction_precision, raw.stream_handle, raw.stream_identity,
      raw.communicator_f_handle, raw.communicator_datatype_f_handle, raw.communicator_identity,
      raw.communicator_datatype_identity);
  auto execution =
      std::make_shared<const component::PreparedExecutionContextV1>(parent.for_lane(*lane));
  system.install_prepared_boundary_execution_context(std::move(lane), std::move(execution));
}

/// AMR facade test Program: one explicit rate stage per recursive hierarchy clock.
///
/// AmrProgramContext owns level clocks and conservative catch-up. AmrRuntime remains the spatial
/// engine inspected by tests and exposes no temporal step entry point.
template <int Dim>
inline std::shared_ptr<runtime::program::AmrProgramContext<Dim>>
install_forward_euler_program_context(AmrSystem<Dim>& system, bool solve_default_field) {
  std::vector<int> block_map(static_cast<std::size_t>(system.n_blocks()));
  std::iota(block_map.begin(), block_map.end(), 0);
  if (system.engine() == nullptr)
    throw std::runtime_error("explicit AMR test Program requires the materialized runtime engine");
  auto context = runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("test.clock.macro");
  context->install(
      [context, solve_default_field](double macro_dt) {
        context->advance_hierarchy(macro_dt, [context, solve_default_field](double level_dt) {
          context->set_stage_time(0, 1);
          if (solve_default_field && context->level() == 0)
            (void)consume_solve_outcome(context->solve_default_field_on_coarse_level());

          std::vector<MultiFab<Dim>*> states;
          std::vector<MultiFab<Dim>*> residuals;
          states.reserve(static_cast<std::size_t>(context->n_blocks()));
          residuals.reserve(static_cast<std::size_t>(context->n_blocks()));
          for (int block = 0; block < context->n_blocks(); ++block) {
            MultiFab<Dim>& state = context->state(block);
            MultiFab<Dim>& residual = context->rhs_scratch(1000 + block, 0, state);
            context->rhs_into(block, state, residual, 3000 + block);
            states.push_back(&state);
            residuals.push_back(&residual);
          }
          for (std::size_t block = 0; block < states.size(); ++block)
            context->axpy(*states[block], Real(level_dt), *residuals[block]);
        });
      },
      context);
  // A direct Program replacement revokes every artifact-derived binding authority, including the
  // block map. Publish this fixture's explicit identity map only after the final body is installed.
  system.set_program_block_map(block_map);
  using FluxBudget = typename AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBlockBudget;
  system.install_prepared_amr_program_flux_expression_budget(
      "tests.explicit-amr-program/forward-euler@1",
      std::vector<FluxBudget>(block_map.size(), FluxBudget{1, 1}), 0, 0);
  return context;
}

template <int Dim>
inline void install_forward_euler_program(AmrSystem<Dim>& system, bool solve_default_field) {
  static_cast<void>(install_forward_euler_program_context(system, solve_default_field));
}

}  // namespace pops::test
