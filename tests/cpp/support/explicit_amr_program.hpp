#pragma once

#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/program/amr_program_context.hpp>

#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace pops::test {

/// AMR facade test Program: one explicit rate stage per recursive hierarchy clock.
///
/// AmrProgramContext owns level clocks and conservative catch-up. AmrRuntime remains the spatial
/// engine inspected by tests; no test invokes its legacy temporal step entry point.
inline void install_forward_euler_program(AmrSystem& system) {
  std::vector<int> block_map(static_cast<std::size_t>(system.n_blocks()));
  std::iota(block_map.begin(), block_map.end(), 0);
  system.set_program_block_map(block_map);
  // The facade selects the common AmrRuntime route during lazy construction only when a Program
  // authority already exists. Install a temporary body before materialization; the typed
  // AmrProgramContext below replaces it immediately after the engine becomes available.
  system.install_program_step([](double) {});
  if (!system.uses_runtime_engine() || system.engine() == nullptr)
    throw std::runtime_error("explicit AMR test Program requires the materialized runtime engine");

  auto context = std::make_shared<runtime::program::AmrProgramContext>(system.engine(), &system);
  context->configure_primary_clock("test.clock.macro");
  context->install([context](double macro_dt) {
    context->advance_hierarchy(macro_dt, [context](double level_dt) {
      context->set_stage_time(0, 1);
      (void)context->solve_fields();

      std::vector<MultiFab*> states;
      std::vector<MultiFab*> residuals;
      states.reserve(static_cast<std::size_t>(context->n_blocks()));
      residuals.reserve(static_cast<std::size_t>(context->n_blocks()));
      for (int block = 0; block < context->n_blocks(); ++block) {
        MultiFab& state = context->state(block);
        MultiFab& residual = context->rhs_scratch(1000 + block, 0, state);
        context->rhs_into(block, state, residual, 3000 + block);
        states.push_back(&state);
        residuals.push_back(&residual);
      }
      for (std::size_t block = 0; block < states.size(); ++block)
        context->axpy(*states[block], Real(level_dt), *residuals[block], Real(level_dt),
                      {{1, 1, 1}});
    });
  });
}

}  // namespace pops::test
