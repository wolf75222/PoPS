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
/// engine inspected by tests and exposes no temporal step entry point.
template <int Dim>
inline std::shared_ptr<runtime::program::AmrProgramContext<Dim>>
install_forward_euler_program_context(AmrSystem<Dim>& system) {
  std::vector<int> block_map(static_cast<std::size_t>(system.n_blocks()));
  std::iota(block_map.begin(), block_map.end(), 0);
  if (system.engine() == nullptr)
    throw std::runtime_error("explicit AMR test Program requires the materialized runtime engine");
  auto context = runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("test.clock.macro");
  context->install([context](double macro_dt) {
    context->advance_hierarchy(macro_dt, [context](double level_dt) {
      context->set_stage_time(0, 1);
      if (context->level() == 0)
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
  });
  // A direct Program replacement revokes every artifact-derived binding authority, including the
  // block map. Publish this fixture's explicit identity map only after the final body is installed.
  system.set_program_block_map(block_map);
  return context;
}

template <int Dim>
inline void install_forward_euler_program(AmrSystem<Dim>& system) {
  static_cast<void>(install_forward_euler_program_context(system));
}

}  // namespace pops::test
