#pragma once

#include <pops/runtime/program/program_context.hpp>
#include <pops/runtime/system.hpp>

#include <numeric>
#include <vector>

namespace pops::test {

/// Install the simplest authored whole-system Program used by Uniform facade integration tests.
///
/// This is deliberately a real ProgramContext composition, not a callback into SystemProgramDriver:
/// solve the current fields, evaluate every block rate at the same stage, publish every new state
/// only after all rates have been evaluated, then let the facade own the accepted clock tick.
inline void install_forward_euler_program(System& system) {
  std::vector<int> block_map(static_cast<std::size_t>(system.n_blocks()));
  std::iota(block_map.begin(), block_map.end(), 0);
  system.set_program_block_map(block_map);

  runtime::program::ProgramContext context(&system);
  context.configure_primary_clock("test.clock.macro");
  context.install([context](double dt) {
    context.begin_step(dt);
    context.set_stage_time(0, 1);
    (void)context.solve_fields();

    std::vector<MultiFab*> states;
    std::vector<MultiFab*> next_states;
    states.reserve(static_cast<std::size_t>(context.n_blocks()));
    next_states.reserve(static_cast<std::size_t>(context.n_blocks()));
    for (int block = 0; block < context.n_blocks(); ++block) {
      MultiFab& state = context.state(block);
      MultiFab& residual = context.rhs_scratch(1000 + block, 0, state);
      MultiFab& next = context.scratch_state(2000 + block, 0, state);
      context.rhs_into(block, state, residual, 3000 + block);
      context.lincomb(next, Real(1), state, Real(dt), residual);
      states.push_back(&state);
      next_states.push_back(&next);
    }
    for (std::size_t block = 0; block < states.size(); ++block)
      context.lincomb(*states[block], Real(0), *states[block], Real(1), *next_states[block]);
  });
}

}  // namespace pops::test
