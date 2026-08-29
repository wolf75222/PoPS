#pragma once

#include "native_dso_compiler.hpp"
#include "program_v5_fixture.hpp"

#include <pops/runtime/program/program_execution_services.hpp>
#include <pops/runtime/system.hpp>

#include <cstdint>
#include <fstream>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace pops::test {

namespace explicit_system_program_detail {

using context_type = runtime::program::ProgramExecutionServices<kNativeDimension>;
using callback_type = std::function<void(context_type&, double)>;

inline std::vector<callback_type>& callbacks() {
  static std::vector<callback_type> value;
  return value;
}

}  // namespace explicit_system_program_detail

extern "C" void pops_test_explicit_system_program_callback(std::uint64_t identifier, void* opaque,
                                                           double dt) {
  auto& callbacks = explicit_system_program_detail::callbacks();
  if (opaque == nullptr || identifier >= callbacks.size())
    throw std::logic_error("explicit System ABI-v5 callback received an invalid dispatch token");
  callbacks.at(static_cast<std::size_t>(identifier))(
      *static_cast<explicit_system_program_detail::context_type*>(opaque), dt);
}

/// Install the simplest authored whole-system Program used by Uniform facade integration tests.
///
/// This is deliberately a real ProgramExecutionServices composition, not a callback into a facade stepper:
/// solve the current fields, evaluate every block rate at the same stage, publish every new state
/// only after all rates have been evaluated, then let the facade own the accepted clock tick.
template <int Dim>
inline void install_forward_euler_program(System<Dim>& system) {
  static_assert(Dim == kNativeDimension,
                "the ABI-v5 explicit System fixture is compiled for POPS_NATIVE_DIM");
  using Resource = program_v5::CallbackProgramResource;
  std::vector<Resource> resources;
  resources.reserve(static_cast<std::size_t>(2 * system.n_blocks()));
  for (int block = 0; block < system.n_blocks(); ++block) {
    const auto state = system.block_state(block);
    if (!state)
      throw std::logic_error("explicit System Program requires materialized block states");
    const auto components = static_cast<std::uint32_t>(state->ncomp());
    const auto ghosts = static_cast<std::uint32_t>(state->ghosts()[0]);
    resources.push_back({Resource::Kind::rhs, resources.size(), 0, block, -1, components, ghosts});
    resources.push_back(
        {Resource::Kind::state, resources.size(), 0, block, -1, components, ghosts});
  }

  auto& callbacks = explicit_system_program_detail::callbacks();
  const auto callback_identifier = static_cast<std::uint64_t>(callbacks.size());
  callbacks.emplace_back([](explicit_system_program_detail::context_type& context, double dt) {
    context.begin_step(dt);
    context.set_stage_time(0, 1);
    (void)consume_solve_outcome(context.solve_fields());

    std::vector<MultiFab<Dim>*> states;
    std::vector<MultiFab<Dim>*> next_states;
    states.reserve(static_cast<std::size_t>(context.n_blocks()));
    next_states.reserve(static_cast<std::size_t>(context.n_blocks()));
    for (int block = 0; block < context.n_blocks(); ++block) {
      MultiFab<Dim>& state = context.state(block);
      const auto resource_slot = static_cast<runtime::program::ProgramCacheSlot>(2 * block);
      MultiFab<Dim>& residual = context.rhs_scratch(resource_slot, 0, state);
      MultiFab<Dim>& next = context.scratch_state(resource_slot + 1, 0, state);
      context.rhs_into(block, state, residual, 3000 + block);
      context.lincomb(next, Real(1), state, Real(dt), residual);
      states.push_back(&state);
      next_states.push_back(&next);
    }
    for (std::size_t block = 0; block < states.size(); ++block)
      context.lincomb(*states[block], Real(0), *states[block], Real(1), *next_states[block]);
  });
#if !defined(POPS_TEST_TMPDIR)
  throw std::runtime_error("explicit System ABI-v5 fixture requires POPS_TEST_TMPDIR");
#else
  static std::size_t fixture_index = 0;
  const std::string prefix =
      std::string(POPS_TEST_TMPDIR) + "/explicit_system_program_" + std::to_string(++fixture_index);
  const std::string source_path = prefix + ".cpp";
  const std::string library_path = prefix + ".so";
  {
    std::ofstream source(source_path);
    if (!source)
      throw std::runtime_error("cannot create explicit System ABI-v5 fixture source");
    source << program_v5::callback_program_source(
        callback_identifier, "tests.explicit-system/forward-euler@1", "test.clock.macro",
        system.block_names(), resources, "pops_test_explicit_system_program_callback", "uniform");
  }
  const auto compiled = native_dso::compile_shared(source_path, library_path);
  if (!compiled.ok) {
    native_dso::report_compile_failure("explicit_system_program", compiled);
    throw std::runtime_error("explicit System ABI-v5 fixture compilation failed");
  }
  system.install_program(library_path);
#endif
}

}  // namespace pops::test
