#pragma once

#include "euler_amr_fixture_criteria.hpp"
#include "explicit_amr_program.hpp"

#include <pops/runtime/dynamic/dynlib.hpp>

#include <string_view>

namespace pops::test {

template <int Dim>
std::shared_ptr<runtime::program::AmrProgramContext<Dim>> install_scalar_history_fixture(
    AmrSystem<Dim>& system, const std::string& path) {
  if (system.uses_runtime_engine())
    throw std::logic_error("scalar history fixture must install before bootstrap");
  constexpr char clock[] = "clock.macro";
  scalar_history.install<Dim>(system, clock);
  system.install_program(path);
  system.mark_bound();
  const auto library = dynlib::open(path);
  if (library == nullptr)
    throw std::runtime_error("cannot open the installed scalar history fixture");
  using Context = runtime::program::AmrProgramContext<Dim>;
  using Getter = void (*)(AmrSystem<Dim>*, std::shared_ptr<Context>*);
  const auto get = reinterpret_cast<Getter>(dynlib::sym(library, "pops_test_history_context"));
  std::shared_ptr<Context> context;
  try {
    const auto primary_clock = reinterpret_cast<const char* (*)()>(
        dynlib::sym(library, "pops_program_checkpoint_primary_clock_identity"));
    const char* declared_clock = primary_clock != nullptr ? primary_clock() : nullptr;
    if (declared_clock == nullptr || std::string_view(declared_clock) != clock)
      throw std::runtime_error("scalar history criterion clock differs from the installed Program");
    if (get == nullptr)
      throw std::runtime_error("installed scalar history fixture has no Context export");
    get(&system, &context);
  } catch (...) {
    dynlib::close(library);
    throw;
  }
  dynlib::close(library);
  return context;
}

}  // namespace pops::test
