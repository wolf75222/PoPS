#pragma once

#include "euler_amr_fixture_criteria.hpp"
#include "explicit_amr_program.hpp"

#include <pops/runtime/dynamic/dynlib.hpp>

namespace pops::test {

template <int Dim>
std::shared_ptr<runtime::program::AmrProgramContext<Dim>> install_scalar_history_fixture(
    AmrSystem<Dim>& system, const std::string& path) {
  if (system.uses_runtime_engine())
    throw std::logic_error("scalar history fixture must install before bootstrap");
  scalar_history.install<Dim>(system);
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
