#pragma once

#include <string>
#include <type_traits>

namespace pops::test {

template <class F>
int RunTestBody(F fn, const char* test_name) {
  if constexpr (std::is_invocable_r_v<int, F, int, char**>) {
    int argc = 1;
    std::string arg0(test_name);
    char* argv[] = {arg0.data(), nullptr};
    return fn(argc, argv);
  } else {
    return fn();
  }
}

}  // namespace pops::test
