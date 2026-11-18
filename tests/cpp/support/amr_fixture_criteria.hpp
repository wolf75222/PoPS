#pragma once

#include <cstddef>
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pops::test {

struct FixtureCriterion {
  const char* identity;
  const char* state;
  const char* block;
  const char* variable;
  const char* clock;
  const char* provider;
  int component;

  std::string program_hash(const std::string& base) const { return base + "/" + identity; }

  template <int Dim, class Runtime>
  void install(Runtime& system) const {
    system.set_generated_tagging(identity, {state}, {block}, {{variable}}, {{component}},
                                 {std::vector<std::size_t>(Dim, 0)}, {}, {}, {}, 0, clock,
                                 provider);
  }
};

inline void append_fixture_criterion(std::ostream& output, const char* path) {
  std::ifstream fragment(path);
  if (!fragment)
    throw std::runtime_error("cannot read generated AMR fixture criterion");
  output << '\n' << fragment.rdbuf();
}

#include "amr_fixture_criteria_generated.hpp"

}  // namespace pops::test
