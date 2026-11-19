#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pops::test {

struct FixtureCriterionField {
  const char* state;
  const char* block;
  const char* variable;
  int component;
};

template <std::size_t Size>
struct FixtureCriterionGroup {
  const char* identity;
  const char* provider;
  std::array<FixtureCriterionField, Size> fields;

  const char* state_for(std::string_view block) const {
    for (const auto& field : fields)
      if (field.block == block)
        return field.state;
    throw std::invalid_argument("unregistered Forward Euler fixture block");
  }

  template <int Dim, class Runtime>
  void install(Runtime& system, const char* clock = "test.clock.macro") const {
    std::vector<std::string> states, blocks;
    std::vector<std::vector<std::string>> variables;
    std::vector<std::vector<int>> components;
    for (const auto& field : fields) {
      states.emplace_back(field.state);
      blocks.emplace_back(field.block);
      variables.push_back({field.variable});
      components.push_back({field.component});
    }
    system.set_generated_tagging(identity, states, blocks, variables, components,
                                 {Size, std::vector<std::size_t>(Dim, 0)}, {}, {}, {}, 0, clock,
                                 provider);
  }
};

#include "euler_amr_fixture_criteria_generated.hpp"

}  // namespace pops::test
