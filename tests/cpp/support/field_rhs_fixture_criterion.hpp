#pragma once

#include "amr_fixture_criteria.hpp"

namespace pops::test {

// Both Field RHS sources declare one identity literal shared by the exported Program,
// the SSA plan and its barriers. Compose at that fixture boundary before compilation.
inline std::string qualify_fixture_program_source(std::string source, const std::string& base,
                                                  const FixtureCriterion& criterion) {
  const auto literal = "\"" + base + "\"";
  const auto position = source.find(literal);
  if (position == std::string::npos ||
      source.find(literal, position + literal.size()) != std::string::npos)
    throw std::runtime_error("expected one Field RHS Program identity literal");
  source.replace(position, literal.size(), "\"" + criterion.program_hash(base) + "\"");
  return source;
}

}  // namespace pops::test
