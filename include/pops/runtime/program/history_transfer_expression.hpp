#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pops::runtime::program {

/// Source basis numbers are attempt-local: equal integers in different ring
/// slots do not establish equal evaluations. Project each (slot, basis) once
/// into a unique immutable namespace before applying a temporal matrix.
template <class Expression, class Project>
std::vector<Expression> distinct_history_flux_sources(const std::vector<Expression>& source,
                                                      Project&& project) {
  std::vector<Expression> result;
  result.reserve(source.size());
  std::uint64_t next = 0;
  for (const auto& expression : source) {
    Expression projected;
    for (const auto& [identity, term] : expression) {
      (void)identity;
      if (!term.basis)
        throw std::invalid_argument("HistoryTransfer source has no immutable flux basis");
      if (next == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("HistoryTransfer source identity space exhausted");
      auto basis = project(term.basis);
      if (!basis)
        throw std::invalid_argument("HistoryTransfer projection returned no flux basis");
      basis->identity = next;
      projected.emplace(next++,
                        typename Expression::mapped_type{std::move(basis), term.coefficient});
    }
    result.push_back(std::move(projected));
  }
  return result;
}

}  // namespace pops::runtime::program
