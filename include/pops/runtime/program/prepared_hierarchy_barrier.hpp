#pragma once

#include <pops/core/identity/prepared_provider.hpp>
#include <pops/parallel/collective_exception.hpp>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace pops::runtime::program {
/// Exact operation identity and collective agreement for closed hierarchy continuations.
/// Contexts authenticate attempts, owner candidates and clocks before admitting these ports.
class PreparedHierarchyBarrier {
 public:
  enum class PortKind : std::uint8_t {
    complete,
    map,
    linear_solve,
    field_publication,
    spatial_rhs,
    named_field_solve,
    field_warm_start,
    nonlinear_solve,
    control
  };

  template <class RootWindow>
  static std::string operation_contract(std::uint8_t kind, std::int64_t operation_id,
                                        std::string_view program_hash,
                                        std::string_view dependency_contract,
                                        std::int64_t numerator, std::int64_t denominator,
                                        std::uint64_t attempt, std::uint64_t sequence,
                                        std::string_view hierarchy_authority,
                                        const RootWindow& window) {
    ExactContractBuilder contract;
    contract.text("pops.amr-hierarchy-barrier.v1")
        .scalar(kind)
        .scalar(operation_id)
        .text(program_hash)
        .bytes(dependency_contract)
        .scalar(numerator)
        .scalar(denominator)
        .scalar(attempt)
        .scalar(sequence)
        .bytes(hierarchy_authority);
    // Level ordinals do not distinguish operations: every port names the common root window.
    for (const auto& clock : {window.begin, window.end})
      contract.scalar(std::int32_t{clock.level})
          .scalar(clock.macro_step)
          .scalar(clock.phase.numerator)
          .scalar(clock.phase.denominator)
          .scalar(clock.physical_time);
    return std::move(contract).release();
  }

  template <class Ports, class Fields>
  static void agree_ports(const Ports& ports, Fields& fields, const ExecutionLane& lane) {
    const auto& first = ports.front();
    const bool map = first.kind == PortKind::map;
    const bool internal =
        first.kind == PortKind::linear_solve || first.kind == PortKind::field_publication ||
        first.kind == PortKind::spatial_rhs || first.kind == PortKind::named_field_solve ||
        first.kind == PortKind::field_warm_start || first.kind == PortKind::nonlinear_solve ||
        first.kind == PortKind::control;
    std::exception_ptr error;
    try {
      for (std::size_t level = 0; level < ports.size(); ++level) {
        const auto& port = ports[level];
        if (port.kind != first.kind || port.identity != first.identity ||
            (map && port.target != first.target) || (port.field != nullptr) != map ||
            static_cast<bool>(port.next) != (map || internal) ||
            static_cast<bool>(port.once) != internal ||
            ((map || internal) && port.identity.empty()) ||
            (!map && !internal && !port.identity.empty()))
          throw std::logic_error("AMR levels reached different qualified continuation barriers");
        fields[level] = port.field;
      }
    } catch (...) {
      error = std::current_exception();
    }
    collectively_rethrow_exception(error, lane,
                                   "AMR continuation barrier agreement failed collectively");
    const auto category = map ? (first.target ? "map-target" : "map-source")
                          : internal
                              ? (first.kind == PortKind::linear_solve        ? "linear-solve"
                                 : first.kind == PortKind::field_publication ? "field-publication"
                                 : first.kind == PortKind::spatial_rhs       ? "spatial-rhs"
                                 : first.kind == PortKind::named_field_solve ? "named-field-solve"
                                 : first.kind == PortKind::nonlinear_solve   ? "nonlinear-solve"
                                 : first.kind == PortKind::control           ? "control"
                                                                             : "field-warm-start")
                              : "complete";
    const std::array<std::pair<std::string_view, std::string_view>, 2> contracts{
        {{"amr-continuation-contract", first.identity}, {"amr-continuation-kind", category}}};
    if (!all_ranks_agree_exact_ordered_byte_pairs(contracts, lane))
      throw std::logic_error("AMR continuation barrier differs between ranks");
  }
};
}  // namespace pops::runtime::program
