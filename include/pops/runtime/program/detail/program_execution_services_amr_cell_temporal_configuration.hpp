
#pragma once

#include <pops/runtime/program/cell_temporal_partition.hpp>
#include <pops/runtime/program/same_level_cell_temporal_provider.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace pops::runtime::program {

struct CellTemporalConfiguration {
  std::string clock;
  std::int64_t tick_denominator = 1;
  /// Authored rung of the finest configured level. Coarser rungs are derived from the exact
  /// power-of-two parent/child clock relations so every level takes one FE batch per window.
  int rung = 0;
  std::vector<int> level_rungs;
  std::vector<SameLevelCellTemporalForwardEulerRoute> routes;
  std::vector<std::uint64_t> level_cell_counts;
  std::uint64_t topology_epoch = 0;
  std::uint64_t materialization_generation = 0;
  std::string exact_contract;
};

/// Fully prepared, detached cell-temporal state.  The DSO builds this from the immutable
/// preparation topology; ProgramExecutionPreparationImage owns it until the accepted facade is
/// bound, at which point adoption is a no-throw exchange of already allocated state.
template <int Dim>
struct PreparedCellTemporalExecution final {
  CellTemporalConfiguration configuration;
  CellTemporalPartitionAcceptedState partition;
  /// Three disjoint candidate-prepared value arenas.  The workspace is consumed by hot level
  /// groups; accepted and rollback pools remain invisible until transaction publication/restore.
  /// Keeping them explicit avoids both publication-time allocation and candidate-state escape.
  std::vector<std::shared_ptr<SameLevelCellIntegratedFluxPackDiagnostic<Dim>>>
      diagnostic_workspace;
  std::vector<std::shared_ptr<SameLevelCellIntegratedFluxPackDiagnostic<Dim>>>
      accepted_diagnostics;
  std::vector<std::shared_ptr<SameLevelCellIntegratedFluxPackDiagnostic<Dim>>>
      rollback_diagnostics;
};

}  // namespace pops::runtime::program
