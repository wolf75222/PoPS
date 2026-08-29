
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
