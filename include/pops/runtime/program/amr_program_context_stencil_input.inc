public:
/// A stencil observes an exact consumed physical solve, never the unfilled
/// ghosts of its component-observation buffer. The provider owns CF and BCs.
const field_type& prepare_stencil_field_solution(std::int64_t key, const std::string& identity,
                                                 const field_type& observation, int component,
                                                 std::int64_t consumer, int input,
                                                 const std::array<int, Dim>& requested) const {
  auto& solver = configured_hierarchy_field_solver_(key);
  const auto& lane = prepared_execution_lane();
  const auto& source = solver.solution(active_level_);
  Extent<Dim> extent;
  prepare_spatial_collectively([&] {
    if (identity != hierarchy_field_resources_.at(key).field_identity || consumer < 0 ||
        input < 0 || component < 0 || component >= source.ncomp() || observation.ncomp() != 1 ||
        observation.layout() != source.layout() ||
        observation.distribution() != source.distribution() ||
        observation.local_rank() != source.local_rank())
      throw std::invalid_argument("stencil field lost its exact solve/component authority");
    for (int axis = 0; axis < Dim; ++axis) {
      if (requested[axis] < 0 || requested[axis] > source.ghosts()[axis])
        throw std::invalid_argument("stencil field extension exceeds its provider's prepared halo");
      extent[axis] = requested[axis];
    }
  });
  Real mismatch = 0;
  prepare_spatial_collectively([&] {
    for (std::size_t patch = 0; patch < observation.local_size(); ++patch) {
      const auto observed = observation.fab(patch).view();
      const auto solved = source.fab(patch).view();
      mismatch = std::max(
          mismatch,
          for_each_cell_reduce_max(observation.box(patch), [=] POPS_HD(const Index<Dim>& cell) {
            return observed(cell, 0) == solved(cell, component) ? Real(0) : Real(1);
          }));
    }
    device_fence();
  });
  if (all_reduce_max(mismatch, lane) != Real(0))
    throw std::invalid_argument(
        "stencil field observation is stale relative to its consumed solve");
  return solver.prepare_solution_extension(active_level_, extent, lane);
}

private:
