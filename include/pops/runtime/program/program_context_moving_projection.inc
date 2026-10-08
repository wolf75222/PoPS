// Included in ProgramContext's public section. User physical expressions and
// quadrature policies are device functors, not runtime-recognized model names.
  template <int Components, class CoordinateLaw, class FaceConstruction, class CellConstruction>
  RuntimeIntervalEvaluation<Dim> project_moving_interval(
      const std::string& identity, int program_block, const std::string& physical_frame,
      const std::string& quadrature, CoordinateLaw coordinate_law,
      FaceConstruction face_construction, CellConstruction cell_construction,
      Real geometry_tolerance) const {
    const auto& lane = prepared_execution_lane();
    std::optional<field_type> sampled;
    std::exception_ptr error;
    try {
      if constexpr (Dim != 1) {
        throw std::invalid_argument("moving interval projection needs prepared 1D geometry");
      } else {
        const auto& input = state(program_block);
        if (input.ncomp() != Components || input.ghosts()[0] < 1 ||
            !std::isfinite(geometry_tolerance) || geometry_tolerance < Real(0))
          throw std::invalid_argument("moving interval projection needs exact state components, "
                                      "one halo and finite nonnegative geometry tolerance");
        sampled.emplace(input);
      }
    } catch (...) { error = std::current_exception(); }
    collectively_rethrow_exception(error, lane, "moving interval sampling failed collectively");
    // The existing prepared boundary session exchanges the real owned state.
    // It completes before the local-only interval producer starts device work.
    fill_boundary(*sampled);
    return evaluate_moving_interval(identity, program_block, physical_frame, quadrature,
        [&](const runtime::multiblock::BoundaryEvaluationPoint& point) -> MovingIntervalInputs<Dim> {
      if constexpr (Dim == 1) {
        const auto& previous = moving_interval_geometry(identity);
        const auto reference = geometry();
        const MovingIntervalTime time{static_cast<Real>(point.physical_time),
            static_cast<Real>(point.physical_time + point.dt), static_cast<Real>(point.dt)};
        MovingIntervalInputs<Dim> values;
        values.coordinates = previous.coordinates;
        values.swept_volumes = previous.swept_volumes;
        values.source = scalar_field_like_(*sampled, Components, 0);
        for (std::size_t patch = 0; patch < sampled->local_size(); ++patch) {
          const auto box = sampled->box(patch);
          values.physical_flux.emplace_back(box, Components);
          values.face_density.emplace_back(box, Components);
          const auto input = std::as_const(*sampled).fab(patch).view();
          const auto old_position = previous.coordinates[patch].template field<0>().view();
          const auto old_measure = previous.measures.fab(patch).view();
          const auto position = values.coordinates[patch].template field<0>().view();
          const auto sweep = values.swept_volumes[patch].template field<0>().view();
          const auto physical = values.physical_flux[patch].template field<0>().view();
          const auto density = values.face_density[patch].template field<0>().view();
          const auto source = values.source.fab(patch).view();
          const auto invalid_law = for_each_cell_reduce_max(nd::face_box(box, 0),
              [=] POPS_HD(const Index<Dim>& face) {
            const Real ref = reference.face_coordinate(0, face[0]);
            const Real begin = coordinate_law(ref, time.begin);
            const Real end = coordinate_law(ref, time.end);
            position(face) = end;
            sweep(face) = end-old_position(face);
            return !Kokkos::isfinite(begin) || !Kokkos::isfinite(end) ||
                Kokkos::abs(begin-old_position(face)) > geometry_tolerance ? Real(1) : Real(0);
          });
          if (invalid_law != Real(0))
            throw StepAttemptRejected(SolveStatus::kInvalidEvaluation, "moving-coordinate-law",
                                      "coordinate law does not continue the accepted geometry");
          for_each_cell(nd::face_box(box, 0), [=] POPS_HD(const Index<Dim>& face) {
            Index<Dim> left = face; --left[0];
            StateVec<Components> lower{}, upper{};
            for (int component = 0; component < Components; ++component) {
              lower[component] = input(left, component);
              upper[component] = input(face, component);
            }
            const MovingFaceGeometry face_geometry{
                reference.face_coordinate(0, face[0]), old_position(face), position(face), sweep(face)};
            const auto result = face_construction(lower, upper, face_geometry, time);
            for (int component = 0; component < Components; ++component) {
              physical(face, component) = result.physical_amount[component];
              density(face, component) = result.density[component];
            }
          });
          for_each_cell(box, [=] POPS_HD(const Index<Dim>& cell) {
            Index<Dim> right = cell; ++right[0];
            StateVec<Components> value{};
            for (int component = 0; component < Components; ++component)
              value[component] = input(cell, component);
            const MovingCellGeometry cell_geometry{reference.cell_coordinate(0, cell[0]),
                (old_position(cell)+old_position(right))/Real(2),
                (position(cell)+position(right))/Real(2), old_measure(cell),
                position(right)-position(cell)};
            const auto result = cell_construction(value, cell_geometry, time);
            for (int component = 0; component < Components; ++component)
              source(cell, component) = result[component];
          });
        }
        return values;
      } else {
        throw std::invalid_argument("moving interval projection needs prepared 1D geometry");
      }
    });
  }
