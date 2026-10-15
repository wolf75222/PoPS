public:
/// Retain the exact integrated faces produced by a source-authored stencil.
/// No equation, detector, or numerical-method selection occurs in this adapter.
void attach_generated_face_basis(int program_block, const field_type& rhs, int rate_id,
                                 const std::vector<nd::FaceField<Dim>>& faces,
                                 std::string_view temporal_family) const {
  if (active_attempt_states_.empty())
    return;
  const int runtime_block = sys_block(program_block);
  require_rate_identity_(rate_id);
  require_same_field_contract_(rhs, state(program_block), "AMR generated face RHS");
  const ExecutionLane& lane = prepared_execution_lane();
  std::array<field_type, Dim> integrated;
  prepare_spatial_collectively([&] {
    if (faces.size() != rhs.local_size())
      throw std::invalid_argument("AMR generated faces differ from the local patch count");
    for (int axis = 0; axis < Dim; ++axis) {
      if (rhs.ghosts()[axis] < 1)
        throw std::invalid_argument("AMR generated face storage needs an upper-face slot");
      integrated[axis] =
          field_type(rhs.layout(), rhs.distribution(), rhs.local_rank(), rhs.ncomp(), rhs.ghosts());
      integrated[axis].set_val(Real(0));
    }
    for (std::size_t local = 0; local < rhs.local_size(); ++local)
      if (faces[local].cell_box() != rhs.box(local) || faces[local].ncomp() != rhs.ncomp())
        throw std::invalid_argument("AMR generated faces differ from the exact RHS layout");
  });
  std::exception_ptr copy_error;
  try {
    for (std::size_t local = 0; local < rhs.local_size(); ++local) {
      const auto source = faces[local].view();
      for (int axis = 0; axis < Dim; ++axis) {
        const auto destination = integrated[axis].fab(local).view();
        const auto input = source.axes[axis];
        const int components = rhs.ncomp();
        for_each_cell(nd::face_box(rhs.box(local), axis), [=] POPS_HD(const Index<Dim>& face) {
          for (int component = 0; component < components; ++component)
            destination(face, component) = input(face, component);
        });
      }
    }
    device_fence();
  } catch (...) {
    copy_error = std::current_exception();
    try {
      device_fence();
    } catch (...) {
    }
  }
  if (all_reduce_max(copy_error ? 1L : 0L, lane)) {
    if (lane.size() == 1 && copy_error)
      std::rethrow_exception(copy_error);
    throw std::runtime_error("AMR generated face copy failed collectively");
  }
  const auto point = boundary_evaluation_point(rate_id);
  const ::pops::amr::ClockWindow interval{
      {active_level_, point.tick, current_interval_begin_phase_, current_interval_start_time_},
      {active_level_, point.tick, current_interval_end_phase_,
       current_interval_start_time_ + current_dt_}};
  FluxExpressionRegistry candidate_registry;
  std::vector<std::size_t> candidate_counts;
  std::uint64_t candidate_identity = 0;
  prepare_spatial_collectively([&] {
    candidate_registry = active_flux_expressions_;
    candidate_counts = active_flux_basis_counts_;
    candidate_identity = next_active_flux_basis_identity_;
  });
  prepare_active_flux_basis_impl_(
      runtime_block, point, rate_id, FluxBasisProvider::GeneratedFace, temporal_family,
      runtime_->topology_epoch(), runtime_->materialization_generation(), rhs, nullptr, &integrated,
      nullptr, interval, candidate_registry, candidate_counts, candidate_identity);
  static_assert(std::is_nothrow_swappable_v<FluxExpressionRegistry>);
  static_assert(std::is_nothrow_swappable_v<std::vector<std::size_t>>);
  active_flux_expressions_.swap(candidate_registry);
  active_flux_basis_counts_.swap(candidate_counts);
  next_active_flux_basis_identity_ = candidate_identity;
}

private:
