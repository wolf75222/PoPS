/// @file
/// @brief Explicit diffusion hooks into the existing AMR accepted flux-expression authority.

void neg_div_flux_default_with_faces_into(int program_block, field_type& stage_state,
                                          field_type& rhs, int rate_id,
                                          std::vector<nd::FaceField<Dim>>& faces,
                                          std::string_view temporal_family = {}) const {
  const int runtime_block = sys_block(program_block);
  require_rate_identity_(rate_id);
  require_same_field_contract_(stage_state, rhs, "AMR Program default flux residual");
  const auto point = boundary_evaluation_point(rate_id);
  const auto& evaluation =
      active_attempt_states_.empty()
          ? facade_->evaluate_prepared_amr_block_level_flux_at(runtime_block, point, stage_state)
          : facade_->evaluate_prepared_amr_block_level_flux_at(
                runtime_block, point, stage_state, active_level_ - 1,
                staged_parent_for_block_(runtime_block));
  copy_valid_(evaluation.residual, rhs);
  std::exception_ptr copy_error;
  try {
    faces = evaluation.integrated_face_fluxes;
  } catch (...) {
    copy_error = std::current_exception();
  }
  const ExecutionLane& lane = prepared_execution_lane();
  if (all_reduce_max(copy_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && copy_error)
      std::rethrow_exception(copy_error);
    throw std::runtime_error("AMR retained transport face copy failed collectively");
  }
  if (!active_attempt_states_.empty())
    attach_active_flux_basis_(runtime_block, evaluation, rhs, rate_id,
                              FluxBasisProvider::PreparedDefaultFlux, temporal_family);
  count_kernel_();
}

/// Register the conservative flux corresponding to the prepared constitutive rate. Prepared
/// diffusion supplies G in RHS = +div(G), while the shared reflux ledger represents RHS = -div(F);
/// hence F = -coefficient * G. The ledger applies the metric face measure exactly once.
void attach_diffusive_flux_basis(int program_block, const field_type& rhs, int rate_id,
                                 const std::vector<nd::FaceField<Dim>>& density_faces,
                                 Real coefficient, std::string_view temporal_family = {}) const {
  if (active_attempt_states_.empty())
    return;
  const int runtime_block = sys_block(program_block);
  require_rate_identity_(rate_id);
  require_same_field_contract_(rhs, state(program_block), "AMR diffusive flux RHS");
  const ExecutionLane& lane = prepared_execution_lane();
  std::array<field_type, Dim> integrated;
  std::exception_ptr allocation_error;
  try {
    if (!std::isfinite(coefficient) || !(coefficient > Real(0)) ||
        density_faces.size() != rhs.local_size())
      throw std::invalid_argument(
          "AMR diffusive face payload has an invalid coefficient or layout");
    for (int axis = 0; axis < Dim; ++axis) {
      integrated[axis] =
          field_type(rhs.layout(), rhs.distribution(), rhs.local_rank(), rhs.ncomp(), rhs.ghosts());
      integrated[axis].set_val(Real(0));
    }
  } catch (...) {
    allocation_error = std::current_exception();
  }
  if (all_reduce_max(allocation_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && allocation_error)
      std::rethrow_exception(allocation_error);
    throw std::runtime_error("AMR diffusive face payload allocation failed collectively");
  }
  long layout_error = 0;
  for (std::size_t local = 0; local < rhs.local_size(); ++local)
    if (density_faces[local].cell_box() != rhs.box(local) ||
        density_faces[local].ncomp() != 3 * rhs.ncomp())
      layout_error = 1;
  if (all_reduce_max(layout_error, lane) != 0)
    throw std::invalid_argument(
        "AMR diffusive face payload differs from the active patch collectively");
  for (std::size_t local = 0; local < rhs.local_size(); ++local) {
    const auto density = density_faces[local].view();
    for (int axis = 0; axis < Dim; ++axis) {
      const auto destination = integrated[axis].fab(local).view();
      const Real scale = -coefficient;
      const int components = rhs.ncomp();
      for_each_cell(nd::face_box(rhs.box(local), axis), [=] POPS_HD(const Index<Dim>& face) {
        for (int component = 0; component < components; ++component)
          destination(face, component) = scale * density.axes[axis](face, component);
      });
    }
  }
  device_fence();

  const auto point = boundary_evaluation_point(rate_id);
  const ::pops::amr::ClockWindow interval{
      {active_level_, point.tick, current_interval_begin_phase_, current_interval_start_time_},
      {active_level_, point.tick, current_interval_end_phase_,
       current_interval_start_time_ + current_dt_}};
  FluxExpressionRegistry candidate_registry;
  std::vector<std::size_t> candidate_counts;
  std::uint64_t candidate_identity = 0;
  std::exception_ptr candidate_error;
  try {
    candidate_registry = active_flux_expressions_;
    candidate_counts = active_flux_basis_counts_;
    candidate_identity = next_active_flux_basis_identity_;
  } catch (...) {
    candidate_error = std::current_exception();
  }
  if (all_reduce_max(candidate_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && candidate_error)
      std::rethrow_exception(candidate_error);
    throw std::runtime_error("AMR diffusive flux-expression candidate copy failed collectively");
  }
  prepare_active_flux_basis_impl_(
      runtime_block, point, rate_id, FluxBasisProvider::DiffusiveFace, temporal_family,
      runtime_->topology_epoch(), runtime_->materialization_generation(), rhs, nullptr, &integrated,
      nullptr, interval, candidate_registry, candidate_counts, candidate_identity);
  static_assert(std::is_nothrow_swappable_v<FluxExpressionRegistry>);
  static_assert(std::is_nothrow_swappable_v<std::vector<std::size_t>>);
  active_flux_expressions_.swap(candidate_registry);
  active_flux_basis_counts_.swap(candidate_counts);
  next_active_flux_basis_identity_ = candidate_identity;
}
