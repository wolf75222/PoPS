/// Joint principal-state preparation through each block's exact AMR ghost authority.
template <std::size_t Count>
void prepare_principal_inputs(const std::array<field_type*, Count>& inputs,
                              const std::array<int, Count>& blocks, int evaluation,
                              const std::array<std::int64_t, Count>& sources) const {
  prepare_spatial_collectively([&] {
    require_rate_identity_(evaluation);
    if constexpr (Count == 0)
      throw std::invalid_argument("principal AMR preparation needs a complete input group");
    for (std::size_t row = 0; row < Count; ++row) {
      if (inputs[row] == nullptr || sources[row] < 0)
        throw std::invalid_argument("principal AMR preparation lost an exact SSA input");
      const int block = sys_block(blocks[row]);
      require_same_field_contract_(*inputs[row], state(blocks[row]),
                                   "principal AMR state input");
      if (facade_->prepared_amr_block_level_active_mask(block, active_level_) != nullptr)
        throw std::invalid_argument("principal AMR transport has no embedded-boundary face authority");
      for (std::size_t other = 0; other < row; ++other)
        if (blocks[row] == blocks[other])
          throw std::invalid_argument("principal AMR input group repeats a block");
    }
  });
  for (std::size_t row = 0; row < Count; ++row) {
    if (active_attempt_states_.empty()) {
      prepare_generated_state(blocks[row], *inputs[row], evaluation);
    } else {
      // A separate exact snapshot per physical row also preserves transformed
      // stage inputs when the parent level has already completed its update.
      const auto trace = capture_rhs_input_trace(blocks[row], *inputs[row], evaluation, sources[row]);
      prepare_generated_state(blocks[row], *inputs[row], evaluation, &trace);
    }
  }
}

/// Register the projection of an already checked joint face evaluation. The
/// ordinary exact coefficient algebra and transactional reflux own acceptance.
void attach_principal_flux(int program_block, field_type& rhs, int evaluation,
                            std::vector<nd::FaceField<Dim>> integrated_faces,
                            std::string_view temporal_family) const {
  const int block = sys_block(program_block);
  std::optional<level_evaluation_type> projected;
  prepare_spatial_collectively([&] {
    require_rate_identity_(evaluation);
    require_same_field_contract_(rhs, state(program_block), "principal AMR projected RHS");
    if (integrated_faces.size() != rhs.local_size())
      throw std::invalid_argument("principal AMR faces do not cover the exact row layout");
    for (std::size_t local = 0; local < rhs.local_size(); ++local)
      if (integrated_faces[local].cell_box() != rhs.box(local) ||
          integrated_faces[local].ncomp() != rhs.ncomp())
        throw std::invalid_argument("principal AMR face projection differs from its physical row");
    projected.emplace();
    projected->point = boundary_evaluation_point(evaluation);
    projected->topology_epoch = runtime_->topology_epoch();
    projected->materialization_generation = runtime_->materialization_generation();
    projected->residual = rhs;
    projected->integrated_face_fluxes = std::move(integrated_faces);
  });
  if (!active_attempt_states_.empty())
    attach_active_flux_basis_(block, *projected, rhs, evaluation,
                              FluxBasisProvider::PreparedDefaultFlux, temporal_family);
}
