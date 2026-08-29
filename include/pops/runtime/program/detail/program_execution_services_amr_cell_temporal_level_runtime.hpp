
class CellTemporalLevelRuntime {
 public:
  static constexpr int dimension = Dim;

  CellTemporalLevelRuntime(const AmrStorageTopologyAdapter& owner,
                           const CellTemporalConfiguration& configuration, int level)
      : owner_(&owner), configuration_(&configuration), level_(level) {
    const BoundaryTopology<Dim> topology = owner.facade_->program_prepared_amr_boundary_topology_();
    for (int axis = 0; axis < Dim; ++axis)
      periodicity_[static_cast<std::size_t>(axis)] =
          topology.is_periodic(Face<Dim>{axis, BoundarySide::lower}) &&
          topology.is_periodic(Face<Dim>{axis, BoundarySide::upper});
    integrated_flux_.reserve(configuration.routes.size());
    final_residuals_.assign(configuration.routes.size(), nullptr);
    evaluations_.assign(configuration.routes.size(), nullptr);
    for (const auto& route : configuration.routes) {
      const field_type& state =
          *owner.active_attempt_states_[static_cast<std::size_t>(route.runtime_block)];
      std::array<field_type, Dim> route_flux{};
      for (int axis = 0; axis < Dim; ++axis) {
        route_flux[static_cast<std::size_t>(axis)] = same_level_cell_temporal_detail::field_like(
            state, same_level_cell_temporal_detail::face_boxes(state.layout(), axis),
            Extent<Dim>{});
        route_flux[static_cast<std::size_t>(axis)].set_val(Real(0));
      }
      integrated_flux_.push_back(std::move(route_flux));
    }
  }

  [[nodiscard]] std::uint64_t topology_epoch() const noexcept {
    return owner_->runtime_->topology_epoch();
  }
  [[nodiscard]] std::uint64_t materialization_generation() const noexcept {
    return owner_->runtime_->materialization_generation();
  }
  [[nodiscard]] std::size_t same_level_cell_route_count() const noexcept {
    return configuration_->routes.size();
  }
  [[nodiscard]] int same_level_cell_level_count() const noexcept { return owner_->nlev(); }
  [[nodiscard]] int same_level_cell_active_level() const noexcept { return level_; }
  [[nodiscard]] std::uint64_t same_level_cell_level_cell_count(int level) const noexcept {
    if (level < 0 || static_cast<std::size_t>(level) >= configuration_->level_cell_counts.size())
      return 0;
    return configuration_->level_cell_counts[static_cast<std::size_t>(level)];
  }
  [[nodiscard]] int same_level_cell_runtime_block(std::size_t route) const noexcept {
    return configuration_->routes[route].runtime_block;
  }
  [[nodiscard]] int same_level_cell_program_block(std::size_t route) const noexcept {
    return configuration_->routes[route].program_block;
  }
  [[nodiscard]] int same_level_cell_rhs_id(std::size_t route) const noexcept {
    return configuration_->routes[route].rhs_id;
  }
  [[nodiscard]] field_type& same_level_cell_state(std::size_t route) noexcept {
    return *owner_->active_attempt_states_[static_cast<std::size_t>(
        configuration_->routes[route].runtime_block)];
  }
  [[nodiscard]] Geometry<Dim> same_level_cell_geometry() const {
    return owner_->facade_->program_prepared_amr_level_geometry_(level_);
  }
  [[nodiscard]] const std::array<bool, Dim>& same_level_cell_periodicity() const noexcept {
    return periodicity_;
  }
  [[nodiscard]] std::string_view same_level_cell_state_identity(std::size_t route) const {
    return owner_
        ->active_block_identities_[static_cast<std::size_t>(same_level_cell_runtime_block(route))];
  }
  [[nodiscard]] std::string_view same_level_cell_flux_provider_identity(std::size_t) const {
    return kSameLevelTransportEulerStageFluxProvider;
  }
  [[nodiscard]] std::string_view same_level_cell_flux_parameter_contract(std::size_t) const {
    return configuration_->exact_contract;
  }
  [[nodiscard]] std::string_view same_level_cell_stage_snapshot_contract(std::size_t) const {
    return configuration_->exact_contract;
  }

  [[nodiscard]] multiblock::BoundaryEvaluationPoint same_level_cell_evaluation_point(
      CellTemporalRungBatchDescriptor batch) const {
    if (batch.end_tick <= batch.begin_tick || batch.tick_denominator <= 0)
      throw std::invalid_argument("cell-local AMR batch has an invalid exact clock window");
    const std::int64_t interval_ticks =
        owner_->cell_temporal_interval_target_tick_ - owner_->cell_temporal_interval_begin_tick_;
    if (interval_ticks <= 0)
      throw std::logic_error("cell-local AMR interval has no exact tick extent");
    const auto relative_phase = [&](std::int64_t tick) {
      return ::pops::amr::Rational{tick - owner_->cell_temporal_interval_begin_tick_,
                                   interval_ticks};
    };
    const auto begin_phase = relative_phase(batch.begin_tick);
    const auto end_phase = relative_phase(batch.end_tick);
    const double physical_begin =
        owner_->current_interval_start_time_ + begin_phase.value() * owner_->current_dt_;
    const double batch_dt = (end_phase - begin_phase).value() * owner_->current_dt_;
    return {.clock = configuration_->clock,
            .tick = owner_->active_subcycling_window_.begin.macro_step,
            .level = level_,
            .substep = owner_->logical_substep_,
            .stage = 0,
            .stage_fraction = {0, 1},
            .dt = batch_dt,
            .physical_time = physical_begin};
  }

  void prepare_same_level_cell_stage_snapshot(std::size_t route,
                                              const multiblock::BoundaryEvaluationPoint& point,
                                              field_type& snapshot, const ExecutionLane& lane) {
    owner_->require_prepared_lane_(lane, "cell-local AMR halo snapshot");
    const int block = same_level_cell_runtime_block(route);
    owner_->facade_->prepare_generated_amr_block_level_state(
        block, point, snapshot, level_ - 1, owner_->staged_parent_for_block_(block));
  }

  void capture_same_level_negative_flux_divergence(std::size_t route,
                                                   const multiblock::BoundaryEvaluationPoint& point,
                                                   const field_type& immutable_snapshot,
                                                   field_type& residual,
                                                   const std::array<field_type*, Dim>& fluxes) {
    const int block = same_level_cell_runtime_block(route);
    auto& stage = const_cast<field_type&>(immutable_snapshot);
    const auto& evaluation = owner_->facade_->program_evaluate_prepared_amr_block_level_flux_at_(
        block, point, stage, level_ - 1, owner_->staged_parent_for_block_(block));
    std::exception_ptr local_error;
    try {
      owner_->copy_valid_(evaluation.residual, residual);
      if (evaluation.integrated_face_fluxes.size() != residual.local_size())
        throw std::logic_error("cell-local AMR evaluation lost its local face fluxes");
      for (int axis = 0; axis < Dim; ++axis) {
        field_type& destination = *fluxes[static_cast<std::size_t>(axis)];
        for (std::size_t local = 0; local < destination.local_size(); ++local)
          copy_face_axis_(axis, evaluation.integrated_face_fluxes[local], destination.fab(local));
      }
      device_fence();
    } catch (...) {
      local_error = std::current_exception();
    }
    const ExecutionLane& lane = owner_->prepared_execution_lane();
    if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("cell-local AMR face-flux extraction failed collectively");
    }
    evaluations_[route] = &evaluation;
  }

  void prepare_same_level_cell_flux_metadata(
      std::span<const SameLevelCellTemporalRouteCandidate<Dim>> candidates) {
    if (candidates.size() != configuration_->routes.size() ||
        evaluations_.size() != candidates.size())
      throw std::logic_error("cell-local AMR finalize lost a route evaluation");
    for (std::size_t route = 0; route < candidates.size(); ++route) {
      const auto& candidate = candidates[route];
      if (candidate.route != route || candidate.source == nullptr ||
          candidate.residual == nullptr || candidate.integrated_face_fluxes == nullptr ||
          candidate.candidate == nullptr || !(candidate.dt > Real(0)) ||
          candidate.begin_tick >= candidate.end_tick ||
          candidate.tick_denominator != configuration_->tick_denominator)
        throw std::invalid_argument("cell-local AMR finalize received a foreign route candidate");
      for (int axis = 0; axis < Dim; ++axis)
        pops::saxpy(integrated_flux_[route][static_cast<std::size_t>(axis)], candidate.dt,
                    (*candidate.integrated_face_fluxes)[static_cast<std::size_t>(axis)]);
      final_residuals_[route] = candidate.residual;
    }
  }

  void publish_same_level_cell_flux_metadata() noexcept {
    // Per-rung accumulation remains attempt-local. The complete, single-basis route pack is
    // prepared once by finalize_same_level_cell_flux_metadata at the synchronization barrier.
  }
  void prepare_same_level_cell_attempt_finalize_local() {
    if (evaluations_.size() != configuration_->routes.size() ||
        final_residuals_.size() != configuration_->routes.size())
      throw std::logic_error("cell-local AMR final flux route pack is incomplete");
    const Real interval_dt = static_cast<Real>(owner_->current_dt_);
    if (!(interval_dt > Real(0)))
      throw std::logic_error("cell-local AMR final flux interval has invalid duration");
    local_flux_expressions_.emplace(owner_->active_flux_expressions_);
    local_flux_counts_.emplace(owner_->active_flux_basis_counts_);
    local_next_identity_ = owner_->next_active_flux_basis_identity_;
    for (std::size_t route = 0; route < configuration_->routes.size(); ++route) {
      if (evaluations_[route] == nullptr || final_residuals_[route] == nullptr)
        throw std::logic_error("cell-local AMR final flux lost its route evaluation");
      for (int axis = 0; axis < Dim; ++axis)
        pops::scale(integrated_flux_[route][static_cast<std::size_t>(axis)], Real(1) / interval_dt);
    }
    device_fence();
  }

  void finalize_same_level_cell_flux_metadata() {
    if (!local_flux_expressions_ || !local_flux_counts_)
      throw std::logic_error("cell-local AMR final flux lacks local preparation");
    FluxExpressionRegistry& candidate_registry = *local_flux_expressions_;
    std::vector<std::size_t>& candidate_counts = *local_flux_counts_;
    std::uint64_t candidate_identity = local_next_identity_;
    const ExecutionLane& lane = owner_->prepared_execution_lane();
    std::exception_ptr local_error;
    for (std::size_t route = 0; route < configuration_->routes.size(); ++route) {
      owner_->prepare_cell_temporal_flux_basis_(
          same_level_cell_runtime_block(route), *evaluations_[route], *final_residuals_[route],
          same_level_cell_rhs_id(route), integrated_flux_[route],
          owner_->cell_temporal_interval_begin_tick_, owner_->cell_temporal_interval_target_tick_,
          candidate_registry, candidate_counts, candidate_identity);
      local_error = nullptr;
      try {
        FluxExpression expression = owner_->scaled_flux_expression_(
            candidate_registry.at(final_residuals_[route]), ExactPolynomial{{1, {1, 1}}});
        owner_->require_flux_expression_budget_(expression);
        candidate_registry[&same_level_cell_state(route)] = std::move(expression);
      } catch (...) {
        local_error = std::current_exception();
      }
      if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
        if (lane.size() == 1 && local_error)
          std::rethrow_exception(local_error);
        throw std::runtime_error("cell-local AMR final route expression failed collectively");
      }
    }
    prepared_flux_expressions_.emplace(std::move(candidate_registry));
    prepared_flux_counts_.emplace(std::move(candidate_counts));
    prepared_next_identity_ = candidate_identity;
    local_flux_expressions_.reset();
    local_flux_counts_.reset();
  }
  void commit_same_level_cell_flux_metadata() noexcept {
    if (!prepared_flux_expressions_ || !prepared_flux_counts_)
      std::terminate();
    owner_->active_flux_expressions_.swap(*prepared_flux_expressions_);
    owner_->active_flux_basis_counts_.swap(*prepared_flux_counts_);
    owner_->next_active_flux_basis_identity_ = prepared_next_identity_;
    prepared_flux_expressions_.reset();
    prepared_flux_counts_.reset();
    local_flux_expressions_.reset();
    local_flux_counts_.reset();
    std::fill(evaluations_.begin(), evaluations_.end(), nullptr);
  }
  void discard_same_level_cell_flux_metadata() noexcept {
    prepared_flux_expressions_.reset();
    prepared_flux_counts_.reset();
    local_flux_expressions_.reset();
    local_flux_counts_.reset();
    prepared_next_identity_ = 0;
    local_next_identity_ = 0;
    std::fill(evaluations_.begin(), evaluations_.end(), nullptr);
    std::fill(final_residuals_.begin(), final_residuals_.end(), nullptr);
  }

 private:
  template <int Axis = 0>
  static void copy_face_axis_(int axis, const nd::FaceField<Dim>& source, Fab<Dim>& destination) {
    if constexpr (Axis < Dim) {
      if (axis == Axis) {
        Kokkos::deep_copy(destination.storage(), source.template field<Axis>().storage());
        return;
      }
      copy_face_axis_<Axis + 1>(axis, source, destination);
    } else {
      throw std::out_of_range("cell-local AMR face axis is outside the exact rank");
    }
  }

  const AmrStorageTopologyAdapter* owner_ = nullptr;
  const CellTemporalConfiguration* configuration_ = nullptr;
  int level_ = 0;
  std::array<bool, Dim> periodicity_{};
  std::vector<const level_evaluation_type*> evaluations_;
  std::vector<std::array<field_type, Dim>> integrated_flux_;
  std::vector<const field_type*> final_residuals_;
  std::optional<FluxExpressionRegistry> prepared_flux_expressions_;
  std::optional<std::vector<std::size_t>> prepared_flux_counts_;
  std::uint64_t prepared_next_identity_ = 0;
  std::optional<FluxExpressionRegistry> local_flux_expressions_;
  std::optional<std::vector<std::size_t>> local_flux_counts_;
  std::uint64_t local_next_identity_ = 0;
};
