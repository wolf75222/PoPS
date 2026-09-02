
class CellTemporalLevelRuntime {
 public:
  static constexpr int dimension = Dim;

  CellTemporalLevelRuntime(const AmrStorageTopologyAdapter& owner,
                           const CellTemporalConfiguration& configuration, int level)
      : owner_(&owner),
        configuration_(&configuration),
        level_(level),
        configured_route_count_(configuration.routes.size()),
        configured_level_count_(configuration.level_rungs.size()),
        configured_tick_denominator_(configuration.tick_denominator),
        configured_level_rung_(level >= 0 && static_cast<std::size_t>(level) <
                                                 configuration.level_rungs.size()
                                   ? configuration.level_rungs[static_cast<std::size_t>(level)]
                                   : -1),
        configured_topology_epoch_(configuration.topology_epoch),
        configured_materialization_generation_(configuration.materialization_generation) {
    if (level < 0 || level >= owner.nlev())
      throw std::invalid_argument("cell-local AMR resident level is outside its prepared topology");
    if (owner.preparation_view_ != nullptr) {
      if (owner.preparation_view_->periodic_faces.size() != static_cast<std::size_t>(2 * Dim))
        throw std::logic_error("cell-local AMR preparation image has no periodic-face authority");
    }
    if (owner.preparation_view_ != nullptr) {
      for (int axis = 0; axis < Dim; ++axis)
        periodicity_[static_cast<std::size_t>(axis)] =
            owner.preparation_view_->periodic_faces.at(static_cast<std::size_t>(2 * axis)) &&
            owner.preparation_view_->periodic_faces.at(static_cast<std::size_t>(2 * axis + 1));
    } else {
      // The facade returns a value authority.  Keep that value alive while reading it rather
      // than retaining the address of a temporary; candidate preparation never enters this
      // accepted-runtime branch.
      const BoundaryTopology<Dim> topology =
          owner.facade_->program_prepared_amr_boundary_topology_();
      for (int axis = 0; axis < Dim; ++axis)
        periodicity_[static_cast<std::size_t>(axis)] =
            topology.is_periodic(Face<Dim>{axis, BoundarySide::lower}) &&
            topology.is_periodic(Face<Dim>{axis, BoundarySide::upper});
    }
    integrated_flux_.reserve(configuration.routes.size());
    final_residuals_.assign(configuration.routes.size(), nullptr);
    evaluations_.assign(configuration.routes.size(), nullptr);
    // The provider borrows this point throughout each warmed batch.  Bind the owned clock once
    // during resident construction; batch execution below only overwrites scalar coordinates.
    evaluation_point_.clock = configuration.clock;
    for (const auto& route : configuration.routes) {
      const field_type& state = state_for_route_(route.runtime_block);
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
  /// Exact retained payload of the runtime facade shared by one level's provider/executor pair.
  /// The provider's candidate/snapshot fields are charged by the provider; this method covers
  /// the level-owned integrated face image and its fixed pointer/evaluation tables.
  [[nodiscard]] std::uint64_t resident_storage_bytes() const {
    const auto checked_add = [](std::uint64_t& total, std::uint64_t value) {
      if (value > std::numeric_limits<std::uint64_t>::max() - total)
        throw std::overflow_error("cell-temporal level runtime resident storage overflows uint64");
      total += value;
    };
    const auto vector_bytes = [](const auto& values) -> std::uint64_t {
      using value_type = typename std::remove_reference_t<decltype(values)>::value_type;
      if constexpr (requires { values.capacity(); }) {
        if (values.capacity() > std::numeric_limits<std::uint64_t>::max() / sizeof(value_type))
          throw std::overflow_error("cell-temporal level runtime vector storage overflows uint64");
        return static_cast<std::uint64_t>(values.capacity()) * sizeof(value_type);
      }
      return 0;
    };
    std::uint64_t total = 0;
    checked_add(total, vector_bytes(integrated_flux_));
    checked_add(total, vector_bytes(final_residuals_));
    checked_add(total, vector_bytes(evaluations_));
    for (const auto& route_flux : integrated_flux_) {
      checked_add(total, vector_bytes(route_flux));
      for (const field_type& flux : route_flux)
        checked_add(total, flux.resident_storage_bytes());
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(&evaluation_point_.clock);
    const auto end = begin + sizeof(evaluation_point_.clock);
    const auto data = reinterpret_cast<std::uintptr_t>(evaluation_point_.clock.data());
    if (!(data >= begin && data < end))
      checked_add(total, static_cast<std::uint64_t>(evaluation_point_.clock.capacity()) + 1U);
    return total;
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
    const int block = configuration_->routes[route].runtime_block;
    if (!owner_->active_attempt_states_.empty())
      return *owner_->active_attempt_states_[static_cast<std::size_t>(block)];
    return const_cast<field_type&>(
        owner_->preparation_view_
            ->block_prototypes[static_cast<std::size_t>(block)][static_cast<std::size_t>(level_)]);
  }
  [[nodiscard]] Geometry<Dim> same_level_cell_geometry() const {
    if (owner_->preparation_view_ != nullptr)
      return owner_->preparation_view_->level_geometries.at(static_cast<std::size_t>(level_));
    return owner_->facade_->program_prepared_amr_level_geometry_(level_);
  }
  [[nodiscard]] const std::array<bool, Dim>& same_level_cell_periodicity() const noexcept {
    return periodicity_;
  }
  [[nodiscard]] std::string_view same_level_cell_state_identity(std::size_t route) const {
    if (owner_->active_block_identities_.empty())
      return configuration_->exact_contract;
    return owner_
        ->active_block_identities_[static_cast<std::size_t>(same_level_cell_runtime_block(route))];
  }

  [[nodiscard]] bool rebind_active_candidates_noexcept() const noexcept {
    if (owner_->active_attempt_states_.size() != configuration_->routes.size())
      return false;
    for (const auto& route : configuration_->routes) {
      const auto index = static_cast<std::size_t>(route.runtime_block);
      if (index >= owner_->active_attempt_states_.size() ||
          owner_->active_attempt_states_[index] == nullptr)
        return false;
      const field_type& prototype = owner_->preparation_view_ != nullptr
                                        ? owner_->preparation_view_->block_prototypes[index].at(
                                              static_cast<std::size_t>(level_))
                                        : *owner_->active_attempt_states_[index];
      const field_type& candidate = *owner_->active_attempt_states_[index];
      if (candidate.layout() != prototype.layout() || candidate.ncomp() != prototype.ncomp() ||
          candidate.ghosts() != prototype.ghosts())
        return false;
    }
    return true;
  }

  void rebind_configuration_noexcept(const CellTemporalConfiguration& configuration) noexcept {
    // ``configuration_`` initially denotes the candidate object supplied to resident
    // construction.  PreparedCellTemporalExecution subsequently moves that object into the
    // accepted image, so dereferencing the old pointer here would be a use-after-move (and a
    // noexcept terminate during installation).  Validate against the immutable scalar witness
    // captured at construction, then rebind to the moved-to authority.
    if (configuration.routes.size() != configured_route_count_ ||
        configuration.level_rungs.size() != configured_level_count_ ||
        configuration.tick_denominator != configured_tick_denominator_ || level_ < 0 ||
        static_cast<std::size_t>(level_) >= configuration.level_rungs.size() ||
        configuration.level_rungs[static_cast<std::size_t>(level_)] != configured_level_rung_ ||
        configuration.topology_epoch != configured_topology_epoch_ ||
        configuration.materialization_generation != configured_materialization_generation_)
      std::terminate();
    configuration_ = &configuration;
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

  [[nodiscard]] const multiblock::BoundaryEvaluationPoint& same_level_cell_evaluation_point(
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
    evaluation_point_.tick = owner_->active_subcycling_window_.begin.macro_step;
    evaluation_point_.level = level_;
    evaluation_point_.substep = owner_->logical_substep_;
    evaluation_point_.stage = 0;
    evaluation_point_.stage_fraction = {0, 1};
    evaluation_point_.dt = batch_dt;
    evaluation_point_.physical_time = physical_begin;
    return evaluation_point_;
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
      if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>)
        // The resident direct copies below read completed face-flux kernels from the host.
        same_level_cell_temporal_detail::resident_execution_fence();
      for (int axis = 0; axis < Dim; ++axis) {
        field_type& destination = *fluxes[static_cast<std::size_t>(axis)];
        for (std::size_t local = 0; local < destination.local_size(); ++local)
          copy_face_axis_(axis, evaluation.integrated_face_fluxes[local], destination.fab(local));
      }
      same_level_cell_temporal_detail::resident_execution_fence();
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
    for (std::size_t route = 0; route < configuration_->routes.size(); ++route) {
      if (evaluations_[route] == nullptr || final_residuals_[route] == nullptr)
        throw std::logic_error("cell-local AMR final flux lost its route evaluation");
      for (int axis = 0; axis < Dim; ++axis)
        pops::scale(integrated_flux_[route][static_cast<std::size_t>(axis)], Real(1) / interval_dt);
    }
    same_level_cell_temporal_detail::resident_execution_fence();
  }

  void finalize_same_level_cell_flux_metadata() {
    // Cell-local FE is admitted only without inter-block/interface coupling.  Its authoritative
    // flux product is therefore the fixed-slot, time-integrated diagnostic carrier owned by the
    // provider, not a per-attempt symbolic FluxExpressionRegistry.  Cloning that map here used
    // to allocate nodes and strings on every candidate step even though it was necessarily empty.
    // Keep the exact route/evaluation validation, but publish no second metadata authority.
    for (std::size_t route = 0; route < configuration_->routes.size(); ++route) {
      if (evaluations_[route] == nullptr || final_residuals_[route] == nullptr)
        throw std::logic_error("cell-local AMR final flux lost its prepared route metadata");
    }
  }
  void commit_same_level_cell_flux_metadata() noexcept {
    std::fill(evaluations_.begin(), evaluations_.end(), nullptr);
    std::fill(final_residuals_.begin(), final_residuals_.end(), nullptr);
  }
  void discard_same_level_cell_flux_metadata() noexcept {
    std::fill(evaluations_.begin(), evaluations_.end(), nullptr);
    std::fill(final_residuals_.begin(), final_residuals_.end(), nullptr);
  }

 private:
  [[nodiscard]] const field_type& state_for_route_(int runtime_block) const {
    if (!owner_->active_attempt_states_.empty()) {
      const auto index = static_cast<std::size_t>(runtime_block);
      if (index >= owner_->active_attempt_states_.size() ||
          owner_->active_attempt_states_[index] == nullptr)
        throw std::logic_error("cell-local AMR resident construction has no active block state");
      return *owner_->active_attempt_states_[index];
    }
    if (owner_->preparation_view_ == nullptr)
      throw std::logic_error("cell-local AMR resident construction has no detached prototype");
    return owner_->preparation_view_->block_prototypes.at(static_cast<std::size_t>(runtime_block))
        .at(static_cast<std::size_t>(level_));
  }
  template <int Axis = 0>
  static void copy_face_axis_(int axis, const nd::FaceField<Dim>& source, Fab<Dim>& destination) {
    if constexpr (Axis < Dim) {
      if (axis == Axis) {
        const auto& source_storage = source.template field<Axis>().storage();
        const auto& destination_storage = destination.storage();
        if (source_storage.extent(0) != destination_storage.extent(0))
          throw std::invalid_argument(
              "cell-local AMR face-flux extraction changed its resident face layout");
        if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>) {
          // Kokkos::deep_copy creates a profiling label for every host dispatch.  Both face
          // carriers are resident HostSpace views; the caller fences before reading them, so the
          // transfer itself can remain allocation-free.
          std::copy_n(source_storage.data(), source_storage.extent(0), destination_storage.data());
        } else {
          Kokkos::deep_copy(::pops::detail::default_execution_space(), destination_storage,
                            source_storage);
        }
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
  std::size_t configured_route_count_ = 0;
  std::size_t configured_level_count_ = 0;
  std::int64_t configured_tick_denominator_ = 0;
  int configured_level_rung_ = -1;
  std::uint64_t configured_topology_epoch_ = 0;
  std::uint64_t configured_materialization_generation_ = 0;
  std::array<bool, Dim> periodicity_{};
  mutable multiblock::BoundaryEvaluationPoint evaluation_point_{};
  std::vector<const level_evaluation_type*> evaluations_;
  std::vector<std::array<field_type, Dim>> integrated_flux_;
  std::vector<const field_type*> final_residuals_;
};
