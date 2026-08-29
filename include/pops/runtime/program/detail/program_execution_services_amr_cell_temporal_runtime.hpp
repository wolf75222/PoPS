[[nodiscard]] std::uint64_t cell_temporal_level_cell_count_(int runtime_block, int level) const {
  const field_type& state = facade_->program_prepared_amr_block_state_(runtime_block, level);
  std::uint64_t count = 0;
  for (const Box<Dim>& patch : state.layout().boxes()) {
    const std::int64_t points = patch.numPts();
    if (points <= 0 ||
        static_cast<std::uint64_t>(points) > std::numeric_limits<std::uint64_t>::max() - count)
      throw std::overflow_error("cell-local AMR topology exceeds uint64_t");
    count += static_cast<std::uint64_t>(points);
  }
  return count;
}

[[nodiscard]] std::uint64_t cell_temporal_block_major_offset_(
    const CellTemporalConfiguration& configuration, std::size_t route, int level) const {
  std::uint64_t offset = 0;
  for (std::size_t prior = 0; prior <= route; ++prior) {
    const int stop = prior == route ? level : nlev();
    const int block = configuration.routes[prior].runtime_block;
    for (int prior_level = 0; prior_level < stop; ++prior_level) {
      const std::uint64_t count = cell_temporal_level_cell_count_(block, prior_level);
      if (count > std::numeric_limits<std::uint64_t>::max() - offset)
        throw std::overflow_error("cell-local AMR block-major identity exceeds uint64_t");
      offset += count;
    }
  }
  return offset;
}

[[nodiscard]] std::vector<int> cell_temporal_level_rungs_(int finest_rung) const {
  const auto relations = facade_->program_prepared_temporal_relations_();
  if (nlev() <= 0 || relations.size() + 1 != static_cast<std::size_t>(nlev()))
    throw std::logic_error(
        "cell-local AMR rung derivation lacks one exact relation per live transition");
  std::vector<int> level_rungs(static_cast<std::size_t>(nlev()), finest_rung);
  for (std::size_t child = relations.size(); child != 0; --child) {
    const auto ratio = relations[child - 1].temporal_ratio();
    if (ratio.numerator <= 0 || ratio.denominator != 1)
      throw std::invalid_argument(
          "cell-local AMR requires integral power-of-two temporal refinement ratios");
    const auto refinement = static_cast<std::uint64_t>(ratio.numerator);
    if ((refinement & (refinement - 1)) != 0)
      throw std::invalid_argument(
          "cell-local AMR requires power-of-two temporal refinement ratios");
    int exponent = 0;
    for (std::uint64_t value = refinement; value > 1; value >>= 1)
      ++exponent;
    const int child_rung = level_rungs[child];
    if (child_rung > 30 - exponent)
      throw std::invalid_argument("cell-local AMR derived rung exceeds its bounded domain");
    level_rungs[child - 1] = child_rung + exponent;
  }
  return level_rungs;
}

[[nodiscard]] CellTemporalPartitionAcceptedState cell_temporal_full_partition_(
    const CellTemporalConfiguration& configuration, std::int64_t synchronization_tick) const {
  CellTemporalPartitionAcceptedState result;
  result.kind = TemporalPartitionKind::CellLocal;
  result.provider_identity = std::string(kSameLevelTransportEulerStageFluxProvider);
  result.topology_epoch = runtime_->topology_epoch();
  result.synchronization_tick = synchronization_tick;
  result.tick_denominator = configuration.tick_denominator;
  for (int level = 0; level < nlev(); ++level)
    for (std::size_t route = 0; route < configuration.routes.size(); ++route) {
      const int block = configuration.routes[route].runtime_block;
      std::uint64_t cell = cell_temporal_block_major_offset_(configuration, route, level);
      const field_type& state = facade_->program_prepared_amr_block_state_(block, level);
      for (const Box<Dim>& patch : state.layout().boxes())
        for (std::int64_t ordinal = 0; ordinal < patch.numPts(); ++ordinal)
          result.cells.push_back({level, cell++,
                                  configuration.level_rungs.at(static_cast<std::size_t>(level)),
                                  synchronization_tick});
    }
  validate_cell_temporal_partition_state(result);
  return result;
}

static constexpr bool cell_temporal_host_execution_supported_ =
    std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace> &&
    Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible;

void require_cell_temporal_execution_envelope_() const {
  if constexpr (!cell_temporal_host_execution_supported_)
    throw std::invalid_argument(
        "cell-local AMR execution requires a host default execution and memory space");
  const BoundaryTopology<Dim> topology = facade_->program_prepared_amr_boundary_topology_();
  for (int axis = 0; axis < Dim; ++axis)
    if (!topology.is_periodic(Face<Dim>{axis, BoundarySide::lower}) ||
        !topology.is_periodic(Face<Dim>{axis, BoundarySide::upper}))
      throw std::invalid_argument(
          "cell-local AMR execution requires periodic boundaries on every physical face");
}

void prepare_same_level_cell_temporal_execution_(
    std::string clock, std::int64_t tick_denominator, int rung,
    std::span<const SameLevelCellTemporalForwardEulerRoute> authored_routes) const {
  require_facade_execution_();
  const ExecutionLane& lane = prepared_execution_lane();
  std::optional<CellTemporalConfiguration> candidate;
  std::optional<CellTemporalPartitionAcceptedState> partition;
  std::exception_ptr local_error;
  try {
    if (cell_temporal_configuration_ || clock.empty() || tick_denominator <= 0 || rung < 0 ||
        rung > 30 || authored_routes.empty() ||
        authored_routes.size() != static_cast<std::size_t>(facade_->program_n_blocks_()))
      throw std::invalid_argument(
          "cell-local AMR execution requires one complete FE route per runtime block");
    require_cell_temporal_execution_envelope_();
    const auto& prepared_hierarchy = facade_->prepared_amr_multiblock_hierarchy_();
    if (prepared_hierarchy.coupling_count() != 0 ||
        prepared_hierarchy.has_interface_flux_provider())
      throw std::invalid_argument(
          "cell-local AMR execution currently requires an uncoupled multi-block hierarchy");
    candidate.emplace();
    candidate->clock = std::move(clock);
    candidate->tick_denominator = tick_denominator;
    candidate->rung = rung;
    candidate->routes.assign(authored_routes.begin(), authored_routes.end());
    for (auto& route : candidate->routes) {
      if (route.program_block < 0 || route.rhs_id < 0)
        throw std::invalid_argument("cell-local AMR route has a negative typed identity");
      const int mapped = sys_block(route.program_block);
      if (route.runtime_block < 0)
        route.runtime_block = mapped;
      if (route.runtime_block != mapped)
        throw std::invalid_argument("cell-local AMR route differs from its Program block map");
    }
    std::sort(candidate->routes.begin(), candidate->routes.end(),
              [](const auto& left, const auto& right) {
                return std::tie(left.runtime_block, left.program_block, left.rhs_id) <
                       std::tie(right.runtime_block, right.program_block, right.rhs_id);
              });
    for (std::size_t index = 0; index < candidate->routes.size(); ++index) {
      if (candidate->routes[index].runtime_block != static_cast<int>(index) ||
          (index != 0 &&
           candidate->routes[index - 1].program_block == candidate->routes[index].program_block))
        throw std::invalid_argument(
            "cell-local AMR routes are not a bijection over the complete block pack");
      for (int level = 0; level < nlev(); ++level) {
        const field_type& reference = facade_->program_prepared_amr_block_state_(0, level);
        const field_type& state =
            facade_->program_prepared_amr_block_state_(candidate->routes[index].runtime_block,
                                                       level);
        if (state.layout() != reference.layout() ||
            state.distribution() != reference.distribution() ||
            state.local_rank() != reference.local_rank())
          throw std::invalid_argument(
              "cell-local AMR routes do not share one prepared hierarchy topology");
      }
    }
    candidate->topology_epoch = runtime_->topology_epoch();
    candidate->materialization_generation = runtime_->materialization_generation();
    candidate->level_rungs = cell_temporal_level_rungs_(candidate->rung);
    candidate->level_cell_counts.clear();
    candidate->level_cell_counts.reserve(static_cast<std::size_t>(nlev()));
    for (int level = 0; level < nlev(); ++level)
      candidate->level_cell_counts.push_back(cell_temporal_level_cell_count_(0, level));
    ExactContractBuilder contract;
    contract.text("pops.amr-program.cell-local-forward-euler")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(candidate->clock)
        .scalar(candidate->tick_denominator)
        .scalar(std::int32_t{candidate->rung})
        .scalar(candidate->topology_epoch)
        .scalar(candidate->materialization_generation)
        .text(lane.identity())
        .bytes(runtime_->spatial_contract())
        .text("host-default-execution-and-memory")
        .presence(cell_temporal_host_execution_supported_)
        .scalar(std::uint64_t{2 * Dim})
        .scalar(static_cast<std::uint64_t>(prepared_hierarchy.coupling_count()))
        .presence(prepared_hierarchy.has_interface_flux_provider())
        .scalar(static_cast<std::uint64_t>(candidate->routes.size()))
        .sequence(candidate->level_rungs,
                  [](ExactContractBuilder& item, int level_rung) {
                    item.scalar(std::int32_t{level_rung});
                  })
        .sequence(candidate->level_cell_counts,
                  [](ExactContractBuilder& item, std::uint64_t count) { item.scalar(count); });
    const BoundaryTopology<Dim> topology = facade_->program_prepared_amr_boundary_topology_();
    for (int axis = 0; axis < Dim; ++axis)
      contract.presence(topology.is_periodic(Face<Dim>{axis, BoundarySide::lower}))
          .presence(topology.is_periodic(Face<Dim>{axis, BoundarySide::upper}));
    for (const auto& route : candidate->routes)
      contract.scalar(std::int32_t{route.program_block})
          .scalar(std::int32_t{route.runtime_block})
          .scalar(std::int32_t{route.rhs_id});
    candidate->exact_contract = std::move(contract).release();
    const double scaled_time = facade_->program_time_() * static_cast<double>(tick_denominator);
    if (!std::isfinite(scaled_time) || scaled_time < 0.0 ||
        !(scaled_time < static_cast<double>(std::numeric_limits<std::int64_t>::max())) ||
        std::floor(scaled_time) != scaled_time)
      throw std::invalid_argument("cell-local AMR accepted time has no exact tick encoding");
    const auto synchronization_tick = static_cast<std::int64_t>(scaled_time);
    if (synchronization_tick % (std::int64_t{1} << candidate->level_rungs.front()) != 0)
      throw std::invalid_argument(
          "cell-local AMR accepted time is not aligned to its coarsest derived rung");
    partition.emplace(cell_temporal_full_partition_(*candidate, synchronization_tick));
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("cell-local AMR route preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"cell-local-amr-route-pack", candidate->exact_contract}}, lane))
    throw std::invalid_argument("cell-local AMR route table differs between execution ranks");
  cell_temporal_configuration_.emplace(std::move(*candidate));
  accepted_temporal_partition_ = std::move(*partition);
  cell_temporal_diagnostics_.clear();
}

static std::int64_t cell_temporal_phase_tick_(std::int64_t begin, std::int64_t extent,
                                              ::pops::amr::Rational phase) {
  if (phase.denominator <= 0 || phase.numerator < 0 || phase.numerator > phase.denominator ||
      extent < 0 || extent % phase.denominator != 0)
    throw std::invalid_argument("cell-local AMR subcycling phase has no exact tick boundary");
  const std::int64_t unit = extent / phase.denominator;
  if (phase.numerator != 0 && unit > std::numeric_limits<std::int64_t>::max() / phase.numerator)
    throw std::overflow_error("cell-local AMR subcycling phase exceeds int64_t");
  const std::int64_t offset = unit * phase.numerator;
  if (begin > std::numeric_limits<std::int64_t>::max() - offset)
    throw std::overflow_error("cell-local AMR subcycling tick exceeds int64_t");
  return begin + offset;
}

void advance_same_level_cell_temporal_(double dt) const {
  require_facade_execution_();
  refresh_resources_();
  requalify_cell_temporal_configuration_();
  if (!cell_temporal_configuration_ || !std::isfinite(dt) || !(dt > 0.0) ||
      accepted_temporal_partition_.kind != TemporalPartitionKind::CellLocal ||
      accepted_temporal_partition_.provider_identity != kSameLevelTransportEulerStageFluxProvider)
    throw std::logic_error("cell-local AMR execution is not prepared");
  const double scaled = dt * static_cast<double>(cell_temporal_configuration_->tick_denominator);
  if (!std::isfinite(scaled) || !(scaled > 0.0) ||
      !(scaled < static_cast<double>(std::numeric_limits<std::int64_t>::max())) ||
      std::floor(scaled) != scaled)
    throw std::invalid_argument("cell-local AMR dt has no bounded exact tick extent");
  const auto extent = static_cast<std::int64_t>(scaled);
  const std::int64_t stride = std::int64_t{1} << cell_temporal_configuration_->level_rungs.front();
  if (extent != stride)
    throw std::invalid_argument(
        "cell-local AMR dt must produce exactly one FE batch on its coarsest level window");
  const std::int64_t begin = accepted_temporal_partition_.synchronization_tick;
  if (extent > std::numeric_limits<std::int64_t>::max() - begin)
    throw std::overflow_error("cell-local AMR target tick exceeds int64_t");
  const std::int64_t target = begin + extent;
  std::vector<std::shared_ptr<SameLevelCellIntegratedFluxPackDiagnostic<Dim>>> diagnostics;
  std::vector<std::string> diagnostic_clock_identities;
  std::size_t diagnostic_slot = 0;
  CellTemporalPartitionAcceptedState target_partition;
  std::string target_partition_contract;
  std::exception_ptr preparation_error;
  try {
    target_partition = cell_temporal_full_partition_(*cell_temporal_configuration_, target);
    ExactContractBuilder target_contract;
    target_contract.text("pops.amr-program.cell-local-target-partition")
        .scalar(std::uint32_t{1})
        .text(target_partition.provider_identity)
        .scalar(target_partition.topology_epoch)
        .scalar(target_partition.synchronization_tick)
        .scalar(target_partition.tick_denominator)
        .sequence(target_partition.cells,
                  [](ExactContractBuilder& item, const CellTemporalPartitionRecord& cell) {
                    item.scalar(std::int32_t{cell.level})
                        .scalar(cell.cell)
                        .scalar(std::int32_t{cell.rung})
                        .scalar(cell.accepted_tick);
                  });
    target_partition_contract = std::move(target_contract).release();
    std::size_t level_groups = 1;
    std::size_t level_multiplicity = 1;
    for (const auto& relation : facade_->program_prepared_temporal_relations_()) {
      const auto ratio = relation.temporal_ratio();
      if (ratio.numerator <= 0 || ratio.denominator <= 0)
        throw std::logic_error("cell-local AMR temporal relation has an invalid ratio");
      const auto quotient = static_cast<std::size_t>(ratio.numerator / ratio.denominator);
      const auto remainder = ratio.numerator % ratio.denominator;
      const std::size_t children = quotient + (remainder == 0 ? 0 : 1);
      level_multiplicity = checked_product_(level_multiplicity, children,
                                            "cell-local AMR diagnostic level-group count");
      if (level_multiplicity > std::numeric_limits<std::size_t>::max() - level_groups)
        throw std::length_error("cell-local AMR diagnostic level-group count exceeds size_t");
      level_groups += level_multiplicity;
    }
    diagnostics.resize(level_groups);
    diagnostic_clock_identities.assign(level_groups, cell_temporal_configuration_->clock);
    for (auto& diagnostic : diagnostics)
      diagnostic = std::make_shared<SameLevelCellIntegratedFluxPackDiagnostic<Dim>>();
  } catch (...) {
    preparation_error = std::current_exception();
  }
  const ExecutionLane& lane = prepared_execution_lane();
  if (all_reduce_max(preparation_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && preparation_error)
      std::rethrow_exception(preparation_error);
    throw std::runtime_error("cell-local AMR target partition preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"cell-local-amr-target-partition", target_partition_contract}}, lane))
    throw std::invalid_argument("cell-local AMR target partition differs between execution ranks");
  advance_prepared_hierarchy_(
      dt,
      [&](double) {
        int level = 0;
        std::int64_t level_begin = 0;
        std::int64_t level_target = 0;
        std::optional<CellTemporalLevelRuntime> runtime;
        std::shared_ptr<SameLevelCellIntegratedFluxPackDiagnostic<Dim>> diagnostic;
        std::string clock_identity;
        std::exception_ptr local_error;
        try {
          level = active_level_;
          level_begin =
              cell_temporal_phase_tick_(begin, extent, active_subcycling_window_.begin.phase);
          level_target =
              cell_temporal_phase_tick_(begin, extent, active_subcycling_window_.end.phase);
          cell_temporal_interval_begin_tick_ = level_begin;
          cell_temporal_interval_target_tick_ = level_target;
          runtime.emplace(*this, *cell_temporal_configuration_, level);
          if (diagnostic_slot >= diagnostics.size())
            throw std::logic_error(
                "cell-local AMR execution exceeded its preallocated level-group slots");
          diagnostic = diagnostics[diagnostic_slot];
          clock_identity = std::move(diagnostic_clock_identities[diagnostic_slot]);
          ++diagnostic_slot;
        } catch (...) {
          local_error = std::current_exception();
        }
        if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
          if (lane.size() == 1 && local_error)
            std::rethrow_exception(local_error);
          throw std::runtime_error("cell-local AMR level runtime preparation failed collectively");
        }
        auto partition = prepare_same_level_transport_euler_partition_pack<Dim>(
            *runtime, level, level_begin, cell_temporal_configuration_->tick_denominator,
            cell_temporal_configuration_->level_rungs.at(static_cast<std::size_t>(level)),
            prepared_execution_lane());
        using provider_type =
            PreparedSameLevelTransportEulerPackStageFluxProvider<Dim, CellTemporalLevelRuntime>;
        provider_type provider(*runtime, partition, diagnostic, std::move(clock_identity),
                               prepared_execution_lane());
        PreparedBatchedCellTemporalExecutor<provider_type> executor(
            std::move(partition), std::move(provider), prepared_execution_lane());
        executor.begin_attempt(level_target);
        executor.advance_to_barrier();
        executor.commit();
      },
      "advance_same_level_cell_temporal");
  // ``diagnostics`` is a preallocated upper bound. Shrinking destroys only unused trailing
  // shared_ptr slots and cannot allocate; every used slot was already closed collectively.
  diagnostics.resize(diagnostic_slot);
  static_assert(std::is_nothrow_move_assignable_v<CellTemporalPartitionAcceptedState>);
  accepted_temporal_partition_ = std::move(target_partition);
  cell_temporal_diagnostics_.swap(diagnostics);
  cell_temporal_interval_begin_tick_ = target;
  cell_temporal_interval_target_tick_ = target;
}

void requalify_cell_temporal_configuration_() const {
  if (!cell_temporal_configuration_ ||
      (cell_temporal_configuration_->topology_epoch == runtime_->topology_epoch() &&
       cell_temporal_configuration_->materialization_generation ==
           runtime_->materialization_generation()))
    return;
  const ExecutionLane& lane = prepared_execution_lane();
  std::optional<CellTemporalConfiguration> candidate;
  std::optional<CellTemporalPartitionAcceptedState> partition;
  std::exception_ptr local_error;
  try {
    candidate.emplace(*cell_temporal_configuration_);
    require_cell_temporal_execution_envelope_();
    const auto& prepared_hierarchy = facade_->prepared_amr_multiblock_hierarchy_();
    if (prepared_hierarchy.coupling_count() != 0 ||
        prepared_hierarchy.has_interface_flux_provider())
      throw std::invalid_argument(
          "cell-local AMR hierarchy requalification found unsupported global coupling");
    candidate->topology_epoch = runtime_->topology_epoch();
    candidate->materialization_generation = runtime_->materialization_generation();
    candidate->level_rungs = cell_temporal_level_rungs_(candidate->rung);
    candidate->level_cell_counts.clear();
    candidate->level_cell_counts.reserve(static_cast<std::size_t>(nlev()));
    for (int level = 0; level < nlev(); ++level)
      candidate->level_cell_counts.push_back(cell_temporal_level_cell_count_(0, level));
    ExactContractBuilder contract;
    contract.text("pops.amr-program.cell-local-forward-euler")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(candidate->clock)
        .scalar(candidate->tick_denominator)
        .scalar(std::int32_t{candidate->rung})
        .scalar(candidate->topology_epoch)
        .scalar(candidate->materialization_generation)
        .text(lane.identity())
        .bytes(runtime_->spatial_contract())
        .text("host-default-execution-and-memory")
        .presence(cell_temporal_host_execution_supported_)
        .scalar(std::uint64_t{2 * Dim})
        .scalar(static_cast<std::uint64_t>(prepared_hierarchy.coupling_count()))
        .presence(prepared_hierarchy.has_interface_flux_provider())
        .scalar(static_cast<std::uint64_t>(candidate->routes.size()))
        .sequence(candidate->level_rungs,
                  [](ExactContractBuilder& item, int level_rung) {
                    item.scalar(std::int32_t{level_rung});
                  })
        .sequence(candidate->level_cell_counts,
                  [](ExactContractBuilder& item, std::uint64_t count) { item.scalar(count); });
    const BoundaryTopology<Dim> topology = facade_->program_prepared_amr_boundary_topology_();
    for (int axis = 0; axis < Dim; ++axis)
      contract.presence(topology.is_periodic(Face<Dim>{axis, BoundarySide::lower}))
          .presence(topology.is_periodic(Face<Dim>{axis, BoundarySide::upper}));
    for (const auto& route : candidate->routes) {
      if (sys_block(route.program_block) != route.runtime_block)
        throw std::logic_error(
            "cell-local AMR retained route changed its authenticated Program block map");
      contract.scalar(std::int32_t{route.program_block})
          .scalar(std::int32_t{route.runtime_block})
          .scalar(std::int32_t{route.rhs_id});
    }
    candidate->exact_contract = std::move(contract).release();
    partition.emplace(cell_temporal_full_partition_(
        *candidate, accepted_temporal_partition_.synchronization_tick));
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("cell-local AMR hierarchy requalification failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"cell-local-amr-requalified-route-pack", candidate->exact_contract}}, lane))
    throw std::invalid_argument(
        "cell-local AMR requalified route table differs between execution ranks");
  cell_temporal_configuration_.emplace(std::move(*candidate));
  accepted_temporal_partition_ = std::move(*partition);
  cell_temporal_diagnostics_.clear();
}
