void apply_reflux_payload_(field_type& coarse, const Index<Dim>& cell,
                           const std::vector<Real>& correction) const {
  for (std::size_t local = 0; local < coarse.local_size(); ++local) {
    if (!coarse.box(local).contains(cell))
      continue;
    const FieldView<Real, Dim> values = coarse.fab(local).view();
    for (int component = 0; component < static_cast<int>(correction.size()); ++component) {
      const Real increment = correction[static_cast<std::size_t>(component)];
      for_each_cell(Box<Dim>{cell, cell}, [=] POPS_HD(const Index<Dim>& index) {
        values(index, component) += increment;
      });
    }
    return;
  }
}

void attach_active_flux_basis_(int runtime_block, const level_evaluation_type& evaluation,
                               field_type& rhs, int rhs_identity,
                               FluxBasisProvider provider) const {
  const ::pops::amr::ClockWindow interval{
      {active_level_, evaluation.point.tick, current_interval_begin_phase_,
       current_interval_start_time_},
      {active_level_, evaluation.point.tick, current_interval_end_phase_,
       current_interval_start_time_ + current_dt_}};
  if (static_flux_tables_.bound) {
    FluxExpressionRegistry unused_registry;
    prepare_active_flux_basis_impl_(
        runtime_block, evaluation.point, rhs_identity, provider, evaluation.topology_epoch,
        evaluation.materialization_generation, rhs, &evaluation, nullptr, nullptr, interval,
        unused_registry, active_flux_basis_counts_, next_active_flux_basis_identity_);
    return;
  }
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
  const ExecutionLane& lane = prepared_execution_lane();
  if (all_reduce_max(candidate_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && candidate_error)
      std::rethrow_exception(candidate_error);
    throw std::runtime_error("AMR Program flux-expression candidate copy failed collectively");
  }
  prepare_active_flux_basis_impl_(runtime_block, evaluation.point, rhs_identity, provider,
                                  evaluation.topology_epoch, evaluation.materialization_generation,
                                  rhs, &evaluation, nullptr, nullptr, interval, candidate_registry,
                                  candidate_counts, candidate_identity);
  static_assert(std::is_nothrow_swappable_v<FluxExpressionRegistry>);
  static_assert(std::is_nothrow_swappable_v<std::vector<std::size_t>>);
  active_flux_expressions_.swap(candidate_registry);
  active_flux_basis_counts_.swap(candidate_counts);
  next_active_flux_basis_identity_ = candidate_identity;
}

void prepare_cell_temporal_flux_basis_(int runtime_block, const level_evaluation_type& evaluation,
                                       const field_type& rhs, int rhs_identity,
                                       const std::array<field_type, Dim>& integrated_face_fluxes,
                                       std::int64_t begin_tick, std::int64_t end_tick,
                                       FluxExpressionRegistry& candidate_registry,
                                       std::vector<std::size_t>& candidate_counts,
                                       std::uint64_t& candidate_identity) const {
  std::optional<::pops::amr::ClockWindow> interval;
  std::exception_ptr local_error;
  try {
    const std::int64_t extent =
        cell_temporal_interval_target_tick_ - cell_temporal_interval_begin_tick_;
    if (extent <= 0 || begin_tick < cell_temporal_interval_begin_tick_ ||
        end_tick > cell_temporal_interval_target_tick_ || begin_tick >= end_tick)
      throw std::logic_error("cell-local AMR flux basis lies outside its active interval");
    const auto local_begin =
        ::pops::amr::Rational{begin_tick - cell_temporal_interval_begin_tick_, extent};
    const auto local_end =
        ::pops::amr::Rational{end_tick - cell_temporal_interval_begin_tick_, extent};
    const auto span = active_subcycling_window_.end.phase - active_subcycling_window_.begin.phase;
    interval.emplace(
        ::pops::amr::ClockWindow{{active_level_, active_subcycling_window_.begin.macro_step,
                                  active_subcycling_window_.begin.phase + span * local_begin,
                                  current_interval_start_time_ + local_begin.value() * current_dt_},
                                 {active_level_, active_subcycling_window_.end.macro_step,
                                  active_subcycling_window_.begin.phase + span * local_end,
                                  current_interval_start_time_ + local_end.value() * current_dt_}});
  } catch (...) {
    local_error = std::current_exception();
  }
  const ExecutionLane& lane = prepared_execution_lane();
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("cell-local AMR flux-basis interval preparation failed collectively");
  }
  prepare_active_flux_basis_impl_(runtime_block, evaluation.point, rhs_identity,
                                  FluxBasisProvider::ExactFace, evaluation.topology_epoch,
                                  evaluation.materialization_generation, rhs, &evaluation,
                                  &integrated_face_fluxes, nullptr, *interval, candidate_registry,
                                  candidate_counts, candidate_identity);
}

void prepare_active_flux_basis_impl_(
    int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point, int rhs_identity,
    FluxBasisProvider provider, std::uint64_t topology_epoch,
    std::uint64_t materialization_generation, const field_type& rhs,
    const level_evaluation_type* evaluation, const std::array<field_type, Dim>* exact_face_fluxes,
    const std::array<field_type*, Dim>* named_cell_fluxes, const ::pops::amr::ClockWindow& interval,
    FluxExpressionRegistry& candidate_registry, std::vector<std::size_t>& candidate_counts,
    std::uint64_t& candidate_identity) const {
  const ExecutionLane& lane = prepared_execution_lane();
  const long active = active_attempt_states_.empty() ? 0L : 1L;
  const long active_minimum = all_reduce_min(active, lane);
  const long active_maximum = all_reduce_max(active, lane);
  if (active_minimum != active_maximum)
    throw std::logic_error("AMR Program flux-basis activity differs between execution ranks");
  if (active_maximum == 0)
    return;

  using fragment_role_type = ::pops::amr::reflux::FaceLedgerRole;
  struct PendingFace {
    fragment_role_type role = fragment_role_type::Coarse;
    int axis = 0;
    Index<Dim> face{};
    Index<Dim> coarse_face{};
    double measure = 0.0;
  };

  std::size_t block = 0;
  std::optional<std::uint32_t> declared_basis_slot;
  std::optional<std::size_t> static_cursor_restore;
  std::optional<std::size_t> static_count_restore;
  std::optional<std::uint64_t> static_identity_restore;
  bool static_prepared = false;
  // Legacy/non-table fallback storage is constructed only after the static carrier has declined
  // the call.  A bound Program never even creates these dynamic containers on its RHS route.
  std::optional<std::vector<PendingFace>> pending;
  std::optional<std::string> pending_contract;
  std::exception_ptr preparation_error;
  try {
    if (runtime_block < 0 ||
        static_cast<std::size_t>(runtime_block) >= active_attempt_states_.size() ||
        active_attempt_states_[static_cast<std::size_t>(runtime_block)] == nullptr)
      throw std::logic_error("AMR Program flux evaluation has no active block candidate");
    block = static_cast<std::size_t>(runtime_block);
    if (block >= active_block_identities_.size() || block >= active_outgoing_flux_.size() ||
        block >= active_incoming_flux_.size() || block >= prepared_rhs_basis_bounds_.size() ||
        block >= prepared_coefficient_term_bounds_.size() || block >= candidate_counts.size())
      throw std::logic_error("AMR Program flux evaluation has an incomplete active block pack");
    if (evaluation != nullptr)
      require_same_field_contract_(evaluation->residual, rhs,
                                   "AMR Program flux-expression RHS basis");
    if ((exact_face_fluxes != nullptr && named_cell_fluxes != nullptr) ||
        (evaluation == nullptr && exact_face_fluxes == nullptr && named_cell_fluxes == nullptr))
      throw std::logic_error("AMR Program flux basis requires one exact face provider");
    const bool prepared_evaluation = provider == FluxBasisProvider::PreparedResidual ||
                                     provider == FluxBasisProvider::PreparedDefaultFlux;
    if ((!prepared_evaluation && provider != FluxBasisProvider::ExactFace &&
         provider != FluxBasisProvider::NamedCell) ||
        (prepared_evaluation &&
         (evaluation == nullptr || exact_face_fluxes != nullptr || named_cell_fluxes != nullptr)) ||
        (provider == FluxBasisProvider::ExactFace &&
         (evaluation == nullptr || exact_face_fluxes == nullptr || named_cell_fluxes != nullptr)) ||
        (provider == FluxBasisProvider::NamedCell &&
         (evaluation != nullptr || exact_face_fluxes != nullptr || named_cell_fluxes == nullptr)))
      throw std::logic_error(
          "AMR Program flux basis provider differs from its frozen operator route");
    const double interval_dt = interval.end.physical_time - interval.begin.physical_time;
    const auto expected_stage =
        exact_face_fluxes == nullptr ? stage_time_ : ::pops::amr::Rational{0, 1};
    if (rhs_identity < 0 || point.clock.empty() || point.stage < 0 ||
        point.level != active_level_ || point.substep != logical_substep_ ||
        point.stage_fraction != expected_stage ||
        (provider != FluxBasisProvider::ExactFace && point.stage != rhs_identity) ||
        !std::isfinite(point.dt) || !(point.dt > 0.0) || !std::isfinite(point.physical_time) ||
        (exact_face_fluxes == nullptr && point.dt != interval_dt) ||
        topology_epoch != runtime_->topology_epoch() ||
        materialization_generation != runtime_->materialization_generation() ||
        (evaluation != nullptr && exact_face_fluxes == nullptr && named_cell_fluxes == nullptr &&
         evaluation->integrated_face_fluxes.size() != rhs.local_size()) ||
        active_block_identities_[block].empty() || !(interval.begin.phase < interval.end.phase) ||
        !std::isfinite(interval.begin.physical_time) || !std::isfinite(interval_dt) ||
        !(interval_dt > 0.0))
      throw std::logic_error(
          "AMR Program flux-expression basis differs from its active evaluation interval");
    const auto static_term_applies = [&](std::uint32_t term_slot) {
      if (term_slot >= static_flux_tables_.terms.size())
        throw std::logic_error("AMR Program static flux term slot is outside its carrier");
      const auto& term = static_flux_tables_.terms[term_slot];
      if (term.basis_slot >= static_flux_tables_.bases.size())
        throw std::logic_error("AMR Program static flux term has no declared basis");
      const int declared_level = static_flux_tables_.bases[term.basis_slot].level;
      return declared_level < 0 || declared_level == active_level_;
    };
    const bool static_cancellation =
        static_flux_tables_.bound &&
        std::none_of(static_flux_tables_.term_slots_by_runtime_block.at(block).begin(),
                     static_flux_tables_.term_slots_by_runtime_block.at(block).end(),
                     static_term_applies);
    if (prepared_rhs_basis_bounds_[block] == 0 ||
        (prepared_coefficient_term_bounds_[block] == 0 && !static_cancellation))
      throw std::logic_error(
          "flux-producing AMR Program block has no authenticated expression budget");
    if (candidate_counts.at(block) >= prepared_rhs_basis_bounds_[block])
      throw std::length_error(
          "AMR Program flux evaluations exceed their authenticated RHS-basis bound");
    if (static_flux_tables_.bound) {
      if (block >= static_flux_tables_.basis_slots_by_runtime_block.size())
        throw std::logic_error("AMR Program flux table carrier has no runtime block slot");
      const auto& ordered = static_flux_tables_.basis_slots_by_runtime_block[block];
      const std::size_t position = candidate_counts.at(block);
      if (position == 0)
        static_flux_tables_.next_basis_by_runtime_block[block] = 0;
      std::optional<std::uint32_t> applicable_slot;
      std::size_t applicable_position = 0;
      for (const std::uint32_t slot : ordered) {
        const auto& candidate = static_flux_tables_.bases.at(slot);
        if (candidate.level >= 0 && candidate.level != active_level_)
          continue;
        if (applicable_position++ == position) {
          applicable_slot = slot;
          break;
        }
      }
      if (!applicable_slot)
        throw std::length_error(
            "AMR Program flux evaluation exceeds its declared static basis occurrences");
      const auto& declared = static_flux_tables_.bases.at(*applicable_slot);
      if (declared.runtime_block != block || declared.rhs_identity != rhs_identity ||
          declared.provider != static_cast<std::uint8_t>(provider) ||
          (declared.level >= 0 && declared.level != active_level_) ||
          declared.stage != point.stage_fraction)
        throw std::invalid_argument(
            "AMR Program flux evaluation differs from its declared static basis occurrence");
      if (static_flux_tables_.next_basis_by_runtime_block[block] != position)
        throw std::logic_error(
            "AMR Program flux basis slot order was not reset for this level group");
      static_cursor_restore = static_flux_tables_.next_basis_by_runtime_block[block];
      ++static_flux_tables_.next_basis_by_runtime_block[block];
      declared_basis_slot = declared.basis_slot;
    }
    if (candidate_identity == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("AMR Program flux basis identity exhausted uint64_t");

    if (static_flux_tables_.bound) {
      static_count_restore = candidate_counts.at(block);
      static_identity_restore = candidate_identity;
      if (!declared_basis_slot || *declared_basis_slot >= static_flux_basis_payloads_.size() ||
          *declared_basis_slot >= static_flux_basis_active_.size())
        throw std::logic_error("AMR Program flux basis has no resident static payload slot");
      const auto& declared = static_flux_tables_.bases.at(*declared_basis_slot);
      FluxBasis& resident = static_flux_basis_payloads_[*declared_basis_slot];
      resident.identity = *declared_basis_slot;
      resident.runtime_block = block;
      resident.level = active_level_;
      // The bound two-table route consumes only the scalar stage coordinates.  Graph/rate/
      // application identities have already been authenticated into the cold ledger routes;
      // copying the runtime strings here would make every RHS attachment allocator-visible.
      resident.point.tick = point.tick;
      resident.point.level = point.level;
      resident.point.substep = point.substep;
      resident.point.stage = point.stage;
      resident.point.stage_fraction = point.stage_fraction;
      resident.point.dt = point.dt;
      resident.point.physical_time = point.physical_time;
      resident.rhs_identity = rhs_identity;
      resident.provider = provider;
      resident.window = interval;
      resident.face_count = 0;
      const bool has_final_term = std::any_of(
          static_flux_tables_.term_slots_by_runtime_block.at(block).begin(),
          static_flux_tables_.term_slots_by_runtime_block.at(block).end(), [&](std::uint32_t term) {
            return static_flux_tables_.terms.at(term).basis_slot == *declared_basis_slot;
          });
      const auto fill_route = [&](multiblock_flux_ledger_type* ledger, fragment_role_type role) {
        if (ledger == nullptr || !has_final_term)
          return;
        const auto route = std::find_if(
            declared.face_routes.begin(), declared.face_routes.end(), [&](const auto& candidate) {
              return candidate.ledger == ledger && candidate.level == active_level_ &&
                     candidate.role == role;
            });
        if (route == declared.face_routes.end())
          throw std::invalid_argument("AMR Program flux basis has no resident topology route");
        const Geometry<Dim> geometry = facade_->program_prepared_amr_level_geometry_(active_level_);
        for (const auto& template_face : route->faces) {
          if (resident.face_count >= resident.faces.size())
            throw std::length_error(
                "AMR Program flux basis exceeds its frozen resident face slots");
          FluxBasisFace& face = resident.faces[resident.face_count++];
          if (face.flux_density.size() != declared.components ||
              face.flux_density.size() != static_cast<std::size_t>(rhs.ncomp()))
            throw std::invalid_argument(
                "AMR Program flux basis payload differs from its frozen slot");
          face.role = role;
          face.axis = template_face.axis;
          face.face = template_face.face;
          face.coarse_face = template_face.coarse_face;
          face.face_measure = face_measure_(geometry, face.axis);
          if (!(face.face_measure > 0.0) || !std::isfinite(face.face_measure))
            throw std::invalid_argument(
                "AMR Program flux basis has an invalid resident face measure");
          if (named_cell_fluxes != nullptr)
            collective_named_face_payload_into_(*named_cell_fluxes, rhs, face.axis, face.face,
                                                face.flux_density);
          else if (exact_face_fluxes != nullptr)
            collective_face_payload_into_(*exact_face_fluxes, rhs, face.axis, face.face,
                                          face.flux_density);
          else
            collective_face_payload_into_(*evaluation, rhs, face.axis, face.face,
                                          face.flux_density);
          for (Real& component : face.flux_density) {
            if (!std::isfinite(static_cast<double>(component)))
              throw std::invalid_argument("AMR Program flux basis contains a non-finite payload");
            if (named_cell_fluxes == nullptr)
              component /= static_cast<Real>(face.face_measure);
          }
        }
      };
      fill_route(active_outgoing_flux_[block], fragment_role_type::Coarse);
      fill_route(active_incoming_flux_[block], fragment_role_type::Fine);
      static_flux_basis_active_[*declared_basis_slot] = 1;
      ++candidate_counts[block];
      candidate_identity = std::max(candidate_identity, resident.identity);
      static_prepared = true;
    }

    if (!static_prepared) {
      pending.emplace();
      if (active_outgoing_flux_[block] != nullptr) {
        const Geometry<Dim> geometry = facade_->program_prepared_amr_level_geometry_(active_level_);
        for (const ProgramInterfaceFace& interface :
             program_interface_faces_(static_cast<std::size_t>(active_level_)))
          pending->push_back({fragment_role_type::Coarse, interface.axis, interface.coarse_face,
                              interface.coarse_face, face_measure_(geometry, interface.axis)});
      }
      if (active_incoming_flux_[block] != nullptr) {
        const std::size_t parent = static_cast<std::size_t>(active_level_ - 1);
        const auto& hierarchy = facade_->prepared_amr_multiblock_hierarchy_();
        const auto ratio =
            hierarchy.topology_runtime().hierarchy().layout(parent + 1).ratio_from_parent();
        const ::pops::amr::reflux::FaceRefinementMapping<Dim> mapping{
            hierarchy.topology_runtime().hierarchy().layout(parent).domain().lo,
            hierarchy.topology_runtime().hierarchy().layout(parent + 1).domain().lo};
        const Geometry<Dim> geometry = facade_->program_prepared_amr_level_geometry_(active_level_);
        for (const ProgramInterfaceFace& interface : program_interface_faces_(parent)) {
          ::pops::amr::reflux::CoarseFaceRefluxKey<Dim> query;
          query.owner = std::string(active_block_identities_[block]);
          query.state = query.owner + "/state";
          query.levels = {static_cast<int>(parent), static_cast<int>(parent + 1)};
          query.axis = interface.axis;
          query.coarse_face = interface.coarse_face;
          query.attempt = active_subcycling_attempt_;
          query.macro_step = point.tick;
          query.window_begin = interval.begin.phase;
          query.window_end = interval.end.phase;
          std::size_t fine_count = 1;
          for (int axis = 0; axis < Dim; ++axis)
            if (axis != interface.axis)
              fine_count = checked_product_(fine_count, static_cast<std::size_t>(ratio[axis]),
                                            "AMR Program fine-face enumeration");
          const ::pops::amr::reflux::MetricRefluxBudget budget{fine_count, fine_count, 1};
          for (const Index<Dim>& fine_face :
               ::pops::amr::reflux::fine_faces_for_coarse_face(query, ratio, mapping, budget))
            pending->push_back({fragment_role_type::Fine, interface.axis, fine_face,
                                interface.coarse_face, face_measure_(geometry, interface.axis)});
        }
      }
      ExactContractBuilder exact;
      exact.text("pops.amr-program.cell-local-pending-flux-faces")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{runtime_block})
          .scalar(std::int32_t{rhs_identity})
          .scalar(static_cast<std::uint8_t>(provider))
          .text(point.clock)
          .scalar(point.tick)
          .scalar(std::int32_t{point.level})
          .scalar(std::int32_t{point.substep})
          .scalar(std::int32_t{point.stage})
          .scalar(point.stage_fraction.numerator)
          .scalar(point.stage_fraction.denominator)
          .scalar(point.dt)
          .scalar(point.physical_time)
          .text(point.graph_identity)
          .text(point.rate_identity)
          .text(point.application_identity)
          .scalar(candidate_identity)
          .scalar(static_cast<std::uint64_t>(candidate_counts[block]))
          .scalar(std::uint64_t{pending->size()});
      for (const PendingFace& face : *pending) {
        exact.scalar(std::uint32_t{face.role == fragment_role_type::Coarse ? 0u : 1u})
            .scalar(std::int32_t{face.axis});
        for (int axis = 0; axis < Dim; ++axis)
          exact.scalar(face.face[axis]);
        for (int axis = 0; axis < Dim; ++axis)
          exact.scalar(face.coarse_face[axis]);
        exact.scalar(face.measure);
      }
      pending_contract.emplace(std::move(exact).release());
    }
  } catch (...) {
    if (static_flux_tables_.bound && declared_basis_slot) {
      if (*declared_basis_slot < static_flux_basis_active_.size())
        static_flux_basis_active_[*declared_basis_slot] = 0;
      if (*declared_basis_slot < static_flux_basis_payloads_.size())
        static_flux_basis_payloads_[*declared_basis_slot].face_count = 0;
      if (static_cursor_restore && block < static_flux_tables_.next_basis_by_runtime_block.size())
        static_flux_tables_.next_basis_by_runtime_block[block] = *static_cursor_restore;
      if (static_count_restore)
        candidate_counts[block] = *static_count_restore;
      if (static_identity_restore)
        candidate_identity = *static_identity_restore;
    }
    preparation_error = std::current_exception();
  }
  if (all_reduce_max(preparation_error ? 1L : 0L, lane) != 0) {
    if (static_flux_tables_.bound && declared_basis_slot) {
      if (*declared_basis_slot < static_flux_basis_active_.size())
        static_flux_basis_active_[*declared_basis_slot] = 0;
      if (*declared_basis_slot < static_flux_basis_payloads_.size())
        static_flux_basis_payloads_[*declared_basis_slot].face_count = 0;
      if (static_cursor_restore && block < static_flux_tables_.next_basis_by_runtime_block.size())
        static_flux_tables_.next_basis_by_runtime_block[block] = *static_cursor_restore;
      if (static_count_restore)
        candidate_counts[block] = *static_count_restore;
      if (static_identity_restore)
        candidate_identity = *static_identity_restore;
    }
    if (facade_->prepared_amr_multiblock_hierarchy_().lane().size() == 1 && preparation_error)
      std::rethrow_exception(preparation_error);
    throw std::runtime_error("AMR Program face-flux preparation failed collectively");
  }
  if (static_prepared)
    return;
  if (!pending || !pending_contract)
    throw std::logic_error("AMR Program dynamic flux fallback has no prepared face witness");
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"cell-local-amr-pending-flux-faces", *pending_contract}}, lane))
    throw std::invalid_argument(
        "AMR Program pending face-flux order differs between execution ranks");

  std::vector<FluxBasisFace> faces;
  preparation_error = nullptr;
  try {
    faces.reserve(pending->size());
  } catch (...) {
    preparation_error = std::current_exception();
  }
  if (all_reduce_max(preparation_error ? 1L : 0L, lane) != 0) {
    if (facade_->prepared_amr_multiblock_hierarchy_().lane().size() == 1 && preparation_error)
      std::rethrow_exception(preparation_error);
    throw std::runtime_error("AMR Program face-flux basis reservation failed collectively");
  }

  for (const PendingFace& face : *pending) {
    std::vector<Real> payload =
        named_cell_fluxes != nullptr
            ? collective_named_face_payload_(*named_cell_fluxes, rhs, face.axis, face.face)
        : exact_face_fluxes != nullptr
            ? collective_face_payload_(*exact_face_fluxes, rhs, face.axis, face.face)
            : collective_face_payload_(*evaluation, rhs, face.axis, face.face);
    std::optional<FluxBasisFace> prepared_face;
    std::exception_ptr payload_error;
    try {
      if (!(face.measure > 0.0) || !std::isfinite(face.measure))
        throw std::invalid_argument("AMR Program flux basis has an invalid face measure");
      for (Real& component : payload) {
        if (!std::isfinite(static_cast<double>(component)))
          throw std::invalid_argument("AMR Program flux basis contains a non-finite payload");
        // Native finite-volume providers retain face-integrated fluxes, while the named
        // cell-centered provider above derives the authenticated face density directly.  The
        // metric ledger always receives a density and multiplies its face measure exactly once.
        if (named_cell_fluxes == nullptr)
          component /= static_cast<Real>(face.measure);
      }
      prepared_face.emplace(FluxBasisFace{face.role, face.axis, face.face, face.coarse_face,
                                          face.measure, std::move(payload)});
      faces.push_back(std::move(*prepared_face));
    } catch (...) {
      payload_error = std::current_exception();
    }
    if (all_reduce_max(payload_error ? 1L : 0L, lane) != 0) {
      if (facade_->prepared_amr_multiblock_hierarchy_().lane().size() == 1 && payload_error)
        std::rethrow_exception(payload_error);
      throw std::runtime_error("AMR Program face-flux basis failed collectively");
    }
  }

  const std::uint64_t identity =
      declared_basis_slot ? *declared_basis_slot : candidate_identity + 1;
  if (static_flux_tables_.bound) {
    if (!declared_basis_slot || *declared_basis_slot >= static_flux_basis_payloads_.size() ||
        *declared_basis_slot >= static_flux_basis_active_.size())
      throw std::logic_error("AMR Program flux basis has no resident static payload slot");
    FluxBasis& resident = static_flux_basis_payloads_[*declared_basis_slot];
    resident = FluxBasis{identity,     block,    active_level_, point,
                         rhs_identity, provider, interval,      std::move(faces)};
    static_flux_basis_active_[*declared_basis_slot] = 1;
    ++candidate_counts[block];
    candidate_identity = std::max(candidate_identity, identity);
    return;
  }
  std::optional<FluxExpression> prepared_expression;
  std::exception_ptr expression_error;
  try {
    auto basis = std::make_shared<const FluxBasis>(FluxBasis{
        identity, block, active_level_, point, rhs_identity, provider, interval, std::move(faces)});
    FluxExpression expression;
    expression.emplace(identity, FluxExpressionTerm{std::move(basis), {{0, {1, 1}}}});
    require_flux_expression_budget_(expression);
    prepared_expression.emplace(std::move(expression));
  } catch (...) {
    expression_error = std::current_exception();
  }
  if (all_reduce_max(expression_error ? 1L : 0L, lane) != 0) {
    if (facade_->prepared_amr_multiblock_hierarchy_().lane().size() == 1 && expression_error)
      std::rethrow_exception(expression_error);
    throw std::runtime_error("AMR Program flux-expression attachment failed collectively");
  }
  expression_error = nullptr;
  try {
    candidate_registry[&rhs] = std::move(*prepared_expression);
    ++candidate_counts[block];
    candidate_identity = std::max(candidate_identity, identity);
  } catch (...) {
    expression_error = std::current_exception();
  }
  if (all_reduce_max(expression_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && expression_error)
      std::rethrow_exception(expression_error);
    throw std::runtime_error("AMR Program flux-expression registry failed collectively");
  }
}
