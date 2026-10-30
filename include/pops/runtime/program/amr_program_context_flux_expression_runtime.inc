void materialize_active_flux_expression_(
    std::size_t runtime_block, const field_type& candidate,
    const FluxExpression* prepared_expression = nullptr) const {
  long retained = 0;
  prepare_spatial_collectively([&] {
    using fragment_key_type = ::pops::amr::reflux::FaceFluxFragmentKey<Dim>;
    using fragment_role_type = ::pops::amr::reflux::FaceLedgerRole;
    if (runtime_block >= active_attempt_states_.size() ||
        runtime_block >= active_incoming_flux_.size() ||
        runtime_block >= active_outgoing_flux_.size() ||
        runtime_block >= active_block_identities_.size() ||
        active_attempt_states_[runtime_block] != &candidate ||
        active_block_identities_[runtime_block].empty())
      throw std::logic_error("AMR Program final flux expression has no canonical block candidate");

    const FluxExpression expression =
        prepared_expression ? *prepared_expression : active_flux_expression_(candidate);
    require_flux_expression_budget_(expression);
    retained = retain_spatial_materialization(runtime_block, candidate, expression);
    if (retained)
      return;
    const std::string& state_identity = facade_->prepared_amr_block_state_identity_(runtime_block);
    std::vector<std::string> stage_identities;
    stage_identities.reserve(expression.size());
    for (const auto& [identity, term] : expression) {
      if (!term.basis || term.basis->identity != identity ||
          term.basis->runtime_block != runtime_block || term.basis->level != active_level_ ||
          term.coefficient.size() != 1 || term.coefficient.begin()->first != 1)
        throw std::invalid_argument(
            "AMR Program final flux coefficient is not a supported exact dt integral");
      const ::pops::ExactWeight weight = term.coefficient.begin()->second;
      const FluxBasis& basis = *term.basis;
      if (basis.window.begin.level != active_level_ || basis.window.end.level != active_level_ ||
          basis.point.clock.empty() || basis.point.level != active_level_ ||
          basis.point.tick != basis.window.begin.macro_step ||
          basis.point.substep != logical_substep_ || basis.point.stage < 0 ||
          basis.rhs_identity < 0 ||
          (basis.provider != FluxBasisProvider::PreparedResidual &&
           basis.provider != FluxBasisProvider::PreparedDefaultFlux &&
           basis.provider != FluxBasisProvider::ExactFace &&
           basis.provider != FluxBasisProvider::NamedCell &&
           basis.provider != FluxBasisProvider::DiffusiveFace &&
           basis.provider != FluxBasisProvider::GeneratedFace) ||
          (basis.provider != FluxBasisProvider::ExactFace &&
           basis.point.stage != basis.rhs_identity) ||
          basis.window.begin.macro_step != active_subcycling_window_.begin.macro_step ||
          basis.window.end.macro_step != active_subcycling_window_.end.macro_step ||
          basis.window.begin.phase < active_subcycling_window_.begin.phase ||
          active_subcycling_window_.end.phase < basis.window.end.phase ||
          !(basis.window.begin.phase < basis.window.end.phase) ||
          basis.point.stage_fraction.denominator <= 0 || basis.point.stage_fraction.numerator < 0 ||
          basis.point.stage_fraction.numerator > basis.point.stage_fraction.denominator)
        throw std::invalid_argument(
            "AMR Program flux basis lies outside its canonical level/substep window");
      const double duration = active_flux_interval_duration_(basis.window);
      if (!std::isfinite(duration) || !(duration > 0.0))
        throw std::invalid_argument("AMR Program flux basis has an invalid physical duration");
      (void)declared_flux_temporal_family_(basis);
      stage_identities.push_back(flux_expression_stage_identity_(basis, weight));
    }
    if (multiblock_flux_ledger_type* incoming = active_incoming_flux_[runtime_block];
        incoming != nullptr) {
      using incoming_entry_type = typename multiblock_flux_ledger_type::Entry;
      using face_coordinates_type = std::array<int, Dim>;
      const std::string owner(active_block_identities_[runtime_block]);
      const auto levels = ::pops::amr::reflux::LevelTransition{active_level_ - 1, active_level_};
      // Select the pending coarse fragments of this block transition once, grouped by axis and
      // coarse face in ledger order: a fine face then reads only the fragments of its own coarse
      // face instead of every pending fragment of its axis.
      std::array<std::map<face_coordinates_type, std::vector<const incoming_entry_type*>>, Dim>
          coarse_entries;
      for (int axis = 0; axis < Dim; ++axis)
        for (const auto& entry : incoming->pending_entries(axis))
          if (entry.key.owner == owner && entry.key.state == state_identity &&
              entry.key.levels == levels && entry.key.axis == axis &&
              entry.key.attempt == active_subcycling_attempt_ &&
              entry.key.clock.macro_step == active_subcycling_window_.begin.macro_step &&
              entry.key.role == fragment_role_type::Coarse)
            coarse_entries[static_cast<std::size_t>(axis)]
                          [::pops::amr::reflux::detail::coordinate_array(entry.key.coarse_face)]
                              .push_back(&entry);
      // The temporal families of the fine faces of each coarse face, over every flux basis.
      std::array<std::map<face_coordinates_type, std::set<std::string>>, Dim> fine_families;
      if (!declared_flux_temporal_families_.empty())
        for (const auto& [source_identity, source_term] : expression) {
          (void)source_identity;
          const auto& source = *source_term.basis;
          for (const FluxBasisFace& source_face : source.faces)
            if (source_face.role == fragment_role_type::Fine)
              fine_families[static_cast<std::size_t>(source_face.axis)]
                           [::pops::amr::reflux::detail::coordinate_array(source_face.coarse_face)]
                               .emplace(declared_flux_temporal_family_(source));
        }
      const std::vector<const incoming_entry_type*> no_entries;
      // The comparison depends only on the coarse face, so its first fine face decides it.
      std::array<std::set<face_coordinates_type>, Dim> compared_coarse_faces;
      bool compared_face = false;
      for (const auto& [identity, term] : expression) {
        (void)identity;
        for (const FluxBasisFace& face : term.basis->faces) {
          if (face.role != fragment_role_type::Fine)
            continue;
          compared_face = true;
          const std::size_t axis = static_cast<std::size_t>(face.axis);
          const face_coordinates_type coarse_face =
              ::pops::amr::reflux::detail::coordinate_array(face.coarse_face);
          if (!compared_coarse_faces[axis].insert(coarse_face).second)
            continue;
          const auto selected = coarse_entries[axis].find(coarse_face);
          const auto& entries =
              selected == coarse_entries[axis].end() ? no_entries : selected->second;
          const std::size_t coarse_count = entries.size();
          bool exact_operator_pack = false;
          if (declared_flux_temporal_families_.empty()) {
            // Legacy producers have no separate spatial authority. Preserve the
            // complete operator/stage/weight identity instead of inventing one.
            exact_operator_pack =
                coarse_count == stage_identities.size() &&
                std::all_of(
                    stage_identities.begin(), stage_identities.end(), [&](const auto& stage) {
                      return std::any_of(entries.begin(), entries.end(), [&](const auto* entry) {
                        return entry->key.stage == stage;
                      });
                    });
          } else {
            std::set<std::string> coarse_families;
            for (const auto* entry : entries) {
              ::pops::amr::reflux::validate_face_flux_fragment(entry->key, entry->measure);
              require_declared_flux_fragment_stage_(runtime_block, *entry);
              coarse_families.emplace(entry->key.temporal_family);
            }
            // A physical flux family is independent of the number, dates and
            // identities of its local evaluations. Retain every original stage
            // and weight. Metric reflux independently verifies each family's
            // unit quadrature, exact temporal coverage, physical duration and
            // complete coarse/fine geometry before applying the correction.
            const auto fine = fine_families[axis].find(coarse_face);
            exact_operator_pack = fine != fine_families[axis].end() && !fine->second.empty() &&
                                  fine->second == coarse_families;
          }
          if (!exact_operator_pack) {
            std::string expected;
            for (const auto& stage : stage_identities) {
              if (!expected.empty())
                expected += ", ";
              expected += stage;
            }
            std::string observed;
            for (const auto* entry : entries) {
              if (!observed.empty())
                observed += ", ";
              observed += entry->key.stage;
            }
            throw std::runtime_error(
                "AMR Program coarse/fine flux operator identities differ before face-flux "
                "publication: expected " +
                std::to_string(stage_identities.size()) + " stage(s), observed " +
                std::to_string(coarse_count) + " [expected: " + expected +
                "; observed: " + observed + "]");
          }
        }
      }
      if (!compared_face) {
        bool coarse_face_exists = false;
        for (int axis = 0; axis < Dim && !coarse_face_exists; ++axis)
          coarse_face_exists = !coarse_entries[static_cast<std::size_t>(axis)].empty();
        if (coarse_face_exists)
          throw std::runtime_error(
              "AMR Program coarse/fine flux operator identities differ before face-flux "
              "publication");
      }
    }

    std::size_t stage_index = 0;
    for (const auto& [identity, term] : expression) {
      const FluxBasis& basis = *term.basis;
      const ::pops::ExactWeight weight = term.coefficient.begin()->second;
      const double duration = active_flux_interval_duration_(basis.window);
      const ::pops::amr::Rational stage_phase =
          basis.window.begin.phase +
          (basis.window.end.phase - basis.window.begin.phase) * basis.point.stage_fraction;
      const double stage_physical_time =
          basis.window.begin.physical_time + basis.point.stage_fraction.value() * duration;
      for (const FluxBasisFace& face : basis.faces) {
        multiblock_flux_ledger_type* ledger = face.role == fragment_role_type::Coarse
                                                  ? active_outgoing_flux_[runtime_block]
                                                  : active_incoming_flux_[runtime_block];
        if (ledger == nullptr)
          throw std::logic_error(
              "AMR Program flux basis targets no active hierarchy-transition ledger");
        fragment_key_type key;
        key.owner = std::string(active_block_identities_[runtime_block]);
        key.state = state_identity;
        key.levels = face.role == fragment_role_type::Coarse
                         ? ::pops::amr::reflux::LevelTransition{active_level_, active_level_ + 1}
                         : ::pops::amr::reflux::LevelTransition{active_level_ - 1, active_level_};
        key.axis = face.axis;
        key.face = face.face;
        key.coarse_face = face.coarse_face;
        key.clock = {active_level_, basis.window.begin.macro_step, stage_phase,
                     stage_physical_time};
        key.temporal_family = basis.temporal_family;
        key.stage = stage_identities[stage_index];
        key.attempt = active_subcycling_attempt_;
        key.role = face.role;
        ledger->accumulate(
            std::move(key),
            {weight, basis.window.begin.phase, basis.window.end.phase, duration, face.face_measure},
            face.flux_density);
      }
      ++stage_index;
    }
  });
  const auto& lane = prepared_execution_lane();
  if (all_reduce_min(retained, lane) != all_reduce_max(retained, lane))
    throw std::logic_error("composite implicit materialization proof differs between ranks");
}

std::vector<std::uint8_t> spatial_weighted_fragments(
    const multiblock_flux_ledger_type& ledger) const {
  checkpoint_detail::Writer writer;
  for (int axis = 0; axis < Dim; ++axis) {
    const auto& entries = ledger.published_entries(axis);
    writer.size(entries.size());
    for (const auto& entry : entries)
      checkpoint_detail::write_face_fragment(writer, entry);
  }
  return std::move(writer).take();
}
