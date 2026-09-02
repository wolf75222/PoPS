void materialize_active_flux_expression_(std::size_t runtime_block,
                                         const field_type& candidate) const {
  using fragment_key_type = ::pops::amr::reflux::FaceFluxFragmentKey<Dim>;
  using fragment_role_type = ::pops::amr::reflux::FaceLedgerRole;
  if (runtime_block >= active_attempt_states_.size() ||
      runtime_block >= active_incoming_flux_.size() ||
      runtime_block >= active_outgoing_flux_.size() ||
      runtime_block >= active_block_identities_.size() ||
      active_attempt_states_[runtime_block] != &candidate ||
      active_block_identities_[runtime_block].empty())
    throw std::logic_error("AMR Program final flux expression has no canonical block candidate");

  if (static_flux_tables_.bound) {
    if (runtime_block >= static_flux_tables_.basis_slots_by_runtime_block.size() ||
        runtime_block >= static_flux_tables_.term_slots_by_runtime_block.size())
      throw std::logic_error("AMR Program final flux has no compact runtime block carrier");
    const auto& basis_slots = static_flux_tables_.basis_slots_by_runtime_block[runtime_block];
    const auto applicable_basis_count = std::count_if(
        basis_slots.begin(), basis_slots.end(), [&](std::uint32_t basis_slot) {
          if (basis_slot >= static_flux_tables_.bases.size())
            throw std::logic_error("AMR Program final flux has an invalid basis carrier slot");
          const int declared_level = static_flux_tables_.bases[basis_slot].level;
          return declared_level < 0 || declared_level == active_level_;
        });
    if (static_flux_tables_.next_basis_by_runtime_block[runtime_block] !=
        static_cast<std::size_t>(applicable_basis_count))
      throw std::invalid_argument(
          "AMR Program final flux omitted one or more declared basis occurrences");
    const auto active_slot_slice = [&](const auto& route) -> std::span<const std::uint32_t> {
      if (route.substep_count == 0 || route.slots.size() % route.substep_count != 0)
        throw std::logic_error(
            "AMR Program final flux route has an invalid resident substep shape");
      const std::size_t substep = route.role == fragment_role_type::Fine
                                      ? static_cast<std::size_t>(logical_substep_)
                                      : std::size_t{0};
      if (logical_substep_ < 0 || substep >= route.substep_count)
        throw std::invalid_argument(
            "AMR Program final flux active child substep exceeds its resident route slices");
      const std::size_t faces_per_substep = route.slots.size() / route.substep_count;
      return std::span<const std::uint32_t>(route.slots)
          .subspan(substep * faces_per_substep, faces_per_substep);
    };
    for (const std::uint32_t term_slot :
         static_flux_tables_.term_slots_by_runtime_block[runtime_block]) {
      const auto& term = static_flux_tables_.terms.at(term_slot);
      if (term.basis_slot >= static_flux_tables_.bases.size())
        throw std::logic_error("AMR Program final flux term has no declared basis");
      const int declared_level = static_flux_tables_.bases[term.basis_slot].level;
      if (declared_level >= 0 && declared_level != active_level_)
        continue;
      if (term.basis_slot >= static_flux_basis_payloads_.size() ||
          term.basis_slot >= static_flux_basis_active_.size() ||
          static_flux_basis_active_[term.basis_slot] == 0)
        throw std::invalid_argument("AMR Program final flux term has no active resident basis");
      const FluxBasis& basis = static_flux_basis_payloads_[term.basis_slot];
      if (basis.identity != term.basis_slot || basis.runtime_block != runtime_block ||
          basis.level != active_level_)
        throw std::invalid_argument(
            "AMR Program final flux resident basis differs from its table slot");
      const double duration = basis.window.end.physical_time - basis.window.begin.physical_time;
      if (!std::isfinite(duration) || !(duration > 0.0))
        throw std::invalid_argument("AMR Program final flux resident basis has invalid duration");
      const ::pops::amr::Rational stage_phase =
          basis.window.begin.phase +
          (basis.window.end.phase - basis.window.begin.phase) * basis.point.stage_fraction;
      const double stage_physical_time =
          basis.window.begin.physical_time + basis.point.stage_fraction.value() * duration;
      std::size_t coarse_face = 0;
      std::size_t fine_face = 0;
      if (basis.face_count > basis.faces.size())
        throw std::logic_error("AMR Program resident basis face count exceeds its frozen slots");
      for (const FluxBasisFace& face :
           std::span<const FluxBasisFace>(basis.faces).first(basis.face_count)) {
        multiblock_flux_ledger_type* ledger = face.role == fragment_role_type::Coarse
                                                  ? active_outgoing_flux_[runtime_block]
                                                  : active_incoming_flux_[runtime_block];
        if (ledger == nullptr)
          throw std::logic_error(
              "AMR Program flux basis targets no active hierarchy-transition ledger");
        const auto route = std::find_if(
            term.ledger_routes.begin(), term.ledger_routes.end(), [&](const auto& candidate) {
              return candidate.ledger == ledger && candidate.level == active_level_ &&
                     candidate.role == face.role;
            });
        if (route == term.ledger_routes.end())
          throw std::invalid_argument(
              "AMR Program flux table has no topology-bound resident ledger route");
        const std::span<const std::uint32_t> active_slots = active_slot_slice(*route);
        const std::size_t face_slot =
            face.role == fragment_role_type::Coarse ? coarse_face++ : fine_face++;
        if (face_slot >= active_slots.size())
          throw std::invalid_argument(
              "AMR Program flux topology has more faces than its prepared resident slots");
        ledger->accumulate_prepared(
            active_slots[face_slot],
            {active_level_, basis.window.begin.macro_step, stage_phase, stage_physical_time},
            active_subcycling_attempt_,
            {term.coefficient, basis.window.begin.phase, basis.window.end.phase, duration,
             face.face_measure},
            std::span<const Real>(face.flux_density));
      }
      const auto require_complete_route = [&](multiblock_flux_ledger_type* ledger,
                                              fragment_role_type role, std::size_t consumed) {
        if (ledger == nullptr)
          return;
        const auto route = std::find_if(
            term.ledger_routes.begin(), term.ledger_routes.end(), [&](const auto& candidate) {
              return candidate.ledger == ledger && candidate.level == active_level_ &&
                     candidate.role == role;
            });
        if (route == term.ledger_routes.end() || consumed != active_slot_slice(*route).size())
          throw std::invalid_argument(
              "AMR Program flux topology differs from its complete resident face route");
      };
      require_complete_route(active_outgoing_flux_[runtime_block], fragment_role_type::Coarse,
                             coarse_face);
      require_complete_route(active_incoming_flux_[runtime_block], fragment_role_type::Fine,
                             fine_face);
    }
    return;
  }

  const FluxExpression expression = active_flux_expression_(candidate);
  require_flux_expression_budget_(expression);
  std::vector<std::string> stage_identities;
  stage_identities.reserve(expression.size());
  for (const auto& [identity, term] : expression) {
    if (!term.basis || term.basis->identity != identity ||
        term.basis->runtime_block != runtime_block || term.basis->level != active_level_ ||
        term.coefficient.size() != 1 || term.coefficient.begin()->first != 1)
      throw std::invalid_argument(
          "AMR Program final flux coefficient is not a supported exact dt integral");
    const ::pops::amr::Rational weight = term.coefficient.begin()->second;
    if (weight.denominator <= 0 ||
        ::pops::amr::Rational{weight.numerator, weight.denominator} != weight)
      throw std::invalid_argument(
          "AMR Program final flux coefficient lost its canonical rational metadata");
    const FluxBasis& basis = *term.basis;
    if (basis.window.begin.level != active_level_ || basis.window.end.level != active_level_ ||
        basis.point.clock.empty() || basis.point.level != active_level_ ||
        basis.point.tick != basis.window.begin.macro_step ||
        basis.point.substep != logical_substep_ || basis.point.stage < 0 ||
        basis.rhs_identity < 0 ||
        (basis.provider != FluxBasisProvider::PreparedResidual &&
         basis.provider != FluxBasisProvider::PreparedDefaultFlux &&
         basis.provider != FluxBasisProvider::ExactFace &&
         basis.provider != FluxBasisProvider::NamedCell) ||
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
    const double duration = basis.window.end.physical_time - basis.window.begin.physical_time;
    if (!std::isfinite(duration) || !(duration > 0.0))
      throw std::invalid_argument("AMR Program flux basis has an invalid physical duration");
    stage_identities.push_back(
        "pops.program-flux-expression.v1/provider/" +
        std::to_string(static_cast<unsigned int>(basis.provider)) + "/rhs/" +
        std::to_string(basis.rhs_identity) + "/point-stage/" + std::to_string(basis.point.stage) +
        "/basis/" + std::to_string(identity) + "/dt-power/1/weight/" +
        std::to_string(weight.numerator) + "/" + std::to_string(weight.denominator) + "/stage/" +
        std::to_string(basis.point.stage_fraction.numerator) + "/" +
        std::to_string(basis.point.stage_fraction.denominator));
  }

  if (multiblock_flux_ledger_type* incoming = active_incoming_flux_[runtime_block];
      incoming != nullptr) {
    const std::string owner(active_block_identities_[runtime_block]);
    const std::string state = owner + "/state";
    const auto levels = ::pops::amr::reflux::LevelTransition{active_level_ - 1, active_level_};
    const auto coarse_entry = [&](const auto& entry, int axis, const Index<Dim>& coarse_face) {
      return entry.key.owner == owner && entry.key.state == state && entry.key.levels == levels &&
             entry.key.axis == axis && entry.key.coarse_face == coarse_face &&
             entry.key.attempt == active_subcycling_attempt_ &&
             entry.key.clock.macro_step == active_subcycling_window_.begin.macro_step &&
             entry.key.role == fragment_role_type::Coarse;
    };
    bool compared_face = false;
    for (const auto& [identity, term] : expression) {
      (void)identity;
      for (const FluxBasisFace& face : term.basis->faces) {
        if (face.role != fragment_role_type::Fine)
          continue;
        compared_face = true;
        const auto& entries = incoming->pending_entries(face.axis);
        const auto same_coarse_face = [&](const auto& entry) {
          return coarse_entry(entry, face.axis, face.coarse_face);
        };
        const std::size_t coarse_count = static_cast<std::size_t>(
            std::count_if(entries.begin(), entries.end(), same_coarse_face));
        const bool exact_operator_pack =
            coarse_count == stage_identities.size() &&
            std::all_of(
                stage_identities.begin(), stage_identities.end(), [&](const std::string& stage) {
                  return std::any_of(entries.begin(), entries.end(), [&](const auto& entry) {
                    return same_coarse_face(entry) && entry.key.stage == stage;
                  });
                });
        if (!exact_operator_pack) {
          std::string observed;
          for (const auto& entry : entries)
            if (same_coarse_face(entry)) {
              if (!observed.empty())
                observed += ", ";
              observed += entry.key.stage;
            }
          throw std::runtime_error(
              "AMR Program coarse/fine flux operator identities differ before face-flux "
              "publication: expected " +
              std::to_string(stage_identities.size()) + " stage(s), observed " +
              std::to_string(coarse_count) + " [" + observed + "]");
        }
      }
    }
    if (!compared_face) {
      bool coarse_face_exists = false;
      for (int axis = 0; axis < Dim && !coarse_face_exists; ++axis)
        coarse_face_exists = std::any_of(
            incoming->pending_entries(axis).begin(), incoming->pending_entries(axis).end(),
            [&](const auto& entry) {
              return entry.key.owner == owner && entry.key.state == state &&
                     entry.key.levels == levels && entry.key.axis == axis &&
                     entry.key.attempt == active_subcycling_attempt_ &&
                     entry.key.clock.macro_step == active_subcycling_window_.begin.macro_step &&
                     entry.key.role == fragment_role_type::Coarse;
            });
      if (coarse_face_exists)
        throw std::runtime_error(
            "AMR Program coarse/fine flux operator identities differ before face-flux "
            "publication");
    }
  }

  std::size_t stage_index = 0;
  for (const auto& [identity, term] : expression) {
    (void)identity;
    const FluxBasis& basis = *term.basis;
    const ::pops::amr::Rational weight = term.coefficient.begin()->second;
    const double duration = basis.window.end.physical_time - basis.window.begin.physical_time;
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
      key.state = key.owner + "/state";
      key.levels = face.role == fragment_role_type::Coarse
                       ? ::pops::amr::reflux::LevelTransition{active_level_, active_level_ + 1}
                       : ::pops::amr::reflux::LevelTransition{active_level_ - 1, active_level_};
      key.axis = face.axis;
      key.face = face.face;
      key.coarse_face = face.coarse_face;
      key.clock = {active_level_, basis.window.begin.macro_step, stage_phase, stage_physical_time};
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
}
