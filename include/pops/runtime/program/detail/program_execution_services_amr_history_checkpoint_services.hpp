// The static ABI carrier deliberately keeps the active RHS payload in a reusable image.  Numeric
// histories, however, outlive both a candidate attempt and that carrier.  Their provenance image
// is therefore independently allocated during cold subcycling prime and only overwritten in
// place below.  In particular, do not retain a shared pointer to static_flux_basis_payloads_.
static void copy_static_history_flux_basis_preallocated_(FluxBasis& destination,
                                                         const FluxBasis& source,
                                                         std::uint64_t identity) {
  if (destination.faces.size() != source.faces.size())
    throw std::logic_error("AMR Program static history flux face shape changed after bind");
  if (!source.point.clock.empty() || !source.point.graph_identity.empty() ||
      !source.point.rate_identity.empty() || !source.point.application_identity.empty())
    throw std::logic_error("AMR Program static flux carrier exposed an unprepared string");
  destination.identity = identity;
  destination.runtime_block = source.runtime_block;
  destination.level = source.level;
  // `clock` was copied into the per-history image during cold prime.  The static carrier omits
  // textual point identities from its hot representation, so preserve that authenticated value.
  destination.point.tick = source.point.tick;
  destination.point.level = source.point.level;
  destination.point.substep = source.point.substep;
  destination.point.stage = source.point.stage;
  destination.point.stage_fraction = source.point.stage_fraction;
  destination.point.dt = source.point.dt;
  destination.point.physical_time = source.point.physical_time;
  destination.rhs_identity = source.rhs_identity;
  destination.provider = source.provider;
  destination.window = source.window;
  destination.face_count = source.face_count;
  if (destination.face_count > destination.faces.size())
    throw std::logic_error("AMR Program static history flux face count exceeds its frozen image");
  for (std::size_t index = 0; index < source.faces.size(); ++index) {
    const FluxBasisFace& input = source.faces[index];
    FluxBasisFace& output = destination.faces[index];
    if (output.flux_density.size() != input.flux_density.size())
      throw std::logic_error("AMR Program static history flux density shape changed after bind");
    output.role = input.role;
    output.axis = input.axis;
    output.face = input.face;
    output.coarse_face = input.coarse_face;
    output.face_measure = input.face_measure;
    std::copy(input.flux_density.begin(), input.flux_density.end(), output.flux_density.begin());
  }
}

void require_static_history_flux_provenance_(const std::vector<field_type>& ring,
                                             const std::vector<FluxExpression>& expressions) const {
  if (!static_flux_tables_.bound || expressions.empty() ||
      expressions.size() != ring.size())
    throw std::logic_error("AMR Program static history flux image is not cold-primed");
  for (const FluxExpression& expression : expressions) {
    for (const auto& [basis_slot, term] : expression) {
      if (!term.basis || basis_slot >= static_flux_basis_payloads_.size() ||
          basis_slot >= static_flux_basis_active_.size() ||
          basis_slot >= static_flux_tables_.bases.size() ||
          term.basis->faces.size() != static_flux_basis_payloads_[basis_slot].faces.size())
        throw std::logic_error("AMR Program static history flux image drifted after bind");
      if (term.basis->point.clock.empty())
        throw std::logic_error("AMR Program static history flux image lost its clock authority");
    }
  }
}

void store_static_history_flux_provenance_(const std::vector<field_type>& ring,
                                           std::vector<FluxExpression>& expressions,
                                           bool initialize) const {
  require_static_history_flux_provenance_(ring, expressions);
  const std::size_t slots = initialize ? expressions.size() : std::size_t{1};
  for (std::size_t slot = 0; slot < slots; ++slot) {
    for (auto& [basis_slot, term] : expressions[slot]) {
      FluxBasis* image = const_cast<FluxBasis*>(term.basis.get());
      if (image == nullptr)
        throw std::logic_error("AMR Program static history flux image has no basis");
      const FluxBasis& active = static_flux_basis_payloads_[basis_slot];
      if (static_flux_basis_active_[basis_slot] == 0) {
        image->identity = kStaticHistoryFluxInactiveIdentity;
        image->face_count = 0;
        continue;
      }
      copy_static_history_flux_basis_preallocated_(*image, active, basis_slot);
    }
  }
}

[[nodiscard]] std::map<std::string, std::vector<FluxExpression>>
prepare_static_history_flux_provenance_at_bind_(const PreparedFluxTableCarrier& tables,
                                                const std::vector<FluxBasis>& payloads) const {
  const auto& manager = runtime_state().hist_;
  if (manager.histories.size() != history_flux_expressions_.size() ||
      manager.owner.size() != history_flux_expressions_.size())
    throw std::logic_error("AMR Program static history flux registry changed before bind");
  for (auto& [key, ring] : manager.histories) {
    const auto previous = history_flux_expressions_.find(key);
    if (previous == history_flux_expressions_.end() || previous->second.size() != ring.size())
      throw std::logic_error("AMR Program static history flux ring depth changed before bind");
  }
  return prepare_static_history_flux_provenance_from_sealed_history_(
      tables, payloads, manager.owner, history_levels_, history_flux_expressions_, primary_clock_);
}

[[nodiscard]] static std::map<std::string, std::vector<FluxExpression>>
prepare_static_history_flux_provenance_from_sealed_history_(
    const PreparedFluxTableCarrier& tables, const std::vector<FluxBasis>& payloads,
    const std::map<std::string, int>& history_owners,
    const std::map<std::string, int>& history_levels,
    const std::map<std::string, std::vector<FluxExpression>>& history_flux_expressions,
    std::string_view primary_clock) {
  if (!tables.bound)
    return history_flux_expressions;
  if (payloads.size() != tables.bases.size())
    throw std::logic_error("AMR Program static history flux payload carrier is not dense");
  std::vector<ExactPolynomial> coefficients(tables.bases.size());
  for (const auto& term : tables.terms) {
    if (term.basis_slot >= coefficients.size() || term.coefficient.numerator == 0 ||
        term.coefficient.denominator <= 0)
      throw std::logic_error("AMR Program static history flux term is malformed after bind");
    add_exact_polynomial_(coefficients[term.basis_slot], {{1, term.coefficient}});
  }

  std::map<std::string, std::vector<FluxExpression>> candidate;
  for (const auto& [key, restored_slots] : history_flux_expressions) {
    const auto owner = history_owners.find(key);
    const auto level = history_levels.find(key);
    if (owner == history_owners.end() || level == history_levels.end() || owner->second < 0)
      throw std::logic_error("AMR Program static history flux has a foreign history identity");
    std::vector<FluxExpression> slots(restored_slots.size());
    for (std::size_t slot = 0; slot < slots.size(); ++slot) {
      const FluxExpression& restored = restored_slots[slot];
      for (const auto& basis : tables.bases) {
        if (basis.runtime_block != static_cast<std::uint32_t>(owner->second) ||
            (basis.level >= 0 && basis.level != level->second) ||
            coefficients[basis.basis_slot].empty())
          continue;
        FluxBasis image = payloads[basis.basis_slot];
        image.identity = kStaticHistoryFluxInactiveIdentity;
        image.runtime_block = basis.runtime_block;
        image.level = basis.level >= 0 ? basis.level : level->second;
        image.rhs_identity = basis.rhs_identity;
        image.provider = static_cast<FluxBasisProvider>(basis.provider);
        image.point.level = image.level;
        image.point.stage_fraction = basis.stage;
        image.face_count = 0;
        image.point.clock = primary_clock;
        const auto retained = restored.find(basis.basis_slot);
        if (retained != restored.end() && retained->second.basis &&
            retained->second.basis->identity == basis.basis_slot) {
          const FluxBasis& restored_basis = *retained->second.basis;
          if (restored_basis.runtime_block != image.runtime_block ||
              restored_basis.level != image.level ||
              restored_basis.rhs_identity != image.rhs_identity ||
              restored_basis.provider != image.provider)
            throw std::logic_error(
                "AMR Program restored static history flux identity differs from its successor "
                "(block " +
                std::to_string(restored_basis.runtime_block) + " -> " +
                std::to_string(image.runtime_block) + ", level " +
                std::to_string(restored_basis.level) + " -> " + std::to_string(image.level) +
                ", rhs " + std::to_string(restored_basis.rhs_identity) + " -> " +
                std::to_string(image.rhs_identity) + ", provider " +
                std::to_string(static_cast<unsigned int>(restored_basis.provider)) + " -> " +
                std::to_string(static_cast<unsigned int>(image.provider)) + ")");
          image.identity = restored_basis.identity;
          image.point = restored_basis.point;
          image.window = restored_basis.window;
          image.face_count = image.faces.size();
          // Regrid transfers are keyed by the complete geometric face identity, never by the old
          // vector position.  Exact successor faces retain their accepted density; newly exposed
          // faces keep the cold-primed zero image and are supplied by the pending parent remap.
          for (std::size_t successor_index = 0; successor_index < image.faces.size();
               ++successor_index) {
            FluxBasisFace& successor_face = image.faces[successor_index];
            const auto matches = [&](const FluxBasisFace& previous_face) {
              return previous_face.role == successor_face.role &&
                     previous_face.axis == successor_face.axis &&
                     previous_face.face == successor_face.face &&
                     previous_face.coarse_face == successor_face.coarse_face &&
                     previous_face.face_measure == successor_face.face_measure;
            };
            // A basis can visit the same geometric face through several sealed ledger routes.
            // The route identity is represented by its stable occurrence ordinal within the
            // complete geometric key; never fall back to the unrelated global vector position.
            std::size_t occurrence = 0;
            for (std::size_t prior = 0; prior < successor_index; ++prior)
              if (matches(image.faces[prior]))
                ++occurrence;
            if (restored_basis.face_count > restored_basis.faces.size())
              throw std::logic_error(
                  "AMR Program restored static history flux face count exceeds its image");
            auto previous = restored_basis.faces.begin();
            const auto previous_end = restored_basis.faces.begin() +
                                      static_cast<std::ptrdiff_t>(restored_basis.face_count);
            for (; previous != previous_end; ++previous) {
              if (!matches(*previous))
                continue;
              if (occurrence == 0)
                break;
              --occurrence;
            }
            if (previous == previous_end)
              continue;
            if (previous->flux_density.size() != successor_face.flux_density.size())
              throw std::logic_error(
                  "AMR Program restored static history flux density differs from its successor");
            std::copy(previous->flux_density.begin(), previous->flux_density.end(),
                      successor_face.flux_density.begin());
          }
        }
        if (!slots[slot]
                 .emplace(basis.basis_slot,
                          FluxExpressionTerm{std::make_shared<FluxBasis>(std::move(image)),
                                             coefficients[basis.basis_slot]})
                 .second)
          throw std::logic_error("AMR Program static history flux basis is not unique");
      }
    }
    candidate.emplace(key, std::move(slots));
  }
  if (candidate.size() != history_flux_expressions.size() ||
      history_owners.size() != history_flux_expressions.size() ||
      history_levels.size() != history_flux_expressions.size())
    throw std::logic_error("AMR Program static history flux registry changed before bind");
  return candidate;
}

static FluxExpression clone_history_flux_expression_for_mutation_bind_(
    const FluxExpression& source) {
  FluxExpression result;
  for (const auto& [identity, term] : source) {
    if (!term.basis)
      throw std::logic_error("AMR Program history mutation has no bindable flux basis");
    auto basis = std::make_shared<FluxBasis>(*term.basis);
    if (!result.emplace(identity, FluxExpressionTerm{std::move(basis), term.coefficient}).second)
      throw std::logic_error("AMR Program history mutation has a duplicate flux occurrence");
  }
  return result;
}

static void require_history_flux_expression_mutation_shape_(const FluxExpression& destination,
                                                            const FluxExpression& source) {
  if (destination.size() != source.size())
    throw std::logic_error("AMR Program history flux mutation shape changed after bind");
  auto target = destination.begin();
  auto input = source.begin();
  for (; input != source.end(); ++input, ++target) {
    if (target->first != input->first || !target->second.basis || !input->second.basis ||
        target->second.coefficient.size() != input->second.coefficient.size())
      throw std::logic_error("AMR Program history flux mutation identity changed after bind");
    const FluxBasis& target_basis = *target->second.basis;
    const FluxBasis& input_basis = *input->second.basis;
    if (target_basis.faces.size() != input_basis.faces.size() ||
        target_basis.point.clock.capacity() < input_basis.point.clock.size() ||
        target_basis.point.graph_identity.capacity() < input_basis.point.graph_identity.size() ||
        target_basis.point.rate_identity.capacity() < input_basis.point.rate_identity.size() ||
        target_basis.point.application_identity.capacity() <
            input_basis.point.application_identity.size())
      throw std::logic_error("AMR Program history flux mutation capacity changed after bind");
    for (std::size_t face = 0; face < target_basis.faces.size(); ++face)
      if (target_basis.faces[face].flux_density.size() !=
          input_basis.faces[face].flux_density.size())
        throw std::logic_error("AMR Program history flux mutation face shape changed after bind");
    auto target_coefficient = target->second.coefficient.begin();
    auto input_coefficient = input->second.coefficient.begin();
    for (; input_coefficient != input->second.coefficient.end();
         ++input_coefficient, ++target_coefficient)
      if (target_coefficient->first != input_coefficient->first)
        throw std::logic_error("AMR Program history flux mutation polynomial changed after bind");
  }
}

static void copy_history_mutation_string_(std::string& destination, const std::string& source) {
  if (destination.capacity() < source.size())
    throw std::length_error("AMR Program history mutation string capacity changed after bind");
  destination.assign(source.data(), source.size());
}

static void copy_history_flux_expression_for_mutation_(FluxExpression& destination,
                                                       const FluxExpression& source) {
  require_history_flux_expression_mutation_shape_(destination, source);
  auto target = destination.begin();
  auto input = source.begin();
  for (; input != source.end(); ++input, ++target) {
    FluxBasis& target_basis = *const_cast<FluxBasis*>(target->second.basis.get());
    const FluxBasis& input_basis = *input->second.basis;
    target_basis.identity = input_basis.identity;
    target_basis.runtime_block = input_basis.runtime_block;
    target_basis.level = input_basis.level;
    copy_history_mutation_string_(target_basis.point.clock, input_basis.point.clock);
    target_basis.point.tick = input_basis.point.tick;
    target_basis.point.level = input_basis.point.level;
    target_basis.point.substep = input_basis.point.substep;
    target_basis.point.stage = input_basis.point.stage;
    target_basis.point.stage_fraction = input_basis.point.stage_fraction;
    target_basis.point.dt = input_basis.point.dt;
    target_basis.point.physical_time = input_basis.point.physical_time;
    copy_history_mutation_string_(target_basis.point.graph_identity,
                                  input_basis.point.graph_identity);
    copy_history_mutation_string_(target_basis.point.rate_identity,
                                  input_basis.point.rate_identity);
    copy_history_mutation_string_(target_basis.point.application_identity,
                                  input_basis.point.application_identity);
    target_basis.rhs_identity = input_basis.rhs_identity;
    target_basis.provider = input_basis.provider;
    target_basis.window = input_basis.window;
    target_basis.face_count = input_basis.face_count;
    if (target_basis.face_count > target_basis.faces.size())
      throw std::logic_error("AMR Program history flux mutation face count exceeds bind image");
    for (std::size_t face = 0; face < target_basis.faces.size(); ++face) {
      FluxBasisFace& target_face = target_basis.faces[face];
      const FluxBasisFace& input_face = input_basis.faces[face];
      target_face.role = input_face.role;
      target_face.axis = input_face.axis;
      target_face.face = input_face.face;
      target_face.coarse_face = input_face.coarse_face;
      target_face.face_measure = input_face.face_measure;
      std::copy(input_face.flux_density.begin(), input_face.flux_density.end(),
                target_face.flux_density.begin());
    }
    auto target_coefficient = target->second.coefficient.begin();
    auto input_coefficient = input->second.coefficient.begin();
    for (; input_coefficient != input->second.coefficient.end();
         ++input_coefficient, ++target_coefficient)
      target_coefficient->second = input_coefficient->second;
  }
}

/// Bindings are prepared for every configured level, while the native history manager owns only
/// materialized levels.  Keep the inactive rows explicit rather than inventing a live map source
/// for a level that cannot yet be observed.  A cold rebind must replace this sentinel before that
/// level can participate in an accepted image.
static constexpr std::size_t kInactiveHistoryMutationOrdinal_ =
    std::numeric_limits<std::size_t>::max();

void prime_history_mutation_workspace_at_bind_() const {
  if (!active_attempt_states_.empty())
    throw std::logic_error("AMR Program history mutation cannot cold-prime during a candidate");
  auto& manager = runtime_state().hist_;
  std::vector<PreparedHistoryMutationSlot> slots;
  slots.reserve(manager.histories.size());
  constexpr std::string_view rotation_tag = "pops.amr.history-rotate.v2";
  std::size_t maximum_clock_filter_capacity = 0;
  std::size_t rotation_capacity = rotation_tag.size() + 3 * sizeof(std::uint64_t);
  for (auto& [key, ring] : manager.histories) {
    const auto decoded = decode_history_key_(key);
    const auto level = history_levels_.find(key);
    const auto clock = manager.clock_identity.find(key);
    const auto dts = manager.slot_dt.find(key);
    const auto flux = history_flux_expressions_.find(key);
    if (!decoded || level == history_levels_.end() || clock == manager.clock_identity.end() ||
        dts == manager.slot_dt.end() || flux == history_flux_expressions_.end() ||
        decoded->first != level->second || ring.empty() || ring.size() != dts->second.size() ||
        ring.size() != flux->second.size())
      throw std::logic_error("AMR Program history mutation registry is incomplete at bind");
    PreparedHistoryMutationSlot slot;
    slot.name = std::move(decoded->second);
    slot.key = key;
    slot.clock_identity = clock->second;
    slot.level = level->second;
    const auto initialized = manager.initialized.find(key);
    const auto pending = manager.store_pending.find(key);
    const auto fill_count = manager.fill_count.find(key);
    const auto depth = manager.depth.find(key);
    if (initialized == manager.initialized.end() || pending == manager.store_pending.end() ||
        fill_count == manager.fill_count.end() || depth == manager.depth.end())
      throw std::logic_error("AMR Program history mutation scalar registry is incomplete at bind");
    slot.live_ring = std::addressof(ring);
    slot.live_dts = std::addressof(dts->second);
    slot.live_expressions = std::addressof(flux->second);
    slot.live_initialized = std::addressof(initialized->second);
    slot.live_store_pending = std::addressof(pending->second);
    slot.live_fill_count = std::addressof(fill_count->second);
    slot.live_depth = std::addressof(depth->second);
    slot.live_clock_identity = std::addressof(clock->second);
    const auto pending_remap = pending_history_remaps_.find(key);
    slot.live_pending_remap = pending_remap == pending_history_remaps_.end()
                                  ? nullptr
                                  : std::addressof(pending_remap->second);
    slot.rollback_ring.reserve(ring.size());
    for (const field_type& field : ring)
      slot.rollback_ring.push_back(make_scratch_(field, field.ncomp(), field.ghosts()));
    slot.dts = dts->second;
    slot.expressions.reserve(flux->second.capacity());
    for (const FluxExpression& expression : flux->second)
      slot.expressions.push_back(clone_history_flux_expression_for_mutation_bind_(expression));
    const std::size_t store_capacity =
        sizeof("pops.amr.history-store.v2") + slot.key.size() + 6 * sizeof(std::uint64_t);
    slot.store_contract.reserve(store_capacity);
    maximum_clock_filter_capacity =
        std::max(maximum_clock_filter_capacity, slot.clock_identity.size());
    rotation_capacity += slot.key.size() + slot.clock_identity.size() + 3 * sizeof(std::uint64_t);
    slots.push_back(std::move(slot));
  }
  rotation_capacity += maximum_clock_filter_capacity;
  prepared_history_rotation_contract_.reserve(rotation_capacity);
  prepared_history_mutation_slots_.swap(slots);
  auto& staging = accepted_state_staging_;
  if (!staging.prepared_envelope ||
      staging.history_slot_bindings.size() != staging.history_slot_pool.size() ||
      staging.pending_history_keys.size() != staging.pending_history_remap_slots.size())
    throw std::logic_error("AMR Program history ordinal staging envelope is not cold-prepared");
  std::vector<std::size_t> history_ordinals;
  history_ordinals.reserve(staging.history_slot_bindings.size());
  for (const auto& binding : staging.history_slot_bindings) {
    if (binding.state_slot >= staging.history_slot_pool.size())
      throw std::logic_error("AMR Program history ordinal has an invalid staging slot");
    const int level = staging.history_slot_pool[binding.state_slot].level;
    if (level < 0)
      throw std::logic_error("AMR Program history ordinal has a negative configured level");
    if (static_cast<std::size_t>(level) >= runtime_->hierarchy().num_levels()) {
      history_ordinals.push_back(kInactiveHistoryMutationOrdinal_);
      continue;
    }
    std::size_t match = prepared_history_mutation_slots_.size();
    for (std::size_t index = 0; index < prepared_history_mutation_slots_.size(); ++index) {
      const auto& candidate = prepared_history_mutation_slots_[index];
      if (candidate.key != binding.key || binding.source_slot >= candidate.rollback_ring.size())
        continue;
      if (match != prepared_history_mutation_slots_.size())
        throw std::logic_error("AMR Program history ordinal has an ambiguous bind key");
      match = index;
    }
    if (match == prepared_history_mutation_slots_.size())
      throw std::logic_error("AMR Program history ordinal has no bind-sealed source");
    history_ordinals.push_back(match);
  }
  std::vector<const AmrProgramPendingHistoryRemap*> pending_ordinals;
  pending_ordinals.reserve(staging.pending_history_keys.size());
  for (const std::string& key : staging.pending_history_keys) {
    const auto pending = pending_history_remaps_.find(key);
    // No remap marker is the normal initial bind state.  Preserve that absent ordinal explicitly;
    // a cold rebind after remap/import installs the exact map-owned source before Candidate can
    // read it.  Inventing a map node here would turn an absent marker into a false lag read.
    if (pending == pending_history_remaps_.end()) {
      pending_ordinals.push_back(nullptr);
      continue;
    }
    if (pending->first != key || pending->second.key != key)
      throw std::logic_error("AMR Program pending-history ordinal changed its exact key");
    pending_ordinals.push_back(std::addressof(pending->second));
  }
  accepted_history_binding_mutation_slots_.swap(history_ordinals);
  accepted_pending_history_ordinal_sources_.swap(pending_ordinals);
  accepted_history_ordinal_owner_ = std::addressof(manager);
  accepted_history_ordinal_epoch_ = resource_epoch_;
  accepted_history_ordinal_generation_ = resource_generation_;
  prepared_history_mutation_epoch_ = resource_epoch_;
  prepared_history_mutation_generation_ = resource_generation_;
}

void rebind_history_mutation_workspace_preallocated_after_restore_() const noexcept {
  auto& manager = runtime_state().hist_;
  if (prepared_history_mutation_slots_.size() != manager.histories.size() ||
      history_levels_.size() != manager.histories.size() ||
      history_flux_expressions_.size() != manager.histories.size() ||
      manager.clock_identity.size() != manager.histories.size() ||
      manager.slot_dt.size() != manager.histories.size() ||
      manager.initialized.size() != manager.histories.size() ||
      manager.store_pending.size() != manager.histories.size() ||
      manager.fill_count.size() != manager.histories.size() ||
      manager.depth.size() != manager.histories.size() ||
      accepted_history_binding_mutation_slots_.size() !=
          accepted_state_staging_.history_slot_bindings.size() ||
      accepted_pending_history_ordinal_sources_.size() !=
          accepted_state_staging_.pending_history_keys.size())
    std::terminate();

  auto histories = manager.histories.begin();
  auto levels = history_levels_.begin();
  auto clocks = manager.clock_identity.begin();
  auto dts = manager.slot_dt.begin();
  auto fluxes = history_flux_expressions_.begin();
  auto initialized = manager.initialized.begin();
  auto pending = manager.store_pending.begin();
  auto fills = manager.fill_count.begin();
  auto depths = manager.depth.begin();
  for (PreparedHistoryMutationSlot& slot : prepared_history_mutation_slots_) {
    if (histories == manager.histories.end() || levels == history_levels_.end() ||
        clocks == manager.clock_identity.end() || dts == manager.slot_dt.end() ||
        fluxes == history_flux_expressions_.end() || initialized == manager.initialized.end() ||
        pending == manager.store_pending.end() || fills == manager.fill_count.end() ||
        depths == manager.depth.end() || slot.key != histories->first || slot.key != levels->first ||
        slot.key != clocks->first || slot.key != dts->first || slot.key != fluxes->first ||
        slot.key != initialized->first || slot.key != pending->first || slot.key != fills->first ||
        slot.key != depths->first || slot.level != levels->second ||
        slot.clock_identity != clocks->second || histories->second.size() != dts->second.size() ||
        histories->second.size() != fluxes->second.size() ||
        histories->second.size() != slot.rollback_ring.size())
      std::terminate();
    slot.live_ring = std::addressof(histories->second);
    slot.live_dts = std::addressof(dts->second);
    slot.live_expressions = std::addressof(fluxes->second);
    slot.live_initialized = std::addressof(initialized->second);
    slot.live_store_pending = std::addressof(pending->second);
    slot.live_fill_count = std::addressof(fills->second);
    slot.live_depth = std::addressof(depths->second);
    slot.live_clock_identity = std::addressof(clocks->second);
    slot.live_pending_remap = nullptr;
    for (auto remap = pending_history_remaps_.begin(); remap != pending_history_remaps_.end();
         ++remap) {
      if (remap->first == slot.key) {
        slot.live_pending_remap = std::addressof(remap->second);
        break;
      }
      if (slot.key < remap->first)
        break;
    }
    ++histories;
    ++levels;
    ++clocks;
    ++dts;
    ++fluxes;
    ++initialized;
    ++pending;
    ++fills;
    ++depths;
  }
  if (histories != manager.histories.end() || levels != history_levels_.end() ||
      clocks != manager.clock_identity.end() || dts != manager.slot_dt.end() ||
      fluxes != history_flux_expressions_.end() || initialized != manager.initialized.end() ||
      pending != manager.store_pending.end() || fills != manager.fill_count.end() ||
      depths != manager.depth.end())
    std::terminate();

  for (std::size_t index = 0; index < accepted_history_binding_mutation_slots_.size(); ++index) {
    const auto& binding = accepted_state_staging_.history_slot_bindings[index];
    if (binding.state_slot >= accepted_state_staging_.history_slot_pool.size())
      std::terminate();
    const std::size_t ordinal = accepted_history_binding_mutation_slots_[index];
    const int level = accepted_state_staging_.history_slot_pool[binding.state_slot].level;
    if (level < 0)
      std::terminate();
    if (static_cast<std::size_t>(level) >= runtime_->hierarchy().num_levels()) {
      if (ordinal != kInactiveHistoryMutationOrdinal_)
        std::terminate();
      continue;
    }
    if (ordinal >= prepared_history_mutation_slots_.size() ||
        prepared_history_mutation_slots_[ordinal].key != binding.key ||
        binding.source_slot >= prepared_history_mutation_slots_[ordinal].rollback_ring.size())
      std::terminate();
  }
  for (std::size_t index = 0; index < accepted_pending_history_ordinal_sources_.size(); ++index) {
    const std::string& key = accepted_state_staging_.pending_history_keys[index];
    const AmrProgramPendingHistoryRemap* source = nullptr;
    for (auto remap = pending_history_remaps_.begin(); remap != pending_history_remaps_.end();
         ++remap) {
      if (remap->first == key) {
        source = std::addressof(remap->second);
        break;
      }
      if (key < remap->first)
        break;
    }
    accepted_pending_history_ordinal_sources_[index] = source;
  }
  accepted_history_ordinal_owner_ = std::addressof(manager);
  accepted_history_ordinal_epoch_ = resource_epoch_;
  accepted_history_ordinal_generation_ = resource_generation_;
  prepared_history_mutation_epoch_ = resource_epoch_;
  prepared_history_mutation_generation_ = resource_generation_;
}

auto& prepared_history_mutation_slot_(std::string_view name) const {
  if (prepared_history_mutation_epoch_ != resource_epoch_ ||
      prepared_history_mutation_generation_ != resource_generation_)
    throw std::logic_error("AMR Program history mutation workspace is stale after topology change");
  PreparedHistoryMutationSlot* result = nullptr;
  for (PreparedHistoryMutationSlot& slot : prepared_history_mutation_slots_) {
    if (slot.level != active_level_ || slot.name != name)
      continue;
    if (result != nullptr)
      throw std::logic_error("AMR Program history mutation has an ambiguous finite slot");
    result = &slot;
  }
  if (result == nullptr)
    throw std::out_of_range("AMR Program history is not registered on the active level");
  if (result->live_ring == nullptr || result->live_dts == nullptr ||
      result->live_expressions == nullptr || result->live_initialized == nullptr ||
      result->live_store_pending == nullptr || result->live_fill_count == nullptr ||
      result->live_depth == nullptr || result->live_clock_identity == nullptr)
    throw std::logic_error("AMR Program history mutation lost its cold-bound live ordinal");
  return *result;
}

static void append_history_mutation_u64_(std::string& bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<char>((value >> shift) & std::uint64_t{0xff}));
}

static void append_history_mutation_bytes_(std::string& bytes, std::string_view value) {
  append_history_mutation_u64_(bytes, static_cast<std::uint64_t>(value.size()));
  bytes.append(value.data(), value.size());
}

std::string_view prepare_history_store_contract_(auto& slot, bool initialized) const {
  constexpr std::string_view tag = "pops.amr.history-store.v2";
  const std::size_t required = tag.size() + slot.key.size() + 6 * sizeof(std::uint64_t);
  if (slot.store_contract.capacity() < required)
    throw std::length_error("AMR Program history-store contract was not cold-primed");
  slot.store_contract.clear();
  slot.store_contract.append(tag);
  append_history_mutation_bytes_(slot.store_contract, slot.key);
  append_history_mutation_u64_(slot.store_contract, static_cast<std::uint64_t>(active_level_));
  append_history_mutation_u64_(slot.store_contract, std::bit_cast<std::uint64_t>(current_dt_));
  append_history_mutation_u64_(slot.store_contract,
                               static_cast<std::uint64_t>(slot.rollback_ring.size()));
  append_history_mutation_u64_(slot.store_contract, initialized ? 1U : 0U);
  return slot.store_contract;
}

std::string_view prepare_history_rotate_contract_(
    std::optional<std::string_view> clock_identity) const {
  constexpr std::string_view tag = "pops.amr.history-rotate.v2";
  std::string& contract = prepared_history_rotation_contract_;
  std::size_t required =
      tag.size() + 3 * sizeof(std::uint64_t) + (clock_identity ? clock_identity->size() : 0U);
  for (const PreparedHistoryMutationSlot& slot : prepared_history_mutation_slots_) {
    if (slot.level != active_level_ || (clock_identity && slot.clock_identity != *clock_identity))
      continue;
    required += slot.key.size() + slot.clock_identity.size() + 3 * sizeof(std::uint64_t);
  }
  if (contract.capacity() < required)
    throw std::length_error("AMR Program history-rotate contract was not cold-primed");
  contract.clear();
  contract.append(tag);
  append_history_mutation_u64_(contract, static_cast<std::uint64_t>(active_level_));
  append_history_mutation_bytes_(contract, clock_identity.value_or(std::string_view{}));
  std::size_t selected = 0;
  for (const PreparedHistoryMutationSlot& slot : prepared_history_mutation_slots_) {
    if (slot.level != active_level_ || (clock_identity && slot.clock_identity != *clock_identity))
      continue;
    if (slot.live_store_pending == nullptr)
      throw std::logic_error("AMR Program history rotation lost a bind-sealed pending slot");
    append_history_mutation_bytes_(contract, slot.key);
    append_history_mutation_bytes_(contract, slot.clock_identity);
    append_history_mutation_u64_(contract, *slot.live_store_pending ? 1U : 0U);
    ++selected;
  }
  append_history_mutation_u64_(contract, static_cast<std::uint64_t>(selected));
  return contract;
}

void store_history_(const std::string& name, const field_type& value) const {
  auto& manager = runtime_state().hist_;
  if (accepted_history_ordinal_owner_ != std::addressof(manager) ||
      accepted_history_ordinal_epoch_ != resource_epoch_ ||
      accepted_history_ordinal_generation_ != resource_generation_)
    throw std::logic_error("AMR Program history mutation ordinals were not rebound cold");
  PreparedHistoryMutationSlot* candidate = prepare_history_mutation_collectively_(
      [&]() {
        PreparedHistoryMutationSlot& staged = prepared_history_mutation_slot_(name);
        auto& ring = *staged.live_ring;
        auto& dts = *staged.live_dts;
        auto& expressions = *staged.live_expressions;
        const bool initialized = *staged.live_initialized;
        if (ring.empty())
          throw std::out_of_range("AMR Program history is not registered on the active level");
        require_same_field_contract_(ring.front(), value, "AMR Program history store");
        if (!std::isfinite(current_dt_) || !(current_dt_ > 0.0))
          throw std::logic_error("AMR Program history store has no positive exact interval");
        if (expressions.size() != ring.size() || staged.rollback_ring.size() != ring.size() ||
            staged.dts.size() != ring.size() || staged.expressions.size() != ring.size() ||
            dts.size() != ring.size())
          throw std::logic_error("AMR Program history flux provenance differs from its ring depth");
        for (std::size_t slot = 0; slot < ring.size(); ++slot) {
          require_same_field_contract_(staged.rollback_ring[slot], ring[slot],
                                       "AMR Program history rollback image");
          copy_valid_(ring[slot], staged.rollback_ring[slot]);
          staged.dts[slot] = dts[slot];
        }
        if (!static_flux_tables_.bound)
          throw std::logic_error(
              "AMR Program history store requires bind-sealed static flux provenance");
        require_static_history_flux_provenance_(ring, expressions);
        (void)prepare_history_store_contract_(staged, initialized);
        return &staged;
      },
      [](const PreparedHistoryMutationSlot* staged) -> std::string_view {
        return staged->store_contract;
      },
      "AMR Program history store");

  auto& ring = *candidate->live_ring;
  auto& expressions = *candidate->live_expressions;
  auto& dts = *candidate->live_dts;
  const bool initialized = *candidate->live_initialized;
  // Public history() references name individual ring elements.  Keep those elements at stable
  // addresses.  A MultiFab valid-copy has no allocation path after its exact field contract has
  // been prepared above; nevertheless, converge a device/local exception before metadata moves.
  std::size_t published = 0;
  std::exception_ptr copy_error;
  try {
    const std::size_t stores = initialized ? std::size_t{1} : ring.size();
    for (; published < stores; ++published)
      copy_valid_(value, ring[published]);
  } catch (...) {
    copy_error = std::current_exception();
  }
  const ExecutionLane& lane = prepared_execution_lane();
  if (all_reduce_max(copy_error ? 1L : 0L, lane) != 0) {
    std::exception_ptr rollback_error;
    try {
      // A throwing valid-copy may have written part of the current slot before the loop counter
      // advances.  Restore every stable element, not merely completed slots, on every rank.
      for (std::size_t slot = 0; slot < candidate->rollback_ring.size(); ++slot)
        copy_valid_(candidate->rollback_ring[slot], ring[slot]);
    } catch (...) {
      rollback_error = std::current_exception();
    }
    (void)all_reduce_max(rollback_error ? 1L : 0L, lane);
    if (lane.size() == 1 && copy_error)
      std::rethrow_exception(copy_error);
    throw std::runtime_error("AMR Program history store valid-copy failed collectively");
  }
  dts.front() = static_cast<Real>(current_dt_);
  if (!initialized)
    for (std::size_t slot = 1; slot < dts.size(); ++slot)
      dts[slot] = static_cast<Real>(current_dt_);
  store_static_history_flux_provenance_(ring, expressions, !initialized);
  *candidate->live_initialized = true;
  *candidate->live_store_pending = true;
}

void rotate_histories_(std::optional<std::string_view> clock_identity) const {
  auto& manager = runtime_state().hist_;
  if (accepted_history_ordinal_owner_ != std::addressof(manager) ||
      accepted_history_ordinal_epoch_ != resource_epoch_ ||
      accepted_history_ordinal_generation_ != resource_generation_)
    throw std::logic_error("AMR Program history mutation ordinals were not rebound cold");
  (void)prepare_history_mutation_collectively_(
      [&]() {
        for (const PreparedHistoryMutationSlot& slot : prepared_history_mutation_slots_) {
          if (slot.level != active_level_ ||
              (clock_identity && slot.clock_identity != *clock_identity))
            continue;
          if (slot.live_ring == nullptr || slot.live_dts == nullptr ||
              slot.live_expressions == nullptr || slot.live_clock_identity == nullptr ||
              slot.live_fill_count == nullptr || slot.live_store_pending == nullptr ||
              slot.live_ring->size() != slot.live_dts->size() ||
              slot.live_ring->size() != slot.live_expressions->size() ||
              slot.live_ring->size() != slot.rollback_ring.size() ||
              *slot.live_depth != static_cast<int>(slot.live_ring->size()) ||
              *slot.live_clock_identity != slot.clock_identity)
            throw std::runtime_error("AMR Program history registry is incomplete");
        }
        return prepare_history_rotate_contract_(clock_identity);
      },
      [](std::string_view staged) { return staged; }, "AMR Program history rotation");
  for (const PreparedHistoryMutationSlot& slot : prepared_history_mutation_slots_) {
    if (slot.level != active_level_ || (clock_identity && slot.clock_identity != *clock_identity))
      continue;
    auto& ring = *slot.live_ring;
    for (std::size_t slot = ring.size(); slot-- > 1;)
      std::swap(ring[slot], ring[slot - 1]);
    auto& dts = *slot.live_dts;
    for (std::size_t slot = dts.size(); slot-- > 1;)
      std::swap(dts[slot], dts[slot - 1]);
    auto& expressions = *slot.live_expressions;
    for (std::size_t slot = expressions.size(); slot-- > 1;)
      std::swap(expressions[slot], expressions[slot - 1]);
    if (*slot.live_store_pending) {
      *slot.live_fill_count = std::min(static_cast<int>(ring.size()), *slot.live_fill_count + 1);
      *slot.live_store_pending = false;
      if (slot.live_pending_remap != nullptr)
        slot.live_pending_remap->consumed = true;
      // Keep the bind-primed scratch node resident.  Its logical activation is carried solely by
      // the pending marker; erasing the node here would change the next hot transaction snapshot
      // shape and force an allocation on a later deferred retry.
    }
  }
}

std::optional<ScheduleCoordinate> schedule_coordinate_(ScheduleDomainKind kind,
                                                       const std::string& clock,
                                                       const std::string& stage_identity,
                                                       int level) const {
  return clock_schedule_.coordinate(kind, clock, stage_identity, level, active_level_,
                                    static_cast<std::int64_t>(macro_step()));
}

OperatorFingerprint operator_topology_(const field_type& prototype) const {
  require_same_layout_(prototype, state(0), "AMR Program operator topology");
  OperatorFingerprint fingerprint =
      ::pops::detail::layout_fingerprint(prototype, program_resource_vector_distribution());
  ::pops::detail::fingerprint_geometry(fingerprint, geometry());
  ::pops::detail::fingerprint_mix(fingerprint, runtime_->spatial_contract());
  ::pops::detail::fingerprint_mix(fingerprint, runtime_->topology_epoch());
  ::pops::detail::fingerprint_mix(fingerprint, runtime_->materialization_generation());
  ::pops::detail::fingerprint_mix(fingerprint, static_cast<std::uint64_t>(active_level_));
  return fingerprint;
}

OperatorEvaluationSnapshot current_operator_snapshot_(OperatorFingerprint authority,
                                                      OperatorFingerprint topology,
                                                      OperatorFingerprint resources,
                                                      std::uint64_t revision) const {
  refresh_resources_();
  const std::uint64_t maximum_generation =
      std::max(runtime_->topology_epoch(), runtime_->materialization_generation());
  if (maximum_generation == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("AMR Program topology revision exhausted uint64_t");
  const double evaluation_time =
      static_cast<double>(physical_time()) + stage_time_.value() * current_dt_;
  return {authority,
          revision,
          static_cast<std::int64_t>(macro_step()),
          stage_time_.numerator,
          stage_time_.denominator,
          std::bit_cast<std::uint64_t>(current_dt_),
          std::bit_cast<std::uint64_t>(evaluation_time),
          maximum_generation + 1,
          topology,
          resources};
}

static void require_same_layout_(const field_type& left, const field_type& right,
                                 std::string_view operation) {
  if (left.layout() != right.layout() || left.distribution() != right.distribution() ||
      left.local_rank() != right.local_rank() || left.local_size() != right.local_size())
    throw std::invalid_argument(std::string(operation) + " fields have different exact layouts");
}
static void require_same_field_contract_(const field_type& left, const field_type& right,
                                         std::string_view operation) {
  require_same_layout_(left, right, operation);
  if (left.ncomp() != right.ncomp())
    throw std::invalid_argument(std::string(operation) + " fields have different components");
}
static void require_scalar_stencil_(const field_type& output, const field_type& input,
                                    int output_components, std::string_view operation) {
  require_same_layout_(output, input, operation);
  if (input.ncomp() != 1 || output.ncomp() != output_components)
    throw std::invalid_argument(std::string(operation) + " has an invalid component contract");
  for (int axis = 0; axis < Dim; ++axis)
    if (input.ghosts()[axis] < 1)
      throw std::invalid_argument(std::string(operation) + " requires one ghost per axis");
}
void require_boundary_point_(const runtime::multiblock::BoundaryEvaluationPoint& point,
                             std::string_view operation) const {
  if (point.level != active_level_ || point.clock.empty() || point.stage < 0 ||
      point.stage_fraction.denominator <= 0)
    throw std::invalid_argument(std::string(operation) + " has a foreign evaluation point");
}

void require_current_boundary_point_exact_(
    const runtime::multiblock::BoundaryEvaluationPoint& point, std::string_view operation) const {
  require_rate_identity_(point.stage);
  const double physical_time = current_interval_start_time_ + stage_time_.value() * current_dt_;
  if (primary_clock_.empty() || !std::isfinite(current_dt_) || !(current_dt_ > 0.0) ||
      point.clock != primary_clock_ ||
      point.tick != static_cast<std::int64_t>(facade_->program_macro_step_()) ||
      point.level != active_level_ || point.substep != logical_substep_ ||
      point.stage_fraction != stage_time_ || point.dt != current_dt_ ||
      point.physical_time != physical_time || !point.graph_identity.empty() ||
      !point.rate_identity.empty() || !point.application_identity.empty())
    throw std::invalid_argument(std::string(operation) +
                                " has a stale or foreign exact evaluation point");
}

static std::string history_key_(const std::string& name, int level) {
  if (name.empty() || level < 0)
    throw std::invalid_argument("AMR Program history key is invalid");
  return "pops.amr.level-history.v1/" + std::to_string(level) + "/" + std::to_string(name.size()) +
         ":" + name;
}

void require_history_owner_(int program_owner) const {
  if (program_owner < 0 || sys_block(program_owner) < 0)
    throw std::invalid_argument("AMR Program history has a foreign block owner");
}

field_type& deferred_history_lag_(const std::string& key,
                                  const std::vector<field_type>& ring) const {
  auto& manager = runtime_state().hist_;
  struct DeferredCandidate {
    AmrProgramPendingHistoryRemap marker;
    std::map<std::string, field_type> scratches;
    FluxExpressionRegistry expressions;
    std::uint64_t next_identity = 0;
    std::string exact_contract;
  };
  DeferredCandidate candidate = prepare_history_mutation_collectively_(
      [&]() {
        const auto pending = pending_history_remaps_.find(key);
        if (pending == pending_history_remaps_.end())
          throw std::logic_error("AMR Program deferred history lag has no pending remap");
        const auto& marker = pending->second;
        if (marker.consumed || ring.size() != 2 || marker.child_level != active_level_ ||
            marker.published_topology_epoch != runtime_->topology_epoch() ||
            marker.published_materialization_generation != runtime_->materialization_generation() ||
            !manager.store_pending.at(key) || manager.slot_dt.at(key).size() != 2 ||
            static_cast<double>(manager.slot_dt.at(key)[0]) != current_dt_ ||
            static_cast<double>(manager.slot_dt.at(key)[1]) != marker.source_dt ||
            marker.temporal_denominator != 1 ||
            (marker.temporal_numerator != 1 && marker.temporal_numerator != 2) ||
            marker.target_dt != marker.source_dt / static_cast<double>(marker.temporal_numerator))
          throw std::runtime_error(
              "AMR Program deferred history lag lost its exact IntegralOnly authority");
        const auto provenance = history_flux_expressions_.find(key);
        if (provenance == history_flux_expressions_.end() ||
            provenance->second.size() != ring.size())
          throw std::logic_error(
              "AMR Program deferred history lag lacks its persistent ring provenance");
        DeferredCandidate staged{
            marker, deferred_history_lag_scratches_, {}, next_active_flux_basis_identity_, {}};
        auto [scratch, inserted] = staged.scratches.try_emplace(key, ring.front());
        (void)inserted;
        require_same_field_contract_(scratch->second, ring.front(),
                                     "AMR Program deferred history lag");
        if (marker.temporal_numerator == 1)
          copy_valid_(ring[1], scratch->second);
        else
          pops::lincomb(scratch->second, Real(0.5), ring[0], Real(0.5), ring[1]);

        // A map swap preserves every node address.  Rebind all existing deferred-scratch
        // registry entries to their staged nodes before the old map is replaced, otherwise a
        // repeated deferred read could leave a live expression pointing at a destroyed map.
        staged.expressions = active_flux_expressions_;
        for (const auto& [scratch_key, live_scratch] : deferred_history_lag_scratches_) {
          const auto live_expression = staged.expressions.find(&live_scratch);
          if (live_expression == staged.expressions.end())
            continue;
          const auto staged_scratch = staged.scratches.find(scratch_key);
          if (staged_scratch == staged.scratches.end())
            throw std::logic_error("AMR Program deferred history scratch staging lost a live key");
          FluxExpression expression = std::move(live_expression->second);
          staged.expressions.erase(live_expression);
          staged.expressions.emplace(&staged_scratch->second, std::move(expression));
        }

        FluxExpression current_expression;
        FluxExpression lag_expression;
        FluxExpression detached;
        if (marker.temporal_numerator == 1) {
          auto [rehydrated_lag, next_identity] = rehydrated_history_flux_expression_(
              key, 1, active_flux_expressions_, staged.next_identity);
          staged.next_identity = next_identity;
          lag_expression = std::move(rehydrated_lag);
          if (!lag_expression.empty()) {
            staged.expressions[&ring[1]] = lag_expression;
            detached = lag_expression;
          }
        } else {
          current_expression = provenance->second[0];
          auto [rehydrated_lag, next_identity] = rehydrated_history_flux_expression_(
              key, 1, active_flux_expressions_, staged.next_identity);
          lag_expression = std::move(rehydrated_lag);
          if (current_expression.empty() != lag_expression.empty())
            throw std::logic_error("AMR Program deferred history lag has no lag provenance");
          staged.next_identity = next_identity;
          if (!current_expression.empty()) {
            staged.expressions[&ring[1]] = lag_expression;
            detached = scaled_flux_expression_(current_expression,
                                               ExactPolynomial{{0, ::pops::amr::Rational(1, 2)}});
            add_flux_expression_(
                detached, scaled_flux_expression_(
                              lag_expression, ExactPolynomial{{0, ::pops::amr::Rational(1, 2)}}));
            require_flux_expression_budget_(detached);
          }
        }
        const std::size_t detached_term_count = detached.size();
        if (!detached.empty())
          staged.expressions[&scratch->second] = std::move(detached);
        ExactContractBuilder contract;
        contract.text("pops.amr.deferred-history-lag.v1")
            .bytes(key)
            .scalar(marker.parent_level)
            .scalar(marker.child_level)
            .scalar(marker.prior_topology_epoch)
            .scalar(marker.prior_materialization_generation)
            .scalar(marker.published_topology_epoch)
            .scalar(marker.published_materialization_generation)
            .scalar(marker.accepted_macro_step)
            .scalar(marker.temporal_numerator)
            .scalar(marker.temporal_denominator)
            .scalar(std::bit_cast<std::uint64_t>(marker.source_dt))
            .scalar(std::bit_cast<std::uint64_t>(marker.target_dt))
            .scalar(staged.next_identity)
            .scalar(static_cast<std::uint64_t>(current_expression.size()))
            .scalar(static_cast<std::uint64_t>(lag_expression.size()))
            .scalar(static_cast<std::uint64_t>(detached_term_count))
            .scalar(static_cast<std::uint64_t>(staged.scratches.size()))
            .scalar(static_cast<std::uint64_t>(staged.expressions.size()));
        staged.exact_contract = std::move(contract).release();
        return staged;
      },
      [](const DeferredCandidate& staged) -> const std::string& { return staged.exact_contract; },
      "AMR Program deferred history lag");
  static_assert(std::is_nothrow_swappable_v<decltype(deferred_history_lag_scratches_)>);
  static_assert(std::is_nothrow_swappable_v<FluxExpressionRegistry>);
  deferred_history_lag_scratches_.swap(candidate.scratches);
  active_flux_expressions_.swap(candidate.expressions);
  next_active_flux_basis_identity_ = candidate.next_identity;
  // Profiling is observational: after the no-throw publication boundary it must not turn a
  // committed deferred scratch/provenance pair into a reported failed history read.
  try {
    count_kernel_();
  } catch (...) {
  }
  return deferred_history_lag_scratches_.at(key);
}

field_type& history_slot_(const std::string& name, int lag, bool zero_start, int components) const {
  const std::string key = history_key_(name, active_level_);
  auto& manager = runtime_state().hist_;
  const auto found = manager.histories.find(key);
  if (found == manager.histories.end() || lag < 0 || lag >= manager.depth.at(key))
    throw std::out_of_range("AMR Program history slot is absent");
  field_type& result = found->second[static_cast<std::size_t>(lag)];
  if (components >= 0 && result.ncomp() != components)
    throw std::invalid_argument("AMR Program history component contract differs");
  if (!manager.initialized.at(key)) {
    if (!zero_start)
      throw std::runtime_error("AMR Program history has not been initialized");
    result.set_val(Real(0));
  }
  if (lag == 1) {
    const auto pending = pending_history_remaps_.find(key);
    if (pending != pending_history_remaps_.end() && !pending->second.consumed)
      return deferred_history_lag_(key, found->second);
  }
  rehydrate_history_flux_expression_(key, lag, result);
  return result;
}
