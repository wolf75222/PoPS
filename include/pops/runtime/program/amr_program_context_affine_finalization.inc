// Included inside AmrProgramContext. All algebraic coefficients come from the
// authored Program; this service owns snapshots, measure transport and publication.
private:
struct AffineMacroFinalization {
  double start = 0.0;
  double requested = 0.0;
  std::uint64_t topology_epoch = 0, materialization_generation = 0;
  std::vector<std::vector<field_type>> entry;
  std::vector<std::vector<field_type>> provisional;
  HistoryManager<Dim> entry_histories;
  std::map<std::string, std::vector<FluxExpression>> entry_history_flux;
  AffineMeasureTower measures;
  bool ready = false;
  bool accepted = false;
};
mutable std::optional<AffineMacroFinalization> affine_finalization_;

std::optional<int> affine_field_owner_(const field_type& field) const {
  if (!affine_finalization_)
    return std::nullopt;
  const auto& frame = *affine_finalization_;
  std::optional<int> owner;
  for (const auto* tower : {&frame.entry, &frame.provisional})
    for (std::size_t block = 0; block < tower->size(); ++block)
      for (std::size_t level = 0; level < tower->at(block).size(); ++level)
        if (&field == &tower->at(block).at(level)) {
          if (!frame.ready || active_level_ != static_cast<int>(level) ||
              frame.topology_epoch != runtime_->topology_epoch() ||
              frame.materialization_generation != runtime_->materialization_generation())
            throw std::invalid_argument("affine snapshot requires its exact prepared support");
          if (owner && *owner != static_cast<int>(block))
            throw std::logic_error("affine snapshot aliases multiple runtime owners");
          owner = static_cast<int>(block);
        }
  return owner;
}

void require_affine_finalization_(const char* operation) const {
  if (!affine_finalization_ || !affine_finalization_->ready || active_level_ != 0)
    throw std::logic_error(std::string(operation) + " requires the post-reflux parent scope");
}

std::vector<std::vector<field_type>> copy_affine_tower_() const {
  const auto& hierarchy = facade_->prepared_amr_multiblock_hierarchy_();
  std::vector<std::vector<field_type>> fields(hierarchy.block_count());
  for (std::size_t block = 0; block < fields.size(); ++block) {
    fields[block].reserve(hierarchy.level_count());
    for (std::size_t level = 0; level < hierarchy.level_count(); ++level)
      fields[block].emplace_back(
          facade_->prepared_amr_block_state(static_cast<int>(block), static_cast<int>(level)));
  }
  return fields;
}

public:
void begin_affine_finalization(double requested, std::initializer_list<int> program_blocks) const {
  require_facade_execution_();
  refresh_resources_();
  std::exception_ptr error;
  try {
    // Failed parent attempts retain no usable finalization authority. Replacing
    // this ephemeral snapshot never alters accepted State or persistent history.
    affine_finalization_.reset();
    const auto& hierarchy = facade_->prepared_amr_multiblock_hierarchy_();
    if (!std::isfinite(requested) || !(requested > 0) ||
        !std::isfinite(facade_->time() + requested) ||
        !(facade_->time() + requested > facade_->time()))
      throw std::invalid_argument("affine finalization requires a finite requested macro interval");
    std::set<int> owners;
    for (int block : program_blocks)
      owners.insert(sys_block(block));
    if (owners.size() != hierarchy.block_count())
      throw std::invalid_argument(
          "affine finalization must own the complete participating hierarchy");
    if (!pending_history_remaps_.empty())
      throw std::logic_error("affine finalization requires resolved incoming history remaps");
    for (const auto& [key, level] : history_levels_) {
      const auto decoded = decode_history_key_(key);
      if (!decoded || decoded->first != level ||
          !affine_history_transfers_.contains(decoded->second))
        throw std::invalid_argument(
            "affine finalization lacks an explicit retained-history transfer");
    }
    AffineMacroFinalization frame;
    frame.start = facade_->time();
    frame.requested = requested;
    frame.topology_epoch = runtime_->topology_epoch();
    frame.materialization_generation = runtime_->materialization_generation();
    frame.entry = copy_affine_tower_();
    frame.entry_histories = runtime_state().hist_;
    frame.entry_history_flux = history_flux_expressions_;
    frame.measures = capture_affine_measures_();
    affine_finalization_.emplace(std::move(frame));
  } catch (...) {
    error = std::current_exception();
  }
  converge_owner_reduction_(error, prepared_execution_lane(), "affine macro entry capture");
}

void prepare_affine_finalization() const {
  std::exception_ptr error;
  try {
    if (!affine_finalization_ || affine_finalization_->ready || !multiblock_subcycling_ ||
        !multiblock_subcycling_has_accepted_step_ ||
        multiblock_subcycling_->has_attempt_candidates())
      throw std::logic_error("affine finalization requires one complete refluxed provisional step");
    validate_affine_measures_();
    affine_finalization_->provisional = copy_affine_tower_();
    affine_finalization_->ready = true;
  } catch (...) {
    error = std::current_exception();
  }
  converge_owner_reduction_(error, prepared_execution_lane(), "affine provisional capture");
}

field_type& attempt_start(int program_block) const {
  require_affine_finalization_("attempt_start");
  return affine_finalization_->entry.at(static_cast<std::size_t>(sys_block(program_block)))
      .at(static_cast<std::size_t>(active_level_));
}

Real composite_pair(int program_block, bool left_entry, bool right_entry,
                    std::initializer_list<double> component_weights) const {
  const auto& lane = prepared_execution_lane();
  FiniteCompensatedSum accumulated;
  std::exception_ptr error;
  try {
    require_affine_finalization_("composite_pair");
    const auto& frame = *affine_finalization_;
    const int owner = sys_block(program_block);
    const auto& left = (left_entry ? frame.entry : frame.provisional).at(owner);
    const auto& right = (right_entry ? frame.entry : frame.provisional).at(owner);
    const std::vector<double> weights(component_weights);
    for (std::size_t level = 0; level < left.size(); ++level) {
      const auto& a = left[level];
      const auto& b = right.at(level);
      require_same_field_contract_(a, b, "composite_pair");
      if (weights.size() != static_cast<std::size_t>(a.ncomp()))
        throw std::invalid_argument("composite_pair component count differs");
      const auto& measure = frame.measures.at(owner).at(level);
      const auto* active = measure.active ? &*measure.active : nullptr;
      const auto* fraction = measure.fraction ? &*measure.fraction : nullptr;
      const auto& coverage = measure.coverage;
      mf_arith_detail::validate_measure(a, {active, nullptr}, "composite_pair");
      mf_arith_detail::validate_measure(a, {fraction, nullptr}, "composite_pair volume fraction");
      mf_arith_detail::validate_measure(a, {&coverage, nullptr}, "composite_pair");
      const auto& geometry = measure.geometry;
      double volume = 1.;
      for (int axis = 0; axis < Dim; ++axis)
        volume *= geometry.spacing(axis);
      if (!(volume > 0.) || !std::isfinite(volume))
        throw std::invalid_argument(
            "composite_pair requires finite positive physical cell volumes");
      for (int component = 0; component < a.ncomp(); ++component) {
        const double coefficient = weights[static_cast<std::size_t>(component)] * volume;
        if (!std::isfinite(coefficient))
          throw std::invalid_argument("composite_pair has a nonfinite component weight");
        if (coefficient == 0.)
          continue;
        for (std::size_t local = 0; local < a.local_size(); ++local) {
          const auto sum = pops::for_each_cell_reduce_finite_sum(
              a.box(local),
              detail::CompositeIntegralKernel<Dim, detail::CompositePairIntegrand<Dim>>{
                  {a.fab(local).view(), b.fab(local).view(), component},
                  active ? active->fab(local).view() : FieldView<const Real, Dim>{},
                  coverage.fab(local).view(),
                  fraction ? fraction->fab(local).view() : FieldView<const Real, Dim>{},
                  Real{1},
                  active != nullptr,
                  fraction != nullptr});
          if (!sum.finite())
            throw std::overflow_error("composite_pair has nonfinite active values");
          if (!a.distribution().replicated() || lane.rank() == 0) {
            accumulated.add(Real(coefficient) * sum.high);
            accumulated.add(Real(coefficient) * sum.low);
          }
        }
      }
    }
    if (!accumulated.finite())
      throw std::overflow_error("composite_pair has nonfinite accumulation");
  } catch (...) {
    error = std::current_exception();
  }
  converge_owner_reduction_(error, lane, "physical composite pairing");
  return pops::collective_finite_compensated_sum(accumulated, lane.communicator());
}

template <class Integrand>
Real composite_integral(int program_block, Integrand integrand) const {
  const auto& lane = prepared_execution_lane();
  FiniteCompensatedSum accumulated;
  std::exception_ptr error;
  try {
    require_affine_finalization_("composite_integral");
    const auto& frame = *affine_finalization_;
    const int owner = sys_block(program_block);
    const auto& entry = frame.entry.at(owner);
    const auto& current = frame.provisional.at(owner);
    for (std::size_t level = 0; level < entry.size(); ++level) {
      const auto& a = entry[level];
      const auto& b = current.at(level);
      require_same_field_contract_(a, b, "composite_integral");
      const auto& measure = frame.measures.at(owner).at(level);
      const auto* active = measure.active ? &*measure.active : nullptr;
      const auto* fraction = measure.fraction ? &*measure.fraction : nullptr;
      const auto& coverage = measure.coverage;
      mf_arith_detail::validate_measure(a, {active, nullptr}, "composite_integral");
      mf_arith_detail::validate_measure(a, {fraction, nullptr},
                                        "composite_integral volume fraction");
      mf_arith_detail::validate_measure(a, {&coverage, nullptr}, "composite_integral");
      const auto& geometry = measure.geometry;
      double volume = 1.;
      for (int axis = 0; axis < Dim; ++axis)
        volume *= geometry.spacing(axis);
      if (!(volume > 0.) || !std::isfinite(volume))
        throw std::invalid_argument("composite_integral requires finite positive cell volumes");
      for (std::size_t local = 0; local < a.local_size(); ++local) {
        integrand.entry = a.fab(local).view();
        integrand.current = b.fab(local).view();
        const auto sum = pops::for_each_cell_reduce_finite_sum(
            a.box(local),
            detail::CompositeIntegralKernel<Dim, Integrand>{
                integrand, active ? active->fab(local).view() : FieldView<const Real, Dim>{},
                coverage.fab(local).view(),
                fraction ? fraction->fab(local).view() : FieldView<const Real, Dim>{}, Real(volume),
                active != nullptr, fraction != nullptr});
        if (!sum.finite())
          throw std::overflow_error("composite_integral has nonfinite active values");
        if (!a.distribution().replicated() || lane.rank() == 0) {
          accumulated.add(sum.high);
          accumulated.add(sum.low);
        }
      }
    }
    if (!accumulated.finite())
      throw std::overflow_error("composite_integral has nonfinite accumulation");
  } catch (...) {
    error = std::current_exception();
  }
  converge_owner_reduction_(error, lane, "authored physical composite integral");
  return pops::collective_finite_compensated_sum(accumulated, lane.communicator());
}

void accept_increment(double scale, double duration) const {
  const auto& lane = prepared_execution_lane();
  std::exception_ptr error;
  std::string contract;
  ::pops::amr::reflux::AffineIntervalPublication publication;
  std::unique_ptr<interface_flux_ledger_type> prepared_interface;
  AffineHistories prepared_histories;
  try {
    require_affine_finalization_("accept_increment");
    auto& frame = *affine_finalization_;
    if (frame.accepted)
      throw std::logic_error("affine macro increment was already accepted");
    publication = {frame.start, frame.requested, duration, scale};
    ::pops::amr::reflux::validate_affine_interval_publication(publication);
    validate_affine_measures_();
    if (!interface_flux_ledger_)
      throw std::logic_error("affine finalization requires its interface ledger authority");
    prepared_interface = std::make_unique<interface_flux_ledger_type>(*interface_flux_ledger_);
    prepared_interface->finalize_affine_publication(publication);
    prepared_histories = prepare_affine_histories_(publication);
    ExactContractBuilder bytes;
    bytes.text("pops.affine-macro-publication.v1")
        .scalar(frame.start)
        .scalar(frame.requested)
        .scalar(duration)
        .scalar(scale);
    for (const auto& [key, history] : prepared_histories) {
      bytes.text(key).scalar(history.fill_count).scalar(history.samples.size());
      for (const auto& sample : history.samples)
        sample.append_contract(bytes);
    }
    contract = std::move(bytes).release();
  } catch (...) {
    error = std::current_exception();
  }
  converge_owner_reduction_(error, lane, "affine macro publication contract");
  const std::array<std::pair<std::string_view, std::string_view>, 1> contract_parts{
      {{"affine-macro-publication", contract}}};
  if (!all_ranks_agree_exact_ordered_byte_pairs(contract_parts, lane))
    throw std::invalid_argument("affine macro publication differs between ranks");
  auto& frame = *affine_finalization_;
  multiblock_subcycling_->finalize_affine_macro(
      publication, frame.entry, frame.provisional,
      [&](std::size_t block, std::size_t level, const field_type& candidate) {
        facade_->validate_prepared_amr_state_publication_candidate(
            static_cast<int>(block), static_cast<int>(level), candidate);
      },
      [&](auto& candidates, const auto& clocks, const auto& histories) {
        prepare_accepted_publication_halos_(duration, candidates, clocks, histories);
      });
  // Prepared before any accepted State mutation. The outer AcceptedSnapshot
  // restores both this ledger and the hierarchy if a later guard rejects.
  publish_affine_histories_(prepared_histories);
  interface_flux_ledger_.swap(prepared_interface);
  interface_flux_commit_guard_.reset();
  runtime_state().set_computed_duration(duration);
  record_scalar("pops.frontier.duration", static_cast<Real>(duration));
  frame.accepted = true;
}

void finish_affine_finalization() const {
  if (!affine_finalization_ || !affine_finalization_->accepted)
    throw std::logic_error("computed AMR Program did not accept its macro increment");
  affine_finalization_.reset();
}
