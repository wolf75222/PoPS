// Composite equations at a local stage consume a frozen macro-entry bundle.
// Resource-level selection never impersonates another level's live attempt.
private:
using TemporalFieldKey = std::pair<std::int64_t, int>;
struct TemporalFieldDeclaration {
  int owner = -1;
  std::string name;
  std::size_t slots = 0;
  std::string state, space, clock, interpolation, identity;
  HistoryTransferCallback callback;
};
struct TemporalFieldLevelImage {
  std::vector<field_type> fields;
  std::vector<TemporalSampleSource> sources;
  std::optional<field_type> evaluated;
  std::string evaluation_authority;
};
struct TemporalFieldImage {
  PreparedResourceAttempt attempt;
  std::uint64_t epoch = 0, generation = 0, macro_step = 0;
  double time = 0, duration = 0;
  std::vector<TemporalFieldLevelImage> levels;
};
mutable std::map<TemporalFieldKey, TemporalFieldDeclaration> temporal_field_declarations_;
mutable std::map<TemporalFieldKey, TemporalFieldImage> temporal_field_images_;
mutable bool temporal_field_declarations_sealed_ = false;

void prepare_temporal_field_images_(double duration) const {
  const auto& lane = prepared_execution_lane();
  if (all_reduce_max(temporal_field_declarations_.empty() ? 0L : 1L, lane) == 0) {
    temporal_field_declarations_sealed_ = true;
    return;
  }
  std::string authority;
  interaction_phase(lane, [&] {
    ExactContractBuilder exact;
    exact.text("pops.temporal-field-declarations.v1");
    for (const auto& [key, d] : temporal_field_declarations_)
      exact.scalar(key.first)
          .scalar(key.second)
          .scalar(d.owner)
          .text(d.name)
          .scalar(d.slots)
          .text(d.state)
          .text(d.space)
          .text(d.clock)
          .text(d.interpolation)
          .bytes(d.identity);
    authority = std::move(exact).release();
  });
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"temporal-field-declarations", authority}}, lane))
    throw std::invalid_argument("temporal field declarations differ across ranks");
  temporal_field_declarations_sealed_ = true;
  if (all_reduce_min(long(nlev()), lane) != all_reduce_max(long(nlev()), lane))
    throw std::invalid_argument("temporal field source level counts differ across ranks");
  std::map<TemporalFieldKey, TemporalFieldImage> staged;
  for (const auto& [key, d] : temporal_field_declarations_) {
    TemporalFieldImage image;
    std::string source_authority;
    interaction_phase(lane, [&] {
      if (!resource_attempt().visible())
        throw std::logic_error("temporal field capture requires a live macro attempt");
      image.attempt = resource_attempt();
      image.epoch = runtime_->topology_epoch();
      image.generation = runtime_->materialization_generation();
      image.macro_step = facade_->macro_step();
      image.time = facade_->time();
      image.duration = duration;
      const auto& history = runtime_state().hist_;
      ExactContractBuilder exact;
      exact.text("pops.temporal-field-macro-source.v1")
          .bytes(d.identity)
          .scalar(key.first)
          .scalar(key.second)
          .scalar(d.owner)
          .scalar(image.epoch)
          .scalar(image.generation)
          .scalar(image.macro_step)
          .scalar(image.attempt.ordinal())
          .scalar(image.time)
          .scalar(duration);
      for (int level = 0; level < nlev(); ++level) {
        const auto name = history_key_(d.name, level);
        if (pending_history_remaps_.contains(name) || history.store_pending.at(name) ||
            history.owner.at(name) != d.owner || history.state_identity.at(name) != d.state ||
            history.space_identity.at(name) != d.space ||
            history.clock_identity.at(name) != d.clock ||
            history.interpolation_identity.at(name) != d.interpolation ||
            history.histories.at(name).size() != d.slots)
          throw std::invalid_argument(
              "temporal field keeper has a foreign or pending source contract");
        TemporalFieldLevelImage level_image;
        const auto& accepted = facade_->prepared_amr_block_state(d.owner, level);
        ExactContractBuilder point;
        point.text("pops.temporal-field-native-macro-point.v1")
            .bytes(d.identity)
            .text(d.state)
            .text(d.space)
            .text(d.clock)
            .scalar(d.owner)
            .scalar(level)
            .scalar(image.epoch)
            .scalar(image.generation)
            .scalar(image.macro_step)
            .scalar(image.attempt.ordinal())
            .scalar(image.time)
            .scalar(duration);
        level_image.fields.emplace_back(accepted);
        level_image.sources.push_back(
            TemporalSampleSource::point(image.time, std::move(point).release()));
        const auto& samples = history.validated_samples(name);
        for (std::size_t slot = 0; slot < d.slots; ++slot) {
          const auto& field = history.histories.at(name)[slot];
          require_same_field_contract_(field, accepted, "temporal field retained source");
          level_image.fields.emplace_back(field);
          level_image.sources.push_back(TemporalSampleSource::history(
              samples[slot], history.initialized.at(name), history.slot_dt.at(name)[slot]));
        }
        exact.scalar(level);
        for (const auto& source : level_image.sources)
          source.append_contract(exact);
        image.levels.push_back(std::move(level_image));
      }
      source_authority = std::move(exact).release();
    });
    if (!all_ranks_agree_exact_ordered_byte_pairs({{"temporal-field-source", source_authority}},
                                                  lane))
      throw std::invalid_argument("temporal field source authority differs across ranks");
    interaction_phase(lane, [&] { staged.emplace(key, std::move(image)); });
  }
  temporal_field_images_.swap(staged);
}

public:
void declare_temporal_field_source(std::int64_t solve, int input, int block,
                                   const std::string& name, std::size_t slots,
                                   const std::string& state, const std::string& space,
                                   const std::string& clock, const std::string& interpolation,
                                   const std::string& identity,
                                   HistoryTransferCallback callback) const {
  const TemporalFieldKey key{solve, input};
  const auto found = temporal_field_declarations_.find(key);
  const int owner = sys_block(block);
  if (solve < 0 || input < 0 || slots < 2 || name.empty() || state.empty() || space.empty() ||
      clock.empty() || interpolation.empty() || identity.empty() || !callback)
    throw std::invalid_argument("temporal field declaration requires complete source authority");
  if (found != temporal_field_declarations_.end()) {
    const auto& old = found->second;
    if (old.owner != owner || old.name != name || old.slots != slots || old.state != state ||
        old.space != space || old.clock != clock || old.interpolation != interpolation ||
        old.identity != identity)
      throw std::invalid_argument("temporal field source key was rebound");
    return;
  }
  if (temporal_field_declarations_sealed_)
    throw std::logic_error("temporal field source declarations are sealed");
  temporal_field_declarations_.emplace(
      key, TemporalFieldDeclaration{owner, name, slots, state, space, clock, interpolation,
                                    identity, std::move(callback)});
}

double temporal_field_evaluation_time(std::int64_t solve, int input) const {
  const auto& image = temporal_field_images_.at({solve, input});
  // A local endpoint must be the same point as its macro-window endpoint. Adding
  // independently rounded local durations can put a valid final stage beyond that
  // window (e.g. (.08 + .02) + .02 > .08 + .04). Compose the exact clock phases
  // first, then use the one frozen macro origin/duration for every level.
  if (current_interval_begin_phase_ < ::pops::amr::Rational{0, 1} ||
      ::pops::amr::Rational{1, 1} < current_interval_end_phase_ ||
      !(current_interval_begin_phase_ < current_interval_end_phase_) ||
      stage_time_ < ::pops::amr::Rational{0, 1} || ::pops::amr::Rational{1, 1} < stage_time_)
    throw std::invalid_argument("temporal field evaluation has a foreign exact clock phase");
  const auto phase = current_interval_begin_phase_ +
                     stage_time_ * (current_interval_end_phase_ - current_interval_begin_phase_);
  return image.time + image.duration * phase.value();
}

const field_type& temporal_field_input(std::int64_t solve, int input, const field_type& current,
                                       int origin_level, double target_time) const {
  const auto& lane = prepared_execution_lane();
  const TemporalFieldKey key{solve, input};
  std::vector<Real> weights;
  std::string authority;
  interaction_phase(lane, [&] {
    const auto& d = temporal_field_declarations_.at(key);
    auto& image = temporal_field_images_.at(key);
    if (!image.attempt.visible() || !image.attempt.same_attempt(resource_attempt()))
      throw std::invalid_argument("temporal field source belongs to a revoked or foreign attempt");
    if (image.epoch != runtime_->topology_epoch() ||
        image.generation != runtime_->materialization_generation())
      throw std::invalid_argument("temporal field source belongs to a foreign topology generation");
    if (image.macro_step != facade_->macro_step() || image.time != facade_->time())
      throw std::invalid_argument("temporal field source belongs to a foreign macro clock");
    if (origin_level < 0 || origin_level >= nlev() || active_level_ < 0 || active_level_ >= nlev())
      throw std::invalid_argument("temporal field source belongs to a foreign level");
    if (!std::isfinite(target_time) ||
        target_time != temporal_field_evaluation_time(solve, input) || target_time < image.time ||
        target_time > image.time + image.duration)
      throw std::invalid_argument("temporal field source belongs to a foreign exact point");
    require_same_field_contract_(current, image.levels.at(origin_level).fields.front(),
                                 "temporal field active stage source");
    if (active_level_ == origin_level)
      return;
    auto& selected = image.levels.at(active_level_);
    HistoryTransferInput arguments;
    arguments.exact_coefficient_bytes = max_exact_coefficient_bytes_;
    arguments.accepted_time = Real(target_time);
    arguments.temporal_ratio = Real(1);
    for (const auto& source : selected.sources) {
      arguments.times.push_back(Real(source.time));
      arguments.intervals.push_back(Real(source.interval));
      arguments.parameters.push_back(source.available ? Real(1) : Real(0));
    }
    const auto result = d.callback(arguments);
    if (result.rows.size() != 1 || result.rows[0].time != target_time ||
        std::bit_cast<std::uint64_t>(double(result.rows[0].interval)) != 0)
      throw std::invalid_argument(
          "temporal field graph must evaluate its exact point with zero interval");
    validate_history_transfer_weights(result.rows[0]);
    if (result.rows[0].weights.size() != selected.sources.size())
      throw std::invalid_argument("temporal field target source width differs from matrix");
    for (std::size_t i = 0; i < selected.sources.size(); ++i)
      if (!result.rows[0].exact_weights.empty() && !result.rows[0].exact_weights[i].zero() &&
          !selected.sources[i].available)
        throw std::invalid_argument(
            "temporal field reconstruction uses an unavailable exact source");
    weights = result.rows[0].weights;
    ExactContractBuilder invocation;
    invocation.text("pops.temporal-field-point-evaluation.v1")
        .bytes(d.identity)
        .scalar(solve)
        .scalar(input)
        .scalar(origin_level)
        .scalar(active_level_)
        .scalar(image.epoch)
        .scalar(image.generation)
        .scalar(image.macro_step)
        .scalar(image.attempt.ordinal())
        .scalar(image.time)
        .scalar(image.duration)
        .scalar(target_time);
    append_history_transfer_weight_contract(invocation, result.rows[0]);
    authority = temporal_reconstruction_contract(selected.sources, weights,
                                                 std::move(invocation).release());
  });
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"temporal-field-evaluation", authority}}, lane))
    throw std::invalid_argument("temporal field evaluation differs across ranks");
  if (active_level_ == origin_level)
    return current;
  auto& selected = temporal_field_images_.at(key).levels.at(active_level_);
  interaction_phase(lane, [&] {
    if (!selected.evaluated)
      selected.evaluated.emplace(selected.fields.front());
    auto& result = *selected.evaluated;
    result.set_val(Real(0));
    for (std::size_t i = 0; i < weights.size(); ++i)
      if (weights[i] != Real(0))
        pops::saxpy(result, weights[i], selected.fields[i]);
    Kokkos::fence();
    selected.evaluation_authority = authority;
  });
  return *selected.evaluated;
}
