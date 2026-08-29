/// Copy one exact valid-cell component span without exposing distributed storage to generated
/// code.  Aliasing copies select their component direction before the kernel launches, so an
/// overlapping in-place pack cannot overwrite a value that has not yet been read.
void copy_component_span(field_type& destination, int destination_component,
                         const field_type& source, int source_component,
                         int component_count) const {
  if (component_count <= 0 || destination_component < 0 || source_component < 0 ||
      destination_component > destination.ncomp() - component_count ||
      source_component > source.ncomp() - component_count)
    throw std::invalid_argument("AMR Program component-span copy has an invalid range");
  require_same_layout_(destination, source, "AMR Program component-span copy");
  for (std::size_t local = 0; local < destination.local_size(); ++local) {
    if (destination.global_index(local) != source.global_index(local))
      throw std::logic_error("AMR Program component-span copy found inconsistent local ownership");
  }
  if (&destination == &source && destination_component == source_component)
    return;
  const bool copy_backward = &destination == &source && destination_component > source_component &&
                             destination_component < source_component + component_count;
  for (std::size_t local = 0; local < destination.local_size(); ++local) {
    const FieldView<Real, Dim> output = destination.fab(local).view();
    const FieldView<const Real, Dim> input = std::as_const(source).fab(local).view();
    for_each_cell(destination.box(local), [=] POPS_HD(const Index<Dim>& cell) {
      if (copy_backward) {
        for (int offset = component_count; offset-- > 0;)
          output(cell, destination_component + offset) = input(cell, source_component + offset);
      } else {
        for (int offset = 0; offset < component_count; ++offset)
          output(cell, destination_component + offset) = input(cell, source_component + offset);
      }
    });
  }
  count_kernel_();
}

void copy_grown_component_span(field_type& destination, int destination_component,
                               const field_type& source, int source_component,
                               int component_count) const {
  if (component_count <= 0 || destination_component < 0 || source_component < 0 ||
      destination_component > destination.ncomp() - component_count ||
      source_component > source.ncomp() - component_count)
    throw std::invalid_argument("AMR Program grown component-span copy has an invalid range");
  require_same_layout_(destination, source, "AMR Program grown component-span copy");
  if (destination.ghosts() != source.ghosts())
    throw std::invalid_argument("AMR Program grown component-span copy ghosts differ");
  for (std::size_t local = 0; local < destination.local_size(); ++local) {
    if (destination.global_index(local) != source.global_index(local) ||
        destination.fab(local).grown_box() != source.fab(local).grown_box())
      throw std::logic_error(
          "AMR Program grown component-span copy found inconsistent local ownership");
  }
  if (&destination == &source && destination_component == source_component)
    return;
  const bool copy_backward = &destination == &source && destination_component > source_component &&
                             destination_component < source_component + component_count;
  for (std::size_t local = 0; local < destination.local_size(); ++local) {
    const FieldView<Real, Dim> output = destination.fab(local).view();
    const FieldView<const Real, Dim> input = std::as_const(source).fab(local).view();
    for_each_cell(destination.fab(local).grown_box(), [=] POPS_HD(const Index<Dim>& cell) {
      if (copy_backward) {
        for (int offset = component_count; offset-- > 0;)
          output(cell, destination_component + offset) = input(cell, source_component + offset);
      } else {
        for (int offset = 0; offset < component_count; ++offset)
          output(cell, destination_component + offset) = input(cell, source_component + offset);
      }
    });
  }
  count_kernel_();
}

/// Register one level-qualified exact-ranked history ring.  The generated AMR installer invokes
/// this while constructing each level bundle, so one authored history maps to one immutable
/// layout contract per active level instead of a 2-D or owner-erased global buffer.
void register_history(const std::string& name, int lag, int ncomp, int program_owner,
                      const std::string& state_identity, const std::string& space_identity,
                      const std::string& clock_identity,
                      const std::string& interpolation_identity) const {
  if (name.empty() || lag < 1 || program_owner < 0 || state_identity.empty() ||
      space_identity.empty() || clock_identity.empty() || interpolation_identity.empty())
    throw std::invalid_argument(
        "AMR Program history requires complete owner/state/space/clock identities");
  const int runtime_owner = sys_block(program_owner);
  refresh_resources_();
  // The host invokes history materialization while the Program image is still detached.  At that
  // point the exact block prototype is carried by PreparedAmrTopologyView; consulting facade_
  // would both be null and would leak the accepted hierarchy into DSO preparation.
  const field_type& prototype = preparation_view_ != nullptr
                                    ? preparation_view_->block_prototypes
                                          .at(static_cast<std::size_t>(runtime_owner))
                                          .at(static_cast<std::size_t>(active_level_))
                                    : facade_->program_prepared_amr_block_state_(runtime_owner,
                                                                                active_level_);
  const int components = ncomp < 0 ? prototype.ncomp() : ncomp;
  if (components < 1)
    throw std::invalid_argument("AMR Program history component count must be positive");
  const std::string key = history_key_(name, active_level_);
  auto& manager = runtime_state().hist_;
  const int depth = lag + 1;
  const auto found = manager.histories.find(key);
  if (found != manager.histories.end()) {
    const field_type& retained = found->second.front();
    if (manager.depth.at(key) != depth || manager.owner.at(key) != runtime_owner ||
        retained.layout() != prototype.layout() ||
        retained.distribution() != prototype.distribution() ||
        retained.local_rank() != prototype.local_rank() || retained.ncomp() != components ||
        retained.ghosts() != prototype.ghosts() ||
        manager.state_identity.at(key) != state_identity ||
        manager.space_identity.at(key) != space_identity ||
        manager.clock_identity.at(key) != clock_identity ||
        manager.interpolation_identity.at(key) != interpolation_identity)
      throw std::runtime_error(
          "AMR Program history identity changed after exact-ranked registration");
    history_levels_.insert_or_assign(key, active_level_);
    return;
  }

  std::vector<field_type> ring;
  ring.reserve(static_cast<std::size_t>(depth));
  for (int slot = 0; slot < depth; ++slot)
    ring.push_back(make_scratch_(prototype, components, prototype.ghosts()));
  manager.histories.emplace(key, std::move(ring));
  manager.depth[key] = depth;
  manager.initialized[key] = false;
  manager.fill_count[key] = 0;
  manager.store_pending[key] = false;
  manager.owner[key] = runtime_owner;
  manager.state_identity[key] = state_identity;
  manager.space_identity[key] = space_identity;
  manager.clock_identity[key] = clock_identity;
  manager.interpolation_identity[key] = interpolation_identity;
  manager.slot_dt[key] = std::vector<Real>(static_cast<std::size_t>(depth), Real(0));
  history_flux_expressions_.emplace(key,
                                    std::vector<FluxExpression>(static_cast<std::size_t>(depth)));
  history_levels_.emplace(key, active_level_);
  if (history_epoch_ == std::numeric_limits<std::uint64_t>::max()) {
    history_epoch_ = runtime_->topology_epoch();
    history_generation_ = runtime_->materialization_generation();
  }
}

field_type& history(const std::string& name, int lag, int program_owner) const {
  require_history_owner_(program_owner);
  return history_slot_(name, lag, /*zero_start=*/false, /*components=*/-1);
}
field_type& history(const std::string& name, int lag = 1) const {
  return history_slot_(name, lag, /*zero_start=*/false, /*components=*/-1);
}
field_type& history_zero_start(const std::string& name, int lag, int ncomp,
                               int program_owner) const {
  require_history_owner_(program_owner);
  return history_slot_(name, lag, /*zero_start=*/true, ncomp);
}
field_type& history_zero_start(const std::string& name, int lag, int ncomp = -1) const {
  return history_slot_(name, lag, /*zero_start=*/true, ncomp);
}

void store_history(const std::string& name, const field_type& value, int program_owner) const {
  require_history_owner_(program_owner);
  store_history_(name, value);
}
void store_history(const std::string& name, const field_type& value) const {
  store_history_(name, value);
}

void rotate_histories() const {
  rotate_histories_(std::nullopt);
}
void rotate_histories(const std::string& clock_identity) const {
  if (clock_identity.empty())
    throw std::invalid_argument("AMR Program history rotation requires a clock identity");
  rotate_histories_(clock_identity);
}

void interpolate_history_linear(field_type& output, const std::string& name, int max_lag,
                                int program_owner, const std::string& source_clock,
                                const std::string& target_clock, int target_step,
                                Real target_offset) const {
  require_history_owner_(program_owner);
  if (max_lag < 1 || !std::isfinite(static_cast<double>(target_offset)))
    throw std::invalid_argument("AMR linear history interpolation has an invalid target");
  const std::string key = history_key_(name, active_level_);
  auto& manager = runtime_state().hist_;
  const auto found = manager.histories.find(key);
  if (found == manager.histories.end() || manager.depth.at(key) <= max_lag ||
      !manager.initialized.at(key))
    throw std::runtime_error(
        "AMR linear history interpolation requires an initialized retained ring");
  require_same_field_contract_(output, found->second.front(), "AMR linear history interpolation");

  const double source_ticks = static_cast<double>(clock_schedule_.ticks_per_macro(source_clock));
  const double target_ticks = static_cast<double>(clock_schedule_.ticks_per_macro(target_clock));
  const double coordinate =
      (static_cast<double>(target_step) + static_cast<double>(target_offset)) * source_ticks /
      target_ticks;
  if (!std::isfinite(coordinate) || coordinate > 0.0 || coordinate < -static_cast<double>(max_lag))
    throw std::runtime_error(
        "AMR linear history interpolation target lies outside retained timestamps");
  if (coordinate == 0.0) {
    copy_valid_(found->second.front(), output);
    count_kernel_();
    return;
  }

  const int older_lag = static_cast<int>(std::ceil(-coordinate));
  if (older_lag < 1 || older_lag > max_lag)
    throw std::runtime_error("AMR linear history interpolation could not select bracketing slots");
  double newer_time = static_cast<double>(physical_time());
  double older_time = newer_time;
  double bracket_dt = 0.0;
  for (int selected_lag = 1; selected_lag <= older_lag; ++selected_lag) {
    const double interval =
        static_cast<double>(manager.slot_dt.at(key)[static_cast<std::size_t>(selected_lag)]);
    if (!std::isfinite(interval) || !(interval > 0.0))
      throw std::runtime_error(
          "AMR linear history interpolation requires positive exact slot timestamps");
    bracket_dt = interval;
    older_time = newer_time - interval;
    if (selected_lag != older_lag)
      newer_time = older_time;
  }
  const double logical_fraction = coordinate + static_cast<double>(older_lag);
  const double target_time = older_time + logical_fraction * bracket_dt;
  const double alpha = (target_time - older_time) / (newer_time - older_time);
  if (!std::isfinite(alpha) || alpha < 0.0 || alpha > 1.0)
    throw std::runtime_error(
        "AMR linear history interpolation target does not bracket retained timestamps");
  lincomb(output, Real(1) - static_cast<Real>(alpha),
          found->second[static_cast<std::size_t>(older_lag)], static_cast<Real>(alpha),
          found->second[static_cast<std::size_t>(older_lag - 1)]);
}

bool cache_should_update(ProgramCacheSlot slot, int every_n) const {
  const bool due = runtime_state().cache_.is_due(slot, macro_step(), every_n);
  runtime_state().profiler_.count(due ? "cache_misses" : "cache_hits");
  return due;
}
void cache_store_aux(ProgramCacheSlot slot) const {
  (void)runtime_state().cache_.plan_entry(slot);
}
void cache_restore_aux(ProgramCacheSlot slot) const {
  (void)runtime_state().cache_.plan_entry(slot);
}
void cache_store_scratch(ProgramCacheSlot slot, const field_type& scratch) const {
  runtime_state().cache_.store(slot, scratch, macro_step());
}
void cache_restore_scratch(ProgramCacheSlot slot, field_type& scratch) const {
  runtime_state().cache_.restore_into(slot, scratch);
}
void cache_accumulate_dt(ProgramCacheSlot slot, Real dt) const {
  runtime_state().cache_.accumulate_dt(slot, dt);
}
Real cache_effective_dt(ProgramCacheSlot slot, Real dt) const {
  return runtime_state().cache_.effective_dt(slot, dt);
}

bool schedule_domain_occurs(ScheduleDomainKind kind, const std::string& clock,
                            const std::string& stage_identity, int level) const {
  return schedule_coordinate_(kind, clock, stage_identity, level).has_value();
}
bool schedule_is_due(ProgramCacheSlot slot, int every_n, ScheduleDomainKind kind, const std::string& clock,
                     const std::string& stage_identity, int level) const {
  (void)runtime_state().cache_.plan_entry(slot);
  if (every_n <= 0)
    throw std::invalid_argument("AMR Program schedule has an invalid period");
  const auto coordinate = schedule_coordinate_(kind, clock, stage_identity, level);
  return coordinate && coordinate->value % every_n == 0;
}
bool schedule_at_start(ScheduleDomainKind kind, const std::string& clock,
                       const std::string& stage_identity, int level) const {
  const auto coordinate = schedule_coordinate_(kind, clock, stage_identity, level);
  return coordinate && coordinate->value == 0;
}
bool schedule_decision(ProgramCacheSlot slot, bool due, bool cache_backed) const {
  (void)runtime_state().cache_.plan_entry(slot);
  return runtime_state().profiler_.schedule_decision(due, cache_backed);
}
[[noreturn]] void scheduler_error(const std::string& message) const {
  throw std::runtime_error(message.empty() ? "AMR Program scheduled node is unavailable" : message);
}

const field_type* pointwise_active_mask(int program_block, const field_type& field) const {
  refresh_resources_();
  const int runtime_block = sys_block(program_block);
  const field_type& accepted =
      facade_->program_prepared_amr_block_state_(runtime_block, active_level_);
  require_same_layout_(field, accepted, "AMR Program pointwise mask");
  const field_type* const active =
      facade_->program_prepared_amr_block_level_active_mask_(runtime_block, active_level_);
  if (active != nullptr)
    require_same_layout_(*active, accepted, "AMR Program pointwise active mask");
  return active;
}

/// Reduce generated per-cell status over every live owner level.  Inactive values, including
/// inactive NaNs, are ignored.  An active negative or non-finite status is failure code 3; a
/// globally empty active domain is 0; otherwise the lane maximum finite severity is returned.
Real pointwise_status_max(int program_block, const field_type& status,
                          const field_type* active_cells, const ExecutionLane& lane) const {
  require_prepared_lane_(lane, "AMR Program pointwise status");
  std::exception_ptr local_error;
  long ncomp_error = 1;
  long mask_error = 1;
  MaskedMaxLocalResult local;
  try {
    ncomp_error = status.ncomp() == 1 ? 0L : 1L;
    mask_error = 0;
    for_each_owner_active_level_(
        program_block, status, nullptr,
        [&](const field_type& level_status, const field_type*, const field_type* expected) {
          if (&level_status == &status && active_cells != expected)
            mask_error = 1;
          if (ncomp_error != 0)
            return;
          const MaskedMaxLocalResult part =
              pops::reduce_masked_max_local(level_status, 0, expected);
          local.has_active = local.has_active || part.has_active;
          local.has_invalid = local.has_invalid || part.has_invalid;
          local.maximum = std::max(local.maximum, part.maximum);
        },
        true);
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR Program pointwise status reduction failed collectively");
  }
  if (all_reduce_max(ncomp_error, lane) != 0)
    throw std::invalid_argument("AMR Program pointwise status requires exactly one component");
  if (all_reduce_max(mask_error, lane) != 0)
    throw std::invalid_argument("AMR Program pointwise status received a foreign active-cell mask");
  if (all_reduce_max(local.has_invalid ? 1L : 0L, lane) != 0)
    return Real(3);
  if (all_reduce_max(local.has_active ? 1L : 0L, lane) == 0)
    return Real(0);
  const Real maximum = static_cast<Real>(all_reduce_max(static_cast<double>(local.maximum), lane));
  return std::isfinite(maximum) ? maximum : Real(3);
}
