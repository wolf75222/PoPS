void store_history_(const std::string& name, const field_type& value) const {
  const std::string key = history_key_(name, active_level_);
  auto& manager = runtime_state().hist_;
  const auto found = manager.histories.find(key);
  if (found == manager.histories.end())
    throw std::out_of_range("AMR Program history is not registered on the active level");
  require_same_field_contract_(found->second.front(), value, "AMR Program history store");
  if (!std::isfinite(current_dt_) || !(current_dt_ > 0.0))
    throw std::logic_error("AMR Program history store has no positive exact interval");
  found->second.front() = value;
  auto& dts = manager.slot_dt.at(key);
  if (dts.size() != found->second.size())
    throw std::logic_error("AMR Program history dt ledger differs from its ring depth");
  if (!manager.initialized.at(key)) {
    for (std::size_t slot = 1; slot < found->second.size(); ++slot) {
      found->second[slot] = value;
      dts[slot] = static_cast<Real>(current_dt_);
    }
  }
  manager.initialized[key] = true;
  manager.store_pending[key] = true;
  dts.front() = static_cast<Real>(current_dt_);
}

void rotate_histories_(std::optional<std::string> clock_identity) const {
  auto& manager = runtime_state().hist_;
  std::vector<std::string> selected;
  for (const auto& [key, level] : history_levels_) {
    if (level != active_level_)
      continue;
    const auto ring = manager.histories.find(key);
    const auto clock = manager.clock_identity.find(key);
    const auto dts = manager.slot_dt.find(key);
    if (ring == manager.histories.end() || clock == manager.clock_identity.end() ||
        dts == manager.slot_dt.end() || ring->second.size() != dts->second.size() ||
        static_cast<int>(ring->second.size()) != manager.depth.at(key))
      throw std::runtime_error("AMR Program history registry is incomplete");
    if (!clock_identity || clock->second == *clock_identity)
      selected.push_back(key);
  }
  for (const std::string& key : selected) {
    auto& ring = manager.histories.at(key);
    for (std::size_t slot = ring.size(); slot-- > 1;)
      std::swap(ring[slot], ring[slot - 1]);
    auto& dts = manager.slot_dt.at(key);
    for (std::size_t slot = dts.size(); slot-- > 1;)
      std::swap(dts[slot], dts[slot - 1]);
    if (manager.store_pending.at(key)) {
      manager.fill_count[key] =
          std::min(static_cast<int>(ring.size()), manager.fill_count.at(key) + 1);
      manager.store_pending[key] = false;
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
      point.tick != static_cast<std::int64_t>(facade_->macro_step()) ||
      point.level != active_level_ || point.substep != logical_substep_ ||
      point.stage_fraction != stage_time_ || point.dt != current_dt_ ||
      point.physical_time != physical_time || !point.graph_identity.empty() ||
      !point.rate_identity.empty() || !point.application_identity.empty())
    throw std::invalid_argument(std::string(operation) +
                                " has a stale or foreign exact evaluation point");
}
