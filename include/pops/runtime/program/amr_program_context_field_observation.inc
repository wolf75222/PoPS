// An observation is a value, not a later lookup into a mutable solver workspace.
// The exact tower also supplies the producer authority of persistent local leaves.
// No collective reduction is moved in time.
class HierarchyFieldObservationSnapshot {
 public:
  const field_type& local_field() const {
    if (!attempt_.visible())
      throw std::invalid_argument("field observation belongs to a revoked attempt");
    return fields_.at(static_cast<std::size_t>(origin_level_));
  }
  std::uint64_t retained_bytes() const noexcept { return retained_bytes_; }

 private:
  friend class AmrProgramContext;
  const AmrProgramContext* owner_ = nullptr;
  PreparedResourceAttempt attempt_;
  std::uint64_t epoch_ = 0, generation_ = 0, retained_bytes_ = 0;
  std::int64_t solve_ = -1, observation_ = -1;
  int origin_level_ = -1, component_ = -1;
  bool retains_halo_ = false;
  ContinuationFieldSupport::Ref support_;
  std::vector<field_type> fields_, active_;
};

// Weak ownership authenticates an exact copied leaf without retaining abandoned
// observations or treating an unknown allocation with the same layout as solved.
mutable std::vector<std::weak_ptr<const HierarchyFieldObservationSnapshot>>
    hierarchy_field_observations_;

std::shared_ptr<const HierarchyFieldObservationSnapshot> hierarchy_field_observation_source(
    const field_type& field) const {
  for (const auto& weak : hierarchy_field_observations_) {
    const auto snapshot = weak.lock();
    if (!snapshot ||
        &snapshot->fields_.at(static_cast<std::size_t>(snapshot->origin_level_)) != &field)
      continue;
    if (snapshot->owner_ != this || !snapshot->attempt_.visible() ||
        !snapshot->attempt_.same_attempt(resource_attempt()) ||
        snapshot->epoch_ != runtime_->topology_epoch() ||
        snapshot->generation_ != runtime_->materialization_generation())
      throw std::invalid_argument("field observation storage has revoked or foreign authority");
    return snapshot;
  }
  return {};
}

bool recognizes_hierarchy_field_observation(const field_type& field) const {
  const auto snapshot = hierarchy_field_observation_source(field);
  if (!snapshot)
    return false;
  if (snapshot->origin_level_ != active_level_)
    throw std::invalid_argument("field observation storage belongs to another level");
  return true;
}

std::shared_ptr<const HierarchyFieldObservationSnapshot> capture_hierarchy_field_observation(
    std::int64_t solve, std::int64_t observation, int component, std::uint64_t byte_capacity,
    bool retain_halo = false, std::string_view field_provider = {}) const {
  auto& provider = configured_hierarchy_field_solver_(solve);
  const auto& lane = prepared_execution_lane();
  std::uint64_t bytes = sizeof(HierarchyFieldObservationSnapshot) +
                        sizeof(std::weak_ptr<const HierarchyFieldObservationSnapshot>);
  std::string identity;
  auto charge = [&](std::uint64_t count, std::uint64_t width) {
    if (!byte_capacity || bytes > byte_capacity ||
        (width && count > (byte_capacity - bytes) / width))
      throw std::length_error("immutable field observation exceeds its byte capacity");
    bytes += count * width;
  };
  // Account for the actual local payload and conservative container/layout
  // storage before making any snapshot allocation. Empty ranks still vote.
  prepare_spatial_collectively([&] {
    if (observation < 0 || component < 0 || !resource_attempt().visible() || active_level_ < 0 ||
        active_level_ >= nlev() || provider.level_count() != nlev())
      throw std::invalid_argument("field observation requires an exact live solve and level");
    if (!field_provider.empty()) {
      const auto& scientific = continuation_field_provider_(field_provider);
      if (!scientific.routes.contains({solve, component}) ||
          scientific.prototypes.at(active_level_).ncomp() != 1)
        throw std::invalid_argument(
            "field observation differs from its installed scientific route");
    }
    if (!field_provider.empty()) {
      charge(1, sizeof(ContinuationFieldSupport));
      charge(field_provider.size(), 1);
      charge(facade_->installed_program_hash().size() + 112, 1);
    }
    charge(static_cast<std::uint64_t>(nlev()), 2 * sizeof(field_type));
    for (int level = 0; level < nlev(); ++level) {
      const auto& source = provider.solution(level);
      const auto& active = provider.active_cell_mask(level);
      if (component >= source.ncomp() || active.ncomp() != 1 ||
          active.layout() != source.layout() || active.distribution() != source.distribution() ||
          active.local_rank() != source.local_rank() || active.local_size() != source.local_size())
        throw std::invalid_argument(
            "field observation changed its solved tuple or active quotient");
      if (retain_halo)
        for (int axis = 0; axis < Dim; ++axis)
          if (source.ghosts()[axis] < 1)
            throw std::invalid_argument("field observation gradient requires a solved halo");
      charge(source.layout().size(),
             4 * (sizeof(Box<Dim>) + sizeof(Index<Dim>) + sizeof(std::size_t)));
      charge(source.local_size(),
             2 * (sizeof(typename field_type::fab_type) + sizeof(std::size_t)));
      for (std::size_t patch = 0; patch < source.local_size(); ++patch) {
        auto grown = source.box(patch);
        for (int axis = 0; axis < Dim; ++axis)
          grown = grown.grow(axis, 1);
        charge(grown.numPts(), sizeof(Real));
        charge(active.fab(patch).grown_box().numPts(), sizeof(Real));
      }
    }
    ExactContractBuilder contract;
    contract.text("pops.immutable-field-observation@1")
        .scalar(solve)
        .scalar(observation)
        .scalar(component)
        .scalar(byte_capacity)
        .scalar(retain_halo)
        .scalar(active_level_)
        .scalar(nlev())
        .scalar(runtime_->topology_epoch())
        .scalar(runtime_->materialization_generation())
        .scalar(resource_attempt().ordinal());
    contract.text(field_provider);
    identity = std::move(contract).release();
  });
  const std::array<std::pair<std::string_view, std::string_view>, 1> authorities{
      {{"immutable-field-observation", identity}}};
  if (!all_ranks_agree_exact_ordered_byte_pairs(authorities, lane))
    throw std::invalid_argument("field observation snapshot authority differs between ranks");
  std::shared_ptr<HierarchyFieldObservationSnapshot> pending;
  Real invalid = 0;
  prepare_spatial_collectively([&] {
    pending = std::make_shared<HierarchyFieldObservationSnapshot>();
    pending->owner_ = this;
    pending->attempt_ = resource_attempt();
    pending->epoch_ = runtime_->topology_epoch();
    pending->generation_ = runtime_->materialization_generation();
    pending->retained_bytes_ = bytes;
    pending->solve_ = solve;
    pending->observation_ = observation;
    pending->component_ = component;
    pending->origin_level_ = active_level_;
    pending->retains_halo_ = retain_halo;
    if (!field_provider.empty()) {
      ExactContractBuilder point;
      point.text("pops.field-observation.point@1")
          .scalar(observation)
          .scalar(current_interval_start_time_)
          .scalar(current_dt_)
          .scalar(stage_time_.numerator)
          .scalar(stage_time_.denominator);
      const auto point_bytes = std::move(point).release();
      const std::vector<std::uint8_t> point_data(point_bytes.begin(), point_bytes.end());
      const auto origin =
          std::string("pops.field-observation.point@1:sha256:") + identity::sha256_hex(point_data);
      pending->support_ = std::make_shared<const ContinuationFieldSupport>(ContinuationFieldSupport{
          facade_->installed_program_hash(), origin, -1, active_level_, pending->epoch_,
          pending->generation_, std::string(field_provider)});
    }
    pending->fields_.reserve(static_cast<std::size_t>(nlev()));
    pending->active_.reserve(static_cast<std::size_t>(nlev()));
    for (int level = 0; level < nlev(); ++level) {
      const auto& source = provider.solution(level);
      pending->fields_.push_back(make_scratch_(source, 1, uniform_ghosts_(1)));
      pending->active_.emplace_back(provider.active_cell_mask(level));
      auto& target = pending->fields_.back();
      for (std::size_t patch = 0; patch < source.local_size(); ++patch) {
        const auto input = source.fab(patch).view();
        const auto output = target.fab(patch).view();
        const auto copied = retain_halo ? target.fab(patch).grown_box() : source.box(patch);
        for_each_cell(copied, [=] POPS_HD(const Index<Dim>& cell) {
          output(cell, 0) = input(cell, component);
        });
        invalid = std::max(invalid, for_each_cell_reduce_max(
                                        source.box(patch), [=] POPS_HD(const Index<Dim>& cell) {
                                          const Real value = input(cell, component);
                                          return std::isfinite(value) ? Real(0) : Real(1);
                                        }));
      }
    }
    Kokkos::fence();
  });
  if (all_reduce_max(invalid, lane) != Real(0))
    throw StepAttemptRejected(SolveStatus::kInvalidEvaluation, StepAttemptDisposition::kReject, 0,
                              "field_evaluation", "nonfinite field observation snapshot");
  prepare_spatial_collectively([&] {
    std::erase_if(hierarchy_field_observations_, [](const auto& weak) { return weak.expired(); });
    hierarchy_field_observations_.push_back(pending);
  });
  return pending;
}

void require_hierarchy_field_observation_(
    const std::shared_ptr<const HierarchyFieldObservationSnapshot>& snapshot) const {
  if (!snapshot || snapshot->owner_ != this || !snapshot->attempt_.visible() ||
      !snapshot->attempt_.same_attempt(resource_attempt()) ||
      snapshot->epoch_ != runtime_->topology_epoch() ||
      snapshot->generation_ != runtime_->materialization_generation() ||
      snapshot->origin_level_ != active_level_ || snapshot->solve_ < 0 ||
      snapshot->observation_ < 0 || snapshot->fields_.size() != snapshot->active_.size() ||
      snapshot->fields_.size() != static_cast<std::size_t>(nlev()))
    throw std::invalid_argument("field observation has foreign snapshot authority");
}

void observe_hierarchy_field_gradient(
    const std::shared_ptr<const HierarchyFieldObservationSnapshot>& snapshot,
    field_type& output) const {
  prepare_spatial_collectively([&] {
    require_hierarchy_field_observation_(snapshot);
    if (!snapshot->retains_halo_)
      throw std::invalid_argument("field observation did not retain the solved halo");
  });
  observe_hierarchy_field_gradient_(snapshot->local_field(), output, 0);
}

Real reduce_hierarchy_field_observation(
    const std::shared_ptr<const HierarchyFieldObservationSnapshot>& snapshot,
    std::string_view kind) const {
  const auto& lane = prepared_execution_lane();
  const bool maximum = kind == "max", minimum = kind == "min";
  Real local = maximum   ? -std::numeric_limits<Real>::infinity()
               : minimum ? std::numeric_limits<Real>::infinity()
                         : Real(0);
  prepare_spatial_collectively([&] {
    require_hierarchy_field_observation_(snapshot);
    if (kind != "sum" && kind != "abs_sum" && !maximum && !minimum)
      throw std::invalid_argument("unknown immutable field observation reduction");
    for (std::size_t level = 0; level < snapshot->fields_.size(); ++level) {
      const auto& field = snapshot->fields_[level];
      const auto& active = snapshot->active_[level];
      if (kind == "sum")
        local += pops::reduce_active_sum_local(field, 0, &active);
      else if (kind == "abs_sum")
        local += pops::reduce_active_abs_sum_local(field, 0, &active);
      else if (pops::reduce_masked_max_local(field, 0, &active).has_active) {
        if (maximum)
          local = std::max(local, pops::reduce_active_max_local(field, 0, &active));
        else
          local = std::min(local, pops::reduce_active_min_local(field, 0, &active));
      }
    }
    Kokkos::fence();
  });
  if (maximum)
    return static_cast<Real>(all_reduce_max(static_cast<double>(local), lane));
  if (minimum)
    return static_cast<Real>(all_reduce_min(static_cast<double>(local), lane));
  return static_cast<Real>(all_reduce_sum(static_cast<double>(local), lane));
}
