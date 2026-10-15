// Included in AmrProgramContext. Topology is representation, not a transfer law.
private:
using FieldTopology = FieldTopologyView<Dim>;
using TopologyViews = std::map<std::pair<int, int>, FieldTopology>;
struct RegridFieldQualification {
  int block, level;
  typename FieldTopology::Side side;
};
struct ContinuationRegridFrame {
  PreparedResourceAttempt attempt;
  std::map<int, ContinuationStore<Dim>> sources, targets;
  TopologyViews source_views, target_views;
  std::map<int, int> selected_sources;
  std::map<const field_type*, RegridFieldQualification> source_fields, target_fields, copied_fields;
  std::vector<FieldTopology> derived_views;
  std::vector<std::shared_ptr<const field_type>> allocated_fields;
  double time = 0.;
  std::int64_t step = 0;
  std::uint64_t source_epoch = 0, source_generation = 0;
  std::uint64_t target_epoch = 0, target_generation = 0;
  bool target_ready = false, published = false;
};
mutable PreparedResourceCache continuation_regrid_lifetime_;
mutable std::optional<ContinuationRegridFrame> continuation_regrid_;

TopologyViews capture_continuation_topologies_(typename FieldTopology::Side side,
                                               PreparedResourceAttempt attempt) const {
  TopologyViews views;
  for (int block = 0; block < n_blocks(); ++block)
    for (int level = 0; level < nlev(); ++level) {
      const int owner = sys_block(block);
      const auto& field = facade_->prepared_amr_block_state(owner, level);
      const auto* active = facade_->prepared_amr_block_level_active_mask(owner, level);
      const auto* fraction = facade_->prepared_amr_block_level_volume_fraction(owner, level);
      auto data = std::make_shared<typename FieldTopology::Data>(typename FieldTopology::Data{
          attempt, level, block, runtime_->topology_epoch(), runtime_->materialization_generation(),
          side, facade_->prepared_amr_level_geometry(level), field.layout(), field.distribution(),
          field.local_rank(), field.ghosts(), field.ncomp(),
          active ? std::make_shared<const field_type>(*active) : nullptr,
          fraction ? std::make_shared<const field_type>(*fraction) : nullptr,
          std::make_shared<const field_type>(
              facade_->prepared_amr_block_level_coverage_mask(owner, level))});
      views.emplace(std::pair{block, level}, FieldTopology(std::move(data)));
    }
  return views;
}

ContinuationRegridFrame& require_continuation_regrid_() const {
  if (!continuation_regrid_ || !continuation_regrid_->attempt.visible())
    throw std::logic_error("Continuation regrid has no live source/candidate transaction");
  return *continuation_regrid_;
}

bool continuation_regrid_publication_deferred_() const noexcept {
  return continuation_regrid_ && continuation_regrid_->attempt.visible() &&
         !continuation_regrid_->target_ready && !continuation_regrid_->published;
}

public:
bool continuation_regrid_phase() const noexcept {
  return continuation_regrid_ && continuation_regrid_->attempt.visible() &&
         continuation_regrid_->target_ready && !continuation_regrid_->published;
}

void require_field_topology(const FieldTopology& view) const {
  view.require();
  const auto& frame = require_continuation_regrid_();
  const bool source = view.side() == FieldTopology::Side::Source;
  if (!source && !frame.target_ready)
    throw std::logic_error("target topology is not materialized for the regrid graph");
  const auto& table = source ? frame.source_views : frame.target_views;
  const auto found = table.find({view.program_block(), view.level()});
  const bool registered =
      found != table.end() && (found->second.same_view(view) ||
                               std::any_of(frame.derived_views.begin(), frame.derived_views.end(),
                                           [&](const auto& item) { return item.same_view(view); }));
  if (!registered || view.topology_epoch() != (source ? frame.source_epoch : frame.target_epoch) ||
      view.materialization_generation() !=
          (source ? frame.source_generation : frame.target_generation) ||
      (!source && (frame.target_epoch != runtime_->topology_epoch() ||
                   frame.target_generation != runtime_->materialization_generation())))
    throw std::invalid_argument(
        "Field topology belongs to another owner, layout or regrid attempt");
}

FieldTopology continuation_source_topology(int block, std::int64_t level) const {
  const auto& frame = require_continuation_regrid_();
  if (level < 0 || level >= static_cast<std::int64_t>(frame.sources.size()) - 1)
    throw std::out_of_range("source Field topology level is outside the immutable forest");
  const auto view = frame.source_views.at({block, static_cast<int>(level)});
  require_field_topology(view);
  return view;
}
FieldTopology continuation_target_topology(int block, std::int64_t level) const {
  const auto& frame = require_continuation_regrid_();
  if (level < 0 || level >= nlev())
    throw std::out_of_range("target Field topology level is outside the candidate forest");
  const auto view = frame.target_views.at({block, static_cast<int>(level)});
  require_field_topology(view);
  return view;
}

private:
std::optional<RegridFieldQualification> regrid_field_qualification_(const field_type& field) const {
  const auto& frame = require_continuation_regrid_();
  for (const auto* values : {&frame.source_fields, &frame.target_fields, &frame.copied_fields}) {
    const auto found = values->find(&field);
    if (found != values->end())
      return found->second;
  }
  return std::nullopt;
}
FieldTopology derive_field_topology_(const FieldTopology& storage, const field_type& field) const {
  require_field_topology(storage);
  auto& frame = require_continuation_regrid_();
  const auto qualification = regrid_field_qualification_(field);
  if (!qualification || qualification->block != storage.program_block())
    throw std::invalid_argument("Field topology template has no exact Continuation owner");
  if (storage.side() == FieldTopology::Side::Source &&
      (qualification->side != FieldTopology::Side::Source ||
       qualification->level != storage.level()))
    throw std::invalid_argument(
        "Field topology template belongs to another immutable source level");
  if (storage.side() == FieldTopology::Side::Source ||
      qualification->side == FieldTopology::Side::Target) {
    if (qualification->level != storage.level() || field.layout() != storage.layout() ||
        field.distribution() != storage.distribution() ||
        field.local_rank() != storage.local_rank())
      throw std::invalid_argument("Field topology template has a different qualified layout");
  }
  auto data = std::make_shared<typename FieldTopology::Data>(*storage.data_);
  data->ghosts = field.ghosts();
  data->components = field.ncomp();
  FieldTopology view(std::move(data));
  frame.derived_views.push_back(view);
  return view;
}

public:
FieldTopology continuation_source_topology(int block, std::int64_t level,
                                           const field_type& field) const {
  return derive_field_topology_(continuation_source_topology(block, level), field);
}
FieldTopology continuation_target_topology(int block, std::int64_t level,
                                           const field_type& field) const {
  return derive_field_topology_(continuation_target_topology(block, level), field);
}
void continuation_bind_regrid_field_(const field_type& source,
                                     const field_type& destination) const {
  if (!continuation_regrid_phase())
    return;
  auto& frame = require_continuation_regrid_();
  const auto qualification = regrid_field_qualification_(source);
  if (!qualification) {
    frame.copied_fields.erase(&destination);
    return;
  }
  require_same_field_contract_(source, destination, "qualified regrid Field copy");
  frame.copied_fields.insert_or_assign(&destination, *qualification);
}
void continuation_bind_regrid_import_(const field_type& field,
                                      const FieldTopology& topology) const {
  require_field_topology(topology);
  topology.require_field(field);
  if (topology.side() != FieldTopology::Side::Target)
    throw std::invalid_argument("Field import requires an exact target topology");
  require_continuation_regrid_().copied_fields.insert_or_assign(
      &field, RegridFieldQualification{topology.program_block(), topology.level(),
                                       FieldTopology::Side::Target});
}

// Called before the engine replaces any topology, masks or Field storage.
bool begin_continuation_regrid() const {
  require_facade_execution_();
  refresh_resources_();
  return continuation_local_phase(
      [&] {
        const auto root = amr_continuation_accepted_.find(-1);
        if (root == amr_continuation_accepted_.end() || !root->second.initialized())
          return false;  // Explicit bootstrap has no accepted scientific forest yet.
        if (continuation_regrid_ || level_continuation_attempt_active_)
          throw std::logic_error(
              "Continuation regrid cannot overlap an unfinished scientific attempt");
        continuation_require_forest_(amr_continuation_accepted_);
        ContinuationRegridFrame frame;
        frame.attempt = continuation_regrid_lifetime_.begin_attempt();
        frame.sources = amr_continuation_accepted_;
        frame.time = facade_->time();
        frame.step = static_cast<std::int64_t>(facade_->macro_step());
        frame.source_epoch = runtime_->topology_epoch();
        frame.source_generation = runtime_->materialization_generation();
        frame.source_views =
            capture_continuation_topologies_(FieldTopology::Side::Source, frame.attempt);
        for (const auto& [level, store] : frame.sources) {
          if (level < 0)
            continue;
          continuation_detail::walk_value<Dim>(
              store.read(), store.configuration().byte_capacity,
              [&](const ContinuationValue<Dim>& value) {
                if (value.kind() == ContinuationKind::Field)
                  frame.source_fields.emplace(
                      &value.field(), RegridFieldQualification{value.field_owner(), level,
                                                               FieldTopology::Side::Source});
                return true;
              },
              [](const ContinuationValue<Dim>&) {});
        }
        continuation_regrid_.emplace(std::move(frame));
        return true;
      },
      "immutable complete Continuation regrid source capture");
}

// Called only after target hierarchy and its declared geometry providers exist.
void prepare_continuation_regrid() const {
  refresh_resources_();
  continuation_local_phase(
      [&] {
        auto& frame = require_continuation_regrid_();
        if (frame.target_ready || frame.published)
          throw std::logic_error("Continuation regrid target was already prepared");
        frame.target_epoch = runtime_->topology_epoch();
        frame.target_generation = runtime_->materialization_generation();
        frame.target_views =
            capture_continuation_topologies_(FieldTopology::Side::Target, frame.attempt);
        for (int block = 0; block < n_blocks(); ++block)
          for (int level = 0; level < nlev(); ++level)
            frame.target_fields.emplace(
                &facade_->prepared_amr_block_state(sys_block(block), level),
                RegridFieldQualification{block, level, FieldTopology::Side::Target});
        continuation_parent_working_ = frame.sources.at(-1);
        continuation_parent_phase_ = false;
        continuation_children_.clear();
        level_continuation_pending_.clear();
        level_continuation_macro_bases_.clear();
        level_continuation_working_.clear();
        level_continuation_macro_start_ = level_continuation_macro_end_ = frame.time;
        level_continuation_macro_attempt_ = frame.attempt;
        level_continuation_attempt_active_ = true;
        level_continuation_sync_ = true;
        frame.target_ready = true;
      },
      "complete Continuation regrid target preparation");
}

void begin_regrid_invocation() const {
  const auto& frame = require_continuation_regrid_();
  if (!frame.target_ready || frame.published)
    throw std::logic_error("regrid invocation requires an unpublished target hierarchy");
  program_values_.revoke_();
  require_continuation_regrid_().copied_fields.clear();
  (void)prepared_resources_.begin_attempt(prepared_execution_lane());
  drain_program_value_writes_();
  rhs_input_traces_.clear();
  current_dt_ = 0.;
  current_interval_start_time_ = frame.time;
  current_interval_begin_phase_ = current_interval_end_phase_ = {0, 1};
  stage_time_ = {0, 1};
  logical_substep_ = 0;
}

std::int64_t continuation_regrid_source_count() const {
  return static_cast<std::int64_t>(require_continuation_regrid_().sources.size()) - 1;
}

void end_continuation_regrid() const noexcept {
  continuation_regrid_lifetime_.reject_attempt();
  continuation_regrid_.reset();
  // Any retained target is reauthenticated from the newly accepted Store.
  // Old source aliases must not carry a previous generation across this seam.
  revoke_continuation_field_copies_();
}

void install_continuation_regrid_graph(std::function<void()> graph) const {
  if (!graph)
    throw std::invalid_argument("Continuation regrid requires a compiled closed graph");
  facade_->install_program_continuation_regrid_hooks(
      [this] { return begin_continuation_regrid(); },
      [this, graph = std::move(graph)](bool changed) {
        if (!changed) {
          end_continuation_regrid();
          return;
        }
        prepare_continuation_regrid();
        graph();
        continuation_local_phase(
            [&] {
              if (!require_continuation_regrid_().published)
                throw std::logic_error(
                    "Continuation regrid graph did not publish its complete forest");
            },
            "complete Continuation regrid publication");
        end_continuation_regrid();
      });
}
