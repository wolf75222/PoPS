// Immutable same-SSA parent inputs for synchronous source/transport composition.
private:
struct RhsInputTraceRecord {
  std::string program, clock;
  std::uint64_t epoch = 0, generation = 0, attempt = 0;
  ::pops::amr::ClockWindow window;
  ::pops::amr::Rational stage;
  int block = -1, level = -1, evaluation = -1;
  std::int64_t source = -1;
  field_type snapshot;
};
using RhsInputTraceKey = std::tuple<int, int, int>;
using RhsInputTraceRegistry =
    std::map<RhsInputTraceKey, std::shared_ptr<const RhsInputTraceRecord>>;
mutable RhsInputTraceRegistry rhs_input_traces_;

static bool same_rhs_input_window_(const RhsInputTraceRecord& left,
                                    const RhsInputTraceRecord& right) {
  const auto same_clock = [](const auto& a, const auto& b) {
    return a.macro_step == b.macro_step && a.phase == b.phase &&
           a.physical_time == b.physical_time;
  };
  return left.program == right.program && left.clock == right.clock &&
         left.epoch == right.epoch && left.generation == right.generation &&
         left.attempt == right.attempt && left.block == right.block &&
         left.evaluation == right.evaluation && left.source == right.source &&
         left.stage == right.stage && same_clock(left.window.begin, right.window.begin) &&
         same_clock(left.window.end, right.window.end);
}

public:
RhsInputTrace capture_rhs_input_trace(int program_block, const field_type& input,
                                       int evaluation, std::int64_t source) const {
  const ExecutionLane& lane = prepared_execution_lane();
  RhsInputTrace trace;
  RhsInputTraceRegistry candidate;
  std::exception_ptr error;
  try {
    require_rate_identity_(evaluation);
    const int runtime_block = sys_block(program_block);
    if (source < 0 || active_attempt_states_.empty() || active_level_ < 0 ||
        scratch_prototype_owner_(input) != runtime_block)
      throw std::invalid_argument("AMR RHS input trace has no exact active block/SSA owner");
    for (const auto& relation : facade_->prepared_program_temporal_relations())
      if (relation.temporal_ratio() != ::pops::amr::Rational(1, 1))
        throw std::invalid_argument(
            "AMR source-transformed RHS inputs require synchronous equal clock windows");
    auto record = std::make_shared<RhsInputTraceRecord>();
    record->program = facade_->installed_program_hash();
    record->clock = primary_clock_;
    record->epoch = runtime_->topology_epoch();
    record->generation = runtime_->materialization_generation();
    record->attempt = active_subcycling_attempt_;
    record->window = active_subcycling_window_;
    record->stage = stage_time_;
    record->block = runtime_block;
    record->level = active_level_;
    record->evaluation = evaluation;
    record->source = source;
    if (record->program.empty() || record->clock.empty())
      throw std::logic_error("AMR RHS input trace has no compiled Program/clock authority");
    const RhsInputTraceKey key{runtime_block, active_level_, evaluation};
    if (rhs_input_traces_.contains(key))
      throw std::invalid_argument("AMR RHS input trace requires a unique SSA evaluation per attempt");
    if (active_level_ > 0) {
      const auto parent = rhs_input_traces_.find({runtime_block, active_level_ - 1, evaluation});
      if (parent == rhs_input_traces_.end() || !parent->second ||
          !same_rhs_input_window_(*parent->second, *record))
        throw std::invalid_argument(
            "AMR RHS input trace lacks its matching parent SSA evaluation/window/stage");
      trace.parent_ = parent->second;
    }
    // MultiFab copies may share storage. Allocate explicitly and copy the valid
    // cells now, before any later generated scratch or accepted-state reuse.
    record->snapshot = make_scratch_(input, input.ncomp(), input.ghosts());
    copy_valid_(input, record->snapshot);
    device_fence();
    trace.owner_ = this;
    trace.input_ = &input;
    trace.current_ = record;
    candidate = rhs_input_traces_;
    candidate.emplace(key, std::move(record));
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane, "AMR RHS input trace capture failed collectively");
  rhs_input_traces_.swap(candidate);
  return trace;
}

private:
const field_type* rhs_input_parent_(int runtime_block, const field_type& input, int evaluation,
                                    const RhsInputTrace* trace) const {
  if (trace == nullptr)
    return staged_parent_for_block_(runtime_block);
  const field_type* parent = nullptr;
  std::exception_ptr error;
  try {
    if (active_attempt_states_.empty() || trace->owner_ != this || trace->input_ != &input ||
        !trace->current_)
      throw std::invalid_argument("AMR RHS input trace is foreign to this evaluation");
    const auto& record = *trace->current_;
    const auto found = rhs_input_traces_.find({runtime_block, active_level_, evaluation});
    if (found == rhs_input_traces_.end() || found->second != trace->current_ ||
        record.program != facade_->installed_program_hash() || record.clock != primary_clock_ ||
        record.epoch != runtime_->topology_epoch() ||
        record.generation != runtime_->materialization_generation() ||
        record.attempt != active_subcycling_attempt_ || record.level != active_level_ ||
        record.block != runtime_block || record.evaluation != evaluation ||
        record.stage != stage_time_ || record.window.begin != active_subcycling_window_.begin ||
        record.window.end != active_subcycling_window_.end)
      throw std::invalid_argument("AMR RHS input trace is stale or has a different exact stage");
    require_same_field_contract_(input, record.snapshot, "AMR RHS immutable input trace");
    if (active_level_ == 0) {
      if (trace->parent_)
        throw std::invalid_argument("root AMR RHS input trace cannot name a parent");
    } else {
      const auto expected = rhs_input_traces_.find({runtime_block, active_level_ - 1, evaluation});
      if (!trace->parent_ || expected == rhs_input_traces_.end() ||
          expected->second != trace->parent_ || !same_rhs_input_window_(*trace->parent_, record))
        throw std::invalid_argument("AMR RHS input trace lost its exact immutable parent");
      parent = &trace->parent_->snapshot;
    }
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, prepared_execution_lane(),
                                 "AMR RHS input trace consumption failed collectively");
  return parent;
}

public:
