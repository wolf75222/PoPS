private:
using program_value_slot_type = typename ProgramValueAuthority<Dim>::Slot;
int program_value_level_() const noexcept { return active_level_; }
std::string program_value_identity_() const { return facade_->installed_program_hash(); }

program_value_slot_type program_value_slot_(int program_block, const field_type& field) const {
  refresh_resources_();
  const int owner = sys_block(program_block);
  if (scratch_prototype_owner_(field) != owner)
    throw std::invalid_argument("AMR SSA buffer belongs to another runtime block");
  if (&field == live_attempt_state_(owner,active_level_))
    return {ProgramValueStorage::State,active_level_,owner,-1,0,{},0};
  // During an active stage, an accepted image is not the stage root. It has its own AcceptedValue.
  if (active_attempt_states_.empty() &&
      &field == &facade_->prepared_amr_block_state(owner,active_level_))
    return {ProgramValueStorage::State,active_level_,owner,-1,0,{},0};
  for (const auto& [key,scratch] : scratches_) {
    if (&field != &scratch) continue;
    if (std::get<1>(key) != active_level_ || std::get<2>(key) != owner)
      throw std::invalid_argument("AMR SSA scratch names a foreign owner or level");
    const auto kind = std::get<0>(key) == ScratchKind::Rhs ? ProgramValueStorage::RhsScratch :
        std::get<0>(key) == ScratchKind::State ? ProgramValueStorage::StateScratch :
                                               ProgramValueStorage::ScalarScratch;
    return {kind,active_level_,owner,std::get<3>(key),std::get<4>(key),{},0};
  }
  const auto& histories = runtime_state().hist_;
  for (const auto& [key,ring] : histories.histories)
    for (std::size_t lag = 0; lag < ring.size(); ++lag)
      if (&field == &ring[lag]) {
        const auto decoded = decode_history_key_(key);
        if (!decoded || decoded->first != active_level_ || histories.owner.at(key) != owner)
          throw std::invalid_argument("AMR SSA history names a foreign owner or level");
        return {ProgramValueStorage::History,active_level_,owner,-1,0,
                decoded->second,static_cast<int>(lag)};
      }
  throw std::invalid_argument("AMR stage SSA has no exact context-owned slot");
}

public:
program_accepted_value_type capture_program_accepted_field_value(
    std::int64_t field_node, int program_block, int level) const {
  require_program_value_identity_();
  refresh_resources_();
  if (level < 0 || static_cast<std::size_t>(level) >= runtime_->hierarchy().num_levels())
    throw std::invalid_argument("accepted Field source names an absent AMR level");
  const auto planned = std::find_if(program_values_.state_->plan.fields.begin(),
      program_values_.state_->plan.fields.end(), [&](const auto& edge) {
        return edge.field_node == field_node && edge.program_block == program_block;
      });
  if (planned == program_values_.state_->plan.fields.end() ||
      !planned->accepted_only)
    throw std::invalid_argument("active AMR Field stage cannot be substituted by an accepted image");
  const int owner = sys_block(program_block);
  const auto& field = facade_->prepared_amr_block_state(owner,level);
  auto point = boundary_evaluation_point(0);
  point.level = level;
  point.tick = static_cast<std::int64_t>(facade_->macro_step());
  point.substep = 0;
  point.stage = 0;
  point.stage_fraction = {0,1};
  point.dt = 0.0;
  point.physical_time = facade_->time();
  return program_values_.accepted_(field_node,program_block,
      program_value_slot_type{ProgramValueStorage::State,level,owner,-1,0,{},0},
      field,resource_attempt(),std::move(point));
}

std::function<void()> require_program_accepted_field_value(
    const program_accepted_value_type& value, std::int64_t field_node,
    int program_block, int level, const field_type& input) const {
  require_program_value_identity_();
  refresh_resources_();
  const auto planned = std::find_if(program_values_.state_->plan.fields.begin(),
      program_values_.state_->plan.fields.end(), [&](const auto& edge) {
        return edge.field_node == field_node && edge.program_block == program_block;
      });
  if (planned == program_values_.state_->plan.fields.end() ||
      !planned->accepted_only)
    throw std::invalid_argument("active AMR Field stage requires its declared SSA source");
  if (&input != &facade_->prepared_amr_block_state(sys_block(program_block),level))
    throw std::invalid_argument("accepted Field source is not its exact owned hierarchy image");
  return program_values_.require_accepted_field_(value,field_node,program_block,level,input,resource_attempt());
}

private:
using named_field_request_type = runtime::system::FieldSolveRequest<Dim>;
struct StagedNamedFieldValues {
  runtime::multiblock::BoundaryEvaluationPoint point;
  std::string field;
  std::uint64_t epoch = 0, generation = 0, sequence = 0, attempt = 0;
  std::vector<typename named_field_request_type::Source> sources;
};
mutable std::map<std::int64_t, std::vector<StagedNamedFieldValues>> staged_named_field_values_;

StagedNamedFieldValues prepare_named_field_values_(
    const runtime::multiblock::BoundaryEvaluationPoint& point, std::int64_t field_node,
    std::string_view field, std::initializer_list<ProgramFieldValueOverride> overrides) const {
  require_boundary_point_(point,"AMR SSA Field stage");
  require_program_value_identity_();
  if (point.level != active_level_ || field_node < 0 || field.empty() ||
      point != boundary_evaluation_point(point.stage))
    throw std::invalid_argument("AMR SSA Field stage lost its exact consumer point");
  StagedNamedFieldValues result{point,std::string(field),runtime_->topology_epoch(),
      runtime_->materialization_generation(),mapping_continuation_ ? mapping_continuation_->internal_sequence : 0,
      active_subcycling_attempt_,{}};
  std::set<int> seen;
  for (const auto& item : overrides) {
    if (!item.state || item.source_ssa < 0 || !seen.insert(item.program_block).second)
      throw std::invalid_argument("AMR SSA Field stage requires unique successful SSA sources");
    const int owner = sys_block(item.program_block);
    require_same_field_contract_(*item.state,facade_->prepared_amr_block_state(owner,active_level_),
                                 "AMR SSA Field source");
    auto value = program_value(item.source_ssa,item.program_block,*item.state);
    auto validate = require_program_field_value(value,field_node,item.program_block,*item.state);
    result.sources.push_back({std::move(value),std::move(validate)});
  }
  for (const auto& edge : program_values_.state_->plan.fields) {
    if (edge.field_node != field_node) continue;
    if (seen.contains(edge.program_block)) continue;
    if (!edge.accepted_only)
      throw std::invalid_argument("AMR Field stage lacks its declared successful SSA source");
    auto value = capture_program_accepted_field_value(field_node,edge.program_block,active_level_);
    const auto& accepted = facade_->prepared_amr_block_state(sys_block(edge.program_block),active_level_);
    auto validate = require_program_accepted_field_value(value,field_node,edge.program_block,active_level_,accepted);
    result.sources.push_back({std::move(value),std::move(validate)});
  }
  if (result.sources.empty())
    throw std::invalid_argument("AMR Field stage has no installed source authority");
  return result;
}

public:
/// Each executing level retains its completed native values before the common hierarchy barrier.
void stage_program_field_values_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, std::int64_t field_node,
    std::string_view field, std::initializer_list<ProgramFieldValueOverride> overrides) const {
  const auto& lane = prepared_execution_lane();
  std::exception_ptr error;
  try {
    if (!mapping_continuation_ || !mapping_continuation_->executing)
      throw std::logic_error("AMR Field stage requires its executing hierarchy continuation");
    auto stage = prepare_named_field_values_(point,field_node,field,overrides);
    auto& levels = staged_named_field_values_[field_node];
    if (levels.size() != static_cast<std::size_t>(active_level_))
      throw std::logic_error("AMR Field stage is duplicate, missing, or out of hierarchy order");
    levels.push_back(std::move(stage));
  } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error,lane,"AMR Field source staging failed collectively");
}

[[nodiscard]] SolveOutcome solve_fields_from_program_values_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, std::int64_t field_node,
    std::string_view field, std::initializer_list<ProgramFieldValueOverride> overrides) const {
  refresh_resources_();
  const auto& lane = prepared_execution_lane();
  std::optional<named_field_request_type> request;
  std::exception_ptr error;
  try {
    require_boundary_point_(point,"AMR SSA Field solve");
    require_program_value_identity_();
    if (point != boundary_evaluation_point(point.stage))
      throw std::invalid_argument("AMR SSA Field solve lost its context-owned consumer point");
    auto found = staged_named_field_values_.find(field_node);
    std::vector<StagedNamedFieldValues> single;
    const std::vector<StagedNamedFieldValues>* levels = nullptr;
    if (found != staged_named_field_values_.end()) {
      if (active_level_ != 0 || !mapping_continuation_ || mapping_continuation_->executing ||
          mapping_continuation_->ports.front().kind != MappingPortKind::named_field_solve)
        throw std::logic_error("composite Field solve requires the unique root hierarchy callback");
      levels = &found->second;
    } else {
      if (nlev() != 1)
        throw std::invalid_argument("composite Field solve requires every exact level stage; accepted fallback is forbidden");
      single.push_back(prepare_named_field_values_(point,field_node,field,overrides));
      levels = &single;
    }
    if (levels->size() != static_cast<std::size_t>(nlev()))
      throw std::invalid_argument("composite Field solve lacks a live hierarchy source level");
    std::vector<typename named_field_request_type::Source> sources;
    for (std::size_t level = 0; level < levels->size(); ++level) {
      const auto& staged = levels->at(level);
      auto expected = point;
      expected.level = static_cast<int>(level);
      if (staged.point != expected || staged.field != field ||
          staged.epoch != runtime_->topology_epoch() ||
          staged.generation != runtime_->materialization_generation() ||
          staged.attempt != active_subcycling_attempt_ ||
          staged.sequence != (mapping_continuation_ ? mapping_continuation_->internal_sequence : 0))
        throw std::invalid_argument("composite Field stage lost clock, attempt, topology, or materialization authority");
      for (const auto& source : staged.sources) {
        source.validate();
        sources.push_back(source);
      }
    }
    request.emplace(named_field_request_type(std::string(field),point,std::move(sources)));
  } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error,lane,"AMR SSA Field request failed collectively");
  // Request owns the successful-write proofs until its candidate solve has completed.
  staged_named_field_values_.erase(field_node);
  return facade_->solve_fields_from_request_at_in_place_(*request);
}

private:
struct StagedFieldWarmStart {
  runtime::multiblock::BoundaryEvaluationPoint point;
  std::string slot, identity;
  int owner = -1;
  std::int64_t source_value = -1;
  int source_subslot = -1;
  const field_type* source = nullptr;
  std::shared_ptr<const field_type> image;
  PreparedResourceAttempt resource;
  std::uint64_t epoch = 0, generation = 0, sequence = 0, attempt = 0;
};
mutable std::map<std::int64_t, std::vector<StagedFieldWarmStart>> staged_field_warm_starts_;

public:
void stage_field_warm_start(const std::string& provider_slot, int program_block,
                            const field_type& potential, int evaluation_id,
                            const std::string& publication_identity) const {
  const auto& lane = prepared_execution_lane();
  std::exception_ptr error;
  try {
    refresh_resources_();
    const auto point = boundary_evaluation_point(evaluation_id);
    require_boundary_point_(point,"AMR field warm-start stage");
    if (!mapping_continuation_ || !mapping_continuation_->executing ||
        provider_slot.empty() || publication_identity.empty() || potential.ncomp() != 1)
      throw std::invalid_argument("AMR field warm start requires its exact scalar hierarchy stage");
    const int owner = sys_block(program_block);
    const auto found = std::find_if(scratches_.begin(),scratches_.end(),[&](const auto& item) {
      return &item.second == &potential;
    });
    if (found == scratches_.end() || std::get<0>(found->first) != ScratchKind::Scalar ||
        std::get<1>(found->first) != active_level_ || std::get<2>(found->first) != owner)
      throw std::invalid_argument("AMR warm start is not the exact owner/level scalar endpoint family");
    auto& levels = staged_field_warm_starts_[evaluation_id];
    if (levels.size() != static_cast<std::size_t>(active_level_))
      throw std::invalid_argument("AMR warm-start stage is duplicate, missing, or out of order");
    device_fence();
    levels.push_back({point,provider_slot,publication_identity,owner,
        std::get<3>(found->first),std::get<4>(found->first),&potential,
        std::make_shared<const field_type>(potential),resource_attempt(),runtime_->topology_epoch(),
        runtime_->materialization_generation(),mapping_continuation_->internal_sequence,
        active_subcycling_attempt_});
  } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error,lane,"AMR warm-start source staging failed collectively");
}

void publish_staged_field_warm_start(int evaluation_id) const {
  const auto& lane = prepared_execution_lane();
  std::exception_ptr error;
  std::vector<const field_type*> images;
  const StagedFieldWarmStart* root = nullptr;
  try {
    const auto found = staged_field_warm_starts_.find(evaluation_id);
    if (active_level_ != 0 || !mapping_continuation_ || mapping_continuation_->executing ||
        mapping_continuation_->ports.front().kind != MappingPortKind::field_warm_start ||
        found == staged_field_warm_starts_.end() || found->second.size() != static_cast<std::size_t>(nlev()))
      throw std::invalid_argument("AMR warm start requires one complete root hierarchy publication");
    root = &found->second.front();
    if (root->point != boundary_evaluation_point(evaluation_id))
      throw std::invalid_argument("AMR warm start lost its current root point");
    for (std::size_t level = 0; level < found->second.size(); ++level) {
      const auto& staged = found->second[level];
      auto expected = root->point;
      expected.level = static_cast<int>(level);
      const auto scalar = scratches_.find(ScratchKey{ScratchKind::Scalar,static_cast<int>(level),
          staged.owner,staged.source_value,staged.source_subslot});
      if (staged.point != expected || staged.slot != root->slot || staged.identity != root->identity ||
          staged.owner != root->owner || !staged.resource.visible() ||
          !staged.resource.same_attempt(resource_attempt()) ||
          staged.epoch != runtime_->topology_epoch() || staged.generation != runtime_->materialization_generation() ||
          staged.sequence != mapping_continuation_->internal_sequence ||
          staged.attempt != active_subcycling_attempt_ || scalar == scratches_.end() ||
          &scalar->second != staged.source || !staged.image)
        throw std::invalid_argument("AMR warm start lost scalar, clock, attempt, or hierarchy authority");
      images.push_back(staged.image.get());
    }
  } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error,lane,"AMR warm-start hierarchy admission failed collectively");
  facade_->publish_program_field_warm_start(root->point,root->slot,root->owner,images,root->identity);
  staged_field_warm_starts_.erase(evaluation_id);
}
