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
  program_values_.require_accepted_edge_(field_node,program_block,
      "active AMR Field stage cannot be substituted by an accepted image");
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
  program_values_.require_accepted_edge_(field_node,program_block,
      "active AMR Field stage requires its declared SSA source");
  if (&input != &facade_->prepared_amr_block_state(sys_block(program_block),level))
    throw std::invalid_argument("accepted Field source is not its exact owned hierarchy image");
  return program_values_.require_accepted_field_(value,field_node,program_block,level,input,resource_attempt());
}

private:
using field_stages_type = PreparedProgramFieldStages<Dim>;
using named_field_request_type = typename field_stages_type::request_type;
mutable field_stages_type prepared_field_stages_;

typename field_stages_type::Stamp field_stage_stamp_(
    const runtime::multiblock::BoundaryEvaluationPoint& point) const {
  return {point,runtime_->topology_epoch(),runtime_->materialization_generation(),
      mapping_continuation_ ? mapping_continuation_->internal_sequence : 0,active_subcycling_attempt_};
}

typename field_stages_type::Sources prepare_named_field_values_(
    const runtime::multiblock::BoundaryEvaluationPoint& point, std::int64_t field_node,
    std::string_view field, std::initializer_list<ProgramFieldValueOverride> overrides) const {
  require_boundary_point_(point,"AMR SSA Field stage");
  require_program_value_identity_();
  if (point.level != active_level_ || field_node < 0 || field.empty() ||
      point != boundary_evaluation_point(point.stage))
    throw std::invalid_argument("AMR SSA Field stage lost its exact consumer point");
  return field_stages_type::prepare_sources(field_stage_stamp_(point),field_node,
      std::string(field),program_values_.state_->plan,overrides,
      [&](const auto& item) -> typename named_field_request_type::Source {
        const int owner = sys_block(item.program_block);
        require_same_field_contract_(*item.state,facade_->prepared_amr_block_state(owner,active_level_),
                                     "AMR SSA Field source");
        auto value = program_value(item.source_ssa,item.program_block,*item.state);
        auto validate = require_program_field_value(value,field_node,item.program_block,*item.state);
        return {std::move(value),std::move(validate)};
      }, [&](int block) -> typename named_field_request_type::Source {
        auto value = capture_program_accepted_field_value(field_node,block,active_level_);
        const auto& accepted = facade_->prepared_amr_block_state(sys_block(block),active_level_);
        auto validate = require_program_accepted_field_value(value,field_node,block,active_level_,accepted);
        return {std::move(value),std::move(validate)};
      });
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
    prepared_field_stages_.stage_sources(field_node,prepare_named_field_values_(point,field_node,field,overrides));
  } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error,lane,"AMR Field source staging failed collectively");
}

[[nodiscard]] SolveOutcome solve_fields_from_program_values_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, std::int64_t field_node,
    std::string_view field, std::initializer_list<ProgramFieldValueOverride> overrides) const {
  refresh_resources_();
  return prepared_field_stages_.solve_sources(field_node,prepared_execution_lane(),[&] {
    require_boundary_point_(point,"AMR SSA Field solve");
    require_program_value_identity_();
    if (point != boundary_evaluation_point(point.stage))
      throw std::invalid_argument("AMR SSA Field solve lost its context-owned consumer point");
    const bool root_callback = mapping_continuation_ && !mapping_continuation_->executing &&
        mapping_continuation_->ports.front().kind == MappingPortKind::named_field_solve;
    return prepared_field_stages_.prepare_request(field_stage_stamp_(point),field_node,
        std::string(field),nlev(),root_callback,
        [&] { return prepare_named_field_values_(point,field_node,field,overrides); },
        [](const auto& name,const auto& at,auto sources) {
          return named_field_request_type(name,at,std::move(sources)); });
  },[&](const auto& request) { return facade_->solve_fields_from_request_at_in_place_(request); });
}

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
    prepared_field_stages_.stage_warm_start(field_stage_stamp_(point),provider_slot,publication_identity,
        owner,std::get<3>(found->first),std::get<4>(found->first),potential,resource_attempt());
  } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error,lane,"AMR warm-start source staging failed collectively");
}

void publish_staged_field_warm_start(int evaluation_id) const {
  prepared_field_stages_.publish_warm_start(evaluation_id,prepared_execution_lane(),[&] {
    const bool root_callback = mapping_continuation_ && !mapping_continuation_->executing &&
        mapping_continuation_->ports.front().kind == MappingPortKind::field_warm_start;
    return prepared_field_stages_.prepare_warm_start(
        field_stage_stamp_(boundary_evaluation_point(evaluation_id)),evaluation_id,nlev(),root_callback,
        resource_attempt(),[&](int level,const auto& staged) {
          const auto scalar = scratches_.find(ScratchKey{ScratchKind::Scalar,level,
              staged.owner,staged.source_value,staged.source_subslot});
          return scalar != scratches_.end() && &scalar->second == staged.source;
        });
  },[&](const auto& publication) {
    const auto& root = *publication.root;
    facade_->publish_program_field_warm_start(root.stamp.point,root.slot,root.owner,publication.images,root.identity);
  });
}
