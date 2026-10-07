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
      (level == active_level_ && !planned->accepted_only))
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
      (level == active_level_ && !planned->accepted_only))
    throw std::invalid_argument("active AMR Field stage requires its declared SSA source");
  if (&input != &facade_->prepared_amr_block_state(sys_block(program_block),level))
    throw std::invalid_argument("accepted Field source is not its exact owned hierarchy image");
  return program_values_.require_accepted_field_(value,field_node,program_block,level,input,resource_attempt());
}

[[nodiscard]] SolveOutcome solve_fields_from_program_values_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, std::int64_t field_node,
    std::string_view field, std::initializer_list<ProgramFieldValueOverride> overrides) const {
  using Request = runtime::system::FieldSolveRequest<Dim>;
  refresh_resources_();
  const auto& lane = prepared_execution_lane();
  std::optional<Request> request;
  std::exception_ptr error;
  try {
    require_boundary_point_(point,"AMR SSA Field solve");
    require_program_value_identity_();
    if (point.level != active_level_ || field_node < 0 || field.empty())
      throw std::invalid_argument("AMR SSA Field solve has an invalid active point or identity");
    if (point != boundary_evaluation_point(point.stage))
      throw std::invalid_argument("AMR SSA Field solve lost its context-owned consumer point");
    std::vector<typename Request::Source> sources;
    std::vector<typename Request::LegacySource> legacy;
    std::set<int> seen;
    for (const auto& item : overrides) {
      if (!item.state || !seen.insert(item.program_block).second)
        throw std::invalid_argument("AMR SSA Field solve has a null or duplicate stage");
      const int owner = sys_block(item.program_block);
      require_same_field_contract_(*item.state,facade_->prepared_amr_block_state(owner,active_level_),
                                   "AMR SSA Field source");
      if (item.source_ssa >= 0) {
        auto value = program_value(item.source_ssa,item.program_block,*item.state);
        auto validate = require_program_field_value(value,field_node,item.program_block,*item.state);
        sources.push_back({std::move(value),std::move(validate)});
      } else {
        if (program_values_.state_->edges.contains({field_node,item.program_block}))
          throw std::invalid_argument("AMR V2 Field source cannot use the legacy source route");
        legacy.push_back({owner,active_level_,std::make_shared<field_type>(*item.state)});
      }
    }
    for (const auto& edge : program_values_.state_->plan.fields) {
      if (edge.field_node != field_node) continue;
      if (!seen.contains(edge.program_block) && !edge.accepted_only)
        throw std::invalid_argument("active AMR Field solve lacks its declared SSA stage override");
      for (std::size_t level = 0; level < runtime_->hierarchy().num_levels(); ++level) {
        if (static_cast<int>(level) == active_level_ && seen.contains(edge.program_block)) continue;
        auto value = capture_program_accepted_field_value(field_node,edge.program_block,
                                                         static_cast<int>(level));
        const auto& accepted = facade_->prepared_amr_block_state(sys_block(edge.program_block),
                                                                static_cast<int>(level));
        auto validate = require_program_accepted_field_value(value,field_node,edge.program_block,
                                                             static_cast<int>(level),accepted);
        sources.push_back({std::move(value),std::move(validate)});
      }
    }
    request = Request(std::string(field),point,std::move(sources),std::move(legacy));
  } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error,lane,"AMR SSA Field request failed collectively");
  return facade_->solve_fields_from_request_at_in_place_(*request);
}
