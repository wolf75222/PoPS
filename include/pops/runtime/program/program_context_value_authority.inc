// Shared public entry points. Storage admission is supplied by the exact ranked context.
public:
using program_value_type = typename ProgramValueAuthority<Dim>::Value;
using program_accepted_value_type = typename ProgramValueAuthority<Dim>::AcceptedValue;
using program_value_write_ticket_type = typename ProgramValueAuthority<Dim>::WriteTicket;
struct ProgramFieldValueOverride {
  int program_block = -1;
  const field_type* state = nullptr;
  std::int64_t source_ssa = -1;
};

void install_program_value_plan(ProgramValuePlan plan) const {
  const auto& lane = prepared_execution_lane();
  std::shared_ptr<typename ProgramValueAuthority<Dim>::State> candidate_state;
  std::exception_ptr error;
  std::string contract;
  try {
    ProgramValueAuthority<Dim> candidate;
    candidate.install_(plan);
    candidate_state = std::move(candidate.state_);
    if (program_values_.state_->installed && program_values_.state_->plan != plan)
      throw std::logic_error("Program value plan is immutable after installation");
    ExactContractBuilder proof;
    proof.text("pops.ProgramValuePlan@1").text(plan.program_identity);
    proof.scalar(plan.producers.size());
    for (const auto& producer : plan.producers) {
      proof.scalar(producer.ssa).scalar(producer.program_block).scalar(producer.storage)
          .scalar(producer.storage_id).scalar(producer.subslot).text(producer.history)
          .scalar(producer.lag).scalar(producer.inputs.size());
      for (auto input : producer.inputs) proof.scalar(input);
      proof.scalar(producer.input_branches.size());
      for (const auto& branch : producer.input_branches) {
        proof.scalar(branch.size());
        for (auto input : branch) proof.scalar(input);
      }
    }
    proof.scalar(plan.fields.size());
    for (const auto& edge : plan.fields)
      proof.scalar(edge.field_node).scalar(edge.program_block).scalar(edge.source).scalar(edge.accepted_only);
    contract = std::move(proof).release();
  } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error,lane,"Program SSA plan preparation failed collectively");
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"pops.ProgramValuePlan@1",contract}},lane))
    throw std::invalid_argument("Program SSA plan differs across ranks");
  if (!program_values_.state_->installed) {
    // Keep the issuer state address stable: an already-installed lifetime callback may hold its
    // weak identity. Every potentially throwing allocation happened in the detached candidate.
    std::swap(program_values_.state_->plan,candidate_state->plan);
    program_values_.state_->producers.swap(candidate_state->producers);
    program_values_.state_->edges.swap(candidate_state->edges);
    program_values_.state_->installed = true;
  }
}

program_value_write_ticket_type begin_program_value_write(
    std::int64_t ssa, int program_block, const field_type& output,
    std::initializer_list<program_value_type> inputs = {}) const {
  const auto& lane = prepared_execution_lane();
  std::optional<program_value_write_ticket_type> ticket;
  std::exception_ptr error;
  try {
    program_values_.drain_failed_();
    require_program_value_identity_();
    const auto slot = program_value_slot_(program_block, output);
    ticket = program_values_.begin_(ssa,program_block,slot,output,resource_attempt(),
        boundary_evaluation_point(0),std::vector<program_value_type>(inputs),false);
  } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error,lane,"Program SSA write admission failed collectively");
  return std::move(*ticket);
}

program_value_type complete_program_value_write(program_value_write_ticket_type&& ticket) const {
  // Successful launch does not establish successful device completion. The values keep allocation
  // owners even if this fence throws; an incomplete ticket is never consumable.
  const auto& lane = prepared_execution_lane();
  std::exception_ptr error;
  try {
    device_fence();
    program_values_.prepare_complete_(ticket);
  } catch (...) { error = std::current_exception(); }
  if (all_reduce_max(error ? 1L : 0L,lane) != 0) {
    program_values_.quarantine_(ticket);
    collectively_rethrow_exception(error,lane,"Program SSA producer completion failed collectively");
  }
  return program_values_.complete_(std::move(ticket));
}

program_value_type capture_program_value(std::int64_t ssa, int program_block,
                                        const field_type& input) const {
  const auto& lane = prepared_execution_lane();
  std::optional<program_value_write_ticket_type> ticket;
  std::exception_ptr error;
  try {
    program_values_.drain_failed_();
    require_program_value_identity_();
    ticket = program_values_.begin_(ssa,program_block,
        program_value_slot_(program_block,input),input,resource_attempt(),
        boundary_evaluation_point(0),{},true);
  } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error,lane,"Program SSA root capture failed collectively");
  return complete_program_value_write(std::move(*ticket));
}

program_value_type alias_program_value(std::int64_t ssa, const program_value_type& source) const {
  require_program_value_identity_();
  return program_values_.alias_(ssa,source);
}

program_value_type program_value(std::int64_t ssa, int program_block,
                                const field_type& input) const {
  require_program_value_identity_();
  (void)program_value_slot_(program_block,input);
  return program_values_.value_(ssa,program_block,program_value_level_(),input,resource_attempt());
}

std::function<void()> require_program_field_value(const program_value_type& value,
    std::int64_t field_node, int program_block, const field_type& input) const {
  require_program_value_identity_();
  (void)program_value_slot_(program_block,input);
  return program_values_.require_field_(value,field_node,program_block,
      program_value_level_(),input,resource_attempt());
}

std::function<void(PreparedResourceAction)> program_resource_lifetime_() const {
  auto resources = prepared_resources_.lifetime_callback();
  auto revoke = program_values_.rejection_callback_();
  auto drain = program_values_.drain_callback_();
  return [resources = std::move(resources),revoke = std::move(revoke),
          drain = std::move(drain)](PreparedResourceAction action) {
    if (action == PreparedResourceAction::reject) revoke();
    resources(action);
    if (action != PreparedResourceAction::reject) drain();
  };
}

private:
void drain_program_value_writes_() const {
  if (!program_values_.state_->installed) return;
  std::exception_ptr error;
  try { program_values_.drain_failed_(); } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error,prepared_execution_lane(),
                                 "Program SSA retained writer drain failed collectively");
}
void require_program_value_identity_() const {
  if (!program_values_.state_->installed ||
      program_values_.state_->plan.program_identity != program_value_identity_())
    throw std::logic_error("Program SSA authority differs from the installed compiled Program");
}
public:
