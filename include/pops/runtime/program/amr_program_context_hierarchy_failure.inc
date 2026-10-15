// Recoverable control remains inside the same unpublished hierarchy attempt.
public:
template <class Handler>
void push_hierarchy_failure_handler(std::int64_t operation, std::string_view graph,
                                    std::string_view dependency, Handler&& handler) const {
  if (!mapping_continuation_ || !mapping_continuation_->executing || operation < 0 ||
      graph != mapping_continuation_->program_hash || dependency.empty() || active_level_ < 0 ||
      static_cast<std::size_t>(active_level_) >= mapping_continuation_->failures.size())
    throw std::logic_error("hierarchy recovery scope has no exact executing authority");
  MappingFailureHandler prepared;
  prepared.operation = operation;
  ExactContractBuilder identity;
  identity.text("pops.hierarchy.recovery-scope@1")
      .text(graph)
      .scalar(operation)
      .bytes(dependency)
      .scalar(active_subcycling_attempt_)
      .scalar(mapping_continuation_->internal_sequence)
      .bytes(mapping_continuation_->hierarchy_authority);
  prepared.identity = std::move(identity).release();
  prepared.callback = std::forward<Handler>(handler);
  if (!prepared.callback)
    throw std::invalid_argument("empty hierarchy recovery handler");
  mapping_continuation_->failures[static_cast<std::size_t>(active_level_)].push_back(
      std::move(prepared));
}

void close_hierarchy_failure_handlers(std::int64_t operation) const {
  require_hierarchy_failure_handlers_(operation);
  for (auto& stack : mapping_continuation_->failures)
    stack.pop_back();
}

private:
void require_hierarchy_failure_handlers_(std::int64_t operation) const {
  if (!mapping_continuation_ || mapping_continuation_->failures.empty())
    throw std::logic_error("hierarchy recovery scope is absent");
  const auto& stacks = mapping_continuation_->failures;
  const auto& first = stacks.front();
  if (first.empty() || first.back().operation != operation)
    throw std::logic_error("hierarchy recovery scope was superseded");
  for (const auto& stack : stacks)
    if (stack.size() != first.size() || stack.back().operation != operation ||
        stack.back().identity != first.back().identity || !stack.back().callback)
      throw std::logic_error("AMR levels have different recovery scopes");
}

bool recover_mapping_failure_(std::exception_ptr failure) const {
  // Terminal exceptions keep their original disposition. A consumed native
  // rejection has already crossed the collective rejection protocol unchanged.
  try {
    std::rethrow_exception(failure);
  } catch (const StepAttemptRejected&) {
  } catch (...) {
    return false;
  }
  auto& frame = *mapping_continuation_;
  const long available = !frame.failures.empty() && !frame.failures.front().empty() ? 1L : 0L;
  const auto& lane = prepared_execution_lane();
  if (all_reduce_min(available, lane) != all_reduce_max(available, lane))
    throw std::logic_error("hierarchy recovery scope presence differs across ranks");
  if (available == 0)
    return false;
  const auto operation = frame.failures.front().back().operation;
  std::vector<MappingLevelPort> prepared;
  std::exception_ptr error;
  try {
    require_hierarchy_failure_handlers_(operation);
    prepared.resize(frame.ports.size());
    ExactContractBuilder contract;
    contract.text("pops.hierarchy.recovery-dispatch@1")
        .bytes(frame.failures.front().back().identity)
        .scalar(frame.internal_sequence);
    const auto identity = std::move(contract).release();
    for (std::size_t level = 0; level < prepared.size(); ++level) {
      auto& port = prepared[level];
      port.kind = MappingPortKind::control;
      port.identity = identity;
      port.once = [] {};
      port.next = [callback = frame.failures[level].back().callback, failure] {
        callback(failure);
      };
    }
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, prepared_execution_lane(),
                                 "hierarchy recovery preparation");
  // Agreement precedes either replacing pending callbacks or popping any scope.
  PreparedHierarchyBarrier::agree_ports(prepared, frame.fields, prepared_execution_lane());
  frame.ports.swap(prepared);
  for (auto& stack : frame.failures)
    stack.pop_back();
  frame.executing = false;
  return true;
}
