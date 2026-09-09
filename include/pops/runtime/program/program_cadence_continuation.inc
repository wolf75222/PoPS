// Included inside ProgramRuntimeState<Dim>. The ordinary step and suspended execution share this
// dispatcher, so preparation, cadence windows, balance accounting and clock publication have one owner.
struct CadenceContinuation {
  PreparedCadenceStep cadence;
  PreparedCadenceSubstep partition;
  double accepted_time = 0.0;
  double requested_dt = 0.0;
  int accepted_macro_step = 0;
  int substep = 0;
  bool numerical_started = false;
  bool executing = false;
  std::function<void()> next;
  std::string port_identity;
  bool target_port = false;
  std::vector<MultiFab<Dim>*> port_fields;
};
// Accepted-state copies deliberately omit transient callbacks and borrowed stage buffers.
struct EphemeralCadenceContinuation : std::optional<CadenceContinuation> {
  using std::optional<CadenceContinuation>::operator=;
  EphemeralCadenceContinuation() = default;
  EphemeralCadenceContinuation(const EphemeralCadenceContinuation&) noexcept {}
  EphemeralCadenceContinuation& operator=(const EphemeralCadenceContinuation&) noexcept {
    this->reset();
    return *this;
  }
};
EphemeralCadenceContinuation cadence_continuation_;

void suspend_program_map(std::string identity, bool target, std::vector<MultiFab<Dim>*> fields,
                         std::function<void()> next) {
  if (!cadence_continuation_ || !cadence_continuation_->executing ||
      !cadence_continuation_->port_identity.empty() || !next || identity.empty() ||
      fields.empty() ||
      std::any_of(fields.begin(), fields.end(), [](auto* field) { return field == nullptr; }))
    throw std::logic_error(
        "Program map suspension requires one executing region and qualified fields");
  auto& frame = *cadence_continuation_;
  frame.port_identity = std::move(identity);
  frame.target_port = target;
  frame.port_fields = std::move(fields);
  frame.next = std::move(next);
}

const std::vector<MultiFab<Dim>*>& program_map_fields(const std::string& identity,
                                                      bool target) const {
  if (!cadence_continuation_ || cadence_continuation_->executing ||
      cadence_continuation_->port_identity != identity ||
      cadence_continuation_->target_port != target || cadence_continuation_->port_fields.empty())
    throw std::logic_error("Program map read/write does not match its suspended qualified port");
  return cadence_continuation_->port_fields;
}

void release_program_map(const std::string& identity, bool target) {
  (void)program_map_fields(identity, target);
  cadence_continuation_->port_identity.clear();
  cadence_continuation_->port_fields.clear();
}

void cancel_cadence_continuation() noexcept {
  cadence_continuation_.reset();
  cadence_dispatch_active_ = false;
  end_balance_due_window();
}

std::string advance_cadence_region(double& physical_time_cursor, int& macro_step_cursor, double dt,
                                   const std::string& runtime) {
  if (!cadence_continuation_) {
    if (cadence_dispatch_active_ || !step_)
      throw std::logic_error(runtime + " Program region dispatch is reentrant or has no Program");
    CadenceContinuation candidate;
    candidate.accepted_time = physical_time_cursor;
    candidate.accepted_macro_step = macro_step_cursor;
    candidate.requested_dt = dt;
    candidate.cadence = prepare_cadence_step(physical_time_cursor, macro_step_cursor, dt, runtime);
    if (macro_step_cursor == std::numeric_limits<int>::max())
      throw std::overflow_error(runtime + " Program cadence macro-step counter overflow");
    if (candidate.cadence.due) {
      validate_cadence_partition(candidate.cadence, substeps_, runtime);
      if (macro_step_cursor < candidate.cadence.window_steps - 1)
        throw std::logic_error(runtime + " Program cadence window starts before macro-step zero");
      begin_balance_due_window(macro_step_cursor, runtime);
    }
    cadence_continuation_ = std::move(candidate);
    cadence_dispatch_active_ = true;
  }
  auto& frame = *cadence_continuation_;
  if (frame.executing || frame.requested_dt != dt || !frame.port_identity.empty())
    throw std::logic_error(runtime +
                           " Program resume requires the same dt and a consumed map port");
  try {
    while (frame.cadence.due && frame.substep < substeps_) {
      frame.executing = true;
      if (!frame.numerical_started) {
        frame.partition = prepare_cadence_substep(frame.cadence, frame.substep, substeps_, runtime);
        physical_time_cursor = frame.partition.start;
        macro_step_cursor = frame.accepted_macro_step - (frame.cadence.window_steps - 1);
        last_dt_ = static_cast<Real>(frame.partition.dt);
        frame.numerical_started = true;
        step_(frame.partition.dt);
      } else {
        if (!frame.next)
          throw std::logic_error(runtime + " suspended Program has no continuation");
        auto next = std::move(frame.next);
        next();
      }
      frame.executing = false;
      if (!frame.port_identity.empty())
        return frame.port_identity;
      if (frame.next)
        throw std::logic_error(runtime + " Program continuation has no qualified barrier");
      physical_time_cursor = frame.partition.end;
      ++frame.substep;
      frame.numerical_started = false;
    }
    const auto cadence = frame.cadence;
    const int accepted_macro_step = frame.accepted_macro_step;
    physical_time_cursor = frame.accepted_time;
    macro_step_cursor = accepted_macro_step;
    end_balance_due_window();
    commit_cadence_step(cadence, runtime);
    physical_time_cursor = cadence.window_end;
    complete_balance_step(cadence.due);
    macro_step_cursor = accepted_macro_step + 1;
    cancel_cadence_continuation();
    return {};
  } catch (...) {
    physical_time_cursor = frame.accepted_time;
    macro_step_cursor = frame.accepted_macro_step;
    cancel_cadence_continuation();
    throw;
  }
}

void dispatch_cadence_step(double& physical_time_cursor, int& macro_step_cursor, double dt,
                           const std::string& runtime) {
  if (cadence_continuation_ || cadence_dispatch_active_)
    throw std::logic_error(runtime + " Program cadence dispatch is non-reentrant");
  const double accepted_time = physical_time_cursor;
  const int accepted_step = macro_step_cursor;
  try {
    const std::string port =
        advance_cadence_region(physical_time_cursor, macro_step_cursor, dt, runtime);
    if (!port.empty())
      throw std::logic_error(runtime +
                             " Program map requires its resolved multi-layout coordinator");
  } catch (...) {
    physical_time_cursor = accepted_time;
    macro_step_cursor = accepted_step;
    cancel_cadence_continuation();
    throw;
  }
}
