// Prepared AMR subcycling execution helpers; included inside the engine class.

template <class Callback>
void invoke_collectively_(Callback&& callback, std::string_view message) const {
  enum class ExceptionKind : long { None = 0, StepRejected = 1, Ordinary = 2 };
  ExceptionKind kind = ExceptionKind::None;
  bool pod_rejection = false;
  ::pops::runtime::program::ProgramStepRejectRecord pod_record{};
  std::string rejection_payload;
  std::exception_ptr local_error;
  try {
    callback();
  } catch (const ::pops::runtime::program::ProgramStepRejectSignal& rejected) {
    pod_record = rejected.record;
    pod_rejection = true;
    kind = ExceptionKind::StepRejected;
  } catch (const ::pops::runtime::program::StepAttemptRejected& rejected) {
    try {
      rejection_payload = encode_step_rejection_(rejected);
      kind = ExceptionKind::StepRejected;
    } catch (...) {
      kind = ExceptionKind::Ordinary;
      local_error = std::current_exception();
    }
  } catch (...) {
    kind = ExceptionKind::Ordinary;
    local_error = std::current_exception();
  }
  try {
    ::pops::device_fence(hot_fence_label_);
  } catch (...) {
    kind = ExceptionKind::Ordinary;
    local_error = std::current_exception();
  }

  const auto communicator = hierarchy_->lane().communicator();
  const long ordinary = kind == ExceptionKind::Ordinary ? 1L : 0L;
  const long rejected = kind == ExceptionKind::StepRejected ? 1L : 0L;
  if (all_reduce_max(ordinary, communicator) != 0) {
    if (hierarchy_->lane().size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error(std::string(message));
  }
  const long local_pod_rejection = pod_rejection ? 1L : 0L;
  const long any_pod_rejection = all_reduce_max(local_pod_rejection, communicator);
  const long any_legacy_rejection =
      all_reduce_max(kind == ExceptionKind::StepRejected && !pod_rejection ? 1L : 0L, communicator);
  if (any_pod_rejection != 0 && any_legacy_rejection != 0)
    throw std::runtime_error("collective step rejection mixed typed and legacy transports");
  if (any_pod_rejection != 0) {
    const auto bytes = [](const ::pops::runtime::program::ProgramStepRejectRecord& value) {
      return std::string_view(reinterpret_cast<const char*>(&value), sizeof(value));
    };
    ::pops::runtime::program::ProgramStepRejectRecord selected{};
    if (all_reduce_min(local_pod_rejection, communicator) != 0) {
      if (!all_ranks_agree_exact_ordered_byte_pairs({{"step-rejection-v5", bytes(pod_record)}},
                                                    communicator))
        throw std::runtime_error("collective typed step rejection differs between ranks");
      selected = pod_record;
    } else {
      const long local_root = pod_rejection ? static_cast<long>(hierarchy_->lane().rank())
                                            : static_cast<long>(hierarchy_->lane().size());
      const long root = all_reduce_min(local_root, communicator);
      if (root < 0 || root >= static_cast<long>(hierarchy_->lane().size()))
        throw std::runtime_error("collective typed step rejection lost its authority");
      if (pod_rejection)
        selected = pod_record;
      broadcast_bytes_inplace(reinterpret_cast<char*>(&selected), sizeof(selected),
                              static_cast<int>(root), communicator);
      const long mismatch =
          pod_rejection && std::memcmp(&pod_record, &selected, sizeof(selected)) != 0 ? 1L : 0L;
      if (all_reduce_max(mismatch, communicator) != 0)
        throw std::runtime_error("collective typed step rejection differs between rejecting ranks");
    }
    if (selected.status > static_cast<std::uint32_t>(SolveStatus::kSafeguardFailure) ||
        selected.reserved != 0 ||
        selected.disposition >
            static_cast<std::uint32_t>(::pops::runtime::program::StepAttemptDisposition::kReject))
      throw std::runtime_error("collective typed step rejection has invalid fields");
    throw ::pops::runtime::program::ProgramStepRejectSignal(selected);
  }
  if (all_reduce_max(rejected, communicator) == 0)
    return;

  // When every rank rejected, authenticate the complete typed envelope exactly. When only a
  // subset rejected, the first rejecting rank is the deterministic control authority. Its byte
  // envelope is broadcast with reductions on the lane communicator so every participant follows
  // the same collective sequence without requiring ownership of the communicator observer.
  std::string selected_payload;
  if (all_reduce_min(rejected, communicator) != 0) {
    if (!all_ranks_agree_exact_ordered_byte_pairs({{"step-rejection", rejection_payload}},
                                                  communicator))
      throw std::runtime_error("collective step rejection fields differ between ranks");
    selected_payload = std::move(rejection_payload);
  } else {
    const long local_root = rejected != 0 ? static_cast<long>(hierarchy_->lane().rank())
                                          : static_cast<long>(hierarchy_->lane().size());
    const long root = all_reduce_min(local_root, communicator);
    if (root < 0 || root >= static_cast<long>(hierarchy_->lane().size()))
      throw std::runtime_error("collective step rejection lost its typed envelope");

    const bool authoritative = hierarchy_->lane().rank() == root;
    const long invalid_length =
        authoritative && rejection_payload.size() >
                             static_cast<std::size_t>(std::numeric_limits<long>::max())
            ? 1L
            : 0L;
    if (all_reduce_max(invalid_length, communicator) != 0)
      throw std::length_error("collective step rejection envelope exceeds long capacity");
    const long encoded_length = all_reduce_max(
        authoritative ? static_cast<long>(rejection_payload.size()) : 0L, communicator);
    if (encoded_length <= 0)
      throw std::runtime_error("collective step rejection envelope is empty");

    long allocation_failed = 0;
    try {
      if (authoritative)
        selected_payload = rejection_payload;
      selected_payload.resize(static_cast<std::size_t>(encoded_length));
    } catch (...) {
      allocation_failed = 1;
    }
    if (all_reduce_max(allocation_failed, communicator) != 0)
      throw std::bad_alloc();
    broadcast_bytes_inplace(selected_payload.data(), selected_payload.size(),
                            static_cast<int>(root), communicator);
    const long typed_envelope_mismatch =
        rejected != 0 && rejection_payload != selected_payload ? 1L : 0L;
    if (all_reduce_max(typed_envelope_mismatch, communicator) != 0)
      throw std::runtime_error("collective step rejection fields differ between rejecting ranks");
  }
  const StepRejectionEnvelope envelope = decode_step_rejection_(selected_payload);
  throw ::pops::runtime::program::StepAttemptRejected(
      envelope.status, envelope.disposition, envelope.reason_code, envelope.phase, envelope.detail);
}

struct StepRejectionEnvelope {
  SolveStatus status = SolveStatus::kInvalidInput;
  ::pops::runtime::program::StepAttemptDisposition disposition =
      ::pops::runtime::program::StepAttemptDisposition::kReject;
  std::uint32_t reason_code = 0;
  std::string phase;
  std::string detail;
};

static void append_u64_(std::string& bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<char>((value >> shift) & 0xffu));
}

static std::uint64_t read_u64_(std::string_view bytes, std::size_t& cursor) {
  if (cursor > bytes.size() || bytes.size() - cursor < 8)
    throw std::runtime_error("collective step rejection envelope is truncated");
  std::uint64_t value = 0;
  for (int byte = 0; byte < 8; ++byte)
    value = (value << 8u) | static_cast<unsigned char>(bytes[cursor++]);
  return value;
}

static void append_text_(std::string& bytes, std::string_view value) {
  append_u64_(bytes, static_cast<std::uint64_t>(value.size()));
  bytes.append(value.data(), value.size());
}

static std::string read_text_(std::string_view bytes, std::size_t& cursor) {
  const std::uint64_t encoded_size = read_u64_(bytes, cursor);
  if (encoded_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error("collective step rejection text exceeds size_t");
  const std::size_t size = static_cast<std::size_t>(encoded_size);
  if (cursor > bytes.size() || size > bytes.size() - cursor)
    throw std::runtime_error("collective step rejection text is truncated");
  std::string value(bytes.substr(cursor, size));
  cursor += size;
  return value;
}

static std::string encode_step_rejection_(
    const ::pops::runtime::program::StepAttemptRejected& rejected) {
  std::string bytes("pops.step-rejection.v1");
  append_u64_(bytes, static_cast<std::uint64_t>(rejected.status()));
  append_u64_(bytes, static_cast<std::uint64_t>(rejected.disposition()));
  append_u64_(bytes, rejected.reason_code());
  append_text_(bytes, rejected.phase());
  append_text_(bytes, rejected.detail());
  return bytes;
}

static StepRejectionEnvelope decode_step_rejection_(std::string_view bytes) {
  constexpr std::string_view prefix = "pops.step-rejection.v1";
  if (!bytes.starts_with(prefix))
    throw std::runtime_error("collective step rejection envelope has another schema");
  std::size_t cursor = prefix.size();
  const std::uint64_t status = read_u64_(bytes, cursor);
  const std::uint64_t disposition = read_u64_(bytes, cursor);
  const std::uint64_t reason_code = read_u64_(bytes, cursor);
  if (status > static_cast<std::uint64_t>(SolveStatus::kSafeguardFailure) ||
      disposition >
          static_cast<std::uint64_t>(::pops::runtime::program::StepAttemptDisposition::kReject) ||
      reason_code > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("collective step rejection envelope has invalid enum fields");
  StepRejectionEnvelope result;
  result.status = static_cast<SolveStatus>(status);
  result.disposition = static_cast<::pops::runtime::program::StepAttemptDisposition>(disposition);
  result.reason_code = static_cast<std::uint32_t>(reason_code);
  result.phase = read_text_(bytes, cursor);
  result.detail = read_text_(bytes, cursor);
  if (cursor != bytes.size())
    throw std::runtime_error("collective step rejection envelope has trailing bytes");
  return result;
}

void require_mutable_state_image_(const MutableStateImage& image) const {
  if (image.exact_contract != exact_contract_ ||
      image.accepted_histories.size() != accepted_histories_.size() ||
      image.accepted_clocks.size() != accepted_clocks_.size() ||
      image.accepted_ledgers.size() != accepted_ledgers_.size())
    throw std::logic_error("AMR subcycling rollback image differs from its prepared authority");
  for (std::size_t block = 0; block < accepted_histories_.size(); ++block) {
    const auto& target_history = accepted_histories_[block];
    const auto& source_history = image.accepted_histories[block];
    const auto& target_clock = accepted_clocks_[block];
    const auto& source_clock = image.accepted_clocks[block];
    const auto& target_ledgers = accepted_ledgers_[block];
    const auto& source_ledgers = image.accepted_ledgers[block];
    if (target_history.size() != source_history.size() ||
        target_clock.size() != source_clock.size() ||
        target_ledgers.size() != source_ledgers.size())
      throw std::logic_error("AMR subcycling rollback image changed its block shape");
    for (std::size_t level = 0; level < target_history.size(); ++level) {
      if (target_history[level].has_value() != source_history[level].has_value() ||
          target_clock[level].has_value() != source_clock[level].has_value())
        throw std::logic_error("AMR subcycling rollback image changed its level presence");
      if (target_history[level] &&
          (!same_field_shape_(target_history[level]->older, source_history[level]->older) ||
           !same_field_shape_(target_history[level]->newer, source_history[level]->newer)))
        throw std::logic_error("AMR subcycling rollback image changed its history field shape");
    }
    for (std::size_t parent = 0; parent < target_ledgers.size(); ++parent) {
      if (target_ledgers[parent].size() != source_ledgers[parent].size())
        throw std::logic_error("AMR subcycling rollback image changed its ledger shape");
      for (std::size_t invocation = 0; invocation < target_ledgers[parent].size(); ++invocation)
        target_ledgers[parent][invocation].require_preallocated_copy_from(
            source_ledgers[parent][invocation]);
    }
  }
}

void copy_history_matrix_preallocated_(
    std::vector<std::vector<std::optional<AcceptedHistory>>>& destination,
    const std::vector<std::vector<std::optional<AcceptedHistory>>>& source) const {
  for (std::size_t block = 0; block < source.size(); ++block)
    for (std::size_t level = 0; level < source[block].size(); ++level)
      if (source[block][level]) {
        auto& target = *destination[block][level];
        const auto& input = *source[block][level];
        copy_field_(input.older, target.older);
        copy_field_(input.newer, target.newer);
        target.window = input.window;
      }
}

static void copy_ledger_matrix_preallocated_(
    std::vector<std::vector<std::vector<ledger_type>>>& destination,
    const std::vector<std::vector<std::vector<ledger_type>>>& source) {
  for (std::size_t block = 0; block < source.size(); ++block)
    for (std::size_t parent = 0; parent < source[block].size(); ++parent)
      for (std::size_t invocation = 0; invocation < source[block][parent].size(); ++invocation)
        destination[block][parent][invocation].copy_from_preallocated(
            source[block][parent][invocation]);
}

static void require_ledger_matrix_copy_(
    const std::vector<std::vector<std::vector<ledger_type>>>& destination,
    const std::vector<std::vector<std::vector<ledger_type>>>& source) {
  for (std::size_t block = 0; block < source.size(); ++block)
    for (std::size_t parent = 0; parent < source[block].size(); ++parent)
      for (std::size_t invocation = 0; invocation < source[block][parent].size(); ++invocation)
        destination[block][parent][invocation].require_preallocated_copy_from(
            source[block][parent][invocation]);
}

bool uses_resident_ledger_slots_() const noexcept {
  for (const auto& block : candidate_ledgers_)
    for (const auto& parent : block)
      for (const ledger_type& ledger : parent)
        if (!ledger.resident_slots_bound())
          return false;
  return true;
}

void require_live_() const {
  std::exception_ptr local_error;
  try {
    if (hierarchy_ == nullptr || runtime_ == nullptr ||
        runtime_ != std::addressof(hierarchy_->topology_runtime()) ||
        hierarchy_->collective_contract() != hierarchy_contract_ ||
        hierarchy_->block_count() != accepted_ledgers_.size() ||
        hierarchy_->level_count() != relations_.size() + 1 ||
        runtime_->topology_epoch() != topology_epoch_ ||
        runtime_->materialization_generation() != materialization_generation_)
      throw std::invalid_argument("prepared multi-block AMR subcycling engine is stale");
    spatial_plan_.require_live(hierarchy_->topology_runtime());
  } catch (...) {
    local_error = std::current_exception();
  }
  collectively_rethrow_(*hierarchy_, local_error,
                        "prepared multi-block AMR subcycling liveness failed collectively");
}

static bool same_field_shape_(const field_type& source, const field_type& destination) {
  return source.layout() == destination.layout() &&
         source.distribution() == destination.distribution() &&
         source.local_rank() == destination.local_rank() && source.ncomp() == destination.ncomp() &&
         source.ghosts() == destination.ghosts() &&
         source.local_global_indices() == destination.local_global_indices() &&
         source.local_size() == destination.local_size();
}

void copy_field_(const field_type& source, field_type& destination) const {
  if (!same_field_shape_(source, destination))
    throw std::invalid_argument("multi-block AMR resident workspace shape changed");
  if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible) {
    // Kokkos::deep_copy constructs a profiling label on each dispatch.  For resident host
    // storage that allocation is both avoidable and inside the accepted hot path.  Synchronize
    // prior kernels once, then copy directly between the already allocated views.  This applies
    // to every host-accessible backend: OpenMP also exposes the same resident storage directly.
    ::pops::device_fence(hot_fence_label_);
    for (std::size_t local = 0; local < source.local_size(); ++local) {
      const auto& source_storage = source.fab(local).storage();
      const auto& destination_storage = destination.fab(local).storage();
      std::copy_n(source_storage.data(), source_storage.extent(0), destination_storage.data());
    }
  } else {
    for (std::size_t local = 0; local < source.local_size(); ++local)
      Kokkos::deep_copy(destination.fab(local).storage(), source.fab(local).storage());
  }
}

void reset_attempt_workspace_() {
  std::fill(ledger_cursors_.begin(), ledger_cursors_.end(), std::size_t{0});
  for (std::size_t block = 0; block < candidate_ledgers_.size(); ++block)
    for (std::size_t parent = 0; parent < candidate_ledgers_[block].size(); ++parent)
      for (ledger_type& ledger : candidate_ledgers_[block][parent]) {
        while (ledger.in_transaction())
          ledger.rollback();
        ledger.clear();
      }
}

void partition_into_workspace_(std::size_t parent_level, const ::pops::amr::ClockWindow& parent) {
  const relation_type& relation = relations_.at(parent_level);
  if (parent.begin.level != relation.parent_level() ||
      parent.end.level != relation.parent_level() || !(parent.begin.phase < parent.end.phase) ||
      !(parent.begin.physical_time < parent.end.physical_time))
    throw std::runtime_error("AMR parent clock window does not match its relation");
  const auto ratio = relation.temporal_ratio();
  if (!ratio.integral() &&
      relation.remainder_policy() == ::pops::amr::RemainderPolicy::IntegralOnly)
    throw std::runtime_error(
        "non-integral AMR temporal relation requires an explicit remainder policy");
  const std::size_t count = child_substeps_.at(parent_level).size();
  const std::int64_t full = ratio.numerator / ratio.denominator;
  if (full < 0 || static_cast<std::size_t>(full + (ratio.integral() ? 0 : 1)) != count)
    throw std::logic_error("multi-block AMR resident child workspace lost its prepared shape");
  const auto span = parent.end.phase - parent.begin.phase;
  const auto nominal = span / ratio;
  const double span_time = parent.end.physical_time - parent.begin.physical_time;
  const auto physical = [&](const ::pops::amr::Rational& phase) {
    return parent.begin.physical_time + span_time * ((phase - parent.begin.phase) / span).value();
  };
  ::pops::amr::Rational cursor = parent.begin.phase;
  for (std::int64_t child = 0; child < full; ++child) {
    const auto next = cursor + nominal;
    child_substeps_[parent_level][static_cast<std::size_t>(child)] = {
        {{relation.child_level(), parent.begin.macro_step, cursor, physical(cursor)},
         {relation.child_level(), parent.begin.macro_step, next, physical(next)}},
        false};
    cursor = next;
  }
  if (cursor < parent.end.phase)
    child_substeps_[parent_level].back() = {
        {{relation.child_level(), parent.begin.macro_step, cursor, physical(cursor)},
         {relation.child_level(), parent.begin.macro_step, parent.end.phase,
          physical(parent.end.phase)}},
        true};
}

void stage_parent_(std::size_t parent_level, const ::pops::amr::ClockWindow& parent_window,
                   const ::pops::amr::ClockStamp& target) {
  std::exception_ptr local_error;
  try {
    for (std::size_t block = 0; block < hierarchy_->block_count(); ++block) {
      auto& qualified = temporal_states_[block][parent_level];
      qualified.older.clock = parent_window.begin;
      qualified.newer.clock = parent_window.end;
      qualified.target.clock = target;
      qualified.target.clock.level = static_cast<int>(parent_level);
      for (std::size_t local = 0; local < staged_[block][parent_level].local_size(); ++local) {
        const auto prepared = ::pops::numerics::time::amr::prepare_linear_time_interpolation(
            hierarchy_->topology_runtime(), parent_level,
            std::as_const(older_[block][parent_level].fab(local)).view(),
            std::as_const(candidates_[block][parent_level].fab(local)).view(),
            staged_[block][parent_level].fab(local).view(), staged_[block][parent_level].box(local),
            qualified.older, qualified.newer, qualified.target,
            {0, 0, 0, staged_[block][parent_level].ncomp()});
        execute_prepared_transfer(prepared);
      }
    }
    ::pops::device_fence(hot_fence_label_);
  } catch (...) {
    local_error = std::current_exception();
  }
  collectively_rethrow_(*hierarchy_, local_error,
                        "multi-block AMR parent-time interpolation failed collectively");
}

template <class Advance, class Reflux>
void advance_level_recursive_(std::size_t level, const ::pops::amr::ClockWindow& window,
                              int substep, const std::vector<const field_type*>& staged_parent,
                              const std::vector<ledger_type*>& incoming_flux, std::uint64_t attempt,
                              Advance& advance_level, Reflux& reflux) {
  for (std::size_t block = 0; block < hierarchy_->block_count(); ++block)
    copy_field_(candidates_[block][level], older_[block][level]);

  const std::size_t invocation =
      level < relations_.size() ? ledger_cursors_[level]++ : std::size_t{0};
  if (level < relations_.size()) {
    for (std::size_t block = 0; block < hierarchy_->block_count(); ++block) {
      ledger_type& ledger = candidate_ledgers_[block][level].at(invocation);
      ledger.begin(attempt);
      outgoing_packs_[level][block] = &ledger;
    }
  }

  auto& group = level_groups_[level];
  for (std::size_t block = 0; block < hierarchy_->block_count(); ++block) {
    LevelAdvanceContext& context = group[block];
    context.substep = substep;
    context.attempt = attempt;
    context.window = window;
    context.staged_parent = level == 0 ? nullptr : staged_parent.at(block);
    context.incoming_flux = level == 0 ? nullptr : incoming_flux.at(block);
    context.outgoing_flux = level == relations_.size() ? nullptr : outgoing_packs_[level][block];
  }

  invoke_collectively_([&] { advance_level(LevelAdvanceGroup(group)); },
                       "multi-block AMR level-group callback failed collectively");
  for (std::size_t block = 0; block < hierarchy_->block_count(); ++block) {
    AcceptedHistory& history = *candidate_histories_[block][level];
    copy_field_(older_[block][level], history.older);
    copy_field_(candidates_[block][level], history.newer);
    history.window = window;
    candidate_clocks_[block][level] = window.end;
  }

  if (level == relations_.size())
    return;
  std::exception_ptr partition_error;
  try {
    partition_into_workspace_(level, window);
  } catch (...) {
    partition_error = std::current_exception();
  }
  collectively_rethrow_(*hierarchy_, partition_error,
                        "multi-block AMR temporal partition failed collectively");

  for (std::size_t child = 0; child < child_substeps_[level].size(); ++child) {
    const auto& child_window = child_substeps_[level][child].window;
    stage_parent_(level, window, child_window.begin);
    advance_level_recursive_(level + 1, child_window, static_cast<int>(child), staged_packs_[level],
                             outgoing_packs_[level], attempt, advance_level, reflux);
  }

  // The recursive child has already synchronized its own descendants, so this is finest-first.
  for (std::size_t block = 0; block < hierarchy_->block_count(); ++block)
    invoke_collectively_([&] { outgoing_packs_[level][block]->commit(); },
                         "multi-block AMR flux-ledger commit failed collectively");
  const auto ratio =
      hierarchy_->topology_runtime().hierarchy().layout(level + 1).ratio_from_parent();
  const ::pops::amr::reflux::FaceRefinementMapping<Dim> mapping{
      hierarchy_->topology_runtime().hierarchy().layout(level).domain().lo,
      hierarchy_->topology_runtime().hierarchy().layout(level + 1).domain().lo};
  for (std::size_t block = 0; block < hierarchy_->block_count(); ++block) {
    RefluxContext context{block,
                          hierarchy_->block_identity(block),
                          level,
                          attempt,
                          window,
                          candidates_[block][level],
                          candidates_[block][level + 1],
                          *outgoing_packs_[level][block],
                          ratio,
                          mapping};
    invoke_collectively_([&] { reflux(context); },
                         "multi-block AMR reflux callback failed collectively");
    average_down_[level][block]->execute(hierarchy_->topology_runtime(), level + 1,
                                         std::as_const(candidates_[block][level + 1]),
                                         candidates_[block][level], hierarchy_->lane());
    copy_field_(candidates_[block][level], candidate_histories_[block][level]->newer);
  }
}

hierarchy_type* hierarchy_ = nullptr;
runtime_type* runtime_ = nullptr;
std::uint64_t topology_epoch_ = std::numeric_limits<std::uint64_t>::max();
std::uint64_t materialization_generation_ = std::numeric_limits<std::uint64_t>::max();
std::string hierarchy_contract_;
std::vector<relation_type> relations_;
PreparedAmrSubcyclePlan<Dim, MemorySpace> spatial_plan_;
::pops::amr::reflux::FaceFluxLedgerBudget flux_budget_{};
typename hierarchy_type::ProgramBlockMap program_map_;
std::string exact_contract_;
std::string hot_fence_label_ = "pops.amr-program.subcycling.hot-fence";
HistoryMatrix accepted_histories_;
ClockMatrix accepted_clocks_;
LedgerMatrix accepted_ledgers_;
CandidateMatrix candidates_;
CandidateMatrix older_;
CandidateMatrix staged_;
HistoryMatrix candidate_histories_;
ClockMatrix candidate_clocks_;
LedgerMatrix candidate_ledgers_;
TemporalStateMatrix temporal_states_;
std::vector<std::string> block_identities_;
std::vector<std::vector<field_type*>> level_packs_;
std::vector<std::vector<LevelAdvanceContext>> level_groups_;
std::vector<std::vector<const field_type*>> staged_packs_;
std::vector<std::vector<ledger_type*>> outgoing_packs_;
std::vector<std::vector<std::unique_ptr<average_down_type>>> average_down_;
std::vector<std::vector<::pops::amr::ChildSubstep>> child_substeps_;
std::vector<std::size_t> ledger_cursors_;
std::vector<std::size_t> ledger_invocations_;
const std::vector<const field_type*> empty_field_pointers_{};
const std::vector<ledger_type*> empty_ledger_pointers_{};
const std::optional<::pops::amr::ClockStamp> empty_clock_{};
const std::optional<AcceptedHistory> empty_history_{};
const std::vector<ledger_type> empty_ledgers_{};
std::uint64_t next_attempt_ = 0;
std::uint64_t last_accepted_attempt_ = 0;
std::vector<std::vector<field_type>>* attempt_candidates_ = nullptr;
