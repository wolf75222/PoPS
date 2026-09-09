// Closed internal barrier protocol shared by every level of one persistent AMR attempt.
public:
enum class HierarchyBarrierKind : std::uint8_t { linear_solve = 1, field_publication = 2 };

template <class Once, class Next>
void suspend_hierarchy_barrier(HierarchyBarrierKind kind, std::int64_t operation_id,
                               std::string_view compiled_program_hash,
                               std::string_view exact_dependency_contract, std::int64_t numerator,
                               std::int64_t denominator, Once&& callback_once,
                               Next&& callback_next) const {
  if (!mapping_continuation_ || !mapping_continuation_->executing || active_level_ < 0 ||
      static_cast<std::size_t>(active_level_) >= mapping_continuation_->ports.size())
    throw std::logic_error("AMR hierarchy barrier requires an executing level region");
  if (kind != HierarchyBarrierKind::linear_solve && kind != HierarchyBarrierKind::field_publication)
    throw std::invalid_argument("AMR hierarchy barrier kind is unknown");
  if (operation_id < 0 || compiled_program_hash.empty() ||
      compiled_program_hash != mapping_continuation_->program_hash ||
      exact_dependency_contract.empty())
    throw std::invalid_argument(
        "AMR hierarchy barrier lacks its installed Program/dependency authority");
  if (denominator <= 0 || numerator < 0 || numerator > denominator)
    throw std::invalid_argument("AMR hierarchy barrier fraction lies outside its step");
  const ::pops::amr::Rational fraction(numerator, denominator);
  if (fraction.numerator != numerator || fraction.denominator != denominator ||
      stage_time_ != fraction)
    throw std::invalid_argument(
        "AMR hierarchy barrier fraction differs from its exact active stage");
  auto& frame = *mapping_continuation_;
  auto& port = frame.ports[static_cast<std::size_t>(active_level_)];
  if (port.kind != MappingPortKind::complete || port.next || port.once || port.field)
    throw std::logic_error("AMR level reached multiple barriers in one region");
  if (frame.internal_sequence == std::numeric_limits<std::uint64_t>::max() ||
      !multiblock_subcycling_ || !multiblock_subcycling_->has_synchronized_groups() ||
      active_subcycling_attempt_ != multiblock_subcycling_->synchronized_attempt())
    throw std::logic_error("AMR hierarchy barrier lost its exact synchronized attempt");

  std::function<void()> once(std::forward<Once>(callback_once));
  std::function<void()> next(std::forward<Next>(callback_next));
  if (!once || !next)
    throw std::invalid_argument(
        "AMR hierarchy barrier requires its unique callback and continuation");
  const auto root = multiblock_subcycling_->synchronized_level_group(0);
  if (root.empty())
    throw std::logic_error("AMR hierarchy barrier has no root clock authority");
  ExactContractBuilder contract;
  contract.text("pops.amr-hierarchy-barrier.v1")
      .scalar(static_cast<std::uint8_t>(kind))
      .scalar(operation_id)
      .text(compiled_program_hash)
      .bytes(exact_dependency_contract)
      .scalar(numerator)
      .scalar(denominator)
      .scalar(active_subcycling_attempt_)
      .scalar(frame.internal_sequence)
      .bytes(frame.hierarchy_authority);
  // Every level names the common root window; a level ordinal is not a different operation.
  for (const auto& clock : {root.front().window.begin, root.front().window.end})
    contract.scalar(std::int32_t{clock.level})
        .scalar(clock.macro_step)
        .scalar(clock.phase.numerator)
        .scalar(clock.phase.denominator)
        .scalar(clock.physical_time);
  port.identity = std::move(contract).release();
  port.kind = kind == HierarchyBarrierKind::linear_solve ? MappingPortKind::linear_solve
                                                         : MappingPortKind::field_publication;
  port.once = std::move(once);
  port.next = std::move(next);
}

private:
enum class MappingPortKind : std::uint8_t { complete, map, linear_solve, field_publication };

void prepare_mapping_barrier_authority_() const {
  // This facade getter participates in MPI even when prepared. Call it before any local
  // allocation/validation, while all ranks are still in the common preparation sequence.
  const auto& hierarchy = facade_->prepared_amr_multiblock_hierarchy_();
  std::exception_ptr error;
  try {
    auto& frame = *mapping_continuation_;
    frame.program_hash = facade_->installed_program_hash();
    ExactContractBuilder authority;
    authority.bytes(hierarchy.collective_contract())
        .scalar(hierarchy.topology_runtime().topology_epoch())
        .scalar(hierarchy.topology_runtime().materialization_generation());
    frame.hierarchy_authority = std::move(authority).release();
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, prepared_execution_lane(),
                                 "AMR mapping barrier authority preparation failed collectively");
}

void agree_continuation_ports_() const {
  auto& frame = *mapping_continuation_;
  const auto& first = frame.ports.front();
  const bool map = first.kind == MappingPortKind::map;
  const bool internal = first.kind == MappingPortKind::linear_solve ||
                        first.kind == MappingPortKind::field_publication;
  std::exception_ptr error;
  try {
    for (std::size_t level = 0; level < frame.ports.size(); ++level) {
      const auto& port = frame.ports[level];
      if (port.kind != first.kind || port.identity != first.identity ||
          (map && port.target != first.target) || (port.field != nullptr) != map ||
          static_cast<bool>(port.next) != (map || internal) ||
          static_cast<bool>(port.once) != internal ||
          ((map || internal) && port.identity.empty()) ||
          (!map && !internal && !port.identity.empty()))
        throw std::logic_error("AMR levels reached different qualified continuation barriers");
      frame.fields[level] = port.field;
    }
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, prepared_execution_lane(),
                                 "AMR continuation barrier agreement failed collectively");
  const auto category =
      map ? (first.target ? "map-target" : "map-source")
      : internal
          ? (first.kind == MappingPortKind::linear_solve ? "linear-solve" : "field-publication")
          : "complete";
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-continuation-contract"), first.identity},
           {std::string_view("amr-continuation-kind"), std::string_view(category)}},
          prepared_execution_lane()))
    throw std::logic_error("AMR continuation barrier differs between ranks");
}
