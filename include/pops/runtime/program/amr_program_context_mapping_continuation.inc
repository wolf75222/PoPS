// One synchronous hierarchy attempt spans every qualified map barrier. Per-level closures own
// only IR temporaries; the existing subcycling engine owns candidate states, clocks and flux ledgers.
#include <pops/runtime/program/amr_program_context_hierarchy_barriers.inc>

public:
void suspend_map(std::string identity, bool target, field_type& field,
                 std::function<void()> continuation) const {
  if (!mapping_continuation_ || !mapping_continuation_->executing || identity.empty() ||
      !continuation || active_level_ < 0 ||
      static_cast<std::size_t>(active_level_) >= mapping_continuation_->ports.size())
    throw std::logic_error("AMR map suspension requires an executing qualified level region");
  auto& port = mapping_continuation_->ports[static_cast<std::size_t>(active_level_)];
  if (port.kind != MappingPortKind::complete || port.field || port.next || port.once)
    throw std::logic_error("AMR level reached more than one map barrier in one region");
  port.kind = MappingPortKind::map;
  port.identity = std::move(identity);
  port.target = target;
  port.field = &field;
  port.next = std::move(continuation);
}

template <class Body, class Completed>
void advance_mapping_hierarchy(double dt, Body&& body, std::shared_ptr<AmrProgramContext> owner,
                               Completed&& completed) const {
  if (!std::isfinite(dt) || !(dt > 0) || !owner || owner.get() != this)
    throw std::invalid_argument("AMR mapping hierarchy requires its exact context and positive dt");
  if (mapping_continuation_ || !active_attempt_states_.empty())
    throw std::logic_error("AMR mapping hierarchy cannot nest another attempt");
  refresh_resources_();
  require_facade_execution_();
  std::unique_ptr<MappingContinuation> pending;
  std::exception_ptr error;
  try {
    for (const auto& relation : facade_->prepared_program_temporal_relations())
      if (relation.temporal_ratio() != ::pops::amr::Rational(1, 1))
        throw std::logic_error(
            "AMR stage mapping requires common-time continuation for recursive subcycling");
    pending = std::make_unique<MappingContinuation>();
    pending->dt = dt;
    std::function<void(double)> prepared_body(std::forward<Body>(body));
    if (!prepared_body)
      throw std::invalid_argument("AMR mapping continuation body is empty");
    pending->completed = std::forward<Completed>(completed);
    pending->ports.resize(static_cast<std::size_t>(nlev()));
    pending->fields.resize(static_cast<std::size_t>(nlev()));
    for (auto& port : pending->ports)
      port.next = [prepared_body, dt] { prepared_body(dt); };
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, prepared_execution_lane(),
                                 "AMR mapping continuation preparation failed collectively");
  mapping_continuation_ = std::move(pending);
  swap_level_attempt_envelope_(mapping_continuation_->prior);
  try {
    interface_flux_commit_guard_.reset();
    begin_step(dt);
    prepare_multiblock_subcycling_engine_();
    prepare_mapping_barrier_authority_();
    begin_interface_flux_attempt_();
    const ::pops::amr::ClockWindow root{
        {0, static_cast<std::int64_t>(facade_->macro_step()), {0, 1}, facade_->time()},
        {0, static_cast<std::int64_t>(facade_->macro_step()), {1, 1}, facade_->time() + dt}};
    multiblock_subcycling_->begin_synchronized(root);
    resume_mapping_hierarchy_(std::move(owner));
  } catch (...) {
    cancel_mapping_hierarchy_();
    throw;
  }
}

private:
struct MappingLevelPort {
  MappingPortKind kind = MappingPortKind::complete;
  std::string identity;
  bool target = false;
  field_type* field = nullptr;
  std::function<void()> next;
  std::function<void()> once;
};
struct MappingContinuation {
  double dt = 0;
  bool executing = false;
  bool envelopes_prepared = false;
  std::uint64_t internal_sequence = 0;
  std::string program_hash;
  std::string hierarchy_authority;
  LevelAttemptEnvelope prior;
  std::function<void()> completed;
  std::vector<MappingLevelPort> ports;
  std::vector<field_type*> fields;
};
mutable std::unique_ptr<MappingContinuation> mapping_continuation_;

void cancel_mapping_hierarchy_() const noexcept {
  if (multiblock_subcycling_)
    multiblock_subcycling_->abort_synchronized();
  if (interface_flux_ledger_ && interface_flux_ledger_->in_transaction())
    interface_flux_ledger_->rollback();
  synchronized_level_envelopes_.clear();
  clear_active_multiblock_group_();
  if (mapping_continuation_)
    swap_level_attempt_envelope_(mapping_continuation_->prior);
  mapping_continuation_.reset();
}

void resume_mapping_hierarchy_(std::shared_ptr<AmrProgramContext> owner) const {
  if (!mapping_continuation_ || mapping_continuation_->executing || !multiblock_subcycling_ ||
      !multiblock_subcycling_->has_synchronized_attempt())
    throw std::logic_error("AMR mapping continuation lost its exact live hierarchy attempt");
  auto& frame = *mapping_continuation_;
  bool suspended = false;
  try {
    for (;;) {
      bool internal = false;
      multiblock_subcycling_->resume_synchronized([&](multiblock_level_group_type group) {
        if (!frame.envelopes_prepared) {
          std::exception_ptr error;
          try {
            prepare_synchronized_level_envelopes_(group, frame.dt);
            frame.envelopes_prepared = true;
          } catch (...) {
            error = std::current_exception();
          }
          collectively_rethrow_exception(error, prepared_execution_lane(),
                                         "AMR mapping level envelopes failed collectively");
        }
        frame.executing = true;
        for (int level = 0; level < nlev(); ++level) {
          runtime::program::collective_step_rejection_phase(
              prepared_execution_lane().communicator(),
              {"pops.amr-map-region.v1", "amr-map-region", false, false},
              "AMR mapping level region failed collectively", [&] {
                with_program_attempt_level(level, [&] {
                  auto& port = frame.ports[static_cast<std::size_t>(level)];
                  auto next = std::exchange(port.next, std::function<void()>{});
                  port = MappingLevelPort{};
                  if (!next)
                    throw std::logic_error("AMR mapping level lost its next region");
                  next();
                });
              });
        }
        frame.executing = false;
        agree_continuation_ports_();
        const auto kind = frame.ports.front().kind;
        suspended = kind == MappingPortKind::map;
        internal =
            kind == MappingPortKind::linear_solve || kind == MappingPortKind::field_publication;
        if (internal) {
          runtime::program::collective_step_rejection_phase(
              prepared_execution_lane().communicator(),
              {"pops.amr-hierarchy-barrier.v1", "amr-hierarchy-barrier", false, false},
              "AMR hierarchy barrier callback failed collectively", [&] {
                with_program_attempt_level(0, [&] {
                  auto once = std::exchange(frame.ports.front().once, std::function<void()>{});
                  once();
                });
              });
          ++frame.internal_sequence;
        }
        if (!suspended && !internal) {
          for (int level = 0; level < nlev(); ++level)
            runtime::program::collective_step_rejection_phase(
                prepared_execution_lane().communicator(),
                {"pops.amr-map-flux.v1", "amr-map-flux", false, false},
                "AMR mapping flux materialization failed collectively", [&] {
                  with_program_attempt_level(level, [&] {
                    materialize_active_shared_flux_();
                    for (std::size_t block = 0; block < active_attempt_states_.size(); ++block)
                      materialize_active_flux_expression_(block, *active_attempt_states_[block]);
                  });
                });
        }
        clear_active_multiblock_group_();
      });
      if (internal)
        continue;
      if (suspended) {
        const auto& port = frame.ports.front();
        facade_->suspend_program_map(
            port.identity, port.target, frame.fields,
            [owner = std::move(owner)] { owner->resume_mapping_hierarchy_(owner); },
            multiblock_subcycling_->synchronized_attempt());
        return;
      }
      multiblock_subcycling_->finish_synchronized(
          [&](multiblock_reflux_context_type& reflux) { reconcile_multiblock_reflux_(reflux); },
          [&](std::size_t block, std::size_t level, const field_type& candidate) {
            facade_->validate_prepared_amr_state_publication_candidate(
                static_cast<int>(block), static_cast<int>(level), candidate);
          },
          [&](std::size_t level, std::span<field_type*> pack) {
            stage_prepared_publication_candidates_(level, pack);
          });
      commit_interface_flux_attempt_();
      multiblock_subcycling_has_accepted_step_ = true;
      synchronized_level_envelopes_.clear();
      clear_active_multiblock_group_();
      swap_level_attempt_envelope_(frame.prior);
      auto completed = std::move(frame.completed);
      mapping_continuation_.reset();
      if (completed)
        completed();
      return;
    }
  } catch (...) {
    cancel_mapping_hierarchy_();
    throw;
  }
}
