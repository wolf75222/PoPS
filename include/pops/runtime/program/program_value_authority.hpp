/// Native ownership and successful-production authority for Field-source SSA values.
#pragma once

#include <pops/mesh/storage/multifab.hpp>
#include <pops/core/foundation/kokkos_env.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <pops/runtime/program/prepared_resource_lifetime.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace pops::runtime::program {
template <int Dim> class ProgramContext;
template <int Dim, class MemorySpace> class AmrProgramContext;

enum class ProgramValueStorage : std::uint8_t {
  State, RhsScratch, StateScratch, ScalarScratch, History, Alias
};

struct ProgramValueProducer {
  std::int64_t ssa = -1;
  int program_block = -1;
  ProgramValueStorage storage = ProgramValueStorage::State;
  std::int64_t storage_id = -1;
  int subslot = 0;
  std::string history;
  int lag = 0;
  std::vector<std::int64_t> inputs;
  std::vector<std::vector<std::int64_t>> input_branches;
  friend bool operator==(const ProgramValueProducer&, const ProgramValueProducer&) = default;
};
struct ProgramValueFieldEdge {
  std::int64_t field_node = -1;
  int program_block = -1;
  std::int64_t source = -1;
  bool accepted_only = false;
  friend bool operator==(const ProgramValueFieldEdge&, const ProgramValueFieldEdge&) = default;
};
struct ProgramValuePlan {
  std::string program_identity;
  std::vector<ProgramValueProducer> producers;
  std::vector<ProgramValueFieldEdge> fields;
  friend bool operator==(const ProgramValuePlan&, const ProgramValuePlan&) = default;
};

/// @1. Only a context may admit owned storage or publish a successful producer.
/// A pointer/layout, SSA label or copied MultiFab alone cannot mint a value. The installed
/// dependency plan binds every source to its actual context-owned slot and incarnation.
template <int Dim>
class ProgramValueAuthority {
  using field_type = MultiFab<Dim>;
  using Point = multiblock::BoundaryEvaluationPoint;
  struct Slot {
    ProgramValueStorage kind;
    int level, owner;
    std::int64_t storage_id;
    int subslot;
    std::string history;
    int lag;
    friend bool operator==(const Slot&, const Slot&) = default;
    friend bool operator<(const Slot& a, const Slot& b) {
      return std::tie(a.kind,a.level,a.owner,a.storage_id,a.subslot,a.history,a.lag) <
             std::tie(b.kind,b.level,b.owner,b.storage_id,b.subslot,b.history,b.lag);
    }
  };
  struct Entry {
    const field_type* buffer = nullptr;
    std::uint64_t incarnation = 0;
  };
  struct Record {
    Slot slot;
    const field_type* buffer;
    std::shared_ptr<const field_type> retained;
    std::uint64_t incarnation, generation;
    std::int64_t ssa;
    int program_block;
    PreparedResourceAttempt attempt;
    Point birth;
    bool complete = false;
    std::vector<typename field_type::fab_type::storage_type> allocation_owners;
  };
  struct State {
    ProgramValuePlan plan;
    bool installed = false;
    std::uint64_t next_incarnation = 0, generation = 0;
    std::map<std::int64_t, ProgramValueProducer> producers;
    std::map<std::pair<std::int64_t,int>, std::int64_t> edges;
    std::map<Slot, Entry> slots;
    std::map<std::pair<std::int64_t,int>, std::shared_ptr<Record>> values;
    std::map<std::pair<std::int64_t,int>, std::shared_ptr<Record>> quarantined;
    // A failed completion fence must retain allocation owners even if the context unwinds.
    // Only a subsequent successful drain releases this exceptional self-retention.
    std::shared_ptr<State> failed_fence_owner;
  };

 public:
  static constexpr unsigned contract_version = 1;
  class Value {
   public:
    Value() = default;
    void validate() const { (void)require_(); }
    const field_type& state() const { return *require_().retained; }
    const Point& birth_point() const { return require_().birth; }
    int level() const { return require_().slot.level; }
    int runtime_block() const { return require_().slot.owner; }
    std::int64_t source_ssa() const { return require_().ssa; }
   private:
    friend class ProgramValueAuthority;
    Value(std::weak_ptr<State> owner, std::shared_ptr<const Record> record)
        : owner_(std::move(owner)), record_(std::move(record)) {}
    const Record& require_() const {
      const auto owner = owner_.lock();
      if (!owner || !record_ || !record_->complete || !record_->attempt.visible() ||
          record_->generation != owner->generation)
        throw std::logic_error("Program SSA value has expired native production authority");
      const auto slot = owner->slots.find(record_->slot);
      const auto value = owner->values.find({record_->ssa,record_->slot.level});
      if (slot == owner->slots.end() || slot->second.buffer != record_->buffer ||
          slot->second.incarnation != record_->incarnation || value == owner->values.end() ||
          value->second.get() != record_.get())
        throw std::logic_error("Program SSA value no longer owns its exact slot incarnation");
      require_allocations_(*record_);
      return *record_;
    }
    std::weak_ptr<State> owner_;
    std::shared_ptr<const Record> record_;
  };

  /// Accepted hierarchy images do not claim a stage SSA produced on another level.
  class AcceptedValue {
   public:
    AcceptedValue() = default;
    void validate() const { (void)require_(); }
    const field_type& state() const { return *require_().retained; }
    const Point& birth_point() const { return require_().birth; }
    int level() const { return require_().slot.level; }
    int runtime_block() const { return require_().slot.owner; }
   private:
    friend class ProgramValueAuthority;
    AcceptedValue(std::weak_ptr<State> owner, std::shared_ptr<const Record> record,
                  std::int64_t field_node)
        : owner_(std::move(owner)), record_(std::move(record)), field_node_(field_node) {}
    const Record& require_() const {
      const auto owner = owner_.lock();
      if (!owner || !record_ || !record_->attempt.visible() ||
          record_->generation != owner->generation)
        throw std::logic_error("accepted Field source has expired native authority");
      const auto slot = owner->slots.find(record_->slot);
      if (slot == owner->slots.end() || slot->second.buffer != record_->buffer ||
          slot->second.incarnation != record_->incarnation)
        throw std::logic_error("accepted Field source lost its exact hierarchy slot");
      require_allocations_(*record_);
      return *record_;
    }
    std::weak_ptr<State> owner_;
    std::shared_ptr<const Record> record_;
    std::int64_t field_node_ = -1;
  };

  class WriteTicket {
   public:
    WriteTicket(const WriteTicket&) = delete;
    WriteTicket& operator=(const WriteTicket&) = delete;
    WriteTicket(WriteTicket&&) noexcept = default;
    WriteTicket& operator=(WriteTicket&&) noexcept = default;
   private:
    friend class ProgramValueAuthority;
    WriteTicket(std::weak_ptr<State> owner, std::shared_ptr<Record> record)
        : owner_(std::move(owner)), record_(std::move(record)) {}
    std::weak_ptr<State> owner_;
    std::shared_ptr<Record> record_;
  };

 private:
  friend class ProgramContext<Dim>;
  template <int, class> friend class AmrProgramContext;
  ProgramValueAuthority() : state_(std::make_shared<State>()) {}
  ProgramValueAuthority(const ProgramValueAuthority&) = delete;
  ProgramValueAuthority& operator=(const ProgramValueAuthority&) = delete;
  ProgramValueAuthority(ProgramValueAuthority&&) = default;
  ProgramValueAuthority& operator=(ProgramValueAuthority&&) = default;
  static void require_allocations_(const Record& record) {
    if (record.buffer->local_size() != record.allocation_owners.size())
      throw std::logic_error("Program source replaced its native allocation set");
    for (std::size_t local = 0; local < record.allocation_owners.size(); ++local)
      if (record.buffer->fab(local).storage().data() != record.allocation_owners[local].data())
        throw std::logic_error("Program source lost its exact native allocation incarnation");
  }

  void install_(ProgramValuePlan plan) const {
    if (state_->installed) {
      if (state_->plan != plan)
        throw std::logic_error("Program value plan is immutable after installation");
      return;
    }
    if (plan.program_identity.empty() || plan.fields.empty())
      throw std::invalid_argument("Program value plan requires Program identity and Field edges");
    std::map<std::int64_t,ProgramValueProducer> producers;
    for (const auto& p : plan.producers) {
      if (p.ssa < 0 || p.program_block < 0 || p.subslot < 0 || p.lag < 0 ||
          static_cast<unsigned>(p.storage) > static_cast<unsigned>(ProgramValueStorage::Alias) ||
          !producers.emplace(p.ssa,p).second ||
          (p.storage == ProgramValueStorage::Alias && p.inputs.size() != 1) ||
          ((p.storage == ProgramValueStorage::State || p.storage == ProgramValueStorage::History) &&
           (!p.inputs.empty() || !p.input_branches.empty() || p.storage_id != -1 || p.subslot != 0)) ||
          (p.storage == ProgramValueStorage::History && p.history.empty()) ||
          ((p.storage == ProgramValueStorage::RhsScratch || p.storage == ProgramValueStorage::StateScratch ||
            p.storage == ProgramValueStorage::ScalarScratch) && p.storage_id < 0))
        throw std::invalid_argument("Program value plan contains an invalid producer");
    }
    std::map<std::pair<std::int64_t,int>,std::int64_t> edges;
    std::set<std::int64_t> visiting, closure;
    const auto visit = [&](auto&& self, std::int64_t id, int block) -> void {
      const auto found = producers.find(id);
      if (found == producers.end() || found->second.program_block != block)
        throw std::invalid_argument("Program value plan has a foreign or missing input");
      if (closure.contains(id)) return;
      if (!visiting.insert(id).second)
        throw std::invalid_argument("Program value plan has a foreign, missing or cyclic input");
      for (const auto input : found->second.inputs) self(self,input,block);
      for (const auto& branch : found->second.input_branches)
        for (const auto input : branch) self(self,input,block);
      visiting.erase(id);
      closure.insert(id);
    };
    for (const auto& edge : plan.fields) {
      if (edge.field_node < 0 || edge.program_block < 0 ||
          !edges.emplace(std::make_pair(edge.field_node,edge.program_block),edge.source).second)
        throw std::invalid_argument("Program value plan contains a duplicate or invalid Field edge");
      if (edge.accepted_only) {
        if (edge.source != -1)
          throw std::invalid_argument("accepted Field edge cannot claim a stage SSA");
      } else visit(visit,edge.source,edge.program_block);
    }
    if (closure.size() != producers.size())
      throw std::invalid_argument("Program value plan must contain exactly the Field-source closure");
    state_->plan = std::move(plan);
    state_->producers = std::move(producers);
    state_->edges = std::move(edges);
    state_->installed = true;
  }

  void revoke_() const noexcept {
    ++state_->generation;
    state_->values.clear();
    state_->slots.clear();
  }
  void revoke_buffer_(const field_type& field) const noexcept {
    for (auto it = state_->slots.begin(); it != state_->slots.end();) {
      if (it->second.buffer == &field) it = state_->slots.erase(it); else ++it;
    }
    for (auto it = state_->values.begin(); it != state_->values.end();) {
      if (it->second->buffer == &field) it = state_->values.erase(it); else ++it;
    }
  }
  void quarantine_(WriteTicket& ticket) const noexcept {
    if (ticket.owner_.lock() != state_ || !ticket.record_) return;
    state_->failed_fence_owner = state_;
    auto node = state_->values.extract({ticket.record_->ssa,ticket.record_->slot.level});
    if (!node.empty()) {
      node.key().first = static_cast<std::int64_t>(ticket.record_->incarnation);
      state_->quarantined.insert(std::move(node));
    }
  }
  void drain_failed_() const {
    if (!state_->failed_fence_owner) return;
    device_fence();
    state_->quarantined.clear();
    state_->failed_fence_owner.reset();
  }
  std::function<void()> rejection_callback_() const {
    return [weak = std::weak_ptr<State>(state_)] {
      if (const auto state = weak.lock()) {
        ++state->generation;
        state->values.clear();
        state->slots.clear();
      }
    };
  }
  std::function<void()> drain_callback_() const {
    return [weak = std::weak_ptr<State>(state_)] {
      if (const auto state = weak.lock(); state && state->failed_fence_owner) {
        device_fence();
        state->quarantined.clear();
        state->failed_fence_owner.reset();
      }
    };
  }
  const ProgramValueProducer& producer_(std::int64_t ssa, int block) const {
    const auto found = state_->producers.find(ssa);
    if (!state_->installed || found == state_->producers.end() || found->second.program_block != block)
      throw std::invalid_argument("SSA producer is outside the installed Field-source plan");
    return found->second;
  }
  std::uint64_t admit_(const Slot& slot, const field_type& field) const {
    auto found = state_->slots.find(slot);
    if (found != state_->slots.end() && found->second.buffer == &field) return found->second.incarnation;
    if (state_->next_incarnation == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("Program value slot incarnation exhausted");
    const auto ordinal = ++state_->next_incarnation;
    state_->slots[slot] = Entry{&field,ordinal};
    return ordinal;
  }
  void require_slot_(const ProgramValueProducer& p, const Slot& slot) const {
    if (p.storage != slot.kind || p.storage_id != slot.storage_id || p.subslot != slot.subslot ||
        p.history != slot.history || p.lag != slot.lag)
      throw std::invalid_argument("SSA producer does not name its installed native storage slot");
  }
  WriteTicket begin_(std::int64_t ssa, int block, const Slot& slot, const field_type& field,
                     const PreparedResourceAttempt& attempt, Point birth,
                     const std::vector<Value>& inputs, bool capture) const {
    const auto& p = producer_(ssa,block);
    if (p.storage != ProgramValueStorage::Alias) require_slot_(p,slot);
    const bool root = p.storage == ProgramValueStorage::State || p.storage == ProgramValueStorage::History;
    std::vector<std::int64_t> ids;
    for (const auto& value : inputs) ids.push_back(value.require_().ssa);
    if (!attempt.visible() || capture != root ||
        (ids != p.inputs && std::find(p.input_branches.begin(),p.input_branches.end(),ids) == p.input_branches.end()))
      throw std::invalid_argument("SSA production lacks its planned exact inputs or active attempt");
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto& source = inputs[i].require_();
      if (inputs[i].owner_.lock() != state_ ||
          source.program_block != block || source.slot.level != slot.level ||
          !source.attempt.same_attempt(attempt))
        throw std::invalid_argument("SSA production input differs from the installed dependency edge");
    }
    if (p.storage == ProgramValueStorage::Alias &&
        (inputs.size() != 1 || inputs.front().record_->buffer != &field ||
         inputs.front().record_->slot != slot))
      throw std::invalid_argument("in-place SSA alias does not inherit its exact input storage");
    // Two planned read roots may bind the same unchanged state/history slot. Reading that slot
    // does not mint a new storage incarnation or revoke an earlier independent read capture.
    if (!capture) revoke_buffer_(field);
    const auto incarnation = admit_(slot,field);
    auto record = std::make_shared<Record>(Record{slot,&field,{},
        incarnation,state_->generation,ssa,block,attempt,std::move(birth),false});
    record->allocation_owners.reserve(field.local_size());
    for (std::size_t local = 0; local < field.local_size(); ++local)
      record->allocation_owners.push_back(field.fab(local).storage());
    state_->values[{ssa,slot.level}] = record;
    return WriteTicket(state_,std::move(record));
  }
  void prepare_complete_(WriteTicket& ticket) const {
    if (ticket.owner_.lock() != state_ || !ticket.record_ || !ticket.record_->attempt.visible())
      throw std::logic_error("SSA snapshot preparation has no live write ticket");
    const auto& record = *ticket.record_;
    const auto slot = state_->slots.find(record.slot);
    const auto found = state_->values.find({record.ssa,record.slot.level});
    if (record.generation != state_->generation || slot == state_->slots.end() ||
        slot->second.buffer != record.buffer || slot->second.incarnation != record.incarnation ||
        found == state_->values.end() || found->second != ticket.record_)
      throw std::logic_error("SSA storage was revoked before snapshot preparation");
    if (record.buffer->local_size() != record.allocation_owners.size())
      throw std::logic_error("SSA producer replaced its native allocation set");
    for (std::size_t local = 0; local < record.allocation_owners.size(); ++local)
      if (record.buffer->fab(local).storage().data() != record.allocation_owners[local].data())
        throw std::logic_error("SSA producer replaced its owned allocation incarnation");
    // MultiFab is deep-owning. Snapshot only after the real producer fence, never before a write.
    ticket.record_->retained = std::make_shared<field_type>(*record.buffer);
    device_fence();
  }
  Value complete_(WriteTicket&& ticket) const {
    auto record = std::exchange(ticket.record_,{});
    if (ticket.owner_.lock() != state_ || !record || !record->retained || !record->attempt.visible() ||
        record->generation != state_->generation)
      throw std::logic_error("SSA write ticket is foreign, failed or expired");
    const auto found = state_->values.find({record->ssa,record->slot.level});
    const auto slot = state_->slots.find(record->slot);
    if (found == state_->values.end() || found->second != record || slot == state_->slots.end() ||
        slot->second.buffer != record->buffer || slot->second.incarnation != record->incarnation)
      throw std::logic_error("SSA write ticket was superseded before successful completion");
    record->complete = true;
    return Value(state_,std::move(record));
  }
  Value alias_(std::int64_t ssa, const Value& input) const {
    const auto& source = input.require_();
    const auto& p = producer_(ssa,source.program_block);
    if (input.owner_.lock() != state_ || p.storage != ProgramValueStorage::Alias ||
        p.inputs != std::vector<std::int64_t>{source.ssa})
      throw std::invalid_argument("SSA alias differs from its installed native edge");
    auto record = std::make_shared<Record>(source);
    record->ssa = ssa;
    state_->values[{ssa,source.slot.level}] = record;
    return Value(state_,std::move(record));
  }
  Value value_(std::int64_t ssa, int block, int level, const field_type& field,
               const PreparedResourceAttempt& attempt) const {
    (void)producer_(ssa,block);
    const auto found = state_->values.find({ssa,level});
    if (found == state_->values.end() || found->second->buffer != &field ||
        !found->second->attempt.same_attempt(attempt))
      throw std::invalid_argument("Field source has no successful exact SSA producer");
    Value value(state_,found->second);
    value.validate();
    return value;
  }
  std::function<void()> require_field_(const Value& value, std::int64_t node, int block,
      int level, const field_type& field, const PreparedResourceAttempt& attempt) const {
    const auto& record = value.require_();
    const auto edge = state_->edges.find({node,block});
    if (value.owner_.lock() != state_ || edge == state_->edges.end() || edge->second != record.ssa ||
        record.program_block != block || record.slot.level != level || record.buffer != &field ||
        !record.attempt.same_attempt(attempt))
      throw std::invalid_argument("Field solve does not consume its planned native SSA source");
    return [value] { value.validate(); };
  }
  AcceptedValue accepted_(std::int64_t node, int block, const Slot& slot,
      const field_type& field, const PreparedResourceAttempt& attempt, Point birth) const {
    if (!state_->installed || !state_->edges.contains({node,block}) || !attempt.visible())
      throw std::invalid_argument("accepted Field source is outside the installed Field plan");
    const auto incarnation = admit_(slot,field);
    auto record = std::make_shared<Record>(Record{slot,&field,std::make_shared<field_type>(field),
        incarnation,state_->generation,-1,block,attempt,std::move(birth),true});
    record->allocation_owners.reserve(field.local_size());
    for (std::size_t local = 0; local < field.local_size(); ++local)
      record->allocation_owners.push_back(field.fab(local).storage());
    return AcceptedValue(state_,std::move(record),node);
  }
  std::function<void()> require_accepted_field_(const AcceptedValue& value, std::int64_t node,
      int block, int level, const field_type& field, const PreparedResourceAttempt& attempt) const {
    const auto& record = value.require_();
    if (value.owner_.lock() != state_ || value.field_node_ != node || record.program_block != block ||
        record.slot.level != level || record.buffer != &field || !record.attempt.same_attempt(attempt))
      throw std::invalid_argument("accepted Field source differs from its exact hierarchy binding");
    return [value] { value.validate(); };
  }
  std::shared_ptr<State> state_;
};
}  // namespace pops::runtime::program
