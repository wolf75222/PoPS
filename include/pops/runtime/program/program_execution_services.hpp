#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>
#include <pops/numerics/time/amr/levels/amr_clock.hpp>
#include <pops/runtime/config/runtime_params.hpp>
#include <pops/runtime/program/clock_schedule.hpp>
#include <pops/runtime/program/profiler.hpp>

namespace pops::runtime::program {

/// Backend-independent Program operations shared by every execution topology.
///
/// The provider owns topology and storage only.  It exposes the narrow
/// ``program_execution_*`` hooks below; generated Program operations themselves live here exactly
/// once.  This is deliberately CRTP instead of a virtual facade: a generated artifact still calls
/// the concrete context directly, and the compiler resolves each provider hook without a second
/// runtime dispatch table.
template <class Provider>
class ProgramExecutionServices {
 public:
  struct FieldStageOverride {
    int program_block = -1;
    const MultiFab* state = nullptr;
  };

  struct CouplingStateOverride {
    int program_block = -1;
    MultiFab* state = nullptr;
  };

  void set_stage_time(std::int64_t numerator, std::int64_t denominator) const {
    if (denominator <= 0 || numerator < 0 || numerator > denominator)
      throw std::runtime_error("Program stage time is outside [0,1]");
    stage_time_ = amr::Rational(numerator, denominator);
  }

  void configure_primary_clock(const std::string& clock) const {
    clock_schedule_.configure_primary_clock(clock);
    primary_clock_ = clock;
  }

  void declare_clock_relation(const std::string& parent, const std::string& child,
                              int count) const {
    clock_schedule_.declare_relation(parent, child, count);
  }

  bool schedule_domain_occurs(ScheduleDomainKind kind, const std::string& clock,
                              const std::string& stage_identity, int level) const {
    return schedule_coordinate_(kind, clock, stage_identity, level).has_value();
  }

  bool schedule_is_due(int node_id, int every_n, ScheduleDomainKind kind, const std::string& clock,
                       const std::string& stage_identity, int level) const {
    if (node_id < 0 || every_n <= 0)
      throw std::runtime_error("Program schedule requires a valid node and positive period");
    const auto coordinate = schedule_coordinate_(kind, clock, stage_identity, level);
    return coordinate && coordinate->value % every_n == 0;
  }

  bool schedule_at_start(ScheduleDomainKind kind, const std::string& clock,
                         const std::string& stage_identity, int level) const {
    const auto coordinate = schedule_coordinate_(kind, clock, stage_identity, level);
    return coordinate && coordinate->value == 0;
  }

  bool schedule_decision(int node_id, bool due, bool cache_backed) const {
    if (node_id < 0)
      throw std::runtime_error("Program schedule decision requires a valid node");
    return profiler().schedule_decision(due, cache_backed);
  }

  ClockScheduleState::SubcycleScope subcycle_scope(const std::string& parent,
                                                   const std::string& child, int count) const {
    return clock_schedule_.subcycle(parent, child, count);
  }

  void synchronize_sample_and_hold(const std::string& source, const std::string& target, int step,
                                   Real offset) const {
    clock_schedule_.synchronize_sample_and_hold(source, target, step, static_cast<double>(offset));
  }

  int sys_block(int program_block) const {
    const std::vector<int>& block_map = provider_().program_execution_block_map_();
    if (block_map.empty())
      throw std::runtime_error(
          "Program execution has no explicit program-to-runtime block map; positional block "
          "identity is not supported");
    if (program_block < 0 || program_block >= static_cast<int>(block_map.size()))
      throw std::runtime_error("Program block index " + std::to_string(program_block) +
                               " is outside the explicit runtime block map [0, " +
                               std::to_string(block_map.size()) + ")");
    const int mapped = block_map[static_cast<std::size_t>(program_block)];
    const int count = provider_().program_execution_block_count_();
    if (mapped < 0 || mapped >= count)
      throw std::runtime_error("Program block index " + std::to_string(program_block) +
                               " maps to invalid runtime block index " + std::to_string(mapped) +
                               " for a runtime with " + std::to_string(count) + " blocks");
    return mapped;
  }

  int n_blocks() const { return provider_().program_execution_block_count_(); }

  Real physical_time() const { return provider_().program_execution_physical_time_(); }

  void record_scalar(const std::string& name, Real value) const {
    provider_().program_execution_record_scalar_(name, value);
  }

  RuntimeParams program_params(int block) const {
    return provider_().program_execution_params_(block);
  }

  void set_field_logical_timepoint(const std::string& field,
                                   const FieldLogicalTimePoint& point) const {
    provider_().program_execution_set_field_timepoint_(field, point);
  }

  void set_field_boundary_parameters(const std::string& field,
                                     const std::vector<double>& parameters) const {
    provider_().program_execution_set_field_parameters_(field, parameters);
  }

  void set_field_boundary_kernel(const std::string& field,
                                 const CompiledFieldBoundaryKernel& kernel) const {
    provider_().program_execution_set_field_kernel_(field, kernel);
  }

  Profiler& profiler() const { return provider_().program_execution_profiler_(); }

  ProfileScope profile_node(const std::string& name) const {
    return ProfileScope(profiler(), name);
  }

  void profile_record(const std::string& name, std::chrono::steady_clock::time_point start) const {
    const auto end = std::chrono::steady_clock::now();
    profiler().record(name, std::chrono::duration<double>(end - start).count());
  }

  void count_kernel(std::int64_t by = 1) const { profiler().count("kernels", by); }

  void count_scratch(const MultiFab& field) const {
    Profiler& service = profiler();
    if (!service.enabled())
      return;
    service.count("scratch_allocs");
    std::int64_t bytes = 0;
    for (int local = 0; local < field.local_size(); ++local)
      bytes += field.fab(local).size() * static_cast<std::int64_t>(sizeof(Real));
    service.count_max("scratch_peak_bytes", bytes);
  }

  int macro_step() const { return provider_().program_execution_macro_step_(); }

  [[noreturn]] void scheduler_error(const std::string& what) const {
    throw std::runtime_error("pops Program scheduler: " + what);
  }

 protected:
  mutable ClockScheduleState clock_schedule_;
  mutable std::string primary_clock_;
  mutable amr::Rational stage_time_{0, 1};

 private:
  const Provider& provider_() const { return static_cast<const Provider&>(*this); }

  std::optional<ScheduleCoordinate> schedule_coordinate_(ScheduleDomainKind kind,
                                                         const std::string& clock,
                                                         const std::string& stage_identity,
                                                         int level) const {
    return clock_schedule_.coordinate(kind, clock, stage_identity, level,
                                      provider_().program_execution_active_level_(), macro_step());
  }
};

}  // namespace pops::runtime::program
