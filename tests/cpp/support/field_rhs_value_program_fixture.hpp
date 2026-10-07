#pragma once

#include <string>

namespace pops::test::field_rhs_value_program {

// A real source-built Program installation for bounded ownership/SSA unit witnesses. Its DSO
// exposes the actual installer-created context; tests never manufacture a Context identity or Value.
inline constexpr const char* identity = "tests.field-rhs-value-program/owned-lincomb-v2";

inline std::string source() {
  return R"CPP(
#include <pops/runtime/program/program_context.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/config/route_ids.hpp>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>

namespace {
constexpr int D = pops::kNativeDimension;
using UniformContext = pops::runtime::program::ProgramContext<D>;
using AmrContext = pops::runtime::program::AmrProgramContext<D>;
constexpr const char* identity = "tests.field-rhs-value-program/owned-lincomb-v2";
std::map<void*, std::weak_ptr<UniformContext>> uniform_contexts;
std::map<void*, std::weak_ptr<AmrContext>> amr_contexts;

pops::runtime::program::ProgramValuePlan plan() {
  using namespace pops::runtime::program;
  return {identity,
    {{0, 0, ProgramValueStorage::State, -1, 0, {}, 0, {}},
     {1, 0, ProgramValueStorage::StateScratch, 1, 0, {}, 0, {0}},
     {2, 0, ProgramValueStorage::Alias, -1, 0, {}, 0, {1}},
     {3, 0, ProgramValueStorage::StateScratch, 3, 0, {}, 0, {0}}},
    {{10, 0, 1}, {11, 0, 3}, {12, 0, 2}}};
}

template<class Context>
void produce(const std::shared_ptr<Context>& context) {
    context->set_stage_time(1, 2);
    auto root = context->capture_program_value(0, 0, context->state(0));
    auto& stage = context->scratch_state(1, 0, context->state(0));
    auto write = context->begin_program_value_write(1, 0, stage, {root});
    context->lincomb(stage, pops::Real(1), context->state(0), pops::Real(0), context->state(0));
    auto produced = context->complete_program_value_write(std::move(write));
    auto alias = context->alias_program_value(2, produced);
    auto& alternative = context->scratch_state(3, 0, context->state(0));
    auto alternate_write = context->begin_program_value_write(3, 0, alternative, {root});
    context->lincomb(alternative, pops::Real(2), context->state(0), pops::Real(0), context->state(0));
    auto alternate = context->complete_program_value_write(std::move(alternate_write));
    context->require_program_field_value(produced, 10, 0, stage)();
    context->require_program_field_value(alias, 12, 0, stage)();
    context->require_program_field_value(alternate, 11, 0, alternative)();
}
void install_body(const std::shared_ptr<UniformContext>& context) {
  context->configure_primary_clock("test.field-rhs-v2.clock");
  context->install([context](double dt) {
    context->install_program_value_plan(plan()); // The official loader has published its hash.
    context->begin_step(dt);
    produce(context);
  });
}
void install_body(const std::shared_ptr<AmrContext>& context) {
  context->configure_primary_clock("test.field-rhs-v2.clock");
  context->install([context](double dt) {
    context->install_program_value_plan(plan());
    context->advance_hierarchy(dt, [context](double) { produce(context); });
  }, context);
}
}

extern "C" const char* pops_program_abi_key() { return POPS_ABI_KEY_LITERAL; }
extern "C" const char* pops_program_route_manifest() { return pops::kRouteRegistrySignature; }
extern "C" const char* pops_program_name() { return "owned-field-rhs-value-unit-program"; }
extern "C" const char* pops_program_hash() { return identity; }
extern "C" int pops_program_operator_authority_count() { return 0; }
extern "C" std::uint64_t pops_program_operator_authority_word(int, int) { return 0; }
extern "C" int pops_program_block_count() { return 1; }
extern "C" const char* pops_program_block_name(int i) { return i == 0 ? "material" : ""; }
extern "C" int pops_program_param_count() { return 1; }
extern "C" int pops_program_param_block(int) { return 0; }
extern "C" int pops_program_param_index(int) { return 0; }
extern "C" double pops_program_param_default(int) { return 1.25; }
extern "C" bool pops_program_has_dt_bound() { return false; }
extern "C" bool pops_program_has_flux_expression() { return false; }
extern "C" int pops_program_flux_expression_budget_count() { return 1; }
extern "C" std::uint64_t pops_program_flux_rhs_basis_bound(int) { return 0; }
extern "C" std::uint64_t pops_program_flux_coefficient_term_bound(int) { return 0; }
extern "C" std::uint64_t pops_program_interface_coupling_application_bound() { return 0; }
extern "C" std::uint64_t pops_program_interface_coupling_identity_character_bound() { return 0; }
extern "C" int pops_program_checkpoint_history_count() { return 0; }
extern "C" const char* pops_program_checkpoint_history_name(int) { return ""; }
extern "C" int pops_program_checkpoint_history_owner(int) { return 0; }
extern "C" const char* pops_program_checkpoint_history_state_identity(int) { return ""; }
extern "C" const char* pops_program_checkpoint_history_space_identity(int) { return ""; }
extern "C" const char* pops_program_checkpoint_history_clock_identity(int) { return ""; }
extern "C" const char* pops_program_checkpoint_history_interpolation_identity(int) { return ""; }
extern "C" int pops_program_checkpoint_history_depth(int) { return 0; }
extern "C" int pops_program_checkpoint_history_components(int) { return 0; }
extern "C" int pops_program_checkpoint_logical_clock_count() { return 1; }
extern "C" const char* pops_program_checkpoint_logical_clock_identity(int i) {
  return i == 0 ? "test.field-rhs-v2.clock" : "";
}
extern "C" const char* pops_program_checkpoint_primary_clock_identity() {
  return "test.field-rhs-v2.clock";
}
extern "C" const char* pops_program_checkpoint_temporal_provider_identity() {
  return "pops.temporal-partition.global@1";
}
extern "C" std::uint64_t pops_program_checkpoint_temporal_cell_capacity() { return 0; }
extern "C" std::uint64_t pops_program_checkpoint_temporal_cells_per_topology_cell() { return 0; }
extern "C" int pops_module_operator_count() { return 0; }
extern "C" const char* pops_module_operator_owner(int) { return ""; }
extern "C" const char* pops_module_operator_name(int) { return ""; }
extern "C" const char* pops_module_operator_kind(int) { return ""; }
extern "C" const char* pops_module_operator_signature(int) { return ""; }
extern "C" const char* pops_module_operator_requirements(int) { return ""; }
extern "C" int pops_module_state_space_count() { return 1; }
extern "C" const char* pops_module_state_space_name(int i) { return i == 0 ? "U" : ""; }
extern "C" const char* pops_module_state_space_owner(int i) { return i == 0 ? "material" : ""; }
extern "C" int pops_module_field_space_count() { return 0; }
extern "C" const char* pops_module_field_space_name(int) { return ""; }
extern "C" const char* pops_module_field_space_owner(int) { return ""; }

extern "C" void pops_install_program(pops::System<D>* system) {
  auto context = pops::runtime::program::make_program_execution_provider(system);
  install_body(context);
  uniform_contexts[system] = context;
}
extern "C" void pops_install_program_amr(pops::AmrSystem<D>* system) {
  auto context = pops::runtime::program::make_program_execution_provider(system);
  install_body(context);
  amr_contexts[system] = context;
}
extern "C" void* pops_test_field_rhs_uniform_context(pops::System<D>* system) {
  return uniform_contexts.at(system).lock().get();
}
extern "C" void* pops_test_field_rhs_amr_context(pops::AmrSystem<D>* system) {
  return amr_contexts.at(system).lock().get();
}
)CPP";
}
} // namespace pops::test::field_rhs_value_program
