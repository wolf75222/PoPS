#pragma once

#include "amr_runtime_authority.hpp"
#include "native_dso_compiler.hpp"
#include "program_v5_fixture.hpp"

#include <pops/runtime/program/program_execution_services.hpp>

#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::test {

/// Explicit test-only access for legacy native regrid fixtures that must mutate an already
/// prepared runtime. Production and Program code use the lease-owned accepted_amr_runtime() view
/// or the private ProgramExecutionServices seam; this helper is intentionally not a public API.
template <int Dim>
struct AmrSystemTestAccess {
  using runtime_type = runtime::amr::AmrRuntime<Dim, typename AmrSystem<Dim>::memory_space>;

  static runtime_type* engine(AmrSystem<Dim>& system) {
    const auto accepted = system.accepted_amr_runtime();
    if (!accepted)
      return nullptr;
    return system.program_engine_();
  }

  static const runtime_type* engine(const AmrSystem<Dim>& system) {
    const auto accepted = system.accepted_amr_runtime();
    if (!accepted)
      return nullptr;
    return system.program_engine_();
  }
};

namespace explicit_amr_program_detail {

using context_type = runtime::program::ProgramExecutionServices<kNativeDimension>;
using callback_type = std::function<void(context_type&, double)>;

inline std::vector<callback_type>& callbacks() {
  static std::vector<callback_type> value;
  return value;
}

}  // namespace explicit_amr_program_detail

extern "C" void pops_test_explicit_amr_program_callback(std::uint64_t identifier, void* opaque,
                                                        double dt) {
  auto& callbacks = explicit_amr_program_detail::callbacks();
  if (opaque == nullptr || identifier >= callbacks.size())
    throw std::logic_error("explicit AMR ABI-v5 callback received an invalid dispatch token");
  callbacks.at(static_cast<std::size_t>(identifier))(
      *static_cast<explicit_amr_program_detail::context_type*>(opaque), dt);
}

/// AMR facade test Program: one explicit rate stage per recursive hierarchy clock.
///
/// ProgramExecutionServices owns level clocks and conservative catch-up. AmrRuntime remains the spatial
/// engine inspected by tests and exposes no temporal step entry point.
template <int Dim>
inline void install_explicit_amr_callback_program(
    AmrSystem<Dim>& system, std::string_view identity, std::string_view clock,
    const std::vector<std::string>& program_blocks,
    const std::vector<program_v5::CallbackProgramResource>& resources,
    const std::vector<program_v5::CallbackProgramFieldRoute>& field_routes,
    explicit_amr_program_detail::callback_type callback,
    const std::vector<program_v5::CallbackProgramHistory>& histories = {},
    const std::vector<program_v5::CallbackProgramClockRelation>& clock_relations = {},
    const std::optional<std::vector<pops::runtime::program::ProgramFluxBudgetRecord>>&
        flux_budgets = std::nullopt,
    const program_v5::CallbackProgramTransactionAuthorities& transaction_authorities = {},
    const std::optional<program_v5::CallbackProgramCellTemporalAuthority>& cell_temporal =
        std::nullopt,
    const std::vector<program_v5::CallbackProgramFluxBasisOccurrence>& flux_basis_occurrences = {},
    const std::vector<program_v5::CallbackProgramFaceFluxStage>& face_flux_stages = {},
    const std::optional<program_v5::CallbackProgramHierarchyTensorAuthority>& hierarchy_tensor =
        std::nullopt,
    const std::vector<pops::runtime::system::AuxiliaryConsumerProviderPlan<kNativeDimension>>&
        auxiliary_consumer_plans = {}) {
  static_assert(Dim == kNativeDimension,
                "the ABI-v5 explicit AMR fixture is compiled for POPS_NATIVE_DIM");
  if (identity.empty() || clock.empty() || !callback)
    throw std::invalid_argument(
        "explicit AMR callback Program requires exact callback authorities");
  const auto runtime_blocks = system.block_names();
  if (program_blocks.empty() || program_blocks.size() != runtime_blocks.size())
    throw std::logic_error("explicit AMR callback Program requires declared blocks");
  std::vector<std::string> remaining = runtime_blocks;
  for (const std::string& block : program_blocks) {
    const auto found = std::find(remaining.begin(), remaining.end(), block);
    if (found == remaining.end())
      throw std::invalid_argument(
          "explicit AMR callback Program block order is not an exact runtime permutation");
    remaining.erase(found);
  }
  // The process-lifetime registry is deliberately the callback ABI owner: generated MODULE code
  // holds only its dense token and never a System/context pointer.  The callback receives the
  // typed v5 service object for one dispatch and cannot retain an AMR facade.
  auto& callbacks = explicit_amr_program_detail::callbacks();
  const auto callback_identifier = static_cast<std::uint64_t>(callbacks.size());
  callbacks.emplace_back(std::move(callback));
#if !defined(POPS_TEST_TMPDIR)
  throw std::runtime_error("explicit AMR ABI-v5 fixture requires POPS_TEST_TMPDIR");
#else
  static std::size_t fixture_index = 0;
  const std::string prefix =
      std::string(POPS_TEST_TMPDIR) + "/explicit_amr_callback_" + std::to_string(++fixture_index);
  const std::string source_path = prefix + ".cpp";
  const std::string library_path = prefix + ".so";
  const auto compiled = native_dso::compile_shared_collectively(
      identity,
      [&]() {
        return program_v5::callback_program_source(
            callback_identifier, identity, clock, program_blocks, resources,
            "pops_test_explicit_amr_program_callback", "amr", field_routes, transaction_authorities,
            histories, clock_relations, flux_budgets, cell_temporal, flux_basis_occurrences,
            face_flux_stages, hierarchy_tensor, auxiliary_consumer_plans);
      },
      source_path, library_path, "explicit_amr_callback_program");
  if (!compiled.ok) {
    throw std::runtime_error("explicit AMR ABI-v5 callback compilation failed");
  }
  system.install_program(compiled.library_path);
#endif
}

template <int Dim>
inline void install_explicit_amr_callback_program(
    AmrSystem<Dim>& system, std::string_view identity, std::string_view clock,
    const std::vector<program_v5::CallbackProgramResource>& resources,
    const std::vector<program_v5::CallbackProgramFieldRoute>& field_routes,
    explicit_amr_program_detail::callback_type callback,
    const std::vector<program_v5::CallbackProgramHistory>& histories = {},
    const std::vector<program_v5::CallbackProgramClockRelation>& clock_relations = {},
    const std::optional<std::vector<pops::runtime::program::ProgramFluxBudgetRecord>>&
        flux_budgets = std::nullopt,
    const program_v5::CallbackProgramTransactionAuthorities& transaction_authorities = {},
    const std::optional<program_v5::CallbackProgramCellTemporalAuthority>& cell_temporal =
        std::nullopt,
    const std::vector<program_v5::CallbackProgramFluxBasisOccurrence>& flux_basis_occurrences = {},
    const std::vector<program_v5::CallbackProgramFaceFluxStage>& face_flux_stages = {},
    const std::optional<program_v5::CallbackProgramHierarchyTensorAuthority>& hierarchy_tensor =
        std::nullopt,
    const std::vector<pops::runtime::system::AuxiliaryConsumerProviderPlan<kNativeDimension>>&
        auxiliary_consumer_plans = {}) {
  install_explicit_amr_callback_program<Dim>(
      system, identity, clock, system.block_names(), resources, field_routes, std::move(callback),
      histories, clock_relations, flux_budgets, transaction_authorities, cell_temporal,
      flux_basis_occurrences, face_flux_stages, hierarchy_tensor, auxiliary_consumer_plans);
}

template <int Dim>
inline void install_forward_euler_program_execution_services(AmrSystem<Dim>& system,
                                                             bool solve_default_field) {
  static_assert(Dim == kNativeDimension,
                "the ABI-v5 explicit AMR fixture is compiled for POPS_NATIVE_DIM");
  using Resource = program_v5::CallbackProgramResource;
  const int block_count = system.n_blocks();
  const int level_count = system.n_levels();
  if (block_count < 1 || level_count < 1)
    throw std::logic_error("explicit AMR Program requires materialized hierarchy levels");
  const std::vector<std::string> program_blocks = system.block_names();
  if (program_blocks.size() != static_cast<std::size_t>(block_count))
    throw std::logic_error("explicit AMR Program block authority is incomplete");
  std::vector<Resource> resources;
  resources.reserve(static_cast<std::size_t>(block_count * level_count));
  std::vector<program_v5::CallbackProgramFluxBasisOccurrence> flux_basis_occurrences;
  std::vector<program_v5::CallbackProgramFaceFluxStage> face_flux_stages;
  flux_basis_occurrences.reserve(resources.capacity());
  face_flux_stages.reserve(resources.capacity());
  for (int level = 0; level < level_count; ++level) {
    for (int block = 0; block < block_count; ++block) {
      const auto state = system.prepared_amr_block_state(block, level);
      if (!state)
        throw std::logic_error("explicit AMR Program resource has no materialized block state");
      const std::size_t slot = resources.size();
      const int rhs_identity = 3000 + block;
      const std::string resource_identity = "tests.explicit-amr/forward-euler/rhs/" +
                                            std::to_string(block) + "/level/" +
                                            std::to_string(level);
      Resource resource{Resource::Kind::rhs,
                        slot,
                        0,
                        block,
                        level,
                        static_cast<std::uint32_t>(state->ncomp()),
                        static_cast<std::uint32_t>(state->ghosts()[0])};
      resource.value_id = static_cast<std::uint64_t>(rhs_identity);
      resource.identity = resource_identity;
      resource.occurrence_path = resource_identity + "/occurrence";
      resource.owner = program_blocks.at(static_cast<std::size_t>(block));
      resource.clock = "test.clock.macro";
      resources.push_back(std::move(resource));

      const auto dense_slot = static_cast<std::uint32_t>(slot);
      flux_basis_occurrences.push_back(
          {dense_slot, dense_slot, block, level, rhs_identity, 0, 0, 1,
           resource_identity + "/flux-basis", resource_identity + "/flux-basis/occurrence",
           program_blocks.at(static_cast<std::size_t>(block)), "test.clock.macro"});
      face_flux_stages.push_back(
          {dense_slot, dense_slot, dense_slot, 1, 1, 1, resource_identity + "/face-flux",
           resource_identity + "/face-flux/occurrence",
           program_blocks.at(static_cast<std::size_t>(block)), "test.clock.macro"});
    }
  }
  const std::optional<std::vector<runtime::program::ProgramFluxBudgetRecord>> flux_budgets{
      std::vector<runtime::program::ProgramFluxBudgetRecord>(static_cast<std::size_t>(block_count),
                                                             {1, 1, 0, 0})};

  install_explicit_amr_callback_program<Dim>(
      system, "tests.explicit-amr/forward-euler@1", "test.clock.macro", resources, {},
      [solve_default_field, block_count](explicit_amr_program_detail::context_type& context,
                                         double macro_dt) {
        context.advance_hierarchy(
            macro_dt, [&context, solve_default_field, block_count](double level_dt) {
              context.set_stage_time(0, 1);
              if (solve_default_field && context.level() == 0)
                (void)consume_solve_outcome(context.solve_default_field_on_coarse_level());

              std::vector<MultiFab<Dim>*> states;
              std::vector<MultiFab<Dim>*> residuals;
              states.reserve(static_cast<std::size_t>(context.n_blocks()));
              residuals.reserve(static_cast<std::size_t>(context.n_blocks()));
              for (int block = 0; block < context.n_blocks(); ++block) {
                MultiFab<Dim>& state = context.state(block);
                const auto slot = static_cast<runtime::program::ProgramCacheSlot>(
                    context.level() * block_count + block);
                MultiFab<Dim>& residual = context.rhs_scratch(slot, 0, state);
                context.rhs_into(block, state, residual, 3000 + block);
                states.push_back(&state);
                residuals.push_back(&residual);
              }
              for (std::size_t block = 0; block < states.size(); ++block)
                context.axpy(*states[block], Real(level_dt), *residuals[block]);
            });
      },
      {}, {}, flux_budgets, {}, std::nullopt, flux_basis_occurrences, face_flux_stages);
}

template <int Dim>
inline void install_forward_euler_program(AmrSystem<Dim>& system, bool solve_default_field) {
  install_forward_euler_program_execution_services(system, solve_default_field);
}

}  // namespace pops::test
