/// @file
/// @brief Private compile-time-ranked storage shared by the System implementation TUs.

#pragma once

#include <pops/runtime/system.hpp>

#include <pops/runtime/system/system_block_store.hpp>
#include <pops/runtime/system/system_boundary_registry.hpp>
#include <pops/runtime/system/system_coupling_registry.hpp>
#include <pops/runtime/system/system_domain.hpp>
#include <pops/runtime/system/exact_named_field.hpp>
#include <pops/runtime/system/system_lifecycle.hpp>
#include <pops/runtime/program/program_runtime_state.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <exception>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {

inline void require_assembling(const runtime::system::SystemLifecycle& lifecycle,
                               const char* operation) {
  if (lifecycle.frozen())
    throw std::runtime_error(
        std::string("System::") + operation +
        ": the composition is frozen once pops.bind completes; create a fresh runtime to change "
        "its structure");
}

template <int Dim>
struct System<Dim>::Impl {
  using domain_type = runtime::system::SystemDomain<Dim>;
  using block_store_type = SystemBlockStore<Dim>;
  using boundary_registry_type = runtime::system::SystemBoundaryRegistry<Dim>;
  using field_type = MultiFab<Dim>;
  using Species = typename block_store_type::BlockState;

  domain_type domain_;
  SystemConfig<Dim>& cfg = domain_.cfg;
  Box<Dim>& dom = domain_.dom;
  Geometry<Dim>& geom = domain_.geom;
  mesh::BoxArray<Dim>& ba = domain_.ba;
  mesh::Distribution<Dim>& dm = domain_.dm;
  Index<Dim>& local_rank = domain_.local_rank;
  std::array<bool, Dim>& periodicity = domain_.periodicity;
  field_type& aux = domain_.aux;
  int& aux_ncomp_ = domain_.aux_ncomp;

  block_store_type blocks_;
  std::vector<Species>& sp = blocks_.blocks;
  boundary_registry_type boundary_registry_;
  runtime::system::SystemCouplingRegistry<Dim> coupling_;
  runtime::system::SystemLifecycle lifecycle_;
  runtime::program::ProgramRuntimeState<Dim> program_;

  double t = 0.0;
  int macro_step_ = 0;
  std::string last_dt_reason_;
  std::string poisson_solver_ = "geometric_mg";
  std::string poisson_bc_ = "auto";
  double poisson_abs_tol_ = 0.0;
  double poisson_rel_tol_ = 1.0e-10;
  int poisson_max_iterations_ = 2000;

  using exact_field_type = runtime::system::ExactNamedField<Dim>;
  using component_field_solver_type = runtime::field::PreparedFieldSolverComponent<Dim>;

  struct FieldProviderBinding {
    std::string identity;
    std::string block;
    std::string key;
    double coefficient = 1.0;
  };

  struct FieldPlan {
    std::string plan_identity;
    std::string provider_identity;
    std::string output_owner_identity;
    std::string output_block;
    std::string output_key;
    std::vector<FieldProviderBinding> providers;
    std::string backend_provider_route;
    std::vector<std::string> boundary_kind;
    std::vector<double> boundary_alpha;
    std::vector<double> boundary_beta;
    std::vector<double> boundary_value;
    std::vector<std::string> boundary_state_blocks;
    std::vector<int> boundary_state_components;
    std::vector<std::string> boundary_field_blocks;
    std::vector<std::string> boundary_field_keys;
    std::vector<int> boundary_field_components;
    std::vector<double> boundary_parameters;
    std::string nullspace_provider_identity;
    PreparedProviderOptions nullspace_options;
    std::string topology_provider_kind;
    std::string topology_provenance;
    std::string topology_digest;
    double reaction = 0.0;
    bool has_reaction = false;
    bool has_boundary_kernel = false;
    bool has_nonlinear_plan = false;
  };

  struct ConfiguredFieldSolverProvider {
    std::string family_route;
    std::string exact_identity;
    PreparedProviderOptions options;
    double relative_tolerance = 0.0;
    double absolute_tolerance = 0.0;
    int maximum_iterations = 0;
  };

  std::shared_ptr<exact_field_type> default_field_;
  std::map<std::string, std::shared_ptr<exact_field_type>> named_fields_;
  std::shared_ptr<exact_field_type> active_field_;
  std::map<std::string, FieldPlan> field_plans_;
  std::map<std::string, ConfiguredFieldSolverProvider> configured_field_solver_providers_;
  std::map<std::string, std::shared_ptr<component_field_solver_type>>
      component_field_solver_providers_;
  bool field_plan_consensus_verified_ = false;
  std::string default_nullspace_provider_identity_;
  PreparedProviderOptions default_nullspace_options_;

  static std::string exact_field_plan_contract(const FieldPlan& plan) {
    ExactContractBuilder contract;
    contract.text("pops.system.exact-ranked-field-plan")
        .scalar(std::uint32_t{1})
        .text(plan.plan_identity)
        .text(plan.provider_identity)
        .text(plan.output_owner_identity)
        .text(plan.output_block)
        .text(plan.output_key)
        .sequence(plan.providers,
                  [](ExactContractBuilder& item, const FieldProviderBinding& row) {
                    item.text(row.identity).text(row.block).text(row.key).scalar(row.coefficient);
                  })
        .text(plan.backend_provider_route)
        .sequence(plan.boundary_kind,
                  [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
        .sequence(plan.boundary_alpha)
        .sequence(plan.boundary_beta)
        .sequence(plan.boundary_value)
        .sequence(plan.boundary_state_blocks,
                  [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
        .sequence(plan.boundary_state_components)
        .sequence(plan.boundary_field_blocks,
                  [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
        .sequence(plan.boundary_field_keys,
                  [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
        .sequence(plan.boundary_field_components)
        .sequence(plan.boundary_parameters)
        .text(plan.nullspace_provider_identity)
        .presence(!plan.nullspace_provider_identity.empty());
    if (!plan.nullspace_provider_identity.empty())
      contract.bytes(plan.nullspace_options.exact_contract());
    contract.text(plan.topology_provider_kind)
        .text(plan.topology_provenance)
        .text(plan.topology_digest)
        .presence(plan.has_reaction);
    if (plan.has_reaction)
      contract.scalar(plan.reaction);
    contract.presence(plan.has_boundary_kernel).presence(plan.has_nonlinear_plan);
    return std::move(contract).release();
  }

  void require_field_plan_consensus() {
    if (field_plan_consensus_verified_)
      return;
    std::string bytes;
    std::exception_ptr local_error;
    try {
      ExactContractBuilder registry;
      registry.text("pops.system.exact-ranked-field-plan-registry").scalar(std::uint32_t{1});
      for (const auto& [slot, plan] : field_plans_) {
        registry.text(slot).bytes(exact_field_plan_contract(plan));
        const auto configured =
            configured_field_solver_providers_.find(plan.backend_provider_route);
        const auto component = component_field_solver_providers_.find(plan.backend_provider_route);
        if ((configured == configured_field_solver_providers_.end()) ==
            (component == component_field_solver_providers_.end()))
          throw std::runtime_error(
              "System field plan must select exactly one installed exact-ranked backend route");
        if (configured != configured_field_solver_providers_.end())
          registry.text(configured->second.exact_identity)
              .bytes(configured->second.options.exact_contract());
        else
          registry.text(component->second->provider_identity())
              .bytes(component->second->collective_contract());
      }
      bytes = std::move(registry).release();
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L) != 0) {
      if (n_ranks() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("System field-plan registry preparation failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs({{"system-field-plan-registry", bytes}}))
      throw std::runtime_error("System: ordered resolved field plans differ across MPI ranks");
    field_plan_consensus_verified_ = true;
  }

  struct AcceptedSnapshot {
    std::vector<field_type> states;
    double time = 0.0;
    int macro_step = 0;

    explicit AcceptedSnapshot(const Impl& owner) : time(owner.t), macro_step(owner.macro_step_) {
      states.reserve(owner.sp.size());
      for (const Species& block : owner.sp)
        states.push_back(block.U);
    }

    void restore(Impl& owner) const {
      if (states.size() != owner.sp.size())
        throw std::logic_error("System transaction snapshot composition changed");
      for (std::size_t block = 0; block < states.size(); ++block)
        owner.sp[block].U = states[block];
      owner.t = time;
      owner.macro_step_ = macro_step;
    }
  };

  std::unique_ptr<AcceptedSnapshot> external_step_transaction_;
  bool external_step_transaction_committed_ = false;

  explicit Impl(const SystemConfig<Dim>& config) : domain_(config) {}

  Species& find(const std::string& name) { return blocks_.find(name); }
  const Species& find(const std::string& name) const { return blocks_.find(name); }
  int index(const std::string& name) const { return blocks_.index(name); }

  const typename boundary_registry_type::InstalledBoundary* boundary_for(
      const std::string& name) const noexcept {
    return boundary_registry_.find_boundary(name);
  }

  void publish_boundary_to_block(const std::string& name) {
    Species& block = find(name);
    const auto* installed = boundary_for(name);
    block.boundary = installed == nullptr ? nullptr : installed->authority;
    if (installed != nullptr)
      block.state_identity = installed->state_identity;
  }

  template <class Function>
  decltype(auto) execute_step_transaction(Function&& function) {
    AcceptedSnapshot snapshot(*this);
    try {
      return std::forward<Function>(function)();
    } catch (...) {
      snapshot.restore(*this);
      throw;
    }
  }
};

template <int Dim>
void validate_system_config(const SystemConfig<Dim>& config) {
  config.validate_spatial_domain();
  if (config.coordinate_system != runtime_config_detail::cartesian_coordinate_system<Dim>())
    throw std::invalid_argument(
        "System Cartesian core requires a dimension-qualified Cartesian provider");
}

}  // namespace pops
