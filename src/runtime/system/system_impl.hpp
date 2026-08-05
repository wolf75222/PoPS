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

#include <algorithm>
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
  std::shared_ptr<exact_field_type> default_field_;
  std::map<std::string, std::shared_ptr<exact_field_type>> named_fields_;
  std::shared_ptr<exact_field_type> active_field_;

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
