/// @file
/// @brief Out-of-line transactional field-publication helpers for AmrRuntime.
///
/// Included at the end of amr_runtime.hpp after the complete AmrRuntime definition. This is not a
/// standalone API header: it keeps component-scoped aux packing/publication and field-solve rollback
/// mechanics out of the already large runtime class body.

#pragma once

#include <pops/runtime/amr/amr_runtime.hpp>  // the class this file defines members of

#include <exception>
#include <set>

namespace pops {
namespace detail {

/// Scope-bound rollback for a field solve. The callable is noexcept so a rejected report or an
/// exception restores the previously published fields while preserving the original failure.
template <class Rollback, class Release>
class FieldSolveTransaction {
 public:
  FieldSolveTransaction(Rollback rollback, Release release)
      : rollback_(std::move(rollback)), release_(std::move(release)) {}
  FieldSolveTransaction(const FieldSolveTransaction&) = delete;
  FieldSolveTransaction& operator=(const FieldSolveTransaction&) = delete;
  ~FieldSolveTransaction() noexcept {
    if (!committed_)
      rollback_();
    release_();
  }

  void commit() noexcept { committed_ = true; }

 private:
  Rollback rollback_;
  Release release_;
  bool committed_ = false;
};

struct CopyAuxComponentKernel {
  ConstArray4 source;
  Array4 destination;
  int source_component = 0;
  int destination_component = 0;

  POPS_HD void operator()(int i, int j) const {
    destination(i, j, destination_component) = source(i, j, source_component);
  }
};

inline void add_aux_component(std::set<int>& components, int component, int width) {
  if (component >= 0 && component < width)
    components.insert(component);
}

}  // namespace detail

inline std::vector<int> AmrRuntime::default_aux_components() const {
  std::set<int> components;
  for (int component = 0; component < 3; ++component)
    detail::add_aux_component(components, component, aux_ncomp_);
  for (const auto& [component, _] : static_aux_)
    detail::add_aux_component(components, component, aux_ncomp_);
  return {components.begin(), components.end()};
}

inline std::vector<int> AmrRuntime::named_aux_components(const std::string* selected) const {
  std::set<int> components;
  for (const auto& [name, field] : named_fields_) {
    if (selected != nullptr && name != *selected)
      continue;
    detail::add_aux_component(components, field.phi_comp, aux_ncomp_);
    detail::add_aux_component(components, field.gx_comp, aux_ncomp_);
    detail::add_aux_component(components, field.gy_comp, aux_ncomp_);
  }
  return {components.begin(), components.end()};
}

inline std::vector<int> AmrRuntime::field_solve_aux_components(const FieldSolveScope& scope) const {
  std::set<int> components;
  if (scope.default_field) {
    const std::vector<int> defaults = default_aux_components();
    components.insert(defaults.begin(), defaults.end());
  }
  if (scope.named_fields != NamedFieldSnapshotScope::kNone) {
    const std::string* selected = scope.named_fields == NamedFieldSnapshotScope::kSelected
                                      ? scope.selected_named_field
                                      : nullptr;
    const std::vector<int> named = named_aux_components(selected);
    components.insert(named.begin(), named.end());
  }
  return {components.begin(), components.end()};
}

inline std::vector<MultiFab> AmrRuntime::allocate_aux_component_carriers_(
    const std::vector<int>& components) const {
  std::vector<MultiFab> packed;
  if (components.empty())
    return packed;
  packed.reserve(aux_.size());
  for (const MultiFab& source : aux_) {
    packed.emplace_back(source.box_array(), source.dmap(), static_cast<int>(components.size()),
                        source.n_grow());
  }
  return packed;
}

inline void AmrRuntime::copy_aux_components_to_(std::vector<MultiFab>& packed,
                                                const std::vector<int>& components) const {
  if (components.empty()) {
    if (!packed.empty())
      throw std::invalid_argument("empty aux component set requires no carriers");
    return;
  }
  if (packed.size() != aux_.size())
    throw std::invalid_argument("aux carrier hierarchy depth mismatch");
  device_fence();
  for (std::size_t level = 0; level < aux_.size(); ++level) {
    const MultiFab& source = aux_[level];
    MultiFab& destination = packed[level];
    if (destination.box_array().boxes() != source.box_array().boxes() ||
        destination.dmap().ranks() != source.dmap().ranks() ||
        destination.ncomp() != static_cast<int>(components.size()) ||
        destination.n_grow() != source.n_grow())
      throw std::invalid_argument("aux carrier crossed an exact hierarchy layout");
    for (int li = 0; li < source.local_size(); ++li) {
      const ConstArray4 src = source.fab(li).const_array();
      const Array4 dst = destination.fab(li).array();
      const Box2D grown = source.fab(li).grown_box();
      for (std::size_t packed_component = 0; packed_component < components.size();
           ++packed_component)
        for_each_cell(grown, detail::CopyAuxComponentKernel{src, dst, components[packed_component],
                                                            static_cast<int>(packed_component)});
    }
  }
  device_fence();
}

inline void AmrRuntime::unpack_aux_components(const std::vector<MultiFab>& packed,
                                              const std::vector<int>& components) noexcept {
  if (components.empty())
    return;
  if (packed.size() != aux_.size())
    std::terminate();
  device_fence();
  for (std::size_t level = 0; level < aux_.size(); ++level) {
    const MultiFab& source = packed[level];
    MultiFab& destination = aux_[level];
    if (source.box_array().boxes() != destination.box_array().boxes() ||
        source.dmap().ranks() != destination.dmap().ranks() ||
        source.local_size() != destination.local_size() ||
        source.ncomp() != static_cast<int>(components.size()) ||
        source.n_grow() != destination.n_grow())
      std::terminate();
    for (int li = 0; li < source.local_size(); ++li) {
      const ConstArray4 src = source.fab(li).const_array();
      const Array4 dst = destination.fab(li).array();
      const Box2D grown = destination.fab(li).grown_box();
      for (std::size_t packed_component = 0; packed_component < components.size();
           ++packed_component)
        for_each_cell(grown,
                      detail::CopyAuxComponentKernel{src, dst, static_cast<int>(packed_component),
                                                     components[packed_component]});
    }
  }
  device_fence();
}

inline AmrRuntime::AuxPublicationWorkspace& AmrRuntime::acquire_aux_publication_workspace_(
    const std::vector<int>& components, bool refined_values) {
  for (AuxPublicationWorkspace& workspace : aux_publication_workspaces_) {
    if (workspace.topology_generation == topology_materialization_generation_ &&
        workspace.refined_values == refined_values && workspace.components == components)
      return workspace;
  }
  AuxPublicationWorkspace candidate = make_aux_publication_workspace_(
      components, refined_values, topology_materialization_generation_);
  aux_publication_workspaces_.push_back(std::move(candidate));
  return aux_publication_workspaces_.back();
}

inline void AmrRuntime::apply_named_aux_bc(MultiFab& packed, const std::vector<int>& components,
                                           const Box2D& level_domain,
                                           const BCRec& level_bc) {
  if (named_aux_bc_.empty())
    return;
  for (std::size_t packed_component = 0; packed_component < components.size(); ++packed_component) {
    const auto policy = named_aux_bc_.find(components[packed_component]);
    if (policy == named_aux_bc_.end())
      continue;
    fill_physical_bc(packed, level_domain, aux_halo_override(level_bc, policy->second),
                     static_cast<int>(packed_component));
  }
}

inline void AmrRuntime::publish_aux_components(const std::vector<int>& components) {
  if (components.empty())
    return;
  AuxPublicationWorkspace& workspace =
      acquire_aux_publication_workspace_(components, /*refined_values=*/false);
  copy_aux_components_to_(workspace.packed, components);
  std::vector<MultiFab>& packed = workspace.packed;
  Box2D level_domain = dom_;
  BCRec level_bc = aux_bc_;
  fill_ghosts_profiled(packed[0], level_domain, level_bc);
  apply_named_aux_bc(packed[0], components, level_domain, level_bc);
  for (int level = 1; level < nlev_; ++level) {
    level_domain = level_domain.refine(kAmrRefRatio);
    level_bc.dx /= Real(kAmrRefRatio);
    level_bc.dy /= Real(kAmrRefRatio);
    const bool replicated_parent = level == 1 && replicated_coarse_;
    const CommunicatorView communicator =
        replicated_parent ? CommunicatorView{} : world_communicator_view();
    workspace.coarse_transfers.at(static_cast<std::size_t>(level - 1))
        .apply(packed[static_cast<std::size_t>(level - 1)],
               packed[static_cast<std::size_t>(level)],
               topology_materialization_generation_, communicator);
    fill_ghosts_profiled(packed[static_cast<std::size_t>(level)], level_domain, level_bc);
    apply_named_aux_bc(packed[static_cast<std::size_t>(level)], components, level_domain, level_bc);
  }
  unpack_aux_components(packed, components);
}

/// Publish components whose valid cells already carry a resolved value on every AMR level.  The
/// Coarse-authoritative publication above deliberately injects the parent across the child's whole
/// grown box; doing that here would destroy independently solved fine valid cells. FillPatch instead
/// materializes parent values only in coarse/fine ghosts, then lets the same-level/physical halo
/// fill override the regions for which a fine-level authority exists.
inline void AmrRuntime::publish_refined_aux_components(const std::vector<int>& components) {
  if (components.empty())
    return;
  AuxPublicationWorkspace& workspace =
      acquire_aux_publication_workspace_(components, /*refined_values=*/true);
  copy_aux_components_to_(workspace.packed, components);
  std::vector<MultiFab>& packed = workspace.packed;
  Box2D level_domain = dom_;
  BCRec level_bc = aux_bc_;
  fill_ghosts_profiled(packed[0], level_domain, level_bc);
  // AuxHalo is a coarse-level authoring policy by contract. Fine patches retain the shared
  // physical BC while independently solved valid cells remain authoritative.
  apply_named_aux_bc(packed[0], components, level_domain, level_bc);
  for (int level = 1; level < nlev_; ++level) {
    level_domain = level_domain.refine(kAmrRefRatio);
    level_bc.dx /= Real(kAmrRefRatio);
    level_bc.dy /= Real(kAmrRefRatio);
    MultiFab& fine = packed[static_cast<std::size_t>(level)];
    const bool replicated_parent = level == 1 && replicated_coarse_;
    const CommunicatorView communicator =
        replicated_parent ? CommunicatorView{} : world_communicator_view();
    workspace.coarse_transfers.at(static_cast<std::size_t>(level - 1))
        .apply(packed[static_cast<std::size_t>(level - 1)], fine,
               topology_materialization_generation_, communicator);
    fill_ghosts_profiled(fine, level_domain, level_bc);
    apply_named_aux_bc(fine, components, level_domain, level_bc);
  }
  unpack_aux_components(packed, components);
}

inline AmrRuntime::FieldSolveSnapshot& AmrRuntime::capture_field_solve_snapshot(
    const FieldSolveScope& scope) {
  if (field_solve_transaction_active_)
    throw std::logic_error(
        "AmrRuntime field solves are sequential and cannot be re-entered");
  if (scope.named_fields == NamedFieldSnapshotScope::kSelected &&
      scope.selected_named_field == nullptr)
    throw std::invalid_argument("selected field-solve scope requires an exact field identity");

  const std::string selected = scope.selected_named_field == nullptr
                                   ? std::string{}
                                   : *scope.selected_named_field;
  const std::vector<int> components = field_solve_aux_components(scope);
  const auto includes = [&](const std::string& name) {
    return scope.named_fields == NamedFieldSnapshotScope::kAll ||
           (scope.named_fields == NamedFieldSnapshotScope::kSelected && name == selected);
  };
  const auto same_scope = [&](const FieldSolveSnapshot& snapshot) {
    return snapshot.scope_default_field == scope.default_field &&
           snapshot.scope_named_fields == scope.named_fields &&
           snapshot.scope_selected_named_field == selected;
  };
  const auto compatible = [&](const FieldSolveSnapshot& snapshot) {
    if (!same_scope(snapshot) ||
        snapshot.topology_generation != topology_materialization_generation_ ||
        snapshot.aux_components != components || snapshot.has_default != scope.default_field ||
        snapshot.packed_aux.size() != (components.empty() ? 0u : aux_.size()))
      return false;
    for (std::size_t level = 0; level < aux_.size(); ++level) {
      const MultiFab& packed = snapshot.packed_aux[level];
      const MultiFab& live = aux_[level];
      if (packed.box_array().boxes() != live.box_array().boxes() ||
          packed.dmap().ranks() != live.dmap().ranks() ||
          packed.ncomp() != static_cast<int>(components.size()) ||
          packed.n_grow() != live.n_grow())
        return false;
    }
    if (scope.default_field &&
        (!same_exact_multifab_layout_(snapshot.default_phi,
                                     default_field_solver_->phi_level(0)) ||
         !same_exact_multifab_layout_(snapshot.default_rhs,
                                     default_field_solver_->rhs_level(0))))
      return false;
    std::size_t expected_named = 0;
    for (const auto& [name, field] : named_fields_) {
      if (!includes(name))
        continue;
      ++expected_named;
      const auto found = snapshot.named.find(name);
      if (found == snapshot.named.end())
        return false;
      const auto& state = found->second;
      using Storage = FieldSolveSnapshot::NamedFieldState::Storage;
      const Storage expected_storage =
          !field.solver ? Storage::kUnallocated
                        : (field.solver->couples_hierarchy_levels() ? Storage::kComposite
                                                                    : Storage::kLevelLocal);
      if (state.storage != expected_storage)
        return false;
      const std::size_t levels =
          field.solver ? static_cast<std::size_t>(field.solver->level_count()) : 0u;
      if (state.phi.size() != levels || state.rhs.size() != levels)
        return false;
      for (std::size_t level = 0; level < levels; ++level)
        if (!same_exact_multifab_layout_(state.phi[level],
                                        field.solver->phi_level(static_cast<int>(level))) ||
            !same_exact_multifab_layout_(state.rhs[level],
                                        field.solver->rhs_level(static_cast<int>(level))))
          return false;
    }
    return snapshot.named.size() == expected_named;
  };

  FieldSolveSnapshot* workspace = nullptr;
  for (FieldSolveSnapshot& candidate : field_solve_rollback_workspaces_)
    if (same_scope(candidate)) {
      workspace = &candidate;
      break;
    }
  if (workspace == nullptr || !compatible(*workspace)) {
    FieldSolveSnapshot candidate;
    candidate.topology_generation = topology_materialization_generation_;
    candidate.scope_default_field = scope.default_field;
    candidate.scope_named_fields = scope.named_fields;
    candidate.scope_selected_named_field = selected;
    candidate.aux_components = components;
    candidate.packed_aux = allocate_aux_component_carriers_(components);
    if (scope.default_field) {
      candidate.has_default = true;
      const MultiFab& phi = default_field_solver_->phi_level(0);
      const MultiFab& rhs = default_field_solver_->rhs_level(0);
      candidate.default_phi =
          MultiFab(phi.box_array(), phi.dmap(), phi.ncomp(), phi.n_grow());
      candidate.default_rhs =
          MultiFab(rhs.box_array(), rhs.dmap(), rhs.ncomp(), rhs.n_grow());
    }
    for (const auto& [name, field] : named_fields_) {
      if (!includes(name))
        continue;
      FieldSolveSnapshot::NamedFieldState state;
      if (field.solver) {
        state.storage = field.solver->couples_hierarchy_levels()
                            ? FieldSolveSnapshot::NamedFieldState::Storage::kComposite
                            : FieldSolveSnapshot::NamedFieldState::Storage::kLevelLocal;
        state.phi.reserve(static_cast<std::size_t>(field.solver->level_count()));
        state.rhs.reserve(static_cast<std::size_t>(field.solver->level_count()));
        for (int level = 0; level < field.solver->level_count(); ++level) {
          const MultiFab& phi = field.solver->phi_level(level);
          const MultiFab& rhs = field.solver->rhs_level(level);
          state.phi.emplace_back(phi.box_array(), phi.dmap(), phi.ncomp(), phi.n_grow());
          state.rhs.emplace_back(rhs.box_array(), rhs.dmap(), rhs.ncomp(), rhs.n_grow());
        }
      }
      candidate.named.emplace(name, std::move(state));
    }
    if (workspace == nullptr) {
      field_solve_rollback_workspaces_.push_back(std::move(candidate));
      workspace = &field_solve_rollback_workspaces_.back();
    } else {
      *workspace = std::move(candidate);
    }
  }

  device_fence();
  copy_aux_components_to_(workspace->packed_aux, workspace->aux_components);
  if (workspace->has_default) {
    PureFieldAlgebra::copy_allocated(workspace->default_phi,
                                     default_field_solver_->phi_level(0));
    PureFieldAlgebra::copy_allocated(workspace->default_rhs,
                                     default_field_solver_->rhs_level(0));
  }
  for (auto& [name, state] : workspace->named) {
    const NamedField& field = named_fields_.at(name);
    state.nullspace = field.nullspace;
    state.level_nullspace = field.level_nullspace;
    state.nullspace_ready = field.nullspace_ready;
    if (!field.solver)
      continue;
    for (int level = 0; level < field.solver->level_count(); ++level) {
      PureFieldAlgebra::copy_allocated(state.phi[static_cast<std::size_t>(level)],
                                       field.solver->phi_level(level));
      PureFieldAlgebra::copy_allocated(state.rhs[static_cast<std::size_t>(level)],
                                       field.solver->rhs_level(level));
    }
  }
  device_fence();
  field_solve_transaction_active_ = true;
  return *workspace;
}

inline void AmrRuntime::restore_field_solve_snapshot(FieldSolveSnapshot& snapshot) noexcept {
  try {
    if (snapshot.topology_generation != topology_materialization_generation_)
      std::terminate();
    device_fence();
    if (snapshot.has_default) {
      PureFieldAlgebra::copy_allocated(default_field_solver_->phi_level(0),
                                       snapshot.default_phi);
      PureFieldAlgebra::copy_allocated(default_field_solver_->rhs_level(0),
                                       snapshot.default_rhs);
    }
    unpack_aux_components(snapshot.packed_aux, snapshot.aux_components);
    for (auto& [name, state] : snapshot.named) {
      const auto found = named_fields_.find(name);
      if (found == named_fields_.end())
        std::terminate();
      NamedField& field = found->second;
      using Storage = FieldSolveSnapshot::NamedFieldState::Storage;
      if (state.storage == Storage::kUnallocated) {
        invalidate_named_field_solver(field);
      } else {
        const bool composite = state.storage == Storage::kComposite;
        if (!field.solver || field.solver->couples_hierarchy_levels() != composite ||
            state.phi.size() != static_cast<std::size_t>(field.solver->level_count()) ||
            state.rhs.size() != state.phi.size())
          std::terminate();
        for (int level = 0; level < field.solver->level_count(); ++level) {
          PureFieldAlgebra::copy_allocated(field.solver->phi_level(level),
                                           state.phi[static_cast<std::size_t>(level)]);
          PureFieldAlgebra::copy_allocated(field.solver->rhs_level(level),
                                           state.rhs[static_cast<std::size_t>(level)]);
        }
      }
      if (!state.nullspace_ready) {
        field.nullspace_workspace.reset();
        field.level_nullspace_workspaces.clear();
        field.nullspace_rhs_levels.clear();
        field.nullspace_phi_levels.clear();
      }
      std::swap(field.nullspace, state.nullspace);
      field.level_nullspace.swap(state.level_nullspace);
      std::swap(field.nullspace_ready, state.nullspace_ready);
    }
    device_fence();
  } catch (...) {
    std::terminate();
  }
}

template <class Solve>
inline SolveReport AmrRuntime::run_field_solve_transaction(const FieldSolveScope& scope,
                                                           Solve&& solve) {
  FieldSolveSnapshot& snapshot = capture_field_solve_snapshot(scope);
  detail::FieldSolveTransaction rollback(
      [this, &snapshot]() noexcept { restore_field_solve_snapshot(snapshot); },
      [this]() noexcept { release_field_solve_snapshot_(); });
  SolveReport report = std::forward<Solve>(solve)();
  if (report.solved())
    rollback.commit();
  return report;
}

}  // namespace pops
