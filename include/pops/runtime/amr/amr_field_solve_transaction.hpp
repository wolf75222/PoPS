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
template <class Rollback>
class FieldSolveTransaction {
 public:
  explicit FieldSolveTransaction(Rollback rollback) : rollback_(std::move(rollback)) {}
  FieldSolveTransaction(const FieldSolveTransaction&) = delete;
  FieldSolveTransaction& operator=(const FieldSolveTransaction&) = delete;
  ~FieldSolveTransaction() noexcept {
    if (!committed_)
      rollback_();
  }

  void commit() noexcept { committed_ = true; }

 private:
  Rollback rollback_;
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
  for (const auto& [component, _] : named_aux_)
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

inline std::vector<int> AmrRuntime::field_solve_aux_components(
    const FieldSolveScope& scope) const {
  std::set<int> components;
  if (scope.default_field) {
    const std::vector<int> defaults = default_aux_components();
    components.insert(defaults.begin(), defaults.end());
  }
  if (scope.named_fields != NamedFieldSnapshotScope::kNone) {
    const std::string* selected =
        scope.named_fields == NamedFieldSnapshotScope::kSelected ? scope.selected_named_field
                                                                 : nullptr;
    const std::vector<int> named = named_aux_components(selected);
    components.insert(named.begin(), named.end());
  }
  return {components.begin(), components.end()};
}

inline std::vector<MultiFab> AmrRuntime::pack_aux_components(
    const std::vector<int>& components) {
  std::vector<MultiFab> packed;
  if (components.empty())
    return packed;
  device_fence();
  packed.reserve(aux_.size());
  for (const MultiFab& source : aux_) {
    packed.emplace_back(source.box_array(), source.dmap(), static_cast<int>(components.size()),
                        source.n_grow());
    MultiFab& destination = packed.back();
    for (int li = 0; li < source.local_size(); ++li) {
      const ConstArray4 src = source.fab(li).const_array();
      const Array4 dst = destination.fab(li).array();
      const Box2D grown = source.fab(li).grown_box();
      for (std::size_t packed_component = 0; packed_component < components.size();
           ++packed_component)
        for_each_cell(grown, detail::CopyAuxComponentKernel{
                                 src, dst, components[packed_component],
                                 static_cast<int>(packed_component)});
    }
  }
  device_fence();
  return packed;
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
    if (source.local_size() != destination.local_size() ||
        source.ncomp() != static_cast<int>(components.size()))
      std::terminate();
    for (int li = 0; li < source.local_size(); ++li) {
      const ConstArray4 src = source.fab(li).const_array();
      const Array4 dst = destination.fab(li).array();
      const Box2D grown = destination.fab(li).grown_box();
      for (std::size_t packed_component = 0; packed_component < components.size();
           ++packed_component)
        for_each_cell(grown, detail::CopyAuxComponentKernel{
                                 src, dst, static_cast<int>(packed_component),
                                 components[packed_component]});
    }
  }
  device_fence();
}

inline void AmrRuntime::apply_named_aux_bc(MultiFab& packed,
                                           const std::vector<int>& components) {
  if (named_aux_bc_.empty())
    return;
  for (std::size_t packed_component = 0; packed_component < components.size();
       ++packed_component) {
    const auto policy = named_aux_bc_.find(components[packed_component]);
    if (policy == named_aux_bc_.end())
      continue;
    fill_physical_bc(packed, dom_, aux_halo_override(aux_bc_, policy->second),
                     static_cast<int>(packed_component));
  }
}

inline void AmrRuntime::publish_aux_components(const std::vector<int>& components) {
  if (components.empty())
    return;
  std::vector<MultiFab> packed = pack_aux_components(components);
  fill_ghosts_profiled(packed[0], dom_, aux_bc_);
  apply_named_aux_bc(packed[0], components);
  for (int level = 1; level < nlev_; ++level)
    detail::coupler_inject_aux_mb(packed[static_cast<std::size_t>(level - 1)],
                                  packed[static_cast<std::size_t>(level)],
                                  /*replicated_parent=*/(level == 1) && replicated_coarse_);
  unpack_aux_components(packed, components);
}

/// Publish components whose valid cells already carry a resolved value on every AMR level.  The
/// coarse-only publication above deliberately injects the parent across the child's whole grown
/// box; doing that here would destroy the independently solved fine valid cells.  FillPatch instead
/// materializes parent values only in coarse/fine ghosts, then lets the same-level/physical halo
/// fill override the regions for which a fine-level authority exists.
inline void AmrRuntime::publish_refined_aux_components(
    const std::vector<int>& components) {
  if (components.empty())
    return;
  std::vector<MultiFab> packed = pack_aux_components(components);
  Box2D level_domain = dom_;
  BCRec level_bc = aux_bc_;
  fill_ghosts_profiled(packed[0], level_domain, level_bc);
  // AuxHalo is a coarse-level authoring policy by contract. Fine patches retain the shared
  // physical BC while independently solved valid cells remain authoritative.
  apply_named_aux_bc(packed[0], components);
  for (int level = 1; level < nlev_; ++level) {
    level_domain = level_domain.refine(kAmrRefRatio);
    level_bc.dx /= Real(kAmrRefRatio);
    level_bc.dy /= Real(kAmrRefRatio);
    MultiFab& fine = packed[static_cast<std::size_t>(level)];
    // The local-parent path is correct for both distributed and replicated parents, including a
    // replicated multi-box coarse layout whose fine grown box crosses a coarse tile boundary.
    mf_fill_fine_ghosts_spatial_mb(
        fine, packed[static_cast<std::size_t>(level - 1)],
        /*replicated_parent=*/false);
    fill_ghosts_profiled(fine, level_domain, level_bc);
  }
  unpack_aux_components(packed, components);
}

inline AmrRuntime::FieldSolveSnapshot AmrRuntime::capture_field_solve_snapshot(
    const FieldSolveScope& scope) {
  device_fence();
  FieldSolveSnapshot snapshot;
  snapshot.aux_components = field_solve_aux_components(scope);
  snapshot.packed_aux = pack_aux_components(snapshot.aux_components);
  if (scope.default_field) {
    snapshot.has_default = true;
    snapshot.default_phi = mg_.phi();
    snapshot.default_rhs = mg_.rhs();
  }
  for (const auto& [name, field] : named_fields_) {
    if (scope.named_fields == NamedFieldSnapshotScope::kNone ||
        (scope.named_fields == NamedFieldSnapshotScope::kSelected &&
         (scope.selected_named_field == nullptr || name != *scope.selected_named_field)))
      continue;
    FieldSolveSnapshot::NamedFieldState state;
    state.nullspace = field.nullspace;
    state.level_nullspace = field.level_nullspace;
    state.nullspace_ready = field.nullspace_ready;
    if (field.fac) {
      state.storage = FieldSolveSnapshot::NamedFieldState::Storage::kComposite;
      for (int level = 0; level < field.fac->n_levels(); ++level) {
        state.phi.push_back(field.fac->phi_level(level));
        state.rhs.push_back(field.fac->rhs_level(level));
      }
    } else if (!field.level_mg.empty()) {
      state.storage = FieldSolveSnapshot::NamedFieldState::Storage::kLevelLocal;
      for (const auto& solver : field.level_mg) {
        state.phi.push_back(solver->phi());
        state.rhs.push_back(solver->rhs());
      }
    }
    snapshot.named.emplace(name, std::move(state));
  }
  return snapshot;
}

inline void AmrRuntime::restore_field_solve_snapshot(FieldSolveSnapshot&& snapshot) noexcept {
  device_fence();
  if (snapshot.has_default) {
    mg_.phi() = std::move(snapshot.default_phi);
    mg_.rhs() = std::move(snapshot.default_rhs);
  }
  unpack_aux_components(snapshot.packed_aux, snapshot.aux_components);
  for (auto& [name, state] : snapshot.named) {
    const auto found = named_fields_.find(name);
    if (found == named_fields_.end())
      std::terminate();
    NamedField& field = found->second;
    using Storage = FieldSolveSnapshot::NamedFieldState::Storage;
    if (state.storage == Storage::kUnallocated) {
      field.mg.reset();
      field.level_mg.clear();
      field.fac.reset();
    } else if (state.storage == Storage::kComposite) {
      if (!field.fac || state.phi.size() != static_cast<std::size_t>(field.fac->n_levels()) ||
          state.rhs.size() != state.phi.size())
        std::terminate();
      for (int level = 0; level < field.fac->n_levels(); ++level) {
        field.fac->phi_level(level) = std::move(state.phi[static_cast<std::size_t>(level)]);
        field.fac->rhs_level(level) = std::move(state.rhs[static_cast<std::size_t>(level)]);
      }
    } else {
      if (field.fac || state.phi.size() != field.level_mg.size() ||
          state.rhs.size() != state.phi.size())
        std::terminate();
      for (std::size_t level = 0; level < field.level_mg.size(); ++level) {
        field.level_mg[level]->phi() = std::move(state.phi[level]);
        field.level_mg[level]->rhs() = std::move(state.rhs[level]);
      }
    }
    field.nullspace = std::move(state.nullspace);
    field.level_nullspace = std::move(state.level_nullspace);
    field.nullspace_ready = state.nullspace_ready;
  }
}

template <class Solve>
inline SolveReport AmrRuntime::run_field_solve_transaction(const FieldSolveScope& scope,
                                                           Solve&& solve) {
  auto snapshot = capture_field_solve_snapshot(scope);
  detail::FieldSolveTransaction rollback(
      [this, &snapshot]() noexcept { restore_field_solve_snapshot(std::move(snapshot)); });
  SolveReport report = std::forward<Solve>(solve)();
  if (report.solved())
    rollback.commit();
  return report;
}

}  // namespace pops
